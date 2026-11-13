"""``PatchMatch``: the patch-matching keypoint model over any ``PatchFeatures`` backbone (§26).

Zero training. The user marks the skeleton on a few example frames (§26 "Example sets"); every
detection is then matched against those marks. The engine is the method the internal evaluations
measured, the arm the design rests on: ``skeleton_peaks_hard``.

1. **View** (``view.py``): a square around each detection's box, rendered to ``crop_px``.
2. **Example bank** (``bank.py``), built once at ``load()``: each visible mark on an example
   frame is one row — the backbone's patch grid sampled bilinearly at the click, renormalised.
   A second bank holds the same examples **mirrored**, pair labels swapped: its best score per
   target is ``extras.mirrored``, what the track stage weighs to settle left/right over time.
3. **Similarity maps**: one ``bank @ grid`` per detection; the max over a target's rows; cells
   outside the detection's mask dilated by one patch set to −inf (``needs_mask``).
4. **Candidates** and 5. **Decode** (``decode.py``).
6. **Output**: one ``FrameResult`` per ``InstanceFrame``, one row per detection — ``bbox``
   (the detection's), ``detection_row``, ``confidence`` NaN (``reports_confidence = False``),
   ``points``, ``point_scores`` (the chosen candidate's cosine), ``candidates`` with their
   scores (``n_candidates = 5``), and ``extras``: ``ref_index (n, K, 5) int16`` (which example
   row won each candidate) and ``mirrored (n, K) float32``.

A concrete method names its backbone (``backbone_class``) and nothing else; the install
contract, ``check_backend`` and ``resolved`` are ``Model``'s, and readiness / install / the
disk check delegate to the backbone, since the method installs nothing of its own
(``shares_weights_with``). ``torch`` is imported inside the methods that need it (Tenet 13).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from detail.core.dataclasses import FrameResult, ModelReadiness
from detail.core.examples import ExampleSet
from detail.core.installed import is_importable
from detail.core.keypoints import InstanceFrame, ScoreSpec
from detail.core.protocols import KeypointModel
from detail.core.skeleton import Skeleton
from detail.models.patch_match import decode as _decode
from detail.models.patch_match.bank import ExampleBank, sample_at
from detail.models.patch_match.view import (
    DEFAULT_CROP_PX,
    View,
    allowed_cells,
    body_scale,
    make_view,
    render,
)

#: Example crops run through the backbone this many at a time at ``load()``.
_EXAMPLE_BATCH = 16


@dataclass
class _Learned:
    """What ``load()`` learns from one example set: the bank, the mirrored bank, the skeleton
    statistics, and the examples' own grids for held-out scoring. One per class for a
    per-class method (§26 *One example set per detection class*), keyed None for one set.
    """

    bank: ExampleBank
    mirror_bank: ExampleBank
    stats: _decode.SkeletonStats
    held: dict


class PatchMatch(KeypointModel):
    """Keypoints by matching a backbone's patch features to a few marked example frames."""

    architecture = "vit"
    supported_backends = ["cuda", "mps", "cpu"]
    needs_examples = True
    per_detection = True  # one instance per detection: a crop around each (§26 *View*)
    needs_mask = True
    n_candidates = _decode.SKEL_TOPK
    # Whether a cosine separates "not visible" is §26 open question 1, unanswered: no default
    # cutoff until it is, so a consumer shows every placed point unless asked otherwise.
    score_spec = ScoreSpec(
        meaning="cosine similarity", low=-1.0, high=1.0, default_cutoff=None
    )
    default_track_method = "temporal_decode"
    #: The constants that moved a point, recorded on every run under ``matching:`` (§26).
    matching = dict(_decode.MATCHING)

    def __init__(
        self,
        variant: str | None = None,
        *,
        examples: ExampleSet | None = None,
        example_images: Sequence[np.ndarray] | None = None,
        class_examples: (
            dict[str, tuple[ExampleSet, Sequence[np.ndarray]]] | None
        ) = None,
        decode: str | None = None,
        side_rule: str | None = None,
        crop_px: int | None = None,
    ) -> None:
        """Built with the example set it learns from (§26: an example set is a config input,
        not a prompt cue), or with one per detection class (``class_examples``: ``{class:
        (set, images)}``, every set on one skeleton), in which case each detection is matched
        against its own class's set. ``decode`` / ``side_rule`` / ``crop_px`` left ``None``
        resolve against the skeleton (``resolve_settings``); the stage writes the resolved
        values back into its config (Tenet 9)."""
        super().__init__(variant)
        if decode is not None and decode not in _decode.DECODES:
            raise ValueError(
                f"unknown decode {decode!r}; expected one of {_decode.DECODES}"
            )
        if side_rule is not None and side_rule not in _decode.SIDE_RULES:
            raise ValueError(
                f"unknown side_rule {side_rule!r}; expected one of {_decode.SIDE_RULES}"
            )
        if crop_px is not None and crop_px % self.patch_size:
            raise ValueError(
                f"crop_px must be a multiple of {self.patch_size}; got {crop_px}"
            )
        if (examples is None) != (example_images is None):
            raise ValueError("pass examples and example_images together")
        if class_examples is not None and examples is not None:
            raise ValueError("pass examples or class_examples, not both")
        if class_examples is not None and not class_examples:
            raise ValueError("class_examples names no class")
        #: The sets it learns from, keyed by detection class; None keys the one set.
        self._sources: dict[str | None, tuple[ExampleSet, list[np.ndarray]]] = {}
        if examples is not None:
            self._sources[None] = (examples, list(example_images))
        for cls, (set_, images) in sorted((class_examples or {}).items()):
            self._sources[str(cls)] = (set_, list(images))
        digests = {s.skeleton.digest() for s, _ in self._sources.values()}
        if len(digests) > 1:
            raise ValueError("every class's example set must be on the same skeleton")
        for set_, images in self._sources.values():
            if len(images) != len(set_.frames):
                raise ValueError(
                    f"one image per example frame: {len(set_.frames)} frames, "
                    f"{len(images)} images"
                )
        self._sources = {
            key: (set_, [np.asarray(im) for im in images])
            for key, (set_, images) in self._sources.items()
        }
        #: The one set, or the first class's: its skeleton is every class's.
        self.examples = next(iter(self._sources.values()))[0] if self._sources else None
        #: Whether each detection is matched against its own class's set.
        self.per_class = class_examples is not None
        if self.examples is not None:
            self.settings = self.resolve_settings(
                self.examples.skeleton, decode, side_rule, crop_px
            )
        else:
            self.settings = {
                "decode": decode,
                "side_rule": side_rule,
                "crop_px": int(crop_px) if crop_px is not None else DEFAULT_CROP_PX,
            }
        self._backbone = None
        self._learned: dict[str | None, _Learned] = {}
        self._plan: _decode.DecodePlan | None = None

    # -- the backbone ------------------------------------------------------- #
    @classmethod
    @abstractmethod
    def backbone_class(cls) -> type:
        """The ``EmbeddingModel`` with ``PatchFeatures`` this method reads (imported lazily)."""

    @property
    def patch_size(self) -> int:
        return int(self.backbone_class().patch_size)

    def _make_backbone(self):
        return self.backbone_class()(variant=self.weight)

    def readiness(self) -> ModelReadiness:
        return self._make_backbone().readiness()

    def is_installed(self) -> bool:
        return self._make_backbone().is_installed()

    def install(self, progress_cb=None) -> None:
        self._make_backbone().install(progress_cb)

    # -- settings ----------------------------------------------------------- #
    @classmethod
    def resolve_settings(
        cls,
        skeleton: Skeleton,
        decode: str | None,
        side_rule: str | None,
        crop_px: int | None,
    ) -> dict:
        """The defaulted settings resolved against the skeleton, or a ``ValueError`` naming
        what to change (Tenet 9: resolved once, written back, so a run records what it did).

        ``decode`` defaults to ``skeleton`` when the skeleton has edges, else ``per_point``;
        ``side_rule`` to ``auto`` (decided per pair from the clicks at load) when the skeleton
        decode runs and there is a left/right pair, else ``off``. An explicit choice the
        skeleton cannot honour is refused rather than ignored.
        """
        has_pairs = bool(skeleton.pairs)
        if decode is None:
            decode = "skeleton" if skeleton.edges else "per_point"
        elif decode not in _decode.DECODES:
            raise ValueError(
                f"unknown decode {decode!r}; expected one of {_decode.DECODES}"
            )
        elif decode == "skeleton" and not skeleton.edges:
            raise ValueError(
                "decode 'skeleton' needs edges between the keypoints; add edges to the "
                "example set's skeleton, or set decode to 'per_point'"
            )
        if side_rule is None:
            side_rule = "auto" if decode == "skeleton" and has_pairs else "off"
        elif side_rule not in _decode.SIDE_RULES:
            raise ValueError(
                f"unknown side_rule {side_rule!r}; expected one of {_decode.SIDE_RULES}"
            )
        elif side_rule != "off":
            if decode != "skeleton":
                raise ValueError(
                    f"the left/right rule {side_rule!r} works through the skeleton decode; "
                    "set decode to 'skeleton', or the rule to 'off'"
                )
            if not has_pairs:
                raise ValueError(
                    f"the left/right rule {side_rule!r} needs a left/right pair in the "
                    "skeleton; pair a left point with its right partner, or set the rule "
                    "to 'off'"
                )
            if (
                side_rule == "midline"
                and not _decode.DecodePlan.from_skeleton(skeleton).midline_path
            ):
                raise ValueError(
                    "the 'midline' rule needs a midline: join at least two unpaired points "
                    "with edges, or use 'hard'"
                )
        patch = cls.backbone_class().patch_size
        resolved_crop = int(crop_px) if crop_px is not None else DEFAULT_CROP_PX
        if resolved_crop % patch:
            raise ValueError(
                f"crop_px must be a multiple of {patch}; got {resolved_crop}"
            )
        return {"decode": decode, "side_rule": side_rule, "crop_px": resolved_crop}

    def skeleton(self) -> Skeleton:
        if self.examples is None:
            raise ValueError(
                f"{self.model_id}'s skeleton is its example set's; construct it with one"
            )
        return self.examples.skeleton

    # -- load: the banks and the learned skeleton ---------------------------- #
    def load(self, device: str) -> None:
        if self.examples is None:
            raise ValueError(f"{self.model_id} needs an example set to load")
        if not is_importable("torch"):
            raise RuntimeError(f"{self.model_id} needs PyTorch: pip install torch")
        for set_, _images in self._sources.values():
            set_.check_runnable()
        backbone = self._make_backbone()
        backbone.load(device)  # local files only, never the network (§5)
        self._backbone = backbone
        self._plan = _decode.DecodePlan.from_skeleton(self.examples.skeleton)
        # One backbone, one bank per set: each class learns from its own clicks only.
        self._learned = {
            key: self._learn(set_, images)
            for key, (set_, images) in self._sources.items()
        }

    def _learn(
        self, examples: ExampleSet, example_images: list[np.ndarray]
    ) -> _Learned:
        """The banks, skeleton statistics and held-out grids from one set's clicks."""
        skeleton = examples.skeleton
        swap = {
            a: b
            for left, right in skeleton.pairs
            for a, b in ((left, right), (right, left))
        }
        crop_px = self.settings["crop_px"]
        views: list[View] = []
        crops: list[np.ndarray] = []
        clicks: list[tuple[dict, float]] = []
        for frame, rgb in zip(examples.frames, example_images):
            wh = (rgb.shape[1], rgb.shape[0])
            view = make_view(frame.bbox, wh, crop_px, self.patch_size)
            views.append(view)
            crops.append(render(rgb, view))
            pts = {
                m.target: (m.x * wh[0], m.y * wh[1])
                for m in frame.marks
                if m.kind == "point"
                and m.visible
                and m.positive
                and m.target in skeleton.names
            }
            clicks.append((pts, body_scale(frame.bbox, wh)))
        grids = self._grids(crops)
        mirrored = self._grids([np.ascontiguousarray(c[:, ::-1]) for c in crops])
        rows: dict[str, list] = {}
        mirror_rows: dict[str, list] = {}
        for i, (grid, flipped, view, (pts, _scale)) in enumerate(
            zip(grids, mirrored, views, clicks)
        ):
            for name, (x, y) in pts.items():
                u, v = view.to_view(x, y)
                rows.setdefault(name, []).append((i, sample_at(grid, u, v, view.size)))
                mirror_rows.setdefault(swap.get(name, name), []).append(
                    (i, sample_at(flipped, view.size - u, v, view.size))
                )
        return _Learned(
            bank=ExampleBank.build(skeleton.names, rows),
            mirror_bank=ExampleBank.build(skeleton.names, mirror_rows),
            stats=_decode.learn_skeleton(
                clicks, self._plan, self.settings["side_rule"]
            ),
            # Kept for score_held_out: the examples' own grids and rows, so each example
            # can be scored against the others without running the backbone again.
            held={"grids": grids, "views": views, "clicks": clicks, "rows": rows},
        )

    def _learned_for(self, sf: InstanceFrame, det_row: int) -> _Learned:
        """The bank a detection is matched against: its class's, or the one set's.

        Raises:
            ValueError: A per-class method handed a detection with no class, or of a class
                it has no set for (the runner filters by class, so this is a wiring bug).
        """
        if not self.per_class:
            return self._learned[None]
        cls = (sf.classes or {}).get(det_row)
        if cls not in self._learned:
            raise ValueError(
                f"detection row {det_row} on frame {sf.frame_idx} is of class {cls!r}, "
                f"which has no example set; this method has sets for "
                f"{sorted(k for k in self._learned if k is not None)}"
            )
        return self._learned[cls]

    def _grids(self, images: list[np.ndarray]) -> list:
        out: list = []
        for i in range(0, len(images), _EXAMPLE_BATCH):
            out.extend(self._backbone.patch_grids(images[i : i + _EXAMPLE_BATCH]))
        return out

    # -- predict ------------------------------------------------------------- #
    # Split in three for keypoint_detect's stages (§26 *Execution*): crops and mask cells on
    # the prep thread, the backbone and the bank matmuls on the model's thread (every copy to
    # the host there), decode and rows on the writer thread. ``finish_batch`` only reads
    # ``_plan`` / ``settings`` and each detection's ``_Learned``, which ``load()`` alone writes.
    def predict_batch(self, frames: list[InstanceFrame]) -> list[FrameResult]:
        return self.finish_batch(self.run_prepared(self.prepare_batch(frames)))

    def prepare_batch(self, frames: list[InstanceFrame]) -> tuple:
        """Each detection's view, crop, allowed cells and bank: CPU only."""
        self._require_loaded()
        # (frame index, detection_row, view, allowed, body scale, learned)
        dets: list[tuple] = []
        crops: list[np.ndarray] = []
        for fi, sf in enumerate(frames):
            wh = (sf.frame.shape[1], sf.frame.shape[0])
            for det_row in sorted(sf.boxes):
                box = sf.boxes[det_row]
                view = make_view(box, wh, self.settings["crop_px"], self.patch_size)
                crops.append(render(sf.frame, view))
                mask = sf.masks.get(det_row) if sf.masks else None
                allowed = allowed_cells(mask, view, wh)
                learned = self._learned_for(sf, det_row)
                dets.append((fi, det_row, view, allowed, body_scale(box, wh), learned))
        return frames, dets, crops

    def run_prepared(self, prepared: tuple) -> tuple:
        """The backbone over the crops and both banks' similarities, copied to the host once
        per bank: ``(frames, dets, sims, mirror_sims, grid_hw)``, numpy only. A per-class
        method's banks differ in size, so each bank's detections are stacked on their own.
        """
        frames, dets, crops = prepared
        if not crops:
            return frames, dets, None, None, None
        import torch

        grids = self._grids(crops)
        sims: list = [None] * len(dets)
        mirror: list = [None] * len(dets)
        by_bank: dict[int, list[int]] = {}
        for i, det in enumerate(dets):
            by_bank.setdefault(id(det[5]), []).append(i)
        for idx in by_bank.values():
            learned = dets[idx[0]][5]
            own = torch.stack([learned.bank.sims(grids[i]) for i in idx]).cpu().numpy()
            own_mirror = (
                torch.stack([learned.mirror_bank.sims(grids[i]) for i in idx])
                .cpu()
                .numpy()
            )
            for j, i in enumerate(idx):
                sims[i], mirror[i] = own[j], own_mirror[j]
        if len(by_bank) == 1:  # one set: one array per bank, as the writer reads it
            sims, mirror = np.stack(sims), np.stack(mirror)
        return frames, dets, sims, mirror, (grids[0].shape[1], grids[0].shape[2])

    def finish_batch(self, raw: tuple) -> list[FrameResult]:
        """Candidates, decode and one ``FrameResult`` per frame, from host arrays."""
        frames, dets, sims, mirror_sims, grid_hw = raw
        names = self._plan.names
        k_pts, n_cand = len(names), self.n_candidates
        per_frame: dict[int, list[dict]] = {fi: [] for fi in range(len(frames))}
        for i, (fi, det_row, view, allowed, scale, learned) in enumerate(dets):
            sf = frames[fi]
            wh = (sf.frame.shape[1], sf.frame.shape[0])
            maps, winners = learned.bank.maps_from_sims(sims[i], *grid_hw, allowed)
            cands = _decode.candidates(maps, view)
            placed = self._place(cands, view, scale, learned.stats)
            pool = _decode.candidate_pool(cands, self._plan)
            mirror_maps, _ = learned.mirror_bank.maps_from_sims(
                mirror_sims[i], *grid_hw, allowed
            )
            mirrored = ExampleBank.best_from_maps(mirror_maps)
            row = {
                "detection_row": det_row,
                "bbox": np.asarray(sf.boxes[det_row], np.float32),
                "points": np.full((k_pts, 2), np.nan, np.float32),
                "point_scores": np.full(k_pts, np.nan, np.float32),
                "candidates": np.full((k_pts, n_cand, 2), np.nan, np.float32),
                "candidate_scores": np.full((k_pts, n_cand), np.nan, np.float32),
                "ref_index": np.full((k_pts, n_cand), -1, np.int16),
                "mirrored": np.full(k_pts, np.nan, np.float32),
            }
            for j, name in enumerate(names):
                if name in placed:
                    (x, y), score, _cell = placed[name]
                    row["points"][j] = (x / wh[0], y / wh[1])
                    row["point_scores"][j] = score
                for c, ((x, y), score, (gy, gx)) in enumerate(
                    pool.get(name, [])[:n_cand]
                ):
                    row["candidates"][j, c] = (x / wh[0], y / wh[1])
                    row["candidate_scores"][j, c] = score
                    row["ref_index"][j, c] = winners[name][gy, gx]
                if name in mirrored:
                    row["mirrored"][j] = mirrored[name]
            per_frame[fi].append(row)
        results = []
        for fi, sf in enumerate(frames):
            rows = per_frame[fi]
            n = len(rows)

            def stack(key, shape, dtype):
                if n == 0:
                    return np.zeros((0, *shape), dtype)
                return np.stack([r[key] for r in rows]).astype(dtype)

            results.append(
                FrameResult(
                    frame_idx=sf.frame_idx,
                    t_ns=sf.t_ns,
                    n=n,
                    confidence=np.full(n, np.nan, np.float32),
                    bbox=stack("bbox", (4,), np.float32),
                    detection_row=np.asarray(
                        [r["detection_row"] for r in rows], dtype=np.int32
                    ),
                    points=stack("points", (k_pts, 2), np.float32),
                    point_scores=stack("point_scores", (k_pts,), np.float32),
                    candidates=stack("candidates", (k_pts, n_cand, 2), np.float32),
                    candidate_scores=stack(
                        "candidate_scores", (k_pts, n_cand), np.float32
                    ),
                    extras={
                        "ref_index": stack("ref_index", (k_pts, n_cand), np.int16),
                        "mirrored": stack("mirrored", (k_pts,), np.float32),
                    },
                )
            )
        return results

    def _require_loaded(self) -> None:
        if self._backbone is None or not self._learned or self._plan is None:
            raise RuntimeError(
                f"{type(self).__name__}.load() must run before predict_batch()"
            )

    def score_held_out(self, progress=None) -> list[dict]:
        """Each example frame scored by the **other** examples: the user's clicks are the
        answer key, so this is how far the method lands from where a person put each point,
        and what score it gave, on frames like the ones it learned from.

        One row per (example frame, point the user marked visible): ``frame`` (index in the
        set), ``point``, ``score``, ``error_px`` (NaN when the method placed nothing) and
        ``body_px`` (the subject's ``sqrt(w*h)``). Reuses the grids ``load()`` computed: the
        bank is rebuilt without frame *i*'s rows (a stack, no backbone call) and the skeleton
        statistics without its clicks. No mask is applied (example frames keep none), so this
        is the method on the cell grid alone. Empty with fewer than two example frames.
        ``progress(done, total)`` hears each example frame as it is scored.

        A per-class method scores each class's frames against that class's other frames
        only — its own bank, as a detection of that class is matched — and each row also
        carries ``class``; ``frame`` is then the index within that class's set.
        """
        if not self._learned or self._plan is None:
            raise RuntimeError(f"{type(self).__name__}.load() must run first")
        total = sum(len(learned.held["clicks"]) for learned in self._learned.values())
        out: list[dict] = []
        done = 0
        for key, learned in self._learned.items():
            rows = self._held_out_rows(learned, progress, done, total)
            if self.per_class:
                rows = [{**row, "class": key} for row in rows]
            out.extend(rows)
            done += len(learned.held["clicks"])
        return out

    def _held_out_rows(self, learned: _Learned, progress, done: int, total: int):
        held = learned.held
        clicks, views, grids = held["clicks"], held["views"], held["grids"]
        if len(clicks) < 2:
            return []
        names = self._plan.names
        out: list[dict] = []
        for i, (pts, scale) in enumerate(clicks):
            if progress is not None:
                progress(done + i, total)
            if not pts:
                continue
            others = {
                n: [r for r in held["rows"].get(n, []) if r[0] != i] for n in names
            }
            try:
                bank = ExampleBank.build(names, others)
            except ValueError:
                continue  # no other example marks anything visible
            stats = _decode.learn_skeleton(
                [c for j, c in enumerate(clicks) if j != i],
                self._plan,
                self.settings["side_rule"],
            )
            maps, _winners = bank.score_maps(grids[i], None)
            placed = self._place(
                _decode.candidates(maps, views[i]), views[i], scale, stats
            )
            for name, (cx, cy) in pts.items():
                if name in placed:
                    (x, y), score, _cell = placed[name]
                    error = float(np.hypot(x - cx, y - cy))
                else:
                    score, error = float("nan"), float("nan")
                out.append(
                    {
                        "frame": i,
                        "point": name,
                        "score": float(score),
                        "error_px": error,
                        "body_px": float(scale),
                    }
                )
        return out

    def _place(
        self,
        cands: dict,
        view: View,
        scale_px: float,
        stats: _decode.SkeletonStats,
    ) -> dict:
        """One subject's points from its candidates: ``{target: (xy, score, cell)}``."""
        if self.settings["decode"] == "per_point":
            return _decode.decode_per_point(cands)
        return _decode.decode_skeleton(
            cands, self._plan, stats, scale_px, view.patch_px
        )

    def resolved(self) -> dict:
        """What ``load()`` decided from the clicks — each left/right pair's side rule (``auto``
        picks per pair) — and the matching constants the run should record; per class
        (``by_class``) for a per-class method, since each class learns from its own clicks.
        Empty before load.
        """
        if self._plan is None or not self._learned:
            return {}

        def side_rules(learned: _Learned) -> dict:
            return {
                f"{left}/{right}": learned.stats.rules.get((left, right), "none")
                for left, right in self._plan.pairs
            }

        if self.per_class:
            return {
                "by_class": {
                    key: {"side_rules": side_rules(learned)}
                    for key, learned in self._learned.items()
                },
                "matching": dict(_decode.MATCHING),
            }
        return {
            "side_rules": side_rules(self._learned[None]),
            "matching": dict(_decode.MATCHING),
        }
