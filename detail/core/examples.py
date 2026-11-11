"""Example sets (§26 "Example sets"): decoded frames with the user's marks on them.

An example set is experiment-level — one set serves every recording, which is the
held-out-animal protocol the research measured (clicks on one animal, applied to others). With
``purpose: "example"`` its marks are the input a patch-matching keypoint method learns from:
user input, the way an exemplar clip is in §20, so they build the method legitimately and
nothing here reads ground truth. With ``purpose: "audit"`` the same storage holds the frames a
person labels to *score* a run (§26 "Auditing"); the two purposes never mix — a frame cannot be
added to a set of the other purpose, and the scorer excludes the frames a run recorded as its
examples.

Today a mark is a clicked **point** (or "not visible"). The ``kind`` field reserves ``region``
and ``box``, and ``positive: false`` reserves "not this", so a later readout adds a kind rather
than a schema. **Unknown fields are kept**: ``from_dict`` carries anything it does not know in
``extra`` and ``to_dict`` writes it back, so a save by older code never deletes what newer code
added (the storage-review class D7).

Pure data and validation. The files (``examples/<id>/examples.yaml`` + ``frames/<sha>.png``)
are ``storage/example_sets.py``'s (§26 *Sequencing*, PR 2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from detail.core.skeleton import Skeleton

#: The kinds of mark a frame may carry. Only ``point`` has a readout today.
MARK_KINDS = ("point", "region", "box")
#: What a set is for. Example marks build a method; audit marks score one.
PURPOSES = ("example", "audit")
_MARK_KEYS = frozenset({"target", "kind", "x", "y", "visible", "positive"})
_FRAME_KEYS = frozenset(
    {
        "image",
        "recording_id",
        "tracking_run_id",
        "track_id",
        "part_id",
        "frame_idx",
        "t_ns",
        "bbox",
        "marks",
        "sampled_by",
        "round",
        "skipped",
    }
)
_SET_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "name",
        "purpose",
        "created_at",
        "skeleton",
        "skeleton_ref",
        "frames",
        "sources",
        "digest",
    }
)


def _digest(content: dict) -> str:
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ExampleMark:
    """One mark on one frame: a ``target`` (a skeleton point's name) and where it is.

    A ``point`` that is ``visible`` carries normalized full-frame ``x``, ``y``; a point marked
    "not visible" carries none. ``positive: False`` is reserved for "not this" (a negative
    example); ``kind`` ``region`` / ``box`` are reserved and carried without a readout.
    """

    target: str
    kind: str = "point"
    x: float | None = None
    y: float | None = None
    visible: bool = True
    positive: bool = True
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in MARK_KINDS:
            raise ValueError(
                f"mark on {self.target!r} has kind {self.kind!r}; expected one of {MARK_KINDS}"
            )
        if self.kind == "point" and self.visible and (self.x is None or self.y is None):
            raise ValueError(f"visible point mark on {self.target!r} has no x, y")

    def to_dict(self) -> dict:
        out: dict = {"target": self.target, "kind": self.kind}
        if self.x is not None and self.y is not None:
            out["x"] = float(self.x)
            out["y"] = float(self.y)
        out["visible"] = bool(self.visible)
        if not self.positive:
            out["positive"] = False
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ExampleMark":
        kind = str(d.get("kind", "point"))
        if kind not in MARK_KINDS:
            raise ValueError(
                f"unknown mark kind {kind!r}; expected one of {MARK_KINDS}"
            )
        return cls(
            target=str(d["target"]),
            kind=kind,
            x=None if d.get("x") is None else float(d["x"]),
            y=None if d.get("y") is None else float(d["y"]),
            visible=bool(d.get("visible", True)),
            positive=bool(d.get("positive", True)),
            extra={k: v for k, v in d.items() if k not in _MARK_KEYS},
        )


@dataclass
class ExampleFrame:
    """One copied-in frame, where it came from, and the marks on it.

    ``marks`` holds the targets decided on this frame; a target absent from it has not been
    marked yet (different from a mark saying it is not visible). ``part_id`` is the video
    part's id (``part_001``), never a position in a list. ``frame_idx`` counts within that part
    and is kept for display; rows match on ``t_ns``.
    """

    image: str  # relative to the set's directory, posix: "frames/<sha>.png"
    recording_id: str
    tracking_run_id: str
    track_id: int
    part_id: str
    frame_idx: int
    t_ns: int
    bbox: tuple[float, float, float, float]  # the tracked subject's box, normalized
    marks: list[ExampleMark] = field(default_factory=list)
    sampled_by: str | None = None  # how the sampler chose it; None = hand-picked
    round: int | None = None  # audit sets: the round it was drawn in
    skipped: bool = False  # audit sets: the person could not label it
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple[str, str, int, int]:
        """``(recording_id, part_id, t_ns, track_id)`` — how a run's rows name this frame."""
        return (self.recording_id, self.part_id, int(self.t_ns), int(self.track_id))

    def mark_for(self, target: str) -> ExampleMark | None:
        for mark in self.marks:
            if mark.target == target:
                return mark
        return None

    def to_dict(self) -> dict:
        out: dict = {
            "image": self.image,
            "recording_id": self.recording_id,
            "tracking_run_id": self.tracking_run_id,
            "track_id": int(self.track_id),
            "part_id": self.part_id,
            "frame_idx": int(self.frame_idx),
            "t_ns": int(self.t_ns),
            "bbox": [float(v) for v in self.bbox],
        }
        if self.sampled_by is not None:
            out["sampled_by"] = self.sampled_by
        if self.round is not None:
            out["round"] = int(self.round)
        if self.skipped:
            out["skipped"] = True
        out["marks"] = [m.to_dict() for m in self.marks]
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ExampleFrame":
        return cls(
            image=str(d["image"]),
            recording_id=str(d["recording_id"]),
            tracking_run_id=str(d["tracking_run_id"]),
            track_id=int(d["track_id"]),
            part_id=str(d["part_id"]),
            frame_idx=int(d["frame_idx"]),
            t_ns=int(d["t_ns"]),
            bbox=tuple(float(v) for v in d["bbox"]),
            marks=[ExampleMark.from_dict(m) for m in d.get("marks") or []],
            sampled_by=None if d.get("sampled_by") is None else str(d["sampled_by"]),
            round=None if d.get("round") is None else int(d["round"]),
            skipped=bool(d.get("skipped", False)),
            extra={k: v for k, v in d.items() if k not in _FRAME_KEYS},
        )


@dataclass
class ExampleSet:
    """A skeleton copy, its purpose, and the frames with their marks.

    ``sources`` (audit sets) maps recording id -> tracking run id: the draw's population, kept
    so a later round draws from the same one.
    """

    id: str
    name: str
    purpose: str
    created_at: str
    skeleton: Skeleton
    frames: list[ExampleFrame] = field(default_factory=list)
    sources: dict[str, str] | None = None
    # The library skeleton this set's copy came from (§26 *The Keypoints manager*):
    # ``{id, name, digest}``; None for a set made before the library, or from a file.
    skeleton_ref: dict | None = None
    extra: dict = field(default_factory=dict)

    # -- what a method or an audit asks ------------------------------------- #
    def visible_counts(self) -> dict[str, int]:
        """Per skeleton point, the number of frames that mark it visible."""
        counts = {name: 0 for name in self.skeleton.names}
        for frame in self.frames:
            for mark in frame.marks:
                if mark.visible and mark.positive and mark.target in counts:
                    counts[mark.target] += 1
        return counts

    def labeled(self, frame_index: int) -> bool:
        """Every skeleton point has a mark (or "not visible") on this frame, and it was not
        skipped."""
        frame = self.frames[frame_index]
        marked = {m.target for m in frame.marks}
        return not frame.skipped and all(n in marked for n in self.skeleton.names)

    def pending(self) -> list[int]:
        """Indices of frames still to mark, in the order they are to be shown."""
        return [
            i
            for i, frame in enumerate(self.frames)
            if not frame.skipped and not self.labeled(i)
        ]

    def counts(self) -> dict[str, int]:
        """``drawn``, ``labeled``, ``skipped`` and ``pending`` frames."""
        labeled = sum(1 for i in range(len(self.frames)) if self.labeled(i))
        skipped = sum(1 for f in self.frames if f.skipped)
        return {
            "drawn": len(self.frames),
            "labeled": labeled,
            "skipped": skipped,
            "pending": len(self.frames) - labeled - skipped,
        }

    def next_round(self) -> int:
        return max((f.round or 0 for f in self.frames), default=0) + 1

    def check_runnable(self) -> None:
        """Raise ``ValueError`` saying what to fix before a run can learn from this set."""
        if self.purpose != "example":
            raise ValueError(
                f"example set {self.id!r} has purpose {self.purpose!r}; a run learns only "
                "from an example set, never from audit labels"
            )
        try:
            self.skeleton.validate()
        except ValueError as exc:
            raise ValueError(f"example set {self.id!r}: {exc}") from None
        if not self.frames:
            raise ValueError(
                f"example set {self.id!r} has no frames; sample frames and mark the "
                "keypoints on them first"
            )
        names = set(self.skeleton.names)
        for i, frame in enumerate(self.frames):
            for mark in frame.marks:
                mark.validate()
            stale = sorted({m.target for m in frame.marks} - names)
            if stale:
                raise ValueError(
                    f"example set {self.id!r} frame {i} has marks for {stale}, which "
                    "the skeleton does not name"
                )
        missing = [n for n, c in self.visible_counts().items() if c == 0]
        if missing:
            raise ValueError(
                f"example set {self.id!r}: no frame marks {missing} visible; mark "
                "each keypoint on at least one frame where it can be seen"
            )

    # -- identity and serialization --------------------------------------- #
    def digest(self, image_hashes: dict[str, str]) -> str:
        """A content hash of the skeleton copy, every mark and each image's own hash.

        ``image_hashes`` maps a frame's ``image`` to its file hash (the storage layer knows
        the files); swapping an image under the same name changes the digest. The set's
        ``name`` is a label and is left out.
        """
        content = {
            "skeleton": self.skeleton.to_dict(),
            "frames": [
                {
                    "image": f.image,
                    "sha256": image_hashes.get(f.image),
                    "key": list(f.key()),
                    "marks": sorted(
                        (m.to_dict() for m in f.marks), key=lambda m: m["target"]
                    ),
                }
                for f in self.frames
            ],
        }
        return _digest(content)

    def to_dict(self, image_hashes: dict[str, str] | None = None) -> dict:
        """The YAML body, ``schema_version`` first (Tenet 7); ``digest`` when the storage
        layer passes the image hashes."""
        out: dict = {
            "schema_version": 1,
            "id": self.id,
            "name": self.name,
            "purpose": self.purpose,
            "created_at": self.created_at,
            "skeleton": self.skeleton.to_dict(),
        }
        if self.skeleton_ref is not None:
            out["skeleton_ref"] = dict(self.skeleton_ref)
        if self.sources is not None:
            out["sources"] = dict(self.sources)
        out["frames"] = [f.to_dict() for f in self.frames]
        out.update(self.extra)
        if image_hashes is not None:
            out["digest"] = self.digest(image_hashes)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "ExampleSet":
        purpose = str(d.get("purpose") or "example")
        if purpose not in PURPOSES:
            raise ValueError(
                f"example set {d.get('id')!r} has purpose {purpose!r}; expected one of "
                f"{PURPOSES}"
            )
        sources = d.get("sources")
        return cls(
            id=str(d["id"]),
            name=str(d.get("name") or d["id"]),
            purpose=purpose,
            created_at=str(d.get("created_at") or ""),
            skeleton=Skeleton.from_dict(d["skeleton"]),
            frames=[ExampleFrame.from_dict(f) for f in d.get("frames") or []],
            sources=(
                None
                if sources is None
                else {str(k): str(v) for k, v in sources.items()}
            ),
            skeleton_ref=(
                dict(d["skeleton_ref"]) if d.get("skeleton_ref") is not None else None
            ),
            extra={k: v for k, v in d.items() if k not in _SET_KEYS},
        )
