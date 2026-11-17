"""Frame codebook + temporal decode (§22) — a behavior vocabulary without comparing bouts.

The other shape a temporal method can take. `dtw_finch` segments first and compares every pair of
bouts; this labels every *frame* against a codebook and lets the stretches fall out of a decode.
Measured against each other on real annotations: a tie on AMI and a loss on boundaries. So
the reason to reach for this one is not accuracy — it is where the cost sits.

**Where the quadratic lives, which is the whole argument.** Both methods have one.

- `dtw_finch`'s is over **bouts**, and the bout count grows with the study. Every bout has to be
  compared to be clustered, which is why a cap there needs a separate assignment pass.
- This method's is the **decode**, ``O(rows x codewords^2)``: linear in the study, quadratic in
  ``codewords``, which is a setting the user chooses rather than a consequence of the data. On a
  multi-recording study it is the wall-clock cost, and it runs at the finest rung whatever
  granularity is chosen later. "Assigning every frame is linear, and so is the decode" once stood
  here while a pooled multi-recording fit spent hours in the decode behind a stalled bar.

**The sample and the codeword count are two settings, because one knob doing both could not be
fixed (#573).** ``fit_rows`` was a fixed total pooled across the study: it thinned every video as
the study grew, and it also set the codeword count, since FINCH's finest level is about a quarter
of the rows it is given. A share of each recording (``fit_share``) fixes the thinning — but fed to
FINCH, a share makes the codeword count, and the decode, grow with the study. So the codebook is
built the pose two-step way: k-means into ``codewords`` groups, average linkage over them
(`_codebook_levels`). Internal evaluations measured it against the ``fit_rows`` method held out,
and found it steadier than FINCH across samples.

**The decode is not optional.** Without it this loses to the bout path outright; with it the two
tie, and `raw < min-run < Viterbi` is the most replicated effect measured. It lives here
rather than in the module because §22 puts it here: suppressing flicker is the method's business.

**One decode, at the finest rung, inherited upward.** This departs from the research protocol,
which decodes separately at each codebook size, and it is a deliberate trade:

- The decode needs *every row's features*, and a granularity change re-derives it. Holding those
  in the fit is hundreds of megabytes per recording; reloading them per change is seconds of I/O
  on every turn of an authored knob (§21).
- The codebook's rungs **nest**, so decoding once at the finest rung and mapping labels upward is
  a lookup — instant at every setting, and it makes instances nest the way `dtw_finch`'s do, so a
  name means the same thing across a granularity change.

What it costs is fidelity to the measured protocol: a coarse setting inherits a decode whose
transition prior was estimated over the finest rung's states, rather than one estimated at that
setting. Named here, and the coarse-decode path is what to try first if this method ever
underperforms its own study.

**Measured, because §22 requires it rather than an assumption.** On a real recording a
granularity change costs milliseconds,
because it is a label lookup and a run-length encode. A re-decode would be about a second per
recording per change. That is the difference between granularity being an authored knob here and
being a launch parameter, which §22 explicitly allows a method to end up needing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from detail.analysis import clip_query as clip_core
from detail.analysis import decode as decode_mod
from detail.analysis import segmentation
from detail.cluster_methods._common import (
    kmeans_level,
    measured_frame_time,
    vector_node_separations,
)
from detail.configs.temporal_codebook import TemporalCodebookConfig
from detail.core.dataclasses import (
    ClipScoring,
    ClusterInstance,
    ClusterTree,
    GranularitySpec,
    MemberPlaces,
)
from detail.core.protocols import TemporalClusterMethod
from detail.core.protocols import member_groups_of
from detail.core.registries import register_temporal_cluster_method

SEC_TO_NS = 1_000_000_000


class _Fit:
    """One fitted codebook plus the decoded labeling it produced. Opaque to the module (§22).

    Deliberately small on disk. The codebook is built from ``sample`` (a few thousand rows) and the
    decode is stored as **one label per row** at the finest rung — a few hundred kilobytes for a
    half-hour recording, against the hundreds of megabytes the rows' features themselves would be.
    Everything a coarser granularity needs is a lookup from there.
    """

    def __init__(
        self,
        sample: np.ndarray,
        levels: list[np.ndarray],
        fine_labels: np.ndarray,
        fine_distance: np.ndarray,
        min_run_s: float,
        recording_ids: tuple[str, ...],
        recording_index: np.ndarray,
        subjects: np.ndarray,
        frame_idx: np.ndarray,
        t_ns: np.ndarray,
        dim: int,
        seed: int = 0,
        tree: ClusterTree | None = None,
        sample_rows: np.ndarray | None = None,
    ) -> None:
        self.sample = sample  # (sampled, D) unit vectors the codebook was built on
        self.levels = levels  # nested partitions of `sample`, finest first
        # (rows,) the decoded codeword of **every** row, at the finest rung. The one thing that
        # had to be computed with the full trajectories in hand.
        self.fine_labels = np.asarray(fine_labels, dtype=np.int64)
        # (rows,) cosine distance from each row to the codeword it was decoded into. One float per
        # row — the same order of storage as the labels — and it is what lets `rank_members`
        # rank by how typical a run is rather than by how long it happens to be.
        self.fine_distance = np.asarray(fine_distance, dtype=np.float32)
        # The floor the fit resolved, carried so `rank_members` and `instances_at` cannot
        # disagree about which runs exist: the protocol hands params to one and not the other.
        self.min_run_s = float(min_run_s)
        self.recording_ids = recording_ids
        self.recording_index = recording_index  # (rows,)
        self.subjects = subjects  # (rows,)
        self.frame_idx = frame_idx  # (rows,)
        self.t_ns = t_ns  # (rows,) int64
        self.dim = dim
        self.seed = int(seed)
        # (sampled,) the row each sample member was taken from. A shard written before the sample
        # was drawn per recording has none, and its sample was one evenly spaced pass over the
        # pooled rows, which is what the fallback reconstructs.
        self.sample_rows = (
            np.asarray(sample_rows, dtype=np.int64)
            if sample_rows is not None
            else np.linspace(0, len(fine_labels) - 1, num=len(sample)).astype(int)
        )
        self._tree = tree
        self._labels: dict[int, np.ndarray] = {}
        self._row_labels: dict[int, np.ndarray] = {}
        self._codebooks: dict[int, np.ndarray] = {}

    @property
    def counts(self) -> list[int]:
        return [int(level.max()) + 1 for level in self.levels]

    @property
    def granularity(self) -> GranularitySpec:
        """This fit's own knob: a codebook size, with the hierarchy's partitions as the rungs.

        The range runs down to 1 whatever the rungs, because every count is a cut of the tree. A
        two-step codebook has one rung (its codewords), and a minimum read off the rungs would pin
        the knob there — the same trap the pose two-step method's ``minimum_clusters`` avoids.
        """
        counts = self.counts
        return GranularitySpec(
            kind="cluster_count",
            minimum=1 if counts else 0,
            maximum=max(counts) if counts else 0,
            rungs=tuple(counts),
        )

    @property
    def tree(self) -> ClusterTree:
        """The merge order behind ``levels``, rebuilt for a shard written before it was stored."""
        if self._tree is None:
            self._tree = segmentation.cluster_tree_from_vectors(
                self.sample, self.levels
            )
        return self._tree

    def labels_at(self, granularity: int) -> np.ndarray:
        """The partition of the **codebook sample** at ``granularity``, memoized.

        A rung returns its own stored labels; between rungs it is a cut of the merge tree, which
        both nests and costs no ``(sampled, sampled)`` allocation.
        """
        want = int(granularity)
        if want not in self._labels:
            for level, count in zip(self.levels, self.counts):
                if count == want:
                    self._labels[want] = level
                    break
            else:
                tree = self.tree
                self._labels[want] = tree.cut_at(want)[tree.leaf_of_row]
        return self._labels[want]

    def codebook_at(self, granularity: int) -> np.ndarray:
        """Unit centroids at ``granularity``, memoized.

        Empty clusters are dropped rather than left at zero: a zero centroid is equidistant from
        everything under cosine and would collect whatever the real ones did not want.
        """
        want = int(granularity)
        if want not in self._codebooks:
            labels = self.labels_at(want)
            n = int(labels.max()) + 1
            means = np.zeros((n, self.dim), dtype=np.float64)
            for cluster in range(n):
                members = self.sample[labels == cluster]
                if len(members):
                    means[cluster] = members.mean(axis=0)
            means = means[np.linalg.norm(means, axis=1) > 0]
            norms = np.linalg.norm(means, axis=1, keepdims=True)
            self._codebooks[want] = (means / np.maximum(norms, 1e-12)).astype(
                np.float32
            )
        return self._codebooks[want]

    def row_labels_at(self, granularity: int) -> np.ndarray:
        """Every row's codeword at ``granularity``, by mapping the stored decode upward.

        A lookup rather than a re-decode, which is what keeps an authored knob instant. Correct
        because the rungs nest: a finest-rung codeword belongs to exactly one coarser codeword, so
        the map is well defined and reading it off any sample member of that fine cluster gives it.
        """
        want = int(granularity)
        if want not in self._row_labels:
            fine_to_want = segmentation.parent_map(self.levels[0], self.labels_at(want))
            self._row_labels[want] = fine_to_want[self.fine_labels]
        return self._row_labels[want]

    def __len__(self) -> int:
        """Rows — frames, not codebook entries."""
        return len(self.fine_labels)


def _blocks_of(t_ns: np.ndarray, max_gap_ns: int | None) -> list[tuple[int, int]]:
    """Continuity blocks: the stretches the decode runs over, split at tracking gaps."""
    t_ns = np.asarray(t_ns, dtype=np.int64)
    if max_gap_ns is None:
        period = clip_core.nominal_period_ns(t_ns)
        max_gap_ns = int(round(1.5 * period)) if period > 0 else None
    if max_gap_ns is None or not len(t_ns):
        return [(0, len(t_ns))]
    return [(int(a), int(b)) for a, b in clip_core.segment_runs(t_ns, max_gap_ns)]


def _runs_of(labels: np.ndarray, min_rows: int, blocks) -> list[tuple[int, int, int]]:
    """Maximal runs of one label within each block, as inclusive ``(start, end, label)``.

    Runs never cross a block: a tracking gap separates two stretches even when the label matches,
    because the animal was not observed in between and an instance spanning that would claim time
    nobody saw.
    """
    out = []
    for lo, hi in blocks:
        for start, stop, label in decode_mod.run_spans(labels, lo, hi):
            if stop - start >= min_rows:
                out.append((int(start), int(stop) - 1, int(label)))
    return out


@register_temporal_cluster_method("codebook_viterbi")
class CodebookViterbiMethod(TemporalClusterMethod):
    """Label every frame against a codebook, decode the sequence, encode the runs."""

    method_id = "codebook_viterbi"
    config_model = TemporalCodebookConfig
    # Every frame gets a codeword; the decode moves labels around but assigns nothing to nothing.
    has_noise = False
    supports_apply = False
    #: k-means builds the codebook: scikit-learn, from the `analysis` extra, imported in the method.
    requires = ("sklearn",)
    shard_suffix = ".npz"

    granularity = GranularitySpec(kind="cluster_count", minimum=2, maximum=0, rungs=())

    def granularity_of(self, fit: _Fit) -> GranularitySpec:
        return fit.granularity

    def hierarchy_of(self, fit: _Fit) -> ClusterTree:
        """The merge order behind this fit's rungs."""
        return fit.tree

    def fit(
        self,
        trajectories: Sequence[tuple[str, int, np.ndarray, np.ndarray]],
        params: dict,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> _Fit:
        """Build the codebook, label every frame, decode once at the finest rung."""
        model = TemporalCodebookConfig.model_validate(params)
        report = progress_cb or (lambda fraction, phase: None)
        report(0.0, "loading")

        if not trajectories:
            raise ValueError(
                "temporal clustering needs at least one trajectory to fit on"
            )

        max_gap_ns = (
            None if model.max_gap_s is None else int(round(model.max_gap_s * SEC_TO_NS))
        )
        window_ns = int(round(model.window_s * SEC_TO_NS))

        rec_ids: list[str] = []
        feats, rec_idx, subjects, frames, times, blocks = [], [], [], [], [], []
        sample_rows: list[np.ndarray] = []
        offset = 0
        for recording_id, subject, features, t_ns in trajectories:
            features = np.asarray(features)
            t_ns = np.asarray(t_ns, dtype=np.int64)
            if not len(features):
                continue
            if recording_id not in rec_ids:
                rec_ids.append(recording_id)
            window = max(1, clip_core.band_rows(t_ns, window_ns)) if window_ns else 1
            pooled, row_of_x, _runs = segmentation.build_features(
                features,
                _blocks_of(t_ns, max_gap_ns),
                window=window,
                pool="mean",
                feat_stride=1,
            )
            rows = np.asarray(row_of_x)
            feats.append(_unit(np.asarray(pooled, dtype=np.float64)))
            rec_idx.append(
                np.full(len(rows), rec_ids.index(recording_id), dtype=np.int64)
            )
            subjects.append(np.full(len(rows), int(subject), dtype=np.int64))
            frames.append(np.arange(len(features), dtype=np.int64)[rows])
            times.append(t_ns[rows])
            blocks.extend(
                (offset + a, offset + b) for a, b in _blocks_of(t_ns[rows], max_gap_ns)
            )
            if len(rows):
                # The same share of every trajectory, so each video is sampled at one density
                # whatever the study's size — a fixed total thinned every video as it grew.
                take = min(
                    len(rows), max(2, math.ceil(len(rows) * model.fit_share / 100))
                )
                sample_rows.append(
                    offset + np.linspace(0, len(rows) - 1, num=take).astype(int)
                )
            offset += len(rows)

        if not feats:
            raise ValueError("every selected trajectory was empty; nothing to cluster")

        all_features = np.ascontiguousarray(np.vstack(feats))
        all_t_ns = np.concatenate(times)

        index = np.concatenate(sample_rows)
        sample = np.ascontiguousarray(all_features[index])
        report(0.2, f"building a codebook from {len(sample):,} rows")
        levels = _codebook_levels(sample, model)

        labeling = f"labeling {len(all_features):,} frames"
        report(0.55, labeling)
        fit = _Fit(
            sample=sample,
            levels=levels,
            fine_labels=np.zeros(len(all_features), dtype=np.int64),
            fine_distance=np.zeros(len(all_features), dtype=np.float32),
            min_run_s=model.min_run_s,
            recording_ids=tuple(rec_ids),
            recording_index=np.concatenate(rec_idx),
            subjects=np.concatenate(subjects),
            frame_idx=np.concatenate(frames),
            t_ns=all_t_ns,
            dim=sample.shape[1],
            seed=model.seed,
            sample_rows=index,
        )
        finest = max(fit.counts)
        centroids = fit.codebook_at(finest)
        n_rows = len(all_features)

        # Scoring every row against the codebook is a (rows, states) matrix if done in one
        # matmul over the whole pooled study — memory this method's cost argument (bounded by
        # `codewords`, not by the study's size) never budgeted for. No block crosses a recording
        # boundary (they are built per-trajectory above), so chunking by continuity block bounds
        # peak memory to one recording's rows regardless of how many are pooled — the scale a
        # single dense matmul was fine at.
        # Pooling many recordings against a large codebook once allocated two
        # float64 matrices of tens of GB at once and the OOM killer took the process.
        raw = np.empty(n_rows, dtype=np.int64)
        rows_scored = 0
        for lo, hi in blocks:
            raw[lo:hi] = np.argmax(all_features[lo:hi] @ centroids.T, axis=1)
            rows_scored += hi - lo
            report(0.55 + 0.15 * rows_scored / max(n_rows, 1), labeling)

        report(0.7, "decoding")
        sigma_sq = decode_mod.estimate_sigma_sq(
            sample, centroids, np.asarray(np.argmax(sample @ centroids.T, axis=1))
        )
        transition = decode_mod.fit_transition(raw, blocks, len(centroids))

        # The decode already restarts at each continuity block (no state carries across one), so
        # decoding block-by-block is identical to decoding them all in one call — it just never
        # materializes a (rows, states) emission or similarity matrix over the pooled study.
        #
        # Both loops report per block, weighted by the block's *rows* rather than by its index:
        # continuity blocks differ by orders of magnitude, and `i / len(blocks)` would price a
        # one-row block the same as a 90,000-row one, racing the bar ahead and then stalling it.
        # The decode is O(rows × states²) with the state count fixed, so rows are what a block
        # costs. The message is held constant across the loop so the runner's `ThrottledProgress`
        # governs the cadence — it forces an event on a *changed* message, which is for phase
        # boundaries, not for every block.
        #
        # The granularity floor this leaves: one continuity block is one atomic `viterbi` call, so
        # a gap-free single recording still reports once. Nothing can subdivide it without
        # changing the decode — the forward pass carries state across the whole block.
        fine_labels = np.empty(n_rows, dtype=np.int64)
        fine_distance = np.empty(n_rows, dtype=np.float32)
        rows_decoded = 0
        for lo, hi in blocks:
            chunk = all_features[lo:hi]
            emission_chunk = decode_mod.gaussian_emission(chunk, centroids, sigma_sq)
            labels_chunk = decode_mod.viterbi(
                emission_chunk, [(0, hi - lo)], transition
            )
            fine_labels[lo:hi] = labels_chunk
            # The chosen centroid's own cosine distance, gathered row-by-row (the same trick
            # `estimate_sigma_sq` already uses) rather than sliced out of a full similarity
            # matrix nothing else needs.
            fine_distance[lo:hi] = 1.0 - (chunk * centroids[labels_chunk]).sum(axis=1)
            rows_decoded += hi - lo
            report(0.7 + 0.2 * rows_decoded / max(n_rows, 1), "decoding")
        fit.fine_labels = fine_labels
        fit.fine_distance = fine_distance
        report(1.0, "decoding")
        return fit

    def instances_at(
        self, fit: _Fit, granularity: int, params: dict
    ) -> list[ClusterInstance]:
        """Run-length encode the decoded labeling at this granularity.

        The decode already ran, once, at the finest rung; this maps its labels upward and encodes
        the runs. So a granularity change is a lookup and an encode, not another decode — which is
        what makes the knob authored rather than a launch parameter (§21).
        """
        model = TemporalCodebookConfig.model_validate(params)
        min_rows = max(1, int(round(model.min_run_s * _rows_per_second(fit))))
        max_gap_ns = (
            None if model.max_gap_s is None else int(round(model.max_gap_s * SEC_TO_NS))
        )
        labels = fit.row_labels_at(granularity)
        blocks = _unit_blocks(fit, max_gap_ns)
        return [
            ClusterInstance(start, end, cluster)
            for start, end, cluster in _runs_of(labels, min_rows, blocks)
        ]

    def member_leaves(self, fit: _Fit) -> np.ndarray:
        """Every frame's decoded finest codeword, which is its leaf of the tree."""
        return np.asarray(fit.fine_labels, dtype=np.int64)

    def member_places(self, fit: _Fit) -> MemberPlaces:
        """Every decoded frame's recording, subject and measured time (see ``MemberPlaces``).

        Blocks are cut per recording and subject first, then at tracking gaps, so a frame's time
        never reaches into the next video or across time nobody observed.
        """
        t_ns = np.asarray(fit.t_ns, dtype=np.int64)
        blocks = _unit_blocks(fit, None)
        return MemberPlaces(
            recording_ids=tuple(fit.recording_ids),
            recording_index=np.asarray(fit.recording_index, dtype=np.int64),
            subject=np.asarray(fit.subjects, dtype=np.int64),
            start_ns=t_ns,
            end_ns=t_ns,
            time_ns=measured_frame_time(t_ns, blocks),
        )

    def rank_members(
        self, fit: _Fit, leaves: np.ndarray
    ) -> list[tuple[ClusterInstance, float]]:
        """The node's runs, most typical first: a run's mean distance to its decoded codewords.

        A run's score is the mean cosine distance from its rows to the codeword they were decoded
        into — how *typical* the stretch is, not how long it is. Length alone would rank a long
        drifting stretch above a short unambiguous one, which is the opposite of what a person
        naming a node wants to look at.

        The runs are the ones ``instances_for`` gives the node with everything else unlabelled,
        under the fit's own floor rather than params (the protocol hands params to
        ``instances_for`` and not to this), so the ordering is over runs that exist.
        """
        groups = np.full(fit.tree.n_leaves, -1, dtype=np.int64)
        groups[np.asarray(leaves, dtype=np.int64)] = 0
        runs = [
            i
            for i in self.instances_for(fit, groups, {"min_run_s": fit.min_run_s})
            if i.cluster_id == 0
        ]
        scored = sorted(
            (
                (float(np.mean(fit.fine_distance[i.start_row : i.end_row + 1])), i)
                for i in runs
            ),
            key=lambda pair: pair[0],
        )
        return [(instance, score) for score, instance in scored]

    def clip_scoring(
        self, fit: _Fit, params: dict, trajectories: Sequence | None = None
    ) -> ClipScoring:
        """Each frame's distance to the codeword it was decoded into, which no node changes.

        So a clip's score is its frames' mean distance, the very score ``rank_members`` sorts a
        run by, and the cache keeps it as a sum per leaf run. The minimum length is the fit's own
        floor, as ``rank_members`` uses, in frames. The continuity blocks are ``instances_for``'s
        under ``params``, so an explicit ``max_gap_s`` cuts clips exactly where it cuts instances.
        """
        return ClipScoring(
            kind="distance",
            min_members=max(1, int(round(fit.min_run_s * _rows_per_second(fit)))),
            join=True,
            member_distance=np.asarray(fit.fine_distance, dtype=np.float64),
            member_block_start=self.row_block_starts(fit, params),
        )

    def row_block_starts(self, fit: _Fit, params: dict) -> np.ndarray:
        """The first row of every continuity block ``instances_for`` cuts runs on, under
        ``params``: the same blocks, so a fused instance and a clip stop where a run does.
        """
        model = TemporalCodebookConfig.model_validate(params)
        max_gap_ns = (
            None if model.max_gap_s is None else int(round(model.max_gap_s * SEC_TO_NS))
        )
        starts = np.zeros(len(fit.fine_labels), dtype=bool)
        for lo, hi in _unit_blocks(fit, max_gap_ns):
            if hi > lo:
                starts[lo] = True
        return starts

    def node_separations(self, fit: _Fit) -> np.ndarray:
        """Centroid distance over spread for every node, from the codebook sample."""
        return vector_node_separations(fit.tree, fit.sample, fit.levels[0])

    def instances_for(
        self,
        fit: _Fit,
        leaf_groups: np.ndarray,
        params: dict,
        member_groups: np.ndarray | None = None,
    ) -> list[ClusterInstance]:
        """Run-length encode the decoded labelling mapped through ``leaf_groups``.

        The decode ran once, at the finest rung, so this is a lookup and an encode, exactly as a
        cut is: ``instances_at`` is this with the cut's clusters as the groups. ``member_groups``
        overrides frames after the lookup and before the encode (``member_groups_of``).
        """
        model = TemporalCodebookConfig.model_validate(params)
        min_rows = max(1, int(round(model.min_run_s * _rows_per_second(fit))))
        max_gap_ns = (
            None if model.max_gap_s is None else int(round(model.max_gap_s * SEC_TO_NS))
        )
        groups = member_groups_of(leaf_groups, self.member_leaves(fit), member_groups)
        blocks = _unit_blocks(fit, max_gap_ns)
        return [
            ClusterInstance(start, end, group)
            for start, end, group in _runs_of(groups, min_rows, blocks)
        ]

    def fit_summary(self, fit: _Fit) -> dict:
        """The rows labeled and the rows the codebook was built from.

        The second is the one worth recording: it grows with the study, at the same share of every
        recording, and a user reading a fit should see how many rows the codebook actually saw.
        """
        return {
            "n_labeled_rows": int(len(fit)),
            "n_codebook_rows": int(len(fit.sample)),
        }

    def resolve(self, fit: _Fit, instance) -> dict:
        """Where a run of frames sits. Its span is its first and last row's, not one row's."""
        start, end = int(instance.start_row), int(instance.end_row)
        if fit.recording_index[start] != fit.recording_index[end]:
            return {}
        return {
            "recording_id": fit.recording_ids[int(fit.recording_index[start])],
            "subject": int(fit.subjects[start]),
            "start_frame": int(fit.frame_idx[start]),
            "end_frame": int(fit.frame_idx[end]),
            "t_ns": int(fit.t_ns[start]),
            # The **last row's** timestamp, not a period past it. Adding the period is right for a
            # coverage span, which measures time observed, and wrong for an instance: consecutive
            # runs would then end exactly where the next begins, and §19 reads two spans sharing
            # an instant as overlapping — which an `exclusive` manifest refuses. The bout method
            # ends on its last row for the same reason.
            "end_t_ns": int(fit.t_ns[end]),
        }

    def serialize(self, fit: _Fit, path: Path) -> None:
        np.savez_compressed(
            path,
            sample=fit.sample,
            sample_rows=fit.sample_rows,
            levels=np.stack(fit.levels) if fit.levels else np.zeros((0, 0), np.int64),
            fine_labels=fit.fine_labels,
            fine_distance=fit.fine_distance,
            min_run_s=fit.min_run_s,
            recording_ids=np.asarray(fit.recording_ids, dtype=object),
            recording_index=fit.recording_index,
            subjects=fit.subjects,
            frame_idx=fit.frame_idx,
            t_ns=fit.t_ns,
            seed=fit.seed,
            **segmentation.tree_arrays(fit.tree),
        )

    def load(self, path: Path) -> _Fit:
        with np.load(path, allow_pickle=True) as data:
            sample = data["sample"]
            return _Fit(
                sample=sample,
                levels=[np.asarray(row) for row in data["levels"]],
                fine_labels=data["fine_labels"],
                fine_distance=data["fine_distance"],
                min_run_s=float(data["min_run_s"]),
                recording_ids=tuple(str(r) for r in data["recording_ids"]),
                recording_index=data["recording_index"],
                subjects=data["subjects"],
                frame_idx=data["frame_idx"],
                t_ns=data["t_ns"],
                dim=sample.shape[1] if sample.size else 0,
                # A fit from an older build also carries `projection_points`,
                # `projection_layout` and `coords` (the retired 2-D layout); they are not read.
                seed=int(data["seed"]),
                tree=segmentation.tree_from_arrays(data),
                sample_rows=(
                    data["sample_rows"] if "sample_rows" in data.files else None
                ),
            )


def _codebook_levels(sample: np.ndarray, model: TemporalCodebookConfig) -> list:
    """One level: the sample's k-means groups, compacted to 0..K-1. The codebook's finest rung.

    k-means rather than FINCH's first level, because FINCH's finest level is whatever size the
    sample makes it (about a quarter of the rows), and that size is the decode's state count: a
    share-based sample would have made the decode grow with the study. The tree above the groups
    is average linkage (``cluster_tree_from_vectors``), the pose two-step shape.
    """
    return kmeans_level(sample, model.codewords, model.seed)


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0.0, norms, 1.0)


def _rows_per_second(fit: _Fit) -> float:
    """Rows per second from the fit's own timestamps — what every seconds knob resolves against.

    **Per unit, never pooled.** Every recording's clock starts at zero, so the study's timestamps
    sorted together interleave dozens of timelines, and the median step between them is the jitter
    between recordings rather than a frame period: pooled, it read a rate hundreds of
    times too high, turned a sub-second floor into thousands of rows, and kept no run.
    """
    t_ns = np.asarray(fit.t_ns, dtype=np.int64)
    steps = np.concatenate(
        [np.diff(t_ns[lo:hi]) for lo, hi in _unit_seams(fit)] or [np.zeros(0, np.int64)]
    )
    steps = steps[steps > 0]
    return 1e9 / float(np.median(steps)) if len(steps) else 1.0


def _unit_seams(fit: _Fit) -> list[tuple[int, int]]:
    """``[lo, hi)`` row ranges of one recording and one subject each."""
    unit = np.asarray(fit.recording_index, dtype=np.int64) * (
        int(np.max(fit.subjects, initial=0)) + 1
    ) + np.asarray(fit.subjects, dtype=np.int64)
    edges = [0, *(np.nonzero(np.diff(unit) != 0)[0] + 1).tolist(), len(unit)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def _unit_blocks(fit: _Fit, max_gap_ns: int | None) -> list[tuple[int, int]]:
    """Continuity blocks over the pooled rows: cut per recording and subject first, then at gaps.

    ``_blocks_of`` on the pooled timeline cuts only where time jumps *forward*; the next
    recording starts back at zero, which is a step backwards, so a run of one label could carry
    on from the end of one video into the start of the next — a "run" no recording holds, which
    ``resolve`` then rightly refuses to place.
    """
    t_ns = np.asarray(fit.t_ns, dtype=np.int64)
    blocks = []
    for lo, hi in _unit_seams(fit):
        blocks.extend((lo + a, lo + b) for a, b in _blocks_of(t_ns[lo:hi], max_gap_ns))
    return blocks
