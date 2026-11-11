"""DTW + FINCH temporal clustering (§22) — a vocabulary of behaviors, one cluster per kind of bout.

Segment each trajectory into bouts, compare every pair with banded DTW, then group the matrix with
FINCH. The pieces are the shared ones in `analysis/bout_segmentation.py` and
`analysis/segmentation.py`, in use since §17; what is new here is the
protocol around them, which turns a single flat fit into an **authored** one: the matrix is computed
once and the granularity is chosen afterwards.

**Why the rule is the method rather than a parameter of one.** ``granularity`` and ``has_noise`` are
class declarations, and both change meaning with the rule: FINCH offers a cluster count whose
partitions are nested and labels every bout; HDBSCAN offers a ``min_cluster_size`` and leaves some
unassigned. One method taking ``cluster_rule`` as a parameter would declare a knob and a coverage
guarantee that are false for some of its own settings. FINCH is the default on measurement — binseg
+ FINCH at k=30 beat HDBSCAN by +0.125 mean AMI on both folds — and the
other rules can register as their own methods with no change to the module.

**Granularity is cheap here, and that is measured rather than hoped.** §21 requires that changing it
not trigger the fit. The DTW matrix depends only on the segmentation knobs, so a re-cluster reads it
and re-runs the grouping alone — the property the removed `cluster_fit.refit_from_cache` relied
on ("seconds, not hours"). That is what makes the granularity an authored decision for this method.

**The cost is the comparison, and it is quadratic.** ``max_segments`` bounds it by evenly
subsampling bouts, and a capped fit looks exactly like a full one on every screen — so
``subsample_note`` exists and the interpretation surface states it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from detail.analysis import bout_segmentation
from detail.analysis import clip_query as clip_core
from detail.analysis import segmentation
from detail.configs.temporal_dtw_finch import TemporalDtwFinchConfig
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
    """One fitted vocabulary of bouts. Opaque to the module (§22) — only this method reads it.

    Holds the DTW matrix and, per bout, everything needed to resolve it back to pixels: which
    recording, which subject, and the frame/time span. That is the same content the removed
    `storage/clusterers.py`'s distance cache carried, because it was the same matrix serving the
    same purpose — a re-cluster without a re-segmentation.

    **Two kinds of bout, in one order.** The provenance arrays run ``K`` *clustered* bouts followed
    by ``D`` *assigned* ones: the clustered bouts are what the matrix compared and the vocabulary
    was learned from, the assigned ones are what ``max_segments`` kept out of that comparison and
    which were placed against the result afterwards. ``dist`` covers only the first ``K``, because
    the matrix is about what was clustered. Everything a *user*
    reads — instances, the cluster table, coverage — covers all ``K + D``, because a bout that was
    segmented is a bout that happened.
    """

    def __init__(
        self,
        dist: np.ndarray,
        recording_ids: tuple[str, ...],
        recording_index: np.ndarray,
        subjects: np.ndarray,
        start_frames: np.ndarray,
        end_frames: np.ndarray,
        start_t_ns: np.ndarray,
        end_t_ns: np.ndarray,
        assigned_fine_cluster: np.ndarray | None = None,
        assigned_distance: np.ndarray | None = None,
        tree: ClusterTree | None = None,
    ) -> None:
        self.dist = dist  # (K, K) float32 banded-DTW distances between clustered bouts
        self.recording_ids = recording_ids
        # All (K + D,), clustered first then assigned.
        self.recording_index = recording_index  # index into `recording_ids`
        self.subjects = subjects
        self.start_frames = start_frames
        self.end_frames = end_frames
        self.start_t_ns = start_t_ns  # int64
        self.end_t_ns = end_t_ns  # int64
        # (D,) each assigned bout's cluster in the **finest** rung, and the DTW cost of that
        # placement. The fine cluster is enough for every granularity because the rungs nest; the
        # distance is kept because "far from every medoid" is worth being able to see later, even
        # though it is not a calibrated score and is not published as one.
        self.assigned_fine_cluster = (
            np.zeros(0, dtype=np.int64)
            if assigned_fine_cluster is None
            else np.asarray(assigned_fine_cluster, dtype=np.int64)
        )
        self.assigned_distance = (
            np.zeros(0, dtype=np.float64)
            if assigned_distance is None
            else np.asarray(assigned_distance, dtype=np.float64)
        )
        self._levels: list[np.ndarray] | None = None
        self._tree = tree
        self._labels: dict[int, np.ndarray] = {}
        self._all_labels: dict[int, np.ndarray] = {}

    @property
    def n_clustered(self) -> int:
        """Bouts the matrix compared — what the vocabulary was learned from."""
        return len(self.dist)

    @property
    def n_assigned(self) -> int:
        """Bouts placed against the vocabulary afterwards, because a cap kept them out of it."""
        return len(self.assigned_fine_cluster)

    @property
    def n_bouts(self) -> int:
        """Every bout segmentation found. All of them are labeled and all of them are instances."""
        return self.n_clustered + self.n_assigned

    @property
    def levels(self) -> list[np.ndarray]:
        """FINCH's own nested partitions of the matrix, finest first. Built once, memoized."""
        if self._levels is None:
            self._levels = segmentation.finch_hierarchy(
                np.asarray(self.dist, dtype=np.float64)
            )
        return self._levels

    @property
    def tree(self) -> ClusterTree:
        """The merge order behind ``levels``, rebuilt for a shard written before it was stored.

        The dense builder, because a bout has no vector to average — the K x K matrix the fit
        already carries is what the cluster-level linkage comes from.
        """
        if self._tree is None:
            self._tree = segmentation.cluster_tree_from_distances(
                np.asarray(self.dist, dtype=np.float64), self.levels
            )
        return self._tree

    @property
    def counts(self) -> list[int]:
        return [int(level.max()) + 1 for level in self.levels]

    @property
    def granularity(self) -> GranularitySpec:
        """This fit's own knob: a cluster count, with its hierarchy's partitions as the rungs.

        On the fit rather than on the method, because the range and the rungs are properties of
        *this* hierarchy and are not known until it exists.
        """
        counts = self.counts
        return GranularitySpec(
            kind="cluster_count",
            minimum=min(counts) if counts else 0,
            maximum=max(counts) if counts else 0,
            rungs=tuple(counts),
        )

    def all_labels_at(self, granularity: int) -> np.ndarray:
        """The label of **every** bout at ``granularity`` — clustered ones, then assigned ones.

        The assigned bouts were placed once, against the finest rung's medoids, and are never
        re-compared. That is exact at every setting rather than an approximation of one, and the
        reason is nesting: ``_refine_to`` starts from a level of the hierarchy and only ever
        merges, and the levels themselves nest, so **every** granularity's partition is a union of
        finest-rung clusters. A bout that belongs to fine cluster *c* therefore belongs to
        whichever cluster *c* is part of at any coarser setting, and reading the label off any
        clustered member of *c* gives it.

        The same nesting is what lets ``member_leaves`` give an assigned bout its leaf, so labels on
        any node reach it without re-running the placement, which would be minutes on a real study.
        """
        want = int(granularity)
        if want not in self._all_labels:
            labels = self.labels_at(want)
            if not self.n_assigned:
                self._all_labels[want] = labels
            else:
                # fine cluster -> its cluster at `want`, read off any clustered member. Every
                # fine cluster has one: it is a partition of the clustered bouts.
                fine_to_want = segmentation.parent_map(self.levels[0], labels)
                self._all_labels[want] = np.concatenate(
                    [labels, fine_to_want[self.assigned_fine_cluster]]
                )
        return self._all_labels[want]

    def labels_at(self, granularity: int) -> np.ndarray:
        """The partition of the **clustered** bouts at ``granularity``, memoized.

        Clustered only, because this is what the medoids and the member ranking read — both are
        about the comparison that actually happened. What a user reads goes
        through ``all_labels_at``.

        A rung is returned directly; between rungs it is a cut of the merge tree. The matrix is
        already in hand here either way, so what the tree buys this method is not the memory it
        buys the pose one — it is that **every** setting now nests, not only the rungs.
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

    def __len__(self) -> int:
        """**Clustered** bouts — what the matrix holds. Use ``n_bouts`` for everything a user
        sees, which includes the assigned ones."""
        return self.n_clustered


def _segmentation_params(model: TemporalDtwFinchConfig) -> dict:
    """The seconds→nanoseconds shape `bout_segmentation.segment_recording` consumes.

    Built from this method's own model rather than borrowed from ``resolve_fit_config``, which also
    does experiment-aware validation the module has already done by this point.
    """
    return {
        "window_ns": int(round(model.window_s * SEC_TO_NS)),
        "min_len_ns": int(round(model.min_len_s * SEC_TO_NS)),
        "band_ns": int(round(model.band_s * SEC_TO_NS)),
        "max_gap_ns": (
            None if model.max_gap_s is None else int(round(model.max_gap_s * SEC_TO_NS))
        ),
        "pool": model.pool,
        "feat_stride": model.feat_stride,
        "detector": model.detector,
        "detector_params": bout_segmentation.detector_params(model),
    }


@register_temporal_cluster_method("dtw_finch")
class DtwFinchTemporalMethod(TemporalClusterMethod):
    """Segment into bouts, compare them with banded DTW, group the matrix with FINCH."""

    method_id = "dtw_finch"
    config_model = TemporalDtwFinchConfig
    # FINCH labels every bout it was given. That is a statement about the *rule*, not about
    # coverage of the recording: bouts do not tile a timeline, so a run can be fully assigned and
    # still cover well under 100% of observed time.
    has_noise = False
    supports_apply = False
    shard_suffix = ".npz"
    # A bout is an extent segmentation found, not a run assembled from labels, so two adjacent
    # bouts the user has given one name stay two instances. The frame-labelling methods have no
    # such claim to contradict — their instances are runs of like-labelled rows, and a name simply
    # makes the run longer.
    instances_are_runs = False

    # The kind of knob, renderable before any fit exists. The rungs belong to a particular
    # hierarchy, so they come from `granularity_of(fit)`.
    granularity = GranularitySpec(kind="cluster_count", minimum=2, maximum=0, rungs=())

    def granularity_of(self, fit: _Fit) -> GranularitySpec:
        """The knob for a *particular* fit — its range and its nested rungs."""
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
        """Cut every trajectory into bouts, pool them, and compare every pair.

        The phases are wildly unequal in cost — the comparison dwarfs the segmentation — so the
        fraction is pinned inside the long step and the phase name carries the movement, which is
        what §22 asks a global-fraction runner to do. ``pairwise_dtw`` reports inside its own loop,
        so the bar does move through the expensive part.
        """
        model = TemporalDtwFinchConfig.model_validate(params)
        report = progress_cb or (lambda fraction, phase: None)
        report(0.0, "loading")

        if not trajectories:
            raise ValueError(
                "temporal clustering needs at least one trajectory to fit on"
            )

        seg_params = _segmentation_params(model)
        report(0.05, "segmenting")
        recording_data: list[dict] = []
        bands: list[int] = []
        subject_of: list[int] = []
        for recording_id, subject, features, t_ns in trajectories:
            features = np.asarray(features)
            if len(features) == 0:
                continue
            frame_idx = np.arange(len(features), dtype=np.int64)
            bouts, band, _resolved, _trace = bout_segmentation.segment_recording(
                features, frame_idx, np.asarray(t_ns, dtype=np.int64), seg_params
            )
            if not bouts:
                continue
            recording_data.append(
                {
                    "recording_id": recording_id,
                    "emb": features,
                    "frame_idx": frame_idx,
                    "t_ns": np.asarray(t_ns, dtype=np.int64),
                    "bouts": bouts,
                }
            )
            bands.append(band)
            subject_of.extend([int(subject)] * len(bouts))

        if not recording_data:
            raise ValueError(
                "no bouts were found in the selected recordings — the trajectories may be too "
                "short, or the minimum bout length too long for this embed cadence"
            )

        pooled_emb, pooled_bouts, origin = bout_segmentation.pool_recordings(
            recording_data
        )
        subjects_all = np.asarray(subject_of, dtype=np.int64)
        n_before = len(pooled_bouts)

        # Subsample here rather than inside `pairwise_dtw`, so `origin` stays row-aligned with the
        # matrix — the rule the removed `cluster_fit.fit_vocabulary` followed, for the same reason.
        # The cap bounds the **comparison**, not the output. Whatever it leaves out is set aside
        # here and placed against the vocabulary below, so a bout that was segmented is a bout
        # that gets a label — the difference between sampling the fit and truncating the result.
        keep = segmentation.subsample_indices(len(pooled_bouts), model.max_segments)
        if keep is None:
            held_bouts: list[tuple[int, int]] = []
            held_origin: list[tuple] = []
            held_subjects = np.zeros(0, dtype=np.int64)
        else:
            kept_set = set(int(i) for i in keep)
            held = [i for i in range(len(pooled_bouts)) if i not in kept_set]
            held_bouts = [pooled_bouts[i] for i in held]
            held_origin = [origin[i] for i in held]
            held_subjects = subjects_all[np.asarray(held, dtype=np.int64)]
            pooled_bouts = [pooled_bouts[i] for i in keep]
            origin = [origin[i] for i in keep]
            subjects_all = subjects_all[np.asarray(keep, dtype=np.int64)]

        n_bouts = len(pooled_bouts)
        report(0.15, f"comparing {n_bouts:,} bouts")
        band = max(bands) if bands else 0

        # `pairwise_dtw` reports `(pairs_done, pairs_total)`, and pairs are not bouts: 400 bouts
        # is 79,800 pairs. The fraction wants the pair counts, the *message* wants the bout count,
        # and printing the pair total as a bout count told a user their 400-bout fit was comparing
        # eighty thousand of them.
        def on_pair(pairs_done: int, pairs_total: int) -> None:
            report(
                0.15 + 0.7 * (pairs_done / max(pairs_total, 1)),
                f"comparing {n_bouts:,} bouts",
            )

        _kept, dist = segmentation.pairwise_dtw(
            pooled_emb, pooled_bouts, band, None, on_pair
        )

        report(0.85, "clustering")
        # Provenance runs clustered-then-assigned, matching how `_Fit` indexes everything.
        all_origin = list(origin) + list(held_origin)
        recording_ids: list[str] = []
        recording_index = []
        for row in all_origin:
            if row[0] not in recording_ids:
                recording_ids.append(row[0])
            recording_index.append(recording_ids.index(row[0]))

        fine, distances = self._place_held(
            pooled_emb,
            np.asarray(dist, dtype=np.float64),
            pooled_bouts,
            held_bouts,
            band,
            report,
        )

        fit = _Fit(
            dist=np.asarray(dist, dtype=np.float32),
            recording_ids=tuple(recording_ids),
            recording_index=np.asarray(recording_index, dtype=np.int64),
            subjects=np.concatenate([subjects_all, held_subjects]),
            start_frames=np.asarray([row[1] for row in all_origin], dtype=np.int64),
            end_frames=np.asarray([row[2] for row in all_origin], dtype=np.int64),
            start_t_ns=np.asarray([row[3] for row in all_origin], dtype=np.int64),
            end_t_ns=np.asarray([row[4] for row in all_origin], dtype=np.int64),
            assigned_fine_cluster=fine,
            assigned_distance=distances,
        )
        if n_before != fit.n_bouts:
            # The invariant the cap now rests on. Stated rather than assumed because losing a
            # bout here is silent by construction: the vocabulary still looks right, the run
            # still completes, and only the instance count is quietly short.
            raise RuntimeError(
                f"segmentation found {n_before} bouts but the fit holds {fit.n_bouts}; the cap "
                "dropped some instead of setting them aside"
            )
        report(1.0, "clustering")
        return fit

    @staticmethod
    def _place_held(
        pooled_emb: np.ndarray,
        dist: np.ndarray,
        kept_bouts: list[tuple[int, int]],
        held_bouts: list[tuple[int, int]],
        band: int,
        report,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place every bout the cap kept out of the comparison, against the **finest** rung.

        Nearest medoid, using ``bout_segmentation.bout_medoid_distance`` so an assigned bout is measured by
        exactly the metric the vocabulary was built with rather than a lookalike.

        **The finest rung, and once.** A bout placed there resolves at every granularity through
        the nesting (see ``_Fit.all_labels_at``), so this is paid once at fit time instead of on
        every turn of the granularity knob — which is what keeps that knob instant. It is also the
        cheap end of the trade: with ``k`` clustered bouts the finest rung holds roughly ``k/6``
        clusters, so placing ``d`` bouts costs ``d·k/6`` comparisons against the ``k²/2`` the
        matrix already cost, and against the ``(k+d)²/2`` of not capping at all.
        """
        if not held_bouts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

        levels = segmentation.finch_hierarchy(dist)
        finest = levels[0] if levels else np.zeros(len(dist), dtype=np.int64)
        medoids = [
            segmentation.medoid_order(
                dist, [int(i) for i in np.flatnonzero(finest == cluster)]
            )[0][0]
            for cluster in range(int(finest.max()) + 1)
        ]
        medoid_emb = [
            pooled_emb[kept_bouts[m][0] : kept_bouts[m][1] + 1] for m in medoids
        ]

        report(0.87, f"assigning {len(held_bouts):,} bouts")
        fine = np.zeros(len(held_bouts), dtype=np.int64)
        chosen = np.zeros(len(held_bouts), dtype=np.float64)
        for i, (r0, r1) in enumerate(held_bouts):
            bout = pooled_emb[r0 : r1 + 1]
            costs = [
                bout_segmentation.bout_medoid_distance(bout, m, band)
                for m in medoid_emb
            ]
            best = int(np.argmin(costs))
            fine[i] = best
            chosen[i] = costs[best]
            if i % 64 == 0:
                report(
                    0.87 + 0.06 * (i / max(len(held_bouts), 1)),
                    f"assigning {len(held_bouts):,} bouts",
                )
        return fine, chosen

    def instances_at(
        self, fit: _Fit, granularity: int, params: dict
    ) -> list[ClusterInstance]:
        """One instance per bout, carrying the cluster it lands in at this granularity.

        No run-length encoding and no minimum-run floor: a bout is already an extent, and one too
        short to be a behavior was dropped by ``min_len_s`` before the comparison existed. The
        ``start_row`` / ``end_row`` here index the **fit's bouts**, not trajectory rows — the
        module resolves them through the fit's own frame and time arrays.

        **Every** bout, clustered and assigned alike. A cap bounds what the vocabulary is learned
        from, not what it labels: for a method whose job is finding every occurrence of a
        behavior, an instance missing because its bout lost a coin flip is missing data.
        """
        labels = fit.all_labels_at(granularity)
        return [
            ClusterInstance(int(i), int(i), int(label))
            for i, label in enumerate(labels)
        ]

    def member_leaves(self, fit: _Fit) -> np.ndarray:
        """Every bout's finest cluster: clustered bouts, then assigned ones.

        The same order ``instances_at`` indexes bouts in. An assigned bout was placed against the
        finest rung's medoids, so its finest cluster is its leaf exactly as a clustered bout's is.
        """
        clustered = np.asarray(fit.levels[0], dtype=np.int64)
        if not fit.n_assigned:
            return clustered
        return np.concatenate([clustered, fit.assigned_fine_cluster])

    def member_places(self, fit: _Fit) -> MemberPlaces:
        """Every bout's recording, subject and measured time: its span plus one frame step.

        The step is the bout's own, measured as its span over its frame count, so a bout's time
        comes from its timestamps; a one-frame bout takes its recording's median step, and a
        recording of one-frame bouts the run's. Clustered bouts first, then assigned, as
        ``member_leaves`` orders them.
        """
        start = np.asarray(fit.start_t_ns, dtype=np.int64)
        end = np.asarray(fit.end_t_ns, dtype=np.int64)
        frames = np.asarray(fit.end_frames, dtype=np.int64) - np.asarray(
            fit.start_frames, dtype=np.int64
        )
        span = np.maximum(end - start, 0)
        own = np.where(frames > 0, span // np.maximum(frames, 1), -1)
        recording = np.asarray(fit.recording_index, dtype=np.int64)
        measured = own[own > 0]
        run_step = int(np.median(measured)) if len(measured) else 0
        step = own.copy()
        for index in np.unique(recording):
            mask = recording == index
            here = own[mask & (own > 0)]
            fallback = int(np.median(here)) if len(here) else run_step
            step[mask & (own <= 0)] = fallback
        return MemberPlaces(
            recording_ids=tuple(fit.recording_ids),
            recording_index=recording,
            subject=np.asarray(fit.subjects, dtype=np.int64),
            start_ns=start,
            end_ns=end,
            time_ns=span + np.maximum(step, 0),
        )

    def rank_members(
        self, fit: _Fit, leaves: np.ndarray
    ) -> list[tuple[ClusterInstance, float]]:
        """The node's clustered bouts, nearest its medoid first (``medoid_order`` over the node).

        Medoid rather than nearest-centroid because this is a distance-based rule and there is no
        centroid to be near: what a node *is* is its most central member (§22). ``medoid_order``
        ranks every member and reuses the bout DTW matrix.

        Clustered bouts only: an assigned bout was never compared, so it has no distance to rank
        by. It still counts in every figure the cache sums.
        """
        members = np.flatnonzero(np.isin(fit.levels[0], np.asarray(leaves)))
        order = segmentation.medoid_order(
            np.asarray(fit.dist, dtype=np.float64), [int(m) for m in members]
        )
        return [
            (ClusterInstance(int(row), int(row), -1), float(distance))
            for row, distance in order
        ]

    def clip_scoring(
        self, fit: _Fit, params: dict, trajectories: Sequence | None = None
    ) -> ClipScoring:
        """Medoid order depends on the node and does not add up over leaves, so a node's clips
        are ranked on demand by ``rank_members``. A bout is its own clip: two bouts never join.
        """
        return ClipScoring(kind="medoid", min_members=1, join=False)

    def node_separations(self, fit: _Fit) -> np.ndarray:
        """Mean DTW distance between the children over their mean within-child distance.

        Batched through leaf-level sums of the bout matrix, so every node costs one dot product:
        ``onehot.T @ dist @ onehot`` gives the summed distance between any two leaves, and a
        node's sums are sums over its leaves.
        """
        tree = fit.tree
        leaves, nodes = tree.n_leaves, tree.n_nodes
        fine = np.asarray(fit.levels[0], dtype=np.int64)
        onehot = np.zeros((len(fine), leaves), dtype=np.float64)
        onehot[np.arange(len(fine)), fine] = 1.0
        dist = np.asarray(fit.dist, dtype=np.float64)
        leaf_sums = onehot.T @ dist @ onehot
        under = np.zeros((nodes, leaves), dtype=np.float64)
        under[np.arange(leaves), np.arange(leaves)] = 1.0
        for merge, (a, b) in enumerate(np.asarray(tree.children, dtype=np.int64)):
            under[leaves + merge] = under[a] + under[b]
        counts = under @ onehot.sum(axis=0)
        through = under @ leaf_sums
        out = np.full(nodes, np.nan, dtype=np.float64)

        def within(x: int) -> float:
            pairs = counts[x] * (counts[x] - 1)
            return float(through[x] @ under[x]) / pairs if pairs > 0 else np.nan

        for merge, (a, b) in enumerate(np.asarray(tree.children, dtype=np.int64)):
            pairs = counts[a] * counts[b]
            if pairs <= 0:
                continue
            between = float(through[a] @ under[b]) / pairs
            spreads = [v for v in (within(int(a)), within(int(b))) if np.isfinite(v)]
            spread = float(np.mean(spreads)) if spreads else np.nan
            out[leaves + merge] = (
                between / spread if np.isfinite(spread) and spread > 0 else np.nan
            )
        return out

    def instances_for(
        self,
        fit: _Fit,
        leaf_groups: np.ndarray,
        params: dict,
        member_groups: np.ndarray | None = None,
    ) -> list[ClusterInstance]:
        """One instance per bout, carrying its leaf's group, or ``member_groups``' override.

        No run-length encoding, as in ``instances_at``: a bout is already an extent, and this
        method's ``instances_are_runs`` is false, so two bouts are never fused. A bout is a whole
        clip, so an override takes the bout out whole; there is no leftover piece.
        """
        groups = member_groups_of(leaf_groups, self.member_leaves(fit), member_groups)
        return [
            ClusterInstance(int(i), int(i), int(group))
            for i, group in enumerate(groups)
        ]

    def resolve(self, fit: _Fit, instance) -> dict:
        """Where a bout sits. One row of the fit's provenance, since an instance *is* a bout."""
        start, end = int(instance.start_row), int(instance.end_row)
        if fit.recording_index[start] != fit.recording_index[end]:
            return {}
        return {
            "recording_id": fit.recording_ids[int(fit.recording_index[start])],
            "subject": int(fit.subjects[start]),
            "start_frame": int(fit.start_frames[start]),
            "end_frame": int(fit.end_frames[end]),
            "t_ns": int(fit.start_t_ns[start]),
            "end_t_ns": int(fit.end_t_ns[end]),
        }

    def fit_summary(self, fit: _Fit) -> dict:
        """How many bouts formed the vocabulary, and how many were placed against it.

        An assigned bout is a weaker claim than a clustered one — it was matched to a codebook it
        had no part in building — and a reader who cannot tell them apart cannot weigh that.
        """
        return {
            "n_fit_bouts": int(fit.n_clustered),
            "n_assigned_bouts": int(fit.n_assigned),
        }

    def subsample_note(self, fit: _Fit) -> str | None:
        """What the cap changed, or ``None`` when every bout was compared.

        A statement about the *fit*, not a warning about missing output: the bouts held out of the
        comparison are still labeled and still instances. What they did not do is help decide what
        the clusters are, which is a weaker claim than a clustered bout's and is why it is said.
        """
        if not fit.n_assigned:
            return None
        return (
            f"vocabulary learned from {fit.n_clustered:,} of {fit.n_bouts:,} bouts, evenly "
            f"sampled. All {fit.n_bouts:,} are labeled: the other {fit.n_assigned:,} were "
            "matched to the nearest cluster afterwards rather than helping to form one."
        )

    def serialize(self, fit: _Fit, path: Path) -> None:
        """Write the fit as one npz.

        The matrix is the thing worth keeping and it is the whole cost: 400 MB at K=10k in
        float32, which is why it is stored at that width and why ``max_segments`` exists.
        """
        np.savez_compressed(
            path,
            dist=np.asarray(fit.dist, dtype=np.float32),
            recording_ids=np.asarray(fit.recording_ids, dtype=object),
            recording_index=fit.recording_index,
            subjects=fit.subjects,
            start_frames=fit.start_frames,
            end_frames=fit.end_frames,
            start_t_ns=fit.start_t_ns,
            end_t_ns=fit.end_t_ns,
            assigned_fine_cluster=fit.assigned_fine_cluster,
            assigned_distance=fit.assigned_distance,
            **segmentation.tree_arrays(fit.tree),
        )

    def load(self, path: Path) -> _Fit:
        with np.load(path, allow_pickle=True) as data:
            return _Fit(
                dist=data["dist"],
                recording_ids=tuple(str(r) for r in data["recording_ids"]),
                recording_index=data["recording_index"],
                subjects=data["subjects"],
                start_frames=data["start_frames"],
                end_frames=data["end_frames"],
                start_t_ns=data["start_t_ns"],
                end_t_ns=data["end_t_ns"],
                assigned_fine_cluster=data["assigned_fine_cluster"],
                assigned_distance=data["assigned_distance"],
                # A fit from an older build also carries `projection_points`,
                # `projection_layout`, `coords` and `seed` (the retired 2-D layout and its
                # seed); they are not read.
                tree=segmentation.tree_from_arrays(data),
            )


def rows_per_second(t_ns: np.ndarray) -> float:
    """Rows per second from a trajectory's own timestamps — the cadence every seconds knob
    resolves against (Tenet 8)."""
    period = clip_core.nominal_period_ns(np.asarray(t_ns, dtype=np.int64))
    return (SEC_TO_NS / period) if period > 0 else 0.0
