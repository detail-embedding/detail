"""``NodeView`` — every surface reads a clustering run's nodes through this (§22).

The read half of the node cache. Opening one reads ``node_cache.zarr`` into memory once; after
that, every question about a node is arithmetic over those arrays: a sum over the node's leaves
for anything additive, and a pass over the node's leaf runs for its clips. **Nothing here decodes
video or reads a track**, which is what makes selecting a node instant on the GUI thread and why
the CLI can print the same figures without a fit in hand.

**Every node lists every clip** (``clips``): one per action, each of its leaf runs on its own
(never joined with its neighbours), kept when it reaches the method's minimum length, with its
typicality from the cache's per-run figures. There is no size floor.

A **concentration** is never reported without its floor. A node always has a most-common
recording or group, so "71% from box 3" means nothing until it is set beside what a random draw
of the same size from the run would give. ``distribution`` computes that floor on the spot: the
node's member count redrawn from the run's own distribution, 200 times, seeded by the node id so
a card says the same thing every time it is opened.
"""

from dataclasses import dataclass

import numpy as np

from detail.analysis import node_cache as tables_lib
from detail.api import cluster_filter
from detail.storage import node_cache as store

#: Redraws behind a concentration floor, and the percentile taken.
FLOOR_DRAWS = 200
FLOOR_PERCENTILE = 95.0

#: The orders a node's clips come in (``ClipTable.order``).
CLIP_SORTS = ("typical", "atypical", "random")


@dataclass(frozen=True)
class NodeSummary:
    """What a node's header says."""

    node: int
    parent: int  # -1 at the root
    children: tuple[int, ...]  # () on a leaf, else exactly two
    time_ns: int
    members: int
    share: float  # of the whole run's member time
    separation: float  # NaN on a leaf


@dataclass(frozen=True)
class Distribution:
    """A node's share per value of one dimension, beside the run's, with its floor."""

    values: tuple[str, ...]
    node: np.ndarray  # (V,) share of the node's time per value
    run: np.ndarray  # (V,) share of the run's time per value
    top: str  # the value the node is most over-represented in
    ratio: float  # node share / run share at ``top``
    floor: float  # FLOOR_PERCENTILE of the same ratio under random draws of the node's size
    flagged: bool  # ratio beats its floor


@dataclass(frozen=True)
class ClipTable:
    """Every clip of one node, columnar, in time order (recording, subject, start).

    ``end_ns`` is the clip's last member's own end, inclusive, the convention every instance
    ends on (§19): a clip of one frame starts and ends on that frame. (A cache from before
    ``run_last_ns`` existed gives the end of the covered time instead.)
    ``typicality`` is the method's own distance to the node's centre: **smaller is more
    typical**, the convention of ``rank_members``, and ``NaN`` where it is not known (a method
    ranked on demand before it is ranked, or a cache built before typicality was kept).
    ``leaf`` is the leaf of the merge tree the clip's run sits in: a clip is one leaf run, so
    ``(recording, subject, start_ns, end_ns)`` names it and ``leaf`` says which cluster it came
    from, whatever the labels say now.
    """

    recording_ids: tuple[str, ...]
    recording: np.ndarray  # (C,) int32 into recording_ids
    subject: np.ndarray  # (C,) int64
    start_ns: np.ndarray  # (C,) int64
    end_ns: np.ndarray  # (C,) int64
    cx: np.ndarray  # (C,) float32, mean frame-normalized position, NaN where unknown
    cy: np.ndarray  # (C,) float32
    members: np.ndarray  # (C,) int64, the members (frames or bouts) the clip holds
    typicality: np.ndarray  # (C,) float64
    leaf: np.ndarray  # (C,) int64, the leaf of the merge tree each clip's run is in

    def __len__(self) -> int:
        return len(self.start_ns)

    def take(self, index) -> "ClipTable":
        """These clips at ``index`` (an index array or a boolean mask), in that order."""
        index = np.asarray(index)
        return ClipTable(
            recording_ids=self.recording_ids,
            recording=self.recording[index],
            subject=self.subject[index],
            start_ns=self.start_ns[index],
            end_ns=self.end_ns[index],
            cx=self.cx[index],
            cy=self.cy[index],
            members=self.members[index],
            typicality=self.typicality[index],
            leaf=self.leaf[index],
        )

    def with_typicality(self, typicality: np.ndarray) -> "ClipTable":
        return ClipTable(
            recording_ids=self.recording_ids,
            recording=self.recording,
            subject=self.subject,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            cx=self.cx,
            cy=self.cy,
            members=self.members,
            typicality=np.asarray(typicality, dtype=np.float64),
            leaf=self.leaf,
        )

    def order(self, sort: str = "typical", seed: int = 0) -> np.ndarray:
        """Indices in ``sort`` order (``CLIP_SORTS``), ties and unknowns in time order.

        ``typical`` puts the smallest typicality first and ``atypical`` the largest, unknown
        (``NaN``) last either way; ``random`` is a shuffle seeded by ``seed`` (the node id, so
        a node's random order is the same on every click).

        Raises:
            ValueError: an unknown sort.
        """
        if sort not in CLIP_SORTS:
            raise ValueError(f"unknown sort {sort!r}: one of {', '.join(CLIP_SORTS)}")
        if sort == "random":
            return np.random.default_rng(int(seed)).permutation(len(self))
        score = np.asarray(self.typicality, dtype=np.float64)
        key = score if sort == "typical" else -score
        return np.argsort(key, kind="stable")

    def rows(self, index=None) -> list[dict]:
        """The clips as ``{recording_id, subject, start_t_ns, end_t_ns, frame_ns, jpeg, cx, cy,
        typicality, leaf}`` dicts, the shape the clip window and the montage take; ``jpeg`` is
        ``b""`` (nothing is decoded here). ``clip_ref(row)`` (``api/clip_edits.py``) turns one
        into the reference the take-out and review calls take."""
        index = range(len(self)) if index is None else index
        out = []
        for i in index:
            start, end = int(self.start_ns[i]), int(self.end_ns[i])
            out.append(
                {
                    "recording_id": self.recording_ids[int(self.recording[i])],
                    "subject": int(self.subject[i]),
                    "start_t_ns": start,
                    "end_t_ns": end,
                    "frame_ns": (start + end) // 2,
                    "jpeg": b"",
                    "cx": float(self.cx[i]),
                    "cy": float(self.cy[i]),
                    "typicality": float(self.typicality[i]),
                    "leaf": int(self.leaf[i]),
                }
            )
        return out


def _empty_clips(recording_ids) -> ClipTable:
    i64, f32 = np.zeros(0, np.int64), np.zeros(0, np.float32)
    return ClipTable(
        recording_ids=tuple(recording_ids),
        recording=np.zeros(0, np.int32),
        subject=i64,
        start_ns=i64,
        end_ns=i64,
        cx=f32,
        cy=f32,
        members=i64,
        typicality=np.zeros(0, np.float64),
        leaf=i64,
    )


class NodeView:
    """One run's node cache, in memory, with the tree it indexes."""

    def __init__(
        self, cache: store.NodeCache, tree, recording_names: dict | None = None
    ):
        self._cache = cache
        self._tree = tree
        self._leaves = tables_lib.node_leaf_index(tree)
        self._parent = tree.parent
        t = cache.leaves
        self._node_time = self._node_sums(t.leaf_time_ns.astype(np.int64))
        self._node_members = self._node_sums(t.leaf_members.astype(np.int64))
        self._run_time = int(t.leaf_time_ns.sum())
        self._run_rec_time = t.leaf_rec_time_ns.sum(axis=0)
        self.recording_names = dict(recording_names or {})
        self._sorted_runs = None
        #: ``[(run indices, from name, to name | None)]``: the run's clip edits, resolved to the
        #: cache's runs (``set_clip_edits``).
        self._clip_edits: list[tuple[np.ndarray, str, str | None]] = []

    #: Set by the run handle that opened this view: why its clips no longer match the fit, or
    #: ``None`` when they do (``stale_reason``).
    stale: str | None = None

    def _node_sums(self, per_leaf: np.ndarray) -> np.ndarray:
        """``(n_nodes, ...)`` a per-leaf figure summed up the tree, bottom-up in one pass."""
        per_leaf = np.asarray(per_leaf)
        leaves = self._tree.n_leaves
        out = np.zeros((self._tree.n_nodes, *per_leaf.shape[1:]), dtype=per_leaf.dtype)
        out[:leaves] = per_leaf[:leaves]
        for merge, (a, b) in enumerate(np.asarray(self._tree.children, dtype=np.int64)):
            out[leaves + merge] = out[a] + out[b]
        return out

    @property
    def built_from(self) -> dict:
        """What the cache was built from: the fit shard's name, size and time."""
        return dict(self._cache.built_from)

    @property
    def clip_rule(self) -> dict:
        """``{"kind", "min_members", "join"}``: how this run's clips are scored and kept (``join``:
        whether a leaf's touching members made one run when the cache was built)."""
        rule = {"kind": "none", "min_members": 1, "join": True}
        rule.update(self._cache.clips or {})
        return rule

    @property
    def ranks_on_demand(self) -> bool:
        """Whether typicality comes from the fit when a node is first asked (``medoid``)."""
        return self.clip_rule["kind"] == "medoid"

    def stale_reason(self, current: dict) -> str | None:
        """Why this cache does not describe the fit ``current`` digests, or ``None`` if it does.

        Compares the fit shard's **name and size** only. Its modification time is recorded but
        not compared: copying or syncing an experiment rewrites every file time, and flagging
        every run's clips after a copy would teach users to ignore the warning. A cache built
        before clips were listed from leaf runs (no ``run_rows``) is reported too: its nodes
        still open, but their clips have no minimum length or typicality until it is rebuilt.
        So is a bout method's cache built under rule 1 (``storage/node_cache.RULES``), whose runs
        each ended a bout's span too late. A frame method's rule-1 cache is not: a frame starts
        where it ends, so the two rules give it the same runs, and flagging every such run would
        teach users to ignore the banner. (Every bout method's clips are unjoined, ``join``
        false, and every frame method's joined.)
        """
        built = self._cache.built_from
        for key, what in (("fit", "fit file"), ("fit_size", "fit size")):
            if key in built and key in current and built[key] != current[key]:
                return (
                    f"the {what} is {current[key]!r} now but the clips were built from "
                    f"{built[key]!r}"
                )
        if self._cache.leaves.run_rows is None:
            return (
                "the clips were built before every clip of every cluster was listed, so they "
                "have no typicality order"
            )
        if int(built.get(store.RULES_KEY, 1)) < 2 and not self.clip_rule["join"]:
            return (
                "the clusters' times were built when a bout's time was counted from its end, so "
                "each bout ran on past its end"
            )
        return None

    @classmethod
    def open(cls, run_dir, tree, recording_names: dict | None = None) -> "NodeView":
        """Read ``run_dir``'s node cache. Raises ``FileNotFoundError`` if it has none."""
        return cls(store.read_node_cache(run_dir), tree, recording_names)

    # --- the tree ------------------------------------------------------------------------

    @property
    def tree(self):
        return self._tree

    @property
    def root(self) -> int:
        return int(self._tree.n_nodes - 1)

    @property
    def recording_ids(self) -> tuple[str, ...]:
        return self._cache.leaves.recording_ids

    def children(self, node: int) -> tuple[int, ...]:
        node = int(node)
        if node < self._tree.n_leaves:
            return ()
        a, b = self._tree.children[node - self._tree.n_leaves]
        return (int(a), int(b))

    def leaves_under(self, node: int) -> np.ndarray:
        return self._leaves[int(node)]

    def ancestors(self, node: int) -> list[int]:
        """The path from the root down to ``node``'s parent, root first."""
        out = []
        current = int(self._parent[int(node)])
        while current >= 0:
            out.append(current)
            current = int(self._parent[current])
        return list(reversed(out))

    def node_times(self) -> np.ndarray:
        """``(n_nodes,)`` member time of every node, the icicle's widths."""
        return self._node_time

    # --- one node ------------------------------------------------------------------------

    def summary(self, node: int) -> NodeSummary:
        node = int(node)
        return NodeSummary(
            node=node,
            parent=int(self._parent[node]),
            children=self.children(node),
            time_ns=int(self._node_time[node]),
            members=int(self._node_members[node]),
            share=self._node_time[node] / self._run_time if self._run_time else 0.0,
            separation=float(self._cache.leaves.separation[node]),
        )

    def separation_median(self) -> float:
        """The run's median split separation: what a node's own figure is read against."""
        values = self._cache.leaves.separation[self._tree.n_leaves :]
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else float("nan")

    def occupancy(self, node: int) -> np.ndarray:
        """``(R,)`` the node's member time per recording, in ``recording_ids`` order."""
        return self._cache.leaves.leaf_rec_time_ns[self._leaves[int(node)]].sum(axis=0)

    def rate(self, node: int) -> np.ndarray:
        """``(R,)`` the node's share of each recording's time over its share of the run's.

        1.0 means the recording holds this node exactly as often as the run as a whole does;
        the which-videos strip draws this with 1.0 as its reference line.
        """
        here = self.occupancy(node).astype(np.float64)
        total = here.sum()
        if not total or not self._run_time:
            return np.zeros(len(here))
        run = self._run_rec_time.astype(np.float64) / self._run_time
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(run > 0, (here / total) / run, 0.0)

    def distribution(self, node: int, groups: dict[str, str]) -> Distribution:
        """The node's time over the values of one recording-level dimension, with its floor.

        ``groups`` maps recording id to its value in that dimension (a metadata column, or the
        recording itself). Recordings with no value are left out of both sides.
        """
        values = tuple(sorted({v for v in groups.values() if v != ""}))
        index = {v: i for i, v in enumerate(values)}
        mapping = np.array(
            [index.get(groups.get(r, ""), -1) for r in self.recording_ids],
            dtype=np.int64,
        )
        keep = mapping >= 0
        here = np.bincount(
            mapping[keep], weights=self.occupancy(node)[keep], minlength=len(values)
        )
        run = np.bincount(
            mapping[keep],
            weights=self._run_rec_time[keep].astype(np.float64),
            minlength=len(values),
        )
        node_share = here / here.sum() if here.sum() else np.zeros(len(values))
        run_share = run / run.sum() if run.sum() else np.zeros(len(values))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = np.where(run_share > 0, node_share / run_share, 0.0)
        top = int(np.argmax(ratios)) if len(values) else 0
        floor = self._floor(node, run_share)
        ratio = float(ratios[top]) if len(values) else 0.0
        return Distribution(
            values=values,
            node=node_share,
            run=run_share,
            top=values[top] if values else "",
            ratio=ratio,
            floor=floor,
            flagged=bool(len(values) > 1 and ratio > floor),
        )

    def _floor(self, node: int, run_share: np.ndarray) -> float:
        """The over-representation a random draw of this node's size reaches by chance."""
        members = int(self._node_members[int(node)])
        if members <= 0 or len(run_share) < 2 or not run_share.sum():
            return float("inf")
        rng = np.random.default_rng(int(node))
        draws = rng.multinomial(members, run_share / run_share.sum(), size=FLOOR_DRAWS)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = np.where(run_share > 0, (draws / members) / run_share, 0.0)
        return float(np.percentile(ratios.max(axis=1), FLOOR_PERCENTILE))

    def positions(self, node: int, recordings: list[int] | None = None) -> np.ndarray:
        """``(B, B)`` position counts for the node, over all recordings or the given indices."""
        counts = self._cache.leaves.leaf_rec_pos[self._leaves[int(node)]]
        if recordings is not None:
            counts = counts[:, recordings]
        return counts.sum(axis=(0, 1))

    def has_positions(self) -> bool:
        """Whether this run recorded any positions at all (a run before kinematics has none)."""
        return bool(self._cache.leaves.leaf_rec_pos.any())

    # --- clips ---------------------------------------------------------------------------

    def _runs_sorted(self) -> np.ndarray:
        """The leaf runs' order by (recording, subject, start), computed once per view."""
        if self._sorted_runs is None:
            t = self._cache.leaves
            self._sorted_runs = np.lexsort(
                (t.run_start_ns, t.run_subject.astype(np.int64), t.run_recording)
            )
        return self._sorted_runs

    def clips(self, node: int) -> ClipTable:
        """Every clip of ``node``, in time order, with its typicality where the cache has it.

        **A clip is one action: one leaf run under the node**, never joined with the runs beside
        it. A leaf run is a maximal stretch of members in one leaf of the tree, inside one of the
        method's continuity blocks (a bout method's runs are its bouts), so the root lists every
        action of the run and a node's clips are exactly its children's clips. A bout that
        crosses two sub-clusters plays as two clips; joining them made the root one clip per
        recording. A run shorter than the method's minimum instance length is dropped, as the
        method drops such a run (a rule per run, so it holds at every node alike).
        A clip's typicality is its run's own figure (``ClipScoring``): its mean member distance,
        or ``1 - cos`` between its vector sum and the node's centroid, a sum over the node's
        leaves. One vectorised pass over the runs; nothing is decoded.
        """
        t = self._cache.leaves
        rule = self.clip_rule
        order = self._runs_sorted()
        inside = np.zeros(self._tree.n_leaves, dtype=bool)
        inside[self._leaves[int(node)]] = True
        idx = order[inside[t.run_leaf[order].astype(np.int64)]]
        if t.run_rows is not None:
            rows = t.run_rows[idx].astype(np.int64)
            idx = idx[rows >= int(rule["min_members"])]
        if not len(idx):
            return _empty_clips(self.recording_ids)
        members = (
            np.ones(len(idx), dtype=np.int64)
            if t.run_rows is None
            else t.run_rows[idx].astype(np.int64)
        )
        end = t.run_end_ns if t.run_last_ns is None else t.run_last_ns
        typicality = np.full(len(idx), np.nan)
        if rule["kind"] == "distance" and t.run_dist_sum is not None:
            typicality = t.run_dist_sum[idx].astype(np.float64) / members
        elif rule["kind"] == "centroid" and t.run_vec_sum is not None:
            centre = t.leaf_vec_sum[self._leaves[int(node)]].sum(axis=0)
            sums = t.run_vec_sum[idx].astype(np.float64)
            norms = np.linalg.norm(sums, axis=1) * np.linalg.norm(centre)
            with np.errstate(invalid="ignore", divide="ignore"):
                typicality = np.where(norms > 0, 1.0 - (sums @ centre) / norms, np.nan)
        return ClipTable(
            recording_ids=self.recording_ids,
            recording=t.run_recording[idx].astype(np.int32),
            subject=t.run_subject[idx].astype(np.int64),
            start_ns=t.run_start_ns[idx].astype(np.int64),
            end_ns=end[idx].astype(np.int64),
            cx=self._run_column(t.run_cx, idx),
            cy=self._run_column(t.run_cy, idx),
            members=members,
            typicality=np.asarray(typicality, dtype=np.float64),
            leaf=t.run_leaf[idx].astype(np.int64),
        )

    @staticmethod
    def _run_column(values, idx) -> np.ndarray:
        """A per-run mean position at ``idx``; NaN where the cache kept none."""
        if values is None:
            return np.full(len(idx), np.nan, dtype=np.float32)
        return np.asarray(values, dtype=np.float32)[idx]

    # --- filters -------------------------------------------------------------------------

    def match_counts(
        self,
        selection: cluster_filter.Selection,
        metadata: dict | None = None,
        column_types: dict | None = None,
        spans: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(matched, total)``, each ``(n_nodes,)``: every node's members meeting ``selection``.

        Counted from ``leaf_rec_pos`` (members per leaf, recording and cell of the frame), so
        exact in members for every node: ``pos[leaves][:, recordings][..., by, bx]`` summed over
        ``pos[leaves]``. Members with no known position are in neither. A run without positions
        at all is counted in member time instead (``leaf_rec_time_ns``), which a region cannot
        narrow.

        A **time-in-session** range (``spans``, ``cluster_filter.session_window``) counts only the
        time inside it: each leaf and recording's count is scaled by the share of its leaf runs'
        time that lies inside the range's stretch of that recording, so a run half inside adds
        half. Exact on its own and with recording-level conditions; with a region as well, the
        cache keeps no position per moment, so the two are combined as independent within a leaf
        and recording (each leaf-recording's members in the region, times its share of time in
        range).

        Raises:
            ValueError: a region on a run with no positions, a metadata condition with no
                ``metadata`` (``cluster_filter.recording_mask``), or a session range with no
                ``spans``.
        """
        t = self._cache.leaves
        recordings = cluster_filter.recording_mask(
            selection, self.recording_ids, metadata, column_types
        )
        cells = cluster_filter.cell_mask(selection)
        window = cluster_filter.session_window(selection, self.recording_ids, spans)
        inside = None if window is None else self.session_shares(window)
        if self.has_positions():
            # Reduced per (leaf, recording) before anything is copied: casting the whole
            # (L, R, 16, 16) table and then indexing it cost two copies of it at float64.
            per_rec = self._pos_by_recording()
            total = per_rec.sum(axis=1)
            if cells is None:
                in_cells = per_rec
            else:
                in_cells = self._pos_in_cells(cells)
        else:
            if cells is not None:
                raise ValueError(
                    "this run recorded no positions, so a region of the frame cannot narrow it"
                )
            in_cells = t.leaf_rec_time_ns.astype(np.float64)
            total = in_cells.sum(axis=1)
        if inside is not None:
            in_cells = in_cells * inside
        matched = in_cells[:, recordings].sum(axis=1)
        return self._node_sums(matched), self._node_sums(total)

    def session_shares(self, window) -> np.ndarray:
        """``(L, R) float64``: the share of each leaf's member time in each recording that lies
        inside ``window`` (``cluster_filter.session_window``), ``0`` where it has none.

        Summed from the leaf runs, which tile every member's time: each run adds its overlap with
        the window over its own covered time (``run_end_ns - run_start_ns``), so a run half in
        the window adds half. A run of no length counts whole when its start is inside the
        half-open window, so two ranges that meet never both count it. One
        vectorised pass over the runs (``bincount``), no table kept.
        """
        known, start, end = window
        t = self._cache.leaves
        leaves, recordings = t.leaf_rec_time_ns.shape
        rec = t.run_recording.astype(np.int64)
        s = t.run_start_ns.astype(np.int64)
        e = t.run_end_ns.astype(np.int64)
        lo, hi = start[rec], end[rec]
        length = e - s
        overlap = np.clip(np.minimum(e, hi) - np.maximum(s, lo), 0, None)
        point = (length <= 0) & (s >= lo) & (s < hi)
        inside = np.where(length > 0, overlap, point.astype(np.int64))
        inside = np.where(known[rec], inside, 0).astype(np.float64)
        whole = np.where(length > 0, length, 1).astype(np.float64)
        cell = t.run_leaf.astype(np.int64) * recordings + rec
        size = leaves * recordings
        in_time = np.bincount(cell, weights=inside, minlength=size)
        all_time = np.bincount(cell, weights=whole, minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(all_time > 0, in_time / np.maximum(all_time, 1), 0.0)
        return share.reshape(leaves, recordings)

    def _pos_by_recording(self) -> np.ndarray:
        """``(L, R) int64`` members with a known position, per leaf and recording; kept."""
        if getattr(self, "_pos_lr", None) is None:
            self._pos_lr = self._cache.leaves.leaf_rec_pos.sum(
                axis=(2, 3), dtype=np.int64
            )
        return self._pos_lr

    def _pos_in_cells(self, cells: np.ndarray) -> np.ndarray:
        """``(L, R) int64`` members in ``cells``. Copies the smaller side of the mask only: a
        region over most of the frame is counted as the whole minus what lies outside it.
        """
        pos = self._cache.leaves.leaf_rec_pos
        flat = pos.reshape(pos.shape[0], pos.shape[1], -1)
        mask = np.asarray(cells, dtype=bool).ravel()
        if mask.sum() <= mask.size // 2:
            return flat[:, :, np.flatnonzero(mask)].sum(axis=2, dtype=np.int64)
        outside = flat[:, :, np.flatnonzero(~mask)].sum(axis=2, dtype=np.int64)
        return self._pos_by_recording() - outside

    def match_shares(
        self,
        selection: cluster_filter.Selection,
        metadata: dict | None = None,
        column_types: dict | None = None,
        spans: dict | None = None,
    ) -> np.ndarray:
        """``(n_nodes,)`` each node's share of members meeting ``selection``; ``NaN`` where the
        node has nothing to count (``match_counts``)."""
        matched, total = self.match_counts(selection, metadata, column_types, spans)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, matched / np.maximum(total, 1), np.nan)

    def recording_spans(self) -> dict[str, tuple[int, int]]:
        """``{recording_id: (first_ns, last_ns)}`` the members cover, for drawing a timeline."""
        t = self._cache.leaves
        out: dict[str, tuple[int, int]] = {}
        for index, recording_id in enumerate(self.recording_ids):
            mask = t.run_recording == index
            if mask.any():
                out[recording_id] = (
                    int(t.run_start_ns[mask].min()),
                    int(t.run_end_ns[mask].max()),
                )
        return out

    # --- labels --------------------------------------------------------------------------

    def set_clip_edits(self, edits, labels: dict) -> None:
        """Take the run's clip edits (``storage/clip_edits.py``) into ``label_time`` and
        ``labelled_runs``, so the label bar and BROWSE's preview agree with the export.

        ``labels`` is the state's ``{label id: name}``, because the readers here are handed
        ``{node: name}``. Each edit is matched to the cache's leaf runs that lie inside its span in
        its recording and subject, the same rule the derivation uses (``api/clip_edits.py``).
        """
        from detail.storage import clip_edits as clip_edits_store

        t = self._cache.leaves
        end = t.run_end_ns if t.run_last_ns is None else t.run_last_ns
        index = {rid: i for i, rid in enumerate(self.recording_ids)}
        resolved = []
        for edit in clip_edits_store.latest(edits):
            recording = index.get(edit["recording_id"])
            from_name = labels.get(edit["from_label"])
            if recording is None or not from_name:
                continue
            runs = np.flatnonzero(
                (t.run_recording == recording)
                & (t.run_subject.astype(np.int64) == int(edit["subject"]))
                & (t.run_start_ns >= int(edit["t_ns"]))
                & (end <= int(edit["end_t_ns"]))
            )
            if len(runs):
                resolved.append((runs, from_name, labels.get(edit["to_label"] or "")))
        self._clip_edits = resolved

    def _run_nodes(self, node_names: dict, apply_edits: bool = True) -> np.ndarray:
        """``(U,)`` the labelled node each leaf run reports under, ``-1`` for none, after the clip
        edits when ``apply_edits``.

        An edit applies to a run whose node, or a node above it, carries the label it was taken
        out of; a run whose label has since changed is left as the labels say. A clip moved to a
        label reports under that label's first node in ``node_names``, or unlabelled when
        ``node_names`` has none (a lane drawn for one node).
        """
        named = {int(n) for n, name in node_names.items() if name}
        groups = np.asarray(self._tree.leaf_groups(named), dtype=np.int64)
        t = self._cache.leaves
        nodes = groups[t.run_leaf.astype(np.int64)] if len(t.run_leaf) else groups[:0]
        if not apply_edits or not self._clip_edits:
            return nodes
        nodes = nodes.copy()
        fold = {str(n): str(name).casefold() for n, name in node_names.items() if name}
        first_of: dict[str, int] = {}
        for node in sorted(fold, key=int):
            first_of.setdefault(fold[node], int(node))
        for runs, from_name, to_name in self._clip_edits:
            target = -1 if not to_name else first_of.get(to_name.casefold(), -1)
            for run in runs:
                node = int(nodes[run])
                if node < 0:
                    continue
                chain = [node, *self.ancestors(node)]
                if any(fold.get(str(c)) == from_name.casefold() for c in chain):
                    nodes[run] = target
        return nodes

    def label_time(self, node_names: dict) -> dict[str, int]:
        """``{node id: member time}`` per labelled node, most specific label winning, plus
        ``"unlabelled"``: the histogram the labels panel draws, updated with no reprojection.

        A clip taken out of a label (``set_clip_edits``) moves its run's covered time to
        ``"unlabelled"``, or to the label it was moved to.
        """
        named = {int(n) for n, name in node_names.items() if name}
        groups = self._tree.leaf_groups(named)
        out: dict[str, int] = {}
        for leaf, time in enumerate(self._cache.leaves.leaf_time_ns):
            key = "unlabelled" if groups[leaf] < 0 else str(int(groups[leaf]))
            out[key] = out.get(key, 0) + int(time)
        if self._clip_edits:
            t = self._cache.leaves
            before = np.asarray(groups, dtype=np.int64)[t.run_leaf.astype(np.int64)]
            after = self._run_nodes(node_names)
            for run in np.flatnonzero(before != after):
                time = int(t.run_end_ns[run]) - int(t.run_start_ns[run])
                for node, sign in ((before[run], -1), (after[run], 1)):
                    key = "unlabelled" if node < 0 else str(int(node))
                    out[key] = out.get(key, 0) + sign * time
        return out

    def cluster_of_spans(self, spans: list, labelled: set[int]) -> list[int | None]:
        """For each ``(recording_id, subject, t0_ns, t1_ns)``, the labelled node holding most of it.

        A bout reports a **label**, and the clusters of one label fuse (§22), so one instance can
        cross two of them; asking which cluster it belongs to means asking which holds most of its
        time. Answered from the cache's leaf runs: each run overlapping the span counts its overlap
        (plus one, so an instant still counts) toward the deepest node of ``labelled`` above its
        leaf. ``None`` where nothing labelled overlaps.

        Each span reads only the runs that can overlap it: the runs are sorted once per unit, and
        a unit's runs tile its time without overlapping, so their starts and ends are both sorted
        and two ``searchsorted`` calls bound the window. The audit asks this of every instance in a
        label, so a scan of every run per span would be quadratic in the run's size.
        """
        t = self._cache.leaves
        if not len(t.run_leaf):
            return [None for _ in spans]
        units, starts, ends, leaves = self._runs_by_unit()
        owner = self._tree.leaf_groups({int(n) for n in labelled})[leaves]
        index = {rid: i for i, rid in enumerate(self.recording_ids)}
        out: list[int | None] = []
        for recording_id, subject, t0, t1 in spans:
            bounds = units.get((index.get(recording_id), int(subject or 0)))
            if bounds is None:
                out.append(None)
                continue
            lo, hi = bounds
            a = lo + int(np.searchsorted(ends[lo:hi], int(t0), side="left"))
            b = lo + int(np.searchsorted(starts[lo:hi], int(t1), side="right"))
            if b <= a:
                out.append(None)
                continue
            overlap = np.minimum(ends[a:b], int(t1)) - np.maximum(starts[a:b], int(t0))
            mine = owner[a:b]
            # Runs are half-open, so a run that only touches the span adds nothing. An instant
            # (t0 == t1) overlaps every run by zero: it belongs to the run that contains it, not
            # to the neighbour whose end it shares — the tie once went to the earlier run.
            if (overlap > 0).any():
                keep = (overlap > 0) & (mine >= 0)
            else:
                keep = (starts[a:b] <= int(t0)) & (ends[a:b] > int(t0)) & (mine >= 0)
            if not keep.any():
                out.append(None)
                continue
            weight: dict[int, int] = {}
            for node, amount in zip(mine[keep], overlap[keep]):
                weight[int(node)] = weight.get(int(node), 0) + int(amount) + 1
            out.append(max(weight, key=weight.get))
        return out

    def _runs_by_unit(self):
        """The leaf runs sorted by (recording, subject, start), with each unit's slice — built once
        per view, since the cache does not change under it."""
        if getattr(self, "_run_index", None) is None:
            t = self._cache.leaves
            order = np.lexsort(
                (t.run_start_ns, t.run_subject.astype(np.int64), t.run_recording)
            )
            recording = t.run_recording[order].astype(np.int64)
            subject = t.run_subject[order].astype(np.int64)
            units: dict = {}
            begin = 0
            for i in range(1, len(order) + 1):
                if i == len(order) or (
                    recording[i] != recording[begin] or subject[i] != subject[begin]
                ):
                    units[(int(recording[begin]), int(subject[begin]))] = (begin, i)
                    begin = i
            self._run_index = (
                units,
                t.run_start_ns[order].astype(np.int64),
                t.run_end_ns[order].astype(np.int64),
                t.run_leaf[order].astype(np.int64),
            )
        return self._run_index

    def labelled_runs(self, node_names: dict, apply_edits: bool = True) -> list[dict]:
        """The labels ethogram: leaf runs mapped through the labels, merged, unlabelled dropped.

        ``[{recording_id, subject, start_ns, end_ns, node}]``. A preview built from the cache, so
        it follows every label at once; BROWSE shows the exact instances once the output catches
        up. Clips taken out of a label are dropped, or moved, as the export has them
        (``set_clip_edits``); ``apply_edits=False`` gives the clusters as fitted, the raw lane.
        """
        t = self._cache.leaves
        node_of_run = self._run_nodes(node_names, apply_edits)
        out: list[dict] = []
        for i in range(len(t.run_leaf)):
            node = int(node_of_run[i])
            if node < 0:
                continue
            here = {
                "recording_id": self.recording_ids[int(t.run_recording[i])],
                "subject": int(t.run_subject[i]),
                "start_ns": int(t.run_start_ns[i]),
                "end_ns": int(t.run_end_ns[i]),
                "node": node,
            }
            last = out[-1] if out else None
            if (
                last is not None
                and last["node"] == node
                and last["recording_id"] == here["recording_id"]
                and last["subject"] == here["subject"]
                and here["start_ns"] <= last["end_ns"]
            ):
                last["end_ns"] = max(last["end_ns"], here["end_ns"])
            else:
                out.append(here)
        return out
