"""What a cluster-view filter selects, as arithmetic the GUI and a script share (§22).

A **selection** narrows a clustering run to the members that meet every active condition: values of
a metadata column, a range of a number or date column, a set of recordings, and a set of cells of the
frame. Metadata and recordings reduce to a set of recordings (metadata is per recording); a region
is a set of cells of the 16×16 position grid the node cache counts members in. So every node's
matching share is exact from ``leaf_rec_pos`` (``NodeView.match_shares``), and a clip matches when
its recording and its mean position do (``clip_matches``).

**Time in session** is a range ``(lo, hi)`` of ``0..1`` over each recording's **whole video**, its
first frame (``0``) to its last (``1``): the first and last ``t_ns`` of its active timestamp map
(``session_spans``; a multi-part recording's map runs from the start of its first part, so the
span is the whole recording). It is normalised per recording, so ``(0, 0.25)`` is the first
quarter of every recording whatever its length. A run that clustered only part of a video simply
has no members in part of the range. A recording with no span (no timestamp map) never matches a
session range, as a recording with no value never matches a metadata range.

How an edge clip counts follows Location's split between shares and clips: a node's matching
share counts only the **time** inside the range (a run half inside adds half its time,
``NodeView.match_counts``), while a clip matches when its **midpoint** is inside
(``clip_matches``), the way a clip matches a region by its mean position.

**OR within a column, AND across columns.** Two values of one column widen the selection; a
second column narrows it. A recording with no value in a column never matches a range on it, and
matches a value condition only through ``""`` (no value).

**Clusters** (``leaves``, key :data:`CLUSTERS`) is the condition the difference screen makes
(§22, §24): the leaves of the merge tree under the clusters a saved node comparison kept. It is a
fact about a cluster, not a member, and it fits the same arithmetic because a member sits in
exactly one leaf: a kept cluster's share is 100%, its parent's the fraction of its time under
kept children, a sibling branch's 0%, so the tree draws the lineage; a clip matches when its own
leaf (``ClipTable.leaf``) is in the set. ``leaves_source`` names what picked them (the comparison
id) for the page that shows it, and takes no part in the arithmetic. An empty set is a real
condition — a screen that kept nothing — and matches nothing, the way a value nobody has does.

Pure: numpy and the metadata parsers, no zarr, no Qt — except ``session_spans``, which reads each
recording's timestamp map and is what a caller hands the rest as ``spans``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from detail.analysis.node_cache import POSITION_BINS
from detail.storage import metadata as metadata_store

#: Condition keys besides a metadata column's ``("meta", column)``.
VIDEO = ("video",)
LOCATION = ("location",)
SESSION = ("session",)
CLUSTERS = ("clusters",)


@dataclass(frozen=True)
class Selection:
    """The conditions in force. Every field left at its default selects everything.

    - ``values``: ``{column: {value, ...}}`` for a category column; ``""`` is "no value".
    - ``ranges``: ``{column: (low, high)}`` for a number or date column, inclusive, ``None`` an
      open end. Number bounds are numbers; date bounds are ISO strings (``"2026-07-02"``).
    - ``recording_ids``: the recording ids kept, or ``None`` for all of them.
    - ``cells``: ``{(bx, by), ...}`` cells of the ``POSITION_BINS`` grid over the frame
      (``bx`` across, ``by`` down), or ``None`` for the whole frame.
    - ``session``: ``(lo, hi)`` fractions of each recording's whole video, ``0 <= lo <= hi <= 1``,
      or ``None`` for all of it. ``(0, 1)`` is stored as ``None``: the whole session is no filter.
    - ``leaves``: the leaves of the merge tree whose members match (the clusters a difference
      screen kept, as leaves), or ``None`` for every leaf; ``leaves_source`` names what picked
      them, a saved comparison's id, and is carried for the page, not the arithmetic.

    Raises:
        ValueError: a ``session`` outside ``0..1`` or with ``lo > hi``, or a ``leaves_source``
            with no ``leaves``.
    """

    values: Mapping[str, frozenset] = field(default_factory=dict)
    ranges: Mapping[str, tuple] = field(default_factory=dict)
    recording_ids: frozenset | None = None
    cells: frozenset | None = None
    session: tuple[float, float] | None = None
    leaves: frozenset | None = None
    leaves_source: str | None = None

    def __post_init__(self) -> None:
        if self.leaves_source is not None and self.leaves is None:
            raise ValueError(
                f"leaves_source {self.leaves_source!r} names what picked the leaves, but no "
                f"leaves were given"
            )
        if self.leaves is not None:
            object.__setattr__(self, "leaves", frozenset(int(v) for v in self.leaves))
        if self.session is None:
            return
        lo, hi = (float(v) for v in self.session)
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"a time-in-session range is two fractions 0 <= lo <= hi <= 1, got "
                f"{self.session!r}"
            )
        whole = lo <= 0.0 and hi >= 1.0
        object.__setattr__(self, "session", None if whole else (lo, hi))

    @property
    def is_empty(self) -> bool:
        """Whether nothing narrows: every member matches."""
        return (
            not self.values
            and not self.ranges
            and self.recording_ids is None
            and self.cells is None
            and self.session is None
            and self.leaves is None
        )

    @property
    def needs_metadata(self) -> bool:
        return bool(self.values or self.ranges)

    def keys(self) -> list[tuple]:
        """Every condition in force, one key each: ``("meta", column)``, then :data:`VIDEO`,
        then :data:`LOCATION`, then :data:`SESSION`, then :data:`CLUSTERS` — the order a page
        lists them in."""
        out: list[tuple] = [("meta", c) for c in (*self.values, *self.ranges)]
        if self.recording_ids is not None:
            out.append(VIDEO)
        if self.cells is not None:
            out.append(LOCATION)
        if self.session is not None:
            out.append(SESSION)
        if self.leaves is not None:
            out.append(CLUSTERS)
        return out

    def without(self, key: tuple) -> "Selection":
        """This selection with the condition ``key`` removed."""
        if key == VIDEO:
            return replace(self, recording_ids=None)
        if key == LOCATION:
            return replace(self, cells=None)
        if key == SESSION:
            return replace(self, session=None)
        if key == CLUSTERS:
            return replace(self, leaves=None, leaves_source=None)
        column = key[1]
        return replace(
            self,
            values={k: v for k, v in self.values.items() if k != column},
            ranges={k: v for k, v in self.ranges.items() if k != column},
        )

    def only(self, key: tuple) -> "Selection":
        """Just the condition ``key``: what a per-condition ✓ / ✗ is asked of, so the
        breakdown and the match use one rule (``clip_matches``)."""
        if key == VIDEO:
            return Selection(recording_ids=self.recording_ids)
        if key == LOCATION:
            return Selection(cells=self.cells)
        if key == SESSION:
            return Selection(session=self.session)
        if key == CLUSTERS:
            return Selection(leaves=self.leaves, leaves_source=self.leaves_source)
        column = key[1]
        if column in self.values:
            return Selection(values={column: self.values[column]})
        return Selection(ranges={column: self.ranges[column]})


def recording_mask(
    selection: Selection,
    recording_ids,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
    column_types: Mapping[str, str] | None = None,
) -> np.ndarray:
    """``(R,) bool``: which of ``recording_ids`` meet every recording-level condition.

    ``metadata`` is ``{recording_id: {column: value}}`` (``Experiment.recording_metadata``);
    ``column_types`` defaults to ``infer_column_types(metadata)``.

    Raises:
        ValueError: a metadata condition with no ``metadata``, a range on a category column, or
            a range bound that does not parse as the column's type.
    """
    ids = list(recording_ids)
    keep = np.ones(len(ids), dtype=bool)
    if selection.recording_ids is not None:
        keep &= np.array([r in selection.recording_ids for r in ids], dtype=bool)
    if not selection.needs_metadata:
        return keep
    if metadata is None:
        raise ValueError("a metadata condition needs the experiment's metadata")
    types = dict(
        column_types
        if column_types is not None
        else metadata_store.infer_column_types(metadata)
    )

    def value_of(recording_id: str, column: str) -> str:
        return str((metadata.get(recording_id) or {}).get(column) or "").strip()

    for column, wanted in selection.values.items():
        allowed = {str(v).strip() for v in wanted}
        keep &= np.array([value_of(r, column) in allowed for r in ids], dtype=bool)
    for column, (low, high) in selection.ranges.items():
        kind = types.get(column, "category")
        if kind == "number":
            parse = metadata_store.parse_number
        elif kind == "date":
            parse = metadata_store.parse_date
        elif kind == "time":
            parse = metadata_store.parse_time
        else:
            raise ValueError(
                f"column {column!r} is a category; filter it by values, not a range"
            )
        bounds = []
        for bound in (low, high):
            if bound is None:
                bounds.append(None)
                continue
            parsed = parse(str(bound))
            if parsed is None:
                raise ValueError(f"{bound!r} is not a {kind} for column {column!r}")
            bounds.append(parsed)
        lo, hi = bounds
        inside = []
        for recording_id in ids:
            value = parse(value_of(recording_id, column))
            inside.append(
                value is not None
                and (lo is None or value >= lo)
                and (hi is None or value <= hi)
            )
        keep &= np.array(inside, dtype=bool)
    return keep


def cell_mask(selection: Selection, bins: int = POSITION_BINS) -> np.ndarray | None:
    """``(bins, bins) bool`` indexed ``[by, bx]`` like ``leaf_rec_pos``, or ``None`` for no region.

    Raises:
        ValueError: a cell outside the grid.
    """
    if selection.cells is None:
        return None
    mask = np.zeros((bins, bins), dtype=bool)
    for bx, by in selection.cells:
        if not (0 <= int(bx) < bins and 0 <= int(by) < bins):
            raise ValueError(f"cell {(bx, by)} is outside the {bins}×{bins} grid")
        mask[int(by), int(bx)] = True
    return mask


def leaf_mask(selection: Selection, n_leaves: int) -> np.ndarray | None:
    """``(n_leaves,) bool``: which leaves the clusters condition keeps, or ``None`` with none.

    Raises:
        ValueError: a leaf outside the tree.
    """
    if selection.leaves is None:
        return None
    mask = np.zeros(int(n_leaves), dtype=bool)
    for leaf in selection.leaves:
        if not (0 <= int(leaf) < n_leaves):
            raise ValueError(f"leaf {leaf} is outside a tree of {n_leaves} leaves")
        mask[int(leaf)] = True
    return mask


def session_window(
    selection: Selection, recording_ids, spans: Mapping[str, tuple[int, int]] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``(known, start_ns, end_ns)`` per recording, each ``(R,)``: the stretch of each recording's
    video ``selection.session`` covers, and whether the recording has a span at all; ``None``
    with no session range.

    ``spans`` is ``{recording_id: (first_ns, last_ns)}`` (``session_spans``). The window is
    **half-open**, ``[first + lo · (last - first), first + hi · (last - first))`` in ``t_ns``, both
    edges rounded by the one rule (``_edge``), so ``[0, x]`` and ``[x, 1]`` meet at the same
    nanosecond and nothing on the line is in both. A range from ``0`` or to ``1`` is open at that
    end: ``1`` is the end of the recording, and a run's covered time reaches past the last frame by
    one frame step, which would otherwise sit in no range at all, so the two halves would not add
    up to the whole and the last run of a session could never be wholly at its end.

    Raises:
        ValueError: a session range with no ``spans``.
    """
    if selection.session is None:
        return None
    if spans is None:
        raise ValueError(
            "a time-in-session range needs each recording's session span (session_spans)"
        )
    lo, hi = selection.session
    ids = list(recording_ids)
    known = np.array([r in spans for r in ids], dtype=bool)
    first = np.array([spans[r][0] if r in spans else 0 for r in ids], dtype=np.int64)
    last = np.array([spans[r][1] if r in spans else 0 for r in ids], dtype=np.int64)
    length = (last - first).astype(np.float64)
    bounds = np.iinfo(np.int64)
    if lo <= 0.0:
        start = np.full(len(ids), bounds.min, dtype=np.int64)
    else:
        start = _edge(first, length, lo)
    if hi >= 1.0:
        end = np.full(len(ids), bounds.max, dtype=np.int64)
    else:
        end = _edge(first, length, hi)
    return known, start, end


def _edge(first: np.ndarray, length: np.ndarray, fraction: float) -> np.ndarray:
    """The ``t_ns`` at ``fraction`` of each recording: one rounding for a range's start and end
    alike, so the end of ``[a, x)`` is the start of ``[x, b)`` to the nanosecond."""
    return first + np.floor(float(fraction) * length).astype(np.int64)


def clip_matches(
    selection: Selection,
    clips,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
    column_types: Mapping[str, str] | None = None,
    spans: Mapping[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """``(C,) bool``: which of a ``ClipTable``'s clips meet every condition.

    A clip's recording must pass ``recording_mask``, and under a region its mean position must
    fall in a selected cell; a clip with no known position never matches a region. Under a
    time-in-session range its **midpoint** (``(start_ns + end_ns) // 2``, the frame a montage
    tile stands for) must fall inside the range's stretch of its recording (``spans``,
    ``session_window``): half-open, ``[lo, hi)``, with a range from ``0`` or to ``1`` open at that
    end, so ``[0, x]`` and ``[x, 1]`` split a recording's clips with none in both, a clip exactly
    on ``x`` going to the later one. A clip of a recording with no span never matches one.
    Under a clusters condition the clip's own leaf must be one of the kept leaves.
    """
    by_recording = recording_mask(
        selection, clips.recording_ids, metadata, column_types
    )
    recording = np.asarray(clips.recording, dtype=np.int64)
    out = by_recording[recording]
    if selection.leaves is not None:
        leaf = np.asarray(clips.leaf, dtype=np.int64)
        out = out & np.isin(leaf, np.fromiter(selection.leaves, dtype=np.int64))
    window = session_window(selection, clips.recording_ids, spans)
    if window is not None:
        known, start, end = window
        mid = (
            np.asarray(clips.start_ns, dtype=np.int64)
            + np.asarray(clips.end_ns, dtype=np.int64)
        ) // 2
        out = (
            out & known[recording] & (mid >= start[recording]) & (mid < end[recording])
        )
    cells = cell_mask(selection)
    if cells is None:
        return out
    cx = np.asarray(clips.cx, dtype=np.float64)
    cy = np.asarray(clips.cy, dtype=np.float64)
    known = np.isfinite(cx) & np.isfinite(cy)
    bins = cells.shape[0]
    bx = np.clip((np.nan_to_num(cx) * bins).astype(np.int64), 0, bins - 1)
    by = np.clip((np.nan_to_num(cy) * bins).astype(np.int64), 0, bins - 1)
    return out & known & cells[by, bx]


def session_spans(experiment_dir, recording_ids) -> dict[str, tuple[int, int]]:
    """``{recording_id: (first_ns, last_ns)}``: each recording's whole video, from its active
    timestamp map (Tenet 8: the frame count and times come from the map, never from cv2). A
    recording with no map, or an empty one, is left out, and so matches no session range.

    Reads only the first and last part's map (through the mtime-cached reader), since ``t_ns``
    runs from the start of the first part. Does disk I/O: call it off the UI thread.
    """
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    out: dict[str, tuple[int, int]] = {}
    for recording_id in recording_ids:
        span = recording_store.session_span_ns(
            experiment_store.recording_path(experiment_dir, recording_id)
        )
        if span is not None:
            out[recording_id] = span
    return out
