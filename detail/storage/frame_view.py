"""The §19 frame view — the behavior artifact's intervals, materialized per frame.

Intervals are the run-length encoding of a per-frame array, and the two hold the same
information. ``instances.csv`` stores the intervals because two things do not survive the
expansion — bout identity (two abutting instances of one label merge into one rising edge) and
a per-instance score (a windowed matcher's confidence means nothing smeared across frames, and
an audit decision attaches to an *instance*). But most consumers want frames, so the expansion
is materialized here rather than re-derived, differently, by each of them.

One zarr group per ``(recording_id, subject)`` under ``analysis/<an_NN>/frames/``::

    frames/<recording_id>__<subject>.zarr
      t_ns           int64  (N,)   the frame time axis
      observed       bool   (N,)   was this frame inside an analyzed span
      instance_id    str    (B,)   ordinal i in a label array refers to instance_id[i - 1]
      <source_label> int32  (N,)   0 = not active, else the bout's ordinal

**``observed`` is the reason this file is required rather than convenient.** ``coverage.csv``
gives ``observed_ns`` for a whole ``(recording, subject)``, which cannot answer "how much of the
first ten minutes was observed" — that needs to know *where* the unobserved rows are, not merely
how many. A denominator taken from the whole recording and applied to a sub-window inflates or
deflates every rate in proportion to where the tracking gaps happen to fall, and nothing about
the resulting table looks wrong (§23).

**Cadence does not make a frame unobserved.** A trajectory has holes from two causes — tracking
gaps and frame cadence — and only the first is a gap in *observation*. A run at
``every_n_frames=5`` covered the same wall-clock time as one at 1; it merely sampled it. Marking
only the sampled frames would report a fifth of the real observed time and make every rate come
out five times too high. :func:`observed_mask` is where that rule lives, so a producer cannot get
it subtly wrong on its own.

**Keyed by ``source_label``, never by the user's name.** Renaming a cluster then rewrites nothing
— the naming map applies at read time — and a cluster carrying two names is one array read twice
with one ordinal, so the double count is prevented by construction rather than by care. Naming a
*branch* above the cut is not a rename: it fuses the bouts under it into one ``source_label``.

**Each view says what it was built from, and an authored edit writes none** (#325). The edit
writes the intervals and, if any view's :func:`built_from` digest no longer matches them, marks the
run pending (:func:`set_pending`); every reader here then refuses until
``AuthoredRun.bring_frame_views_current`` has rebuilt the mismatched units. A view is a
materialized expansion that most edits' surfaces never read, so it is paid for once, by whoever
needs it, rather than by every name.

**Intervals stay authoritative.** This is a materialized view, not a second truth: if the two ever
disagree, regenerate. Which is also why the ordinal below is allowed to be positional.
"""

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detail.analysis.clip_query import (
    default_max_gap_ns,
    nominal_period_ns,
    segment_runs,
)
from detail.core import atomic
from detail.core.ids import sorted_by_id
from detail.core.utils import coerce_series_key
from detail.storage import behavior_output
from detail.storage import zarr_store

SCHEMA_VERSION = 1

FRAMES_DIRNAME = behavior_output.FRAMES_DIRNAME
T_NS_ARRAY = "t_ns"
OBSERVED_ARRAY = "observed"
INSTANCE_ID_ARRAY = "instance_id"

# The array names the group reserves for itself; a source_label may not collide with one.
_RESERVED = frozenset({T_NS_ARRAY, OBSERVED_ARRAY, INSTANCE_ID_ARRAY})

# Matches every scalar column in ``storage/zarr_io.py``. ``shards=`` stays unset: the shard
# policy DESIGN_PLAN §9 describes belongs to the (unbuilt) StorageLocation backend, and nothing
# in the repo sets one.
_CHUNK = 10_000

# A source_label becomes a zarr array name, so it has to be a safe single token.
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_-]+$")

_NO_SUBJECT = "none"


#: The group attribute recording what a unit's view was built from — :func:`built_from`'s digest.
BUILT_FROM_ATTR = "built_from"

#: Dropped into ``frames/`` while any view may disagree with ``instances.csv``, removed once none
#: does. An authored edit leaves the views to be brought current by whoever reads them next
#: (§21), so this is the one thing standing between a reader and a stale view.
PENDING_FILENAME = "PENDING"

#: Part of every digest, so a change to what this module writes invalidates every view built by
#: the old code instead of letting them pass as current.
_WRITER_VERSION = 1


class FrameViewError(ValueError):
    """The pieces handed to the writer do not describe a frame view."""


class FrameViewsPendingError(RuntimeError):
    """A run's frame views are out of date after an edit and have not been brought current.

    Deliberately not a ``ValueError`` / ``OSError`` / ``KeyError``: readers catch those to treat a
    missing view as "fewer numbers", and a stale view must never be read as a missing one.
    """


def frames_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / FRAMES_DIRNAME


def is_pending(run_dir: Path | str) -> bool:
    """Whether some view of this run may disagree with its ``instances.csv``."""
    return (frames_dir(run_dir) / PENDING_FILENAME).is_file()


def set_pending(run_dir: Path | str, pending: bool) -> None:
    """Record (or clear) that the views may be out of date.

    Set **before** the write that makes them so, never after: a kill between an ``instances.csv``
    rewrite and a later mark would leave a stale view with nothing saying so.
    """
    marker = frames_dir(run_dir) / PENDING_FILENAME
    if pending:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    else:
        marker.unlink(missing_ok=True)


def refuse_if_pending(run_dir: Path | str) -> None:
    """Raise rather than let a reader see a view an edit has moved past.

    Raises:
        FrameViewsPendingError: the run's views have not been brought current since an edit.
    """
    if is_pending(run_dir):
        raise FrameViewsPendingError(
            f"the frame views in {Path(run_dir).name} are out of date after an edit. Bring them "
            "current through the run first (AuthoredRun.bring_frame_views_current, which the "
            "measures, finish and export paths already call)."
        )


def built_from(instances, coverage, gap_ns: int | None) -> str:
    """A digest of everything a unit's view is a function of: its bouts, its coverage, its gap.

    Stored on the view, so whether a view is current is a fact about the view itself rather than
    about a flag somewhere else that can drift from it.
    """
    bouts = sorted(
        (row.instance_id, str(row.source_label), int(row.t_ns), int(row.end_t_ns))
        for row in behavior_output.bouts(list(instances))
    )
    payload = json.dumps(
        [
            _WRITER_VERSION,
            bouts,
            [
                int(coverage.start_t_ns),
                int(coverage.end_t_ns),
                int(coverage.observed_ns),
                int(coverage.n_observed),
            ],
            None if gap_ns is None else int(gap_ns),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stored_built_from(
    run_dir: Path | str, recording_id: str, subject: int | None
) -> str | None:
    """The digest a unit's view was built from, or ``None`` if it has no view or predates one."""
    path = unit_path(run_dir, recording_id, subject)
    if not path.is_dir():
        return None
    value = zarr_store.open_group(path, mode="r").attrs.get(BUILT_FROM_ATTR)
    return None if value is None else str(value)


def unit_filename(recording_id: str, subject: int | None) -> str:
    """``<recording_id>__<subject>.zarr`` — the flat per-unit store name (§19)."""
    return f"{recording_id}__{_NO_SUBJECT if subject is None else subject}.zarr"


def unit_path(run_dir: Path | str, recording_id: str, subject: int | None) -> Path:
    return frames_dir(run_dir) / unit_filename(recording_id, subject)


def parse_unit(stem: str) -> tuple[str, int | None]:
    """Recover ``(recording_id, subject)`` from a store's stem.

    Split from the right: a recording id is a minted ``recording_NNN`` and carries no ``__``,
    but splitting from the left would break the moment one did.
    """
    recording_id, _, raw = stem.rpartition("__")
    if not recording_id:
        raise FrameViewError(
            f"frame-view store {stem!r} is not <recording_id>__<subject>"
        )
    return recording_id, None if raw == _NO_SUBJECT else coerce_series_key(raw)


def observed_mask(
    axis_t_ns, analyzed_t_ns, max_gap_ns: int | None = None
) -> np.ndarray:
    """Which frames of ``axis_t_ns`` fall inside a continuous analyzed span.

    ``analyzed_t_ns`` are the times of the rows the analysis actually held — an embedding
    trajectory's rows, say. Those are sparse for two different reasons, and the whole point of
    this function is that only one of them is a hole in *observation*:

    - a **tracking gap** is genuinely unobserved, and splits the analyzed span in two;
    - **frame cadence** is not. A run at ``every_n_frames=5`` still covered that wall-clock
      time. Treating its unsampled frames as unobserved would divide every rate by the cadence.

    So the spans come from :func:`segment_runs`, which splits only where a step exceeds
    ``max_gap_ns`` (defaulting to ~4x the observed cadence), and every axis frame inside a span
    is observed.

    **A row covers its sampling interval, not just its own instant.** A run's last sampled row
    sits up to one cadence short of where the run actually ended, so ending the span there would
    shave up to ``cadence - 1`` frames off every run — small, but in the direction that makes a
    coarser cadence look like less observation, which is exactly the bias this function exists
    to remove. Each span therefore runs to ``last_row + period``, and the dense and sampled
    versions of the same coverage come out equal.

    Raises:
        FrameViewError: ``axis_t_ns`` is not sorted — the mask is built by binary search.
    """
    axis = np.asarray(axis_t_ns, dtype=np.int64)
    rows = np.asarray(analyzed_t_ns, dtype=np.int64)
    if axis.size and np.any(np.diff(axis) < 0):
        raise FrameViewError("axis_t_ns must be non-decreasing")
    mask = np.zeros(axis.size, dtype=bool)
    if rows.size == 0 or axis.size == 0:
        return mask
    gap = default_max_gap_ns(rows) if max_gap_ns is None else max_gap_ns
    period = nominal_period_ns(rows)
    for start, end in segment_runs(rows, gap):
        if end <= start:
            continue
        lo = int(np.searchsorted(axis, rows[start], side="left"))
        hi = int(np.searchsorted(axis, rows[end - 1] + period, side="left"))
        # No `max(hi, lo + 1)`: `hi == lo` means no axis frame falls inside this run's span at
        # all, and forcing one marked `axis[lo]` — the first frame *after* the run, which may
        # sit far outside it. A run that covers no frame contributes no observed frame.
        mask[lo:hi] = True
    return mask


@dataclass
class FrameView:
    """A read frame view. Arrays are lazy zarr handles — a 100-hour unit is 173 MB of ``t_ns``
    alone, so nothing is materialized until asked for."""

    recording_id: str
    subject: int | None
    root: object
    #: Materialized ``(axis, observed, durations)``, filled on first use by :meth:`arrays`.
    _arrays: tuple | None = field(default=None, repr=False, compare=False)

    def arrays(self) -> tuple:
        """``(t_ns, observed, frame_durations)`` as numpy, materialized once per view.

        A windowed measure asks for the denominator once per *window*, and a sliding window over
        a long recording asks hundreds of times — re-reading the same two zarr arrays and
        recomputing the same durations every time. Nothing about a view changes while it is open,
        so the read happens once and the caller pays it back over every window.

        Scoped by the view's own lifetime, deliberately: ``measures.measure_run`` drops each unit
        before loading the next, so the cache is one subject, not a cohort — which is the same
        bound the class docstring already promises.
        """
        if self._arrays is None:
            axis = np.asarray(self.t_ns[:], dtype=np.int64)
            mask = np.asarray(self.observed[:], dtype=bool)
            self._arrays = (axis, mask, frame_durations(axis))
        return self._arrays

    @property
    def n_frames(self) -> int:
        return int(self.root[T_NS_ARRAY].shape[0])

    @property
    def t_ns(self):
        return self.root[T_NS_ARRAY]

    @property
    def observed(self):
        return self.root[OBSERVED_ARRAY]

    @property
    def source_labels(self) -> list[str]:
        """The producer ids this unit carries an array for."""
        return sorted(set(self.root.array_keys()) - _RESERVED)

    def instance_ids(self) -> list[str]:
        """``instance_id`` by ordinal — ordinal ``i`` is ``instance_ids()[i - 1]``."""
        if INSTANCE_ID_ARRAY not in self.root:
            return []
        return [str(v) for v in self.root[INSTANCE_ID_ARRAY][:]]


def _validate_labels(labels) -> None:
    for label in labels:
        if label in _RESERVED:
            raise FrameViewError(
                f"source_label {label!r} collides with a reserved array name "
                f"({sorted(_RESERVED)})"
            )
        if not _SAFE_LABEL.match(label):
            raise FrameViewError(
                f"source_label {label!r} is not a safe zarr array name; it must contain "
                "only letters, digits, '-' and '_'"
            )


def _bouts(instances, recording_id: str, subject: int | None) -> list:
    """This unit's bouts, deduplicated by ``instance_id`` and in time order.

    ``instances.csv`` is one row per ``(bout, label)``, so a cluster the user named twice
    appears twice over one span. The frame view is keyed by ``source_label`` and carries the
    bout, so the duplicate rows collapse — which is what keeps a twice-named cluster from
    occupying two ordinals and being counted twice by a consumer reading frames. The collapse
    itself is ``behavior_output.bouts``, shared with the audit, which counted rows until it
    was not.

    Instances belonging to a **different** unit are refused rather than written in. Nothing
    here reads ``recording_id`` or ``subject`` otherwise — they name the file, not the rows —
    so a foreign instance whose span happened to hit this axis was silently given an ordinal
    in this subject's store, and ``instance_id`` hashes the unit, so it was not even collapsed
    away as a duplicate.
    """
    unit = (recording_id, subject)
    foreign = [row.instance_id for row in instances if row.unit != unit]
    if foreign:
        raise FrameViewError(
            f"instance {foreign[0]} belongs to a different (recording_id, subject) than the "
            f"{unit} frame view being written. A frame view holds one unit; writing another "
            "unit's rows into it makes every measure over this subject count another's."
        )
    return sorted(
        behavior_output.bouts(list(instances)),
        key=lambda r: (r.t_ns, r.instance_id),
    )


def write_frame_view(
    run_dir: Path | str,
    *,
    recording_id: str,
    subject: int | None,
    t_ns,
    observed,
    instances,
    built_from: str | None = None,
) -> Path:
    """Write one unit's frame view; return its path.

    Called once per ``(recording_id, subject)`` rather than once per run, so peak memory stays
    proportional to one subject. At 60 Hz a 100-hour unit is 21.6M rows — holding a whole
    cohort's expansion at once is not an option.

    ``instances`` are this unit's rows from the artifact (either the per-bout records or the
    expanded per-label rows; duplicates collapse by ``instance_id``). Each bout takes an ordinal,
    written into its ``source_label``'s array over the frames its span covers, with ``0``
    meaning not active.

    The ordinal is positional, which is safe precisely because this is a **materialized view**:
    it is rebuilt whenever the artifact is, and the ``instance_id`` array carries the join back
    to the stable id the audit uses.

    ``built_from`` is :func:`built_from`'s digest of what this view expands, stored on the group
    and written inside the staged store, so a view carries it only once the whole view exists.

    Raises:
        FrameViewError: mismatched array lengths, an unsafe ``source_label``, or an instance
            whose span does not lie on the axis.
    """
    axis = np.asarray(t_ns, dtype=np.int64)
    mask = np.asarray(observed, dtype=bool)
    if mask.shape != axis.shape:
        raise FrameViewError(
            f"observed has {mask.shape} entries for a {axis.shape} frame axis"
        )
    if axis.size and np.any(np.diff(axis) < 0):
        raise FrameViewError("t_ns must be non-decreasing")

    bouts = _bouts(instances, recording_id, subject)
    _validate_labels({row.source_label for row in bouts})

    columns: dict[str, np.ndarray] = {}
    for ordinal, row in enumerate(bouts, start=1):
        if row.end_t_ns < row.t_ns:
            raise FrameViewError(
                f"instance {row.instance_id} ends before it starts "
                f"({row.t_ns} > {row.end_t_ns})"
            )
        # Addressed by **time**, never by frame index: an embedding part's `frame_idx` is
        # part-local (no cross-part offset is applied), so indexing a recording-global axis by
        # frame number silently misplaces every instance on a multi-part recording. `t_ns` is
        # global by construction.
        lo = int(np.searchsorted(axis, row.t_ns, side="left"))
        hi = int(np.searchsorted(axis, row.end_t_ns, side="right"))
        if hi <= lo:
            raise FrameViewError(
                f"instance {row.instance_id} spans [{row.t_ns}, {row.end_t_ns}] which covers "
                "no frame on this axis — the artifact and the axis disagree"
            )
        column = columns.setdefault(
            row.source_label, np.zeros(axis.size, dtype=np.int32)
        )
        # Refuse rather than overwrite. Bouts arrive in time order, so a later one writing over
        # an occupied stretch silently truncated the earlier bout — or erased it outright when
        # it contained it, leaving an ordinal that appears in `instance_id` and nowhere in the
        # column. The frame view would then disagree with `instances.csv`, which §19 calls
        # authoritative, and a duration read off the frames would be short with nothing wrong
        # on the face of it.
        clash = np.nonzero(column[lo:hi])[0]
        if clash.size:
            occupant = bouts[int(column[lo + int(clash[0])]) - 1]
            raise FrameViewError(
                f"instances {occupant.instance_id} and {row.instance_id} both carry "
                f"source_label {row.source_label!r} and overlap on this axis. One frame holds "
                "one ordinal per label, so writing both would truncate the earlier bout and "
                "make the frame view disagree with instances.csv."
            )
        column[lo:hi] = ordinal

    path = unit_path(run_dir, recording_id, subject)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Built beside the store and swapped in, never cleared in place. `mode="w"` empties the
    # existing directory first, and an authored run *rewrites* its frame view on every
    # reprojection (a granularity change, a naming) while a reader may hold it open — the GUI
    # measures panel is one. Clearing under a reader raced, intermittently, as
    # `[Errno 66] Directory not empty`: a build in four on macOS. The swap is the same
    # discipline the YAML writers already use, one level up from a file.
    staged = path.with_name(path.name + ".tmp")
    shutil.rmtree(staged, ignore_errors=True)
    # Through `zarr_store`, so every chunk and metadata write below rides out the
    # transient network-share replace. Wrapping only this line — which writes one small
    # `zarr.json` — while the `create_array` calls that write every chunk file went bare
    # is the bug that killed a long run on an SMB share.
    root = zarr_store.open_group(staged, mode="w")
    root.attrs["schema_version"] = SCHEMA_VERSION  # Tenet 7 — before any array
    if built_from is not None:
        root.attrs[BUILT_FROM_ATTR] = built_from
    root.create_array(T_NS_ARRAY, data=axis, chunks=(_CHUNK,))
    root.create_array(OBSERVED_ARRAY, data=mask, chunks=(_CHUNK,))
    ids = root.create_array(
        INSTANCE_ID_ARRAY, shape=(len(bouts),), chunks=(_CHUNK,), dtype="str"
    )
    if bouts:
        ids[:] = np.array([row.instance_id for row in bouts], dtype=object)
    for label, column in sorted(columns.items()):
        root.create_array(label, data=column, chunks=(_CHUNK,))

    # Rename the old store aside before moving the new one in: a replace refuses a
    # non-empty directory, and deleting the displaced copy afterwards is harmless even if
    # something is still reading it (POSIX keeps an open file alive through unlink).
    # `atomic.replace`, not `os.replace`: these are directory renames on a share while a
    # reader may hold the store open, which is the same sharing violation the store seam
    # rides out for the writes above — and the one write the store cannot see.
    displaced = path.with_name(path.name + ".old")
    shutil.rmtree(displaced, ignore_errors=True)
    if path.exists():
        atomic.replace(path, displaced)
    atomic.replace(staged, path)
    shutil.rmtree(displaced, ignore_errors=True)
    return path


def read_frame_view(
    run_dir: Path | str, recording_id: str, subject: int | None
) -> FrameView:
    """Open one unit's frame view.

    Raises:
        FileNotFoundError: no store for this unit.
        FrameViewsPendingError: an edit left the run's views out of date (see :func:`set_pending`).
    """
    refuse_if_pending(run_dir)
    path = unit_path(run_dir, recording_id, subject)
    if not path.is_dir():
        raise FileNotFoundError(f"no frame view at {path}")
    root = zarr_store.open_group(path, mode="r")
    return FrameView(recording_id=recording_id, subject=subject, root=root)


def _subject_sort_key(subject: int | None) -> tuple:
    """None, then every int in order — never compares a ``None`` against an int directly
    (mixing them in a bare comparison raises ``TypeError``)."""
    if subject is None:
        return (0, 0)
    return (1, subject)


def _in_unit_order(
    units: list[tuple[str, int | None]], recording_order: list[str]
) -> list[tuple[str, int | None]]:
    """Units grouped by ``recording_order``, then by subject within each recording.

    Split out from :func:`discover_frame_views` so the two orderings stay visibly separate:
    the *recording id* must go through ``sorted_by_id`` (it is a minted ``recording_NNN``,
    and a text sort breaks past the pad), while *subject* sorts by ``_subject_sort_key``. The
    guard over ``discover_*`` bodies bans a bare ``sorted(`` precisely so the first ordering
    cannot quietly become the second.
    """
    rank = {rid: i for i, rid in enumerate(recording_order)}
    return sorted(units, key=lambda u: (rank[u[0]], _subject_sort_key(u[1])))


def discover_frame_views(run_dir: Path | str) -> list[tuple[str, int | None]]:
    """Every ``(recording_id, subject)`` this run carries a frame view for, in id order.

    By scanning, like every other discovery in the artifact — no central list.

    Ordered through ``sorted_by_id`` on the recording id, not by text: a recording id is a
    minted ``recording_NNN``, so a plain sort puts ``recording_10`` before ``recording_9`` the
    moment an experiment outgrows the pad, and every listing built on this inherits it.

    Raises:
        FrameViewsPendingError: an edit left the run's views out of date (see :func:`set_pending`).
    """
    refuse_if_pending(run_dir)
    root = frames_dir(run_dir)
    if not root.is_dir():
        return []
    units = [parse_unit(p.stem) for p in root.iterdir() if p.suffix == ".zarr"]
    return _in_unit_order(units, sorted_by_id({rid for rid, _ in units}))


def frame_durations(t_ns) -> np.ndarray:
    """Per-frame duration in ns — the gap to the next frame, the median step for the last.

    The trailing frame has no successor to measure against, and assuming zero would silently
    shorten every window that ends at the recording's end.
    """
    axis = np.asarray(t_ns, dtype=np.int64)
    if axis.size == 0:
        return np.zeros(0, dtype=np.int64)
    if axis.size == 1:
        return np.zeros(1, dtype=np.int64)
    steps = np.diff(axis)
    # The trailing frame's stand-in is the subject's cadence, which is exactly what
    # `nominal_period_ns` computes — already imported above, rather than reimplemented here.
    return np.concatenate([steps, [nominal_period_ns(axis)]]).astype(np.int64)


def observed_ns_in(
    view: FrameView, start_t_ns: int | None = None, end_t_ns: int | None = None
) -> int:
    """Observed time inside ``[start_t_ns, end_t_ns]`` — §23's windowed denominator.

    The reason the frame view exists: ``coverage.observed_ns`` answers this for a whole
    recording and cannot answer it for a window, because it records how much was unobserved and
    not *where*. Frame-quantized — a frame counts toward the window it starts in.

    The window is therefore **half-open**, ``[start_t_ns, end_t_ns)``. Treating both ends as
    inclusive put the frame at a shared boundary in *both* neighbouring windows and credited it
    a full period to each, so the measure was not additive — two halves of a recording summed
    to more than the whole — and a window one period long reported twice its own length. This
    is §23's denominator, so that inflates it and deflates every rate divided by it.

    Both bounds default to the whole recording, which makes the no-window case the same code
    path rather than a special one.
    """
    axis, mask, durations = view.arrays()
    inside = mask
    if start_t_ns is not None:
        inside = inside & (axis >= int(start_t_ns))
    if end_t_ns is not None:
        inside = inside & (axis < int(end_t_ns))
    return int(durations[inside].sum())
