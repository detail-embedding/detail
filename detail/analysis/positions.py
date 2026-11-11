"""Where each behavior instance happened — one position per bout, from its frames (§19).

The aggregation half of the spatial check. §14's ``kinematics/`` says where the subject was on
every embedded frame; this reduces that to one point per instance, which is what a map of "where
does this behavior happen" plots.

**The mean over the bout, not its first frame.** A bout is an extent, and the question the map
answers is *where does this happen*, not *where does it start* — a rear that begins mid-arena and
ends against a wall is one event in one place. The per-frame path stays in the kinematics file for
anything that wants it.

Pure over arrays: numpy only, no zarr, no paths, no Qt.
"""

import numpy as np

#: What an instance with no usable frames reports. NaN rather than a coordinate, for the reason
#: `analysis/geometry.py` gives: zero is the top-left corner, and a plotted zero is a claim.
NO_POSITION = (float("nan"), float("nan"))


def instance_positions(instances, kinematics_by_unit: dict) -> dict:
    """``{instance_id: (cx_norm, cy_norm)}`` — each instance's mean position.

    ``instances`` are §19 ``BehaviorInstance`` rows; ``kinematics_by_unit`` maps
    ``(recording_id, subject)`` to that unit's kinematics columns (``t_ns``, ``cx_norm``,
    ``cy_norm``), as :func:`detail.api.run_preview.load_track_kinematics` returns them.

    Spans are matched by **time**, inclusive at both ends, which is what §19 means by a span and
    is the only key the two files share — the artifact stores no row indices, and deriving them
    would re-invent a join the timestamps already make exact.

    Looked up with ``searchsorted`` rather than a mask per instance: a boolean mask over the whole
    trajectory costs O(instances x rows), which on a real run is thousands of bouts against
    hundreds of thousands of rows — seconds of work for an answer that is a binary search.
    """
    out: dict[str, tuple[float, float]] = {}
    prepared = {
        unit: _sorted_columns(cols) for unit, cols in kinematics_by_unit.items()
    }
    for row in instances:
        unit = (row.recording_id, row.subject)
        columns = prepared.get(unit)
        if columns is None:
            out[row.instance_id] = NO_POSITION
            continue
        t_ns, cx, cy = columns
        lo = int(np.searchsorted(t_ns, int(row.t_ns), side="left"))
        hi = int(np.searchsorted(t_ns, int(row.end_t_ns), side="right"))
        out[row.instance_id] = _mean_position(cx[lo:hi], cy[lo:hi])
    return out


def _sorted_columns(columns: dict):
    """``(t_ns, cx, cy)`` in ascending time — ``searchsorted`` is wrong on unsorted input.

    The kinematics file is written in frame order, which is time order, so this is normally a
    no-op. It is not assumed: an unsorted array makes ``searchsorted`` return a silently wrong
    span rather than raising, and every position built on it would be a plausible number.
    """
    t_ns = np.asarray(columns["t_ns"], dtype=np.int64)
    cx = np.asarray(columns["cx_norm"], dtype=np.float64)
    cy = np.asarray(columns["cy_norm"], dtype=np.float64)
    if t_ns.size and not np.all(np.diff(t_ns) >= 0):
        order = np.argsort(t_ns, kind="stable")
        t_ns, cx, cy = t_ns[order], cx[order], cy[order]
    return t_ns, cx, cy


def _mean_position(cx: np.ndarray, cy: np.ndarray):
    """Mean of the finite rows, or ``NO_POSITION`` when the bout has none.

    ``nanmean`` of an all-NaN slice warns and returns NaN; the suite runs with warnings as
    errors, and "this bout had no describable frame" is a state to return rather than a warning
    to emit.
    """
    finite = np.isfinite(cx) & np.isfinite(cy)
    if not finite.any():
        return NO_POSITION
    return (float(cx[finite].mean()), float(cy[finite].mean()))
