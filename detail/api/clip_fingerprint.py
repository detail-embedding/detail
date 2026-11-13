"""What sets one cluster apart, for the clip window: its few most distinctive measurements, the
cluster's spread and the recordings' spread on each, and where the clip on screen falls.

**The cluster page's measurements, at the window's size.** The same descriptors
(``analysis.descriptors``) and the same ranking (``descriptor_ranking.rank_selection``, Cliff's
delta against the rest) as the Interpretation band, so a measurement the window calls
distinctive is one the page would too. What differs is how much is read: the band measures every
bout of the run, ~70 s on a real run, behind a button; a window opened to label one cluster
cannot wait that long. So this reads **the cluster's most typical members** (``max_members``,
from at most ``max_units`` animal-recordings, most typical first) and compares them with
windows of the members' own lengths spread across **the same animals' whole recordings** — the
band's ``BASELINE_RECORDING`` comparison, drawn from fewer recordings. The figure says both
numbers, so it is never mistaken for the band's.

Off the UI thread, and it reports: one track read per animal-recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detail.core.utils import subject_key

#: The comparison windows drawn across the animals' recordings, in total.
BASELINE_WINDOWS = 200
#: The label a member's row carries; a comparison window's is ``None`` (rank_selection's rule).
_THIS = "this"
#: The shortest span a member is measured over, centred on it. A pose cluster's member is one
#: frame, which has no speed or change; half a second around it is the movement the posture sits
#: in, and the window says the figures are over at least this much.
MIN_SPAN_NS = 500_000_000


def widened(start_ns: int, end_ns: int) -> tuple[int, int]:
    """A member's span, grown about its middle to at least :data:`MIN_SPAN_NS`."""
    start_ns, end_ns = int(start_ns), int(end_ns)
    short = MIN_SPAN_NS - (end_ns - start_ns)
    if short <= 0:
        return start_ns, end_ns
    return start_ns - short // 2, end_ns + (short - short // 2)


@dataclass(frozen=True)
class FingerprintRow:
    """One distinctive measurement: its values over the cluster's members and over the
    comparison windows, the clip's own value (``None`` when it could not be read), and how far
    the cluster sits from the rest (``delta``, Cliff's, -1..1)."""

    name: str
    label: str
    unit: str
    cluster: np.ndarray
    baseline: np.ndarray
    clip: float | None
    delta: float
    percentile: float


@dataclass(frozen=True)
class Fingerprint:
    rows: list[FingerprintRow]
    n_members: int
    n_units: int
    n_baseline: int


def cluster_fingerprint(
    experiment,
    run,
    node: int,
    clip=None,
    *,
    max_members: int = 40,
    max_units: int = 10,
    top: int = 3,
    progress=None,
    tracks: dict | None = None,
) -> Fingerprint:
    """The ``top`` measurements that most set ``node`` apart (§22), and ``clip``'s values on
    them. ``clip`` is ``(recording_id, subject, start_ns, end_ns)`` or ``None``.
    ``progress(stage, fraction, detail)`` hears each animal-recording read. ``tracks`` (a dict
    the caller keeps) holds each track read, so :func:`clip_measures` for the next clip reads
    nothing again.

    Raises nothing for an unreadable track: it is left out, and the counts say what was read.
    """
    from detail.analysis.descriptors import DESCRIPTORS, available, describe
    from detail.analysis.descriptor_ranking import rank_selection
    from detail.api.frame_slices import between
    from detail.core import run_selection

    members = []
    units: list[tuple[str, int]] = []
    # Every clip of the node, most typical first (``NodeView.clips``); a method ranked on
    # demand loads its fit here, on the caller's worker.
    for member in run.typical_members(int(node)):
        unit = (str(member["recording_id"]), subject_key(member["subject"]))
        if unit not in units:
            if len(units) >= max_units:
                continue
            units.append(unit)
        members.append(member)
        if len(members) >= max_members:
            break
    clip_unit = None if clip is None else (str(clip[0]), subject_key(clip[1]))
    if clip_unit is not None and clip_unit not in units:
        units.append(clip_unit)
    embed_runs = run_selection.runs_by_recording(run.state(), {u[0] for u in units})
    spans = [widened(m["start_t_ns"], m["end_t_ns"]) for m in members]
    lengths = np.array(
        [end - start for start, end in spans] or [MIN_SPAN_NS], dtype=np.int64
    )
    per_unit = max(BASELINE_WINDOWS // max(len(units), 1), 1)

    values: dict[str, list[float]] = {}
    labels: list = []
    clip_values: dict[str, float] = {}
    names: list[str] | None = None
    n_members = n_baseline = n_units = 0

    def add(bout, label) -> None:
        nonlocal names
        if names is None:
            names = [
                n for n in available(bout) if not DESCRIPTORS[n].measures_span
            ] or []
        described = describe(bout, names=names)
        for name in names:
            values.setdefault(name, []).append(described.get(name, np.nan))
        labels.append(label)

    for index, unit in enumerate(units):
        if progress is not None:
            progress(
                "measuring",
                index / max(len(units), 1),
                f"{unit[0]} · {index + 1} of {len(units)}",
            )
        trajectory = _cached(tracks, experiment, unit, embed_runs.get(unit[0]))
        if trajectory is None or not len(trajectory.t_ns):
            continue
        n_units += 1
        for member, (start, end) in zip(members, spans):
            if (str(member["recording_id"]), subject_key(member["subject"])) == unit:
                add(between(trajectory, start, end), _THIS)
                n_members += 1
        for start, end in _windows(trajectory.t_ns, lengths, per_unit, seed=index):
            add(between(trajectory, start, end), None)
            n_baseline += 1
        if clip is not None and unit == clip_unit and names:
            clip_values = describe(
                between(trajectory, *widened(clip[2], clip[3])), names=names
            )
    if progress is not None:
        progress("measuring", 1.0, f"{n_units} recordings read")
    labels_arr = np.asarray(labels, dtype=object)
    ranked = rank_selection(values, labels, _THIS)[:top]
    rows = []
    for item in ranked:
        column = np.asarray(values[item.name], dtype=np.float64)
        mine = np.array([lab == _THIS for lab in labels_arr.tolist()], dtype=bool)
        clip_value = clip_values.get(item.name)
        rows.append(
            FingerprintRow(
                name=item.name,
                label=DESCRIPTORS[item.name].label,
                unit=DESCRIPTORS[item.name].unit,
                cluster=column[mine],
                baseline=column[~mine],
                clip=(
                    None
                    if clip_value is None or not np.isfinite(clip_value)
                    else float(clip_value)
                ),
                delta=float(item.delta),
                percentile=float(item.percentile),
            )
        )
    return Fingerprint(rows, n_members, n_units, n_baseline)


def clip_measures(
    experiment, run, clip, names: list[str], tracks: dict | None = None
) -> dict[str, float]:
    """One clip's values on ``names`` (``(recording_id, subject, start_ns, end_ns)``), over
    the same widened span the fingerprint measured its members on; ``{}`` when its track cannot
    be read."""
    from detail.analysis.descriptors import describe
    from detail.api.frame_slices import between
    from detail.core import run_selection

    unit = (str(clip[0]), int(clip[1]))
    embed_run_id = run_selection.runs_by_recording(run.state(), {unit[0]}).get(unit[0])
    trajectory = _cached(tracks, experiment, unit, embed_run_id)
    if trajectory is None or not len(trajectory.t_ns) or not names:
        return {}
    return describe(between(trajectory, *widened(clip[2], clip[3])), names=list(names))


def _cached(tracks: dict | None, experiment, unit, embed_run_id):
    if tracks is None:
        return _trajectory(experiment, unit, embed_run_id)
    if unit not in tracks:
        tracks[unit] = _trajectory(experiment, unit, embed_run_id)
    return tracks[unit]


def _trajectory(experiment, unit, embed_run_id):
    """One animal's whole track in this recording, or ``None`` if it cannot be read."""
    from detail.api.frame_slices import track_slice

    recording_id, subject = unit
    try:
        recording = experiment.get_recording(recording_id)
        record = next(
            (e for e in recording.embedding_runs if e.get("id") == embed_run_id), {}
        )
        # The kinematics sidecar when the run wrote one, so the mask's speed and area are
        # measured here as they are on the cluster profile; the boxes otherwise.
        return track_slice(recording, {"id": embed_run_id, **record}, subject)
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return None


def _windows(t_ns, lengths: np.ndarray, n: int, *, seed: int) -> list[tuple[int, int]]:
    """``n`` windows spread evenly across a track, each a member's length — the measurements
    band's ``_baseline_spans`` rule (a second copy, kept beside its reader: the band's lives in
    the GUI package, which this one may not import). Seeded, so the same window opens on the
    same picture."""
    times = np.asarray(t_ns, dtype=np.int64)
    if len(times) < 2 or n <= 0:
        return []
    first, last = int(times[0]), int(times[-1])
    drawn = np.random.default_rng(seed).choice(lengths, size=n)
    out = []
    for start, length in zip(np.linspace(first, last, n), drawn):
        end = min(int(start) + int(length), last)
        if end > int(start):
            out.append((int(start), end))
    return out
