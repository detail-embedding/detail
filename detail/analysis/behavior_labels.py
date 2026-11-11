"""Ground-truth behavior labels as a per-frame integer sequence (shared primitive).

This is the scoring-neutral half of what ``pipeline/query_eval.py`` does inline: read a recording's
``behavior``-role stream and turn its labeled intervals into a per-frame label array. It is lifted
here — rather than reused from query_eval — so the clustering eval and a future auto-namer share one
interval→frame reader without importing query_eval's fold-splitting, IoU/point scoring, or
``iou_threshold``/``point_window_s`` semantics, which are specific to per-clip retrieval AP.

The one generalization over query_eval's ``_interval_mask``: that returns a boolean mask for *one*
behavior; ``frames_to_labels`` returns a *multi-class* integer label per frame (behavior id, or
``unlabeled`` for a frame in no interval), which is what a clustering metric scores against.

Single-label per frame: if intervals overlap at a frame (co-occurring behaviors), the first in
stream order wins — deterministic, and multi-label scoring is out of Phase-1 scope.
"""

from collections import namedtuple

import numpy as np

from detail.storage import streams as streams_store

# Per-frame ground truth over a whole recording timeline.
FrameLabels = namedtuple("FrameLabels", ["labels", "id_to_label", "t_ns"])


class AmbiguousBehaviorStreamError(ValueError):
    """A recording carries several behavior streams and the caller named none of them."""


class PredictedBehaviorStreamError(ValueError):
    """A method's own output was named where ground truth was required."""


def ground_truth_streams(recording) -> list[dict]:
    """The recording's ``behavior``-role streams that hold a human's scoring (§19).

    A prediction re-ingested as a stream is structurally identical to an annotation — same
    role, same shape, same importer — so every surface that offers "the ground truth" has to
    filter on ``origin`` or it will offer a method its own output back.
    """
    return [
        entry
        for entry in recording.streams_by_role("behavior")
        if not streams_store.is_predicted(entry)
    ]


def resolve_behavior_stream(
    recording, stream_name: str | None, *, allow_predicted: bool = False
) -> str | None:
    """The behavior stream to read, resolved once, or ``None`` if the recording carries none.

    The single place the implicit choice is made. It used to be four copies of
    ``streams_by_role("behavior")[0]`` — in this module, in ``query_eval``, and twice in
    ``api/recording`` — and ``recording_frame_labels`` reached two of them in one call.

    That fallback is ordered by insertion, so **adding** a stream to a recording changes what
    every unnamed reader scores. It is not hypothetical: adding ``climb_rear`` (2 labels) beside
    ``full_behavior`` (7) silently moved `clusterer diagnose` (since removed) onto a different
    vocabulary than the eval it was meant to explain, and nothing anywhere errored.

    So with several streams and no name, this raises rather than picking. One stream is
    unambiguous and resolves silently; none returns ``None`` and the caller degrades to the
    label-free path, which is a normal state for an unannotated recording.

    **Predicted streams are excluded** (§19). A clustering run's ethogram re-ingests through the
    same importer as a hand-scored CSV and is otherwise indistinguishable from one, so
    without this a method could be scored — or named — against its own output, and nothing
    would fail. Naming one explicitly raises rather than quietly scoring it; leaving it
    unnamed simply resolves past it. ``allow_predicted`` is for a caller that genuinely wants
    a prediction's intervals (rendering them, comparing them against truth) rather than
    treating them as truth.

    Raises:
        AmbiguousBehaviorStreamError: several ground-truth streams, none named.
        PredictedBehaviorStreamError: the named stream holds a method's output.
    """
    if stream_name is not None:
        if not allow_predicted:
            entry = next(
                (
                    s
                    for s in recording.streams_by_role("behavior")
                    if s.get("name") == stream_name
                ),
                None,
            )
            if entry is not None and streams_store.is_predicted(entry):
                raise PredictedBehaviorStreamError(
                    f"stream {stream_name!r} on recording {recording.id!r} holds predicted "
                    "labels (origin: predicted), not ground truth. Scoring or naming a "
                    "method against its own output measures nothing — a label may score a "
                    "method, never build one."
                )
        return stream_name
    streams = (
        recording.streams_by_role("behavior")
        if allow_predicted
        else ground_truth_streams(recording)
    )
    if not streams:
        return None
    if len(streams) > 1:
        names = [str(s.get("name")) for s in streams]
        raise AmbiguousBehaviorStreamError(
            f"recording {recording.id!r} carries {len(names)} behavior streams "
            f"({', '.join(sorted(names))}) and none was named. Pass the stream explicitly — "
            "which one is read would otherwise depend on the order they were added, so the "
            "same command can score a different ground truth tomorrow."
        )
    return str(streams[0]["name"])


def recording_behavior_intervals(
    recording,
    *,
    behaviors=None,
    stream_name: str | None = None,
    allow_predicted: bool = False,
) -> list[tuple[int, int, str]]:
    """The recording's ``(start_t_ns, end_t_ns, label)`` behavior intervals (Section 3).

    ``stream_name`` picks the ``behavior``-role stream (video-aligned ``t_ns``); omitted, it is
    resolved by ``resolve_behavior_stream``, which raises rather than guess when the recording
    carries more than one. ``behaviors`` optionally restricts to a subset. Empty if the
    recording carries no such stream. Scoring-neutral — the generalization of query_eval's
    per-behavior ``_behavior_intervals`` to all behaviors.

    Predicted streams are excluded unless ``allow_predicted`` (§19) — see
    ``resolve_behavior_stream``.
    """
    name = resolve_behavior_stream(
        recording, stream_name, allow_predicted=allow_predicted
    )
    if name is None:
        return []
    rows = recording.stream_events(name)
    intervals = [(int(r[0]), int(r[1]), str(r[2])) for r in rows]
    if behaviors is not None:
        allowed = set(behaviors)
        intervals = [iv for iv in intervals if iv[2] in allowed]
    return intervals


def frames_to_labels(
    t_ns,
    intervals: list[tuple[int, int, str]],
    *,
    behavior_types: dict[str, str] | None = None,
    point_window_ns: int = 0,
    unlabeled: int = -1,
    label_to_id: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Per-frame integer behavior label for frame times ``t_ns`` (multi-class).

    A ``state`` behavior labels a frame whose time lies in ``[start, end]``; a ``point`` behavior
    labels a frame within ``±point_window_ns`` of the interval start (mirroring query_eval's point
    mode). Frames in no interval get ``unlabeled``. ``behavior_types`` maps a label to
    ``"state"``/``"point"`` (default: all ``state``). Pass ``label_to_id`` to fix the behavior→id
    map across recordings; otherwise ids are assigned to the sorted labels present.

    Returns:
        The ``int64`` per-frame label array and the ``id → label`` map for reporting.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    if label_to_id is None:
        present = sorted({label for _s, _e, label in intervals})
        label_to_id = {label: i for i, label in enumerate(present)}
    types = behavior_types or {}
    out = np.full(t.size, unlabeled, dtype=np.int64)
    for start, end, label in intervals:
        if label not in label_to_id:
            continue
        if types.get(label, "state") == "point":
            hit = np.abs(t - start) <= point_window_ns
        else:
            hit = (t >= start) & (t <= end)
        fill = hit & (out == unlabeled)  # first interval in stream order wins
        out[fill] = label_to_id[label]
    id_to_label = {i: label for label, i in label_to_id.items()}
    return out, id_to_label


def recording_frame_labels(
    recording,
    *,
    behaviors=None,
    point_window_ns: int = 0,
    unlabeled: int = -1,
    label_to_id: dict[str, int] | None = None,
    stream_name: str | None = None,
) -> FrameLabels:
    """Per-frame ground-truth behavior label over a recording's whole video timeline.

    Ties ``frame_timestamps`` + ``recording_behavior_intervals`` + ``behavior_types`` together — the
    accessor the clustering eval scores against. ``labels`` is ``unlabeled`` on every frame the
    annotation did not cover, so a caller scores only ``labels != unlabeled``.

    ``stream_name`` selects the ground-truth stream (default: the recording's first). The
    intervals and the point/state types must come from the **same** stream, or a point behavior
    in one is scored as a state from the other.
    """
    t_ns = recording.frame_timestamps()
    # Resolved once and reused: the intervals and the point/state types have to come from the
    # *same* stream, and resolving twice is two chances to get different answers.
    name = resolve_behavior_stream(recording, stream_name)
    intervals = recording_behavior_intervals(
        recording, behaviors=behaviors, stream_name=name
    )
    labels, id_to_label = frames_to_labels(
        t_ns,
        intervals,
        behavior_types=recording.behavior_types(name),
        point_window_ns=point_window_ns,
        unlabeled=unlabeled,
        label_to_id=label_to_id,
    )
    return FrameLabels(labels=labels, id_to_label=id_to_label, t_ns=t_ns)
