"""What every producer of the §19 behavior artifact writes besides its instances.

A module that emits labeled behavior in time owes two things beyond ``instances.csv``: a
**coverage row** per analyzed unit, and the **frame view** that says *where* the observed time
sits. Both are easy to get subtly wrong in the same two ways -- the span's end is the last row's
timestamp plus its period, and a tracking gap is unobserved while frame cadence is not -- so they
live here once rather than once per producer.

Query wrote both first and pose clustering wrote neither, which is exactly the shape this module
exists to stop: `output_manifest` advertises the frames artifact for *every* producer (it returns
``behavior_output_artifacts()``), so a producer that skips it ships a run page offering a thing
that is not there, and `analysis/measures.py` refuses every windowed measure on its runs while
answering the same question on its sibling's.

**The caller supplies the analyzed timestamps.** They are not derivable here, and not
interchangeable between producers: query's are an embedding trajectory's rows, while pose
clustering's come out of ``load_features``, which segments on a stricter gap and drops
``window - 1`` rows per run when a smoothing window is set. Each producer builds its coverage from
its own rows, so the frame view must be built from the same ones or the denominator disagrees with
the coverage it is supposed to refine.
"""

from pathlib import Path

import numpy as np

from detail.storage import experiment as experiment_store


def require_embedded_track(
    experiment_root, embed_run_ids: dict[str, str], track_id: int
) -> None:
    """Refuse a selection where some recording's own embedding run lacks ``track_id``.

    Per recording, because the runs are matched on settings and embedding settings do not
    include which tracks were embedded (§6.5), so two recordings' matching runs can cover
    different subjects.

    Raises:
        ValueError: A recording's run has no such track; the message names what it has.
    """
    from detail.api.run_preview import embedding_track_ids

    for recording_id, embed_run_id in embed_run_ids.items():
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        tracks = embedding_track_ids(recording_dir, embed_run_id)
        if track_id not in tracks:
            raise ValueError(
                f"recording {recording_id!r} has no track {track_id} in embedding "
                f"run {embed_run_id!r}; it has {sorted(tracks)}"
            )


def coverage_for(recording_id: str, subject: int, t_ns, gap_ns: int | None):
    """The §19 coverage row for one analyzed unit — the denominator every rate ships with.

    Two rules that are easy to get wrong: the span's ``end_t_ns`` is the last row's timestamp
    *plus its period* (a row covers its sampling interval, not just its instant), and
    ``observed_ns`` sums the continuous runs, so a tracking gap is subtracted while frame cadence
    is not.
    """
    from detail.analysis import clip_query as clip_core
    from detail.storage import behavior_output

    period = int(clip_core.nominal_period_ns(t_ns))
    gap = clip_core._gap_for(t_ns, gap_ns)
    observed_ns = sum(
        int(t_ns[end - 1]) + period - int(t_ns[start])
        for start, end in clip_core.segment_runs(t_ns, gap)
        if end > start
    )
    return behavior_output.CoverageRow(
        recording_id=recording_id,
        subject=subject,
        start_t_ns=int(t_ns[0]),
        end_t_ns=int(t_ns[-1]) + period,
        observed_ns=int(observed_ns),
        n_observed=int(len(t_ns)),
    )


class UnitFrames:
    """One analyzed unit's embedded rows as ``(t_ns, frame_idx)``, for placing an instance on the video.

    **An instance's frame numbers are looked up by its time, here, and nowhere else.** A method
    works in rows — positions in a sample or a trajectory — and a row position is a video frame only
    when no frame was ever dropped, strided or split across parts. Pose clustering wrote sample row
    numbers as frames and both temporal methods wrote trajectory positions, so clips came from the
    wrong place on every recording tracking had lost frames on. ``t_ns`` is what every method does
    carry correctly, so the frame is derived from it.
    """

    def __init__(self, t_ns, frame_idx) -> None:
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.frame_idx = np.asarray(frame_idx, dtype=np.int64)

    def frame_at(self, t_ns: int) -> int:
        """The video frame of the embedded row nearest ``t_ns``."""
        at = int(np.searchsorted(self.t_ns, int(t_ns)))
        if at >= len(self.t_ns) or (
            at > 0 and int(t_ns) - self.t_ns[at - 1] < self.t_ns[at] - int(t_ns)
        ):
            at -= 1
        return int(self.frame_idx[max(at, 0)])


def write_artifact(
    experiment_root,
    run_dir,
    *,
    instances,
    coverage,
    manifest: dict,
    analyzed_t_ns: dict,
    embed_runs: dict,
    gap_ns: int | None = None,
    progress=None,
    reuse: bool = False,
    example_ids: set | None = None,
) -> None:
    """Write the §19 artifact: the three files, the instance features, and the frame views.

    ``example_ids`` are the instances the user's own clips overlap (a query run's,
    ``query_runs.example_instance_ids``), written as ``instance_features.csv``'s ``is_example``;
    ``None`` for a producer that is given no clips, which writes no such column.

    **With ``reuse`` (an authored edit) no frame view is written.** A view is the one part of the
    artifact whose cost is the experiment's rather than the edit's — most of a branch
    name's commit, nearly all of it per-array zarr overhead that no tuning removes (#325).
    So an edit writes what the workspace reads, marks the views pending if any of them no longer
    matches, and :func:`bring_frame_views_current` rebuilds exactly those, once, for the first
    reader that needs them. Positions carry over by ``instance_id``.

    Without ``reuse`` (the compute's first write, a rebuild, an edit on a run left stale by a
    failed one) everything is written now and nothing on disk is trusted: rebuilding is the
    repair for a disagreement, so it must not trust what it is repairing.

    The pending mark goes down **before** ``instances.csv`` moves and comes up only when no view
    disagrees with it, so a kill anywhere in between leaves the views refusing to be read rather
    than silently stale.
    """
    from detail.storage import behavior_output
    from detail.storage import frame_view

    frame_view.set_pending(run_dir, True)
    behavior_output.write_behavior_output(
        run_dir, instances=instances, coverage=coverage, manifest=manifest
    )
    write_instance_features(
        experiment_root,
        run_dir,
        instances=instances,
        embed_runs=embed_runs,
        carry=reuse,
        example_ids=example_ids,
    )
    if reuse:
        if not _views_out_of_date(
            experiment_root,
            run_dir,
            instances=instances,
            coverage=coverage,
            gap_ns=gap_ns,
        ):
            frame_view.set_pending(run_dir, False)
        return
    write_frame_views(
        experiment_root,
        run_dir,
        coverage=coverage,
        instances=instances,
        analyzed_t_ns=analyzed_t_ns,
        gap_ns=gap_ns,
        progress=progress,
    )
    frame_view.set_pending(run_dir, False)


def bring_frame_views_current(
    experiment_root, run_dir, *, rows_for, gap_ns: int | None = None, progress=None
) -> int:
    """Rebuild the frame views an edit left out of date; return how many were rebuilt.

    Free when nothing is pending. Otherwise reads the artifact on disk — the intervals are
    authoritative (§19) — and rewrites only the units whose view was built from something else,
    so a session of ten names pays for the views once, and only for the recordings those names
    actually moved.

    ``rows_for(unit)`` returns that unit's analyzed ``t_ns``: the producer's own rows, which only
    the producer knows how to load (see this module's docstring). Called only for a unit being
    rebuilt.
    """
    from detail.storage import behavior_output
    from detail.storage import frame_view

    if not frame_view.is_pending(run_dir):
        return 0
    artifact = behavior_output.read_behavior_output(run_dir)
    by_unit = _by_unit(artifact.instances)
    stale = _views_out_of_date(
        experiment_root,
        run_dir,
        instances=artifact.instances,
        coverage=artifact.coverage,
        gap_ns=gap_ns,
    )
    for index, row in enumerate(stale):
        if progress is not None:
            progress("writing frame views", index / len(stale), row.recording_id)
        _write_unit_view(
            experiment_root,
            run_dir,
            row,
            by_unit.get(row.unit, []),
            rows_for(row.unit),
            gap_ns,
        )
    frame_view.set_pending(run_dir, False)
    return len(stale)


def _by_unit(instances) -> dict[tuple, list]:
    out: dict[tuple, list] = {}
    for row in instances:
        out.setdefault(row.unit, []).append(row)
    return out


def _views_out_of_date(
    experiment_root, run_dir, *, instances, coverage, gap_ns
) -> list:
    """The coverage rows whose unit's view was built from something other than ``instances``.

    A unit with no view is out of date unless it cannot have one — a recording with no video has
    no dense axis, and ``write_frame_views`` skips it for good.
    """
    from detail.storage import frame_view

    by_unit = _by_unit(instances)
    out = []
    for row in coverage:
        want = frame_view.built_from(by_unit.get(row.unit, []), row, gap_ns)
        have = frame_view.stored_built_from(run_dir, row.recording_id, row.subject)
        if have == want:
            continue
        if have is None and not _video_parts(experiment_root, row.recording_id)[2]:
            continue
        out.append(row)
    return out


def _video_parts(experiment_root, recording_id: str) -> tuple[Path, dict, list]:
    """``(recording_dir, recording.yaml, its video parts)`` — no parts means no dense axis."""
    from detail.storage import recording as recording_store

    recording_dir = experiment_store.recording_path(experiment_root, recording_id)
    data = recording_store.read_recording(recording_dir)
    return recording_dir, data, data.get("video", {}).get("files", [])


def write_instance_features(
    experiment_root,
    run_dir,
    *,
    instances,
    embed_runs: dict,
    carry: bool = False,
    example_ids: set | None = None,
) -> None:
    """§19's per-instance features — for now, where each instance happened.

    Here for the same reason ``write_frame_views`` is: a module that emits behavior in time owes
    the artifact more than its instances, and the pieces that are easy to get subtly wrong live
    once rather than once per producer. Position is one of those — it has to come from §14's
    ``kinematics/`` and be matched by time, and a producer that reconstructed it from ``det_row``
    instead would get a plausible wrong answer (that back-reference is part-relative and names a
    frame's representative detection, not the subject).

    ``embed_runs`` maps ``(recording_id, subject)`` to the embedding run its rows came from,
    keyed like ``analyzed_t_ns`` and for the same reason: it is the producer that knows, and a
    future producer with a per-recording embed run needs no new signature.

    **A run whose embedding predates ``kinematics/`` writes an empty file, not no file.** The
    difference matters to a reader: no file is "this artifact is older than features", an empty
    one is "features were computed and there was nothing to say", and only the first is fixable
    by ``recording backfill-kinematics``.

    With ``carry``, a row already on disk is kept for any instance whose ``instance_id`` it has —
    the id hashes the unit and the span, so the same id is the same stretch of the same subject —
    and only a recording holding a new instance has its kinematics read. A row with no position is
    computed again, so a unit backfilled since the last write fills in on its next edit, as it did
    when every edit recomputed every position.
    """
    from detail.analysis import positions as positions_lib
    from detail.api.run_preview import load_track_kinematics
    from detail.storage import behavior_output

    # Carrying needs a file to carry from. Without one the run either predates kinematics or has
    # never been written, and whether to write `{}` or nothing depends on whether *any* unit has
    # kinematics — which only reading them all can say, so that case takes the full path.
    previous = behavior_output.read_instance_features(run_dir) if carry else None
    carried: dict[str, tuple] = {}
    pending = []
    for row in behavior_output.bouts(list(instances)):
        position = (previous or {}).get(row.instance_id)
        if position is not None and None not in position:
            carried[row.instance_id] = position
        else:
            pending.append(row)
    wanted = None if previous is None else {row.unit for row in pending}

    kinematics: dict[tuple, dict] = {}
    for (recording_id, subject), embed_run_id in embed_runs.items():
        if embed_run_id is None:
            continue
        if wanted is not None and (recording_id, subject) not in wanted:
            continue
        recording_dir = experiment_store.recording_path(experiment_root, recording_id)
        found = load_track_kinematics(recording_dir, embed_run_id, int(subject))
        if found is not None:
            kinematics[(recording_id, subject)] = found
    if previous is None and not kinematics:
        return
    behavior_output.write_instance_features(
        run_dir,
        {**carried, **positions_lib.instance_positions(pending, kinematics)},
        example_ids=example_ids,
    )


def write_frame_views(
    experiment_root,
    run_dir,
    *,
    coverage,
    instances,
    analyzed_t_ns: dict,
    gap_ns: int | None = None,
    progress=None,
) -> None:
    """§19's per-frame expansion, one zarr per analyzed unit.

    Written by the producer rather than left to a consumer because ``coverage.observed_ns``
    records how much time was observed and not *where*: without this, a measure over any
    sub-window has no honest denominator, and ``analysis/measures.py`` refuses to compute one
    rather than substituting the whole-recording number.

    The axis is the recording's **dense** frame timeline, not the analyzed rows, because
    ``observed_mask`` is what separates the two reasons a row can be missing — a tracking gap is
    genuinely unobserved, frame cadence is not — and it needs the dense axis to tell them apart.
    Marking an ``every_n_frames=5`` run's unsampled frames unobserved would divide every rate by
    five.

    ``analyzed_t_ns`` maps ``(recording_id, subject)`` to the rows that unit's coverage was built
    from. ``gap_ns`` is the producer's continuity gap and is passed through to ``observed_mask``:
    resolving it here rather than letting the mask default is what keeps
    ``observed_ns_in(view) == coverage.observed_ns`` true for a producer that sets one. It held
    only for the default gap before, while the docstring claimed it held by construction.
    """
    by_unit = _by_unit(instances)
    for index, row in enumerate(coverage):
        if progress is not None:
            progress(
                "writing frame views",
                index / max(len(coverage), 1),
                row.recording_id,
            )
        rows = analyzed_t_ns.get(row.unit)
        if rows is None or len(rows) == 0:
            continue
        _write_unit_view(
            experiment_root, run_dir, row, by_unit.get(row.unit, []), rows, gap_ns
        )


def _write_unit_view(
    experiment_root, run_dir, coverage, instances, rows, gap_ns
) -> bool:
    """Write one unit's view, stamped with what it was built from.

    ``False`` when the unit cannot have one — no video, so no dense axis; the intervals stay
    authoritative regardless.
    """
    from detail.analysis import clip_query as clip_core
    from detail.storage import frame_view
    from detail.storage import recording as recording_store

    recording_dir, data, parts = _video_parts(experiment_root, coverage.recording_id)
    if not parts:
        return False
    # The cached twin: an axis is the same file on every rebuild of a session, and 48 uncached
    # reads were 2 of the 12 s a branch name took.
    axis = np.concatenate(
        [
            recording_store.load_active_timestamps_cached(recording_dir, data, i).values
            for i in range(len(parts))
        ]
    )
    frame_view.write_frame_view(
        Path(run_dir),
        recording_id=coverage.recording_id,
        subject=coverage.subject,
        t_ns=axis,
        observed=frame_view.observed_mask(
            axis, rows, max_gap_ns=clip_core._gap_for(rows, gap_ns)
        ),
        instances=instances,
        built_from=frame_view.built_from(instances, coverage, gap_ns),
    )
    return True
