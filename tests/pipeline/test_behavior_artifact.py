"""An authored edit writes the intervals and defers the frame views — and a stale view is never read.

The class, in both directions:

- **An edit whose cost is the experiment's rather than its own** (#325). Naming one cluster
  reprojected every ``(recording, subject)`` frame view — most of the commit's time.
  Diffing fixed the name at the cut, where no bout moves; it could not fix a name on a *branch*,
  which fuses and relabels the bouts beneath it in nearly every recording, and whose views are
  per-array zarr overhead that no tuning removes. So an edit writes no view at all: each
  view records what it was built from, and the first reader rebuilds exactly the ones that differ.
- **Deferral done by trust instead of by content.** Deciding "this edit cannot have moved a
  boundary" from *which operation ran* is false one level up, and keeping a view that disagrees
  with ``instances.csv`` fails nowhere — every measure over it stays a plausible number. So a view
  is current only when its own digest says so, and while any view might not be, the run is marked
  pending and **every reader refuses** rather than reading it. The mark goes down before
  ``instances.csv`` moves, so a kill mid-edit leaves a refusal, never a silent stale view.

Every test compares with a **full** write of the same inputs, and a store counts as rewritten when
its directory is a different one (the writer builds beside the old store and swaps it in, so a
rewrite always yields a new inode while the old one still exists).
"""

import filecmp
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from detail.pipeline import behavior_artifact
from detail.storage import behavior_output as bo
from detail.storage import frame_view
from detail.storage import recording as recording_store
from detail.storage import snapshots
from detail.storage import zarr_store

STEP = 10_000_000
N_ROWS = 100
RECORDINGS = ("recording_001", "recording_002")
UNITS = [(recording_id, 0) for recording_id in RECORDINGS]


@pytest.fixture
def cohort(experiment_dir, recording_dir, make_embedding_run):
    """Two recordings, each with a dense video axis and an embedding run carrying kinematics.

    Two, because a single unit cannot tell "rebuilt the changed one" from "rebuilt everything".
    Position drifts across each recording, so a carried position that belonged to another span
    would come out a different number rather than passing.
    """
    second = experiment_dir / "recordings" / RECORDINGS[1]
    shutil.copytree(recording_dir, second)
    record = yaml.safe_load((second / "recording.yaml").read_text(encoding="utf-8-sig"))
    record.update(id=RECORDINGS[1], name=RECORDINGS[1])
    (second / "recording.yaml").write_text(
        yaml.dump(record, sort_keys=False), encoding="utf-8"
    )
    t_ns = _rows()
    for recording_id in RECORDINGS:
        directory = experiment_dir / "recordings" / recording_id
        recording_store.write_timestamps_csv(
            directory / "video/timestamps/generated/part_001.csv", t_ns
        )
        make_embedding_run(
            directory,
            np.zeros((N_ROWS, 4), dtype=np.float32),
            np.arange(N_ROWS, dtype=np.int64),
            t_ns,
            cx=np.linspace(0.0, 1.0, N_ROWS),
        )
    return experiment_dir


def _rows(n=N_ROWS):
    return np.arange(n, dtype=np.int64) * STEP


def _bout(recording_id, source_label, start, end):
    return bo.BehaviorInstance(
        recording_id=recording_id,
        subject=0,
        t_ns=start * STEP,
        end_t_ns=end * STEP,
        start_frame=start,
        end_frame=end,
        source_label=source_label,
    )


def _bouts(*, fused_on=()):
    """Two bouts per recording; a recording in ``fused_on`` has them joined under one label —
    the shape naming a branch above the cut gives."""
    out = []
    for recording_id in RECORDINGS:
        if recording_id in fused_on:
            out.append(_bout(recording_id, "9", 10, 60))
        else:
            out += [_bout(recording_id, "3", 10, 30), _bout(recording_id, "4", 31, 60)]
    return out


def _manifest(names: dict) -> dict:
    return {
        "producer": {"module_id": "temporal_clustering", "run_id": "an_01"},
        "origin": "predicted",
        "label_space": "pooled",
        "overlap": "exclusive",
        "vocabulary": sorted({n for labels in names.values() for n in labels}),
        "names": names,
        "behavior_types": {n: "state" for labels in names.values() for n in labels},
    }


def _write(experiment_dir, run_dir, instances, *, names=None, reuse, rows=None):
    rows = rows or {}
    analyzed = {unit: rows.get(unit, _rows()) for unit in UNITS}
    behavior_artifact.write_artifact(
        experiment_dir,
        run_dir,
        instances=instances,
        coverage=[
            behavior_artifact.coverage_for(*unit, t_ns, None)
            for unit, t_ns in analyzed.items()
        ],
        manifest=_manifest(names or {}),
        analyzed_t_ns=analyzed,
        embed_runs={unit: "emb_01" for unit in UNITS},
        reuse=reuse,
    )


def _bring_current(experiment_dir, run_dir, rows=None):
    rows = rows or {}
    return behavior_artifact.bring_frame_views_current(
        experiment_dir, run_dir, rows_for=lambda unit: rows.get(unit, _rows())
    )


def _identities(run_dir) -> dict:
    return {
        unit: os.stat(frame_view.unit_path(run_dir, *unit)).st_ino for unit in UNITS
    }


def _rewritten(run_dir, before: dict) -> set:
    after = _identities(run_dir)
    return {unit for unit in UNITS if after[unit] != before[unit]}


def _assert_same_as_a_full_write(experiment_dir, run_dir, instances, **kw):
    """The deferred artifact, once current, must be the artifact: identical files and arrays."""
    full = Path(run_dir).parent / "full"
    _write(experiment_dir, full, instances, reuse=False, **kw)
    for name in ("instances.csv", "coverage.csv", "instance_features.csv"):
        assert filecmp.cmp(run_dir / name, full / name, shallow=False), name
    for unit in UNITS:
        kept = zarr_store.open_group(frame_view.unit_path(run_dir, *unit), mode="r")
        fresh = zarr_store.open_group(frame_view.unit_path(full, *unit), mode="r")
        assert sorted(kept.array_keys()) == sorted(fresh.array_keys()), unit
        for key in fresh.array_keys():
            assert np.array_equal(kept[key][:], fresh[key][:]), (unit, key)


def test_naming_a_cluster_leaves_nothing_to_rebuild(cohort):
    """A name at the cut moves no bout, so no view is out of date and no reader pays."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = _identities(run_dir)

    names = {"3": ["rearing"]}
    _write(cohort, run_dir, _bouts(), names=names, reuse=True)

    assert not frame_view.is_pending(run_dir)
    assert _rewritten(run_dir, before) == set()
    labelled = {r.label for r in bo.read_behavior_output(run_dir).instances}
    assert "rearing" in labelled, "the name must still reach instances.csv"
    _assert_same_as_a_full_write(cohort, run_dir, _bouts(), names=names)


def test_an_edit_that_moves_bouts_writes_no_view_and_refuses_readers(cohort):
    """Naming a branch fuses and relabels bouts. The edit must not pay for the views — and must
    not let anyone read the old ones either."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = _identities(run_dir)

    changed = _bouts(fused_on={RECORDINGS[1]})
    _write(cohort, run_dir, changed, names={"9": ["moving"]}, reuse=True)

    assert _rewritten(run_dir, before) == set(), "an edit wrote a frame view"
    assert frame_view.is_pending(run_dir)
    with pytest.raises(frame_view.FrameViewsPendingError):
        frame_view.read_frame_view(run_dir, *UNITS[0])
    with pytest.raises(frame_view.FrameViewsPendingError):
        frame_view.discover_frame_views(run_dir)


def test_bringing_views_current_rebuilds_only_the_units_an_edit_moved(cohort):
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = _identities(run_dir)

    changed = _bouts(fused_on={RECORDINGS[1]})
    _write(cohort, run_dir, changed, names={"9": ["moving"]}, reuse=True)
    assert _bring_current(cohort, run_dir) == 1

    assert _rewritten(run_dir, before) == {UNITS[1]}
    assert not frame_view.is_pending(run_dir)
    assert _bring_current(cohort, run_dir) == 0, "a second call must be free"
    _assert_same_as_a_full_write(cohort, run_dir, changed, names={"9": ["moving"]})


def test_several_edits_pay_for_the_views_once(cohort):
    """The point of deferring: a naming session rebuilds at the end, not per name."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = _identities(run_dir)

    _write(cohort, run_dir, _bouts(fused_on={RECORDINGS[0]}), reuse=True)
    _write(cohort, run_dir, _bouts(fused_on=set(RECORDINGS)), reuse=True)
    assert _rewritten(run_dir, before) == set()

    assert _bring_current(cohort, run_dir) == 2
    _assert_same_as_a_full_write(cohort, run_dir, _bouts(fused_on=set(RECORDINGS)))


def test_an_edit_that_moves_bouts_back_needs_no_rebuild(cohort):
    """Current is a fact about each view, not about the edit history: undoing a change before
    anyone read the views leaves them exactly right, and the mark comes back up."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)

    _write(cohort, run_dir, _bouts(fused_on={RECORDINGS[1]}), reuse=True)
    assert frame_view.is_pending(run_dir)
    _write(cohort, run_dir, _bouts(), reuse=True)

    assert not frame_view.is_pending(run_dir)


def test_a_unit_whose_coverage_changed_is_rebuilt(cohort):
    """Same bouts, different analyzed rows: the ``observed`` mask moves, so the view does."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = _identities(run_dir)

    rows = {UNITS[0]: _rows(N_ROWS - 20)}
    _write(cohort, run_dir, _bouts(), reuse=True, rows=rows)
    assert _bring_current(cohort, run_dir, rows) == 1

    assert _rewritten(run_dir, before) == {UNITS[0]}
    _assert_same_as_a_full_write(cohort, run_dir, _bouts(), rows=rows)


def test_a_view_that_does_not_say_what_it_was_built_from_is_not_trusted(cohort):
    """Views written before this change carry no digest; the first edit's reader rebuilds them
    rather than assuming the old code wrote what the new one would."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    for unit in UNITS:
        group = zarr_store.open_group(frame_view.unit_path(run_dir, *unit), mode="r+")
        del group.attrs[frame_view.BUILT_FROM_ATTR]

    _write(cohort, run_dir, _bouts(), names={"3": ["rearing"]}, reuse=True)

    assert frame_view.is_pending(run_dir)
    assert _bring_current(cohort, run_dir) == 2


def test_an_edit_that_dies_leaves_the_views_refusing(cohort, monkeypatch):
    """The mark goes down before ``instances.csv`` moves. A kill between the two writes is what
    an exception here stands in for — nothing catches either."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)

    real = bo.write_behavior_output

    def die(*a, **k):
        real(*a, **k)  # instances.csv has moved — the window the ordering exists for
        raise RuntimeError("killed after instances.csv")

    monkeypatch.setattr(bo, "write_behavior_output", die)
    with pytest.raises(RuntimeError):
        _write(cohort, run_dir, _bouts(fused_on={RECORDINGS[1]}), reuse=True)

    assert frame_view.is_pending(run_dir)
    with pytest.raises(frame_view.FrameViewsPendingError):
        frame_view.read_frame_view(run_dir, *UNITS[1])


def test_a_snapshot_refuses_to_freeze_pending_views(cohort):
    """A snapshot copies the views and is never rewritten, so a stale copy would be stale for
    good. Pinning through the run brings them current first; pinning past it refuses."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    _write(cohort, run_dir, _bouts(fused_on={RECORDINGS[1]}), reuse=True)

    with pytest.raises(frame_view.FrameViewsPendingError):
        snapshots.pin(run_dir, reason="finish")


def test_a_rebuild_trusts_nothing_on_disk(cohort):
    """``reuse=False`` is the repair path; a repair that kept what it repairs would be none."""
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = _identities(run_dir)

    _write(cohort, run_dir, _bouts(), reuse=False)

    assert _rewritten(run_dir, before) == set(UNITS)
    assert not frame_view.is_pending(run_dir)


def test_a_kept_instance_carries_its_position_without_rereading_kinematics(cohort):
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, _bouts(), reuse=False)
    before = bo.read_instance_features(run_dir)
    assert before and all(None not in p for p in before.values())

    # With the kinematics gone, a re-read would come back empty and write no positions.
    shutil.rmtree(_kinematics(cohort, RECORDINGS[0]))
    _write(cohort, run_dir, _bouts(), names={"3": ["rearing"]}, reuse=True)

    assert bo.read_instance_features(run_dir) == before


def test_a_kept_unit_fills_in_positions_it_did_not_have(cohort):
    """Every edit used to recompute every position, so a unit backfilled after the run was made
    filled in on the next edit. Carrying rows over must not turn that hole permanent."""
    run_dir = cohort / "analysis" / "an_01"
    kinematics = _kinematics(cohort, RECORDINGS[0])
    aside = kinematics.with_name("kinematics_aside")
    kinematics.rename(aside)
    _write(cohort, run_dir, _bouts(), reuse=False)
    first = bo.read_instance_features(run_dir)
    assert any(None in p for p in first.values()), "the fixture must start with a hole"

    aside.rename(kinematics)  # the backfill
    _write(cohort, run_dir, _bouts(), reuse=True)

    assert all(None not in p for p in bo.read_instance_features(run_dir).values())
    _assert_same_as_a_full_write(cohort, run_dir, _bouts())


def _kinematics(experiment_dir, recording_id) -> Path:
    return (
        experiment_dir
        / "recordings"
        / recording_id
        / "embeddings"
        / "emb_01"
        / "track_0"
        / "kinematics"
    )


def test_an_edit_with_nothing_to_place_still_says_it_looked(cohort):
    """Absent and empty are different answers (§19): no file means "predates kinematics", which
    is fixable; an empty one means "measured, nothing to place". Reading kinematics only for new
    instances reads none when there are no instances, and wrote no file for a freshly fitted run
    whose granularity was not yet chosen — the actionable state, reported for the wrong reason.
    """
    run_dir = cohort / "analysis" / "an_01"
    _write(cohort, run_dir, [], reuse=True)

    assert bo.read_instance_features(run_dir) == {}
