"""An export can be narrowed to the recordings you want, and says which targets honour it.

**The class: a cost that can only be avoided by choosing, with no way to choose.** An export
that wrote one video per recording of a whole run did not fit the drive it was written to, when
one recording was wanted (issue #710). `ExportScope` could narrow
by cluster, selection and frame range, never by recording. The design note had ruled it out
(`docs/export_design.md` §1: "writing one file per recording gives the reader the same choice ...
the narrowing is picking a file"), and the reasoning fails exactly where the cost is the
*writing*: picking a file after 117 GB are written is not a choice.

What these pin, so the scope cannot quietly become a field some targets read:

- the value itself: normalised on the way in (ids, once each, tuple or list), describes itself,
  and survives the job record's JSON and YAML round trips (a tuple in a record writes
  ``!!python/tuple`` and the run does its work and then dies reading itself back);
- **every target that declares ``"recordings"`` in its ``scopes`` writes only those recordings**,
  checked by running it on a two-recording fixture (the sweep names the targets, so a new one is
  covered the day it is added), and every run function that walks the recordings of a run declares
  the scope (a fail-closed source scan, the converse: a target that reads per-recording data and
  forgot to say so would be exported whole again);
- a recording the export does not span is refused where the caller can see it, not written as an
  empty folder.

Where it stops (`fix-the-class` step 3.5): ``metadata`` and ``bundle`` describe the whole
experiment and do not declare the scope; the Export dialog offers the recordings list for a
processing run only (an analysis run is narrowed by recording through the API and ``--for-recording``).
"""

from __future__ import annotations

import importlib.util
import inspect
import json

import pytest
import yaml

from detail.api import export_targets, exporters
from detail.api.exporters import ExportError, ExportScope, ExportSubject
from detail.core.dataclasses import ExportJob
from tests.api.test_keypoint_export import keypoint_runs  # noqa: F401  (a fixture)


# --------------------------------------------------------------------------- #
# the value
# --------------------------------------------------------------------------- #
def test_recordings_are_normalised_to_ids_in_order_each_once():
    scope = ExportScope(recording_ids=["b", "a", "b", 7])
    assert scope.recording_ids == ("b", "a", "7")
    assert scope == ExportScope(
        recording_ids=("b", "a", "7")
    )  # list or tuple, same scope


def test_a_scope_with_recordings_is_not_everything():
    assert ExportScope().is_all()
    assert not ExportScope(recording_ids=("a",)).is_all()


def test_it_says_which_recordings_it_names():
    assert ExportScope(recording_ids=("r1",)).describe() == "recording r1"
    assert ExportScope(recording_ids=("r1", "r2", "r3")).describe() == "3 recordings"
    # beside another narrowing, not instead of it
    assert (
        ExportScope(cluster="4", recording_ids=("r1",)).describe()
        == "cluster 4, recording r1"
    )
    assert ExportScope(frames=(5, 9)).describe() == "frames 5-9"  # unchanged


def test_includes_means_every_recording_when_none_are_named():
    assert ExportScope().includes("anything")
    scope = ExportScope(recording_ids=("a",))
    assert scope.includes("a") and not scope.includes("b")


def test_the_scope_survives_the_job_record_round_trip():
    """The run reads its own record back to finish; an unsafe type there fails *after* the work."""
    from detail.api.experiment import _scope_dict

    as_dict = _scope_dict(ExportScope(recording_ids=("r2", "r1"), frames=(3, 1)))
    assert as_dict == {"frames": [1, 3], "recording_ids": ["r2", "r1"]}
    assert yaml.safe_load(yaml.safe_dump(as_dict)) == as_dict  # not `!!python/tuple`
    job = ExportJob(
        job_id="export_001",
        subject_kind="run",
        subject_id="run_001",
        target_ids=["tracks"],
        destination={"root": "x", "rel": ""},
        folder="f",
        created_at="2026-10-08T00:00:00Z",
        scope=as_dict,
    )
    again = ExportJob.from_dict(json.loads(json.dumps(job.to_dict())))
    assert ExportScope(**again.scope) == ExportScope(
        recording_ids=("r2", "r1"), frames=(3, 1)
    )


def test_an_everything_scope_writes_an_empty_dict():
    from detail.api.experiment import _scope_dict

    assert _scope_dict(ExportScope()) == {}


# --------------------------------------------------------------------------- #
# the targets: each one that says it honours the scope does
# --------------------------------------------------------------------------- #
def _declaring() -> set[str]:
    return {t.id for t in exporters.all_targets() if "recordings" in t.scopes}


#: Every target that declares the scope, and where it is shown to honour it. A new target
#: that declares ``"recordings"`` fails `test_every_declaring_target_is_exercised` until it has
#: a test here, which is the sweep's whole job.
EXERCISED = {
    "masks": "test_masks_narrow_to_the_recordings_asked_for",
    "tracks": "test_tracks_narrow_to_the_recordings_asked_for",
    "vectors": "test_vectors_narrow_to_the_recordings_asked_for",
    "instances": "test_behavior_targets_narrow_to_the_recordings_asked_for",
    "ethogram": "test_behavior_targets_narrow_to_the_recordings_asked_for",
    "frame_labels": "test_behavior_targets_narrow_to_the_recordings_asked_for",
    "clips": "test_the_behavior_rows_clips_are_cut_from_are_narrowed",
    "keypoints": "test_the_keypoint_targets_narrow_to_the_recordings_asked_for",
    "keypoints_dlc": "test_the_keypoint_targets_narrow_to_the_recordings_asked_for",
    "keypoints_sleap": "test_the_keypoint_targets_narrow_to_the_recordings_asked_for",
    "keypoints_nwb": "test_the_keypoint_targets_narrow_to_the_recordings_asked_for",
    "keypoints_video": "test_the_keypoint_targets_narrow_to_the_recordings_asked_for",
}


def test_every_declaring_target_is_exercised():
    assert _declaring() == set(EXERCISED), (
        "a target declares (or stopped declaring) the recordings scope without a test of it "
        f"in this module: {sorted(_declaring() ^ set(EXERCISED))}"
    )
    for target_id, test_name in EXERCISED.items():
        assert callable(
            globals().get(test_name)
        ), f"{target_id}: no test named {test_name}"


#: What a run function calls when it walks the recordings of a run or an analysis run.
_WALKERS = (
    "_detection_parts(",
    "_tracking_runs(",
    "_embedding_parts(",
    "_keypoint_sources(",
    "_narrow(",
    "discover_frame_views(",
)


def test_every_target_that_walks_recordings_declares_the_scope():
    """The converse: a `run` that walks a run's recordings and says nothing is exported whole."""
    silent = [
        f"{t.subject_kind}/{t.id}"
        for t in exporters.all_targets()
        if any(w in inspect.getsource(t.run) for w in _WALKERS)
        and "recordings" not in t.scopes
    ]
    assert not silent, f"walk a run's recordings without declaring the scope: {silent}"


def test_the_scan_would_catch_a_walker_that_forgot_to_declare(monkeypatch):
    target = exporters.target("run", "tracks")
    forgot = type(target)(**{**target.__dict__, "scopes": ()})
    monkeypatch.setattr(exporters, "_TARGETS", [forgot])
    silent = [
        t.id
        for t in exporters.all_targets()
        if any(w in inspect.getsource(t.run) for w in _WALKERS)
        and "recordings" not in t.scopes
    ]
    assert silent == ["tracks"]


def _only_recordings(result, pattern: str) -> set[str]:
    """The recording ids named by the files of ``result`` that match ``pattern``'s suffix."""
    return {
        name.split("_" + pattern)[0]
        for name in result.files
        if name.endswith("_" + pattern)
    }


@pytest.fixture
def two(processed_recording):
    """Two recordings that tell apart: ``recording_002`` has an extra detection row."""
    built = processed_recording
    square = built.square(4, 20, 4, 20)
    built.add_recording(
        "recording_002",
        detections=[
            (0, 0, (0.1, 0.1, 0.5, 0.5), square),
            (1, 0, (0.2, 0.2, 0.6, 0.6), square),
        ],
        object_ids=[0, 0],
    )
    return built


def _export(built, run_id, target_id, parent, scope=None, **options):
    subject = ExportSubject("run", built.experiment.dir, run_id)
    return exporters.write_export(
        subject, [exporters.target("run", target_id)], parent, scope=scope, **options
    )


@pytest.mark.parametrize(
    "chosen, expected",
    [
        (None, {"recording_001", "recording_002"}),
        (("recording_002",), {"recording_002"}),
        (("recording_001",), {"recording_001"}),
        (("recording_001", "recording_002"), {"recording_001", "recording_002"}),
    ],
)
def test_masks_narrow_to_the_recordings_asked_for(two, tmp_path, chosen, expected):
    scope = ExportScope(recording_ids=chosen) if chosen else None
    result = _export(two, two.detection_run_id, "masks", tmp_path, scope)
    assert _only_recordings(result, "masks.json") == expected


@pytest.mark.parametrize(
    "chosen, expected",
    [
        (None, {"recording_001", "recording_002"}),
        (("recording_002",), {"recording_002"}),
    ],
)
def test_tracks_narrow_to_the_recordings_asked_for(two, tmp_path, chosen, expected):
    scope = ExportScope(recording_ids=chosen) if chosen else None
    result = _export(two, two.tracking_run_id, "tracks", tmp_path, scope)
    assert _only_recordings(result, "tracks.csv") == expected


def test_vectors_narrow_to_the_recordings_asked_for(two, make_embedding_run, tmp_path):
    import csv

    import numpy as np
    from detail.storage import runs as runs_store

    ids = ("recording_001", "recording_002")
    for index, rid in enumerate(ids):
        make_embedding_run(
            two.experiment.get_recording(rid).dir,
            np.full((3, 4), index, dtype=np.float32),
            np.arange(3),
            np.arange(3) * 1000,
            embed_run_id="emb_07",
        )
    runs_store.write_run(
        two.experiment.dir,
        "emb_07",
        {
            "id": "emb_07",
            "module_id": "embedding",
            "recording_ids": list(ids),
            "per_recording": {
                rid: {"run_id": "emb_07", "status": "complete"} for rid in ids
            },
            "status": "complete",
        },
    )

    def recordings_in_index(scope):
        result = _export(two, "emb_07", "vectors", tmp_path, scope)
        with open(result.dest / "embeddings_index.csv", encoding="utf-8-sig") as handle:
            return {row["recording_id"] for row in csv.DictReader(handle)}

    assert recordings_in_index(None) == set(ids)
    assert recordings_in_index(ExportScope(recording_ids=("recording_002",))) == {
        "recording_002"
    }


@pytest.fixture
def two_keypoint_recordings(keypoint_runs):
    """The keypoint run of `test_keypoint_export`, over ``recording_002`` and a copy of it
    named differently as ``recording_003``: two real recordings with real stage output, so a
    narrowing is told apart by the file each writes (``Mouse 7, day 2`` / ``Mouse 8, day 3``).
    """
    import shutil

    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store
    from detail.storage import runs as runs_store

    it, recording = keypoint_runs
    source = recording.dir
    copy = experiment_store.recording_path(it.experiment.dir, "recording_003")
    shutil.copytree(source, copy)
    data = recording_store.read_recording(copy)
    data["id"] = "recording_003"
    data["name"] = "Mouse 8, day 3"
    recording_store.write_recording(copy, data)
    for run_id in ("run_010", "run_011"):
        record = runs_store.read_run(it.experiment.dir, run_id)
        record["recording_ids"] = ["recording_002", "recording_003"]
        record["per_recording"]["recording_003"] = dict(
            record["per_recording"]["recording_002"]
        )
        runs_store.write_run(it.experiment.dir, run_id, record)
    return it


KEYPOINT_STEMS = {
    "recording_002": export_targets.safe_name("Mouse 7, day 2"),
    "recording_003": export_targets.safe_name("Mouse 8, day 3"),
}


@pytest.mark.parametrize(
    "target_id",
    [
        "keypoints",
        "keypoints_dlc",
        pytest.param(
            "keypoints_sleap",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("h5py") is None,
                reason="h5py is not installed",
            ),
        ),
        pytest.param(
            "keypoints_nwb",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("pynwb") is None,
                reason="pynwb is not installed",
            ),
        ),
        "keypoints_video",
    ],
)
def test_the_keypoint_targets_narrow_to_the_recordings_asked_for(
    two_keypoint_recordings, tmp_path, target_id
):
    """Every keypoint format writes a file per recording (or per video part of one), named for
    the recording: the narrowing is the set of names that appear. ``keypoints_nwb`` is checked
    on what the writer *attempted*, because a ``pynwb`` that cannot build the file on the box
    (it does not, on this Windows one, with and without this change) fails the target after
    the recordings were chosen, not before."""
    it = two_keypoint_recordings
    subject = ExportSubject("run", it.experiment.dir, "run_011")
    target = exporters.target("run", target_id)

    def stems_in(result):
        names = list(result.files) + [
            f.name for f in result.dest.rglob("*") if f.is_file()
        ]
        return {
            rid for rid, stem in KEYPOINT_STEMS.items() if any(stem in n for n in names)
        }

    whole = exporters.write_export(subject, [target], tmp_path / "all")
    one = exporters.write_export(
        subject,
        [target],
        tmp_path / "one",
        scope=ExportScope(recording_ids=("recording_003",)),
    )
    if target_id != "keypoints_nwb":
        assert stems_in(whole) == {"recording_002", "recording_003"}
        assert stems_in(one) == {"recording_003"}
    else:
        assert stems_in(one) <= {"recording_003"}, "nothing else was attempted"


def test_behavior_targets_narrow_to_the_recordings_asked_for(authored_cohort, tmp_path):
    import csv

    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")

    def run(target_id, name, scope):
        result = exporters.write_export(
            subject,
            [exporters.target("analysis_run", target_id)],
            tmp_path / name,
            scope=scope,
        )
        return result

    def instance_recordings(result):
        with open(result.dest / "instances.csv", encoding="utf-8-sig") as handle:
            return {row["recording_id"] for row in csv.DictReader(handle)}

    everything = run("instances", "all", None)
    assert (
        len(instance_recordings(everything)) >= 2
    ), "the fixture must span two recordings"
    chosen = ExportScope(recording_ids=("r1",))
    assert instance_recordings(run("instances", "i", chosen)) == {"r1"}

    ethogram = run("ethogram", "e", chosen)
    assert {f for f in ethogram.files if f.endswith("_ethogram.csv")} == {
        "r1_ethogram.csv"
    }

    labels = run("frame_labels", "f", chosen)
    assert labels.files and all(
        f.startswith("r1") for f in labels.files if f.endswith(".csv")
    )


def test_the_behavior_rows_clips_are_cut_from_are_narrowed():
    """`clips` takes its bouts from `_narrow`, which the instance and ethogram exports share, so
    narrowing it narrows the clips; cutting real video is covered by `test_behavior_clips`.
    """
    from types import SimpleNamespace

    rows = [
        SimpleNamespace(recording_id="r1", instance_id="a", source_label="1"),
        SimpleNamespace(recording_id="r2", instance_id="b", source_label="1"),
    ]
    kept = export_targets._narrow(rows, ExportScope(recording_ids=("r2",)))
    assert [r.instance_id for r in kept] == ["b"]
    # and beside another narrowing rather than instead of it
    assert (
        export_targets._narrow(rows, ExportScope(recording_ids=("r2",), cluster="2"))
        == []
    )


# --------------------------------------------------------------------------- #
# refusing what the export does not span
# --------------------------------------------------------------------------- #
def test_a_recording_the_run_does_not_span_is_refused_up_front(two, tmp_path):
    out = tmp_path / "exports"
    with pytest.raises(ExportError, match="recording_999 is not one of the recordings"):
        two.experiment.run_export(
            "run",
            two.detection_run_id,
            ["masks"],
            out,
            scope=ExportScope(recording_ids=("recording_999",)),
        )
    assert not out.exists(), "not even the destination was created for a refused export"


def test_estimating_refuses_the_same_way(two):
    with pytest.raises(ExportError, match="not one of the recordings"):
        two.experiment.estimate_export(
            "run",
            two.detection_run_id,
            ["masks"],
            scope=ExportScope(recording_ids=("recording_999",)),
        )


def test_an_experiment_export_has_no_recordings_to_choose_between(two, tmp_path):
    with pytest.raises(ExportError, match="no recordings to choose between"):
        two.experiment.run_export(
            "experiment",
            "",
            ["metadata"],
            tmp_path,
            scope=ExportScope(recording_ids=("r1",)),
        )


def test_recordings_of_a_run_are_the_ones_it_completed(two):
    from detail.storage import runs as runs_store

    record = runs_store.read_run(two.experiment.dir, two.detection_run_id)
    record["per_recording"]["recording_002"]["status"] = "failed"
    runs_store.write_run(two.experiment.dir, two.detection_run_id, record)

    subject = ExportSubject("run", two.experiment.dir, two.detection_run_id)
    assert exporters.recordings_of(subject) == ["recording_001"]
    assert (
        exporters.recordings_of(ExportSubject("experiment", two.experiment.dir)) == []
    )
