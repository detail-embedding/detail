import contextlib
import json
import os

import cv2
import numpy as np
import pytest

import detail.api.recording as api_recording
from detail.api.experiment import Experiment
from detail.core.dataclasses import (
    ComputeTarget,
    Detection,
    FrameResult,
    ModelReadiness,
    PointCue,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.protocols import DetectionModel, EmbeddingModel
from detail.configs.tracking import DEFAULT_TRACKER
from detail.core.registries import (
    EMBEDDING_MODELS,
    MODELS,
    TRACKERS,
    TRAINED_LOADERS,
    register_embedding_model,
    register_model,
    register_trained_loader,
)
from detail.storage import models as models_store
from detail.storage import recording as recording_store
from detail.storage import video as video_store
from detail.storage.zarr_io import (
    DetectionPartWriter,
    PartZarrReader,
    TracksSummaryWriter,
)


def _write_part(path, frames):
    with DetectionPartWriter(path) as w:
        for frame_idx, dets in frames:
            for d_idx, (bbox, conf, cname) in enumerate(dets):
                w.add_row(
                    frame_idx, frame_idx * 1000, d_idx, cname, float(conf), bbox, ""
                )


def _drift(base, f, step=0.01):
    x1, y1, x2, y2 = base
    return (x1 + step * f, y1, x2 + step * f, y2)


def _setup(tmp_path):
    """A experiment + recording with one completed detection run (det_01)."""
    proj = Experiment.create("proj", tmp_path)
    recording = proj.create_recording("recording_001")
    det_dir = recording.dir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    _write_part(
        det_dir / "part_001.zarr",
        [(f, [(_drift((0.1, 0.1, 0.2, 0.2), f), 0.9, "mouse")]) for f in range(4)],
    )
    # The visible trackers read the mask grid from the detection run's envelope.
    (det_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete", "output": "detections/det_01/"}
    ]
    recording_store.write_recording(recording.dir, data)
    return recording


def test_run_tracking_end_to_end(tmp_path):
    recording = _setup(tmp_path)
    run = recording.run_tracking("det_01", "anchor_tracker", {"iou_threshold": 0.3})
    run.wait()
    assert run.status == "complete"

    assign = PartZarrReader(
        recording.dir / "tracking" / run.id / "assignments" / "part_001.zarr"
    ).read()["object_id"]
    assert assign.tolist() == [0, 0, 0, 0]

    runs = recording.tracking_runs
    assert len(runs) == 1
    assert runs[0]["id"] == run.id
    assert runs[0]["detection_run_id"] == "det_01"
    assert runs[0]["output"] == f"tracking/{run.id}/"
    assert runs[0]["status"] == "complete"


def test_run_tracking_accepts_no_hyperparams(tmp_path):
    """``hyperparams=None`` means "the tracker's own defaults", not a crash.

    ``config.get("hyperparams", {})`` returns None when the key is *present and null*, which
    reaches ``set(hyperparams)`` as a bare TypeError rather than as any kind of validation
    error. Reachable only now that the argument is optional.
    """
    recording = _setup(tmp_path)
    run = recording.run_tracking("det_01", "anchor_tracker")
    run.wait()
    assert run.status == "complete"
    assert recording.tracking_runs[0]["hyperparams"]["iou_threshold"] == 0.1


def test_run_tracking_with_no_tracker_records_the_one_it_resolved(tmp_path):
    """A resolved default is persisted (Tenet 9), not left for a later reader to re-derive.

    ``validate`` discards the validated model and reads the raw config dict, so the model's
    default reaches nothing on its own — without the write-back the run record would simply
    not name a tracker, and a run that cannot say what it ran is not reproducible from its own
    metadata. Worse than missing: **Start from** and ``retry_run`` both re-read this record, so
    an unnamed tracker silently means "whatever the default is the day you read it".

    Asserted on the record rather than on the run's output because this fixture's detections
    carry no masks, which the default tracker's contiguity gate needs — see
    ``tests/pipeline/test_tracking_defaults.py`` for what that costs.
    """
    recording = _setup(tmp_path)
    run = recording.run_tracking("det_01")
    run.wait()

    [record] = recording.tracking_runs
    assert record["tracker_id"] == DEFAULT_TRACKER
    # And the hyperparams it ran with, not an empty dict: the same rule one field over.
    assert set(record["hyperparams"]) == set(
        TRACKERS[DEFAULT_TRACKER].HYPERPARAMS_SCHEMA
    )


def test_run_tracking_unknown_detection_run(tmp_path):
    recording = _setup(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        recording.run_tracking("det_99", "anchor_tracker", {})


def test_run_tracking_incomplete_detection_run(tmp_path):
    recording = _setup(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"][0]["status"] = "partial"
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="not complete"):
        recording.run_tracking("det_01", "anchor_tracker", {})


def test_run_tracking_unknown_tracker(tmp_path):
    recording = _setup(tmp_path)
    with pytest.raises(ValueError, match="unknown tracker_id"):
        recording.run_tracking("det_01", "nope", {})


def test_run_tracking_bad_hyperparams(tmp_path):
    recording = _setup(tmp_path)
    with pytest.raises(ValueError, match="unknown hyperparams"):
        recording.run_tracking("det_01", "anchor_tracker", {"bogus": 1})


def test_run_tracking_unknown_arena(tmp_path):
    recording = _setup(tmp_path)
    with pytest.raises(ValueError, match="unknown arena"):
        recording.run_tracking("det_01", "anchor_tracker", {}, arena_ids=["arena_x"])


def test_run_tracking_rejects_multiple_arenas(tmp_path):
    recording = _setup(tmp_path)
    a = recording.define_arena(
        "main", [(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]
    )
    b = recording.define_arena(
        "zone", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])]
    )
    with pytest.raises(ValueError, match="at most one arena_id"):
        recording.run_tracking("det_01", "anchor_tracker", {}, arena_ids=[a, b])


# --- define_arena ----------------------------------------------------------- #
def _full_frame():
    return [(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]


def test_define_arena_returns_id_and_appears_in_arenas(tmp_path):
    recording = _make_recording(tmp_path)
    arena_id = recording.define_arena("main", _full_frame())
    assert arena_id == "arena_001"
    assert recording.arenas == [{"id": "arena_001", "name": "main", "n_vertices": 4}]


def test_define_arena_sequential_ids(tmp_path):
    recording = _make_recording(tmp_path)
    a = recording.define_arena("main", _full_frame())
    b = recording.define_arena(
        "zone", [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])]
    )
    assert (a, b) == ("arena_001", "arena_002")


def test_define_arena_duplicate_name_raises(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_arena("main", _full_frame())
    with pytest.raises(ValueError, match="name 'main'"):
        recording.define_arena("main", _full_frame())


def test_clear_arenas_removes_definition(tmp_path):
    recording = _make_recording(tmp_path)
    arena_id = recording.define_arena("main", _full_frame())
    arena_dir = recording.dir / "arenas" / arena_id
    assert recording.arenas and arena_dir.is_dir()
    recording.clear_arenas()
    assert recording.arenas == []
    assert not arena_dir.exists()  # keyframes dir removed too
    # idempotent + lets a new arena be defined fresh
    recording.clear_arenas()
    assert recording.define_arena("main", _full_frame()) == "arena_001"


def test_arena_keyframes_roundtrip(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_arena("main", _full_frame())
    assert recording.arena_keyframes("arena_001") == _full_frame()
    with pytest.raises(ValueError, match="unknown arena_id"):
        recording.arena_keyframes("arena_999")


# --- ROIs and POIs (analysis-time spatial definitions) ---------------------- #


def test_define_roi_appears_in_rois_with_active_none(tmp_path):
    recording = _make_recording(tmp_path)
    roi_id = recording.define_roi("lever_zone", _full_frame())
    assert roi_id == "roi_001"
    assert recording.rois == [
        {"id": "roi_001", "name": "lever_zone", "n_vertices": 4, "active": None}
    ]
    # roundtrip + unknown id
    assert recording.roi_keyframes("roi_001") == _full_frame()
    with pytest.raises(ValueError, match="unknown roi id"):
        recording.roi_keyframes("roi_999")


def test_define_roi_with_active_interval_persists(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_roi("phase", _full_frame(), active=(100, 200))
    assert recording.rois[0]["active"] == {"start": 100, "end": 200}


def test_roi_at_frame_respects_active_interval(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_roi("phase", _full_frame(), active=(100, 200))
    assert recording.roi_at_frame("roi_001", 150) is not None
    assert recording.roi_at_frame("roi_001", 50) is None  # before the window
    assert recording.roi_at_frame("roi_001", 250) is None  # after the window


def test_delete_roi_is_unconditional(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_roi("zone", _full_frame())
    recording.delete_roi("roi_001")
    assert recording.rois == []
    assert not (recording.dir / "rois" / "roi_001").exists()


def test_define_point_single_vertex(tmp_path):
    recording = _make_recording(tmp_path)
    point_id = recording.define_point("lever", [(0, [(0.5, 0.5)])])
    assert point_id == "point_001"
    assert recording.points == [
        {"id": "point_001", "name": "lever", "n_vertices": 1, "active": None}
    ]
    assert recording.point_keyframes("point_001") == [(0, [(0.5, 0.5)])]


def test_define_point_rejects_polygon(tmp_path):
    recording = _make_recording(tmp_path)
    with pytest.raises(ValueError, match="exactly 1 vertex"):
        recording.define_point("bad", _full_frame())


def test_point_at_frame_and_delete(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_point("lever", [(0, [(0.5, 0.5)])], active=(10, 20))
    got = recording.point_at_frame("point_001", 15)
    assert got is not None and got.shape == (1, 2)
    assert recording.point_at_frame("point_001", 5) is None
    recording.delete_point("point_001")
    assert recording.points == []


def test_set_rois_replaces_all_in_one_transaction(tmp_path):
    recording = _make_recording(tmp_path)
    recording.set_rois(
        [
            {"name": "a", "keyframes": _full_frame(), "active": None},
            {"name": "b", "keyframes": _full_frame(), "active": (0, 900)},
        ]
    )
    assert [r["name"] for r in recording.rois] == ["a", "b"]
    # a second "Save all" replaces the whole set
    ids = recording.set_rois([{"name": "c", "keyframes": _full_frame()}])
    assert ids == ["roi_001"]
    assert [r["name"] for r in recording.rois] == ["c"]


def test_set_points_replaces_all(tmp_path):
    recording = _make_recording(tmp_path)
    recording.set_points([{"name": "lever", "keyframes": [(0, [(0.5, 0.5)])]}])
    assert [p["name"] for p in recording.points] == ["lever"]
    recording.set_points([])  # clear all
    assert recording.points == []


def _square(offset=0.0):
    return [(0.1 + offset, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9 + offset)]


# --- one door for a setup arriving from elsewhere (§10) ---------------------- #
# `receive_spatial` is the single door, so every one of these six methods answers with
# `(added, flattened, refused)` rather than a count: a count cannot say that a polygon
# lost its camera motion on the way, or that a name was left behind and why, and both are
# changes to what the caller asked for that leave a well-formed result behind.


def test_export_import_spatial_csv_roundtrip(tmp_path):
    """The whole setup leaves in one file and arrives through the same door as a copy."""
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois([{"name": "zone", "keyframes": _full_frame()}])
    src.set_points([{"name": "lever", "keyframes": [(0, [(0.5, 0.5)])]}])
    out = tmp_path / "regions.csv"
    src.export_spatial_csv(out)
    dst = exp.create_recording("r2")

    assert dst.import_spatial_csv(out) == (["zone", "lever"], [], [])
    assert [r["name"] for r in dst.rois] == ["zone"]
    assert [p["name"] for p in dst.points] == ["lever"]


def test_import_spatial_csv_refuses_a_windowed_region(tmp_path):
    """A CSV is pulled under the same rules as a recording, active window included.

    Exporting and re-importing is the obvious way round a refusal, so the policy has to
    live at the destination rather than at either source — otherwise "it would not copy,
    so I saved it to a file" quietly works and the window lands on a timeline it does not
    describe.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois([{"name": "zone", "keyframes": _full_frame(), "active": (0, 30)}])
    out = tmp_path / "regions.csv"
    src.export_spatial_csv(out)
    dst = exp.create_recording("r2")

    added, _flattened, refused = dst.import_spatial_csv(out)

    assert added == []
    assert [name for name, _ in refused] == ["zone"]
    assert "active frame window" in refused[0][1]
    assert dst.rois == []


def test_import_spatial_csv_refuses_a_colliding_name_and_keeps_the_rest(tmp_path):
    """Same per-name grain through the CSV door as through the recording one."""
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois(
        [
            {"name": "zone", "keyframes": [(0, _square())]},
            {"name": "nest", "keyframes": [(0, _square())]},
        ]
    )
    out = tmp_path / "regions.csv"
    src.export_spatial_csv(out)
    dst = exp.create_recording("r2")
    mine = [(0, _square(0.05))]
    dst.define_roi("zone", mine)

    added, flattened, refused = dst.import_spatial_csv(out)

    assert added == ["nest"] and flattened == []
    assert [name for name, _reason in refused] == ["zone"]
    assert [r["name"] for r in dst.rois] == ["zone", "nest"]
    assert dst.roi_keyframes(dst.rois[0]["id"]) == mine


def test_import_spatial_from_adds_beside_what_is_already_there(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois([{"name": "zone", "keyframes": _full_frame()}])
    dst = exp.create_recording("r2")
    dst.define_roi("existing", _full_frame())

    assert dst.import_spatial_from(src) == (["zone"], [], [])
    assert [r["name"] for r in dst.rois] == ["existing", "zone"]


def test_import_spatial_from_refuses_a_colliding_name_and_keeps_the_rest(tmp_path):
    """The refusal grain for regions/points is the **name**, not the kind.

    Refusing the whole set because one name clashes would kill "pull one more zone from a
    template", which is the ordinary use; replacing the collision would be worse still,
    since this recording's own polygon is the one drawn against its own video. So the one
    name is refused and the others land — and the surviving polygon is asserted by its
    keyframes, because a same-named replacement passes a name-only check.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois(
        [
            {"name": "zone", "keyframes": [(0, _square())]},
            {"name": "nest", "keyframes": [(0, _square())]},
        ]
    )
    dst = exp.create_recording("r2")
    mine = [(0, _square(0.05))]
    dst.define_roi("zone", mine)

    added, flattened, refused = dst.import_spatial_from(src)

    assert added == ["nest"] and flattened == []
    assert [name for name, _reason in refused] == ["zone"]
    assert "already exists on" in refused[0][1]
    assert [r["name"] for r in dst.rois] == ["zone", "nest"]  # one zone, not two
    assert dst.roi_keyframes(dst.rois[0]["id"]) == mine


def test_import_spatial_from_flattens_camera_motion_and_names_it(tmp_path):
    """A moving polygon tracks *that* camera's pan, so only its frame-0 shape travels."""
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois(
        [{"name": "panning zone", "keyframes": [(0, _square()), (300, _square(0.05))]}]
    )
    dst = exp.create_recording("r2")

    added, flattened, refused = dst.import_spatial_from(src)

    assert added == ["panning zone"] and flattened == ["panning zone"] and refused == []
    assert dst.roi_keyframes(dst.rois[0]["id"]) == [(0, _square())]


def test_import_spatial_from_requires_something_that_can_travel(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")  # nothing drawn
    dst = exp.create_recording("r2")
    dst.define_roi("keep", _full_frame())

    with pytest.raises(ValueError, match="to copy"):
        dst.import_spatial_from(src)
    assert [r["name"] for r in dst.rois] == ["keep"]


def test_propagate_spatial_from_adds_and_never_clobbers(tmp_path):
    """The cross-recording copy adds; it never replaces.

    A name already on the target is refused rather than overwritten or suffixed, and the
    rest still arrive — regions are a set, so one clash is not a reason to refuse the lot.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.set_rois(
        [
            {"name": "zone", "keyframes": _full_frame()},
            {"name": "extra", "keyframes": _full_frame()},
        ]
    )
    dst = exp.create_recording("r2")
    dst.define_roi("zone", _full_frame())

    added, flattened, refused = dst.propagate_spatial_from(src, "roi")

    assert added == ["extra"] and flattened == []
    assert [name for name, _ in refused] == ["zone"]
    assert sorted(r["name"] for r in dst.rois) == ["extra", "zone"]


def test_propagate_spatial_from_with_nothing_to_take_is_not_an_error(tmp_path):
    """Nothing to take is a *report*, not an exception.

    It is called once per kind, so raising here let one fully-refused kind abort a
    transfer whose other kinds were perfectly transferable — and the caller could see
    neither half.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")  # no ROIs
    dst = exp.create_recording("r2")
    dst.define_roi("keep", _full_frame())

    assert dst.propagate_spatial_from(src, "roi") == ([], [], [])
    assert [r["name"] for r in dst.rois] == ["keep"]


def test_a_windowed_region_is_reported_and_the_rest_still_travel(tmp_path):
    """One region's active window must not strand the kinds beside it.

    A source with a windowed ROI and a transferable POI used to import *nothing*: the ROI
    pass raised, and the POI never got a chance.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.define_roi("timed zone", _full_frame(), active=(5000, 9000))
    src.define_point("lever", [(0, [(0.5, 0.5)])])
    dst = exp.create_recording("r2")

    added, _flattened, refused = dst.import_spatial_from(src)

    assert added == ["lever"], "the point was stranded by the region's refusal"
    assert [name for name, _ in refused] == ["timed zone"]
    assert "active frame window" in refused[0][1]
    assert dst.rois == []


def test_import_field_of_view_from_onto_an_empty_recording(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.define_arena("rig", [(0, _square())])
    dst = exp.create_recording("r2")

    assert dst.import_field_of_view_from(src) == (["rig"], [], [])
    assert [a["name"] for a in dst.arenas] == ["rig"]
    assert dst.arena_keyframes(dst.arenas[0]["id"]) == [(0, _square())]


def test_import_field_of_view_from_is_refused_when_one_is_already_there(tmp_path):
    """A field of view is refused **whole**, unlike the per-name regions.

    A second one is not a richer setup — the batch tracking picker intersects arena *ids*
    across recordings — so having one at all is the refusal, and the target keeps its own
    polygon. This used to `clear_arenas()` and copy over the top after a Yes/No, which is
    the same operation the *push* direction refuses outright: one setup arriving from
    elsewhere had two contradictory rules depending on which way you started it.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.define_arena("rig", [(0, _square())])
    dst = exp.create_recording("r2")
    mine = [(0, _square(0.05))]
    dst.define_arena("old", mine)

    with pytest.raises(ValueError, match="already has a field of view"):
        dst.import_field_of_view_from(src)
    assert [a["name"] for a in dst.arenas] == ["old"]
    assert dst.arena_keyframes(dst.arenas[0]["id"]) == mine


def test_import_field_of_view_leaves_a_run_dependent_polygon_alone(tmp_path):
    """The polygon a completed tracking run filtered on is never swapped out from under it.

    Refused now for having a field of view at all, which is the stronger reason — the
    property worth asserting is the one below it: the run's arena id still resolves to the
    polygon that run recorded.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.define_arena("rig", [(0, _square())])
    dst = exp.create_recording("r2")
    in_use = [(0, _square(0.05))]
    arena_id = dst.define_arena("in use", in_use)
    data = recording_store.read_recording(dst.dir)
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "arena_ids": [arena_id]}
    ]
    recording_store.write_recording(dst.dir, data)

    with pytest.raises(ValueError):
        dst.import_field_of_view_from(src)
    assert [a["name"] for a in dst.arenas] == ["in use"]  # untouched
    assert dst.arena_keyframes(arena_id) == in_use
    assert dst.arena_dependents(arena_id) == ["track_01"]


def test_import_field_of_view_requires_a_source(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")  # no field of view drawn
    dst = exp.create_recording("r2")  # and none here either, so nothing is in the way
    with pytest.raises(ValueError, match="to copy"):
        dst.import_field_of_view_from(src)
    assert dst.arenas == []


def test_import_field_of_view_and_calibration_from_csv(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.define_arena("rig", [(0, _square())])
    src.set_rois([{"name": "zone", "keyframes": _full_frame()}])
    src.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    out = tmp_path / "setup.csv"
    src.export_spatial_csv(out)

    dst = exp.create_recording("r2")
    _set_resolution(dst, 1920, 1080)
    assert dst.import_field_of_view_csv(out) == (["rig"], [], [])
    assert [a["name"] for a in dst.arenas] == ["rig"]
    assert dst.import_calibration_csv(out)["pixels_per_mm"] == pytest.approx(4.8)
    assert dst.pixels_per_mm == pytest.approx(4.8)
    # The same file still feeds the regions import, which takes only the ROIs/POIs.
    assert dst.import_spatial_csv(out) == (["zone"], [], [])
    assert [r["name"] for r in dst.rois] == ["zone"]


def test_import_field_of_view_csv_is_refused_when_one_is_already_there(tmp_path):
    """The CSV door obeys the same rule as the recording one — that is the point of it."""
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    src.define_arena("rig", [(0, _square())])
    out = tmp_path / "setup.csv"
    src.export_spatial_csv(out)
    dst = exp.create_recording("r2")
    mine = [(0, _square(0.05))]
    dst.define_arena("old", mine)

    with pytest.raises(ValueError, match="already has a field of view"):
        dst.import_field_of_view_csv(out)
    assert [a["name"] for a in dst.arenas] == ["old"]
    assert dst.arena_keyframes(dst.arenas[0]["id"]) == mine


def test_import_calibration_from_recording_keeps_its_reference_resolution(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    dst = exp.create_recording("r2")
    _set_resolution(dst, 960, 540)

    dst.import_calibration_from(src)
    # The block travels with the resolution it was picked at; the reader rescales, so a
    # half-width recording reads half the pixels-per-mm.
    assert dst.distance_calibration["ref_resolution"] == [1920, 1080]
    assert dst.pixels_per_mm == pytest.approx(2.4)


def test_import_calibration_is_refused_when_one_is_already_set(tmp_path):
    """There is one pixel scale per recording, so it is refused whole, like the field of
    view — and the number already there is what a future run would use, so it survives.
    """
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    dst = exp.create_recording("r2")
    _set_resolution(dst, 1920, 1080)
    dst.set_pixels_per_mm(99.0)

    with pytest.raises(ValueError, match="already has a distance calibration"):
        dst.import_calibration_from(src)
    assert dst.pixels_per_mm == pytest.approx(99.0)


def test_import_calibration_csv_is_refused_when_one_is_already_set(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    out = tmp_path / "setup.csv"
    src.export_spatial_csv(out)
    dst = exp.create_recording("r2")
    _set_resolution(dst, 1920, 1080)
    dst.set_pixels_per_mm(99.0)

    with pytest.raises(ValueError, match="already has a distance calibration"):
        dst.import_calibration_csv(out)
    assert dst.pixels_per_mm == pytest.approx(99.0)


def test_import_calibration_requires_a_source(tmp_path):
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    dst = exp.create_recording("r2")
    with pytest.raises(ValueError, match="no distance calibration"):
        dst.import_calibration_from(src)


def _set_resolution(recording, width, height):
    from detail.storage import recording as recording_store

    data = recording_store.read_recording(recording.dir)
    data["video"] = {"resolution": [width, height], "files": [], "storage": "default"}
    recording_store.write_recording(recording.dir, data)


def test_distance_calibration_from_points(tmp_path):
    recording = _make_recording(tmp_path)
    _set_resolution(recording, 1920, 1080)
    # two points half the width apart at 1920 wide → 960 px across 200 mm → 4.8 px/mm
    block = recording.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    assert abs(block["pixels_per_mm"] - 4.8) < 1e-9
    assert recording.distance_calibration["pixels_per_mm"] == block["pixels_per_mm"]
    assert abs(recording.pixels_per_mm - 4.8) < 1e-9
    recording.clear_distance_calibration()
    assert recording.distance_calibration is None and recording.pixels_per_mm is None


def test_set_pixels_per_mm_scalar(tmp_path):
    recording = _make_recording(tmp_path)
    _set_resolution(recording, 1920, 1080)
    recording.set_pixels_per_mm(5.0)
    assert recording.distance_calibration["pixels_per_mm"] == 5.0
    assert recording.distance_calibration["points"] is None


def test_calibration_requires_video(tmp_path):
    recording = _make_recording(tmp_path)  # metadata-only, no video
    with pytest.raises(ValueError, match="no imported video"):
        recording.set_pixels_per_mm(5.0)


def test_arena_dependents_block_clear(tmp_path):
    recording = _make_recording(tmp_path)
    arena_id = recording.define_arena("main", _full_frame())
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "arena_ids": [arena_id]}
    ]
    recording_store.write_recording(recording.dir, data)
    assert recording.arena_dependents(arena_id) == ["track_01"]
    with pytest.raises(ValueError, match="delete those runs first"):
        recording.clear_arenas()
    assert recording.arenas  # untouched while a tracking run uses it


def test_delete_arena_removes_one_keeps_others(tmp_path):
    recording = _make_recording(tmp_path)
    a = recording.define_arena("first", _full_frame())
    b = recording.define_arena("second", _full_frame())
    recording.delete_arena(a)
    assert [x["id"] for x in recording.arenas] == [b]  # only the named one is gone
    assert not (recording.dir / "arenas" / a).exists()  # its keyframes dir too
    assert (recording.dir / "arenas" / b / "keyframes.csv").is_file()  # other intact


def test_delete_arena_refuses_when_used_by_run(tmp_path):
    recording = _make_recording(tmp_path)
    arena_id = recording.define_arena("main", _full_frame())
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "arena_ids": [arena_id]}
    ]
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="delete those runs first"):
        recording.delete_arena(arena_id)
    assert recording.arenas  # untouched while a tracking run uses it


def test_define_arena_empty_keyframes_raises(tmp_path):
    recording = _make_recording(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        recording.define_arena("main", [])


def test_define_arena_vertex_drift_raises(tmp_path):
    recording = _make_recording(tmp_path)
    keyframes = [
        (0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        (10, [(0.1, 0.1), (0.9, 0.9), (0.5, 0.5)]),  # 3 verts
    ]
    with pytest.raises(ValueError, match="expected 4"):
        recording.define_arena("main", keyframes)


def test_define_arena_writes_keyframes_csv(tmp_path):
    recording = _make_recording(tmp_path)
    recording.define_arena("main", _full_frame())
    assert (recording.dir / "arenas" / "arena_001" / "keyframes.csv").is_file()


def test_run_tracking_rejects_active_run(tmp_path):
    recording = _setup(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"] = [
        {
            "id": "track_01",
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "status": "running",
            "output": "tracking/track_01/",
        }
    ]
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="already active"):
        recording.run_tracking("det_01", "anchor_tracker", {})


# --- run_detection ---------------------------------------------------------- #
H, W = 24, 32

FAKE_MODEL_SOURCE = """
import numpy as np
from detail.core.dataclasses import Detection, FrameResult, TextCue
from detail.core.protocols import DetectionModel
from detail.core.registries import register_model


@register_model("fake_detector")
class FakeDetector(DetectionModel):
    model_id = "fake_detector"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return FrameResult.from_detections(
            0, 0, [Detection(mask, (0.25, 0.25, 0.75, 0.75), "mouse", 0.9, None)]
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]
"""


class _FakeDetector(DetectionModel):
    model_id = "fake_detector"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


@contextlib.contextmanager
def _registered_fake_model():
    register_model("fake_detector")(_FakeDetector)
    try:
        yield
    finally:
        MODELS.pop("fake_detector", None)


class _FakeClosedSet(_FakeDetector):
    """A trained (closed-set) model: no cue types, its own baked-in classes; one
    ``predict`` returns detections for every class (the real TrainedYOLO contract)."""

    supported_cue_types = []

    def __init__(self, weights_path=None, classes=None):
        self._classes = list(classes or [])

    @property
    def classes(self):
        return list(self._classes)

    def readiness(self):
        return ModelReadiness(
            ready=True, reason="ok", message="", action_url=None, repo_id=None
        )

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 2, : w // 2] = True
        return FrameResult.from_detections(
            0,
            0,
            [
                Detection(mask, (0.1, 0.1, 0.3, 0.3), "mouse", 0.9, None),
                Detection(mask, (0.4, 0.4, 0.6, 0.6), "cage", 0.8, None),
            ],
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


@contextlib.contextmanager
def _registered_closed_set_loader():
    register_trained_loader("fake_closed")(_FakeClosedSet)
    try:
        yield
    finally:
        TRAINED_LOADERS.pop("fake_closed", None)


def _write_trained_model(experiment_root, classes=("mouse", "cage")):
    mdir = models_store.model_dir(experiment_root, "model_001")
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "status": "complete",
            "model_family": "fake_closed",
            "classes": [{"id": i, "name": n} for i, n in enumerate(classes)],
            "weights": {"best": "best.pt", "last": None},
        },
    )


def _cpu():
    return ComputeTarget("cpu", None, None, None)


def _text_prompt():
    return Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))])


def _make_recording(tmp_path):
    return Experiment.create("proj", tmp_path).create_recording("recording_001")


def _write_video(path, n_frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()


def _count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _add_video(recording):
    sd = recording.dir
    _write_video(sd / "video" / "part_001.mp4", n_frames=5)
    n = _count_frames(sd / "video" / "part_001.mp4")
    recording_store.write_timestamps_csv(
        sd / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(n, dtype=np.int64) * 1000,
    )
    data = recording_store.read_recording(sd)
    data["video"] = {
        "storage": "default",
        "fps_container": 30.0,
        "fps_actual": 30.0,
        "fps_source": "container",
        "resolution": [W, H],
        "total_duration_s": n / 30.0,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    recording_store.write_recording(sd, data)
    return n


def test_run_detection_unknown_model(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with pytest.raises(ValueError, match="unknown model"):
        recording.run_detection("nope", _text_prompt(), _cpu())


def test_run_detection_unsupported_cue(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    prompt = Prompt(targets=[PromptTarget("mouse", PointCue([(0.1, 0.2)], [1]))])
    with _registered_fake_model():
        with pytest.raises(ValueError, match="does not support cue types"):
            recording.run_detection("fake_detector", prompt, _cpu())


def test_run_detection_no_video(tmp_path):
    recording = _make_recording(tmp_path)  # no video added
    with _registered_fake_model():
        with pytest.raises(ValueError, match="no video parts"):
            recording.run_detection("fake_detector", _text_prompt(), _cpu())


def test_run_detection_end_to_end(tmp_path, monkeypatch):
    # fake model importable by the API process and the worker subprocess
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_model_mod.py").write_text(FAKE_MODEL_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.setenv(
        "PYTHONPATH", str(mod_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    monkeypatch.setattr(
        api_recording,
        "_PIPELINE_RUNNERS",
        api_recording._PIPELINE_RUNNERS + ("fake_model_mod",),
    )

    recording = _make_recording(tmp_path)
    n = _add_video(recording)
    run = recording.run_detection("fake_detector", _text_prompt(), _cpu())
    run.wait()
    assert run.status == "complete"

    data = PartZarrReader(
        recording.dir / "detections" / run.id / "part_001.zarr"
    ).read()
    assert data["frame_idx"].shape[0] == n
    assert [str(x) for x in data["class_name"]] == ["mouse"] * n
    assert recording.detection_runs[0]["id"] == run.id
    assert recording.detection_runs[0]["status"] == "complete"
    # The config model's model_dump() must reproduce the stored record shape unchanged.
    assert recording.detection_runs[0]["model_id"] == "fake_detector"
    assert recording.detection_runs[0]["conf_threshold"] == 0.5
    assert "weights_source" in recording.detection_runs[0]


# --- run_prompt_test (Section 8) -------------------------------------------- #
def test_run_prompt_test_unknown_model(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with pytest.raises(ValueError, match="unknown model"):
        recording.run_prompt_test(
            "no_such_model",
            [_text_prompt()],
            n_sample_frames=2,
            compute=_cpu(),
        )


def test_run_prompt_test_unsupported_cue(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    prompt = Prompt(targets=[PromptTarget("mouse", PointCue([(0.1, 0.2)], [1]))])
    with _registered_fake_model():
        with pytest.raises(ValueError, match="does not support cue types"):
            recording.run_prompt_test(
                "fake_detector",
                [prompt],
                n_sample_frames=2,
                compute=_cpu(),
            )


def test_run_prompt_test_zero_samples_rejected(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_fake_model():
        with pytest.raises(ValueError, match="must be > 0"):
            recording.run_prompt_test(
                "fake_detector",
                [_text_prompt()],
                n_sample_frames=0,
                compute=_cpu(),
            )


def test_run_prompt_test_empty_prompts_rejected(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_fake_model():
        with pytest.raises(ValueError, match="prompts must be non-empty"):
            recording.run_prompt_test(
                "fake_detector",
                [],
                n_sample_frames=2,
                compute=_cpu(),
            )


def test_run_prompt_test_no_video(tmp_path):
    recording = _make_recording(tmp_path)  # no video added
    with _registered_fake_model():
        with pytest.raises(ValueError, match="no video parts"):
            recording.run_prompt_test(
                "fake_detector",
                [_text_prompt()],
                n_sample_frames=2,
                compute=_cpu(),
            )


# --- closed-set (trained/local) preview (#16) ------------------------------- #
def test_run_prompt_test_closed_set_previews_selected_classes(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_closed_set_loader():
        _write_trained_model(recording.dir.parents[1])
        results = recording.run_prompt_test(
            "model_001",
            [],  # closed-set takes no text prompt
            n_sample_frames=2,
            compute=_cpu(),
            display_classes=["mouse"],
        )
    assert results
    for r in results:  # one column for the single selected class, filtered to it
        assert set(r.results) == {"prompt_0"}
        assert all(d.class_name == "mouse" for d in r.results["prompt_0"].detections())


def test_run_prompt_test_closed_set_defaults_to_all_classes(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_closed_set_loader():
        _write_trained_model(recording.dir.parents[1])
        results = recording.run_prompt_test(
            "model_001", [], n_sample_frames=2, compute=_cpu()
        )
    assert all(set(r.results) == {"prompt_0", "prompt_1"} for r in results)


def test_run_prompt_test_closed_set_rejects_text_prompt(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_closed_set_loader():
        _write_trained_model(recording.dir.parents[1])
        with pytest.raises(ValueError, match="closed-set"):
            recording.run_prompt_test(
                "model_001",
                [_text_prompt()],
                n_sample_frames=2,
                compute=_cpu(),
            )


def test_run_prompt_test_unknown_display_class(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_closed_set_loader():
        _write_trained_model(recording.dir.parents[1])
        with pytest.raises(ValueError, match="not one of"):
            recording.run_prompt_test(
                "model_001",
                [],
                n_sample_frames=2,
                compute=_cpu(),
                display_classes=["frog"],
            )


def test_run_prompt_test_unsupported_backend(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with _registered_fake_model():
        with pytest.raises(ValueError, match="not supported"):
            recording.run_prompt_test(
                "fake_detector",
                [_text_prompt()],
                n_sample_frames=2,
                compute=ComputeTarget("cuda", [0], None, None),  # fake is cpu-only
            )


# --- run_embedding ---------------------------------------------------------- #
EMB_RENDERER = [
    {"type": "init_binary_mask"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 32},
]


class _FakeEmbedder(EmbeddingModel):
    model_id = "fake_embedder"
    output_dim = 8
    supported_backends = ["cpu"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def embed(self, image):
        return np.zeros(8, dtype=np.float32)

    def embed_batch(self, images):
        n = images.shape[0] if hasattr(images, "shape") else len(list(images))
        return np.zeros((n, 8), dtype=np.float32)


@contextlib.contextmanager
def _registered_embedder():
    register_embedding_model("fake_embedder")(_FakeEmbedder)
    try:
        yield
    finally:
        EMBEDDING_MODELS.pop("fake_embedder", None)


def _setup_tracking(tmp_path):
    recording = _make_recording(tmp_path)
    src = tmp_path / "raw.mp4"
    _write_video(src, n_frames=5)
    recording.add_video_part(src)  # embedding decodes the source, so it must exist
    track_dir = recording.dir / "tracking" / "track_01"
    track_dir.mkdir(parents=True)
    with TracksSummaryWriter(track_dir / "tracks.zarr") as w:
        w.add_track(0, "mouse", 0, 3, 4, 0, 0.9)
    data = recording_store.read_recording(recording.dir)
    # A complete detection -> tracking lineage (embedding reads the source detection saw).
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete", "downsampling_run_id": None}
    ]
    data["tracking_runs"] = [
        {
            "id": "track_01",
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "status": "complete",
            "output": "tracking/track_01/",
        }
    ]
    recording_store.write_recording(recording.dir, data)
    return recording


def test_run_embedding_tracking_not_found(tmp_path):
    recording = _setup_tracking(tmp_path)
    with _registered_embedder():
        with pytest.raises(ValueError, match="not found"):
            recording.run_embedding("track_99", [0], "fake_embedder", EMB_RENDERER)


def test_run_embedding_tracking_not_complete(tmp_path):
    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"][0]["status"] = "running"
    recording_store.write_recording(recording.dir, data)
    with _registered_embedder():
        with pytest.raises(ValueError, match="not complete"):
            recording.run_embedding("track_01", [0], "fake_embedder", EMB_RENDERER)


def test_run_embedding_track_id_missing(tmp_path):
    recording = _setup_tracking(tmp_path)
    with _registered_embedder():
        with pytest.raises(ValueError, match="not in tracking run"):
            recording.run_embedding("track_01", [5], "fake_embedder", EMB_RENDERER)


def test_validate_resolves_constructor_to_renderer_spec(tmp_path):
    from detail.pipeline.embedding import EmbeddingModule

    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "model_id": "fake_embedder",
        "constructor": "full_frame",  # GUI path: a named constructor, no spec
        "compute": ComputeTarget("cpu", None, None, None),
    }
    with _registered_embedder():
        out = EmbeddingModule().validate(recording.dir, data, config)
    assert out["constructor"] == "full_frame"
    # validate now defaults the resize target to the model's input size (§14), so the
    # GUI path (no explicit resize) renders at the model input instead of full canvas.
    assert out["renderer_spec"] == [
        {"type": "init_frame"},
        {"type": "pad_to_square"},
        {"type": "resize", "size": _FakeEmbedder.input_size},
    ]


def test_track_ids_lists_run_tracks(tmp_path):
    recording = _setup_tracking(tmp_path)
    assert recording.track_ids("track_01") == [0]  # tracks.zarr has one track
    assert recording.track_ids("track_99") == []  # no tracks.zarr -> empty


def test_track_ids_caches_repeated_reads(tmp_path, monkeypatch):
    """A completed run's tracks.zarr is read once and cached (the GUI reads it per
    selected recording on every selection change) — and the returned list is a copy, so
    a caller mutating it cannot corrupt the cache."""
    from detail.api import recording as recording_api
    from detail.storage import read_cache

    read_cache.clear_read_cache()
    recording = _setup_tracking(tmp_path)
    real = recording_api.PartZarrReader
    calls = {"n": 0}

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(recording_api, "PartZarrReader", counting)
    assert recording.track_ids("track_01") == [0]
    assert recording.track_ids("track_01") == [0]
    assert calls["n"] == 1  # tracks.zarr opened once; the second read is cached

    ids = recording.track_ids("track_01")
    ids.append(999)  # caller mutation must not leak into the cache
    assert recording.track_ids("track_01") == [0]


def test_validate_all_tracks_resolves_to_every_track(tmp_path):
    from detail.pipeline.embedding import EmbeddingModule

    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    config = {
        "tracking_run_id": "track_01",
        "all_tracks": True,  # ignore track_ids, embed everything in the run
        "track_ids": [],
        "model_id": "fake_embedder",
        "constructor": "full_frame",
        "compute": ComputeTarget("cpu", None, None, None),
    }
    with _registered_embedder():
        out = EmbeddingModule().validate(recording.dir, data, config)
    assert out["track_ids"] == [0]  # resolved from tracks.zarr, per recording
    # The config model validates but must not alter/strip the stored record's fields.
    assert out["model_id"] == "fake_embedder"
    assert out["tracking_run_id"] == "track_01"
    assert (
        "renderer_spec" in out
    )  # resolved from the constructor, preserved through the model


def test_validate_without_constructor_or_spec_renders_the_default_constructor(
    tmp_path, monkeypatch
):
    from detail.pipeline import embedding as embedding_pipeline
    from detail.pipeline.embedding import EmbeddingModule
    from detail.transforms.constructors import DEFAULT_CONSTRUCTOR

    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "model_id": "fake_embedder",
        "compute": ComputeTarget("cpu", None, None, None),
    }
    resolved = []

    def fake_resolve(recording_dir, tracking_run_id, track_ids, constructor, **kwargs):
        resolved.append(constructor)
        return [{"type": "init_binary_mask"}]

    monkeypatch.setattr(embedding_pipeline, "resolve_renderer_spec", fake_resolve)
    with _registered_embedder():
        out = EmbeddingModule().validate(recording.dir, data, config)
    assert resolved == [DEFAULT_CONSTRUCTOR]
    assert (
        out["constructor"] == DEFAULT_CONSTRUCTOR
    )  # written back, so the record names it


def test_run_embedding_records_the_constructor_name(tmp_path, monkeypatch):
    # The record must carry the constructor name (a constructor-axis comparison reads it); the
    # resolved renderer_spec alone does not name it. run_embedding forwards it into the run config.
    recording = _make_recording(tmp_path)
    captured = {}

    def _capture(module_id, config):
        captured["module_id"] = module_id
        captured["config"] = config

    monkeypatch.setattr(recording, "run", _capture)
    recording.run_embedding(
        "track_01", [0], "m", EMB_RENDERER, constructor="full_frame"
    )
    assert captured["module_id"] == "embedding"
    assert captured["config"]["constructor"] == "full_frame"


def test_validate_allows_a_second_concurrent_embedding_run(tmp_path):
    # An already-running embedding no longer blocks a new one: outputs are per-run isolated and the
    # recording.yaml record is reserved/updated under the recording lock, so concurrent embeddings
    # (e.g. one per GPU) are safe. Reverting the guard removal makes this raise "already active".
    from detail.pipeline.embedding import EmbeddingModule

    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["embedding_runs"] = [{"id": "emb_01", "status": "running"}]
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "model_id": "fake_embedder",
        "constructor": "full_frame",
        "compute": ComputeTarget("cpu", None, None, None),
    }
    with _registered_embedder():
        out = EmbeddingModule().validate(recording.dir, data, config)
    assert (
        out["model_id"] == "fake_embedder"
    )  # validate passed the (removed) active-run guard


def test_run_embedding_unknown_model(tmp_path):
    recording = _setup_tracking(tmp_path)
    with pytest.raises(ValueError, match="unknown model_id"):
        recording.run_embedding("track_01", [0], "nope", EMB_RENDERER)


def test_run_embedding_unknown_transform(tmp_path):
    recording = _setup_tracking(tmp_path)
    with _registered_embedder():
        with pytest.raises(ValueError, match="unknown transform"):
            recording.run_embedding(
                "track_01", [0], "fake_embedder", [{"type": "bogus"}]
            )


def test_run_embedding_invalid_cuda_indices(tmp_path, monkeypatch):
    """compute.device_indices that exceed device_count raises before submit."""
    from detail.core.dataclasses import ComputeTarget

    recording = _setup_tracking(tmp_path)

    class _FakeTorch:
        class cuda:
            @staticmethod
            def device_count():
                return 1

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch)
    with _registered_embedder():
        with pytest.raises(ValueError, match="not available"):
            recording.run_embedding(
                "track_01",
                [0],
                "fake_embedder",
                EMB_RENDERER,
                compute=ComputeTarget("cuda", [0, 5], None, None),
            )


# --- run_render_preview (Section 14) ---------------------------------------- #
def test_run_render_preview_zero_samples_rejected(tmp_path):
    recording = _setup_tracking(tmp_path)
    with pytest.raises(ValueError, match="must be > 0"):
        recording.run_render_preview("track_01", [0], EMB_RENDERER, n_sample_frames=0)


def test_run_render_preview_tracking_not_found(tmp_path):
    recording = _setup_tracking(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        recording.run_render_preview("track_99", [0], EMB_RENDERER, n_sample_frames=2)


def test_run_render_preview_tracking_not_complete(tmp_path):
    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["tracking_runs"][0]["status"] = "running"
    recording_store.write_recording(recording.dir, data)
    with pytest.raises(ValueError, match="not complete"):
        recording.run_render_preview("track_01", [0], EMB_RENDERER, n_sample_frames=2)


def test_run_render_preview_track_id_missing(tmp_path):
    recording = _setup_tracking(tmp_path)
    with pytest.raises(ValueError, match="not in tracking run"):
        recording.run_render_preview("track_01", [5], EMB_RENDERER, n_sample_frames=2)


def test_run_render_preview_unknown_transform(tmp_path):
    recording = _setup_tracking(tmp_path)
    with pytest.raises(ValueError, match="unknown transform"):
        recording.run_render_preview(
            "track_01", [0], [{"type": "bogus"}], n_sample_frames=2
        )


# --- run dependents (deletion safety) --------------------------------------- #
def test_run_dependents_maps_downstream(tmp_path):
    recording = _make_recording(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete"},
        {"id": "det_02", "status": "complete"},
    ]
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    ]
    recording_store.write_recording(recording.dir, data)
    assert recording.run_dependents("detection", "det_01") == [("tracking", "track_01")]
    assert recording.run_dependents("detection", "det_02") == []  # nothing builds on it
    assert recording.run_dependents("embedding", "emb_01") == []  # terminal stage


def test_keypoint_runs_depend_on_their_detection_run_and_their_tracking_run(tmp_path):
    """§26 (2026-10-21): a keypoint detect run hangs off a detection run, and a keypoint track
    run reads a tracking run beside its detect run, so deleting either upstream must name it.
    """
    recording = _make_recording(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [{"id": "det_01", "status": "complete"}]
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    ]
    data["keypoint_detect_runs"] = [
        {"id": "kd_01", "status": "complete", "detection_run_id": "det_01"}
    ]
    data["keypoint_track_runs"] = [
        {
            "id": "kt_01",
            "status": "complete",
            "detect_run_id": "kd_01",
            "tracking_run_id": "track_01",
        }
    ]
    recording_store.write_recording(recording.dir, data)
    assert recording.run_dependents("detection", "det_01") == [
        ("keypoint_detect", "kd_01"),
        ("tracking", "track_01"),
    ]
    assert recording.run_dependents("tracking", "track_01") == [
        ("keypoint_track", "kt_01")
    ]
    assert recording.run_dependents("keypoint_detect", "kd_01") == [
        ("keypoint_track", "kt_01")
    ]


def test_dependency_edges_come_from_the_module_registry():
    """The edges a delete is checked against are derived from each module's ``lineage_ref``
    (and ``aux_refs``, §6.5), not a hand map: a hand map named one downstream per stage, and the
    second stage to read tracking (keypoint detect, §26) would have been orphaned silently.
    """
    from detail.api import recording as recording_api
    from detail.executor.runners import PROCESSING_MODULES

    assert recording_api._downstream_edges("detection") == [
        # §26 (2026-10-21): keypoint detect stores unassigned instances on a detection run
        ("keypoint_detect", "keypoint_detect_runs", "detection_run_id"),
        ("tracking", "tracking_runs", "detection_run_id"),
    ]
    assert recording_api._downstream_edges("tracking") == [
        ("embedding", "embedding_runs", "tracking_run_id"),
        # keypoint track assigns the instances to this tracking run's subjects (aux_refs)
        ("keypoint_track", "keypoint_track_runs", "tracking_run_id"),
    ]
    assert (
        recording_api._downstream_edges("embedding") == []
    )  # derive is a batch module
    # The guard: every module that declares a parent appears among that parent's edges, so a
    # new stage cannot be registered and left out of delete-safety.
    for module in PROCESSING_MODULES.values():
        refs = []
        if module.lineage_ref is not None:
            refs.append(module.lineage_ref)
        refs.extend(getattr(module, "aux_refs", ()))
        for parent, ref_field in refs:
            assert (module.module_id, module.runs_key, ref_field) in (
                recording_api._downstream_edges(parent)
            ), f"{module.module_id} reads {parent} but is not among its dependents"


def test_dependency_edges_fill_the_registry_before_reading_it(monkeypatch):
    """Nothing on the delete path imports the runner modules, so an empty registry would read
    as "no dependents" and let a delete through. ``_downstream_edges`` imports them first;
    this stands in for that import with one that refills the registry, and empties it.
    """
    from detail.api import recording as recording_api
    from detail.executor.runners import PROCESSING_MODULES

    saved = dict(PROCESSING_MODULES)
    calls = []

    def refill(modules):
        calls.append(tuple(modules))
        PROCESSING_MODULES.update(saved)

    monkeypatch.setattr(recording_api, "import_runner_modules", refill)
    PROCESSING_MODULES.clear()
    try:
        assert recording_api._downstream_edges("detection") == [
            ("keypoint_detect", "keypoint_detect_runs", "detection_run_id"),
            ("tracking", "tracking_runs", "detection_run_id"),
        ]
    finally:
        PROCESSING_MODULES.update(saved)
    assert calls and "detail.pipeline.tracking" in calls[0]


# --- run_downsampling (Section 4) ------------------------------------------ #
# --- get_frame ------------------------------------------------------------- #
def test_get_frame_returns_rgb_uint8(tmp_path):
    recording = _make_recording(tmp_path)
    n = _add_video(recording)
    frame = recording.get_frame(0, 0)
    assert frame.shape == (H, W, 3)
    assert frame.dtype == np.uint8
    # different frames decode to distinct content
    assert not np.array_equal(frame, recording.get_frame(0, n - 1))


def test_get_frame_bad_part_index_raises(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with pytest.raises(IndexError, match="part_index"):
        recording.get_frame(99, 0)


def test_get_frame_negative_frame_idx_raises(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with pytest.raises(IndexError, match="must be >= 0"):
        recording.get_frame(0, -1)


def test_get_frame_out_of_range_frame_idx_raises(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    with pytest.raises(IndexError, match="could not decode"):
        recording.get_frame(0, 9999)


# --- the whole pipeline through the public API, over real subprocesses ------ #
FAKE_MODELS_SOURCE = """
import numpy as np
from detail.core.dataclasses import Detection, FrameResult, TextCue
from detail.core.protocols import DetectionModel, EmbeddingModel
from detail.core.registries import register_embedding_model, register_model


@register_model("fake_detector")
class FakeDetector(DetectionModel):
    model_id = "fake_detector"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return FrameResult.from_detections(
            0, 0, [Detection(mask, (0.25, 0.25, 0.75, 0.75), "mouse", 0.9, None)]
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


@register_embedding_model("fake_embedder")
class FakeEmbedder(EmbeddingModel):
    model_id = "fake_embedder"
    output_dim = 8
    supported_backends = ["cpu"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def embed(self, image):
        return np.full(8, float(np.asarray(image).mean()), dtype=np.float32)

    def embed_batch(self, images):
        return np.stack(
            [
                np.full(8, float(np.asarray(img).mean()), dtype=np.float32)
                for img in images
            ],
            axis=0,
        )
"""


def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    mod_dir = tmp_path / "pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_models.py").write_text(FAKE_MODELS_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.setenv(
        "PYTHONPATH", str(mod_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    monkeypatch.setattr(
        api_recording,
        "_PIPELINE_RUNNERS",
        api_recording._PIPELINE_RUNNERS + ("fake_models",),
    )

    recording = _make_recording(tmp_path)
    # fully API-driven input side
    src = tmp_path / "raw.mp4"
    _write_video(src, n_frames=5)
    n = _count_frames(src)
    recording.add_video_part(src)

    det = recording.run_detection("fake_detector", _text_prompt(), _cpu()).wait()
    assert det.status == "complete"
    track = recording.run_tracking(
        det.id, "anchor_tracker", {"iou_threshold": 0.3}
    ).wait()
    assert track.status == "complete"
    emb = recording.run_embedding(track.id, [0], "fake_embedder", EMB_RENDERER).wait()
    assert emb.status == "complete"

    embz = PartZarrReader(
        recording.dir / "embeddings" / emb.id / "track_0" / "part_001.zarr"
    ).read()
    assert embz["embedding"].shape == (n, 8)  # one detection/frame, all track 0
    assert recording.embedding_runs[0]["status"] == "complete"


# --- add_video_part --------------------------------------------------------- #
def test_add_video_part(tmp_path):
    recording = _make_recording(tmp_path)
    src = tmp_path / "raw.mp4"
    _write_video(src, n_frames=5)
    n = _count_frames(src)
    recording.add_video_part(src)

    v = recording_store.read_recording(recording.dir)["video"]
    assert v["storage"] == "default"
    assert round(v["fps_container"]) == 30
    assert v["resolution"] == [W, H]
    assert len(v["files"]) == 1
    part = v["files"][0]
    assert part["file"] == "video/part_001.mp4"
    assert (recording.dir / "video" / "part_001.mp4").is_file()
    tsmap = recording_store.TimestampMap.from_csv(
        recording.dir / part["timestamps"]["generated"]
    )
    assert len(tsmap) == n
    assert tsmap.t_ns(0) == 0


def test_parts_join_in_time_whatever_each_file_starts_at(tmp_path, write_vfr_video):
    """Two parts are one strictly increasing clock across the join (Tenet 8).

    The first file starts its PTS at 3 s and the second at 0. Before maps were
    re-based the first part kept its 3 s start while the second was offset only by
    the first's *duration*, so the second began 1 s before the first ended. The
    offset is integer ns from the maps, the arithmetic ``reorder_video_parts`` uses,
    so putting the parts in the order they already have rewrites nothing.
    """
    recording = _make_recording(tmp_path)
    recording.add_video_part(write_vfr_video(tmp_path / "a.mp4", start_ms=3000))
    recording.add_video_part(write_vfr_video(tmp_path / "b.mp4"))

    data = recording_store.read_recording(recording.dir)
    first, second = (
        recording_store.load_active_timestamps(recording.dir, data, i).values
        for i in (0, 1)
    )
    assert first[0] == 0
    assert second[0] == video_store.part_duration_ns(first, 50.0)
    assert (np.diff(np.concatenate([first, second])) > 0).all()

    csvs = [
        recording.dir / f["timestamps"]["generated"] for f in data["video"]["files"]
    ]
    before = [c.read_bytes() for c in csvs]
    recording_store.reorder_video_parts(recording.dir, [0, 1])
    assert [c.read_bytes() for c in csvs] == before


def test_add_video_part_missing_file_raises(tmp_path):
    recording = _make_recording(tmp_path)
    with pytest.raises(FileNotFoundError):
        recording.add_video_part(tmp_path / "nope.mp4")


# --- add_video_part: reference in place (Section 2) ------------------------- #
def _register_root(tmp_path, monkeypatch, alias="nas"):
    """Isolate storage.yaml to tmp and register a video root at <tmp>/external."""
    from detail.storage import locations, video_roots

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    root = tmp_path / "external"
    root.mkdir(exist_ok=True)
    video_roots.register_video_root(alias, root)
    return root


def test_add_video_part_reference_in_place(tmp_path, monkeypatch):
    root = _register_root(tmp_path, monkeypatch)
    src = root / "day3" / "m27.mp4"
    src.parent.mkdir()
    _write_video(src, n_frames=5)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")

    v = recording_store.read_recording(recording.dir)["video"]
    assert v["storage"] == "reference"
    part = v["files"][0]
    assert "file" not in part  # no recording-relative copy
    assert part["source"] == {"root": "nas", "rel": "day3/m27.mp4"}
    assert part["frame_count"] == 5
    assert set(part["fingerprint"]) == {"size", "sha256_ends", "mtime_ns"}
    # the video is NOT copied into the recording, but the timestamp map still is
    assert not (recording.dir / "video" / "part_001.mp4").exists()
    assert (recording.dir / part["timestamps"]["generated"]).is_file()
    # the seam resolves the part to the external file, and presence works through it
    resolved = recording_store.resolve_part_video_path(recording.dir, part)
    assert resolved.samefile(src)
    assert recording.video_available


def test_reference_presence_detects_a_moved_file(tmp_path, monkeypatch):
    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    assert recording.video_available
    src.unlink()  # the user moved/deleted the external file
    assert (
        not recording.video_available
    )  # detected through the resolver, not a stale copy


def test_reference_requires_a_registered_root(tmp_path, monkeypatch):
    _register_root(tmp_path, monkeypatch)  # registers "nas", not "ghost"
    src = tmp_path / "external" / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    with pytest.raises(ValueError, match="not registered"):
        recording.add_video_part(src, storage="reference", video_root="ghost")


def test_reference_rejects_a_video_outside_the_root(tmp_path, monkeypatch):
    _register_root(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere" / "m.mp4"
    outside.parent.mkdir()
    _write_video(outside, n_frames=4)
    recording = _make_recording(tmp_path)
    with pytest.raises(ValueError, match="not under video root"):
        recording.add_video_part(outside, storage="reference", video_root="nas")


def test_reference_auto_remembers_the_video_folder(tmp_path, monkeypatch):
    # No video_root given (the default): the tool remembers the video's own folder — the
    # caller references a path, never picks a location.
    from detail.storage import locations, video_roots

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    folder = tmp_path / "library" / "day3"
    folder.mkdir(parents=True)
    src = folder / "m.mp4"
    _write_video(src, n_frames=5)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference")  # no video_root → auto

    part = recording_store.read_recording(recording.dir)["video"]["files"][0]
    assert part["source"]["rel"] == "m.mp4"  # relative to the auto-remembered folder
    assert video_roots.video_root(part["source"]["root"]) == folder.resolve()
    assert recording.video_available
    assert not (recording.dir / "video" / "part_001.mp4").exists()  # not copied


# --- verify / relink / the pre-run gate (Section 2) ------------------------- #
def test_verify_videos_reports_a_changed_reference(tmp_path, monkeypatch):
    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    assert recording.verify_videos() == []
    with open(src, "ab") as fh:
        fh.write(b"tampered")  # same path, new content
    issues = recording.verify_videos()
    assert [i.kind for i in issues] == ["changed_fingerprint"]


def test_relink_accepts_a_moved_file(tmp_path, monkeypatch):
    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    moved = root / "sub" / "m2.mp4"
    moved.parent.mkdir()
    src.rename(moved)  # moved within the same root, content unchanged
    assert not recording.video_available
    assert recording.relink_video_part(0, moved) is None  # nothing newly registered
    part = recording_store.read_recording(recording.dir)["video"]["files"][0]
    assert part["source"] == {"root": "nas", "rel": "sub/m2.mp4"}
    assert recording.video_available


def _forget_video_roots(tmp_path, monkeypatch):
    """Point storage.yaml at an empty one — the experiment is now on another machine."""
    from detail.storage import locations

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg2")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg2" / "storage.yaml")


def test_relink_registers_the_parts_root_when_this_machine_lacks_it(
    tmp_path, monkeypatch
):
    # An experiment made on another machine names a root this one has never heard of. One
    # Locate registers *that* root where the file shows it lives, so every other recording
    # referencing it resolves without a Locate of its own.
    from detail.storage import video_roots

    root = _register_root(tmp_path, monkeypatch)
    experiment = Experiment.create("proj", tmp_path)
    recs = [experiment.create_recording(f"recording_00{i}") for i in (1, 2)]
    for i, rec in enumerate(recs):
        src = root / "day1" / f"m{i}.mp4"
        _write_video(src, n_frames=4)
        rec.add_video_part(src, storage="reference", video_root="nas")
    _forget_video_roots(tmp_path, monkeypatch)
    elsewhere = tmp_path / "mounted" / "library"
    elsewhere.parent.mkdir()
    root.rename(elsewhere)
    assert [r.verify_videos()[0].kind for r in recs] == ["missing_root"] * 2

    assert recs[0].relink_video_part(0, elsewhere / "day1" / "m0.mp4") == "nas"

    assert video_roots.video_root("nas").resolve() == elsewhere.resolve()
    part = recording_store.read_recording(recs[0].dir)["video"]["files"][0]
    assert part["source"] == {"root": "nas", "rel": "day1/m0.mp4"}
    assert [r.verify_videos() for r in recs] == [[], []]


def test_relink_registers_the_new_folder_when_no_root_holds_it(tmp_path, monkeypatch):
    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    moved = tmp_path / "moved" / "m.mp4"
    moved.parent.mkdir()
    src.rename(moved)  # out of every registered root

    assert recording.relink_video_part(0, moved) == "moved"

    part = recording_store.read_recording(recording.dir)["video"]["files"][0]
    assert part["source"] == {"root": "moved", "rel": "m.mp4"}
    assert recording.verify_videos() == []


def test_relink_of_a_renamed_file_under_an_unregistered_root_leaves_that_root_alone(
    tmp_path, monkeypatch
):
    # The file no longer sits at the root's recorded ``rel``, so its location says nothing
    # about where the root is. Registering the root there would misplace every sibling.
    from detail.storage import video_roots

    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    _forget_video_roots(tmp_path, monkeypatch)
    renamed = tmp_path / "moved" / "renamed.mp4"
    renamed.parent.mkdir()
    src.rename(renamed)

    assert recording.relink_video_part(0, renamed) == "moved"

    assert video_roots.video_root("nas") is None
    assert recording.verify_videos() == []


def test_relink_refuses_a_changed_file(tmp_path, monkeypatch):
    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=4)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    other = root / "different.mp4"
    _write_video(other, n_frames=7)  # a different video
    with pytest.raises(ValueError, match="different content"):
        recording.relink_video_part(0, other)


def test_relink_refuses_a_copied_part(tmp_path):
    recording = _make_recording(tmp_path)
    src = tmp_path / "raw.mp4"
    _write_video(src, n_frames=4)
    recording.add_video_part(src)  # copied
    with pytest.raises(ValueError, match="copied part"):
        recording.relink_video_part(0, src)


def test_run_gate_rejects_a_changed_reference_video(tmp_path, monkeypatch):
    # The hard pre-run gate (a detection run's validate) refuses to start on a changed
    # referenced video — the "never process changed data as good data" guarantee.
    from detail.pipeline.detection import _validate_source_files

    root = _register_root(tmp_path, monkeypatch)
    src = root / "m.mp4"
    _write_video(src, n_frames=6)
    recording = _make_recording(tmp_path)
    recording.add_video_part(src, storage="reference", video_root="nas")
    with open(src, "ab") as fh:
        fh.write(b"tampered")
    data = recording_store.read_recording(recording.dir)
    # The video gate itself, as detection's validate calls it — before any model is
    # resolved, so the guarantee holds on a box with no GPU.
    with pytest.raises(ValueError, match="content changed"):
        _validate_source_files(recording.dir, data)


def test_add_video_part_reports_progress_stages(tmp_path):
    recording = _make_recording(tmp_path)
    src = tmp_path / "raw.mp4"
    _write_video(src, n_frames=5)
    stages = []
    recording.add_video_part(src, progress=lambda s, f, d: stages.append((s, f)))
    names = [s for s, _ in stages]
    assert "Copying" in names  # full import reports the copy stage
    assert stages[-1][1] == 1.0  # copy finishes at fraction 1.0


def test_add_video_part_multipart_global_timestamps(tmp_path):
    recording = _make_recording(tmp_path)
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write_video(a, n_frames=5)
    _write_video(b, n_frames=4)
    recording.add_video_part(a)
    recording.add_video_part(b)

    v = recording_store.read_recording(recording.dir)["video"]
    assert [f["file"] for f in v["files"]] == [
        "video/part_001.mp4",
        "video/part_002.mp4",
    ]
    t1 = recording_store.TimestampMap.from_csv(
        recording.dir / v["files"][0]["timestamps"]["generated"]
    )
    t2 = recording_store.TimestampMap.from_csv(
        recording.dir / v["files"][1]["timestamps"]["generated"]
    )
    # part 2 timestamps are global — they start after part 1 ends
    assert t2.t_ns(0) > t1.t_ns(len(t1) - 1)


def test_video_parts_accessor(recording_dir):
    sess = api_recording.Recording(recording_dir.parent.parent, "recording_001")
    parts = sess.video_parts
    assert len(parts) == 1
    assert parts[0]["file"] == "video/part_001.mp4"


def test_part_frame_counts(recording_dir):
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(7, dtype=np.int64) * 1000,
    )
    sess = api_recording.Recording(recording_dir.parent.parent, "recording_001")
    assert sess.part_frame_counts == [7]


def test_set_name_persists(recording_dir):
    sess = api_recording.Recording(recording_dir.parent.parent, "recording_001")
    sess.set_name("Mouse A")
    assert sess.name == "Mouse A"
    assert (
        api_recording.Recording(recording_dir.parent.parent, "recording_001").name
        == "Mouse A"
    )


def test_reorder_video_parts_blocked_after_runs(recording_dir):
    sess = api_recording.Recording(recording_dir.parent.parent, "recording_001")
    data = recording_store.read_recording(sess.dir)
    data["detection_runs"] = [{"id": "det_01", "status": "complete"}]
    recording_store.write_recording(sess.dir, data)
    with pytest.raises(ValueError, match="cannot reorder"):
        sess.reorder_video_parts([0])


# --- video availability (video-excluded import handling, §15) --------------- #
def test_video_available(tmp_path):
    recording = _make_recording(tmp_path)
    assert recording.video_available is False  # no video imported
    _add_video(recording)
    assert recording.video_available is True
    for mp4 in (recording.dir / "video").glob("*.mp4"):  # video-excluded import
        mp4.unlink()
    assert recording.video_available is False  # entry kept, blob gone


def test_get_frame_missing_video_raises_filenotfound(tmp_path):
    recording = _make_recording(tmp_path)
    _add_video(recording)
    for mp4 in (recording.dir / "video").glob("*.mp4"):
        mp4.unlink()
    with pytest.raises(FileNotFoundError, match="video not present"):
        recording.get_frame(0, 0)


def test_import_calibration_is_refused_onto_a_video_of_another_shape(tmp_path):
    """Both carry-over doors refuse a change of shape, and leave nothing behind."""
    exp = Experiment.create("proj", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.set_distance_calibration([(0.5, 0.25), (0.5, 0.75)], 200.0)
    out = tmp_path / "setup.csv"
    src.export_spatial_csv(out)
    dst = exp.create_recording("r2")
    _set_resolution(dst, 1280, 1024)

    with pytest.raises(ValueError, match="different video shape"):
        dst.import_calibration_from(src)
    with pytest.raises(ValueError, match="different video shape"):
        dst.import_calibration_csv(out)
    assert dst.distance_calibration is None


def test_a_calibration_whose_video_changed_shape_is_refused_on_read(tmp_path):
    """The video can be replaced after the calibration is set; the read refuses too."""
    exp = Experiment.create("proj", tmp_path)
    rec = exp.create_recording("r1")
    _set_resolution(rec, 1920, 1080)
    rec.set_pixels_per_mm(5.0)
    assert rec.calibration_refusal() is None
    _set_resolution(rec, 1080, 1920)

    assert "1080x1920" in rec.calibration_refusal()
    with pytest.raises(ValueError, match="different video shape"):
        rec.pixels_per_mm


def test_embedding_validate_asks_the_install_whether_it_runs_on_the_backend(tmp_path):
    """``Model.check_backend`` (§5) is the install's answer, after ``supported_backends``
    gave the class's: an embedder that supports cpu in principle but cannot run on it on this
    machine refuses at validate, with its own reason."""
    from detail.core.registries import EMBEDDING_MODELS
    from detail.pipeline.embedding import EmbeddingModule

    class _CpuRefusing(_FakeEmbedder):
        model_id = "cpu_refusing_embedder"

        def check_backend(self, backend):
            raise ValueError(f"this install of {self.model_id} cannot run on {backend}")

    recording = _setup_tracking(tmp_path)
    data = recording_store.read_recording(recording.dir)
    config = {
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "model_id": "cpu_refusing_embedder",
        "constructor": "full_frame",
        "compute": ComputeTarget("cpu", None, None, None),
    }
    EMBEDDING_MODELS["cpu_refusing_embedder"] = _CpuRefusing
    try:
        with pytest.raises(ValueError, match="cannot run on cpu"):
            EmbeddingModule().validate(recording.dir, data, config)
    finally:
        EMBEDDING_MODELS.pop("cpu_refusing_embedder", None)


def test_the_keypoint_run_lists_are_read_off_the_live_recording(tmp_path):
    """§26: `run_spec._recording_run_data` reads a module's runs by ``runs_key`` off the live
    Recording, so without `keypoint_detect_runs` / `keypoint_track_runs` every picker of a
    keypoint run listed nothing in the app, while tests that hand the dicts in passed.
    """
    from detail.api import run_spec
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.api.experiment import Experiment
    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
    from detail.storage import recording as recording_store

    import_runner_modules(PIPELINE_RUNNERS)
    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("recording_001")
    detect_identity = {
        f: "x" for f in PROCESSING_MODULES["keypoint_detect"].identity_fields
    }
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"] = [
        {
            "id": "kd_01",
            "status": "complete",
            **detect_identity,
            "model_id": "dinov3_match",
        }
    ]
    data["keypoint_track_runs"] = [
        {
            "id": "kt_01",
            "status": "complete",
            "detect_run_id": "kd_01",
            "method": "temporal_decode",
            "config": {"despike": True},
        }
    ]
    recording_store.write_recording(recording.dir, data)
    assert recording.keypoint_detect_runs[0]["id"] == "kd_01"
    assert recording.keypoint_track_runs[0]["id"] == "kt_01"
    for module_id, run_id in (
        ("keypoint_detect", "kd_01"),
        ("keypoint_track", "kt_01"),
    ):
        groups = run_spec.group_specs([recording], module_id)
        assert len(groups) == 1 and groups[0]["recordings"] == [
            "recording_001"
        ], module_id
        assert (
            run_spec.resolve_spec_for_recording(recording, module_id, groups[0]["spec"])
            == run_id
        )
