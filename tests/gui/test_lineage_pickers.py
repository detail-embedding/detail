"""A form with two upstream pickers offers only the pairs its stage accepts.

**The class: two inputs of one stage that must share a run, picked independently.** Keypoint
track takes a keypoint detect run and a tracking run, and gives the poses only to tracks of the
detections they were placed on; embedding takes a tracking run and a keypoint track run whose
poses went to it; track merge takes tracking runs of one detection run. Each stage checks this per
recording in ``validate`` and refuses the pair, correctly — but each form listed every run in the
selection beside every other, so it offered pairs it would then refuse. Reported: keypoint detect
``run_011`` (on ``run_001``'s detection) offered beside tracking ``run_010`` (a merge on
``run_007``'s), the two detection runs told apart only as "from run_001" / "from run_007" deep in
long labels, and the launch refused with "tracking run 'track_05' is on detection run 'det_03', but
keypoint detect run 'kd_03' is on 'det_01'".

How it hid: every picker was right about its own input, and the rule lives in the *pair*. It is a
fact about runs, not settings — the same run is ``det_03`` in one recording and ``det_02`` in the
next — so it is compared recording by recording (``_common.spec_reads`` / ``lineage_fits``), and an
option that fits on none is greyed with the reason by ``UpstreamSpecPicker.restrict``.

**The default half: a tie broken by list order.** When two settings covered the same recordings a
picker opened on the first by label, and a label leads with its experiment run, so of two detection
runs of one model it opened on ``run_001`` — the older — and the keypoint detect run above went onto
it without anyone choosing it. A tie now goes to the newest (``_common.widest_group``), and a plain
combo refilled by ``repopulate_spec_combo`` must name its ``fallback``.

**The next-step half: one input preselected, the other left to its default.** A *Run embedding →*
from a keypoint track run preselects that run, and the tracking row opened on its own default, where
the preselected row is greyed. The preselected run's spec names its partner (a keypoint track run's
``aux.tracking``, a tracking run's ``parent`` detection), and the form opens the other picker there.

The guard drives each form with the stage's own check where the stage exposes one
(``keypoint_track._resolve_tracking_run``), over every pair the form leaves enabled, so a pair
offered and refused fails here rather than at launch.

Swept (``UpstreamSpecPicker(`` / ``PooledSpecPicker(`` / ``repopulate_spec_combo(`` under
``detail/gui``): keypoint track, embedding and track merge fixed; keypoint detect's *Test on
tracking* already filtered to the chosen detection run (its own copy, left: it hides rather than
greys, because the row only appears when a recording is ambiguous); keypoint distill and the
analysis forms take one upstream each, so they have no pair.
"""

from types import SimpleNamespace

import pytest

from detail.api import run_spec
from detail.api.experiment import Experiment
from detail.gui.modules.registry import GUI_MODULES
from detail.pipeline.keypoint_track import _resolve_tracking_run
from detail.storage import recording as recording_store

# Two detection runs of one model, differing only in a class threshold — the reported pair.
_OLD, _NEW = "2026-10-30T22:42:00Z", "2026-11-03T17:48:00Z"
# (recording, {experiment run: per-recording detection run id}): the run_007 detection is
# det_03 in one recording and det_02 in the next, as in the reported experiment.
_RECORDINGS = (
    ("recording_017", {"run_001": "det_01", "run_007": "det_03"}),
    ("recording_018", {"run_001": "det_01", "run_007": "det_02"}),
)


def _detection(run_id: str, processing_run: str, threshold_fp: str, created: str):
    return {
        "id": run_id,
        "status": "complete",
        "processing_run_id": processing_run,
        "model_id": "model_001",
        "model_variant": None,
        "weights_source": "local",
        "conf_threshold": 0.3,
        "prompt_fingerprint": "p",
        "prompt_classes": ["Intruder", "Resident"],
        "class_conf_thresholds_fingerprint": threshold_fp,
        "output": f"detections/{run_id}/",
        "created_at": created,
    }


def _tracking(
    run_id, processing_run, detection_run_id, tracker_id, class_name, created
):
    return {
        "id": run_id,
        "status": "complete",
        "processing_run_id": processing_run,
        "detection_run_id": detection_run_id,
        "tracker_id": tracker_id,
        "arena_ids": [],
        "class_name": class_name,
        "hyperparams": {},
        "every_n_frames": 1,
        "output": f"tracking/{run_id}/",
        "created_at": created,
    }


def _keypoint_detect(run_id, processing_run, detection_run_id, created):
    identity = {
        "model_id": "dinov3_match",
        "model_variant": "base",
        "weights_source": "hf",
        "backbone_id": None,
        "backbone_variant": None,
        "skeleton_digest": "s",
        "example_set_digest": "e",
        "n_candidates": 4,
        "every_n_frames": 1,
        "crop_px": 256,
        "decode": "argmax",
        "side_rule": "none",
        "matching": "greedy",
        "min_pose_confidence": 0.0,
    }
    return {
        "id": run_id,
        "status": "complete",
        "processing_run_id": processing_run,
        "detection_run_id": detection_run_id,
        "output": f"keypoint_detect/{run_id}/",
        "created_at": created,
        **identity,
    }


def _keypoint_track(run_id, processing_run, detect_run_id, tracking_run_id, created):
    return {
        "id": run_id,
        "status": "complete",
        "processing_run_id": processing_run,
        "detect_run_id": detect_run_id,
        "tracking_run_id": tracking_run_id,
        "tracking_identity": "t",
        "match": {"min_inside": 0.5, "mask_grow": 0},
        "method": "temporal_decode",
        "config": {},
        "output": f"keypoint_track/{run_id}/",
        "created_at": created,
    }


@pytest.fixture
def two_lineages(tmp_path):
    """Two recordings, each with both detection runs and, on each, a tracking run, a keypoint
    detect run and a keypoint track run — the reported experiment's shape."""
    experiment = Experiment.create("p", tmp_path)
    recordings = []
    for name, det in _RECORDINGS:
        recording = experiment.create_recording(name)
        old, new = det["run_001"], det["run_007"]
        data = recording_store.read_recording(recording.dir)
        data["detection_runs"] = [
            _detection(old, "run_001", "a", _OLD),
            _detection(new, "run_007", "b", _NEW),
        ]
        data["tracking_runs"] = [
            _tracking("track_01", "run_002", old, "anchor_tracker", "Intruder", _OLD),
            _tracking("track_02", "run_003", old, "anchor_tracker", "Resident", _OLD),
            _tracking(
                "track_05", "run_010", new, "track_merge", "Intruder+Resident", _NEW
            ),
        ]
        data["keypoint_detect_runs"] = [
            _keypoint_detect("kd_03", "run_011", old, _OLD),
            _keypoint_detect("kd_04", "run_012", new, _NEW),
        ]
        data["keypoint_track_runs"] = [
            _keypoint_track("kt_01", "run_013", "kd_03", "track_01", _OLD),
            _keypoint_track("kt_02", "run_014", "kd_04", "track_05", _NEW),
        ]
        recording_store.write_recording(recording.dir, data)
        recordings.append(recording)
    return recordings


def _rows(combo):
    """``[(index, spec, enabled)]`` for a combo's spec rows."""
    model = combo.model()
    return [
        (i, combo.itemData(i), model.item(i).isEnabled())
        for i in range(combo.count())
        if isinstance(combo.itemData(i), dict)
    ]


def _own_run(recording, module_id: str, spec: dict) -> dict:
    run_id = run_spec.resolve_spec_for_recording(recording, module_id, spec)
    data = recording_store.read_recording(recording.dir)
    return next(r for r in data[f"{module_id}_runs"] if r["id"] == run_id)


def _form(module_id: str, recordings):
    """A module's form as the launch dialog fills it: both halves, each read off-thread."""
    form = GUI_MODULES[module_id].config_widget()
    form.apply_eligible_recordings(
        recordings, form.resolve_eligible_recordings(recordings)
    )
    form.apply_recordings(recordings, form.resolve_recordings(recordings))
    return form


def test_keypoint_track_offers_only_tracking_the_stage_accepts(qapp, two_lineages):
    """Every tracking row left enabled, beside every detect run, passes the stage's own
    check on every recording the form would launch; the others are greyed, saying why.
    """
    form = _form("keypoint_track", two_lineages)
    by_id = {r.id: r for r in two_lineages}
    detect_rows = _rows(form._detect)
    assert len(detect_rows) == 2
    greyed = 0
    for detect_index, detect_spec, _ in detect_rows:
        form._detect.setCurrentIndex(detect_index)
        for index, tracking_spec, enabled in _rows(form._tracking):
            if not enabled:
                greyed += 1
                tip = form._tracking.itemData(index, 3)  # Qt.ItemDataRole.ToolTipRole
                assert "another detection run" in tip
                continue
            form._tracking.setCurrentIndex(index)
            covered = form.covered_recording_ids()
            assert covered, form._tracking.itemText(index)
            for recording_id in covered:
                recording = by_id[recording_id]
                detect = _own_run(recording, "keypoint_detect", detect_spec)
                tracking = _own_run(recording, "tracking", tracking_spec)
                _resolve_tracking_run(  # raises on the mismatched pair
                    recording.dir,
                    recording_store.read_recording(recording.dir),
                    SimpleNamespace(
                        detection_run_id=detect["detection_run_id"],
                        run_id=detect["id"],
                    ),
                    tracking["id"],
                )
    # Each detect run greys the other lineage's rows: the merge for run_011, the two
    # single-class runs for run_012 — so the loop above was not vacuously green.
    assert greyed == 3


def test_keypoint_track_refilters_when_the_detect_run_changes(qapp, two_lineages):
    """The reported pair: run_011 greys run_010 (the merge on run_007's detection), and
    choosing run_012 instead offers it."""
    form = _form("keypoint_track", two_lineages)

    def enabled_trackers():
        return {
            spec["identity"]["tracker_id"]
            for _i, spec, on in _rows(form._tracking)
            if on
        }

    def choose_detect(processing_run):
        for index in range(form._detect.count()):
            if form._detect.itemText(index).startswith(processing_run):
                form._detect.setCurrentIndex(index)
                return
        raise AssertionError(f"no {processing_run} row")

    choose_detect("run_011")
    assert enabled_trackers() == {"anchor_tracker"}
    choose_detect("run_012")
    assert enabled_trackers() == {"track_merge"}


def test_embedding_offers_only_keypoint_runs_on_its_tracking_run(qapp, two_lineages):
    """The embedding's keypoint track row follows its tracking run: a keypoint track run
    whose poses went to another tracking run is greyed (§14 *Lineage* refuses it)."""
    form = _form("embedding", two_lineages)
    by_id = {r.id: r for r in two_lineages}
    checked = 0
    for t_index in range(form._tracking.count()):
        tracking_spec = form._tracking.itemData(t_index)
        form._tracking.setCurrentIndex(t_index)
        for index, kt_spec, enabled in _rows(form._keypoint):
            for recording in two_lineages:
                tracking = _own_run(recording, "tracking", tracking_spec)
                kt = _own_run(recording, "keypoint_track", kt_spec)
                fits = kt["tracking_run_id"] == tracking["id"]
                assert enabled == fits, (form._tracking.itemText(t_index), kt["id"])
                checked += 1
            if enabled:
                form._keypoint.setCurrentIndex(index)
                assert form._keypoint_picker.covered() == set(by_id)
    assert checked == 3 * 2 * 2


def test_track_merge_offers_track_b_only_on_track_as_detection_run(qapp, two_lineages):
    """Track B follows Track A: the stage merges tracks of one detection run only."""
    form = _form("track_merge", two_lineages)
    a, b = form._rows
    for a_index in range(a.count()):
        a_spec = a.itemData(a_index)
        a.setCurrentIndex(a_index)
        for _index, b_spec, enabled in _rows(b):
            for recording in two_lineages:
                same = (
                    _own_run(recording, "tracking", a_spec)["detection_run_id"]
                    == _own_run(recording, "tracking", b_spec)["detection_run_id"]
                )
                assert enabled == same
    # The merge alone is on run_007's detection, so with it as Track A, Track B has no
    # other source and the form says so rather than "run tracking first".
    merge = next(
        i for i in range(a.count()) if "track_merge" in str(a.itemData(i)["identity"])
    )
    a.setCurrentIndex(merge)
    b.setCurrentIndex(next(i for i, _s, on in _rows(b) if on))
    assert "more than once" in form.validation_error()


@pytest.mark.parametrize(
    ("module_id", "upstream", "run_id"),
    [
        # From the older tracking run: the detect picker leaves its newest-first default.
        ("keypoint_track", "tracking", "track_01"),
        # From the older keypoint track run: the tracking row moves to its tracking run.
        ("embedding", "keypoint_track", "kt_01"),
    ],
)
def test_a_next_step_launch_opens_the_other_half_of_its_pair(
    qapp, two_lineages, module_id, upstream, run_id
):
    """A next-step launch preselects one input of the pair; the other opens on its match
    rather than on its own default, where the preselected row would be greyed."""
    recording = two_lineages[0]
    want = run_spec.spec_from_run(recording, upstream, run_id)
    form = GUI_MODULES[module_id].config_widget()
    form.prefer_upstream_spec(upstream, want)
    form.apply_eligible_recordings(
        two_lineages, form.resolve_eligible_recordings(two_lineages)
    )
    form.apply_recordings(two_lineages, form.resolve_recordings(two_lineages))
    picker = (
        form._tracking_picker
        if module_id == "keypoint_track"
        else form._keypoint_picker
    )
    chosen = picker.spec()
    assert run_spec.freeze_spec(chosen) == run_spec.freeze_spec(want)
    assert form.covered_recording_ids() == {r.id for r in two_lineages}


@pytest.mark.parametrize(
    "picker_kind", ["upstream", "pooled", "tracking_form", "keypoint_detect_form"]
)
def test_a_coverage_tie_opens_on_the_newest_run(qapp, two_lineages, picker_kind):
    """Two detection runs of one model over the same recordings: every picker opens on the
    newer (run_007), not on whichever label sorts first (run_001)."""
    from detail.gui.modules._common import PooledSpecPicker, UpstreamSpecPicker

    if picker_kind == "upstream":
        picker = UpstreamSpecPicker("detection")
        picker.apply(picker.resolve(two_lineages, {}), len(two_lineages))
        chosen = picker.spec()
    elif picker_kind == "pooled":
        picker = PooledSpecPicker("detection")
        picker.apply(picker.resolve(two_lineages, {}), len(two_lineages), names={})
        (chosen,) = picker.specs()
    elif picker_kind == "tracking_form":
        chosen = _form("tracking", two_lineages)._detection.currentData()
    else:
        chosen = _form("keypoint_detect", two_lineages)._picker.spec()
    assert chosen["identity"]["class_conf_thresholds_fingerprint"] == "b"
