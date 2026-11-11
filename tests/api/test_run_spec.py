"""Run-spec identity (§6.5 "Upstream run specs"): equivalence, distinctness,
normalization, grouping, per-recording resolution, and the legacy job.json fallback."""

import json
import types

import pytest

from detail.api import run_spec
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    Prompt,
    PromptTarget,
    TextCue,
    TrackingJob,
)
from detail.executor import rundir
from detail.executor.runners import PROCESSING_MODULES, import_runner_modules
from detail.pipeline.detection import prompt_classes, prompt_fingerprint


@pytest.fixture(autouse=True)
def _runner_modules():
    import_runner_modules(PIPELINE_RUNNERS)
    run_spec._LEGACY_CACHE.clear()
    yield
    run_spec._LEGACY_CACHE.clear()


def _det(rid, **over):
    record = {
        "id": rid,
        "status": "complete",
        "output": f"detections/{rid}/",
        "created_at": "2026-01-01T00:00",
        "model_id": "sam3",
        "model_variant": "large",
        "weights_source": "pretrained",
        "conf_threshold": 0.25,
        "prompt_fingerprint": "abc123",
        "prompt_classes": ["mouse"],
    }
    record.update(over)
    return record


def _track(rid, detection_run_id="det_01", **over):
    record = {
        "id": rid,
        "status": "complete",
        "output": f"tracking/{rid}/",
        "created_at": "2026-01-01T00:00",
        "detection_run_id": detection_run_id,
        "tracker_id": "anchor_tracker",
        "arena_ids": [],
        "hyperparams": {
            "conf_threshold": 0.5,
            "iou_threshold": 0.5,
            "max_missed_frames": 5,
        },
    }
    record.update(over)
    return record


def _key(recording_data, module_id, record, recording_dir=None):
    return run_spec.freeze_spec(
        run_spec.spec_of(recording_data, module_id, record, recording_dir)
    )


# --------------------------------------------------------------------------- #
# Equivalence: the same settings under different per-recording ordinals
# --------------------------------------------------------------------------- #
def test_spec_equivalence_across_ordinals():
    data_a = {"detection_runs": [_det("det_02")]}
    data_b = {"detection_runs": [_det("det_01")]}
    assert _key(data_a, "detection", _det("det_02")) == _key(
        data_b, "detection", _det("det_01")
    )


# --------------------------------------------------------------------------- #
# Distinctness — the three error cases the user named
# --------------------------------------------------------------------------- #
def test_distinct_weights_source():
    a = _key({}, "detection", _det("det_01", weights_source="pretrained"))
    b = _key({}, "detection", _det("det_01", weights_source="trained"))
    assert a != b


def test_distinct_tracker_conf():
    data = {"detection_runs": [_det("det_01")]}
    a = _key(data, "tracking", _track("t1", hyperparams={"conf_threshold": 0.3}))
    b = _key(data, "tracking", _track("t2", hyperparams={"conf_threshold": 0.7}))
    assert a != b


def test_distinct_detection_lineage():
    data = {
        "detection_runs": [
            _det("det_01", model_id="sam3"),
            _det("det_02", model_id="model_001"),
        ]
    }
    spec_a = run_spec.spec_of(data, "tracking", _track("t1", detection_run_id="det_01"))
    spec_b = run_spec.spec_of(data, "tracking", _track("t2", detection_run_id="det_02"))
    # identical tracker settings — only the upstream detection differs
    assert spec_a["identity"] == spec_b["identity"]
    assert run_spec.freeze_spec(spec_a) != run_spec.freeze_spec(spec_b)


# --------------------------------------------------------------------------- #
# The two false-merge fixes: prompt (detection) and arena (tracking)
# --------------------------------------------------------------------------- #
def test_distinct_prompt_fingerprint():
    a = _key({}, "detection", _det("det_01", prompt_fingerprint="aaa"))
    b = _key({}, "detection", _det("det_01", prompt_fingerprint="bbb"))
    assert a != b


def test_prompt_fingerprint_distinguishes_prompts():
    mouse = Prompt(
        targets=[PromptTarget(class_name="mouse", cue=TextCue(text="mouse"))]
    )
    cage = Prompt(targets=[PromptTarget(class_name="cage", cue=TextCue(text="cage"))])
    assert prompt_fingerprint(mouse) != prompt_fingerprint(cage)
    assert prompt_fingerprint(mouse) == prompt_fingerprint(
        Prompt(targets=[PromptTarget(class_name="mouse", cue=TextCue(text="mouse"))])
    )
    assert prompt_fingerprint(None) == prompt_fingerprint(None)


def test_detection_label_shows_prompt_classes():
    """Same model + confidence, different prompt → distinguishable in the picker label
    (not just in the opaque spec key). Regression: the label used to omit the prompt,
    so two prompts rendered identically."""
    mouse = run_spec.spec_of(
        {}, "detection", _det("d1", prompt_fingerprint="h1", prompt_classes=["mouse"])
    )
    rat = run_spec.spec_of(
        {}, "detection", _det("d2", prompt_fingerprint="h2", prompt_classes=["rat"])
    )
    label_mouse, label_rat = run_spec.spec_label(mouse), run_spec.spec_label(rat)
    assert label_mouse != label_rat
    assert "mouse" in label_mouse and "rat" in label_rat


def test_distinct_arena_ids():
    data = {"detection_runs": [_det("det_01")]}
    a = _key(data, "tracking", _track("t1", arena_ids=[]))
    b = _key(data, "tracking", _track("t2", arena_ids=["arena_a"]))
    assert a != b


# --------------------------------------------------------------------------- #
# Normalization: float noise, key order, JSON round-trip
# --------------------------------------------------------------------------- #
def test_freeze_stable_across_noise_and_json_roundtrip():
    data = {"detection_runs": [_det("det_01")]}
    spec = run_spec.spec_of(
        data, "tracking", _track("t1", hyperparams={"b": 2, "a": 0.30000000004})
    )
    other = run_spec.spec_of(
        data, "tracking", _track("t2", hyperparams={"a": 0.3, "b": 2})
    )
    # rounded float + reordered keys are the same identity
    assert run_spec.freeze_spec(spec) == run_spec.freeze_spec(other)
    # a JSON round-trip (as the stored config takes) does not change the key
    roundtripped = json.loads(json.dumps(spec))
    assert run_spec.freeze_spec(roundtripped) == run_spec.freeze_spec(spec)


# --------------------------------------------------------------------------- #
# Grouping across recordings — coverage, split across batches
# --------------------------------------------------------------------------- #
def _recording(tmp_path, sid, detection_runs):
    # group_specs reads a Recording's run-list properties (named by runs_key), so a
    # lightweight stand-in just needs those attributes + id/dir.
    return types.SimpleNamespace(
        id=sid,
        dir=tmp_path / sid,
        detection_runs=detection_runs,
        tracking_runs=[],
        embedding_runs=[],
        downsampling_runs=[],
    )


def test_group_specs_covers_split_batches(tmp_path):
    # Same detection settings, different ordinals + dates (as if split across launches).
    recordings = [
        _recording(tmp_path, "s0", [_det("det_01", created_at="2026-01-01T00:00")]),
        _recording(tmp_path, "s1", [_det("det_01", created_at="2026-02-01T00:00")]),
        _recording(tmp_path, "s2", [_det("det_02", created_at="2026-03-01T00:00")]),
    ]
    groups = run_spec.group_specs(recordings, "detection")
    assert len(groups) == 1
    assert sorted(groups[0]["recordings"]) == ["s0", "s1", "s2"]

    # A recording whose detection used a different prompt forms a separate spec.
    recordings.append(
        _recording(tmp_path, "s3", [_det("det_01", prompt_fingerprint="different")])
    )
    groups = run_spec.group_specs(recordings, "detection")
    assert len(groups) == 2
    coverage = {tuple(sorted(g["recordings"])) for g in groups}
    assert ("s0", "s1", "s2") in coverage
    assert ("s3",) in coverage


def test_group_specs_ignores_incomplete_runs(tmp_path):
    recordings = [
        _recording(tmp_path, "s0", [_det("det_01", status="running")]),
    ]
    assert run_spec.group_specs(recordings, "detection") == []


# --------------------------------------------------------------------------- #
# Per-recording resolution
# --------------------------------------------------------------------------- #
def test_resolve_spec_in_recording_newest_wins():
    data = {
        "detection_runs": [
            _det("det_01", conf_threshold=0.25, created_at="2026-01-01T00:00"),
            _det("det_02", conf_threshold=0.25, created_at="2026-02-01T00:00"),
            _det("det_03", conf_threshold=0.50),
        ]
    }
    match = run_spec.spec_of(data, "detection", _det("x", conf_threshold=0.25))
    assert run_spec.resolve_spec_in_recording(data, "detection", match) == "det_02"
    absent = run_spec.spec_of(data, "detection", _det("x", conf_threshold=0.9))
    assert run_spec.resolve_spec_in_recording(data, "detection", absent) is None


def test_same_spec_resolves_to_each_recordings_own_ordinal():
    """The crux of the fix: one detection spec resolves to a *different* per-recording
    run id when the same settings landed on different ordinals across recordings."""
    spec = run_spec.spec_of({}, "detection", _det("x"))
    data_a = {"detection_runs": [_det("det_02")]}  # same settings, ordinal det_02
    data_b = {"detection_runs": [_det("det_01")]}  # same settings, ordinal det_01
    assert run_spec.resolve_spec_in_recording(data_a, "detection", spec) == "det_02"
    assert run_spec.resolve_spec_in_recording(data_b, "detection", spec) == "det_01"


def test_spec_from_run_builds_named_run_spec():
    recording = types.SimpleNamespace(
        id="s1",
        dir=None,
        detection_runs=[_det("det_01")],
        tracking_runs=[],
        embedding_runs=[],
        downsampling_runs=[],
    )
    spec = run_spec.spec_from_run(recording, "detection", "det_01")
    assert spec["module"] == "detection"
    assert run_spec.freeze_spec(spec) == _key(
        {"detection_runs": [_det("det_01")]}, "detection", _det("det_01")
    )
    with pytest.raises(ValueError, match="no detection run"):
        run_spec.spec_from_run(recording, "detection", "det_99")


def test_resolve_upstream_spec_injects_run_id():
    data = {"detection_runs": [_det("det_07")]}
    module = PROCESSING_MODULES["tracking"]
    spec = run_spec.spec_of(data, "detection", _det("det_07"))
    config = {"detection_spec": spec, "tracker_id": "anchor_tracker"}
    module.resolve_upstream_spec("s1", data, config)
    assert config["detection_run_id"] == "det_07"
    assert "detection_spec" not in config


def test_resolve_upstream_spec_raises_when_absent():
    data = {"detection_runs": [_det("det_07", conf_threshold=0.9)]}
    module = PROCESSING_MODULES["tracking"]
    spec = run_spec.spec_of({}, "detection", _det("x", conf_threshold=0.1))
    with pytest.raises(ValueError, match="matching the selected spec"):
        module.resolve_upstream_spec("s1", data, {"detection_spec": spec})


def test_detection_roots_at_the_video():
    """Detection has no upstream *run* — it reads the recording's video, thinned by its
    own every_n_frames — so its lineage parent is the root sentinel."""
    module = PROCESSING_MODULES["detection"]
    assert module.lineage_ref is None
    spec = run_spec.spec_of({}, "detection", {"id": "det_01", "model_id": "sam3"})
    assert spec["parent"] == run_spec.ROOT_SPEC


def test_resolve_upstream_spec_is_a_no_op_without_a_parent():
    """Detection declares neither ``lineage_ref`` nor ``aux_refs`` (§6.5), so the resolver
    leaves its config untouched — ``aux_refs`` joining the guard changed nothing for it.
    """
    module = PROCESSING_MODULES["detection"]
    assert module.lineage_ref is None and module.aux_refs == ()
    config = {"model_id": "sam3", "conf_threshold": 0.5}
    module.resolve_upstream_spec("s1", {}, config)
    assert config == {"model_id": "sam3", "conf_threshold": 0.5}


def test_resolve_upstream_spec_noop_for_explicit_id():
    module = PROCESSING_MODULES["tracking"]
    config = {"detection_run_id": "det_03"}
    module.resolve_upstream_spec("s1", {}, config)
    assert config == {"detection_run_id": "det_03"}


# --------------------------------------------------------------------------- #
# Labels carry lineage
# --------------------------------------------------------------------------- #
def test_spec_label_includes_lineage():
    data = {"detection_runs": [_det("det_01", model_id="sam3", conf_threshold=0.25)]}
    label = run_spec.spec_label(run_spec.spec_of(data, "tracking", _track("t1")))
    assert "anchor_tracker" in label
    assert " from " in label
    assert "sam3" in label


# --------------------------------------------------------------------------- #
# Drift guard + legacy fallback
# --------------------------------------------------------------------------- #
def test_identity_fields_are_recorded():
    """Every declared identity field (and the lineage ref field) is actually written
    by the module's ``extra_record_fields`` — the single-source guard."""
    cases = {
        "detection": {"model_id": "m", "prompt": None},
        "tracking": {
            "detection_run_id": "d",
            "tracker_id": "t",
            "arena_ids": [],
            "hyperparams": {},
        },
        "embedding": {"tracking_run_id": "t", "model_id": "m", "track_ids": []},
    }
    for module_id, config in cases.items():
        module = PROCESSING_MODULES[module_id]
        record = module.extra_record_fields(config)
        for field in module.identity_fields:
            assert field in record, f"{module_id}.{field} declared but not recorded"
        if module.lineage_ref is not None:
            _upstream, ref_field = module.lineage_ref
            assert (
                ref_field in record
            ), f"{module_id}.{ref_field} (lineage) not recorded"


def test_legacy_hyperparams_recovered_from_job_json(tmp_path):
    sdir = tmp_path / "s1"
    run_dir = sdir / "tracking" / "track_01"
    run_dir.mkdir(parents=True)
    hp = {"conf_threshold": 0.5, "iou_threshold": 0.5, "max_missed_frames": 5}
    rundir.init_run_dir(
        run_dir,
        TrackingJob(
            job_id="track_01",
            recording_id="s1",
            track_id="track_01",
            detection_run_id="det_01",
            tracker_id="anchor_tracker",
            hyperparams=hp,
            arena_ids=[],
            created_at="2026-01-01T00:00",
        ),
    )
    # A legacy tracking record: no `hyperparams` key (written before it was recorded).
    legacy = {
        "id": "track_01",
        "status": "complete",
        "output": "tracking/track_01/",
        "created_at": "2026-01-01T00:00",
        "detection_run_id": "det_01",
        "tracker_id": "anchor_tracker",
        "arena_ids": [],
    }
    data = {"detection_runs": [_det("det_01")], "tracking_runs": [legacy]}
    legacy_key = _key(data, "tracking", legacy, sdir)
    modern_key = _key(
        {"detection_runs": [_det("det_01")]}, "tracking", _track("t9", hyperparams=hp)
    )
    assert legacy_key == modern_key


def test_legacy_unrecoverable_field_never_merges(tmp_path):
    sdir = tmp_path / "s1"  # no job.json written -> unrecoverable
    legacy_a = {
        "id": "track_01",
        "status": "complete",
        "output": "tracking/track_01/",
        "detection_run_id": "det_01",
        "tracker_id": "anchor_tracker",
        "arena_ids": [],
    }
    legacy_b = dict(legacy_a, id="track_02", output="tracking/track_02/")
    data = {"detection_runs": [_det("det_01")]}
    # Two legacy runs missing the same field must not collapse together.
    assert _key(data, "tracking", legacy_a, sdir) != _key(
        data, "tracking", legacy_b, sdir
    )


def test_legacy_detection_fingerprint_recovered_from_job_json(tmp_path):
    sdir = tmp_path / "s1"
    run_dir = sdir / "detections" / "det_01"
    run_dir.mkdir(parents=True)
    prompt = Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))])
    rundir.init_run_dir(
        run_dir,
        DetectionJob(
            job_id="det_01",
            recording_id="s1",
            model_id="sam3",
            prompt=prompt,
            conf_threshold=0.25,
            compute=ComputeTarget("cpu", None, None, None),
            created_at="2026-01-01T00:00",
            model_variant="large",
            weights_source="pretrained",
        ),
    )
    # A legacy detection record: no prompt_fingerprint / prompt_classes (written before
    # they were recorded) — both are recovered from the run's job.json.
    legacy = {
        "id": "det_01",
        "status": "complete",
        "output": "detections/det_01/",
        "created_at": "2026-01-01T00:00",
        "model_id": "sam3",
        "model_variant": "large",
        "weights_source": "pretrained",
        "conf_threshold": 0.25,
    }
    legacy_key = _key({"detection_runs": [legacy]}, "detection", legacy, sdir)
    modern = _det(
        "det_02",
        prompt_fingerprint=prompt_fingerprint(prompt),
        prompt_classes=prompt_classes(prompt),
    )
    modern_key = _key({"detection_runs": [modern]}, "detection", modern)
    assert legacy_key == modern_key


def test_legacy_unrecoverable_detection_never_merges_across_recordings(tmp_path):
    # Two recordings each hold a first-of-its-kind legacy detection at the same ordinal
    # (det_01) with no readable job.json. Keyed on the run id alone they would merge
    # across recordings — the dangerous false-merge; the recording-scoped sentinel splits
    # them so each stays its own kind (a safe false-split).
    legacy = {
        "id": "det_01",
        "status": "complete",
        "output": "detections/det_01/",
        "model_id": "sam3",
        "model_variant": "large",
        "weights_source": "pretrained",
        "conf_threshold": 0.25,
    }
    data = {"detection_runs": [legacy]}
    assert _key(data, "detection", legacy, tmp_path / "s1") != _key(
        data, "detection", legacy, tmp_path / "s2"
    )


# --- model id vs model name in labels (§11) --------------------------------- #
def _detection_spec(model_id="model_001", conf=0.5, weights="trained"):
    return {
        "module": "detection",
        "identity": {
            "model_id": model_id,
            "conf_threshold": conf,
            "weights_source": weights,
        },
    }


def test_spec_label_shows_a_model_name_when_it_has_one():
    """A run record keys on the model id, which is permanent; the user reads the name they
    gave it. Showing the bare id is what got reported as "the model name is truncated".
    """
    label = run_spec.spec_label(_detection_spec(), {"model_001": "Fish_YOLO"})
    assert label == "Fish_YOLO · model_001 @ 0.50 (trained)"


def test_spec_label_keeps_the_bare_id_without_a_map():
    """No map: the label is exactly what it was, so api/run_spec stays usable with no
    experiment (the CLI and tests call it that way)."""
    assert run_spec.spec_label(_detection_spec()) == "model_001 @ 0.50 (trained)"


def test_spec_label_leaves_a_built_in_model_alone():
    """A built-in model's id *is* its name — `sam3 · sam3` would be noise, and the map has
    no entry for it anyway."""
    spec = _detection_spec(model_id="sam3", weights="pretrained")
    assert run_spec.spec_label(spec, {"model_001": "Fish_YOLO"}) == "sam3 @ 0.50"
    # ...and an id whose "name" is just the id again doesn't get doubled either
    assert run_spec.spec_label(spec, {"sam3": "sam3"}) == "sam3 @ 0.50"


def test_spec_label_falls_back_for_a_model_deleted_since_the_run():
    """A run outlives its model; the label still has to render (the recording-name rule)."""
    assert run_spec.spec_label(_detection_spec(), {}) == "model_001 @ 0.50 (trained)"


def test_spec_label_names_the_model_in_a_lineage_parent_too():
    """The parent chain goes through the same renderer, so "from <detection>" gets it."""
    spec = {
        "module": "tracking",
        "identity": {"tracker_id": "iou"},
        "parent": _detection_spec(),
    }
    label = run_spec.spec_label(spec, {"model_001": "Fish_YOLO"})
    assert "from Fish_YOLO · model_001" in label


def test_resolve_upstream_spec_resolves_an_auxiliary_input_too():
    """§6.5 auxiliary inputs: embedding's keypoint track run (§14) rides a batch config as
    ``keypoint_track_spec`` and resolves per recording exactly as the upstream does."""
    track_record = {
        "id": "kt_04",
        "status": "complete",
        "detect_run_id": "kd_02",
        "tracking_identity": "x",
        "match": {"min_inside": 0.5, "mask_grow": 0.05},
        "method": "temporal_decode",
        "config": {"despike": True, "smooth_window_s": 0.133},
    }
    detect_record = {
        "id": "kd_02",
        "status": "complete",
        "model_id": "dinov3_match",
        # every identity field present: a missing one is recovered to a per-run sentinel
        # that never merges, which is the legacy fallback, not what this test is about
        **{f: "x" for f in PROCESSING_MODULES["keypoint_detect"].identity_fields},
    }
    data = {
        "keypoint_detect_runs": [detect_record],
        "keypoint_track_runs": [track_record],
    }
    module = PROCESSING_MODULES["embedding"]
    assert module.aux_refs == (("keypoint_track", "keypoint_run_id"),)
    spec = run_spec.spec_of(data, "keypoint_track", track_record)
    config = {"keypoint_track_spec": spec, "tracking_run_id": "track_01"}
    module.resolve_upstream_spec("s1", data, config)
    assert config["keypoint_run_id"] == "kt_04" and "keypoint_track_spec" not in config
    other = {**track_record, "id": "kt_05", "config": {"despike": False}}
    with pytest.raises(ValueError, match="no complete keypoint_track run matching"):
        module.resolve_upstream_spec(
            "s1",
            {"keypoint_detect_runs": [detect_record], "keypoint_track_runs": [other]},
            {"keypoint_track_spec": spec},
        )
    untouched = {"keypoint_run_id": "kt_01", "tracking_run_id": "track_01"}
    module.resolve_upstream_spec("s1", data, untouched)
    assert untouched == {"keypoint_run_id": "kt_01", "tracking_run_id": "track_01"}
