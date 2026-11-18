"""The tracker a run gets when nobody names one, and that it is written down.

**The class.** A default resolved at launch and not persisted is not a default, it is a
reading of the table on the day you looked. `CLAUDE.md`: *a resolved default is persisted*.
Tracking had the sharper version of the problem — the field was declared **required**, so the
model had no default at all, and each surface supplied its own: the CLI hardcoded
``iou_tracker`` in a decorator, the GUI combo inherited ``sorted(TRACKERS)[0]``. The same tool
answered "what does a default tracking run do?" two different ways, one of them decided by
alphabetical order.

**Why the write-back is not optional.** ``TrackingModule.validate`` calls
``TrackingConfig.model_validate(config)`` and *discards the instance*; every line after it
reads the raw ``config`` dict. So giving the model a default changes nothing on its own — a
config that omits the key would reach ``build_job`` as a ``KeyError`` instead of as the
default. ``validate`` has to resolve it and put it back, which is what it already does for
``every_n_frames`` and ``hyperparams``.

The surface half of this lives elsewhere, because no one file can see all three:
``tests/configs/test_cli_defaults_match_models.py`` (no CLI option invents a default) and
``tests/gui/test_launch.py`` (the form opens on the model's default, not on sort order).
"""

import pytest

from detail.configs.tracking import DEFAULT_TRACKER, TrackingConfig
from detail.core.errors import UserInputError
from detail.core.registries import TRACKERS
from detail.pipeline.tracking import TrackingModule

COMPLETE_DETECTION = {
    "detection_runs": [
        {"id": "det_01", "status": "complete", "output": "detections/det_01/"}
    ]
}


@pytest.fixture(autouse=True)
def _detection_on_disk(recording_dir):
    """The output the record above describes: validate refuses a complete run whose folder
    is gone (`test_a_detection_run_whose_folder_is_gone_is_refused_at_validate`)."""
    (recording_dir / "detections" / "det_01" / "part_000.zarr").mkdir(parents=True)


def _validate(recording_dir, config):
    return TrackingModule().validate(recording_dir, dict(COMPLETE_DETECTION), config)


@pytest.mark.parametrize("gone", ["folder", "parts"])
def test_a_detection_run_whose_folder_is_gone_is_refused_at_validate(
    recording_dir, gone
):
    """It validated, submitted, then failed, with the cause only in `job.json` (#31, 50)."""
    import shutil

    det = recording_dir / "detections" / "det_01"
    shutil.rmtree(det if gone == "folder" else det / "part_000.zarr")
    with pytest.raises(UserInputError) as caught:
        _validate(recording_dir, {"detection_run_id": "det_01"})
    assert str(caught.value) == (
        "detection run 'det_01' is recorded as complete but its output is not on disk "
        f"(detections/det_01/part_*.zarr in recording {recording_dir.name!r}); "
        "run detection again, or delete that run"
    )


def test_keypoint_detection_and_embedding_check_their_upstream_folder_too(
    recording_dir,
):
    """The two other stages that read an upstream run's folder, through the same check."""
    import inspect

    from detail.pipeline import embedding, keypoint_detect

    for module, kind in (
        (keypoint_detect.KeypointDetectModule, "detection"),
        (embedding.EmbeddingModule, "tracking"),
    ):
        source = " ".join(inspect.getsource(module.validate).split())
        assert f'require_run_output( recording_dir, "{kind}"' in source.replace(
            "require_run_output(recording_dir", "require_run_output( recording_dir"
        ), f"{module.__name__}.validate does not check its {kind} run's folder"


def test_a_config_with_no_tracker_resolves_to_the_default(recording_dir):
    resolved = _validate(recording_dir, {"detection_run_id": "det_01"})
    assert resolved["tracker_id"] == DEFAULT_TRACKER


def test_the_resolved_tracker_is_written_back_into_the_config(recording_dir):
    """Not just returned — the record is built from this dict, so it has to carry the value."""
    config = {"detection_run_id": "det_01"}
    _validate(recording_dir, config)
    assert config["tracker_id"] == DEFAULT_TRACKER, (
        "validate resolved the default but did not persist it; the run record is built from "
        "this dict, so the run would not name the tracker it ran"
    )


def test_the_resolved_default_reaches_the_run_record(recording_dir):
    """The property that matters to a user: the record names the tracker.

    `extra_record_fields` reads ``config["tracker_id"]`` directly, so without the write-back
    this is a KeyError rather than a missing key — but either way the run cannot say what it
    ran, which is the thing being guarded.
    """
    module = TrackingModule()
    config = module.validate(
        recording_dir, dict(COMPLETE_DETECTION), {"detection_run_id": "det_01"}
    )
    record = module.extra_record_fields(config)
    assert record["tracker_id"] == DEFAULT_TRACKER


def test_an_explicit_tracker_is_never_overridden(recording_dir):
    """Guard the guard: the fallback must be a fallback, not an assignment.

    An `or` that ran unconditionally would pass every test above while quietly ignoring what
    the user asked for — so probe with a registered tracker that is *not* the default.
    """
    other = next(t for t in sorted(TRACKERS) if t != DEFAULT_TRACKER)
    resolved = _validate(
        recording_dir, {"detection_run_id": "det_01", "tracker_id": other}
    )
    assert resolved["tracker_id"] == other


def test_the_default_names_a_registered_tracker():
    """A default that resolves to nothing fails at launch, in front of the user."""
    assert DEFAULT_TRACKER in TRACKERS
    assert TrackingConfig.model_fields["tracker_id"].default == DEFAULT_TRACKER


def test_hyperparams_left_unset_become_the_trackers_own_defaults(recording_dir):
    """``None`` means "the tracker decides", and must not reach the tracker as ``None``.

    ``config.get("hyperparams", {})`` returns None for a key that is present and null — the
    shape an optional API argument produces — and `validate_hyperparams` then does
    ``set(None)``. A bare TypeError, from an input that is entirely legal.
    """
    resolved = _validate(
        recording_dir, {"detection_run_id": "det_01", "hyperparams": None}
    )
    schema = TRACKERS[DEFAULT_TRACKER].HYPERPARAMS_SCHEMA
    assert resolved["hyperparams"] == {
        name: default for name, (_type, default) in schema.items()
    }


def test_an_unknown_tracker_still_raises(recording_dir):
    """The default replaces "you must name one", not "it must be real"."""
    with pytest.raises(ValueError, match="unknown tracker_id"):
        _validate(recording_dir, {"detection_run_id": "det_01", "tracker_id": "nope"})
