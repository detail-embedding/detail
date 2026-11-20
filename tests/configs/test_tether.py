"""The tether option: one ``cable`` class, recorded as auxiliary from the job to the model (§12).

**The class.** An option that adds a class is only half done if the fact that it is auxiliary
is lost on the way: the labeling job knows it, and tracking reads the trained model. Each hop
(job → ``dataset.yaml`` → ``model.yaml`` → the promoted copy) is checked here or beside the
writer it belongs to (``tests/pipeline/test_training.py``, ``tests/models/test_distilled.py``).
"""

import pytest

from detail.configs.labeling import TETHER_CLASS
from detail.configs.labeling import TETHER_PROMPT
from detail.configs.labeling import LabelingConfig
from detail.configs.labeling import with_tether
from detail.core.dataclasses import ComputeTarget
from detail.core.dataclasses import LabelingClass
from detail.core.dataclasses import LabelingJob
from detail.core.dataclasses import TrackingJob
from detail.pipeline import labeling
from detail.storage import datasets as datasets_store

MOUSE = LabelingClass(class_id=0, name="mouse", prompt="a mouse")


def test_off_leaves_the_classes_alone():
    assert with_tether([MOUSE], False) == ([MOUSE], [])
    assert LabelingConfig.model_fields["tether"].default is False


def test_on_adds_the_cable_after_the_users_classes():
    classes, auxiliary = with_tether([MOUSE], True)
    assert classes == [MOUSE, LabelingClass(class_id=1, name="cable", prompt="cable")]
    assert auxiliary == ["cable"]


def test_the_prompt_is_cable_not_tether():
    """SAM 3 found no tether at all where it found the cable (the pilot, §12)."""
    assert (TETHER_CLASS, TETHER_PROMPT) == ("cable", "cable")


@pytest.mark.parametrize(
    "clash",
    [
        LabelingClass(class_id=1, name="Cable", prompt="a wire"),
        LabelingClass(class_id=1, name="lead", prompt="cable"),
    ],
)
def test_a_class_that_is_already_the_cable_is_refused(clash):
    with pytest.raises(ValueError, match="already the cable"):
        with_tether([MOUSE, clash], True)


def _job(auxiliary_classes):
    classes, _ = with_tether([MOUSE], bool(auxiliary_classes))
    return LabelingJob(
        job_id="dataset_001",
        dataset_id="dataset_001",
        name="d",
        source_recording_ids=["recording_001"],
        classes=classes,
        n_frames=10,
        logit_threshold=0.5,
        min_contour_area=100,
        val_split=0.15,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-01-01T00:00:00Z",
        auxiliary_classes=list(auxiliary_classes),
    )


def test_the_jobs_round_trip_the_new_fields():
    job = _job(["cable"])
    assert LabelingJob.from_dict(job.to_dict()).auxiliary_classes == ["cable"]
    legacy = job.to_dict()
    del legacy["auxiliary_classes"]
    assert LabelingJob.from_dict(legacy).auxiliary_classes == []
    track = TrackingJob(
        job_id="t",
        recording_id="r",
        track_id="t",
        detection_run_id="det_01",
        tracker_id="anchor_tracker",
        hyperparams={},
        arena_ids=[],
        created_at="x",
        exclude_classes=["cable"],
    )
    assert TrackingJob.from_dict(track.to_dict()).exclude_classes == ["cable"]
    legacy = track.to_dict()
    del legacy["exclude_classes"]
    assert TrackingJob.from_dict(legacy).exclude_classes == []


@pytest.mark.parametrize("auxiliary", [["cable"], []])
def test_the_dataset_records_its_auxiliary_classes(tmp_path, auxiliary):
    job = _job(auxiliary)
    per_class = {c.class_id: 1 for c in job.classes}
    labeling._finalize(tmp_path, job, per_class, 1, 0, 0, {}, interrupted=False)
    record = datasets_store.read_dataset_yaml(tmp_path)
    assert record["auxiliary_classes"] == auxiliary
    assert [c["name"] for c in record["classes"]] == [c.name for c in job.classes]
