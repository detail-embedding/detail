"""The negative prompt: one more labelled class, recorded by name from the job onwards (§12).

**The class.** An option that adds a class is only half done if the fact that the class is a
negative one is lost on the way: the labeling job knows it, and the model's loader drops it.
Each hop (job → ``dataset.yaml`` → ``model.yaml`` → the promoted copy) is checked here or
beside the writer it belongs to (``tests/pipeline/test_labeling.py``,
``tests/pipeline/test_training.py``, ``tests/models/test_distilled.py``); the loader's drop is
``tests/models/test_negative_classes.py``.
"""

import pytest

from detail.configs.labeling import LabelingConfig
from detail.configs.labeling import with_negative_prompt
from detail.core.dataclasses import ComputeTarget
from detail.core.dataclasses import LabelingClass
from detail.core.dataclasses import LabelingJob

MOUSE = LabelingClass(class_id=0, name="mouse", prompt="a mouse")


@pytest.mark.parametrize("negative_prompt", [None, "", "   "])
def test_no_negative_prompt_leaves_the_classes_alone(negative_prompt):
    assert with_negative_prompt([MOUSE], negative_prompt) == ([MOUSE], [])
    assert LabelingConfig.model_fields["negative_prompt"].default is None


def test_a_negative_prompt_adds_a_class_after_the_users_named_after_it():
    """After the user's classes, so they keep their ids and a trained model's output classes
    are still the first ones; named after the prompt, so the dataset reads as typed."""
    rat = LabelingClass(class_id=3, name="rat", prompt="a rat")
    classes, negatives = with_negative_prompt([MOUSE, rat], " cable ")
    assert classes == [
        MOUSE,
        rat,
        LabelingClass(class_id=4, name="cable", prompt="cable"),
    ]
    assert negatives == ["cable"]


@pytest.mark.parametrize(
    "clash",
    [
        LabelingClass(class_id=1, name="Cable", prompt="a wire"),
        LabelingClass(class_id=1, name="lead", prompt=" CABLE"),
    ],
)
def test_a_negative_prompt_that_is_one_of_the_classes_is_refused(clash):
    with pytest.raises(ValueError, match="negative prompt 'cable' is also class"):
        with_negative_prompt([MOUSE, clash], "cable")


def _job(**over) -> LabelingJob:
    classes, negatives = with_negative_prompt([MOUSE], "cable")
    fields = {
        "job_id": "dataset_001",
        "dataset_id": "dataset_001",
        "name": "d",
        "source_recording_ids": ["recording_001"],
        "classes": classes,
        "n_frames": 10,
        "logit_threshold": 0.5,
        "min_contour_area": 100,
        "val_split": 0.15,
        "compute": ComputeTarget("cpu", None, None, None),
        "created_at": "2026-01-01T00:00:00Z",
        "negative_classes": negatives,
    }
    fields.update(over)
    return LabelingJob(**fields)


def test_the_job_round_trips_its_negative_classes():
    job = _job()
    assert LabelingJob.from_dict(job.to_dict()).negative_classes == ["cable"]


def test_a_job_recorded_before_the_field_has_none():
    legacy = _job().to_dict()
    del legacy["negative_classes"]
    assert LabelingJob.from_dict(legacy).negative_classes == []
