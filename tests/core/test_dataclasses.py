import json
import typing

import pytest

from detail.core.dataclasses import (
    BatchProgressEvent,
    BoxCue,
    ComputeTarget,
    DatasetExportJob,
    Detection,
    DetectionJob,
    DetectionProgressEvent,
    EmbeddingJob,
    EmbeddingProgressEvent,
    FrameResult,
    ImageCue,
    Job,
    JobStatus,
    LabelingClass,
    LabelingJob,
    LabelingProgressEvent,
    PointCue,
    ProgressEvent,
    Prompt,
    PromptTarget,
    PromptTestJob,
    PromptTestResult,
    TextCue,
    TrackingJob,
    TrackingProgressEvent,
    TrainingJob,
    TrainingProgressEvent,
    job_from_dict,
    job_to_dict,
    progress_event_from_dict,
    progress_event_to_dict,
)


def _cuda(indices=None):
    return ComputeTarget(
        backend="cuda",
        device_indices=indices,
        cloud_provider=None,
        cloud_instance=None,
    )


def _text_prompt():
    return Prompt(
        targets=[
            PromptTarget(class_name="mouse", cue=TextCue("mouse")),
            PromptTarget(class_name="food_pellet", cue=TextCue("food pellet")),
        ]
    )


def _detection_job():
    return DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="sam3",
        every_n_frames=4,
        prompt=_text_prompt(),
        conf_threshold=0.7,
        compute=_cuda([0, 1]),
        created_at="2026-05-22T10:00:00Z",
    )


def _labeling_job():
    return LabelingJob(
        job_id="lbl_01",
        dataset_id="dataset_001",
        name="mouse-pellet labels",
        source_recording_ids=["recording_001", "recording_002"],
        classes=[
            LabelingClass(class_id=0, name="mouse", prompt="mouse"),
            LabelingClass(class_id=1, name="pellet", prompt="food pellet"),
        ],
        n_frames=25000,
        logit_threshold=0.5,
        min_contour_area=64,
        val_split=0.1,
        compute=_cuda(None),
        created_at="2026-05-22T10:00:00Z",
        model_id="fake_yolo",
        model_variant="11l",
    )


def _training_job():
    return TrainingJob(
        job_id="train_01",
        model_id="mouse_yolo",
        dataset_id="dataset_001",
        trainer_id="ultralytics_yolo",
        base_model="yolo11s-seg.pt",
        hyperparams={"epochs": 100, "imgsz": 640},
        compute=_cuda([0]),
        created_at="2026-05-22T10:00:00Z",
    )


def _tracking_job():
    return TrackingJob(
        job_id="track_01",
        recording_id="recording_001",
        track_id="track_01",
        detection_run_id="det_01",
        tracker_id="anchor_tracker",
        hyperparams={"iou_threshold": 0.3, "max_gap": 5},
        arena_ids=["arena_main"],
        created_at="2026-05-22T10:00:00Z",
    )


def _embedding_job():
    return EmbeddingJob(
        job_id="emb_01",
        recording_id="recording_001",
        embedding_run_id="emb_01",
        tracking_run_id="track_01",
        track_ids=[0, 1],
        model_id="dinov3_convnext_base",
        renderer_spec=[
            {"type": "init_binary_mask"},
            {"type": "crop_to_bbox", "padding": 0.1},
        ],
        batch_size=32,
        num_workers=4,
        compute=_cuda([0]),
        created_at="2026-05-22T10:00:00Z",
    )


def _dataset_export_job():
    return DatasetExportJob(
        job_id="export_01",
        dataset_id="dataset_001",
        name="mouse_set",
        source_run_id="run_003",
        frame_selection={"mode": "every_n", "n": 10},
        val_split=0.15,
        min_contour_area=100,
        created_at="2026-05-22T10:00:00Z",
    )


JOB_BUILDERS = [
    (DetectionJob, _detection_job),
    (LabelingJob, _labeling_job),
    (TrainingJob, _training_job),
    (DatasetExportJob, _dataset_export_job),
    (TrackingJob, _tracking_job),
    (EmbeddingJob, _embedding_job),
]


@pytest.mark.parametrize("cls,build", JOB_BUILDERS)
def test_job_json_round_trip(cls, build):
    job = build()
    restored = cls.from_dict(json.loads(json.dumps(job.to_dict())))
    assert restored == job


@pytest.mark.parametrize("cls,build", JOB_BUILDERS)
def test_job_to_dict_is_json_native(cls, build):
    # json.dumps raises TypeError on any non-serializable value (e.g. a stray ndarray).
    json.dumps(build().to_dict())


def test_labeling_job_model_defaults_and_round_trips():
    job = _labeling_job()  # model_id="fake_yolo", model_variant="11l"
    assert job.to_dict()["model_id"] == "fake_yolo"
    restored = LabelingJob.from_dict(job.to_dict())
    assert (restored.model_id, restored.model_variant) == ("fake_yolo", "11l")
    legacy = job.to_dict()  # a record written before labeling was model-agnostic
    del legacy["model_id"]
    del legacy["model_variant"]
    restored = LabelingJob.from_dict(legacy)
    assert restored.model_id == "sam3" and restored.model_variant is None


def test_detection_job_carries_model_variant():
    job = DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="fake_yolo",
        prompt=_text_prompt(),
        conf_threshold=0.5,
        compute=_cuda([0]),
        created_at="t",
        model_variant="11m",
    )
    assert job.to_dict()["model_variant"] == "11m"
    assert DetectionJob.from_dict(job.to_dict()).model_variant == "11m"
    legacy = job.to_dict()  # an old record written before the field existed
    del legacy["model_variant"]
    assert DetectionJob.from_dict(legacy).model_variant is None


def test_detection_job_weights_source_defaults_and_round_trips():
    job = DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id="model_001",
        prompt=Prompt(targets=[]),  # closed-set: no prompt cues
        conf_threshold=0.5,
        compute=_cuda([0]),
        created_at="t",
        weights_source="trained",
    )
    assert job.to_dict()["weights_source"] == "trained"
    assert DetectionJob.from_dict(job.to_dict()).weights_source == "trained"
    legacy = job.to_dict()  # a record written before the field existed
    del legacy["weights_source"]
    assert DetectionJob.from_dict(legacy).weights_source == "pretrained"


def test_text_prompt_serializes_to_yaml_shape():
    target = _text_prompt().to_dict()["targets"][0]
    assert target == {"class_name": "mouse", "cue_type": "text", "cue": "mouse"}


def test_point_cue_round_trip():
    prompt = Prompt(
        targets=[
            PromptTarget(
                class_name="mouse",
                cue=PointCue(points=[(0.1, 0.2), (0.3, 0.4)], labels=[1, 0]),
            )
        ]
    )
    restored = Prompt.from_dict(json.loads(json.dumps(prompt.to_dict())))
    assert restored == prompt
    assert restored.targets[0].cue.points == [(0.1, 0.2), (0.3, 0.4)]


def test_empty_prompt_round_trips():
    prompt = Prompt(targets=[])
    restored = Prompt.from_dict(json.loads(json.dumps(prompt.to_dict())))
    assert restored == prompt
    assert restored.targets == []


def test_box_cue_round_trip():
    prompt = Prompt(
        targets=[
            PromptTarget(class_name="mouse", cue=BoxCue(boxes=[(0.1, 0.2, 0.3, 0.4)]))
        ]
    )
    restored = Prompt.from_dict(json.loads(json.dumps(prompt.to_dict())))
    assert restored == prompt


def test_image_cue_serializes_by_path():
    prompt = Prompt(
        targets=[
            PromptTarget(
                class_name="mouse", cue=ImageCue(image=None, path="prompts/m.png")
            )
        ]
    )
    d = prompt.to_dict()
    assert d["targets"][0] == {
        "class_name": "mouse",
        "cue_type": "image",
        "cue": "prompts/m.png",
    }
    restored = Prompt.from_dict(json.loads(json.dumps(d)))
    assert restored == prompt


def test_image_cue_without_path_raises():
    import numpy as np

    prompt = Prompt(
        targets=[
            PromptTarget(
                class_name="mouse",
                cue=ImageCue(image=np.zeros((4, 4), dtype=np.uint8), path=None),
            )
        ]
    )
    with pytest.raises(ValueError, match="must be persisted"):
        prompt.to_dict()


def test_job_status_serializes_as_string():
    assert json.dumps({"status": JobStatus.COMPLETE}) == '{"status": "complete"}'
    assert JobStatus("partial") is JobStatus.PARTIAL
    assert JobStatus.PARTIAL.value == "partial"


def test_job_union_members():
    assert set(typing.get_args(Job)) == {
        DetectionJob,
        LabelingJob,
        TrainingJob,
        DatasetExportJob,
        TrackingJob,
        EmbeddingJob,
    }


def test_progress_event_union_members():
    assert set(typing.get_args(ProgressEvent)) == {
        DetectionProgressEvent,
        LabelingProgressEvent,
        TrainingProgressEvent,
        TrackingProgressEvent,
        EmbeddingProgressEvent,
        BatchProgressEvent,
    }


def test_compute_target_round_trip():
    compute = _cuda([0, 1])
    assert ComputeTarget.from_dict(compute.to_dict()) == compute


# --- tagged (discriminated) serialization for the executor boundary --------- #
@pytest.mark.parametrize(
    "tag,build",
    [
        ("detection", _detection_job),
        ("labeling", _labeling_job),
        ("training", _training_job),
        ("tracking", _tracking_job),
        ("embedding", _embedding_job),
    ],
)
def test_job_tagged_round_trip(tag, build):
    job = build()
    d = job_to_dict(job)
    assert d["type"] == tag
    assert job_from_dict(json.loads(json.dumps(d))) == job


def test_job_to_dict_rejects_non_job():
    with pytest.raises(TypeError, match="not a known Job"):
        job_to_dict(object())


def test_job_from_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown job type"):
        job_from_dict({"type": "nope"})


def _progress_events():
    return [
        ("detection", DetectionProgressEvent(1, 2, 100, 200, 5.0, 10.0)),
        ("labeling", LabelingProgressEvent(50, 100, 1, 2, 5.0, None)),
        ("training", TrainingProgressEvent(3, 100, {"box_map50": 0.85}, 5.0, 9.0)),
        ("tracking", TrackingProgressEvent(1, 2, 100, 200, 2, 5.0, None)),
        ("embedding", EmbeddingProgressEvent(0, 1, 2, 100, 200, 5.0, 1.0)),
    ]


@pytest.mark.parametrize("tag,event", _progress_events())
def test_progress_event_tagged_round_trip(tag, event):
    d = progress_event_to_dict(event)
    assert d["type"] == tag
    assert progress_event_from_dict(json.loads(json.dumps(d))) == event


def test_progress_event_to_dict_rejects_unknown():
    with pytest.raises(TypeError, match="not a known ProgressEvent"):
        progress_event_to_dict(object())


def test_progress_event_message_defaults_none_and_round_trips():
    assert DetectionProgressEvent(1, 2, 100, 200, 5.0, 10.0).message is None
    event = DetectionProgressEvent(0, 1, 0, 100, 0.0, None, message="loading model")
    d = progress_event_to_dict(event)
    assert d["message"] == "loading model"
    assert progress_event_from_dict(json.loads(json.dumps(d))) == event


def test_embedding_progress_event_carries_units():
    event = EmbeddingProgressEvent(
        track_id=3,
        parts_done=0,
        parts_total=1,
        frames_done=10,
        frames_total=10,
        elapsed_s=0.0,
        eta_s=None,
        units_done=2,
        units_total=5,
    )
    d = progress_event_to_dict(event)
    assert (d["units_done"], d["units_total"]) == (2, 5)
    assert progress_event_from_dict(json.loads(json.dumps(d))) == event


def test_prompt_test_types_are_preview_not_executor_jobs():
    import numpy as np

    job = PromptTestJob(
        recording_id="recording_001",
        model_id="sam3",
        prompts=[_text_prompt()],
        n_sample_frames=8,
        compute=_cuda(None),
    )
    assert job.n_sample_frames == 8
    # synchronous preview type — not submitted to the executor
    assert PromptTestJob not in typing.get_args(Job)

    result = PromptTestResult(
        frame_idx=0, t_ns=0, frame_image=np.zeros((4, 4, 3), np.uint8), results={}
    )
    assert result.results == {}


# --- FrameResult: the shared row container every model returns (§5) --------- #
def _rows():
    import numpy as np

    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:4] = True
    return [
        Detection(mask, (0.1, 0.2, 0.5, 0.6), "mouse", 0.9, None),
        Detection(None, None, "cage", 0.4, 7),
    ]


def test_frame_result_round_trips_detection_rows():
    """``from_detections`` is how a detection model builds the container and ``detections()``
    is what the writer, the overlays and the labeling export read: the rows survive the trip,
    including a row with no box (a NaN row inside, ``None`` outside) and one with an id.
    """
    import numpy as np

    result = FrameResult.from_detections(3, 120_000_000, _rows())
    assert (result.frame_idx, result.t_ns, result.n) == (3, 120_000_000, 2)
    assert result.class_name == ["mouse", "cage"]
    assert result.confidence.dtype == np.float32 and result.bbox.shape == (2, 4)
    assert np.isnan(result.bbox[1]).all()
    assert result.track_id.tolist() == [-1, 7]
    back = result.detections()
    assert [d.class_name for d in back] == ["mouse", "cage"]
    assert back[0].bbox == pytest.approx((0.1, 0.2, 0.5, 0.6))
    assert back[1].bbox is None and back[1].mask is None
    assert (back[0].object_id, back[1].object_id) == (None, 7)
    assert back[0].mask.sum() == 4


def test_frame_result_without_identities_has_no_track_id_column():
    """``track_id`` is ``None`` before tracking — a detection has no identity yet — rather
    than a column of placeholders a reader could mistake for ids."""
    result = FrameResult.from_detections(0, 0, [_rows()[0]])
    assert result.track_id is None
    assert result.detections()[0].object_id is None
    empty = FrameResult.from_detections(0, 0, [])
    assert empty.n == 0 and empty.detections() == []


def test_frame_result_refuses_a_column_of_the_wrong_length():
    """Every array's first axis is ``n``: a mismatched column is refused at construction,
    not discovered as a misaligned row in the zarr."""
    import numpy as np

    with pytest.raises(ValueError, match="confidence has 1 rows but n is 2"):
        FrameResult(frame_idx=0, t_ns=0, n=2, confidence=np.zeros(1, np.float32))
    with pytest.raises(ValueError, match=r"extras\['heat'\] has 3 rows but n is 2"):
        FrameResult(frame_idx=0, t_ns=0, n=2, extras={"heat": np.zeros((3, 5))})
    keypoints = FrameResult(
        frame_idx=0,
        t_ns=0,
        n=2,
        track_id=np.array([0, 1], np.int32),
        points=np.full((2, 5, 2), np.nan, np.float32),
        point_scores=np.zeros((2, 5), np.float32),
    )
    assert (
        keypoints.class_name is None
    )  # a keypoint model leaves the detection fields unset
    assert keypoints.detection_row is None
    with pytest.raises(ValueError, match="detection_row has 1 rows but n is 2"):
        FrameResult(frame_idx=0, t_ns=0, n=2, detection_row=np.zeros(1, np.int32))
    instances = FrameResult(
        frame_idx=0,
        t_ns=0,
        n=2,
        detection_row=np.array([4, -1], np.int32),
        points=np.zeros((2, 5, 2), np.float32),
    )
    assert instances.detection_row.tolist() == [4, -1]
