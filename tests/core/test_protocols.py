import numpy as np
import pytest

from detail.core.dataclasses import (
    FrameResult,
    RenderContext,
    TextCue,
)
from detail.core.keypoints import ScoreSpec
from detail.core.skeleton import Skeleton
from detail.core.protocols import (
    DetectionModel,
    EmbeddingModel,
    KeypointModel,
    KeypointTrackMethod,
    MaskTransform,
    Model,
    PatchFeatures,
    Tracker,
    Trainer,
)

ABCS = [
    Model,
    DetectionModel,
    Trainer,
    Tracker,
    EmbeddingModel,
    KeypointModel,
    KeypointTrackMethod,
    PatchFeatures,
    MaskTransform,
]


# --- complete minimal implementations (also serve as reference stubs) -------- #
class FakeDetectionModel(DetectionModel):
    model_id = "fake"
    supported_cue_types = [TextCue]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def predict(self, frame, prompt):
        return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=[])

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


class FakeTrainer(Trainer):
    trainer_id = "fake"
    model_family = "fake"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]

    def is_ready(self):
        return True

    def prepare_dataset(self, dataset_dir, work_dir):
        return work_dir

    def train(
        self, job, dataset_dir, model_dir, device, progress_cb=None, stop_event=None
    ):
        pass

    def resume(self, job, model_dir, device, progress_cb=None, stop_event=None):
        pass

    def checkpoint_path(self, model_dir):
        return None

    def best_weights_path(self, model_dir):
        return None

    def final_metrics(self, model_dir):
        return {}

    def metric_history(self, model_dir):
        return []

    def validate_hyperparams(self, hyperparams):
        return dict(hyperparams)

    def resolved_settings(self, model_dir):
        return {}


class FakeTracker(Tracker):
    tracker_id = "fake"
    supported_backends = ["cpu"]

    def validate_hyperparams(self, hyperparams):
        return dict(hyperparams)

    def track(
        self,
        job,
        detection_run_dir,
        output_dir,
        arena_polygons,
        progress_cb=None,
        stop_event=None,
    ):
        pass


class FakeEmbeddingModel(EmbeddingModel):
    model_id = "fake"
    output_dim = 8
    supported_backends = ["cpu"]

    def load(self, device):
        pass

    def is_ready(self):
        return True

    def embed(self, image):
        return np.zeros(self.output_dim, dtype=np.float32)

    def embed_batch(self, images):
        return images


class FakeKeypointModel(KeypointModel):
    model_id = "fake_kp"
    supported_backends = ["cpu"]
    per_detection = True
    score_spec = ScoreSpec("heatmap peak", 0.0, 1.0, None)
    default_track_method = "despike_smooth"

    def load(self, device):
        pass

    def skeleton(self):
        return Skeleton(names=("a", "b"))

    def predict_batch(self, frames):
        return []


class FakeTrackMethod(KeypointTrackMethod):
    method_id = "fake_track"
    config_model = dict

    def track(self, rows, skeleton, config, **kw):
        return None


class FakeTransform(MaskTransform):
    def apply(self, ctx):
        return ctx

    def to_spec(self):
        return {"type": "fake"}


COMPLETE_IMPLS = [
    FakeDetectionModel,
    FakeTrainer,
    FakeTracker,
    FakeEmbeddingModel,
    FakeKeypointModel,
    FakeTrackMethod,
    FakeTransform,
]


@pytest.mark.parametrize("abc", ABCS)
def test_abstract_base_cannot_instantiate(abc):
    with pytest.raises(TypeError):
        abc()


@pytest.mark.parametrize("impl", COMPLETE_IMPLS)
def test_complete_impl_instantiates(impl):
    instance = impl()
    assert isinstance(instance, impl)


def test_incomplete_impl_cannot_instantiate():
    class PartialTracker(Tracker):
        tracker_id = "partial"
        supported_backends = ["cpu"]

        def validate_hyperparams(self, hyperparams):
            return hyperparams

        # track() intentionally not implemented

    with pytest.raises(TypeError):
        PartialTracker()


def test_transform_apply_returns_context():
    ctx = RenderContext(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        mask=None,
        bbox=None,
        image=None,
        frame_idx=0,
        t_ns=0,
        class_name="mouse",
        confidence=0.9,
        object_id=0,
        aux={},
    )
    assert FakeTransform().apply(ctx) is ctx


def test_default_readiness_install_is_ready():
    # A model that fetches nothing inherits ready / no-op defaults (Section 5);
    # is_ready() derives from readiness(). The fakes above override is_ready, so
    # this asserts the base behaviour directly.
    class BareModel(DetectionModel):
        model_id = "bare"
        supported_cue_types = [TextCue]
        supported_backends = ["cpu"]

        def load(self, device):
            pass

        def predict(self, frame, prompt):
            return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=[])

        def predict_batch(self, frames, prompts):
            return []

    model = BareModel()
    r = model.readiness()
    assert r.ready is True and r.reason == "ok"
    assert model.is_ready() is True
    assert model.install() is None


def test_importing_protocols_does_not_import_torch():
    # Fresh interpreter: importing core.protocols must not pull in any backend
    # (Tenet 13). torch is referenced only under TYPE_CHECKING.
    import subprocess
    import sys

    code = "import detail.core.protocols, sys; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_model_protocol_shares_the_one_base():
    """§5 "Model protocol": the install / load contract is written once, on ``Model``, and
    the per-kind protocols add only their ``predict``. A third copy (the one #626 carried for
    keypoints) is what this base exists to prevent."""
    assert issubclass(DetectionModel, Model) and issubclass(EmbeddingModel, Model)
    for name in ("load", "install", "readiness", "is_installed", "is_ready"):
        assert getattr(DetectionModel, name) is getattr(Model, name)
        assert getattr(EmbeddingModel, name) is getattr(Model, name)


def test_model_base_hooks_default_to_nothing():
    """``check_backend`` says nothing and ``resolved`` records nothing unless a model has
    something to say; ``license`` / ``shares_weights_with`` are unset."""
    model = FakeEmbeddingModel()
    assert model.check_backend("cpu") is None
    assert model.resolved() == {}
    assert model.license is None and model.shares_weights_with is None


def test_keypoint_model_is_on_the_shared_base_with_its_own_predict():
    """§26 "KeypointModel protocol": the install contract is Model's; the class adds what the
    detect stage reads before it runs (examples, mask, candidates, the score's meaning).
    """
    assert issubclass(KeypointModel, Model)
    for name in ("load", "install", "readiness", "is_installed", "is_ready"):
        assert getattr(KeypointModel, name) is getattr(Model, name)
    assert (KeypointModel.needs_examples, KeypointModel.needs_mask) == (False, False)
    assert KeypointModel.n_candidates == 1
    model = FakeKeypointModel()
    assert model.skeleton().names == ("a", "b")
    assert model.score_spec.meaning == "heatmap peak"
    assert model.check_backend("cpu") is None and model.resolved() == {}
    assert KeypointModel.reports_confidence is False
    with pytest.raises(TypeError, match="reports no instance confidence"):
        model.set_min_pose_confidence(0.25)  # only a method that reports one takes it


def test_every_keypoint_method_says_whether_it_works_per_detection():
    """``per_detection`` has no default on the base (§26): the detect stage decides from it
    which frames to visit and whether to hand over detections, so a method that forgot it
    must not silently get one. Swept over every registered method and the student."""
    import detail.models.patch_match  # noqa: F401  (registers dinov3_match)
    from detail.core.registries import KEYPOINT_MODELS
    from detail.models.yolo_pose import TrainedYoloPose

    assert "per_detection" not in vars(KeypointModel)
    methods = [*KEYPOINT_MODELS.values(), TrainedYoloPose]
    assert len(methods) >= 2
    for cls in methods:
        assert isinstance(getattr(cls, "per_detection", None), bool), cls.__name__
        if (
            cls.reports_confidence
        ):  # it must take the threshold, not inherit the refusal
            assert (
                cls.set_min_pose_confidence is not KeypointModel.set_min_pose_confidence
            ), cls.__name__
