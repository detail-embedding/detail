"""Machine-global imported-model catalog (models/imported.py).

The catalog dir is isolated per test by the autouse ``_isolate_model_cache``
fixture (DETAIL_MODEL_DIR -> tmp), so these never touch the real machine.

**Class guarded here and in ``tests/api/test_models.py``: a catalog path that assumes every
model is a detector.** Before §12 *Keypoint models in the catalog*, an imported pose ``.pt``
was read by the detection loader (which accepts ``task: pose``), labelled ``role: detection``
by ``list_models``, offered to Quick segment and counted as a ready detector by the setup
checklist, and failed only when built; promote dropped the skeleton it is rebuilt from. Nothing
failed where any of that was written. The rule: every catalog reader asks
``imported.is_pose_record`` (the task) before treating a record as a detector, and a pose
record carries its skeleton copy, digest, library ref and sha256.
"""

import hashlib

import pytest

from detail.core.protocols import DetectionModel
from detail.core.registries import MODELS, TRAINED_LOADERS
from detail.core.skeleton import Skeleton
from detail.models import imported as imported_store
from detail.storage import skeleton_library


class _StubLoader(DetectionModel):
    """A trained-loader stand-in whose ``introspect`` reads no real checkpoint."""

    model_id = "stub"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def __init__(self, weights_path, classes, *, negative_class_ids=()):
        super().__init__(None)
        self.weights_path = weights_path
        self.classes = classes

    @staticmethod
    def introspect(weights_path):
        return ["mouse", "pellet"], "segment"

    def load(self, device): ...

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


@pytest.fixture
def stub_family(monkeypatch):
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubLoader)
    return "stubfam"


def _weights(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights")
    return w


def test_import_reads_classes_and_writes_record(tmp_path, stub_family):
    mid = imported_store.import_model(
        _weights(tmp_path), family=stub_family, name="mouse_yolo11s"
    )
    assert mid == "mouse_yolo11s"
    rec = imported_store.read(mid)
    assert rec["source"] == "imported"
    assert rec["model_family"] == "stubfam"
    assert [c["name"] for c in rec["classes"]] == ["mouse", "pellet"]  # from introspect
    assert (imported_store.model_dir(mid) / "weights/best.pt").is_file()
    assert imported_store.discover() == ["mouse_yolo11s"]


def test_import_name_defaults_to_filename(tmp_path, stub_family):
    w = tmp_path / "mymodel.pt"
    w.write_bytes(b"w")
    assert imported_store.import_model(w, family=stub_family) == "mymodel"


def test_import_rejects_duplicate(tmp_path, stub_family):
    imported_store.import_model(_weights(tmp_path), family=stub_family, name="dup")
    with pytest.raises(ValueError, match="already exists"):
        imported_store.import_model(_weights(tmp_path), family=stub_family, name="dup")


def test_import_rejects_registry_id_collision(tmp_path, stub_family, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _StubLoader)
    with pytest.raises(ValueError, match="built-in model id"):
        imported_store.import_model(_weights(tmp_path), family=stub_family, name="sam3")


def test_import_rejects_reserved_model_nnn(tmp_path, stub_family):
    with pytest.raises(ValueError, match="reserved experiment-model"):
        imported_store.import_model(
            _weights(tmp_path), family=stub_family, name="model_002"
        )


def test_import_unknown_family_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown model family"):
        imported_store.import_model(_weights(tmp_path), family="nope")


def test_import_missing_weights_raises(tmp_path, stub_family):
    with pytest.raises(FileNotFoundError):
        imported_store.import_model(tmp_path / "absent.pt", family=stub_family)


def test_list_and_remove(tmp_path, stub_family):
    mid = imported_store.import_model(_weights(tmp_path), family=stub_family, name="m")
    assert [r["id"] for r in imported_store.list_records()] == [mid]
    imported_store.remove(mid)
    assert imported_store.discover() == []


def test_import_stamps_created_at_and_empty_metadata(tmp_path, stub_family):
    # Every imported model gets a created_at stamp and the metadata keys (empty when
    # unset) so the catalog can always render them — the fields were absent before §12.
    mid = imported_store.import_model(_weights(tmp_path), family=stub_family, name="m")
    rec = imported_store.read(mid)
    assert rec["created_at"]  # a timestamp, not missing
    assert rec["made_by"] == "" and rec["description"] == "" and rec["notes"] == ""
    assert rec["preview"] is None


def test_import_captures_metadata_and_preview(tmp_path, stub_family):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG fake")
    mid = imported_store.import_model(
        _weights(tmp_path),
        family=stub_family,
        name="operant_v2",
        made_by="labmate",
        description="operant chamber",
        notes="room 1",
        preview=img,
    )
    rec = imported_store.read(mid)
    assert rec["made_by"] == "labmate"
    assert rec["description"] == "operant chamber"
    assert rec["notes"] == "room 1"
    assert rec["preview"] == "preview.png"
    assert (
        imported_store.model_dir(mid) / "preview.png"
    ).read_bytes() == b"\x89PNG fake"


def test_set_preview_replaces_a_stale_preview(tmp_path, stub_family):
    mid = imported_store.import_model(_weights(tmp_path), family=stub_family, name="m")
    mdir = imported_store.model_dir(mid)
    first = tmp_path / "a.jpg"
    first.write_bytes(b"jpg")
    assert imported_store.set_preview(mdir, first) == "preview.jpg"
    second = tmp_path / "b.png"
    second.write_bytes(b"png")
    assert imported_store.set_preview(mdir, second) == "preview.png"
    # the prior .jpg is gone — never leave a stale preview beside the new one
    assert sorted(p.name for p in mdir.glob("preview.*")) == ["preview.png"]


# --- a pose checkpoint (§12 *Keypoint models in the catalog*) ---------------------- #
# The task is read first and decides what the model is; a pose .pt needs a skeleton that
# fits its K and flip_idx. Both readers are stubbed: the detection loader's introspect
# reports the task, TrainedYoloPose.introspect the (K, flip_idx) — no ultralytics loads.
RAT = Skeleton(
    names=("nose", "left_ear", "right_ear", "tail_base"),
    edges=(("nose", "left_ear"), ("nose", "right_ear"), ("nose", "tail_base")),
    pairs=(("left_ear", "right_ear"),),
)


class _PoseStubLoader(_StubLoader):
    @staticmethod
    def introspect(weights_path):
        return ["animal"], "pose"


@pytest.fixture
def pose_checkpoint(monkeypatch):
    """The probe family reads ``task: pose``; returns a setter for the checkpoint's
    ``(K, flip_idx)`` (default: RAT's shape)."""
    from detail.models import yolo_pose

    monkeypatch.setitem(TRAINED_LOADERS, imported_store._PROBE_FAMILY, _PoseStubLoader)
    shape = {"value": (4, [0, 2, 1, 3])}
    monkeypatch.setattr(
        yolo_pose.TrainedYoloPose,
        "introspect",
        staticmethod(lambda weights_path: shape["value"]),
    )
    return lambda k, flip: shape.update(value=(k, flip))


def test_a_pose_checkpoint_imports_as_a_keypoint_model(tmp_path, pose_checkpoint):
    mid = imported_store.import_model(
        _weights(tmp_path), skeleton=RAT, skeleton_name="rat overhead", name="rat_pose"
    )
    rec = imported_store.read(mid)
    assert imported_store.is_pose_record(rec)
    assert rec["role"] == "keypoints" and rec["model_family"] == "yolo_pose"
    assert rec["skeleton"] == RAT.to_dict()
    assert rec["skeleton_digest"] == RAT.digest()
    assert rec["skeleton_ref"]["name"] == "rat overhead"
    assert skeleton_library.discover() == [rec["skeleton_ref"]["id"]]
    assert rec["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert "teacher" not in (rec.get("provenance") or {})  # an imported .pt has none


def test_a_pose_checkpoint_without_a_skeleton_is_refused(tmp_path, pose_checkpoint):
    with pytest.raises(
        ValueError, match="pose \\(keypoint\\) checkpoint; pass the skeleton"
    ):
        imported_store.import_model(_weights(tmp_path), name="rat_pose")
    assert imported_store.discover() == []


@pytest.mark.parametrize(
    "shape, message",
    [
        ((17, list(range(17))), "model places 17 points but the skeleton has 4"),
        ((4, [3, 1, 2, 0]), r"model swaps \(nose, tail_base\)"),
        ((4, None), "no flip_idx"),
    ],
    ids=["wrong K", "mismatched pairs", "no flip_idx with pairs"],
)
def test_a_skeleton_that_does_not_fit_is_refused(
    tmp_path, pose_checkpoint, shape, message
):
    pose_checkpoint(*shape)
    with pytest.raises(ValueError, match=message):
        imported_store.import_model(_weights(tmp_path), skeleton=RAT, name="rat_pose")
    assert imported_store.discover() == []
    assert skeleton_library.discover() == []  # nothing added for a refused import


def test_a_detection_family_given_for_a_pose_checkpoint_is_refused(
    tmp_path, pose_checkpoint
):
    # Refused, not overridden: a typed family is how a pose .pt became a detector (§12).
    with pytest.raises(ValueError, match="imports as 'yolo_pose', not 'yolo'"):
        imported_store.import_model(
            _weights(tmp_path), family="yolo", skeleton=RAT, name="rat_pose"
        )
    # naming the right family is fine
    mid = imported_store.import_model(
        _weights(tmp_path), family="yolo_pose", skeleton=RAT, name="rat_pose"
    )
    assert imported_store.read(mid)["model_family"] == "yolo_pose"


def test_the_pose_family_or_a_skeleton_for_a_detection_checkpoint_is_refused(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(TRAINED_LOADERS, imported_store._PROBE_FAMILY, _StubLoader)
    with pytest.raises(ValueError, match="not a pose one"):
        imported_store.import_model(_weights(tmp_path), family="yolo_pose")
    with pytest.raises(ValueError, match="a skeleton applies only to a pose"):
        imported_store.import_model(_weights(tmp_path), skeleton=RAT)
    # with no family a detection checkpoint lands as the probe family, as before
    mid = imported_store.import_model(_weights(tmp_path), name="det")
    rec = imported_store.read(mid)
    assert rec["model_family"] == imported_store._PROBE_FAMILY
    assert not imported_store.is_pose_record(rec) and "skeleton" not in rec


def test_an_id_naming_a_built_in_keypoint_method_is_refused(tmp_path, stub_family):
    from detail.core.registries import KEYPOINT_MODELS

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(KEYPOINT_MODELS, "dinov3_match", _StubLoader)
        with pytest.raises(ValueError, match="built-in model id"):
            imported_store.import_model(
                _weights(tmp_path), family=stub_family, name="dinov3_match"
            )
