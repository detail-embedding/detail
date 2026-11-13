"""Imported keypoint models through ONNX (§26 Phase 4): sleap-nn exports, run with onnxruntime.

The fixtures under ``data/sleap_onnx`` are real sleap-nn 0.3.3 exports of tiny models trained on a
mouse video (made by the spike's ``d_fixtures.py``): ``single_instance`` (greyscale, input scale
0.5, so the exporter's antialiased-resize path), ``centered_instance`` (RGB, crop 64) and the
``topdown`` bundle. ``external_weights`` is ``single_instance`` saved with its weights in
``model.onnx.data``, which the graph names — what an export looks like when ``onnx`` was not
installed to embed the metadata and re-save the graph whole. ``no_metadata`` is
``centered_instance`` with its embedded metadata removed. ``expected.json`` is sleap-nn's own
answer on ``frame_rgb.npy`` through its *torch* export wrappers — the known-answer guard, as
SuperAnimal is held to DeepLabCut's: DETAIL's onnxruntime path must reproduce it.

The catalog is isolated per test by the autouse ``_isolate_model_cache`` fixture.
"""

import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from detail.core.keypoints import InstanceFrame  # noqa: E402
from detail.core.skeleton import Skeleton  # noqa: E402
from detail.models import imported as imported_store  # noqa: E402
from detail.models import onnx_keypoints  # noqa: E402
from detail.models.onnx_keypoints import ImportedKeypointModel  # noqa: E402

DATA = Path(__file__).parent / "data" / "sleap_onnx"
EXPECTED = json.loads((DATA / "expected.json").read_text(encoding="utf-8-sig"))
FRAME = np.load(DATA / "frame_rgb.npy")
H, W = FRAME.shape[:2]
NAMES = [
    "nose", "left_ear", "right_ear", "neck_base", "left_shoulder", "right_shoulder",
    "mid_back", "left_hip", "right_hip", "tail_base", "tail_mid", "tail_tip",
]  # fmt: skip


def _loaded(preset: str, name: str | None = None) -> ImportedKeypointModel:
    model_id = onnx_keypoints.import_model(DATA / preset, name=name or f"m_{preset}")
    model = ImportedKeypointModel(model_id)
    model.load("cpu")
    return model


def _predict(model, subjects):
    """Each ``(frame, box)`` as a one-detection ``InstanceFrame``; its row's points and scores
    back, so a test reads one subject the way the method placed it."""
    results = model.predict_batch(
        [
            InstanceFrame(frame_idx=i, t_ns=i, frame=frame, boxes={0: tuple(box)})
            for i, (frame, box) in enumerate(subjects)
        ]
    )
    return [
        types.SimpleNamespace(xy=r.points[0], score=r.point_scores[0]) for r in results
    ]


def _topdown_box(index: int):
    """A small box on the top-down output's ``index``-th instance centroid."""
    cx, cy = EXPECTED["topdown"]["centroids"][index]
    return ((cx - 3) / W, (cy - 3) / H, (cx + 3) / W, (cy + 3) / H)


@pytest.mark.parametrize("preset", onnx_keypoints.PRESETS)
def test_detail_reproduces_sleap_nn_on_each_model_type(preset):
    expected = EXPECTED[preset]
    if preset == "topdown":
        pick = int(np.flatnonzero(expected["instance_valid"])[0])
        box = _topdown_box(pick)
        want_xy = np.asarray(expected["peaks"][pick])
        want_score = np.asarray(expected["peak_vals"][pick])
    else:
        box = tuple(expected.get("box", (0.0, 0.0, 1.0, 1.0)))
        want_xy = np.asarray(expected["peaks"])
        want_score = np.asarray(expected["peak_vals"])
    [got] = _predict(_loaded(preset), [(FRAME, box)])
    np.testing.assert_allclose(got.xy * [W, H], want_xy, atol=1e-3)
    np.testing.assert_allclose(got.score, want_score, atol=1e-4)
    assert got.xy.dtype == np.float32 and got.xy.shape == (len(NAMES), 2)


@pytest.mark.parametrize("preset", onnx_keypoints.PRESETS)
def test_a_batch_answers_as_its_frames_one_at_a_time(preset):
    model = _loaded(preset)
    box = _topdown_box(0) if preset == "topdown" else (0.3, 0.25, 0.75, 0.8)
    crops = [(FRAME, box)] * 3
    batch = _predict(model, crops)
    single = _predict(model, crops[:1])[0]
    for result in batch:
        np.testing.assert_allclose(result.xy, single.xy, equal_nan=True)


def test_import_records_what_the_export_declares():
    model_id = onnx_keypoints.import_model(DATA / "single_instance", name="mice")
    record = imported_store.read(model_id)
    assert imported_store.role_of(record) == "keypoints"
    assert record["model_family"] == "onnx_keypoints" and record["format"] == "sleap_nn"
    assert record["preset"] == "single_instance"
    assert record["input"]["channels"] == 1 and record["input"]["crop_size"] is None
    assert record["export"]["sleap_nn_version"] == "0.3.3"
    assert "checkpoint_path" not in json.dumps(record)  # a path on the training machine
    skeleton = Skeleton.from_dict(record["skeleton"])
    assert list(skeleton.names) == NAMES
    # pairs inferred from the names — a starting value, settable
    assert ("left_ear", "right_ear") in skeleton.pairs
    assert ("left_hip", "right_hip") in skeleton.pairs
    assert len(record["sha256"]) == 16


def test_an_export_keeping_its_weights_beside_it_brings_them(tmp_path):
    """A graph that names a model.onnx.data beside it loads nothing without it; the import
    copies both, so the model runs after the export folder is gone."""
    source = tmp_path / "export"
    shutil.copytree(DATA / "external_weights", source)
    model_id = onnx_keypoints.import_model(source / "model.onnx", name="mice")
    shutil.rmtree(source)
    weights = imported_store.model_dir(model_id) / "weights"
    assert sorted(p.name for p in weights.iterdir()) == [
        "model.onnx",
        "model.onnx.data",
    ]
    model = ImportedKeypointModel(model_id)
    model.load("cpu")
    [got] = _predict(model, [(FRAME, (0, 0, 1, 1))])
    np.testing.assert_allclose(
        got.xy * [W, H], EXPECTED["single_instance"]["peaks"], atol=1e-3
    )


def test_an_export_missing_its_weights_file_is_refused_at_import(tmp_path):
    source = tmp_path / "export"
    shutil.copytree(DATA / "external_weights", source)
    (source / "model.onnx.data").unlink()
    with pytest.raises(ValueError, match="not a readable ONNX model|does not load"):
        onnx_keypoints.import_model(source, name="mice")
    assert imported_store.discover(role=None) == []  # nothing half-written left behind


def test_the_metadata_sidecar_stands_in_for_embedded_metadata(tmp_path):
    """sleap-nn embeds its metadata only when ``onnx`` was installed at export, and always
    writes export_metadata.json beside the file."""
    source = tmp_path / "export"
    source.mkdir()
    shutil.copy(DATA / "no_metadata" / "model.onnx", source / "model.onnx")
    shutil.copy(
        DATA / "centered_instance" / "export_metadata.json",
        source / "export_metadata.json",
    )
    assert onnx_keypoints.read_export(source)["preset"] == "centered_instance"


def test_an_onnx_that_is_not_a_sleap_export_is_refused():
    with pytest.raises(ValueError, match="carries no sleap-nn metadata"):
        onnx_keypoints.import_model(DATA / "no_metadata", name="anon")


def _with_metadata(tmp_path, **changes) -> Path:
    source = tmp_path / "export"
    source.mkdir()
    shutil.copy(DATA / "no_metadata" / "model.onnx", source / "model.onnx")
    metadata = json.loads(
        (DATA / "centered_instance" / "export_metadata.json").read_text(
            encoding="utf-8-sig"
        )
    )
    metadata.update(changes)
    (source / "export_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return source


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"model_type": "bottomup"}, "groups them afterwards"),
        ({"model_type": "centroid"}, "one point per animal"),
        ({"model_type": "multi_class_topdown"}, "identities"),
        ({"model_type": "segmentation"}, "unknown sleap-nn model type"),
        ({"model_type": "topdown"}, r"missing \['centroids', 'instance_valid'\]"),
        ({"input_channels": 1}, "1 input channel"),
        ({"node_names": ["a", "a"]}, "missing or repeated"),
        ({"crop_size": None}, "must declare its crop_size"),
    ],
)
def test_an_export_detail_cannot_use_says_why(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        onnx_keypoints.read_export(_with_metadata(tmp_path, **changes))


def test_given_roles_must_name_the_exports_keypoints_in_order():
    reordered = Skeleton(names=tuple(reversed(NAMES)))
    with pytest.raises(ValueError, match="must be the export's, in order"):
        onnx_keypoints.import_model(DATA / "centered_instance", skeleton=reordered)
    plain = Skeleton(names=tuple(NAMES))  # no pairs: the user's, not the names'
    model_id = onnx_keypoints.import_model(DATA / "centered_instance", skeleton=plain)
    assert Skeleton.from_dict(imported_store.read(model_id)["skeleton"]).pairs == ()


def test_roles_can_change_and_names_cannot():
    model_id = onnx_keypoints.import_model(DATA / "centered_instance", name="mice")
    current = Skeleton.from_dict(imported_store.read(model_id)["skeleton"])
    unpaired = Skeleton(names=current.names, edges=current.edges)
    onnx_keypoints.set_skeleton(model_id, unpaired)
    assert ImportedKeypointModel(model_id).skeleton().pairs == ()
    renamed = Skeleton(names=("snout", *current.names[1:]))
    with pytest.raises(ValueError, match="cannot change"):
        onnx_keypoints.set_skeleton(model_id, renamed)


def test_top_down_keeps_the_animal_on_the_tracked_box_and_no_other():
    model = _loaded("topdown")
    valid = np.flatnonzero(EXPECTED["topdown"]["instance_valid"])
    for index in valid:
        [got] = _predict(model, [(FRAME, _topdown_box(int(index)))])
        np.testing.assert_allclose(
            got.xy * [W, H], EXPECTED["topdown"]["peaks"][index], atol=1e-3
        )
    # a box no centroid falls in: the tracked animal is not one SLEAP found
    centroids = np.asarray(EXPECTED["topdown"]["centroids"])
    corner = (0.0, 0.0, 2 / W, 2 / H)
    assert not ((centroids[:, 0] <= 2) & (centroids[:, 1] <= 2)).any()
    [none] = _predict(model, [(FRAME, corner)])
    assert np.isnan(none.xy).all() and np.isnan(none.score).all()


def test_a_centred_crop_past_the_frame_edge_is_zero_padded_and_moved_back():
    """The crop around a subject at the corner reaches outside the frame; it is padded with
    zeros, as SLEAP pads, and the points come back moved by the crop's corner."""
    model = _loaded("centered_instance")
    box = (0.0, 0.0, 0.2, 0.2)  # centre (9.6, 7.2): the 64 px crop starts at (-22, -25)
    [got] = _predict(model, [(FRAME, box)])
    left, top = int(round(0.1 * W - 32)), int(round(0.1 * H - 32))
    crop = np.zeros((3, 64, 64), np.uint8)
    crop[:, -top:, -left:] = FRAME.transpose(2, 0, 1)[:, : 64 + top, : 64 + left]
    peaks = model._session.run(None, {"image": crop[None]})[0][0]
    np.testing.assert_allclose(got.xy * [W, H], peaks + [left, top], atol=1e-3)


def test_an_unplaced_point_carries_no_score():
    xy, score = onnx_keypoints._result(
        np.array([[np.nan, np.nan], [10.0, 5.0]]), np.array([0.0, 0.7]), (20, 10)
    )
    assert np.isnan(xy[0]).all() and np.isnan(score[0])
    assert xy[1].tolist() == pytest.approx([0.5, 0.5])
    assert score[1] == pytest.approx(0.7)


def test_an_id_cannot_shadow_a_built_in_method():
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    with pytest.raises(ValueError, match="collides with a built-in"):
        onnx_keypoints.import_model(DATA / "centered_instance", name="dinov3_match")


def test_readiness_names_a_missing_onnxruntime(monkeypatch):
    model_id = onnx_keypoints.import_model(DATA / "centered_instance", name="mice")
    assert ImportedKeypointModel(model_id).readiness().ready
    monkeypatch.setattr(onnx_keypoints, "backend_available", lambda: False)
    status = ImportedKeypointModel(model_id).readiness()
    assert not status.ready and status.reason == "missing_dependency"
    assert "keypoints_onnx" in status.message


def test_the_record_says_which_file_ran():
    """A re-import under the same id would otherwise be invisible in a run's record."""
    model = _loaded("centered_instance")
    resolved = model.resolved()["imported"]
    assert resolved["preset"] == "centered_instance"
    assert resolved["sha256"] == imported_store.read(model.model_id)["sha256"]
    assert resolved["sleap_nn_version"] == "0.3.3"


def test_the_import_records_the_frame_size_sleap_matches_frames_to():
    """The training config's max_height / max_width, embedded in the export — the size the
    graph was traced at, and the one sleap-nn's predictor size-matches every frame to.
    """
    for preset in ("single_instance", "topdown"):
        model = _loaded(preset)
        assert model._input["frame_size"] == [H, W], preset
    assert _loaded("centered_instance")._input["frame_size"] is None  # fed crops


@pytest.mark.parametrize(
    "shape,ratio,filled",
    [
        ((2 * H, 2 * W), 0.5, (H, W)),  # larger: shrunk to fit
        ((H // 2, W // 2), 2.0, (H, W)),  # smaller: enlarged to fit
        ((H, W // 2), 1.0, (H, W // 2)),  # narrower: kept, padded at the right
        ((H // 2, W), 1.0, (H // 2, W)),  # shorter: kept, padded at the bottom
        ((H, W), 1.0, (H, W)),  # already the size: untouched
    ],
)
def test_a_frame_is_size_matched_as_sleap_does(shape, ratio, filled):
    model = _loaded("single_instance")
    frame = np.full(shape + (3,), 200, np.uint8)
    matched, got_ratio = model._size_match(frame)
    assert matched.shape == (H, W, 3) and got_ratio == pytest.approx(ratio)
    assert (matched[: filled[0], : filled[1]] == 200).all()
    assert (matched[filled[0] :] == 0).all() and (matched[:, filled[1] :] == 0).all()


@pytest.mark.parametrize("preset", ["single_instance", "topdown"])
def test_points_found_on_a_size_matched_frame_come_back_to_the_frame(
    preset, monkeypatch
):
    """The graph answers in the size-matched frame's pixels; the result is in the frame's own.
    For top-down the centroids come back too, before the tracked box is asked about them.
    """
    model = _loaded(preset)
    k = len(NAMES)
    peaks = np.tile(np.array([[10.0, 20.0]], np.float32), (k, 1))
    if preset == "topdown":
        out = {
            "centroids": np.array([[[30.0, 30.0], [5.0, 5.0], [0.0, 0.0]]], np.float32),
            "instance_valid": np.array([[True, True, False]]),
            "peaks": np.stack([peaks, peaks * 2, peaks * 0])[None],
            "peak_vals": np.ones((1, 3, k), np.float32),
        }
    else:
        out = {"peaks": peaks[None], "peak_vals": np.ones((1, k), np.float32)}
    monkeypatch.setattr(model, "_run", lambda batch: out)
    big = np.zeros((2 * H, 2 * W, 3), np.uint8)  # size-matched at ratio 0.5
    box = (
        55 / (2 * W),
        55 / (2 * H),
        65 / (2 * W),
        65 / (2 * H),
    )  # around (60, 60) = 30 / 0.5
    [got] = _predict(model, [(big, box)])
    np.testing.assert_allclose(got.xy[0] * [2 * W, 2 * H], [20.0, 40.0])


def test_a_scaled_centred_instance_export_is_refused_with_the_way_out(tmp_path):
    """sleap-nn 0.3.3 freezes such a graph's crop resize at the training frame size (a 256 px
    crop came back with peaks out to 728 px); its top-down bundle resizes crops correctly.
    """
    with pytest.raises(ValueError, match="Export the top-down bundle instead"):
        onnx_keypoints.read_export(_with_metadata(tmp_path, input_scale=0.5))


@pytest.mark.parametrize("preset", ["centered_instance", "topdown"])
def test_an_rgb_model_is_fed_rgb(preset):
    """The training frames are infrared — red, green and blue equal — so the known answer above
    cannot tell RGB from BGR. On a frame whose channels differ it can: fed BGR, the centred
    model's points move by up to 50 px."""
    expected = json.loads(
        (DATA / "expected_color.json").read_text(encoding="utf-8-sig")
    )[preset]
    frame = np.load(DATA / "frame_color.npy")
    if preset == "topdown":
        pick = int(np.flatnonzero(expected["instance_valid"])[0])
        cx, cy = expected["centroids"][pick]
        box = ((cx - 3) / W, (cy - 3) / H, (cx + 3) / W, (cy + 3) / H)
        want = expected["peaks"][pick]
    else:
        box, want = tuple(expected["box"]), expected["peaks"]
    [got] = _predict(_loaded(preset), [(frame, box)])
    np.testing.assert_allclose(got.xy * [W, H], want, atol=1e-3)


# --------------------------------------------------------------------------- #
# Asked for CUDA: CUDA or a refusal, never a quiet CPU run
# --------------------------------------------------------------------------- #
def _fake_ort(available, started=("CPUExecutionProvider",)):
    """An onnxruntime whose provider list and CUDA start-up the test decides — onnxruntime
    itself falls back to the CPU when CUDA will not start, so ``started`` is what the session
    ends up on. ``asked`` records the providers each session was opened with."""
    asked = []

    def session(path, providers):
        asked.append(providers)
        return types.SimpleNamespace(get_providers=lambda: list(started))

    ort = types.SimpleNamespace(
        __version__="1.30.0",
        cuda_version="13.0",
        get_available_providers=lambda: list(available),
        InferenceSession=session,
    )
    return ort, asked


def test_cuda_under_the_cpu_only_onnxruntime_is_refused_not_run_on_the_cpu(monkeypatch):
    """The first build opened a CPU session here, under a run recorded as CUDA."""
    model = ImportedKeypointModel(
        onnx_keypoints.import_model(DATA / "single_instance", name="mice")
    )
    ort, asked = _fake_ort(["AzureExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(onnx_keypoints, "_ort", lambda: ort)
    model.check_backend("cpu")
    with pytest.raises(
        onnx_keypoints.OnnxCudaUnavailableError, match="CPU-only build.*keypoints_onnx"
    ):
        model.check_backend("cuda")
    with pytest.raises(onnx_keypoints.OnnxCudaUnavailableError, match="CPU-only"):
        model.load("0")
    assert asked == [] and model._session is None


def test_a_gpu_onnxruntime_that_cannot_start_cuda_is_refused(monkeypatch):
    model = ImportedKeypointModel(
        onnx_keypoints.import_model(DATA / "single_instance", name="mice")
    )
    ort, _asked = _fake_ort(["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(onnx_keypoints, "_ort", lambda: ort)
    with pytest.raises(
        onnx_keypoints.OnnxCudaUnavailableError, match="could not start CUDA.*CUDA 13.0"
    ):
        model.load("0")
    assert model._session is None


def test_the_session_opens_on_the_cuda_device_the_run_names(monkeypatch):
    """``resolve_device`` hands a CUDA run "0", "1" or "0,1"; the first build opened every one
    on device 0."""
    model = ImportedKeypointModel(
        onnx_keypoints.import_model(DATA / "single_instance", name="mice")
    )
    ort, asked = _fake_ort(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        started=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    monkeypatch.setattr(onnx_keypoints, "_ort", lambda: ort)
    for device in ("0", "1", "1,0"):
        model.load(device)
    model.load("cpu")
    assert asked == [
        [("CUDAExecutionProvider", {"device_id": 0})],
        [("CUDAExecutionProvider", {"device_id": 1})],
        [("CUDAExecutionProvider", {"device_id": 1})],
        ["CPUExecutionProvider"],
    ]
