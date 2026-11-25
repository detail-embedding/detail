"""TrainedYOLO: closed-set apply of a locally-trained model (§12).

The load+predict test needs a real segmentation ``.pt`` to exercise the closed-set path end
to end without training one. It **builds** that checkpoint from Ultralytics' own bundled
architecture config rather than downloading pretrained weights.

That is not a stylistic preference. The download version of this test parked a full-suite run
until the 120 s watchdog killed it, and it would have done so on every run: ``tmp_path`` is
fresh each time, so the weights were re-fetched from the network on every invocation. Its
``try/except -> skip`` caught a download that *failed* and did nothing for one that *stalled* —
a hang is not an exception. A test whose runtime depends on a remote host is not bounded by
anything, and the class of bug is "a test that reaches the network without a bound", not
"this particular URL was slow that day".

Building from the ``.yaml`` gives randomly-initialised weights, which is exactly right here:
nothing asserts *what* is detected, only that a closed-set model loads, runs, ignores the
prompt, and returns a ``FrameResult``. The predictions of a pretrained net were never the
subject.
"""

import sys

import numpy as np
import pytest

from detail.core.dataclasses import (
    FrameResult,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.registries import TRAINED_LOADERS
from detail.models import trained_yolo
from detail.models._ultralytics import import_ultralytics
from detail.models.trained_yolo import TrainedYOLO


def test_registered_under_yolo_family():
    assert TRAINED_LOADERS.get("yolo") is TrainedYOLO


def test_closed_set_attrs():
    # the trained model is closed-set: no prompt cues, runs on CPU/MPS/CUDA
    assert TrainedYOLO.supported_cue_types == []
    assert TrainedYOLO.supported_backends == ["cpu", "mps", "cuda"]
    assert TrainedYOLO.supports_batching is True


def test_init_stores_weights_and_classes(tmp_path):
    weights = tmp_path / "best.pt"
    model = TrainedYOLO(weights, ["mouse", "pellet"], negative_class_ids=())
    assert model._weights_path == weights
    assert model._classes == ["mouse", "pellet"]


def test_predict_before_load_raises(tmp_path):
    model = TrainedYOLO(tmp_path / "best.pt", ["mouse"], negative_class_ids=())
    with pytest.raises(RuntimeError, match="load.. must be called"):
        model.predict(np.zeros((8, 8, 3), np.uint8), Prompt(targets=[]))


def test_readiness_missing_dependency(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    # None in sys.modules makes ``import ultralytics`` raise ImportError
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    status = TrainedYOLO(weights, ["mouse"], negative_class_ids=()).readiness()
    assert not status.ready
    assert status.reason == "missing_dependency"


def test_readiness_missing_weights(tmp_path):
    pytest.importorskip("ultralytics")
    status = TrainedYOLO(
        tmp_path / "absent.pt", ["mouse"], negative_class_ids=()
    ).readiness()
    assert not status.ready
    assert status.reason == "missing_weights"
    assert "absent.pt" in status.message


def test_readiness_ready_when_weights_present(tmp_path):
    pytest.importorskip("ultralytics")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    assert TrainedYOLO(weights, ["mouse"], negative_class_ids=()).readiness().ready


def _build_seg_weights(path):
    """A real, loadable segmentation checkpoint, built offline from the bundled config.

    Goes through ``import_ultralytics`` rather than importing ultralytics directly — that is
    the one seam that sets ``YOLO_OFFLINE`` before the first import, which is the only moment
    it can be set (the value is snapshotted into module constants at import time, and it must
    be the string "true"; "1" is silently ignored).
    """
    import_ultralytics()
    from ultralytics import YOLO

    # A `.yaml` names an architecture in the installed package; a `.pt` names a release asset
    # to be fetched. Only the first is a local operation.
    YOLO("yolo11n-seg.yaml").save(str(path))
    return path


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_load_and_predict_ignores_prompt(tmp_path):
    pytest.importorskip("ultralytics")
    weights = _build_seg_weights(tmp_path / "stand_in-seg.pt")

    model = TrainedYOLO(
        weights, classes=["person"], negative_class_ids=()
    )  # closed-set class list
    model.load("cpu")
    model.set_conf_threshold(0.05)
    frame = np.zeros((64, 64, 3), np.uint8)
    # closed-set ignores the prompt — pass a non-empty one and it must still run
    result = model.predict(
        frame, Prompt(targets=[PromptTarget("ignored", TextCue("ignored"))])
    )
    assert isinstance(result, FrameResult)
    assert isinstance(result.detections(), list)  # may be empty on a blank frame


def _detecting_seg_weights(path):
    """A stand-in segmentation checkpoint whose random weights *do* detect.

    A freshly built net scores every anchor near zero and finds nothing, and two paths that
    agree on nothing prove nothing. Lifting the class logits makes nearly every anchor score,
    so the comparison below has hundreds of boxes and masks per frame to disagree on.
    """
    import torch

    import_ultralytics()
    from ultralytics import YOLO

    torch.manual_seed(0)
    built = YOLO("yolo11n-seg.yaml")
    for branch in built.model.model[-1].cv3:
        branch[-1].bias.data += 12.0  # from about -10 (Ultralytics' prior) to about +2
    built.save(str(path))
    return path


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_batch_is_ultralytics_predict_split_at_its_own_stages(tmp_path):
    # Class guarded: **the split predict (§7 *Execution*: letterbox on the prep thread, the
    # forward on the inference thread) returns something other than Ultralytics' own
    # ``predict``** — a skipped /255, a lost BGR flip, a letterbox padded differently — or an
    # Ultralytics release moves a stage the split calls by hand.
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    weights = _detecting_seg_weights(tmp_path / "detecting-seg.pt")
    classes = ["mouse"]
    model = TrainedYOLO(weights, classes, negative_class_ids=())
    model.load("cpu")
    model.set_conf_threshold(0.05)
    reference = YOLO(str(weights))
    rng = np.random.default_rng(0)

    def frame(h, w):
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    batches = [
        [frame(96, 128) for _ in range(3)],
        [frame(96, 128)],
        [frame(96, 128), frame(64, 160)],  # mixed shapes: the full-square letterbox
    ]
    found = 0
    for frames in batches:
        got = model.predict_batch(frames, [Prompt(targets=[])] * len(frames))
        want = reference.predict(
            frames, conf=0.05, device="cpu", retina_masks=True, verbose=False
        )
        assert len(got) == len(want) == len(frames)
        for image, result, ultra in zip(frames, got, want):
            h, w = image.shape[:2]
            expected = FrameResult.from_detections(
                0,
                0,
                trained_yolo._detections(
                    trained_yolo._host_arrays(ultra),
                    classes,
                    w,
                    h,
                    negative_class_ids=(),
                ),
            ).detections()
            detections = result.detections()
            assert len(detections) == len(expected)
            for a, b in zip(detections, expected):
                assert (a.bbox, a.confidence, a.class_name) == (
                    b.bbox,
                    b.confidence,
                    b.class_name,
                )
                np.testing.assert_array_equal(a.mask, b.mask)
            found += len(detections)
    assert found > 100  # the comparison saw boxes and masks, not two empty lists

    # Still one batched forward pass per call.
    predictor = model._split.predictor(batches[0])
    forwards = []
    inference = predictor.inference

    def counted(im, *args, **kwargs):
        forwards.append(im.shape[0])
        return inference(im, *args, **kwargs)

    predictor.inference = counted
    model.predict_batch(batches[0], [Prompt(targets=[])] * 3)
    assert forwards == [3]


def _block_connections(monkeypatch):
    """Make any outbound socket connection raise. Creating a socket stays legal — only
    reaching a remote host is what this forbids."""
    import socket

    def refuse(*a, **k):
        raise AssertionError("this test must not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_the_stand_in_weights_are_built_without_touching_the_network(
    tmp_path, monkeypatch
):
    """The guard for the class: a test that reaches the network has no bound on its runtime.

    This one downloaded a release asset into a fresh ``tmp_path`` on *every* full-suite run,
    and its ``try/except -> skip`` covered a failed download but not a stalled one — so a slow
    host parked the whole run until the 120 s watchdog dumped stacks and killed it. Blocking
    connections is what makes a reintroduced download fail loudly and immediately instead of
    hanging.
    """
    pytest.importorskip("ultralytics")
    _block_connections(monkeypatch)

    weights = _build_seg_weights(tmp_path / "offline-seg.pt")
    assert weights.is_file()
    assert weights.stat().st_size > 0

    model = TrainedYOLO(weights, classes=["person"], negative_class_ids=())
    model.load("cpu")
    result = model.predict(np.zeros((64, 64, 3), np.uint8), Prompt(targets=[]))
    assert isinstance(result, FrameResult)


def test_the_network_block_would_catch_a_download(tmp_path, monkeypatch):
    """Guard the guard: a blocker that blocks nothing passes forever."""
    import socket

    _block_connections(monkeypatch)
    with pytest.raises(AssertionError, match="must not reach the network"):
        socket.create_connection(("example.invalid", 80))
    with socket.socket() as sock:  # `with`, or the unclosed socket is a ResourceWarning
        with pytest.raises(AssertionError, match="must not reach the network"):
            sock.connect(("example.invalid", 80))


def _non_checkpoints(tmp_path) -> list:
    """Files a user picks for ``model import`` by mistake, each of which made Ultralytics
    8.4.215 raise from ``YOLO(path)`` (#33 finding 7)."""
    import torch

    (tmp_path / "notes.pt").write_text(
        "some notes\n", encoding="utf-8"
    )  # text, renamed
    whole = _build_seg_weights(tmp_path / "whole.pt").read_bytes()
    (tmp_path / "cut.pt").write_bytes(whole[: len(whole) // 2])  # a copy cut short
    torch.save({"a": 1}, tmp_path / "dict.pt")  # a PyTorch file holding no model
    torch.save([1, 2, 3], tmp_path / "list.pt")
    torch.save(torch.nn.Linear(2, 2).state_dict(), tmp_path / "weights.pt")
    return [
        tmp_path / n for n in ("notes.pt", "cut.pt", "dict.pt", "list.pt", "weights.pt")
    ]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_file_that_is_not_a_checkpoint_is_refused_by_name(tmp_path):
    """The real Ultralytics on the real files: every one is a ``UserInputError`` naming the
    file, whatever ``YOLO(path)`` raised. Where Ultralytics is not installed the same
    reader is driven with each observed error by ``tests/test_cli_user_errors.py``."""
    pytest.importorskip("ultralytics")
    from detail.core.errors import UserInputError

    for path in _non_checkpoints(tmp_path):
        with pytest.raises(
            UserInputError, match=f"{path.name} is not a YOLO .pt checkpoint"
        ):
            TrainedYOLO.introspect(path)
    classes, task = TrainedYOLO.introspect(tmp_path / "whole.pt")
    assert task == "segment" and classes
