import contextlib
import subprocess
import sys
import types

import numpy as np
import pytest

from detail.core.dataclasses import Prompt, PromptTarget, TextCue
from detail.core.registries import MODELS
from detail.models.sam3 import SAM3


def test_registered_and_attrs():
    assert MODELS.get("sam3") is SAM3
    assert SAM3.hf_repo == "facebook/sam3"
    assert SAM3.gated is True
    assert SAM3.supported_cue_types == [TextCue]
    assert SAM3.supported_backends == ["cuda"]  # NVIDIA-only; no CPU/MPS path
    assert SAM3.supports_batching is False  # single-frame only (inherits the default)


class _FakeBackbone:
    def forward_text(self, texts, device):
        _FakeProcessor.encoded.append(texts[0])
        return {"language_features": f"features of {texts[0]}"}


class _FakeModel:
    def __init__(self):
        self.backbone = _FakeBackbone()

    def to(self, device):
        return self

    def eval(self):
        return self

    def _get_dummy_prompt(self):
        return "no boxes"


class _FakeProcessor:
    """The upstream surface ``SAM3._ground`` drives: ``set_image``, the backbone's
    ``forward_text``, ``_get_dummy_prompt`` and ``_forward_grounding``."""

    last: dict = {}
    encoded: list = []  # every text forward_text was asked to encode
    grounded: list = []  # the text features each grounding call found in its state

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        _FakeProcessor.last["device"] = device

    def set_image(self, image):
        _FakeProcessor.last["image_size"] = image.size
        return {"backbone_out": {"vision": "image features"}}

    def _forward_grounding(self, state):
        features = state["backbone_out"]["language_features"]
        assert state["geometric_prompt"] == "no boxes"
        _FakeProcessor.grounded.append(features)
        _FakeProcessor.last["prompt"] = features.removeprefix("features of ")
        return {
            "masks": [np.array([[True, False], [False, False]])],
            "boxes": [[0.0, 0.0, 1.0, 1.0]],  # absolute xyxy on a 2x2 frame
            "scores": [0.87],
        }

    def set_confidence_threshold(self, threshold):
        _FakeProcessor.last["confidence_threshold"] = threshold


@pytest.fixture
def fake_sam3(monkeypatch, fake_hf_snapshot):
    builder = types.ModuleType("sam3.model_builder")
    builder.last = {}

    def _build(**kwargs):
        builder.last.update(kwargs)
        return _FakeModel()

    builder.build_sam3_image_model = _build
    proc = types.ModuleType("sam3.model.sam3_image_processor")
    proc.Sam3Processor = _FakeProcessor
    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", types.ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", proc)
    monkeypatch.setattr(_FakeProcessor, "encoded", [])
    monkeypatch.setattr(_FakeProcessor, "grounded", [])
    # load() resolves the checkpoint from our cache and will not download it.
    snapshot = fake_hf_snapshot(SAM3.hf_repo, files=SAM3.hf_files)
    return types.SimpleNamespace(builder=builder, snapshot=snapshot)


def _frame():
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_load_passes_an_explicit_checkpoint_from_our_cache(fake_sam3):
    """The builder's download_ckpt_from_hf takes no cache_dir and would fetch a
    second copy into the default HF cache; it is skipped only when checkpoint_path
    is not None. Passing it is what makes load() offline."""
    SAM3().load("cpu")
    assert fake_sam3.builder.last["checkpoint_path"] == str(
        fake_sam3.snapshot / "sam3.pt"
    )


def test_load_raises_actionable_when_not_installed(fake_sam3):
    """load() never downloads — an absent checkpoint must fail here, pointing at
    install(), rather than silently fetching GBs inside a pool worker subprocess."""
    (fake_sam3.snapshot / "sam3.pt").unlink()
    with pytest.raises(FileNotFoundError, match="model install"):
        SAM3().load("cpu")


def test_predict_maps_outputs(fake_sam3):
    model = SAM3()
    model.load("cpu")
    prompt = Prompt(targets=[PromptTarget(class_name="fish", cue=TextCue("fish"))])
    result = model.predict(_frame(), prompt)

    assert len(result.detections()) == 1
    det = result.detections()[0]
    assert det.class_name == "fish"
    assert det.confidence == pytest.approx(0.87)
    assert det.bbox == (0.0, 0.0, 0.5, 0.5)  # normalized by the 2x2 frame
    assert det.mask.dtype == bool and det.mask.shape == (2, 2)
    assert _FakeProcessor.last["prompt"] == "fish"


def test_pins_non_zero_cuda_device_at_load_and_predict(fake_sam3, monkeypatch):
    # --device 1 -> resolve_device "1" -> cuda:1. SAM3 allocates current-device
    # tensors both at build time (the decoder's coord cache, in __init__) and at
    # inference (a worker thread defaulting to cuda:0), so the current device must
    # be pinned to the model's in BOTH load (before build) and predict.
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(str(d)))
    monkeypatch.setattr(torch, "autocast", lambda **k: contextlib.nullcontext())
    model = SAM3()
    model.load("1")
    assert model._device == "cuda:1"
    assert _FakeProcessor.last["device"] == "cuda:1"  # processor pinned to model
    assert calls == ["cuda:1"]  # device set before building (coord cache)
    model.predict(_frame(), Prompt(targets=[PromptTarget("fish", TextCue("fish"))]))
    assert calls == ["cuda:1", "cuda:1"]  # and again on the inference thread


def test_predict_multi_target_tags_each_class(fake_sam3):
    model = SAM3()
    model.load("cpu")
    prompt = Prompt(
        targets=[
            PromptTarget(class_name="fish", cue=TextCue("fish")),
            PromptTarget(class_name="crab", cue=TextCue("crab")),
        ]
    )
    result = model.predict(_frame(), prompt)
    assert [d.class_name for d in result.detections()] == ["fish", "crab"]


def test_each_prompt_text_is_encoded_once_per_loaded_model(fake_sam3):
    """The text encode is the same every frame, so it runs once per text per load; every
    grounding still sees its own target's features (§7: SAM3 encodes each prompt once).
    """
    model = SAM3()
    model.load("cpu")
    prompt = Prompt(
        targets=[
            PromptTarget("mouse", TextCue("mouse")),
            PromptTarget("rat", TextCue("rat")),
        ]
    )
    for _ in range(3):
        model.predict(_frame(), prompt)
    assert _FakeProcessor.encoded == ["mouse", "rat"]
    assert _FakeProcessor.grounded == ["features of mouse", "features of rat"] * 3
    model.load("cpu")  # a new model: its features are its own
    model.predict(_frame(), prompt)
    assert _FakeProcessor.encoded == ["mouse", "rat", "mouse", "rat"]


#: ``Sam3Processor.set_text_prompt``'s statements (docstring dropped) as ``SAM3._ground``
#: mirrors them, cached encode aside. If SAM3 changes the method this list stops matching:
#: re-read it and bring ``_ground`` along before updating the list.
_SET_TEXT_PROMPT = [
    "if 'backbone_out' not in state:\n"
    "    raise ValueError('You must call set_image before set_text_prompt')",
    "text_outputs = self.model.backbone.forward_text([prompt], device=self.device)",
    "state['backbone_out'].update(text_outputs)",
    "if 'geometric_prompt' not in state:\n"
    "    state['geometric_prompt'] = self.model._get_dummy_prompt()",
    "return self._forward_grounding(state)",
]


def test_ground_mirrors_the_upstream_set_text_prompt():
    """``_ground`` re-does ``Sam3Processor.set_text_prompt`` with the encode cached, through
    the processor's private ``_forward_grounding``; nothing in its own behaviour would show
    an upstream change, so the guard is the upstream method's source. Read from the file,
    not imported (no torch): skipped where the sam3 package is not installed."""
    import ast
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("sam3")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("the sam3 backend package is not installed")
    source = Path(next(iter(spec.submodule_search_locations)))
    source = source / "model" / "sam3_image_processor.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    (method,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "set_text_prompt"
    ]
    body = method.body[1:] if ast.get_docstring(method) else method.body
    assert [ast.unparse(stmt) for stmt in body] == _SET_TEXT_PROMPT


def test_predict_before_load_raises():
    model = SAM3()
    with pytest.raises(RuntimeError, match="load"):
        model.predict(
            _frame(),
            Prompt(targets=[PromptTarget("fish", TextCue("fish"))]),
        )


def test_set_conf_threshold_forwards_to_processor(fake_sam3):
    model = SAM3()
    model.load("cpu")
    model.set_conf_threshold(0.25)
    assert _FakeProcessor.last["confidence_threshold"] == 0.25


def test_set_conf_threshold_before_load_raises():
    model = SAM3()
    with pytest.raises(RuntimeError, match="load.. must be called"):
        model.set_conf_threshold(0.25)


def test_predict_rejects_non_text_cue(fake_sam3):
    from detail.core.dataclasses import BoxCue

    model = SAM3()
    model.load("cpu")
    prompt = Prompt(
        targets=[PromptTarget("fish", BoxCue(boxes=[(0.0, 0.0, 1.0, 1.0)]))]
    )
    with pytest.raises(TypeError, match="TextCue"):
        model.predict(_frame(), prompt)


def test_module_import_pulls_no_backend():
    # Importing the model module registers "sam3" but must not import torch or the
    # sam3 backend (Tenet 13).
    code = (
        "import detail.models.sam3, sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'sam3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_readiness_flags_missing_backend_package(monkeypatch):
    # The honesty fix: weights on disk aren't enough — the sam3 package must import, or
    # the run crashes deep in the worker at `import sam3`. is_importable catches its
    # absence without importing the backend (Tenet 13).
    monkeypatch.setattr("detail.models.sam3.is_importable", lambda m: False)
    readiness = SAM3().readiness()
    assert readiness.ready is False
    assert readiness.reason == "missing_dependency"
    assert readiness.action_url  # points at the sam3 backend setup


def test_readiness_defers_to_weights_when_backend_present(
    monkeypatch, fake_hf_snapshot
):
    fake_hf_snapshot(SAM3.hf_repo, files=SAM3.hf_files)
    monkeypatch.setattr("detail.models.sam3.is_importable", lambda m: True)
    readiness = SAM3().readiness()
    assert readiness.reason != "missing_dependency"  # backend ok → weights check
    assert readiness.ready is True
