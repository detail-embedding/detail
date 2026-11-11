import sys
import types

import numpy as np
import pytest

from detail.core.registries import EMBEDDING_MODELS
from detail.models.dinov3 import (
    DINOv3ConvNextBase,
    DINOv3ViT,
    _read_output_dim,
)
from detail.models.hf_access import model_cache_dir


def test_registered_and_attrs():
    assert EMBEDDING_MODELS.get("dinov3_convnext_base") is DINOv3ConvNextBase
    # hf_repo is resolved from the variant at construction; default is convnext_base
    assert (
        DINOv3ConvNextBase().hf_repo
        == "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    )
    assert "convnext_base" in DINOv3ConvNextBase.variants()
    assert DINOv3ConvNextBase("convnext_small").hf_repo.endswith(
        "convnext-small-pretrain-lvd1689m"
    )
    assert DINOv3ConvNextBase.gated is True
    assert DINOv3ConvNextBase.supported_backends == ["cuda", "mps", "cpu"]


def test_output_dim_uninitialized_before_load():
    # Instance-level default before load() runs; the runner reads this *after* load.
    assert DINOv3ConvNextBase().output_dim == 0


# --- readiness surfaces missing backend deps (Tenet 17) --------------------- #
def test_readiness_reports_missing_transformers(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        return None if name == "transformers" else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    status = DINOv3ConvNextBase().readiness()
    assert status.ready is False
    assert status.reason == "missing_dependency"
    assert "transformers" in status.message


# --- output_dim is read from the model config, not hardcoded ---------------- #
def test_read_output_dim_convnext_uses_last_hidden_size():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(hidden_sizes=[128, 256, 512, 1024])
    )
    assert _read_output_dim(model) == 1024


def test_read_output_dim_vit_uses_hidden_size():
    model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=768))
    # No hidden_sizes attr -> falls back to hidden_size.
    assert _read_output_dim(model) == 768


def test_read_output_dim_convnext_takes_precedence_over_hidden_size():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(hidden_sizes=[1024], hidden_size=999)
    )
    assert _read_output_dim(model) == 1024


# --- embed / embed_batch wired through a mocked transformers stack ---------- #
class _FakeProcessor:
    def __init__(self, size=None):
        self.last_batch_size = 0
        # Echo the configured target size into the emitted pixel_values, so a test can
        # confirm the model is fed exactly the resolution load() asked for. Defaults to 224
        # when load() passes no size override (the "processor silently re-resizes" case).
        self._h = (size or {}).get("height", 224)
        self._w = (size or {}).get("width", 224)
        # the normalisation statistics patch_grids reads (DINOv3 uses ImageNet's)
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def __call__(self, images, return_tensors):
        import torch

        self.last_batch_size = len(images)
        return _FakeInputs(
            {"pixel_values": torch.zeros((len(images), 3, self._h, self._w))}
        )

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        _FakeProcessor.last_kwargs = kwargs
        return cls(size=kwargs.get("size"))


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeOutputs:
    def __init__(self, pooler):
        import torch

        self.pooler_output = pooler
        self.last_hidden_state = torch.zeros((pooler.shape[0], 5, pooler.shape[1]))


class _FakeModel:
    def __init__(self, output_dim):
        self.config = types.SimpleNamespace(hidden_sizes=[output_dim])
        self._output_dim = output_dim

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        _FakeModel.last_kwargs = kwargs
        return cls(output_dim=1024)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        import torch

        pv = inputs["pixel_values"]
        _FakeModel.last_pixel_shape = tuple(
            pv.shape
        )  # what resolution the model received
        batch = pv.shape[0]
        return _FakeOutputs(
            torch.arange(batch * self._output_dim, dtype=torch.float32).reshape(
                batch, self._output_dim
            )
        )


@pytest.fixture
def fake_transformers(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = _FakeProcessor
    transformers.AutoModel = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _frame():
    return np.zeros((224, 224, 3), dtype=np.uint8)


def test_load_reads_our_cache_and_never_fetches(fake_transformers):
    """The bug: load() steered the cache with os.environ["HF_HUB_CACHE"] after
    huggingface_hub had already snapshotted it into a module constant, so weights
    landed in the default cache while is_installed() probed model_cache_dir() — a
    dead guard and a live auth_check per recording. cache_dir must be passed
    explicitly, and local_files_only keeps load() off the network entirely."""
    DINOv3ConvNextBase().load("cpu")
    for fake in (_FakeProcessor, _FakeModel):
        assert fake.last_kwargs["cache_dir"] == str(model_cache_dir())
        assert fake.last_kwargs["local_files_only"] is True


def test_load_sets_output_dim_from_model_config(fake_transformers):
    model = DINOv3ConvNextBase()
    model.load("cpu")
    assert model.output_dim == 1024  # set by _FakeModel.config.hidden_sizes


def test_embed_batch_returns_b_by_d(fake_transformers):
    model = DINOv3ConvNextBase()
    model.load("cpu")
    out = model.embed_batch([_frame(), _frame(), _frame()])
    arr = np.asarray(out)
    assert arr.shape == (3, 1024)
    assert arr.dtype == np.float32


def test_embed_single_image_returns_d(fake_transformers):
    model = DINOv3ConvNextBase()
    model.load("cpu")
    vec = model.embed(_frame())
    assert vec.shape == (1024,)


def test_embed_before_load_raises():
    model = DINOv3ConvNextBase()
    with pytest.raises(RuntimeError, match="load.. must be called"):
        model.embed(_frame())


# --- ViT-style fallback: pooler_output is None, last_hidden_state is pooled - #
class _FakeViTOutputs:
    def __init__(self, last_hidden_state):
        self.pooler_output = None
        self.last_hidden_state = last_hidden_state


class _FakeViTModel(_FakeModel):
    def __init__(self, output_dim):
        super().__init__(output_dim)
        # ViT-style: no hidden_sizes, just hidden_size
        self.config = types.SimpleNamespace(hidden_size=output_dim)

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        return cls(output_dim=768)

    def __call__(self, **inputs):
        import torch

        batch = inputs["pixel_values"].shape[0]
        # (B, N+1=5, D)
        return _FakeViTOutputs(torch.ones((batch, 5, self._output_dim)))


@pytest.fixture
def fake_transformers_vit(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = _FakeProcessor
    transformers.AutoModel = _FakeViTModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_vit_pooling_falls_back_to_mean_of_last_hidden_state(fake_transformers_vit):
    model = DINOv3ConvNextBase()
    model.load("cpu")
    out = model.embed_batch([_frame()])
    arr = np.asarray(out)
    assert arr.shape == (1, 768)
    # mean of ones is 1 — verifies the pooling path
    np.testing.assert_allclose(arr, 1.0)


# --- G6: spatial read-outs from the ConvNeXt final map (B, C, H, W) --------- #
def _pooled(readout, feat):
    from detail.models.dinov3 import _pooled_features

    return np.asarray(
        _pooled_features(
            types.SimpleNamespace(pooler_output=None, last_hidden_state=feat), readout
        )
    )


def test_readout_validation_rejects_unknown():
    with pytest.raises(ValueError, match="unknown readout"):
        DINOv3ConvNextBase(readout="nonsense")


def test_readout_default_is_pooler_and_unchanged():
    assert DINOv3ConvNextBase().readout == "pooler"


def test_readout_gap_gmp_are_distinct_pools_of_the_final_map():
    import torch

    # (B=1, C=3, H=2, W=2): each channel a known 2x2 patch grid.
    feat = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 4.0], [4.0, 4.0]],
                [[-1.0, -3.0], [0.0, 0.0]],
            ]
        ]
    )
    gap = _pooled("gap", feat)
    gmp = _pooled("gmp", feat)
    gap_gmp = _pooled("gap_gmp", feat)
    np.testing.assert_allclose(gap[0], [1.5, 4.0, -1.0])  # spatial mean per channel
    np.testing.assert_allclose(gmp[0], [3.0, 4.0, 0.0])  # spatial max per channel
    assert gap_gmp.shape == (1, 6)  # concat doubles the width
    np.testing.assert_allclose(gap_gmp[0], [1.5, 4.0, -1.0, 3.0, 4.0, 0.0])


def test_readout_gap_gmp_doubles_output_dim(fake_transformers):
    model = DINOv3ConvNextBase(readout="gap_gmp")
    model.load("cpu")
    assert model.output_dim == 2048  # 2 * config hidden_sizes[-1] (1024)


# --- DINOv3 ViT: registration, read-outs, and the CLS/register prefix slice --- #
class _FakeViTFull:
    """A realistic ViT output: pooler_output = the CLS token, and last_hidden_state laid out as
    ``[CLS, 4 registers, 4 patches]`` with the prefix set to a sentinel (99) so a correct pool over
    the patch tokens (1,2,3,4 -> mean 2.5) ignores it. Records the resolution it was fed.
    """

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        _FakeViTFull.last_kwargs = kwargs
        return cls()

    def __init__(self):
        self.config = types.SimpleNamespace(hidden_size=4, num_register_tokens=4)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        import torch

        pv = inputs["pixel_values"]
        _FakeViTFull.last_pixel_shape = tuple(pv.shape)
        b = pv.shape[0]
        cls_row = torch.full((b, 1, 4), 7.0)  # token 0 = CLS (== pooler_output)
        registers = torch.full((b, 4, 4), 99.0)  # 4 registers (must be excluded)
        patches = torch.stack(
            [torch.full((b, 4), float(i + 1)) for i in range(4)], dim=1
        )  # 4 patch tokens: 1,2,3,4
        last_hidden_state = torch.cat([cls_row, registers, patches], dim=1)  # (b, 9, 4)
        return types.SimpleNamespace(
            pooler_output=cls_row[:, 0, :], last_hidden_state=last_hidden_state
        )


@pytest.fixture
def fake_transformers_vit_full(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = _FakeProcessor
    transformers.AutoModel = _FakeViTFull
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def test_vit_registered_and_attrs():
    assert EMBEDDING_MODELS.get("dinov3_vit") is DINOv3ViT
    assert DINOv3ViT().hf_repo == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert DINOv3ViT.default_variant() == "vit_large"
    assert DINOv3ViT.input_size == 224
    assert set(DINOv3ViT.variants()) == {"vit_small", "vit_base", "vit_large"}
    assert "cls_patch" in DINOv3ViT.READOUTS
    assert DINOv3ViT.VARIANT_OUTPUT_DIMS["vit_large"] == 1024
    assert DINOv3ViT.gated is True


def test_vit_load_passes_cache_and_local_only(fake_transformers_vit_full):
    DINOv3ViT().load("cpu")
    for fake in (_FakeProcessor, _FakeViTFull):
        assert fake.last_kwargs["cache_dir"] == str(model_cache_dir())
        assert fake.last_kwargs["local_files_only"] is True


def test_vit_gap_excludes_cls_and_register_prefix(fake_transformers_vit_full):
    # gap over the patch tokens (1,2,3,4) = 2.5, NOT the CLS (7) + register (99) prefix.
    # The pre-fix bug (pooling all 9 tokens) would give (7 + 99*4 + 10) / 9 = 45.9.
    model = DINOv3ViT(readout="gap")
    model.load("cpu")
    out = np.asarray(model.embed_batch([_frame()]))
    np.testing.assert_allclose(out, 2.5)


def test_vit_pooler_is_the_cls_token(fake_transformers_vit_full):
    model = DINOv3ViT(readout="pooler")
    model.load("cpu")
    out = np.asarray(model.embed_batch([_frame()]))
    np.testing.assert_allclose(out, 7.0)  # pooler_output = CLS


def test_vit_cls_patch_concats_cls_and_patch_mean(fake_transformers_vit_full):
    model = DINOv3ViT(readout="cls_patch")
    model.load("cpu")
    assert model.output_dim == 8  # cls_patch doubles: 2 * hidden_size (4)
    out = np.asarray(model.embed_batch([_frame()]))
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0], [7, 7, 7, 7, 2.5, 2.5, 2.5, 2.5])


def test_convnext_has_no_cls_patch_readout():
    with pytest.raises(ValueError, match="unknown readout"):
        DINOv3ConvNextBase(readout="cls_patch")


# --- the point of the feature: the model is fed exactly input_size ----------- #
@pytest.mark.parametrize("size,expected", [(None, 224), (384, 384)])
def test_input_size_reaches_the_convnext_model(fake_transformers, size, expected):
    """Constructs with input_size and asserts the model's forward receives a tensor of exactly
    that resolution. FAILS if the processor-size override is removed (the 'silently re-resizes
    to 224' trap), since the fake processor then defaults to 224."""
    model = DINOv3ConvNextBase(input_size=size)
    model.load("cpu")
    model.embed_batch([_frame(), _frame()])
    assert _FakeModel.last_pixel_shape == (2, 3, expected, expected)
    assert _FakeProcessor.last_kwargs["size"] == {
        "height": expected,
        "width": expected,
    }


def test_input_size_reaches_the_vit_model(fake_transformers_vit_full):
    model = DINOv3ViT(input_size=512, readout="gap")
    model.load("cpu")
    model.embed_batch([_frame()])
    assert _FakeViTFull.last_pixel_shape == (1, 3, 512, 512)
    assert _FakeProcessor.last_kwargs["size"] == {"height": 512, "width": 512}


# --- PatchFeatures: dense patch grids for patch matching (§14, §26) ---------- #
def test_vit_declares_patch_features_and_the_patch_size():
    from detail.core.protocols import PatchFeatures

    assert issubclass(DINOv3ViT, PatchFeatures) and DINOv3ViT.patch_size == 16
    assert not issubclass(DINOv3ConvNextBase, PatchFeatures)


def test_patch_grids_are_the_normalised_patch_tokens_without_the_prefix(
    fake_transformers_vit_full,
):
    """A 32x32 image is 2x2 patches at 16 px; the fake's 4 patch tokens are the constants
    1, 2, 3, 4 behind a CLS and 4 register tokens (99): a grid that includes the prefix or
    skips the normalisation cannot come out as four unit vectors of 0.5."""
    model = DINOv3ViT()
    model.load("cpu")
    [grid] = model.patch_grids([np.zeros((32, 32, 3), np.uint8)])
    arr = np.asarray(grid)
    assert arr.shape == (4, 2, 2)  # (C, H/16, W/16)
    np.testing.assert_allclose(np.linalg.norm(arr, axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(arr, 0.5, atol=1e-6)
    assert _FakeViTFull.last_pixel_shape == (1, 3, 32, 32)  # never resized


def test_patch_grids_refuse_sides_that_are_not_a_multiple_of_the_patch(
    fake_transformers_vit_full,
):
    model = DINOv3ViT()
    model.load("cpu")
    with pytest.raises(ValueError, match="multiples of 16; got 30x30"):
        model.patch_grids([np.zeros((30, 30, 3), np.uint8)])


def test_patch_grids_before_load_raises():
    with pytest.raises(
        RuntimeError, match="load\\(\\) must be called before patch_grids"
    ):
        DINOv3ViT().patch_grids([np.zeros((32, 32, 3), np.uint8)])


# --- the CPU / device split the embedding worker runs on two threads (§14) ---- #
class _PixelProcessor(_FakeProcessor):
    """Emits pixel_values from the images themselves, and records every device copy, so the
    split can be checked for both identity and where the copy happens."""

    to_calls: list = []

    def __call__(self, images, return_tensors):
        import torch

        arrays = [np.asarray(im, dtype=np.float32) for im in images]
        pv = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2) / 255.0
        return _RecordingInputs({"pixel_values": pv})


class _RecordingInputs(dict):
    def to(self, device):
        _PixelProcessor.to_calls.append(device)
        return self


class _PixelModel(_FakeModel):
    def __call__(self, **inputs):
        pv = inputs["pixel_values"]
        flat = pv.reshape(pv.shape[0], -1)
        return _FakeOutputs(flat[:, : self._output_dim].repeat(1, 1))


@pytest.fixture
def pixel_transformers(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = _PixelProcessor
    transformers.AutoModel = _PixelModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    _PixelProcessor.to_calls = []


def test_embed_batch_is_exactly_its_two_halves(pixel_transformers):
    """The worker calls prepare_batch on one thread and embed_prepared on another. Their
    composition must be embed_batch, and only the device half may copy to the device."""
    rng = np.random.default_rng(3)
    images = [rng.integers(0, 256, (40, 40, 3), dtype=np.uint8) for _ in range(5)]
    model = DINOv3ConvNextBase()
    model.load("cpu")

    whole = np.asarray(model.embed_batch(images))
    copies_by_whole = len(_PixelProcessor.to_calls)
    prepared = model.prepare_batch(images)
    assert len(_PixelProcessor.to_calls) == copies_by_whole  # the CPU half never copies
    split = np.asarray(model.embed_prepared(prepared))
    assert len(_PixelProcessor.to_calls) == copies_by_whole + 1

    assert split.dtype == whole.dtype == np.float32
    np.testing.assert_array_equal(split, whole)
    # Composition alone cannot see a CPU half that mangles the batch (both sides would go
    # through it), so pin each row to its own image, computed without the model at all.
    expected = np.stack(
        [
            (im.astype(np.float32).transpose(2, 0, 1) / np.float32(255.0)).reshape(-1)[
                : model.output_dim
            ]
            for im in images
        ]
    )
    np.testing.assert_array_equal(whole, expected)


def test_both_halves_refuse_to_run_before_load():
    model = DINOv3ConvNextBase()
    with pytest.raises(RuntimeError, match="load.. must be called"):
        model.prepare_batch([_frame()])
    with pytest.raises(RuntimeError, match="load.. must be called"):
        model.embed_prepared({})
