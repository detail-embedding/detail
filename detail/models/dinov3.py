"""DINOv3 embedding models — ConvNeXt and ViT backbones (Section 14).

Weights are gated on Hugging Face (``facebook/dinov3-*-pretrain-lvd1689m``); readiness
and install come from ``HFHubModel``. ``torch`` and ``transformers`` are imported lazily
inside ``load`` / ``embed_batch`` (Tenet 13), so importing this module pulls in no
backend and registration works even when the torch extra is not installed.

Both backbones share ``_DINOv3Base``; a subclass only declares its variants, dims, and
read-outs. ``output_dim`` is read from the model's config at ``load()`` time and exposed
as an instance attribute (it differs per variant, and doubles for the concat read-outs).

The model's input resolution is ``input_size`` (a per-run override, defaulting to the
class value): ``load()`` configures the HF image processor to that square, so the size the
model actually sees is the size asked for — not silently re-resized back to the default.
ConvNeXt is convolutional and ViT is patch-16 + RoPE, so both accept sizes above 224.
"""

import contextlib

import numpy as np

from detail.core.dataclasses import ModelReadiness
from detail.core.installed import is_importable
from detail.core.protocols import EmbeddingModel, PatchFeatures
from detail.core.registries import register_embedding_model
from detail.core.utils import torch_device
from detail.models.hf_access import HFHubModel, model_cache_dir


class _DINOv3Base(HFHubModel, EmbeddingModel):
    """Shared DINOv3 backbone logic; subclasses set the variants, dims, and read-outs.

    ``_has_prefix_tokens`` marks a ViT (a ``[CLS] + registers + patches`` token sequence
    whose prefix tokens must be dropped before the spatial pools); ConvNeXt leaves it
    False (its final map is ``(B, C, H, W)`` with no prefix).
    """

    gated = True
    # Every DINOv3 repo sits behind one Meta agreement: accepting it for any backbone or
    # size grants the whole collection, so the setup flow asks for access once.
    hf_gate = "dinov3"
    supported_backends = ["cuda", "mps", "cpu"]
    _has_prefix_tokens = False

    def __init__(
        self,
        variant: str | None = None,
        readout: str = "pooler",
        input_size: int | None = None,
    ) -> None:
        super().__init__(variant)  # resolves self.variant + self.weight (the HF repo)
        if readout not in self.READOUTS:
            raise ValueError(
                f"unknown readout {readout!r}; expected one of {self.READOUTS}"
            )
        self.readout = readout
        # Per-run resolution: instance value overrides the class default; the renderer is
        # sized from the same config value (resolve_renderer_spec), so the two agree.
        self.input_size = (
            int(input_size) if input_size is not None else type(self).input_size
        )
        self.hf_repo = self.weight  # HFHubModel readiness/install/load use this
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._num_prefix = 0  # populated by load() for a ViT (CLS + register tokens)
        self.output_dim = 0  # populated by load() from the model config

    def readiness(self) -> ModelReadiness:
        """Backend deps first, then HFHubModel's cache-first weight check.

        ``load`` imports ``torch`` + ``transformers`` lazily; without them the run
        fails deep in the worker with a bare ``ModuleNotFoundError``. Surface it as
        a not-ready reason here (``find_spec`` doesn't import the backend, so
        Tenet 13 holds) so the pre-run check explains exactly what to install.
        """
        for module in ("torch", "transformers"):
            if not is_importable(module):
                return ModelReadiness(
                    ready=False,
                    reason="missing_dependency",
                    message=(
                        "DINOv3 needs PyTorch and transformers — install them with "
                        "`pip install torch transformers`."
                    ),
                    action_url="https://huggingface.co/docs/transformers/installation",
                    repo_id=self.hf_repo,
                )
        return super().readiness()

    def load(self, device: str) -> None:
        import torch  # noqa: F401 — ensures the backend is present before the pool worker runs

        from transformers import AutoImageProcessor, AutoModel

        # cache_dir + local_files_only, passed explicitly: install() put the weights
        # under model_cache_dir(), and load() must read exactly there and never fetch
        # (§5). An env var cannot do this — huggingface_hub binds HF_HUB_CACHE into a
        # module constant at import, and readiness() has already imported it.
        cache_dir = str(model_cache_dir())
        self._device = torch_device(device)
        # size= overrides the processor's target square so the model sees input_size, not
        # the processor's baked-in default. At the class default this equals the config
        # value (a no-op, so prior runs stay byte-identical). Both DINOv3 processors are
        # DINOv3ViTImageProcessorFast with size={"height","width"} and no center crop.
        self._processor = AutoImageProcessor.from_pretrained(
            self.hf_repo,
            cache_dir=cache_dir,
            local_files_only=True,
            size={"height": self.input_size, "width": self.input_size},
        )
        model = AutoModel.from_pretrained(
            self.hf_repo, cache_dir=cache_dir, local_files_only=True
        )
        model.to(self._device)
        model.eval()
        self._model = model
        self._num_prefix = (
            1 + int(getattr(model.config, "num_register_tokens", 0))
            if self._has_prefix_tokens
            else 0
        )
        # gap_gmp / cls_patch concatenate two D-vectors, so they double the width.
        self.output_dim = _read_output_dim(model) * (
            2 if self.readout in ("gap_gmp", "cls_patch") else 1
        )

    def embed(self, image: np.ndarray) -> np.ndarray:
        batch = self.embed_batch([image])  # (1, D)
        return np.asarray(batch[0])

    def _require_loaded(self) -> None:
        if self._model is None or self._processor is None:
            raise RuntimeError(
                f"{type(self).__name__}.load() must be called before embed_batch()"
            )

    def embed_batch(self, images):
        return self.embed_prepared(self.prepare_batch(images))

    def prepare_batch(self, images: list[np.ndarray]) -> object:
        """The HF image processor over the whole batch, on the CPU: the same call, on the
        same batch, that ``embed_batch`` has always made — only the device copy moved out.
        """
        self._require_loaded()
        pil_images = [_to_pil(_to_numpy(img)) for img in images]
        return self._processor(images=pil_images, return_tensors="pt")

    def embed_prepared(self, prepared: object) -> "torch.Tensor":
        import torch

        self._require_loaded()
        inputs = prepared.to(self._device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = self._model(**inputs)
        return _pooled_features(outputs, self.readout, self._num_prefix).float().cpu()


@register_embedding_model("dinov3_convnext_base")
class DINOv3ConvNextBase(_DINOv3Base):
    """DINOv3 ConvNeXt, a self-supervised image encoder (Meta); one image renders to one vector.

    The ConvNeXt backbone size is the model variant; each maps to its gated HF
    repo. ``output_dim`` is read from the loaded model's config, so all sizes work
    through the one class. (Verify the small/tiny/large repo ids on the box.)
    """

    model_id = "dinov3_convnext_base"
    architecture = "convnext"
    VARIANTS = {
        "convnext_tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "convnext_small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        "convnext_base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
        "convnext_large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    }
    DEFAULT_VARIANT = "convnext_base"
    # The repo's file manifest: what is_installed() verifies and what install()
    # downloads (allow_patterns), so the two agree by construction. Verified against
    # the convnext_base snapshot on the box; the LVD-1689m convnext repos share this
    # layout, but check on the box if a variant ships pytorch_model.bin instead —
    # is_installed() would then read False forever. (Same caveat class as input_size.)
    hf_files = ["config.json", "preprocessor_config.json", "model.safetensors"]
    # The class-default processor input; a run may override it via input_size (the
    # resolution knob). ConvNeXt is convolutional, so a larger input simply yields a
    # larger final map that the read-out pools to the same width.
    input_size = 224
    # The default-variant (convnext_base) embedding dimension, for the model catalog / docs.
    # load() overrides the instance value with the actual per-variant dim read from the config.
    output_dim = 1024
    # Embedding dim per variant, from each repo's config (hidden_sizes[-1]) — the docs read this.
    VARIANT_OUTPUT_DIMS = {
        "convnext_tiny": 768,
        "convnext_small": 768,
        "convnext_base": 1024,
        "convnext_large": 1536,
    }

    # Feature read-outs from the ConvNeXt final map (B, C, H, W). "pooler" (default) is the
    # model's own global-avg-pool + layernorm output — byte-identical to prior runs. The
    # others re-pool last_hidden_state: "gap" avg without the final layernorm, "gmp" spatial
    # max (the most-activated position — better for localized behaviors), "gap_gmp" their
    # concatenation (2*C). ConvNeXt has no CLS token, so cls_patch is ViT-only.
    READOUTS = ("pooler", "gap", "gmp", "gap_gmp")


@register_embedding_model("dinov3_vit")
class DINOv3ViT(_DINOv3Base, PatchFeatures):
    """DINOv3 ViT, a self-supervised image encoder (Meta); one image renders to one vector.

    ViT ships a real ``[CLS]`` token (``pooler_output``) and 4 register tokens that absorb
    high-norm artifacts, leaving clean patch tokens. The spatial read-outs pool the patch
    tokens only (the ``[CLS]`` + register prefix is dropped, read from
    ``config.num_register_tokens`` at load). (Verify the s/b/l repo ids + file manifest on
    the box.)
    """

    model_id = "dinov3_vit"
    architecture = "vit"
    VARIANTS = {
        "vit_small": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "vit_base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "vit_large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    DEFAULT_VARIANT = "vit_large"
    hf_files = ["config.json", "preprocessor_config.json", "model.safetensors"]
    input_size = 224
    output_dim = (
        1024  # default-variant (vit_large) dim; load() reads the real per-variant dim
    )
    VARIANT_OUTPUT_DIMS = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
    }
    _has_prefix_tokens = True

    # "pooler" (default) is the CLS token. "cls_patch" concatenates the CLS token with the
    # mean of the patch tokens (the DINOv3 image descriptor, 2*D). "gap"/"gmp"/"gap_gmp" pool
    # the patch tokens (CLS + registers excluded).
    READOUTS = ("pooler", "cls_patch", "gap", "gmp", "gap_gmp")
    #: The ViT's patch: a ``patch_grids`` image's sides must be multiples of it (§14).
    patch_size = 16

    def patch_grids(self, images) -> list:
        """One L2-normalised patch-feature grid ``(C, H/16, W/16)`` per RGB uint8 image —
        the per-patch features ``dinov3_match`` matches (§26), not a pooled read-out.

        The final layer's patch tokens after the model's final LayerNorm (DINOv3's own
        recommendation for dense features), CLS and register tokens dropped. The images are
        normalised with the processor's statistics but never resized, so a patch is exactly
        16 image pixels; they must share one shape. fp16 on CUDA, fp32 elsewhere.
        """
        import torch
        import torch.nn.functional as F

        if self._model is None or self._processor is None:
            raise RuntimeError("DINOv3ViT.load() must be called before patch_grids()")
        batch = np.stack([np.asarray(im, dtype=np.uint8) for im in images])
        _, height, width, _ = batch.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"patch_grids needs sides that are multiples of {self.patch_size}; got "
                f"{width}x{height}"
            )
        mean = torch.tensor(self._processor.image_mean, device=self._device)
        std = torch.tensor(self._processor.image_std, device=self._device)
        x = torch.from_numpy(batch).to(self._device).permute(0, 3, 1, 2).float()
        x = (x / 255.0 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        on_gpu = self._device.startswith("cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if on_gpu
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            tokens = self._model(pixel_values=x).last_hidden_state
        patches = tokens[:, self._num_prefix :, :].float()
        grid = patches.reshape(
            len(batch), height // self.patch_size, width // self.patch_size, -1
        )
        grid = F.normalize(grid.permute(0, 3, 1, 2), dim=1)
        return list((grid.half() if on_gpu else grid).unbind(0))


def _read_output_dim(model) -> int:
    """Get the model's feature dimension from its config.

    transformers ConvNextModel exposes the final stage's channel count as
    ``config.hidden_sizes[-1]``; ViT models expose ``config.hidden_size``. Try both
    so the same helper serves either DINOv3 backbone.
    """
    config = model.config
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])
    return int(config.hidden_size)


def _to_numpy(image) -> np.ndarray:
    return image if isinstance(image, np.ndarray) else np.asarray(image)


def _to_pil(image: np.ndarray):
    from PIL import Image

    return Image.fromarray(image)


def _pooled_features(outputs, readout: str = "pooler", num_prefix: int = 0):
    """Pool the model output to a (B, D) tensor per ``readout``.

    ``num_prefix`` is the count of non-patch prefix tokens (``[CLS]`` + registers) on a ViT
    sequence — 0 for a ConvNeXt map, which has none. The spatial read-outs pool the patch
    tokens only; a ViT's prefix tokens (esp. the high-norm register tokens) are excluded.

    - ``pooler``: the model's ``pooler_output`` (ConvNeXt: global-avg-pool; ViT: ``[CLS]``).
    - ``gap`` / ``gmp``: avg / max over the patch positions.
    - ``gap_gmp``: their concatenation, ``(B, 2D)``.
    - ``cls_patch`` (ViT only): ``[CLS]`` concatenated with the patch-token mean, ``(B, 2D)``.
    """
    import torch

    if readout == "pooler":
        pooler = getattr(outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        feat = (
            outputs.last_hidden_state
        )  # defensive fallback (real models have a pooler)
        if feat.dim() == 4:
            return feat.flatten(2).mean(dim=2)
        return feat[:, num_prefix:, :].mean(dim=1)

    feat = outputs.last_hidden_state
    if feat.dim() == 4:  # ConvNeXt final map (B, C, H, W) — no prefix tokens
        tokens = feat.flatten(2)  # (B, C, S)
        cls = None
    else:  # ViT token sequence (B, N, D): drop the [CLS] + register prefix
        cls = feat[:, 0, :]
        tokens = feat[:, num_prefix:, :].transpose(1, 2)  # (B, D, S_patch)
    if readout == "gap":
        return tokens.mean(dim=2)
    if readout == "gmp":
        return tokens.amax(dim=2)
    if readout == "gap_gmp":
        return torch.cat([tokens.mean(dim=2), tokens.amax(dim=2)], dim=1)
    if readout == "cls_patch":
        if cls is None:
            raise ValueError("cls_patch readout requires a ViT backbone (a CLS token)")
        return torch.cat([cls, tokens.mean(dim=2)], dim=1)
    raise ValueError(f"unknown readout {readout!r}")
