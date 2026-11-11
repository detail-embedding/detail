"""In-process setup checks (§1): disk / credential / weight-cache probes.

These answer from state that doesn't depend on importing a backend, so they run in the
app's interpreter (a re-check reflects them immediately). Import-sensitive probes live in
``checks/environment.py`` and run in a throwaway subprocess instead.
"""

from detail.configs.embedding import EmbeddingConfig
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.core.protocols import Check
from detail.core.registries import register_check
from detail.help import HelpTopic


@register_check("hf_token")
class HuggingFaceToken(Check):
    check_id = "hf_token"
    label = "Hugging Face token"
    group = "Hugging Face"
    fix = HelpTopic("getting-started", "hugging-face-access")

    def run(self, target=None) -> CheckResult:
        from detail.storage import credentials

        if credentials.get_hf_token() is not None:
            return CheckResult(self.check_id, CheckStatus.OK, "token configured")
        return CheckResult(
            self.check_id, CheckStatus.WARN, "no token — gated models can't download"
        )


class _WeightsCheck(Check):
    """Shared body: is a model's weights cached on disk (no network, no import)?"""

    group = "Models"
    model_id: str

    def run(self, target=None) -> CheckResult:
        from detail.api import models as api_models

        try:
            installed = api_models.is_installed(self.model_id)
        except Exception as exc:  # a probe never crashes the checker
            return CheckResult(
                self.check_id, CheckStatus.UNKNOWN, f"{type(exc).__name__}: {exc}"
            )
        if installed:
            return CheckResult(self.check_id, CheckStatus.OK, "weights installed")
        return CheckResult(self.check_id, CheckStatus.WARN, "weights not installed")


@register_check("sam3_weights")
class Sam3Weights(_WeightsCheck):
    check_id = "sam3_weights"
    label = "SAM 3 weights"
    model_id = "sam3"
    fix = HelpTopic("gui", "models")


@register_check("dinov3_weights")
class Dinov3Weights(_WeightsCheck):
    check_id = "dinov3_weights"
    label = "DINOv3 weights"
    # The default embedding model, at its default size (``is_installed`` with no variant) —
    # the weights a run with no ``--model`` needs. Read from the config, not written out: this
    # pointed at ConvNeXt-B while the default was another model.
    model_id = EmbeddingConfig.model_fields["model_id"].default
    fix = HelpTopic("gui", "models")
