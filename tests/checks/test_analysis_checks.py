"""The optional Cluster profile step (§1): its check, its signal, its row.

The guards:

- **The check is registered, so `detail doctor` and the GUI checklist report it**, from the one
  registry both read — and it runs in the fresh-subprocess probe, so an install made while the
  app is open shows on the next re-check.
- **Optional means it never fails and never blocks.** The check WARNs at worst (so `doctor`
  exits 0 over it), and the step never guides: setup is complete without it.
- **The step is ✓ only when SHAP is in**, and otherwise names what is missing.
"""

from __future__ import annotations

import dataclasses

import pytest

from detail.api import checks as api_checks
from detail.api import extras
from detail.api.readiness import signals_from
from detail.checks import import_checks
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.core.registries import CHECKS
from detail.gui import readiness_guide
from tests.gui.test_readiness_guide import _s


def test_the_check_is_registered_for_doctor_and_the_checklist():
    import_checks()
    ids = {info.check_id for info in api_checks.list_checks()}
    assert "shap" in ids
    assert CHECKS["shap"].isolation == "subprocess", "an install mid-session must show"


def test_the_shap_check_reports_the_version_and_only_warns(monkeypatch):
    monkeypatch.setattr(extras, "package_version", lambda name: "0.46.0")
    result = CHECKS["shap"]().run()
    assert result.status == CheckStatus.OK and "0.46.0" in result.detail
    monkeypatch.setattr(extras, "package_version", lambda name: None)
    result = CHECKS["shap"]().run()
    assert result.status == CheckStatus.WARN
    assert "permutation importance" in result.detail


def test_the_signal_carries_the_check():
    signals = signals_from([CheckResult("shap", CheckStatus.OK, "")])
    assert signals.shap is True
    assert signals_from([CheckResult("shap", CheckStatus.WARN, "")]).shap is False


def _step(signals):
    (step,) = [
        s
        for s in readiness_guide.readiness_steps(signals)
        if s.kind == readiness_guide.PROFILE_DESCRIBE_STEP
    ]
    return step


@pytest.mark.parametrize("shap, done", [(False, False), (True, True)])
def test_the_step_is_done_only_with_shap(shap, done):
    step = _step(_s(shap=shap))
    assert step.done is done and step.guides is False
    if done:
        assert step.label == "Cluster profile"
    else:
        assert step.label == "Optional: Cluster profile (needs SHAP)"


def test_the_optional_step_never_blocks_setup():
    everything = dict(
        gpu=True,
        hf_token=True,
        sam3=True,
        sam3_backend=True,
        sam3_weights=True,
        detection_model=True,
        distilled_detector=True,
        keypoint_student=True,
    )
    assert readiness_guide.is_setup_complete(_s(**everything))
    assert readiness_guide.next_step_kind(_s(**everything)) is None
    assert (
        readiness_guide.guides_step(
            readiness_guide.PROFILE_DESCRIBE_STEP, _s(**everything)
        )
        is None
    ), "no banner for an optional step"


def test_the_step_is_built_from_the_signal_fields_by_name():
    # A ReadinessSignals with a field missing fails to construct; this pins that the field
    # exists rather than riding on a default that hides a missing mapping.
    names = {f.name for f in dataclasses.fields(_s())}
    assert "shap" in names
