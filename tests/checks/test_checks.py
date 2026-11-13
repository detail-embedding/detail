"""Live checker framework (§1): the registry, each check body, and the runner."""

import os
import subprocess

import pytest

from detail.api import checks as api_checks
from detail.checks import import_checks
from detail.core.dataclasses import CheckStatus
from detail.core.registries import CHECKS


def _by_id(results):
    return {r.check_id: r for r in results}


def test_registry_lists_expected_checks():
    import_checks()
    assert {
        "torch",
        "cuda",
        "nvidia_gpu",
        "sam3_backend",
        "triton",
        "dinov3_backend",
        "psutil",
        "hf_token",
        "sam3_weights",
        "dinov3_weights",
    } <= set(CHECKS)


def test_env_checks_read_find_spec(monkeypatch):
    present = {"torch", "transformers", "sam3", "psutil"}
    monkeypatch.setattr("detail.checks.environment._has", lambda m: m in present)
    res = _by_id(api_checks.run_checks(in_process=True))
    assert res["torch"].status == CheckStatus.OK
    assert res["sam3_backend"].status == CheckStatus.OK
    assert res["dinov3_backend"].status == CheckStatus.OK
    assert res["psutil"].status == CheckStatus.OK

    present.difference_update({"sam3", "torch", "psutil"})
    res = _by_id(api_checks.run_checks(in_process=True))
    assert res["torch"].status == CheckStatus.FAIL
    assert res["sam3_backend"].status == CheckStatus.FAIL
    assert res["psutil"].status == CheckStatus.FAIL


def test_cuda_check_both_branches(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert api_checks.run_check("cuda").status == CheckStatus.OK
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert api_checks.run_check("cuda").status == CheckStatus.WARN


def test_nvidia_gpu_check_asks_the_driver_not_torch(monkeypatch):
    # The check `cuda` cannot answer: a CPU-only torch wheel says "no CUDA device" on a box
    # whose NVIDIA driver is right there. So this one reads nvidia-smi, never torch.
    monkeypatch.setattr(
        "detail.checks.environment.hostinfo.nvidia_smi", lambda: ("560.94", "12.6")
    )
    result = api_checks.run_check("nvidia_gpu")
    assert result.status == CheckStatus.OK
    assert "560.94" in result.detail and "12.6" in result.detail
    monkeypatch.setattr(
        "detail.checks.environment.hostinfo.nvidia_smi", lambda: (None, None)
    )
    result = api_checks.run_check("nvidia_gpu")
    # No NVIDIA GPU is not a problem to fix, so it neither warns nor fails.
    assert result.status == CheckStatus.SKIPPED
    assert "no NVIDIA driver" in result.detail


def test_triton_is_skipped_exactly_off_windows():
    # The property is the platform gate, not one platform's answer. Written as
    # "== SKIPPED" it asserted the Linux branch unconditionally and could not pass on
    # the platform the check exists for; on Windows the check really runs, so the most
    # this can say there is that it did not skip.
    status = api_checks.run_check("triton").status
    if os.name == "nt":
        assert status != CheckStatus.SKIPPED
    else:
        assert status == CheckStatus.SKIPPED


def test_hf_token_check(monkeypatch):
    monkeypatch.setattr("detail.storage.credentials.get_hf_token", lambda: "tok")
    assert api_checks.run_check("hf_token").status == CheckStatus.OK
    monkeypatch.setattr("detail.storage.credentials.get_hf_token", lambda: None)
    assert api_checks.run_check("hf_token").status == CheckStatus.WARN


def test_weights_checks(monkeypatch):
    monkeypatch.setattr(
        "detail.api.models.is_installed", lambda mid, *a, **k: mid == "sam3"
    )
    assert api_checks.run_check("sam3_weights").status == CheckStatus.OK
    assert api_checks.run_check("dinov3_weights").status == CheckStatus.WARN


def test_the_dinov3_check_asks_for_the_default_embedding_model(monkeypatch):
    """The setup check points a new user at the weights a run with no ``--model`` needs —
    the config default at its default size (no variant) — not a model of its own choosing.
    """
    from detail.configs.embedding import EmbeddingConfig

    asked = []
    monkeypatch.setattr(
        "detail.api.models.is_installed",
        lambda mid, *a, **k: asked.append((mid, a, k)) or True,
    )
    api_checks.run_check("dinov3_weights")
    assert asked == [(EmbeddingConfig.model_fields["model_id"].default, (), {})]


def test_run_check_unknown_raises():
    with pytest.raises(KeyError):
        api_checks.run_check("nope")


def test_inline_check_degrades_on_raise(monkeypatch):
    import_checks()

    def _raise(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(CHECKS["hf_token"], "run", _raise)
    assert api_checks.run_check("hf_token").status == CheckStatus.UNKNOWN


def test_subprocess_path_parses_json_past_chatter(monkeypatch):
    import json

    canned = {"torch": {"check_id": "torch", "status": "ok", "detail": "installed"}}

    class _Proc:
        stdout = "Error importing something\n" + json.dumps(canned)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    res = _by_id(api_checks.run_checks())
    assert res["torch"].status == CheckStatus.OK and res["torch"].detail == "installed"
    # a subprocess check the probe didn't answer degrades, not crashes
    assert res["cuda"].status == CheckStatus.UNKNOWN


def test_subprocess_degrades_on_timeout(monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=1)

    monkeypatch.setattr(subprocess, "run", _boom)
    res = _by_id(api_checks.run_checks())
    assert res["torch"].status == CheckStatus.UNKNOWN


def test_real_probe_emits_valid_json():
    # One real spawn: the child returns a parseable status for every subprocess check.
    res = _by_id(api_checks.run_checks())
    for cid in ("torch", "cuda", "sam3_backend", "dinov3_backend", "triton", "psutil"):
        assert res[cid].status in set(CheckStatus)
