"""The active guided layer (§1): sidebar next-step marker, dismissal, page banners."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QSettings  # noqa: E402

from detail.api.readiness import ReadinessSignals  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.gui.test_configuration_pages import _SyncBackgroundCall  # noqa: E402


def _s(**kw):
    kw.setdefault(
        "sam3", kw.get("sam3_backend", False) and kw.get("sam3_weights", False)
    )
    base = dict(
        gpu=False,
        hf_token=False,
        sam3=False,
        sam3_backend=False,
        sam3_weights=False,
        detection_model=False,
        has_experiment=False,
    )
    base.update(kw)
    return ReadinessSignals(**base)


def _window(tmp_path, monkeypatch):
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return MainWindow()


# The sidebar next-step ``●`` marker was removed with the banner-nav redesign — banner
# modes are full-width (no rail), so the next step now shows on the Configuration checklist
# (``_ReadinessPanel``); its ordering logic is covered in tests/gui/test_readiness_guide.py.


# --- dismissal (QSettings, test-mode) --------------------------------------- #
def test_dismissal_persists_and_suppresses_banner(qapp, tmp_path, monkeypatch):
    from detail.gui.app import _DISMISSED_KEY

    QSettings().remove(_DISMISSED_KEY)  # clean slate in the isolated test store
    window = _window(tmp_path, monkeypatch)
    window._readiness = _s(gpu=True)  # both HF + model incomplete → banners available
    assert window._banner_for("global_models") is not None
    window._dismiss_step("global_models")
    assert window._banner_for("global_models") is None  # dismissed → suppressed
    # persisted: a fresh window reads the same dismissal back
    assert "global_models" in _window(tmp_path, monkeypatch)._dismissed


def test_dismissal_resets_when_the_step_becomes_done(qapp, tmp_path, monkeypatch):
    from detail.gui.app import _DISMISSED_KEY

    QSettings().remove(_DISMISSED_KEY)
    window = _window(tmp_path, monkeypatch)
    window._dismiss_step("global_models")
    assert "global_models" in window._dismissed
    # the model step is now done → _on_readiness prunes its stale dismissal
    window._on_readiness(_s(gpu=False, detection_model=True))
    assert "global_models" not in window._dismissed


# --- page banners + change notification ------------------------------------- #
def test_hf_banner_dismiss_and_success_wire_callbacks(qapp, monkeypatch):
    dismissed, changed = [], []
    page = pages.HuggingFacePage(
        banner="Next step — Hugging Face access.",
        on_banner_dismiss=lambda: dismissed.append(True),
        on_readiness_changed=lambda: changed.append(True),
    )
    assert page._banner is not None
    page._banner._dismiss()  # the × calls on_banner_dismiss + hides
    assert dismissed == [True] and page._banner.isHidden()

    monkeypatch.setattr(pages, "BackgroundCall", _SyncBackgroundCall)
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)
    monkeypatch.setattr(
        "detail.gui.pages.credentials.set_hf_token", lambda t: "octocat"
    )
    page._token.setText("hf_x")
    page._save_token()
    assert changed == [True]  # a saved token flips a signal → notify the shell


def test_model_install_notifies_only_on_success(qapp, monkeypatch):
    monkeypatch.setattr("detail.gui.pages.QMessageBox.critical", lambda *a, **k: None)
    changed = []
    page = pages.GlobalModelsPage(on_readiness_changed=lambda: changed.append(True))
    page._on_install_done("")  # success
    assert changed == [True]
    changed.clear()
    page._on_install_done("boom")  # error → no notification
    assert changed == []


def test_gpu_page_never_gets_a_banner(qapp, tmp_path, monkeypatch):
    # GPU is terminal when no NVIDIA is present — it never guides, so no banner.
    window = _window(tmp_path, monkeypatch)
    window._readiness = _s(gpu=False)
    assert window._banner_for("gpu_cuda") is None
