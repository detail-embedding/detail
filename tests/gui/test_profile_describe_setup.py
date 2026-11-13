"""Configuration ▸ Cluster profile (``gui/profile_describe_setup.py``, §1).

The guards:

- **Every surface of the step is built**: the page, the dialog the profile window opens, and
  the SHAP section in each state a user meets (not installed / installed).
- **The install runs off the UI thread**, through ``run_progress`` with a Cancel — the button's
  work is observed running on a thread that is not the GUI's.
- **The button is there when it helps and says why when it is not**: Install SHAP greys once
  SHAP is in.
- **The checklist and the profile window reach the step**: the checklist's row navigates to it,
  and a fallen-back profile offers Install SHAP….

The probe (a fresh-subprocess check run) is stood in so each state is exactly the one named; the
install is stood in too, so nothing here installs anything.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton  # noqa: E402

from detail.api import extras  # noqa: E402
from detail.core.dataclasses import CheckResult, CheckStatus  # noqa: E402
from detail.gui import profile_describe_setup as setup  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402


@pytest.fixture
def probes(tmp_path, monkeypatch):
    """The SHAP probe stood in; flip ``state`` to choose what it answers."""
    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path)
    state = {"shap": False}

    def shap_state():
        ok = state["shap"]
        result = CheckResult(
            "shap",
            CheckStatus.OK if ok else CheckStatus.WARN,
            "shap 0.46.0" if ok else "",
        )
        extra = extras.ExtraStatus("analysis", ("shap",), () if ok else ("shap",))
        return (
            result,
            extra,
            "uv pip install --python python shap",
            ("0.46.0" if ok else None),
        )

    monkeypatch.setattr(setup, "_shap_state", shap_state)
    return state


def _page(qapp):
    page = setup.ProfileDescribePage()
    page.resize(900, 1000)
    page.show()
    assert drain_until(qapp, lambda: page.shap.recheck.isEnabled(), 10_000)
    return page


def test_the_page_and_its_dialog_build(qapp, probes):
    page = _page(qapp)
    assert page.shap is not None
    assert not hasattr(page, "describe")
    dialog = setup.ProfileDescribeSetupDialog()
    assert isinstance(dialog.page, setup.ProfileDescribePage)


def test_shap_missing_offers_the_install_and_the_command(qapp, probes):
    page = _page(qapp)
    assert "not installed" in page.shap.status.text()
    assert page.shap.install.isEnabled()
    assert page.shap._command is not None and page.shap.paste.isVisibleTo(page)
    assert "permutation importance" in page.shap.status.text()


def test_shap_installed_shows_its_version_and_greys_the_install(qapp, probes):
    probes["shap"] = True
    page = _page(qapp)
    assert page.shap.status.text().startswith("SHAP 0.46.0 is installed")
    assert not page.shap.install.isEnabled()
    assert page.shap.install.toolTip() == "SHAP is installed"
    assert page.shap._command is None and page.shap.paste.isHidden()


def test_install_shap_runs_off_the_ui_thread_then_checks_again(
    qapp, probes, monkeypatch
):
    seen = {}

    def install(extra, progress):
        seen["thread"] = threading.current_thread()
        seen["extra"] = extra
        progress("installing", 0.5, "Installing collected packages: shap")
        probes["shap"] = True
        return extras.ExtraStatus(extra, ("shap",), ())

    monkeypatch.setattr(extras, "install_extra", install)
    page = _page(qapp)
    page.shap.install.click()
    assert drain_until(qapp, lambda: "is installed" in page.shap.status.text(), 10_000)
    assert seen["extra"] == "analysis"
    assert seen["thread"] is not threading.main_thread()


def test_a_failed_install_shows_what_the_installer_said(qapp, probes, monkeypatch):
    def install(extra, progress):
        raise extras.InstallError("ERROR: No matching distribution found for shap")

    monkeypatch.setattr(extras, "install_extra", install)
    page = _page(qapp)
    page.shap.install.click()
    assert drain_until(qapp, lambda: page.shap.log.isVisibleTo(page), 10_000)
    assert "No matching distribution" in page.shap.log.toPlainText()


def test_the_checklist_row_opens_the_step(qapp):
    from detail.gui import pages
    from tests.gui.test_configuration_panel import _signals

    opened = []
    panel = pages._ReadinessPanel(opened.append)
    panel._render(_signals(gpu=True))
    buttons = {
        b.objectName(): b for b in panel.findChildren(QPushButton) if b.objectName()
    }
    buttons["profile_describe"].click()
    assert opened == ["profile_describe"]


def test_a_profile_that_fell_back_offers_install_shap(qapp):
    from types import SimpleNamespace

    from detail.analysis import explainers
    from detail.gui import cluster_profile

    def profile(explainer, note):
        return SimpleNamespace(
            explanation=SimpleNamespace(explainer=explainer, note=note)
        )

    fallback = explainers.FALLBACK_EXPLAINER
    assert cluster_profile.fell_back(profile(fallback, "shap is not installed"))
    assert not cluster_profile.fell_back(profile(fallback, ""))  # asked for by name
    assert not cluster_profile.fell_back(profile(explainers.DEFAULT_EXPLAINER, ""))
