"""Every remote-capable analysis module's launch form offers *Run on* (§6 "Remote analysis").

**The class: a capability one surface of N can reach.** ``remote_capable`` is declared on the
module, and the API and CLI take ``compute`` from it, but the launch form is written by hand per
module: a module made remote-capable without a *Run on* row is remote everywhere but the GUI,
and nothing fails. So this is driven by the backend's own declaration, not a list here: a module
that declares itself remote-capable is checked the day it does, and its form must offer each
registered target and send the one shown as ``compute``.

The row is one helper (``gui/modules/_common.py::add_run_on_row``), shared by every such form
since there were four of them.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QComboBox  # noqa: E402

from detail.api._runner_modules import PIPELINE_RUNNERS  # noqa: E402
from detail.configs.remote import RemoteTargetConfig  # noqa: E402
from detail.executor.runners import (  # noqa: E402
    ANALYSIS_MODULES,
    import_runner_modules,
)
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, surfaced  # noqa: E402
from detail.storage import remote_targets  # noqa: E402

import_runner_modules(PIPELINE_RUNNERS)

REMOTE_CAPABLE = [
    module_id
    for module_id in surfaced(ANALYSIS_GUI_MODULES)
    if getattr(ANALYSIS_MODULES.get(module_id), "remote_capable", False)
]


@pytest.fixture
def tillicum(tmp_path, monkeypatch):
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )
    remote_targets.register_remote_target(
        RemoteTargetConfig(name="tillicum", host="tillicum", remote_workdir="/x")
    )


def _run_on(widget) -> QComboBox:
    combos = [
        combo
        for combo in widget.findChildren(QComboBox)
        if combo.findData("tillicum") >= 0 and combo.itemData(0) is None
    ]
    assert len(combos) == 1, "no Run on row offering the registered target"
    return combos[0]


def test_the_clustering_modules_are_among_them():
    assert {
        "temporal_clustering",
        "profile_describe",
        "cluster_lookalikes",
    } <= set(REMOTE_CAPABLE)


@pytest.mark.parametrize("module_id", REMOTE_CAPABLE)
def test_the_form_sends_the_target_it_shows(qapp, tillicum, module_id):
    widget = ANALYSIS_GUI_MODULES[module_id].config_widget()
    run_on = _run_on(widget)
    assert run_on.currentData() is None, "this computer is the default"
    assert "compute" not in widget.config()

    run_on.setCurrentIndex(run_on.findData("tillicum"))

    assert widget.config()["compute"]["backend"] == "remote"
    assert widget.config()["compute"]["remote_target"] == "tillicum"


@pytest.mark.parametrize("module_id", REMOTE_CAPABLE)
def test_loading_settings_shows_where_they_ran(qapp, tillicum, module_id):
    widget = ANALYSIS_GUI_MODULES[module_id].config_widget()
    run_on = _run_on(widget)

    widget.load_config({"compute": {"backend": "remote", "remote_target": "tillicum"}})
    assert run_on.currentData() == "tillicum"
    widget.load_config({"compute": None})
    assert run_on.currentData() is None
