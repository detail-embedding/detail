"""INTERPRET's **Cluster tools ▾** (§22, user decision 2026-11-10).

rd's per-cluster tools — Cluster profile…, Look-alike clusters…, Find look-alikes…, Profile
clusters… — sat as a row of buttons over rd's tree. On main's INTERPRET they are one
menu button on the label bar's field row, and every entry that cannot run is greyed with why
(``requirements``: a disabled control says why). What is pinned here, through the page a user
reaches (``ClusterWorkspace`` → INTERPRET):

- the button opens the menu, and the menu holds the four entries, in that order;
- on a snapshot the job entries are greyed with the snapshot's reason, and the profile is not;
- *Find look-alikes…* opens the launch dialog on this run (the job, as on rd).

The windows behind the entries have their own tests: ``test_cluster_profile.py``,
``test_lookalikes_gui.py``, ``test_profile_describe_gui.py``.
"""

from __future__ import annotations

import pytest

from detail.gui.cluster_workspace import ClusterWorkspace
from tests.gui.conftest import drain_until

ENTRIES = [
    "Cluster profile…",
    "Look-alike clusters…",
    "Find look-alikes…",
    "Profile clusters…",
]


def _interpret(qapp, experiment, handle):
    workspace = ClusterWorkspace(experiment, handle)
    interpret = workspace._interpret
    assert drain_until(qapp, lambda: interpret._view is not None), "never loaded"
    return workspace, interpret


def _entry(interpret, text: str):
    (action,) = [
        a for a in interpret.cluster_tools_menu().actions() if a.text() == text
    ]
    return action


def test_the_button_opens_one_menu_of_the_four_tools(qapp, fitted):
    experiment, run_id = fitted
    _workspace, interpret = _interpret(
        qapp, experiment, experiment.temporal_clustering_run(run_id)
    )
    button = interpret._tools
    assert button.text() == "Cluster tools ▾" and button.isEnabled()
    assert interpret.label_bar().isAncestorOf(button), "on the label bar"
    button.click()
    menu = interpret._tools_menu
    try:
        assert menu.isVisible(), "the click opens the menu"
        assert [a.text() for a in menu.actions()] == ENTRIES
    finally:
        menu.hide()
    view = interpret._view
    interpret.select(int(view.children(view.root)[0]))
    assert all(_entry(interpret, text).isEnabled() for text in ENTRIES)


def test_a_snapshot_greys_the_jobs_with_the_reason(qapp, fitted):
    experiment, run_id = fitted
    handle = experiment.temporal_clustering_run(run_id)
    view = handle.node_view()
    snapshot = handle.view(handle.pin(reason="finish"))
    _workspace, interpret = _interpret(qapp, experiment, snapshot)
    interpret.select(int(view.children(view.root)[0]))
    for text in ENTRIES[1:]:
        action = _entry(interpret, text)
        assert not action.isEnabled() and "snapshot" in action.toolTip(), text
    assert _entry(interpret, "Cluster profile…").isEnabled(), "reading is not changing"


def test_find_lookalikes_opens_the_launch_dialog_on_this_run(qapp, fitted, monkeypatch):
    from detail.gui.launch import LaunchAnalysisDialog

    experiment, run_id = fitted
    _workspace, interpret = _interpret(
        qapp, experiment, experiment.temporal_clustering_run(run_id)
    )
    opened: list = []

    def fake_exec(self):
        opened.append(self.config_widget())
        return 0

    monkeypatch.setattr(LaunchAnalysisDialog, "exec", fake_exec)
    _entry(interpret, "Find look-alikes…").trigger()
    (form,) = opened
    assert drain_until(qapp, lambda: form.config().get("source_run_id") == run_id)
