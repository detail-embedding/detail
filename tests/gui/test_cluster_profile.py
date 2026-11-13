"""The Cluster profile window (``gui/cluster_profile.py``) and the three doors into it.

Guards:

- **The verdict gates the chart.** A profile whose measures do not separate the cluster opens
  with the chart folded behind *Show anyway*; one that does shows it.
- **A saved profile opens as saved** — settings and numbers — with no recompute.
- **Each door is offered only where it is wired** (the tree's menu when a host listens, the clip
  window's button when the page provides ``open_profile``), and INTERPRET's doors — **Cluster
  tools ▾** and the tree's right-click — open the same window on the cluster chosen.
"""

import pytest

pytest.importorskip("PyQt6")

import detail.api.cluster_profile as cluster_profile  # noqa: E402
from tests.api.test_cluster_profile import _node_with_members, _track  # noqa: E402


@pytest.fixture
def saved(fitted_profile, monkeypatch):
    experiment, run_id = fitted_profile
    run = experiment.temporal_clustering_run(run_id)
    run.rebuild_node_cache()
    view = run.node_view()
    node = _node_with_members(view)
    by_unit: dict = {}
    for m in view.sample_runs(node, n=100_000, seed=0):
        by_unit.setdefault((m["recording_id"], int(m["subject"])), []).append(m)
    monkeypatch.setattr(
        cluster_profile,
        "_trajectory",
        lambda _e, key, _r, _ts: _track(by_unit, key, fast=True),
    )
    return experiment, run, node


def _dialog(experiment, run, node):
    from detail.gui.cluster_profile import ClusterProfileDialog

    dialog = ClusterProfileDialog(experiment, run, node, title=f"Cluster #{node}")
    dialog.resize(960, 760)
    dialog.show()
    return dialog


def test_a_node_with_no_profile_says_so_and_offers_one(qapp, saved):
    experiment, run, node = saved
    dialog = _dialog(experiment, run, node)
    assert dialog._compute.text() == "Profile this cluster"
    assert dialog.config()["compare"] == "rest"


def test_a_separating_profile_shows_its_chart(qapp, saved):
    from PyQt6.QtWidgets import QCheckBox, QGroupBox

    experiment, run, node = saved
    profile = experiment.profile_cluster(
        str(run.id), node, {"explainer": "logistic_regression", "n_per_side": 40}
    )
    assert profile.explanation.separated
    dialog = _dialog(experiment, run, node)  # opens the saved one
    assert dialog._compute.text() == "Recompute"
    assert dialog.config()["explainer"] == "logistic_regression"
    titles = [b.title() for b in dialog.findChildren(QGroupBox)]
    assert "What sets it apart" in titles
    assert not [c for c in dialog.findChildren(QCheckBox) if "Show anyway" in c.text()]


def test_a_profile_that_does_not_separate_folds_its_chart(qapp, saved, monkeypatch):
    from PyQt6.QtWidgets import QCheckBox

    experiment, run, node = saved
    # No planted speed (the API test's premise): what is left is weak, and the demanding margin
    # makes the profile say it does not separate. With the planted speed a temporal fit's bouts
    # separate cleanly enough to clear even this margin.
    by_unit: dict = {}
    for m in run.node_view().sample_runs(node, n=100_000, seed=0):
        by_unit.setdefault((m["recording_id"], int(m["subject"])), []).append(m)
    monkeypatch.setattr(
        cluster_profile,
        "_trajectory",
        lambda _e, key, _r, _ts: _track(by_unit, key, fast=False),
    )
    experiment.profile_cluster(
        str(run.id),
        node,
        {"explainer": "logistic_regression", "n_per_side": 40, "margin": 0.5},
    )
    dialog = _dialog(experiment, run, node)
    (reveal,) = [c for c in dialog.findChildren(QCheckBox) if "Show anyway" in c.text()]
    chart = reveal.parent().layout().itemAt(1).widget()
    assert not chart.isVisible()
    reveal.setChecked(True)
    assert chart.isVisible()


def test_render(qapp, saved, tmp_path):
    """Laid out and drawn, so a broken chart or a squashed row shows in the image."""
    experiment, run, node = saved
    experiment.profile_cluster(
        str(run.id), node, {"explainer": "logistic_regression", "n_per_side": 40}
    )
    dialog = _dialog(experiment, run, node)
    for _ in range(5):
        qapp.processEvents()
    image = dialog.grab().toImage()
    assert image.width() > 0 and image.height() > 0
    dialog.grab().save(str(tmp_path / "cluster_profile.png"))


def test_the_tree_offers_a_profile_only_when_a_host_listens(qapp):
    from tests.gui.test_node_icicle import _ChainView

    from detail.gui.node_icicle import NodeIcicle

    icicle = NodeIcicle()
    icicle.set_view(_ChainView())
    texts = lambda: [a.text() for a in icicle.context_menu(6).actions()]  # noqa: E731
    assert "Cluster profile…" not in texts()
    icicle.profile_wanted.connect(lambda _node: None)
    assert "Cluster profile…" in texts()


def test_the_clip_window_offers_a_profile_only_when_the_page_provides_one(qapp, saved):
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.clip_cluster_info import ClipComparePanel

    experiment, _run, _node = saved

    class _Context:
        def experiment(self):
            return experiment

        def sibling(self):
            return None

        def member_clips(self, n):
            return []

        def title(self):
            return "cluster #1"

    opened = []

    class _WithProfile(_Context):
        def open_profile(self, parent):
            opened.append(parent)

    def buttons(context):
        panel = ClipComparePanel(context)
        return {b.objectName(): b for b in panel.findChildren(QPushButton)}

    assert "clip_profile_open" not in buttons(_Context())
    found = buttons(_WithProfile())
    found["clip_profile_open"].click()
    assert len(opened) == 1


def _interpret(qapp, fitted, monkeypatch):
    """INTERPRET on a real fit, loaded, with the profile window stubbed to record its opening."""
    from detail.gui import cluster_profile as gui_profile
    from detail.gui.cluster_workspace import ClusterWorkspace
    from tests.gui.conftest import drain_until

    opened = []

    class _Stub:
        def __init__(self, experiment, run, node, *, title, parent=None):
            opened.append((node, title))

        def exec(self):
            return 0

    monkeypatch.setattr(gui_profile, "ClusterProfileDialog", _Stub)
    experiment, run_id = fitted
    workspace = ClusterWorkspace(experiment, experiment.temporal_clustering_run(run_id))
    interpret = workspace._interpret
    assert drain_until(qapp, lambda: interpret._view is not None)
    return workspace, interpret, opened


def test_cluster_tools_opens_the_profile_of_the_open_cluster(qapp, fitted, monkeypatch):
    """INTERPRET's door: **Cluster tools ▾ → Cluster profile…**, on the cluster open in the tree
    — greyed with why on the root, which is the whole run."""
    _workspace, interpret, opened = _interpret(qapp, fitted, monkeypatch)
    view = interpret._view
    menu = interpret.cluster_tools_menu()
    (profile,) = [a for a in menu.actions() if a.text() == "Cluster profile…"]
    interpret.select(view.root)
    interpret.cluster_tools_menu()
    assert not profile.isEnabled() and "whole run" in profile.toolTip()
    node = int(view.children(view.root)[0])
    interpret.select(node)
    interpret.cluster_tools_menu()
    assert profile.isEnabled()
    profile.trigger()
    assert opened == [(node, f"Cluster #{node}")]


def test_the_trees_right_click_opens_the_profile(qapp, fitted, monkeypatch):
    _workspace, interpret, opened = _interpret(qapp, fitted, monkeypatch)
    view = interpret._view
    node = int(view.children(view.root)[0])
    menu = interpret._icicle.context_menu(node)
    (profile,) = [a for a in menu.actions() if a.text() == "Cluster profile…"]
    profile.trigger()
    assert [n for n, _t in opened] == [node]
    root_menu = interpret._icicle.context_menu(int(view.root))
    (whole,) = [a for a in root_menu.actions() if a.text() == "Cluster profile…"]
    assert not whole.isEnabled() and "whole run" in whole.toolTip()


def test_another_cluster_is_asked_for_only_when_comparing_against_one(qapp, saved):
    from detail.gui._pydantic_form import set_any_field_value
    from detail.configs.cluster_profile import ClusterProfileConfig

    experiment, run, node = saved
    dialog = _dialog(experiment, run, node)
    against, compare = dialog._fields["against"], dialog._fields["compare"]
    assert not against.isEnabled()
    assert dialog.config()["against"] is None
    assert dialog.config()["leave_out_place"] is True
    compare.setCurrentIndex(compare.findData("node"))
    assert against.isEnabled()
    set_any_field_value(ClusterProfileConfig, "against", against, 3)
    assert dialog.config()["against"] == 3
    ClusterProfileConfig.model_validate(dialog.config())
    compare.setCurrentIndex(compare.findData("rest"))
    assert not against.isEnabled() and dialog.config()["against"] is None
    ClusterProfileConfig.model_validate(dialog.config())


def test_a_comparison_with_another_cluster_says_what_place_alone_does(qapp, saved):
    """The window draws the place fit's line (``place_text``, the same line the CLI prints)
    for a comparison against another cluster, and no such line for one against the rest.
    """
    from PyQt6.QtWidgets import QLabel

    from tests.api.test_cluster_profile import _apart_from

    experiment, run, node = saved
    other = _apart_from(run.node_view(), node)
    settings = {"explainer": "logistic_regression", "n_per_side": 40}
    pair = experiment.profile_cluster(
        str(run.id), node, {**settings, "compare": "node", "against": other}
    )
    dialog = _dialog(experiment, run, node)
    dialog.show_profile(pair)
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    assert cluster_profile.place_text(pair) in texts
    dialog.show_profile(experiment.profile_cluster(str(run.id), node, settings))
    texts = [label.text() for label in dialog.findChildren(QLabel) if label.isVisible()]
    assert not [t for t in texts if t.startswith("Position and facing alone")]
