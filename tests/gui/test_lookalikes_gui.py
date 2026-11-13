"""INTERPRET's Look-alike clusters window (§22 "Look-alike clusters").

What a person sees and does, on a real fit (``fitted``) with comparisons saved beside it the way
a cluster profile against another cluster saves them:

- the window lists each group with its members, its proposed name filled in, and why the
  members are grouped — every link with both of its scores; a member whose label would also cover
  clusters named for something else names them and starts unticked, the rest start ticked;
- clicking a member selects it in the tree, so its card and clips show;
- *Label as one…* shows what it would do before anything is written, and writes one label for
  the ticked members through the page's own label queue — an unticked member is left alone;
- a run whose labels cannot change says why, rather than offering a button that does nothing;
- the job's launch form runs it here or on a registered remote target, and a run started from
  a previous one keeps its target;
- Run with every pair asks first — how many pairs, about how long, and what the labelled mode
  would compare, counted off the UI thread — and nothing launches on Cancel; labelled asks
  nothing; a saved ``pairs: names`` opens on the default.

The read is off the UI thread (``BackgroundCall``); the write is the page's label queue.
"""

import dataclasses
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from detail.gui import lookalikes as lookalikes_gui
from detail.gui.cluster_workspace import ClusterWorkspace
from detail.storage import analysis_runs
from tests.api.test_cluster_lookalikes import save_comparison
from tests.gui.subjects import a_lookalike_group


def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """A modal box nobody stubbed is a failure, not a hung run."""
    raised: list[str] = []

    def critical(parent, title, text, *args, **kwargs):
        raised.append(f"{title}: {text}")
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(critical))
    yield
    assert not raised, f"a background call failed: {raised}"


@pytest.fixture
def grouped(qapp, fitted):
    """Three clusters from both halves of the tree, chained by two look-alike pairs, and a
    fourth pair the body measures told apart."""
    experiment, run_id = fitted
    handle = experiment.temporal_clustering_run(run_id)
    view = handle.node_view()
    run_dir = analysis_runs.analysis_dir(experiment.dir, run_id)
    left, right = list(view.children(view.root))[:2]
    a1, a2 = list(view.children(left))[:2]
    b1 = list(view.children(right))[0]
    save_comparison(run_dir, a1, b1, 0.52)
    save_comparison(run_dir, a2, b1, 0.55)
    save_comparison(run_dir, a1, right, 0.95)
    workspace = ClusterWorkspace(experiment, handle)
    interpret = workspace._interpret
    assert _drain(qapp, lambda: interpret._view is not None), "Interpret never loaded"
    return handle, workspace, interpret, (a1, a2, b1)


def tool(interpret, starts: str):
    """The **Cluster tools ▾** entry whose text starts with ``starts``, as the menu opens."""
    menu = interpret.cluster_tools_menu()
    (action,) = [a for a in menu.actions() if a.text().startswith(starts)]
    return action


def _open_window(qapp, interpret):
    action = tool(interpret, "Look-alike")
    assert action.isEnabled(), action.toolTip()
    action.trigger()
    window = interpret._lookalikes_window
    assert window is not None
    assert _drain(qapp, lambda: window._groups), "the groups were never read"
    return window


def test_the_window_lists_the_group_its_links_and_a_proposed_name(qapp, grouped):
    handle, _workspace, interpret, nodes = grouped
    window = _open_window(qapp, interpret)
    assert len(window._groups) == 1
    top = window._tree.topLevelItem(0)
    members = {int(top.child(i).text(0)[1:]) for i in range(top.childCount())}
    assert members == set(nodes)
    group = window._groups[0]
    assert sorted(window.chosen_nodes()) == sorted(group.ticked)
    for i in range(top.childCount()):
        node = int(top.child(i).text(0)[1:])
        said = top.child(i).text(lookalikes_gui._SWEEP_COLUMN)
        assert bool(said) == (node in group.sweeps), node
    why = window._why.text()
    assert ("would also label" in why) == bool(group.sweeps)
    assert why.count("body measures") == 2, "both links, not the pair told apart"
    assert "position and facing alone 90%" in why
    assert "1 group, held together by 2 look-alike pairs" in window._status.text()


def test_clicking_a_member_selects_it_in_the_tree(qapp, grouped):
    _handle, _workspace, interpret, nodes = grouped
    window = _open_window(qapp, interpret)
    member = window._tree.topLevelItem(0).child(1)
    window._tree.itemClicked.emit(member, 0)
    assert interpret._selected == int(member.text(0)[1:])


def test_label_as_one_writes_one_label_for_the_ticked_members(
    qapp, grouped, monkeypatch
):
    handle, _workspace, interpret, nodes = grouped
    window = _open_window(qapp, interpret)
    top = window._tree.topLevelItem(0)
    for i in range(top.childCount()):
        top.child(i).setCheckState(0, Qt.CheckState.Checked)
    left_out = top.child(top.childCount() - 1)
    left_out.setCheckState(0, Qt.CheckState.Unchecked)
    window._name.setText("rearing")
    shown = []

    def exec_(dialog):
        shown.append(dialog)
        return True

    monkeypatch.setattr(lookalikes_gui.LabelGroupDialog, "exec", exec_)
    assert window._label.isEnabled()
    window._label.click()
    assert shown, "what it would do is shown before anything is written"
    assert _drain(qapp, lambda: interpret._writer is None and not interpret._pending)
    names = handle.state().node_names
    kept = int(left_out.text(0)[1:])
    labelled = [n for n in nodes if n != kept]
    assert {names.get(str(n)) for n in labelled} == {"rearing"}
    assert str(kept) not in names, "an unticked member is left alone"
    source = handle.state().node_label_sources[str(labelled[0])]
    assert source["from"] == "lookalikes"


def test_a_run_that_cannot_change_says_why(qapp):
    class Run:
        id = "an_01"

        def lookalike_groups(self):
            return [a_lookalike_group()]

    window = lookalikes_gui.LookalikesWindow(
        Run(), editable=False, why_not="This run is finished. Reopen it to change it."
    )
    assert _drain(qapp, lambda: window._groups)
    assert not window._label.isEnabled()
    assert "finished" in window._label.toolTip()
    assert window._name.text() == "Rear_to_right", "the group's proposed name"
    assert "#1067 would also label #739 (3.9% of the run)" in window._why.text()
    unticked = [
        window._tree.topLevelItem(0).child(i)
        for i in range(window._tree.topLevelItem(0).childCount())
        if window._tree.topLevelItem(0).child(i).checkState(0)
        == Qt.CheckState.Unchecked
    ]
    assert [item.text(0) for item in unticked] == ["#1067"]


def test_label_as_one_says_what_a_ticked_member_would_sweep_in(qapp):
    from PyQt6.QtWidgets import QLabel

    from detail.api.cluster_lookalikes import plan_join

    group = a_lookalike_group()
    parent = [-1] * 1400
    plan = plan_join({}, parent, [1067, 755], "rearing up")
    dialog = lookalikes_gui.LabelGroupDialog(plan, "rearing up", sweeps=group.sweeps)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "#1067 — also labels #739 (3.9% of the run)" in text
    assert "Label them first" in text
    clean = plan_join({}, parent, [755], "rearing up")
    dialog = lookalikes_gui.LabelGroupDialog(clean, "rearing up", sweeps=group.sweeps)
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "also labels" not in text and "Label them first" not in text


def test_the_jobs_result_lists_each_pair(qapp, tmp_path):
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES

    record = {
        "module_id": "cluster_lookalikes",
        "source_run_id": "an_17",
        "counts": {"complete": 2, "failed": 1},
        "looks_alike": 1,
        "groups": [{"nodes": [755, 1067], "name": "rearing up", "links": 1}],
        "pairs": [
            {"a": 755, "b": 1067, "status": "complete", "accuracy": 0.52,
             "place_accuracy": 0.91, "looks_alike": True},
            {"a": 755, "b": 1179, "status": "complete", "accuracy": 0.95,
             "place_accuracy": 0.6, "looks_alike": False},
            {"a": 803, "b": 1179, "status": "failed", "error": "no track"},
        ],
    }  # fmt: skip
    widget = ANALYSIS_GUI_MODULES["cluster_lookalikes"].result_widget(None, record)
    from PyQt6.QtWidgets import QLabel, QTableView, QTableWidget

    text = " ".join(label.text() for label in widget.findChildren(QLabel))
    assert "2 complete, 1 failed" in text and "“rearing up”" in text
    assert widget.findChildren(QTableWidget) or widget.findChildren(QTableView)


def test_the_job_runs_on_a_chosen_remote_target(qapp, monkeypatch):
    """*Run on* sends the job to a registered target (where a run's pairs can take hours), and a
    run started from a previous one comes back on the same target."""
    from detail.api import remote as api_remote
    from detail.configs.remote import RemoteTargetConfig
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES

    monkeypatch.setattr(
        api_remote,
        "list_remote_targets",
        lambda: [RemoteTargetConfig(name="tillicum", host="t", remote_workdir="/x")],
    )
    widget = ANALYSIS_GUI_MODULES["cluster_lookalikes"].config_widget()
    assert "compute" not in widget.config(), "this computer by default"
    widget._where.setCurrentIndex(widget._where.findData("tillicum"))
    compute = widget.config()["compute"]
    assert (compute["backend"], compute["remote_target"]) == ("remote", "tillicum")

    seeded = ANALYSIS_GUI_MODULES["cluster_lookalikes"].config_widget()
    seeded.load_config({"compute": compute})
    assert seeded.config()["compute"]["remote_target"] == "tillicum"
    seeded.load_config({"compute": None})
    assert "compute" not in seeded.config()


# --- the warning before every pair ------------------------------------------------------- #
def _counts(n_all=1225, n_labelled=96, tops=4):
    from detail.api.cluster_lookalikes import PairCounts

    return PairCounts(
        n_all=n_all,
        n_labelled=n_labelled,
        n_labelled_tops=tops,
        skipped_all=0,
        skipped_labelled=0,
        per_pair_s=32.0,
        basis="about 32s (the time budget)",
    )


@pytest.fixture
def launch(qapp, fitted, monkeypatch):
    """The launch dialog on *Find look-alike clusters* with the fit chosen, as *Find
    look-alikes…* opens it; the count is stood in (and records the thread it ran on), the
    question records what it asked, and a launch records its config."""
    import threading

    from detail.gui import launch as launch_mod
    from detail.gui.launch import LaunchAnalysisDialog

    experiment, run_id = fitted
    seen: dict = {"asked": [], "launched": [], "threads": [], "answer": False}

    def count(run, config=None):
        seen["threads"].append(threading.current_thread())
        return _counts()

    def ask(parent, title, question):
        seen["asked"].append(question)
        return seen["answer"]

    def run_analysis(module_id, config):
        seen["launched"].append(config)
        return SimpleNamespace(id="an_99")

    monkeypatch.setattr(experiment, "count_lookalike_pairs", count)
    monkeypatch.setattr(experiment, "run_analysis", run_analysis)
    monkeypatch.setattr(launch_mod, "ask_before_launch", ask)
    dialog = LaunchAnalysisDialog(
        experiment,
        seed={"module_id": "cluster_lookalikes", "params": {}},
        selection={"source_run_id": str(run_id)},
    )
    assert _drain(qapp, lambda: dialog._config._run.currentData() == str(run_id))
    return dialog, seen


def _set_pairs(dialog, mode: str) -> None:
    combo = dialog._config._fields["pairs"]
    combo.setCurrentIndex(combo.findData(mode))


def test_every_pair_asks_first_off_the_ui_thread_and_cancel_launches_nothing(
    qapp, launch
):
    import threading

    dialog, seen = launch
    assert dialog._config.config()["pairs"] == "all", "every pair by default"
    dialog._run()
    assert _drain(qapp, lambda: seen["asked"]), "Run with every pair never asked"
    (question,) = seen["asked"]
    assert "compute intensive" in question.text and "1,225 pairs" in question.text
    assert "'labelled'" in question.text and "96 pairs" in question.text
    assert question.go == "Run every pair"
    assert seen["threads"] and threading.main_thread() not in seen["threads"]
    qapp.processEvents()
    assert not seen["launched"], "Cancel still launched"


def test_run_every_pair_launches_once_answered(qapp, launch):
    dialog, seen = launch
    seen["answer"] = True
    dialog._run()
    assert _drain(
        qapp, lambda: seen["launched"]
    ), "answering Run every pair launched nothing"
    assert seen["launched"][0]["pairs"] == "all"


def test_labelled_launches_without_a_question(qapp, launch):
    dialog, seen = launch
    _set_pairs(dialog, "labelled")
    dialog._run()
    assert _drain(qapp, lambda: seen["launched"]), "labelled never launched"
    assert not seen["asked"] and not seen["threads"], "labelled asked or counted"
    assert seen["launched"][0]["pairs"] == "labelled"


def test_the_question_says_when_nothing_is_labelled(qapp):
    from detail.gui.modules.analysis_cluster_lookalikes import every_pair_question

    experiment = SimpleNamespace(
        count_lookalike_pairs=lambda run, config=None: _counts(n_labelled=0, tops=0)
    )
    question = every_pair_question(experiment, "an_01", {"pairs": "all"})
    assert "no cluster is labelled yet" in question.text
    nothing = SimpleNamespace(
        count_lookalike_pairs=lambda run, config=None: _counts(n_all=0, n_labelled=0)
    )
    assert (
        every_pair_question(nothing, "an_01", {"pairs": "all"}) is None
    ), "no pair at all is the job's refusal to explain, not a question"
    done = SimpleNamespace(
        count_lookalike_pairs=lambda run, config=None: dataclasses.replace(
            _counts(), skipped_all=1225, skipped_labelled=96
        )
    )
    assert (
        every_pair_question(done, "an_01", {"pairs": "all"}) is None
    ), "every pair already compared costs nothing to ask about"


def test_the_question_box_defaults_to_cancel(qapp, monkeypatch):
    """The real box: the go button carries the verb, and Cancel is the default."""
    from PyQt6.QtWidgets import QWidget

    from detail.gui import space_prompt

    built = {}

    def exec_(box):
        built["default"] = box.defaultButton().text().replace("&", "")
        built["buttons"] = {b.text() for b in box.buttons()}
        built["text"] = box.text()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    question = space_prompt.LaunchQuestion("1,225 pairs, about 11h.", "Run every pair")
    assert (
        space_prompt.ask_before_launch(QWidget(), "Launch analysis", question) is False
    )
    assert "Run every pair" in built["buttons"]
    assert built["default"] == "Cancel"
    assert built["text"] == "1,225 pairs, about 11h."


def test_a_saved_names_mode_opens_on_every_pair(qapp):
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES

    widget = ANALYSIS_GUI_MODULES["cluster_lookalikes"].config_widget()
    widget.load_config({"pairs": "labelled"})
    assert widget.config()["pairs"] == "labelled"
    fresh = ANALYSIS_GUI_MODULES["cluster_lookalikes"].config_widget()
    fresh.load_config({"pairs": "names", "min_name_overlap": 0.33})
    assert fresh.config()["pairs"] == "all"
