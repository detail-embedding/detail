"""RemoteTargetsPage + _RemoteTargetDialog (§6) — the Configuration-mode surface for
managing Slurm cluster profiles."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from detail.api import checks as api_checks  # noqa: E402
from detail.api import remote as api_remote  # noqa: E402
from detail.configs.remote import RemoteTargetConfig  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.storage import remote_targets  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_targets(tmp_path, monkeypatch):
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )


def _register(name="tillicum", **overrides):
    fields = dict(name=name, host=name, remote_workdir="/scratch/x")
    fields.update(overrides)
    api_remote.add_remote_target(RemoteTargetConfig(**fields))


# --- _RemoteTargetDialog ------------------------------------------------------- #
def test_dialog_add_mode_values_round_trip(qapp):
    dialog = pages._RemoteTargetDialog()
    dialog._name.setText("tillicum")
    dialog._host.setText("tillicum")
    dialog._workdir.setText("/scratch/x")
    dialog._partition.setText("gpu-a40")
    dialog._modules.setText("cuda/12.4, gcc/13.2")
    dialog._gpus_per_job.setValue(2)
    config = dialog.values()
    assert config.name == "tillicum"
    assert config.partition == "gpu-a40"
    assert config.modules == ["cuda/12.4", "gcc/13.2"]
    assert config.gpus_per_job == 2
    assert config.account is None  # left blank -> None, not ""
    assert config.cost_per_gpu_hour is None  # left at 0 ("not set") -> None, not 0.0


def test_dialog_cost_per_gpu_hour_round_trips(qapp):
    dialog = pages._RemoteTargetDialog()
    dialog._name.setText("tillicum")
    dialog._host.setText("tillicum")
    dialog._workdir.setText("/scratch/x")
    dialog._cost_per_gpu_hour.setValue(0.5)
    assert dialog.values().cost_per_gpu_hour == 0.5


def test_dialog_cost_per_gpu_hour_is_framed_and_resizable(qapp):
    """Every QLineEdit row in this form grows when the dialog is resized wider; a spin
    box does not by default (Fixed horizontal policy), which would leave this the one
    row that stays put. It should match its neighbors and show a visible frame."""
    from PyQt6.QtWidgets import QSizePolicy

    dialog = pages._RemoteTargetDialog()
    field = dialog._cost_per_gpu_hour
    assert field.hasFrame()
    assert field.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert field.minimumWidth() >= 2 * field.sizeHint().width()


def test_dialog_edit_mode_prefills_cost_per_gpu_hour(qapp):
    existing = RemoteTargetConfig(
        name="tillicum",
        host="tillicum",
        remote_workdir="/scratch/x",
        cost_per_gpu_hour=0.75,
    )
    dialog = pages._RemoteTargetDialog(existing=existing)
    assert dialog._cost_per_gpu_hour.value() == 0.75


def test_dialog_offers_the_model_defaults_for_a_new_target(qapp):
    """The defaults are the config model's, not numbers retyped in the form: the form used to
    say `04:00:00` in two places of its own, which is how a changed default reaches the CLI and
    the Python API and not the dialog."""
    dialog = pages._RemoteTargetDialog()
    dialog._name.setText("tillicum")
    dialog._host.setText("tillicum")
    dialog._workdir.setText("/scratch/x")

    config = dialog.values()

    defaults = RemoteTargetConfig.model_fields
    assert dialog._walltime.text() == defaults["walltime"].default == "24:00:00"
    assert (
        dialog._max_concurrent_jobs.value() == defaults["max_concurrent_jobs"].default
    )
    assert (config.walltime, config.max_concurrent_jobs) == ("24:00:00", 8)


def test_dialog_a_blank_walltime_falls_back_to_the_models_default(qapp):
    dialog = pages._RemoteTargetDialog()
    dialog._name.setText("tillicum")
    dialog._host.setText("tillicum")
    dialog._workdir.setText("/scratch/x")
    dialog._walltime.setText("  ")

    assert dialog.values().walltime == "24:00:00"


def test_dialog_round_trips_concurrent_jobs_and_prefills_it_on_edit(qapp):
    dialog = pages._RemoteTargetDialog()
    dialog._name.setText("tillicum")
    dialog._host.setText("tillicum")
    dialog._workdir.setText("/scratch/x")
    dialog._max_concurrent_jobs.setValue(5)
    config = dialog.values()
    assert config.max_concurrent_jobs == 5

    edit = pages._RemoteTargetDialog(existing=config)

    assert edit._max_concurrent_jobs.value() == 5


def test_dialog_edit_keeps_a_saved_targets_four_hour_walltime(qapp):
    """A target registered before the default changed stores `04:00:00`; opening Edit must show
    that, not quietly offer the new default, so the person sees what the job will be killed at.
    """
    existing = RemoteTargetConfig(
        name="tillicum",
        host="tillicum",
        remote_workdir="/scratch/x",
        walltime="04:00:00",
    )

    dialog = pages._RemoteTargetDialog(existing=existing)

    assert dialog._walltime.text() == "04:00:00"


def test_dialog_tooltips_carry_the_models_descriptions(qapp):
    dialog = pages._RemoteTargetDialog()
    fields = RemoteTargetConfig.model_fields

    assert dialog._walltime.toolTip() == fields["walltime"].description
    assert (
        dialog._max_concurrent_jobs.toolTip()
        == fields["max_concurrent_jobs"].description
    )


def test_dialog_edit_mode_prefills_and_disables_name(qapp):
    existing = RemoteTargetConfig(
        name="tillicum",
        host="tillicum",
        remote_workdir="/scratch/x",
        partition="gpu-a40",
    )
    dialog = pages._RemoteTargetDialog(existing=existing)
    assert dialog._name.text() == "tillicum"
    assert not dialog._name.isEnabled()
    assert dialog._partition.text() == "gpu-a40"
    assert dialog.windowTitle() == "Edit target"


def test_dialog_values_raises_on_invalid_name(qapp):
    dialog = pages._RemoteTargetDialog()
    dialog._name.setText("Bad Name")
    dialog._host.setText("h")
    dialog._workdir.setText("/x")
    with pytest.raises(ValueError):
        dialog.values()


# --- RemoteTargetsPage ---------------------------------------------------------- #
def test_page_builds_empty_with_actions_disabled(qapp, monkeypatch):
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: True)
    page = pages.RemoteTargetsPage()
    assert page._selected() is None
    assert not page._edit_btn.isEnabled()
    assert not page._remove_btn.isEnabled()
    assert not page._test_btn.isEnabled()
    assert not page._push_creds_btn.isEnabled()
    assert not page._ssh_session_btn.isEnabled()


def test_page_lists_registered_targets(qapp):
    _register("tillicum")
    _register("hyak", host="klone.hyak.uw.edu")
    page = pages.RemoteTargetsPage()
    assert {t.name for t in page._targets} == {"tillicum", "hyak"}


def test_page_table_shows_concurrent_jobs_and_walltime(qapp):
    """The two settings that decide how a run behaves on the cluster are readable without
    opening Edit: a target still on `04:00:00` is the thing worth noticing."""
    _register("tillicum", max_concurrent_jobs=5, walltime="1-00:00:00")
    _register("hyak", host="klone.hyak.uw.edu", walltime="04:00:00")
    page = pages.RemoteTargetsPage()
    model = page._table.model()
    rows = {
        model.item(r, 0).text(): (
            model.item(r, page._COLUMNS.index("Concurrent jobs")).text(),
            model.item(r, page._COLUMNS.index("Walltime")).text(),
        )
        for r in range(model.rowCount())
    }

    assert rows == {"tillicum": ("5", "1-00:00:00"), "hyak": ("8", "04:00:00")}


def test_page_table_shows_cost_per_gpu_hour_when_set(qapp):
    _register("tillicum", cost_per_gpu_hour=0.5)
    _register("hyak", host="klone.hyak.uw.edu")  # no rate configured
    page = pages.RemoteTargetsPage()
    model = page._table.model()
    col = page._COLUMNS.index("Cost/GPU-hr")
    cell_texts = [model.item(row, col).text() for row in range(model.rowCount())]
    assert "$0.50" in cell_texts
    assert "—" in cell_texts


def test_selecting_a_row_enables_row_actions(qapp, monkeypatch):
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: True)
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    assert page._selected().name == "tillicum"
    assert page._edit_btn.isEnabled()
    assert page._remove_btn.isEnabled()
    assert page._test_btn.isEnabled()
    assert page._push_creds_btn.isEnabled()
    assert page._ssh_session_btn.isEnabled()


def test_ssh_session_button_greyed_when_unsupported(qapp, monkeypatch):
    """Plain Linux outside WSL has no reliable terminal-launch command — grey out,
    don't hide, even with a target selected."""
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: False)
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    assert not page._ssh_session_btn.isEnabled()
    # every other row action is unaffected by terminal-launch support
    assert page._edit_btn.isEnabled()
    assert page._test_btn.isEnabled()


def test_open_ssh_session_launches_the_terminal(qapp, monkeypatch):
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: True)
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(api_checks, "ssh_refresh_argv", lambda host: ["echo", host])
    captured = {}
    monkeypatch.setattr(
        pages.subprocess, "Popen", lambda argv: captured.setdefault("argv", argv)
    )
    page._open_ssh_session()
    assert captured["argv"] == ["echo", "tillicum"]


def test_open_ssh_session_reports_a_launch_failure(qapp, monkeypatch):
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: True)
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(api_checks, "ssh_refresh_argv", lambda host: ["nope"])

    def _raise(argv):
        raise OSError("no such terminal")

    monkeypatch.setattr(pages.subprocess, "Popen", _raise)
    monkeypatch.setattr(pages.QMessageBox, "critical", lambda *a, **k: None)
    page._open_ssh_session()  # must not raise


def test_remove_target_confirms_then_removes(qapp, monkeypatch):
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    page._remove_target()
    assert api_remote.list_remote_targets() == []
    assert page._selected() is None  # table rebuilt, nothing selected


def test_remove_target_cancelled_confirm_keeps_it(qapp, monkeypatch):
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    page._remove_target()
    assert [t.name for t in api_remote.list_remote_targets()] == ["tillicum"]


def test_add_target_writes_through_the_dialog(qapp, monkeypatch):
    page = pages.RemoteTargetsPage()

    class _FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            from PyQt6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted

        def values(self):
            return RemoteTargetConfig(
                name="tillicum", host="tillicum", remote_workdir="/scratch/x"
            )

    monkeypatch.setattr(pages, "_RemoteTargetDialog", _FakeDialog)
    page._add_target()
    assert [t.name for t in api_remote.list_remote_targets()] == ["tillicum"]


def test_test_target_reports_status(qapp, monkeypatch):
    from detail.api.remote import RemoteTargetStatus

    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(
        api_remote,
        "test_remote_target",
        lambda name: RemoteTargetStatus(ok=True, detail="connected to 'tillicum'"),
    )
    page._test_target()
    assert _drain(qapp, lambda: "connected" in page._status.text())


def _status_with_build(state, **fields):
    from detail.api.remote import RemoteTargetStatus
    from detail.core import buildinfo

    build = buildinfo.BuildComparison(
        state,
        target="tillicum",
        host="tillicum",
        local_commit="c2a1dd8",
        remote_commit="a1b2c3d",
        remote_root="/gpfs/home/x/detail",
        **fields,
    )
    return RemoteTargetStatus(ok=True, detail="connected to 'tillicum'", build=build)


def _color_of(label):
    from PyQt6.QtGui import QPalette

    return label.palette().color(QPalette.ColorRole.WindowText)


def test_test_target_shows_the_update_command_when_the_cluster_is_behind(
    qapp, monkeypatch
):
    from detail.core import buildinfo
    from detail.gui.status_color import WARN

    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(
        api_remote,
        "test_remote_target",
        lambda name: _status_with_build(buildinfo.BEHIND, commits=4),
    )
    page._test_target()
    assert _drain(qapp, lambda: "behind" in page._status.text())

    text = page._status.text()
    assert "connected to 'tillicum'" in text  # the connection line is still there
    assert "ssh tillicum 'git -C /gpfs/home/x/detail pull --ff-only'" in text
    assert _color_of(page._status) == WARN  # reachable, but not a clean pass


def test_test_target_is_green_only_when_the_cluster_matches(qapp, monkeypatch):
    from detail.core import buildinfo
    from detail.gui.status_color import READY

    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(
        api_remote,
        "test_remote_target",
        lambda name: _status_with_build(buildinfo.SAME),
    )
    page._test_target()
    assert _drain(qapp, lambda: "same build" in page._status.text())

    assert _color_of(page._status) == READY


def test_push_credentials_reports_status(qapp, monkeypatch):
    from detail.api.remote import RemoteTargetStatus

    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    monkeypatch.setattr(
        api_remote,
        "push_credentials",
        lambda name: RemoteTargetStatus(
            ok=True, detail="pushed to 'tillicum': Hugging Face access"
        ),
    )
    page._push_credentials()
    assert _drain(qapp, lambda: "pushed" in page._status.text())


def test_push_credentials_reports_no_local_credentials(qapp, monkeypatch):
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)

    def _raise(name):
        raise ValueError("no credentials are configured on this machine yet")

    monkeypatch.setattr(api_remote, "push_credentials", _raise)
    page._push_credentials()
    assert _drain(qapp, lambda: "no credentials are configured" in page._status.text())


def _drain(qapp, predicate, timeout_ms=5000):
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


# --- Update DETAIL… -------------------------------------------------------------- #
def _build(state, **fields):
    from detail.core import buildinfo

    return buildinfo.BuildComparison(
        state,
        target="tillicum",
        host="tillicum",
        local_commit="c2a1dd8",
        remote_commit="a1b2c3d",
        remote_root="/gpfs/home/x/detail",
        **fields,
    )


def _update_page(qapp, monkeypatch, state, *, answer, **fields):
    """A page with one selected target whose comparison is ``state``; ``answer`` is what the
    confirm dialog returns. Every call that would reach the cluster is recorded."""
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: True)
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    calls = {"asked": [], "updated": []}
    build = _build(state, **fields)
    monkeypatch.setattr(api_remote, "check_remote_build", lambda name: build)

    def ask(*args, **kwargs):
        calls["asked"].append(args[2])
        return answer

    monkeypatch.setattr(QMessageBox, "question", ask)
    return page, calls, build


def test_update_button_needs_a_target_and_says_why(qapp, monkeypatch):
    monkeypatch.setattr(api_checks, "ssh_terminal_supported", lambda: True)
    page = pages.RemoteTargetsPage()
    assert not page._update_btn.isEnabled()
    assert page._update_btn.text() == "Update DETAIL…"
    _register("tillicum")
    page = pages.RemoteTargetsPage()
    page._table.selectRow(0)
    assert page._update_btn.isEnabled()


def test_update_checks_then_asks_then_updates_and_shows_the_result(qapp, monkeypatch):
    from detail.gui.status_color import READY

    page, calls, build = _update_page(
        qapp,
        monkeypatch,
        "behind",
        answer=QMessageBox.StandardButton.Yes,
        commits=4,
    )

    def update(name, *, before=None, progress=None, **kwargs):
        calls["updated"].append((name, before))
        progress("Updating", 0.3, "git pull --ff-only")
        return api_remote.RemoteUpdate(
            True, "Updated DETAIL on 'tillicum': a1b2c3d -> c2a1dd8.", before
        )

    monkeypatch.setattr(api_remote, "update_remote_build", update)
    page._update_target()
    assert not page._update_btn.isEnabled(), "busy while it works"
    assert _drain(qapp, lambda: "Updated DETAIL on" in page._status.text())

    (question,) = calls["asked"]
    assert "4 commit(s) behind this app (a1b2c3d → c2a1dd8)" in question
    assert "git pull --ff-only" in question and "/gpfs/home/x/detail" in question
    assert "no run is going" in question
    assert calls["updated"] == [
        ("tillicum", build)
    ], "the comparison shown is the one used"
    assert _color_of(page._status) == READY
    assert _drain(qapp, lambda: page._update_btn.isEnabled())


def test_declining_the_question_changes_nothing_on_the_cluster(qapp, monkeypatch):
    page, calls, _build_ = _update_page(
        qapp, monkeypatch, "behind", answer=QMessageBox.StandardButton.No, commits=2
    )
    monkeypatch.setattr(
        api_remote,
        "update_remote_build",
        lambda *a, **k: calls["updated"].append(a) or None,
    )

    page._update_target()
    assert _drain(qapp, lambda: len(calls["asked"]) == 1)
    assert _drain(qapp, lambda: page._update_btn.isEnabled())

    assert calls["updated"] == []
    assert "behind this app" in page._status.text(), "the check's finding stays shown"


@pytest.mark.parametrize("state", ["same", "ahead", "diverged", "unknown"])
def test_only_a_target_that_is_behind_is_ever_asked_about(qapp, monkeypatch, state):
    from detail.gui.status_color import READY, WARN

    page, calls, build = _update_page(
        qapp,
        monkeypatch,
        state,
        answer=QMessageBox.StandardButton.Yes,
        reason="it could not be read",
    )
    monkeypatch.setattr(
        api_remote,
        "update_remote_build",
        lambda *a, **k: calls["updated"].append(a) or None,
    )

    page._update_target()
    assert _drain(qapp, lambda: page._status.text() == build.describe())
    assert _drain(qapp, lambda: page._update_btn.isEnabled())

    assert calls["asked"] == [] and calls["updated"] == []
    assert _color_of(page._status) == (READY if state == "same" else WARN)


def test_a_refused_update_shows_gits_words_and_is_not_green(qapp, monkeypatch):
    from detail.gui.status_color import WARN

    page, calls, _build_ = _update_page(
        qapp, monkeypatch, "behind", answer=QMessageBox.StandardButton.Yes, commits=1
    )
    monkeypatch.setattr(
        api_remote,
        "update_remote_build",
        lambda name, *, before=None, progress=None, **k: api_remote.RemoteUpdate(
            False,
            "git refused the update on 'tillicum', which is unchanged: fatal: Not "
            "possible to fast-forward, aborting.",
            before,
            output="fatal: Not possible to fast-forward, aborting.",
        ),
    )

    page._update_target()
    assert _drain(qapp, lambda: "git refused the update" in page._status.text())

    assert "Not possible to fast-forward" in page._status.text()
    assert _color_of(page._status) == WARN


def test_a_failed_check_is_an_error_not_a_question(qapp, monkeypatch):
    from detail.gui.status_color import ERROR

    page, calls, _build_ = _update_page(
        qapp, monkeypatch, "behind", answer=QMessageBox.StandardButton.Yes
    )

    def boom(name):
        raise RuntimeError("could not reach 'tillicum'")

    monkeypatch.setattr(api_remote, "check_remote_build", boom)
    page._update_target()
    assert _drain(qapp, lambda: "could not reach" in page._status.text())

    assert calls["asked"] == []
    assert _color_of(page._status) == ERROR


def test_the_status_follows_each_stage_while_it_updates(qapp, monkeypatch):
    import threading

    page, calls, _build_ = _update_page(
        qapp, monkeypatch, "behind", answer=QMessageBox.StandardButton.Yes, commits=1
    )
    proceed = threading.Event()

    def update(name, *, before=None, progress=None, **kwargs):
        progress("Updating", 0.3, "git pull --ff-only in /gpfs/home/x/detail")
        assert proceed.wait(20)
        return api_remote.RemoteUpdate(True, "Updated DETAIL on 'tillicum'.", before)

    monkeypatch.setattr(api_remote, "update_remote_build", update)
    page._update_target()
    try:
        assert _drain(
            qapp, lambda: "Updating — git pull --ff-only" in page._status.text()
        )
    finally:
        proceed.set()
    assert _drain(qapp, lambda: "Updated DETAIL on 'tillicum'." in page._status.text())
