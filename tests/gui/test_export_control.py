"""The one Export control: placed by the shell, scoped by the page (#336).

**The class this guards.** *Seven bespoke exporters, seven placements, no shared contract.* The
capability that prompted the report already existed twice — `export_clusterer_videos` and
`export_cluster_videos` — and was reachable from neither page the user was on, because each was
built onto whichever page its author was looking at. A control any page constructs is a control
some pages will not.

So the placement is structural: the shell owns the button, and its subject comes from the opener
that showed the page. A page cannot hand-list, fall behind, or forget it. What a page *does*
contribute is scope, through `ExportScopeSurface` — the same "shell asks the page" shape
`gui/editing.py` uses.

The drift half — *a result page grew a panel whose data no export reaches* — is
`test_export_reach.py`.
"""

import pytest

from tests.gui.conftest import drain_until  # noqa: E402

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget  # noqa: E402

from detail.api.exporters import ExportScope, ExportSubject  # noqa: E402
from detail.gui import export as gui_export  # noqa: E402

pytestmark = pytest.mark.guard


class _Scoped(QWidget):
    def __init__(self, scope):
        super().__init__()
        self._scope = scope

    def export_scope(self):
        return self._scope


def _embed(child) -> QWidget:
    """A child inside a plain page, the way a result widget sits inside RunDetailPage."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel("some page chrome"))
    layout.addWidget(child)
    return page


# --------------------------------------------------------------------------- #
# scope: the page's half of the contract
# --------------------------------------------------------------------------- #
def test_scope_is_found_on_an_embedded_widget(qapp):
    """A result widget is embedded, so asking only the top-level page finds nothing.

    That failure is silent and expensive: no scope means "everything", so a user who selected
    one cluster and clicked Export would get the whole run without being told.
    """
    scope = ExportScope(cluster="7")
    assert gui_export.scope_of(_embed(_Scoped(scope))) == scope


def test_scope_is_found_on_the_page_itself(qapp):
    scope = ExportScope(instance_ids=("a", "b"))
    assert gui_export.scope_of(_Scoped(scope)) == scope


def test_a_page_with_no_selection_scopes_to_everything(qapp):
    assert gui_export.scope_of(_embed(_Scoped(None))) is None
    assert gui_export.scope_of(QWidget()) is None
    assert gui_export.scope_of(None) is None


def test_the_scope_is_read_when_the_menu_opens_not_when_the_page_was_built(qapp):
    """A scope snapshotted at construction is the selection from before the user chose."""
    surface = _Scoped(None)
    page = _embed(surface)
    assert gui_export.scope_of(page) is None
    surface._scope = ExportScope(cluster="3")
    assert gui_export.scope_of(page) == ExportScope(cluster="3")


# --------------------------------------------------------------------------- #
# the control
# --------------------------------------------------------------------------- #
def test_the_button_hides_when_the_subject_can_export_nothing(qapp, tmp_path):
    button = gui_export.ExportButton()
    button.set_subject(None)
    assert button.isHidden()
    button.set_subject(ExportSubject("analysis_run", tmp_path, "an_99"))
    assert button.isHidden(), "a run with no artifact on disk offers nothing"


def test_the_button_appears_for_a_subject_with_targets(qapp, authored_cohort):
    button = gui_export.ExportButton()
    button.set_subject(
        ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")
    )
    # `isHidden` rather than `isVisible`: the button has no shown ancestor here, so
    # `isVisible()` is False whatever `setVisible` was called with.
    assert not button.isHidden()


# --------------------------------------------------------------------------- #
# the dialog
# --------------------------------------------------------------------------- #
def _dialog(cohort, target_ids=("clips", "instances", "ethogram"), scope=None):
    """A dialog whose clip target can actually run.

    The fixture's run records no `embed_run_id`, so clips would be greyed out — and a form
    with its clip knobs hidden is not the form these tests are about.
    """
    from detail.api import exporters
    from detail.gui.export import ExportDialog
    from detail.storage import analysis_runs as analysis_store

    run_dir = analysis_store.analysis_dir(cohort.experiment.dir, "an_01")
    record = analysis_store.read_analysis_yaml(run_dir)
    record.setdefault("params", {})["embed_run_id"] = "emb_01"
    analysis_store.write_analysis_yaml(run_dir, record)

    subject = ExportSubject("analysis_run", cohort.experiment.dir, "an_01")
    targets = [exporters.target("analysis_run", t) for t in target_ids]
    return ExportDialog(subject, targets, scope)


def test_the_dialog_lists_everything_the_screen_can_produce(qapp, authored_cohort):
    """One dialog with a checklist, not a menu of dialogs.

    The menu's defect was structural: having already picked one target, the dialog's scope row
    could only narrow *that one*, so its "Everything" was a claim about a single file.
    """
    dialog = _dialog(authored_cohort)
    assert set(dialog._include) == {"clips", "instances", "ethogram"}
    assert all(c.isChecked() for c in dialog._include.values())
    assert all(c.isEnabled() for c in dialog._include.values())


def test_the_bundle_is_offered_but_not_ticked(qapp, authored_cohort):
    """It round-trips and the others do not, so it is a deliberate choice rather than a default."""
    from detail.api import exporters
    from detail.gui.export import ExportDialog

    subject = ExportSubject("experiment", authored_cohort.experiment.dir)
    targets = exporters.targets_for(subject)
    dialog = ExportDialog(subject, targets, None)
    assert dialog._include["bundle"].isChecked() is False
    assert dialog._include["metadata"].isChecked() is True


def test_the_dialog_shows_the_folder_it_will_create(qapp, authored_cohort):
    """Named before anything is written, so you know what you are about to get."""
    dialog = _dialog(authored_cohort)
    shown = dialog._folder.text()
    assert shown.endswith("/")
    assert "an_01" in shown and "export" in shown


def test_an_untickable_export_is_refused_rather_than_writing_nothing(
    qapp, authored_cohort, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    dialog = _dialog(authored_cohort)
    dialog._out.setText("/tmp/somewhere")
    for check in dialog._include.values():
        check.setChecked(False)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    dialog._export()
    assert warned and "at least one" in warned[0]


def test_the_dialog_offers_the_three_flavours_for_clips(qapp, authored_cohort):
    from detail.api import clip_render

    dialog = _dialog(authored_cohort)
    assert set(dialog._renders) == set(clip_render.RENDERS)
    assert dialog._renders[clip_render.OVERLAY].isChecked()
    assert not dialog._crop.isChecked(), "the full field of view is the default"


def test_a_dialog_with_no_clip_target_shows_no_clip_knobs(qapp, authored_cohort):
    """A form offering "clip size" for a CSV is a form nobody trusts."""
    dialog = _dialog(authored_cohort, target_ids=("instances", "ethogram"))
    assert dialog._renders == {}
    assert dialog._size is None


def test_the_scope_row_is_absent_when_there_is_nothing_to_choose(qapp, authored_cohort):
    """A picker with one entry cannot be operated, and its single entry reads as a claim.

    That was the reported defect: "Everything" was the only option and was not everything.
    """
    assert _dialog(authored_cohort, scope=None)._scope_pick is None


def test_the_scope_row_appears_with_a_selection_and_prefers_it(qapp, authored_cohort):
    dialog = _dialog(authored_cohort, scope=ExportScope(cluster="c01"))
    assert dialog._scope_pick is not None
    items = [dialog._scope_pick.itemText(i) for i in range(dialog._scope_pick.count())]
    assert items == ["Cluster c01", "Everything in this run"]
    assert dialog._scope_pick.currentData() is True, "the selection is preselected"


def test_the_clip_options_reach_the_writer(qapp, authored_cohort):
    """A knob the form draws and never passes on is the quieter half of a hand-listed config."""
    from detail.api import clip_render

    dialog = _dialog(authored_cohort)
    dialog._renders[clip_render.MASK].setChecked(True)
    dialog._renders[clip_render.OVERLAY].setChecked(False)
    dialog._crop.setChecked(True)
    dialog._size.setValue(320)
    dialog._max_per_cluster.setValue(5)
    dialog._codec.set_codec("mp4v")
    assert dialog._options() == {
        "renders": (clip_render.MASK,),
        "crop": True,
        "size": 320,
        "max_per_cluster": 5,
        "codec": "mp4v",
    }


def test_the_included_files_are_named_on_each_row(qapp, authored_cohort):
    """The half that makes this more than Save-As, as filenames rather than a description."""
    from detail.api import exporters

    dialog = _dialog(authored_cohort)
    tip = dialog._include["instances"].toolTip()
    for name in exporters.target("analysis_run", "instances").included_files():
        assert name in tip, f"{name} is not named on the row that writes it"


# --------------------------------------------------------------------------- #
# grey out, don't hide
# --------------------------------------------------------------------------- #
def _recording_dialog(experiment, recording_id="r1"):
    from detail.api import exporters
    from detail.gui.export import ExportDialog

    subject = ExportSubject("recording", experiment.dir, recording_id)
    return ExportDialog(subject, exporters.targets_for(subject), None)


def test_a_target_that_cannot_run_is_shown_disabled_with_its_reason(
    qapp, authored_cohort
):
    """The house rule, and the reason for it.

    A hidden row leaves the reader unable to tell "DETAIL cannot export my streams" from "this
    recording has no streams yet" — and only the second is something they can act on. The
    fixture's recordings have no streams and no geometry, which is the state a new user is in.
    """
    dialog = _recording_dialog(authored_cohort.experiment)
    assert set(dialog._include) == {
        "geometry",
        "streams",
        "timestamps",
        "video",
        "clip",
    }

    off = {k: c for k, c in dialog._include.items() if not c.isEnabled()}
    assert off, "nothing was disabled, so this asserts nothing"
    for key, check in off.items():
        assert "\u2014" in check.text(), f"{key} is greyed out but does not say why"
        assert not check.isChecked(), f"{key} is disabled but ticked"


def test_the_reason_names_the_missing_thing(qapp, authored_cohort):
    dialog = _recording_dialog(authored_cohort.experiment)
    text = dialog._include["streams"].text()
    assert "no streams" in text, f"the streams row says {text!r}"


def test_referenced_video_says_the_files_are_already_yours(qapp, authored_cohort):
    """Not "unavailable": a reference-in-place video is not missing, it is somewhere else.

    Copying it back out would be DETAIL exporting the user's own file to the user, so the row
    is off — and it has to say that rather than reading as a failure.
    """
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    directory = experiment_store.recording_path(authored_cohort.experiment.dir, "r1")
    data = recording_store.read_recording(directory)
    data["video"] = {"files": [{"source": {"root": "rig", "rel": "day1/clip.mp4"}}]}
    recording_store.write_recording(directory, data)

    dialog = _recording_dialog(authored_cohort.experiment)
    text = dialog._include["video"].text()
    assert not dialog._include["video"].isEnabled()
    assert "referenced in place" in text, f"the video row says {text!r}"


def test_the_save_row_is_labelled_save_to(qapp, authored_cohort):
    from PyQt6.QtWidgets import QFormLayout

    dialog = _recording_dialog(authored_cohort.experiment)
    form = dialog.findChild(QFormLayout)
    labels = [
        form.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
        for i in range(form.rowCount())
        if form.itemAt(i, QFormLayout.ItemRole.LabelRole) is not None
    ]
    assert "Save to:" in labels, labels
    assert dialog._out.placeholderText() == ""


# --------------------------------------------------------------------------- #
# the bundle's own knobs
# --------------------------------------------------------------------------- #
def _experiment_dialog(experiment):
    from detail.api import exporters
    from detail.gui.export import ExportDialog

    subject = ExportSubject("experiment", experiment.dir)
    return ExportDialog(subject, exporters.targets_for(subject), None)


def test_the_bundle_exposes_its_video_and_dataset_knobs(qapp, authored_cohort):
    """They existed on the API from the start and no surface exposed them.

    So the GUI always wrote a slim bundle — no video, no datasets — with nothing on screen
    saying so. A default only the person reading the source could know about.
    """
    dialog = _experiment_dialog(authored_cohort.experiment)
    assert dialog._bundle_video is not None
    assert dialog._bundle_datasets is not None
    assert not dialog._bundle_video.isChecked(), "slim stays the default"


def test_the_bundle_knobs_reach_the_writer(qapp, authored_cohort):
    dialog = _experiment_dialog(authored_cohort.experiment)
    dialog._bundle_video.setChecked(True)
    dialog._bundle_datasets.setChecked(True)
    dialog._bundle_gzip.setChecked(True)
    options = dialog._options()
    assert options["include_video"] is True
    assert options["include_datasets"] is True
    assert options["gzip"] is True


def test_ticking_a_bundle_knob_asks_for_a_new_estimate(
    qapp, authored_cohort, monkeypatch
):
    """The three boxes change what the bundle packs and how long it takes, and none of them
    was connected: ticking "Source video" left the estimate quoting a slim bundle."""
    dialog = _experiment_dialog(authored_cohort.experiment)
    # Observed through what `_changed` does (it restarts the estimate's debounce timer), not by
    # patching `_changed`: the signals hold the bound method they were connected to, so a patch
    # made afterwards is never called and the test would fail with the wiring present.
    for box in (dialog._bundle_video, dialog._bundle_datasets, dialog._bundle_gzip):
        dialog._estimate_timer.stop()
        box.toggle()
        assert dialog._estimate_timer.isActive(), f"{box.text()!r} does not re-estimate"


def test_the_experiment_pages_export_opens_this_dialog_on_the_bundle(
    qapp, authored_cohort
):
    """`ExperimentOverviewPage` asks for `only=("bundle",)`: the archive ticked, nothing else,
    so its Export does what it always did, through the one dialog (#916)."""
    from detail.api import exporters
    from detail.gui.export import ExportDialog

    subject = ExportSubject("experiment", authored_cohort.experiment.dir)
    dialog = ExportDialog(
        subject, exporters.targets_for(subject), None, only=("bundle",)
    )
    ticked = {tid for tid, box in dialog._include.items() if box.isChecked()}
    assert ticked == {"bundle"}
    assert not dialog._bundle_gzip.isChecked(), "uncompressed stays the default"


def test_a_screen_with_no_bundle_shows_no_bundle_knobs(qapp, authored_cohort):
    """A form offering "source video" for a CSV export is a form nobody trusts."""
    dialog = _dialog(authored_cohort, target_ids=("instances",))
    assert dialog._bundle_video is None


# --------------------------------------------------------------------------- #
# Export… submits a run — it does not export
# --------------------------------------------------------------------------- #
# **The class.** *A form that draws a knob and never passes it on.* The option tests above
# assert `_options()` in isolation, which proved something while the dialog itself called
# `write_export` on the next line. Now it submits a run, and a value that reaches `_options()`
# and not `run_export` is a setting the user chose and the export ignored — the quieter half of
# a hand-listed config, and invisible to every test that stops at the form.


def _submit(qapp, dialog, monkeypatch, out, *, fail=None):
    """Click Export with `run_export` spied on, and drain until the background call lands."""
    from detail.api.experiment import Experiment

    seen = {}

    def spy(self, subject_kind, subject_id, target_ids, parent, **kwargs):
        seen.update(
            subject=(subject_kind, subject_id),
            target_ids=list(target_ids),
            parent=str(parent),
            **kwargs,
        )
        if fail is not None:
            raise fail
        return _Handle()

    monkeypatch.setattr(Experiment, "run_export", spy, raising=False)
    dialog._out.setText(str(out))
    dialog._export()
    # Drain until the *slot* has run, not until the spy has. `seen` is filled on the worker
    # thread and would let this return before `done`/`failed` was ever delivered — the recorded
    # "draining on a value that is already truthy" trap, one thread over. `_export` disables the
    # buttons, so either a handle arrived or they came back.
    #
    # Through `drain_until`, which is bounded by **wall clock**. This was `for _ in range(400):
    # processEvents()`, a spin count -- and `processEvents` returns immediately when nothing is
    # queued, so 400 bare pumps burn through in microseconds while the `BackgroundCall` is still
    # running. Green on an idle box, flaky inside a loaded suite, which is what `drain_until`'s own
    # docstring says and why it was centralised: five files had grown their own copy. This was an
    # unmigrated sixth, and the broken kind. It cost two intermittent failures in this file under
    # `-n auto` (#580, and `test_a_submitted_export_hands_its_handle_up_and_closes` seen on
    # another branch), each green 3/3 in isolation.
    assert drain_until(
        qapp,
        lambda: dialog.run_handle is not None or dialog._buttons.isEnabled(),
    ), "the export call never landed: neither a handle nor the buttons came back"
    return seen


class _Handle:
    id = "export_001"


def test_export_submits_a_run_carrying_what_the_form_says(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """Every choice on the form reaches the run: targets, destination, scope and the knobs."""
    from detail.api import clip_render
    from detail.api.exporters import ExportScope

    dialog = _dialog(authored_cohort, scope=ExportScope(cluster="c01"))
    dialog._include["ethogram"].setChecked(False)
    dialog._renders[clip_render.MASK].setChecked(True)
    dialog._renders[clip_render.OVERLAY].setChecked(False)
    dialog._crop.setChecked(True)
    dialog._size.setValue(320)

    seen = _submit(qapp, dialog, monkeypatch, tmp_path / "out")

    assert seen["subject"] == ("analysis_run", "an_01")
    assert seen["target_ids"] == [
        "clips",
        "instances",
    ], "the unticked target was not sent"
    assert seen["parent"] == str(tmp_path / "out")
    assert seen["scope"] == ExportScope(cluster="c01")
    assert seen["renders"] == (clip_render.MASK,)
    assert seen["crop"] is True and seen["size"] == 320


def test_export_does_not_write_anything_itself(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """The regression this PR exists to prevent, and one "it exported" would not catch.

    A dialog that still called `write_export` would pass every assertion about the *files* and
    be exactly the modal-overlay-for-forty-minutes behaviour #354 removed.
    """
    from detail.api import exporters

    called = []
    monkeypatch.setattr(
        exporters, "write_export", lambda *a, **k: called.append(a) or _Handle()
    )
    dialog = _dialog(authored_cohort)
    _submit(qapp, dialog, monkeypatch, tmp_path / "out")

    assert not called, "the dialog exported inline instead of submitting a run"


def test_a_submitted_export_hands_its_handle_up_and_closes(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """`run_handle` is how the shell learns to open the run; without it the run is invisible."""
    dialog = _dialog(authored_cohort)
    _submit(qapp, dialog, monkeypatch, tmp_path / "out")

    assert dialog.run_handle is not None
    assert dialog.run_handle.id == "export_001"
    assert dialog.result() == int(dialog.DialogCode.Accepted)


def test_a_refused_export_leaves_the_form_open_and_usable(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """Submission is where a bad target is caught, so this is the path a user actually meets.

    A submit-and-close flow makes the failure path easy to forget: the dialog must **not** close,
    and its controls must come back, or the user is left with a dead window.
    """
    from detail.api.exporters import ExportError

    messages = []
    monkeypatch.setattr(
        "detail.gui.export.QMessageBox.critical",
        lambda *a, **k: messages.append(a[-1]),
    )
    dialog = _dialog(authored_cohort)
    _submit(qapp, dialog, monkeypatch, tmp_path / "out", fail=ExportError("nope"))

    assert dialog.run_handle is None
    assert dialog.result() != int(dialog.DialogCode.Accepted), "it closed on a failure"
    assert dialog._buttons.isEnabled(), "the form never came back"
    assert any("nope" in str(m) for m in messages), "the failure was silent"


def test_the_button_forwards_the_handle_to_whoever_is_listening(qapp):
    """`ExportButton` cannot open a run page and must not learn how — it emits."""
    from detail.gui.export import ExportButton

    button = ExportButton()
    got = []
    button.launched.connect(got.append)
    button.launched.emit(_Handle())
    assert [h.id for h in got] == ["export_001"]


# --------------------------------------------------------------------------- #
# nothing holds you while it runs
# --------------------------------------------------------------------------- #
def _shell(qapp, experiment):
    """The real `MainWindow`, opened on an experiment, with its own export notice.

    Through the shell rather than a stand-in: the whole claim here is about what happens to the
    *screen you were on*, and a stand-in has no screen.
    """
    from detail.gui.app import MainWindow

    window = MainWindow()
    window._experiment = experiment
    return window


class _LaunchedHandle:
    """What `ExportButton.launched` carries. Named apart from `_Handle` above deliberately: two
    classes of one name in one module means the later one silently wins, which is what a second
    `_Handle` here did to every test using the first."""

    def __init__(self, run_id):
        self.id = run_id


def _an_export_record(
    experiment, run_id="ex_01", targets=("instances",), subject_kind="analysis_run"
):
    from detail.storage import export_runs as export_store

    export_store.write_export_yaml(
        export_store.export_dir(experiment.dir, run_id),
        {
            "id": run_id,
            "subject_kind": subject_kind,
            "subject_id": "an_01" if subject_kind == "analysis_run" else "",
            "target_ids": list(targets),
            "folder": "cohort_an_01_export_2026-09-12_140420",
            "destination": {"root": "nowhere", "rel": "out"},
            "params": {},
            "status": "running",
            "created_at": "2026-09-12T14:04:20Z",
        },
    )
    return run_id


def test_starting_an_export_does_not_take_the_screen_away(qapp, authored_cohort):
    """The behaviour this change exists for.

    Every other launch in this app navigates to the run it started, which is right when starting
    that run is why you opened the app. An export is an errand off a screen you are in the
    middle of *using* — a cluster you were naming, a plot you were reading — and swapping that
    screen for a progress bar costs more than the bar is worth.
    """
    from PyQt6.QtWidgets import QLabel

    window = _shell(qapp, authored_cohort.experiment)
    mine = QLabel("the screen I was on")
    window._show(mine, "somewhere")
    _an_export_record(authored_cohort.experiment)

    window._on_export_launched(_LaunchedHandle("ex_01"))

    assert (
        window._content.widget(0) is mine
    ), "the shell navigated away from the screen the export was started from"


def _opened_runs(window) -> list:
    """Every export run page the shell is asked to open. Patches the destination, not the wiring:
    the call under test is the one `_on_export_launched` makes."""
    opened: list = []
    window._open_export_run = lambda run_id, live=None: opened.append((run_id, live))
    return opened


def test_an_experiment_export_goes_to_its_bar(qapp, authored_cohort):
    """Exporting the whole experiment is not an errand off a screen in use. It is minutes, and a
    user who started one had to find it in Active processes and then open it (#916) — so the
    shell goes to the run's page, where the bar is."""
    from PyQt6.QtWidgets import QLabel

    window = _shell(qapp, authored_cohort.experiment)
    window._show(QLabel("the overview"), "overview")
    _an_export_record(
        authored_cohort.experiment, targets=("bundle",), subject_kind="experiment"
    )
    opened = _opened_runs(window)
    handle = _LaunchedHandle("ex_01")

    window._on_export_launched(handle)

    assert opened == [
        ("ex_01", handle)
    ], "the launch handle is what puts the bar up at once"
    assert window._export_notice.isHidden()


def test_unsaved_edits_keep_an_experiment_export_on_screen(qapp, authored_cohort):
    """Going to the bar is a navigation, so it asks first like every other; answering Cancel
    keeps the screen, and the notice still says the export started."""
    window = _shell(qapp, authored_cohort.experiment)
    _an_export_record(
        authored_cohort.experiment, targets=("bundle",), subject_kind="experiment"
    )
    opened = _opened_runs(window)
    window._may_leave = lambda: False

    window._on_export_launched(_LaunchedHandle("ex_01"))

    assert opened == []
    assert not window._export_notice.isHidden()


def test_a_run_export_still_stays_put(qapp, authored_cohort):
    """The errand case is unchanged: an export of a run never takes the screen."""
    window = _shell(qapp, authored_cohort.experiment)
    _an_export_record(authored_cohort.experiment)
    opened = _opened_runs(window)

    window._on_export_launched(_LaunchedHandle("ex_01"))

    assert opened == []
    assert not window._export_notice.isHidden()


def test_the_notice_says_what_started_and_where_it_is_going(qapp, authored_cohort):
    """A notice that only says "started" sends you to look for the answer it could have given.

    Both halves are read off the **record**, not the form, so the notice and the Active-processes
    row cannot disagree: one fact a person reads has one formatter.
    """
    window = _shell(qapp, authored_cohort.experiment)
    _an_export_record(authored_cohort.experiment)

    window._on_export_launched(_LaunchedHandle("ex_01"))

    assert not window._export_notice.isHidden()
    text = window._export_notice._label.text()
    assert "nstances" in text, f"the notice does not say what it started: {text!r}"
    assert "cohort_an_01_export_2026-09-12_140420" in text, text
    assert "Active processes" in text, text
    assert text == " ".join(
        window._export_notice._label.text().split()
    ), "the notice was composed by appending to its own text"


def test_the_notice_and_the_listing_spell_the_path_the_same_way(
    qapp, authored_cohort, monkeypatch, tmp_path
):
    """One fact a person reads, one formatter.

    The notice and the Active-processes row print the same destination, and for a day the notice
    printed ``/Users/me/Downloads/…`` beside a table printing ``~/Downloads/…``. Both go through
    `exporters.describe_run` now, which is the only way the two cannot drift — a second
    abbreviation written next to whichever surface was being worked on is the recorded class.
    """
    from detail.api import exporters
    from detail.pipeline.export_run import EXPORT_ROOTS_KEY
    from detail.storage import roots

    home = tmp_path / "home"
    (home / "Downloads").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda _cls: home))
    roots.register_root(EXPORT_ROOTS_KEY, "dl", home / "Downloads")

    window = _shell(qapp, authored_cohort.experiment)
    run_id = _an_export_record(authored_cohort.experiment)
    record = authored_cohort.experiment.export_record(run_id)
    record["destination"] = {"root": "dl", "rel": ""}
    from detail.storage import export_runs as export_store

    export_store.write_export_yaml(
        export_store.export_dir(authored_cohort.experiment.dir, run_id), record
    )

    _, where = exporters.describe_run(authored_cohort.experiment.export_record(run_id))
    assert where.startswith("~/"), f"the shared formatter stopped abbreviating: {where}"

    window._on_export_launched(_LaunchedHandle(run_id))
    assert where in window._export_notice._label.text()


def test_the_notice_survives_a_record_it_cannot_read(qapp, authored_cohort):
    """A notice is the *reassurance*; it must never be the reason a launch looks like a failure.

    The run is submitted and going by the time this is called — there is nothing left to fail.
    """
    window = _shell(qapp, authored_cohort.experiment)

    window._on_export_launched(_LaunchedHandle("ex_99"))  # no such record

    assert not window._export_notice.isHidden()
    assert window._export_notice._label.text()


def test_the_notice_clears_on_the_next_navigation(qapp, authored_cohort):
    """It belongs to the screen it was raised on. Once you have gone to look at something, it
    has said its piece — and a stale "export started" strip riding along every page is the kind
    of chrome people learn to stop reading."""
    from PyQt6.QtWidgets import QLabel

    window = _shell(qapp, authored_cohort.experiment)
    _an_export_record(authored_cohort.experiment)
    window._on_export_launched(_LaunchedHandle("ex_01"))
    assert not window._export_notice.isHidden()

    window._show(QLabel("somewhere else"), "elsewhere")

    assert window._export_notice.isHidden()


def test_the_notice_routes_to_active_processes(qapp, authored_cohort):
    """The one route worth offering, and the only reason the strip has a button at all. A notice
    naming a place with no way to get there makes the reader do the navigation it just described.
    """
    window = _shell(qapp, authored_cohort.experiment)
    _an_export_record(authored_cohort.experiment)
    window._on_export_launched(_LaunchedHandle("ex_01"))

    went = []
    # Patch the destination, not the wiring: the connection under test is the one the shell made
    # at construction, and re-connecting here would test a connection this test wrote.
    window._switch_mode = lambda mode: went.append(mode)
    window._export_notice.show_processes.emit()

    assert went == ["runs"]


# --------------------------------------------------------------------------- #
# dark figures — offered where a figure is drawn, sent only when asked
# --------------------------------------------------------------------------- #
def test_dark_figures_are_offered_when_a_target_draws_one(qapp, authored_cohort):
    dialog = _dialog(authored_cohort)  # instances + ethogram draw figures
    assert dialog._dark_figures is not None
    assert "theme" not in dialog._options(), "light is every target's own default"
    dialog._dark_figures.setChecked(True)
    assert dialog._options()["theme"] == "dark"


def test_no_dark_figures_control_where_nothing_draws_one(qapp, authored_cohort):
    """A control for nothing is a question the user cannot answer."""
    from detail.api import exporters
    from detail.gui.export import ExportDialog

    subject = ExportSubject("experiment", authored_cohort.experiment.dir)
    targets = [t for t in exporters.targets_for(subject) if not t.figure_files]
    assert targets, "the fixture must offer something that draws no figure"
    dialog = ExportDialog(subject, targets, None)
    assert dialog._dark_figures is None


def _keypoints_run_subject(built):
    from detail.storage import runs as runs_store

    recording = built.recording
    run_dir = recording.dir / "keypoints" / "kp_01"
    run_dir.mkdir(parents=True)
    (run_dir / "keypoints.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    video = recording.dir / "video" / "part_001.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"\0")  # availability asks only that the file is here
    runs_store.write_run(
        built.experiment.dir,
        "run_kp",
        {
            "id": "run_kp",
            "module_id": "keypoints",
            "per_recording": {recording.id: {"run_id": "kp_01", "status": "complete"}},
            "status": "complete",
        },
    )
    return ExportSubject("run", built.experiment.dir, "run_kp")
