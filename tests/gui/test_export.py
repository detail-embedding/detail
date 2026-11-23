"""GUI export / import surfaces (§15): the Experiment page's Export and the Home-page Import,
including the collision → replace confirmation.

Both used to run in a `BackgroundCall` whose only feedback was the button's label (#916). Export
is now the shared Export dialog submitting an export run (listed under Active processes); Import
runs behind `run_progress`, a bar with a Cancel.
"""

import cv2
import numpy as np

from detail.api.experiment import Experiment
from detail.gui import pages
from detail.storage import export as export_store
from detail.storage import locations


def _video(path, n=4, w=16, h=16):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 7) % 250 + 1, np.uint8))
    vw.release()
    return path


def _experiment(root):
    proj = Experiment.create("study", root)
    proj.create_recording("recording_001", metadata={"animal_id": "M001"})
    proj.get_recording("recording_001").add_video_part(_video(root / "_src.mp4"))
    return proj


# --- the Experiment page's Export ----------------------------------------------- #
def test_the_overview_export_opens_the_shared_dialog_and_hands_the_run_up(
    qapp, tmp_path, monkeypatch
):
    """Not a `BackgroundCall` behind a dialog of its own any more: the one Export dialog, with
    the bundle ticked, whose run the shell announces and Active processes lists."""
    from detail.gui import export as gui_export

    proj = Experiment.create("study", tmp_path)
    asked = []
    handle = object()

    def fake_open_export(parent, subject, scope=None, *, only=(), seed=None):
        asked.append((subject.kind, subject.experiment_dir, only))
        return handle

    monkeypatch.setattr(gui_export, "open_export", fake_open_export)
    launched = []
    page = pages.ExperimentOverviewPage(
        proj, on_open_section=lambda k: None, on_export_launched=launched.append
    )
    page._export_btn.click()

    assert asked == [("experiment", proj.dir, ("bundle",))]
    assert launched == [handle], "the submitted run never reached the shell"


def test_a_cancelled_export_dialog_hands_nothing_up(qapp, tmp_path, monkeypatch):
    from detail.gui import export as gui_export

    monkeypatch.setattr(gui_export, "open_export", lambda *a, **k: None)
    launched = []
    page = pages.ExperimentOverviewPage(
        Experiment.create("study", tmp_path),
        on_open_section=lambda k: None,
        on_export_launched=launched.append,
    )
    page._export_btn.click()
    assert launched == []


# --- Home-page Import -------------------------------------------------------- #
def _sync_run_progress(monkeypatch, *, cancel=False):
    """`run_progress` run inline, recording what the work reported.

    ``cancel`` raises the way the overlay's Cancel does, at the first report.
    """
    from detail.core.progress import OperationCancelled
    from detail.gui import busy

    emitted = []

    def run(target, work, on_done, *, on_error=None, on_cancel=None, **_kw):
        def emit(*event):
            emitted.append(event)
            if cancel:
                raise OperationCancelled()

        try:
            result = work(emit)
        except OperationCancelled:
            on_cancel()
            return
        except Exception as exc:  # noqa: BLE001 — mirrors run_progress's own boundary
            on_error(exc)
            return
        on_done(result)

    monkeypatch.setattr(busy, "run_progress", run)
    return emitted


def _home(monkeypatch, tmp_path, archive):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    monkeypatch.setattr(locations, "experiments_root", lambda: ws)
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(archive), "")
    )
    return ws


def test_home_import_places_and_opens(qapp, tmp_path, monkeypatch):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "study.tar")
    ws = _home(monkeypatch, tmp_path, archive)
    emitted = _sync_run_progress(monkeypatch)

    told = []
    monkeypatch.setattr(pages.QMessageBox, "information", lambda *a: told.append(a))

    opened = []
    page = pages.HomePage(on_open=opened.append)
    page._import_experiment()

    assert len(opened) == 1
    assert opened[0].dir == ws / "study"
    assert (ws / "study" / "config.yaml").is_file()
    assert told == [], "a notice about video when every recording's video is here"
    assert page._import_btn.isEnabled()
    stages = {stage for stage, _fraction, _detail in emitted}
    assert {"extracting", "verifying"} <= stages, "the bar was never driven"


def test_home_import_cancelled_places_nothing(qapp, tmp_path, monkeypatch):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "study.tar")
    ws = _home(monkeypatch, tmp_path, archive)
    _sync_run_progress(monkeypatch, cancel=True)
    errors = []
    monkeypatch.setattr(
        pages.QMessageBox, "critical", lambda *a, **k: errors.append(a[-1])
    )

    opened = []
    page = pages.HomePage(on_open=opened.append)
    page._import_experiment()

    assert opened == [] and errors == []  # a cancel is not a failure
    assert list(ws.iterdir()) == [], "a cancelled import left something behind"
    assert page._import_btn.isEnabled()


def test_home_import_collision_then_replace(qapp, tmp_path, monkeypatch):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "study.tar")
    ws = _home(monkeypatch, tmp_path, archive)
    export_store.import_experiment(archive, ws)  # pre-existing same-id experiment
    _sync_run_progress(monkeypatch)
    monkeypatch.setattr(
        pages.QMessageBox,
        "question",
        lambda *a, **k: pages.QMessageBox.StandardButton.Yes,
    )

    opened = []
    page = pages.HomePage(on_open=opened.append)
    page._import_experiment()  # collision → confirm Yes → replace → open

    assert len(opened) == 1
    assert opened[0].dir == ws / "study"
    assert page._import_btn.isEnabled()


def test_home_import_replace_prompt_lists_what_it_deletes(qapp, tmp_path, monkeypatch):
    """Yes in this prompt deleted the experiment's copied video, datasets and export records
    when the bundle was slim, and the prompt said only "Replace the existing experiment with
    the imported one?" (#29). It lists them now, in the CLI's words
    (`api.experiment.replace_warning`), from the list the import's refusal carries, so nothing
    walks the experiment on the UI thread."""
    from detail.api.exporters import format_size
    from detail.storage import experiment as experiment_store

    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "study.tar")  # slim
    ws = _home(monkeypatch, tmp_path, archive)
    existing = export_store.import_experiment(
        proj.export(tmp_path / "full.tar", include_video=True), ws
    )
    (experiment_store.exports_dir(existing) / "export_01").mkdir(parents=True)
    (video,) = (existing / "recordings").rglob("*.mp4")
    _sync_run_progress(monkeypatch)
    asked = []

    def answer(_parent, _title, text, *rest):
        asked.append(text)
        return pages.QMessageBox.StandardButton.No

    monkeypatch.setattr(pages.QMessageBox, "question", answer)

    pages.HomePage(on_open=lambda _e: None)._import_experiment()

    assert asked == [
        "An experiment named 'study' already exists.\n\n"
        "Replacing it deletes these, and the bundle does not bring them back:\n"
        f"\u2022 1 video file(s), {format_size(video.stat().st_size)}: the bundle carries "
        "no video (it can be put back from the original files with Locate video, or "
        "`detail recording relink`)\n"
        "\u2022 1 export record(s), export_01: no bundle carries them (the exported files "
        "themselves are not touched)\n\n"
        "Replace it with the imported one?"
    ]
    assert video.is_file(), "declining replaced anyway"


def test_home_import_collision_declined(qapp, tmp_path, monkeypatch):
    proj = _experiment(tmp_path / "a")
    archive = proj.export(tmp_path / "study.tar")
    ws = _home(monkeypatch, tmp_path, archive)
    export_store.import_experiment(archive, ws)
    _sync_run_progress(monkeypatch)
    monkeypatch.setattr(
        pages.QMessageBox,
        "question",
        lambda *a, **k: pages.QMessageBox.StandardButton.No,
    )

    opened = []
    page = pages.HomePage(on_open=opened.append)
    page._import_experiment()  # collision → declined → nothing opened, button restored

    assert opened == []
    assert page._import_btn.isEnabled()


def test_home_import_bad_archive_reports(qapp, tmp_path, monkeypatch):
    bad = tmp_path / "bad.tar"
    bad.write_bytes(b"not a tar archive at all")
    _home(monkeypatch, tmp_path, bad)
    _sync_run_progress(monkeypatch)
    errors = []
    monkeypatch.setattr(
        pages.QMessageBox, "critical", lambda *a, **k: errors.append(a[-1])
    )

    opened = []
    page = pages.HomePage(on_open=opened.append)
    page._import_experiment()

    assert opened == []
    assert errors  # the failure surfaced as a critical message
    assert page._import_btn.isEnabled()


def test_home_import_says_how_long_is_left(qapp, tmp_path, monkeypatch):
    """The import's fraction is weighed by measured seconds, so its overlay may extrapolate one;
    a user importing a 17 GB experiment wants the time left, not just a bar (#916)."""
    from detail.gui import busy

    _home(monkeypatch, tmp_path, tmp_path / "study.tar")
    asked = {}
    monkeypatch.setattr(busy, "run_progress", lambda *_a, **kw: asked.update(kw))

    pages.HomePage(on_open=lambda _e: None)._import_experiment()

    assert asked.get("eta") is True, asked


def test_home_import_names_the_recordings_whose_video_was_not_found(
    qapp, tmp_path, monkeypatch
):
    """An import looks for referenced video where the exporting machine had it (#916). What it
    could not find, it says, by recording name, with the one thing to do about it: the user had
    found out by opening a recording and seeing its video missing."""
    from detail.storage import recording as recording_store

    proj = _experiment(tmp_path / "a")
    directory = proj.get_recording("recording_001").dir
    data = recording_store.read_recording(directory)
    for part in data["video"]["files"]:
        (directory / part.pop("file")).unlink()
        part["source"] = {
            "root": "rig",
            "rel": "day1/clip.mp4",
        }  # a root nobody registered
    recording_store.write_recording(directory, data)
    proj.get_recording("recording_001").set_name("Mouse 7 day 1")
    archive = proj.export(tmp_path / "study.tar")
    _home(monkeypatch, tmp_path, archive)
    _sync_run_progress(monkeypatch)
    told = []
    monkeypatch.setattr(
        pages.QMessageBox,
        "information",
        lambda _parent, _title, text: told.append(text),
    )

    opened = []
    pages.HomePage(on_open=opened.append)._import_experiment()

    assert len(opened) == 1, "the experiment did not open"
    assert told == [
        "Video for 1 recording(s) was not found on this computer: Mouse 7 day 1. A file "
        "at the same path that is a different video counts as not found.\n\n"
        "Open one of them and use Locate video on its file. Every recording whose video "
        "is in the same folder will find it too."
    ]


# --- a refused export ----------------------------------------------------------- #
def test_a_refused_export_gives_every_tick_box_back_the_state_it_had(
    qapp, processed_recording, monkeypatch
):
    """#35 finding 15. Going busy greyed every box, and coming back set each to "enabled if
    it is enabled now" — which read the greying itself, so after a refused submit no box
    could be ticked again although the error said the form was left usable. A target that
    cannot run here stays greyed; every other box comes back."""
    from detail.gui.qt import QMessageBox

    from detail.api import exporters
    from detail.api.exporters import ExportSubject
    from detail.gui import export as gui_export
    from tests import waits

    built = processed_recording
    subject = ExportSubject("run", built.experiment.dir, built.detection_run_id)
    dialog = gui_export.ExportDialog(subject, exporters.targets_for(subject), None)
    boxes = {
        **dialog._include,
        **{f"render:{k}": c for k, c in dialog._renders.items()},
    }
    before = {name: box.isEnabled() for name, box in boxes.items()}
    assert any(before.values()), "the fixture offers nothing to tick"
    assert not all(
        before.values()
    ), "the fixture greys nothing, so a greyed box is untested"

    def refuse(*_args, **_kwargs):
        raise ValueError("cannot create the destination")

    monkeypatch.setattr(Experiment, "run_export", refuse)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a[-1]))
    dialog._out.setText(str(built.experiment.dir / "out"))
    dialog._estimate = None
    dialog._export()
    waits.until(qapp, lambda: shown, "the refused export's message")

    assert shown == ["cannot create the destination"]
    assert {name: box.isEnabled() for name, box in boxes.items()} == before
    dialog.deleteLater()
    qapp.processEvents()


# --- the Export dialog: Source video ticked with no video to pack ---------------- #
def _bundle_dialog(qapp, proj, monkeypatch):
    """The Export dialog on the experiment's bundle, Source video ticked, its note asked for
    now and waited on (the answer comes off the UI thread)."""
    from detail.api import exporters
    from detail.api.exporters import ExportSubject
    from detail.gui import export as gui_export
    from tests import waits

    answered = []
    real = gui_export.ExportDialog._on_video_note
    monkeypatch.setattr(
        gui_export.ExportDialog,
        "_on_video_note",
        lambda self, g, note: (answered.append(note), real(self, g, note)),
    )
    subject = ExportSubject("experiment", proj.dir)
    dialog = gui_export.ExportDialog(
        subject, exporters.targets_for(subject), None, only=("bundle",)
    )
    dialog._bundle_video.setChecked(True)
    dialog._estimate_timer.stop()
    dialog._start_estimate()
    waits.until(qapp, lambda: answered, "the bundle's video note")
    return dialog, answered


def test_export_dialog_says_when_source_video_would_pack_none(
    qapp, tmp_path, monkeypatch
):
    """`experiment export --include-video` prints why it packed no video (#29); the dialog
    says it, in the same words, while Source video is ticked."""
    proj = _experiment(tmp_path)
    (copied,) = (proj.dir / "recordings").rglob("*.mp4")
    copied.unlink()  # what an experiment imported without its video holds

    dialog, answered = _bundle_dialog(qapp, proj, monkeypatch)

    assert not dialog._bundle_video_note.isHidden()
    assert dialog._bundle_video_note.text() == (
        "No video will be packed: 1 recording(s) have no video on this machine (put it "
        "back with Locate video, or `detail recording relink`)."
    )


def test_export_dialog_says_nothing_when_source_video_has_video_to_pack(
    qapp, tmp_path, monkeypatch
):
    proj = _experiment(tmp_path)

    dialog, answered = _bundle_dialog(qapp, proj, monkeypatch)

    assert answered == [None]
    assert dialog._bundle_video_note.isHidden()
