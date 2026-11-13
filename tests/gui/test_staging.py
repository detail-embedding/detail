"""The background-uploads dialog (``gui/staging.py``): watch, stop and resume a distill's video upload.

**The class guarded: a feature only the CLI can see.** ``Distill`` with *Upload videos in the
background* starts a process that outlives the app and copies a library for an hour; the only way
to watch or stop it was ``detail remote staging-status`` / ``staging-stop``, so a GUI user saw
nothing happening and had no way to stop 100+ GB of copying. These tests drive the dialog through
the real ``Experiment.video_staging`` / ``stop_video_staging`` over real state files, on the real
worker thread, rather than handing the dialog a canned list: the thread, the liveness read and the
stop are where this goes wrong.

Two properties are asserted rather than assumed. **Both calls run off the UI thread**: the liveness
check shells out and the stop kills a process tree, and a slow read in a poll every two seconds is
the frozen-window class (`tests/gui/test_ui_thread_blocking.py`). **A table with no rows is a claim**,
so it stays hidden until the first read lands.
"""

import threading

import pytest

from detail.api.experiment import Experiment
from detail.executor import local, video_staging
from detail.gui import staging
from detail.gui.tables import row_id
from tests.gui.conftest import drain_until


def _state(root, target, **over):
    state = video_staging._initial_state(target, ["recording_001"])
    state.update(pid=4242, files_done=3, files_total=8, bytes_total=116_000_000_000)
    state.update(over)
    video_staging._write_state(root, target, state)
    return state


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "coordinator_alive", lambda *_a: True)
    return Experiment.create("p", tmp_path)


def _cells(dialog):
    model = dialog._table.view.model()
    return [
        [model.index(r, c).data() for c in range(model.columnCount())]
        for r in range(model.rowCount())
    ]


def _read(qapp, dialog):
    dialog.refresh()
    assert drain_until(qapp, lambda: not dialog._reading), "the read never landed"


def test_rows_carry_the_uploads_own_numbers():
    ids, rows = staging.staging_rows(
        [
            {
                "target": "tillicum",
                "status": "running",
                "files_done": 3,
                "files_total": 8,
                "bytes_done": 12_000_000_000,
                "bytes_current": 500_000_000,
                "bytes_total": 116_000_000_000,
                "eta_s": 3600,
                "message": "uploading videos · 3/8 files",
                "error": None,
            },
            {
                "target": "hyak",
                "status": "failed",
                "files_done": 0,
                "files_total": 8,
                "bytes_done": 0,
                "bytes_total": 116_000_000_000,
                "eta_s": None,
                "message": "failed",
                "error": "RuntimeError: disk quota exceeded",
            },
        ]
    )

    assert ids == ["tillicum", "hyak"]
    assert rows[0] == (
        "tillicum",
        "running",
        "3/8",
        "12.5/116.0 GB",  # the file in flight counts, so it moves through one big file
        "1h00m",
        "uploading videos · 3/8 files",
    )
    assert rows[1][4:] == (
        "—",
        "RuntimeError: disk quota exceeded",
    ), "no ETA; error wins"


def test_the_table_stays_hidden_until_the_first_read_lands(qapp, experiment):
    """Rendered early, "(no background uploads…)" tells someone with an upload running that
    there is none."""
    dialog = staging.BackgroundUploadsDialog(experiment)

    assert dialog._table.isHidden()
    _read(qapp, dialog)
    assert not dialog._table.isHidden()


def test_with_no_uploads_it_says_how_to_start_one(qapp, experiment):
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)

    assert _cells(dialog) == []
    assert "Upload videos in the background" in dialog._table._notice.text()


def test_it_shows_the_recorded_uploads_read_through_the_api(qapp, experiment):
    _state(experiment.dir, "tillicum", bytes_done=12_000_000_000, eta_s=3600.0)
    dialog = staging.BackgroundUploadsDialog(experiment)

    _read(qapp, dialog)

    (row,) = _cells(dialog)
    assert row[:5] == ["tillicum", "running", "3/8", "12.0/116.0 GB", "1h00m"]
    assert dialog._caption.text() == "1 upload(s), 1 running"


def test_both_calls_run_off_the_ui_thread(qapp, experiment, monkeypatch):
    """`video_staging` asks the OS whether each process is alive and `stop_video_staging` kills a
    process tree: a poll of either on the UI thread, every two seconds, freezes the window.
    """
    _state(experiment.dir, "tillicum")
    threads = {}
    real_read, real_stop = experiment.video_staging, experiment.stop_video_staging
    monkeypatch.setattr(
        experiment,
        "video_staging",
        lambda: threads.setdefault("read", threading.current_thread()) and real_read(),
    )
    monkeypatch.setattr(
        experiment,
        "stop_video_staging",
        lambda t=None: threads.setdefault("stop", threading.current_thread())
        and real_stop(t),
    )
    monkeypatch.setattr(local, "kill_coordinator", lambda *_a: None)
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    dialog._table.view.selectRow(0)
    dialog._stop.click()
    assert drain_until(qapp, lambda: "stop" in threads and not dialog._reading)

    assert threads["read"] is not threading.main_thread()
    assert threads["stop"] is not threading.main_thread()


def test_stop_is_offered_only_for_a_running_upload(qapp, experiment):
    _state(experiment.dir, "tillicum")
    _state(experiment.dir, "hyak", status="complete")
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    assert not dialog._stop.isEnabled(), "nothing selected"

    for target, expected in (("tillicum", True), ("hyak", False)):
        row = next(r for r, (t, *_rest) in enumerate(_cells(dialog)) if t == target)
        dialog._table.view.selectRow(row)
        assert dialog._stop.isEnabled() is expected, target


def test_stop_stops_that_upload_and_refreshes(qapp, experiment, monkeypatch):
    _state(experiment.dir, "tillicum")
    killed = []
    monkeypatch.setattr(local, "kill_coordinator", lambda root, pid: killed.append(pid))
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    dialog._table.view.selectRow(0)

    dialog._stop.click()

    assert drain_until(
        qapp, lambda: _cells(dialog) and _cells(dialog)[0][1] == "cancelled"
    )
    assert killed == [4242]
    assert not dialog._stop.isEnabled(), "a stopped upload cannot be stopped again"


def test_a_poll_does_not_take_the_selection_from_someone_about_to_stop(
    qapp, experiment
):
    _state(experiment.dir, "tillicum")
    _state(experiment.dir, "hyak")
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    hyak = next(r for r, (t, *_rest) in enumerate(_cells(dialog)) if t == "hyak")
    dialog._table.view.selectRow(hyak)

    _read(qapp, dialog)  # what the two-second timer does

    assert dialog._selected_target() == "hyak"
    assert dialog._stop.isEnabled()


def test_a_failed_read_says_so_and_does_not_stop_polling(qapp, experiment, monkeypatch):
    def broken():
        raise OSError("share unavailable")

    monkeypatch.setattr(experiment, "video_staging", broken)
    dialog = staging.BackgroundUploadsDialog(experiment)

    _read(qapp, dialog)

    assert "share unavailable" in dialog._caption.text()
    assert not dialog._reading, "a failed read must not wedge every later poll"


def test_it_polls_while_shown_and_stops_polling_when_hidden(qapp, experiment):
    dialog = staging.BackgroundUploadsDialog(experiment)
    assert not dialog._timer.isActive()

    dialog.show()
    assert dialog._timer.isActive() and dialog._timer.interval() == staging.POLL_MS
    assert drain_until(qapp, lambda: not dialog._reading)
    dialog.hide()

    assert not dialog._timer.isActive(), "a closed dialog is not polling"


def _select(dialog, target):
    row = next(r for r, (t, *_rest) in enumerate(_cells(dialog)) if t == target)
    dialog._table.view.selectRow(row)


def test_resume_is_offered_for_an_upload_that_is_not_running_and_stop_for_one_that_is(
    qapp, experiment
):
    """Exactly one of the two is live for any row: stopping a stopped upload and resuming a
    running one are both no-ops the buttons should not invite."""
    _state(experiment.dir, "tillicum")
    _state(experiment.dir, "hyak", status="cancelled")
    _state(experiment.dir, "ember", status="failed")
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    assert not dialog._resume.isEnabled(), "nothing selected"

    for target, stop, resume in (
        ("tillicum", True, False),
        ("hyak", False, True),
        ("ember", False, True),
    ):
        _select(dialog, target)
        assert (dialog._stop.isEnabled(), dialog._resume.isEnabled()) == (
            stop,
            resume,
        ), target


def test_resume_restarts_that_upload_off_the_ui_thread_and_refreshes(
    qapp, experiment, monkeypatch
):
    _state(experiment.dir, "hyak", status="cancelled")
    started = []

    def fake_start(target, recording_ids=None):
        started.append((target, threading.current_thread()))
        return _state(experiment.dir, target, status="running")

    monkeypatch.setattr(experiment, "start_video_staging", fake_start)
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    _select(dialog, "hyak")

    dialog._resume.click()

    assert drain_until(
        qapp, lambda: _cells(dialog) and _cells(dialog)[0][1] == "running"
    )
    assert [t for t, _ in started] == ["hyak"]
    assert started[0][1] is not threading.main_thread()
    assert dialog._stop.isEnabled() and not dialog._resume.isEnabled()


def test_a_resume_that_cannot_start_says_why_and_keeps_the_row(
    qapp, experiment, monkeypatch
):
    _state(experiment.dir, "hyak", status="failed")

    def refuse(target, recording_ids=None):
        raise ValueError("no remote target registered as 'hyak'")

    monkeypatch.setattr(experiment, "start_video_staging", refuse)
    dialog = staging.BackgroundUploadsDialog(experiment)
    _read(qapp, dialog)
    _select(dialog, "hyak")

    dialog._resume.click()

    assert drain_until(qapp, lambda: "Could not resume" in dialog._caption.text())
    assert "no remote target registered" in dialog._caption.text()
    assert dialog._resume.isEnabled(), "the person can try again"


# --- a distill run's own page: the one-line status and the button ------------------------ #
def _run_record(experiment, *, config=None, status="running"):
    from detail.storage import distill_runs as distill_store

    record = {
        "id": "distill_001",
        "name": "m",
        "status": status,
        "stages": {
            "label": {"status": "complete", "dataset_id": "dataset_001"},
            "train": {"status": "pending", "model_id": None},
        },
    }
    if config is not None:
        record["config"] = config
    distill_store.write_run(experiment.dir, "distill_001", record)


def _run_page(experiment):
    from detail.gui import pages

    return pages.DistillRunDetail(experiment, "distill_001")


def _upload_label(page):
    labels = page.findChildren(staging.UploadStatusLabel)
    return labels[0] if labels else None


def _upload_button(page):
    return next(
        (
            b
            for b in page.findChildren(staging.QPushButton)
            if b.text() == "Background uploads…"
        ),
        None,
    )


_STAGED = {"stage_videos": True, "stage_video_target": "tillicum"}


@pytest.mark.parametrize(
    "state, expected",
    [
        (None, "not started yet (to tillicum)"),
        (
            dict(
                status="running",
                files_done=12,
                files_total=80,
                bytes_done=17_000_000_000,
                bytes_current=2_100_000_000,
                bytes_total=116_000_000_000,
                eta_s=2700.0,
            ),
            "uploading to tillicum · 12/80 files · 19.1/116.0 GB · ~45m00s left",
        ),
        (
            dict(
                status="running",
                files_done=0,
                files_total=80,
                bytes_done=0,
                bytes_total=116_000_000_000,
                eta_s=None,
            ),
            "uploading to tillicum · 0/80 files · 0.0/116.0 GB",
        ),
        (
            dict(
                status="complete",
                files_done=80,
                files_total=80,
                bytes_done=116_000_000_000,
                bytes_total=116_000_000_000,
            ),
            "on tillicum · 80 files, 116.0 GB",
        ),
        (
            dict(
                status="failed",
                files_done=3,
                files_total=80,
                bytes_done=0,
                bytes_total=1,
                error="RuntimeError: disk quota exceeded",
            ),
            "upload to tillicum failed · RuntimeError: disk quota exceeded"
            " — resume it from Background uploads…",
        ),
        (
            dict(
                status="cancelled",
                files_done=3,
                files_total=80,
                bytes_done=0,
                bytes_total=1,
            ),
            "upload to tillicum stopped at 3/80 files"
            " — resume it from Background uploads…",
        ),
    ],
)
def test_the_summary_says_where_the_videos_are_in_the_uploads_own_numbers(
    state, expected
):
    assert staging.staging_summary("tillicum", state) == expected


def test_a_staged_distill_shows_its_upload_on_its_page(qapp, experiment):
    _run_record(experiment, config=_STAGED)
    _state(
        experiment.dir,
        "tillicum",
        files_done=12,
        files_total=80,
        bytes_done=19_100_000_000,
        eta_s=2700.0,
    )
    page = _run_page(experiment)
    label = _upload_label(page)

    assert drain_until(qapp, lambda: not label._reading)

    assert label.text() == (
        "uploading to tillicum · 12/80 files · 19.1/116.0 GB · ~45m00s left"
    )
    assert _upload_button(page) is not None


def test_a_distill_without_the_option_shows_nothing_about_uploads(qapp, experiment):
    """No row and no button for a run that never asked, nor for one recorded before the option
    existed (a record with no `config` at all, which is every run on disk today)."""
    for config in (None, {"stage_videos": False, "stage_video_target": None}):
        _run_record(experiment, config=config)
        page = _run_page(experiment)

        assert _upload_label(page) is None
        assert _upload_button(page) is None


def test_the_line_reads_off_the_ui_thread_and_follows_the_upload(
    qapp, experiment, monkeypatch
):
    """The upload can be stopped, failed or resumed from the dialog while this page is open, so
    the line keeps reading, and the read asks the OS about a process: never the UI thread.
    """
    _run_record(experiment, config=_STAGED)
    _state(experiment.dir, "tillicum")
    threads = []
    real = experiment.video_staging
    monkeypatch.setattr(
        experiment,
        "video_staging",
        lambda: threads.append(threading.current_thread()) or real(),
    )
    page = _run_page(experiment)
    label = _upload_label(page)
    assert drain_until(qapp, lambda: not label._reading)
    assert label.text().startswith("uploading to tillicum")
    assert label._timer.isActive() and label._timer.interval() == staging.LABEL_POLL_MS

    _state(experiment.dir, "tillicum", status="cancelled")
    label.refresh()
    assert drain_until(
        qapp, lambda: label.text().startswith("upload to tillicum stopped")
    )

    assert threads and all(t is not threading.main_thread() for t in threads)


def test_the_line_says_so_when_it_cannot_read_the_upload(qapp, experiment, monkeypatch):
    _run_record(experiment, config=_STAGED)

    def broken():
        raise OSError("share unavailable")

    monkeypatch.setattr(experiment, "video_staging", broken)
    page = _run_page(experiment)
    label = _upload_label(page)

    assert drain_until(qapp, lambda: "share unavailable" in label.text())
    assert not label._reading, "a failed read must not wedge every later one"


def test_the_runs_button_opens_the_dialog_on_the_same_experiment(
    qapp, experiment, monkeypatch
):
    _run_record(experiment, config=_STAGED)
    opened = []
    monkeypatch.setattr(
        staging.BackgroundUploadsDialog,
        "exec",
        lambda self: opened.append(self._experiment),
    )
    page = _run_page(experiment)

    _upload_button(page).click()

    assert opened == [experiment]


def test_the_models_page_opens_it(qapp, experiment, monkeypatch):
    from detail.gui import pages

    opened = []
    monkeypatch.setattr(
        staging.BackgroundUploadsDialog,
        "exec",
        lambda self: opened.append(self._experiment),
    )
    page = pages.ModelsPage(experiment, on_open=lambda _id: None)
    button = next(
        b
        for b in page.findChildren(pages.QPushButton)
        if "Background uploads" in b.text()
    )

    button.click()

    assert opened == [experiment]
    assert (
        row_id is not None
    )  # the table helper the dialog selects through is importable
