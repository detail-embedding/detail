"""`ExportRunDetailPage` against a real, completed export run (#475).

`tests/gui/test_smoke_construct.py` only ever builds this page against a non-existent run id,
which hits the "(record not found)" branch and never reaches `_render()` — not the "Wrote to"
label, not the outcomes table, not the "Open folder" button. So the two things a user actually
looks at on this page (where did my files go, and can I get to them) had no test at all: a
record/folder desync (`_unique_dir` renaming the destination `write_export` actually used, never
written back) and a discarded `QDesktopServices.openUrl` return value both shipped invisibly.
"""

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel, QPushButton  # noqa: E402

from detail.api import export_targets  # noqa: E402,F401  (registers the catalogue)
from detail.gui import open_external  # noqa: E402
from detail.gui import pages as gui_pages  # noqa: E402


def _run(cohort, targets, out, **kwargs):
    handle = cohort.experiment.run_export(
        "analysis_run", "an_01", targets, out, **kwargs
    )
    return handle.wait()


class _Child:
    """A spawned launcher that has already exited."""

    def wait(self, timeout=None):
        return 1


def _open_folder_button(page):
    return next(
        (b for b in page.findChildren(QPushButton) if b.text() == "Open folder"), None
    )


def test_the_page_shows_where_the_export_really_wrote_and_offers_open_folder(
    qapp, authored_cohort, tmp_path
):
    from detail.api import exporters

    out = tmp_path / "out"
    handle = _run(authored_cohort, ["instances"], out)
    record = authored_cohort.experiment.export_record(handle.id)
    destination = exporters.destination_of(record)
    assert destination.is_dir(), "the fixture itself must have written somewhere real"

    page = gui_pages.ExportRunDetailPage(authored_cohort.experiment, handle.id)

    labels = [w.text() for w in page.findChildren(QLabel)]
    assert any(
        str(destination) in text for text in labels
    ), f"'Wrote to' does not name the real destination: {labels}"

    button = _open_folder_button(page)
    assert (
        button is not None
    ), "a completed export with a real folder must offer Open folder"


def test_open_folder_targets_the_actual_destination(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """The click must reach the OS with the *real* path — not a stale, pre-collision name."""
    from detail.api import exporters

    out = tmp_path / "out"
    handle = _run(authored_cohort, ["instances"], out)
    destination = exporters.destination_of(
        authored_cohort.experiment.export_record(handle.id)
    )

    opened = []
    monkeypatch.setattr(
        open_external.QDesktopServices,
        "openUrl",
        lambda url: opened.append(Path(url.toLocalFile())) or True,
    )

    page = gui_pages.ExportRunDetailPage(authored_cohort.experiment, handle.id)
    _open_folder_button(page).click()

    assert opened == [destination]


def test_open_folder_warns_instead_of_silently_doing_nothing(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """`QDesktopServices.openUrl` returns `False` with no default `file://` handler registered
    (a headless/minimal Linux desktop, a misconfigured `xdg-open`) — the exact shape of "I
    clicked it and nothing happened." The click must say so rather than swallow it.
    """
    from PyQt6.QtWidgets import QMessageBox

    out = tmp_path / "out"
    handle = _run(authored_cohort, ["instances"], out)

    monkeypatch.setattr(open_external.QDesktopServices, "openUrl", lambda url: False)
    # Pinned to a non-WSL host: on a WSL dev box the fallback would launch a real Explorer.
    monkeypatch.setattr(
        open_external.api_checks, "wsl_explorer_argv", lambda path: None
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warned.append(a) or None
    )

    page = gui_pages.ExportRunDetailPage(authored_cohort.experiment, handle.id)
    _open_folder_button(page).click()

    assert warned, "a failed openUrl must not be a silent no-op"


def test_open_folder_falls_back_to_explorer_inside_wsl(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """A WSL distro has no `file://` launcher, so `openUrl` is always False there. The click
    must hand the folder to Windows Explorer instead of only saying it could not."""
    from PyQt6.QtWidgets import QMessageBox

    handle = _run(authored_cohort, ["instances"], tmp_path / "out")
    monkeypatch.setattr(open_external.QDesktopServices, "openUrl", lambda url: False)
    argv = ["explorer.exe", "\\\\wsl$\\Ubuntu\\out"]
    monkeypatch.setattr(
        open_external.api_checks, "wsl_explorer_argv", lambda path: argv
    )
    ran = []
    monkeypatch.setattr(
        open_external.subprocess, "Popen", lambda a, **k: ran.append(a) or _Child()
    )
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warned.append(a) or None
    )

    page = gui_pages.ExportRunDetailPage(authored_cohort.experiment, handle.id)
    _open_folder_button(page).click()

    assert ran == [argv]
    assert not warned


def test_open_folder_still_warns_when_the_explorer_fallback_fails(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    handle = _run(authored_cohort, ["instances"], tmp_path / "out")
    monkeypatch.setattr(open_external.QDesktopServices, "openUrl", lambda url: False)
    monkeypatch.setattr(
        open_external.api_checks,
        "wsl_explorer_argv",
        lambda path: ["explorer.exe", "x"],
    )

    def broken(argv, **kwargs):
        raise FileNotFoundError("explorer.exe")

    monkeypatch.setattr(open_external.subprocess, "Popen", broken)
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: warned.append(a) or None
    )

    page = gui_pages.ExportRunDetailPage(authored_cohort.experiment, handle.id)
    _open_folder_button(page).click()

    assert warned


def test_a_renamed_folder_is_what_the_page_actually_shows(
    qapp, authored_cohort, tmp_path, monkeypatch
):
    """The GUI-facing half of the record/folder desync guarded at the API level in
    `tests/api/test_export_run.py::test_a_renamed_folder_is_reflected_in_the_record`.
    """
    from detail.api import exporters

    out = tmp_path / "out"
    monkeypatch.setattr(exporters, "folder_name", lambda subject, when=None: "collide")
    _run(authored_cohort, ["instances"], out)
    second = _run(authored_cohort, ["instances"], out)

    page = gui_pages.ExportRunDetailPage(authored_cohort.experiment, second.id)
    labels = [w.text() for w in page.findChildren(QLabel)]
    assert any(
        str(out / "collide_2") in text for text in labels
    ), f"the page still shows the pre-collision folder: {labels}"
    assert _open_folder_button(page) is not None
