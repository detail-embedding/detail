"""The *Meta-information…* button on an analysis run's page, and the window it opens.

What is guarded:

- **Where the button is**: under the run's details block and above **Results**, on any analysis
  run, since the page is shared by every module (asked of a run that shows Results, and of one
  that does not).
- **What the window shows**: one tab per file, named for it, holding the file's text as on disk.
- **That Export writes what is shown**, off the UI thread, and says where.
"""

from __future__ import annotations

import json
import zipfile

from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QPushButton, QTabWidget

from detail.api import run_info as api_run_info
from detail.gui import run_info as gui_run_info
from detail.gui.pages import AnalysisRunDetailPage
from tests.api.test_run_info import a_run
from tests.gui.conftest import drain_until

BUTTON = "Meta-information…"


def _index(page, predicate) -> int:
    for i, widget in enumerate(page._body_widgets):
        if predicate(widget):
            return i
    raise AssertionError("not on the page")


def _is_details(widget) -> bool:
    return any(label.text() == "Module:" for label in widget.findChildren(QLabel))


def _is_button(widget) -> bool:
    return any(b.text() == BUTTON for b in widget.findChildren(QPushButton)) or (
        isinstance(widget, QPushButton) and widget.text() == BUTTON
    )


def _opened(monkeypatch) -> list:
    opened = []
    monkeypatch.setattr(
        gui_run_info.RunInfoWindow, "exec", lambda self: opened.append(self)
    )
    return opened


def test_the_button_sits_under_the_details_and_above_results(qapp, authored_cohort):
    run_dir = authored_cohort.run_dir()
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "job": {"job_id": "an_01", "config": {}},
                "progress": None,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    page = AnalysisRunDetailPage(authored_cohort.experiment, "an_01")

    details = _index(page, _is_details)
    button = _index(page, _is_button)
    results = _index(page, lambda w: isinstance(w, QLabel) and w.text() == "Results")

    assert details < button < results


def test_the_button_is_its_own_size_not_the_pages_width(qapp, tmp_path):
    """Reported: the button ran the full width of the page. It is as wide as its words."""
    page = AnalysisRunDetailPage(a_run(tmp_path), "an_01")
    page.resize(1200, 600)
    page.show()
    drain_until(qapp, lambda: False, timeout_ms=200)
    button = next(b for b in page.findChildren(QPushButton) if b.text() == BUTTON)

    assert button.width() == button.sizeHint().width()
    assert button.width() < page.width() // 3


def test_the_button_opens_every_file_of_the_run(qapp, tmp_path, monkeypatch):
    experiment = a_run(tmp_path)
    opened = _opened(monkeypatch)
    page = AnalysisRunDetailPage(experiment, "an_01")
    button = next(b for b in page.findChildren(QPushButton) if b.text() == BUTTON)

    button.click()

    assert drain_until(qapp, lambda: opened)
    window = opened[0]
    tabs = window.findChild(QTabWidget)
    info = experiment.analysis_info("an_01")
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        f.name for f in info.files
    ]
    for i, f in enumerate(info.files):
        assert tabs.widget(i).toPlainText() == f.text
        assert tabs.widget(i).isReadOnly()


def test_export_writes_the_files_shown_and_says_where(qapp, tmp_path):
    experiment = a_run(tmp_path)
    info = experiment.analysis_info("an_01")
    window = gui_run_info.RunInfoWindow(info)

    window.export_to(tmp_path / "meta")

    out = tmp_path / "meta.zip"
    assert drain_until(qapp, lambda: "Exported to" in window._status.text())
    assert str(out) in window._status.text()
    with zipfile.ZipFile(out) as archive:
        assert archive.namelist() == [f"an_01/{f.name}" for f in info.files]


def test_a_long_file_scrolls_in_its_tab_and_every_tab_name_is_in_view(qapp):
    """Rendered on an_17: a 230-line ``analysis.yaml`` sized the text to the whole file, so the
    window scrolled and took the tabs with it, and the window, as wide as the widest file, hid
    the last two tab names behind arrows."""
    names = [f"file_with_a_long_name_{i}.yaml" for i in range(5)]
    long_text = "".join(f"- recording_{i:03d}\n" for i in range(230))
    window = gui_run_info.RunInfoWindow(
        api_run_info.RunInfo(
            "an_17", tuple(api_run_info.InfoFile(n, long_text) for n in names)
        )
    )
    window.show()
    drain_until(qapp, lambda: False, timeout_ms=200)
    tabs = window.findChild(QTabWidget)
    bar = tabs.tabBar()

    assert bar.width() >= bar.sizeHint().width()
    assert tabs.widget(0).verticalScrollBar().maximum() > 0, "the file scrolls"
    assert window._scroll.verticalScrollBar().maximum() == 0, "the window does not"


def test_a_run_with_no_files_says_so_and_offers_no_export(qapp):
    window = gui_run_info.RunInfoWindow(api_run_info.RunInfo("an_05", ()))
    export = next(b for b in window.findChildren(QPushButton) if b.text() == "Export…")
    assert not export.isEnabled()
    assert not window.findChildren(QPlainTextEdit)
    assert any(
        label.text() == "an_05 keeps no YAML files."
        for label in window.findChildren(QLabel)
    )
