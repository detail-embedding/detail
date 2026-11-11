"""A run's results page fits the window the app is designed for, without scrolling.

**Reported.** *"The results pages for detection, tracking and embedding are still difficult to
navigate. I don't think these pages ever need to be scrollable. The video player and the buttons
below should be locked to the bottom of the screen and the video player (or embedding graph,
output) should be rescaled if the window is smaller."*

**The class: a page whose parts each insisted on a size, so the page could only grow.** The
run-detail page carried six rows of run metadata with a fixed-height config box, a table sized to
the *display's* row budget, a result view that opened with its own recording dropdown, and a
picture with a 480x360 floor. Every one of those was reasonable alone; together they asked for
more height than the window has, so the shell scrolled and the transport under the video sat
below the bottom edge — the control you need while watching, reachable only by scrolling away
from the thing you are watching. Measured at the design size: 893, 940 and 930px of page in a
643px content area.

Three halves to the fix, and this measures the result of all of them: the list and the results
share one pane (`pages.RunResults`), the run's details fold behind a line that says what the run
is, and the picture's minimum became a *floor* rather than a size — so the results space is what
absorbs a resize.

**Measured through the real shell**, not through a `PageScroll` built in the test: the content
area is what the window is left with after the rail, the banner and the mode bar, and only the
shell knows how much that is. It is also the number that moves when any of those change, which
is exactly when this should start failing. The window is the 1440x810 target
`test_dialog_space.py` pins; the content area under it is measured, not assumed, so this reads
the same on a machine whose fonts are bigger.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QWidget  # noqa: E402

from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers module UIs)
from detail.gui.app import MainWindow  # noqa: E402
from detail.gui.pages import RunDetailPage, RunResults  # noqa: E402
from detail.gui.scroll import PageScroll  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests import waits  # noqa: E402
from tests.gui import subjects  # noqa: E402

pytestmark = pytest.mark.guard

#: The window this app is designed for — `test_dialog_space.py`'s target, where it guards the
#: size a window *opens* at. This guards what a page must fit inside one.
TARGET_W, TARGET_H = 1440, 810


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """One experiment with a completed run of every stage, on real on-disk outputs."""
    return subjects.an_experiment_with_completed_runs(tmp_path_factory.mktemp("fits"))


def _enclosing_scroll(widget: QWidget) -> PageScroll:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, PageScroll):
            return parent
        parent = parent.parentWidget()
    raise AssertionError("the shell did not put this page in a PageScroll")


@pytest.fixture
def shell(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()
    window.resize(TARGET_W, TARGET_H)
    window.show()
    qapp.processEvents()
    yield window
    window.close()


def _open_run(qapp, shell, experiment, module_id: str):
    """Open the run page the way the shell does, and wait for its results to arrive.

    With its Notes band folded: open by default, it costs the video 65px at this size, and the
    user chose (2026-11-02) a box that is there unless folded over a page that fits unfolded.
    Folding is how this window gets the room back, and that state is what has to fit.
    """
    record = next(r for r in experiment.processing_runs if r["module_id"] == module_id)
    page = RunDetailPage(experiment, record["id"])
    shell._show(page, f"Run {record['id']}")
    # The results open off the UI thread, and it is the *loaded* page that has to fit: an empty
    # one always does. Wait for the picture or the scatter before measuring.
    waits.until(qapp, lambda: not _still_loading(page), f"the {module_id} results")
    for _ in range(3):
        qapp.processEvents()
    return page


def _still_loading(page) -> bool:
    from detail.gui.run_preview import DetectionRunPreview, EmbeddingRunPreview

    for preview in page.findChildren(DetectionRunPreview):
        if not hasattr(preview, "_scrubber"):
            return True
    for preview in page.findChildren(EmbeddingRunPreview):
        if preview._preview is None:
            return True
    return False


@pytest.mark.parametrize("module_id", ["detection", "tracking", "embedding"])
def test_a_run_page_fits_the_window_without_scrolling(
    qapp, shell, experiment, module_id
):
    page = _open_run(qapp, shell, experiment, module_id)
    scroll = _enclosing_scroll(page)
    overflow = scroll.verticalScrollBar().maximum()
    assert overflow == 0, (
        f"the {module_id} results page is {page.height()}px in the "
        f"{scroll.viewport().height()}px content area of a {TARGET_W}x{TARGET_H} window, so "
        f"{overflow}px of it — the transport under the video, and the run's own buttons — is "
        "reachable only by scrolling"
    )


@pytest.mark.parametrize("module_id", ["detection", "tracking", "embedding"])
def test_the_results_take_the_room_the_rest_of_the_page_does_not(
    qapp, shell, experiment, module_id
):
    """And the room goes to the results, not to a spacer under them.

    A page that fits can still waste the window: the trailing stretch every page ends with
    would take the slack and leave the video at its floor with empty page beneath it. The
    results pane declares itself expanding for exactly this, so it is what grows — measured at
    42% of the page with the spacer left in place at stretch 0, which is how that was found.
    """
    page = _open_run(qapp, shell, experiment, module_id)
    results = page.findChild(RunResults)
    assert results is not None
    share = results.height() / page.height()
    assert share > 0.55, (
        f"the results are {results.height()}px of a {page.height()}px page ({share:.0%}) — "
        "the rest of the window went to a spacer rather than to the video or the scatter"
    )


def test_the_run_details_start_folded(qapp, shell, experiment):
    """The line above them says what the run is; the rest is read once.

    Not a style choice — those six rows are 188px of a 643px content area, and this is the page
    that has to hold a video *and* its transport *and* the run's buttons.
    """
    from detail.gui.collapse import CollapsibleBand

    page = _open_run(qapp, shell, experiment, "tracking")
    band = page.findChild(CollapsibleBand)
    assert band is not None, "the run's details are not in a band that can be folded"
    assert not band.is_open()
    band.set_open(True)  # and they are still there, one click away
    qapp.processEvents()
    from PyQt6.QtWidgets import QLabel

    shown = [label.text() for label in band.findChildren(QLabel) if label.isVisible()]
    assert any("Config" in text for text in shown), shown
    assert any("From" in text or "Created" in text for text in shown), shown
