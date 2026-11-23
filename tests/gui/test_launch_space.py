"""A launch dialog's config area is big enough to be read, not a porthole.

**The class.** *A list that keeps a constant hint whatever it holds, starving the pane beside it.*
`QListWidget.sizeHint()` is ~192px regardless of row count — the same constant the table version
of this class records — so a launch dialog listing **one** recording held 192px of a 520px window
and left its config form a **117px** viewport onto 444px of content.

**Why nothing failed.** The form is inside a `PageScroll`, so it *did* scroll: the scrollbar was
there, the range was right, every control reachable. But a 117px porthole sits exactly beside the
Behaviors table, so the scrollbar reads as *that table's* and the dialog looks like it ends. It
was reported as "windows open with no scroll bar, hiding content" — the scrollbar was there and
the report was correct anyway.

**Why a helper check would not have caught it.** `sizing.fit_list_height` existed, `comparison.py`
called it four times, and `LaunchRunDialog` had a *private copy* of it (`_fit_recordings_height`,
using `setFixedHeight` — a floor and a ceiling at once). `LaunchAnalysisDialog` had neither. A
test that the helper works passes; a test that some dialog calls it passes. So this asserts the
**viewport the user actually gets**, on both dialogs, at their own opening size.
"""

import pytest

from detail.gui.scroll import PageScroll

from tests.gui import subjects

pytestmark = pytest.mark.guard

#: Measured: with the list capped, one recording leaves 285px. Below ~200 the form shows a table
#: and nothing else, which is the state that was reported. Not a style number — the point at
#: which the pane stops being readable as a form.
MIN_CONFIG_VIEWPORT = 200


def _dialogs(experiment):
    from detail.gui.launch import LaunchAnalysisDialog, LaunchRunDialog

    return [LaunchRunDialog(experiment), LaunchAnalysisDialog(experiment)]


@pytest.mark.parametrize("index", [0, 1], ids=["run", "analysis"])
def test_the_config_area_is_not_a_porthole(qapp, tmp_path, index):
    experiment = subjects.an_experiment(tmp_path)
    dialog = _dialogs(experiment)[index]
    dialog.show()
    qapp.processEvents()
    qapp.processEvents()
    try:
        area = dialog.findChild(PageScroll)
        assert area is not None, "the config form is not in a scroll area at all"
        viewport = area.viewport().height()
        assert viewport >= MIN_CONFIG_VIEWPORT, (
            f"the config area opens at {viewport}px of a {dialog.height()}px dialog — the "
            f"recordings list is {dialog._recordings.height()}px for "
            f"{dialog._recordings.count()} row(s)"
        )
    finally:
        dialog.close()


@pytest.mark.parametrize("index", [0, 1], ids=["run", "analysis"])
def test_the_recordings_list_is_capped_to_its_rows(qapp, tmp_path, index):
    """A one-row list must not hold a five-row box.

    Asserts the height against the row height rather than against a constant, so it stays true
    on a machine whose font makes rows taller.
    """
    experiment = subjects.an_experiment(tmp_path)
    dialog = _dialogs(experiment)[index]
    dialog.show()
    qapp.processEvents()
    try:
        listing = dialog._recordings
        count = listing.count()
        assert count, "no recordings listed, so this asserts nothing"
        row = listing.sizeHintForRow(0)
        assert (
            listing.height() <= row * min(count, 8) + 24
        ), f"{count} row(s) of {row}px are held in {listing.height()}px"
    finally:
        dialog.close()


def test_the_cap_would_be_missed_without_it(qapp, tmp_path):
    """Guard the guard: an uncapped list must fail the check above.

    Without this the two tests pass on a build where the helper silently stopped working — the
    number they assert is only meaningful if the unfixed shape is known to breach it.
    """
    from detail.gui.qt import QListWidget

    listing = QListWidget()
    listing.addItem("one recording")
    listing.show()
    qapp.processEvents()
    try:
        row = listing.sizeHintForRow(0)
        assert (
            listing.sizeHint().height() > row * 1 + 24
        ), "an uncapped one-row QListWidget no longer over-reports, so the cap is untested"
    finally:
        listing.close()
