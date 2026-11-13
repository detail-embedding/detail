"""A recording picker shows what a recording is, and filters on it (issue #663).

**The class: a picker that asks for the one thing a person cannot choose by.** Every list of
recordings showed the recording's name and nothing else, and in a cohort of eighty a name like
``2024-09-27_12-41-09_BOX1_PM_9_DrugOn_S20`` says when and where, not the strain, dose or brain
state the run is being made for. The columns were already in the experiment (Recording
information) and one page away from every picker.

The second half of the class is what a filter can do wrong once it exists, and these tests pin
it: **a filter hides rows, it never changes the selection.** A recording ticked and then
filtered out is still ticked and still submitted, and the count says so; *Select all* / *Clear*
act on the rows shown only. A filter that dropped or kept a selection silently would shrink or
grow the very set a run is about to use, with a plausible-looking list and nothing failing.

The last test is the sweep: nine pickers were wired in one change, so the guard discovers them
from source rather than listing them, and a picker added later has to attach the filter or say
why not.
"""

import ast
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QAbstractItemView, QListWidget  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import launch  # noqa: E402
from detail.gui.recording_filter import MAX_FILTER_VALUES, RecordingFilter  # noqa: E402
from detail.gui.recording_items import fill_recording_list  # noqa: E402

REAL_ROWS = 16


def _cohort(tmp_path) -> Experiment:
    """Sixteen recordings: two strains x eight, alternating drug on/off, a unique animal id."""
    experiment = Experiment.create("proj", tmp_path)
    animal = 0
    for strain in ("CD1", "C57"):
        for box in range(1, 9):
            animal += 1
            on = animal % 2
            experiment.add_recording(
                f"BOX{box}_{'DrugOn' if on else 'DrugOff'}_{strain}_{animal}",
                {
                    "Animal ID": str(animal),
                    "Strain": strain,
                    "Dose": "10" if on else "9",
                    "Brain_state": "on" if on else "off",
                    "Sex": "",
                },
            )
    return experiment


@pytest.fixture
def cohort(tmp_path):
    return _cohort(tmp_path)


@pytest.fixture
def picker(qapp, cohort):
    """A plain recording list under a filter, ``(list, filter, experiment)``."""
    listwidget = QListWidget()
    fill_recording_list(listwidget, cohort.recordings)
    bar = RecordingFilter(listwidget, cohort)
    qapp.processEvents()
    return listwidget, bar, cohort


def _shown(listwidget) -> list[str]:
    return [
        listwidget.item(row).text()
        for row in range(listwidget.count())
        if not listwidget.isRowHidden(row)
    ]


def _choose(bar: RecordingFilter, column: str, value: str) -> None:
    combo = bar._combos[column]
    combo.setCurrentIndex(combo.findData(value))


def test_every_row_carries_its_recordings_information(picker):
    listwidget, bar, experiment = picker
    first = experiment.recordings[0]

    text = bar.information_text(first.id)
    tooltip = bar.information_tooltip(first.id)

    # in the order of the columns, a dash where the recording has nothing: the reserved
    # acquisition_setup column (always first) and Sex are both blank
    assert text == "– · 1 · CD1 · 10 · on · –"
    assert "Strain: CD1" in tooltip and "Dose: 10" in tooltip
    assert "Sex" not in tooltip  # a blank is not worth a line
    assert bar.information_text("no_such_recording") == ""


def test_text_filter_matches_the_name_and_any_columns_value(picker):
    listwidget, bar, _ = picker

    bar._text.setText("c57")  # in the *name* of the C57 rows and in their Strain column
    assert len(_shown(listwidget)) == REAL_ROWS // 2

    bar._text.setText(
        "drugon cd1"
    )  # every word has to match, in any column or the name
    assert len(_shown(listwidget)) == REAL_ROWS // 4

    bar._text.setText("brain_state")  # a column *name* is not a value: nothing matches
    assert _shown(listwidget) == []

    bar._text.setText("")
    assert len(_shown(listwidget)) == REAL_ROWS


def test_a_column_with_few_values_gets_a_dropdown_and_a_unique_one_does_not(picker):
    _, bar, _ = picker

    assert {"Strain", "Dose", "Brain_state"} <= set(bar._combos)
    assert "Animal ID" not in bar._combos  # 16 values: the list itself
    assert "Sex" not in bar._combos  # one value (blank) has nothing to choose
    assert REAL_ROWS > MAX_FILTER_VALUES


def test_dropdown_values_sort_as_numbers(picker):
    _, bar, _ = picker
    combo = bar._combos["Dose"]

    values = [combo.itemData(i) for i in range(1, combo.count())]

    assert values == ["9", "10"]  # not "10" < "9"


def test_dropdowns_and_the_text_box_narrow_together(picker):
    listwidget, bar, _ = picker

    _choose(bar, "Strain", "CD1")
    assert len(_shown(listwidget)) == REAL_ROWS // 2
    _choose(bar, "Brain_state", "on")
    assert len(_shown(listwidget)) == REAL_ROWS // 4
    bar._text.setText("BOX1")
    assert len(_shown(listwidget)) == 1

    bar.clear_filter()
    assert len(_shown(listwidget)) == REAL_ROWS
    assert not bar.filter_active()


def test_a_row_with_no_recording_is_hidden_by_a_column_filter_not_by_the_text_box(
    qapp, cohort
):
    listwidget = QListWidget()
    fill_recording_list(listwidget, cohort.recordings)
    listwidget.addItem("(not a recording)")
    bar = RecordingFilter(listwidget, cohort)
    qapp.processEvents()

    bar._text.setText("not a")
    assert _shown(listwidget) == ["(not a recording)"]
    bar._text.setText("")
    _choose(bar, "Strain", "CD1")
    assert "(not a recording)" not in _shown(listwidget)


def test_a_list_that_is_rebuilt_is_filtered_again(qapp, picker):
    listwidget, bar, experiment = picker
    _choose(bar, "Strain", "C57")
    assert len(_shown(listwidget)) == REAL_ROWS // 2

    fill_recording_list(
        listwidget, experiment.recordings
    )  # a picker rebuilding its rows
    qapp.processEvents()

    assert len(_shown(listwidget)) == REAL_ROWS // 2
    assert bar._combos["Strain"].currentData() == "C57"  # and the choice survived


def test_select_shown_and_clear_shown_leave_hidden_rows_alone(qapp, cohort):
    listwidget = QListWidget()
    listwidget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    fill_recording_list(listwidget, cohort.recordings)
    bar = RecordingFilter(listwidget, cohort, select_buttons=True)
    qapp.processEvents()

    _choose(bar, "Strain", "CD1")
    bar.select_shown()
    assert len(listwidget.selectedItems()) == REAL_ROWS // 2

    _choose(bar, "Strain", "C57")  # the CD1 rows are hidden now, and still selected
    assert len(listwidget.selectedItems()) == REAL_ROWS // 2
    bar.clear_shown()  # clears the C57 rows it shows (none were selected) ...
    assert (
        len(listwidget.selectedItems()) == REAL_ROWS // 2
    )  # ... not the hidden CD1 ones


# --------------------------------------------------------------------------- #
# the Launch dialog: the screenshot in the issue
# --------------------------------------------------------------------------- #
@pytest.fixture
def run_dialog(qapp, cohort, monkeypatch):
    """The processing Launch dialog over the cohort, with every recording eligible."""
    monkeypatch.setattr(launch, "_eligible", lambda recording, upstream: True)
    dialog = launch.LaunchRunDialog(cohort)
    qapp.processEvents()
    return dialog


def _bar(dialog) -> RecordingFilter:
    return dialog.findChild(RecordingFilter)


def test_the_launch_dialog_filters_by_the_experiments_columns(run_dialog):
    bar = _bar(run_dialog)

    _choose(bar, "Strain", "C57")

    assert len(_shown(run_dialog._recordings)) == REAL_ROWS // 2
    assert "8 of 16 recordings shown" in bar._count.text()


def test_select_all_and_clear_act_on_the_rows_shown(run_dialog):
    bar = _bar(run_dialog)
    _choose(bar, "Strain", "CD1")

    run_dialog._select_all_recordings()
    picked = run_dialog._selected_recordings()
    assert len(picked) == REAL_ROWS // 2

    _choose(bar, "Strain", "C57")
    run_dialog._select_all_recordings()  # the C57 rows join; the hidden CD1 rows stay
    assert len(run_dialog._selected_recordings()) == REAL_ROWS

    run_dialog._clear_all_recordings()  # clears only the C57 rows it shows
    assert run_dialog._selected_recordings() == picked


def test_a_selection_survives_a_filter_and_the_count_says_so(run_dialog, qapp):
    bar = _bar(run_dialog)
    run_dialog._recordings.item(0).setCheckState(Qt.CheckState.Checked)
    run_dialog._recordings.item(1).setCheckState(Qt.CheckState.Checked)
    chosen = run_dialog._selected_recordings()
    assert len(chosen) == 2

    bar._text.setText("nothing matches this")
    qapp.processEvents()

    assert _shown(run_dialog._recordings) == []
    assert run_dialog._selected_recordings() == chosen  # still submitted
    assert "2 selected (2 hidden by the filter)" in bar._count.text()


def test_the_analysis_dialog_has_the_filter_too(qapp, cohort):
    dialog = launch.LaunchAnalysisDialog(cohort)
    qapp.processEvents()

    assert _bar(dialog) is not None
    assert "Strain" in _bar(dialog)._columns


def test_rows_are_still_selected_by_their_ids_not_their_text(run_dialog, qapp):
    """The row's text now has company (a painted column); the ids still come from the row."""
    bar = _bar(run_dialog)
    _choose(bar, "Brain_state", "off")
    run_dialog._select_all_recordings()

    ids = run_dialog._selected_recordings()

    names = run_dialog._experiment.recording_names()
    assert all("DrugOff" in names[i] for i in ids)
    assert len(ids) == REAL_ROWS // 2


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
_GUI = Path(launch.__file__).resolve().parent

#: Files that build a list of recordings and attach no ``RecordingFilter``, and why.
_OWN_FILTER = {
    "label_browse.py": (
        "the label browser filters its recordings with its own search box and a metadata "
        "column/value pair, as part of what the page is for"
    ),
}


def _is_recording_list_assignment(node: ast.AST) -> bool:
    """``self.<something with 'recording', 'targets' or 'score_on'> = QListWidget()``."""
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if not (isinstance(func, ast.Name) and func.id == "QListWidget"):
        return False
    return any(
        isinstance(t, ast.Attribute)
        and any(word in t.attr for word in ("recording", "_targets", "_score_on"))
        for t in node.targets
    )


def _filters_attached(tree: ast.AST) -> int:
    return sum(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in _ATTACHES)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in _ATTACHES)
        )
        for node in ast.walk(tree)
    )


_ATTACHES = {"RecordingFilter", "with_filter"}


def test_every_picker_of_recordings_attaches_the_filter():
    """Discovered from source: a list of recordings built under ``detail/gui`` is under a filter.

    The count rule rather than a name: a file with two lists needs two attachments, so wiring
    one and forgetting its twin (the Launch dialogs are such a pair) still fails.
    """
    unfiltered = []
    found = 0
    for path in sorted(_GUI.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        lists = sum(_is_recording_list_assignment(n) for n in ast.walk(tree))
        if not lists:
            continue
        found += lists
        relative = path.relative_to(_GUI).as_posix()
        if relative in _OWN_FILTER:
            continue
        if _filters_attached(tree) < lists:
            unfiltered.append(
                f"{relative} ({lists} list(s), {_filters_attached(tree)} filter(s))"
            )
    assert found >= 9, "the scan found too few recording lists: its pattern has drifted"
    assert unfiltered == [], (
        f"these build a list of recordings without attaching RecordingFilter: {unfiltered}. "
        "Wrap the list with `with_filter(...)` (or put a `RecordingFilter` above it), or add the "
        "file to _OWN_FILTER with the reason it filters on its own."
    )


def test_the_exemptions_still_exempt_something():
    """Guard the guard: an entry for a file that no longer builds such a list is dead."""
    for name in _OWN_FILTER:
        tree = ast.parse((_GUI / name).read_text(encoding="utf-8-sig"))
        assert any(_is_recording_list_assignment(n) for n in ast.walk(tree)), name
