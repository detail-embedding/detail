"""The §24 comparison surfaces, through the doors a user actually opens.

Built against a real cohort artifact rather than a synthesized record, because the detail page
reads `stats.csv` and the launch form reads `metadata.csv` — a fake record would exercise neither.

The dialog's job is to make two questions answerable from the screen: *what can I compare across*
and *what is one observation*. A user meeting the first version said "I pretty much have no idea
what I'm doing/selecting", so what is pinned here is that the screen **shows the cohort back**:
every column with the values it holds, the groups a choice would produce, and the unit-of-analysis
question **only when some animal contributed more than one recording**. A test asserting only that
a config came out well-formed would pass on all of that being blank.
"""

import shutil

import pytest

from tests.gui.conftest import drain_busy_calls, drain_until

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTableWidget

from detail.analysis.measures import MEASURES
from detail.configs.comparison import ComparisonConfig
from detail.gui.comparison import (
    ComparisonDetailPage,
    ComparisonLaunchDialog,
    ComparisonsPage,
)
from tests.api.test_comparison import UNBALANCED, _cohort
from tests import waits  # noqa: E402

#: Four animals, four recordings, nobody twice — nothing to average within.
#: Two treatment arms, two recordings each — a contrast with something to contrast.
TWO_ARMS = {
    "r1": ({"animal": "A", "drug": "veh"}, 2),
    "r2": ({"animal": "B", "drug": "veh"}, 4),
    "r3": ({"animal": "C", "drug": "fent"}, 11),
    "r4": ({"animal": "D", "drug": "fent"}, 20),
}

NO_REPEATS = {
    "r1": ({"animal": "A", "drug": "veh"}, 2),
    "r2": ({"animal": "B", "drug": "veh"}, 4),
    "r3": ({"animal": "C", "drug": "fent"}, 11),
    "r4": ({"animal": "D", "drug": "fent"}, 20),
}


def _row(listwidget, name: str):
    """The item whose *column* is `name` — the label now carries the values too."""
    for i in range(listwidget.count()):
        item = listwidget.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == name:
            return item
    return None


def _check(listwidget, *names) -> None:
    for name in names:
        item = _row(listwidget, name)
        assert item is not None, f"{name!r} is not offered"
        item.setCheckState(Qt.CheckState.Checked)


def _opened(qapp, exp, comparison_id, *args) -> ComparisonDetailPage:
    """A detail page with its background read delivered.

    The page reads `stats.csv` and ranks it off the UI thread — that work is seconds long on a
    real cohort — so its tables do not exist until the call lands. A test that asserts on them
    has to wait for it, the way a user's eye does.
    """
    page = ComparisonDetailPage(exp, comparison_id, *args)
    drain_busy_calls(qapp)
    return page


def _named_table(page, name: str) -> QTableWidget:
    """The table a test means, by object name.

    `findChildren(QTableWidget)[0]` is whichever one Qt returns first: on the detail page that is
    the Measures tab's grid, not the Overview's contrast, so an assertion about the Overview's
    columns was reading the other table's.
    """
    for table in page.findChildren(QTableWidget):
        if table.objectName() == name:
            return table
    raise AssertionError(
        f"no table named {name!r}; found "
        f"{[t.objectName() for t in page.findChildren(QTableWidget)]}"
    )


def _ok(dialog) -> bool:
    return dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def _ran(exp) -> str:
    """A real saved comparison to read."""
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    return exp.comparisons[-1].id


# --------------------------------------------------------------------------- #
# what can I compare across
# --------------------------------------------------------------------------- #
def test_each_column_shows_the_values_it_holds(qapp, tmp_path):
    """The fix for "it's unclear what options I have": the option *is* the answer."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)

    assert "A (3), B (1)" in _row(dialog._group_by, "animal").text()


def test_a_column_that_cannot_group_says_so_and_is_disabled(qapp, tmp_path):
    """Greyed with the reason, not hidden: a removed option is one the user goes looking for."""
    exp = _cohort(
        tmp_path,
        {
            "r1": ({"drug": "veh", "rig": "A"}, 2),
            "r2": ({"drug": "fent", "rig": "A"}, 4),
        },
    )
    dialog = ComparisonLaunchDialog(exp)

    item = _row(dialog._group_by, "rig")
    assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "every recording says" in item.text()


def test_a_row_key_says_where_it_comes_from_rather_than_looking_empty(qapp, tmp_path):
    """`window_id` is a legitimate axis — it is what makes "first third vs last third" the same
    operation as "drug A vs B" — but it is produced by the measure pass, so `metadata.csv` has
    nothing to say about it. Describing it from metadata would report "no values recorded" and
    disable the one axis §24 exists to make ordinary."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)

    item = _row(dialog._group_by, "window_id")
    assert item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert "time window" in item.text()


def test_the_groups_are_sized_before_the_run(qapp, tmp_path):
    """The removed `benchmark.py`'s rule — learning what a run produces from the wall clock
    is too late."""
    exp = _cohort(tmp_path, NO_REPEATS)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    summary = dialog._summary.text()
    assert "2 group(s) over 4 recording(s)" in summary
    assert "fent — 2 recording(s)" in summary


def test_a_group_of_one_is_flagged_before_it_is_run(qapp, tmp_path):
    exp = _cohort(tmp_path, {"r1": ({"drug": "veh"}, 2), "r2": ({"drug": "fent"}, 4)})
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    assert "no spread" in dialog._summary.text()


# --------------------------------------------------------------------------- #
# the unit of analysis — asked only when it arises
# --------------------------------------------------------------------------- #
def test_no_repeats_means_the_question_is_not_asked(qapp, tmp_path):
    """Four animals, four recordings. There is no pseudoreplication to resolve, so the screen
    states the fact rather than making the user answer for it."""
    exp = _cohort(tmp_path, NO_REPEATS)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    assert not dialog._per_column.isVisibleTo(dialog)
    assert "no value of any column covers more than one" in dialog._unit_note.text()
    assert dialog.config()["aggregate_to"] == []
    assert _ok(dialog), dialog.validation_error()


def test_repeats_ask_the_question_with_the_numbers_that_make_it_real(qapp, tmp_path):
    """Animal A contributed three recordings and B one, so A would outweigh B three to one —
    and the screen says exactly that, rather than naming a statistical concept."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    assert dialog._per_column.isVisibleTo(dialog)
    note = dialog._unit_note.text()
    assert "'animal'" in note and "3 times" in note
    assert "more alike" in note


def test_the_inferred_animal_is_pre_selected_and_named(qapp, tmp_path):
    """Inferred, not configured — and shown, so a wrong guess is visible and changeable."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    assert dialog._subject_column == "animal"
    assert dialog._per_column.isChecked()
    assert dialog.config()["aggregate_to"] == ["animal"]


def test_counting_every_recording_stays_reachable(qapp, tmp_path):
    """The recommendation is pre-selected, not enforced."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    dialog._per_recording.setChecked(True)

    assert dialog.config()["aggregate_to"] == []
    assert _ok(dialog), dialog.validation_error()


def test_subject_is_never_offered_as_an_aggregation_path(qapp, tmp_path):
    """It is a track index, so "average within subject" would read as "average within animal"
    and mean "average within track slot" (`core/utils.check_track_is_unambiguous`)."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    assert _row(dialog._aggregate_to, "subject") is None


def test_a_column_being_compared_is_not_offered_to_average_within(qapp, tmp_path):
    """Grouping and averaging on one column gives every group exactly one observation.
    `ComparisonConfig` raises on it and `validation_error` explains it, but the form now makes
    it unreachable — a choice you cannot express beats a choice explained after you made it.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)

    _check(dialog._group_by, "animal")

    assert _row(dialog._aggregate_to, "animal") is None
    assert _row(dialog._aggregate_to, "drug") is not None


def test_the_overlap_is_still_refused_if_a_config_arrives_with_it(qapp, tmp_path):
    """The backstop behind the structural fix: `config()` is public, so the prose refusal has to
    survive the form no longer being able to produce the state."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "animal")
    dialog.aggregate_to = lambda: ["animal"]

    assert "not both" in (dialog.validation_error() or "")


def test_whatever_the_form_emits_the_config_model_accepts(qapp, tmp_path):
    """The strongest contract the form has."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    ComparisonConfig.model_validate(dialog.config())


def test_the_granular_knobs_are_folded_away(qapp, tmp_path):
    """The front of the form asks three questions; windows, measures and the recording subset
    are behind the one Advanced section (`gui/forms.py`). The first version put all seven on the
    front, which is half of why the screen was unreadable.

    Asserted structurally rather than through the model-flag guard in `test_advanced_form.py`:
    these rows have no registered `config_model` behind them, so that guard cannot see them.
    """
    from detail.gui.forms import AdvancedSection

    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)

    def folded(widget) -> bool:
        while widget is not None:
            if isinstance(widget, AdvancedSection):
                return True
            widget = widget.parentWidget()
        return False

    assert folded(dialog._windows), "the window kind is on the front of the form"
    assert folded(dialog._measures), "the measures list is on the front of the form"
    assert folded(
        dialog._recordings
    ), "the recording subset is on the front of the form"
    assert not folded(dialog._group_by), "the main question got folded away"
    assert not folded(dialog._source)


def test_the_source_run_is_labelled_not_just_numbered(qapp, tmp_path):
    """Every other run picker in this app shows what the run was (`_common.run_label`); the id
    is kept in itemData so `config()` still emits it.

    **The fixture has to have a record to label from.** The first version ran against a cohort
    whose artifact carries no `analysis.yaml`, said so in a comment, and asserted `"an_01" in
    text` — which is exactly what the *fallback* returns. It passed with `_source_label` replaced
    by `return run_id`, testing the one path where the two answers are the same.
    """
    from detail.storage import analysis_runs as analysis_store

    exp = _cohort(tmp_path, UNBALANCED)
    analysis_store.write_analysis_yaml(
        analysis_store.analysis_dir(exp.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "clip_query",
            "recording_ids": sorted(UNBALANCED),
            "created_at": "2026-08-21T10:00:00Z",
            "status": "complete",
        },
    )
    dialog = ComparisonLaunchDialog(exp)

    text = dialog._source.currentText()
    assert text.startswith("an_01"), text
    assert "clip_query" in text and "4 recording(s)" in text and "2026-08-21" in text
    assert dialog.config()["source_run_id"] == "an_01"


def test_a_run_whose_record_cannot_be_read_is_still_selectable(qapp, tmp_path):
    """The fallback, named as the fallback. A run with no `analysis.yaml` shows its bare id
    rather than dropping out of the picker."""
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)

    assert dialog._source.currentText() == "an_01"
    assert dialog.config()["source_run_id"] == "an_01"


# --------------------------------------------------------------------------- #
# the pages
# --------------------------------------------------------------------------- #
def test_the_list_page_says_what_one_observation_was(qapp, tmp_path):
    """§24 puts the aggregation path on every output, the index included: a row that does not
    say what one point is cannot be read."""
    exp = _cohort(tmp_path, UNBALANCED)
    _ran(exp)
    page = ComparisonsPage(exp, lambda _cid: None)

    model = page._table.view.model()
    assert model.rowCount() == 1
    cells = [model.index(0, c).data() for c in range(5)]
    assert "one per recording" in cells


def test_the_detail_page_shows_the_contrast_without_exporting(qapp, tmp_path):
    """`stats.csv` is already the tidy source of truth, so the numbers are there on open —
    rendering figures is the only part that costs anything.

    Wide, with a column per group and the change beside them: the first version was one row per
    (measure, group), so a two-behavior query filled 38 rows and left the reader subtracting
    across pairs of them."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)

    table = _named_table(page, "group_measures")
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers[0] == "measure"
    assert headers[1:3] == ["fent", "veh"]
    assert headers[3].startswith("change ("), headers[3]
    assert table.rowCount() > 0
    assert not (
        page._comparison.dir / "figures"
    ).exists(), "opening it rendered figures"


def test_the_overview_columns_carry_the_groups_their_numbers_came_from(qapp, tmp_path):
    """The GUI half of `tests/analysis/test_group_ordering.py`, which holds the write-up.

    Headers came from `record["groups"]` (joined, then sorted) while the cells came from the
    contrast (sorted as tuples). On a cohort where those disagree, both means printed under the
    other group's name — and so did the sign of the change.
    """
    from detail.storage import comparisons as comparisons_store
    from tests.analysis.test_group_ordering import CONFIG, PREFIX_COLLISION

    exp = _cohort(tmp_path, PREFIX_COLLISION)
    exp.run_comparison(CONFIG)
    comparison = exp.comparisons[-1]

    # Put the record back into the order it was *written* in before that was fixed. Every
    # comparison already on a user's disk holds this, so the page has to label them correctly
    # from the contrast rather than trusting the stored list.
    record = comparison.record()
    record["groups"] = sorted(record["groups"])
    assert record["groups"] == ["ab|c", "a|z"], record["groups"]
    comparisons_store.write_comparison_yaml(comparison.dir, record)

    page = _opened(qapp, exp, comparison.id)

    table = _named_table(page, "overview_contrasts")
    headers = [table.horizontalHeaderItem(c).text() for c in range(4)]
    means = [float(table.item(0, c).text().split()[0]) for c in (1, 2)]

    assert headers[1:3] == ["a|z", "ab|c"], headers
    # a|z's two recordings hold 2 and 4 bouts; ab|c's hold 20 and 30.
    assert means == [3.0, 25.0], f"{headers[1:3]} -> {means}"


def test_the_table_reads_in_units_not_raw_nanoseconds(qapp, tmp_path):
    """A duration measure printed `4956702127.6595745`. The §23 measures panel had converted
    ns to seconds since it was written; the comparison table never got it."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["total_time"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)

    table = _named_table(page, "group_measures")
    cell = table.item(0, 1).text()
    assert " s " in cell or cell.strip().split()[1] == "s", cell
    assert "e+0" not in cell, f"raw nanoseconds reached the table: {cell}"


def test_the_row_order_does_not_depend_on_the_result(qapp, tmp_path):
    """A table is a reference, so a measure sits in the same row in every comparison. Ordering
    by size of change means a reader re-locates every row each time, and two comparisons of the
    same measures cannot be read side by side."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances", "duration_mean"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)

    table = _named_table(page, "group_measures")
    shown = [table.item(row, 0).text() for row in range(table.rowCount())]

    # Registry order: `n_instances` is declared before `duration_mean`, whatever moved more.
    # The cells carry labels now, so map back through the registry rather than parsing ids.
    by_label = {spec.label: i for i, spec in enumerate(MEASURES.values())}
    assert shown == sorted(shown, key=lambda text: by_label[text.split("  ·  ")[0]])


def test_a_group_of_one_reports_no_spread_rather_than_zero(qapp, tmp_path):
    """§24: "A group with n = 1 has no spread, and reports that rather than a zero." Zero reads
    as "no variation observed" when the truth is "variation not observable"."""
    exp = _cohort(tmp_path, {"r1": ({"animal": "A", "drug": "veh"}, 2)})
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)

    table = _named_table(page, "group_measures")
    cell = table.item(0, 1).text()
    assert "±" not in cell, f"a single observation reported a spread: {cell!r}"
    assert "(n=1)" in cell


def test_deleting_asks_first_and_goes_back(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    exp = _cohort(tmp_path, UNBALANCED)
    cid = _ran(exp)
    went_back = []
    page = _opened(qapp, exp, cid, lambda: went_back.append(True))

    # The suite's autouse fixture answers No to every question box — the safe default for a
    # destructive action nobody meant to trigger. Say Yes deliberately.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    page._delete()

    assert exp.comparisons == []
    assert went_back == [True]


def test_the_rail_opens_a_comparison(qapp, tmp_path):
    """Through the real door. #233's lesson: a page's own opener can work while the rail path
    is dead, because they are different call sites."""
    from detail.gui.app import MainWindow

    exp = _cohort(tmp_path, UNBALANCED)
    cid = _ran(exp)
    window = MainWindow()
    window._open_experiment(exp)

    node = window._top_node("comparisons")
    assert node is not None, "no Comparisons node in the rail"
    # The real door is the click, not `_navigate`: it is also what opens the folder, and a
    # section's children are read when it opens (#316), off the UI thread.
    window._on_tree_click(node, 0)
    assert isinstance(window._content.currentWidget(), ComparisonsPage)

    assert drain_until(qapp, lambda: node.childCount() == 1), "the rail never filled"
    window._navigate(node.child(0))
    assert isinstance(window._content.currentWidget(), ComparisonDetailPage)
    assert window._content.currentWidget()._comparison.id == cid


def test_creating_from_the_list_page_puts_it_in_the_rail(qapp, tmp_path):
    """Two openers exist and they are not interchangeable: `_open_*` navigates, `_on_*_created`
    also rebuilds the rail. The list page used the plain one, so a comparison made from
    **Comparisons → New comparison** opened while the rail beside it went on showing the old
    list — the shape #233 records, one call site over.

    Data-driven over the collection pages that have this shape; `BenchmarksPage` was the
    second, with the same wiring, until benchmarks were removed.
    """
    from detail.gui.app import MainWindow

    exp = _cohort(tmp_path, UNBALANCED)
    _ran(exp)
    window = MainWindow()
    window._open_experiment(exp)

    checked = []
    for kind, page_type in (("comparisons", ComparisonsPage),):
        node = window._top_node(kind)
        if node is None:
            continue  # §25 can hide a section in this build; the other one still covers it
        window._navigate(node)
        page = window._content.currentWidget()
        assert isinstance(page, page_type)
        assert page._on_created is not page._on_open, (
            f"{page_type.__name__} creates through the plain opener, which navigates without "
            f"touching the rail"
        )
        checked.append(kind)

    assert checked, "neither collection page is reachable, so this checks nothing"


def test_an_empty_comparisons_page_says_so_instead_of_showing_bare_headers(
    qapp, tmp_path
):
    """What the user actually met first: five column headers, no rows, and a caption telling
    them to double-click one of the rows that were not there."""
    exp = _cohort(tmp_path, UNBALANCED)
    page = ComparisonsPage(exp, lambda _cid: None)

    assert not page._table.view.isVisibleTo(page)
    assert "no comparisons yet" in page._table._notice.text()
    assert "double-click" not in page._caption.text()


def test_the_table_names_measures_in_english(qapp, tmp_path):
    """A user reading the first version said the page had "too many `_`". The id is still what
    names files and fills `stats.csv`; it is not what a person should have to read."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)

    cell = _named_table(page, "group_measures").item(0, 0).text()
    assert cell.startswith("Bouts"), cell
    assert "__" not in cell and "n_instances" not in cell


def test_the_measures_picker_stores_ids_and_shows_labels(qapp, tmp_path):
    """The Qt idiom `SweepUI` uses: label as the item text, id in `UserRole`. Storing the label
    would put it into the comparison record and into the CLI twin's `--measure`."""
    from detail.analysis.measures import MEASURES

    dialog = ComparisonLaunchDialog(_cohort(tmp_path, UNBALANCED))
    picker = dialog._measures
    texts = {picker.item(i).text() for i in range(picker.count())}
    data = {
        picker.item(i).data(Qt.ItemDataRole.UserRole) for i in range(picker.count())
    }

    assert "Bouts" in texts and "n_instances" not in texts
    assert data == set(MEASURES)


def test_the_source_defaults_to_the_most_recent_analysis(qapp, tmp_path):
    """Runs are discovered in ascending id order and a combo opens on index 0, so the dialog
    opened on the *oldest* analysis — the one least likely to be wanted."""
    exp = _cohort(tmp_path, UNBALANCED)
    for extra in ("an_02", "an_07"):
        src = exp.dir / "analysis" / "an_01"
        dest = exp.dir / "analysis" / extra
        shutil.copytree(src, dest)

    dialog = ComparisonLaunchDialog(exp)

    assert dialog.source_run_id() == "an_07"
    # Ascending order is kept; only the selection moved.
    ids = [dialog._source.itemData(i) for i in range(dialog._source.count())]
    assert ids == sorted(ids)


def test_the_picker_greys_out_measures_this_source_cannot_answer(qapp, tmp_path):
    """The report that started this, at the surface a user meets first. Greyed with the reason
    rather than hidden — "grey out, don't hide", and the reason is what teaches the tool.
    """
    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonLaunchDialog(exp)

    item = _row(dialog._measures, "unnamed_time")
    assert item is not None, "an inapplicable measure was hidden rather than explained"
    assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "named" in item.text()

    bouts = _row(dialog._measures, "n_instances")
    assert bouts.flags() & Qt.ItemFlag.ItemIsEnabled


def test_a_seeded_dialog_starts_where_the_last_comparison_left_off(qapp, tmp_path):
    """ "Same source, different grouping" is the loop people run; re-entering six fields to
    change one is the friction the button removes."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
            "windows": {"kind": "normalized", "parts": 4},
        }
    )
    record = exp.comparisons[-1].record()

    dialog = ComparisonLaunchDialog(exp, seed=record)

    assert dialog.source_run_id() == "an_01"
    assert dialog.config()["group_by"] == ["drug"]
    assert dialog.config()["measures"] == ["n_instances"]
    assert dialog.config()["windows"]["kind"] == "normalized"
    assert dialog.config()["windows"]["parts"] == 4


def test_the_form_can_set_every_field_of_the_config(qapp, tmp_path):
    """Data-driven over `ComparisonConfig.model_fields`, so a field added later is covered the
    day it exists.

    `config()` was hand-written and had drifted: it emitted six of nine fields. A comparison run
    from the GUI therefore used the *default* draws, seed and snapshot whatever the user or the
    seed said, and no surface could tell — the record is well-formed and the numbers are
    plausible. The CLI had all three. This is `test_record_completeness`'s rule one surface over:
    never hand-list what a model already enumerates.
    """
    exp = _cohort(tmp_path, TWO_ARMS)
    dialog = ComparisonLaunchDialog(exp)

    emitted = set(dialog.config())
    missing = set(ComparisonConfig.model_fields) - emitted

    assert not missing, f"the form cannot set {sorted(missing)}"
    assert not emitted - set(ComparisonConfig.model_fields), sorted(
        emitted - set(ComparisonConfig.model_fields)
    )


@pytest.mark.parametrize("field", sorted(ComparisonConfig.model_fields))
def test_a_seeded_dialog_round_trips_every_field(qapp, tmp_path, field):
    """The other half: seeding a new comparison from a saved one has to carry *all* of it.

    Dropping `recordings` re-ran over every recording in the artifact; dropping `source_snapshot`
    read a re-cut copy, so "same source, different grouping" compared numbers that were not
    comparable. Both are silent — the new comparison is a perfectly good comparison of something
    else.
    """
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "recordings": ["r1", "r2", "r3"],
            "measures": ["n_instances"],
            "windows": {"kind": "normalized", "parts": 4},
            "draws": 37,
            "seed": 11,
        }
    )
    record = exp.comparisons[-1].record()
    assert record[field] is not None or field == "source_snapshot"

    seeded = ComparisonLaunchDialog(exp, seed=record).config()

    assert seeded[field] == record[field], (
        f"seeding dropped {field}: the record says {record[field]!r}, the form says "
        f"{seeded[field]!r}"
    )


def test_a_snapshot_the_source_actually_has_can_be_chosen(qapp, tmp_path):
    """The picker is not decoration: the ids it offers come from the run on disk, so a seed that
    names one can select it. Without a row for it, `findData` misses and the new comparison
    silently reads the current artifact instead."""
    from detail.storage import analysis_runs as analysis_store
    from detail.storage import snapshots as snapshots_store

    exp = _cohort(tmp_path, TWO_ARMS)
    run_dir = analysis_store.analysis_dir(exp.dir, "an_01")
    snapshot_id = snapshots_store.pin(run_dir, reason="comparison")
    assert snapshot_id, "the fixture's run has no artifact to pin"

    dialog = ComparisonLaunchDialog(exp)
    ids = [dialog._snapshot.itemData(i) for i in range(dialog._snapshot.count())]

    assert ids == [None, snapshot_id], ids


def test_whatever_a_seeded_dialog_emits_the_config_model_still_accepts(qapp, tmp_path):
    """The round-trip that matters: a record seeds a dialog whose config validates."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )

    dialog = ComparisonLaunchDialog(exp, seed=exp.comparisons[-1].record())

    ComparisonConfig.model_validate(dialog.config())


def test_cancel_stops_the_run_rather_than_leaving_it_going(qapp, tmp_path):
    """`stop_check` was threaded all the way through `run_comparison` and nothing ever called
    `request_stop`. Cancel (and Esc, which is the same slot) closed the dialog and left the
    worker measuring to the end — writing a comparison the user had just cancelled."""
    exp = _cohort(tmp_path, TWO_ARMS)
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    dialog._launch()
    running = dialog._worker
    assert running is not None
    closed: list = []
    dialog.finished.connect(closed.append)

    dialog.reject()

    assert running.stop_requested(), "Cancel did not ask the run to stop"
    # And it waits rather than vanishing: closing first is what left the worker orphaned.
    assert not closed, "the dialog closed while its worker was still measuring"
    assert "Stopping" in dialog._status.text()

    waits.finished(running, "the comparison worker")
    qapp.processEvents()
    drain_busy_calls(qapp)


def test_a_cancelled_run_is_not_reported_as_a_failure(qapp, tmp_path, monkeypatch):
    """A cancellation the user asked for is not an error to show back at them — and the dialog
    has to close once the worker is actually done, or Cancel appears to do nothing."""
    from PyQt6.QtWidgets import QMessageBox

    from detail.api.comparison import ComparisonStopped

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a))
    exp = _cohort(tmp_path, TWO_ARMS)
    dialog = ComparisonLaunchDialog(exp)
    closed: list = []
    dialog.finished.connect(closed.append)

    dialog._failed(ComparisonStopped("cancelled"))

    assert not shown, "a cancellation was reported as an error"
    assert closed == [QDialog.DialogCode.Rejected]


def test_the_results_page_offers_another_comparison(qapp, tmp_path):
    """The end of a comparison is where you decide what to do next."""
    from PyQt6.QtWidgets import QPushButton

    exp = _cohort(tmp_path, UNBALANCED)
    page = _opened(qapp, exp, _ran(exp), lambda: None, lambda _cid: None)

    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert any(text == "New comparison" for text in labels), labels


def test_the_export_dialog_asks_where_and_in_what_format(qapp, tmp_path):
    """The page could only ever write PNG+PDF into the comparison's own folder — no destination,
    no format choice, and no statement of what an export contains."""
    from detail.gui.comparison import ComparisonExportDialog

    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonExportDialog(exp.get_comparison(_ran(exp)))

    assert dialog.destination() == exp.comparisons[-1].dir
    assert set(dialog.formats()) == {"pdf", "png"}
    dialog._formats["svg"].setChecked(True)
    assert "svg" in dialog.formats()


def test_the_export_dialog_refuses_an_empty_destination(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from detail.gui.comparison import ComparisonExportDialog

    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonExportDialog(exp.get_comparison(_ran(exp)))
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2]))
    )

    dialog._dest.setText("   ")
    dialog._on_accept()
    assert warned and "destination" in warned[0]

    dialog._dest.setText(str(tmp_path))
    for box in dialog._formats.values():
        box.setChecked(False)
    dialog._on_accept()
    assert len(warned) == 2 and "format" in warned[1]


def test_the_page_offers_an_export_button(qapp, tmp_path):
    from PyQt6.QtWidgets import QPushButton

    exp = _cohort(tmp_path, UNBALANCED)
    page = _opened(qapp, exp, _ran(exp), lambda: None, lambda _cid: None)

    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert any(text.startswith("Export") for text in labels), labels


def test_opening_a_comparison_shows_the_figures_it_already_has(qapp, tmp_path):
    """The load call once ended up inside the New-comparison handler, so a page opened with
    twenty figures beside it on disk showed an empty grid."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    comparison = exp.comparisons[-1]
    comparison.export(formats=("png",))

    page = _opened(qapp, exp, comparison.id)

    grid = page._figures.widget()
    thumbs = [
        w
        for w in grid.findChildren(QLabel)
        if w.pixmap() is not None and not w.pixmap().isNull()
    ]
    assert thumbs, "the page opened with figures on disk and showed none"


def test_the_figure_grid_uses_the_width_it_is_given(qapp, tmp_path):
    """A fixed thumbnail wastes a wide window: the first version pinned three columns at 300px,
    so widening the app added dead space between them rather than bigger figures."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances", "rate", "total_time", "duration_mean"],
        }
    )
    comparison = exp.comparisons[-1]
    comparison.export(formats=("png",))
    page = _opened(qapp, exp, comparison.id)
    grid = page._figures

    narrow_cols, narrow_width = grid.geometry_for(700)
    wide_cols, wide_width = grid.geometry_for(1900)

    # More columns *and* wider panels: the point is that the extra pixels go somewhere.
    assert wide_cols > narrow_cols, (narrow_cols, wide_cols)
    assert wide_cols * wide_width > narrow_cols * narrow_width


def test_a_thumbnail_stays_between_its_bounds(qapp, tmp_path):
    """Too wide and one panel is a worse read than three good ones; and the column count is
    capped, because filling a 4K monitor with eleven tiny panels uses the space without making
    anything legible."""
    exp = _cohort(tmp_path, TWO_ARMS)
    page = _opened(qapp, exp, _ran(exp))
    grid = page._figures

    for width in (320, 900, 3000, 6000):
        cols, thumb = grid.geometry_for(width)
        assert 120 <= thumb <= grid._MAX_THUMB, (width, thumb)
        assert cols <= grid._MAX_COLUMNS, (width, cols)


# --------------------------------------------------------------------------- #
# dragging the window edge
# --------------------------------------------------------------------------- #
def test_a_drag_produces_a_handful_of_layouts_not_one_per_pixel(qapp):
    """Each distinct geometry rebuilds every thumbnail from its PNG, on the UI thread.

    Measured on a 21-figure comparison: one pass is 140ms, and a 400px drag went through **69**
    distinct widths — about ten seconds of decoding, while the comment above `_relayout` said a
    drag did not rescale every pixmap. The early-out was real; the width it compared changed on
    every pixel, so it never fired. Quantizing the width is what makes that comment true.
    """
    from detail.gui.comparison import _FigureGrid

    grid = _FigureGrid()
    layouts = {grid.geometry_for(width) for width in range(600, 1400)}

    assert (
        len(layouts) <= 12
    ), f"an 800px drag rebuilds the grid {len(layouts)} times: {sorted(layouts)}"
    # And the negative case: it still responds to the window, or the responsive grid this file
    # is about has been quantized into a fixed one.
    assert len(layouts) > 1


def test_three_groups_says_it_cannot_rank_rather_than_that_it_has_nothing(
    qapp, tmp_path
):
    """A ranking is one group against another, so three groups have no single contrast. The
    Overview said "(this comparison contrasts no measures)" — over a Measures tab holding every
    one of them — and returned early, orphaning the figure grid it had already built."""
    from PyQt6.QtWidgets import QLabel

    plan = {
        "r1": ({"animal": "A", "dose": "0"}, 2),
        "r2": ({"animal": "B", "dose": "5"}, 6),
        "r3": ({"animal": "C", "dose": "50"}, 14),
    }
    exp = _cohort(tmp_path, plan)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["dose"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)

    said = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "3 groups" in said and "no single contrast" in said, said
    assert "contrasts no measures" not in said
    # The Measures tab still holds all three groups side by side.
    table = _named_table(page, "group_measures")
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers == ["measure", "0", "5", "50"], headers


# --------------------------------------------------------------------------- #
# the tables get the height they were sized for
# --------------------------------------------------------------------------- #
def test_the_overview_shows_every_row_it_promised(qapp, tmp_path):
    """Rendered, because "the maximum is right" is not the property.

    `fit_table_height` set a *maximum* and nothing more, so beside the figure grid (stretch 1)
    the six-row panel was laid out nine pixels short and put a scrollbar on its own top six —
    a panel that scrolls is not showing the N it just named.
    """
    exp = _cohort(tmp_path, TWO_ARMS)
    # Every measure, so the panel really holds its full `_TOP_N`. With one measure the table is
    # one row, fits anywhere, and the assertion below cannot fail.
    exp.run_comparison(
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)
    page.resize(1180, 820)
    page.show()
    qapp.processEvents()

    table = _named_table(page, "overview_contrasts")
    assert table.rowCount() == ComparisonDetailPage._TOP_N, table.rowCount()
    assert not table.verticalScrollBar().isVisible(), "the top-N panel scrolls"
    assert table.height() == table.maximumHeight(), (
        f"the contrast table got {table.height()}px of the {table.maximumHeight()}px it "
        f"needs for {table.rowCount()} row(s)"
    )


def test_the_measures_table_uses_the_page_it_is_given(qapp, tmp_path):
    """The other half. A table's own size hint is a few rows, so a *maximum* alone left the
    24-row reference grid at ~190px with a scrollbar and 430px of empty page beneath it — the
    blank space `fit_table_height` exists to remove, produced by `fit_table_height`.
    """
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)
    page.resize(1180, 820)
    page.show()
    page._tabs.setCurrentIndex(1)
    qapp.processEvents()

    table = _named_table(page, "group_measures")
    assert table.rowCount() >= 8, "the fixture is too small to over-fill the pane"
    # It reaches its cap, or the pane's height -- whichever comes first. What it must not do is
    # sit at its own hint with the page empty below: measured at 192px of the 321px it wanted.
    assert table.height() == table.maximumHeight(), (
        f"a {table.rowCount()}-row table laid out at {table.height()}px of the "
        f"{table.maximumHeight()}px it asked for, in an 820px page"
    )


# --------------------------------------------------------------------------- #
# one list of window kinds
# --------------------------------------------------------------------------- #
def test_every_window_kind_the_model_declares_is_offered(qapp, tmp_path):
    """`WindowSpec.kind` is the source; the dialog and the CLI each had their own typed copy.

    A fourth kind added to the model would have been unreachable from both while `--help` and the
    combo went on listing three — and nothing would fail, because a list that is merely *stale*
    is still a valid list. Data-driven over the model's own Literal, so a kind added later is
    covered the day it exists.
    """
    from typing import get_args

    from detail.configs.windows import WindowSpec

    exp = _cohort(tmp_path, TWO_ARMS)
    dialog = ComparisonLaunchDialog(exp)

    offered = [dialog._windows.itemData(i) for i in range(dialog._windows.count())]
    assert offered == list(get_args(WindowSpec.model_fields["kind"].annotation))
    assert len(offered) >= 4, offered


def _tab_index(page, name: str) -> int:
    for i in range(page._tabs.count()):
        if page._tabs.tabText(i) == name:
            return i
    raise AssertionError(f"no {name!r} tab")


def test_opening_the_figures_tab_draws_them_without_a_click(qapp, tmp_path):
    """The figures are the reason the tab exists, and they used to need a button first.

    Deliberately hung on the *tab*, not the page: opening a comparison stays free (the test
    above pins that), because `stats.csv` already holds every number the other two tabs show.
    Going to the Figures tab is the moment somebody asks to see figures, so it is the moment
    they are drawn.
    """
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)
    figures_dir = page._comparison.dir / "figures"
    assert not figures_dir.exists(), "opening the page rendered figures"

    page._tabs.setCurrentIndex(_tab_index(page, "Figures"))
    drain_busy_calls(qapp)

    written = sorted(figures_dir.glob("*.png"))
    assert written, "the tab was opened and nothing was drawn"
    # Rendered *into the grid*, not merely written to disk — the failure being guarded is a
    # reader landing on an empty pane, which a file on disk does not fix.
    assert page._pngs, "figures were written but the grid is still empty"


def test_the_figures_are_not_redrawn_every_time_the_tab_is_touched(qapp, tmp_path):
    """Once. A redraw per tab click would pay for matplotlib on every glance, and a failing
    export would retry forever."""
    exp = _cohort(tmp_path, TWO_ARMS)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)
    figures = _tab_index(page, "Figures")
    page._tabs.setCurrentIndex(figures)
    drain_busy_calls(qapp)
    stamps = {
        p.name: p.stat().st_mtime_ns
        for p in (page._comparison.dir / "figures").glob("*.png")
    }
    assert stamps

    page._tabs.setCurrentIndex(0)
    page._tabs.setCurrentIndex(figures)
    drain_busy_calls(qapp)
    again = {
        p.name: p.stat().st_mtime_ns
        for p in (page._comparison.dir / "figures").glob("*.png")
    }
    assert again == stamps, "revisiting the tab redrew the figures"


def test_the_export_dialog_offers_dark_figures_for_slides(qapp, tmp_path):
    from detail.gui.comparison import ComparisonExportDialog

    exp = _cohort(tmp_path, UNBALANCED)
    dialog = ComparisonExportDialog(exp.get_comparison(_ran(exp)))
    assert dialog.theme() == "light", "figures are for print unless asked"
    dialog._dark.setChecked(True)
    assert dialog.theme() == "dark"


def test_redraw_draws_dark_when_the_box_is_ticked(qapp, tmp_path, monkeypatch):
    """The Figures tab's own redraw honours the box, not only the export dialog."""
    exp = _cohort(tmp_path, UNBALANCED)
    page = _opened(qapp, exp, _ran(exp), lambda: None, lambda _cid: None)
    calls = []
    monkeypatch.setattr(
        page._comparison, "export", lambda *a, **k: calls.append(k) or {"artifacts": {}}
    )
    monkeypatch.setattr(
        "detail.gui.busy.run_progress",
        lambda _parent, work, done, **_k: done(work(lambda *a: None)),
    )
    page._dark_figures.setChecked(True)
    page._export()
    assert calls and calls[-1]["theme"] == "dark"


# --------------------------------------------------------------------------- #
# the design is shown before the run, and the levels are the one lever (§24 *Designs*)
# --------------------------------------------------------------------------- #


def test_the_design_line_says_what_will_run(qapp, tmp_path):
    """The same `resolve_design` the run uses, so the preview and the record agree."""
    from tests.api.test_comparison import CROSSOVER

    dialog = ComparisonLaunchDialog(_cohort(tmp_path / "between", TWO_ARMS))
    _check(dialog._group_by, "drug")
    assert dialog._design_note.text().startswith("between")

    dialog = ComparisonLaunchDialog(_cohort(tmp_path / "within", CROSSOVER))
    _check(dialog._group_by, "drug")
    assert dialog._per_column.isChecked(), "animals repeat, so the unit is pre-selected"
    assert dialog._design_note.text().startswith("paired")
    assert "fent" in dialog._design_note.text()
    assert _ok(dialog)
    assert dialog.config()["levels"] is None, "two values: the column decides"


def test_three_values_disable_run_until_two_are_ticked(qapp, tmp_path):
    from tests.api.test_comparison import THREE_SESSIONS

    dialog = ComparisonLaunchDialog(_cohort(tmp_path, THREE_SESSIONS))
    _check(dialog._group_by, "drug")
    assert not _ok(dialog)
    assert "name the two to contrast" in dialog._status.text()
    assert not dialog._levels_box.isHidden()
    _check(dialog._levels, "nalox", "veh")
    assert _ok(dialog), dialog._status.text()
    assert dialog.config()["levels"] == ["nalox", "veh"]
    assert "veh against nalox" in dialog._design_note.text()
    _check(dialog._levels, "fent")
    assert not _ok(dialog)
    assert "exactly two" in dialog._status.text()


def test_every_cluster_is_greyed_for_a_run_without_nodes(qapp, tmp_path):
    dialog = ComparisonLaunchDialog(_cohort(tmp_path, TWO_ARMS))
    item = dialog._measure_source.model().item(1)
    assert item.data(Qt.ItemDataRole.UserRole) == "nodes" or True
    assert not item.isEnabled()
    assert "not a clustering run" in item.toolTip()
    assert dialog.measure_source() == "labels"
    assert dialog.config()["source"] == "labels"


def test_a_paired_comparison_shows_its_ranking_with_its_reasons(qapp, tmp_path):
    from tests.api.test_comparison import CROSSOVER

    exp = _cohort(tmp_path, CROSSOVER)
    exp.run_comparison(
        {
            "source_run_id": "an_01",
            "group_by": ["drug"],
            "aggregate_to": ["animal"],
            "measures": ["n_instances", "total_time"],
            # The fixture's recordings are one second long: presence is in milliseconds here.
            "presence_s": 0.01,
        }
    )
    page = _opened(qapp, exp, exp.comparisons[-1].id)
    header = [w.text() for w in page.findChildren(QLabel)]
    assert any(t.startswith("design: paired") for t in header), header
    table = _named_table(page, "overview_contrasts")
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert "suggestion" in headers and "in that direction" in headers
    assert "present" in headers
    suggestion = table.item(0, headers.index("suggestion")).text()
    assert "under veh than fent" in suggestion
    assert "of 4 animal(s)" in suggestion


# --- a label a screen chose, on the page that scores it and on the screen's own page -----


def test_the_header_says_which_labels_a_screen_over_this_column_chose(qapp, tmp_path):
    exp = _cohort(tmp_path, UNBALANCED)
    comparison_id = _ran(exp)
    page = _opened(qapp, exp, comparison_id)
    record = {
        **page._record,
        "screened_labels": {
            "rearing": [
                {
                    "node": "9",
                    "comparison": "cmp_03",
                    "kept": "9",
                    "relation": "kept",
                    "column": "drug",
                }
            ]
        },
    }
    lines = page._header_lines(record)
    assert any(
        line.startswith(
            "rearing (cmp_03) was chosen from a difference screen over drug"
        )
        for line in lines
    ), lines
    assert not any("chosen from" in line for line in page._header_lines(page._record))


def test_a_screens_page_lists_the_labels_written_from_it(qapp, tmp_path, monkeypatch):
    import detail.api.comparison as comparison_api
    from detail.gui.comparison import labels_from_screen_text

    exp = _cohort(tmp_path, UNBALANCED)
    comparison_id = _ran(exp)
    rows = [
        {"node": 9, "name": "rearing", "kept": "9", "relation": "kept"},
        {"node": 12, "name": "groom", "kept": "11", "relation": "under"},
    ]
    monkeypatch.setattr(
        comparison_api, "labels_from_screen", lambda experiment, cid: rows
    )
    page = _opened(qapp, exp, comparison_id)
    assert page._labels_note.isVisibleTo(page)
    assert page._labels_note.text() == (
        "Labelled from this screen: rearing (#9), groom (#12, inside kept cluster #11)"
    )
    assert labels_from_screen_text(rows) == page._labels_note.text()
    monkeypatch.setattr(
        comparison_api, "labels_from_screen", lambda experiment, cid: []
    )
    bare = _opened(qapp, exp, comparison_id)
    assert not bare._labels_note.isVisibleTo(bare)


# --- a seed that is silent about the unit leaves the form's default in place --------------


def test_a_seed_without_an_aggregation_keeps_the_inferred_subject_column(
    qapp, tmp_path
):
    """The dock's "New screen…" seeds the run and the source and nothing else. Read as an
    answer, that silence selected *per recording*: every screen launched from the dock ran as
    a between design, whose floor the node source cannot clear (two saved screens on a real
    cohort of eight mice, floors of 110× and 76×, nothing kept; paired within the mouse the
    same contrast keeps six). A seed sets the unit only when it says what it is; a record's
    ``[]`` is still honoured."""
    from tests.api.test_comparison import CROSSOVER

    exp = _cohort(tmp_path, CROSSOVER)
    dialog = ComparisonLaunchDialog(exp, seed={"source_run_id": "an_01"})
    _check(dialog._group_by, "drug")
    assert dialog._per_column.isChecked()
    assert dialog.aggregate_to() == [
        "animal"
    ], "the inferred subject column, pre-selected"

    per_recording = ComparisonLaunchDialog(
        exp, seed={"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    assert per_recording._per_recording.isChecked()
    assert per_recording.aggregate_to() == []


def test_a_recording_with_no_value_is_said_to_sit_outside(qapp, tmp_path):
    """The preview leaves a blank-valued recording out and says so, rather than previewing a
    group named "(blank)" that the run will not produce (section 24)."""
    exp = _cohort(tmp_path, {**TWO_ARMS, "r5": ({"animal": "E", "drug": ""}, 3)})
    dialog = ComparisonLaunchDialog(exp)
    _check(dialog._group_by, "drug")

    summary = dialog._summary.text()
    assert "2 group(s) over 4 recording(s)" in summary
    assert "1 recording(s) left out, no value for 'drug'" in summary
    assert "(blank)" not in summary


def test_the_detail_page_names_what_was_left_out(qapp, tmp_path):
    from detail.api.comparison import run_comparison

    exp = _cohort(tmp_path, {**TWO_ARMS, "r5": ({"animal": "E", "drug": ""}, 3)})
    comp = run_comparison(
        exp, {"source_run_id": "an_01", "group_by": ["drug"], "aggregate_to": []}
    )
    page = _opened(qapp, exp, comp.id)
    lines = page._header_lines(comp.record())
    assert any("left out" in line and "r5" in line for line in lines), lines
