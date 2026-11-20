"""The comparison dialog and detail page over a location-export run (§24 *The locations source*).

**A source offered for a run that cannot answer it.** A location-export run holds no behaviors
and a behavior run holds no positions. The *Measure* picker greys what the chosen run cannot
offer, with the reason, and moves off a choice that just became impossible: a form that left
"Labelled behaviors" selected over a location export would launch a comparison that is refused.

**A form that starts somewhere the model does not.** An optional number box opens on "unset",
and unset is a real choice here (read every row) that is not the model's default (five a
second). Left unseeded, a comparison launched from the form read the run at another rate than
one launched from the command line.

**A verdict worded for another rule.** A locations comparison judges each measure by its own
test. A detail page that printed the family's floor beside it would say the wrong thing was
the judge.

**A form that describes recordings the run never read.** *Compare across*, *Average within*
and every count beside them were worked out over every recording in the experiment, whatever
the chosen run covered: over an export of three recordings out of fifty the form listed every
animal and every value, and inferred repeats the run would never see. They are described over
what the run covers now (``api.comparison.source_recordings``, the rule the run applies), and a
recording outside it is greyed in *Recordings* rather than offered.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtWidgets import QLabel, QTableWidget  # noqa: E402

from PyQt6.QtCore import Qt  # noqa: E402

from detail.analysis import cohort as cohort_lib  # noqa: E402
from detail.analysis import location_measures  # noqa: E402
from detail.configs.comparison import ComparisonConfig  # noqa: E402
from detail.storage import analysis_runs  # noqa: E402
from detail.storage import metadata as md  # noqa: E402
from detail.gui.comparison import (  # noqa: E402
    _LOCATION_FIELDS,
    ComparisonDetailPage,
    ComparisonLaunchDialog,
    _checked,
    _set_checked,
)
from tests.api.test_comparison import CROSSOVER, _cohort  # noqa: E402
from tests.gui.conftest import drain_busy_calls  # noqa: E402
from tests.api.test_comparison_locations import (
    EXPORTED,
    PAIRED,
    cohort,
    paired_plan,
)  # noqa: E402
from tests.location_fixtures import write_run  # noqa: E402


def _sources(dialog) -> dict[str, bool]:
    """``{source: offered}`` as the *Measure* picker shows it."""
    model = dialog._measure_source.model()
    return {
        dialog._measure_source.itemData(i): model.item(i).isEnabled()
        for i in range(dialog._measure_source.count())
    }


def _shaped(dialog) -> None:
    """Tick the paired design: across ``session``, averaged within ``animal``."""
    _set_checked(dialog._group_by, {"session"})
    dialog._per_column.setChecked(True)
    _set_checked(dialog._aggregate_to, {"animal"})
    dialog._refresh()


def test_a_location_export_offers_movement_and_nothing_else(qapp, tmp_path):
    dialog = ComparisonLaunchDialog(cohort(tmp_path, paired_plan()))
    assert dialog.source_run_id() == "an_01"
    assert (
        dialog.measure_source() == "locations"
    )  # moved there: nothing else is possible
    assert _sources(dialog) == {"labels": False, "nodes": False, "locations": True}
    reasons = dialog._source_reasons()
    assert (
        reasons["labels"]
        == reasons["nodes"]
        == "a location-export run holds no behaviors"
    )
    # no snapshot to read, and the windows still apply
    assert not dialog._snapshot.isEnabled() and dialog._snapshot.toolTip()
    assert dialog._windows.isEnabled()
    assert dialog.config()["source_snapshot"] is None


def test_a_behavior_run_does_not_offer_movement_and_says_why(qapp, tmp_path):
    exp = _cohort(tmp_path, {"a": ({"drug": "veh"}, 2), "b": ({"drug": "fent"}, 9)})
    dialog = ComparisonLaunchDialog(exp)
    assert dialog.measure_source() == "labels"
    assert _sources(dialog)["locations"] is False
    assert "location-export run" in dialog._source_reasons()["locations"]
    assert not any(widget.isEnabled() for widget in dialog._location_fields.values())


def test_the_measures_offered_are_the_location_measures(qapp, tmp_path):
    dialog = ComparisonLaunchDialog(cohort(tmp_path, paired_plan()))
    offered = [dialog._measures.item(i).text() for i in range(dialog._measures.count())]
    assert offered == [spec.label for spec in location_measures.MEASURES.values()]
    assert dialog.config()["measures"] is None  # none ticked: all of them
    _set_checked(dialog._measures, {"distance", "body_length"})
    assert dialog.config()["measures"] == ["distance", "body_length"]


def test_the_read_settings_open_on_the_models_defaults_and_only_for_this_source(
    qapp, tmp_path
):
    dialog = ComparisonLaunchDialog(cohort(tmp_path, paired_plan()))
    assert set(dialog._location_fields) == set(_LOCATION_FIELDS)
    # every field of the model that starts with `location_` has a row here
    assert set(_LOCATION_FIELDS) == {
        name for name in ComparisonConfig.model_fields if name.startswith("location_")
    }
    config = dialog.config()
    for name in _LOCATION_FIELDS:
        assert config[name] == ComparisonConfig.model_fields[name].default, name
        assert dialog._location_fields[name].isEnabled(), name
        assert dialog._location_fields[name].toolTip(), name


def test_the_read_settings_come_back_as_they_were_given_including_every_row(
    qapp, tmp_path
):
    dialog = ComparisonLaunchDialog(cohort(tmp_path, paired_plan()))
    wanted = {
        "location_sample_fps": None,  # every row: a value, not an absence
        "location_smooth_s": 2.0,
        "location_moving_threshold": 3.5,
        "location_jump_speed": 12.0,
    }
    dialog.load_config(wanted)
    assert {name: dialog.config()[name] for name in wanted} == wanted
    assert dialog._location_fields["location_sample_fps"].text() == "every row"
    dialog.load_config({"location_sample_fps": 10.0})
    assert dialog.config()["location_sample_fps"] == 10.0


def test_what_the_form_builds_runs_and_seeds_the_next_one(qapp, tmp_path):
    exp = cohort(tmp_path, paired_plan())
    dialog = ComparisonLaunchDialog(exp)
    _shaped(dialog)
    assert dialog.validation_error() is None
    assert "paired" in dialog._design_note.text()
    config = dialog.config()
    ComparisonConfig.model_validate(config)  # what the form emits is a whole config
    comp = exp.run_comparison(config)
    record = comp.record()
    assert record["source"] == "locations" and record["design"] == "paired"
    seeded = ComparisonLaunchDialog(exp, seed=record)
    assert seeded.measure_source() == "locations"
    assert _checked(seeded._group_by) == ["session"]
    for name in _LOCATION_FIELDS:
        assert seeded.config()[name] == record[name], name


@pytest.fixture
def location_page(qapp, tmp_path):
    exp = cohort(tmp_path, paired_plan())
    comp = exp.run_comparison(PAIRED)
    page = ComparisonDetailPage(exp, comp.id)
    drain_busy_calls(qapp)
    return page


def _texts(page) -> str:
    return "\n".join(label.text() for label in page.findChildren(QLabel))


def test_the_detail_page_says_each_measure_is_tested_on_its_own(location_page):
    text = _texts(location_page)
    assert "Each measure is tested on its own" in text
    assert "Floor (95th percentile" not in text and "No permutation floor" not in text
    assert "movement and body shape from location export an_01" in text
    assert "thin parts removed" in text
    # ordered by evidence, and the page says so rather than "ranked on the effect"
    assert "ordered by each measure's own test" in text
    assert "ranked on the effect" not in text


def test_the_detail_page_shows_the_verdict_by_each_measures_own_p(location_page):
    tables = {t.objectName(): t for t in location_page.findChildren(QTableWidget)}
    table = next(
        t
        for t in tables.values()
        if "p ≤ 0.05"
        in [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    )
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert "p (own · q)" in headers and "clears floor" not in headers
    names = [table.item(r, 0).text() for r in range(table.rowCount())]
    verdict = headers.index("p ≤ 0.05")
    by_name = {
        table.item(r, 0).text(): table.item(r, verdict).text()
        for r in range(table.rowCount())
    }
    assert by_name["Body length"] == "yes" and by_name["Body width"] == "no"
    assert not any("_" in name for name in names)  # labels, never raw ids


def test_the_detail_page_renders_at_a_laptop_size(location_page, qapp, tmp_path):
    location_page.resize(1100, 700)
    location_page.show()
    qapp.processEvents()
    picture = tmp_path / "locations_detail.png"
    assert location_page.grab().save(str(picture))
    assert picture.stat().st_size > 5_000  # a page with content, not a blank pixmap


# --- what the form is described over ----------------------------------------------------- #
#: Animals 0 to 3, both sessions: what the second export covers.
FIRST_FOUR = [f"{s}{a}" for a in range(4) for s in ("s", "d")]


def _partial_exports(tmp_path):
    """Sixteen recordings in an_01's export, eight of them in an_02's, and one recording in
    the experiment that neither covers, with a session and a batch nothing else has."""
    plan = paired_plan()
    exp = cohort(tmp_path, plan)
    for recording in plan:
        batch = "1" if recording in FIRST_FOUR else "2"
        md.upsert_row(exp.dir, recording, {**plan[recording][0], "batch": batch})
    exp.create_recording("x1")
    md.upsert_row(exp.dir, "x1", {"animal": "Z", "session": "other", "batch": "3"})
    run_dir = analysis_runs.analysis_dir(exp.dir, "an_02")
    write_run(run_dir, {r: {0: plan[r][1]} for r in FIRST_FOUR})
    analysis_runs.write_analysis_yaml(
        run_dir,
        {"id": "an_02", "module_id": "location_export", "status": "complete",
         "recording_ids": FIRST_FOUR, "params": dict(EXPORTED),
         "per_recording": {r: {"status": "complete"} for r in FIRST_FOUR}},
    )  # fmt: skip
    return exp


def _row(listwidget, name: str):
    (item,) = [
        listwidget.item(i)
        for i in range(listwidget.count())
        if listwidget.item(i).data(Qt.ItemDataRole.UserRole) == name
    ]
    return item


def _described(exp, recordings, column: str) -> str:
    summary = cohort_lib.describe_columns(
        exp.recording_metadata, list(recordings), [column]
    )[0]
    return summary.usable_for_grouping or summary.describe()


def test_the_columns_are_described_over_the_recordings_the_run_covers(qapp, tmp_path):
    exp = _partial_exports(tmp_path)
    dialog = ComparisonLaunchDialog(exp)
    assert dialog.source_run_id() == "an_02"  # the newest
    for column in ("session", "animal"):
        text = _row(dialog._group_by, column).text()
        assert text.endswith(_described(exp, FIRST_FOUR, column)), text
        assert "other" not in text and "Z" not in text.split("—")[-1]
    # one batch among these eight: it cannot define groups here, and says so
    batch = _row(dialog._group_by, "batch")
    assert not batch.flags() & Qt.ItemFlag.ItemIsEnabled
    # a recording the run did not export is greyed, says why, and is never sent
    outside = _row(dialog._recordings, "s5")
    assert not outside.flags() & Qt.ItemFlag.ItemIsEnabled
    assert outside.text().endswith("(not in an_02)")
    assert _row(dialog._recordings, "s1").flags() & Qt.ItemFlag.ItemIsEnabled
    _shaped(dialog)
    assert "an_02 covers 8 of this experiment's 17 recordings" in dialog._summary.text()
    assert "2 group(s) over 8 recording(s)" in dialog._summary.text()
    averaged = _row(dialog._aggregate_to, "animal").text()
    assert averaged.endswith(_described(exp, FIRST_FOUR, "animal")), averaged


def test_choosing_another_run_describes_its_recordings_and_keeps_the_ticks(
    qapp, tmp_path
):
    exp = _partial_exports(tmp_path)
    dialog = ComparisonLaunchDialog(exp)
    _shaped(dialog)
    dialog._source.setCurrentIndex(dialog._source.findData("an_01"))
    everything = list(paired_plan())
    assert (
        _row(dialog._group_by, "session")
        .text()
        .endswith(_described(exp, everything, "session"))
    )
    assert _row(dialog._group_by, "batch").flags() & Qt.ItemFlag.ItemIsEnabled
    assert _checked(dialog._group_by) == ["session"]
    assert _row(dialog._recordings, "s5").text() == "s5"  # offered again, as named
    assert _row(dialog._recordings, "x1").text().endswith("(not in an_01)")
    assert "covers 16 of this experiment's 17" in dialog._summary.text()
    # and back: what an_02 does not cover drops out of a selection made over an_01
    from detail.gui.recording_items import select_recording_ids

    select_recording_ids(dialog._recordings, ["s1", "s5"])
    dialog._source.setCurrentIndex(dialog._source.findData("an_02"))
    assert dialog.config()["recordings"] == ["s1"]


def test_a_behavior_result_is_described_over_its_coverage_too(qapp, tmp_path):
    """The rule is not the locations source's: a behavior result over some recordings."""
    exp = _cohort(tmp_path, CROSSOVER)
    exp.create_recording("x1")
    md.upsert_row(exp.dir, "x1", {"animal": "Z", "drug": "other"})
    dialog = ComparisonLaunchDialog(exp)
    assert dialog.measure_source() == "labels"
    text = _row(dialog._group_by, "drug").text()
    assert text.endswith(_described(exp, list(CROSSOVER), "drug")), text
    assert "other" not in text
    assert _row(dialog._recordings, "x1").text().endswith("(not in an_01)")
