"""*Estimate…* beside **Min contour area** on the labeling forms, and the window it opens.

Two forms carry the field with a recording picker (**Distill new model**'s detector page and
**Label a dataset**), so both carry the button: a capability added to one embedder is the
recorded way the other goes without (``wire-every-surface``). The button sits in the field's
own row under Advanced Settings, measures the recordings the form has selected (none selected
means all of them, as the run reads it), and **Use this value** writes into the field, which
Advanced Settings then counts as changed.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton  # noqa: E402

from detail.analysis.contour_area import MinAreaEstimate  # noqa: E402
from detail.analysis.contour_area import RecordingAreas  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import fit as fit_gui  # noqa: E402
from detail.gui import min_area_estimate as est_gui  # noqa: E402
from detail.gui.forms import AdvancedSection  # noqa: E402
from tests.analysis.test_contour_area import measured_experiment  # noqa: E402,F401
from tests.gui.conftest import drain_until  # noqa: E402

FORMS = ("distill", "label")


def _form(kind, experiment):
    if kind == "distill":
        return fit_gui.DistillDialog(experiment)
    return fit_gui.GenerateLabelingDialog(experiment)


def _estimate(suggestion=450):
    row = RecordingAreas(
        "recording_001",
        "M5",
        435,
        778.0,
        {1: 953.0, 5: 1000.0, 50: 1926.0, 95: 2500.0, 99: 2600.0},
        3000,
        {50: 106.0, 95: 575.0, 99: 826.0},
    )
    reason = "450 px: ..." if suggestion else "No value separates them: ..."
    return MinAreaEstimate([row], suggestion, 476.5, 106.0, reason, ["a warning"])


@pytest.mark.parametrize("kind", FORMS)
def test_the_button_sits_beside_the_field_under_advanced(qapp, tmp_path, kind):
    dialog = _form(kind, Experiment.create("p", tmp_path))
    row = dialog._min_area_row
    assert row.estimate_button.text() == "Estimate…"
    assert row.estimate_button.parent() is row
    assert dialog._min_area.parent() is row
    node = row.parent()
    while node is not None and not isinstance(node, AdvancedSection):
        node = node.parent()
    assert node is not None, "Min contour area left Advanced Settings"


@pytest.mark.parametrize("kind", FORMS)
@pytest.mark.parametrize("select", [False, True])
def test_estimate_measures_the_selection_and_use_writes_the_field(
    qapp, measured_experiment, monkeypatch, kind, select  # noqa: F811
):
    opened = []

    class Chosen:
        def __init__(self, experiment, recording_ids, *, parent, run_on_open):
            opened.append((experiment, recording_ids, run_on_open))

        def exec(self):
            return True

        def chosen_value(self):
            return 450

    monkeypatch.setattr(est_gui, "MinAreaEstimateDialog", Chosen)
    dialog = _form(kind, measured_experiment)
    expected = None
    if select:
        dialog._recordings.item(0).setSelected(True)
        expected = [dialog._recordings.item(0).data(0x0100)]
    dialog._min_area_row.estimate_button.click()
    ((experiment, recording_ids, run_on_open),) = opened
    assert experiment is measured_experiment
    assert run_on_open is True
    if select:
        assert recording_ids == expected
    else:
        assert recording_ids is None
    assert dialog._min_area.value() == 450


def test_a_value_written_back_counts_as_an_advanced_change(
    qapp, measured_experiment, monkeypatch  # noqa: F811
):
    dialog = fit_gui.DistillDialog(measured_experiment)
    advanced = dialog._form.advanced()
    before = advanced.changed_count()
    dialog._min_area.setValue(450)
    assert advanced.changed_count() == before + 1


def test_the_window_shows_the_table_and_hands_back_the_suggestion(qapp, tmp_path):
    dialog = est_gui.MinAreaEstimateDialog(Experiment.create("p", tmp_path), None)
    assert not dialog._use.isEnabled()
    assert dialog._use.toolTip() == "Nothing has been measured yet"
    dialog.show_estimate(_estimate())
    model = dialog._table.model()
    assert model.rowCount() == 1
    assert model.item(0, 0).text() == "M5"
    assert model.item(0, 3).text() == "953"
    assert dialog._summary.text().startswith("Suggested: 450 px")
    assert not dialog._warnings.isHidden()
    assert dialog._use.isEnabled()
    dialog._use.click()
    assert dialog.chosen_value() == 450


def test_no_suggestion_greys_use_and_says_why(qapp, tmp_path):
    dialog = est_gui.MinAreaEstimateDialog(Experiment.create("p", tmp_path), None)
    dialog.show_estimate(_estimate(suggestion=None))
    assert dialog._summary.text().startswith("No suggestion.")
    assert not dialog._use.isEnabled()
    assert "separates" in dialog._use.toolTip()


def test_the_window_states_what_the_estimate_cannot_see(qapp, tmp_path):
    dialog = est_gui.MinAreaEstimateDialog(Experiment.create("p", tmp_path), None)
    texts = [label.text() for label in dialog.findChildren(type(dialog._summary))]
    assert est_gui.LIMIT_NOTE in texts
    assert "lower bound" in est_gui.LIMIT_NOTE


def test_the_real_estimate_runs_off_the_ui_thread_and_fills_the_window(
    qapp, measured_experiment  # noqa: F811
):
    dialog = est_gui.MinAreaEstimateDialog(measured_experiment, None, run_on_open=True)
    assert not dialog._again.isEnabled()  # running
    drain_until(qapp, lambda: dialog.estimate() is not None, 30_000)
    assert dialog._table.model().rowCount() == 1
    assert dialog._again.isEnabled()
    assert isinstance(dialog._again, QPushButton)
