"""The shared number boxes, and two guards that keep every form using them.

Qt pads a spin box's text to exactly ``decimals`` places and its validator rejects any
candidate carrying more, so a 4-decimal confidence box shows ``0.5000`` and then swallows
every digit typed into it. ``gui/spin.py`` fixes that once; these tests pin each half of the
fix, and the guards stop a new form from hand-building a box that misses it.
"""

from pathlib import Path

import pytest

import detail
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.detection import DetectionConfig
from detail.configs.labeling import LabelingConfig
from detail.configs.query import QueryConfig
from detail.configs.temporal_dtw_finch import TemporalDtwFinchConfig
from detail.executor.runners import import_runner_modules
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui._pydantic_form import (
    field_slider,
    field_widget,
    optional_spin,
    optional_value,
    set_optional_value,
)
from detail.gui.modules.registry import (
    ANALYSIS_GUI_MODULES,
    GUI_MODULES,
)
from detail.gui.qt import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QEvent,
    QFocusEvent,
    QPoint,
    QPointF,
    QValidator,
    QWheelEvent,
    Qt,
)
from detail.gui.spin import (
    FRACTION,
    SECONDS,
    FloatSpin,
    IntSpin,
    ThresholdSlider,
)
from detail.gui.tools.registry import TOOL_GUIS, import_tool_uis


def _fraction_box() -> FloatSpin:
    spin = FloatSpin(FRACTION)
    spin.setRange(0.0, 1.0)
    spin.setValue(0.5)
    return spin


# --------------------------------------------------------------------------- #
# The reported bug: you could not type into a confidence threshold
# --------------------------------------------------------------------------- #
def test_the_display_trims_its_padding_zeros(qapp):
    spin = _fraction_box()
    assert spin.lineEdit().text() == "0.5"  # not "0.5000"
    for value, shown in (
        (0.05, "0.05"),
        (1.0, "1.0"),
        (0.0, "0.0"),
        (0.1234, "0.1234"),
    ):
        spin.setValue(value)
        assert spin.lineEdit().text() == shown
        assert spin.value() == value  # the trim is display-only


def test_a_digit_typed_into_a_full_field_is_accepted(qapp):
    """The whole point. At 4 decimals Qt renders 0.5000 and rejects anything longer, so
    every keystroke was swallowed until you deleted part of the value first."""
    spin = _fraction_box()
    padded = QDoubleSpinBox()
    padded.setRange(0.0, 1.0)
    padded.setDecimals(4)
    padded.setValue(0.5)

    typed = spin.lineEdit().text() + "5"  # append a digit to what is on screen
    assert spin.validate(typed, len(typed))[0] == QValidator.State.Acceptable
    # ...whereas appending to the padded rendering is rejected outright
    padded_typed = padded.lineEdit().text() + "5"
    assert padded.validate(padded_typed, len(padded_typed))[0] == (
        QValidator.State.Invalid
    )


def test_stepping_and_typing_still_round_trip(qapp):
    spin = _fraction_box()
    spin.stepBy(1)
    assert spin.value() == pytest.approx(0.55)
    assert spin.lineEdit().text() == "0.55"
    spin.lineEdit().setText("0.75")
    spin.interpretText()
    assert spin.value() == pytest.approx(0.75)


def test_the_box_is_no_narrower_than_a_full_precision_entry(qapp):
    """Qt sizes a spin box from its displayed min/max text, so trimming alone shrank a 0-1
    box from 72 px to 49 px — too narrow to see 0.9999 while typing it."""
    trimmed = _fraction_box()
    padded = QDoubleSpinBox()
    padded.setRange(0.0, 1.0)
    padded.setDecimals(4)
    assert trimmed.sizeHint().width() >= padded.sizeHint().width()
    assert trimmed.minimumSizeHint().width() >= padded.minimumSizeHint().width()


def test_focusing_selects_the_value_so_typing_replaces_it(qapp):
    spin = _fraction_box()
    spin.focusInEvent(QFocusEvent(QEvent.Type.FocusIn, Qt.FocusReason.MouseFocusReason))
    qapp.processEvents()  # the select is deferred past the mouse press
    assert spin.lineEdit().selectedText() == "0.5"


def _wheel(widget) -> QWheelEvent:
    return QWheelEvent(
        QPointF(5, 5),
        widget.mapToGlobal(QPoint(5, 5)).toPointF(),
        QPoint(0, 120),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_an_unfocused_box_ignores_the_wheel(qapp):
    """Scrolling a long config dialog must not shift a threshold under the pointer."""
    spin = _fraction_box()
    counts = IntSpin()
    counts.setRange(0, 512)
    counts.setValue(32)
    for widget, before in ((spin, 0.5), (counts, 32)):
        widget.clearFocus()
        qapp.sendEvent(widget, _wheel(widget))
        assert widget.value() == before


# --------------------------------------------------------------------------- #
# field_widget: one field's widget, entirely from the model
# --------------------------------------------------------------------------- #
def test_field_widget_takes_its_range_and_help_from_the_model(qapp):
    spin = field_widget(DetectionConfig, "conf_threshold")
    assert isinstance(spin, FloatSpin)
    assert (spin.minimum(), spin.maximum()) == (0.0, 1.0)  # from ge/le
    assert spin.toolTip() == DetectionConfig.model_fields["conf_threshold"].description


def test_field_widget_infers_the_kind_from_the_field(qapp):
    """A 0-1 field is a fraction, a seconds field is a duration, an int is a count — decided
    once here rather than at each call site."""
    assert field_widget(DetectionConfig, "conf_threshold").singleStep() == 0.05
    assert isinstance(field_widget(DetectionConfig, "every_n_frames"), IntSpin)
    band = field_widget(TemporalDtwFinchConfig, "band_s")
    # Three decimals, because this project's durations routinely run shorter than a frame
    # (one frame at 59 Hz is 0.017 s) and a two-decimal cap cannot express one — it was
    # rounding `abd_smooth_half_s`, 0.034, down to 0.03 on the way into the form.
    assert band.decimals() == 3 and band.singleStep() == pytest.approx(0.1)


def test_an_exclusive_bound_is_stepped_inside(qapp):
    """``val_split`` is ``lt=1.0``. The box has to stop below 1.0 — it cannot offer a value
    the validator would reject — and the bound used to be missed entirely, leaving the
    generated widget ranged 0 to a billion."""
    spin = field_widget(LabelingConfig, "val_split")
    assert spin.maximum() < 1.0


def test_ui_max_caps_the_display_without_moving_the_model_bound(qapp):
    spin = field_widget(DetectionConfig, "batch_size")
    assert spin.maximum() == 512  # the declared ui_max
    assert DetectionConfig.model_fields["batch_size"].metadata  # ge=1, no le
    DetectionConfig(model_id="sam3", batch_size=4096)  # the model still accepts more


# --------------------------------------------------------------------------- #
# optional_spin: one "unset" convention instead of four
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model, field",
    [
        (DetectionConfig, "batch_size"),  # int, ge=1
        (QueryConfig, "trajectory_split_gap_s"),  # float, ge=0
    ],
)
def test_optional_spin_round_trips_none_through_its_sentinel(qapp, model, field):
    spin = optional_spin(model, field, "(auto)")
    assert optional_value(spin) is None
    assert spin.specialValueText() == "(auto)"
    set_optional_value(spin, 5)
    assert optional_value(spin) == 5
    set_optional_value(spin, None)
    assert optional_value(spin) is None


def test_a_meaningful_zero_stays_reachable(qapp):
    """The sentinel sits one step *below* the field's floor, not on 0 — a split gap of 0
    (never split) is a real value, and three modules used to make it mean "auto"."""
    spin = optional_spin(QueryConfig, "trajectory_split_gap_s", "(auto)")
    spin.setValue(0.0)
    assert optional_value(spin) == 0.0


# --------------------------------------------------------------------------- #
# Guard 1: no form hand-builds a bare Qt spin box
# --------------------------------------------------------------------------- #
_GUI = Path(detail.__file__).parent / "gui"

# (file relative to gui/, construction) still building a raw Qt spin box, with a reason.
# Expected to stay empty: use IntSpin / FloatSpin, or field_widget / optional_spin when a
# config model describes the parameter.
ALLOWED: set[tuple[str, str]] = set()


def _bare_constructions() -> set[tuple[str, str]]:
    found = set()
    for path in _GUI.rglob("*.py"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            for call in ("QSpinBox()", "QDoubleSpinBox()"):
                if call in line:
                    found.add((str(path.relative_to(_GUI)), call))
    return found


def test_no_gui_file_builds_a_bare_qt_spin_box():
    offenders = _bare_constructions() - ALLOWED
    assert not offenders, (
        "these build a raw Qt spin box, which pads its display and then refuses the digits "
        "you type into it. Use detail/gui/spin.py's IntSpin / FloatSpin, or field_widget / "
        f"optional_spin when a config model describes the parameter: {sorted(offenders)}"
    )


def test_the_allow_list_has_no_stale_entries():
    stale = ALLOWED - _bare_constructions()
    assert not stale, f"ALLOWED entries that no longer exist: {sorted(stale)}"


def test_the_scan_would_actually_catch_one():
    """Guard the guard: a source scan that matches nothing passes vacuously forever."""
    assert "QDoubleSpinBox()" in "        self._conf = QDoubleSpinBox()"


# --------------------------------------------------------------------------- #
# Guard 2: every built form's spin boxes are the shared ones
# --------------------------------------------------------------------------- #
def _all_forms() -> dict[str, object]:
    import_runner_modules(PIPELINE_RUNNERS)  # populate TRACKERS (tracking knob rows)
    import_tool_uis()
    return {**GUI_MODULES, **ANALYSIS_GUI_MODULES, **TOOL_GUIS}


@pytest.mark.parametrize("form_id", sorted(_all_forms()))
def test_every_form_spin_box_is_a_shared_one(qapp, form_id):
    """The source scan cannot see a row built at runtime — the tracker hyperparameters are
    generated from the selected tracker's schema, so they only exist once the form is.
    """
    widget = _all_forms()[form_id].config_widget()
    strangers = [
        type(spin).__name__
        for spin in widget.findChildren(QAbstractSpinBox)
        if not isinstance(spin, (IntSpin, FloatSpin))
    ]
    assert not strangers, f"{form_id} has non-shared spin boxes: {strangers}"


def test_a_tracker_knob_can_reach_a_value_the_tracker_accepts(qapp):
    """``iou_weight`` defaults to 1.0, and the form used to guess its range as
    ``min(0, default)..max(1, default)`` — pinning the box at the default and making a
    value the CLI and API take (validate_hyperparams checks types, not ranges)
    unreachable in the GUI."""
    from detail.gui.modules.tracking import _TrackingConfig

    widget = _TrackingConfig()
    widget._tracker.setCurrentText("iou_single_tracker")
    assert widget._hp_widgets["iou_weight"].maximum() > 1.0


def test_seconds_and_fraction_boxes_are_distinguishable(qapp):
    """The kind is what makes a duration step by 0.1 and a ratio by 0.05."""
    assert FloatSpin(SECONDS).singleStep() == pytest.approx(0.1)
    assert FloatSpin(FRACTION).singleStep() == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# ThresholdSlider — the shared control for a value you sweep to see its effect
# --------------------------------------------------------------------------- #
def test_field_slider_takes_its_range_step_and_help_from_the_model(qapp):
    from detail.configs.display import ConfidenceView

    slider = field_slider(ConfidenceView, "min_conf")
    assert isinstance(slider, ThresholdSlider)
    assert slider._step == 0.01  # the field's ui_step, not a number typed in the widget
    slider.set_value(1.0)
    assert slider.value() == 1.0
    slider.set_value(0.0)
    assert slider.value() == 0.0
    assert "confidence" in slider.toolTip().lower()  # the field's own help


def test_field_slider_refuses_a_field_it_cannot_lay_out(qapp):
    from detail.configs.detection import DetectionConfig
    from detail.configs.quick_segment import QuickSegmentConfig

    # An integer field: no integer track exists yet, so say so rather than mis-render.
    with pytest.raises(ValueError, match="integer"):
        field_slider(QuickSegmentConfig, "n_frames")
    # A non-scalar field.
    with pytest.raises(ValueError, match="bounded number"):
        field_slider(DetectionConfig, "model_id")


def test_the_slider_and_its_box_stay_in_step_without_looping(qapp):
    from detail.configs.display import ConfidenceView

    slider = field_slider(ConfidenceView, "min_conf")
    slider._slider.setValue(70)  # a drag
    assert slider.value() == pytest.approx(0.70)
    slider._spin.setValue(0.25)  # typing
    assert slider._slider.value() == 25


def test_a_drag_emits_once_when_it_settles_not_once_per_pixel(qapp):
    """Each emit re-renders a frame. Undebounced, a drag across the track queues a
    render per pixel and the picture lags the handle."""
    from detail.configs.display import ConfidenceView

    slider = field_slider(ConfidenceView, "min_conf")
    seen = []
    slider.threshold_changed.connect(seen.append)
    for notch in range(10, 60, 5):  # a drag
        slider._slider.setValue(notch)
    assert seen == [], "a drag must not emit while it is still moving"
    slider._slider.sliderReleased.emit()  # letting go emits immediately
    assert seen == [pytest.approx(0.55)]


def test_set_value_seeds_without_emitting(qapp):
    """Seeding from a caller is not a user edit — emitting would re-render the view
    that is still being built."""
    from detail.configs.display import ConfidenceView

    slider = field_slider(ConfidenceView, "min_conf")
    seen = []
    slider.threshold_changed.connect(seen.append)
    slider.set_value(0.8)
    assert slider.value() == pytest.approx(0.8)
    assert seen == []


def test_an_unfocused_slider_ignores_the_wheel(qapp):
    """Same reason the spin boxes do: scrolling a dialog must not nudge the threshold
    that happens to sit under the pointer."""
    from detail.configs.display import ConfidenceView

    slider = field_slider(ConfidenceView, "min_conf")
    slider.set_value(0.5)
    before = slider.value()
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    slider._slider.wheelEvent(event)
    assert slider.value() == before


def test_the_confidence_row_reads_its_label_and_bounds_from_the_model(qapp):
    from detail.gui.confidence import ConfidenceRow, showing_caption

    row = ConfidenceRow()
    assert row.value() == 0.5  # ConfidenceView's default
    row.set_value(0.25)
    assert row.value() == 0.25
    row.set_caption(showing_caption(3, 214))
    assert row._caption.text() == "showing 3 of 214"


def test_the_confidence_row_says_what_it_thresholds_and_that_it_slides(qapp):
    """Every surface that shows this also carries a FrameScrubber, itself a horizontal
    slider. An unnamed track next to another unnamed track reads as the same control, so
    the heading has to name the thing being thresholded, not describe the effect."""
    from detail.gui.qt import QFrame

    from detail.configs.display import ConfidenceView
    from detail.gui.confidence import HINT, ConfidenceRow

    row = ConfidenceRow()
    heading = row._heading.text()
    assert "confidence" in heading.lower()
    assert heading.startswith(
        ConfidenceView.model_fields["min_conf"].json_schema_extra["label"]
    )
    assert HINT in heading
    # A titled box, so it does not read as another bare scrubber track.
    assert row.frameShape() == QFrame.Shape.StyledPanel
