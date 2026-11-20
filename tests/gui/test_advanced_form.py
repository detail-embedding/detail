"""Advanced Settings: the collapsible section, the routing, and a fail-closed guard.

The section is where a form puts the knobs a user rarely touches. Three things have to hold
or the surface does more harm than the wall of spin boxes it replaces:

- what is hidden is *declared* by the model, not decided form by form — so the guard below
  builds every config form and fails if a field flagged ``advanced`` is sitting in a main
  body, which is what silently happens when a form is written without ``ConfigForm``;
- hiding a row never changes the run — ``config()`` reads the same collapsed or expanded;
- a hidden value that differs from what the form opened with says so, and opens itself.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFormLayout, QWidget  # noqa: E402

from detail.api._runner_modules import PIPELINE_RUNNERS  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.configs.temporal_dtw_finch import TemporalDtwFinchConfig  # noqa: E402
from detail.configs.describe import describe_config, iter_config_models  # noqa: E402
from detail.configs.detection import DetectionConfig  # noqa: E402
from detail.executor.runners import import_runner_modules  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.forms import AdvancedSection, ConfigForm  # noqa: E402
from detail.gui.help_ui import FieldLabel, field_label  # noqa: E402
from detail.gui._pydantic_form import (  # noqa: E402
    field_widget,
    optional_spin,
)
from detail.gui.modules.registry import (  # noqa: E402
    ANALYSIS_GUI_MODULES,
    GUI_MODULES,
)
from detail.gui.spin import IntSpin  # noqa: E402
from detail.gui.tools.registry import TOOL_GUIS, import_tool_uis  # noqa: E402


def _spin(value: int) -> IntSpin:
    spin = IntSpin()
    spin.setRange(0, 100)
    spin.setValue(value)
    return spin


# --------------------------------------------------------------------------- #
# The section
# --------------------------------------------------------------------------- #
def test_a_section_starts_collapsed_and_counts_what_it_hides(qapp):
    host = QWidget()
    form = ConfigForm(host)
    form.addRow("Batch size:", _spin(8), advanced=True)
    form.addRow("Workers:", _spin(2), advanced=True)
    form.finish()

    section = form.advanced()
    assert not section.is_expanded()
    assert not section._body.isVisible()
    assert section._header.text() == "▶  Advanced Settings (2)"


def test_the_header_spans_the_panel(qapp):
    """The toggle is the width of what it opens, not a word tucked into the left margin."""
    from PyQt6.QtWidgets import QSizePolicy

    host = QWidget()
    form = ConfigForm(host)
    form.addRow("Batch size:", _spin(8), advanced=True)
    form.finish()

    assert (
        form.advanced()._header.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Expanding
    )


def test_the_header_is_set_in_the_same_type_as_the_parameter_labels(qapp):
    """Reproduces the platform trap that shipped a visibly smaller header.

    Qt lets a platform register a default font *per widget class*, and macOS gives
    ``QToolButton`` 10pt against ``QLabel``'s 13 — so the header rendered smaller than the
    parameter labels beside it while every font comparison in a test passed. It passed
    because the offscreen platform CI and the suite run under registers no per-class fonts
    at all, and because an unpolished widget still reports the application font.

    So the condition is created here rather than waited for: register a small class font,
    build the section, and require the header to have kept the label's size.
    """
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

    reference = QApplication.font("QLabel")
    originals = {cls: QApplication.font(cls) for cls in ("QToolButton", "QPushButton")}
    try:
        for cls in originals:
            shrunk = QFont(reference)
            shrunk.setPointSizeF(reference.pointSizeF() - 3)
            QApplication.setFont(shrunk, cls)

        host = QWidget()
        form = ConfigForm(host)
        form.addRow("Batch size:", _spin(8), advanced=True)
        form.finish()
        header = form.advanced()._header
        header.ensurePolished()  # class fonts are applied on polish, not construction

        assert header.font().pointSizeF() == reference.pointSizeF()
        assert header.font().bold() == reference.bold()
        # ...and the guard itself works: an unpinned button of either class does shrink.
        for cls in (QPushButton, QLabel):
            probe = cls()
            probe.ensurePolished()
        bare = QPushButton()
        bare.ensurePolished()
        assert bare.font().pointSizeF() < reference.pointSizeF()
    finally:
        for cls, font in originals.items():
            QApplication.setFont(font, cls)


def test_a_note_row_is_not_counted_as_a_parameter(qapp):
    """A warning or note added with an empty label is not a knob — counting it would tell
    the user there is more behind the header than there is."""
    host = QWidget()
    form = ConfigForm(host)
    form.addRow("Batch size:", _spin(8), advanced=True)
    form.advanced().form().addRow("", QWidget())
    form.finish()

    assert form.advanced()._header.text() == "▶  Advanced Settings (1)"


def test_toggling_the_header_shows_and_hides_the_body(qapp):
    host = QWidget()
    form = ConfigForm(host)
    form.addRow("Batch size:", _spin(8), advanced=True)
    form.finish()
    section = form.advanced()

    section._header.click()
    assert section.is_expanded()
    assert section._header.text().startswith("▼")
    section._header.click()
    assert not section.is_expanded()
    assert section._header.text().startswith("▶")


def test_the_header_reports_a_value_changed_since_the_form_opened(qapp):
    host = QWidget()
    form = ConfigForm(host)
    batch = _spin(8)
    form.addRow("Batch size:", batch, advanced=True)
    form.addRow("Workers:", _spin(2), advanced=True)
    form.finish()

    batch.setValue(64)
    assert form.advanced()._header.text() == "▶  Advanced Settings (2 · 1 changed)"
    batch.setValue(8)  # back to what it opened with: no longer a change
    assert form.advanced()._header.text() == "▶  Advanced Settings (2)"


def test_an_unset_optional_field_is_not_a_change(qapp):
    """``optional_spin`` parks "unset" one step below the field's floor. Read as a number
    that would be a change on every open; read as ``None`` (what ``optional_value`` returns)
    it is the state the form opened in."""
    host = QWidget()
    form = ConfigForm(host)
    form.addRow(
        "Max bouts:",
        optional_spin(TemporalDtwFinchConfig, "max_segments", "all"),
        advanced=True,
    )
    form.finish()

    assert form.advanced().changed_count() == 0


def test_a_section_holding_a_changed_value_opens_itself_on_first_show(qapp):
    """The Refine path: a seed fills the form before the dialog is shown, so the parameter
    that made the old run different has to be on screen, not folded away."""
    host = QWidget()
    form = ConfigForm(host)
    batch = _spin(8)
    form.addRow("Batch size:", batch, advanced=True)
    form.finish()

    batch.setValue(64)  # as load_config would, before the dialog is shown
    host.show()
    try:
        assert form.advanced().is_expanded()
    finally:
        host.hide()


def test_an_unchanged_section_stays_collapsed_on_show(qapp):
    host = QWidget()
    form = ConfigForm(host)
    form.addRow("Batch size:", _spin(8), advanced=True)
    form.finish()

    host.show()
    try:
        assert not form.advanced().is_expanded()
    finally:
        host.hide()


def test_a_section_the_user_collapsed_stays_collapsed(qapp):
    host = QWidget()
    form = ConfigForm(host)
    batch = _spin(8)
    form.addRow("Batch size:", batch, advanced=True)
    form.finish()
    section = form.advanced()

    section._header.click()  # opened by hand...
    section._header.click()  # ...and closed again
    batch.setValue(64)
    host.show()
    try:
        assert not section.is_expanded()  # the user's choice wins over the auto-open
    finally:
        host.hide()


def test_an_empty_section_is_hidden(qapp):
    """A form can ask for a section and then add nothing to it (a tracker with no knobs).
    A header over nothing is a dead control."""
    host = QWidget()
    form = ConfigForm(host)
    form.section()  # created, never filled
    form.finish()

    assert form.advanced().isHidden()


def test_clearing_a_section_rebuilds_its_count(qapp):
    """Tracking rebuilds its knobs whenever the tracker changes."""
    host = QWidget()
    form = ConfigForm(host)
    form.addRow("Old knob:", _spin(1), advanced=True)
    form.finish()
    assert form.advanced()._header.text() == "▶  Advanced Settings (1)"

    form.advanced().clear()
    form.advanced().form().addRow("New knob:", _spin(2))
    form.advanced().form().addRow("Another:", _spin(3))
    form.finish()
    assert form.advanced()._header.text() == "▶  Advanced Settings (2)"


def test_switching_tracker_rebuilds_its_knob_section(qapp):
    """The real rebuild path, through the tracking form.

    Its knobs are wiped and re-added whenever the tracker changes. The section has to end up
    describing the *new* tracker — not a stale count, and not an exception from reading the
    widgets Qt just deleted out from under the baseline.
    """
    from detail.gui.modules.tracking import _KNOBS

    widget = GUI_MODULES["tracking"].config_widget()
    section = widget.findChild(AdvancedSection)
    assert _KNOBS in section.groups()  # the knobs sit under their own heading

    widget._tracker.setCurrentText("anchor_tracker")
    anchor_knobs = section.labeled_rows()
    widget._hp_widgets["conf_threshold"].setValue(0.75)
    assert "changed" in section._header.text()

    # A tracker the form offers. `setCurrentText` past a combo that does not hold the id is a
    # silent no-op, so naming a hidden one (§25) would leave the form on `anchor_tracker` and
    # quietly test nothing.
    widget._tracker.setCurrentText("iou_single_tracker")
    assert widget._tracker.currentText() == "iou_single_tracker"
    # The knobs, plus the one row above them that is not a knob: *Leave out classes* (§13).
    assert section.labeled_rows() == len(widget._hp_widgets) + 1
    assert (
        section.labeled_rows() != anchor_knobs
    )  # a different tracker, different knobs
    assert (
        "changed" not in section._header.text()
    )  # the new knobs are at their defaults


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_a_field_label_row_routes_by_the_model(qapp):
    host = QWidget()
    form = ConfigForm(host, DetectionConfig)
    conf = field_widget(DetectionConfig, "conf_threshold")
    batch = field_widget(DetectionConfig, "batch_size")  # the model flags this one
    form.addRow(field_label("detection", "conf_threshold", "Confidence:"), conf)
    form.addRow(field_label("detection", "batch_size", "Batch size:"), batch)
    form.finish()

    assert form.main.getWidgetPosition(conf)[0] >= 0
    assert form.main.getWidgetPosition(batch)[0] < 0
    assert form.advanced().form().getWidgetPosition(batch)[0] >= 0


def test_a_named_group_gets_its_own_section(qapp):
    host = QWidget()
    form = ConfigForm(
        host, TemporalDtwFinchConfig
    )  # declares Segmentation + Clustering
    for name in ("pool", "max_segments"):
        form.addRow(field_label("dtw_finch", name, f"{name}:"), _spin(1))
    form.finish()

    assert sorted(form.advanced().groups()) == ["Clustering", "Segmentation"]
    assert len(host.findChildren(AdvancedSection)) == 1  # one toggle, two headings


def test_an_explicit_flag_overrides_the_model(qapp):
    """The escape hatch for a row with no model field — and for a form that means it."""
    host = QWidget()
    form = ConfigForm(host, DetectionConfig)
    conf = field_widget(DetectionConfig, "conf_threshold")  # not flagged in the model
    form.addRow(
        field_label("detection", "conf_threshold", "Confidence:"), conf, advanced=True
    )
    form.finish()

    assert form.advanced().form().getWidgetPosition(conf)[0] >= 0


def test_a_label_the_model_does_not_describe_stays_in_the_main_body(qapp):
    """Tracker knobs carry a ``field_label`` naming something ``TrackingConfig`` has no field
    for. That must read as "not advanced", not raise."""
    host = QWidget()
    form = ConfigForm(host, DetectionConfig)
    widget = _spin(1)
    form.addRow(field_label("detection", "not_a_field", "Invented:"), widget)
    form.finish()

    assert form.main.getWidgetPosition(widget)[0] >= 0


def test_a_row_is_reachable_wherever_it_landed(qapp):
    """A form that greys or hides a row by its field widget must not care which layout the
    row is in — ``main.labelForField`` returns nothing for a row that moved into a section.
    """
    host = QWidget()
    form = ConfigForm(host, DetectionConfig)
    batch = field_widget(DetectionConfig, "batch_size")
    form.addRow(field_label("detection", "batch_size", "Batch size:"), batch)
    form.finish()

    label = form.label_for(batch)
    assert isinstance(label, FieldLabel)
    form.set_row_visible(batch, False)
    assert not batch.isVisible()


# --------------------------------------------------------------------------- #
# The guard: no flagged field may sit in a form's main body
# --------------------------------------------------------------------------- #
# (form_id, module_id, field) rows that deliberately show a flagged field. Each needs a
# reason — the field is this form's whole subject, not a knob it happens to expose.
EXEMPT: set[tuple[str, str, str]] = set()


def _registry_forms() -> dict:
    import_runner_modules(PIPELINE_RUNNERS)  # populate TRACKERS (tracking's knob rows)
    import_tool_uis()
    return {**GUI_MODULES, **ANALYSIS_GUI_MODULES, **TOOL_GUIS}


# The launch surfaces outside the module registries — the guard has to reach them too, and
# a form that opted *out* of ConfigForm is exactly what it exists to catch.
_DIALOGS = (
    "GenerateDatasetDialog",
    "GenerateLabelingDialog",
    "TrainModelDialog",
    "DistillDialog",
    # The launch dialog. It was not listed, and did not use `ConfigForm` — which is how the
    # comparison dialog shipped with its window knobs on the front of the form and nothing
    # failed. A launch dialog is a config form; the list has to say so.
    "ComparisonLaunchDialog",
)


def _build_dialog(name: str, tmp_path) -> QWidget:
    from detail.gui import fit as gui_fit

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    if name == "ComparisonLaunchDialog":
        from detail.gui import comparison as gui_comparison

        return gui_comparison.ComparisonLaunchDialog(experiment)
    return getattr(gui_fit, name)(experiment)


def _config_models() -> dict:
    return {module_id: model for module_id, model in iter_config_models()}


def _advanced_fields(module_id: str, models: dict) -> set[str]:
    model = models.get(module_id)
    if model is None:
        return set()
    return {p["name"] for p in describe_config(model) if p["advanced"]}


def _main_body_fields(widget: QWidget) -> list[tuple[str, str]]:
    """``(module_id, field)`` for every ``field_label`` row *outside* a collapsible section."""
    in_sections = [
        form
        for section in widget.findChildren(AdvancedSection)
        for form in section._forms()
    ]
    found = []
    for layout in widget.findChildren(QFormLayout):
        if any(layout is form for form in in_sections):
            continue
        for row in range(layout.rowCount()):
            item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            label = item.widget() if item else None
            if isinstance(label, FieldLabel):
                found.append((label.module_id, label.field))
    return found


def _offenders(form_id: str, widget: QWidget, models: dict) -> list[str]:
    return [
        f"{module_id}.{field}"
        for module_id, field in _main_body_fields(widget)
        if field in _advanced_fields(module_id, models)
        and (form_id, module_id, field) not in EXEMPT
    ]


@pytest.mark.parametrize("form_id", sorted(_registry_forms()))
def test_no_advanced_field_sits_in_a_module_form_body(qapp, form_id):
    widget = _registry_forms()[form_id].config_widget()
    offenders = _offenders(form_id, widget, _config_models())
    assert not offenders, (
        f"{form_id} shows model fields flagged advanced in its main body: {offenders}. "
        "Build the form with gui/forms.py::ConfigForm (which routes them), or add an "
        "EXEMPT entry with the reason the field belongs on the front of this form."
    )


@pytest.mark.parametrize("form_id", _DIALOGS)
def test_no_advanced_field_sits_in_a_dialog_body(qapp, tmp_path, form_id):
    widget = _build_dialog(form_id, tmp_path)
    offenders = _offenders(form_id, widget, _config_models())
    assert not offenders, (
        f"{form_id} shows model fields flagged advanced in its main body: {offenders}. "
        "Build the form with gui/forms.py::ConfigForm (which routes them), or add an "
        "EXEMPT entry with the reason the field belongs on the front of this form."
    )


def test_the_guard_catches_a_flagged_field_left_in_the_main_body(qapp):
    """The guard itself. A fail-closed check that cannot fail is decoration — this is the
    shape it exists to catch: a form built on a plain QFormLayout, so a flagged field is
    rendered like any other row."""
    host = QWidget()
    layout = QFormLayout(host)
    layout.addRow(
        field_label("detection", "batch_size", "Batch size:"),
        field_widget(DetectionConfig, "batch_size"),
    )

    assert _offenders("a_hand_rolled_form", host, _config_models()) == [
        "detection.batch_size"
    ]


def test_exempt_entries_are_all_still_live(qapp, tmp_path):
    """An EXEMPT entry whose row moved or vanished is stale — it would quietly cover a
    different field later."""
    models = _config_models()
    live = set()
    for form_id in _DIALOGS:
        widget = _build_dialog(form_id, tmp_path / form_id)
        for module_id, field in _main_body_fields(widget):
            if field in _advanced_fields(module_id, models):
                live.add((form_id, module_id, field))
    for form_id, ui in _registry_forms().items():
        for module_id, field in _main_body_fields(ui.config_widget()):
            if field in _advanced_fields(module_id, models):
                live.add((form_id, module_id, field))
    assert not (EXEMPT - live), f"stale EXEMPT entries: {EXEMPT - live}"


# --------------------------------------------------------------------------- #
# Hiding a row must never change the run
# --------------------------------------------------------------------------- #
def _make_config_readable(widget: QWidget, tmp_path) -> None:
    """Give a form the input its ``config()`` needs before it will read.

    Quick segment's config indexes its chosen videos, so an empty picker raises rather than
    returning a config — nothing to do with visibility, but it has to be fed to ask the
    question below.
    """
    videos = getattr(widget, "_videos", None)
    if videos is not None and videos.count() == 0:
        videos.addItem(str(tmp_path / "clip.mp4"))


@pytest.mark.parametrize("form_id", sorted(_registry_forms()))
def test_config_reads_the_same_collapsed_or_expanded(qapp, tmp_path, form_id):
    widget = _registry_forms()[form_id].config_widget()
    sections = widget.findChildren(AdvancedSection)
    if not sections:
        pytest.skip(f"{form_id} has no advanced section")
    _make_config_readable(widget, tmp_path)
    collapsed = widget.config()
    for section in sections:
        section.set_expanded(True)
    assert widget.config() == collapsed
