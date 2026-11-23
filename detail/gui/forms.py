"""Config-form layout: the rows a user reads, and the ones they only sometimes need.

A run's config model can carry twenty-odd parameters. Rendered flat they are a wall of spin
boxes with the three that matter buried in it, so a form keeps its granular knobs behind one
**Advanced Settings** toggle at the bottom, under a divider, and opens showing the rest.

There is exactly one toggle per form. Opening it reveals *everything* it holds at once; a
form with a lot of knobs sub-divides them with a titled rule per group (Segmentation /
Clustering), which is a heading, not a second thing to click.

Which knobs are advanced is the **model's** call, not the form's: a field declares
``json_schema_extra={"advanced": True}`` (or a group title, ``{"advanced": "Clustering"}``,
when one undivided list would be a wall of its own), exactly like its ``label`` / ``unit`` /
``ui_max`` hints. ``ConfigForm.addRow`` recovers the field from the row's ``FieldLabel`` and
routes it, so adopting this in a form is one line — and a *new* form gets it for free, by
flagging the field and writing the row the way every other row is written. Rows with no model
field behind them (a compute target, a tracker's knobs, a trainer's hyperparameters) pass
``advanced=True``, or a group title, at the call site.

Two behaviours belong here rather than to each form, because a hidden setting that silently
differs from what the form opened with is the failure this whole surface could introduce:

- The header counts what it is hiding, and says how much of it has been **changed** —
  ``▶ Advanced Settings (11 · 2 changed)``.
- The section **opens itself** the first time the form is shown while holding a change.
  Refine seeds a form through ``load_config`` before the dialog is shown, so the parameter
  that made the old run different is on screen, not folded away.

"Changed" is measured against the state the form *opened* in, taken in ``finish()`` — not
against the model default, because forms legitimately open on something else (detection opens
at batch 1, embedding at batch 32, while the model defaults differ).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from detail.gui.help_ui import FieldLabel

ADVANCED = "Advanced Settings"

# The body's left inset, so the opened rows read as belonging to the header above them.
_INDENT = 16

# The header's share of the panel width, against one stretch unit of margin either side.
_HEADER_SPAN = 18

_UNREADABLE = object()  # a widget whose value this module can't read: not counted


def row_value(widget: QWidget | None):
    """A form field's current value, for comparing against the state the form opened in.

    Covers the widget kinds the config forms actually use. A spin box carrying a
    ``specialValueText`` reads ``None`` on its floor, matching ``optional_value`` — otherwise
    an ``X | None`` field parked on "unset" would look like a number.

    A widget Qt has already deleted reads as unreadable rather than raising: a form that
    rebuilds its rows (tracking, on tracker change) drops the old widgets, and a header
    refresh must not become an exception on the way out.
    """
    if widget is None:
        return _UNREADABLE
    try:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            if widget.specialValueText() and widget.value() == widget.minimum():
                return None
            return widget.value()
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return widget.currentText() if data is None else data
        if isinstance(widget, QLineEdit):
            return widget.text()
        getter = getattr(widget, "value", None)  # EveryNRow, ThresholdSlider, …
        if callable(getter):
            return getter()
    except (RuntimeError, TypeError):
        return _UNREADABLE
    return _UNREADABLE


# Whichever of these a widget has is how it announces an edit. First match wins — a spin box
# has valueChanged, a check box toggled, a composite row its own ``changed``.
_CHANGE_SIGNALS = (
    "valueChanged",
    "toggled",
    "currentIndexChanged",
    "textChanged",
    "changed",
)


def _on_change(widget: QWidget, slot) -> None:
    for name in _CHANGE_SIGNALS:
        signal = getattr(widget, name, None)
        if signal is not None and hasattr(signal, "connect"):
            signal.connect(lambda *_args: slot())
            return


def _rule(shadow: QFrame.Shadow, width: int = 1) -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(shadow)
    line.setLineWidth(width)
    return line


class _GroupRule(QWidget):
    """A full-width rule with a group's name set into it — a heading, not a control.

    Drawn one weight lighter than the divider above ``Advanced Settings`` (``Plain`` against
    its ``Sunken``), so the nesting reads without a stylesheet or a nested box.
    """

    def __init__(self, title: str) -> None:
        super().__init__()
        box = QHBoxLayout(self)
        box.setContentsMargins(0, 8, 0, 2)
        label = QLabel(title)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        box.addWidget(_rule(QFrame.Shadow.Plain), 1)
        box.addWidget(label)
        box.addWidget(_rule(QFrame.Shadow.Plain), 1)


class AdvancedSection(QFrame):
    """A form's one collapsible home for the knobs a user rarely touches.

    Under a divider sits a flat text-triangle header; under that, everything the section
    holds — ungrouped rows first, then one titled rule per group. No icons, no stylesheet, no
    animation (GUI Style). Starts collapsed, except on the first show when it is already
    holding something the form did not open with.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._expanded = False
        self._toggled_by_user = False
        self._shown_once = False
        self._baseline: dict[QWidget, object] = {}
        self._connected: set[QWidget] = set()
        self._groups: dict[str, tuple[_GroupRule, QFormLayout]] = {}

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 6, 0, 0)
        # Two weights above the group rules inside, so the nesting reads top-down:
        # this divider closes the form's own rows, the rules only sub-divide what is
        # behind the toggle.
        box.addWidget(_rule(QFrame.Shadow.Plain, width=2))
        # A plain framed button: the span is the point, and a flat one has no visible bounds
        # to span with until you hover it. Deliberately **not** a QToolButton — macOS
        # registers a class-specific default font for that class (10pt against QLabel's 13)
        # and the header came out visibly smaller than the parameter labels beside it. The
        # font is pinned to the label font besides, so no platform can shrink it again; the
        # trap is invisible under the offscreen platform, which registers no per-class fonts,
        # so it is reproduced in tests/gui/test_advanced_form.py instead.
        self._header = QPushButton()
        self._header.setFont(QApplication.font("QLabel"))
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._header.clicked.connect(self._on_header_clicked)
        # Centred, spanning most of the panel: the toggle is the width of what it opens,
        # not a word tucked into the left margin. Its font is left alone, so it reads at
        # the same weight and size as the parameter labels above it.
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addStretch(1)
        header_row.addWidget(self._header, _HEADER_SPAN)
        header_row.addStretch(1)
        box.addLayout(header_row)

        self._body = QWidget()
        self._body_box = QVBoxLayout(self._body)
        self._body_box.setContentsMargins(_INDENT, 0, 0, 0)
        self._ungrouped = QFormLayout()
        self._body_box.addLayout(self._ungrouped)
        box.addWidget(self._body)
        self._body.setVisible(False)
        self._sync_header()

    # -- the rows ----------------------------------------------------------- #
    def form(self, group: str | None = None) -> QFormLayout:
        """The layout rows go into: the ungrouped one, or a named group's, created on first
        use. Groups render in the order they were first asked for, after the ungrouped rows.
        """
        if group is None:
            return self._ungrouped
        existing = self._groups.get(group)
        if existing is not None:
            return existing[1]
        rule = _GroupRule(group)
        form = QFormLayout()
        self._body_box.addWidget(rule)
        self._body_box.addLayout(form)
        self._groups[group] = (rule, form)
        return form

    def groups(self) -> list[str]:
        return list(self._groups)

    def clear(self, group: str | None = None) -> None:
        """Drop the rows of one group (or of every layout) — for a form that rebuilds its
        knobs, e.g. tracking when the tracker changes."""
        # Forget the widgets before Qt deletes them: ``removeRow`` destroys each one, and a
        # baseline still holding them is a set of dangling references.
        self._baseline.clear()
        self._connected.clear()
        targets = [self.form(group)] if group is not None else list(self._forms())
        for form in targets:
            while form.rowCount():
                form.removeRow(0)
        self._sync_header()

    def _forms(self):
        yield self._ungrouped
        for _rule_widget, form in self._groups.values():
            yield form

    def labeled_rows(self) -> int:
        """Rows carrying a label — what the header counts. A bare note or warning row (added
        with an empty label) is not a parameter, so it isn't counted as one.

        A row the form has **hidden** is not counted either: it is not a knob the user is
        being offered, so putting it in "Advanced Settings (N)" overstates what opening the
        section would show. This matters wherever one control decides which other rows
        apply — the model picker in ``modules/detection.py``, the method picker in the
        querying form, which carries every registered method's rows and shows one method's.
        ``isHidden`` rather than ``isVisible``: nothing has a shown ancestor while a form is being
        built, so ``isVisible`` is False for every row at that point and the count would come out
        zero.
        """
        total = 0
        for form in self._forms():
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                widget = item.widget() if item else None
                if widget is None or widget.isHidden():
                    continue
                if isinstance(widget, FieldLabel) or widget.findChildren(QToolButton):
                    total += 1
                elif getattr(widget, "text", lambda: "")().strip():
                    total += 1
        return total

    def _field_widgets(self):
        for form in self._forms():
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                widget = item.widget() if item else None
                if widget is not None:
                    yield widget

    # -- changed-since-opened ------------------------------------------------ #
    def snapshot(self) -> None:
        """Take the current state as the baseline "changed" is measured against, and hide the
        heading of any group that ended up with no rows."""
        self._baseline = {}
        for widget in self._field_widgets():
            value = row_value(widget)
            if value is not _UNREADABLE:
                self._baseline[widget] = value
            if widget not in self._connected:
                self._connected.add(widget)
                _on_change(widget, self._sync_header)
        for rule, form in self._groups.values():
            rule.setVisible(bool(form.rowCount()))
        self._sync_header()

    def changed_count(self) -> int:
        """How many of this section's rows differ from what the form opened with.

        A hidden row is skipped for the same reason it is not counted in
        ``labeled_rows``: it is not on offer. Without this, a user who tries one method's
        knob and switches to another leaves an edited row behind that nobody can see, and
        the section announces a change and opens itself to show nothing. Its baseline is
        still kept, so the row is judged correctly if it is shown again.
        """
        changed = 0
        for widget, baseline in self._baseline.items():
            if widget.isHidden():
                continue
            value = row_value(widget)
            if value is _UNREADABLE:
                continue
            try:
                if value != baseline:
                    changed += 1
            except (TypeError, ValueError):  # an exotic value that won't compare
                continue
        return changed

    # -- expansion ----------------------------------------------------------- #
    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._body.setVisible(expanded)
        self._sync_header()

    def _on_header_clicked(self) -> None:
        self._toggled_by_user = True
        self.set_expanded(not self._expanded)

    def reveal_if_changed(self) -> None:
        """Open, if this section hides a value that differs from the form's baseline.

        The rule lives here rather than at the two moments it can become true, because they are
        genuinely two: a form seeded *before* the dialog is shown (Start from applied to a form
        built a moment ago) learns it in `showEvent`, and a form seeded *while* the dialog is on
        screen (the user picking a different starting point) learns it there. Written once, the
        two cannot disagree about what "buries the parameter that made the old run different"
        means.

        A section the user has opened or closed themselves is left alone: their click outranks
        this, and re-opening a section somebody just closed reads as the app arguing.
        """
        if not self._toggled_by_user and self.changed_count():
            self.set_expanded(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            # A seeded form fills its widgets before the dialog is shown, so this is the first
            # moment the section can tell whether it hides a non-default value.
            self.reveal_if_changed()
        self._sync_header()

    def _sync_header(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        changed = self.changed_count()
        inside = f"{self.labeled_rows()}"
        if changed:
            inside += f" · {changed} changed"
        self._header.setText(f"{arrow}  {ADVANCED} ({inside})")


class ConfigForm:
    """A config form: a main ``QFormLayout`` plus one ``AdvancedSection`` beneath it.

    Build it over the widget (or layout) that hosts the form, add rows the way a plain
    ``QFormLayout`` takes them, and call ``finish()`` once the form's opening values are set.
    """

    def __init__(self, host: QWidget | QLayout, model_cls=None) -> None:
        if isinstance(host, QWidget):
            self._box = QVBoxLayout(host)
        else:
            self._box = QVBoxLayout()
            host.addLayout(self._box)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._main = QFormLayout()
        self._box.addLayout(self._main)
        self._model_cls = model_cls
        self._advanced: AdvancedSection | None = None

    # -- rows ---------------------------------------------------------------- #
    @property
    def main(self) -> QFormLayout:
        """The main (always-visible) layout, for code that manipulates rows directly."""
        return self._main

    def addRow(self, label, field=None, *, advanced: bool | str | None = None) -> None:
        """Add a row, to the main body or to the advanced section.

        ``advanced`` overrides; otherwise a ``field_label`` row asks the model whether its
        field is advanced, and anything else stays in the main body. ``True`` puts the row
        directly under the header, a string puts it under that group's heading. Signature
        mirrors ``QFormLayout.addRow`` so an existing form's rows migrate untouched.
        """
        target = self._target(label, advanced)
        if field is None:
            target.addRow(label)
        else:
            target.addRow(label, field)

    def section(self, group: str | None = None) -> QFormLayout:
        """The advanced layout rows go into — for a composite row that adds itself to a
        ``QFormLayout`` (``ComputeTargetRow.add_to``). ``group`` names a heading within it.
        """
        return self.advanced().form(group)

    def advanced(self) -> AdvancedSection:
        """The form's advanced section, created on first use."""
        if self._advanced is None:
            self._advanced = AdvancedSection()
            self._box.addWidget(self._advanced)
        return self._advanced

    def setContentsMargins(self, *margins: int) -> None:
        self._main.setContentsMargins(*margins)

    def _target(self, label, advanced: bool | str | None) -> QFormLayout:
        if advanced is None:
            advanced = self._declared(label)
        if not advanced:
            return self._main
        return self.section(None if advanced is True else advanced)

    def _declared(self, label) -> bool | str:
        """What the model says about this row's field — ``False`` for a row with no field
        behind it, or a label naming something the model doesn't describe (a tracker knob).
        """
        if self._model_cls is None or not isinstance(label, FieldLabel):
            return False
        from detail.gui._pydantic_form import field_advanced

        try:
            return field_advanced(self._model_cls, label.field)
        except KeyError:
            return False

    # -- reaching a row, wherever it landed ----------------------------------- #
    # A form that greys or hides a row by its field widget must not care which layout the
    # row ended up in: the moment a field is flagged advanced, ``main.labelForField``
    # returns None for it and the label silently stops following its field.
    def _layouts(self):
        yield self._main
        if self._advanced is not None:
            yield from self._advanced._forms()

    def _layout_of(self, widget: QWidget) -> QFormLayout | None:
        for layout in self._layouts():
            if layout.getWidgetPosition(widget)[0] >= 0:
                return layout
        return None

    def set_row_visible(self, widget: QWidget, visible: bool) -> None:
        """Show or hide ``widget``'s whole row — its label and its ``?`` included."""
        layout = self._layout_of(widget)
        if layout is None:
            widget.setVisible(visible)
            return
        if hasattr(layout, "setRowVisible"):
            layout.setRowVisible(widget, visible)
            return
        widget.setVisible(visible)  # older Qt: hide the field + its label
        label = layout.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    # -- lifecycle ------------------------------------------------------------ #
    def finish(self) -> None:
        """Close the form: hide the section if it ended up empty, and take the baseline the
        "changed" count is measured against. Call it once the opening values are set — and
        again after a rebuild that replaces the section's rows."""
        if self._advanced is None:
            return
        self._advanced.setVisible(bool(self._advanced.labeled_rows()))
        self._advanced.snapshot()

    def mark_baseline(self) -> None:
        """Re-take the baseline, for a form whose rows are populated after construction (the
        launch dialog pushes an experiment and a recording selection into the form before a
        seed is applied)."""
        if self._advanced is not None:
            self._advanced.snapshot()
