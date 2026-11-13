"""Build Qt form widgets from a pydantic config model — the single source for a parameter's
type, range, step, help, and label (a widget can't drift from the CLI or validator).

Four entry points, in order of preference:

- ``build_fields(model)`` — every plain scalar field at once, when a form's rows are just the
  model's fields.
- ``field_widget(model, name)`` — **one** field, for a form that interleaves rows the model
  doesn't describe (an upstream run picker, a prompt, a compute target) or needs its own row
  order. Reach for this instead of hand-building a spin box beside a generated one.
- ``optional_spin(model, name, auto_text)`` — an ``X | None`` field whose ``None`` means
  "work it out for me", rendered as a position one step below the field's own floor.
- ``field_slider(model, name)`` — a bounded field the user *sweeps* rather than types, for
  filtering what a finished result shows (``configs/display.ConfidenceView``). Same metadata,
  a ``ThresholdSlider`` instead of a bare box.

Whether a field is *shown* by default is the model's call too: ``field_advanced`` reads the
field's ``advanced`` flag and ``gui/forms.py::ConfigForm`` routes the row accordingly.

Fields with no scalar shape — enums, free text, experiment-populated combos, nested inputs —
stay hand-built; use ``field_meta`` so *their* help and bounds still come from the model, and set
their initial value yourself.

**Every builder here seeds the widget with the model's default**, in one place
(``_seed_default``). A surface that wants something other than the model's default sets it
afterwards.

Read through ``configs/describe.py``, the same source the CLI help and the generated
parameter reference read, so a widget cannot show a different range than the docs.
"""

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from PyQt6.QtWidgets import QCheckBox, QComboBox, QWidget

from detail.configs.describe import describe_config
from detail.gui.spin import (
    COUNT,
    FRACTION,
    PLAIN,
    SECONDS,
    FloatSpin,
    IntSpin,
    ThresholdSlider,
    float_format,
)

_BIG = 1_000_000_000

_NUMERIC = ("integer", "number")


def _field(model_cls: type[BaseModel], name: str) -> dict:
    for field in describe_config(model_cls):
        if field["name"] == name:
            return field
    raise KeyError(f"{model_cls.__name__} has no field {name!r}")


def field_meta(model_cls: type[BaseModel], name: str) -> dict:
    """``{label, help, minimum, maximum}`` for one field — for hand-built widgets to read their
    metadata from the model instead of hard-coding it."""
    field = _field(model_cls, name)
    return {key: field[key] for key in ("label", "help", "minimum", "maximum")}


def field_advanced(model_cls: type[BaseModel], name: str) -> bool | str:
    """Whether a field belongs under a form's collapsible advanced section.

    ``False`` (the default for any field that doesn't declare it) keeps the row in the main
    body; ``True`` puts it under "Advanced settings"; a string names its own section, so a
    form with a lot of knobs can split them. ``gui/forms.py::ConfigForm`` routes on this — a
    form never decides for itself which of its model's fields are advanced.
    """
    return _field(model_cls, name)["advanced"]


def field_kind(field: dict) -> str:
    """The ``gui/spin.py`` kind for a described field.

    Declared ``ui`` wins; otherwise an integer is a count, a field carrying seconds is a
    duration, a bounded 0-1 field is a fraction, and anything else is plain.
    """
    if field["ui"]:
        return field["ui"]
    if field["type"] == "integer":
        return COUNT
    if field["unit"] == "s" or field["name"].endswith("_s"):
        return SECONDS
    lo, hi = field["minimum"], field["maximum"]
    if lo is not None and lo >= 0 and hi is not None and hi <= 1:
        return FRACTION
    return PLAIN


def _range(field: dict, minimum, maximum, epsilon) -> tuple:
    """The spin box's range: explicit override, then the field's ``ui_max`` display ceiling,
    then the model's own bound.

    An exclusive bound (``gt`` / ``lt``) is stepped *inside* by ``epsilon``, because a spin box
    can only offer values it can land on — ``val_split`` is ``lt=1.0``, so the box stops just
    under 1.0 rather than offering a value the validator would reject.

    ``epsilon`` is the **smallest increment the box can represent** (``10**-decimals``), not its
    single step. Using the step assumed the step is small next to the field's values, and
    ``query_eval.tau`` is where that broke: ``gt=0.0`` with a 0.1 step put the floor at 0.1,
    above the field's own default of 0.07, so the form clamped and silently shipped a different
    temperature than the model declares.
    """
    if minimum is not None:
        lo = minimum
    elif field["minimum"] is not None:
        lo = field["minimum"]
    elif field["exclusive_minimum"] is not None:
        lo = field["exclusive_minimum"] + epsilon
    else:
        lo = None
    if maximum is not None:
        hi = maximum
    elif field["ui_max"] is not None:
        hi = field["ui_max"]
    elif field["maximum"] is not None:
        hi = field["maximum"]
    elif field["exclusive_maximum"] is not None:
        hi = field["exclusive_maximum"] - epsilon
    else:
        hi = None
    return lo, hi


def _epsilon(kind: str, decimals: int) -> float:
    """The smallest amount a box of this kind can move — what an exclusive bound steps by."""
    return 1.0 if kind == COUNT else 10.0**-decimals


def _number_widget(field: dict, kind, minimum, maximum, step) -> QWidget:
    resolved = kind or field_kind(field)
    decimals, default_step = float_format(resolved)
    if step is None:
        step = field["ui_step"] or default_step
    lo, hi = _range(field, minimum, maximum, _epsilon(resolved, decimals))
    if resolved == COUNT:
        widget: QWidget = IntSpin()
        widget.setRange(
            int(lo) if lo is not None else 0, int(hi) if hi is not None else _BIG
        )
        widget.setSingleStep(int(step))
    else:
        widget = FloatSpin(resolved)
        widget.setRange(
            float(lo) if lo is not None else 0.0,
            float(hi) if hi is not None else float(_BIG),
        )
        widget.setSingleStep(step)
    return widget


def _seed_default(widget: QWidget, model_cls: type[BaseModel], name: str) -> None:
    """Put the model's default into a freshly built widget.

    An ``X | None`` defaulting to ``None`` is left where it is: "work it out for me" has no
    number to show, and its affordance belongs to :func:`optional_spin`.
    """
    default = model_cls.model_fields[name].default
    if default is PydanticUndefined or default is None:
        return
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(default))
    else:
        widget.setValue(default)


def field_widget(
    model_cls: type[BaseModel],
    name: str,
    *,
    kind: str | None = None,
    minimum=None,
    maximum=None,
    step=None,
) -> QWidget:
    """The widget for one model field, carrying its range, step, precision and help.

    **Seeded with the model's default**, like every other builder here. A widget starts at its
    *minimum*, and the minimum is the field's own bound, so a caller that forgets to seed opens
    the form on the bottom of every range — legal numbers, a form that renders, a run that
    completes, and nothing to see. That shipped once already (``build_fields``' docstring has the
    case); this door had the same hole, and four call sites had fallen through it.

    ``kind`` / ``minimum`` / ``maximum`` / ``step`` override what the model says, for the rare
    surface that needs a narrower view than the field allows. Prefer declaring ``ui`` /
    ``ui_max`` on the field so every surface agrees. A surface wanting a *different* starting
    value still sets one afterwards.
    """
    field = _field(model_cls, name)
    if field["type"] == "boolean":
        widget: QWidget = QCheckBox()
    elif field["type"] in _NUMERIC:
        widget = _number_widget(field, kind, minimum, maximum, step)
    else:
        raise ValueError(
            f"{model_cls.__name__}.{name} is {field['type'] or 'not a scalar'}; "
            "build it by hand and take its help from field_meta"
        )
    if field["help"]:
        widget.setToolTip(field["help"])
    _seed_default(widget, model_cls, name)
    return widget


def field_slider(
    model_cls: type[BaseModel],
    name: str,
    *,
    kind: str | None = None,
    minimum=None,
    maximum=None,
    step=None,
    debounce_ms: int = 120,
) -> QWidget:
    """A ``ThresholdSlider`` for one model field, carrying its range, step and help.

    For a bounded value the user *sweeps* to see its effect rather than types once — today
    the confidence a detection result is displayed at, on every surface that shows one.

    Raises:
        ValueError: If the field is non-numeric, integral (no integer slider exists yet),
            or unbounded on either side — a slider needs both ends to lay out its track.
    """
    field = _field(model_cls, name)
    if field["type"] not in _NUMERIC:
        raise ValueError(
            f"{model_cls.__name__}.{name} is {field['type'] or 'not a scalar'}; "
            "a slider needs a bounded number"
        )
    resolved = kind or field_kind(field)
    if resolved == COUNT:
        raise ValueError(
            f"{model_cls.__name__}.{name} is an integer; ThresholdSlider is float-only "
            "so far — give it an integer track before sliding a count"
        )
    if step is None:
        step = field["ui_step"] or float_format(resolved)[1]
    lo, hi = _range(
        field, minimum, maximum, _epsilon(resolved, float_format(resolved)[0])
    )
    if lo is None or hi is None:
        raise ValueError(
            f"{model_cls.__name__}.{name} is unbounded ({lo}, {hi}); a slider needs both "
            "ends — declare them on the field or pass minimum/maximum"
        )
    widget = ThresholdSlider(
        float(lo),
        float(hi),
        step=float(step),
        kind=resolved,
        debounce_ms=debounce_ms,
    )
    if field["help"]:
        widget.setToolTip(field["help"])
    return widget


def optional_spin(
    model_cls: type[BaseModel],
    name: str,
    auto_text: str,
    *,
    kind: str | None = None,
    maximum=None,
    step=None,
) -> QWidget:
    """A spin box for an ``X | None`` field whose ``None`` means "work it out for me".

    "Unset" sits one step *below* the field's own floor and renders as ``auto_text``, so it
    can't collide with a value the field allows — a split gap of 0 (never split) stays
    reachable, and a ``ge=1`` count still reads "unset" at 0 the way it always has. No
    separate checkbox, and no way to submit a value the model would reject. Read it back with
    ``optional_value``, set it with ``set_optional_value``.
    """
    field = _field(model_cls, name)
    widget = _number_widget(field, kind, None, maximum, step)
    # One step below the range the field itself allows, so "unset" can never be confused
    # with a value — and a meaningful 0 (a split gap of 0 never splits) stays reachable.
    unset = widget.minimum() - (
        1 if field["type"] == "integer" else widget.singleStep()
    )
    widget.setRange(unset, widget.maximum())
    widget.setSpecialValueText(auto_text)
    widget.setValue(unset)
    # Seeded with the model's default when it has one, as `build_fields` seeds a plain field:
    # an `X | None` whose default is a number (a batch's fixed count per side) otherwise
    # opened on "unset" and launched the other behaviour, with nothing on screen to say so.
    default = model_cls.model_fields[name].default
    if default is not None and default is not PydanticUndefined:
        widget.setValue(default)
    if field["help"]:
        widget.setToolTip(field["help"])
    return widget


def optional_value(spin):
    """The spin box's value, or ``None`` when it sits on its "unset" position."""
    return None if spin.value() == spin.minimum() else spin.value()


def set_optional_value(spin, value) -> None:
    """Show ``value``, or the "unset" position for ``None`` — the load_config twin."""
    spin.setValue(spin.minimum() if value is None else value)


def build_fields(model_cls: type[BaseModel]) -> dict[str, tuple[str, QWidget]]:
    """``{field_name: (label, widget)}`` for each plain scalar field, **seeded with the model's
    default**, with range + tooltip from the model. The caller adds the rows and reads
    ``widget.value()`` in its ``config()``. ``X | None`` fields are omitted — they need a chosen
    "unset" affordance, so build those with ``optional_spin``.

    Seeding here rather than leaving it to the caller, because leaving it to the caller is a
    defect that ships silently. A `QSpinBox` starts at its **minimum**, and the minimum is the
    field's own `ge` bound — so a form that forgets the three-line seeding loop launches runs
    with every knob at the bottom of its range, and there is nothing to see: the form renders,
    the numbers are legal, the run completes. It happened to pose clustering, whose `fit_rows`
    went out at 100 instead of 6,000 and produced a vocabulary far coarser than anyone asked
    for; the only symptom was somebody saying the result looked wrong.

    A surface that genuinely wants a different starting value still sets one afterwards — this
    changes the default from "the bottom of the range" to "what the model says", which is what
    every caller wanted anyway.
    """
    out: dict[str, tuple[str, QWidget]] = {}
    for field in describe_config(model_cls):
        if field["optional"] or field["type"] not in _NUMERIC + ("boolean",):
            continue
        out[field["name"]] = (field["label"], field_widget(model_cls, field["name"]))
    return out


#: What an ``X | None`` knob reads as when it sits on its "unset" position and the method
#: resolves the value itself. Matches the house wording (`optional_spin`'s callers).
AUTO_TEXT = "auto"


def choice_combo(model_cls: type[BaseModel], name: str) -> QComboBox:
    """A combo for a ``Literal[...]`` field, seeded with the model's default.

    The values come from ``describe_config``'s ``choices`` — the schema's ``enum`` — rather
    than from ``get_args`` on the annotation at the call site, which is how three forms
    currently each grow their own copy of one model's option list.
    """
    field = _field(model_cls, name)
    if not field["choices"]:
        raise ValueError(f"{model_cls.__name__}.{name} is not a choice field")
    widget = QComboBox()
    if field["optional"]:
        # ``Literal[...] | None``: None is "let the method resolve it", first and default.
        widget.addItem(AUTO_TEXT, None)
    for value in field["choices"]:
        widget.addItem(str(value), value)
    default = model_cls.model_fields[name].default
    if default is not PydanticUndefined and default in field["choices"]:
        widget.setCurrentIndex(widget.findData(default))
    if field["help"]:
        widget.setToolTip(field["help"])
    return widget


def any_field_widget(model_cls: type[BaseModel], name: str) -> QWidget:
    """The widget for **any** shape of scalar field, seeded with the model's default.

    The three builders above each cover one shape — a plain number or bool
    (:func:`field_widget`), an ``X | None`` (:func:`optional_spin`), a ``Literal``
    (:func:`choice_combo`) — and a caller that wants to render a *whole model it does not
    know* has to dispatch between them. Every caller that tried ended up rendering only the
    shape it remembered: ``build_fields`` skips the other two, so a form built on it alone
    silently drops an optional or a choice knob. The removed `DtwQueryConfig` was 2 of 5 fields that way.

    **Raises on a shape it cannot render**, deliberately. A field that quietly fails to
    appear is the defect this exists to remove, and it is invisible — the form builds, the
    run launches, the parameter is simply never settable. Failing here surfaces at
    construction, which the GUI smoke tests already run.
    """
    field = _field(model_cls, name)
    if field["choices"]:
        return choice_combo(model_cls, name)
    if field["optional"]:
        if field["type"] not in _NUMERIC:
            raise ValueError(
                f"{model_cls.__name__}.{name} is an optional {field['type']}; only an "
                "optional number has an 'unset' affordance (optional_spin)"
            )
        return optional_spin(model_cls, name, AUTO_TEXT)
    return field_widget(model_cls, name)


def any_field_value(model_cls: type[BaseModel], name: str, widget: QWidget):
    """Read back what :func:`any_field_widget` built — the twin, dispatched the same way.

    Keyed off the *field*, not the widget's class: an ``X | None`` spin and a plain one are
    both spin boxes, and telling them apart by type would come down to whether a special
    value text happens to be set.
    """
    field = _field(model_cls, name)
    if field["choices"]:
        return widget.currentData()
    if field["optional"]:
        return optional_value(widget)
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    return widget.value()


def set_any_field_value(
    model_cls: type[BaseModel], name: str, widget: QWidget, value
) -> None:
    """Write ``value`` into what :func:`any_field_widget` built — the third of the trio.

    ``any_field_widget`` builds and ``any_field_value`` reads; without a setter every caller
    that wants to *restore* a config hand-rolls the dispatch, and each one covers only the
    shapes its author remembered. That is the same drift the other two exist to remove, and it
    is why ``load_config`` was a silent no-op in four modules.

    Dispatched off the *field*, exactly as its twin is: an ``X | None`` spin and a plain one
    are both spin boxes, and a ``Literal`` combo carries its value in ``itemData`` rather than
    its text.

    A ``value`` outside the field's range is **clamped by the widget**, not rejected — the
    widget's range is the model's, so a stored value that no longer fits (a bound tightened
    since the run) lands on the nearest legal one instead of refusing to load the rest of the
    form. An unknown choice leaves the combo where it was.
    """
    field = _field(model_cls, name)
    if field["choices"]:
        if value is None and field["optional"]:
            widget.setCurrentIndex(0)  # the AUTO_TEXT row choice_combo put first
            return
        index = widget.findData(value)
        if index >= 0:
            widget.setCurrentIndex(index)
        return
    if field["optional"]:
        set_optional_value(widget, value)
        return
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
        return
    if value is None:
        return
    widget.setValue(int(value) if field["type"] == "integer" else float(value))
