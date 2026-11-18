"""Every widget a model field builds opens on that field's default, not on its floor.

**The class.** A number box starts at its *minimum*, and the minimum is the field's own bound.
So a builder that carries a field's range without carrying its value hands the user a form where
every knob sits at the bottom of what it allows — and there is nothing to see. The form renders,
the numbers are legal, the run launches, the result is merely wrong. It shipped that way once:
pose clustering's ``fit_rows`` went out at 100 instead of 6,000 and produced a vocabulary far
coarser than anyone asked for, and the only symptom was somebody saying the result looked wrong.

**How it hid the second time.** That first fix went into ``build_fields``, and its docstring
argued the case at length — but ``field_widget``, the door a form uses when it interleaves model
rows with hand-built ones, was left deliberately unseeded, with a docstring saying callers
"usually have their own starting value in mind". Four of them did not: the comparison dialog's
four window knobs offered a 0.01-second tile width where the model says 600, and detection opened
its batch size on 1. Every one of those forms had a passing construction test.

**The rule.** Seeding lives in one place (``_pydantic_form._seed_default``) and every builder goes
through it, so a new builder inherits it rather than re-deciding. The test below is data-driven
over every scalar field of every registered config model, so a new model is covered the day it is
added — and it asserts the *value the user reads*, not that the widget was built.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QCheckBox  # noqa: E402
from pydantic_core import PydanticUndefined  # noqa: E402

from detail.configs.describe import describe_config, iter_config_models  # noqa: E402
from detail.configs.windows import WindowSpec  # noqa: E402
from detail.gui._pydantic_form import (  # noqa: E402
    any_field_value,
    any_field_widget,
    build_fields,
    field_widget,
)


def _models():
    """Every registered config model, plus the nested ones a dialog builds fields from
    directly — `iter_config_models` walks the registries, and `WindowSpec` is reached
    through `ComparisonConfig.windows` rather than being registered itself."""
    seen = {WindowSpec}
    for _module_id, model_cls in iter_config_models():
        seen.add(model_cls)
    return sorted(seen, key=lambda m: m.__name__)


def _cases():
    for model_cls in _models():
        for field in describe_config(model_cls):
            default = model_cls.model_fields[field["name"]].default
            if default is PydanticUndefined or field["choices"] or field["optional"]:
                continue
            if field["type"] not in ("integer", "number", "boolean"):
                continue
            yield pytest.param(
                model_cls,
                field["name"],
                default,
                id=f"{model_cls.__name__}.{field['name']}",
            )


CASES = list(_cases())


@pytest.fixture(autouse=True)
def _discard_widgets(qapp):
    """Destroy what each test built.

    The `QApplication` is session-scoped, so a widget nobody deletes lives until the process
    exits. This file builds one per field of every registered config model; left behind, they
    accumulate into a measurable drag on the shared event loop and a screen tested minutes
    later misses a wall-clock deadline for reasons that have nothing to do with it.
    """
    before = set(qapp.allWidgets())
    yield
    for widget in qapp.allWidgets():
        if widget not in before:
            widget.deleteLater()
    qapp.processEvents()


#: Models `build_fields` builds no scalar widget for, and why. A claim, like `UNIT_EXEMPTIONS`:
#: `test_every_no_scalar_entry_is_still_true` fails on one that has grown a field.
NO_SCALAR_FIELDS = {
    "LocationExportConfig": "only list-valued ROI/POI selections, no numbers or toggles",
    "TrackingConfig": "a tracker id plus that tracker's own hyperparameter dict",
    "CsvStreamConfig": "a column name, a choice, and a list of channels — every value is "
    "read off the CSV being imported, so there is no number or toggle to seed",
    "MedpcSessionConfig": "an optional subject, box and session number that pick one block "
    "of the file being imported; every one defaults to unset and is read off that file",
    "BorisSelectionConfig": "an optional observation id and subject that pick rows of the "
    "export being imported; both default to unset and are read off that file",
}


def test_every_no_scalar_entry_is_still_true(qapp):
    """Guard the guard's exemptions: a model that has since grown a scalar field should be
    checked, not excused."""
    still_empty = []
    for model_cls in _models():
        if model_cls.__name__ not in NO_SCALAR_FIELDS:
            continue
        fields = build_fields(model_cls)
        if any(
            model_cls.model_fields[n].default not in (PydanticUndefined, None)
            for n in fields
        ):
            still_empty.append(model_cls.__name__)
    assert (
        not still_empty
    ), f"NO_SCALAR_FIELDS excuses models that now have scalar fields: {still_empty}"


def test_there_are_fields_to_check():
    """Guard the guard: a parametrisation that silently empties passes forever."""
    assert len(CASES) > 20, f"only {len(CASES)} scalar fields found; the sweep broke"


@pytest.mark.parametrize("model_cls,name,default", CASES)
def test_field_widget_opens_on_the_models_default(qapp, model_cls, name, default):
    widget = field_widget(model_cls, name)
    shown = widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
    assert shown == pytest.approx(default), (
        f"{model_cls.__name__}.{name} opens on {shown!r}, but the model says {default!r} "
        f"(a spin with no seeded value shows its floor)"
    )


@pytest.mark.parametrize("model_cls,name,default", CASES)
def test_any_field_widget_round_trips_the_default(qapp, model_cls, name, default):
    """The dispatching door and its reader agree on an untouched form."""
    widget = any_field_widget(model_cls, name)
    assert any_field_value(model_cls, name, widget) == pytest.approx(default)


@pytest.mark.parametrize("model_cls", _models(), ids=lambda m: m.__name__)
def test_a_whole_form_opens_on_the_models_defaults(qapp, model_cls):
    """The consequence that matters: what a form emits when nobody touches it is what the
    model would have produced on its own.

    Parametrized over the *model* rather than the field, deliberately. Per-field it rebuilt
    the entire form once per field — thousands of undeleted `QWidget`s across the sweep, which
    slowed the shared `QApplication` enough to time out an unrelated screen's off-thread load
    later in the run. A test that perturbs its neighbours is a broken test.
    """
    fields = build_fields(model_cls)
    checked = 0
    for name, (_label, widget) in fields.items():
        default = model_cls.model_fields[name].default
        if default is PydanticUndefined or default is None:
            continue
        shown = widget.isChecked() if isinstance(widget, QCheckBox) else widget.value()
        assert shown == pytest.approx(default), f"{model_cls.__name__}.{name}"
        checked += 1

    # A parametrisation that asserts nothing is a green tick for a model nobody checked.
    # `LocationExportConfig` and `TrackingConfig` build no scalar widgets at all — every field is
    # a choice or a nested input — so they ran the loop zero times and reported a pass. Listed,
    # with the reason, rather than silently tolerated.
    if not checked:
        assert model_cls.__name__ in NO_SCALAR_FIELDS, (
            f"{model_cls.__name__} has no scalar field to check, so this test says nothing "
            f"about it. Add it to NO_SCALAR_FIELDS with the reason, or give it a field."
        )


def test_an_unseeded_widget_would_be_caught(qapp):
    """Guard the guard, the other half: the assertion is capable of failing.

    Simulates the defect by putting a widget back on its floor, the state a builder that
    carried the range but not the value would leave it in.
    """
    from detail.configs.detection import DetectionConfig

    default = DetectionConfig.model_fields["conf_threshold"].default
    widget = field_widget(DetectionConfig, "conf_threshold")
    assert widget.value() == default

    widget.setValue(widget.minimum())
    assert widget.value() != default, (
        "the floor and the default coincide here, so this field can no longer show that an "
        "unseeded widget is distinguishable -- pick another"
    )
