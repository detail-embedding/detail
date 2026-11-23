"""Tests for the pydantic→Qt form helper (detail/gui/_pydantic_form.py)."""

from detail.configs.detection import DetectionConfig
from detail.configs.embedding import EmbeddingConfig
from detail.gui._pydantic_form import (
    build_fields,
    field_meta,
)
from detail.gui.qt import QSpinBox


def test_build_fields_generates_scalar_from_model(qapp):
    fields = build_fields(DetectionConfig)
    # every_n_frames is a clean int scalar → generated with range + tooltip + label from the model.
    label, widget = fields["every_n_frames"]
    assert isinstance(widget, QSpinBox)
    assert widget.minimum() == 1  # from Field(ge=1)
    assert (
        widget.toolTip() == DetectionConfig.model_fields["every_n_frames"].description
    )
    assert label == "Use every"
    # model_variant is Optional (anyOf) → skipped; the caller builds those by hand.
    assert "model_variant" not in fields


def test_field_meta_reads_an_optional_field():
    # field_meta digs the bound + help out of an `int | None` field for hand-built widgets.
    meta = field_meta(EmbeddingConfig, "every_n_frames")
    assert meta["minimum"] == 1
    assert "multiple of" in meta["help"]
    assert meta["label"] == "Use every"


def test_build_fields_starts_every_widget_at_the_model_default(qapp):
    """A form that forgets to seed its widgets ships every knob at the bottom of its range.

    The class: **a QSpinBox starts at its minimum, and the minimum is the field's own bound.** So
    a form built from `build_fields` that never sets initial values launches runs with, say,
    `fit_rows` at 100 instead of 6,000 — and nothing looks wrong. The form renders, the numbers
    are legal, the run completes, and the only symptom is somebody saying the result is too
    coarse. That is what happened to pose clustering.

    Four forms carried an identical three-line seeding loop to avoid it, which is what makes this
    a mechanism rather than a slip: the fifth form forgot, as a fifth form eventually would.
    Seeding lives in `build_fields` now, and this is its guard, over every model that reaches the
    parameter reference so a new config is covered the day it is added.
    """
    from pydantic_core import PydanticUndefined

    from detail.configs.describe import iter_config_models
    from detail.gui._pydantic_form import build_fields

    checked = 0
    for module_id, model_cls in iter_config_models():
        for name, (_label, widget) in build_fields(model_cls).items():
            default = model_cls.model_fields[name].default
            if default is PydanticUndefined:
                continue
            if hasattr(widget, "isChecked"):
                value, tolerance = widget.isChecked(), 0
            else:
                value = widget.value()
                # Compared at the box's own precision, not exactly. A seconds box shows two
                # decimals, so a 0.034 s default renders 0.03 — a deliberate display choice
                # rather than a wrong number. One representable step still catches what this
                # guard is for: a field left at its minimum (`fit_rows` at 100 against a 6,000
                # default) or clamped outside its range by an exclusive bound (`tau` at 0.1
                # against 0.07).
                tolerance = 10.0 ** -getattr(widget, "decimals", lambda: 0)()
            assert abs(value - default) <= tolerance, (
                f"{module_id}.{name} starts at {value!r} but its model default is "
                f"{default!r} — a form built from this ships the wrong number"
            )
            checked += 1
    assert checked > 20, f"only {checked} fields checked; the walk found almost nothing"


def test_an_optional_choice_offers_auto_and_reads_back_none(qapp):
    """A ``Literal[...] | None`` renders as a combo whose first row is "auto" (None) — before,
    ``any_field_widget`` refused it as an optional string and the knob could not be shown.
    No shipped config has such a field yet (§26's will); the model is local.
    """
    from typing import Literal

    from pydantic import BaseModel, Field

    from detail.configs.describe import describe_config
    from detail.gui._pydantic_form import (
        AUTO_TEXT,
        any_field_value,
        any_field_widget,
        set_any_field_value,
    )

    class Knob(BaseModel):
        rule: Literal["hard", "soft", "off"] | None = Field(
            None, description="how", json_schema_extra={"label": "Rule"}
        )
        fixed: Literal["a", "b"] = Field(
            "b", description="which", json_schema_extra={"label": "Fixed"}
        )

    combo = any_field_widget(Knob, "rule")
    choices = next(f["choices"] for f in describe_config(Knob) if f["name"] == "rule")
    assert [combo.itemText(i) for i in range(combo.count())] == [AUTO_TEXT, *choices]
    assert any_field_value(Knob, "rule", combo) is None
    set_any_field_value(Knob, "rule", combo, "soft")
    assert any_field_value(Knob, "rule", combo) == "soft"
    set_any_field_value(Knob, "rule", combo, None)
    assert any_field_value(Knob, "rule", combo) is None
    # a plain Literal gets no "auto" row and still opens at its default
    fixed = any_field_widget(Knob, "fixed")
    assert [fixed.itemText(i) for i in range(fixed.count())] == ["a", "b"]
    assert any_field_value(Knob, "fixed", fixed) == "b"
