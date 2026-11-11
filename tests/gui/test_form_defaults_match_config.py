"""A form's choice opens on its config model's default, never on whichever row is first.

**The class.** ``detail/configs/`` declares each parameter's default, and the CLI and API read
it. A GUI combo that is filled and left alone opens on index 0 instead — the first row of
whatever order it was filled in — and that is a second, silent answer to "what does a default
run do?". Nothing fails: both values are legal, the run completes, and the result is merely
computed with a setting nobody chose. ``temporal_clustering`` opened on ``codebook_viterbi``
(sorted first) while config, CLI and API defaulted to ``dtw_finch`` (the default is
``codebook_viterbi`` now, chosen on cost, and the form opens on it because it is the default,
not because it sorts first); the embedding form opened
on whichever model was registered first and then on whichever size happened to be installed.
``tracking`` and ``pose_clustering`` (since removed) had each been fixed one at a time before
this guard existed (see the tracking form's comments), which is the sign that it is a class.

**The rule.** Every labelled combo row in every registered module form — the row's
``field_label`` names its config field — opens on that field's declared default, when the
model declares one. Data-driven over the GUI registries and ``iter_config_models()``, so a form
added tomorrow is covered the day it lands.

A field whose default is ``None`` ("the model's own default", "none picked yet") has nothing to
compare a row against here and is skipped; the embedding size is checked by
``test_the_embedding_form_keeps_its_default_size_when_not_installed`` instead.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QComboBox, QFormLayout, QWidget  # noqa: E402

from detail.configs.describe import iter_config_models  # noqa: E402
from detail.configs.embedding import EmbeddingConfig  # noqa: E402
from detail.configs.temporal_clustering import TemporalClusteringConfig  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.help_ui import FieldLabel  # noqa: E402
from detail.gui.model_availability import selected_model_id  # noqa: E402
from tests.gui.test_advanced_form import _registry_forms  # noqa: E402

_SUFFIXES = ("  — ", "  (")  # an availability reason, a "(default)" marker


def _models() -> dict:
    return dict(iter_config_models())


def _opens_on(combo: QComboBox):
    """The value a combo is on: its item data (a model row's ``id``), else its base label."""
    data = combo.currentData()
    if isinstance(data, dict):
        return data.get("id")
    if data is not None:
        return data
    text = combo.currentText()
    for suffix in _SUFFIXES:
        text = text.split(suffix)[0]
    return text


def _combo_rows(widget: QWidget):
    """``(module_id, field, combo)`` for every ``field_label`` row whose field is a combo."""
    for layout in widget.findChildren(QFormLayout):
        for row in range(layout.rowCount()):
            label_item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field_item = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
            label = label_item.widget() if label_item else None
            field = field_item.widget() if field_item else None
            if isinstance(label, FieldLabel) and isinstance(field, QComboBox):
                yield label.module_id, label.field, field


def _offences(widget: QWidget, models: dict) -> list[str]:
    found = []
    for module_id, name, combo in _combo_rows(widget):
        model = models.get(module_id)
        field = model.model_fields.get(name) if model is not None else None
        if field is None or field.is_required():
            continue
        default = field.get_default(call_default_factory=True)
        if default is None:
            continue
        opened = _opens_on(combo)
        if opened != default:
            found.append(
                f"{module_id}.{name} opens on {opened!r}; "
                f"{model.__name__} defaults to {default!r}"
            )
    return found


FORMS = sorted(_registry_forms())


def test_the_sweep_reaches_the_forms_it_was_written_for(qapp):
    """Guard the guard: a sweep that paired no rows would pass forever."""
    forms = _registry_forms()
    checked = {
        (module_id, name)
        for form_id in ("embedding", "temporal_clustering", "tracking")
        for module_id, name, _combo in _combo_rows(forms[form_id].config_widget())
    }
    assert {
        ("embedding", "model_id"),
        ("embedding", "constructor"),
        ("temporal_clustering", "method"),
        ("tracking", "tracker_id"),
    } <= checked


@pytest.mark.parametrize("form_id", FORMS)
def test_every_choice_opens_on_its_config_default(qapp, form_id):
    widget = _registry_forms()[form_id].config_widget()
    offences = _offences(widget, _models())
    assert not offences, (
        f"{form_id}: " + "; ".join(offences) + ". Select the config model's default "
        "(`findData(Model.model_fields[name].default)`), not whichever row is first."
    )


def test_the_guard_catches_a_combo_left_on_its_first_row(qapp):
    """The negative case: the temporal-clustering bug, rebuilt by hand, must be reported."""
    from detail.gui.forms import ConfigForm
    from detail.gui.help_ui import field_label

    host = QWidget()
    form = ConfigForm(host, TemporalClusteringConfig)
    default = TemporalClusteringConfig.model_fields["method"].default
    other = next(m for m in ("codebook_viterbi", "dtw_finch") if m != default)
    combo = QComboBox()
    combo.addItems([other, default])  # the non-default first, index 0 left alone
    form.addRow(field_label("temporal_clustering", "method", "Method:"), combo)
    form.finish()
    assert _offences(host, _models()) == [
        f"temporal_clustering.method opens on {other!r}; "
        f"TemporalClusteringConfig defaults to {default!r}"
    ]
    combo.setCurrentIndex(1)
    assert not _offences(host, _models())


@pytest.mark.parametrize(
    "missing",
    ["default_size", "whole_model"],
    ids=["another-size-installed", "no-size-installed"],
)
def test_the_embedding_form_keeps_its_default_size_when_not_installed(
    qapp, monkeypatch, missing
):
    """With the default size's weights absent the form stays on the default model *and size*,
    greyed and labelled, and refuses to launch with the reason — it never opens on another
    size or model that happens to be installed, which would change every vector the run
    writes. Both shapes: another size of the default model is installed (the size picker had
    somewhere to move to), and no size of it is (the model picker had)."""
    from detail.api import models as api_models
    from detail.core.registries import EMBEDDING_MODELS

    default_model = EmbeddingConfig.model_fields["model_id"].default
    default_size = EMBEDDING_MODELS[default_model].default_variant()

    def is_installed(model_id, variant=None):
        if model_id != default_model:
            return True
        return missing == "default_size" and variant != default_size

    monkeypatch.setattr(api_models, "is_installed", is_installed)
    widget = _registry_forms()["embedding"].config_widget()
    assert selected_model_id(widget._model) == default_model
    assert widget._variant.currentData() == default_size
    assert "not installed" in widget._variant.currentText()
    error = widget.validation_error()
    assert error is not None and "not installed" in error
