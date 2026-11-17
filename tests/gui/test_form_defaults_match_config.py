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

**The same class for a spin box** (#35): a widget built from its model (``field_widget``,
``build_fields``) and then given a hand-written ``setRange`` or a literal ``setValue`` is a second
answer to the model's range or default. Two labeling dialogs capped ``val_split`` at 0.9 while
the third went to the model's bound. A fail-closed scan of ``detail/gui`` refuses both, with a
negative case; a cap the GUI wants is the field's ``ui_max``.
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


# --- a model-built widget keeps the model's range and default -------------------- #
#: Model-built widgets a surface deliberately re-seeds, keyed ``file:attribute``, and why.
HAND_SET: dict[str, str] = {
    "fit.py:_n_frames": (
        "`LabelingConfig.n_frames` is required — the model has no default to seed — so the "
        "dialog supplies its own starting count"
    ),
}


def _target_name(node) -> str | None:
    import ast

    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _model_built(value) -> bool:
    """``field_widget(...)``, or ``fields["x"]`` out of ``build_fields`` (unpacked)."""
    import ast

    if isinstance(value, ast.Call):
        func = value.func
        return getattr(func, "id", getattr(func, "attr", None)) == "field_widget"
    return (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id == "fields"
    )


def hand_set_model_widgets(name: str, text: str) -> list[str]:
    """``file:attribute`` for each model-built widget a function then gives a hand-written
    range (``setRange``) or a literal value (``setValue(<constant>)``)."""
    import ast

    found = []
    for fn in ast.walk(ast.parse(text)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        built = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _model_built(node.value):
                for target in node.targets:
                    elts = target.elts if isinstance(target, ast.Tuple) else [target]
                    built.add(_target_name(elts[-1]))
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and _target_name(node.func.value) in built
            ):
                continue
            literal = bool(node.args) and isinstance(node.args[0], ast.Constant)
            if node.func.attr == "setRange" or (
                node.func.attr == "setValue" and literal
            ):
                found.append(f"{name}:{_target_name(node.func.value)}")
    return found


def test_no_model_built_widget_is_given_a_hand_written_range_or_default():
    """#35 finding 6. ``fit.py`` capped ``val_split`` at 0.9 by hand in two dialogs while the
    third used the model's bound, and a dozen widgets built from their model were then set to
    a literal — the model's own default retyped (``detection``'s batch box to 1, where the
    model's default is "the model's own"). Two answers to one question, which drift apart the
    first time either moves. A cap the GUI wants goes on the field as ``ui_max``."""
    from pathlib import Path

    gui = Path(__file__).resolve().parents[2] / "detail" / "gui"
    found = []
    for source in sorted(gui.rglob("*.py")):
        found += hand_set_model_widgets(source.name, source.read_text(encoding="utf-8"))
    unexcused = sorted(set(found) - set(HAND_SET))
    assert not unexcused, f"hand-set model-built widgets: {unexcused}"
    stale = sorted(set(HAND_SET) - set(found))
    assert not stale, f"HAND_SET excuses what no longer happens: {stale}"


def test_the_hand_set_scan_would_catch_a_cap_and_a_retyped_default():
    text = (
        "def build(self):\n"
        "    label, self._split = fields['val_split']\n"
        "    self._split.setRange(0.0, 0.9)\n"
        "    self._conf = field_widget(Config, 'conf')\n"
        "    self._conf.setValue(0.5)\n"
        "    self._other = QDoubleSpinBox()\n"
        "    self._other.setValue(0.5)\n"
        "    self._conf.setValue(seed)\n"
    )
    assert hand_set_model_widgets("x.py", text) == ["x.py:_split", "x.py:_conf"]
