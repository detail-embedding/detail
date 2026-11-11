"""A registry method's parameters reach every surface, for every registered method.

The class: **a registry whose members each declare a ``config_model``, whose parameters reach
a surface only by being retyped at the call site.** Both method registries had it, in
different disguises, and neither showed up as a failure anywhere:

- Querying sent ``method_params={}`` from the GUI form *and* the CLI, so **none** of ``dtw``'s
  five parameters was settable from either. The form rendered, the run launched, the knobs
  were simply not there.
- Pose clustering built ``build_fields(FinchPoseConfig)`` against a hand-written tuple of four
  field names, and its CLI had one ``--fit-rows`` option. Correct only while ``finch`` is the
  only method registered, and **1 of 4** parameters in the CLI — an undocumented surface gap,
  which `CLAUDE.md` calls the bug rather than the omission.

It was already replicating: a third copy of the same tuple-and-hardcoded-model shape existed
in an unmerged ``analysis_temporal_clustering.py`` before this was fixed.

Nothing behavioural can see any of it — a parameter that never arrives is indistinguishable
from a parameter left at its default — so the guard is registry parity. Everything below is
data-driven over ``QUERY_METHODS`` (and was over ``POSE_CLUSTER_METHODS`` until pose clustering
was removed), so a method registered tomorrow, or another registry wired the same way, is
covered the day it is added.

The temporal-clustering form is swept too, by its own four tests at the end. It lists each
method's rows in a hand-written ``METHOD_FIELDS`` tuple beside a hand-written ``METHOD_MODELS``
map, which is the shape named above, so the tests ask the registry what each method declares
and never read either table.

``build_fields`` alone is not enough to render a model, and that is the trap underneath all
of it: it skips ``X | None`` and ``Literal`` fields, which was 3 of the removed ``dtw``'s 5.
``_pydantic_form.any_field_widget`` dispatches across all three shapes and *raises* on one it
cannot render, so a new field shape fails at construction instead of silently vanishing.
"""

import pytest

pytest.importorskip("PyQt6")

from click.testing import CliRunner  # noqa: E402
from PyQt6.QtWidgets import QCheckBox, QComboBox  # noqa: E402

from detail.cli import main  # noqa: E402
from detail.configs.describe import describe_config  # noqa: E402
from detail.configs.method_params import resolve_method_params  # noqa: E402
from detail.gui.modules.analysis_query import (  # noqa: E402
    _QueryConfig,
    _query_methods,
)
from detail.gui.modules.analysis_temporal_clustering import (  # noqa: E402
    _TemporalClusteringConfig,
)
from detail.pipeline import temporal_clustering  # noqa: E402

#: (surface id, the registry, the launch form, the CLI command path). One row per registry
#: that carries method parameters — a new one is added here and inherits every test below.
SURFACES = [
    ("query", _query_methods, _QueryConfig, ("analysis", "query")),
    # Pose clustering's row stood here until the module was removed. The temporal-clustering
    # form builds its method rows its own way (text items, its own `_widgets`) and its command
    # takes named options, not `--method-param`, so it has its own cases: TEMPORAL_CASES.
]

CASES = [
    (surface, method_id, methods()[method_id], widget_cls, command)
    for surface, methods, widget_cls, command in SURFACES
    for method_id in sorted(methods())
]


def _ids(case):
    return f"{case[0]}:{case[1]}"


def test_there_is_a_method_to_check():
    """Guard the guard: every test below is parametrized over the registries, so an empty
    one would make the whole file pass while checking nothing."""
    assert CASES


def _select(widget, method_id):
    """Point the form's method combo at ``method_id`` and let it rebuild its rows."""
    index = widget._method.findData(method_id)
    assert index >= 0, f"{method_id} is registered but not offered by the form"
    widget._method.setCurrentIndex(index)
    return widget.config()["method_params"]


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_form_offers_every_parameter_the_method_declares(qapp, case):
    _surface, method_id, method_cls, widget_cls, _command = case
    params = _select(widget_cls(), method_id)
    assert set(params) == set(method_cls.config_model.model_fields), (
        f"{method_id}'s launch form does not carry every parameter of its config model. "
        "A knob that is not on the form cannot be set, and nothing fails when it is missing."
    )


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_form_opens_on_the_method_s_own_defaults(qapp, case):
    """Not the bottom of each range — a form that forgets to seed launches every run at the
    field's `ge` bound, which is legal, silent, and wrong (see ``build_fields``)."""
    _surface, method_id, method_cls, widget_cls, _command = case
    params = _select(widget_cls(), method_id)
    assert params == method_cls.config_model().model_dump()


def _perturb(model_cls, name, widget) -> bool:
    """Move a widget off its opening value. False when the field has only one legal value."""
    if isinstance(widget, QComboBox):
        if widget.count() < 2:
            return False
        widget.setCurrentIndex((widget.currentIndex() + 1) % widget.count())
        return True
    if isinstance(widget, QCheckBox):
        widget.setChecked(not widget.isChecked())
        return True
    step = widget.singleStep()
    target = widget.value() + step
    if target > widget.maximum():
        target = widget.value() - step
    if target < widget.minimum() or target == widget.value():
        return False
    widget.setValue(target)
    return True


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_parameter_a_user_changes_reaches_the_config(qapp, case):
    """The other half. A row can exist, be edited, and still not be read back — which is
    what a hand-written ``for name in METHOD_FIELDS`` does the moment a field is added.
    """
    _surface, method_id, method_cls, widget_cls, _command = case
    widget = widget_cls()
    opened = _select(widget, method_id)
    rows = widget._method_fields._widgets[method_id]
    unmoved = []
    for name in method_cls.config_model.model_fields:
        if not _perturb(method_cls.config_model, name, rows[name]):
            continue
        if widget.config()["method_params"][name] == opened[name]:
            unmoved.append(name)
    assert (
        not unmoved
    ), f"{method_id}: changing these rows did not change the run's config: {unmoved}"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_only_the_selected_method_s_rows_are_shown(qapp, case):
    """Every method's rows are built once; the others are hidden rather than rebuilt. A form
    showing all of them would offer knobs the run will never receive."""
    _surface, method_id, _method_cls, widget_cls, _command = case
    widget = widget_cls()
    _select(widget, method_id)
    for other, rows in widget._method_fields._widgets.items():
        for name, row in rows.items():
            assert row.isHidden() == (
                other != method_id
            ), f"{other}.{name} visibility is wrong while {method_id} is selected"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_cli_takes_every_parameter_the_method_declares(case):
    """Typed as ``NAME=VALUE`` strings and coerced by the method's own model, so the CLI
    never holds a second copy of any field's type."""
    _surface, method_id, method_cls, _widget_cls, _command = case
    raw = {
        field["name"]: str(field["default"])
        for field in describe_config(method_cls.config_model)
        if field["default"] is not None  # `None` means unset; you omit it instead
    }
    resolved = resolve_method_params(method_cls(), raw)
    assert set(resolved) == set(method_cls.config_model.model_fields)


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_the_command_offers_method_param(case):
    """Parity between the two surfaces: a command that takes ``--method`` takes its
    parameters too, or the CLI is offering a choice it cannot configure."""
    _surface, _method_id, _method_cls, _widget_cls, command = case
    result = CliRunner().invoke(main, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert (
        "--method-param" in result.output
    ), f"`{' '.join(command)}` offers --method but no way to set the method's parameters"


def test_an_unknown_parameter_is_refused_rather_than_ignored():
    """Pydantic drops extras silently, so ``--method-param fusin=mean`` would read as
    accepted and run on the default. That is the same defect in a new place."""
    from detail.pipeline.query import resolve_method

    with pytest.raises(ValueError, match="has no parameter"):
        resolve_method_params(resolve_method("soft_knn"), {"kk": "5"})


def test_a_field_shape_the_renderer_cannot_draw_raises_rather_than_vanishing(qapp):
    """Guard the guard, at the seam that actually decides whether the parity above holds.

    With today's models every field renders, so the parity tests pass forever and prove
    nothing about the *next* field. What protects that one is ``any_field_widget`` refusing a
    shape it cannot draw: the failure mode being guarded against is a knob that quietly is
    not there, so a loud error at construction is the fix, and this is the case that proves
    it is still loud. ``build_fields``, which these forms used to be built on, silently
    returns nothing for the same field.
    """
    from pydantic import BaseModel, Field

    from detail.gui._pydantic_form import any_field_widget, build_fields

    class _Unrenderable(BaseModel):
        note: str = Field("hello", description="free text, not a scalar knob")

    with pytest.raises(ValueError, match="build it by hand"):
        any_field_widget(_Unrenderable, "note")
    assert "note" not in build_fields(_Unrenderable), (
        "build_fields skipping the field silently is the behaviour any_field_widget "
        "exists to replace; if this changed, the raise above may no longer be needed"
    )


# --- the temporal-clustering form ---------------------------------------------------------- #
#: (method id, the config model its registered method declares). Read from the registry, never
#: from the form's own tables: those tables are what is being checked.
TEMPORAL_CASES = [
    (method_id, temporal_clustering.resolve_method(method_id).config_model)
    for method_id in _TemporalClusteringConfig._method_ids()
]


def test_there_is_a_temporal_method_to_check():
    """Guard the guard, as above: an empty registry would pass every test below."""
    assert TEMPORAL_CASES


def _select_temporal(widget, method_id):
    """Point the temporal form's method combo at ``method_id``; its items are text."""
    assert (
        widget._method.findText(method_id) >= 0
    ), f"{method_id} is registered but not offered by the form"
    widget._method.setCurrentText(method_id)
    return widget.config()["method_params"]


@pytest.mark.parametrize("case", TEMPORAL_CASES, ids=lambda case: case[0])
def test_the_temporal_form_offers_every_parameter_the_method_declares(qapp, case):
    method_id, model_cls = case
    params = _select_temporal(_TemporalClusteringConfig(), method_id)
    assert set(params) == set(model_cls.model_fields), (
        f"{method_id}'s launch form does not carry every parameter of its config model. "
        "A knob that is not on the form cannot be set, and nothing fails when it is missing."
    )


@pytest.mark.parametrize("case", TEMPORAL_CASES, ids=lambda case: case[0])
def test_the_temporal_form_opens_on_the_method_s_own_defaults(qapp, case):
    method_id, model_cls = case
    params = _select_temporal(_TemporalClusteringConfig(), method_id)
    assert params == model_cls().model_dump()


@pytest.mark.parametrize("case", TEMPORAL_CASES, ids=lambda case: case[0])
def test_every_temporal_parameter_a_user_changes_reaches_the_config(qapp, case):
    method_id, model_cls = case
    widget = _TemporalClusteringConfig()
    opened = _select_temporal(widget, method_id)
    rows = widget._widgets[method_id]["widgets"]
    unmoved = []
    for name in model_cls.model_fields:
        if not _perturb(model_cls, name, rows[name]):
            continue
        if widget.config()["method_params"][name] == opened[name]:
            unmoved.append(name)
    assert (
        not unmoved
    ), f"{method_id}: changing these rows did not change the run's config: {unmoved}"


@pytest.mark.parametrize("case", TEMPORAL_CASES, ids=lambda case: case[0])
def test_only_the_selected_temporal_method_s_rows_are_shown(qapp, case):
    method_id, _model_cls = case
    widget = _TemporalClusteringConfig()
    _select_temporal(widget, method_id)
    for other, rows in widget._rows.items():
        assert rows, f"{other} has no rows on the form"
        for layout, index in rows:
            assert layout.isRowVisible(index) == (
                other == method_id
            ), f"{other} row {index} visibility is wrong while {method_id} is selected"
