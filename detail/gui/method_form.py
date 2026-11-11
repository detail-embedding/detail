"""A registry method's parameters, rendered from the method's own config model.

§20's query methods are a registry (so were §22's pose-clustering methods, until pose clustering
was removed): a method declares a ``config_model`` and the module never looks inside it. Both
modules' surfaces nevertheless reached
those parameters by **retyping them at the call site** — the querying form and CLI sent
``method_params={}`` and offered none of `dtw`'s five, while the pose-clustering form built
``build_fields(FinchPoseConfig)`` against a hand-written tuple of four names and its CLI
offered one of them. So a method's knobs were reachable only if a surface had been written
with that method in mind, and a second registered method's parameters were dropped by both
with nothing to see: the form renders, the run launches, the parameter is simply never
settable.

This builds every registered method's rows once and shows the selected method's. Two
consequences worth knowing:

- The rows go through ``ConfigForm.addRow`` individually, **not** into a container, because
  a method's model declares per-field ``advanced`` flags and the flagged ones have to reach
  the advanced section while the others stay in the body. A widget that held them together
  could not split them.
- Hidden rows are excluded from the advanced section's count and its "changed" tally
  (``gui/forms.py``), or a form carrying three methods would advertise every method's knobs
  at once and open itself over an edit nobody can see.

A leaf: it imports ``_pydantic_form`` and ``help_ui`` and nothing from ``detail.gui.pages``
or ``detail.gui.modules``. Both of this package's shipped import cycles ran through a shared
helper that lived inside a package whose ``__init__`` imports the world
(``tests/gui/test_import_cycles.py``).
"""

from PyQt6.QtWidgets import QWidget

from detail.configs.describe import describe_config
from detail.gui._pydantic_form import (
    any_field_value,
    any_field_widget,
    set_any_field_value,
    field_advanced,
)
from detail.gui.help_ui import field_label


class MethodFields:
    """Every registered method's parameter rows on one form, one method's shown at a time.

    ``methods`` maps a method id to the class declaring ``config_model`` — a registry dict
    (``QUERY_METHODS``, ``POSE_CLUSTER_METHODS``) passed straight in, so a method registered
    tomorrow is offered without touching a surface.

    The help anchor for each row is the **method id**, so a knob documents itself on the
    method's own generated page rather than the module's.
    """

    def __init__(self, form, methods: dict[str, type]) -> None:
        self._form = form
        self._models = {
            method_id: cls.config_model
            for method_id, cls in methods.items()
            if getattr(cls, "config_model", None) is not None
        }
        self._widgets: dict[str, dict[str, QWidget]] = {}
        for method_id in sorted(self._models):
            model = self._models[method_id]
            row_widgets: dict[str, QWidget] = {}
            for field in describe_config(model):
                name = field["name"]
                widget = any_field_widget(model, name)
                form.addRow(
                    field_label(method_id, name, f"{field['label']}:"),
                    widget,
                    advanced=field_advanced(model, name),
                )
                row_widgets[name] = widget
            self._widgets[method_id] = row_widgets

    def method_ids(self) -> list[str]:
        """The methods this carries rows for, sorted — what a picker should offer."""
        return sorted(self._widgets)

    def show(self, method_id: str | None) -> None:
        """Show ``method_id``'s rows and hide every other method's."""
        for other, widgets in self._widgets.items():
            for widget in widgets.values():
                self._form.set_row_visible(widget, other == method_id)

    def set_params(self, method_id: str | None, params: dict) -> None:
        """Write a previous run's method parameters back into that method's rows.

        The setter twin of :meth:`params`, and it has to exist here rather than at each call
        site for the same reason ``set_any_field_value`` does: a module that hand-rolled the
        dispatch would restore only the widget shapes its author remembered, and a knob that
        quietly keeps its default is invisible.

        A key the method does not declare is ignored — a run fit with an older version of the
        method carries parameters this one has never heard of, and dropping them is right.
        """
        widgets = self._widgets.get(method_id or "")
        if not widgets:
            return
        model = self._models[method_id]
        for name, widget in widgets.items():
            if name in params:
                set_any_field_value(model, name, widget, params[name])

    def params(self, method_id: str | None) -> dict:
        """The selected method's parameters, read back in its model's own types.

        ``{}`` for a method with no rows or none selected — which is what the module's
        config carried before any of this existed, so an unselected picker still validates
        with the same message it always did.
        """
        widgets = self._widgets.get(method_id or "")
        if not widgets:
            return {}
        model = self._models[method_id]
        return {
            name: any_field_value(model, name, widget)
            for name, widget in widgets.items()
        }
