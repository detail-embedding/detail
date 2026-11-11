"""Fail-closed help coverage: every labeled row of every launch form carries a ``?``.

The field ``?`` rollout can silently miss a row (a composite widget, a
dynamically-built knob, a nested-input group) — import and construction both pass
while the field has no in-app docs. This test builds every module / analysis /
tool config form and asserts each labeled row either has a ``?`` help button or is
on the explicit ``EXEMPT`` allow-list (display-only rows, or a sub-row documented by
a sibling). A new form field with no ``?`` fails here until it is documented or
exempted with a reason.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFormLayout, QLabel, QToolButton  # noqa: E402

from detail.api._runner_modules import PIPELINE_RUNNERS  # noqa: E402
from detail.executor.runners import import_runner_modules  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import (  # noqa: E402
    ANALYSIS_GUI_MODULES,
    GUI_MODULES,
)
from detail.gui.tools.registry import TOOL_GUIS, import_tool_uis  # noqa: E402

# (form_id, label_text) rows that legitimately carry no ``?``. Each needs a reason.
EXEMPT: set[tuple[str, str]] = {
    # Compute-target sub-rows: the "Backend:" row of the same group carries the
    # compute-target ?, so the device / source sub-rows don't repeat it.
    ("detection", "CUDA device:"),
    ("embedding", "CUDA device:"),
    ("keypoint_detect", "CUDA device:"),
    ("quick_segment", "CUDA device:"),
    # The shared example-set picker's own two rows: the detect form's "Examples:" row
    # around it carries the example_set_id ?, so its Skeleton / Example set sub-rows don't
    # repeat it (the same reasoning as the compute-target sub-rows above).
    ("keypoint_detect", "Skeleton:"),
    ("keypoint_detect", "Example set:"),
    # Downsampling: a read-only display of the source video resolution, and a mode
    # selector whose target field ("Resize to (px):") carries the ?.
    #
    # The launch dialog below has **no user documentation to point a ? at**: the dataset rows
    # would each need a term written for them. A ? that opens the wrong page is worse than none.
    # They are exempt until that screen is documented, not because the rows do not deserve
    # help — and being listed here means a *new* row on the dialog still has to justify itself.
    ("GenerateDatasetDialog", "Name:"),
    ("GenerateDatasetDialog", "Source run:"),
    ("GenerateDatasetDialog", "Frames:"),
    ("GenerateDatasetDialog", "N:"),
    ("GenerateDatasetDialog", "Validation split:"),
    ("GenerateDatasetDialog", "Min contour area (px):"),
}


def _row_label_and_help(item) -> tuple[str | None, bool]:
    """A form row's label text and whether it contains a ``?`` help button."""
    if item is None or item.widget() is None:
        return (None, False)
    widget = item.widget()
    has_help = isinstance(widget, QToolButton) or bool(widget.findChildren(QToolButton))
    if isinstance(widget, QLabel):
        text = widget.text()
    else:
        labels = widget.findChildren(QLabel)
        text = labels[0].text() if labels else ""
    return (text, has_help)


class _DialogForm:
    """A launch dialog, wrapped so it looks like a registry form to the sweep below."""

    def __init__(self, build) -> None:
        self._build = build

    def config_widget(self):
        return self._build()


def _launch_dialogs(tmp_path) -> dict[str, object]:
    """The config surfaces that are **not** in any registry.

    These were the whole gap. A launch dialog is a config form — it asks the same kind of
    questions with the same kind of rows — but `GUI_MODULES` / `ANALYSIS_GUI_MODULES` /
    `TOOL_GUIS` do not contain one, so nothing here ever built one and every house rule about
    form rows went unenforced on them. Measured when this was written: registry forms carried a
    `?` on 94 of 97 advanced rows; the launch dialogs were at 0/7, 0/2, 1/6 and 6/12.
    """
    from detail.api.experiment import Experiment
    from detail.gui import comparison as gui_comparison
    from detail.gui import fit as gui_fit

    experiment = Experiment.create("p", tmp_path)
    experiment.create_recording("recording_001")
    return {
        "ComparisonLaunchDialog": _DialogForm(
            lambda: gui_comparison.ComparisonLaunchDialog(experiment)
        ),
        "GenerateDatasetDialog": _DialogForm(
            lambda: gui_fit.GenerateDatasetDialog(experiment)
        ),
    }


def _all_forms() -> dict[str, object]:
    import_runner_modules(PIPELINE_RUNNERS)  # populate TRACKERS (tracking knob rows)
    import_tool_uis()
    return {**GUI_MODULES, **ANALYSIS_GUI_MODULES, **TOOL_GUIS}


@pytest.mark.parametrize("form_id", sorted(_all_forms()))
def test_every_labeled_form_row_has_help_or_is_exempt(qapp, form_id):
    widget = _all_forms()[form_id].config_widget()
    missing = []
    for layout in widget.findChildren(QFormLayout):
        for row in range(layout.rowCount()):
            text, has_help = _row_label_and_help(
                layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            )
            if not text or not text.strip():
                continue
            if not has_help and (form_id, text) not in EXEMPT:
                missing.append(text)
    assert not missing, (
        f"{form_id} form has labeled rows with no ? help and not in EXEMPT: "
        f"{missing}. Wire a field/help ? or add an entry to EXEMPT with a reason."
    )


def test_exempt_entries_are_all_still_present(qapp, tmp_path):
    """An EXEMPT row that no longer exists is stale — remove it so the allow-list
    can't quietly cover a row that was renamed or now carries a ?."""
    # Both sets, or a launch dialog's entry looks stale merely because this walk never built one
    # — which is the same blind spot that let those rows go uncovered in the first place.
    forms = {**_all_forms(), **_launch_dialogs(tmp_path)}
    live = set()
    for form_id, ui in forms.items():
        widget = ui.config_widget()
        for layout in widget.findChildren(QFormLayout):
            for row in range(layout.rowCount()):
                text, has_help = _row_label_and_help(
                    layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                )
                if text and text.strip() and not has_help:
                    live.add((form_id, text))
    stale = EXEMPT - live
    assert not stale, f"stale EXEMPT entries (row gone or now has ?): {stale}"


#: Named here rather than derived, because building one needs an experiment on disk and
#: parametrisation happens at collection time.
LAUNCH_DIALOGS = (
    "ComparisonLaunchDialog",
    "GenerateDatasetDialog",
)


@pytest.mark.parametrize("form_id", LAUNCH_DIALOGS)
def test_every_launch_dialog_row_has_help_or_is_exempt(qapp, tmp_path, form_id):
    """The same rule, on the surfaces no registry lists.

    Kept a separate test rather than folded into `_all_forms` because a dialog needs an
    experiment on disk and the registry forms do not — but it is the same assertion, reading the
    same `EXEMPT` table, so a row here is held to exactly the standard a module form's row is.
    """
    widget = _launch_dialogs(tmp_path)[form_id].config_widget()
    missing = []
    for layout in widget.findChildren(QFormLayout):
        for row in range(layout.rowCount()):
            text, has_help = _row_label_and_help(
                layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
            )
            if not text or not text.strip():
                continue
            if not has_help and (form_id, text) not in EXEMPT:
                missing.append(text)
    assert not missing, (
        f"{form_id} has labeled rows with no ? help and not in EXEMPT: {missing}. "
        f"Wire a field/help ? or add an entry to EXEMPT with a reason."
    )


def test_the_launch_dialog_list_matches_what_the_builder_knows(qapp, tmp_path):
    """Guard the guard: a name dropped from `LAUNCH_DIALOGS` would silently stop being checked."""
    assert set(LAUNCH_DIALOGS) == set(_launch_dialogs(tmp_path))
