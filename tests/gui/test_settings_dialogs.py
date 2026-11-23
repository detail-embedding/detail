"""A settings dialog shows its whole form, and still does after the form is swapped.

**Reported.** "Launch Analysis Settings Window needs to be taller (important buttons are hidden
for query). All settings dialog popups should probably be big enough to show all [settings]
except advanced always. In this same dialog, the Edit non-examples button [is] cut off."

**One class, two symptoms: a size decided at show, for content that changes after it.** The
launch dialogs opened at a typed 560x520 and sized their width floor once, in ``showEvent``. Both
then replace their whole form when you pick another module, and nothing re-asked either
question — so Query's behavior buttons sat below the fold of a form that fit on the screen, and its
row of five buttons was crushed to the first module's width. ``Dialog._refit`` now runs at show
**and** whenever the body changes; a dialog declaring ``fits_content`` also grows to its form.

**Discovered, not listed.** Which dialogs are settings dialogs is read off what they hold — a
module's ``ConfigWidget`` or a ``ConfigForm``'s ``AdvancedSection`` — across every dialog the
builder can make, so a settings dialog added tomorrow fails here until it says ``fits_content``.
"""

import pytest

from detail import surfaces
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers module UIs)
from detail.gui.modules import registry
from detail.gui.qt import QComboBox, QLabel, QListWidget, QPushButton
from tests.gui import subjects

pytestmark = pytest.mark.guard

#: Settings dialogs that do not declare ``fits_content``, and why. Each is a claim, checked by
#: `test_every_exemption_is_still_true`.
EXEMPT: dict[str, str] = {}


#: A name per recording, none of which contains its id — so an id on screen is detectable.
_NAMES = ("Fish tank A", "Fish tank B", "Fish tank C", "Fish tank D")


def _named(experiment):
    for recording, name in zip(experiment.recordings, _NAMES):
        recording.set_name(name)
    return experiment


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """One experiment with a completed run per stage, every recording named. Shared by the
    launch-dialog tests, which read it and write nothing."""
    return _named(
        subjects.an_experiment_with_completed_runs(tmp_path_factory.mktemp("settings"))
    )


@pytest.fixture
def fresh_experiment(tmp_path):
    """Its own experiment, for a sweep that builds every dialog: building some of them
    writes to the experiment, and a later dialog reading that is a different test."""
    return _named(subjects.an_experiment_with_completed_runs(tmp_path))


def _settle(qapp) -> None:
    # The refit is queued behind the change that asks for it, and a refit can itself resize.
    for _ in range(4):
        qapp.processEvents()


def _assert_shows_its_form(dialog) -> None:
    """Every row outside *Advanced settings* in view, and every button whole.

    Collects every problem before failing: the reported dialog had two at once, and an early
    assert on the first would hide whether the second was fixed.
    """
    problems = []
    scroll = dialog._scroll
    overflow = scroll.verticalScrollBar().maximum()
    screen = dialog.screen().availableGeometry().height()
    frame = dialog.frameGeometry().height() - dialog.height()
    if overflow and dialog.height() < screen - frame - 1:
        problems.append(
            f"{dialog.height()}px tall with {overflow}px of its form scrolled out of view, "
            f"on a {screen}px screen it has room to grow into"
        )
    body = scroll.widget()
    viewport = scroll.viewport()
    buttons = [b for b in body.findChildren(QPushButton) if b.isVisibleTo(body)]
    # A button given an explicit width (a `−` beside a list, at 28px) is sized on purpose; one
    # left to its hint can only fall below it when the row it sits in is starved.
    crushed = [
        (b.text(), b.width(), b.minimumSizeHint().width())
        for b in buttons
        if b.minimumWidth() == 0 and b.width() < b.minimumSizeHint().width()
    ]
    if crushed:
        problems.append(f"buttons narrower than their own text: {crushed}")
    # And none running off the right edge. The dialog scrolls vertically only, so a body wider
    # than its viewport is not scrolled to — it is cut, and whatever sits at the end of a row
    # of buttons is what goes.
    cut = [
        (b.text(), b.mapTo(viewport, b.rect().topRight()).x())
        for b in buttons
        if b.mapTo(viewport, b.rect().topRight()).x() >= viewport.width()
    ]
    if cut:
        problems.append(
            f"buttons past the {viewport.width()}px viewport's right edge: {cut}"
        )
    assert not problems, f"{type(dialog).__name__}: " + "; ".join(problems)


def _launch_modules():
    out = []
    for dialog in ("LaunchRunDialog", "LaunchAnalysisDialog"):
        table = (
            registry.GUI_MODULES
            if dialog == "LaunchRunDialog"
            else registry.ANALYSIS_GUI_MODULES
        )
        out += [(dialog, module_id) for module_id in sorted(registry.surfaced(table))]
    return out


@pytest.mark.parametrize(("dialog_name", "module_id"), _launch_modules())
def test_a_module_picked_after_the_dialog_opened_still_fits(
    qapp, experiment, dialog_name, module_id
):
    """The reported path: open the dialog, *then* pick the module — the form arrives after
    the show that sized the window."""
    from detail.gui import launch

    dialog = getattr(launch, dialog_name)(experiment)
    dialog.show()
    _settle(qapp)
    try:
        index = dialog._module_combo.findData(module_id)
        assert index >= 0, f"{module_id} is not offered"
        # Away and back, so the module's form is always one swapped in after the show — even
        # the one the dialog happens to open on.
        dialog._module_combo.setCurrentIndex(0 if index else 1)
        _settle(qapp)
        dialog._module_combo.setCurrentIndex(index)
        _settle(qapp)
        _assert_shows_its_form(dialog)
    finally:
        dialog.close()


def _holds_a_settings_form(dialog) -> bool:
    from detail.gui.forms import AdvancedSection
    from detail.gui.modules.registry import ConfigWidget

    return bool(
        dialog.findChildren(ConfigWidget) or dialog.findChildren(AdvancedSection)
    )


def _settings_dialogs(experiment):
    recording = experiment.get_recording("recording_001")
    found = []
    for cls in subjects.buildable_dialogs():
        dialog = subjects.build_dialog(cls, experiment, recording)
        if _holds_a_settings_form(dialog):
            found.append(cls)
        dialog.close()
        dialog.deleteLater()
    return found


def test_every_settings_dialog_says_it_fits_its_content(qapp, fresh_experiment):
    undeclared = [
        cls.__name__
        for cls in _settings_dialogs(fresh_experiment)
        if not cls.fits_content and cls.__name__ not in EXEMPT
    ]
    assert not undeclared, (
        f"{undeclared} hold a config form but do not set `fits_content = True`, so they open "
        "at whatever size was typed and scroll a form that would fit on the screen"
    )


def test_there_are_settings_dialogs_to_check(qapp, fresh_experiment):
    """Guard the guard: a discovery that finds nothing passes forever."""
    found = {cls.__name__ for cls in _settings_dialogs(fresh_experiment)}
    assert {"LaunchRunDialog", "LaunchAnalysisDialog", "DistillDialog"} <= found, found


def test_every_exemption_is_still_true():
    """An exemption names a hidden surface. The day it is switched back on, it is a real
    settings dialog again and has to declare itself."""
    for name, surface in EXEMPT.items():
        assert surface in surfaces.HIDDEN, (
            f"{name} is exempt because {surface} is hidden, and it no longer is — give it "
            "`fits_content = True` and drop the exemption"
        )


@pytest.mark.parametrize(
    "name",
    [
        "ComparisonLaunchDialog",
        "DistillDialog",
        "GenerateDatasetDialog",
        "GenerateLabelingDialog",
        "TrainModelDialog",
    ],
)
def test_a_settings_dialog_opens_with_its_form_in_view(qapp, experiment, name):
    (cls,) = [c for c in subjects.buildable_dialogs() if c.__name__ == name]
    dialog = subjects.build_dialog(
        cls, experiment, experiment.get_recording("recording_001")
    )
    dialog.show()
    _settle(qapp)
    try:
        _assert_shows_its_form(dialog)
    finally:
        dialog.close()


@pytest.mark.parametrize(("dialog_name", "module_id"), _launch_modules())
def test_a_launch_dialog_names_recordings_by_name(
    qapp, experiment, dialog_name, module_id
):
    """§11: the id is the tool's, the name is the user's. `LaunchAnalysisDialog` listed
    ``recording_001`` while its processing twin, forty lines up, listed names — past a
    guard that scanned for three exact spellings of the mistake. This one reads what the
    dialog actually shows, for every module either dialog offers."""
    from detail.gui import launch

    dialog = getattr(launch, dialog_name)(experiment)
    try:
        dialog._module_combo.setCurrentIndex(dialog._module_combo.findData(module_id))
        shown = [
            dialog._recordings.item(i).text() for i in range(dialog._recordings.count())
        ]
        names = [recording.name for recording in experiment.recordings]
        assert shown and all(
            text.startswith(name) for text, name in zip(shown, names)
        ), shown
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        for combo in dialog.findChildren(QComboBox):
            texts += [combo.itemText(i) for i in range(combo.count())]
        for listing in dialog.findChildren(QListWidget):
            texts += [listing.item(i).text() for i in range(listing.count())]
        ids = [recording.id for recording in experiment.recordings]
        leaked = [text for text in texts if any(rid in text for rid in ids)]
        assert not leaked, f"a recording id where its name belongs: {leaked}"
    finally:
        dialog.close()
