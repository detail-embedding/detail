"""Export frame… / Export video… / Export points… in the run viewers, through the page that opens
them (issue #690).

**The class: a capability added to one of two entry points to the same widget.** The detection and
tracking viewers had Export buttons from the day the export engine learned the preview targets. They
reached the viewer through the module's `result_widget`, which passes the experiment and the run the
buttons need. The run-detail page stopped using that path: it lists the run's recordings and builds each
one's viewer through `ModuleUI.preview_widget(recording, run_id)`, which had no way to carry either, so
the viewer built without them hid its own buttons ("a caller that predates the export buttons"). Nothing
failed. Every viewer test built the widget directly, with the context, which is the one way it worked.

So these build the viewer **the way a user reaches it**: `RunDetailPage` on a real experiment with a
completed run of each module, and `RecordingRunDetailPage` for the run that has no experiment-level
record. What decides whether the buttons exist is whether the page handed the viewer an experiment-level
run, so that is what is asserted, and the click is followed to the dialog it opens.
"""

import inspect

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton  # noqa: E402

import detail.gui.export as export_module  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import GUI_MODULES, ModuleUI  # noqa: E402
from detail.gui.pages import RecordingRunDetailPage, RunDetailPage  # noqa: E402
from detail.gui.run_preview import DetectionRunPreview, TrackingRunPreview
from tests.gui import subjects  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402

VIEWER = {
    "detection": DetectionRunPreview,
    "tracking": TrackingRunPreview,
}
VIDEO_EXPORTS = ("Export frame…", "Export video…")


@pytest.fixture
def experiment(tmp_path):
    return subjects.an_experiment_with_completed_runs(tmp_path)


def _buttons(viewer) -> dict[str, QPushButton]:
    return {
        b.text(): b for b in viewer.findChildren(QPushButton) if "Export" in b.text()
    }


def _built(qapp, viewer, *names) -> dict[str, QPushButton]:
    """The viewer's export buttons once it has read its run (they are built after the load)."""
    assert drain_until(
        qapp, lambda: all(name in _buttons(viewer) for name in names), 15_000
    ), f"{names} never appeared; there are {sorted(_buttons(viewer))}"
    return _buttons(viewer)


def _run_page(experiment, module_id):
    record = next(r for r in experiment.processing_runs if r["module_id"] == module_id)
    return record, RunDetailPage(experiment, record["id"])


@pytest.fixture
def opened(monkeypatch):
    """What `open_export` was asked for. The dialog is the shared one, tested on its own."""
    calls = []
    monkeypatch.setattr(
        export_module,
        "open_export",
        lambda parent, subject, scope=None, *, only=(), seed=None: calls.append(
            {"subject": subject, "scope": scope, "only": only, "seed": seed}
        ),
    )
    return calls


@pytest.mark.parametrize("module_id", ["detection", "tracking"])
def test_the_viewer_on_the_run_detail_page_offers_the_video_exports(
    qapp, experiment, module_id
):
    _record, page = _run_page(experiment, module_id)
    (viewer,) = page.findChildren(VIEWER[module_id])

    buttons = _built(qapp, viewer, *VIDEO_EXPORTS)

    assert all(not buttons[name].isHidden() for name in VIDEO_EXPORTS)


@pytest.mark.parametrize("module_id", ["detection", "tracking"])
def test_exporting_from_the_page_is_of_the_experiment_s_run_and_the_recording_on_screen(
    qapp, experiment, opened, module_id
):
    record, page = _run_page(experiment, module_id)
    (viewer,) = page.findChildren(VIEWER[module_id])
    buttons = _built(qapp, viewer, *VIDEO_EXPORTS)

    buttons["Export video…"].click()

    (call,) = opened
    assert call["subject"].kind == "run"
    # the experiment's run (run_00N), not the recording's own det_01 / track_01 / kp_01
    assert call["subject"].id == record["id"]
    assert call["only"] == ("preview_video",)
    assert call["seed"]["recording_id"] == "recording_001"
    assert call["scope"] is None


@pytest.mark.parametrize("module_id", ["detection", "tracking"])
def test_only_a_keypoints_viewer_offers_the_points_export(qapp, experiment, module_id):
    _record, page = _run_page(experiment, module_id)
    (viewer,) = page.findChildren(VIEWER[module_id])

    buttons = _built(qapp, viewer, *VIDEO_EXPORTS)

    assert "Export points…" not in buttons


def test_a_run_with_no_experiment_level_record_has_nothing_to_export_from(
    qapp, experiment
):
    """A run launched per recording (the API, the CLI) has a `track_01` and no `run_00N`, and an
    export is *of* the run. The viewer built for it says so by not offering a button that cannot
    succeed — a choice the page makes (`parent_run_id=None`), not a silent gap."""
    page = RecordingRunDetailPage(experiment, "recording_001", "tracking", "track_01")
    (viewer,) = page.findChildren(TrackingRunPreview)

    assert drain_until(qapp, lambda: hasattr(viewer, "_scrubber"), 15_000)

    assert all(b.isHidden() for b in _buttons(viewer).values())


def test_the_base_declares_the_export_context_as_required():
    """A new module copies the base's signature, so the base is where the rule starts."""
    parameters = inspect.signature(ModuleUI.preview_widget).parameters

    for name in ("experiment_dir", "parent_run_id"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize("module_id", sorted(GUI_MODULES))
def test_every_per_recording_viewer_is_given_the_export_context(module_id):
    """A module that overrides `preview_widget` must take the experiment and its run, as required
    keywords: a module that dropped them would build a viewer that looks complete and has no Export,
    which is exactly how the run-detail page lost it. The base class says it the same way.
    """
    ui = GUI_MODULES[module_id]
    if type(ui).preview_widget is ModuleUI.preview_widget:
        pytest.skip("this module has no per-recording viewer")

    parameters = inspect.signature(ui.preview_widget).parameters

    for name in ("experiment_dir", "parent_run_id"):
        assert name in parameters, f"{module_id}.preview_widget does not take {name}"
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty, (
            f"{module_id}.preview_widget gives {name} a default: a caller that forgets it "
            "gets a viewer with no Export instead of an error"
        )
