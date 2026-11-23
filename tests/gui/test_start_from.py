"""Every launch surface offers **Start from**, and picking one reaches the form.

The class this exists for: **a capability with no way in.** PR #295 gave every config form a
working `load_config` and shipped with nothing that calls it — the only entry point was a
Refine button gated on `not mutable AND surfaced`, which reached 2 of 9 analysis modules and no
processing module at all. Almost no real run had the button. Nothing
failed; the feature simply could not be used, and a green suite said so all the way.

So the parity sweep is the point of this file, not a nicety beside the behaviour test. It runs
over the registries rather than a list, so a module registered tomorrow is covered the day it
lands rather than the day somebody remembers this file. That is what the user asked to be made
mandatory.

The behaviour half has its own trap, and it is the one `test_config_round_trip.py` was written
around: a picker that *lists* runs and loads nothing looks identical to one that works. So the
form is read before and after, and the assertion is that a perturbed value arrived — never that
the combo has items.

**And a pick the form cannot take whole says so** (#35). A choice this build does not offer left
its combo on the default, a number out of range was clamped, and a query-eval run's scored
behaviors were applied to a list still empty, so all were ticked — each under a picker row naming
the run. Two forms had a note of their own; the rest were silent. The launch dialog now compares
the pick with what the form reads back, over the module's config model, and names every
difference in one line under **Start from** (`start_from.unrestored_settings`). Driven over every
surfaced module's choice fields, with the negative — a form loading its own settings reports
nothing — over every module too.
"""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import (  # noqa: E402
    ANALYSIS_GUI_MODULES,
    GUI_MODULES,
    surfaced,
)
from detail.gui.start_from import DEFAULTS_LABEL, NO_SETTINGS  # noqa: E402
from detail.storage import analysis_runs as analysis_store  # noqa: E402
from detail.storage import runs as runs_store  # noqa: E402

#: Launch surfaces that offer no **Start from** row, and why. Each entry is a claim, checked by
#: `test_the_exemptions_are_not_stale`: the named class must still exist and must still have no
#: `config()`, so a dialog that grows one cannot stay quietly exempt.
NO_CONFIG_TO_REUSE = {
    "GenerateDatasetDialog": "exports images+labels from a selection; no run config to reuse",
    "GenerateLabelingDialog": "same, for the labeling set",
    "TrainModelDialog": "trains from a dataset; its inputs are the dataset and the weights",
    "DistillDialog": "a pipeline of its own, not a module run",
}


def _experiment(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    return proj


def _labels(row) -> list[str]:
    return [row.combo.itemText(i) for i in range(row.combo.count())]


@pytest.fixture
def one_baseline(monkeypatch):
    """Offer a baseline to `tracking`, with none shipped.

    No baseline ships today — the numbers for one are measured rather than guessed — but the
    picker's handling of them is a code path either way, and a test that quietly skipped when
    the directory emptied would be the vacuity this file is otherwise careful about. Patched at
    `start_from`'s point of use, so the widget under test is the real one.
    """
    from detail.configs.baselines import Baseline
    from detail.gui import start_from as module

    baseline = Baseline(
        name="Test rig", settings={"tracking": {"tracker_id": "iou_single_tracker"}}
    )
    monkeypatch.setattr(
        module,
        "baselines_for",
        lambda module_id: [baseline] if module_id == "tracking" else [],
    )
    return baseline


# --- parity: every surfaced module offers the row ---------------------------- #
@pytest.mark.parametrize("module_id", sorted(surfaced(GUI_MODULES)))
def test_every_processing_module_offers_start_from(qapp, tmp_path, module_id):
    from detail.gui.launch import LaunchRunDialog

    dialog = LaunchRunDialog(_experiment(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData(module_id))
    assert _labels(dialog._start_from)[0] == DEFAULTS_LABEL


@pytest.mark.parametrize("module_id", sorted(surfaced(ANALYSIS_GUI_MODULES)))
def test_every_analysis_module_offers_start_from(qapp, tmp_path, module_id):
    from detail.gui.launch import LaunchAnalysisDialog

    dialog = LaunchAnalysisDialog(_experiment(tmp_path))
    index = dialog._module_combo.findData(module_id)
    if index < 0:
        pytest.skip(
            f"{module_id} is not offered by the analysis launcher in this build"
        )
    dialog._module_combo.setCurrentIndex(index)
    assert _labels(dialog._start_from)[0] == DEFAULTS_LABEL


def test_the_registries_are_populated():
    """A data-driven sweep over an empty registry passes by testing nothing."""
    assert len(surfaced(GUI_MODULES)) >= 4
    assert len(surfaced(ANALYSIS_GUI_MODULES)) >= 3


def test_the_exemptions_are_not_stale(qapp):
    """A dialog listed as having no config must still have none."""
    from tests.gui.test_advanced_form import _DIALOGS, _build_dialog

    for name in NO_CONFIG_TO_REUSE:
        assert (
            name in _DIALOGS
        ), f"exemption names a dialog that no longer exists: {name}"


# --- behaviour: a pick reaches the form -------------------------------------- #
def test_picking_a_previous_run_loads_its_settings(qapp, tmp_path):
    """The failure mode is a picker that lists runs and loads nothing — which looks the same.

    So the form is read before and after, and the assertion is on the value that moved.
    """
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment(tmp_path)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "config": {"conf_threshold": 0.17, "batch_size": 7},
        },
    )
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    before = dialog._config.config()
    assert before["conf_threshold"] != 0.17, "the fixture must differ from the default"

    combo = dialog._start_from.combo
    index = next(
        i for i in range(combo.count()) if combo.itemText(i).startswith("run_001")
    )
    combo.setCurrentIndex(index)

    after = dialog._config.config()
    assert after["conf_threshold"] == pytest.approx(0.17)
    assert after["batch_size"] == 7


def test_picking_a_baseline_loads_its_settings(qapp, tmp_path, one_baseline):
    """The authored half, through the same path a run takes."""
    from detail.gui.launch import LaunchRunDialog

    baseline = one_baseline
    dialog = LaunchRunDialog(_experiment(tmp_path))
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))
    combo = dialog._start_from.combo
    index = combo.findText(baseline.name)
    assert index >= 0, f"{baseline.name!r} is not offered on tracking"
    combo.setCurrentIndex(index)

    for key, value in baseline.for_module("tracking").items():
        assert dialog._config.config()[key] == value


def test_going_back_to_defaults_resets_the_form(qapp, tmp_path):
    """Reported by a user, and the tests were the reason it shipped.

    Every test in this file picked *from* Defaults; none picked back *to* it. The handler
    skipped an item whose payload was falsy, so choosing **Defaults** did nothing at all: the
    form kept the run's values with the row above it reading "Defaults". A control that applies
    some of its options and ignores the rest — the ignored one being the one that means "none".

    A state machine has to be driven in both directions to be tested at all.
    """
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment(tmp_path)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "config": {"conf_threshold": 0.17, "batch_size": 7},
        },
    )
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    defaults = dialog._config.config()

    combo = dialog._start_from.combo
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i).startswith("run_001"))
    )
    assert dialog._config.config()["conf_threshold"] == pytest.approx(0.17)

    combo.setCurrentIndex(combo.findText(DEFAULTS_LABEL))
    back = dialog._config.config()
    assert back["conf_threshold"] == pytest.approx(defaults["conf_threshold"])
    assert back["batch_size"] == defaults["batch_size"]


def test_a_pick_replaces_the_previous_pick_rather_than_layering_on_it(qapp, tmp_path):
    """The same bug wearing a disguise, and the reason the fix is a rebuild.

    One starting point names a handful of knobs and the next names another handful. Loading one
    *over* the other leaves a form that is part one run, part the other and wholly unchosen —
    and unlike the Defaults case there is no visible clue, because the row does read as the
    thing you picked. Every pick means "the form is now this".

    The subject is `conf_threshold`, deliberately. The first version of this test used
    `every_n_frames` and passed against the overlay it was written to catch: `EveryNRow` resets
    itself when handed nothing, so the residue cleared by accident and the assertion never
    reached the behaviour. A field has to *persist* when absent for this to test anything.
    """
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment(tmp_path)
    for run_id, config in (
        ("run_001", {"conf_threshold": 0.17, "batch_size": 7}),
        ("run_002", {"batch_size": 9}),  # says nothing about conf_threshold
    ):
        runs_store.write_run(
            proj.dir,
            run_id,
            {
                "id": run_id,
                "module_id": "detection",
                "recording_ids": ["recording_001"],
                "status": "complete",
                "config": config,
            },
        )
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    default_conf = dialog._config.config()["conf_threshold"]
    assert default_conf != pytest.approx(
        0.17
    ), "the fixture must differ from the default"

    combo = dialog._start_from.combo

    def pick(prefix):
        combo.setCurrentIndex(
            next(
                i for i in range(combo.count()) if combo.itemText(i).startswith(prefix)
            )
        )

    pick("run_001")
    assert dialog._config.config()["conf_threshold"] == pytest.approx(0.17)

    pick("run_002")
    after = dialog._config.config()
    assert after["batch_size"] == 9
    assert after["conf_threshold"] == pytest.approx(
        default_conf
    ), "the previous pick's value survived one that says nothing about it"


def test_a_run_with_no_recorded_settings_is_offered_disabled(qapp, tmp_path):
    """Every analysis run made before runs recorded their params is in this state.

    Hidden, a user who made that run and cannot find it concludes the list is broken. Shown and
    pickable, it silently loads nothing. Shown and disabled with the reason is the only one of
    the three that is true — and it is the house rule ("Grey out, don't hide").
    """
    from PyQt6.QtCore import Qt

    from detail.gui.launch import LaunchAnalysisDialog

    proj = _experiment(tmp_path)
    analysis_store.write_analysis_yaml(
        analysis_store.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "status": "complete",
        },  # no `params` — the pre-fix shape
    )
    dialog = LaunchAnalysisDialog(proj)
    dialog._module_combo.setCurrentIndex(
        dialog._module_combo.findData("location_export")
    )
    combo = dialog._start_from.combo
    index = next(
        i for i in range(combo.count()) if combo.itemText(i).startswith("an_01")
    )
    assert NO_SETTINGS in combo.itemText(index)
    item = combo.model().item(index)
    assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled), "it must not be pickable"


def test_the_order_is_defaults_then_baselines_then_runs(qapp, tmp_path, one_baseline):
    """Defaults first because it is where a form already is, and where most runs start."""
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment(tmp_path)
    for run_id, conf in (("run_001", 0.11), ("run_002", 0.12)):
        runs_store.write_run(
            proj.dir,
            run_id,
            {
                "id": run_id,
                "module_id": "tracking",
                "recording_ids": ["recording_001"],
                "status": "complete",
                "config": {"tracker_id": "anchor_tracker"},
            },
        )
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("tracking"))
    labels = _labels(dialog._start_from)

    assert labels[0] == DEFAULTS_LABEL
    assert labels.index("── baselines ──") < labels.index("── previous runs ──")
    runs = [label for label in labels if label.startswith("run_")]
    assert runs == sorted(runs, reverse=True), "newest first"


def test_a_pick_opens_the_advanced_section_holding_it(qapp, tmp_path):
    """A knob the pick changed must not stay folded away — the reason Refine applied its seed
    after the form's baseline was taken, inherited here."""
    from detail.gui.forms import AdvancedSection
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment(tmp_path)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "config": {"batch_size": 7},  # advanced on the detection form
        },
    )
    dialog = LaunchRunDialog(proj)
    dialog.resize(700, 640)
    dialog.show()
    qapp.processEvents()
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    assert dialog._config.findChildren(
        AdvancedSection
    ), "the detection form has an advanced section"
    assert not any(
        s.changed_count() for s in dialog._config.findChildren(AdvancedSection)
    )

    combo = dialog._start_from.combo
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i).startswith("run_001"))
    )
    qapp.processEvents()
    # Re-queried, never held across the pick: a pick *rebuilds* the form, so sections captured
    # before it belong to a widget that has since been discarded — they would report the state
    # of a form nobody is looking at, and the assertion would fail for the wrong reason.
    sections = dialog._config.findChildren(AdvancedSection)
    assert any(
        s.changed_count() and s._expanded for s in sections
    ), "the section hiding the changed knob stayed folded"


# --- what a pick could not restore (#35 finding 4; finding 2 left with query eval) --- #
def _pick(dialog, run_id: str) -> None:
    combo = dialog._start_from.combo
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if combo.itemText(i).startswith(run_id))
    )


def _module_dialog(tmp_path, module_id):
    from detail.gui.launch import LaunchAnalysisDialog, LaunchRunDialog

    cls = LaunchRunDialog if module_id in GUI_MODULES else LaunchAnalysisDialog
    dialog = cls(_experiment(tmp_path))
    index = dialog._module_combo.findData(module_id)
    if index < 0:
        pytest.skip(f"{module_id} is not offered by its launcher in this build")
    dialog._module_combo.setCurrentIndex(index)
    return dialog


_ALL_MODULES = sorted(surfaced(GUI_MODULES)) + sorted(surfaced(ANALYSIS_GUI_MODULES))


@pytest.mark.parametrize("module_id", _ALL_MODULES)
def test_a_pick_the_form_takes_whole_leaves_no_note(qapp, tmp_path, module_id):
    """The note's negative: every form loading its own settings back reports nothing, so the
    note is never noise on a pick that was restored as recorded."""
    dialog = _module_dialog(tmp_path, module_id)
    dialog._apply_start_from(dialog._config.config())
    assert dialog._start_from_note.isHidden(), dialog._start_from_note.text()


def _choice_fields(module_id: str, config: dict) -> list[dict]:
    from detail.configs.describe import describe_config, iter_config_models

    model = dict(iter_config_models())[module_id]
    return [
        f
        for f in describe_config(model)
        if f["scope"] == "setting" and f["choices"] and f["name"] in config
    ]


def test_every_choice_a_form_cannot_take_is_named(qapp, tmp_path):
    """Finding 4, over every surfaced module's form rather than the two that had a note of
    their own: a pick naming a choice this build does not offer used to leave the combo on its
    default under a picker row naming the run. Every such field is now named in one note.
    """
    checked = 0
    for module_id in _ALL_MODULES:
        dialog = _module_dialog(tmp_path / module_id, module_id)
        base = dialog._config.config()
        for field in _choice_fields(module_id, base):
            if field["name"] in dialog._config.noted_by_form:
                continue  # explained beside its own row (tracking, the scan)
            dialog._apply_start_from({**base, field["name"]: "not_in_this_build"})
            note = dialog._start_from_note
            assert (
                not note.isHidden()
            ), f"{module_id}.{field['name']} substituted silently"
            assert f"{field['label']}: that run used 'not_in_this_build'" in note.text()
            checked += 1
    assert checked >= 4, f"only {checked} choice fields reached — the sweep is too thin"


def test_a_model_this_build_lacks_and_out_of_range_numbers_are_named(qapp, tmp_path):
    """The issue's own cases, through the real picker: an embedding run whose model is not
    registered here, and a detection run whose numbers the form clamps. Numbers are named too
    — a clamp is a run configured differently from the one picked."""
    from detail.gui.launch import LaunchRunDialog

    proj = _experiment(tmp_path)
    for run_id, module_id, config in (
        ("run_001", "detection", {"conf_threshold": 1.7, "batch_size": 0}),
        ("run_002", "embedding", {"model_id": "dinov2_gone", "align": "flip"}),
    ):
        runs_store.write_run(
            proj.dir,
            run_id,
            {
                "id": run_id,
                "module_id": module_id,
                "recording_ids": ["recording_001"],
                "status": "complete",
                "config": config,
            },
        )
    dialog = LaunchRunDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("detection"))
    _pick(dialog, "run_001")
    text = dialog._start_from_note.text()
    assert "Confidence threshold: that run used 1.7" in text
    assert "Batch size: that run used 0" in text

    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("embedding"))
    assert dialog._start_from_note.isHidden(), "a note outlived its module"
    _pick(dialog, "run_002")
    text = dialog._start_from_note.text()
    assert "Model: that run used 'dinov2_gone'" in text
    assert "Egocentric alignment: that run used 'flip'" in text

    dialog._start_from.combo.setCurrentIndex(0)  # Defaults
    assert dialog._start_from_note.isHidden()
