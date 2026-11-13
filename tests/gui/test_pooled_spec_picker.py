"""The analysis upstream picker that may tick several sets of settings (§6.5).

``PooledSpecPicker`` exists because recordings whose upstream came through different chains —
two strains, each detected by its own model — share no one set of settings, so a single-choice
picker could only ever pool half of them. These drive it with synthetic groups (the
``run_spec.group_specs`` shape), and drive each analysis form that embeds it the way the launch
dialog does; the resolution itself is guarded in ``tests/api/test_analysis_selection.py``.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt

import detail.gui.modules.analysis_location  # noqa: F401 — registers its form
import detail.gui.modules.analysis_temporal_clustering  # noqa: F401 — registers its form
from detail.gui.modules._common import PooledSpecPicker
from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, ConfigWidget


def _spec(n: int) -> dict:
    return {"module": "embedding", "identity": {"n": n}, "parent": {"root": True}}


def _group(n: int, recordings: list[str]) -> dict:
    return {
        "spec": _spec(n),
        "label": f"settings {n}",
        "recordings": recordings,
        "runs": {sid: "emb_01" for sid in recordings},
    }


#: The strain split: settings 1 covers one recording, settings 2 the other two.
SPLIT = [_group(1, ["r1"]), _group(2, ["r2", "r3"])]


def _tick(picker: PooledSpecPicker, row: int, on: bool = True) -> None:
    picker.widget.item(row).setCheckState(
        Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
    )


def test_the_widest_settings_are_ticked_first_and_emit_the_single_spec(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply(SPLIT, 3, names={})

    assert [picker.widget.item(i).text() for i in range(2)] == [
        "1/3 recordings  ·  settings 1",
        "2/3 recordings  ·  settings 2",
    ]
    # One tick is the record shape every run before multi-select wrote.
    assert picker.selection() == {"embedding_spec": _spec(2)}
    assert picker.covered() == {"r2", "r3"}
    assert picker.validation_error() is None


def test_ticking_a_second_setting_pools_every_recording(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply(SPLIT, 3, names={})
    fired = []
    picker.changed.connect(lambda _item: fired.append(True))

    _tick(picker, 0)

    assert fired, "a tick must be announced, or the dialog keeps its old narrowing"
    assert picker.selection() == {"embedding_specs": [_spec(1), _spec(2)]}
    assert picker.covered() == {"r1", "r2", "r3"}
    assert picker.validation_error() is None


def test_ticks_survive_a_refresh_that_still_offers_them(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply(SPLIT, 3, names={})
    _tick(picker, 0)

    picker.apply(
        SPLIT, 3, names={}
    )  # the launch dialog re-applies on every recording change

    assert picker.specs() == [_spec(1), _spec(2)]


def test_a_recording_in_two_ticked_settings_is_refused_in_the_form(qapp):
    """What the run refuses, said before launch rather than after it."""
    picker = PooledSpecPicker("embedding")
    picker.apply([_group(1, ["r1", "r2"]), _group(2, ["r2", "r3"])], 3, names={})
    _tick(picker, 0)
    _tick(picker, 1)

    assert picker.overlap() == ["r2"]
    assert "r2" in (picker.validation_error() or "")


def test_no_settings_is_not_narrowed_and_says_why(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply([], 2, names={})

    assert picker.covered() is None  # never the empty set: it would grey every row
    assert picker.selection() == {}
    assert "no completed embedding run" in (picker.validation_error() or "")


#: Every analysis form that embeds the picker, with its upstream. A claim, checked below
#: against the source, so a form that starts embedding it is covered the day it does.
EMBEDDERS = {
    "temporal_clustering": "embedding",
    "location_export": "tracking",
}


def test_every_embedder_is_listed():
    from pathlib import Path

    gui = Path(__file__).resolve().parents[2] / "detail" / "gui" / "modules"
    found = sorted(
        path.stem.removeprefix("analysis_")
        for path in gui.glob("analysis_*.py")
        if "PooledSpecPicker(" in path.read_text(encoding="utf-8-sig")
    )
    listed = sorted(
        {"location_export": "location"}.get(module_id, module_id)
        for module_id in EMBEDDERS
    )
    assert found == listed


@pytest.mark.parametrize("module_id", sorted(EMBEDDERS))
def test_each_form_re_narrows_and_emits_the_list_when_a_second_setting_is_ticked(
    qapp, module_id
):
    upstream = EMBEDDERS[module_id]
    groups = [{**g, "spec": {**g["spec"], "module": upstream}} for g in SPLIT]
    widget = ANALYSIS_GUI_MODULES[module_id].config_widget()
    recordings = [SimpleNamespace(id=sid) for sid in ("r1", "r2", "r3")]
    picker = next(
        value for value in vars(widget).values() if isinstance(value, PooledSpecPicker)
    )
    # Each form through its own hook: the clustering forms resolve over the selection, the
    # location form over every eligible recording (with that pool's size as the total).
    # (The launch dialog's own test: whether the form overrides the base no-op.)
    if (
        type(widget).apply_eligible_recordings
        is not ConfigWidget.apply_eligible_recordings
    ):
        widget.apply_eligible_recordings(
            recordings, {"groups": groups, "total": len(recordings)}
        )
    else:
        widget.apply_recordings(recordings, {"groups": groups})
    narrowed = []
    widget.coverage_changed.connect(lambda: narrowed.append(True))

    _tick(picker, 0)

    assert narrowed, "the launch dialog must hear that the covered recordings changed"
    assert widget.covered_recording_ids() == {"r1", "r2", "r3"}
    config = widget.config()
    assert config[f"{upstream}_specs"] == [groups[0]["spec"], groups[1]["spec"]]
    assert f"{upstream}_spec" not in config
    assert widget.validation_error() is None


# --------------------------------------------------------------------------- #
# A run chosen per recording (§6.5 per-recording overrides)
# --------------------------------------------------------------------------- #
#: The zebrafish shape: r1's tracking redone with settings 2 beside a bad run of settings 1.
REDONE = [
    {**_group(1, ["r1", "r2"]), "runs": {"r1": "emb_01", "r2": "emb_01"}},
    {**_group(2, ["r1"]), "runs": {"r1": "emb_02"}},
]


def test_a_run_chosen_per_recording_lifts_the_two_settings_refusal(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply(REDONE, 2, names={"r1": "fish 3", "r2": "fish 4"})
    _tick(picker, 1)
    assert "fish 3" in picker.validation_error()  # named, not the id

    heard = []
    picker.changed.connect(heard.append)
    picker.overrides.set_chosen({"r1": "emb_02"})

    assert heard == [None], "a form must hear a per-recording choice like a tick"
    assert picker.validation_error() is None
    assert picker.selection()["embedding_overrides"] == {"r1": "emb_02"}
    assert picker.covered() == {"r1", "r2"}
    assert "fish 3 → emb_02" in picker.overrides.note.text()


def test_a_choice_no_longer_offered_is_dropped(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply(REDONE, 2, names={})
    picker.overrides.set_chosen({"r1": "emb_02"})
    heard = []
    picker.changed.connect(heard.append)
    picker.apply(REDONE[:1], 2, names={})  # settings 2's run deleted
    assert picker.overrides.chosen() == {}
    assert heard and "embedding_overrides" not in picker.selection()


def test_the_dialog_lists_recordings_with_a_choice_first_and_returns_the_chosen(qapp):
    from detail.gui.modules._common import (
        AS_RESOLVED,
        RunOverridesDialog,
        run_options,
    )

    options = run_options(REDONE)
    dialog = RunOverridesDialog(
        "embedding", options, {"r1": "fish 3", "r2": "fish 4"}, {"r1": "emb_02"}
    )
    assert list(dialog._combos) == ["r1", "r2"]
    assert dialog.chosen() == {"r1": "emb_02"}
    dialog._combos["r1"].setCurrentIndex(0)
    assert dialog._combos["r1"].currentText() == AS_RESOLVED
    assert dialog.chosen() == {}
    dialog._combos["r2"].setCurrentIndex(1)
    assert dialog.chosen() == {"r2": "emb_01"}


@pytest.mark.parametrize("module_id", sorted(EMBEDDERS))
def test_each_form_shows_the_per_recording_row_and_emits_the_choice(qapp, module_id):
    """Every embedder puts the picker's panel (the list *and* the per-recording row) in its
    layout — a form that placed only the list would offer no way to name a run."""
    upstream = EMBEDDERS[module_id]
    groups = [{**g, "spec": {**g["spec"], "module": upstream}} for g in REDONE]
    widget = ANALYSIS_GUI_MODULES[module_id].config_widget()
    recordings = [SimpleNamespace(id=sid, name=f"fish {sid}") for sid in ("r1", "r2")]
    picker = next(
        value for value in vars(widget).values() if isinstance(value, PooledSpecPicker)
    )
    assert widget.isAncestorOf(picker.overrides.button)
    resolved = {"groups": groups, "total": 2, "names": {"r1": "fish 3"}}
    if (
        type(widget).apply_eligible_recordings
        is not ConfigWidget.apply_eligible_recordings
    ):
        widget.apply_eligible_recordings(recordings, resolved)
    else:
        widget.apply_recordings(recordings, resolved)
    narrowed = []
    widget.coverage_changed.connect(lambda: narrowed.append(True))

    picker.overrides.set_chosen({"r1": "emb_02"})

    assert narrowed
    assert widget.config()[f"{upstream}_overrides"] == {"r1": "emb_02"}
