"""The analysis upstream picker that may tick several sets of settings (§6.5).

``PooledSpecPicker`` exists because recordings whose upstream came through different chains —
two strains, each detected by its own model — share no one set of settings, so a single-choice
picker could only ever pool half of them. These drive it with synthetic groups (the
``run_spec.group_specs`` shape), and drive each analysis form that embeds it the way the launch
dialog does; the resolution itself is guarded in ``tests/api/test_analysis_selection.py``.
"""

from types import SimpleNamespace

import pytest

import detail.gui.modules.analysis_location  # noqa: F401 — registers its form
import detail.gui.modules.analysis_temporal_clustering  # noqa: F401 — registers its form
from detail.gui.modules._common import PooledSpecPicker
from detail.gui.modules.registry import ANALYSIS_GUI_MODULES
from detail.gui.qt import Qt


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
    picker.apply(SPLIT, 3)

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
    picker.apply(SPLIT, 3)
    fired = []
    picker.changed.connect(lambda _item: fired.append(True))

    _tick(picker, 0)

    assert fired, "a tick must be announced, or the dialog keeps its old narrowing"
    assert picker.selection() == {"embedding_specs": [_spec(1), _spec(2)]}
    assert picker.covered() == {"r1", "r2", "r3"}
    assert picker.validation_error() is None


def test_ticks_survive_a_refresh_that_still_offers_them(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply(SPLIT, 3)
    _tick(picker, 0)

    picker.apply(SPLIT, 3)  # the launch dialog re-applies on every recording change

    assert picker.specs() == [_spec(1), _spec(2)]


def test_a_recording_in_two_ticked_settings_is_refused_in_the_form(qapp):
    """What the run refuses, said before launch rather than after it."""
    picker = PooledSpecPicker("embedding")
    picker.apply([_group(1, ["r1", "r2"]), _group(2, ["r2", "r3"])], 3)
    _tick(picker, 0)
    _tick(picker, 1)

    assert picker.overlap() == ["r2"]
    assert "r2" in (picker.validation_error() or "")


def test_no_settings_is_not_narrowed_and_says_why(qapp):
    picker = PooledSpecPicker("embedding")
    picker.apply([], 2)

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
