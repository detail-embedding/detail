"""Config-form gating for the behavior-query eval (§16): the IoU threshold and the
point window are each editable only when a checked behavior is of the matching type —
a state-only eval never uses the point window; a point-only eval never uses IoU."""

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from detail.gui.modules.analysis_query_eval import _QueryEvalConfig  # noqa: E402


class _FakeRecording:
    """The minimal annotated recording the config form's resolve/apply reads.

    ``streams`` maps a behavior-stream name to its ``{behavior: type}`` vocabulary;
    the default single stream keeps the pre-selector behaviour. ``covered_behaviors``
    / ``behavior_types`` take an optional stream name, as the real Recording does —
    ``None`` meaning the only behavior stream, and **refused** when there are several,
    as the real one refuses (``AmbiguousBehaviorStreamError``). This fake once returned the
    first stream there, which is how the form's swallowed failure on a real recording with
    two streams passed every test here.
    """

    is_annotated = True
    embedding_runs: list = []
    id = "recording_000"
    name = "m1"

    def __init__(self, types: dict, streams: dict | None = None):
        self._streams = streams if streams is not None else {"behavior": dict(types)}

    def _resolve(self, name: str | None) -> dict:
        if name is None:
            if len(self._streams) > 1:
                from detail.analysis.behavior_labels import (
                    AmbiguousBehaviorStreamError,
                )

                raise AmbiguousBehaviorStreamError("several streams, none named")
            name = next(iter(self._streams), None)
        return self._streams.get(name, {})

    def streams_by_role(self, role: str) -> list:
        return [{"name": name} for name in self._streams] if role == "behavior" else []

    def covered_behaviors(self, name: str | None = None) -> list:
        return list(self._resolve(name))

    def behavior_types(self, name: str | None = None) -> dict:
        return dict(self._resolve(name))


def _config_for(types: dict) -> _QueryEvalConfig:
    cfg = _QueryEvalConfig()
    recordings = [_FakeRecording(types)]
    cfg.apply_recordings(
        recordings, cfg.resolve_recordings(recordings)
    )  # checks all behaviors
    return cfg


def _enabled(cfg: _QueryEvalConfig, name: str) -> bool:
    return cfg._fields[name][1].isEnabled()


def test_type_fields_gate_on_the_checked_behaviors(qapp):
    state_only = _config_for({"climb": "state", "groom": "state"})
    assert _enabled(state_only, "iou_threshold")
    assert not _enabled(state_only, "point_window_s")

    mixed = _config_for({"climb": "state", "bite": "point"})
    assert _enabled(mixed, "iou_threshold")
    assert _enabled(mixed, "point_window_s")

    point_only = _config_for({"bite": "point"})
    assert _enabled(point_only, "point_window_s")
    assert not _enabled(point_only, "iou_threshold")


def _config_for_streams(streams: dict) -> _QueryEvalConfig:
    cfg = _QueryEvalConfig()
    recordings = [_FakeRecording({}, streams=streams)]
    cfg.apply_recordings(recordings, cfg.resolve_recordings(recordings))
    return cfg


def test_stream_picker_lists_every_behavior_stream(qapp):
    cfg = _config_for_streams({"early": {"climb": "state"}, "full": {"groom": "state"}})
    items = [cfg._stream.itemData(i) for i in range(cfg._stream.count())]
    assert items == [None, "early", "full"]  # None = "— choose a stream —"
    assert cfg._stream.itemText(0) == "— choose a stream —"
    assert cfg._stream.currentData() is None


def test_a_single_stream_keeps_the_first_stream_default(qapp):
    """The unnamed choice stays where it means something: one stream, no ambiguity."""
    cfg = _config_for_streams({"behavior": {"climb": "state"}})
    assert cfg._stream.itemText(0) == "(first behavior stream)"
    assert cfg._checked_behaviors() == ["climb"]
    assert cfg._stream_notice.isHidden()


def test_behaviors_follow_the_selected_stream(qapp):
    """The list must be the chosen stream's vocabulary — showing the first stream's
    labels while scoring another is how a recording silently gets skipped."""
    cfg = _config_for_streams(
        {"early": {"climb": "state"}, "full": {"groom": "state", "rear": "state"}}
    )
    assert cfg._checked_behaviors() == []  # no stream chosen yet: nothing to score
    cfg._stream.setCurrentIndex(cfg._stream.findData("full"))
    assert sorted(cfg._checked_behaviors()) == ["groom", "rear"]


def test_selected_stream_reaches_the_config(qapp):
    cfg = _config_for_streams({"early": {"climb": "state"}, "full": {"groom": "state"}})
    assert cfg.config()["stream_name"] is None
    cfg._stream.setCurrentIndex(cfg._stream.findData("full"))
    assert cfg.config()["stream_name"] == "full"


def test_stream_picker_does_not_reread_recordings_on_change(qapp):
    """Switching streams must be a cached lookup: the vocabulary is resolved once,
    off the UI thread (no O(recordings) disk I/O on a combo change)."""
    calls = []

    class _Counting(_FakeRecording):
        def covered_behaviors(self, name=None):
            calls.append(name)
            return super().covered_behaviors(name)

    cfg = _QueryEvalConfig()
    recordings = [
        _Counting({}, streams={"early": {"climb": "state"}, "full": {"g": "state"}})
    ]
    cfg.apply_recordings(recordings, cfg.resolve_recordings(recordings))
    before = len(calls)
    cfg._stream.setCurrentIndex(cfg._stream.findData("full"))
    assert len(calls) == before


def test_two_ground_truth_streams_on_a_real_recording_name_the_problem(qapp, tmp_path):
    """Through the real launch dialog and a real recording, not the fake.

    A recording scored twice (two scorers, or a corrected re-import) has two ground-truth
    streams, and the backend refuses to pick one unnamed. The form used to ask it to anyway,
    off the UI thread; the failure was swallowed, and the behavior list and stream picker
    stayed empty with nothing said. Now the form fills, offers no "first stream", and says
    which recording needs a choice.
    """
    import numpy as np
    from detail.api.experiment import Experiment
    from detail.gui.launch import LaunchAnalysisDialog, _apply_to_config

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("m1")
    import cv2

    vid = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    for _ in range(10):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    recording.add_video_part(vid)
    for name, label in (("scorer_a", "groom"), ("scorer_b", "rear")):
        labels = tmp_path / f"{name}.csv"
        labels.write_text(
            f"start_frame,end_frame,label\n1,3,{label}\n", encoding="utf-8"
        )
        recording.add_stream(name, raw_path=labels, importer="behavior_csv")

    dialog = LaunchAnalysisDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("query_eval"))
    cfg = dialog._config
    # The dialog's own resolve-then-apply, on the real recording (the check box is gated on
    # an embedding run this fixture has no reason to build).
    _apply_to_config(dialog, [recording], cfg.resolve_recordings([recording]))

    items = [cfg._stream.itemData(i) for i in range(cfg._stream.count())]
    assert items == [None, "scorer_a", "scorer_b"]
    assert cfg._stream.itemText(0) == "— choose a stream —"
    assert cfg._stream_notice.text() == (
        "m1 carries more than one ground-truth label stream. Choose which one to score "
        "against."
    )
    assert not cfg._stream_notice.isHidden()
    cfg._stream.setCurrentIndex(cfg._stream.findData("scorer_b"))
    assert cfg._checked_behaviors() == ["rear"]
    assert cfg.config()["stream_name"] == "scorer_b"

    dialog.resize(720, 760)
    dialog.show()
    qapp.processEvents()
    out = (
        Path(os.environ.get("DETAIL_RENDER_DIR", tmp_path))
        / "query_eval_two_streams.png"
    )
    assert dialog.grab().save(str(out))
    dialog.close()


def test_a_failed_resolution_is_said_not_swallowed(qapp, tmp_path):
    """Whatever else goes wrong reading the selection, the dialog says so in its status line
    rather than leaving an empty form with no reason."""
    from detail.api.experiment import Experiment
    from detail.gui.launch import (
        LaunchAnalysisDialog,
        _resolve_recordings_in_background,
    )

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("m1")
    dialog = LaunchAnalysisDialog(proj)
    dialog._module_combo.setCurrentIndex(dialog._module_combo.findData("query_eval"))

    def _boom(_recordings):
        raise ValueError("stream 'x' could not be read")

    dialog._config.resolve_recordings = _boom
    dialog._resolve_gen += 1
    _resolve_recordings_in_background(dialog, [object()], dialog._resolve_gen)
    for _ in range(200):
        qapp.processEvents()
        if dialog._status.text():
            break
        import time

        time.sleep(0.01)
    assert dialog._status.text() == (
        "Could not read the selected recordings: stream 'x' could not be read"
    )
