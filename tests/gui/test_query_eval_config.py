"""Config-form gating for the behavior-query eval (§16): the IoU threshold and the
point window are each editable only when a checked behavior is of the matching type —
a state-only eval never uses the point window; a point-only eval never uses IoU."""

import pytest

pytest.importorskip("PyQt6")

from detail.gui.modules.analysis_query_eval import _QueryEvalConfig  # noqa: E402


class _FakeRecording:
    """The minimal annotated recording the config form's resolve/apply reads.

    ``streams`` maps a behavior-stream name to its ``{behavior: type}`` vocabulary;
    the default single stream keeps the pre-selector behaviour. ``covered_behaviors``
    / ``behavior_types`` take an optional stream name, as the real Recording does —
    ``None`` meaning the first behavior stream.
    """

    is_annotated = True
    embedding_runs: list = []

    def __init__(self, types: dict, streams: dict | None = None):
        self._streams = streams if streams is not None else {"behavior": dict(types)}

    def _resolve(self, name: str | None) -> dict:
        if name is None:
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
    assert items == [None, "early", "full"]  # None = "(first behavior stream)"
    assert cfg._stream.currentData() is None


def test_behaviors_follow_the_selected_stream(qapp):
    """The list must be the chosen stream's vocabulary — showing the first stream's
    labels while scoring another is how a recording silently gets skipped."""
    cfg = _config_for_streams(
        {"early": {"climb": "state"}, "full": {"groom": "state", "rear": "state"}}
    )
    assert sorted(cfg._checked_behaviors()) == ["climb"]  # default = first stream
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


def test_load_config_restores_the_track(qapp):
    cfg = _QueryEvalConfig()
    cfg.load_config({"track_id": 3})
    assert cfg._fields["track_id"][1].value() == 3
