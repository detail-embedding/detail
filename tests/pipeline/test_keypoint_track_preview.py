"""``keypoint_track_preview`` (§26 *Previewing a track method*): the track stage over a window.

**The class this guards: a preview that drifts from the run it previews.** A preview is only
worth its seconds if what it shows is what the run will write, and a second implementation of
"track these rows" agrees with the stage until the day one of them changes. So the preview reads
the stage's rows and runs the stage's method, and this pins it: inside the window, every setting's
points equal a real ``keypoint_track.run`` with that setting, frame for frame. The fixture's box
jitters and spikes, so the filters move points; a preview that skipped them, decoded the window
without its margin (smoothing near the cut would then differ), or ran a different path fails.

One recording of 240 frames at 30 fps, one subject whose box jitters by a pixel and jumps every 40
frames, and a real per-detection detect run (the fake method from ``test_keypoint_detect``).
"""

import cv2
import numpy as np
import pytest

from detail.analysis import keypoint_track_diagnostics as diagnostics
from detail.configs.keypoint_track import (
    PREVIEW_MAX_RECORDINGS,
    KeypointTrackPreviewWindow,
    tune_settings,
)
from detail.core.registries import KEYPOINT_MODELS
from detail.pipeline import keypoint_track
from detail.pipeline import keypoint_track_preview as preview
from detail.storage import recording as recording_store
from detail.storage.keypoint_runs import KeypointDetectRun, KeypointTrackRun
from tests.pipeline.test_keypoint_detect import _FakeKeypointModel, _config, _run

N_FRAMES = 240
FRAME_NS = 33_333_333  # 30 fps
#: The window every test previews: 3 s in, 2 s long, with the margin inside the video.
WINDOW = {"starts": {"recording_002": 3.0}, "duration_s": 2.0}
VARIANT = {"transition_sigma_body": 3.0, "smooth": False}


def _box(frame: int) -> tuple[int, int]:
    """The box's top-left pixel: a one-pixel jitter, and a jump every 40 frames."""
    col = 4 + frame % 3 + (8 if frame % 40 == 20 else 0)
    row = 4 + (frame // 5) % 2
    return row, col


def _moving_recording(it, recording_id: str):
    sq = it.square
    detections = []
    for f in range(N_FRAMES):
        r, c = _box(f)
        bbox = (c / 40, r / 40, (c + 16) / 40, (r + 16) / 40)
        detections.append((f, 0, bbox, sq(r, r + 16, c, c + 16)))
    recording = it.add_recording(
        recording_id,
        detections,
        [0] * N_FRAMES,
        frame_ns=FRAME_NS,
        n_timestamps=N_FRAMES,
    )
    video = recording.dir / "video" / "part_001.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (40, 40)
    )
    for i in range(N_FRAMES):
        writer.write(np.full((40, 40, 3), i % 250, np.uint8))
    writer.release()
    _run(it, recording, _config())
    data = recording_store.read_recording(recording.dir)
    data["keypoint_detect_runs"] = [
        {"id": "kd_01", "status": "complete", "model_id": "fake_kp", "output": "x"}
    ]
    recording_store.write_recording(recording.dir, data)
    return recording


@pytest.fixture
def moving(processed_recording, monkeypatch):
    monkeypatch.setitem(KEYPOINT_MODELS, "fake_kp", _FakeKeypointModel)
    it = processed_recording
    return it, _moving_recording(it, "recording_002")


def _launch(**config) -> dict:
    """The track form's launch config: temporal_decode on kd_01 over track_01."""
    return {
        "detect_run_id": "kd_01",
        "tracking_run_id": "track_01",
        "method": "temporal_decode",
        "config": {},
        **config,
    }


def _stage(it, recording, method_config: dict, run_id: str) -> dict:
    """A real track run with ``method_config``; its subject 0 rows on the part."""
    module = keypoint_track.KeypointTrackModule()
    data = recording_store.read_recording(recording.dir)
    config = module.validate(recording.dir, data, _launch(config=method_config))
    job = module.build_job(recording.id, run_id, config)
    run_dir = module.run_dir_for(it.experiment.dir, job)
    run_dir.mkdir(parents=True, exist_ok=True)
    keypoint_track.run(job, run_dir, lambda _e: None, None)
    return KeypointTrackRun(recording.dir, run_id).rows(0, "part_001")


def test_inside_the_window_every_setting_is_the_stage_run(moving):
    it, recording = moving
    result = it.experiment.preview_keypoint_tracking(
        _launch(), ["recording_002"], variants=[VARIANT], **WINDOW
    )
    assert [s.name for s in result.settings] == [
        preview.DETECTION_ONLY,
        preview.CURRENT,
        "transition 3, smooth off",
    ]
    window = result.windows[0]
    assert (window.start_ns, window.end_ns) == (3 * 10**9, 5 * 10**9)
    for setting, config, run_id in (
        (result.settings[1], {}, "kt_01"),
        (result.settings[2], VARIANT, "kt_02"),
    ):
        ran = _stage(it, recording, config, run_id)
        shown = setting.tracks["recording_002"][0]
        assert len(shown.frame_idx) == 60  # 2 s at 30 fps
        assert np.all((shown.t_ns >= window.start_ns) & (shown.t_ns < window.end_ns))
        at = np.searchsorted(ran["frame_idx"], shown.frame_idx)
        np.testing.assert_array_equal(ran["frame_idx"][at], shown.frame_idx)
        np.testing.assert_allclose(shown.points, ran["points"][at], atol=1e-6)
    # Detection alone is the detect run's own placement, untouched.
    detected = KeypointDetectRun(recording.dir, "kd_01").rows("part_001")
    shown = result.settings[0].tracks["recording_002"][0]
    np.testing.assert_allclose(shown.points, detected["points"][shown.frame_idx])
    # The setting the stage ran is not detection: the filters moved points in the window.
    assert not np.allclose(
        result.settings[1].tracks["recording_002"][0].points, shown.points
    )


def test_every_setting_is_described_with_detection_alone_as_the_first_row(moving):
    it, _recording = moving
    result = it.experiment.preview_keypoint_tracking(
        _launch(), ["recording_002"], variants=[VARIANT], **WINDOW
    )
    detection = result.settings[0]
    assert detection.overrides is None
    for setting in result.settings:
        assert set(setting.figures) == {"recording_002", preview.ALL}
        assert setting.figures[preview.ALL] == setting.figures["recording_002"]
    # The box jumps 8 px on frames 100 and 140 of the window's 90..150. Detection places every
    # point; the default transition scale (0.15 of a 16 px body) calls both points missing on
    # both jump rows rather than follow them; transition 3 follows them.
    placed = [s.figures[preview.ALL]["placed"] for s in result.settings]
    assert placed == pytest.approx([1.0, 1 - 4 / 120, 1.0])
    assert detection.figures[preview.ALL]["jumps_per_min"] > 0
    assert result.settings[1].figures[preview.ALL]["median_move"] <= (
        detection.figures[preview.ALL]["median_move"]
    )
    assert result.tracked == ("nose", "tail")
    assert result.config_of("transition 3, smooth off")["smooth"] is False


def test_progress_assigns_then_tracks_then_measures_and_moves_within_one_subject(
    moving, monkeypatch
):
    """One recording, one subject: what moves the bar is inside the work, not the loop."""
    monkeypatch.setattr("detail.core.progress.PROGRESS_INTERVAL_S", 0.0)
    it, _recording = moving
    events = []
    it.experiment.preview_keypoint_tracking(
        _launch(),
        ["recording_002"],
        variants=[VARIANT],
        progress=lambda *e: events.append(e),
        **WINDOW,
    )
    stages = [e[0] for e in events]
    order = [s for i, s in enumerate(stages) if i == 0 or stages[i - 1] != s]
    assert order == ["assigning", "tracking", "measuring"]
    assigning = [f for s, f, _d in events if s == "assigning"]
    assert any(0 < f < 1 for f in assigning)
    assert assigning == sorted(assigning) and assigning[-1] == 1.0
    tracking = [f for s, f, _d in events if s == "tracking"]
    assert any(0 < f < 1 for f in tracking) and tracking[-1] == 1.0
    assert events[-1][:2] == ("measuring", 1.0)


def test_the_window_defaults_to_the_middle_and_stops_at_its_part(moving):
    it, recording = moving
    data = recording_store.read_recording(recording.dir)
    last_ns = (N_FRAMES - 1) * FRAME_NS
    middle = preview.resolve_window(
        recording.dir, data, KeypointTrackPreviewWindow(duration_s=2.0)
    )
    assert middle[:2] == (0, "part_001")
    assert middle[2] == (last_ns - 2 * 10**9) // 2
    late = preview.resolve_window(
        recording.dir, data, KeypointTrackPreviewWindow(start_s=7.0, duration_s=30.0)
    )
    assert (late[2], late[3]) == (7 * 10**9, last_ns + 1)
    with pytest.raises(ValueError, match="past the end of recording"):
        preview.resolve_window(
            recording.dir, data, KeypointTrackPreviewWindow(start_s=60.0)
        )


@pytest.mark.parametrize(
    "recordings, variants, fragment",
    [
        (["recording_002"] * (PREVIEW_MAX_RECORDINGS + 1), [], "at most 3 recordings"),
        (["recording_002"], [{"smooth": False}] * 3, "at most 2 settings"),
        (["recording_002"], [{"despike_windows": 3}], "no setting despike_windows"),
        (["recording_002"], [{"transition_sigma_body": -1.0}], "greater than 0"),
    ],
)
def test_a_preview_names_what_it_refuses(moving, recordings, variants, fragment):
    it, _recording = moving
    with pytest.raises(ValueError, match=fragment):
        it.experiment.preview_keypoint_tracking(
            _launch(), recordings, variants=variants, **WINDOW
        )


def test_a_launch_the_stage_would_refuse_is_refused_by_the_preview(moving):
    it, _recording = moving
    with pytest.raises(ValueError, match="not found in recording"):
        it.experiment.preview_keypoint_tracking(
            _launch(detect_run_id="kd_09"), ["recording_002"], **WINDOW
        )


def test_without_recordings_it_takes_those_the_config_covers(moving):
    it, _recording = moving
    result = it.experiment.preview_keypoint_tracking(_launch(), duration_s=2.0)
    assert [w.recording_id for w in result.windows] == ["recording_002"]


def test_a_grid_search_is_a_preview_of_every_tune_setting(moving):
    """The grid search shares one assignment and one decode per transition scale across 36
    settings; each must still be what a preview with that setting as its *current* settings
    shows — a preview that decodes one path, so nothing is shared, and whose current column
    is the stage run (the test above). A path cached under the wrong key or a filter leaking
    between settings fails here."""
    it, _recording = moving
    grid = it.experiment.preview_keypoint_tracking(
        _launch(), ["recording_002"], grid=True, **WINDOW
    )
    assert grid.grid
    assert [s.name for s in grid.settings] == [
        preview.DETECTION_ONLY,
        preview.CURRENT,
    ] + [preview.variant_name(o) for o in tune_settings()]
    for setting in grid.settings[2:]:
        alone = it.experiment.preview_keypoint_tracking(
            _launch(config=setting.overrides), ["recording_002"], **WINDOW
        ).setting(preview.CURRENT)
        np.testing.assert_array_equal(
            setting.tracks["recording_002"][0].points,
            alone.tracks["recording_002"][0].points,
            err_msg=setting.name,
        )
        assert setting.figures == alone.figures, setting.name


def test_the_front_keeps_a_trade_and_drops_what_is_beaten(moving):
    it, _recording = moving
    result = it.experiment.preview_keypoint_tracking(
        _launch(), ["recording_002"], grid=True, **WINDOW
    )
    figures = {s.name: s.figures[preview.ALL] for s in result.settings[1:]}
    assert result.front == diagnostics.pareto_front(figures)
    assert preview.DETECTION_ONLY not in result.front  # the floor, never a candidate
    # The default transition scale calls the box's two jumps missing and gains nothing for
    # it; smoothing without despike smears each jump into its neighbours, off the confident
    # detections. Both are beaten. Following the jumps (filters off) trades jumps for
    # agreement, and despiking trades the other way: both stay.
    for name in figures:
        if name.startswith("transition 0.15") or (
            "smooth on" in name and "despike off" in name
        ):
            assert name not in result.front, name
    assert "transition 3, smooth off, despike off" in result.front
    assert "transition 3, smooth off, despike on" in result.front
    ranked = result.ranked()
    assert [s.name for s in ranked[:2]] == [preview.DETECTION_ONLY, preview.CURRENT]
    on = [s.name for s in ranked[2:] if s.name in result.front]
    assert ranked[2 : 2 + len(on)] == [result.setting(n) for n in on]
    agreement = [result.setting(n).figures[preview.ALL]["agreement"] for n in on]
    assert agreement == sorted(agreement, reverse=True)


@pytest.mark.parametrize(
    "launch, variants, fragment",
    [
        (_launch(), [{"smooth": False}], "pass no variants"),
        (
            _launch(method="despike_smooth"),
            None,
            "grid search covers temporal_decode settings",
        ),
    ],
)
def test_a_grid_search_names_what_it_refuses(moving, launch, variants, fragment):
    it, _recording = moving
    with pytest.raises(ValueError, match=fragment):
        it.experiment.preview_keypoint_tracking(
            launch, ["recording_002"], variants=variants, grid=True, **WINDOW
        )
