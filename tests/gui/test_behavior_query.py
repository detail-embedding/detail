"""Shared clip surfaces (detail/gui/behavior_query.py): the example-clip picker and the
clip player most clip views embed. Built against a real recording (video + detection +
tracking + embedding).

The scan form re-picks the seed's embed run while the recordings offer it
(``prefer_selection``, outranked by a pick by hand), so the clips a scan restores come back with
the run they were marked on.
"""

import json

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.dataclasses import TrackingJob  # noqa: E402
from detail.gui import frame_source  # noqa: E402
from detail.gui.behavior_query import (  # noqa: E402
    _CoverageBar,
    ClipExampleDialog,
    ClipSpec,
    _ClipPlayer,
)
from detail.storage import recording as recording_store  # noqa: E402
from detail.storage.zarr_io import (  # noqa: E402
    DetectionPartWriter,
    EmbeddingPartWriter,
)
from detail.trackers.anchor_tracker import AnchorTracker  # noqa: E402
from tests import waits  # noqa: E402

_FRAMES = (0, 2, 4)  # frames with a detection (the 6-frame video)


@pytest.fixture(autouse=True)
def _flush_decodes(qapp):
    """Let the scrubber's background decode finish before teardown deletes its signals."""
    yield
    from PyQt6.QtCore import QThreadPool

    QThreadPool.globalInstance().waitForDone(2000)
    qapp.processEvents()


def _tiny_video(path, n=6, w=32, h=24):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 30) % 255, np.uint8))
    vw.release()
    return path


def _append_run(recording_dir, key, entry):
    data = recording_store.read_recording(recording_dir)
    data.setdefault(key, []).append(entry)
    recording_store.write_recording(recording_dir, data)


def _embed_video_experiment(tmp_path):
    """A experiment with one recording carrying a video + det_01 → track_01 → emb_01."""
    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    sess.add_video_part(_tiny_video(tmp_path / "clip.mp4"))
    sdir = sess.dir

    det_dir = sdir / "detections" / "det_01"
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        for f in _FRAMES:
            w.add_row(f, f * 1000, 0, "mouse", 0.9, (0.1, 0.1, 0.4, 0.4), "")
    # The visible trackers read the mask grid from the detection run's envelope.
    (det_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    _append_run(
        sdir,
        "detection_runs",
        {"id": "det_01", "status": "complete", "downsampling_run_id": None},
    )
    AnchorTracker().track(
        TrackingJob(
            job_id="track_01",
            recording_id="recording_001",
            track_id="track_01",
            detection_run_id="det_01",
            tracker_id="anchor_tracker",
            hyperparams={},
            arena_ids=[],
            created_at="2026-01-01T00:00:00Z",
        ),
        det_dir,
        sdir / "tracking" / "track_01",
    )
    _append_run(
        sdir,
        "tracking_runs",
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"},
    )

    emb_dir = sdir / "embeddings" / "emb_01"
    rng = np.random.default_rng(0)
    with EmbeddingPartWriter(emb_dir / "track_0" / "part_001.zarr", output_dim=8) as w:
        for di, f in enumerate(_FRAMES):
            w.add_row(f, f * 1000, di, rng.standard_normal(8).astype(np.float32))
    (emb_dir / "embedding.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "emb_01",
                "tracking_run_id": "track_01",
                "model_id": "dinov3_convnext_base",
                "model_variant": None,
                "constructor": "full_frame",
                "output_dim": 8,
                "track_ids": [0],
                "renderer_spec": [{"type": "init_binary_mask"}],
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _append_run(
        sdir,
        "embedding_runs",
        {"id": "emb_01", "status": "complete", "tracking_run_id": "track_01"},
    )
    return proj


def _add_second_embedding_run(proj, embed_run_id="emb_02"):
    """A second embedding run on ``recording_001``, over the same ``track_01`` tracking run
    ``_embed_video_experiment`` already built."""
    sess = proj.recordings[0]
    sdir = sess.dir
    emb_dir = sdir / "embeddings" / embed_run_id
    rng = np.random.default_rng(1)
    with EmbeddingPartWriter(emb_dir / "track_0" / "part_001.zarr", output_dim=8) as w:
        for di, f in enumerate(_FRAMES):
            w.add_row(f, f * 1000, di, rng.standard_normal(8).astype(np.float32))
    (emb_dir / "embedding.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": embed_run_id,
                "tracking_run_id": "track_01",
                "model_id": "dinov3_convnext_base",
                "model_variant": None,
                "constructor": "full_frame",
                "output_dim": 8,
                "track_ids": [0],
                "renderer_spec": [{"type": "init_binary_mask"}],
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _append_run(
        sdir,
        "embedding_runs",
        {"id": embed_run_id, "status": "complete", "tracking_run_id": "track_01"},
    )


def _wait_continuity(qapp, dialog, timeout_ms=3000):
    """Block until the dialog's off-thread continuity read lands (marking is greyed out
    until it does, so a test that marks straight away would race it)."""
    from PyQt6.QtTest import QTest

    waited = 0
    while dialog._continuity is None and waited < timeout_ms:
        qapp.processEvents()
        QTest.qWait(10)
        waited += 10
    assert dialog._continuity is not None, "continuity never loaded"
    return dialog


def _clip_note(dialog, row: int) -> str:
    """The Note cell of the clip table's ``row`` — where a pre-loaded clip's verdict shows."""
    return dialog._clip_model.item(row, 5).text()


def test_example_dialog_marks_clips(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    assert not dialog._mark_start.isEnabled()  # greyed until continuity is known
    _wait_continuity(qapp, dialog)
    assert dialog._mark_start.isEnabled() and dialog._mark_end.isEnabled()
    assert dialog._preview.recording_frame_count("recording_001", "emb_01") == 6
    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    dialog._scrubber.set_frame(4, decode=False)
    dialog._mark(start=False)
    assert len(dialog.examples) == 1
    ex = dialog.examples[0]
    assert ex["recording_id"] == "recording_001" and ex["start_t_ns"] <= ex["end_t_ns"]


def _lit(button) -> bool:
    """Whether a button is filled with the status green — what it *shows*, not a flag."""
    from PyQt6.QtGui import QPalette

    from detail.gui.status_color import READY

    return (
        button.autoFillBackground()
        and button.palette().color(QPalette.ColorRole.Button) == READY
    )


def test_mark_start_shows_it_is_on_until_the_clip_is_finished(qapp, tmp_path):
    """The 10/4 ask: a pending start turns the button green and ticks the timeline.

    Before this, only a status line above the video said a start was waiting — somewhere other
    than where the next press goes, which is how "Mark start isn't working" came up mid-demo.
    """
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    assert not _lit(dialog._mark_start)
    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    assert _lit(dialog._mark_start), "a pending start left the button looking idle"
    assert dialog._scrubber.marks() == ((0,), ()), "no tick where the start was marked"

    dialog._scrubber.set_frame(4, decode=False)
    dialog._mark(start=False)
    assert len(dialog.examples) == 1
    assert not _lit(dialog._mark_start)
    # The tick goes; the finished clip stays on the timeline as a span (#628).
    ticks, _spans = dialog._scrubber.marks()
    assert ticks == () and [band[:2] for band in dialog._scrubber.bands()] == [(0, 4)]


def test_a_reload_drops_a_pending_start_visibly_and_says_so(qapp, tmp_path):
    """A start that silently vanished when the track spinner moved was one of 10/4's silent states."""
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    dialog._scrubber.set_frame(2, decode=False)
    dialog._mark(start=True)
    assert dialog.pending_start() is not None

    dialog._reload()

    assert dialog.pending_start() is None
    assert not _lit(dialog._mark_start)
    assert dialog._scrubber.marks() == ((), ())
    assert "dropped" in dialog._span_status.text()


def test_a_reload_with_nothing_pending_says_nothing_about_a_start(qapp, tmp_path):
    """Guard the guard: the message is about a start that existed, not a reload as such."""
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    dialog._reload()
    assert "dropped" not in dialog._span_status.text()


def test_cancelling_clears_a_pending_start(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    dialog._scrubber.set_frame(2, decode=False)
    dialog._mark(start=True)
    dialog.reject()
    assert dialog.pending_start() is None and not _lit(dialog._mark_start)


def test_the_example_dialog_shows_time_from_the_player_not_its_own_label(
    qapp, tmp_path
):
    """Its private readout sat mid-row; the player owns the time now, beside the frame number."""
    from PyQt6.QtWidgets import QLabel

    from detail.gui.timefmt import format_time

    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    dialog._scrubber.set_frame(3, decode=False)
    expected = format_time(dialog._preview.frame_t_ns("recording_001", "emb_01", 3))
    assert dialog._scrubber.time_text() == expected
    times = [
        w
        for w in dialog.findChildren(QLabel)
        if w.text().startswith(expected) and w not in dialog._scrubber.time_labels()
    ]
    assert not times, "a second time readout is still in the dialog"


def _one_clip_of(track_id) -> list[dict]:
    clip = {
        "recording_id": "recording_001",
        "track_id": track_id,
        "start_t_ns": 0,
        "end_t_ns": 4000,
    }
    return [{"name": "close contact", "examples": [clip], "negatives": []}]


def test_start_from_a_scan_picks_the_scans_embed_run(qapp, tmp_path):
    """The clips a scan restores are bound to its embed run, so the run comes back with them
    whenever the recordings offer it — a pick by hand still wins.
    """
    from detail.gui.launch import LaunchAnalysisDialog

    proj = _embed_video_experiment(tmp_path)
    _add_second_embedding_run(proj)
    seed = {
        "id": "an_01",
        "module_id": "query",
        "recording_ids": ["recording_001"],
        "params": {
            "embed_run_id": "emb_02",
            "track_id": 0,
            "method": "soft_knn",
            "behaviors": [{"id": "q01", **_one_clip_of(0)[0]}],
        },
    }
    dialog = LaunchAnalysisDialog(proj, seed=seed)
    dialog._select_all_recordings()
    embed = dialog._config._embed
    assert [embed.itemData(i) for i in range(embed.count())] == ["emb_01", "emb_02"]
    assert embed.currentData() == "emb_02"

    embed.setCurrentIndex(0)
    embed.activated.emit(0)  # the user picks the other run by hand
    dialog._select_all_recordings()
    assert embed.currentData() == "emb_01", "a pick by hand is outranked"


def test_edit_clips_frame_is_full_resolution_and_zooms(qapp, tmp_path):
    """Edit clips is where a clip's edge is found, so its frame zooms like the player's.

    It used to hand the pane a copy already scaled to 400x300 — zooming that would only have
    magnified the shrunk copy — so the view must hold the frame at its own size.
    """
    from detail.gui.image_view import ImageView

    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    assert isinstance(dialog._src, ImageView) and dialog._src.zoomable
    assert dialog._block.picture is dialog._src
    assert dialog._block.controls is dialog._scrubber
    assert _drain(qapp, lambda: dialog._src.pixmap_size() is not None)
    size = dialog._src.pixmap_size()
    assert (size.width(), size.height()) == (32, 24), "the pane holds a pre-scaled copy"


def test_example_dialog_reports_coverage_and_brackets_the_pending_run(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    # 3 embedded frames (0, 2, 4) out of a 6-frame video, all one continuous stretch.
    assert dialog._continuity["n_rows"] == 3
    assert dialog._continuity["runs"] == [(0, 4)]
    assert (
        "50.0%" in dialog._status.text()
        and "1 continuous stretches" in dialog._status.text()
    )

    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    # A pending start outlines the stretch it landed in — how far the end may be dragged.
    assert dialog._coverage._bracket == (0, 4)
    assert "1 embedded frames" in dialog._span_status.text()  # start == end so far
    dialog._scrubber.set_frame(4, decode=False)  # the span grows as you scrub
    assert "3 embedded frames" in dialog._span_status.text()
    dialog._mark(start=False)
    assert dialog._coverage._bracket is None  # cleared once the clip is added


def test_example_dialog_refuses_a_clip_that_crosses_a_gap(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    # Embedded frames are 2000 ns apart, so a 1000 ns split gap makes every step a break —
    # the fragmented-track case, without needing an hour of video to reproduce it.
    dialog = _wait_continuity(
        qapp,
        ClipExampleDialog(proj, proj.recordings, "emb_01", max_gap_ns=1000),
    )
    assert (
        len(dialog._continuity["runs"]) == 3
    )  # every embedded frame is its own stretch

    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    dialog._scrubber.set_frame(4, decode=False)
    dialog._mark(start=False)
    assert dialog.examples == []  # refused at the point of marking, not at search time
    assert "cannot use" in dialog._span_status.text()
    assert "breaks in tracking" in dialog._span_status.text()

    # A clip inside one stretch is still accepted.
    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    dialog._mark(start=False)
    assert len(dialog.examples) == 1


def test_example_dialog_flags_preloaded_clips_that_no_longer_resolve(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    stale = [
        {
            "recording_id": "recording_001",
            "track_id": 0,
            "start_t_ns": 0,
            "end_t_ns": 4000,
        }
    ]
    dialog = _wait_continuity(
        qapp,
        ClipExampleDialog(
            proj, proj.recordings, "emb_01", initial=stale, max_gap_ns=1000
        ),
    )
    from PyQt6.QtTest import QTest

    for _ in range(100):  # the initial check is its own background call
        qapp.processEvents()
        QTest.qWait(10)
        if "⚠" in _clip_note(dialog, 0):
            break
    assert "⚠" in _clip_note(dialog, 0)
    assert "breaks in tracking" in _clip_note(dialog, 0)


def test_example_dialog_frame_view_no_detection(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    rgb, silhouette, _actual, _t_ns = dialog._preview.frame_view(
        "recording_001", "emb_01", 0, 1  # frame 1 has no detection
    )
    assert rgb is not None and silhouette is None


def _write_clip_run(proj, run_id="an_01"):
    from detail.storage import analysis_runs as ar
    from detail.storage.clip_query_csv import ClipRow, write_clip_csv

    adir = ar.analysis_dir(proj.dir, run_id)
    adir.mkdir(parents=True, exist_ok=True)
    write_clip_csv(
        adir / "matches.csv",
        [
            ClipRow(
                "query",
                0,
                "recording_001",
                0,
                "emb_01",
                0,
                100_000_000,
                0,
                3,
                0.0,
                query_id="q01",
                query_name="rear",
            ),
            ClipRow(
                "match",
                1,
                "recording_001",
                0,
                "emb_01",
                0,
                100_000_000,
                0,
                3,
                0.05,
                query_id="q01",
                query_name="rear",
            ),
        ],
    )
    ar.write_analysis_yaml(
        adir,
        {
            "id": run_id,
            "module_id": "clip_query",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "embed_run_id": "emb_01",
            "track_id": 0,
            "params": {"fusion": "min", "band_ns": 500_000_000, "threshold": 0.1},
            "queries": [
                {
                    "id": "q01",
                    "name": "rear",
                    "fusion": "mean",
                    "band_ns": 800_000_000,
                    "threshold": 0.14,
                    "examples": [
                        {
                            "recording_id": "recording_001",
                            "track_id": 0,
                            "start_t_ns": 0,
                            "end_t_ns": 2000,
                        }
                    ],
                }
            ],
            "artifacts": [
                {"path": "matches.csv", "kind": "clip_matches", "label": "m"}
            ],
        },
    )
    return ar.read_analysis_yaml(adir)


def test_a_removed_modules_run_can_be_read_but_not_relaunched(qapp, tmp_path):
    """Reading a finished run survives the module's removal; starting another one does not.

    Written when `clip_query` was hidden (§25); it was then removed, and the same two halves
    hold for a run it left on disk.

    This used to be asserted on the Refine button, which was gated on `is_surfaced` for exactly
    this reason. The button is gone — the Start-from picker replaced it — so the rule now rests
    on the launcher, which enumerates only surfaced modules. Worth keeping pointed at the thing
    that enforces it rather than deleting with the button: the failure it prevents is a form for
    one module opened carrying another module's config.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.launch import LaunchAnalysisDialog
    from detail.gui.pages import AnalysisRunDetailPage

    proj = _embed_video_experiment(tmp_path)
    _write_clip_run(proj)

    page = AnalysisRunDetailPage(proj, "an_01")
    assert [
        b for b in page.findChildren(QPushButton) if b.text() == "Open output folder"
    ], "the run itself still renders — the half the toggle leaves alone"

    dialog = LaunchAnalysisDialog(proj)
    assert dialog._module_combo.findData("clip_query") < 0


def test_clip_player_one_frame_play_snaps_back(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 0, "q"))  # a 1-frame clip
    assert player._len == 1
    player.play()  # pressing Play on a 1-frame clip
    assert not player.is_playing()  # snaps back, doesn't stick on "Pause"
    assert not player.play_button().isChecked()


def test_clip_player_set_clip_and_compute(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 100_000_000, "q"))
    assert player._len >= 1
    lo, hi = player.bounds()
    assert lo <= hi
    payload = player._compute(0)
    view, reason = payload
    assert reason is None  # the fixture's video is present and decodes
    # `frame_masked` returns (rgb, actual, t_ns) — one picture, the footage with the subject
    # marked. The old 4-tuple carried a second pane holding the *input image*, which answers a
    # question about the method rather than about the animal.
    assert len(view) == 3
    player._display(0, payload)  # renders source + silhouette without raising
    player._scrubber._has_rendered = (
        True  # a frame is up, so play runs rather than arming
    )
    player.play()
    assert player.is_playing() == (player._len > 1)
    player.pause()
    assert not player.is_playing()


def test_clip_player_play_stops_at_end_no_loop(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 200_000_000, "q"))
    if player._len <= 1:
        return  # a 1-frame clip has nothing to play
    player._scrubber._has_rendered = True
    player.play()  # start playing (as a click would)
    for _ in range(player._len + 3):  # drive past the end
        player._scrubber._advance()
    assert not player.is_playing()  # stopped at the end
    assert (
        player._scrubber.frame() == player._len - 1
    )  # rests on the last frame, no wrap


def _group(name, examples, matches):
    return {"name": name, "examples": examples, "matches": matches}


class _FakeCompare:
    """Records the grouped queries a launch handler would open the compare dialog with."""

    seen: dict = {}

    def __init__(self, experiment, queries, parent=None) -> None:
        _FakeCompare.seen = {"queries": queries}

    def exec(self) -> int:
        return 0


def _drop_the_video_blob(proj):
    """Make the recording a video-excluded export: the part entry and its timestamps
    stay, the ``.mp4`` goes. This is the shape an exported bundle arrives in when the
    pixels were left on the source machine — every zarr the picker reads is intact."""
    recording = proj.recordings[0]
    data = recording_store.read_recording(recording.dir)
    for part in data["video"]["files"]:
        recording_store.resolve_part_video_path(recording.dir, part).unlink()
    assert not recording.video_available
    return proj


def test_example_dialog_refuses_to_mark_without_the_video(qapp, tmp_path):
    """The reported bug: the picker gated marking on the *embeddings* (read from zarr,
    which a video-excluded export still carries in full), so marking opened over a pane
    that never rendered and clips were committed to frames nobody saw."""
    proj = _drop_the_video_blob(_embed_video_experiment(tmp_path))
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    _wait_continuity(qapp, dialog)

    assert not dialog._mark_start.isEnabled()
    assert not dialog._mark_end.isEnabled()
    assert "isn't on this machine" in dialog._status.text()

    # Refused in the model too, not only by the greyed button.
    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    dialog._scrubber.set_frame(4, decode=False)
    dialog._mark(start=False)
    assert dialog.examples == []
    assert "cannot mark" in dialog._span_status.text()


def test_example_dialog_names_the_missing_video_instead_of_a_blank_pane(qapp, tmp_path):
    proj = _drop_the_video_blob(_embed_video_experiment(tmp_path))
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")

    _view, reason = dialog._compute(0)
    assert reason == "missing"
    dialog._display(0, (None, reason))
    assert dialog._src.text() == "(source video not in this export)"


def test_clip_player_names_the_missing_video_instead_of_a_blank_pane(qapp, tmp_path):
    proj = _drop_the_video_blob(_embed_video_experiment(tmp_path))
    player = _ClipPlayer(proj, context_s=0)
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 100_000_000, "q"))

    payload = player._compute(0)
    assert payload[1] == "missing"
    player._display(0, payload)
    assert player._src.text() == "(source video not in this export)"


def test_example_dialog_still_marks_when_the_video_is_present(qapp, tmp_path):
    """Guard the guard above: the refusal must be about the missing pixels, not a
    picker that stopped accepting clips altogether."""
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    _wait_continuity(qapp, dialog)
    assert dialog._mark_start.isEnabled()
    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    dialog._scrubber.set_frame(4, decode=False)
    dialog._mark(start=False)
    assert len(dialog.examples) == 1


class _LongRecording:
    """Stands in for a long real part, as long as a typical recording.


    The video fixtures here are six frames long, where a whole-recording trim strip and a
    windowed one are indistinguishable — which is exactly why the scale bug survived the
    suite and only showed up against real data.
    """

    PERIOD_NS = 16_000_000  # 62.5 fps
    N_FRAMES = 108_064

    def recording_frame_count(self, _recording_id, _embed_run_id):
        return self.N_FRAMES

    def t_ns_to_frame(self, _recording_id, _embed_run_id, t_ns):
        return int(min(max(round(t_ns / self.PERIOD_NS), 0), self.N_FRAMES - 1))

    def frame_t_ns(self, _recording_id, _embed_run_id, frame):
        return int(frame) * self.PERIOD_NS


def _long_clip(player, start_s=223.4, length_s=0.85):
    player._preview = _LongRecording()
    clip = ClipSpec(
        "recording_001",
        0,
        "emb_01",
        int(start_s * 1e9),
        int((start_s + length_s) * 1e9),
        "match",
    )
    player.set_clip(clip)
    return clip


def test_trim_window_is_context_around_the_clip_not_the_whole_recording(qapp, tmp_path):
    """A 0.85 s clip on a 29-minute part: spanning the whole timeline gives ~2.8 s per
    pixel, so the clip is a third of a pixel wide and its edges cannot be grabbed."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)

    first, last = player.window()
    window_s = (last - first) * _LongRecording.PERIOD_NS / 1e9
    assert window_s < 20, f"window is {window_s:.0f}s — the drag scale collapses"
    clip_lo, clip_hi = player.clip_span()
    assert first < clip_lo and last > clip_hi  # room on both sides to extend outward


def test_panning_slides_the_window_without_changing_its_scale(qapp, tmp_path):
    """Growing the window on every edge hit made the scale coarser each time — a few steps
    and a clip is too small to work with, with no way back. Sliding keeps the
    seconds-per-pixel the annotator started with, however far they scrub."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    before = player.window()
    size = before[1] - before[0]

    assert player.pan(forward=True) is True
    after = player.window()
    assert after[1] - after[0] == size, "the window changed size instead of sliding"
    assert after[0] > before[0] and after[1] > before[1]

    assert player.pan(forward=False) is True
    assert player.window() == before  # and back again, same size


def test_panning_repeatedly_never_degrades_the_scale(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    size = player.window()[1] - player.window()[0]

    for _ in range(12):
        player.pan(forward=True)

    assert player.window()[1] - player.window()[0] == size


def test_back_to_clip_returns_the_window_to_the_match(qapp, tmp_path):
    """Panning can take the strip a long way from the clip; there has to be one step back
    rather than a long scrub."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    home = player.window()
    for _ in range(6):
        player.pan(forward=True)
    assert player.window() != home

    assert player.recenter() is True
    assert player.window() == home
    lo, hi = player.clip_span()
    assert home[0] <= lo and hi <= home[1]  # the clip is visible again


def test_a_non_trim_player_still_spans_only_its_clip(qapp, tmp_path):
    """The compare view and the example player must be untouched by trim mode."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    _long_clip(player)
    assert player.window() == player.clip_span()


def test_trim_scrub_positions_map_back_through_the_window_offset(qapp, tmp_path):
    """A dragged edge is a scrubber position, not a recording frame — forgetting the
    window's offset would write a span from the wrong part of the video."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)

    first, _last = player.window()
    assert player.frame_to_t_ns(0) == first * _LongRecording.PERIOD_NS
    assert player.frame_to_t_ns(10) == (first + 10) * _LongRecording.PERIOD_NS


def test_coverage_strip_is_solid_where_tracking_is_continuous_when_zoomed_in(
    qapp, tmp_path
):
    """Zoomed past one frame per pixel (the trim window), a histogram puts 0 or 1 frame in
    each bin, so a fully-tracked stretch renders as a barcode of alternating dark and blank
    columns — which reads as a run of tracking gaps, the one thing this strip exists to
    show."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    bar = _CoverageBar(player.scrubber())
    frames = np.arange(400, dtype=np.int32)
    bar.set_continuity(frames, [(0, 399)], 400)

    density = bar._density(840)  # more pixels than frames

    assert density.min() == 1.0, "continuous tracking rendered with blank columns"


def test_coverage_strip_still_shows_a_real_gap_when_zoomed_in(qapp, tmp_path):
    """Guard the guard: solid everywhere would hide the gaps too."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    bar = _CoverageBar(player.scrubber())
    frames = np.concatenate([np.arange(0, 100), np.arange(200, 400)]).astype(np.int32)
    bar.set_continuity(frames, [(0, 99), (200, 399)], 400)

    density = bar._density(840)

    assert density.min() == 0.0 and density.max() == 1.0
    assert 0.2 < float((density == 0).mean()) < 0.35  # ~the 100-frame hole in 400


def test_scrubbing_to_the_window_edge_slides_it(qapp, tmp_path):
    """Reaching the edge means the behavior runs on past what the window happens to show.
    Stopping there is an artifact of how much context was loaded, not of the video."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    before = player.window()
    grew: list = []
    player.window_changed.connect(lambda: grew.append(player.window()))

    player.scrubber().set_frame(player._len - 1, decode=False)  # scrub to the end

    assert grew, "the window did not move at its edge"
    assert grew[-1][1] > before[1]


def test_edge_walk_starts_at_video_speed_and_ramps_to_a_cap():
    """A short hold must be frame-precise, a sustained one must actually get somewhere, and
    the top speed is capped low: this moves the *view*, and losing your place is worse than
    arriving slowly."""
    from detail.gui.behavior_query import _EDGE_MAX_STEP, edge_walk_step

    assert edge_walk_step(0) == 1  # first touch is exactly one frame
    assert edge_walk_step(3) == 1  # still precise through the first ~0.5 s
    assert edge_walk_step(4) == 2  # then ramps
    assert edge_walk_step(8) == _EDGE_MAX_STEP == 3
    assert edge_walk_step(10_000) == _EDGE_MAX_STEP  # and never past the cap
    assert edge_walk_step(-5) == 1  # a negative hold is not a rewind


def test_holding_at_the_edge_walks_at_the_ramped_rate(qapp, tmp_path):
    """Panning straight from the scrub signal ran away: sliding re-bases the playhead, the
    in-progress drag re-pins it to the edge, and that pans again — half a window per
    mouse-move. Driving it from a timer decouples the speed from the mouse."""
    from detail.gui.behavior_query import edge_walk_step

    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    before = player.window()

    player.scrubber().set_frame(player._len - 1, decode=False)  # grab the edge

    assert player._edge_pan.isActive(), "holding the edge did not start the walk"
    assert player.window()[0] == before[0] + 1, "first touch slid more than a frame"

    expected = 1
    for tick in range(1, 9):  # as the timer would fire
        expected += edge_walk_step(tick)
        player._pan_one_frame()
    assert player.window()[0] == before[0] + expected
    assert player.window()[1] - player.window()[0] == before[1] - before[0]


def test_regrabbing_the_edge_starts_over_at_video_speed(qapp, tmp_path):
    """Otherwise a second nudge inherits the previous hold's speed and overshoots."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    player.scrubber().set_frame(player._len - 1, decode=False)
    for _ in range(10):
        player._pan_one_frame()
    player.scrubber().set_frame(player._len // 2, decode=False)  # let go

    mid = player.window()
    player.scrubber().set_frame(player._len - 1, decode=False)  # grab again

    assert player.window()[0] == mid[0] + 1


def test_the_edge_walk_stops_when_the_playhead_comes_back_inside(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    player.scrubber().set_frame(player._len - 1, decode=False)
    assert player._edge_pan.isActive()

    player.scrubber().set_frame(player._len // 2, decode=False)

    assert not player._edge_pan.isActive()


def test_moving_to_another_match_stops_the_edge_walk(qapp, tmp_path):
    """A timer left running would keep sliding the window under the next clip."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    player.scrubber().set_frame(player._len - 1, decode=False)
    assert player._edge_pan.isActive()

    _long_clip(player, start_s=300.0)

    assert not player._edge_pan.isActive()


def test_autoplay_stops_at_the_clip_not_the_window(qapp, tmp_path):
    """The scrubber spans a window of surrounding context so the user can scrub into it.
    Playback must not follow: playing on past the clip runs to the end of the recording,
    which is neither what a match preview is for nor anything the user asked to watch.
    """
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    window_before = player.window()
    _lo, clip_end = player.clip_span()
    player._scrubber.set_frame(player._playback_range()[0], decode=False)
    player._scrubber._has_rendered = True
    player.play()

    for _ in range(500):  # far more than the clip is long
        player._scrubber._advance()

    assert not player.is_playing(), "autoplay ran past the clip"
    assert player.window()[0] + player._scrubber.frame() == clip_end
    assert player.window() == window_before, "playback panned the window"


def test_scrubbing_past_the_clip_still_works_after_playback_stops(qapp, tmp_path):
    """Stopping playback at the clip must not stop the *user* from looking further —
    that is the whole point of the surrounding window."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    _long_clip(player)
    _lo, clip_end = player.clip_span()

    beyond = clip_end - player.window()[0] + 20
    player._scrubber.set_frame(beyond, decode=False)

    assert player.window()[0] + player._scrubber.frame() > clip_end


def test_a_non_trim_player_still_stops_at_the_end_of_its_clip(qapp, tmp_path):
    """Guard the guard: the compare view's players must not grow a window they do not have."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 200_000_000, "q"))
    if player._len <= 1:
        return  # a 1-frame clip has nothing to play
    window_before = player.window()
    player._scrubber._has_rendered = True
    player.play()
    for _ in range(player._len + 3):  # drive past the end
        player._scrubber._advance()

    assert not player.is_playing()
    assert player.window() == window_before  # and no window appeared to grow


def test_coverage_outside_the_kept_span_is_drawn_back(qapp, tmp_path):
    """The span has to read as the subject and its surroundings as context; one uniform
    weight of ink makes the eye hunt for it."""
    from detail.gui.behavior_query import _OUTSIDE_SPAN_ALPHA

    assert 0.0 < _OUTSIDE_SPAN_ALPHA < 1.0  # dimmed, never erased
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=2, walk_edges=True)
    bar = _CoverageBar(player.scrubber())
    bar.resize(400, bar.height())
    bar.set_continuity(np.arange(400, dtype=np.int32), [(0, 399)], 400)
    bar.set_bracket((100, 200))

    bar.set_draggable(False)
    assert bar._muted_range(0, 400) is None  # a read-only strip is drawn evenly

    bar.set_draggable(True)
    inside = bar._muted_range(0, 400)
    assert inside is not None and inside[0] < inside[1]
    # ...and it covers the span, not the whole strip
    assert inside[0] > 0 and inside[1] < 400


def test_playback_runs_at_the_recordings_own_frame_rate(qapp, tmp_path):
    """Real time, measured from the recording — not a fixed timer.

    The player advanced one frame every 120 ms for every recording: ~8 fps, whatever the
    camera did. On a 60 fps recording that is **1/7th speed**, which is not a considered
    choice about reviewing behaviour — it is a constant that predates the data. Judging
    whether a clip is the behaviour depends on seeing it at the speed it happened.
    """
    from detail.gui.behavior_query import _ClipPlayer, _interval_ms

    assert _interval_ms(30.0) == 33
    assert _interval_ms(60.0) == 17

    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    # Asserted through the scrubber's own timer now that it owns playback — the rate is
    # still the recording's, it is just no longer a second timer in this class.
    player._preview.frame_rate = lambda *_a, **_k: 60.0
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 4000, "x", score=0.5))
    assert player._scrubber._play_timer.interval() == 17

    player._preview.frame_rate = lambda *_a, **_k: 25.0
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 4000, "y", score=0.5))
    assert player._scrubber._play_timer.interval() == 40


def test_example_dialog_accepts_a_gap_crossing_clip_when_the_method_pools_frames(
    qapp, tmp_path
):
    """The pain point, at the surface a user meets it.

    Same fixture and same marks as `test_example_dialog_refuses_a_clip_that_crosses_a_gap`
    — every embedded frame is its own stretch at this split gap — but under a method that
    pools exemplar frames rather than matching a path. The clip is taken, and the break is
    *described* rather than used to refuse it: a user who marked across a dropout should know
    the animal was not seen for part of what they marked.

    The pair is the guard. One of these alone passes with the rule hardcoded either way.
    """
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(
        qapp,
        ClipExampleDialog(
            proj,
            proj.recordings,
            "emb_01",
            max_gap_ns=1000,
            require_contiguous=False,
        ),
    )
    assert len(dialog._continuity["runs"]) == 3

    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    # No bracket: outlining the containing stretch would claim a limit this method does not
    # impose, which is the shared-widget failure — a cue set beside a rule it does not follow.
    assert dialog._coverage._bracket is None
    dialog._scrubber.set_frame(4, decode=False)
    dialog._mark(start=False)

    assert len(dialog.examples) == 1
    assert dialog.examples[0]["start_t_ns"] < dialog.examples[0]["end_t_ns"]
    status = dialog._span_status.text()
    assert "cannot use" not in status
    assert "breaks in tracking" in status
    assert "skipped" in status


# --- masks on the frame, instead of a silhouette beside it --------------------------------- #
def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    """Spin the event loop until ``predicate`` holds; every load here is off-thread."""
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


def test_the_clip_editor_reads_the_cheap_path_until_masks_are_asked_for(qapp, tmp_path):
    """The toggle has to change **which decode runs**, not just what is drawn.

    This is the whole point of it. The silhouette pane it replaces cost several times what
    the frame alone does — so a toggle that decoded the same
    thing and hid it would look identical on screen and leave playback exactly as slow, which is
    the version of this change that does nothing.

    Asserted by watching which accessor is called, because the two return the same shape and a
    rendered frame cannot tell you which produced it.
    """
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    called: list[str] = []
    dialog._preview.frame_plain = lambda *a, **k: called.append("plain") or None
    dialog._preview.frame_masked = lambda *a, **k: called.append("masked") or None

    dialog._masks.setChecked(False)  # on by default, so this is a toggle and redraws
    # The redraw decodes on a pool thread and may land before or after the call below; let it
    # settle and clear, so what is asserted is this call's path rather than a race with it.
    _drain(qapp, lambda: bool(called), timeout_ms=2_000)
    called.clear()
    dialog._compute(0)
    assert called == ["plain"], "masks off must not decode or draw them"

    dialog._masks.setChecked(True)
    # Ticking it redraws the frame on screen, which decodes on a pool thread and may land
    # before or after the assertion below. Let that settle and clear, so what is asserted is
    # the *next* decode's path rather than a race with the redraw.
    _drain(qapp, lambda: bool(called), timeout_ms=2_000)
    called.clear()
    dialog._compute(0)
    assert called == ["masked"], "masks on must take the path that draws them"


def test_toggling_masks_redraws_the_frame_already_on_screen(qapp, tmp_path):
    """Otherwise the tick does nothing until you move, which reads as a broken control.

    The frame is re-requested at the position it is already at — the cheap way to say "decode
    this again, differently", and it works whether playing or paused.
    """
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    dialog._scrubber.set_total(50)
    dialog._scrubber.set_frame(17, decode=False)
    asked: list[int] = []

    def note(frame):
        asked.append(frame)
        # The `(payload, reason)` pair `frame_source.decode` returns — `_display` unpacks it,
        # and a bare None would fail there rather than telling us what was asked for.
        return None, frame_source.UNDECODABLE

    dialog._scrubber.set_decoder(note)

    dialog._masks.setChecked(False)  # on by default, so the toggle is to off

    assert _drain(
        qapp, lambda: 17 in asked
    ), "toggling Masks left the frame on screen as it was"


def test_the_clip_editor_shows_one_pane(qapp, tmp_path):
    """The silhouette pane is gone, not hidden.

    A hidden pane still holds its share of the width in some layouts, and the reason to remove
    it was the space as much as the decode: it took two fifths of the row to answer a question
    about the *method* beside footage you are watching to judge an animal.
    """
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")

    assert not hasattr(dialog, "_sil"), "the silhouette pane is still being built"


# --- masks are a control, on every player that can reach one ------------------------------- #
def test_every_clip_player_offers_the_mask_toggle(qapp, tmp_path):
    """A capability that lands on the surface being edited and reaches no other.

    Thirteen surfaces are built on `_ClipPlayer` — the clustering result, the behavior scan, the
    audit tab, measurements, the spatial map, the annotation pages. Eleven took the default
    `masked=True` and had no way to turn it off, so they drew the outline whether anyone was
    studying it or not and paid **10.39 ms a frame against 4.64** for it. Two more surfaces had
    the choice because somebody was looking at those two.

    Asserted on the class rather than per call site, because that is what makes it true for the
    fourteenth: the toggle is built in `__init__`, so a new player cannot ship without it.
    """
    proj = _embed_video_experiment(tmp_path)

    for kwargs in (
        {"context_s": 0},
        {"masked": True, "context_s": 0},
        {"masked": False, "context_s": 0},
        {"context_s": 2, "walk_edges": True},
    ):
        player = _ClipPlayer(proj, **kwargs)
        assert player._masks is not None, f"no mask control on _ClipPlayer({kwargs})"
        assert player._masks in player._scrubber.findChildren(
            type(player._masks)
        ), f"the mask control is not in the player's control row for {kwargs}"


def test_the_constructor_flag_sets_where_the_toggle_starts(qapp, tmp_path):
    """`masked=` used to fix the behaviour; it is the opening position now.

    Both directions, because "always ticked" passes a one-sided check and is the same bug as
    "never ticked" — a surface that cannot turn masks off is what this replaces.
    """
    proj = _embed_video_experiment(tmp_path)

    assert _ClipPlayer(proj, masked=True, context_s=0).masks_shown()
    assert not _ClipPlayer(proj, masked=False, context_s=0).masks_shown()
    assert _ClipPlayer(
        proj, context_s=0
    ).masks_shown(), "the default stays masked, as every surface had it"


def test_a_player_decodes_masks_only_while_the_toggle_is_on(qapp, tmp_path):
    """The toggle has to change which decode runs, not just what is drawn.

    A toggle that decoded the same thing and hid it would look identical and leave playback
    exactly as slow — the version of this change that does nothing.
    """
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, masked=False, context_s=0)
    player.set_clip(ClipSpec("recording_001", 0, "emb_01", 0, 200_000_000, "q"))
    called: list[str] = []
    player._preview.frame_plain = lambda *a, **k: called.append("plain") or None
    player._preview.frame_masked = lambda *a, **k: called.append("masked") or None

    player._compute(0)
    assert called == ["plain"]

    player._masks.setChecked(True)
    _drain(qapp, lambda: bool(called) and called[-1] == "masked", timeout_ms=2_000)
    called.clear()
    player._compute(0)
    assert called == ["masked"]


def _inner_clip() -> ClipSpec:
    """A clip *inside* the six-frame, 30 fps fixture video — frames 2..3, not the whole of it.

    A clip covering all of it leaves the padding nowhere to go and every context amount is
    identical, which is how the first version of the popup test passed against padding.
    """
    return ClipSpec("recording_001", 0, "emb_01", 66_666_666, 100_000_000, "q")


def test_a_popup_opened_without_context_spans_the_clip_and_nothing_either_side(
    qapp, tmp_path
):
    """The query grids open the popup on a match judged on its own: the bar is the clip."""
    from detail.gui.clip_popup import ClipPopup

    proj = _embed_video_experiment(tmp_path)
    popup = ClipPopup(proj, _inner_clip(), context_s=0)

    player = popup.player()
    assert player.clip_span()[1] > player.clip_span()[0]
    assert player.window() == player.clip_span()
    assert player.scrubber().marks() == (
        (),
        (),
    ), "a span the width of the bar says nothing"


def test_context_pads_the_window_and_marks_the_clip_on_the_bar(qapp, tmp_path):
    """The clustering page's ask: a second either side, with the clip itself marked.

    The fixture is 0.2 s long, so a second either side clamps to the video at both ends — the
    clamp is part of what is asserted: a clip near the start or end gets what footage there is
    rather than an error.
    """
    from detail.gui.clip_popup import ClipPopup

    proj = _embed_video_experiment(tmp_path)
    popup = ClipPopup(proj, _inner_clip(), context_s=1)
    player = popup.player()

    first, last = player.window()
    lo, hi = player.clip_span()
    assert (first, last) == (
        0,
        5,
    ), "the padding did not clamp to the video's own frames"
    assert first < lo and hi < last
    assert player.scrubber().marks() == ((), ((lo - first, hi - first),))
    assert (
        player.scrubber().frame() == 0
    ), "playback must open at the window's left edge"
    assert player.scrubber().playback_range() == (
        0,
        last - first,
    ), "playback stopped at the clip; the context is there to be watched"


def test_changing_context_rebuilds_the_window_and_keeps_the_frame(qapp, tmp_path):
    """Changing how much surrounds the clip is not a request to start again."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    player.set_clip(_inner_clip())
    lo, _hi = player.clip_span()
    player.scrubber().set_frame(0, decode=False)  # the clip's first frame

    player.set_context_s(1)

    assert player.window() == (0, 5)
    assert player.window()[0] + player.scrubber().frame() == lo
    assert player.scrubber().marks()[
        1
    ], "the clip's span was not drawn after the change"

    player.set_context_s(0)
    assert player.window() == player.clip_span()
    assert player.scrubber().marks() == ((), ())


def test_a_clip_that_fills_most_of_the_bar_is_not_bracketed(qapp, tmp_path):
    """A cluster member that is one unbroken run of a 30-minute video filled the bar edge to
    edge with "the clip" — brackets on the bar's own ends mark nothing. Below 90% they show.
    """
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=1)
    player.set_clip(_inner_clip())
    assert player.window() == (0, 5)

    player._clip_span = (0, 4)  # 5 of 6 frames: 83%
    player._apply_playback_range()
    assert player.scrubber().marks()[1] == ((0, 4),)

    player._clip_span = (0, 5)  # the whole bar
    player._apply_playback_range()
    assert player.scrubber().marks() == ((), ())


def test_the_coverage_strip_has_something_to_show_only_where_tracking_breaks(
    qapp, tmp_path
):
    """A fully tracked recording drew one solid, unlabelled block — "a bar I don't understand"."""
    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=0)
    bar = _CoverageBar(player.scrubber())
    frames = np.arange(1000, dtype=np.int32)

    bar.set_continuity(frames, [(0, 999)], 1000)
    assert not bar.has_gaps(), "one stretch over the whole video is not a gap"
    bar.set_continuity(frames, [(0, 400), (450, 999)], 1000)
    assert bar.has_gaps()
    bar.set_continuity(frames, [(0, 800)], 1000)
    assert bar.has_gaps(), "a fifth of the video untracked at the end is a gap"


def test_the_example_dialog_shows_the_coverage_strip_only_with_a_gap(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _wait_continuity(qapp, ClipExampleDialog(proj, proj.recordings, "emb_01"))
    # The fixture tracks frames 0-4 of a six-frame video: a sixth untracked.
    assert dialog._coverage.has_gaps()
    assert not dialog._coverage.isHidden()
    dialog._coverage.set_continuity(np.arange(6, dtype=np.int32), [(0, 5)], 6)
    dialog._coverage.setVisible(dialog._coverage.has_gaps())  # as `_on_continuity` does
    assert dialog._coverage.isHidden()


def test_an_amount_the_menu_does_not_offer_is_refused(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    with pytest.raises(ValueError, match="offered amounts"):
        _ClipPlayer(proj, context_s=3)


def test_the_player_shows_the_recordings_time_from_its_timestamp_map(qapp, tmp_path):
    """Read from the map the pipeline stamps, sliced to the window — never frame ÷ fps."""
    from detail.gui.timefmt import format_time

    proj = _embed_video_experiment(tmp_path)
    player = _ClipPlayer(proj, context_s=1)
    player.set_clip(_inner_clip())
    clip = _inner_clip()
    t_ns = player._preview.timestamps(clip.recording_id, clip.embed_run_id)

    player.scrubber().set_frame(3, decode=False)

    assert player.scrubber().time_text() == format_time(
        int(t_ns[player.window()[0] + 3])
    )


# --------------------------------------------------------------------------- #
# One pass, every behavior (#628)
# --------------------------------------------------------------------------- #
def _two_behaviors(negatives=None):
    return [
        {"name": "attack", "examples": [], "negatives": list(negatives or [])},
        {"name": "sniffing", "examples": [], "negatives": []},
    ]


def _multi_dialog(qapp, proj, **kwargs):
    kwargs.setdefault("behaviors", _two_behaviors())
    return _wait_continuity(
        qapp,
        ClipExampleDialog(proj, proj.recordings, "emb_01", max_gap_ns=10_000, **kwargs),
    )


def _mark_span(dialog, first: int, last: int) -> None:
    dialog._scrubber.set_frame(first, decode=False)
    dialog._mark(start=True)
    dialog._scrubber.set_frame(last, decode=False)
    dialog._mark(start=False)


def _behavior_at_frame_2(proj) -> dict:
    """One behavior whose clip starts on video frame 2 — at that frame's own timestamp,
    which is the video's, not the fixture's embedding rows'."""
    from detail.api.run_preview import ClipQueryPreview

    preview = ClipQueryPreview(proj.dir)
    clip = {
        "recording_id": "recording_001",
        "track_id": 0,
        "start_t_ns": preview.frame_t_ns("recording_001", "emb_01", 2),
        "end_t_ns": preview.frame_t_ns("recording_001", "emb_01", 4),
    }
    return {"name": "attack", "examples": [clip], "negatives": []}


def _choose_behavior(dialog, index: int) -> None:
    dialog._behavior_table.selectRow(index)


def test_one_pass_marks_examples_and_non_examples_into_each_behavior(qapp, tmp_path):
    """Two behaviors with examples and non-examples was four passes over the video; here it is
    one, and the behavior row plus the kind toggle decide which list a clip joins."""
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 2)  # attack, example — the dialog opens on the first behavior
    _choose_behavior(dialog, 1)
    dialog._as_negative.setChecked(True)
    _mark_span(dialog, 0, 4)
    assert "'sniffing' as a non-example" in dialog._span_status.text()

    attack, sniffing = dialog.behaviors
    assert len(attack["examples"]) == 1 and attack["negatives"] == []
    assert sniffing["examples"] == [] and len(sniffing["negatives"]) == 1
    assert dialog._clip_model.rowCount() == 2
    kinds = {dialog._clip_model.item(r, 1).text() for r in range(2)}
    assert kinds == {"example", "non-example"}
    # The counts beside the player follow.
    assert dialog._behavior_model.item(1, 2).text() == "1"


def test_a_clip_that_is_two_behaviors_is_marked_once_per_behavior(qapp, tmp_path):
    """A clip that is "approached but did not attack": switch the row and mark the span again."""
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 4)
    _choose_behavior(dialog, 1)
    _mark_span(dialog, 0, 4)
    attack, sniffing = dialog.behaviors
    assert attack["examples"] == sniffing["examples"]
    assert attack["examples"][0] is not sniffing["examples"][0]


def test_the_dialog_adds_a_behavior_under_the_setup_tables_rule(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    assert dialog._add_behavior_named("  ") == "A behavior needs a name."
    assert "already listed" in dialog._add_behavior_named("attack")
    assert "already listed" in dialog._status.text()
    assert dialog._add_behavior_named(" rearing ") is None
    assert [b["name"] for b in dialog.behaviors] == ["attack", "sniffing", "rearing"]
    _mark_span(dialog, 0, 2)  # the added behavior is the one being marked
    assert len(dialog.behaviors[2]["examples"]) == 1


def test_a_dialog_opened_with_no_behaviors_says_to_add_one(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj, behaviors=[])
    assert dialog._annotating.text() == "Add a behavior to annotate"
    # No header row over nothing: a line saying what to do, like the empty clip list.
    assert dialog._behavior_table.isHidden() and not dialog._no_behaviors.isHidden()
    assert not dialog._rename_button.isEnabled() and dialog._rename_button.toolTip()
    dialog._mark(start=True)
    assert "add a behavior first" in dialog._span_status.text()


def test_delete_removes_the_selected_clip_by_button_and_by_key(qapp, tmp_path):
    from PyQt6.QtGui import QShortcut

    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 2)
    _mark_span(dialog, 0, 4)
    assert dialog._delete.isEnabled()

    dialog._clip_table.selectRow(0)
    dialog._delete.click()
    assert len(dialog.behaviors[0]["examples"]) == 1
    assert "deleted" in dialog._span_status.text()

    keys = {
        s.key().toString(): s
        for s in dialog._clip_table.findChildren(QShortcut)
        if s.parent() is dialog._clip_table
    }
    assert {"Del", "Backspace"} <= set(keys)
    dialog._clip_table.selectRow(0)
    keys["Del"].activated.emit()
    assert dialog.behaviors[0]["examples"] == []
    assert not dialog._delete.isEnabled() and dialog._delete.toolTip()


def test_selecting_a_clip_seeks_the_player_to_it(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj, behaviors=[_behavior_at_frame_2(proj)])
    dialog._scrubber.set_frame(0, decode=False)
    dialog._clip_table.selectRow(0)
    assert dialog._scrubber.frame() == 2


def test_selecting_a_clip_on_another_track_switches_to_it_then_seeks(qapp, tmp_path):
    from PyQt6.QtTest import QTest

    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj, behaviors=[_behavior_at_frame_2(proj)])
    dialog._track.setValue(3)
    dialog._reload()
    for _ in range(300):  # the other track's read lands (it has no rows, so it fails)
        qapp.processEvents()
        QTest.qWait(10)
        if "loading" not in dialog._mark_start.text():
            break
    dialog._clip_table.selectRow(0)
    for _ in range(300):
        qapp.processEvents()
        QTest.qWait(10)
        if dialog._continuity is not None and dialog._pending_seek is None:
            break
    assert dialog._track.value() == 0
    assert dialog._scrubber.frame() == 2


def test_non_examples_are_greyed_with_the_reason_and_kept_when_unusable(qapp, tmp_path):
    """§20 under a method that cannot use them: greyed, never hidden, and what was already
    marked is kept and labelled rather than dropped."""
    proj = _embed_video_experiment(tmp_path)
    old = {
        "recording_id": "recording_001",
        "track_id": 0,
        "start_t_ns": 0,
        "end_t_ns": 2000,
    }
    reason = "dtw scores distance to the examples."
    dialog = _multi_dialog(
        qapp, proj, behaviors=_two_behaviors([old]), negatives_reason=reason
    )
    assert not dialog._as_negative.isEnabled() and not dialog._as_negative.isHidden()
    assert dialog._as_negative.toolTip() == reason
    assert dialog._behavior_model.item(0, 2).text() == "1 (not used)"
    assert dialog._clip_model.item(0, 5).text() == f"⚠ not used: {reason}"
    assert dialog.behaviors[0]["negatives"] == [old]


def test_mark_buttons_say_they_are_loading(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01", behaviors=[])
    assert dialog._mark_start.text() == "Mark start (loading…)"
    assert not dialog._mark_start.isEnabled() and dialog._mark_start.toolTip()
    _wait_continuity(qapp, dialog)
    assert dialog._mark_start.text() == "Mark start" and dialog._mark_start.isEnabled()


def test_a_track_change_says_it_dropped_the_pending_start(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    dialog._scrubber.set_frame(2, decode=False)
    dialog._mark(start=True)
    dialog._reload()
    assert "dropped" in dialog._span_status.text()
    assert dialog._pending_start is None and dialog._coverage._bracket is None


def test_a_frame_with_no_timestamp_says_why_it_cannot_be_marked(
    qapp, tmp_path, monkeypatch
):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)

    def no_timestamp(*_a):
        raise IndexError("frame 5 out of range")

    monkeypatch.setattr(dialog._preview, "frame_t_ns", no_timestamp)
    dialog._mark(start=True)
    assert "has no timestamp" in dialog._span_status.text()


def test_a_second_mark_start_says_it_moved_the_start(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    dialog._scrubber.set_frame(0, decode=False)
    dialog._mark(start=True)
    dialog._scrubber.set_frame(2, decode=False)
    dialog._mark(start=True)
    assert "start moved to" in dialog._status.text()


def test_the_one_list_mode_keeps_the_hidden_callers_shape(qapp, tmp_path):
    """Clip query and clip annotation (hidden, §25) still pass ``initial`` and read
    ``.examples``; they get no behavior table and no kind toggle."""
    proj = _embed_video_experiment(tmp_path)
    clip = {
        "recording_id": "recording_001",
        "track_id": 0,
        "start_t_ns": 0,
        "end_t_ns": 2000,
    }
    dialog = _wait_continuity(
        qapp, ClipExampleDialog(proj, proj.recordings, "emb_01", initial=[clip])
    )
    assert dialog._behavior_table.isHidden() and dialog._as_negative.isHidden()
    assert dialog._annotating.isHidden()
    assert dialog._clip_table.isColumnHidden(0) and dialog._clip_table.isColumnHidden(1)
    _mark_span(dialog, 0, 4)
    assert len(dialog.examples) == 2 and dialog.examples[0] == clip


def test_mark_examples_opens_every_behavior_and_takes_back_the_result(
    qapp, tmp_path, monkeypatch
):
    """The setup table's one button: a "Start from" config reopens every behavior's examples
    and non-examples, and what the dialog returns — names added there included — replaces
    the table's behaviors."""
    from PyQt6.QtWidgets import QDialog

    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES

    proj = _embed_video_experiment(tmp_path)
    clip = {
        "recording_id": "recording_001",
        "track_id": 0,
        "start_t_ns": 0,
        "end_t_ns": 2000,
    }
    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    widget.on_experiment_set(proj)
    widget.on_recordings_changed(proj.recordings)
    widget.load_config(
        {
            "method": "soft_knn",
            "behaviors": [
                {"name": "attack", "examples": [clip], "negatives": [clip]},
                {"name": "sniffing", "examples": [clip], "negatives": []},
            ],
        }
    )
    widget._table.selectRow(1)
    seen = {}

    def accept(dialog):
        seen["behaviors"] = [dict(b) for b in dialog.behaviors]
        seen["active"] = dialog._active
        seen["reason"] = dialog._negatives_reason
        dialog._add_behavior_named("rearing")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ClipExampleDialog, "exec", accept)
    widget._mark_examples.click()
    assert seen["behaviors"][0]["negatives"] == [clip]
    assert seen["active"] == 1 and seen["reason"] is None
    assert [b["name"] for b in widget._behaviors] == ["attack", "sniffing", "rearing"]
    assert widget._model.item(0, 2).text() == "1"


def test_the_banner_names_what_is_being_annotated(qapp, tmp_path):
    """The banner over the Mark pair, not a sentence in the side column, says where the next
    clip goes — the "next clip goes to 'hunch' as an example" line read as noise."""
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    assert dialog._annotating.text() == "Annotating: attack"
    _choose_behavior(dialog, 1)
    dialog._as_negative.setChecked(True)
    assert dialog._annotating.text() == "Annotating: sniffing — non-examples"


def test_rename_and_remove_live_in_the_dialog(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 4)
    assert "already listed" in dialog._rename_behavior_named("sniffing")
    assert dialog._rename_behavior_named("bite") is None
    assert dialog.behaviors[0]["name"] == "bite"
    assert dialog._clip_model.item(0, 0).text() == "bite"  # the clip list follows
    dialog._remove_behavior()
    assert [b["name"] for b in dialog.behaviors] == ["sniffing"]
    assert dialog._clip_model.rowCount() == 0
    assert "removed 'bite' and its 1 clip" in dialog._status.text()


def test_progress_fills_toward_three_examples(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)

    def bar():
        index = dialog._behavior_model.index(0, 1)
        return dialog._behavior_table.indexWidget(index)

    assert bar().text() == "0/3"
    for _ in range(3):
        _mark_span(dialog, 0, 4)
    assert bar().text() == "3 ✓" and bar().value() == bar().maximum()


def test_an_empty_clip_list_is_a_line_saying_how_to_start(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    assert not dialog._no_clips.isHidden() and dialog._clip_table.isHidden()
    _mark_span(dialog, 0, 4)
    assert dialog._no_clips.isHidden() and not dialog._clip_table.isHidden()


def test_the_recording_picker_counts_each_recordings_clips(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    name = proj.recordings[0].name
    assert dialog._recording.itemText(0) == name
    _mark_span(dialog, 0, 4)
    assert dialog._recording.itemText(0) == f"{name} · 1 clip"


def test_balance_across_offers_the_metadata_columns(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    proj.set_recording_metadata("recording_001", {"treatment": "saline"})
    dialog = _multi_dialog(qapp, proj)
    assert dialog._balance.findData("treatment") >= 0
    assert dialog._balance.findData("acquisition_setup") == -1  # blank everywhere
    _mark_span(dialog, 0, 4)
    dialog._balance.setCurrentIndex(dialog._balance.findData("treatment"))
    assert "every treatment" in dialog._hint.text()


def test_balance_across_is_greyed_without_metadata(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    assert not dialog._balance.isEnabled() and "no recording metadata" in (
        dialog._balance.toolTip()
    )


def test_clips_draw_on_the_timeline_in_their_behaviors_colour(qapp, tmp_path):
    """#627's marks: an example filled in its behavior's colour, a non-example hollow, and a
    pending start's tick alongside them."""
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 2)
    _choose_behavior(dialog, 1)
    dialog._as_negative.setChecked(True)
    _mark_span(dialog, 0, 4)
    ticks, _spans = dialog._scrubber.marks()
    assert ticks == ()
    assert set(dialog._scrubber.bands()) == {
        (0, 2, dialog._behavior_color(0), False),
        (0, 4, dialog._behavior_color(1), True),
    }
    assert dialog._behavior_color(0) != dialog._behavior_color(1)
    dialog._scrubber.set_frame(3, decode=False)
    dialog._mark(start=True)
    assert dialog._scrubber.marks()[0] == (3,)
    assert len(dialog._scrubber.bands()) == 2  # the tick does not wipe the clips
    dialog._reload()
    assert dialog._scrubber.bands() == ()  # nothing drawn until the subject is read
    _wait_continuity(qapp, dialog)
    assert len(dialog._scrubber.bands()) == 2


def test_the_setup_table_is_a_summary_with_one_way_in(qapp):
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES

    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    texts = {b.text() for b in widget.findChildren(QPushButton)}
    assert "Mark examples…" in texts
    assert not texts & {"Add behavior", "Rename", "Remove", "Edit clips"}


def test_the_clip_list_fills_its_column(qapp, tmp_path):
    """It shows every clip rather than three rows and a scrollbar.

    It used to fill its column to get there; in a pop-up every list now shows all of its rows
    (`sizing.shows_every_row`), so the table hugs them and the window grows to hold it — which
    answers the same complaint without blank grid under the last clip.
    """
    from detail.gui.sizing import hidden_extent

    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 4)
    dialog.show()
    for _ in range(4):
        qapp.processEvents()
    try:
        table = dialog._clip_table
        assert table.model().rowCount() >= 1
        assert hidden_extent(table)[1] == 0, "a clip is scrolled out of the list"
    finally:
        dialog.close()


def test_the_video_top_is_level_with_the_checklist(qapp, tmp_path):
    """Pinned, not centred: a video whose shape leaves slack above it floated below the
    checklist beside it — and the slack goes under the Mark pair, not between it and the
    transport."""
    from PyQt6.QtCore import QPoint

    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    dialog.resize(1400, 740)
    dialog.show()
    for _ in range(20):
        qapp.processEvents()
    try:
        assert dialog._block.picture.y() == 0
        block_top = dialog._block.mapTo(dialog, QPoint(0, 0)).y()
        table_top = dialog._behavior_table.mapTo(dialog, QPoint(0, 0)).y()
        assert abs(block_top - table_top) <= 2
        transport_bottom = dialog._scrubber.mapTo(
            dialog, QPoint(0, dialog._scrubber.height())
        ).y()
        box = dialog._mark_start.parentWidget()  # the Annotating box
        box_top = box.mapTo(dialog, QPoint(0, 0)).y()
        assert 0 <= box_top - transport_bottom <= 12
    finally:
        dialog.close()


def test_widening_the_window_widens_the_clip_list_too(qapp, tmp_path):
    """A fixed checklist kept every extra pixel for the video, and the clip list's last
    columns stayed cut off however wide the window got. Note shows only when a clip has one.
    """
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 4)
    dialog.show()
    try:
        dialog.resize(1100, 740)
        qapp.processEvents()
        narrow = dialog._checklist.width()
        dialog.resize(1600, 740)
        qapp.processEvents()
        assert dialog._checklist.width() > narrow
        table = dialog._clip_table
        assert table.isColumnHidden(5), "an empty Note column took the width"
        used = sum(
            table.columnWidth(c) for c in range(5) if not table.isColumnHidden(c)
        )
        assert used <= table.viewport().width(), "the last column is still cut off"
    finally:
        dialog.close()


def test_track_shows_only_once_clips_are_on_two_animals(qapp, tmp_path):
    proj = _embed_video_experiment(tmp_path)
    dialog = _multi_dialog(qapp, proj)
    _mark_span(dialog, 0, 4)
    assert dialog._clip_table.isColumnHidden(4), "one animal: Track says nothing"
    second = dict(dialog.behaviors[0]["examples"][0], track_id=1)
    dialog.behaviors[1]["examples"].append(second)
    dialog._refresh_clips()
    assert not dialog._clip_table.isColumnHidden(4)


def test_the_transport_is_never_cut_off_at_the_default_size(qapp, tmp_path):
    """At 1100x740 a 16:9 video's button row was cut in half: the "tracked" strip arrived after
    the first layout and nothing re-measured. Too small a window scrolls instead."""
    from detail.gui.requirements import a_players_controls_get_the_height_they_need

    original = _tiny_video
    globals()["_tiny_video"] = lambda path, n=6, w=64, h=36: original(path, n, w, h)
    try:
        proj = _embed_video_experiment(tmp_path)
    finally:
        globals()["_tiny_video"] = original
    # Shown *before* the track is read, as a user opens it: the strip arrives after layout.
    dialog = ClipExampleDialog(
        proj, proj.recordings, "emb_01", max_gap_ns=10_000, behaviors=[]
    )
    dialog.resize(1100, 740)
    dialog.show()
    try:
        qapp.processEvents()
        _wait_continuity(qapp, dialog)
        for _ in range(10):
            qapp.processEvents()
        a_players_controls_get_the_height_they_need(dialog._scrubber)
    finally:
        dialog.close()


def test_masks_are_on_by_default_like_every_other_player(qapp, tmp_path):
    """Mark examples opened with masks off, the one player (with query-eval's explorer, asserted
    in `test_run_detail.py`) a user found unmasked against the tool's default."""
    proj = _embed_video_experiment(tmp_path)
    dialog = ClipExampleDialog(proj, proj.recordings, "emb_01")
    assert dialog._masks.isChecked()
