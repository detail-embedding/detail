"""The *Min contour area* estimate (``detail.analysis.contour_area``) and its API and CLI.

The estimate sizes a labeling filter from the video alone: one grey frame every few seconds,
a median background, and the largest moving blob taken as the animal. What is pinned here:

- **The suggestion rule**, on the Linear Track pilot's own numbers (the case it was built for)
  and on the edges: a conflict suggests nothing, nothing moving suggests nothing, and a rounded
  value never falls to the floor.
- **Sampling reads the timestamp map**, not a frame rate: spacing is even in time on a
  variable-rate part, a long recording widens the spacing to stay within the cap, and a part's
  frame count is its map's length.
- **A real video end to end**: a bright rectangle moving over a dark background is measured at
  its own area, and a small moving speck lands in "other".
- **Progress moves within one recording** (``long-op-progress``): a single-part,
  single-recording run reports a fraction strictly between 0 and 1 while reading, and its phase
  goes Reading → Measuring.
"""

from fractions import Fraction

from click.testing import CliRunner
import numpy as np
import pytest

from detail.analysis import contour_area as ca
from detail.api.experiment import Experiment
from detail.cli import main
from detail.core import progress as progress_mod

W, H = 200, 150
ANIMAL = (40, 30)  # px: 1,200 px
SPECK = 10  # px side: 100 px


def _row(name, *, p1, p50_other, p95_other, animal_min=None):
    return ca.RecordingAreas(
        recording_id=name,
        name=name,
        n_frames=435,
        animal_min=animal_min if animal_min is not None else p1 - 10,
        animal={1: p1, 5: p1 + 100, 50: p1 * 2, 95: p1 * 3, 99: p1 * 3},
        n_other=1000,
        other={50: p50_other, 95: p95_other, 99: p95_other + 300},
    )


# --- the suggestion rule ---------------------------------------------------- #
def test_the_linear_track_pilot_numbers_suggest_450_and_flag_the_overlap():
    rows = [
        _row("M2", p1=1191, p50_other=202, p95_other=726),
        _row("M4", p1=1189, p50_other=126, p95_other=837),
        _row("M5", p1=953, p50_other=106, p95_other=575),
    ]
    value, ceiling, floor, reason, warnings = ca.suggest(rows)
    assert value == 450
    assert ceiling == pytest.approx(476.5)
    assert floor == 202
    assert "M5" in reason
    assert len(warnings) == 1
    for name in ("M2", "M4", "M5"):
        assert name in warnings[0]


def test_when_the_typical_other_blob_reaches_half_the_animal_nothing_is_suggested():
    value, ceiling, floor, reason, _ = ca.suggest(
        [_row("a", p1=400, p50_other=250, p95_other=300)]
    )
    assert value is None
    assert ceiling == 200 and floor == 250
    assert "No value separates them" in reason


def test_nothing_moving_suggests_nothing():
    empty = ca.RecordingAreas("a", "a", 10, None, {}, 0, {})
    value, ceiling, floor, reason, warnings = ca.suggest([empty])
    assert (value, ceiling, floor, warnings) == (None, None, None, [])
    assert "No moving animal" in reason


def test_no_other_blobs_puts_the_floor_at_zero():
    row = ca.RecordingAreas("a", "a", 10, 900.0, {1: 1000.0, 50: 2000.0}, 0, {})
    value, _ceiling, floor, _reason, warnings = ca.suggest([row])
    assert floor == 0.0
    assert value == 500
    assert warnings == []


def test_a_rounded_value_never_falls_to_the_floor():
    # ceiling 145 rounds to 100, which is not above a floor of 120: keep the ceiling itself.
    value, *_ = ca.suggest([_row("a", p1=290, p50_other=120, p95_other=130)])
    assert value == 145


@pytest.mark.parametrize(
    "value, expected", [(476.5, 450), (1189, 1000), (97, 95), (8.9, 8), (0, 0)]
)
def test_round_down(value, expected):
    assert ca.round_down(value) == expected


# --- sampling on the timestamp map ------------------------------------------ #
def test_sampling_is_even_in_time_on_a_variable_rate_part():
    # 10 ms apart for 1 s, then 50 ms apart for 2 s: a frame-count stride would sample the
    # first second five times as densely as the rest.
    t = np.concatenate(
        [np.arange(100) * 10_000_000, 1_000_000_000 + np.arange(40) * 50_000_000]
    ).astype(np.int64)
    (idx,) = ca.sample_indices([t])
    assert np.all(np.diff(t[idx]) == 500_000_000)


def test_a_long_recording_widens_the_spacing_to_the_cap():
    t = np.arange(10_000, dtype=np.int64) * 10_000_000  # 100 s at 100 fps
    (idx,) = ca.sample_indices([t], sample_every_s=1.0, max_samples=20)
    assert len(idx) <= 21
    # The span over the cap (99.99 s / 20), less one frame of rounding up to the next frame.
    assert np.diff(t[idx]).min() >= 99_990_000_000 // 20 - 10_000_000


def test_parts_are_sampled_on_one_clock_and_counted_by_their_maps():
    first = np.arange(50, dtype=np.int64) * 100_000_000  # 0 .. 4.9 s
    second = 5_000_000_000 + np.arange(30, dtype=np.int64) * 100_000_000
    a, b, empty = ca.sample_indices(
        [first, second, np.zeros(0, np.int64)], sample_every_s=1.0
    )
    assert list(a) == [0, 10, 20, 30, 40]
    assert list(b) == [0, 10, 20]
    assert b.max() < len(second)
    assert len(empty) == 0


def test_a_short_sample_is_flagged():
    row = ca.RecordingAreas("a", "short clip", 2, 900.0, {1: 1000.0, 50: 1000.0}, 0, {})
    *_, warnings = ca.suggest([row])
    assert any("short clip" in w and "Fewer than" in w for w in warnings)


# --- a real video ------------------------------------------------------------ #
def _frame(i: int) -> np.ndarray:
    grey = np.full((H, W), 30, np.uint8)
    x = 10 + (i * 11) % (W - ANIMAL[0] - 20)
    grey[60 : 60 + ANIMAL[1], x : x + ANIMAL[0]] = 210
    sx = 20 + (i * 13) % (W - 40)
    grey[10 : 10 + SPECK, sx : sx + SPECK] = 200
    return np.repeat(grey[:, :, None], 3, axis=2)


def _write_video(path, n_frames=60):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(1, 1))
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 4_000_000
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(_frame(i), format="bgr24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


@pytest.fixture
def measured_experiment(tmp_path):
    experiment = Experiment.create("p", tmp_path / "exp")
    recording = experiment.add_recording("mouse a")
    recording.add_video_part(_write_video(tmp_path / "a.mp4"))
    experiment.add_recording("no video")
    return experiment


def _video(experiment, name):
    from detail.storage import recording as recording_store

    recording = next(r for r in experiment.recordings if r.name == name)
    data = recording_store.read_recording(recording.dir)
    parts = [
        (
            recording_store.resolve_part_video_path(recording.dir, part),
            recording_store.load_active_timestamps(recording.dir, data, i).values,
        )
        for i, part in enumerate(data["video"]["files"])
    ]
    return ca.RecordingVideo(recording.id, recording.name, parts, (W, H))


def test_a_moving_rectangle_is_measured_at_its_own_area(measured_experiment):
    result = ca.estimate([_video(measured_experiment, "mouse a")])
    (row,) = result.per_recording
    assert row.n_frames == 15  # one every 4 s of a 60 s video
    area = ANIMAL[0] * ANIMAL[1]
    assert 0.85 * area <= row.animal[50] <= 1.15 * area
    assert row.n_other >= row.n_frames // 2
    assert 0.5 * SPECK**2 <= row.other[50] <= 1.5 * SPECK**2
    assert result.floor < result.suggestion <= result.ceiling


def test_progress_moves_within_one_recording_and_names_its_phases(
    measured_experiment, monkeypatch
):
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    events = []
    ca.estimate(
        [_video(measured_experiment, "mouse a")],
        progress=lambda *e: events.append(e),
    )
    reading = [f for stage, f, _ in events if stage.startswith("Reading")]
    assert any(0 < f < 1 for f in reading)
    assert reading == sorted(reading)
    stages = [stage.split(" ")[0] for stage, _, _ in events]
    assert stages.index("Measuring") > stages.index("Reading")
    assert events[-1][:2] == ("Done", 1.0)


def test_estimate_refuses_no_recordings():
    with pytest.raises(ValueError, match="at least one recording"):
        ca.estimate([])


# --- the API and the CLI ----------------------------------------------------- #
def test_the_api_reads_every_recording_with_video_by_default(measured_experiment):
    events = []
    result = measured_experiment.estimate_min_contour_area(
        progress=lambda *e: events.append(e)
    )
    assert [r.name for r in result.per_recording] == ["mouse a"]
    assert events and events[-1][1] == 1.0


def test_the_api_refuses_a_named_recording_without_video(measured_experiment):
    no_video = next(r for r in measured_experiment.recordings if r.name == "no video")
    with pytest.raises(ValueError, match="'no video' has no video"):
        measured_experiment.estimate_min_contour_area([no_video.id])


def test_the_cli_prints_the_table_and_the_suggestion(measured_experiment):
    out = CliRunner().invoke(
        main,
        [
            "dataset",
            "estimate-min-area",
            "--experiment",
            str(measured_experiment.dir),
            "--recording",
            "mouse a",
        ],
    )
    assert out.exit_code == 0, out.output
    assert "mouse a" in out.output
    assert "suggested --min-contour-area:" in out.output


def test_the_cli_names_what_it_refuses(measured_experiment):
    out = CliRunner().invoke(
        main,
        [
            "dataset",
            "estimate-min-area",
            "--experiment",
            str(measured_experiment.dir),
            "--recording",
            "no video",
        ],
    )
    assert out.exit_code != 0
    assert "has no video" in out.output
