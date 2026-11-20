"""The *Min contour area* estimate (``detail.analysis.contour_area``) and its API and CLI.

The estimate sizes a labeling filter from the video alone: one grey frame every few seconds,
a median background, and the moving blobs of animal size taken as the animals. What is pinned
here:

- **It is species-agnostic.** The class this guards against is *a size or a contrast of one
  kind of subject built into the method*: a fixed grey-level threshold, a fixed smoothing
  kernel, "the largest blob is the animal", or a species word in what the user reads. Each
  synthetic case is one that such an assumption gets wrong: a subject a few pixels across (a
  fixed 5x5 opening erases it), a low-contrast subject (a fixed threshold of 20 misses it), a
  large dark subject on a light background, two subjects at once (the second would be called
  noise), and the same scene at three times the scale (the suggestion must scale with it).
- **The suggestion rule** on a worked example (the Linear Track pilot's numbers, as inputs
  only) and on the edges: a conflict suggests nothing, nothing moving suggests nothing, and a
  rounded value never falls to the floor.
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


# --- species-agnostic: synthetic subjects of very different size and contrast ---- #
N_FRAMES = 15


def _scene(size, *, background, noise, seed):
    """``N_FRAMES`` flat ``background`` frames with Gaussian ``noise``, and the generator."""
    width, height = size
    rng = np.random.default_rng(seed)
    frames = [
        float(background) + rng.normal(0, noise, (height, width))
        for _ in range(N_FRAMES)
    ]
    return frames, rng


def _render(frames, rng, boxes):
    out = []
    for grey, placed in zip(frames, boxes):
        img = grey.copy()
        for x, y, w, h, level in placed:
            img[y : y + h, x : x + w] = level + rng.normal(0, 2.0, (h, w))
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def _moving(size, w, h, level, *, bands=1, background, noise=2.0, seed=0):
    """Frames with a ``w`` x ``h`` subject at grey ``level`` in each of ``bands`` horizontal
    bands. Each visits five positions, so no pixel is covered in more than a fifth of the
    frames and the median background is clean."""
    width, height = size
    frames, rng = _scene(size, background=background, noise=noise, seed=seed)
    band_h = height // bands
    boxes = []
    for i in range(N_FRAMES):
        placed = []
        for b in range(bands):
            slot = (i + 2 * b) % 5
            x = 2 + slot * ((width - w - 4) // 4)
            y = b * band_h + (band_h - h) // 2
            placed.append((x, y, w, h, level))
        boxes.append(placed)
    return _render(frames, rng, boxes)


def _animal_median(frames):
    found = ca.measure(frames)
    return float(np.median(found.animal)), found


def test_a_subject_a_few_pixels_across_is_measured_not_erased():
    frames = _moving((320, 240), 4, 4, 90, background=20)
    median, found = _animal_median(frames)
    assert found.n_animals == 1
    assert 0.7 * 16 <= median <= 1.3 * 16
    assert found.kernels == (0, 0)  # smoothing scaled to a 4 px subject is no smoothing


def test_a_low_contrast_subject_is_found_by_an_adaptive_threshold():
    frames = _moving((240, 160), 30, 20, 140, background=128, noise=1.5)
    median, found = _animal_median(frames)
    assert found.threshold < 12
    assert 0.85 * 600 <= median <= 1.15 * 600


def test_a_large_dark_subject_on_a_light_background_is_measured():
    frames = _moving((240, 160), 80, 60, 40, background=200)
    median, found = _animal_median(frames)
    assert found.n_animals == 1
    assert 0.9 * 4800 <= median <= 1.1 * 4800
    assert found.kernels[1] > 9  # smoothing grew with the subject


def test_two_subjects_are_both_animals_and_named_in_the_result():
    frames = _moving((240, 200), 20, 15, 200, bands=2, background=30)
    median, found = _animal_median(frames)
    assert found.n_animals == 2
    assert 0.85 * 300 <= median <= 1.15 * 300
    assert len(found.animal) == 2 * N_FRAMES
    assert not len(found.other) or np.max(found.other) < 300 / 3
    row = ca.summarize("r", "pair", N_FRAMES, found.animal, found.other, 2)
    *_, warnings = ca.suggest([row])
    assert any("More than one animal" in w and "pair (2)" in w for w in warnings)


def test_the_suggestion_scales_with_the_subject():
    def ceiling(scale):
        frames = _moving(
            (120 * scale, 90 * scale), 10 * scale, 8 * scale, 200, background=40
        )
        found = ca.measure(frames)
        row = ca.summarize("r", "r", N_FRAMES, found.animal, found.other)
        return ca.suggest([row])[1]

    small, large = ceiling(1), ceiling(3)
    assert 0.8 * 9 <= large / small <= 1.2 * 9


def test_nothing_the_user_reads_names_a_species():
    """The method and its words are for any subject; species belong in the issue's example."""
    import inspect
    import pathlib
    import re

    from detail import cli
    from detail.gui import min_area_estimate

    root = pathlib.Path(ca.__file__).resolve().parents[2]
    texts = {
        "contour_area.py": inspect.getsource(ca),
        "min_area_estimate.py": inspect.getsource(min_area_estimate),
        "the CLI command": inspect.getsource(cli.dataset_estimate_min_area.callback),
    }
    gui = (root / "docs" / "gui.md").read_text(encoding="utf-8")
    texts["docs/gui.md"] = next(
        line for line in gui.splitlines() if line.startswith("**Min contour area**")
    )
    cli_doc = (root / "docs" / "cli.md").read_text(encoding="utf-8")
    start = cli_doc.index("- `estimate-min-area")
    texts["docs/cli.md"] = cli_doc[start : cli_doc.index("\n- `", start + 1)]
    species = re.compile(
        r"\b(mouse|mice|rats?|tails?|flies|fly|zebrafish|larvae?|worms?)\b", re.I
    )
    named = {where: species.findall(text) for where, text in texts.items()}
    assert not any(named.values()), named


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
    recording = experiment.add_recording("subject a")
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
    result = ca.estimate([_video(measured_experiment, "subject a")])
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
        [_video(measured_experiment, "subject a")],
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
    assert [r.name for r in result.per_recording] == ["subject a"]
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
            "subject a",
        ],
    )
    assert out.exit_code == 0, out.output
    assert "subject a" in out.output
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
