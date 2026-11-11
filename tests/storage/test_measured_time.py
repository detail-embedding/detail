"""A video part's duration comes from its measured timestamps, never from advertised fps.

`generate_timestamps` demuxes real PTS precisely so `t_ns` is measured (Tenet 8). Computing
a part's *duration* as `len(t_ns) / container_fps` throws that away at the last step and
re-introduces the error the measurement exists to remove — the same fallacy Tenet 8 already
forbids in its mirror image (`CAP_PROP_FRAME_COUNT` = duration x advertised fps).

It matters because a part's duration is what the *next* part's offset is computed from, so
the error accumulates down the recording's global time base. A container's advertised
fps can differ from its true PTS rate by several percent, and the error is paid per part.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import detail
from detail.storage import recording
from detail.storage.video import NS_PER_SECOND, part_duration_ns

# A true 50 fps part, described by a container that claims 62.5.
_TRUE_STEP_NS = NS_PER_SECOND // 50
_LYING_FPS = 62.5


def _part(n_frames: int, start_ns: int = 0) -> np.ndarray:
    return (start_ns + np.arange(n_frames) * _TRUE_STEP_NS).astype(np.int64)


# --------------------------------------------------------------------------- #
# the measured helper
# --------------------------------------------------------------------------- #
def test_duration_is_the_measured_span_not_the_advertised_rate():
    t_ns = _part(50)  # 50 frames at a true 50 fps == exactly 1 second
    assert part_duration_ns(t_ns, _LYING_FPS) == NS_PER_SECOND
    # what the old expression gave: 50 / 62.5 = 0.8 s, a 20% under-report
    assert part_duration_ns(t_ns, _LYING_FPS) != round(
        len(t_ns) / _LYING_FPS * NS_PER_SECOND
    )


def test_duration_includes_the_last_frame_s_interval():
    """The span runs first-frame-start to last-frame-start; a part lasts one interval
    longer, or two parts butted together would overlap by a frame."""
    t_ns = _part(3)
    span = int(t_ns[-1]) - int(t_ns[0])
    assert part_duration_ns(t_ns, _LYING_FPS) == span + _TRUE_STEP_NS


def test_duration_of_degenerate_parts():
    assert part_duration_ns(_part(0), 50.0) == 0  # no frames, no duration
    # one frame has no interval to measure — the only place the advertised rate survives
    assert part_duration_ns(_part(1), 50.0) == _TRUE_STEP_NS


# --------------------------------------------------------------------------- #
# through the real reorder path, on real files
# --------------------------------------------------------------------------- #
def _lying_recording(experiment_dir):
    """Two parts whose timestamps are a true 50 fps while `fps_actual` claims 62.5 —
    the shape of a real file whose header disagrees with its PTS."""
    sdir = recording.create_recording(experiment_dir, "s")
    ts_dir = sdir / "video" / "timestamps" / "generated"
    ts_dir.mkdir(parents=True, exist_ok=True)
    recording.write_timestamps_csv(ts_dir / "part_001.csv", _part(100))
    recording.write_timestamps_csv(
        ts_dir / "part_002.csv", _part(50, start_ns=2 * NS_PER_SECOND)
    )
    data = recording.read_recording(sdir)
    data["video"] = {
        "storage": "default",
        "fps_container": _LYING_FPS,
        "fps_actual": _LYING_FPS,
        "fps_source": "container",
        "resolution": [10, 10],
        "total_duration_s": 3.0,
        "files": [
            {
                "index": i,
                "file": f"video/part_{i + 1:03d}.mp4",
                "timestamps": {
                    "generated": f"video/timestamps/generated/part_{i + 1:03d}.csv",
                    "user": None,
                    "active": "generated",
                },
            }
            for i in range(2)
        ],
    }
    recording.write_recording(sdir, data)
    return sdir, ts_dir


def test_reorder_offsets_come_from_measured_spans(experiment_dir):
    """Reorder rewrites every part's timestamps on disk, so using the advertised rate here
    *degrades* a recording whose stored timestamps were correct."""
    sdir, ts_dir = _lying_recording(experiment_dir)

    recording.reorder_video_parts(sdir, [1, 0])  # 50-frame part first

    moved_first = recording.TimestampMap.from_csv(ts_dir / "part_002.csv").values
    moved_second = recording.TimestampMap.from_csv(ts_dir / "part_001.csv").values
    assert moved_first[0] == 0
    # the 50-frame part really lasts 1.0 s, so the 100-frame part starts there —
    # the advertised rate would have put it at 50/62.5 = 0.8 s
    assert moved_second[0] == NS_PER_SECOND
    assert moved_second[0] != round(50 / _LYING_FPS * NS_PER_SECOND)
    # ...and the moved part keeps its own measured cadence
    assert int(moved_second[1] - moved_second[0]) == _TRUE_STEP_NS


def test_reorder_is_lossless_when_the_header_is_honest(experiment_dir):
    """The correcting case must not disturb a recording whose fps really is its rate:
    reorder twice and the timestamps come back to where they started."""
    sdir, ts_dir = _lying_recording(experiment_dir)
    data = recording.read_recording(sdir)
    data["video"]["fps_actual"] = 50.0  # honest header
    recording.write_recording(sdir, data)
    before = recording.TimestampMap.from_csv(ts_dir / "part_001.csv").values.copy()

    recording.reorder_video_parts(sdir, [1, 0])
    recording.reorder_video_parts(sdir, [1, 0])

    after = recording.TimestampMap.from_csv(ts_dir / "part_001.csv").values
    assert list(after) == list(before)


# --------------------------------------------------------------------------- #
# the guard: this arithmetic must not come back
# --------------------------------------------------------------------------- #
_BAD = re.compile(r"len\([^)]*\)\s*/\s*[\w.]*fps")

# Sites where dividing by an advertised rate is correct, with the reason. Nothing is
# listed today; an entry here needs a comment saying why the measurement is unavailable.
_ALLOWED: set[tuple[str, int]] = set()


def _sources():
    root = Path(detail.__file__).parent
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_duration_is_computed_from_an_advertised_fps():
    offenders = []
    for path in _sources():
        for n, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if _BAD.search(line) and (path.name, n) not in _ALLOWED:
                offenders.append(
                    f"{path.relative_to(Path(detail.__file__).parent)}:{n}"
                )
    assert not offenders, (
        "a frame count divided by an advertised fps re-introduces the error the PTS map "
        f"exists to remove, and a part's duration sets the next part's offset: {offenders}. "
        "Use storage.video.part_duration_ns, which measures the span."
    )


def test_the_scan_would_catch_the_original():
    """Guard the guard — a scan that matches nothing passes vacuously forever."""
    assert _BAD.search("        duration_s = len(t_ns) / eff_fps")
    assert _BAD.search("        durations_s.append(len(values) / fps)")
    assert not _BAD.search("        out.append(int(round(i / fps * NS_PER_SECOND)))")


@pytest.mark.parametrize("n", [2, 7, 1000])
def test_measured_duration_round_trips_a_constant_rate(n):
    """n frames at a constant true rate last exactly n intervals."""
    assert part_duration_ns(_part(n), _LYING_FPS) == n * _TRUE_STEP_NS
