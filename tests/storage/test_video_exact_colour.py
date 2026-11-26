"""The writer's exact-colour path and the packet join (§22 "Recording visualization").

**The class:** a picture meant to sit on a slide's background must decode to the background's
exact value. swscale's default bgr24 -> yuv420p path left pure white at (251, 253, 250), a faint
box on a white slide; ``exact_colour`` writes H.264 through PyAV and converts to BT.709 limited
range itself. Off by default, because the preview cache encodes through this writer and changing
every caller's conversion would mix previews made before and after (CLAUDE.md, *Expensive derived
data*) — so the default path is pinned here too.
"""

import av
import cv2
import numpy as np
import pytest

from detail.storage import video
from detail.storage.video import concat_mp4, open_video_writer

needs_pyav_h264 = pytest.mark.skipif(
    not video._pyav_h264(), reason="this PyAV build has no H.264 encoder"
)


def _write(path, colours, *, exact, w=64, h=48, frames_each=4):
    writer, _codec = open_video_writer(path, 25.0, w, h, exact_colour=exact)
    for rgb in colours:
        for _ in range(frames_each):
            writer.write(np.full((h, w, 3), rgb[::-1], np.uint8))
    writer.release()


def _decoded(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


@needs_pyav_h264
@pytest.mark.parametrize("rgb", [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_exact_colour_decodes_neutral_greys_exactly(tmp_path, rgb):
    path = tmp_path / "exact.mp4"
    _write(path, [rgb], exact=True)
    for frame in _decoded(path):
        assert tuple(int(v) for v in frame[24, 32]) == rgb


@needs_pyav_h264
def test_exact_colour_tags_the_stream_bt709_limited(tmp_path):
    path = tmp_path / "exact.mp4"
    _write(path, [(255, 255, 255)], exact=True)
    with av.open(str(path)) as c:
        cc = c.streams.video[0].codec_context
        assert (int(cc.colorspace), int(cc.color_range)) == (1, 1)


def test_the_default_path_is_unchanged(tmp_path):
    """Pins the previews' encode: without the flag every caller still gets cv2's writer."""
    writer, codec = open_video_writer(tmp_path / "default.mp4", 25.0, 64, 48)
    writer.release()
    assert isinstance(writer, cv2.VideoWriter) and codec in ("avc1", "mp4v")


def test_exact_colour_falls_back_where_pyav_has_no_h264(tmp_path, monkeypatch):
    """No libx264 in PyAV: the export still writes, through cv2, rather than failing."""
    monkeypatch.setattr(video, "_pyav_h264", lambda: False)
    writer, _codec = open_video_writer(
        tmp_path / "fallback.mp4", 25.0, 64, 48, exact_colour=True
    )
    writer.release()
    assert isinstance(writer, cv2.VideoWriter)


@needs_pyav_h264
def test_joined_parts_play_through_without_a_gap(tmp_path):
    parts = []
    for i, rgb in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        part = tmp_path / f"part_{i}.mp4"
        _write(part, [rgb], exact=True, frames_each=10)
        parts.append(part)
    out = concat_mp4(parts, tmp_path / "joined.mp4")
    frames = _decoded(out)
    assert len(frames) == 30
    with av.open(str(out)) as c:
        assert float(c.duration / 1e6) == pytest.approx(30 / 25.0, abs=0.05)
    assert frames[5][24, 32].argmax() == 0 and frames[15][24, 32].argmax() == 1


@needs_pyav_h264
def test_joining_nothing_or_mismatched_parts_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no parts"):
        concat_mp4([], tmp_path / "x.mp4")
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _write(a, [(0, 0, 0)], exact=True)
    _write(b, [(0, 0, 0)], exact=True, w=32, h=32)
    with pytest.raises(ValueError, match="parts must match"):
        concat_mp4([a, b], tmp_path / "y.mp4")
