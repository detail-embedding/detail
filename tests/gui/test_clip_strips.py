"""The strips under the clip window's slider (``gui/clip_strips.py``): one thin row per curve
the API hands them, an untracked lane, the reading under the pointer — and nothing read or
named by the widget itself. The scan beside this file (``test_clip_strips_scan.py``) is the
guard for that last part; these are the behaviour.
"""

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from detail.api.clip_curves import ClipCurves, Curve  # noqa: E402
from detail.gui.clip_strips import ClipStrips  # noqa: E402
from detail.gui.scrubber import FrameScrubber  # noqa: E402

#: A strip is a line of fine print, not the 30px row it replaced.
_ROW_MAX = 20


def _curves():
    frames = np.arange(10, 15)
    return ClipCurves(
        window=(10, 14),
        frames=frames,
        t_ns=(frames * 1e9 / 30).astype(np.int64),
        curves=[
            Curve(
                "speed",
                "Speed",
                "px/s",
                np.array([1.0, 2.0, np.nan, 4.0, 3.0]),
                floor_zero=True,
            ),
            Curve(
                "bend",
                "Body bend",
                "°",
                np.array([-10.0, 0.0, np.nan, 20.0, 5.0]),
                signed=True,
            ),
        ],
        untracked=[(12, 12)],
        note="at the embedding cadence",
    )


@pytest.fixture
def strips(qapp):
    host = QWidget()
    layout = QVBoxLayout(host)
    scrubber = FrameScrubber(playable=False)
    scrubber.set_total(5)
    widget = ClipStrips(scrubber)
    scrubber.set_underlay(widget)
    layout.addWidget(scrubber)
    host.resize(640, 240)
    host.show()
    QApplication.processEvents()
    widget._host = host  # keep it alive
    return widget


def test_one_thin_row_per_curve_and_an_untracked_lane(strips):
    strips.set_curves(_curves())
    rows = strips._rows()
    assert [kind for kind, _top, _h in rows] == ["curve:0", "curve:1", "untracked"]
    assert strips.names() == ["speed", "bend"]
    assert max(h for kind, _top, h in rows if kind.startswith("curve:")) <= _ROW_MAX
    assert strips.height() <= 3 * (_ROW_MAX + 2) + 4


def test_the_reading_under_the_pointer_names_the_curve_its_unit_and_the_cadence(
    strips,
):
    strips.set_curves(_curves())
    left, span = strips._travel()
    rows = strips._rows()
    _kind, top, height = rows[0]
    tip = strips.describe(left + span * 0.25, top + height / 2)  # frame 11
    assert tip.startswith("Speed: 2.00 px/s") and "embedding cadence" in tip
    _kind, top, height = rows[1]
    assert "not tracked here" in strips.describe(left + span * 0.5, top + 1)  # frame 12
    _kind, top, height = rows[2]
    assert "Not tracked" in strips.describe(left + span * 0.5, top + 1)
    assert strips.describe(left, top + 1) == "Tracked."


def test_an_empty_read_shows_its_note_where_the_rows_would_be(strips):
    empty = ClipCurves(
        window=(0, 4),
        frames=np.arange(5),
        t_ns=np.zeros(5, np.int64),
        curves=[],
        untracked=[],
        note="no kinematics for emb_01 — run detail recording backfill-kinematics",
    )
    strips.set_curves(empty)
    assert strips._rows() == [] and strips.names() == []
    assert "backfill-kinematics" in strips.describe(10, 3)
    strips.set_message("No strips: boom")
    assert strips.curves() is None and strips.describe(10, 3) == "No strips: boom"


def test_the_playhead_and_the_clip_are_drawn(strips, tmp_path):
    strips.set_curves(_curves())
    strips.set_clip_span((1, 3))
    strips.set_frame(2)
    shot = strips.grab().toImage()
    pixels = {
        shot.pixel(x, y)
        for x in range(0, shot.width(), 7)
        for y in range(0, shot.height(), 3)
    }
    assert len(pixels) > 2, "the rows, the clip's shade and the playhead are drawn"
