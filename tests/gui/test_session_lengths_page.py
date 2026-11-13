"""Time in session on the real page, over twelve recordings from 8 s to 120 min (§22).

``tests/api/test_session_lengths.py`` holds the arithmetic; this drags the slider with the mouse on
the INTERPRET page of the same fit (``session_lengths``) and reads what a user reads: the
"n of N clips match" line, the slider's own clip count, and which montage tiles wear the match
frame. Each expected answer is computed here from the clips' times and the spans the fixture's maps
were written with, in exact fractions, never through ``cluster_filter``.
"""

from fractions import Fraction

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest


def _drain(qapp, predicate, timeout_ms: int = 20_000) -> bool:
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


def _at(slider, fraction: float) -> QPoint:
    """The point ``fraction`` of the way along the track, set in by half a knob."""
    inset = slider.KNOB_W_PX / 2
    x = inset + fraction * (slider.width() - 2 * inset)
    return QPoint(int(round(x)), slider.height() // 2)


def _drag(widget, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(widget, end)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=end)


def _expected(clips, spans, lo, hi) -> list[bool]:
    """Each clip's midpoint inside ``[lo, hi]`` of its own recording's whole video."""
    lo, hi = Fraction(str(lo)), Fraction(str(hi))
    out = []
    for i in range(len(clips.start_ns)):
        first, last = spans[clips.recording_ids[int(clips.recording[i])]]
        mid = (int(clips.start_ns[i]) + int(clips.end_ns[i])) // 2
        out.append(lo <= Fraction(mid - first, last - first) <= hi)
    return out


@pytest.fixture
def page(qapp, session_lengths, monkeypatch):
    from detail.gui import preview_jobs
    from detail.gui.cluster_workspace import ClusterWorkspace

    experiment, run_id, layout = session_lengths
    # The application's preview queue, as the shell installs one per experiment and shuts it
    # down on quitting. The page queues this run's twelve proxies on it (real videos, up to
    # 120 min), which this test never waits for; left to the page's own queue they built for
    # ~24 s after the test ended, past the teardown's wait under a loaded run.
    queue = preview_jobs.PreviewQueue(experiment)
    monkeypatch.setattr(preview_jobs, "current", lambda: queue)
    workspace = ClusterWorkspace(experiment, experiment.temporal_clustering_run(run_id))
    workspace.resize(1500, 1000)
    workspace.show()
    interpret = workspace._interpret
    assert _drain(
        qapp,
        lambda: interpret._view is not None and interpret._root_clips is not None,
    ), "INTERPRET never loaded"
    interpret.select(interpret._view.root)
    assert _drain(qapp, lambda: len(interpret.montage()._clips) > 0)
    row = interpret.filters().field_row("Time in session")
    row.set_open(True)
    qapp.processEvents()
    yield interpret, layout
    workspace.close()
    queue.shutdown()


def test_dragging_time_in_session_counts_and_frames_what_each_recording_s_range_holds(
    qapp, page
):
    """The ends, a sliver at each end and the middle, picked by dragging the handles: the line
    counts the clips whose midpoint is in each recording's own range, across all twelve
    recordings, and the root's montage frames exactly those."""
    interpret, layout = page
    filters = interpret.filters()
    slider = filters.session.slider
    assert slider.width() > 200, "the slider must be laid out for the handles to land"
    assert (
        interpret._spans == layout.spans
    ), "the page read every recording's whole span"
    root = interpret._root_clips
    view = interpret._view
    node_clips = view.clips(view.root)

    def check(lo: float, hi: float) -> int:
        assert filters.selection().session == (lo, hi)
        expected = _expected(root, layout.spans, lo, hi)
        matched = sum(expected)
        assert 0 < matched < len(expected), (lo, hi)
        assert filters.count_label.text() == (
            f"{matched:,} of {len(expected):,} clips match"
        )
        assert filters.session.span_label.text().endswith(
            f" · {matched:,} clip{'s' if matched != 1 else ''}"
        )
        flags = _expected(node_clips, layout.spans, lo, hi)
        order = interpret.montage_order()
        tiles = interpret.montage()._clips
        assert len(tiles) == len(order)
        assert [t.matching for t in tiles] == [flags[i] for i in order]
        return matched

    # The left end: the first 1% of every recording.
    _drag(slider, _at(slider, 1.0), _at(slider, 0.01))
    first = check(0.0, 0.01)
    # The right end: the last 1%.
    _drag(slider, _at(slider, 0.01), _at(slider, 1.0))
    _drag(slider, _at(slider, 0.0), _at(slider, 0.99))
    last = check(0.99, 1.0)
    # The middle tenth.
    _drag(slider, _at(slider, 0.99), _at(slider, 0.45))
    _drag(slider, _at(slider, 1.0), _at(slider, 0.55))
    middle = check(0.45, 0.55)
    assert (first, last, middle) == tuple(
        sum(_expected(root, layout.spans, lo, hi))
        for lo, hi in ((0.0, 0.01), (0.99, 1.0), (0.45, 0.55))
    )
