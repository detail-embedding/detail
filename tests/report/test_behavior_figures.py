"""The pictures a result screen shows, as files (report/behavior_figures.py).

**The class these guard.** *A figure that renders perfectly and says nothing.* Every failure here
is invisible to a test that checks a figure came back: a lane drawn only to its last bout, a
behavior that is blue in one figure and green in the next, a heavy-tailed axis that puts a
thousand bouts in one pixel. So each test names the property that makes the picture readable and
asserts *that*, on the artists the figure actually contains.

Qt-free by construction — these are matplotlib figures, which is what lets the CLI produce them
on a machine with no display.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from detail.report import behavior_figures as bf  # noqa: E402
from detail.storage import behavior_output as bo  # noqa: E402

_NS = 1_000_000_000


def _instance(recording, name, start_s, end_s, source="c01"):
    return bo.BehaviorInstance(
        recording,
        0,
        int(start_s * _NS),
        int(end_s * _NS),
        int(start_s * 30),
        int(end_s * 30),
        source,
        name,
    )


def _coverage(recording, seconds=60.0):
    return bo.CoverageRow(recording, 0, 0, int(seconds * _NS), int(seconds * _NS), 1800)


def _bars(ax):
    """Every block drawn in the lanes — the lane tracks and the bouts — as ``(x_min, x_max)``."""
    return [
        (float(path.vertices[:, 0].min()), float(path.vertices[:, 0].max()))
        for collection in ax.collections
        for path in collection.get_paths()
    ]


# --------------------------------------------------------------------------- #
# the ethogram
# --------------------------------------------------------------------------- #
def test_the_lane_spans_the_observed_time_not_the_last_bout():
    """A recording that stops behaving is not a recording that stopped.

    Drawing a lane only as far as its last bout silently shortens it, and the reader has no way
    to tell a quiet animal from a truncated recording — the two look identical.
    """
    rows = [_instance("r1", "rearing", 1.0, 2.0)]
    fig = bf.ethogram(rows, [_coverage("r1", seconds=60.0)])
    try:
        ax = fig.axes[0]
        extents = [high for _, high in _bars(ax)]
        assert (
            max(extents) >= 59.0
        ), f"the lane stops at {max(extents):.1f}s but the recording observed 60s"
    finally:
        plt.close(fig)


def test_every_recording_gets_a_lane_even_with_no_bouts():
    """An animal that did nothing is a finding; a missing row is a bug that looks like one."""
    rows = [_instance("r1", "rearing", 1.0, 2.0)]
    fig = bf.ethogram(rows, [_coverage("r1"), _coverage("r2")])
    try:
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["r2", "r1"]
    finally:
        plt.close(fig)


def test_a_behavior_keeps_its_colour_across_figures():
    """One behavior, one colour, in every picture of the same export.

    The colours come from a sorted assignment rather than first-seen order, because first-seen
    depends on which bout happens to come first — so recording B's figure would colour
    "rearing" differently from recording A's, and the two could not be laid side by side.
    """
    a = [_instance("r1", "rearing", 0.0, 1.0), _instance("r1", "walking", 2.0, 3.0)]
    b = [_instance("r2", "walking", 0.0, 1.0), _instance("r2", "rearing", 2.0, 3.0)]
    assert bf._label_colors({r.label for r in a}) == bf._label_colors(
        {r.label for r in b}
    )


def test_the_axis_is_seconds_not_nanoseconds():
    """Every artifact time is `t_ns` (Tenet 8) and no reader thinks in nanoseconds."""
    fig = bf.ethogram([_instance("r1", "rearing", 1.0, 2.0)], [_coverage("r1", 60.0)])
    try:
        ax = fig.axes[0]
        assert "s" in ax.get_xlabel()
        assert (
            max(high for _, high in _bars(ax)) < 1000
        ), "the axis is in nanoseconds, so every bout is at the same coordinate"
    finally:
        plt.close(fig)


def test_a_large_vocabulary_draws_no_legend():
    """A 60-cluster key would be taller than the figure and tells you nothing you can read."""
    rows = [
        _instance("r1", f"b{i:02d}", i, i + 0.5, source=f"c{i:02d}") for i in range(20)
    ]
    fig = bf.ethogram(rows, [_coverage("r1")])
    try:
        assert fig.axes[0].get_legend() is None
    finally:
        plt.close(fig)

    small = [_instance("r1", "rearing", 0.0, 1.0)]
    fig = bf.ethogram(small, [_coverage("r1")])
    try:
        assert fig.axes[0].get_legend() is not None, "a small key is worth drawing"
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# time per behavior
# --------------------------------------------------------------------------- #
def test_time_is_summed_over_bouts_not_rows():
    """A multi-label bout is one span and several rows; summing rows double-counts its time.

    `behavior_output.bouts` is the shared answer to this, and a figure that re-derives it
    would be a second implementation that disagrees with the numbers on screen.
    """
    duration = 4.0
    one_bout = _instance("r1", "rearing", 0.0, duration)
    twice = [
        one_bout,
        _instance("r1", "grooming", 0.0, duration),
    ]  # same span, two labels
    fig = bf.time_by_behavior(twice)
    try:
        widths = [patch.get_width() for patch in fig.axes[0].patches]
        assert (
            max(widths) <= duration + 0.01
        ), f"a {duration}s span was counted as {max(widths)}s"
    finally:
        plt.close(fig)


def test_the_longest_behavior_is_drawn_first():
    """An unordered bar chart makes the reader do the sorting."""
    rows = [
        _instance("r1", "brief", 0.0, 1.0, source="c00"),
        _instance("r1", "long", 2.0, 12.0, source="c01"),
    ]
    fig = bf.time_by_behavior(rows)
    try:
        fig.canvas.draw()
        # Where each label *is* on the rendered figure, not its index in the tick list: an
        # inverted axis lists the top label first, and tick order is not reading order.
        by_height = sorted(
            fig.axes[0].get_yticklabels(),
            key=lambda t: t.get_window_extent().y0,
        )
        labels = [t.get_text() for t in by_height]
        assert labels[-1] == "long", f"the longest behavior is not at the top: {labels}"
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# bout durations
# --------------------------------------------------------------------------- #
def test_a_heavy_tail_gets_a_log_axis():
    """Otherwise a thousand half-second bouts land in one pixel beside a 30s outlier."""
    rows = [_instance("r1", "rearing", i, i + 0.2) for i in range(10)]
    rows.append(_instance("r1", "rearing", 100.0, 160.0))
    fig = bf.bout_durations(rows)
    try:
        assert fig.axes[0].get_xscale() == "log"
    finally:
        plt.close(fig)


def test_a_narrow_spread_stays_linear():
    """A log axis on a 2x spread buys nothing and crowds "0.9" against "1.0"."""
    rows = [_instance("r1", "rearing", i, i + 0.5 + 0.1 * (i % 3)) for i in range(8)]
    fig = bf.bout_durations(rows)
    try:
        assert fig.axes[0].get_xscale() == "linear"
    finally:
        plt.close(fig)


def test_the_log_axis_is_labelled_in_plain_numbers():
    """`2 x 10^-1` has to be decoded before a duration can be judged.

    Rendered rather than inspected: the formatter object is not the label, and a formatter that
    is set but never reached would pass an `isinstance` check.
    """
    rows = [_instance("r1", "rearing", i, i + 0.2) for i in range(10)]
    rows.append(_instance("r1", "rearing", 100.0, 160.0))
    fig = bf.bout_durations(rows)
    try:
        fig.canvas.draw()
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels() if t.get_text()]
        assert labels, "the axis drew no labels at all"
        assert not any(
            "10" in text and "$" in text for text in labels
        ), f"the durations are labelled in scientific notation: {labels}"
    finally:
        plt.close(fig)


def test_bout_durations_are_seconds():
    rows = [_instance("r1", "rearing", 0.0, 2.0)]
    fig = bf.bout_durations(rows)
    try:
        ax = fig.axes[0]
        assert "s" in ax.get_xlabel()
        offsets = ax.collections[0].get_offsets()
        assert abs(offsets[0][0] - 2.0) < 0.01, f"a 2s bout plotted at {offsets[0][0]}"
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# both formats, from one figure
# --------------------------------------------------------------------------- #
def test_a_figure_saves_as_png_and_svg(tmp_path):
    """PNG to look at, SVG for a paper. The SVG is the one a reader cannot tell is missing."""
    from detail.report import figures

    fig = bf.ethogram([_instance("r1", "rearing", 0.0, 1.0)], [_coverage("r1")])
    written = figures.save_figure(fig, tmp_path / "ethogram", formats=("png", "svg"))
    assert sorted(p.name for p in written) == ["ethogram.png", "ethogram.svg"]
    assert (
        (tmp_path / "ethogram.svg").read_text(encoding="utf-8-sig").startswith("<?xml")
    )
    assert (tmp_path / "ethogram.png").stat().st_size > 1000
