"""The pure half of a recording visualization (§22 "Recording visualization"): the layout rule,
the type sizes, the montage grid, the tree cut and the frame colours."""

import numpy as np
import pytest

from detail.analysis import recording_visualization as rv

ASPECTS = {"linear track": 800 / 176, "16:9 arena": 16 / 9, "square field": 1.0}
TILES = {"full screen": (1920, 1080), "2 across": (960, 1080), "3x3": (640, 360)}


def _inside(box: rv.Box, w: int, h: int) -> bool:
    return (
        box.x >= -0.5
        and box.y >= -0.5
        and box.x + box.w <= w + 0.5
        and box.y + box.h <= h + 0.5
    )


def _overlap(a: rv.Box, b: rv.Box) -> bool:
    return not (
        a.x + a.w <= b.x + 0.5
        or b.x + b.w <= a.x + 0.5
        or a.y + a.h <= b.y + 0.5
        or b.y + b.h <= a.y + 0.5
    )


@pytest.mark.parametrize("aspect", list(ASPECTS.values()), ids=list(ASPECTS))
@pytest.mark.parametrize("tile", list(TILES.values()), ids=list(TILES))
@pytest.mark.parametrize("location", [True, False])
def test_layout_keeps_every_panel_inside_the_tile_and_apart(aspect, tile, location):
    w, h = tile
    lay = rv.layout_for(w, h, aspect, location)
    panels = [lay.video, lay.ethogram, lay.now, lay.outline, lay.pose]
    if location:
        assert lay.location is not None
        panels.append(lay.location)
    else:
        assert lay.location is None
    for box in panels:
        assert _inside(box, w, h), (box, tile)
        assert box.w > 0 and box.h > 0
    # The video keeps its shape, and nothing is drawn on it.
    assert lay.video.w / lay.video.h == pytest.approx(aspect, rel=1e-6)
    for box in panels[1:]:
        assert not _overlap(lay.video, box), (box, lay.video)
    assert not _overlap(lay.pose, lay.outline)
    if location:
        assert not _overlap(lay.pose, lay.location)


@pytest.mark.parametrize("tile", list(TILES.values()), ids=list(TILES))
def test_layout_gives_the_video_the_larger_arrangement(tile):
    """A wide strip spans the tile's width (stack); a square video spans its height on a wide tile
    (side) — whichever leaves the video more area."""
    w, h = tile
    strip = rv.layout_for(w, h, ASPECTS["linear track"], True)
    assert strip.mode == "stack"
    square = rv.layout_for(w, h, ASPECTS["square field"], True)
    if w / h > 1.2:
        assert square.mode == "side"
    for lay in (strip, square):
        free = (h - lay.video.h) if lay.mode == "stack" else (w - lay.video.w)
        total = h if lay.mode == "stack" else w
        assert free >= rv.MIN_PANEL_SHARE * total - 1


def test_type_scales_with_the_tile_and_never_falls_under_its_floor():
    big = rv.font_sizes(1920, 1080)
    small = rv.font_sizes(640, 360)
    assert big["behavior"] == pytest.approx(rv.FONT_PX["behavior"])
    for key, floor in rv.FONT_FLOOR_PX.items():
        assert small[key] >= floor
        assert big[key] >= small[key]


def test_a_bad_tile_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        rv.layout_for(0, 1080, 1.0, True)


@pytest.mark.parametrize("n", range(1, rv.MAX_MONTAGE + 1))
def test_montage_tiles_fill_the_screen_in_even_pixels(n):
    tiles = rv.montage_tiles(n, 1920, 1080)
    assert len(tiles) == n
    cols, rows = rv.montage_grid(n)
    for t in tiles:
        assert t.w == 1920 // cols // 2 * 2 and t.h == 1080 // rows // 2 * 2
        assert int(t.w) % 2 == 0 and int(t.h) % 2 == 0
    assert len({(t.x, t.y) for t in tiles}) == n


def test_a_montage_of_more_than_nine_is_refused():
    with pytest.raises(ValueError, match="1 to 9"):
        rv.montage_grid(10)


def _tree(heights_by_merge, children):
    return np.asarray(children, dtype=np.int32), np.asarray(
        heights_by_merge, dtype=float
    )


def test_display_heights_lift_a_parent_to_its_children():
    # leaves 0..3; merge 0 -> node 4 (0,1) at 0.5; merge 1 -> node 5 (2,3) at 0.2;
    # merge 2 -> node 6 (4,5) at 0.3: below its child 4, as a transition boundary allows.
    children, height = _tree([0.5, 0.2, 0.3], [[0, 1], [2, 3], [4, 5]])
    shown = rv.display_heights(children, height, 4)
    assert shown[6] == pytest.approx(0.5)
    for m, (a, b) in enumerate(children):
        assert shown[4 + m] >= shown[a] and shown[4 + m] >= shown[b]


@pytest.mark.parametrize("seed", range(5))
def test_a_cut_is_always_a_partition_of_the_leaves(seed):
    rng = np.random.default_rng(seed)
    n = 30
    active = list(range(n))
    children, height = [], []
    for m in range(n - 1):
        a, b = rng.choice(len(active), 2, replace=False)
        x, y = active[a], active[b]
        for i in sorted((a, b), reverse=True):
            active.pop(i)
        children.append([x, y])
        height.append(float(rng.random()))  # deliberately non-monotone
        active.append(n + m)
    children, height = np.asarray(children), np.asarray(height)
    for cut in (0.05, 0.3, 0.7, 2.0):
        nodes = rv.cut_nodes(children, height, n, cut)
        leaves = []
        for node in nodes:
            stack = [node]
            while stack:
                cur = stack.pop()
                if cur < n:
                    leaves.append(cur)
                else:
                    stack.extend(children[cur - n])
        assert sorted(leaves) == list(range(n))
    assert rv.cut_nodes(children, height, n, 10.0) == [2 * n - 2]


def test_a_cut_numbers_clusters_largest_first():
    children, height = _tree([0.1, 0.2, 0.9], [[0, 1], [4, 2], [5, 3]])
    nodes = rv.cut_nodes(children, height, 4, 0.5)
    assert nodes == [5, 3]  # three leaves, then one


def test_label_raster_colours_spans_and_marks_gaps():
    t_ns = np.arange(10) * 10
    spans = [
        {"start_ns": 0, "end_ns": 30, "node": 7},
        {"start_ns": 50, "end_ns": 70, "node": 9},
    ]
    embedded = np.ones(10, bool)
    embedded[8] = False
    codes = rv.label_raster(spans, t_ns, {7: 0}, embedded)
    assert list(codes[:4]) == [0, 0, 0, 0]
    assert codes[5] == rv.GREY  # node 9 is not coloured
    assert codes[8] == rv.NO_POSE


def test_colours_come_from_the_chart_theme_per_mode():
    from detail.core import chart_theme

    assert rv.colour_of("light", 0) == chart_theme.LIGHT.identity(0)
    assert rv.colour_of("dark", 2) == chart_theme.DARK.identity(2)
    assert rv.colour_of("dark", rv.GREY) == rv.THEMES["dark"]["grey"]
    assert rv.colour_of("light", rv.NO_POSE) == rv.THEMES["light"]["gap"]


@pytest.mark.parametrize(
    "extent, bar", [(360.7, 100.0), (80.0, 20.0), (5.0, 1.0), (1000.0, 200.0)]
)
def test_scale_bar_is_a_round_length(extent, bar):
    assert rv.nice_scale_bar(extent) == bar
