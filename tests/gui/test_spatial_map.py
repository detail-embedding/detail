"""The spatial map — where a behavior happens, over the field of view.

Every assertion here is about what the picture *claims*, because the failures this widget can
have are all claims rather than crashes: a square that says "nowhere" when it means "not
measured", a caption that says "one dot per instance" over a density plot, or a concentration
warning that never fires on the one recording it exists to catch.

That last one is not hypothetical — it is how this file's own rule was found. The first version
pooled every recording before checking, so a plainly left-piled recording beside two uniform ones
came out at 56% and said nothing while the offender itself sat at 80%. It was visible in a
screenshot and invisible to every assertion.
"""

import pathlib

import numpy as np
import pytest

from detail.gui import spatial_map
from detail.gui.sizing import lines
from detail.gui.spatial_map import SpatialMap


def _uniform(n, seed=0):
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)))


def _left_piled(n, seed=1):
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(0.0, 0.25, n), rng.uniform(0.05, 0.95, n)))


def _placed(points, subject=0, step=1_000):
    """Coordinates as the placement dicts the panel now takes.

    The panel plays a clicked dot, so it needs identity as well as position: a dot that reported
    only where it was could not be turned back into the clip it stands for.
    """
    return [
        {
            "instance_id": f"i{i}",
            "cx": float(x),
            "cy": float(y),
            "subject": subject,
            "t_ns": step * (i + 1),
            "end_t_ns": step * (i + 2),
        }
        for i, (x, y) in enumerate(points)
    ]


class _FakeExperiment:
    """All `_ClipPlayer` touches at construction is `experiment.dir`, which it hands to a
    `ClipQueryPreview` that reads nothing until it is asked for a frame."""

    def __init__(self, tmp_path=None):
        self.dir = tmp_path or pathlib.Path(".")


def _shown(qapp, by_recording, **kw) -> SpatialMap:
    """`outlines`, `aspects` and `embed_run_ids` are required on the widget — facts about the
    recording, not choices — so the helper supplies empty ones and a test that cares overrides.
    """
    widget = SpatialMap(_FakeExperiment())
    widget.resize(900, 240)
    widget.show()
    kw.setdefault("outlines", {})
    kw.setdefault("aspects", {})
    kw.setdefault("embed_run_ids", {rid: "emb_01" for rid in by_recording})
    widget.show_instances(
        {rid: _placed(points) for rid, points in by_recording.items()}, **kw
    )
    for _ in range(5):
        qapp.processEvents()
    return widget


# --- what the picture is, and what it says it is -------------------------------------------


def test_one_square_per_recording_in_the_order_given(qapp):
    widget = _shown(
        qapp, {"rec_a": _uniform(20), "rec_b": _uniform(30), "rec_c": _uniform(10)}
    )
    assert len(widget._panels) == 3
    labels = [label.text() for _h, _s, label in widget._panels]
    assert labels[0].startswith("rec_a") and labels[2].startswith("rec_c")
    # The count is on the label, not only in a tooltip: which square is which has to be
    # readable at a glance, because square N pairs with ethogram row N above it.
    assert "30" in labels[1]
    assert "one square per recording (3)" in widget._caption.text()


def test_density_replaces_points_and_the_caption_says_which(qapp):
    widget = _shown(qapp, {"rec_a": _uniform(spatial_map.DENSITY_ABOVE + 500)})
    assert "density" in widget._caption.text()
    assert "one dot per instance" not in widget._caption.text()

    sparse = _shown(qapp, {"rec_a": _uniform(50)})
    assert "one dot per instance" in sparse._caption.text()
    assert "density" not in sparse._caption.text()


def test_no_instances_says_so_rather_than_drawing_an_empty_box(qapp):
    widget = _shown(qapp, {"rec_a": []})
    assert "nothing to place" in widget._caption.text()
    assert widget._note.text() == ""


# --- the warning, which is the reason the widget exists ------------------------------------


def test_a_lopsided_distribution_is_named_not_left_to_the_eye(qapp):
    widget = _shown(qapp, {"rec_a": _left_piled(200)})
    assert "left third" in widget._note.text()
    assert "%" in widget._note.text()


def test_one_bad_recording_is_caught_even_beside_good_ones(qapp):
    """The pooled-only version of this rule missed exactly this case.

    Two uniform recordings dilute one piled recording below any pooled threshold, and "the video
    is bad in that corner" is a fact about a *recording* — so the check runs per recording and
    names the offender.
    """
    widget = _shown(
        qapp,
        {"rec_a": _left_piled(200), "rec_b": _uniform(200), "rec_c": _uniform(200)},
    )
    note = widget._note.text()
    assert "rec_a" in note, note
    assert "left third" in note


def test_an_even_distribution_says_nothing(qapp):
    """The negative case. A warning that always fires is the same bug as one that never does."""
    widget = _shown(qapp, {"rec_a": _uniform(400), "rec_b": _uniform(400, seed=7)})
    assert widget._note.text() == ""
    assert widget._note.isHidden()


def test_too_few_points_to_judge_says_nothing(qapp):
    """With five bouts a "share" is one bout moving, and a warning would be noise."""
    widget = _shown(qapp, {"rec_a": _left_piled(5)})
    assert widget._note.text() == ""


# --- the pure helpers ------------------------------------------------------------------------


def test_concentration_names_the_worst_direction():
    piled_low = {"r": [(0.5, 0.9)] * 50}
    assert "bottom third" in spatial_map.concentration_note(piled_low)


def test_undefined_positions_are_ignored_rather_than_plotted():
    """A NaN is "not measured", and it must not become a corner."""
    points = [(np.nan, np.nan)] * 40 + [(0.1, 0.5)] * 60
    grid = spatial_map.density_grid(points, 10)
    assert grid.sum() == 60


def test_the_density_grid_is_in_screen_order():
    """Row 0 is the top of the box. Transposed, every map is mirrored and still looks plausible."""
    grid = spatial_map.density_grid([(0.05, 0.05)] * 10, 10)
    assert grid[0, 0] == 10
    assert grid[9, 9] == 0


def test_redrawing_with_fewer_recordings_drops_the_extra_squares(qapp):
    """Selecting a cluster present in fewer recordings must not leave stale squares behind."""
    widget = _shown(qapp, {"a": _uniform(10), "b": _uniform(10), "c": _uniform(10)})
    widget.show_instances(
        {"a": _placed(_uniform(10))}, outlines={}, aspects={}, embed_run_ids={"a": "e"}
    )
    for _ in range(5):
        qapp.processEvents()
    assert len(widget._panels) == 1
    assert widget._panels[0][2].text().startswith("a")


# --- the field of view, and the size of the box it is drawn in ---------------------------
def test_the_square_draws_the_field_of_view_it_is_given(qapp):
    """A dot at (0.2, 0.7) of a bare rectangle is a coordinate; over the arena it is a place.

    The widget has taken `outlines` since it was written and `docs/gui.md` has described the
    dots as drawn "over the field of view" all along — both callers passed nothing, so every
    square rendered blank. This asserts the polygon reaches the square, which is the part that
    was missing, rather than that the call returns.
    """
    arena = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    widget = _shown(qapp, {"rec_a": _uniform(20)}, outlines={"rec_a": arena})
    _holder, square, _label = widget._panels[0]
    assert square._outline == arena


def test_a_recording_with_no_arena_still_draws_its_box(qapp):
    """No arena drawn on a recording is not an error: the frame *is* the field of view, and the
    coordinates are normalized to it either way."""
    widget = _shown(qapp, {"rec_a": _uniform(20)}, outlines={})
    _holder, square, _label = widget._panels[0]
    assert square._outline == []


def test_the_squares_can_be_dragged_bigger_and_smaller(qapp):
    """The gesture the clip strip teaches one band up. Six recordings at a legible size and one
    you can read a corner bias off are different pictures, and which you want changes with the
    question — so it is the user's call, not a constant.
    """
    widget = _shown(qapp, {"rec_a": _uniform(20)})
    _holder, square, _label = widget._panels[0]
    before = square.width()

    widget.resize_by(60)
    qapp.processEvents()
    assert square.width() > before
    assert square.width() == square.height()  # a squashed arena moves every point in it


def test_the_drag_is_clamped_at_both_ends(qapp):
    """Below the floor a dot cannot be placed in the box; above the ceiling one recording fills
    the screen and the comparison this panel exists for is gone."""
    widget = _shown(qapp, {"rec_a": _uniform(20)})
    _holder, square, _label = widget._panels[0]

    for _ in range(50):
        widget.resize_by(100)
    assert square.width() == lines(spatial_map.MAX_SIDE_LINES)

    for _ in range(50):
        widget.resize_by(-100)
    assert square.width() == lines(spatial_map.MIN_SIDE_LINES)


def test_a_resize_redraws_rather_than_reloading(qapp):
    """The points are normalized, so growing a square is a repaint. If a drag went back to the
    run for data it would be a disk read per pixel of travel."""
    widget = _shown(qapp, {"rec_a": _uniform(20)})
    _holder, square, _label = widget._panels[0]
    points_before = square._points.copy()

    widget.resize_by(40)
    qapp.processEvents()
    assert (square._points == points_before).all()


def test_the_boundary_is_drawn_inside_the_canvas_not_on_its_edge(qapp):
    """The complaint this fixes: the frame read as the widget edge, so a dot at cx=0 sat on top
    of it and the box looked like a blank canvas.

    `plots.bound_axes` already gives every pyqtgraph axis this slack, for the stated reason that
    "a symbol sitting exactly on the boundary would otherwise be drawn half outside the frame".
    These squares are hand-painted and never got it.
    """
    widget = _shown(qapp, {"rec_a": _uniform(20)})
    _holder, square, _label = widget._panels[0]
    box = square._box()

    pad = lines(spatial_map.PAD_LINES)
    assert box.left() >= pad - 1  # inset from the canvas, not flush against it
    assert box.top() >= pad - 1
    assert box.right() <= square.width() - pad + 1
    assert box.width() == box.height()  # a squashed arena moves every point in it


# --- looking closer: a square opens, a dot answers -----------------------------------------


def test_a_redraw_repoints_the_squares_at_the_recordings_they_now_show(qapp):
    """The reuse this exists for: square 0 held `rec_a`, and after a redraw with a different
    cohort it holds `rec_x`. A click has to reach `rec_x`.

    Panels are reused rather than rebuilt, so the index bound into the click is stable while the
    recording behind it is not — the lookup goes through `_titles` as it stands *now*.
    """
    widget = _shown(qapp, {"rec_a": _uniform(20), "rec_b": _uniform(30)})
    widget.show_instances(
        {"rec_x": _placed(_uniform(12)), "rec_y": _placed(_uniform(9))},
        outlines={},
        aspects={},
        embed_run_ids={"rec_x": "emb_01", "rec_y": "emb_01"},
    )
    qapp.processEvents()

    picked: list[dict] = []
    widget.instance_picked.connect(picked.append)
    widget._panels[0][1].clicked.emit(0.5, 0.5)

    assert widget._titles[0] == "rec_x"
    assert picked, "the click reached no instance"
    assert widget._player._clip.recording_id == "rec_x"


def test_the_boundary_has_canvas_around_it_not_background(qapp, app_theme):
    """Read off the rendered pixels, because the geometry was already right and invisible.

    `_box()` was inset from the widget and the pen was near-black, and
    `test_the_boundary_is_drawn_inside_the_canvas_not_on_its_edge` passed on both — but the white
    fill covered only `box`, so the line landed exactly on the white/background edge. On a dark
    theme that is the edge of the white square and nothing else: the outline was drawn every
    time and could not be seen. Reported from a screenshot after the geometry check shipped.

    So the property is not "the box is inset". It is that the padding around the box is
    **canvas** — something the line can sit on — which only the image can answer. Asked in both
    chart themes: the canvas is the theme's surface and the boundary its axis ink, so the
    property is "padding is surface, and the edge stands off it", not "white, and dark".
    """
    from PyQt6.QtGui import QColor

    widget = SpatialMap(_FakeExperiment())
    widget.setAutoFillBackground(True)
    widget.resize(900, 260)
    widget.show()
    widget.show_instances(
        {"rec_a": _placed(_uniform(20))},
        outlines={},
        aspects={},
        embed_run_ids={"rec_a": "e"},
    )
    for _ in range(5):
        qapp.processEvents()

    _holder, square, _label = widget._panels[0]
    image = square.grab().toImage()
    box = square._box()
    # `grab()` renders at the screen's device pixel ratio, so on a retina display the image is
    # twice the widget's logical size and every coordinate below has to be scaled. Sampling
    # unscaled lands 2x off and reads whatever happens to be there.
    scale = image.width() / max(square.width(), 1)
    row = int(box.center().y() * scale)
    left = box.left() * scale

    surface = QColor(app_theme.surface)
    outside = image.pixelColor(int(left) - int(3 * scale), row)
    assert outside.name() == surface.name(), (
        f"the padding around the field of view is {outside.name()}, not the {app_theme.name} "
        f"canvas {surface.name()} — the boundary is drawn against the background and cannot "
        "be seen"
    )

    span = max(int(2 * scale), 2)
    edge = max(
        abs(image.pixelColor(int(left) + dx, row).lightness() - surface.lightness())
        for dx in range(-span, span + 1)
    )
    assert edge > 100, (
        f"no boundary standing off the {app_theme.name} canvas at the edge of the field of "
        f"view (lightness difference {edge})"
    )


# --- the frame's shape ----------------------------------------------------------------------


def test_the_box_is_the_frames_shape_not_a_square(qapp):
    """`cx` and `cy` are fractions of the frame's *width* and *height*, so a box drawn square
    stretches a 16:9 video by 1.78x vertically and moves every dot in it.

    Nothing about that looks wrong: the box is a box, each dot is somewhere legal, and the only
    signal is that a centred animal appears to reach the top and bottom of the arena.
    """
    widget = _shown(qapp, {"rec_a": _uniform(20)}, aspects={"rec_a": 16 / 9})
    _holder, square, _label = widget._panels[0]
    box = square._box()

    assert box.width() / box.height() == pytest.approx(16 / 9, rel=0.02)
    assert square.width() == square.height(), "the panel itself still tiles as a square"
    assert box.width() <= square.width() and box.height() <= square.height()


def test_a_recording_with_no_aspect_given_stays_square(qapp):
    """The fallback is the old behaviour, so a caller that has no resolution to offer — a
    recording with no video imported — draws what it drew before rather than nothing.
    """
    widget = _shown(qapp, {"rec_a": _uniform(20)})
    _holder, square, _label = widget._panels[0]
    box = square._box()

    assert box.width() == pytest.approx(box.height())


def test_frame_aspect_reads_the_recordings_resolution():
    """And falls back to 1.0 rather than raising, because a recording with no video imported is
    an ordinary state on a page that lists every recording in the run."""
    from detail.gui.spatial_map import frame_aspect

    class _Rec:
        def __init__(self, resolution):
            self.video_resolution = resolution

    assert frame_aspect(_Rec((640, 480))) == pytest.approx(4 / 3)
    assert frame_aspect(_Rec((1920, 1080))) == pytest.approx(16 / 9)
    assert frame_aspect(_Rec(None)) == 1.0


# --- the cue and the wiring are the same fact -----------------------------------------------


#: Inputs to the map that are genuinely a **choice** the calling surface makes, so a default is
#: right. Everything else on `show_instances` describes the *recording*, and a fact with a
#: default is a fact that can go missing on a surface nobody was looking at.
_CHOICES = {
    "colour": "the surface tints the dots however it tints things — a run colour, a cutoff",
}


def test_a_fact_about_the_recording_is_never_optional():
    """Three capabilities landed on this widget as optional parameters, and all three reached
    exactly one of the two surfaces that embed it: the field-of-view outline, the frame's
    aspect, and the click that opens a recording. Query kept rendering an unlabelled square box
    each time, and nothing failed — the default *is* the old behaviour, which is what makes the
    gap silent.

    So the rule is at the signature: an input that decides whether the picture is **correct**
    has no default, and the second surface stops compiling the day the first one gains a
    capability. An input that is a real choice is listed in `_CHOICES` with its reason, which
    is a line somebody has to write rather than a default somebody can forget.
    """
    import inspect

    offenders = []
    for owner in (SpatialMap.show_instances,):
        for name, param in inspect.signature(owner).parameters.items():
            if name in ("self", "args", "kwargs") or name in _CHOICES:
                continue
            if param.default is not inspect.Parameter.empty:
                offenders.append(f"{owner.__qualname__}({name}=...)")

    assert not offenders, (
        f"these default rather than requiring the caller to say: {offenders}. A default here "
        "means a surface that never passes it draws a plausible wrong picture and no test "
        "fails. Make it required, or add it to _CHOICES with the reason it is a choice."
    )


def test_the_choices_list_does_not_outlive_its_parameters():
    """Guard the guard: an allowlist entry for a parameter that no longer exists excuses
    nothing and hides the next one that takes its name."""
    import inspect

    known = set()
    for owner in (SpatialMap.show_instances,):
        known.update(inspect.signature(owner).parameters)

    stale = sorted(set(_CHOICES) - known)
    assert not stale, f"_CHOICES names parameters that are gone: {stale}"


def test_every_surface_that_embeds_the_map_is_covered_by_a_test():
    """Discovered, not listed — the point is that a *third* surface is covered the day it is
    added rather than the day someone remembers it.

    This is the check that would have caught all three gaps at once: it fails when a file
    builds a `SpatialMap` and no test constructs that surface, which is exactly the state
    `analysis_query.py` was in while the workspace grew three capabilities it never got.
    """
    import pathlib

    gui = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui"
    tests = pathlib.Path(__file__).resolve().parent
    embedders = {
        py.stem
        for py in gui.rglob("*.py")
        if "SpatialMap(" in py.read_text(encoding="utf-8-sig")
        and py.stem != "spatial_map"
    }
    covered = " ".join(
        py.read_text(encoding="utf-8-sig") for py in tests.rglob("test_*.py")
    )

    # Guard the guard: a discovery that finds nothing passes vacuously forever, and the whole
    # point is that it keeps finding the *next* surface.
    # One today: the clustering page draws its maps with the node card's own density squares
    # (§22) and no longer embeds SpatialMap. The behavior-scan page still does.
    assert (
        "analysis_query" in embedders
    ), f"the sweep stopped finding embedders: {embedders}"

    missing = sorted(stem for stem in embedders if stem not in covered)
    assert not missing, (
        f"these surfaces embed the spatial map and no test names them: {missing}. A shared "
        "widget's capabilities reach one surface at a time unless every surface is built."
    )


# --- a hundred recordings ---------------------------------------------------------------------


def test_a_click_plays_the_nearest_instance_rather_than_needing_a_hit(qapp):
    """A dot is a few pixels wide at the default square size and they overlap, so requiring a
    hit would make most of the row unclickable.

    Nearest also makes the rule the same in density mode, where there are no dots to hit at
    all — one behaviour to learn rather than two, and no state where clicking does nothing.
    """
    widget = _shown(qapp, {"rec_a": [(0.1, 0.1), (0.9, 0.9)]})
    picked: list[dict] = []
    widget.instance_picked.connect(picked.append)

    widget._panels[0][1].clicked.emit(0.8, 0.75)  # near the second, on neither

    assert picked and picked[0]["cx"] == 0.9
    assert widget._player._clip is not None, "nothing was played"
    assert int(widget._player._clip.start_t_ns) == int(picked[0]["t_ns"])


def test_a_hundred_recordings_scroll_rather_than_pooling_into_one_box(qapp):
    """The panel used to overlay every square into one above six recordings — so at the scale a
    real experiment runs at, the comparison it exists for was gone and the caption said so.

    A hundred squares is 16,500px, which is what the scroll area is for. Asserted on the panel
    count, because "the caption mentions a hundred" would pass over a single pooled box.
    """
    widget = _shown(qapp, {f"rec_{i:03d}": _uniform(5, seed=i) for i in range(100)})

    assert len(widget._panels) == 100
    assert widget._titles[0] == "rec_000" and widget._titles[-1] == "rec_099"
    assert widget._scroll.horizontalScrollBar().maximum() > 0, "the row does not scroll"


def test_the_row_scrolls_sideways_and_leaves_the_page_its_wheel(qapp):
    """`PageScroll` gives up the wheel to any descendant scroll area that has somewhere to go —
    it asks the *vertical* bar's range, so a row that scrolls horizontally must keep its
    vertical bar off or the page can no longer be scrolled through the map.
    """
    from PyQt6.QtCore import Qt

    widget = _shown(qapp, {f"rec_{i:03d}": _uniform(5, seed=i) for i in range(100)})

    assert (
        widget._scroll.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    assert widget._scroll.verticalScrollBar().maximum() == 0


def test_the_caption_describes_the_square_in_front_of_you_not_the_pool(qapp):
    """The painting decides density per square (`len(self._points)`); the caption asked the
    **pooled** total. At a hundred recordings of fifty instances each that said "density" over
    squares drawing fifty dots apiece — a caption describing a picture that was not on screen.
    """
    many_small = _shown(
        qapp, {f"rec_{i:02d}": _uniform(60, seed=i) for i in range(40)}
    )  # 2,400 pooled, 60 per square

    assert "density" not in many_small._caption.text()
    assert "one dot per instance" in many_small._caption.text()


def test_one_dense_recording_still_says_density(qapp):
    """Guard the guard: a caption that never says "density" would pass the test above."""
    dense = _shown(qapp, {"rec_a": _uniform(spatial_map.DENSITY_ABOVE + 50)})

    assert "density" in dense._caption.text()
    assert (
        "nearest instance" in dense._caption.text()
    ), "it does not say what a click does"


# --- saying how long the thing you clicked lasted ----------------------------------------------


def _one(cx, cy, t_ns, end_t_ns):
    return [
        {
            "instance_id": "i0",
            "cx": cx,
            "cy": cy,
            "subject": 0,
            "t_ns": t_ns,
            "end_t_ns": end_t_ns,
        }
    ]


def test_a_played_clip_says_which_recording_where_and_how_long(qapp):
    """An instance is a *run* of frames, and the run is a single frame about as often as it is
    three seconds — so the same click produced a still one time and a second of footage the
    next, and the two looked like different bugs rather than two lengths of the same thing.
    """
    widget = _shown(qapp, {"rec_a": []})
    widget.show_instances(
        {"rec_a": _one(0.5, 0.5, 17_000_000_000, 17_990_000_000)},
        outlines={},
        aspects={},
        embed_run_ids={"rec_a": "emb_01"},
    )
    widget._panels[0][1].clicked.emit(0.5, 0.5)

    said = widget._played.text()
    assert "rec_a" in said
    assert "0:00:17" in said, f"it does not say where in the recording: {said!r}"
    assert "0.99 s" in said, f"it does not say how long: {said!r}"


def test_a_single_frame_instance_says_so_rather_than_reading_0_of_0(qapp):
    """The reported confusion, at its root.

    `pipeline/pose_clustering.py` (since removed) built `end_t_ns` as `t_ns[end]`, so a one-row
    run had **zero** duration. The player was handed a zero-length span and its frame counter read
    `0 / 0` — which does not read as "one frame", it reads as broken, and it is why some dots
    seemed to play and others seemed to do nothing.
    """
    widget = _shown(qapp, {"rec_a": []})
    widget.show_instances(
        {"rec_a": _one(0.5, 0.5, 17_000_000_000, 17_000_000_000)},  # start == end
        outlines={},
        aspects={},
        embed_run_ids={"rec_a": "emb_01"},
    )
    widget._panels[0][1].clicked.emit(0.5, 0.5)

    said = widget._played.text()
    assert "single frame" in said, f"a one-frame instance is unexplained: {said!r}"
    assert " s (" not in said, f"it claims a duration it does not have: {said!r}"


def test_the_caption_says_what_an_instance_is(qapp):
    """Said once, where the picture is. Without it the varying clip lengths read as the surface
    behaving inconsistently rather than as the data being uneven."""
    widget = _shown(qapp, {"rec_a": _uniform(20)})

    assert "run of frames" in widget._caption.text()


def test_redrawing_clears_the_last_clip_it_named(qapp):
    """The label is about the *last click*, so a redraw must not leave it describing an instance
    from a cutoff or granularity nobody is looking at any more."""
    widget = _shown(qapp, {"rec_a": []})
    widget.show_instances(
        {"rec_a": _one(0.5, 0.5, 1_000_000_000, 2_000_000_000)},
        outlines={},
        aspects={},
        embed_run_ids={"rec_a": "emb_01"},
    )
    widget._panels[0][1].clicked.emit(0.5, 0.5)
    assert widget._played.text()

    widget.show_instances(
        {"rec_a": _placed(_uniform(5))},
        outlines={},
        aspects={},
        embed_run_ids={"rec_a": "emb_01"},
    )

    assert widget._played.text() == ""
