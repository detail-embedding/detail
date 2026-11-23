"""The Measurements band: a ranked scan of what a cluster is, with the top few drawn.

Every failure below was found by opening the app on a real run and looking at the picture. None
of them fails an assertion that the widget was built.

**A panel with no answer in it.** The first version drew all nine measurements, equally, in a row
each. On a real run one or two measurements carry most of the between-cluster variance — so the
other rows buried the ones worth reading, and the reported experience was "the layout just
isn't working".
The band ranks now, and the guard is that the *drawn* set follows the ranking rather than the
registry: permute the values and different measurements must be drawn.

**Two computations of one thing, disagreeing.** Twice, in one afternoon:

- the drill-in's cluster lane took its axis range from the run's values while the plot beneath it
  pooled the run *with* the members, so the two "identical" ranges came out different and
  the plots drifted apart across the axis — a tick no longer over its own data, both plots
  rendering perfectly. The lane now takes the strip's own range.
- the table marked three rows as drawn while two were drawn, because "what is drawn" was derived
  in both places. One list now feeds both.

**An axis nobody drew.** Nine plots with `hideAxis("bottom")` stack nine different
percentile-clipped scales with nothing on screen to say what any of them is. Reported as "I'm
confused by the x axis for all the plots".

**A player holding a third of the width with nothing in it**, from the moment the tab opened —
which is why the plots were too thin to read. It arrives with something to play.

**Numbers kept under ids the run had moved past (#515).** The class: *a view that caches what it
derived from a run under the run's id, while an authored edit changes what the run states and
leaves the id alone.* The band measured on the first click and kept every row under the cluster
id it had then. A granularity change or a name moves those ids, so a cluster new to the cut drew
the run-wide spread over zero members, which looks finished. A committed cutoff did the
same to a behavior scan's matches.

How it hid: every click was fast and every picture was well-formed, because the cache was
consistent with itself — only not with the run. And the obvious repair rests on a premise that
is false one level up: "a granularity change only relabels" holds for ``dtw_finch`` alone, since
``finch`` and ``codebook_viterbi`` re-encode their runs at every cut.

The rule: what is kept is keyed on what it was read **from** (``Measured`` carries the spans and
groups of every artifact behind it, the posture sources' included) and checked before each draw;
when any of them moved, the band clears and says so. Measuring itself waits for a button, because
it is a minute on a real cohort and the vocabulary is still moving while you interpret. The
end-to-end guards drive the real edit paths — ``test_cluster_workspace.py``'s granularity and name,
``test_query_workspace.py``'s cutoff — and the one below that could not be seen from here: after
an edit the workspace re-selected the row that was already selected, which emits nothing, so the
whole pane went on describing a cluster the cut no longer had.
"""

import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from detail.analysis.descriptor_ranking import STRONG  # noqa: E402
from detail.gui.measurements import (  # noqa: E402
    BASELINE_INSTANCES,
    BASELINE_RECORDING,
    TABLE_HEADERS,
    DRAWN,
    WEAK_DRAWN,
    MeasurementsPanel,
)
from detail.gui.plots import (
    DENSITY_ABOVE,
    DescriptorStrip,
    fmt_value,
    swarm,
)  # noqa: E402
from detail.gui.qt import QPointF, QPushButton, Qt

#: Three real descriptor names, so the panel's registry lookups resolve.
SEPARATED, MIDDLE, FLAT = "bbox_aspect", "tortuosity", "duration_s"


def _shown(name: str) -> str:
    from detail.analysis.descriptors import DESCRIPTORS

    return DESCRIPTORS[name].label


def _instance(recording: str, at: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        recording_id=recording, subject=0, t_ns=at, end_t_ns=at + 1_000_000_000
    )


@pytest.fixture
def panel(qapp, tmp_path) -> MeasurementsPanel:
    """A real experiment behind it: the band carries a clip player, and a player with nothing to
    resolve a recording against cannot be built."""
    from detail.api.experiment import Experiment

    return MeasurementsPanel(
        Experiment.create("p", tmp_path),
        "bouts",
        "behavior",
        key="test/measurements",
        baseline=BASELINE_INSTANCES,
    )


def _cohort(n_groups: int = 4, per_group: int = 30, seed: int = 0):
    """A run where one measurement separates the groups cleanly, one weakly, one not at all."""
    rng = np.random.default_rng(seed)
    labels = [str(g) for g in range(n_groups) for _ in range(per_group)]
    run = {
        SEPARATED: np.concatenate(
            [rng.normal(g * 10.0, 0.2, per_group) for g in range(n_groups)]
        ).tolist(),
        MIDDLE: np.concatenate(
            [rng.normal(g * 0.6, 1.0, per_group) for g in range(n_groups)]
        ).tolist(),
        FLAT: rng.normal(0, 1, n_groups * per_group).tolist(),
    }
    return run, labels


def _members(run: dict, labels: list, group: str):
    rows = [i for i, label in enumerate(labels) if label == group]
    members = {name: [values[i] for i in rows] for name, values in run.items()}
    ids = {name: [_instance("rec", i * 1_000_000_000) for i in rows] for name in run}
    return members, ids


def _show(panel, group: str = "0", **kwargs):
    run, labels = _cohort(**kwargs)
    members, ids = _members(run, labels, group)
    panel._selected = group
    panel.show_measurements(run, members, ids, labels)
    return run, labels


# --- the ranking is what gets drawn ---------------------------------------------------- #
def test_the_drawn_measurements_follow_the_ranking(panel):
    """The guard for the whole redesign, and it has to be driven by *permuting the data*.

    Asserting "three rows were drawn" passes against a panel that draws the first three in
    registry order — which is exactly what it used to do. So the fixture is built with a
    different measurement separating the groups, and the drawn set must follow it.
    """
    _show(panel)
    assert [name for name, _row in panel._rows][0] == SEPARATED

    # Same panel, same registry, one measurement swapped for another: the drawn set must move.
    run, labels = _cohort()
    run[SEPARATED], run[FLAT] = run[FLAT], run[SEPARATED]
    members, ids = _members(run, labels, "0")
    panel.show_measurements(run, members, ids, labels)
    assert [name for name, _row in panel._rows][0] == FLAT


def _column(panel, column: int) -> list:
    model = panel._table.model()
    return [model.index(row, column).data() for row in range(model.rowCount())]


def test_the_table_lists_every_measurement_and_marks_only_the_drawn(panel):
    """The table is the scan and the plots are the shortlist. A table listing only what is drawn
    cannot say "and it is ordinary on everything else", which is half the description.
    """
    run, _labels = _show(panel)
    assert set(_column(panel, 0)) == {_shown(name) for name in run}
    assert sum(1 for cell in _column(panel, 5) if cell) == len(panel._rows) < len(run)


def test_the_table_and_the_plots_agree_about_what_is_drawn(panel):
    """They disagreed: three ticks over two plots, because each derived the set for itself.

    **Every branch, and the no-strong one especially.** The two computations agree wherever the
    cluster has strong measurements — they only diverge where the drawn count changes, which is
    exactly the case a cohort of well-separated groups never reaches. Written over strong groups
    alone, this guard passed against its own reverted fix.
    """

    def marked_and_drawn():
        marked = {
            name for name, mark in zip(_column(panel, 0), _column(panel, 5)) if mark
        }
        return marked, {_shown(name) for name, _row in panel._rows}

    for group in ("0", "1", "2", "3"):
        _show(panel, group=group)
        assert any(r.strong for r in panel._ranked), "this half needs a strong effect"
        marked, drawn = marked_and_drawn()
        assert marked == drawn, f"group {group}"

    # The branch they actually differed on: nothing strong, so fewer plots are drawn.
    rng = np.random.default_rng(1)
    labels = [str(g) for g in range(4) for _ in range(30)]
    run = {name: rng.normal(0, 1, 120).tolist() for name in (SEPARATED, MIDDLE, FLAT)}
    members, ids = _members(run, labels, "0")
    panel._selected = "0"
    panel.show_measurements(run, members, ids, labels)
    assert not any(r.strong for r in panel._ranked), "this half needs no strong effect"
    marked, drawn = marked_and_drawn()
    assert marked == drawn, "nothing strong"

    panel._toggle_all()
    marked, drawn = marked_and_drawn()
    assert marked == drawn, "drawing all"


def test_a_cluster_distinctive_on_nothing_says_so(panel):
    """A state real runs have: no measurement separates the cluster.

    Drawing nine flat rows and no sentence is a page that failed to load; the state has an
    answer of its own."""
    rng = np.random.default_rng(1)
    labels = [str(g) for g in range(4) for _ in range(30)]
    run = {name: rng.normal(0, 1, 120).tolist() for name in (SEPARATED, MIDDLE, FLAT)}
    members, ids = _members(run, labels, "0")
    panel._selected = "0"
    panel.show_measurements(run, members, ids, labels)

    assert not any(
        r.strong for r in panel._ranked
    ), "the fixture must have no strong effect"
    assert not panel._note.isHidden()
    assert "No measurement separates" in panel._note.text()
    assert str(STRONG) in panel._note.text(), "the threshold that decided it"
    assert len(panel._rows) == WEAK_DRAWN
    assert "none of them strong" in panel._caption.text()


def test_one_group_with_nothing_else_measured_draws_nothing(panel):
    """The case that stays degenerate: one behavior and no background, so the grey spread and the
    dark marks are the same data — a distribution centred on itself."""
    run, _labels = _cohort(n_groups=1, per_group=40)
    labels = ["only"] * len(next(iter(run.values())))
    members, ids = _members(run, labels, "only")
    panel._selected = "only"
    panel.show_measurements(run, members, ids, labels)

    assert panel._ranked == []
    assert panel._rows == [], "nothing may be drawn against itself"
    assert not panel._note.isHidden()
    assert "one behavior" in panel._note.text()


def test_one_group_against_background_still_ranks(panel):
    """The reported failure and its fix. Four of five real query runs have a single behavior, so
    the band withheld everything and drew nothing — which is also why there was nothing to click.

    Background rows carry no group: in the comparison, out of the partition. The behavior ranks
    against the recording while the *run line* stays honestly absent.
    """
    run, _labels = _cohort(n_groups=1, per_group=40)
    n = len(next(iter(run.values())))
    labels = ["q01" if i < n // 4 else None for i in range(n)]
    members, ids = _members(run, labels, "q01")
    panel._selected = "q01"
    panel.show_measurements(run, members, ids, labels)

    assert panel._ranked, "the behavior must rank against the recording"
    assert panel._rows, "and something must be drawn"
    assert panel._run_ranked == [], "one behavior is still no partition"
    assert "one behavior" in panel._run_line.text()
    assert "rest of the recording" in panel._run_line.text()


def test_the_run_line_does_not_move_with_the_selection(panel):
    """It is a property of the partition. A line that changed per cluster would be read as a
    fact about the cluster."""
    _show(panel, group="0")
    first = panel._run_line.text()
    _show(panel, group="2")
    assert panel._run_line.text() == first
    assert "separate them most" in first


# --- drawing more ---------------------------------------------------------------------- #
def test_drawing_all_is_reversible(panel):
    run, _labels = _show(panel)
    assert len(panel._rows) <= DRAWN
    panel._toggle_all()
    assert len(panel._rows) == len(run)
    assert sum(1 for cell in _column(panel, 5) if cell) == len(run)
    panel._toggle_all()
    assert len(panel._rows) <= DRAWN


def test_the_controls_look_like_the_rest_of_the_gui(panel):
    """Reported: "it doesn't look like any of the other buttons". `setFlat` appeared nowhere else
    in `detail/gui/` — the one control that mattered was styled as text, by me, and nothing in the
    package agreed with it."""
    _show(panel)
    for button in panel.findChildren(QPushButton):
        assert (
            not button.isFlat()
        ), f"{button.text()!r} is flat and nothing else in the GUI is"
    assert panel._more.text() and not panel._more.isHidden()


def test_a_table_row_opens_that_measurement(panel):
    """The discoverability fix: every row is an entry point, with the package's own cue for it.

    Driven through `doubleClicked` — the signal the view emits — not by calling the handler,
    which would prove the handler works and nothing about whether a user reaches it.
    """
    _show(panel)
    row = _column(panel, 0).index(_shown(FLAT))
    panel._table.doubleClicked.emit(panel._table.model().index(row, 0))
    assert panel._open == FLAT
    assert [name for name, _r in panel._rows] == [FLAT]


def test_the_table_says_its_rows_are_clickable(panel):
    """`_table(navigable=True)` is the shared cue — pointing hand plus a trailing "›" column —
    and it is set only where a double-click is wired, which here it is."""
    _show(panel)
    assert panel._table.model().columnCount() == len(TABLE_HEADERS) + 1
    # The cue sits on the viewport, which is what the pointer is actually over.
    assert panel._table.viewport().cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_every_column_drags_normally_while_the_name_takes_the_slack(panel, qapp):
    """Handing the slack to the name column first relocked every other column to
    `ResizeToContents` (#463); then, with `Stretch` on the name, a border dragged right stayed
    put and its column grew leftward — reported from this table on a clustering run. The page
    sweep in `test_table_sorting` builds no populated band and never saw either; this fills one,
    shows it, and drags each border with the mouse."""
    from tests.gui.layout_checks import assert_border_follows_the_drag, locked_columns

    _show(panel)
    panel.resize(1200, 900)
    panel.show()
    qapp.processEvents()
    table = panel._table
    header = table.horizontalHeader()
    assert locked_columns(table) == []
    widest = max(range(header.count()), key=header.sectionSize)
    assert widest == 0, f"column {widest} took the slack, not the measurement name"
    for column in range(len(TABLE_HEADERS)):
        assert_border_follows_the_drag(table, column, dx=30)


# --- the drill-in ------------------------------------------------------------------------ #
def test_the_drill_in_lane_and_plot_share_one_x_axis(panel, qapp):
    """Two plots stacked on one quantity must map a value to the same place, or the tick marking
    a group's median stops sitting over its own data — which is what happened, by 15px, because
    each computed "the same" robust range from a different pool.

    Shown and resized first: an unlaid-out plot reports its default viewport and every position
    below comes out equal for the wrong reason.
    """
    _show(panel)
    panel.resize(900, 600)
    panel.show()
    qapp.processEvents()
    panel._open_measurement(SEPARATED)
    qapp.processEvents()

    import pyqtgraph as pg

    plots = panel._plots.findChildren(pg.PlotWidget)
    assert len(plots) == 2, "a lane and a plot"

    def at(plot, value):
        view = plot.getViewBox()
        return plot.mapTo(
            panel, plot.mapFromScene(view.mapViewToScene(QPointF(value, 0.0)))
        ).x()

    values = np.asarray(panel._values[SEPARATED], dtype=float)
    for value in np.percentile(values, [5, 50, 95]):
        positions = [at(plot, float(value)) for plot in plots]
        assert max(positions) - min(positions) <= 1.0, f"drifted apart at {value}"


def test_the_drill_in_returns_to_the_scan(panel):
    _show(panel)
    drawn_before = [name for name, _row in panel._rows]
    panel._open_measurement(SEPARATED)
    assert panel._table is None, "the scan's table is gone, not merely hidden"
    assert not panel._back.isHidden()
    panel._open_measurement(None)
    assert [name for name, _row in panel._rows] == drawn_before
    assert panel._back.isHidden()


# --- the player ------------------------------------------------------------------------- #
def test_the_player_is_absent_until_something_is_picked(panel):
    """It used to hold a third of the width from the moment the tab opened, which is why the
    plots were too thin to read."""
    _show(panel)
    assert panel._player is None
    assert panel._split.count() == 1

    panel._on_picked(SEPARATED, 0)
    assert panel._player is not None
    assert panel._split.count() == 2


def test_clicking_a_mark_names_the_member_behind_it(panel):
    """A point you can see is odd is only useful if you can then look at it."""
    run, labels = _cohort()
    members, ids = _members(run, labels, "1")
    seen = []
    panel.instance_picked.connect(seen.append)
    panel._selected = "1"
    panel.show_measurements(run, members, ids, labels)

    panel._on_picked(SEPARATED, 3)
    assert seen == [ids[SEPARATED][3]]
    assert ids[SEPARATED][3].recording_id in panel._played.text()
    assert not panel._played.isHidden()


def test_a_pick_out_of_range_is_ignored(panel):
    _show(panel)
    panel._on_picked(SEPARATED, 10_000)
    assert panel._player is None


# --- the strip ---------------------------------------------------------------------------- #
def test_the_strip_draws_its_axis(qapp):
    """The reported bug. An axis carries a quantity; a hidden one carries nothing, and nine of
    them stacked carry nine invisible scales.

    Shown first, and asserted on the space the axis *reserves*: an ``AxisItem`` is a graphics
    item, so it has no ``isHidden``, and its ``isVisible`` is false for a plot nobody has shown —
    which would pass this test against a strip that hides the axis, for the wrong reason.
    """
    strip = DescriptorStrip("speed", "px/frame")
    strip.show_values([1.0, 2.0, 3.0], [2.0])
    strip.resize(400, 120)
    strip.show()
    qapp.processEvents()

    axis = strip.plot.getAxis("bottom")
    assert axis.height() > 0, "a hidden axis reserves nothing"
    assert "speed" in axis.labelText and "px/frame" in axis.labelText

    from detail.gui.plots import cluster_lane

    lane = cluster_lane([1.0, 2.0], 1.5, 0.0, 3.0)
    lane.resize(400, 34)
    lane.show()
    qapp.processEvents()
    assert lane.getAxis("bottom").height() == 0, "the lane borrows the strip's axis"


def test_the_background_is_the_run_and_the_marks_are_the_selection(qapp):
    strip = DescriptorStrip("speed", "px/frame")
    strip.show_values(list(np.linspace(0.0, 100.0, 500)), [1.0, 2.0, 3.0])
    assert len(strip._points.getData()[0]) == 3
    assert strip.x_range[1] > 50, "the axis must span the run, not just the members"


def test_a_skewed_quantity_stays_readable(qapp):
    """A thousand bouts under a second and ten over a minute: on a full-range axis every one of
    the thousand lands in the first pixel."""
    strip = DescriptorStrip("duration", "s")
    strip.show_values([0.5] * 1000 + [60.0] * 10, [0.5, 0.6])
    assert (
        strip.x_range[1] < 10.0
    ), "the long tail set the axis and squashed everything else"


def test_a_value_past_the_range_sits_on_the_end(qapp):
    strip = DescriptorStrip("duration", "s")
    strip.show_values([0.5] * 1000 + [60.0] * 10, [0.5, 500.0])
    assert np.asarray(strip._points.getData()[0]).max() <= strip.x_range[1] + 1e-6


def test_an_undefined_value_is_not_drawn_at_zero(qapp):
    """NaN means the descriptor could not be computed; a mark on the baseline would say it was
    computed and came out low."""
    strip = DescriptorStrip("speed", "px/frame")
    strip.show_values([1.0, 2.0, 3.0], [2.0, float("nan")])
    assert len(strip._points.getData()[0]) == 1


def test_every_value_identical_gives_an_axis_with_an_inside(qapp):
    strip = DescriptorStrip("aspect", "h/w")
    strip.show_values([2.0] * 20, [2.0] * 5)
    assert strip.x_range[1] > strip.x_range[0]


def test_nothing_to_draw_is_not_a_crash(qapp):
    strip = DescriptorStrip("speed", "px/frame")
    strip.show_values([], [])
    strip.show_values([float("nan")], [float("nan")])


def test_many_members_become_a_density_and_few_stay_clickable(qapp):
    """Dots are worth having because you can click one; past the threshold they are a solid rule
    and clicking a smear is not a gesture."""
    sparse = DescriptorStrip("speed", "px/frame")
    sparse.show_values(list(np.linspace(0, 10, 200)), list(np.linspace(0, 10, 20)))
    assert not sparse.is_dense and sparse._points is not None

    dense = DescriptorStrip("speed", "px/frame")
    dense.show_values(
        list(np.linspace(0, 10, 2000)), list(np.linspace(0, 10, DENSITY_ABOVE + 1))
    )
    assert dense.is_dense
    assert (
        dense._points is None
    ), "a density has no points to click, and must not pretend to"


def test_the_caption_says_which_picture_it_is(panel):
    """ "each mark is one" and "taller means more" are different claims about the same marks."""
    _show(panel, per_group=20)
    assert "each mark is one" in panel._legend.text()

    _show(panel, per_group=DENSITY_ABOVE + 20, n_groups=2)
    assert "taller means more" in panel._legend.text()


def test_the_swarm_stacks_rather_than_overwriting(qapp):
    """Height carries count. On one line, thirty members and three hundred are one picture."""
    piled = swarm(np.zeros(20), 0.0, 1.0)
    assert len(np.unique(piled)) > 1, "twenty members at one value drew on one line"
    spread = swarm(np.linspace(0.0, 1.0, 20), 0.0, 1.0)
    assert len(np.unique(spread)) == 1, "members at distinct values share a baseline"


def test_dragging_the_band_changes_the_row_height_and_remembers(panel):
    """The house answer to "these are too skinny": a handle, not a bigger constant."""
    _show(panel)
    row = panel._rows[0][1]
    before = row.plot.maximumHeight()
    panel.resize_by(40)
    assert row.plot.maximumHeight() > before

    from detail.gui.sizing import remembered_size

    assert remembered_size("test/measurements", 0) == panel._row_px


def test_a_row_is_a_range_not_a_fixed_height(qapp):
    """One number is a floor and a ceiling at once, so a short pane crops the row instead of
    squeezing it — and what a plot loses off the bottom is its axis."""
    strip = DescriptorStrip("speed", "px/frame")
    strip.set_row_height(120)
    assert strip.plot.minimumHeight() < strip.plot.maximumHeight()


def test_a_unit_is_written_the_way_that_unit_is_written():
    assert fmt_value(0.4123, "/s") == "0.412/s"
    assert fmt_value(3.0, "count") == "3"
    assert fmt_value(1.8, "px/frame") == "1.8 px/frame"
    # A range and a formula are not units: `0.15 0–1` is what one format string produces.
    assert fmt_value(0.15, "0–1") == "0.15"
    assert fmt_value(0.15, "1-cos") == "0.15"
    assert fmt_value(float("nan"), "s") == "—"


def test_the_baseline_has_no_default(qapp, tmp_path):
    """A shared widget's input that decides whether the output is *correct* has no default: a
    default spelling "as before" is where a missing fact hides, and the other surface goes on
    rendering a plausible wrong picture with nothing failing."""
    import inspect

    from detail.api.experiment import Experiment

    parameter = inspect.signature(MeasurementsPanel.__init__).parameters["baseline"]
    assert parameter.default is inspect.Parameter.empty

    with pytest.raises(ValueError, match="unknown baseline"):
        MeasurementsPanel(
            Experiment.create("p", tmp_path),
            "bouts",
            "behavior",
            key="k",
            baseline="whatever",
        )


def test_every_embedder_states_its_baseline():
    """Discovered rather than listed, so a third surface is covered the day it is added."""
    import ast
    import pathlib

    import detail.gui

    root = pathlib.Path(detail.gui.__file__).parent
    seen = {}
    for py in sorted(root.rglob("*.py")):
        if py.name == "measurements.py":
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8-sig"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "MeasurementsPanel"
            ):
                seen[py.name] = {kw.arg for kw in node.keywords}
    assert seen, "no embedder found — the scan is looking in the wrong place"
    for name, keywords in seen.items():
        assert (
            "baseline" in keywords
        ), f"{name} builds the band without saying what it compares against"


def test_a_descriptor_that_measures_the_span_is_dropped_against_chosen_spans(
    tmp_path, qapp
):
    """A background window is as long as we made it, so comparing `duration` against one compares
    a measurement with a parameter. It read `duration 1.07 s · above 100%` — a fact about the
    sampler wearing the clothes of a finding, and the only number on the panel that was certain
    to be meaningless.

    Declared on the descriptor (`measures_span`), so a second one added later is dropped without
    editing the panel; asserted here through the registry rather than against a literal, so the
    guard follows the declaration.
    """
    from detail.analysis.descriptors import DESCRIPTORS

    span_measures = {n for n, d in DESCRIPTORS.items() if d.measures_span}
    assert span_measures, "the flag must be set on something, or this passes vacuously"

    class _Run:
        id = "r"

        def state(self):
            return type("S", (), {"embed_run_id": None})()

        def artifact(self):
            return type("A", (), {"instances": []})()

    from detail.api.experiment import Experiment
    from detail.gui.measurements import measure_run

    experiment = Experiment.create("p", tmp_path)
    for baseline in (BASELINE_INSTANCES, BASELINE_RECORDING):
        assert measure_run(_Run(), experiment, baseline=baseline).rows == {}

    # And the panel keeps them under the instance baseline, so the drop is about the *pool*.
    panel_kept = MeasurementsPanel(
        experiment, "bouts", "behavior", key="k1", baseline=BASELINE_INSTANCES
    )
    run, labels = _cohort()
    members, ids = _members(run, labels, "0")
    panel_kept._selected = "0"
    panel_kept.show_measurements(run, members, ids, labels)
    assert span_measures & {r.name for r in panel_kept._ranked}


def _run_with_instances(lengths_s):
    """A run whose artifact holds one instance per length, all on one subject."""
    from types import SimpleNamespace

    instances = [
        SimpleNamespace(
            recording_id="recording_001",
            subject=0,
            t_ns=int(i * 10e9),
            end_t_ns=int(i * 10e9 + s * 1e9),
            source_label="0",
        )
        for i, s in enumerate(lengths_s)
    ]

    class _Run:
        id = "r"

        def state(self):
            return SimpleNamespace(embed_run_id=None)

        def artifact(self):
            return SimpleNamespace(instances=instances)

    return _Run()


class _Trajectory:
    """Just enough of a `FrameSlice` for `measure_run`: a length and its timestamps."""

    def __init__(self, seconds: int = 100) -> None:
        self.t_ns = np.arange(0, seconds * 10**9, 10**8, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.t_ns)


def test_the_span_drop_is_reached_by_a_run_that_has_something_to_measure(monkeypatch):
    """The test above measures a run with **no** instances, which returns ``{}`` with or without
    the drop — so it stayed green with the drop deleted. This one reaches the loop: real
    instances, a trajectory to measure them on, and every descriptor answering.
    """
    import detail.analysis.descriptors as descriptors
    import detail.api.frame_slices as frame_slices
    import detail.gui.measurements as measurements

    span_measures = {n for n, d in descriptors.DESCRIPTORS.items() if d.measures_span}
    assert span_measures, "the flag must be set on something, or this passes vacuously"
    everything = list(descriptors.DESCRIPTORS)
    monkeypatch.setattr(measurements, "_load_slice", lambda *a: (_Trajectory(), None))
    # No experiment behind the run, so no pose run to lend postures.
    monkeypatch.setattr(measurements, "_posture_candidates", lambda *a: ({}, {}))
    monkeypatch.setattr(frame_slices, "between", lambda t, s, e: (s, e))
    monkeypatch.setattr(descriptors, "available", lambda bout, **k: everything)
    monkeypatch.setattr(
        descriptors, "describe", lambda bout, names=None, **k: {n: 1.0 for n in names}
    )

    run = _run_with_instances([1, 2, 3])
    kept = measurements.measure_run(run, None, baseline=BASELINE_INSTANCES).rows
    assert span_measures <= set(kept), "the instance baseline keeps the span measures"
    dropped = measurements.measure_run(run, None, baseline=BASELINE_RECORDING).rows
    assert (
        dropped
    ), "the fixture must reach the measuring loop, or this passes vacuously"
    assert not span_measures & set(dropped), (
        f"{sorted(span_measures & set(dropped))} compared against windows whose length we "
        "chose — a measurement set against a parameter"
    )


def test_background_windows_take_the_instances_own_lengths():
    """Not their median. One length for every window makes a length-sensitive measure a fact
    about the sampler, and nothing called `_baseline_spans` until this test, so a pool drawn
    at one fixed length passed everything.
    """
    from detail.gui.measurements import _baseline_spans, _span_lengths_ns

    lengths = _span_lengths_ns(_run_with_instances([1, 2, 5]).artifact().instances)
    assert sorted(lengths.tolist()) == [10**9, 2 * 10**9, 5 * 10**9]
    trajectory = _Trajectory()
    last = int(trajectory.t_ns[-1])
    # A window running past the recording's end is clipped there, so it takes no drawn length.
    drawn = {
        end - start for start, end in _baseline_spans(trajectory, lengths) if end < last
    }
    assert drawn <= set(lengths.tolist()), "a window took a length no instance has"
    assert len(drawn) > 1, (
        "every background window has one length, so a length-sensitive measure compares "
        "against the sampler's setting rather than the run"
    )


def test_a_run_with_nothing_measurable_names_the_fix(panel):
    """A fixable state, not a blank panel — the same distinction `SpatialMap.show_unavailable`
    makes, and the same remedy named."""
    panel.show_measurements({}, {}, {}, [])
    assert panel._rows == []
    assert "backfill-kinematics" in panel._note.text()


# --- #515: measured when asked, and never drawn past the run it was measured from -------- #
def _bout(label, at: int = 0, recording: str = "rec", subject: int = 0):
    from types import SimpleNamespace

    return SimpleNamespace(
        recording_id=recording,
        subject=subject,
        t_ns=at,
        end_t_ns=at + 1_000_000_000,
        source_label=label,
    )


class _Artifacts:
    """A run and an experiment whose artifacts a test can move under the panel."""

    def __init__(self, run_instances, postures=None):
        from types import SimpleNamespace

        self.run = SimpleNamespace(
            id="an_01",
            artifact=lambda: SimpleNamespace(instances=list(self.run_instances)),
        )
        self.run_instances = list(run_instances)
        self.postures = dict(postures or {})

    def authored_run(self, run_id):
        from types import SimpleNamespace

        if run_id not in self.postures:
            return None
        return SimpleNamespace(
            artifact=lambda: SimpleNamespace(instances=list(self.postures[run_id]))
        )


def _measured_under(world):
    from detail.gui.measurements import Measured, artifact_spans

    return Measured(
        rows={},
        spans=artifact_spans(world.run.artifact()),
        posture_runs={
            run_id: artifact_spans(world.authored_run(run_id).artifact())
            for run_id in world.postures
        },
        embed_run_ids={},
    )


def test_an_artifact_that_still_states_what_was_measured_is_current():
    from detail.gui.measurements import what_changed

    world = _Artifacts([_bout(1), _bout(2, at=2_000_000_000)], {"an_02": [_bout(7)]})
    assert what_changed(_measured_under(world), world.run, world) is None


@pytest.mark.parametrize(
    "edit",
    [
        # The issue's case: a granularity change or a name moves the ids, the spans stay put.
        pytest.param(lambda i: [_bout(9, at=b.t_ns) for b in i], id="relabel"),
        # Two of the three methods re-encode their runs at a new cut: the same ids, new spans.
        pytest.param(
            lambda i: [_bout(b.source_label, at=b.t_ns + 1) for b in i], id="respan"
        ),
        # A name fuses consecutive bouts: fewer instances, one of them longer.
        pytest.param(lambda i: i[:1], id="fuse"),
        # A lowered cutoff keeps more matches.
        pytest.param(lambda i: i + [_bout(1, at=9_000_000_000)], id="more-matches"),
    ],
)
def test_any_edit_to_what_the_run_states_is_seen(edit):
    """Each half alone misses an edit: the ids survive a granularity change on the clusters
    present at both cuts, and the spans survive a name that fuses nothing."""
    from detail.gui.measurements import RUN_CHANGED, what_changed

    world = _Artifacts([_bout(1), _bout(2, at=2_000_000_000)])
    measured = _measured_under(world)
    world.run_instances = edit(world.run_instances)
    assert what_changed(measured, world.run, world) == RUN_CHANGED


def test_a_pose_run_the_postures_come_from_is_checked_too():
    """The posture measurements are joined from another run's frame views, so that run's cut is
    as much an input as this one's — and it can be edited in its own workspace."""
    from detail.gui.measurements import what_changed

    world = _Artifacts([_bout(1)], {"an_02": [_bout(7)]})
    measured = _measured_under(world)
    world.postures["an_02"] = [_bout(8)]
    assert what_changed(measured, world.run, world) == "an_02"
    # Gone altogether is changed too, not current.
    del world.postures["an_02"]
    assert what_changed(measured, world.run, world) == "an_02"


def test_selecting_measures_nothing_and_offers_the_button(panel, monkeypatch):
    """The first click used to start a 70 s measure mid-interpretation. It must not start one
    at all: the panel says what measuring costs and waits."""
    import detail.gui.measurements as measurements

    def refuse(*args, **kwargs):
        raise AssertionError("a selection measured the run")

    monkeypatch.setattr(measurements, "measure_run", refuse)
    world = _Artifacts([_bout(1)])
    panel.select(world.run, 1)
    assert panel._measured is None
    assert panel._measure_button.text() == measurements.MEASURE
    assert not panel._measure_button.isHidden()
    assert panel._rows == [] and panel._table is None
