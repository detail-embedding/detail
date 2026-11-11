"""Measures compute numbers a reader can trust, over windows that add up (§23).

Three classes guarded here, all of which ship green:

**A wrong window-boundary rule.** An instance straddling a window edge has to be handled, and the
right handling differs per measure — but getting it wrong produces plausible numbers that do not
add up. Measuring duration *by overlap* truncates every bout at the edge and reports bouts shorter
than they were; nothing raises. The signal is additivity: a measure declared additive must sum
across adjacent windows to the value over their union.

**A denominator from the wrong scope.** ``coverage.observed_ns`` answers "how much was observed"
for a whole recording and cannot answer it for a window, because it records how much was unobserved
and not *where*. Substituting it into a sub-window inflates or deflates every rate in proportion to
where the tracking gaps happen to fall.

**Counting rows instead of bouts.** ``instances.csv`` is one row per ``(bout, label)``, so a cluster
named twice appears twice over one span. The §19 audit counted rows and reported precision 0.667
where the truth was 0.5.
"""

import numpy as np
import pytest

from detail.analysis import measures as m
from detail.analysis import windows as w
from detail.storage import behavior_output as bo
from detail.storage import frame_view as fv

PERIOD = 10_000_000  # 10 ms — 100 Hz, so 100 frames is one readable second
N_FRAMES = 100


def _axis(n=N_FRAMES):
    return (np.arange(n) * PERIOD).astype(np.int64)


def _manifest(**over):
    base = {
        "producer": {"module_id": "test", "run_id": "an_01"},
        "origin": "predicted",
        "label_space": "pooled",
        "vocabulary": ["rearing"],
        "names": {"q01": ["rearing"]},
        "overlap": "exclusive",
        "behavior_types": {"rearing": "state"},
    }
    base.update(over)
    return base


def _bout(first, last, source="q01", label="rearing"):
    t = _axis()
    return bo.BehaviorInstance(
        "rec", 0, int(t[first]), int(t[last]), first, last, source, label
    )


def _artifact(tmp_path, instances, *, observed=None, manifest=None, view=True):
    t = _axis()
    obs = np.ones(N_FRAMES, bool) if observed is None else observed
    coverage = [
        bo.CoverageRow(
            "rec", 0, 0, N_FRAMES * PERIOD, int(obs.sum()) * PERIOD, int(obs.sum())
        )
    ]
    bo.write_behavior_output(
        tmp_path,
        instances=instances,
        coverage=coverage,
        manifest=manifest or _manifest(),
    )
    if view:
        fv.write_frame_view(
            tmp_path,
            recording_id="rec",
            subject=0,
            t_ns=t,
            observed=obs,
            instances=instances,
        )
    return tmp_path


def _pick(rows, measure, label="rearing"):
    return [r for r in rows if r["measure"] == measure and r["label"] == label]


def _value(rows, measure, label="rearing"):
    return _pick(rows, measure, label)[0]["value"]


# --- the numbers, asserted rather than smoke-tested ------------------------------------- #


def test_the_whole_recording_numbers_are_the_hand_computed_ones(tmp_path):
    """Two bouts: frames 10-12 (three frames) and frame 50 alone (one)."""
    run = _artifact(tmp_path, [_bout(10, 12), _bout(50, 50)])
    rows = m.measure_run(run)
    assert _value(rows, "n_instances") == 2
    assert _value(rows, "total_time") == 4 * PERIOD  # 3 frames + 1 frame
    assert _value(rows, "time_fraction") == pytest.approx(4 / N_FRAMES)
    assert _value(rows, "duration_mean") == 2 * PERIOD
    assert _value(rows, "rate") == pytest.approx(2.0)  # 2 bouts in one observed second
    assert _value(rows, "latency_to_first") == 10 * PERIOD


def test_a_single_frame_bout_lasts_one_period_not_zero(tmp_path):
    """Invisible in any fixture whose bouts are long, and wrong by one period per bout
    everywhere else. ``end_t_ns - t_ns`` is zero here; the measured span is half-open.
    """
    run = _artifact(tmp_path, [_bout(50, 50)])
    rows = m.measure_run(run)
    assert _value(rows, "total_time") == PERIOD
    assert _value(rows, "duration_mean") == PERIOD


# --- the headline guard ------------------------------------------------------------------ #


@pytest.mark.parametrize("measure", ["total_time", "n_instances"])
def test_an_additive_measure_sums_across_adjacent_windows(tmp_path, measure):
    """The guard that catches a wrong boundary rule. One bout deliberately straddles the seam."""
    run = _artifact(tmp_path, [_bout(5, 7), _bout(45, 55), _bout(80, 82)])
    whole = m.measure_run(run, windows_for=lambda c: w.whole(0, N_FRAMES * PERIOD))
    halves = m.measure_run(
        run,
        windows_for=lambda c: w.absolute(
            0, N_FRAMES * PERIOD, every_ns=N_FRAMES * PERIOD // 2
        ),
    )
    assert sum(r["value"] for r in _pick(halves, measure)) == _value(whole, measure)


def test_every_additive_measure_is_actually_additive(tmp_path):
    """Data-driven over the registry, so a new measure declaring ``additive`` is covered the day
    it registers rather than when someone remembers to extend a parametrize list."""
    run = _artifact(tmp_path, [_bout(5, 7), _bout(45, 55), _bout(80, 82)])
    whole = m.measure_run(run, windows_for=lambda c: w.whole(0, N_FRAMES * PERIOD))
    halves = m.measure_run(
        run,
        windows_for=lambda c: w.absolute(
            0, N_FRAMES * PERIOD, every_ns=N_FRAMES * PERIOD // 2
        ),
    )
    additive = [k for k, spec in m.MEASURES.items() if spec.additive]
    assert additive, "no additive measures — this guard would pass vacuously"
    for measure in additive:
        assert sum(r["value"] for r in _pick(halves, measure)) == _value(
            whole, measure
        ), measure


def test_an_onset_measure_reports_a_straddling_bout_at_full_length(tmp_path):
    """Truncating at the window edge would report 5 frames for an 11-frame bout, and the table
    would look entirely reasonable."""
    run = _artifact(tmp_path, [_bout(45, 55)])
    halves = m.measure_run(
        run,
        windows_for=lambda c: w.absolute(
            0, N_FRAMES * PERIOD, every_ns=N_FRAMES * PERIOD // 2
        ),
    )
    first = [r for r in _pick(halves, "duration_mean") if r["window_id"] == "abs_00"][0]
    assert first["value"] == 11 * PERIOD  # frames 45..55 inclusive
    assert first["n"] == 1


# --- denominators ------------------------------------------------------------------------ #


def test_a_window_denominator_is_its_own_observed_time(tmp_path):
    """The gap sits only in the second half, so the halves must not share a denominator."""
    observed = np.ones(N_FRAMES, bool)
    observed[60:70] = False
    run = _artifact(tmp_path, [_bout(5, 7)], observed=observed)
    halves = m.measure_run(
        run,
        windows_for=lambda c: w.absolute(
            0, N_FRAMES * PERIOD, every_ns=N_FRAMES * PERIOD // 2
        ),
    )
    denoms = {
        r["window_id"]: r["denominator_ns"] for r in _pick(halves, "time_fraction")
    }
    assert denoms["abs_00"] == 50 * PERIOD
    assert denoms["abs_01"] == 40 * PERIOD  # ten frames unobserved


def test_every_rate_row_carries_a_denominator(tmp_path):
    run = _artifact(tmp_path, [_bout(5, 7)])
    for row in m.measure_run(run):
        spec = m.MEASURES[row["measure"]]
        if spec.needs_denominator:
            assert row["denominator_ns"], row


def test_a_window_with_no_observed_time_yields_no_measurement(tmp_path):
    """Not a zero rate. A 0.0 would enter a group mean as a real observation of "never
    happened", which is the absence-versus-zero error one layer up."""
    observed = np.ones(N_FRAMES, bool)
    observed[50:] = False
    run = _artifact(tmp_path, [_bout(5, 7)], observed=observed)
    rows = m.measure_run(
        run,
        windows_for=lambda c: w.absolute(
            0, N_FRAMES * PERIOD, every_ns=N_FRAMES * PERIOD // 2
        ),
    )
    second = [r for r in rows if r["window_id"] == "abs_01" and r["measure"] == "rate"]
    assert second == []


def test_without_a_frame_view_only_the_whole_window_is_answerable(tmp_path):
    """``coverage.observed_ns`` is exactly right for its own span and answers nothing smaller."""
    run = _artifact(tmp_path, [_bout(5, 7)], view=False)
    whole = m.measure_run(run)
    assert _value(whole, "time_fraction") == pytest.approx(3 / N_FRAMES)
    with pytest.raises(m.MeasureError, match="frame view"):
        m.measure_run(
            run,
            windows_for=lambda c: w.absolute(
                0, N_FRAMES * PERIOD, every_ns=PERIOD * 50
            ),
        )


# --- bouts, labels, refusals -------------------------------------------------------------- #


def test_a_twice_named_cluster_does_not_double_count(tmp_path):
    """One bout, two names. Counting rows would report two instances and twice the time."""
    manifest = _manifest(
        vocabulary=["rearing", "near_wall"], names={"q01": ["rearing", "near_wall"]}
    )
    run = _artifact(tmp_path, [_bout(10, 12)], manifest=manifest)
    rows = m.measure_run(run)
    for label in ("rearing", "near_wall"):
        assert _value(rows, "n_instances", label) == 1
        assert _value(rows, "total_time", label) == 3 * PERIOD


def test_a_vocabulary_label_that_never_occurs_is_a_real_zero(tmp_path):
    """Absent from the table it would be indistinguishable from a recording nobody analyzed —
    the distinction §19's coverage exists to preserve."""
    manifest = _manifest(vocabulary=["rearing", "grooming"])
    run = _artifact(tmp_path, [_bout(10, 12)], manifest=manifest)
    rows = m.measure_run(run)
    assert _value(rows, "n_instances", "grooming") == 0
    assert _value(rows, "total_time", "grooming") == 0


def test_n_matches_what_the_measure_consumed(tmp_path):
    """A value over three bouts reported with an n of thirty is caught by no arithmetic check."""
    run = _artifact(tmp_path, [_bout(5, 7), _bout(45, 47), _bout(80, 82)])
    rows = m.measure_run(run)
    assert _value(rows, "n_instances") == 3
    assert _pick(rows, "n_instances")[0]["n"] == 3
    assert _pick(rows, "duration_mean")[0]["n"] == 3


def test_a_point_behavior_reports_no_duration(tmp_path):
    """Half-open measurement would hand an instant exactly one period — frame quantization
    dressed as a duration. Count and rate stay meaningful and are computed."""
    manifest = _manifest(behavior_types={"rearing": "point"})
    run = _artifact(tmp_path, [_bout(10, 10), _bout(50, 50)], manifest=manifest)
    rows = m.measure_run(run)
    assert _value(rows, "total_time") == ""
    assert _value(rows, "duration_mean") == ""
    assert _value(rows, "n_instances") == 2
    assert _value(rows, "rate") == pytest.approx(2.0)


def test_transitions_are_refused_on_an_overlapping_label_set(tmp_path):
    """ "The next label" is undefined once two can be active at once — refused rather than
    answered with whichever row happened to sort first."""
    manifest = _manifest(overlap="allowed")
    run = _artifact(tmp_path, [_bout(10, 12), _bout(50, 52)], manifest=manifest)
    assert _pick(m.measure_run(run), "transitions") == []


def test_transitions_count_ordered_pairs(tmp_path):
    """One row per **ordered pair**: `label` is the from, `label_to` is the to
    (`DESIGN_PLAN` §23's measure table).

    It counted the *destinations* and left the from-label blank — `prev` was bound in the loop
    and never read — so rearing→grooming and grooming→rearing both landed in a row keyed
    `("", …)`, and every from-label's transitions were summed into one number. A transition
    matrix is the from-label; without it the measure answers "how often did anything precede
    this", which nothing asked.
    """
    manifest = _manifest(
        vocabulary=["rearing", "grooming"],
        names={"q01": ["rearing"], "q02": ["grooming"]},
    )
    run = _artifact(
        tmp_path,
        [_bout(10, 12), _bout(20, 22, "q02", "grooming"), _bout(30, 32)],
        manifest=manifest,
    )
    rows = m.measure_run(run)
    pairs = {
        (r["label"], r["label_to"]): r["value"]
        for r in rows
        if r["measure"] == "transitions"
    }
    assert pairs == {
        ("rearing", "grooming"): 1.0,
        ("grooming", "rearing"): 1.0,
    }


def test_a_transition_is_counted_from_the_label_it_leaves(tmp_path):
    """The asymmetry the old shape could not express. Three rearing bouts and one grooming, in
    the order R G R R: rearing leaves to grooming once and to rearing once; grooming leaves to
    rearing once. Summed over destinations — what it used to report — those are
    indistinguishable from any other arrangement with the same destination counts.
    """
    manifest = _manifest(
        vocabulary=["rearing", "grooming"],
        names={"q01": ["rearing"], "q02": ["grooming"]},
    )
    run = _artifact(
        tmp_path,
        [
            _bout(10, 12),
            _bout(20, 22, "q02", "grooming"),
            _bout(30, 32),
            _bout(40, 42),
        ],
        manifest=manifest,
    )
    pairs = {
        (r["label"], r["label_to"]): r["value"]
        for r in m.measure_run(run)
        if r["measure"] == "transitions"
    }

    assert pairs == {
        ("rearing", "grooming"): 1.0,
        ("rearing", "rearing"): 1.0,
        ("grooming", "rearing"): 1.0,
    }


def test_the_last_bout_of_a_unit_transitions_to_nothing(tmp_path):
    """There is no next label after the final bout, so it contributes no row rather than a row
    to an invented destination."""
    manifest = _manifest(
        vocabulary=["rearing", "grooming"],
        names={"q01": ["rearing"], "q02": ["grooming"]},
    )
    run = _artifact(
        tmp_path,
        [_bout(10, 12), _bout(20, 22, "q02", "grooming")],
        manifest=manifest,
    )
    pairs = {
        (r["label"], r["label_to"]): r["value"]
        for r in m.measure_run(run)
        if r["measure"] == "transitions"
    }

    assert pairs == {("rearing", "grooming"): 1.0}


def test_unknown_measures_are_named_not_ignored(tmp_path):
    run = _artifact(tmp_path, [_bout(10, 12)])
    with pytest.raises(m.MeasureError, match="unknown measure"):
        m.measure_run(run, measures=["not_a_measure"])


def test_every_registered_measure_declares_a_known_rule():
    """Registry parity: a new measure is covered the day it registers."""
    for measure_id, spec in m.MEASURES.items():
        assert spec.rule in m.RULES, measure_id
        assert spec.scope in ("label", "unit"), measure_id
        assert callable(spec.fn), measure_id


def test_the_table_writes_every_declared_column(tmp_path):
    run = _artifact(tmp_path, [_bout(10, 12)])
    out = m.write_measures(tmp_path / "measures.csv", m.measure_run(run))
    header = out.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert header == m.COLUMNS
