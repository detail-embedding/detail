"""Tests for the §19 reliability audit (detail/storage/behavior_audit.py).

Most of these assert a *refusal to produce a number*. That is the point of the file: an
unaudited artifact must read as unaudited, and a rate over zero decisions is not a rate. A
default in either place looks exactly like a real estimate and is the failure the audit
exists to prevent.

The other half is direction. `Match.score` is a distance everywhere in this codebase, so
"retained by the cutoff" is `score <= cutoff` — and an artifact that declares the opposite
convention must be read the opposite way. Assuming one silently inverts precision.

**Recorded class: an estimate over a stratified draw read back as a plain share.** The draw gives
each score bin about the same number of clips however many candidates it holds, so precision has to
weight each bin by its size among what the cutoff keeps, in the bins the draw used. Read as the
plain share of judged clips it reported 0.62 against a true 0.40, with an interval that covered the
truth in 17% of audits, and nothing looked wrong. The guards are the section headed "a stratified
draw is read back with its strata's weights", seeded over thirty audits of one run.
"""

from detail.storage import behavior_audit as audit
from detail.storage import behavior_output as bo


def _instance(score, ident, **kw):
    data = dict(
        recording_id="recording_001",
        subject=0,
        t_ns=ident * 1_000,
        end_t_ns=ident * 1_000 + 500,
        start_frame=ident,
        end_frame=ident + 1,
        source_label="q01",
        score=score,
    )
    data.update(kw)
    return bo.BehaviorInstance(**data)


def _instances(scores):
    return [_instance(score, i) for i, score in enumerate(scores)]


def _manifest(direction="lower_is_better"):
    return {"score": {"meaning": "cosine distance", "direction": direction}}


# --------------------------------------------------------------------------- #
# the log
# --------------------------------------------------------------------------- #
def test_append_and_read_round_trip(tmp_path):
    audit.append_audit(tmp_path, "abc", score=0.2, human_label=True, drawn_at="t0")
    assert audit.read_audit(tmp_path) == [
        audit.AuditRow("abc", 0.2, True, "t0"),
    ]


def test_a_decision_is_stamped_when_it_was_made(tmp_path):
    row = audit.append_audit(tmp_path, "abc", score=0.2, human_label=False)
    assert row.drawn_at.endswith("Z") and len(row.drawn_at) == 20


def test_an_undecided_row_is_none_not_false(tmp_path):
    # Drawn into the sample but not yet judged. Counting it as a rejection would report a
    # low precision for a sample nobody finished.
    audit.append_audit(tmp_path, "abc", score=0.2, human_label=None)
    assert audit.read_audit(tmp_path)[0].human_label is None


def test_re_auditing_appends_and_the_latest_wins(tmp_path):
    audit.append_audit(tmp_path, "abc", score=0.2, human_label=True, drawn_at="t0")
    audit.append_audit(tmp_path, "abc", score=0.2, human_label=False, drawn_at="t1")
    assert len(audit.read_audit(tmp_path)) == 2  # the full trail is kept
    latest = audit.latest_decisions(tmp_path)
    assert len(latest) == 1 and latest[0].human_label is False


def test_a_torn_trailing_line_is_tolerated(tmp_path):
    audit.append_audit(tmp_path, "abc", score=0.2, human_label=True)
    audit.append_audit(tmp_path, "def", score=0.4, human_label=False)
    with open(tmp_path / bo.AUDIT_FILENAME, "a", newline="", encoding="utf-8") as f:
        f.write("ghi,0.5\n")
    assert [r.instance_id for r in audit.read_audit(tmp_path)] == ["abc", "def"]


def test_has_audit_reflects_the_file(tmp_path):
    assert audit.has_audit(tmp_path) is False
    audit.append_audit(tmp_path, "abc", score=0.2, human_label=True)
    assert audit.has_audit(tmp_path) is True


# --------------------------------------------------------------------------- #
# sampling: stratified, not top-N
# --------------------------------------------------------------------------- #
def test_the_sample_is_not_the_best_ranked_instances(tmp_path):
    """The optimism guard.

    Reviewing the top of the ranking reviews the easiest instances, and precision comes out
    flattering. A stratified draw has to reach the worst scores too.
    """
    instances = _instances([i / 100 for i in range(100)])
    drawn = audit.draw_audit_sample(instances, per_bin=2, n_bins=10, seed=0)
    # Not a subset check: `per_bin=2, n_bins=10` draws 20 ids and the top slice holds 10, so
    # a 20-element set can never be a subset of it — the assertion was true against a literal
    # top-N draw. Assert the spread instead, which is the property.
    edges = audit.bin_edges([row.score for row in instances], 10)
    covered = {audit._bin_of(row.score, edges) for row in drawn}
    assert covered == set(
        range(10)
    ), f"the draw missed bins {sorted(set(range(10)) - covered)}"
    assert max(row.score for row in drawn) > 0.8
    assert min(row.score for row in drawn) < 0.2


def test_each_bin_contributes_at_most_per_bin(tmp_path):
    instances = _instances([i / 100 for i in range(100)])
    drawn = audit.draw_audit_sample(instances, per_bin=3, n_bins=10, seed=0)
    edges = audit.bin_edges([r.score for r in instances], 10)
    counts: dict[int, int] = {}
    for row in drawn:
        counts[audit._bin_of(row.score, edges)] = (
            counts.get(audit._bin_of(row.score, edges), 0) + 1
        )
    assert counts and max(counts.values()) <= 3


def test_a_sparse_bin_contributes_everything_it_has(tmp_path):
    instances = _instances([0.0, 0.05, 0.95])
    drawn = audit.draw_audit_sample(instances, per_bin=5, n_bins=10, seed=0)
    assert len(drawn) == 3


def test_the_binned_draw_is_deterministic_for_a_seed(tmp_path):
    instances = _instances([i / 100 for i in range(100)])
    first = audit.draw_audit_sample(instances, per_bin=2, seed=7)
    second = audit.draw_audit_sample(instances, per_bin=2, seed=7)
    assert [r.instance_id for r in first] == [r.instance_id for r in second]


def test_an_excluded_instance_is_never_drawn(tmp_path):
    """A confirmed match that became an exemplar built the method; it cannot also score it.

    The two label sets are kept separate so the constraint is checkable rather than
    remembered — this is the check.
    """
    instances = _instances([i / 100 for i in range(100)])
    excluded = {row.instance_id for row in instances[::3]}
    drawn = audit.draw_audit_sample(
        instances, per_bin=10, n_bins=5, seed=0, exclude_instance_ids=excluded
    )
    assert drawn and not {row.instance_id for row in drawn} & excluded


def test_scoreless_instances_are_not_sampled(tmp_path):
    # There is no bin for them; including them would not be stratified by score.
    assert audit.draw_audit_sample(_instances([None, None]), per_bin=5) == []


# --------------------------------------------------------------------------- #
# absence reads as absence
# --------------------------------------------------------------------------- #
def test_no_audit_yields_none_everywhere(tmp_path):
    instances = _instances([0.1, 0.2])
    assert audit.precision_at_cutoff(tmp_path, instances, cutoff=0.5) is None
    assert audit.precision_by_score_bin(tmp_path, instances) is None


def test_a_cutoff_that_retains_nothing_judged_reports_n_zero_not_a_rate(tmp_path):
    instances = _instances([0.8, 0.9])
    audit.append_audit(tmp_path, instances[0].instance_id, score=0.8, human_label=True)
    out = audit.precision_at_cutoff(
        tmp_path, instances, cutoff=0.1, manifest=_manifest()
    )
    assert out["n"] == 0 and out["precision"] is None  # never a fabricated 0.0


def test_an_unjudged_bin_reports_no_precision(tmp_path):
    instances = _instances([0.0, 0.9])
    audit.append_audit(tmp_path, instances[0].instance_id, score=0.0, human_label=True)
    bins = audit.precision_by_score_bin(tmp_path, instances, n_bins=2)
    assert bins[0]["precision"] == 1.0 and bins[0]["n"] == 1
    assert bins[-1]["precision"] is None and bins[-1]["n"] == 0


# --------------------------------------------------------------------------- #
# direction
# --------------------------------------------------------------------------- #
def test_precision_reads_the_declared_direction(tmp_path):
    """The same decisions under the two conventions must not retain the same instances.

    Score is a distance here, so a cutoff keeps the *low* scores. An artifact declaring
    higher_is_better keeps the high ones. Assuming either silently inverts the answer.
    """
    instances = _instances([0.1, 0.9])
    audit.append_audit(tmp_path, instances[0].instance_id, score=0.1, human_label=True)
    audit.append_audit(tmp_path, instances[1].instance_id, score=0.9, human_label=False)

    lower = audit.precision_at_cutoff(
        tmp_path, instances, cutoff=0.5, manifest=_manifest("lower_is_better")
    )
    higher = audit.precision_at_cutoff(
        tmp_path, instances, cutoff=0.5, manifest=_manifest("higher_is_better")
    )
    assert lower["n"] == 1 and lower["precision"] == 1.0  # kept the confirmed 0.1
    assert higher["n"] == 1 and higher["precision"] == 0.0  # kept the rejected 0.9


def test_precision_over_everything_when_there_is_no_cutoff(tmp_path):
    instances = _instances([0.1, 0.9])
    audit.append_audit(tmp_path, instances[0].instance_id, score=0.1, human_label=True)
    audit.append_audit(tmp_path, instances[1].instance_id, score=0.9, human_label=False)
    out = audit.precision_at_cutoff(
        tmp_path, instances, cutoff=None, manifest=_manifest()
    )
    assert out["n"] == 2 and out["precision"] == 0.5


def test_an_undecided_instance_does_not_count_against_precision(tmp_path):
    instances = _instances([0.1, 0.2])
    audit.append_audit(tmp_path, instances[0].instance_id, score=0.1, human_label=True)
    audit.append_audit(tmp_path, instances[1].instance_id, score=0.2, human_label=None)
    out = audit.precision_at_cutoff(
        tmp_path, instances, cutoff=1.0, manifest=_manifest()
    )
    assert out["n"] == 1 and out["precision"] == 1.0


# --------------------------------------------------------------------------- #
# the artifact notices
# --------------------------------------------------------------------------- #
def test_the_manifest_reports_audited_once_the_file_exists(tmp_path):
    instances = _instances([0.1])
    bo.write_behavior_output(
        tmp_path,
        instances=instances,
        coverage=[
            bo.CoverageRow("recording_001", 0, 0, 10_000, 9_000, 90),
        ],
        manifest={
            "producer": {"module_id": "query", "run_id": "an_01"},
            "origin": "predicted",
            "label_space": "pooled",
            "vocabulary": [],
            "score": {"meaning": "distance", "direction": "lower_is_better"},
        },
    )
    assert bo.read_behavior_output(tmp_path).audited is False

    audit.append_audit(tmp_path, instances[0].instance_id, score=0.1, human_label=True)
    # the flag is derived, so it needs a rewrite to be reflected — the reprojection §21 does
    bo.write_behavior_output(
        tmp_path,
        instances=instances,
        coverage=[bo.CoverageRow("recording_001", 0, 0, 10_000, 9_000, 90)],
        manifest={
            "producer": {"module_id": "query", "run_id": "an_01"},
            "origin": "predicted",
            "label_space": "pooled",
            "vocabulary": [],
            "score": {"meaning": "distance", "direction": "lower_is_better"},
        },
    )
    assert bo.read_behavior_output(tmp_path).audited is True


# --------------------------------------------------------------------------- #
# class guard: the audit counts bouts, never rows
# --------------------------------------------------------------------------- #
# `instances.csv` is one row per (bout, label), so a cluster the user named twice appears
# twice over one span — one `instance_id`, one human decision, two rows. Counting rows counts
# that decision twice, which is the exact failure §19 introduced `instance_id` to prevent, and
# nothing about the arithmetic looks wrong while it happens.
#
# Driven over the four entry points rather than written four times, so a fifth consumer is
# covered the day it is added. The sibling one file over (`frame_view`) solved this and said
# why; this module is the one that missed it.
def _twice_named(tmp_path):
    """Two bouts, one of them named twice, with one human decision each.

    Truth: 2 bouts, 1 confirmed, precision 0.5. Counting rows gives 3 and 0.667.
    """
    rows = [
        _instance(0.10, 1, label="rearing"),
        _instance(0.10, 1, label="near wall"),  # same span -> same instance_id
        _instance(0.90, 2, label="rearing"),
    ]
    ids = {row.instance_id for row in rows}
    assert len(ids) == 2, "fixture no longer builds a twice-named bout"
    audit.append_audit(tmp_path, rows[0].instance_id, score=0.10, human_label=True)
    audit.append_audit(tmp_path, rows[2].instance_id, score=0.90, human_label=False)
    return rows


def test_precision_at_cutoff_counts_a_twice_named_bout_once(tmp_path):
    rows = _twice_named(tmp_path)
    out = audit.precision_at_cutoff(tmp_path, rows, cutoff=None, manifest=_manifest())
    assert out["n"] == 2, "counted label assignments instead of bouts"
    assert out["n_confirmed"] == 1
    assert out["precision"] == 0.5


def test_precision_by_score_bin_counts_a_twice_named_bout_once(tmp_path):
    rows = _twice_named(tmp_path)
    bins = audit.precision_by_score_bin(tmp_path, rows, n_bins=2)
    assert sum(b["n_instances"] for b in bins) == 2
    assert sum(b["n"] for b in bins) == 2
    low = next(b for b in bins if b["n_confirmed"])
    assert low["n_instances"] == 1 and low["n"] == 1 and low["precision"] == 1.0


def test_the_audit_sample_never_shows_one_bout_twice(tmp_path):
    # The reviewer-facing half: a duplicated row spends a bin's budget showing the same clip
    # again, so the sample is smaller than it looks and other bouts go unreviewed.
    rows = _twice_named(tmp_path)
    drawn = audit.draw_audit_sample(rows, per_bin=5, n_bins=2, seed=0)
    ids = [row.instance_id for row in drawn]
    assert len(ids) == len(set(ids)) == 2


# --------------------------------------------------------------------------- #
# the interval, the per-recording split, and the two draw strategies
# --------------------------------------------------------------------------- #
def _inst(i, label, *, score=None, recording="r1"):
    return bo.BehaviorInstance(
        recording_id=recording,
        subject=0,
        t_ns=i * 1000,
        end_t_ns=i * 1000 + 500,
        start_frame=i,
        end_frame=i,
        source_label=label,
        score=score,
    )


def test_a_clustering_artifact_yields_a_sample_rather_than_nothing():
    """The regression test for the defect this work exists to fix.

    A clustering run writes no score (`pipeline/temporal_clustering.py` builds its instances
    without one), and the score-bin draw skips scoreless bouts by design. So an audit of a
    clustering artifact returned an **empty list** — not an error, not a warning, just no clips to
    review, while the spec claimed a module "gets Audit by emitting the artifact".
    """
    unscored = [_inst(i, f"cluster_{i % 10}") for i in range(100)]
    assert (
        audit.draw_audit_sample(unscored) == []
    ), "the scored path used to be the only path"

    drawn = audit.draw_sample(unscored, budget=25)
    assert len(drawn) == 25
    assert (
        len({row.source_label for row in drawn}) == 10
    ), "every cluster is represented"


def test_the_strategy_is_chosen_by_the_artifact_not_the_caller():
    scored = [_inst(i, "rear", score=i / 100.0) for i in range(100)]
    unscored = [_inst(i, f"cluster_{i % 4}") for i in range(100)]
    assert audit._strata(scored)[0] == "score_bin"
    assert audit._strata(unscored)[0] == "label"


def test_the_budget_is_what_is_asked_for():
    scored = [_inst(i, "rear", score=i / 100.0) for i in range(100)]
    for budget in (1, 7, 25, 50):
        assert len(audit.draw_sample(scored, budget=budget)) == budget
    assert audit.draw_sample(scored, budget=0) == []


def test_a_budget_larger_than_the_pool_returns_the_pool():
    """No topping up from nowhere, and no infinite loop looking for it."""
    scored = [_inst(i, "rear", score=i / 10.0) for i in range(6)]
    assert len(audit.draw_sample(scored, budget=25)) == 6


def test_a_second_batch_draws_new_instances_and_the_thinnest_stratum_first():
    """ "Review 25 more" has to be additive. Serving the thinnest stratum first is what makes
    the second batch buy resolution where the first left it thin."""
    scored = [_inst(i, "rear", score=i / 100.0) for i in range(100)]
    first = audit.draw_sample(scored, budget=25)
    second = audit.draw_sample(
        scored, budget=25, exclude_instance_ids=[r.instance_id for r in first]
    )
    assert not ({r.instance_id for r in first} & {r.instance_id for r in second})
    edges = audit.bin_edges([r.score for r in scored])
    counts: dict = {}
    for row in [*first, *second]:
        counts[audit._bin_of(row.score, edges)] = (
            counts.get(audit._bin_of(row.score, edges), 0) + 1
        )
    assert max(counts.values()) - min(counts.values()) <= 1, f"uneven strata: {counts}"


def test_the_budgeted_draw_is_deterministic_for_a_seed():
    scored = [_inst(i, "rear", score=i / 100.0) for i in range(100)]
    a = [r.instance_id for r in audit.draw_sample(scored, budget=25, seed=3)]
    b = [r.instance_id for r in audit.draw_sample(scored, budget=25, seed=3)]
    assert a == b
    assert a != [r.instance_id for r in audit.draw_sample(scored, budget=25, seed=4)]


def test_precision_ships_with_a_wilson_interval_that_widens_as_n_shrinks(tmp_path):
    """An interval, and Wilson specifically. The normal approximation runs past 1.0 at 9 of 10
    and collapses to zero width at 10 of 10 — both wrong in the direction that flatters, and
    neither looks wrong on screen. Asserted as an ordering, not against hardcoded widths.
    """
    widths = []
    for n in (10, 25, 50):
        lo, hi = audit.wilson_interval(round(0.9 * n), n)
        assert 0.0 <= lo < hi <= 1.0
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True), f"interval did not tighten: {widths}"
    assert audit.wilson_interval(0, 0) is None
    # the two cases the normal interval gets wrong
    assert audit.wilson_interval(9, 10)[1] <= 1.0
    assert audit.wilson_interval(10, 10)[0] < 1.0, "certainty from ten observations"


def test_a_precision_from_too_few_decisions_is_flagged_rather_than_hidden(tmp_path):
    """It is still returned — a caller may legitimately want the raw rate — but it carries the
    signal a surface renders "too few" from. Hiding it in the estimator was the first attempt
    and it destroyed a number tests and exports need."""
    instances = [_inst(i, "rear", score=i / 10.0) for i in range(10)]
    for row in instances[:3]:
        audit.append_audit(tmp_path, row.instance_id, score=row.score, human_label=True)
    got = audit.precision_at_cutoff(tmp_path, instances, cutoff=None)
    assert got["n"] == 3
    assert got["precision"] == 1.0
    assert got["below_min_n"] is True
    assert got["interval"][0] < 0.6, "n=3 at 1.00 must not read as certainty"


def test_precision_is_reported_per_recording_with_its_own_n(tmp_path):
    """Proportional, not balanced — so one recording can be thin, and says so."""
    instances = [
        *[_inst(i, "rear", score=0.9, recording="r1") for i in range(10)],
        *[_inst(100 + i, "rear", score=0.9, recording="r2") for i in range(10)],
    ]
    for row in instances[:9]:  # nine from r1
        audit.append_audit(tmp_path, row.instance_id, score=row.score, human_label=True)
    for row in instances[10:12]:  # two from r2
        audit.append_audit(
            tmp_path, row.instance_id, score=row.score, human_label=False
        )

    rows = {
        r["recording_id"]: r for r in audit.precision_by_recording(tmp_path, instances)
    }
    assert rows["r1"]["n"] == 9 and rows["r1"]["precision"] == 1.0
    assert rows["r1"]["below_min_n"] is False
    assert rows["r2"]["n"] == 2 and rows["r2"]["below_min_n"] is True
    assert rows["r1"]["n_instances"] == rows["r2"]["n_instances"] == 10


def test_no_audit_means_no_per_recording_estimate(tmp_path):
    """Absence reads as absence here too, never as a zero."""
    instances = [_inst(i, "rear", score=0.9) for i in range(5)]
    assert audit.precision_by_recording(tmp_path, instances) is None


# --------------------------------------------------------------------------- #
# "unclear" and "never looked at" are not the same state
# --------------------------------------------------------------------------- #
# The class: **two different states encoded as the same value.** `human_label` is `None` for
# an instance drawn into a sample and for one a reviewer judged *unclear*, and for a long time
# the log could not tell them apart — both wrote an empty cell. Nothing failed. Every metric
# was right, because a metric drops both: an unclear carries no evidence either way, so it is
# excluded from precision exactly like an unjudged clip.
#
# What it cost was everything that asks *has a person looked at this*. `audit_sample` offers
# undecided rows back before drawing fresh, so an unclear came round again as though it had
# never been seen — an ambiguous clip could be handed to the same reviewer forever — and no
# surface could list a reviewer's unclears for them to resolve, which is the one re-review
# that measurably tightens the interval.
#
# `decided` is the split. A metric keeps asking `human_label is not None`; anything asking
# whether a person has ruled asks `decided`.
def test_an_unclear_is_a_decision_and_an_undrawn_clip_is_not(tmp_path):
    audit.append_audit(tmp_path, "drawn", score=0.4)
    audit.append_audit(tmp_path, "unclear", human_label=None, decided=True)
    audit.append_audit(tmp_path, "yes", human_label=True)

    rows = {row.instance_id: row for row in audit.read_audit(tmp_path)}
    assert rows["drawn"].decided is False
    assert rows["unclear"].decided is True and rows["unclear"].human_label is None
    assert rows["yes"].decided is True and rows["yes"].human_label is True


def test_an_unclear_survives_a_round_trip_through_the_file(tmp_path):
    """It is the *cell* that has to carry the difference — an in-memory flag would be lost the
    moment the log was read back, which is the only way anything reads it."""
    audit.append_audit(tmp_path, "u", human_label=None, decided=True)
    text = (tmp_path / bo.AUDIT_FILENAME).read_text(encoding="utf-8-sig")
    assert audit.UNCLEAR_CELL in text
    assert audit.read_audit(tmp_path)[0].decided is True


def test_the_two_predicates_disagree_exactly_on_an_unclear(tmp_path):
    """The whole point of the split, stated as the invariant. A metric asks
    `human_label is not None` and must keep dropping an unclear, which carries no evidence
    either way; anything asking whether a person has ruled asks `decided` and must keep it.
    """
    audit.append_audit(tmp_path, "drawn", score=0.4)
    audit.append_audit(tmp_path, "unclear", human_label=None, decided=True)
    audit.append_audit(tmp_path, "yes", human_label=True)
    audit.append_audit(tmp_path, "no", human_label=False)
    rows = audit.read_audit(tmp_path)

    scored = {r.instance_id for r in rows if r.human_label is not None}
    ruled = {r.instance_id for r in rows if r.decided}
    assert scored == {"yes", "no"}
    assert ruled == {"yes", "no", "unclear"}
    assert ruled - scored == {"unclear"}


def test_a_label_written_without_the_flag_is_still_a_decision(tmp_path):
    """Guard on the inference. Defaulting `decided` to False instead would mean a caller that
    passes a judgment and forgets the flag silently records "never looked at"."""
    audit.append_audit(tmp_path, "y", human_label=False)
    assert audit.read_audit(tmp_path)[0].decided is True


# coverage — an audit drawn against one vocabulary is not a sample of another
# --------------------------------------------------------------------------- #
def _audit_all(run_dir, instances, labels):
    """Record one judgement per instance, in order."""
    for instance, label in zip(bo.bouts(instances), labels):
        audit.append_audit(
            run_dir, instance.instance_id, score=instance.score, human_label=label
        )


def test_coverage_is_one_when_nothing_has_changed(tmp_path):
    """The negative case first, so every assertion below cannot pass on any reprojection at all."""
    rows = _instances([0.1, 0.2, 0.3, 0.4])
    _audit_all(tmp_path, rows, [True, True, False, True])

    coverage = audit.decision_coverage(tmp_path, rows)
    assert coverage == {"n_decisions": 4, "n_applicable": 4, "coverage": 1.0}
    at = audit.precision_at_cutoff(tmp_path, rows, cutoff=1.0, manifest=_manifest())
    assert at["precision"] == 0.75 and at["reason"] is None


def test_coverage_falls_when_the_vocabulary_is_re_cut(tmp_path):
    """`instance_id` is a content hash of the bout *as produced*, so a re-cut vocabulary yields
    different ids and the old decisions stop applying. Before this was reported, they simply
    dropped out of the intersection and `n` shrank with nothing failing."""
    rows = _instances([0.1, 0.2, 0.3, 0.4])
    _audit_all(tmp_path, rows, [True, True, False, True])

    # Two of the four bouts survive the re-cut; the other two are different spans now.
    recut = rows[:2] + [_instance(0.5, 90), _instance(0.6, 91)]
    coverage = audit.decision_coverage(tmp_path, recut)
    assert coverage["n_decisions"] == 4
    assert coverage["n_applicable"] == 2
    assert coverage["coverage"] == 0.5


def test_a_number_is_withheld_below_the_floor_and_says_why(tmp_path):
    rows = _instances([0.1, 0.2, 0.3, 0.4])
    _audit_all(tmp_path, rows, [True, True, False, True])
    recut = rows[:2] + [_instance(0.5, 90), _instance(0.6, 91)]

    at = audit.precision_at_cutoff(tmp_path, recut, cutoff=1.0, manifest=_manifest())
    assert at["precision"] is None
    assert at["coverage"] == 0.5
    assert "2 of 4" in at["reason"] and "50% coverage" in at["reason"]


def test_the_withheld_number_would_have_flattered_the_method(tmp_path):
    """The reason this is a refusal and not a footnote. Whether an instance survives a re-cut is
    not independent of whether it was judged right — clusters merge because their contents
    resemble each other — so the survivors are a biased subset. Here the two that survive are
    both confirmed, and the old code would have reported 1.00 against a true 0.75.
    """
    rows = _instances([0.1, 0.2, 0.3, 0.4])
    _audit_all(tmp_path, rows, [True, True, False, False])
    honest = audit.precision_at_cutoff(tmp_path, rows, cutoff=1.0, manifest=_manifest())
    assert honest["precision"] == 0.5

    recut = rows[:2] + [
        _instance(0.5, 90),
        _instance(0.6, 91),
    ]  # only the confirmed two survive
    biased = audit.precision_at_cutoff(
        tmp_path, recut, cutoff=1.0, manifest=_manifest()
    )
    assert (
        biased["n"] == 2 and biased["n_confirmed"] == 2
    )  # what it *would* have said: 1.00
    assert biased["precision"] is None, "a flattering number is worse than no number"


def test_a_partly_covered_audit_above_the_floor_still_reports(tmp_path):
    """The floor is a floor, not a demand for perfection: 90% coverage yields a precision *and*
    its coverage, so a caller can render both."""
    rows = _instances([i / 10 for i in range(10)])
    _audit_all(tmp_path, rows, [True] * 9 + [False])
    recut = rows[:9] + [_instance(9.9, 99)]

    at = audit.precision_at_cutoff(tmp_path, recut, cutoff=99.0, manifest=_manifest())
    assert at["coverage"] == 0.9
    assert at["precision"] == 1.0 and at["reason"] is None


def test_every_estimate_carries_its_coverage(tmp_path):
    """Both, not just the one that was easy to reach — a caller that renders any of them must
    be able to say what it rests on."""
    rows = _instances([0.1, 0.2, 0.3, 0.4])
    _audit_all(tmp_path, rows, [True, True, False, True])
    recut = rows[:2] + [_instance(0.5, 90), _instance(0.6, 91)]

    at = audit.precision_at_cutoff(tmp_path, recut, cutoff=1.0, manifest=_manifest())
    bins = audit.precision_by_score_bin(tmp_path, recut, manifest=_manifest())

    assert at["coverage"] == 0.5
    assert at["reason"]
    assert all(row["coverage"] == 0.5 and row["precision"] is None for row in bins)


def test_an_unjudged_row_does_not_count_toward_coverage(tmp_path):
    """A drawn-but-unjudged row is not evidence, so counting it would understate coverage and
    withhold a number that is fine."""
    rows = _instances([0.1, 0.2])
    audit.append_audit(
        tmp_path, bo.bouts(rows)[0].instance_id, score=0.1, human_label=True
    )
    audit.append_audit(tmp_path, "never-judged", score=0.2, human_label=None)

    coverage = audit.decision_coverage(tmp_path, rows)
    assert coverage == {"n_decisions": 1, "n_applicable": 1, "coverage": 1.0}


def test_no_judgements_is_not_zero_coverage(tmp_path):
    """No evidence differs from evidence that no longer applies, and a caller renders them
    differently."""
    audit.append_audit(tmp_path, "abc", score=0.1, human_label=None)
    assert audit.decision_coverage(tmp_path, _instances([0.1]))["coverage"] is None


def test_a_snapshot_keeps_full_coverage_after_the_run_is_re_cut(tmp_path):
    """An audit is evidence about *one* artifact state (§19), and a snapshot copies it beside the
    artifact it was taken from (§21) — so the frozen number stays fully covered however far the
    working run moves afterwards."""
    from detail.storage import snapshots

    rows = _instances([0.1, 0.2, 0.3, 0.4])
    bo.write_behavior_output(
        tmp_path,
        instances=rows,
        coverage=[
            bo.CoverageRow("recording_001", 0, 0, 10_000, 10_000, 100),
        ],
        manifest={
            "producer": {"module_id": "clip_query", "run_id": "an_01"},
            "origin": "predicted",
            "label_space": "pooled",
            "vocabulary": ["rearing"],
            "names": {"q01": ["rearing"]},
            "overlap": "allowed",
            "behavior_types": {"rearing": "state"},
            "score": {"meaning": "cosine distance", "direction": "lower_is_better"},
        },
    )
    _audit_all(tmp_path, rows, [True, True, False, True])
    snapshot_id = snapshots.pin(tmp_path, reason="finish")
    frozen_dir = snapshots.snapshot_dir(tmp_path, snapshot_id)
    frozen = bo.read_behavior_output(frozen_dir)

    at = audit.precision_at_cutoff(
        frozen_dir, frozen.instances, cutoff=1.0, manifest=frozen.manifest
    )
    assert at["coverage"] == 1.0 and at["precision"] == 0.75

    # ...and the *contrast*, which is the whole claim. Nothing above re-cuts anything, so
    # without this the test proves only that `pin` copies the log — while its name and
    # docstring promise the frozen number survives the working run moving on.
    # Genuinely different spans, so genuinely different content hashes — `_instances` keys
    # `t_ns` off the index, so re-scoring the same four indices reproduces the same ids and
    # would have shown 100% coverage of a "re-cut" that never happened.
    recut = [_instance(0.5, i) for i in range(500, 504)]
    working = audit.precision_at_cutoff(
        tmp_path, recut, cutoff=1.0, manifest=frozen.manifest, universe=recut
    )
    assert working["coverage"] == 0.0, "the re-cut run should have lost its decisions"
    assert working["precision"] is None and working["reason"]
    still = audit.precision_at_cutoff(
        frozen_dir, frozen.instances, cutoff=1.0, manifest=frozen.manifest
    )
    assert still["coverage"] == 1.0 and still["precision"] == 0.75


# --------------------------------------------------------------------------- #
# coverage asks whether an instance still exists, never whether it is selected
# --------------------------------------------------------------------------- #
# The class: **a run-level property computed against a caller's slice.** `decision_coverage`
# exists to catch a re-cut vocabulary — ids are content hashes, so re-cutting drops decisions
# out of the intersection and the survivors are a biased subset. But every caller estimates a
# *slice* (one behaviour, one cutoff), and measuring against the slice counts ordinary scoping
# as loss: two behaviours put each at 50%, and raising a cutoff took a run to 8%. Both refused
# to report any number, blaming "the vocabulary changed" when nothing had — and the refusal
# then crashed the surface that had to render it.
def _pair(label, n, start=0):
    return [
        bo.BehaviorInstance(
            recording_id="r1",
            subject=0,
            t_ns=i * 1000,
            end_t_ns=i * 1000 + 500,
            start_frame=i,
            end_frame=i,
            source_label=label,
            score=0.5,
        )
        for i in range(start, start + n)
    ]


def _judge_all(run_dir, rows):
    for row in rows:
        audit.append_audit(run_dir, row.instance_id, score=row.score, human_label=True)


MANIFEST = {"score": {"direction": "higher_is_better"}}


def test_scoping_to_one_behavior_does_not_read_as_a_re_cut(tmp_path):
    rear, groom = _pair("rear", 10), _pair("groom", 10, start=100)
    _judge_all(tmp_path, rear + groom)

    scoped = audit.precision_at_cutoff(
        tmp_path, rear, cutoff=None, manifest=MANIFEST, universe=rear + groom
    )
    assert scoped["coverage"] == 1.0
    assert scoped["precision"] == 1.0, "a second behaviour made the first unreportable"


def test_a_cutoff_that_drops_instances_does_not_read_as_a_re_cut(tmp_path):
    """§20's property: **one audit serves every cutoff.** `_audit_pool` enumerates at cutoff
    zero precisely so a decision still attaches to its span after the cutoff moves."""
    rows = _pair("rear", 10)
    _judge_all(tmp_path, rows)
    kept = rows[:2]  # what a tighter cutoff leaves in the artifact

    e = audit.precision_at_cutoff(
        tmp_path, kept, cutoff=None, manifest=MANIFEST, universe=rows
    )
    assert e["coverage"] == 1.0
    assert e["precision"] is not None


def test_a_genuinely_re_cut_vocabulary_still_refuses(tmp_path):
    """The negative case — #245's whole reason for existing has to keep working, or the fix
    above is just a way of never refusing."""
    rows = _pair("rear", 10)
    _judge_all(tmp_path, rows)
    recut = _pair("rear", 10, start=500)  # different spans, so different content hashes

    e = audit.precision_at_cutoff(
        tmp_path, recut, cutoff=None, manifest=MANIFEST, universe=recut
    )
    assert e["coverage"] == 0.0
    assert e["precision"] is None and e["reason"]


def test_every_per_recording_row_carries_the_coverage_the_headline_does(tmp_path):
    """Or a row prints a rate the figure above it is refusing to print."""
    rows = _pair("rear", 10)
    _judge_all(tmp_path, rows)
    recut = _pair("rear", 10, start=500)

    by_recording = audit.precision_by_recording(
        tmp_path, recut, manifest=MANIFEST, universe=recut
    )
    assert by_recording, "no rows to check"
    for row in by_recording:
        assert "coverage" in row and row["reason"]
        assert row["precision"] is None


# --------------------------------------------------------------------------- #
# nested labels: one judgment answered at the deepest true level
# --------------------------------------------------------------------------- #
def _nested(i, path):
    return bo.BehaviorInstance(
        recording_id="r1",
        subject=0,
        t_ns=i * 1000,
        end_t_ns=i * 1000 + 500,
        start_frame=i,
        end_frame=i,
        source_label=path,
        label=path.rsplit("/", 1)[-1],
        label_path=path,
    )


def test_an_answer_survives_the_round_trip(tmp_path):
    audit.append_audit(tmp_path, "abc", human_label=False, answer="rearing")
    (row,) = audit.read_audit(tmp_path)
    assert row.answer == "rearing" and row.human_label is False and row.decided


def test_a_log_written_before_answers_existed_migrates_in_place(tmp_path):
    """The old four-column header is rewritten once, atomically, on the first new append —
    the old rows read an empty answer and keep every judgment they recorded."""
    path = tmp_path / bo.AUDIT_FILENAME
    path.write_text(
        "instance_id,score,human_label,drawn_at\nold,,true,t0\n", encoding="utf-8"
    )
    audit.append_audit(tmp_path, "new", human_label=False, answer="rearing")
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == ",".join(
        audit.AUDIT_COLUMNS
    )
    old, new = audit.read_audit(tmp_path)
    assert (old.instance_id, old.human_label, old.answer) == ("old", True, "")
    assert (new.instance_id, new.answer) == ("new", "rearing")


def test_the_choices_run_deepest_first_and_a_flat_label_keeps_yes_and_no():
    assert audit.answer_choices("rearing/supported") == [
        ("rearing › supported", "rearing/supported", True),
        ("rearing, not supported", "rearing", False),
        ("not rearing", audit.NOTHING_CELL, False),
    ]
    assert audit.answer_choices("a/b/c")[1:3] == [
        ("a › b, not c", "a/b", False),
        ("a, not b", "a", False),
    ]
    # Exactly the row a flat audit always wrote: no answer, just the yes/no.
    assert audit.answer_choices("rearing") == [
        ("rearing", "", True),
        ("not rearing", "", False),
    ]


def test_one_answer_reads_correctly_at_every_level():
    """The table that makes D worth having. Each judgment of a "rearing/supported" clip,
    scored against the sub-label and against the parent."""
    own = "rearing/supported"

    def row(human_label, answer=""):
        return audit.AuditRow("x", None, human_label, "t0", answer=answer)

    cases = [
        # (judgment,                       as supported, as rearing)
        (row(True, own), True, True),
        (row(False, "rearing"), False, True),
        (row(False, audit.NOTHING_CELL), False, False),
        (row(None), None, None),  # unclear carries nothing
        # Written before answers existed: a yes is a yes to its own label, and so to the
        # parent; a no rules out only its own label, and is not guessed for the parent.
        (row(True), True, True),
        (row(False), False, None),
    ]
    for judgment, as_child, as_parent in cases:
        assert audit.verdict(judgment, own, own) is as_child, judgment
        assert audit.verdict(judgment, own, "rearing") is as_parent, judgment
    # "rearing, not supported" is still not grooming — a sibling is outside the answer.
    assert audit.verdict(row(False, "rearing"), own, "grooming") is False


def test_a_parent_is_scored_over_everything_under_it(tmp_path):
    """Precision for "rearing" counts its own instances and its sub-label's, through
    `verdict`; precision for the sub-label counts only its own."""
    rows = [_nested(i, "rearing/supported") for i in range(4)] + [
        _nested(10 + i, "rearing") for i in range(2)
    ]
    for r in rows[:4]:
        audit.append_audit(tmp_path, r.instance_id, human_label=False, answer="rearing")
    for r in rows[4:]:
        audit.append_audit(tmp_path, r.instance_id, human_label=True)
    child = audit.precision_at_cutoff(
        tmp_path, rows, cutoff=None, target_path="rearing/supported"
    )
    parent = audit.precision_at_cutoff(
        tmp_path, rows, cutoff=None, target_path="rearing"
    )
    assert (child["n"], child["n_confirmed"]) == (4, 0)
    assert (parent["n"], parent["n_confirmed"]) == (6, 6)
    # Without a target the old reading stands: each judgment against its own label.
    flat = audit.precision_at_cutoff(tmp_path, rows, cutoff=None)
    assert (flat["n"], flat["n_confirmed"]) == (6, 2)


def test_a_label_does_not_claim_a_sibling_that_shares_its_prefix(tmp_path):
    """ "rear" is not a parent of "rearing": a prefix match on the string would score one
    behavior with another's judgments."""
    rows = [_nested(1, "rearing"), _nested(2, "rear")]
    audit.append_audit(tmp_path, rows[0].instance_id, human_label=True)
    est = audit.precision_at_cutoff(tmp_path, rows, cutoff=None, target_path="rear")
    assert est["n"] == 0


# --------------------------------------------------------------------------- #
# cluster verdicts: a stratified estimate, an adaptive draw
# --------------------------------------------------------------------------- #
def _yes_no(yes: int, no: int) -> list:
    return [True] * yes + [False] * no


def test_one_stratum_is_exactly_the_plain_estimate():
    """A label of one cluster reads as it always did."""
    plain = audit.estimate(_yes_no(9, 3))
    strat = audit.stratified_estimate({"a": (40, _yes_no(9, 3))})
    assert strat["precision"] == plain["precision"]
    assert strat["interval"] == plain["interval"]
    assert strat["unjudged_share"] == 0.0


def test_the_label_figure_is_weighted_by_each_clusters_size():
    """The draw gives a small cluster more than its share, so pooling would tilt the label toward
    it. A big pure cluster and a small impure one, judged equally: the label is mostly pure.
    """
    groups = {"big": (90, _yes_no(10, 0)), "small": (10, _yes_no(0, 10))}
    strat = audit.stratified_estimate(groups)
    assert abs(strat["precision"] - 0.9) < 1e-9
    pooled = audit.estimate(_yes_no(10, 10))["precision"]
    assert pooled == 0.5, "the pooled figure this replaces"


def test_unjudged_clusters_are_reported_not_hidden():
    groups = {"a": (60, _yes_no(8, 2)), "b": (40, [])}
    strat = audit.stratified_estimate(groups)
    assert abs(strat["unjudged_share"] - 0.4) < 1e-9
    assert abs(strat["precision"] - 0.8) < 1e-9


def test_a_cluster_judged_all_yes_does_not_claim_certainty():
    strat = audit.stratified_estimate(
        {"a": (50, _yes_no(10, 0)), "b": (50, _yes_no(10, 0))}
    )
    low, high = strat["interval"]
    assert low < 1.0 and high == 1.0


def test_the_projection_narrows_as_clips_are_added():
    """What a button shows before a round: the interval at the same share over more clips."""
    now = audit.estimate(_yes_no(22, 3))
    after = audit.projected_interval(now, 25)
    assert after == audit.wilson_interval(44, 50)
    assert audit.ci_width({"interval": after}) < audit.ci_width(now)
    assert audit.projected_interval(audit.estimate([]), 25) is None


def test_a_stratified_projection_uses_the_effective_n():
    groups = {"a": (300, _yes_no(9, 1)), "b": (30, _yes_no(5, 5))}
    figure = audit.stratified_estimate(groups)
    assert figure["n_effective"] != figure["n"]
    low, high = audit.projected_interval(figure, 0)
    assert (round(low, 9), round(high, 9)) == tuple(
        round(v, 9) for v in figure["interval"]
    )


def _cluster_instances(sizes: dict) -> tuple[list, dict]:
    rows, stratum_of = [], {}
    i = 0
    for cluster, size in sizes.items():
        for _ in range(size):
            row = _nested(i, "climb")
            rows.append(row)
            stratum_of[row.instance_id] = cluster
            i += 1
    return rows, stratum_of


def test_precision_since_counts_only_fresh_judgments(tmp_path):
    rows = [_nested(i, "climb") for i in range(4)]
    for r in rows[:2]:
        audit.append_audit(
            tmp_path, r.instance_id, human_label=False, drawn_at="2026-01-01T00:00:00Z"
        )
    for r in rows[2:]:
        audit.append_audit(
            tmp_path, r.instance_id, human_label=True, drawn_at="2026-03-01T00:00:00Z"
        )
    every = audit.precision_at_cutoff(tmp_path, rows, cutoff=None, target_path="climb")
    fresh = audit.precision_at_cutoff(
        tmp_path, rows, cutoff=None, target_path="climb", since="2026-02-01T00:00:00Z"
    )
    assert (every["n"], fresh["n"]) == (4, 2)
    assert fresh["precision"] == 1.0
    assert (every["n_before"], fresh["n_before"]) == (0, 2), "it says what it set aside"


# --------------------------------------------------------------------------- #
# a check: fixed answers, quick vs per cluster, the cost up front
# --------------------------------------------------------------------------- #
def test_a_checks_answers_come_from_its_scope_never_from_a_clip():
    """The keys must not re-map between clips: one list for the whole check, the scope's branch
    first, then "not" the scope."""
    branch = [
        "climb and rear",
        "climb and rear/climb corner",
        "climb and rear/climb front wall",
    ]
    options = audit.answer_options("climb and rear", branch + ["grooming"])
    assert [text for text, _a, _m in options] == [
        "climb and rear, other",
        "climb and rear › climb corner",
        "climb and rear › climb front wall",
        "not climb and rear",
    ]
    assert audit.answer_options("grooming", ["grooming"]) == (
        audit.answer_choices("grooming")
    ), "a root label with no sub-labels keeps the flat yes/no"


def test_checking_a_sub_label_asks_only_about_it():
    """The reported problem: checking *climb front wall* also asked about *climb and rear*,
    with two ways to say no that differed only about the parent."""
    branch = [
        "climb and rear",
        "climb and rear/climb corner",
        "climb and rear/climb front wall",
    ]
    own = "climb and rear/climb front wall"
    options = audit.answer_options(own, branch)
    assert [(text, answer) for text, answer, _m in options] == [
        ("climb front wall", own),
        ("not climb front wall", audit.ruled_out(own)),
    ]


def test_ruling_out_a_sub_label_says_nothing_about_its_parent():
    own = "climb and rear/climb front wall"
    no = audit.AuditRow(
        "x", None, audit.human_label_for(audit.ruled_out(own), False, own), "t0",
        answer=audit.ruled_out(own),
    )  # fmt: skip
    assert no.human_label is False
    assert audit.verdict(no, own, own) is False
    assert (
        audit.verdict(no, own, "climb and rear") is None
    ), "not counted for the parent"
    yes = audit.AuditRow("y", None, True, "t0", answer=own)
    assert audit.verdict(yes, own, "climb and rear") is True, "a yes still counts above"
    assert audit.answer_text(no, own) == "not climb front wall"
    # Ruling out a parent rules out everything under it.
    parent_no = audit.AuditRow(
        "z", None, False, "t0", answer=audit.ruled_out("climb and rear")
    )
    assert audit.verdict(parent_no, own, own) is False
    assert audit.verdict(parent_no, own, "climb and rear") is False


def test_none_of_these_reads_as_not_the_outermost_label():
    """A check of a root label records "not climb" as the bare separator; it read as "no"."""
    row = audit.AuditRow("x", None, False, "t0", answer=audit.NOTHING_CELL)
    assert audit.answer_text(row, "climb") == "not climb"
    assert audit.answer_text(row, "climb/corner") == "not climb"


def test_a_sibling_answer_reads_as_the_sibling():
    own = "climb and rear/climb front wall"
    row = audit.AuditRow("x", None, False, "t0", answer="climb and rear/climb corner")
    assert audit.answer_text(row, own) == "climb and rear › climb corner"


def test_naming_a_sibling_is_a_miss_for_the_clip_and_a_hit_for_the_parent():
    own = "climb and rear/climb front wall"
    row = audit.AuditRow(
        "x", None, audit.human_label_for("climb and rear/climb corner", False, own), "t0",
        answer="climb and rear/climb corner",
    )  # fmt: skip
    assert row.human_label is False
    assert audit.verdict(row, own, own) is False
    assert audit.verdict(row, own, "climb and rear") is True


def test_a_labels_round_draws_in_proportion_to_cluster_size():
    rows, stratum_of = _cluster_instances({"big": 300, "small": 30, "mid": 70})
    drawn = audit.draw_sample(rows, budget=20, stratum_of=stratum_of)
    counts = {}
    for row in drawn:
        counts[stratum_of[row.instance_id]] = (
            counts.get(stratum_of[row.instance_id], 0) + 1
        )
    for key, size in {"big": 300, "small": 30, "mid": 70}.items():
        quota = 20 * size / 400
        assert abs(counts.get(key, 0) - quota) <= 1, (key, counts, quota)


# --------------------------------------------------------------------------- #
# class guard: a stratified draw is read back with its strata's weights
# --------------------------------------------------------------------------- #
# The draw spreads a batch evenly across score bins — that is what keeps it from being a top-N
# sample — and the bins hold very different numbers of instances. Precision read back as the
# plain share of the judged clips therefore over-counts the thin bins: on a skewed score
# distribution, where most candidates score low and are mostly wrong, it read 0.62 where the
# truth was 0.40 and its interval covered the truth in 17% of audits. Nothing looked wrong: a
# share of judged clips is a plausible number with a plausible interval. The clustering half of
# the audit had already learned this (`stratified_estimate`, weighted by cluster size); the
# score-bin half had not. The rule: an estimate over a stratified draw weights each stratum by
# its size in the population the cutoff keeps, in the strata the draw used.
_SKEW_CUTOFF = 0.2


def _skewed(n: int = 400, seed: int = 0):
    """Candidates whose scores pile up low, with precision rising with score — the shape a
    query search has at no cutoff. Returns ``(rows, scores, truth)``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    scores = rng.beta(1.2, 4.0, n)
    truth = rng.random(n) < np.clip((scores - 0.05) / 0.6, 0, 1) ** 1.5
    return [_instance(float(s), i) for i, s in enumerate(scores)], scores, truth


def _audit_two_rounds(run_dir, rows, truth_of: dict, seed: int) -> None:
    """Two batches of 25 the way a reviewer takes them, every clip judged truthfully."""
    done: set = set()
    for batch in range(2):
        for row in audit.draw_sample(
            rows, budget=25, seed=seed * 7 + batch, exclude_instance_ids=done
        ):
            audit.append_audit(
                run_dir,
                row.instance_id,
                score=row.score,
                human_label=truth_of[row.instance_id],
            )
            done.add(row.instance_id)


def test_precision_at_a_cutoff_is_weighted_by_score_bin(tmp_path):
    """Over many audits of one run, the estimate averages to the true precision at the cutoff
    and its interval covers it about 95% of the time; the plain share of the judged clips does
    neither."""
    import numpy as np

    rows, scores, truth = _skewed()
    truth_of = {row.instance_id: bool(t) for row, t in zip(rows, truth)}
    true = float(truth[scores >= _SKEW_CUTOFF].mean())
    weighted, plain, covered = [], [], []
    for seed in range(30):
        run_dir = tmp_path / f"audit_{seed}"
        run_dir.mkdir()
        _audit_two_rounds(run_dir, rows, truth_of, seed)
        figure = audit.precision_at_cutoff(
            run_dir, rows, cutoff=_SKEW_CUTOFF, manifest=_manifest("higher_is_better")
        )
        weighted.append(figure["precision"])
        low, high = figure["interval"]
        covered.append(low <= true <= high)
        plain.append(
            np.mean(
                [
                    row.human_label
                    for row in audit.latest_decisions(run_dir)
                    if row.score >= _SKEW_CUTOFF
                ]
            )
        )
    assert abs(np.mean(plain) - true) > 0.1, (
        "the fixture no longer separates a weighted estimate from the plain share, so this "
        "test cannot fail"
    )
    assert abs(np.mean(weighted) - true) < 0.03, (np.mean(weighted), true)
    assert sum(covered) >= 24, f"the interval covered the truth in {sum(covered)} of 30"


def test_the_per_recording_split_is_weighted_the_same_way(tmp_path):
    """One recording holds the whole run, so its row must be the headline — a split that read
    the plain share would disagree with the figure above it."""
    rows, _scores, truth = _skewed()
    _audit_two_rounds(
        tmp_path, rows, {r.instance_id: bool(t) for r, t in zip(rows, truth)}, seed=1
    )
    manifest = _manifest("higher_is_better")
    head = audit.precision_at_cutoff(
        tmp_path, rows, cutoff=_SKEW_CUTOFF, manifest=manifest
    )
    (split,) = audit.precision_by_recording(
        tmp_path, rows, cutoff=_SKEW_CUTOFF, manifest=manifest
    )
    assert split["precision"] == head["precision"]
    assert split["interval"] == head["interval"]


def test_a_draw_over_several_behaviors_bins_each_on_its_own_scores():
    """Pooled edges put a behavior whose scores sit in 0-0.1 into one bin of a range that runs to
    1.0, so a draw "across ten bins" drew from one of its."""
    low = [_instance(0.001 * i, i, source_label="q01") for i in range(100)]
    high = [
        _instance(0.5 + 0.005 * i, 1000 + i, source_label="q02") for i in range(100)
    ]
    drawn = audit.draw_sample(low + high, budget=40)
    for label, rows in (("q01", low), ("q02", high)):
        edges = audit.bin_edges([r.score for r in rows])
        bins = {audit._bin_of(r.score, edges) for r in drawn if r.source_label == label}
        assert bins == set(range(10)), (label, sorted(bins))


def test_bins_are_recorded_at_the_first_draw_and_never_move(tmp_path):
    import numpy as np

    first = audit.fix_bins(tmp_path, {"q01": np.linspace(0.0, 1.0, 11)}, seed=3)
    text = (tmp_path / bo.AUDIT_BINS_FILENAME).read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1\n")

    again = audit.fix_bins(
        tmp_path,
        {"q01": np.linspace(0.0, 2.0, 11), "q02": np.linspace(0.5, 0.9, 11)},
        seed=4,
    )
    assert np.array_equal(again["q01"], first["q01"]), "a recorded label's bins moved"
    assert np.array_equal(again["q02"], np.linspace(0.5, 0.9, 11))
    stored = audit.read_bins(tmp_path)
    assert set(stored) == {"q01", "q02"}
    assert np.array_equal(stored["q01"], first["q01"])


def test_the_estimate_cuts_where_the_recorded_bins_cut(tmp_path):
    """Given edges, the estimate weights by them rather than re-deriving its own from whatever
    rows it was handed — the same bins the draw used."""
    import numpy as np

    rows = _instances([0.1, 0.2, 0.3, 0.9])
    audit.append_audit(tmp_path, rows[0].instance_id, score=0.1, human_label=True)
    audit.append_audit(tmp_path, rows[3].instance_id, score=0.9, human_label=False)
    manifest = _manifest("higher_is_better")
    # One bin over everything: the plain share of the two judged clips.
    one = audit.precision_at_cutoff(
        tmp_path,
        rows,
        cutoff=None,
        manifest=manifest,
        edges={"q01": np.array([0.0, 1.0])},
    )
    # Two bins: three instances weigh the judged yes, one the judged no.
    two = audit.precision_at_cutoff(
        tmp_path,
        rows,
        cutoff=None,
        manifest=manifest,
        edges={"q01": np.array([0.0, 0.5, 1.0])},
    )
    assert one["precision"] == 0.5
    assert two["precision"] == 0.75


def test_clips_drawn_before_the_bins_were_recorded_are_counted_and_said(tmp_path):
    """A draw made before bins were recorded per behavior — one spanning every behavior cut its
    bins over all of them — is weighted by the recorded bins all the same (post-stratified), and
    the figure says how many of its clips that covers."""
    import numpy as np

    from detail.storage import audit_figures

    rows = _instances([0.1, 0.2, 0.3, 0.9])
    audit.append_audit(
        tmp_path, rows[0].instance_id, score=0.1, human_label=True,
        drawn_at="2020-01-01T00:00:00Z",
    )  # fmt: skip
    audit.append_audit(
        tmp_path, rows[1].instance_id, score=0.2, human_label=True,
        drawn_at="2020-01-01T00:00:00Z",
    )  # fmt: skip
    manifest = _manifest("higher_is_better")
    before = audit.precision_at_cutoff(tmp_path, rows, cutoff=None, manifest=manifest)
    assert (
        before["n_drawn_before_bins"] == 2
    ), "no bins recorded: every clip predates them"

    audit.fix_bins(tmp_path, {"q01": np.linspace(0.1, 0.9, 11)}, seed=0)
    audit.append_audit(tmp_path, rows[3].instance_id, score=0.9, human_label=False)
    after = audit.precision_at_cutoff(tmp_path, rows, cutoff=None, manifest=manifest)
    assert after["n_drawn_before_bins"] == 2
    assert after["n"] == 3
    assert any(
        note.startswith("2 audited clips were drawn before")
        for note in audit_figures.notes(after)
    )
