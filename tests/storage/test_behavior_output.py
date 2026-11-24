"""Tests for the §19 behavior output artifact (detail/storage/behavior_output.py).

The three files are one artifact, and most of what is tested here is a claim the artifact
makes about itself rather than a value it stores: that a bout's id survives reprojection (or
every audited decision silently re-points), that a two-label cluster is countable without
double counting, and that an instance can never sit outside the coverage table that is
supposed to say what was analyzed.

**The class: a lookup that misses every key, reported as the value zero.** ``0s`` meant two
different things — "this cluster accounts for no time", which is real and worth seeing, and "the
artifact was projected under a previous id space", which is every run interpreted before the ids
changed. The second renders as ``0s`` on *every* row, beneath real row counts, so it reads as a
finding about the data rather than as a stale projection — and it repairs itself invisibly on the
next reprojection, so nobody ever catches it in the act.

Both sum to zero, so a total cannot tell them apart: only the **keys** separate them. The check is
therefore an **intersection** of the two id sets, and the empty artifact is its schema edge.
"""

import csv
from dataclasses import fields
from unittest.mock import patch

import pytest
import yaml

from detail.storage import behavior_output as bo
from detail.storage import experiment as experiment_store


def _instance(**kw) -> bo.BehaviorInstance:
    data = dict(
        recording_id="recording_001",
        subject=0,
        t_ns=1_000,
        end_t_ns=6_000,
        start_frame=30,
        end_frame=180,
        source_label="q01",
    )
    data.update(kw)
    return bo.BehaviorInstance(**data)


def _coverage(**kw) -> bo.CoverageRow:
    data = dict(
        recording_id="recording_001",
        subject=0,
        start_t_ns=0,
        end_t_ns=600_000,
        observed_ns=540_000,
        n_observed=5400,
    )
    data.update(kw)
    return bo.CoverageRow(**data)


def _manifest(**kw) -> dict:
    data = dict(
        producer={"module_id": "clip_query", "run_id": "an_07"},
        origin="predicted",
        label_space="pooled",
        vocabulary=["rearing"],
        names={"q01": ["rearing"]},
        score={
            "meaning": "cosine distance to exemplar",
            "direction": "lower_is_better",
        },
        overlap="allowed",
        behavior_types={"rearing": "state"},
    )
    data.update(kw)
    return data


def _write(tmp_path, *, instances=None, coverage=None, manifest=None):
    return bo.write_behavior_output(
        tmp_path,
        instances=[_instance()] if instances is None else instances,
        coverage=[_coverage()] if coverage is None else coverage,
        manifest=_manifest() if manifest is None else manifest,
    )


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_row_dataclasses_match_their_columns():
    # Every row schema's fields == its COLUMNS list, so a write/read cannot drop a field.
    assert {f.name for f in fields(bo.BehaviorInstance)} == set(bo.INSTANCE_COLUMNS)
    assert {f.name for f in fields(bo.CoverageRow)} == set(bo.COVERAGE_COLUMNS)


def test_headers_are_exactly_the_column_lists(tmp_path):
    _write(tmp_path)
    with open(tmp_path / bo.INSTANCES_FILENAME, newline="", encoding="utf-8-sig") as f:
        assert next(csv.reader(f)) == bo.INSTANCE_COLUMNS
    with open(tmp_path / bo.COVERAGE_FILENAME, newline="", encoding="utf-8-sig") as f:
        assert next(csv.reader(f)) == bo.COVERAGE_COLUMNS


def test_manifest_is_schema_versioned_first(tmp_path):
    _write(tmp_path)
    text = (tmp_path / bo.MANIFEST_FILENAME).read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == "schema_version: 1"  # Tenet 7
    assert yaml.safe_load(text)["schema_version"] == 1


def test_round_trip(tmp_path):
    _write(tmp_path)
    back = bo.read_behavior_output(tmp_path)
    # `label_path` comes back **filled from the label**, because a producer with no hierarchy
    # still has a one-level path and an empty cell there would mean two different things: "this
    # producer has no hierarchy" and "this bout is unlabelled".
    assert back.instances == [_instance(label="rearing", label_path="rearing")]
    assert back.coverage == [_coverage()]
    assert back.manifest["producer"] == {"module_id": "clip_query", "run_id": "an_07"}


def test_a_one_frame_bout_survives_the_round_trip(tmp_path):
    # Both ends inclusive: a bout on a single frame is start == end, not a zero-length span.
    row = _instance(t_ns=5_000, end_t_ns=5_000, start_frame=0, end_frame=0)
    _write(tmp_path, instances=[row])
    back = bo.read_behavior_output(tmp_path).instances[0]
    assert (back.t_ns, back.end_t_ns) == (5_000, 5_000)
    assert (back.start_frame, back.end_frame) == (0, 0)  # frame 0 is a real frame


def test_a_label_containing_a_comma_round_trips(tmp_path):
    manifest = _manifest(
        vocabulary=["rearing, supported"], names={"q01": ["rearing, supported"]}
    )
    _write(tmp_path, manifest=manifest)
    assert bo.read_behavior_output(tmp_path).instances[0].label == "rearing, supported"


# --------------------------------------------------------------------------- #
# instance identity, and the double count it exists to make countable
# --------------------------------------------------------------------------- #
def test_instance_id_is_stable_across_a_reprojection(tmp_path):
    """The audit-dangling guard.

    ``audit.csv`` keys human decisions by ``instance_id`` and the artifact is rewritten every
    time a user drags a cutoff or names a cluster. If the id moved with row position, every
    stored decision would silently re-point at a different bout after the first reprojection.

    Each projection builds its instances **fresh**, the way a producer re-derives them from
    its native output. Reusing one Python object across both writes would carry its id along
    and the test would pass against a row counter — it did, until this comment existed.
    """

    def survivor():
        return _instance(t_ns=9_000, end_t_ns=14_000, score=0.31)

    def dropped():
        return _instance(t_ns=1_000, end_t_ns=6_000, score=0.42)

    _write(tmp_path, instances=[dropped(), survivor()])
    before = {
        row.t_ns: row.instance_id for row in bo.read_behavior_output(tmp_path).instances
    }

    # A tighter cutoff drops the first bout and the user renames the cluster. The surviving
    # bout is now row 0 rather than row 1: its id must not have moved with it.
    _write(
        tmp_path,
        instances=[survivor()],
        manifest=_manifest(vocabulary=["grooming"], names={"q01": ["grooming"]}),
    )
    after = {
        row.t_ns: row.instance_id for row in bo.read_behavior_output(tmp_path).instances
    }
    assert after[9_000] == before[9_000]
    assert after[9_000] == bo.instance_id_for("recording_001", 0, "q01", 9_000, 14_000)


def test_a_two_label_cluster_is_two_rows_and_one_bout(tmp_path):
    """Rows are label assignments; bouts are what a duration sum must count.

    Summing durations over these two rows double counts the same five seconds, and nothing
    about the arithmetic looks wrong when it does.
    """
    manifest = _manifest(
        vocabulary=["rearing", "near wall"],
        names={"q01": ["rearing", "near wall"]},
    )
    _write(tmp_path, manifest=manifest)
    back = bo.read_behavior_output(tmp_path)
    assert len(back.instances) == 2
    assert sorted(row.label for row in back.instances) == ["near wall", "rearing"]
    assert back.bout_count == 1
    assert len({row.instance_id for row in back.instances}) == 1


def test_reprojecting_the_artifact_does_not_multiply_its_rows(tmp_path):
    """§21 rewrites this artifact on every authored change, and a rewrite reads it back first.

    Expansion fans a bout out per label, so fanning an already-fanned row out again gives
    ``labels x labels`` rows for one bout. Nothing downstream notices: ``bout_count`` counts
    distinct ``instance_id`` and keeps reporting one while ``instances.csv`` grows every pass.
    """
    manifest = _manifest(
        vocabulary=["rearing", "near wall"],
        names={"q01": ["rearing", "near wall"]},
    )
    _write(tmp_path, manifest=manifest)
    for _ in range(3):
        back = bo.read_behavior_output(tmp_path)
        bo.write_behavior_output(
            tmp_path,
            instances=back.instances,
            coverage=back.coverage,
            manifest=manifest,
        )
    final = bo.read_behavior_output(tmp_path)
    assert len(final.instances) == 2  # not 2 -> 4 -> 16 -> 256
    assert final.bout_count == 1
    assert sorted(row.label for row in final.instances) == ["near wall", "rearing"]


def test_bouts_collapses_a_twice_named_cluster_without_reordering(tmp_path):
    # The shared collapse every consumer must run before it counts. Order is the input's,
    # because `_expand` promises the producer's order survives a write.
    manifest = _manifest(
        vocabulary=["rearing", "near wall"],
        names={"q01": ["rearing", "near wall"]},
    )
    rows = [
        _instance(t_ns=9_000, end_t_ns=11_000, source_label="q01"),
        _instance(t_ns=1_000, end_t_ns=6_000, source_label="q01"),
    ]
    _write(tmp_path, instances=rows, manifest=manifest)
    read_back = bo.read_behavior_output(tmp_path)
    assert len(read_back.instances) == 4  # two bouts x two labels
    collapsed = bo.bouts(read_back.instances)
    assert len(collapsed) == 2
    assert [row.t_ns for row in collapsed] == [
        9_000,
        1_000,
    ]  # producer order, not sorted


def test_an_unnamed_cluster_keeps_its_source_label_and_an_empty_label(tmp_path):
    # A discovered cluster has an id before it has a name and may never get one.
    manifest = _manifest(vocabulary=[], names={})
    _write(tmp_path, instances=[_instance(source_label="7")], manifest=manifest)
    row = bo.read_behavior_output(tmp_path).instances[0]
    assert row.label == "" and row.source_label == "7"


def test_a_producer_set_label_survives_an_empty_naming_map(tmp_path):
    # Ground truth arrives already labeled and carries no naming map; expansion must not
    # wipe the labels it came with.
    manifest = _manifest(
        origin="ground_truth", vocabulary=["rearing"], names={}, score=None
    )
    instances = [_instance(label="rearing", score=None)]
    _write(tmp_path, instances=instances, manifest=manifest)
    assert bo.read_behavior_output(tmp_path).instances[0].label == "rearing"


def test_an_integer_cluster_id_names_the_same_bout_as_its_string(tmp_path):
    # YAML round-trips a cluster id as whatever it was written as; the lookup must not miss
    # on 7 vs "7".
    manifest = _manifest(vocabulary=["rearing"], names={7: "rearing"})
    _write(tmp_path, instances=[_instance(source_label="7")], manifest=manifest)
    assert bo.read_behavior_output(tmp_path).instances[0].label == "rearing"


# --------------------------------------------------------------------------- #
# coverage: absence is not zero
# --------------------------------------------------------------------------- #
def test_a_covered_recording_with_no_instances_is_a_real_zero(tmp_path):
    # The whole point of coverage: "we looked and it never happened" is a row in coverage
    # and no rows in instances, which is different from the recording being absent.
    _write(tmp_path, instances=[], coverage=[_coverage(), _coverage(subject=1)])
    back = bo.read_behavior_output(tmp_path)
    assert back.instances == []
    assert {row.unit for row in back.coverage} == {
        ("recording_001", 0),
        ("recording_001", 1),
    }


def test_an_instance_outside_coverage_is_refused(tmp_path):
    with pytest.raises(bo.BehaviorOutputError, match="no coverage.csv row"):
        _write(tmp_path, instances=[_instance(recording_id="recording_002")])


def test_duplicate_coverage_rows_are_refused(tmp_path):
    with pytest.raises(bo.BehaviorOutputError, match="more than one row"):
        _write(tmp_path, coverage=[_coverage(), _coverage()])


def test_observed_ns_is_stored_beside_the_span_it_is_the_denominator_for(tmp_path):
    _write(tmp_path, coverage=[_coverage(end_t_ns=600_000, observed_ns=420_000)])
    row = bo.read_behavior_output(tmp_path).coverage[0]
    # Tracked time is not recording duration — the gap is the point of the column.
    assert row.observed_ns == 420_000 and row.end_t_ns - row.start_t_ns == 600_000


def test_an_instance_outside_its_coverage_span_is_refused(tmp_path):
    # Being *in* a covered unit is not being inside the covered span. Accepting this wrote a
    # well-formed artifact whose time_fraction came out at 20.0 and raised nothing.
    with pytest.raises(bo.BehaviorOutputError, match="only analyzed over"):
        _write(
            tmp_path,
            instances=[_instance(t_ns=700_000, end_t_ns=800_000)],
            coverage=[_coverage(start_t_ns=0, end_t_ns=600_000)],
        )


def test_an_instance_may_touch_either_end_of_its_coverage_span(tmp_path):
    # Both ends inclusive, so the boundary cases are covered, not refused — a containment
    # check written with the wrong comparison would reject a legitimate edge bout.
    _write(
        tmp_path,
        instances=[_instance(t_ns=0, end_t_ns=600_000)],
        coverage=[_coverage(start_t_ns=0, end_t_ns=600_000)],
    )
    assert bo.read_behavior_output(tmp_path).bout_count == 1


def test_coverage_cannot_observe_more_time_than_its_own_span(tmp_path):
    # observed_ns is time seen *inside* [start_t_ns, end_t_ns] and is the denominator every
    # rate ships with; one larger than its own span deflates every rate built on it.
    with pytest.raises(bo.BehaviorOutputError, match="ns of a span only"):
        _write(tmp_path, coverage=[_coverage(end_t_ns=1_000, observed_ns=999_999)])


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_a_label_outside_the_vocabulary_is_refused(tmp_path):
    manifest = _manifest(vocabulary=["grooming"], names={"q01": ["rearing"]})
    with pytest.raises(bo.BehaviorOutputError, match="not in the manifest vocabulary"):
        _write(tmp_path, manifest=manifest)


def test_a_producer_set_label_outside_the_vocabulary_is_refused(tmp_path):
    # The other half of the vocabulary check: a label the producer put on the row itself,
    # rather than one reached through the naming map. Only the naming half was covered.
    manifest = _manifest(vocabulary=["rearing"], names={})
    with pytest.raises(bo.BehaviorOutputError, match="not in the manifest vocabulary"):
        _write(tmp_path, instances=[_instance(label="grooming")], manifest=manifest)


def _overlapping_pair():
    return [
        _instance(t_ns=1_000, end_t_ns=6_000, source_label="q01"),
        _instance(t_ns=5_000, end_t_ns=9_000, source_label="q02"),
    ]


_TWO_SOURCES = {"q01": ["rearing"], "q02": ["rearing"]}


def test_exclusive_overlap_is_enforced_against_the_rows(tmp_path):
    # The manifest's own claim, checked. §23 reads `exclusive` as "at most one label active at
    # a time" and refuses `transitions` without it, so a stored `exclusive` the rows contradict
    # does not fail — it builds a transition table on an undefined "next label".
    with pytest.raises(bo.BehaviorOutputError, match="overlap: exclusive"):
        _write(
            tmp_path,
            instances=_overlapping_pair(),
            manifest=_manifest(overlap="exclusive", names=_TWO_SOURCES),
        )


def test_overlap_allowed_accepts_exactly_what_exclusive_refuses(tmp_path):
    # The paired negative: the same rows, and only the declaration differs. Without this the
    # check above could pass by refusing overlapping spans unconditionally.
    _write(
        tmp_path,
        instances=_overlapping_pair(),
        manifest=_manifest(overlap="allowed", names=_TWO_SOURCES),
    )
    assert bo.read_behavior_output(tmp_path).bout_count == 2


def test_exclusive_overlap_is_judged_per_unit(tmp_path):
    # Two subjects may occupy the same wall-clock span — they are separate units, and judging
    # them together would refuse every multi-subject artifact that declares exclusivity.
    rows = [
        _instance(subject=0, t_ns=1_000, end_t_ns=6_000),
        _instance(subject=1, t_ns=1_000, end_t_ns=6_000),
    ]
    _write(
        tmp_path,
        instances=rows,
        coverage=[_coverage(subject=0), _coverage(subject=1)],
        manifest=_manifest(overlap="exclusive"),
    )
    assert bo.read_behavior_output(tmp_path).bout_count == 2


def test_scores_without_a_declared_direction_are_refused(tmp_path):
    with pytest.raises(bo.BehaviorOutputError, match="score.direction"):
        _write(
            tmp_path,
            instances=[_instance(score=0.4)],
            manifest=_manifest(score={"meaning": "distance"}),
        )


def test_an_unscored_artifact_needs_no_direction(tmp_path):
    # Human-authored rows have no score, so the block is optional when nothing carries one.
    _write(
        tmp_path,
        instances=[_instance(score=None)],
        manifest=_manifest(origin="ground_truth", score=None),
    )
    assert bo.read_behavior_output(tmp_path).instances[0].score is None


@pytest.mark.parametrize(
    "manifest_kw, fragment",
    [
        ({"producer": {"module_id": "clip_query"}}, "producer.run_id"),
        ({"origin": "guessed"}, "origin must be one of"),
        ({"label_space": "per_animal"}, "label_space must be one of"),
        ({"overlap": "sometimes"}, "overlap must be one of"),
        ({"behavior_types": {"rearing": "episodic"}}, "behavior_types"),
    ],
)
def test_manifest_domain_errors(tmp_path, manifest_kw, fragment):
    with pytest.raises(bo.BehaviorOutputError, match=fragment):
        _write(tmp_path, manifest=_manifest(**manifest_kw))


def test_an_inverted_span_is_refused(tmp_path):
    with pytest.raises(bo.BehaviorOutputError, match="ends before it starts"):
        _write(tmp_path, instances=[_instance(t_ns=6_000, end_t_ns=1_000)])


def test_nothing_is_written_when_validation_fails(tmp_path):
    # Tenet 10: validate everything before writing anything. A refused write must not leave
    # a half-artifact that discovery would pick up.
    with pytest.raises(bo.BehaviorOutputError):
        _write(tmp_path, instances=[_instance(recording_id="recording_002")])
    assert not (tmp_path / bo.INSTANCES_FILENAME).exists()
    assert not bo.has_behavior_output(tmp_path)


# --------------------------------------------------------------------------- #
# class guard: a writer checks its values, not only its vocabulary
# --------------------------------------------------------------------------- #
# The class: a writer that validates every field is *present and in its enum*, and never that
# the numbers agree with each other. Before this table `_validate` ran nine checks and every
# one was a domain check, so an instance could sit entirely outside the span its own coverage
# row declared, write clean, and yield a time_fraction of 20.0.
#
# Each row is (what the artifact claims, the fragment that must name it). A second writer of
# this shared artifact adds its rows here rather than growing a private test — the sibling that
# skipped these checks is exactly the one that had to be found by review.
#
# Non-vacuity: each case differs from a writable artifact only in the value under test, and the
# paired positives sit beside the individual checks above
# (`test_an_instance_may_touch_either_end_of_its_coverage_span`,
# `test_overlap_allowed_accepts_exactly_what_exclusive_refuses`).
_INCONSISTENT_VALUES = [
    pytest.param(
        dict(instances=[_instance(t_ns=-5_000, end_t_ns=6_000)]),
        "only analyzed over",
        id="instance-starts-before-its-coverage",
    ),
    pytest.param(
        dict(instances=[_instance(t_ns=1_000, end_t_ns=900_000)]),
        "only analyzed over",
        id="instance-ends-after-its-coverage",
    ),
    pytest.param(
        dict(coverage=[_coverage(end_t_ns=1_000, observed_ns=999_999)]),
        "ns of a span only",
        id="observes-more-time-than-its-span-is-long",
    ),
    pytest.param(
        dict(coverage=[_coverage(start_t_ns=600_000, end_t_ns=0)]),
        "ends before it starts",
        id="inverted-coverage-span",
    ),
    pytest.param(
        dict(instances=[_instance(t_ns=6_000, end_t_ns=1_000)]),
        "ends before it starts",
        id="inverted-instance-span",
    ),
    pytest.param(
        dict(
            instances=_overlapping_pair(),
            manifest=_manifest(overlap="exclusive", names=_TWO_SOURCES),
        ),
        "overlap: exclusive",
        id="rows-contradict-the-declared-exclusivity",
    ),
]


@pytest.mark.parametrize("kwargs,fragment", _INCONSISTENT_VALUES)
def test_an_artifact_whose_numbers_disagree_is_refused(tmp_path, kwargs, fragment):
    with pytest.raises(bo.BehaviorOutputError, match=fragment):
        _write(tmp_path, **kwargs)
    assert not (tmp_path / bo.MANIFEST_FILENAME).exists()  # refused before any byte


# --------------------------------------------------------------------------- #
# label_space synonym
# --------------------------------------------------------------------------- #
def test_the_segmentation_label_space_word_is_normalized(tmp_path):
    # Older fits wrote "global" for what §19 calls "pooled".
    _write(tmp_path, manifest=_manifest(label_space="global"))
    assert (
        yaml.safe_load(
            (tmp_path / bo.MANIFEST_FILENAME).read_text(encoding="utf-8-sig")
        )["label_space"]
        == "pooled"
    )
    assert bo.read_behavior_output(tmp_path).manifest["label_space"] == "pooled"


def test_a_per_recording_label_space_is_kept(tmp_path):
    _write(tmp_path, manifest=_manifest(label_space="per_recording"))
    assert bo.read_behavior_output(tmp_path).manifest["label_space"] == "per_recording"


# --------------------------------------------------------------------------- #
# audited is derived, never stored by the producer
# --------------------------------------------------------------------------- #
def test_audited_is_false_without_an_audit_file(tmp_path):
    _write(tmp_path)
    assert bo.read_behavior_output(tmp_path).audited is False


def test_audited_is_derived_from_the_audit_file_not_from_the_producer(tmp_path):
    # A producer claiming audited=True with no audit.csv is overruled: the fact is checkable
    # from the store, so a stored copy of it is a second writer on one fact.
    (tmp_path).mkdir(parents=True, exist_ok=True)
    _write(tmp_path, manifest=_manifest(audited=True))
    assert bo.read_behavior_output(tmp_path).audited is False

    (tmp_path / bo.AUDIT_FILENAME).write_text(
        "instance_id,score,human_label,drawn_at\n", encoding="utf-8"
    )
    _write(tmp_path, manifest=_manifest(audited=False))
    assert bo.read_behavior_output(tmp_path).audited is True


def test_a_review_of_every_clip_counts_as_audited(tmp_path):
    """A run reviewed clip by clip (``clip_review.csv``, §19/§20/§22) and never audited one by
    one is audited: reading ``audit.csv`` alone reported it as never checked."""
    from detail.storage import clip_review
    from detail.storage.clip_edits import ClipRef

    _write(tmp_path)
    assert bo.read_behavior_output(tmp_path).audited is False
    clip_review.append_answers(
        tmp_path,
        [(ClipRef("recording_001", 0, 10, 40), True)],
        node=None,
        label_path="q01",
    )
    _write(tmp_path)
    assert bo.read_behavior_output(tmp_path).audited is True


# --------------------------------------------------------------------------- #
# atomicity
# --------------------------------------------------------------------------- #
def test_write_is_atomic_no_tmp_left(tmp_path):
    _write(tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_a_crash_mid_manifest_leaves_the_previous_artifact_intact(tmp_path):
    _write(tmp_path)
    with patch(
        "detail.storage.behavior_output.yaml.dump", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            _write(tmp_path, manifest=_manifest(vocabulary=["rearing"]))
    assert bo.read_behavior_output(tmp_path).manifest["schema_version"] == 1


# --------------------------------------------------------------------------- #
# tolerant reader
# --------------------------------------------------------------------------- #
# The literal header of an artifact written before the readability columns existed — spelled
# out rather than derived from INSTANCE_COLUMNS so it stays a real historical file.
_NO_FRAMES_HEADER = (
    "recording_id,subject,t_ns,end_t_ns,label,source_label,instance_id,score"
)
_NO_FRAMES_ROW = "recording_001,0,1000,6000,rearing,q01,abc123,0.4"


def test_read_tolerates_a_csv_missing_an_optional_column(tmp_path):
    _write(tmp_path)
    (tmp_path / bo.INSTANCES_FILENAME).write_text(
        _NO_FRAMES_HEADER + "\n" + _NO_FRAMES_ROW + "\n", encoding="utf-8"
    )
    row = bo.read_behavior_output(tmp_path).instances[0]
    assert row.t_ns == 1_000 and row.label == "rearing" and row.score == 0.4
    # The frames were not recorded; -1 says so rather than claiming frame 0.
    assert (row.start_frame, row.end_frame) == (-1, -1)


def test_read_keeps_an_instance_id_written_by_an_older_scheme(tmp_path):
    # Audit decisions join on this id, so a stored one is preserved rather than recomputed.
    _write(tmp_path)
    (tmp_path / bo.INSTANCES_FILENAME).write_text(
        _NO_FRAMES_HEADER + "\n" + _NO_FRAMES_ROW + "\n", encoding="utf-8"
    )
    assert bo.read_behavior_output(tmp_path).instances[0].instance_id == "abc123"


def test_read_missing_a_required_column_raises(tmp_path):
    _write(tmp_path)
    (tmp_path / bo.INSTANCES_FILENAME).write_text(
        "label,score\nrearing,0.4\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing columns"):
        bo.read_behavior_output(tmp_path)


def test_read_without_a_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=bo.MANIFEST_FILENAME):
        bo.read_behavior_output(tmp_path)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def test_discovery_scans_for_the_manifest(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    analysis = experiment_store.analysis_dir(proj)
    _write(analysis / "an_01")
    # A run that produced native output but no artifact — an interrupted run cannot honestly
    # state coverage, so it is simply not a behavior output.
    (analysis / "an_02").mkdir(parents=True, exist_ok=True)
    (analysis / "an_02" / bo.INSTANCES_FILENAME).write_text(
        "recording_id\n", encoding="utf-8"
    )
    assert bo.discover_behavior_outputs(proj) == ["an_01"]


def test_discovery_orders_past_the_id_pad(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    analysis = experiment_store.analysis_dir(proj)
    for run_id in ("an_9", "an_10", "an_100"):
        _write(analysis / run_id)
    assert bo.discover_behavior_outputs(proj) == ["an_9", "an_10", "an_100"]


def test_discovery_of_an_experiment_without_an_analysis_dir_is_empty(tmp_path):
    assert bo.discover_behavior_outputs(tmp_path / "nope") == []


# --------------------------------------------------------------------------- #
# the artifact declaration producers reuse
# --------------------------------------------------------------------------- #
def test_artifact_manifest_names_every_component(tmp_path):
    paths = {entry["path"] for entry in bo.behavior_output_artifacts()}
    assert paths == {
        bo.INSTANCES_FILENAME,
        bo.COVERAGE_FILENAME,
        bo.MANIFEST_FILENAME,
        bo.FRAMES_DIRNAME,
        bo.FEATURES_FILENAME,
    }
    _write(tmp_path)
    # The three files this writer owns. Two components are written *by the producer*, because
    # only a producer has what they need: the frame view needs a time axis
    # (``storage/frame_view.py``), and the features need the embedding run the rows came from
    # (``pipeline/behavior_artifact.write_instance_features``).
    producer_written = {bo.FRAMES_DIRNAME, bo.FEATURES_FILENAME}
    for entry in bo.behavior_output_artifacts():
        if entry["path"] not in producer_written:
            assert (tmp_path / entry["path"]).is_file()


# --------------------------------------------------------------------------- #
# the artifact-to-stream projection, and the boundary it carries
# --------------------------------------------------------------------------- #
# §19 makes re-ingest a deliberate user action rather than an automatic write, because that is
# what keeps the prediction/ground-truth boundary structural rather than remembered. The whole
# leakage rule rests on the `origin` column this writes, and it had no test at all.
def _projected(
    tmp_path, *, instances=None, manifest=None, recording_id="recording_001"
):
    _write(tmp_path, instances=instances, manifest=manifest)
    out = tmp_path / "stream.csv"
    bo.write_stream_csv(out, bo.read_behavior_output(tmp_path), recording_id)
    with open(out, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_the_stream_projection_carries_the_artifact_origin(tmp_path):
    # A predicted artifact must re-ingest as a stream that declares itself, or a prediction
    # becomes readable as ground truth the moment someone imports it back.
    rows = _projected(tmp_path, manifest=_manifest(origin="predicted"))
    assert rows and {row["origin"] for row in rows} == {"predicted"}


def test_a_ground_truth_artifact_projects_as_ground_truth(tmp_path):
    # The paired direction: the column is the artifact's own value, not a constant. Without
    # this, the test above passes just as well against a hardcoded "predicted".
    rows = _projected(tmp_path, manifest=_manifest(origin="ground_truth"))
    assert rows and {row["origin"] for row in rows} == {"ground_truth"}


def test_the_projection_is_readable_by_the_importer_it_targets(tmp_path):
    """Its whole purpose is to be re-imported, so assert the real importer takes it.

    A CSV that merely has the right column names is the failure a header test cannot see.
    """
    from detail.importers.behavior import BehaviorCsvImporter

    _write(tmp_path, manifest=_manifest(origin="predicted"))
    out = tmp_path / "stream.csv"
    bo.write_stream_csv(out, bo.read_behavior_output(tmp_path), "recording_001")

    importer = BehaviorCsvImporter()
    assert importer.can_import(out)
    result = importer.import_stream(out, tmp_path / "imported", {})
    assert result.metadata.get("origin") == "predicted"


def test_the_projection_keeps_only_the_named_rows(tmp_path):
    # A behavior stream's label is required, and an unnamed cluster has nothing to call
    # itself, so it is dropped rather than written with an empty label.
    rows = _projected(
        tmp_path,
        instances=[
            _instance(t_ns=1_000, end_t_ns=6_000, source_label="q01"),
            _instance(t_ns=7_000, end_t_ns=9_000, source_label="unnamed"),
        ],
        manifest=_manifest(names={"q01": ["rearing"]}),
    )
    assert [row["label"] for row in rows] == ["rearing"]


# --- how much time a producer's label accounts for ------------------------------------------


def test_the_period_is_observed_time_over_rows():
    coverage = [
        bo.CoverageRow(
            recording_id="r1",
            subject=0,
            start_t_ns=0,
            end_t_ns=1_000,
            observed_ns=800,
            n_observed=8,
        ),
    ]
    assert bo.period_by_recording(coverage) == {"r1": 100}


# --------------------------------------------------------------------------- #
# instance_features.csv — what we measured about an instance
# --------------------------------------------------------------------------- #
def test_instance_features_round_trip(tmp_path):
    bo.write_instance_features(tmp_path, {"aaa": (0.25, 0.75), "bbb": (0.5, 0.5)})
    assert bo.read_instance_features(tmp_path) == {
        "aaa": (0.25, 0.75),
        "bbb": (0.5, 0.5),
    }


def test_an_unknown_position_is_an_empty_cell_not_the_string_nan(tmp_path):
    """Every other nullable column in this artifact writes an empty cell for "not known", and a
    literal ``nan`` reads back as a float that compares unequal to itself in whatever consumes
    it."""
    bo.write_instance_features(tmp_path, {"aaa": (float("nan"), float("nan"))})
    text = (tmp_path / bo.FEATURES_FILENAME).read_text(encoding="utf-8-sig")
    assert "nan" not in text.lower()
    assert bo.read_instance_features(tmp_path) == {"aaa": (None, None)}


def test_absent_features_and_empty_features_are_different_answers(tmp_path):
    """An artifact written before features existed has **no file**; one whose producer found
    nothing to place has an **empty file**. Only the first is actionable — `recording
    backfill-kinematics` fixes it — so a reader has to be able to tell them apart.

    They used to share one encoding (`{}` for both), which is the recorded class in
    `test_behavior_audit.py`: every number stayed right while the surface adjusted a setting
    that was never the problem.
    """
    _write(tmp_path)  # a complete artifact, no features file
    assert not (tmp_path / bo.FEATURES_FILENAME).exists()
    assert bo.read_instance_features(tmp_path) is None  # absent

    bo.write_instance_features(tmp_path, {})  # the producer looked and found nothing
    assert (tmp_path / bo.FEATURES_FILENAME).is_file()
    assert bo.read_instance_features(tmp_path) == {}  # empty, and not the same answer


def test_features_are_written_in_a_stable_order(tmp_path):
    """A reprojection that changes nothing rewrites a byte-identical file, so a diff over an
    experiment shows only what really moved."""
    bo.write_instance_features(tmp_path, {"ccc": (0.1, 0.1), "aaa": (0.2, 0.2)})
    first = (tmp_path / bo.FEATURES_FILENAME).read_bytes()
    bo.write_instance_features(tmp_path, {"aaa": (0.2, 0.2), "ccc": (0.1, 0.1)})
    assert (tmp_path / bo.FEATURES_FILENAME).read_bytes() == first
