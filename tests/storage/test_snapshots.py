"""Tests for pinned artifact copies (detail/storage/snapshots.py, §21).

The bug this module exists for is invisible to a behavioural test of anything else: an authored
run's artifact is reprojected on every edit, so ``run_id`` names a mutable slot rather than a
result, and a comparison that read it last week can disagree with one that reads it today while
both stay well-formed. What is asserted here is therefore mostly about *identity* — that a pin
freezes content rather than referencing it, that pinning twice without an edit does not mint a
second copy, and that a half-written snapshot is not discoverable.
"""

import yaml

import pytest

from detail.storage import behavior_output as bo
from detail.storage import snapshots


def _instance(**kw) -> bo.BehaviorInstance:
    data = dict(
        recording_id="recording_001",
        subject=0,
        t_ns=1_000,
        end_t_ns=6_000,
        start_frame=30,
        end_frame=180,
        source_label="c01",
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
        producer={"module_id": "temporal_clustering", "run_id": "an_15"},
        origin="predicted",
        label_space="pooled",
        vocabulary=["rearing"],
        names={"c01": ["rearing"]},
        overlap="exclusive",
        behavior_types={"rearing": "state"},
    )
    data.update(kw)
    return data


def _write(run_dir, *, instances=None, manifest=None):
    return bo.write_behavior_output(
        run_dir,
        instances=[_instance()] if instances is None else instances,
        coverage=[_coverage()],
        manifest=_manifest() if manifest is None else manifest,
    )


# --------------------------------------------------------------------------- #
# the mint rule
# --------------------------------------------------------------------------- #
def test_a_pin_freezes_the_artifact_rather_than_referencing_it(tmp_path):
    """The whole point: the copy must not move when the run is reprojected."""
    _write(tmp_path)
    sid = snapshots.pin(tmp_path, reason="finish")

    _write(tmp_path, instances=[_instance(t_ns=2_000, end_t_ns=7_000)])

    frozen = bo.read_behavior_output(snapshots.snapshot_dir(tmp_path, sid))
    assert [i.t_ns for i in frozen.instances] == [1_000]
    assert [i.t_ns for i in bo.read_behavior_output(tmp_path).instances] == [2_000]


def test_pinning_twice_without_an_edit_returns_the_same_snapshot(tmp_path):
    """Minted by difference, not by action — otherwise exporting twice makes two identical
    snapshots and the list becomes the `final_v2_REAL_final` pile this design exists to avoid.
    """
    _write(tmp_path)
    first = snapshots.pin(tmp_path, reason="export")
    second = snapshots.pin(tmp_path, reason="comparison")
    assert first == second
    assert snapshots.discover_snapshots(tmp_path) == [first]


def test_editing_after_a_pin_produces_a_second_snapshot(tmp_path):
    """The negative half of the rule above: a real change must not be folded into the
    existing snapshot, or the first result silently becomes the second."""
    _write(tmp_path)
    first = snapshots.pin(tmp_path, reason="finish")
    _write(
        tmp_path,
        instances=[_instance(source_label="c02")],
        manifest=_manifest(
            names={"c02": ["grooming"]},
            vocabulary=["grooming"],
            behavior_types={"grooming": "state"},
        ),
    )
    second = snapshots.pin(tmp_path, reason="export")

    assert second != first
    assert snapshots.discover_snapshots(tmp_path) == [first, second]
    assert snapshots.read_snapshot(tmp_path, second)["parent"] == first


def test_an_audit_decision_alone_does_not_mint_a_snapshot(tmp_path):
    """Adding a human decision does not change what the instances are. Minting on it would put a
    snapshot behind every audited clip."""
    _write(tmp_path)
    first = snapshots.pin(tmp_path, reason="finish")
    (tmp_path / bo.AUDIT_FILENAME).write_text(
        "instance_id,score,human_label,drawn_at\nabc,0.9,true,2026-01-01\n",
        encoding="utf-8",
    )
    assert snapshots.pin(tmp_path, reason="export") == first


def test_a_run_with_no_artifact_pins_nothing(tmp_path):
    """A fit whose granularity is unset has no artifact (§21), and finalizing it is legal — so
    this is a real state rather than an error."""
    assert snapshots.pin(tmp_path, reason="finish") is None
    assert snapshots.discover_snapshots(tmp_path) == []


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #
def test_the_record_says_why_it_exists_and_what_produced_it(tmp_path):
    _write(tmp_path)
    sid = snapshots.pin(
        tmp_path,
        reason="comparison",
        state={"granularity": 12, "names": {"c01": "rearing"}},
        code_sha="abc1234",
    )
    record = snapshots.read_snapshot(tmp_path, sid)
    assert record["reason"] == "comparison"
    assert record["state"]["granularity"] == 12
    assert record["code_sha"] == "abc1234"
    assert record["parent"] is None
    assert record["content_hash"] == snapshots.artifact_hash(tmp_path)


def test_the_audit_is_copied_so_its_precision_is_fixed(tmp_path):
    """An audit is evidence about one artifact state (§19). Leaving it behind would let a
    snapshot's precision move with a later reprojection."""
    _write(tmp_path)
    (tmp_path / bo.AUDIT_FILENAME).write_text(
        "instance_id,score,human_label,drawn_at\nabc,0.9,true,2026-01-01\n",
        encoding="utf-8",
    )
    sid = snapshots.pin(tmp_path, reason="finish")
    assert (snapshots.snapshot_dir(tmp_path, sid) / bo.AUDIT_FILENAME).is_file()


def test_an_unknown_reason_is_refused(tmp_path):
    _write(tmp_path)
    with pytest.raises(ValueError, match="reason must be one of"):
        snapshots.pin(tmp_path, reason="because")


def test_the_expensive_pass_is_never_copied(tmp_path):
    """A snapshot is the artifact, not the fit — the fit is where the size actually is."""
    _write(tmp_path)
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "dist.npy").write_bytes(b"x" * 1024)
    sid = snapshots.pin(tmp_path, reason="finish")
    assert not (snapshots.snapshot_dir(tmp_path, sid) / "fit").exists()


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def test_a_half_written_snapshot_is_not_discoverable(tmp_path):
    """`snapshot.yaml` is the commit point, so a crash mid-copy leaves a directory that is
    skipped rather than a snapshot that lies about its contents."""
    _write(tmp_path)
    sid = snapshots.pin(tmp_path, reason="finish")
    (snapshots.snapshot_dir(tmp_path, sid) / snapshots.SNAPSHOT_FILENAME).unlink()
    assert snapshots.discover_snapshots(tmp_path) == []
    assert snapshots.latest_snapshot(tmp_path) is None


def test_ids_are_allocated_in_numeric_order(tmp_path):
    _write(tmp_path)
    assert snapshots.next_snapshot_id(tmp_path) == "snap_01"
    snapshots.pin(tmp_path, reason="finish")
    assert snapshots.next_snapshot_id(tmp_path) == "snap_02"


def test_the_record_is_written_atomically_with_schema_version_first(tmp_path):
    _write(tmp_path)
    sid = snapshots.pin(tmp_path, reason="finish")
    sdir = snapshots.snapshot_dir(tmp_path, sid)
    assert not list(sdir.glob("*.tmp"))
    with open(sdir / snapshots.SNAPSHOT_FILENAME, encoding="utf-8-sig") as handle:
        assert next(iter(yaml.safe_load(handle))) == "schema_version"


def test_a_record_from_another_schema_is_refused(tmp_path):
    _write(tmp_path)
    sid = snapshots.pin(tmp_path, reason="finish")
    path = snapshots.snapshot_dir(tmp_path, sid) / snapshots.SNAPSHOT_FILENAME
    record = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    record["schema_version"] = 99
    path.write_text(yaml.dump(record, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        snapshots.read_snapshot(tmp_path, sid)


def test_artifact_hash_is_none_without_an_artifact(tmp_path):
    assert snapshots.artifact_hash(tmp_path) is None
