"""`AuthoredRun.coverage_summary`: how much of each video was analysed, and how much of that is labelled.

Two ratios a user reads together — the first is tracking and embedding, the second this run — so
each is checked against a fixture whose answer is known, and the snapshot view is checked to report
its frozen numbers rather than the live run's.
"""

from detail.storage import behavior_output as bo


def test_a_recording_with_no_timestamp_map_has_no_video_length(authored_cohort):
    """No map, no length: ``None`` rather than a figure built from a guessed duration."""
    rows = authored_cohort.run().coverage_summary()

    assert {row["recording_id"] for row in rows} == {"r1", "r2"}
    for row in rows:
        assert row["video_ns"] is None
        assert row["analysed_ns"] > 0
        assert 0 < row["in_instance_ns"] <= row["analysed_ns"]


def test_instance_time_uses_the_rule_the_cluster_table_uses(authored_cohort):
    """The line, the table and measures agree on what an instance lasts."""
    artifact = bo.read_behavior_output(authored_cohort.run_dir())
    periods = bo.period_by_recording(artifact.coverage)
    by_recording: dict[str, int] = {}
    for row in bo.bouts(artifact.instances):
        by_recording[row.recording_id] = by_recording.get(
            row.recording_id, 0
        ) + bo.instance_ns(row, periods)

    rows = authored_cohort.run().coverage_summary()

    assert {r["recording_id"]: r["in_instance_ns"] for r in rows} == by_recording


def test_a_snapshot_view_reports_its_frozen_coverage(authored_cohort):
    cohort = authored_cohort
    before = cohort.run().coverage_summary()
    snapshot_id = cohort.run().pin(reason="export")

    artifact = bo.read_behavior_output(cohort.run_dir())
    bo.write_behavior_output(
        cohort.run_dir(),
        instances=list(bo.bouts(artifact.instances))[:1],
        coverage=list(artifact.coverage),
        manifest=artifact.manifest,
    )

    assert cohort.run().coverage_summary() != before
    assert cohort.run().view(snapshot_id).coverage_summary() == before
