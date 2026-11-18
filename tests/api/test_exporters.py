"""The one export writer: its own folder, its sidecar, and its manifest.

The behaviours here are the ones no individual target should have to remember, which is the whole
argument for a single writer rather than seventeen functions that each nearly do this.
"""

import csv

import pytest

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportScope, ExportSubject


def _subject(cohort, kind="analysis_run", run_id="an_01"):
    return ExportSubject(kind, cohort.experiment.dir, run_id)


def _rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _export(cohort, target_ids, parent, *, subject=None, **kwargs):
    """Run one or more targets, the way both surfaces do."""
    subject = subject or _subject(cohort)
    targets = [exporters.target(subject.kind, t) for t in target_ids]
    return exporters.write_export(subject, targets, parent, **kwargs)


@pytest.fixture
def two_clusters(authored_cohort):
    """An artifact with **two** source labels, one of them unnamed.

    `authored_cohort` writes every instance under one label, so "narrow to one cluster" would
    remove nothing and pass against its own reverted fix — the fixture-that-never-diverges trap
    (CLAUDE.md). The unnamed cluster is the other half: it is what makes the ethogram's
    fall-back-to-`source_label` branch reachable at all.
    """
    from detail.storage import analysis_runs as analysis_store
    from detail.storage import behavior_output as bo

    run_dir = analysis_store.analysis_dir(authored_cohort.experiment.dir, "an_01")
    artifact = bo.read_behavior_output(run_dir)
    instances = []
    for index, row in enumerate(artifact.instances):
        label = "c01" if index % 2 == 0 else "c02"
        instances.append(
            bo.BehaviorInstance(
                row.recording_id,
                row.subject,
                row.t_ns,
                row.end_t_ns,
                row.start_frame,
                row.end_frame,
                label,
                "rearing" if label == "c01" else "",
            )
        )
    manifest = dict(artifact.manifest)
    manifest["names"] = {"c01": ["rearing"], "c02": []}
    bo.write_behavior_output(
        run_dir, instances=instances, coverage=artifact.coverage, manifest=manifest
    )
    labels = {r.source_label for r in bo.read_behavior_output(run_dir).instances}
    assert labels == {"c01", "c02"}, "the fixture must actually hold two clusters"
    return authored_cohort


# --------------------------------------------------------------------------- #
# every export gets its own folder
# --------------------------------------------------------------------------- #
def test_the_export_creates_its_own_named_folder(authored_cohort, tmp_path):
    """Not loose files in a folder you chose — a folder that says what it is."""
    result = _export(authored_cohort, ["instances"], tmp_path)

    assert result.dest.parent == tmp_path
    assert result.dest.is_dir()
    name = result.dest.name
    assert "cohort" in name, "the experiment is not in the folder name"
    assert "temporal_clustering" in name, "what produced it is not in the folder name"
    assert "an_01" in name, "the run is not in the folder name"
    assert "export" in name
    assert (result.dest / "instances.csv").is_file()


def test_the_folder_name_carries_the_date_and_time(authored_cohort):
    """So two exports of the same run are distinguishable, and orderable, by name."""
    import re
    from datetime import datetime

    subject = _subject(authored_cohort)
    name = exporters.folder_name(subject, when=datetime(2026, 9, 10, 14, 32))
    assert name.endswith("export_2026-09-10_143200")
    assert re.search(r"\d{4}-\d{2}-\d{2}_\d{6}$", name)


def test_two_exports_back_to_back_do_not_collide(authored_cohort, tmp_path):
    """The reason there is no cleanup step: a fresh folder cannot overwrite the last one.

    Back to back **in the same second**, which the timestamp alone cannot separate — two CSV
    exports of one run take milliseconds. An export that landed in the previous export's folder
    would mix two results under one manifest describing both.
    """
    parent = tmp_path / "somewhere"
    subject = _subject(authored_cohort)
    target = exporters.target("analysis_run", "instances")

    first = exporters.write_export(subject, [target], parent)
    second = exporters.write_export(subject, [target], parent)

    assert first.dest != second.dest, "the second export reused the first one's folder"
    assert first.dest.is_dir() and (first.dest / "instances.csv").is_file()
    assert second.dest.is_dir() and (second.dest / "instances.csv").is_file()
    assert len(list(parent.iterdir())) == 2


def test_an_export_leaves_a_users_own_files_alone(authored_cohort, tmp_path):
    parent = tmp_path / "somewhere"
    parent.mkdir()
    mine = parent / "my_own_notes.csv"
    mine.write_text("keep me", encoding="utf-8")

    _export(authored_cohort, ["instances"], parent)
    assert mine.read_text(encoding="utf-8-sig") == "keep me"


def test_an_export_never_writes_outside_its_own_folder(authored_cohort, tmp_path):
    # An empty directory of its own: `tmp_path` also holds the fixture's experiment, so
    # asserting against it would compare the export to somebody else's tree.
    parent = tmp_path / "somewhere"
    parent.mkdir()
    result = _export(authored_cohort, ["instances", "ethogram"], parent)
    assert sorted(p.name for p in parent.iterdir()) == [result.dest.name]


# --------------------------------------------------------------------------- #
# several targets, one folder
# --------------------------------------------------------------------------- #
def test_several_targets_land_in_one_folder(two_clusters, tmp_path):
    """One decision, one folder — the whole point of a checklist over a menu of dialogs."""
    result = _export(two_clusters, ["instances", "ethogram"], tmp_path)
    names = set(result.files)
    assert "instances.csv" in names
    assert "coverage.csv" in names
    assert "r1_ethogram.csv" in names
    assert set(result.target_ids) == {"instances", "ethogram"}


def test_asking_for_no_target_is_refused(authored_cohort, tmp_path):
    with pytest.raises(exporters.ExportError, match="at least one"):
        exporters.write_export(_subject(authored_cohort), [], tmp_path)


def test_the_progress_bar_folds_each_target_into_the_whole(two_clusters, tmp_path):
    """Three targets must not run the bar to 100% three times.

    The aggregate-fold rule (DESIGN_PLAN §6) one surface over: a bar that finishes and restarts
    reads as a run that finished and did not.
    """
    seen = []
    _export(
        two_clusters,
        ["instances", "ethogram"],
        tmp_path,
        progress=lambda stage, fraction, detail="": seen.append(fraction),
    )
    assert seen, "no progress at all"
    assert seen == sorted(seen), f"the bar went backwards: {seen}"
    assert max(seen) <= 1.0
    assert any(
        0.0 < f < 1.0 for f in seen
    ), "the bar never moved between start and finish"


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #
def test_the_manifest_names_every_file_it_wrote(authored_cohort, tmp_path):
    result = _export(authored_cohort, ["instances"], tmp_path)

    manifest = exporters.read_manifest(result.dest)
    assert manifest["schema_version"] == 1
    assert list(manifest)[0] == "schema_version", "schema_version is the first key"
    assert manifest["targets"] == ["instances"]
    assert set(manifest["files"]) == set(result.files)
    for name in manifest["files"]:
        assert (result.dest / name).is_file(), f"{name} is recorded but not on disk"


def test_a_declared_data_file_is_actually_written(authored_cohort, tmp_path):
    """What makes the declaration checkable rather than decorative.

    `instances` declares `coverage.csv` beside its artifact, so the export has to produce it —
    a target that declared a file it never wrote would be a Save-As button with a promise on it.
    """
    target = exporters.target("analysis_run", "instances")
    result = _export(authored_cohort, ["instances"], tmp_path)

    assert (
        target.data_files
    ), "this target declares no data files, so this asserts nothing"
    for name in target.data_files:
        assert name in result.files, f"{name} is declared but was not written"
        assert (result.dest / name).is_file()


# --------------------------------------------------------------------------- #
# params.csv — derived, never hand-listed
# --------------------------------------------------------------------------- #
def test_params_sidecar_carries_the_run_record_params(authored_cohort, tmp_path):
    """The settings that produced the run, beside what it produced.

    Read from `record["params"]`, which `runners.job_params` already derived from the job. A
    second derivation here would be the `test_record_completeness` class one surface over.
    """
    from detail.storage import analysis_runs as analysis_store

    run_dir = analysis_store.analysis_dir(authored_cohort.experiment.dir, "an_01")
    record = analysis_store.read_analysis_yaml(run_dir)
    record["params"] = {"granularity": 4, "method": "finch", "track_id": 0}
    analysis_store.write_analysis_yaml(run_dir, record)

    result = _export(authored_cohort, ["instances"], tmp_path)
    rows = {r["parameter"]: r["value"] for r in _rows(result.dest / "params.csv")}
    assert rows == {"granularity": "4", "method": "finch", "track_id": "0"}
    manifest = exporters.read_manifest(result.dest)
    assert (
        "params.csv" in manifest["files"]
    ), "the sidecar is recorded like any other file"


def test_a_subject_with_no_run_params_ships_no_sidecar(authored_cohort, tmp_path):
    """An experiment has no run parameters, so an empty params.csv would be a lie."""
    result = _export(
        authored_cohort,
        ["metadata"],
        tmp_path,
        subject=ExportSubject("experiment", authored_cohort.experiment.dir),
    )
    assert not (result.dest / "params.csv").exists()


def test_every_target_promises_the_sidecars(authored_cohort):
    """`included_files` is what a surface shows, so it must be the whole list."""
    target = exporters.target("analysis_run", "instances")
    assert target.included_files()[-2:] == (
        exporters.PARAMS_NAME,
        exporters.MANIFEST_NAME,
    )
    assert "coverage.csv" in target.included_files()


# --------------------------------------------------------------------------- #
# empty
# --------------------------------------------------------------------------- #
def _name_an_empty_cluster(cohort, source="c09", name="sleep"):
    """Give the artifact a named cluster with no instances: real, and nothing to write."""
    from detail.storage import behavior_output as bo

    run_dir = cohort.run_dir()
    artifact = bo.read_behavior_output(run_dir)
    manifest = dict(artifact.manifest)
    manifest["names"] = {**(manifest.get("names") or {}), source: [name]}
    manifest["vocabulary"] = [*(manifest.get("vocabulary") or []), name]
    bo.write_behavior_output(
        run_dir,
        instances=artifact.instances,
        coverage=artifact.coverage,
        manifest=manifest,
    )


def test_an_export_with_nothing_to_write_says_so(authored_cohort, tmp_path):
    """An empty folder reads as a failure; a manifest that says why does not.

    The cluster is one the run has and that holds no instance: an unknown one is refused
    (`test_an_unknown_cluster_is_refused_naming_the_clusters`), which is a different answer.
    """
    _name_an_empty_cluster(authored_cohort)
    result = _export(
        authored_cohort, ["instances"], tmp_path, scope=ExportScope(cluster="c09")
    )
    assert result.is_empty
    assert not result.files
    manifest = exporters.read_manifest(result.dest)
    assert manifest["files"] == []
    assert "cluster c09" in manifest["empty"]


# --------------------------------------------------------------------------- #
# what was asked is refused, never ignored (#31, findings 40-43)
# --------------------------------------------------------------------------- #
def test_a_scope_a_target_does_not_take_is_refused_and_nothing_is_written(
    two_clusters, tmp_path
):
    """`--target instances --frames 0:3` wrote every instance and recorded a frames scope."""
    with pytest.raises(exporters.ExportError, match="cannot be narrowed to a frame"):
        _export(
            two_clusters,
            ["instances"],
            tmp_path / "out",
            scope=ExportScope(frames=(0, 3)),
        )
    assert not (tmp_path / "out").exists(), "a refused export creates no folder"


@pytest.mark.parametrize("target_id", ["ethogram", "frame_labels", "instances"])
def test_an_unknown_cluster_is_refused_naming_the_clusters(
    frame_labelled, tmp_path, target_id
):
    with pytest.raises(exporters.ExportError) as caught:
        _export(
            frame_labelled,
            [target_id],
            tmp_path / "out",
            scope=ExportScope(cluster="zz"),
        )
    assert str(caught.value) == (
        "this run has no cluster 'zz'. Its clusters: c01 (rearing), c02"
    )
    assert not (tmp_path / "out").exists()


def test_a_cluster_given_by_its_name_says_which_id_it_is(two_clusters, tmp_path):
    with pytest.raises(exporters.ExportError, match="'rearing' is the name of c01"):
        _export(
            two_clusters, ["ethogram"], tmp_path, scope=ExportScope(cluster="rearing")
        )


def test_frames_past_the_end_are_refused(frame_labelled, tmp_path):
    """The fixture's frame views are 100 frames long: 99 is the last one."""
    with pytest.raises(exporters.ExportError) as caught:
        _export(
            frame_labelled,
            ["frame_labels"],
            tmp_path / "past",
            scope=ExportScope(frames=(5000, 6000)),
        )
    assert "has 100 frames (0-99)" in str(caught.value)
    last = _export(
        frame_labelled,
        ["frame_labels"],
        tmp_path / "last",
        scope=ExportScope(frames=(99, 6000)),
    )
    assert [int(r["frame_idx"]) for r in _frame_rows(last)] == [99]


def test_an_unknown_selection_is_refused(authored_cohort, tmp_path):
    with pytest.raises(exporters.ExportError, match="no instance 'not-an-id'"):
        _export(
            authored_cohort,
            ["instances"],
            tmp_path,
            scope=ExportScope(instance_ids=("not-an-id",)),
        )


def test_a_target_asked_for_twice_is_refused(authored_cohort, tmp_path):
    with pytest.raises(exporters.ExportError, match="'instances' is asked for twice"):
        _export(authored_cohort, ["instances", "instances"], tmp_path)


def test_the_scope_line_names_the_cluster_and_the_frames(frame_labelled, tmp_path):
    result = _export(
        frame_labelled,
        ["frame_labels"],
        tmp_path,
        scope=ExportScope(cluster="c01", frames=(0, 9)),
    )
    assert exporters.read_manifest(result.dest)["scope"] == "cluster c01, frames 0-9"


def _whole_report():
    """A copy of `instances` that declares no scopes: a whole-subject report, as `performance` is."""
    import dataclasses

    real = exporters.target("analysis_run", "instances")

    def write(subject, dest, scope, progress, **options):
        out = dest / "report.md"
        out.write_text("the whole run\n", encoding="utf-8")
        return [out]

    return dataclasses.replace(real, id="report", label="Report", scopes=(), run=write)


@pytest.fixture
def fake_targets(monkeypatch):
    """Let the stand-in targets below answer `unavailable_reason` like a real one."""
    for target_id in ("report", "a", "b"):
        monkeypatch.setitem(
            export_targets._REASONS, ("analysis_run", target_id), lambda subject: None
        )


def test_a_whole_run_report_beside_a_narrowed_target_is_named_in_the_manifest(
    two_clusters, tmp_path, fake_targets
):
    subject = _subject(two_clusters)
    targets = [exporters.target("analysis_run", "instances"), _whole_report()]
    result = exporters.write_export(
        subject, targets, tmp_path, scope=ExportScope(cluster="c01")
    )
    manifest = exporters.read_manifest(result.dest)
    assert manifest["scope"] == "cluster c01"
    assert manifest["whole"] == ["report"]
    plain = exporters.write_export(subject, targets, tmp_path / "plain")
    assert "whole" not in exporters.read_manifest(plain.dest)


def test_a_narrowing_only_whole_run_reports_were_asked_for_is_refused(
    two_clusters, tmp_path, fake_targets
):
    with pytest.raises(exporters.ExportError, match="narrows nothing"):
        exporters.write_export(
            _subject(two_clusters),
            [_whole_report()],
            tmp_path,
            scope=ExportScope(cluster="c01"),
        )


def test_a_file_two_targets_share_is_listed_once(two_clusters, tmp_path, fake_targets):
    """Every keypoint target writes `coverage.csv`; five of them listed it five times."""
    import dataclasses

    def shared(name):
        def write(subject, dest, scope, progress, **options):
            out = dest / "coverage.csv"
            out.write_text("x\n", encoding="utf-8")
            return [out, dest / f"{name}.csv"]

        return write

    real = exporters.target("analysis_run", "instances")
    a = dataclasses.replace(real, id="a", run=shared("a"))
    b = dataclasses.replace(real, id="b", run=shared("b"))
    result = exporters.write_export(_subject(two_clusters), [a, b], tmp_path)
    assert [f for f in result.files if f == "coverage.csv"] == ["coverage.csv"]


# --------------------------------------------------------------------------- #
# scope
# --------------------------------------------------------------------------- #
def test_scope_narrows_to_one_cluster(two_clusters, tmp_path):
    every = _export(two_clusters, ["instances"], tmp_path / "all")
    scoped = _export(
        two_clusters, ["instances"], tmp_path / "one", scope=ExportScope(cluster="c01")
    )
    all_rows = _rows(every.dest / "instances.csv")
    some = _rows(scoped.dest / "instances.csv")

    assert {r["source_label"] for r in all_rows} == {"c01", "c02"}
    assert {r["source_label"] for r in some} == {"c01"}
    assert 0 < len(some) < len(all_rows), "narrowing must actually remove rows"


def test_scope_narrows_to_an_explicit_selection(authored_cohort, tmp_path):
    """ "Export a selection" is the same call with the rows named."""
    every = _export(authored_cohort, ["instances"], tmp_path / "all")
    picked = [r["instance_id"] for r in _rows(every.dest / "instances.csv")[:2]]

    result = _export(
        authored_cohort,
        ["instances"],
        tmp_path / "some",
        scope=ExportScope(instance_ids=tuple(picked)),
    )
    rows = _rows(result.dest / "instances.csv")
    assert [r["instance_id"] for r in rows] == picked
    assert exporters.read_manifest(result.dest)["scope"] == "2 selected"
    assert not result.is_empty


# --------------------------------------------------------------------------- #
# the catalogue's own edges
# --------------------------------------------------------------------------- #
def test_a_target_is_refused_for_the_wrong_subject_kind(authored_cohort, tmp_path):
    with pytest.raises(exporters.ExportError, match="is for a analysis_run"):
        exporters.write_export(
            ExportSubject("experiment", authored_cohort.experiment.dir),
            [exporters.target("analysis_run", "instances")],
            tmp_path,
        )


def test_targets_for_lists_everything_and_available_targets_narrows(authored_cohort):
    """Grey out, don't hide — so the two questions are asked separately.

    `targets_for` is what a surface *shows*; `available_targets` is what can run. Collapsing
    them was the original mistake: a hidden row leaves the reader unable to tell "DETAIL cannot
    export this" from "there is none of it yet".
    """
    subject = _subject(authored_cohort)
    listed = {t.id for t in exporters.targets_for(subject)}
    runnable = {t.id for t in exporters.available_targets(subject)}

    assert listed == {
        "instances",
        "ethogram",
        "frame_labels",
        "clips",
        "performance",
    }
    assert runnable == {"instances", "ethogram", "frame_labels"}
    assert "clips" in listed - runnable, "the fixture records no embedding run"
    assert exporters.unavailable_reason(
        exporters.target("analysis_run", "clips"), subject
    )
    # The performance report is the same shape one target over, and the fixture is the case its
    # reason exists for: a run with no activity log. It is listed and greyed, with a sentence
    # naming the missing thing rather than the word "unavailable".
    assert "performance" in listed - runnable
    assert "activity log" in exporters.unavailable_reason(
        exporters.target("analysis_run", "performance"), subject
    )


def test_an_analysis_run_with_no_artifact_can_run_nothing(authored_cohort):
    """Still listed — with a reason — so the control simply does not appear for it."""
    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_99")
    assert exporters.available_targets(subject) == []
    reasons = {
        t.id: exporters.unavailable_reason(t, subject)
        for t in exporters.targets_for(subject)
    }
    assert reasons and all(r for r in reasons.values()), reasons


def test_an_unavailable_target_is_refused_with_its_reason(authored_cohort, tmp_path):
    """Refused at the writer too, not only greyed in one surface.

    A CLI caller naming a target whose inputs are absent gets the same sentence the dialog
    would have shown, rather than a traceback from inside the exporter.
    """
    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_99")
    with pytest.raises(exporters.ExportError, match="not produced a behavior result"):
        exporters.write_export(
            subject, [exporters.target("analysis_run", "instances")], tmp_path
        )


def test_ethogram_writes_one_file_per_recording(two_clusters, tmp_path):
    result = _export(two_clusters, ["ethogram"], tmp_path)
    files = [f for f in result.files if f.endswith("_ethogram.csv")]
    assert sorted(files) == ["r1_ethogram.csv", "r2_ethogram.csv"]
    rows = _rows(result.dest / "r1_ethogram.csv")
    assert list(rows[0]) == ["start_frame", "end_frame", "label", "subject", "origin"]
    assert all(r["label"] for r in rows), "every span carries a label"


def test_an_unnamed_cluster_keeps_its_own_id_in_the_ethogram(two_clusters, tmp_path):
    """Dropping the unnamed rows would silently shorten the ethogram.

    An unnamed cluster is still a thing that happened, so it goes out under the producer's own
    id rather than not at all — and the fixture has one, or this branch is never reached.
    """
    result = _export(two_clusters, ["ethogram"], tmp_path)
    labels = set()
    for name in ("r1_ethogram.csv", "r2_ethogram.csv"):
        labels |= {r["label"] for r in _rows(result.dest / name)}
    assert labels == {
        "rearing",
        "c02",
    }, "the named cluster goes out under its name and the unnamed one under its id"


# --------------------------------------------------------------------------- #
# Per-frame labels — the regressor a GLM takes (#524)
# --------------------------------------------------------------------------- #
#: Frames the fixture's tracker "lost" — kept as rows, flagged by `observed`.
_LOST = range(90, 100)


@pytest.fixture
def frame_labelled(two_clusters):
    """`two_clusters` with its frame views rebuilt to match, and a lost stretch at the end.

    `authored_cohort` writes every frame observed and one cluster, so an export that dropped
    unobserved rows, or merged the unnamed cluster into the named one, would pass against it.
    """
    import numpy as np

    from detail.storage import behavior_output as bo
    from detail.storage import frame_view as fv

    run_dir = two_clusters.run_dir()
    artifact = bo.read_behavior_output(run_dir)
    axis = (np.arange(100) * 10_000_000).astype(np.int64)
    observed = np.ones(100, bool)
    observed[list(_LOST)] = False
    for recording in ("r1", "r2"):
        fv.write_frame_view(
            run_dir,
            recording_id=recording,
            subject=0,
            t_ns=axis,
            observed=observed,
            instances=[i for i in artifact.instances if i.recording_id == recording],
        )
    return two_clusters


def _frame_rows(result, recording="r1"):
    return _rows(result.dest / f"{recording}_subject0_frame_labels.csv")


def test_frame_labels_keep_every_frame_and_flag_the_unobserved(
    frame_labelled, tmp_path
):
    """A lost frame is a row with `observed = 0`, never a missing row and never a silent 0.

    Dropping it shortens the table; zero-filling it without the flag reads "not analyzed" as
    "behavior absent" and moves the denominator of every rate computed off the file.
    """
    result = _export(frame_labelled, ["frame_labels"], tmp_path)
    assert sorted(f for f in result.files if f.endswith("_frame_labels.csv")) == [
        "r1_subject0_frame_labels.csv",
        "r2_subject0_frame_labels.csv",
    ]
    rows = _frame_rows(result)
    assert len(rows) == 100, "one row per video frame, observed or not"
    assert [int(r["frame_idx"]) for r in rows] == list(range(100))
    assert int(rows[7]["t_ns"]) == 70_000_000
    assert {int(r["frame_idx"]) for r in rows if r["observed"] == "0"} == set(_LOST)


def test_frame_labels_carry_a_column_per_behavior_and_one_per_unnamed_cluster(
    frame_labelled, tmp_path
):
    """r1's bouts alternate c01 (named "rearing") and c02 (unnamed): frames 0 and 6 are
    rearing, frame 3 is the unnamed cluster, and frame 50 is neither."""
    rows = _frame_rows(_export(frame_labelled, ["frame_labels"], tmp_path))
    assert list(rows[0]) == ["frame_idx", "t_ns", "observed", "rearing", "c02"]
    assert (rows[0]["rearing"], rows[0]["c02"]) == ("1", "0")
    assert (rows[3]["rearing"], rows[3]["c02"]) == ("0", "1")
    assert (rows[6]["rearing"], rows[6]["c02"]) == ("1", "0")
    assert (rows[50]["rearing"], rows[50]["c02"]) == ("0", "0")


def test_two_clusters_sharing_a_name_are_one_column(frame_labelled, tmp_path):
    from detail.storage import behavior_output as bo

    run_dir = frame_labelled.run_dir()
    artifact = bo.read_behavior_output(run_dir)
    manifest = dict(artifact.manifest)
    manifest["names"] = {"c01": ["rearing"], "c02": ["rearing"]}
    bo.write_behavior_output(
        run_dir,
        instances=artifact.instances,
        coverage=artifact.coverage,
        manifest=manifest,
    )
    rows = _frame_rows(_export(frame_labelled, ["frame_labels"], tmp_path))
    assert list(rows[0]) == ["frame_idx", "t_ns", "observed", "rearing"]
    assert [rows[i]["rearing"] for i in (0, 3, 6, 50)] == ["1", "1", "1", "0"]


def test_a_name_that_is_another_clusters_id_is_refused(frame_labelled, tmp_path):
    """Naming c01 "c02" while c02 is unnamed would merge two different things into one
    column; the export says so instead of guessing which one the column means."""
    from detail.storage import behavior_output as bo

    run_dir = frame_labelled.run_dir()
    artifact = bo.read_behavior_output(run_dir)
    manifest = dict(artifact.manifest)
    manifest["names"] = {"c01": ["c02"], "c02": []}
    manifest["vocabulary"] = ["rearing", "c02"]
    bo.write_behavior_output(
        run_dir,
        instances=artifact.instances,
        coverage=artifact.coverage,
        manifest=manifest,
    )
    result = _export(frame_labelled, ["frame_labels"], tmp_path)
    assert result.status == "failed"
    assert "unnamed cluster" in result.failures[0].error
    assert not any(f.endswith("_frame_labels.csv") for f in result.files)


def test_frame_labels_honour_a_cluster_and_a_frame_range(frame_labelled, tmp_path):
    one = _frame_rows(
        _export(
            frame_labelled,
            ["frame_labels"],
            tmp_path / "cluster",
            scope=ExportScope(cluster="c02"),
        )
    )
    assert list(one[0]) == ["frame_idx", "t_ns", "observed", "c02"]
    assert len(one) == 100

    span = _frame_rows(
        _export(
            frame_labelled,
            ["frame_labels"],
            tmp_path / "range",
            scope=ExportScope(frames=(85, 95)),
        )
    )
    assert [int(r["frame_idx"]) for r in span] == list(range(85, 96))
    assert [r["observed"] for r in span] == ["1"] * 5 + ["0"] * 6


# --------------------------------------------------------------------------- #
# Where each bout happened — the 9/3 ask, "export must link clusters to XY"
# --------------------------------------------------------------------------- #
def _place(cohort, positions, run_id="an_01"):
    """Give the artifact an `instance_features.csv` holding ``positions``."""
    from detail.storage import behavior_output

    behavior_output.write_instance_features(cohort.run_dir(run_id), positions)


def test_instances_carry_the_position_of_each_bout(authored_cohort, tmp_path):
    artifact = _rows_of_artifact(authored_cohort)
    _place(authored_cohort, {artifact[0]["instance_id"]: (0.25, 0.75)})

    rows = _rows(
        _export(authored_cohort, ["instances"], tmp_path).dest / "instances.csv"
    )
    placed = {r["instance_id"]: r for r in rows}[artifact[0]["instance_id"]]
    assert float(placed["cx_norm"]) == pytest.approx(0.25)
    assert float(placed["cy_norm"]) == pytest.approx(0.75)


def test_a_run_with_no_positions_has_no_position_columns(authored_cohort, tmp_path):
    """Absent columns and blank cells are two different answers, so they look different.

    No `instance_features.csv` means the run predates §14's kinematics, which
    `detail recording backfill-kinematics` fixes — so the header says the export could not know.
    A blank cell under a header that exists means the bout itself had no placeable frame, which
    needs no action. Collapsing the two is the `human_label=None` class
    (`tests/storage/test_behavior_audit.py`).
    """
    rows = _rows(
        _export(authored_cohort, ["instances"], tmp_path).dest / "instances.csv"
    )
    assert (
        "cx_norm" not in rows[0]
    ), "the fixture writes no features, so there is nothing to say"


def test_an_unplaceable_bout_is_blank_under_a_column_that_exists(
    authored_cohort, tmp_path
):
    """The other half of the pair above — the state that is *not* actionable."""
    artifact = _rows_of_artifact(authored_cohort)
    _place(
        authored_cohort,
        {
            artifact[0]["instance_id"]: (float("nan"), float("nan")),
            artifact[-1]["instance_id"]: (0.5, 0.5),
        },
    )

    rows = {
        r["instance_id"]: r
        for r in _rows(
            _export(authored_cohort, ["instances"], tmp_path).dest / "instances.csv"
        )
    }
    assert rows[artifact[0]["instance_id"]]["cx_norm"] == ""
    assert rows[artifact[-1]["instance_id"]]["cx_norm"] == "0.5"


def _rows_of_artifact(cohort, run_id="an_01"):
    from detail.storage import behavior_output

    return [
        {"instance_id": row.instance_id}
        for row in behavior_output.read_behavior_output(
            cohort.run_dir(run_id)
        ).instances
    ]


# --------------------------------------------------------------------------- #
# The upstream run targets — a trajectory and a set of masks
# --------------------------------------------------------------------------- #
def _run_subject(built, run_id):
    return ExportSubject("run", built.experiment.dir, run_id)


def _export_run(built, run_id, target_id, parent, **kwargs):
    subject = _run_subject(built, run_id)
    return exporters.write_export(
        subject, [exporters.target("run", target_id)], parent, **kwargs
    )


def test_a_tracking_run_can_export_its_tracks(processed_recording, tmp_path):
    """The complaint `docs/export_design.md` §1 opens with: a tracking run had no way out."""
    subject = _run_subject(processed_recording, processed_recording.tracking_run_id)
    runnable = {t.id for t in exporters.available_targets(subject)}
    assert "tracks" in runnable, "a completed tracking run offers its trajectory"

    result = _export_run(
        processed_recording, processed_recording.tracking_run_id, "tracks", tmp_path
    )
    rows = _rows(result.dest / "recording_001_tracks.csv")
    assert len(rows) == 1, "one row per (frame, track)"
    assert rows[0]["frame_idx"] == "0" and rows[0]["object_id"] == "0"
    # px == norm * 40 on this fixture, so a wrong scaling is visible rather than plausible.
    assert float(rows[0]["bbox_cx_norm"]) == pytest.approx(0.3)
    assert float(rows[0]["bbox_cx_px"]) == pytest.approx(12.0)


def _four_detections(built):
    """Rewrite the fixture's one recording with four detection rows.

    A cadence test needs something to be partway *through*; with a single row the only fractions
    reachable are 0 and 1, and the assertion silently requires a second recording — at which
    point it is the outer count moving the bar and the single-unit case the rule exists for is
    dead (CLAUDE.md, *assert the inner pair on a single-unit fixture*).
    """
    import shutil

    from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter

    recording_dir = built.recording.dir
    det_part = recording_dir / "detections" / "det_01" / "part_001.zarr"
    assign_part = (
        recording_dir / "tracking" / "track_01" / "assignments" / "part_001.zarr"
    )
    for path in (det_part, assign_part):
        shutil.rmtree(path)
    mask = built.square(4, 20, 4, 20)
    with DetectionPartWriter(det_part) as writer:
        for frame in range(4):
            writer.add_row(
                frame,
                frame * 1000,
                0,
                "mouse",
                0.9,
                (0.1, 0.1, 0.5, 0.5),
                built.rle(mask),
            )
    with AssignmentPartWriter(assign_part) as writer:
        for _ in range(4):
            writer.add_object_id(0)


@pytest.mark.parametrize(
    "run_attribute, target_id",
    [("tracking_run_id", "tracks"), ("detection_run_id", "masks")],
)
def test_an_upstream_export_moves_its_bar_within_one_recording(
    processed_recording, tmp_path, run_attribute, target_id, monkeypatch
):
    """A single-recording export must still show movement, or it reads as hung.

    Measured on a real 30-minute pair: `tracks` is **minutes**, because the compute behind it
    decodes a mask per detection row. `masks` reads hundreds of thousands of rows. Either way the
    recording boundary is far too coarse to be the only thing that moves the bar, and a one-
    recording export is the ordinary case rather than the edge one.

    Parametrized over both targets because they report through different code: one folds a
    callback the analysis compute drives, the other counts its own rows, and a test written
    against whichever was in front of the author leaves the other frozen.
    """
    from detail.core import progress as progress_module

    # Both emits are throttled, as a per-row callback has to be. Zero the interval so every
    # one lands — the hook the throttle exposes for exactly this, rather than sleeping.
    monkeypatch.setattr(progress_module, "PROGRESS_INTERVAL_S", 0.0)
    _four_detections(processed_recording)

    seen = []
    _export_run(
        processed_recording,
        getattr(processed_recording, run_attribute),
        target_id,
        tmp_path,
        progress=lambda stage, fraction, detail="": seen.append(fraction),
    )
    assert any(
        0.0 < f < 1.0 for f in seen
    ), f"the bar never moved mid-recording: {seen}"
    assert seen == sorted(seen), f"the bar went backwards: {seen}"


def test_the_experiment_bundle_reports_from_inside_the_pack(
    processed_recording, tmp_path, monkeypatch
):
    """The bundle reported ``packing 0%`` before one long call and ``packing 100%`` after it, so
    an export run of a 17 GB experiment sat at zero for the whole pack (#916). It hands the
    callback to the pack now, which reports its own phases."""
    from detail.core import progress as progress_module

    monkeypatch.setattr(progress_module, "PROGRESS_INTERVAL_S", 0.0)
    seen = []
    exporters.write_export(
        ExportSubject("experiment", processed_recording.experiment.dir),
        [exporters.target("experiment", "bundle")],
        tmp_path,
        progress=lambda stage, fraction, detail="": seen.append((stage, fraction)),
    )
    stages = [stage for stage, _fraction in seen]
    assert stages.index("hashing") < stages.index("packing"), stages
    fractions = [fraction for _stage, fraction in seen]
    assert any(0.0 < f < 1.0 for f in fractions), seen
    assert fractions == sorted(fractions), f"the bar went backwards: {seen}"


def test_a_gzipped_bundle_is_gzipped_and_keeps_its_tar_name(
    processed_recording, tmp_path
):
    """The option the Experiment page's own dialog had, carried to the catalogue when that dialog
    was retired. Named `.tar` either way, as `experiment export --gzip` names it."""
    import gzip

    from detail.storage import export as export_store

    result = exporters.write_export(
        ExportSubject("experiment", processed_recording.experiment.dir),
        [exporters.target("experiment", "bundle")],
        tmp_path,
        gzip=True,
    )
    (archive,) = [result.dest / f for f in result.files if f.endswith(".tar")]
    with gzip.open(archive) as handle:
        handle.read(1)  # raises BadGzipFile on a plain tar
    assert export_store.read_manifest(archive)["n_files"] > 0


def test_the_video_export_moves_its_bar_inside_a_part(
    processed_recording, tmp_path, monkeypatch
):
    """A part was copied whole and reported when it ended, so a one-part recording sat at 0%
    until the export was done (#916). It copies through `copy_with_progress` now; the chunk is
    cut small here so this fixture's short video is several of them."""
    import functools

    from detail.core import progress as progress_module
    from detail.storage import video as video_store

    monkeypatch.setattr(progress_module, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        video_store,
        "copy_with_progress",
        functools.partial(video_store.copy_with_progress, chunk_size=256),
    )
    # The fixture records a part it never writes; the copy needs bytes, not a decodable video.
    (processed_recording.recording.dir / "video" / "part_001.mp4").write_bytes(
        bytes(4096)
    )
    subject = ExportSubject(
        "recording",
        processed_recording.experiment.dir,
        processed_recording.recording.id,
    )
    target = exporters.target("recording", "video")
    assert exporters.unavailable_reason(target, subject) is None
    seen = []
    result = exporters.write_export(
        subject,
        [target],
        tmp_path,
        progress=lambda stage, fraction, detail="": seen.append(fraction),
    )
    assert any(f.endswith(".mp4") for f in result.files), result.files
    assert any(0.0 < f < 1.0 for f in seen), f"the bar never moved mid-part: {seen}"
    assert seen == sorted(seen), f"the bar went backwards: {seen}"


def test_the_tracks_export_is_the_file_the_analysis_run_writes(
    processed_recording, tmp_path
):
    """One mechanism, two doors — asserted on the **bytes**, not on the column list.

    The hazard this guards is the comparison-figures class: a second implementation of "the
    same" export that answers differently, where both files look right on their own. Anything
    short of byte equality (a rounding difference, a column order, a line terminator) is a
    divergence a reader can only find by diffing two exports they had no reason to compare.
    """
    from detail.storage import location_export

    result = _export_run(
        processed_recording, processed_recording.tracking_run_id, "tracks", tmp_path
    )
    elsewhere = tmp_path / "via_the_analysis_door"
    processed_recording.recording.export_locations("track_01", elsewhere)

    assert (result.dest / "recording_001_tracks.csv").read_bytes() == (
        elsewhere / location_export.TRACKS_FILE
    ).read_bytes()


def test_tracks_are_written_per_recording(processed_recording, tmp_path):
    """One file per recording rather than a scope: the narrowing is picking a file."""
    processed_recording.add_recording("recording_002")
    result = _export_run(
        processed_recording, processed_recording.tracking_run_id, "tracks", tmp_path
    )
    assert sorted(f for f in result.files if f.endswith("_tracks.csv")) == [
        "recording_001_tracks.csv",
        "recording_002_tracks.csv",
    ]


def test_a_detection_run_exports_its_masks_without_re_encoding_them(
    processed_recording, tmp_path
):
    """The RLE that comes out is the string that went in.

    `mask_rle` already *is* the COCO counts string, so the export is a passthrough and
    `core/utils.decode_mask` is not on this path (`DESIGN_PLAN.md` §17). Note what this does and
    does not catch: a *lossless* decode-and-re-encode would pass it, because it produces the same
    counts — what it catches is the transformation that quietly is not lossless, a row-major
    re-encode or an h/w transposition, each of which yields a counts string that still parses.
    """
    import json

    from detail.storage.zarr_io import PartZarrReader

    result = _export_run(
        processed_recording, processed_recording.detection_run_id, "masks", tmp_path
    )
    document = json.loads(
        (result.dest / "recording_001_masks.json").read_text(encoding="utf-8-sig")
    )

    stored = PartZarrReader(
        processed_recording.recording.dir / "detections" / "det_01" / "part_001.zarr"
    ).read()
    assert [a["segmentation"]["counts"] for a in document["annotations"]] == [
        str(v) for v in stored["mask_rle"]
    ]
    assert document["annotations"][0]["segmentation"]["size"] == [
        processed_recording.H,
        processed_recording.W,
    ]


def test_an_exported_mask_and_its_box_share_one_coordinate_space(
    processed_recording, tmp_path
):
    """A COCO box is absolute pixels and ours is normalized, so the conversion is the risk.

    Asserted by decoding the mask and checking the box **contains** it: a normalized box left
    unscaled, an x/y swap, or an h/w transposition all still produce a plausible four-number
    list, and none of them survives this.
    """
    import json

    from detail.core.utils import decode_mask

    result = _export_run(
        processed_recording, processed_recording.detection_run_id, "masks", tmp_path
    )
    document = json.loads(
        (result.dest / "recording_001_masks.json").read_text(encoding="utf-8-sig")
    )
    annotation = document["annotations"][0]

    mask = decode_mask(
        annotation["segmentation"]["counts"], tuple(annotation["segmentation"]["size"])
    )
    rows, columns = mask.nonzero()
    x, y, width, height = annotation["bbox"]
    assert x <= columns.min() and columns.max() < x + width
    assert y <= rows.min() and rows.max() < y + height
    assert annotation["area"] == pytest.approx(float(mask.sum()))


def test_the_mask_export_records_both_resolutions(processed_recording, tmp_path):
    """Coordinates are in the mask resolution; a consumer wanting video pixels needs to know."""
    import json

    result = _export_run(
        processed_recording, processed_recording.detection_run_id, "masks", tmp_path
    )
    info = json.loads(
        (result.dest / "recording_001_masks.json").read_text(encoding="utf-8-sig")
    )["info"]
    assert info["mask_resolution"] == [processed_recording.H, processed_recording.W]
    assert info["video_resolution"] == [processed_recording.W, processed_recording.H]
    assert info["detection_run_id"] == "det_01"


def test_feature_vectors_come_off_an_embedding_run(
    processed_recording, make_embedding_run, tmp_path
):
    """The `vectors` target on the run that produces them.

    This was parametrized over `embedding` and `derive`, because a `derive` run wrote the same
    `emb_NN` and was once refused by a check spelled `module_id == "embedding"`. The `derive`
    module has been removed; what a record it left behind is told is the test below.
    """
    module_id = "embedding"
    import numpy as np
    from detail.storage import runs as runs_store

    recording_id = processed_recording.recording.id
    make_embedding_run(
        processed_recording.recording.dir,
        np.zeros((3, 4), dtype=np.float32),
        np.arange(3),
        np.arange(3) * 1000,
        embed_run_id="emb_07",
    )
    runs_store.write_run(
        processed_recording.experiment.dir,
        "emb_07",
        {
            "id": "emb_07",
            "module_id": module_id,
            "recording_ids": [recording_id],
            "per_recording": {recording_id: {"run_id": "emb_07", "status": "complete"}},
            "status": "complete",
        },
    )

    subject = _run_subject(processed_recording, "emb_07")
    assert "vectors" in {t.id for t in exporters.available_targets(subject)}
    result = _export_run(processed_recording, "emb_07", "vectors", tmp_path)
    assert "embeddings.npy" in result.files
    assert "embeddings_index.csv" in result.files


def test_a_run_made_by_a_removed_module_is_refused_by_name(
    processed_recording, make_embedding_run
):
    """A record outlives the module that wrote it, and the refusal has to say that.

    "This is a derive run; these come from an embedding run" reads as the user having picked
    the wrong run. The run is the right kind of data; this version cannot read it as one.
    """
    import numpy as np
    from detail.storage import runs as runs_store

    recording_id = processed_recording.recording.id
    make_embedding_run(
        processed_recording.recording.dir,
        np.zeros((3, 4), dtype=np.float32),
        np.arange(3),
        np.arange(3) * 1000,
        embed_run_id="emb_07",
    )
    runs_store.write_run(
        processed_recording.experiment.dir,
        "emb_07",
        {
            "id": "emb_07",
            "module_id": "derive",
            "recording_ids": [recording_id],
            "per_recording": {recording_id: {"run_id": "emb_07", "status": "complete"}},
            "status": "complete",
        },
    )

    subject = _run_subject(processed_recording, "emb_07")
    listed = {t.id for t in exporters.available_targets(subject)}
    assert "vectors" not in listed
    vectors = {t.id: t for t in exporters.targets_for(subject)}["vectors"]
    reason = exporters.unavailable_reason(vectors, subject)
    assert "'derive'" in reason and "no longer has" in reason


def test_a_run_of_the_wrong_kind_says_which_kind_it_needs(processed_recording):
    """The sentence names the run that produces it *and* the run you are on.

    "Unavailable" would leave the reader unable to act; "masks come from a detection run; this
    is a tracking run" tells them where to go.
    """
    tracking = _run_subject(processed_recording, processed_recording.tracking_run_id)
    reason = exporters.unavailable_reason(exporters.target("run", "masks"), tracking)
    assert reason is not None
    assert "detection run" in reason and "tracking run" in reason


def test_every_run_target_is_offered_on_the_run_that_makes_it(processed_recording):
    """Both new targets, from the catalogue, on a real run record — not from the table."""
    for run_id, expected in (
        (processed_recording.tracking_run_id, "tracks"),
        (processed_recording.detection_run_id, "masks"),
    ):
        subject = _run_subject(processed_recording, run_id)
        runnable = {t.id for t in exporters.available_targets(subject)}
        assert expected in runnable
        assert "vectors" not in runnable, "feature vectors come from an embedding run"


# --------------------------------------------------------------------------- #
# One target's failure is not the export's
# --------------------------------------------------------------------------- #
# **The class.** *A loop over N independent pieces of work where one exception ends all of
# them.* Inline that is tolerable — you read the traceback and retry. As a run you walked away
# from for an hour it is the difference between "7 of 8 done, clips failed because a video
# moved" and an empty folder with no manifest saying what happened. The failure is invisible to
# a happy-path test, because a loop that never raises behaves identically either way.


def test_a_clip_knob_does_not_break_the_target_beside_it(authored_cohort, tmp_path):
    """Ticking two targets means **one** options dict reaching both of their runners.

    Reported from the app as `_run_performance() got an unexpected keyword argument 'renders'`:
    every target worked alone, and the tests exercised them alone, so the one runner that could
    not ignore another's knobs only failed in combination. The signature guard in
    `test_export_registry.py` covers all thirteen; this is the behaviour it protects.
    """
    from detail.storage import activity
    from detail.storage import analysis_runs as analysis_store

    # `performance` is only offered for a run that kept an activity log, and the fixture's did
    # not — so without this the pair never runs and the test passes by never reaching the bug.
    run_dir = analysis_store.analysis_dir(authored_cohort.experiment.dir, "an_01")
    activity.ActivityLog(run_dir).sample(
        {"unit": "r1", "phase": "working", "frames_done": 1, "frames_total": 1},
        force=True,
    )

    subject = _subject(authored_cohort)
    targets = [
        exporters.target("analysis_run", "instances"),
        exporters.target("analysis_run", "performance"),
    ]
    assert (
        exporters.unavailable_reason(targets[1], subject) is None
    ), "the fixture must actually be able to run the performance target"
    result = exporters.write_export(
        subject, targets, tmp_path, renders=("overlay",), crop=True, size=320
    )

    assert result.status == "complete", [o.error for o in result.failures]
    assert "instances.csv" in result.files


def _exploding(target_id: str, message: str = "the video moved"):
    """A copy of a real target whose `run` raises — the catalogue is frozen, so rebuild it."""
    import dataclasses

    real = exporters.target("analysis_run", target_id)

    def boom(*a, **k):
        raise FileNotFoundError(message)

    return dataclasses.replace(real, run=boom)


def test_one_failed_target_does_not_stop_the_others(two_clusters, tmp_path):
    """The others still write, the manifest still lands, and the run says `partial`."""
    subject = _subject(two_clusters)
    targets = [
        exporters.target("analysis_run", "instances"),
        _exploding("ethogram"),
    ]
    result = exporters.write_export(subject, targets, tmp_path)

    assert result.status == "partial"
    assert "instances.csv" in result.files, "the healthy target still wrote"
    assert not any(
        f.endswith("_ethogram.csv") for f in result.files
    ), "the failed target wrote nothing"

    manifest = exporters.read_manifest(result.dest)
    assert manifest["status"] == "partial"
    outcomes = {o["target"]: o for o in manifest["outcomes"]}
    assert outcomes["instances"]["status"] == "complete"
    assert outcomes["ethogram"]["status"] == "failed"
    assert (
        "the video moved" in outcomes["ethogram"]["error"]
    ), "a failed target has to say why — the whole point of keeping the manifest"


def test_every_target_failing_is_a_failed_export(two_clusters, tmp_path):
    """`partial` and `failed` share a rendering, so a test for one passes against the other."""
    subject = _subject(two_clusters)
    result = exporters.write_export(
        subject, [_exploding("instances"), _exploding("ethogram")], tmp_path
    )

    assert result.status == "failed"
    assert [o.target_id for o in result.failures] == ["instances", "ethogram"]
    assert exporters.read_manifest(result.dest)["status"] == "failed"


def test_a_failed_export_is_not_reported_as_an_empty_one(two_clusters, tmp_path):
    """Two different states that would otherwise share one encoded value.

    "Nothing to write" means every target ran fine and had nothing to say — actionable only by
    changing the scope. A target that *failed* wrote nothing for a different reason entirely, and
    collapsing them would send the reader to look at their filter instead of at the error.
    """
    subject = _subject(two_clusters)
    result = exporters.write_export(subject, [_exploding("instances")], tmp_path)

    assert not result.is_empty, "a failure is not an empty export"
    assert result.status == "failed"
    assert "empty" not in (exporters.read_manifest(result.dest) or {})


def test_a_clean_export_carries_no_failure_block(two_clusters, tmp_path):
    """Guard the guard: if every export said `status`, its presence would mean nothing."""
    result = _export(two_clusters, ["instances"], tmp_path)
    manifest = exporters.read_manifest(result.dest)

    assert "status" not in manifest and "outcomes" not in manifest
    assert result.status == "complete"
    assert result.failures == ()


def test_the_bundle_and_its_folder_are_named_after_the_experiment(
    processed_recording, tmp_path
):
    """The bundle read ``id`` from ``config.yaml``, a key it never has, so every archive was
    ``experiment.tar`` and every export folder was named after the experiment's directory. The
    name is the user's, and it is what they look for once the archive leaves its folder (#916).
    A space is not a path character anywhere, but it is spelled as every other export does.
    """
    experiment = processed_recording.experiment
    experiment.set_metadata(name="Top Down Study")
    assert (
        experiment.dir.name != "Top Down Study"
    ), "the fixture must tell the two apart"

    result = exporters.write_export(
        ExportSubject("experiment", experiment.dir),
        [exporters.target("experiment", "bundle")],
        tmp_path,
    )

    assert "Top_Down_Study.tar" in result.files, result.files
    assert result.dest.name.startswith("Top_Down_Study_export_"), result.dest.name


def test_a_recording_with_only_a_calibration_can_export_its_geometry(
    authored_cohort, tmp_path
):
    """The geometry file carries the calibration, so a calibration alone is something to write.

    `export list` said "nothing has been drawn" while the spatial editor's Export and
    `recording export-spatial` both wrote the file (#31, finding 52).
    """
    from detail.storage import experiment as experiment_store
    from detail.storage import spatial as spatial_store

    subject = ExportSubject("recording", authored_cohort.experiment.dir, "r1")
    geometry = exporters.target("recording", "geometry")
    assert exporters.unavailable_reason(geometry, subject) == (
        "nothing has been drawn or calibrated on this recording yet"
    )
    directory = experiment_store.recording_path(authored_cohort.experiment.dir, "r1")
    spatial_store.write_calibration_scalar(directory, 2.5, (40, 40))
    assert exporters.unavailable_reason(geometry, subject) is None

    result = exporters.write_export(subject, [geometry], tmp_path)
    rows = _rows(result.dest / "r1_spatial.csv")
    assert [r["kind"] for r in rows] == ["calibration"]


@pytest.mark.parametrize("call", ["run_export", "estimate_export"])
def test_the_experiment_refuses_a_scope_before_it_reserves_a_run(
    authored_cohort, tmp_path, call
):
    """Each way into an export checks on its own: the CLI asks `estimate_export` first, so a
    script calling `run_export` directly is covered only by this."""
    from detail.api.experiment import Experiment

    experiment = Experiment(authored_cohort.experiment.dir)
    with pytest.raises(exporters.ExportError, match="cannot be narrowed to a frame"):
        getattr(experiment, call)(
            "analysis_run",
            "an_01",
            ["instances"],
            tmp_path / "out",
            scope=ExportScope(frames=(0, 3)),
        )
    assert experiment.export_runs == []
    assert not (tmp_path / "out").exists()
