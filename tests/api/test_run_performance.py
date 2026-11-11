"""The performance report: this machine, and how long the run took on it.

**What it is for.** Three hours of detection is fast or slow depending on the box, and the
comparison people need is across labs — the same run on an older GPU and a newer one can differ
by hours. So the report carries the machine beside the numbers and is a file somebody can
paste. **Local only; no telemetry** — a deliberate decision, and this suite asserts the
absence rather than trusting it.

**Its timings are derived, not recorded.** Nothing in the tree stores when a recording started or
finished, so the report reads the activity log the run already writes. That is why the two are one
piece of work: a second bookkeeping path would be a second thing to keep true.
"""

from __future__ import annotations

import datetime as dt

import pytest

from detail.api import exporters
from detail.api.exporters import ExportSubject
from detail.storage import activity
from detail.storage import runs as runs_store

_BASE = dt.datetime(2026, 9, 10, 13, 30, tzinfo=dt.timezone.utc)


def _run_with_log(tmp_path, units, run_dir=None):
    """A run dir whose log says each unit ran for a given span at a given frame count."""
    run_dir = run_dir or runs_store.run_dir(tmp_path, "det_07")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = activity.ActivityLog(run_dir, interval_s=0.0)
    real = activity._now_iso
    try:
        for unit, (start_s, end_s), frames in units:
            for k, offset in enumerate((start_s, end_s)):
                when = _BASE + dt.timedelta(seconds=offset)
                activity._now_iso = lambda w=when: w.isoformat(timespec="seconds")
                log.sample(
                    activity.make_row(
                        elapsed_s=float(offset),
                        unit=unit,
                        unit_index=1,
                        unit_total=len(units),
                        phase="detecting",
                        done=frames if k else 0,
                        total=frames,
                    )
                )
    finally:
        activity._now_iso = real
    return run_dir


def _write(tmp_path, units):
    _run_with_log(tmp_path, units)
    # A detection run: its log's total is frames, so the report may show a rate.
    runs_store.write_run(tmp_path, "det_07", {"id": "det_07", "module_id": "detection"})
    dest = tmp_path / "out"
    dest.mkdir(exist_ok=True)
    written = exporters.target("run", "performance").run(
        ExportSubject("run", tmp_path, "det_07"), dest
    )
    return dest, written, (dest / "performance.md").read_text(encoding="utf-8-sig")


def test_a_slow_recording_is_visible_as_a_number(tmp_path):
    """The row that makes the report worth having.

    A user's `recording_004` ran beside the next one because of an orphaned worker (#289) and took
    3.6x as long. In a run summary that reported only the total, that is a slightly slow run; per
    unit, it is one row that does not match its neighbours — which is a diagnosis.
    """
    _dest, _written, report = _write(
        tmp_path,
        [
            ("recording_001", (0, 540), 88412),
            ("recording_002", (560, 1130), 91004),
            ("recording_004", (1150, 3470), 102880),
        ],
    )
    rows = {
        line.split("|")[1].strip(): line
        for line in report.splitlines()
        if line.startswith("| recording_")
        and line.count("|") == 8  # the per-unit table
    }
    fast = float(rows["recording_001"].split("|")[5])
    slow = float(rows["recording_004"].split("|")[5])
    assert slow < fast / 3, f"{slow} vs {fast} — the outlier does not stand out"


def test_each_unit_is_charged_only_its_own_span(tmp_path):
    """The gap between one unit finishing and the next starting belongs to neither.

    Folding it into a recording would make whichever ran first look slow — and on a run whose
    setup is minutes, that is the whole apparent difference between recordings.
    """
    _dest, _written, report = _write(
        tmp_path, [("a", (0, 100), 1000), ("b", (600, 700), 1000)]
    )
    spans = [line for line in report.splitlines() if line.startswith("| a |")]
    assert "1m40s" in spans[0], spans  # 100s, not 700s


def test_the_machine_is_named_beside_the_numbers(tmp_path):
    _dest, _written, report = _write(tmp_path, [("a", (0, 100), 1000)])
    assert "## Machine" in report
    for field in ("GPU", "CPU", "RAM (GB)", "OS", "detail", "torch"):
        assert f"| {field} |" in report, field


def test_ram_is_a_real_number_not_just_a_present_row(tmp_path):
    """`psutil` is a core dependency (#394), so on a normal install RAM is a measurement, not
    the "unknown" a plain install always produced. A test that only checked the row existed —
    as the one above does — would pass forever on a value that never stopped being "unknown";
    this one reads the number itself.
    """
    _dest, _written, report = _write(tmp_path, [("a", (0, 100), 1000)])
    lines = {
        line.split("|")[1].strip(): line.split("|")[2].strip()
        for line in report.splitlines()
        if line.startswith("| ")
    }
    assert lines["RAM (GB)"] != "unknown"
    assert float(lines["RAM (GB)"]) > 0


def test_the_data_behind_the_table_is_written_too(tmp_path):
    """A report that shipped only its rendering would be the Save-PNG button the catalogue's
    rule exists to prevent — so the CSV and the YAML are `data_files`, and both land."""
    dest, written, _report = _write(tmp_path, [("a", (0, 100), 1000)])
    assert {p.name for p in written} == {
        "performance.md",
        "timings.csv",
        "phases.csv",
        "machine.yaml",
    }
    assert (
        (dest / "timings.csv")
        .read_text(encoding="utf-8-sig")
        .startswith("unit,started,seconds,frames,fps,peak_ram_gb,peak_gpu_gb")
    )
    assert (
        (dest / "phases.csv")
        .read_text(encoding="utf-8-sig")
        .startswith("unit,phase,seconds")
    )


def test_a_run_with_no_log_reports_nothing_rather_than_guessing(tmp_path):
    """Runs from before the log existed have none, and there is nothing to reconstruct one from.

    A report that filled those in from `created_at` and the current time would be plausible and
    wrong — which is the worse failure, because it is the one nobody checks.
    """
    run_dir = runs_store.run_dir(tmp_path, "det_07")
    run_dir.mkdir(parents=True)
    dest = tmp_path / "out"
    dest.mkdir()
    exporters.target("run", "performance").run(
        ExportSubject("run", tmp_path, "det_07"), dest
    )
    report = (dest / "performance.md").read_text(encoding="utf-8-sig")
    assert "no activity log" in report
    assert "unknown" in report


def test_the_target_says_why_it_cannot_run(tmp_path):
    """Grey out, don't hide — and the greyed row names the missing thing."""
    run_dir = runs_store.run_dir(tmp_path, "det_07")
    run_dir.mkdir(parents=True)
    reason = exporters.unavailable_reason(
        exporters.target("run", "performance"),
        ExportSubject("run", tmp_path, "det_07"),
    )
    assert reason and "activity log" in reason


def test_a_run_with_a_log_can_run_it(tmp_path):
    """The pair. A reason that is always truthy greys out every run forever, and the suite above
    would not notice — every assertion there is about a run that has no log."""
    _run_with_log(tmp_path, [("a", (0, 100), 1000)])
    assert (
        exporters.unavailable_reason(
            exporters.target("run", "performance"),
            ExportSubject("run", tmp_path, "det_07"),
        )
        is None
    )


@pytest.mark.parametrize("kind", ["run", "analysis_run", "distill"])
def test_every_run_kind_has_the_report(kind):
    """Asked for by name for clustering as well as detection — temporal clustering can also take
    hours, and that is the number people want to know. A per-kind exception is how a
    summary becomes wrong for half the tool."""
    assert exporters.target(kind, "performance") is not None


def test_the_report_makes_no_network_call():
    """The line drawn in the meeting, asserted rather than trusted.

    Read from the source: a behavioural test would have to be offline to prove anything, and the
    thing being promised is that the code has no such call in it at all.
    """
    import ast
    import pathlib

    from detail.core import hostinfo

    tree = ast.parse(pathlib.Path(hostinfo.__file__).read_text(encoding="utf-8-sig"))
    names = {
        node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
        for node in ast.walk(tree)
    }
    for banned in ("urlopen", "requests", "post", "urlretrieve", "httpx", "socket"):
        assert banned not in names, f"hostinfo reaches the network via {banned}"


# --------------------------------------------------------------------------- #
# the machine the run ran on, its peak memory, and its phases
# --------------------------------------------------------------------------- #
_JOB_MACHINE = {"gpu": "Job GPU 5090", "torch": "2.9.0", "os": "JobOS", "ram_gb": 64}
_EXPORTER_MACHINE = {"gpu": "Exporting laptop", "torch": "2.9.0", "os": "LapOS"}


def _job_facts(job_dir, *, ram_gb, gpu_gb):
    from detail.storage import state_yaml

    job_dir.mkdir(parents=True, exist_ok=True)
    state_yaml.write_state_yaml(job_dir / "machine.yaml", _JOB_MACHINE)
    state_yaml.write_state_yaml(
        job_dir / "resources.yaml",
        {
            "peak_ram_bytes": int(ram_gb * 2**30),
            "gpu_backend": "cuda" if gpu_gb else None,
            "peak_gpu_reserved_bytes": int(gpu_gb * 2**30) if gpu_gb else None,
        },
    )


def _processing_run(tmp_path):
    """A processing run whose recording ran in its own worker, as on disk: the coordinator's log
    in ``runs/run_001/``, the worker's facts under ``recordings/recording_001/detections/``.
    """
    from detail.storage import experiment as experiment_store

    run_dir = _run_with_log(tmp_path, [("recording_001", (0, 100), 1000)])
    runs_store.write_run(
        tmp_path,
        "det_07",
        {
            "id": "det_07",
            "module_id": "detection",
            "per_recording": {
                "recording_001": {"run_id": "det_01", "status": "complete"}
            },
        },
    )
    job_dir = (
        experiment_store.recording_path(tmp_path, "recording_001")
        / "detections"
        / "det_01"
    )
    return run_dir, job_dir


def _export(tmp_path, kind="run", run_id="det_07"):
    dest = tmp_path / "out"
    dest.mkdir(exist_ok=True)
    exporters.target(kind, "performance").run(
        ExportSubject(kind, tmp_path, run_id), dest
    )
    return dest, (dest / "performance.md").read_text(encoding="utf-8-sig")


def test_the_machine_is_the_one_the_run_ran_on_not_the_exporter(tmp_path, monkeypatch):
    """A run exported on a laptop after it ran on a GPU box used to report the laptop — the
    wrong hardware beside right timings, the worst kind of wrong, since nothing looks off.
    """
    from detail.core import hostinfo

    monkeypatch.setattr(hostinfo, "collect", lambda *a, **k: dict(_EXPORTER_MACHINE))
    _run_dir, job_dir = _processing_run(tmp_path)
    _job_facts(job_dir, ram_gb=12.5, gpu_gb=7.25)
    _dest, report = _export(tmp_path)
    assert "Job GPU 5090" in report
    assert "Exporting laptop" not in report
    row = next(
        line for line in report.splitlines() if line.startswith("| recording_001")
    )
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[-2:] == ["12.5", "7.25"], cells


def test_a_run_that_recorded_no_machine_says_the_table_is_this_computer(
    tmp_path, monkeypatch
):
    """The pair: with no record the exporter's machine is all there is, and the heading must say
    so rather than pass it off as the machine that did the work."""
    from detail.core import hostinfo

    monkeypatch.setattr(hostinfo, "collect", lambda *a, **k: dict(_EXPORTER_MACHINE))
    _processing_run(tmp_path)
    _dest, report = _export(tmp_path)
    assert "Exporting laptop" in report
    assert "did not record the one it ran on" in report


def test_a_distill_reports_its_stages_without_a_fake_fps(tmp_path):
    """A distill's log counts a fraction scaled to 1000, not frames — so a rate computed from it
    would be invented. Its stages are its rows, each with the memory its own job used.
    """
    from detail.storage import datasets, distill_runs, models

    run_dir = distill_runs.run_dir(tmp_path, "distill_001")
    run_dir.mkdir(parents=True)
    distill_runs.write_run(
        tmp_path,
        "distill_001",
        {
            "id": "distill_001",
            "stages": {
                "label": {"status": "complete", "dataset_id": "ds_01"},
                "train": {"status": "complete", "model_id": "model_001"},
            },
        },
    )
    _job_facts(datasets.dataset_dir(tmp_path, "ds_01"), ram_gb=3.0, gpu_gb=6.0)
    _job_facts(models.model_dir(tmp_path, "model_001"), ram_gb=9.0, gpu_gb=7.5)
    log = activity.ActivityLog(run_dir, interval_s=0.0)
    real = activity._now_iso
    try:
        for offset, unit, phase in [
            (0, "labeling", "starting labeling"),
            (300, "labeling", "labeling frames"),
            (600, "training", "starting training"),
            (660, "training", "epoch 1/100"),
            (4200, "training", "epoch 100/100"),
        ]:
            when = _BASE + dt.timedelta(seconds=offset)
            activity._now_iso = lambda w=when: w.isoformat(timespec="seconds")
            log.sample(
                activity.make_row(
                    elapsed_s=float(offset),
                    unit=unit,
                    phase=phase,
                    done=500,
                    total=1000,
                )
            )
    finally:
        activity._now_iso = real
    dest, report = _export(tmp_path, kind="distill", run_id="distill_001")
    rows = {
        line.split("|")[1].strip(): [c.strip() for c in line.split("|")[1:-1]]
        for line in report.splitlines()
        if line.startswith(("| labeling |", "| training |")) and line.count("|") == 8
    }
    assert rows["labeling"][3:] == ["—", "—", "3.0", "6.0"], rows
    assert rows["training"][3:] == ["—", "—", "9.0", "7.5"], rows
    phases = (dest / "phases.csv").read_text(encoding="utf-8-sig")
    assert "training,training epochs,3540.0" in phases, phases


def _timings(dest) -> list[dict]:
    import csv

    with open(dest / "timings.csv", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_an_analysis_whose_log_counts_a_fraction_reports_no_rate(tmp_path):
    """Most analyses count their bar as a fraction out of 1000 (``core.progress.split_units``).
    Divided by seconds that is a plausible fps for a run that has no frame rate — the shape of
    the keypoint-tracking bug, where a two-pass count doubled the benchmark's fps. The module
    says what its log counts; here it says not frames, so the cells are blank."""
    from detail.storage import analysis_runs

    run_dir = analysis_runs.analysis_dir(tmp_path, "an_01")
    analysis_runs.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "temporal_clustering"}
    )
    _run_with_log(tmp_path, [("", (0, 400), 1000)], run_dir=run_dir)
    dest, report = _export(tmp_path, kind="analysis_run", run_id="an_01")
    (row,) = _timings(dest)
    assert row["seconds"] == "400.0"
    assert (row["frames"], row["fps"]) == ("0", ""), row
    unit_rows = [line for line in report.splitlines() if line.count("|") == 8]
    assert unit_rows and not any(
        "2.5" in line for line in unit_rows
    )  # 1000 / 400, the rate that would have been invented


def test_a_run_with_no_known_module_reports_no_rate(tmp_path):
    """The default is the blank, not the division: a record naming no module this build has
    cannot say what its log counted."""
    _run_with_log(tmp_path, [("recording_001", (0, 100), 1000)])
    runs_store.write_run(tmp_path, "det_07", {"id": "det_07", "module_id": "gone"})
    dest, _report = _export(tmp_path)
    (row,) = _timings(dest)
    assert (row["frames"], row["fps"]) == ("0", ""), row


def test_a_detection_run_reports_its_frames_and_rate(tmp_path):
    """The pair: a module whose log counts frames keeps its rate."""
    dest, _written, _report = _write(tmp_path, [("recording_001", (0, 100), 3000)])
    (row,) = _timings(dest)
    assert (row["frames"], row["fps"]) == ("3000", "30.0"), row
