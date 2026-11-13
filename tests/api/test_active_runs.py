"""What is running, across every experiment.

**The class this closes.** Five detail pages each knew whether the one run they were showing was
still going; nothing knew whether *anything* was. So "did last night's detection finish" meant
remembering which experiment it was in. The records were always there — every run kind has a
`discover_*` and a status — and nobody was asking across them.

**Imports** are the family that is not a run (the last section below), and this file holds the
listing half of the *news delivered where nobody is looking* class written up in
`tests/gui/test_active_processes.py`. A listing that drops finished rows cannot be looked at in time
when what it lists finishes inside a second, which an import on a local disk does, so a finished
import stays listed — every family does now, until archived (§ active-run archiving). The two
things that are not a finished import, a recording no import ever touched (whose `import_status`
property defaults a missing key to "ready") and a metadata-only shell, are each pinned with a real
finished import beside them, so neither exclusion can pass by listing nothing.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from detail.api import active
from detail.api.experiment import Experiment
from detail.storage import recording as recording_store
from detail.storage import runs as runs_store


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return tmp_path


def _run(experiment_dir, run_id, status, created_at, module="detection", **extra):
    runs_store.run_dir(experiment_dir, run_id).mkdir(parents=True, exist_ok=True)
    runs_store.write_run(
        experiment_dir,
        run_id,
        {
            "id": run_id,
            "module_id": module,
            "recording_ids": [],
            "config": {},
            "per_recording": {},
            "created_at": created_at,
            "status": status,
            **extra,
        },
    )


def test_it_spans_experiments(workspace):
    """The whole point. A per-experiment listing already existed; what did not was one that
    crosses them, which is the only kind that answers "is anything running"."""
    for name in ("alpha", "beta"):
        exp = Experiment.create(name, workspace)
        _run(exp.dir, "det_01", "running", "2026-09-10T12:00:00Z")

    found = active.list_active()
    assert {r.experiment_name for r in found} == {"alpha", "beta"}


def test_finished_runs_stay_listed_too(workspace):
    """Every family keeps its finished runs now, not just export/import — until archived
    (`test_archiving_a_run_removes_it_from_the_listing` below)."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "complete", "2026-09-10T12:00:00Z")
    _run(exp.dir, "det_02", "running", "2026-09-10T12:30:00Z")

    assert {r.run_id for r in active.list_active()} == {"det_01", "det_02"}


def test_a_local_run_shows_where_as_local(workspace):
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "running", "2026-09-10T12:00:00Z")
    assert active.list_active()[0].where == "local"
    assert active.list_active()[0].cost == ""


def test_a_remote_run_shows_where_and_a_configured_cost(workspace):
    exp = Experiment.create("alpha", workspace)
    _run(
        exp.dir,
        "det_01",
        "running",
        "2026-09-10T12:00:00Z",
        compute={"backend": "remote", "remote_target": "tillicum"},
        estimated_cost_usd=1.5,
    )
    row = active.list_active()[0]
    assert row.where == "remote: tillicum"
    assert row.cost == "$1.50"


def test_a_multi_recording_run_shows_its_aggregate_cost(workspace):
    """A `run`-kind record never carries a bare `estimated_cost_usd` of its own — only
    `RunOrchestrator._write_record`'s aggregate `total_estimated_cost_usd`, summed from
    each per-recording entry's own `estimated_cost_usd` (`detail/api/experiment.py`).
    `test_a_remote_run_shows_where_and_a_configured_cost` above writes the bare key
    directly, which never exercises this path — the run-detail page already reads the
    aggregate key (`detail/gui/pages.py`), so the listing here has to as well."""
    exp = Experiment.create("alpha", workspace)
    _run(
        exp.dir,
        "run_006",
        "complete",
        "2026-09-10T12:00:00Z",
        compute={"backend": "remote", "remote_target": "tillicum"},
        per_recording={
            "recording_001": {"status": "complete", "estimated_cost_usd": 0.11},
        },
        total_estimated_cost_usd=0.11,
    )
    row = active.list_active()[0]
    assert row.cost == "$0.11"


def test_a_remote_run_with_no_rate_configured_shows_where_but_no_cost(workspace):
    exp = Experiment.create("alpha", workspace)
    _run(
        exp.dir,
        "det_01",
        "running",
        "2026-09-10T12:00:00Z",
        compute={"backend": "remote", "remote_target": "tillicum"},
    )
    row = active.list_active()[0]
    assert row.where == "remote: tillicum"
    assert row.cost == ""


def test_a_finished_run_shows_its_duration(workspace):
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "complete", "2026-09-10T12:00:00Z", duration_s=125)

    row = active.list_active()[0]
    assert row.duration == "2m05s"


def test_a_run_with_no_duration_yet_shows_a_blank_duration(workspace):
    """Still going, or a legacy record from before this field existed — either way, a blank cell
    rather than a formatted ``None``."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "running", "2026-09-10T12:00:00Z")

    assert active.list_active()[0].duration == ""


def test_failed_runs_are_listed_with_the_active_ones(workspace):
    """A run that stopped overnight is exactly as urgent as one still going, and harder to find:
    an active run at least has a page you remember launching."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "failed", "2026-09-10T12:00:00Z")

    listed = active.list_active()
    assert [r.run_id for r in listed] == ["det_01"]
    assert listed[0].is_failed

    assert active.list_active(include_failed=False) == []


def test_newest_first(workspace):
    """The run being asked about is almost always the last one started."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "old", "running", "2026-09-01T09:00:00Z")
    _run(exp.dir, "new", "running", "2026-09-10T09:00:00Z")

    assert [r.run_id for r in active.list_active()] == ["new", "old"]


def test_a_run_with_no_timestamp_sorts_last_rather_than_crashing(workspace):
    """A record written before `created_at` existed, or one half-written. The listing is a view;
    a view that raises because one row is odd tells the user less than one missing a row.
    """
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "stamped", "running", "2026-09-10T09:00:00Z")
    _run(exp.dir, "bare", "running", "")

    assert [r.run_id for r in active.list_active()] == ["stamped", "bare"]


def test_one_unreadable_experiment_does_not_empty_the_list(workspace, monkeypatch):
    """The failure that would make this useless the first time a share hiccups.

    Asserted because the alternative is silent: a `list_active` that raised would leave the page
    showing "nothing is running" to somebody with four jobs in flight — which is not an error
    message, it is a wrong answer.
    """
    good = Experiment.create("good", workspace)
    _run(good.dir, "det_01", "running", "2026-09-10T12:00:00Z")
    Experiment.create("bad", workspace)

    real = Experiment.processing_runs.fget

    def explode(self):
        if self.dir.name == "bad":
            raise PermissionError("the share went away")
        return real(self)

    monkeypatch.setattr(Experiment, "processing_runs", property(explode), raising=False)
    assert [r.run_id for r in active.list_active()] == ["det_01"]


def test_a_row_carries_what_it_takes_to_open_it(workspace):
    """The page routes rather than reports, so a row has to name the experiment *and* the kind —
    opening a run against the wrong experiment reads its record from the wrong place."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "running", "2026-09-10T12:00:00Z", module="detection")

    (run,) = active.list_active()
    assert run.kind == "run"
    assert run.experiment_dir == exp.dir
    assert run.label == "detection"


# --------------------------------------------------------------------------- #
# One table, and the surfaces that read it
# --------------------------------------------------------------------------- #
# **The class.** *A family of run is added to one enumeration and not its siblings.* Six
# families each answered three questions — what is running, what page shows it, how do I stop
# it — from three hand-written lists, and they drifted in both directions: the double-click
# router knew 2 of 6, the quit guard knew 4. Nothing failed either time, because a row nobody
# can open and a row nobody clicks look identical, and a quit prompt that does not appear is
# indistinguishable from having nothing to say.


def test_every_run_kind_names_a_cancel_method_that_exists():
    """The table hands the quit path a method name; a typo there is a silent no-cancel."""
    for spec in active.RUN_KINDS:
        assert hasattr(Experiment, spec.cancel), (
            f"run kind {spec.kind!r} names Experiment.{spec.cancel}, which does not exist — "
            f"quitting during one of these would fail to cancel it"
        )


def test_every_single_job_run_kind_is_listed():
    """`_JOB_RUN_KINDS` is the attach/reconcile table; this one is the "what is running" table.

    They cover different sets on purpose — processing and distill are coordinator-backed and are
    outside the first — but every kind in the first must appear in the second, or a run you can
    reattach to is a run the workspace cannot tell you is going. The naming differs by one:
    `_JOB_RUN_KINDS` says `analysis` where a navigation kind says `analysis_run`.
    """
    listed = {spec.kind for spec in active.RUN_KINDS}
    for kind in Experiment._JOB_RUN_KINDS:
        expected = "analysis_run" if kind == "analysis" else kind
        assert (
            expected in listed
        ), f"{kind!r} can be attached and reconciled but never appears in the active listing"


def test_the_listing_covers_the_coordinator_kinds_too():
    """The two that are *not* in `_JOB_RUN_KINDS`, named so dropping one fails here."""
    listed = {spec.kind for spec in active.RUN_KINDS}
    assert {"run", "distill"} <= listed


@pytest.mark.parametrize(
    "ref_name",
    ["AnalysisRunRef", "DatasetRef", "ModelRef", "ExportRunRef"],
)
def test_every_listing_ref_carries_created_at(ref_name):
    """A listing sorted on `created_at` needs every kind to have one.

    Four of the five did not. `_runs_of` reads the field per record, so the Active-runs
    "started" column was blank for those kinds and `list_active`'s sort put them last in
    arbitrary order — found by rendering the page against the real workspace, because a blank
    cell and an unsorted tail both look like data.

    Asserted on the **ref type**, which is where the field was missing. A record that happens to
    have no `created_at` is a different and legitimate state, and asserting on a built
    experiment would conflate the two.
    """
    import dataclasses

    from detail.api import experiment as experiment_mod

    fields = {f.name for f in dataclasses.fields(getattr(experiment_mod, ref_name))}
    assert "created_at" in fields, (
        f"{ref_name} carries no `created_at`, so its rows show a blank 'started' column and "
        f"sort last in `list_active`"
    )


@pytest.mark.parametrize(
    "ref_name",
    ["AnalysisRunRef", "DatasetRef", "ModelRef", "ExportRunRef"],
)
def test_every_listing_ref_carries_duration_and_compute(ref_name):
    """Same class as created_at, for § run location, duration, cost: a ref missing
    `compute`/`duration_s`/`estimated_cost_usd` shows a blank Where/Duration/Cost
    forever for that whole family, no matter what set_*_status merges into its
    record on disk — `_field`'s dataclass branch can only read what the ref
    declares."""
    import dataclasses

    from detail.api import experiment as experiment_mod

    fields = {f.name for f in dataclasses.fields(getattr(experiment_mod, ref_name))}
    missing = {"compute", "duration_s", "estimated_cost_usd"} - fields
    assert not missing, f"{ref_name} carries no {missing}"


@pytest.mark.parametrize("spec", active.RUN_KINDS, ids=lambda s: s.kind)
def test_every_run_kind_reads_a_real_accessor(spec):
    """`records` names an `Experiment` attribute; a typo is a family that lists nothing."""
    assert hasattr(Experiment, spec.records), (
        f"run kind {spec.kind!r} reads Experiment.{spec.records}, which does not exist — "
        f"runs of this kind would silently never be listed"
    )


@pytest.mark.parametrize("spec", active.RUN_KINDS, ids=lambda s: s.kind)
@pytest.mark.parametrize("shape", ["dict", "ref"], ids=["dict-record", "typed-ref"])
def test_a_kind_with_an_active_record_is_reported_as_active(
    workspace, monkeypatch, spec, shape
):
    """Driven over the table, so a seventh kind is covered the day it is added.

    Both record shapes, because the stores disagree: processing and distill hand back plain
    dicts and the rest hand back typed refs. That split is why this used to be three loops in
    three shapes, and a fourth shape is how a family gets added to one of them and not the rest.
    """

    class _Ref:
        id, status = "x_01", "running"

    record = {"id": "x_01", "status": "running"} if shape == "dict" else _Ref()
    experiment = Experiment.create("p", workspace)
    # `monkeypatch`, not a bare setattr + delattr: these are real properties on `Experiment`, so
    # deleting one to "restore" it would leave the class without it for every later test.
    monkeypatch.setattr(
        Experiment, spec.records, property(lambda self: [record]), raising=False
    )
    assert active.active_ids(experiment).get(spec.kind) == ["x_01"]


# --------------------------------------------------------------------------- #
# finished runs stay listed, and archiving is how you dismiss one
# --------------------------------------------------------------------------- #
def _export_record(experiment_dir, run_id, status, created_at, targets=("instances",)):
    """An export record the way `run_export` writes one, minus the job it submits."""
    from detail.storage import export_runs as export_store

    rdir = export_store.export_dir(experiment_dir, run_id)
    export_store.write_export_yaml(
        rdir,
        {
            "id": run_id,
            "subject_kind": "analysis_run",
            "subject_id": "an_01",
            "target_ids": list(targets),
            "folder": f"cohort_{run_id}_export",
            "destination": {"root": "nowhere", "rel": "out"},
            "params": {},
            "status": status,
            "created_at": created_at,
        },
    )


def test_a_finished_export_stays_listed(workspace):
    """A finished export has no other home: its files are in a folder outside the experiment
    and its record is reachable from no page in the app once it is done. Dropped from this
    listing it was reachable from nowhere at all, which is the state the user reported.
    """
    exp = Experiment.create("alpha", workspace)
    _export_record(exp.dir, "ex_01", "complete", "2026-09-10T12:00:00Z")

    found = active.list_active()
    assert [r.run_id for r in found] == ["ex_01"]
    assert found[0].kind == "export"


def test_archiving_a_run_removes_it_from_the_listing(workspace):
    """The one way a finished run stops being listed now that every family keeps one — a pure
    view dismissal, asserted here at the API layer that both the CLI and the GUI read.
    """
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "complete", "2026-09-10T12:00:00Z")

    active.archive_run(exp.dir, "run", "det_01")

    assert active.list_active() == []


def test_archiving_one_run_does_not_hide_another(workspace):
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "complete", "2026-09-10T12:00:00Z")
    _run(exp.dir, "det_02", "complete", "2026-09-10T12:30:00Z")

    active.archive_run(exp.dir, "run", "det_01")

    assert [r.run_id for r in active.list_active()] == ["det_02"]


def test_archiving_a_run_is_idempotent(workspace):
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "complete", "2026-09-10T12:00:00Z")

    active.archive_run(exp.dir, "run", "det_01")
    active.archive_run(exp.dir, "run", "det_01")

    assert active.list_active() == []


def test_archiving_does_not_touch_an_active_run_with_the_same_id_elsewhere(workspace):
    """A pure view flag, never consulted for a status still in `wanted` — archiving is a
    dismiss for a *finished* row, not a way to hide a run still going."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "running", "2026-09-10T12:00:00Z")

    active.archive_run(exp.dir, "run", "det_01")

    assert [r.run_id for r in active.list_active()] == ["det_01"]


def test_archiving_a_failed_run_removes_it_when_failures_are_included(workspace):
    """A looked-at failure can be dismissed like any other finished row."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "failed", "2026-09-10T12:00:00Z")

    active.archive_run(exp.dir, "run", "det_01")

    assert active.list_active() == []


def test_excluding_failures_hides_them_regardless_of_archive_state(workspace):
    """`include_failed=False` means "not this" outright — it must not be overridden by the
    run happening to be unarchived, or the flag would stop meaning what its name says the
    moment every family started keeping finished rows."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "failed", "2026-09-10T12:00:00Z")

    assert active.list_active(include_failed=False) == []


def test_a_cancelled_run_is_listed(workspace):
    """`FAILED_STATUSES` is ``("failed", "cancelled")`` and the filter asked for the literal
    ``"failed"``, so a cancelled run was in neither bucket and appeared **nowhere** — not here,
    and not as active. It is terminal and it is not a success, which is exactly the pair this
    screen exists to show."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "cancelled", "2026-09-10T12:00:00Z")

    assert [r.run_id for r in active.list_active()] == ["det_01"]


def test_a_cancelled_run_is_not_counted_as_running(workspace):
    """And it is not *active* either, which is the half a surface gets wrong when it asks
    ``not is_failed``: cancelled is neither, and a screen counting it as running claims work is
    in flight that stopped."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "cancelled", "2026-09-10T12:00:00Z")

    (run,) = active.list_active()
    assert not run.is_active
    assert not run.is_failed


# --------------------------------------------------------------------------- #
# what an export row says
# --------------------------------------------------------------------------- #
def test_an_export_row_names_its_targets_not_the_word_export(workspace):
    """The fallback six families want and this one cannot use.

    A row's label is ``module_id`` or the family's own name, which is right for every family
    that runs a registered module. An export runs none, so every cell of a column headed
    *exporting* read the literal string "export" — rendering, sorting, and saying nothing.
    """
    exp = Experiment.create("alpha", workspace)
    _export_record(
        exp.dir, "ex_01", "running", "2026-09-10T12:00:00Z", targets=("instances",)
    )

    (run,) = active.list_active()
    assert run.label != "export", "the row still says the family name"
    assert "nstances" in run.label, run.label


def test_an_export_row_carries_where_the_files_went(workspace):
    """The question a *finished* export exists to answer, and the one no run id can.

    Every other output is found from its run id; an export's files are in a folder the user
    picked, outside the experiment. With the alias unregistered — which is what an experiment
    carried to another machine looks like — the folder name is still true, so it is shown
    rather than a blank.
    """
    exp = Experiment.create("alpha", workspace)
    _export_record(exp.dir, "ex_01", "complete", "2026-09-10T12:00:00Z")

    (run,) = active.list_active()
    assert run.detail, "the row says nothing about where the export went"
    assert "cohort_ex_01_export" in run.detail


def test_a_run_of_another_family_carries_no_detail(workspace):
    """The negative case for the column: `detail` is the one extra thing *this* family has to
    say, not a field every row fills with something."""
    exp = Experiment.create("alpha", workspace)
    _run(exp.dir, "det_01", "running", "2026-09-10T12:00:00Z")

    (run,) = active.list_active()
    assert run.detail == ""


def test_a_describer_that_raises_does_not_empty_the_listing(workspace, monkeypatch):
    """A view that vanishes because one of its sources is sick tells the user less than a view
    with one cell missing — `_safe`'s rule for a whole store, applied to the describer that runs
    once per record."""
    exp = Experiment.create("alpha", workspace)
    _export_record(exp.dir, "ex_01", "running", "2026-09-10T12:00:00Z")

    def boom(_record):
        raise RuntimeError("catalogue is on fire")

    monkeypatch.setattr(
        active,
        "RUN_KINDS",
        tuple(
            spec if spec.kind != "export" else replace(spec, describe=boom)
            for spec in active.RUN_KINDS
        ),
    )

    (run,) = active.list_active()
    assert run.run_id == "ex_01"
    assert run.label == "export", "the generic label is the fallback, not a crash"


# --------------------------------------------------------------------------- #
# Imports — the family that is not a run
# --------------------------------------------------------------------------- #
# **The class.** *A long operation that keeps its state outside the shape every surface reads.*
# An import writes no run record: its status lives on the recording as `import_status`, in a
# vocabulary only that family speaks (`queued` / `importing`, where every run says `pending` /
# `running`). So the one screen whose whole job is "is anything going on" said nothing about the
# longest thing a new user does — and nothing could fail over it, because there was no row to be
# missing. `Experiment.import_runs` is the adapter; these are the properties that make it one.


def _importing(experiment, recording_id, status, name=None):
    """A recording mid-import — the three-line idiom that was copied at each call site."""
    recording = experiment.create_recording(recording_id)
    if name is not None:
        recording.set_name(name)
    recording_store.set_import_status(recording.dir, status)
    return recording


@pytest.mark.parametrize(
    "import_status, listed_as",
    [
        ("queued", "pending"),
        ("importing", "running"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
    ],
)
def test_an_unfinished_import_is_listed_in_the_shared_vocabulary(
    workspace, import_status, listed_as
):
    """Translated at the adapter, not at each reader.

    `queued` and `importing` are `pending` and `running` in a dialect one family speaks. Doing
    the translation once is what lets a single listing, a single quit guard and a single
    `is_active` cover a family whose status lives on a recording; the alternative is teaching
    every one of those readers the dialect, which is six places to forget it.
    """
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "rec_01", import_status)

    (row,) = active.list_active()
    assert row.kind == "import"
    assert row.status == listed_as
    assert row.run_id == "rec_01"


def test_a_running_import_reads_as_active_and_a_failed_one_as_failed(workspace):
    """The two predicates every surface counts with. `is_active` is `status in ACTIVE_STATUSES`,
    so an untranslated `importing` would have been counted as *finished* by the page's summary —
    a row on screen, described as the opposite of what it is."""
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "rec_01", "importing")
    _importing(exp, "rec_02", "failed")

    rows = {row.run_id: row for row in active.list_active()}
    assert rows["rec_01"].is_active and not rows["rec_01"].is_failed
    assert rows["rec_02"].is_failed and not rows["rec_02"].is_active


def _with_video(recording, created_at=None):
    """Give a recording the video section an import writes — through the real store."""
    record = recording_store.read_recording(recording.dir)
    record["video"] = {"files": [{"index": 0, "file": "video/part_001.mp4"}]}
    if created_at is not None:
        record["created_at"] = created_at
    recording_store.write_recording(recording.dir, record)
    return recording


def test_a_finished_import_stays_listed_as_complete(workspace):
    """On a local disk an import is over within a second of being asked for.

    The report that prompted this: two imports, one of them 533 MB, both finished inside the
    second they started, and the Imports section said "nothing importing" to the person who had
    just done it. A section that drops finished rows can never be looked at in time.
    """
    exp = Experiment.create("alpha", workspace)
    _with_video(_importing(exp, "rec_01", "ready"))

    (row,) = active.list_active()
    assert row.kind == "import"
    assert row.status == "complete"
    assert not row.is_active and not row.is_failed


def test_a_recording_that_brought_nothing_in_is_not_a_finished_import(workspace):
    """A shell created with metadata only is `ready` the moment it exists and carried nothing in.

    The real finished import is in the fixture so this cannot pass by listing nothing at all.
    """
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "rec_metadata_only", "ready")
    _with_video(_importing(exp, "rec_imported", "ready"))

    assert [row.run_id for row in active.list_active()] == ["rec_imported"]


def test_a_recording_no_import_ever_touched_is_not_listed(workspace):
    """`Recording.import_status` defaults a missing key to "ready", which would make every
    recording made before the field existed — and every one given video by `recording add-video`,
    which never writes it — read as a finished import. The adapter reads the key instead.
    """
    exp = Experiment.create("alpha", workspace)
    _with_video(exp.create_recording("rec_added_by_cli"))
    _with_video(_importing(exp, "rec_imported", "ready"))

    assert [row.run_id for row in active.list_active()] == ["rec_imported"]


def test_finished_imports_all_stay_listed_newest_first(workspace):
    """Unbounded now, like every other family — newest first, across the workspace."""
    exp = Experiment.create("alpha", workspace)
    for day in range(1, 13):
        _with_video(
            _importing(exp, f"rec_{day:02d}", "ready"),
            created_at=f"2026-09-{day:02d}T12:00:00Z",
        )

    assert [row.run_id for row in active.list_active()] == [
        f"rec_{day:02d}" for day in range(12, 0, -1)
    ]


def test_the_quit_guard_does_not_count_a_finished_import(workspace):
    """Listed is not running. A finished import in the quit prompt would ask the user to cancel
    something that is already done."""
    exp = Experiment.create("alpha", workspace)
    _with_video(_importing(exp, "rec_done", "ready"))
    _importing(exp, "rec_going", "importing")

    assert active.active_ids(exp).get("import") == ["rec_going"]


def test_an_import_row_names_the_recording_the_user_named(workspace):
    """§11: the id is the tool's, the name is the user's.

    Every other family's id *is* what the user calls it — `run_004`, `det_01` — so the generic
    row prints the id and is right. This listing is read at a glance across every experiment,
    which is the worst place to ask somebody to recognise `recording_007`. The id still travels,
    because it is what the opener needs and what the CLI speaks.
    """
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "recording_007", "importing", name="Habituation day 1")

    (row,) = active.list_active()
    assert row.label == "Habituation day 1"
    assert row.run_id == "recording_007"


def test_the_quit_guard_counts_an_unfinished_import(workspace):
    """`active_ids` is the question `closeEvent` asks. An import it does not count is one the
    window closes on without a word — and unlike every run, an import does not survive that.
    """
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "rec_01", "importing")
    _importing(exp, "rec_02", "ready")

    assert active.active_ids(exp).get("import") == ["rec_01"]


def test_the_import_family_is_the_one_that_does_not_survive_quitting(workspace):
    """The fact the quit prompt is built on, asserted against the table rather than a string
    compare in the dialog. A run's workers are detached processes and outlive the app, which is
    why the prompt offers to leave them going; an import is a thread inside it."""
    survives = {spec.kind: spec.survives_quit for spec in active.RUN_KINDS}

    assert survives["import"] is False
    assert all(
        value for kind, value in survives.items() if kind != "import"
    ), "every other family is a detached worker and does outlive the app"


def test_cancel_import_records_the_decision(workspace):
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "rec_01", "queued")

    exp.cancel_import("rec_01")

    assert exp.get_recording("rec_01").import_status == "cancelled"


def test_cancel_import_is_a_no_op_once_the_import_is_terminal(workspace):
    """Quitting sweeps every kind's active ids; a recording whose import already finished must
    not be re-labelled by that sweep — the same contract `cancel_export` keeps."""
    exp = Experiment.create("alpha", workspace)
    _importing(exp, "rec_01", "ready")

    exp.cancel_import("rec_01")

    assert exp.get_recording("rec_01").import_status == "ready"
