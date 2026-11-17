"""What is running, across every experiment — one question, asked once.

**Why this is not a page.** Six kinds of run each keep their records in their own store, and five
detail pages each answered "is this one still going?" for the one run they were showing. Nothing
answered it for the workspace, so the only way to find out whether last night's detection
finished was to remember which experiment it was in and open it.

The information was always there — every kind has a `discover_*` and a status — so this is a
listing over what exists rather than new bookkeeping. It lives in the API rather than the GUI for
the usual reason: the CLI wants the same answer, and a run does not stop being active because
nobody has a window open.

**It routes; it does not report.** A row carries enough to recognise and rank a run — what, where,
how far, since when — and then hands over to that run's own page, which is where the bar, the
activity log and the per-recording detail already live. Rendering progress here as well would be
the second event→render mapping DESIGN_PLAN.md:1351 warns about.

**On cost.** This walks every experiment's run directories, which is the shape this repo has a
recorded class about: an O(N) sweep of individually-cheap reads is nothing on a local disk and
seconds over SMB. So it reads one record per run and nothing else — no zarr, no outputs.

The **import** family is the one exception, and it is a deliberate one: an import keeps its
status on the recording, so listing imports is one `recording.yaml` per recording rather than
one record per run. `Experiment.import_runs` pays that under `read_cache.pinned()`, which is
what holds it to a single round trip each instead of one per field —
`tests/gui/test_active_processes.py` asserts the slope rather than a total, because a wall
clock on a local disk measures nothing here. It matters more than usual because `active_ids`
is read **on the UI thread** by the quit guard; `list_active` is expected to run off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from detail.core.dataclasses import ACTIVE_STATUSES, FAILED_STATUSES


@dataclass(frozen=True)
class ActiveRun:
    """One run that has not reached a terminal status, and where to find it.

    ``kind`` is the export/navigation kind the rest of the app already routes on, so a surface can
    open this run without a second mapping from "what sort of thing is this" to "which page".
    """

    kind: str
    experiment_dir: Path
    experiment_name: str
    run_id: str
    label: str
    status: str
    created_at: str
    #: The one extra thing this family's row has to say, or "" for a family with nothing.
    #: An export's is **where its files went** — a question no other run kind has, and the whole
    #: reason to come back to a *finished* one. A `detail` rather than a second `label` because
    #: it is its own column and its own sort.
    detail: str = ""

    @property
    def is_failed(self) -> bool:
        """Ended in failure — ``failed`` only. A **cancelled** run is not failed (the user
        stopped it) and not active; a surface that needs a bucket for every status reads
        `gui/active_processes.py::SUMMARY_BUCKET` rather than inferring one from these two.
        """
        return self.status == "failed"

    @property
    def is_active(self) -> bool:
        """Still going. The complement is *terminal*, which includes complete and cancelled —
        neither of which `is_failed` covers, so asking "not failed" counts a finished run as
        running."""
        return self.status in ACTIVE_STATUSES


@dataclass(frozen=True)
class RunKind:
    """One family of run, and everything a surface needs to know about it.

    **Why this is a table.** Six families each answer the same three questions — what is
    running, what page shows it, how do I stop it — and each question was its own hand-written
    list. They drifted, in both directions and silently: the Active-runs double-click router
    (`gui/app.py`) knew 2 of the 6, so four kinds of row drew the navigable cue and did nothing
    when clicked; the quit guard knew 4, so closing the window during a **distill** run — the
    longest thing this app does — neither warned nor offered to cancel. Nothing failed in either
    case, because a missing row and a row nobody clicks look identical.

    So the families are declared once, here, and the surfaces read them. A surface that
    genuinely cannot be derived from this table — the GUI's opener maps a kind to a *method* —
    is keyed by `kind` and checked against it by a parity guard instead.

    `records` and `cancel` are ``Experiment`` **attribute names** rather than bound callables,
    because this table is built at import time and an experiment is not — and because a name is
    something a guard can check exists, where a lambda is not.
    """

    kind: str
    label: str
    records: str
    cancel: str
    #: How many **finished** runs of this family the listing keeps, newest first. Zero for every
    #: family whose finished runs already have a permanent home — a completed detection run is
    #: on its experiment's Runs page, and listing it here as well would bury the one still going.
    #: An export has no such home: it writes into a folder outside the experiment and its record
    #: is reachable from nothing once it is done. So it keeps its recent history here, which is
    #: a fact about where a family *lives*, not a display preference.
    #:
    #: An **import** keeps its history for a different reason, and it is not that one: a finished
    #: import does have a home, its experiment's Recordings page. It is that an import on a local
    #: disk is over within a second of being asked for — both imports in the report that
    #: prompted this finished inside the second they started, one of them 533 MB — so a
    #: user who goes to look is always too late to see it running, and a section that drops
    #: finished rows reads to them as "nothing happened".
    keep_finished: int = 0
    #: Whether this family's work outlives the app that started it. Every *run* does: its
    #: workers are detached processes, which is the whole reason the quit prompt offers to
    #: leave them running. An **import** does not — it is a thread in this process, driven by
    #: an in-memory queue that cannot resume (§3), so quitting ends it. The quit prompt is the
    #: one surface that has to know the difference, because it is the surface that tells the
    #: user what is about to happen; a family that gets this wrong does not fail, it lies.
    survives_quit: bool = True
    #: ``(record) -> (what it is, the extra column)``. Defaults to reading ``module_id``, which
    #: is right for every family that runs a registered module and wrong for the one that does
    #: not: an export's record names no module, so the fallback printed the literal string
    #: "export" in every cell of a column headed *exporting*. The family answers; the listing
    #: renders.
    describe: Callable[[Any], tuple[str, str]] | None = None


def _field(record, name: str, default=""):
    """One accessor for both record shapes.

    Processing and distill stores hand back plain dicts; datasets, models and analyses hand
    back typed refs. That difference is why `_runs_of` used to be three loops in
    three shapes, which is what made a fourth easy to forget.
    """
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


#: How many finished exports the listing keeps. Bounded rather than unbounded because this is a
#: *workspace-wide* listing: an experiment with three hundred exports would push the two that are
#: running off the screen, which is the recorded class of a widget that is correct at the size it
#: was written at and useless at the size the tool is for.
_KEEP_FINISHED_EXPORTS = 10

#: How many finished imports the listing keeps. The same bound for the same reason — one screen,
#: capped across the workspace — and a separate number because the reason a finished import is
#: listed at all is different (see `RunKind.keep_finished`), so the two can come apart.
_KEEP_FINISHED_IMPORTS = 10


def _describe_export(record) -> tuple[str, str]:
    """An export row: the target *labels*, and the folder the files went to.

    A local import because `api.exporters` loads the whole target catalogue, and this module is
    imported by the quit guard — which asks what is running and has no business paying for a
    registry it will not read.
    """
    from detail.api import exporters

    # Through `_field`, because the stores disagree about shape: processing hands back dicts and
    # the export store hands back a typed ref. Reading one shape is how a describer returns two
    # blank strings against a perfectly good record and nothing fails.
    return exporters.describe_run(
        {
            key: _field(record, key)
            for key in ("subject_kind", "target_ids", "folder", "destination")
        }
    )


def _describe_import(record) -> tuple[str, str]:
    """An import row: the recording being imported, by **name**.

    Every other family's id is what the user calls it — `run_004`, `det_01` — so the generic
    row prints the id and is right. A recording's id is the tool's and its name is the user's
    (§11), and this listing is read at a glance across experiments, which is the worst place to
    ask someone to recognise `recording_007`. The id still has its own column; this decides
    which of the two is the thing you read first.
    """
    return str(_field(record, "name") or _field(record, "id")), ""


#: Every family of run this workspace can have going, in the order a listing shows them.
#:
#: `kind` is the navigation kind the rest of the app already routes on, so a row carries what a
#: surface needs to open it without a second mapping. `label` is what a row says when the record
#: names no module of its own.
RUN_KINDS: tuple[RunKind, ...] = (
    RunKind("run", "run", "processing_runs", "cancel_run"),
    RunKind("analysis_run", "analysis", "analysis_runs", "cancel_analysis"),
    RunKind("distill", "distill", "distill_runs", "cancel_distill"),
    RunKind("dataset", "dataset", "datasets", "cancel_dataset"),
    RunKind("model", "model", "models", "cancel_model"),
    RunKind(
        "export",
        "export",
        "export_runs",
        "cancel_export",
        keep_finished=_KEEP_FINISHED_EXPORTS,
        describe=_describe_export,
    ),
    # The one family that is not a run. Its status lives on the recording rather than in a run
    # record, so `Experiment.import_runs` is the adapter that translates it into this table's
    # vocabulary — see that property for why the translation is there and not here. It belongs
    # in this table for the reason the table exists: an import is the longest thing a new user
    # does, and until it was here, the screen that answers "is anything going on" said no.
    RunKind(
        "import",
        "import",
        "import_runs",
        "cancel_import",
        keep_finished=_KEEP_FINISHED_IMPORTS,
        survives_quit=False,
        describe=_describe_import,
    ),
)


def _safe(fn, default):
    """A store that cannot be read contributes nothing rather than hiding every other run.

    One unreadable experiment — a half-copied folder, a permission on a share — must not empty
    the list. This is a *view*, and a view that vanishes because one of its sources is sick tells
    the user less than a view with one row missing.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def active_ids(experiment) -> dict[str, list[str]]:
    """``{kind: [run id, ...]}`` for one experiment's runs that are still going.

    The quit guard's question, asked of the same table the listing uses — it had its own
    four-family copy, and `clusterer` and `distill` were not in it.
    """
    out: dict[str, list[str]] = {}
    for spec in RUN_KINDS:
        ids = [
            str(_field(record, "id"))
            for record in _safe(lambda s=spec: getattr(experiment, s.records), [])
            if _field(record, "status") in ACTIVE_STATUSES
        ]
        if ids:
            out[spec.kind] = ids
    return out


def _runs_of(experiment) -> list[ActiveRun]:
    name = _safe(lambda: experiment.name, "") or Path(experiment.dir).name

    def make(kind, run_id, label, detail, status, created_at):
        return ActiveRun(
            kind=kind,
            experiment_dir=Path(experiment.dir),
            experiment_name=name,
            run_id=str(run_id or ""),
            label=label,
            status=str(status or ""),
            created_at=str(created_at or ""),
            detail=detail,
        )

    found: list[ActiveRun] = []
    for spec in RUN_KINDS:
        for record in _safe(lambda s=spec: getattr(experiment, s.records), []):
            # A family that describes itself is asked; one that does not falls back to the
            # module it ran. A describer that raises must not empty the listing — the row is
            # still worth showing with the generic label, which is what it had before.
            label, detail = _field(record, "module_id") or spec.label, ""
            if spec.describe is not None:
                label, detail = _safe(
                    lambda r=record, d=spec.describe: d(r), (spec.label, "")
                )
            found.append(
                make(
                    spec.kind,
                    _field(record, "id"),
                    label,
                    detail,
                    _field(record, "status"),
                    _field(record, "created_at"),
                )
            )
    return found


def list_active(*, include_failed: bool = True) -> list[ActiveRun]:
    """Every run that is pending or running, newest first, across every experiment.

    ``include_failed`` also returns runs that ended badly — ``failed`` or ``cancelled`` — and
    **keeps returning them until the run is deleted.** Nothing records that a row has been seen:
    there is no seen or dismissed field on any run record, and this listing reads records and
    writes none. So a failed or cancelled run stays listed, in every experiment, until someone
    deletes or retries it; that is what `docs/gui.md` (*Active processes*) tells the user. They
    are not *active*, and they are the other thing someone opening this is looking for: a run
    that stopped overnight is exactly as urgent as one still going, and finding it should not
    require remembering which experiment it was in. `FAILED_STATUSES`, not the literal
    ``"failed"``: a **cancelled** run is terminal and not failed, so asking for one string
    listed it nowhere.

    A family that declares `keep_finished` also contributes that many of its most recent
    *finished* runs. Exports and imports do, for two different reasons — see
    `RunKind.keep_finished`.
    """
    from detail.api.experiment import Experiment

    wanted = set(ACTIVE_STATUSES) | (set(FAILED_STATUSES) if include_failed else set())
    keep = {spec.kind: spec.keep_finished for spec in RUN_KINDS if spec.keep_finished}
    out: list[ActiveRun] = []
    spare: dict[str, list[ActiveRun]] = {}
    for experiment in _safe(Experiment.list_all, []):
        for run in _runs_of(experiment):
            if run.status in wanted:
                out.append(run)
            elif run.kind in keep:
                spare.setdefault(run.kind, []).append(run)
    # Newest first: the run someone is asking about is almost always the last one they started.
    # `created_at` is an ISO stamp, so a string sort is a time sort — and a run missing one sorts
    # last rather than crashing the listing.
    newest = lambda r: r.created_at or ""  # noqa: E731
    for kind, finished in spare.items():
        # Capped **across the workspace**, not per experiment: the listing is one screen, so a
        # per-experiment cap would be a workspace-wide cap of N x however many experiments there
        # happen to be — a bound that grows with the thing it is bounding.
        out.extend(sorted(finished, key=newest, reverse=True)[: keep[kind]])
    return sorted(out, key=newest, reverse=True)
