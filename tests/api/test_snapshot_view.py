"""Reading a snapshot back as the interpretation it was (§21).

**The class.** A read-only view of a mutable thing that shares the mutable thing's write paths.
Every read on an authored run resolves through `state()` and `artifact()`, so pointing those two at
a frozen copy renders the whole interpretation as it was — but the view still addresses the *run's*
directory for its lock, its status and its fit. A write that slipped through would therefore land on
the live run, not on the copy being looked at: the damage would not even be local to the thing on
screen. `finalize()` is the sharpest instance, because it does not go through `_edit` and would
finish a run somebody had only opened to read.

**How it hides.** Nothing fails at the line where it is written. A view built from the right
snapshot shows the right numbers whether or not its writers refuse; the defect only appears the day
somebody presses a button on it. So the guard is not "does it read correctly" but "does every
declared writer refuse", driven off `AuthoredRun._WRITE_METHODS` — plus a scan that fails when a
sixth writer joins the class without joining the list, since a writer nobody declared is a writer
nobody guarded.

**The rule.** A read-only handle refuses *every* write, and the list of writes is declared in the
class rather than remembered by whoever adds the next one.
"""

import ast
import inspect
import pathlib

import pytest

from detail.api import authored_run as authored
from detail.api.authored_run import AuthoredRun, ReadOnlyRunError
from detail.api.experiment import Experiment
from detail.api.query import QueryRun
from detail.storage import behavior_output as bo


class QueryRunStub(QueryRun):
    """A `QueryRun` with no run directory: `snapshot_interpretation` reads the record it is
    handed and nothing else, which is the point of asking the module rather than the store.
    """

    def __init__(self):  # noqa: D107 - deliberately skips the handle's setup
        pass


# --------------------------------------------------------------------------- #
# what a view reads
# --------------------------------------------------------------------------- #
def test_the_view_reports_the_interpretation_as_pinned(authored_cohort):
    """The point of the whole feature: what the snapshot was, not what the run is now."""
    cohort = authored_cohort
    cohort.name({"c01": "rearing"})
    snapshot_id = cohort.run().pin(reason="finish")

    cohort.name({"c01": "grooming", "c02": "still"})
    view = cohort.run().view(snapshot_id)

    assert view.state().names == {"c01": "rearing"}
    assert view.state().granularity == 4
    assert cohort.run().state().names == {"c01": "grooming", "c02": "still"}


def test_the_views_instances_come_from_the_frozen_copy(authored_cohort):
    """Read, never re-derived. Re-deriving under later code is what makes an old result
    unreproducible — and it would hide a snapshot whose state no longer reproduces its own
    instances, which is worth being able to notice."""
    cohort = authored_cohort
    snapshot_id = cohort.run().pin(reason="export")
    view = cohort.run().view(snapshot_id)
    assert len(view.instances()) == 6

    artifact = bo.read_behavior_output(cohort.run_dir())
    bo.write_behavior_output(
        cohort.run_dir(),
        instances=list(bo.bouts(artifact.instances))[:1],
        coverage=list(artifact.coverage),
        manifest=artifact.manifest,
    )
    assert len(cohort.run().instances()) == 1
    assert len(cohort.run().view(snapshot_id).instances()) == 6


def test_the_view_carries_where_it_came_from(authored_cohort):
    cohort = authored_cohort
    snapshot_id = cohort.run().pin(reason="finish")
    view = cohort.run().view(snapshot_id)
    assert view.snapshot_id == snapshot_id
    assert view.snapshot_record()["reason"] == "finish"
    assert view.id == "an_01", "it is still that run, read at a moment"


# --------------------------------------------------------------------------- #
# what a view refuses
# --------------------------------------------------------------------------- #
#: How to invoke each declared writer, and the operation name its own guard should report.
#:
#: ``_edit`` and ``_record`` are the two funnels: they are *given* the operation, so their refusal
#: names the caller's word rather than the method's. Everything else reports its own name.
_CALLS = {
    "_edit": (lambda v: v._edit("rename", lambda state: None), "rename"),
    "_record": (lambda v: v._record("audit_sample", {}), "audit_sample"),
    "pin": (lambda v: v.pin(reason="finish"), "pin"),
    "label_audit": (lambda v: v.label_audit("some-instance", True), "label_audit"),
    "audit_sample": (lambda v: v.audit_sample(), "audit_sample"),
    # The mixin's own override is this one's guard, so its refusal names the operation rather
    # than the method — the same shape as the two funnels above.
    "_write_state": (lambda v: v._write_state(None), "write its state"),
    "_set_stale": (lambda v: v._set_stale(True), "mark its artifact stale"),
    "rebuild_node_cache": (
        lambda v: v.rebuild_node_cache(),
        "rebuild node cache",
    ),
}


def _call(view, method: str) -> str:
    """Invoke one declared writer; return the operation its refusal should name."""
    invoke, op = _CALLS.get(method, (lambda v: getattr(v, method)(), method))
    invoke(view)
    return op


@pytest.mark.parametrize("method", AuthoredRun._WRITE_METHODS)
def test_every_declared_writer_refuses_on_a_view(authored_cohort, method):
    """Data-driven over the declared list, so a writer that joins the class is covered the day it
    is added rather than the day somebody remembers this file.

    **The message must name the operation being refused**, which is what makes each case prove its
    own guard. Without that this passed for `finalize` with `finalize`'s guard deleted, because
    `finalize` calls `pin` and *pin* refused — a test that is green for a reason unrelated to the
    thing it names. Removing any one guard now fails its own case.
    """
    cohort = authored_cohort
    view = cohort.run().view(cohort.run().pin(reason="finish"))
    with pytest.raises(ReadOnlyRunError) as excinfo:
        _call(view, method)
    message = str(excinfo.value)
    assert "snap" in message, "it says which snapshot it is refusing for"
    op = _CALLS.get(method, (None, method))[1]
    assert f"will not {op}" in message, (
        f"{method} was refused by some *other* method's guard ({message!r}), so this case does not "
        f"cover {method}'s own."
    )


def test_a_refused_write_leaves_the_live_run_alone(authored_cohort):
    """The reason the refusal matters. A view addresses the *run's* directory, so an unguarded
    `finalize()` would finish the run somebody had only opened to look at."""
    cohort = authored_cohort
    view = cohort.run().view(cohort.run().pin(reason="export"))
    with pytest.raises(ReadOnlyRunError):
        view.finalize()
    assert cohort.run().status == "open"


def test_naming_through_a_view_refuses(authored_cohort):
    """The user-facing path, not just the funnel underneath it."""
    cohort = authored_cohort
    view = cohort.run().view(cohort.run().pin(reason="finish"))
    with pytest.raises(ReadOnlyRunError):
        view.name_nodes({0: "wall-lean"})
    assert cohort.run().state().node_names == {}


# --------------------------------------------------------------------------- #
# the guard on the guard: a writer nobody declared is a writer nobody refuses
# --------------------------------------------------------------------------- #
# Every call that puts something on disk under the run directory.
#
# The list is the scan's whole reach, and it has been short twice. `append_audit` and `_record`
# joined when the §19 audit landed. `_log` — `AuthoredRun`'s own documented write hook — was missing
# until a review added a method whose entire body was `self._log({...})` and watched this file stay
# green. The concrete state writers are here because a subclass could call one directly instead of
# going through `_write_state`, which is one line away from how `_write_state` is itself written.
_WRITE_CALLS = {
    "_write_state",
    "set_analysis_status",
    "set_artifact_stale",
    "_project",
    "append_audit",
    "_record",
    "_log",
    "write_behavior_output",
    "write_frame_view",
    "write_instance_features",
    "write_analysis_yaml",
    "write_temporal_state",
    "write_query_state",
    # An edit defers the frame views and a reader brings them current (#325): both write the
    # run's `frames/`, the first by marking it pending.
    "_project_frames",
    "set_pending",
    # The clip review's log (`storage/clip_review.py`), written by the temporal run's mixin.
    "append_answers",
}

#: Every class a snapshot view is an instance of.
#:
#: The scan read only `AuthoredRun` at first, which made its own docstring's promise false: a writer
#: added to `QueryRun` — a class every Query view inherits — was invisible to it forever.
_SCANNED = (
    ("detail/api/authored_run.py", "AuthoredRun"),
    ("detail/api/temporal_clustering.py", "TemporalClusteringRun"),
    ("detail/api/query.py", "QueryRun"),
    ("detail/api/clip_edits.py", "ClipEditOperations"),
    ("detail/api/clip_review.py", "ClipReviewOperations"),
)

#: Writers a view inherits but does not need to refuse, with the reason.
_NOT_A_RUN_WRITE = {
    # Reads the fit and writes nothing; the name matches because it *takes* a run dir.
    "_load_fit",
}


def _writing_methods() -> set[str]:
    """Methods of every class a view is, whose body writes under the run directory.

    A source scan rather than a runtime check, because the defect is a method that *exists* and was
    never added to the list — there is nothing to observe at runtime until someone calls it.
    """
    root = pathlib.Path(inspect.getsourcefile(authored)).parents[2]
    found = set()
    for relative, class_name in _SCANNED:
        tree = ast.parse((root / relative).read_text(encoding="utf-8-sig"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name in _NOT_A_RUN_WRITE:
                continue
            for node in ast.walk(method):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", None)
                    if name in _WRITE_CALLS or (
                        name == "pin" and getattr(node.func.value, "id", "") != "self"
                    ):
                        found.add(method.name)
    return found


def test_every_method_that_writes_is_declared(authored_cohort):
    from detail.api.temporal_clustering import TemporalClusteringRun

    declared = set(AuthoredRun._WRITE_METHODS) | set(
        TemporalClusteringRun._WRITE_METHODS
    )
    undeclared = _writing_methods() - declared
    assert not undeclared, (
        f"{sorted(undeclared)} write to the run directory but are not in _WRITE_METHODS, so a "
        "snapshot view would not refuse them. Add them to the tuple and call "
        "_refuse_if_read_only()."
    )


def test_the_scan_would_catch_an_undeclared_writer():
    """Guard the guard: a scan that matches nothing passes vacuously forever.

    Asserting the scan is non-empty and contains `_edit` — a *declared* method — proves it matches
    something, not that it would flag an undeclared one. So this synthesizes the real case: a
    method whose whole body is `self._log(...)`, which is exactly the shape that slipped through
    when `_log` was missing from `_WRITE_CALLS`.
    """
    assert _writing_methods(), "the scan found no writers at all, so it proves nothing"

    source = """
class AuthoredRun:
    def annotate(self, note):
        self._log({"op": "annotate", "note": note})
"""
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    hits = set()
    for method in cls.body:
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) in _WRITE_CALLS
            ):
                hits.add(method.name)
    assert hits == {"annotate"}, "the scan would not flag an undeclared writer"


# --------------------------------------------------------------------------- #
# the edges
# --------------------------------------------------------------------------- #
def test_a_snapshot_with_no_recorded_state_says_so(authored_cohort):
    """Snapshots pinned before a comparison pinned through the run handle carry `state: {}` — real
    runs on disk have them. The state dataclass would raise a bare KeyError; this says what is
    actually wrong and what is still readable."""
    from detail.analysis import provenance
    from detail.storage import snapshots as snapshots_store

    cohort = authored_cohort
    snapshot_id = snapshots_store.pin(
        cohort.run_dir(), reason="comparison", code_sha=provenance.code_version()
    )
    with pytest.raises(ValueError) as excinfo:
        cohort.run().view(snapshot_id)
    message = str(excinfo.value)
    assert "recorded no settings" in message
    assert "snapshot_artifact" in message, "it names what still works"


def test_a_missing_snapshot_raises_rather_than_rendering_the_live_run(authored_cohort):
    with pytest.raises(FileNotFoundError):
        authored_cohort.run().view("snap_99")


def test_every_authored_module_can_read_its_snapshots_back(authored_cohort):
    """Data-driven over the accessor table, so a fourth authored module has to decide.

    Query was left out at first, on the grounds that a strip of clips is not a vocabulary. That
    missed the part that matters: a Query snapshot's interpretation is its behaviors and their
    **cutoffs**, and "which cutoff did I ship?" is exactly the question a pinned result exists to
    answer. `None` remains a legitimate answer for a module that genuinely authors nothing, but it
    has to be a decision — the failure mode is a module quietly inheriting the default.
    """
    from detail.api.query import QueryRun
    from detail.api.temporal_clustering import TemporalClusteringRun

    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    declared = {
        "temporal_clustering": TemporalClusteringRun,
        "query": QueryRun,
    }
    # Driven off the registry, not off a literal typed here. A hand-listed set catches "you added an
    # accessor without a view class" and cannot catch the direction that actually fails — a module
    # marked `mutable` that never got an accessor, which is how the removed `clip_annotation` sat
    # outside this for its whole life.
    authored = {
        m for m, mod in ANALYSIS_MODULES.items() if getattr(mod, "mutable", False)
    }
    # No exemption today. Clip annotation was the one — authored, but a decision log that pinned
    # nothing — and it has been removed. A new one is a decision with a reason, written here.
    exempt: set[str] = set()
    assert authored - exempt == set(declared), (
        f"authored modules {sorted(authored - exempt - set(declared))} have no snapshot accessor, "
        f"and {sorted(set(declared) - authored)} are declared but not authored. Add the accessor "
        "and a snapshot_view_class, or exempt it here with a reason."
    )
    assert set(Experiment._AUTHORED_RUN_ACCESSORS) == set(declared)
    for module_id, cls in declared.items():
        assert cls.snapshot_view_class is not None, module_id
        assert cls.snapshot_view_class._state_class is not None, module_id


def test_each_module_summarizes_a_snapshot_in_its_own_terms(authored_cohort):
    """The bug this replaced a hardcoded summary to fix.

    Every surface used to read `granularity` and `names` straight out of the record — clustering's
    shape and nobody else's — so a fully calibrated scan with two named behaviors rendered
    `Clusters — · Named 0 · unnamed`. Wrong information rather than missing information, and
    invisible, because a blank cell is a plausible cell.
    """
    import dataclasses

    from detail.storage.query_runs import Behavior, QueryState

    clustering = authored_cohort.run()
    authored_cohort.name({"c01": "rearing"})
    snapshot_id = clustering.pin(reason="finish")
    summary = clustering.snapshot_interpretation(clustering.snapshot(snapshot_id))
    # §22: a clustering snapshot is the labels it carried. This cohort's state was written
    # with a granularity, which the headline still states because the snapshot froze it.
    assert summary.columns == ("Labels", "Vocabulary")
    assert summary.cells == ("1", "rearing")
    assert "taken at 4 clusters" in summary.headline

    # The scan's own shape, against the state that used to render as blank.
    state = QueryState(
        embed_run_id="emb_01",
        track_id=0,
        method="soft_knn",
        behaviors=[
            Behavior("b1", "supported rearing", cutoff=0.72),
            Behavior("b2", "grooming", cutoff=None),
        ],
    )
    record = {"snapshot_id": "snap_01", "state": dataclasses.asdict(state)}
    scan = QueryRunStub().snapshot_interpretation(record)
    assert scan.columns == ("Behaviors", "Calibrated", "Cutoffs", "Vocabulary")
    assert scan.cells == ("2", "1", "0.72", "supported rearing, grooming")
    assert "2 behaviors · 1 calibrated" in scan.headline


# --------------------------------------------------------------------------- #
# the freeze itself: pinned != live, for every authored module
# --------------------------------------------------------------------------- #
def _diverged_query(tmp_path):
    """A scan whose pinned calibration differs from its live one."""
    from tests.api.test_query import _searched

    exp, run = _searched(tmp_path)
    behavior = run.behaviors()[0]
    run.set_cutoff(behavior.id, 0.30)
    snapshot_id = run.pin(reason="finish")
    run.set_cutoff(behavior.id, 0.90)
    return exp.query_run(run.id), snapshot_id, 0.30, 0.90


def test_a_scan_view_reads_the_pinned_cutoff_not_the_live_one(tmp_path):
    """The bug this file could not see.

    `QueryRun.state()` read the run directory directly instead of going through `_read_state`, so
    the mixin's override was bypassed and every read followed: the cutoff line, the confidence box,
    the frequency curve and the behavior names all showed today's values under a banner saying
    "Read-only", with the frozen instance list beside them making it look consistent.

    The reason it shipped is that no fixture made the snapshot differ from the run, so every
    assertion compared a value to itself.
    """
    run, snapshot_id, pinned, live = _diverged_query(tmp_path)
    view = run.view(snapshot_id)

    assert view.state().behaviors[0].cutoff == pinned
    assert run.state().behaviors[0].cutoff == live
    assert view.behaviors()[0].cutoff == pinned, "every read follows state()"


@pytest.mark.parametrize("module_id", sorted(Experiment._AUTHORED_RUN_ACCESSORS))
def test_every_modules_view_diverges_from_its_live_run(
    module_id, authored_cohort, tmp_path
):
    """Data-driven, so a fourth authored module inherits the one property the feature *is*.

    Asserting `view.state() == view._read_state()` alongside it pins the delegation directly: the
    Query bug was exactly a `state()` that stopped consulting `_read_state`.
    """
    if module_id == "query":
        run, snapshot_id, pinned, live = _diverged_query(tmp_path)
        view = run.view(snapshot_id)
        assert view.state().behaviors[0].cutoff != run.state().behaviors[0].cutoff
    elif module_id == "temporal_clustering":
        cohort = authored_cohort
        assert cohort.module_id == module_id
        snapshot_id, pinned, live = cohort.pin_then_change()
        view = cohort.run().view(snapshot_id)
        assert (view.state().granularity, dict(view.state().names)) == pinned
        assert (
            cohort.run().state().granularity,
            dict(cohort.run().state().names),
        ) == live
    else:
        pytest.skip(
            f"{module_id} has no cheap fixture here; covered by its own GUI tests"
        )
    assert view.state() == view._read_state(), "state() must go through the seam"


def test_a_views_audit_decisions_are_the_snapshots_own(authored_cohort):
    """A snapshot copies `audit.csv`, and the precision panel has to read *that* one.

    Reading by run directory mixed a frozen numerator with live decisions — a snapshot whose own
    file said 3 judgments reported 6, a number that was never true of anything. The write path is
    deliberately untouched: it still addresses the run and is still refused.
    """
    cohort = authored_cohort
    instances = [i.instance_id for i in cohort.run().instances()]
    cohort.judge(instances[:3], verdict=True)
    snapshot_id = cohort.run().pin(reason="finish")
    cohort.judge(instances[3:6], verdict=False)

    view = cohort.run().view(snapshot_id)
    from detail.storage import behavior_audit as audit

    frozen = [r for r in audit.latest_decisions(view._audit_dir()) if r.decided]
    current = [
        r for r in audit.latest_decisions(cohort.run()._audit_dir()) if r.decided
    ]

    assert len(frozen) == 3, "the snapshot's own decisions"
    assert len(current) == 6, "the live run kept accumulating"
    assert view._audit_dir() != view._run_dir()
