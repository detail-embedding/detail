"""Every export target names the data behind its artifact — the rule most likely to decay.

**The class.** *An export that ships a picture instead of the thing the picture was made from.*
The line is: exporting a video file is the easy part; what matters is exporting the data
needed to recreate that video. Without it, "any analysis you show should be exportable" degrades into a Save
PNG button, one target at a time, and nothing fails: the export runs, the file is there, and it
is merely useless to the person who wanted to build on it.

**How it would hide.** A target is added for the surface in front of you, it writes the artifact
that surface draws, and that is the whole feature working. The missing half is invisible until
somebody downstream tries to rebuild the figure and finds they have a movie of it. So the check
cannot be "did the export write something" — it has to be a declaration, made at registration,
that a reviewer can read.

**The rule.** Every `ExportTarget` declares **exactly one** of:

- `data_files` — the **filenames** it writes beside the artifact that carry the rows,
  coordinates or masks needed to redraw it elsewhere; or
- `artifact_is_data` — the artifact is itself the data (a CSV export is its own input).

Filenames rather than a sentence about them, for two reasons: it is what a surface can show
somebody deciding whether to click, and it is checkable against what the export actually wrote.

Data-driven over the catalogue, so a target added tomorrow is covered the day it is added rather
than the day someone remembers this file.
"""

import pytest

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters

pytestmark = pytest.mark.guard

#: Subject kinds the catalogue must be able to export *something* from. These are the
#: navigation kinds the GUI routes on, so a kind with no targets is a page whose Export
#: control can never appear.
_SUBJECT_KINDS = ("experiment", "recording", "run", "analysis_run", "distill")


def test_every_subject_kind_has_at_least_one_target():
    have = {t.subject_kind for t in exporters.all_targets()}
    missing = sorted(set(_SUBJECT_KINDS) - have)
    assert not missing, (
        f"these subject kinds declare no export target, so their pages can never show the "
        f"Export control: {missing}"
    )


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_target_declares_its_data(target):
    """Exactly one of `data_files` / `artifact_is_data`."""
    assert bool(target.data_files) != bool(target.artifact_is_data), (
        f"{target.id!r} must declare exactly one of `data_files` (the files carrying the data "
        f"behind the artifact) or `artifact_is_data` — it declares "
        f"{'both' if target.data_files else 'neither'}"
    )
    for name in target.data_files:
        # A trailing slash names a directory, for a target whose file names depend on the run.
        assert name.endswith(
            (".csv", ".npy", ".yaml", ".json", "/")
        ), f"{target.id!r} lists {name!r} as a data file; that is not a readable data format"


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_the_included_list_always_carries_the_sidecars(target):
    """Whatever a target declares, params.csv and export.yaml land beside it.

    Listed once here rather than by every target, so the list a surface shows is the whole
    list and cannot be short by whatever one target forgot.
    """
    included = target.included_files()
    assert included[-2:] == (exporters.PARAMS_NAME, exporters.MANIFEST_NAME)
    for name in target.data_files:
        assert name in included


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_target_describes_its_artifact(target):
    assert (
        len(target.artifact.split()) >= 4
    ), f"{target.id!r} does not say what it writes; `artifact` is shown to the user"
    assert (
        target.label and target.label[0].isupper()
    ), f"{target.id!r} needs a human label — it is what the Export menu shows"


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_target_is_reachable_through_the_catalogue(target):
    """`target()` finds every registered entry — the lookup both surfaces use."""
    assert exporters.target(target.subject_kind, target.id) is target


def test_every_target_can_say_why_it_is_off():
    """A greyed-out row has to name the missing thing, so every target needs its sentence.

    The rule is **grey out, don't hide**: a target whose inputs are absent is still listed, and
    the reader's next question is "so what do I do". A target missing from this table crashes
    the dialog rather than quietly not appearing — caught at import time instead of in front of
    a user.
    """
    missing = [
        f"{t.subject_kind}.{t.id}"
        for t in exporters.all_targets()
        if (t.subject_kind, t.id) not in export_targets._REASONS
    ]
    assert not missing, f"no unavailability reason for: {missing}"


def test_an_unavailable_reason_names_the_missing_thing(tmp_path):
    """ "Unavailable" is not actionable; "this recording has no streams imported yet" is.

    Driven against an experiment with nothing in it, which is where a new user meets these.
    """
    from detail.api.experiment import Experiment
    from detail.api.exporters import ExportSubject

    experiment = Experiment.create("empty", tmp_path)
    experiment.create_recording("r1")
    subject = ExportSubject("recording", experiment.dir, "r1")

    seen = {}
    for target in exporters.targets_for(subject):
        reason = exporters.unavailable_reason(target, subject)
        seen[target.id] = reason
        if reason is not None:
            assert (
                len(reason.split()) >= 5
            ), f"{target.id!r} says {reason!r}, which does not tell anyone what to do"
            assert "unavailable" not in reason.lower()
    assert seen, "a fresh recording listed no targets at all"
    assert any(
        v is not None for v in seen.values()
    ), "nothing was unavailable on an empty recording, so this asserts nothing"


def test_the_scan_would_catch_a_target_that_declares_neither():
    """Guard the guard: the rule must be refusable, or it passes vacuously forever."""
    with pytest.raises(ValueError, match="exactly one of"):
        exporters.register(
            exporters.ExportTarget(
                id="_undeclared",
                label="Undeclared",
                subject_kind="experiment",
                artifact="something with no stated inputs at all",
                run=lambda *a, **k: [],
            )
        )


def test_the_scan_would_catch_a_target_that_declares_both():
    with pytest.raises(ValueError, match="exactly one of"):
        exporters.register(
            exporters.ExportTarget(
                id="_overdeclared",
                label="Overdeclared",
                subject_kind="experiment",
                artifact="something that declares its data twice over",
                data_files=("rows.csv",),
                artifact_is_data=True,
                run=lambda *a, **k: [],
            )
        )


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_a_declared_figure_is_a_figure(target):
    """`figure_files` holds pictures; `data_files` holds data. Keeping them apart is the point.

    A target that listed only PNGs under `data_files` would satisfy the "names its data" rule
    while shipping exactly the Save-PNG button the rule exists to prevent — so the split is
    checked rather than trusted.
    """
    for name in target.figure_files:
        # A trailing slash names a *directory* of figures, for a target whose figure names are
        # not knowable in advance — a comparison draws one per measurement, and how many
        # depends on the run.
        assert name.endswith(
            (".png", ".svg", ".pdf", "/")
        ), f"{target.id!r} lists {name!r} as a figure; that is not an image format"
    for name in target.data_files:
        assert not name.endswith((".png", ".svg", ".pdf")), (
            f"{target.id!r} lists the picture {name!r} as data. A figure is an artifact, not "
            f"the data behind one — put it in `figure_files`."
        )


def test_a_target_that_draws_declares_every_format_it_draws():
    """PNG for looking at, SVG for a paper — both, or the declaration is short.

    The one that matters is SVG: a reader who only ever sees the PNG cannot tell it is missing.
    """
    for target in exporters.all_targets():
        stems = {
            name.rsplit(".", 1)[0]
            for name in target.figure_files
            if not name.endswith(
                "/"
            )  # a directory carries whatever formats were asked for
        }
        for stem in stems:
            for suffix in ("png", "svg"):
                assert (
                    f"{stem}.{suffix}" in target.figure_files
                ), f"{target.id!r} draws {stem} but does not declare its .{suffix}"


# --------------------------------------------------------------------------- #
# Every processing module's runs have a way out
# --------------------------------------------------------------------------- #
# **The class.** *A registry grows a member and a parallel table does not, so the member works
# and one fact about it silently does not exist.* All four processing modules share the `run`
# subject kind, so `test_every_result_kind_has_an_export_target` was satisfied by whichever
# module happened to have a target — `vectors` alone satisfied it, while a detection run and a
# tracking run could export nothing but a performance report. That is the complaint
# `docs/export_design.md` §1 opens with ("your tracking data requires an analysis run to get
# out"), sitting behind a green suite because the existing guard counted *kinds*, not producers.
#
# `performance` is excluded on purpose: it is available for every run, so a sweep that counted
# it would pass for a module with no data export at all — precisely the state being guarded.


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_every_target_can_ignore_another_targets_options(target):
    """**The class.** *A registry member that does not implement the signature its contract
    declares.*

    `ExportTarget.run`'s contract is `(subject, dest, scope, progress, **options)` — "each
    target takes what it recognises and ignores the rest" — because one export is **one call per
    target with one shared options dict**. `_run_performance` was the one runner of thirteen
    without `**options`, so ticking *Performance report* beside *Clips* died on
    `unexpected keyword argument 'renders'`. Reported from the app.

    What makes it invisible: every target works perfectly **alone**, and the tests exercise them
    alone. It is only the combination that breaks, and there are more combinations than anyone
    writes tests for — so the check is on the signature, where it is one assertion for all of
    them and covers a target added tomorrow.
    """
    import inspect

    parameters = inspect.signature(target.run).parameters.values()
    assert any(p.kind is p.VAR_KEYWORD for p in parameters), (
        f"{target.subject_kind}.{target.id}'s runner takes no **options, so it raises when it "
        f"is exported alongside a target that has knobs — see ExportTarget.run's contract"
    )


def test_the_sweep_would_catch_a_runner_that_cannot_ignore_options():
    """Guard the guard: the signature check must actually refuse one."""

    def narrow(subject, dest, scope=None, progress=None):
        return []

    bad = exporters.ExportTarget(
        id="_narrow",
        label="Narrow",
        subject_kind="experiment",
        artifact="a runner that cannot ignore another target's knobs",
        artifact_is_data=True,
        run=narrow,
    )
    with pytest.raises(AssertionError, match="no \\*\\*options"):
        test_every_target_can_ignore_another_targets_options(bad)


def _run_target_ids() -> set:
    return {t.id for t in exporters.all_targets() if t.subject_kind == "run"}


def test_every_processing_module_declares_how_its_runs_export():
    """A module added tomorrow has to say how its output gets out, or fail here."""
    from detail.executor.runners import PROCESSING_MODULES

    declared = set(export_targets._RUN_TARGETS)
    missing = sorted(set(PROCESSING_MODULES) - declared)
    assert not missing, (
        f"these processing modules have no entry in `_RUN_TARGETS`, so a user on one of their "
        f"runs is offered nothing but a performance report: {missing}. Add the target its runs "
        f"produce, or the id of an existing one it shares."
    )
    stale = sorted(declared - set(PROCESSING_MODULES))
    assert not stale, f"`_RUN_TARGETS` names modules that do not exist: {stale}"


def test_every_declared_run_target_is_registered():
    """The table names target ids; a typo there is a module that quietly exports nothing."""
    named = {tid for ids in export_targets._RUN_TARGETS.values() for tid in ids}
    unknown = sorted(named - _run_target_ids())
    assert not unknown, f"`_RUN_TARGETS` names unregistered targets: {unknown}"


def test_every_run_target_that_names_a_producer_can_say_so():
    """The refusal sentence is a second table, and the two have to agree.

    `_RUN_TARGETS` is the fact (which module produces what); `_RUN_SOURCE` is how a refusal
    words it. A target in the first and not the second raises `KeyError` inside the reason
    function — at the moment a user opens the Export dialog on the wrong kind of run.
    """
    named = {tid for ids in export_targets._RUN_TARGETS.values() for tid in ids}
    assert named == set(export_targets._RUN_SOURCE), (
        f"`_RUN_TARGETS` and `_RUN_SOURCE` disagree: "
        f"{sorted(named ^ set(export_targets._RUN_SOURCE))}"
    )


def test_the_sweep_would_catch_a_module_with_no_way_out(monkeypatch):
    """Guard the guard: register a module with no entry and the sweep must fail."""
    from detail.executor.runners import PROCESSING_MODULES

    monkeypatch.setitem(PROCESSING_MODULES, "_unexportable", object())
    with pytest.raises(AssertionError, match="_unexportable"):
        test_every_processing_module_declares_how_its_runs_export()


def test_the_sweep_would_catch_a_target_id_that_does_not_exist(monkeypatch):
    monkeypatch.setitem(export_targets._RUN_TARGETS, "detection", ("no_such_target",))
    with pytest.raises(AssertionError, match="no_such_target"):
        test_every_declared_run_target_is_registered()
