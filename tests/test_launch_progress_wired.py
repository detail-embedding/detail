"""Every surface that launches a processing run drives the launch's progress callback.

**The class.** `Experiment.run_processing` and `retry_run` read every selected recording twice
before they start anything — validation, then the disk-space estimate — and report through a
``progress(stage, fraction, detail)`` callback. A caller that does not pass one gets the old
behaviour, which is not an error: the launch works, and the user watches a single unchanging
sentence ("Validating 71 recording(s)…") for minutes, indistinguishable from a hang. That is how a
71-recording embedding launch read as stuck. Nothing fails at the call that omits it, so this is
invisible to every behavioural test — the callback is optional by design, for the API.

**The rule** (CLAUDE.md: *every caller drives that callback*). Each call to ``run_processing`` /
``retry_run`` in the CLI and the GUI passes ``progress=``. The API itself is the one place it may
be absent, and a run started from code with no one watching needs no bar.

**The guard** reads the source of ``detail/cli*.py`` and ``detail/gui/`` and fails on a call that
omits it, so a seventh launching command or a third launch dialog is held to the rule the day it
is written. The negative case feeds the scan a call without the keyword and asserts it is found.

**Where it stops, deliberately.** `run_analysis` is a single ``module.validate`` over the whole
selection, not a loop over recordings, so there is nothing per-recording to report; the analysis
launch dialog ignores the callback it is handed. The clusterer launchers (hidden, §25) call
``fit_clusterer``, a different API, and are left as they are.
"""

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
LAUNCHES = frozenset({"run_processing", "retry_run"})


def _surface_files() -> list[pathlib.Path]:
    detail = REPO / "detail"
    return sorted([*detail.glob("cli*.py"), *(detail / "gui").rglob("*.py")])


def launches_without_progress(source: str) -> list[int]:
    """Line numbers of calls to a launching method that do not pass ``progress=``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in LAUNCHES):
            continue
        if not any(keyword.arg == "progress" for keyword in node.keywords):
            found.append(node.lineno)
    return found


def test_every_cli_and_gui_launch_drives_the_progress_callback():
    offenders = {}
    for path in _surface_files():
        lines = launches_without_progress(path.read_text(encoding="utf-8-sig"))
        if lines:
            offenders[path.relative_to(REPO).as_posix()] = lines
    assert not offenders, (
        "these calls launch a processing run without passing progress=, so the user watches "
        f"one unchanging sentence through the validation: {offenders}"
    )


def test_the_scan_catches_a_launch_without_the_callback():
    """Negative case: a scan that matched nothing would pass forever."""
    omitted = "run = proj.run_processing('embedding', ids, config)\n"
    retried = "handle = experiment.retry_run(run_id, ignore_free_space=True)\n"
    wired = "run = proj.run_processing('embedding', ids, config, progress=bar)\n"

    assert launches_without_progress(omitted) == [1]
    assert launches_without_progress(retried) == [1]
    assert launches_without_progress(wired) == []


def test_the_scan_reads_the_files_that_launch():
    """And it is looking where launches are: without this, an emptied file list is a green."""
    launching = {
        path.relative_to(REPO).as_posix()
        for path in _surface_files()
        if any(
            name in path.read_text(encoding="utf-8-sig")
            for name in (".run_processing(", ".retry_run(")
        )
    }
    assert {
        "detail/cli.py",
        "detail/cli_keypoints.py",
        "detail/gui/launch.py",
        "detail/gui/pages.py",
    } <= launching
