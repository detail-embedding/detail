"""Every CLI option that takes a recording takes its **name** (§11).

The class: a CLI option that names a recording, handed on to the API as typed. The run APIs
take ids; what a user types is the name, so the translation has to happen in the CLI or the
name reaches the run as a directory that does not exist. ``detail analysis run --recording
"larva A"`` advertised "Recording name (or id)" and died in ``validate()`` with
``FileNotFoundError: no recording.yaml in …/recordings/larva A`` — a traceback, not even a
clean refusal — while the ``process`` commands beside it resolved through
``_resolve_recording_refs``. The sweep found eleven more: ``analysis query`` (``--search``,
``--example``), ``pose-clustering``, ``temporal-clustering``, ``query-eval``, ``comparison
run``, ``dataset label``, ``distill run``, ``remote staging-start``, ``export --recording``,
and ``_spec_from_ref`` behind every ``--*-ref RECORDING:run`` selector.

It hid because nothing is wrong at the line where it is written: ``list(recording_ids)`` is
correct for an id, and every fixture names its recordings by id. Only a test that types a
**name** sees it — so the behaviour tests here rename the recording first.

The rule: a recording a user types reaches a resolver (``_recording``,
``_resolve_recording_refs``, ``Experiment.resolve_recording`` or a helper that calls one)
before anything else reads it. The scan below enumerates the click tree, so a new command is
covered the day it is added; a param that is not resolved is either hidden (§25 — a deliberate
non-fix, checked against ``surfaces.HIDDEN``), not a recording at all, or resolved further
down, and each of those is a claim this file checks rather than trusts.
"""

import ast
import inspect
import re
import textwrap

import click
import pytest
from click.testing import CliRunner

from detail import cli
from detail import surfaces
from detail.api.experiment import Experiment
from detail.cli import main

#: Calls that turn what was typed into a recording id (or refuse it cleanly).
RESOLVERS = frozenset(
    {
        "_recording",
        "_resolve_recording_refs",
        "resolve_recording",
        "_override_selection",
        "_spec_selection",
        "_spec_from_ref",
        "_embedding_selection",
        "_examples_by_id",
        "_resolve_subject",
    }
)

#: Uses that only look at the value — a test, a count, a message. `_echo_track_tune` prints the
#: re-run hint with the ref as it was typed.
_NEUTRAL_CALLS = frozenset(
    {"len", "bool", "echo", "secho", "ClickException", "UsageError", "_echo_track_tune"}
)

#: Resolved further down, by a callee that resolves its own refs. Each names the callee the
#: command must actually call, which the scan checks.
RESOLVED_DOWNSTREAM = {
    # `Experiment.propagate_spatial` resolves its source and its targets.
    **{
        (f"experiment propagate-{kind}", param): "_propagate_cli"
        for kind in ("fov", "calibration", "rois", "points")
        for param in ("source_ref", "only")
    },
    ("analysis query", "examples"): "_examples_by_id",
    # The keypoint API takes refs and resolves them itself (`api/keypoints.py`,
    # `api/keypoint_distill.py`), so these CLI commands hand them on as typed.
    ("keypoints examples sample", "recordings"): "sample_example_frames",
    ("keypoints examples sample", "from_recordings"): "sample_example_frames",
    ("keypoints examples add-frame", "recording"): "add_example_frame",
    ("keypoints track-preview", "recording_ids"): "preview_keypoint_tracking",
    ("keypoints track-preview", "starts"): "preview_keypoint_tracking",
    ("keypoints setup start", "teacher_specs"): "start_keypoint_setup",
    **{
        (command, param): "_teacher_args"
        for command in ("dataset keypoints", "keypoints test", "keypoints distill")
        for param in ("recording_ids", "detect_run_refs", "tracking_run_refs")
    },
}

#: Matched by the predicate, and not a reference to an existing recording.
NOT_A_REF = {
    ("analysis audit", "by_recording"): "a flag",
    ("keypoints examples sample", "from_recordings"): "a count of recordings",
}


def _commands(group, path=()):
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _commands(command, (*path, name))
        else:
            yield " ".join((*path, name)), command


def _hidden(path: str) -> bool:
    hidden = {
        p for entry in surfaces.HIDDEN.values() if entry.in_effect for p in entry.cli
    }
    return any(path == p or path.startswith(p + " ") for p in hidden)


_NAMES_A_RECORDING = re.compile(
    r"recording[:\[]|\(id or name\)|\(an id or a name\)|name \(or id\)|recording name",
    re.IGNORECASE,
)


def _takes_a_recording(param: click.Parameter) -> bool:
    if "experiment_dir" == param.name:
        return False
    help_text = getattr(param, "help", None) or ""
    metavar = str(getattr(param, "metavar", None) or "")
    return (
        "recording" in (param.name or "")
        or "RECORDING" in metavar
        or bool(_NAMES_A_RECORDING.search(help_text))
    )


def _call_name(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _calls_a_resolver(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and _call_name(n) in RESOLVERS for n in ast.walk(node)
    )


def _use_kind(use: ast.Name, parents: dict) -> str:
    """``resolved`` / ``neutral`` / ``raw`` for one read of the param."""
    child, node = use, parents.get(use)
    while node is not None:
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in RESOLVERS:
                return "resolved"
            if name in _NEUTRAL_CALLS:
                return "neutral"
            return f"raw:{name}"
        if isinstance(node, ast.comprehension) and child is node.iter:
            owner = parents[node]
            body = [
                getattr(owner, f) for f in ("elt", "key", "value") if hasattr(owner, f)
            ]
            return (
                "resolved" if any(map(_calls_a_resolver, body)) else "raw:comprehension"
            )
        if isinstance(node, ast.For) and child is node.iter:
            return "resolved" if any(map(_calls_a_resolver, node.body)) else "raw:for"
        if isinstance(node, (ast.If, ast.While, ast.IfExp)) and child is node.test:
            return "neutral"
        if isinstance(node, (ast.BoolOp, ast.Compare, ast.JoinedStr)):
            return "neutral"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return "neutral"
        if isinstance(node, ast.stmt):
            return f"raw:{type(node).__name__}"
        child, node = node, parents.get(node)
    return "raw"


def _callback_tree(command: click.Command) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(command.callback))
    return next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
    )


def _sites():
    """``(command path, param name, verdict)`` for every recording param in the CLI."""
    for path, command in _commands(main):
        params = [p for p in command.params if _takes_a_recording(p)]
        if not params:
            continue
        tree = _callback_tree(command)
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for param in params:
            kinds = [
                _use_kind(n, parents)
                for n in ast.walk(tree)
                if isinstance(n, ast.Name)
                and n.id == param.name
                and isinstance(n.ctx, ast.Load)
            ]
            raw = [k for k in kinds if k.startswith("raw")]
            ok = "resolved" in kinds and not raw
            yield path, param.name, "resolved" if ok else ", ".join(raw) or "unused"


def test_every_recording_option_reaches_a_resolver():
    calls = {}
    for path, command in _commands(main):
        tree = _callback_tree(command)
        calls[path] = {_call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    unresolved = []
    for path, param, verdict in _sites():
        key = (path, param)
        if verdict == "resolved" or key in NOT_A_REF or _hidden(path):
            continue
        if key in RESOLVED_DOWNSTREAM and RESOLVED_DOWNSTREAM[key] in calls[path]:
            continue
        unresolved.append(f"{path} {param}: {verdict}")
    assert not unresolved, (
        "these CLI options take a recording and hand it on as typed — resolve each through "
        "_recording / _resolve_recording_refs (a name is what a user types, §11):\n  "
        + "\n  ".join(unresolved)
    )


@pytest.mark.parametrize("helper", sorted(RESOLVERS - {"resolve_recording"}))
def test_every_resolver_resolves(helper):
    """``RESOLVERS`` is a claim too: each CLI helper on it calls another resolver. The old
    ``_spec_from_ref`` sat beside the resolvers and called ``proj.get_recording`` instead.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(cli, helper))))
    called = {_call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert called & (RESOLVERS - {helper}), helper


def test_every_exemption_still_excuses_something():
    """An allowlist entry that names no live, unresolved param is a claim nobody re-checks."""
    sites = {(path, param): verdict for path, param, verdict in _sites()}
    stale = [
        key
        for key in (*RESOLVED_DOWNSTREAM, *NOT_A_REF)
        if sites.get(key, "resolved") == "resolved"
    ]
    assert not stale, stale


def test_the_scan_flags_a_raw_recording_list():
    """The negative case: the shape the bug had is reported, the fixed shape is not."""

    def verdict(body: str) -> str:
        tree = ast.parse(textwrap.dedent(body)).body[0]
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "refs"]
        return ",".join(_use_kind(n, parents) for n in uses)

    assert verdict("""
        def f(proj, refs):
            config = {"source_recording_ids": list(refs)}
        """).startswith("raw")
    assert verdict("""
        def f(proj, refs):
            config = {"source_recording_ids": refs}
        """).startswith("raw")
    assert verdict("""
        def f(proj, refs):
            config = {"source_recording_ids": _resolve_recording_refs(proj, refs)}
        """) == "resolved"
    assert verdict("""
        def f(proj, refs):
            ids = [proj.resolve_recording(r).id for r in refs]
        """) == "resolved"


# --------------------------------------------------------------------------- #
# Behaviour: a recording renamed away from its id, typed by name.
# --------------------------------------------------------------------------- #
@pytest.fixture
def named(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001").set_name("larva A")
    proj.create_recording("recording_002").set_name("larva: B")
    return proj


def _capture_analysis(monkeypatch) -> dict:
    seen: dict = {}

    def capture(self, module_id, config):
        seen.update(config)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "run_analysis", capture)
    return seen


def test_analysis_run_takes_a_recording_name(named, monkeypatch):
    seen = _capture_analysis(monkeypatch)
    result = CliRunner().invoke(
        main,
        [
            "analysis", "run", "--experiment", str(named.dir), "--recording", "larva A",
            "--recording", "recording_002", "--tracking", "track_01",
        ],
    )  # fmt: skip
    assert "stop here" in result.output, result.output
    assert seen["source_recording_ids"] == ["recording_001", "recording_002"]


def test_an_unknown_recording_is_a_clean_refusal(named, monkeypatch):
    _capture_analysis(monkeypatch)
    result = CliRunner().invoke(
        main,
        [
            "analysis", "run", "--experiment", str(named.dir), "--recording", "larva Z",
            "--tracking", "track_01",
        ],
    )  # fmt: skip
    assert result.exit_code != 0
    assert "no recording 'larva Z'" in result.output and "larva A" in result.output
    assert "Traceback" not in result.output


def test_a_run_ref_takes_a_recording_name_with_a_colon(named, monkeypatch):
    from detail.api import run_spec

    monkeypatch.setattr(
        run_spec,
        "spec_from_run",
        lambda recording, module, run_id: (recording.id, run_id),
    )
    assert cli._spec_from_ref(named, "tracking", "larva: B:track_02") == (
        "recording_002",
        "track_02",
    )


def test_an_export_subject_takes_a_recording_name(named):
    subject = cli._resolve_subject(named.dir, None, "larva A", None, None)
    assert (subject.kind, subject.id) == ("recording", "recording_001")


def test_an_example_clip_takes_a_recording_name(named):
    _, clip = cli._parse_example("rear=larva A:0:1.0-2.0", 0)
    assert cli._examples_by_id(named, {"rear": [clip]})["rear"][0]["recording_id"] == (
        "recording_001"
    )
