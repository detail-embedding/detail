"""A run's pre-flight treats ``"remote"`` as a routing decision, not a local backend (§6).

The class: an entry point that accepts a ``ComputeTarget`` checks it as if the job will run
on *this* machine. Two shapes, both invisible to a behavioural test that only ever passes a
local target:

- ``compute.backend not in supported_backends`` — no component lists ``"remote"``, so the
  check refuses every Tillicum job. The capability check reads
  ``core.utils.effective_backend(compute)``, which maps remote to the ``cuda`` it runs as.
- ``if not trainer.is_ready()`` (or a model's local access check) left unconditional — it
  demands this machine have a framework or weights for a job that never runs here. Gate it
  on ``compute.backend != "remote"``.

``distill()`` was fixed for both, by inspection, and ``label_dataset`` / ``train_model``
before it; ``distill_keypoints`` was written later beside them and repeated both, so a
keypoint student could not be trained on the cluster at all. Fixed by giving it
``distill()``'s guard, and pinned here as a scan because the next entry point will be
written the same way.

Two sites are allowed, keyed by (file, function):

- ``core/utils.py::resolve_device`` maps a target to a *local* device string; a remote job
  is rewritten to cuda before it reaches it, so refusing ``remote`` there is correct.
- ``api/recording.py::_validate_prompt_test`` — the prompt-test preview always runs here,
  so the bug there is that a remote target reaches it at all; tracked as its own fix.
"""

import ast
from pathlib import Path

import detail

PACKAGE = Path(detail.__file__).parent

ALLOWED = {
    ("detail/core/utils.py", "resolve_device"),
    ("detail/api/recording.py", "_validate_prompt_test"),
}


def _is_backend_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "backend"


def _calls_is_ready(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "is_ready"
        for n in ast.walk(node)
    )


def _mentions_remote(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Constant) and n.value == "remote" for n in ast.walk(node)
    )


def local_only_preflight(source: str) -> list[tuple[str, int]]:
    """``(function, line)`` of every check that treats a compute target as local.

    A raw ``<x>.backend in / not in <...>`` membership test, or an ``if`` that consults
    ``is_ready()`` without also deciding on ``"remote"``.
    """
    found = []

    def visit(node: ast.AST, func: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = node.name
        if (
            isinstance(node, ast.Compare)
            and _is_backend_attr(node.left)
            and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
        ):
            found.append((func, node.lineno))
        if (
            isinstance(node, ast.If)
            and _calls_is_ready(node.test)
            and not _mentions_remote(node.test)
        ):
            found.append((func, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, func)

    visit(ast.parse(source), "<module>")
    return found


def _offenders() -> list[str]:
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent).as_posix()
        if rel.startswith("detail/gui/"):
            continue  # the GUI's readiness labels describe this machine, by design
        for func, line in local_only_preflight(path.read_text(encoding="utf-8-sig")):
            if (rel, func) not in ALLOWED:
                offenders.append(f"{rel}:{line} ({func})")
    return offenders


def test_no_preflight_treats_a_remote_target_as_local():
    offenders = _offenders()
    assert not offenders, (
        "a compute check that refuses or gates a remote job on this machine's backend — "
        "use effective_backend(compute) for the capability check and skip local "
        "readiness when compute.backend == 'remote' (see this module's docstring):\n"
        + "\n".join(offenders)
    )


def test_every_allowance_still_excuses_something():
    scanned = {
        (path.relative_to(PACKAGE.parent).as_posix(), func)
        for path in PACKAGE.rglob("*.py")
        for func, _ in local_only_preflight(path.read_text(encoding="utf-8-sig"))
    }
    assert ALLOWED <= scanned, f"stale allowance(s): {sorted(ALLOWED - scanned)}"


def test_the_scan_would_catch_a_raw_backend_check():
    source = (
        "def launch(compute, trainer):\n"
        "    if compute.backend not in trainer.supported_backends:\n"
        "        raise ValueError\n"
        "def wrapped(train_compute, model):\n"
        "    if (\n"
        "        train_compute.backend\n"
        "        in model.supported_backends\n"
        "    ):\n"
        "        pass\n"
    )
    assert [f for f, _ in local_only_preflight(source)] == ["launch", "wrapped"]


def test_the_scan_would_catch_an_unconditional_readiness_gate():
    source = (
        "def launch(compute, trainer):\n"
        "    if not trainer.is_ready():\n"
        "        raise ValueError\n"
    )
    assert local_only_preflight(source) == [("launch", 2)]


def test_the_scan_passes_the_remote_aware_spellings():
    source = (
        "def launch(compute, trainer):\n"
        "    if compute.backend != 'remote' and not trainer.is_ready():\n"
        "        raise ValueError\n"
        "    if effective_backend(compute) not in trainer.supported_backends:\n"
        "        raise ValueError\n"
    )
    assert local_only_preflight(source) == []
