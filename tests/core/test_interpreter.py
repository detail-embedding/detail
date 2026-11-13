"""An interpreter DETAIL starts must not import from the folder it happens to start in.

**The class: an interpreter started from a directory whose contents it then trusts.**
``python -m x``, ``python -`` and ``python -c`` put their working directory first on
``sys.path``, and DETAIL starts interpreters in directories it does not choose. A source
checkout there has no ``__init__.py`` at its root, so Python takes it for an empty *namespace
package* of that name, and an editable install's finder (which sits *after* the path finder in
``sys.meta_path``) never gets to answer for the top-level name. Submodules still resolve through
that finder, so most imports work and the failure surfaces far from its cause.

**Twice, on the real cluster.**

1. ``remote_python``'s probes run ``python -`` over ssh, whose working directory is ``$HOME``,
   where the install guide put DETAIL's checkout as ``~/detail``. The build check's
   ``from detail import __version__`` failed with ``cannot import name '__version__' from
   'detail' (unknown location)``, which reads as "this cluster is out of date" and was the
   opposite. The fix then was a prefix that dropped ``sys.path[0]`` from every *probe* snippet,
   and its write-up recorded the worker as safe because it "runs from a run directory, not from
   ``$HOME``".
2. That verdict was wrong. A Slurm job starts where ``sbatch`` ran, and DETAIL runs ``sbatch``
   over ssh, so from ``$HOME``, where the guide put SAM 3's checkout as ``~/sam3``. A
   fresh-setup keypoint distill's labelling job died at "loading model" with
   ``TypeError: expected str, bytes or os.PathLike object, not NoneType``: SAM 3's builder asks
   ``pkg_resources`` for a file inside the ``sam3`` package, and the namespace package has no
   ``__file__``. On Tillicum, from ``~``, ``print(sam3)`` gave ``<module 'sam3' (namespace) from
   ['/gpfs/home/<user>/sam3']>``.

The local launches had the same shape: a worker, coordinator, shepherd or video stager inherits
whatever folder the user launched the app from, and the guides put ``~/sam3`` beside
``~/detail``.

**The rule.** Every interpreter DETAIL starts runs with ``-P`` (``safe_path``, Python 3.11+), via
``core/interpreter.py``: ``module_argv`` locally, ``remote_python_command`` in a remote shell.
``-P`` drops only the start-up entry; ``PYTHONPATH`` stays (the model below puts its runner on
``PYTHONPATH``, so a flag that dropped it would fail here too). The probe prefix is retired: one
mechanism, not two.

**Where the fix stops.** ``docs_access.build_site`` and ``detail docs --serve`` run ``mkdocs`` with
``cwd`` set to DETAIL's own source tree, a folder DETAIL chose, whose ``detail/`` is the real
package; they are listed in ``_CHOSEN_CWD`` and a stale entry fails.

Modelled with a real interpreter: a ``shadowpkg/`` checkout (no ``__init__.py``) in the starting
folder, and the installed ``shadowpkg`` served only by a finder appended to ``sys.meta_path`` (from
a ``sitecustomize`` on ``PYTHONPATH``), as an editable install's is.
"""

import ast
import os
import pathlib
import re
import shlex
import subprocess
import sys
import textwrap

from detail.core import interpreter
from detail.executor import remote_python
from detail.executor.schedulers import slurm

PACKAGE = pathlib.Path(interpreter.__file__).resolve().parents[1]

_SITECUSTOMIZE = textwrap.dedent("""
    import importlib.abc, importlib.util, sys
    from pathlib import Path

    class _EditableFinder(importlib.abc.MetaPathFinder):
        root = Path({root!r})

        def find_spec(self, name, path=None, target=None):
            if name != "shadowpkg":
                return None
            init = self.root / "shadowpkg" / "__init__.py"
            return importlib.util.spec_from_file_location(
                name, init, submodule_search_locations=[str(init.parent)]
            )

    sys.meta_path.append(_EditableFinder())
    """)

#: What the worker did that failed: resolve a file inside the package from its location.
_RUNNER = "import shadowpkg\nprint(getattr(shadowpkg, '__file__', None))\n"


def _layout(tmp_path):
    """``(start, env, installed_init)``: a folder holding a ``shadowpkg`` checkout, an environment
    whose ``shadowpkg`` is an editable-style install, and where that install's code is.
    """
    start = tmp_path / "home"
    (start / "shadowpkg").mkdir(
        parents=True
    )  # the checkout: a repo root, no __init__.py
    (start / "shadowpkg" / "README.md").write_text("checkout", encoding="utf-8")
    installed = tmp_path / "installed"
    (installed / "shadowpkg").mkdir(parents=True)
    init = installed / "shadowpkg" / "__init__.py"
    init.write_text("", encoding="utf-8")
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(
        _SITECUSTOMIZE.format(root=str(installed)), encoding="utf-8"
    )
    (shim / "runner.py").write_text(_RUNNER, encoding="utf-8")
    # The child prints a path, which may hold any character: both ends of the pipe say UTF-8.
    env = {**os.environ, "PYTHONPATH": str(shim), "PYTHONIOENCODING": "utf-8"}
    return start, env, init


def _run(argv, start, env, stdin=None) -> str:
    done = subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        encoding="utf-8",
        cwd=start,
        env=env,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr[-600:]
    return done.stdout.strip()


def _remote_argv(rest: str) -> list[str]:
    """``remote_python_command(rest)`` as the remote shell splits it, with this interpreter."""
    argv = shlex.split(interpreter.remote_python_command(rest))
    assert argv[0] == "python"
    return [sys.executable, *argv[1:]]


def test_the_model_reproduces_the_failure_without_the_flag(tmp_path):
    """Guard the guard: the plain launch must get the empty checkout, or the tests below prove
    nothing."""
    start, env, _ = _layout(tmp_path)

    assert _run([sys.executable, "-m", "runner"], start, env) == "None"
    assert _run([sys.executable, "-"], start, env, stdin=_RUNNER) == "None"


def test_a_local_launch_imports_the_installed_package_wherever_it_starts(tmp_path):
    start, env, init = _layout(tmp_path)

    assert _run(interpreter.module_argv("runner"), start, env) == str(init)


def test_a_remote_module_launch_imports_the_installed_package_from_home(tmp_path):
    start, env, init = _layout(tmp_path)

    assert _run(_remote_argv("-m runner"), start, env) == str(init)


def test_a_remote_snippet_imports_the_installed_package_from_home(tmp_path):
    start, env, init = _layout(tmp_path)

    assert _run(_remote_argv("-"), start, env, stdin=_RUNNER) == str(init)


def test_the_sbatch_script_starts_the_worker_through_the_safe_command():
    target = slurm.RemoteTargetConfig(name="t", host="h", remote_workdir="/scratch/w")
    script = slurm.render_sbatch_script(target, "job_1", "/scratch/w/runs/r1", "m")

    expected = interpreter.remote_python_command(
        "-m detail.executor.worker /scratch/w/runs/r1"
    )
    assert script.rstrip("\n").splitlines()[-1] == expected


def test_remote_python_sends_its_snippet_through_the_safe_command(monkeypatch):
    sent = []

    def fake_run_remote(host, command, **kwargs):
        sent.append((command, kwargs["stdin_text"]))
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(remote_python.ssh_transport, "run_remote", fake_run_remote)
    monkeypatch.setattr(remote_python, "activation_prefix", lambda target: "act && ")

    class Target:
        host = "h"
        name = "t"

    remote_python.remote_python(Target(), "print(1)\n", "failed")

    assert sent == [("act && " + interpreter.remote_python_command("-"), "print(1)\n")]


# --- the scan: no interpreter started any other way --------------------------------------- #

#: Launches whose working directory DETAIL sets itself, and why that is enough.
_CHOSEN_CWD = {
    "detail/docs_access.py": "mkdocs build runs with cwd= DETAIL's own source tree",
    "detail/cli.py": "`detail docs --serve` runs mkdocs with cwd= DETAIL's own source tree",
}

#: A shell command that starts ``python`` with a flag: ``python -m``, ``python3 -``, ``python -c``.
_SHELL_PYTHON = re.compile(r"(?:^|[\s;&|(])python3?\s+-")


def _docstring_nodes(tree) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _is_sys_executable(node) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def unsafe_launches(source: str) -> list[int]:
    """Lines that start an interpreter other than through ``core/interpreter.py``: an argv
    holding ``sys.executable`` and ``-m`` / ``-c`` / ``-``, or a string that is a shell
    ``python -…`` command. Docstrings and ``usage:`` messages are prose and are skipped.
    """
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            flags = {
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
            if any(_is_sys_executable(e) for e in node.elts) and flags & {
                "-m",
                "-c",
                "-",
            }:
                found.append(node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and not node.value.lstrip().startswith("usage:")
            and _SHELL_PYTHON.search(node.value)
        ):
            found.append(node.lineno)
    return sorted(set(found))


def _package_launches() -> dict[str, list[int]]:
    hits = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent).as_posix()
        if rel == "detail/core/interpreter.py":
            continue
        lines = unsafe_launches(path.read_text(encoding="utf-8-sig"))
        if lines:
            hits[rel] = lines
    return hits


def test_every_interpreter_detail_starts_goes_through_core_interpreter():
    hits = _package_launches()
    unexcused = {rel: lines for rel, lines in hits.items() if rel not in _CHOSEN_CWD}

    assert unexcused == {}, (
        "start interpreters with core.interpreter.module_argv / remote_python_command, "
        f"so the starting folder is not on sys.path: {unexcused}"
    )


def test_every_chosen_cwd_entry_still_excuses_a_launch():
    """An entry is a claim: one whose launch has gone, or moved to the helper, fails."""
    hits = _package_launches()

    assert sorted(rel for rel in _CHOSEN_CWD if rel not in hits) == []


def test_the_scan_would_catch_the_spellings_a_person_writes():
    assert unsafe_launches('a = [sys.executable, "-m", "detail.executor.worker", d]\n')
    assert unsafe_launches("a = (\n    sys.executable,\n    '-c',\n    code,\n)\n")
    assert unsafe_launches('lines.append(f"python -m detail.executor.worker {d}")\n')
    assert unsafe_launches('command = f"{activation_prefix(target)}python -"\n')
    assert unsafe_launches('cmd = "module load x && python3 -c pass"\n')


def test_the_scan_leaves_the_helper_and_prose_alone():
    assert not unsafe_launches(
        "argv = interpreter.module_argv('detail.executor.worker')\n"
    )
    assert not unsafe_launches(
        'def f():\n    """Invoked as ``python -m detail.executor.worker``."""\n'
    )
    assert not unsafe_launches('argv = [sys.executable, "--version"]\n')
    assert not unsafe_launches(
        'raise SystemExit("usage: python -m detail.executor.worker <run_dir>")\n'
    )
    assert not unsafe_launches('msg = "install it with uv"\n')
