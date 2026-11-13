"""A snippet run on the cluster must not see the login directory's own folders.

**The class: an interpreter started from a directory whose contents it then trusts.** `python -`
puts the working directory first on `sys.path`, and an ssh login's working directory is `$HOME`,
where DETAIL's checkout is, by the install instructions, a folder called `detail`. A repo root has
no `__init__.py`, so `import detail` from `$HOME` is a namespace package, `<module 'detail'
(unknown location)>`, and an editable install's finder (which sits *after* the path finder in
`sys.meta_path`) never gets to answer for the top-level name. Submodules still resolved through
that finder, so every probe that only imported `detail.<something>` worked, and the first one that
asked for a name on `detail` itself (the build check: `from detail import __version__`) failed:

    could not read the DETAIL build on 'tillicum': ImportError: cannot import name
    '__version__' from 'detail' (unknown location)

which reads as "this cluster is out of date" and is the opposite: it was current. Found on the
real cluster (`cwd /gpfs/home/sagolden`, `find_spec('detail').origin None`,
`submodule_search_locations ['/gpfs/home/sagolden/detail']`, no `__init__.py` there).

Modelled here with a real interpreter: a `detail/` folder with no `__init__.py` in the working
directory, and the real package served only by a finder appended to `sys.meta_path`, as an
editable install's is. Without `probe_script`'s prefix the failure is reproduced; with it the name
is found.
"""

import os
import subprocess
import sys
import textwrap

from detail.executor import remote_python

#: What the cluster's environment does for an editable install: a finder *after* the path finder.
_EDITABLE_INSTALL = textwrap.dedent("""
    import importlib.abc, importlib.util, sys

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self, root):
            self.root = root

        def find_spec(self, name, path=None, target=None):
            if name != "detail" and not name.startswith("detail."):
                return None
            parts = name.split(".")
            base = self.root.joinpath(*parts)
            init = base / "__init__.py"
            if init.is_file():
                return importlib.util.spec_from_file_location(
                    name, init, submodule_search_locations=[str(base)]
                )
            if base.with_suffix(".py").is_file():
                return importlib.util.spec_from_file_location(name, base.with_suffix(".py"))
            return None

    from pathlib import Path
    sys.meta_path.append(_Finder(Path({root!r})))
    """)


def _cluster(tmp_path):
    """``(home, installed)``: a login directory holding a `detail` repo root, and the package."""
    home = tmp_path / "home"
    (home / "detail").mkdir(parents=True)  # the checkout: a repo root, no __init__.py
    (home / "detail" / "README.md").write_text("DETAIL", encoding="utf-8")
    installed = tmp_path / "installed"
    (installed / "detail" / "core").mkdir(parents=True)
    (installed / "detail" / "__init__.py").write_text(
        '__version__ = "9.9"\n', encoding="utf-8"
    )
    (installed / "detail" / "core" / "__init__.py").write_text("", encoding="utf-8")
    (installed / "detail" / "core" / "buildinfo.py").write_text(
        "def report():\n    from detail import __version__\n    return {'version': __version__}\n",
        encoding="utf-8",
    )
    return home, installed


def _run(tmp_path, script: str) -> subprocess.CompletedProcess:
    home, installed = _cluster(tmp_path)
    # `-S` and no PYTHONPATH: this interpreter must know no `detail` but the model's, not this
    # repo's (a venv's own editable install, or the test run's path, would answer for it).
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-S", "-"],
        input=_EDITABLE_INSTALL.format(root=str(installed)) + script,
        capture_output=True,
        text=True,
        cwd=home,
        env=env,
        timeout=60,
    )


_BUILD_PROBE = (
    "from detail.core import buildinfo\nprint(buildinfo.report()['version'])\n"
)


def test_the_probe_that_failed_on_the_cluster_is_reproduced_without_the_prefix(
    tmp_path,
):
    """Guard the guard: the model must fail the way the cluster did, or the test below proves
    nothing."""
    done = _run(tmp_path, _BUILD_PROBE)

    assert done.returncode != 0
    assert "cannot import name '__version__' from 'detail' (unknown location)" in (
        done.stderr
    )


def test_a_snippet_sent_through_probe_script_finds_detail_whatever_the_login_dir_holds(
    tmp_path,
):
    done = _run(tmp_path, remote_python.probe_script(_BUILD_PROBE))

    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.strip() == "9.9"


def test_a_pythonpath_entry_is_the_environments_choice_even_when_it_is_the_cwd(
    tmp_path,
):
    """Only the start-up entry is dropped. A first version dropped every entry equal to the
    working directory and so removed a `PYTHONPATH` that named it on purpose (found by
    `tests/api/test_remote_build.py`, which runs the real snippet from the checkout with
    `PYTHONPATH` set to it and got the wrong copy of `detail`)."""
    home = tmp_path / "home"
    (home / "detail").mkdir(parents=True)
    (home / "detail" / "__init__.py").write_text(
        '__version__ = "7.7"\n', encoding="utf-8"
    )
    done = subprocess.run(
        [sys.executable, "-S", "-"],
        input=remote_python.probe_script(
            "from detail import __version__\nprint(__version__)\n"
        ),
        capture_output=True,
        text=True,
        cwd=home,
        env={**os.environ, "PYTHONPATH": str(home)},
        timeout=60,
    )

    assert done.returncode == 0, done.stderr[-600:]
    assert done.stdout.strip() == "7.7"


def test_every_snippet_goes_out_through_probe_script(monkeypatch):
    """`remote_python` is the one place Python is run on a cluster
    (`test_remote_version_skew`); what it sends must be the prefixed text."""
    sent = []

    def fake_run_remote(host, command, **kwargs):
        sent.append(kwargs["stdin_text"])
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(remote_python.ssh_transport, "run_remote", fake_run_remote)
    monkeypatch.setattr(remote_python, "activation_prefix", lambda target: "")

    class Target:
        host = "h"
        name = "t"

    remote_python.remote_python(Target(), "print(1)\n", "failed")

    assert sent == [remote_python.probe_script("print(1)\n")]
    assert sent[0].endswith("print(1)\n") and sent[0] != "print(1)\n"
