"""Source-scanning drift guards for the model-cache contract (§5).

The bug these exist for is invisible to behavioural tests: ``load()`` steered HF's
cache with an ``os.environ`` write that huggingface_hub had already snapshotted at
import, so the weights silently landed in a different cache than ``is_installed()``
probed. Nothing was wrong at the line where it was written, every test passed, and
the guard stayed dead for months — until a transient Hub 504 failed half a batch
of recordings. Same idiom as ``tests/storage/test_path_serialization.py``: scan the
source for the shape, because no value assertion can see it.

Pure source parsing — these run in CI with no backends installed.
"""

import ast
import re
from pathlib import Path

import detail

PACKAGE = Path(detail.__file__).parent
CACHED_CALLS = {"from_pretrained", "snapshot_download", "hf_hub_download"}


def _python_files():
    return sorted(p for p in PACKAGE.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE.parent).as_posix()


def test_hf_loader_calls_pass_cache_dir_explicitly():
    """``model_cache_dir()`` is the single owner of the weight location, and every
    call site passes it explicitly — an env var provably cannot (huggingface_hub
    binds HF_HUB_CACHE at import, and readiness() imports it before load() runs).

    AST, not a line regex: black at 88 cols wraps these calls across lines, so a
    per-line pattern would silently stop matching and pass by accident forever.
    """
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name not in CACHED_CALLS:
                continue
            if not any(kw.arg == "cache_dir" for kw in node.keywords):
                offenders.append(f"{_rel(path)}:{node.lineno} {name}(...)")
    assert not offenders, (
        "These calls resolve the HF cache implicitly, so they read/write whatever "
        "location huggingface_hub bound at import — not model_cache_dir(). Pass "
        "cache_dir=str(model_cache_dir()) explicitly:\n  " + "\n  ".join(offenders)
    )


def test_no_hf_cache_env_writes():
    """Writing HF_HUB_CACHE / HF_HOME / TRANSFORMERS_CACHE is always a no-op here:
    huggingface_hub snapshots them into module constants at import time, and
    readiness() (via _require_hf) has already imported it before load() runs. There
    is no ordering that makes this work — the fix is an explicit cache_dir=."""
    pattern = re.compile(
        r"""os\.environ(?:\.setdefault\(|\[)\s*["'](HF_|HUGGINGFACE_|TRANSFORMERS_CACHE)"""
    )
    offenders = [
        f"{_rel(path)}:{i}"
        for path in _python_files()
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "Setting an HF cache env var is a silent no-op (the constant is already "
        "bound). Pass cache_dir= to the call instead:\n  " + "\n  ".join(offenders)
    )


def _imports_ultralytics(node) -> bool:
    if isinstance(node, ast.Import):
        return any(
            a.name == "ultralytics" or a.name.startswith("ultralytics.")
            for a in node.names
        )
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        return mod == "ultralytics" or mod.startswith("ultralytics.")
    return False


def test_every_ultralytics_import_is_muzzled_first():
    """ultralytics resolves ONLINE with a live DNS probe at import and snapshots it
    into an Events singleton that POSTs to Google Analytics on every predict/train,
    plus a PyPI check on train. All of it is decided at *import*, so YOLO_OFFLINE
    must be set before the first one — SETTINGS["sync"]=False afterwards is a no-op.

    Checked per *function*, not per file: every lazy import site is a candidate for
    being the process's first, so each must call import_ultralytics() ahead of its
    own import. A file-level exemption would wave through a new method that forgot —
    which is precisely the drift this guards.
    """
    seam = PACKAGE / "models" / "_ultralytics.py"
    offenders = []
    for path in _python_files():
        if path == seam:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            imports = [n.lineno for n in ast.walk(scope) if _imports_ultralytics(n)]
            if not imports:
                continue
            muzzles = [
                n.lineno
                for n in ast.walk(scope)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "import_ultralytics"
            ]
            first_muzzle = min(muzzles, default=None)
            offenders += [
                f"{_rel(path)}:{line} (in {scope.name})"
                for line in imports
                if first_muzzle is None or first_muzzle > line
            ]
    # A module-level import is a Tenet 13 violation too, but catch it here as well:
    # nothing can muzzle ahead of it.
    for path in _python_files():
        if path == seam:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        offenders += [
            f"{_rel(path)}:{n.lineno} (module level)"
            for n in tree.body
            if _imports_ultralytics(n)
        ]
    assert not offenders, (
        "Call models/_ultralytics.py::import_ultralytics() before importing "
        "ultralytics, in the same function, so YOLO_OFFLINE is set before the "
        "process's first import:\n  " + "\n  ".join(offenders)
    )


# --- the suite's own network reach ------------------------------------------ #
TESTS = Path(__file__).resolve().parent.parent
# Ultralytics treats a `.pt` argument as a release asset to fetch and a `.yaml` as an
# architecture in the installed package. Only the second is a local operation.
_FETCHES_WEIGHTS = re.compile(r"""YOLO\(\s*["'][^"']+\.pt["']""")


def _test_files():
    # This module is excluded from its own scan: its guard-the-guard case has to *contain*
    # the forbidden shape as a sample string in order to prove the pattern still matches it.
    return sorted(p for p in TESTS.rglob("test_*.py") if p != Path(__file__).resolve())


def _rel_test(path: Path) -> str:
    return path.relative_to(TESTS.parent).as_posix()


def test_no_test_downloads_model_weights():
    """A test that fetches a remote asset has no bound on its runtime.

    ``test_trained_yolo`` did exactly this, into a fresh ``tmp_path``, on every full-suite
    run. Its ``try/except -> skip`` handled a download that *failed* and did nothing for one
    that *stalled*, so a slow host parked the run until the 120 s watchdog killed it — which
    reads as a hung suite rather than as a slow test. It builds the checkpoint from the
    bundled ``.yaml`` now, and this scan is what stops the next one being written.
    """
    offenders = [
        f"{_rel_test(p)}:{i}"
        for p in _test_files()
        for i, line in enumerate(p.read_text(encoding="utf-8-sig").splitlines(), 1)
        if _FETCHES_WEIGHTS.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, (
        "a test fetches model weights over the network, so its runtime depends on a remote "
        "host and a stall parks the whole suite. Build the checkpoint locally instead — "
        f'`YOLO("<name>.yaml").save(path)`: {offenders}'
    )


def test_a_url_read_in_a_test_carries_a_timeout():
    """The same class one step out: a fetch that *is* legitimate still needs a bound, or a
    hung remote host is indistinguishable from a hung suite."""
    offenders = []
    for path in _test_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name not in {"urlopen", "urlretrieve"}:
                continue
            if not any(kw.arg == "timeout" for kw in node.keywords):
                offenders.append(f"{_rel_test(path)}:{node.lineno}")
    assert not offenders, f"a URL read in a test must pass timeout=: {offenders}"


def test_the_weight_download_scan_would_catch_one():
    """Guard the guard: a scan that matches nothing passes vacuously forever."""
    assert _FETCHES_WEIGHTS.search('    YOLO("yolo11n-seg.pt")')
    assert _FETCHES_WEIGHTS.search("        model = YOLO('best.pt')")
    # The local forms must stay legal, or the scan forbids the fix it exists to require.
    assert not _FETCHES_WEIGHTS.search('    YOLO("yolo11n-seg.yaml").save(str(path))')
    assert not _FETCHES_WEIGHTS.search("    model = YOLO(weights_path)")
