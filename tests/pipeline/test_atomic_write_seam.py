"""Fail-closed guard: a YAML/JSON record this package rewrites while something else reads it
is written through ``tmp`` + ``core.atomic.replace``, never a plain truncating
``open(path, "w")`` — and every reader of one goes through ``core.atomic.read_text``, never a
bare ``open`` / ``Path.read_text``.

**The class, two mirror-image halves of the same hard rule (CLAUDE.md's atomic-write rule):**

(1) **A write side that never rides the atomic seam at all.**
    ``pipeline/embedding.py::_write_embedding_yaml`` wrote ``embedding.yaml`` with a plain
    truncating ``open(path, "w")`` — no ``tmp`` file, no ``atomic.replace`` — so a GUI
    thread polling "has this embedding run finished" could observe the file **empty or
    half-written**, not merely hit a transient ``PermissionError`` the way an already-atomic
    writer's readers can. ``api/derive_run.py::_write_derived_yaml`` (since removed) wrote
    the same ``embedding.yaml`` for a *derived* run and had the identical bug — a second,
    independent writer of the same file the first fix alone would have missed. The three trackers
    (``anchor_tracker.py``, ``iou_single_tracker.py``, ``iou_tracker.py``) each carried a
    byte-identical copy of the same bug writing ``track.yaml``, one instance per tracker
    rather than one behind a shared write path — the N-parallel-implementations shape this
    project's own class-fix rule warns will drift. All five are fixed by routing through
    ``tmp`` + ``atomic.replace`` (the trackers now share ``trackers/_common.py::write_track_yaml``,
    the same lift ``write_handoff`` already got in that module).

(2) **A raw reader of an atomically-written file.** ``os.replace`` on Windows fails with a
    sharing violation while a reader holds the destination open, and a reader opening it
    mid-replace fails the same way — ``core/atomic.py`` retries both, so both the replace and
    the read must go through it. The readers of ``embedding.yaml`` / ``track.yaml`` in
    ``api/derive_run.py`` (since removed), ``api/run_preview.py`` (twice — the module-level
    accessor and ``EmbeddingPreview``), ``pipeline/kinematics_backfill.py`` (both files it reads),
    ``pipeline/render_preview.py``, ``pipeline/dataset_export.py``, ``analysis/locations.py``
    and ``pipeline/embedding.py`` itself (three ``track.yaml`` reads) now do, and so does
    ``api/exporters.py::read_manifest``, whose ``export.yaml`` was already written atomically.

(3) **The guard's own scope was the hole.** It scanned ``pipeline``, ``api``, ``analysis`` and
    ``trackers`` on the reasoning that ``storage/`` holds the canonical writers — true of the
    writers, and silently assumed of the readers beside them. Ten ``storage/`` readers of files
    their own module writes through ``atomic.replace`` read them with a bare ``open``:
    ``state_yaml.read_state_yaml`` (pose / query / temporal state, #317, seen as a
    ``PermissionError`` *on the read* under full-suite load), ``annotation_runs``,
    ``clusterers``, ``comparisons``, ``datasets``, ``models`` and ``snapshots``, plus
    ``credentials.yaml`` and ``storage.yaml`` — and ``experiment.write_config`` was a plain
    truncating write of ``config.yaml``, which an import rewrites (``next_recording_id``) while
    Home reads it per card. Outside ``storage/``, ``gui/comparison.py`` read
    ``behavior_output.yaml`` raw. The read scan also knew only ``yaml.safe_load``, so
    ``executor/rundir.py::read_envelope`` — which carried a hand-written copy of
    ``retry_on_lock`` rather than the seam — was invisible to it. The scope is the whole
    package now, and ``json.load`` / ``json.loads`` are reads too: a scope is a claim about
    where the class cannot occur, and nothing checked that one.

**How it hid.** The two halves hide differently. A truncating write tears on *every* OS — the
file is empty from the ``open`` until the dump finishes — but the window is microseconds and a
reader has to land in it, so it is rare on an idle box and routine on one running several
agents' suites at once; nothing fails deterministically on any platform. The raw-read half is
Windows-only: on POSIX ``os.replace`` never contends with an open reader, so Linux CI cannot
see it at all.

**Two scans**, both on the AST so a name in a docstring or a monkeypatched target is not a
hit. Each runs per scope — every function, and the module body — so a finding names the
function it is in and ``ALLOWED`` can key on that identity rather than a whole file:

- **raw write**: YAML/JSON (``dump`` / ``safe_dump`` / ``dumps``) written to a file by
  ``with open(<p>, "w")`` or ``with <p>.open("w")``, or by ``<p>.write_text(...)``, where the
  written path is never the *source* of an ``atomic.replace`` in the same scope. Matching the
  source, not merely the presence of a replace, is what keeps a function that commits one file
  atomically from excusing a second, truncating write beside it.
- **raw read**: ``yaml.safe_load`` / ``json.load`` / ``json.loads`` fed from a bare
  ``<p>.read_text(...)`` (never ``atomic.read_text(...)``, the seam itself), from a name that
  holds one, or from a handle — or its ``.read()`` — bound by a read-mode ``open(...)`` /
  ``<p>.open(...)``.

Scoped to the whole of ``detail/``. A site that is not an instance — a file a user hands in, one
shipped inside the package, a one-shot artifact nothing reads back — is an ``ALLOWED`` entry
with its reason, not a directory left out of the scan.
"""

import ast
from pathlib import Path

import detail

_PKG = Path(detail.__file__).parent

#: A raw write or read that is not an instance of this class, keyed ``<file>::<function>`` and
#: each with its reason. Claims, not exemptions: ``test_every_allowlist_entry_still_matches``
#: fails on one whose function no longer trips the scan.
_WRITE_ONCE_ARTIFACT = (
    "writes a one-shot artifact into a fresh output folder for a person to open; nothing in "
    "detail/ reads it back or rewrites it, so there is no concurrent reader to tear"
)
_USER_INPUT = (
    "reads a file a person authored and handed in; no code in detail/ writes it, so there is "
    "no replace for the read to race"
)
_PACKAGED = "reads YAML shipped inside the package; nothing rewrites it at run time"
_READ_BACK_LATER = (
    "reads back an artifact this code wrote in an earlier, finished call; its writer is not "
    "running when it is read, so there is no replace for the read to race"
)
_BEST_EFFORT_POLL = (
    "a best-effort poll that answers None on a transient PermissionError and is asked again "
    "next tick; a retry would stall the poll loop for the update it saves"
)
ALLOWED = {
    "api/export_targets.py::_run_masks": _WRITE_ONCE_ARTIFACT,
    "api/measures.py::export": _WRITE_ONCE_ARTIFACT,
    "analysis/comparison_report.py::_write_manifest": _WRITE_ONCE_ARTIFACT,
    "pipeline/prompt_test.py::save_prompt_test_results": _WRITE_ONCE_ARTIFACT,
    "pipeline/render_preview.py::_save_results": _WRITE_ONCE_ARTIFACT,
    "storage/location_export.py::write_location_tables": _WRITE_ONCE_ARTIFACT,
    "storage/stream_presets.py::export_preset": _WRITE_ONCE_ARTIFACT,
    "storage/heading_definitions.py::export_definition": _WRITE_ONCE_ARTIFACT,
    "tools/quick_segment.py::_save_sampled": _WRITE_ONCE_ARTIFACT,
    "storage/datasets.py::write_data_yaml": (
        "written once while the dataset is built; its one reader is a trainer, and "
        "Experiment.train refuses a dataset whose status is not complete"
    ),
    "trainers/ultralytics_yolo.py::_training_data_yaml": (
        "writes the descriptor ultralytics reads, in the same call, before training starts; "
        "and reads the dataset's data.yaml, whose entry above says why that is not raced"
    ),
    "storage/labels_manifest.py::load_labels_manifest": _USER_INPUT,
    "storage/manifest.py::load_manifest": _USER_INPUT,
    "storage/stream_presets.py::import_preset": _USER_INPUT,
    "storage/heading_definitions.py::import_definition": _USER_INPUT,
    "api/skeletons.py::resolve_skeleton": _USER_INPUT,
    "api/skeletons.py::skeleton_source_name": _USER_INPUT,
    "api/skeletons.py::import_skeleton": _USER_INPUT,
    "api/keypoints.py::import_example_set": _USER_INPUT,
    "configs/baselines.py::load_baselines": _PACKAGED,
    "report/exporter.py::previous_artifacts": _READ_BACK_LATER,
    "tools/quick_segment.py::read_sampled": _READ_BACK_LATER,
    "trackers/_common.py::read_handoff": (
        "read once, at start, by the worker resuming the run — the worker that wrote it has "
        "exited, so there is no replace for the read to race"
    ),
    "pipeline/_detection_worker.py::read_progress_json": _BEST_EFFORT_POLL,
    "pipeline/_embedding_worker.py::read_progress_json": _BEST_EFFORT_POLL,
}

_DUMPS = ("dump", "safe_dump", "dumps")


def _module_paths():
    yield from sorted(_PKG.rglob("*.py"))


def _scopes(tree: ast.AST):
    """``(name, node)`` for the module body and every function, however nested."""
    yield "<module>", tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node


def _own_nodes(scope: ast.AST):
    """Every node in ``scope`` except those inside a nested function, which is its own scope."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(node))


def _same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def _is_attr_call(node: ast.AST, attr: str, receiver: str | None = None) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != attr:
        return False
    if receiver is None:
        return True
    return isinstance(node.func.value, ast.Name) and node.func.value.id == receiver


def _is_parse(node: ast.AST) -> bool:
    """A parse of a record's text: ``yaml.safe_load``, ``json.load`` or ``json.loads``."""
    return (
        _is_attr_call(node, "safe_load", "yaml")
        or _is_attr_call(node, "load", "json")
        or _is_attr_call(node, "loads", "json")
    )


def _is_raw_read_text(node: ast.AST) -> bool:
    """A bare ``<p>.read_text(...)`` — but not ``atomic.read_text(...)``, the seam itself."""
    return _is_attr_call(node, "read_text") and not _is_attr_call(
        node, "read_text", "atomic"
    )


def _open_call(node: ast.AST) -> tuple[ast.expr, str] | None:
    """``(path, mode)`` for ``open(p, mode)`` or ``p.open(mode)``; ``None`` if not an open or
    its mode is not a literal (a computed mode is neither flagged as a read nor a write).
    """
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
        path, positional = node.args[0], node.args[1:2]
    elif isinstance(node.func, ast.Attribute) and node.func.attr == "open":
        path, positional = node.func.value, node.args[0:1]
    else:
        return None
    mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), None)
    if mode is None and positional:
        mode = positional[0]
    if mode is None:
        return path, "r"
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return path, mode.value
    return None


def _is_truncating_write(mode: str) -> bool:
    # Append ("a") never truncates, so it cannot show a reader an empty file — the
    # provenance ``history.jsonl`` appenders are not this class.
    return bool(set(mode) & set("wx"))


def _is_read_mode(mode: str) -> bool:
    return not set(mode) & set("wax+")


def _contains_dump(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in _DUMPS
        for n in ast.walk(node)
    )


def _write_offenders(tree: ast.AST) -> list[tuple[str, int]]:
    """``(scope, line)`` for each YAML/JSON write whose path is never committed by an
    ``atomic.replace`` in the same scope — a plain truncating write of the file itself.
    """
    found = []
    for name, scope in _scopes(tree):
        nodes = list(_own_nodes(scope))
        committed = [
            n.args[0] for n in nodes if _is_attr_call(n, "replace", "atomic") and n.args
        ]
        writes = []
        for node in nodes:
            if isinstance(node, ast.With):
                for item in node.items:
                    opened = _open_call(item.context_expr)
                    if (
                        opened
                        and _is_truncating_write(opened[1])
                        and _contains_dump(node)
                    ):
                        writes.append((opened[0], node.lineno))
            elif _is_attr_call(node, "write_text") and _contains_dump(node):
                writes.append((node.func.value, node.lineno))
        for path, lineno in writes:
            if not any(_same(path, src) for src in committed):
                found.append((name, lineno))
    return sorted(set(found), key=lambda f: f[1])


def _read_offenders(tree: ast.AST) -> list[tuple[str, int]]:
    """``(scope, line)`` for each YAML/JSON parse fed by a raw read."""
    found = []
    for name, scope in _scopes(tree):
        nodes = list(_own_nodes(scope))
        handles = set()
        raw_text = set()
        for node in nodes:
            if isinstance(node, ast.With):
                for item in node.items:
                    opened = _open_call(item.context_expr)
                    if (
                        opened
                        and _is_read_mode(opened[1])
                        and isinstance(item.optional_vars, ast.Name)
                    ):
                        handles.add(item.optional_vars.id)
            elif isinstance(node, ast.Assign) and _is_raw_read_text(node.value):
                raw_text.update(t.id for t in node.targets if isinstance(t, ast.Name))
        for node in nodes:
            if not (_is_parse(node) and node.args):
                continue
            arg = node.args[0]
            if _is_attr_call(arg, "read") and isinstance(arg.func.value, ast.Name):
                arg = arg.func.value
            if (
                _is_raw_read_text(arg)
                or (isinstance(arg, ast.Name) and arg.id in handles | raw_text)
                or _open_call(arg) is not None
            ):
                found.append((name, node.lineno))
    return sorted(set(found), key=lambda f: f[1])


def _offenders() -> list[str]:
    out = []
    for path in _module_paths():
        rel = path.relative_to(_PKG).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for scope, lineno in _read_offenders(tree) + _write_offenders(tree):
            if f"{rel}::{scope}" not in ALLOWED:
                out.append(f"detail/{rel}:{lineno} ({scope})")
    return sorted(set(out))


def test_every_concurrently_read_record_rides_the_atomic_seam():
    offenders = _offenders()
    assert not offenders, (
        "a concurrently-read experiment file must be written through tmp + "
        "core.atomic.replace and read back through core.atomic.read_text — a plain "
        "truncating write lets a reader observe a torn file, and a raw read of an "
        "atomically-replaced file can hit a transient PermissionError on Windows:\n"
        + "\n".join(offenders)
    )


def test_every_allowlist_entry_still_matches():
    # An entry that excuses nothing has gone stale — drop it rather than let it stand unread.
    for key, reason in ALLOWED.items():
        rel, scope = key.split("::")
        path = _PKG / rel
        assert path.is_file(), f"detail/{rel} is gone; drop its entry ({reason})"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        scopes = {s for s, _ in _read_offenders(tree) + _write_offenders(tree)}
        assert scope in scopes, f"{key} no longer trips this scan; drop its entry"


def _writes(*lines: str) -> list[int]:
    return [line for _, line in _write_offenders(ast.parse("\n".join(lines)))]


def _reads(*lines: str) -> list[int]:
    return [line for _, line in _read_offenders(ast.parse("\n".join(lines)))]


def test_the_read_scan_would_catch_a_with_open_read():
    caught = _reads(
        "def read_x(path):",
        "    with open(path, encoding='utf-8-sig') as f:",
        "        return yaml.safe_load(f)",
    )
    assert caught == [3]


def test_the_read_scan_would_catch_a_method_form_open_read():
    caught = _reads(
        "def read_x(path):",
        "    with path.open(encoding='utf-8-sig') as f:",
        "        return yaml.safe_load(f.read())",
    )
    assert caught == [3]


def test_the_read_scan_would_catch_a_bare_read_text():
    assert _reads("yaml.safe_load(path.read_text(encoding='utf-8-sig'))") == [1]


def test_the_read_scan_would_catch_read_text_held_in_a_name():
    caught = _reads(
        "def read_x(path):",
        "    text = path.read_text(encoding='utf-8-sig')",
        "    return yaml.safe_load(text)",
    )
    assert caught == [3]


def test_the_read_scan_skips_the_seam_itself():
    assert _reads("yaml.safe_load(atomic.read_text(path))") == []
    assert _reads("json.loads(atomic.read_text(path))") == []


def test_the_read_scan_would_catch_a_json_load_from_a_handle():
    # rundir.read_envelope's shape before it took the seam: invisible to a yaml-only scan.
    caught = _reads(
        "def read_x(path):",
        "    with open(path, encoding='utf-8-sig') as f:",
        "        return json.load(f)",
    )
    assert caught == [3]


def test_the_read_scan_would_catch_a_json_loads_of_a_bare_read_text():
    assert _reads("json.loads(path.read_text(encoding='utf-8-sig'))") == [1]


def test_the_scan_covers_every_module_in_the_package():
    # The class hid in `storage/` because the scope named four subpackages; a module left out
    # of the scan has to be argued for in ALLOWED, not implied by a directory list.
    scanned = set(_module_paths())
    assert scanned == set(_PKG.rglob("*.py"))
    assert _PKG / "storage" / "state_yaml.py" in scanned


def test_the_write_scan_would_catch_a_plain_truncating_write():
    caught = _writes(
        "def write_x(path, data):",
        "    with open(path, 'w', encoding='utf-8') as f:",
        "        yaml.dump(data, f)",
    )
    assert caught == [2]


def test_the_write_scan_would_catch_a_method_form_open_write():
    caught = _writes(
        "def write_x(path, data):",
        "    with path.open('w', encoding='utf-8') as f:",
        "        json.dump(data, f)",
    )
    assert caught == [2]


def test_the_write_scan_would_catch_a_write_text_dump():
    caught = _writes(
        "def write_x(path, data):",
        "    path.write_text(yaml.dump(data), encoding='utf-8')",
    )
    assert caught == [2]


def test_the_write_scan_would_catch_a_module_level_write():
    caught = _writes(
        "with open(PATH, 'w', encoding='utf-8') as f:",
        "    yaml.dump(DATA, f)",
    )
    assert caught == [1]


def test_the_write_scan_would_catch_a_write_beside_an_unrelated_replace():
    # One file committed atomically must not excuse a second, truncating write beside it.
    caught = _writes(
        "def write_x(path, other, data):",
        "    with open(path, 'w', encoding='utf-8') as f:",
        "        yaml.dump(data, f)",
        "    tmp = other.parent / (other.name + '.tmp')",
        "    with open(tmp, 'w', encoding='utf-8') as f:",
        "        yaml.dump(data, f)",
        "    atomic.replace(tmp, other)",
    )
    assert caught == [2]


def test_the_write_scan_skips_a_tmp_plus_atomic_replace_write():
    caught = _writes(
        "def write_x(path, data):",
        "    tmp = path.parent / (path.name + '.tmp')",
        "    with open(tmp, 'w', encoding='utf-8') as f:",
        "        yaml.dump(data, f)",
        "    atomic.replace(tmp, path)",
    )
    assert caught == []


def test_the_write_scan_skips_an_append():
    # Append never truncates: the provenance appenders cannot show a reader an empty file.
    caught = _writes(
        "def append_x(path, entry):",
        "    with open(path, 'a', encoding='utf-8') as f:",
        "        f.write(json.dumps(entry) + '\\n')",
    )
    assert caught == []


def test_the_write_scan_skips_a_read_mode_open():
    # A read-mode `with open(...)` feeding yaml.safe_load is the read scan's job, not this one.
    caught = _writes(
        "def read_x(path):",
        "    with open(path, encoding='utf-8-sig') as f:",
        "        return yaml.safe_load(f)",
    )
    assert caught == []
