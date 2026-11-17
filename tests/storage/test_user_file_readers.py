"""A file a user hands in is read through one reader per format, never a bare parser call.

**The class.** A user's ordinary mistake — a YAML with a missing bracket, a CSV saved by
Excel in the Windows encoding, a PNG picked instead of a skeleton file — reached them as the
exception the parsing library raised: a ``yaml.parser.ParserError`` traceback, a
``UnicodeDecodeError`` traceback, or ``Error: utf-8`` (#33). Nothing is wrong at the line
that calls ``yaml.safe_load`` or ``csv.DictReader``; the call works on every file a test
fixture writes. Only a malformed file shows it, and each reader had its own copy of the call.

**The rule.** YAML a user hands in or edits goes through
``detail/storage/user_yaml.py`` (``read_user_yaml`` / ``parse_user_yaml``); a CSV a user
hands in goes through ``detail/importers/_csv_common.py::read_user_csv``. Both turn the
mistake into a ``UserInputError`` naming the file and the line, which the CLI root shows as
one ``Error:`` line (``tests/test_cli_user_errors.py``).

**Where it stops.** Every other parser call in ``detail/`` is listed below with the number
of calls and why it may stay: a record DETAIL wrote and reads back for itself (a damaged
one is a listing's problem, ``storage/dirscan.read_each``), or a user file a later slice of
#33 moves onto the opener. The counts are exact, so a new bare call in a listed file fails
too, and an entry whose calls are gone fails until it is removed.
"""

import ast
import pathlib
import textwrap

import detail

_PACKAGE = pathlib.Path(detail.__file__).parent

#: Parser calls, by the function name they end in.
_YAML_CALLS = frozenset({"safe_load", "load", "full_load", "unsafe_load"})
_CSV_CALLS = frozenset({"reader", "DictReader"})

#: The readers themselves.
_YAML_SEAM = "storage/user_yaml.py"
_CSV_SEAM = "importers/_csv_common.py"

_RECORD = "a record DETAIL writes and reads back for itself"
_LATER = "a user file not yet on the reader: a later slice of #33"

#: ``{path under detail/: (calls, why)}`` for ``yaml.safe_load``.
YAML_ALLOWED: dict[str, tuple[int, str]] = {
    "analysis/locations.py": (1, f"track.yaml, {_RECORD}"),
    "api/exporters.py": (1, f"an export's manifest.yaml, {_RECORD}"),
    "api/run_preview.py": (3, f"run records, {_RECORD}"),
    "api/skeletons.py": (2, "a best-effort scan that already skips a damaged file"),
    "configs/baselines.py": (1, "baselines shipped inside the package"),
    "gui/comparison.py": (1, "a best-effort read that already skips a damaged file"),
    "pipeline/dataset_export.py": (1, f"track.yaml, {_RECORD}"),
    "pipeline/embedding.py": (5, f"track.yaml, {_RECORD}"),
    "pipeline/keypoint_dataset.py": (1, f"track.yaml, {_RECORD}"),
    "pipeline/kinematics_backfill.py": (2, f"run records, {_RECORD}"),
    "pipeline/pose_match.py": (1, f"track.yaml, {_RECORD}"),
    "pipeline/render_preview.py": (1, f"track.yaml, {_RECORD}"),
    "pipeline/subject_frames.py": (2, f"track.yaml, {_RECORD}"),
    "storage/analysis_runs.py": (1, f"analysis.yaml, {_RECORD}"),
    "storage/behavior_output.py": (1, f"a run manifest, {_RECORD}"),
    "storage/comparisons.py": (1, f"comparison.yaml, {_RECORD}"),
    "storage/credentials.py": (1, "machine credentials DETAIL writes"),
    "storage/datasets.py": (1, f"dataset.yaml, {_RECORD}"),
    "storage/distill_runs.py": (1, f"distill.yaml, {_RECORD}"),
    "storage/example_sets.py": (3, f"an experiment's example sets, {_RECORD}"),
    "storage/export_runs.py": (1, f"export.yaml, {_RECORD}"),
    "storage/id_allocator.py": (1, f"the id high-water marks, {_RECORD}"),
    "storage/heading_definitions.py": (1, f"the experiment's own store, {_RECORD}"),
    "storage/keypoint_runs.py": (2, f"run records, {_RECORD}"),
    "storage/locations.py": (1, "the machine's storage.yaml, which DETAIL writes"),
    "storage/models.py": (1, f"model.yaml, {_RECORD}"),
    "storage/runs.py": (1, f"run records, {_RECORD}"),
    "storage/skeleton_library.py": (1, f"the machine's skeleton library, {_RECORD}"),
    "storage/snapshots.py": (1, f"snapshot records, {_RECORD}"),
    "storage/state_yaml.py": (1, f"authored-run state, {_RECORD}"),
    "storage/stream_presets.py": (1, f"the experiment's own store, {_RECORD}"),
    "storage/tool_runs.py": (2, f"tool run records, {_RECORD}"),
    "trainers/ultralytics_yolo.py": (3, "files the trainer and Ultralytics write"),
}

#: ``{path under detail/: (calls, why)}`` for ``csv.reader`` / ``csv.DictReader``.
CSV_ALLOWED: dict[str, tuple[int, str]] = {
    "analysis/comparison_report.py": (2, f"a comparison's tables, {_RECORD}"),
    "cli.py": (1, f"the stream-preset batch table, {_LATER}"),
    "gui/comparison.py": (1, f"a comparison's tables, {_RECORD}"),
    "importers/behavior.py": (2, f"a behavior CSV (finding 34), {_LATER}"),
    "importers/boris.py": (2, f"a BORIS export (finding 34), {_LATER}"),
    "storage/append_log.py": (2, f"an append-only log, {_RECORD}"),
    "storage/behavior_output.py": (3, f"run tables, {_RECORD}"),
    "storage/location_export.py": (1, f"a location export, {_RECORD}"),
    "storage/metadata.py": (3, f"metadata CSVs (finding 32), {_LATER}"),
    "storage/query_eval_csv.py": (
        2,
        f"query-eval tables, {_RECORD}; `query-eval --out` is checked up front in cli.py",
    ),
    "storage/recording.py": (1, f"a timestamps CSV, {_RECORD}"),
    "storage/spatial.py": (1, f"keyframes.csv, {_RECORD}"),
    "storage/streams.py": (1, f"a canonical stream table, {_RECORD}"),
    "trainers/ultralytics_yolo.py": (1, "Ultralytics' results.csv"),
}


def _parser_calls(source: str, names: frozenset[str], module: str) -> int:
    """How many calls in ``source`` reach ``<module>.<name>`` or a bare imported ``name``.

    Read from the parse tree, so a call wrapped onto two lines counts and a docstring that
    names the call does not.
    """
    imported = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            imported |= {a.asname or a.name for a in node.names if a.name in names}
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in names
            and isinstance(func.value, ast.Name)
            and func.value.id == module
        ):
            count += 1
        elif isinstance(func, ast.Name) and func.id in imported:
            count += 1
    return count


def _drift(
    names: frozenset[str], module: str, seam: str, allowed: dict[str, tuple[int, str]]
) -> list[str]:
    found = {}
    for py in sorted(_PACKAGE.rglob("*.py")):
        rel = py.relative_to(_PACKAGE).as_posix()
        if rel == seam:
            continue
        n = _parser_calls(py.read_text(encoding="utf-8-sig"), names, module)
        if n:
            found[rel] = n
    problems = []
    for rel, n in found.items():
        if rel not in allowed:
            problems.append(
                f"{rel}: {n} bare {module} call(s); use the reader in {seam}"
            )
        elif allowed[rel][0] != n:
            problems.append(f"{rel}: {n} call(s), the allowlist says {allowed[rel][0]}")
    problems += [
        f"{rel}: listed but has no {module} call; remove the entry"
        for rel in allowed
        if rel not in found
    ]
    return problems


def test_user_yaml_is_read_through_the_one_reader():
    assert _drift(_YAML_CALLS, "yaml", _YAML_SEAM, YAML_ALLOWED) == []


def test_a_user_csv_is_read_through_the_one_opener():
    assert _drift(_CSV_CALLS, "csv", _CSV_SEAM, CSV_ALLOWED) == []


def test_every_exemption_says_why():
    for rel, (count, why) in {**YAML_ALLOWED, **CSV_ALLOWED}.items():
        assert count > 0 and len(why) > 10, rel


def test_the_scan_would_catch_a_bare_safe_load():
    source = textwrap.dedent("""
        import yaml
        from yaml import safe_load as parse

        def a(p):
            return yaml.safe_load(
                p.read_text()
            )

        def b(text):
            return parse(text)

        def c():
            '''Never call yaml.safe_load( here.'''
        """)
    assert _parser_calls(source, _YAML_CALLS, "yaml") == 2


def test_the_scan_would_catch_a_bare_csv_reader():
    source = textwrap.dedent("""
        import csv
        from csv import DictReader

        def a(f):
            return list(csv.DictReader(f)), next(csv.reader(f))

        def b(f):
            return DictReader(f)

        def c(f):
            return csv.writer(f)
        """)
    assert _parser_calls(source, _CSV_CALLS, "csv") == 3
