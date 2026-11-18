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
one is a listing's problem, ``storage/dirscan.read_each``), or a user file whose shape the
opener does not read (a BORIS export may be tab-separated), with what checks it instead. The
counts are exact, so a new bare call in a listed file fails
too, and an entry whose calls are gone fails until it is removed.

**Numbers on the command line (#33 section C).** ``click``'s ``float`` and ``FloatRange``
accept ``nan`` and ``inf``, which then reached ``round(seconds * 1e9)`` as a ``ValueError`` or
``OverflowError`` traceback, or were stored as a cut-off nothing could pass. Every float
option in ``detail/cli*.py`` takes its type from ``detail/cli_numbers.py``;
:func:`float_option_violations` fails on a ``type=float`` or a ``FloatRange`` call anywhere
else in those modules.

**Checkpoints (#33 finding 7).** ``model import`` of a file that is not a YOLO checkpoint (a
text file, an empty or cut-short ``.pt``, a PyTorch file of something else) showed whatever
``YOLO(path)`` raised: a ``TypeError``, ``AttributeError`` or ``RuntimeError`` traceback, or
a missing-``onnxruntime`` error for an ``.onnx``, which with ``onnxruntime`` installed would
have been stored as ``weights/best.pt``. A checkpoint a user picks is read by
``detail/models/_ultralytics.py::read_checkpoint``, which checks the suffix and size first and
turns the three types Ultralytics raised into a ``UserInputError`` naming the file. Every
other ``YOLO(...)`` call is listed in :data:`CHECKPOINT_ALLOWED` with why it may stay.
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
    "api/export_targets.py": (1, f"a stream's canonical CSV, {_RECORD}"),
    "cli.py": (
        1,
        "the stream-preset batch table: headerless, with # comments, so not the opener's "
        "shape; its offset cell goes through core.utils.seconds_to_t_ns",
    ),
    "gui/comparison.py": (1, f"a comparison's tables, {_RECORD}"),
    "importers/behavior.py": (
        1,
        "can_import's one-row header sniff, which answers False on any problem; the "
        "import itself reads through the opener",
    ),
    "importers/boris.py": (
        2,
        "a BORIS export, which BORIS writes tab- or comma-separated: the opener refuses "
        "a tab-separated file as a spreadsheet mistake. Decoding is checked in boris.py",
    ),
    "storage/append_log.py": (2, f"an append-only log, {_RECORD}"),
    "storage/behavior_output.py": (3, f"run tables, {_RECORD}"),
    "storage/location_export.py": (1, f"a location export, {_RECORD}"),
    "storage/query_eval_csv.py": (
        2,
        f"query-eval tables, {_RECORD}; `query-eval --out` is checked up front in cli.py",
    ),
    "storage/recording.py": (1, f"a timestamps CSV, {_RECORD}"),
    "storage/spatial.py": (1, f"keyframes.csv, {_RECORD}"),
    "storage/streams.py": (1, f"a canonical stream table, {_RECORD}"),
    "trainers/ultralytics_yolo.py": (1, "Ultralytics' results.csv"),
}


#: The Ultralytics checkpoint reader, and the other ``YOLO(...)`` calls with why each may stay.
_CHECKPOINT_SEAM = "models/_ultralytics.py"
_LOADS_ITS_OWN = (
    "load() of the weights/best.pt DETAIL copied into a catalog or experiment, after "
    "read_checkpoint accepted the file at import or training wrote it"
)
CHECKPOINT_ALLOWED: dict[str, tuple[int, str]] = {
    "models/trained_yolo.py": (1, _LOADS_ITS_OWN),
    "models/yolo_pose.py": (1, _LOADS_ITS_OWN),
    "trainers/ultralytics_yolo.py": (
        2,
        "a base model named from Ultralytics' own list, and the last.pt a training wrote; "
        "a failure there is the training job's recorded error",
    ),
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
        try:
            source = py.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            # Another test's negative case plants a module in `detail/` for a moment
            # (`tests/test_docs_generated_pages.py`); one gone again holds no call.
            continue
        n = _parser_calls(source, names, module)
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


def test_a_picked_checkpoint_is_read_through_the_one_reader():
    """A checkpoint a user picks is read by ``_ultralytics.read_checkpoint``, which refuses a
    file that is not one by name; a bare ``YOLO(path)`` showed Ultralytics' own exception
    (#33 finding 7)."""
    assert (
        _drift(frozenset({"YOLO"}), "ultralytics", _CHECKPOINT_SEAM, CHECKPOINT_ALLOWED)
        == []
    )


def test_the_scan_would_catch_a_bare_yolo_checkpoint_read():
    source = textwrap.dedent("""
        def introspect(path):
            import_ultralytics()
            from ultralytics import YOLO

            return YOLO(str(path)).names

        def other(path):
            import ultralytics

            return ultralytics.YOLO(
                str(path)
            )
        """)
    assert _parser_calls(source, frozenset({"YOLO"}), "ultralytics") == 2


def test_every_exemption_says_why():
    for rel, (count, why) in {
        **YAML_ALLOWED,
        **CSV_ALLOWED,
        **CHECKPOINT_ALLOWED,
    }.items():
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


#: The one module the CLI's float types live in.
_NUMBER_SEAM = "cli_numbers.py"


def float_option_violations(source: str) -> list[str]:
    """Each ``type=float`` keyword and each ``FloatRange(...)`` call in ``source``, by line.

    Read from the parse tree: ``click.FloatRange(``, an imported bare ``FloatRange(`` and
    ``type=float`` on any call all count; a docstring or comment naming them does not.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "FloatRange":
            found.append(f"line {node.lineno}: FloatRange(...)")
        for kw in node.keywords:
            if (
                kw.arg == "type"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "float"
            ):
                found.append(f"line {node.lineno}: type=float")
    return found


def test_every_cli_float_option_refuses_nan_and_inf():
    problems = []
    for py in sorted(_PACKAGE.glob("cli*.py")):
        if py.name == _NUMBER_SEAM:
            continue
        for hit in float_option_violations(py.read_text(encoding="utf-8-sig")):
            problems.append(f"{py.name} {hit}; use the types in detail/{_NUMBER_SEAM}")
    assert problems == []


def test_the_scan_would_catch_a_plain_float_option():
    source = textwrap.dedent('''
        import click
        from click import FloatRange

        @click.option("--a", type=float)
        @click.option(
            "--b",
            type=click.FloatRange(
                min=0
            ),
        )
        @click.option("--c", type=FloatRange(0, 1))
        def cmd(a, b, c):
            """Never type=float or click.FloatRange( here."""
            return int(a), float(b)
        ''')
    assert len(float_option_violations(source)) == 3
