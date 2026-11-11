"""The machine's ``storage.yaml`` is parsed once, and again only when it changes on disk.

**The class.** A machine-level config file read and parsed on every call, from a lookup that
sits under a per-item path. The cost is the file's size times the number of items, and neither
factor is visible where the lookup is written.

**How it hid.** Each lookup was one small file, and ``preview_cache_dir`` said in its docstring
that it was "read on every call (never cached)" on purpose, so a test that set the environment
variable saw the new value. Three things then compounded:

- A montage tile asks where the preview cache is twice per picture (``previews.cache_dir``).
- The file grows by one alias for every folder a video is imported from, and test runs had
  been registering their temporary folders in the developer's real file, so each parse
  was slow and every tile picture paid for two of them.
- **Every test, and every probe used to investigate, set ``DETAIL_PREVIEW_DIR``**, which
  returns before the file is read. The slow path was the one a user is on and the one nothing
  exercised, so the measurements were fast while the user watched a slow view.

**The rule.** ``locations.read_storage`` is the one reader. It costs a ``stat``; the parse
happens once per ``(mtime, size)``. Every lookup in ``locations``, ``roots`` and
``video_roots`` goes through it, and nothing else in the package parses the file. The one
exception is the writer's read-modify-write (``roots.register_root``), which reads the disk
directly under its lock: merging into a cached copy that missed another process's write would
write that process's alias away.

What is asserted here is a **count of parses**, never a time: a timing assertion on a machine
that also runs the suite fails for reasons that have nothing to do with the code.

A new public function in any of the three modules fails ``test_every_public_function_is_
accounted_for`` until it is listed as a reader (and so swept) or as something else, with why.
"""

import ast
import inspect
import os
import pathlib

import pytest
import yaml

from detail.storage import locations
from detail.storage import previews
from detail.storage import read_cache
from detail.storage import roots
from detail.storage import video_roots

REPO = pathlib.Path(__file__).resolve().parents[2]

#: How many times each reader is called. Any number above one tells a cached reader from an
#: uncached one; this is a tile grid's worth.
CALLS = 60


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine with a ``storage.yaml`` that names every setting and a few aliases.

    The preview-cache environment variable is **removed**: with it set the file is never read,
    which is how every earlier test and probe missed this.
    """
    config = tmp_path / "cfg"
    config.mkdir()
    monkeypatch.setattr(locations, "CONFIG_DIR", config)
    monkeypatch.setattr(locations, "STORAGE_PATH", config / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "DetailExperiments")
    monkeypatch.delenv(locations.PREVIEW_DIR_ENV, raising=False)
    videos = tmp_path / "videos"
    exports = tmp_path / "exports"
    (videos / "session_01").mkdir(parents=True)
    exports.mkdir()
    _write_externally(
        {
            "storage": {
                "default": {
                    "root": (tmp_path / "workspace").as_posix(),
                    "model_cache": (tmp_path / "models").as_posix(),
                    "preview_cache": (tmp_path / "previews").as_posix(),
                }
            },
            "video_roots": {"videos": videos.as_posix()},
            "export_roots": {"exports": exports.as_posix()},
        }
    )
    return tmp_path


@pytest.fixture
def parses(monkeypatch):
    """Every parse of the file, counted where the package does it."""
    seen = []
    real = locations._parse_storage

    def counted():
        seen.append(1)
        return real()

    monkeypatch.setattr(locations, "_parse_storage", counted)
    return seen


def _write_externally(data: dict) -> None:
    """Replace the file the way another process would: no cache of ours is told."""
    path = locations.STORAGE_PATH
    tmp = path.with_name(path.name + ".other-process.tmp")
    tmp.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# every reader, by name
# --------------------------------------------------------------------------- #

#: Each public reader of the file, with a call that reaches the file on the fixture machine.
READERS = {
    "locations.read_storage": lambda tmp: locations.read_storage(),
    "locations.experiments_root": lambda tmp: locations.experiments_root(),
    "locations.list_experiment_dirs": lambda tmp: locations.list_experiment_dirs(),
    "locations.configured_model_cache": lambda tmp: locations.configured_model_cache(),
    "locations.configured_preview_cache": lambda tmp: (
        locations.configured_preview_cache()
    ),
    "locations.preview_cache_dir": lambda tmp: locations.preview_cache_dir(),
    "roots.list_roots": lambda tmp: roots.list_roots("export_roots"),
    "roots.root": lambda tmp: roots.root("export_roots", "exports"),
    "roots.resolve_ref": lambda tmp: roots.resolve_ref(
        "export_roots", {"root": "exports", "rel": "run_01"}
    ),
    # A path inside a registered root: it finds the alias and registers nothing.
    "roots.resolve_or_register": lambda tmp: roots.resolve_or_register(
        "export_roots", tmp / "exports" / "run_01"
    ),
    "video_roots.list_video_roots": lambda tmp: video_roots.list_video_roots(),
    "video_roots.video_root": lambda tmp: video_roots.video_root("videos"),
    "video_roots.resolve_video_ref": lambda tmp: video_roots.resolve_video_ref(
        {"root": "videos", "rel": "session_01/a.mp4"}
    ),
    "video_roots.resolve_or_register_root": lambda tmp: (
        video_roots.resolve_or_register_root(tmp / "videos" / "session_01" / "a.mp4")
    ),
}

#: Public functions that are not swept above, and why. An entry is a claim.
NOT_READERS = {
    "locations.read_storage_fresh": (
        "the uncached read for the writer's read-modify-write; who may call it is "
        "test_nothing_else_reads_the_file"
    ),
    "roots.register_root": "the writer; test_a_registration_* cover it",
    "video_roots.register_video_root": "the writer, by its video name",
}


def _public_functions() -> set[str]:
    found = set()
    for module in (locations, roots, video_roots):
        short = module.__name__.rsplit(".", 1)[1]
        for name, value in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(value):
                continue
            if value.__module__ == module.__name__:
                found.add(f"{short}.{name}")
    return found


def test_every_public_function_is_accounted_for():
    assert _public_functions() == set(READERS) | set(NOT_READERS)
    assert not set(READERS) & set(NOT_READERS)


@pytest.mark.parametrize("name", sorted(READERS))
def test_a_reader_parses_an_unchanged_file_once(name, machine, parses):
    results = [READERS[name](machine) for _ in range(CALLS)]
    # Exactly one, not "at most": zero would mean this call never reached the file and the
    # test proved nothing about it.
    assert len(parses) == 1, f"{name}: {len(parses)} parses in {CALLS} calls"
    assert all(result == results[0] for result in results)


def test_all_the_readers_together_share_one_parse(machine, parses):
    for _ in range(5):
        for call in READERS.values():
            call(machine)
    assert len(parses) == 1


def test_the_tile_path_asks_for_the_cache_location_without_a_parse(machine, parses):
    """What a montage tile calls: the cache root, and a proxy's folder under it."""
    expected = machine / "previews"
    for _ in range(CALLS):
        assert previews.cache_dir() == expected
        assert previews.proxy_dir("1024-abc") == expected / previews.PROXIES_DIR / (
            "1024-abc"
        )
    assert len(parses) == 1


def test_the_parse_count_does_not_grow_with_the_file(machine, parses):
    """The shape that was slow: a file with a couple of thousand aliases."""
    data = locations.read_storage()
    data["video_roots"] = {f"conformance0_{i}": f"/gone/{i}" for i in range(2000)}
    _write_externally(data)
    parses.clear()
    for _ in range(CALLS):
        locations.preview_cache_dir()
        video_roots.video_root("conformance0_1999")
    assert len(parses) == 1


def test_an_uncached_reader_would_be_caught(machine, parses, monkeypatch):
    """The negative case: the reader as it was before, one parse per call."""
    monkeypatch.setattr(locations, "read_storage", locations.read_storage_fresh)
    for _ in range(CALLS):
        locations.preview_cache_dir()
    assert len(parses) == CALLS


# --------------------------------------------------------------------------- #
# a cache that is never stale
# --------------------------------------------------------------------------- #


def test_a_rewrite_by_another_process_is_seen_by_the_next_read(machine, parses):
    assert locations.preview_cache_dir() == machine / "previews"
    data = locations.read_storage()
    data["storage"]["default"]["preview_cache"] = (machine / "elsewhere").as_posix()
    data["video_roots"]["second"] = (machine / "videos" / "session_01").as_posix()
    _write_externally(data)
    assert locations.preview_cache_dir() == machine / "elsewhere"
    assert video_roots.video_root("second") == machine / "videos" / "session_01"
    assert len(parses) == 2  # the first read, then one for the new file


def test_a_registration_is_seen_by_the_next_read(machine):
    assert roots.root("video_roots", "added") is None
    folder = machine / "videos" / "session_01"
    roots.register_root("video_roots", "added", folder)
    assert roots.root("video_roots", "added") == folder.resolve()
    assert roots.resolve_or_register("video_roots", folder / "a.mp4") == "added"


def test_a_registration_is_seen_inside_a_pin(machine):
    """A pinned read skips the ``stat``, so the writer has to drop the entry itself."""
    one = machine / "one"
    two = machine / "two"
    one.mkdir()
    two.mkdir()
    with read_cache.pinned():
        roots.register_root("video_roots", "moved", one)
        assert roots.root("video_roots", "moved") == one.resolve()
        # Same alias, a path of the same length: the file's size does not change.
        roots.register_root("video_roots", "moved", two)
        assert roots.root("video_roots", "moved") == two.resolve()


def test_a_registration_merges_into_the_disk_not_the_cache(machine):
    """Another process's alias survives our write even when our cache could not see it.

    The cache's signature is ``(mtime, size)``. A rewrite of the same size inside one tick of
    the filesystem clock is its blind spot, built here by restoring the timestamp. Merging a
    new alias into the stale copy would drop the other process's change.
    """
    path = locations.STORAGE_PATH
    one = machine / "one"
    two = machine / "two"
    one.mkdir()
    two.mkdir()
    roots.register_root("video_roots", "theirs", one)
    assert roots.root("video_roots", "theirs") == one.resolve()  # warm the cache
    before = path.stat()
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    data["video_roots"]["theirs"] = two.resolve().as_posix()
    _write_externally(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
    # The fixture is in the blind spot: the cached reader still has the old value.
    assert roots.root("video_roots", "theirs") == one.resolve()

    roots.register_root("video_roots", "ours", one)

    assert roots.root("video_roots", "theirs") == two.resolve()
    assert roots.root("video_roots", "ours") == one.resolve()


def test_a_caller_cannot_edit_what_the_next_caller_reads(machine):
    first = locations.read_storage()
    first["video_roots"]["planted"] = "/nowhere"
    first["storage"]["default"]["preview_cache"] = "/nowhere"
    assert "planted" not in locations.read_storage()["video_roots"]
    assert locations.preview_cache_dir() == machine / "previews"


def test_a_missing_file_gives_the_defaults_and_is_not_remembered(machine):
    locations.STORAGE_PATH.unlink()
    assert locations.read_storage() == {}
    assert locations.read_storage_fresh() == {}
    assert locations.configured_preview_cache() is None
    assert roots.list_roots("video_roots") == {}
    _write_externally({"video_roots": {"back": (machine / "videos").as_posix()}})
    assert roots.root("video_roots", "back") == machine / "videos"


# --------------------------------------------------------------------------- #
# nothing else parses the file
# --------------------------------------------------------------------------- #

_READ_CALLS = frozenset(
    {"safe_load", "load", "full_load", "unsafe_load", "read_text", "read_bytes", "read"}
)
_UNCACHED = frozenset({"read_storage_fresh", "_parse_storage"})

#: The only places that may read the file from disk or reach the uncached read, and why.
ALLOWED = {
    ("detail/storage/locations.py", "_parse_storage"): "the one parse",
    ("detail/storage/locations.py", "read_storage"): "hands the parse to the cache",
    ("detail/storage/locations.py", "read_storage_fresh"): "the uncached read itself",
    ("detail/storage/roots.py", "register_root"): (
        "read-modify-write under the file lock: it must merge into what is on disk"
    ),
}


def _names_the_file(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "STORAGE_PATH"
    if isinstance(node, ast.Attribute):
        return node.attr == "STORAGE_PATH"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value.strip()
        return text.endswith("storage.yaml") and " " not in text
    return False


def _called(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else None


def _mentions(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.ImportFrom):
        return {alias.name for alias in node.names}
    return set()


def _own_nodes(scope: ast.AST):
    """The nodes of one function (or the module), without the functions nested in it."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(node))


def _file_readers(source: str) -> set[str]:
    """Functions that read ``storage.yaml`` from disk or reach the uncached read.

    A function is one when it names the file (the constant, or a path ending in
    ``storage.yaml``) and also calls something that reads, or when it mentions the uncached
    reader at all. Module-level code counts as ``<module>``.
    """
    tree = ast.parse(source)
    scopes = [("<module>", tree)] + [
        (node.name, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    found = set()
    for name, scope in scopes:
        nodes = list(_own_nodes(scope))
        if isinstance(scope, ast.Module):
            nodes = [
                node
                for node in nodes
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        names_file = any(_names_the_file(node) for node in nodes)
        reads = any(_called(node) in _READ_CALLS for node in nodes)
        uncached = any(_mentions(node) & _UNCACHED for node in nodes)
        if (names_file and reads) or uncached:
            found.add(name)
    return found


def test_nothing_else_reads_the_file():
    found = {}
    for path in sorted((REPO / "detail").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        for name in _file_readers(path.read_text(encoding="utf-8-sig")):
            found[(rel, name)] = True
    assert set(found) == set(ALLOWED), (
        "storage.yaml is read through locations.read_storage(), which parses it once per "
        "change on disk. A second parser, or a new caller of the uncached read, is the "
        f"per-call parse this file guards against: {sorted(set(found) ^ set(ALLOWED))}"
    )


@pytest.mark.parametrize(
    "source",
    [
        # the reader as it was
        "def _read():\n"
        "    if not STORAGE_PATH.is_file():\n"
        "        return {}\n"
        "    return yaml.safe_load(atomic.read_text(STORAGE_PATH)) or {}\n",
        # through the module, onto a local name first
        "def aliases():\n"
        "    path = locations.STORAGE_PATH\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        return yaml.safe_load(handle)\n",
        # the path spelled out rather than the constant
        "def aliases():\n"
        "    text = (locations.CONFIG_DIR / 'storage.yaml').read_text()\n"
        "    return parse(text)\n",
        "def aliases():\n    return json.loads(STORAGE_PATH.read_bytes())\n",
        "def aliases():\n    return open(f'{CONFIG_DIR}/storage.yaml').read()\n",
        # a reader that reaches for the uncached read
        "def video_root(alias):\n"
        "    return locations.read_storage_fresh()['video_roots'].get(alias)\n",
        "from detail.storage.locations import _parse_storage as parse\n",
        "def video_root(alias):\n    return _parse_storage()['video_roots'][alias]\n",
    ],
)
def test_the_scan_would_catch_a_second_parser(source):
    assert _file_readers(source)


@pytest.mark.parametrize(
    "source",
    [
        "def video_root(alias):\n"
        "    return locations.read_storage().get('video_roots', {}).get(alias)\n",
        # the writer: names the file, dumps, never reads
        "def _write(data):\n"
        "    path = locations.STORAGE_PATH\n"
        "    with open(tmp, 'w', encoding='utf-8') as handle:\n"
        "        yaml.dump(data, handle, sort_keys=False)\n"
        "    atomic.replace(tmp, path)\n",
        # a message that mentions the file, beside an unrelated read
        "def explain(path):\n"
        "    text = path.read_text()\n"
        "    raise ValueError(f'{text} is not registered in storage.yaml on this machine')\n",
        # a nested function's read belongs to the nested function, not its parent
        "STORAGE_PATH = CONFIG_DIR / 'storage.yaml'\n"
        "def other(path):\n"
        "    return path.read_text()\n",
    ],
)
def test_the_scan_passes_what_is_not_a_parse(source):
    assert not _file_readers(source)
