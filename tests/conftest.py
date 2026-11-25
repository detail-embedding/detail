import gc
import importlib.util
import itertools
import os
import sys

import pytest
import numpy as np
import zarr
import yaml
from pathlib import Path

# --- test tiers ---------------------------------------------------------------------- #
# The full suite is ~5m, which is fine for a merge gate and too slow for the edit loop —
# and longer than the default command budget an agent runs it under, so an un-tiered
# `pytest -q` gets killed part-way and reads as a hang. The gate below the merge button is
# four selections instead: `-m guard`, `-m "unit and not guard"`, `-m "gui and not guard"`
# and `-m slow`. They partition the suite, and `-m "not slow"` is the coarse fallback when
# nothing narrower applies. The edit loop selects by *change* rather than by tier —
# `scripts/select_tests.py` over the table in `tests/impact.py` — which is a different
# question from this one and deliberately not a gate; see CLAUDE.md, "Running tests".
#
# The tiers themselves live in `tests/tiers.py` — one table, derived from the path, never
# written on the test. A marker an author has to remember is a marker that will be missing
# from the file that needed it most; deriving it means a new test file inherits its
# directory's tier the day it is added. This hook is the only thing that applies them, and
# `tests/test_test_tiers.py` is the guard on the table.

_TIERS_MODULE = "detail_test_tiers"


def _load_tiers():
    """Load ``tests/tiers.py`` by path.

    Not ``import tests.tiers``: a *different* ``tests`` package is installed in
    site-packages by one of our dependencies and wins the name, and ``import tiers``
    depends on ``tests/`` being on ``sys.path``, which ``--import-mode=importlib`` does
    not guarantee. The file's location relative to this one is the only fact that is
    always true. ``tests/test_test_tiers.py`` loads it the same way and under the same
    ``sys.modules`` key, so both see one module object.
    """
    if _TIERS_MODULE not in sys.modules:
        path = Path(__file__).parent / "tiers.py"
        spec = importlib.util.spec_from_file_location(_TIERS_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_TIERS_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[_TIERS_MODULE]


_EXPERIMENTS_MODULE = "detail_test_experiments"


def _load_experiments():
    """Load ``tests/experiments.py`` by path, for the same reason ``tiers.py`` is.

    That module holds the bodies of the three experiment fixtures below, so a fixture at a
    wider scope can build the same thing without the setup being written out twice — see its
    docstring. Loaded by path and under a shared ``sys.modules`` key so every consumer sees
    one module object, exactly as ``_load_tiers`` does and for the same collision.
    """
    if _EXPERIMENTS_MODULE not in sys.modules:
        path = Path(__file__).parent / "experiments.py"
        spec = importlib.util.spec_from_file_location(_EXPERIMENTS_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_EXPERIMENTS_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[_EXPERIMENTS_MODULE]


_SOURCE_TREE_MODULE = "detail_test_source_tree"


def _load_source_tree():
    """Load ``tests/source_tree.py`` by path, for the same reason ``tiers.py`` is.

    ``tests/test_source_tree_writes.py`` loads it under the same key, so its negative cases
    read the records of the hook this process installed.
    """
    if _SOURCE_TREE_MODULE not in sys.modules:
        path = Path(__file__).parent / "source_tree.py"
        spec = importlib.util.spec_from_file_location(_SOURCE_TREE_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_SOURCE_TREE_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[_SOURCE_TREE_MODULE]


def pytest_configure(config):
    """Refuse any write into the repository tree from here on, in this process.

    Before collection, so a module that writes at import is caught too. The class and the rule
    are in ``tests/test_source_tree_writes.py``.
    """
    _load_source_tree().install(config.rootpath)


@pytest.fixture(autouse=True)
def _no_source_tree_writes():
    """Fail a test whose write into the repository tree was refused, even if it caught the error.

    Autouse and function-scoped, so it is set up first and torn down last: a write from another
    fixture's teardown is still this test's. A write recorded *before* the test began came from
    collection or from a thread that outlived an earlier test; it is reported here rather than
    dropped.
    """
    source_tree = _load_source_tree()
    earlier = source_tree.take()
    yield
    refused = source_tree.take()
    assert not (earlier or refused), (
        "writes into the repository tree were refused — a test writes under tmp_path "
        f"(tests/test_source_tree_writes.py): during this test {refused}; before it began "
        f"{earlier}"
    )


def pytest_collection_modifyitems(config, items):
    """Attach each test's cost tier and, where it is one, the `guard` marker.

    Exactly one of `unit` / `gui` / `slow` per item, plus `guard` orthogonally — so the
    four selections above cover every test once. A file the table does not cover gets no
    cost marker at all and is therefore selected by *no* stage of the gate; that is the
    failure `tests/test_test_tiers.py::test_every_test_file_resolves_to_exactly_one_cost_tier`
    exists to make loud, since nothing about the run itself would say so.
    """
    tiers = _load_tiers()
    root = config.rootpath
    for item in items:
        try:
            rel = Path(item.path).relative_to(root).as_posix()
        except ValueError:  # a test outside the rootdir is nobody's tier
            continue
        cost = tiers.cost_tier(rel)
        if cost is not None:
            item.add_marker(getattr(pytest.mark, cost))
        if tiers.is_guard(rel):
            item.add_marker(pytest.mark.guard)


#: Machine-local state a test could write, by the env var that relocates it. Each gets a
#: session-wide floor below its per-test isolation (``_machine_state_floor``).
_MACHINE_STATE_ENV = {
    "DETAIL_MODEL_DIR": "model_cache_floor",
    "DETAIL_PREVIEW_DIR": "preview_cache_floor",
    "DETAIL_DOCS_CACHE": "docs_cache_floor",
}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Run the cyclic collector between tests, on this (the UI) thread, once a GUI test is up.

    The ``qapp`` fixture installs ``detail.gui.ui_thread_gc`` — automatic collection off, so a
    background thread can never tear down a widget's Python side under the UI thread. Its timer
    runs only while a test pumps the event loop; this runs the collection that is due at every
    test's end, so the tests after a GUI test (which pump nothing) still collect.
    """
    yield
    if not gc.isenabled() and "detail.gui.ui_thread_gc" in sys.modules:
        sys.modules["detail.gui.ui_thread_gc"].collect_due()


@pytest.fixture(autouse=True, scope="session")
def _machine_state_floor(tmp_path_factory):
    """A **session-wide** throwaway home for every piece of machine-local state, below the
    per-test isolation: the model cache, the video preview cache, the built docs site,
    ``~/.config/detail`` (storage.yaml, credentials), and the experiments root.

    A background thread can outlive its test: a GUI readiness check (``GlobalModelsPage`` →
    ``is_installed``), or a page's preview lookup (``key_for_video``, which writes the
    cache's ``index.json``). When it runs, the per-test monkeypatch has already reverted, so
    without a floor it falls back to the real machine's state. That happened twice: the model
    cache on a network share faulted the process, and a GUI test's tmp video path was found
    in a developer's real ``~/.cache/detail/previews/index.json``. Reverting to this floor
    keeps a stray thread in the basetemp. ``tests/test_machine_state_isolation.py`` is the
    guard: it reverts a test's own isolation and asks where everything resolves.
    """
    previous = {name: os.environ.get(name) for name in _MACHINE_STATE_ENV}
    for name, dirname in _MACHINE_STATE_ENV.items():
        os.environ[name] = str(tmp_path_factory.mktemp(dirname))
    with pytest.MonkeyPatch.context() as mp:
        _point_machine_storage_at(mp, tmp_path_factory.mktemp("machine_config_floor"))
        yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


_ISOLATION_DIRS = itertools.count()


def _isolation_dir(tmp_path_factory, name: str) -> Path:
    """A throwaway directory for one autouse isolation fixture, named from a counter.

    ``mktemp(name)`` numbers the directory *for* you, and to pick the number it lists the
    whole basetemp. Every autouse fixture that takes one therefore makes the basetemp
    bigger *and* every later listing slower — the pair here costs O(tests²), and it slows
    down pytest's own ``tmp_path`` alongside it. Measured directly, per call, as the
    basetemp fills: 1.34 / 2.03 / 3.01 / 5.70 ms at 500 / 2k / 4k / 8k entries, against a
    flat 0.05 ms for a named ``mkdir``.

    End to end, on an otherwise idle machine: ``-m "unit and not guard"`` (1935 tests,
    paired runs) 61.7s / 66.7s against 54.4s / 58.5s, and the full suite 341.5s / 339.9s
    against 289.7s / 286.7s. The counter gives the same guarantee the numbering did — a
    fresh directory per fixture per test — without the listing, and each xdist worker has
    its own basetemp, so the count cannot collide across workers.
    """
    return tmp_path_factory.mktemp(f"{name}_{next(_ISOLATION_DIRS)}", numbered=False)


@pytest.fixture(autouse=True)
def _isolate_model_cache(tmp_path_factory, monkeypatch):
    """Point the machine-global model cache (HF downloads + the imported-model
    catalog) at a fresh throwaway dir per test, so tests never read from or
    pollute the real machine's ``~/.cache/detail``. Reverts to the session
    floor above (never the real cache) so an escaping thread stays isolated."""
    monkeypatch.setenv(
        "DETAIL_MODEL_DIR", str(_isolation_dir(tmp_path_factory, "model_cache"))
    )


@pytest.fixture(autouse=True)
def _isolate_preview_cache(tmp_path_factory, monkeypatch):
    """Point the machine-local video preview cache (§3) at a fresh throwaway dir per test,
    so no test reads a real machine's previews or writes proxies into ``~/.cache/detail``.
    The env var is read on every call, so it reaches this process and a spawned worker.
    """
    monkeypatch.setenv(
        "DETAIL_PREVIEW_DIR", str(_isolation_dir(tmp_path_factory, "preview_cache"))
    )


class _HeldLock:
    """Another process holding the preview cache's build lock (``held_build_lock``)."""

    def __init__(self, process) -> None:
        self._process = process

    def release(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(10)


@pytest.fixture
def held_build_lock():
    """Another **process** holding the preview cache's build lock until ``release()`` (or the
    test ends) — a CLI build running while the GUI wants to build, which a lock taken in a
    thread of this process does not reproduce on every platform."""
    import subprocess

    from detail.executor import local
    from detail.storage import previews as preview_store

    root = preview_store.cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    script = (
        "import sys\n"
        "from detail.core import atomic\n"
        "with atomic.file_lock(sys.argv[1]):\n"
        "    print('held', flush=True)\n"
        "    sys.stdin.read()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root / preview_store._LOCK_FILE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        **local.new_process_group_kwargs(),
    )
    held = _HeldLock(process)
    try:
        assert process.stdout.readline().strip() == "held"
        yield held
    finally:
        held.release()
        if process.poll() is None:
            process.kill()
            process.wait(10)
        process.stdout.close()


@pytest.fixture(autouse=True)
def _isolate_machine_storage(tmp_path_factory, monkeypatch):
    """Point ``~/.config/detail/storage.yaml`` at a throwaway dir per test.

    That file holds the machine-local video-root registry, which video import writes to.
    Isolating it per test was previously each test's own job, done ad-hoc — so any test
    that imported a video could write to the developer's real config, and one that forgot
    failed with a confusing ``storage.yaml.tmp -> storage.yaml`` error instead. Covering it
    once here matches ``_isolate_model_cache`` above: a test never touches real machine
    state, and never has to remember to say so.

    Both halves are needed. The **env var** is what reaches a spawned import worker, which
    re-imports ``locations`` and would otherwise resolve the real path; the **monkeypatch**
    is what redirects this process, where the module was imported long before the fixture
    ran. Credentials live in the same directory and follow the same override.

    **The experiments root moves with it.** With no root in storage.yaml — and the isolated
    one has none — ``locations.experiments_root()`` falls back to ``DEFAULT_ROOT``, the real
    ``~/Documents/DetailExperiments``. That was each test's own job too, and the tool-run case
    of ``tests/storage/test_id_reuse.py`` forgot: every full run left two stub tool runs in a
    developer's real ``tools/`` folder, and the app's Quick tools rail raised ``KeyError:
    'tool_id'`` on them. Only the monkeypatch half applies: nothing spawned resolves the root
    (the tools workspace, ROI exports, experiment listing, creation and import are all
    in-process); a worker that did would need an env var.
    """
    _point_machine_storage_at(
        monkeypatch, _isolation_dir(tmp_path_factory, "machine_config")
    )


def _point_machine_storage_at(monkeypatch, config_dir: Path) -> None:
    """``_isolate_machine_storage``'s body, for a session-scoped builder that imports video: a
    wider-scoped fixture is set up *before* the function-scoped autouse ones, so it is not yet
    isolated and has to say so itself."""
    from detail.storage import credentials, locations

    monkeypatch.setenv("DETAIL_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(locations, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(locations, "STORAGE_PATH", config_dir / "storage.yaml")
    monkeypatch.setattr(credentials, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(
        credentials, "CREDENTIALS_PATH", config_dir / "credentials.yaml"
    )
    monkeypatch.setattr(locations, "DEFAULT_ROOT", config_dir / "DetailExperiments")


# The component registries these cover are plain ``dict[str, type]`` in registries.py
# with pure-insert register functions, so a snapshot/restore is exact. (PROCESSING_MODULES
# / ANALYSIS_MODULES are excluded: their registration wires several maps — registry,
# JOB_RUNNERS, event maps — at once, so restoring one dict alone would desync them.)
_ISOLATED_REGISTRIES = (
    "MODELS",
    "TRAINED_LOADERS",
    "TRAINERS",
    "TRACKERS",
    "EMBEDDING_MODELS",
    "KEYPOINT_MODELS",
    "KEYPOINT_TRACK_METHODS",
    "TRANSFORMS",
    "STREAM_IMPORTERS",
    "TOOLS",
    "QUERY_METHODS",
)


def _prepopulate_registries() -> None:
    """Import the production backend modules so every isolated registry holds its real
    entries *before* the snapshot below. Registration is a side effect of import, and Python
    caches modules — so once a registry entry is stripped, an ``import_*`` call can't re-add
    it. Without this, a test that first-imports a backend and then has it restored away left
    the registry empty for later tests **in any run that didn't collect the pipeline test
    files** (a subset like ``pytest tests/gui``); the full suite hid it because collection
    imports those modules up front. Pre-populating here makes every subset match the full
    suite. Idempotent (cached ``import_module``) and backend-free (Tenet 13), so it is cheap
    to call per test — the first call in a fresh subset is a one-time import, the rest are
    ``sys.modules`` hits."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules
    from detail.importers import import_stream_importers
    from detail.query_methods import import_query_methods
    from detail.tools import import_tools
    from detail.trainers import import_trainers

    # MODELS, TRAINED_LOADERS, TRAINERS, EMBEDDING_MODELS, TRACKERS (via pipeline.tracking),
    # TRANSFORMS (via pipeline.embedding):
    import_runner_modules(PIPELINE_RUNNERS)
    import_stream_importers()  # STREAM_IMPORTERS
    import_tools()  # TOOLS
    import_query_methods()  # QUERY_METHODS
    import_trainers()  # TRAINERS, which nothing else imports on its own


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Restore the component registries after each test so a fake model/tracker/etc.
    registered mid-test can't leak into a later test — an in-body ``register_model(...)``
    used to persist for the whole pytest process, breaking any later test whose behaviour
    depends on registry contents or order (e.g. ``sorted(MODELS)[0]``). The baseline is
    pre-populated with the production entries (``_prepopulate_registries``) so it is the same
    whether or not the run collected the pipeline test files — only a test's *own* in-body
    registration is stripped. Dicts are cleared and repopulated in place (never reassigned),
    so modules that did ``from detail.core.registries import MODELS`` still see the restored
    contents."""
    from detail.core import registries

    _prepopulate_registries()
    saved = {name: dict(getattr(registries, name)) for name in _ISOLATED_REGISTRIES}
    yield
    for name in _ISOLATED_REGISTRIES:
        live = getattr(registries, name)
        live.clear()
        live.update(saved[name])


@pytest.fixture(autouse=True)
def _isolate_pool_worker_state():
    """Restore the GPU pool workers' process-wide state after each test.

    ``pool_initializer`` runs once per worker *process* in production, so it writes module
    globals (``_WORKER_STATE``) and ``CUDA_VISIBLE_DEVICES`` freely. Tests run it — or a
    synchronous fake pool that does its job — inside the pytest process, and what it wrote stayed
    for every later test on that worker: a two-GPU fake left ``share = 2``, and
    ``tests/core/test_cpu.py`` then read a render pool of 3 threads where the rule gives 7, in
    whichever full run put the two files on one worker. Dicts are cleared and refilled in place,
    because the workers read their own module's dict.
    """
    from detail.pipeline import _detection_worker, _embedding_worker

    states = (_detection_worker._WORKER_STATE, _embedding_worker._WORKER_STATE)
    saved = [dict(state) for state in states]
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    yield
    for state, before in zip(states, saved):
        state.clear()
        state.update(before)
    if cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda


def _split_detail_modules() -> set[str]:
    """``detail.*`` modules whose ``sys.modules`` entry is not what their parent package holds."""
    split = set()
    for name, module in list(sys.modules.items()):
        if not name.startswith("detail.") or module is None:
            continue
        parent, _, child = name.rpartition(".")
        if getattr(sys.modules.get(parent), child, module) is not module:
            split.add(name)
    return split


@pytest.fixture(autouse=True)
def _no_split_modules():
    """Fail the test that leaves a ``detail`` module existing twice, rather than a later victim.

    Deleting ``detail.models.*`` from ``sys.modules`` and re-importing rebinds each submodule
    on its parent package; putting the old entries back into ``sys.modules`` does not rebind
    them. From then on a string-target patch (``"detail.models.sam3.is_importable"``) walks
    the package attributes into the new copy while the code under test reads the old one, so
    the patch silently misses — in whichever test the scheduler put next on that worker.
    Two SAM 3 tests failed 12 and 7 of 86 full runs that way, never alone. Only splits that appear
    *during* a test count: a package attribute that legitimately shadows a submodule is
    split before the test too. The split is repaired before failing so the run's remaining
    tests are not victims of it.
    """
    before = _split_detail_modules()
    yield
    new = _split_detail_modules() - before
    if not new:
        return
    for name in new:
        parent, _, child = name.rpartition(".")
        setattr(sys.modules[parent], child, sys.modules[name])
    pytest.fail(
        "this test left two copies of "
        + ", ".join(sorted(new))
        + ": sys.modules holds one and the parent package another, so a later string-target "
        "monkeypatch misses. Re-import in a subprocess instead of deleting from sys.modules.",
        pytrace=False,
    )


# A plausible 40-hex commit sha; it only has to round-trip refs/ -> snapshots/.
FAKE_HF_SHA = "e619358a5fe272554f17d2fb6cc65a679326bac0"


@pytest.fixture
def fake_hf_snapshot():
    """Materialize a repo in the HF cache layout under ``model_cache_dir()``.

    Not a storage mock: it writes a real temp directory in an *external* cache's
    on-disk format and lets the real ``huggingface_hub`` read it, so tests pin the
    library's actual behaviour rather than our belief about it.

    ``refs/<rev>`` + ``snapshots/<sha>/<file>`` is the whole contract both readers
    need — ``try_to_load_from_cache`` resolves the ref to a sha then stats the file,
    and ``snapshot_download(local_files_only=True)`` only checks that the snapshot
    dir exists. No ``blobs/``, no symlinks, no content: on Windows hf_hub moves the
    blob into the snapshot path anyway (``blobs/`` is empty there), and on POSIX the
    snapshot entry is a relative symlink into ``blobs/`` — neither reader follows or
    verifies it here.

    ``_isolate_model_cache`` (autouse, above) already points the cache at a fresh
    tmp dir, so this always lands somewhere throwaway.
    """
    from detail.models import hf_access

    def _make(repo_id: str, files: list[str], sha: str = FAKE_HF_SHA):
        root = hf_access.model_cache_dir() / f"models--{repo_id.replace('/', '--')}"
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(sha, encoding="utf-8")
        snapshot = root / "snapshots" / sha
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in files:
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        return snapshot

    return _make


@pytest.fixture
def experiment_dir(tmp_path) -> Path:
    """Minimal valid experiment directory.

    The body is ``tests/experiments.py::build_experiment_dir`` so a wider-scoped fixture can
    build the same thing without writing the setup out again (Tenet 16 forbids that, and a
    module-scoped fixture cannot request this one).
    """
    return _load_experiments().build_experiment_dir(tmp_path)


@pytest.fixture
def recording_dir(experiment_dir) -> Path:
    """Minimal valid recording directory with recording.yaml and all stage subdirs.

    Body in ``tests/experiments.py::build_recording_dir`` -- see ``experiment_dir``.
    """
    return _load_experiments().build_recording_dir(experiment_dir)


@pytest.fixture
def sample_detection_zarr(tmp_path) -> Path:
    """Detection zarr with N=100 rows and correct schema. Use for storage and pipeline tests."""
    N = 100
    path = tmp_path / "part_001.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["schema_version"] = 1
    root.create_array("frame_idx", data=np.arange(N, dtype=np.int32), chunks=(10000,))
    root.create_array(
        "t_ns", data=np.arange(N, dtype=np.int64) * 33_333_333, chunks=(10000,)
    )
    root.create_array(
        "detection_idx", data=np.zeros(N, dtype=np.int32), chunks=(10000,)
    )
    root.create_array(
        "confidence", data=np.full(N, 0.9, dtype=np.float32), chunks=(10000,)
    )
    root.create_array(
        "bbox", data=np.zeros((N, 4), dtype=np.float32), chunks=(10000, 4)
    )
    # vlen columns: native v3 variable-length UTF-8 (StringDType); create then assign
    # (create_array forbids data= together with dtype=). mask_rle holds the ASCII
    # COCO-RLE counts string — see "Mask encoding" in DESIGN_PLAN Section 9.
    cn = root.create_array("class_name", shape=(N,), chunks=(10000,), dtype="str")
    cn[:] = np.array(["mouse"] * N, dtype=object)
    mr = root.create_array("mask_rle", shape=(N,), chunks=(1000,), dtype="str")
    mr[:] = np.array([""] * N, dtype=object)
    return path


@pytest.fixture
def sample_assignment_zarr(tmp_path) -> Path:
    """Assignment zarr parallel to sample_detection_zarr (N=100 rows)."""
    N = 100
    path = tmp_path / "assignments" / "part_001.zarr"
    path.parent.mkdir(parents=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["schema_version"] = 1
    root.create_array("object_id", data=np.zeros(N, dtype=np.int32), chunks=(10000,))
    return path


@pytest.fixture
def write_vfr_video():
    """Factory: write a **variable-frame-rate** video with pairwise-distinct frames.

    The fixture for frame-index correctness. On a constant-rate file a
    frame-number seek happens to be right, so it hides the desync; here no single
    fps maps frame number to time (the first half is spaced 10ms, the second 30ms),
    which is what a real camera's VFR output does. Asking cv2's
    ``CAP_PROP_POS_FRAMES`` seek for frame 25 of a 60-frame file like this returns
    the picture of frame 33 — and reports 25 back — which is exactly the preview
    overlay desync ``read_frame_at`` exists to prevent.

    Two options reproduce what real containers do to the *start* of a video, which
    is where a timestamp map can disagree with the frames a decoder shows:

    - ``start_ms`` — the first frame's PTS. Cameras and MPEG-TS files rarely start at
      0; the map must anyway (``t_ns[0] == 0``).
    - ``b_frames`` — B-frames per group. With a **negative** ``start_ms`` this makes
      an MP4 **edit list**: the frames before 0 stay in the file as ``is_discard``
      packets (the decoder needs them as references) and no decoder shows them. So
      ``start_ms=-50, b_frames=2`` keeps 60 packets and shows 55 frames, the first
      being frame 5. Without B-frames the muxer simply drops negative-PTS frames,
      which does not exercise the edit list.

    Returns the path; frame ``i`` is recoverable as a distinct BGR colour triple.
    """
    import av
    from fractions import Fraction

    def _frame_bgr(i: int, h: int, w: int) -> np.ndarray:
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :, 0] = (i * 37) % 256
        frame[:, :, 1] = (i * 91 + 20) % 256
        frame[:, :, 2] = (i * 13 + 50) % 256
        return frame

    def _write(
        path,
        n_frames: int = 60,
        width: int = 64,
        height: int = 48,
        *,
        start_ms: int = 0,
        b_frames: int = 0,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=Fraction(50, 1))
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            if b_frames:
                stream.codec_context.options = {"bf": str(b_frames)}
            stream.time_base = Fraction(1, 1000)
            stream.codec_context.time_base = Fraction(1, 1000)
            pts = start_ms
            for i in range(n_frames):
                frame = av.VideoFrame.from_ndarray(
                    _frame_bgr(i, height, width), format="bgr24"
                )
                frame.pts = pts
                frame.time_base = Fraction(1, 1000)
                pts += 10 if i < n_frames // 2 else 30
                container.mux(stream.encode(frame))
            container.mux(stream.encode())
        return path

    _write.frame_bgr = _frame_bgr
    return _write


@pytest.fixture
def make_embedding_run():
    """Factory: write a per-track embedding run (store + embedding.yaml) into a recording,
    and index it in ``recording.yaml`` when the recording has one — as the stage does.

    Hands back ``tests/experiments.py::write_embedding_run`` -- see ``experiment_dir`` for why
    the body lives there. Uses the real ``EmbeddingPartWriter`` (no storage mocks) so the
    schema matches the pipeline.
    """
    return _load_experiments().write_embedding_run


@pytest.fixture
def processed_recording(tmp_path):
    """An experiment with one recording carrying a complete detection **and** tracking run,
    recorded both per-recording and as experiment-level run records.

    The shape the upstream export targets need and no fixture had: `tracks` and `masks` hang off
    a `run` subject, which is `runs/<id>/run.yaml` plus the per-recording output it points at, so
    a fixture that writes only the recording half cannot reach them at all.

    Returns a builder with:

    ``it.experiment``                 the `Experiment`
    ``it.recording``                  the `Recording`
    ``it.detection_run_id``           the experiment-level run id for detection (``det_01``)
    ``it.tracking_run_id``            the experiment-level run id for tracking (``trk_01``)
    ``it.H`` / ``it.W``               mask + video resolution (40x40, so px == norm * 40)
    ``it.rle(mask)``                  a mask as the ASCII COCO-RLE string the zarr stores
    ``it.square(r0, r1, c0, c1)``     a rectangular mask
    ``it.add_recording(id, ...)``     a second recording with its own runs, joined to both records
    """
    import json

    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store
    from detail.storage import runs as runs_store
    from detail.storage.zarr_io import AssignmentPartWriter, DetectionPartWriter

    SIZE = 40  # mask resolution == video resolution, so px == norm * 40 in an assertion
    H = W = SIZE

    def _rle(mask) -> str:
        from pycocotools import mask as mask_utils

        return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))[
            "counts"
        ].decode("ascii")

    def _square(r0, r1, c0, c1):
        mask = np.zeros((H, W), np.uint8)
        mask[r0:r1, c0:c1] = 1
        return mask

    class _Built:
        H = W = SIZE
        detection_run_id = "det_01"
        tracking_run_id = "trk_01"
        rle = staticmethod(_rle)
        square = staticmethod(_square)

        def __init__(self):
            self.experiment = Experiment.create("p", tmp_path)
            self._recordings: list[str] = []
            self.recording = self.add_recording("recording_001")

        def add_recording(self, recording_id, detections=None, object_ids=None):
            """A recording with a 1-part 40x40 video and a complete ``det_01`` / ``track_01``.

            ``detections`` is ``[(frame_idx, detection_idx, bbox, mask), ...]`` and
            ``object_ids`` is parallel — the track each detection row was assigned to.
            """
            if detections is None:
                detections = [(0, 0, (0.1, 0.1, 0.5, 0.5), _square(4, 20, 4, 20))]
                object_ids = [0]
            recording = self.experiment.create_recording(recording_id)
            sdir = recording.dir

            rel = "video/timestamps/generated/part_001.csv"
            (sdir / rel).parent.mkdir(parents=True, exist_ok=True)
            recording_store.write_timestamps_csv(
                sdir / rel, np.arange(5, dtype=np.int64) * 1000
            )
            data = recording_store.read_recording(sdir)
            data["video"] = {
                "fps_actual": 30.0,
                "resolution": [W, H],
                "files": [
                    {
                        "index": 0,
                        "file": "video/part_001.mp4",
                        "timestamps": {
                            "generated": rel,
                            "user": None,
                            "active": "generated",
                        },
                    }
                ],
            }
            # keypoint_detect hangs off a detection run (§26), so the recording lists it.
            data["detection_runs"] = [{"id": "det_01", "status": "complete"}]
            data["tracking_runs"] = [
                {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
            ]
            recording_store.write_recording(sdir, data)

            det_dir = sdir / "detections" / "det_01"
            det_dir.mkdir(parents=True)
            (det_dir / "job.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "mask_resolution": [H, W],
                        "job": {"downsampling_run_id": None},
                    }
                ),
                encoding="utf-8",
            )
            with DetectionPartWriter(det_dir / "part_001.zarr") as writer:
                for frame, index, bbox, mask in detections:
                    writer.add_row(
                        frame, frame * 1000, index, "mouse", 0.9, bbox, _rle(mask)
                    )

            trk_dir = sdir / "tracking" / "track_01"
            (trk_dir / "assignments").mkdir(parents=True)
            (trk_dir / "track.yaml").write_text(
                yaml.dump({"detection_run_id": "det_01"}), encoding="utf-8"
            )
            with AssignmentPartWriter(
                trk_dir / "assignments" / "part_001.zarr"
            ) as writer:
                for object_id in object_ids:
                    writer.add_object_id(object_id)

            self._recordings.append(recording_id)
            self._write_records()
            return recording

        def _write_records(self):
            for run_id, module_id, stage_id in (
                (self.detection_run_id, "detection", "det_01"),
                (self.tracking_run_id, "tracking", "track_01"),
            ):
                runs_store.write_run(
                    self.experiment.dir,
                    run_id,
                    {
                        "id": run_id,
                        "module_id": module_id,
                        "recording_ids": list(self._recordings),
                        "per_recording": {
                            sid: {"run_id": stage_id, "status": "complete"}
                            for sid in self._recordings
                        },
                        "status": "complete",
                    },
                )

    return _Built()


class _CohortTopology:
    """What an authored run is on disk: module, method, state class and storage module.

    A seam because ``authored_cohort`` was written once against both clustering modules; pose
    clustering's half went with the module (``docs/removed_capabilities.md``), and a restore puts
    it back here.
    """

    def __init__(self, name):
        if name == "temporal":
            from detail.api.temporal_clustering import TemporalClusteringRun
            from detail.storage import temporal_runs

            self.module_id = "temporal_clustering"
            self.method = "dtw_finch"
            self.storage = temporal_runs
            self.state_class = temporal_runs.TemporalClusteringState
            self.read_state_name = "read_temporal_state"
            self.write_state_name = "write_temporal_state"
            self.state_path = temporal_runs.temporal_yaml_path
            self.run_class = TemporalClusteringRun
        else:
            raise ValueError(f"no cohort topology {name!r}")
        self.name = name


#: The opening granularity of an ``authored_cohort`` run, and the one ``pin_then_change`` moves
#: the live run to. Both are cuts the fixture's merge tree really has (it has 22 leaves).
_COHORT_GRANULARITY = 4
_COHORT_CHANGED_GRANULARITY = 9


@pytest.fixture(scope="session")
def _temporal_cohort_fit(tmp_path_factory):
    """The ``fit/`` folder of ``authored_cohort``'s runs: one tiny real ``dtw_finch`` fit.

    Built once per session and copied into each run, because ``authored_cohort`` writes a fit per
    test and again per ``add_run``, and a bout fit costs a numba warm-up the frame-level FINCH
    shard it replaced did not. 48 bouts of 8 kinds give a merge tree of 22 leaves — deep enough
    that the root's children are internal nodes and that granularities 4 and 9 are both real cuts.
    """
    from detail.cluster_methods import import_temporal_cluster_methods
    from detail.core.registries import TEMPORAL_CLUSTER_METHODS
    from detail.storage import temporal_runs

    import_temporal_cluster_methods()
    method = TEMPORAL_CLUSTER_METHODS["dtw_finch"]()
    features = _bouty_trajectory(n_bouts=48, bout_rows=20, kinds=8)
    t_ns = np.arange(len(features), dtype=np.int64) * _POSE_FIT_STEP
    fit = method.fit(
        [("r1", 0, features, t_ns)],
        method.config_model(**_TEMPORAL_METHOD_PARAMS).model_dump(),
    )
    run_dir = tmp_path_factory.mktemp("temporal_cohort_fit")
    temporal_runs.fit_dir(run_dir).mkdir(parents=True, exist_ok=True)
    method.serialize(fit, temporal_runs.fit_path(run_dir, method.shard_suffix))
    return temporal_runs.fit_dir(run_dir)


@pytest.fixture
def authored_cohort(tmp_path, _temporal_cohort_fit):
    """An experiment holding one authored temporal-clustering run with a real §19 artifact.

    The cheap half of an authored run: `analysis.yaml`, `temporal_clustering.yaml`, the three
    artifact files, a frame view and a tiny real ``dtw_finch`` fit — everything a snapshot copies,
    a comparison measures and a snapshot view reads, without a run through the executor. The fit
    is tiny but real, because naming resolves a cluster to a node of the fit's merge tree and a
    run with no fit cannot answer that. Tests that need a realistic vocabulary (the scatter, the
    cluster table, a real reprojection) use `tests/gui/conftest.py::fitted`; tests about the
    *state* around it use this.

    Returns a builder with:

    ``cohort.experiment``            the `Experiment`
    ``cohort.add_run(run_id)``       a second authored run in the same experiment
    ``cohort.name({id: name})``      set names on disk, no reprojection
    ``cohort.run(run_id)``           the authored run handle (a ``cohort.run_class``)
    ``cohort.read_state(run_id)`` / ``cohort.write_state(state, run_id)``
                                     the run's state file, so no test names a storage module
    ``cohort.state_path(run_id)`` / ``cohort.history_path(run_id)``
    ``cohort.topology``              module id, method, state class, storage module
    """
    return _build_authored_cohort(tmp_path, "temporal", _temporal_cohort_fit)


def _build_authored_cohort(tmp_path, topology_name, temporal_fit_dir):
    import numpy as np

    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs
    from detail.storage import behavior_output as bo
    from detail.storage import frame_view as fv
    from detail.storage import metadata as md

    topology = _CohortTopology(topology_name)
    storage = topology.storage

    n_frames, period = 100, 10_000_000
    axis = (np.arange(n_frames) * period).astype(np.int64)
    recordings = ("r1", "r2")

    class _Cohort:
        def __init__(self):
            self.topology = topology
            self.run_class = topology.run_class
            self.module_id = topology.module_id
            self.experiment = Experiment.create("cohort", tmp_path)
            for recording in recordings:
                self.experiment.create_recording(recording)
                md.upsert_row(self.experiment.dir, recording, {"animal": "A"})
            self.add_run("an_01")

        def run_dir(self, run_id="an_01"):
            return analysis_runs.analysis_dir(self.experiment.dir, run_id)

        def run(self, run_id="an_01"):
            return self.experiment.authored_run(run_id)

        def read_state(self, run_id="an_01"):
            return getattr(storage, topology.read_state_name)(self.run_dir(run_id))

        def write_state(self, state, run_id="an_01"):
            return getattr(storage, topology.write_state_name)(
                self.run_dir(run_id), state
            )

        def state_path(self, run_id="an_01"):
            return topology.state_path(self.run_dir(run_id))

        def history_path(self, run_id="an_01"):
            return self.run_dir(run_id) / storage.HISTORY_FILENAME

        def pin_then_change(self, run_id="an_01"):
            """Pin a snapshot, then move the live run so the two **differ**.

            The shape every snapshot test needs and none had. With the pinned state equal to the
            live state, an assertion about "the frozen value" compares a value to itself: the whole
            feature could be removed — both mixin overrides replaced with ``super()`` — and the GUI
            suites stayed green. A fixture that never diverges cannot tell a working freeze from a
            broken one.

            Returns ``(snapshot_id, pinned, live)`` where each value is ``(granularity, names)``.
            """
            self.name({"c01": "rearing"}, run_id)
            pinned_state = self.read_state(run_id)
            pinned = (pinned_state.granularity, dict(pinned_state.names))
            snapshot_id = self.run(run_id).pin(reason="finish")
            assert snapshot_id, "nothing to pin"

            state = self.read_state(run_id)
            state.granularity = _COHORT_CHANGED_GRANULARITY
            state.resolved_names = {"c01": "grooming", "c02": "still"}
            self.write_state(state, run_id)
            live = (
                _COHORT_CHANGED_GRANULARITY,
                {"c01": "grooming", "c02": "still"},
            )
            assert pinned != live, "the fixture must actually diverge"
            return snapshot_id, pinned, live

        def judge(self, instance_ids, verdict=True, run_id="an_01"):
            """Record audit decisions on the **live** run, without going through the handle."""
            from detail.storage import behavior_audit as audit

            for instance_id in instance_ids:
                audit.append_audit(
                    self.run_dir(run_id), instance_id, human_label=verdict, decided=True
                )

        def name(self, names, run_id="an_01"):
            state = self.read_state(run_id)
            state.names.update({str(k): str(v) for k, v in names.items()})
            self.write_state(state, run_id)

        def add_fit(self, run_id):
            """A real, tiny fit, so naming can resolve a cluster to a tree node: a copy of the
            session's one ``dtw_finch`` fit."""
            import shutil

            shutil.copytree(temporal_fit_dir, storage.fit_dir(self.run_dir(run_id)))

        def add_run(self, run_id, *, granularity=_COHORT_GRANULARITY):
            run_dir = self.run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            analysis_runs.write_analysis_yaml(
                run_dir,
                {"id": run_id, "module_id": topology.module_id, "status": "open"},
            )
            self.add_fit(run_id)
            self.write_state(
                topology.state_class(
                    embed_run_id="emb_01",
                    track_id=0,
                    method=topology.method,
                    source_recording_ids=list(recordings),
                    granularity=granularity,
                ),
                run_id,
            )
            instances, coverage = [], []
            for recording in recordings:
                for i in range(3):
                    instances.append(
                        bo.BehaviorInstance(
                            recording,
                            0,
                            int(axis[i * 3]),
                            int(axis[i * 3 + 1]),
                            i * 3,
                            i * 3 + 1,
                            "c01",
                            "rearing",
                        )
                    )
                coverage.append(
                    bo.CoverageRow(
                        recording, 0, 0, n_frames * period, n_frames * period, n_frames
                    )
                )
            bo.write_behavior_output(
                run_dir,
                instances=instances,
                coverage=coverage,
                manifest={
                    "producer": {"module_id": topology.module_id, "run_id": run_id},
                    "origin": "predicted",
                    "label_space": "pooled",
                    "vocabulary": ["rearing"],
                    "names": {"c01": ["rearing"]},
                    "overlap": "exclusive",
                    "behavior_types": {"rearing": "state"},
                },
            )
            for recording in recordings:
                fv.write_frame_view(
                    run_dir,
                    recording_id=recording,
                    subject=0,
                    t_ns=axis,
                    observed=np.ones(n_frames, bool),
                    instances=[i for i in instances if i.recording_id == recording],
                )
            return self.experiment

    return _Cohort()


# --- the synthetic trajectories' cadence, shared by the clustering fit templates below ------ #
# These sized the pose-clustering fit templates, removed with pose clustering
# (`docs/removed_capabilities.md`); the temporal and codebook templates below still use them.

#: The embed cadence this project's runs actually have (~59 rows/s).
_POSE_FIT_STEP = 17_000_000

#: Frames the video runs on past the last embedded row, so the frame view's dense video timeline
#: has an unobserved tail; without one, an all-True observed mask would prove nothing.
_POSE_FIT_VIDEO_TAIL = 60


# --- one real temporal-clustering fit, shared ------------------------------------------ #
# The montage's real topology is a temporal run — a member is a stretch of time with a start
# and an end — and the pose fit (since removed) alone hid every defect that depended on that (a temporal run
# whose node cache listed nothing shipped). The segmentation settings cut a short synthetic
# trajectory at this cadence; the shipped defaults are tuned for a real recording, where a
# 0.27 s minimum bout is a few rows and here would be most of the trajectory.
_TEMPORAL_METHOD_PARAMS = {
    "window_s": 0.05,
    "min_len_s": 0.1,
    "band_s": 0.05,
    "penalty": 1.0,
    "min_size_s": 0.05,
}


def _bouty_trajectory(n_bouts=12, bout_rows=30, dim=8, kinds=3, seed=0):
    """A trajectory made of repeating bout kinds, so segmentation has transitions to find —
    the shape ``tests/pipeline/test_temporal_clustering_module.py`` fits, written here rather
    than imported because a conftest that imports a test module collects it twice."""
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(kinds, 2, dim))
    rows = []
    for i in range(n_bouts):
        ramp = np.linspace(0.0, 1.0, bout_rows)[:, None]
        path = anchors[i % kinds, 0] * (1 - ramp) + anchors[i % kinds, 1] * ramp
        rows.append(path + 0.05 * rng.normal(size=path.shape))
    return np.vstack(rows).astype(np.float32)


@pytest.fixture(scope="session")
def _temporal_fit_template(tmp_path_factory):
    """One real temporal fit (``dtw_finch`` over the bouty trajectory, with kinematics), built
    once per session; never handed to a test directly."""
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    experiments = _load_experiments()
    root = tmp_path_factory.mktemp("temporal_fit_template")
    experiment_dir = experiments.build_experiment_dir(root)
    recording_dir = experiments.build_recording_dir(experiment_dir)

    features = _bouty_trajectory()
    n = len(features)
    frame_idx = np.arange(n, dtype=np.int64)
    t_ns = frame_idx * _POSE_FIT_STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(n + _POSE_FIT_VIDEO_TAIL, dtype=np.int64) * _POSE_FIT_STEP,
    )
    cx = np.linspace(0.0, 1.0, n)
    experiments.write_embedding_run(
        recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01", cx=cx, cy=cx
    )
    run = Experiment(experiment_dir).run_analysis(
        "temporal_clustering",
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": "dtw_finch",
            "method_params": dict(_TEMPORAL_METHOD_PARAMS),
        },
    )
    run.wait()
    return experiment_dir, run.id


@pytest.fixture
def fitted_temporal(experiment_dir, recording_dir, _temporal_fit_template):
    """A private copy of the session's temporal fit, as ``(experiment, run_id)``."""
    import shutil

    from detail.api.experiment import Experiment

    template, run_id = _temporal_fit_template
    shutil.copytree(template, experiment_dir, dirs_exist_ok=True)
    return Experiment(experiment_dir), run_id


_SESSION_LENGTHS_MODULE = "detail_test_session_lengths"


def _load_session_lengths():
    """Load ``tests/session_lengths.py`` by path, as ``_load_experiments`` does and for the same
    collision with the installed ``tests`` package."""
    if _SESSION_LENGTHS_MODULE not in sys.modules:
        path = Path(__file__).parent / "session_lengths.py"
        spec = importlib.util.spec_from_file_location(_SESSION_LENGTHS_MODULE, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_SESSION_LENGTHS_MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[_SESSION_LENGTHS_MODULE]


@pytest.fixture(scope="session")
def _session_lengths_template(tmp_path_factory):
    """One real temporal fit over twelve recordings from 8 s to 120 min, two of them imported in
    parts through the real importer (``tests/session_lengths.py``); built once per session and
    never handed to a test directly. Returns ``(experiment_dir, run_id, layout)``."""
    from detail.api.experiment import Experiment

    experiments = _load_experiments()
    root = tmp_path_factory.mktemp("session_lengths_template")
    with pytest.MonkeyPatch.context() as mp:
        _point_machine_storage_at(mp, root / "machine_config")
        experiment_dir = root / "experiment"
        experiment_dir.mkdir()
        experiments.build_experiment_dir(experiment_dir)
        layout = _load_session_lengths().build(
            experiment_dir, root / "videos", experiments
        )
        run = Experiment(experiment_dir).run_analysis(
            "temporal_clustering",
            {
                "source_recording_ids": list(layout.spans),
                "embed_run_id": "emb_01",
                "track_id": 0,
                "method": "dtw_finch",
                "method_params": dict(_TEMPORAL_METHOD_PARAMS),
            },
        )
        run.wait()
    return experiment_dir, run.id, layout


@pytest.fixture
def session_lengths(tmp_path, _session_lengths_template):
    """A private copy of the twelve-recording temporal fit, as ``(experiment, run_id, layout)``:
    ``layout`` (``tests/session_lengths.py::Layout``) is each recording's span read straight off
    the maps written, for computing expected answers without the code under test."""
    import shutil

    from detail.api.experiment import Experiment

    template, run_id, layout = _session_lengths_template
    experiment_dir = tmp_path / "experiment"
    shutil.copytree(template, experiment_dir)
    return Experiment(experiment_dir), run_id, layout


@pytest.fixture(scope="session")
def _codebook_fit_template(tmp_path_factory):
    """One real ``codebook_viterbi`` fit over the bouty trajectory, built once per session.

    The frame-run twin of ``_temporal_fit_template``: its members are frames and its instances are
    runs of them, so a clip taken out of the middle of an instance splits it — the topology a bout
    method, whose clip is a whole bout, cannot show. Never handed to a test directly.

    **With kinematics**, drifting across the recording, so the node cache's positions have
    something to hold: the node-cache tests that a frame method needs moved here from the removed
    pose-clustering fit (``docs/removed_capabilities.md``)."""
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    experiments = _load_experiments()
    root = tmp_path_factory.mktemp("codebook_fit_template")
    experiment_dir = experiments.build_experiment_dir(root)
    recording_dir = experiments.build_recording_dir(experiment_dir)

    features = _bouty_trajectory()
    n = len(features)
    frame_idx = np.arange(n, dtype=np.int64)
    t_ns = frame_idx * _POSE_FIT_STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(n + _POSE_FIT_VIDEO_TAIL, dtype=np.int64) * _POSE_FIT_STEP,
    )
    cx = np.linspace(0.0, 1.0, n)
    experiments.write_embedding_run(
        recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01", cx=cx, cy=cx
    )
    run = Experiment(experiment_dir).run_analysis(
        "temporal_clustering",
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": "codebook_viterbi",
            "method_params": {"fit_share": 20.0, "codewords": 20, "min_run_s": 0.05},
        },
    )
    run.wait()
    return experiment_dir, run.id


@pytest.fixture
def fitted_codebook(experiment_dir, recording_dir, _codebook_fit_template):
    """A private copy of the session's ``codebook_viterbi`` fit, as ``(experiment, run_id)``."""
    import shutil

    from detail.api.experiment import Experiment

    template, run_id = _codebook_fit_template
    shutil.copytree(template, experiment_dir, dirs_exist_ok=True)
    return Experiment(experiment_dir), run_id


@pytest.fixture(params=["frames", "bouts"])
def fitted_clustering(
    request,
    experiment_dir,
    recording_dir,
    _codebook_fit_template,
    _temporal_fit_template,
):
    """``(experiment, run)`` over **both** clustering topologies, as the authored run object:
    a ``codebook_viterbi`` fit, whose members are frames, and a ``dtw_finch`` one, whose members
    are bouts. A test that reads a clustering run's members runs twice, once per shape, so a
    change right on one and wrong on the other fails. (The frame half was a pose-clustering fit
    until pose clustering was removed.)"""
    import shutil

    from detail.api.experiment import Experiment

    template, run_id = (
        _codebook_fit_template if request.param == "frames" else _temporal_fit_template
    )
    shutil.copytree(template, experiment_dir, dirs_exist_ok=True)
    proj = Experiment(experiment_dir)
    return proj, proj.authored_run(run_id)
