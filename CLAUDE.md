# DETAIL — Agent Instructions

## Source of Truth

**`DESIGN_PLAN.md` is the authoritative specification** for all schemas, zarr layouts, YAML
formats, dataclass fields, validation rules and directory structures. Read the relevant section
before writing a pipeline stage. Do not invent fields, rename columns, or deviate from specified
dtypes or chunk sizes. If the spec is ambiguous, stop and ask — do not infer. The 17 Design Tenets
at the top of `DESIGN_PLAN.md` are non-negotiable; read them before starting any task.

## Start here if you are new to this repo

This file is the rulebook: what must be true of the code. `docs/how_we_work.md` is the judgment
behind it. Procedures are skills in `.claude/skills/`, loaded when they apply. Where to look:

| doing this | read |
|---|---|
| fixing a bug, or judging whether a fix is done | skill `fix-the-class` |
| writing or reviewing a test meant to prevent a class of bug | skill `guard-that-can-fail` |
| anything under `detail/gui/` | skill `gui-surface-change`; rules in `detail/gui/requirements.py` |
| adding a capability, config dimension, registry member, or a shared widget's parameter | skill `wire-every-surface` |
| branching, testing, opening a PR | skill `ship-a-change` |
| deciding whether work goes in the public or the private repository, or moving a feature across | `CONTRIBUTING.md` — *Public and private work* |
| running tests — any pytest run, so its progress and result are visible | skill `run-tests` |
| a stage that reports progress or an ETA | skill `long-op-progress`; contract in `DESIGN_PLAN.md` §6 |
| testing a stage that consumes another's output | skill `schema-edge-tests` |
| optimizing anything, or claiming a speedup | skill `validate-a-perf-claim` |
| adding or regenerating a docs page | skill `add-a-docs-page`; spec in `docs/docs_design.md` |
| filing a bug report | skill `file-a-bug-report`; spec in `docs/bug_reports.md` |
| any schema, zarr layout, YAML format, dataclass field, chunk size | `DESIGN_PLAN.md` — authoritative |
| deciding *which* rule applies, or whether to build at all | `docs/how_we_work.md` R1–R17 |
| a bug shape you recognise | `docs/recorded_classes.md` — the classes already paid for |
| finding where something lives | `docs/package_map.md` |

**No capabilities are hidden from a default build**: `surfaces.HIDDEN` is empty — count it, never
restate it from memory. The rules for a hidden capability are in `detail/surfaces.py`'s module
docstring, guarded by `tests/test_surfaces.py`.

**Sizing a change.** Anything touching a single part of the app is small. Anything that spans
several parts, or where a data structure has to be thought about, is structural — raise it before
building. Two copies of a shape can be tolerated; a third is when it becomes one mechanism.

## User-Facing Documentation

`README.md`, `docs/concepts.md`, `docs/cli.md` and `docs/gui.md` are the user-facing docs;
`DESIGN_PLAN.md` and specs such as `docs/gui_design.md` and `docs/backend_overview.md` are
internal. Every file under `docs/` is a user page in `mkdocs.yml`'s nav or an internal doc in
`exclude_docs` — there is no third state.

**A change that alters anything a user sees or types updates the relevant doc in the same
change**, not in a follow-up. The change → doc table is in `ship-a-change`; adding a page is
`add-a-docs-page`; the voice rules are `docs/docs_design.md` §5.

**Surface parity.** A user-facing capability — a stage, or a config dimension on one — is exposed
in every surface a user reaches it through (Python API, CLI, GUI) in the same change. An
intentional omission is recorded in `docs/cli.md`'s "Not in the CLI" with its reason; the
undocumented gap is the bug, not the omission. The procedure is `wire-every-surface`.

## Workflow

**Every change** goes on a feature branch and is merged through a pull request:

```bash
git checkout main && git pull
git checkout -b descriptive-branch-name
git push -u origin HEAD   # immediately, before any commits
git config branch.<branchname>.merge   # must show refs/heads/<branchname>
```

**That `push -u` is not optional.** A branch created off `origin/main` has `origin/main` as its
upstream, so a later bare `git push` can go to **main directly**. In a worktree, `git fetch` first
and branch off `origin/main` explicitly. Branch names are short, lowercase, hyphen-separated.

**At session start, before each push, and again immediately before merging:**
`git fetch origin main && git log --oneline HEAD..origin/main`. If a commit there touches a file
you are editing, read it before continuing. **Non-empty right before a merge means merge `main` in
and re-run the suite before merging**: two green branches can make a red `main` with no textual
conflict. Read the *N passed* line yourself before merging, and never cite a selected run as
evidence a change is green. Tests, docs and the PR body are the `ship-a-change` skill.

## Filing a bug report

A report from a machine the fixer cannot reach is the only record of that machine's state.
**Collect ruthlessly, conclude tentatively**: capture what decays first (the terminal scrollback)
and include the one cheap command that separates the hypotheses still standing. Procedure: the
`file-a-bug-report` skill. Spec: `docs/bug_reports.md`. Form: `.github/ISSUE_TEMPLATE/bug_report.md`.

## Hard Rules

These apply to every file in `detail/`. A violation is a bug, not a style preference. Where a rule
names a guard, the guard's module docstring holds the history of the bug behind it. A rule's
*Guard:* is the test that protects it, in any tier; the `guard` tier is narrower (*Test tiers*).

**Lazy backend imports.** `import torch`, `from ultralytics import YOLO`, `import sam3` and any
other optional backend import appear only inside methods of the concrete implementation class,
never at module top level. An import under `if TYPE_CHECKING:` (annotations only) is allowed.

**No absolute paths in experiment files.** Any YAML the pipeline writes uses paths relative to the
experiment root or recording directory. A string containing `/Users/`, `/home/` or `C:\` is a bug.

**`schema_version: 1` comes first**: the literal first key of every YAML, and
`root.attrs["schema_version"] = 1` immediately after a zarr group is opened. No exceptions.

**Atomic writes for concurrently-read experiment files.** A YAML/JSON that a background worker
rewrites while something else may read it is written to a sibling `*.tmp`, then moved into place
with `core.atomic.replace` — never bare `os.replace` / `Path.replace`, never `open(path, "w")`.
Guards: `tests/storage/test_zarr_seam.py`, `tests/pipeline/test_atomic_write_seam.py`.

**Storage uses the zarr v3 API (zarr-python ≥ 3.2), opened through `storage/zarr_store.py`.** A
bare `zarr.open_group` is a bug. Create arrays with `create_array`, grow them with resize +
tail-assignment (there is no `Array.append`), resume from `arr.shape[0]`. Variable-length columns
use `dtype="str"`, never `object`/`object_codec` or `dtype="bytes"`. Follow "Zarr write idiom
(v3)" in `DESIGN_PLAN.md` §9 exactly. Guard: `tests/storage/test_zarr_seam.py`.

**No Qt imports outside `gui/`.** `from PyQt6 import ...` is prohibited in `detail/core/`,
`storage/`, `pipeline/`, `models/`, `trackers/`, `transforms/`, `executor/` and `api/`.

**No credentials in experiment files.** HF tokens, API keys and other credentials live in
`~/.config/detail/credentials.yaml` or environment variables only.

**Model weights: `install()` is the only downloader, and the cache location is always an explicit
argument** (`DESIGN_PLAN.md` §5). `is_installed()` is a pure disk check; `readiness()` uses the
network only when not installed; `load()` never touches the network — it passes a path under
`model_cache_dir()` explicitly (`cache_dir=` + `local_files_only=True`, or a `checkpoint_path`).
An un-installed model fails at `validate()`; a run never auto-installs. **Never configure a
backend's cache with an environment variable** — backends read those at import, so a later write
does nothing. The one exception, `YOLO_OFFLINE`, is set only in
`models/_ultralytics.py::import_ultralytics()`, which a function must call before it imports
ultralytics, in the same function.
**`hf_files` is a model's file manifest**: what `is_installed()` verifies (per file, with
`try_to_load_from_cache`, never `is_cached()` or `snapshot_download(local_files_only=True)`) and
what `install()` downloads. Guard: `tests/models/test_no_implicit_network.py`.

**Expensive derived data survives an update.** The video preview cache, the node cache and model
weights are the user's. Orphaning or silently mixing them (a key's inputs, a file or folder name,
the cache location, an encode parameter) is a user-facing breaking change. To invalidate on
purpose, follow `DESIGN_PLAN.md` §3 *Compatibility contract*: version it, rebuild only what
changed, tell the user first. Run outputs are never invalidated.
Guard: `tests/storage/test_cache_compat.py` (the preview and node caches); the model cache's
default location is pinned in `tests/models/test_hf_access.py`.

**No FFmpeg** (Tenet 15). No `subprocess.run(["ffmpeg", ...])`, `os.system("ffmpeg ...")` or
library that shells out to it in pipeline code. Use `cv2.VideoCapture` for sequential frame
reading and writing, and `PyAV` for container metadata and PTS extraction.

**Sequential video reading only** (Tenet 11). Never call `cap.set(cv2.CAP_PROP_POS_FRAMES, ...)`
inside a batch processing loop — only at part start and in preview/prompt-test paths. Never hold a
whole video's frames; a buffer of at most `batch_size` frames for batched inference is allowed.

**Timestamps are always `t_ns: int64`** (Tenet 8): nanoseconds from the start of the first video
part, under that exact name. Float seconds are never a time reference in zarr, YAML or dataclasses.

**Frame counts come from the timestamp map, not cv2.** A part's frame count is
`len(timestamp_map)`, via `recording_store.load_active_timestamps` (or its `_cached` twin in a
loop over recordings). `cv2.CAP_PROP_FRAME_COUNT` is an estimate that can over-report; it is
allowed only in preview/prompt-test paths and for Quick segment's clip length.

**Long-running operations must surface progress.** A silent long operation looks like a hang.
A multi-recording bar folds in the inner fraction, the runner fills `eta_s`, and the phase
`message` advances as the phase changes; the full contract is `DESIGN_PLAN.md` §6 *Progress
reporting*.
- Executor-backed runners emit `ProgressEvent`s. Each surface has exactly one tested
  `event → render` mapping (CLI and GUI are deliberately separate); a second one is the bug.
- A synchronous API method that does real work takes a `progress` callback, and **every caller
  drives it**. A new long path, or a flag that makes a quick one long, ships with progress and a
  test that it is wired. `Recording.export_locations` is an outstanding violation.
  Guard: `tests/test_progress_callers.py`.
- GUI interactive loads (a preview, a frame decode, a one-shot export) run off the UI thread
  through `gui/busy.py` (`run_busy` / `run_progress`), never as a raw slow call in a widget
  `__init__` or slot. Guard: `tests/gui/test_ui_thread_blocking.py`.

**Module run configs are pydantic models** in `detail/configs/<module>.py` (`DESIGN_PLAN.md`
§6.5): the single source for each parameter's type, bounds, default, help and label. `validate()`
calls `Model.model_validate(config)` and keeps the checks that need recording data. It returns
`{**config, **model.model_dump()}` only when the model covers every key; for a subset model it
returns the enriched `config` unchanged — never merge a partial `model_dump()`. The GUI form and
CLI `--help` read the field metadata; hand-writing a range, help or default in a widget or
`@click.option` is the bug. A new module ships a config model (`config_model`).

**A resolved default is persisted; a headline metric ships with its floor** (Tenet 9).
- Never hand-write `record["params"]`: derive it from the job (`runners.job_params`), and build
  the job by carrying config fields by name (`query_eval.build_job`).
  Guard: `tests/configs/test_record_completeness.py`.
- Resolve a defaulted input in `validate()` and write the resolved value back. Raise rather than
  guess when the choice is ambiguous. Never resolve from mutable or order-dependent state.
- A raw score ships with its floor in the same output, and the floor's coverage matches the real
  prediction (`cluster_metrics.one_cluster_floor`). Guard: `tests/analysis/test_cluster_floor.py`.

**Labels score a method; they never build one.** Ground truth is read by the code computing the
metric and by nothing else. If a label decides what the method is *given* — which frames go in an
index, which name a cluster carries, which rows are sampled — no user can reach that score.
- A method's inputs are what a user could hand it: clips they chose, a model they trained, config.
- "Background" or "negatives" are sampled from the *unlabeled* remainder, positives and all.
- A label-derived artifact records where it came from and is applied somewhere else.
- A supervised step is fine when it is the thing being measured, never upstream of a number
  reported as unsupervised quality. A metric with documented optimism (clustering's majority-vote
  per-behavior F1) is not a leak: the test is whether a user could reproduce the number.
- Guard: `tests/pipeline/test_label_leakage.py` permutes the labels with the user's inputs held
  fixed; predictions must not change.

**A recording's id is the tool's, its name is the user's** (`DESIGN_PLAN.md` §3).
`Experiment.add_recording(name)` mints `recording_NNN`; `create_recording(recording_id)` is the
explicit form for scripts. Never derive an id from a filename or display one where a name will do
(`experiment.recording_names()`, `resolve_recording(ref)`). Names are unique: the tool suffixes a
collision it caused and refuses one a user typed. Guard: `tests/api/test_recording_identity.py`.

**Fixing a bug means fixing its class.** A fix that removes the symptom and leaves its siblings is
not done. Name the class without naming the file, derive a `grep` and sweep the repo, fix the
mechanism rather than the site, record fixed / deliberately-not for every site found, ship a
guard, retire a guard whose class has become impossible, and write the class into the guard's
module docstring. Procedure: `fix-the-class`, then `guard-that-can-fail`.
`docs/recorded_classes.md` indexes the classes already paid for; re-breaking one is a regression.

## Package Layout

`docs/package_map.md` says where things live and what a listing cannot: which module may import a
backend, which writer is canonical, which registry a file populates. `tests/` mirrors `detail/`.

## Code Style

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

- **Imports.** No relative imports — full package paths (`from detail.core import ...`), one per
  line, ordered `__future__` → stdlib → third-party → local, lexicographic within each group.
- **Naming.** `snake_case` modules, functions, variables; `CapWords` classes and exceptions;
  `ALL_CAPS` constants; `_leading_underscore` for private module-level names.
- **Formatting.** `black`, line length 88, 4 spaces. No backslash continuation, no semicolons.
- **Types.** Required on all public functions and methods. `x | None`, not `Optional[x]`.
- **Docstrings.** One line for simple functions; `Args:` / `Returns:` / `Raises:` for public API
  methods. Comments only for a non-obvious *why*; name identifiers well instead.
- **Exceptions.** Raise specific built-ins, or custom subclasses of one ending in `Error`. No bare
  `except:` or `except Exception:` unless re-raising or at an explicit isolation boundary (a
  subprocess worker writing `job.json`). Keep `try` blocks minimal — only the line that can raise.
- **An isolation boundary may not span the bookkeeping about the work it isolates.** A handler
  that records `status: failed` wraps the job and nothing else — not the progress emit, the
  run-record write or the poll of the worker's envelope. Publish progress through `best_effort`
  (`core/progress.py`). Guards: `tests/test_bookkeeping_boundaries.py`,
  `tests/api/test_bookkeeping_isolation.py`.
- **Patterns.** No mutable defaults; `is None` for None checks; `for key in d:`; `with` for files
  and resources; no module-level side effects (file reads, network calls).

## GUI Style

The Qt GUI is minimal: function over polish. **The rules are not listed here.** They are
predicates in `detail/gui/requirements.py`, each with a counterexample it must reject, swept over
every page, dialog and widget by `tests/gui/test_requirements.py` and printed to
`docs/reference/gui-requirements.md` at docs-build time; `docs/gui_design.md` §3.1 has the
reasoning. Add a new rule there as a predicate, never as prose alone. `requirements.py` wins when
copies disagree.

## Testing

### Running tests

A full run passes `-n auto --dist loadfile`: the suite is single-threaded without it, and
`loadfile` keeps a file's tests on one worker so expensive shared fixtures are built once.

```bash
# The edit loop — what this change touches, plus every guard (`--explain` shows which):
pytest -q $(python scripts/select_tests.py)
# Tighter, while iterating on one file — no guards, so run the line above before you push:
pytest -q $(python scripts/select_tests.py --no-guards)
# Before pushing — everything the branch touched since main:
pytest -q $(python scripts/select_tests.py --against origin/main)
# The gate, before merging — the whole suite, once:
pytest -q -n auto --dist loadfile
```

- **Run what the change touches while you work, and everything once, at the end.**
  `scripts/select_tests.py` prints nothing when the honest answer is "everything", so each line
  falls back to a full run. It is a convenience, not a gate (`tests/impact.py` is its table).
- **The full local run is the gate.** Do not count on CI to stop a merge, and there is
  deliberately no pre-push hook. Re-check `HEAD..origin/main` right before merging; if it is
  non-empty, merge `main` in and run the gate again. Say in the PR which command you ran.
- **Run `pre-commit install` once per clone**; `tests/test_pre_commit_config.py` fails without it.
- Pytest is configured in `pyproject.toml`. Do not pass `--rootdir`, `pythonpath` or `addopts`.
- **Give the full suite an explicit 10-minute budget**; a run killed by a shorter command timeout
  looks like a hang. `-m "not slow"` is a coarse fallback when the selector cannot narrow.
- A run stops after 10 failures (`--maxfail=0` lifts that). Coverage (`--cov=detail`) is a
  deliberate, slower run; `codecov.yml` says why nothing gates on it.
- **Never wait on a stalled run.** `scripts/suite_watchdog.py` ends a run that makes no progress
  for 300s (exit 3) and prints what it waited on. Read that; do not re-run blindly.

**Test tiers.** `tests/tiers.py` is the one table: what a test costs (`unit` / `gui` / `slow`),
derived from the file's directory, and whether it is a `guard`: listed in `GUARD_FILES`, added by
hand. Exceptions carry a reason and are checked by `tests/test_test_tiers.py`. A guard-tier file is
never `slow`. If a run ends in a stack dump from the per-test hang watchdog
(`faulthandler_timeout`), read the dump; do not raise the timeout.

### What a new test costs

**Ship a guard — and its cost is part of the change.** Neither obligation is a wall-clock budget:
- **Coverage must not drop** for the package you touched. Nothing records a baseline: run
  `--cov=detail` on `main` and on your branch, and compare that package. Replacing many slow tests
  with one shared fixture is therefore allowed.
- **A module starts at most one real executor run** (`run_analysis` / `run_processing`) unless
  `tests/test_real_run_budget.py` says otherwise. Build it once in a shared fixture and copy the
  result per test. That table only comes down.

### Writing tests

- **Fixtures.** Pipeline tests use the shared fixtures in `tests/conftest.py` (`DESIGN_PLAN.md`
  Tenet 16); never define equivalent setup inline. Use real directories via `tmp_path` and
  **never mock a storage primitive** (`zarr.open`, `yaml.dump`, `Path.mkdir`). The only legitimate
  mock is an external model inference call.
- Tests live in `tests/{package}/test_{module}.py`, import by full package path (no `sys.path`
  changes), and are named `test_{what_it_does}`.
- Every zarr output test asserts column names, dtypes and chunk sizes explicitly (Tenet 16).
- Every validation check has a test that triggers it and asserts the exception type and a
  meaningful fragment of the message.
- **GUI work.** Every page, dialog, run-preview widget and module config/result UI the app can
  open is instantiated by a test, verified through the real entry point, and checked by reading a
  rendered image (`gui-surface-change`).
- **Test the edges.** Derive cases from the documented edge properties of the schema a stage
  consumes and from the workflow shapes: fit ≠ applied, multi- and single-recording
  (`schema-edge-tests`).
- **Long-running stages.** A runner that emits a `ProgressEvent` only at a part boundary shows no
  live bar on a single-part run; test the cadence (`long-op-progress`).
- **Performance claims.** Prove the bottleneck first, benchmark on realistic data, control the
  confounds, validate through the real run path against a measured baseline
  (`validate-a-perf-claim`).
- **Mocking.** Scope a mock to the function or `with` block and patch at the point of use
  (`detail.pipeline.training.TrainedYOLO`, not `detail.models.trained_yolo.TrainedYOLO`). Never
  `sys.modules["x"] = MagicMock()` at module level.
- **Errors and warnings.** Raise a specific exception that says what is wrong and what to fix,
  never `warnings.warn` or a broad `try/except`. `pytest` runs with `filterwarnings = error`: a
  warning we emit is a bug to fix. A harmless warning a dependency emits about the host gets a
  *narrow* `ignore` in `pyproject.toml`, never a blanket one.

### CI runs Linux + CPU — don't assume the dev box

The CI workflow targets Linux with CPU-only torch and headless PyQt6; development is often on
macOS with a GPU. Fix the root cause of a platform difference, not the symptom:
- **No accelerator.** CPU-reachable code must not require CUDA/MPS. `pin_memory=True` warns on a
  host that cannot pin; that warning is harmless, so it has one of `pyproject.toml`'s narrow
  `ignore`s.
- **`ps` / process semantics.** Linux `ps` truncates the command column with no tty; use
  `ps -ww`. Do not assume signal or process-group behavior matches macOS.
- **Concurrency timing.** A torn read or an unreaped subprocess appears on a slower machine. Make
  the write atomic or reap the child; do not widen a timeout.
- **Never assert the backend from memory.** Call `torch.cuda.is_available()`, and validate
  accelerator-dependent behavior on the actual device against a measured baseline.

### Windows is untested by CI — its own platform gotchas

- **Path serialization.** Store relative paths with `.as_posix()`, never
  `str(path.relative_to(base))`. Guard: `tests/storage/test_path_serialization.py`, the
  source-scan idiom other guards copy.
- **Spawning and killing workers.** Spawn through `local.new_process_group_kwargs()`, real workers
  and test stand-ins alike: `start_new_session=True` does nothing on Windows, and killing a child
  that shares the console group also kills the test runner. Reference `signal.SIGKILL` only on the
  POSIX branch.
