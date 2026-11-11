---
name: long-op-progress
description: Make a long operation report progress that actually moves, and test the cadence. Use when adding or changing a long-running stage, runner, analysis module, or any operation that reports progress, a phase message, or an ETA.
---

# A long operation must surface progress that moves

A silent long op is a bug, because it is indistinguishable from a hang. So is one whose bar sits at
0 and jumps to 100, and one whose phase label is frozen on `starting`.

The contract is `DESIGN_PLAN.md` §6 *Progress reporting*, which is the fuller document and holds the
rendering invariants. This is the procedure for building and testing one.

## Four cases, one standard

- **Executor-backed runners** (detection / tracking / embedding / keypoints / analysis) emit
  `ProgressEvent`s. The CLI renders them through `tqdm` (`_stream_progress`), the GUI through a
  progress bar off the same events.
- **Synchronous API methods that do real work** — `Recording.add_video_part`'s import + timestamp
  generation, `Recording.backfill_kinematics` — take a `progress` callback and report through it.
  Not being an executor job does not make an operation instantaneous.
- **Every caller drives that callback.** A CLI command wires it to a bar (`_StageProgress` for the
  `(stage, fraction, detail)` shape, `_stream_progress` for `ProgressEvent`s); the GUI wires it to
  its bar. Adding a long path — or a flag that turns a quick op into a long one, e.g.
  `--every-n-frames` on `add-video` — without surfacing progress is the bug, and needs a test that
  the callback is wired.
- **GUI interactive loads** (a preview opening, a frame decode or seek, a one-shot export behind a
  button) are *indeterminate*, so they do not fit `ProgressEvent`. They run off the UI thread
  through `gui/busy.py` — `run_busy` or `run_progress` — never as a raw slow call in a widget
  `__init__` or slot. Guarded by `tests/gui/test_ui_thread_blocking.py`.

`Recording.export_locations` is an outstanding **violation** of this rule, not an example of it: it
computes location tables across a whole tracking run, takes `(tracking_run_id, out_dir)` and nothing
else, and the CLI calls it bare.

## How to emit

- **In-loop (synchronous) runners** — a runner that owns its own loop rather than polling workers
  (trackers, dataset export, labeling, every analysis module) — wrap `progress_cb` in
  `core/progress.py::ThrottledProgress` and call it **inside the frame loop**; force the final
  event. Do not hand-roll a `last_emit` or `% N` throttle, and do not emit once per part.
- **Multiprocess coordinators** (detection, embedding) poll the worker progress files at
  `PROGRESS_INTERVAL_S`. **Training** is the only per-epoch exception.

**Weigh the loop, do not count it.** A fraction built from `(i + 1) / len(things)` claims the things cost the same. A tiny continuous span beside a huge one was reported as half the search, so the bar jumped most of the way in a moment and then sat still for most of the run; two exemplars of very different length were reported 50/50. Weigh each iteration by the work it will do — for a path matcher, span rows × usable exemplar rows — and let a step that does no work be worth zero.

## The cadence test, and the trap in it

Run the stage on a single-part fixture with `PROGRESS_INTERVAL_S` patched to `0.0` and assert an
intermediate `0 < frames_done < frames_total` event, not just a terminal one, **and a populated
`eta_s`** on that mid-run event.

**Assert the *inner* pair, on a *single*-unit fixture.** A cadence test written as
`0 < recordings_done < recordings_total` cannot pass with one unit, so it silently requires a
multi-unit fixture — and then passes while the single-unit case, the one the rule exists for, is
dead. Two search modules, since removed, shipped that test over a two-recording fixture, and
both bars sat at 0 for an entire one-recording search before jumping to 100. The fraction a surface
renders is `(recordings_done + frames_done/frames_total) / recordings_total`, so the inner pair is
what moves within a unit.

**Cut the fixture to one of *everything* the stage loops over.** The same trap sits one level in:
§20's method cadence test pinned the run count at one and said so in its docstring, then used
**three** exemplar clips — and `fused_curve` reports once per clip, so `0 < f < 1` was satisfied by
the exemplar count alone. The one-clip case, the ordinary first search, emitted `[1.0, 1.0]` and
nothing else until the search ended. Parametrize over the inner count rather than picking a comfortable
one. When every structural
loop is cut to one, whatever still moves the bar is inside the algorithm — and that is where the
hook belongs.

**A stage with distinct phases needs a phase-message test too.** The cadence test only proves the
*number* moves; it passes while the `message` sits frozen on `starting` through a long per-recording
load and segmentation, where the fraction is pinned to the recording boundary and cannot move.
Assert the runner streams the distinct phase labels in order — temporal clustering's `loading` →
`clustering` → `persisting` → `projecting` → `caching` — and **not** a lone `starting`. Template:
`tests/pipeline/test_temporal_clustering_module.py::test_the_runner_streams_distinct_phases`.
Collapse the phase emits back to one and the test must fail.

**A stage flowing through `ProcessingProgress`** also needs an aggregate-fold test (that `inner`
moves the aggregate bar — `tests/gui/test_workers.py::test_processing_progress_fraction_folds_inner_within_recording`),
and a **durable-run reattach test**: a fresh `Experiment.attach_run` streams a surviving run to a
terminal status, the coordinator completes with the launching app closed, and reconcile fails a
killed coordinator's run.

Trackers are covered data-driven over `TRACKERS` (`tests/trackers/test_tracker_progress.py`), so a
new tracker is caught automatically. Add the equivalent for a new module.

**Revert the in-loop emit and the test must fail** — see `guard-that-can-fail`.

## The rendering invariants

Full spec in `DESIGN_PLAN.md` §6. In short: a multi-recording aggregate bar folds the inner fraction
(`(recordings_done + inner_fraction) / recordings_total`, never `recordings_done/recordings_total`)
and always shows recordings completed/total; the runner populates `eta_s` via
`core/progress.py::remaining_seconds` and every surface renders it; every long op sets a phase
`message` both surfaces show **and advances it as the phase changes**; and **each surface has exactly
one tested `event → render` mapping** — the GUI's `processing_progress_fraction` /
`fit_progress_fraction`, the CLI's `_stream_*` tqdm drivers. A second hand-rolled mapping inside a
page or worker is the bug. CLI and GUI are deliberately **not** unified onto one function, because
tqdm owns its own ETA.
