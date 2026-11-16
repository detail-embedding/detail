# Export design — one backbone, every surface

Internal design doc (agent-facing), the export counterpart to `gui_design.md`. The spec the
export catalogue was built against.

**PR and issue numbers here (`#336`, `#710`, ...) are from the private repository this project was
developed in.** That history is not published, so a number cannot be looked up from here; the
sentence around it says what the change was.

It settles the open questions raised in user review. This is *what to build*.

The rule this exists to satisfy: **any analysis the app shows can be exported.**

And the sharper half, which is what makes it more than a Save-PNG button: exporting a rendered
video is the easy part; what matters is exporting the data needed to recreate that video.

---

## 1. What was wrong before this design

*History: the state this design replaced. `export_clusterer_videos`, `ClustererDetailPage` and the
clusterers have since been removed, and the clustering workspace now has the shell's Export
control (`gui/cluster_workspace.py::export_scope`).*

Seventeen `export_*` functions across `api/`, `storage/` and `analysis/`, behind **seven**
unrelated GUI dialogs, with no shared contract. Three consequences, all observed:

- **Capabilities with no way in.** `export_clusterer_videos` writes full-frame clips with masks
  drawn on, cluster-sorted, plus `clips.csv` and `clusters.csv` — most of what review
  asked for. It is reachable only from `ClustererDetailPage`. A user on an *analysis run's*
  cluster workspace, which has no export of any kind, concluded the feature did not exist.
- **Three different answers to "what is an export".** `dataset_export` is a registered executor
  runner with a run dir; `location_export` is an AnalysisModule; the other fifteen are synchronous
  API calls. Nothing says which is right.
- **Your tracking data requires an analysis run to get out.** `tracks.csv` exists and is excellent.
  You reach it by launching a *location-export analysis*, which nobody looking for "my tracking
  data" would think to do.

## 2. The scoping rule

> **Export what DETAIL made or holds, not a file that is already yours.**

The first version said "not what you handed it" and left video and streams out entirely. That was
too strict: a recording now exports its streams in DETAIL's aligned form, its frame timestamps,
and the video parts DETAIL imported a copy of. What stays out is a video part referenced in place,
which is already the user's, under the video root it was registered from (the `video` target in
`api/export_targets.py` skips it). Tracks, masks, embeddings and analysis results did not exist
before we computed them, and have no other way out. This is what keeps the target list short
enough to stay meaningful.

## 3. The two kinds of export, in one list

Both live in the same catalogue; they differ by one declared flag, not by mechanism.

| | **bundle** | **data** |
| --- | --- | --- |
| purpose | move the whole thing to another machine or person | get one thing into someone else's tools |
| format | DETAIL's own (`tar` + `export.yaml`) | open (CSV, mp4, npy) |
| round-trips | **yes** — importable back | never |
| covers | everything, opaque | one named thing, documented columns |

`ExportTarget.portable` distinguishes them so a surface can group them. Merging them into one
undifferentiated menu is how the menu stops meaning anything; splitting them into two mechanisms
is how we end up back at seven islands.

## 4. The model

An export is **a subject, a target, and a scope.**

```
SUBJECT               TARGET                          SCOPE
what you export from  what it can produce             how much of it

experiment ─────────► bundle (portable)               —
                      metadata table (CSV)            —

recording ──────────► scene geometry (CSV)            —
                      streams (aligned CSV)           —
                      frame timestamps (CSV)          —
                      video (imported copies)         —
                      clip  (raw|overlay|mask)        frames

detection run ──────► masks (COCO JSON)               recordings | all
                      performance report

tracking run ───────► tracks (CSV)                    recordings | all
                      performance report

embedding run ──────► feature vectors (npy + CSV)     recordings | all
                      performance report

keypoint detect / ──► keypoints (long CSV)            recordings | all
track run             keypoints for DeepLabCut         recordings | all
                      keypoints for SLEAP (HDF5)       recordings | all
                      keypoints as NWB                 recordings | all
                      keypoints video (mp4 + CSV)      recordings | all
                      performance report

analysis run ───────► clips  (raw|overlay|mask)       cluster | selection | recordings | all
                      instances (CSV, with XY)         cluster | selection | recordings | all
                      ethogram (CSV + picture)         cluster | selection | recordings | all
                      frame labels (CSV, per frame)   cluster | frames | recordings | all
                      performance report

comparison ─────────► report (figures + tables)       —

distill run ────────► performance report              —
```

**A recordings scope exists, and the first draft was wrong to refuse it (#710).** The table once
wrote "recording | all" for the run targets, and this paragraph struck it out: it would have been
the system's first per-recording narrowing, "a field on `ExportScope`, a branch in `_narrow`, a CLI
flag, and a GUI control the run page has no way to supply", and writing **one file per recording**
gave the reader the same choice, since "the narrowing is picking a file".

That reasoning holds when writing a file is cheap and fails exactly where it is not. From a
keypoints run's page, **Export…** with **Keypoints video** ticked wrote every recording's video,
far more than the drive had free, when one recording was wanted; picking a file after the
writing is not a choice. `masks`, `tracks` and `vectors` write one file per recording and have the
same shape at a smaller size. And the objection that no surface could supply it was already untrue
of the run page, whose list *is* the choice of recording on screen. So:

- `ExportScope.recording_ids` narrows an export to those recordings. `None`/empty is every
  recording, as for the other scopes, and it is normalised (ids, in order, once each).
- A target says it honours the scope by listing **`"recordings"` in its `scopes`**, which is a
  promise: `tests/api/test_export_recordings_scope.py` runs each one on a two-recording fixture,
  and a fail-closed scan fails any target that walks a run's recordings without declaring it.
  `metadata` and `bundle` describe the whole experiment and do not declare it, and `export list`
  shows the difference.
- A recording the export does not span is refused where the caller can see it
  (`Experiment.run_export`, `estimate_export`), not written as an empty folder.
- Surfaces: the API (`ExportScope(recording_ids=…)`), the CLI (`--for-recording`, repeatable, an
  id or a name), and the Export dialog's **Recordings** list. The run page's recording list is an
  `ExportScopeSurface` like a cluster workspace: it opens the dialog on the recording on screen,
  and the list is where the choice is widened. The list is offered for a processing run; an
  analysis run is narrowed by recording through the API and the CLI.

`SCOPE_KINDS` names `cluster`, `selection`, `frames` and `recordings`.

### Three parts

- **The catalogue** — `detail/api/exporters.py`, Qt-free. Each subject kind declares its targets.
  This is the single source both surfaces read.
- **One writer** — takes a plan, writes the files, drives progress, writes the sidecars, and
  records what it wrote.
- **Scope** — supplied by the caller. The CLI passes flags; the GUI passes what is selected.
  The catalogue never knows about widgets.

```
                    ┌──────────────────────────┐
                    │  catalogue (no Qt)       │
                    │  subject ─► [targets]    │
                    └──────────┬───────────────┘
                  reads        │        reads
            ┌──────────────────┴───────────────────┐
            ▼                                      ▼
      detail export …                        [Export ▾]
      (CLI, first-class)                     (GUI, placed by the shell)
            └──────────────────┬───────────────────┘
                               ▼
                    ┌──────────────────────────┐
                    │  one writer              │
                    │  progress · sidecars ·   │
                    │  manifest · cleanup      │
                    └──────────────────────────┘
```

## 5. Every export ships the data behind it

A target declares **both** halves, or states in one sentence why the artifact *is* the data (a CSV
export is its own input). This is enforced by a registry sweep, not by convention — it is the rule
most likely to decay.

```
~/flail_clips/
  cluster_03_recording_004_1200-1340.mp4     the artifact
  cluster_03_recording_007_0880-0995.mp4
  clips.csv        which bout, recording, frames, cluster   the inputs
  clusters.csv     what each cluster is
  params.csv       the settings that produced the run
  notes.md         the subject's notes (DESIGN_PLAN §15 "Entry notes"), when it has any
  export.yaml      what this export wrote  ← the manifest
```

`params.csv` is derived from `runners.job_params`, never hand-listed — CLAUDE.md's
`test_record_completeness` rule, one surface over. `export.yaml` is what makes re-exporting delete
only our own files and never a user's `.png` that happens to share the folder (the recorded
`test_comparison_figures_shape.py` class).

## 6. Long exports — every export is a run

The requirement: a user starts an export of everything, walks away, and comes back later to a
finished result.

**This section used to make `durable` a property of the target**, with a CSV staying inline and
clips being submitted as a job. That was the wrong classification, and the change that made
every export a run (#354) dropped it. `tracks` takes seconds on a test fixture and far longer
on a real 1080p recording — same target, same code. *Long* is a property of the data, so a per-target flag is wrong the first time
somebody exports `instances` across a sixty-recording cohort.

So every export is submitted as an executor job with a run dir, and inherits progress, ETA,
cancel, reattach and an activity log. Measured, the whole cost is under a second, and what it
buys is one mechanism instead of
two: one progress rendering, one error path, and no class of drift where the two quietly disagree
about the bytes they write.

**A fast export still reads as fast.** The cost is felt in the surface, not the CPU: sending
somebody to a run page to read "complete" after a 300 ms CSV is worse than a message box. So the
GUI submits, waits briefly on the handle, and *then* decides — finished, say what it wrote and
close; still going, show the run. One execution path, two presentations chosen by observation.
The CLI needs nothing: it streams, and a short run streams nothing.

**One request is one run**, however many targets it covers — one folder, one manifest, one row,
with the targets as the run's units. Splitting a single checklist into a run per target would
produce several folders for what was asked for once.

**Each target is isolated.** A target that raises no longer takes the rest of the export with it:
its outcome is recorded, the loop continues, and the run settles `partial` (some failed) or
`failed` (all did), with the manifest describing what actually landed. Inline, an exception was
tolerable because you were watching; as a run you walked away from, it is the difference between
"7 of 8 done, clips failed because a video moved" and an empty folder.

**The destination is an alias, never a path.** Everything in a job reaches the run record through
`runners.job_params`, and a record is `.yaml` — which `storage/export.py::portability_offenses`
scans and *refuses the bundle* on. An absolute destination would not fail a test; it would make
`detail experiment export` refuse the experiment permanently. So the job carries `{root, rel}`
into the `export_roots` section of `storage.yaml`, resolved in the worker — the same indirection
reference-in-place video already uses, generalised into `storage/roots.py` on its second user.

### What it will cost is told before it is asked (#710)

A run you can walk away from is a run you can walk away from *into a full disk*. The export that
wrote far more than the drive had free gave no hint beforehand of either number, and the dialog
that started it had one button. So the catalogue says what an export will cost, and every surface
says it before the first byte is written.

- **`ExportTarget.estimate`** is a callable `(subject, scope, options) -> Estimate`, and **every
  target declares one**. `Estimate(bytes, seconds, basis)` is a figure plus the sentence saying how
  it was reached; either number may be `None` with the reason in `basis` (the report, whose size
  depends on what is installed). A target
  without one is the bug, and `tests/api/test_export_estimates.py` fails on it by sweeping the
  registry — a target added tomorrow is covered the day it is added, and the figure for one
  recording must come out smaller than the figure for both
  (`test_an_estimate_follows_the_recordings_asked_for`).
- **`exporters.estimate(subject, targets, scope=, parent=)`** asks each target, isolating a
  failure to that target's row, and returns an `ExportEstimate`: per-target figures, the total
  (`complete` is false when any target could not be sized, so the total is read as *at least*), and
  `free_bytes` at the destination, from which `fits` follows. `Experiment.estimate_export` is the
  API door; `describe_estimate` is the one sentence both surfaces print.
- **How a figure is reached is one of three, in this order of preference.** *Counted* — the work
  is in files already on disk (rows in a detection run × bytes per row). *Modelled* — a constant
  measured once on a real recording and written beside the estimator with the date and the
  recording (`export_estimates.py`'s constants each carry their measurement; a test fails on one
  that does not). *Timed* — video, whose cost no constant predicts: a short sample from the middle
  of the first video is decoded and encoded under the real settings, and the rate scales to the
  frames wanted. Measured against real exports it overestimates the time, and that side of the error
  is the one to be on: an overestimate of a long export is a pleasant surprise.
- **Free space is measured where the export goes**, by `free_bytes_at` on the destination's nearest
  existing parent. The write rate is probed with a 16 MiB `fsync`ed file (cached for two minutes,
  deleted after), because a network share and a local SSD differ by two orders of magnitude and the
  seconds figure is dominated by the disk for every target but video.
- **An estimate is not a promise, and says so**: the dialog's tooltip lists each target's `basis`
  under the line "Estimates, not promises".
- **It refuses on the one thing it knows.** When `fits` is false the dialog turns **Export** off and
  says how much is free; `export run` exits non-zero unless `--ignore-free-space` is given. Time is
  never a refusal.

## 7. Keeping the GUI and the catalogue in sync

Two different drifts, two different fixes.

**The menu falling behind the catalogue** is structural: the shell builds the control *from* the
catalogue, and no page hand-lists anything. It cannot happen.

**A page growing a panel with no export** is caught by a sweep, in the shape this repo already
uses for discovering surfaces rather than listing them (`test_empty_states.py` walks every
`IndexTable`; `test_dialog_space.py` discovers dialogs):

```
for every result page:
    build it
    for every table and plot on it:
        its underlying data must be reachable in some export target
        ... or be listed as exempt, with a reason
```

**The rule is the loose one:** the *data behind* a panel must be reachable, not that each panel
has its own file. A cluster scatter of 40k points and `instances.csv` are the same export. The
strict reading generates exemptions faster than exports, and an allowlist nobody believes is worse
than no allowlist.

## 8. `location_export` — one door per thing

Splitting it on a principle rather than by taste:

| | what it needs | verdict |
| --- | --- | --- |
| `tracks.csv` | only the tracking run's own output | **a re-encoding → an export target on the tracking run** |
| `roi_relations.csv` | tracking **+ arena geometry** | computes new facts → **stays an analysis module** |
| `point_relations.csv` | tracking **+ POI geometry** | same |

A format change is an export. A computation is an analysis. After the split the analysis module
keeps the two relation tables and stops being the only way to get a trajectory out.

**As built: the analysis module still writes `tracks.csv` too, and that is the point.** "Stops
being the only way" is what the rule asked for; moving the file would change `record["artifacts"]`
for every location-export run already on disk and buy nothing. What makes two doors safe is that
there is one mechanism behind them — the export target calls the same
`analysis.locations.compute_location_tables` and the same `storage.location_export._write_csv`, and
`tests/api/test_exporters.py` asserts the two files are **byte-identical** for the same run. A
second implementation of "the same" export is the class where both files look right and disagree.

## 9. Build order

1. **The catalogue + the one writer + `detail export`.** CLI complete, GUI not yet wired.
2. **Clip flavours** — `render` in `{raw, overlay, mask}` written in one decode pass (decode
   dominates; re-decoding three times is the cost of separate calls), plus a crop toggle.
   Full field of view stays the default.
3. **Tracks as a target** on the tracking run, and the `location_export` split.
4. **The GUI control**, placed once by the shell, and the drift sweep.
5. **The cluster workspace** — the screen users reach for export from, which had no export at all then.
6. *Follow-up PR:* convert the other six exporters onto the contract, behaviour unchanged.

Steps 1–5 are this push. Step 6 is deliberately separate so this change does not rewrite seven
working features at once.

### What actually landed, and when

This list said "steps 1–5 are this push" and three pieces of step 3 were not in it — while the user
docs shipped describing them. Recorded here because the failure is the one this file is otherwise
about: a rule written down, believed, and never implemented.

| | in the first export change (#351) | later |
| --- | --- | --- |
| catalogue + writer + `detail export` | ✅ | |
| clip flavours + crop | ✅ | |
| `tracks` on a tracking run | ✗ *(documented as landed)* | ✅ |
| `masks` on a detection run | ✗ | ✅ |
| `cx_norm` / `cy_norm` on `instances.csv` | ✗ | ✅ |
| `export list` printing why a target is off | ✗ *(documented as landed)* | ✅ |
| the GUI control + the drift sweep | ✅ | |
| the cluster workspace | ✅ | |
| the other six exporters converted (step 6) | | still open |
| every export is a run (§6 above) | ✗ | ✅ (#354, the change that made every export a run) — and the `durable` flag it was specified as turned out to be the wrong idea |

The sweep that could not see it was `tests/gui/test_export_reach.py`, which counts **navigation
kinds** — and all four processing modules share the `run` kind, so `vectors` alone satisfied it.
`tests/api/test_export_registry.py` now sweeps `PROCESSING_MODULES` instead, and `performance` is
excluded from that count because it is available for every run and would make it vacuous.

## 10. Tests

- **Registry sweep**: every subject kind declares at least one target; every target declares its
  inputs or its reason. Data-driven, so a module added tomorrow is covered.
- **Drift sweep**: §7, over every result page.
- **Content, not existence**: a mask-only frame is binary, not merely a file that exists; a
  raw clip has no overlay drawn on it. `clips.csv` row count equals the files written.
- **Empty**: exporting zero instances writes an empty manifest and says so, rather than leaving an
  empty directory.
- **Cleanup**: a re-export removes what its own `export.yaml` recorded and nothing else.
- **Parity**: the CLI and the GUI produce byte-identical output for the same target and scope —
  the guard against the catalogue growing a second implementation.
- **Narrowing**: every target that declares `"recordings"` is run on a two-recording fixture and
  writes only the one named (`tests/api/test_export_recordings_scope.py`); a fail-closed scan fails
  a target that walks a run's recordings and does not declare it.
- **Cost**: every target has an estimator; counted and modelled figures are compared with the
  bytes actually written; the dialog's estimate follows the ticks, comes off the UI thread and
  drops a stale answer (`tests/api/test_export_estimates.py`,
  `tests/gui/test_export_recordings.py`).

## 11. Open

- **OQ-EXP-C** — where does the GUI control live on a page that is a dialog rather than a page
  (the query finish step)? Review asked for export to replace *Open folder* there.

### Settled

- **OQ-EXP-A** — do embeddings export as `.npy`, CSV, or both? **`.npy` plus a row-index CSV**: the
  `vectors` target writes `embeddings.npy` and `embeddings_index.csv`. CSV of a 768-d vector per
  frame is enormous.
- **OQ-EXP-B** — does `detail export` take `--run` and infer the subject, or an explicit
  `--subject`? **Both**: the subject is inferred from whichever id flag is given (`--recording`,
  `--run`, `--analysis-run`, `--distill-run`), and `--subject` is needed only for the experiment,
  which has no id flag.

## 12. Specified, not built: two additions on the embedding run (2026-10-20)

Asked for alongside an embedding-space figure, and specified as tool features because they
are useful without it: anyone analysing exported embeddings needs to know where the animal was and
how big its silhouette was, and anyone checking a run needs to see what the model was fed. The
figure does not depend on either; it can read the same data through the Python API.

### 12.1 `vectors` carries the kinematics

`embeddings_index.csv` is `row, recording_id, track_id, part, frame_idx, t_ns`. The embedding run
already stores a kinematics row per embedded frame (DESIGN_PLAN §14 *`kinematics/`*), in the same
order. Append, when the run has kinematics:

`cx_norm, cy_norm, mask_area_norm, bbox_x1, bbox_y1, bbox_x2, bbox_y2`

(`bbox_norm`'s four values, union xyxy, normalized.) A run that predates kinematics gets **no
columns at all**, not blank ones — the three-state rule `_instance_row` already follows: an absent
header says the export could not know, a blank cell says "we looked and there was nothing", and
collapsing the two is the `human_label=None` class. A recording whose kinematics part is missing
while others have one is refused for that recording with the `detail recording backfill-kinematics`
hint, never written with a ragged header.

- The estimator grows by the columns' bytes; `docs/cli.md`'s column list for `vectors` moves in the
  same change.
- Guards: the three states (with, predating, one recording missing) on the existing `vectors`
  fixture; the rows join back to `embeddings.npy` by `row` with kinematics in the same order.

### 12.2 `crops` — the subject at chosen frames, as images

A new target on an embedding run (`subject_kind="run"`, added to `_RUN_TARGETS["embedding"]`
beside `vectors`). Per frame, two PNGs and one index:

```
crops/
  raw/<recording name>_t<track>_<t_ns>.png     source frame cropped to the union box, padded
  model/<recording name>_t<track>_<t_ns>.png   the image the embedding model saw
  crops.csv                                     recording_id, recording_name, track_id, part,
                                                frame_idx, t_ns, raw_path, model_path
```

- **`model/`** is rendered through the run's resolved renderer spec, alignment included, so it is
  exactly what was embedded, not a re-derivation. `pipeline/render_preview.py::run_render_preview`
  samples its own rows and cannot be handed a list, so the per-row step it and the embedding worker
  share (decode → `RenderContext` → `MaskRenderer`) is what gets reused — lifted to a function if it
  is not one already, never copied.
- **Which frames — a scope, not an option.** `ExportScope` gains `frame_keys: tuple[(recording_id,
  track_id, t_ns), ...]`, normalised like `recording_ids` (in order, each once). The class docstring
  already says why a "how much" belongs in the scope: the manifest's `scope` line and the "nothing to
  write" sentence read it through `describe()`, which says `"N listed frames"`. The existing
  `frames` range does not fit — it is one recording-global range, and this is a scattered list
  across recordings. Only `crops` declares `"frame_keys"` in its `scopes` for now; a target that
  does not declare it refuses a scope carrying it. Surfaces: API (`ExportScope(frame_keys=…)`), CLI
  (`--frames-csv <path>` with those three columns, read into the scope), GUI not in this change —
  no page has a frame list to hand it yet, and that omission goes in `docs/cli.md`'s parity notes
  the way the analysis-run recordings scope did.
- **With no frame list**: a seeded sample of `n` frames (option, default 200) spread across the
  scoped recordings, and the seed is recorded in `export.yaml`.
- **Options**: `pad` (fraction of the box, default 0.1), `which` (`raw` | `model` | `both`, default
  `both`).
- **Reading video**: frames sorted by `(recording, part, frame_idx)` and read forward, one pass per
  part (Tenet 11) — a seek per frame is the preview carve-out, and an export of thousands of frames
  is not a preview. Frame location comes from the embedding index (`t_ns` → part, `frame_idx`), never
  from `CAP_PROP_FRAME_COUNT`.
- **No video**: a recording without its source video (a slim bundle, an unlinked external file) is
  refused for that recording with a sentence naming it; the rest are written.
- **Cost and progress**: an estimator (frames × two PNGs at the crop size, and decode time per part
  read), progress per frame with the phase `decoding <recording>` advancing per part — the §6
  contract and the long-op rule.
- **Guards**: on the shared embedding fixture — `model/` is byte-identical to the renderer's
  output for the same row through the embedding path; `raw/` at `pad=0` has the box's pixel size; a `frame_keys` scope writes
  exactly those rows; a source scan for a seek inside the frame loop (the Tenet 11 idiom); the
  no-video refusal; the estimate against bytes written (`tests/api/test_export_estimates.py`).

### 12.3 Build order

Two independent PRs, either first. Each moves `docs/cli.md` (and `docs/gui.md` where the dialog
shows it) in the same change, per `ship-a-change`.
