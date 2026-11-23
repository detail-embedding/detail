# DETAIL GUI Design

> **Design/architecture doc — not a user guide.** For how to *use* the GUI, see
> [`docs/gui.md`](gui.md); for the CLI, [`docs/cli.md`](cli.md). This doc records
> the design. The few pieces not built yet say so (*planned*, *pending*, *not built*).
> `(new)` in §2 and §4 marks an API method that did not exist when the flow was
> designed; every one of those is built now.
>
> **History not published.** The issue and pull-request numbers below are from the private
> repository this project was developed in. That history is not published, so a number
> cannot be looked up from here; each sentence that cites one also says what the change was.

Working design doc for the desktop GUI. The goal is a **strong foundation that
guides future changes** plus a **buildable MVP**. `DESIGN_PLAN.md` remains the
authoritative spec for the backend; this doc covers the frontend and references
the plan (§N) for contracts it builds on.

Status: **built** — the shell and navigation (the banner modes, the project and
Quick tools rails), launch/run-progress, per-module config UIs, prompt-test (matrix) +
render-preview, the scene-geometry canvas, and detection, tracking and keypoint run
previews; the embedding input-image constructors + scrubber and the embedding result view
(Stage 1: coverage + PCA point cloud); deletion at every level (experiment / recording /
run / arena / stream / dataset / model) under one cohesive strategy (§1.1); run-table
navigation + rich run labels; video import with a smooth, staged progress bar; stream
ingestion (MED-PC, events CSV, time-series CSV and behavior-label importers, with add /
sync-to-video / edit-offset / delete and an event-raster + video preview); the Recording
information page (grouping variables, §1.3); the Model manager with install/readiness and
the Configuration setup checklist; every analysis module's config and result UI;
comparisons (§24); the **fit/train surface** (the project rail's Datasets / Models —
generate-dataset + train-model dialogs, distill, live progress, trained-model closed-set
detection — §1.2); **Active processes**; the **Keypoints manager** (§26); per-page
**Notes**; and **Version History**. Not built: the embedding result view's Stage 2 (a
linked scrub) and the alignment correction table (§3). The API additions are folded into
`DESIGN_PLAN.md` §15 ("API additions for the GUI").

---

## 1. Scope & principles

**The structural rules are checked, and they write themselves down.**
`detail/gui/requirements.py` holds every rule the app's widgets are swept against — one
function each, whose docstring *is* the sentence printed to `reference/gui-requirements.md`,
whose body is the check, and which cannot be registered without a counterexample it must
reject. `tests/gui/test_requirements.py` builds every page the shell opens and every dialog,
walks their widgets, and runs the grid.

Read that page for **what** is enforced; read this document for **why**. The two cannot
disagree, which is the point: this file once said the window geometry and rail split were
persisted while no Qt geometry call existed anywhere in the package.

Rules that are written down and *not* yet swept are listed in `NOT_YET_SWEPT`, with a reason
each, under a ceiling that only comes down. Adding to it is the visible act of writing a
guideline with no check behind it.

A rule does not have to be a *requirement in this registry* to be enforced. Some are about a
registry, an API or a click sequence, and the right check for those is a sweep or a behavioural
test rather than a walk over widgets; those are listed in `COVERED_ELSEWHERE`, each naming the
guard that covers it. The distinction matters for the same reason the ceiling does — the backlog
once said fifteen rules were unchecked when four of them had a guard, so the number that is
supposed to say "how much is left" was measuring something else.

**The GUI is a thin client over the Python API.** Every action calls a
`Experiment` / `Recording` method; nothing touches `recording.yaml`, zarr, job
dataclasses, or paths directly (Tenet 2). It is one of several frontends (CLI,
scripts) over the same API.

**Fixed constraints (decided):**
- **Style** — `detail/gui/requirements.py`, with the generated list at
  `reference/gui-requirements.md`: `Fusion` (as `gui/style.py`'s `DetailStyle`, scrollbars
  without arrow buttons), no QSS, greyscale + status colors only, monospace for ids/paths,
  single `QMainWindow`, no icons. §3.1 carries the reasoning.
- **Navigation** — two-level: a top **mode banner** (Experiments · Active processes ·
  Configuration · Model manager · Keypoints manager · Quick tools · Docs) is the global
  scope switch; most modes are a **full-width
  page** whose detail opens as a **drill-in** sub-page; a **tree rail** appears in two
  scopes — inside a project (the sections of the open experiment) and in **Quick tools**
  (each tool, expanding to its saved runs). The top bar is read-only
  identity; the shell keeps no history (see §3). Run launch/review live at the **experiment** level,
  data-entry at the **recording**.
- **Modules** — **bespoke per-module UI behind a registry-driven shell**: the
  shell enumerates the module registry and hosts each module's own page; adding
  a module is additive (no shell surgery). Mirrors the backend's three module
  families (processing §6.5, fit/train §12, analysis §16).
- **Long operations** — never block the UI thread; every run goes through a
  `QThread`, progress bar fed by `RunHandle.progress()`, errors from `job.json`.
- **Gate per-item actions on a capability predicate, never show them
  unconditionally.** An action valid for only *some* of the items a control
  operates on (e.g. sync/edit-offset for only external-clock streams —
  `clocks.needs_alignment`) is enabled from the selected item's capability, not
  built for every item and left to fail on click. An unconditional button that
  crashes or no-ops for the wrong item is the bug (this recurred once already —
  Sync-to-video shown for a video-anchored behavior stream). Test the disabled
  state, not just the enabled one.

**MVP scope (in):** home/experiments; experiment-data editing + recording management;
the per-recording processing modules (detection / tracking / embedding) launched as
**experiment-level runs over a recording selection**; the arena canvas; prompt-test +
render-preview iteration; run summary/management; the experiment runs index; the
**fit/train surface** (datasets + models — §1.2).

**After the MVP (all built since):** the analysis modules (§16) and their result pages;
grouping variables (§11) on the Recording information page; the keypoint stages (§26);
streams (§3), with importers for MED-PC, event CSVs, time-series (sampled) CSVs and
behavior labels. The alignment correction table (§3) is the one stream piece not built.

### 1.2 Fit/train surface (§12)

The fit/train family is **experiment-scoped** (not per-recording), and it lives in the
**experiment rail**: `EXPERIMENT_SECTIONS` (`gui/sections.py`) is Recording information /
Recordings / **Datasets** / **Models** / Processing pipeline / Analysis / Comparisons — the
training sections before the pipeline, in the order the work is done, since a dataset and a
model are what a detection run is given. Each expands to its datasets / trained models;
`DatasetsPage` / `ModelsPage` are the index pages. Keypoint **example sets** have no section
or page of their own: they are reached through the example-set picker
(`gui/example_set_picker.py`) wherever a keypoint model that needs examples is chosen.

It was a drill-in off the Model manager until #136 reworked that page and dropped the
Trained-models link, which left `ModelsPage` reachable only through `_on_model_deleted` — found
by a user deleting a model. A rail node is a door no page rework can remove, and
`tests/gui/test_reachability.py::test_a_delete_lands_somewhere_the_rail_can_take_you` holds every
after-delete landing to being a rail page.

**`ModelsPage`** holds two tables. **Models** (newest first; `EXPERIMENT_MODEL_COLUMNS`, with a
**Source** column naming the distill run that trained each — derived from the distill records,
which name the model their train stage produced — or `trained`) and **Distill runs**. The runs
stay a table of their own rather than rows of the first: a run that failed while labelling has
no model, and its page is where **Resume** lives. Two launch buttons: **＋ Distill new model**
(label + train, one durable run) and **Train from dataset…** (train on an existing dataset;
unavailable until one is complete). There is no separate Distill-runs page.

A launch or delete of a dataset, model or distill run calls `_refresh_training`, which re-reads
the rail (the new item is a rail child) and the page on screen.

- **Generate dataset** (`gui/fit.py::GenerateDatasetDialog`) — pick a completed
  detection/tracking run, a frame selection (all / every-N), `val_split`, and
  `min_contour_area`; calls `Experiment.generate_dataset`. The **only** surface — a
  run-detail "Export to dataset…" shortcut existed and was removed: it duplicated this
  dialog on every completed detection and tracking run (one launch surface per module).
- **Train model** (`gui/fit.py::TrainModelDialog`) — pick a complete dataset, a
  trainer (with a readiness line from `trainer.is_ready()`), a base model, and the
  trainer's hyperparameters (rendered from `HYPERPARAMS_SCHEMA`); calls
  `Experiment.train_model`.
- Both return a `DatasetRun` / `ModelRun` handle; the detail page shows live
  progress via `gui/workers.py::FitProgressWorker` and rebuilds on completion.
- **Apply a trained model:** the Detection config form lists built-in, experiment-trained
  and machine-imported models in **one** picker. Selecting a closed-set (trained /
  imported) model hides the prompt rows, shows the model's class list, and submits an
  empty prompt; the validator resolves `weights_source` from the id. It used to be two
  pickers behind a *Weights source* toggle that hid one or the other — see *Model
  availability* in §3.2 for why that had to go.

Dataset/model **deletion** follows the §1.1 strategy: each detail page carries a
`Delete dataset` / `Delete model` button (hidden while the object is still being
generated/trained). Active work blocks the delete — the dataset's export or a
model training from it; the model's training or an active detection run using it —
while completed dependents dangle as lineage (`Experiment.delete_dataset` /
`delete_model`).

### 1.3 Recording import + metadata (§3, §11)

A recording is an experimental unit carrying *optional* video, streams, and
**grouping variables** — video is no longer required to create one. Surfaces:

- **Recording information page** (the rail's first section, under the **Overview** node —
  `Section("metadata", "Recording information")`; the *label* is the user-facing name, the
  `metadata` id is internal) — the
  experiment `metadata.csv` as a cohort table, one row per recording. **View-only by default**:
  click *Edit* to change cells and add/delete columns. Edit mode is a **transaction** — every
  change is staged in the table and nothing touches disk until *Save*, and *Cancel* discards all
  of it — which is why this is the one table in the package that declares itself an editor
  (`requirements.EDITOR_PROPERTY`). Three column groups are never editable in either mode:
  `recording_id`, the derived **file** column (the video's name, full path in its tooltip), and
  one **completion** checkbox per thing a recording can have (`pages._COMPLETION_COLUMNS`: Field
  of view, Distance calibration, ROI, POI, Streams, Detection, Tracking, Embedding). Import CSV /
  Export CSV are whole-table file operations and live in view mode; import is
  merge-only (sets the cells present, skips blanks) and rejects unknown `recording_id`s —
  nothing applied on any error (DESIGN_PLAN §11). Double-click a row in view mode to open that
  recording. The recording detail page edits its own row inline.
- **Add recordings** dialog — non-blocking: each row is a recording with an *optional*
  video and per-row grouping-variable cells; "Add empty recording" makes a
  metadata-only one. Create hands the batch to the app **import queue** and closes
  immediately. The **whole bout appears at once** in the Recordings table: empty ones
  `ready`, video ones `queued` → `importing` → `ready` / `failed` (the `import_status`
  column), with a **Source** column showing each queued recording's pending video path
  (held in the in-memory queue, never persisted — Tenet 6). A "Cancel imports" button
  stops the queue and marks the remaining `queued` recordings `cancelled` (tags kept);
  progress surfaces on the status bar: a bar plus one line naming the recording in flight,
  its position (`3/21`), its phase and the ETA, rendered from the queue's folded
  `ImportProgress` by `workers.import_progress_fraction` (DESIGN_PLAN §3). Not on the
  Active processes page, which routes rather than reports. A crash reconciles orphaned `queued`/
  `importing` → `failed` on the next open.
- **Import bundle** — pick a bundle folder, validate the whole YAML manifest
  (errors → a report, import nothing), confirm, enqueue. Multi-part video + streams
  + metadata in one file (DESIGN_PLAN §3). Bulk-imported streams land at offset 0 →
  a later **Sync to video** pass on the recording page aligns them if needed.

### 1.1 Deletion (cohesive strategy)

One model for deleting anything in an experiment, so new objects follow it without
re-litigating. Every delete is an API method (`Experiment` / `Recording`); the GUI
only confirms and calls it.

**Principles**

1. **Confirm first.** A delete shows `QMessageBox.question` (Yes/No, standard
   text) naming the object and that it's irreversible. A confirmation is for an
   effect the user cannot take back — a delete, an overwrite, a discard, a killed
   run, or an action whose scope or cost has no other surface; every call site is
   declared in `requirements._CONFIRMATIONS` (GUI Style).
2. **Refuse, don't orphan.** A delete is blocked when a *downstream computation*
   would lose its inputs. The error names the blockers and what to delete first.
   (`Experiment.delete_run` blocks on `Recording.run_dependents`; `clear_arenas`
   blocks on `arena_dependents`.)
3. **Delete owns its outputs.** It removes the object's directory/files **and**
   its record, idempotently (missing → skip). Outputs first, then the index
   entry — mirrors `delete_run`.
4. **Indexes prune; computations block.** A record that merely *points at* the
   object (a processing run's `per_recording` map) is **pruned** when the object
   goes, and a record left empty is removed. A record *computed from* it
   **blocks** (principle 2). This is the rule for "delete a recording that a
   processing run included": prune, don't block — the run's real outputs lived
   in the recording and are going with it.
5. **Owned children go together.** Deleting a container deletes everything it
   owns in one confirmed step (a recording takes its runs, arenas, video, masks);
   internal dependencies don't block because they're removed as a unit.
6. **Delete lives on the detail page.** The page for an object carries its
   `Delete <object>` button (like Run detail's "Delete run"); on success it
   navigates to the parent index via an `on_deleted` callback. List-level
   "Remove selected" is reserved for *uncommitted* staging rows (the create
   dialog), never persisted objects.

**Per-object** (current + planned)

| Object | Blocked when | Removes | Prunes |
|---|---|---|---|
| Processing run | a per-recording output has a downstream run | per-recording outputs + run record | — |
| Per-recording run | a downstream run uses it | run dir + `recording.yaml` entry | — |
| Arena | a tracking run used it | arena files + entry | — |
| Stream | (nothing today; a future §16 analysis join would) | stream dir + `recording.yaml` entry | — |
| **Recording** | (nothing today; a future cross-recording analysis would) | the whole recording dir | itself from every processing-run `per_recording` (emptied run record deleted) |
| Video part | any run exists on the recording | part file + entry (+ reindex) | — |
| **Dataset** | its export is active, or a model is *training* from it | the dataset dir (images + labels) | — |
| **Model** | its training is active, or an *active* detection run uses it | the model dir (weights + run) | — |
| **Experiment** | (nothing — top of the hierarchy) | the whole experiment dir | — |

Dataset/Model follow principle 2 strictly: a *completed* model copies its weights
and a *completed* run's masks are already written, so neither blocks its dataset/
model — the lineage reference just dangles. Only **active** work blocks.

---

## 2. User flows

Each step is tagged **— Screen — `API call`**. `(new)` flagged an API method
that didn't exist at design time; all are now built (see §4 status).

### F0 — Experiments
Land on a list of all experiments; create or open one.
1. View experiments, pick one — **Experiments** (the banner's first mode) — experiment discovery
2. New experiment → F2, or open → the experiment's Overview

### F1 — First-run setup *(one-time, machine-local)*
1. Enter HF token — **Configuration ▸ Hugging Face** (a drill-in sub-page, §3.1) — `credentials.set_hf_token`
2. Check readiness / install weights — **Model manager (§4.6)** — `model_readiness`, `install_model`

### F2 — Create an experiment
1. Create experiment (name/description) — **Experiments → New experiment** — `Experiment.create`
2. The new experiment opens on its Overview; add recordings from its **Recordings** section → F3

### F2b — Edit experiment & manage recordings *(ongoing, inside the experiment)*
1. Edit experiment data (name/description) — **Overview** — `set_metadata` *(new)*
2. **Add recordings** (any time) — **Recordings** — the import queue → `add_recording`
3. Open/edit a recording → F3

There are no per-stage defaults to set: a run's configuration comes from the launch dialog's
**Start from** row (§3.2).

### F3 — Recording setup & edit
Edit all recording-level data where it lives (raw inputs + metadata; run outputs are read-only).
1. Add video part(s) — **Recording · Overview** — `add_video_part`
2. Edit recording metadata — **Recording** — *(new)*
3. Set up scene geometry (field of view, calibration, ROIs, POIs) → F3a
4. Add streams (MED-PC, events CSV, time-series CSV or behavior labels) — **Recording · Streams** — `add_stream`; align to video via **Sync to video** — `set_stream_offset` / `frame_t_ns`; **Preview** opens an event-raster timeline linked to a scrubbable video frame (alignment applied, doubles as a sync check) — `stream_preview` / `frame_timestamps`
5. Grouping-variable values — the **Recording information** table (§1.3) or the recording's own row
6. *(not built)* the alignment correction table (§3)

### F3a — Set up scene geometry *(one interactive canvas)*
1. Open the scene-geometry dialog; scrub frames — **Recording · Scene geometry** — `get_frame`
2. Select a kind (field of view / calibration / ROI / POI) and edit it on the shared canvas;
   the other three stay drawn as dimmed context
3. **Save** writes every kind at once and keeps the dialog open (greyed while nothing has
   changed); **Close** asks before discarding unsaved edits

### F4 — Launch a processing run *(over a recording selection)*
Single launch path: a run = one module + one config over 1..N recordings, fanning
out to a per-recording execution in each. The recording page has no launch control; a
completed run's page offers **Run <next stage> →**, which opens the launch dialog on that
run's recordings.
1. Choose module + recording selection — **Processing pipeline → New run**
2. Configure (with F5 / F6 iteration for detection / embedding)
3. Run — ***`Experiment.run_processing(module_id, recording_ids, config)` (new)*** → fans out to per-recording `Recording.run` (§6.5) → monitor (F7)

### F5 — Iterate on a detection prompt *(pre-step to F4 detection)*
1. Test prompts on N sample frames — **Detection config · Test prompts** — `run_prompt_test`
2. Compare frames×prompts overlay grid → tweak → repeat
3. Settle → chosen prompt **pre-fills** the detection run config

### F6 — Iterate on the renderer *(pre-step to F4 embedding)*
1. Preview renderer on N samples — **Embedding config · Preview** — `run_render_preview`
2. Compare source | rendered grid → tweak → repeat
3. Settle → chosen spec pre-fills the embedding run config

### F7 — Run summary / management
One page per run, reachable from the experiment runs index.
1. While running: aggregate + per-recording progress — `RunHandle.progress()`
2. On complete: results summary (status + per-recording counts; prompt/renderer grids for those tools) — run records
3. Manage: cancel / retry; on failure, show the `job.json` error and **retry** the incomplete
   recordings with the same config (4.14). A changed config is a new run, seeded from this one
   through the launch dialog's **Start from** row

### F8 — Experiment runs index
1. All processing runs, oldest first (store order) — **Processing pipeline** — experiment-level run index *(new)*
2. Multiple runs per module; open one → F7. Fit/train runs are under **Datasets** / **Models**,
   analysis runs under **Analysis**.

### Flows added after the MVP (all built)
- Set grouping variables (`metadata.csv`, §11) — the Recording information page (§1.3)
- Add + align a stream (§3) — F3 step 4, 4.15–4.17
- Fit a model → apply per recording (processing) → analysis (experiment context) — §1.2, the Analysis section

---

## 3. Navigation & the module-UI contract

### 3.1 Navigation

**Navigation is two-level. A top mode banner — Experiments · Active processes ·
Configuration · Model manager · Keypoints manager · Quick tools · Docs — is the global scope
switch: clicking a mode shows that mode's landing
page. Sub-navigation within a mode is *drill-in* — the landing lists the mode's items and an
`Open →` swaps in a full-width sub-page (breadcrumb in the top bar), the way a phone Settings
screen works. A left tree rail appears in exactly two scopes, one at a time: the *project
view*, reached by opening an experiment from Experiments — it holds that one experiment's
Overview and sections —
and *Quick tools*, which holds each registered tool with its saved runs beneath (the rail is
up across the whole mode, its landing included). Everywhere else a mode renders full-width
with no sidebar. The top bar is read-only identity; the shell keeps no history and has no Back
button — the banner (and, where there is one, the rail) always offer a way anywhere.**

**An in-page mode switch is not a Back button.** A page may have modes of its own and a button
that returns to its default one: an analysis run's `<run_id>` link on the path line (from a
full-window view back to the run's landing page, §3.2) and snapshot mode's "Back to live run".
Each changes what *this* page shows and names where it goes; none walks a history of
pages, which is the thing the shell does not keep.

```
Banner:  [ Experiments ] [ Active processes ] [ Configuration ] [ Model manager ]
         [ Keypoints manager ] [ Quick tools ] [ Docs ]               <running build>

Experiments        full-width — one card per experiment (double-click a card → that project)
Active processes   full-width — every run and export in flight, across every experiment
Configuration      full-width — the setup checklist; each step drills in:
                     GPU / CUDA                          accelerator status + CUDA doctor (§1)
                     Hugging Face access                 token for gated downloads (§1)
                     SAM 3 backend package               its own page: the pip install commands
                     SAM 3 weights                       → opens the Model manager (§5)
                     Install a runnable detection model  → the Model manager (no NVIDIA GPU only)
                     See it segment a clip               → opens Quick tools
                     Recommended: distill a fast model   → opens the Model manager (§12)
                   then the Preview cache line (where previews live, their size, Clear)
Model manager      full-width — one page: Global models (imported/promoted, metadata +
                     preview), the active experiment's trained models, and the Hugging Face
                     foundation models (suggested first, then installed); four rows each, then
                     scroll; ＋ Distill new model / Import model / Restore from zip on top; a
                     row drills into its detail page (§5/§12)
Keypoints manager  full-width — the machine-wide skeleton library (§26)
Quick tools        one-off tools (DESIGN_PLAN.md §27, Tools), with a rail of its own (below)
Docs               opens the bundled/local docs

Quick tools rail (up across the whole mode, its landing included):
  Quick segment          ▸    the tool's launch page + its saved runs; expand → each run,
                              oldest first, labelled `run_003  ·  complete`

Project rail (only inside a project — open an experiment from Experiments):
  Overview                    the experiment's own page: name, description, Export, Delete,
                              a summary row per section (not a section itself)
  Recording information       grouping variables + completion (§11)
  Recordings             ▸    expand: its recordings (click → recording detail)
  Datasets               ▸    expand: its training datasets (§12)
  Models                 ▸    trained models + distill runs; expand: its trained models (§12)
  Processing pipeline    ▸    the runs index (detection/tracking/embedding); expand → each run,
                              oldest first (every rail section lists in store order)
  Analysis               ▸    expand: its analysis runs (§16)
  Comparisons            ▸    expand: its comparisons (§24)
```

- **One interaction rule for every rail node**, in either rail. A click **always shows the
  clicked node's page**; a **folder** click *also* **toggles** its expansion (open ↔ closed) —
  identically for every section (Recordings / Datasets / Models / Processing pipeline /
  Analysis / Comparisons) and for every tool. The **▸ arrow** is the same toggle. A **leaf** just
  navigates. (A click on an already-open folder that only collapsed it and left the prior
  screen up was the bug.)
  *Contract:* `_on_tree_click` owns expansion and applies it uniformly; `_navigate` is the
  single dispatcher for both rails (kinds are distinct across them) and only says **what to
  navigate to**, never how to expand — so a new section or tool kind inherits the toggle
  for free. Hand-rolling a section's own expand/collapse is the bug. (Standard tree
  behavior — no ambiguous double-click.)
- **The rail selection follows every way into a page.** A rail click, a landing's `Open`
  button, a table double-click and the jump after a run all funnel through the one opener
  for that destination, which moves `currentItem` (`_select_rail_node`). A path that shows
  a page while the rail still highlights the previous node is the bug.
- **A landing/folder page is a *summary of its contents*, never a duplicate of a sub-page.**
  A drill-in landing or a rail folder shows a summary of what's inside — it must never
  re-render one of its own child pages (the experiment root once re-showed the
  **Recordings** table — that was the bug). Two shapes, chosen by what it holds:
    - **Mixed container** (an **experiment**) → an **overview dashboard**:
      an at-a-glance header (counts) then one row per section, each with a live count/status
      and an **Open →** that drills into it — every row is both a summary and the nav to that
      section. It never embeds a sub-page's table. (The **Model manager** is a bespoke
      three-table page — Global models + the experiment's models + Hugging Face — not a dashboard, since its content *is*
      the models, not a set of sections.)
    - **Homogeneous collection** (**Recordings**, **Runs**, **Analysis**,
      **Comparisons**, **Datasets**, **Models**, a **tool's** saved runs) → a
      **table of its items** (one row per child, double-click → its detail) + the
      folder's primary **New…/Add…** action + a one-line summary. This is the index the
      collection folders already render; a rail folder expands to list the same items.
      **One exception: the Experiments list** (the banner's first mode) renders a **card
      per experiment** — a cached still from the middle of its footage beside its name,
      recording count and per-stage completed-run counts. It is the front door rather than
      a work surface, and its job is *recognition*: in a table every experiment looks
      alike, and the one thing that identifies yours is what the footage looks like. The
      still is cached per experiment (`Experiment.preview_image`) and loaded off the UI
      thread, so the list never decodes video while it is on screen.

    A landing that shows **nothing** (a dead node) or **re-shows a child's page** is the bug.
    Build the summary from cached metadata / `read_cache`, never an O(recordings) disk sweep
    on the UI thread.

    **A collection page reads its rows off the UI thread, and says so while it does.** The
    recordings and analysis indexes compute their rows in a `BackgroundCall` (Qt-free
    builders, `pages._recording_rows` / `_analysis_rows`) and install the table when they
    land. Two consequences, both load-bearing: the builder wraps its loop in
    `read_cache.pinned()`, because a row wants several fields of one record and each
    `Recording` property is its own read — 9.0 round trips per recording became 2.0 (the
    floor: one listing probe, one read), and 541 over sixty recordings became none at all on
    the UI thread; and the table shows a
    **reading** notice rather than its empty one, since an empty state is a claim about the
    data and zero rows mean nothing until the read is done. A page in flight sets `_reading`,
    which is what the empty-state guard settles on.
> **Two copies of these rules exist, and this is the one with no test behind it.**
> `detail/gui/requirements.py` holds each as a predicate with a counterexample, checked by
> `tests/gui/test_requirements.py` and generating `reference/gui-requirements.md` at docs-build
> time; the bullets below restate them. Only the first can fail. So: **when they disagree,
> `requirements.py`
> wins** — it is the one that is executed — and the table-sizing bullet below had already
> drifted to a constant that no longer exists before anyone noticed. What earns a bullet's
> keep here is the *why*: the measurement that produced the rule, and the failure it was
> written against. Add a new rule as a predicate in `requirements.py` first; restate it here
> only if there is reasoning that does not fit in a docstring.
>
> There were **three** until the third — `CLAUDE.md`'s 468-line GUI Style section — was deleted
> for being the copy that could not fail, and already the incomplete one: `requirements.py`
> carried six enforced rules it never mentioned, four of them about video surfaces. Measured
> bullet by bullet, 84.5% of that section was enforced by a named test and a further 12.7% was
> already stated here, leaving three rules with no other home; those are below, each saying what
> does and does not check it. `CLAUDE.md` now points here and at the generated list. This
> paragraph is the sentence that was finally acted on.

- **Navigable tables signal that a row opens something.** A table whose rows open a
  detail/result page opens on **double-click** and indicates it — a pointing-hand cursor
  over the rows + a trailing **"›"** column, plus a one-line "double-click to open" hint
  on the index/collection pages. Centralized via `_table(navigable=True)` and set only at
  the navigable call sites; a **selection-only** table (row-select just enables buttons —
  e.g. the Hugging Face table) or a **static** display must not get it, or the cue misleads.
  Classify every new table navigable vs selection-only.
- **A page's create action sits above the list it fills.** *New run*, *Add recordings*,
  *Import model* — the button that adds an item goes **above** the table, never under it. A list
  is the thing on a page that grows, so a button beneath it moves further away with every item
  added, until on the page where it matters most you scroll to reach the control you came for
  (reported from a real experiment: *"as the run list gets long it gets buried and you have to
  scroll to find this"*). Buttons that act on the **selected** row — Install, Remove, Check
  access — are a different thing and stay below the table, where they read as operating on what
  is above them. Enforced for every page by `requirements.py`'s
  `a_page_puts_its_create_action_above_the_list_it_fills`, which is why a page added tomorrow
  inherits it; an exemption carries its reason, as every allowlist here does.
- **Index/list tables size to their content, up to a row budget.** A `QTableView` defaults
  to vertical `Expanding`, so a table with a few rows stretches and shows blank space below
  its rows; and its `sizeHint` is a constant 192px ≈ 5 rows *whatever* it holds, so a cap
  alone is a height it never reaches. **Three numbers, three jobs**: `fit_table_height` assigns
  the **hint** (on the instance — what the binding's virtual dispatch reads) as the display height, the
  **minimum** as `MIN_VISIBLE_ROWS` = 4, how far it may be squeezed, and the **maximum** as the
  content height. The row budget is `max_visible_rows(row_height)` — what fits on *this* screen
  — which replaced a constant 16 that could never have been right on a display it was not picked
  on. One number for the hint and the minimum makes a table rigid in the pane that needed it to
  yield. `fixed=True` for a bounded top-N panel that must never scroll.
  So a short table hugs its
  rows and a long one keeps its own scrollbar. (The content area *does* scroll — `PageScroll`,
  below — but that is a backstop for a page too tall to fit, not a licence for every table to
  spill into it.) A **primary-content grid** meant to fill its page/dialog (the metadata
  cohort grid is the exception — the user wanted it to fit; but the add-recordings dialog, the
  behavior-query dialogs, and split-pane tables) opts out — it uses `addWidget(table, 1)` and
  does not call `_fit_table_height`. The **width** is the same rule on the other axis:
  `ResizeToContents` sizes each column to its content and says nothing about the leftover, so
  a two-column summary occupying 133px of a 547px pane trails ~410px of empty grid on every
  row. `fit_table_width` (`gui/tables.py`) caps it; a fill table opts out of both.
- **Every table sorts and resizes.** This has required sortable headers since before it was
  true anywhere — `gui/tables.py`'s `make_columns_resizable`
  (drag a column border; starting widths still fit content) and `enable_sorting`
  (click a header to sort) are the two calls, wired into both shared builders
  (`pages._table`, on by default) so a new table gets both for free.
  A caller opts a specific table out of sorting only with a documented reason — a bounded
  top-N panel, a table whose row order *is* its claim (registry order, a fixed reference
  position), or one where sorting cannot be made safe (a `QTableWidget`'s vertical header
  labels do not travel with a sort). Never opt out of resizing. A column that takes the spare
  width gets it from `tables.give_slack_to` (or `fill_width` for the last), never the `Stretch`
  resize mode: a stretched column cannot be dragged, and a column to its right dragged wider
  grows *leftward*, the stretched one giving the width back — reported on the clustering
  measurements table. Every section is `Interactive` (`tests/gui/test_table_sorting.py`).
  **A row-click/selection handler must read identity via `tables.row_id`
  (stamped with `stamp_row_id`), never `self._ids[index.row()]`** — a row's *position* in
  the view is only that record's position while nothing can reorder it, and once sorting is
  on, a handler that trusts position silently opens, plays, or overwrites the wrong record.
  A `QTableWidget` repopulated in place (not rebuilt from scratch) goes through
  `tables.refill_table_widget`: Qt resorts a `QTableWidget` after every single `setItem`
  call while sorting is enabled, which scrambles a row-by-row fill loop that assumes row
  `r` stays row `r` until it finishes writing it.
- **An empty index says so; it never shows bare headers.** A collection page (Recordings,
  Runs, Analysis, Comparisons, Datasets, Models)
  is built on `pages.IndexTable`, which swaps the table for a one-line notice when it has no
  rows, and its caption comes from `_index_caption(summary, rows=n)`, which drops the
  "double-click a row to open" hint at zero. A zero-row table still draws its header strip, so
  without this a fresh experiment reads as a broken table inviting a click on rows that are not
  there. The notice follows the house idiom — parenthesised, lowercase — and, unlike the
  embedded result views, **names the action that would fill it** (`(no comparisons yet — New
  comparison… contrasts a behavior between groups of your recordings)`), because these are the
  pages a new user lands on. Guarded by `tests/gui/test_empty_states.py`, data-driven over every
  index page.
- **The page scrolls; it never forces the window.** The content area is a `PageScroll`
  (`gui/scroll.py`). Qt clamps `resize()` to the layout minimum, so before this a page taller
  than the screen grew the *window* past the display rather than clipping — measured, a 920px
  analysis page in a 739px content area with no way to reach the bottom. The page is sized to
  its **hint** rather than its floor (floor-sizing scrolls less but lays every pane out at its
  minimum), and the container routes the wheel, because pyqtgraph accepts wheel events and never
  propagates them. Scrolling is a backstop: a page that fits still fills the viewport and shows
  no scrollbar, and lowering a pane's floor is always better than spending the difference in
  scroll.
- **A dialog gets the same treatment from its base class, not from a helper it remembers to
  call.** Every dialog inherits `gui/dialog.py::Dialog`: build rows into `body()`, add the
  button box with `set_buttons()`, never `QVBoxLayout(self)`. The base owns the top-level
  layout — one `PageScroll` around the body with the buttons pinned outside it, so the window
  can always be squeezed onto the screen and Cancel/OK never scroll away — and adds the
  trailing stretch on `showEvent`, only when nothing already takes the slack. The predecessor is
  the lesson: `sizing.finish_dialog` was the right rule reachable only by the dialogs that
  called it, **8 of 44**, and is now deleted so there is nothing left to call instead. A policy
  on a base class every dialog already inherits is reached by construction. **A second scroller
  inside a dialog is the bug** — `launch.py` removed its own, since a same-axis scroller nested
  in another is one scroll too many. One scroll per window.
- **A size that holds text or a graph is in lines, not pixels; and a plot gets a range, never
  a fixed height.** `sizing.lines(n)` is this package's `em` (Qt Widgets has no `rem`): a raw
  pixel constant is correct on exactly the screen it was picked on, and it cannot follow a user
  who has *asked* for larger text — the boxes stay put and only the words grow. Store the **line
  count** at module scope and call `lines()` where the widget is built: `lines()` reads the
  application font, and there is no application at import time (a module-level call hangs the
  import). For a plot, `setFixedHeight` is the bug — one number is a floor and a ceiling at once,
  so a short pane cannot squeeze it and crops it instead, and what crops is the x-axis and its
  labels. Measured: a 444px dial column in a 334px viewport, 304px of it two `setFixedHeight`
  calls. pyqtgraph itself imposes no floor (`minimumSizeHint` 68x68) and its axis chrome is a
  constant ~40px at any height, so a *ranged* plot shrinks its data area and keeps its axis.
  Guarded by `tests/gui/test_pane_floors.py`.
- **A widget declares the floor its own contents need — never a number that undercuts them.**
  `setMinimumHeight(640)` on a workspace whose children need 933 reads as a floor and acts as a
  licence: Qt squeezes to 640 and the children are clipped inside, silently. Leave it unset and
  the floor is the contents' own, honest by construction; `add_body_widget` already gives an
  expanding child the stretch, which is the job those numbers were added to do. Note
  `setMinimumSize(0, 0)` does **not** lower a floor — zero is the default, so Qt falls back to
  the hint; only a non-zero minimum or `QSizePolicy.Ignored` does.
- **A result view's secondary bands fold, and remember it.** Where a workspace stacks parts that
  each want their height and none of which is wanted at once — a behavior scan's Clips,
  Ethograms, Locations and Interpretation; BROWSE's ethogram — the secondary ones go in a
  `collapse.CollapsibleBand`. Folding is the only move that changes the *sum* rather than
  redistributing it: a hidden widget is outside its layout, so its floor leaves with it
  (measured on the clustering INTERPRET tab when it still had bands: its floor went 949 → 839 →
  621px as two bands folded). Result bands open by default — the clips are how a behavior is
  recognised, and a screen that opens having hidden its own evidence is worse than one that
  scrolls; a run page's *Run details* is the exception and starts folded — and the state is
  kept per band in `QSettings`, because the page is rebuilt on every navigation. **Sections fold
  by their header; panes fold to a labelled strip.** A folded band keeps its titled header, so it
  reads as something you closed. Krita shipped per-pane collapse and removed it ("even power users
  were not using these options"), because a collapsed pane was an unlabelled sliver — so a side
  panel here folds to a strip with its **name written down it** ("Navigation", "Filters · 2
  active", "Labels"). One gesture for every side panel (`gui/grip_splitter.py`, user decision
  2026-11-09): drag its grip narrower than the panel's minimum and it snaps shut to the ~16px
  strip, never to zero. The strip is its own way back: an expand mark (› for a left panel, ‹ for
  a right one) sits at its centre in place of the middle grip dot — exactly where an open grip has its dots,
  so the place to aim does not move when the panel shuts — and a click anywhere on it, or a pull
  outward, opens the panel. The pull opens it the moment it leaves the strip and then follows the
  pointer: before 2026-11-09 nothing moved until the pointer had travelled the panel's whole
  minimum (160px on the navigation, 180 on AUDIT's list), which read as a strip that cannot be
  dragged. The strip shows a pointing hand and "Click or drag to show <name>"; double-clicking the
  grip, open or shut, still toggles. It reopens at the width it last had. The mark is painted by
  the handle, not a button beside the panel, so it is allowed by
  `a_side_panel_has_no_hide_button` while a separate ‹ button is not. Only side panels fold — lists, the navigation, filters,
  checklists, summary columns; main content and video players stop at their minimum. There are no
  hide/show buttons of a panel's own (the navigation's ‹ / › and Interpret's "◂ Hide filters" are
  gone); Ctrl+\ still toggles the navigation. Each splitter declares its foldable panes
  (`set_foldable`, first or last pane only). Enforced by four requirements —
  `a_split_is_built_as_a_grip_splitter`, `a_split_remembers_where_it_was_left`,
  `a_pane_never_drags_to_nothing`, `a_side_panel_has_no_hide_button` — and the gestures by
  `tests/gui/test_grip_splitter.py`. Distinct from `forms.AdvancedSection`, which shares the
  glyph and none of the job. Guarded by `tests/gui/test_collapse.py`. **Not the clustering
  INTERPRET tree**: it drags (a `ResizeHandle`) and does not fold, because that page is the tree
  and its clips and a folded tree leaves nothing to pick from (user decision 2026-11-07; pinned by
  `tests/gui/test_workspace_parity.py`).
- **One scrollbar, one zoom.** Every scrollbar is a real `QScrollBar` under `DetailStyle`: Fusion's
  drawing with no arrow buttons, the handle over the whole track
  (`a_scrollbar_has_no_arrow_buttons`). Every 1-D zoom (the cluster tree, the ethograms) holds a
  `gui/zoom.py` `ZoomWindow`, takes its gestures from `zoom_wheel` / `zoom_pinch`, and shows where
  it is with an `OverviewBar`, which is a scrollbar and so the same control as the rest
  (`zoom_lives_in_one_place`). The overviews used to be bands each widget drew for itself, and
  looked like nothing else beside them. An overview bar is hidden at the whole span with its room kept
  (`retainSizeWhenHidden`): greyed, it drew as an empty outlined box; hidden without its room, it
  would resize the picture under the pointer on the first zoom.
- **A binding is only real on the platform it can be produced on.** `pyqtgraph`'s 3-D view
  implements pan and binds it to middle-drag and Ctrl+left-drag; a MacBook trackpad has no middle
  button, and macOS turns Ctrl+left into a *right* button event before Qt sees it (Apple QA1362),
  which `GLViewWidget` has no branch for. The capability was documented, implemented and
  unreachable — so the scatter could be turned and never moved. When binding a gesture, name the
  device the user actually has; and when a view can be moved out of frame, give it a way back
  (**Fit view**). The 3-D scatter this was found on is gone; the rule stands for the next view
  that can be moved out of frame — which INTERPRET's node tree is: it zooms on the wheel (a
  trackpad pinch arrives as one), pans on a plain drag, and carries its own **Whole tree** (and
  Esc) to get back.
- **The window's geometry and every split persist.** `window/geometry` in `QSettings`, saved in
  `closeEvent` and restored at the end of `MainWindow.__init__`. `restoreGeometry` rather than a
  remembered width and height, because it clamps to the available screen — the case that matters
  when a laptop is reopened without the monitor it was docked to — and its return value is
  checked, so an unreadable blob gives the default instead of leaving the window wherever Qt left
  it. **Every split remembers its pane widths and which side panels are shut**, per splitter, under
  `panes/<key>/` (`sizes`, `open` — the width a shut panel reopens at — and `shut`), written on
  every drag and fold and read by `GripSplitter.remember()` when the page is rebuilt — so a width
  the user set survives `_discard_content`, which rebuilds the page on every navigation, and the
  session. The navigation is one of them (`panes/navigation/`). The keys from before one scheme
  existed (`window/rail_split`, `window/nav_hidden`, `window/nav_width`,
  `splitters/interpret_filters_width` / `_folded`, `splitters/<key>`) are read once and removed. A
  saved split in which a pane that cannot fold is at zero is not restored. Other per-page state
  (selections, scroll positions) is not persisted. Guarded by `tests/gui/test_window_state.py`,
  which also scans for the Qt calls themselves — this rule was written down and believed for a long
  time while none of it existed — and `tests/gui/test_grip_splitter.py` (a second instance reads
  what the first left, through real `QSettings`).
- **A view open full window fits a laptop window, because nothing scrolls it.** The run page asks
  for no height with a view open (`AnalysisRunDetailPage.minimumSizeHint`), so a view whose rows
  want more than the window is clipped, not scrolled. Measured on INTERPRET at 1440×847 before the
  fix: a 764px column in 660px, the clip grid's bottom 94px unreachable, while a comment in
  `cluster_workspace.py` said the page scrolls. So a view's own floor is sized to fit: INTERPRET's
  tree gives height back to 120px before its clips drop under two rows, from size hints rather than
  from heights that have already overflowed. `tests/gui/test_laptop_layout.py` opens every view of
  a real run in the shell at 1440×847 and fails on any visible widget past its visible parent (a
  scroll area's contents excused only when that area can actually scroll).
- **A surface with nothing in it yet shows the one action, not the empty frame.** Where a page
  has a workspace that only makes sense once there is data — the audit tab's three columns, which
  need a reviewed sample before any of them says anything — the pre-data state is a single
  full-width call to action, and the workspace arrives with the data it is for. The failure it
  replaces is not a missing message but *four* regions each describing their own absence: a list
  with no rows, a prompt adrift in a void, two tables reading "not audited", a pair of empty
  axes. That reads as a page that failed to load. Distinct from `IndexTable`'s empty notice,
  which is the same rule for a collection: there the table *is* the page, so the notice replaces
  it in place. Key the switch on "has this ever had data", never on the current filter — an empty
  scope after an audit is an empty view, not an empty tab. The query AUDIT tab this was written
  for, and its guard, went with the one-by-one audit (2026-11-22); the rule stands for the next
  surface that has a pre-data state.
- **Geometry is measured after the layout has run, and a scrolled row has two layouts.** A
  widget's `x()` and `width()` are its *previous* values until Qt lays it out, and it has no
  reason to do so before the next paint — so anything that reads geometry to position something
  else (centring a divider, deciding which cells are on screen) must force the pass first. Inside
  a `widgetResizable` `QScrollArea` that means **both** levels: the scroll area sizes the scrolled
  widget, so activating only the inner layout measures it at its old width. `_body.adjustSize()`
  then `layout.activate()`; both are ~0.3ms. Traced twice in one session — a divider left 40px
  shy of centre, and 300 thumbnail decodes queued in the wrong order because every cell reported
  `width() == 0`. Guarded by `tests/gui/test_decode_priority.py` (the reuse guard,
  `test_strip_reuse.py`, went with the query confidence strip it measured, 2026-11).
- **Pages top-justify.** Content packs at the **top** of the content area, with empty
  space falling to the bottom — never spread down the page or floated in the middle. End
  a page's top-level layout with **`addStretch()`** (the standard idiom) so headings,
  forms, and content-hugging tables stay at the top. The exception is a page whose
  **primary content is a fill element** — a `QSplitter`, a `QScrollArea` grid, a large
  preview/canvas, or a table added with `addWidget(table, 1)` — which legitimately expands
  to fill instead; such a page needs no trailing stretch. A page with small content, no
  fill child, and no trailing stretch is the bug (it spreads once its table stops
  stretching). Every new content-area page ends with a trailing stretch unless it has a
  genuine fill child. **This applies to a pane, not only a page** — a splitter half, a tab
  body, anything handed more height than its hint. The audit tab's stats pane shipped
  without one and Qt spread the slack across its headings, laying out a 17px label at 324px.
  Measured by `tests/gui/layout_checks.py::assert_top_justified`, which is applied to every
  page in the smoke corpus (`PAGES` in `tests/gui/test_smoke_construct.py`). That corpus is not
  every page, which is why `NOT_YET_SWEPT` still lists this rule.
- **An axis states the domain of its quantity.** pyqtgraph gives every `PlotWidget` free pan
  and zoom, so a view that is never told otherwise scrolls to confidence −4 and precision 3
  — coordinates the quantity cannot take, on an axis that goes on labelling them. Call
  `bound_axes(plot, x=…, y=…)` (`gui/plots.py`) at every construction site, with `None` for
  an axis that genuinely has no domain (projection units), and `InfiniteLine.setBounds` on
  every movable line — a draggable cutoff is a value on its way to disk. Guarded by a
  fail-closed source scan (`tests/gui/test_plot_bounds.py`).
- **A mode is full-width unless it owns a rail.** Experiments / Active processes /
  Configuration / Model manager / Keypoints manager / Docs are banner modes that each render
  full-width with no sidebar; their detail is a
  **drill-in** sub-page, not a rail node. The left tree rail belongs to exactly two scopes,
  and the shell names which one the shared tree holds (`self._rail`: `"project"` |
  `"tools"` | `None`) so a refresh helper keys off the *rail on screen*, never off "the tree
  is visible" — a background import finishing while Quick tools is up must not rebuild the
  project rail into the tools tree. The scopes: a **project** (opened from Experiments),
  holding an **Overview** node and then that one experiment's sections — **Recording
  information / Recordings / Datasets / Models / Processing pipeline / Analysis /
  Comparisons**; the experiment's identity/settings live on its Overview, not a separate Data
  node, and Overview is a rail node of its own rather than a section.
  And **Quick tools**, holding one folder per registered tool with that tool's saved runs
  beneath it — a new tool appears there with no shell change, since it self-registers.
  Opening a project also makes it the **active** experiment (the target of a new run and the
  import queue). The **Model manager** mode is one machine-global page — **Global models**
  (imported/promoted, with metadata + a preview + a detail page), the **active experiment's**
  trained models, and the **Hugging Face** foundation models, with **＋ Distill new model** /
  **Import model** / **Restore from zip** on top. Datasets
  and distill runs are in the project rail, not linked from here.
- **Built once per scope, persistent within it.** The rail is (re)built when a project opens
  or Quick tools is entered, reading only *that* scope's lists — that experiment's (never an
  O(all-recordings) sweep), or one sweep of the tools workspace grouped by tool (never one
  listing per tool). Navigating between its nodes never rebuilds the rail, so expansion and
  scroll survive every click; a structural change (a run/recording/analysis added or deleted;
  a tool run launched or deleted) refreshes the rail on screen in place
  (`_refresh_active_experiment` / `_refresh_tools_rail`), preserving which folders were open —
  including one the user collapsed against the builder's default.
- **A project section reads its children when it opens, off the UI thread.** Building the
  project rail touches no disk: the folders come from the declaration (`gui/sections.py`),
  and `_fill_section` reads one section's children in a `BackgroundCall` when it is expanded
  — driven off `itemExpanded`, so the rail click, the post-refresh re-expand and
  `_go_to_section` all get it from the one toggle rule. Whether a node is a **folder** is
  therefore its declaration (`_FOLDER`), never `childCount()`: a closed section has no
  children, so a predicate that asks the children answers "not a folder" and the section
  never opens. The **▸** is declared for the same reason — Qt draws it only for an item that
  *has* children, so lazily-read sections would show none at all — and the fill hands that
  decision back to Qt once the contents are known, so a section that really is empty drops
  its arrow. The re-expand after a rebuild *is* the invalidation, so a refresh re-reads
  exactly the folders that were open and nothing else. Building the rail no longer makes any
  round trips on the UI thread, and a refresh costs only the open sections' own reads. On the
  network share these workspaces are held on, every one of those round trips froze the window,
  on each of the places a structural change refreshes (#316).
- **Identity** — the top bar shows the current context as plain text, e.g.
  `experiment_a  ›  recording_001`. That's where names are unambiguous.
- **The screen this is designed for is 1440x810 logical pixels.** Not a guess and not the panel
  size: what a layout has to fit is the resolution *after* OS scaling. A 1920x1080 laptop at
  Windows' own recommended 125% reports **1536x864** to the app — which is why that is the third
  most common desktop resolution worldwide (Statcounter). A MacBook at default scaling reports a
  little more than 1440x810. 1440x810
  sits under every machine in play, above the declining 1366x768, and is a round 16:9. Adobe
  Premiere Pro, the nearest commercial comparison, states a 1920x1080 minimum; Fiji, napari and
  EthoVision publish none at all. **A window does not get all of it** — a title bar and the menu
  bar take roughly 60px, so `USABLE_H = TARGET_H - 60` (against `TARGET_W, TARGET_H = 1440, 810`)
  is what a dialog's opening size is measured against. Three dialogs had drifted past it —
  1040x800, 1120x780, 1240x760 — because nothing checked, and a window that opens taller than the
  display puts its own bottom under the Dock. Dialogs scroll now, so this is the size they **open
  at**, not a floor they impose. The constants and the check live in
  `tests/gui/test_dialog_space.py`, whose comment defers to this bullet for the reasoning and the
  market share behind it.
- **A plot follows the chart theme, and sets its range after its items.** Build it with
  `plot_theme.themed_plot()` and colour its marks by role (`theme.identity(i)`, `theme.neutral`,
  `theme.handle`) — never `setBackground("w")` or a literal pen. This **reverses** the rule that
  stood here, "a plot is on white": in a dark window every plot was a lit panel with grey tick
  labels, and the eleven copies of `setBackground("w")` could only change one at a time. A themed
  plot is a card one step lighter than the window in either theme, turns off pyqtgraph's SI-prefix
  rewrite, and re-themes on a light/dark switch. The design and its reasoning are
  `docs/chart_design.md`; this half is swept by `a_plot_takes_its_colours_from_the_chart_theme`.
  `setXRange` + `disableAutoRange` **after** adding items, since a range set before them did not
  survive a splitter drag. Stacked plots sharing an x set the same left-axis width, or their
  plotting areas start at different x and a bar stops sitting under its own point. The *domain*
  half of this rule is the `bound_axes` bullet above and is swept; these two clauses are checked by
  nothing, and sit here rather than in `NOT_YET_SWEPT` because that ratchet is at its ceiling and
  raising it is the act it exists to make visible.
- **Forms**: `QFormLayout` (labels left, fields right). Tab navigates, Enter submits, Esc cancels.
  Invalid input shows a red border + text below — no inline animations or floating labels.
  `NOT_YET_SWEPT` carries this as *"Forms: QFormLayout, tab/enter/esc — no predicate yet"*.
- **Run config**: a modal `QDialog` for launching a new run (focuses submit / cancel); an inline
  `QWidget` for property edits. "Do something" stays distinct from "edit something". The modal half
  is swept by `a_dialog_opens_modally`; nothing asserts the inline half.
- **A window watched beside its page is non-modal**, and is the exception the modal rule names in
  `requirements._NON_MODAL_DIALOGS` with its reason: the clip window (`clip_popup.open_clip_window`
  — one per page, the next double-click shows its clip in the same window), a cluster's pop-out
  window (`cluster_window.ClusterWindows` — one per cluster, labelling through the page's own
  queue), and the enlarged pane. Each is the opening page's child, so it
  closes with the page; each opener goes through the one function that opens it, so "a second
  open raises the first" is written once (`test_every_opener_goes_through_the_one_non_modal_opener`
  scans for a page building the clip window itself). Review (2026-11-09) turned the clip window
  from modal to this: a modal one blocked the page it was opened to help judge.
- **External links look external.** A button that opens a browser ends with a trailing ` ↗` (a text
  glyph — consistent with "text labels only", not an icon set). Use the `_link_button` helper in
  `gui/pages.py`. Spot-checked on one page by
  `tests/gui/test_configuration_pages.py::test_hf_page_lists_external_links_and_gated_models`; no
  sweep reaches the rest.
- **Credential surfaces offer a way out.** Any page that stores a credential (e.g. the HF token)
  gives a clear replace / sign-out path, and surfaces when the value is pinned by an env var it
  cannot change. `NOT_YET_SWEPT` carries this as *"one surface exists; a rule needs a second"*.
- **No rounded corners, no shadows, no animations.** Shadows and animations are swept by
  `the_package_animates_nothing`; rounded corners are checked by nothing.

(Supersedes the earlier sketches — the Home/Experiment context-switch + breadcrumb/Back, the
three-group persistent tree, and a rail on *every* mode — with a top banner, drill-in
sub-pages, and a tree rail scoped to the open project or to Quick tools.)

### 3.2 The module-UI contract

Adding a module's UI must be additive — no shell surgery — the GUI analogue of
the backend registry (Tenet 1). A **gui-side module registry** lives entirely in
`detail/gui/` (backend modules stay Qt-free per the hard rule), keyed by
`module_id` and paired with the backend `PROCESSING_MODULES` entry for metadata
(`upstream`, `runs_key`, …).

Each entry is a `ModuleUI` supplying bespoke widgets:

```python
class ModuleUI(ABC):
    module_id: str                          # pairs with the backend module

    @property
    def label(self) -> str: ...
        # the picker's display name, from configs.describe, so the picker and the
        # module's docs page cannot disagree — they did, from a hand-written string here

    def config_widget(self) -> ConfigWidget: ...
        # the run-config form body; its interactive iterate-before-you-commit step
        # is a preview QDialog launched by a BUTTON on this form (detection ->
        # prompt-test F5; embedding -> render-preview F6; analysis query ->
        # "Mark examples…")

    def result_widget(self, experiment, run_record: dict) -> QWidget | None: ...
        # a view of the WHOLE run's results, for the run page (F7). Every analysis
        # module has one; no processing module does — `None`, the default.
        # The *record*, not a run handle — a completed run is a dict on disk.

    def preview_widget(self, recording, run_id: str) -> QWidget | None: ...
        # ONE recording's results (its frames, tracks or embeddings) — what a processing
        # module answers. The run page builds it for the recording selected in its list;
        # `None` means no per-recording view, and the page falls back to result_widget.

    def snapshot_widget(
        self, experiment, run_record: dict, snapshot_id: str
    ) -> QWidget | None: ...
        # the same results, read-only, as they were when a snapshot was pinned (§21).
        # `None` — the default — means this module authors nothing worth reading back,
        # and is what tells the run page to list its snapshots without a clickable cue.
```

**`config_widget` returns a `ConfigWidget`, not any `QWidget`.** The base is the contract:
`config()` / `validation_error()` for the launch, `load_config()` + a `config_model` for
reading a previous run's settings back (§3.2 "Start from"), and the
`resolve_recordings` / `apply_recordings` split that keeps a per-recording read off the UI
thread. A form that subclasses `QWidget` directly compiles and then silently has no
round trip and no off-thread path.

**The same shape, one registry over: the importer-UI contract.** A `StreamImporter`
declaring a `config_model` (§3 "Importer configuration") contributes an `ImporterUI` in
`gui/importers/`, keyed by `importer_id` and paired with the backend `STREAM_IMPORTERS`
entry:

```python
class ImporterUI(ABC):
    importer_id: str                        # pairs with the backend importer

    def config_band(self, parent=None) -> ConfigBand: ...
        # the band the Add-stream dialog shows under the importer picker.
        # `set_source(path)` reads the chosen file, `config()` returns the importer's
        # config dict, `problem()` is the sentence that refuses the import, and
        # `suggested_name()` offers a stream name read off the file.
```

`config_band` is the whole contract — one method. `snapshot_widget` belongs to `ModuleUI`
above, not here: an importer authors no run to pin.

**An analysis run page is a landing page, then full-window views.** Every analysis run opens on
a landing page — the run's facts, Notes (folded; opened from the shell's **Notes** button), the
live phase rows while it runs, its
snapshots, and **one card per result view**: the view's name, what it is for, and one status
line — then Open output folder / Cancel / Delete / Start new analysis. A card opens its view to
fill the window under a single row of tabs, one per view, to switch straight to another (none for
a one-view module), and `Read-only · <snapshot>` in snapshot mode. The way back is the shell's
path line, which reads `<experiment> › <run_id> › <view>` while a view is open, `<run_id>` a link
to the landing page: the page names its open view (`context_view` / `context_view_changed`) and
the shell draws the link and calls `close_view`. A `‹ <run_id>` button at the head of the row read
as one more view (review round 15). No
run facts, notes or tab bar inside a view; the shell's own context row stays.

```
landing                                      a view, full window
Analysis: an_32                              exp › an_32 › Audit        (an_32 is a link back)
                                             ╭Interpret╮╭Audit╮╭Browse╮╭Measures╮
Module / Status / Recordings                 ┌──────────────────────────────────────────────┐
(Notes, when opened)                         │                                              │
Results                                      │   the view, filling the window               │
┌ Interpret ›          ┐ ┌ Audit ›        ┐   │                                              │
│ Label behaviors: …   │ │ Check whether …│   │                                              │
│ 3 labels             │ │                │   └──────────────────────────────────────────────┘
└──────────────────────┘ └────────────────┘
[Open output folder] [Delete analysis]
```

The views are the module's to declare and the page's to host (`gui/result_views.py`, re-exported
from the registry):

```python
@dataclass(frozen=True)
class ResultView:
    name: str; purpose: str; widget: QWidget
    status: Callable[[], str] | None        # the card's status line — cheap, one small read

def result_views(widget, purpose="") -> list[ResultView]
    # widget.views() if it declares them, else one "Results" view with the module's purpose

class MultiViewWorkspace(QWidget):           # a result widget with several views
    view_changed = Signal(str)               # a hand-off inside it moves the page's row
    def views(self) -> list[ResultView]: ...
    def show_view(self, name: str) -> None: ...
    def current_view(self) -> str: ...
```

`ModuleUI` adds `purpose` (the one-view card's line) and `planned_views()` — the cards a run will
have, shown greyed with "Results appear when the run finishes" while it computes, when there is
no result widget to ask. The clustering and behavior-scan workspaces (the two
`MultiViewWorkspace`s) keep their tab widget with its bar hidden: the row is the one switch. Views are built **eagerly**, as the inline
result was, so cross-view hand-offs keep working; the open view is page state, like snapshot
mode, so a rebuild does not drop the user back on the landing page. With a view open the page
asks the shell's scroll for no height, so the view is held to the window rather than scrolled.
Esc is not bound to the way back. It was held back for AUDIT's review pane, which cleared its
selection of clips with it; review round 13 replaced that selection with click-to-remove, and since
round 14 Esc cancels a drag over the review grid — handled by the grid while it has focus, so
binding the way back to Esc elsewhere would still need to leave a drag in progress alone.

**Snapshot mode is a page mode, not a drill-in.** Double-clicking a snapshot row does not navigate:
the run detail page re-renders *itself* as that snapshot — a banner naming it on the landing page,
the snapshot list with the current row marked, and `snapshot_widget` in place of `result_widget`
behind the view cards. There is no second
read-only screen, because the workspaces are already generic over the run handle: a snapshot view
satisfies the same contract and refuses every write. The mode is **page state**, so it survives the
page rebuilding itself; a click-handler local would drop the user back to live values under the same
layout with no event. Controls that would change the snapshot are greyed with the reason (which is
*not* the same reason as a finished run — one is undone by Reopen, the other by leaving the
snapshot); selection, playback and navigation stay live.

**Selection-driven disk work runs off the UI thread.** The `config_widget`'s
`ConfigWidget` base splits any per-recording disk read a selection triggers (spec groups,
a tracking run's track ids) into `resolve_recordings(recordings) -> payload` (heavy, **Qt-free**,
run on a worker thread) + `apply_recordings(recordings, payload)` (UI-only, installs the
result). The launch dialog drives the split through a `BackgroundCall` with a generation
guard (latest selection wins); a form with no heavy read just overrides the synchronous
`on_recordings_changed`. This is the "no O(recordings) disk I/O on the UI thread" rule in
`CLAUDE.md` — a launch/selection surface that reads disk per recording on the UI thread is
the bug.

**A worker names the widget it reports to.** Every GUI thread is a `workers.GuiWorker`
(`BackgroundCall`, `StageProgressWorker`, and the per-module ones), constructed with a
required `receiver=` — the widget the result is *for*. A delivery arriving after that
widget's C++ side is destroyed is dropped rather than run. This is **not** what the
generation guard above does: a generation expresses *supersession* (a later selection owns
the view), and nothing bumps one when a page is torn down, because the shell tears pages
down with `deleteLater` and no page code runs. Both are needed, and a slot connected as a
closure — the usual `lambda result, g=generation:` — has no receiver for Qt to disconnect
on its own. `tests/gui/test_worker_lifetime.py` is the guard.

**`config()` and `load_config()` are inverses, over the fields that are settings.** A config
form has to give a configuration back as well as produce one — that is what the **Start from**
row is (`gui/start_from.py`, §3.2 below). `load_config` takes the plain config
dict `config()` returns; the launch dialog flattens a run record into that shape once
(`launch.seed_config`, `params` winning over the copies a runner promotes to the record's top
level) so a form never digs through two levels. Before that flattening every implementation
carried its own `params.get(k) or record.get(k)` fallback and no two agreed on which keys
survived a Refine, which is precisely why the pair could not be stated as a round trip.

**The Start from row is where a configuration comes from.** One `StartFromRow`
(`gui/start_from.py`) at the top of the launch dialog's form — `form.addRow("Start from:", …)`
— offering three sources in the order a person looks for them: **Defaults** (the pydantic
model's own, always first and always present), **baselines** shipped in `configs/baselines/`
(settings, never selections, so one travels between experiments and machines), and **previous
runs**, newest first. Picking one means "the form is now this", so it rebuilds rather than
overlaying — a form left part one thing and part another, with nothing on screen saying so, is
the bug. A run recorded before the form could take a config back is offered **disabled with the
reason**, not hidden ("grey out, don't hide"): a user who made that run and cannot find it here
would reasonably conclude the list is broken. The payload rides in `itemData` and the label is
display only — `currentText()` is a sentence, not an id, the `selected_model_id` lesson one
surface over. This replaced a *Refine* button gated on a run being non-mutable, which reached
2 of 9 analysis modules and no processing module at all.

A run config is two kinds of thing wearing one dict, and the pydantic field says which:

```python
window_s:     Field(..., json_schema_extra={"label": "Window"})                     # setting
embed_run_id: Field(..., json_schema_extra={"label": "Embedding run",
                                            "scope": "selection"})                  # selection
```

A **setting** is a number, a choice, a threshold — reusable, and restored. A **selection** names
an artifact (`embed_run_id`, `source_recording_ids`, `source_run_id`, `tracking_run_id`); the
dialog re-derives those from the current recording selection and re-asks, because a config that
carries a data reference is the thing that kills this feature in every tool that has tried it.
Classification lives on the field, never in a denylist inside a consumer — the drift rule
`runners.job_params` and `test_record_completeness` already exist to enforce.

`ConfigWidget.load_config` has a **generic base**: declare `config_model` on the form and every
field it built through `build_fields` is restored for free; a form with hand-built widgets
overrides, sets those, and calls `super().load_config(config)` for the rest. `MethodFields` has
the matching `set_params`, and `_pydantic_form.set_any_field_value` is the setter twin of
`any_field_value` — so no surface hand-rolls a dispatch that covers only the widget shapes its
author remembered.

Guarded by `tests/gui/test_config_round_trip.py`, which perturbs a form's own config and
asserts the form moves to it. It is data-driven over the module registries and the launch
dialogs, and a surface that can *produce* a config but not read one back fails rather than
being skipped.

**The form's granular knobs collapse, and the model says which.** A config form is built
on `gui/forms.py::ConfigForm` — a main `QFormLayout` plus **one** `AdvancedSection` beneath
it, opened by a single toggle under a divider. Opening it reveals everything it holds at
once; a form with a lot of knobs sub-divides them with a titled rule per group, which is a
heading, not a second thing to click. Which parameters are advanced is declared on the
pydantic field (`json_schema_extra={"advanced": True}`, or a group title such as
`{"advanced": "Clustering"}`) beside its `label` / `unit` / `ui_max`; `ConfigForm.addRow`
recovers the field from the row's `FieldLabel` and routes it, so a form's rows are written
exactly as they were. A row with no model field behind it (a compute target, a tracker's
knobs, a trainer's hyperparameters) passes `advanced=True`, or a group title, at the call
site. The section opens collapsed, counts what it hides, marks what has changed since the
form opened, and opens itself on first show when it holds a change — so a **Refine** never
buries the parameter that made the old run different.
`tests/gui/test_advanced_form.py` is fail-closed: it builds every config form and fails if a
flagged field is sitting in a main body, so a form that opts out of `ConfigForm` cannot
silently ignore the flag.

**One launch surface per module — including analysis modules (§16).** A module is
reached through exactly one surface: its `config_widget()` in the launch dialog. Any
"scrub / test / preview / tune before committing" step is a preview `QDialog` launched
from a button on that form — **not** a new sidebar node or standalone authoring page.
This holds for processing (§6.5) *and* analysis (§16) modules alike; a bespoke page that
duplicates a config form is a bug. When adding a module's interaction, copy how
detection/embedding already do it rather than inventing a parallel surface.

The **shell stays generic**: it enumerates modules, renders the recording-selection
+ launch dialog (hosting the module's `config_widget`), runs the job on a
`QThread` with a progress bar, maintains the runs index, and builds the
run-summary scaffold (hosting `preview_widget` for the selected recording, or `result_widget`
for a whole-run view). Launching calls
`Experiment.run_processing(...)`; nothing module-specific lives in the shell.

**One approved exception: a stateful, compute-triggering `result_widget`.** For almost
every module the `result_widget` is a *static* view of a finished run. An **authored run**
(Query, temporal clustering) is the deliberate exception: its result widget is a workspace whose
operations re-run compute in place on the same open run. This is allowed because the run is
*authored data*, not a computed artifact — the whole point of the module is the loop. The
carve-out is narrow: the mutable affordances live **inside** the result widget, so the only shell
change is letting `AnalysisRunDetailPage` render results while the run status is `open` (not just
`complete`/`partial`).
Do **not** read this as
license for a second launch surface or a bespoke authoring page — the launch is still the
one `config_widget`, and a new module needs an equally strong "authored, not computed"
justification before its result view may trigger compute.

**It is a lifecycle, not one exception (§21).** DESIGN_PLAN §21 turns the shape into a lifecycle: an **authored run** is one whose compute finishes while the *run*
stays open for a user to work on, edited in place rather than seeded into a new run. Two
consequences for a module UI:

- **The module declares it, not the UI.** `AnalysisModule.mutable` is the flag, so the API
  and CLI read the same fact the shell does. The GUI `ModuleUI.mutable` mirrors it
  (`mutable = QueryModule.mutable`) rather than restating it.
- **The shell asks the flag, never a list of statuses.** `AnalysisRunDetailPage` renders
  results for a terminal status, *plus* `AUTHORED_STATUSES` when the module is mutable. A
  hardcoded tuple of status strings goes stale the day a module grows a lifecycle, and
  nothing fails when it does — which is why §21 asks for the flag specifically.

What still holds from the narrow version: the affordances live inside the result widget, the
launch is still the one `config_widget`, and "authored, not computed" still has to be argued
for a new module rather than assumed.

**Every number is a shared box, and comes from the model where one exists.** Never build a
bare `QSpinBox` / `QDoubleSpinBox`: Qt pads the display to exactly `decimals` places and its
validator then rejects any longer candidate, so a 4-decimal confidence box shows `0.5000`
and silently swallows every digit typed into it. `detail/gui/spin.py`'s `IntSpin` /
`FloatSpin` trim the display, keep the width a full-precision entry needs, select on focus,
and ignore the wheel unless focused. A float box's `kind` (`FRACTION` / `SECONDS` / `PLAIN`)
fixes its decimals and step in one place.

Reach for those directly only when nothing describes the parameter (an interactive tool's
own view knob). When a pydantic config model does, build it from the model:

| what you have | use |
|---|---|
| a form whose rows *are* the model's scalar fields | `build_fields(model)` |
| one field, in a form with rows the model doesn't describe | `field_widget(model, name)` |
| an `X \| None` field whose `None` means "work it out for me" | `optional_spin(model, name, auto_text)` + `optional_value` |

`field_widget` is what stops the all-or-nothing choice that produced 47 hand-built boxes:
a form that interleaves a run picker or a preview button no longer has to abandon the
generated path for *every* row. A widget's range, step, precision and help all come from the
field, so it cannot disagree with the CLI or the parameter reference. Where a form wants a
narrower view than the field allows, declare it on the field (`ui_max` for a spin ceiling,
`ui_step`, `ui` to override the inferred kind) rather than passing a number at one call site
— a second surface for the same parameter then can't pick a different one. Tracker
hyperparameters, which are not pydantic fields, take their bounds from
`configs/tracker_params.py` for the same reason.

Two tests hold the line: a source scan for bare constructions, and a walk over every built
config form asserting its spin boxes are the shared ones (`tests/gui/test_spin_widgets.py`).

**The compute target is one widget.** `ComputeTargetRow` (`detail/gui/compute_target.py`)
owns the backend combo, the CUDA device number, and the greying that ties them together;
`add_to(form, prefix)` places its two rows, `set_backends` narrows it to what a chosen model
supports, `compute_target()` reads it. Nine copies had drifted apart before it existed, one
of which had lost the greying entirely.

`set_backends` offers only what the model **and this machine** have in common
(`core/utils.available_backends`). Passing a component's own list straight through put
`cuda` in front of a Mac user for a CUDA-only model, and Qt made it stick:
`best_supported_backend(["cuda"])` falls back to `"cpu"`, and `setCurrentText("cpu")` on a
combo holding only `"cuda"` is a **no-op** — so the form sat on a device that did not exist
here with its index box live, and the run died on `requested cuda device(s) [0] not
available` once the dialog had closed.

**An editable page can be asked what it would lose.** A page with a Save button drops
whatever was typed the moment you click elsewhere; the metadata table is where that cost
the most. `gui/editing.py` is the mechanism rather than a per-page fix: implement
`unsaved_summary` / `save_edits` / `discard_edits`, and the shell prompts Save / Discard /
Cancel before leaving.

The check runs at the **navigation entry points** (`_on_tree_click`, `_switch_mode`, the
`_go_to_*` drill-ins, `_open_experiment`, `closeEvent`), *not* in `_show`. `_show` is the
single funnel every page swap passes through and looks like the obvious hook, but it has
~39 callers and by the time it runs the replacement page is built and the rail selection
has moved — Cancel could no longer leave you where you were. Two consequences worth
keeping: pending work is computed by **comparing against the store** (entering edit mode
and changing nothing is not unsaved work, and a prompt that fires on that is one people
learn to dismiss), and a **failing save stays put** and says why, since proceeding would
discard the edits the user asked to keep. Modal dialogs are out — they already have Cancel
semantics. Guarded by `tests/gui/test_editing.py`: every entry point must ask, and every
content page with a Save-like button must implement the protocol.

**Model availability: one rule for every picker.** A model list never offers something that
cannot run here without saying so. `gui/model_availability.py` is the single entry point —
`apply_availability(combo, role)` marks every row with its reason, opens the form on one
that can run, and returns the sentence to show when none can:

- reasons are **backend first** (`— needs CUDA`), then disk (`— not installed`, from
  `api.models.is_installed`, never `readiness()`, which reaches the network on the UI
  thread). On a Mac, downloading a CUDA-only model would not help, so its hardware is the
  useful thing to say. A model with sizes counts as installed if *any* size is; the variant
  row (`mark_variants`) carries the rest.
- nothing runnable → a non-selectable `(no model available)` row becomes current, an amber
  note appears, and `validation_error()` refuses the run with the same sentence. A greyed
  row sitting in a closed combo reads as *selected*, which is the lie this exists to remove.
- an id lives in the combo's `itemData` and is read back through `selected_model_id`, because
  a label now carries its reason — `currentText()` is a label, not an id.

Detection's two pickers merged for this: with the list split behind a mode switch, "is there
anything I can run?" could only be answered by toggling it, and the answer on a Mac was yes,
under the branch you were not looking at. GUI tests run on a pinned "everything installed,
every backend present" machine (`tests/gui/conftest.py::capable_machine`) so only the
availability tests depend on the host; they opt out by requesting `real_machine`.

**Adding a module's GUI = register a `ModuleUI` under its `module_id`.** The new
module then appears in the launch picker with its config + result UI, no shell
changes — the cheap-to-add goal realized at the GUI layer, with bespoke UI per
module. *(MVP: bespoke `ModuleUI` for the built processing modules; a generic
auto-form fallback for modules without one is deferred.)*

## 4. MVP screens → API

Each screen: **purpose**, the **actions/values → API call**, and key **states**.
`(new)` flagged a then-missing API method/accessor; all are now built (status
list at the end of this section).

### 4.1 Experiments *(banner mode; `HomePage`)*
- **Purpose:** pick, open, create or import an experiment.
- **→ API:** **one card per experiment** (`HomePage`, not a table — a work surface's job here is
  recognition, per §3.1) ← `Experiment.list_all()` — the managed workspace (storage `default`
  root, §2). No folder picker (experiments live in the managed workspace, not arbitrary folders);
  `[New experiment]` → 4.2; `[Import experiment]` → `Experiment.import_archive`. Machine-local
  setup is the **Configuration** banner mode, not a dialog off this page; the readiness
  checklist lives there too.
- **States:** empty (no experiments → no cards; the two buttons below the list are the way in).

### 4.2 New-experiment dialog *(modal)*
- **Purpose:** create an experiment.
- **→ API:** name (required) + optional description → `Experiment.create(name)` then `set_metadata(description=…)`; created in the managed workspace (no folder picker, no video). On success, the new experiment opens on its Overview (4.3); recordings are added from its Recordings section (F2).

### 4.3 Experiment overview *(the rail's Overview node; it absorbed the former Data tab)*
- **Purpose:** view/edit experiment-level data (edit-where-it-lives), and a summary of the sections.
- **→ API:** name/description (editable, `[Save]`) → `set_metadata`; the folder (read-only); `[Export experiment…]`; one summary row per section with a live count and an **Open →**; `[Delete experiment]` (confirm) → `Experiment.delete` → the Experiments list (§1.1). There are no per-stage defaults (a configuration comes from the launch dialog's Start from row, §3.2), and grouping variables are edited on the **Recording information** page (§1.3).
- **States:** clean / edited (Save).

### 4.4 Experiment · Recordings
- **Purpose:** manage the cohort's recordings.
- **→ API:** recordings table (Recording · Status · Source · Parts · Streams · Field of view · Detections) ← `Experiment.recordings` (+ per-recording `Recording` props); `[Add recordings]` → 4.12 → the import queue → `add_recording`; `[Import bundle]` (§1.3); `[Import behavior labels]`; `[Cancel imports]`; double-click a row → Recording detail (4.8).
- **States:** reading (rows are read off the UI thread); empty (a notice naming Add recordings).

### 4.5 Experiment · Processing pipeline *(the runs index)*
- **Purpose:** runs index + launch.
- **→ API:** runs table, oldest first in store order like every rail section (Run · Module · Recordings · Status · Started) ← experiment run index `(new)`; `[New run]` → 4.13; double-click a row → Run detail (4.14).
- **States:** empty; the caption counts running and failed runs.

### 4.6 Model manager *(banner mode, machine-global — §5)*
- **Purpose:** one catalog of all available models — the Hugging Face foundation models
  (registries) + imported and promoted weights. On top: `[＋ Distill new model]` /
  `[Import model]` / `[Restore from zip]`. Experiment-free; a model's detail page is a
  **drill-in**, shown full-width, not a rail node. The per-experiment *trained* models (§1.2)
  live in the project rail; this page shows the active experiment's in a table of their own
  (`EXPERIMENT_MODEL_COLUMNS`, shared with `ModelsPage`), whose rows open the model inside its
  experiment. Every table here holds `ModelManagerPage.TABLE_ROWS` (4) and scrolls past it.
  The Hugging Face table lists `api.models.suggested_models()` first, then installed, then the
  rest.
- **Tables:** **Global models** (Name · From project(s) · Made by · Created · Kind; `[Export]` /
  `[Delete]` under it); **Models in <experiment>** (`EXPERIMENT_MODEL_COLUMNS`); **Hugging Face**
  (Id · Size / variant · Role · Architecture · Source · Status · Classes — one row per model and
  variant, because install state is per variant; `[Check access]` / `[Install]` under it).
- **→ API:** `api.models.list_models` / `model_readiness` / `install_model` / `is_installed`
  / `import_model` / `import_model_archive` / `export_model` / `remove_imported_model` (the
  experiment-free catalog; `Experiment`'s model methods delegate to it).

### 4.7 *(retired — there is no Preferences dialog)*
The machine-local credential settings this specified live in the **Configuration** mode's
**Hugging Face** step (`pages.HuggingFacePage` → `credentials.set_hf_token`), a drill-in
sub-page described in §3.1. `Preferences` appears nowhere in `detail/gui/`; this entry
contradicted §3.1 inside one file for as long as both were here. Section number kept so the
cross-references above do not silently renumber.

### 4.8 Recording detail — Overview tab *(content-area detail)*
- **Purpose:** view/edit a recording's video + metadata.
- **→ API:** video-parts table ← `Recording.video_parts` (GUI must not read `recording.yaml` directly, Tenet 2); `[Add video part]` → 4.11 → `add_video_part`; recording name edit → `set_name`; **Run history** table (stage, id, "From run" = `processing_run_id`, status) — double-click a row → its result view: an experiment-backed run opens its experiment run (4.14); a per-recording run launched via the CLI/API (no experiment record) opens a recording-scoped result view (4.8a) keyed on `(recording, stage, run_id)`; `[Delete recording]` (confirm) → `Experiment.delete_recording` → recordings list (§1.1).
- **States:** empty (no video → prompt add).

### 4.8a Recording run detail — per-recording result *(content-area detail)*
- **Purpose:** view one per-recording stage run (`det_01` / `track_01` / `emb_01`)
  that was launched per recording via the CLI/API and so has **no** experiment run
  record — opened from §4.8's Run history. The experiment-backed case opens §4.14
  instead.
- **→ API:** `RecordingRunDetailPage(experiment, recording_id, module_id, run_id)` reads the
  stage record's status from `getattr(recording, f"{module_id}_runs")`; on `complete`
  it asks the module for `preview_widget(recording, run_id)` — the same per-recording view
  the run page's results list builds, so no new preview class; on failure it shows the
  reason read from the run dir's `job.json`; a missing run shows "(run not found)".
- **States:** complete (result view) / failed (reason) / not found.

### 4.9 Recording detail — Scene geometry section
- **Purpose:** reach the one spatial-setup surface. The section summarises what the recording
  has (field of view, distance calibration, regions, points) and opens 4.10 to edit any of it.
- **→ API:** `Recording.arenas` / `rois` / `points` / calibration for the summary; the button
  **Edit scene geometry…** opens `arena.SceneGeometryDialog`
  (`RecordingDetailPage._open_scene_geometry`, the one call site).
- **Naming:** the user-facing term is **field of view**; `arena` survives as the *kind id* and
  the storage directory (`SpatialKind(id="arena", label="Field of view")`). A single field of
  view per recording — a deliberate reversal (2026-08-07) — so this is not a list to manage.

### 4.9b Recording detail — Streams section *(content-area, on the recording page)*
- **Purpose:** import + align + inspect time-series streams (§3).
- **→ API:** streams table (Name · Type · Rows · Offset (s)) ← `Recording.streams`; `[Add stream]` → 4.15 → `add_stream`; `[Preview]` → 4.16 → `stream_preview` / `frame_timestamps`; `[Sync to video]` → 4.17 → `set_stream_offset` / `frame_t_ns`; `[Edit offset]` → `set_stream_offset`; `[Delete stream]` (confirm) → `delete_stream` (§1.1).
- **States:** empty (no streams → just `[Add stream]`). **`[Sync to video]` and `[Edit offset]` are enabled per the *selected* stream's clock** — only an external-clock stream (`stream_seconds`, e.g. MED-PC) needs alignment; a video-anchored stream (`video_frames` / `video_seconds`, e.g. behavior labels) is already aligned, so both are disabled with a reason (`clocks.needs_alignment`, DESIGN_PLAN §3 "Stream alignment"). Sync additionally needs a video part.

### 4.10 Scene-geometry dialog *(one canvas, modal)*
- **Purpose:** every piece of a recording's spatial setup on **one frame** —
  `arena.SceneGeometryDialog`, covering all four `SPATIAL_KINDS`: **Field of view**,
  **Distance calibration**, **Regions of interest**, **Points of interest**.
- **Why one canvas, not four editors.** These were four page sections, five dialogs and four
  editing idioms for one thing a person sets up in one sitting — twenty of the recording page's
  twenty-six buttons were spatial. The separate editors could not see each other, so an ROI was
  drawn on a frame that did not show the field of view, which is why a read-only *Preview
  spatial setup* dialog had to exist at all. Here the other definitions are always drawn as
  context, so preview is this dialog's resting state (select nothing) rather than a fifth
  surface to keep in step — and the frame decodes once instead of four times.
- **→ API:** frame scrub → `get_frame`; select a definition to edit, siblings stay dimmed, a
  checkbox hides one; **Save** writes every kind at once and keeps the dialog open, greyed
  while nothing has changed — the dialog's one sign of pending edits — and **Close** asks
  before discarding them. Saving used to close the dialog, which fought the *Copy to other
  recordings…* / *Import from…* / *Export as CSV…* actions beside it: they refuse while edits
  are pending. A field of view a tracking run depends on is shown, drawn, and **not**
  editable — the run recorded that polygon, so changing it would make the run's record a lie.
- **States:** locked field of view (with the reason); validation → `QMessageBox`.

### 4.11 Add-video dialog *(modal)*
- **→ API:** file picker (exists-check) → `add_video_part`.

### 4.12 New-recording / add-recordings dialog *(modal)*
- **→ API:** one row per recording (name + optional video, plus per-row grouping-variable
  cells). **Create hands the batch to the app import queue (`gui/import_queue.py`) and closes
  immediately** — it does not block on a progress bar, and there is no rollback-on-cancel,
  because nothing is half-created to roll back. The whole bout appears at once in the Recordings
  table: empty recordings `ready`, video ones `queued` → `importing` → `ready` / `failed`.
  §1.3 has the full state model, including what "Cancel imports" leaves behind and how a crash
  reconciles orphans.
- This entry previously specified a blocking bar with stage text and rollback, via
  `workers.create_recordings`. That function and `workers.BulkCreateRecordingsWorker` had no
  caller, and were removed in #509: a second import → render mapping, the shape §6 forbids.

### 4.13 Launch-run dialog *(modal — the new experiment-level run path)*
- **Purpose:** configure + launch a processing run over a recording selection.
- **→ API:**
  - module picker ← `PROCESSING_MODULES` (registry, §6.5) paired with its `ModuleUI` (§3.2)
  - recording multi-select ← `Experiment.recordings`, each annotated with **eligibility** (is `module.upstream` complete for it? derived from per-recording run status) — ineligible recordings disabled with a reason
  - config ← the module's `config_widget()`, which embeds its iteration tool (prompt-test F5 / render-preview F6) and pre-fills the settled config
  - compute target (backend / devices)
  - `[Run]` → `Experiment.run_processing(module_id, recording_ids, config)` `(new)` → fans out to per-recording `Recording.run` (§6.5)
- **States:** validation (no/zero-eligible recordings, invalid config → inline errors).

### 4.14 Run detail — summary / management *(content-area detail)*
- **Purpose:** progress + results + manage one run.
- **→ API:**
  - header: module, config summary, recording list, overall status ← run index record `(new)`
  - running: `run_activity.LiveRunPanel` — an aggregate bar + status + **one row per unit**
    (`gui/run_units.py`), fed by `RunHandle.progress()` on a `QThread`; `[Cancel run]` →
    `Experiment.cancel_run`. **One shape, a unit per run kind**: processing loops
    *recordings*, analysis loops *phases*, distill loops *stages* (each with its own inner
    unit — frames labelled, epochs trained), an export loops *targets* — each surface says what
    its units are, and flattening them into one stream of text is what made them all look alike.
    Above `GROUP_ABOVE` = 8 units the finished and the waiting collapse to counts, because at 21
    recordings only two rows are news (the one running and the one that failed); below it
    everything lists, because at three a count is worse than the list it replaces. Every widget
    in the panel has a **pinned height** — it updates ~twice a second for hours above ordinary
    content, so any height derived from text makes the whole page twitch. There is no activity
    log on the page: it held a timestamped band once, and the unit rows replaced it because the
    band only restated them a line at a time. `activity.csv` is still written for every run
    (`storage/activity.py`) — the performance export reads it and a bug report attaches it.
    A new run kind supplies its unit noun and its rows; it does not add a second progress
    mapping (§3.2's one `event → render` rule).
  - complete: `RunResults` — the run's recordings as one list (Recording · Run id · Status) that
    is also the picker: select a row and that recording's results appear under it from the
    module's `preview_widget(recording, run_id)` (the detection / tracking / keypoint viewers,
    the embedding PCA point cloud); double-click a row → that recording's detail, 4.8. A module
    with no per-recording view falls back to `result_widget(experiment, run_record)`. A
    completed run also offers **Run <next stage> →** (F4).
  - failed / cancelled: per-recording error (from the run record) + `[Retry N incomplete
    recording(s)]` → `Experiment.retry_run(run_id)` → `_start_live` streams it on this page
  - `[Delete run]` (terminal runs only, confirm) → `Experiment.delete_run` → runs index (§1.1)
- **States:** pending / running / complete / partial / failed; per-recording breakdown.

**Retry is in-place, not a pre-filled re-launch.** This screen previously specified
`[Re-run] → 4.13 pre-filled`, i.e. seeding a *new* run from the failed one. That splits one
logical result set across two records: `per_recording` already maps each recording to its own
stage run id, so a run record is a **map with holes in it**, and the natural repair is to
fill the holes — same run id, rolling up to `complete` once every recording is. So the button
gates on the *incomplete count*, not the run-wide status (a cancelled run with unfinished
recordings gets it too), and its work goes through `BackgroundCall`: retry validates every
incomplete recording and spawns a coordinator, which is O(recordings) disk I/O and must not run
on the UI thread (§1.1). Seeding a fresh run remains the right shape for the **Start from** row
(`gui/start_from.py`), where the point is a **different** config.

### 4.15 Add-stream dialog *(modal)*
- **Purpose:** import one stream (§3).
- **→ API:** name + importer picker ← `STREAM_IMPORTERS`; a data-file picker + one file picker per the importer's `aux_inputs` (MED-PC: the `.MPC` program); `[Import stream]` → `add_stream(name, raw_path, importer, aux_paths)`. Synchronous (files are small); errors → `QMessageBox.critical`.

### 4.16 Stream-preview dialog *(modal)*
- **Purpose:** inspect a stream + confirm its alignment.
- **→ API:** an event-raster timeline (one lane per event type) linked to a scrubbable video frame, the alignment `offset_ns` applied so it doubles as a sync check ← `stream_preview` (events/lanes) + `get_frame` / `frame_timestamps`. Falls back to a stream-local timeline when the recording has no video.

### 4.17 Sync-to-video dialog *(modal)*
- **Purpose:** set a stream's alignment offset by clicking the frame where it starts.
- **→ API:** a whole-recording frame scrubber + an anchor-event picker; `[Set offset]` → `offset_ns = frame_t_ns(frame) − anchor_t_ns` → `set_stream_offset`. Needs a video part.
- **Reached only for an external-clock stream** (`clocks.needs_alignment`); a video-anchored stream never opens this dialog (its `[Sync to video]` is disabled). The anchor picker reads `t_ns` + `label` by index, so it tolerates either row shape (events / interval).


### API additions §4 needed (status)
All **built**:
1. `Experiment.run_processing(...)` + runs index + per-recording breakdown — **built** (`storage/runs.py`; a detached coordinator + `AttachedRun` handle, §15).
2. `Experiment.set_metadata` (name/description; `description` stored field) — **built**.
3. `Recording.part_frame_counts` (per-part counts → global→(part, local) for the arena slider) — **built**.
4. `Recording.video_parts` read accessor (Tenet 2) — **built**.
5. `Recording.set_name` (recording metadata edit) — **built**.
6. **Experiments-list discovery** — `Experiment.list_all()` over the managed workspace (storage `default` root, §2; `storage/locations.py`) — **built**.

## 5. Package layout

> **This section used to describe a 7-file package and a planned `pages/` / `dialogs/` split.**
> Neither is the shipped layout: `detail/gui/` is flat modules plus three code subpackages,
> and the flat arrangement is what the shared-leaf rules below depend on — a helper inside a
> package whose `__init__` imports the world is an import cycle whose failure depends on order
> (`tests/gui/test_import_cycles.py`). The graduation is not planned; the grouping below is by
> **role**, which is what a reader needs. `docs/package_map.md` is the package-wide map.

```
detail/gui/
├── __main__.py          # entry point: python -m detail.gui
├── app.py               # MainWindow shell: banner modes, both rails, content routing
├── pages.py             # the §4 pages + the shared `_table` / `IndexTable` builders
├── sections.py          # EXPERIMENT_SECTIONS — the rail's nodes, declared once
│
│  # --- shared leaves: the widgets and rules every surface inherits -------------
├── style.py             # the one Qt style: Fusion, scrollbars without arrow buttons
├── dialog.py            # the Dialog base: body() / set_buttons(), scroll + top-justify
├── scroll.py            # PageScroll — the page scrolls, the window never grows
├── sizing.py            # lines() (this package's em), mono/emphasis fonts, fit_list_height
├── tables.py            # sorting, resizing, row identity (stamp_row_id / row_id), heights
├── forms.py             # ConfigForm + AdvancedSection
├── flow.py              # a row of controls that wraps instead of being squeezed
├── spin.py              # IntSpin / FloatSpin / ThresholdSlider — never a bare QSpinBox
├── _pydantic_form.py    # field_widget / build_fields / optional_spin — a form from a config model
├── plots.py             # bound_axes and the shared plot conventions
├── plot_theme.py        # the pyqtgraph / QPainter adapter for the chart design language
├── zoom.py              # ZoomWindow + OverviewBar — one horizontal zoom for every widget
├── collapse.py          # CollapsibleBand — a foldable section with a drag handle
├── grip_splitter.py     # GripSplitter — side panels that fold to a labelled strip, remembered
├── busy.py              # run_busy / run_progress / decode_overlay — off-thread loads
├── workers.py           # GuiWorker (every thread names its receiver=) + BackgroundCall
├── lifetime.py          # whether a Qt object is still there to be talked to
├── ui_thread_gc.py      # the cyclic garbage collector runs on the UI thread only
├── throttle.py          # a drag that reports at most every 50 ms, then once on release
├── menus.py             # a menu built for one opening is deleted when it closes
├── timefmt.py           # one way to print a recording time
├── image_view.py        # the one pane that shows a picture; `zoomable` has no default
├── full_frame.py        # a thumbnail that opens its frame full size and zoomable
├── scrubber.py          # FrameScrubber — the shared frame/playback control
├── frame_source.py      # why a scrub position produced no pixels, and what to say
├── decode_order.py      # which thumbnails to decode first
├── clip_grid.py         # one thumbnail's pool decode, and how many decimals a score needs
├── cluster_strip.py     # the clip strip a cluster/behaviour is recognised from
├── quick_preview.py     # the small player a strip puts in a thumbnail's place
├── clip_popup.py        # the window a quick preview opens into
├── clip_panes.py        # the clip window's picture: footage, masks alone, pose alone
├── clip_pane_window.py  # one of those side panes, large, with its own playback and export
├── clip_strips.py       # the clip window's per-frame strips under the slider
├── advanced_player.py   # the clip window's Advanced player: footage, masks, keypoints, strips
├── skeleton_points.py   # a choice of a skeleton's points (Trail points)
├── show_menu.py         # what a montage's tiles show — the Show ▾ menu
├── animation_export.py  # Export video… / Export GIF… for a picture that plays
├── montage_export.py    # Export ▾ and pick mode — one export for every clip grid
├── recording_items.py   # filling a recording picker, and reading it back
├── confidence.py        # the confidence-view control every detection result carries
├── method_form.py       # a registry method's parameters, from its own config model
├── editing.py           # the unsaved-edits protocol the shell prompts on
├── notes.py             # the Notes box every entry page carries, saved as you type
├── selectable.py        # all user-facing text is selectable
├── status_color.py      # the status palette and the one way to apply it
├── requirements.py      # the checked structural requirements; generates its own doc page
│
│  # --- feature surfaces ---------------------------------------------------------
├── launch.py            # 4.13 launch dialog; hosts ModuleUI.config_widget
├── start_from.py        # the Start from row: defaults / baselines / previous runs
├── run_activity.py      # LiveRunPanel — the live run page
├── run_units.py         # one row per unit, a unit per run kind, GROUP_ABOVE
├── unit_rows.py         # the rows themselves: name, bar, status
├── result_views.py      # ResultView / MultiViewWorkspace — an analysis run's cards and views
├── active_processes.py  # the Active processes banner mode
├── import_queue.py      # the non-blocking recording-import queue (§1.3)
├── preview_jobs.py      # video preview builds, queued in the background (§3)
├── node_cache_jobs.py   # node-cache builds that outlive the page that started them (§22)
├── arena.py             # SceneGeometryDialog — 4.10, all four SPATIAL_KINDS
├── streams.py           # 4.15–4.17 add-stream / preview / sync-to-video
├── preview.py           # prompt-test / render-preview dialogs
├── run_preview.py       # detection / tracking / input-image / embedding viewers
├── behavior_query.py    # the Query surface + the shared _ClipPlayer
├── example_spread.py    # how a behavior's example clips spread over the cohort
├── cluster_workspace.py # the clustering result workspace
├── node_workspace.py    # INTERPRET: pick a cluster, watch every clip of it, label it
├── node_icicle.py       # INTERPRET's node tree, drawn as an icicle
├── live_montage.py      # every clip of a cluster, all playing, under the tree
├── cluster_filters.py   # the cluster page's filter dock
├── match_key.py         # the match key atop the filter dock, and the label swatches
├── cluster_window.py    # one cluster's clips in a window of their own
├── label_panel.py       # INTERPRET's label bar, and the label arithmetic every surface shares
├── label_history.py     # INTERPRET's undo: one history of label changes per run
├── density_map.py       # where in the frame: the 16×16 position grid and a region picker
├── measurements.py      # the shared measurements band (both result pages)
├── spatial_map.py       # the shared spatial map (both result pages)
├── comparison.py        # §24 comparisons
├── export.py            # the one Export control, placed by the shell
├── fit.py               # §12 distill: dataset + train dialogs
├── training_curve.py    # the distill training curve
├── snapshots.py         # §21 pinned artifacts on screen
├── query_review.py      # §19/§20 AUDIT on a query run: every clip a cutoff keeps
├── review_marks.py      # AUDIT's ordered writes and undo, shared by both run kinds
├── label_check.py       # §19/§22 clustering AUDIT: the list of labels/clusters + the page
├── label_clean.py       # label_check's review pane: every clip of a cluster, click to remove, Finish
├── audit_table.py       # the one colour INTERPRET rings a reviewed cluster in
├── node_audit.py        # AUDIT's figure (a reviewed cluster's census) on INTERPRET's card
├── label_browse.py      # clustering BROWSE
├── keypoints_manager.py # the Keypoints manager banner mode: the skeleton library (§26)
├── skeleton_editor.py   # the skeleton editor, a skeleton drawn on a plain canvas (§26)
├── example_set_picker.py# the example-set picker wherever a keypoint model needs examples (§26)
├── keypoint_labeling.py # the labeling screen: marking an example set's frames (§26)
├── keypoint_run_preview.py # the keypoint run viewers (§26)
├── keypoint_test.py     # Test cut-off… / Test keypoints — testing a keypoint teacher (§12)
├── keypoint_distill.py  # the keypoint half of ＋ Distill new model (§12)
├── readiness_guide.py   # Configuration mode: the setup checklist
├── cuda_doctor.py       # Configuration mode: the GPU/CUDA doctor
├── model_availability.py# apply_availability / selected_model_id — every model picker
├── compute_target.py    # ComputeTargetRow — never a fresh backend combo + device spin
├── help_ui.py           # the in-app help popover and its ? button
├── version_history.py   # Version History: the running build and what changed
│
│  # --- registries: all module/importer/tool Qt lives here ----------------------
├── importers/           # ImporterUI + ConfigBand + IMPORTER_GUIS (§3.2)
├── modules/             # ModuleUI + GUI_MODULES (§3.2) — per-module config + result
├── tools/               # ToolUI + register_tool_ui (DESIGN_PLAN.md §27, Tools)
└── resources/           # app_icon.png
```

`widgets/image_grid.py` was in the planned tree and is gone for good: the thumbnail grid it
named was deleted once nothing showed it, and `image_view.py` is the single pane that replaced
the ad-hoc picture widgets.

- The **shell** (`app.py`) is module-agnostic; everything stage-specific lives in
  `gui/modules/`, registered by `module_id` and paired with the backend
  `PROCESSING_MODULES` entry (§3.2). Adding a module's UI = a file in
  `gui/modules/` + one `register_module_ui` call (`register_analysis_module_ui` for an
  analysis module, which lands in the separate `ANALYSIS_GUI_MODULES`).
- `gui/` is the **only** package allowed Qt imports (CLAUDE.md hard rule); it
  reaches the backend exclusively through `Experiment` / `Recording` (Tenet 2).

**The Add-stream dialog is one surface, driven by the importer** (§3, 4.15). The **data
file comes first** and `can_import` identifies it; a file nothing claims falls back to the
single importer declaring a `config_model`, because a CSV needing a column mapping is by
definition one the sniffing cannot recognize. Below the picker sit two bands: the
importer's own (its `ConfigBand`, absent for an importer with no parameters) and the
dialog's own — name, *Labels are*, and **Alignment** — each shown or greyed from a
**declared** property (`role == "behavior"`, `clocks.needs_alignment(clock)`), never a
hand-written list of importer ids, so a new importer inherits the right rows the day it
registers.

**Alignment happens inside the import**, not as an errand afterwards. *Sync to video…*
opens `SyncStreamDialog` from the form against the parsed-but-unwritten rows and returns an
offset; the same dialog, entered by stream name from the Recording page, writes one for a
stream that already exists. Adding a *second* surface for either is the bug.

**No mockup.** The earlier `scripts/gui_mockup.py` exploration has been removed;
the design here is settled, so `detail/gui/` is built directly against it.

**Build order.** The GUI-driving backend additions have landed (see §4 status),
so: shell (Home/Experiment contexts, content routing, workspace listing) → launch dialog
+ `ModuleUI` registry + run-detail (the `run_processing` surface) → per-module
`ModuleUI`s + the arena canvas.

---

## Open decisions / API gaps

- **Built (see §4 status):** `Experiment.run_processing` + runs index, `Experiment.set_metadata`,
  `Recording.video_parts` / `part_frame_counts` / `set_name`.
- **Experiments workspace** — experiments live under the storage `default` root (§2, `storage/locations.py`); the Experiments mode lists them via `Experiment.list_all()`. **Built.** Full storage.yaml config / cloud `StorageLocation` still pending.
- **Per-field editability** — principle settled (edit raw/metadata where it
  lives; run outputs read-only). Exact editable fields enumerated per screen in §4.
- **MVP result depth** — status + per-recording counts + prompt/renderer grids;
  richer overlay browsing deferred.
