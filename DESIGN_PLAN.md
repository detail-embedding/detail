# DETAIL Design Plan

This document is the approved design for the DETAIL infrastructure redesign. Components are added here after discussion and approval. It is the source of truth for implementation.

---


## Definitions

**Experiment** — A complete cohort study. One experiment contains all recordings, all processing outputs, all analysis runs, all trained models, and all datasets for a given experiment. The unit of sharing and archival.

**Recording** — One experimental unit: one animal, one experimental bout. A recording is a named entity carrying **grouping variables** (§11) and, *optionally*, **video** and **streams** — none is required to create one (a stream-only or metadata-only recording is valid; each stage simply offers itself only when its input is present). When video is present it may span multiple files (parts) if the recording was split, but all parts are continuous and share the same physical setup, fps, resolution, arena definitions, and stream alignment. If the animal, condition, or arena changes, it is a new recording.

**Part** — One video file within a recording. Parts are ordered and together constitute the full recording. A recording with a single video file has one part.

**Arena** — A named polygon (normalized coordinates) that defines the valid detection region within a recording's recording space. Used to filter out-of-bounds detections during processing. A recording may have multiple named arenas. Arenas may differ between parts if the camera moved.

**ROI (Region of Interest)** — A named polygon (normalized coordinates) used during behavioral analysis to spatially tag or query detections (e.g., "what poses are most common near the lever?"). Independent from arenas — ROIs are analysis-time definitions, arenas are processing-time filters.

**Detection Run** — One execution of a detection model over some or all parts of a recording, using a specific prompt and compute configuration. Multiple detection runs may exist per recording (different models, prompts, or confidence thresholds).

**Analysis Run** — One execution of an analysis module (location export, query, query evaluation, temporal clustering — §16) over a specified set of recordings and parameters. Multiple analysis runs may exist per module. Analysis runs live at the experiment level and reference recordings by ID.

**Grouping Variables** — Recording-level metadata (animal ID, treatment, condition, etc.) used to compare groups during analysis. Defined at the experiment level as a CSV with one row per recording.

**Skeleton** — The named body points a keypoints run places, with the edges that join them and the left/right symmetry pairs among them (§26). Names are labels; nothing is inferred from them. A skeleton is copied into every example set and every run that uses it.

**Example set** — An experiment-level collection of decoded frames with the user's marks on them (today: one clicked point per skeleton point, or "not visible"), the input a patch-matching keypoint method learns from (§26). Its sibling with `purpose: audit` holds the frames that *score* a run and never build one.

**Heading definition** — Two sets of skeleton point names, front and back; the body heading on a frame is the direction from the mean of the back set to the mean of the front set. Chosen per embedding run for egocentric alignment (§14); never part of the skeleton.

---

## Status

| Component | Status |
|-----------|--------|
| Experiment Creation | Approved |
| Recording Structure | Approved |
| Storage Abstraction | Approved; **not built** beyond local paths. `storage/locations.py` resolves the `default` alias's local `root`, `model_cache` and `preview_cache`, plus the `video_roots` / `export_roots` aliases; the `StorageLocation` abstraction (`video_uri` / `fs_store`), cloud aliases (`type: s3` / `gdrive`), `volume_label` and zarr sharding are design only. |
| Model Abstraction | Approved |
| Executor / Cloud Compute | Approved |
| Detection Jobs | Approved |
| Detection Outputs | Approved |
| Spatial Definitions | Approved |
| Grouping Variables | Approved |
| Custom Model Training | Approved |
| Tracking | Approved |
| Embedding Generation | Approved |
| Python API and CLI | Approved |
| Processing Modules (§6.5) | Approved |
| Module Families & Analysis Modules (§16) | **Implemented.** Four modules in `ANALYSIS_MODULES`: `location_export`, `query`, `query_eval`, `temporal_clustering`. Both kinds once listed as pending now exist — `query` / `temporal_clustering` / `query_eval` pool across the cohort, and `query_eval` joins a behavior stream (§3). |
| Annotation Import/Export (§17) | **COCO masks export built** — the `masks` export target on a detection run (`api/export_targets.py`), through the export backbone. Not built: MOTS / MOT for tracked runs, the `with_frames` bundle, and import (Phase 2). |
| Export and provenance (§18) | **Built** — `report/` (`style.py`, `tables.py`, `figures.py`, `exporter.py`) and `analysis/provenance.py` (`capture` / `drift`); §24's comparison export is the writer. A Query run's §19 artifact carries a captured provenance block; a temporal clustering run's block is empty, and only a saved comparison is checked for drift (`Comparison.drift`). |
| Behavior Output (§19) | **Built, and produced.** `storage/behavior_output.py` (the three files, writer + tolerant reader + discovery-by-scanning), `storage/frame_view.py` (the per-frame expansion §23 measures against), `storage/behavior_audit.py` (the optional audit), `storage/append_log.py` (the shared append primitive), and the `origin` boundary that keeps a re-ingested prediction from being read as ground truth. **Two producers emit it**: §20's Query and §22's temporal clustering. Each writes the frame view too (#237), so a windowed §23 measure answers on any of them; a producer that skipped it would leave every sub-window question refusing on its runs alone, which is what the parity guard exists to stop. The measures and comparison layers above it are specified in §23 / §24 and built. |
| Query (§20) | **Built** — `pipeline/query.py`, `api/query.py` (an authored run, §21), `storage/query_runs.py`, the `query_methods/` registry, CLI `analysis query` / `query-calibrate` / `export-query`, and the GUI calibration workspace. It is the first producer of the §19 artifact. One method, **`soft_knn`**, the default — chosen on measured cost, while which kind of method retrieves better stayed unmeasured. Its AUDIT tab (§19 reliability, `gui/audit_tab.py`) is built: blind, randomized, 25 clips a batch, precision with a Wilson interval and a per-recording split. §22's workspace has its own (`gui/label_check.py`); the two share the §19 statistics, not a widget. |
| Authored Runs (§21) | **Built.** `api/authored_run.py` carries the contract — open vs partial, one writer, `finalize()` / `reopen()`, refuse-if-finalized, reproject-on-edit. `mutable` is declared on `AnalysisModule` so the API and CLI know it as well as the GUI; two modules set it (`query`, `temporal_clustering`). Of the shared pieces recorded as extraction targets, the run handle base `AuthoredRun` was extracted; `Experiment.run_authored` and a shared `MutableRunWorkspace` were not (each module keeps its own launch path and workspace). |
| Behavior Discovery (§22) | **Temporal built** — `pipeline/temporal_clustering.py`, the `cluster_methods/` registry, CLI `analysis temporal-clustering` / `temporal-interpret`, and the INTERPRET workspace; it emits the §19 artifact. The pose half of this section is design, not built: its protocol and an empty method registry (`POSE_CLUSTER_METHODS`) are in the tree, and no pose clustering module is registered. **Node-centric interpretation built** (labels on tree nodes, no granularity, the node cache listing every node's clips with their typicality and a rebuild, the icicle shaded by the cluster-view filters, the label bar, and a live montage of every clip from the preview cache). Labels are stored things with ids (one label across several clusters, the parent rule, rename and merge). AUDIT is a list of labels and clusters beside a review pane, with the figures in INTERPRET's label bar (§19, §22); on a temporal run every clip of a cluster is reviewed, the wrong ones marked (removed or moved, `clip_edits`, faded in place on INTERPRET too) and the finish recorded as a census (`clip_review.csv`, `finish_review`) that gives the cluster's exact precision, built; and BROWSE is the labelled-data browser (filters, ethogram, list). Not yet built from §22's text: the descriptor histograms behind *Measurements that set it apart*, arena-normalized maps, BROWSE's location filter, and re-attaching audit answers after a branch is sub-labelled. |
| Measures (§23) | **Built, on all three surfaces** — `analysis/measures.py` (eleven measures, each declaring how it meets a window boundary and what unit its value carries) and `analysis/windows.py` (whole / absolute / normalized / sliding, the model in `configs/windows.py`). Pure windowed queries over the §19 artifact; a windowed denominator needs the frame view's observed mask, which every producer now writes. `noise_time` is deliberately absent (see *The measures*). Reachable as `Experiment.measures(run_id)`, as `detail measures export` / `list`, and as the Measures panel on an analysis run's page. |
| Comparison (§24) | **Built, on all three surfaces** — `analysis/compare.py`, `analysis/comparison_report.py`, `analysis/node_measures.py`, `configs/comparison.py`, `storage/comparisons.py`, `api/comparison.py`, CLI `comparison run` / `export` / `list` / `delete`. Descriptive only, saved with its outputs; the aggregation path is declared, never defaulted, the design (between / paired / trend / profile) is resolved from the chosen column and recorded, the node source screens a clustering run's unlabelled nodes, and every ranking ships with its permutation floor. The locations source compares distance, time moving, speed and body shape read from a location-export run, each measure by its own test. The GUI section is **static by design**: a saved comparison holds its outputs; the live screen over a clustering run's clusters is INTERPRET's Differences line (§22), which reads a saved node comparison back. |
| Surface Visibility (§25) | **Built** (#203) — `detail/surfaces.py`, sections declared once in `gui/sections.py`, fifteen enumeration points, `DETAIL_SHOW_ALL` escape hatch. Nothing is hidden: the capabilities it hid were removed from the tree on 2026-11-10; the mechanism stays. Nothing is staged. This sentence is a copy of `surfaces.HIDDEN` and had already drifted once, so `tests/test_surfaces.py` now checks it against the table. Hiding a *module or method* still needs the docs generator and the static `mkdocs.yml` nav taught about it. |
| Keypoints (§26) | **Built; audit not kept** — spec rewritten 2026-10-07 from the keypoints review, replacing the integration branch `keypoints` (#626) rather than merging it: `dinov3_match` on a shared patch-matching engine plus a distilled YOLO-pose student; two stages, `keypoint_detect → keypoint_track`; alignment is §14's, distill §12's, the shared model layer §5's. Every planned piece landed except the audit, which was parked by the user (2026-10-09) and its API and CLI later removed, so the verdict reads built without it; the detect form's *Test keypoints* (no cut-off field, by the user's decision) has landed too. **The 2026-10-21 revision is built:** detect stores unassigned instances per part on a detection run (`keypoint_detect`'s `upstream` is `"detection"`); keypoint track assigns them to a tracking run's subjects by mask (`pipeline/pose_match.py`; §26 *Assignment*). |
| Tools (§27) | **Built** — the `Tool` protocol, the `TOOLS` registry, `storage/tool_runs.py` (machine-global `tools/<tool_id>/run_NNN/tool.yaml`), `api/tools.py`, the GUI's Quick tools mode, and CLI `detail tools`. One tool is registered: Quick segment. Each tool's CLI command is hand-written. |
| GUI | Built: shell + navigation, launch/run-progress, per-module config UIs, prompt-test (matrix), arena editor (time-varying), detection + tracking run previews; embedding input-image constructors + scrubber; embedding result view (Stage 1: a PCA point cloud beside the clicked point's frame and model input); deletion at every level (experiment/recording/run/arena) with a cohesive strategy (gui_design §1.1); run-table navigation + rich run labels; smooth import progress; the model manager + readiness/CUDA-doctor onboarding; every analysis module's config and result UI; the training surface (datasets, trained models, distill); the §21 authored-run workspaces (query calibration, temporal INTERPRET); the §19 reliability audit, which both of them share by emitting the artifact; and §24's comparison screen (`gui/comparison.py`: a list page, a launch dialog and a saved result). Pending: embedding result view Stage 2 (linked scrub). |



---

## Design Tenets

These are the non-negotiable principles that govern all implementation decisions. When in doubt, refer here first.

---

### Architecture

**1. Protocol + registry, never hard-coded implementations.**
Every major component — detection models, trainers, trackers, embedding models, mask transforms — is hidden behind an abstract protocol and registered by string ID. Callers reference `model_id: "sam3"` or `tracker_id: "anchor_tracker"`, never import or instantiate concrete classes directly. Adding a new backend means implementing the protocol and calling `register_X` — nothing else changes.

**2. The Python API is the only boundary between frontend and backend.**
The GUI, CLI, and research scripts all call `Experiment` and `Recording` methods. Nothing outside the Python API reads or writes `recording.yaml`, mutates zarr stores, constructs job dataclasses, or resolves paths. If a GUI widget is manipulating a path string directly, that logic belongs in the API layer.

**3. Core has zero Qt dependency.**
Pipeline logic, executors, protocols, and the Python API import nothing from PyQt/PySide. `LocalExecutor` runs jobs as subprocesses. The GUI's thin progress workers (`gui/workers.py`: `RunProgressWorker`, `FitProgressWorker`) read a run handle's `ProgressEvent`s off the UI thread and forward them as Qt signals. A violation is any `from PyQt6 import ...` outside `gui/`.

**4. Jobs are serializable data structures, not function calls.**
A job is a plain dataclass expressible as JSON. Everything needed to reconstruct and run it — model ID, hyperparams, compute target, input references — is in the job. No lambdas, no closures, no runtime state. This enables cloud submission, re-running a failed or stopped job from its record, and exact reproduction of any run.

---

### Data and Storage

**5. Source data is immutable. Each stage writes to its own directory.**
Detection zarrs are never modified after the detection run completes. Tracking writes `object_id` assignments to `tracking/track_01/assignments/` — never back to `detections/`. Embedding writes to `embeddings/`. If a stage needs to annotate earlier output, it writes a parallel artifact alongside it, not into it.

**6. No absolute paths anywhere in experiment files.**
`recording.yaml`, `config.yaml`, `track.yaml`, `embedding.yaml`, and all other experiment-owned files store paths relative to the experiment root or recording directory. Machine-specific paths live only in `~/.config/detail/storage.yaml`. A violation is any string containing `/Users/`, `/home/` or `C:\` in an experiment file.

**7. Schema version on every output file.**
Every YAML and zarr group written by the pipeline carries `schema_version: 1`. This is the first field written, before any computed content. No exceptions. Without it there is no safe migration path when formats change.

**8. Timestamps are `int64` nanoseconds from the start of the first video part.**
The field is always named `t_ns` and always typed `int64`. Float seconds are never used as a time reference in pipeline data. Integer nanosecond comparison is exact; float seconds accumulate drift. "The start of the first video part" is its first **shown** frame, not the container's clock: each part's map is part-local from 0 (the first shown frame's presentation time is subtracted) and placed in the recording by an integer-ns offset — the summed measured durations of the parts before it. A map has exactly one row per frame a decoder shows, so map index `i` is the `i`-th frame `cv2.VideoCapture` returns; a sequential pass that finds the two counts disagree raises rather than guess (§3). For a recording with no video, the reference is the stream's own native clock (there is no video frame to anchor to); stream alignment offsets are simply not meaningful until video is added.

**9. Transparent lineage.**
Every output artifact records the IDs of the inputs and the full parameters that produced it. `tracking/track_01/track.yaml` records `detection_run_id` and the full `hyperparams` dict. `embeddings/emb_01/embedding.yaml` records `tracking_run_id`, `model_id`, `model_variant`, the named `constructor` (if any), and the resolved `renderer_spec` (reproduces the run on its own). A run must be fully reproducible from its own metadata alone.

**The recorded value is the *resolved* one.** An input left unset is resolved in `validate()` and written back, so the record carries the answer rather than the request — `location_analysis` resolves a tracking spec into a per-recording map, `detection` writes back `weights_source` / `batch_size` / `model_variant`, `tracking` refuses an ambiguous multi-arena request rather than picking one. A default resolved at run time from mutable or order-dependent state (the *first* behavior stream, `next(iter(VARIANTS))`, a disk-discovered recording set) makes the same recorded config mean something different later, which is not reproducibility however complete the record looks. Where the choice is genuinely ambiguous, raise instead of picking.

This applies to **analysis runs** (§16), which arrived after this tenet and did not inherit it: `analysis/<id>/analysis.yaml`'s `params` is derived from the job (`runners.job_params`), never hand-listed, and the job carries every config field by name. Guarded by `tests/configs/test_record_completeness.py`. The one deliberate exception is the experiment-level `runs/<id>/run.yaml`, which stores a **spec** rather than a resolution because each recording resolves it to a different run id — the resolution lives in each recording's own record.

---

### Execution and I/O

**10. Validate everything before writing anything.**
All pre-run checks complete before any output directory is created or any compute begins. A job that fails validation leaves no partial output on disk. All failures raise specific, actionable exceptions. The `confirm_fn` callback is reserved only for genuine required-configuration decisions where there is no correct default (e.g., conflicting FPS values where the user must specify which is authoritative). It is not a mechanism for soft-confirming questionable inputs.

**11. Video is always read sequentially. Never load all frames.**
Frame reading within a part is always forward-sequential. No `cap.set(CAP_PROP_POS_FRAMES, ...)` during batch processing loops — only at part start, for lightweight preview and prompt-test operations, and in the **labeling sampler** (`pipeline/_dataset_common.py::exact_sampled_frames`), which reads a sparse random sample by seeking to each index. That last exception is allowed because a labeling frame's index is a *label*, not a join key, so it may seek — but each frame is decoded **exactly** (`read_frame_at` with the part's timestamp map, never cv2's `CAP_PROP_POS_FRAMES`), and a frame the decoder cannot land on is redrawn rather than labelled under another frame's index. It is **not** allowed for pipeline data that must stay frame-exact — detection, embedding, and the run-export dataset source keep forward-only `sequential_frames` so each frame lines up with its zarr row. Never accumulate an unbounded / whole-video list of frames; a **bounded** buffer of up to `batch_size` frames for batched GPU inference is allowed (memory stays O(batch_size), not O(video)). This is a hard memory constraint for long recordings.

**12. Zarr writes use buffer-flush. Status reflects partial progress.**
Outputs are written to zarr incrementally in fixed-size buffers, never accumulated in memory and dumped at the end. Each flush grows the array along axis 0 — `n = arr.shape[0]; arr.resize((n + m, *arr.shape[1:])); arr[n:] = buf` (zarr v3 has no `Array.append`). Flush buffer sizes are a whole multiple of the chunk size so a completed flush writes whole chunks. A long job that fails or is stopped is **re-run**, not continued: a processing run's retry (`Experiment.retry_run`, §15) runs each incomplete recording again, whole, under a fresh stage run id, so nothing appends to a half-written store. Only these pick up where they stopped: tracking resumes at a part boundary (§13, `part_handoff.json`); a cancelled or failed training or distill resumes from its last checkpoint (`Experiment.resume_training` / `resume_distill`, §12); and Query's runner skips search shards already on disk (§20). Job status has a `"partial"` state for a run that was interrupted and kept what it had written. The storage layer targets **zarr v3** (zarr-python ≥ 3.2); see "Zarr write idiom (v3)" in Section 9 for the canonical create/grow/encode pattern.

**13. Backend-specific imports are lazy and module-local.**
`from ultralytics import YOLO`, `import torch`, and any other backend import appears only inside the methods of the concrete implementation class, never at module top level. Excluding a backend means not importing its module — the registry never sees it and nothing else breaks.

**14. Credentials never enter experiment files.**
HF tokens, API keys, and any other credentials live in `~/.config/detail/credentials.yaml` or environment variables. If a credential appears in `config.yaml` or `recording.yaml` it is a bug.

**15. Frame reading uses `cv2.VideoCapture` and `PyAV` only. No FFmpeg.**
FFmpeg CLI and any library that shells out to it are prohibited for any operation that touches individual frames. FFmpeg's frame extraction is not guaranteed frame-accurate — it can drop frames, misalign PTS, or silently re-encode. `cv2.VideoCapture` is used for the pipeline stages' sequential frame reading and for writing. `PyAV` is used for container metadata and PTS extraction, and for decoding a frame found by its presentation time rather than by a forward walk — the preview and prompt-test reads (`video.read_frame_at`), the labeling sampler, and the display-only video preview proxy (§3). This applies to detection, embedding, and any future stage that reads video frames. A violation is any `subprocess.run(["ffmpeg", ...])` or equivalent in pipeline code.

---

### Quality

**16. Full test coverage is required for all pipeline code.**
Every protocol implementation, validation check, zarr schema, API method, and resume path must have tests. Silent wrong results are worse than crashes — an undetected frame drop or incorrect embedding corrupts an entire experiment. Specific requirements:
- Every validation check has a test that triggers it and verifies the correct exception
- Every zarr output has shape and dtype assertion tests
- Resume paths are tested by simulating a mid-run interruption and verifying the resumed output is identical to an uninterrupted run
- Protocol compliance tests run against every registered implementation (model, trainer, tracker, embedding model, transform)
- Integration tests use real temporary directories — no mocking of the storage layer

**Shared test fixtures (`tests/conftest.py`):**

All pipeline tests draw from a common set of fixtures. These must be implemented before any pipeline tests are written. Every agent task that writes tests must use these fixtures — never define equivalent setup inline.

The sample below is the contract — each fixture's name, scope and result — not its current source. Three of them (`experiment_dir`, `recording_dir`, `make_embedding_run`) now delegate their bodies to `tests/experiments.py` so a **wider-scoped** fixture can build the same thing without the setup being written out twice, which the paragraph above forbids: a module- or session-scoped fixture cannot request a function-scoped one, and a test that needs one expensive artifact built once and copied per test has no other seam. The fixtures remain the interface, in `tests/conftest.py`, unchanged in behaviour.

```python
import pytest
import numpy as np
import zarr
import yaml
from pathlib import Path


@pytest.fixture
def experiment_dir(tmp_path) -> Path:
    """Minimal valid experiment directory."""
    for d in ("recordings", "analysis", "models", "datasets"):
        (tmp_path / d).mkdir()
    (tmp_path / "config.yaml").write_text(yaml.dump({
        "schema_version": 1,
        "name": "test_experiment",
        "created_at": "2026-01-01T00:00:00Z",
    }))
    return tmp_path


@pytest.fixture
def recording_dir(experiment_dir) -> Path:
    """Minimal valid recording directory with recording.yaml and all stage subdirs."""
    s = experiment_dir / "recordings" / "recording_001"
    for d in ("video/timestamps/generated", "detections", "tracking", "embeddings"):
        (s / d).mkdir(parents=True)
    (s / "recording.yaml").write_text(yaml.dump({
        "schema_version": 1,
        "id": "recording_001",
        "name": "recording_001",
        "created_at": "2026-01-01T00:00:00Z",
        "video": {
            "storage": "default",
            "fps_container": 30.0,
            "fps_actual": 30.0,
            "fps_source": "container",
            "resolution": [640, 480],
            "total_duration_s": 10.0,
            "files": [{
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }],
        },
    }))
    return s


@pytest.fixture
def sample_detection_zarr(tmp_path) -> Path:
    """Detection zarr with N=100 rows and correct schema. Use for storage and pipeline tests."""
    N = 100
    path = tmp_path / "part_001.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs["schema_version"] = 1
    root.create_array("frame_idx",     data=np.arange(N, dtype=np.int32),              chunks=(10000,))
    root.create_array("t_ns",          data=np.arange(N, dtype=np.int64) * 33_333_333, chunks=(10000,))
    root.create_array("detection_idx", data=np.zeros(N, dtype=np.int32),               chunks=(10000,))
    root.create_array("confidence",    data=np.full(N, 0.9, dtype=np.float32),         chunks=(10000,))
    root.create_array("bbox",          data=np.zeros((N, 4), dtype=np.float32),        chunks=(10000, 4))
    # vlen columns: native v3 variable-length UTF-8 (StringDType); create then assign
    # (create_array forbids passing both data= and dtype=). mask_rle holds the ASCII
    # COCO-RLE counts string — see "Mask encoding" in Section 9.
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
```

**Schema assertion pattern** — every zarr output test must assert column names, dtypes, and chunk sizes explicitly:

```python
def test_detection_zarr_schema(sample_detection_zarr):
    root = zarr.open_group(str(sample_detection_zarr), mode="r")
    assert set(root.array_keys()) == {
        "frame_idx", "t_ns", "detection_idx", "class_name", "confidence", "bbox", "mask_rle"
    }
    assert root["frame_idx"].dtype == np.int32
    assert root["t_ns"].dtype == np.int64
    assert root["confidence"].dtype == np.float32
    assert root["bbox"].shape[1] == 4
    assert root["frame_idx"].chunks == (10000,)
    assert root["mask_rle"].chunks == (1000,)
```

**17. Fail informatively. Warnings are almost never the right choice.**
If a condition deviates from expected behavior, raise a specific exception with a clear, actionable message that tells the caller exactly what is wrong and what to do. Do not warn and proceed — warnings are ignored in scripts and habituate users to dismissing real problems. The only legitimate use of a non-fatal diagnostic is for conditions that are purely observational and cannot affect correctness (e.g., a disk space advisory after a run completes successfully). Every entry in a validation table defaults to a hard error. A "Warn" classification must be explicitly justified in the design. If user judgment is genuinely required, the call fails with a message telling the user which parameter to set — it does not auto-proceed.

---

## 1. Experiment Creation

### What happens when a user creates a new experiment

The user provides a name and selects a parent directory (or storage location). The tool creates the following structure:

```
my_experiment/
├── config.yaml
├── stream_presets.yaml   # saved importer setups (§3), if any have been saved
├── heading_definitions.yaml  # saved headings for alignment (§14), if any have been saved
├── examples/             # example and audit sets (§26): ex_NN/{examples.yaml, frames/}
├── recordings/
├── analysis/
├── models/
├── datasets/
├── runs/                 # experiment-level processing run records (§15): run_NNN/
├── comparisons/          # saved comparisons (§24): cmp_NN/, created by the first one
├── distill/              # keypoint distill runs (§12): distill_NNN/, created by the first one
└── exports/              # export run records, not the exported files (§18), created by the first one
```

`Experiment.create` makes `recordings/`, `analysis/`, `models/`, `datasets/` and `runs/`
(`storage/experiment.py::SUBDIRS`); every other folder above is created by the first thing that
writes into it.

The experiment directory is fully portable. It contains no absolute paths, no machine-specific configuration, and no credentials.

### `config.yaml`

Written at experiment creation with identity only.

*A top-level `defaults:` block was specified here and removed.* It was built, tested, and written by nothing: `set_defaults` had no caller outside its own tests, and the GUI rendered a "Per-stage defaults" table that was always empty. Pre-filling a form is now done by **Start from** (`gui/start_from.py`), which offers the model's defaults, named baselines from `configs/baselines/` (the mechanism is built; none ship yet, see that folder's README), and the settings of any previous run of that module. The difference is that those are all *concrete* — a file or a run somebody made — where `defaults:` was an abstract preference a user had to author and maintain with no surface for doing so.

What `Experiment.create` writes (`storage/experiment.py::create_experiment`); `Experiment.set_metadata` edits `name` and can add a `description`:

```yaml
schema_version: 1
name: "my_experiment"
created_at: "2026-05-21T10:00:00Z"
```

### What is explicitly excluded from `config.yaml`

- Video paths (recordings are discovered by scanning `recordings/`)
- Output paths (all outputs are relative to experiment root)
- HF token or any credentials
- Pipeline parameters of any kind (each run carries its full config in its own `job.json`)
- Machine-specific paths of any kind

### Credentials

The Hugging Face token and any other credentials live outside the experiment:

- File: `~/.config/detail/credentials.yaml`
- Or environment variable: `DETAIL_HF_TOKEN`

**Resolution order.** A credential is resolved by the first source that provides
it, in this order:

1. The matching environment variable (e.g. `DETAIL_HF_TOKEN`) — for headless
   and cloud runs.
2. `~/.config/detail/credentials.yaml`.
3. For the HF token only: the user's existing Hugging Face login
   (`huggingface_hub.get_token()` — `HF_TOKEN` / `~/.cache/huggingface/token`), so
   a user who has already run `huggingface-cli login` needs no extra setup.

`credentials.yaml` is keyed by provider so other secrets (cloud API keys, §6)
live alongside the HF token, and is written with `0600` permissions:

```yaml
huggingface: "hf_xxx"
# modal: "..."   # future providers add a key here
```

The API exposes `set_hf_token(token)` (validates via `whoami`, returns the
account name); resolution is read-only everywhere else. A credential is never
read from or written to a experiment file (Tenet 14).

---

## 2. Storage Abstraction

### Problem

Video files and output data may live on a local SSD, an external hard drive, or cloud storage (S3, GCS, GDrive). The experiment config must remain portable across machines regardless of where data is stored.

> **Built vs. spec.** Only local paths are built (`storage/locations.py`): the `default`
> alias's `root`, `model_cache` and `preview_cache`, and the `video_roots` / `export_roots`
> aliases. Cloud aliases (`type: s3` / `gdrive`), `volume_label`, the `StorageLocation`
> abstraction below and zarr sharding are design, not built.

### Two-level config

**Level 1 — experiment config** (`config.yaml`): references storage locations by name only. Never contains absolute paths.

**Level 2 — machine-local storage config** (`~/.config/detail/storage.yaml`): maps names to actual paths or URIs on the current machine. Never committed or shared.

```yaml
# ~/.config/detail/storage.yaml
storage:
  default:
    type: local
    root: "~/Documents/DetailExperiments"
    model_cache: "~/.cache/detail/models"  # optional; auto-downloaded weights
    preview_cache: "~/.cache/detail/previews"  # optional; display-only video previews (§3)

  lab_drive:
    type: local
    root: "/Volumes/LaCie/rodent_data"
    volume_label: "LaCie"    # optional: warn if drive not mounted

  s3_archive:
    type: s3
    root: "s3://my-bucket/detail"
    options:
      profile: "lab-aws"

  gdrive:
    type: gdrive
    root: "gdrive://folder-id/detail"

# Machine-local roots for reference-in-place video (§3). An alias here is what a part's
# `source: {root, rel}` resolves against, so the experiment file holds no machine path
# (Tenet 6) and moving a whole video library is one re-point of the alias.
video_roots:
  lab_nas: "/Volumes/nas/behaviour/2026"
  scratch: "~/Movies/inbox"
```

If a storage alias cannot be resolved (drive not mounted, credentials missing), the tool reports which alias failed — not a generic "file not found."

The `default` alias's optional `model_cache` key is where auto-downloaded model weights live (the `model_cache_dir()` single-owner location, §5), letting several machines share one weight cache. Resolution order: the `DETAIL_MODEL_DIR` env var, then `default.model_cache`, then the `~/.cache/detail/models/` default. The resolved path is passed to the backend explicitly (`cache_dir=`) — never steered via `HF_HUB_CACHE`/`HF_HOME`, which are snapshotted at import time (§5).

The `default` alias's optional `preview_cache` key is where this machine keeps its display-only video previews (§3 *Video preview cache*). Resolution order: the `DETAIL_PREVIEW_DIR` env var, then `default.preview_cache`, then `~/.cache/detail/previews/` (`storage/locations.py::preview_cache_dir`, resolved on every call).

**`storage.yaml` has one reader, and it parses the file once.** `locations.read_storage()` is what every lookup above goes through (the experiments root, the model cache, the preview cache, a `video_roots` / `export_roots` alias). It is cached on the file's `(mtime, size)` (`read_cache.cached_document`), so a lookup costs one `stat` and the file is parsed again only when it changes on disk — an edit by hand or by another process is seen by the next call, with no restart. The one writer (`roots.register_root`) reads the file fresh under its lock (`read_storage_fresh`), because a merge into a cached copy can drop another process's entry, and drops the cached entry after its replace (`read_cache.forget_document`). The resolved *location* is still never cached: the environment variable is read on every call. A lookup sits under per-item paths — a montage tile asks for the preview cache location at least twice — so a parse per lookup made a tile's cost grow with the file's size (two 82 ms parses per tile on a 227 KB file). Guard: `tests/storage/test_machine_config_reads.py`.

### `StorageLocation` abstraction (not built)

All storage backends would implement two methods:

```python
class StorageLocation:
    def video_uri(self, path: str) -> str:
        """
        Returns a URI suitable for cv2.VideoCapture.
        Local:  filesystem path string
        Cloud:  presigned HTTPS URL (TTL covers job duration)
        """

    def fs_store(self, path: str) -> zarr.abc.store.Store:
        """
        Returns a zarr v3 store.
        Local:  zarr.storage.LocalStore (no network overhead)
        Cloud:  zarr.storage.FsspecStore via s3fs / gcsfs
        """
```

### Large file handling

**Video files during processing**: `cv2.VideoCapture` supports HTTP/HTTPS URLs with range requests when OpenCV is built with its libavformat backend enabled (the standard distribution build). MP4 files are seekable over HTTP. Cloud video is accessed via presigned URL — only the byte ranges needed per frame are fetched. No full download. Note: this uses libavformat as a linked library, not the FFmpeg CLI — it is consistent with Tenet 15, which prohibits shelling out to the FFmpeg executable.

**Zarr stores (embeddings, analysis outputs)**: Zarr natively speaks fsspec and fetches only the chunks overlapping a requested slice. A `(100000, 768)` feature array chunked as `(1000, 768)` means loading 1000 frames downloads ~3MB regardless of total store size. Dask arrays from zarr are lazy by default. On object-store backends, arrays could additionally be **sharded** (`create_array(..., shards=...)`) so many chunks live in one storage object — see "Sharding (v3)" in Section 9; no writer passes `shards=` today. This is the main reason the storage layer targets zarr v3.

**Local storage**: Both methods are no-ops — `video_uri()` returns a filesystem path, `fs_store()` returns a `LocalStore`. No network calls, no overhead. The local backend writes chunks-only (no shards) by default.

---

## 3. Recording Structure

### What is a recording

A recording is one logical recording: one or more video files representing continuous recording, plus any time-series data streams collected simultaneously.

### Directory structure

Each category (video, each stream) gets its own folder. Each stream folder holds the original imported files (`raw/`) and the canonical converted data (`canonical/`). A raw file keeps its own name unless an earlier file of the same stream already has it, when it is prefixed with its role (`program_same.txt`); the manifest's `raw` list is what says which file is which. A stream name that differs from an existing one only by case is refused, since it is the same folder on macOS and Windows. Video parts have a `timestamps/generated/` folder holding each part's frame timestamp map. Each stage's runs write under their own folder (Tenet 5), named in the stage's section.

```
recordings/
└── recording_001/
    ├── recording.yaml
    ├── video/
    │   ├── part_001.mp4            # a copied part; a referenced part has no file here
    │   └── timestamps/
    │       └── generated/
    │           ├── part_001.csv
    │           └── part_002.csv
    ├── streams/
    │   ├── medpc/
    │   │   ├── raw/                # the importer's source files, verbatim
    │   │   │   ├── m1_day3.txt
    │   │   │   └── operant.MPC
    │   │   └── canonical/
    │   │       └── part_001.csv    # one continuous canonical file per stream
    │   └── accelerometer/
    │       ├── raw/
    │       │   └── accel_export.csv
    │       └── canonical/
    │           └── part_001.csv
    ├── arenas/
    │   └── arena_001/
    │       └── keyframes.csv
    ├── rois/
    │   └── roi_001/
    │       └── keyframes.csv
    ├── points/
    │   └── point_001/
    │       └── keyframes.csv
    ├── detections/                 # det_NN/ (§9)
    ├── tracking/                   # track_NN/ (§13)
    ├── embeddings/                 # emb_NN/ (§14)
    ├── keypoint_detect/            # kd_NN/ (§26)
    └── keypoint_track/             # kt_NN/ (§26)
```

### `recording.yaml`

All file paths are relative to the recording directory. The `video:` block is written by `Recording.add_video_part`, each `streams:` entry by `Recording.add_stream` (`api/recording.py`).

```yaml
schema_version: 1
id: "recording_001"
name: "recording_001"
created_at: "2026-05-21T10:00:00Z"

video:
  storage: "default"
  fps_container: 30.0
  fps_actual: 30.0
  fps_source: "container"       # the only value written today
  resolution: [1920, 1080]
  total_duration_s: 3600.0
  files:
    # A **copied** part: the tool owns the bytes, at a recording-relative path.
    - index: 0
      id: "part_001"            # stable stem for zarr / progress names
      timestamps:
        generated: "video/timestamps/generated/part_001.csv"
        user: null              # always null: no user map can be uploaded yet
        active: "generated"
      file: "video/part_001.mp4"
      origin:                   # provenance only — where it was copied *from*
        root: "scratch"
        rel: "day3/cam1.mp4"
      fingerprint:              # of the copy; absent on a part copied before #29
        size: 2147483648
        mtime_ns: 1747821600000000000
        sha256_ends: "c3d4…"
    # A **reference-in-place** part: the bytes stay where the user put them.
    - index: 1
      id: "part_002"
      timestamps:
        generated: "video/timestamps/generated/part_002.csv"
        user: null
        active: "generated"
      source:                   # resolved through storage.yaml's `video_roots` (§2)
        root: "lab_nas"
        rel: "day3/cam2.mp4"
      fingerprint:              # integrity.file_fingerprint: detects a moved / replaced / truncated file
        size: 2147483648
        mtime_ns: 1747821600000000000
        sha256_ends: "a1b2…"    # sha256 of the first and last 1 MiB
      frame_count: 53946        # len(timestamp_map), so a presence check needs no decode

streams:
  - name: "medpc"
    type: "events"              # legacy; `shape` is the axis readers use
    shape: "event"
    role: null
    clock: "stream_seconds"
    columns: ["t_ns", "code", "label"]
    dtypes: {t_ns: "int64", code: "int64", label: "str"}
    importer: "medpc"
    alignment:
      offset_ns: 3200000000     # constant shift: stream t=0 → video t_ns
      active: "offset"
    raw:
      - {index: 0, file: "streams/medpc/raw/m1_day3.txt", role: "data"}
      - {index: 1, file: "streams/medpc/raw/operant.MPC", role: "program"}
    files:
      - index: 0
        file: "streams/medpc/canonical/part_001.csv"
    metadata:                   # the importer's own: MED-PC keeps header fields and the code→label map
      subject: "M001"
      msn: "operant"
      event_map: {"1": "lever_press", "2": "reward"}

  - name: "accelerometer"
    type: "sampled"
    shape: "continuous"
    role: null
    clock: "stream_seconds"
    columns: ["t_ns", "x", "y", "z"]
    dtypes: {t_ns: "int64", x: "float", y: "float", z: "float"}
    importer: "timeseries_csv"
    alignment:
      offset_ns: 0
      active: "offset"
    raw:
      - {index: 0, file: "streams/accelerometer/raw/accel_export.csv", role: "data"}
    files:
      - index: 0
        file: "streams/accelerometer/canonical/part_001.csv"
    metadata:                   # the mapping actually used, `auto` resolved
      source: "timeseries_csv"
      channels: ["x", "y", "z"]
      source_columns: ["x", "y", "z"]
      time_column: "time_s"
      time_kind: "seconds"
    sample_rate_hz: 200.0       # only when the importer reports one
```

A `behavior`-role stream also carries `origin` (`ground_truth` | `predicted`, §19). Readers default
missing keys via `.get`. The alignment correction table (`generated` / `user` files under an
`alignment/` folder, *Stream alignment* below) is not built, so no stream has that folder.

**Copied or referenced, and a reader never branches on it.** A part either carries `file` (the
tool copied the bytes into the recording, at a recording-relative path) or `source: {root, rel}`
(the bytes stay where the user put them). Neither stores an absolute path — Tenet 6 — because
`root` is an *alias* into `storage.yaml`'s `video_roots`, so re-pointing one alias moves an entire
video library and no experiment file changes. **Every** video reader — the run workers, previews,
the presence gate — resolves through `recording_store.resolve_part_video_path`, which is the one
place that branches; reference mode is transparent to everything above it.

`fingerprint` (`integrity.file_fingerprint`) is on both kinds, for different jobs. On a
referenced part it detects a file that was moved, replaced or truncated under us, and every reader
checks it (§15). On a copied part it is the copy's own, written by `add_video_part` and by a
put-back (amended in #29, owner decision; a part copied before that has none): nothing reads it to
find or gate pixels, since the tool owns the copy, and it is what **putting the video back**
checks a candidate file against. Two fields are only on one kind. `frame_count`, on a referenced
part, is `len(timestamp_map)` recorded at import, so a presence check costs no decode. And
`origin` — on a *copied* part — is the same `{root, rel}` shape used as **provenance only**:
deliberately not spelled `source`, because `resolve_part_video_path` keys on that name to find
pixels, so reusing it would send every read to the external original and deleting that original
would break runs whose footage we hold.

**Putting a copied video back** (#29, owner decision). A bundle without video keeps a copied
part's entry and timestamps but not its file, so on the machine that imported it the part reports
`missing_file`. `Recording.relink_video_part` — Locate video in the GUI, `recording relink` in the
CLI — accepts a copied part **whose file is missing** and copies the chosen file back in to its
`file`, through a sibling temporary file and `core.atomic.replace`, so a cancel leaves no half
file; the part stays copied, keeps its `origin`, and records the copy's `fingerprint`. A copied
part whose file is present is refused. The file is accepted only if its frame times match the
part's stored generated map **exactly** — the same count and every `t_ns` equal, both re-based to
start at 0 (the stored map carries the earlier parts' offset), read the way an import reads them
(`video.probe_video` + `video.generate_timestamps`) — and its resolution equals `video.resolution`
(`video_integrity.part_video_mismatch`, which names what differs). A part that recorded a
`fingerprint` must match it too. Locate offers the file the part was copied from first:
`Recording.parts_to_locate` resolves `origin` on this machine and offers it when a file is there,
so no absolute path is stored to make the offer (Tenet 6). It reports `Reading timestamps` then
`Copying` through `progress`; the GUI runs it behind `run_progress` with Cancel.

### Ordering

The `index` field is the canonical ordering for both video files and stream files. Filename and filesystem order are ignored. The GUI exposes reorder UI that writes back to `recording.yaml`. Nothing reads files in any order except what `recording.yaml` declares.

### Time reference

All timestamps are **integer nanoseconds from the start of the first video file** (`t=0`), stored as `int64`. This is the universal join key across all streams and video frames. A recording with no video has no video anchor; its streams use their own native clock.

### Recording import (queue + bundle manifest)

Recordings are imported **non-blockingly**: the GUI hands a batch to an app-level queue (`gui/import_queue.py`) and returns immediately; one background worker drains it batch by batch. Within a batch, `api/imports.py::import_pending` imports recordings **in parallel** when `core/utils.transcode_parallelism()` is above 1 (`min(4, cpu_count)`, overridden by `DETAIL_MAX_TRANSCODE_PARALLELISM`) and more than one recording has video or streams to import, and sequentially otherwise. At submit, **all** the recording shells are created at once (dir + metadata row + tags) so the whole bout is visible immediately — metadata-only recordings land `ready`, ones with pending video/streams land `queued`. `recording.yaml`'s `import_status` (`ready` | `queued` | `importing` | `failed` | `cancelled`) is persisted so a crashed import is visible + deletable, not stuck; on experiment open, orphaned `queued`/`importing` (no resume — in-memory queue) are reconciled to `failed` (`Experiment.reconcile_imports`). Every entry path produces the same `RecordingImportSpec` (`api/imports.py`: id + grouping vars + *optional* ordered video parts + streams) executed by one path — `create_shell` (fast) then `import_data` (video parts in order, then streams; the slow, deferrable part). The **source video path is never persisted** (Tenet 6 — it's external/absolute); a queued recording's pending source is held in the in-memory queue and shown in the Recordings view. Cancel keeps completed recordings and marks the remaining `queued` ones `cancelled` (tags preserved, not deleted).

**Previews after import.** A finished import queues the recording's display-only previews
(§3 *Video preview cache*) on the GUI's background queue; the CLI's `recording add-video` builds
the new part's proxy right after the import with its own bar. Neither blocks the import or
changes its status: a recording is `ready` without its previews.

**Import progress** (§6 rules apply). `import_data` reports one fraction per recording that rises through every part and phase: parts weighed by file size, and within a copied part the timestamp scan takes `TIMESTAMP_SHARE` (0.6, measured) and the copy the rest; a referenced part is all scan. Each phase (`Reading timestamps` → `Copying` → `Importing stream <name>`) is forced past the throttle. `ImportProgressFold` (`api/imports.py`) folds the per-recording callbacks into an `ImportProgress(recordings_done, recordings_total, in_flight: tuple[(name, fraction, phase)], failed, elapsed_s, eta_s)`, whose `fraction` is `(recordings_done + Σ in-flight fractions) / recordings_total`, so concurrent recordings under transcode parallelism each count. It counts completions, not `import_pending`'s `index` (a batch position, not a done-count, under parallelism), and ignores a tick that lands after its recording finished. The GUI queue keeps one fold across every batch until it goes idle and renders it on the status bar through `gui/workers.py::import_progress_fraction`; the CLI's `recording import` renders the same fold through `_StageProgress`.

**Bundle manifest.** A *bundle* is a folder with a YAML `manifest` + the files it references (paths resolve relative to the folder). One nested file expresses multi-part video, streams, and metadata:

```yaml
schema_version: 1
recordings:
  - id: recording_001              # path-safe; unique in file; not an existing recording
    metadata: {animal_id: M001, treatment: drug_A}
    video: [videos/a.mp4, videos/b.mp4]    # optional; list order = part order
    streams:
      - name: medpc
        importer: medpc          # a registered STREAM_IMPORTERS id
        raw: medpc/m1.txt
        aux: {program: medpc/prog.MPC}     # keys = the importer's aux_inputs keys
        config: {}               # optional; the importer's own config_model (e.g. a CSV's
                                 # time_column / time_kind / channels)
        offset_ns: 0             # optional, default 0 (sync to video later)
  - id: recording_002              # metadata-only recording (no video/streams) — valid
    metadata: {animal_id: M002}
```

**Validation is upfront + exhaustive** (Tenet 10 — `storage/manifest.py::validate_manifest`): structure + `schema_version`; each id path-safe, unique-in-file, not colliding with an existing recording; each video path resolves under the bundle, opens (probe), and all of a recording's parts share fps + resolution; each stream's importer is registered, name path-safe + unique-in-recording, `raw` + every required `aux` key present with the right extension; `config` valid against that importer's `config_model` (and absent for an importer that declares none); `offset_ns` an int. All problems are reported together; on any error, **nothing is imported**. Warnings (a reused source path) don't block. The CLI (`detail recording import --bundle <dir> [--dry-run]`) and the GUI ("Import bundle…") share `load_manifest` + `validate_manifest` + `api.imports.build_specs`; deep parse/decode failures surface per recording at import time (that recording goes `failed`, the batch continues).

Using integer nanoseconds rather than float seconds avoids floating point drift at high sample rates (e.g., 30kHz neural signals over a 1-hour recording). Integer comparison is exact; any two samples with the same `t_ns` value are provably simultaneous.

If video is split across multiple files, `t_ns` counts continuously across part boundaries: part *k*'s map is its part-local map plus the integer-ns sum of `part_duration_ns` over the parts before it (`storage/recording.py::video_end_ns`, the same arithmetic `reorder_video_parts` uses), so the concatenated maps are strictly increasing whatever PTS each file starts at.

### Video preview cache

A **display-only** small copy of every video part, so that many clip tiles can play at once
(the clip montage draws 25+ tiles at 15 fps; live decode of 1080p source at that rate is about
everything a laptop can decode, and drawing at full frame then shrinking is slow per tile).
**Machine-local, outside every experiment**: never exported (`storage/export.py` needs no rule —
it only walks the experiment), clearing it deletes no experiment data, it reads at local-disk
speed when the experiment is on a share, and each machine builds its own.

**Where.** `storage/locations.py::preview_cache_dir()` (§2): `DETAIL_PREVIEW_DIR` →
`storage.yaml` `default.preview_cache` → `~/.cache/detail/previews/`. Layout
(`storage/previews.py`):

```
<preview_cache>/
├── index.json                     # resolved video path → {size, mtime_ns, key}; one stat per part on re-check
└── proxies/<key>/                 # key = "<size>-<sha256_ends[:20]>" of integrity.file_fingerprint
    ├── proxy.mp4                  # ≤480×270 (aspect kept, never enlarged, even sides), 15 fps,
    │                              #   H.264 (avc1) via video.open_video_writer, mp4v where cv2 lacks H.264
    ├── frames.zarr/
    │   └── src_frame  (n,) int64  chunks=(100000,)   # proxy frame i shows source frame src_frame[i]
    └── masks/<tracking_run_id>-<digest>.zarr/        # one per complete tracking run
        ├── proxy_frame (N,) int64    chunks=(10000,)    # rows sorted by proxy_frame, then object_id
        ├── object_id   (N,) int32    chunks=(10000,)    # the tracking run's animal
        ├── bbox        (N, 4) float32 chunks=(10000, 4) # x0,y0,x1,y1 normalized (the detection rows')
        └── mask_rle    (N,) str      chunks=(1000,)     # ASCII COCO-RLE counts at the proxy's (h, w)
```

Both zarr groups open through `zarr_store.open_group` with `schema_version: 1` the first attr;
`frames.zarr` attrs carry `width`, `height`, `source_width`, `source_height`, `fps`, `codec`,
`n_source_frames`; a mask store's carry `built_from` (= its `<digest>`), `tracking_run_id`,
`detection_run_id`, `proxy_key`, `width`, `height`, `n_proxy_frames`.

**Keys.** A proxy is keyed by the **source file's content** (size + head/tail hash), not by
recording: a copied-in part and its referenced original share one, and a re-encoded file gets a
new one. It is computed from the file at build time and remembered in `index.json` (machine-local,
so absolute paths are fine there — it is never an experiment file), never read from a part's
recorded `fingerprint`: a copied part made before #29 has none.
Masks are keyed by **tracking run** and part, because which animal is which (`object_id`) exists
only in tracking's assignments; they hang under their proxy because they are indexed by its
frames. `<digest>` = `masks_digest(tracking entry, detection entry)` — ids, `created_at` and
`every_n_frames` from `recording.yaml` — so a run re-made under a reused id is a new store, and
asking whether masks are missing costs one stat.

**Build** (`api/previews.py`), staged in a sibling `<name>.<uuid>.tmp` directory (unique per
build, never per process) and swapped with `core.atomic.replace`; a cancelled or failed build
removes its staging and leaves nothing. **Every build and every clear holds the cache's one build
lock** (`storage/previews.py::build_lock`, `core.atomic.file_lock` on `<cache>/.build.lock`:
across threads and processes, re-entrant within a thread). Builds are sequential anyway; the lock
is what keeps two builders of one file (a GUI and a CLI, an old queue stopping) and a clear from
writing into each other. **A build waits for it visibly**: it tries the lock without blocking and,
while another holder has it, reports `Waiting for another preview build to finish · <phase>`
every `atomic.WAIT_POLL_S` (0.25 s) through its progress callback — so the bar says why it sits
still and a cancel (the callback raising) lands while it waits. A blocking `flock` held by
another process parked the GUI's worker at 0% with no report and no way to stop it. A folder is **built only when whole** — `proxy.mp4` and
`frames.zarr/zarr.json` (`proxy_complete`), a mask store's `zarr.json` (`masks_complete`) — never
because it exists: a swap that finds a part-present folder (a clear that could not finish) removes
it first. Under the lock every `*.tmp` staging folder belongs to a dead build, so each batch
sweeps them first (`sweep_stale_staging`; index temp files, written outside the lock, only when
older than an hour). **A batch skips a failed part**: `build_tasks` returns a
`PreviewBuildReport(built, failed)`, a cancel (`OperationCancelled`) still stops the batch,
`cache previews build` prints each failure and exits non-zero (`recording add-video` warns and
exits 0: its import succeeded), and the GUI keeps "N builds failed" on its status-bar line with
the reasons in its tooltip. A part whose proxy failed has its mask builds reported as
`skipped: this part's proxy failed (<why>)` rather than tried — each would decode the whole part
again to build the proxy it draws on, and fail the same way.

- *proxy*: one sequential decode of the part from frame 0 (`video.read_selected_frames`: no seek,
  Tenet 11; PyAV, no FFmpeg executable, Tenet 15). A frame is kept when its `t_ns` reaches the next
  1/15 s slot after the part's first frame (`proxy_frame_selection`; 0.1 ms slack so a 60 fps
  frame on a slot edge is not pushed to the next), so a slower source keeps every frame and a VFR
  one skips no time. Kept frames are shrunk in the decoder (swscale, area) and written. The proxy
  is sized from **the part's own** video (`video.probe_video`), not the recording's resolution:
  parts of one recording can differ. `cv2.VideoWriter.write` raises nothing on a full disk, so
  before the swap the written file's frames are counted from its packets
  (`proxy_frames_in_file`, PyAV) and a file short of the frames written fails the build — a short
  proxy marked complete would never be rebuilt.
- *masks*: no video decode. For each proxy frame, the rows of the tracking run's nearest
  *processed* frame (`round(src_frame / every_n) * every_n`, so a strided run still shows its
  animal); unassigned rows (`object_id < 0`) skipped, an animal's fragments OR-merged and their
  boxes unioned, the mask shrunk (area, >50%) and re-encoded with `pycocotools`. It reads its own
  part only (`run_preview._build_part`, the cached timestamp map): another part's video being
  absent does not fail it.

**There is no second timestamp map.** The proxy container's own clock is not a time reference;
a proxy frame's time is `timestamps[src_frame[i]]` from the part's active map (Tenet 8).

**When it is built.** CLI: `recording add-video` builds the new part's proxy after the import
with its own bar (`--no-preview` skips); a preview that fails there is a warning naming
`detail cache previews build`, and the command still exits 0 — the video was added. `detail cache
previews build` builds what is missing.
GUI: an app-owned queue (`gui/preview_jobs.py`, one build at a time, a permanent status-bar
bar) is filled when an import finishes, when an experiment is opened (everything missing — a new
machine, an archive import, a cleared cache), and when a tracking run finishes on its run page;
a page that needs some recordings first calls `prioritise`. The missing check runs in the
worker, so a recording queued again while it is the one building is run again after it.
Switching experiments stops the old queue's build before the new queue starts. Progress is the `(stage, fraction, detail)` callback weighed by source frames (masks at
0.35 of a proxy, measured), phase line `proxy · <recording> · part i of n` → `masks <trk> · …`.

**Read** (`PreviewReader.open(recording, part, tracking_run_id)` → `.clip(object_id, start_t_ns,
end_t_ns, tile_px)` → `PreviewClip.frame_at(offset)` / iteration): a crop fixed per clip
(`clip_montage.crop_box` over the stored boxes, in proxy pixels), shrunk to the tile, with the
mask outline and an 8-frame (0.5 s) fading trail drawn **at tile size** (`visualize`'s fade,
ring and tint) — or the filled mask on white without footage. The proxy is decoded through
`video.FrameCursor` (walks forward, lets go of the file when idle), seeking by the proxy file's
own packet times, read once per file. Raises `PreviewMissingError` when the proxy is not
built here; masks asked for and not built yet open **footage-only** (`masks_missing` set,
whole-frame crop, no outline), so a montage can play while they build.

**Who reads it.** Display surfaces only (the clip montage, later the behavior-query clip grid,
the clip window's opening playback, node thumbnails). Never a pipeline / analysis / tracker /
model / executor / transform / method / trainer module (`tests/storage/test_preview_isolation.py`,
a source scan), never an export (exports read the source), and not the full-frame viewers
(detection / tracking runs, keypoint labelling, arena, prompt test), which need every frame at
full size.

**Clearing.** `clear_preview_cache()` (Configuration's *Preview cache* line, `detail cache clear
--previews`) removes `proxies/` and `index.json` only, never the configured folder itself, and
waits on the build lock — a build in flight stops (the GUI cancels its queue first) before
anything is deleted. A file that cannot be deleted (in use on Windows) raises rather than leaving
a silently half-cleared cache.
Measured sizes and build times are in the PR that added it; mp4v is roughly twice the size of H.264.

#### Compatibility contract

**A user who updates the app keeps the previews they built.** Rebuilding them is the most
expensive thing an update can do to a user by accident: minutes of proxy and mask building per
hour of video (`MASK_WORK_SHARE` is the masks' measured share of that work), on every computer
that opens the experiment, with the montages empty meanwhile. So a change
that makes built entries unfindable, or wrong, is a **user-facing breaking change** (CLAUDE.md
*Expensive derived data*), and `tests/storage/test_cache_compat.py` pins every value below so
that one cannot happen as a side effect.

**What each entry depends on.** An entry is found by a formula over these, never by a version:

| entry | found by | depends on (changing any one orphans every entry of this kind) |
| --- | --- | --- |
| cache root | `locations.preview_cache_dir()` | `DETAIL_PREVIEW_DIR`, `storage.yaml` `storage.default.preview_cache`, the default `~/.cache/detail/previews/` |
| `proxies/<key>/` | `proxy_key(integrity.file_fingerprint(path))` | the fingerprint's inputs: file size + sha256 of the first and last `integrity._CHUNK` (1 MiB); the key format `"<size>-<sha256_ends[:20]>"`; the folder name `proxies` |
| `proxy.mp4` + `frames.zarr` | `proxy_complete(folder)` | the file names, `frames.zarr/zarr.json` existing; `src_frame` int64, chunks `(100000,)`; attrs `schema_version` (first), `width`, `height`, `source_width`, `source_height`, `fps`, `codec`, `n_source_frames` |
| `masks/<trk>-<digest>.zarr` | `masks_path(key, trk, masks_digest(trk_entry, det_entry))` | the digest's five inputs (tracking `id`, `created_at`, `every_n_frames`; detection `id`, `created_at`, as `str()`, `"|"`-joined, sha1 `[:12]`); its proxy's key; columns `proxy_frame` int64 / `object_id` int32 / `bbox` float32 `(N,4)` at 10,000-row chunks, `mask_rle` str at 1,000 |
| `index.json` | resolved path → `{size, mtime_ns, key}` | nothing expensive: losing it costs a re-fingerprint (two 1 MiB reads per part), not a rebuild |

**What is not in any key: how a proxy was made.** `PROXY_FPS` (15), `PROXY_MAX_WH` (480×270),
`proxy_size`'s rounding (aspect kept, never enlarged, even sides), `proxy_frame_selection`'s
slots and 0.1 ms slack, the codec order (`avc1`, then `mp4v`; encoder defaults otherwise, so
keyframes ~250 frames apart), and the in-decoder area shrink. Changing one of these does **not**
rebuild anything: proxies already on disk are found by their unchanged key and kept, so old and
new mix, and the read side (which uses the current `PROXY_FPS` and `TRAIL_FRAMES`) draws the
old ones by the new rules. A parameter change is therefore a deliberate change too. Likewise,
`schema_version` is written to both zarr groups and **no reader checks it today**; it becomes a
check the first time a version is bumped.

**`integrity.file_fingerprint` is not the preview cache's to change.** The same fingerprint is
recorded in `recording.yaml` for referenced videos and compared by `video_integrity`, `export`
and `analysis/provenance`; a new formula would report every referenced video as changed. A
preview change that needs a new key adds to the key, it does not change the fingerprint.

**Changing it deliberately.** When a change truly must invalidate (a wrong frame selection, a
codec that will not play, a mask format the reader can no longer read):

1. **Name the one store that changes** — proxies, masks, or the index — and rebuild only that.
   A mask change keeps every proxy: it changes the mask store's name (a version in the digest's
   inputs or in the file name), never the proxy key.
2. **Version it where it is checked.** Add a format number (a `proxy_format` / `masks_format`
   attr, compared by `proxy_complete` / `masks_complete`, or a suffix on the store name) so the
   missing-check finds old entries as missing. Never re-purpose `schema_version` for a parameter
   change; it describes the layout.
3. **Keep reading what still reads.** If old entries are still correct to show (a cosmetic
   change), leave them and let new builds use the new rule; only a wrong picture forces a
   rebuild. The node cache's *older caches still read* is the model (§22).
4. **Tell the user once, before the bar starts**: what is being rebuilt, why, and roughly how
   long (the figures above per hour of video), on the GUI's preview status line and in
   `detail cache previews build`'s output; and in the PR body and release note, so someone
   updating a lab machine before a session can plan for it.
5. **Remove what is orphaned.** Old stores under old names are disk nobody reads; the build
   sweeps them under the build lock (`sweep_stale_staging` is where), or the note says to run
   `detail cache clear --previews`.
6. **Update the pins** in `tests/storage/test_cache_compat.py` in the same commit, and this
   table.

#### Expensive derived data and what rebuilding costs

The same rule for everything a user has already paid to compute. A *cache* may be invalidated
deliberately, with a version and a message; a *run output* never is.

| data | where | rebuilt by | cost of a rebuild | invalidated by | rule |
| --- | --- | --- | --- | --- | --- |
| video preview cache (§3) | machine-local, `preview_cache_dir()` | the GUI queue, `cache previews build` | minutes per hour of video per computer (above); hours on a big experiment | the key inputs above; silently mixed by an encode-parameter change | Compatibility contract above; pinned by `tests/storage/test_cache_compat.py` |
| node cache (§22) | `<run>/node_cache.zarr` | **Rebuild clips**, `--rebuild-cache` | seconds per run; no video decode | a `built_from` mismatch, including `RULES` (`NodeView.stale_reason`) | Allowed when deliberate: bump `storage/node_cache.RULES` with a line in its list, so the INTERPRET banner says why and **Rebuild clips** fixes it. `RULES = 2` (a bout's run ended one span late, `8d9afd6a`) is the worked example: stale only for the bout methods it changed, `schema_version` kept at 1 because the arrays kept their names and meaning. Pinned in the same guard |
| model weights | `model_cache_dir()` (`~/.cache/detail/models/`) | `install()` (Configuration, `detail model install`) | a download of GBs (SAM 3 and DINOv3 are multi-GB each) | a changed `hf_files` manifest, repo id or revision, or cache-dir resolution: `is_installed()` turns false | Change a manifest only for a new or corrected model; the run path never re-downloads (Hard Rules, *Model weights*) |
| run outputs: detections, tracking, embeddings, analysis results | inside the experiment | re-running the stage | GPU hours on a big experiment, and the user's own decisions (labels, audits) attached to the run | **nothing**: not a cache | A schema change keeps old runs readable (an old column read as absent, a reader that branches on what the store has); never a "re-run to upgrade" |


### Frame timestamp maps

Each video part has a `timestamps.csv` mapping `frame_idx → t_ns`. This is generated at import time from either container PTS (presentation timestamps, extracted via PyAV) or computed from `fps_actual` as a fallback. PTS are read from the container **packets** (demux — no full decode, fast on large files), trusted on their own integrity (every video packet has a PTS; sorted strictly increasing); only when that can't be established does `generate_timestamps` fall back to a full **decode** pass (the conservative ground truth). It deliberately does *not* gate on cv2's estimated `CAP_PROP_FRAME_COUNT` — that estimate is commonly off by a frame on real H.264 and would spuriously force a decode on every large import.

**The map describes the frames a decoder shows.** Two container facts are removed on both paths: packets an MP4 **edit list** trims (`Packet.is_discard` — kept in the file as decode references, never output by cv2 or PyAV's decode; counting them made the map longer than the video and shifted every detection row), and the file's **start time** (the first shown frame's PTS is subtracted, so `t_ns[0] == 0`; MPEG-TS, camera files and B-frame reordering all start above 0). Each PTS is rounded to integer ns *before* the origin is subtracted, so a reader re-deriving a frame's `t_ns` from its PTS reproduces the map exactly.

**A count mismatch is an error, not a dropped frame.** The detection worker pairs cv2's `i`-th frame with `map[i]`. If cv2 returns more frames than the map has, it raises `FrameCountMismatchError` (no `t_ns = -1` rows). If cv2 stops early, the file's packets are recounted (`video.shown_frame_count`): agreeing with the map means the frames exist and cv2 could not decode them — a damaged tail, recorded in `dropped_frames` — and disagreeing raises. Embedding and keypoints read detection's rows, so they inherit the check. A user-uploaded replacement map (e.g., from a hardware sync signal), stored alongside the generated one with `recording.yaml`'s `active` choosing between them, is **not built**: nothing writes a user map, every part's `timestamps.user` is `null`, and `active` is always `generated`. The readers already go through `active`, so the upload is the missing piece.

```
frame_idx,t_ns
0,0
1,16666701
2,33333398
```

A run that thins with `every_n_frames` needs no second map: its rows carry the source frame's own index and `t_ns` (§4).

Loading and `frame_idx → t_ns` lookup of these maps live in `storage/recording`. Canonical `t_ns` ↔ float-seconds conversions (`t_ns_to_seconds` / `seconds_to_t_ns`) live in `core/utils`. `seconds_to_t_ns` is the **one checked conversion**: it refuses a value that is not finite or whose nanoseconds do not fit int64, naming where it came from (`what=`), and every importer and every typed offset goes through it (`check_t_ns` for a value already in ns).

**The map is authoritative for frame *retrieval*, not just for counting.** `storage/video.py::read_frame_at` — the seek behind every preview (detection / tracking / input-image / prompt-test) — takes the part's map and seeks by `t_ns`, so the picture returned is the `frame_idx`-th in presentation order: the same indexing the sequential detection pass assigns, which is what makes an overlay keyed by `frame_idx` land on the right image. Without a map it falls to the packet-PTS index, and failing that to a bounded sequential decode (exact by construction, capped by `SEQUENTIAL_DECODE_FRAME_CAP`). What it must never do is seek by frame *number* via cv2's `CAP_PROP_POS_FRAMES`: that converts the ordinal to a time using the container's *advertised* fps, so on a variable-frame-rate or fps-skewed file it returns the wrong picture **and** reports the requested index back, desyncing overlays by an amount that grows with the frame number. Passing the map is also the fast path — it replaces a whole-file packet scan (≈11 s on a 1 GB part) with a CSV read (≈0.1 s). Because a map is part-local from 0 plus the part's recording offset, the seek converts with one constant per part (`video._map_shift_ns`: `map[0]` minus the stream's start time), and a frame is accepted only if its converted `t_ns` is in the map — so a later part or a late-starting file takes the fast path too, and a map that does not describe the file still falls back. The packet-PTS index skips `is_discard` packets in the same loop the import uses, so the two cannot disagree on an edit-listed file.

### Stream alignment

**Alignment applies only to an external-clock stream.** A `video_frames` / `video_seconds`
stream (e.g. behavior labels — BORIS, interval CSV) is authored *against the video itself*,
so it is anchored at import and its `offset_ns` is definitionally 0; there is nothing to
sync. Only the default `stream_seconds` clock (an independent device time, e.g. MED-PC) is
aligned by a read-time offset. The canonical predicate is `storage/clocks.needs_alignment(clock)`
(`clock not in VIDEO_ANCHORED_CLOCKS`), and **every alignment surface must gate on it** — the
API `set_stream_offset` is a no-op-shaped edit for a video-anchored stream, and the GUI must
not expose a sync / edit-offset control for one (see `docs/gui_design.md`). Offering an
alignment action for a stream that needs none is a bug, not a harmless extra.

Each stream has its own clock. The `alignment` block in `recording.yaml` maps stream time to video `t_ns`:

- **Simple case** (most common, **built**): constant signed `offset_ns` — `video_t_ns = stream_t_ns + offset_ns` (positive if the stream began after the video, negative if before). Canonical data stays stream-local and is never rewritten; alignment is applied at read time (`storage/streams.apply_alignment`), so re-aligning is a one-number edit.
- **Advanced case** (not yet built): user-provided correction table CSV mapping `stream_t_ns → video_t_ns` for non-linear drift or hardware sync data.

**Every alignment door gates on the stored clock, in storage (#30).** `streams_store.check_offset_allowed`
refuses a non-zero offset for a video-anchored clock; it is called by `Recording.add_stream` (the
resolved clock) and by `streams_store.set_stream_offset` (the stored clock), the only two writers of
a stream's `alignment` (`tests/storage/test_alignment_doors.py` scans for a third). The offset is
read only through `streams_store.stream_offset_ns`, from `alignment.offset_ns`.

**Exporting a stream (the `streams` target, amended pending owner confirmation).** The exported CSV
keeps the canonical columns, `t_ns` in the stream's own clock, and adds `video_t_ns` right after it
(`video_end_t_ns` after `end_t_ns` for an interval stream), each `apply_alignment` of its row, so the
file joins tracks on video time without the canonical data being rewritten. A `behavior` stream's
file also carries `origin`, since the `behavior_csv` importer reads it back. `streams.csv` lists
`stream, shape, origin, role, clock, offset_ns, file`, with `origin` empty for a stream that is not
labels and `offset_ns` the stored offset.

**The offset is set at import, not only afterwards.** `Recording.add_stream` takes `offset_ns` and writes it into the manifest in the same write as the rest of the stream (`--offset` on `recording add-stream`, `offset_ns:` in the bundle manifest, the alignment row on the GUI's import form). It defaults to 0, which *asserts* the stream began with the video rather than merely leaving the field unfilled, so a stream that needs a different number is corrected once rather than discovered later. A non-zero offset on a **video-anchored** stream is refused rather than stored, checked against the clock the import *resolved* (a frame-kind CSV reports `video_frames` from a `stream_seconds` importer, so the declared clock would wrongly accept it); storing a number that read time then ignores is worse than refusing it. Afterwards the offset is editable via `Recording.set_stream_offset`, `recording set-stream-offset`, or the GUI **"Sync to video frame"** helper — scrub to the frame where a chosen stream anchor event occurs and the offset is set to `frame_t_ns(frame) − anchor_t_ns` (`Recording.frame_t_ns` maps a recording-global frame to its video `t_ns`). The correction-table upload (with a `generated`/`user` `active` toggle) remains the documented future extension.

**Alignment correction CSV format:**
```
stream_t_ns,video_t_ns
0,3200000000
1000000,3201000000
```

### Importer configuration

`STREAM_IMPORTERS` holds five today:

| `importer_id` | shape | `role` | `clock` | `config_model` |
|---|---|---|---|---|
| `medpc` | event | — | `stream_seconds` | `MedpcSessionConfig` (which session of a several-session file; the `.MPC` program is an `aux_input`, not config) |
| `behavior_csv` | interval | `behavior` | `video_frames` | — |
| `boris` | interval | `behavior` | `video_seconds` | `BorisSelectionConfig` (which observation and subject) |
| `events_csv` | event | — | `stream_seconds` | — |
| `timeseries_csv` | continuous | — | `stream_seconds` (`video_frames` when the time column is frames) | `CsvStreamConfig` |

`timeseries_csv` is the worked example for everything below: a continuous CSV cannot be read
without being told which column is time and which are channels, so it is the importer the config
channel exists for.

**A config can also pick within a file (#30, amended pending owner confirmation).** A MED-PC data
file can hold every box of a day (one `Start Date:` block each) and a BORIS export several
observations and subjects. A recording is one animal in one session, so such a file is **refused
until its importer's config names one** (`configs/stream_select.py`: `MedpcSessionConfig`
`{subject, box, session}`, `BorisSelectionConfig` `{observation, subject}`), and the refusal lists
what the file holds. They are never merged and never cut to the last block. Every field defaults to
`None`, so a one-session file imports exactly as before with no config. The chosen block is recorded
in the stream `metadata` (`session` / `sessions_in_file`, `observation` / `observations_in_file`,
`subjects`). Reached through the same config channel as the CSV mapping: `add_stream(config=)`,
`add-stream --medpc-subject/--medpc-box/--medpc-session/--boris-observation/--boris-subject`, a
bundle manifest's `config:`, and a picker band in the GUI's Add stream dialog. A selection is about
one *file*, so a saved preset does not carry it (`ConfigBand.preset_config`, and `--save-preset`
saves only the CSV mapping).

**`maps_unrecognized_files`** (class attribute, default `False`) says an importer's config can
make it read a file nothing recognizes; the Add stream dialog falls back to such an importer for an
unclaimed file. It is declared rather than inferred from `config_model`, since the MED-PC and BORIS
configs only pick within files those importers already recognize.

**`resolve_clock(path, config)`** returns the clock an import of that file would record, without
importing it (default: the declared `clock`; the CSV importers override it from the time column).
Every surface gating an alignment control on a not-yet-imported file reads this, never the class
attribute: a `frame` time column is `video_frames` from a `stream_seconds` importer.

A `StreamImporter` declares an optional **`config_model`** — a pydantic model single-sourcing
its parameters (type / bounds / default / help / label) for the GUI form, the CLI, and the
generated docs page, exactly as a `Tool` and a `ProcessingModule` do (§6.5). `import_stream`
takes that model as a dict in its fourth argument, `config`, which is `None` when the caller
supplied none. An importer with no `config_model` ignores it, and passing a config to one is
refused.

**This is what keeps a configurable importer inside the registry.** Without a config channel
there is nowhere to tell an importer *how* to read a file, so the first attempt at explicit
CSV column mapping was written as a free function outside `STREAM_IMPORTERS` — and what went
with it was every surface the registry feeds: the GUI picker, `add-stream --importer`, and
the bundle manifest, whose validation rejects an unregistered importer id outright. The
capability worked from the one dialog it was built for and was unreachable from everywhere
else.

`configs/describe.py::iter_config_models` walks `STREAM_IMPORTERS` with the other registries,
so an importer's parameters reach the Parameters reference, `detail help`, the unit-convention
scan and the advanced-form guard the day the model exists;
`tests/configs/test_describe.py` fails if that walk is ever dropped.

**One file makes one stream.** An importer returns a single `StreamImportResult` per source
file, carrying whatever channels the config selected (`columns = ["t_ns", *channels]`). It
never emits a stream per column: `alignment.offset_ns` is per stream and channels recorded by
one device share one clock, so a 40-column pose export split into 40 streams is 40 copies of
one offset to keep in step by hand, and 40 verbatim copies of the source file in `raw/`
(forty copies of the same bytes). A channel may not be named `t_ns` or `frame` — those
are the canonical schema's own columns, and a channel taking one overwrites the int64 time
anchor's dtype with `float` (Tenet 8) while the manifest stays well-formed.

### Stream presets

An importer's config is the answer to "how is this lab's CSV shaped", and that answer is the
same for every recording in an experiment while the *file* and the *offset* are not. A
**preset** saves the first half: `<experiment_dir>/stream_presets.yaml`, a list of
`{name, importer, config, offset_ns?}` (`schema_version: 1`, atomic + `file_lock`, the same
read-modify-write hazard as `recording.yaml`'s `streams:` list). `storage/stream_presets.py`
owns it; `export_preset` / `import_preset` move one entry through a standalone file.

**A preset carries the mapping and not the alignment.** `offset_ns` is optional and absent by
default, because each video and logger pair has its own skew: one number applied to sixty
recordings is one recording's answer given to all of them, and it would be wrong in a way
nothing downstream can see. The batch surface takes an offset **per recording** instead --
`Experiment.apply_stream_preset(preset, [(recording, file, offset_ns), ...], name=)`, which
reports through the standard `progress(index, total, id, fraction, status)` callback and
skips a failing recording rather than sinking the batch (the `import_behavior_labels`
contract). The CLI reads that list from a `recording,file[,offset_s]` table. An `offset_ns`
on a preset is for the deterministic rig, where one trigger starts camera and logger.

The stored `config` is **not** validated by the storage layer: it is checked by the importer
at use time, through the same path a direct import takes. A second copy of each importer's
rules here is exactly the drift the config channel exists to remove.

### Stream shapes

| `shape` | Description | Example streams |
|------|-------------|-----------------|
| `continuous` | A signal sampled over time | heart rate, accelerometer, temperature |
| `event` | Sparse named timestamps | key presses, stimulus delivery |
| `interval` | Labelled spans with a start and an end | behavior annotations |

`shape` replaced the legacy `type` (`sampled` / `events`), which the manifest still carries; a
stream written before `shape` existed reads its shape from `type`
(`storage/streams.py::stream_shape`). New shapes are added as new `shape` values without
changing the schema.

### Canonical stream format

The CSV format below is the **internal canonical format**. All stream data is converted to it at import time. The pipeline never reads original external formats.

**Raw stream files and Tenet 6.** `streams/<name>/raw/` holds the importer's source files stored *verbatim*, and an external tool routinely records its own machine paths in them (a BORIS export names the annotator's media file in its `Source` column). Decision — mirroring the §12 ruling on trainer-owned `models/<id>/run/`: `raw/` is **importer-owned source data**, not relied on for portability. DETAIL stores it, never parses it, and reads `canonical/` instead, which DETAIL writes and keeps relative. The export portability guard (`storage/export.py::portability_offenses`) therefore skips `raw/` — scanning it made any experiment carrying a third-party export permanently un-exportable, contradicting this section's mandate to keep raw files verbatim.

**Sampled stream:**
```
t_ns,bpm
0,72
10000000,72
```

**Events stream:**
```
t_ns,label
45300000000,stimulus_on
120700000000,stimulus_off
```

An importer may add columns beyond the minimal pair when the source carries
stable extra fields — the manifest `columns` is per-stream. The MED-PC importer
(below) emits `t_ns,code,label`, keeping the instrument's integer event code
alongside the human label. Label fields are CSV-quoted (some MED-PC labels
contain commas), so the canonical writer (`storage/streams.py`) uses the `csv`
module, never manual string joins.

### Stream importers

Stream data comes from a variety of external tools in arbitrary formats. A
`StreamImporter` adapter (`core/protocols.py`) handles conversion to canonical
format at import time; concrete importers live in `detail/importers/`,
self-register into `STREAM_IMPORTERS` (Tenet 1), and are imported via
`import_stream_importers()`. Adding support for a new device or file format is
one new importer class — additive, no change to the storage/API/GUI seam.

```python
class StreamImporter(ABC):
    importer_id: str
    display_name: str
    stream_type: str                 # legacy; superseded by result.shape
    role: str | None = None          # semantic role a consumer checks (e.g. "behavior")
    clock: str = "stream_seconds"    # how native time reaches t_ns (see "Clocks" below)
    aux_inputs: list[AuxInput] = []   # extra files beyond the primary data file
    config_model: type | None = None  # its own pydantic parameters, or None when the
                                      # format is self-describing ("Importer configuration")
    maps_unrecognized_files: bool = False  # its config can read a file nothing recognizes

    def resolve_clock(self, path: Path, config: dict | None = None) -> str:
        """The clock importing this file would record (default: ``clock``)."""

    def can_import(self, path: Path) -> bool:
        """Returns True if this importer recognizes the primary data file."""

    def import_stream(self, data_path: Path, out_dir: Path,
                      aux_paths: dict[str, Path],
                      config: dict | None = None) -> StreamImportResult:
        """Parse the data file (+ any aux_paths) into a StreamImportResult. The
           importer parses (staying recording-agnostic); the storage layer resolves
           any frame/time columns to t_ns, writes the canonical CSV, and composes
           the recording.yaml manifest."""
```

**Two axes classify a stream (Section 3).** Beyond the shape (`shape` —
`event`/`interval`/`continuous`, generalizing the legacy `type`), a stream declares a
**`role`** (the semantic meaning a *consuming module* checks — the first is `behavior`)
and a **`clock`** (how the importer's native time reaches canonical video `t_ns`). Roles
grow with consumers, not producers: a role exists only when a module must branch on it
(streams have no consumer yet, so `behavior` is the first). The manifest records
`role`/`shape`/`clock`/`dtypes` alongside `columns`; `read_canonical` types each column
by `dtypes` (with a name-based fallback for legacy MED-PC streams). Discovery is by role
(`streams_by_role`); `stream_dependents` consults a `STREAM_CONSUMERS` registry (empty in
stage 1) so a future consumer registers without editing the API.

**Clocks (`storage/clocks.py`).** Clock resolution is a reusable seam separate from format
parsing (Approach Y — the importer stays recording-agnostic; `Recording.add_stream` applies
the resolver with the recording's active timestamp map). An importer declares which columns
are native-time and the import layer fills the rest: `video_frames` fills `t_ns` from a
source frame column via the forward map; `video_seconds` (BORIS) emits `t_ns` and the
layer fills the convenience frame columns via the inverse map; `stream_seconds` (MED-PC)
emits `t_ns` itself and a read-time offset aligns it. Importers sharing a clock share a
resolver.

The importer parses to canonical rows; the storage layer owns the canonical CSV
format and the manifest (single writer, per Tenet 5/2). An importer that needs
more than one file declares `aux_inputs` (each an `AuxInput(key, label,
extensions)`); the API/GUI collect them into the `aux_paths` dict. The registry
can try each importer's `can_import()` in order, or the user selects one
explicitly (the GUI's importer picker).

**MED-PC (Med Associates) — the first concrete importer.** Operant-chamber
recordings export a data `.txt` (a `Key: value` header, single-letter scalar
variables, and array variables dumped index-prefixed) plus a program `.MPC`
(MedState Notation) whose `Event stamps ( array E )` comment block maps each
integer event code to a label. Events live in two parallel arrays written in
lockstep — `E[i]` (event code) and `T[i]` (time in **seconds**, last value ==
`H`, the recording length); empty slots use the sentinel `-987.987`. `MedPCImporter`
(`stream_type="events"`, `aux_inputs=[AuxInput("program", …, [".mpc"])]`) pairs
E with T, drops sentinels, maps codes→labels, and emits `events` rows
`(t_ns, code, label)` in stream-local time, with the code→label map kept in the
manifest `metadata`. **This is the one layout it supports, and it refuses others rather than
guess (#30, pending owner confirmation):** no `E`/`T` arrays is refused by name, and a `T` past the
session length `H` (the sign of `T` logged in ticks, which would import every time too large) is
refused. A trailing run of `(code 0, time 0)` pairs (a program's unused, unsealed array space) is
dropped and counted in `metadata.notes`; a code-0 event between real events is refused unless the
event map names code 0. A file of several sessions is narrowed by `MedpcSessionConfig` (above).

### Behavior labels (ground truth)

Human-scored behavior ground truth is a stream (behavior labels are the first stream *consumer's* input,
built as an events/interval stream, not a separate annotation layer). Two importers, both
`role="behavior"`, `shape="interval"`, emitting the canonical columns
`t_ns, end_t_ns, label, subject, start_frame, end_frame` (`subject` is the per-track
framework — written, nullable, and unused until subject attribution is built; `end_frame`
is inclusive):

- **`behavior_csv`** — a plain interval CSV `start_frame,end_frame,label[,subject]`,
  `clock="video_frames"`; the import layer resolves the frames to `t_ns` against the
  recording's active map and keeps the frames alongside.
- **`boris`** — a BORIS **tabular-events** export (CSV/TSV): one row per event marker
  (`Behavior`, `Behavior type` ∈ `START`/`STOP`/`POINT`, `Time` in seconds). Pairs each
  START→STOP into an interval and emits a POINT as a zero-length interval, on the
  `video_seconds` clock (seconds → `t_ns`; the import layer fills the frame columns via the
  inverse map). `Subject`→`subject` (names kept in `metadata`, the per-row int stays null);
  `behavior_types` follow from the markers (verified against a real export). An export holding
  several `Observation id`s or subjects is refused until `BorisSelectionConfig` names one of each,
  and START/STOP pairing happens within the chosen rows. A label scored both as START/STOP and as
  POINT is refused naming the rows of each kind, since `behavior_types` holds one kind per label
  (#30, pending owner confirmation).

Overlapping intervals of different labels are first-class (co-occurring behaviors);
same-label overlaps are left untouched on import.

**Per-behavior metadata (for the eval).** An ethogram scores several behaviors, so the
stream's `metadata` records two maps the eval consumes: **`covered_behaviors`** — the scored
vocabulary (importers default it to the labels present; `add_stream(covered_behaviors=…)` /
`--covered-behaviors` / a `{file, covered_behaviors}` manifest entry override it to declare
behaviors *scored but never observed*, which must be a superset of the labels present) — and
**`behavior_types`** — each label's `point`/`state` type (BORIS's *Behavior type* column, else
inferred: all-zero-length → point, else state). A recording is eligible to eval behavior A iff
`A ∈ covered_behaviors` (`Recording.is_annotated_for`), so a "miss" is a real false negative, not
an unscored behavior; there is **no coverage-window** (an annotation covers a behavior for the
whole recording or not at all).

Two surfaces, one path: per-recording `Recording.add_stream(..., importer="behavior_csv")` and bulk
`Experiment.import_behavior_labels(manifest)` (a `schema_version: 1` + `labels: {recording_id: file}`
manifest — each value a filename or a `{file, covered_behaviors}` map — onto existing recordings,
each resolved against its own map, one bad file skipped-and-recorded). CLI: `recording add-stream`
/ `recording import-labels`. Legacy streams lacking the metadata fall back to inference. The first
consumer is the **behavior-query eval** (§16).

### Recording id and name

A recording's **id is the tool's; its name is the user's.** The id names the recording's
directory and is what every run record, export and analysis row stores; the name is the label
every surface displays.

- **Minting.** `Experiment.add_recording(name, metadata=None)` is the application path: it mints
  the id from a counter in `config.yaml` (`storage/experiment.py::next_recording_id`:
  `recording_000`, `recording_001`, …, zero-padded). The counter only goes up, so a deleted
  recording's id is never handed to the next one.
- **The explicit form.** `Experiment.create_recording(recording_id, metadata=None)` creates a
  recording at an id the caller chooses, with the name set to the same string. It is for scripts
  and tests, not for a surface adding a recording a user is naming.
- **Never derive an id from a filename.** The video's filename is a good starting *name* and a
  bad identity: the file gets moved, renamed or joined by a second part.
- **Names are unique.** A collision the tool caused (an import proposing two files with the same
  stem) is suffixed: `m27_day3` becomes `m27_day3_2` (`Experiment.unique_name`). A name a user
  typed (`Recording.set_name`) is refused with a `ValueError` if another recording has it.
- **Lookup.** `Experiment.recording_names()` is the `{recording_id: name}` display map.
  `Experiment.resolve_recording(ref)` takes what a user types: an exact id wins, then a unique
  name; anything else raises `KeyError` rather than guessing. Display a name wherever one will do.

Guard: `tests/api/test_recording_identity.py`.

### Ids are never reused

The recording rule holds for **every** id the tool mints as `<stem>_<n>`: models, datasets,
processing, distill, analysis and export runs, comparisons, example sets, skeletons, tool runs, a
recording's own runs (`det_NN`, `track_NN`, `emb_NN`, `kd_NN`, `kt_NN`) and its ROIs and POIs. A
deleted thing's id is never handed to the next thing of its kind, because finished runs, distill
records, comparisons, `roi_relations.csv` and export folders go on naming it.

- **One allocator**, `storage/id_allocator.py`. Each family keeps a high-water mark in
  `.ids.yaml` beside it (`models/.ids.yaml`, `tools/.ids.yaml`, a recording's own `.ids.yaml` for
  its runs and regions), keyed by stem, rewritten atomically and only upward. A claim holds an
  exclusive lock on the sibling `.ids.lock` and, for a family stored as directories, creates the
  id's directory with `exist_ok=False`, so two processes never take one id.
- **The next id** is the larger of the mark and one past the highest id on disk. An experiment
  made before the mark existed therefore continues from its highest id: no stored id changes, and
  an id reused before the mark existed cannot be detected.
- **ROI and POI ids are stable**: a rename, a shape edit or the scene editor's Save keeps a
  region's id (`replace_defs` takes each kept region's `id` back), and a new region gets a
  never-used one.
- **Not on the allocator, each for a reason**: recordings keep their own counter in `config.yaml`
  (above); arenas take the lowest free number, because every record naming one is a tracking run
  and an arena a tracking run names can be neither deleted nor redefined; snapshots are never
  deleted; a clustering run's label ids (`L<n>`, §22) still take one past the highest *(open: a
  high-water mark in the run's state)*.
- **Ids are local.** A skeleton id (`sk_NN`) means something only on the machine that minted it,
  so whatever compares a `skeleton_ref` across experiments or machines matches its `digest` (§26).

Guard: `tests/storage/test_id_reuse.py`.

---

## 4. Frame cadence (`every_n_frames`)

> Replaced the downsampling stage (a §4 processing module that transcoded a second copy of
> the video). Removed 2026-08; a one-time migration script clears the old artifacts and a
> permanent guard refuses any run record still carrying a `downsampling_run_id`.

A run processes every Nth frame of the recording's video. `every_n_frames` is a config field on
**detection**, **tracking**, **embedding** and **keypoint detection** (§26); `1` (the default) is
every frame. The keypoint distill's
label stage (`keypoint_dataset`, §12) has a field of the same name with a different meaning: a
stride over the teacher's rows when drawing candidate frames, so neighbouring frames of one
posture are not drawn together.

**One video, one frame space.** The stride selects which frames a stage looks at; it never
renumbers them. A row's `frame_idx` always indexes the source video, so `t_ns`, the part frame
offsets, the arena lookup and every downstream join are unchanged at any stride. There is no
second video that any stage analyses, and no second timestamp map. The one other copy of the
footage is the machine-local **video preview** (§3 *Video preview cache*): a shrunk 15 fps proxy
for display only, outside the experiment, never read by a stage
(`tests/storage/test_preview_isolation.py`), whose every frame names the source frame it shows
(`src_frame`) so its times come from the part's own map.

**One execution rule, at every stage:** keep a frame iff `frame_idx % every_n_frames == 0`.
Because the stride is source-relative and every upstream stride divides it, no stage needs to
know its upstream's cadence at *run* time — only at validation.

**Composition.** A downstream stage may only thin further by a whole multiple of what it
received (`core/utils.resolve_downstream_every_n`): after detection at 2, tracking and embedding
may take 2, 4, 6, … Asking for 3 raises and names the reachable values — those frames were never
processed, so rounding to a neighbour would silently change what was analysed (Tenet 17). Omitting
it inherits the upstream cadence.

**Identity.** `every_n_frames` is in each module's `identity_fields` (two runs at different cadence
are different specs, and the upstream picker labels them `· every Nth frame`) and in embedding's
recipe keys, so runs at different cadence are never judged comparable. It is also in
`identity_defaults` — absent from a record written before this existed, it reads as `1`, which is
what those runs did.

**Trackers age at the run's cadence.** `max_missed_frames` / `max_missed` count frames the run
*processed*, so the trackers step `frame_walk(groups, every_n_frames)` rather than every integer.
Walking every integer against a strided run would expire a track after `max_missed` **source**
frames — at a stride above the tolerance, before a single processed frame had passed.

**Cost.** Demux is still sequential over every frame (Tenet 11 forbids seeking), so a stride saves
the colour convert, the render and the model — the dominant costs — not the demux. Detection uses
`cap.grab()` and pays `retrieve()` only on kept frames. Progress stays in source-frame space,
which is what the wall clock tracks.

**No resize.** Nothing in the pipeline shrinks a frame: masks are stored at source resolution.
SAM 3's processor resizes to the model's own input, so source resolution never reached the GPU
cost; embedding's `input_size` still bounds what its model sees.

---

## 5. Model Abstraction

Detection models are hidden behind a consistent protocol. The model's architecture, prompt type, weight source, and execution location are internal details — all callers see the same interface.

### Prompt types

A prompt is a list of targets — one per class or object the user wants detected. Each target has a class name and a cue. This supports single-class and multi-class detection with the same structure.

```python
@dataclass
class TextCue:
    text: str

@dataclass
class PointCue:
    points: list[tuple[float, float]]
    labels: list[int]          # 1=foreground, 0=background

@dataclass
class BoxCue:
    boxes: list[tuple[float, float, float, float]]   # x1,y1,x2,y2 normalized

@dataclass
class ImageCue:
    image: np.ndarray | None   # runtime pixels; None after deserialization until loaded
    path: str | None           # experiment-relative path once persisted; used for serialization

Cue = TextCue | PointCue | BoxCue | ImageCue

@dataclass
class PromptTarget:
    class_name: str
    cue: Cue

@dataclass
class Prompt:
    targets: list[PromptTarget]
```

Example — two-class text prompt:
```python
Prompt(targets=[
    PromptTarget(class_name="mouse",       cue=TextCue("mouse")),
    PromptTarget(class_name="food_pellet", cue=TextCue("food pellet")),
])
```

Models declare which cue types they accept. New cue types are added as new dataclasses without changing the `Prompt` or `PromptTarget` structure.

### Prompt serialization

A `Prompt` round-trips to and from plain JSON-serializable dicts so it can live inside a `DetectionJob` (Tenet 4) and `recording.yaml`. Each target serializes with a `cue_type` discriminator and a `cue` payload:

| `cue_type` | `cue` payload |
|------------|---------------|
| `"text"`   | the text string |
| `"point"`  | `{"points": [[x, y], ...], "labels": [1, 0, ...]}` |
| `"box"`    | `{"boxes": [[x1, y1, x2, y2], ...]}` |
| `"image"`  | a experiment-relative path string — never inline pixels |

```yaml
prompt:
  targets:
    - class_name: "mouse"
      cue_type: "text"
      cue: "mouse"
```

`Prompt.to_dict()` performs no file I/O. An `ImageCue` must already carry a `path` — the API layer writes the image under the run directory before serializing. If `path` is `None`, `to_dict()` raises `ValueError` telling the caller to persist the image first. `Prompt.from_dict()` rebuilds an image cue as `ImageCue(image=None, path=...)`; the model loads the pixels from `path` when needed. This keeps image bytes out of `recording.yaml` / `job.json` and honors Tenet 6 (no absolute paths).

### Standardized output

Every model that locates something in a frame — a detection model (this section), a keypoint
model (§26) — returns one `FrameResult`: a frame's instances as equal-length rows, each field
optional and present when the model produces it. Results include `t_ns` so they are immediately
joinable with stream data. Specified 2026-10-07 and built in PR 1 of the §26 build: `FrameResult.from_detections`
is how a detection model builds it, `detections()` is the row view the detection code reads.

```python
@dataclass
class FrameResult:
    frame_idx: int                          # index within the video part
    t_ns: int                               # from the active timestamp map (Tenet 8)
    n: int                                  # rows; every array below has n as its first axis
    track_id: np.ndarray | None             # (n,) int32; None before tracking (a detection has no identity yet)
    class_name: list[str] | None            # (n,); detection sets it, a keypoint model leaves it None
    confidence: np.ndarray | None           # (n,) float32; the instance's own score
    bbox: np.ndarray | None                 # (n, 4) float32, normalized xyxy
    mask: list[np.ndarray | None] | None    # n binary (H, W) masks
    points: np.ndarray | None               # (n, K, 2) float32, normalized full-frame; NaN = not placed
    point_scores: np.ndarray | None         # (n, K) float32, the method's own; NaN with points
    candidates: np.ndarray | None           # (n, K, C, 2) float32; absent when the model declares C = 1
    candidate_scores: np.ndarray | None     # (n, K, C) float32
    detection_row: np.ndarray | None        # (n,) int32: the detection part row a per-detection keypoint
                                            #   method placed each instance on; -1 = from no detection (§26)
    extras: dict[str, np.ndarray]           # method-only arrays, first axis n (§26 names dinov3_match's)

@dataclass
class Detection:                            # one row of a detection FrameResult, the view detection code reads
    mask: np.ndarray | None
    bbox: tuple[float, float, float, float] | None
    class_name: str
    confidence: float
    object_id: int | None                   # = track_id once tracking has assigned one
```

`FrameResult.detections()` yields the `Detection` rows; `DetectionResult` (the former
`frame_idx` / `t_ns` / `detections` triple) is retired in favour of it, and every caller of
`predict` / `predict_batch` moves in the same change (`pipeline/_detection_worker.py`,
`pipeline/prompt_test.py`, `pipeline/labeling.py`, `pipeline/training.py`,
`tools/quick_segment.py`, `gui/preview.py`, and the two models). The detection zarr (§9) does
not change: the writer reads the same fields from the new container. What the shared container
buys is §26: a model that returns boxes **and** points (a YOLO-pose student) writes one result,
and the keypoint stage's storage mirrors these field names.

### Compute targeting

Users specify what hardware to run on. This is set per detection run, not globally, so different runs can use different resources.

```python
@dataclass
class ComputeTarget:
    backend: str                        # "cpu" | "cuda" | "mps" | "cloud"
    device_indices: list[int] | None    # for cuda: [0], [0,1], etc.; None = GPU 0 (see below)
    cloud_provider: str | None          # for cloud: "modal" | "aws" | etc.
    cloud_instance: str | None          # for cloud: "A100" | "g4dn.xlarge" | etc.
```

The `ComputeTarget` rides on the job (no `config.yaml` default; §1). The model receives only the resolved device string from `core/utils.resolve_device`: Ultralytics-style indices for CUDA (`"0"`, `"0,1"`), or `"mps"` / `"cpu"`. A torch-backed model's `load` turns that into a torch device with `core/utils.torch_device` (`"0,1"` → `"cuda:0"`). `device_indices: None` resolves to GPU 0 — spreading a run over every available GPU when none are named is not built; name the indices to use more than one. The `"cloud"` backend and the two `cloud_*` fields are design (§6): `resolve_device` refuses `"cloud"`.

### Model variants (size axis)

A model family that ships in several sizes/backbones declares them on the class
via the `VariantedModel` mixin (the base of `Model`, so shared by `DetectionModel`, `EmbeddingModel` and `KeypointModel`):
`VARIANTS: dict[str, str]` (variant name → backend weight identifier, e.g. a
`.pt` filename or an HF repo) and a `DEFAULT_VARIANT`. The model is **constructed
with the chosen variant** — `model_cls(variant)` — which resolves `self.variant`
and `self.weight` (and raises `ValueError` on an unknown variant); `load(device)`
is unchanged. `model_cls.variants()` lists the options for the GUI/CLI; a model
with no size options leaves `VARIANTS` empty and is constructed with
`variant=None`. The chosen variant rides on the job (`DetectionJob.model_variant`
/ `EmbeddingJob.model_variant`, `None` = default), is recorded in the run record,
and is surfaced as `--variant` (CLI) and a "Size / variant" combo (GUI).
Registered families: `dinov3_convnext_base` (convnext tiny/small/base/large, default
`convnext_base`), `dinov3_vit` (vit small/base/large, default `vit_large`) and the keypoint
model `dinov3_match` (§26; `vit_large`, `vit_base`, default `vit_large`); `sam3` has none
(its builder is fixed).

### Model protocol

One base, `Model`, carries what every model shares — identity, backends, the install / load
contract, the two hooks §26 generalised — and `DetectionModel`, `EmbeddingModel` (§14) and
`KeypointModel` (§26) each add their own `predict`. The base is the only place the install
contract is written; the three copies it replaces (one per protocol, the third on #626) are the
reason it exists (PR 1 of the §26 build).

```python
class Model(VariantedModel, ABC):
    model_id: str
    supported_backends: list[str]   # e.g. ["cuda", "cpu"] — resolves the device (Section 12)
    license: ModelLicense | None = None      # shown before install when the weights' terms restrict use
    shares_weights_with: str | None = None   # another model_id whose download this one reuses (§26 dinov3_match)
    VARIANTS: dict[str, str] = {}   # variant name -> weight id; empty = single model
    DEFAULT_VARIANT: str | None = None

    def __init__(self, variant: str | None = None) -> None:
        """Resolve the variant (None -> DEFAULT_VARIANT -> first); set self.weight."""

    def check_backend(self, backend: str) -> None:
        """Raise when this *install* cannot run on a backend the class supports (a CPU-only
        runtime asked for CUDA), so validate refuses before the run starts. Default: nothing."""

    def resolved(self) -> dict:
        """What load() decided that the config does not say, written to the run record under
        ``resolved`` (Tenet 9). Default: {}."""

    # is_installed / load / install / readiness / is_ready: the contract below, unchanged.

class DetectionModel(Model):
    supported_cue_types: list[type[Cue]]   # [] = closed set (a trained YOLO); [TextCue] = prompt-capable

    def is_installed(self) -> bool:
        """Pure disk check: are this variant's `hf_files` all present in the cache?
        Never touches the network."""

    def load(self, device: str) -> None:
        """Load present weights into memory on the given device (resolved from
        ComputeTarget). Assumes install() has run. **Never touches the network**:
        resolve the weights to a local path under model_cache_dir() and pass it to
        the backend explicitly (cache_dir= + local_files_only=True, or a resolved
        checkpoint path), and raise if they are absent rather than downloading —
        this runs in a pool worker with no progress channel, so a download here is
        invisible by construction. Must call model.eval() and optionally
        torch.compile(model) before returning."""

    def install(self, progress_cb: Callable[[InstallProgress], None] | None = None) -> None:
        """Make this model's weights present locally (the one-time, possibly large
        download). Idempotent — returns immediately if already present. Default is a
        no-op; weight-bearing models inherit a source mixin (see "Model readiness,
        install, and credentialed access")."""

    def readiness(self) -> ModelReadiness:
        """Structured, cache-first status: ready / no_token / no_access / offline.
        The network is consulted only when the weights are absent. Default: ready."""

    def is_ready(self) -> bool:
        """Thin wrapper: ``return self.readiness().ready``."""

    def set_conf_threshold(self, threshold: float) -> None:
        """Push the run's confidence threshold into the model. Default: no-op.

        Backends with an internal score filter (e.g. SAM3's
        ``Sam3Processor.confidence_threshold``, default 0.5) override this so the
        caller's threshold is actually effective end-to-end. Backends without an
        internal threshold inherit the no-op and rely on the pipeline's
        post-predict filter at ``job.conf_threshold``. Called by the pipeline
        runner immediately after ``load()``."""

    def predict(self, frame: np.ndarray, prompt: Prompt) -> FrameResult:
        """All forward passes must be wrapped in torch.inference_mode()."""

    def predict_batch(self, frames: list[np.ndarray], prompts: list[Prompt]) -> list[FrameResult]:
        """All forward passes must be wrapped in torch.inference_mode().
        CUDA implementations should also wrap with torch.autocast(device_type='cuda', dtype=torch.float16)."""

    # Optional: the three halves of predict_batch, so the detection worker can run the CPU
    # work on other threads while the GPU runs the forward (§7 *Execution*). A model that
    # overrides them keeps predict_batch(f, p) == finish_batch(run_prepared(prepare_batch(f, p)))
    # exactly; the defaults run the whole call in run_prepared.
    def prepare_batch(self, frames: list[np.ndarray], prompts: list[Prompt]) -> object:
        """CPU only, safe off the model's thread. Default: (frames, prompts)."""

    def run_prepared(self, prepared: object) -> object:
        """The device work, on the model's thread. Every device-to-host copy happens here:
        a copy made on another thread would queue behind the next batch on the same CUDA
        stream. Default: predict_batch(*prepared)."""

    def finish_batch(self, raw: object) -> list[FrameResult]:
        """Host arrays to rows, on the worker's writer thread. Default: raw unchanged."""
```

`TrainedYOLO` splits `predict_batch` at Ultralytics' own stage boundaries (`models/_ultralytics.py::SplitPredict`): `prepare_batch` is the letterbox, `run_prepared` the device copy, forward, NMS and full-resolution masks, and `finish_batch` the `Detection`s. SAM3 keeps the defaults, one frame at a time; it encodes each text prompt once per loaded model rather than every frame, since the encode is the same each time. `SAM3._ground` is `Sam3Processor.set_text_prompt` with that one change, and `tests/models/test_sam3.py` fails if the upstream method changes.

Adding a new model (any architecture, any cue type, any number of classes) requires implementing this protocol. Nothing else changes.

### Model registry

Same pattern as `Trainer`, `Tracker`, and `EmbeddingModel` — detection models self-register on import and are looked up by `model_id`.

```python
MODELS: dict[str, type[DetectionModel]] = {}

def register_model(model_id: str):
    def decorator(cls):
        MODELS[model_id] = cls
        return cls
    return decorator
```

**One lookup across every model registry.** `MODELS` (detection), `EMBEDDING_MODELS` (§14) and
`KEYPOINT_MODELS` (§26) are separate dicts because each runner dispatches on its own, but a
`model_id` is one namespace: `api.models` resolves an id through a single
`all_models() -> {model_id: (role, cls)}` view, and registering an id already present in another
registry raises at import. The hand-written chain `MODELS.get(id) or EMBEDDING_MODELS.get(id)`
(four copies in `api/models.py` on 2026-10-07) resolved a duplicate to the detection model
silently; the view replaces it (PR 1 of the §26 build).

Concrete implementations live in `models/` and import their backend (`torch`, `sam3`, ...) only inside their own methods — never at module top level. Excluding a model means not importing its module: the registry never sees it and nothing else breaks.

### Weight sourcing

`weights_source` takes exactly three values; `resolve.py` raises on anything else (§12 states the resolution rules).

| `weights_source` | Example | Where weights live |
|----------|---------|-------------------|
| `pretrained` (default) | SAM3, DINOv3 | `model_cache_dir()` — machine-global cache, not per-experiment. Resolved through the `MODELS` registry. |
| `trained` | Custom YOLO trained by this experiment (§12) | `experiment/models/{model_id}/` — `model.yaml` + the loader registered for its `model_family` |
| `imported` | A `.pt` dropped in by the user | `model_cache_dir()/imported/{model_id}/` — machine-global, the same `model.yaml` shape as `trained` |

Pretrained weights are not experiment-specific and are not stored inside the experiment directory.

### Model readiness, install, and credentialed access

Weights live in many places — a gated Hugging Face repo (SAM3, DINOv3), an
ungated hub repo, a direct URL, files the user drops in `experiment/models/`, or
nowhere at all (a cloud API). So **fetching weights is the model's own
responsibility**, expressed through three per-model methods:

- **`is_installed()`** — are the weights on this disk? A **pure disk check**: it
  resolves each file of the model's manifest under `model_cache_dir()` and never
  touches the network, ever.
- **`readiness()`** — *can* we obtain them? Cache-first: `is_installed()` first,
  network only in the not-installed case (for the gated-access UX).
- **`install(progress_cb)`** — make the weights present on disk. **The only method
  that may download.**
- **`load(device)`** — load present weights into memory for a run. **Never touches
  the network**: it resolves a local path under `model_cache_dir()` and passes it to
  the backend explicitly (`cache_dir=` + `local_files_only=True`, or a resolved
  checkpoint path). It runs inside a pool worker subprocess with no progress channel,
  so a download there is invisible by construction and violates the progress rule.

`model_cache_dir()` is the **single owner of the weight location**, and every call
site passes it **explicitly**. Configuring it through a process env var
(`HF_HUB_CACHE`, `HF_HOME`, `TRANSFORMERS_CACHE`) is prohibited: `huggingface_hub`
snapshots those into module constants at **import** time, so an `os.environ` write
after the first `import huggingface_hub` is a silent no-op — and `readiness()`
imports it before `load()` ever runs, so "set it earlier" is not a fix. Getting this
wrong splits the cache in two and falsifies `is_installed()` forever, which puts a
live network call on every run.

`readiness()` is cache-first and touches the network only when it must:

1. `is_installed()` → `ready`. No token check, no network call — the steady state,
   and it works offline. `is_installed()` verifies **each file of the manifest
   individually** (`huggingface_hub.try_to_load_from_cache`, a pure `isfile` under
   `cache_dir`). It must not be built on `snapshot_download(local_files_only=True)`:
   that returns the snapshot folder if the *directory* merely exists and ignores
   `allow_patterns` entirely ("we can't check if all the files are actually there" —
   its own source), so a half-finished download would report installed.
2. Not present (a download would be needed):
   - no token in the store → `no_token` (action: the HF token-settings URL)
   - token present → a gated-access check (`huggingface_hub.auth_check`),
     classified by the HTTP status of the response (robust across hf_hub
     versions, which may raise a base `HfHubHTTPError` or a typed subclass):
     - granted → `not_installed` (**not ready**): access is fine, but the weights
       are absent and `load()` will not fetch them. Actionable: run `install()`.
     - HTTP 403 → `no_access` (gated and not granted, or a fine-grained token
       lacking the "public gated repos" permission; action: the gate-page URL)
     - HTTP 401/404 → `bad_token` (token invalid or cannot see the repo)
     - connection failure (an `OSError`) → `offline` — *"not downloaded and no
       network to fetch it"* (graceful); an unexpected HTTP status is surfaced,
       not hidden

```python
@dataclass
class ModelReadiness:
    ready: bool
    reason: str                 # "ok" | "not_installed" | "no_token" | "no_access" | "offline" | "bad_token"
    message: str                # human, actionable
    action_url: str | None      # token settings, or the exact gate page
    repo_id: str | None
```

**Source mixins** carry the shared implementation so authors do not reimplement
it. A weight-bearing model composes a mixin and declares a couple of attributes;
the protocol's `readiness()`/`install()` defaults (ready / no-op) cover models
with nothing to fetch (and keep the test fakes working unchanged).

| Mixin | `readiness()` | `install()` |
|-------|---------------|-------------|
| `HFHubModel` (built, `models/hf_access.py`) | per-file `try_to_load_from_cache` → `auth_check` (when `gated`) | `snapshot_download(cache_dir=model_cache_dir(), allow_patterns=hf_files)`, forwarding progress |
| `URLModel` (not built) | cache-check | download from a URL + checksum verify |
| `UserProvidedModel` (not built) | are the files under `experiment/models/{model_id}/`? | no-op; raise an actionable "place the weights here" error |
| `APIModel` (not built) | API key present / endpoint reachable | validate key (no weights) |

Only `HFHubModel` exists. Trained and imported models (§12) resolve their weights through
`models/resolve.py` and the `model.yaml` beside them rather than a source mixin.

`hf_files` is the model's **file manifest, not a download-size optimization**: it is
what `is_installed()` verifies and what `install()` downloads, so the two agree by
construction. A model that leaves it `None` can only fall back to snapshot-folder
presence — declare it.

```python
@register_model("sam3")
class SAM3(HFHubModel, DetectionModel):
    model_id = "sam3"
    hf_repo = "facebook/sam3"
    gated = True
    supported_cue_types = [TextCue]
    # load()/predict() import the backend lazily (Tenet 13)
```

Hub models cache to `model_cache_dir()`: the `DETAIL_MODEL_DIR` env var, then
`storage.yaml`'s `default.model_cache` (§2), then `~/.cache/detail/models/`. `trained`
models resolve their path under `experiment/models/{model_id}/` (`models/resolve.py`), so
those readiness checks are handed the experiment context by the API layer. Trainers keep their existing
`is_ready()` (Section 12) for base weights; the `readiness()`/`install()` split
can extend to them later without an interface change.

The same logic backs the existing pre-run check "model weights present / API key
set" (Section 7): validation calls `readiness()` and, if not ready — **including
`not_installed`** — raises `ModelNotReadyError` carrying `message` and `action_url`
(Tenet 17). It never silently proceeds and never uses `confirm_fn` (Tenet 10).

Because `readiness()` is cache-first, an installed model validates with **zero
network calls**: validating a large batch makes zero HTTP requests, and a
transient Hub outage cannot fail a run whose weights are already on disk. Validation
never auto-installs — `install()` is a user-triggered step with its own progress
surface, and the run path's `load()` happens in a pool worker subprocess with no
channel to report a multi-GB download.

**Install is an API-level step, not an executor job.** It is one-time machine I/O
to the shared cache, so the API exposes it directly (Section 15) and the GUI
shows a cancellable download dialog separate from the run's frame progress.
`InstallProgress` (bytes done/total, current file) is a plain callback payload,
not a `ProgressEvent` on the executor boundary. A cloud worker installs into its
own cache the same way. If cloud/cancel parity ever needs the executor, install
can become a job later without changing the model-side methods.


---

## 6. Executor and Cloud Compute

The Executor protocol separates **what** a job does from **where** it runs. The GUI and CLI never call model code directly — they submit a `Job` to an executor and listen for progress.

### Executor protocol

```python
class Executor(ABC):
    def submit(self, job: Job) -> str:                      # returns job_id
    def status(self, job_id: str) -> JobStatus: ...
    def cancel(self, job_id: str) -> None: ...
    def stream_progress(self, job_id: str) -> Iterator[ProgressEvent]: ...
```

`Job`, `JobStatus`, and `ProgressEvent` are the shared types that cross the executor boundary. All are defined in `core/dataclasses.py`:

```python
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"      # interrupted; kept what it had written (Tenet 12)
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"  # via Executor.cancel

Job = (
    DetectionJob
    | LabelingJob
    | TrainingJob
    | DatasetExportJob
    | TrackingJob
    | EmbeddingJob
)

ProgressEvent = (
    DetectionProgressEvent
    | LabelingProgressEvent
    | TrainingProgressEvent
    | TrackingProgressEvent
    | EmbeddingProgressEvent
    | BatchProgressEvent     # a batch module's inner event (§6.5); carries no eta_s
)
```

These unions name the first-party types in `core/dataclasses.py`. A processing or analysis module (and a runner such as `keypoint_dataset` or `export_run`) registers its own job and event classes on import (*Job and progress-event registration*, §6.5), so the registry holds more types than the unions list.

`JobStatus` is a `str`-subclass enum (not `StrEnum`, which is 3.11+) so its members serialize as their plain string value through `json.dumps` / `yaml.dump` — `job.json` and `recording.yaml` store `status: complete`, etc., with no special handling.

One implementation is built, and a second is designed:

- `LocalExecutor` (built) — runs jobs as subprocesses on the local machine. No Qt dependency. Used in all local workflows including scripts, the CLI and the GUI.
- `CloudExecutor` (not built) — would serialize the job, submit it to a cloud service, poll for completion and stream progress back, without changing any caller code. Everything below about cloud compute is design.

### Jobs are fully serializable

A job is a data structure, not a function call. Everything required to reconstruct and run the job is expressible as JSON. See Section 7 for the full `DetectionJob` definition — the same serialization principle applies to all job types. Each job (and `Prompt` and `ComputeTarget`) realizes this with `to_dict` / `from_dict` methods; `from_dict(json.loads(json.dumps(job.to_dict())))` reproduces the job exactly.

### Data access and cloud compute

Cloud workers need direct access to input and output data. This is where the storage abstraction pays off: if input video and output stores are on S3, the cloud worker reads and writes directly with no data transfer through the local machine.

Storage and compute pairings:

| Compute | Storage | Works |
|---------|---------|-------|
| Local | Local | Yes |
| Local | Cloud | Yes |
| Cloud | Cloud | Yes — natural pairing |
| Cloud | Local | No — avoid this combination |

### Progress reporting

`stream_progress()` returns an `Iterator[ProgressEvent]` in all cases. Scripts iterate it directly. The GUI consumes a run handle's `progress()` on a `QThread` (`gui/workers.py`: `RunProgressWorker` for a processing run, `FitProgressWorker` for analysis, training and distill runs) and forwards each event as a Qt signal — the executor itself has no Qt dependency.

Every `ProgressEvent` carries an optional `message: str | None = None` — ephemeral phase text (e.g. `"loading model"`, `"loading detections"`) shown above the CLI bar; `None` on ordinary frame ticks. Runners emit progress per frame (throttled to ~0.5s, since each emit writes `job.json`), not only at part boundaries, so a single-part run still shows a live bar. **This cadence is enforced, not just intended.** Synchronous in-loop runners (the trackers, dataset export, labeling, and the §16 analysis modules) wrap `progress_cb` in the shared `core/progress.py::ThrottledProgress` — one initial (forced) event, throttled in-loop emits at `PROGRESS_INTERVAL_S`, a forced final event — so emitting only at a part/unit boundary (the recurring "no live bar on a single part" bug) is impossible by construction. The multiprocess coordinators (detection, embedding) instead emit by **polling** their worker progress files at the same `PROGRESS_INTERVAL_S` (a `worker.join(timeout)` loop is already incremental). The one exception is **training**, which reports per epoch — its natural unit. A `TRACKERS`-parametrized cadence test (`tests/trackers/test_tracker_progress.py`) plus per-stage cadence tests assert an intermediate `frames_done < frames_total` event on a single-part fixture, so a runner that regresses to coarse-boundary emission fails the suite. Every stage — embedding included — reports a real `frames_done` / `frames_total`, so the CLI keys the bar on frames for a true %/ETA. For embedding `frames_total` is the run-wide count of **unique frames** the selected tracks occupy (detections that share a frame, e.g. `anchor_tracker` secondaries, are OR-merged into one image, so one frame = one model input); it is pre-counted from the assignment + detection zarrs before the model loads so the denominator is stable from the first event. `EmbeddingProgressEvent` additionally carries `units_done` / `units_total` — the run-level (track, part) unit count — which the CLI shows (with the current `track_id`) in the bar postfix.

**Report measured work, not counted steps, and report it from wherever the work actually is.** Two rules, both learned from §20's search on a real recording, where the bar barely moved and the ETA was far off. First, **a fraction built by counting structural steps claims they cost the same, and they do not**: a one-row continuous span beside a long one was reported as half the search, and two exemplars of very different length were reported 50/50. Weigh each step by the work it will do (for a path matcher, `span_rows x usable exemplar rows`), and let a step that does no work count for nothing. Second, **the finest thing to report against is often inside the algorithm rather than at its structural boundaries**: with one exemplar over one continuous stretch every layer above the matcher has exactly one call to make, so the search reported only its terminal `1.0`. `retrieval.subsequence_dtw` therefore reports from its own DP row loop, whose per-iteration cost is uniform, which makes a linear read of it honest rather than nominal. The query runner (`pipeline/query.py`) dropped the nominal `LOAD_SHARE = 0.2` that used to stand in for the missing movement: the trajectory load is a negligible share of a unit, so it claims no share of the fraction and reports through its phase message instead. One copy survives: `analysis/clip_query.query_tracks`, which `query_eval` runs, still weights its bar with `_LOAD_SHARE = 0.2` (the bar only; its ETA comes from the folded fraction).

```python
class RunProgressWorker(GuiWorker):          # GuiWorker is a QThread (gui/workers.py)
    tick = pyqtSignal(int, str)              # permille (0..1000), status text
    progress = pyqtSignal(object)            # the aggregate ProcessingProgress itself
    finished_run = pyqtSignal()

    def run(self) -> None:
        for event in self._handle.progress():
            fraction, status = processing_progress_fraction(event)
            self.emit_to_ui(self.tick, round(fraction * 1000), status)
            self.emit_to_ui(self.progress, event)
        self.emit_to_ui(self.finished_run)
```

The run is submitted by the API (`Experiment.run_processing` and its siblings), not by the worker: the worker only reads the handle, so it can reattach to a run that outlived the window.

Cloud execution (not built) would write progress events to a status channel (S3 key, Redis, webhook) that the local machine polls independently of the job. `stream_progress()` abstracts over both.

**Aggregation, ETA, phase text, and per-surface rendering (invariants).** Four rules stop the per-surface renderers from drifting — the recurring class of bug where one surface silently drops a field (a frozen multi-recording bar; a missing ETA; a silent model load):

1. **Aggregate bars fold the inner fraction, and the inner fraction is monotonic across a recording's phases.** A multi-recording run yields `ProcessingProgress(recordings_done, recordings_total, current_recording, inner, elapsed_s)`, where `inner` is the active recording's live `ProgressEvent`. The bar fraction is `(recordings_done + inner_fraction) / recordings_total`, **never** `recordings_done / recordings_total` — else the bar sits frozen through a long recording. Nothing marshaling progress to a surface may drop `inner`, and the aggregate always shows **recordings completed / total**. A runner with *several* within-recording phases must not let one phase reset the fold: a phase with no natural sub-count (e.g. temporal clustering's `clustering` after the DTW `comparing` phase) **holds the prior fraction**, it does not report 0 — else the bar snaps back mid-recording (the "99% → 0% at clustering" bug). The runner populates the inner `frames_done`/`frames_total` so the fraction rises monotonically (phase-weighted where phases differ in cost), and a boundary emit that hands the count to `recordings_done` sends `frames_total = 0`. (`gui/workers.py::processing_progress_fraction` / `fit_progress_fraction`.)

2. **`eta_s` is populated by the runner and rendered by every surface.** Every `ProgressEvent` carries `eta_s: float | None`, except `BatchProgressEvent`, which is only ever the `inner` of a `ProcessingProgress` whose whole-run ETA comes from `elapsed_s`. A runner with a `done`/`total` denominator (frames, epochs, or recordings/units) populates it via `core/progress.py::remaining_seconds(fraction, elapsed_s)` — the linear `elapsed·(1−f)/f` estimate — rather than leaving it `None`. The GUI renders it through `_with_eta` onto the status line (the multi-recording bar shows a **whole-run** ETA from `ProcessingProgress.elapsed_s`); the CLI's tqdm shows its own rate-based `{remaining}`, so it needs no `eta_s` but must not override tqdm's default `bar_format` — **unless the bar's `n` is not a count of work.** A rate estimate is only meaningful over a counter; the distill bar sets `n` from the stage-*weighted* fraction (there is no common unit between labeling frames and training epochs), so tqdm's `{remaining}` re-derived the very ~5x under-report the fold had just been fixed to avoid, printed beside the runner's correct `eta=`. That bar suppresses `{remaining}` and shows the runner's number alone. A bar whose `n` counts frames, epochs or recordings keeps tqdm's default.

   **The ETA's denominator is the work the emitting coordinator did — not always the bar's.** `remaining_seconds` is `elapsed·(1−f)/f`, which assumes `elapsed` bought *all* of `f`. That holds only while the two measure the same work. A **retry** (§15) breaks it: the bar legitimately starts at `29/60` (those recordings really are complete), but `elapsed_s` starts at zero for the fresh coordinator, so feeding it the run-wide fraction infers a ~30× throughput and reads *"30m left"* on a 15 h retry. `ProcessingProgress.recordings_at_start` carries the mark the coordinator began from, and the ETA divides by the work since it — `(recordings_done + inner − at_start) / (recordings_total − at_start)` — while the bar keeps spanning the run. It is `0` for a fresh run, which is the identity, so nothing else changes. Any future path that resumes a partly-done bar owes the same distinction: **a plausible ETA is indistinguishable from a correct one**, so this is pinned by `tests/gui/test_workers.py::test_processing_progress_eta_measures_only_this_coordinators_work` rather than left to review.

3. **Every long op announces its phase via `message`, and the message *advances* as the phase does.** Model load, weight download, transcode, "loading detections" — any silent stretch sets `message`; both surfaces render it (the CLI writes it above the bar, the GUI appends it to the status). A long op that only moves a bar with no text reads as hung — **and so does one that sets `message` *once* and holds it through a multi-step op** (the "frozen at `starting`" bug): a static label over a stretch where the numeric fraction also can't move (a per-recording load + segmentation before the fraction-bearing inner loop) is indistinguishable from a hang. A runner with distinct phases names *each* as it enters it and **forces that phase-boundary emit past the `ThrottledProgress` throttle** so the text never lags the work — e.g. temporal clustering names `loading` → `clustering` → `persisting` → `projecting` → `caching`, with the method's own sub-phases (`segmenting`, `comparing`, `clustering`) as the note on `clustering`, not one `starting`. (Mid-run weight *download* streaming into the run bar — `.load()` has no progress hook — is the one gap: a pre-load `"loading model"` message is the pragmatic bar; the standalone `model install` surface shows the file + bytes on both surfaces.)

4. **One mapping per surface, covered by a test — not one mapping across surfaces.** Each surface has exactly one `event → render` function per event shape: the GUI's `processing_progress_fraction` / `fit_progress_fraction` / `import_progress_fraction` (+ `_with_eta`); the CLI's `_stream_*` tqdm drivers and `_import_with_progress`. Rendering is never hand-rolled a second time inside a page or worker. CLI and GUI are deliberately **not** unified onto a single function — tqdm counts `n/total` and owns its own ETA, so a shared fraction mapping would fight it. The invariant is that each surface's one mapping is **tested**, so a dropped field fails CI.

**Testing.** Beyond the cadence test (patch `PROGRESS_INTERVAL_S=0`, assert an intermediate `0 < done < total`) and a **populated `eta_s`** on a mid-run event, any stage flowing through `ProcessingProgress` ships an **aggregate-fold test** (that `inner` moves the aggregate bar; template `tests/gui/test_workers.py::test_processing_progress_fraction_folds_inner_within_recording`) and the durable-run path ships a **reattach test** (a fresh `attach_run` streams a surviving run to terminal; §15). Reverting any of these emits must fail its test.

### Run directory and `job.json`

`LocalExecutor` IPC is filesystem-only. `submit` resolves the job's **run directory** — co-located with the stage's outputs; each runner registers a `run_dir_for(experiment_root, job)` resolver alongside its runner in `JOB_RUNNERS` — writes `job.json`, and spawns one worker subprocess (`python -m detail.executor.worker <run_dir>`). Stages self-register their runners via the modules named in `DETAIL_RUNNER_MODULES`, imported by both the executor and the worker.

`job.json` is a **generic envelope** the executor reads, plus **stage-specific payload** keys it never touches:

```json
{
  "schema_version": 1,
  "status": "running",
  "job": {"type": "detection", "job_id": "det_01", "...": "..."},
  "progress": {"type": "detection", "parts_done": 1, "...": "..."},
  "error": null,
  "pid": 41237
}
```

- `status` — `JobStatus`; what `Executor.status()` returns.
- `job` — the tagged job (`job_to_dict`); the worker reconstructs the right class with `job_from_dict` and dispatches on `type`.
- `progress` — the latest tagged `ProgressEvent` (`progress_event_to_dict`) or `null`; `stream_progress()` polls and yields it on change until `status` is terminal.
- `error` — message recorded if the worker catches an exception.
- `pid` — the worker's process id, recorded by `LocalExecutor.submit` right after the spawn; what Cancel signals and the PID-reuse guard checks.
- stage payload (e.g. detection's `parts[]` and `mask_resolution`, Section 7/9) lives alongside these keys and is managed entirely by the runner.

Writes are atomic — a unique temp file per write (`tempfile.mkstemp`), moved into place with `core.atomic.replace` — so the polling reader never sees a torn file. Two processes write it during a run's startup (the executor records the worker's `pid` while the worker writes `status: running`); the unique temp name is what keeps one writer's replace from consuming the other's temp, and the last replace wins. Once running, the worker is the only writer and the executor only reads, so no locking is needed. One worker subprocess runs per job; multi-GPU part-parallelism lives inside the detection runner, not the executor.

**Stopping a run is two operations, and the difference is not a tuning parameter.**

- **Stop and keep** (`request_stop`; offered for training, `Experiment.stop_model`) writes its own `stop` marker and *never* kills. The worker polls it, sets the runner's `stop_event`, and exits with status **`complete`** — a finish the user chose, not an interruption; the run records what it produced. A trainer runs its own completion path. Its marker is not Cancel's because the worker must know which was asked: it shared the `cancel` flag until #486, so every stopped model read `partial` and was offered Resume on a checkpoint the backend had already finalized.
- **Cancel** (`cancel_run_dir`) is graceful-then-force: it requests the `cancel` flag, gives a loop-based runner three seconds to land its own terminal status, then kills the worker's **process group** (SIGTERM, then SIGKILL), because a blocking backend ignores the flag and a user pressing Cancel wants it to stop. It writes `cancelled` only if the worker did not already land a terminal status, which preserves a clean `partial`. It takes a run *directory*, not a live handle, so it works from a fresh process after the GUI restarted — the orphan case. A PID-reuse guard (`_is_our_worker`) keeps it from signalling a recycled pid.

### `activity.csv` — what happened, when

Every surface renders progress into a label the next tick replaces, so after two hours the screen shows the same sentence whether the run is working or wedged. `runs/<id>/activity.csv` (`storage/activity.py`, via `AppendLog`) keeps the rows instead of overwriting them: the last one carries a clock and there is nothing after it, which is the whole property. A user watching a long detection had no way to tell and ran a second tool beside the app to get timestamped lines out of it (#341); `docs/bug_reports.md` asks for exactly this, and *a missing log is its own bug*.

```
ts, elapsed_s, unit, unit_index, unit_total, phase, done, total, eta_unit_s, eta_run_s
```

`ts` is a **wall clock**, not an elapsed: "nothing since 14:02" is the question, and an elapsed cannot answer it without knowing when you are reading. `unit` is the recording a processing run is on, empty for a single-job run. `eta_unit_s` is that unit's own remaining and `eta_run_s` the whole run's — two denominators, never conflated, since recordings differ by hours and the run-wide rate says nothing about whether *this* one is stuck.

Its cadence is its own: `ACTIVITY_INTERVAL_S = 5.0`, not `PROGRESS_INTERVAL_S` (0.5s, a bar frame), which would write 72,000 fsynced rows for a ten-hour run. A row is forced at every phase and unit boundary whatever the clock says, so the lines that mark a change are never the ones the throttle dropped. The throttle drops rather than holds back, so the log keeps the last sample it dropped and writes it before the next boundary's row: a unit is timed from its first sample to its **last**, not to whichever earlier row the clock kept (#32). Every writer calls `ActivityLog.finish` at its terminal status, which writes that held sample and then a row stamped at the run's end in the last unit and phase, so the run's span and its last unit end when the run did. `ts` carries milliseconds; at whole seconds a 4 s recording read anything from 3 to 5 s. Writing is `best_effort` like the progress writers beside it — publishing is a statement *about* the work, and a raised write would reach a runner's isolation boundary and be recorded as the work itself failing. Three writers: the orchestrator's emit, the single-job worker, and the distill coordinator. It lives in `storage/` and not the GUI because the run outlives the window, because the GUI only ever sees a rendered string, and because a hung run is a bug report.

### `machine.yaml` and `resources.yaml` — what a job ran on, and its peak memory

The worker (`executor/worker.py`) writes both into its run directory when the job ends, whatever its status, through `storage/run_facts.py::record`, which is `best_effort` for the reason above. They sit with the **worker**, not with whatever wrote `activity.csv`: a processing run's log belongs to its coordinator, which only relays, while each recording ran in its own worker under `recordings/<id>/<output_path>`; a distill's stages ran as a dataset job (`datasets/<id>/`) and a training job (`models/<id>/`). The Performance report export maps each logged unit to that directory (`export_targets._job_dirs`). It prints frames and fps only for a module declaring `progress_counts_frames = True` (required on every `ProcessingModule` / `AnalysisModule`, no default): an analysis's `frames_total` is usually a per-mille fraction (`split_units`), and a multi-pass stage folds its passes into one frame count rather than reporting steps (keypoint tracking reported twice its frames until it did). Guard: `tests/executor/test_progress_counts_frames.py`.

- `machine.yaml` — `core/hostinfo.collect(load_torch=False)`, the same flat facts the report prints. Read at the end so a GPU job's already-imported torch is described; a job that never imported torch writes `not loaded by this run` for the torch fields rather than importing it. The report prefers this over the exporting computer, and says so when a run has none.
- `resources.yaml` — `core/resource_peaks.PeakSampler`, a daemon thread sampling once a second: `peak_ram_bytes` (RSS of the worker **plus every descendant**, summed — DataLoader workers and multi-GPU spawn children included), `gpu_backend`, `peak_gpu_reserved_bytes`, `peak_gpu_allocated_bytes` (CUDA's own peak counters, or sampled `torch.mps.driver_allocated_memory`), `sample_interval_s`. GPU fields are `null` — not measured — when the worker never imported torch or did its GPU work in spawned children.

The report's per-phase table (`activity.phase_timings`) charges each phase until the next one starts in the same unit, normalises `epoch N/M` to `training epochs` and a `" · "` detail away, and names blank rows before a unit's first named phase `starting` and blank rows after it `processing`.


---

## 6.5 Processing Modules

A **processing module** is one pipeline stage that consumes a single upstream input and writes its own derived output (Tenet 5), runs through the executor (§6), and is referenced by string id (Tenet 1). Detection, tracking and embedding were the first three; keypoint detection and keypoint tracking (§26) followed. The contract below makes a stage a *registry entry* so that adding one is an additive engineering task — implement a runner + its storage schema, register a module, and surface it everywhere a user reaches it (API method, CLI command, GUI) — or record a deliberate omission in `docs/cli.md`'s "Not in the CLI" — and never a change to the run backbone, the executor, other modules, or any central map.

The executor/runner seam (§6) is already stage-agnostic (`JOB_RUNNERS`, the uniform `run(job, run_dir, progress_cb, stop_event)` signature, the generic `job.json` envelope). This section adds the layer **above** that seam — a module abstraction and one generic API run path, `Recording.run` — which was hand-written per stage before it (`Recording.run_detection` / `run_tracking` / `run_embedding` were structurally identical; they are now thin wrappers over `Recording.run`).

### Scope

`ProcessingModule` covers **executor-backed, output-producing** stages only. The synchronous inspection tools — prompt-test (§8) and render-preview (§14) — are **not** processing modules: they have no runner, no zarr output, and no `recording.yaml` run record. Each attaches to the module whose configuration it informs (prompt-test → detection, render-preview → embedding) and stays a plain synchronous API method.

**Batch modules (`BatchProcessingModule`).** The default run model is one isolated job **per recording** (the coordinator sequences `Recording.run` calls, Tenet 5). A stage that needs **cross-recording state** — a transform fit *pooled across the whole selection* for comparability — can't fit that: it must see all selected recordings at once. Such a module sets `batch = True` and implements `validate_batch(experiment_root, recording_ids, config)` + `run_batch(experiment, recording_ids, config, emit_progress, cancel)` instead of the per-recording hooks; `RunOrchestrator` then calls `run_batch` **once** over the whole subset and `run_processing` validates once. It reuses the rest of the run spine unchanged — the run record, the detached coordinator, progress, reattach/reconcile — so it still appears in the New-run dropdown like any module; only the fan-out is replaced by a single job. Retry is whole-run (a batch is atomic — all recordings complete or all fail), not per-recording gap-fill. **No registered module is a batch module today.** `BatchProcessingModule` (`executor/runners.py`) and the orchestrator's single-job path are built; the stage that motivated them — a derived embedding fitting one transform (`drop_top` &c.) across the selection and writing each recording's `emb_NN` — was not built.

The experiment-scoped `AnalysisModule` (cross-recording aggregation, plots, reports) is a deliberately separate concept, specified and built in §16. The shared run machinery (the registry pattern, the generic run backbone, run-records, lineage, `RunHandle`, and the GUI module-shell) is factored so it is not welded to recording scope, which is what let that sibling be added without a refactor of this spine.

### The contract

```python
class ProcessingModule(ABC):
    module_id: str        # "detection" — registry key, job tag, recording.yaml root, GUI nav id
    id_prefix: str        # "det" -> det_01, det_02, ...
    runs_key: str         # "detection_runs" — the recording.yaml list this module appends to
    upstream: str | None  # "video" | another module_id | None — the single input dependency
    job_class: type[Job]  # the typed, serializable Job dataclass (registered dynamically)
    config_model: type[BaseModel] | None = None  # the pydantic config model (§6.5, below)

    def validate(self, recording_dir: Path, recording_data: dict, config: dict) -> dict:
        """All pre-run checks (Tenet 10); raise specific errors; return normalized
        config (may enrich it with derived values build_job/run_record consume)."""

    def build_job(self, recording_id: str, run_id: str, config: dict) -> Job:
        """Construct the typed job from validated config."""

    def runner(self, job, run_dir, progress_cb, stop_event) -> None:
        """The stage's compute — delegates to the existing runner body (unchanged).
        Named `runner` (not `run`) to avoid clashing with the generic Recording.run."""

    def run_dir_for(self, experiment_root, job) -> Path:
        """Where this run's output (and job.json) live — co-located with outputs."""

    def output_path(self, run_id: str) -> str:
        """Recording-relative output dir recorded in recording.yaml (e.g. detections/det_01/)."""

    # Provided by the base class; modules override only the extras hook.
    def run_record(self, run_id: str, config: dict, status: str) -> dict:
        record = {
            "id": run_id,
            "status": status,
            "output": self.output_path(run_id),
            "created_at": utc_timestamp(),
            **self.extra_record_fields(config),   # module keys incl. the upstream run id
        }
        if config.get("processing_run_id") is not None:   # the run_NNN that launched it (§15)
            record["processing_run_id"] = config["processing_run_id"]
        return record

    def extra_record_fields(self, config: dict) -> dict:
        return {}
```

The `recording.yaml` run record stays a lightweight index — the upstream run id among the module's extras is its lineage pointer. The **full** input ids + parameters live in the run's `job.json` (the serialized `Job`), which is what satisfies Tenet 9; the record shape per stage is unchanged from today.

The contract is an **ABC subclassed per stage** (consistent with the protocols in `core/protocols.py`): the base supplies the standard `run_record` skeleton, and each stage overrides `validate` / `build_job` / `runner` / `run_dir_for` / `output_path` / `extra_record_fields`. `runner` and `run_dir_for` delegate to the stage's existing module-level runner function and run-dir resolver, so the (often multi-GPU) compute bodies are untouched. `validate` carries the logic that lives in today's `_validate_X` methods, unchanged in substance.

### Config models (pydantic)

A module's run config is a **`pydantic.BaseModel`** in `detail/configs/<module>.py` — the
single source for each parameter's type, bounds, default, help, and label (a `Field(default, ge=,
le=, description=, json_schema_extra={"label": ..., "unit": ...})`). One declaration feeds four
surfaces: `validate()` (structure + scalar/cross-field checks), the CLI `--help`
(`Model.model_fields[name].description`), the GUI form (`gui/_pydantic_form.py::build_fields`
renders scalar fields as Qt widgets with range + tooltip; `field_meta` reads any field's metadata
for hand-built widgets), and the generated docs parameter reference (`describe_config` →
`docs/reference/parameters.md`). Hand-writing a parameter's range / help / default in a widget or
`@click.option` is a bug.

**Validation split.** `validate()` calls `Model.model_validate(config)` for the model's checks
(bounds via `ge`/`le`; cross-field via a `@model_validator`), then keeps the **experiment-aware**
checks that need `recording_data` (an upstream run exists, resolution/fps limits, comparability). It
returns the config dict. **Two return shapes by coverage:** when the model covers *every* key
(`temporal_clustering`), `{**config, **cfg.model_dump()}` (+ any derived keys) — the
`model_dump()` reproduces the dict, so the stored run-record shape and the executor Job are
unchanged. When the model covers a **subset** (detection, embedding, `location_export`, `query`) — nested inputs (`Prompt` /
`ComputeTarget`) or fields `validate()` *enriches* (`weights_source`, `track_ids`, `renderer_spec`)
live outside it — `model_validate` only enforces the bounds/shape and `validate()` returns the
enriched `config` unchanged; **do not merge a partial `model_dump()`**, it reverts the enrichment
(e.g. an `all_tracks`-resolved `track_ids` back to `[]`). `pydantic.ValidationError` subclasses
`ValueError`, so existing error handling and `pytest.raises(ValueError)` still hold.

**Status:** done. Every registered processing and analysis module declares a `config_model`, and
`describe_config` renders the generated parameter reference (`docs/reference/parameters.md`) from
them — plus the config-bearing runners that are not modules and so have no registry to walk
(`labeling`, `keypoint_dataset`, `comparison`, the keypoint track methods), listed in
`configs/describe.py`. A new
module ships a config model; ask `config_model` on the module rather than keeping a roster here,
which is what went stale.

### Registration

A **single decorator wires all three seams** — there is no way to half-register a module:

```python
@register_processing_module          # 1) PROCESSING_MODULES[module_id] = instance
class DetectionModule(ProcessingModule):  # 2) JOB_RUNNERS[module_id] = RunnerSpec(run, run_dir_for)
    module_id = "detection"                # 3) dynamic job-type map: module_id -> job_class
    id_prefix = "det"
    runs_key = "detection_runs"
    upstream = "video"
    job_class = DetectionJob
    ...
```

`PROCESSING_MODULES: dict[str, ProcessingModule]` is the registry: what a module *is* comes from it, never from a hand-kept map. Which modules get *imported* is still a hand-kept list — the `PIPELINE_RUNNERS` tuple in `api/_runner_modules.py` (pipeline, trainer and model modules alike), which the executor hands each worker subprocess in the `DETAIL_RUNNER_MODULES` env var — so a build excludes a module by dropping it from that tuple, exactly as for backends.

### The generic run backbone

One real implementation (`Recording.run`, condensed); the typed per-stage methods are thin, back-compatible wrappers:

```python
def run(self, module_id: str, config: dict) -> RunHandle:
    m = PROCESSING_MODULES[module_id]
    data = recording_store.read_recording(self.dir)
    config = m.resolve_upstream_spec(self.dir, data, config)   # a batch spec -> this recording's run id
    config = m.validate(self.dir, data, config)
    # Reserve the id and commit the record as PENDING under the recording lock, so
    # concurrent runs on one recording never collide on the id or lose an update.
    run_id = self._reserve_run(
        m.runs_key, m.id_prefix, lambda rid: m.run_record(rid, config, "pending")
    )
    ex = LocalExecutor(self._experiment_dir, runner_modules=_PIPELINE_RUNNERS)
    try:
        ex.submit(m.build_job(self.id, run_id, config))
    except Exception:
        self._set_run_status(m.runs_key, run_id, "failed")   # never an orphaned PENDING
        raise
    self._set_run_status(m.runs_key, run_id, ex.status(run_id).value)
    return _RecordedRun(ExecutorRunHandle(ex, run_id), self, run_id, m.runs_key)

def run_detection(self, model_id, prompt, compute, conf_threshold=0.5, confirm_fn=None,
                  model_variant=None, batch_size=None, every_n_frames=1) -> RunHandle:
    return self.run("detection", {"model_id": model_id, "model_variant": model_variant,
                                  "prompt": prompt, "compute": compute,
                                  "conf_threshold": conf_threshold,
                                  "batch_size": batch_size,
                                  "every_n_frames": every_n_frames})
```

`run_detection` accepts `confirm_fn` and passes it nowhere (Tenet 10 reserves it; no check uses it today).

`_RecordedRun` becomes a **single parameterized class** (constructed with `runs_key`) rather than the `DetectionRun` / `TrackingRun` / `EmbeddingRun` subclasses, which differed only by their `_LIST_KEY`.

### Dependency model

`upstream` is a single declaration per module — the minimal model that covers the linear chains we have and feeds both input-existence validation and the GUI's progression view:

| Module | `upstream` | Note |
|--------|-----------|------|
| `detection` | `"video"` | reads the recording's video, thinned by its own `every_n_frames` (§4) |
| `tracking` | `"detection"` | |
| `embedding` | `"tracking"` | |
| `keypoint_detect` | `"detection"` | §26 (revised 2026-10-21; `"tracking"` before) — raw per-instance placements, no subject, never filtered |
| `keypoint_track` | `"keypoint_detect"` | §26; assigns instances to a tracking run's subjects, then decodes and filters per subject; re-runs whole, no resume |

The chain reads `video → detection → tracking → embedding`, and **forks at detection**: `detection → tracking → embedding` and `detection → keypoint_detect → keypoint_track`. A module that needs **multiple** upstreams (a true DAG) stays out of scope; the second inputs are declared as **auxiliary inputs** — `ProcessingModule.aux_refs`, `(module_id, record_ref_field)` pairs a run *may* read beside its lineage parent — and there are two: embedding's alignment (§14), `(("keypoint_track", "keypoint_run_id"),)`, and keypoint tracking's subjects (§26 *Assignment*), `(("tracking", "tracking_run_id"),)`. One constraint keeps the chain linear: the auxiliary run and the module's own lineage must meet at one run — embedding refuses a keypoint track run whose `tracking_run_id` is not the embedding's tracking run, and keypoint track refuses a tracking run not on its detect run's detection run. Eligibility and the GUI progression read `upstream` alone; the auxiliary run's **spec** (its settings and lineage, never its id) is one of the module's own identity fields — embedding's `keypoint_track_identity`, keypoint track's `tracking_identity` — so per-recording runs with the same settings still match and the spec keeps its `{module, identity, parent}` shape; a batch run resolves a `<module>_spec` per recording by the same machinery as the upstream spec; and the run is a dependent of what it read (`_downstream_edges`, derived from the registry, §15 *Run deletion*).

### Upstream run specs

A batch run (`Experiment.run_processing` over N recordings) must pick a **per-recording upstream** — which detection run to track, which tracking run to embed. The per-recording run id (`det_NN`) cannot be that selector: it is a **per-recording monotonic counter** (`api/recording.py::_next_run_id`), so the *first* run in every recording is `det_01` regardless of what it is. Matching on it both **false-negatives** (the same logical batch is `det_02` in one recording, `det_01` in another → "no common run") and **false-merges** (two different runs share an ordinal → silent mixing). The fix: select by a canonical **run spec**, resolved to each recording's own run.

**Definition.** A spec is a run's identity — its own output-affecting settings **plus, recursively, the spec of its upstream input** (full lineage). It is a plain JSON-native structure so it round-trips through the stored run record:

```python
{"module": "tracking",
 "identity": {"tracker_id": "anchor_tracker", "arena_ids": [], "hyperparams": {...}},
 "parent": {"module": "detection", "identity": {...}, "parent": {"root": True}}}
```

Two runs are interchangeable iff their specs are equal. `api/run_spec.py::freeze_spec` normalizes a spec to a hashable key: floats rounded to 1e-6 (coarser than any UI-representable value — conf step 0.05 / 4 decimals — so repr noise and the JSON round-trip never split two real values), dicts → sorted `(key, frozen)` tuples, lists → tuples. The frozen key is **stable across a JSON round-trip** (list vs tuple canonicalized identically), which the stored-config path relies on.

**Identity per module** (record keys; `batch_size`/`created_at`/output path are *not* identity — they don't change output). The spec is derived from the **recording.yaml run record** (`extra_record_fields`), so two fields the record did not carry are added — both available in `config` at record time:

| module | `identity_fields` | `lineage_ref` (upstream_module, ref_field) | record change |
|--------|-------------------|--------------------------------------------|---------------|
| detection | `model_id, model_variant, weights_source, conf_threshold, every_n_frames, prompt_fingerprint, prompt_classes` | `None` (roots at the recording's video) | **+`prompt_fingerprint`** (short digest of `Prompt.to_dict()`; the prompt is the class definition for open-set models — omitting it false-merges "mouse" vs "cage") **and +`prompt_classes`** (the sorted target names — the human-readable half, redundant for equivalence so it can only ever false-*split*) |
| tracking | `tracker_id, arena_ids, hyperparams, every_n_frames` | `("detection","detection_run_id")` | **+`hyperparams`** (`config["hyperparams"]`, enriched in validate; was `job.json`-only) |
| embedding | `model_id, model_variant, constructor, every_n_frames, align, heading_definition, max_gap_s, mirror_hold_s, facing_camera_ratio, keypoint_track_identity` (the last six are alignment's, §14; `keypoint_track_identity` is the auxiliary keypoint track run's spec, `None` without alignment) | `("tracking","tracking_run_id")` (+ `aux_refs = (("keypoint_track","keypoint_run_id"),)`) | — (the constructor **name**, not the resolved `renderer_spec` — centered recipes size per-recording, so the resolved spec is intentionally recording-specific and must not be identity) |
| keypoint_detect (§26) | `model_id, model_variant, weights_source, backbone_id, backbone_variant, skeleton_digest, example_set_digest, n_candidates, every_n_frames, crop_px, decode, side_rule, matching` (the matching constants as one dict), `min_pose_confidence` | `("detection","detection_run_id")` | compute, batch size and the display cut-off are not identity |
| keypoint_track (§26) | `tracking_identity` (the tracking run's spec, never its id — ids restart per recording, §6.5 *auxiliary inputs*), `match` (`min_inside`, `mask_grow`), `method`, `config` (every field of the method's config, filter constants and decode cut-off included); the detect run's identity by lineage | `("keypoint_detect","detect_run_id")` | the display cut-off is not identity |

`every_n_frames` is identity on detection, tracking, embedding and keypoint_detect (§4): the same config at a different cadence is a different output. The first three carry `identity_defaults = {"every_n_frames": 1}`, which is how a record written before the field existed reads — without it those runs would false-*split* from equivalent ones written since; embedding's also defaults its alignment fields (`align: "none"` and the rest) for the same reason. keypoint_detect arrived after the field and needs none.

`lineage_ref` is distinct from `upstream`: detection's `upstream` is `"video"` (eligibility) and it has no lineage parent at all. These live as class attributes on `ProcessingModule` (record keys diverge from config-model field names — prompt/weights_source/hyperparams are enriched, not modelled — so they are **not** derived from config metadata). An **analysis** module declares `selects = (upstream_module, config_ref_field)` — explicit, since the ref field is not always `{module}_run_id` (`query`, `query_eval` and `temporal_clustering` select an embedding run by `embed_run_id`); analysis reuses the processing-module spec machinery for whatever it selects.

**Storage: derived on the fly** — no persisted hash, no migration; specs recompute from records. A **legacy** record (written before the two record additions) recovers the missing field from its `job.json` via `ProcessingModule.recompute_identity_from_job` (detection → `prompt_fingerprint`, tracking → `hyperparams`); unrecoverable → a per-run sentinel that only ever *false-splits* (safe), never false-merges. `freeze_spec`'s **false-merge is the dangerous direction** — when unsure whether a field affects output, include it.

**Resolution — three sites, two mechanisms.** The batch config carries `<upstream>_spec` (JSON), not a per-recording id; the **stored** run-record config keeps the spec so the detached coordinator re-resolves per recording.

*Processing* (per-recording fan-out) uses `ProcessingModule.resolve_upstream_spec` (→ `run_spec.resolve_into_config`), which injects a single resolved local id before `validate` at **(A)** `Recording.run` (coordinator runtime) and **(B)** `run_processing`'s up-front pre-validate loop. For tracking/embedding the config model makes the ref field *required*, so a spec-only config fails `model_validate`. Resolution must precede validate — and every live site does it.

*Analysis* is a **different mechanism** — one experiment-level job, not a fan-out, so `AnalysisModule` has no `resolve_upstream_spec`. **(C)** `Experiment.run_analysis` calls `module.validate`, and the module's validate calls `AnalysisModule.resolve_selection` → `run_spec.resolve_analysis_selection`, driven by the module's `selects = (upstream_module, ref_field)`. It resolves to a `{sid: run_id}` **map** from, in order: `<upstream>_spec` (each recording's own run with those settings); `<upstream>_specs`, a list (each recording's own run of the **one** listed spec it has a complete run of — **refused** when it has none or several, so list order never picks; for recordings whose upstream came through different chains, e.g. two detection models for two strains, which no single spec covers; passing it with `<upstream>_spec` is refused; when the list resolves to more than one spec and the upstream is `embedding`, the pooled runs must also pass `run_preview.validate_comparability`, because equal specs no longer vouch for the recipe and neither clustering module checks it itself; the GUI's `PooledSpecPicker` and a repeated `--embed-ref` / `--tracking-ref` emit it, and one tick or ref still emits `<upstream>_spec`); a single `ref_field` id (that id in every recording, checked present and complete); or neither (the one spec every selected recording has a complete run of, **refused** when there is none — naming the closest spec's missing recordings — or several, Tenet 9). The config is enriched with `<ref_field>s` (the map) and `<ref_field>` (the id when every recording's agrees, else `None`), which `build_job` freezes into the Job (the runner loops recordings internally and keys the map). Because analysis resolves once at launch (no coordinator re-run), the job carries the resolved map, not the spec. Readers go through `core/run_selection.py::run_for_recording`, which answers both the map and a run recorded before it (whose single id meant that id in every recording). Migrated: `location_export` (`tracking_run_ids`) and `temporal_clustering` (`embed_run_ids`, on the job, the state and the record's `params`). `query` and `query_eval` still match the embedding run on its id.

The explicit-id path (CLI names a run) stays working at all three — spec is additive.

**Labels always carry lineage** (`spec_label` → `anchor_tracker @ conf 0.25 · from sam3/large @ 0.25`) so "same tracker params, different detection model" is visible in the picker, not silently identical.

**Known limitations (v1).** A trained/imported model whose weights change under a stable `model_id` (`resume_training`, re-import) is a `weights_source`-level match — the weights *content* is not fingerprinted. Detection identity uses the whole `Prompt.to_dict()`, so an `ImageCue`'s per-run path can false-split two otherwise-equal image prompts (text/point/box prompts — the common case — are content-stable). Both are acceptable false-*splits* / narrow, and documented rather than merged. **Embedding identity is the constructor name, model, cadence and alignment — deliberately not `track_ids`**: an `all_tracks` run resolves to different concrete ids per recording, so including them would false-*split* the same logical run across recordings. The trade-off is a possible false-*merge* — two embedding runs over different track subsets of the same tracking parent share a spec — so an analysis module's spec-select of an embedding run (`embed_run_ids`, `--embed-ref`) can under-cover; documented rather than merged.

### Job and progress-event registration

Per-stage `Job` and `ProgressEvent` dataclasses stay typed (so nested types like `Prompt` and `renderer_spec` keep their `to_dict`/`from_dict` validation), but the central `_JOB_TYPES` / `_PROGRESS_EVENT_TYPES` maps in `core/dataclasses.py` are **replaced by dynamic registration** — a module registers its `(module_id → job_class)` (and event class) on import. Adding a module therefore registers its serialization without editing `core/`.

This imposes an **ordering requirement**: `job_from_dict` requires the owning module imported before deserialization. The worker and executor already call `import_runner_modules(...)` before touching jobs, so the runtime path holds; tests that deserialize a job must import its module first (the previous static map tolerated not doing so).


---

## 7. Detection Jobs

> Implemented as a `ProcessingModule` (§6.5), `upstream = "video"`. The schemas below are unchanged.

### Config vs job parameters

There are no experiment-level defaults: `config.yaml` holds identity only (§1). A launch form starts from **Start from** — the config model's defaults, a baseline, or a previous run's settings — and every parameter is set per job.

A detection job carries its full configuration (`core/dataclasses.py`):

```python
@dataclass
class DetectionJob:
    job_id: str
    recording_id: str
    model_id: str
    prompt: Prompt                      # empty for a closed-set (trained) model
    conf_threshold: float
    compute: ComputeTarget
    created_at: str
    model_variant: str | None = None    # None = the model's default variant
    weights_source: str = "pretrained"  # "pretrained" | "trained" | "imported" (§5, §12); resolved in validate
    batch_size: int = 1                 # frames per GPU forward pass (batch-capable models)
    every_n_frames: int = 1             # process every Nth source frame (1 = every frame, §4)
```

If the prompt contains an `ImageCue`, the API layer persists the image under the run directory and stores its experiment-relative path in the serialized job (see Prompt serialization, Section 5) before the job is submitted — the job dict never carries inline pixels.

Detection emits a `DetectionProgressEvent` on the executor's progress stream. Parts run in parallel across GPUs, so the coordinator aggregates per-part counts into one event; per-part detail lives in `job.json`:

```python
@dataclass
class DetectionProgressEvent:
    parts_done: int
    parts_total: int
    frames_done: int        # aggregate across all parallel parts
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None  # ephemeral phase text, e.g. "loading model"
```

Multiple detection runs per recording are supported — different models, prompts, or compute. Each gets its own ID and output directory. Each run is a `detection_runs` entry in `recording.yaml`, appended when the run is **submitted** (as `pending`, under the recording lock) and updated as its status changes, so a running or failed run is listed too; deleting a run removes its entry (§15 *Run deletion*). The entry is a lightweight index (`ProcessingModule.run_record` + `DetectionModule.extra_record_fields`); the prompt and compute live in the run's `job.json`:

```yaml
detection_runs:
  - id: "det_01"
    status: "complete"
    output: "detections/det_01/"
    created_at: "2026-05-22T10:00:00Z"
    model_id: "sam3"
    model_variant: null
    weights_source: "pretrained"
    conf_threshold: 0.5
    prompt_fingerprint: "3f9c2a71d0b48e15"   # sha1 of Prompt.to_dict(), 16 hex — run identity (§6.5)
    prompt_classes: ["mouse"]
    batch_size: 1
    every_n_frames: 2
    processing_run_id: "run_004"         # only when launched by an experiment-level run (§15)
```

### Execution: decode thread → prep → GPU → writer

Each part is one forward walk of its video (Tenet 11), batched to `batch_size`. `_detection_worker.run_part` overlaps four stages through `core/lookahead.py`, so the GPU does not wait on the CPU:

```
decode thread          prep thread              calling thread           writer thread
grab/retrieve, stride  model.prepare_batch  ─►  model.run_prepared  ─►   model.finish_batch, conf filter,
(2 batches ahead)      (2 batches ahead)        forward only             RLE encode, zarr rows, progress
```

- **The model's halves** are the optional split in the `DetectionModel` protocol (§5). A model that does not split runs `predict_batch` whole on the calling thread, with only decode and the writes overlapped.
- **The calling thread** checks the stop flag before each forward. On a stop the batches already computed are written, and `frames_done` is the source frames walked through the last batch written, so it never counts a frame with no rows.
- **One writer thread** keeps rows in frame order. A write error fails the part after the batches before it. The zarr writer is opened and closed on the calling thread, after the writer has joined.
- **Errors keep their position.** The frame-count rule (`FrameCountMismatchError`, the dropped tail) runs on the decode thread and is raised to the caller after every batch before it. Every stage is a generator closed by the one after it, so no thread outlives the part.
- **Rows are byte-identical to the serial loop**: the same calls on the same batches, in order. `tests/pipeline/test_detection_worker.py` holds the worker to rows built without it, and `tests/models/test_trained_yolo.py` holds `SplitPredict` to Ultralytics' own `predict`.

### Multi-GPU parallel processing

One independent process is assigned per video part, round-robin across available GPUs. Each worker reads its video part sequentially from start — no frame seeking, no shared state, no communication during execution. This is functionally identical to running independent detection scripts manually.

```
Recording: part_001.mp4, part_002.mp4, part_003.mp4 — 2 GPUs
  → part_001 → GPU 0
  → part_002 → GPU 1
  → part_003 → GPU 0
```

If parts > GPUs, remaining parts queue on the same GPUs after their first part finishes. If parts < GPUs, excess GPUs are idle — acceptable.

A list of indices in `ComputeTarget.device_indices` names the GPUs to use; `null` runs on GPU 0 alone (§5 *Compute targeting* — using every available GPU when none are named is not built).

**Implementation notes:**
- **Single-device short-circuit.** When `compute.backend != "cuda"` or `len(device_indices) <= 1`, detection runs inline (no `ProcessPoolExecutor`) — zero subprocess overhead for the common single-GPU / CPU / MPS case.
- **GPU pinning via env.** Each pool worker sets `CUDA_VISIBLE_DEVICES` to its assigned index **before any torch import**, then uses `cuda:0`. Cleaner than passing `cuda:N` strings, and required to make `fork`-vs-`spawn` CUDA semantics safe.
- **`spawn` multiprocessing context.** Forking after CUDA init corrupts driver state; the pool uses `multiprocessing.get_context("spawn")`.
- **Per-worker model load.** One `ProcessPoolExecutor(max_workers=1)` per assigned GPU; the pool's `initializer` loads the model once on that GPU. Parts dispatched round-robin reuse the loaded model.
- **Cross-process channel.** File-based: each worker writes its per-part progress JSON under `<run_dir>/parts/<part_id>.json` (a sibling `.tmp` moved into place with `core.atomic.replace`). Stop signal is `<run_dir>/stop.flag`; workers poll before each forward pass (*Execution* above).
- **Stop = failed (not partial).** A stopped detection run is reported as `status="failed"` rather than `"partial"`: detection does not resume, so `partial` would promise something it cannot do. A failed or stopped run is re-run, not continued (Tenet 12). Embedding treats a stop the same way (§14).
- **One worker failure ≠ run failure (mid-run); aggregate failure is final.** A worker that raises returns `status="failed"` to the coordinator via `pool_run_part`'s exception capture, and other parts continue. After all parts complete, the coordinator raises `RuntimeError` if any part failed — surfacing to the executor envelope as `status="failed"` for the whole run. A GPU out-of-memory error follows this path (*Validation checks* below).

### Directory structure

```
detections/
└── det_01/
    ├── job.json          # the envelope (§6) + detection's payload — written before any process launches
    ├── parts/
    │   ├── part_001.json # live per-part progress, written by the worker for that part
    │   └── part_002.json
    ├── part_001.zarr     # written incrementally by worker, one zarr per video part
    ├── part_002.zarr
    ├── activity.csv      # timestamped progress rows (§6)
    ├── worker.log        # the worker's stdout + stderr
    ├── machine.yaml      # what it ran on (§6), written when the job ends
    └── resources.yaml    # peak memory (§6), written when the job ends
```

### Monitoring

Each part's worker writes a small progress file, `parts/<part_id>.json` — `{frames_done, frames_total, status}` — atomically (a sibling `.tmp` moved into place with `core.atomic.replace`), and best-effort: a failed write never fails the part. The coordinator polls these files every `PROGRESS_INTERVAL_S` (0.5 s), folds them into one `DetectionProgressEvent` and publishes it to `job.json`'s `progress`, which is what the executor's `stream_progress()` yields. Filesystem-only IPC, no pipes or sockets.

When every part has finished, the coordinator writes the per-part results into `job.json` as the `parts` payload, beside `mask_resolution` (§9):

```json
{
  "parts": [
    {
      "id": "part_001",
      "file": "video/part_001.mp4",
      "gpu": 0,
      "status": "complete",
      "frames_done": 3600,
      "frames_total": 3600,
      "mask_resolution": [1080, 1920],
      "dropped_frames": [],
      "error": null
    },
    {
      "id": "part_002",
      "file": "video/part_002.mp4",
      "gpu": 1,
      "status": "failed",
      "frames_done": 0,
      "frames_total": 0,
      "mask_resolution": null,
      "dropped_frames": [],
      "error": "RuntimeError: detection ran out of GPU memory at batch_size=8 (…) — lower the batch size"
    }
  ]
}
```

`file` is the part's recording-relative path for a copied part and its stem (`part_002`) for a referenced one; `gpu` is `null` on the inline single-device path. There is **no stall detection**: nothing times out a worker whose progress stops moving. `activity.csv` (§6) is how a person tells a slow run from a stuck one.

### Recovery

Results are written to zarr incrementally using buffer-flush, so a worker that crashes mid-part leaves the frames written so far on disk — but detection does **not** resume from them. A failed or stopped detection run is re-run, not continued (Tenet 12): a processing run's retry (`Experiment.retry_run`, §15) runs each incomplete recording again, whole, under a fresh `det_NN`, so nothing appends to a half-written store.

**Part status states:**

| Status | Meaning |
|--------|---------|
| `pending` | Not started (the coordinator's own view before a part's progress file exists) |
| `running` | Its worker is processing it |
| `complete` | Every frame walked and written |
| `failed` | The worker raised, or the run was stopped; the error says which. Any failed part fails the run |

### Validation checks

**Pre-run — before any process launches** (`DetectionModule.validate`):

| Check | Action on failure |
|-------|-------------------|
| `model_id` resolves (pretrained, experiment-trained or machine-imported) | Error |
| Model supports the requested backend, and this install can run on it (`check_backend`) | Error — names the backends it needs |
| `batch_size` > 1 only on a model that batches | Error — set batch size to 1 |
| Requested CUDA indices exist (`check_cuda_indices`) | Error — list available devices |
| Prompt fits the model: none for a closed-set model, at least one target and only supported cue types otherwise | Error — list accepted cue types |
| Model `readiness().ready` — weights present / token / access (Section 5) | Error — raise `ModelNotReadyError` with the actionable message and `action_url` |
| The recording has video parts and every part is present and unchanged (`video_integrity`) | Error — list missing or changed files |
| No active detection run already on this recording | Error — prevent duplicate concurrent runs |
| Estimated VRAM fits on each assigned GPU | **Not built** |
| Estimated output size vs available disk space | **Not built** |

**During a part** (`_detection_worker.run_part`):

| Check | Action on failure |
|-------|-------------------|
| Model loads on its device (once per device, before any part) | The run fails with the load error |
| cv2 returns more frames than the timestamp map has | `FrameCountMismatchError` — part `failed` (§3 *Frame timestamp maps*) |
| cv2 stops early | The file's packets are recounted: agreeing with the map means a damaged tail, recorded in `dropped_frames`; disagreeing raises — part `failed` |
| Detection output has valid shape, no NaN/inf; mask dimensions match the frame | **Not built** |

Zero detections on a frame is valid — the frame is walked and simply writes no rows. `frames_done` counts it, so downstream stages know it was processed and returned nothing.

**After the parts:**

| Check | Action on failure |
|-------|-------------------|
| All parts are `complete` | The coordinator raises `RuntimeError` naming each failed part; the run is `failed` |

**Out of GPU memory:** a forward pass that runs out of memory is re-raised as `RuntimeError("detection ran out of GPU memory at batch_size=N (frame resolution …) — lower the batch size")`, which fails that part; other GPU workers continue, and the run fails with the message after they finish.

---

## 8. Prompt Testing

A lightweight pre-detection step for evaluating prompts before committing to a full run. Runs on M randomly sampled frames with N prompts in parallel, returns results immediately for visual inspection. No chunking, no recovery, no zarr output.

Prompt-test always samples the recording's own video and stamps `t_ns` from its active timestamp map, so its results join on the same time reference as everything else.

```python
@dataclass
class PromptTestJob:
    recording_id: str
    model_id: str
    prompts: list[Prompt]          # N prompts to compare
    n_sample_frames: int           # M random frames
    compute: ComputeTarget
    model_variant: str | None = None
```

There is no threshold on the job: the run keeps every proposal, and `Recording.run_prompt_test(..., min_conf=0.0)` takes the threshold as a **view** — what the saved images are drawn at and counted at (§9 *Confidence threshold*).

Results are returned directly — one entry per sampled frame, one `FrameResult` (§5) per prompt:

```python
@dataclass
class PromptTestResult:
    part_index: int                   # the video part the frame came from
    frame_idx: int                    # local to that part
    t_ns: int
    frame_image: np.ndarray
    results: dict[str, FrameResult]   # prompt label ("prompt_0", …) → that prompt's detections
```

The GUI displays a grid: M rows (frames) × N columns (prompts) with masks overlaid, plus a confidence slider that re-draws and re-counts the grid at any threshold (see "Confidence threshold" in §9 below — the preview kept every proposal, so this costs no re-run). User selects the best-performing prompt and reads off the threshold, then sets both on a full detection run.

Because the preview keeps every proposal, a mask per proposal per sampled frame per prompt is far more than a run at 0.5 ever held; `run_prompt_test_on_parts` therefore takes an `on_result` sink so a caller can consume each frame and drop it rather than accumulating full-resolution masks across the whole grid.

Results are optionally saved for later review (`save_as=<name>`; path relative to recording directory):

```
recordings/recording_001/detections/
└── prompt_tests/
    └── <name>/
        ├── test.json          # model_id, min_conf, n_sample_frames, compute; the sampled frames,
        │                      #   the prompts and their colours, and per-frame counts at min_conf
        └── frames/
            ├── part_000_frame_000042_prompt_0.jpg
            └── part_000_frame_000042_prompt_1.jpg
```

A result is identified by its `(part_index, frame_idx)`, never by its position: `sampled_frames` lists every position requested, while a sample that does not decode is skipped, so the results (and `test.json`'s `frame_results`, each of which names its part) can be shorter. The part is in the overlay name because `frame_idx` is local to its part.

Sampled frames are exact. Each is decoded with `video.read_frame_at` and the part's timestamp map (§3 *Frame timestamp maps*), so the picture is the requested presentation index, and `PromptTestResult.frame_idx` reports the frame actually decoded.

---

## 9. Detection Outputs

### Directory structure

Detection outputs live inside the recording they were derived from. All other processing stages that consume detection results look here.

```
my_experiment/
├── config.yaml
├── metadata.csv                        # grouping variables — one row per recording
├── heading_definitions.yaml            # saved heading definitions — see Section 14
├── recordings/
│   └── recording_001/
│       ├── recording.yaml
│       ├── video/
│       │   ├── part_001.mp4
│       │   └── part_002.mp4
│       ├── streams/
│       │   └── heart_rate/
│       │       ├── raw/
│       │       └── canonical/
│       ├── arenas/
│       │   └── arena_001/
│       │       └── keyframes.csv
│       ├── rois/
│       │   └── roi_001/
│       │       └── keyframes.csv
│       ├── points/                     # points of interest — see Section 10
│       ├── detections/
│       │   ├── prompt_tests/
│       │   │   └── test_01/
│       │   │       ├── test.json
│       │   │       └── frames/
│       │   └── det_01/
│       │       ├── job.json
│       │       ├── parts/              # per-part progress files the workers write
│       │       ├── part_001.zarr
│       │       └── part_002.zarr
│       ├── tracking/                   # tracking runs — see Section 13
│       ├── embeddings/                 # embedding runs — see Section 14
│       ├── keypoint_detect/            # keypoint detect runs — see §26
│       └── keypoint_track/             # keypoint track runs — see §26
├── runs/                               # one record per processing run (run_NNN) — see Section 15
├── analysis/                           # flat, one directory per run — see Section 16
│   ├── an_01/                          # analysis.yaml + that run's outputs
│   └── an_02/                          # the module is a field in the record, not a directory
├── comparisons/                        # saved comparisons (cmp_NN) — see §24
├── exports/                            # export run records; the exported files go to the user's folder
├── examples/                           # keypoint example sets — see §26
├── distill/                            # distill runs (label → train, one record) — see Section 12
├── models/
└── datasets/
```

`analysis/` is at experiment level because analysis runs may span multiple recordings. It is flat: one directory per run (`an_NN`), and the module that produced a run is a field in its `analysis.yaml`, not a directory level; each run is independent. `models/` and `datasets/` are experiment-level and shared across recordings.

### Per-part zarr schema

One zarr store per video part. Detection-indexed — one row per detection, not one row per frame. Multiple detections per frame produce multiple rows with the same `frame_idx`.

```
part_001.zarr/
├── frame_idx        (N,) int32       chunks=(10000,)   # frame this detection belongs to
├── t_ns             (N,) int64       chunks=(10000,)   # from active timestamp map — not recomputed
├── detection_idx    (N,) int32       chunks=(10000,)   # index within the frame (0, 1, 2...)
├── class_name       (N,) vlen-str    chunks=(10000,)
├── confidence       (N,) float32     chunks=(10000,)
├── bbox             (N, 4) float32   chunks=(10000, 4) # x1,y1,x2,y2 normalized
└── mask_rle         (N,) vlen-str    chunks=(1000,)    # ASCII COCO-RLE counts — larger per element, smaller chunks
```

Both vlen columns (`class_name`, `mask_rle`) use zarr v3's native variable-length UTF-8 dtype (`dtype="str"`, numpy `StringDType`), which is a stable, specified v3 data type. The `mask_rle` column stores the **ASCII RLE counts string**, not raw bytes — see Mask encoding below for why this is both correct and durable.

`object_id` is intentionally absent. Tracking writes identity to its own parallel zarr (`tracking/track_01/assignments/`). The detection zarr is immutable after writing.

Frames with zero detections have no rows. Coverage (which frames were processed) is derived from `job.json` part status — no redundant `processed_frames` array needed.

### Mask encoding

Masks are stored as COCO-style Run-Length Encoding (RLE). A 1080p binary mask is ~2MB as a bool array; as RLE a typical animal-sized mask is 1–5KB — 400–2000x smaller. Encoding and decoding uses `pycocotools`.

`pycocotools` returns `rle["counts"]` as a **bytes** object, but the compressed-RLE format guarantees those bytes are printable **ASCII**. We therefore store the counts as a `str` (vlen-UTF-8), not as raw bytes. This is deliberate: zarr v3's variable-length *bytes* dtype is an unstable extension (it emits `UnstableSpecificationWarning` and "may be unreadable by other Zarr libraries" / may change across zarr versions), whereas the variable-length *UTF-8* dtype is stable and specified. Storing the ASCII counts as a string is the durable choice and round-trips to byte-identical masks.

```python
from pycocotools import mask as mask_utils

# Encode (write path): store the ASCII counts string
rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
counts_str = rle["counts"].decode("ascii")        # -> goes into the mask_rle vlen-str column

# Decode (read path): reassemble the rle dict; size comes from job.json mask_resolution
rle = {"counts": counts_str.encode("ascii"), "size": mask_resolution}  # [H, W]
binary_mask = mask_utils.decode(rle)
```

The mask resolution (height, width) is recorded once at the detection run level in `job.json` — it is the same for all masks in a run, so it is not stored per row.

The detection pipeline performs this encode/decode. `storage/` imports no `pycocotools`: the writer receives the already-encoded ASCII `counts` string and the reader returns it verbatim.

### Zarr write idiom (v3)

This is the canonical pattern for **every** per-part zarr the pipeline writes (detection here, tracking assignments in Section 13, embeddings in Section 14, keypoints in §26). The storage layer targets zarr-python ≥ 3.2.

```python
import numpy as np

from detail.storage import zarr_store

root = zarr_store.open_group(path, mode="w")  # the retrying store — never bare zarr.open_group
root.attrs["schema_version"] = 1              # Tenet 7 — first write, before any array

# Create empty, grow on flush. Numeric columns can pass data= directly; vlen columns
# must create with shape+dtype then assign (create_array forbids data= together with dtype=).
frame_idx = root.create_array("frame_idx", shape=(0,), chunks=(10000,), dtype="int32")
mask_rle  = root.create_array("mask_rle",  shape=(0,), chunks=(1000,),  dtype="str")

def append_rows(arr, buf: np.ndarray) -> None:
    """Grow a zarr array along axis 0 and write buf into the new tail (replaces v2 .append)."""
    n = arr.shape[0]
    arr.resize((n + len(buf), *arr.shape[1:]))
    arr[n:] = buf

# Buffer-flush (Tenet 12): accumulate a fixed number of rows, then
#   append_rows(frame_idx, np.asarray(buf_frame_idx, dtype="int32"))
#   append_rows(mask_rle,  np.asarray(buf_counts_str, dtype=object))   # python str elements
# Reopen: every column is cut to the common length; arr.shape[0] is then the rows on disk.
```

- **Open / version:** `storage/zarr_store.py::open_group(...)`, then `root.attrs["schema_version"] = 1` before any array (Tenet 7). **Never `zarr.open_group` directly.** zarr commits each chunk file and each metadata rewrite with its own `tmp.replace(dst)`, which on a network share intermittently returns a transient `PermissionError` (WinError 5/32); `RetryingLocalStore` rides that out for *every* write the group goes on to make. The mitigation was applied per call site once — one `retry_on_lock` around each `create_array` — and that is precisely how the next module to write a zarr ended up covered on its `open_group` and bare on the four calls that write chunk data. A store is the seam every write passes through; a call site is not. Our own renames (a staged store swapped into place) are `core.atomic.replace`. Both are enforced by `tests/storage/test_zarr_seam.py`.
- **dtypes:** numeric → `"int32"` / `"int64"` / `"float32"`; both vlen columns → `dtype="str"` (StringDType). There is no `object`/`object_codec` route in v3 — `dtype=object` raises, and `dtype="bytes"` silently stores object pointers (corrupt). Always use `"str"` for vlen.
- **Grow / reopen:** `append_rows` (resize + tail-assign); `arr.shape[0]` is the row count on disk.

**Detection-indexed writers — whole frames on disk.** The detection zarr has multiple rows per frame. Detection does not resume: a failed or stopped detection run is re-run, each incomplete recording whole under a fresh run id (Section 7). The writer (`storage/zarr_io.py::PartZarrWriter`) still keeps two rules, so whatever reaches disk ends on a whole frame — as does the keypoint detect store, which has the same layout and does not resume either (§26):

- **Flush on frame boundaries.** A flush is emitted only when the buffer is full *and* a new frame begins, so every flush ends on a complete frame — a frame's rows are never split across a crash-survivable boundary.
- **Truncate to common length on reopen.** A crash can leave columns at different lengths (some resized/assigned, some not). On reopen, every column is truncated to the minimum length across columns, which lands exactly on the last completed flush. This is the standard recovery step for all per-part writers. Reopen also verifies each column's trailing shape against the expected schema (e.g. the `(D,)` embedding dimension); a mismatch is a hard error, so reopening a store with the wrong schema fails loudly instead of corrupting data.

The writer exposes `rows_written` (`= arr.shape[0]`) and the reader `last_frame_idx` (rows are frame-ordered, so this is the last row's `frame_idx`).

### Sharding (v3)

**Not built.** Nothing in the tree sets `shards=`: every store is chunks-only, which is right for the local disk, the only backend that exists (Section 2's `StorageLocation` abstraction is not built either). The design, for when an object-store backend lands: a "part" produces one small chunk file per array per ~`chunk_size` rows. On a local disk that is fine; on an object store the file count drives cost and latency. zarr v3 can pack many chunks into one storage object via **sharding** (`create_array(..., chunks=(C,), shards=(S,))` with `S` a multiple of `C`). The shard policy would be owned by the `StorageLocation` backend: the local backend chunks-only (`shards=None`); object-store backends a shard size (default: `S = 10 * C` along axis 0, tunable). Flush buffer sizes follow Tenet 12 — a whole multiple of the chunk size, not a divisor of it. Read granularity and the schema's `chunks=` values are unchanged by sharding (`arr.chunks` stays the inner chunk; `arr.shards` reports the shard).

### Confidence threshold

Two thresholds, and they are not the same thing.

**A detection run's threshold** is specified at job creation. Only detections at or above it are written. Recorded in `job.json` for reproducibility. To try a different threshold, create a new detection run — the rows below it were never stored, so nothing later can recover them. This is deliberate: a full run over an experiment writes an RLE mask per row, and keeping every proposal would multiply the store without bound.

**A preview's threshold is a view, not a filter.** The prompt test (§8) and the Quick segment tool (§27) push `KEEP_ALL_PROPOSALS` (`configs/display.py`) into `set_conf_threshold` and keep everything the model returns, because SAM3 filters inside `Sam3Processor` *before* returning and a threshold applied at run time destroys the low-confidence rows rather than hiding them. The threshold is then chosen while looking at the result: `ConfidenceView.min_conf`, rendered as a slider in the GUI (`gui/confidence.ConfidenceRow`, shared by every surface that shows detections) and as `--min-conf` in the CLI. It changes what is drawn, counted and exported — never what was run, so the same result answers every threshold without re-running. §13's tracking-preview slider is the same control and should reuse the same widget.

A count computed at run time cannot answer a question asked at a different threshold, so a view that shows totals must recount from the stored confidences rather than reuse a stored number.

### `job.json` complete schema

The generic executor envelope (`schema_version`, `status`, `job`, `progress`, `error`, and the coordinator's `pid` once it starts — see "Run directory and `job.json`" in Section 6) plus the detection-specific payload (`mask_resolution`, `parts[]`) that the detection runner merges in when the run ends (`rundir.update_payload`). The `job` block is `job_to_dict(DetectionJob)`; each `parts[]` entry is the per-part result `_detection_worker.run_part` returns. Live per-part progress is not here: workers write it to `parts/<part_id>.json` beside `job.json` while the run is going.

```json
{
  "schema_version": 1,
  "status": "complete",
  "job": {
    "type": "detection",
    "job_id": "det_01",
    "recording_id": "recording_001",
    "every_n_frames": 2,
    "model_id": "sam3",
    "model_variant": null,
    "weights_source": "pretrained",
    "prompt": {
      "targets": [
        {"class_name": "mouse", "cue_type": "text", "cue": "mouse"}
      ]
    },
    "conf_threshold": 0.7,
    "batch_size": 8,
    "compute": {"backend": "cuda", "device_indices": [0, 1], "cloud_provider": null, "cloud_instance": null},
    "created_at": "2026-05-22T10:00:00Z"
  },
  "progress": null,
  "error": null,
  "pid": 41237,
  "mask_resolution": [1080, 1920],
  "parts": [
    {
      "id": "part_001",
      "file": "video/part_001.mp4",
      "gpu": 0,
      "status": "complete",
      "frames_done": 3600,
      "frames_total": 3600,
      "mask_resolution": [1080, 1920],
      "dropped_frames": [],
      "error": null
    },
    {
      "id": "part_002",
      "file": "video/part_002.mp4",
      "gpu": 1,
      "status": "complete",
      "frames_done": 3200,
      "frames_total": 3200,
      "mask_resolution": [1080, 1920],
      "dropped_frames": [4521],
      "error": null
    }
  ]
}
```

`dropped_frames` lists frames the decoder could not deliver (a read failure or a truncated tail); a stopped part records none. `error` is the part's own failure message, `null` when it completed.

---

## 10. Spatial Definitions

Spatial definitions describe the physical recording space and are stored per recording. They are user-defined by drawing polygons on video frames in the GUI.

### Arenas

An arena is a named polygon that defines the valid detection region within a recording's recording space. The detection run stores every detection; a tracking run that names an arena treats detections outside the arena's polygon at that frame as ineligible (see *Tracking with arenas* below and Section 13). A recording may have any number of named arenas.

### ROIs (Regions of Interest)

An ROI is a named polygon used during behavioral analysis to spatially tag or query detections — e.g., "what poses are most common near the lever?" ROIs are independent from arenas: arenas are processing-time filters, ROIs are analysis-time queries. A recording may have any number of named ROIs.

### Points of Interest (POIs)

A POI is a named **point** (a single normalized vertex) marking a static landmark in the recording space — a lever, a nose-poke port, a food hopper. Like ROIs, POIs are **analysis-time** definitions (e.g. "time spent within 5 cm of the lever"), not processing filters. A POI is structurally a one-vertex polygon: it reuses the same `keyframes.csv` format with `n_vertices = 1`, so a moving camera can be tracked by adding keyframes, while the common static case is a single keyframe. A recording may have any number of named POIs.

POIs are **distinct from pose keypoints**: a POI is a *user-placed* landmark fixed in the apparatus, whereas pose keypoints are per-frame *animal body* landmarks produced by the keypoint stages (§26).

### Active interval (ROIs and POIs)

An ROI or POI may be present for only a **fraction** of the recording. Its `recording.yaml` entry may carry an optional `active: {start, end}` (recording-global frame indices, inclusive); outside that interval the definition is **not present** and per-frame lookup returns nothing. When `active` is absent (or null) the definition spans the whole video (the default, clamped as below). **Arenas have no active interval** — a processing-time filter is always in effect — so this field applies to ROIs and POIs only.

### Directory structure

Arenas, ROIs, and POIs follow the same layout. Each named definition gets its own directory with a stable numeric ID. The human-readable name, vertex count, and (ROIs/POIs) optional active interval are stored in `recording.yaml`.

```
recording_001/
├── arenas/
│   ├── arena_001/
│   │   └── keyframes.csv
│   └── arena_002/
│       └── keyframes.csv
├── rois/
│   ├── roi_001/
│   │   └── keyframes.csv
│   └── roi_002/
│       └── keyframes.csv
└── points/
    └── point_001/
        └── keyframes.csv
```

### `recording.yaml` spatial section

```yaml
arenas:
  - id: "arena_001"
    name: "main_arena"
    n_vertices: 4
  - id: "arena_002"
    name: "nest_zone"
    n_vertices: 6

rois:
  - id: "roi_001"
    name: "lever_zone"
    n_vertices: 4
    active: null                 # whole video (default)
  - id: "roi_002"
    name: "test_phase_zone"
    n_vertices: 4
    active: {start: 1500, end: 3000}   # recording-global frames, inclusive

points:
  - id: "point_001"
    name: "lever"
    n_vertices: 1
    active: null
```

Arenas omit `active` entirely (always in effect). Renaming an arena, ROI, or POI updates only the `name` field in `recording.yaml` — the directory ID is stable.

### Keyframe CSV format

Each arena and ROI stores a list of keyframes. The column count is fixed at `1 + 2 * n_vertices` and determined by the `n_vertices` declared in `recording.yaml`.

```
frame_idx,v0x,v0y,v1x,v1y,v2x,v2y,v3x,v3y
0,0.10,0.10,0.90,0.10,0.90,0.90,0.10,0.90
1500,0.15,0.12,0.88,0.12,0.88,0.92,0.15,0.92
```

All coordinates are normalized to [0, 1] — (0, 0) is top-left, (1, 1) is bottom-right.

### Per-frame polygon lookup

`frame_idx` in `keyframes.csv` is **recording-global** — a cumulative frame counter where 0 is the first frame of `part_001` and each subsequent part continues the count. For a single-part recording this is identical to the part-local frame index. A multi-part recording where the arena shifts at a part boundary expresses that by adding a keyframe at (or just after) that boundary's global frame.

For any frame F (global), the active polygon is computed as follows:

1. Find the two keyframes immediately before and after F (by `frame_idx`).
2. Linearly interpolate each vertex between them.
3. If F is before the first keyframe or after the last, clamp to the nearest keyframe (no extrapolation).

For an ROI/POI with an `active` interval, F outside `[start, end]` returns **no polygon** (the definition is not present); clamping applies only within the active span. Arenas (and ROIs/POIs without an interval) are present for every frame.

The vertex count is fixed per definition — the user draws an N-vertex polygon (or a 1-vertex point) at creation and the vertices move, but the topology does not change.

### Tracking with arenas

A tracking run accepts **at most one** `arena_id`. Multi-arena combination semantics (union vs intersection vs include/exclude) are not yet specified; a recording may still hold many arenas, and different tracking runs may each pick a different one. `TrackingJob.arena_ids` remains a list to preserve forward compatibility — the API rejects more than one element today and the field can carry combination semantics later without a schema change.

Within the selected arena, a detection's bounding-box center (in the same normalized coordinates as the polygon) is tested against the interpolated polygon at that frame. Inside-or-on-edge keeps the detection eligible for matching and track birth; outside fixes `object_id = -1` for that row. Active tracks still age on every processed frame regardless of whether any candidates passed the filter.

---

## 11. Grouping Variables

Grouping variables are recording-level metadata used to compare groups during analysis (e.g., treatment vs. control, animal ID, condition). They live in a single CSV at the experiment root.

### `metadata.csv`

```
recording_id,animal_id,treatment,group,notes
recording_001,M001,drug_A,experimental,
recording_002,M002,vehicle,control,
recording_003,M001,vehicle,control,day 2
```

- `recording_id` is the key column — always present, auto-populated when a new recording is created.
- All other columns are user-defined. New columns can be added at any time.
- Type inference from values (numeric, string, datetime) is applied at analysis time.
- The file is the single source of truth. There is no generated/user split — all content is user-owned.

**Editing.** Cell-by-cell in the GUI's **Recording information** section (+ Add column), per-row in the Add recordings dialog, or in bulk via **CSV import** (`Experiment.import_metadata_csv` / `detail experiment import-metadata --csv`; GUI "Import CSV"). Import is **merge-only**: it sets the cells present, leaves recordings/columns not in the CSV untouched, and adds new columns. Requirements: a UTF-8 CSV with non-empty, non-duplicate headers and one **key column** — `recording_id`, else `name`, else a video-file-name column (`metadata.VIDEO_KEY_HEADERS`: `file`, `file_name`, `filename`, `video`, `video_file`, `video_name`, compared ignoring case, spaces and hyphens), or any column the caller names as `key_column` (matched as a video file name unless it is `recording_id` / `name`). With none recognised and none named, the import is refused with `MissingKeyColumnError`, which carries the header so a surface can ask (the GUI does; the CLI takes `--key`). A **video file name** matches on its base name, case-insensitive, extension optional, against every video part of every recording (`Experiment._video_file_matches` — the same rule `apply_stream_preset(by_video=True)` uses, #330). Key values must be non-empty; **every row must match exactly one existing recording, and no two rows the same one** — an unknown key, a file name shared by several recordings, or a duplicate is a hard error, all reported at once, and the whole import is rejected (nothing applied; Tenet 10). The key column and the derived export columns `name` / `file` are never stored as values. **Blank cells are skipped** (kept, never cleared). Export writes `recording_id`, `name`, `file` (the first video part's file name), then every column. `storage/metadata.py`: `load_import_csv` + `validate_import` (pure) → `apply_import` (reuses `upsert_row` / `add_column`).

---

## 12. Custom Model Training

> **Fit / train family — one of three module families (see §16).** This section specifies *detection-model* training concretely, but the pattern generalizes: a fit module consumes data (an image dataset here; **embeddings** for a behavior classifier — design, not built), supervised (with labels) or unsupervised (clustering), and produces a **reusable model** registered into the model registry. That model is then consumed by an *apply* step — which is an ordinary processing module (§6.5) naming the fitted model's id in its config, exactly as detection takes `model_id="sam3"`. Applying a model is therefore **not** a separate family. When a non-detection fit module is built it follows the same module-contract + registry pattern as `ProcessingModule`. A behavior **classifier** (supervised, over embeddings) is still unbuilt.
>
> **Implemented (detection training, this section):** the `Trainer` protocol + `TRAINERS` registry, `UltralyticsYOLOTrainer`, `pipeline/dataset_export.py` + `pipeline/training.py`, the `Experiment.generate_dataset` / `train_model` API (+ `DatasetRun` / `ModelRun` handles), the trained-model *apply* path (`TrainedYOLO` + `weights_source="trained"`, closed-set), dataset/model deletion (`docs/gui_design.md` §1.1), and the GUI fit surface (the **Datasets** and **Models** sections of an experiment). **Also implemented:** the fresh SAM3-labeling dataset source (`pipeline/labeling.py`, `Experiment.label_dataset`, the `dataset label` CLI, and the GUI `GenerateLabelingDialog`). **Not yet built:** a supervised behavior classifier.

Training a custom detection model is a two-phase workflow: dataset generation followed by model training. Both phases are model-agnostic — the training backend is a pluggable implementation of the `Trainer` protocol, analogous to how `DetectionModel` abstracts inference backends in Section 5.

> **Dataset source — design decision (current direction).** A training dataset's
> *format* is **source-agnostic** (images + YOLO polygon labels + `dataset.yaml`,
> which records its `source`). The **primary, first-built source is exporting from
> an existing detection/tracking run** — decode that run's masks into labels +
> extract the frames, **no new inference**, fully local and testable; a *tracking*
> run is preferred so labels can be restricted to tracked / in-arena detections.
> The **fresh SAM3 labeling pass** (sampling frames + running SAM3, the original
> Phase 1 design below) is **built** (`pipeline/labeling.py`) as a *second* source
> for "labels without a full detection run" — it writes the same dataset format,
> with no change to the trainer or the apply path. Everything downstream of the
> labels (frame reading, image extraction, mask→polygon, val split, the trainer,
> trained-model detection) is built **once** and shared across sources
> (`pipeline/_dataset_common.py`). This is the whole training workflow's *distill
> SAM3 → a fast YOLO* story; the run-export source just reuses masks you already
> computed instead of recomputing them.

### Two phases

**Phase 1 — Dataset Generation (export from a run; current direction).** The user picks an existing detection/tracking run (the experiment run + its per-recording runs) and a frame selection; the tool decodes that run's masks into YOLO polygon labels and extracts the chosen frames as images into `experiment/datasets/{dataset_id}/`. *(Alternate source — fresh SAM3 labeling, built (`pipeline/labeling.py`):* the user selects source recordings + classes (each with a SAM3 text prompt), the tool samples N frames proportionally across recordings and runs SAM3 per frame (all class prompts against the one encoded image). Same YOLO-seg output; gated on SAM3.)*

**Phase 2 — Model Training**: The user picks a dataset, a trainer, and a base pretrained model. A `TrainingJob` is submitted to the executor. The trainer writes its run artifacts into `experiment/models/{model_id}/`. On completion `model.yaml` is finalized (`status: complete`, `model_family`, `classes`, relative `weights.best/last`); the model is discovered by **scanning** `experiment/models/` (no central index — same pattern as recordings/runs/datasets) and is immediately available to any detection run via `weights_source: "trained"` (Section 5).

---

### Dataset export from a run (primary source)

`Experiment.generate_dataset(name, source_run_id, frame_selection, val_split, …)` builds a dataset from an existing **detection or tracking run** — the experiment processing run (`run_NNN`) and its per-recording runs. No new inference: it reuses the run's stored masks. A *tracking* run source can restrict to assigned tracks / in-arena detections (drop `object_id < 0`) for clean labels.

**Frame reading is forward-sequential, not seek-based (Tenet 11).** Export is a batch pipeline operation, so it reads each recording's source video (the same one detection used) **start-to-finish with `cv2` `cap.read()`**, picking off the selected frames as it passes them. This is required both by Tenet 11 *and* for correctness: jumping to a frame in H.264 can land on a neighbouring keyframe, so the extracted image would not match the mask used as its label. Reading sequentially in the same order detection processed the part guarantees image ↔ label frame-exactness. (A sparse selection still reads the whole video — pure I/O, no inference.)

For each selected frame that has a mask in the run: write the frame as `images/{train|val}/<stem>.jpg` (cv2), and write one YOLO label line per detection — `class_id x1 y1 x2 y2 …` (normalized polygon from `cv2.findContours` on the decoded mask, dropping contours under `min_contour_area`). Class ids come from the run's `class_name`s (a stable name→id map recorded in `dataset.yaml`). The val split is stratified per recording. The job is a `DatasetExportJob` (executor job: progress via `LabelingProgressEvent`). Examples are staged as they are written and split once the set is known (`_dataset_common.StagedExamples`). A stop keeps what was exported and records the dataset as `status: partial`; there is no resume — a failed or unwanted partial export is re-run.

```python
@dataclass
class DatasetExportJob:
    job_id: str
    dataset_id: str
    name: str
    source_run_id: str        # experiment processing run (tracking | detection)
    frame_selection: dict     # e.g. {"mode": "every_n", "n": 10} | {"mode": "all"}
    val_split: float
    min_contour_area: int     # px; drop tiny contours
    created_at: str
```

`dataset.yaml` records `source: {kind: "run_export", run_id: …, module: …, frame_selection: …}` (`module` is the source run's `module_id`, detection or tracking) so the dataset is reproducible and the format stays **source-agnostic** — the SAM3 labeling pass (below) writes the same layout with a different `source`.

> *Mask → polygon* is the only new conversion (we already store masks as COCO-RLE for detection): decode RLE → `cv2.findContours` → flatten + normalize to `[0,1]` against the frame size. A label line is one polygon and an instance is routinely more than one piece — a mask splits at a thin neck, a tail, a limb across the body — so **every** contour at or above `min_contour_area` is kept and the pieces are spliced into one ring through zero-width corridors (the shape Ultralytics' own COCO converter writes, `data/converter.py::merge_multi_segment`). Keeping only the largest piece is the bug this replaced: the piece that detaches is always an extremity, and Ultralytics derives the label's *box* from the polygon it is handed (`ops.segments2boxes`), so the box shrank with it.

---

### `Trainer` protocol

Callers never interact with training backends directly. All training goes through this protocol, analogous to `DetectionModel` for inference.

```python
class Trainer(ABC):
    trainer_id: str                        # e.g. "ultralytics_yolo"
    model_family: str                      # which DetectionModel family this produces
    supported_cue_types: list[type[Cue]]   # cue types the trained model will accept
    supported_backends: list[str]          # ["cpu", "cuda", "mps"] — subset ComputeTarget.backend
    config_model: type | None = None       # pydantic hyperparameter model (§6.5); None until declared

    def is_ready(self) -> bool:
        """Base weights downloadable/present and runtime installed."""

    def prepare_dataset(self, dataset_dir: Path, work_dir: Path) -> Path:
        """
        Converts or validates the experiment dataset into the format this trainer requires.
        Returns path to the trainer-specific dataset descriptor (e.g. data.yaml).
        Output is cached in work_dir — repeated calls are fast.
        The dataset_dir is never modified.
        """

    def train(
        self,
        job: TrainingJob,
        dataset_dir: Path,
        model_dir: Path,
        device: str,
        progress_cb: Callable[[TrainingProgressEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None: ...

    def resume(
        self,
        job: TrainingJob,
        dataset_dir: Path,
        model_dir: Path,
        device: str,
        progress_cb: Callable[[TrainingProgressEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None: ...

    def checkpoint_path(self, model_dir: Path) -> Path | None:
        """Last checkpoint available for resume, or None."""

    def best_weights_path(self, model_dir: Path) -> Path | None:
        """Best weights after training completes, or None."""

    def final_metrics(self, model_dir: Path) -> dict[str, float]:
        """Trainer-specific metrics dict — recorded into model.yaml on completion."""

    def metric_history(self, model_dir: Path) -> list[dict[str, float]]:
        """One row per finished epoch, oldest first (`epoch` plus the trainer's metrics),
        read from the trainer's own on-disk record. Empty before the first epoch ends."""

    def validate_hyperparams(self, hyperparams: dict) -> dict:
        """
        Validates hyperparams dict against this trainer's config model.
        Fills in defaults for missing keys. Raises ValueError on unknown or wrong-type keys.
        """

    def resolved_settings(self, model_dir: Path) -> dict:
        """What the backend actually trained with — its name, version and every resolved
        setting, machine paths left out — recorded as model.yaml → trained_with."""
```

---

### Trainer registry

Trainers self-register by importing their module. A build that excludes a backend simply does not import that module — the registry never sees it and the rest of the system is unaffected. This is the guaranteed mechanism for commercial or restricted builds to exclude specific backends such as Ultralytics.

```python
TRAINERS: dict[str, type[Trainer]] = {}

def register_trainer(trainer_id: str):
    def decorator(cls):
        TRAINERS[trainer_id] = cls
        return cls
    return decorator
```

`UltralyticsYOLOTrainer` lives in `trainers/ultralytics_yolo.py` and does `from ultralytics import YOLO` only inside its own methods — never at module top level. If that module is not imported, Ultralytics is never referenced.

```python
# trainers/ultralytics_yolo.py
@register_trainer("ultralytics_yolo")
class UltralyticsYOLOTrainer(Trainer):
    trainer_id = "ultralytics_yolo"
    model_family = "yolo"
    supported_cue_types = []          # closed-set: the trained model takes no prompt cues
    supported_backends = ["cpu", "cuda", "mps"]

    # configs/training.py — the single source for each hyperparameter's type, bounds and
    # default: epochs 100, batch_size 32, imgsz 640, patience 100 (extra="forbid").
    config_model = UltralyticsYOLOConfig

    def validate_hyperparams(self, hyperparams: dict) -> dict:
        return self.config_model.model_validate(hyperparams).model_dump()

    def train(self, job, dataset_dir, model_dir, device, ...):
        from ultralytics import YOLO    # only imported here
        ...
```

Adding a new training backend requires only implementing `Trainer`, calling `register_trainer`, and importing the module at startup — nothing else changes.

---

### `Trainer` protocol has no Ultralytics dependency

The following have zero imports from `ultralytics` or any other training framework:

- `Trainer` protocol
- `TrainingJob`, `LabelingJob`, `TrainingProgressEvent`, `LabelingProgressEvent` dataclasses
- `resolve_device()` utility
- Dataset generation code (writes plain text files)
- `model.yaml` / `dataset.yaml` read/write

The dataset label format (class ID + normalized polygon coordinates in `.txt` files) is a plain text convention. Reading and writing it requires only Python stdlib — no Ultralytics import.

---

### Shared `resolve_device` utility

Both `DetectionModel.load()` (Section 5) and `Trainer.train()` need `ComputeTarget` → device string mapping. This lives in one shared utility so neither protocol re-implements it.

```python
def resolve_device(compute: ComputeTarget, supported_backends: list[str]) -> str:
    """
    Maps ComputeTarget to a device string.
    Raises ValueError if compute.backend is not in supported_backends, or if it is
    supported but has no local device mapping (e.g. "cloud").

    cuda + device_indices=[0, 1]  →  "0,1"   (Ultralytics-compatible format)
    cuda + device_indices=None    →  "0"
    mps                           →  "mps"
    cpu                           →  "cpu"

    Note: the "0,1" multi-GPU string format is Ultralytics-compatible.
    PyTorch-native trainers using DataParallel or DDP need a list of ints, not a string.
    Those trainers should parse the returned string or inspect compute.device_indices directly.
    """
```

---

### Standard PyTorch model loading requirements

All `DetectionModel`, `EmbeddingModel`, and future inference protocol implementations that build a PyTorch module themselves must follow this pattern in `load()`:

```python
def load(self, device: str) -> None:
    self.model = build_model(...)
    self.model.to(device)
    self.model.eval()                          # required — disables dropout, sets BatchNorm to inference mode
```

All inference forward passes must be wrapped in `torch.inference_mode()` (not `torch.no_grad()` — `inference_mode` is faster and more memory-efficient). CUDA implementations must additionally wrap with `torch.autocast(device_type='cuda', dtype=torch.float16)` for mixed-precision throughput (`models/sam3.py`, `models/dinov3.py`). Ultralytics-backed models (`models/_ultralytics.py`) run under `torch.inference_mode()` and leave precision to Ultralytics' own predictor.

`torch.compile` is not used by any model today. It is an optimization to measure on the real workload (skill `validate-a-perf-claim`) before adopting, not a requirement.

---

### Compute targeting for training

Identical ideology to inference (Section 5). `ComputeTarget` is set per training job, not globally. Multi-GPU maps `device_indices` to the trainer's own multi-device API (Ultralytics uses `device=[0,1]`, PyTorch uses `DataParallel` or `DistributedDataParallel` — each trainer handles its own translation after `resolve_device` validates the request).

MPS and CPU are supported if the trainer declares them in `supported_backends`. `UltralyticsYOLOTrainer.supported_backends = ["cpu", "cuda", "mps"]` — Ultralytics supports MPS training on Apple Silicon. A trainer that only runs on CUDA declares `supported_backends = ["cuda"]` and the executor rejects MPS/CPU jobs before any work starts.

---

### Job dataclasses

`hyperparams` is a free `dict` so `TrainingJob` remains a single serializable class with no subclassing. Validation of trainer-specific parameters happens in `trainer.validate_hyperparams()` before the job starts.

```python
@dataclass
class LabelingClass:
    class_id: int
    name: str
    prompt: str       # SAM3 text prompt for this class

@dataclass
class LabelingJob:
    job_id: str
    dataset_id: str
    name: str                          # the dataset's display name
    source_recording_ids: list[str]    # empty = all recordings in experiment
    classes: list[LabelingClass]
    n_frames: int
    logit_threshold: float
    min_contour_area: int            # pixels; reject masks smaller than this
    val_split: float                 # fraction held out for validation (0.0 = no val set)
    compute: ComputeTarget
    created_at: str
    sampling: str = "stratified"     # within-part draw: "stratified" | "uniform"
    model_id: str = "sam3"           # the prompt-capable teacher
    model_variant: str | None = None
    background_share: float = 0.0    # cap on empty-label frames; the API passes
                                     # LabelingConfig's resolved value (default 0.1)

@dataclass
class TrainingJob:
    job_id: str
    model_id: str
    dataset_id: str
    trainer_id: str        # key into TRAINERS registry
    base_model: str        # pretrained starting point — trainer interprets this string
    hyperparams: dict      # trainer-specific; validated by trainer.validate_hyperparams()
    compute: ComputeTarget
    created_at: str

@dataclass
class TrainingProgressEvent:
    epoch: int
    total_epochs: int
    metrics: dict[str, float]    # trainer-specific keys, e.g. {"box_map50": 0.85}
    elapsed_s: float
    eta_s: float | None
    message: str | None = None   # the phase line (§6 Progress reporting)

@dataclass
class LabelingProgressEvent:
    frames_done: int
    frames_total: int
    classes_done: int            # classes fully labeled across the sampled frames
    classes_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None
```

Dataset generation (Phase 1) is a long SAM3 run over the sampled frames and emits `LabelingProgressEvent` on the executor's progress stream, the same way detection, tracking, and embedding report progress. Training (Phase 2) emits the epoch-based `TrainingProgressEvent`.

`base_model` is a string the trainer owns. For `UltralyticsYOLOTrainer` this is a model name like `"yolo11s-seg.pt"` that Ultralytics auto-downloads. For a future trainer it could be a HuggingFace model ID or a path to a local checkpoint.

---

### Dataset directory structure

```
datasets/
└── dataset_001/
    ├── dataset.yaml
    ├── data.yaml              ← path: .  (portable; the trainer trains from a machine-local copy)
    ├── images/
    │   ├── train/
    │   └── val/               ← absent if val_split: 0.0
    └── labels/
        ├── train/
        └── val/
```

`data.yaml` (plain YAML, no Ultralytics import required to read or write). `path: .` keeps the file portable, but Ultralytics resolves a relative `path:` against the current working directory, not this file — so `UltralyticsYOLOTrainer` trains from a copy in the machine-local weights cache whose `path` is the absolute dataset directory (`storage/datasets.py::write_data_yaml`, `trainers/ultralytics_yolo.py::_training_data_yaml`). A pose dataset adds `kpt_shape` and `flip_idx` (§12 *Distill keypoints*):

```yaml
path: .
train: images/train
val: images/val
names:
  0: mouse
  1: food_pellet
```

`dataset.yaml` — as the SAM3 labeling source writes it (`pipeline/labeling.py`); the run-export source writes the same keys with `source: {kind: run_export, run_id, module, frame_selection}` and no `background_share` / `n_background` / `skipped` / `per_class`:

```yaml
schema_version: 1
id: "dataset_001"
name: "mouse_and_pellet"
created_at: "2026-05-22T10:00:00Z"

source:
  kind: "labeling"
  model: "sam3"               # the prompt-capable teacher (LabelingJob.model_id)
  model_variant: null
  recording_ids: ["recording_001", "recording_002", "recording_003"]
  classes:
    - {class_id: 0, name: "mouse", prompt: "mouse"}
    - {class_id: 1, name: "food_pellet", prompt: "food pellet"}
  n_frames: 10000
  logit_threshold: 0.5
  min_contour_area: 100
  sampling: "stratified"

classes:
  - id: 0
    name: "mouse"
  - id: 1
    name: "food_pellet"

val_split: 0.15
background_share: 0.1       # labeling source: the cap on empty-label frames
n_train: 8494               # split drawn over the examples written, so
n_val: 1506                 # n_val / (n_train + n_val) is val_split (per-recording rounding)
n_background: 1000          # empty-label frames among them, <= background_share of the total
skipped: {undecodable: 0, below_min_area: 3}
per_class: {"0": 8120, "1": 2290}   # labelled instances per class id
status: "complete"          # "running" | "complete" | "partial" | "failed"
```

### Frame sampling *(SAM3 labeling source)*

> The `LabelingJob` / `LabelingClass` types and the two subsections below specify
> the fresh-SAM3-labeling source (`pipeline/labeling.py`, see the design-decision
> note at the top of §12). They write the same dataset layout as the run-export
> source. Frames are read by seeking, each decoded exactly (`read_frame_at` with the
> part's timestamp map; Tenet 11's labeling-sampler exception); a frame the decoder cannot
> land on is skipped and redrawn from the same part, seeded by the dataset id.

Sampling is proportional to recording frame count — longer recordings contribute more frames. The total `n_frames` is distributed across source recordings weighted by their frame counts. Within each video, `LabelingJob.sampling` picks the draw: `"stratified"` (the default) spreads the frames evenly across time, `"uniform"` draws them at random without replacement.

The val split is stratified by recording: the val fraction is drawn independently from each recording so no recording is entirely absent from validation. It is drawn **after** labelling, over the examples actually written (`_dataset_common.StagedExamples`), so the realised share is the configured one — frames that yield no example never hold a validation slot. Every dataset source (run export, labeling, keypoint dataset) writes through it.

### Multi-class labeling

For each sampled frame, SAM3 is run once per class using that class's text prompt. Each class's detected contour is written as a separate label line (with its class ID) in the same `.txt` file. A frame is saved if at least one class produced a valid detection. A frame where the teacher found **none** of the classes is a background example — an empty label file, which Ultralytics trains on as one — kept up to `background_share` of the dataset (`LabelingConfig`, default 0.1, 0–0.5; at most `floor(share × labelled / (1 − share))`, chosen seeded by the dataset id). A frame where it found something but every mask fell below `min_contour_area` is skipped, not background: an animal may be there. A class absent from a frame has no row in the label file — YOLO interprets this as that class not being present in the image, which is the correct treatment for instance segmentation datasets.

---

### Trained model directory structure

The trainer is directed to write its run into `models/{model_id}/run/`. For Ultralytics this is accomplished via `model.train(project=<models/model_001>, name="run")`. All training artifacts stay inside the experiment directory and are portable.

```
models/
└── model_001/
    ├── model.yaml
    ├── job.json               ← the executor envelope (Section 6); models/<id>/ is the run dir
    ├── preview.png            ← one overlay frame of the model's own prediction
    └── run/                   ← trainer output directory
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        └── args.yaml
```

`model.yaml` — `Experiment.train_model` writes the first block at submit (`status: running`); the training runner (`pipeline/training.py`) fills in the rest when the run ends. A failed run records `status: failed` and an `error` message. A model trained on a pose dataset also carries `task: pose` and the `skeleton` it was trained on (§12 *Distill keypoints*), and its `model_family` is `yolo_pose`:

```yaml
schema_version: 1
id: "model_001"               # also the training job's id
name: "mouse_seg_v1"
trainer_id: "ultralytics_yolo"
dataset_id: "dataset_001"
base_model: "yolo11s-seg.pt"

model_family: "yolo"          # trainer.model_family — resolves the trained-apply loader

classes:
  - id: 0
    name: "mouse"
  - id: 1
    name: "food_pellet"

weights:
  best: "run/weights/best.pt"
  last: "run/weights/last.pt"

preview: "preview.png"      # one overlay frame of the model's own prediction, captured when
                            # training completes (relpath, or null); travels on promote

metrics:
  final_epoch: 87
  box_map50: 0.923
  mask_map50: 0.891

hyperparams:                # what was asked (UltralyticsYOLOConfig, defaults filled) —
  epochs: 100               # written at submit, so a live page has the epoch count
  batch_size: 32
  imgsz: 640
  patience: 100

trained_with:               # trainer.resolved_settings(): what the backend actually ran
  backend: "ultralytics"
  version: "8.4.70"
  args: {task: segment, epochs: 100, batch: 32, imgsz: 640, lr0: 0.01, seed: 0, ...}
                            # run/args.yaml minus machine paths (data, project, save_dir)

stopped_at_epoch: 37        # present only when Stop and keep ended the run early: the model is
                            # complete, and this says it did not run its course (absent otherwise)

status: "complete"          # "pending" | "running" | "complete" | "failed" | "partial"
```

`weights.best` and `weights.last` are set by calling `trainer.best_weights_path()` and `trainer.checkpoint_path()` — the paths are trainer-specific and recorded here, not assumed.

---

### Resume

On resume (`Experiment.resume_training(model_id)` re-submits the same job): the training runner finds a checkpoint via `trainer.checkpoint_path(model_dir)` and calls `trainer.resume(job, dataset_dir, model_dir, device, ...)` instead of `train`. For `UltralyticsYOLOTrainer` this is `YOLO(last_pt_path).train(resume=True)`. The checkpoint location and resume mechanism are fully encapsulated in the trainer — the executor calls the same interface regardless of backend.

**Only a killed or crashed run is resumable** (`RESUMABLE_TRAINING_STATUSES` = `cancelled` / `failed`). A run whose `train()` *returned* — all epochs, Stop and keep, or a cancel caught cleanly at an epoch boundary (`partial`) — has been finalized by Ultralytics, which strips `last.pt` of its `epoch` and `optimizer`. Given such a checkpoint, Ultralytics 8.4.120's `train(resume=True)` does not fail: it warns and starts **a new training from its defaults** (`coco8-seg.yaml`, batch 16, its own runs dir). So `UltralyticsYOLOTrainer.resume` checks the loaded checkpoint itself and raises, and `resume_training` refuses `partial` before submitting (#486).

---

### No `config.yaml` additions

Datasets and models are experiment-level artifacts with no index in `config.yaml`, which holds the experiment's identity only (Section 1). Each is discovered by **scanning** `datasets/` and `models/` for its `dataset.yaml` / `model.yaml` (`storage/dirscan.py`), the same pattern as recordings and runs. Training defaults are not stored per experiment either: each parameter's default lives in its config model (`configs/training.py::UltralyticsYOLOConfig`, `configs/labeling.py::LabelingConfig`), and each submitted job carries its full resolved values.

---

### Integration with detection (Section 5)

A trained model is referenced in detection runs by `model_id` with `weights_source: "trained"`. It is a normal `DetectionModel` subclass (`TrainedYOLO`) — same `load`/`predict` interface as SAM3, but it loads **local trained weights** (`best.pt`) and its class list from `model.yaml` instead of downloading.

```yaml
detection_runs:
  - id: "det_01"
    model_id: "model_001"
    weights_source: "trained"
    ...
```

**Model-id resolution** (`models/resolve.py::resolve_detection_model`). `weights_source` is the discriminator, so there is no ambiguity between a registered backend and a experiment model:
- `weights_source: "pretrained"` (default) → look up `model_id` in the `MODELS` registry (`"sam3"`, …), as today.
- `weights_source: "trained"` → read `experiment/models/<model_id>/model.yaml` (must be `status: complete`), take `model_family` → the `DetectionModel` registered under that family in the **`TRAINED_LOADERS`** registry (`register_trained_loader("yolo")` on `TrainedYOLO`), and instantiate it with the resolved `best.pt` path + class list. `TRAINED_LOADERS` is a separate axis from `MODELS` (keyed by family, not model id), so a trained loader is *not* a `MODELS` entry; the *backend* is still registry-based (Tenet 1) and `model_id` is a weights/instance reference. The training runner records `model_family` (from `trainer.model_family`) into `model.yaml` on completion so resolution can find the loader.
- `weights_source: "imported"` → read the **machine-global** imported-model catalog (`model_cache_dir()/imported/<id>/model.yaml`, `models/imported.py`) — same `model.yaml` + `TRAINED_LOADERS` shape as a trained model, only the directory is machine-wide (shared across every experiment — the data-defined twin of `MODELS`; `_load_from_model_yaml` is shared with the trained branch). `import_model` reads class names straight from the checkpoint via a per-family `introspect` hook on the loader (`TrainedYOLO.introspect`), so importing is just dropping in a `.pt`; ids that collide with a registry id or the reserved `model_NNN` shape are rejected to keep the namespace unambiguous.

**Auto source resolution** (`resolve_detection_model_auto`). New code resolves a `model_id` *without* being told its source: registry → experiment-trained → machine-imported, returning `(model, weights_source)`. The id namespaces are disjoint by construction, so the first hit is unambiguous. Detection's `validate` calls this, **records** the discovered `weights_source` on the run, and the worker re-resolves with the explicit source. The user surface is therefore a single polymorphic `--model <id>` (CLI) / one merged "Local models" picker (GUI) — no `weights_source` flag.

**Closed-set, no-prompt detection (extends §5/§9).** A trained model is **closed-set** — its classes are fixed by training; you don't prompt it. So a `DetectionModel` may declare `supported_cue_types = []` ("takes no cues"). Detection's `validate` treats such a model specially: it requires an **empty prompt** (`Prompt(targets=[])`) and rejects any prompt cues; `predict` ignores the prompt entirely, runs the model, and returns detections labeled with the model's own trained class names (from `model.yaml`). Symmetrically, an open-vocabulary model (non-empty `supported_cue_types`) now requires at least one prompt target. Slice 3 detects **all** of the trained model's classes; a per-class checkbox selection in the detection config (subset filtering, surfaced via an API accessor — Tenet 2) is a deferred GUI refinement (Slice 4).

**Ultralytics output paths (Tenet 6).** The trainer writes its run into `models/<id>/run/` — and Ultralytics' *own* files there (`args.yaml`, etc.) may embed absolute machine paths. Decision: `run/` is **trainer-owned scratch**, not relied on for portability; DETAIL's authoritative files — `model.yaml`, `dataset.yaml`, and the trainer `data.yaml` (`path: .`) — stay relative and are what every other stage reads. The one file read back is `run/args.yaml`, Ultralytics' resolved settings, copied into `model.yaml → trained_with` with its version and **without** any value that is a path on this machine — so the record says what trained the model and still holds no absolute path. `ultralytics` is pinned to one minor series (`>=8.4,<8.5`) because its defaults decide every setting this tool does not set.

---

### Validation checks

**Pre-labeling:**

| Check | Action on failure |
|-------|-------------------|
| Source recordings have video files | Error |
| The labeling model is ready (weights installed) | Error — `ValueError` naming what is missing; a run never downloads (Section 5 `install()`) |
| `n_frames` > total available frames across source recordings | Error — requested frame count exceeds what exists across source recordings |
| Two classes with identical prompts | Error — duplicate prompts produce duplicate labels |
| Output dataset directory writable | Error |

**Post-labeling — not built.** No check runs after labeling today. `dataset.yaml` records `per_class` (labelled instances per class), `n_train` and `n_val`, so the facts the checks below would read are on disk. If built, the two count checks report their numbers on the dataset page rather than emit a warning (a warning nobody reads is not a check):

| Check | Action on failure |
|-------|-------------------|
| Fewer than 100 labeled examples for any class | Report per-class counts |
| Class imbalance > 10:1 | Report per-class counts |
| Val set empty when `val_split > 0` | Error |

**Pre-training:**

| Check | Action on failure |
|-------|-------------------|
| Dataset `status: "complete"` | Error |
| `trainer_id` present in TRAINERS registry | Error — list available trainers |
| `trainer.is_ready()` | Error — report what is missing |
| `compute.backend` in `trainer.supported_backends` | Error — list what this trainer supports |
| Requested GPU(s) accessible | Error — list available devices |
| `trainer.validate_hyperparams()` passes | Error with field-level detail |
| Model directory has no existing `complete` run | Error — prevent silent overwrite |

**Per-epoch:**

| Check | Action on failure |
|-------|-------------------|
| Stop signal received | Trainer sets stop flag at epoch boundary; Stop and keep → `complete` + `stopped_at_epoch`, a cleanly caught Cancel → `partial` |

**Post-training:**

| Check | Action on failure |
|-------|-------------------|
| `trainer.best_weights_path()` exists | Error — mark `failed` |
| Metrics readable via `trainer.final_metrics()` | Error if unreadable — mark run `partial`; metrics must be recorded for the run to be `complete` |

---

### Distill orchestrator (label → train, one durable run)

"Distill a model" is the SAM3 → YOLO workflow presented as **one** action: label a frame
sample with a prompt-capable model, then train a closed-set YOLO detector on those labels.
It is an **experiment-level, stage-keyed durable run** — the run-level twin of a processing
run (§15), sequencing **stages** (`label → train`) instead of recordings. It is **not** a
`ProcessingModule` (those are per-recording, one output per recording); a distill produces one
dataset + one model across all selected recordings. The orchestrator **composes the existing
`Experiment.label_dataset` and `Experiment.train_model` verbatim** — it adds no new inference
code — and each stage keeps its own run directory (`datasets/<id>/`, `models/<id>/`); the train
stage can resume from its checkpoint, the label stage cannot (see **Resume** below).

**Detached coordinator.** `Experiment.distill(...)` validates up-front (the union of the
labeling and training pre-flight checks), reserves the run record, and spawns a **detached
coordinator** (`python -m detail.executor.distill_coordinator <distill/<id>/>`, its own process
group — survives the launching app closing), distinct from the processing-run coordinator (§15).
The coordinator owns the two child worker processes in turn: it calls `label_dataset(...).wait()`,
then `train_model(dataset_id, ...).wait()`, folding each stage's native progress
(`LabelingProgressEvent` frame-fraction; `TrainingProgressEvent` per-epoch) into one
`DistillProgress`. Reconcile-on-open (`reconcile_distill_runs`, evidence-based on the coordinator
pid + each stage's on-disk `dataset.yaml`/`model.yaml` status) is the backstop for a killed
coordinator, exactly as for processing runs.

**Storage** mirrors `runs/` (§15): `distill/<id>/` holds `distill.yaml` (atomic, `schema_version`
first), `progress.json` (the folded `DistillProgress`, poll-read across processes), `coordinator.pid`,
and a `cancel` sentinel. A parallel store (`storage/distill_runs.py`), not `runs.py` — a distill run
is stage-keyed, not `module_id` + `per_recording`, and must not appear in the processing-runs index
(same precedent as `analysis_runs.py`, `tool_runs.py`).

```yaml
# distill/<id>/distill.yaml
schema_version: 1
id: "distill_001"
name: "mouse_fast"
status: "running"                # running | complete | partial | failed | cancelled
created_at: "2026-08-07T18:22:04Z"
config:                          # serialized DistillConfig + the experiment-aware method inputs
  name: "mouse_fast"
  base_model: "yolo11s-seg.pt"
  trainer_id: "ultralytics_yolo"
  classes: [{class_id: 0, name: "mouse", prompt: "a mouse"}]
  source_recording_ids: ["rec_001", "rec_002"]
  labeling: {model_id: "sam3", model_variant: null, n_frames: 500, sampling: "stratified",
             logit_threshold: 0.5, min_contour_area: 100, val_split: 0.15,
             background_share: 0.1}
  training: {epochs: 100, batch_size: 32, imgsz: 640, patience: 100}
  label_compute: {backend: "cuda", device_indices: null, cloud_provider: null, cloud_instance: null}
  train_compute: {backend: "cuda", device_indices: null, cloud_provider: null, cloud_instance: null}
stages:
  label: {status: "complete", dataset_id: "dataset_003"}
  train: {status: "running",  model_id: "model_002"}
resume: false
# error: "..."                   # added when the run fails
```

**`DistillConfig`** (`configs/distill.py`, pydantic, `extra="forbid"`) **composes** the existing
config models — it does **not** re-home their fields (§6.5): `name`, `base_model`
(`"yolo11s-seg.pt"`), `trainer_id` (`"ultralytics_yolo"`), `labeling: LabelingConfig`,
`training: UltralyticsYOLOConfig`. The experiment-aware inputs — `classes`, `source_recording_ids`,
and the **two** compute targets — are `Experiment.distill(...)` arguments validated in the method
(SAM3 labeling is CUDA-only; the trainer accepts cpu/cuda/mps, so the two computes default
independently), exactly as `label_dataset` keeps `classes`/`compute` out of `LabelingConfig`.

**Progress fold.** `DistillProgress` (`core/dataclasses.py`): `phase` (`"labeling"` | `"training"`,
plus `"registering"` as a keypoint distill's third stage), `stage_index`, `stage_count` (2 here),
`inner_fraction`, `overall_fraction` (stage-weighted:
`{"labeling": 0.45, "training": 0.55}`), `elapsed_s`, `eta_s`, and an advancing `message`. A
phase-boundary emit is forced at each stage transition so the message never sits frozen (§6
progress rule).

**The stage weights are for the bar only — never for the ETA.** They are nominal (labeling's
share of the bar is far larger than its share of wall clock), so `remaining_seconds(overall, elapsed)`
— which the fold once used — inferred a throughput several times too high and reported a
fraction of the real time left, worst at the labeling→training boundary. `eta_s` comes from the **running
stage** instead: the stage runner's own `eta_s` where it has one (the trainer already computes an
exact per-epoch estimate, which the fold previously discarded), else `remaining_seconds(inner_fraction,
now − stage_start)` over a clock that resets at each phase boundary. During training — the last stage
— stage-remaining *is* run-remaining, so the number is exact; during labeling it is labeling's own
remaining time, because a whole-run estimate is unknowable before `total_epochs` exists. Renderers
must not pair a distill event's weighted fraction with its wall-clock `elapsed_s` as a fallback: that
reconstructs the same bug downstream (`gui/workers.py::fit_progress_fraction` trusts the event's
`eta_s` alone). This is §6's *"the ETA's denominator is the work the emitting coordinator did — not
always the bar's"* (see the retry case above), which distill re-broke; a plausible ETA is
indistinguishable from a correct one, so it is pinned by
`tests/api/test_distill.py::test_training_eta_tracks_the_real_remaining_time`.

**Resume** (`resume_distill(run_id)`) reconciles first, then skips a **complete** label stage
(reusing its `dataset_id`) and resumes a **cancelled or failed** model via the existing `resume_training` (a `partial` one is finished, so the train stage starts a new model).
Labeling has no partial-resume seam (`label_dataset` mints a fresh `dataset_NNN`), so a non-complete
label stage re-labels from scratch. `delete_distill(run_id)` prunes only the `distill/<id>/` index —
never the produced dataset, trained model, or a promoted global model.

**Surfaces (parity):** `Experiment.distill(...)` (API) · `detail distill run|list|status|attach|resume|cancel|delete` (CLI) · one bespoke `DistillDialog` (GUI, with a `Test prompts…` preview) — distill is §12, so a bespoke surface does not conflict with "one launch surface per module" (that governs §6.5/§16 modules).

### Distill keypoints (label with the patch method or a keypoint detect run → train YOLO-pose → register)

> Specified 2026-10-07 (PR 8 of the §26 build); built 2026-10-08. The mechanics are those of
> #682 (distill a keypoints run into a YOLO-pose student, open against the `keypoints` branch),
> ported onto §5's `Model` base and §26's two stages, with one change: the student sees the
> **whole frame**, not the teacher's crop.

The keypoint twin of the distill above: a durable run of `kind: keypoints` whose stages are
**`label → train → register`**. It reuses the orchestrator, the detached coordinator, cancel /
resume / attach and the run page; only the stage bodies differ.

- **Teacher, one of two** (decided 2026-10-08, the user's review; **built 2026-10-09**). **The patch method
  itself**: `dinov3_match` with an **example set** (the picker of §26 *Example sets follow the
  model that needs them*), run on the sampled frames to label them, the way the detector distill
  lets SAM 3 label; its labels carry no smoothing over time. **Or a finished `keypoint_detect`
  run** (revised 2026-10-09 from the user's review; it replaces the `keypoint_track` run, so the two
  teachers are symmetric): its stored instances, read back through `KeypointDetectRun.rows` and
  given to the subjects of a tracking run on its detection run by §26 *Assignment*'s function
  (the teacher record carries that `tracking_run_id`; revised 2026-10-21), are the labels,
  unsmoothed, and nothing runs on the accelerator while labelling. **Both teachers are
  raw**; a track run's smoothing is not taught ("I don't like the asymmetry … one is smoothed and the
  other isn't"). The dialog asks which (the patch method first: it needs no keypoint runs); the
  record carries `teacher: {kind: "model" | "keypoint_detect", …}`, and an older record's
  `track_runs` reads as `kind: keypoint_track`, never re-run. The model teacher differs from the
  detect-run teacher only in where the label stage reads its points. **The score cut-off is the
  user's choice, made in Test cut-off…** before launch (below), and the label stage keeps the
  teacher's raw output (`teacher.npz` beside `dataset.yaml`: per used frame and track, the points,
  their scores, the candidates and their scores, aligned with `frames`) so a cut-off can later be
  re-applied without the GPU; **no re-cut surface yet**. **As built:** the model teacher
  is resolved and **copied at launch** by `keypoint_detect.resolve_method` (the function
  keypoint detect's `validate` uses: set runnable, weights installed, backend supported, the
  example-set snapshot, the settings) and built by `keypoint_detect.build_method` from that copy,
  never the live set; the job carries it as `KeypointDatasetJob.teacher` with a `compute` target.
  Its frames are every tracked frame on which every detection is assigned to a track (the same
  rule), drawn spread over time (the user chose spread over varied postures, 2026-10-09); the
  labels are full frames, one line per tracked animal (the user chose full frames over crops).
  The method is given each sampled frame's **detections** — boxes and masks — in batches, as the
  detect stage gives them (§26 revision 2026-10-21), and its instances are given to the tracking
  run's subjects by §26 *Assignment*'s function; an assigned instance's placement on that frame
  alone is the label. A subject no instance was assigned to is still taught with its box and no
  visible points. A frame
  that cannot be decoded exactly is counted (`excluded.undecodable`) and the dataset says
  `n_requested` and `n_from_teacher`. `dinov3_match` sets no default cut-off, so the cut-off is
  required. No teacher window (a proposed `teacher_window_s`, declined). Either teacher's placements
  join the example set's own clicks, the hand labels the teacher was built from. Both are the
  user's input; neither is an audit label.
- **Test cut-off…** (built 2026-10-09; the user chose a pre-launch sample "for continuity with how
  detection is run", detection's *Test prompts…*). `pipeline/keypoint_preview.py::run_teacher_test`
  draws about a dozen frames **through the label stage's own sampler** (`_gather` → `_share` +
  `_slices` → `_decode`, seeded), so the sample is what the distill would label, has the teacher
  place them (the patch method runs; a detect run's rows are read), and writes nothing. For a method
  with `score_held_out()` (`PatchMatch`) it adds the **held-out check**: each example frame placed by
  a bank rebuilt from the others (one `load()`, no backbone call per frame) and measured against
  the user's click. `cutoff_report(rows, cutoff)` splits those by the cut-off and gives, overall
  and per point, n, the median error **in pixels and as % of body size** (`sqrt(w·h)` of the
  tracking box) and the share within 5% of body size; a median from fewer than 8 placements is
  withheld. Surfaces: `Experiment.test_keypoints` (the distill's teacher keywords plus `n_frames`,
  `seed`, `every_n_frames`, `exclude_audit_s`), `detail keypoints test`, and
  `gui/keypoint_test.py::KeypointTestDialog` (cards with points filled when taught and hollow
  below the cut-off, a slider that redraws without re-running, *Another sample*, *Use this
  cut-off*), which the distill form's *Test cut-off…* opens. The keypoint detect form's *Test
  keypoints* (§26 PR 4b) opens its subclass `KeypointDetectTestDialog`, laid out as detection's
  *Test prompts* (Recordings and Sample frames rows, *Run keypoint test*, the slider above the
  frames, the held-out check on one line with *Per point*) and with no *Use this cut-off*: a
  detect run stores raw points. It passes the form's method settings (`settings=`: `decode`,
  `side_rule`, `crop_px`, also `keypoints test --decode/--side-rule/--crop-px`), so the test
  places points the way the run will.
- **Label** (`pipeline/keypoint_dataset.py`, job `keypoint_dataset`, config
  `KeypointDatasetConfig`). Frames are sampled from the teacher's rows, shared across recordings
  by row count, one draw per equal time slice, seeded by the dataset id; each is decoded
  **exactly** (`read_frame_at`, the Tenet 11 labeling-sampler exception), because the label is
  that frame's point. A frame is used only when **every** detection in it is assigned to a track,
  so no animal in a training image goes unlabelled. One label line per tracked animal: the
  tracking box, then each skeleton point as `x y v` with `v = 2` only where the point was placed
  with a score at or above the method's cut-off (`score_spec.default_cutoff`, resolved and
  recorded) and `v = 0` otherwise — the box is still taught. `flip_idx` in `data.yaml` comes from
  the skeleton's symmetry pairs, so Ultralytics' flip augmentation does not teach left/right
  swaps. **Audit frames, and every frame within `exclude_audit_s` of one, are never used**
  (labels score a method; they never build one). The dataset records every frame it used
  (`dataset.yaml → frames`), so a later audit of the student can exclude them.
- **Train.** The §12 trainer unchanged; the task comes from the `-pose` base checkpoint
  (`yolo11s-pose.pt`), and `train_model` refuses a base model that cannot learn the dataset's
  labels. `data.yaml` gains `kpt_shape: [K, 3]` and `flip_idx`; dataset, job and model records
  gain `task: pose`. Metrics read `mAP50(P)`, and the best epoch is ranked by it.
- **Register.** Promotes the student as `<name> <distill_NNN>` into the catalog with
  `role: keypoints`, `model_family: yolo_pose`, the skeleton (copied, names and pairs), the
  weights' sha256 and the teacher runs. A resume after a failed register does not retrain.
- **The student** is `TrainedYoloPose`, a closed-set `KeypointModel` (§26): `n_candidates = 1`,
  a confidence of 0 is NaN, never a guess; it loads local weights only, through the
  `import_ultralytics` seam. `ModelRef.role` lets the detection picker skip pose models.

**Surfaces (parity):** `Experiment.distill_keypoints(...)`, `Experiment.keypoint_dataset(...)`
(API) · `detail keypoints distill`, `detail dataset keypoints`, the register stage in
`distill status` (CLI) · **Distill keypoints…** on the experiment's Models page, fields generated
from `KeypointDatasetConfig` and the trainer's config; the run page shows the three stage rows
(GUI). The machine-wide Model manager has no distill button: a keypoint distill needs an
experiment's keypoint runs.

### Promote a trained model to the global catalog (on-demand, with provenance)

A trained model is immediately usable **in its own experiment** (detection resolves
`weights_source: "trained"` against `experiment/models/<id>/`). To reuse it **across experiments**,
`Experiment.promote_model(model_id, name=None)` copies it into the machine-global imported catalog
(`model_cache_dir()/imported/<id>/`, `models/imported.py`) — the same directory + `model.yaml` shape
as an imported model, so detection reaches it unchanged with `weights_source: "imported"` (no new
resolution branch). Promotion is **on-demand** (a discrete action, not a distill stage), so a user
can iterate on several distill attempts and promote only the keeper.

The promoted record adds a `source` discriminator, user-facing **catalog metadata**, and a
`provenance` block so a user can see (and describe) where a global model came from — the model manager
is a first-class home so a model isn't "lost to time in a folder." `source` becomes an enum alongside
`"imported"`:

```yaml
# model_cache_dir()/imported/<id>/model.yaml (promoted or imported)
source: "distilled"              # vs "imported" (dropped-in .pt)
created_at: "2026-08-07T19:40:11Z"   # stamped on import/promote (imported models get one too)
made_by: "labmate"               # user-typed; "" when unset. No user-management system — just a field
description: "operant self-admin, box 3"   # "what it was for"; "" when unset
notes: ""                        # optional free text
preview: "preview.png"           # model's own overlay frame (relpath, or absent)
provenance:
  source_experiments:            # a LIST — one model can name several sources later (distill
    - {name: "rotation_pilot", id: "rotation_pilot"}   # writes one entry today
  source_dataset_id: "dataset_003"
  base_model: "yolo11s-seg.pt"
  trainer_id: "ultralytics_yolo"
  distilled_at: "2026-08-07T19:40:11Z"
  metrics: {box_map50: 0.92, mask_map50: 0.89}
```

`made_by` / `description` / `notes` are captured in the import dialog and editable on the model detail
page; `created_at` shows alongside a `last-updated` (equal until in-place iteration exists). Readers
accept the **legacy single-source** shape (`provenance.source_experiment_name`/`_id`) and treat it as a
one-element `source_experiments` list, so models promoted before this schema still render.

Provenance is **metadata only** — names, ids, and metrics, never an absolute path into the source
experiment (Tenet 6) — and the weights (and preview) are **copied**, so the global model stays
self-contained and survives deleting the source experiment. **Surfaces (parity):**
`Experiment.promote_model(...)` (API) · `detail train promote <model_id> [--name]` (CLI, in the `train`
group — it promotes a trained model) · a "Promote to global…" button on the trained-model detail page
(GUI). The catalog metadata + preview + export have their own parity trio (`api/models.py` ·
`detail model show|export|set-meta|import --made-by/--description/--notes/--preview` · the Model
manager detail page).

### Keypoint models in the catalog (2026-10-21)

**A keypoint (YOLO-pose) model moves through the catalog exactly as a detection model does**:
promote → export a zip → restore the zip on another machine, or import an Ultralytics pose `.pt`;
then it is offered in any experiment's keypoint detect form. Before this, the keypoint lookup read
only the current experiment's students, promote dropped the skeleton (so a promoted student could
not be rebuilt), an imported pose `.pt` landed as a *detection* model that failed when built, and
no GUI control restored a zip at all, though the docs said one did.

```yaml
# model_cache_dir()/imported/<id>/model.yaml — a keypoint model
role: keypoints                  # absent / "detection" for a detection model; derived from task on read
task: pose
model_family: yolo_pose
skeleton: {names: [...], edges: [...], pairs: [...]}   # the copy the model uses — always, whatever the library holds
skeleton_digest: "4b7e…"
skeleton_ref: {id: "sk_03", name: "rat overhead", digest: "4b7e…"}   # the library entry it was matched or added to
weights_sha256: "…"
provenance:
  teacher: {kind: "keypoint_detect", distill_run_id: "distill_002", …}   # a promoted student's; absent for an imported .pt
```

- **The skeleton travels inside `model.yaml`** and the model always uses that copy. On promote,
  restore and import, `ensure_in_library` adds it to the machine's skeleton library unless an entry
  (or a preset) with the same digest exists; a taken name becomes "name (2)", "name (3)"; an existing
  entry is never edited. The library entry is only so the skeleton can be reused, for example to
  mark example frames.
- **An external pose `.pt`** carries the number of points `K` and Ultralytics' `flip_idx` (which
  point each becomes under a left-right flip), not names or edges. The user supplies a skeleton — a
  preset, a library id or a YAML — and `check_against_model` refuses one with a different `K`, or
  whose left/right pairs give a different `flip_idx`. A checkpoint with **no** `flip_idx` is refused
  with a skeleton that has pairs (they cannot be checked); a skeleton without pairs passes. The check
  **cannot** catch a pair named the wrong way round, which gives the same `flip_idx`.
- **The family comes from the checkpoint's task**, never typed: a pose checkpoint imports as
  `yolo_pose`; `--family` is checked against it.
- **Lookup order** for a keypoint method is built-in → this experiment's students → the catalog, as
  for detection. A keypoint detect run resolves and **writes back `weights_source`** (`"trained"` |
  `"imported"`), as detection does (Tenet 9): an experiment model and a catalog model can share an id.
- **Pose models are kept out of detection**: catalog rows take `role` from `task`; the detection
  picker, Quick segment's model list and detection resolution skip or refuse `task: pose`.
- **Export still starts from the catalog** (promote first), as for detection. **Surfaces:** the Model
  manager's Kind column (detector / keypoints), a **Restore from zip** button, the import dialog's
  skeleton step, the global model page's skeleton; `detail model import --skeleton`, `model list|show`
  printing role and skeleton, `model restore` naming the library entry; the API's `import_model(…,
  skeleton=)`.

---

## 13. Tracking

> Implemented as a `ProcessingModule` (§6.5), `upstream = "detection"`. The schemas below are unchanged.

Tracking assigns a consistent `object_id` to each detection across frames — linking the same animal through time. A **track** is the sequence of detections sharing the same `object_id`. Downstream stages (embedding, analysis) select which track(s) to use, which handles single-animal and multi-animal cases without special-casing: for one animal the user selects track 0; for two animals, tracks 0 and 1.

Tracking subsumes what would otherwise be a separate filtering step. The confidence threshold and arena restriction are inputs to the tracker — detections that fail them are never linked into a track. What the downstream stage receives is not a filtered list but a set of identified subjects.

Multiple tracking runs per detection run are supported. Downstream stages declare a dependency on a specific tracking run ID.

`configs/tracking.DEFAULT_TRACKER` is the single source for the model, the CLI option and the GUI's opening selection, and `TrackingModule.validate` writes the resolved value back into the config so the run record names the tracker it ran (Tenet 9). It is `anchor_tracker`: the recordings this tool has been pointed at hold one animal, and a single-subject tracker gives that animal one track id for the whole recording, which is what every later stage reads. The field was required for a long time, which sounds like neutrality and is not — it left each surface to supply its own, and they disagreed without anything failing.

### `Tracker` protocol

```python
class Tracker(ABC):
    tracker_id: str
    supported_backends: list[str]   # most trackers are ["cpu"] only

    def validate_hyperparams(self, hyperparams: dict) -> dict:
        """Fill defaults, raise ValueError on unknown or wrong-type keys."""

    def track(
        self,
        job: TrackingJob,
        detection_run_dir: Path,
        output_dir: Path,
        arena_polygons: dict[str, PolygonKeyframes] | None,
        part_frame_offsets: list[int] | None = None,
        progress_cb: Callable[[TrackingProgressEvent], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None: ...
```

`arena_polygons` maps arena ID to its keyframe table (Section 10). `None` or empty dict means no arena restriction. `part_frame_offsets[i]` is the recording-global frame index of part `i`'s frame 0 — used for the per-frame arena lookup and for recording-global frame numbering in the output (a tracker can't derive it from detections alone: zero-detection frames leave no rows, §9); `None` means part-local indexing (single-part or no-arena runs). `tracking.run` always supplies it. Parts are processed in index order; the tracker maintains track continuity across part boundaries and **resumes** at them (see below).

### Tracker registry

Same pattern as `Trainer` and `DetectionModel` — self-register on import. Excluding a tracker backend means not importing its module.

```python
TRACKERS: dict[str, type[Tracker]] = {}

def register_tracker(tracker_id: str):
    def decorator(cls):
        TRACKERS[tracker_id] = cls
        return cls
    return decorator
```

### Job dataclasses

```python
@dataclass
class TrackingJob:
    job_id: str
    recording_id: str
    track_id: str
    detection_run_id: str
    tracker_id: str         # key into TRACKERS registry
    hyperparams: dict       # validated by tracker.validate_hyperparams()
    arena_ids: list[str]    # empty = no arena restriction
    created_at: str
    every_n_frames: int = 1 # resolved from the detection run (§4)

@dataclass
class TrackingProgressEvent:
    parts_done: int
    parts_total: int
    frames_done: int
    frames_total: int
    n_active_tracks: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None
```

### Cross-part continuity and resume

**Cross-part continuity + resume (all trackers)**: after each completed part the tracker atomically commits `part_handoff.json` — `{schema_version, parts_completed, state}`, where `state` is that tracker's own live cross-part snapshot (`anchor_tracker`: the subject's summary accumulators; `iou_single_tracker`: also the subject bbox and missed-frame count). The handoff write is the commit point, so it always follows the part's assignment write. A re-invoked `track()` reads the handoff, skips the completed parts, restores `state`, and continues — a run interrupted at a part boundary resumes to output identical to an uninterrupted one (Tenet 12). The per-part assignment writer opens `mode="a"`, so a reprocessed part's stale store (and `tracks.zarr`, re-emitted from the accumulators) is cleared before the fresh write to avoid appended duplicates. `stop_event` yields status `partial`; a fresh run (no handoff) starts at part 0.

### `AnchorTracker` — confidence-anchored single-subject tracker

```python
@register_tracker("anchor_tracker")
class AnchorTracker(Tracker):
    tracker_id = "anchor_tracker"
    supported_backends = ["cpu"]

    HYPERPARAMS_SCHEMA = {
        "num_subjects":       (int,   1),   # restricted to 1 for now
        "conf_threshold":     (float, 0.1), # primary/anchor minimum
        "secondary_conf":     (float, 0.05),# secondary minimum (≤ conf_threshold)
        "iou_threshold":      (float, 0.1), # secondary must overlap an anchor by ≥ this
        "max_secondary":      (int,   2),   # cap per anchor
        "require_contiguous": (bool,  False),# veto masks with >1 connected component
    }
```

`require_contiguous` is **off by default**. On, it is a *veto*, not a preference: `select_subject`
walks candidates in score order and skips every one whose mask is more than one connected region,
so a frame where the detector found the animal clearly ends up with no subject at all.


A mask that splits as the animal moves costs single frames; an animal holding a pose whose mask
is stably split costs long runs. `iou_single_tracker` has always
defaulted it off for the related reason below; `AnchorTracker` has no continuity to fall through
to, so it simply loses the frame. Taking the best candidate when none is contiguous — the gate as
a *preference* — is the better fix and is not yet built.

For the known-subject-count case (one animal per recording) at a low detection threshold, a multi-subject IoU tracker spawns a track per spurious detection. `AnchorTracker` instead, per frame: (1) keeps the single top-confidence detection above `conf_threshold` whose bbox center is in the arena, as the subject anchor (`object_id 0`) — and, when `require_contiguous` is on, whose mask is a single connected component (a `cv2.connectedComponents` contiguity gate); (2) attaches nearby lower-confidence detections (above `secondary_conf`, bbox IoU ≥ `iou_threshold` with the anchor, up to `max_secondary`) with the **same** `object_id` — recovering fragments when the detector splits the animal across detections; (3) leaves everything else `object_id = -1`. Identity is trivial for one subject (track 0 in every frame); `num_subjects > 1` requires a cross-frame association layer and is rejected by `validate_hyperparams` until built. The mask resolution for the contiguity decode is read once from the detection run's `job.json` envelope (`mask_resolution`).

**A track may hold more than one detection per frame.** The assignment zarr already permits this (one `object_id` per detection row; the id simply repeats within a frame). The embedding stage (§14) groups a track's detection rows by frame and **OR-merges their masks into one subject silhouette per frame** — so the secondaries become full-silhouette recovery, and there is exactly one embedded image per frame. This is a no-op for a tracker run that assigns ≤1 detection per track-frame (e.g. `iou_single_tracker` at its default `max_merge = 0`, where each group has size 1). The render-preview (§14) merges the same way so its input-image preview matches what is embedded.

### `IoUSingleTracker` — IoU-continuity single-subject tracker

```python
@register_tracker("iou_single_tracker")
class IoUSingleTracker(Tracker):
    tracker_id = "iou_single_tracker"
    supported_backends = ["cpu"]

    HYPERPARAMS_SCHEMA = {
        "conf_threshold":     (float, 0.05), # min conf to be a subject candidate
        "secondary_conf":     (float, 0.05), # min conf for an attachable fragment (≤ conf_threshold)
        "merge_iou":          (float, 0.1),  # bbox-overlap pre-filter for fragment merging
        "merge_mask_iou":     (float, 0.5),  # mask-overlap floor: rejects glass reflections
        "max_merge":          (int,   0),    # fragments merged per frame (0 = cleanest silhouette)
        "iou_weight":         (float, 1.0),  # weight of the IoU-continuity bonus in the score
        "max_missed":         (int,   30),   # frames the subject may be absent before re-acquire
        "require_contiguous": (bool,  False),# off: IoU continuity already locks onto the subject
    }
```

Single-subject (`object_id 0`) like `AnchorTracker`, but selection has **memory**. `AnchorTracker` picks the per-frame argmax-confidence detection with no state, so a one-frame confidence spike on a distractor steals the subject. `IoUSingleTracker` instead walks the full detection span (a zero-detection frame still ages the subject) and per frame:

1. while the subject is **locked** (seen within `max_missed` frames), picks the arena/conf-gated candidate maximizing `conf + iou_weight · IoU(candidate, last_subject_bbox)` — a distant high-confidence distractor earns no bonus and loses to the slightly-lower-confidence subject sitting where it was last frame; when nothing overlaps, the bonus is ~0 and selection falls back to confidence;
2. else (cold, or absent for more than `max_missed` frames) re-acquires by plain confidence;
3. attaches overlapping lower-confidence fragments (`object_id 0`) through a two-stage gate — bbox IoU ≥ `merge_iou` (cheap pre-filter), then **mask** IoU ≥ `merge_mask_iou`, which rejects a glass reflection whose bbox overlaps the subject but whose pixels are a disjoint region. `max_merge = 0` (the validated default) disables merging for the cleanest single silhouette.

`require_contiguous` defaults **False**, as it does on `AnchorTracker`: IoU continuity already identifies the subject, and the contiguity gate otherwise rejects the subject's *split* mask and lets selection fall through to a contiguous distractor (subject teleports). The subject bbox + missed count carry across parts and are serialized into `part_handoff.json` for resume. The validated defaults are tuned on single-subject rat recordings (a sensible default, not universal). **Known limitation (detector-side, not the tracker):** a fully curled / balled-up subject gets ~no detector confidence, so no candidate clears `conf_threshold` and that frame has no subject; `max_missed` bridges the identity gap but cannot recover the missing mask.

### Tracking run directory structure

Tracking runs live at recording level as a sibling of `detections/` — consistent with the layout reserved in Section 9. The dependency on a specific detection run is recorded in `track.yaml`, not expressed through directory nesting.

```
recordings/recording_001/
├── detections/
│   └── det_01/
│       ├── job.json
│       ├── part_001.zarr        ← detection rows (bbox, mask_rle, confidence, t_ns, ...)
│       └── part_002.zarr
└── tracking/
    └── track_01/
        ├── track.yaml           ← params, status, subject_labels
        ├── part_handoff.json    ← active track state serialized after each part
        ├── tracks.zarr/         ← one row per track, summary metadata
        └── assignments/
            ├── part_001.zarr    ← object_id (N,) parallel to det_01/part_001.zarr
            └── part_002.zarr
```

The three data artifacts and their purpose:

| Artifact | Indexed by | Purpose |
|----------|-----------|---------|
| `track.yaml` | — | Params, status, subject labels |
| `tracks.zarr` | track ID — T rows | What tracks exist: time span, frame count, mean confidence. Used by preview UI and downstream config |
| `assignments/part_00N.zarr` | detection row — N rows | For detection row `i`, what track does it belong to. Fast O(1) lookup for embedding |

### Per-part assignment zarr

One row per detection row in the corresponding source part zarr. Parallel arrays — row `i` of `assignments/part_001.zarr` corresponds to row `i` of `det_01/part_001.zarr`.

```
assignments/part_001.zarr/
└── object_id     (N,) int32    chunks=(10000,)    # track ID; -1 = unassigned (failed filter or not linked)
```

The source detection zarr is never modified, and it has no `object_id` column (see *Relationship to `object_id`* below). Downstream stages always read `object_id` from `assignments/` and everything else (`bbox`, `mask_rle`, `confidence`, `t_ns`) from the source detection zarr.

### Track summary zarr

One row per track. Used by the preview UI and by downstream stage configuration to list available tracks.

```
tracks.zarr/
├── track_id       (T,) int32
├── class_name     (T,) vlen-str
├── first_frame    (T,) int32
├── last_frame     (T,) int32
├── n_detections   (T,) int32       # frames where a detection was linked
├── n_gaps         (T,) int32       # processed frames missed within the track span (§4: at a stride, a frame never looked at is not missed)
└── mean_conf      (T,) float32
```

### `track.yaml`

```yaml
schema_version: 1
id: "track_01"
created_at: "2026-06-03T10:00:00Z"
detection_run_id: "det_01"
tracker_id: "anchor_tracker"
hyperparams:
  num_subjects: 1
  conf_threshold: 0.1
  iou_threshold: 0.1
arena_ids: []

subject_labels: null    # null until identity is assigned
                        # when populated: {0: "M001", 1: "M002"}

status: "complete"      # "pending" | "running" | "complete" | "failed" | "partial"
n_tracks: 2
n_detections_assigned: 87210
n_detections_unassigned: 1222
```

### `recording.yaml` additions

Tracking runs are recorded as a top-level list, parallel to `detection_runs`. The `detection_run_id` field records the dependency.

```yaml
detection_runs:
  - id: "det_01"
    ...
    status: "complete"
    output: "detections/det_01/"

tracking_runs:
  - id: "track_01"
    detection_run_id: "det_01"
    tracker_id: "anchor_tracker"
    status: "complete"
    output: "tracking/track_01/"
```

### Arena restriction

Before matching, the tracker checks each detection's bounding box center against the active arena polygon at that frame (Section 10 interpolation). Detections outside the arena get `object_id = -1` and are never used to extend or start a track. If `arena_ids` is empty no spatial check is performed. Both registered trackers (`anchor_tracker`, `iou_single_tracker`) apply it, through the `arena_polygons` the `Tracker` protocol carries.

### Preview window

**Not built.** What exists is the preview of a *finished* tracking run (`gui/run_preview.py::TrackingRunPreview`, a frame viewer over the run's stored assignments). The pre-run, in-memory preview below is design:

Samples N frames from the video. For each sample frame, loads detection rows from the source zarr, applies current hyperparams and arena restriction in memory, runs a lightweight in-memory version of the tracker over the sample set, and overlays results: each track ID rendered in a distinct color. The user sees whether the tracker correctly links the subject across sample frames.

Controls:

- Confidence threshold slider
- IoU threshold slider
- Max missed frames spinner
- Arena polygon editor (same drawing UI as Section 10)

The preview is entirely in memory — no tracking run zarr is written until the user confirms. Confirming submits a `TrackingJob` to the executor.

### Animal identity

Track IDs are integers local to a tracking run (assigned in order of first appearance). Animal identity — linking `track_id: 0` to `"M001"` — is a separate mapping layer.

**`subject_labels`** in `track.yaml` is the slot for this mapping. Every tracker writes it as `null`, and **nothing populates it yet** — neither a manual labelling surface nor a re-identification stage is built. No structural change is required when that workflow is added.

**Not built: selecting a track by animal ID.** `core/utils.py::resolve_track_id` below exists, but no stage calls it: downstream stages take an `int` track ID. Until this lands, `core/utils.py::check_track_is_unambiguous` refuses to pool a multi-track run across recordings by its integer track ID, since the same integer can name a different animal in each recording. The design — a "which track" parameter accepting either an `int` (direct track ID) or a `str` (animal ID resolved through `subject_labels`):

```python
def resolve_track_id(track_id: int | str, subject_labels: dict | None) -> int:
    if isinstance(track_id, int):
        return track_id
    if not subject_labels:
        raise ValueError(f"Cannot resolve '{track_id}': no subject_labels on this tracking run")
    matches = [tid for tid, aid in subject_labels.items() if aid == track_id]
    if not matches:
        raise ValueError(f"animal_id '{track_id}' not found in subject_labels")
    return matches[0]
```

Once built: analysis runs written before identity assignment reference `track_id: 0`; after labels are assigned, new runs can reference `track_id: "M001"`. Same analysis code, same zarr structure, no migration.

### Relationship to `object_id` in Sections 5 and 9

Tracking runs are the authoritative and exclusive source of identity. The detection zarr has no `object_id` column — it was removed to eliminate permanently-dead data. `Detection.object_id` in the Section 5 dataclass is retained for in-memory use during detection model output, but is never written to the detection zarr. Multiple independent tracking runs can reference the same detection zarr without any conflict.

### Validation checks

**Pre-tracking:**

| Check | Action on failure |
|-------|-------------------|
| Detection run `status: "complete"` | Error |
| `tracker_id` present in TRACKERS registry | Error — list available trackers |
| `tracker.validate_hyperparams()` passes | Error with field-level detail |
| Arena IDs exist in recording arenas | Error |
| No active tracking run already running on this detection run | Error |

**Per-part:**

| Check | Action on failure |
|-------|-------------------|
| Source part zarr readable | Error — mark run `failed` |
| `part_handoff.json` absent | Not an error: the run starts at part 0. A later part is only reached through the handoff, so there is no "non-first part without a handoff" case to check |

**Post-part — not built** (no tracker checks this today):

| Check | Action on failure |
|-------|-------------------|
| `object_id` array length matches source zarr row count | Error — mark run `partial` |

**Post-tracking:**

| Check | Action on failure |
|-------|-------------------|
| All parts complete | Report which failed/partial; the run can be re-run (`detail process retry`) |
| At least one track produced | Error — zero tracks produced; raise with message indicating hyperparams may be too strict |

---

## 14. Embedding Generation

> Implemented as a `ProcessingModule` (§6.5), `upstream = "tracking"`. The render-preview tool (below) is a synchronous inspection helper, not a module. The schemas below are unchanged.

The embedding stage reads tracking results, renders each valid detection into an image using a configurable pipeline of transforms, and passes the rendered images through an embedding model to produce per-frame feature vectors. The rendered image — not the raw video frame — is what the model receives. This rendering step is where CV research on mask representation happens; it is designed to be maximally flexible and composable.

Two independent abstractions:

- **`MaskRenderer`** — takes raw detection components (frame, mask, bbox) and produces a rendered image. Composed from a pipeline of `MaskTransform` steps. Fully serializable so every experiment is reproducible.
- **`EmbeddingModel`** — takes a rendered image and returns a feature vector. Same protocol pattern as `DetectionModel`, `Trainer`, and `Tracker`.

---

### `RenderContext` and `MaskTransform` protocol

Each transform receives a context, does one operation, and returns the updated context. A `MaskRenderer` is a sequence of transforms applied in order.

```python
@dataclass
class RenderContext:
    frame: np.ndarray | None                        # original RGB frame (H, W, 3); None when no
                                                    # transform reads pixels and decode was skipped
    mask: np.ndarray | None                         # binary mask (H, W) bool, decoded from COCO RLE
    bbox: tuple[float, float, float, float] | None  # x1,y1,x2,y2 normalized
    image: np.ndarray | None                        # current rendered image — updated by each transform
    frame_idx: int
    t_ns: int
    class_name: str
    confidence: float
    object_id: int
    aux: dict   # open slot: secondary masks, metadata, research extensions (e.g. second animal's mask)

class MaskTransform(ABC):
    def apply(self, ctx: RenderContext) -> RenderContext: ...
    def to_spec(self) -> dict: ...    # serializes to {"type": "...", param: value, ...}
```

`aux` carries anything a transform needs outside the standard fields. Today the alignment steps use it: the worker puts the frame's heading there (`aux["heading_deg"]`, `aux["heading_source"]`, `aux["align_mirrored"]`) for `align_rotate` / `align_mirror` (*Egocentric alignment* below). **Not built:** multi-animal interaction frames — loading a second track's detection for the same frame into `aux["track_1_mask"]` for a transform to read.

A renderer reaches a worker as its spec (`EmbeddingJob.renderer_spec`) and is rebuilt there with `MaskRenderer.from_spec`, so every transform must round-trip through `to_spec`. Rows are rendered on a thread pool inside the worker (§14 *Execution*), not in DataLoader worker processes; transforms are dataclasses, so they stay picklable for the multi-GPU process pool all the same.

---

### `MaskRenderer`

```python
class MaskRenderer:
    def __init__(self, transforms: list[MaskTransform]): ...

    def render(self, ctx: RenderContext) -> np.ndarray:
        for t in self.transforms:
            ctx = t.apply(ctx)
        return ctx.image

    @classmethod
    def from_spec(cls, spec: list[dict]) -> "MaskRenderer":
        return cls([
            TRANSFORMS[s["type"]](**{k: v for k, v in s.items() if k != "type"})
            for s in spec
        ])

    def to_spec(self) -> list[dict]:
        return [t.to_spec() for t in self.transforms]
```

---

### Transform registry

```python
TRANSFORMS: dict[str, type[MaskTransform]] = {}

def register_transform(name: str):
    def decorator(cls):
        TRANSFORMS[name] = cls
        return cls
    return decorator
```

First transforms to implement:

```python
@register_transform("init_binary_mask")
class InitBinaryMask(MaskTransform):
    """White mask on black background, same resolution as source frame.
    Always the first transform for binary mask workflows. Sets ctx.image."""

@register_transform("crop_to_bbox")
class CropToBbox(MaskTransform):
    padding: float = 0.0    # fractional padding around bbox on each side

@register_transform("pad_to_square")
class PadToSquare(MaskTransform):
    """Pad shorter axis to match longer axis. No cropping, no distortion."""

@register_transform("resize")
class Resize(MaskTransform):
    size: int    # both dimensions — output is (size, size, C)

@register_transform("apply_mask")
class ApplyMask(MaskTransform):
    """Keep ctx.frame's pixels under the mask, flatten the rest to `background`.
    Leads a masked-RGB pipeline (initializes ctx.image from ctx.frame); resizes
    the mask to the frame when their resolutions differ."""
    background: int = 0

@register_transform("center_in_square")
class CenterInSquare(MaskTransform):
    """Place ctx.image, unscaled, centered in a size×size black canvas — so
    different-sized masks keep their true relative scale. Larger-than-canvas
    images are center-cropped."""
    size: int
```

Also registered (`transforms/basic.py`): `init_frame` (the whole decoded frame — the `full_frame` constructor), `fill_background_noise` and `fill_background_bands`, and the alignment steps `align_rotate` / `align_mirror` (*Egocentric alignment* below).

Example — binary mask pipeline (first implementation):

```python
renderer_spec = [
    {"type": "init_binary_mask"},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 224},
]
```

Example — masked-RGB pipeline (real pixels under the mask):

```python
renderer_spec = [
    {"type": "apply_mask", "background": 0},
    {"type": "crop_to_bbox", "padding": 0.1},
    {"type": "pad_to_square"},
    {"type": "resize", "size": 224},
]
```

### Named constructors

`transforms/constructors.py` exposes a small set of named recipes (the GUI's
"Input image" choices) that build a `renderer_spec`. The GUI offers them by name
with no further controls; the builder functions take explicit parameters so the
API can drive finer experiments. Current set (`CONSTRUCTORS`; `DEFAULT_CONSTRUCTOR` is
`masked_color_centered_contrast`):

- `silhouette_centered` — `crop_to_bbox → init_binary_mask → center_in_square`.
- `masked_color_centered` — `crop_to_bbox → apply_mask → center_in_square`.
- `masked_color_centered_contrast` (the default) — `masked_color_centered` on the
  max-contrast background color: `crop_to_bbox → apply_mask(background=bg) →
  center_in_square(fill=bg)`.
- `full_frame` — `init_frame → pad_to_square`: the whole decoded frame, no mask and no
  crop (the scene baseline).

The three `*_centered` recipes size their shared canvas to the **largest mask
across the tracks being embedded** (`center_in_square`'s `size`), so every mask
keeps its relative size. That size is data-dependent, and is resolved at a
**scope** the run chooses (`canvas_scope`, `per_recording` by default):

- **`per_recording`** — `EmbeddingModule.validate` (and
  `Recording.resolve_constructor` for the preview) computes it from *this*
  recording via `pipeline.embedding.largest_mask_px` and bakes the concrete
  `renderer_spec` into the job. Across recordings this **normalizes**: every
  subject's own largest mask becomes its canvas, so body-size differences between
  animals disappear.
- **`shared`** — `EmbeddingModule.resolve_shared_config` resolves one value
  across the whole selected subset *before* the per-recording `validate` loop
  (the canvas is the **max**, the only size that still contains every subject's
  largest mask) and passes it in. This keeps the recordings on one absolute scale,
  at the cost of a smaller subject filling less of the model input.

The `masked_color_centered_contrast` background color is also data-dependent, but it
has **no scope to choose**: `resolve_shared_config` always resolves it once across the
selection, from the subjects' mean masked color pooled over every selected recording
(`_max_contrast_rgb`, in RGB: the order the renderer paints and the model reads), and every
recording's `validate` bakes that one color in. A
color resolved per recording is a cue that identifies the recording in every frame the
encoder sees, so it is never what a cross-recording comparison wants. A single-recording
run resolves it from that recording, which is the same thing over a selection of one.
`embedding.yaml` records `bg_color_scope: "shared"` on every new run; a run written
before this records `per_recording`, or nothing, and is read by the color it resolved.

The API path (`run_embedding` with an explicit `renderer_spec`) bypasses both and
records the spec verbatim.

**Comparability follows the canvas scope, not the resolved canvas** (`validate_comparability`).
Two runs are comparable when their *recipe identity* matches — the named
`constructor`, `model_id`, `model_variant`, `readout`, `input_size`, and every part of
`renderer_spec` other than a per-recording canvas. A canvas whose recorded scope is
`per_recording` **may differ between recordings** (that difference is the
normalization, not a mismatch); a `shared` one must be identical, and the two runs'
canvas scopes must themselves agree. The background color is compared **by value** on
every step that carries it (`apply_mask`, `center_in_square`, and the `align_rotate`
fill alignment copies), whatever `bg_color_scope` a run recorded: the same color is the
same image, and two colors are a recording cue. A run with no named `constructor` has no
recipe identity to compare, so it keeps strict whole-`renderer_spec` equality. Both
scopes are recorded in `embedding.yaml` and the job; an absent `canvas_scope` (a run
written before it existed) reads as `per_recording`, which is what it was.

**Render at the model's input size (not the canvas).** `resolve_renderer_spec`
appends a final `resize` to the embedding model's `EmbeddingModel.input_size`
(DINOv3 = 224) when the caller sends no explicit `resize` — so the renderer hands the
model exactly its input, instead of a large canvas the HF processor re-downscales
*every frame* (the dominant cost at 1080p masks). All three surfaces resolve the same
target — the model's own input size: the CLI's `--resize` (which **defaults to unset**,
deferring to the model, not a hard-coded 224), the GUI (which sends none → the model
default), and the input-image scrubber preview. `Resize` uses `cv2.INTER_AREA` when
shrinking (antialiased, matching the processor's own downscale).

**Crop-first, and crop to the subject.** The `*_centered` recipes lead with
`crop_to_bbox`, which crops the *source* (`ctx.mask`, and `ctx.frame` if present) when
no image is built yet — so `init_binary_mask` / `apply_mask` build only the subject
region, never a full-resolution image cropped away afterward. This is output-identical
to cropping after **when the mask and frame share a resolution** — the production
invariant, since detection emits frame-resolution masks. If a masked-color render is
ever fed a frame and mask of *different* resolutions, crop and mask→image stop
commuting exactly (each grid rounds the bbox independently), so the subject boundary can
shift; the mask-only silhouettes are always identical (single grid). The worker threads
the per-frame **union** bbox (`_embedding_worker.embed_one`, matching `render_preview`)
into `RenderContext.bbox`; previously `bbox=None` made `crop_to_bbox` a no-op and
`center_in_square` center-cropped the *frame*, dropping edge subjects to blank —
preview and embedding now agree. **Reproducibility:** two changes shift the pixels a
stored `renderer_spec` produces, so re-running an old spec differs from the original run
(re-embed; query and clustering already compare by spec). (1) A spec containing `crop_to_bbox`
now crops to the subject (the bbox is threaded in; it was a no-op before). (2) `Resize`
now uses `INTER_AREA` on any downscale — the interpolation is not encoded in the spec,
so *every* shrinking spec, including a plain `silhouette_centered`, re-renders slightly
differently.

**Mask-only renders skip decode and run in parallel.** A renderer whose transforms
all leave `MaskTransform.reads_frame` False (`ApplyMask` and `InitFrame` set it) needs no video
(`MaskRenderer.needs_frame`), so `DetectionMaskDataset` opens no `cv2.VideoCapture` and
renders from the detection mask alone. Those rows are independent, so the worker renders
them on a bounded thread pool ahead of the GPU consumer (the RLE-decode + numpy render
release the GIL, overlapping the forward); the GIL-heavy preprocess stays on the
consumer (cheap at 224px), which raises throughput several-fold over the serial baseline.
The pixel-reading path keeps the sequential
`cap.read()` walk (Tenet 11).

---

### Render preview

Before launching an embedding run the user can preview the input images two ways.
The **input-image scrubber** (`Recording.input_image_preview` → `InputImagePreview`,
GUI `InputImageScrubber`, opened from the embedding form) scrubs every source frame,
showing the original frame beside the constructed model input for one track at a time
(empty where the track has no detection). The older **render preview**
(`Recording.run_render_preview`) is **API-only**: it renders a random sample of
detections and, with `save_as`, writes source + rendered JPGs and `preview.json` under
`embeddings/render_previews/<save_as>/`. No GUI or CLI surface calls it (`docs/cli.md`
lists the renderer preview under "Not in the CLI"). Both run the `MaskRenderer` only —
no embedding model required.

Render-preview reads the recording's video at the detection rows' own frame indices, so the source-frame column shows exactly what the model was looking at when it produced the rows being rendered.

```python
@dataclass
class RenderPreviewJob:
    tracking_run_id: str
    track_ids: list[int]
    renderer_spec: list[dict]
    n_sample_frames: int

@dataclass
class RenderPreviewResult:
    frame_idx: int
    t_ns: int
    track_id: int
    source_frame: np.ndarray     # original video frame with mask overlay — shown for reference
    rendered_image: np.ndarray   # output of the renderer pipeline — what the model will receive
    det_row: int                 # back-reference to source detection zarr row
```

Samples are drawn uniformly across all parts and all requested `track_ids`. **Not built:** a GUI over this sample — two columns per sample (source frame with mask overlaid, rendered image) that re-runs the renderer on the already-loaded sample when a transform parameter changes.

---

### `EmbeddingModel` protocol and registry

```python
class EmbeddingModel(ABC):
    model_id: str
    output_dim: int
    supported_backends: list[str]

    def is_installed(self) -> bool: ...   # pure disk check, never the network

    def load(self, device: str) -> None:
        """Load present weights into memory (assumes install() has run). Never touches
        the network — resolve a local path under model_cache_dir() and pass it explicitly,
        raising if absent (§5). Must call model.eval()."""

    def install(self, progress_cb: Callable[[InstallProgress], None] | None = None) -> None:
        """Make weights present locally (one-time download). Idempotent; default no-op.
        Same readiness/install/mixin pattern as DetectionModel (Section 5)."""

    def readiness(self) -> ModelReadiness: ...

    def is_ready(self) -> bool: ...   # return self.readiness().ready

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Single image convenience method. (H,W,C) → (D,). Wraps embed_batch internally."""

    def embed_batch(self, images: torch.Tensor) -> torch.Tensor:
        """(B,H,W,C) tensor → (B,D) tensor.
        Must be wrapped in torch.inference_mode().
        CUDA implementations should also wrap with torch.autocast(device_type='cuda', dtype=torch.float16).
        Caller writes output as .cpu().numpy() to zarr."""

    # Optional split of embed_batch (concrete defaults; a model opts in by overriding both)
    def prepare_batch(self, images: list[np.ndarray]) -> object:
        """The CPU half: what the model does to a batch before it touches the device.
        Needs no GPU; safe on a thread other than the one calling embed_prepared.
        Default: returns images unchanged."""

    def embed_prepared(self, prepared: object) -> torch.Tensor:
        """The device half, given prepare_batch's output. Default: embed_batch(prepared)."""

EMBEDDING_MODELS: dict[str, type[EmbeddingModel]] = {}

def register_embedding_model(model_id: str): ...
```

**The split is what lets the worker take a model's preprocessing off the inference thread**
(§14 *Execution*). A model that overrides it must keep `embed_batch(x) ==
embed_prepared(prepare_batch(x))` exactly; DINOv3 does it by writing `embed_batch` as that
composition. Its `prepare_batch` is the HF image processor over the **whole batch**, the same
call on the same batch as before the split, and `embed_prepared` is the device copy, the forward
pass and the pooling (`tests/models/test_dinov3.py` pins both the composition and each row to its
own image). Models that do not opt in (and any direct caller of `embed_batch` such as the
subject-frame picker) are unchanged: their `embed_batch` simply runs on the inference thread.

---

### `PatchFeatures` — dense patch features, a capability an embedding model declares

> Specified 2026-10-07 and built in PR 3 of the §26 build: `DINOv3ViT(_DINOv3Base, PatchFeatures)`.

A patch-matching keypoint method (§26) needs the backbone's **per-patch** features, not the
pooled vector an embedding reads out. That is a capability of the same model and the same
weights, so it is declared on the embedding model as an ABC mixin rather than registered as a
second model:

```python
class PatchFeatures(ABC):                     # mixed into an EmbeddingModel that offers it
    patch_size: int                           # DINOv3 ViT: 16; the image's sides must be multiples of it

    def patch_grids(self, images: list[np.ndarray]) -> list:
        """One L2-normalised feature grid (C, H/patch_size, W/patch_size) per RGB uint8 image —
        the final layer's patch tokens after the model's final LayerNorm, CLS and register
        tokens dropped, never resized (a patch is exactly patch_size image pixels). fp16 on CUDA,
        fp32 elsewhere. After load(); never touches the network."""
```

`DINOv3ViT` is the first (`dinov3_vit`; all three sizes, patch 16). The Model manager shows the
capability on the model's row — *embedding · patch features* — so one download is listed once
and read for both uses. Final layer plus the final LayerNorm is DINOv3's own recommendation for
dense features; an intermediate layer, if ever tried, is normalised another way and measured
first.

---

### `EmbeddingProgressEvent`

```python
@dataclass
class EmbeddingProgressEvent:
    track_id: int
    parts_done: int
    parts_total: int
    frames_done: int            # run-wide: completed units' frames + the in-flight unit's live count
    frames_total: int           # run-wide: every unit's frames, counted before the run starts
    elapsed_s: float
    eta_s: float | None         # from frames_done / frames_total
    units_done: int = 0         # (track, part) units completed across the run
    units_total: int = 0        # total units
    message: str | None = None  # ephemeral phase text, e.g. "loading model"
```

---

### `EmbeddingJob`

```python
@dataclass
class EmbeddingJob:
    job_id: str
    recording_id: str
    embedding_run_id: str
    tracking_run_id: str
    track_ids: list[int]        # which tracks to embed
    model_id: str
    renderer_spec: list[dict]   # serializable MaskRenderer pipeline — exact record of what was used
    batch_size: int
    num_workers: int            # not read: §14 *Execution* renders on threads; kept for runs on disk
    compute: ComputeTarget
    created_at: str
    model_variant: str | None = None      # None = the model's default variant
    constructor: str | None = None        # named recipe that produced renderer_spec, if any
    readout: str | None = None            # feature read-out; None = the model's default
    input_size: int | None = None         # model input resolution (px); None = the model's default
    canvas_scope: str = "per_recording"   # scope the canvas was resolved at (Named constructors)
    bg_color_scope: str = "shared"        # scope the background color was resolved at
    every_n_frames: int = 1               # a multiple of the stride detection ran at (§4)
    align: str = "none"                   # Egocentric alignment; "none" leaves the fields below at defaults
    keypoint_run_id: str | None = None    # the keypoint track run (kt_NN) the heading reads
    heading_definition: dict | None = None  # {front: [...], back: [...]}, resolved
    heading_cutoff: float | None = None   # the track run's decode cut-off, if it had one
    max_gap_s: float = 0.5
    mirror_hold_s: float = 1.0
    facing_camera_ratio: float = 0.5
```

---

### Execution: decode thread → render pool → GPU

Each `(track_id, part)` is processed as one independent pass producing one zarr. This matches the per-track-per-part zarr structure and avoids sharing `cv2.VideoCapture` across part boundaries.

`DetectionMaskDataset` is **lazy and sequential**: `walk_frames()` opens one `cv2.VideoCapture` and walks it forward (Tenet 11 — no seeking after open), yielding `(i, frame_idx, rgb_frame)` for each row. Frames between the track's rows are skipped with `cap.grab()`: still decoded, since an H.264 frame can be a reference for the next, but never converted — the same walk detection and keypoint detection make. `_render_row(i, frame_idx, frame)` decodes and OR-merges the row's masks, records its kinematics geometry, and runs the `MaskRenderer`. Iterating the dataset composes the two serially; that serial order is the reference the worker is held to.

```python
class DetectionMaskDataset:
    def __init__(
        self,
        video_path: Path,
        frame_indices: np.ndarray,   # sorted ascending inside __init__
        t_ns: np.ndarray,
        det_rows: np.ndarray,
        mask_rles_per_frame: list[list[str]],
        mask_size: tuple[int, int],
        renderer: MaskRenderer,
        track_id: int,
        bboxes: list[tuple] | None = None,
        aux_per_frame: list[dict] | None = None,
    ): ...

    def walk_frames(self):           # the only reader of the video
        # yields (i, frame_idx, rgb_frame)

    def __iter__(self):              # serial: _render_row over walk_frames
        # yields {"image": np.ndarray, "frame_idx": int, "t_ns": int, "det_row": int}
```

**The worker overlaps five stages** (`_embedding_worker.embed_one` and `_render_batches`, all through `core/lookahead.py`), so the GPU stops waiting on the CPU:

```
decode thread      render pool                  prep thread           calling thread          writer thread
walk_frames() ─►  _render_row, in frame   ─►   model.prepare_batch ─► model.embed_prepared ─►  zarr rows, kinematics,
one forward walk  order (window 3×batch)       (2 batches ahead)     forward only            progress file (2 ahead)
```

- **The prep thread** makes the model's CPU call on each whole batch (`prepare_batch`, protocol above). It is the same call on the same batch the model made inside `embed_batch` before the split, so it is identical by construction.
- **The calling thread** runs only `embed_prepared` (device copy, forward pass, copy back) inside `torch.inference_mode()`, which is thread-local and so stays with the forward. The stop flag is checked before each forward.
- **One writer thread** puts each batch's embedding and kinematics rows on disk and updates the progress file, so rows stay in order. `frames_done` is the rows it wrote. On a stop the batches already computed are written; a write error fails the unit after the batches before it. The zarr writers are opened and closed on the calling thread, after the writer has joined.

- **A mask-only renderer opens no video**: rows go straight to the render pool (`needs_frame` below).
- **Rows are byte-identical to the serial iteration**: the same `_render_row` on the same frame, handed back in frame order. `tests/pipeline/test_embedding_worker.py` holds every renderer family to that, and fails on a scrambled order, a dropped frame, or a row rendered from another row's frame.
- **Threads, not processes.** cv2 decode / convert / resize / border, pycocotools' RLE decode and torch release the GIL, so threads overlap; processes would pickle each decoded frame through a pipe and need one `VideoCapture` per worker, each seeking to its slice boundary — which can snap to a keyframe and pair an embedding with the wrong frame. `EmbeddingJob.num_workers` is not read (it is kept on the job because runs on disk carry it).
- **Batches are lists of numpy images.** The model builds its own tensor (`dinov3`'s HF processor in `prepare_batch`), so a collated, pinned tensor would only be converted back.
- **Threads per worker** follow `core/cpu.py`: the cores this process may run on (its affinity set — a SLURM allocation, not the node), less one, at most 8, split among the per-GPU workers of a multi-GPU run.
- **Memory is bounded** (Tenet 11): at most `decode depth + window + 2` rows are held at once — at batch 32 and 8 render threads about 110 decoded frames, ~0.7 GB at 1080p, reached only while render is the slower stage. Never O(video).
- **Errors and stop are the serial path's.** A video that ends early raises after the rows before it. Every stage is a generator closed by the one after it (stop flag, a model or write error), so the writer, prep, render and decode threads all stop and join with the unit, releasing the `VideoCapture`.

**Detection (Section 7)** uses the same seam: decode, the model's CPU half, the forward and the writes on their own threads (§7 *Execution*).

**`ctx.frame` carries the real decoded video frame** — or `None` when the renderer reads no pixels. Only `apply_mask` and `init_frame` consume the frame's pixels (`MaskTransform.reads_frame` True); a renderer built from the mask alone (the silhouette constructors) has `MaskRenderer.needs_frame` False, so the worker skips decode and passes `frame=None` (`crop_to_bbox` still crops the frame when it *is* present, but never requires it). Note: `init_binary_mask` sizes its output from `ctx.mask`, not `ctx.frame`, since the mask is what's being rendered — the frame and the detection mask are sized independently.

---

### Multi-GPU embedding

When `compute.backend == "cuda"` and `len(device_indices) > 1`, the coordinator parallelizes across GPUs. Otherwise (single device, MPS, or CPU) it short-circuits to the inline path above — zero subprocess overhead.

**Work unit: `(track_id, part_index)`.** Each unit is one pass of the worker (decode → render → GPU, above) producing one zarr at `embeddings/<run_id>/track_<tid>/part_<NNN>.zarr`. Per-unit zarrs are independent stores — workers never contend on the same file. With `N_tracks × N_parts` units and `K` GPUs, this exposes maximum parallelism (including when `N_tracks=1` — still K-way across parts).

**Topology: one `ProcessPoolExecutor(max_workers=1)` per GPU.** Each pool's `initializer` pins `CUDA_VISIBLE_DEVICES=<idx>` *before* importing torch, then loads the model + builds the `MaskRenderer` once and stashes them in worker-local state. Tasks dispatched to the same pool reuse that state, so model load is amortized across all units that GPU handles. Units are dispatched round-robin: `units[i]` goes to `pools[i % K]`. We need one initializer per GPU but `ProcessPoolExecutor`'s initializer is shared across all workers — so we create `K` separate single-worker pools rather than one `K`-worker pool. Same shape as multi-GPU detection (§7).

**Stop = fail** (not partial). The coordinator touches `<run_dir>/stop.flag`; workers poll it between batches and exit early with `status="failed"`. Embedding has no resume support, so a partial run would be misleading — surfacing as `failed` is honest. Mirrors detection's stop contract.

**Cross-process channel is file-based.** Per-unit progress JSONs land at `<run_dir>/parts/track_<tid>_<part_id>.json`, written atomically (tmp + `core.atomic.replace`) so readers never see torn writes. Progress events are emitted whenever a unit's future completes — scoped to that unit's `track_id`, with `parts_done` climbing as units for that track complete (out-of-submission-order is fine since each track's parts are independent).

**Worker output_dim round-trip.** Because the model loads inside the worker (not the coordinator), the worker's status dict includes `output_dim`. The coordinator reads the first non-zero value to write `embedding.yaml`.

**Cuda-index validation.** `EmbeddingModule.validate` calls `core.utils.check_cuda_indices` when `backend == "cuda"`, so a request for a non-existent GPU fails before any subprocess spawns.

---

### Embedding run directory structure

```
recordings/recording_001/
├── detections/
├── tracking/
└── embeddings/
    └── emb_01/
        ├── embedding.yaml
        ├── track_0/
        │   ├── part_001.zarr/
        │   │   ├── frame_idx    (M,) int32       chunks=(10000,)
        │   │   ├── t_ns         (M,) int64       chunks=(10000,)
        │   │   ├── det_row      (M,) int32       chunks=(10000,)   ← row index in source detection zarr
        │   │   └── embedding    (M, D) float32   chunks=(1000, D)  ← D=model.output_dim
        │   ├── part_002.zarr/
        │   └── kinematics/                                          ← sibling, see below
        │       ├── part_001.zarr/
        │       └── part_002.zarr/
        └── track_1/
            └── part_001.zarr/
```

`M` = frames in this part where the track has a valid detection. `D` = `model.output_dim`. **`frame_idx` is part-local**: it counts from 0 inside this part, as every per-part store does (§5 `FrameResult.frame_idx`), while `t_ns` is on the recording's clock. The readers (`run_preview.load_track_trajectory` / `load_track_timeline`) return it **recording-wide**, adding the part's start frame from `recording_store.part_start_frames` (keyed by part stem), so anything built from them (query and clustering instances, frame-span examples, the coverage band, Measurements) numbers frames across the whole recording. `det_row` is a back-reference to the source detection zarr row — allows re-rendering with a different pipeline or inspecting the original mask without re-running detection.

**`det_row` is valid only within the part that wrote it, and names the frame's *representative* detection.** A detection is identified by `(track_id, part_index, det_row)` — the embedding zarr stores no part, so the part is whichever file the row is in, and a flat concatenated read resolves one part's index against another's. Where a frame carries several detections of the subject (anchor secondaries), they are OR-merged into one silhouette and `det_row` records the highest-confidence one only. Both facts make `det_row` a *traceability* handle, not a geometry source: reconstructing where the animal was from it mis-resolves across parts and returns the largest fragment's box rather than the animal's.

---

### `kinematics/` — what the subject was, per embedded frame

The pipeline's home for **per-frame, per-subject derived quantities**: one row per embedded frame,
in the same order, describing the silhouette the embedding was taken from. Written by the embedding
pass, which already decodes and OR-merges the mask, so these cost a `.sum()` and a moment
calculation on an array that is in memory anyway.

```
kinematics/part_001.zarr/
  frame_idx       (M,) int32     chunks=(10000,)     joins to the embedding part, same order
  t_ns            (M,) int64     chunks=(10000,)
  cx_norm         (M,) float32   chunks=(10000,)     subject centroid, x, [0, 1]
  cy_norm         (M,) float32   chunks=(10000,)     subject centroid, y, [0, 1]
  mask_area_norm  (M,) float32   chunks=(10000,)     mask pixels ÷ frame pixels
  bbox_norm       (M, 4) float32 chunks=(10000, 4)   union xyxy over the frame's detections
```

**A sibling, not extra columns on the embedding part.** Tenet: a stage that annotates earlier output
writes a parallel artifact alongside it, never into it. The measured reason is in
`api/run_preview.py::load_track_timeline` — a long track's embedding column is hundreds of MB and
slow to read, against the small columns beside it. "Where was the animal" must not load a
wall of floats. It also means a backfill writes a small new file rather than rewriting the
embeddings, and the keypoint stages (§26) write their own per-frame layer beside this one, joined
on `(track_id, t_ns)`, without touching either.

**Everything is normalized**, matching the detection zarr's `bbox` convention, so a value is
comparable across recordings of different resolutions. Absolute pixels are recoverable by multiplying by the frame size; the reverse is not
recoverable without knowing it.

**`cx_norm` / `cy_norm` are the *mask* centroid**, falling back to the centroid of `bbox_norm` when
the mask is empty or absent, and NaN when there is no detection to describe. NaN is "undefined", not
zero — a zero would place the animal in the top-left corner, which is a plausible position and
therefore the worst possible filler.

**`bbox_norm` is the union over the frame's detections**, not the representative's box — the same
quantity `render_preview._Candidate` computes, and the reason this is captured here rather than
reconstructed from `det_row` later.

**Optional on read.** A run made before this existed has no `kinematics/`, which readers treat as
absent rather than as an error; `detail recording backfill-kinematics` regenerates it from the
detection zarr without a GPU or a model.

---

### `embedding.yaml`

`output_dim` is reported by the loaded model (`EmbeddingModel.output_dim`, read from the underlying config), not set by the user. The runner records whatever the model reports — `1024` for the default ViT-L, other DINOv3 variants differ (e.g. ViT-S/B return `384`/`768`, ConvNeXt-Base `1024`).

**The default embedding model is `dinov3_vit` at `vit_large`, input 224 px**, declared once on
`configs/embedding.py::EmbeddingConfig.model_id` and `DINOv3ViT.DEFAULT_VARIANT` (the input size
is the model's own). `validate` writes the resolved `model_id`, `model_variant` and
`constructor` back, so a run launched without them records what ran. The CLI `--model` options,
the GUI form and the Model manager's suggested install all read the same two declarations.

**Channel order.** cv2 decodes BGR; `RenderContext.frame` and every embedding model read RGB.
`DetectionMaskDataset` converts each frame once, as it is read (`core/visualize.bgr_to_rgb`),
so the worker and the in-process path, and every model behind them, see RGB; the max-contrast
background is computed and stored in RGB. Runs made before this embedded BGR frames — a red
subject reached the model in the blue channel — and record no `channel_order`; it reads as
`"bgr"`, a recipe key (`api/run_preview._RECIPE_KEYS`),
so query, clustering and evaluation refuse to pool them with RGB runs.

```yaml
schema_version: 1
id: "emb_01"
created_at: "2026-06-03T10:00:00Z"
tracking_run_id: "track_01"
model_id: "dinov3_vit"
model_variant: "vit_large"  # the resolved size, written back by validate
constructor: "masked_color_centered_contrast"   # named recipe that produced renderer_spec, or null
readout: null               # which model output is taken as the vector (null = the model's default)
input_size: null            # the model input edge in px (null = the model's own, 224 here)
output_dim: 1024
track_ids: [0]
canvas_scope: "per_recording"   # what a *_centered recipe sizes its canvas over
bg_color_scope: "shared"   # the background colour: always one per run (a run before that says per_recording)
channel_order: "rgb"       # the pixel order the model was handed; absent on a run before the fix = "bgr"
renderer_spec:              # the resolved spec — reproduces the run by itself; colours are RGB
  - type: "crop_to_bbox"
  - type: "apply_mask"
    background: [0, 0, 255]
  - type: "center_in_square"
    size: 774
    fill: [0, 0, 255]
  - type: "resize"
    size: 224
batch_size: 32
compute:
  backend: "cuda"
  device_indices: [0]
  cloud_provider: null
  cloud_instance: null
align: "none"               # §14 egocentric alignment; the five fields below are its settings
keypoint_run_id: null
heading_definition: null
heading_cutoff: null
max_gap_s: 0.5
mirror_hold_s: 1.0
facing_camera_ratio: 0.5
heading_coverage: {}
status: "complete"          # "pending" | "running" | "complete" | "failed" | "partial"
```

---

### No `config.yaml` additions

`config.yaml` holds the experiment's identity only (Section 1). The embedding defaults live in `configs/embedding.py::EmbeddingConfig` (model, batch size, constructor), and each run records its resolved values in `embedding.yaml` and its job.

---

### `recording.yaml` additions

```yaml
embedding_runs:
  - id: "emb_01"
    tracking_run_id: "track_01"
    model_id: "dinov3_vit"
    ...                         # plus the run's identity fields (EmbeddingModule.extra_record_fields)
    status: "complete"
    output: "embeddings/emb_01/"
```

---

### Re-running a failed or stopped run

Embedding does not resume. A stop or a failed unit fails the run (*Multi-GPU embedding* above: stop = fail); the run is re-run — `Experiment.retry_run` / `detail process retry` re-runs each incomplete recording whole, under a fresh run id, so nothing appends into a half-written zarr (Section 7).

---

### Egocentric alignment (reads a `keypoint_track` run)

> Specified 2026-10-07 (PR 6 of the §26 build); built 2026-10-08. Replaces the alignment built on the
> `keypoints` branch (#626): the heading no longer comes from a skeleton's `front`/`back`, a frame
> with no heading is never embedded unturned, the hold looks both ways, and the canvas is sized
> from the turned masks rather than the box diagonal.

An embedding run may turn each subject image so the animal faces one way before the model sees
it. The heading comes from a **keypoint track run** (§26) through a **heading definition**:

```python
class EmbeddingConfig(BaseModel):            # additions
    align: Literal["none", "rotate", "mirror"] = "none"
    keypoint_run_id: str | None = None       # a keypoint_track run (kt_NN) on the same tracking run
    heading: str | None = None               # a saved heading definition's name (heading_definitions.yaml)
    max_gap_s: float = 0.5                   # interpolate a missing heading across a gap up to this long
    mirror_hold_s: float = 1.0               # mirror: the new side must hold this long before the flip
    facing_camera_ratio: float = 0.5         # heading axis shorter than this × its typical length = undefined
```

- **Heading definition** — `{front: [names], back: [names]}`, saved per experiment by name in
  `heading_definitions.yaml` (the stream-preset pattern, §3: a combo plus *Save as…* on the form;
  export / import / delete in the CLI), and offered only for skeletons that have every name it
  lists. The heading on a row is `atan2(dy, dx)` in **pixels** (normalized x and y differ in
  length on a non-square frame; y down, 0° = facing +x) from the mean of the back set to the mean
  of the front set, over the points that are present and at or above the track run's decode
  cut-off. A set, not a pair, so one missing point moves the mean a little instead of losing the
  heading (keypoint-MoSeq's `anterior_bodyparts` / `posterior_bodyparts`); a single pair is the
  one-point case. The resolved definition is copied into `embedding.yaml`.
- **Undefined on a row** when every point of one set is missing, or when the axis is shorter than
  `facing_camera_ratio` × that track's typical axis length (its 95th percentile over the run) —
  from the side an animal facing the camera foreshortens its axis to noise, the cause of the
  axis flips measured from the side.
- **Gaps.** A frame's heading is the row's own when defined; across a gap of defined rows no
  longer than `max_gap_s` it is interpolated in time the short way round the circle, looking
  **both** directions; a frame in a longer gap, or before the first / after the last defined row
  (whatever the distance: an edge hold would be a guess), **gets no embedding row** and is counted in coverage. No frame is embedded
  unaligned inside an aligned run — an unturned frame is a different picture, and nothing
  downstream could tell it apart.
- **`rotate`** (overhead): turn by the heading so the animal faces +x. **`mirror`** (side view):
  no turn; flip left-right when the heading points left (`cos < 0`), and only after the new side
  has held for `mirror_hold_s` (no per-frame flicker at ±90°). `none` clears every field above, so
  two unaligned runs never differ in a setting that did nothing.
- **Rendering.** `with_alignment(spec, align)` inserts `align_rotate` / `align_mirror` after the
  subject image is built and cropped, before the first `center_in_square` / `pad_to_square` /
  `resize` (a resize to a square is anisotropic on a non-square crop and would bend the axis).
  `align_rotate` turns the whole crop, re-crops to the **turned mask**, and the recipe centres that
  box on the canvas as every centred recipe does (not the mask centroid, as #626's text said: the
  code centred the box, and the box is what the unaligned recipes centre). Exposed corners take the
  recipe's background. Alignment applies to the centred recipes (`masked_color_centered_contrast`,
  `masked_color_centered`, `silhouette_centered`); `full_frame` is never modified and refuses it;
  an explicit `renderer_spec` must carry exactly the step `align` names.
- **Canvas.** A pre-pass over the stored masks, each turned by its frame's heading, sizes the
  shared canvas to the **largest turned mask box + 2 px** (nearest-neighbour edge) — the same rule
  the unaligned recipes use, applied after turning, so aligned and unaligned frames draw the
  animal at about the same size. CPU only, no video decode, and cached by (tracking run, keypoint
  track run, heading definition). **Largest bound first:** turned by the frame's heading θ, a
  `w × h` mask box spans at most `(w−1)|cos θ| + (h−1)|sin θ|` (and that with `w`, `h` swapped)
  — `core.utils.mask_boxes` reads the box from the RLE without a decode — so frames are taken in
  descending bound and the pass stops at the first that cannot beat the largest so far. The
  frames it does decode are measured from each pixel column's top and bottom pixel
  (`turned_mask_extent`; a column's extreme along any direction is one of its ends). Both exact.
  Decoding and turning every pixel was slow; the bound makes a typical recording fast. The worst
  case is an animal held still at a diagonal heading, where no bound prunes and every mask is
  decoded. It runs inside `validate`, so its progress is the launch's per-recording step (§15
  *Validation*). The box-diagonal canvas it replaces drew most animals smaller than their
  unaligned size.
- **Recorded per frame** on the embedding part zarr, present only when `align != "none"`:
  `heading_deg (M,) float32` (the heading used, interpolated or not; the turn direction alignment
  discards), `heading_source (M,) int8` (0 own row, 1 interpolated), `align_mirrored (M,) bool`.
- **Coverage and alerts.** The run viewer shows, per track, where frames were lost at each step —
  tracking, keypoints, heading (defined / interpolated / dropped) — as `_TrackTimeline` rows, plus
  each point's coverage; the launch form shows the same count before the run. An alert (an amber
  label with the one action that fixes it) when heading coverage is under 95%, the longest heading
  gap is over 2 s, or any point's coverage is under 80%. The three thresholds are policy, not
  measured.
- **Identity and comparability.** `align`, the heading definition, `max_gap_s`, `mirror_hold_s`,
  `facing_camera_ratio`, and the keypoint track run's **identity** (§26 *Identity*: method,
  skeleton digest, example-set digest, track method and settings — never its id) join embedding's
  `identity_fields` and `_RECIPE_KEYS`. Per-recording runs with the same settings still pool; axes
  from different methods or skeletons do not, and aligned never compares with unaligned.
- **Lineage** — embedding's one second input (§6.5 *auxiliary inputs*):
  `aux_refs = (("keypoint_track", "keypoint_run_id"),)`, and validate refuses a keypoint track run
  whose `tracking_run_id` is not the embedding's own tracking run (§26 revision 2026-10-21: a
  keypoint track run names the tracking run it assigned to, rather than descending from it), so
  eligibility stays single-upstream. A
  batch run passes `keypoint_track_spec`, resolved per recording by the same machinery as the
  upstream spec; the run spec gains an `aux` entry keyed on that identity.

**Pre-run (embedding with alignment):**

| Check | Action on failure |
|---|---|
| `align != "none"` without `keypoint_run_id` or `heading`, or either without `align` | Error |
| Keypoint track run complete, from the same tracking run, covering the embedded track ids | Error |
| Every name in the heading definition is in that run's skeleton | Error — name the missing points |
| The input image crops to the subject (`full_frame` does not) | Error — name the recipes that do |
| An explicit `renderer_spec` whose alignment steps disagree with `align` | Error |
| Heading coverage below the alert thresholds | Alert on the form, not an error |

**Later, with research (§26 *Deferred*):** one rule for every camera tilt — turn about the floor's
vertical axis, so overhead is an in-plane rotation, side is a mirror and an oblique camera is a
floor-plane warp then a rotation, with the tilt a per-rig fact — and a study of whether alignment
helps retrieval and clustering at all: none / keypoint alignment / embeddings averaged over 4 or 8
rotations / alignment with ±15° jitter, with an invariance curve (DINOv3 was trained with random
horizontal flips and no rotations, so a mirror may change its embedding very little) and a
heading-leakage check (can a kNN recover heading from the embedding).

---

### Validation checks

**Pre-run:**

| Check | Action on failure |
|-------|-------------------|
| Tracking run `status: "complete"` | Error |
| All `track_ids` exist in `tracks.zarr` | Error |
| `model_id` in EMBEDDING_MODELS registry | Error |
| Model `readiness().ready` (Section 5) | Error — `ModelNotReadyError` with message + `action_url` |
| `compute.backend` in `model.supported_backends` | Error |
| All transform types in `renderer_spec` registered in TRANSFORMS | Error — list unknown transforms |
| Output directory writable | Error |
| Existing `complete` run at this output path | Error — prevent silent overwrite |

**Per-part:**

| Check | Action on failure |
|-------|-------------------|
| Video part readable | Error — the unit fails, so the run is `failed` (any failed unit fails an embedding run) |
| Source detection zarr readable | Error |
| Assignment zarr readable | Error |
| Model loads successfully on device | Error — mark run `failed` |

**Post-part — not built** (nothing compares the row count today):

| Check | Action on failure |
|-------|-------------------|
| Embedding row count matches expected frame count for this track/part | Error — row count mismatch; the unit fails |

**Post-run:**

| Check | Action on failure |
|-------|-------------------|
| All parts complete for all requested tracks | Report which failed/partial; the run can be re-run (`detail process retry`) |

---

## 15. Python API and CLI

All core functionality is exposed through a Python API that has no Qt dependency. The GUI, scripts, and CLI all go through this API to change experiment state — none of them writes `recording.yaml`, zarr stores or job dataclasses, or resolves internal paths, itself. This is the single boundary between frontend and backend.

### The rule

> **Nothing outside the Python API mutates experiment state or resolves internal paths.**

The GUI and the CLI both launch detection as `Experiment.run_processing("detection", recording_ids, config)`. A research script can do the same, or call `recording.run_detection(...)`, a thin wrapper over the same backbone (`Recording.run`, §6.5). The implementation is in one place.

**As built, the rule holds for writes but not for every read.** Several GUI modules read through storage modules directly — `storage/recording.py::read_recording` (e.g. `gui/keypoint_run_preview.py`), `storage/behavior_output.py` and `storage/behavior_audit.py` (the audit and label surfaces), `storage/metadata.py`. One GUI write bypasses the API: cancelling queued imports marks them through `storage/recording.py::set_import_status` (`gui/import_queue.py`).

---

### `Experiment` class

The listings below sketch the main entry points; they are **not** the full surface. `Experiment` has about 150 public members and `Recording` about 100; the full, current list is the generated Python API reference (`docs/reference/api.md`, built from the docstrings of `detail/api/experiment.py` and `detail/api/recording.py`).

```python
class Experiment:
    @classmethod
    def create(cls, name: str, parent_dir: Path | str | None = None) -> "Experiment":
        """Creates experiment directory structure and config.yaml."""

    @classmethod
    def open(cls, experiment_dir: Path | str) -> "Experiment":
        """Loads an existing experiment. Discovers recordings by scanning recordings/."""

    @property
    def recordings(self) -> list["Recording"]: ...

    # A recording's id is the tool's, its name is the user's (§3). add_recording mints the
    # next recording_NNN id (the user-facing way to add one); create_recording takes an
    # explicit id (the form for scripts).
    def add_recording(self, name: str, metadata: dict[str, str] | None = None) -> "Recording": ...
    def create_recording(self, recording_id: str, metadata: dict[str, str] | None = None) -> "Recording": ...
    def get_recording(self, recording_id: str) -> "Recording": ...
    def resolve_recording(self, ref: str) -> "Recording": ...   # a name or an id
    def recording_names(self) -> dict[str, str]: ...           # id -> name

    # Experiment identity — config.yaml fields; the directory is not renamed.
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def set_metadata(
        self, name: str | None = None, description: str | None = None
    ) -> None: ...

    # Processing runs over a recording selection (§15) — spawns a detached coordinator
    # that fans out to per-recording runs; the handle reattaches by polling on disk.
    def run_processing(
        self,
        module_id: str,
        recording_ids: list[str],
        config: dict,
        progress: Callable[[str, float, str], None] | None = None,
    ) -> "AttachedRun": ...
    def attach_run(self, run_id: str) -> "AttachedRun": ...
    def retry_run(self, run_id: str, progress=None) -> "AttachedRun": ...  # re-run incomplete recordings
    @property
    def processing_runs(self) -> list[dict]: ...   # experiment-level runs index

    @property
    def datasets(self) -> list[DatasetRef]: ...

    @property
    def models(self) -> list[ModelRef]: ...

    # §12: a dataset from an existing detection/tracking run (no new inference) ...
    def generate_dataset(
        self,
        name: str,
        source_run_id: str,
        frame_selection: dict | None = None,
        val_split: float = 0.15,
        min_contour_area: int = 100,
    ) -> "DatasetRun": ...

    # ... or from a fresh labeling pass with a prompt-capable model (SAM3 by default)
    def label_dataset(
        self,
        name: str,
        classes: list[LabelingClass],
        n_frames: int,
        source_recording_ids: list[str] | None = None,
        logit_threshold: float = 0.5,
        min_contour_area: int = 100,
        val_split: float = 0.15,
        compute: ComputeTarget | None = None,
        sampling: str = "stratified",
        model_id: str = "sam3",
        model_variant: str | None = None,
        background_share: float = 0.1,   # LabelingConfig's default
    ) -> "DatasetRun": ...

    def train_model(
        self,
        name: str,
        dataset_id: str,
        trainer_id: str,
        base_model: str,
        hyperparams: dict | None = None,
        compute: ComputeTarget | None = None,
    ) -> "ModelRun": ...

    # Credentials and model install (Section 5). set_hf_token writes the
    # machine-global credentials file, not experiment state.
    def set_hf_token(self, token: str) -> str:
        """Validate via whoami, store with 0600 perms, return the account name."""

    def check_model_access(self, model_id: str, variant: str | None = None) -> ModelReadiness:
        """Proactive readiness for a model — drives 'set up access' UI states."""

    def install_model(
        self,
        model_id: str,
        progress_cb: Callable[[InstallProgress], None] | None = None,
        *,
        variant: str | None = None,
    ) -> None:
        """Ensure a model's weights are present; raise ModelNotReadyError if blocked."""

```

---

### `Recording` class

```python
class Recording:
    id: str
    name: str

    # Setup
    def add_video_part(self, path: Path | str, storage: str = "default",
                       video_root: str | None = None, progress=None,
                       should_stop=None) -> None: ...
    # Streams — see Section 3. aux_paths supplies the importer's declared
    # aux_inputs (e.g. {"program": ".MPC"} for MED-PC). Read/edit/delete mirror
    # the arena accessors; offset is applied at read time.
    def add_stream(self, name: str, raw_path: Path | str, importer: str,
                   aux_paths: dict[str, Path | str] | None = None,
                   covered_behaviors: list[str] | None = None, origin: str | None = None,
                   config: dict | None = None, offset_ns: int = 0) -> None: ...
    @property
    def streams(self) -> list[dict]: ...                 # manifest entries
    def stream_events(self, name: str) -> list[tuple]: ...  # canonical rows (stream-local)
    def stream_preview(self, name: str): ...                # StreamPreview: events + lanes for the timeline preview
    def set_stream_offset(self, name: str, offset_ns: int) -> None: ...
    def stream_dependents(self, name: str) -> list[str]: ...  # [] until §16 consumes streams
    def delete_stream(self, name: str) -> None: ...
    def frame_t_ns(self, global_frame_idx: int) -> int: ...  # recording-global frame → video t_ns
    def frame_timestamps(self) -> np.ndarray: ...  # all frames' video t_ns (preview timeline axis)

    # Spatial — see Section 10. Each keyframe is (frame_idx, [(x, y), ...]) with
    # recording-global frame_idx and normalized vertices. Returns the auto-assigned
    # stable id (e.g. "arena_001") which is what TrackingJob.arena_ids refers to.
    # `active` is an ROI's / POI's optional (start, end) interval (Section 10).
    def define_arena(
        self,
        name: str,
        keyframes: list[tuple[int, list[tuple[float, float]]]],
    ) -> str: ...
    def define_roi(
        self,
        name: str,
        keyframes: list[tuple[int, list[tuple[float, float]]]],
        active: tuple[int, int] | None = None,
    ) -> str: ...
    def define_point(
        self,
        name: str,
        keyframes: list[tuple[int, list[tuple[float, float]]]],   # one vertex per keyframe
        active: tuple[int, int] | None = None,
    ) -> str: ...

    @property
    def arenas(self) -> list[dict]: ...   # recording.yaml entries: id, name, n_vertices
    @property
    def rois(self) -> list[dict]: ...     # id, name, n_vertices, active
    @property
    def points(self) -> list[dict]: ...   # id, name, n_vertices (1), active

    # Read accessors (Tenet 2)
    @property
    def video_parts(self) -> list[dict]: ...        # [{index, file, timestamps}, ...]
    @property
    def part_frame_counts(self) -> list[int]: ...   # frames per part (arena slider, §10)
    def set_name(self, name: str) -> None: ...       # edit display name; id/dir unchanged

    # Pipeline stages — every stage runs through the generic backbone (§6.5); the
    # typed methods are thin wrappers, all returning a RunHandle.
    def run(self, module_id: str, config: dict) -> RunHandle: ...

    def run_detection(
        self, model_id: str, prompt: Prompt, compute: ComputeTarget,
        conf_threshold: float = 0.5, model_variant: str | None = None,
        batch_size: int | None = None, every_n_frames: int = 1,
    ) -> RunHandle: ...

    def run_tracking(
        self, detection_run_id: str, tracker_id: str | None = None,
        hyperparams: dict | None = None, arena_ids: list[str] | None = None,
        arena_name: str | None = None, every_n_frames: int | None = None,
    ) -> RunHandle: ...

    def run_embedding(
        self, tracking_run_id: str, track_ids: list[int], model_id: str | None = None,
        renderer_spec: list[dict] | None = None, batch_size: int = 32,
        compute: ComputeTarget | None = None, model_variant: str | None = None,
        constructor: str | None = None, readout: str | None = None,
        input_size: int | None = None, every_n_frames: int | None = None,
        *, align: str = "none", keypoint_run_id: str | None = None,
        heading: str | None = None, max_gap_s: float = 0.5,
        mirror_hold_s: float = 1.0, facing_camera_ratio: float = 0.5,
    ) -> RunHandle: ...

    # Synchronous inspection tools (not modules — §6.5 scope): return results directly.
    # The prompt test keeps every proposal; min_conf is the threshold its saved images and
    # counts are drawn at — a view, not a filter (Section 9 *Confidence threshold*).
    def run_prompt_test(
        self, model_id: str, prompts: list[Prompt], n_sample_frames: int,
        compute: ComputeTarget, save_as: str | None = None, seed: int | None = None,
        model_variant: str | None = None, display_classes: list[str] | None = None,
        min_conf: float = 0.0, on_result=None,
    ) -> list[PromptTestResult]: ...

    def run_render_preview(
        self, tracking_run_id: str, track_ids: list[int], renderer_spec: list[dict],
        n_sample_frames: int, save_as: str | None = None, seed: int | None = None,
    ) -> list[RenderPreviewResult]: ...

    # Queries — run-record dicts from recording.yaml
    @property
    def detection_runs(self) -> list[dict]: ...
    @property
    def tracking_runs(self) -> list[dict]: ...
    @property
    def embedding_runs(self) -> list[dict]: ...
```

---

### Run handles

Every `run_X()` method returns a run handle immediately. The handle exposes progress streaming and blocking wait.

```python
class RunHandle(ABC):
    id: str
    status: str     # mirrors job status states from each section

    def progress(self) -> Iterator[ProgressEvent]:
        """Yields progress events as the job runs. Returns immediately when complete."""

    def wait(self) -> "RunHandle":
        """Blocks until the run is complete or failed. Returns self for chaining."""

    def cancel(self) -> None: ...
```

Script usage:

```python
det = recording.run_detection(...).wait()
track = recording.run_tracking(det.id, ...).wait()
emb = recording.run_embedding(track.id, ...).wait()
```

Or with progress:

```python
run = recording.run_embedding(...)
for event in run.progress():
    print(f"{event.parts_done}/{event.parts_total} parts, eta {event.eta_s:.0f}s")
```

---

### No `confirm_fn` — every validation problem raises

There are no soft warnings in the pipeline, so there is nothing for a run to ask about: a required-configuration decision or a validation failure raises an exception before anything runs (Tenet 10), in scripts, the CLI and the GUI alike. An earlier design passed a `confirm_fn` callback (a dialog in the GUI, `click.confirm` in the CLI) for warnings that needed a decision; no surface passes one. `Recording.run_detection` still accepts a `confirm_fn` keyword and ignores it.

---

### GUI uses the same API

The GUI launches through the same API (`Experiment.run_processing`, from `gui/launch.py`) and never writes `recording.yaml`, zarr stores or job dataclasses itself (the one exception is noted under *The rule* above). Qt workers (`gui/workers.py`) are thin adapters that consume a handle off the UI thread:

```python
class RunProgressWorker(GuiWorker):
    tick = pyqtSignal(int, str)          # permille (0..1000), status text
    progress = pyqtSignal(object)        # the ProcessingProgress event itself
    finished_run = pyqtSignal()

    def run(self) -> None:
        for event in self._handle.progress():          # an AttachedRun
            fraction, status = processing_progress_fraction(event)
            self.emit_to_ui(self.tick, round(fraction * 1000), status)
            self.emit_to_ui(self.progress, event)
        self.emit_to_ui(self.finished_run)
```

The worker's only job is bridging `Iterator[ProgressEvent]` to Qt signals; `processing_progress_fraction` is the GUI's one `event → render` mapping for a processing run (§6). `FitProgressWorker` does the same for a dataset, training or distill run. All business logic is in the API.

---

### API additions for the GUI (status)

The GUI design (`docs/gui_design.md`) drove these additions; all are now **implemented** (all preserve Tenet 2). They are described here, not folded into the listings above, which sketch the main entry points only; the full API is `docs/reference/api.md`.

**`Experiment.run_processing(module_id, recording_ids, config) -> AttachedRun`** — the experiment-level batch construct. A *processing run* applies one processing module (§6.5) with one config across a **selection of recordings**, fanning out to a per-recording `Recording.run(module_id, config)`. Per-recording outputs still write into each `recordings/<id>/…` (Tenet 5); this orchestrates and indexes only. As built:

- **Record & discovery** — each run is `runs/<run_id>/run.yaml` (experiment-level `runs/` dir, in `storage/runs.py`), discovered by scanning `runs/`. The record holds module_id, config, recording_ids, the per-recording run-id map, aggregate status, created_at. Experiment-monotonic `run_001`… ids; the index (`Experiment.processing_runs`) groups by `module_id`. A single-recording run is a selection of one.
- **Validation** — every selected recording is validated up front (`module.validate`) before anything runs (Tenet 10). It reads each recording's outputs, so over a large selection it is minutes, and it reports through the `(stage, fraction, detail)` `progress` callback (§6): `Resolving shared settings` while `resolve_shared_config` reads each recording (it calls its `on_recording(recording_id)` hook first), then `Validating`, each step naming the recording and its place (`Rat 12 (3/48)`). `retry_run(run_id, progress=)` reports its re-validation the same way. The CLI drives a bar (`_with_stage_progress`); the launch dialog and the run page's *Retry* replace their status line with each report. Recordings count equally — the bar does not move *within* one, and a module's `validate` has no progress channel of its own.
- **Execution — durable, via a detached run coordinator.** `run_processing` reserves `run.yaml`, then spawns a **detached coordinator subprocess** (`executor/coordinator.py`, `python -m …`, own process group like a worker) and returns an **`AttachedRun`** that polls the run's `progress.json`. The coordinator runs the orchestration loop (`api/experiment.py::RunOrchestrator`): **serial by default**, **concurrent when the module opts in** via `ProcessingModule.recording_parallelism()` (default `1`; GPU stages stay `1`). **No registered module opts in today**, so every processing run is serial; a CPU-bound module would return a core count (as video import does with `core.utils.transcode_parallelism`, env `DETAIL_MAX_TRANSCODE_PARALLELISM`, in its own queue). Each recording is still its own detached subprocess + run dir; a `> 1` degree runs them through a bounded `ThreadPoolExecutor`. Because orchestration lives in the coordinator, **the whole run keeps going and completes with the launching app (GUI or CLI) closed** — the app is a thin client. The coordinator publishes `ProcessingProgress` (`recordings_done / recordings_total`, current recording, inner per-recording event, run-level `elapsed_s` for a whole-run ETA) to `runs/<id>/progress.json` (atomic, unique-temp — several recording threads emit at once); `AttachedRun.progress()` polls it and yields until terminal. **Reattach** from any process — a GUI reopen (`Experiment.attach_active_run`) or `detail process attach <run_id>`. `cancel()` writes the cancel sentinel, cancels every in-flight recording, and kills the coordinator.
- **Single-job runs reattach the same way** — `AttachedJobRun` + `Experiment.attach_job_run(kind, run_id)` / `attach_active_job_run(...)`, with typed wrappers per kind (`attach_active_analysis_run` / `_dataset_run` / `_model_run` / `_export_run`) and CLI `attach` on each group. Analysis (§16), dataset export, training (§12) and exports are **one job, not a fan-out**, so they have no coordinator — but they have the same problem: the launch-time handles (`AnalysisRun`, `DatasetRun`, …) wrap an `ExecutorRunHandle` over a `LocalExecutor` that keeps run dirs **in memory**, populated by `submit`, so **no process can stream a run it did not launch**. `AttachedJobRun` polls the run dir's `job.json` (status + progress + pid — what the worker writes) instead. The four kinds (analysis, dataset, model, export) differ only in where their record lives, so one table (`Experiment._JOB_RUN_KINDS`, `(run_dir, read_record, set_status, discover)`) drives every attach, and the caller passes a guarded write that adopts a terminal job status **only where the runner never finalized the record**. It is also these runs' **SIGKILL backstop**: a dead worker pid with an active record is failed rather than streamed forever — without which a killed analysis run reads `running` for good *and cannot be deleted*, since the delete methods refuse an active run. `Experiment.reconcile_job_runs()` runs the same check over every kind on experiment open (the GUI calls it beside `reconcile_runs()`); `reconcile_analysis_runs()` is its analysis slice, which `detail analysis list` calls. A fifth run kind inherits all of it by adding a table entry, and `tests/api/test_job_reattach.py` is data-driven over the table so the tests come with it.
- **Reconcile on open** — `Experiment.reconcile_runs()` / `_reconcile_run(record)` (beside `reconcile_imports`) folds an orphaned run to its real status, keyed on the **coordinator pid** (the liveness authority): coordinator alive → leave `running` (a client reattaches); coordinator gone → adopt each recording's terminal `job.json` result (recovering work finished after close), keep a genuinely-live detached recording worker `running` (`worker_alive`, PID-reuse guarded), fail a provably-gone one, and **fail recordings the coordinator never launched**. This is the SIGKILL backstop that guarantees a terminal status even if the coordinator is force-killed; `AttachedRun.progress()` runs the same reconcile when it observes the coordinator gone, so it never hangs.
- **Failure policy** — a per-recording failure is recorded and the batch **continues**; status is `complete` only if every recording completed, else `failed`, with the per-recording breakdown preserved.
- **Retry — gap-fill the same record, in place.** `Experiment.retry_run(run_id)` (→ `process retry`, → the run page's **Retry N incomplete recordings**) re-runs a *terminal* run's recordings that aren't `complete` — failed, cancelled, or never launched — into the **same run id and record**, so one logical result set stays one record and rolls up to `complete` once every recording is. It is **not** a pre-filled re-launch of a new run: `per_recording` already maps each recording to its own stage run id, so the record is a map with holes and the repair is to fill them. Seeding a new run is the right shape for *Refine*, where the point is a different config. Mechanics: `retry_run` reconciles first (a SIGKILLed coordinator reads `running` forever and no CLI path calls `reconcile_runs`, so a status gate alone would reject a retryable run), refuses an active run, restores the typed config (the record holds `_jsonable`'s flattened dicts, and validators read `compute.backend` / `prompt.targets` as objects), validates every incomplete recording and **names them all** if any fails (Tenet 10 — nothing mutates until all pass), resets those entries to `pending`, **clears the run-level cancel sentinel** (`runs_store.clear_cancel`; a cancelled run's marker outlives its coordinator and would stop the fresh one on its first recording — the run-level twin of `rundir.clear_cancel`), seeds `progress.json` so a reattach reads the real standing rather than the finished run's, then re-spawns the coordinator. The coordinator seeds `RunOrchestrator` with the record's `per_recording` and drives only what is pending — a fresh run passes an empty map, so every recording is pending and behaviour is unchanged. **A retried recording is re-run whole**, under a *fresh* stage run id from `Recording.run`, so nothing appends into a half-written zarr; resuming mid-recording (§7's `partial`) is not built.

Also built: `Experiment.set_metadata` (name/description; `description` is a `config.yaml` field), `Recording.video_parts`, `Recording.video_resolution`, `Recording.part_frame_counts` (per-part counts for the arena/preview sliders), `Recording.set_name`, `Recording.reorder_video_parts` (re-offsets global `t_ns`, Tenet 8; blocked once any run exists), `Recording.get_frame(part, local)` (preview decode, Tenet 11 exemption).

**Run deletion** — `Experiment.delete_run(run_id)` removes a processing run's per-recording outputs (zarr stores + `recording.yaml` entries) and the `runs/<id>` record. It refuses if a downstream run depends on it (`Recording.run_dependents(module_id, run_id)` walks detection→{tracking→embedding, keypoint_detect→keypoint_track} and each module's `aux_refs` — embedding's keypoint track run, keypoint track's tracking run — the edges derived from the module registry — §26), and refuses an active run; nothing is removed until the run is clear (Tenet 10). The GUI's run-detail view has a delete button (terminal runs only) and, now, a per-recording **Error** column.

**Per-recording error surfacing** — `RunHandle.error` / `Executor.error(job_id)` (default `None`; `LocalExecutor` reads `job.json`'s `error`). The coordinator's `RunOrchestrator` records `run.error` into `per_recording` when a run does not complete, so the worker's reason (e.g. "zero tracks produced") reaches the GUI instead of a bare "failed".

**Arena editing (§10)** — `Recording.define_arena` (multi-keyframe, time-varying), `Recording.arena_keyframes(arena_id)` (reload into the editor), `Recording.clear_arenas` (redraw; refuses if a tracking run used the arena), `Recording.arena_dependents(arena_id)`. Tracking records `arena_ids` in its `recording.yaml` entry so the dependency check is precise; the GUI arena canvas (drag/add/delete vertices, scrub + set keyframes) lives on the recording page.

**Model variant (size) axis** — see §5. Detection/embedding runs carry an optional `model_variant`; the typed wrappers (`run_detection`, `run_embedding`, `run_prompt_test`) and the CLI (`--variant`) accept it; the GUI shows a "Size / variant" combo.

**Run previews (in-GUI, no annotated video, §15)** — `Recording.detection_preview(run_id)` / `Recording.tracking_preview(run_id)` (in `api/run_preview.py`) return read-only per-frame accessors: decode the run's source frame on demand + slice that frame's boxes/masks (and, for tracking, the assignment's `object_id`) lazily — only a frame-index column is held in memory, so memory stays flat on hour-long runs. `TrackingPreview.track_presence()` gives per-track presence over the timeline. The GUI's detection/tracking result views are a frame viewer (scrub/step, overview strip, masks toggle; tracking adds per-track colors + a presence timeline) — `gui/run_preview.py`. `Recording.input_image_preview(tracking_run_id, renderer_spec)` → `InputImagePreview` extends this for the embedding **input-image scrubber** (`gui/run_preview.py` `InputImageScrubber`): per source frame, the original frame beside the constructed model input for one track. `Recording.resolve_constructor(name, tracking_run_id, track_ids)` resolves a named constructor to a concrete `renderer_spec` (computing the shared canvas from the recording's masks) for both the preview and the embedding config. `Recording.track_ids(tracking_run_id)` lists a run's tracks (for the config's "All tracks" + visible-ids); embedding `validate` resolves an `all_tracks` config flag per recording.

**Embedding result view (§14, Stage 1)** — `Recording.embedding_preview(run_id)` → `EmbeddingPreview` (`api/run_preview.py`): run `metadata()`, per-track `coverage()` (frames + parts), `pca_2d(track_ids)` over the pooled embeddings (one point per embedded frame, with locators back to the frame), and `frame_view(track, stem, frame_idx)` (the original frame + model input behind a point, via the run's stored `renderer_spec`). `pca_project_2d` is a plain numpy SVD projection — the lightweight inspection PCA, distinct from the §16 analysis stack. The GUI `EmbeddingRunPreview` (`gui/run_preview.py`) is a pyqtgraph point cloud colored by track beside the clicked point's frame and model input — three columns in one row (the per-track coverage table it once had was removed; `coverage()` is still on the API). (Stage 2, the linked slider scrub, is pending.)

**Fit/train surface (§12)** — `Experiment.generate_dataset(name, source_run_id, frame_selection, val_split, min_contour_area)` exports a YOLO dataset from a completed detection/tracking run (reusing its masks; forward-sequential read — Tenet 11) → `DatasetRun`; `Experiment.train_model(name, dataset_id, trainer_id, base_model, hyperparams, compute)` trains via a `TRAINERS` backend → `ModelRun`. Both execute on the executor; the GUI shows live progress via `gui/workers.py::FitProgressWorker`, and the runners write a **terminal** status (`complete` / `partial` / `failed`) so a crashed run is never left stuck at `running` (it would otherwise be hidden from delete). `Experiment.datasets` / `Experiment.models` (scan-discovered `DatasetRef` / `ModelRef`) + `Experiment.dataset_record` / `model_record` (full yaml, for detail views) back the experiment's **Datasets** / **Models** sections. A trained model is *applied* by a detection run carrying `weights_source="trained"` (closed-set: empty prompt, `TrainedYOLO` resolved via `TRAINED_LOADERS` by `model_family`); the GUI detection config lists built-in, experiment-trained and machine-imported models in one picker, with no source toggle — the detection validator resolves `weights_source` from the model id (§12 *Auto source resolution*).

**Deletion (cohesive strategy, gui_design §1.1)** — one model for deleting anything: confirm first; refuse (don't orphan) when a *downstream computation* loses its inputs; delete owns its outputs; an index that merely *points at* an object is pruned while a record *computed from* it blocks; delete lives on the detail page and navigates to the parent index. `Experiment.delete()` (whole experiment), `Experiment.delete_recording(recording_id)` (whole recording; prunes it from every processing-run `per_recording`, removing an emptied run record — never blocked, since a run only indexes a recording), `Experiment.delete_run(run_id)` (above), `Experiment.delete_dataset(dataset_id)` / `Experiment.delete_model(model_id)` (§12; block only on *active* work — the object's own export/training, a model still training from a dataset, or an active detection run using a model — while a completed dependent's lineage just dangles), arena clear. Per-recording run records carry `processing_run_id` (the `run_NNN` that launched them) + `created_at` so the GUI can map `run_NNN` ↔ a per-recording id and label runs descriptively in selectors and the recording "Run history".

**Experiment export / import — move a experiment between machines (§1-2, §15)** — a experiment is already self-contained with relative paths (Tenet 6), so this is selective packaging + an integrity manifest + safe placement, not path rewriting. `Experiment.export(dest, *, include_video=False, include_datasets=False, gzip=False)` packs the experiment tree into a single `tar` (`<id>.segproj.tar`) with a top-level `export.yaml` manifest (`schema_version` first, experiment id/name, `contents` flags, payload sha256). **Slim by default** — outputs + per-part timestamps + metadata + models; **source video and datasets excluded** (heavy; datasets regenerable). **`exports/` is never packed**: export run records say where *this* machine wrote copies, under root names another machine does not have, and while an export run packs, its own record there is rewritten by every report. **An earlier export archive saved inside the experiment is never packed** (recognised by name, `*.tar` / `*.tar.gz` / `*.tgz` and their `.partial`, and a finished one confirmed by its manifest, so a user's own `.tar` still travels); a symlinked folder is not followed, and `contents.symlinked_folders_left_out` names each one, which `experiment export` prints (#29). The exporter refuses if a DETAIL file holds an absolute path (Tenet 6 guard, scoped past `models/*/run/` trainer scratch). `Experiment.import_archive(archive, *, replace=False, name=None)` validates the manifest (schema_version, payload sha256 — truncation guard) and places the experiment into the managed workspace, staging to a temp dir then moving atomically (nothing placed on error — Tenet 10): **hand-off** semantics — create, or replace a same-id experiment (`replace=True`), or `name` for a separate copy; no 3-way merge (assumes sequential use). **Replace deletes what the bundle does not carry, and says so first** (#29): `storage/export.py::replace_losses` reads the manifest's `contents` flags and walks the existing experiment with the export's own filters, returning the video files (count, bytes) and datasets a bundle without them leaves behind, the export records no bundle carries, and earlier archives in the folder. A refused import carries it (`ExperimentExistsError.losses`, a `FileExistsError`), so the GUI asks without walking the experiment on its UI thread; `api.experiment.replace_warning` words it once for the GUI prompt and `experiment import --replace`, which prints it before replacing and does not ask. `storage/export.py` does the work; the CLI (`detail experiment export/import`) is the headless VM↔local path. **Both report progress** (#916) through a `progress(stage, fraction, detail)` callback: export `scanning → checking → hashing → packing`, import `reading → extracting → verifying → placing`, one fraction across the passes with each file weighed by its measured per-file and per-byte cost (`storage/export.py::_PassCost`), moving inside a large file as well as between files. The digest and the tar are two reads, so each file's `(size, mtime_ns)` is taken from the open handle when hashed and compared before and after it is packed; a mismatch raises `ExperimentChangedError` naming the file rather than writing an archive that fails its integrity check on import. The archive is written as `<dest>.partial` and renamed when complete. Import reaches 1.0 *before* it replaces a same-id experiment and reports nothing after, so a callback that raises to cancel can never land between the delete and the rename. The CLI drives `_StageProgress`. The GUI mirrors it: **Export experiment…** on the experiment overview opens the one Export dialog (`gui/export.py::open_export`) with the catalogue's `bundle` target ticked — include-video / datasets / gzip are that target's options — so the export is an export run (#354), listed under Active processes with its bar and ETA and surviving the window closing; the catalogue's bundle names the archive after the experiment (`<name>.tar`, `core.utils.safe_path_part`), and launching an export whose subject is the whole experiment goes straight to that run's page (`app.py::_on_export_launched`; other exports stay on the screen they were started from and show a notice). The run's ETA is the running target's own (`PhaseClock`), since the bar gives each target an equal share; **Import experiment** on the Home page (file pick → import → open; a same-id collision prompts to replace) runs behind `gui/busy.py::run_progress`, a bar with Cancel and the time left (`eta=True`: the import's fraction is time-weighted). (A video-excluded import keeps the `video` entries + timestamps; the `.mp4` blobs are absent → frame previews degrade to a "video not in this export" message — graceful no-video handling; the video can be put back from the original files, §3 *Putting a copied video back*.) **An export asked to include video that packs none says why** (#29): `experiment export --include-video` prints `no video was packed: …` when the manifest says `video_included: false`, and the GUI's Export dialog shows the same reasons under **Source video** while it is ticked (`Experiment.video_not_packed`, off the UI thread); `api.experiment.no_video_packed_reasons` words them once (recordings whose copied video is not on this machine, recordings whose video is used in place). Replace's video line says the video can be put back with Locate. **Referenced video is found again on import** (#916). A referenced part names its video by a `video_roots` alias, which means nothing on a machine that never registered it, so an imported experiment's video was missing even when both machines reach the same server, until a user located one file by hand. The manifest therefore carries `video_roots: {alias: <the exporting machine's directory>}` for every alias the experiment's recordings name. That is a machine path in the *archive's* manifest, never in an experiment file (Tenet 6): the manifest is not placed with the experiment, and it is a hint the importer verifies, not a path anything resolves. Import adds a `finding video` phase between `verifying` and `placing` (only when a recording references video; a few small reads per root, not weighed in the bar): for each alias this machine cannot resolve to the right videos, the manifest's directory is used when the first referenced file found there matches its recorded fingerprint, the same evidence a Locate gives (`api/recording.py::_relink_root`). Then, for what is still missing, **beside every folder that did hold video** (`storage/export.py::_beside`): the folder itself, a folder of the alias's name inside it, beside it, or around it, the name also tried without a `_N` dedup suffix (auto aliases are folder names, `roots._auto_alias`), and each folder found is looked beside in turn. A lab keeps a study as one folder per day, and the manifest's paths are no help when the two machines spell the server differently or the archive predates them; one folder this machine knows is enough. A guessed folder probes at most 3 of the root's videos, and is used only on a fingerprint match. A share that does not answer is asked once per import, not once per root (`_Reachable`). The search only reads; what it found is applied after the last report, so a cancel leaves this machine's `storage.yaml` as it was. A free alias is registered as is. An alias this machine already uses for another folder is never re-pointed (auto aliases are folder names, so two machines' `lab` can differ): the found folder gets a root of this machine's own, or reuses the deepest one already holding it, and the staged `recording.yaml`s are rewritten to it before placement. **A referenced file that is present is checked against its recorded fingerprint by the resolver** (`storage/recording.py::resolve_part_video_path`, cached per `(path, size, mtime_ns)` by `integrity.cached_fingerprint`): a different video at the reference raises `video_roots.VideoNotRecordedError`, a `MissingVideoRootError` and a `FileNotFoundError`, so the presence gate (`Recording.video_available`) and every frame reader treat it as not on this machine rather than reading it. A root alias this machine already uses for another folder holding a same-named file used to pass every reader (#29); the run gate (`video_integrity.validate_source_videos`) always checked, and keeps its own uncached comparison. `Experiment.referenced_video_missing()` lists what is still not found, a different video included; the CLI prints it after `imported:` and the GUI names those recordings in a notice once the experiment opens, each pointing at Locate. An archive made before this carries no `video_roots` and imports as before.

**Import progress (§3)** — `Recording.add_video_part(path, progress=None)` copies the part in at its original rate and resolution; a stage that wants fewer frames sets a `every_n_frames` on the run (§4). `progress(stage, fraction, detail)` reports the import stage (Reading timestamps → Copying). `DINOv3.readiness` now reports missing `torch`/`transformers` (not-ready) so the pre-run check explains what to install instead of failing deep in the worker.

**Managed workspace:** experiments live under the `default` storage location's root (§2, `storage/locations.py`). `Experiment.list_all()` lists them and `Experiment.create(name)` defaults there — so the GUI Home lists/creates experiments with no folder picker (experiments are tracked in one managed location, not imported from arbitrary folders). The full `StorageLocation` surface (storage.yaml config UI, cloud backends) is still to come.

---

### Entry notes

A user's own free-text notes on any entry in the experiment: what a run was for, what they noticed
in it. One note per entry, kept **beside the entry's record** as `notes.md` in the entry's own
directory (`storage/notes.py`):

| kind | directory |
|---|---|
| `experiment` (no id) | `<experiment>/` |
| `recording` | `recordings/<recording_id>/` |
| `run` | `runs/<run_id>/` |
| `analysis_run` | `analysis/<an_NN>/` |
| `dataset` | `datasets/<dataset_id>/` |
| `model` | `models/<model_id>/` |
| `distill` | `distill/<run_id>/` |
| `export_run` | `exports/<run_id>/` |
| `comparison` | `comparisons/<cmp_NN>/` |

The kind names are the GUI's navigation kinds and the export subject kinds (§17/#336), so a name
means one thing on every surface.

- **A file of its own, not a record field.** A processing run's coordinator rewrites `run.yaml`
  for the whole run, so a note stored there would race it; the sidecar has one writer and can be
  edited while the run is live. Written atomically (tmp + `core.atomic.replace`).
- **Plain UTF-8 text, not YAML** — so it carries no `schema_version`. DETAIL never parses it, and
  it may quote a path the user typed; in a YAML file that is a Tenet 6 violation the experiment
  export refuses to pack, in `notes.md` it is prose. The experiment export scans only
  `.yaml`/`.yml`/`.csv`.
- **No note is no file.** Saving an empty or blank note deletes `notes.md`. A note is never
  written into a directory that does not exist, so a save after the entry was deleted fails
  rather than resurrecting a directory discovery would mistake for a half-written entry.
- **It travels with the entry.** Deleting an entry removes its directory; an experiment export
  packs the tree; a catalogue export (`api/exporters.write_export`) copies the subject's note into
  the export folder as `notes.md`, listed in `export.yaml`'s `files`. Analysis snapshots (§21) do
  not copy it: a snapshot pins a *result*, and a note is commentary on the entry that can change.

**Surfaces.** API: `Experiment.notes(kind, entry_id="") -> str` and
`Experiment.set_notes(kind, entry_id, text)` (`ValueError` for an unknown kind or a missing/extra
id, `FileNotFoundError` for no such entry). CLI: `detail notes show --experiment DIR KIND [ID]`
and `detail notes set --experiment DIR KIND [ID] (--text T | --file F | --clear)`; every kind is a
choice today — a kind hidden later (§25) would be left out of `KIND`'s choices while the API
still reads and writes it (hiding governs launching). GUI: a **Notes** box (`gui/notes.py::NotesBox`) on every entry page, below the
entry's details and above its results, **autosaved** — 800 ms after typing stops, on focus-out,
and when the box is hidden (page swap, rebuild, window close). A page that rebuilds its body keeps
its one box across rebuilds (`place_notes`), so a run finishing a recording does not take the
cursor from someone typing. **The box folds from the shell, only where you are**: the box has no fold
control of its own; the shell's top-right bar carries a checkable **Notes** button beside Export… /
Version History / Info, reading `Notes ●` when the page's entry has a saved note
(`NotesBox.has_note`), bound to the page on screen (`MainWindow._bind_page_notes`: `findChild` at
`_show`, and `notes.notes_hub().created` for a box placed after the page was shown) and hidden on a
page with no notes. **Folded on every page and every view until opened there, and never
remembered** — not across pages, not across sessions (review round 4: "I don't want the notes
visibly to propagate across pages"; the app-wide `bands/notes` setting it replaces is gone). A box
is open or folded per *place* (`NotesBox.set_place`: its page, or `view:<name>` for each
full-window view of an analysis run), kept only while the page lives, so a rebuild keeps it and a
page opened again starts folded. Folded, the box is hidden outright, so a processing run's page,
which must fit 1440×810 unscrolled (`test_run_page_fits.py`), pays nothing for it.

---

### Visualization helpers

Multiple stages need to draw masks, bounding boxes, polygons, and points onto frames for inspection: prompt testing (§8), render preview (§14), tracking preview (§13), arena/ROI drawing (§10), and any GUI panel that overlays detection or tracking output. The drawing primitives shared by all of these live in **`core/visualize.py`** as pure functions over `np.ndarray` frames — no Qt (Tenet 3), no model backends (Tenet 13), no experiment state (Tenet 2). Callers compose by chaining; the GUI wraps the resulting numpy arrays in `QImage` later.

```python
# Color palette + per-key assignment
PALETTE_OKABE_ITO: tuple[tuple[int, int, int], ...]

def assign_colors(
    keys: list[str],
    *,
    seed: int | None = None,
    palette: tuple[tuple[int, int, int], ...] = PALETTE_OKABE_ITO,
) -> dict[str, tuple[int, int, int]]:
    """Assign one color per key. Deterministic given a seed (re-runs match);
    random when seed is None. Cycles when len(keys) > len(palette)."""

# Drawing primitives — each takes an RGB uint8 frame, returns a new array
def draw_mask(frame, mask, color, *, alpha=0.5) -> np.ndarray: ...
def draw_bbox(frame, bbox_xyxy_norm, color, *, label=None, thickness=2) -> np.ndarray: ...

# Thin loop helper for the recurring "draw every detection in this list" pattern
def draw_detections(frame, detections, color) -> np.ndarray: ...
```

The palette is the chart theme's identity colours (`core/chart_theme.py`): the Okabe-Ito color-blind safe set minus black and yellow, in the order the chart checks chose (blue, orange, bluish green, sky blue, vermillion, reddish purple), so a track or cluster is one colour on a frame and in the chart beside it. The default mask tint is named separately (`MASK_TINT`, amber), so it did not move when the order did. It is overridable per call for callers with strict branding/contrast requirements. The chart design as a whole — both themes, the roles, the forms — is `docs/chart_design.md`.

`draw_polygon` (for arenas/ROIs) and `draw_point` (for `PointCue` handles and vertex editing) are added when arena drawing / point-cue testing actually need them, not speculatively.

Pure functions, no class hierarchy: there is no state to share between calls. Callers chain `frame = draw_mask(frame, m, c); frame = draw_bbox(frame, b, c, label=...)` directly. The same calls work from `pipeline/prompt_test.py`, from a future tracking-preview runner, and from the GUI.

---

### CLI entry points

Each command is a thin click wrapper over the `Experiment` / `Recording` API. The CLI lives in `detail/cli.py`, with the `keypoints` and `skeleton` groups in `detail/cli_keypoints.py` and `detail/cli_skeletons.py`; the entry point `detail` is registered via `[project.scripts]` in `pyproject.toml`, and `python -m detail` works the same way.

**MVP commands (built):**

```bash
# Experiment and recording lifecycle
detail experiment create my_experiment --dir /path/to/parent
detail experiment status --experiment /path
detail recording create --experiment /path --name recording_001   # --name is taken as the id (create_recording)
detail recording add-video --experiment /path --recording recording_001 --video /path/to/video.mp4
detail recording status --experiment /path --recording recording_001

# Pipeline runs (block until complete; print ProgressEvents to stderr)
detail detection run --experiment /path --recording recording_001 --model sam3 --class mouse --prompt "mouse" [--every-n-frames 2]
detail tracking run --experiment /path --recording recording_001 --detection det_01 --tracker anchor_tracker --conf-threshold 0.1 [--arena arena_001]
detail embedding run --experiment /path --recording recording_001 --tracking track_01 --tracks 0,1   # --model defaults to dinov3_vit (vit_large)

# Credentials + model install (machine-local; experiment param optional)
detail hf-token hf_xxxxx
detail model check sam3
detail model install sam3
```

**Not built (§26):** the keypoints audit (`keypoints audit`; parked, its API and CLI removed). Built: `keypoints examples|detect|track|distill|test|export`, `dataset keypoints`, `process keypoint_detect|keypoint_track`, and on the API `Recording.run_keypoint_detect` / `run_keypoint_track` / `keypoints`, `Experiment.distill_keypoints` / `keypoint_dataset`. The full command tree is `docs/cli.md` and `detail --help`; the block above is only the core path.

**Deferred from the CLI** (the user-facing list, with reasons, is `docs/cli.md`'s "Not in the CLI"):

- **Arena drawing** — polygon input on the command line is awkward; the GUI is the right tool.
- **Render preview** — an interactive inspection workflow; on the API (`Recording.run_render_preview`) and, as the input-image scrubber, in the GUI. The detection prompt test *is* in the CLI (`detail detection prompt-test`).
- **Multi-class detection prompts via CLI flags** — v2 adds `--prompts-from-json <file>` if needed.
- **`--non-interactive`** — no API call asks for confirmation (*No `confirm_fn`* above; Tenet 17 prefers hard errors), so the flag would be a no-op.

Training and dataset generation are in the CLI: `detail dataset create|label|keypoints`, `detail train run|resume|stop|promote`, `detail distill run|…` (§12).

---

## 16. Module Families and Analysis Modules

The pipeline's extensibility (Tenet 1) is organized into **three module families, distinguished by what a module outputs**. All three plug in the same way — a contract + a registry + a generic run backbone over the executor seam (§6) — so adding one is additive (implement runner + storage, register, ship a UI), never a change to the spine.

| Family | Output | Scope | Run backbone | Examples |
|--------|--------|-------|--------------|----------|
| **Processing** (§6.5) | derived data (zarr) | per-recording | `Recording.run(module_id, config)` | detection, tracking, embedding, keypoint detect, keypoint track |
| **Fit / train** (§12) | a reusable **model** | data in (often cross-recording) | `Experiment.train_model`; `Experiment.distill` / `distill_keypoints` (label, then train, in one run) | detection-model training; detector distill; keypoint distill |
| **Analysis** (this section) | artifacts (CSVs / plots / stats) | experiment (one or many recordings) | `Experiment.run_analysis(module_id, config)` | location export (per-recording); query, query eval, temporal clustering (cohort) |

A research workflow chains them: **fit** a model → **apply** it per recording (processing) → **analyze** across a recording selection (analysis — iterating per recording, like location export, or pooling across the cohort). "Applying a fitted model" is *not* a fourth family — it is a processing module whose config names the model id. A behavior classifier or pose clusterer therefore decomposes into a fit module (produces the model) and an apply processing module (produces per-frame labels / cluster ids), with an optional analysis module summarizing the cohort.

### Analysis modules

An analysis module is **experiment-scoped**: it consumes derived data for a selection of recordings (referencing them by id) — *iterating* per recording and emitting a per-recording result (like location export) **or** *pooling* across the cohort — optionally reading grouping variables (§11), and produces free-form artifacts (figures, tables, reports) rather than a fixed schema. It reuses the executor seam (§6) and the `RunHandle` machinery; it differs from a processing module in **scope** (experiment vs recording), **input model** (multi-source vs a single `upstream`), and **output** (artifacts manifest vs a zarr schema).

```python
class AnalysisModule(ABC):
    module_id: str            # registry key, job tag, GUI nav id
    id_prefix: str            # "an" -> an_01, an_02, ...
    requires: list[str]       # artifact kinds consumed, e.g. ["embedding"] —
                              #   the multi-source analogue of ProcessingModule.upstream
    job_class: type[Job]
    phases: tuple[str, ...]   # the run's named steps, in order — required, no default:
                              #   a run page shows the phase in flight and what is ahead
    progress_counts_frames: bool   # is frames_total a count of frames? required, no
                              #   default; only then may a report divide by seconds as fps
    progress_event_class: type | None = None
    config_model: type | None = None   # the pydantic config model (§6.5)
    selects: tuple[str, str] | None = None   # (upstream module, config ref field) read
                              #   per recording, e.g. ("embedding", "embed_run_id")
    mutable: bool = False     # an authored run (§21): stays open after its compute

    def validate(self, experiment_dir, config: dict) -> dict:
        """Confirm the selected recordings + runs exist and are complete across the
        cohort; raise specific errors (Tenet 10). Returns normalized config."""

    def resolve_selection(self, experiment_dir, recording_ids, config) -> dict:
        """Concrete: drives `selects` — {<ref>s: {recording: run_id}, <ref>: the one id
        when every recording's agrees, else None} for validate() to write back."""

    def build_job(self, run_id: str, config: dict) -> Job: ...
    def runner(self, job, run_dir, progress_cb, stop_event) -> None: ...
    def run_dir_for(self, experiment_root, job): ...           # analysis/<run_id>/

    def output_manifest(self, run_id: str, config: dict) -> list[dict]:
        """Artifacts produced — [{path, kind, label}, ...] — for the result view."""
```

`register_analysis_module` wires `ANALYSIS_MODULES` + `JOB_RUNNERS` + the dynamic job-type map — a strict **subset** of `register_processing_module` (§6.5). An analysis run is a single **experiment-level** job (one `analysis/<run_id>/`, one executor job — the dataset-export/training execution shape of §12, *not* the per-recording fan-out of a processing run), so the module carries none of the recording-scoped record fields (`runs_key`, `output_path`, `run_record`); `build_job(run_id, config)` takes no `recording_id`. `module_id` is **experiment-globally unique** across all three families — the `JOB_RUNNERS` tag space is shared, so registration raises on a collision rather than silently clobbering. `Experiment.run_analysis(module_id, config)` is the generic backbone, mirroring `Experiment.generate_dataset`: resolve module → `validate` → allocate `an_NN` → `build_job` → pre-write `analysis.yaml` (status `running`, so the run is discoverable immediately) → submit one job → return a `RunHandle`.

**Recording and discovery.** Each run writes `analysis/<run_id>/` with its own `analysis.yaml` (`schema_version: 1` first, written **atomically** — sibling `*.tmp` + `core.atomic.replace` — because the runner subprocess rewrites it on completion while the GUI's progress view reads it; Tenet "atomic writes for concurrently-read experiment files"). The experiment discovers runs by **scanning `analysis/`** — the same way it discovers datasets/models — so there is no central mutable list.

**Inputs.** `requires` names the artifact kinds; the config carries `source_recording_ids` plus the run id(s) the module needs. The upstream run is resolved **per recording**, on settings rather than on the shared id (`selects` + `resolve_selection`, §6.5 "Upstream run specs"): `validate` writes back a `{recording: run_id}` map (location export's `tracking_run_ids`) and the single id only when every recording's agrees, so a selection whose recordings hold the same settings under different ids still resolves. `validate` checks every referenced run exists and is complete before any compute.

**Outputs.** No fixed schema: the module writes figures/tables/reports into its run dir and declares them via the artifacts manifest. Full lineage (recording ids + run ids + params) lives in the run's `job.json` (Tenet 9).

**Location export was the first module; four are registered now.** **Location export** (per-track position relative to a recording's ROIs/POIs, §10) loops the selected recordings and emits a **per-recording** result, which made it the simplest first consumer of this machinery. The cohort-pooling modules followed on the same contract: `query` (§20), `temporal_clustering` (§22) and `query_eval`, which pool across the selection. **Streams** (§3) have a consumer too: `query_eval` reads a behavior stream as its ground truth (`requires=["embedding", "behavior"]`, below). Grouping variables (§11) are built, and `metadata.csv` is what §24's comparisons group by.

### Location export (the first analysis module)

`LocationExportModule` (`pipeline/location_analysis.py`, `module_id="location_export"`,
`requires=["tracking"]`) turns a completed tracking run into per-frame, per-track
**spatial features relative to a recording's ROIs/POIs** — three tidy CSVs
(`tracks` / `roi_relations` / `point_relations`) keyed by `t_ns` (so they join the
streams CSV, §3). Its config is the selection, `{source_recording_ids, tracking_run_id}`,
and three measurement options, `{sample_fps, remove_thin_parts, thin_part_scale}`
(`configs/location_export.py`, the one place their bounds and defaults are written).
`validate` resolves the selection to each recording's own complete tracking run
(`selects = ("tracking", "tracking_run_id")`), writing back the `tracking_run_ids`
map the job carries; `tracking_run_id` stays set only when every recording's agrees.
It writes the resolved options back too, the job carries them by name, and the record's
`params` and each recording's `locations.yaml` name them (Tenet 9).

**The body.** `tracks` carries, after every column an older export has, the body's centre,
area, length, width and long-axis direction (`body_*`). The body is the track's mask, or
with `remove_thin_parts` the largest piece left by a morphological opening, which removes a
tail, a limb or an antenna. The option is off by default and never forced: the tool is
species-agnostic, and for some animals the thin part is the body. Three rules, each from a
failure on real recordings (detail#75):

- **The opening is sized from the animal, never from the calibration**: its diameter is
  `thin_part_scale` × √(the track's median mask area), so a wrong `pixels_per_mm` cannot change
  a body's shape. One opening per track per recording, from masks spread over the recording.
  Masks too coarse for the opening asked for are refused, not left untouched.
- **Length and width are four standard deviations** of the body's area along and across its
  principal axis (`body_axis: shape` in the manifest). A head-to-tail axis from a keypoint
  track is planned and not built; it needs a second upstream selection per recording.
- **With thin parts kept, `body_*` equals `mask_*`** in the columns they share, to the last
  digit: both come from one function.

**Sampling.** `sample_fps` keeps, for each track, the tracked frame nearest each tick of a
clock running from the recording's start; unset, every tracked frame. Nothing is interpolated,
so a gap in tracking stays a gap, and all three tables hold the same frames. Progress counts
the rows kept.

The region and point tables are measured on the whole mask whatever the options.
The runner loops the recordings,
calling the pure compute `analysis/locations.compute_location_tables` →
`storage/location_export.write_location_tables` into
`analysis/<run_id>/<recording_id>/`; `output_manifest` lists those CSVs as paths
**relative** to the run dir. Progress is per-recording; deleting a tracking run is
blocked only while an analysis run reading it is still *active* (a completed run
copied its CSVs and dangles).

**Two surfaces, one compute.** The same `compute_location_tables` backs both this
recorded run and the **synchronous** `Recording.export_locations` / CLI `locations
export` — the lightweight, scriptable per-recording path (no executor, no record).
The recorded analysis run (API `Experiment.run_analysis`, CLI `analysis run`, and the
GUI Analysis tab) is the surface that adds progress, batch, history, and
navigate-away safety. Feature details (the column schemas, the OR-merge, the
normalized/pixel/mm units — the `_mm` columns come from the recording's distance
calibration, blank when uncalibrated — the part-frame-offset lift) live with the compute
in `analysis/locations.py`.

### The analysis run page

Every analysis run, whatever its module, opens on one page class (`gui/pages.py::AnalysisRunDetailPage`)
in two modes. **The landing page**: the run's facts (module, status, recordings, error), its Notes,
the live phase rows while it runs, the stale-results notice, the snapshot banner and table (§21),
**one card per result view** — name, purpose, one status line — the fallback §23 Measures for a
module whose result holds none, and Open output folder / Cancel / Delete / Start new analysis.
**A view**, opened from its card: one row (a `QTabBar` with a tab per view — none for a one-view
module — and `Read-only · <snapshot>` in snapshot mode) and the view filling the rest of the window.
The way back is the shell's path line: with a view open it reads `<experiment> › <run_id> › <view>`
and `<run_id>` is a link to the landing page (`context_view` / `context_view_changed` /
`close_view`, drawn by `MainWindow._render_context`).
The page's one Notes box moves between the two (`_place_notes_box`): under the view row while a
view is open, back to its place on the landing page when it closes, so the shell's Notes button
opens a box the user can see. Moved, not copied: one box, one autosave — and open or folded per
place (`set_place`), so opening it in a view leaves the landing page folded unless it was opened
there.

- **The views are the module's to declare** (`gui/result_views.py`, re-exported from
  `gui/modules/registry.py`): a result widget that defines `views()` lists `ResultView(name,
  purpose, widget, status)`s; any other result widget is one view, "Results", carrying the
  `ModuleUI.purpose` line. The clustering and behavior-scan (query) workspaces
  (`ClusterWorkspace`, `_QueryWorkspace`) are `MultiViewWorkspace`s — their tab widget kept, its bar hidden, with `show_view` /
  `current_view` / `view_changed` so a hand-off inside one (AUDIT → "Open in INTERPRET") moves the
  page's row. `ModuleUI.planned_views()` names the views before the run has results, for the
  cards shown greyed ("Results appear when the run finishes") while it computes.
- **Built eagerly, shown on demand.** The result widget is built with the page, as it always was;
  the landing page only hides the view host. The open view is page state (`_open_view`), like
  snapshot mode's `_viewing`, so `_build()` keeps it.
- **The view is held to the window.** With a view open the page asks the shell's `PageScroll` for
  no height, so the view fills the viewport instead of the page scrolling to its preferred height.
  The shell's own context row (Export, Version history) stays. Export asks the view in front; the
  landing page exports the whole run.
- **The `<run_id>` link is an in-page mode switch, not a Back button** — the shell keeps no history
  (`docs/gui_design.md` §3.1). Esc is not bound to it: AUDIT uses Esc.

### The clip engine and the clip CSV (shared)


The clip engine is backend-agnostic and used by §20's Query, the `query_eval` scorer below and
§22's temporal clustering: `analysis/retrieval.py` (cosine-cost banded subsequence DTW,
`match_curve` / `query_multi`) and `analysis/clip_query.py` (trajectory load, continuity
`segment_runs`, `span_to_rows`, `band_rows`, multi-exemplar fusion, `query_tracks`). Bout
segmentation helpers live in `analysis/bout_segmentation.py`.

**The clip CSV** (`storage/clip_query_csv.py`, `schema_version` first): one tidy table,
`query_id`/`query_name` the leading columns, then `kind` (`query` | `match`), `rank`,
`recording_id`, `track_id`, `embed_run_id`, `start_t_ns`, `end_t_ns`, `start_frame`,
`end_frame`, `length`, `score`, `example_idx`. Query rows (the exemplars) carry no score/rank.
Read by name (`DictReader`) so column order is safe; the query columns are optional on read (a
pre-multi-query CSV reads as one query `q01`/`query`). Today it is an **example list**, not a
results file: a montage's clip-list export writes it (`ClusterMontage.export_clip_list`, every clip a
`query` row of one query named after the node), and `detail analysis query --from-csv` reads its
rows as example clips grouped by `query_name`. No run writes `match` rows any more; the kind stays
in the schema so an older results CSV still reads. Written atomically.

### Behavior-query eval (retrieval scored against ground truth)

The **first consumer of the behavior stream** (§3): score how well the clip-query engine
retrieves a labeled behavior, to compare embedding recipes (mask resolution, embed model,
input-image constructor) and query params. Two surfaces, one compute, both built: the
**scriptable** CLI `analysis query-eval`, and the recorded `QueryEvalModule` (progress, batch,
history) with its GUI report (`gui/modules/analysis_query_eval.py`) — see *Output + surface*
below.

**Compute (pure, no I/O — `analysis/query_eval.py`).** Temporal IoU over `t_ns` spans; a greedy
one-to-one match of the engine's rank-ordered predictions to ground truth (**state** behaviors
by IoU ≥ `iou_threshold`, **point** behaviors by a `±point_window` tolerance on the prediction
midpoint — the type comes from the stream's `behavior_types`); COCO-style all-point **AP** plus
precision/recall/best-F1. `seeded_order` gives a reproducible **held-out** exemplar draw.

**Two method families (the `method` config Literal).** The **windowed matchers** (`dtw`,
`clip_pooled_mean`/`max`, `keyframe`) produce variable-position spans scored by the temporal-IoU
match above (span-detection AP + the nulls below). The **per-frame classifier** `soft_knn` (the set
is `configs/query_eval.py::_PERFRAME_METHODS`) instead reframes retrieval as classification: score
**every** trajectory frame, rank the whole set, and integrate a threshold-free **per-frame AP**
(`analysis/query_eval.py::perframe_metrics`, the vectorized twin of `average_precision` — here the
ranked list is every frame, not a handful of spans, so the Python per-element loops would dominate a
leave-one-out run; ties are grouped at distinct scores so the AP is order-independent). A frame is
positive iff its `t_ns` lies in a scored interval (**state**) or within `±point_window_s` of a
scored point (**point**) — the same type split as the matcher path (`_interval_mask`). `soft_knn`
scores a frame by a softmax(cos/`tau`) vote over its `k` nearest **labeled** frames = the held-out
exemplar frames ∪ `n_negatives` sampled background frames (`retrieval.soft_knn_frame_scores`),
class-balancing the rare positives. **Leakage-free:** only the held-out exemplar frames are excluded
from the scored test set (so a positive never matches itself); the sampled negatives are the
neighbor index but **stay** in the test set — a background frame that is also a negative is a
legitimate true negative and self-ranks low, and excluding them let a large `n_negatives` fake
AP=1. There is no proposal cutoff, so `precision`/`recall`/`tp`/`fp`/`fn` in the row are at the
**best-F1 operating point** (not predict-all), while `ap` stays threshold-free; the per-frame path
fills **no** `overlap`/null table (those are span-only) and persists the frame score curve
(`similarity/<behavior>.npz`) plus fold-0 GT + best-F1 per-frame TP/FP runs for the figures
(visualization only — the AP is threshold-free). The old fixed-proposal-cutoff span method
(`frame_span` / `top_percent`) is **removed**.

**Every windowed-matcher headline ships with its null (`overlap_metrics`; the per-frame classifiers
skip it).** A metric with nothing to compare it
against cannot be read: a permissive matcher that covers a third of the recording scores a high
recall whatever it retrieves. So each score is computed twice — once for real, once with the
answer destroyed but the shape kept. The **shifted-time null** slides every predicted span to a
random offset, wrapping inside *that subject's own* timeline (same count, same lengths, same
coverage — only the alignment is gone) and is the null for recall; the **shuffled-rank null**
keeps the spans and outcomes and randomizes the order, and is the null for AP. `lift` is
actual − null. Both are averaged over `NULL_DRAWS` (20) draws, seeded by the run's `seed`.
Reported at each of `OVERLAP_LEVELS` (0.1 / 0.25 / 0.5) plus the configured `iou_threshold` —
module constants, deliberately not config knobs (they are for reading the shape of a result, not
for tuning). Per-level rows aggregate across folds with the same macro mean as the headline, so a
level row and the `metrics.csv` row at that level agree. **Point** behaviors get one row, at
their `point_window_s`. History: a high `recall@any` was once a tiling artifact — no better than its shifted-time null — which is exactly what this makes impossible to miss again.

**Orchestration + scripting entry (`pipeline/query_eval.py::evaluate_query`, the no-executor
entry both surfaces share).** Per behavior: keep recordings that both scored it (`is_annotated_for`) and embed the
track; `validate_comparability` over them; pool the ground-truth intervals. Then **cross-validate**
across `folds` (see below): each fold holds out its `n_exemplars`-subset and, for a **windowed
matcher**, runs `analysis/clip_query.query_tracks` **permissively** (`threshold=2.0`, high
`max_results`, so the PR curve comes from one ranked set — exemplars self-exclude) and scores the
matches vs the *remaining* ground truth (matching per recording, ranked globally); a **per-frame
method** branches to `_perframe_fold` instead (score every frame, per-frame AP, no `query_tracks`). A behavior with `≤ n_exemplars`
pooled ground truth, or none resolvable within one continuous trajectory, is skipped. Emits one
`BehaviorEval` per behavior carrying the **aggregate** metrics + the per-fold list + all folds'
per-clip detail.

**Cross-validation (`folds`).** `folds=1` (default) is a single seeded split — identical headline
to the pre-folds behavior. Otherwise the **resolvable** instances are partitioned (seeded) into
disjoint `n_exemplars`-chunks (standard k-fold): `folds=k` uses the first `k`
(capped at `⌊N_resolvable / n_exemplars⌋`, logged when it caps — no silent truncation); `folds=0`
is leave-one-out (every instance once; requires `n_exemplars=1`). Metrics aggregate **macro** —
mean of AP/precision/recall/best-F1 with their population std (`analysis/query_eval.aggregate_folds`),
so the accuracy is seed-independent with a spread. A multi-fold run caches each subject's trajectory
(`_CachingTrajectories`) so it loads once and queries `k` times rather than reloading per fold; a
single split keeps the lazy, low-memory store.

**Output + surface — two surfaces, one core** (the location-export shape). `storage/query_eval_csv.py`
writes an aggregate metrics table (**one row per `(embedding run, config, behavior)`** — the point
estimates are the fold means, plus `n_folds` + `*_std`; the cross-validation columns are optional
so a pre-folds sweep CSV still reads), a per-fold `folds.csv`, and the per-clip detail —
`predictions.csv` (each retrieved candidate: rank, score, `tp`/`fp`, the matched GT span + IoU/offset)
and `ground_truth.csv` (each interval marked `matched`/`fn`, plus the held-out exemplars) — every row
carrying a `fold` column so all folds are inspectable; plus per-fold PR points. All atomic (tmp +
`core.atomic.replace`). `evaluate_query` returns each `BehaviorEval` carrying the aggregate + per-fold
metrics + all folds' detail (from `analysis/query_eval.match_ranked_detailed`), so both surfaces emit it:
- **Scriptable CLI** — `analysis query-eval` (options single-sourced from `configs/query_eval.py`)
  is **synchronous**: it prints an AP±std line per behavior and **appends** aggregate metrics rows to
  `--out` (read + append + atomic rewrite) so a shell sweep accumulates a benchmark table; `--folds-out` /
  `--pr-out` / `--predictions-out` / `--ground-truth-out` emit the per-fold metrics, curves, and
  per-clip detail. `--behavior` is repeatable; `--search` defaults to every eligible recording.
- **Recorded `QueryEvalModule`** (§16 AnalysisModule, `requires=["embedding","behavior"]`,
  `selects=("embedding","embed_run_id")`) — a durable, progress-bearing run over `analysis/<an_NN>/`
  that writes `metrics.csv` + `folds.csv` + `overlap.csv` (long format: one row per behavior ×
  overlap level, each score beside its null — a separate file so `metrics.csv` stays the
  one-row-per-config sweep table) + `predictions.csv` + `ground_truth.csv` +
  `pr/<behavior>/fold_NN.csv`, read back through `api/run_preview.QueryEvalPreview` (the GUI seam,
  Tenet 2). Its `validate` resolves the search set to every requested recording that is annotated for a
  behavior **and** carries the embed run at the search track, so the pre-written record's
  `recording_ids` is the real universe. Progress folds the query engine's per-subject `progress_cb`
  across `(fold × subject)` within each behavior's band, so even a single-behavior single-recording run
  moves during retrieval, not only at phase boundaries.

The engine's `min_separation` (rows, not seconds) is left at its default rather than exposed.

**Matchers (`method`).** `dtw` (default), `clip_pooled_mean`, `clip_pooled_max` and `keyframe` are
`curve_fn`s: each maps `(database_norm, query_norm, band, params)` to a per-row cost curve that
`retrieval.extract_matches` turns into windows of roughly the example's length (`band` is the
warp/duration tolerance). In the eval, the per-frame classifier `soft_knn` takes the separate
per-frame path documented above, not a `curve_fn` (it scores every frame, so `band` does not
apply). It is not eval-only: the same scoring is §20 Query's one registered method, which forms
instances from the score's peaks (§20, *Reference implementation*). The old `frame_span`
fixed-proposal-cutoff span method has been removed.

⚠️ **Score direction.** `Match.score` is a **distance** everywhere downstream — `retrieval.extract_matches`,
the eval's ranked sort and `predictions.csv` all take ascending as best-first. A matcher that ranks
by a *similarity* (higher is better) must emit `1 - similarity`; inverting this silently reverses
the ranking and collapses AP with no error, and is pinned by a ranking-direction test.

---

## 17. Annotation Import/Export

Two user needs, one mechanism: (a) **import** externally-authored detections/tracks so a experiment can run our tracking/embedding/analysis on labels we did not generate, and (b) **export** our model's detections/tracks so a human can validate/correct them in the annotation tool of their choice. Human annotation itself is **out of scope** — we only read and write the interchange files. The unifying frame is a **lossless round-trip**: export a run → correct it in an external tool → re-import as a new, corrected run.

The data model makes most of this nearly free. Every downstream stage (tracking §13, embedding §14, location export §16) reads **only the per-part zarr (§9/§13) and `job.json`** — never the model. So an imported detection set is a normal detection run; an exported one is a read over a normal run. Imports do **not** introduce a new run family — they *produce a standard `det_NN` (and, for tracks, `track_NN`)* so they land in `detection_runs` / `tracking_runs` and downstream consumes them with no change. Exports are read-only and reuse the §16 "two surfaces, one compute" shape (a synchronous `Recording.export_*` twin plus a recorded `AnalysisModule`).

### Superseded in part: the export *surface* is the export backbone (#336)

This section was written before the export backbone existed, and it specifies a whole delivery
shape of its own — an `AnnotationCodec` registry, a `detail/annotations/` package, an
`AnnotationExportModule`, a `storage/annotation_export.py` manifest, and a separate `annotations`
CLI group. The backbone now owns every one of those jobs for exports: folder naming, the
`params.csv` sidecar, the `export.yaml` manifest, progress, and one CLI plus one GUI control
(`docs/export_design.md`).

So the **COCO masks half of Phase 1 is built**, as the `masks` export target on a detection-run
subject (`detail/api/export_targets.py`), honouring this section's format decisions exactly —
including the RLE passthrough below. Keypoint export targets (DeepLabCut, SLEAP, NWB, the long
CSV) are §26's, on the same registry. What remains §17's, unchanged:

- **MOTS / MOT** for tracked runs, and the `content: boxes` switch.
- **`with_frames`**, the self-contained bundle that extracts the referenced frames as images. The
  shipped target is the annotations-only form; its `file_name` values are the names a bundle
  would use, so the two agree when it arrives.
- **Import** (§17.2) in its entirety, and with it the bidirectional `AnnotationCodec` — which is
  where a codec registry earns its keep, since reading is what needs one adapter per format.
- The stated prerequisite of lifting `dataset_export._gather_parts` belongs to `with_frames`, not
  to the mask export: a detection-run subject resolves its own parts the way `_embedding_parts`
  does, and never reads a video.

### Locked decisions

- **Formats:** COCO instance-seg JSON with RLE `segmentation` (detections; lossless masks; widest tool reach) + **MOTS** (masks **and** track ids, lossless) + **MOT** (the boxes-only sibling for tracked runs). YOLO-seg (polygon → lossy, no tracks) stays the *training* export (§12), not an annotation format. CVAT-XML is reached by piggybacking on COCO/MOTS, not hand-parsed.
- **Build order:** export first (§17.1), then import (§17.2). Export is read-only — no schema synthesis, no frame-mapping — and reuses the mask-decode/frame-read core, so it is the low-risk first slice and it delivers the round-trip's second half.
- **Import requires masks in v1.** `mask_rle` is not nullable and embedding renders the mask silhouette, so an instance with a box but no mask is rejected with a specific error. Synthesizing a filled-rectangle mask from the box is a deliberate later option, not v1.
- **Export payload is selectable** (`with_frames`: a self-contained bundle that extracts the referenced frames as images, default; vs annotations-only that reference frames by number), **with a masks-vs-boxes content switch** (`content`: `masks` | `boxes`).

### Why masks round-trip for free

The detection zarr's `mask_rle` column (§9) already stores the **ASCII COCO-RLE counts string**, and `job.json`'s `mask_resolution` is `[H, W]`. A COCO/MOTS mask is therefore *literally* `{"counts": mask_rle, "size": mask_resolution}` — no decode, no re-encode, byte-identical masks on the way out, and the inverse on the way in. `core/utils.decode_mask` is not on the export path at all. The only per-row conversions are **bbox** (our `bbox` is xyxy **normalized**; COCO wants `[x, y, w, h]` **absolute**, MOT/MOTS want top-left+size absolute) and the **category↔`class_name`** map. Two recurring correctness traps to test at the codec boundary: COCO/MOTS RLE is **column-major over `[h, w]`**, and MOT/MOTS frames/ids are **1-indexed**.

### `AnnotationCodec` protocol (bidirectional) and registry

One adapter per format, self-registering (Tenet 1), discovered via `import_annotation_codecs()` — the same shape as the §3 `StreamImporter` (`STREAM_IMPORTERS`). Defined bidirectional from the start so Phase 2 import is "fill in `read`," not a new abstraction; Phase 1 implements `write` only. Concrete codecs live in a new top-level `detail/annotations/` package (mirroring `importers/`); the ABC sits in `core/protocols.py` with the other backend ABCs, the registry in `core/registries.py`.

```python
class AnnotationCodec(ABC):
    format_id: str                      # registry key, CLI/GUI value ("coco", "mots", "mot")
    display_name: str
    file_extension: str                 # primary artifact suffix (".json", ".txt")
    supported_content: set[str]         # {"masks", "boxes"} a subset — drives valid combos
    carries_tracks: bool                # MOTS/MOT True; COCO False (no native track field)

    @abstractmethod
    def write(self, annotations: "AnnotationSet", out_dir: Path,
              content: str, with_frames: bool) -> list[dict]:
        """Serialize the canonical set into out_dir; return the artifact manifest
        ([{path, kind, label}, ...], paths relative to out_dir). Raise ValueError
        on an unsupported (content, carries_tracks) combination with an actionable
        message (e.g. 'MOT carries boxes only; use content=boxes or format=mots')."""

    @abstractmethod
    def read(self, path: Path, mask_resolution: tuple[int, int]) -> "AnnotationSet":
        """Parse an interchange file into the canonical set in the recording's frame
        space. Phase 2. Raise ValueError on an instance without a mask (v1 rule)."""
```

`register_annotation_codec(format_id)` populates `ANNOTATION_CODECS`; `import_annotation_codecs()` imports the codec modules so the decorators run (exactly `import_stream_importers()`). Adding CVAT/Supervisely/YOLO-seg later is one new file — no change to the compute, the module, the API, the CLI, or the GUI.

### Canonical structure

The in-memory representation both directions share — codecs convert format↔canonical, the storage/zarr layer converts canonical↔run. This is the real extensibility seam (one structure, N formats):

```python
@dataclass
class AnnotationInstance:
    frame_idx: int                 # part-local, matches the detection zarr
    t_ns: int                      # from the recording timestamp map (§8 Tenet)
    class_name: str
    bbox: tuple[float, float, float, float]   # x1,y1,x2,y2 normalized (§9)
    confidence: float = 1.0        # imported labels default to 1.0
    mask_rle: str | None = None    # ASCII COCO-RLE counts; required on import (v1)
    object_id: int | None = None   # track id; None for a detection-only set

@dataclass
class AnnotationSet:
    mask_resolution: tuple[int, int]   # [H, W] — from job.json, the RLE size
    classes: list[str]                 # the category↔class_name map (sorted, stable)
    instances: list[AnnotationInstance]   # frames with no instance simply have none
```

`object_id is None` everywhere ⇒ a detection-only set (export → COCO, import → `det_NN`). Populated ⇒ a tracked set (export → MOTS/MOT, import → `det_NN` + `track_NN`). One code path serves both; track-carrying codecs use `object_id`, others ignore it.

### 17.1 Export (Phase 1)

**Export matrix.** The boxes-only option falls out of each family's sibling format, so four write paths over three codecs:

| Source run | `content="masks"` | `content="boxes"` |
|------------|-------------------|-------------------|
| detection  | COCO, RLE `segmentation` | COCO, `bbox` only (segmentation omitted) |
| tracking   | **MOTS** (RLE + track id) | **MOT** (bbox + track id) |

**Two surfaces, one compute** (the §16 lesson — keep CLI and GUI in lockstep by sharing the compute):

- **Synchronous** `Recording.export_annotations(run_id, out_dir, format, content="masks", with_frames=True) -> Path` — the scriptable per-recording, per-run path, mirroring `Recording.export_locations` (delegates to the read-run → canonical → `codec.write` pipeline).
- **Recorded** `AnnotationExportModule(AnalysisModule)` reached via `Experiment.run_analysis("annotation_export", config)` — batch over a recording selection, progress, history, `analysis/<an_NN>/<recording_id>/` artifacts, navigate-away safety. Config: `{source_recording_ids, source_run_id, format, content, with_frames}`. It loops the recordings like `LocationExportModule`, writing each recording's file(s) under its subdir and declaring them via `output_manifest`; `analysis.yaml` is finalized atomically with per-recording status (the §16 isolation pattern). Progress is per-recording, and per-row within a recording when `with_frames` extraction is the slow part.

  > **`requires` nuance.** `AnnotationExportModule` accepts a detection **or** a tracking run, which the single-list `requires` (§16) does not cleanly express. v1 registers **one** module with the looser requirement and a `validate` that resolves the chosen `source_run_id`'s kind from `detection_runs` / `tracking_runs`; the GUI recording-eligibility check accepts a recording that has a run of either kind. (The alternative — two modules — was rejected as duplicative.)

**Frame extraction.** When `with_frames=True`, frames are read **forward-sequentially** (Tenet 11), exactly as `pipeline/dataset_export.py` already does. The sequential reader is already shared: `pipeline/_dataset_common.sequential_frames`, which dataset export and labeling both call. `_gather_parts` is not — `dataset_export` and `labeling` each keep their own — and lifting it is the remaining prerequisite of `with_frames` (as *Superseded in part* above says), so the pipelines never diverge.

**Storage.** A `storage/annotation_export.py` owns the run-dir layout and an `annotations.yaml` manifest (`schema_version` first; full lineage so the export reproduces from itself), mirroring `storage/location_export.py`. The codec writes the format file(s); storage owns the manifest and the per-recording directory layout (single-writer, Tenet 5).

### 17.2 Import (Phase 2 — outlined, built second)

`codec.read(path, mask_resolution)` → an `AnnotationSet`; the storage layer writes it into a freshly-allocated `det_NN` via the existing `DetectionPartWriter`, and for a tracked set also writes `assignments/part_NN.zarr` (`AssignmentPartWriter`) + `tracks.zarr` (`TracksSummaryWriter`) into a `track_NN`. The run lands in `detection_runs` (+ `tracking_runs`) so it is indistinguishable downstream from an inferred run. Synchronous, mirroring `Recording.add_stream` (parse → write storage → append the record).

- **`job.json` without a fake job.** Import does not run the executor, so it hand-writes the `job.json` envelope (`status: complete`, `mask_resolution`, synthetic `parts[]` with `frames_total` = the part's frame count) plus a `source: {kind: "annotation_import", format, file}` block. `model_id` is the sentinel `"imported"`; **no `Prompt` is synthesized** (downstream reads the zarr + `mask_resolution`, never the `Prompt`). `track.yaml` records `tracker_id: "imported"`. (`weights_source="imported"` is already taken by machine-global model imports §5 — provenance goes in the `source` block, not there.)
- **Preconditions and mappings.** The recording's **video must already be imported** (the part structure + `t_ns` timestamp map must exist) — `t_ns` per instance is looked up, never invented. The codec maps external frame numbers → part-local `frame_idx`; categories → the experiment's `class_name`s (match or register). A frame with no imported instance is written as zero rows, which downstream reads as "no detections there" — the documented meaning of a sparse import.
- **v1 rejects box-only** (see Locked decisions): any instance with `mask_rle is None` fails `read` with a specific message.

### Surfaces

- **API:** `Recording.export_annotations` (P1) / `Recording.import_detections` + `import_tracks` (P2); `Experiment.run_analysis("annotation_export", …)` (P1). `Experiment.run_analysis` is already generic — no change.
- **CLI:** a new `annotations` group — `annotations export …` (synchronous, the quick path, mirroring `locations export`) and a recorded batch command in the same group; `annotations import …` (P2). The recorded path gets its **own** command rather than overloading `analysis run`, whose options are location-export-shaped (`--tracking` required); genericizing `analysis run` is a separate, later cleanup.
- **GUI:** the **Analysis tab** picks up `AnnotationExportModuleUI` automatically once it is registered in `ANALYSIS_GUI_MODULES` (the launch dialog enumerates that registry). *(Superseded for export: the built COCO masks target is reached through the export backbone's one GUI control, not a run-detail button — and there is no run-detail "Export to dataset…" to mirror; a dataset is built with **Generate dataset** on the Datasets page.)* P2 adds a recording-page **"Import annotations…"** entry mirroring "Add stream…". Long ops run off the UI thread (GUI Style). The construction smoke covers the config widget automatically (parametrized over `ANALYSIS_GUI_MODULES`); add a complete `annotation_export` run-record fixture so the run-detail test exercises the new `result_widget`.
- **Docs (same change):** `docs/cli.md` (the `annotations` group), `docs/gui.md` (Analysis tab + the run-detail button), `docs/concepts.md` (the export/import mental model, the lossless round-trip, the format matrix, and the distinction from training dataset export).


---

## 18. Export and provenance (the shared `report/` package)


An **export** is an on-demand step that reads a completed record and writes a tree of
publication tables and figures. It is re-runnable, in any format, with no recompute, and the data
artifacts re-export **byte-identically**. The module-agnostic **`report/`** package holds the
primitives every export is built from; a §24 comparison is the writer today
(`comparison export`, `Comparison.export`), and it writes the tree into the comparison's own
folder, `comparisons/<cmp_NN>/`, unless given another (`--out`). The experiment's `exports/`
folder is something else: the export backbone's run records (`exports/<run_id>/export_run.yaml`
plus the executor envelope), whose files go to a folder the user chose outside the experiment
(`docs/export_design.md`). Of §19's behavior outputs, only Query's carries a provenance block
today (§18.5).

### 18.3 The export tree

The tree holds:

- **`stats.csv`** — the tidy source of truth, one row per observation with its factors joined on
  (the shape a stats package wants); a notebook reshapes it further. Copied in when the tree is
  written somewhere other than the comparison's own folder, so the tree stays self-contained.
- **`tables/`** — `.tex` / `.md` / `.csv` grids pivoted from `stats.csv`. LaTeX is `booktabs`.
- **`ranking.csv`** — when the comparison ranks its measures, the ranking with its floor (§24).
- **`figures/`** — `.pdf` / `.svg` / `.png`, rendered deterministically (stripped dates, fixed
  `svg.hashsalt`), publication-styled (no embedded title, the chart theme's palette and type,
  editable vector text).
- **`summary.md`** — the quick-look summary.
- **`manifest.json`** — the provenance record + per-artifact SHA-256.

### 18.4 The `report/` package

`detail/report/` is **module-agnostic**: `style.py` (`FigureStyle` + rcParams, the matplotlib
adapter for `core/chart_theme.py` — light for print unless an export asks for `theme="dark"`),
`tables.py` (tidy-CSV + `pivot_to_grid` → LaTeX / Markdown / CSV, stdlib only, no pandas),
`figures.py` (styled matplotlib primitives), and `exporter.py` (what a re-export into an existing
tree may delete: only what the last export's manifest says it wrote). A module's **domain** figures
live with the module (§24's are `analysis/comparison_report.py`), never in `report/`.

### 18.5 Provenance and reproducibility

Three levels, distinct guarantees:

- **L1 re-export** (stored outputs → figures/tables): bit-identical, guaranteed (determinism +
  persisted data + write-once inputs). A determinism test pins it.
- **L2 re-score** (stored derived data → numbers): bit-identical where the computation is seeded
  and pure numpy.
- **L3 from raw video**: **captured, not automated**. `analysis/provenance.py::capture` records the
  full DAG (embedding → tracking → detection → video), the model `weights_source`, a content **hash
  of any ground-truth labels**, **video fingerprints** (size + mtime + head/tail hash, not a
  multi-GB full hash), the **code git SHA**, and the environment — in the record and in
  `manifest.json`. `drift()` names what changed (labels edited, video swapped, code moved) so a
  stale result is caught. Where it is used today: §24 comparisons capture at run time and use
  `drift()` — `comparison export` warns on it and the comparison page reads it. Of §19's
  producers, Query captures at the end of its search (`pipeline/query.py::_capture_provenance`)
  and copies the block into its artifact; temporal clustering writes an empty block, and nothing
  checks a behavior output for drift yet. The push-button rebuild driver is deferred;
  cross-hardware bit-identity of GPU neural steps is a documented limit.

## 19. Behavior Output (the shared analysis artifact)

The analytics half of the tool serves two research tasks: **quantifying a known behavior** (how
much of it, compared across groups) and **discovering unknown behavioral structure** (what differs
between groups that nobody named in advance). That is a conceptual model, not two modules — N
analysis modules map onto one task or the other, and the methods underneath are deliberately
interchangeable.

What makes them one system is that both end in the same place: **labeled instances of behavior in
time**, measured and then compared across recording metadata (§11). This section specifies that
artifact. Everything downstream — descriptive measures, frame-level export, ethograms, group
comparison — is written **once** against it, so a method added later (a supervised classifier, a
regression readout, a within-session trend analysis) inherits all of it by *emitting the artifact*,
never by editing the analytics layer.

**It is a derived, additionally-written artifact.** A producing module keeps its native output
(Query's `query.yaml`, `search/` shards and `background.npz`; temporal clustering's
`temporal_clustering.yaml` and `fit/`) exactly as it is and writes the behavior output beside it. Nothing migrates; a run predating the artifact simply lacks it, and can be backfilled
from what it does have. That is also the escape hatch if the schema changes: regenerate, don't
migrate.

### The shape is already the behavior stream's

Human-scored ground truth is imported as a `behavior`-role, `shape="interval"` stream carrying
`t_ns, end_t_ns, label, subject, start_frame, end_frame` (§3 "Behavior labels"). A predicted
behavior track is the same kind of thing, so it is the same schema — not a parallel one. Anything
that reads a behavior stream reads a behavior output, and re-importing a prediction as a stream
(the shape `behavior_csv` imports) is not a special case.

Both ends are **inclusive**: a state behavior labels a frame with `start <= t <= end`
(`analysis/behavior_labels.py::frames_to_labels`), and `end_frame` is inclusive to match.

**`instances.csv`** — the stream's six columns plus five, in this order
(`storage/behavior_output.INSTANCE_COLUMNS`):

| column | meaning |
| --- | --- |
| `recording_id` | the recording |
| `subject` | the track id within it (§13). Always written, never collapsed away — see *Subject* below. |
| `t_ns`, `end_t_ns` | the span, `int64` ns, inclusive (Tenet 8) |
| `start_frame`, `end_frame` | the same span in frames, inclusive, for readability |
| `label` | the user-facing behavior name. **Empty when unnamed.** |
| `label_path` | the label with every label above it, joined by `/` (§22: labels nest). A producer with no hierarchy writes the label itself. |
| `source_label` | the producer's own id — a query behavior's `q01`; a clustering run's label id (`L3`) or `unlabelled` (§22). Always present. |
| `instance_id` | unique per underlying bout — see *Several labels* below |
| `score` | the producer's confidence. Empty for human-authored rows. |

**`label` and `source_label` are both columns, not one.** A discovered cluster has an id before it
has a name and may never get one, so `label` is empty until a human assigns it — that is the
"some clusters labeled, some not" case handled by a column rather than a mode. Keeping the id
alongside also keeps the naming **auditable**: which clusters became "grooming" is readable from
the artifact rather than reconstructed from a separate map.

### Several labels, and the double count it invites

A frame may carry several labels: intervals are rows, overlap is permitted (§3 declares
overlapping intervals of *different* labels first-class), so a frame inside two intervals has two
labels with no special case in the schema.

A **cluster** may also carry several labels — the naming map is `{source_label: [label, ...]}`, so
cluster 7 can be both "rearing" and "near wall". It emits **one instance row per label** over the
same span rather than one row with a multi-valued cell: the schema stays flat, stream
compatibility holds, and a measure over "rearing" picks up cluster 7's bouts with no change. The
schema and the writer allow it; neither producer emits it today — a query behavior has one name,
and a clustering node carries one label, nesting instead through `label_path` (§22).

This is exactly why `instance_id` exists. Rows are no longer one-per-bout, so summing durations
over a two-label cluster **double counts** — and nothing about the arithmetic looks wrong when it
does. With `instance_id`, "how many bouts" and "how many label assignments" are distinct questions
with distinct answers, and a measure that must not double count says so by counting distinct
`instance_id`.

### Frame-level methods, and what "interval" costs them

A per-frame **categorical** labeling — which is what frame-level clustering, a temporal decode, or
a supervised frame classifier produces — is stored as intervals **with no loss**. Run-length
encoding a label sequence into maximal runs is exact, and the tool already does it:
`segmentation.labels_to_bout_rows` is what lets a frame-labeling method reach the same assignment
output as the segment-then-cluster path, which is precisely why an eval can score either
without knowing which produced it. It splits at continuity-run edges, so a label continuing across
a tracking gap yields two rows rather than one claiming the animal did the same thing throughout
the gap. Re-expanding to frames at export (`behavior_labels.frames_to_labels`) recovers the
original sequence on the same `t_ns` grid, which `coverage.csv` records.

So "the artifact stores intervals" is not a restriction on frame-level methods. The distinction
that matters is **categorical vs continuous**, not frame vs interval: a per-frame *label* is an
interval table, a per-frame *value* is not (see *Deliberately out of the MVP*).

One consequence worth watching rather than preventing: a noisy frame labeling that alternates
rapidly produces one interval per frame, and the row count is the tell. The ratio of instances to `n_observed` is
therefore a first-class diagnostic on the artifact, not a footnote: a producer whose instances
approach its observed rows is emitting frame noise, however good its agreement metric looks.

### The frame view — a materialized expansion, not a second source of truth

Intervals are the **run-length encoding** of a per-frame array, and the two hold the same
information. Intervals are what gets stored, for two reasons that are not about size:

- **Instance identity.** From a per-frame boolean you recover bouts from rising edges — unless two
  instances of the same label abut, and then they silently merge and the bout count is wrong. Bout
  count is a headline measure, and `instance_id` is also what keeps a twice-named cluster from
  double-counting.
- **Per-instance score.** A windowed matcher produces one confidence per span; smearing it across
  frames means nothing. And the audit attaches a human decision to an *instance* — there is nothing
  to attach it to in a boolean array.

But most consumers want frames, so the expansion is **materialized as a first-class derived view**
rather than re-derived by each of them. One zarr group per `(recording, subject)`:

```
frames/<recording_id>__<subject>.zarr
  t_ns           int64  (N,)   the frame time axis
  observed       bool   (N,)   was this frame inside an analyzed span — coverage, per frame
  instance_id    str    (B,)   the unit's bouts' instance_ids; ordinal i refers to instance_id[i - 1]
  <source_label> int32  (N,)   0 = not active, else the bout's 1-based ordinal
```

(`storage/frame_view.py`.) The `instance_id` strings are content hashes, so they are stored once
per bout in their own array and each label array holds a small ordinal into it.

- **Keyed by `source_label`, never by the user's name.** Renaming a cluster then rewrites nothing;
  the naming map applies at read time, and a cluster carrying two names is one array read twice with
  one ordinal, so the double count is prevented by construction rather than by care.
- **Each view records what it was built from**: group attribute `built_from`, a digest of the
  unit's bouts (`instance_id`, `source_label`, span), its coverage row and the producer's gap. A
  view is current exactly when its digest matches what `instances.csv` and `coverage.csv` now say
  — a fact about the view, not about which operation last ran (naming a *branch* above the cut
  fuses and relabels the bouts beneath it; naming at the cut moves none). A view without the
  attribute predates it and is treated as out of date.
- **An authored edit writes no view** (§21). It writes the three files and the instance features,
  and if any view no longer matches it leaves `frames/PENDING`. While that marker exists every
  reader (`read_frame_view`, `discover_frame_views`, `snapshots.pin`) raises
  `FrameViewsPendingError` rather than read a stale view; `AuthoredRun.bring_frame_views_current`
  rebuilds exactly the mismatched units and removes it, and every entry point that reads views —
  measures, `pin` (and so finish, export, comparison), the measurements panel — calls it first,
  reporting through `progress`. The marker goes down **before** `instances.csv` moves, so a kill
  mid-edit leaves a refusal, never a silent stale view. The compute's first write and a rebuild
  write every view immediately, so an edit commits quickly and the views are paid once, by the
  first reader.
- **A bout ordinal rather than a boolean**, at identical cost: two abutting instances stay two, and
  through the `instance_id` array an audit decision has something to attach to.
- **Intervals stay authoritative.** If the two ever disagree, regenerate — the same rule §19 already
  gives for the artifact as a whole. This is a materialized view, not a second truth.
- **Self-contained.** `t_ns` is stored rather than referenced from the recording's timestamp map,
  which costs the largest single array and makes the artifact readable on its own — these are files
  people hand to collaborators.

**Sizing, so the trade is explicit.** At 60 Hz, one row per frame:

| | 2.5 h | 100 h |
| --- | --- | --- |
| rows per recording-subject | 540 k | 21.6 M |
| one label array (int32) | 2.2 MB | 86 MB |
| `t_ns` axis (int64) | 4.3 MB | 173 MB |

The view holds one label array per `source_label`: a discovery run stores one per label (plus
`unlabelled`, §22), not one per cluster, and a Query run one per behavior. Against interval storage of
roughly 180 KB per behavior over 100 hours this is about three orders of magnitude more for the
same information, which is the honest cost of not making every consumer write its own expansion.

**What is still not stored is a per-frame *continuous value*** — a kinematic readout, a regressor's
output, a similarity curve. Those are not intervals at all and cannot be expanded from one; see
*Deliberately out of the MVP*.

### `instance_features.csv` — what we *measured about* an instance

`instances.csv` says what an instance **is** — where in time, which behavior, whose. This says
what was measured about it. The split is deliberate: the first is a contract other producers emit
and every reader is written against, while this list is expected to grow (position now; the
per-cluster descriptors — mask speed, area, aspect, displacement — next), and growing it must not
change the artifact's shape or invalidate one already on disk.

| column | meaning |
| --- | --- |
| `instance_id` | the join key — §19's content hash of the bout's identity |
| `cx_norm`, `cy_norm` | where it happened: the **mean** position over the instance's frames, normalized to the field of view |
| `is_example` | `true` when one of the instance's own behavior's example or non-example clips overlaps it (a query run's; absent for a producer given no clips). The audit never draws or counts such an instance (§20 *The audit*) |

Joined on `instance_id` rather than by row position, because both files are rewritten on every
reprojection (a granularity drag, a cutoff commit) and only the hash is stable across one.

**The mean over the bout, not its first frame.** A bout is an extent; the question a spatial map
answers is *where does this happen*, not *where does it start*. The per-frame path stays in §14's
`kinematics/` for anything that wants it.

**Position comes from §14's `kinematics/`, matched by time.** Never reconstructed from the
embedding zarr's `det_row`: that back-reference is valid only within its own part and names the
frame's *representative* detection rather than the OR-merged subject, so a position derived from
it is wrong across a multi-part video and biased toward the largest mask fragment — plausibly,
with nothing failing.

**Absent and empty are different states.** No file means the artifact predates features, and
`recording backfill-kinematics` is what fixes it. An empty file means features were computed and
there was nothing to say. A reader distinguishes them; only the first is actionable.

### `coverage.csv` — absence is not zero

The single most dangerous failure in a group comparison is a missing recording read as a zero. If
"rearing" has no rows in recording X, that is either *we looked and it never happened* (a real 0,
which belongs in the group mean) or *X was not in this analysis run* (missing, which must not be).
An interval table alone cannot tell them apart, and the resulting bar chart is clean, confident,
and wrong.

`coverage.csv` is one row per analyzed `(recording_id, subject)`:

| column | meaning |
| --- | --- |
| `recording_id`, `subject` | the analyzed unit |
| `start_t_ns`, `end_t_ns` | the span the analysis covered, **half-open** — see below |
| `observed_ns` | time **actually observed** — tracking gaps removed, cadence accounted for |
| `n_observed` | rows actually present over that span |

**`end_t_ns` is the span end, not the last frame's timestamp**, and getting this wrong fails in a
way that reads as a different problem. A frame covers its sampling interval, so a fully observed
10-frame recording at 1 kHz has `observed_ns = 10000` while its last frame sits at 9000. A producer
writing that last timestamp as `end_t_ns` therefore claims more observed time than its own span is
long, and the writer refuses it — with a message about a denominator that deflates every rate,
which is baffling when the recording is perfectly correct. The same rule holds in
`frame_view.observed_mask` (a run ends at `last_row + period`) and in `observed_ns_in` (a window is
`[start, end)`); §23 carries the reasoning and extends it to instance durations.

**`observed_ns` is the denominator, and it ships with every rate.** Video duration is not
recording duration is not tracked time: a 20-minute and a 60-minute recording compared on raw
counts is meaningless, and comparing them on rate is *still* wrong if one lost 30% of its frames
to tracking. This is the §16 "a headline metric ships with its floor" discipline applied to
frequency — the denominator travels with the numerator or the number cannot be read.

Coverage also carries the **vocabulary** half of the same distinction, at run level: a label in
the manifest's `vocabulary` with no rows inside coverage is a **real zero**. This mirrors the
ground-truth stream's `covered_behaviors`, which exists so that a miss is a true false negative
rather than an unscored behavior (§3).

### `behavior_output.yaml` — what varies between producers

Everything that differs between a query run, a clustering run, and an imported ethogram lives in a
manifest, **not** in a subclass. The row schema is fixed; the producers vary in metadata. A new
method registers and writes; it does not inherit.

```yaml
schema_version: 1
producer: {module_id: query, run_id: an_07}
origin: predicted            # predicted | ground_truth
label_space: pooled          # pooled | per_recording
overlap: allowed             # allowed | exclusive
vocabulary: [rearing]
names: {q01: [rearing]}      # source_label -> one or more labels; partial by design
behavior_types: {rearing: state}
score: {direction: higher_is_better, meaning: "confidence that the stretch is …"}
method: {method_id: soft_knn, params: {...}, cutoffs: {q01: 0.42}, min_match_s: 0.0}
provenance: {...}            # the §18.5 capture block — upstream DAG, weights, code SHA, env
audited: false               # derived: whether an audit.csv sits beside the artifact
```

That is Query's (`pipeline/query.py`), in the order it is written; `audited` is filled in by the
writer, never taken from the producer. A temporal clustering run's (`pipeline/temporal_clustering.py`)
adds `label_paths` — parallel to `names`, one path per label — gives its `unlabelled` source
label an entry with no name in both (`unlabelled: []`: the fact §23's `unnamed_labels` condition
reads, so `unnamed_time` is measured on every clustering run), writes no `score` (a cluster
assignment has no confidence), declares `overlap: exclusive`, and its `method` carries
`method_id`, `params`, `granularity` and the method's own fit-summary counts.

**`score.direction` is not optional.** A producer's native score may be a confidence (Query's,
higher is better) or a **distance** (query eval's `Match.score`, ascending best-first), and §16's
query eval already carries a warning that inverting it silently reverses
the ranking and collapses AP with no error. Declaring the direction in the artifact stops the same
bug recurring one layer up, where a comparison would rank groups backwards just as quietly.

**`label_space` decides whether labels may be pooled across recordings.** A clustering that pools
its selected recordings into one vocabulary is the normal case; per-recording label spaces exist
only for frame rules such as `tw_finch`, which label each recording independently — which gives
a recurring behavior a different id each time (§22). The producer declares it and the artifact
carries it forward, because a comparison that pools a
per-recording label space across animals is nonsense that renders as a tidy chart. The comparison
layer refuses to mix the two rather than footnote it.

### Where it lives, and why not in the recording

The artifact is written under `analysis/<an_NN>/`, per Tenet 5 — a stage writes only its own
directory. It is **not** auto-written into the recording's `streams/`. Re-ingesting a prediction as
a stream stays a deliberate user action (export the output, then import it as a stream), which keeps
the prediction / ground-truth boundary **structural** rather than a flag a reader has to remember
to check. This matters more than tidiness: the leakage rule in `CLAUDE.md` — labels score a method
and never build one — is only enforceable while a predicted track cannot be picked up by something
that expects ground truth. `resolve_behavior_stream` and `is_annotated_for` must refuse an imported
prediction on `origin`, not on hope.

Discovery is by **scanning** `analysis/*/behavior_output.yaml`, the way runs, datasets, and
models are already discovered — no central mutable list.

**One writer, many producers.** The write path is a single shared function in `storage/`; a module
hands it the pieces. A module that emits the artifact inherits measures, export, and comparison
without touching them, and a module that hand-rolls its own writer is the bug (the "fix the
mechanism, not the site" rule — where N siblings share a shape, drive them from one place).

### Written once, at finalization — which is what makes parallelism a non-question

The artifact is **derived from a run's native output and written once, when the run completes**.
Producing modules keep whatever parallel execution they already have — detection's per-part
workers, embedding's per-(track, part) units, the numba-parallel DTW block in a temporal clustering fit, a
query's per-recording loop — and none of them touch the artifact. The coordinator projects it at
the end, from outputs those workers already wrote. This is the detection shape exactly: parallel
workers write their own zarrs, the coordinator finalizes.

Two properties follow, and both are worth having deliberately:

- **No concurrent-writer story is needed.** Nothing appends to `instances.csv` from N processes, so
  there is no shard-merge, no append-atomicity question, and no interleaving to get wrong. The
  write is a single atomic replace (the concurrently-read-file rule), like every other analysis
  output.
- **A partial run has no behavior output.** An interrupted run cannot honestly state coverage over
  recordings it did not finish, and the coverage table is the thing that separates a real zero from
  a missing one. So the artifact appears at `complete` (or `open`, for an authored run, §21), never
  before — `partial` means the compute was interrupted, not that the artifact is half-true. A
  stopped or failed analysis run is re-run as a new run; nothing resumes one mid-way.

**It is also cheap to regenerate**, because it is a projection of native output plus the authored
state (the cutoff, the naming map). That is what lets an authored run rewrite it when a user names
a cluster or drags a cutoff, with no recompute — renaming a cluster must never re-run a fit. Where
a producer *is* parallel per recording and a future run wants incremental output, the shard-then-
merge version stays available; it is simply not needed for anything in either task today.

### Reproducibility — these numbers go in papers

A behavior output is the artifact a figure is built from, so it carries what is needed to recreate
it, and to detect when it can no longer be trusted. This is Tenet 9 at the point where the numbers
leave the tool.

The machinery exists and is reused, not rebuilt: `analysis/provenance.py::capture` already records
the full upstream DAG per artifact (embedding → tracking → detection → video), the model
`weights_source`, a content hash of any ground-truth labels, video fingerprints, the code git SHA,
and the environment; `drift()` names what changed since (§18.5). A behavior
output carries the same provenance block, and any surface that renders its numbers **warns on
drift** the way a comparison export does. **Partly built:** Query captures the block at the end of
its search and copies it into the artifact; temporal clustering writes an empty one; and no
surface checks a behavior output for drift yet — `drift()` is reached only through comparisons.

Three levels, with the same distinct guarantees §18.5 draws:

- **Re-derive the artifact** from the producing run's stored output plus its authored state —
  deterministic and cheap, since that is exactly what finalization does.
- **Re-run the analysis** from stored embeddings and the recorded config — deterministic where the
  method is (the resolved config is recorded, not the requested one, per Tenet 9's resolved-value
  rule).
- **Rebuild from raw video** — captured, not automated, with cross-hardware bit-identity of GPU
  neural steps a documented limit.

The manifest therefore records the **resolved** configuration, never the requested one. A cutoff
the user dragged, a naming map they authored, a default the module filled in — all are written
back as the values that produced these rows. A record that says "threshold: null, resolved at run
time" is not reproducible however complete it looks.

### The reliability audit — optional, and derived from a stored sample

Auditing is an **optional step**. A user may set a cutoff and take the instances without labeling
any of them, and the artifact is complete and valid without an `audit.csv`. What is not optional is
that the *absence* of an audit reads as absence: a surface showing instances from an unaudited
artifact says it has no reliability estimate, and never substitutes a default, a proxy, or silence.
An unaudited count presented identically to an audited one is the bug — the whole point of the
audit is that a number without one cannot be read.

**A temporal clustering run's precision comes from `clip_review.csv`**, a census of each cluster:
every clip of it reviewed and recorded yes or no (§22, *Precision from reviewing every clip*). Its `audit.csv`, from the one-by-one review
that came before, is never rewritten or merged into the new figure; it stays readable and is shown
as *Earlier one-by-one audit*. Everything below about `audit.csv` still holds for a query run and
for the earlier audit.

When it is done, the estimate is **derived from a stored sample of human decisions**, never stored
as a number. `audit.csv` holds `{instance_id, score, human_label, drawn_at, answer}` — the
instances a human reviewed and what they decided. `answer` arrived with nested labels (below); a
log written before it is migrated in place on its next append (`storage/append_log.py`), and its
rows read an empty answer.

- **A stored scalar goes stale silently.** Precision at cutoff 0.15 says nothing about precision at
  0.25, but a stored `precision: 0.88` renders identically after the cutoff moves. Derived from the
  sample, precision recomputes over whatever audited instances remain above the current cutoff,
  with an honest `n`, and cannot drift from the thing it describes. Same discipline as the derived
  `params` rule (Tenet 9): compute from the store, never maintain a second copy of a fact.
- **Sample stratified by score band, not top-N.** A user reviewing the highest-ranked matches is
  sampling the easiest ones, and precision comes out optimistic. Drawing *k* per score band instead
  estimates precision **as a function of score**, which makes precision at any cutoff an integral
  over the bands above it — so the figure moves live as the cutoff is dragged, and it takes fewer
  labels to get there.
- **The audit sample is disjoint from anything that fed the method.** Where confirmed matches
  become new exemplars, those labels *built* the method and cannot also score it — the leakage rule
  again. An instance that was ever an exemplar is ineligible for the audit, and the two label sets
  are stored separately so the constraint is checkable rather than remembered.
- **Re-auditing appends.** A user who reviews a larger sample later adds rows; nothing is
  recomputed by hand and no earlier decision is discarded. `drawn_at` records when each decision
  was made, so a sample drawn under a different cutoff is still readable for the population it
  belongs to.
- **Every estimate ships with its coverage, and is withheld below a floor.** A decision is keyed by
  `instance_id`, a content hash of the bout *as produced*, so re-cutting the vocabulary changes the
  ids and the old decisions stop applying. Intersecting them with the current instances and
  reporting whatever survives is not merely lossy: whether an instance survives is **not
  independent of what the audit measures** — clusters merge because their contents resemble each
  other, which correlates with whether they were judged right — so the survivors are a biased subset
  and the estimate flatters the method. It is also the strategy information retrieval abandoned;
  bpref, infAP, statAP and condensed-list scoring all exist to replace "unjudged = ignore it"
  (Zobel, SIGIR 1998). So `decision_coverage` reports `n_decisions` / `n_applicable` / `coverage`
  beside every number, and `precision` comes back `None` with a `reason` below
  `MIN_COVERAGE`. **That floor is a policy choice, not a measurement** — the literature supports
  having one and gives no principled value — and the way back from low coverage is to audit more,
  which `draw_audit_sample`'s `exclude_instance_ids` already supports. An audit read from a
  **snapshot** (§21) is always fully covered, because the decisions were copied beside the artifact
  they were made against.
- **A nested label is answered at the deepest level that is true.** A clustering run's labels sit
  on tree nodes (§22), so an instance labelled `rearing/supported` is also a claim of `rearing`.
  A yes/no to the deepest label cannot score the parent — "not supported" says nothing about
  whether it is still rearing — so the reviewer picks the level that is true ("rearing, not
  supported") or "none of these", and `answer` records it: the path, or the bare separator for
  none. A check of one label asks a narrower question and records "not *that label*" instead (see
  *A check is one label* below). `behavior_audit.verdict` reads one answer at any level: yes when the answer is at or under
  the label asked about, no when it is outside it, dropped when unclear. A parent's precision is
  therefore exact, over every instance under it, from the same judgments that score its
  sub-labels. `human_label` is still written (whether the answer is the instance's own label), so
  every reader that knows nothing of paths reads what it always did, and a flat label keeps the
  plain yes/no with an empty `answer`. A row from before `answer` existed is a yes/no to its own
  label: its yes counts for the ancestors too, its no only for its own label.
- **The unlabelled remainder is never audited.** A clustering run reports time under no label as
  one `unlabelled` bucket, which makes no claim a reviewer could confirm, so the sample is drawn
  from labelled instances only (`AuthoredRun.audit_sample`).
- **An edited label says what it set aside.** Its headline counts judgments since its membership
  last changed, and `precision_at_cutoff` returns `n_before`, the judgments that filter dropped, so
  a surface reads "0 judged since, 78 before" rather than "not audited yet" beside a label that was.
- **Cluster intervals, never verdicts: how pure is each cluster a label was given to?** A user
  labels the clusters that looked pure, and the audit reports each one's precision with its 95%
  confidence interval. Whether that is good enough is the reader's call, not the tool's: an earlier
  version said *holds* / *fails* / *undecided* against a target, and that is a judgment the person
  who knows the behavior makes. Each judged instance belongs to the labelled cluster holding most
  of its time (`NodeView.cluster_of_spans`, from the node cache — an instance reports a label, and
  the clusters of one label fuse). `audit_by_cluster` returns each with its size, its share of the
  label and what its clips were answered as. Without a node cache the audit draws by label.
- **The goal is an interval width.** A surface aims at a 95% confidence interval no wider than a
  width the reader sets (`DEFAULT_GOAL_WIDTH` 0.20, ±0.10) and says when it is reached — a fact
  about the sample, not the label. ±0.10 takes 30–60 clips near a precision of 0.9; ±0.05 takes
  several hundred. There is no target precision: a reference line was drawn for a while, and
  went with the chart it was drawn on.
- **A check is one label, with one fixed set of answers.** The question is "What is this?" and the
  answers come from the **scope**, never from a clip (`answer_options`): every label in the scope's
  branch (the scope reading "*scope*, other" when it has sub-labels), "not *scope*", and Unclear —
  the same keys on every clip, which also stops telling the reviewer which label a clip was drawn
  from. Answers are label paths, so naming a sibling sub-label scores a miss for the clip's own
  label and a hit for the parent. **Only the scope is asked about.** A check of *climb front wall*
  once also offered "*climb and rear*, not *climb front wall*" and "not *climb and rear*": two ways
  to say no that differed only about a label the reviewer had not chosen to check. "not *scope*"
  is recorded as `ruled_out(scope)` — the separator then the path, the form `NOTHING_CELL` takes
  with nothing after it — and `verdict` reads it as no for the scope and everything under it and
  as no judgment for the labels above. A yes still counts upward. A root-level scope's "not" is
  `NOTHING_CELL`, and a root-level label with no sub-labels keeps the flat yes/no.
- **A round is 25 clips, of a label or of one cluster.** `audit_sample(path=)` draws a label's
  round across its clusters in proportion to their size (`draw_sample`, Sainte-Laguë);
  `audit_sample(path=, node=)` draws from one cluster only, which is how a reviewer looks harder at
  a cluster that looks doubtful. A round is a sitting, never a count sized to close a gap: "audit 2
  more" is not a sitting. The CLI's `--goal-width` note says roughly where another round would put
  the interval, `projected_interval(estimate, 25)` — Wilson at the current share over `n + 25`
  (the effective n for a label) — never reported as a figure. The GUI shows no projection.
- **What a figure shows is one rule, in one place** (`storage/audit_figures.py`, read by AUDIT's
  table, INTERPRET's card and `analysis audit`). Below `MIN_N_FOR_ESTIMATE` (8) audited clips the
  estimate is withheld and the interval still shown — it is the honest statement of how little a
  few clips say. With nothing left to audit (`remaining` 0, carried on the estimate by
  `audit_precision(path=)` and `audit_by_cluster`) the share is exact: the CI reads *all audited*
  and the width 0.00. A refused estimate has no width either.
- **A cluster's "no" clips can be watched, never re-answered.** Re-judging only the rejections can
  move a precision in one direction, the same reason a query's review offers no yes/no scope. A
  misclick is changed in the round it was made, from the last answers.
- **A label's precision is a stratified estimate**, `Σ w_c·p_c` with `w_c` the cluster's share of
  the label's instances, because a cluster audited on its own holds more judged clips than its
  share and a pooled figure would tilt toward it. The interval is Wilson's on the effective n, each stratum's
  variance taken at `(x+2)/(n+4)` so a 10-of-10 cluster does not claim zero spread; clusters not
  judged yet are left out and reported as `unjudged_share` (the figure is provisional while it is
  above zero). One stratum gives exactly the plain estimate.
- **A label edited after it was audited reports from fresh judgments.** Choosing clusters by how
  they scored and then quoting the label's precision from the same clips raises the figure for the
  wrong reason — the labels-score-never-build rule, applied to the loop this creates. State keeps
  `membership_changed_at` per label, stamped whenever an edit changes which clusters a label (or a
  label under it) holds; a rename stamps nothing, and neither does creating a label, which has no
  audit to set aside. An undo (`restore_labels` with the *before* side of `label_maps()`) puts the
  stamp back with the clusters, so undoing a misclick does not set aside the review made before
  it. The headline counts only judgments made since, and the clips judged before
  can be drawn again — set aside but undrawable, a fully audited label could never be audited
  further. A cluster's interval keeps all of its judgments, because the cluster did not change.
- **A label a difference screen chose carries the screen.** State keeps `label_origins` per
  labelled node (§22 *The difference screen*): the saved `cmp_NN`, the kept cluster, the node's
  relation to it, the column contrasted, the strictness the screen was read at (`alpha`, blank
  for its own floor) and the kept cluster's `family_p` / `own_p` / `q` when the label was
  written. It goes with the node's own label and is restored with
  the other maps (`label_maps()` is four maps; `restore_labels` takes four), so an undo neither
  strands an origin on an unlabelled node nor forgets one on redo.
- **A rename or merge keeps every answer.** Answers are label paths, so renaming *climb* left
  every "climb › corner" answer naming a path that no longer existed, and each yes read as no.
  `LabelOperations._carry_answers` appends a superseding row per decision whose path moved
  (`path_moves`, `carried_path`), with the original `drawn_at`; `_carry_review` does the same
  for `clip_review.csv`. A path moves only through clusters that **kept their label** (a rename,
  a label added above) or through a merge — a cluster given to another label does not take the
  old label's answers with it.
- **No recall.** "Relative recall over the ranked list" — the share of confirmed clips a cutoff
  keeps — was built and never shown, and it is retired: it divided by the judged clips without
  weighting them by the bins they were drawn from, the flaw the precision estimate had, and it says
  nothing about behavior the search never surfaced. A recall figure needs its own design.

### Subject

`subject` is a first-class dimension in every table from the start and is never collapsed away,
even while every current recording has exactly one tracked animal. It already exists in the
underlying zarr schemas (§13) and is written-but-nullable in the behavior stream (§3), so this
costs nothing now.

What multi-animal recordings will need later is **per-subject metadata** — which animal is track 0
in this recording. `metadata.csv` is recording-level today (§11); because the measures table
carries `subject` as a column, that future join goes from keyed on `recording` to keyed on
`(recording, subject)` with no downstream schema change. Designed to, not built.

### The two layers above it (§23, §24 — both built)

- **Measures** (§23) — pure functions over the artifact producing a tidy table, one row per
  `(recording, subject, label, measure)` with its denominator: instance count, total time, time
  fraction, rate, bout-duration mean/median/sd, latency to first, inter-bout interval, and
  label→label transition counts. A registry, so a new measure is additive.
- **Comparison** (§24) — a recording selection → group by keys → aggregate measures → tables,
  figures, raw export. **The grouping key is uniform**: any `metadata.csv` column, plus `subject`,
  plus `window_id` (a time window, §23). That last one is what makes "rearing rate over the course
  of a session" the same operation as "animal 1 vs animal 2" and "drug A vs drug B within one
  animal" — a within-session trend is a grouping axis, not a module. Scope is **descriptive**: tables, effect sizes, figures,
  and a tidy raw export for inferential work elsewhere. The one inference it makes is the floor a
  ranking ships with — a permutation p, with a family-wise p and a Benjamini–Hochberg q beside it
  (§24) — and it fits no models.
- The export tree is §18.3's, written by the module-agnostic `report/` package — `stats.csv` as
  the tidy source of truth (one row per observation with its factors joined on, which is what a
  stats package wants), `tables/`, `figures/`, `summary.md`, `manifest.json`.

**Naming.** "Comparison" means this — the scientific comparison across recording metadata.

### Deliberately out of the MVP

- **A stored per-frame *continuous* series.** A per-frame **categorical** labeling round-trips
  through intervals with no loss and is materialized as the frame view above, so frame-level
  clustering is fully covered. What is deferred is a per-frame **value** — a kinematic readout, a
  regressor's output, a similarity curve — which is not an interval at all, cannot be expanded from
  one, and needs a genuinely second shape. No module in either task produces one yet; the
  location-export CSVs (§16) are the nearest existing thing and stay in their own format until one
  does.
- **Comparison across producing runs.** Cluster 7 in fit A is unrelated to cluster 7 in fit B, so a
  comparison is scoped to one producing run. The bridge, when it is wanted, is the **user's names**
  — a name can mean the same thing across runs in a way an id never can.
- **Per-subject metadata**, above.


## 20. Query (measuring a known behavior across a cohort)

**Query** is the §19 Task-1 module: a user points at a few example clips of a behavior they can
already name, the tool finds every stretch that resembles them across the whole cohort, and the
user calibrates a confidence cutoff by looking at what each setting returns. Its output is the
§19 behavior artifact — instances in time — which the measures and comparison layers then take
across recording metadata (§11).

**It works without ground truth, and that is the design constraint.** A user has no behavior
stream, no labeled frames, and no way to score one method against another on their own data. The
only labels that ever exist in a Query run are the example clips the user marked and — optionally
— the sample they review afterwards. Everything else follows from that:

- **Precision is auditable; recall is not.** You can review what was returned. You cannot know what
  was missed without labeling everything, which is the work the tool exists to avoid. Prioritizing
  precision is therefore not a preference, it is the only direction with a measurable answer, and
  it is why this module is built around calibrate-then-audit rather than around a score.
- **The user cannot evaluate the method.** So the method is not a casual dropdown: it lives in
  advanced settings, and each method's documentation states what is and is not known about it.
- **The research eval is not this.** `query_eval` (§16) draws its exemplars *from ground-truth
  instances* and scores against the held-out remainder. It is a research instrument for comparing
  representations and matchers, and it stays one. Its numbers do not transfer directly here,
  because the tool's exemplars are chosen deliberately by a human — and which clip is chosen is the
  thing that moves the score most, where the research eval draws them at random. A person choosing on purpose is
  neither.

### The run: two phases, one record

**Search** is computed, expensive, and immutable — a pure function of (exemplars, method, params,
embeddings). It fans out over `(behavior × recording × subject)` under the durable coordinator
(§15), because a cohort search is long enough that it must survive the app closing. A stopped or
failed search is re-run as a new run: the runner skips a shard already on disk, but nothing
re-enters a stopped run, so a `partial` search does not resume.

**Calibration** is authored, cheap, and instant — the cutoff, the behavior names, the audit
labels. It triggers no compute. Every change reprojects the §19 artifact from stored search
results.

**No cutoff exists during search.** The search config has no threshold field at all, so there is no
way to spend hours searching and then discover a change of mind costs another search. Adding a
behavior later, as an incremental job into the same run directory, is **not built**: there is no
`add_behavior` on any surface, and a behavior not searched at launch needs a new run.

The record stays open until the user finalizes it, carrying the non-terminal status §21 defines
for authored runs. Each search job still reaches its own terminal executor status; the two status
planes are orthogonal.

### Config: several behaviors, one launch

A run searches **N behaviors at once**, because a user quantifying a cohort almost always wants
more than one and searching each separately would reload every trajectory once per behavior. The
config splits in two:

- **Run-level**: the search recordings, the embedding run, the subject, the method id, the method's
  parameters, the sampling request, and the **minimum match length** (`min_match_s`, seconds,
  0 = none). One run means one method under one configuration.
- **Per behavior**: a name and its example clips — and, in the authored half, its cutoff.

**Method parameters are run-level, not per behavior.** Two reasons, and the second is the real one:
instances produced under different parameters are not comparable, and a run emits **one** §19
artifact that the measures layer treats as homogeneous; and with no ground truth a user has no
signal to tune per behavior against, so a per-behavior knob would be adjusted by impression. A user
who genuinely needs different parameters per behavior launches two runs, which is honest about
what happened. **The cutoff is the exception and is always per behavior**, because each behavior's
confidence distribution is its own and a shared cutoff would mean nothing.

**The minimum match length is a search setting, not a filter.** It reaches the method as
`search(..., min_rows=)`, converted from seconds on each unit's own timeline (a match of `n` rows
lasts `n` frame periods), and the method applies it **while matches are formed**. Filtering
afterwards loses real matches: a peak-and-suppress method lets a brief spike suppress the longer,
lower match beside it, then the filter drops the spike and the stretch returns nothing. Because no
short match ever exists, every reader of the results — the artifact, the calibration curve, the
audit — agrees with no filter of its own. It is run-level for the same reason as the method's
parameters; it sits in the launch form's main settings, not advanced, because a user can state it
for their behavior without a measurement.

**One background sample, shared across behaviors.** Where a method requests one, the module draws
it once — excluding the exemplar spans of *every* behavior in the run, not just the one being
scored. Per-behavior exclusion would make the draw behavior-dependent for no gain; a single draw is
one record, one seed, and identical for every unit.

**Bounds are validated per behavior, and all violations are reported together.** A user who marked
five behaviors and got two of them wrong should see both, not discover the second after fixing the
first and relaunching (Tenet 10 — validate everything before writing anything).

**A failing behavior does not take the others down.** Shards are per `(behavior, recording,
subject)`, so a behavior whose search fails leaves the rest complete and reviewable; the record
names which behaviors finished (`behaviors_searched`, `behaviors_failed`). The alternative — one
failure voiding a cohort-scale search — is the outcome per-behavior shards exist to prevent.

### `QueryMethod` — the protocol

A query method is a small ABC plus a registry (`QUERY_METHODS`), the same shape as `Tracker` and
`TRACKERS` — a `TrackingModule` needs no change when a tracker is added, and `Query` needs none
when a method is. It is deliberately lighter than `DetectionModel`: no weights, no install, no
device management.

```python
class QueryMethod(ABC):
    method_id: str
    config_model: type[BaseModel]        # its own parameters, single-sourced (§6.5)
    sampling: SamplingRequest | None = None   # what background it needs, if any
    exemplars: ExemplarBounds            # min/max clips, min/max total rows,
                                         # and requires_contiguous (below)
    negatives: NegativeSupport           # unusable | optional | required, + its own bounds
    shard_suffix: str                    # names this method's result file; no default, so a
                                         # method that forgets it fails the registry guard
                                         # rather than writing one file and reading another
    score_meaning: str                   # what the artifact's score column means, in the
                                         # method's own words (§19). Defaulted to the
                                         # canonical sentence, since `confidence` is already
                                         # [0, 1] higher-better across methods.

    def prepare(self, exemplars, params, *, negatives=None, sample=None) -> State: ...
    def search(self, state, trajectory, runs, params, *,
               row_offset=0, progress_cb=None, min_rows=0) -> SearchResult: ...
                                         # min_rows: the run's min_match_s in this unit's rows
    def instances_at(self, result, cutoff, params) -> list[Instance]: ...
    def merge(self, chunk_results, state, params) -> SearchResult: ...
    def chunk_overlap_rows(self, state, params) -> int | None: ...  # intra-unit split
    def candidate_cap(self, params) -> int | None: ...
    def serialize(self, result, path) / def load(self, path) -> SearchResult: ...
```

**The module never looks inside a `SearchResult`.** It is whatever the method needs — ranked
candidate spans, a per-row score curve, something a future method invents — and the method owns its
serialization. The module stores it, shards it, and hands it back.

**`instances_at` is where a cutoff becomes instances, and therefore where extent is decided.** This
is the pivotal division. A windowed matcher returns windows of roughly the exemplar's length; a
per-frame method could threshold its score and run-length-encode, or — as `soft_knn` does — form
its extents from the score's peaks before any cutoff and let `instances_at` filter them; either
way its extents follow the behavior. Both are correct definitions of "an instance" for their method, and neither is the module's
business. **A method documents how it defines extent** (`analysis/query_methods_docs.py` is the
generated per-method doc registry), because that definition is what a duration measure will report.
There is no post-hoc boundary-refinement step and no module-level notion of a "true" extent.

**Confidence is canonical: `[0, 1]`, higher is better.** Every method returns it alongside its
native score. The cutoff axis, the audit stratification, and the artifact's score column are then
uniform across methods, and the score-direction hazard §16 carries a warning about — a matcher
emitting a similarity where a distance is expected, silently reversing the ranking with no error —
becomes impossible to express rather than a declaration the module has to trust.

**No parameter carries a label.** `prepare` takes user-chosen exemplars and a module-drawn sample;
`search` takes a trajectory. There is no argument through which ground truth could enter, which is
the leakage rule (CLAUDE.md) made structural rather than conventional. In this module there are no
labels to leak — but the protocol is also what `query_eval` will drive, and there the guarantee is
load-bearing.

### Sampling

Some methods need a **background** — a reference set of frames standing for "not the behavior."
`soft_knn` scores a frame by a vote over its nearest labeled neighbors, which needs a negative
class to vote against; a distance matcher measures distance to the positives and can use
nothing else. Methods differ in what they need, so the requirement is declared and the module
serves it.

**The background has two sources, and the user owns both.** It is the **non-example clips the user
marked** — the same marking surface as the exemplars, in a negative role — plus an optional
**sampled** set of rows. Marked negatives are the valuable half: a human pointing at the thing that
resembles the behavior and is not it ("that rear-like posture is a stretch") carries far more
information than a random frame. (The removed annotation loop used a rejected clip only to *hide*
it rather than to push down what resembles it.)

**A method declares `NegativeSupport`** — `unusable` / `optional` / `required` — with its own
bounds, the negative twin of `ExemplarBounds`. When a method cannot use negatives the control is
**greyed out with the reason** rather than hidden (the GUI rule): a user who marks fifteen careful
non-examples into a method that silently discards them has been misled by an empty affordance.
A `required` method refuses to launch without them, at validate time, naming what it needs.

The sampled half is what the rest of this section governs:

- The method declares a `SamplingRequest`: **size**, **scope** (`subject` | `recording` | `cohort`),
  and a named **strategy** from a small registry the module implements (uniform random over
  observed rows, temporally stratified, …). `sampling = None` means the method needs none.
  **Built: `uniform` only** (`pipeline/query.SAMPLE_STRATEGIES`), drawn once over every searched
  subject — the `cohort` scope `soft_knn` declares. Other strategies and the `subject` /
  `recording` scopes are design, not built: the scope is recorded with the draw but not branched
  on.
- **The module draws it, never the method** — so every draw is recorded identically (strategy,
  scope, size, seed, and the rows drawn), and a run reproduces. Tenet 9's resolved-value rule: the
  record carries the sample that was used, not the request that asked for one.
- The exemplar spans are excluded from the draw and **nothing else is**. With no labels in the
  tool, the remainder is simply the unlabeled trajectory, which is by construction the pool a user
  could reproduce.
- **Scope changes the execution plan.** A `cohort` sample is a barrier drawn once before fan-out,
  written to the run directory, and loaded by each worker — it is too large for the job envelope
  (a few thousand rows at 1024-d is tens of MB) so the job references the file, per Tenet 4. A
  `subject` sample is drawn inside its own unit and needs no barrier.
- **Scope is also a comparability decision, and it belongs in the method's docs.** A background
  drawn per recording calibrates confidence to that recording, so one cutoff means different things
  in different animals. A cohort-wide draw keeps one cutoff meaning one thing. Either way, a
  background drawn from the unlabeled remainder contains the target behavior in proportion to how
  common it is, which can differ widely between recordings for the same behavior; that depresses
  confidence exactly where the behavior is most frequent.
  That is a property of background-using methods, not a defect to be corrected away, and the module
  surfaces it (see *Calibration*) rather than silently adjusting for it.

### Parallelism

Three axes, with different owners.

**Across search units** — `(behavior × recording × subject)` — is the module's, always available,
and needs nothing from the method. It is also **bounded by the longest single unit**: one 40-hour
recording beside five 1-hour ones gives a 40-hour tail however many workers exist.

**Within a unit** is the method's declaration. `chunk_overlap_rows` says whether a continuous
trajectory run may be split and by how much:

- `None` — not splittable, for a method whose recurrence genuinely carries state across the whole
  row axis. A naive split cuts matches at the seam.
- an integer — splittable with that overlap. A method whose scoring is row-local declares `0`; a
  windowed method that can be split declares its longest window so no match is cut.


The module decides **whether** to split, from worker count and unit imbalance; the method owns
**`merge`**, because deduplicating across the overlap is method knowledge and the module still
never looks inside a result.

**Across exemplars** is entirely method-internal and the module never sees it.

Two rules fall out:

- **The split plan must not change between two passes over the same run**, or a "completed" shard
  would no longer mean what it did. Today's rule takes no worker count: the plan is a pure function
  of the trajectory's row count and the method's declared overlap, so the runner computes it per
  unit (`QueryModule._search_unit`) and gets the identical plan every time. A future rule that
  depends on the environment (a worker count) has to move to `build_job` and be recorded.
- **Chunking bounds memory as well as wall clock.** A worker holds one subject's trajectory, and a
  long recording at full frame cadence and 1024 dimensions runs to gigabytes. The same declaration
  that enables intra-unit parallelism lets a worker stream a long run in pieces — so an
  unsplittable method is also the one that can exhaust memory on a long recording, which the module
  knows up front and warns about rather than discovering at hour six.

### Storage

```
analysis/an_NN/
  analysis.yaml       lifecycle status (open until finalized)
  query.yaml          authored state: behaviors, exemplars, method + params, cutoffs, names
  history.jsonl       append-only operation history (§21)
  background.npz      the drawn background sample + its draw record, when the method asks for one
  search/<behavior>/<recording>__<subject>[__chunk_KK].<ext>    method-serialized, opaque
  audit.csv           optional human decisions
  audit.yaml          each behavior's fixed score bins, from its first audit draw
  instances.csv  coverage.csv  behavior_output.yaml            the §19 artifact
  instance_features.csv                                        §19, beside the artifact
  frames/             the §19 frame view (+ frames/PENDING while an edit has left it stale)
  snapshots/<snap_NN>/   frozen copies of the artifact (§21)
  job.json            transient executor envelope
```

One shard per search unit, which the fan-out produces naturally: the runner never recomputes a
shard already on disk. The shard extension is the method's, since the method owns serialization.

### Calibration


Per behavior, one axis — confidence — and everything read against it:

- **The frequency curve**: instances returned at each cutoff. Scrub it and clips from that band
  play, which is the only way a user without labels can judge where to sit.
- **Its floor beside it.** A number with nothing to compare it against cannot be read (§16). For a
  method whose confidence is a per-frame classification, the floor is the behavior's prevalence at
  that cutoff; for a ranking, it is what a shuffled ranking returns. Which floor applies is the
  method's to declare, and it is drawn on the same axes as the curve. **Not built:**
  `QueryRun.frequency_curve` returns `{cutoff, n_instances, capped}` per point and no floor, and
  methods declare none.
- **Per-recording and per-group confidence distributions**, always shown. A single cutoff that
  keeps most of one animal's behavior and little of another's biases the result *along the axis
  being compared*, and that is invisible in a pooled curve. Showing the distributions makes it
  legible; the module does not auto-correct it with per-group cutoffs, which would be tuning toward
  the answer. **Built per recording**, as instances kept at the cutoff (*Three layout decisions*,
  below); a per-group view is not built.
- **The audit overlay**, when one exists: precision against confidence, and the product of the two
  — estimated true instances, which is the number that reaches a figure. **Not built:** the Audit
  tab lists precision per score bin (`audit_precision_by_score_bin`), but nothing draws it on the
  curve and the product is not computed.
- **The cutoff is the user's**, set by looking at what each setting returns. Nothing is suggested
  or pre-applied in v1 (see *Deferred*).
- **The stored edge.** Where a method caps what it stores, the cap is recorded and drawn, because a
  curve that flattens because storage ran out looks exactly like a curve that flattens because the
  animal stopped.

The module builds the curve by calling `instances_at` across the confidence range and caching the
result, so it branches on nothing method-specific. Instance count is **not guaranteed monotonic** in
the cutoff — a method that re-forms extents can split one instance into two as the cutoff tightens
— and the view renders what it is given rather than assuming a shape.

**Zero matches is a first-class state.** A behavior that returns nothing says so, at every surface,
as a result. An empty list that reads as a hang was a recorded failure of the removed annotation
module.

### The audit

Optional, and the only quality signal that exists in this module. §19 specifies the mechanism —
store the labeled sample, derive the metric. What Query adds is the property that keeps it from
becoming a labeling loop.

**The audit measures the method, not the cutoff.** The sample is stratified **uniformly across the
whole confidence range**, independent of any cutoff, and reviewing it yields precision *as a
function of confidence*. The cutoff then reads off that curve. So:

- **Precision at a cutoff is weighted by score bin.** The draw gives every bin the same share of
  clips while the bins hold very different numbers of candidates, so each bin's precision counts by
  how many of the cutoff's instances fall in it (`behavior_audit.stratified_estimate`, the same
  estimator clustering weights by cluster size), with the Wilson interval on the effective n. A bin
  the cutoff cuts through counts its kept part. The plain share of the judged clips is not an
  estimate of anything: on a skewed score distribution it read 0.62 against a true 0.40, and its
  interval covered the truth in 17% of audits.
- **The bins are fixed per behavior.** Ten equal-width bins over the behavior's candidates at no
  cutoff — a pure function of the search, which never changes — recorded in `audit.yaml` at the
  behavior's first draw (`schema_version: 1` first, `{stratified_by, bins: {<id>: {edges, seed,
  fixed_at}}}`). The draw, every estimate and the confidence curve read those edges. A draw across
  several behaviors bins each on its own scores. Clips drawn before a behavior's bins were recorded
  (a draw with no behavior chosen cut one set of bins over every behavior) are weighted by the
  recorded bins all the same, and the figure says how many of its clips that covers.
- **The estimate is over the candidates the cutoff keeps**, not the artifact, so a cutoff below the
  committed one has a population to weight. For `soft_knn`, the one registered method, these are
  the same instances (`instances_at` is a filter of one extraction).

- **One audit serves every cutoff.** Move the cutoff afterwards as often as you like and precision
  follows, with no re-sampling and no further labeling.
- **Order is free** — audit before choosing a cutoff or after; it is an independent act.
- **Sampling is trivial**, because it depends on nothing that changes.

Weighting the sample toward the current cutoff would undo all three: it makes the sample a function
of a decision the user is still making, so every adjustment invalidates part of the evidence and
the flow degenerates into set-cutoff → audit → re-set → re-audit. The cost of not weighting is some
labels spent on obviously-wrong low-confidence clips — cheap to judge, and they anchor the bottom
of the curve.

**The reference cutoff, for methods whose instances are cutoff-dependent.** (Design: `soft_knn` is
not one — its candidates exist before any cutoff.) A per-frame method that thresholds has
no instances until a threshold is chosen, so there is no cutoff-free candidate set to sample. The
module enumerates at a **permissive reference cutoff**, records it with the sample, and reports
precision at or above it — below the reference it is unknown, and says so.

**Labels survive refinement, which is what makes the reference work.** When a stricter cutoff
splits or shrinks an audited instance, the pieces inherit its decision: a sub-span of a confirmed
instance is still one, a sub-span of a rejected span is still not. One bounded error follows and is
documented rather than solved — a span rejected as a whole may contain a short real fragment that a
stricter cutoff isolates, which will inherit the rejection.

**Blind review, in randomized order.** The reviewer does **not** see the confidence while judging,
and clips arrive shuffled rather than score-ordered. Knowing a clip scored 0.95 pulls the judgment
toward accepting it, which inflates precision by exactly the quantity the audit exists to measure —
and it fails silently, since a biased estimate is still a plausible number. The score is revealed
after the decision.

**Precision reports its interval.** 0.88 from n=25 and 0.88 from n=200 are different claims, and a
bare point estimate hides which one is on screen. Same discipline as a headline metric shipping
beside its floor (§16).

**Re-auditing appends.** A later, larger sample draws from instances not already labeled and adds
rows; n grows and the interval tightens. Nothing is recomputed by hand and no earlier decision is
discarded (§19).

**Not stratified by group.** The draw falls across recordings and groups in proportion to where the
candidates are; the per-recording split weights by score bin within each recording and reports its
own `n`, refusing below the floor rather than balancing the draw, which would double the labels for
a fixed number of bins.

**The user's own clips are never audited.** A candidate overlapping one of its behavior's example or
non-example clips is the method finding its input; it is never drawn and a judgment of it is never
counted (`query_runs.example_instance_ids`, §19). It stays in the run's instances and export, marked
`is_example` in `instance_features.csv` and in the export's `instances.csv`.

**Per behavior, never interleaved.** A session reviews one behavior. Alternating "is this rearing?"
with "is this grooming?" is error-prone in a way that corrupts the estimate quietly.

**An unaudited run looks unfinished.** Not blocked — a user may take instances without reviewing
any — but an unaudited count must not render like an audited one, because the difference between a
behavior this works well on and one it does not is invisible without it.

### The workspace — four tabs over one behavior

The authored half is one screen: a behavior selector, and beneath it four tabs
(`gui/modules/analysis_query.QUERY_VIEWS`); the first three are three different jobs on the same
instances. Switching behavior keeps the current tab.

```
 [ Rearing ]  [ Grooming ]
 ┌──────────┬───────────┬──────────┬──────────┐
 │ CALIBRATE│   AUDIT   │  BROWSE  │ MEASURES │
 └──────────┴───────────┴──────────┴──────────┘
```

- **Calibrate** — clips *at* the cutoff, to decide where to cut.
- **Audit** — blind and randomized, to estimate precision.
- **Browse** — the accepted instance set, sortable, read-only, to see what you actually got.
- **Measures** — §23's measures of the live artifact (`pages.MeasuresPanel`); live runs only, so it
  is absent while a snapshot is shown (§21).

**Nothing is a wizard.** Audit before or after choosing a cutoff, browse at any time, move the
cutoff after auditing — which is the whole point of the audit being cutoff-independent (below).
**Browse survives Finish**: completing sets the record status, it does not close the workspace, and
reopening is only needed to *change* something.

The first three exist because they answer different questions and one surface cannot serve all of them.
Calibrate shows the *marginal* clips, because the decision is "is what sits at this level real?".
Audit hides the score, because seeing it biases the judgment. Browse shows everything above the
cutoff sorted however you like, because that is where a systematic error becomes visible — every
match in one recording being the same wall-lean is invisible in a marginal band and obvious in a
list.

```
┌─ Analysis / an_07 · Query · open ─────────────────────────────────────────┐
│  Rearing — 412 instances · precision 0.88 (0.76–0.95, n=50 audited)      │
│                                                                           │
│  instances                                                                │
│   600 ┤····                                        ── precision           │
│       │    ····            ────────                ·· instances           │
│   400 ┤        ·····┊──────                        ▁▂▃ floor (shuffled)   │
│       │             ┊·····                                                │
│   200 ┤             ┊     ········                                        │
│       │      ▁▁▂▂▃▃▄┊▄▅▅▆▆▇▇███············                               │
│     0 ┼─────────────┊───────────────────────────────                      │
│      0.0          0.42                          1.0   confidence          │
│                 ▲ cutoff — drag                                           │
├───────────────────────────────────────────────────────────────────────────┤
│  by recording   rec_A    ▁▂▅█▅▂    rec_B    ▁▃█▆▂▁                        │
│                 rec_C    ▁▁▂▃▂▁    rec_D    ▂▅█▃▁▁                        │
├───────────────────────────────────────────────────────────────────────────┤
│  Clips  ● near cutoff (0.40–0.44)   ○ sampled above                       │
│   ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐                        │
│   │ ▨  ││ ▨  ││ ▨  ││ ▨  ││ ▨  ││ ▨  ││ ▨  ││ ▨  │  click to play         │
│   └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘                        │
│    0.44  0.43  0.43  0.42  0.41  0.41  0.40  0.40                         │
├───────────────────────────────────────────────────────────────────────────┤
│  [ Review 50 clips… ]        [ Add behavior… ]        [ Finish ]          │
└───────────────────────────────────────────────────────────────────────────┘
```

Not built from this sketch: the precision line and the shuffled floor on the curve (*Calibration*
above), and **Add behavior…** (*No cutoff exists during search*). The per-recording row and the clip
strip were built differently, as the decisions below say.

Three layout decisions, each with a reason:

- **Precision and instance count share one axis** despite being different units. The whole point is
  reading them together *at a cutoff*, and splitting them into two panels makes the user eye-trace
  between plots to answer one question.
- **The per-recording view is always visible, not behind a toggle.** It is the panel that reveals a
  cutoff keeping most of one animal's behavior and little of another's — the bias that lands along
  the comparison axis (see *Sampling*), and the one a pooled curve cannot show. It shows **instances
  kept per recording at the current cutoff**, and names a recording contributing far fewer than the
  rest. It was specified as per-recording score *histograms*; users read those as "a pixelated
  graphic with no context" and could not say what they were for, so the panel now states the answer
  rather than drawing the raw shape it is derived from. The distribution remains available through
  `QueryRun.score_distribution`.
- **The clip strip is the ranked list itself, with the cutoff drawn in it.** Scrolling loads clips
  toward more or less confident matches; the cutoff is a line at its own place in the ordering, and
  moving *that* is what sets it. This replaces the marginal-band / sampled-above modes, which were
  two fixed windows onto an ordering the user could not otherwise traverse. The reason is a measured
  one: the frequency curve is a couple of hundred pixels across the whole confidence axis — about
  0.005 per pixel — while the range that decides a cutoff is three decimal places wide, so the curve
  is a readout and the strip is the control. Scrolling never commits; releasing the line does.

### What the workspace reuses, and the lessons that come with it

Every one of these was learned by building a video surface in this repo, and each is written down
where it was learned. A new surface that re-solves one of them will re-solve it worse.

- **`FrameScrubber`** — off-thread, **latest-only** decode: a spam of clicks costs one decode of the
  final frame, not N. Keyboard stepping accelerates while held, giving reach and precision, because
  a mouse slider is ~1 frame/pixel only on short clips and useless on a 400k-frame video. Its
  `busy_changed` overlay is what makes a slow decode say "Decoding frame…" rather than showing a
  blank pane.
- **`_ClipPlayer`'s trim mode** — scrub a *window* around the clip, not the whole recording. Over a
  29-minute part a full timeline puts ~2.8 s behind every pixel, which makes a one-second clip a
  third of a pixel wide and impossible to grab. **Browse needs this**: clicking an instance and
  getting a full-duration scrubber reproduces the exact bug it fixed.
- **Edge-walk panning** — holding the scrubber at the window edge walks the window, ramped and
  capped low, because this moves the *view* and losing your place is worse than arriving slowly.
- **`gui/confidence.py`'s debounced threshold + live caption** — and its lesson, which is the
  calibration curve's lesson verbatim: a stored count cannot answer "how many at 0.7?", so the view
  must **recount**, and one that forgets shows a confident wrong total.
- **Arrow keys scrub the clip; they do not navigate the list.** Settled in the removed annotation
  review tab, kept by `_ClipPlayer` (focusing the player gives the scrubber the arrow keys), and
  counterintuitive enough to be re-broken.

Browse is a sortable instance table beside a `_ClipPlayer` (`_BrowseTab`). The per-recording span
timeline it was to reuse belonged to the annotation workspace, which was removed; Browse has none.

**The clip grid is the one genuinely new widget, and the decode economics shape it.** Playback here
is a 120 ms timer stepping frame by frame — a slideshow, not decoded video — and every thumbnail is
a `cv2` seek, so a strip of eight is eight seeks. Therefore: thumbnails **fill progressively**, each
decoded off-thread with latest-request-wins per cell; moving the cutoff **cancels in-flight decodes**
for the abandoned band; and each player owns its own preview, since sharing decode cache between two
players was already found to be a problem.

**A cell that cannot decode says which reason.** `gui/frame_source.py` keeps *missing* (the video is
not on this machine) and *undecodable* (this position has no picture) distinct, and a surface that
collapses them into an empty pane reports neither. **The audit is the highest-stakes instance of
this**: it shows a clip and takes a judgment, so a frame that did not render must **refuse the
decision** rather than record one against a blank pane — a judgment made against nothing still lands
in the precision estimate. This is the recorded class in `tests/gui/test_frame_source.py`.

### Progress and loading

Four kinds of wait, matching the four cases in the long-operation rule, none of them optional.

| moment | kind | requirement |
| --- | --- | --- |
| the search | executor-backed | `ProgressEvent` per unit, **inner fraction via `core/progress.py::split_units`**, `eta_s` from `remaining_seconds`, rendered through `fit_progress_fraction` and the CLI's `_stream_progress` |
| opening a behavior | GUI interactive load, sized | reading N shards is O(recordings) disk: `load_results` then `frequency_curve` in one `busy.run_progress` job, never inline in a widget `__init__`; the curve is then **cached** per behavior |
| thumbnail decode, scrub, playback | interactive | off-thread, latest-wins, progressive; `busy_changed` on the player |

Three specifics that are easy to get wrong and are not visible in a screenshot:

- **Dragging the cutoff must not reproject the artifact.** Reprojection is cheap *relative to a
  refit*, but for a per-frame method it is threshold-plus-run-length-encode over tens of millions of
  rows per recording. The drag updates the view **from the cached curve**; releasing commits the
  cutoff and reprojects **off-thread**. Conflating the two freezes the GUI on every pixel of a drag.
- **The aggregate bar's unit is not "recordings."** A search unit is `(behavior × recording ×
  subject)`, so the bar reads "7 of 12 searches"; labelling it recordings makes a two-behavior run
  over six recordings claim twelve recordings.
- **The search names its phases and advances them** — `loading → preparing → searching → writing`,
  each forced past the throttle. `preparing` is the background draw, which happens once before
  fan-out and is exactly the phase where the numeric fraction is pinned and a static label reads as
  a hang. That needs the phase-message test, not only the cadence test.

The shard load, `load_results(`, is in `tests/gui/test_ui_thread_blocking.py`'s scan list, so a
later inline call fails CI rather than shipping a freeze. `frequency_curve(`, `candidates_at(` and
`score_distribution(` are deliberately **not**: each is a slice over the candidates `load_results`
already extracted, and the calibration view calls them on every drag *because* they are cheap.
Guarding the load is what keeps them cheap.

### Exemplar guardrails

The exemplars are the one input the user controls and the one place the research transfers cleanly,
because it concerns the input rather than the score:

- **Warn below ~2 s.** A minimum-duration filter lifts the worst case —
  it works on the downside, which is what a guardrail is for.
- **Push toward 3–5 clips.** At one exemplar, which clip is picked moves the score widely;
  several clips steady it.
- **A span must lie inside one continuous trajectory run** *if the method says so*
  (`ExemplarBounds.requires_contiguous`), or it silently spans a tracking gap. Validated with the
  gap named. The bound belongs to the method, not to the module: a path matcher such as DTW
  needs one unbroken stretch, because a clip spanning a gap is not a trajectory the animal traced;
  `soft_knn` — the default — pools every exemplar frame into an unordered index and never consults
  the order, so the same rule would refuse clips it could have used. `True` is the conservative
  default, so a new method is refused a gap-spanning clip until it declares it can take one.

These are warnings and defaults, not refusals: a user who knows their behavior is brief must still
be able to mark a brief clip.

**Where exemplars come from.** Always from the **search set** — a behavior demonstrated on a
recording that is not being quantified cannot be checked against what the run returns, and the
comparability of its embedding run would be one more thing to validate for no gain. Built ways to
supply them, all landing in the same place: marked in the clip dialog (`ClipExampleDialog`), typed
on the CLI (`--example`), or **imported from a clip CSV** (`--from-csv`, §16 *The clip CSV* — for
example a cluster montage's exported clip list). Reusing a previous Query run's exemplars directly
is not built: a Query run writes no clip CSV. Non-examples are marked in the dialog or given with
`--non-example`; `--from-csv` seeds examples only.

**What a method requires is different, and is a refusal.** Every method declares `ExemplarBounds`
— a minimum and maximum number of **clips**, and a minimum and maximum **total rows** across them
— and `validate()` enforces them before any compute (Tenet 10), naming the bound and what was
given. The two constraints are independent: a method that pools every exemplar frame into one
index cares about the total row count and barely about how many clips it came from, while a method
that scores each clip separately and fuses cares about the count, because its cost is linear in it.
A method that genuinely requires exactly one exemplar says `min_clips = max_clips = 1` rather than
failing obscurely on the second.

**Why rows here and seconds above, which looks like a violation and is not.** The unit-convention
guard (`tests/configs/test_unit_conventions.py`) fails a config field declaring a *temporal window*
in frames or rows, because the same count is a different duration at every cadence. These are not
temporal windows. The 2 s minimum is a statement about capturing the behavior, so it is seconds;
`ExemplarBounds`' row counts are a statement about how many vectors the method is given, which is
a sample budget and cadence-independent in exactly the way a window is not. They are also not
config fields — they are declarations on the method class, fixed by its implementation rather than
set by a user. Both facts should be stated where they are declared, or someone will helpfully
convert them to seconds and change what the bound means.

### The artifact

Query writes the §19 behavior output — `instances.csv`, `coverage.csv`, `behavior_output.yaml` —
reprojected whenever calibration changes. `source_label` is the behavior's id, `label` its name,
`score` the method's confidence with its declared meaning. `coverage.csv` records every
`(recording, subject)` searched and its observed time, so a behavior returning nothing in a
recording that *was* searched is a real zero and not a gap.

### Surfaces

- **API** — `Experiment.run_analysis("query", config)` for the search; a `QueryRun` object
  (`Experiment.query_run`) for the authored half (`set_cutoff`, `name_behavior`, `audit_sample`,
  `label_audit`, `finalize` / `reopen`, `export`).
- **CLI** — `analysis query` (launch), `analysis status` / `attach`, `analysis query-calibrate`
  (cutoffs, names, the frequency curve, finalize / reopen), `analysis audit` (reads the audit's
  precision), `export-query`. Marking clips and judging audit clips are GUI-only, recorded in
  `docs/cli.md`'s omission ledger like every other deliberate gap.
- **GUI** — one config form (exemplar and non-example marking through the existing
  `ClipExampleDialog`, method in advanced settings), and the four-tab workspace above. Browse is
  reachable after Finish; only changing something requires a reopen.

### Reference implementation

One method is registered: **`soft_knn`**, the per-frame method and the default
(`configs/query.DEFAULT_METHOD` is the single source for the model, the CLI option and the GUI's
opening selection). It pools every exemplar frame into one index and scores every row, then forms
instances **before any cutoff exists**: it smooths the score, takes its peaks highest first with
non-maximum suppression, and grows each peak to `extent_fraction` of its own height
(`query_methods/soft_knn.py`). So its extents follow the behavior, and `instances_at` is a filter
over those candidates by peak score, not a re-formation — raising the cutoff only removes
instances. It draws a background sample, and its cost is flat in the number of exemplars. It is
the default on cost; which kind of method retrieves better has not been measured.

The protocol is written so that a windowed path matcher fits it too: ranked candidate spans of
roughly exemplar length ± a band, no background, a chunk overlap of `max_exemplar_rows +
band_rows`, cost linear in the exemplar count. `analysis/retrieval.py`'s banded subsequence DTW is
the engine such a method would wrap. Its `band` is a duration tolerance, and the returned span is
the DP's own start-to-end extent, so spans already vary over `L ± band`; a method may expose a
**multiplicative duration range** and derive `band` from it. Cost and memory are linear in `band`,
and the DP has no slope constraint, so a wide band admits degenerate warps and absurdly short
matches once `L - band` clamps at 1 — filtering matches to the declared duration range closes both.



### Guards

- **Registry parity**, data-driven over `QUERY_METHODS`: every method builds, declares a config
  model, returns canonical confidence, and answers `sampling` / `chunk_overlap_rows` /
  `exemplars`. The bounds must be self-consistent (`min <= max`, and a row minimum reachable by
  the clip maximum) — a method declaring `max_clips = 2` and `min_total_rows = 10_000` accepts
  nothing, and does so only when a user tries.
- **Chunk equivalence**, asserted on the output rather than the intermediate: for a method
  declaring an overlap, a chunked search yields the **identical match set** as an unchunked one,
  while the curve is only required to agree to float tolerance (see *Parallelism* — bit-equality is
  not available, because BLAS blocks a sliced matmul differently). Assert the match set, not the
  costs, or the guard fails on numerical noise while missing the failure it exists to catch.
  Otherwise splitting silently changes answers on exactly the long recordings it exists to serve,
  and the disagreement is data-dependent enough to read as noise.
- **Cutoff monotonicity** for methods whose candidates are cutoff-stable: raising the cutoff only
  ever removes instances. `soft_knn` forms its candidates before any cutoff exists, taking peaks
  **highest score first** so a strong match is never suppressed by a weaker neighbour, and
  `instances_at` filters them by peak — so a stricter cutoff keeps a subset of a looser one. It is
  the property the whole calibration workflow rests on, so it is a test rather than an assumption
  (`tests/query_methods/test_soft_knn.py`).
- **Sample determinism**: the recorded draw reproduces the sample.
- **Progress cadence** on a single-unit fixture — the inner pair must move, not just the unit count
  (§6) — and phase messages must advance through `loading → preparing → searching → writing`.
- **Label leakage**: `tests/pipeline/test_label_leakage.py` permutes labels and asserts
  predictions are byte-identical — over `query_eval`'s `evaluate_query` only. Query is not driven by
  it, because Query takes no ground truth at all: no parameter of `QueryMethod` carries a label
  (*The protocol*), so its protection is structural rather than this test.
- **A judgment is refused against a frame nobody saw.** The audit shows a clip and records a
  decision, so a cell whose frame did not decode must reject input and name the reason rather than
  accept a decision against a blank pane — the recorded class in `tests/gui/test_frame_source.py`,
  at its highest-stakes site, since such a decision still lands in the precision estimate.
- **Dragging the cutoff reprojects nothing.** Assert that a drag reads the cached curve and that the
  artifact is rewritten only on commit, off-thread. A test that only checks the number on screen
  passes while the GUI freezes on every pixel.
- **The workspace's slow seam is in the UI-thread scan.** The shard load (`load_results(`) is listed
  in `tests/gui/test_ui_thread_blocking.py`, so a later inline call fails CI rather than shipping a
  freeze; the curve built from it is deliberately not (*Progress and loading*).
- **Every navigable surface is constructed in a test.** The four tabs, the clip grid, and the audit
  review are each built against a fake completed run, because a renamed slot raises at construction
  and nowhere earlier.

### Deferred

- **An automatic cutoff suggester.** The user chooses the cutoff. The seam is left open: with an
  audit in hand the honest form is target-based ("the cutoff that holds precision at 0.9"), read
  off the measured curve; without one, only the distribution's shape is available, which is a guess
  and would have to be labeled as one — and be capable of declining, since a suggester that always
  returns a number teaches people to trust it.
- **Exemplar expansion** — feeding confirmed matches back as new exemplars. Nothing measurable is
  gained past about four exemplars, so the loop's premise is weak.
- **Method comparison by the user**, which would need auditing two runs and comparing their
  precision. Possible on this machinery, out of scope for now.
- **Compute backends.** A per-frame method's core is a large matrix product and a natural GPU
  candidate; `ComputeTarget` already exists (§5). A method may declare supported backends when one
  needs it.
- **Per-behavior method parameters.** Run-level for the reasons above; revisit only if a real case
  appears where one method setting cannot serve two behaviors in one cohort.
- **Multi-subject** beyond `subject` being carried as a dimension throughout (§19).


## 21. Authored Runs

An **authored run** is one whose output depends on human decisions rather than only on its
parameters. It is not a fourth module family (§16) — an authored run is still an analysis module —
but a fourth **lifecycle**, orthogonal to what a module produces.

The distinction is real, not stylistic. A computed run (location export, query-eval) is
reproducible from its inputs and params, immutable once complete, and "refined" by seeding a *new*
run. An authored run cannot be re-derived from parameters, because the parameters were not what
decided it: a person set a cutoff by looking at clips, named a cluster, or judged fifty candidates.
Tenet 9 is met not by re-derivation but by **recording the human input as primary data**, so the
output is deterministic from the run's own recorded state.

**Two modules share this lifecycle**, which is why it is written down once:

| module | expensive immutable pass | authored decisions |
| --- | --- | --- |
| Query (§20) | the cohort search | cutoff, behavior names, audit labels |
| temporal clustering (§22) | the fit + cached comparison (+ node cache) | node labels, audit labels |


**This section is a contract, not a base class.** The distinction is deliberate and the reasoning
is worth keeping, because this is the kind of infrastructure that gets built too early. What is
specified here — the status vocabulary, one-writer-per-fact, `open` vs `partial`, sequential
compute, finalize/reopen — is **cross-cutting**: every authored module picks a status string and
every one meets the shell's "show results when complete" gate, so those decisions get made three
times whether or not they are written down. Writing them down moves the decision from three places
to one, which costs a paragraph and is not speculative.

The **shared code** is a different matter and was deliberately *not* specified up front. A base
class designed from three specs rather than three implementations tends to fit none of them, and
three modules then inherit the misfit. So the expected shapes were recorded at the end of this
section as **extraction targets**, to be lifted once the second authored module existed and the
common parts were visible rather than predicted. That has happened for one of the three:
`AuthoredRun` (`api/authored_run.py`) was extracted once Query and temporal clustering both
existed. An `Experiment.run_authored` launcher and a `MutableRunWorkspace` GUI base were **not
built**, and stay extraction targets (*Extraction targets*, below).

What that leaves for a module being built now: follow the contract, and implement its own state
handling. Two small utilities are the exception — see *Extraction targets*.

### The shape: one expensive pass, then cheap decisions

Every authored run in this design has the same two-part structure. Something long and immutable
runs once — a cohort search, a fit, an O(bouts²) comparison — and then the user makes decisions
that are **cheap by construction**, because they operate on what that pass stored. A cutoff filters
stored candidates; a label on a tree node is a map entry.

This is a design constraint, not an observation. **If an authored decision triggers the expensive
pass, it is not an authored decision — it is a new run**, and it should be modeled as one. A
module whose "cheap" knob turns out to cost hours has to be re-shaped rather than shipped with a
slow interaction, because a decision a user is expected to iterate on cannot cost a coffee break.

### Three kinds of operation

- **State edit** — pure, instant, no executor. Rewrites the state file atomically (tmp +
  `core.atomic.replace`; the concurrently-read-file rule). `set_cutoff`, `name_nodes`.
- **Log append** — one human decision, appended with `flush()` + `fsync()`. Deliberately *not* a
  state edit: the append-vs-rewrite difference is what makes human judgment durable, since a crash
  loses at most the final row and a torn trailing line is tolerated on read. `label_audit`, and
  the clip review's `clip_review.csv` (§22).
- **Compute operation** — a fresh executor job into the **same** run directory, with
  `job_id = run_id`, so re-submitting overwrites only the transient `job.json` while the authored
  state persists. **None is built:** a refit mints a new run (*Snapshots*, below), and adding a
  behavior to a Query run is not built (§20).

These three are named because the **rules differ per kind** — atomic rewrite, crash-safe append,
and same-dir job submission — not because a dispatcher is being specified. How a module routes them
is its own business until there is a second implementation to generalize from.

### State model, and the one invariant

```
analysis/<run_id>/
  analysis.yaml     the discoverable index — the ONLY lifecycle-status source
                    (+ `artifact_stale: true` while the artifact does not match the state)
  <module>.yaml     the authored state. No status. No counters.
  <name>.csv        append-only decision log(s) — primary data
  history.jsonl     append-only operation history — provenance, non-authoritative
  ...               the expensive pass's own outputs (shards, caches, iterations)
  instances.csv coverage.csv behavior_output.yaml   the §19 artifact — derived
  snapshots/<snap_NN>/   frozen copies of the artifact, pinned when a result leaves
  job.json          transient envelope for the current compute operation
```

**One writer per fact.** Status lives in `analysis.yaml` and nowhere else; authored configuration
lives in the state yaml; human decisions live in the log; anything countable is **derived** and
stored nowhere. The framework makes this structural, because dual writers on a single fact is how
these drift, and the drift is silent: both files stay well-formed and disagree.

**Two utilities were worth lifting**, because they are small, they already worked, and getting
either subtly wrong is a data-loss bug rather than a tidiness one: `write_state_yaml` (atomic
rewrite) and a crash-safe `AppendLog` (append + `flush()` + `fsync()`, tolerant of a torn trailing
line on read). Both came from the removed annotation module's `annotation_runs.py` — an extraction
from working code rather than a design from specs. A module that hand-rolls either is the bug.

**Both are lifted.** `AppendLog` is `storage/append_log.py`, extracted when §19's `audit.csv` became
its second consumer; `behavior_audit.py`, `clip_review.py` and `activity.py` use it. It carries the
header-drift migration too, which is not incidental: the header is written once at creation, so a
log started under an older column set goes ragged on the next append and a positional read
silently discards that decision and every one after it. Use it rather than re-deriving it.
`write_state_yaml` is `storage/state_yaml.py`, shared by §20's `query.yaml` and §22's
`temporal_clustering.yaml`.

### Two status planes, orthogonal

- **Record status** — `open` while authoring, `complete` after finalize, `failed`. Non-terminal by
  design.
- **Executor status** — each compute operation is an ordinary job reaching its own terminal state,
  so `stream_progress`, `worker.run_job`, and the GUI progress workers need no special case.

These do not conflict: a run stays open for its whole life while individual jobs inside it
complete. The operation's runner writes the record status back to `open`, **never** `complete` —
only the user closes a run.

**A record may claim an outcome the envelope has not reached; it may never claim a success the
envelope contradicts.** The runner writes `open` *before* projecting the artifact — an open run
is supposed to have one — so a job that dies in the artifact phase leaves a record reading `open`
over a `job.json` reading `failed`, and every surface reads the record. `core/dataclasses.py::
reconciled_status` is the one rule, and every reconciler goes through it (`AnalysisRun.wait`,
`AuthoredRun.wait`, `ExportRun.wait`, `attach_job_run`, `Experiment._reconcile_job_run`):
adopt the envelope's terminal status where the runner never finalized the record, **and** where
the envelope is `failed`/`cancelled` and the record claims anything else. `open` against a
`complete` envelope is untouched — that is the runner's own statement, and the asymmetry is the
whole point. The earlier guard, `status in (None, "running")`, was correct for the computed
modules and silently inapplicable to the authored ones; the reconciler that skips a non-active
record cannot see an authored run at all.

`mutable` moves from the GUI module UI onto the `AnalysisModule` itself, so the API and CLI know it
too, and the hardcoded status tuple in `gui/pages.py` reads the flag instead of listing statuses.

### Compute inside an open run

- **Strictly sequential.** The user acts between operations, so two at once is a bug rather than a
  concurrency problem to solve. Submitting a second *compute* operation while one is running
  **raises**; it does not queue.

  **State edits are serialized rather than refused**, because the sentence above was quietly false
  for them: every state edit runs off the UI thread through `gui/busy.py`, so two quick edits —
  labelling twice, or naming while another label change is still in flight (the granularity slider,
  when there was one, §22) — run two `_edit` calls at once. Both halves then break silently — the state rewrite is read-modify-write,
  so the second writer overwrites the first; and `_project` rewrites the frame views, where two
  writers race on one zarr store (observed twice in five full-suite runs, as `an array exists ...
  at path 't_ns'` and as a vanished `zarr.<uuid>.partial`, both from one call `rmtree`-ing the
  staging directory the other was writing into). `AuthoredRun` holds one lock per run directory
  across the whole of `_edit`, reprojection included. A lock rather than a refusal: the second edit
  is a real thing the user asked for, it is already off the UI thread so waiting costs nothing, and
  raising would surface as an error box for an ordinary double-click.
- **An interrupted operation leaves authored state intact**, because it lives in different files
  from the compute output. The log survives by construction, so a crash mid-operation costs the
  operation and nothing the user typed.

  **What it does not leave intact is the agreement between the state and the artifact**, and the
  original ordering — write the state, log it, then reproject — made that disagreement silent:
  a failed reprojection left a run whose state said one granularity and whose artifact said
  another, with a history line asserting the change had succeeded. §19 says regenerate rather
  than reconcile, so the disagreement does not have to be prevented — but it does have to be
  *recorded*. `_edit` sets `analysis.yaml`'s `artifact_stale` **before** writing the state and
  clears it when the reprojection lands. Before, not in an `except`: a SIGKILL between the two
  writes leaves exactly the state an exception does, and nothing catches that. `reproject()` —
  `detail analysis rebuild`, the GUI's **Rebuild results** — is the way back, and a stale run
  offers that in place of its results rather than showing one setting's numbers under another
  setting's heading.

### `open` is not `partial`, and the artifact depends on the difference

§19 says a `partial` run has no behavior output: compute was interrupted, so `coverage.csv` cannot
honestly state what was analyzed, and coverage is what separates a real zero from a missing
recording.

An **`open`** run is the opposite case — its compute is complete and only the authoring is in
progress. It therefore **has** an artifact, reprojected on every operation that changes what the
instances are, and marked as coming from an open run. Withholding it until finalize would invert
the workflow: the statistics are what the user is *supposed* to be looking at while deciding.

Each operation declares whether it invalidates the artifact. A cutoff or a name does; editing a
description does not. Invalidating is not rewriting everything: an edit writes the intervals, the
coverage and the instance features, and **defers the frame views** to the first reader that needs
them (§19, the frame view) — so a name commits in seconds on a large run, and a session of names
pays for the views once. A reprojection reads each unit's **timeline**, never its embeddings: the
rows an edit needs are a function of the runs and the window alone (`segmentation.feature_rows`),
and reading the full embeddings per edit cost many GB and minutes per name.

### Finalize, and reopening

`finalize()` sets the record to `complete`. It is a statement by the user that they are done, not a
lock: a run can be **reopened**, returning it to `open`. A scientist who finds an error after
finalizing should fix it rather than redo the labeling, and reopening is safe precisely because the
artifact is derived — it regenerates from state that was never thrown away.

The cost is explicit: for an authored run, `complete` means *the user said so*, not *immutable*.
Both events are recorded in `history.jsonl`, so a result can always be traced to the state of the
run when it was finalized. **`finalize()` also pins a snapshot** (below), which is what lets that
cost stay acceptable: the statement survives the reopen that is allowed to follow it.

### Snapshots — what leaves is frozen

The artifact is reprojected on every edit, which is what makes a cutoff or a label cheap enough
to iterate on. The cost is that `run_id` names a **mutable slot** rather than a result: a comparison
that read `an_15` last week and one that reads it today can disagree, both well-formed, with nothing
failing. `reopen()` is a supported workflow, so this is not a case to forbid — it is one to record.

A **snapshot** is a frozen copy of the artifact under `snapshots/<snap_NN>/`, taken when a number
*leaves*:

```
 edit a cutoff    ─┐
 relabel nodes     ├──► nothing. ever.
 scrub clips      ─┘

 run a comparison ─┐
 export           ─├──► pin() ──► artifact hash == newest snapshot? → return it
 finalize         ─┘                differs?                        → write snap_NN
```

- **Editing never pins, and neither does display.** The BROWSE tables and the CLI status read the
  working artifact through `artifact()`; pinning there would fire on every click.
- **Minted by difference, not by action.** `pin` hashes the artifact and returns the existing
  snapshot when nothing changed, so exporting twice yields one. This is the whole defence against
  a pinned directory becoming the `final_v2_REAL_final` pile that copy-on-every-action produces —
  the failure mode the notebook-versioning studies document, where what users actually asked for
  was not a history browser but the parent link a plain copy loses.
- **Copied, not re-derived.** Storing only the authored state and rebuilding on demand is far
  smaller and silently wrong: a later code change then yields different numbers for an old result,
  which is the failure this exists to prevent and which this repo has shipped twice (the
  label-leakage fix, the DTW pair-count fix). The **fit is never copied** — the distance matrix and
  the zarrs stay where they are, which is where the size actually lives.
- **`snapshot.yaml` is written last**, so it is the commit point: a directory without one is a
  half-written snapshot that discovery skips, the same rule `storage/comparisons.py` uses.
- **The audit travels with it.** A snapshot copies `audit.csv`, because an audit is evidence about
  *one* artifact state (§19); precision computed against a later reprojection is a different claim.
  Adding a decision does **not** mint a snapshot — it does not change what the instances are.

The record carries `reason` (`comparison` / `export` / `finish`), `parent`, `content_hash`,
`code_sha` and the authored `state` verbatim, plus the **frame view** so a snapshot can be
*measured* and not merely read — §23's denominator is not regenerable from `instances.csv`, because
`write_frame_views` needs `analyzed_t_ns` from the fit.

**Which** consumer used a snapshot is deliberately *not* stored on it. An export has no stable
identifier to record (its path lies outside the experiment, and an absolute path in an experiment
file is a hard-rule violation) and a comparison id does not exist until after the pin, so the link
is **derived** by scanning `comparisons/` for `source_snapshot` — a stored copy of a fact checkable
from the store is the second writer this section exists to prevent. That is what makes a list of
snapshots readable: not "version 7" but "the one the dose comparison used".

**On screen**, snapshots are a table on the run's own landing page (§16's analysis run page:
the run's facts, then a card per result view) — beside the status they qualify, not inside the
workspace, because a snapshot is a property of the *run* rather than of the
interpretation being authored. `analysis snapshots` is the CLI equivalent, and both ask the run —
`AuthoredRun.snapshot_interpretation` — so the two cannot disagree about what a snapshot was.
A run that has pinned nothing shows no table at all — an empty one would suggest something is
missing when nothing has left yet.

**The row carries the vocabulary.** The first cut paired a short row
(taken · why · granularity · named · used by) with a dialog, and driving it showed the dialog had
nothing to add: it repeated the row and added a code hash. What a person is actually trying to
recall is *which interpretation this was* — so the names go in the row, deduplicated because §22
makes many-to-one naming normal, and the row answers the question without opening anything.

**Opening a snapshot is a mode, not a page.** A navigable cue is earned by a destination
(§3 of `docs/gui_design.md`), and the destination is the interpretation itself, read-only.
Double-clicking a row leaves the run page where it is and re-renders it as that snapshot: a banner
naming it, the snapshot list with the current row marked, and the view cards opening the workspace
driven by `AuthoredRun.view(snapshot_id)` — whose full-window row says `Read-only · <snapshot>`. There is no second read-only screen, because
`gui/cluster_workspace.py` is already generic over the run handle — a parallel screen is what would
let a frozen result and a live one drift while both look right. The mode is **page state**, not a
click-handler local: `_build()` re-renders a shown page, so an incidental refresh would otherwise
drop the user silently back to the live run — same layout, different numbers, no event.

`SnapshotViewMixin` is the whole of it. Every read on an authored run resolves through `state()` and
`artifact()`, so overriding `_read_state` (the record's `state:`) and `artifact` (the snapshot's own
CSVs) re-renders the entire interpretation, and nothing downstream knows a snapshot exists. It is a
subclass rather than a wrapper because a delegating wrapper hands back the run's bound methods,
whose `self.state()` is still the run's. The **fit is read from the run**, not the snapshot: it has
one writer and a re-fit mints a new `an_NN`, so copying the largest file in the run directory into
every snapshot would freeze something that cannot move.

**A view refuses every write, and the writes are declared.** `AuthoredRun._WRITE_METHODS` names them
and each calls `_refuse_if_read_only` first. The list is necessary because `finalize`, `pin` and
`wait` do not go through `_edit` and all write to the *run's* directory — so an unguarded one would
change the live run rather than the copy on screen, and the damage would not even be local to the
thing being looked at. `wait` was found by the scan in `tests/api/test_snapshot_view.py`, not by
reasoning about it, which is why the list is enforced rather than remembered.

**Read-only is not `status`.** A finished run is still editable (finalize is a statement, not a
lock); a snapshot never is. `editable` carries that separately so a greyed control can say which of
the two reasons applies — one is undone by Reopen, the other by going back to the live run.

**Selection is not an edit.** Rotating the scatter, picking a cluster, clicking an ethogram band and
playing a clip stay live on a snapshot. Only what changes what the snapshot *is* goes dead. The live
**Measures** panel is hidden rather than greyed while a snapshot is shown: its numbers describe the
run, and live numbers under a snapshot banner are the confusion this whole section exists to
prevent.

**The module says what a snapshot *was*; the surface never assumes.** `snapshot_interpretation`
returns the columns, the cells and the one-line headline, and the table, the banner and
`analysis snapshots` all render whatever comes back. Clustering answers with its labels (a run
written before §22 dropped the granularity also shows the one it had), Query with its behaviors and their cutoffs. Hardcoding clustering's shape is what made a
fully calibrated Query run read `Clusters — · Named 0 · unnamed` — wrong information rather than
missing information, on a run that was completely interpreted, with nothing failing because a blank
cell is a plausible cell.

**Every authored module reads its snapshots back.** Query was left out at first, on the grounds
that a strip of clips is not a vocabulary. That missed the part that matters: a Query snapshot's
interpretation is its **cutoffs**, and "which cutoff did I ship?" is exactly the question a pinned
result exists to answer. `_QueryWorkspace` takes a run *handle* rather than a run id for the same
reason `ClusterWorkspace` does — building the handle inside a surface is what stops that surface
rendering anything but the live run. `snapshot_view_class = None` remains a legitimate answer for a
module that authors nothing, but it has to be a decision;
`tests/api/test_snapshot_view.py` drives over `_AUTHORED_RUN_ACCESSORS` so a fourth module cannot
inherit the default silently.

**No version tree.** Lineage is stored (`parent`) but not yet surfaced anywhere; it is not a graph
to navigate. In the only controlled study of a comparable feature 2 of 12 analysts ever opened one, and
VisTrails — the scientific-workflow tool built around an explicit version tree — was abandoned in
2018 with no published evidence that scientists navigated it.

`storage/snapshots.py` owns this, and takes `code_sha` and `state` as arguments rather than reaching
for `analysis/provenance.py`, so `storage` keeps its one-way dependency.

### `history.jsonl` — provenance, not data

An append-only record of every authored operation and when it happened: the cutoff was set to 0.4,
cluster 7 was named "rearing", an audit of 50 clips was started, the run was finalized and later
reopened.

- **It answers "how was this number arrived at,"** which is a real question for a published result
  and which the current-state file cannot answer — a state file says the cutoff is 0.4, not that it
  was 0.7 an hour ago.
- **It is explicitly non-authoritative.** The state file is the truth; history is narrative. That is
  a deliberate exception to the one-writer rule and is named here rather than hidden, because the
  alternative — deriving current state by replaying the log — is the reducer design, which is
  deferred (below). Nothing may read `history.jsonl` to decide behavior.
- **It records operations, not rows.** "Audited 50 clips for rearing", not fifty entries: the
  individual decisions are already primary data in the audit log, and duplicating them would
  create the dual writer this section exists to prevent.

### Surfaces

Each authored module ships the full set (surface parity), and two behaviors are **required of every
one** because they are shell-level rather than module-level:

- The result view **freezes while a compute operation runs**, so nobody authors against stale
  results. This is a correctness rule, not a courtesy: labeling against superseded candidates
  silently corrupts the decisions.
- **Refine is suppressed.** An authored run iterates in place rather than seeding a new run, so the
  affordance means nothing and implies immutability that does not hold.

Everything else — what the operations are called, how they are exposed — is the module's, and the
CLI keeps its usual split: headless launch, status, finalize/reopen and state edits, with
interactive review GUI-only and recorded in `docs/cli.md`'s omission ledger.

### The workspace: three generic tabs, one module tab

An authored run that emits the §19 artifact has a workspace of exactly this shape. The "tabs" are
the run page's **views** (§16 *The analysis run page*): each opens full window from a card on the
run's landing page, and a row of view tabs above it switches between them — the workspace holds
them in a tab widget whose bar is hidden (`gui/result_views.MultiViewWorkspace`).

```
 a MultiViewWorkspace (_QueryWorkspace, ClusterWorkspace)
 ┌───────────────────────────────────────────────────────────────────┐
 │  [ entity selector — behaviors, or nothing ]                      │
 │  ┌────────────┬───────┬────────┬─────────────┐                    │
 │  │  MODULE    │ AUDIT │ BROWSE │ MEASURES    │                    │
 │  └────────────┴───────┴────────┴─────────────┘                    │
 │       ▲            ▲       ▲          ▲                           │
 │       │            └───────┴──────────┘                           │
 │  supplied by            generic — read only the artifact          │
 │  the module                                                       │
 └───────────────────────────────────────────────────────────────────┘

   Query (§20)          MODULE = CALIBRATE   confidence curve, cutoff, marginal clips
   Temporal clustering  MODULE = INTERPRET   node tree, filters, montage, labels
   Pose clustering      MODULE = INTERPRET   the same — design only; the module is not built (§22)
```

- **AUDIT** — the §19 reliability audit: blind, randomized, budget-picked, precision with its
  interval. The same statistics wherever it appears, because it estimates precision of a *named
  label* and does not care what produced the label; the layout differs — a query's is scoped by
  its behavior selector (`gui/audit_tab.py`), a clustering run's is one table of labels and
  clusters (`gui/label_check.py`).
- **BROWSE** — the accepted instance set: sort, click to play (a per-recording span timeline was
  specified and is not built). Read-only. On a clustering run it is the **labelled** set (§22,
  *AUDIT and BROWSE on a clustering run*).
- **MEASURES** — §23's whole-recording measures of this artifact, a tab of its own on live runs
  (`pages.MeasuresPanel`); the run page shows the panel directly only for a producer whose result
  holds none.
- **Differences are not a tab of the run page.** The difference screen (§22, *The difference
  screen*) is one of the two tabs of INTERPRET's filter dock: a saved §24 node comparison, ticked,
  colours the tree by its statistic while that tab is open and greys the clips of the rest, one
  click from the montage. The durable record is the comparison itself, in the experiment's
  Comparisons section. A DIFFERENCES tab beside INTERPRET was specified here and never built; the
  dock is the screen it was meant to be, and it already has the properties the tab needed (never
  hides, shades on a fixed scale, next to the clips).

**Three of the four tabs are generic, and not by resemblance.** They read only the artifact and
`metadata.csv`, which is the same fact that makes measures (§23) and comparison (§24) generic. The
GUI therefore inherits the property the artifact exists to create: **a module added later gets
Audit, Browse and Measures by emitting the artifact**, not by building three tabs. *As built,
only Measures is one shared widget (`pages.MeasuresPanel`)*: each module still has its own Audit and
Browse widgets (a query's `AuditTab` / `_BrowseTab`; a clustering run's `label_check` and BROWSE),
which share the §19 statistics rather than a widget.

The module tab is where everything genuinely specific lives — a confidence curve, a node tree and
its cards. That is two things against three whole tabs.

**Shared widgets beneath them**, so no module builds one twice:

| widget | used by |
| --- | --- |
| clip grid — progressive fill, latest-wins per cell, decode-failure reason named | every module tab, and Browse |
| `_ClipPlayer` (exists) | all of them |
| blind review — randomized, score hidden until after, refuses a decision on an undecoded frame | Audit |
| instance table (a per-recording span timeline was the removed annotation workspace's; not built here) | Browse |
| per-recording strip — one small bar per recording | Query's confidence-by-recording, clustering's presence-by-recording |
| debounced threshold + live **recount** (`gui/confidence.py`, exists) | any threshold; a stored count cannot answer "how many at 0.7?" |

**Entity selection is the module's, and may be empty.** Query selects a behavior; clustering selects
nothing (its entity is the cluster, chosen inside the module tab). The generic tabs scope themselves
to the selection when there is one.

### Extraction targets

Recorded as expected shapes, **to be lifted once a second authored module exists** and the common
parts are visible rather than predicted, so the extraction is a planned step rather than a
rediscovery. The second module exists (Query and temporal clustering).

**`AuthoredRun(AnalysisRun)` has been lifted** — `api/authored_run.py`: status read from
`analysis.yaml`, the state-edit path (`_edit`: one lock per run directory, rewrite, reproject),
`finalize()` / `reopen()`, `pin()` and the snapshots, and the audit operations. There is no
operation dispatcher; a subclass supplies `_read_state`, `_write_state`, `_log` (its history) and
`_project`. The two below are **not built** and are still expected shapes.

- **`Experiment.run_authored(module_id, config)`** — the launcher, with the initial compute job
  optional, since a module may open a run with nothing computed yet. Authored runs launch through
  `Experiment.run_analysis` today.
- **`MutableRunWorkspace`** — the GUI base for a stateful result widget that triggers compute: the
  freeze, the progress screen, re-enabling on terminal status. Its **tab split is specified above**
  rather than deferred, because the three generic tabs are generic by a property of the data (they
  read only the artifact) rather than by two surfaces happening to look alike. What stays an
  extraction target is the base class itself — the freeze and progress plumbing. Today
  `_QueryWorkspace` and `ClusterWorkspace` share only `MultiViewWorkspace` (the hidden-bar view
  switching, §16), and no authored run has a compute operation to freeze for (*Three kinds of
  operation*).

The trigger is the second implementation, not a date. If two authored modules are built in
parallel by different people, extract earlier — divergence happens fastest then, and a shared base
is worth more than its risk at that point.

### Guards

- **No dual writers**, data-driven over authored modules: a `<module>.yaml` record carries no
  status key and no derived counters. This cannot be caught behaviorally — both files stay
  well-formed while disagreeing — so it is a scan over what the writers emit.
- **Sequential compute**: submitting an operation while one is running raises rather than queues.
  (Nothing to drive yet: no compute operation is built, *Three kinds of operation*.)
- **`open` yields an artifact, `partial` does not.** Assert the files, not that the call returned.
- **Finalize/reopen round-trips**, and the artifact after a reopen-and-refinalize matches what the
  state says.
- **The status gate reads the flag**, not a list of status strings — the `tests/gui/test_editing.py`
  idiom, so a new authored module is covered the day it registers rather than when someone
  remembers to extend a tuple.
- **Crash safety**: a torn trailing line in an append log is tolerated on read, and the operation
  before it survives.

Every one of these guards the **contract**, which is what this section specifies. None of them
assumes a shared base class, so they stay valid whether or not the extraction happens.

### Deliberately not included

- **A reducer over the log.** Deriving current state by replaying `history.jsonl` would give undo
  for free and would remove the non-authoritative exception above. It is a larger design and is
  deferred; the log's shape is chosen so it stays possible. Label undo exists without it:
  INTERPRET's ⌘Z / ⇧⌘Z (`gui/label_history.py`) records each label write's before and after
  (`AuthoredRun.label_maps`) and undoes by writing the before side back (`restore_labels`). That
  history is the page's, in memory, and is not kept with the run.
- **Concurrent compute operations** and **multi-user editing**. Out of scope, and the sequential
  rule is what keeps them out. (Concurrent *state edits* from one user are serialized by a lock,
  *Compute inside an open run*.)


## 22. Behavior Discovery (pose and temporal clustering)

The §19 Task-2 modules: find structure nobody named in advance, and see how it differs between
groups. Where Query starts from a behavior the user can already name, discovery starts from nothing
and hands back a vocabulary for them to interpret.

**Two modules, split by what a cluster *is*.**

| | **Pose clustering** | **Temporal clustering** |
| --- | --- | --- |
| a cluster is | a posture | a behavior |
| an instance is | a run of frames holding it | a segment of movement |
| priority | first | second |
| status | designed; its protocol and an empty method registry are in the tree | built |

They are separate modules with separate method protocols rather than one module with a mode flag.
The shapes are identical enough that sharing is tempting, and that is the trap: a shared interface
would let a bout method be selected in the pose module, where it answers a different question.
Keeping them apart also lets each grow features the other does not need — the pose module's
vocabulary is something you *look at*, the temporal module's is something that *unfolds*.

**What already exists is sequence clustering, not pose clustering.** Every rule in
`segmentation.CLUSTER_RULES` cuts a trajectory into segments and clusters those. Even the research
"frame path" (`FRAME_RULES`, and the codebook → assign → decode arm) labels frames but run-length-encodes them into bouts and scores against
*behavior* labels — it was built as a comparison arm for behavior clustering. That is evidence that
frame-level labeling is feasible; it is not a design for pose clustering, where the postures
themselves are the object of interest.

### Three phases, the same shape as Query

Both modules are **authored runs** (§21):

- **Fit**: computed, expensive and immutable. It is whatever structure the method needs to compare
  things, and it always includes a full merge tree over its finest clusters (`ClusterTree`).
- **Node cache**: computed once after the fit and rebuildable. It holds the evidence the
  interpretation screen shows for every node of that tree (*The node cache*, below). It is a
  derived view, not an authored decision: it depends on the fit and the tracks, never on a
  label.
- **Labelling**: authored, cheap and instant. A label goes on a **node** of the merge tree. This is
  the analogue of Query's cutoff.
- **Artifact**: the §19 behavior output, reprojected on every label change.

**There is no granularity.** The first version of this module had one: a knob choosing how many
clusters the vocabulary had, a table of the clusters at that cut, and names filed per setting. Names
then moved onto tree nodes, so that a name followed its branch at every setting, and at that point
the cut stopped carrying any decision. The partition a user means is the one their labels draw: a
label on a coarse node covers everything under it, a label on a finer node inside it is more
specific, and the rest is unlabelled. The knob remained as a second, competing way to pick what to
look at, and the screen built around it was reported as too busy and too slow. Users land on the
screen to find behaviors and label them, and the unit they reason about is a cluster, which is a
node. So the node is the unit, and the cut is gone. A run written before this still reads
(*Runs written with a granularity*, below).

### The method protocols

One protocol per module, with the same shape, mirroring `QueryMethod` (§20). The pivotal member is
the same too: **the method turns a partition of its own tree into instances.**

```python
class PoseClusterMethod(ABC):            # and TemporalClusterMethod: one per module, not shared
    method_id: str
    config_model: type[BaseModel]
    granularity: GranularitySpec          # still declared; the screen no longer reads it (below)
    has_noise: bool                       # does this rule leave members unassigned
    supports_apply: bool = False          # reserved; see below
    shard_suffix: str                     # the extension `serialize` writes; no default
    instances_are_runs: bool = True       # may adjacent instances with one label be fused
    requires: tuple[str, ...] = ()        # optional-extra modules, checked at validate

    def fit(self, trajectories, params, progress_cb=None) -> FitResult: ...
    def hierarchy_of(self, fit) -> ClusterTree: ...
    def member_leaves(self, fit) -> np.ndarray: ...          # each member's leaf
    def member_places(self, fit) -> MemberPlaces: ...        # where it sits, its measured time
    def rank_members(self, fit, leaves) -> list[tuple[Instance, float]]: ...
    def clip_scoring(self, fit, params, trajectories=None) -> ClipScoring: ...  # node cache
    def node_separations(self, fit) -> np.ndarray: ...       # one per node
    def instances_for(self, fit, leaf_groups, params) -> list[Instance]: ...
                                         # temporal adds member_groups=None
    def instances_at(self, fit, granularity, params) -> list[Instance]: ...  # old runs
    def resolve(self, fit, instance) -> dict: ...            # temporal only
    def fit_summary(self, fit) -> dict: ...                  # temporal only
    def subsample_note(self, fit) -> str | None: ...         # temporal only
    def assign(self, fit, trajectory, params) -> list[Instance]: ...   # only if supports_apply
    def serialize(self, fit, path) / load(self, path): ...
```

(`core/protocols.py`. `TEMPORAL_CLUSTER_METHODS` holds `codebook_viterbi` and `dtw_finch`;
`POSE_CLUSTER_METHODS` is empty — the pose protocol is design, not built.)

- **`FitResult` is opaque to the module**, as `SearchResult` is in Query. A codebook, a distance
  matrix or a neighbour graph: the method owns its format and its serialization.
- **`member_leaves` is membership without a cut.** For every member the fit labels (a frame for
  pose, a bout for `dtw_finch`, a frame for `codebook_viterbi`), it gives the member's finest
  cluster, which is its **leaf** of the tree. A node's members are the members whose leaf lies
  under it. No granularity is involved, so any node can be asked about. **`member_places`** gives,
  in the same index space, where each member is (recording, subject, start and end `t_ns`) and the
  time it stands for, measured (*Member time*, below).
- **`instances_for` is where the method's own decisions live**: the temporal decode, smoothing, and
  the minimum duration below which a run is not an instance. `leaf_groups` maps each leaf to the
  node it is reported under (or `-1`, unlabelled): a partition of the method's own tree, exactly
  the kind of thing a granularity cut was. **It carries no names.** The module resolves labels to
  that partition (deepest labelled ancestor per leaf) and maps node ids to names afterwards, so a
  method is still handed trajectories, config and its own tree and nothing else, which is what
  `tests/cluster_methods/test_registry.py` keeps structural (CLAUDE.md, "Labels score a method;
  they never build one"). A frame-level labelling flickers, and every flicker would otherwise become an
  instance; suppressing that is the method's business, not a module-level cleanup pass. (This is the
  same division as in Query, where extent belongs to the method.) A run-length method
  (`codebook_viterbi`) encodes runs of equal label inside its continuity blocks and
  applies `min_run_s`; a bout method (`dtw_finch`) relabels its fixed bouts.
- **Cadence and continuity blocks are measured per unit — one recording and one subject — never
  over the pooled rows.** A fit's rows are every trajectory's concatenated, and every recording's
  clock starts at zero, so the median step of the study's timestamps *sorted together* is the
  jitter between recordings rather than a frame period, and a gap rule over the pooled rows never
  cuts at a seam (a step *back* to zero is not a gap). On a large study `codebook_viterbi` read a
  cadence far above the true frame rate, which turned its `min_run_s` into thousands of rows:
  the only runs that long crossed a recording seam, `resolve` rightly refused them, and the run
  kept no instance at all. `_unit_seams` / `_unit_blocks` cut per unit before any gap rule —
  an earlier frame method already did this, and the fix copies its shape. The guard,
  `tests/cluster_methods/test_pooled_timelines.py`, is data-driven over
  `TEMPORAL_CLUSTER_METHODS`, so a new temporal method is covered the day it is added. A fit made
  before the fix keeps its fit: the decode ran per trajectory and is unaffected.
- **`rank_members` makes a node interpretable.** A user cannot label a cluster they cannot see, and
  which members best show what it is depends on how the method grouped them: medoid for a
  distance-based rule, nearest centroid for a codebook. It orders **every** member under a set of
  leaves from most to least typical, as `(instance, distance)` with smaller more typical in every
  method, which is what the *most typical* and *least typical* views show.
- **`node_separations`** says how different each node's two children are, in the method's own
  distance: the distance between the children's centres over their mean within-child spread,
  for every node in one batch (it is additive, so this is cheap). It is shown
  beside the run's median so it has a reference (*The node cache*).
- **`assign` is reserved.** Applying a fit to new recordings is out of the MVP (below), but declaring
  the capability now costs nothing, whereas retrofitting a base contract later is not free.

**Retired with the granularity:** `member_order`, `representatives` and `projection`. Each took a
granularity, and each is **removed** from both protocols and every method, with the `Projection` dataclass and the fit-time
UMAP/PCA layout that fed the scatter (the 3-D view had already been dropped from the screen, because
the tree says "what merged with what", which a scatter never did). Their two settings,
`projection_points` and `projection_layout`, are gone from every method's config;
`resolve_method_params` drops them silently (`RETIRED_PARAMS`) so an old run's `method_params` still
validates, and each method's `load` ignores the `projection_points` / `projection_layout` / `coords`
an older shard carries. **`instances_at` is kept**: old runs were written through it and still read
through it, as `instances_for(fit, tree.leaf_nodes_at(g), params)` relabelled to cluster ids, and the
*Old and new agree* guard below needs it to hold the two together. **`granularity: GranularitySpec`
is kept too**, still a required attribute of both protocols and declared by both methods (each also
has `granularity_of(fit)`); nothing on the screen reads it, and the manifest's `method` block still
records the state's `granularity` for a run written with one.

### Labels, and the three label states

A label is authored and has the shape §19 already specifies: `source_label` carries what the output
row was resolved to, `label` the user's name, and `label_path` the chain of names above it. What
discovery adds is that **labels nest**. A label goes on any node of the merge tree. Each member
resolves to its **deepest labelled ancestor** (`ClusterTree.deepest_named`), so "not moving" over a
whole quiet branch and "resting" on one branch inside it are both kept. A resting frame reads
`label = resting`, `label_path = not moving/resting`; a quiet frame outside "resting" reads
`label = not moving`. A name may not contain `/`, which separates the levels.

**A label is a stored thing, and nodes point at it** (`storage/run_labels.py`). State holds
`labels` (id → name; ids `L1`, `L2`… never reused) and `node_labels` (node → id); the per-node
`node_names` map is derived. Several nodes carrying one label are **one label**, not matching
strings: the output groups by label (`authored_run.label_groups`, one `instances_for` group per
label), so `source_label` is the label id, and two clusters of one label alternating make the same
bouts as their common parent — for a method whose instances are runs (`instances_are_runs`:
`codebook_viterbi`); `dtw_finch`'s bouts are segmenter cuts and stay as cut. A rename changes no id, so it moves no instance and no audit
judgment. Three rules, held by `check_labels` on every edit before anything is written, refusing
rather than repairing:

- a name is unique within the run, ignoring case;
- **every node of a label sits under one parent label** — nesting still comes from the tree, and
  this is what makes a label's path a property of the label;
- a label is never inside its own branch.

The edits are `name_nodes` (by name: an existing name adds to that label; a label's only node given
a new name renames the label), `create_label`, `label_nodes`, `rename_label`, `merge_labels` and
`delete_label`, all through one door (`LabelOperations._label_edit`) that folds an older state,
applies the change, checks the rules and recomputes the resolution. `edit_labels` applies names,
renames and merges as **one** edit, each name read against the labels as the edits before it left
them: the CLI's `--name-node`, `--rename-label` and `--merge-label` run through it, so a command
whose last pair names no label writes nothing (#24; one write per pair had left the earlier pairs
written). The CLI has `--name-node`,
`--rename-label` and `--merge-label`; INTERPRET's label box lists **Add to an existing label**
(greyed, with the reason, where the parent rule forbids it) and its labels list has Rename… and
Merge into….

**Older states.** One written with per-node names reads, fit-free, as one label per distinct name,
so an old snapshot still opens. The next edit, which has the tree, splits a name used under two
parent labels into "name" and "name (2)" and drops a label repeated inside its own branch, and
records what it changed in the edit's history entry. Moving `source_label` from node id to label id
changes every instance id once, so audit judgments recorded before that stop applying, and the
coverage floor reports it.

A member has no label for one of two different reasons, and the difference is meaningful:

| state | `source_label` | `label` | means | shown as |
| --- | --- | --- | --- | --- |
| **labelled** | the label's id (`L3`) | the name | interpreted | its label's colour |
| **unlabelled** | `unlabelled` | empty | a real cluster member nobody has labelled yet | a distinct neutral |
| **noise** | reserved | empty | the method assigned it nowhere | greyed |

**Unlabelled is one bucket.** Unlabelled members keep no finer identity in the output: without a
granularity there is no honest choice of which cluster id to give them, and hundreds of leaf-sized
instances would bury the labelled ones. Unlabelled runs are still emitted as instances, so coverage
and the measures see them. The measure that reports them keeps its id, `unnamed_time` (§23), because
runs on disk carry it.

No schema change: this falls out of the columns §19 already has, plus reserved values.

**Measures report unlabelled and noise separately.** "12% of observed time is noise" and "31% is in
clusters you have not labelled" are different facts about a run's completeness, and collapsing them
into "not attributed" hides which one you are looking at. A method with `has_noise = False` reports
full assignment by construction, and must not be read as more complete than one that admits it
leaves frames out. Every method today declares `has_noise = False`.

### The quality signal

**Precision per label**, exactly as in Query, and the machinery is §19's rather than this section's:

- **After labelling, and on labels only.** An unlabelled node makes no claim, so there is nothing for
  it to be right or wrong about. A labelled one asserts "these instances are rearing", which is a
  claim a human can check.
- **The label, not the node, is the unit.** A user who labels four nodes "rearing" cares whether
  *rearing* is right; the internal coherence of one node is a step along the way.
- Blind, randomized, sampled across the confidence range, appended on re-audit, reported with its
  interval: all inherited.

**Per-node coherence is a labelling aid, not a result.** The label bar's split separation and
the least-typical clips tell a user whether a node is one thing and therefore whether to label it.
That is useful during interpretation, and it is not a number that belongs in an output.

### INTERPRET: the interpretation tab

The workspace is §21's: three generic tabs (Audit, Browse and Measures) plus one module tab. For
both clustering modules that tab is **INTERPRET**. A user arrives there to explore what the fit found
and to label the behaviors they care about, so the screen is built around one loop: **click a node,
decide whether it deserves a label, label it or move on** (to a child, if the node is too broad, or
to another branch).

```
┌─ Analysis / an_32 · Behavior clustering · open ──────────────────────────────────────────┐
│  INTERPRET │ AUDIT │ BROWSE │ MEASURES                                                   │
│ Clips that match         ┃ ■ rearing 14 min                          [−][+][Whole]       │
│ 0% [░▒▓█ ramp|run] 100%  ┃ ████████████████████████████████████████ (shade = share)      │
│ 212 of 1,004 clips match ┃ ▔▔▔▔▔▔▔ ████████████ ███████████ ▌▌▌▌ (thin = 1 px columns)   │
│              [Reset all] ┃ ┌───────────────────────────────────────────────────────────┐ │
│ ▾ Location 1 region [Rst]┃ │ Behavior #41 [rearing]  ■■ [Typical▾][Show▾]─○[Export▾]   │ │
│ ┌──────────────┐         ┃ │ 214 clips · 3.1 min · 4.2% of the run  Labelled · Saved   │ │
│ │ ░▒▓ ┏━━┓ ░▒  │ box     ┃ │ [ name… ][Label][Remove][Labels▾] [Undo][Open in window]  │ │
│ └──────────────┘         ┃ └───────────────────────────────────────────────────────────┘ │
│ In use                   ┃ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  every clip, each looping    │
│ ▾ Video A01 only [Reset] ┃ │▶ ││▶ ││▶ ││▶ ││▶ ││▶ ││▶ ││▶ │  on its own clock; a clip    │
│   [✓] A01_drug  182      ┃ └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘  outside the filters: grey   │
│ Other fields             ┃                                                               │
│ ▸ treatment     3 values ┃                                                               │
│ ▸ weight_g     31 values ┃                                                               │
│ animal               S01 ┃                                                               │
│ sex                    F ┃                                                               │
└───────────────────────────────────────────────────────────────────────────────────────────┘
(the ┃ is the filter dock's grip: dragged shut, the dock is a strip reading
 "Filters · 1 active"; ■■ is the match key, shown while a filter is on)
```

**The tree** is an icicle. Every node is a block whose width is its member time; the root is at the
top and each row is one level down. **A lopsided merge is not a level**: real fits are mostly chains,
one cluster shedding a piece per merge, and drawn merge by merge that is a staircase of identical
bars. A merge whose smaller half holds under 20% of it (a *peel*) gets no row; the pieces it is made
of stand in for it, so what a chain shed sits side by side in one row, largest first. A chain's own
nodes stay clusters, reached with the arrow keys and AUDIT's *Open in INTERPRET*, and when one is
selected its drawn pieces are outlined. **Every block is followed by a 1 px gap** to its right and
below it, in one dark neutral (`node_icicle.GAP_COLOUR`, both themes, so it stands apart from the
light end of the match ramp): edges round to whole pixels, a 2 px slot keeps 1 px of colour, and a
slot under 2 px is 1 px of colour with no gap of its own. The selection is a two-tone ring, 2 px dark
outside 1 px white, inside the block's edge, or round the outside of a thin column. Blocks under 6 px
paint as bare columns in their own fill (their share's shade while a filter is on, so a small
cluster shows without zooming), with no text, and stay individually clickable; only children under 0.5 px,
which are not laid out, show the faint strip behind them. A hover names any block. **Unfiltered it is coloured by labels**: solid where a label was
put, pale where a parent's label covers it, neutral where nothing does. A dot marks nodes already
looked at in this session; no block is marked as lacking clips, since every node lists them (*The
node cache*). Clicking a block selects it. **The tree is a viewport.** Rows are a fixed height and
every level is laid out, so a taller pane shows more levels and a vertical scrollbar (or
Shift+wheel) moves down them. The horizontal zoom is a scale (pixels per root) rather than a share
of the width, so a wider pane shows more of the run at the same zoom. **The tree zooms**: the wheel
(or a trackpad pinch) scales the visible width around the pointer (the tree claims `OWNS_WHEEL`,
since the page scrolls), a drag pans, and a click is told from a drag on release; blocks are laid
out at their on-screen width, so a zoomed-in sliver becomes a labelled block with its own rows.
While zoomed an overview band shows the whole width and the visible window, and moves it. A block's
right-click menu is **Label…**, **Remove label**, **Open in window** (the cluster's pop-out
window, *Cluster windows* below; offered only while the page connects `window_wanted`) and **Zoom
to this**; double-click labels it; **Whole tree** / Esc fits. Arrow keys walk
the tree as drawn (↑ the block above, ↓ the largest block below, ← → along the row), **L** focuses
the label field, **U** removes the node's own label and **1**–**9** apply the label bar's recent
labels. Nothing about the tree moves when a label
lands, so labelling never costs you your place.

**Find clusters** (`gui/cluster_find.py`; on screen "Find" and the page's `WorkspaceTerms.units`,
"Find behaviors" on a temporal run), in the tree's zoom row between the label key and the zoom
buttons (no row of its own). **One search at a time**: the user types one entry and presses
Enter (`cluster_find.parse_entry`): a node id (`1321`, `#1321`), or else a label, matched to each
node's own label whole and ignoring case (`_nodes_found_by`); a new search replaces the last.
Its nodes' blocks wear a band in one colour just outside them (`NodeIcicle.set_found`), on each
piece where one is a chain node; never inside a block (a side at the widget's edge goes
without) and never on the zoom root, which is the whole picture. The colour is the first
`plot_theme.identity` slot the labels do not fill blocks with (`set_reserved`). A chip beside the
field shows it, with a count for a label on several nodes ("climb · 3"). The tree re-roots on the
smallest cluster holding every found node (`cluster_find.common_cluster`): the node itself for
one. An entry that names nothing is a chip saying so ("not in this run", "no behavior has this
label"), with no colour, no band and no zoom. A label change re-reads the search
(`FindClustersBar.refresh`, emitting `rings_changed`): the bands follow and the tree does not
move. Clicking the chip selects its node (a label's largest); **Clear** ends the search and puts
back the view the tree had before the first search zoomed it (`NodeIcicle.view_state` /
`restore_view`). Nothing is filtered, labelled or written. A search lasts the page's life and is
cleared by a re-fit, which makes every id name a different cluster (`tree_identity`).

**Layout** (`gui/node_workspace.py`). A horizontal splitter: the filter dock
(`gui/cluster_filters.FilterDock`) on the left, which folds by the splitter's grip
(`gui/grip_splitter.GripSplitter`, as every side panel does) to a strip reading "Filters", or
"Filters · 2 active" while two filters are on (`cluster_filters.panel_name`), and drags down to
240px, the location map filling its width (its width and folded state persist under `QSettings`
`panes/interpret_filters/`; Location is a line that folds like a field's, open or folded kept
under `interpret/filters_location_open`), and on the right, top to
bottom: the tree with a drag handle under it, as short as 120px
(the house rule for stacked sections; its height persists in `QSettings`) under a one-line key
(`gui/match_key.TreeKeyRow`: each root label's swatch and time on the left, the match ramp on the
right, the zoom buttons at its end; how to use the tree is the row's tooltip), the **label bar**
(`gui/label_panel.LabelBar`, review round 9, option A: between the tree and the clips, so labelling
sits where the eye goes from one to the other), and the montage (`gui/live_montage.LiveMontage`)
taking the rest of the page. `cluster_filters.FilterBar` is not a widget: it keeps the selection
and builds the dock. **Rebuild clips** is a row above it
all shown only while the cache is stale (or from before every node listed its clips), beside the
warning, and a page reopened mid-build attaches before loading anything, so it never shows the empty
workspace under a loading overlay first. Gone with this layout: the node card, the labels panel's
histogram and Ethograms band, and the clip-montage window (*Clip montage* below); the selected
cluster's ethogram lane moved to BROWSE.

**The label bar** (`gui/label_panel.LabelBar`) is three lines in a frame (laptop layout and user
decision A, 2026-11-09: three rows and the montage's own controls row had cost 141px between the
tree and the clips). The first is the open cluster's name ("Behavior #41") in a large bold font,
its label as a large chip in the label's colour (`LabelChip`: filled for its own label, outlined
as *inside “X”* where a parent's label covers it, "not labelled" in muted text otherwise), and at
the line's right end the montage's toolbar (`LiveMontage.detach_controls`, seated by
`LabelBar.add_trailing_widget`: the match key while filtering, then **Most typical ▾**, **Show
▾**, Size and **Export ▾**). The second is short: the figures at its left (clips · time · share
of the run) shortened with "…" to the room left and whole in their tooltip
(`sizing.ElidedLabel`), which also carries the separation against the run's median and, when the
cluster beats its floor, where it is concentrated (`node_workspace.cluster_facts`); at its right
the status — what the last write did ("Labelled #41 “rearing”") and then the write queue's
"Saving…" / "Output up to date." (`LabelBar.status`). Under it the AUDIT line when AUDIT has
judged the cluster. The third is **the label field** (`gui/label_panel.LabelField`), which types a
label onto the selected node: a large field that stretches, then **Label** (the style's default
button, in a larger bold font; Enter in the field submits it), **Remove** and **Labels ▾**, which
leads with the recent labels (`QuickLabels`, no button of its own), up to nine, "1  rearing",
each with its colour swatch, most recently used on this page first and then every label by its
time; the number key (on the INTERPRET tab, not while a text field has focus) sends the same
request as typing the name and pressing Enter, the menu entry fills the field, and a label the
node may not join is greyed in the menu with `label_choices`' reason. Hidden while the run has no
labels. After **Labels ▾**, **Undo** and then the page's actions on the cluster
(`LabelBar.add_action_widget`: INTERPRET's **Open in window**).

**The column fits the window.** A view open full window is held to it, not scrolled
(`AnalysisRunDetailPage.minimumSizeHint`), so the tree's height is computed from the other rows'
size hints (`NodeInterpretTab._fit_tree`), never from heights they were given: it shows every level
while the montage keeps two rows, gives height back down to 120px before the montage loses them, and
a dragged height is held to the same room without being forgotten. The montage's floor is one row.
Guarded at 1440×847 by `tests/gui/test_laptop_layout.py`.
All of it goes through the page's one write queue (`_label` / `_write_next`). The field
(Enter or **Label**; **Remove** its own; it completes from labels in use and says "Sub-label inside *X*" when
a parent's label covers the node). **Labels ▾ lists every label** with
its colour, its time and its share, the most specific label winning (`label_rows`: a labelled parent
shows its sub-labels nested under it, plus an *other* row for its time no sub-label covers). Each
entry's submenu goes to the label, renames it, or merges it (`merge_targets`): Rename and Merge's one
home now the histogram has gone. **Only Enter and Label write** (user, 2026-11-09): a click in the
field only places the cursor (it used to open this list, which took the first click); a click on a
label's name in the list — its row, or *Label as “X”* in its submenu, greyed with the reason
where the parent rule forbids it (`label_choices`) — and a type-ahead suggestion chosen with Enter
both only fill the field, selected, for Enter or **Label** to apply. The numbered recent entries at
the list's top fill the field the same way; their keys 1–9 still label at once. A double-click on a block opens an inline editor there.

**Filters shade; they never hide** (`api/cluster_filter.py`, *The node cache*). The dock's sections
make one `cluster_filter.Selection`, and every change emits at once, a drag at most every 50 ms
while it moves and once on release (`gui/throttle.py`): **Location** (`gui/density_map.RegionPicker`
over the run's position density: a box drawn live while dragging, kept with four corner handles to
resize, dragged from inside to move, a new one dragged from outside, a click toggling one cell;
greyed with the reason on a run with no positions), **Differences** (the dock's other tab, not
a line of this one: the run's saved node comparisons, §24, one tickable at a time, with what each
kept, its scale, its strictness and *New screen…*; ticking one is the leaves under its kept
clusters, `cluster_filter.CLUSTERS`, for the clips, and the tree's colour while that tab is open —
*The difference screen*), **Time in session**
(`cluster_filters.SessionRange`: the `RangeHistogram` with `bars=False`, a plain two-handle slider
over each recording's whole video in `SESSION_STEPS` = 100 steps, ticks *start · middle · end*, a
line *12%–40% of each recording · 840 clips* or *Whole session*; its line folds like Location's,
reading *any* or *12%–40%*, open or folded kept under `interpret/filters_session_open`; greyed with
the reason when no recording of the run has a timestamp map), **Video** (a searchable checklist of
recordings), then a line per metadata column, its values picked by what it holds: a checklist (clips per
value and "(no value)", a search box over 20 values, an "only" on the hovered row) for a category, and for a number,
date or time-of-day column (`storage.metadata.infer_column_types`) with at most 12 distinct *parsed*
values; a
`RangeHistogram` (the bars of clips per value with two handles dragged along them, the span in
the match colour, its from–to under the bars) for one with more. A column with a single value,
blanks not counted, is a greyed line at the bottom of the field list giving its name and that
value (its tooltip says there is nothing to filter); one value plus blanks keeps its checklist.
Field names are bold, with a thin line above each field, so the fields read apart (user, review
round 6). Whether a condition is values or a range is decided by its shape (a set or a
`(lo, hi)` tuple), never by the column's type, so a number checklist is stored as values.
OR within a column, AND across. A section's heading is the column's name only, never its type.
**A field's line is not a button** (`cluster_filters.FieldRow`, review round 8, option B): "▸ name"
in bold with its value count or its filter at the right ("2026-07-08 only", "2 of 3"); a click
anywhere on it, or Space / Enter with the focus on it, opens its values; the pointer tints it and
the focused line is outlined in the palette's highlight. While the field has a filter, a small
**Reset** at the line's right end removes it — checklists, Video and ranges alike; there is no
Reset under the values, and nothing else repeats what the line already says. Location's
**Reset** sits at the right of its heading, shown only while a region is picked; how to pick is
the map's tooltip. **Fixed above the scrolling sections, always** (`FilterDock.top_box`): the match
key (`gui/match_key.MatchKey`: *Clips that match* over the 0–100% ramp) and one line. With no
filter the key is greyed with no marks and the line is the run's clip count ("840 clips"); with
one, the key carries its marks and the line is "n of N clips match" (`clip_matches` at the root;
how fields combine, "A clip matches when it fits every field. Several ticked values in one field:
any of them.", is its tooltip, in both states) with Reset all at its right. The block is one
height in both states, so the fields never move when the first filter goes on (round 7 hid it
and the column jumped). Undoing a filter sits in the dock that sets it, not by the labels (review
round 4: undoing a filter outside the filter column read as part of the label field). Folded,
the dock's strip reads "Filters · n active", so an active filter is never out of sight; Reset all
is only in the open dock. The tree opens at the height that
shows every level (`NodeIcicle.whole_tree_levels`); when that would leave the montage under two
tile rows its rows shrink to fit (`set_fit_rows`, down to `min_row_px`) rather than scroll. The
tree **drags and does not fold** (a `ResizeHandle` under it, no `CollapsibleBand` header: user
decision 2026-11-07, pinned by `test_workspace_parity`). A drag of the handle is kept under
`interpret/tree_height_dragged` and wins from then on, with full-height rows. The montage's **Show ▾** (`gui/show_menu.ShowButton`, 2026-11-09; it replaced a *Show masks*
check box and an *As recorded ▾* menu) ticks the `TileLook` layers — Video, Masks, Mask trail,
Keypoints, one definition in `show_menu.LAYERS` — the orientation and the context (none, 1, 2,
5 s either side, `show_menu.CONTEXT_CHOICES_S`; the page widens its clips by
`live_montage.context_span`), kept per page (`interpret/montage_look` / `_context`,
`label_check/review_look` / `_context`, `interpret/cluster_window_look` / `_context`; the old
app-wide `interpret/montage_masks` read once to start from). Masks off asks
`ClipTiles.frame(..., masks=False)` for the plain picture, its drawn-tile cache keyed on the look,
and washes a non-matching tile out instead of greying its mask. While a filter is on, a matching tile
wears a 2 px frame in the mask tint (`live_montage.match_colours`, the one source for the tiles'
frames and the header's *■ matches filters ■ doesn't* key), masks on or off; with none on, no
frames and no key. A tile's loop line is drawn **under** its picture on a grid-coloured track,
never over the footage: on the footage's black foot the near-black fill vanished and the unplayed
remainder read as a bar running backwards (review round 4). While a selection is on, **every block is
shaded by its matching share** (`NodeView.match_shares`) on the fixed 0–100% ramp
(`plot_theme.match_colour`, the ramp's low end for none or NaN), and a block a label covers shows
a 3 px ink strip along its top in place of its label's colour (`NodeIcicle.fill_of`); the text ink
follows the fill. The key marks the selected cluster's share
with a tall bar and a triangle, labelled *selected 2.3%* above the ramp, and the run's with a
thinner line labelled *run 23%* below it. (A tick per cluster above the ramp was removed in review
round 3: hundreds of them read as a barcode, and the tree's shading already places each cluster.) **The scale never stretches**: an even 43–55% spread is one mid
shade, not a strong pattern.

**The montage** plays **every clip** of the selected node (`NodeView.clips`; for a method ranked on
demand, `run.clips` on a worker, memoised per node) in the order the switch asks — most typical,
least typical, random seeded by the node (`ClipTable.order`). Its tiles paint from this machine's
**video previews** only (`api/clip_tiles.ClipTiles` over `PreviewReader`: one `PreviewClip` per tile,
reopened only when the tile size or its match changes, at most `OPEN_CLIPS` open; a clip of at most
`WHOLE_CLIP_FRAMES` tile frames is drawn whole, in one forward pass, the second time it is asked
for a frame, so a playing tile does not seek back for each frame it skipped), on one picture
worker that serves the newest request per tile — the first picture of every visible tile before
any tile's second, then one row past the view in the scroll direction, and nothing for a tile
scrolled out of view (`LiveMontage.lookahead_indices`, `_PictureWorker.focus`). One `QTimer` at 15 fps advances each visible tile's
own clock, looping at its clip's length; only visible tiles play or decode. Hovering a tile pauses
the others; its tooltip is rank, recording, start, length, each condition's ✓ / ✗ and the clip's
position. **A clip outside the selection still plays, its animal drawn grey** (`ClipTiles`'s `GREY`
tint). The grey and the ✓ / ✗ come from one rule: the per-condition flags are
`clip_matches(selection.only(key), clips)` and the match is their AND, pinned by
`tests/gui/test_cluster_filters.py::test_every_condition_passing_is_exactly_a_match`. A double-click
opens the clip in `ClipPopup` from the source video, stepping through the montage's order.

**The label bar's first two lines** (the first was the montage header until review round 9) are
what survives from the card: the node's name, its label chip, clip count, member time and share;
its split separation against the run's median and, only when `NodeView.distribution` beats its
floor on some recording-level dimension, the strongest concentration with its numbers, both in
the figures' tooltip (they were on the line, and were what got cut at a laptop's width); and —
when AUDIT has judged clips of it — `NodeAuditBlock` as a line of its own, its Move to… / Remove
from the label / Look at its halves in one **Act on it ▾** menu. Since
user decision A (2026-11-09) the bar is three lines: the name, the chip and the montage toolbar;
then the figures at the left of a short line of their own and the status (`LabelBar.status`) at its
right — on the toolbar's line the figures kept 179 of 217 px at 1440×847 with the navigation and
the filters open, and on the field's line the status 80 of 262; then the field, **Undo** and
**Open in window**.

**One montage toolbar everywhere** (2026-11-09, option A; `gui/montage_export.py`). INTERPRET's
montage, the cluster window and AUDIT's review grid carry the same controls — **Most typical ▾ ·
Show ▾ · Size · Export ▾** — because `LiveMontage` builds all four, with `export=` (an
`ExportSource`: the grid's clips as `api.clip_montage.MontageClip` in grid order, the experiment
dir, a `MontageOrigin` for the CSV, a file stem and the reason while empty) and `context_key=`
**required**: a grid cannot be built without its export, and a page adds nothing to the row (AUDIT
hides the order button, its order being fixed). **Export ▾** writes every clip the grid shows, in
its order — *Video of N clips…*, *GIF of N clips…*, *One file per clip…* (stems numbered `NN_`),
*CSV list of clips…* (each clip's own span) — drawn as **Show ▾** says (`show_menu.layers_of`,
`with_context`), from the **source video** through `api.clip_montage.ClusterMontage`, off the UI
thread behind `run_export` / `run_progress` with Cancel; a greyed last entry says it in words
(*Writes what the grid shows: video + masks, as recorded, no context*). The writers are
`MontageExport.write_montage` / `write_files` / `write_list`, one copy for the three grids (the page
and the window each had their own before). The shell's **Export…** no longer lists montage
exports, and the dialog's page-exports section (`gui/export.PageExports`) went with the last page
that offered one.

**Pick mode** (`MontageExport.start_picking`, from *Pick clips to export…*): a tinted bar over the
grid (`PickBar`, shown only while picking: *Pick mode · Click clips to pick them, in order · 3
picked [Pick all] [Clear] [Export 3 ▾] [Quit]*). It starts with **nothing picked** (user decision,
2026-11-09); a click picks a clip — numbered in pick order, `LiveMontage._paint_pick` — or un-picks
it; a drag or Shift-click picks a run through the montage's own `RunPreview` mechanism. The mode
takes the grid's clicks through a **click layer** (`LiveMontage.set_click_layer(ClickLayer)`: hint,
`is_off`, click, run, numbers, `shift_puts_back`), which replaced `set_click_state` and
`MontageClip.dropped` / `pick`; the page's own hint and connections are never touched, so **Quit** or
Esc (a `QShortcut` on the montage; the cluster window also takes Esc anywhere in it) restores the
page's click exactly — AUDIT's removes again. **An export of the picks that finishes leaves pick
mode the same way** (`MontageExport._finished`, reached only from `busy.run_progress`'s success
path); a cancelled or failed one leaves the mode and its picks as they were. The button read
*Done* until 2026-11-09, which looked as if it would export. No ✕ cue is drawn under a layer, and AUDIT's preview
line ignores a layer's runs, so dropping from an export never reads as removing from a label. Picks
are held as clips (recording, subject, start, end), so a re-order or re-mark keeps them; they write
nothing, and a reviewed AUDIT cluster can still be picked from. A double-click's first press is
taken back when the open follows. While picking, both menus write the picks in pick order and are
greyed with *Pick at least one clip.* while there are none.

**Missing previews block the montage area only.** The load asks `clip_tiles.missing` for this run's
recordings (off the UI thread): a recording missing a **proxy** blocks the montage behind the house
bar (`LiveMontage.set_blocked`: its own fraction over this run's recordings, the phase line of the
build in hand, the ETA, Cancel) while the app's queue (`gui/preview_jobs.py`) builds them, moved to
its front (`prioritise`); a recording missing only its **masks** plays footage at once
(`PreviewReader.masks_missing`) and the outline arrives when its build lands. **Cancel** withdraws
this run's recordings and the build in hand if it is one of them (`PreviewQueue.withdraw`), never
another page's; **Build previews** queues them again. The tree, the label field and the filters
never wait.

**Labelling is instant on screen.** The tree, the swatches and the label list update from the node
cache in memory the moment a label is entered. The write (`name_nodes`) and the reprojection it
triggers run in a serialized background worker; a quiet status beside the field says the output is
catching up. A click on a node reads the cache and nothing else, so **selecting a node never
decodes video** (the montage decodes previews on its own worker).

**Surface parity with the behavior-scan page is re-scoped, not kept.** The four stacked sections
(Clips, Ethograms, Locations, Interpretation) were built for a table of clusters at a cut, and the
tree and montage replace them here. The behavior-scan page keeps them. What is still shared, and
must stay shared, is the widgets both pages use: the spatial rendering, the measurements, the
ethogram strip and the §19 audit statistics (each page has its own audit tab).

### AUDIT and BROWSE on a clustering run

Both read the artifact, so both follow a label as soon as the output catches up, and both leave
the `unlabelled` bucket out: it is the run's remainder, not something anyone claimed.

**AUDIT** (§19) reports, never judges: each cluster's precision and each label's, and what they
rest on. Whether a figure is good enough is the reader's call — an earlier table said *holds* /
*fails* / *~16 more clips* against a target, and that call belongs to the person who knows the
behavior. On a clustering run it is **one screen** (`gui/label_check.py`, review round 11): a list
of labels and their clusters on the left, and the **review pane** for one cluster on the right
(`gui/label_clean.py`). A query run keeps its own layout (`gui/audit_tab.py`), scoped by its
behavior selector.

**Every clip is reviewed** (review round 11, the user's decision: "we force people to review all
clips"). Before it, AUDIT was a table, a cleaning scroll that fed no figure, and a separate draw of
25 random clips judged ✓ / ✕ that was the only source of a precision. A draw left most of a
label's clips unseen, and a clip nobody watched cannot be removed, so the two steps are now one:
watch every clip of a cluster, remove the wrong ones, finish. The precision is the census that
finish writes — a count, exact — and the draw, `review_sample` and the judge mode are gone.

Review round 12 settled an undo line, everything about the cluster in the top row, and two-line
list rows. Review round 13 replaced round 11's selection with **click to remove**, said *removed*
everywhere a user reads it (the stored names — `clip_edits`, `take_out_clips`, `taken_out` — are
unchanged), and gave a finished cluster a **reviewed box** and a **split, read-only grid**.
Review round 14 made a run a **drag** as well as a Shift-click, both shown before they act (one
preview), made Finish a large default button with its counts under it, and took the bar off a
reviewed list row.

```
reviewing:
┌ list (230 px, drags) ───┐ ┌ review pane ───────────────────────────────────────────────┐
│ ■ climb              93%│ │ cluster #871  climb › center  ▬▬▬──── reached 40 of 62      │
│   182 clips · 3 of 3 rev│ │                                         [✓ Finish review]   │
│                         │ │                                      60 climb · 2 not climb │
│   cluster #1321 60 clips│ │ Removed 1 from climb › center  [Undo (⌘Z)]                  │
│   ✓ Reviewed    93% · 4 │ │ ┌──┐┌──┐┌──┐┌─✕┐┌──┐┌──┐  every clip, playing; a click        │
│ ■ climb › center        │ │ │  ││  ││  ││░░││  ││  │  removes (faded, ✕, "not climb"    │
│   62 clips · 0 of 1 rev │ │ └──┘└──┘└──┘└──┘└──┘└──┘  tag, in place), again puts back    │
│   cluster #871  62 clips│ │                                                             │
│   ▬▬▬──── reviewing · 2 │ │                                                             │
└─────────────────────────┘ └────────────────────────────────────────────────────────────┘
finished:
                            ┌ review pane ───────────────────────────────────────────────┐
                            │┌──────────────────────────────────────────────────────────┐│
                            ││✓ Reviewed cluster #1321 climb · 8 Nov  95%  57 of 60 are ││
                            ││climb · 3 removed        [Next cluster: #871][Reopen review]││
                            │└──────────────────────────────────────────────────────────┘│
                            │ Removed (3)                                                 │
                            │ ┌─✕┐┌─✕┐┌──┐                                                │
                            │ Kept as climb (57)                                          │
                            │ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  read-only until Reopen review     │
                            └────────────────────────────────────────────────────────────┘
```

- **The list** is a `QTreeWidget` drawn by `ReviewListDelegate`, two lines a row, in a
  `GripSplitter` with the review pane: 230 px until dragged, never under 180 px
  (`LIST_MIN_PX`), the width kept across sessions under `splitters/audit_review_list`. A label: its
  identity colour, bold name and — only once every cluster at or under it is reviewed,
  `audit_figures.complete` of `review_precision(path)` — its combined percentage on the right; then,
  muted, *182 clips · 3 of 3 clusters reviewed* (counting every cluster under it, which is what its
  figure combines). A cluster (its own clusters only, under it): *cluster #871* and *62 clips*; then
  a thin bar (how far its scroll has reached, `LiveMontage.seen`) and *reviewing · 2 removed* /
  *not reviewed* — or, reviewed, **✓ Reviewed** in bold and *97% · 2 removed*, no bar. A
  cluster's clips are its leaf runs whose deepest labelled node is it — the population of
  `review_by_cluster`. Clicking a label opens its first unreviewed cluster; right-click a cluster
  for **Open in INTERPRET**. The label's tooltip carries *Earlier one-by-one audit: …* when
  `audit.csv` has answers — never merged into the figures.
- **The top row** of the review pane (`label_clean.ReviewPane`) holds everything about the cluster
  while it is reviewed: *cluster #871*, the label path, a thin `QProgressBar` with *reached 40 of
  62*, and **✓ Finish review** — a large bold default button, *60 climb · 2 not climb* (live) in
  small type under it, filled with the palette's accent once it can be pressed (the app's style
  draws a default button with only a coloured border), and pressed by Enter — disabled with *Scroll
  to the last clip first* until the last row of tiles has been on screen (`LiveMontage.seen` — a row
  counts once half its pictures' height was in view). The order is fixed, least typical first (no
  order buttons: every clip is reviewed, so the order only decides which come first), and a clip
  never moves while it is reviewed.
- **A click removes** (`LiveMontage.set_click_hint` + `clip_clicked(index, shift)`): a plain click
  removes the clip from the label at once — faded (`MontageClip.faded`), a bold ✕ in its top-right
  (`crossed`), a *not climb* tag in its bottom-left — and a click on a removed clip puts it back.
  **A run** is one step either way, and shown before it is done by one mechanism
  (`LiveMontage.preview` → `RunPreview(kind, put_back, indices)`, emitted as
  `run_marked(indices, put_back)`): a **drag** — press, move past `startDragDistance`, every clip
  from the pressed one to the one under the mouse by index — removes the run, or puts it back when
  it started on a faded clip; a **Shift-click** removes the run from the anchor (the last clicked
  clip, or a drag's end). Either skips the clips already in that state. The preview tints each
  picture in the theme's vermillion (remove) or blue (put back) with a 3px ring; under Shift the
  anchor wears an ink ring. Esc cancels a drag; the grid auto-scrolls while the mouse is within
  36px of its top or bottom. The line under the top row says in bold what letting go will do
  (*Let go to remove 4 clips · Esc cancels*, *Let go to put back 3 clips*, *Shift-click removes 7
  clips*). Nothing previews while the montage is not clickable, so a finished cluster offers no
  run. **Right-click** (`clip_context_requested`) offers *Move to <label>* for each other
  label, most clips first, and *Put back in climb* on a removed or moved clip; a moved clip is faded
  with an *→ climb › left* tag and no ✕. **Double-click** or **Space** opens the clip window
  (`clip_activated`); a double-click's first press was a click, so the pane retracts it (one undo)
  when the double-click lands on the same clip within the system double-click interval. Hovering a
  tile shows the cue — a pointing hand, a faint ✕, and the tooltip *Click: remove from climb ·
  double-click: open* — derived from whether `clip_clicked` has a receiver, never set beside it.
  Every change is written through the page's ordered off-thread queue (`take_out_clips` /
  `put_back_clips`, labels passed by name), with `set_clips(keep_scroll=True)` so nothing moves.
- **Undo**: the line under the top row says *Click a clip to remove it. Drag, or hold Shift, to
  remove a run.* before any change, then what the last one did — *Removed 3 from climb*, *Put back
  1 in climb*, *Moved 1 to climb › left* — with **Undo (⌘Z)** (disabled, *Nothing to undo*, with
  none). The page keeps a per-review history: each change records every clip's mark before it, and
  undo restores those marks through the same ordered writes (`put_back_clips` for a clip that had
  none, `take_out_clips` to its earlier target otherwise). ⌘Z steps back repeatedly; Finish and
  opening another cluster clear the history.
- **A finished cluster** (after Finish, or opened later): the top row and the undo line give way to
  a framed box (`QFrame` StyledPanel — no stylesheet, so no tinted fill): **✓ Reviewed** in the
  status green, *cluster #1321*, the label, the day (`review_by_cluster`'s `reviewed_at`), the
  precision large and bold, *57 of 60 are climb · 3 removed*, **Next cluster: #871** (the next
  unreviewed cluster in list order) and **Reopen review**. The grid is split with
  `set_clips(sections=…)` — *Removed (3)* first, then *Kept as climb (57)*; with none removed only
  the Kept section — one `LiveMontage`, so one playback mechanism. It is read-only: no click
  removes, no hover cue says one would, right-click entries are greyed with the reason; a
  double-click still opens a clip. Reopen review restores the review order and the editing, keeping
  the removals; its tooltip names the figure it will update and how many removals are kept.
- **A snapshot or a finished run** disables clicking, the menu's entries, Undo, Finish and Review
  again with the reason (and INTERPRET's Put back in…); every clip still plays and opens, and Next
  cluster still works. Pose clustering (design, not built: §22's protocol and an empty method
  registry) would have no clip review.

**On INTERPRET**, `NodeIcicle.set_outlines` rings each reviewed cluster in one colour
(`audit_table.audited_colour`): the ring says *reviewed*, not how well. `NodeInterpretTab` reads
`review_by_cluster()` — keeping only clusters with every clip reviewed — and `taken_out()` with the
page's load and again, off the UI thread, whenever AUDIT emits `changed`. The montage keeps every clip of the cluster — **a removed clip stays,
faded, tagged** *not climb* with its ✕ / *→ climb › left* — and right-clicking one offers **Put back in climb**
(`LiveMontage.clip_context_requested`), written through the page's label queue. The label bar says
*48 clips · 6 removed*. Its **AUDIT** line (`gui/node_audit.py`) gives the figure — *precision
0.93 · all 60 clips reviewed* (for a run with only the earlier one-by-one audit: precision, 95% CI,
`n of size reviewed`, CI width against the default goal) — with **Move to…**, **Remove from the label** and **Look at its
halves** (one **Act on it ▾** menu) for the reader who decides the figure calls for one; the two
edits are disabled with the reason on a snapshot or a finished run.

In the earlier one-by-one audit (`audit.csv`, still read where a run has one and for the tooltip), a
judged instance is credited to the cluster holding most of its time
(`NodeView.cluster_of_spans`), since clusters of one label fuse into one instance. Labelling part
of a branch after auditing it re-derives the instances there, so their ids change; the label's
figure then counts only judgments since its membership last changed (`membership_changed_at`),
while a cluster's keeps every judgment, because the cluster did not change. Re-attaching a recorded
answer to the instances that now cover its span is possible from what `answer` stores, and is not
built. BROWSE lists every judged instance with what the audit said.

### Cleaning a label: taking clips out (temporal clustering)

Clustering is never perfect, and a user who has watched a clip and knows it is wrong must not be
made to export it. On a **temporal** clustering run a clip can be **removed** from its label (it
reports as unlabelled) or **moved** to another label. The fit is untouched: the clip stays in its
cluster and in INTERPRET's montage (faded, tagged), so INTERPRET stays true to the raw data.

- **A clip is the montage clip**: one leaf run, `(recording_id, subject, t_ns, end_t_ns)`, `end_t_ns`
  the last member's own time, inclusive. The node cache keeps it stable because the fit never
  changes. `ClipTable` carries each clip's `leaf`. The API's address is
  `storage/clip_edits.ClipRef` (re-exported by `api/clip_edits.py`); `clip_ref()` builds one from a
  `ClipTable.rows()` row (`start_t_ns`), a stored row (`t_ns`) or a 4-tuple.
- **State.** `TemporalClusteringState.clip_edits` is a list of
  `{recording_id, subject, t_ns, end_t_ns, from_label, to_label, at}`: label **ids**, `to_label`
  `null` for removed, `at` ISO-8601 UTC. One entry per clip, the latest winning; *Put back*
  removes the entry. A snapshot copies it with the rest of the state, and a state written before
  the field reads as an empty list.
  ```yaml
  clip_edits:
  - {recording_id: recording_001, subject: 0, t_ns: 510000000, end_t_ns: 765000000,
     from_label: L1, to_label: null, at: '2026-11-08T10:00:00Z'}
  ```
- **API** (`api/clip_edits.ClipEditOperations`, on `TemporalClusteringRun`):
  `take_out_clips(clips, from_label, to_label=None)`, `put_back_clips(clips)`, `taken_out(label=None)`.
  Labels are ids or names. Both edits go through `_edit`: the edit lock, the snapshot and
  finalized refusals, a history line (`take clips out` / `put clips back`), and the reprojection.
  Putting back a clip that has no edit writes nothing.
- **Derivation.** `labelled_instances` builds a per-member override (`api/clip_edits.member_groups`):
  every member inside an edited clip's span (same recording and subject, start at or after the
  clip's, end at or before its inclusive end) gets the target label's group, or `-1`. The temporal
  methods take it as `instances_for(..., member_groups=)` (`protocols.KEEP_GROUP` where the leaf
  decides) and apply it **before** cutting runs: `codebook_viterbi` overrides frames after the leaf
  lookup, so a clip removed from the middle of an instance splits it and `fuse_runs` keeps the two
  sides apart (the hole is its own run); `dtw_finch`'s clip is a whole bout, so it is overridden
  whole. **A leftover piece shorter than the method's minimum (`min_run_s` in rows) is dropped**, as
  any short run is. The clip floor and the instance floor agree: clips use the fit's `min_run_s`,
  instances the run's `method_params.min_run_s`, which the fit was made with. Every export, measure
  and frame view reads `instances.csv`, so all follow.
- **An edit applies only while it still means what it said**: while the clip's label is
  `from_label` or a label nested under it. A rename keeps it (the id is the same); relabelling the
  cluster or deleting the label leaves the entry in the state with no effect (deleting the label a
  clip was moved *to* reports the clip unlabelled, as every clip of that label is). **A merge
  re-points the edits** at the label merged into (`storage/clip_edits.relabel`): the two labels are
  one behavior, so a clip removed from the source stays removed and one moved into it is in the
  target; an edit whose two ends become the same label is dropped. Undo puts them back:
  `label_maps` carries the edits keyed by clip and `restore_labels(..., clip_edits=)` restores them
  (#24).
- **The cache readers follow too**, so the label bar and BROWSE's preview agree with the export:
  `NodeView.set_clip_edits` (called by `node_view()` whenever the state file changes) feeds
  `label_time` (a run's covered time moves to `unlabelled` or the target) and `labelled_runs`
  (`apply_edits=False` gives the clusters as fitted, the lane INTERPRET draws).
- **Pose clustering is not covered.** It is design, not built (§22's protocol and an empty
  `POSE_CLUSTER_METHODS` registry); building it means giving its state a `clip_edits` field and
  handing its methods `member_groups`.
- Not in the CLI: picking a clip means watching it (`docs/cli.md`, *Not in the CLI*).

### Precision from reviewing every clip (temporal clustering)

A temporal clustering run's AUDIT precision comes from **`clip_review.csv`** beside the artifact,
appended through `storage/append_log.AppendLog` (`storage/clip_review.py`). Like `audit.csv` it has
no `schema_version` row; its header is its schema.

| column | meaning |
|---|---|
| `recording_id`, `subject`, `t_ns`, `end_t_ns` | the clip (`ClipRef`) |
| `node` | the labelled cluster it counted toward |
| `label_path` | that cluster's label path then |
| `drawn_at` | when the row was written (the name is kept from the draw that came first) |
| `answer` | `yes` or `no`; `withdrawn` for a clip put back after it was answered; empty only in a row an earlier version logged for a draw |

**Why a census.** Review round 11 replaced the random draw of 25 with a review of every clip (the
user's decision): a draw left most clips unseen, and a clip nobody watched cannot be removed. The
reviewer marks the wrong clips as they go (`take_out_clips`, written at once) and finishes the
cluster, which writes its census.

- **API** (`api/clip_review.ClipReviewOperations`): `finish_review(node)`,
  `review_precision(path, node=None)`, `review_by_cluster(path=None)` (each row's `reviewed_at`:
  when its latest counted answer was written — AUDIT's reviewed box shows its day). `finish_review` writes one
  row per clip of the cluster in **one** append (`AppendLog.append_many`, one `fsync`): `no` for a
  clip removed or moved by an edit that still applies (made from the cluster's label or one above it),
  `yes` for every other. It returns the cluster's `review_by_cluster` row. A later finish supersedes
  the earlier census (`latest_by_path`). Refused on a snapshot and on a finalized run (`_record`);
  an invalid node is refused before anything is logged.
- **The population is the clustering's**: a label's clips are its clusters' leaf runs, each counted
  under its own cluster (its deepest labelled node). `path` covers its own clusters and every
  cluster under it, so a parent's figure combines its own and its sub-labels' by size.
- **The estimate** is `behavior_audit.stratified_estimate` with each cluster a stratum of its clip
  count; a reviewed cluster is exact (`remaining` 0, *all audited*), and a label is exact once every
  cluster under it is. AUDIT and `analysis audit` show a label's figure only then. An answer counts
  when written at or after its cluster's label's `membership_changed_at`, for the label and for a
  cluster alike, **and only under the label path it was written for** (`label_path`): an answer
  says "this clip is (not) climb", so a cluster moved to a newly made label — which has no
  `membership_changed_at` — starts unreviewed rather than showing climb's census. A clip keeps one
  latest answer per path. A rename and a take-out change nothing (a rename carries the answers to
  the new path), and neither does an undo, which restores the label's change time.
- **Putting a clip back withdraws its answers** (the owner's decision, #24): `put_back_clips`
  appends a `withdrawn` row for each path the clip has a current answer under
  (`storage/clip_review.withdraw`), which supersedes the answer without being one. The clip reads
  as not yet reviewed (`remaining` counts it), so its cluster is no longer *all audited* on a "no"
  written because the clip was out; the cluster's other clips keep their answers, and finishing
  again answers it. A version older than this reads the row as an unanswered draw row.
- **An earlier draw's rows stay readable**: a drawn-but-unanswered row judges nothing; a judged one
  counts as that clip's answer (under its `label_path`; a row with none, under any) until a census
  supersedes it, but no cluster reads as reviewed until
  every clip has an answer.
- `clip_review.csv` travels with a snapshot, which reads its own copy.
- **The old `audit.csv` stays readable** and is never merged in. The CLI prints it under *Earlier
  one-by-one audit*. A take-out rederives instances and so changes instance ids, which can drop the
  earlier audit's coverage below `MIN_COVERAGE`; it then reports *no estimate*, as it does after any
  relabelling.

**BROWSE** is the labelled dataset — every labelled instance exactly as written, and where the
ethogram lives: the labelled time per recording, with the cluster selected on INTERPRET as a thin
lane under each strip, labelled or not (`LabelBrowseTab.set_cluster_lane`, from INTERPRET's
`lane_changed`). **Shade there, narrow here**: INTERPRET's filters *shade* the tree by what matches;
BROWSE *narrows* the labelled set by the same dimensions.

```
┌ Filters ─────────┐┌ Ethogram (labelled time that passes, click a bout) ──────────────┐
│ Labels   bouts t ││ Tank A  ▇▇ ▅▅▅   ▇  ▃▃▃▃     ▇▇                                │
│ ☑ rearing        ││ Tank B    ▅▅  ▇▇▇▇    ▃▃  ▅                                    │
│   ☑ supported    │├────────────────────────────────────────────┐┌─────────────────┐ │
│   ☑ other        ││ label        recording  start    dur  audit ││     player      │ │
│ ☑ grooming       ││ rearing›supp Tank A     0:03:12  1.2s  —    ││                 │ │
│ Recordings [ ]   ││ …                                           │└─────────────────┘ │
│ Metadata [col ▾] │└────────────────────────────────────────────┘  [Export selected]  │
└──────────────────┘                                                 [Export all N]     │
```

- **Labels** is the labels panel's nesting (`label_panel.path_rows`, shared) with bouts and time
  per row, and the filter is exact: unticking *rearing* leaves *rearing/supported* showing.
  **Recordings** is a checklist by name with a search box; **Metadata** a column, then its values,
  a recording with no value filed under its own entry.
- **The ethogram** is one strip per recording that passes the recording and metadata filters
  (an empty strip is a finding), coloured by root label in the slots INTERPRET uses
  (`label_panel.root_slots`). A click selects the bout's row and plays it; selecting a row outlines
  its bout (`EthogramStrip.set_highlight`).
- **The list** — label, recording, subject (only when a run has more than one), start, duration
  (`behavior_output.instance_ns`, the last period included), and the audit's answer — sorts on any
  column and multi-selects. It builds the first 2,000 filtered instances, a build cost rather than
  a preference; the ethogram draws them all, and a bout past the cap plays when clicked.
- **Export selected** and **Export all N** hand `ExportScope(instance_ids=…)` to the one export
  dialog; *all* is every filtered instance, not the list's capped rows. A snapshot reads and
  filters and does not export from here.
- **One block, one bottom line.** The block takes its natural height and the page's surplus goes
  below it; the filter lists ask for no height of their own (`QSizePolicy.Ignored`) and divide
  the results', so the filter column ends level with the export buttons. The ethogram shows
  every strip up to six recordings with no handle (a handle that moves nothing reads as
  broken), and scrolls with a drag handle past that.
- **No empty viewers.** Nothing labelled: the tab is one sentence. Filters that leave nothing: the
  ethogram, list and player give way to one sentence and **Clear filters**. Otherwise the player
  opens on the first instance, paused, and *shown, not selected* — a selected row is an export
  scope, and opening the tab must not narrow Export to an instance nobody picked.

Not built: a location filter (per-instance positions are not in the node cache) and ethogram zoom.

### The clip window: three pictures, trails and strips

The window a clip opens into (`gui/clip_popup.py`) is for **seeing** a behaviour, so its Advanced
view shows three pictures of each frame — the footage, the masks alone and the keypoints alone on
white (`gui/clip_panes.py`) — with the picture taking the width and the stretch, and everything
else a row only as tall as it needs: one header row (Advanced / Blind, the clip's length and
place, previous / next), the slider, the strips, the player's controls. Simple is the plain
one-picture player every other surface embeds. Both choices are remembered (`clip_window/advanced`,
`clip_window/blind`); the window opens at 100 x 56 lines capped to 92% of the screen
(`sizing.fit_to_screen`, one helper where the port had three copies), and
`tests/gui/test_clip_popup.py` pins the picture at >= 70% of the window's body at 1500 x 900.

- **The Advanced player is the plain player over its hooks**, not a flag on it
  (`gui/advanced_player.py::AdvancedClipPlayer(_ClipPlayer)`). `_ClipPlayer` is embedded directly by seven
  views and none wants a trail picker, three pictures or strips, so it opens seven seams with
  today's behaviour as the default (`_picture_widget`, `_add_view_controls`, `_decode`,
  `_show_payload`, `_show_unavailable`, `_clip_only`, `_redraw`) and the Advanced player overrides
  exactly those plus `set_clip`; `tests/gui/test_advanced_player.py` asserts the override set, so
  popup-only state landing on the base class fails there first.
- **One settings object, one decode.** The view ticks (Masks / Keypoints on the footage, Trail and
  Trail points, Mask trail, Align pose) are read on the UI thread into a frozen
  `api.clip_views.ClipViewSettings` and handed to the decoder as a value; `ClipQueryPreview.clip_views`
  decodes one frame (`clip_frame`, a `ClipFrame`: the footage, the frame's detections, the placed
  keypoints, the body axis, the trails) and three pure drawers paint the panes from it
  (`draw_footage`, `draw_masks_pane`, `draw_pose_pane`). `ClipFrame.rgb` is `None` when no footage
  was asked for (`footage=False`), and drawing the footage then raises rather than painting a
  white canvas that looks like a frame: the caller that wants no footage draws the panes alone. A
  recording with no `video.resolution` cannot size the panes without the footage, so
  `clip_frame(footage=False)` raises naming the field and **the window decides** (the enlarged pane
  decodes and says so); the API never decodes behind a caller's back. The footage's subject tint
  is `draw_mask`'s own alpha.
- **Trails** (`core/visualize.py`): **Trail** is each ticked keypoint's path over the last
  `CLIP_TRAIL_S` (0.5 s), fading with age, in its role colour (`KEYPOINT_COLORS`), on the footage
  and the pose pane, from the keypoint run the embedding read (`ClipPoints`); body-aligned, each
  past frame is turned by its own axis, so the pane's trail is the path relative to the body. A run
  with no keypoints trails the centre of the animal from the kinematics sidecar's centroids
  instead; `trail_note` says which. **Mask trail** is the subject's masks over the same span under
  the current one, fading (`draw_onion_skin` over `fade_weights`, the fade rule written once),
  read from the detection rows via `MaskHistory` (a `core.lru.CountLRU` of cropped masks), never by
  decoding another frame.
- **Row blocks.** Every detection row a frame or its trail needs comes from a cached block of
  `DetectionPreview.ROW_BLOCK` rows (`_row_block`, `ROW_BLOCKS_KEPT` on a `CountLRU`), keyed on
  `(part stem, block, with_masks)`: the flag that decides the read's cost is in the key, so a
  masks-off caller never touches `mask_rle` (the 4.7 ms `frame_plain` path stays that), and the
  key survives a rebuilt part list, which `id(part)` did not.
- **The strips** (`gui/clip_strips.py`) are `api.clip_curves.read_clip_curves` over the window on
  the bar: **speed** and **area** off the §14 kinematics sidecar through `frame_slices.from_kinematics`
  (the loader `frame_slices`' docstring promised, globalised per part from the timestamp maps) and
  `descriptors.step_curve`, so the step rule cannot drift from the descriptors'; with keypoints and
  a heading, **length**, **bend** and **turning** from `analysis.body_axis.body_shape_series`, the
  code an aligned embedding turns its inputs by. Units are px or mm (`pixels_per_mm`), never a
  body-size unit of the window's own. The strips run at the **embedding cadence** (`lcm` of the
  tracking cadence and `every_n_frames`), with NaN breaks and an untracked lane where the track has
  no row: what the method saw. No sidecar → `curves == []` and a note naming
  `detail recording backfill-kinematics`; nothing is recomputed from RLE. A row is a `Curve` (name,
  label, unit, `signed` / `floor_zero`), so the widget names no curve — `tests/gui/test_clip_strips_scan.py`
  fails if it does.

### Enlarged panes

Clicking the Masks or Keypoints pane (`ClipPanes.enlarge_wanted`, `"masks"` / `"keypoints"`) opens
it in a window of its own (`gui/clip_pane_window.py::ClipPaneWindow`, via
`AdvancedClipPlayer.open_pane`). The cue (a pointing hand through `ImageView.set_rest_cursor`,
"Click to enlarge" on hover) is derived from `receivers(enlarge_wanted) > 0`, so a `ClipPanes`
nobody wired advertises nothing.

- **Full resolution, no footage.** The window draws `clip_views`' own `masks` / `pose` picture at
  the record's frame size with its own `ClipQueryPreview` (a decode context's reader keeps its
  place, so two threads never share one), calling `clip_views(footage=False)`; it and its exports
  decode no video (`test_the_enlarged_pane_decodes_no_video`). A record with no frame size is the
  exception: the window decodes and its note says so.
- **Live, and torn down.** It reads the player's ticks on `view_changed`, its window of frames on
  `window_changed` and its clip on `clip_changed`, so it is **non-modal** and the clip window's
  child (`requirements._NON_MODAL_DIALOGS`), closing with it. **One window per pane**: a second
  click raises the open one (`open_pane_window`, `player.pane_windows`); Close deletes it
  (`WA_DeleteOnClose`), pauses its scrubber and disconnects it from the player (`finished`). The
  port this replaced built a window per click and set none of that, so a closed window stayed
  parented to the main window, still wired, decoding in a loop forever.
- **Exports** (`api.clip_views.export_clip_pane`, through `api.animation.write_animation` and the
  GUI's `animation_export.run_export`, the house `run_progress` mapping): the frames on the
  window's bar — the clip plus its context when Context is on. An mp4 keeps every frame at full
  size and the recording's rate; a GIF samples real time at the asked rate (`pane_frames`, with
  `SAMPLE_SLACK_NS` so an integer-nanosecond boundary picks the frame at it, not the one before)
  at the asked width, both asked by `GifSettingsDialog` (640 px, 12 fps), with one palette for
  all its frames. A cancelled or failed export removes its partial file.

### Clip montage

**A cluster's clips side by side, on one clock, drawn from the source video** (`api/clip_montage.py`,
`detail analysis cluster-montage`, and INTERPRET's **Export ▾**). The window that played it on the
GUI (`gui/clip_montage.ClipMontageDialog`, opened from the tree's right-click) is gone: INTERPRET's
montage plays every clip from the previews (*INTERPRET* above), and this module is what an export,
the CLI and a script write with. `montage_members` gives every clip of the node, most typical first
(through `typical_members`, which reads `NodeView.clips`), and a montage takes `DEFAULT_CLIPS` (25)
of them in `SORTS`' order (`choose_clips`). The sorts are most typical, least typical,
by recording, longest, and **random**: a shuffle of the clips in rank order seeded by the node id
(`sort_clips(seed=)`), so a node's random 25 are the same on every open and a fair look at a large
node rather than its centre alone. A clip's `rank` is its typicality
rank among the members whatever the layout. Every clip starts at once and **each loops from its
own start the moment it ends** (`frame_at`: its frame at `start + (offset mod its cycle)`, the
cycle its length plus one frame step), so no tile ever stands still — a tile that held its last
frame until the longest clip ended read as a stuck one; a pass is the longest clip's length,
after which every clip starts together again. The clock is the footage's rate up to
`MONTAGE_MAX_FPS` (15), and time, not frame count, is what is synchronised (Tenet 8).

- **A tile is the clip window's read, drawn by the clip window's drawers.** `tile` reads one
  `ClipQueryPreview.clip_frame` with `MontageLayers.settings` (a `ClipViewSettings`) and paints
  it with `draw_footage` / `draw_masks_pane` / `draw_pose_pane`, so a tile and the clip window
  never disagree about a frame. The one thing a tile adds is the **outline**: with Raw on the
  animal is outlined in the mask tint at the tile's own scale (`visualize.mask_outline`, its
  fade `fade_weights`', its blend `tint_by_weight`'s), never tinted, so its fur and paws stay
  visible. `MontageLayers.flavour` names the look in `clip_render`'s words (`RAW` / `OVERLAY` /
  `MASK`). Raw off decodes no video (`footage=False`); a recording with no recorded frame size
  is the exception, where the montage, as the caller, reads the footage for that clip
  (`_Prepared.needs_footage`) rather than showing a grey tile.
- **The crop is fixed per clip** (`crop_box` over `ClipQueryPreview.clip_boxes`, the animal's
  boxes read off the detection rows in blocks), so a tile never jitters.
- **Ownership and order.** A `ClusterMontage` is one export's: its decode contexts, per-clip
  reads and threads (`core.cpu.thread_workers()`, the rule the embedding worker and the
  kinematics backfill each had a copy of) are its own, and `close()` shuts the pool and drops
  them, so a finished export holds nothing. It keeps no rendered tiles between exports: within
  one export a frame sampled twice in a row is drawn once (`_export_tile`'s `held`). **Drawing
  before `prepare()` raises** (`RuntimeError`, "call prepare() before drawing") rather than
  preparing from inside a pool task — the port this replaced did, a nested blocking `pool.map`
  on the pool it ran on, reachable only by call order. Concurrent cold reads of one record wait
  for a single parse (`read_cache.computing`, taken by `cached_read`, `cached_document` and
  `read_recording`'s own cache).
- **One grid formula.** `grid_size(columns, rows, tile_px)` is what an export's frame
  (`export_grid`, `compose`) measures with. `fit_label` stays in `api/`: an export draws its
  labels with no Qt.
- **Exports** draw at their own size; a GIF has one palette for all its frames
  (`api.animation`). The clip list is a **clip-query CSV** through `storage.clip_query_csv.write_clip_csv`
  (which grew `extra=` for columns after the canonical ones): every clip an example (`kind="query"`,
  `rank=0`) of the one query named after the node, so `analysis query --from-csv` reads it, with
  `MONTAGE_LIST_EXTRA` (`rank_typical`, `recording_name`, `subject`, `node`, `run_id`, `start_s`,
  `end_s`) after. `cluster_montage()` is the context manager a script or the CLI exports from,
  closed on the way out; the CLI refuses `--max-width` / `--fps` with nothing to picture.
- **On every clip grid** an export runs behind the house bar (`run_progress`, Cancel), from the
  grid's order or the picks, and writes only where the user points. It draws **what the grid
  shows** (`show_menu.layers_of` of **Show ▾**'s look, the orientation included, and its context),
  and **Export ▾** says so; Export… entries had once said "from the source video" while writing
  `DEFAULT_LAYERS` — masks on white, no footage.
- **Aligned** (`MontageLayers.aligned`, `--aligned`): a tile frame turned by `clip_views.turn_matrix`
  about the axis's middle, the crop's side kept, the outline placed by the same turn
  (`_outline_turned`); a frame with no axis is `_as_recorded` and marked. **Context**
  (`with_context`, `cluster_montage(context_s=)`, `--context`): each clip widened either side;
  labels and the clip list name the clip itself.

### Cluster windows, the clip window, and the Aligned orientation (2026-11-09)

**A cluster in a window of its own** (`gui/cluster_window.py`). **Open in window** — on
INTERPRET's label bar after **Undo** (`LabelBar.add_action_widget`) and the tree's right-click menu
(`NodeIcicle.window_wanted`) — opens a non-modal `ClusterWindow(Dialog)` (`requirements.
_NON_MODAL_DIALOGS`): INTERPRET's own `LabelBar` over a `LiveMontage` of the cluster's clips with
its own `TileCatalog` (a catalog is one worker's). The bar is built by the page's
`node_workspace.build_label_bar` — the call that builds the page's — with the montage toolbar on
its first line, its name led by the window's number, and without *Open in window*; it is filled by
the page's `show_cluster_name` / `show_figures` (`cluster_facts`, `clip_count_text`) /
`show_label_state`, so the window and the page cannot drift (user, 2026-11-09; guarded by
`test_cluster_window.py::test_the_windows_top_is_interprets_label_bar_row_for_row`). Nothing sits
between the bar and the grid but the pick bar while picking; "ranking this cluster's clips…" is
said in the figures, as on the page. Its permanent Show
row, Context box, picks bar and export bar went with "one montage toolbar everywhere": the grid has
their height. `ClusterWindows`, owned by the page, keeps **one window per
cluster** (a second open raises the first), numbers each (the lowest number free: in its title,
*① Behavior #1321 · climb · an_32*, and as a disc on its block in the tree,
`NodeIcicle.set_window_marks`), cascades new ones and forgets a closed one. On a reload the page
compares `node_workspace.tree_identity` (the fit's file and size, `NodeView.built_from`, and a
hash of the tree's merges) with the tree it had: the same, each window re-reads its clips; changed
(a re-fit), every window closes and the page says so (*Closed 2 cluster windows: the run was
re-fitted.*), since a node id would name another cluster. Windows are the page's children: they
close with it.

- **Labelling follows the window.** The window's bar is the page's `LabelBar` (field, Label /
  Remove / Labels ▾ with its numbered recent labels, Undo). `build_label_bar(page, montage,
  origin=window)` sends every request to the page's handlers (`_label`, `_rename_label`,
  `_merge_label` — told which window asked, so a prompt opens over it — and `_select_label`), so
  the run, the tree, the bar and the rings move as from the page; `_refresh_labels` and
  `_apply_editability` refresh each window, and the window's status follows the page's
  (`LabelBar.status_changed` → `LabelBar.show_status`). Keys 1–9 press the window's own `QuickLabels`. **INTERPRET stays
  put** (D1 B, 2026-11-09): neither a window's focus nor its label selects that node on the page.
  Enter clicks no button: `gui/dialog.Dialog` takes default and auto-default off every button of a
  non-modal dialog (`requirements.a_non_modal_window_has_no_default_button`) — Remove, the
  window's first button, had been the default, so Enter in the field wrote the label and removed
  it. A label write re-marks the other montages only where a tile's marks changed
  (`node_workspace.marks_changed`). A snapshot or finished run greys the field and Undo with
  `_edit_reason()`.
- **One ⌘Z history** (`gui/label_history.py`) for the page and its windows: every label-state
  write (label, remove, rename, merge, put back) is a `Change` with how to show and write either
  side. A label write reads `AuthoredRun.label_maps()` either side of itself
  (`authored_run.label_change`), and undo / redo write a side back by id through
  `AuthoredRun.restore_labels`, queued behind the write they reverse — names, clusters and each
  label's `membership_changed_at`, so an undone misclick leaves the label's review counting. **Undo (⌘Z)** on the bar
  and each window's row, its tooltip naming the change; greyed with `_edit_reason()`; cleared on a
  re-fit and when a write fails. Not in the CLI (`docs/cli.md`).
- **Picks and the export** are every grid's (*One montage toolbar everywhere*, above): the window
  hands `ExportSource` its clips in the window's order.

**The clip window is non-modal** (`clip_popup.open_clip_window`, `requirements._NON_MODAL_DIALOGS`):
every opener (INTERPRET, AUDIT, the cluster window, the cluster strip, both query grids) calls it;
while the page's window is open the next clip is shown **in it** (`ClipPopup.show_clip`, the
sequence replaced) and raised. Parented to the page; `WA_DeleteOnClose`. A source scan
(`test_every_opener_goes_through_the_one_non_modal_opener`) fails on a page that builds the window
itself. **Export ▾**: *Video of what you see…* / *GIF of what you see…*
(`clip_views.export_clip_view`: `view_picture` — the footage, or the footage with its two panes — of
every frame on the bar, context included, with the window's `ClipViewSettings`) and *This frame as
PNG…* (`write_clip_frame`, off the UI thread through `run_busy`). The plain player's own **Export
video…** is hidden in this window.

**Aligned** (`ClipViewSettings.aligned`, `TileLook.aligned`, `MontageLayers.aligned`) is **one turn**,
`clip_views.turn_matrix(front, back, side, out_wh)`: rotate about the body axis's middle so back→front
points **up** (Align pose's direction), put the middle at the output's centre, scale `side` to the
output's height. The axis ends are the frame's keypoints of the run the embedding read, averaged over
the heading definition's front and back sets at the run's cutoff (`axis_ends`, `body_axis.
row_axis_ends`), in the target picture's own pixels, so the angle is right at any size. Used by the
clip window (`turn_views`: footage, masks pane and pose pane — unless Align pose already aligned it —
turned together at the frame's size), the live tiles (`PreviewClip.set_pose`, fed once per clip by
`ClipTiles._pose` from `TileCatalog.points` → `ClipPoints.rows_at` over the tile frames' source
frames; the mask, its trail — each past mask turned by the *current* frame's turn — and the keypoints
placed by the same matrix) and the exports. **No cache on disk**: the headings are computed as a clip
is opened and kept with it in memory. A frame with no axis is drawn as recorded and marked *no axis*
(`mark_no_axis`); a run with no keypoints or no fitting heading definition draws nothing turned, and
every orientation choice (*As recorded* / *Aligned*: in **Show ▾** on every clip grid, the clip
window) greys *Aligned* with
`ClipPoints.note` / `axis_note`, read off the UI thread with the
page (`clip_views.pose_notes`). Each montage keeps its own choice (`LiveMontage(aligned_key=)`,
required, so no page silently shares another's); the clip window keeps `interpret/clip_window_aligned`.
Measured on a real run, a turned tile frame stays well inside the ~2.2 ms a frame 30 tiles × 15 fps leave the one picture worker on a first pass; later passes are served from `ClipTiles`' drawn-tile cache.

### Every node's clips

**Every node with members has clips, and the cache lists them.** `NodeCacheMixin.clips(node,
progress=)` is `NodeView.clips(node)` with a medoid method's typicality filled in on first ask
(*The node cache*, *Clips and typicality*), and `typical_members(node, limit, progress=)` the same
as rows, most typical first, for the CLI and exports. INTERPRET's montage plays them all; the tree's
old **Preview clips…** and **Clip montage…** items are gone with it. A node whose every stretch is
shorter than the run's minimum instance has no clips, and its montage is empty.

### The node cache

**Computed once after the fit, so that interpretation is instant.** A user has already waited for
the fit. Spending more time up front, in a phase that reports its progress, is better than making
every click wait. The cache lives at `<run>/node_cache.zarr` and is **independent of labels**: it is
a function of the fit and the tracks, so labelling never invalidates it, and a snapshot of the run
reads the same cache. **Nothing in it is decoded from video**, and there is no size floor: every
node of the tree, however small, has every figure and every clip.

**Leaf-additive, so every node has exact figures.** Everything that sums is stored **per leaf** or
**per leaf run**, and a node's value is a sum over the leaves under it, which is one numpy
index-and-sum. That makes every node's figures exact and cheap, however large the tree.

| array | shape | dtype | chunks | what |
| --- | --- | --- | --- | --- |
| `leaf_time_ns` | `(L,)` | `int64` | `(L,)` | member time per leaf (*Member time*) |
| `leaf_members` | `(L,)` | `int64` | `(L,)` | member count per leaf |
| `leaf_rec_time_ns` | `(L, R)` | `int64` | `(L, R)` | member time per leaf and recording |
| `leaf_rec_pos` | `(L, R, 16, 16)` | `uint32` | `(64, R, 16, 16)` | members per leaf, recording and cell of the frame-normalized grid, axes `[by, bx]` |
| `leaf_desc_hist` | `(L, D, 32)` | `uint32` | `(L, D, 32)` | descriptor histograms over run-wide bins (not yet built) |
| `desc_edges` | `(D, 33)` | `float64` | `(D, 33)` | the bins, fixed from run-wide quantiles (not yet built) |
| `recording_ids` | `(R,)` | `str` | `(R,)` | the column order of every `R` axis |
| `desc_names` | `(D,)` | `str` | `(D,)` | descriptor names (§23 descriptors; not yet built) |
| `run_recording` / `run_subject` / `run_start_ns` / `run_end_ns` / `run_leaf` | `(U,)` | `int32` / `int64` / `int64` / `int64` / `int32` | `(65536,)` | leaf runs: maximal stretches of one leaf per unit (one per bout for a bout method); `run_end_ns` is the end of the time covered, the last member's **start** plus its time (`analysis/node_cache.covered_end`: a frame plus its gap, a bout's last frame plus one step), so a run covers exactly its members' time, for the labels ethogram |
| `run_cx` / `run_cy` | `(U,)` | `float32` | `(65536,)` | each run's mean frame-normalized position, `NaN` where unknown |
| `run_rows` | `(U,)` | `int32` | `(65536,)` | members per run |
| `run_last_ns` | `(U,)` | `int64` | `(65536,)` | the run's last member's own end, inclusive (§19): where a clip ends |
| `run_dist_sum` | `(U,)` | `float64` | `(65536,)` | `distance` scoring only: the run's summed member distances |
| `run_vec_sum` | `(U, K)` | `float32` | `(65536, K)` | `centroid` scoring only: the run's member vectors summed in `K ≤ 64` dimensions |
| `leaf_vec_sum` | `(L, K)` | `float64` | `(4096, K)` | `centroid` scoring only: `run_vec_sum` summed per leaf |
| `separation` | `(N,)` | `float32` | `(N,)` | per internal node; `NaN` on leaves |

Metadata distributions are joined at read time from `leaf_rec_time_ns` and `metadata.csv`, so editing
the metadata needs no rebuild. The labels ethogram is built in memory from the leaf-level runs mapped
through the current labels. It is a preview; BROWSE shows the exact instances.

**Clips and typicality.** A node's **clips** (`NodeView.clips(node) -> ClipTable`: recording,
subject, start and end `t_ns`, mean `cx`/`cy`, members, typicality, in time order) are
**one clip per leaf run under the node, never joined with the runs beside it**, kept
when the run holds at least the method's minimum instance length in members. So a leaf's clips
are the runs the method's own `instances_for` makes of that leaf with everything else
unlabelled; a node's clips are exactly its children's; and the root lists every clip of the
run (the count the filter bar's "n of N clips match" reads). A bout that crosses two
sub-clusters plays as two clips: joining touching runs under the node instead made the root one
clip per recording, and a mid node's clips several seconds of mixed behaviour. The minimum length is
the method's rule per run (`_runs_of` drops any run shorter than `min_run_s`), so it applies to a
leaf run unchanged and holds at every node alike. The runs are cut on the method's continuity
blocks, under the run's params (`ClipScoring.member_block_start`): a run never crosses one, so a
`codebook_viterbi` run with an explicit `max_gap_s` cuts its clips where it cuts its instances.
One vectorised pass over the runs, well under a millisecond on a real run. A bout
method's runs are its bouts (`instances_are_runs`).
**Typicality** is each method's own definition, made a sum over runs by the `ClusterMethod.clip_scoring`
hook (`ClipScoring`), and is a distance: smaller is more typical, as `rank_members`' is.

- `distance` (`codebook_viterbi`): a frame's distance to the codeword it was decoded into does not
  depend on the node, so a clip's score is its frames' mean, `run_dist_sum` over `run_rows`, the
  same at every node that holds it. At a leaf, clips and their order equal `rank_members`; above
  one, `rank_members` ranks the node's joined runs (its instances), so the clips are its
  children's in their own scores' order (`tests/api/test_node_members.py`).
- `centroid` (the pose methods' scoring; no registered method uses it today): `1 - cos` between the clip's run's summed member vectors
  and the node's centroid, the sum of `leaf_vec_sum` over its leaves. Every frame is scored, where
  `rank_members` scores the fit sample only (the fit keeps no other vectors), so the build reads
  the fit's trajectories back. The sums are kept in the leading `K = 64` right singular vectors of
  the fit sample (`analysis/node_cache.VECTOR_DIMS`, `vector_basis`: the eigenvectors of the
  `(D, D)` Gram matrix, so no `(S, D)` left factor is ever built): on a fit of pose vectors that keeps each
  node's run order close to the full vectors' order, for 256
  bytes a run against 3 KB.
- `medoid` (`dtw_finch`): the medoid order depends on the node and does not add up, so a node is
  ranked on first ask from the fit by `rank_members` and kept on the run handle per node
  (`NodeCacheMixin.clips`). `medoid_order` alone is
  O(m²) and measured 12 / 67 / 259 ms for 2,000 / 5,000 / 10,000 members on a synthetic matrix.
  A bout assigned after the fit was never compared and stays unranked (`NaN`, sorted last).

`ClipTable.order(sort, seed)` gives `typical`, `atypical` or `random` (a shuffle seeded by the node
id). `rank_members` stays the ranking the CLI and exports read from the fit.

**Filters are arithmetic over the same arrays** (`api/cluster_filter.py`). A `Selection` holds
values per category column, a range per number or date column, a set of recordings, a set of
cells of the 16×16 grid and a time-in-session range; metadata and recordings reduce to a recording
mask (OR within a column,
AND across; `recording_mask`), a region to a cell mask. `NodeView.match_shares(selection, metadata)`
is every node's share of members meeting it, exact from
`leaf_rec_pos[leaves][:, recordings][..., by, bx]` over `leaf_rec_pos[leaves]`, summed in `int64`
with the cells reduced per leaf and recording before any copy (a run with no positions counts
member time from `leaf_rec_time_ns`, and refuses a region);
`cluster_filter.clip_matches(selection, clips, metadata)` tests each clip's recording and mean
position. **Time in session** is `Selection.session = (lo, hi)`, fractions `0 <= lo <= hi <= 1`
of each recording's **whole video**: `0` its first frame and `1` its last, the first and last
`t_ns` of its active timestamp map (`cluster_filter.session_spans`, through
`recording.session_span_ns`; a multi-part recording's span is its first part's first time to its
last part's last, since `t_ns` runs from the start of the first part — Tenet 8, never cv2's frame
count). Normalised per recording, so a range is the same part of every recording. `(0, 1)` is
stored as `None`, no filter. The window is **half-open**, `[lo, hi)`, both edges rounded by one
rule (`cluster_filter.session_window`), so `[0, x]` and `[x, 1]` meet at the same nanosecond. A
range from `0` or to `1` is **open at that end**: a run's covered time reaches past the last frame
by one frame step, and a closed end would put it in no range, so `[0, x]` and `[x, 1]` would not add
up to the whole and the end of a session could never hold its own last run. A recording with no map has no span and matches no session range. A
run that clustered only part of a video has no members in part of the range. **Edge clips are
trimmed, as Location splits shares from clips**: `match_counts(..., spans=)` scales each leaf and
recording's count by `NodeView.session_shares`, the share of its leaf runs' covered time
(`run_end_ns - run_start_ns`) inside the range's window, so a run half inside adds half; and
`clip_matches(..., spans=)` keeps a clip whose midpoint `(start_ns + end_ns) // 2` is inside
`[lo, hi)` (with `hi = 1` reaching the end and `lo = 0` the very start), so `[0, x]` and `[x, 1]`
split the clips exactly, a clip whose midpoint is on `x` going to `[x, 1]`; a run of no length in
`session_shares` follows the same rule. With a region as well, the cache keeps no position per moment, so the two are combined
as independent within a leaf and recording. Computed from the leaf runs at query time, with no
cache table: one `bincount` over the runs, 11 ms on a synthetic 1,000,000 runs
× 5,000 leaves × 72 recordings. `storage/metadata.infer_column_types` says which columns are numbers (every non-empty
value parses as one), ISO dates, times of day (`HH:MM[:SS]`, ranged by seconds after midnight,
`parse_time`), or categories; a date with a UTC offset is read as naive UTC,
so a column mixing the two forms compares. A run with no node cache lists no clips: `clips`
raises `FileNotFoundError` naming `--rebuild-cache`, which the CLI prints. The montage's *least
typical* keeps a clip the method could not rank last, as `ClipTable.order("atypical")` does
(`MontageClip.ranked`).

Written through `storage/zarr_store.open_group`, with `schema_version = 1` first, a `built_from`
digest (the fit shard's name, size and mtime, and `rules`, the rule set the figures were computed
under) and a `clips` attr (`{"kind", "min_members",
"join"}`, the method's `ClipScoring` without its arrays). It is staged at `node_cache.zarr.tmp` and
swapped in with `core.atomic.replace`, the frame view's pattern. A cache whose `built_from` no longer
matches is reported as stale with a rebuild offered, never silently read: INTERPRET shows a banner
naming what changed. The check compares the fit shard's **name and size**; its time is recorded but
not compared, because copying or syncing an experiment rewrites every file time and would flag
every run's clips. **A rule that corrects what an existing array holds bumps `rules`**
(`storage/node_cache.RULES`), not `schema_version`: the arrays keep their names, dtypes and
meaning, so an old cache still reads, and `stale_reason` reports one computed under an older rule
the same way, through the same banner and rebuild. A cache without the key was built under rule 1.
Rule 2 ends a leaf run at its last member's start plus its time; rule 1 added the time to the
member's end, which for a bout counted its span twice (a bout ran past its last frame, over the
next bout, and past a recording's last frame), skewing Time in session shares, the labels ethogram and lanes, and the time a moved clip
carries in `label_time`. Only a bout method's rule-1 cache is reported (`clips.join` false): a frame
starts where it ends, so both rules give a frame method the same runs. **Rebuild** (`run.rebuild_node_cache(progress=)`, the CLI's `--rebuild-cache`,
the tab's **Rebuild clips**) recomputes the whole cache from the fit, reading the fit's
trajectories again only for a `centroid`-scored method (none is registered today); it is not an authored edit, so it is allowed on a finished run and refused on
a snapshot view.

**Caches and runs from before every node listed its clips still open.** Such a cache held per-node
member lists and JPEG thumbnails for nodes above a size floor (`list_*` / `thumb_*` arrays, a
`min_cluster_s` attr, a `pictured` attr) and no per-run sums. Those arrays and attrs are ignored on
read; the cache reads with `run_rows` and its siblings absent, every node still lists clips (one per
leaf run, no minimum length, no typicality), and `stale_reason` says the cache predates
typicality, so INTERPRET offers **Rebuild clips**. The floor's run keys, `min_cluster_s` in launch
configs and params and `cache_floor_s` in state, are dropped on read by one helper
(`configs/clustering_legacy.drop_legacy_floor`): in each state's `from_dict` and in each module's
`validate`, so a run started from an old one launches without it. The columns added under the
same `schema_version` (positions, then the per-run sums) are all additive.

**Member time is measured, never frames ÷ fps.** Member time is in seconds, so pose (whose
members are frames) and temporal (whose members are bouts) mean the same thing by it. A pose member's time is
the measured gap from its `t_ns` to the next analysed member's inside the same continuity block, so a
fit analysed at every *n*-th frame counts *n* frames of time per member. The last member of a block
gets the block's measured step (`nominal_period_ns`). A bout's time is its span plus one period, the
convention of `behavior_output.instance_ns`, with the step measured from the bout's own span and
frame count. A block of one frame takes the run's measured step. No advertised fps is consulted
anywhere: the saved `fps_actual` is itself derived from these timestamps, and an advertised rate
is exactly what disagrees with them when it is wrong. Member time counts
every member, flickers included, and the label bar gives the figure as member time, distinct from the
output's total time, which counts instances only.

**Concentration ships with its floor.** A node always has a most-common recording, animal or group,
so ranking concentration without a reference manufactures a finding on every click. For each
dimension, the cache reader redraws the node's member count from the run's own distribution, 200
times with the node id as seed, and takes the 95th percentile of the largest over-representation.
A chip or flag appears only when the node beats that. A node too small to beat it says nothing, which
is right.

**Separation has a reference too.** `node_separations` is shown next to the run's median over all
internal nodes, so "0.41" reads as clearly different or barely different without a user knowing the
method's distance scale.

### Runs written with a granularity

A run written before this redesign carries `state.granularity` and possibly `names_by_granularity`.
**It still opens.** Its names fold onto nodes as labels (*Labels, and the three label states*), so
the new screen shows them. Its
old snapshots keep the artifact they pinned. On the first label edit, the live run is reprojected the
new way (label-driven, unlabelled as one bucket), and that edit's `history.jsonl` entry records the
migration (`migrated_names`, how many per-granularity names moved onto nodes, and `folded`, what
the fold changed in words; `authored_run.reresolve_labels`). A run with no node cache opens to one action: **Build the node cache**, with progress.

### Progress and loading

The four wait kinds again (§20's table applies unchanged). What differs here:

- **The fit** is executor-backed, and reports a **global** phase fraction rather than per-recording
  progress: `loading → clustering → persisting → projecting → caching` (the module's declared
  `phases`). `TemporalClusteringProgressEvent` (`pipeline/temporal_clustering.py`) carries it in
  the analysis event's shape: its `recordings_done` / `recordings_total` pair counts phases, and
  its `frames_done` / `frames_total` pair (out of 1000) the fraction within the current one, so a
  renderer that folds the inner fraction into the outer count reads a phase bar. **`caching`** runs last, after the artifact is written, and the
  record says `open` only once it ends. It names each step as it goes (member time, positions per
  animal, split separations, clips and typicality) and carries an ETA
  measured over the running phase (`core/progress.PhaseClock`).
- **Rebuilding the cache** reports the same steps through its `progress` callback, driven by the
  CLI's `_StageProgress` and the GUI's `run_progress`, with cancel.
- **Opening INTERPRET** reads the tree and the cache: `busy.run_busy`, never inline.
- **Selecting a node** reads the cache only and needs no busy state.
- **Labelling** updates the screen at once and writes in the background (above). The workspace does
  not freeze for a label; it freezes, as §21 requires, only for a compute operation (a rebuild).

### The difference screen

The output unique to discovery: **which clusters differ most across a metadata column**. Per
cluster, frequency by recording, grouped by the chosen column(s), ranked by effect size.

**It ships with its floor, and this is not optional.** With tens of clusters and a handful of
animals, ranking by difference *always* produces a most-different cluster, whether or not anything
is there. The floor is what the same procedure yields on shuffled group labels, computed over the
same data and shown beside the ranking. Without it the screen manufactures a finding every time it
is run, and it will be run every time.

It is framed as **where to look**, never as a result — the thing it produces is a cluster worth
watching clips of, and the interpretation is the user's. Mechanically it belongs to the comparison
layer (filter, group by metadata, aggregate a measure); discovery is its first consumer.

**It reads the node cache, not the artifact, and it suggests, never writes.** The artifact
collapses unlabelled nodes by design, and the screen exists to find the nodes nobody has labelled
yet; so §24's `source: nodes` measures every candidate node from the leaf-additive tables, resolves
the design from the chosen column (a within-animal dose is a trend, vehicle against drug a
paired contrast, sex a between-unit fold change), ranks on time fraction with bouts per hour beside
it, suppresses nesting by peak picking and ships the floor. A kept node is a *suggestion* with the
sentence that earned it ("less under drug than vehicle, 2.1×; 8 of 10 animals in that direction");
a label is the user's, written from the clips, because a label this screen wrote would be scored
by the comparison that selected it. All of this was measured before any of it was built, in
study records kept outside the public tree.

**It lives in two places, deliberately.** The durable one is the saved §24 node comparison
(`source: nodes`), with its provenance, record, `ranking.csv` and figures — a result, in the
experiment's Comparisons section, and what a label written from the screen cites. The live
one is INTERPRET's **Differences** tab in the filter dock (*INTERPRET: the interpretation tab*):
the run's saved screens, one tickable at a time, and *New screen…*, which opens the comparison
launch dialog seeded to this run and the cluster source. Ticking a screen is a condition on the
clips like the others — **the clusters it kept, as the leaves under them**
(`cluster_filter.Selection.leaves`, key `CLUSTERS`) — so it ANDs with the recording, location and
session filters; the montage greys the clips of clusters that do not differ; and "n of N clips
match" reads as clips of clusters that differ. The page reads each screen's ranking back through
the one record → ranking mapping (`comparison_report.rank_record`), so the dock, the Comparisons
page and an export cannot disagree. One screen at a time: two statistics cannot shade one tree. **A screen with no floor is listed
and cannot be ticked** (`ScreenChoice.leaves` `None`, as for one that cannot be read), with its
reason in full under the list (`cluster_filters.unavailable_text`): nothing it ranked is shown to
differ, and ticked it took the labels' colours off the tree, coloured nothing and greyed every
clip (user, 2026-11-17); so is one with nothing to contrast (`rank_record` `None`). Such a screen
is refused at launch when its shape is knowable (§24 *Known before measuring*). A
tab beside INTERPRET was specified for this (§21) and never built; the dock is the screen it was
meant to be.

**Colour on the tree means one thing at a time** (`NodeIcicle.labels_coloured`, `fill_of`). With
no filter shading the tree and no screen ticked, a block is its label's colour: solid for its own
label, pale under an ancestor's, neutral with none. Once a filter shades the tree, or a ticked
screen colours it or frames what it kept, colour is the share's or the screen's: a block a label
covers shows a 3 px ink strip along its top and its name, never the label's hue, and the swatch
row above the tree draws its squares in the text's colour (`NodeInterpretTab._show_swatches`). A
screen's frames with no field on leave the fill unshaded, and a labelled block is then the neutral
stepped toward ink. The label's hue once rode the top of a shaded block — a blue strip on a yellow
block, the palette's first two being blue and amber, the scale's two poles (user, 2026-11-16:
*we're using color to communicate two different things*). **And blue and yellow are the screen's
alone**: a label's colour comes from `ChartTheme.label_identity`, the identity colours less the
twelve slots within ΔE 15 of either pole in either theme (`LABEL_RESERVED_SLOTS`; blue, amber, sky
and vermillion of the six), on every surface that colours a behavior by its label — this tree and
its label bar, BROWSE's list and lanes, the label check, Find clusters' bands, and the exported
behavior figures (`chart_theme.assign_identities`) — so a label is one colour everywhere and never
a pole's. Twelve label colours remain before they repeat. Labels made before this take the new
colours, on screen and in figures exported from now on. AUDIT's ring keeps its colour; it is a
frame, never a fill, and is not yet reconciled with this rule.

**The dock has two tabs, and the open one decides what the tree's colour means**
(`cluster_filters.FilterDock.tabs`, `FilterBar.open_tab`, `NodeInterpretTab._shade_tree`).
*Filters*, first and the tab the dock opens on, holds the clips-that-match scale, the count line
with Reset all, and every field. *Differences* holds the saved screens, the ticked screen's scale
and its strictness. Each tab's
label counts what it has on (*Differences · 1*, *Filters · 2*), so the tab that is not open still
says it is narrowing the clips or colouring the tree. Reset all removes the Filters tab's fields
and leaves a ticked screen, which is on the other tab. The two scales are two widgets
(`ScreenList.key`, `FilterDock.key`), so neither changes its units. This replaced one scale at
the top of one column that read the screen's statistic *while the screen was the only filter on*
and a share otherwise: ticking a field beside a screen swapped the tree's colours and the scale's
units with nothing said, and the scale sat far from the screen that set it (user, 2026-11-16).

**While the Differences tab is open, the tree reads the ticked screen's statistic** — whatever
else is filtered, and nothing is shaded with no screen ticked. The blocks take the theme's
diverging scale (`plot_theme.effect_colour`): less at the blue pole, more at the yellow, neutral
grey inside the floor and for a cluster the screen has no row for, so only what cleared is
coloured and its sign says which way. The poles differ in lightness as well as hue — blue against
yellow is the contrast a red-green colour-blind reader keeps, where the earlier blue ↔ vermillion
pair sat at one lightness (`tests/core/test_chart_theme.py`). The axis is fixed, as the share
ramp's is, and it is in the unit the floor is in (§24 *The statistic*): t from −10 to +10 on a
screen scored by the moderated t, the default (`match_key.T_EXTENT`; a block sits by its t and
the selected cluster's mark reads *selected t +8.8 (12.4×)*, the t first because it is what the
axis and the floor are, the effect after it because it is what the cluster did), ρ from −1 to +1
for a trend scored by the effect, ÷20 to ×20 for a log ratio (`match_key.effect_extent`). **The
scale's ends say *less* and *more*, and *no difference* sits under the middle between the floor's
ticks** (`match_key.effect_end_words`, `NO_DIFFERENCE_TEXT`): the units (*t −10*, *÷20*) are in
the bar's tooltip (`match_key.scale_text`) and on the selected cluster's mark, because a reader
could not say at a glance what they meant. The selected cluster is the tall bar;
`match_key.effect_position` is the one map from a value to the axis, for the scale and the blocks
alike (`cluster_filters.effect_shades`, `ScreenChoice.axis_value`), and a block's hover carries
the t beside the effect's words. A profile screen's effect is a size whose sign belongs to each
cluster's own pair, not to the measure (§24 *Designs*), so its axis is one-sided, with one floor
tick, the ends *same* and *differs* and the caption *Differs across session* in place of a
direction (`match_key.one_sided`; `effect_position` takes the magnitude) — **and its ramp is its
own, the neutral middle to ink with no hue** (`plot_theme.size_colour`), never the scale's "more"
half, so the yellow never means *more* on one screen and *differs either way* on another. The
selected cluster's mark names its pair and direction (*selected t 6.2 (9.6× more under SA than
REI)*, `NodeEffect.pair`), and the hover names every block's. A block's hover carries the screen's
sentence for it, the same words the Comparisons page prints (`comparison_report.suggestion`), with
*kept*, *nested under a kept cluster* or *inside the floor* after it.

**While the Filters tab is open, the tree reads the fields, and frames what the screen kept.**
Each block is shaded by the share of its clips that match the Filters tab's own fields
(`Selection.without(CLUSTERS)`): the screen does not narrow that share to its kept clusters, and
with no field on nothing is shaded. The share ramp is lightness with no hue (`ChartTheme.
sequential`, `plot_theme.match_colour`, none of it at the low end): it was blue, which is now the
"less" pole. The clusters the ticked screen kept, at the strictness in force, are **framed on
their own block's edge in the colour of their side** — the blue pole or the yellow, the ink of a
profile's ramp — with a dark line either side so the frame reads on a pale fill and a dark one,
and the screen's sentence in the hover (`cluster_filters.kept_marks`, `NodeIcicle.set_kept`). The
frame is the pole's own colour whatever the size, because it says *kept, and which way*; the size
is the Differences tab's to show. A frame only for the screen, not the reverse: a screen keeps a
handful of clusters, and a filter touches most of them, so a frame for "passes the filter" on the
Differences tab would mark most of the tree.

**Strictness: a screen read looser or stricter than its floor, as a view.** Under the ticked
screen the dock shows **Strictness**, a slider over the family-wise levels
`compare.STRICTNESS_LEVELS` (1%, 2%, 5%, 10%, 20%, 30%, 50%; the screen's own 5% floor among
them), with a line saying the level and what the screen keeps there (*p ≤ 0.2 (looser than its
floor) · 6 kept of 12 clearing*). Every level is a lookup over the one saved ranking
(`Ranking.floor_at`, `kept_at`: the same null read at another quantile, the same peak pick), so the
slider never re-ranks and the page hands the dock each level's kept set and leaves
(`cluster_filters.ScreenLevel`). The selection carries the level (`Selection.alpha`, `None` for the
floor), the leaves are that level's, the key's floor ticks move to that level's floor, the tree
colours what clears there and a block's hover reads *kept at p ≤ 0.2*. **It is a view**: the
saved screen's floor, `kept` and record do not move, and a label written at a loosened level says
so (below). The purpose is discovery on a cohort too small to clear a family-wise floor — to see
which clusters would need more animals or another experiment — and every such cluster carries the
numbers that say how far it is from clearing (*The node source*): its family-wise p, its own p and
its q, in the hover, on the Comparisons page and in `ranking.csv`. Because a screen scores by the
moderated t (§24 *The statistic*), loosening admits the clusters the animals most agree on, not
the ones with the largest numbers; on a screen scored by the effect it admits by size, which is
what the floor is in the unit of there. The CLI reads a screen at a
level with `temporal-interpret --from-screen cmp_NN --at-p LEVEL`; the slider itself has no CLI
form, since the columns are in the export.

**A label written while a screen is ticked remembers the screen.** There is no promotion of a
suggestion to a label: the suggestion is where to look, the name is the user's, typed from the
clips as any label is. What the screen did was choose the cluster, and that choice has to travel
with the label, because the label scored later across the same column is the effect the screen
selected it for, reported as if found independently — the *labels score a method* failure one step
removed, and nothing fails when it happens. So the state keeps `label_origins` (node → the saved
screen `cmp_NN`, the cluster it kept, the labelled node's relation to it — *kept*, *under*,
*above* — the column the screen contrasted, copies of how the screen read and the kept
cluster's sentence, so the record reads without the comparison, and the strictness the screen
was read at with the kept cluster's p's — `alpha`, blank for the screen's own floor, and
`family_p` / `own_p` / `q` — so a label chosen at p ≤ 0.3 says so for as long as it exists:
the label bar reads *kept at p ≤ 0.2 · family-wise p 0.14, own p 0.0039, q 0.9*, the screen's
page lists it *at p ≤ 0.2*, and a labels comparison's `screened_labels` note says *kept at p ≤
0.2*; `storage/run_labels.LabelOrigin`).
It is written by `name_nodes` / `label_nodes` with the name, travels with the node's *own* label
(a rename in place keeps it; removal, `delete_label` and a merge that drops the node's own label
lose it, in the one edit door), and is restored with the other three maps by `restore_labels`, so
⌘Z and ⇧⌘Z carry it. INTERPRET records it whenever a screen is ticked at labelling time and the
screen points at the cluster (`run_labels.label_origin`, from the `ScreenChoice`); the CLI's
`--from-screen cmp_NN` records the same thing from the saved screen
(`api.comparison.label_origin`) and refuses a node the screen did not keep, nor sits under or
above one. It shows on the label bar under the figures (*From screen cmp_01 (drug vs vehicle ·
paired): more under drug than vehicle, 12.4× · kept*), on the screen's own page as the labels
written from it (`api.comparison.labels_from_screen`, read from the run as it is now), and in
`--tree` as `[from cmp_NN]`. The consumer is §24: a labels comparison records `screened_labels`.

**Query has the same screen.** "Rearing frequency, vehicle versus drug" is the same operation on
the same artifact — any artifact with named labels compares (a labels comparison, §24), which is
why the comparison is generic rather than discovery's.

### The temporal module

Built on the shared analysis pieces — `analysis/segmentation.py`'s detectors and cluster rules,
`analysis/bout_segmentation.py` and banded DTW for `dtw_finch`, `analysis/decode.py` for
`codebook_viterbi` — behind the protocol above, with the authored phase, the §19 artifact, and the
INTERPRET / AUDIT / BROWSE workspace described in this section.

**`dtw_finch` carries a known scale wall and ships anyway.** Its fit is O(bouts²) banded DTW, which
is infeasible at study scale until bout retrieval is made sub-quadratic (an open item), and
`max_segments` bounds the cost only by evenly subsampling the bouts, which stops being representative as a study grows. This is a documented limitation, not a
defect to fix before shipping: the method is useful at the scale people work at today, and the
subsample is recorded in the run so a result is never quietly built on a fraction of the data.
The surface must **state the subsample when one was applied**, since a fit over 20% of the bouts
and a fit over all of them look identical in every other respect.


**The module's default method is `codebook_viterbi`** (`TemporalClusteringConfig.method`, read by
the CLI, the API and the GUI form alike), chosen on cost: it matches `dtw_finch` on vocabulary quality,
and its cost grows linearly with the pooled video where `dtw_finch`'s
grows with the square of the bout count. `dtw_finch` stays registered and selectable for its
better boundaries on short studies.

### The pose module

**Design, not built.** A pose module was built once and removed on 2026-11-10; what follows is its
design. Its protocol (`PoseClusterMethod`) and an empty `POSE_CLUSTER_METHODS` registry stay in
the tree, so restoring it starts from registering a method. Its shape follows the protocol above, and the open questions are genuinely open — what a
posture vocabulary should be fitted on and how the minimum-duration rule interacts with the
decode. These are decided when it is built and measured,
not asserted here.

Two things are settled. **The vocabulary is postures**, so representatives are frames or short
runs, and the interpretation surface is about recognizing a shape rather than watching an action
unfold. And **the research's frame arm is not the design** — it is evidence that frame-level
labeling holds up against the bout path at matched vocabulary size, which is a reason to build this
rather than a specification of it.

### Surfaces

The temporal module ships the full set (surface parity), and the pose module must when it is built;
both are authored runs, so §21's two required behaviors apply: the result view freezes while a compute operation runs, and Refine is suppressed.

- **API**: launch the fit, then the authored operations (`name_nodes`, unlabel, audit, finalize,
  reopen), `rebuild_node_cache`, the node cache reader (`api/node_view.py`: `clips`,
  `match_shares`) that every surface reads a node through, and `api/cluster_filter.Selection`.
- **CLI**: launch, status and attach; `temporal-interpret` with `--tree` (every
  node: id, parent, member time, share, clip count, label), `--name-node`, `--rebuild-cache`,
  `--finalize` and `--reopen`; `cluster-montage` (with `--sort random`); and export. The montage, the
  shaded tree, the cluster view's filters and labelling by clicking stay
  GUI-only, recorded in `docs/cli.md`'s omission ledger.
- **GUI**: a config form per module and the interpretation workspace above.

### Guards

- **Registry parity** over each module's methods: every one builds, declares a config model and
  `has_noise`, returns a tree, and implements `member_leaves`, `member_places`, `instances_for`,
  `rank_members` and `node_separations`.
- **Membership without a cut**: for every method, the members `member_leaves` puts under a node are
  exactly the members the old cut path gave that node at a granularity where it was on the cut.
- **Old and new agree**: for every method and several granularities,
  `instances_for(fit, tree.leaf_nodes_at(g), params)` equals the granularity cut `instances_at(g)`,
  which both protocols still declare and both temporal methods implement for that reason
  (`tests/cluster_methods/test_hierarchy.py`).
- **Ranking is whole and one-directional**: for every method `rank_members` leaves no member under
  the node out, its distances ascend, and a child ranks only members its parent ranks
  (`tests/cluster_methods/test_member_order.py`, `test_hierarchy.py`).
- **Old shards open**: a fit saved with the retired layout fields loads and ranks as one without.
- **Label resolution**: every member's `label` is its deepest labelled ancestor's name and its
  `label_path` the chain above it; a member under no label is `unlabelled`; the same fit and labels
  always yield identical instances.
- **Member time is measured**: a fit analysed at every third frame, from a video whose
  advertised fps differs from its true rate, has node time equal to its timestamp span, not frames ÷
  fps; a lone frame takes the run's measured step.
- **Leaf additivity**: on a real fit, every cached node figure equals the same figure computed
  directly over that node's members.
- **Cache schema**: names, dtypes and chunks as tabled above; `schema_version` first; the staged
  atomic write; a `built_from` mismatch reported as stale.
- **Every node's clips are the method's instances of it**: for every node, `NodeView.clips` equals
  `instances_for` with the node as the only group; `codebook_viterbi`'s order equals
  `rank_members`' (`tests/api/test_node_cache.py`, `test_node_members.py`).
- **Old runs open**: a cache with member lists, thumbnails and a floor, a state with
  `cache_floor_s` and params with `min_cluster_s` all read, and the keys are dropped on read.
- **Filters are exact**: `match_counts` equals a brute-force count over the members for a region, a
  metadata value, a range and their AND.
- **Time in session trims and normalises** (`tests/api/test_cluster_filter.py`): recordings of
  different lengths, a run straddling either end (only its time inside counts), a clip matched by
  its midpoint, a multi-part recording's span, a tiny range, a recording with no span, and the AND
  with a region and a metadata value. `tests/api/test_session_lengths.py` and
  `tests/gui/test_session_lengths_page.py` run it over a real fit of twelve recordings from 8 s to
  120 min (two imported in parts through the real importer, one on a clock starting at 7 s, one with
  dropped frames), every expected answer in exact fractions: clip counts and every node's share for
  ten ranges, complementary ranges summing to the whole, and the slider dragged on the page.
- **No concentration flag without its floor**: a node drawn from the run's own distribution does not
  flag; an injected single-recording node does.
- **Selecting a node never decodes**: the node-select path opens no video
  (`tests/gui/test_cluster_workspace.py::test_selecting_a_node_reads_no_video`).
- **The three label states round-trip** through the artifact: noise, unlabelled and labelled come
  back distinguishable. Measures report unlabelled time as `unnamed_time`; `noise_time` is
  deliberately absent until a producer emits noise (§23, *The measures*).
- **Runs written with a granularity still open**, and their snapshots still read as pinned.
- **Label leakage** (`tests/pipeline/test_label_leakage.py`): permute the labels, predictions must
  not move. Discovery is unsupervised, so this must hold trivially, and it is exactly the kind of
  place a "seed the labels from ground truth" convenience would quietly break it.

### Deferred

- **Apply to new recordings.** Out of the MVP, with its cost recorded: refitting after adding
  recordings changes every cluster id and **invalidates the user's naming work**, which is
  expensive human effort. That is the reason to build apply eventually, and the reason its slot is
  reserved in the protocol now.
- **Arena-normalized positions.** The Location filter's map uses frame-normalized positions, labelled as
  such. Mapping each recording onto its drawn arena, so recordings filmed at different angles line
  up, is the next step, with a frame-coordinate fallback for a recording with no arena.
- **Cross-run comparison.** Cluster 7 in one fit is unrelated to cluster 7 in another; the bridge,
  when wanted, is the user's names.
- **Sub-quadratic bout retrieval** for the temporal module — the scale wall above.


## 23. Measures

**Measures** turn the §19 artifact into numbers: how much of a behavior, how often, how long, in
what order. They are pure functions — artifact in, tidy table out — with no executor, no Qt, and no
knowledge of which module produced the instances they read. Every producer therefore inherits every
measure, and a measure added later applies to every artifact already on disk.

The output is one row per `(recording, subject, label, label_to, window, measure)`:

```
recording_id, subject, label, label_to, label_display, label_is_remainder, window_id,
window_start_t_ns, window_end_t_ns, measure, value, denominator_ns, n
```

**A parent label's row counts only its remainder, and says so** (amended in #24, owner decision).
Labels nest (§22) and a bout counts under its deepest label, so a label with another nested under
it measures only the time its children did not take. The numbers stay as they are; the row is
*named* for what it counts. `label` keeps the plain name, because it is the join key a comparison
groups on (`compare.MEASURE_KEYS`) and a user joins to `instances.csv` on. `label_display` is what
a person reads — `"<label> (other)"` on such a row, the label itself otherwise — and
`label_is_remainder` marks the row, so a CSV reader can tell without parsing a name. "Is a parent"
is decided once, from the manifest's `label_paths` (`measures.remainder_labels`); a producer with
no hierarchy has none. A comparison stores the list as `remainder_labels` on its record and carries
the same two columns in `stats.csv` and `ranking.csv`, and every surface that titles a measurement
reads them.

`denominator_ns` is not optional and not a convenience column — see *Denominators* below.

**`n` is the number of instances behind `value`**, not a measure in its own right: how many
durations `duration_mean` averaged, how many onsets `rate` counted, how many spans contributed to
`time_fraction`. It is what tells a reader whether a mean over three bouts means anything. For the
`n_instances` measure `value == n`, a harmless tautology and a free consistency check. (An earlier
draft called this column `n_instances`, which collided with the measure of that name and left the
column reading as either "the count" or "the sample size".)

`n` means the same thing in `storage/behavior_audit.py`, which returns it beside a precision as the
number of decisions behind that number. What differs there is `n_instances`, which the audit uses
for a bin's *population* rather than as a measure name — so a reader moving between the two meets
one word with one meaning and one word with two. Worth knowing rather than renaming: `n` is the
sample size in both, which is the one that appears in output a scientist reads.

**`label_to` is empty for every measure but the pairwise ones.** A transition is a property of an
ordered pair, and the alternatives are worse: encoding `"rearing→grooming"` into `label` forces
every reader to parse it and breaks joining on `label`, and giving transitions their own table
forces §24's comparison to grow a second reader — the branch-per-producer shape this whole artifact
exists to avoid. One nullable column keeps a single table and a single code path, and §24 groups by
`(label, label_to)` with no special case.

### Everything is a windowed query, and the answer is exact

A window is a `(start_t_ns, end_t_ns)` interval per recording. **The whole recording is simply the
window that covers everything**, so there is no separate "per recording" code path. The artifact is
intervals, so any windowed question has an exact answer with no binning anywhere: time in a window
is the overlap of the instance intervals with it, and count is the number of onsets inside it.

Windows come from four places, and the model is the same for all of them:

- **Whole recording** — the default.
- **Absolute** — offsets from the recording start (`0–600 s`, `600–1200 s`, …).
- **Normalized** — fractions of the recording (`first third`), for cohorts whose recordings differ
  in length.
- **Event-referenced — not built** — relative to an event (`0–60 s after each lever press`). The
  event lives in a stream (§3), which `query_eval` reads only for its `behavior`-role ground truth;
  this is the shape that would read an event stream. The stream join itself is a follow-on, but
  the window model is designed so it drops in rather than needing a second mechanism. Built kinds
  (`configs/windows.WindowSpec.kind`): `whole`, `absolute`, `normalized`, `sliding`.

**A sliding window sampled at a step is how a measure becomes a curve** — "rearing rate across the
session" is the same measure evaluated at many windows, not a different kind of measure. Binning is
therefore never a property of the data; it is a choice of where to evaluate, and it never changes
what an answer is.

### One convention: half-open, and it reaches further than windows

**A window is `[start_t_ns, end_t_ns)`.** Both bounds inclusive puts a frame on a shared boundary
in *both* neighbouring windows and credits it a full period to each — so two halves of a recording
sum to more than the whole, and a window one period long reports twice its own length. Additivity
is one of this section's guards, so half-open is not merely tidier; it is what makes the guard
passable. Measured on a 100-frame fixture: four adjacent quarter-windows summed to 103 periods
before the fix and 100 after.

**An instance is measured half-open too, as `[t_ns, end_t_ns + period)`** — even though §19 and §3
*store* `end_t_ns` inclusive. The two conventions are deliberate and serve different roles: an
instance is a thing with a first and last frame, which inclusive storage says exactly, and it
matches the behavior stream so an artifact re-ingests unchanged. But *measuring* it inclusive
computes `end_t_ns - t_ns`, which gives **zero for a single-frame bout** and under-reports every
bout by one period. A bout over frames 10–12 covers three frames, so its duration is three periods,
and `[t₁₀, t₁₂ + period)` gives exactly that.

That is the same reasoning `frame_view.observed_mask` applies when a run's span ends at
`last_row + period`: a row covers its sampling interval, not just its own instant. Durations and
denominators are then commensurable — both are counts of frame-periods — which is what
`time_fraction` requires in order to mean anything.

**Coverage's span end follows the same rule**, and this one is easy to get wrong because the
failure is a refusal rather than a wrong number: `coverage.end_t_ns` is the **span end**, not the
last frame's timestamp. A fully-observed 10-frame recording at 1 kHz has `observed_ns = 10000` and
a last frame at 9000, so a producer that writes the last frame's timestamp as `end_t_ns` claims
more observed time than its own span is long and `write_behavior_output` refuses it — with a
message about deflating rates, which is baffling when the recording is perfectly correct.

### Denominators, and why the frame view is required rather than convenient

**Every rate and fraction carries the observed time of its own window.** Video duration is not
recording duration is not tracked time: comparing a 20-minute and a 60-minute recording on raw
counts is meaningless, and comparing them on rate is still wrong if one lost 30% of its frames to
tracking.

Windowing sharpens this into a requirement. `coverage.csv` gives `observed_ns` for a whole
`(recording, subject)`, which cannot answer "how much of the first ten minutes was observed" —
that needs to know *where* the unobserved rows are, not just how many. The frame view's `observed`
mask is exactly that, and summing it over a window slice is the only exact way to get a windowed
denominator.

So the frame view (§19) is not merely a convenience for consumers who prefer arrays: **windowed
measures cannot be computed correctly without it.** A denominator taken from the whole recording
and applied to a sub-window silently inflates or deflates every rate in proportion to where the
tracking gaps happen to fall.

### A measure declares how it meets a window boundary

An instance that straddles the edge of a window has to be handled, and the right handling **differs
per measure**. Getting this wrong does not fail — it produces plausible numbers that do not add up.
Three rules, declared by each measure:

| rule | meaning | measures |
| --- | --- | --- |
| **by onset** | the instance counts if it *starts* in the window, and its full extent is used | count, rate, duration mean/median/sd, latency |
| **by overlap** | the instance contributes the part of itself inside the window | total time, time fraction |
| **by pair** | reads a bout in sequence with its neighbour; a bout is selected when it *starts* in the window. `inter_bout_interval` uses a pair only when both bouts start in the window; `transitions` looks each selected bout's successor up in the unit's whole sequence, so a transition counts where its *first* bout starts | transitions, inter-bout interval |

The distinction is not pedantry. **By onset** keeps counts additive across adjacent windows and
keeps a duration statistic honest — measuring duration *by overlap* would truncate every instance
at the boundary and report bouts shorter than they were. **By overlap** keeps total time additive.
A measure using the other rule breaks one of those properties, and the resulting table still looks
fine.

### The measures

| measure | rule | notes |
| --- | --- | --- |
| `n_instances` | onset | distinct `instance_id` — see below |
| `rate` | onset | `n_instances / denominator_ns` |
| `total_time` | overlap | |
| `time_fraction` | overlap | `total_time / denominator_ns` |
| `duration_mean` / `_median` / `_sd` | onset | full extents, not truncated |
| `latency_to_first` | onset | from window start to the first onset; undefined when none |
| `inter_bout_interval` | pair | gap between consecutive instances of one label |
| `transitions` | pair | one row per ordered pair: `label` is the from, `label_to` the to. Label-scoped, reading `Context.unit_bouts` for each bout's successor — the whole unit's sequence is what "next" means, but the *from* is this measure's own label |
| `noise_time` | overlap | time the method assigned nowhere. **Not implemented** — see below |
| `unnamed_time` | overlap | time in real clusters the user has not named |

**A pairwise measure is tabulated, not plotted.** `transitions` answers per ordered pair, so a
vocabulary of N behaviors is up to N x N values — 400 for a 20-label vocabulary, each of which
would become a two-point dot plot of a single matrix cell. `comparison_report._measure_figures`
skips a `fans_out` measure for that reason: the pairs are in `stats.csv`, in the per-measure
tables, in the contrast grid and in the GUI's Measures tab, one row each. What a transition
*matrix* should look like on a comparison screen is an open §24 question, and drawing N x N wrong
answers is not a way of answering it.

**`noise_time` is specified and deliberately not implemented.** §22 gives noise a *reserved*
`source_label`, but neither §19 nor `storage/behavior_output.py` fixes what that value is, so a
measure choosing one here would invent a convention no writer agreed to, and its guard would be
unexercised — a guard that passes forever. It lands with the first producer that emits noise.
The rationale lives beside the registry in `analysis/measures.py`, which is what keeps it from
drifting from the code.

**`noise_time` and `unnamed_time` are separate, deliberately.** "12% of observed time is noise" and
"31% is in clusters you have not named" are different facts about a run's completeness, and a
single "unattributed" bucket hides which one you are looking at. A method that declares
`has_noise = False` reports zero noise **by construction** (§22) and must not read as more complete
than one that admits it leaves frames out.

**Counting distinct `instance_id` versus counting rows is a per-measure declaration, never a
default, and `behavior_output.bouts()` is the collapse.** A cluster the user gave two names emits
two rows over one bout (§19), so a measure that counts rows counts it twice. This is not
hypothetical: the §19 audit shipped four functions counting rows, and precision came out **0.667
where the truth was 0.5** while a stratified draw spent a whole bin's budget showing the same clip
twice (fixed in #195, which is why `bouts()` is public). `n_instances` and `total_time` must count
the bout **once**; a measure asking "how many label assignments" counts
rows. Nothing about the arithmetic looks wrong when this is confused, which is why it is declared
rather than inferred.

### What a measure refuses

Refusals are part of the contract, because the alternative is a number that cannot be read:

- **Pooling a per-recording label space across recordings.** When the manifest says
  `label_space: per_recording`, cluster 3 in one recording is unrelated to cluster 3 in another
  (§19, §22). A measure may compute per recording and must refuse to aggregate across them.
- **Transitions on an overlapping label set.** "The next label" is undefined once two labels can be
  active at once, so `transitions` declares `requires: ("exclusive",)` and is not computed against
  a manifest that says `overlap: allowed`. The refusal is *upstream of the run*, not an exception
  from it: every picker greys the measure out with `unmet_condition`'s reason attached, and naming
  it explicitly writes no rows rather than a column of zeros. A measure that cannot answer a
  question is absent from the answer, with the reason on the surface that offered it.
- **A window with no observed time.** Zero denominator is not zero rate; it is no measurement, and
  the row is emitted with a null value rather than a division.

### The registry

`@register_measure(measure_id, **fields)` (`analysis/measures.py`) is a decorator that stores a
`MeasureSpec` in `MEASURES` — the same shape as every other registry in the tool (Tenet 1). The
spec's fields are `rule` (the window rule, required), `label` and `description` (required: what a
person reads), `unit`, `additive`, `duration_shaped`, `scope`, `fans_out`, `requires` and
`needs_denominator`. A measure is `fn(selection, ctx) -> value`: `selection` is the `Selection` of
instances its `rule` let through, `ctx` the `Context` (the window, the frame period, the observed
denominator, the unit's ordered bouts). Adding one is additive and reaches every artifact and every
surface at once. `requires` names manifest conditions (`CONDITIONS`: `exclusive`, `unnamed_labels`) so a refusal is
declared rather than raised from inside the computation.

### Surfaces

- **API** — `Experiment.measures(run_id)` returns a `Measures` handle (`api/measures.py`):
  `.table(windows=…, measures=…)` returns the tidy table, and `.export(out_dir, windows=…,
  measures=…)` writes `measures.csv`, `frames.csv` and `measures.yaml`. No measure selection means
  every measure the artifact can answer (`applicable_measures`).
- **CLI** — `detail measures export --run … --out …` writes the same three files, which is also the
  **frame-level export** users asked for: `frames.csv` expands the frame view to a per-frame
  classification. The instance table is deliberately not copied — the artifact already writes
  `instances.csv` beside the run, and a second copy is two files that can disagree.
  `detail measures list` prints every measure with its unit and window rule.
- **GUI** — the **Measures** panel (`gui/pages.MeasuresPanel`) on an analysis run's page, carried
  as its own view by the query and temporal clustering workspaces; the comparison surface (§24)
  reads measures too.

### Guards

- **Additivity.** A measure declared additive must sum across adjacent windows to the value over
  their union. This is the guard that catches a wrong window rule, which is otherwise invisible —
  every individual number looks reasonable.
- **Duration is not truncated.** A bout straddling a window edge reports its full length under an
  onset-rule measure. Revert the rule to overlap and this must fail.
- **Every rate row carries its denominator**, and the denominator is the *window's* observed time,
  not the recording's. Assert the number against a fixture with tracking gaps deliberately placed
  inside one window and not another.
- **A twice-named cluster does not double count** `n_instances` or `total_time` — the collapse
  through `bouts()`, against a fixture where one bout carries two labels.
- **A single-frame bout has one period of duration, not zero.** The inclusive-storage /
  half-open-measurement split is invisible in any fixture whose bouts are long, and wrong by one
  period per bout everywhere else.
- **`n` matches what the measure consumed**: for `n_instances`, `value == n`; for a mean, `n` is the number averaged. A measure that reports a value over three bouts and an `n` of thirty is not caught by any arithmetic check, only by asserting the pair.
- **Each refusal fires**, with its message — data-driven over the registry so a new measure
  declaring `requires` is covered the day it registers.
- **Noise and unnamed stay separate** when `noise_time` lands (it is deliberately absent today, see
  *The measures*): a `has_noise = False` method's zero must then be distinguishable from a method
  that measured no noise. Not built, because there is no `noise_time` to guard.

### Deferred

- **Continuous per-frame values conditioned on a behavior** — mean speed during rearing, a
  regressor's output within a bout. Those need the second artifact shape §19 defers. What a
  location-export run already holds per frame is measured per recording by §24's *locations
  source*: distance, time moving, speed and body shape need no behavior to be defined.
- **The stream join** that makes event-referenced windows real. The window model accommodates it;
  the join is its own work.
- **Cross-artifact measures.** Comparing labels from two different producing runs needs the naming
  bridge, not a measure (§19, §22).


## 24. Comparison

A **comparison** answers the question both research tasks exist to serve: *does this differ between
these groups?* It filters a set of recordings, groups them by whatever columns of `metadata.csv`
(§11) the question needs, aggregates §23 measures, and saves the result — record, tables and
figures together.

**Naming.** "Comparison" means this: the scientific comparison across recording metadata. It needs
no labels and compares animals. Not "evaluations", because that word already names a run scored
against ground truth: each `analysis query-eval` run *is* an evaluation.

```
  analysis/<an_NN>/        an evaluation — one config scored against ground truth
  comparisons/<cmp_NN>/    a comparison  — this section; groups, no labels
```


**Scope is descriptive.** Tables, effect sizes, figures, and a tidy raw export. **No models and no
parametric tests.** The only p-values are read off the ranking floor's own permutation null — a
family-wise p, an item's own p and a Benjamini–Hochberg q (*The floor read as numbers*, below) —
which calibrate a ranking rather than fit a model. Behavioral counts are overdispersed, sessions within
an animal are not independent, and getting inference right needs choices this tool has no business
making on a user's behalf. What it owes them instead is a clean `stats.csv` — one row per
observation with its factors attached — which is exactly what a mixed model in R or statsmodels
wants as input.

### The pipeline

```
filter recordings  ->  measure over windows  ->  aggregate through the path  ->  group  ->  render
```

- **Filter** — which recordings take part: an explicit list (`recordings`; unset, every recording
  in the source artifact). A filter expression over metadata (`drug != vehicle`) is not built.
- **Windows** — §23's window model, unchanged: whole recording, absolute, normalized, or a sliding
  series. A sliding series makes the comparison **time-resolved** — curves per group instead of
  points — with no separate machinery.
- **Measures** — any registered §23 measure, with its denominator. A pairwise measure keys on
  `(label, label_to)` rather than `label` alone (§23), which needs no branch here: the group key is
  a tuple either way.
- **Aggregate** — collapse through the declared path (below).
- **Group** — partition the resulting units by the group-by keys and summarize.

**Group-by keys are uniform**: any column(s) of `metadata.csv`, plus `subject`, plus the window.
That is what makes "drug A vs B", "animal 1 vs 2", "A vs B within animal 1", and "first third vs
last third of the session" one operation rather than four features.

### The aggregation path — declared, never defaulted

The single most consequential choice in a cohort comparison, and the one a tool will otherwise make
silently.

Compare a treatment against baseline in a cohort where one animal contributed four sessions and
another contributed one. Pool recordings and the first animal votes four times, so the group mean
tilts toward whichever animal happens to have more sessions. Average within animal first and each
animal votes once, at a much smaller n. **Both are legitimate and they answer different questions.**

So a comparison declares an **aggregation path** — a sequence of metadata columns to collapse
through before the group statistic:

- `[]` — each `(recording, subject, window)` is an observation.
- `[animal]` — recordings average within animal first; each animal is an observation.
- `[animal, cage]` — and so on, in order.

Any column or columns, so the general case is supported rather than a list of anticipated shapes.
Three rules go with it:

- **There is no default *where it matters*.** A comparison names its path or does not run —
  **when the cohort has repeats**, which is the only case where the choice changes the answer. If
  no subject contributed more than one of the recordings being compared, `[]` and `[animal]` give
  the same numbers, so the surfaces run with one observation per recording and say so rather than
  asking a question with one answer. A silent default in the repeated case is a wrong answer that
  looks like a right one; a demanded choice in the unrepeated case is a demand nobody can act on.
- **The path is printed on every output** — every table, every figure caption, `summary.md`, and
  `stats.csv` as a column. A plot that does not say what one point is cannot be read.
- **The floor's permutation respects it** (below). Shuffling group labels at the recording level
  when the path aggregates to animal would destroy the very structure the path exists to preserve,
  and would make the floor far too easy to clear.

### What a group summary reports, and what it must show

Per `(group, label, measure, window)`: **n**, the central value (mean and median), a spread, and
**every underlying observation**.

- **Show the points.** With a few animals per group, a bar and an error bar hide the entire
  dataset behind two numbers. The default figure is a dot plot with every observation drawn,
  group summary overlaid. This is not a style preference — at small cohort sizes, the points
  *are* the result.
- **n is never implicit.** A group of 3 and a group of 12 rendered identically is the most common
  way a comparison misleads.
- **A group with n = 1 has no spread**, and reports that rather than a zero.
- **Missing is not zero.** A label with no instances in a recording that *was* analyzed is a real
  zero and enters the mean (§19 coverage). A recording that was not analyzed is missing and must
  not. The comparison reports each group's coverage alongside its n, and flags a group whose
  coverage is partial rather than quietly averaging over whatever was there.

### Effect size, and the floor every ranking ships with

For a two-group contrast the comparison reports the **raw difference** and the **ratio** of the
central values. Not a standardized difference: dividing by a pooled spread is a variance model, and
a variance model is the inference this section deliberately does not do. What a within-unit
ranking *scores* by is a different matter, settled below (*The statistic*): the effect a user
reads stays the raw ratio; the order and the floor are taken over the effect studentized across
the units, which is a ranking statistic calibrated by the same permutation, not a model of the
variance.

**Any ranking of many things by difference ships with its floor.** The discovery difference screen
(§22) ranks tens of clusters by how much they differ between groups. With tens of clusters and a
handful of animals, ranking *always* produces a most-different cluster whether or not anything is
there — and it will be run every time, on every comparison, which is exactly the condition under
which a spurious top hit becomes a finding.

The floor is a **permutation**: shuffle the group assignment **among the aggregated units** (the
path's units, per above), recompute the whole ranking, repeat, and report the distribution of the
top effect. It answers "how big is the biggest difference you would see from nothing?" and is drawn
on the same axes as the ranking. This is null calibration, not a hypothesis test — the same
discipline `query_eval` already applies with its shifted-time and shuffled-rank nulls (§16), and
the same rule as "a headline metric ships with its floor" (Tenet 9).

It is computed **in the same pass**, so a ranking cannot be rendered without one — *or* without
saying why there is none. A permutation the data cannot move produces no floor: with one
observation per group every relabelling leaves every fold change unchanged, so the null's maximum
is arithmetically the observed maximum and nothing can ever exceed it. That is withheld with
`no_floor_reason` rather than printed as a threshold nothing cleared, because a blank column and a
column of "no" read as "nothing differed" when the truth is "this design cannot tell".

**Known before measuring, refused before measuring.** Three of those shapes are counts of
recordings and their metadata: a within-unit design where fewer than two units hold the levels
its null moves, a between design where every group holds at most one unit, and fewer than two
groups. `compare.no_floor_before_measuring` is the one rule (the same sentence the ranking
carries afterwards; its counts are upper bounds on the ranking's, so it never refuses what
measuring could have floored), asked in `prepare_comparison` before the measuring pass. A screen
(`source: nodes`) exists to say which clusters clear the floor, so without one it has no output:
it is refused with the reason and what would run (`api.comparison.screen_refusal`), in the launch
dialog before Run, by `comparison run` and by the API alike. A comparison of labelled behaviors
with no floor is still a description, reported as measured: the dialog warns in its preview and
the run goes ahead. The reasons only measuring can know (no measure present in `min_units`
units, no spread for the moderated t) keep the after-measuring backstop.

### Designs: the permutation respects the unit, one level down

The between-unit fold change above is one design, and the difference screen's pre-build study
(kept outside the public tree) measured that it
cannot reach what a cohort study is usually about: over ~1,350 sparse clustering nodes with five
animals a side its family-wise floor sits at a log ratio above 6, so nothing clears, while the same
nodes under a within-animal contrast clear a floor the shuffled control cannot (15 kept against 0
of 20 seeds). The within-unit designs are the same `Observation` list — `key` is the unit after
the aggregation path, `group` is the level — with the permutation the design allows.

**The user chooses the variables; the tool resolves the design.** A comparison names its
`group_by` column and its unit (`aggregate_to`), as it always has, and the design follows from
what that column does *within* the unit. It is resolved when the comparison runs, written to the
record with the sentence every surface prints (Tenet 9), and never configured — "there are too many
combinations to do all", and an override has no meaning a user could state. The one lever kept is
naming two `levels`, which turns any column into a two-level contrast.

| the chosen column … | resolves to | effect per unit | null |
| --- | --- | --- | --- |
| is constant within each unit, or there is no unit, or several columns are chosen | **between** (above) | — (group means over units) | group shuffled **among** units |
| varies within units and holds two values, or the user named two `levels` | **paired** | `log((b + ε) / (a + ε))` over the unit's level means | the sign flipped per unit; **exact** over all 2ⁿ flips when n ≤ 10 |
| varies within units, is numeric with three or more values, no `levels` named | **trend** | Spearman ρ over the unit's levels | the levels shuffled **within** each unit |
| varies within units with three or more non-numeric values and no `levels` | **profile** | the largest `|mean log ratio|` over every pair of levels, each pair the paired contrast over the unit's level means, the pair on the item | the levels shuffled **within** each unit, the same largest pair taken each time |

A recording with no value for a grouping column — blank, or a spelling of *not available* such as `NA`,
`nan` or `none` (`cohort.MISSING_TOKENS`, behind the one predicate `cohort.missing`) — sits outside the
comparison, not only the design: it is left out before anything is grouped (an antagonist session in a dose
column is not a dose, and not a group of its own), the record names it under `left_out`, and the summary,
the detail page, the launch preview and `comparison run` say so in one sentence
(`comparison_report.left_out_sentence`). `levels` on a
numeric column is a paired contrast of those two doses. Prevalence — present at one level and absent
at the other — is not a design of its own: a paired item carries the units present at each level beside
its log ratio, which is what that statistic added. A unit with several recordings at one level
contributes their mean, which is the aggregation path doing what it already does.

**A profile** (three or more named levels within each unit) gives every pair of levels the paired
contrast and ranks each measure by the largest `|mean log ratio|` among the pairs defined for it,
with the pair on the item (`Effect.pair`) and the sentence naming it first: *differs most between
REI and SA: 9.6× less under REI than SA (7 of 8 ID in that direction)*. A pair is defined where
`min_units` units are present at either of its levels, as a paired contrast is; a measure with no
defined pair is undefined. The null shuffles each unit's levels within it and takes the same
largest pair each time, so the floor already pays for choosing the pair and no second correction
is applied; it is sampled, `(k!)^n` configurations being past enumerating, and `Ranking.exact`
says so. With two levels the profile is the paired contrast, item for item. A **between design
over three or more groups** ranks the same way, each item the largest contrast between any two
groups with the pair on it (`Contrast.pair`) and the null the same largest gap under the label
shuffle, where it had no ranking before. `Ranking.profile` says when items carry their own pair,
`Ranking.level_values` lists what was ranked over, and `ranking.csv` carries `pair_a` / `pair_b`.
A numeric column with three or more values stays a trend: the lever for a non-monotone
dose-response is naming two `levels`. A profile's effect is a size whose sign belongs to its
pair, which is what INTERPRET's key reads (§22).

Three rules the within designs share, each from a measurement:

- **An effect over fewer than `min_units` *present* units is undefined, never large.** Presence is
  the unit's level mean of `total_time` reaching `presence_s` at some level. Units with a value
  are not the count: a node present in three of eight rats has a mean log ratio over eight, five of
  them zero, and the trend promoted one-to-three-minute nodes on exactly that. `min_units` is
  resolved from `min_units_fraction` (default one half) × the units in the cohort and recorded.
- **ε is one second of member time** (`pseudo_count_s`), converted into the measure's own unit
  through the observation's denominator, so a node absent at one level has a finite, honest ratio
  and the family-wise floor is not owned by nodes a few animals barely touched. The between design
  over **labels** keeps its unbounded "appeared / vanished" — a label's appearance is a finding,
  not a sparsity artefact — and takes ε only over the node source.
- **The floor is exact when it can be.** The sign-flip null of n units has 2ⁿ configurations; at
  eight units the sampled 95th percentile moved three nodes in and out of a kept set with its seed.
  n ≤ 10 enumerates every flip and the floor is deterministic; `Ranking.exact` says which.

Beside the effect, every within-design item reports `consistency` (units whose own effect shares
the sign), `n_units`, `n_present`, and whether the sign survives leaving any one unit out. The
floor is the same 95th percentile of the largest |effect| over the family, in the same pass, and is
withheld with `no_floor_reason` when nothing can move (fewer than two units holding the levels
the null moves, since a unit at one level of a paired contrast cannot be flipped; no measure
reaching `min_units`).

### The statistic: scored by agreement, not by size

The number a within-unit ranking scores each measure by, and the floor is taken over, is the
measure's **moderated t** (`statistic: moderated_t`, the default): the mean per-unit effect
over its spread across the units, ``t = mean / ((sd + s0) / √n)``, with `sd` the sample spread
of the per-unit effects (`0` for a single unit), `n` the units with an effect and `s0` the median
`sd` over the family's defined measures, fixed from the observed configuration and applied to
every draw of the null (SAM's fudge factor, chosen the simple way). The effect itself — the mean
log ratio, the mean ρ — is what every surface prints; the t decides the order and the verdict
and sits beside it (`Effect.statistic`, `item.score` is whichever the ranking used,
`Ranking.statistic` / `Ranking.s0`).

**Why, from a measurement.** On the mean effect alone the family-wise floor belongs to the
sparsest measures: a cluster a few animals barely touch swings ±4 log units under any sign, so
under the flips it produces the family's largest mean more often than anything else, and the
floor it sets is one no consistent cluster can reach. On an eight-rat cohort the floor sat at
16.8× while the two families eight of eight rats agreed on at 12–15× sat under it, kept 0; the
same screen studentized kept 4, those families. The t makes the null max a fair race: every
measure's t is on one scale, so each has a similar chance of owning the family's largest, which is
the balancing property of the max-t permutation (Westfall–Young). The consequence for §22's
strictness is the one it was built for: loosening admits the clusters the animals most agree on,
not the ones with the largest numbers.

**What stays and what moves.** The null is the design's own, unchanged: under a sign flip the
sum of squares is invariant, so each draw's spread follows from its mean and the observed
configuration stays one of the enumerated draws (an exact own p is still never zero); a trend's
or a profile's within-unit shuffle recomputes the spread per draw. A profile chooses each
measure's pair by the t, with `s0` the median over every defined pair of every measure, so the
floor pays for the choice by the statistic it is taken over. A family where no measure's units
disagree has no `s0`: the ranking withholds the floor with its reason rather than divide by
nothing. `statistic: effect` scores by the mean effect itself. A between design has no per-unit
effects to studentize and ranks on the effect whatever was asked; the record's resolved
`ranking_statistic` says what ran (Tenet 9), and a record written before the field existed
carries neither and ranks on the effect, as it did when it was written (§3: a saved comparison is
never re-read as a different comparison).

**The floor read as numbers.** The null the floor is taken from already says, for every item,
how far it is from clearing, and a between ranking keeps each measure's own draws beside the
family's largest for the same purpose (`_magnitudes`). Every item carries three (`Effect` /
`Contrast`, `ranking.csv`, the ranking tables' *p (family · own)* column, INTERPRET's hover):
- **`family_p`** — the share of the null's draws whose *largest* score reaches the item: the
  floor read as a p. Defined as the smallest family-wise level at which the item clears
  `Ranking.floor_at` (the percentile rule rounds its position in the null, so a bare count sits one
  draw off it at some levels), which makes `family_p ≤ α` and `score > floor_at(α)` agree at
  every level and `Ranking.clears_at` one rule for the dock's slider and the hover's p; at
  `FLOOR_ALPHA` (0.05) it is `clears_floor`. Exact over an enumerated null. The score is the
  item's t or its magnitude, whichever the ranking is on (*The statistic*).
- **`own_p`** — the share of the item's *own* null draws reaching it, ignoring the rest of the
  family: whether the units agree on it by itself. Exact over an enumerated null (the observed
  configuration is one of its draws, so it is never zero); over a sample the observed labelling
  counts as one more draw, the Monte Carlo convention, so its floor is `1 / (draws + 1)`.
- **`q`** — Benjamini–Hochberg over the family's own p's: the smallest false-discovery rate at
  which the item is among the discoveries. Computed because it is the standard answer to a family
  this size, and reported knowing that at eight animals the exact null has 256 configurations, so
  the smallest own p is 1/256 and no cluster can reach q ≤ 0.05 among ~1,350; it becomes
  informative at twelve to sixteen units.
All three are `None` where the effect is undefined or unbounded, or where there is no floor. What
they are for is §22's strictness: a screen read looser than its floor shows the clusters a larger
cohort might clear, each with the numbers that say how far it is. They are not a model: the
permutation is the same one the floor was always built from, and `stats.csv` remains the handoff
to whatever a user tests with.
**The permutation respects the design**: shuffle a paired contrast's levels *across* units and the
pairs break — a unit dealt two of one level drops out of the draw, the null's tail widens and the
floor rises past an effect every unit showed — the guard that pins the between permutation to the
path's unit, applied one level down.

### The node source

A clustering run's **unlabelled** nodes are what the difference screen (§22) exists to screen, and
the §19 artifact collapses them by design. `source: nodes` measures the run's **node cache**
instead — the leaf-additive tables, no frame view, seconds rather than minutes — and emits rows in
`measures.COLUMNS`' own schema, one `label` per candidate node (`node:<id>`), so `observations`,
the summaries, the tables and the figures consume them unchanged. A candidate is a node with at
least `candidate_floor_s` of member time over the run (default 30 s, Phase 0's floor), the root
excluded: the cache has no floor of its own now that every cluster has clips, and every candidate
widens the family the floor is taken over. Four measures, from the registry:
`time_fraction` (the ranking family), `rate`, `total_time`, `duration_mean` — bouts are the node's
leaf runs merged where adjacent — with the recording's analysed time as the denominator. The
hierarchy travels beside the observations as `nodes.csv` (`node_measures.HIERARCHY_COLUMNS`:
`label`, `node`, `parent`, `depth`, `ancestors`, `time_s`, `share`), and
the cache's `built_from` digest is the source the record cites; a stale cache is refused, naming
what changed, never silently read.

**Nesting is suppressed by peak picking.** A node and its parent carry nearly the same time, so
the clearing set is full of slices of one behaviour. Over the items that clear, in rank order: keep
the best, drop its ancestors and descendants, repeat. The below-floor "show more" list is thinned
the same way. `suppress_nesting` is a config field so a flat source turns it off; `Ranking.kept`
is the kept set, `None` where suppression was not applied.

**A large family is exported as its shown set.** 1,350 nodes × four measures is 5,000 dot plots
nobody asked for, so for the node source the per-measurement tables and figures cover the kept
nodes and the first ten below the floor after suppression; `ranking.csv` carries every candidate
with its effect, floor and verdict, and `stats.csv` every observation, so nothing is lost by it.

### The locations source

A behavior artifact says *what* an animal did; a location-export run (§16) says where its body
was and what shape it had, a row per frame. `source: locations` turns one finished
location-export run into one value per recording, track and window for each measure below, in
`measures.COLUMNS`' own schema with an empty `label`, so `observations`, the designs, the
tables and the per-measure figures consume them unchanged. Nothing is re-measured from masks: a
comparison stays seconds, and the per-frame values are the run's
(`analysis/location_measures.py`).

| measure | definition | unit |
|---|---|---|
| `distance` | length of the smoothed path of the body's centre, within unbroken stretches of tracking | m |
| `time_moving` | share of path time faster than `location_moving_threshold` | fraction |
| `speed_moving` | mean speed over that time; no value for an animal that never moved | cm/s |
| `body_length`, `body_width` | median of the run's per-frame length and width | mm |
| `body_elongation` | median of length ÷ width, frame by frame | ratio |
| `body_area` | median area of the body | cm² |

A unit is never `""`: the report reads that as a count, whose pseudo-count is one bout. A
ratio's unit is named and printed as nothing.

Four settings, config fields like any other and written to the record: `location_sample_fps`
(the run's rows are thinned to at most this many a second by the export's own rule, default
5, so runs exported at different rates are measured alike; a run exported slower is read as it
is), `location_smooth_s` (1 s), `location_moving_threshold` (2 cm/s) and
`location_jump_speed` (7.5 body lengths per second). The rules they implement, each from a
failure on real recordings (detail#70):

- **A raw centre path over-counts**, so it is a running mean before it is measured.
- **A gap or a jump is not movement**: the path breaks at a gap in tracking and at a raw step
  faster than the jump speed, and nothing is joined across a break. The rule is a *speed*
  because a length per step means a different thing at every sampling rate
  (`tests/configs/test_unit_conventions.py` is the guard that said so).
- **A window takes the steps that fall in it**: the path is smoothed once per track, so
  distance over adjacent windows sums to the distance over their union.

**What it needs of the run.** The body columns (§16 *The body*): a run from before them is
refused, naming the run to make again. A distance calibration on every recording compared:
one exported without is refused, rather than measured in pixels beside recordings measured in
millimetres. A snapshot is refused, as it is over nodes; a finished run is never rewritten, and
the record's `location_export` says how it was exported (`sample_fps`, `remove_thin_parts`,
`thin_part_scale`), because whether the body has its tail changes every body measure. Drift
reads each compared recording's manifest and the size of its table.

### The verdict

What an item must do to count as a finding is a property of the ranking, recorded as `verdict`
and asked of the ranking (`Ranking.clears_floor` / `clears_at`), never worked out by a surface
from `floor`.

- **`family`** (labels, nodes): clear the floor, the 95th percentile of the largest score over
  the whole family under the null. Right for a screen: hundreds of clusters in one unit, where
  ranking by difference always produces a most-different one.
- **`own`** (locations): reach its own test, `own_p ≤ 0.05`, the share of the null's draws in
  which *this* measure differed at least as much. Right for a handful of measures a person
  chose, in different units. One floor over a distance, a share of time and a length belongs
  to the noisiest of them: on the cohort this was written for, every rat's body was shorter
  under naloxone (10 of 10, by 6%) and no floor shared with a distance that halved could be
  cleared by 6% of anything. The false-discovery rate over the family (`q`) rides beside it
  for a reader who wants the correction.

Under `own` the ranking is ordered by `own_p`, smallest first, because the sizes are not
comparable; it ranks on the effect whatever `statistic` asked (the moderated t borrows spread
across the family, which means nothing across units), and `ranking_statistic` says so; it
draws at least 10,000 shuffles (`draws` is recorded as drawn), because its family is small and
at 200 the smallest p a trend can show is 0.005; `ranking.csv` leaves `floor` blank; and no
ranking figure is drawn, since bars of size across units against a floor nothing was judged by
would chart the wrong question. A trend's null ranks each unit's data once rather than once
per draw (`compare._spearman_null`), which is what makes that many draws a second and leaves
every saved comparison's numbers exactly where they were.

### The saved comparison

A comparison is a **saved object holding its outputs**, not a transient view. It is cheap to
recompute, but these numbers go into papers and a figure without its inputs recorded is not
reproducible.

```
comparisons/<cmp_NN>/
  comparison.yaml     schema_version first, atomic (storage/comparisons.write_comparison_yaml):
                      id; the config as run (ComparisonConfig.as_record: source_run_id,
                      source_snapshot — resolved — source, group_by, aggregate_to, levels,
                      recordings, measures, windows, draws, seed, presence_s, pseudo_count_s,
                      min_units_fraction, candidate_floor_s, suppress_nesting, statistic); the
                      resolved design: design (between, paired, trend or profile),
                      design_column, design_reason, n_units, min_units, ranking_measure,
                      ranking_statistic (what ranked: moderated_t or effect, the latter for
                      a between design whatever was asked; absent on a record written before
                      the statistic existed, which ranks on the effect); groups,
                      n_observations, and left_out, the recordings with no value for a
                      grouping column; provenance, status, created_at;
                      source: labels only — screened_labels, the labels (as measured, from
                      the snapshot's state) whose cluster a node screen over one of the
                      compared columns chose, each with the screen, the kept cluster and the
                      node (§22): the effect of such a label here is not an independent
                      finding, and every surface says so from one function
                      (comparison_report.screened_note / screened_sentence — the suggestion
                      cell, the summary, the detail page's header, `comparison run`)
  stats.csv           the tidy source of truth — one row per observation with its factors
  nodes.csv           source: nodes only — the node hierarchy the ranking is suppressed over

  written by the run and again by every export (comparison_report.write_result_tables):
  ranking.csv         every ranked item: effect, statistic (the moderated t where the ranking
                      is on it; the floor is in its unit), floor, clears, kept, family_p /
                      own_p / q (the floor read as a p, the item's own p, the BH q over the
                      family), consistency, n_units, pair_a / pair_b (the levels or groups
                      the effect is between)
  significant.csv     one row per item that cleared: the group means and s.e.m., the effect,
                      the statistic, the p it was judged by and which p that is (p_kind:
                      own / family-wise, and the bound where a shuffled test could not
                      resolve lower), own_p / family_p / q beside it, n, and how many units
                      went the effect's way — what a paper cites beside the figures
  units.csv           the per-unit values in wide form: one row per unit and measurement, a
                      column per group, in the unit a reader sees — the shape Prism takes
  unit_effects.csv    the per-unit effects a within-unit ranking averaged (a unit's log
                      ratio of its level means, or its Spearman ρ), which were computed for
                      the ranking and thrown away before detail#76
  methods.txt         a paragraph for a Methods section, generated from the record and its
                      ranking (comparison_report.methods_sentence): which test, what was
                      shuffled, how many draws or that it was exact, which p is reported and
                      at what level, the correction beside it, n, and the smallest p the
                      test can report

  written by `comparison export` (Comparison.export), into this folder or --out:
  tables/             per measure: groups as rows, summary columns
  figures/            one panel per measurement (below), the composite `panels`, a curve per
                      windowed measurement, the difference ranking with its floor
  summary.md          the quick-look, with the Methods paragraph
  manifest.json       provenance + per-artifact SHA-256
```

`run_comparison` writes the first three and the result tables; the figures, the per-measure
tables and the summary are rendered from them on export (`analysis/comparison_report.py`), which
copies `stats.csv` along when exporting elsewhere so the tree is self-contained.

**The figure of one measurement** (detail#76, the paper's group-difference figures as the
template). Bars of the group mean with s.e.m., every unit as a dot at a fixed offset so the same
animal sits in the same place in every bar, and a line joining the same unit across the groups it
appears in; its P on the plot, as a bracket over the groups it compares, only when the ranking
judged the measure to differ (`Ranking.clears_floor`), and nothing when it did not: no stars, no
"n.s.". The P is the one the verdict used, and the symbol says which — `P_FWE` under the family
verdict (a measure that had to clear the family's floor, having been found by screening many),
`P` under the own verdict — written `< 1/draws` where a shuffled test could not resolve lower,
never 0. A test against an ordered variable is labelled with the variable (`dose: P = 0.002`),
never "trend", which a reader takes to mean "not quite". Groups are drawn in the order the design
resolved (`group_order`: the levels a user named, reference first; numeric levels in numeric
order; numeric before text), and coloured by what they are (`group_colors`, over every group of
the record so a filter repaints nothing): the reference level grey, an ordered variable's levels
one hue from light to dark (checked monotone), two levels grey and the hue, three or more named
values distinct hues. The hues are the theme's (`theme_hues`): from blue on white, and from orange
on the dark surface, where the ramp runs from the hue towards light (a dark shade sinks into the
background) and a bar's fill tints towards the surface rather than towards white. The difference
ranking draws what cleared the floor plus the first ten below it, its height following the bar
count, with the measure said once on the axis when every bar is of one measure. The axis names
the thing and its unit (`panel_label`: *Rearing (% time)*,
*Distance (m)*; `MeasureSpec.axis`), a ratio's bars start at 1 with the baseline a labelled
tick, an annotated panel reserves headroom for its bracket and an unannotated one does not, and
nothing is clipped to the axes, so a dot at zero is a whole dot in every output. The composite
`panels` figure holds the chosen measurements (every single-window one, in registry order; over
the node source the clusters the screen kept, in rank order; at most `PANELS_MAX`) as one row or
grid at a double column's width with one legend and panel letters. Every figure is 5.5 to 8 pt
type on thin axes with no top or right spine (`core/chart_theme`), and the SVG is rewritten for
editing by hand (`report/svg_edit.py`: no clipping masks, markers as plain paths, a P label as one
text object in the house face), the PDF has editable type, and the formats are the same default
on every surface (`DEFAULT_FORMATS`: the API, the CLI, the GUI's export dialog and redraw, the
catalogue target).

**The export tree and the primitives under it are §18's, reused wholesale** — `report/style.py`,
`tables.py` (`write_tidy_csv`, `pivot_to_grid`, LaTeX/Markdown/CSV), `figures.py`, deterministic
rendering, publication styling.

**Two qualifications, from checking the seam rather than assuming it:**

- **Not a per-module exporter registry.** A comparison has no member module to dispatch on, so it
  exports through the same primitives by its own entry point (`analysis/comparison_report.py`).
- **The points are never hidden behind the bar.** `figures.grouped_dots` takes one value and one
  error per group, which is the shape this section rejects — at these cohort sizes a bar and an
  error bar alone hide the whole dataset. The comparison's per-measure figures use
  `figures.bars_with_units`, which draws the bar *and* every unit on it, joined where the unit
  recurs; `figures.observation_dots` (every observation with the group's median) remains for a
  surface that wants no summary bar.

**Provenance is §18.5's `capture` / `drift`**: the artifact's producing run, its upstream DAG,
model weights, code SHA, environment. A surface rendering these numbers **warns on drift**, because
a stale comparison is indistinguishable from a current one by looking at it.

**A comparison measures a pinned snapshot, and records which one.** The source artifact is
reprojected on every authored edit (§21), so naming only the run would cite a mutable slot: the same
`comparison.yaml` would describe different numbers after a reopen and re-cut. `run_comparison` pins
before measuring — not after, which would leave a window in which what was measured and what was
cited could differ — and writes the *resolved* `source_snapshot` back into the record, the same
discipline as any other resolved default (Tenet 9). `--snapshot` re-runs against an earlier pin.
Records written before the field existed carry no `source_snapshot` and resolve to the run directory,
so nothing already on disk is stranded.

**Drift covers the source artifact too.** `capture` records the hash of *what was measured* and
`drift` compares it against the run's artifact *now*, so a source analysis that has been reopened
and re-cut is named on export. The item means **superseded, not invalid** — the numbers were
measured from a pinned snapshot and stay exactly reproducible; what changed is that a re-run would
not match them. That is the same thing every other drift item already means (a moved code SHA does
not corrupt a computed `stats.csv` either), and the line says so, because a warning read as "your
result is broken" invites discarding a good one.

**Scoped to one producing artifact.** Cluster 7 from one fit is unrelated to cluster 7 from another
(§19, §22), so a comparison reads one artifact. The bridge across runs, when wanted, is the user's
names — a name can mean the same thing in two runs in a way an id never can.

### Surfaces

- **API** — `Experiment.run_comparison(config)` → a `Comparison`; `.export(out_dir=None,
  formats=…)`, `.drift()`, `.record()`, `.ranking()`, `.kept_nodes()`; `Experiment.comparisons` /
  `get_comparison` / `delete_comparison`.
  `api.comparison.screen_comparison(experiment, config)` is the same computation with nothing
  written, through the one `rank_record` mapping the export and the detail page use. No surface
  calls it: the live tab it was written for was not built, and INTERPRET's Differences line (§22)
  reads a saved node comparison back instead. The tests use it to hold a live ranking and a saved
  one together.
- **CLI** — `comparison run` (`--source-run`, `--snapshot`, `--recording` (repeatable: the
  filter), `--group-by`, `--aggregate-to`, `--source labels|nodes`, `--level A --level B`,
  `--measure`, the window options, `--draws`, `--seed`, `--presence-s`, `--pseudo-count-s`,
  `--min-units-fraction`, `--candidate-floor-s`, `--no-suppress-nesting`, `--statistic
  moderated_t|effect`), printing the design it resolved and what the ranking scores by; `comparison export`, `comparison list`, `comparison delete`. `--aggregate-to` is **required when the cohort has
  repeats**, per above, and the refusal names the column it found and how many recordings it
  covers.
- **GUI** — the experiment's **Comparisons** section: list, launch dialog (source, the column,
  the unit, two levels to contrast, with the design it will run shown before Run), and a detail
  view with the tables, the Methods paragraph, the per-measurement panels, the composite, the
  time-resolved curves and the difference ranking — the within-design ranking as effect, units
  in that direction, present, floor and a one-line suggestion per item — plus an export button
  (off-thread).

### Guards

- **The aggregation path reaches every output.** A table, figure or `stats.csv` written without it
  fails — this is a source-scan-plus-render check, because the failure is a plot that is merely
  ambiguous rather than wrong.
- **Aggregation is correct on an unbalanced design.** Fixture: one subject with four recordings and
  one with one. `[]` and `[subject]` must give *different* group means, and both must match a
  hand-computed value. A path that silently pools would pass a balanced fixture.
- **Missing is not zero.** A recording outside coverage does not enter a mean; a recording inside
  coverage with no instances does. Assert both numbers, not that the call returned.
- **No ranking without a floor** — the difference screen cannot render one, enforced at the API.
- **The permutation shuffles at the path's unit level**, not the recording level. Revert it and the
  floor must become measurably easier to clear.
- **The permutation respects the design.** A paired contrast's levels shuffled across units
  instead of its signs flipped within them moves the floor measurably, and past the effect.
- **The negative control, as a test.** A synthetic cohort with no signal keeps nothing after
  suppression in at least 19 of 20 seeds, and a planted node clears in every seed.
- **`min_units` leaves an effect undefined, not large**; **suppression is a peak pick** (no kept
  node is an ancestor or descendant of another); **the exact floor does not move with its seed**.
- **The moderated t, by hand and by what it is for** (`tests/analysis/test_compare_designs.py`):
  over three animals and eight flips every t, `s0`, the null, the floor and the own p's equal a
  reference written out in plain Python; a measure four animals agree on at 2× ranks below a
  tenfold swing on the effect and above it on the t with a smaller family-wise p and the same
  effect; a profile's pair is chosen by the t; a family with no spread gets no floor and its
  reason; a record without `ranking_statistic` ranks on the effect and one naming the t on the
  t; every table, `ranking.csv`, the floor sentence and the record's note carry the t where it
  ranked and not where it did not.
- **The node source is leaf-additive**: every row equals the figure computed directly over the
  node's members.
- **Design resolution is one function**: the launch dialog's preview, the CLI's printed line and
  the record's `design` come from `compare.resolve_design`, and a config the GUI emits is
  enumerated over `ComparisonConfig.model_fields`.
- **n and coverage on every group row.**
- **Determinism**: a re-export is byte-identical (§18.5 L1), and the permutation is seeded.
- **A P on the plot is the verdict's P, drawn only where the verdict holds**
  (`tests/analysis/test_comparison_paper_figures.py`): a bracket appears for exactly the items
  `clears_floor` keeps, carries `P_FWE` under the family verdict and `P` under the own one, is
  the bound form where the shuffled P is zero, and names the column rather than "trend"; groups
  draw in design order with colours fixed by the record's full group list; every unit is drawn
  and joined where it recurs; `significant.csv` holds exactly the kept items with the means the
  figure drew; the per-unit effects in `unit_effects.csv` average to the ranking's effects; the
  Methods paragraph names the draws, the exactness and the verdict the ranking used; the SVG
  holds no clip path and no `<use>`, and one text object per P label; and the default formats
  are one constant on every surface.

### Deferred

- **Inference.** Deliberate. `stats.csv` is the handoff.
- **An ordered series and a paired contrast side by side on a shared y axis** (a dose series
  and a second treatment against the same control, as the paper's figure draws them): two
  comparisons' worth, since one comparison resolves one design (detail#76, item 11).
- **Example bouts for a cluster that differs** — the run's most typical members, one per
  animal, drawn as outlines beside the cluster's bars, which is what tells a reader what
  "cluster 1354" is (detail#76, item 12). The figure of the cluster's numbers is here; the
  figure of what the cluster looks like is the clip engine's, and goes beside it.
- **The within-unit change as a panel of its own** (each unit's difference or ratio to the
  reference), for the small paired effect on a measure with a large spread between units,
  which reads as "no difference" in bars of the mean (detail#76).
- **Occupancy maps** over the locations source: time per place, averaged over each group's
  units. Not a row of `stats.csv`; it needs a frame the recordings share (a region to register
  to, or a check that the views agree) and a figure and table of its own (detail#70).
- **A between-unit relationship to a continuous variable.** A numeric column that is constant
  within each unit resolves to a between design scored by its largest pairwise contrast, not to
  a trend across units (detail#70).
- **Region measures** from `roi_relations.csv` (time in a region, entries), and the share of a
  recording a measure was made from as a measure of its own.
- **Windows over the node source.** The cache's leaf runs make §23's window model applicable; a
  drug effect is time-locked and a behaviour that changes within a session is its own question.
  Whole recording first.
- **A between-unit screen over nodes that can clear** — a presence restriction per group, so
  sparse nodes stop owning the null's tail. Open from Phase 0, not run.
- **Cross-artifact comparison**, above.
- **Event-referenced windows**, which need the stream join (§23).
- **Per-subject metadata.** `metadata.csv` is keyed on recording today; when a recording carries
  several animals, it becomes keyed on `(recording, subject)` and the aggregation path gains
  `subject` as a real column rather than a synthetic key (§19 *Subject*).


## 25. Surface Visibility (deploy toggles)

A build need not show every capability the tree contains. A developer surface means nothing to a
user who has no ground truth, and a module superseded by a newer one should stop being offered, yet
both can stay in the tree — editable, tested, and switchable back on. This section is the
mechanism. Nothing is hidden today (the Status table says so, checked by `tests/test_surfaces.py`).

### What kind of toggle this is, and what follows

In the standard taxonomy (Fowler) a feature toggle is classified on two axes — how long it lives
and how dynamic the decision is. This one is a **permissioning toggle keyed on build audience**:
long-lived, statically decided, never per-request.

Three consequences, and they remove most of the design space:

- **Source-controlled config, edited by a developer.** No runtime store, no admin UI, no dynamic
  evaluation. The decision is made once, in the source, and travels with the build.
- **It carries no removal debt.** A *release* toggle is technical debt with a carrying cost and
  should be ripped out as soon as its feature lands; the usual advice is to file the removal ticket
  when you add the flag. A permissioning toggle inverts that — the surface genuinely is not for
  that audience, and hiding it is the permanent, correct state.
- **It is not a user preference.** There is no "show advanced features" checkbox. A user-facing
  expert mode is a different feature with a different owner (the user, not the build), and this is
  not it.

### One table, asked at enumeration points

The governing rule is to **decouple the decision point from the decision logic**: never scatter
`if visible(...)` through the surfaces. One table, one accessor, and each surface asks it exactly
once — where it *enumerates* what exists, not where it uses one.

```
                  detail/surfaces.py          ← the single table
                  ┌───────────────────────┐
                  │ HIDDEN = { ... }      │
                  │ is_surfaced(name)     │
                  └───────────┬───────────┘
                              │  asked at ENUMERATION points only
   ┌───────────┬──────────┬───┴──────┬─────────────┬────────────┐
   ▼           ▼          ▼          ▼             ▼            ▼
 GUI        GUI launch  CLI        generated    user docs   ✗ Python API
 sections   lists and   commands   module       (marked       never asks
 (declared  pickers     hidden=    pages        prose,
  list +    filtered    True; CLI  (surfaced_   stripped
  opener)               pickers    config_      at build)
                        filtered   models)
```

**Fifteen enumeration points, not N use sites.** `tests/test_surfaces.py::ENUMERATION_POINTS` lists
them: the GUI rail and overview sections, the GUI processing and analysis launch lists, the GUI and
CLI pickers for query methods, trackers and input-image constructors, the GUI embedding-model
picker and model manager, the CLI model list, the CLI's help, and the generated module pages. That
is what makes a data-driven guard possible; a scattered check cannot be enumerated and therefore
cannot be tested for completeness. User docs are checked by their own guards (below).

**An entry is a capability, and a capability has manifestations.** One thing a user might reach
goes by different names in different surfaces — a GUI section `<things>` and a CLI group
`<thing>` — so the entry declares those names beside the decision:

```python
HIDDEN = {
    "<capability>": Hidden(
        reason="why a user of this build has no use for it",
        until=None,                          # what would unhide it; None = permanent
        awaiting=None,                       # set to stage the entry (see *Staging*)
        section="<things>",
        modules=("<module_id>",),            # registered module ids; several for a chain
        methods=("<method_id>",),            # query / cluster / tracker / embedding / constructor ids
        cli=("<thing>", "<other-group> <subcommand>"),
        docs=(("docs/cli.md", "### `<thing>`"), ...),
    ),
}
```

An earlier draft of this section specified **one flat id space with two resolvers**. Implementing
it produced the evidence against that: the correspondence between the section id and the command
name had to be restated as an alias map *inside the test* — a mapping in the wrong place, and one
that would go stale silently. A reverse index over the declared manifestations means a call site
passes whatever id it already holds and needs no translation of its own.

**A capability can span more than one command.** A command that writes a record only another
group's command can read must be hidden with it, or surfacing one without the other strands the
user mid-workflow. Hence `cli` is a tuple.

### User documentation is an enumeration point

`docs/cli.md`, `docs/gui.md`, `docs/concepts.md` and `docs/glossary.md` describe surfaces in
hand-written prose that no registry knows about. A build that hides a capability everywhere else
while its user guide still teaches it is worse than either state alone. So the prose is **marked**,
`<!-- surface:<capability> -->` … `<!-- /surface -->`, and `surfaces.strip_hidden` drops a marked
block whose capability is hidden. Three readers call that one function, so they cannot disagree
about a marker: the site build (`scripts/docs_surface_hook.py`), the in-app help panel
(`help.extract_section`), and the guards. An entry also names the **doc anchors it owns**; a guard
asserts each is absent from the page *after stripping* (marked or deleted), and where a page marks
the capability, that at least one of its anchors exists in the raw source, so a mistyped anchor is
caught. Marking rather than deleting is what keeps the prose editable and switchable back on.

Anchors are **declared, not grepped for**: "comparison" names several different things in this repo —
§24's group contrast, the O(bouts²) DTW bout comparison, and the word's ordinary use — and a
keyword scan would fire on all of them.

**Hiding a *module* is not like hiding a section**, and the difference is not obvious. Each module's
reference page is *generated* from the config-model registry while its nav entry in `mkdocs.yml` is
*static*, so hiding one needs the generator taught to skip it **and** the nav line removed.

Both halves are built. `configs/describe.py::surfaced_config_models` is the docs seam, and it is a
*separate* accessor rather than a filter on `iter_config_models` on purpose: that registry also
feeds `detail help`, the unit-convention scan, the advanced-form guard and the help-anchor map, and
a hidden module's parameters must keep meeting those. **Hidden code that stops being guarded is
hidden code that rots**, and switching it back on becomes a surprise rather than a toggle.

The nav is kept in step by `tests/test_docs_modules_nav.py`, and the two failure directions are not
symmetric. A nav entry with no page breaks `mkdocs build --strict` — loud, if late. A **page with no
nav entry fails nothing at all**: mkdocs publishes every page whether the nav mentions it or not, so
a hidden module would keep a live, reachable reference page. Generated pages are not on disk, so the
`docs/`-scanning guard cannot see them; this is the guard for the generated half.

**Activation is guarded end to end**, verified by activating a staged entry: the §25 docs-anchor
guard demands the prose be marked or removed, the nav guard demands the line be removed, and only
then does `--strict` build clean — stripping the marked prose is also what clears the cross-links
that would otherwise dangle.

### Staging: declared, validated, not in effect

An entry may be waiting on something. Hiding a module before its replacement ships would remove a
capability with nothing behind it — but leaving it undeclared means re-deriving where it appears on
the day it is switched off.

So an entry may set **`awaiting`**: it is declared and validated, and stays surfaced. `awaiting` and
`until` read alike and are opposites — `awaiting` is what must happen before the entry takes effect,
`until` is what would undo it once it has.

**The guards invert for a staged entry, which is the whole value.** In effect, a doc anchor must be
**gone**; staged, it must **exist**. Without that second assertion a doc rewrite would silently
invalidate an anchor string, and the entry would enforce nothing on the day it was switched on — a
failure indistinguishable from a working toggle. Non-vacuity covers staged entries too, so module
ids and command paths are checked now rather than when they matter.

### The refactor it forces, which is worth doing anyway

The GUI's experiment sections were **hardcoded** — a sequence of `self._item(…)` calls
in the rail builder. A guard cannot be data-driven over hardcoded calls, so the sections become a
**declared list** the builder iterates. That is the same "drive N siblings from one table rather
than N parallel implementations that will drift" rule the codebase already applies elsewhere
(`Experiment._JOB_RUN_KINDS` is the pattern), and it is a prerequisite here rather than a bonus.

**The check belongs at the opener, not at the rail.** The GUI already funnels *every* way into a
page — a rail click, a landing button, a table double-click, the jump after a run — through that
destination's single opener. That opener is the toggle point. Guarding only the rail leaves the
other three doors open, and the surface a user reaches by double-clicking a table row is exactly
the one nobody re-checks.

### Surface by surface

| surface | mechanism |
| --- | --- |
| **GUI sections** | declared section list + the check in the destination's opener |
| **user docs** | marked prose stripped by `surfaces.strip_hidden`; declared anchors per entry, absent after stripping while in effect, present while staged |
| **GUI modules** | filter `ANALYSIS_GUI_MODULES` (and the other GUI registries) at enumeration |
| **Method and model pickers** | `Hidden.methods`: every GUI and CLI picker over `QUERY_METHODS`, the cluster-method registries, `TRACKERS`, `EMBEDDING_MODELS` and the input-image `CONSTRUCTORS` filters at enumeration; a hidden id still resolves, so a saved run that names it opens |
| **CLI** | `hidden=True` on the command/group — click's own support |
| **Generated docs** | CLI pages **inherited**: mkdocs-click's `show_hidden` defaults to false, so the CLI flag covers `--help` and the docs site with one mechanism. Module pages: `configs/describe.py::surfaced_config_models`, with the static nav kept in step by `tests/test_docs_modules_nav.py` |
| **Python API** | **never consulted** |

**The API is deliberately not a consumer.** Tenet 2 makes it the boundary; the GUI and CLI are
surfaces over it. A deploy toggle is about what a lab user *encounters*, and research
studies drive the API directly — a toggle that broke a study would be a toggle nobody dares use.

### The escape hatch

`DETAIL_SHOW_ALL=1` reveals every hidden surface without editing the source, so a developer working
against a deployed build does not have to modify code to see one. The trade-off is recorded rather
than hidden: a determined user could find it. That is acceptable here — the audience is lab
scientists, the hidden surfaces are inert rather than dangerous, and the alternative is developers
editing the table locally and accidentally committing it.

### Guards

- **Surface parity**, data-driven over the table: a hidden id appears at **no** enumeration point
  (`test_a_hidden_capability_is_absent_from_every_enumeration_point`, one case per entry of
  `ENUMERATION_POINTS`). A new surface is covered once it is added to that table.
- **Non-vacuity**: every id resolves to a real registered module or a real declared section. A
  typo hides nothing, errors nowhere, and reviews clean — the only way to catch it is to assert
  resolution.
- **Hidden code stays tested.** The suite runs with everything visible, so a hidden module cannot
  rot and switching it back on is a toggle rather than a surprise. (This mirrors the standard
  advice to test the expected production configuration *and* the all-on configuration; testing
  every combination is unnecessary, because these toggles do not interact.)
- **Guard the guard**: a test proving the parity scan would catch a leaked surface, so it cannot
  pass vacuously. **Watch for a filter guarded only in principle** — every in-effect capability is
  a section plus commands, so reverting the *module* registry filter broke no test until one was
  added that hides a live module. A guard whose subject nothing currently exercises is a guard that
  passes forever.
- **A staged entry's anchors exist**, the inverse of the absence check (see *Staging*).



---

## 26. Keypoints (pose estimation)


> **Revised 2026-10-21: detection stores unassigned instances; tracking assigns them.** Before
> this revision `keypoint_detect` ran on a tracking run and stored rows per (track, part), so a
> pose already belonged to a subject when it was written, and the student was handed to tracks by
> box IoU inside the detect stage. That cannot hold more than one animal's pose per frame
> without a subject, and it put identity outside tracking. Every multi-animal pose tool surveyed
> (DeepLabCut, SLEAP, idtracker.ai) stores poses per frame with no identity and assigns identity
> in a separate, re-runnable step; NWB `ndx-pose` and `movement` hold only the assigned layer. So:
> `keypoint_detect` takes a **detection run** (`upstream = "detection"`) and writes **one store
> per part, one row per pose instance, no subject**; `keypoint_track` takes that run **and a
> tracking run on the same detection run**, assigns instances to subjects by mask (*Assignment*),
> writes the assignment beside the detect rows as tracking does beside detections, and then runs
> the method's own cleanup per subject. Patch matching crops around each **detection** rather than
> each tracked subject; the student runs on the whole frame and keeps every pose at or above
> `min_pose_confidence`. Runs in the per-track layout are refused with a re-run message (§26
> landed 2026-10-09). Multi-animal object tracking is not part of this revision: the surfaced
> trackers still hold one subject, and nothing here assumes one.

A **keypoint run** places named body points on the animals in each frame — the nose, the ears,
the tail base, whatever the user's skeleton names — and a **keypoint track run** gives each
placement to a tracked subject and settles it over time. Its consumers are an
embedding run that turns each subject to face one way before the model sees it (§14 *Egocentric
alignment*) and the export targets below; an audit and analysis modules that read points beside
embeddings are not built (*Auditing a keypoint track run*, *Analysis and export*). One stage, several **methods**, chosen per run like a detection model:

| Method (`model_id`) | Training | What the user supplies | Weights |
|---|---|---|---|
| `dinov3_match` (default) | none | a skeleton, and clicks on a few example frames | the DINOv3 ViT already in the catalog (§5, §14 `PatchFeatures`) |
| a distilled student (`TrainedYoloPose`) | §12 *Distill keypoints*, from a finished run | nothing at run time | `models/<id>/`, promoted into the catalog with `role: keypoints` and its skeleton; or a catalog pose model restored from a zip or imported as a `.pt` with a supplied skeleton (§12 *Keypoint models in the catalog*) |

The pretrained SuperAnimal models and imported SLEAP/DeepLabCut exports that #626 built are
**deferred** (*Deferred*, below); the protocol leaves their slots.


### Skeletons: names are labels, structure is declared

A keypoint's **name is arbitrary** — DETAIL never derives meaning from it at run time, and never
guesses a role from it (#626's `infer_roles` and the editor's *Guess from names* are gone: a
guessed side is a silent swap waiting to happen). The only structure DETAIL reads is declared
beside the names, and every piece of it is set by the user:

```python
@dataclass(frozen=True)
class Skeleton:                               # core/skeleton.py
    names: tuple[str, ...]                    # column order of every stored array
    edges: tuple[tuple[str, str], ...] = ()   # drawn, and read by the skeleton decode
    pairs: tuple[tuple[str, str], ...] = ()   # left/right symmetry pairs: (left name, right name)
    layout: tuple[tuple[float, float], ...] | None = None   # where the editor drew each point,
                                              # (x, y) fractions of its canvas; outside equality
                                              # and the digest; nothing downstream reads it

    def validate(self) -> None: ...           # Tenet 17
    def digest(self) -> str: ...              # sha256 over names, edges, pairs — the identity (below)
```

**Layout.** The skeleton editor draws the points on a plain canvas, and where it drew them is kept
with the skeleton (`layout`, per column, fractions of the canvas) so the editor reopens a
skeleton as it was drawn; presets ship one, and a skeleton without one opens on
`core/skeleton.py::default_layout`'s schematic (midline down the middle in column order, each
pair's left member to the left). It is a picture of the skeleton, not part of what it means:
`validate` checks only that it places every point inside the canvas, `digest` and equality
ignore it, and no run, store or method reads it.

This is SLEAP's skeleton (`nodes`, `edges`, `symmetries`); DeepLabCut stores less. There is **no
body axis and no `side` on the skeleton**: no keypoint tool puts the heading there (keypoint-MoSeq
calls its skeleton "for visualization only" and keeps `anterior_bodyparts` / `posterior_bodyparts`
as its own config), and here it is the heading definition an embedding run chooses (§14). The
**midline** is the points in no pair. Which member of a pair is the animal's left is the pair's
order; which *image* side that falls on is learned from the clicks, per pair (*Patch matching*).

**Validation** (`Skeleton.validate`): at least one name; names non-empty and unique; a pair names
two existing, different points and no point is in two pairs; an edge names existing points and is
not a self-loop; no duplicate edge in either direction.

**Presets.** Four built-in skeletons (`quadruped_overhead_12` 12 points, `quadruped_side_5` 5,
`quadruped_overhead_8` 8, `midline_3` 3 — the two research skeletons and Quick pose's two, roles removed),
each with a layout, shown as **read-only rows** of the skeleton library (below). **Renamed
2026-10-09** (the user's review: named for the body plan and view, not a species) from
`rodent_overhead` / `rodent_side` / `rodent_simple` / `elongate`; labelled *Quadruped overhead
view (12 points)*, *Quadruped overhead view (8 points)*, *Quadruped side view (5 points)*, *Midline
(3 points)*. An outright rename, by the user's choice: the old ids no longer resolve. A set or run
saved under one still holds its own copy and runs, and the example-set picker still places it by
digest, and the library's *Used by* count matches it by digest too (§3 *Ids are never reused*). A run counts its
skeleton by the **digest of the copy it holds**, never by a name or a library id — two
experiments that used the same skeleton produce runs with matching identity.

### The Keypoints manager: the skeleton library

> Decided 2026-10-08 from the user's fourth review; built (the library, the manager and the
> picker, which the keypoint detect form and the distill dialog both use). Supersedes the
> per-experiment example-sets page and its skeleton editor, and skeleton templates.

Skeletons are defined in one place, a **machine-wide library**, reached from the top bar's
**Keypoints manager** mode (between *Model manager* and *Quick tools*). A lab reuses a few
skeletons across every experiment, so they belong to none of them.

**Storage.** One directory per skeleton under `model_cache_dir()/skeletons/<sk_NN>/skeleton.yaml`,
beside the machine-wide imported-model catalog (§5), so machines that share a model cache share
their skeletons:

```yaml
schema_version: 1
id: "sk_03"
name: "rat overhead"
created_at: "2026-10-08T21:00:00Z"
skeleton: {names: [...], edges: [...], pairs: [...], layout: [...]}
digest: "9f2c…"            # Skeleton.digest(): names, edges, pairs (never the layout)
derived_from: "sk_01"      # set by Save as new / Duplicate; omitted otherwise
```

**Copy on use, with provenance.** Whatever uses a skeleton — an example set, a keypoint run, a
trained student — holds its **own copy**, plus `skeleton_ref: {id, name, digest}` naming the
library entry it came from. Nothing reads the library to run: a missing or unmounted library never
breaks a set, a run or a model.

**Dependents freeze a skeleton.** A library skeleton is **editable and deletable only while
nothing uses it**. Its dependents are found by a **live search** of the experiments the app knows
(`locations.list_experiment_dirs`) for example sets, keypoint detect and track runs, embedding runs
aligned on them, and trained students whose `skeleton_ref` holds it — matched on the ref's
**digest**, since a library id means something only on the machine that minted it: a ref naming
this entry by id and digest counts for it alone (so a duplicate of a used skeleton stays
editable); a ref whose id names a different skeleton here counts for every entry holding its
digest; a ref with no digest is matched on its id. The search runs off the UI
thread behind a busy overlay, when a skeleton is selected and again on Save or Delete. While it
has dependents, **Save** is replaced by **Save as new** (the edit becomes a new skeleton,
`derived_from` the old one) and **Delete** is greyed, listing what uses it. Once the last dependent
is deleted, the skeleton is editable and removable again. An experiment the app does not know
about is not counted; it holds its own copies, so nothing of it breaks. Presets are always frozen.

**The screen** (revised 2026-10-08 after the user's in-app review: the page must not open on an
editor). The page is a **table**, the shape of the app's other list pages: *Name · Kind · Points ·
Used by* (the presets as read-only rows), **+ New skeleton** and **Import…** above it, **Edit…**,
**Export…** and **Delete** below, double-click to edit. No Duplicate button: editing a preset or a
used skeleton already saves a copy (`skeleton duplicate` stays in the CLI and API). **+ New
skeleton** and **Edit…** open the **skeleton editor dialog**: *Name*, the plain `SkeletonCanvas`
(the scene editor's `ArenaCanvas`, subclassed; no video frame behind it) with the scene editor's
gestures (click adds a point, drag moves, right-click removes a point and its edges, Shift+click a
second point joins or unjoins an edge, scroll zooms, right-drag pans), the selected point's *Point*
name, *Pair with*, *Side*, *Remove point*, then **Save** (or **Save as new**) and **Cancel**, which
asks when there are unsaved edits.

### Example sets follow the model that needs them

An **example set** is the frames on which a user clicked a skeleton's points: the input of a model
that **`needs_examples`** (`dinov3_match`). It belongs to an experiment (its frames come from that
experiment's videos) and holds a copy of its skeleton with `skeleton_ref`. There is **no example-set
page** and no rail node: the example-set flow belongs to the model, and **wherever a model that
`needs_examples` is chosen, the same picker appears**:

```
Skeleton:     [rat overhead ▾]                     (the library, presets included)
Example set:  [overhead rats ▾]  8 / 8 frames      (this experiment's sets on that skeleton)
              [New example set…]  [Mark frames…]  [Delete set…]
```

- **New example set…** opens a window with the set's *Name* and *Skeleton* (opening on the
  picker's skeleton), then the labeling screen on the new set. **Mark frames…** opens the labeling
  screen on the chosen set. **Delete set…** confirms and is refused while a run lists the set.
- **A set's skeleton changes until its first mark** (2026-10-09, the user's review, option C: chose
  both the creation window and a *Skeleton* dropdown in the labeling screen). Marks name points, so
  from the first mark the skeleton is fixed and the dropdown greys, naming the way out (a new set).
  A frame with no marks holds no points and does not fix it. `update_example_set(skeleton=)` is the
  one rule; the screen's greying is a courtesy.
- The picker is one widget (`gui/example_set_picker.py`), shown by: the **keypoint detect** form
  and **Test keypoints…** when the method is `dinov3_match`, and the **Distill keypoints…** dialog
  when its teacher is the patch method (§12). It is hidden for a model that does not need examples.
- A set belongs to a skeleton by its `skeleton_ref.id` when that entry holds the set's skeleton
  (same digest); otherwise (no ref, made before the library or from a bare `Skeleton` or file, or a
  ref whose id names another skeleton here) it matches the first library entry with the same
  digest, and one matching none is listed under *(no library skeleton)*, so no set an experiment
  holds is hidden. An imported set keeps its `skeleton_ref` only when an entry here holds its
  digest, and the ref then names that entry.
- **API / CLI** are unchanged in substance: `create_example_set(name, skeleton)` takes a library
  id, a preset id, a `Skeleton` or a skeleton file; `keypoints examples …` stays the scripted
  route; `keypoints examples edit` renames, and `--skeleton` changes the skeleton until the
  first mark, and
  `keypoints examples presets` is gone (`skeleton list` shows them). The library has its own
  top-level CLI group (`cli_skeletons.py`), `skeleton list | show | create | edit | save-as-new |
  duplicate | export | import | delete`, which refuses `edit --skeleton` and `delete` with the
  dependents listed, as the screen does; `save-as-new` is the screen's *Save as new*.

### Example sets (`dinov3_match`'s input)

An **example set** is the user's clicks on a handful of decoded frames, at experiment level because
one set serves every recording: the held-out-animal results are exactly this — clicks on one
animal, applied to others. Clicks are user input, the way an exemplar clip is in §20; they build
the method legitimately, and nothing here reads ground truth. The same storage, with
`purpose: audit`, holds the frames an audit labels (*Auditing*, below); the two purposes are kept
apart by rule — a frame cannot be added to a set of the other purpose.

```
my_experiment/
└── examples/
    └── ex_01/
        ├── examples.yaml
        └── frames/
            ├── a1f3c9….png      ← the full decoded frame, lossless, named by its content hash
            └── 9c2e71….png
```

```yaml
schema_version: 1
id: "ex_01"
name: "overhead rats"
purpose: "example"                           # "example" | "audit"
created_at: "2026-10-07T10:00:00Z"
skeleton:                                    # a copy (library skeletons are copied on use)
  names: ["nose", "left_ear", "right_ear", "tail_base"]
  edges: [["nose", "left_ear"], ["nose", "right_ear"], ["nose", "tail_base"]]
  pairs: [["left_ear", "right_ear"]]
skeleton_ref: {id: "sk_03", name: "rat overhead", digest: "4b7e…"}   # omitted for a bare skeleton
frames:
  - image: "frames/a1f3c9….png"
    recording_id: "recording_005"
    tracking_run_id: "track_01"
    track_id: 0
    part_id: "part_001"                      # the video part's id (§3), never a position in a list
    frame_idx: 1830                          # within the part; display only
    t_ns: 61061000000
    bbox: [0.53, 0.40, 0.70, 0.86]           # the tracked subject's box, normalized
    sampled_by: "varied:shape+dino"          # how the sampler chose it; omitted when hand-picked
    marks:                                   # normalized full-frame; one per skeleton point at most
      - {target: "nose",      kind: "point", x: 0.61, y: 0.85, visible: true}
      - {target: "left_ear",  kind: "point", visible: false}
      # reserved, no UI yet: kind "region" (rle) · kind "box" (xyxy) · positive: false = "not this"
digest: "9f2c…"                              # over the skeleton copy, every mark, and each image's hash
```

- **Edited in place, written atomically** (`core.atomic.replace`; the labeling screen saves every
  click), **unknown fields kept on read and written back**, so a save by older code never deletes
  what newer code added. No copy-on-write and no `used/` markers — what #626 protected with them is
  protected by the run instead: a `keypoint_detect` run records the set's `digest`, the **image
  names it used** and a **full copy of the marks** (a few KB), so a later edit never changes a run
  and the run never depends on the set still existing.
- **Frames are immutable.** An image is written once under its content-hash name and never
  rewritten; removing a frame from the set leaves the PNG while any run lists it. **Deleting a set
  is refused while a run lists any of its frames**, found by scanning run records — nothing to go
  stale.
- **The digest covers the images by hash**, so swapping an image under the same name changes it.
- **Sampling** (`Experiment.sample_example_frames`, strategy `varied` by default, `spread` the
  alternative): `spread` draws one frame per equal time slice across the chosen recordings, in
  proportion to their tracked frames (Tenet 11's labeling-sampler exception: an example frame's
  index is a label, not a join key); `varied` (#656/#677) groups tracked frames by silhouette
  shape, embeds one stand-in per group with the patch backbone, and picks the most typical frames
  first, then the farthest apart, so a resting animal does not hand the user five copies of one
  posture. Each frame records `sampled_by`. Audit sets always use `spread`: an audit needs an
  unbiased sample. Whether `varied` beats `spread` on accuracy is **not measured** (#656's own
  body).
- **The labeling screen** (*Surfaces*) builds one in the GUI; the API and CLI take a prepared YAML
  as well, so a set can be scripted. Clicks are on the **full frame**, so the stored coordinate is
  the click's own; `dinov3_match` maps it into its crop.

### Shared model layer (§5, §14)

Every keypoint method is a `Model` (§5 *Model protocol*): one base for identity, backends, license,
weights sharing, the install / load contract, `check_backend` and `resolved`. Every method returns
the shared `FrameResult` (§5 *Standardized output*), **one row per pose instance**, with `bbox`,
`confidence`, `points`, `point_scores`, optional `candidates` and method-only `extras`; `track_id`
is left unset — a method never assigns a subject (*Assignment*). A patch-matching method reads
dense features through `PatchFeatures` (§14), a capability the embedding model declares, so one
download serves embedding and keypoints.

### `KeypointModel` protocol and registries

```python
@dataclass
class InstanceFrame:                             # core/keypoints.py
    frame_idx: int
    t_ns: int
    frame: np.ndarray                            # RGB uint8 (H, W, 3), the full decoded frame
    boxes: dict[int, tuple[float, float, float, float]]   # detection_row -> that detection's box, normalized; {} for a whole-frame method
    masks: dict[int, np.ndarray] | None          # detection_row -> its mask (H, W) bool; only when needs_mask

@dataclass(frozen=True)
class ScoreSpec:
    meaning: str                                 # "cosine similarity" | "heatmap peak" | "detector confidence"
    low: float; high: float                      # the range a score can take
    default_cutoff: float | None                 # below this a point reads as not visible; None = none set

class KeypointModel(Model, ABC):                 # core/protocols.py
    needs_examples: ClassVar[bool] = False       # dinov3_match: True — built with an example set
    per_detection: ClassVar[bool]                # dinov3_match: True — one instance per detection; the student: False — whole frame
    needs_mask: ClassVar[bool] = False           # dinov3_match: True — matching is restricted to the detection's mask
    n_candidates: ClassVar[int] = 1              # dinov3_match: 5 — candidates per point the detect stage stores
    reports_confidence: ClassVar[bool] = False   # the student: True — each instance has a confidence min_pose_confidence applies to
    score_spec: ClassVar[ScoreSpec]
    default_track_method: ClassVar[str]          # the KeypointTrackMethod this method's output suits

    def skeleton(self) -> Skeleton: ...          # fixed when the model is built (the example set's, or the model's own)
    def predict_batch(self, frames: list[InstanceFrame]) -> list[FrameResult]:
        """One FrameResult per InstanceFrame, one row per instance found, in order; track_id unset;
        a per-detection method sets each row's detection_row; torch.inference_mode()."""
    # Optional split of predict_batch (*Execution*, below); the defaults run it whole.
    def prepare_batch(self, frames: list[InstanceFrame]) -> object: ...   # CPU half; default: frames
    def run_prepared(self, prepared: object) -> object: ...              # device half, every host copy; default: predict_batch
    def finish_batch(self, raw: object) -> list[FrameResult]: ...        # host half; default: identity
    # A method that splits keeps predict_batch(x) == finish_batch(run_prepared(prepare_batch(x))).

KEYPOINT_MODELS: dict[str, type[KeypointModel]] = {}
def register_keypoint_model(model_id: str): ...
KEYPOINT_TRACK_METHODS: dict[str, type[KeypointTrackMethod]] = {}      # Track methods, below
def register_keypoint_track_method(method_id: str): ...
```

- The model is handed the **whole frame plus every detection's box** (and mask, when it reads
  them) in it. A per-detection method (`dinov3_match`) places one instance per detection; a
  whole-frame method (the YOLO-pose student, later a bottom-up SLEAP export) runs once and returns
  every animal it finds. **Neither assigns a subject**: identity is tracking's, and the one step
  that gives an instance to a subject is the first step of `keypoint_track` (*Assignment*), the
  same for every method. Before this revision the student matched itself to tracks by box IoU ≥ 0.1
  inside the detect stage, and an earlier draft of this bullet specified a box-contains-centroid
  rule that was never built; both are gone.
- **`n_candidates`** is what the detect stage stores per point (*Run directory structure*). A
  closed-set model gives one; the patch-matching method gives its top candidates so the track
  stage can choose among them over time. Method-only values (which example frame won, a mirrored
  match) ride in `FrameResult.extras` and are stored under `extras/`.
- **`score_spec`** says what a score means, because scores are the method's own and never
  comparable across methods: a cosine for `dinov3_match`, a detector confidence for the student. A
  single threshold field cannot serve both, so every consumer that thresholds reads the run's
  method's spec (*Visibility*).
- **An example set is a config input** (`example_set_id`), not a `Prompt` cue: a cue is a
  self-contained prompt payload (§5), and an example set is experiment state the user keeps
  editing. The method is built with the set (`PatchMatch(variant, examples=…)`), `skeleton()` is
  the set's, and `load()` builds the example descriptors from it.
- The install contract, `check_backend` and `resolved` are §5's, inherited. `KeypointModel`
  resolves through the same single lookup as every model (§5 *Model registry*).

### Patch matching: `PatchMatch` and `dinov3_match`

The research method (with its
known-answer checks as tests), ported as one engine in `models/patch_match/` with its decoding
kept private to it, so a second backbone or a second readout changes one piece:

```
PatchFeatures backbone ─► ExampleBank ─► SimilarityMaps ─► point readout (private)
  dinov3_vit               one row per        bank @ grid         peaks → 3×3 soft-argmax →
  (any ViT later)          visible click      max per target      per_point | skeleton decode
```

1. **View.** A square around **each detection's** box — the longer side × 1.3 — rendered to `crop_px`
   (448 by default; a multiple of the backbone's `patch_size`, so 28 × 28 cells at 16 px), black
   outside the frame. Every animal is resized to the same crop, so a patch is a fixed fraction of
   body size whatever the rig.
2. **Example bank**, built once at `load()`: each visible click on an example frame becomes one
   row — the backbone's patch grid sampled bilinearly at the click, renormalised — tagged with its
   target, its example frame and `positive: true`. Rows are never averaged (a mean blurs varied
   appearance; the systems that average need ~16 examples). Negative marks (`positive: false`) and
   region marks have a row type reserved and no readout yet.
3. **Similarity maps**: one `bank @ grid` matrix product per frame, then for each target the
   **max** over its rows — a cell scores as high as its single closest example. Cells outside the
   detection's mask dilated by one patch are set to −inf (`needs_mask`); the mask never changes what
   the backbone sees.
4. **Candidates**: the top local maxima of each map (true peaks, so a runner-up is a competing
   place and not the shoulder of the best peak) and, for pair members, the top cells under greedy
   suppression; each refined to sub-patch position by a soft-argmax over its 3 × 3 neighbourhood
   (`REFINE_TAU`), then mapped back to frame pixels through the view.
5. **Decode.** `per_point` takes each target's best candidate. `skeleton` (the default when the
   skeleton has edges) chooses jointly: the midline as a tree with edge lengths learned from the
   clicks, pairs as two distinct peaks at least a patch apart, and each pair held to its **side
   rule** — `auto` (default) decides per pair from the clicks: `midline` when every example puts
   the pair on opposite sides of the midline polyline, else `hard` when the examples agree on the
   relative order, else none; `hard`, `midline`, `soft`, `off` as explicit choices. Which image
   side "left" falls on is **learned from the clicks** (every example that shows a pair must agree,
   else the pair gets no rule); the local direction a rule needs is derived from the **edges** —
   the midline polyline runs between the two ends of the longest chain through unpaired points,
   and a pair's parent is the midline point both members join — so no head/tail marking is needed:
   any fixed direction works because the learned sign cancels it. The result per pair is recorded
   under `resolved.side_rules`.
6. **Output**, one row per detection: `bbox` (the detection's), `detection_row`, `confidence` NaN
   (`reports_confidence = False`), `points`, `point_scores` (the chosen candidate's cosine),
   `candidates` with their scores (`n_candidates = 5`), and `extras`: `ref_index (K, 5) int16` (which example row won) and
   `mirrored (K,) float32` (the best score with the examples mirrored and their pair labels
   swapped — what the track stage uses to settle left/right over time).

`dinov3_match` is `PatchMatch` on `dinov3_vit` (`vit_large` default, `vit_base`); it declares
`shares_weights_with = "dinov3_vit"` and installs nothing of its own. **The backbone and its
variant are identity**, since the model id no longer says which weights ran. **The matching
constants are tied to the backbone and recorded on the run** (`matching:` in `detect.yaml`):
`refine_tau` 0.01, `skel_tau` 0.05, `topk` 5, `topk_wide` 10, `len_tol` 0.25, `len_floor` 0.02,
`side_penalty` 4.0 — set during the research mostly on ViT-B and never re-tuned for ViT-L; a
cosine threshold that worked for one DINO model needed re-tuning for the next (TokenCut), so they
are provisional and per backbone. Known-answer tests pin the geometry: a point
chosen inside the mask on a frame comes back where it was when that frame is matched against
itself, and moves with the frame when it is shifted.

**Why a crop per detection and not the whole frame.** Every few-shot or feature-matching keypoint
method found (CapeFormer, the graph-based CAPE methods, SCAPE) works on single-instance crops, and
SCAPE names multi-instance scenes as future work; the methods that separate instances on a whole
frame learn to (part affinity fields, X-Pose's instance queries). Patch matching has no such
signal of its own, and a detection already is one: one mask per animal. The crop keeps the scale
the research measured on (a patch is a fixed fraction of body size). A per-detection crop costs a
forward pass per detection rather than per tracked subject — under `anchor_tracker`, up to three
detections where the subject was one.

**`TrainedYoloPose` — the distilled student** (§12 *Distill keypoints*): a closed-set
`KeypointModel` whose skeleton is fixed from the teacher's example set; `n_candidates = 1`;
`score_spec` is the detector's confidence, a confidence of 0 is NaN, never a guess; sees the whole
frame (`per_detection = False`) and returns **every pose whose box confidence is at least
`min_pose_confidence`** (0.25, Ultralytics' own default; `reports_confidence = True`), each with
its box, its confidence and `detection_row = −1` — no matching at detect time; loads local weights only through the
`import_ultralytics` seam; `role: keypoints` on its catalog record keeps it out of every detection
picker. `default_track_method = "despike_smooth"`.

### Job dataclasses and progress

```python
@dataclass
class KeypointDetectJob:                        # core/keypoints.py
    job_id: str; recording_id: str; detect_run_id: str
    detection_run_id: str                        # the detections a per-detection method crops; the cadence and lineage for every method
    min_pose_confidence: float | None            # 0.25 when the method reports_confidence; None (refused if set) otherwise
    model_id: str; model_variant: str | None
    backbone: dict | None                        # {"id": "dinov3_vit", "variant": "vit_large"}; None for a closed-set model
    example_set: dict | None                     # {"id", "digest", "frames": [...], "marks": {...}}; None when not needs_examples
    skeleton: dict                               # the copy the arrays follow
    n_candidates: int
    every_n_frames: int; crop_px: int | None; decode: str | None; side_rule: str | None
    matching: dict | None                        # the constants above, as resolved
    batch_size: int; compute: ComputeTarget; created_at: str
    skeleton_ref: dict | None = None             # {id, name, digest} of the library entry the copy came from
    weights_source: str | None = None            # a student's: "trained" | "imported" (§12)

@dataclass
class KeypointTrackJob:
    job_id: str; recording_id: str; track_run_id: str
    detect_run_id: str; tracking_run_id: str     # the tracking run must be on the detect run's detection run
    match: dict                                  # {min_inside: 0.5, mask_grow: 0.05} (*Assignment*)
    method: str; config: dict                    # the method's config, every field resolved
    skeleton: dict; created_at: str
    skeleton_ref: dict | None = None

@dataclass
class KeypointProgressEvent:                     # both stages; §6's contract
    track_id: int | None                         # None in the detect stage, which has no subjects
    parts_done: int; parts_total: int
    frames_done: int; frames_total: int
    elapsed_s: float; eta_s: float | None
    message: str | None = None                   # "loading model" → "placing keypoints" / "assigning" → "decoding" → "filtering"
```

Phases advance and force an emit at each boundary (§6): the detect stage counts its frames up
front — the detection run's frames for a per-detection method, the frames at the run's cadence for
a whole-frame one (one emit before the count, so a long count is not silence) — then
`placing keypoints` per batch; the track stage emits `assigning` per part, then `decoding` and
`filtering` per (subject, part) — the filter pass #626 ran silently took ~14 s on a 61k-frame part
and read as a hang.

### Run directory structure

Two layers, the shape object detection and tracking already have (§9, §13): the **detect** run is
instance-indexed per part with no subject, as `detections/<run>/part_NNN.zarr` is; the **track**
run writes an assignment column parallel to those rows, as `tracking/<run>/assignments/` is, and
then the per-subject series the consumers read, in embedding's per-(track, part) layout (§14).
The zarr v3 write idiom (§9) exactly — `zarr_store.open_group`, `schema_version` first,
buffer-flush. Column names mirror `FrameResult`.

```
recordings/recording_001/
├── keypoint_detect/
│   └── kd_01/
│       ├── job.json
│       ├── detect.yaml                       ← written when the run STARTS; coverage + timing appended at the end
│       ├── part_001.zarr/                    ← attrs: schema_version, layout: "instances", skeleton_names (K), n_candidates
│       │   ├── frame_idx         (N,) int32            chunks=(10000,)
│       │   ├── t_ns              (N,) int64            chunks=(10000,)
│       │   ├── bbox              (N, 4) float32        chunks=(10000, 4)         normalized xyxy; NaN = none
│       │   ├── confidence        (N,) float32          chunks=(10000,)           the instance's; NaN when the method reports none
│       │   ├── detection_row     (N,) int32            chunks=(10000,)           row index in the detection part; −1 = from no detection
│       │   ├── points            (N, K, 2) float32     chunks=(10000, K, 2)      normalized full-frame; NaN = not placed
│       │   ├── point_scores      (N, K) float32        chunks=(10000, K)         the method's own; NaN with points
│       │   ├── candidates        (N, K, C, 2) float32  chunks=(10000, K, C, 2)   absent when n_candidates = 1
│       │   ├── candidate_scores  (N, K, C) float32     chunks=(10000, K, C)
│       │   └── extras/
│       │       ├── ref_index     (N, K, C) int16       chunks=(10000, K, C)      dinov3_match: the example row that won
│       │       └── mirrored      (N, K) float32        chunks=(10000, K)         dinov3_match: best score, examples mirrored
│       └── part_002.zarr/
└── keypoint_track/
    └── kt_01/
        ├── job.json
        ├── track.yaml
        ├── assignments/
        │   └── part_001.zarr/                ← attrs: schema_version
        │       ├── subject_id        (N,) int32            chunks=(10000,)           parallel to the detect part's rows; −1 = unassigned
        │       └── inside            (N,) float32          chunks=(10000,)           the share that won (*Assignment*); NaN when unassigned
        └── track_0/
            └── part_001.zarr/                ← attrs: schema_version, skeleton_names (K)
                ├── frame_idx         (M,) int32            chunks=(10000,)
                ├── t_ns              (M,) int64            chunks=(10000,)
                ├── instance_row      (M,) int32            chunks=(10000,)           the detect row assigned on this frame; −1 = none
                ├── points            (M, K, 2) float32     chunks=(10000, K, 2)      the final point
                ├── point_scores      (M, K) float32        chunks=(10000, K)         the chosen detection's score
                └── changed           (M, K) int8           chunks=(10000, K)         0 as detected · 1 another candidate chosen · 2 moved by a filter
```

`N` = pose instances in the part: rows written in frame order, **none for a frame with no
instance** (as a detection part writes none for a frame with no detection), any number for a frame
with several. `M` = frames in the part where the subject is tracked, at the run's `every_n_frames`
(§4) — the same rows an embedding run of that track at that cadence has; a subject-frame no
instance was assigned to is a row with `instance_row = −1` and NaN points, so a filter sees the gap
in time. `K` and the column order are the skeleton's, recorded in the yaml **and** as
`skeleton_names` in each zarr's attrs, so rows stay readable after a crash that never reached the
yaml; a reader checks both and fails on a mismatch. `C` = the model's `n_candidates`. Track rows
**match embedding on `(track_id, t_ns)`** (Tenet 8); detect rows are reached from a track row by
`instance_row`, from a detection row by `detection_row`. `frame_idx` counts within its part and is
kept for display. A video part is its **`part_id`** everywhere (`part_001`, the file stem §3 and
the other pipelines use), never a position in a sorted list, which changes meaning when a part is
added. A store without `layout: "instances"` is the earlier per-track layout and is **refused** with
"re-run keypoint detection", never read through a second path.

Sizes: ~1 KB per instance for detect at K = 12, C = 5;
~0.2 KB per frame for track. Raw output is written once and **never rewritten**: changing a filter,
a decode or the assignment is a new track run over the same detect run, CPU only — #626 overwrote
`xy` in place and made every filter change a GPU re-run.

### `detect.yaml` and `track.yaml`

```yaml
# keypoint_detect/kd_01/detect.yaml — written at run start
schema_version: 1
id: "kd_01"
created_at: "2026-10-07T10:00:00Z"
detection_run_id: "det_01"
min_pose_confidence: null                                    # 0.25 for a method that reports_confidence
model_id: "dinov3_match"
model_variant: "vit_large"
weights_source: null                                         # a student's: "trained" (models/) | "imported" (catalog), §12
backbone: {id: "dinov3_vit", variant: "vit_large"}          # null for a closed-set model
example_set:                                                 # null when the method takes none
  id: "ex_01"
  digest: "9f2c…"
  frames: ["frames/a1f3c9….png", "frames/9c2e71….png"]       # the images this run learned from
  marks: {...}                                               # a full copy, so the run never depends on the set
skeleton: {names: [...], edges: [...], pairs: [...]}         # the copy the arrays follow
skeleton_digest: "4b7e…"
skeleton_ref: {id: "sk_03", name: "rat overhead", digest: "4b7e…"}   # the library entry; null for a bare skeleton
n_candidates: 5
every_n_frames: 1
crop_px: 448
decode: "skeleton"
side_rule: "auto"                                            # as asked; what each pair got is under resolved
matching: {refine_tau: 0.01, skel_tau: 0.05, topk: 5, topk_wide: 10, len_tol: 0.25, len_floor: 0.02, side_penalty: 4.0}
resolved: {side_rules: {"left_ear/right_ear": "midline"}}   # Model.resolved(), after load()
batch_size: 32
compute: {backend: "cuda", device_indices: [0]}
# appended when the run ends:
coverage: {part_001: {frames: 61311, frames_with_instance: 61290, instances: 63402}}   # frames walked / with ≥ 1 row / rows
timing: {frames: 61311, seconds: 594.1, frames_per_s: 103.2}
```

```yaml
# keypoint_track/kt_01/track.yaml — written at run start
schema_version: 1
id: "kt_01"
created_at: "2026-10-07T10:30:00Z"
detect_run_id: "kd_01"
tracking_run_id: "track_01"                                  # on the detect run's detection run
match: {min_inside: 0.5, mask_grow: 0.05}                    # *Assignment*
method: "temporal_decode"
config:                                                      # the method's config model, model_dump(): every field, flat
  despike: true
  despike_window_s: 0.1
  despike_k: 3.0
  despike_floor: 0.03
  smooth: true
  smooth_window_s: 0.133
  smooth_kernel: "local_quadratic"
  smooth_truncate: 3.0
  smooth_min_points: 5
  transition_sigma_body: 0.15                                # temporal_decode only, from here down
  missing_penalty: 2.0
  mirror_weight: 1.0
  decode_cutoff: null                                        # the score below which a candidate is "not visible" to the decode; null = the method's default
skeleton: {names: [...], edges: [...], pairs: [...]}
skeleton_digest: "4b7e…"
skeleton_ref: {id: "sk_03", name: "rat overhead", digest: "4b7e…"}
# appended when the run ends:
assignment: {part_001: {instances: 63402, assigned: 61205, unassigned: 2197}}
coverage: {track_0: {part_001: {placed: 61205, total: 61311}}}   # subject-frames with an assigned instance / subject-frames
timing: {frames: 61311, seconds: 14.2}
```

Neither file carries a `status`: **the executor's status is the only status** (`job.json`,
`recording.yaml`), as for every other stage; stop-and-keep lands COMPLETE, and `coverage` says how
much was placed. #626 kept a second status in its yaml, and the two disagreed after a stop.

```yaml
# recording.yaml — run_record() plus each module's extra_record_fields(); job.json holds the full job
keypoint_detect_runs:
  - id: "kd_01"
    status: "complete"
    output: "keypoint_detect/kd_01/"
    created_at: "2026-10-07T10:00:00Z"
    detection_run_id: "det_01"
    min_pose_confidence: null
    model_id: "dinov3_match"
    model_variant: "vit_large"
    weights_source: null
    backbone_id: "dinov3_vit"
    backbone_variant: "vit_large"
    example_set_id: "ex_01"
    example_set_digest: "9f2c…"
    skeleton_digest: "4b7e…"
    n_candidates: 5
    every_n_frames: 1
    crop_px: 448
    decode: "skeleton"
    side_rule: "auto"
    matching: {...}
keypoint_track_runs:
  - id: "kt_01"
    status: "complete"
    output: "keypoint_track/kt_01/"
    created_at: "2026-10-07T10:30:00Z"
    detect_run_id: "kd_01"
    tracking_run_id: "track_01"
    tracking_identity: {...}                                 # the tracking run's spec (§6.5)
    match: {min_inside: 0.5, mask_grow: 0.05}
    method: "temporal_decode"
    config: {...}
    skeleton_digest: "4b7e…"
```

A batch-launched run's entry also carries `processing_run_id`.

### Visibility, status and coverage

- **Visibility is a view, never a filter** (§9's rule for confidence: a threshold applied at run
  time destroys the rows it rejects, and the same run should answer every threshold). Nothing is
  written for it. A consumer reads the run's method's `score_spec.default_cutoff` as its starting
  point — the viewer opens there, the export writes a `visible` column from it and keeps the
  coordinates — and may set its own. A **display** cut-off is not identity. The **decode** cut-off
  the track stage applies when choosing among candidates *is* a track-run setting and identity.
- `points` is NaN only where the method placed nothing (nothing inside the detection's mask, a
  point the student reports absent) or, in a track run, where no instance was assigned — NaN is
  "undefined", as in kinematics (§14), never zero. The track stage never fills an absent point.
- **`min_pose_confidence` is the one threshold applied at run time**, and only to a whole-frame
  method: the student, asked for everything, returns hundreds of near-zero boxes per frame, and a
  row per box would store noise. It is identity and recorded; the per-point visibility above stays
  a view.
- **Coverage.** A detect run records, per part, the frames it walked, the frames with at least one
  instance and the instances; it knows no subjects. A track run records, per part, how many
  instances were assigned and how many were not, and per subject and part the frames placed / the
  frames the subject was tracked. The track run's figure is what consumers read for "how complete
  is this run": §14's alignment refuses to embed frames with no heading and counts them, and the
  viewer draws where frames were lost (§14 *Coverage and alerts*). Per-point coverage (points at or above the
  cut-off) is computed by the reader from `point_scores`; it is not stored.

### Assignment: the first step of every track run

Before any method runs, the track stage gives each detect instance to at most one subject of the
tracking run, per frame, the same way whatever method placed the instances. This is the only place
a pose meets a subject.

1. **Subjects on the frame**: from the tracking run, each subject's rows on that frame
   (`subject_frames`, the grouping every keypoint and sampler path shares, §13's multi-detection
   tracks). Their masks are read as run lengths (`core/rle.py`), never decoded here: a point is
   inside the subject when any row's grown mask holds it, which is the OR without building it.
2. **Score** for each (instance, subject): the **share of the instance's placed points that fall
   inside the subject's mask grown by `mask_grow` × body scale** (`√(w·h)` of the subject's box;
   default 0.05). Points are scaled to the mask's resolution first — a mask is stored at detection
   resolution, which need not be the frame's. An instance with no placed point scores 0 everywhere.
   Growing the mask keeps a nose, ear or tail tip on the mask's edge, which segmentation clips.
3. **Match** one-to-one per frame — each instance to at most one subject, each subject at most one
   instance — maximizing the total score (`scipy.optimize.linear_sum_assignment`), over pairs whose
   score is **at least `min_inside`** (default 0.5); ties go to the higher instance `confidence`.
4. **Write** `assignments/<part>.zarr` (`subject_id`, `inside`) parallel to the detect rows, then
   one series per subject with a row for every subject-frame — the assigned instance's points, or
   NaN with `instance_row = −1` — and hand it to the method.

**A matched pose keeps its points outside the mask**: a tail tip outside the mask is usually the
mask's fault, the match already said whose pose it is, and despike handles a single bad point.
The defaults are **policy, not measured**; both are identity (`match`).

Why the mask and not the box (the student's IoU ≥ 0.1 before this revision): when two animals
touch their boxes overlap heavily and their masks do not. Why here and not in the detect stage:
re-assigning — another tracking run, another threshold — is then a CPU re-run of the track stage
over the same detect run, never a GPU re-run; DeepLabCut and SLEAP split detection from identity
for the same reason. The same function serves the distill's teacher (§12 *Distill keypoints*).

### Track methods

A **track method** turns one subject's assigned placements into final points over time. It owns
its filtering: no filter runs in the detect stage, and no filter applies to a method that did not
declare it — #626 despiked and smoothed every method's output with one setting, and the window was
measured on `dinov3_match` alone. **Each detect method defaults to its own track method in every
surface** (`default_track_method`: `dinov3_match` → `temporal_decode`, the student →
`despike_smooth`), resolved through the one model lookup (§5 *Model registry*) — a surface that
keeps its own table or a fallback id is the bug.

```python
class KeypointTrackMethod(ABC):                 # core/protocols.py
    method_id: str
    config_model: type[BaseModel]               # pydantic (§6.5); every default is identity
    needs_candidates: ClassVar[bool]            # validate refuses it on a detect run with n_candidates = 1

    def track(self, rows: SubjectRows, skeleton: Skeleton, config: BaseModel,
              *, rows_per_second: float, frame_wh: tuple[int, int],
              body_scale_px: np.ndarray) -> TrackPart:   # points (M, K, 2), point_scores (M, K), changed (M, K)
        """One subject's part after *Assignment*: frame_idx, t_ns, instance_row, points, point_scores,
        and candidates / extras when the detect run stored them, NaN where no instance was assigned.
        Pure arrays in, arrays out: the stage reads and writes (unlike Tracker.track, which does its own I/O)."""
```

- **`temporal_decode`** (`needs_candidates = True`; `dinov3_match`'s default). Per point, a Viterbi
  path through the stored candidates plus a *missing* state (the Anipose filter's shape): emission
  is the candidate's score, transition a Gaussian on the pixel distance between consecutive
  choices scaled by body size (`√(box w · h)`) and by the time between rows, so a gap in the track
  costs what its duration earns; the missing state costs `missing_penalty`. Left/right is settled
  over the path: a pair's `mirrored` score in `extras` weighs against its unmirrored one
  (`mirror_weight`), so a swap must hold over frames to win, never flicker. Then the two filters
  below, each switchable. `changed = 1` where another candidate than the detect stage's best was
  chosen. Transition and penalty defaults are **policy**, not measured.
- **`despike_smooth`** (`needs_candidates = False`; the student's default): the filters alone, on
  the best point. **Despike** — a Hampel filter per point across rows: threshold `k` = 3 scaled
  MADs with a floor of 3% of body scale, window `despike_window_s` = 0.1 s each side resolved to
  rows at the recording's cadence (a frame count is a different duration at every frame rate,
  `tests/configs/test_unit_conventions.py`); a jump is replaced by the local median and marked
  `changed = 2`. **Smooth** — each point replaced by a local quadratic fitted to its neighbours,
  Gaussian-weighted **in time** (`t_ns`, so a gap or an every-n run is weighted by real distance),
  standard deviation `smooth_window_s` = 0.133 s, truncated at 3 of them, 5 weighted neighbours
  for the quadratic and the weighted mean below that; an absent point stays absent and absent
  neighbours carry no weight — absence is never filled; a moved point is `changed = 2`. A
  quadratic, not a plain average, because it keeps turning points. Both run per part, in pixels
  (normalized x and y are not isotropic on a non-square frame), vectorized.
- **`point_tracker`** — a learned point tracker seeded by a detect run's points (TAPIR / BootsTAP /
  TAPNext, Apache-2.0; CoTracker3 is non-commercial), in its **online, chunked** mode only (the
  offline mode holds the whole video, Tenet 11). *Deferred*.

Each method's config is a pydantic model (§6.5) and **every one of its fields is identity**, filter
constants included: a run is "the same settings" only if every number that moved a point matches.

### The processing modules

```python
@register_processing_module
class KeypointDetectModule(ProcessingModule):
    module_id = "keypoint_detect"
    id_prefix = "kd"
    runs_key = "keypoint_detect_runs"
    upstream = "detection"
    job_class = KeypointDetectJob
    config_model = KeypointDetectConfig       # configs/keypoint_detect.py
    identity_fields = ("model_id", "model_variant", "backbone_id", "backbone_variant",
                       "skeleton_digest", "example_set_digest", "n_candidates",
                       "every_n_frames", "crop_px", "decode", "side_rule", "matching",
                       "min_pose_confidence")
    lineage_ref = ("detection", "detection_run_id")

@register_processing_module
class KeypointTrackModule(ProcessingModule):
    module_id = "keypoint_track"
    id_prefix = "kt"
    runs_key = "keypoint_track_runs"
    upstream = "keypoint_detect"
    job_class = KeypointTrackJob
    config_model = KeypointTrackConfig        # configs/keypoint_track.py: method + that method's config_model
    identity_fields = ("tracking_identity", "match", "method", "config")   # the tracking run's settings, not its id; every field of the method's config
    lineage_ref = ("keypoint_detect", "detect_run_id")
```

`KeypointDetectConfig`: `detection_run_id`, `model_id` — **no default**: which method fits is a
fact about the animal and the camera, and a default is where a wrong one hides — `model_variant`,
`example_set_id` (required iff `needs_examples`; refused otherwise), `min_pose_confidence` (0.25
for a method that `reports_confidence`; refused otherwise), `decode`, `side_rule`, `crop_px`,
`every_n_frames`, `batch_size`, compute. `validate()` resolves every defaulted input and writes it
back (Tenet 9): the variant, the backbone, `decode` (`skeleton` when the skeleton has edges),
`side_rule` (`off` when it has no pairs; an explicit `hard` without pairs refused), the matching
constants, `min_pose_confidence`, the set's digest, frames and marks, the skeleton copy and digest.
`KeypointTrackConfig`: `detect_run_id`, `tracking_run_id` (default: **the one** complete tracking
run on the detect run's detection run; several → refused, naming them — resolving to "the first"
would make the same command assign differently tomorrow), `match_min_inside` (0.5),
`match_mask_grow` (0.05 of body scale), `method` (default: the detect run's method's
`default_track_method`), and the method's own fields.

**Detect runner** (`pipeline/keypoint_detect.py`). Per part: **one** forward-sequential
`cv2.VideoCapture` walk (Tenet 11). A per-detection method visits the frames that have detections
at the run's cadence and is given every detection on the frame — its box, and its mask decoded
only when the method reads masks; a whole-frame method visits every frame at the cadence. Frames
batched in a bounded buffer of `batch_size`, `predict_batch`, every returned row appended in frame
order, the writer flushing on frame boundaries (Tenet 12). Frames come from the detection part's
rows or the timestamp map, never `CAP_PROP_FRAME_COUNT`. Progress per §6 (*Job dataclasses and
progress*).

**Execution: decode thread → prep → GPU → writer.** The same four stages as detection (§7
*Execution*), through `core/lookahead.py`, so the GPU does not wait on the CPU:

```
decode thread               prep thread              calling thread          writer thread
walk, BGR→RGB, mask decode  model.prepare_batch  ─►  model.run_prepared  ─►  model.finish_batch,
(2 batches ahead)           (2 batches ahead)        forward only            instances appended, progress
```

- **The method's halves** are the optional split in the `KeypointModel` protocol above. The
  student opts in through `models/_ultralytics.py::SplitPredict` (letterbox on the prep thread;
  forward, NMS and the host copies on the calling thread). `dinov3_match` opts in through
  `PatchMatch` (each detection's view, crop and allowed cells on the prep thread; the backbone and
  both banks' `bank @ grid` on the calling thread, copied to the host once per bank per batch;
  candidates, the decode and the rows on the writer thread, from numpy only). A method that does
  not split runs `predict_batch` whole on the calling thread, with only decode and the writes
  overlapped. Every copy off the device happens in `run_prepared`: on another thread it would
  queue behind the next batch's forward.
- **The calling thread** checks the stop event before each forward. On a stop the batches already
  computed are written, so the store still ends on a frame boundary.
- **One writer thread** keeps instances in frame order and opens the part's store on the first
  instance; the store is closed on the calling thread after the writer has joined. A write error
  fails the run after the batches before it.
- **Errors keep their position**: a decode error is raised to the caller after every batch before
  it, and no stage thread outlives the part.
- **Rows are byte-identical to the serial loop**: the same calls on the same batches, in order.
  `tests/pipeline/test_keypoint_detect.py` holds the stage to rows built from the fixture's
  detections, `tests/models/test_ultralytics.py` holds the student's split to Ultralytics' own
  `predict`, and `tests/models/test_patch_match.py` holds `PatchMatch`'s to each detection scored
  alone through the bank's `score_maps`.

**Track runner** (`pipeline/keypoint_track.py`). Per part: read the detect part and the tracking
run's subjects, run *Assignment* and write `assignments/`; then per (subject, part) build the
subject's series, call the method, write the track part; CPU only. **The
tracking run's assignment zarr is joined to the detection rows by row id, never by position**: with
a class-filtered tracking run the assignment covers only the tracked class's rows, and a
positional pairing put every track on the wrong frame after the first dropped row (found in user testing,
#657).

**Launching.** Each stage is launched like tracking is: **Run keypoint track →** on a finished
detect run's page seeds the module and the recordings, or *New run* → `keypoint_track` with
*Start from* a previous run's settings. There is **no combined detect-then-track form** — normal
detection has none, and chaining is to be added for every stage at once, later (*Deferred*).
Re-tracking a detect run is the same launch. **The GUI's next-step buttons follow one line**, detection → tracking → keypoint detect → keypoint track (`ProcessingModule.offered_after`, 2026-10-21, the user's choice): keypoint detect reads only a detection run but is offered from a finished *tracking* run, and keypoint track only from a keypoint detect run — from a tracking run it would be a button with nothing to run. The API and CLI still allow keypoint detect before tracking.

### Identity and comparability

| stage | identity | recorded, not identity |
|---|---|---|
| `keypoint_detect` | the detection run's identity (by lineage), method, variant, `weights_source`, backbone + variant, skeleton digest, example-set digest, `n_candidates`, `every_n_frames`, `crop_px`, `decode`, `side_rule`, the matching constants, `min_pose_confidence` | compute, batch size, timing, coverage |
| `keypoint_track` | the detect run's identity (by lineage), the tracking run's identity (`tracking_identity`; the id is recorded, not identity), `match` (`min_inside`, `mask_grow`), `method`, every field of its config (filters, transition, decode cut-off) | timing, coverage, assignment counts |

What #626 left out and this adds: the skeleton (a run with a renamed or reordered skeleton merged
with one that was not), the backbone, and the filter constants (a changed default false-merged
old and new runs). The §6.5 identity table carries both rows.

### Analysis and export

- **One reader.** `Recording.keypoints(track_run_id, track_id) -> dict[str, dict]`, one entry per
  video part: `{part_id: {frame_idx, t_ns, points, point_scores, changed}}`; the skeleton is the
  run's (`Recording.keypoint_track_run(run_id).skeleton`). `Recording.keypoint_instances(
  detect_run_id, part_id)` for the raw stage — every instance with its box, confidence, detection
  row and candidates, no subject. Every consumer below goes through them.
- **A joined view for hybrid modules — not built.** The design is `frame_table(embedding_run,
  track_run)` keyed on `(track_id, part_id, t_ns)` with the vector, the points and their scores
  and the kinematics row, so a module that reads both never writes its own join; a
  `keypoint_track` run an input an analysis module may `require` (§16), resolved per recording
  like an embedding spec, the analysis run recording its identity and being its dependent.
  `analysis/descriptors.py` already takes extra `FrameSlice` parts, so keypoint parts can plug in
  there with no schema change. **No analysis feature reads keypoints yet**; this is the room for
  them.
- **Export targets** on a `keypoint_track` run (the §17 registry, `export_targets.py`), reachable
  from the API, the CLI and the GUI's one Export control (§18):

| target | read by | layout |
|---|---|---|
| `keypoints` — DETAIL long CSV | pandas, R | one row per frame × track × point: `recording_id, track_id, part_id, frame_idx, t_ns, keypoint, x_px, y_px, x_norm, y_norm, score, visible, changed` |
| `keypoints_dlc` — DeepLabCut CSV + H5 | keypoint-MoSeq, VAME, B-SOiD, A-SOiD, SimBA, movement | one file per video part; columns `scorer / individuals / bodyparts / coords (x, y, likelihood)`; rows by frame index |
| `keypoints_sleap` — SLEAP analysis HDF5 | movement, keypoint-MoSeq, SimBA | `tracks (n_tracks, 2, n_nodes, n_frames)`, `node_names`, `track_names`, `point_scores`, `track_occupancy`, `edge_inds` |
| `keypoints_nwb` — NWB + ndx-pose | DANDI, the NWB tools | `PoseEstimation` with one `PoseEstimationSeries` per point, timestamps, the skeleton; optional extra (`pynwb`, `ndx-pose`) |
| `keypoints_video` — points drawn on the footage | people | every frame, the points at the display cut-off; the tracking overlay opt-in (#689) |

  One file per video part, named by **recording name** (not id); DLC and SLEAP files carry no
  time, so a `timestamps.csv` sidecar (`frame_idx → t_ns`) goes with them; `likelihood` is the
  method's own score, not a probability, and `export.yaml` says so; a missing point is NaN in
  every format, and an option blanks points below the cut-off; the skeleton's edges and pairs are
  written in each format's own form; every export ships `params.csv` (from `job_params`) and the
  run's coverage. The default source is the track run; the detect run is an option and exports its
  instances **unassigned** (an empty `track_id`, an `instance` index per frame — DLC and SLEAP
  call these untracked individuals); candidates go only in DETAIL's own bundle. **The layouts are
  checked against each tool's own loader** (`tests/api/test_keypoint_export.py`): the DLC and
  SLEAP files load through `movement`'s `load_poses`, the SLEAP file's datasets are read with
  `h5py`, and the NWB file round-trips through `pynwb`. This also fixes #626's
  `export run --run kp_NN`, which looked for a `runs/<id>/run.yaml` a single-recording run never
  has.

### Alignment (§14)

How an embedding run reads a `keypoint_track` run — the heading definition, interpolation across
gaps, `rotate` and `mirror`, the turned-mask canvas, coverage, identity — is §14 *Egocentric
alignment*. This section owns only what it reads: the final points, their scores and the method's
`score_spec`.

### Distill (§12)

Training a fast closed-set student from the patch method or a finished `keypoint_detect` run, plus
the example set's clicks, is §12 *Distill keypoints*; the teacher's instances are given to the
tracking run's subjects by *Assignment*'s function. The student itself (`TrainedYoloPose`) is a
`KeypointModel` described above.

### Auditing a keypoint track run

Not built. What is in the tree is the storage: an example set with `purpose: audit` (*Example
sets*) — with per frame `round` and `skipped`, and `sources: {recording_id: tracking_run_id}` — is
still read, and the distill's rule that audit frames never build a student stays (§12).


### Validation checks

**Pre-run (`keypoint_detect`):**

| Check | Action on failure |
|---|---|
| Detection run `status: "complete"` | Error |
| `model_id` in `KEYPOINT_MODELS` or a catalog record with `role: keypoints` | Error — list available methods |
| `min_pose_confidence` in [0, 1], and set only for a method that `reports_confidence` | Error — name the method |
| Model installed (`readiness()`, §5); `compute.backend` in `supported_backends`; `check_backend` passes | `ModelNotReadyError` with the install action / Error |
| Example set given iff the model `needs_examples`; it exists and `purpose: example` | Error |
| Every skeleton point clicked visible on at least one example frame | Error — name the keypoints with none |
| `Skeleton.validate` passes | Error with the offending point |
| `decode = "skeleton"` requires edges; an explicit `hard` / `midline` side rule requires a pair | Error — name the setting to change |
| `crop_px` a multiple of the backbone's `patch_size` | Error — name the nearest values |
| `every_n_frames` not finer than the detection run's, and a multiple of it (§4) | Error |

**Per-part / post-part:** the writer flushes on frame boundaries (Tenet 12); K matches the
skeleton; `candidates` present iff `n_candidates > 1`.

**Pre-run (`keypoint_track`):**

| Check | Action on failure |
|---|---|
| Detect run `status: "complete"` or stopped with coverage; its skeleton readable; its parts in the `instances` layout | Error — a per-track run says "re-run keypoint detection" |
| Tracking run `status: "complete"` and on the detect run's detection run; with none given, exactly one such run | Error — name the runs to choose between |
| `match_min_inside` in (0, 1]; `match_mask_grow` ≥ 0 | Error with the field |
| `method` registered; `needs_candidates` satisfied by the detect run's `n_candidates` | Error — name a method that fits |
| The method's config validates (pydantic) | Error with the field |
| The recording has `fps_actual` when a filter window is in seconds | Error |

### Resume

Neither stage resumes (Tenet 12), the same as detection. A failed or cancelled keypoint run is
**re-run**: a retry (`Experiment.retry_run`) gives each incomplete recording a fresh run id, so
nothing appends into a half-written store (§7). Both modules declare `resumes = False`, so a cancel
is recorded `cancelled`, never `partial` (`tests/executor/test_cancel_status.py`). A stop-and-keep
still lands `complete` with the frames placed so far and their `coverage`. `keypoint_detect`'s
runner always starts on an empty run directory: it has no skip past frames already written.
`keypoint_track` re-runs whole, in seconds.


### Surfaces

Every capability on all three surfaces in the change that adds it (`wire-every-surface`):

- **API.** `Recording.run_keypoint_detect(detection=…, ...)` / `run_keypoint_track(detect=…,
  tracking=…, method=…)` and `Experiment.run_processing` for both modules;
  `Recording.keypoints(kt, track)`, `keypoint_instances(kd, part)` (was `keypoint_candidates(kd,
  track)` before the 2026-10-21 revision); example sets — `Experiment.example_sets()`,
  `create_example_set(name, skeleton)`, `sample_example_frames(...)`, `add_example_frame(...)`,
  `set_example_marks(...)`, `remove_example_frame`, `import_example_set(yaml)`,
  `delete_example_set`, `update_example_set`, `duplicate_example_set`, `export_example_set`,
  `example_set_runs`; the skeleton library in `api/skeletons.py` — `skeletons()` (presets and
  library entries), `skeleton`, `create_skeleton`, `update_skeleton`, `save_as_new`,
  `duplicate_skeleton`, `delete_skeleton`, `dependents`, `export_skeleton`, `import_skeleton`;
  `Experiment.test_keypoints(...)` (a few frames, no run, nothing written); §14's and §12's
  additions.
- **CLI.** `keypoints detect` / `keypoints track` and `process keypoint_detect` /
  `process keypoint_track` (options from the config models);
  `keypoints examples list|create|edit|duplicate|sample|add-frame|show|import|export|delete`;
  `skeleton list|show|create|edit|save-as-new|duplicate|export|import|delete`
  (`detail/cli_skeletons.py`, the library's own group); `keypoints test`; `keypoints export --target keypoints|keypoints_dlc|keypoints_sleap|keypoints_nwb|keypoints_video`;
  `keypoints distill`, `dataset keypoints` (§12). Clicking marks is GUI-only, recorded in `docs/cli.md`'s "Not in the CLI" with the YAML routes as
  the alternatives. The `keypoints` group lives in `detail/cli_keypoints.py`, registered on the
  root group — the one module `cli.py` does not hold, so eight PRs do not edit one region.
- **GUI** (every page copies the closest existing page and follows `gui/requirements.py`): the
  two launch forms (`ConfigForm` over the config models; *Start from* first; the tracking run via
  `UpstreamSpecPicker`; the example set and tracking run as *selection* fields, never restored);
  **Run keypoint track →** on a finished detect run; the run viewers as subclasses of the tracking
  viewer with per-track **loss** rows (tracking, keypoints, heading) and a per-point coverage
  table; the **Keypoints manager** (the skeleton library and its editor dialog, *The Keypoints
  manager*, above) and the **example-set picker** wherever a model that `needs_examples` is chosen
  (*Example sets follow the model that needs them*);
  the **labeling screen** (the *Mark examples* dialog's shape: recording and track on top, player 3
  : frame list 2 in a remembered splitter, an *Annotating: <point>* box under the transport with
  *Not visible (N)* and *Clear*, example frames as slider ticks, removing a saved frame confirms,
  OK reads *Use these frames*, a tool row with *Point* alone enabled); **Test keypoints…** on the
  detect form (detection's *Test prompts…* pattern); the heading picker (§14) as one shared widget;
  *Distill keypoints…* on the Models page (§12); the Model manager saying what each model is for
  (*embedding · patch features · keypoints*); keypoint exports through the one Export control. All
  slow loads off the UI thread through `gui/busy.py`; names shown, ids as item data; sizes in
  `lines()`; a greyed control says why.
- **Docs.** `docs/concepts.md` (what keypoints are, the two stages, example sets, alignment),
  `docs/cli.md`, `docs/gui.md`, the generated parameter reference — each written in the change
  that added its capability.

### Guards

- `tests/test_design_plan_status.py` has no §26 probe, so the Keypoints Status row's verdict is
  unchecked. Its parser reads every two-column table row in the file whose first cell names a
  section, so no two-column table in this section may name a probed section in its first cell.
- The install contract: `tests/models/test_no_implicit_network.py` scans every model module, the
  new ones included; `PatchFeatures.patch_grids` after `load()` only.
- Schema-edge tests (`schema-edge-tests`) for both stages: a part with no subject frames, a track
  absent from a part, `n_candidates = 1` (no `candidates` column), a stopped run's coverage, a
  skeleton whose K differs from the zarr's attrs, a class-filtered tracking run.
- Label leakage: not covered. `tests/pipeline/test_label_leakage.py` does not drive the keypoint
  stages, and the audit whose labels it would permute is not built.
- Export round-trips: each format opened by its own loader.
- UI-thread scan seams (`tests/gui/test_ui_thread_blocking.py`): `keypoint_instances(`,
  `sample_example_frames(`, `add_example_frame(`, `set_example_marks(`, `test_keypoints(`.
  `keypoints(` and the canvas pre-pass are not scanned.
- Identity: `tests/configs/test_identity_completeness.py` covers both modules; a changed filter
  default produces a new identity (`test_record_completeness.py`).
- The `_downstream_edges` guard: every registered module with a `lineage_ref` or
  `aux_refs` appears in the dependents walk.

### Deferred

Named so a later change adds a slot rather than reshaping: **SuperAnimal** (Quadruped,
TopViewMouse; native HRNet-W32, non-commercial license) and **imported SLEAP / DeepLabCut ONNX
exports**, with the catalog `role` field, the `keypoints_onnx` extra and the import screens they
need — both as closed-set `KeypointModel`s; a **parent part** per point; **mapping a closed-set
model's point names** onto an example set's; the **region readout** (patch matching for body-part
masks; similarity peaks as SAM 3 prompts) and **patch-matched objects as a `DetectionModel`**
(pellets, levers; static landmarks located once per recording); **point trackers** as a track
method; a **chained detect-then-track launch**, added for every stage at once; the opt-in **change
gate** (#680, run the method only when the subject's shape changes — off by default and
unmeasured); **pose into the embedding** (#659, points drawn on the mask or appended as numbers — a
hybrid module, for which *Analysis and export* keeps the room); per-viewer **export buttons**
(#693 — exports go through the one Export control); the **camera-tilt alignment rule** and the
**alignment study** (§14 *Later*); **Quick Pose** (an earlier prototype) is superseded by this section's
one protocol and storage.

---

## 27. Tools

A **tool** is a one-off run that needs no experiment: point it at a file, run a model, look at
the result. It serves the moment before there is a project (judging a model on real footage
before committing to an experiment) and a result that can be shown to someone. The user page is
`docs/tools.md`.

### The protocol

`Tool` in `core/protocols.py`:

```python
class Tool(ABC):
    tool_id: str        # registry key + GUI/CLI id
    label: str          # display name
    description: str    # one line, shown in the tools list
    category: str       # grouping in the tools list
    config_model: type  # pydantic model for the scalar parameters (§6.5's configs rule)

    def run(self, config: dict, out_dir: Path,
            progress: Callable[[str, float, str], None] | None = None,
            stop_check: Callable[[], bool] | None = None) -> ToolResult: ...
```

- **Synchronous, in the caller's process.** No executor, job, worker subprocess or
  `ProgressEvent`. `progress(stage, fraction, detail)` is a plain callback: the CLI draws it as a
  bar, and the GUI runs the tool off the UI thread in `gui/workers.py::StageProgressWorker`. The
  long-op rule still holds: a tool drives `progress` inside any loop that can run longer than a
  moment.
- **Stopping is not failing.** A tool whose work is unbounded polls `stop_check()` in that same
  loop. When it returns true the tool finishes early and returns a `ToolResult` with
  `stopped=True` describing what it did complete, rather than raising.
- **It writes only under `out_dir`**, a run directory the framework owns, never into an
  experiment.
- **Non-scalar inputs** (video paths, `Prompt`s, a `ComputeTarget`) are assembled by each surface
  and passed in `config` beside the config model's fields.
- **Models are resolved machine-globally**: `models/resolve.py::resolve_pretrained_or_imported`
  (a built-in, or an imported or distilled catalog model). With no experiment, an experiment's
  trained models are out of reach. `api/models.py::tool_models` lists what a tool can run.

`ToolResult` (`core/dataclasses.py`) is JSON-native: `n_processed`, `frames_total`, `fps`,
`resolution` (width, height), `class_counts`, `outputs` (logical name → path relative to the run
directory), `summary`, `stopped`. Heavy output is written to `out_dir` and named in `outputs`.

### Registry

`TOOLS` in `core/registries.py`, keyed by `tool_id`, filled by `@register_tool(tool_id)`
(Tenet 1). `tools/__init__.py::TOOL_MODULES` lists the tool modules and `import_tools()` imports
them so the registry is populated before any lookup.

The GUI half is a parallel registry, `gui/tools/registry.py::TOOL_GUIS`, filled by
`@register_tool_ui` from `TOOL_UI_MODULES`. A `ToolUI` supplies the config form
(`config_widget()`) and the view of a finished run (`result_widget(manifest)`). It lives under
`gui/` so the core protocol imports no Qt (Tenet 3).

### Where runs are stored

Tool runs are machine-global, in a `tools/` tree beside the experiments in the managed workspace
(`storage/locations.py::experiments_root()`). `storage/tool_runs.py` owns it.

```
<experiments root>/tools/
  <tool_id>/
    run_NNN/
      tool.yaml     # the run record
      ...           # what the tool wrote, named in result.outputs
```

Run ids are `run_NNN`, unique across the whole tools workspace rather than per tool, so one id
names a run without its tool (`detail tools delete run_003`). Runs are found by scanning for
`tool.yaml`, and deleting one removes its directory.

### The run record

`api/tools.py::run_tool` writes `tool.yaml` with `status: running` before the work starts, so a
crash leaves a visible record, then rewrites it when the tool returns or raises. The GUI reads a
manifest while the run rewrites it, so every write goes to a sibling `.tmp` and is moved into place
with `core.atomic.replace`. `schema_version: 1` is the first key.

```yaml
schema_version: 1
id: run_003
tool_id: quick_segment
label: Quick segment
source_video: /path/to/clip.mp4     # the first of source_videos
source_videos: [/path/to/clip.mp4]
model_id: sam3
model_variant: null
prompt: {...}                       # the first of prompts (Prompt.to_dict())
prompts: [{...}]
compute: {...}                      # ComputeTarget.to_dict()
config: {mode: sample, n_frames: 50, whole_video: false}   # exactly the config_model's fields
status: complete                    # running | complete | stopped | failed
created_at: '2026-11-15T10:00:00Z'
result: {...}                       # ToolResult.to_dict(); written when the run returns
error: '...'                        # only when status is failed (no result)
```

- **The source video is recorded by absolute path and read in place, never copied.** A tool run
  is machine-local and outside any experiment, so Tenet 6 (which governs experiment files) does
  not reach it.
- `prompt` and `source_video` are singular aliases of the first entry of the plural keys, kept so
  records written before a run could take several clips or prompts still read.

### Quick segment, the one registered tool

`tools/quick_segment.py` (`tool_id: quick_segment`, category "Detection & segmentation"), with
config model `configs/quick_segment.py::QuickSegmentConfig`: `mode` (`sample` | `consecutive`),
`n_frames`, `whole_video` (consecutive only). It runs a detection model on raw clips with no
import.

- **sample** (the default): `n_frames` spread evenly over each clip, every prompt run on each,
  compared side by side. It is §8's prompt-test engine pointed at raw files. Writes one detection
  store per prompt and clip (`<prompt label>/part_NNN.zarr`), each clip's timestamp map
  (`timestamps/part_NNN.npy`) and a `test.json` index.
- **consecutive**: one clip, one prompt, read straight through from frame 0, writing
  `detections.zarr` (§9's detection schema) and `timestamps.npy`. With `whole_video` it is uncapped,
  and so stoppable. It is the only mode an overlay mp4 can be exported from.

Neither mode writes images or copies the video: overlays are drawn from the source frames and the
saved detections when a view asks, and pictures reach disk only through an explicit export. Both
keep every proposal the model returns (`KEEP_ALL_PROPOSALS`); the threshold a result is shown at
is a view setting (`ConfidenceView.min_conf`, §9 *Confidence threshold*), which is why
`QuickSegmentConfig` has no confidence field. Quick segment picks SAM 3 by default only on a
machine with CUDA (`api/tools.py::default_model_for_machine`) and raises otherwise, unless the
user chooses one of their own imported or distilled models.

### Surfaces

- **API**: `api/tools.py`, a module of functions because there is no experiment: `list_tools()`,
  `run_tool(tool_id, config, *, progress, stop_check)`, `list_tool_runs`, `read_tool_run`,
  `delete_tool_run`, `default_model_for_machine`.
- **GUI**: the shell's **Quick tools** mode (`gui/app.py`): a rail listing each tool with its
  saved runs; `ToolsPage` (the landing, grouped by category), `ToolRunPage` (the tool's form and
  Run, off-thread with a progress bar and Stop) and `ToolRunDetailPage` (the result view and
  Delete), all in `gui/pages.py` and generic over `TOOL_GUIS`.
- **CLI**: the `detail tools` group in `cli.py`: `quick-segment`, `list`, `delete`.
  **Each tool's CLI command is hand-written.** There is no generic `detail tools run <tool_id>`:
  the non-scalar inputs (videos, prompts, model, compute) have no generic flag form, so a new tool
  adds its own click command. `list` and `delete` are generic over every tool.

### Adding a tool

A `Tool` subclass in `detail/tools/` with `@register_tool`, listed in `TOOL_MODULES`; its config
model in `detail/configs/`; a `ToolUI` in `detail/gui/tools/` with `@register_tool_ui`, listed in
`TOOL_UI_MODULES`; its own `detail tools <name>` command; and a section in `docs/tools.md`. The
landing page, the rail, `detail tools list` / `delete` and the config-model reference
(`configs/describe.py::iter_config_models`) pick it up with no other change.
