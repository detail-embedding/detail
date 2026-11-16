# Backend design overview

> **Design/architecture doc — not a user guide.** For how to *use* the tool, see
> [`docs/concepts.md`](concepts.md), [`docs/cli.md`](cli.md), and
> [`docs/gui.md`](gui.md).

This doc outlines the backend functionality and organization of DETAIL. For full design details and specs, see [`DESIGN_PLAN.md`](../DESIGN_PLAN.md).

## Design methodology

The backend functionality is designed to be highly flexible, extensible, and future-proof. All functionality is callable directly through the python API and fully decoupled from the front end GUI.

The tool is designed to support local installs as well as usage of cloud storage and compute for future version releases.

All operations and stages (time series data, detection, model fine-tuning, tracking) are abstracted to allow for the addition of new data types, models and methods with minimal code changes.

Each stage runs as a self-contained, serializable job, which enables crash recovery and exact reproduction of any run.

## Data and organization methodology

- Unified time units in integer nanoseconds
- Data organization fully handled by tool. Video is either copied into the experiment or referenced in place under a machine-local **video root** alias; an experiment file never holds a machine path.
- Each stage writes its own output. Source data and earlier outputs are never modified.
- Every output is self-contained with sufficient meta data to fully describe itself

## Core abstractions

Every model and method sits behind a small interface (an abstract base class) and is registered under a string id. Callers reference the id (`model_id="sam3"`) and never import a concrete class. Adding a backend means implementing the interface and registering it; excluding one means not installing it.

The protocols, each with its own registry. The full list is `core/protocols.py` and `core/registries.py`; the pipeline and analysis modules register in `executor/runners.py` (`PROCESSING_MODULES`, `ANALYSIS_MODULES`).

| Protocol | Defines | Registry | Lives in |
|---|---|---|---|
| `DetectionModel` | how objects are found in a frame | `MODELS` (and `TRAINED_LOADERS` for trained weights) | `models/` |
| `Tracker` | how detections are linked across frames | `TRACKERS` | `trackers/` |
| `EmbeddingModel` | how a rendered image becomes a vector | `EMBEDDING_MODELS` | `models/` |
| `KeypointModel` | where an animal's keypoints are in a frame | `KEYPOINT_MODELS` | `models/` |
| `KeypointTrackMethod` | how keypoints are smoothed and linked over time | `KEYPOINT_TRACK_METHODS` | `pipeline/keypoint_track.py` |
| `Trainer` | how a custom model is trained | `TRAINERS` | `trainers/` |
| `MaskTransform` | one step in the render pipeline | `TRANSFORMS` | `transforms/` |
| `StreamImporter` | converts an external signal format to the canonical CSV, one per device or file format | `STREAM_IMPORTERS` | `importers/` |
| `QueryMethod` | how a behavior scan scores frames against example clips | `QUERY_METHODS` | `query_methods/` |
| `TemporalClusterMethod` | how bouts are clustered into behaviors | `TEMPORAL_CLUSTER_METHODS` | `cluster_methods/` |
| `Tool` | a one-off operation that needs no experiment (`DESIGN_PLAN.md` §27 (Tools)) | `TOOLS` | `tools/` |
| `Check` | a readiness or data-integrity check | `CHECKS` | `checks/` |

Storage is local only. The `StorageLocation` abstraction (a video URI and a data store per backend: local, S3, GCS, GDrive) is designed but not built; `storage/locations.py` resolves the `default` alias to a local path, and `storage/roots.py` maps the video-root and export-root aliases.

Key data types passed across these interfaces:

- `Prompt` / `Cue`: what to detect. A cue is text, points, a box, or an example image.
- `Detection` / `FrameResult`: a model's standardized output, identical across architectures.
- `ComputeTarget`: the requested hardware.
- `*Job` (one per stage and per analysis module): the serializable record that defines and runs it.
- `RenderContext`: the frame, mask, and bbox a transform reads and updates.

## Experiments and recordings

Two containers hold everything:

- **Experiment**: a whole study. Holds all recordings, outputs, trained models, and analyses. The unit of sharing and archival.
- **Recording**: one logical recording (one animal, one bout, one camera setup). May span several video files; each is a **part**. Parts are ordered and continuous. A new animal, condition, or arena is a new recording.

Directory layout (trimmed):

```
my_experiment/
├── config.yaml              # experiment identity only
├── metadata.csv             # one row per recording (see "grouping variables")
├── recordings/
│   └── recording_001/
│       ├── recording.yaml     # the manifest for this recording
│       ├── video/           # original video parts + frame-time maps
│       ├── streams/         # other recorded signals, aligned to video time
│       ├── arenas/  rois/   # named polygons
│       ├── detections/      # one folder per detection run
│       ├── tracking/        # one folder per tracking run
│       └── embeddings/      # one folder per embedding run
├── analysis/                # analysis runs (can span recordings)
├── models/                  # trained / user-provided model weights
└── datasets/                # labeled datasets for training
```

- `config.yaml`: experiment identity only (name, creation time). No paths, parameters, or credentials.
- `recording.yaml`: the per-recording manifest. Lists video parts, streams, arenas, and every run. Files are read in the order it declares, never filesystem order.
- Storage is referenced by alias (`"default"`, `"lab_drive"`, ...), not by absolute path. A machine-local config maps each alias to a real path (cloud buckets are designed, not built), so experiments stay portable across machines.
- Credentials (e.g. a Hugging Face token) live outside the experiment.

## Processing pipeline

Each stage reads an earlier stage's output and writes its own: **ingest → detect → track → embed**, and alongside them, **keypoint detect → keypoint track** (keypoint detect places points on a detection run's animals; keypoint track gives those points to a tracking run's subjects and settles them over time).

### Ingest video and streams

- Video is added to a recording one part at a time.
- At import the tool builds a timestamp map (frame index → `t_ns`) from the container's presentation timestamps, falling back to frame rate. The map has one row per frame a player actually shows — frames an MP4 edit list trims are left out — and starts at 0 whatever time the file's own clock starts at. Each later part starts where the previous one ends.
- `recording.yaml` has a slot for a second, more accurate map (e.g. from a hardware sync signal) and records which one is active, but nothing writes one yet: the generated map is always active.
- **Streams** are other signals recorded alongside the video (heart rate, accelerometer, key events). Each is converted to a common CSV and aligned to video time, so it shares the `t_ns` clock. Any stream and any frame then join on `t_ns`.
- **Use every N frames** is how a run costs less: detection, tracking, embedding and keypoint detect each take one, processing every Nth *source* frame. No copy is made and no frame is renumbered, so a run's rows say which frames it looked at.

### Arenas and ROIs

- **Arena**: a polygon bounding where detections count. A processing-time filter; detections outside it are dropped.
- **ROI** (region of interest): a polygon used during analysis to tag or query detections ("which poses occur near the lever?").
- Independent of each other: arenas shape what is processed, ROIs shape how results are queried.

### Detect

- A detection model finds the animal(s) or object(s) in each frame.
- The target is set by a **prompt**: a list of targets, one per class, each with a cue. A cue is a word (`"mouse"`), points or boxes drawn on a frame, or an example image. Same structure for single- or multi-class.
- A prompt can be tested on a few sample frames before the full run.
- Each part is read start to finish (always sequential; frames are never loaded all at once or seeked mid-run). Output is one array store per part.
- Stores are detection-indexed: one row per detection, not per frame. Three animals in a frame is three rows; none is no rows. Each row holds the box, confidence, class name, and mask.
- Masks are stored as run-length encoding (RLE), shrinking a mask from megabytes to a few KB.
- Compute target (CPU or GPU) is set per run, not globally (see *Compute and GPU support*).

### Track

- Tracking links the same animal across frames by assigning a consistent `object_id`. The detections sharing an id form a **track**.
- Filtering is built in: low-confidence and out-of-arena detections are never linked. Downstream stages receive identified subjects, not a raw list of boxes.
- One animal is one track; two animals are two tracks.
- Detection stores are never modified. Tracking writes:
  - a parallel assignment store (same length, row for row) holding `object_id`
  - a track summary: time spans, frame counts, mean confidence
- Downstream reads the box and mask from the detection store, and `object_id` from the assignment store.
- Track ids start as integers; real identities (track 0 → `"M001"`) can be attached later with no structural change.

### Embed

Each tracked detection becomes a feature vector, an **embedding** (a fixed-length numeric summary of an image). Two steps:

1. **Render** the detection into an image through a pipeline of **transforms**, each doing one operation (init from binary mask, crop to box, apply mask, pad to square, center in square, resize). The pipeline is a serializable list, recorded with the run; mask-only pipelines skip video decode and a final `resize` targets the embedding model's input size (§14):

   ```python
   renderer_spec = [
       {"type": "init_binary_mask"},
       {"type": "pad_to_square"},
       {"type": "resize", "size": 224},
   ]
   ```

2. **Embed** the rendered image through an embedding model, which returns the vector.

- The model sees the rendered image, not the raw frame. The render step is where most mask-representation research happens; swapping the transform list changes the representation with no other code changes.
- Output is per track and per part: one store per part holding the vectors, each tagged with frame, `t_ns`, and a back-reference to the source detection row.

## Compute and GPU support

- Each run chooses its own compute target: CPU, CUDA (Nvidia GPU) or MPS (Apple Silicon). Set per run, not globally, so different runs can use different hardware. `ComputeTarget` has a `cloud` backend, which nothing runs yet.
- `device_indices` selects which GPUs to use; null means GPU 0 alone. Using every available GPU when none are named is not built, so name the indices to use more than one.
- Detection parallelizes across GPUs by part: one process per video part, assigned round-robin across the selected GPUs, each reading its part independently with no shared state. More parts than GPUs simply queue.
- Embedding parallelizes across GPUs by `(track, part)` unit, on the same round-robin pattern: each unit produces an independent per-track-per-part output store, so workers never contend on storage. Within a single GPU, the embedding loop batches frames through `model.embed_batch` to amortize kernel-launch overhead.
- Each model, tracker, and embedder declares which backends it supports. A run is rejected up front if its compute target isn't one of them.
- Cloud compute and cloud storage are designed but slated for a future release.

## Supporting functionality

- **Executor**: runs a job as a process.
  - A local executor runs each job as a subprocess. Jobs are plain data, so a cloud executor could replace it without changing callers; none is built.
  - Progress streams while the job runs.
  - Only some runs resume. Training continues from its last checkpoint and keypoint detect from the last frame it wrote. Detection, tracking and embedding do not resume: a failed processing run re-runs its incomplete recordings whole (`process retry`).
- **Custom model training**: two phases.
  - Generate a labeled dataset by sampling and auto-labeling frames from chosen recordings.
  - Train a model on it with a pluggable trainer. The trained model registers with the experiment and is usable by any detection run.
- **Grouping variables and analysis** (experiment level):
  - `metadata.csv`: one row per recording with the variables compared across groups (animal id, treatment, condition, ...).
  - Analysis runs live at the experiment level since one analysis can span recordings. Four modules exist: location export (a tracking run plus each recording's ROIs/POIs), behavior scan (`query`: finds a behavior from example clips in an embedding run), its evaluation against labels (`query_eval`), and behavior clustering (`temporal_clustering`: bouts from an embedding run's pose trajectories).
  - Comparisons contrast a behavior result between groups of recordings defined by `metadata.csv`.

## Python API and CLI

- Everything runs through the Python API. From a script the pipeline is a short chain; each call returns a handle to wait on:

  ```python
  det   = recording.run_detection(...).wait()
  track = recording.run_tracking(det.id, ...).wait()
  emb   = recording.run_embedding(track.id, ...).wait()
  ```

- The CLI mirrors this (`detection run`, `tracking run`, `embedding run`, ...). The GUI calls the same methods.
- Every `run_*` call returns a handle to stream progress, block until done, or cancel.
- A setting with no safe default is resolved when the run is validated and the resolved value is written to the run record. When the choice is ambiguous the call raises and names the choices, rather than guessing.
