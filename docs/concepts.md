# DETAIL concepts

The shared vocabulary and mental model behind the tool, referenced by the [CLI guide](cli.md) and the [GUI guide](gui.md). Read this once; the guides assume these terms.

!!! note

    Under active development. This describes the current model on `main`; features marked *planned* are not built yet.

## The pipeline at a glance

The pipeline turns behavioral video into per-animal feature embeddings, one stage feeding the next:

```
experiment
└── recording                 one experimental subject/recording (video + optional streams + metadata)
    ├── detection           find the target each frame (text-prompted model)  → det_01
    ├── tracking            link detections into a subject track over time     → track_01
    ├── keypoint detect     place named body points on each detection          → kd_01
    ├── keypoint track      give each pose to a tracked animal, over time      → kt_01
    ├── embedding           embed the tracked subject's masked image per frame → emb_01
    └── analysis            experiment-level: analyze a run across recordings  → an_01
```

Each stage reads the previous stage's output and writes a new, **immutable** output. You don't edit a run; you create another one with different settings. Downstream stages name the specific upstream run they depend on (embedding names a `tracking_run_id`), so multiple runs coexist without conflict.

## Experiment, recording, run

### Experiment

The top-level container. Holds recordings, processing runs, models, and experiment-wide
metadata, in one self-contained, portable directory.

### Recording { #recording }

One experimental unit, typically one subject in one recording. Holds its video, optional event
streams, a field of view, ROIs/POIs, and its row of recording information. Video is optional; a
metadata-only recording is valid. One animal per recording.

A recording has a **name** and an **id**, and they are not the same thing:

| | Name | Id |
|---|---|---|
| Whose it is | Yours | The tool's |
| Starts as | The video's filename | `recording_000`, assigned in order |
| Changeable | Yes, and must stay unique | Never |
| Where it shows | Every screen | Run records, exports, the CLI |

So **never rename a video file to organise your data**. Rename the recording instead. The file
can move (relink it) and a recording can be built from several files, without disturbing what
identifies it. Commands take a name or an id wherever they take `--recording`.

### Video part { #video-part }

A recording's video can arrive in pieces. Each imported file is a part (`part_001`, `part_002`),
concatenated in time. Most recordings have one.

By default a part is **copied into** the recording. A part can instead be **referenced in
place** (`add-video --reference`, or the GUI's reference-in-place checkbox): the file stays where
it is and the recording stores its location plus a content fingerprint, so terabytes are not
duplicated. The tool remembers the file's folder automatically; there is nothing to configure.

A moved or changed referenced file is detected, not silently used. `recording verify` reports it,
and every run refuses to start on a missing or changed video. Fix a moved file with `recording
relink` (or the GUI's **Locate video**); a changed file means remove and re-add. On an experiment
made on another computer, locating one file fixes every recording whose videos share its folder.

### Run

One execution of a stage over a recording, with a specific config. The id carries a stage prefix
and a per-recording counter: `det_01`, `track_01`, `emb_01`. Runs are immutable and re-runnable;
a failed or interrupted run is marked `failed` or `partial` and can be re-run.

### Processing run

An experiment-level run of one stage across a selection of recordings (`run_001`, `run_002`). It
records the per-recording run it created in each recording, and is the unit a training dataset is
built from. Create one with `process detection`, `process tracking`, `process embedding`,
`process keypoint_detect` or `process keypoint_track`.

### Use every N frames

How many frames a run steps by. Detection, tracking, embedding and keypoint detect each take
one, and 4 means
"use frames 0, 4, 8, …". The cost falls roughly in proportion, and each form shows the rate that
lands at (`7.5 fps (source 30 fps)`).

It counts **source** frames at every stage, so nothing is renumbered and a downstream stage can
only go coarser by a multiple of what it received: after detection at 2, embedding can use 2, 4,
6, … Nothing is copied or re-encoded. The recording keeps one video, and a run's rows say which
of its frames were looked at.

### Upstream selection

A batch picks its input (which detection run to track, which tracking run to export) by the run's
**kind** (model, settings, upstream lineage), not its per-recording id. Each recording uses its
own matching run, so `det_01` in one recording and `det_02` in another are recognized as the same
detections and tracked together.

In the GUI, pick the kind from a list labelled with coverage; recordings without a match are
dropped after you confirm. On the CLI, name one example run with `--detection-ref
<RECORDING:det_NN>` (or `--tracking-ref`). The run fails if any selected recording has no match,
so name a run whose kind every selected recording shares. `analysis run` (`--tracking-ref`) and
`analysis temporal-clustering` (`--embed-ref`) take the flag more than once, one example per
kind, so recordings processed through different chains (two detection models, say) each use
the run of the named kind they have.

### Snapshot { #snapshot }

A frozen copy of an analysis run's results, saved automatically whenever a number leaves the run:
an export, a comparison, or pressing Finish.

Some analyses stay open after the computing is done because you are still interpreting them
(labelling clusters, setting a cutoff), so the run itself keeps changing. A
snapshot never changes, so a figure you produced last month is still reachable exactly as it was
after you reopen the run and re-cut it. Read them back from the run's page, or with `analysis
snapshots`.

## The processing stages

### Detection run { #detection-run }

Runs a detection model with a **text prompt** (`"mouse"`) over every frame, writing one row per
detection: bounding box, mask, confidence. Frames with no detection have no rows.

### Confidence threshold { #confidence-threshold }

How sure the model is, 0 to 1. It means two different things depending on where you set it.

On a **detection run** it is a filter: anything below it is never written, and changing your mind
means another run. On a **preview** (the prompt test, and Quick segment) the model runs wide open
and every proposal is kept, so the threshold is only what you are looking at: a slider in the
GUI, `--min-conf` on the CLI.

Slide it down to see what a looser setting would have caught, up to see what a stricter one would
cut, then set the run's `--conf` once you know. A run that filtered the low-confidence rows
cannot get them back.

### Track

The sequence of detections judged to be the same subject over time, identified by an integer
`object_id`. A **tracking run** assigns these ids. For a single subject the track is `0` in every
frame.

### Tracker

The algorithm that turns detections into tracks. The single-subject trackers are `anchor_tracker`
(the default: the single best contiguous detection each frame) and `iou_single_tracker` (followed
frame to frame by IoU continuity, so a distractor cannot steal
it). See the [CLI
guide](cli.md#tracking) for when to use which.

### Embedding run { #embedding-run }

For a chosen track, renders the subject's masked image each frame and embeds it into a feature
vector. One vector per frame the subject appears in.

### Where the subject was { #where-the-subject-was }

An embedding run also records, per frame, the subject's position, mask area and box, normalized
to the frame so two recordings at different resolutions are comparable. The embedding itself
carries no position and no orientation, so a cluster confined to one corner is invisible in the
embedding and obvious on a map.

A run made before this was captured has none. `detail recording backfill-kinematics` regenerates
it from the detection masks without re-embedding.

### Keypoints { #keypoints }

Named body points (the nose, the ears, the tail base) placed on the animals in every frame, in
two stages. **Keypoint detect** (`kd_NN`) runs on a **detection run** and places points on every
animal it detected, frame by frame. It keeps the raw result and gives no pose to any animal
yet: each pose, the method's own score for each point, and the method's top candidates, so a
later stage can choose among them without running the model again. The method `dinov3_match`
works on each detection's crop and learns from an **example set**, a few frames on which you
marked the skeleton; it needs no training. A trained student looks at the whole frame instead,
and keeps each pose it is at least 0.25 confident in (**Minimum pose confidence**; poses below it
are not stored, so changing it means running again). **Keypoint track** (`kt_NN`) runs on a
detect run and a **tracking run** made from the same detection run. It first gives each pose to
a tracked animal: the one whose mask, grown slightly, holds the larger share of the pose's
points, at least half of them by default, and never two poses to one animal on one frame. A
pose that fits no animal stays unassigned, and the run counts how many did. Then it settles
each animal's points over time: for `dinov3_match` a path through the stored candidates that
keeps each point moving plausibly, left/right pairs held to one reading over frames, then a
despike and a smooth; for a student, the despike and the smooth alone. Each detect method opens
on its own track method. It is seconds of CPU, so another tracking run or other settings are a
new run over the same detect run and the raw result is never rewritten. Detect runs made before
this split (one row per tracked animal) are refused with a message to run keypoint detection
again. Either run exports its points for other pose tools (DeepLabCut, SLEAP, NWB), as
DETAIL's own long table, or as a video with the points drawn on; see
[Keypoints, for pose tools](cli.md#keypoints-for-pose-tools). An embedding run can **align** on a track run: through a **heading definition**
(the points whose mean is the front, the points whose mean is the back) each subject image is
turned to face one way before the model sees it, rotated for an overhead camera or mirrored
for a side camera. A frame with no heading within a short gap is not embedded, so an aligned
run never mixes turned and unturned pictures, and it never compares with an unaligned one.
The spec is DESIGN_PLAN §26 and §14.


**A skeleton** is the list of points, in the order the tool stores them, plus the edges that join
them (drawn, and used to keep a placement coherent) and the left/right pairs among them. A name
is just a label: the tool never guesses what a point is from its name, so which point is the
animal's left is something you say, not something it infers. Skeletons live in one library on
the machine, the **Keypoints manager**, shared by every experiment; the built-in presets
(Quadruped overhead view with 12 or 8 points, Quadruped side view, and Midline with 3 points)
are read-only entries in it.
Whatever uses a skeleton keeps its own copy, so a missing library never breaks a run. While
anything uses a skeleton its points stay fixed, and an edit is saved as a new skeleton. A
skeleton also remembers where you drew each point (its layout), so it reopens as you
left it; that is a picture of the skeleton, not part of what it means, and no run reads it.

**An example set** is a skeleton plus a few decoded frames with its points clicked on them,
made where you choose a model that needs one. It is what a
patch-matching keypoint method learns from, the way example clips are what a behavior query
learns from, so it is yours to make and never built from ground truth. One set belongs to the
experiment and serves every recording. The tool samples the frames for you (varied postures by
default, or a spread over time), and you click each point or say it is not visible. To reuse a
skeleton, make a new set on it; a set's skeleton can change until its first mark, and is
fixed after that. A run will
record the set's digest and a copy of the marks it learned from, so a set can be edited afterwards
without altering a run that used it.

**A distilled student** is the keypoint counterpart of [distilling a detector](#detection-models).
The teacher is the patch method itself, run on the sampled frames from an example set and a
tracking run, the way SAM 3 labels frames for a detector, or a finished keypoint detect run,
whose stored placements are read back. Either way the poses are given to the tracking run's
animals the way keypoint track gives them (a detect run uses the one tracking run on its
detection run). Both are raw, frame by frame, with no smoothing over
time, and a **score cut-off** you choose decides which placements are taught as visible; **Test
cut-off…** (or `detail keypoints test`) shows the teacher on a sample of the frames the distill
would draw, and checks it against your example clicks in pixels, before you commit to one. Its
placements, with the example set's own clicks, become a YOLO-pose training set (`task: pose`) that a fast closed-set model is trained on. The student needs no example set at run time, runs on any backend, and is offered as a
keypoint method beside `dinov3_match` once its distill run registers it under `role: keypoints`.
Only frames where every animal is assigned to a track are used, so no animal in a training image
goes unlabelled, and audit frames never are, so the labels that score a method never build it.
The dataset records every frame it used, so a later audit of the student can leave them out,
and keeps the teacher's raw output for those frames.
`detail keypoints distill`, or **＋ Distill new model** with Model type *Keypoints*, runs label, train and
register as one durable run; `detail dataset keypoints` builds the dataset alone. Promote the
student to offer it in every experiment, and export it to use it on another machine (see
[Imported model](#imported-model)).


## Scene geometry

### Field of view { #field-of-view }

An optional polygon that **bounds detection**. Tracking ignores detections whose center falls
outside it. It can move over time (keyframes). The storage kind is `arena`.

Recordings off one rig share it, so it carries over from another recording (or an exported CSV)
rather than being redrawn each time. A copy replaces what is there, and is refused while a
tracking run depends on it.

### Region of interest (ROI) and point of interest (POI) { #roi-poi }

Named **analysis-time** shapes drawn on the video. An ROI is a polygon (a zone); a POI is a
single landmark point (lever, port, hopper). Unlike the field of view they do not touch detection
or tracking; they tag position for analysis, which the location export reads. Edit them per kind
in a manager, and export or import to reuse them across recordings.

### Distance calibration { #distance-calibration }

Two points a known real distance apart give a **pixels-per-mm** scale, so location distances can
be reported in millimetres rather than pixels. Optional, stored per recording, and rescaled if
the video resolution differs. Like the field of view it carries over between recordings, keeping
the resolution it was measured at.

It carries over only onto a video of the **same shape** (aspect ratio). Across a change of shape
one pixels-per-mm is wrong in some direction, so the copy is refused, and a calibration whose
video was replaced by one of another shape is reported as unusable rather than read.

## Streams and labels

### Stream

An aligned time-series imported alongside the video. Each stream has a **role** (what it is, for
example `behavior`) and a **shape** (event, interval, or continuous). MED-PC operant events are a
signal stream; behavior labels are a `behavior`-role interval stream.

Bring your own data as CSV: continuous sensor traces (`timeseries_csv`) or timestamped events
(`events_csv`). A file already shaped for DETAIL needs only a time column named `seconds`,
`frame` or `t_ns`. For any other export, say which column is time and what it holds (a frame
index, seconds, raw `t_ns`, or a UTC timestamp) and which columns to bring in as channels. One
CSV makes one stream carrying those channels.

Streams are aligned to video time and previewable. No stage consumes a signal stream
automatically yet (*planned*), but the location-export CSVs are `t_ns`-keyed, so you can join a
stream on time yourself today.

### Behavior labels

Human-scored ground truth of which behavior occurs over which frames, imported as a
`behavior`-role interval stream. Import from a plain interval CSV
(`start_frame,end_frame,label[,subject]`) or a BORIS export, per recording (`recording add-stream
--importer behavior_csv`) or in bulk (`recording import-labels`).

Each behavior is a **point** (an instant) or a **state** (a duration). The annotation records the
behaviors it **covered**, the scored vocabulary, so a recording is only judged on behaviors it
actually scored.

A stream also records its **origin**: `ground_truth` (a person scored it) or `predicted` (a
method produced it, re-ingested). A predicted stream is never read as ground truth. It does not
make a recording eligible for an evaluation, and naming it for scoring is refused, because a
method scored against its own output produces a good-looking number that measures nothing.
Anything you import is ground truth unless the file says otherwise or you pass `--origin
predicted`. Every label file DETAIL exports carries an `origin` column, so re-importing a
method's output brings it back as `predicted` on its own.

A label file is one animal in one session. A BORIS export holding several observations or
subjects, or a MED-PC file holding several boxes, is refused until you pick one; they are never
merged onto one recording.

## Detection models

### Model and variant { #model-variant }

A stage names a model by id. Three kinds:

| Kind | Id | What it is |
|---|---|---|
| Built-in | `sam3` | Open-set: you give it a text prompt. Some take a size *variant* |
| Trained | `model_NNN` | Fit in your experiment |
| Imported | your id | Your own weights |

Trained and imported models are *local*: closed-set, no prompt, detecting the classes they were
trained on. On the CLI all three are `--model <id>`.

### Dataset

A training dataset of images plus YOLO-format polygon labels (`dataset_001`), from one of two
sources: exported from a processing run, reusing its stored masks with no new inference (`dataset
create`), or built fresh by sampling frames and labeling them with a prompt-capable model
(`dataset label`). The labeling source samples frames *stratified* across each timeline by
default, or *uniform*.

### Trained model

A custom, closed-set detection model fit from a dataset (`model_001`), stored in the experiment
and usable by later detection runs (`--model model_001`), or a keypoint student distilled for
keypoint detect runs. Build with `train run`, `distill` or `keypoints distill`; `train promote`
copies one into the machine-global catalog, a keypoint student with its skeleton.

### Imported model

External weights (a YOLO `.pt`) registered into a **machine-global** catalog, usable in every
experiment: a detection model, or a keypoint (pose) model. The checkpoint's task decides which.
A detection model's class names come from the checkpoint, so sharing it is sharing its `.pt`. A
keypoint model also needs the skeleton its points follow, which you supply on import; it is
checked against the model's number of points and left/right pairs, though not against a pair
named the wrong way round. A promoted model, or one restored from an exported `.zip`, joins the
same catalog, and a keypoint model carries its skeleton with it. Add one with `model import` or
`model restore`, or **Import model** / **Restore from zip** on the GUI's Model manager.

## Analysis

### Analysis run

An experiment-level run of an analysis module over a selection of recordings (`an_01`), written
to `analysis/<id>/`. It reads finished data and produces free-form artifacts as a single job.
Build with `analysis run`.

### Location export { #location-export }

Turns a tracking run into per-frame, per-track **spatial features** around a recording's ROIs and
POIs (`tracks.csv`, `roi_relations.csv`, `point_relations.csv`), keyed by `t_ns` so they join
event streams on time. Distances and positions come in normalized and pixel units, plus
millimetres (`_mm`) when the recording has a distance calibration. Run it big (`analysis run`,
many recordings) or quick (`locations export`, one recording).

### Behavior scan (query) { #query }

**Behavior scan** is the app's name for the `query` module, which is the name `analysis query`
and the run records use. Name a behavior, give a few example clips, and search a whole cohort
for it. Then set **one
cutoff per behavior** and get instances with start and end times, not a frame-by-frame score.

The search is computed once and never changes; the cutoff is yours, and re-projecting at a new
one is instant, so you can move it while watching what falls in and out. Precision is favoured
over recall by design. Build it with `analysis query` and set cutoffs with `analysis
query-calibrate`.

### Behavior-query evaluation { #behavior-query-eval }

Scores how well the query retrieves a labeled behavior against the ground-truth streams.
Held-out example clips seed the query; the rest of the labeled instances are the answer. Its
purpose is to compare configurations: vary the embedding run (embed model and size,
input-image constructor, input size) or the query parameters and read the average precision
side by side.
**Folds** cross-validate for a seed-independent number.

The scoring **method** comes in two families:

- **Windowed matchers** (`dtw`, `clip_pooled_mean`/`max`, `keyframe`) return matches about the
  example's length, scored by temporal overlap as a span-detection average precision. Overlap is
  IoU for state behaviors and a time window for point behaviors.
- The **per-frame classifier** `soft_knn` scores every frame into a threshold-free per-frame
  average precision, voting over the `k` nearest labeled frames (the example frames plus
  `n_negatives` background frames, with `tau` the temperature).

Every score is reported beside its **null**, the same result with the answer destroyed, so the
**lift** over the null is the part that is real.

Run it scriptably (`analysis query-eval`, one CSV row per behavior) or as a recorded GUI run
whose report and **Explore clips** window play the matches. Full detail:
[Behavior-query evaluation](behavior-query-eval.md).

### Behavior clustering

Discover a vocabulary of **behaviors** with no example to point at, pooled across a selection of
recordings with no labels. A cluster is a stretch of movement and an instance is one performance
of it, with a real start and end. The clusters form a tree, from many small ones to a few large
ones.

Two **methods** do this differently, and which you want depends on your study's size.
`codebook_viterbi` (the default) clusters a sample of *frames* (the same share of every
recording) into a set number of codewords, labels every frame against them, and recovers the
stretches with a decode; its expensive step grows with the square of the codeword count, which you
choose, and otherwise in proportion to how much video you pool. `dtw_finch` instead cuts each
subject's trajectory into **bouts** at the moments the pose changes and compares every pair,
which is what lets a slow and a fast performance of the same behavior match, and is why its cost
grows with the square of the bout count: fine for a short study, slow for a long one. Measured
against each other they tie on vocabulary quality, and the bout path draws slightly better
boundaries.

Two things to know:

- Its instances cover **less** than the observed time: stretches shorter than the method's
  minimum (a bout, or a decoded run) are dropped, and gaps in tracking are not behavior. That
  gap is how much of the recording the vocabulary accounts for, not a fault.
- For `dtw_finch`, **Max bouts** bounds how many bouts the comparison sees, because the
  comparison grows with the square of the bout count. Every other bout is still matched to the
  nearest cluster afterwards, so a cap changes what the vocabulary is *learned from*, never what
  gets labeled.

The fit is computed once and stays put. Which clusters get a name, and what it is, are decided
afterwards by labelling nodes of the tree, at whatever level a behavior shows up, and cost
nothing to change.

Run it in the GUI (**Behavior clustering** analysis) or with `analysis temporal-clustering`, then
`analysis temporal-interpret`.


### Measure

A number taken over a stretch of a behavior result: how many instances, how much total time, a
rate (bouts per second of observed time), a mean bout duration. Every measure is computed over a **window** (the whole
recording, a fixed span, a fraction of it) and reported against **observed time**, so a subject
tracked for half a recording is not scored as if it were absent for the other half.

Measures are what a comparison consumes, and you can read them on their own: an analysis run's
page shows them, and `measures export` writes them as CSV. Full list:
[Measures](measures.md).

### Comparison { #comparison }

The group question: does this differ between these animals? Pick a behavior result, filter to the
recordings you want, group them by a column of your recording information (treatment, sex,
timepoint), and read the [measures](measures.md) side by side. Saved to `comparisons/<cmp_NN>/`
and exportable as tables, figures and a `stats.csv`.

Three things to know:

- **You say what one observation is** (`--aggregate-to`) whenever your cohort has repeats.
  Averaging each animal's recordings first, or treating every recording as its own point, give
  different answers as soon as one animal contributed more than one recording, so the tool
  refuses to guess and records the choice on every row. When no animal contributed more than one
  there is nothing to decide, and it runs with one observation per recording.
- It stops at **description**: means, medians, spreads, and a permutation floor saying how big a
  difference shuffling the group labels would produce anyway. The only p-values are the floor's
  own: for each measure or cluster, the share of the shuffles that reach it, against the whole
  family (the floor read as a number) and against its own shuffles alone, with a
  false-discovery q over them, so a cohort too small to clear the floor can still say which
  clusters look consistent. No model sits behind them; `stats.csv` is the handoff to whatever
  you test with.
- **You choose the column; the tool picks the design.** A column that is the same for all of an
  animal's recordings (sex, genotype) compares *between* animals. A column that changes within
  an animal (session, dose) compares *within* each animal: two values are a paired contrast,
  a numeric column with three or more is a trend across them, and three or more named values
  (sessions) are a *profile*: each behavior is scored by the pair of values it differs most
  between, and the floor shuffles each animal's sessions and searches the pairs the same way,
  so picking the pair is already paid for. Name two values of a column to contrast just those
  two. Every result says which design ran, and its floor is the one that
  design allows: shuffling labels across animals for a between design, flipping each animal's
  own direction or shuffling its sessions for a within one, so the pairing is never thrown away.

**Screening clusters nobody has named.** A comparison over a clustering run can measure its
**nodes** instead of its labels: every cluster in the tree, named or not, ranked by how its share of
time differs across your column, with its rate (bouts per second) and mean bout length shown
beside it. The
ranking is a list of *where to look*: a node is kept when its effect clears the floor in enough of
your animals, nested slices of one behavior are collapsed to the best of them, and each kept node
carries a one-line reason ("less under drug than vehicle, 2.1×; 8 of 10 animals in that
direction"). Naming the node is still yours to do, from the clips. A label you write from a
screen remembers it: compare that label across the same column later and the result says the
screen chose the cluster it is scoring, so the effect is not mistaken for an independent finding.

## Export

Getting data out in formats other tools read. Every result surface carries the same **Export**
control, and `detail export` is the same catalogue on the command line. Each export writes its
own folder, named for the experiment, module, run and time, so two never collide.

An export carries the data behind an artifact as well as the artifact: a `clips.csv` naming the
bout each clip file holds, a `params.csv` of the run's settings, your notes on the run, and PNG
plus SVG for anything with a picture. A run exports what it made: masks, tracks, embeddings,
keypoints and analysis results. A recording exports its streams, its frame timestamps, its
scene geometry, a clip of a frame range, and the video parts DETAIL holds a copy of; a video
referenced in place is already yours, under its own folder, and is not copied back out.

The experiment **bundle** is a separate format, and the only export that imports back. See
[Getting your data out](cli.md#getting-your-data-out).

## Where experiments live (the managed workspace)

The GUI and `Experiment.list_all()` discover experiments under a single **managed workspace**, by default `~/Documents/DetailExperiments`. Override it by creating `~/.config/detail/storage.yaml`:

```yaml
storage:
  default:
    root: /data/segexperiments
    model_cache: /data/detail/models   # optional; shared model-weight cache
```

`model_cache` sets where auto-downloaded model weights live (default `~/.cache/detail/models`). Point it at a shared location so several machines use one weight cache instead of each re-downloading. `DETAIL_MODEL_DIR` overrides it for a single process.

The **CLI** is explicit instead: every command takes `--experiment <dir>`, and `experiment create --dir <parent>` drops the experiment wherever you point it. To see it in the GUI too, create it inside the managed workspace (`--dir ~/Documents/DetailExperiments`). An experiment directory is self-contained with relative paths, so you can also copy or [export/import](cli.md#moving-experiments-between-machines) it between machines.

## What an experiment looks like on disk

Plain files: YAML records plus per-stage outputs. Every run is immutable and lands in its own `<stage>/<id>/` folder; a later stage reads the previous stage's output by id.

```text
<experiment>/
├── config.yaml
├── recordings/
│   └── recording_001/
│       ├── recording.yaml               # parts + this recording's run index
│       ├── video/                     # imported video parts
│       ├── detections/det_01/         # per-frame masks (zarr)
│       ├── tracking/track_01/         # tracks + assignments
│       ├── keypoint_detect/kd_01/     # raw poses per frame
│       ├── keypoint_track/kt_01/      # poses given to tracked animals
│       ├── embeddings/emb_01/         # per-track feature vectors
│       ├── arenas/  rois/  points/    # arena / ROI / POI definitions
│       └── streams/                   # aligned event streams (e.g. MED-PC)
├── runs/run_001/                      # experiment processing-run records (batch across recordings)
├── analysis/an_01/                    # analysis outputs (locations, behavior scan, clustering)
│   └── snapshots/snap_01/             # frozen copies, saved whenever a result left this run
├── comparisons/cmp_01/                # saved group comparisons across recording metadata
└── datasets/  models/                 # detection-model training surface
```

## Models, backends, and access

| Model | id | Role | Access | Compute |
|-------|----|------|--------|---------|
| SAM 3 | `sam3` | detection | **gated** (HF token + access; source install) | CUDA (official) |
| Trained / imported YOLO | `model_NNN` / imported id | detection (closed-set) | **local** (no account) | CPU / MPS / CUDA |
| DINOv3 ConvNeXt | `dinov3_convnext_base` (variants `convnext_tiny` / `_small` / `_base` / `_large`) | embedding | **gated** (HF token + access) | CPU / MPS / CUDA |
| DINOv3 ViT | `dinov3_vit` (variants `vit_small` / `vit_base` / `vit_large`); **the default embedding model**, at `vit_large` and 224 px | embedding | **gated** (HF token + access) | CPU / MPS / CUDA |
| DINOv3 patch matching | `dinov3_match`, a keypoint method that reads the `dinov3_vit` weights' patch features (default `vit_large`); learns from an example set, no training | keypoint | **gated** (install `dinov3_vit`) | CPU / MPS / CUDA |
| Keypoint student / imported YOLO-pose | `model_NNN` / imported id | keypoint (closed-set) | **local** (no account) | CPU / MPS / CUDA |

- **Install a model before you run it.** Weights download once, into a global cache shared by every experiment (`~/.cache/detail/models`, or `DETAIL_MODEL_DIR`). `detail model install <id>`, or **Install** on the GUI's **Model manager**, is the only step that downloads. A **run never fetches weights**: it fails immediately if they are missing rather than stalling mid-pipeline on a silent multi-GB download. Once installed, runs need no network.
- **Each model size is its own download.** A model with size variants installs one size at a time: `detail model install <id> --variant <size>`, or the matching row under **Hugging Face** on the GUI's **Model manager**. A run gates on the exact size it was configured with, so installing the default does not make another size runnable.
- **`detail model list` is the pre-flight check:** every model and size, and whether it's installed, with no network calls.
- **A trained or imported YOLO runs anywhere:** closed-set, no Hugging Face account, CPU/MPS/CUDA. It is the distillation target: label a sample with SAM 3, train a YOLO, then run the fast YOLO everywhere. **Distill** does both as one durable run (`detail distill run`, or **＋ Distill new model** on the Model manager) that survives closing the app and resumes a failed one. Labeling (CUDA-only SAM 3) and training take their own compute target.
- **Promote a trained model to reuse it elsewhere.** It is usable in its own experiment at once; `detail train promote`, or the model page's **Promote to global**, joins it to the machine-global catalog as a *distilled* model carrying its provenance (source experiment, dataset, base model, metrics).
- **Gated models** (SAM 3, DINOv3) need a Hugging Face token *and* access granted to each repo (see [Hugging Face access](getting-started.md#hugging-face-access)). `detail model check <id>` explains what's missing. Access alone isn't enough: a granted model still reports `not_installed` until you install it.
- **DINOv3 read-out and input size** are per-run knobs. `--readout` picks how the backbone output becomes one vector (ConvNeXt: `pooler` (default) / `gap` / `gmp` / `gap_gmp`; ViT adds `cls_patch`). `--input-size` sets the model input resolution.
- **Compute** is selected per run: `cpu`, `mps` (Apple Silicon), or `cuda`. CUDA runs can use multiple GPUs (detection and embedding fan out per device).
  {: #compute-target }

## Current scope / planned

Built today, reachable from the API, GUI and CLI:

| | |
|---|---|
| Processing | Detection (SAM 3, trained and imported YOLO), tracking (`anchor_tracker`, `iou_single_tracker`), keypoint detect (`dinov3_match`, a distilled or imported YOLO-pose model) and keypoint track, embedding (DINOv3) |
| Scene geometry | Field of view, ROIs and POIs, distance calibration |
| Streams | MED-PC operant events, CSV sensor traces and events, behavior labels |
| Analysis | Location export, behavior scan (query), behavior-query evaluation, behavior clustering, measures, comparisons |
| Detection and keypoint models | Processing runs, datasets, trained models, the unified **distill** run (a detector, or a keypoint student), promotion to the machine-global catalog |
| Getting data out | The `export` backbone and an **Export** button on every result screen; experiment bundles that import back |
| Tools | One-off, experiment-free runs, starting with **Quick segment**. Runs live in a machine-global `tools/` folder (see the [Tools guide](tools.md)) |

Planned, referenced but not yet usable: multi-animal tracking, and a supervised **behavior
classifier** over embeddings. Behavior clustering is built; a supervised classifier is not.

## Next

- Driving DETAIL from the terminal or a script: [CLI guide](cli.md)
- Driving it from the desktop app: [GUI guide](gui.md)
