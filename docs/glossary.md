# Glossary

Core terms used across the tool.

**Experiment**: the top-level workspace; holds recordings, runs, and configuration.

**Recording**: one experimental unit, typically one subject in one session: its video (optional; one or more video parts), streams, scene geometry and recording information, and the runs made from them.

**Video part**: a single video file within a recording; a recording's footage may span several parts.

**Referenced video**: a video part left where it is instead of copied into the recording (`add-video --reference`). The recording stores its location plus a content fingerprint, so a moved or changed file is detected rather than silently used. **Video root**: the machine-local folder a referenced part lives under, registered by alias so one relink fixes every recording pointing at it.

**Detection**: the stage that finds subject masks or boxes per frame (a model such as SAM 3 or a trained YOLO).

**Text prompt**: what a prompt-capable detection model (e.g. SAM 3) searches for, e.g. `mouse`. Ignored by closed-set local models, which detect the classes they were trained on.
{: #text-prompt }

**Class name**: the label stored on each detection, how it appears in the outputs. Often the same word as the text prompt.
{: #class-name }

**Tracking**: links per-frame detections into per-subject tracks over time.

**Keypoints**: named body points (the nose, the tail base) placed on each animal in every frame. **Keypoint detect** places them on a detection run's animals; **keypoint track** gives each pose to a tracked animal and settles its points over time. See [Keypoints](concepts.md#keypoints).

**Skeleton**: the list of keypoints, in the order the tool stores them, plus the edges that join them and the left/right pairs among them. Skeletons live in the machine's **Keypoints manager** library.

**Example set**: a skeleton plus a few frames with its points marked by hand, which the patch-matching keypoint method learns from.

**Field of view**: a labeled region of the frame that bounds detection; tracking ignores subjects whose center falls outside it. It can move over time (keyframes) and carries over between recordings off the same rig. Stored under the kind `arena`, which is what the older CLI options and on-disk paths still call it.

**ROI (region of interest)**: a labeled polygon in the frame; location export measures each subject's position relative to it (inside / outside, distance).

**POI (point of interest)**: a labeled point in the frame; location export measures each subject's distance to it.

**Embedding**: computes a feature vector per subject per frame (a model such as DINOv3) for similarity and analysis.

**Use every N frames**: how many frames a run steps by. 4 uses frames 0, 4, 8, … Set per run on detection, tracking, embedding and keypoint detect, and shown beside the rate it lands at. Counted in source frames, so a downstream stage can only go coarser by a multiple of it.

**Analysis**: deriving results from tracks and embeddings, an **experiment-level** run over a selection of recordings (location export, behavior scan, behavior clustering), separate from the per-recording stages.

**Location export**: an analysis that writes each subject's per-frame position, body length and width, and spatial features relative to the recording's ROIs/POIs (`tracks.csv` / `roi_relations.csv` / `point_relations.csv`), keyed by `t_ns`.

**Behavior scan**: an analysis that searches a cohort for behaviors you can name, from a few example clips each; the `query` module on the CLI and in run records. See [Behavior scan](concepts.md#query).

**Cutoff**: in a behavior scan, the confidence a match needs to become an instance, set per behavior after the search; higher is stricter. The search stores no cutoff, so changing it never searches again.

**Behavior clustering**: an analysis that finds a vocabulary of behaviors with no labels or examples, as a tree of clusters you then label. See [Behavior clustering](concepts.md#behavior-clustering).

**Measure**: a number taken over a behavior result per recording, subject, behavior and window: a count, total time, a rate, a mean bout duration. See [Measures](measures.md).

**Comparison**: measures of one behavior result read side by side across groups of recordings, with a permutation floor. See [Comparison](concepts.md#comparison).

**Snapshot**: a frozen copy of an analysis run's results, saved automatically whenever a number leaves the run (an export, a comparison, pressing Finish). An analysis you are still interpreting keeps changing; a snapshot does not, so a figure you produced last month stays reachable exactly as it was. Read one back from the run's page or with `analysis snapshots`.
{: #snapshot }

**Run**: one execution of a stage, recorded with its config and outputs. Per-recording stage runs are indexed in the recording (e.g. `det_01`); experiment-level runs live at the experiment.

**Processing run**: an experiment-level batch that applies one processing stage with one config across a selection of recordings (id `run_NNN`); it fans out to a per-recording run in each.

**Analysis run**: an experiment-level analysis over a selection of recordings (id `an_NN`), written under `analysis/`.

**Dataset**: a training set of frames and labels (`dataset_NNN`), built from a processing run's masks, by labelling sampled frames with a prompt-capable model, or from a keypoint teacher.

**Distill**: one durable run that labels a frame sample with a slow, capable teacher and trains a fast YOLO model on it: a detector, or a keypoint student (`distill_NNN`).

**Trained model**: a closed-set model fit in an experiment (`model_NNN`), by `train run` or a distill. **Imported model**: your own YOLO weights registered in the machine-global catalog.

**Stream**: a time-series aligned to video time (e.g. MED-PC operant data), keyed by `t_ns` so it joins the pipeline outputs. Each stream declares a **role** (what it is, e.g. `behavior`) and a **shape** (event / interval / continuous).

**Behavior labels**: human-scored ground truth of which behavior occurs over which frames, imported as a `behavior`-role interval stream (from a plain interval CSV or a BORIS export). Each behavior is a **point** (an instant) or a **state** (a duration), and the annotation records the behaviors it **covered** (the scored vocabulary), so a behavior nobody scored is told apart from one that did not happen.

**Job**: the executor unit (a subprocess) that carries out a run; distinct from the run record it writes.

**`t_ns`**: a timestamp in nanoseconds from the start of the first video part; the universal time reference.

**Track**: one subject's trajectory (identified by a track id) through a recording.

**Band**: the **DTW band** of behavior clustering's `dtw_finch` method: the warp tolerance, in seconds, allowed when comparing two bouts; `0` is a rigid match. It is what lets a slow and a fast performance of the same behavior match.
