# Glossary

Core terms used across the tool.

**Experiment**: the top-level workspace; holds recordings, runs, and configuration.

**Recording**: one recording (subject or experiment); contains one or more video parts and the data derived from them.

**Video part**: a single video file within a recording; a recording's footage may span several parts.

**Referenced video**: a video part left where it is instead of copied into the recording (`add-video --reference`). The recording stores its location plus a content fingerprint, so a moved or changed file is detected rather than silently used. **Video root**: the machine-local folder a referenced part lives under, registered by alias so one relink fixes every recording pointing at it.

**Detection**: the stage that finds subject masks or boxes per frame (a model such as SAM 3 or a trained YOLO).

**Text prompt**: what a prompt-capable detection model (e.g. SAM 3) searches for, e.g. `mouse`. Ignored by closed-set local models, which detect the classes they were trained on.
{: #text-prompt }

**Class name**: the label stored on each detection, how it appears in the outputs. Often the same word as the text prompt.
{: #class-name }

**Tracking**: links per-frame detections into per-subject tracks over time.

**Field of view**: a labeled region of the frame that bounds detection; tracking ignores subjects whose center falls outside it. It can move over time (keyframes) and carries over between recordings off the same rig. Stored under the kind `arena`, which is what the older CLI options and on-disk paths still call it.

**ROI (region of interest)**: a labeled polygon in the frame; location export measures each subject's position relative to it (inside / outside, distance).

**POI (point of interest)**: a labeled point in the frame; location export measures each subject's distance to it.

**Embedding**: computes a feature vector per subject per frame (a model such as DINOv3) for similarity and analysis.

**Use every N frames**: how many frames a run steps by. 4 uses frames 0, 4, 8, … Set per run on detection, tracking and embedding, and shown beside the rate it lands at. Counted in source frames, so a downstream stage can only go coarser by a multiple of it.

**Analysis**: deriving results from tracks and embeddings, an **experiment-level** run over a selection of recordings (location export, behavior scan, pose and behavior clustering), separate from the per-recording stages.

**Location export**: an analysis that writes each subject's per-frame spatial features relative to the recording's ROIs/POIs (`tracks.csv` / `roi_relations.csv` / `point_relations.csv`), keyed by `t_ns`.

**Snapshot**: a frozen copy of an analysis run's results, saved automatically whenever a number leaves the run (an export, a comparison, pressing Finish). An analysis you are still interpreting keeps changing; a snapshot does not, so a figure you produced last month stays reachable exactly as it was. Read one back from the run's page or with `analysis snapshots`.
{: #snapshot }

**Run**: one execution of a stage, recorded with its config and outputs. Per-recording stage runs are indexed in the recording (e.g. `det_01`); experiment-level runs live at the experiment.

**Processing run**: an experiment-level batch that applies one processing stage with one config across a selection of recordings (id `run_NNN`); it fans out to a per-recording run in each.

**Analysis run**: an experiment-level analysis over a selection of recordings (id `an_NN`), written under `analysis/`.

**Stream**: a time-series aligned to video time (e.g. MED-PC operant data), keyed by `t_ns` so it joins the pipeline outputs. Each stream declares a **role** (what it is, e.g. `behavior`) and a **shape** (event / interval / continuous).

**Behavior labels**: human-scored ground truth of which behavior occurs over which frames, imported as a `behavior`-role interval stream (from a plain interval CSV or a BORIS export). Each behavior is a **point** (an instant) or a **state** (a duration), and the annotation records the behaviors it **covered** (the scored vocabulary), so a recording is judged only on behaviors it scored.

**Behavior-query eval**: scores how well the behavior query retrieves a labeled behavior against the ground-truth streams. Held-out example clips seed the query and the matches are scored against the remaining instances, reporting **average precision (AP)** and precision/recall. **Folds** cross-validate for a seed-independent number (rotate exemplar sets, or leave-one-out). Used to compare embedding configurations and query settings, scriptably (`analysis query-eval`) or as a recorded GUI run with an inline report.

**Job**: the executor unit (a subprocess) that carries out a run; distinct from the run record it writes.

**`t_ns`**: a timestamp in nanoseconds from the start of the first video part; the universal time reference.

**Track**: one subject's trajectory (identified by a track id) through a recording.

**Fusion**: how several example clips combine into one score in a behavior query: `min` (match any one) or `mean` (resemble all).

**Band**: the warp tolerance, in time, allowed when matching a clip; `0` is a rigid, exact-length match.

**Threshold**: a distance cutoff for accepting a match; lower is stricter.
