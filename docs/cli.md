# CLI guide

Drive the full pipeline from the terminal: headless machines, batch jobs,
scripting. It mirrors the Python API. New to experiment / recording / run / track?
Read [concepts](concepts.md) first.

> Under active development. Commands track the current `main`. The
> always-current reference for any command is its `--help`:
> `detail --help`, `detail <group> --help`,
> `detail <group> <command> --help`.

The entry point is `detail` (installed with the package) or, equivalently,
`python -m detail`.

An ordinary input mistake, such as a YAML file with a missing bracket, a CSV that is not saved
as UTF-8 or a file that is not the kind a command reads, stops the command with one
`Error:` line that names the file (and the line or row) and says what to change. So does a
mistyped id (a run, a comparison or a tracking run that does not exist), and a folder or disk
that cannot take the write: one you do not have permission to write to, a read-only disk, a
name too long for the disk or with a character it refuses (Windows refuses any of
`< > : " | ? *`), or a full disk. A command that writes one file you name (`--out`,
`--csv`, `--export`) creates a missing folder in that path. A number option refuses `nan` and
`inf`.

A CSV you hand in (metadata, spatial setup, a stream, a clip list) must be saved as
**CSV UTF-8** and **comma-separated**: in Excel, Save As → CSV UTF-8 (Comma delimited). A
file saved with semicolons (the default in some European versions of Excel) or tabs is
refused with a message saying so. A row whose cells are all empty is skipped, and so is an
empty column at the end with no header.

## Contents

- [Setup: credentials & models](#setup-credentials--models)
- [End-to-end walkthrough](#end-to-end-walkthrough)
- [Command reference](#command-reference)
- [Getting your data out](#getting-your-data-out)
- [Tracking: choosing a tracker](#tracking)
- [Compute & multi-GPU](#compute--multi-gpu)
- [Inspecting and troubleshooting runs](#inspecting-and-troubleshooting-runs)
- [Workflows](#workflows)
- [Not in the CLI](#not-in-the-cli)

## Setup: credentials & models

The model-free stage (tracking) and any trained/imported detection
model need no setup; skip to the walkthrough. The **gated** models (SAM 3
detection, DINOv3 embedding) need a Hugging Face token plus access granted to each
repo (see the [Hugging Face access](getting-started.md#hugging-face-access)).

```bash
detail hf-token hf_xxx                 # store the token (machine-local)
detail model list                      # what's installed (no network)
detail model check sam3                # why a model can't be installed
detail model install sam3              # download weights into the global cache
detail model install dinov3_vit                     # the default embedding model (vit_large)
detail model install dinov3_vit --variant vit_base
```

Install every model you plan to run, and every **size** you plan to run. Each
variant is a separate download, so installing the default does not cover the
others; pick the one your run is configured with. `model list` names each:

```
  sam3                                installed      role=detection  arch=sam  source=built-in  gated
  dinov3_convnext_base/convnext_base  not installed  role=embedding  arch=convnext  source=built-in  gated
  dinov3_vit/vit_base                 not installed  role=embedding · patch features  arch=vit  source=built-in  gated
  dinov3_vit/vit_large                installed      role=embedding · patch features  arch=vit  source=built-in  gated
```

A model whose weights serve more than one purpose lists everything they are read for
(the ViT above embeds frames and also provides the patch features the keypoint stages
match against); it is one download and one row.

**A run never downloads weights**: it fails immediately if they are missing, so
`model list` is the pre-flight check.

Once a model is installed, nothing on the run path contacts Hugging Face; runs
work offline.

`model check` explains a model you cannot install yet:

```
sam3: ready=False reason=missing_dependency
  message: SAM 3 needs its backend package (sam3): clone the repo and `pip install -e ../sam3` (see the SAM 3 backend setup).
  action : https://github.com/facebookresearch/sam3
```

A ready model prints `sam3: ready=True reason=ok`. `reason` is `ok` (installed and runnable),
`not_installed` (fetch it with `model install`), `missing_dependency` (a Python package the
model needs is not installed), `missing_weights` (a trained or imported model's weights file
is gone), or one of `no_token` / `no_access` / `bad_token` / `offline`. Each
tells you which setup step to revisit, with a message and an action URL. Add
`--variant <name>` to check one size; without it you get the model's default.
Models are machine-global, so the `model` commands and `hf-token` need no experiment
and write nothing outside the model cache and your credentials. `model check` and
`model install` still accept `--experiment` so older scripts keep working, and ignore it.

Older versions cached weights under `~/.cache/huggingface/hub`. If you have a copy
there, move it into the global cache or just re-install:

```bash
mv ~/.cache/huggingface/hub/models--facebook--dinov3-* ~/.cache/detail/models/
detail model list                      # confirm: installed
```

## End-to-end walkthrough

A complete single-recording run. Detection uses **SAM 3** (gated, GPU); set up
[Hugging Face access](getting-started.md#hugging-face-access) and
`detail model install sam3` first. Set a variable for the experiment path first:

```bash
PROJ=~/Documents/DetailExperiments/study        # inside the managed workspace → also visible in the GUI

# 1. experiment + recording
detail experiment create study --dir ~/Documents/DetailExperiments
#   created experiment: /Users/you/Documents/DetailExperiments/study
detail recording create --experiment $PROJ --name cage_a
#   created recording: cage_a  (id recording_000)

# 2. import a video part
detail recording add-video --experiment $PROJ --recording cage_a --video clip.mp4
#   added video to cage_a (copied): clip.mp4

# 3. detection: text-prompted. --every-n-frames 2 halves the cost by detecting
#    frames 0, 2, 4, … (omit it to detect every frame)
detail detection run --experiment $PROJ --recording cage_a \
    --model sam3 --class mouse --prompt mouse --conf 0.1 --every-n-frames 2 --backend cuda
#   submitted detection: det_01
#   detection: det_01 (status=complete)

# 4. tracking: link detections into the subject track (track 0)
detail tracking run --experiment $PROJ --recording cage_a \
    --detection det_01 --tracker anchor_tracker --conf-threshold 0.1 --secondary-conf 0.05
#   submitted tracking: track_01
#   tracking: track_01 (status=complete)

# 5. embedding: embed the tracked subject each frame (DINOv3, gated)
detail embedding run --experiment $PROJ --recording cage_a \
    --tracking track_01 --tracks 0 --backend cpu   # default model: dinov3_vit, vit_large
#   submitted embedding: emb_01
#   embedding: emb_01 (status=complete)

# 6. inspect
detail recording status --experiment $PROJ --recording cage_a
```

Each command blocks until its run finishes, streaming progress lines, then
prints the final status. Run ids are assigned per recording with a per-stage
prefix and counter: `det_01`, `track_01`, `emb_01` (a second run of the
same stage is `…_02`). Downstream commands name the upstream run by id
(`--detection det_01`, `--tracking track_01`), or, for a
**batch** whose recordings carry different ids for equivalent runs, by *kind* with the
`process`/`analysis` `--*-ref <RECORDING:run_id>` selectors.

> **Real subjects, real prompts.** Detection only writes rows where it finds the
> target. With zero detections the detection run still completes, but tracking fails
> with `zero tracks produced: the detection run found no detections`. Use a matching
> prompt and lower `--conf` if needed.

## Command reference

Options below are the common ones; run `detail <group> <command> --help` for
the complete, current list. A path you pass may start with `~` (your home folder), quoted or
not. Every stage command takes `--experiment` and
`--recording`. For every run parameter's bounds, default, and description in one
table, see the [Parameters reference](reference/parameters.md).

### `experiment`
- `create <name> --dir <parent>`: create `<parent>/<name>/`. Use
  `--dir ~/Documents/DetailExperiments` to put it in the GUI's managed workspace.
- `status --experiment <dir>`: identity + recording list + per-recording run counts. A
  recording whose `recording.yaml` cannot be read (a hand edit, a sync conflict) is listed
  as damaged, with the file and line, and the rest are still shown.
- `import-metadata --experiment <dir> --csv <file> [--key <column>] [--dry-run]`: bulk-set
  recording grouping variables from a CSV (see [Recording information](#recording-information)).
- `export-metadata --experiment <dir> --csv <file>`: write the current metadata to a
  fill-in CSV template (`recording_id`, `name`, `file`, then every column, one row per
  recording).
- `export` / `import`: move an experiment between machines
  (see [Moving experiments](#moving-experiments-between-machines)).
- `propagate-fov --experiment <dir> --source <id> [--only <id> ...]`: copy `--source`'s
  field of view onto every other recording that has none (or just the ones named with
  `--only`, repeatable). **Never replaces**: a recording that already has a field of
  view keeps it and is reported, naming the tracking runs holding it if any do. Delete it
  there first to change it.
- `propagate-calibration --experiment <dir> --source <id> [--only <id> ...]`: same, for
  the distance calibration. It only lands on recordings whose video is the same shape
  (aspect ratio) as the source's. A different shape means a different rig, which needs its
  own calibration; those recordings are reported as skipped. So calibrate one recording per
  rig and propagate from each.
- `propagate-rois` / `propagate-points --experiment <dir> --source <id> [--only <id>
  ...]`: same, for regions/points of interest.

  Two things about a definition do not survive the trip between recordings, because a
  frame index means nothing on a different one:

  - **Camera motion is dropped.** A polygon with several keyframes is copied at its
    frame-0 position, and the command says which ones that happened to.
  - **A region with an active frame window is not copied at all**, and is named. That
    window describes one recording's timeline; keeping it can leave a region that applies
    to no frame, and dropping it would silently widen your claim to the whole video.

  Each definition a recording refuses gets its own line, naming the recording:
  `not copied to <recording>: <definition>: <reason>`. A recording that took nothing is
  also listed as `skipped <recording>: <reason>`.

  The GUI's **Copy to other recordings…** button (beside any of the four setup sections
  on a recording's page) does all four from one dialog, showing which recordings will keep
  what they have before you commit.

### `recording`
- `create --experiment <dir> --name <name>`: e.g. `--name cage_a`. Makes an empty recording
  with that name; the app assigns its id (`recording_000`, …) and prints both. A name already
  taken is refused (see [Naming recordings](#naming-recordings)). The other `recording`
  commands take it back as `--recording <name or id>`.
- `add-video --experiment <dir> --recording <id> --video <file>`: import a video as
  the next part. Repeat to add more parts. The video is imported at its original rate
  and resolution: a stage that wants fewer frames sets `--every-n-frames` on the run, which
  needs no second copy. Add `--reference` to **reference the video in place**: the file is not
  copied; the recording stores its location plus a content fingerprint so a moved or
  changed file is detected. The location (the file's folder) is remembered
  automatically; `--video-root <alias>` pins a specific registered [video
  root](#video-root) instead (advanced). Also available in the GUI (a "Reference in
  place" checkbox). After the import it builds the part's [preview](#cache) with its own
  bar; add `--no-preview` to skip that and build it later with `cache previews build`. If
  the preview fails, the video is still added: the command prints what failed and a
  warning with the `cache previews build` command to run, and exits 0.
- `verify --experiment <dir> --recording <id>`: check every video part is present and (for
  referenced parts) unchanged since it was added: a moved/renamed file reports missing
  (relink it), a content change reports changed (remove and re-add). Exits non-zero if
  anything is wrong. A recording with no video yet says so and exits 0. Runs are gated on the
  same check, so a bad referenced video fails the run before it starts.
- `relink --experiment <dir> --recording <id> --part <N> [--video <file>]`: point a part at
  its video file.
    - A **referenced** part: point it at its moved or renamed file. Accepted only if the file's
      content matches what was recorded (a mismatch is a changed video → remove and re-add). For
      a whole moved library, re-point the [video root](#video-root) instead. On an experiment
      made on another computer, relinking one file registers its video root here, so every
      recording under that root finds its video too.
    - A **copied** part whose video is not on this computer (an experiment
      [imported without its video](#moving-experiments-between-machines)): the file is copied
      back in. It is accepted only if its frame times match the recorded part's exactly (the
      same number of frames, each at the same time) and its resolution matches; the error says
      what differs. A part copied in by this version also has to have the same content. Without
      `--video` it uses the file the part was copied from, when that is still on this computer.
      It shows a bar while it checks and copies. A copied part whose video is here is refused:
      there is nothing to put back.

    ```
    detail recording relink --experiment study_copy --recording m1 --part 0
    #   using the file it was copied from: /Users/you/raw/m1.mp4
    #   put back m1 part 0: copied in from /Users/you/raw/m1.mp4
    ```
- `add-stream …`: import an aligned time-series
  (see [Streams](#importing-event-streams-med-pc)).
- `import --experiment <dir> --bundle <folder> [--dry-run]`: bulk-import many
  recordings from a manifest (see [Bulk import](#bulk-importing-recordings)).
- `export-spatial --experiment <dir> --recording <id> --csv <file>`: write the
  recording's whole spatial setup (field of view, distance calibration, ROIs + POIs) to
  one round-trippable CSV, to share or import elsewhere. Each import below takes its own
  part of that file.
- `import-spatial --experiment <dir> --recording <id> (--from-recording <id> | --csv <file>)`:
  take ROIs + POIs onto the recording from another recording in the same experiment or
  from an exported CSV. Coordinates in the CSV are fractions of the frame (0 to 1), not
  pixels. A name the recording already has is **refused by name** and
  reported; the ones that do not collide still arrive (regions and points are sets, so one
  collision does not refuse the rest). To change one, delete it here first: it is never
  overwritten, and no name is silently suffixed. A file with an invalid row (a point outside
  the frame) imports nothing, so you can fix the file and run the command again.

  A CSV you edited by hand is read as strictly as the tool writes it, and each mistake is
  refused naming the row: an active window with only one of `active_start` / `active_end`
  filled (both blank means the whole video), a definition whose rows disagree on its window,
  a name with a space at the start or end, the same `(kind, name, frame_idx, vertex_index)`
  on two rows, vertex indexes that skip one (`0, 1, 5`), and a column named twice. A name is
  stored without spaces at either end, however it is typed, so a setup you export reads back
  under the same names.

  ```
  $ detail recording import-spatial --experiment exp --recording r2 --from-recording r1
  imported 1/2 region(s)/point(s)
    camera motion dropped (copied at frame 0): panning zone
    not imported: zone already exists on this recording; rename or delete it here first; nothing is overwritten
  ```
- `import-fov --experiment <dir> --recording <id> (--from-recording <id> | --csv <file>)`:
  carry a field of view onto the recording, recordings off one rig share it, so it need
  only be drawn once. Refused **whole**, not per name: a recording that already has a
  field of view keeps it and the command fails saying so. (A second one is not a richer
  setup: the batch tracking picker intersects arena ids across recordings.) Delete it on
  this recording first, along with any tracking runs holding it, replacing it would
  misdescribe the region those runs actually filtered on. A CSV or recording holding more
  than one field of view is refused too, naming them: keep the one you want and import again.
- `import-calibration --experiment <dir> --recording <id> (--from-recording <id> | --csv <file>)`:
  carry a distance calibration onto the recording. Refused whole like the field of view:
  there is one pixel scale, so a recording that already has one keeps it (`clear-calibration`
  removes it). The calibration keeps the resolution it was measured at, so it stays correct
  on a recording whose video is a different size, but not a different *shape*. Onto a video
  with another aspect ratio (1920x1080 onto 1280x1024, say) it is refused, because one
  pixels-per-mm cannot describe a frame whose horizontal and vertical scales differ.
  Calibrate that recording on its own video instead.

  All three are the *pull* twins of the `experiment propagate-*` commands above and obey
  the same rules, so it makes no difference which end you start from: **nothing is ever
  replaced**, camera motion is dropped (the polygon arrives at its frame-0 position), and
  a region carrying an active frame window does not travel at all. Only the refusal
  *grain* differs, per name for regions and points, whole for the field of view and the
  calibration. Exporting to a CSV is not a way round any of it; the rules are applied at
  the recording receiving the setup, not at the source.

- `set-calibration --experiment <dir> --recording <id> --pixels-per-mm <value>`: set the
  recording's distance calibration from a pixels-per-mm scalar (the interactive two-point
  pick is GUI-only). `clear-calibration` removes it.
- `status --experiment <dir> --recording <id>`: name and id, the source video files, video
  parts, resolution/fps, spatial definitions (arenas / ROIs / points), the distance
  calibration, streams, and the recording's detection, tracking and embedding runs. A
  calibration on a recording with no video yet is shown at the resolution it was measured on
  (`2 px/mm at 1280x720 (no video yet)`). Keypoint runs are not listed; `process list` and
  `process status` show them:

  ```
  recording: recording_001  (id recording_001)
    source: /Users/you/videos/clip.mp4
    video parts: 1
    resolution: 160x120 @ 30.0 fps
    arenas: 0  ·  rois: 0  ·  points: 0
    stream  : cues (event) offset=0.000s
    detection: det_01 status=complete
  ```

### `detection run`
`--model <id>` is any detection model: built-in (`sam3`), experiment-trained
(`model_NNN`), or imported (`model import` / `model list`). Built-ins also need
`--class` (the stored label) + `--prompt` (the search text), optional `--variant`
(if the model has size options); trained/imported models are closed-set and take neither.
Plus `--conf` (default `0.5`): a real filter here: a detection run stores only what
clears it, and changing your mind means another run (unlike `prompt-test`, which keeps
everything and lets you re-threshold the view). Plus `--batch-size <N>` (frames per GPU forward pass;
omit for the model default, `4` for YOLO and `1` for SAM3, bounded by GPU memory),
`--every-n-frames N` (process every Nth source frame, 1, the default, is every frame; 4
detects frames 0, 4, 8, … for roughly a quarter of the cost), and
[compute flags](#compute--multi-gpu). Writes `det_NN`.

`--every-n-frames` counts **source** frames, and rows keep that numbering, so a downstream
stage can only go coarser by a multiple of it: after detection at 2, tracking and embedding
may use 2, 4, 6, … Asking for 3 is refused rather than rounded: those frames were never
processed. `tracking run`, `embedding run` and `keypoints detect` take the same flag; omit it
there to keep every frame the stage above produced.

### `detection prompt-test`
A fast QC loop *before* committing to a full `detection run`. For an **open-vocab**
model (`sam3`), compare candidate prompt phrasings with `--class <label>`
(applied to every candidate) and one or more `--prompt <text>` (repeatable, one
phrasing per candidate). For a **trained/local** model (`--model model_NN`, closed-set),
omit `--class`/`--prompt`; it previews its own classes, one result column each.
Narrow with `--show-class <name>` (repeatable; default all). Common flags: `--n`
(frames to sample, default `8`), `--min-conf` (default `0.5`), optional `--variant`,
`--seed`, and [compute flags](#compute--multi-gpu). Prints a
per-column detection summary; `--save-as <name>` also writes overlay JPGs + a
`test.json` under `detections/prompt_tests/<name>/`, one JPG per frame and prompt, named by
video part and frame (`part_000_frame_000042_prompt_0.jpg`). Creates no run.

The model runs **wide open** and keeps every proposal it found, so `--min-conf` chooses
what the summary counts and what the saved overlays draw, not what runs. Re-run with a
different `--min-conf` to see another threshold; nothing was thrown away. (In the GUI the
same control is a slider on the results, so you move it without re-running.) This is why
there is no `--conf`: on this command a threshold is a way of looking, not a setting.

### `tracking run`
`--detection det_NN`, `--tracker {anchor_tracker,iou_single_tracker}`
(default `anchor_tracker`), the tracker's hyperparameters, and optional `--arena <name>`.
Each hyperparameter has a dedicated flag (e.g. `--iou-threshold`,
`--max-secondary`); `--hyperparam KEY=VALUE` (repeatable) sets *any* knob in the
selected tracker's schema (coerced to its declared type) for parameters
without a dedicated flag. See [Choosing a tracker](#tracking). Writes `track_NN`.

**The knobs belong to the tracker, not to the command.** Each tracker declares its own
set, so a flag one accepts another rejects: `--max-secondary` is `anchor_tracker`'s and
`--iou-threshold` is not a knob `iou_single_tracker` has. Setting one the chosen tracker
does not declare fails by name and lists what it does accept. Check the table for the
tracker you picked.

### `embedding run`
`--tracking track_NN`, then either `--tracks 0` (comma-separated track ids, e.g.
`0,1`) or `--all-tracks` (embed every track in the run; mutually exclusive),
`--model` (default `dinov3_vit`, a DINOv3 ViT, at `--variant vit_large` unless you
pick `vit_small` / `vit_base`; or `dinov3_convnext_base`),
optional `--variant`, `--input-size` (model input resolution in px, e.g. `384`; sets the
model's processor size and the render size together; omit for the model's own default;
resizable models only), `--readout` (feature read-out: DINOv3 ConvNeXt takes `pooler`
(default) / `gap` / `gmp` / `gap_gmp`, DINOv3 ViT adds `cls_patch` = CLS ⊕ patch-mean),
`--resize` (advanced: final render size independent of the model input; omit to follow
`--input-size`), `--batch-size` (inference batch, default `32`), and
`--constructor` (the input-image constructor: `masked_color_centered_contrast` (default),
`masked_color_centered`, `silhouette_centered`, or `full_frame`; see
[Input images](input-images.md) for what each renders), plus
[compute flags](#compute--multi-gpu). Writes `emb_NN`. `--model` takes any registered
embedding model id (see `model list`).

**Egocentric alignment.** `--align rotate` (overhead camera) or `--align mirror` (side
camera) turns each subject image so the animal faces right before the model sees it.
The heading comes from a keypoint track run on the same tracking run (`--keypoint-run kt_NN`)
through a saved **heading definition** (`--heading NAME`): the skeleton points whose mean is
the front and the points whose mean is the back, so one missing point moves the mean a little
instead of losing the heading. Save one with `experiment save-heading NAME --front nose,left_ear
--back tail_base`; `list-headings`, `export-heading`, `import-heading` and `delete-heading`
manage them (an imported file holds `name`, `front` and `back`, each a list of points or one
point's name; any other key is refused). A frame with no heading of its own takes one interpolated across a gap no longer
than `--max-gap-s` (default 0.5 s); in a longer gap, or before the first and after the last
heading, the frame is **not embedded** and is counted, so an aligned run never mixes turned
and unturned pictures. `--mirror-hold-s` (default 1 s) keeps a side view from flickering
when the animal faces the camera; `--facing-camera-ratio` (default 0.5) treats a heading axis
shorter than that fraction of the track's typical length as undefined. Alignment needs a
cropped, true-scale input image (`full_frame` is refused), its canvas is sized from the
turned masks, and the mode, the definition and the three settings are part of the run's
identity: an aligned run never compares with an unaligned one. The run's parts gain
`heading_deg`, `heading_source` (0 own, 1 interpolated) and `align_mirrored` per frame, and
`embedding.yaml` records the heading coverage per track and part.


### `keypoints`
Keypoints are placed in two stages. Keypoint detect places them on every animal a detection run
found, frame by frame, and stores each pose with no subject. Keypoint track then gives each pose
to an animal of a tracking run on that same detection run, and settles the points over time. First
what a run learns from, **example sets**, each a skeleton plus a few frames marked on it (see
[Keypoints](concepts.md#keypoints)); then the detect and track commands, and distilling a
finished run into a fast student.

**Example sets.** Create a set on a library skeleton (`detail skeleton list`, below), another
set's skeleton or a skeleton YAML (`names`, `edges`, `pairs`), sample tracked frames into it, add
one chosen frame, show it, delete it:

```bash
detail keypoints examples list --experiment my_experiment
detail keypoints examples create "overhead rats" --skeleton quadruped_overhead_12 --experiment my_experiment
detail keypoints examples sample ex_01 --recording cage_a --recording cage_b -n 5 --experiment my_experiment
detail keypoints examples sample ex_01 --recording cage_a --strategy spread -n 3 --experiment my_experiment
detail keypoints examples add-frame ex_01 --recording cage_a --tracking-run track_01 --part part_001 --frame 1830 --track 0 --experiment my_experiment
detail keypoints examples show ex_01 --experiment my_experiment
detail keypoints examples delete ex_01 --experiment my_experiment
```

Rename a set, change its skeleton (only until its first mark; after that a different skeleton
is a new set), make a new set with the same skeleton, or write a set to a folder another
experiment can import:

```bash
detail keypoints examples edit ex_01 --name "side rats" --experiment my_experiment
detail keypoints examples edit ex_01 --skeleton quadruped_side_5 --experiment my_experiment
detail keypoints examples duplicate ex_01 --name "side rats 2" --experiment my_experiment
detail keypoints examples create "copied" --skeleton ex_01 --experiment other_experiment
detail keypoints examples export ex_01 --out shared/side_rats --experiment my_experiment
```

`sample` reads each recording's only complete tracking run; a recording with several needs
`--tracking-run`. The default strategy, `varied`, reads the shape of every tracked frame, groups
postures and picks the ones that differ (the typical ones first on an empty set); `spread` takes
one frame per equal time slice. A sample decodes frames, so it shows a progress bar. A set's
frames are copied in as PNGs; the marks are made in the GUI's labeling screen (GUI only, see
*Not in the CLI*), or arrive with a prepared file:

```bash
detail keypoints examples import shared/examples.yaml --experiment my_experiment
```

The file is an `examples.yaml` as the tool writes it (`schema_version` first, the skeleton, then
one entry per frame with its `image` path relative to the file and its `marks`), with the images
beside it; `examples export` writes exactly that. Its marks are checked as the labeling screen
checks a click: a point the skeleton names, at most once a frame, and a visible point at `x`, `y`
between 0 and 1 (a fraction of the frame's width and height, not pixels). A bare skeleton file
imports too, as an empty set. The set keeps its link to a library skeleton only when a skeleton on this machine has the same
points, edges and pairs; otherwise it is listed under *(no library skeleton)*. If an image will not open, the import stops and no set is added. Deleting a set a keypoint run lists is refused, so a run always finds its frames.

**Detect.** One run per selection of recordings:

- `keypoints detect --experiment <dir> --recording <id> [--recording <id> …] --model <method>`
  plus **one** of `--detection <det_NN>` (that exact run id, complete in every selected
  recording) or `--detection-ref <RECORDING:det_NN>` (derive the detection kind from one run
  and resolve each recording's own match). `--model` is a registered method, a keypoint student
  trained in this experiment, or a keypoint model from the machine's catalog (`model list`
  shows them with `role=keypoint`), looked up in that order; the run records which
  (`weights_source`: `trained` or `imported`). A method that works on each animal (`dinov3_match`)
  crops around every detection on the frame; a trained student looks at the whole frame. The
  method `dinov3_match` learns from an example set, so it needs `--example-set <ex_NN>`; its
  settings (`--decode`, `--side-rule`, `--crop-px`) can be left out, in which case they are
  resolved against the set's skeleton and recorded on the run. A trained student scores each
  whole pose, and `--min-pose-confidence` (default 0.25) is the lowest score it keeps; a pose
  below it is not stored, so changing it means running again. The patch method has no such
  score and refuses the option. `--every-n-frames` must be a multiple of the detection run's;
  `--batch-size` only changes speed. Writes `run_NNN`, and in each recording a `kd_NN` run
  under `keypoint_detect/` holding, per video part, one row per pose: its frame, the box it was
  placed in, the points, their scores and the method's candidates. No pose belongs to an
  animal yet. `process keypoint_detect` is the same command.

**Track.** Give each pose to a tracked animal, then settle the points over time, on the CPU, in
seconds:

- `keypoints track --experiment <dir> --recording <id> [--recording <id> …]` plus **one** of
  `--detect <kd_NN>` or `--detect-ref <RECORDING:kd_NN>`. `--tracking <track_NN>` names the
  tracking run whose animals get the poses, or `--tracking-ref <RECORDING:track_NN>` derives
  the tracking kind from one run and resolves each recording's own match. Left out, it is the
  one complete tracking run on the detect run's detection run, and when there are several the
  command refuses and names them. On each frame, poses and animals are paired by the share of a pose's points that fall
  inside the animal's mask: each animal gets at most one pose, each pose at most one animal,
  and a pair counts only if that share is at least `--match-min-inside` (default 0.5). The
  mask is grown first by `--match-mask-grow` times the animal's body size (default 0.05), so a
  point just outside the outline still counts. A pose no animal gets stays unassigned. Both
  defaults are choices, not measured values.
  Then, optionally, `--method temporal_decode|despike_smooth` with `--method-param
  NAME=VALUE` for its settings (names and defaults on the method's parameter page, `detail
  docs`). Without `--method`, the detect run's method chooses: `dinov3_match` settles its
  stored candidates along a path through time and its left/right pairs over frames
  (`temporal_decode`); a trained student despikes and smooths its points (`despike_smooth`).
  Every window is in seconds and every setting, the tracking run and both matching numbers
  included, is part of the run's identity. Writes `run_NNN` and a `kt_NN` run under
  `keypoint_track/` holding which animal each pose went to, and per track and part the final
  point, its score and how it came to be (as detected, another candidate, moved by a filter).
  `process keypoint_track` is the same command. Re-running with other settings, or against
  another tracking run, is a new `kt_NN` over the same detect run; the detect run is never
  rewritten.

**Distill.** Turn a teacher into a fast YOLO-pose **student**, a model that places the same
skeleton on its own, with no example set and on any backend, in one durable run of three stages:
label, train, register. `--teacher` says what labels the frames, and both teachers give their
placements unsmoothed, frame by frame. `patch` runs the patch method itself (`dinov3_match`) on
the sampled frames, from an example set and the tracking runs whose boxes it is given, so no
keypoint detect run is needed first. `detect` reads finished keypoint detect runs: their stored
poses are given to the animals of a tracking run on the same detection run, the way `keypoints
track` gives them, and become the labels, so nothing runs on the accelerator while labelling. Either way each
label is one YOLO-pose line per tracked animal (the tracking box, then each point with its
visibility), frames are drawn spread over time where every animal is tracked, the example set's
own clicks join them where the marked animal was alone on the frame, and audit frames, and every
frame within `--exclude-audit-s` of one, are never used. The dataset also keeps what the teacher
gave on each frame it used (`teacher.npz`: points, scores and the other candidates), so a
different cut-off can be applied later without running the teacher again.

**Test first.** `keypoints test` takes the same teacher options and runs the teacher on about a
dozen frames drawn the way the distill would draw them, writing nothing. It prints how many
placed points a cut-off keeps and, for the patch method, a **held-out check**: each example
frame is placed by the others and compared with your own clicks, as a median error in pixels and
as a share of body size, above and below the cut-off, with counts (a median from fewer than 8
placements is not shown). Use it to choose `--score-cutoff`; `--seed` draws another sample.

```bash
detail keypoints test --teacher patch --example-set ex_01 --recording cage_a --cutoff 0.4 --label-backend cuda --experiment my_experiment
detail keypoints distill --teacher patch --example-set ex_01 --recording cage_a --recording cage_b --name fast_rats --n-frames 300 --score-cutoff 0.4 --label-backend cuda --experiment my_experiment
detail keypoints distill --teacher detect --recording cage_a --recording cage_b --name fast_rats --n-frames 300 --score-cutoff 0.5 --experiment my_experiment
detail keypoints distill --teacher detect --detect-run cage_a:kd_02 --name fast_rats --n-frames 300 --epochs 60 --train-backend cuda --experiment my_experiment
detail distill status distill_001 --experiment my_experiment
```

- `keypoints test --experiment <dir> --teacher patch|detect` plus the teacher options below,
  `--every-n-frames` and `--exclude-audit-s` (draw as the distill will), `--n` (frames, default
  12), `--seed` (default 0) and `--cutoff` (default the method's own, if it has one). With
  `--teacher patch`, `--decode`, `--side-rule` and `--crop-px` set the method the way
  `keypoints detect` takes them, so a test before a detect run places points as the run will
  (left out, the method decides, as on a run). Shows a progress bar through loading, labelling
  and the held-out check.
- `keypoints distill --experiment <dir> --name <name> --n-frames N --teacher patch|detect`.
  For `patch`: `--example-set ex_NN`, and the recordings to label as `--recording <name>`
  (repeatable; the recording's only complete tracking run) or `--tracking-run
  RECORDING:track_NN` (a named run, for a recording with several); `--model` (default
  `dinov3_match`), `--model-variant`, `--label-backend` / `--label-device` (where it runs; a
  real number of frames wants CUDA), and `--decode` / `--side-rule` / `--crop-px` as
  `keypoints test` takes them, so the distill labels with the settings you tested the cut-off
  on (left out, the method's own; what was used is recorded on the dataset). The patch method
  has no default cut-off, so
  `--score-cutoff` is required. For `detect`: `--recording <name>` (repeatable; the recording's
  only complete keypoint detect run) or `--detect-run RECORDING:kd_NN` (a named run, for a
  recording with several), and optionally `--tracking-run RECORDING:track_NN` for the
  tracking run whose animals get the poses (left out, the one complete tracking run on the
  detect run's detection run; with several, the command refuses and names them). A mix of the two teachers' options is refused. Label options come from the
  dataset's parameter page (`detail docs`): `--val-split`, `--score-cutoff` (a point placed below it
  is taught as not visible; omit for the teacher method's own default, which is recorded on the
  dataset, and a method with no default asks for one), `--exclude-audit-s`, `--every-n-frames`.
  Training options mirror `train run`: `--base-model` (a `-pose` checkpoint, default
  `yolo11s-pose.pt`; any other is refused), `--trainer`, `--epochs` / `--batch-size` / `--imgsz` /
  `--patience`, `--train-backend` / `--train-device`. One bar folds `labeling → training →
  registering`; Ctrl-C detaches and `distill attach` rejoins. Writes `distill_NNN` with its
  `dataset_NNN` (task `pose`) and `model_NNN`, which the register stage names `<name> distill_NNN`
  and files under `role: keypoints`, so `keypoints detect --model <model_NNN>` can run it.
- `dataset keypoints` builds the dataset alone, with the same teacher and label options (see
  [`dataset`](#dataset)). `distill status` prints the register stage for a keypoint distill;
  `attach`, `resume` (a run that failed at register keeps its trained student and only registers
  it), `cancel` and `delete` work as for any distill run.

**Export.** A finished run's points, for other tools:

- `keypoints export --experiment <dir> --run <run_NNN> --out <dir>` plus `--target` repeated
  from `keypoints`, `keypoints_dlc`, `keypoints_sleap`, `keypoints_nwb`, `keypoints_video`
  (default: `keypoints`), and optionally `--cutoff <score>`, `--blank-below-cutoff`,
  `--source detect` and `--overlay`. The same targets are on `export run`; the formats and the
  options are described under [Keypoints, for pose tools](#keypoints-for-pose-tools).


### `skeleton`
The machine's skeleton library, the CLI twin of the GUI's Keypoints manager. It needs no
`--experiment`: skeletons are shared by every experiment on the machine. The four presets are
listed first and cannot change.

```bash
detail skeleton list
detail skeleton show sk_01
detail skeleton create "rat overhead" --from quadruped_overhead_12
detail skeleton edit sk_01 --name "rat, top view"
detail skeleton edit sk_01 --skeleton my_rats.yaml
detail skeleton save-as-new sk_01 "rat overhead v2" --skeleton my_rats.yaml
detail skeleton duplicate quadruped_side_5 --name "side rats"
detail skeleton export sk_01 shared/rat_overhead.yaml
detail skeleton import shared/rat_overhead.yaml
detail skeleton delete sk_01
```

`show` lists what uses a skeleton: the example sets, keypoint runs, datasets and trained models in
your experiments that copied it. A copy counts when it holds the same points, edges and pairs,
whatever id it was given on the machine where it was made. Only the experiments in the workspace
are searched, not one created with `experiment create --dir` somewhere else. While anything does, `edit --skeleton` and `delete` are refused
with that list; `save-as-new` keeps the edit as a new skeleton instead. A rename is always allowed.

A skeleton file is `names` (a list, even of one point), and optionally `edges` and `pairs`, each a
list of two-name lists (`[[nose, neck]]`); an exported file holds that under `skeleton:` beside its
`name`. A list written as one word (`names: nose`), an edge that is not two names, a key it does not
take (`edgse:`) or a key written twice is refused, naming it.

### `process`
Experiment-level processing runs span a **recording selection** and record a `run_NNN`,
the unit a dataset is built from. (The recording-scoped `detection run` /
`tracking run` above do **not** create one.) `--recording` is repeatable.

Every selected recording is checked before the run starts. A bar names each one as it goes
(`Validating`, after `Resolving shared settings` for an embedding run that reads each recording
for its background colour), then the run's own progress follows.

- `process detection --experiment <dir> --recording <id> [--recording <id> …]` plus the
  same detection options as `detection run` (`--model` for any source; `--class` /
  `--prompt` only for a built-in model; `--conf`, `--variant`, `--every-n-frames`) and
  [compute flags](#compute--multi-gpu). Writes `run_NNN`.
- `process tracking --experiment <dir> --recording <id> [--recording <id> …]` plus the
  tracker options, and **one** of:
  - `--detection <det_NN>`: that exact run id, complete in **every** selected recording.
  - `--detection-ref <RECORDING:det_NN>`: derive the detection *kind* from one named run
    and resolve each recording's own matching detection run. Use this when the recordings'
    detection runs have different ids (e.g. run in separate batches). The kind is the
    run's **full lineage**: model + variant + confidence + prompt + cadence. A
    recording whose detection differs in any of those won't match (and the run fails; name
    a run every selected recording shares).
  Writes `run_NNN`.
- `process embedding --experiment <dir> --recording <id> [--recording <id> …]` plus the same
  embedding options as `embedding run` (`--model`, `--variant`, `--constructor`,
  `--resize`, `--batch-size`, `--tracks` / `--all-tracks`, and the alignment options
  `--align`, `--heading`, `--max-gap-s`, `--mirror-hold-s`, `--facing-camera-ratio`), and
  **one** of `--tracking <track_NN>` (that exact run id, complete in **every** selected
  recording) or `--tracking-ref <RECORDING:track_NN>` (derive the tracking *kind* from one
  run and resolve each recording's own match, for recordings with divergent ids). An aligned
  run names its keypoint track run the same two ways: `--keypoint-run <kt_NN>` or
  `--keypoint-ref <RECORDING:kt_NN>`. Writes `run_NNN`.

  `--canvas-scope` controls how the `*_centered` constructors size their canvas across the selected recordings. `per_recording` (default) sizes the canvas to each recording's own largest mask, so every subject fills the frame and body-size differences between animals are normalized away. `shared` sizes it once to the largest mask over the whole selection, keeping true relative size, but a smaller subject then fills less of the model input.

  It is a no-op on a single recording. Runs made at different scopes are **not**
  interchangeable: a later `query` or `query-eval`
  requires every recording to share the scope (a `per_recording` canvas is allowed to
  differ between recordings; a `shared` one is not).

  The max-contrast background has no flag: one `process embedding` over several recordings
  always picks **one** color for all of them, pooled from every selected recording's
  subject, so the background can't become a cue that identifies the recording. Pass every
  `--recording` you mean to compare in one command. Recordings embedded separately can get
  different colors, and a later `query` or `query-eval` across them refuses them as
  not comparable.
- `process list --experiment <dir>`: the experiment's processing runs.
- `process status <run_NNN> --experiment <dir>`: its module, status, and the
  per-recording run ids it created.
- `process attach <run_NNN> --experiment <dir>`: stream a live run's progress (it keeps
  running with the CLI closed). Ctrl-C detaches; it does not cancel.
- `process retry <run_NNN> --experiment <dir>`: re-run the run's **incomplete** recordings
  into the same record. Every recording that isn't complete (failed, cancelled, or never
  launched) runs again under a fresh per-recording id; completed ones are left alone, and
  the run rolls up to `complete` once all of them are. A bar names each recording as it is
  re-checked; the command refuses if one no longer validates, naming each. A recording is re-run whole (there is no
  resume-from-mid-recording).
- `process cancel <run_NNN> --experiment <dir>`: stop a running run's coordinator and its
  in-flight recordings. Finished recordings keep their results, and recordings the run had not
  reached yet are listed as `cancelled`. The run reads `cancelled` unless every recording had
  already finished; `process retry` picks up the rest.
- Naming the same recording twice in one `process` command (by its name and by its id, say)
  runs it once.

`detection`, `tracking`, and `embedding` have `process` subcommands.

### `dataset`
- `create --experiment <dir> --name <name> --source-run <run_NNN>`: export a YOLO
  dataset from a processing run (detection or tracking). Optional `--every-n N`
  (keep every Nth frame; default all), `--val-split` (default `0.15`),
  `--min-contour-area` (px, default `100`). No new inference; it reuses the run's
  stored masks. Writes `dataset_NNN`.
- `label --experiment <dir> --name <name> --class NAME:PROMPT [--class …] --n-frames N`:
  the second dataset source (no existing run needed). Sample N frames across recordings
  and label them with a prompt-capable model into a YOLO dataset. Options: `--model`
  (default `sam3` (gated)) and `--model-variant` (if the model has size options;
  omit for the model default), `--recording <id>` (repeatable; default every recording),
  `--sampling stratified|uniform` (default `stratified`, which spreads the sampled frames evenly
  across each video's timeline; `uniform` is fully random), `--logit-threshold` (default
  `0.5`), `--min-contour-area` (px, default `100`), `--val-split` (default `0.15`),
  `--background-share` (default `0.1`, at most `0.5`: frames where the model found none of the
  classes are kept with an empty label, up to this share of the dataset, so the detector learns
  what an empty frame looks like; `0` keeps none), plus `--backend` / `--device`. Each sampled
  frame is decoded exactly; one that can't be is replaced by another frame from the same video.
  The validation share is drawn over the frames that made it into the dataset. Writes
  `dataset_NNN`.
- `keypoints --experiment <dir> --name <name> --n-frames N --teacher patch|detect`: the third
  dataset source, a YOLO-pose dataset taught by the patch method or by finished keypoint detect
  runs, the label stage of [`keypoints distill`](#keypoints) on its own. Same teacher and label
  options as there (`--recording`, `--detect-run`, `--tracking-run`, `--example-set`,
  `--val-split`, `--score-cutoff`, `--exclude-audit-s`, `--every-n-frames`). Writes
  `dataset_NNN` with `task: pose` and the teacher's raw output (`teacher.npz`), which `train
  run` trains with a `-pose` base model.
- `list` / `status <dataset_NNN>` / `delete <dataset_NNN>`: inspect and remove
  datasets (`delete` confirms unless `--yes`).
- `attach <dataset_NNN> --experiment <dir>`: stream a live export's progress (it keeps
  running with the CLI closed). Ctrl-C detaches; it does not cancel.

### `train`
- `run --experiment <dir> --name <name> --dataset <dataset_NNN>`: train a custom,
  closed-set detector. `--trainer` (default `ultralytics_yolo`), `--base-model`
  (default `yolo11s-seg.pt`, a seg checkpoint; labels are polygons), `--epochs` /
  `--batch-size` / `--imgsz` / `--patience` (stop when no improvement for N epochs;
  default `100`, `0` disables), `--hyperparam KEY=VALUE`
  (repeatable), and [compute flags](#compute--multi-gpu). Writes `model_NNN`; the
  best weights land in `models/<id>/run/weights/best.pt`.
- `list` / `status <model_NNN>` / `delete <model_NNN>`: inspect and remove trained
  models (`delete` confirms unless `--yes`). `status` prints a `trained with` line: the
  Ultralytics version and the settings asked for. The model's `model.yaml` also keeps every
  setting Ultralytics resolved (`trained_with.args`).
- `attach <model_NNN> --experiment <dir>`: stream a live training run's progress (it keeps
  running with the CLI closed). Ctrl-C detaches; it does not cancel.
- `promote <model_NNN> --experiment <dir> [--name <name>]`: copy a completed trained model into
  the machine-global model catalog so it's selectable in *any* experiment. Records its provenance
  (source experiment, dataset, base model, metrics); the original stays put. A keypoint student
  is promoted with its skeleton, which is also added to the skeleton library unless one with the
  same points, edges and pairs is there; `keypoints detect --model <id>` then finds it in any
  experiment.

### `distill`
One durable run that chains the two steps above, label a frame sample with a prompt-capable
model, then train a YOLO detector on it, so you don't drive `dataset label` → `train run` by
hand. It survives the CLI/app closing; reattach or resume later. The trained model is usable in
this experiment at once; `train promote` it for cross-experiment reuse.

- `run --experiment <dir> --name <name> --class NAME:PROMPT [--class …] --n-frames N`: launch a
  distill run. Labeling options mirror `dataset label` (`--label-model` (default `sam3`),
  `--label-model-variant`, `--sampling`, `--logit-threshold`, `--min-contour-area`, `--val-split`,
  `--background-share`, `--recording <id>` repeatable); training options mirror `train run` (`--base-model`, `--trainer`,
  `--epochs` / `--batch-size` / `--imgsz` / `--patience`). Compute is **per stage**,
  `--label-backend` / `--label-device` (SAM3 needs CUDA) and `--train-backend` / `--train-device`.
  One bar shows the folded `labeling → training` progress with an ETA. Ctrl-C detaches (the run
  keeps going); `distill attach` to rejoin. Writes `distill_NNN` (+ its `dataset_NNN` / `model_NNN`).
- `attach <distill_NNN>` / `resume <distill_NNN>` / `cancel <distill_NNN>`: rejoin a run, resume a
  failed/cancelled one (reuses a complete dataset, resumes a cancelled or crashed training), or stop a running one.
- `list` / `status <distill_NNN>` / `delete <distill_NNN>`: inspect and remove distill runs.
  `delete` removes only the run record: its dataset and trained model stay (delete those with
  `dataset delete` / `train delete`). `status` prints the run's `kind` (`detection`, or
  `keypoints` for one launched by [`keypoints distill`](#keypoints), whose third stage,
  `register`, it prints too).

### `locations export`
`--experiment`, `--recording`, `--tracking track_NN`, `--out <dir>`: a per-recording export
into a new folder inside `--out` (named for the experiment, the recording, the tracking run and
the time, and printed), writing `tracks.csv` / `roi_relations.csv` / `point_relations.csv` + a
`locations.yaml` manifest. Exporting another recording into the same `--out` makes another
folder, so one never replaces the other (per-track position, body length and width, and
position relative to the recording's ROIs/POIs, keyed by `t_ns`). No run record. For a
recorded, batchable version, see `analysis run` below.

Three options choose what is measured and how often. `analysis run` takes the same three:

- `--sample-fps <n>`: rows per second for each subject. Without it, every tracked frame. With
  it, the tracked frame nearest each tick of a clock at that rate, and nothing interpolated.
- `--remove-thin-parts` / `--keep-thin-parts`: measure the body without a tail, limb or
  antenna (the `body_` columns). Off unless you ask; recommended for rats and mice.
- `--thin-part-scale <x>`: how thick a part must be to count as body, as a share of the
  square root of the subject's own mask area. Default 0.12.

`locations.yaml` records all three, and the size of the opening used for each subject.

### `export`
Get data out of DETAIL in open formats. `export list` shows what a subject can produce;
`export run` writes it. See [Getting your data out](#getting-your-data-out).

```bash
detail export list --experiment ./study --analysis-run an_01
detail export run --experiment ./study --analysis-run an_01 --target clips --out ~/exports
```

Options on `export run`: `--target <id>` (repeatable, several targets go into one folder),
`--out <dir>` (where the export folder is created), `--cluster <label>` or repeated
`--instance <id>` to narrow the scope, `--for-recording <id or name>` (repeatable) to export only
those recordings of a run, and `--estimate` to see how big and how long it would be without writing
anything ([below](#how-big-and-how-long)). The subject is whichever of `--recording <id>` /
`--run <run_NNN>` / `--analysis-run <an_NN>` / `--distill-run <distill_NNN>` you pass; pass
none for the experiment itself. `--run` takes the experiment-level processing run id from
`process list`, not a per-recording id such as `track_01`.

A scope is refused, not ignored, when a target you ask for cannot be narrowed that way:
`--target instances --frames 0:3` stops with "Behavior instances cannot be narrowed to a frame
range (only to one cluster, a selection, some recordings)", and nothing is written. `export list`
shows each target's scopes. So is a `--cluster` the run does not have (the error lists its
clusters, with the names you gave them), an `--instance` it does not have, a `--frames` range that
starts past the last frame, and a `--target` given twice. A target that always covers the whole
run, such as `performance`, is written whole beside targets that do narrow, and `export.yaml`
lists it under `whole`; asked for alone with a scope, it is refused.

Figure options, for every target that writes a figure (`instances`, `ethogram`):
`--theme light|dark`. Figures are light, for print, unless you pass `--theme dark` (for slides).
A comparison's figures are exported with `comparison export`, which takes the same `--theme`.

Clip options, for `--target clip` and `--target clips`: `--render raw|overlay|mask`
(repeatable), `--crop` / `--full-frame`, and `--size <px>`. `--target clip` needs
`--frames START:END` (the inclusive frame range to write; without it the export is refused) and
`--embed-run <emb_NN>`, and takes `--track <id>`, which must be a track that embedding run has:

```bash
detail export run --experiment ./study --recording recording_004 --target clip \
    --frames 1200:1340 --embed-run emb_01 --render overlay --out ~/exports
```

### `notes`
Your own notes on one entry, the same text as the **Notes** box on its page in the GUI. Name the
entry by kind and id; `experiment` takes no id. `detail notes show --help` lists the kinds.

```bash
detail notes show --experiment ./study run run_002
detail notes set --experiment ./study run run_002 --text "conf 0.1 picked up the plate edge"
detail notes set --experiment ./study experiment --file notes.txt   # or --file - for stdin
detail notes set --experiment ./study analysis_run an_03 --clear
```

`set` replaces the whole note. Notes live beside the entry as `notes.md`, and an export of the
entry includes them.

### `analysis`
Experiment-level **analysis runs**: recorded, batchable runs over a recording
selection, written to `analysis/<an_NN>/`.

- `run --experiment <dir> --recording <id> [--recording <id> …] [--module location_export]`
  plus **one** of `--tracking <track_NN>` (that id, complete in every selected recording)
  or `--tracking-ref <RECORDING:track_NN>` (derive the tracking kind from one run and
  resolve each recording's own match, for recordings with different run ids; the kind is
  full lineage, incl. the tracking run's own detection kind, and the run fails if any
  selected recording has no match). Repeat `--tracking-ref` to cover recordings tracked
  through different chains (two detection models, say): each recording reads its run of the
  one named kind it has. Runs a
  per-recording analysis module across the selection (writes `an_NN`). For
  `location_export` (the default), output lands in `analysis/<an_NN>/<recording_id>/` as
  the three CSVs. Same output as `locations export`, but recorded, with progress + batch.
  It takes `--sample-fps`, `--remove-thin-parts` and `--thin-part-scale` as
  [`locations export`](#locations-export) does, and refuses them for any other module.
- `query --experiment <dir> --search <id> [--search <id> …] --embed-run <emb_NN> --method <id> [--method-param <name>=<value> …] [--behavior <name> …] --example <[name=]recording:start-end> [--example … | --from-csv <clips.csv>] [--non-example <[name=]recording:start-end> …] [--track 0] [--split-gap <s>] [--min-match <s>] [--strict]`: search a cohort for behaviors you can name, from a few example clips each. One run searches **several behaviors**: declare names with `--behavior rear --behavior groom` and attach clips with a `name=` prefix (`--example rear=s1:0:12.4-13.1`); a bare `--example` makes one default behavior. Spans are **seconds** by default (`s1:0:12.4-13.1`, `recording[:track]:start-end`) or **frames** with an `f` suffix (`s1:0:1240f-1310f`). `--from-csv` reuses a prior clip CSV's rows as examples, which is also how you reuse a previous run's. Writes one opaque shard per (behavior × recording × subject) under `analysis/<an_NN>/search/`. **There is no threshold flag**: the search stores no cutoff, so choosing one afterwards costs nothing and never a second search. The run therefore finishes **open** rather than complete: it stays open until you close it. Turn the results into behavior instances with `analysis query-calibrate`, below. `--method` defaults to [`soft_knn`](modules/soft_knn.md), which is fast; its page says what is and is not known about how well it retrieves. `--min-match <s>` sets the shortest match the search returns, in seconds (default 0, no minimum). It is applied while matches are formed, so a brief spike cannot crowd out a longer match beside it, and changing it means searching again.

  The method's own parameters are set with `--method-param <name>=<value>`, repeatable,
  `--method-param k=30 --method-param duration_max_mult=3`. There is no generated
  option per parameter because `--method` is chosen at parse time; each method's page
  ([soft_knn](modules/soft_knn.md)) lists what it takes, with bounds and defaults. A name the method does not have is
  refused rather than ignored. `--non-example` marks a clip that is *not* the behavior, in the
  same spec as `--example`; only a method whose negatives are `optional` or `required` can use
  them, and a run that passes them to one that cannot is refused with the method's reason.
  A `--non-example` whose behavior name has no `--example` (a typo, or a bare one beside named
  examples) is refused, naming the behaviors that have examples.
- `query-calibrate --experiment <dir> --run <an_NN> [--curve <behavior> [--steps 21]]
  [--cutoff <behavior>=<confidence> …] [--name <behavior>=<name> …] [--finalize] [--reopen]`:
  set the confidence cutoff that turns a query run's stored search into behavior instances.
  Every operation is instant: it rewrites the run's state and rebuilds the behavior output
  from results already on disk; nothing re-searches.

  `--curve q01` prints how many instances each cutoff returns and exits without editing, which
  is how you choose one: there is no ground truth to score against, so you pick by looking at
  what each setting yields. A point flagged *capped* hit the method's own candidate limit,
  a curve that flattens because storage ran out looks exactly like one that flattens because
  the animal stopped. Instance count is not guaranteed to fall as the cutoff rises.

  A behavior with **no cutoff is absent** from the behavior output, not present with zero
  instances: a named behavior with no rows means "we looked and it never happened", which is a
  different claim from "nobody has chosen a threshold yet". `--cutoff q01=none` clears one and
  removes it again. `--name` renames a behavior; its id never moves, so recorded judgments stay
  attached to what they were made about. `--finalize` is your statement that you are done, not
  a lock: a later edit is refused and told to `--reopen` first.

  A search that did not reach every recording (you cancelled it, so it is `partial`, or one
  behavior's search failed on a recording) can still be calibrated, and its behavior output
  covers only the recordings that were searched: a recording a named behavior never looked at
  is left out, not counted as a zero. `--finalize` and `export-query` refuse such a run and name
  the recordings it is missing; re-run the search to cover them.
- `audit --experiment <dir> --run <an_NN> [--behavior <id or label>] [--cutoff <confidence>]
  [--by-recording] [--by-cluster] [--goal-width <width>] [--own-only]`: read the precision an authored run's reliability audit supports, for a
  behavior scan or a clustering run. Reads only: the judging is a blind review of video clips and
  lives in the GUI (see *Not in the CLI*), but the number it buys is reachable from a script.

  Precision always comes with its 95% confidence interval and its `n`: `1.00` from three clips and `1.00`
  from fifty are the same figure and different claims. Below 8 audited clips the precision
  prints as a dash and only the interval shows; once every clip of a label is audited it prints the
  exact share, `precision 0.83  (all 6 audited)`. The GUI's table reads the same way. A run nobody has audited prints
  `(not audited)` rather than a number, and one whose audit no longer covers enough of the
  run (because the vocabulary was re-cut under it) prints why instead of a rate.
  `--by-recording` splits it per video, which is where one cutoff behaving differently in
  different animals becomes visible.

  On a behavior scan the precision is weighted by confidence bin, because the review draws about
  the same number of clips from every bin however many candidates each holds; it is the same
  figure the GUI's Audit tab shows. Clips overlapping the behavior's own example or non-example
  clips are never drawn or counted.

  On a clustering run, labels nest, so `--behavior` takes a label path (`rearing/supported`), and without it every label prints, a parent above its sub-labels, each with its precision, its 95% confidence interval and the number of clips audited.

  A parent's precision counts every instance under it. A clip is judged at the level that is
  true ("rearing, not supported"), so one judgment scores both levels.

  `--by-cluster` lists, under each label, every cluster it was given to: its size, its share of
  the label, and its precision with its 95% confidence interval. With `--behavior` naming a
  parent, its sub-labels' clusters are listed too. It reports; whether a cluster is
  pure enough is yours to say. `--goal-width <width>` adds each interval's width against that
  goal (0.20, for example, is ±0.10) and says when it is reached, or roughly where
  another 25 clips would put the interval, which is a statement about how many clips were
  audited, not about the label. Without it nothing is said about width. Clusters need the run's
  node cache. The label's own precision weights each cluster by its share, and says
  *provisional: 40% of the label not yet audited* while some cluster has nothing audited. After
  you change which clusters a label holds, its precision counts only clips audited since, and the
  line says *edited after the audit: 3 audited since, 25 before*. Renaming or merging a label
  keeps every answer. `--own-only` reports a parent label's own clusters without its
  sub-labels': the row INTERPRET's labels panel shows as the label's *other*.

  On a **temporal clustering** run, the clip review prints first, cluster by cluster. AUDIT reviews every clip of a cluster, so a reviewed cluster's figure is exact, and a label's figure appears only once every cluster under it (its sub-labels' included) is reviewed; until then its line says how many are, and each reviewed cluster shows its clip count, its share of the label and its precision.

  The earlier one-by-one audit follows under its own heading, read as before and never merged
  into the clip review.
- `export-query --experiment <dir> --run <an_NN> --out <dir>`: write a calibrated query run's
  behaviors as one behavior-stream CSV per searched recording, into a new folder inside `--out`
  that the command prints, so exporting again at another cutoff keeps the first export. Each row carries
  `origin: predicted`, so re-importing one can never be read back as ground truth. A recording
  that matched nothing gets a header-only file: the zero is the finding. A run whose search did
  not reach every recording is refused, naming the ones it missed, so a header-only file never
  stands for a recording nobody searched; so is a run with no cutoff set yet (run
  `query-calibrate` first).

  The artifact is pinned as it is exported and the snapshot id is printed, so a figure
  built from these CSVs can cite something that survives a later re-cut of the run.
- `temporal-clustering --experiment <dir> --recording <id> [--recording <id> …]
  [--embed-run <emb_NN> | --embed-ref <RECORDING:emb_NN> [--embed-ref …]] [--track 0] [--method codebook_viterbi]
  [--detector binseg] [--min-bout <s>] [--band <s>] [--max-bouts <n>]
  [--fit-share <percent>] [--codewords <n>] [--min-run <s>]`: discover a vocabulary
  of **behaviors** pooled across the selected recordings. With `--method dtw_finch`, each trajectory is cut into bouts at the
  moments the pose changes, every pair of bouts is compared, and the comparison is grouped; the default, `codebook_viterbi`, is
  described further down. A cluster is a stretch of movement
  and an instance is one performance of it.

  Each recording is read through **its own** embedding run, matched on settings rather than on
  its id. A recording embedded later, or one whose first attempt failed and was retried, holds
  the same settings under a different id (`emb_02` where the rest have `emb_01`), and it still
  joins the others. `--embed-ref <RECORDING:emb_NN>` names the settings by pointing at one run
  that has them. `--embed-run <emb_NN>` is the literal form: that id, in every recording. Leave
  both out and the one set of embedding settings every selected recording has a run of is used;
  the command is **refused** when there is none (it names the recordings that are missing one)
  and when there are several, rather than picking whichever sorted first.

  **Recordings made through different chains.** Settings include everything upstream of the
  embedding, so recordings whose animals were detected by two different models (one per
  strain, say) have no one set of settings in common, even when the embedding itself was made
  identically. Repeat `--embed-ref`, once per chain:
  `--embed-ref recording_000:emb_01 --embed-ref recording_080:emb_01`. Each recording reads
  its run of the one named setting it has. A recording with a run of none of them, or of more
  than one, is refused rather than resolved by order, and the runs must be **comparable** (the
  same embedding model, variant, read-out, input image and size), or the command is refused
  rather than pooling two kinds of vector into one vocabulary.

  For `dtw_finch`, comparing every pair is what lets a slow and a fast performance of the same behavior match,
  and it is why a fit is expensive: the cost grows with the **square** of the bout count.
  `--max-bouts` evenly samples the bouts the comparison sees and is the only bound on it.

  **A cap changes what the vocabulary is learned from, not what gets labeled.** Every remaining
  bout is matched to the nearest cluster afterwards, so the output carries one instance per bout
  however low you set it, which matters because finding every occurrence is the point. What an
  assigned bout does not get is a say in what the clusters are, so the run records both counts and
  `temporal-interpret` prints the split every time.

  **Two methods, and `--method` picks between them.** `dtw_finch` is the above.
  `codebook_viterbi` (the default) clusters a sample of *frames* into a codebook, labels every
  frame against it, then recovers the stretches with a decode, so its cost grows in proportion to
  the video pooled rather than with the square of the bout count. `--fit-share` is the percent of each recording the
  codebook is built from, so every video is sampled at the same density however large the study.
  `--codewords` is how many codewords it has, and the cost knob: the decode costs roughly the
  square of it on every frame pooled, so doubling it roughly quadruples the decode. `--fit-rows`
  was removed when the sample became a share, and is refused with the two flags that replace it. They tie on vocabulary quality; the bout path draws slightly better boundaries. Each
  method's knobs are its own: passing one that belongs to the other is refused rather than
  ignored, with a message naming what the method does accept.

  After the fit, the run prepares the evidence every cluster shows in the GUI, however small the
  cluster: counts, time per recording, positions, and the list of its clips with how typical each
  one is. Nothing is decoded from the video at this step. Time is measured from the recordings' own
  timestamps.

  The fit leaves the run **open**, not complete: a vocabulary is not a result until somebody has
  looked at it and labelled things. Interpret it with `temporal-interpret`.

- `temporal-interpret --experiment <dir> --run <an_NN> [--tree] [--name-node <node>=<label> …]
  [--rename-label <old>=<new> …] [--merge-label <from>=<into> …]
  [--rebuild-cache] [--finalize] [--reopen]`: label the clusters you care
  about. The fit found a whole tree of clusters, from one holding everything down to the finest
  groups, and **a label goes on any node of that tree**. Every operation is quick: it rewrites the
  run's state and rebuilds the behavior output from a fit already on disk; nothing re-fits or re-compares. Each
  command ends with a `coverage:` line per recording: how much of the video was analysed, and how
  much of that is in a bout. A capped fit prints a `note:` line first, every time.

  `--tree` prints the nodes, coarsest level first: each with its parent, its time, its share of the
  run, how many clips it has, and its label (in brackets when a label above it covers it). A
  clip is one action: one unbroken stretch at the finest level of the tree, inside the node, at
  least as long as the run's shortest instance. Stretches next to each other are never joined, so
  the root counts every action in the run and a node's clips add up to its children's. `--name-node <node>=<label>` labels one. A label covers everything beneath the node unless a
  node inside has its own, more specific label, so one coarse label and a finer one under it
  coexist:

  ```
  detail analysis temporal-interpret --experiment $PROJ --run an_07 --tree
  detail analysis temporal-interpret --experiment $PROJ --run an_07 --name-node 118='not moving'
  detail analysis temporal-interpret --experiment $PROJ --run an_07 --name-node 96='resting'
  ```

  Exported, an instance under node 96 reads `label` `resting` and `label_path`
  `not moving/resting`, so you can total up either level. Everything under no label is
  **unlabelled**. A label is one thing however many nodes carry it: `--name-node` with a name that
  already exists adds the node to that label, and the nodes of one label count as one behavior
  (their bouts join when they follow each other). A label's nodes must sit under the same parent
  label, and names are unique ignoring capitals; an edit that breaks either is refused and says
  which nodes. `--name-node 96=` removes a node's label. A label may not contain `/`, which
  separates the levels.

  `--rename-label rearing='rearing up'` renames a label and every node carrying it, and
  `--merge-label scaling=climbing` moves every node of `scaling` to `climbing` and removes
  `scaling`. Both take names and are repeatable. A merge carries the clips removed from
  `scaling` or moved into it (in AUDIT) to `climbing`, so they stay removed or moved. The
  command's `--name-node`, `--rename-label` and `--merge-label` edits are one change, made in
  the order given, so a merge can name the label a rename just made: if any of them fails
  (a label that does not exist, say) nothing is written, and the error names the labels there
  are. It reports its progress. The name `other` is refused: it is what
  the GUI's labels panel calls a label's own remainder.

  `--from-screen cmp_NN` records that the nodes you label with `--name-node` were found by that
  saved cluster screen (`comparison run --source nodes`): the label remembers the screen, the
  cluster it kept and the column it contrasted, so a later `comparison run` of the label over the
  same column can say the screen chose the cluster it scores. Each node must be one the screen
  kept, or sit inside or above one; any other is refused naming what the screen kept. `--tree`
  shows such a label as `climb [from cmp_02]`. `--at-p 0.2` reads the screen looser than its own
  5% floor, as the GUI's **Strictness** slider does: the clusters whose family-wise p is at most
  0.2 count as kept, the refusal names what is kept at that level, and the label records the level
  with the cluster's p's, so it stays marked as a lead rather than a finding.

  `--rebuild-cache` rebuilds every cluster's evidence from the fit, with a progress bar: for a run
  whose clips are reported out of date, or one prepared before every cluster listed its clips
  (such a run still opens; its clips just have no typicality order until it is rebuilt). It does
  not change any label and is allowed on a finished run. `--min-cluster-s` is gone: there is no
  size floor, and a run saved with one ignores it.

- `export-temporal-clustering --experiment <dir> --run <an_NN> --out <dir>`: write one
  behavior-stream CSV per fitted recording, into a new folder inside `--out` that the command
  prints. Each row carries `origin: predicted`, so re-importing
  one can never read back as ground truth.

  The artifact is pinned as it is exported and the snapshot id is printed, so a figure
  built from these CSVs can cite something that survives a later re-cut of the run.


- `cluster-montage --experiment <dir> --run <an_NN> --node <node> [--out <file.mp4|.gif>]
  [--per-clip-gifs <dir>] [--csv <file>] [--clips N] [--sort typical|least-typical|recording|length|random]
  [--raw/--no-raw] [--mask/--no-mask] [--keypoints/--no-keypoints] [--mask-trail/--no-mask-trail]
  [--keypoint-trail/--no-keypoint-trail] [--trail-point <name> …] [--aligned/--as-recorded]
  [--context <s>] [--blind] [--max-width <px>]
  [--fps <n>]`: exports a node's clips from the terminal, for a node of a
  temporal-clustering run. It writes the same files a clip grid's **Export ▾** writes in the app. `--out` writes the montage itself: the node's
  clips side by side, each cropped to its animal and labelled `#rank · recording · start`, all
  starting together and each looping from its start the moment it ends. `--per-clip-gifs`
  writes one GIF per clip into a folder, named `rank_recording_m-ss.t.gif`, and refuses before
  writing anything when that folder already holds a GIF of one of those names; `--csv` writes the
  clips as a list: a clip-query CSV (`analysis query --from-csv` reads it as example clips) with
  the clip's typicality rank, its recording's name, the subject, the times in seconds, the node
  and the run after the standard columns. Give at least one of the three. The defaults: 25 clips,
  most typical first, the mask and its trail on white (a clip grid's **Export ▾** in the app
  writes every clip it shows); `--raw` draws the
  footage with the animal outlined instead. `--sort random` shuffles the cluster's clips, the same
  way every time for the same node, so a montage of 25 is a fair look at a large cluster rather
  than its centre alone. `--aligned` turns each clip so the animal's body points up, frame by
  frame, as **Aligned** does on every clip grid in the app: it needs the keypoint run the
  embedding read and a heading definition, and without them every frame is drawn as recorded; a
  frame whose body axis cannot be measured is drawn as recorded with a small *no axis* mark.
  `--context 1` plays a second of footage either side of each clip (**Show ▾**'s context on a
  clip grid); the labels and the `--csv` list still name each clip's own span. Every cluster has clips, however small. A run fitted before the node
  cache existed has none listed until it is built: the command says so and names
  `temporal-interpret --rebuild-cache`, which builds it. `--max-width` and `--fps` shape a picture, so they
  are refused with only `--csv`; left out, a GIF is 640 pixels at 12 a second and an mp4 is
  200-pixel tiles at the montage's own clock (the footage's rate, at most 15 a second).

  ```bash
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --out rearing.gif
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --per-clip-gifs clips/ --csv clips.csv --sort least-typical --clips 50
  ```

- `snapshots <an_NN> --experiment <dir> [--snapshot <snap_NN>]`: list the run's **pinned
  artifacts**, frozen copies of its results, saved automatically whenever a number left the run
  (an export, a comparison, a finish). Each line gives when it was taken, why, a one-line summary of
  what the run was interpreted as at the time: a clustering run's labels, a scan's behaviors and how
  many were calibrated, and which comparison read it. `--snapshot` shows one in full, including a
  scan's cutoffs; for a clustering run it adds one line per label with its time and share, plus
  unlabelled time. The labels come from the snapshot and the times from the run's prepared
  evidence, so that table reads `unavailable` if the run is gone. The link to a
  comparison is worked out by reading the comparisons back, so it is always current.

  Reopening a run and relabelling it never changes a snapshot.

- `query-eval --experiment <dir> --embed-run <emb_NN> --behavior <label> [--behavior <label> …]
  [--search <id> …] [--stream <name>] [--band 0.5] [--fusion min|mean] [--method dtw]
  [--keyframe-select medoid|distinctive|rare] [--n-keyframes 1] [--iou 0.5] [--point-window 0.5]
  [--n-exemplars 1] [--n-holdout <n>] [--folds 1] [--seed 0] [--split-gap <s>] [--max-results <n>] [--track 0]
  [--k 20] [--tau 0.07] [--n-negatives 500] [--negative-pool unlabeled|labeled_clean]
  --out metrics.csv [--pr-out <dir>] [--predictions-out <csv>] [--ground-truth-out <csv>]
  [--folds-out <csv>] [--overlap-out <csv>]`:
  **score** how well the behavior query retrieves a
  labeled behavior against the `behavior` ground-truth streams. Held-out exemplars
  (`--n-exemplars`, chosen by `--seed`) seed the query; the ranked matches are scored against
  the remaining ground-truth intervals: **state** behaviors by temporal IoU (`--iou`),
  **point** behaviors by a `--point-window` tolerance (the type is captured at import).
  `--stream <name>` picks **which** ground-truth stream to score against. Omit it and a recording
  carrying exactly one behavior stream uses that one; a recording carrying **several stops the run**
  and lists their names, rather than silently scoring one of them (say an early hand-scored subset
  alongside a full ethogram, whose labels would then count as unscored and drop the recording).
  An unknown name fails with the available names rather than scoring nothing. Writes
  **one row per `--behavior`** to `--out` (appended, so a shell loop over configs / embedding
  runs builds a benchmark table): AP, precision, recall, best-F1, and counts.
  `--negative-pool` decides where `soft_knn` draws its negative class. `unlabeled` (default)
  samples everything not held out: the target behavior included, because at query time nobody
  knows which frames those are. `labeled_clean` samples only frames known *not* to be the
  behavior, which needs the ground truth and so reports a score no user can reproduce; it is kept
  to re-read runs recorded before this option existed, and a metrics row names which was used.
  The gap is large: on one real behavior a typical exemplar scores 0.92 clean and 0.50 unlabeled.
  `--n-holdout` fixes **how many instances leave the scored set**, independent of how many seed
  the query (default: `--n-exemplars`). Sweeping `--n-exemplars` without it compares runs whose
  scored sets differ (each extra exemplar removes another instance from the ground truth) so set
  it to the largest `--n-exemplars` in the sweep and `n_gt` stays constant across the benchmark.
  Instances held out but not queried are scored neither way; they leave the test set.
  `--folds` **cross-validates** for a seed-independent accuracy: `1` (default) is a single split;
  `k` rotates `k` distinct exemplar sets; `0` is leave-one-out (every instance once; requires
  `--n-exemplars 1`, and incompatible with `--n-holdout`). Each `--out` row then holds the **mean ± std** over folds (`n_folds` +
  `*_std` columns); `--folds-out` writes the per-fold metrics behind it and `--overlap-out` the
  per-overlap-level scores with their nulls. `--predictions-out` and
  `--ground-truth-out` write the **per-clip detail** (each retrieved candidate tagged `tp`/`fp`
  with its matched GT + IoU; each ground-truth interval `matched`/`fn`, plus the held-out
  exemplars, all with a `fold` column), the same tables the GUI evaluation report reads. This command
  is **synchronous and scriptable** (it builds the append-sweep and does not create a recorded
  run); the recorded, inspectable form is the `query_eval` analysis module run from the GUI (or
  `Experiment.run_analysis`), which writes those same files into `analysis/<an_NN>/`. A recording is
  scored for a behavior only if its annotation *covered* it; `--search` defaults to every eligible
  recording. **Vary the embedding run** to compare embedding configurations (embed model and size,
  input-image constructor, input size); vary `--band`/`--fusion`/`--method`
  for engine params.

  `--method` selects the scoring method, in two families. **Windowed matchers** (span-detection
  AP): `dtw` (default; banded subsequence DTW), `clip_pooled_mean`/`clip_pooled_max` (one pooled
  descriptor per clip) and `keyframe` (the most query-defining frame; `--keyframe-select` picks it
  by `medoid`, `distinctive` or `rare`, and `--n-keyframes` sets how many) return matches roughly the
  **example's length** and are scored by temporal overlap. The **per-frame classifier** (per-frame
  AP) `soft_knn` scores every frame by a softmax-weighted vote over its `--k` nearest labeled frames
  (the example frames plus `--n-negatives` sampled background frames, `--tau` sets the softmax
  temperature), ranks all frames, and integrates a threshold-free per-frame Average Precision, so
  there is no proposal cutoff; `--band` applies to the windowed matchers only.

  [Behavior-query evaluation](behavior-query-eval.md) covers the matching rule and every metric
  reported here.

  **Read every number next to its null.** `--overlap-out` writes recall/precision/AP at overlap
  0.1 / 0.25 / 0.5, each beside a deliberately uninformative version of itself: `recall_null` is
  the same predictions slid to the wrong places (same count, same lengths, wrong times), `ap_null`
  the same outcomes in a random order. `*_lift` is actual − null, the part that is real. A matcher
  that blankets the timeline scores a high recall whatever it retrieves, so a value that doesn't
  clear its null means nothing. The `--out` row carries the same nulls at `--iou`, plus the
  `method` that produced it. E.g.

  ```bash
  for emb in emb_01 emb_02; do
    detail analysis query-eval --experiment $PROJ --embed-run $emb \
        --behavior grooming --behavior rearing --out sweep.csv
  done
  ```
- `list --experiment <dir>`: the experiment's analysis runs.
- `status <an_NN> --experiment <dir>`: its module, status, recordings, **settings**, and
  artifacts. The settings block is what the run was launched with, ready to re-pass:

  ```
    settings:
      behaviors: 2 item(s)
      embed_run_id: emb_02 [selection]
      method: soft_knn
      method_params:
        k: 20
        tau: 0.07
      track_id: 0
      trajectory_split_gap_ns: (not set)
  ```

  `[selection]` marks a parameter that names *another artifact* rather than setting a value.
  Those are scoped to this experiment as it stands, so re-passing one only means what you want
  if that run is still the input you want; the rest carry anywhere. The GUI's **Refine…** makes
  the same split: it fills the form with the settings and re-asks for the selections.
  A run whose job **failed** says `failed` here even if its own record was written before the
  failure, and shows the error the worker recorded. A run whose **results are out of date**
  says so, with the command that fixes it.
- `attach <an_NN> --experiment <dir>`: stream a live run's progress (it keeps running with
  the CLI closed). Ctrl-C detaches; it does not cancel.
- `cancel <an_NN> --experiment <dir>`: stop an active run (kills its worker; a no-op
  once terminal).
- `rebuild <an_NN> --experiment <dir>`: re-derive an open run's results from its own stored
  output and its current settings. The repair for a run whose results are **out of date**:
  a change that was saved but did not finish rebuilding them, so the settings moved and the
  results did not (`status` says so, and the GUI offers **Rebuild results**). The command whose
  rebuild failed says the change was saved, and names the recording it could not read. Until a
  rebuild succeeds, every command that reads the results refuses with this command's name:
  `measures`, `comparison run`, `export-query`, `export-temporal-clustering`, a finalize, a new
  audit batch and a finished clip review. The expensive pass is untouched:
  this costs one pass over the outputs, not another fit or search. It is also the way to
  re-drive a run whose results write **failed**: the fit or search is still on disk, so the
  run does not have to be redone from the start.

  A label or cutoff change writes the results and leaves the per-frame table for
  later, so it returns in seconds. `measures`, `--finalize` and the `export-*` commands bring that
  table up to date first, showing a progress bar only when there is something to rebuild.
- `delete <an_NN> --experiment <dir> [--yes]`: remove the run's `analysis/<an_NN>/`
  directory. Blocked only while the run is active (cancel it first).

### `measures`
Turn a behavior result into numbers: how many bouts, how much total time, a rate against observed
time, a mean bout duration. Any module that produces labeled behavior in time can be measured. The
full list, with what each one reports and what it needs, is the
[Measures reference](measures.md).

- `list`: every measure, with its name, what it answers, the unit its value carries, and the rule
  it uses at a window edge. `--measure` below takes the id in the first column, not the name.
- `export --experiment <dir> --run <an_NN> --out <dir> [--measure <name> …]
  [--windows whole|absolute|normalized|sliding] [--every <seconds>] [--parts N]
  [--length <seconds>] [--step <seconds>]`: measure the run and write three files into a new
  folder inside `<dir>` (printed; a second export never replaces the first), `measures.csv` (one row per recording, subject, label, window and measure, with the
  observed time each rate was taken against; a label with labels nested inside it counts only
  what they leave, so its row has `label_display` "<label> (other)" and `label_is_remainder`
  `True`, while `label` keeps the plain name to join on), `frames.csv` (one row per frame: whether it was
  observed and what was active on it), and `measures.yaml` recording the windows and measures used.

  `--measure` defaults to every measure the source run can answer: one that needs something
  the run does not have (`transitions` needs non-overlapping behaviors; `unnamed_time` needs a
  clustering run) is skipped rather than reported as a column of zeros. `--windows` picks the
  window model: `whole` measures the
  full recording, `absolute` tiles it into fixed windows of `--every` seconds, `normalized` splits
  it into `--parts` equal fractions (for recordings of differing length), and `sliding` walks a
  `--length` window forward by `--step` for a time-resolved curve.

The instance table is not copied here: the run already writes `instances.csv` in
`analysis/<an_NN>/`.

### `comparison`
Group statistics across recording metadata: take a behavior result, split the recordings by a
column of `metadata.csv`, and report how the groups differ.

- `run --experiment <dir> --source-run <an_NN> [--snapshot <snap_NN>] [--source labels|nodes] --group-by <column> --aggregate-to <column|none>
  [--level <value> --level <value>] [--measure <name> …] [--recording <id> …] [--windows whole|absolute|normalized|sliding]
  [--every <seconds>] [--parts N] [--length <seconds>] [--step <seconds>] [--draws N] [--seed N]
  [--presence-s <seconds>] [--pseudo-count-s <seconds>] [--min-units-fraction <share>] [--no-suppress-nesting]
  [--statistic moderated_t|effect]`: measure each unit, group them, and
  save the result to `comparisons/<cmp_NN>/`. Prints the `cmp_NN`, the design it resolved and
  what the ranking scores by. A recording whose video is not on this machine (a slim import,
  an unplugged drive) is measured all the same; the record notes the video as missing.
  `--group-by`, `--aggregate-to`, `--level`, `--measure` and `--recording` are all repeatable.

  **The design follows from the column.** A `--group-by` column that is constant within each
  `--aggregate-to` unit (sex) compares between units; one that varies within units compares
  within them: two values are a paired contrast, three or more numeric values a trend, and three
  or more other values a profile over all of them, each measure reported by the pair of values it
  differs most between. `--level A --level B` contrasts just those two (the second against the
  first), on any column; on a numeric one it replaces the trend. A
  recording with no value in the column (blank, or `NA`) is left out, and the command says which.
  The command prints the design it ran, and the record carries it. Within-unit designs count a
  unit toward the ranking only when the behavior is present in it (`--presence-s` of total time
  at some level), need that in `--min-units-fraction` of the cohort (default half) for an effect
  to be defined, and add `--pseudo-count-s` of time to both sides of a ratio so a behavior absent
  at one level has a finite one.

  **A within-unit ranking scores by agreement, not by size.** By default (`--statistic
  moderated_t`) each measure is ranked, and the floor taken, on its moderated t: the mean
  per-unit effect divided by its spread across the units (plus the family's median spread, so
  exact agreement cannot own the floor). A behavior eight of eight animals agree on at 2x
  outranks one that swings tenfold either way in different animals, and the floor stops belonging
  to the sparsest behaviors. The effect itself (the fold change, the mean rho) is still what
  every table prints; the t sits beside it, in the tables' `t` column and `ranking.csv`'s
  `statistic`. `--statistic effect` ranks on the mean effect itself, as comparisons run before
  this option existed did; those records read back as they were run. A between design has no
  per-unit effects and ranks on the effect whatever is asked; the printed line says which ran.

  **`--source nodes` screens a clustering run's unlabelled clusters.** Every node of its tree with
  at least `--candidate-floor-s` seconds of member time (30 by default) becomes a row
  (`node:<id>`), measured from the node cache over the whole recording: share of time (what the ranking is on), rate, total time and mean
  bout length. Nested nodes that clear the floor are collapsed to the best of them
  (`--no-suppress-nesting` keeps every slice). The run needs a current node cache; a stale one is
  refused with what changed. It always takes those four measures and reads the cache as it is
  now, so `--measure` naming fewer and `--snapshot` are refused. A screen whose recordings cannot
  give a floor is refused before anything is measured, naming why and what would work: one animal
  holding both sessions of a paired contrast, one animal in every group, or a single group (`only
  one animal_id (3) has both habituation and fentanyl_25, so a within-animal_id null has nothing to
  move and no floor can be computed: this screen is refused before measuring, since it could keep
  nothing. Add a second animal_id that has both …`). `--source labels` over the same recordings
  runs, and its summary says *No floor computed*. `export` then writes `ranking.csv` with every node's effect, its
  `statistic` (the moderated t the floor is in the unit of, blank on a ranking by the effect),
  floor and verdict, and draws the tables and figures for the kept nodes and the next ten below
  the floor rather than for thousands of nodes. Each row also carries `family_p` (the share of
  shuffled cohorts whose largest t reaches it: the floor read as a p, at most 0.05 for a node
  that clears), `own_p` (the same share against the node's own shuffles alone, so a cluster
  consistent in its animals but hidden by the size of the family still says so) and `q` (the
  Benjamini–Hochberg false-discovery rate over the family's own p's). Threshold `family_p`
  yourself to read the screen looser or stricter than its floor; that is what the GUI's
  Strictness slider does.

  **A labels comparison says which of its labels a screen chose.** A label written from a
  cluster screen (`--from-screen`, or a screen ticked in the GUI) carries that screen; `run`
  over the same column prints `not independent: climb (cmp_02) was chosen from a difference
  screen over drug …`, the label's `suggestion` cell in `ranking.csv` and `summary.md` ends with
  `chosen from screen cmp_02 (cluster #118 kept)`, and `list` marks the comparison
  `screened=climb`. The number is still the number; the line is what keeps it from being read as
  a finding the screen did not already make.

  **The source is pinned, so the numbers keep meaning what they meant.** An analysis run stays
  editable after it finishes (you can reopen it and re-cut the vocabulary) so a comparison that
  named only the run would describe different numbers next month. Running one saves a **snapshot**
  of the source artifact and records which snapshot it read; the run stays free to change.
  `--snapshot snap_NN` re-runs against an earlier one. Comparisons saved before snapshots existed
  keep working and read the run's current artifact.

  **`--aggregate-to` is required whenever your cohort has repeats.** It is the unit of analysis,
  what one point in the result *is*. `--aggregate-to animal` averages each animal's recordings
  first, so one animal contributes one observation; `--aggregate-to none` treats every recording as
  its own observation. When one animal contributed several recordings the two give different
  answers and the output looks identical either way, so the CLI refuses rather than choosing, and
  the refusal names the column it found and how many recordings it covers, so you know which one to
  pass. When no animal contributed more than one recording there is nothing to decide, and the
  command runs with one observation per recording. It is recorded on every row of `stats.csv` and
  in `summary.md`.

  `--measure` defaults to every measure the source run can answer: one that needs something
  the run does not have (`transitions` needs non-overlapping behaviors; `unnamed_time` needs a
  clustering run) is skipped rather than reported as a column of zeros. `--windows` picks the
  window model: `whole` measures the
  full recording, `absolute` tiles it into fixed windows of `--every` seconds, `normalized`
  splits it into `--parts` equal fractions (for recordings of differing length), and `sliding`
  walks a `--length` window forward by `--step` for a time-resolved comparison. `--draws` sets how
  many permutations the difference ranking's floor is drawn from.
- `export <cmp_NN> --experiment <dir> --out <dir> [--format pdf|svg|png …] [--theme light|dark]`: write the publication
  tree, `stats.csv` (long form, one row per observation), `tables/` (per measure: groups as rows
  with n / coverage / mean / median / sd), `figures/` (every observation as a point, plus the
  difference ranking with its chance floor), `summary.md`, and `manifest.json` with a SHA-256 per
  artifact, plus `ranking.csv` (every ranked measure or node with its effect, its `statistic`
  where the ranking is on the moderated t, floor and verdict, and its `family_p` / `own_p` /
  `q`) and, for a node screen, the `nodes.csv` hierarchy. Warns if the source data has changed since
  the comparison was run, or if a recording's video is not on this machine, so it cannot be
  checked. `--format` is repeatable and takes any format matplotlib writes (`jpg`, `tiff`,
  `eps`, …); an unknown one is refused before anything is drawn. `--theme dark`
  draws the figures on a dark background for slides; the manifest records which theme was used.
- `list --experiment <dir>`: the experiment's comparisons, with the unit of analysis each used.
- `delete <cmp_NN> --experiment <dir> [--yes]`: remove the comparison and its exports. Confirms
  unless `--yes`.

Statistics stop at description: means, medians, spreads, and a permutation floor that says whether
a difference is bigger than shuffling the group labels would produce. The only p-values are the
floor's own (each ranked item's share of the shuffles that reach it, family-wise and on its own,
with a false-discovery q over them), and there is no model behind them; `stats.csv` is the
handoff to whatever you test with.

### `model`
Built-in (registry) model readiness/install, plus the **machine-global imported
model** catalog (shared across every experiment on the machine):

- `list`: every machine-global model (built-in, imported, or promoted from an experiment),
  one line per size, with install
  status, role (`detection`, `embedding` or `keypoint`), and a keypoint model's skeleton. No
  network.
- `check <id> [--experiment <dir>]`: readiness reason + action URL. Consults the
  network only when the model isn't installed.
- `install <id> [--experiment <dir>]`: download weights into the global cache
  (idempotent; prints per-file progress). The only command that downloads.
- `import <weights.pt> [--skeleton <preset|sk_NN|file.yaml>] [--family <family>] [--name <id>]
  [--made-by] [--description] [--notes] [--preview <image>]`: register an Ultralytics
  checkpoint into the machine-global catalog with optional metadata + a preview image. The
  checkpoint's task decides what it becomes, so leave `--family` out (when given, it must
  agree). A detection checkpoint becomes a detector, its class names read from the checkpoint;
  use it as `detection run --model <id>`. A pose checkpoint becomes a keypoint model and needs
  `--skeleton`: a preset, a library skeleton or a skeleton YAML (see `skeleton`). It must have
  the model's number of points, and left/right pairs that match the ones the model swaps when
  it flips an image; a checkpoint that records no swap accepts only a skeleton without pairs.
  The check cannot catch a pair named the wrong way round (left and right swapped). The
  skeleton is added to the library unless an entry with the same skeleton is there, and the
  command prints the entry. Use the model as `keypoints detect --model <id>`. `--skeleton` is
  refused for a detection checkpoint. An import that fails or is stopped part-way adds
  nothing to the catalog, so the same name can be imported again.
- `show <id>`: print one of your models' metadata + provenance (the detail page, as text),
  with its role, and for a keypoint model its skeleton: the library entry, points and pairs.
- `set-meta <id> [--made-by] [--description] [--notes] [--preview <image>]`: edit a
  model's metadata (only the flags you pass change; with none it stops with "nothing to
  change" rather than reporting an update).
- `export <id> <dest.zip>`: bundle a model (weights + `model.yaml` + preview) into a
  `.zip` to share it (shows progress).
- `restore <archive.zip>`: bring a model exported with `export` back into the catalog. A
  keypoint model keeps its own skeleton; it is added to the skeleton library unless an entry
  with the same skeleton is there ("name (2)" when the name is taken by a different one), and
  the command prints the entry it matched or added. A file that is not such an export, one
  whose `model.yaml` is empty or names another id than its folder, or one without the weights
  its `model.yaml` names, is refused and leaves nothing in the catalog; a bare `.pt`
  checkpoint is added with `import` instead.
- `remove <id>`: delete an imported model. It also deletes a folder that an import stopped
  part-way by an earlier version left in the catalog, which kept its name from being used.

### `modules`
List the registered processing modules and their dependency chain.

### `doctor`
Check this machine's setup: PyTorch / CUDA, the SAM 3 backend package, Hugging Face,
and model weights.

It starts with a **System** section, which is what a bug report needs: your build of
DETAIL and its commit, the operating system, the Python version and what kind of
environment it runs in, CPU, RAM, GPU, the NVIDIA driver and the newest CUDA it supports,
which CUDA your PyTorch was built for, and the version of every package DETAIL uses. No line
in it names a folder or your user name, so you can paste the whole section into an issue.

It runs the checks in a fresh subprocess, so it reflects the
*current* install state (handy for debugging a setup over SSH). Each failing check
names the docs page that explains the fix; when a backend package is missing it also
prints the full copy-paste install sequence for this machine, with the real detected
paths, from opening a terminal to verifying (the Windows Triton line only on Windows).
Exits non-zero if any check failed. This is the headless twin of the GUI's
**Configuration ▸ Check readiness** button and **SAM 3 backend** page.

Where the output is not UTF-8 (a redirected stream on Windows, a `LANG=C` terminal) a
character the encoding cannot carry is spelled in ASCII instead: the `✓` prints as `+`,
the `✗` as `x`, a `≥` as `>=`. This holds for every command, so a `detail` command never
stops part-way because a character did not fit.

### `version`
Which build of detail you are running, read from the git checkout you installed from:
the version, the commit, its branch and date, and whether you have local edits on top.
`detail --version` prints the same thing on one line (`v0.1.0 · 2026-09-20 · 033eef0`).

- `--history N`: also list the newest N changes on your branch, one line each with its
  date and commit. On `main` that is one line per merged change.
- `--check`: fetch the latest `main` from GitHub and say how many changes you are behind,
  and how to update. This is the only part that uses the network.

This is the headless twin of the GUI's **Version History** panel.

### `video-root`
Machine-local registry of the directories that reference-in-place videos live under
(stored in `~/.config/detail/storage.yaml`, never in an experiment). Referencing by a
named root keeps machine paths out of `recording.yaml` and makes a moved library a
one-command relink.

- `video-root set <alias> <path>`: register (or re-point) `<alias>` to a directory. Use
  it with `recording add-video --video-root <alias>`. Re-pointing relinks every recording
  that references videos under that alias.
- `video-root list`: list the registered roots.

### `cache`
This computer's video previews: small copies of each video part (480×270 at most, 15 frames a second) that clip montages play from, plus each tracking run's masks shrunk to match. They are kept outside every experiment (`~/.cache/detail/previews/` unless `DETAIL_PREVIEW_DIR` or `preview_cache` in `storage.yaml` says otherwise), are never analysed by any stage and are never exported, so clearing them deletes no data. Expect tens of megabytes per hour of video, and roughly twice that on Windows and Linux, where the video library cannot write H.264.

- `cache previews build --experiment <dir> [--recording <name>]…`: build every preview the
  experiment's recordings are missing (default: all of them), with a progress bar. Safe to
  re-run: built previews are skipped. While another build holds the cache (the GUI, or a
  second terminal), the bar says it is waiting for it. Each failed build is printed, and the
  command exits non-zero.
- `cache previews size`: where the previews are, and how much disk they use.
- `cache clear --previews`: delete every preview on this computer. They are rebuilt when next needed, which takes a while for a large experiment.

`recording add-video` builds the new part's preview after the import. `recording import`
does not; run `cache previews build` afterwards. The GUI builds them in the background.

### `hf-token <token>`
Validate and store a Hugging Face token in
`~/.config/detail/credentials.yaml`.

### `docs [TOPIC]`
Open the documentation in your browser. The site is built into a per-user cache on
first use and served over `file://`, so it works offline. `TOPIC` is an optional page
(`detail docs concepts`, `detail docs reference/parameters`). `--serve` runs a live
local server (`mkdocs serve`) instead.

### `tools quick-segment <video> [<video> …]`
Run a segmentation model on raw clips; no experiment, no import
([Tools guide](tools.md)). `--prompt <text>` is **repeatable and each repeat is a separate
prompt**, compared side by side (comma-separate to put several classes in one prompt:
`--prompt "mouse,cricket"`). `--sample` (default) spreads `--frames` (default 50) evenly
across each clip; `--consecutive` reads straight through from frame 0: one clip, one
prompt, and `--whole-video` removes the limit entirely (that run can take hours; it is
stoppable in the GUI). `--min-conf` (default `0.5`), `--backend` / `--device`, `--model`
(default: auto, SAM 3 on an NVIDIA GPU). Left out, `--backend` takes the best backend the
machine has that the model supports (cuda > mps > cpu), so SAM 3 runs on CUDA with no flag;
the run records the backend it used. A closed-set `--model` (one of your
imported or distilled models) finds its own classes and ignores the prompt's text, but
`--prompt` is still required. `--export out.mp4` renders the overlay clip and
needs `--consecutive`, since sampled frames are not contiguous; without it the command
stops before the run starts. A missing folder in the `--export` path is created.

A run keeps every proposal the model found, so `--min-conf` only chooses what gets
counted in the printed summary and drawn into `--export`. The saved run is the same
either way, and the GUI result view puts a slider on it. Saves a run under
`DetailExperiments/tools/quick_segment/`.

> Repeating `--prompt` used to mean one prompt carrying several classes; it now means
> several prompts to compare. Comma-separate for the old behavior.

- `tools list`: saved tool runs.
- `tools delete <run_id>`: remove one.

## Getting your data out

Every export writes **the data behind the artifact**, not only the artifact: the rows,
coordinates and frame ranges needed to rebuild a figure somewhere else. Each export folder also
gets a `params.csv` (the settings that produced the run), the subject's
[notes](#notes) as `notes.md` when it has any, and an `export.yaml` manifest.

Only what DETAIL made is exportable. Your video and your streams came from you and you still
have them; tracks, masks, embeddings and analysis results did not exist before we computed them.

```bash
detail export list --experiment ./study --analysis-run an_01
```

### Each export gets its own folder

`--out` is *where to put it*, not what to write into. The export creates
`<experiment>_<module>_<run>_export_<date>_<time>/` inside it, so two exports never collide,
nothing you already have there is touched, and the folder is still identifiable months later on
somebody else's machine.

### What each subject can produce

| Subject | Target | Writes |
| --- | --- | --- |
| experiment (no id flag) | `bundle` | one `.tar` of the whole experiment, importable back |
| | `metadata` | `metadata.csv`: one row per recording, one column per grouping variable |
| `--recording <id>` | `geometry` | field of view, distance calibration, every ROI and POI (the **scene geometry**) |
| | `streams` | one CSV per stream: its own time `t_ns` with video time `video_t_ns` beside it (`video_end_t_ns` too for intervals; a behavior stream also carries `origin`), plus `streams.csv` naming each stream's clock and stored alignment offset |
| | `timestamps` | `frame_idx,t_ns` for every frame of every video part |
| | `video` | the video parts DETAIL imported a copy of |
| | `clip` | one mp4 of the frame range `--frames START:END` names, plus `clip.csv` (see [Clips, in three flavours](#clips-in-three-flavours)) |
| `--run <run_NNN>`, a detection run | `masks` | one COCO instance-segmentation JSON per recording |
| `--run <run_NNN>`, a tracking run | `tracks` | one CSV per recording: a row per frame per track |
| `--run <run_NNN>`, an embedding run | `vectors` | `embeddings.npy` + `embeddings_index.csv` |
| `--run <run_NNN>`, a keypoint detect or keypoint track run | `keypoints` | one CSV per recording: a row per frame per track per keypoint, with the score, whether it is visible at the cut-off, and what the track stage changed |
| | `keypoints_dlc` | DeepLabCut CSV per video part (and the same table as H5 with the `keypoints_export` extra), plus `timestamps.csv` |
| | `keypoints_sleap` | the SLEAP analysis HDF5 per video part, plus `timestamps.csv` (needs the `keypoints_export` extra) |
| | `keypoints_nwb` | NWB with ndx-pose per video part (needs the `keypoints_export` extra) |
| | `keypoints_video` | the footage with the points drawn on, one mp4 per video part, plus `keypoints.csv` |
| `--run <run_NNN>`, any processing run | `performance` | `performance.md` (the machine it ran on and how long each part took), plus `timings.csv`, `phases.csv` and `machine.yaml` (see [How long a run took](#how-long-a-run-took-and-on-what)) |
| `--analysis-run <an_NN>` | `clips` | one mp4 per bout + `clips.csv` |
| | `instances` | `instances.csv` (with an `origin` column), `coverage.csv`, and the time-per-behavior and bout-duration figures |
| | `ethogram` | the ethogram as a picture, plus one `start_frame,end_frame,label,subject,origin` CSV per recording, which re-imports as predicted labels |
| | `frame_labels` | one CSV per recording and subject, one row per video frame (see [Per-frame behavior labels](#per-frame-behavior-labels)) |
| | `performance` | the same performance report, for the analysis run |
| `--distill-run <distill_NNN>` | `performance` | the same performance report, one unit per stage |

`export list` shows **every** target the subject can have, and says why one cannot run,
"this recording has no streams imported yet" is something you can act on, where a missing row
would leave you unable to tell that from "DETAIL cannot export streams".

### Video that was referenced in place

`video` exports the parts DETAIL holds a copy of. A part added with **reference in place** is
skipped and the reason says so: that file is already yours, under the video root it was
registered from, and copying it back out would export nothing DETAIL made. A recording whose
parts are all referenced reports `video` as unavailable rather than writing a short export.

Each exported video keeps its part's own number, the same number the `timestamps` target uses.
If part 1 was referenced and parts 2 and 3 were copied, you get `<recording>_part002.mp4` and
`<recording>_part003.mp4`, next to `<recording>_part002_timestamps.csv` and
`<recording>_part003_timestamps.csv`.

The same is true of the experiment `bundle`: it packs the experiment tree, and a referenced
video is not in that tree.

### Per-frame behavior labels

`--target frame_labels` writes the result the way a GLM or a correlation with neural data takes
it: a regressor sampled at the video's frame rate, not a list of bouts. One file per recording
and subject (`<recording>_subject<n>_frame_labels.csv`), one row per video frame:

```
frame_idx,t_ns,observed,rearing,c02
0,0,1,1,0
1,33366667,1,1,0
2,66733333,1,0,1
3,100100000,0,0,0
```

- `frame_idx` counts every frame of the recording from 0, across all its video parts.
- `t_ns` is that frame's time in nanoseconds from the start of the first part, the same
  clock as every other DETAIL table, so the file lines up with your other data without a lookup.
- `observed` is `1` when the frame was analyzed and `0` when it was not, typically because the
  tracker lost the animal there. **An unobserved frame is kept as a row, with `0` in every
  behavior column.** Read those zeros as "not looked at", not "not doing it": filter on
  `observed == 1` before computing a rate, or the lost frames count as time the behavior was
  absent and every rate comes out low.
- One `0`/`1` column per behavior, named as you named it. Clusters sharing a name share one
  column. A cluster nobody has named yet gets a column under its own id (`c02` above), the same
  as in the ethogram. If a behavior's name is also the id of an unnamed cluster the export stops
  and says so, since their columns would otherwise merge; rename one of them.

`--cluster <id>` narrows the file to that cluster's column; `--frames START:END` keeps only
those rows, in every recording's file. A recording that was never analyzed for a subject gets no
file.

### Clips, in three flavours

`--target clips` and `--target clip` take `--render` of `raw`, `overlay` and `mask`. Ask for
several and they come out of **one** decode pass, so three flavours cost about what one does.

`clips` writes one file per bout of an analysis run. `clip` writes **one** file, of the frame
range you name with `--frames START:END`: the same export a clip player's **Export video…**
button runs.

The full field of view is the default: a steady camera, where the surroundings are half the
answer. Crop to the subject when you want the pose at a constant scale for comparing bouts.

A clip holds **every** frame of its span and plays at the rate those frames were recorded at, so
it lasts as long as the behavior did, whatever stride detection or tracking ran at. On a frame
the subject has no detection on, `raw` is the frame as recorded, `overlay` is that same plain
frame, and `mask` is black. A cropped clip keeps the box from the subject's last detection
before that frame; before the first detection in the span, it uses that first one. A span the
subject is never detected across has no clip; its row stays in the CSV with the video column
blank. `fps` in `clips.csv` and `clip.csv` is the rate the clip was written at, so its frame
count divided by `fps` is `dur_s`, to within one frame.

### Keypoints, for pose tools

A keypoint track run (or a keypoint detect run, for the raw placements) exports in the shapes
other pose tools read. Name it by its experiment-level id, `--run <run_NNN>`, not by a
recording's `kt_NN` or `kd_NN`. Every one of them is written from the same rows, so a CSV and an NWB file of the
same export agree point for point. A track run's poses belong to tracked animals. A detect run's
do not: they export **unassigned**, with an empty `track_id` and each pose numbered within its
frame (`instance`), which DeepLabCut and SLEAP read as untracked individuals. `--source detect`
on a track run exports them the same way.

- `keypoints` is DETAIL's own long table, one CSV per recording: a row per frame per track per
  keypoint with `x_px, y_px, x_norm, y_norm, score, visible, changed`. Its `frame_idx` restarts
  at 0 in each video part, the one named in `part_id`; `t_ns` is counted from the start of the
  recording. `changed` is `0` as
  detected, `1` where the track stage chose another candidate, `2` where a filter moved it.
- `keypoints_dlc` is DeepLabCut's multi-animal analysis table, one per video part: a row per
  video frame, columns `scorer / individuals / bodyparts / coords` over `x, y, likelihood`. The
  CSV is always written; the H5 beside it (what keypoint-MoSeq, VAME, B-SOiD, SimBA and movement
  also read) needs the `keypoints_export` extra.
- `keypoints_sleap` is the SLEAP analysis HDF5, one per video part, in SLEAP's own layout
  (`tracks`, `node_names`, `track_names`, `point_scores`, `track_occupancy`, `edge_inds`).
- `keypoints_nwb` is NWB with ndx-pose, one per video part: a `PoseEstimation` per track, a
  series per point with timestamps, and the skeleton.
- `keypoints_video` is the footage with the points drawn on every frame, one mp4 per video part,
  with the same long table beside it as `keypoints.csv`, so the picture is never the only copy.

Things every format shares. A missing point is empty or NaN, never zero. `likelihood` and
`score` are the method's own point score, not a probability: comparable within one run, never
across methods. DeepLabCut and SLEAP files carry no time, so `timestamps.csv` (`frame_idx` to
`t_ns` per part) goes with them. Every export ships `params.csv` and `coverage.csv` (frames
placed per track and part, and each point's count at the cut-off). Files are named by the
recording's name, not its id. When two names would make the same file name (`Mouse 1 (pre)`
and `Mouse 1 [pre]`, or `Mouse` and `mouse`), each of those gets its recording id after the
name, so every recording keeps its own files.

`keypoints export` adds the options the generic `export run` does not have: `--cutoff` sets the
display cut-off (default: the detect method's own), under which a point is written not visible
with its coordinates kept; `--blank-below-cutoff` writes it as missing instead, for a tool that
cannot read a visibility column; `--source detect` on a track run exports the detect run's own
placements, unassigned, so the two can be compared; `--overlay` draws the tracking run's masks and boxes
under the points in the video.

### Figures

Anything with a picture exports the picture as well as the rows, as **PNG** (to look at) and
**SVG** (to edit, or scale for a paper). They are drawn headless, so the CLI produces the same
figures the app does.

### Exporting part of a run

```bash
detail export run --experiment ./study --analysis-run an_01 --target clips --cluster 3 --out ~/c3
detail export run --experiment ./study --analysis-run an_01 --target instances --instance <id> --out ~/picked
```

`--cluster` takes a source label; `--instance` is repeatable and takes `instance_id` values from
`instances.csv`.

`--for-recording` narrows a run's export to the recordings you name: an id or the name you gave
it, repeatable, and it combines with `--cluster`.

```bash
detail export run --experiment ./study --run run_002 --target tracks \
    --for-recording recording_004 --out ~/exports
```

Without it, **every** recording is exported, and for video that is the whole run: one overlay video per recording can add up to hundreds of gigabytes. A target that writes one file for the whole experiment (`metadata`, `bundle`) has no recordings to choose between, and `export list` shows which targets take `recordings` among their scopes. A recording the run does not cover is refused before anything is written.

### How big and how long

`export run` prints an estimate before it starts, from the same catalogue the app's dialog asks:

```bash
detail export run --experiment ./study --run run_002 --target tracks \
    --for-recording recording_004 --out ~/exports --estimate
```

`--estimate` prints the total and one line per target with how the figure was reached, then stops; nothing is written and no folder is made. The sizes are counted from what is on disk where they can be (detections, tracks, keypoint rows), measured by writing a sample through the real writer for the keypoint CSVs, and measured on this machine for video, where a few dozen frames are timed (briefly) and scaled up. They are estimates: video seconds read high, and a total says **at least** when a target could not be sized (the performance report, whose size depends on what is installed, and the SLEAP and NWB keypoint files, which compress their arrays by an amount nothing tells you before they are written; such a row adds nothing to the free-space check). It also says how much room the drive under `--out` has. The keypoint video's drawing time is a fixed figure per frame measured on another machine, not timed on this one.

If the estimate is larger than the free space there, the command **refuses** and says so, rather
than failing an hour in; `--ignore-free-space` starts it anyway, for a drive that frees up or an
estimate you know to be high. A long time is never a refusal.

### How long a run took, and on what

```bash
detail export run --experiment ./study --run run_003 --target performance --out ~/report
detail export run --experiment ./study --distill-run distill_001 --target performance --out ~/report
```

Writes `performance.md`, meant to be pasted: three hours of detection is fast or slow depending on
the box, and that comparison is only useful across machines. It holds:

- **the machine the run ran on**: GPU, driver, CPU, RAM, OS, versions. That is the computer that
  did the work, not the one exporting. A run from before DETAIL recorded it shows this computer and
  says so.
- **per recording**: wall clock, frames, fps, and the **peak RAM** and **peak GPU memory** the
  work used. RAM counts the worker and every process it started. GPU memory is blank when it was
  not measured: a CPU-only stage, or a run spread over several GPUs.
- **per phase**: how long each part took, such as *loading model* against *processing*, or a
  distill's labeling against its training epochs.

Beside it are `timings.csv`, `phases.csv` and `machine.yaml`, the same numbers as data. Pass the
experiment-level id (`run_003`, from `process ...`), an analysis run, or a distill run. For a
distill the units are its stages. Frames and fps are shown only for a stage that counts frames
(detection, tracking, embedding, keypoint detection and keypoint tracking); for a distill and
for an analysis they are blank, and the wall clock is the number to compare.

The timings come from the run's own `activity.csv`, so a run that finished before DETAIL kept one
reports blanks rather than guesses. A recording's wall clock runs from its first logged moment to
its last, and the run's ends when the run did. A run made before DETAIL logged those last moments
shows each recording a few seconds short, so a short recording's fps there reads too high. **Nothing is sent anywhere**: it is a file, and sharing it is
your decision.

### The columns

`clips.csv`, `instance_id`, `source_label` (the cluster's own id), `label` (the name a person
gave it, blank if none), `label_path` (that name and every group it sits inside, outermost
first), `recording_id`, `subject`, `start_frame`, `end_frame`, `t_ns`,
`end_t_ns`, `dur_s`, `fps` (the rate the clip plays at; blank when it has none), and one
`video_<render>` column per flavour. `start_frame` and `end_frame`
count every frame of the recording from 0, across all its video parts, so a bout in the second
video file carries frame numbers past the first file's last frame. The ethogram CSV's
`start_frame,end_frame` count the same way, which is what lets `recording add-stream --importer
behavior_csv` read it back onto the right place. A blank video column means
that bout produced no clip: the track was not detected across it, or it fell outside the video
, and the row is kept because its times are still true.

`instances.csv`: the same identity columns plus `score`, `label_path`, and `cx_norm`,
`cy_norm`: where each bout happened, as a fraction of the frame. `label_path` is the whole chain
of names the bout sits under, outermost first (`not moving/resting`) where `label` is the last
element on its own. Total by `label` for the specific behavior, or by a prefix of `label_path` for
the group it belongs to; each bout appears once either way, so neither double-counts. Measures and
comparisons count the same way: a label with labels inside it counts only what they leave, and
its row is shown as "<label> (other)". A flat
vocabulary's path is just its label. `coverage.csv` gives `observed_ns` per
`(recording_id, subject)`, which is the denominator for any rate you compute: video duration is
not recording duration is not tracked time.

The two position columns are **absent** on a run made before DETAIL recorded per-frame geometry.
`detail recording backfill-kinematics --experiment <dir> --recording <id> --embedding <emb_NN>`
fills it in and the next export has them. Present but **blank** is a different answer: that bout
had no frame we could place.

`<recording>_tracks.csv`: a row per frame per track: `frame_idx` (numbered across the whole
recording, not per video part), `t_ns`, `object_id`, then the bounding box and the mask centroid
in three unit families. `*_norm` is a fraction of the frame, `*_px` is pixels, and `*_mm` /
`*_mm2` are millimetres, **blank unless the recording has a distance calibration**. The same file
an `analysis run location_export` writes, byte for byte; that analysis also gives you each track's
position relative to your ROIs and POIs, which needs geometry this export does not.

Expect **a long run per recording**: the mask centroid means decoding a mask per detection, and a recording holds a great many of them. You do not have to sit through it, see below. `masks`, which hands the same masks over without decoding them, is quick.

### An export keeps going without you

`export run` submits the export as a **run** and streams it. Closing the terminal, or the app,
does not stop it; come back with `export attach`. Ctrl-C while you are watching cancels it (see
[what Ctrl-C does](#inspecting-and-troubleshooting-runs) on each command). A short export is
unchanged; it finishes before there is anything to watch.

```bash
detail export run --experiment ./study --run run_002 --target tracks --out ~/exports
detail export attach --experiment ./study export_001   # come back to it; Ctrl-C detaches, it does not cancel
detail export cancel --experiment ./study export_001   # stop it; what it wrote is kept
```

Exports appear under **Active processes → Exports** in the app, alongside everything else in flight.

**One export is one run**, however many targets you tick: they share a folder and a manifest, and
the run reports each target separately. If one fails the others still write: the run finishes
`partial` and `export.yaml` names which target failed and why, rather than leaving you an empty
folder an hour later.

`<recording>_masks.json`: the detection run as COCO instance segmentation: `images` (one per
frame that has detections, carrying `t_ns` as well as the COCO fields), `annotations` with
`segmentation` as run-length encoding, `bbox`, `area`, `score`, and `categories` from the class
names you prompted with. Coordinates are in the **mask resolution**, which `info` records
alongside the video's, masks were rasterized there, so a box scaled to anything else would not
sit on one. The mask data is the encoding DETAIL stored rather than a re-encoding of it.

`embeddings_index.csv`, `row`, `recording_id`, `track_id`, `part`, `frame_idx`, `t_ns`, one row
per row of `embeddings.npy`. The matrix alone cannot be joined to anything, which is why the
index is not optional. Here `frame_idx` restarts at 0 in each video part, the one named in
`part`. To join it to a file numbered across the whole recording (`<recording>_tracks.csv`,
frame labels), join on `t_ns`, which is the same clock in every file.

### Re-exporting

Every export makes a new folder, so re-exporting never overwrites the last one and never touches
anything else in the directory you pointed at. The same holds for `analysis export-query`,
`analysis export-temporal-clustering`, `measures export` and `locations export`. Inside one
export no two files share a name: names built from your text (a recording's, a cluster's) that
would come out the same carry the recording id or the subject, and a clip export writes one
file per bout even when two subjects did the same thing at the same time.

## Tracking

Every tracker takes `--conf-threshold` and `--arena`.

`--arena` takes the field of view's **name**, and that is what to pass when tracking more
than one recording: a name is the same field of view everywhere, while an arena *id* is
minted per recording, so the same id is a different polygon on the recording next to it.
Each recording resolves the name to its own arena, and one that has no field of view by that
name is refused by name rather than silently tracked whole-frame. An id is still accepted
for a single recording.


The single-subject trackers are for **one animal per recording**, which is what the rest of the
pipeline is built for: a subject is one track id, and every later stage picks one. They differ in
how they decide which detection is the subject on each frame.

**`anchor_tracker`** (default): for a **single known subject** (one animal per recording).
Per frame it keeps the single highest-confidence detection as the subject (track `0`),
optionally absorbs overlapping lower-confidence fragments, and drops everything else.
Good when the detector runs at a low threshold and emits scattered spurious boxes. Key
options:

Turning `--require-contiguous` **on** makes the mask's shape a veto: a frame whose detections all have a split mask: a tail or a limb separated from the body by a few pixels, gets no subject at all, even when the detector was confident. That can discard a noticeable share of the frames that had a detection. Leave it off unless the detector is producing scattered fragments you need filtered.

| Option | Meaning | Default |
|--------|---------|---------|
| `--conf-threshold` | minimum confidence for the subject anchor | `0.1` |
| `--secondary-conf` | minimum confidence to absorb an overlapping fragment | `0.05` |
| `--iou-threshold` | overlap a fragment needs with the anchor | `0.1` |
| `--max-secondary` | cap on fragments absorbed per frame | `2` |
| `--num-subjects` | subjects per frame (must be `1` for now) | `1` |
| `--require-contiguous` / `--no-require-contiguous` | only accept a subject whose mask is one connected region | off |

```bash
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker anchor_tracker \
    --conf-threshold 0.1 --secondary-conf 0.05 --max-secondary 2
```

Multi-subject `anchor_tracker` (`--num-subjects > 1`) is planned and currently
rejected.

**`iou_single_tracker`**: a **single known subject** like `anchor_tracker`, but it
follows the subject frame-to-frame by IoU continuity, so a distractor's one-frame
confidence spike can't steal it, and it rejects glass reflections by mask overlap.
Usually the better single-subject choice. Takes `--conf-threshold` and `--arena`;
set its other knobs with `--hyperparam KEY=VALUE`:

| Knob (`--hyperparam KEY=VALUE`) | Meaning | Default |
|--------|---------|---------|
| `conf_threshold` | minimum confidence for a subject candidate | `0.05` |
| `iou_weight` | weight of the IoU-continuity bonus in the score | `1.0` |
| `max_missed` | frames the subject may vanish before re-acquiring by confidence | `30` |
| `secondary_conf` | minimum confidence for a fragment that can attach to the subject | `0.05` |
| `merge_iou` | bbox-overlap pre-filter before a fragment is tested for merging | `0.1` |
| `merge_mask_iou` | mask-overlap floor to absorb an overlapping fragment | `0.5` |
| `max_merge` | fragments absorbed per frame (`0` = none, cleanest silhouette) | `0` |
| `require_contiguous` | reject fragmented masks (off; continuity already locks on) | `false` |

```bash
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker iou_single_tracker \
    --conf-threshold 0.05 --hyperparam max_missed=45 --hyperparam iou_weight=1.5
```

## Compute & multi-GPU

Detection and embedding take:

- `--backend {cpu,cuda,mps}` (default `cpu`)
- `--device N`: CUDA device index, **repeatable** for multiple GPUs. Left out, a CUDA run uses
  GPU 0 alone, not every GPU the machine has

```bash
# single GPU
detail detection run … --model sam3 --class mouse --prompt mouse --backend cuda --device 0
# two GPUs: work fans out across them
detail detection run … --backend cuda --device 0 --device 1
```

DINOv3 and a trained/imported YOLO run on `cpu`/`mps`/`cuda`; SAM 3 is officially CUDA-only.

**Stage commands default to `cpu`; training, distilling and quick segment pick the best
backend.** The `--backend` of `detection run`, `detection prompt-test`, `embedding run`,
`process detection`, `process embedding`, `keypoints detect` and `dataset label` defaults to
`cpu`, so a scripted or headless run is predictable; pass `--backend cuda` (or `mps`) to use an
accelerator. `train run --backend`, `tools quick-segment --backend`, the `--label-backend` /
`--train-backend` of `distill run` and `keypoints distill`, and the `--label-backend` of
`keypoints test` and `dataset keypoints` have no default: left out, they take the best backend present (cuda > mps > cpu), narrowed to
what the model or trainer supports. The GUI forms and the Python API default the same way.

## Inspecting and troubleshooting runs

- `detail recording status` (above) lists a recording's detection, tracking and embedding runs
  and their status: `complete`, `failed`, `cancelled`, or an active state.
  `process status <run_NNN>` covers the rest, keypoint runs included. No stage picks up where
  it stopped: cancelling detection, tracking, embedding, keypoint detect or keypoint track
  records `cancelled`, and you run it again (`process retry` re-runs it whole). A keypoint
  detect run cancelled by an older version may read `partial`; it does not resume either.
- A blocking command exits non-zero and points you at the details when a run
  doesn't complete. The `attach` commands are the exception: they watch a run and print
  its status, and Ctrl-C leaves the run going, so they exit 0 whatever the status says.

  ```
  Error: tracking run 'track_01' did not complete (status='failed');
  check job.json in the run directory for details.
  ```

  The run directory is under
  `recordings/<id>/<stage>/<run_id>/` inside the experiment; its `job.json` holds the
  `error` field. Fix the cause (e.g. lower `--conf`, pick a better prompt) and
  re-run; runs are immutable, so a new attempt gets the next id.
- **`worker.log` in the run directory has the rest.** `job.json`'s `error` is one line;
  the worker's full output: the traceback, and everything the backend itself printed
  (Ultralytics' training table, a model's load warnings), is written to `worker.log`
  beside it, and `coordinator.log` for a multi-recording run. It appends, so an
  interrupted attempt's log survives the resume that reuses the directory. This is the
  first file to attach to a bug report; it is not echoed to your terminal, where it
  would interleave with the progress bar and vanish when the window closes.
- **A run missing from a list** may have a record that can't be read. If the machine lost
  power or crashed while writing it, the record can be left empty or full of zero bytes.
  Lists (in the app and here) leave that run out, so the rest still opens, and print
  `skipping unreadable record <run id>` to the terminal. The run can't be recovered from
  that record: move its folder out of the experiment.
- Outputs live beside the inputs in the experiment tree
  (`detections/det_01/`, `tracking/track_01/`, `embeddings/emb_01/`); they are
  zarr stores read by the GUI and the Python API.
- **What Ctrl-C does depends on the command.** It **cancels** a run the command started in a
  worker of its own: `detection run`, `tracking run`, `embedding run`, the `analysis` runs,
  `dataset create` / `label` / `keypoints`, `train run` and `export run`. The worker is in that
  command's care, and walking away would leave it going with nothing watching it. It
  **detaches** from an experiment-level processing run (the `process` commands, and `keypoints
  detect` / `keypoints track`, which are `process` runs) and from a distill (`distill run`,
  `keypoints distill`): those keep going in the background, `process attach` or `distill attach`
  rejoins one, and `process cancel` or `distill cancel` stops it. On an `attach` command it
  **detaches** too: you did not start that worker, and reattaching exists precisely so you can
  let go of it again.
- **A run whose command was cut off still gets its real status.** If you close the terminal
  while `detection run` (or another one-recording run) is going, or a script starts one and
  never waits for it, the recording keeps saying `pending` only until the next `detail`
  command on that experiment, or until the app opens it. Either one reads how the run really
  ended and records that.
- **Stopping a run you are not watching:** `<thing> cancel <id>` (`process`, `analysis`,
  `distill`, `export`), `train stop <model_NNN>` for a training run (it finishes the epoch in
  progress and keeps the best weights), the GUI's **Cancel** button, or
  `Experiment.cancel_run` / `cancel_model` / `cancel_dataset` in Python. A dataset has no
  `cancel` subcommand; see [Not in the CLI](#not-in-the-cli).

## Workflows

### Training a custom detector from an experiment

Turn an experiment's detections into a YOLO detector. A dataset is built from a
**experiment-level processing run** (`run_NNN`), so first create one across the
recordings you want labels from, then export and train:

```bash
PROJ=~/Documents/DetailExperiments/study

# 1. experiment-level detection across recordings -> run_001
detail process detection --experiment $PROJ \
    --recording recording_001 --recording recording_002 \
    --model sam3 --class mouse --prompt mouse --conf 0.05 \
    --every-n-frames 2 --backend cuda --device 0
detail process list --experiment $PROJ
#   run_001    module=detection  status=complete   recordings=2

# 2. export a YOLO dataset (every 10th frame) -> dataset_001
detail dataset create --experiment $PROJ --name mouse_yolo \
    --source-run run_001 --every-n 10

# 3. train YOLO-s (segmentation) -> model_001  (uses CUDA, else MPS, when available)
detail train run --experiment $PROJ --name mouse_yolo11s \
    --dataset dataset_001 --base-model yolo11s-seg.pt \
    --epochs 100 --batch-size 16 --patience 20
detail train list --experiment $PROJ

# resume an interrupted run (Ctrl-C / crash) from its last checkpoint:
detail train resume model_001 --experiment $PROJ
detail train status model_001 --experiment $PROJ   # ...including the best epoch so far
detail train stop model_001 --experiment $PROJ     # finish this epoch, keep the best weights
```

The dataset reuses the run's stored masks (no re-detection); the trained model
lands in `models/model_001/run/weights/best.pt`. Training defaults to the best backend
the machine has (**CUDA**, else **MPS**, else CPU); pass `--backend cpu` to force CPU. `train resume` continues
from `run/weights/last.pt` instead of restarting from epoch 0, for a training that was
cancelled or crashed. `train stop` *finishes* the training: the model is `complete`, keeps its
best weights, records the epoch it stopped at (`stopped_at_epoch`), and has nothing to resume.

### Moving experiments between machines

Run heavy stages on a GPU VM, then pull the experiment to your laptop to inspect.

```bash
# on the VM: slim archive (outputs + metadata + models; NO source video/datasets)
detail experiment export --experiment $PROJ --out study.segproj.tar
#   exported: study.segproj.tar
scp study.segproj.tar laptop:~/

# on the laptop: places it in the managed workspace
detail experiment import study.segproj.tar
#   imported: /Users/you/Documents/DetailExperiments/study
```

Include the heavy bits when you need them: `--include-video`, `--include-datasets`, `--gzip`. `--gzip` rarely pays: DETAIL's data files are compressed already, so it barely shrinks the archive and makes the export much slower. An earlier archive saved inside the experiment folder is never packed into a later one. A folder that is a symlink is not followed, and the export names each one it left out:

```
exported: study.segproj.tar
left out raw_videos: it is a symlinked folder, which is not followed
```

On import, `--replace` replaces a same-id experiment and `--name <new>` lands a separate copy. Replacing deletes the whole existing experiment, so anything the archive does not carry is gone afterwards: the video files and datasets a slim archive leaves out, the export records no archive carries, and any earlier archive saved in the experiment folder. `--replace` prints that list before it replaces anything (it does not ask; stop it with Ctrl-C):

```
replacing experiment 'study' deletes what the bundle does not bring back:
  - 3 video file(s), 4.2 GB: the bundle carries no video (it can be put back from the original files with Locate video, or `detail recording relink`)
  - 1 export record(s), export_01: no bundle carries them (the exported files themselves are not touched)
imported: /Users/you/Documents/DetailExperiments/study
```

Without `--replace`, an import onto an existing experiment stops and shows the same list under "--replace would also delete". To keep the video, import with `--name <new>` instead, or export with `--include-video`.

**Putting the video back.** An experiment imported without its video keeps every recording's
timestamps, so its copied video can be put back from the original files: `detail recording relink
--recording <name> --part <N>` copies a file back in when its frame times match the recorded
part's exactly (see [`recording relink`](#recording)). Until then `--include-video` has nothing
of those recordings to pack, and the export says so rather than printing only `exported:`:

```
exported: study.segproj.tar
no video was packed: 2 recording(s) have no video on this machine (put it back with Locate video, or `detail recording relink`)
```

Recordings whose video is used in place are named the same way: that video never travels in an
archive.

Both commands show a progress bar on stderr, relabelled as the work moves through
its phases: export goes `scanning` → `checking` → `hashing` → `packing`, import
goes `reading` → `extracting` → `verifying` → `placing`. A slim export reads every
file twice (once for the checksum, once to pack it), so a large experiment takes
minutes even without its video. If a file changes between those two reads (a run
still writing to the experiment), the export stops and names the file; let the
run finish and export again. Before packing, the export also checks that no DETAIL
file holds an absolute path (a home folder such as `/home/…` or `/Users/…`, or a
drive such as `D:\`), since that path would not exist on the other machine; if one
does, it stops and names the file, the line and the path. Text you wrote is not
checked: your metadata table, a description, a note, or a recorded error message can
mention any folder. The archive is written as `<out>.partial` and
renamed when it is complete, so a stopped export never leaves a file that looks
finished. A video-excluded import is
fully analysis-ready (embeddings, tracking re-tuning, metadata); the
frame-preview surfaces show "source video not in this export", and the clip
picker refuses to mark example clips until the video is on this machine
(put it back with `recording relink`, above).

Video that was referenced in place never travels in the archive, but the
archive records which folder each [video root](#video-root) was on the
exporting machine. Import looks there (`finding video`, after `verifying`),
so video on a server both machines reach is found with nothing to do. It then
looks beside every folder that held video (the folder itself, and folders of
the root's name inside, beside or around it), so a study kept as one folder per
day is found from any one this machine knows, even when the two machines spell
the server differently. A folder
is used only when a video in it matches the recorded fingerprint, and a root
name this machine already uses for another folder is left alone: the imported
recordings get a root of their own. A file at the recorded path that is a
different video (its fingerprint does not match) counts as not found, so a
root name this machine uses for another folder never hands a recording someone
else's video. Whatever is still not found is listed:

```
imported: /Users/you/Documents/DetailExperiments/study
video not found on this machine for 2 recording(s): s1, s2. (A file at the same path that is a different video counts as not found.) Relink one file (`detail recording relink`) and every recording whose video is in the same folder finds it too.
```

The same check applies to every frame preview, frame read and clip export of a
referenced recording, not only at import: a referenced video whose file was
replaced, or whose [video root](#video-root) was re-pointed at a folder with a
different file of the same name, shows as not on this machine instead of
showing the other file's frames.

### Importing event streams (MED-PC)

Attach an aligned operant-behavior stream to a recording:

```bash
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name medpc --importer medpc --data box15.txt --program program.MPC
#   added stream 'medpc' (medpc) to recording_001: 1234 rows
```

`--importer medpc` parses Med Associates `.txt` + `.MPC` into canonical events. It reads one
program layout: event codes in array `E`, their times **in seconds** in array `T`, and an
`Event stamps ( array E )` block in the program that names each code. A program that logs into
other arrays is refused by name, and one that logs `T` in ticks is refused when a time is past
the session length `H`. Unused `(code 0, time 0)` slots after the last event are dropped and
counted (`note:` in the output); a code-0 event between real events is refused unless the
program names code 0.

MED-PC can save every box of a day into one data file. A recording is one session, so a file
holding several is refused, listing them, until you pick one with `--medpc-subject`,
`--medpc-box` or `--medpc-session N` (its position in the file, from 1). Sessions are never
merged. In a bundle manifest the same choice is `config: {subject: PX22}` on the stream.

```bash
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name medpc --importer medpc --data day.txt --program program.MPC --medpc-box 4
```

Align the stream to video time with `--offset` at import or `recording set-stream-offset`
afterwards (see [Aligning a stream](#aligning-a-stream)). An import that is refused adds
nothing to the recording, so you can fix the file and run the command again under the same
name. Previewing it against the video,
and finding the offset by scrubbing, are done in the [GUI](gui.md#recording-detail).

### Importing your own CSV streams

Bring any time-series in as a CSV. Two importers, one for each shape:

```bash
# continuous sensor trace (numeric value channels)
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name heart_rate --importer timeseries_csv --data hr.csv

# generic timestamped events (a label per row)
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name cues --importer events_csv --data cues.csv
```

**CSV format.** The header names exactly one **time column**, which sets the clock:

| Time column | Unit | Clock |
| --- | --- | --- |
| `seconds` | seconds from `t=0` | aligned to video by an offset (sync later) |
| `t_ns` | integer nanoseconds from `t=0` | aligned to video by an offset (sync later) |
| `frame` | video frame index | anchored to the video (needs the video imported) |

- **`timeseries_csv`** (continuous): every other column is a numeric **value channel**, kept
  under its own header name. `sample_rate_hz` is inferred for the `seconds` / `t_ns` clocks.

  ```
  seconds,bpm,temp
  0.0,72,36.5
  0.5,73,36.6
  ```

- **`events_csv`** (event): a **`label`** column is required; an integer **`code`** column is
  optional (blank cells are allowed). Canonical columns are `t_ns,code,label`.

  ```
  seconds,code,label
  1.0,5,press
  2.0,,release
  ```

The importer validates strictly and names the problem: no recognized time column, more than
one time column, a column name repeated in the header, an empty or header-only file, a
`timeseries_csv` with no value columns or a non-numeric cell (or a `label` column, which means
you want `events_csv`), and an `events_csv` with no `label`, an empty label, or a non-integer
`code` / `frame`. A time too large to be one (epoch milliseconds in a `seconds` column, say) is
refused naming its row, as it is by every importer and by `--offset`, rather than stored. The
behavior label importers (`behavior_csv`, `boris`) refuse a repeated column and a header-only
file too. `events_csv`, `behavior_csv` and `boris` match column names in any case, so `Label`
and `label` count as the same column there.

**A CSV that isn't shaped that way.** When the time column is called something else, or you
only want some of the columns, say so instead of editing the file. `--time-column` names the
column carrying time, `--time-is` says what it holds, and `--channel` picks the columns to
bring in (repeat it; `COLUMN=NAME` renames):

```bash
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name pose --importer timeseries_csv --data pose.csv \
    --time-column t_sec --time-is seconds --channel nose_x --channel nose_y=ny
```

`--time-is` takes `frame`, `seconds`, `t_ns`, `utc`, or `auto` (the default, which reads the
kind from the column's name and fails if the name doesn't say). **`utc`** takes an ISO 8601
timestamp or Unix epoch seconds and makes the first row `t=0`, for a logger that knows only
its own wall clock.

One CSV makes **one** stream carrying the channels you named. The alignment offset belongs to
the stream and channels off one device share a clock, so keeping them together means syncing
once rather than once per column.

### Aligning a stream

A stream on the `seconds` / `t_ns` / `utc` clock starts at offset 0, meaning "this began when
the video began". Set the real offset at import, or afterwards:

```bash
# at import: positive if the stream began after the video
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name pose --importer timeseries_csv --data pose.csv \
    --time-column t_sec --time-is seconds --offset 3.48

# afterwards, on a stream that already exists
detail recording set-stream-offset --experiment $PROJ --recording recording_001 \
    --name pose --offset 3.48
```

`video_t_ns = stream_t_ns + offset`. The canonical data is never rewritten, so this is a
one-number edit you can redo at any time. A `frame`-clocked stream and behavior labels are
anchored to the video already, and an offset for one is refused rather than stored, both at
import and by `set-stream-offset`.

The GUI's **Sync to video** finds the same number by scrubbing to the frame where the stream
starts; these flags are how you set it across a whole experiment without opening anything.

### Reusing a mapping across recordings

Working out which column is time and which columns you want is worth doing once. Save it as a
**preset** and apply it to the rest:

```bash
# save the mapping from the import that proved it works
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name pose --importer timeseries_csv --data pose_001.csv \
    --time-column t_sec --time-is seconds --channel nose_x --channel nose_y \
    --offset 3.48 --save-preset dlc_pose

detail experiment list-stream-presets --experiment $PROJ
#   dlc_pose  importer=timeseries_csv  time=t_sec (seconds)  channels: nose_x, nose_y
```

The preset is saved only after the import works. A preset name that is already taken is
refused before anything is imported; choose another name, or delete the old preset first with
`experiment delete-stream-preset`.

Then the rest of the experiment, from a table of one row per recording:

```csv
recording,file,offset_s
recording_002,pose_002.csv,3.512
recording_003,pose_003.csv,3.474
```

```bash
detail experiment apply-stream-preset --experiment $PROJ \
    --preset dlc_pose --name pose --sources sources.csv
#   [1/2] recording_002: done
#   [2/2] recording_003: done
#   imported 2, skipped 0
```

The first column can be a recording's id or name, as above, or its video file name: head
the column `video` (or `file_name`, `video_file` …) and each row is matched the way
[`import-metadata`](#recording-information) matches a file name: base name, any case,
extension optional. A name that matches no recording's video, or several, skips that
row.

```csv
video,file,offset_s
m27_day3.mp4,pose_002.csv,3.512
m27_day4.mp4,pose_003.csv,3.474
```

Paths in the table resolve relative to the table itself. **The offset column is per
recording** and that is the point: a preset carries the mapping, which is the same
everywhere, and not the alignment, which is not: each video and logger pair has its own
skew. Leave the cell empty for 0. A recording whose file fails is reported and skipped, so
one unreadable file doesn't sink the batch; when **every** recording is skipped the command
ends with an error. A header row (`recording`, `Recording ID`, `name`, in any case) is read as
a header, not as a recording.

One recording at a time takes `--preset` instead of `--importer` and the mapping flags.
`--time-column`, `--time-is`, `--channel` and `--covered-behaviors` are refused beside
`--preset`, which carries its own mapping; without `--offset` the preset's own offset applies:

```bash
detail recording add-stream --experiment $PROJ --recording recording_004 \
    --name pose --data pose_004.csv --preset dlc_pose --offset 3.49
```

Move a preset between experiments, or to a labmate:

```bash
detail experiment export-stream-preset --experiment $PROJ dlc_pose --out dlc_pose.yaml
detail experiment import-stream-preset --experiment $OTHER dlc_pose.yaml
detail experiment delete-stream-preset --experiment $PROJ dlc_pose
```

### Importing behavior labels (ground truth)

Import human-scored behavior intervals as a `behavior`-role stream. The recording's
video must already be imported (labels are frame-indexed and anchored to its
timestamps). One recording at a time:

```bash
# plain interval CSV: start_frame,end_frame,label[,subject]
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name behavior --importer behavior_csv --data labels.csv

# or a BORIS tabular-events export (Export events > Tabular events; CSV/TSV)
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name behavior --importer boris --data observations.tsv
```

A BORIS export can hold several observations (`Observation id`) and several subjects. A
recording is one animal in one session, so an export holding more than one of either is refused,
listing them, until you pick with `--boris-observation` and `--boris-subject` (an empty value
picks rows with no subject). They are never merged onto one track. In a bundle manifest the
same choice is `config: {observation: day1, subject: mA}`.

Each behavior is recorded as **point** or **state** (BORIS's *Behavior type* column, else
inferred from durations) so the evaluation scores it correctly. A BORIS behavior scored both as
START/STOP bouts and as POINT events is refused, naming the rows of each kind, since one
behavior cannot be both; and the annotation's **covered
behaviors** (the scored vocabulary) default to the labels present. Add
`--covered-behaviors "groom,rear"` to declare behaviors you *scored but never observed*; the
evaluation treats them as covered (a real absence, not "unscored") rather than skipping the recording.

**Ground truth vs predictions.** A stream also records where its labels came from. Anything you
import is ground truth unless you say otherwise; a label file the tool exported (`analysis export-temporal-clustering`,
`export-query`, the `ethogram`, `instances` and `streams` export targets) carries an `origin`
column and imports with that origin on its own. A file holding rows for several recordings (a
`recording_id` column, as in `instances.csv`) is refused: a stream belongs to one recording. Override either way with
`--origin ground_truth|predicted`.

```bash
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name model_output --importer behavior_csv --data predictions.csv --origin predicted
```

A predicted stream is never read as ground truth: it does not make a recording eligible for an
evaluation, and naming it with `--stream` is refused. Scoring a method against its own output
produces a good-looking number and measures nothing, so the distinction is enforced rather than
left to you to remember. Streams imported before this existed read as ground truth.

Or in bulk onto many existing recordings from a small manifest (a value may be a filename, or a
`{file, covered_behaviors}` mapping):

```yaml
# labels.yaml
schema_version: 1
labels:
  recording_001: labels/s1.csv
  recording_002: {file: labels/s2.csv, covered_behaviors: [groom, rear]}
```

```bash
detail recording import-labels --experiment $PROJ --manifest labels.yaml
#   [1/2] recording_001: done
#   [2/2] recording_002: done
#   imported 2/2 recording(s); skipped 0
```

The whole manifest is validated first (every recording must exist, every file must
resolve, a mapping takes only `file` and `covered_behaviors`, and a recording listed twice
is refused rather than keeping the second); then each file is imported against its own
recording's timestamps. A file that fails to parse is reported and skipped; the batch
continues, and when every file is skipped the command ends with an error. Every file in
one `import-labels` run uses the same `--importer` (`behavior_csv` or `boris`).

### Bulk importing recordings

Create many recordings (video + streams + grouping variables) from one manifest:

```bash
detail recording import --experiment $PROJ --bundle ./bundle --dry-run   # validate first
detail recording import --experiment $PROJ --bundle ./bundle             # then import
```

The bundle folder holds a `manifest.yaml` plus the referenced files; the whole
manifest is validated before anything is imported. One progress bar covers the
whole batch and shows how many recordings are done, the one importing now and its
step, and the estimated time left.

The manifest is a `recordings:` list (paths relative to the bundle folder);
everything except a recording `id` is optional. Every field, and each importer's
required files, is in the [Manifest reference](reference/manifest.md).

The whole manifest is validated before anything is imported: an `id` already in
the experiment, a duplicate id, a key the manifest does not take (a misspelled `vidoe:`, or
`name:`: a recording's name starts as its `id`), `recording_id` under `metadata`, an unknown
importer, a missing file, or video parts that disagree on fps/resolution all abort the
import (nothing is created). A blank `metadata` value (`sex:`) is stored blank.
`--dry-run` runs the same validation and reports without importing.

### Recording information

Record what varies across your recordings (treatment, sex, day, …): one row per
recording, one column per variable. From a CSV:

```bash
detail experiment export-metadata --experiment $PROJ --csv groups.csv   # template to fill in
detail experiment import-metadata --experiment $PROJ --csv groups.csv --dry-run
detail experiment import-metadata --experiment $PROJ --csv groups.csv
```

`export-metadata` writes every recording as a row (`recording_id`, `name`, `file`,
the first video's file name, then each column); fill the blanks offline, then import
it back.

**Key your sheet on whatever you have.** Each row is matched to a recording by one
column, picked in this order:

1. `recording_id`
2. `name`: what you call the recording
3. the **video file name**, in a column headed `file`, `file_name`, `filename`, `video`,
   `video_file` or `video_name` (any case). This is usually the one a sheet made before
   import has.

A video file name matches on its base name, in any case, with or without the extension:
`M27_Day3.MP4`, `m27_day3` and `D:\raw\m27_day3.mp4` all find the recording whose video is
`m27_day3.mp4`, and any part of a multi-part recording counts. If two recordings have
videos with the same name (two rigs, two folders), the import names both and applies
nothing.

If your key column has some other heading, name it with `--key`:

```bash
detail experiment import-metadata --experiment $PROJ --csv groups.csv --key "Session Clip"
```

A column named with `--key` is matched as a video file name, unless it is `recording_id`
or `name`. Without `--key`, a sheet with none of the headings above is refused and the
error lists its columns.

A key column only *matches* rows; it never renames anything (rename on the recording's
page in the GUI). Import is merge-only: blank cells are skipped (existing values kept),
and a key that matches no recording, or two rows that match the same one, abort the
whole import (nothing applied).

Or set them one at a time, which is what a script over 30 animals wants:

```bash
detail experiment add-column --experiment $PROJ dose
detail experiment set-metadata --experiment $PROJ --recording m27_day3 --set dose=0.5 --set day=3
detail experiment delete-column --experiment $PROJ dose      # and its values everywhere
```

A new key in `--set` becomes a column on every recording (blank elsewhere): the
table is one shape, not one per recording.

### Naming recordings

`--recording` takes a **name or an id**. The name is yours (it starts as the video's
filename and you can change it on the recording's page in the GUI); the id is fixed when the
recording is made and never changes. The app assigns it (`recording_000`, in order, never
reused). `recording create --name <name>` names the recording and the app assigns its id; a
name already taken is refused. To choose ids yourself, which is what a script restoring
earlier data wants, give each recording an `id` in a [bulk import](#bulk-importing-recordings)
manifest; its name then starts as that id. `recording status` prints both:

```bash
detail recording status --experiment $PROJ --recording m27_day3
#   recording: m27_day3  (id recording_000)
#     source: /Volumes/rig2/m27_day3.mp4
```

So **never rename a video file to organise your data**, rename the recording. The
file can move (`recording relink`) and a recording can span several files, without
either changing what identifies it.

## Not in the CLI

The CLI mirrors the Python API: anything you can configure through the API or
GUI is either a CLI flag or listed here, with the reason. Most entries are
*deliberate* omissions: interactive, iterate-until-it-looks-right tools that live
in the [GUI](gui.md), not the CLI. A few are gaps, and say so:

- **`--embed-ref` for `analysis query`**: the `--*-ref
  <RECORDING:run_id>` spec selector (choose the upstream by *kind* so a batch's
  recordings can carry different run ids) is wired for `process tracking`, `process
  keypoint_detect` and `keypoints detect` (`--detection-ref`), `process keypoint_track` and
  `keypoints track` (`--detect-ref`, `--tracking-ref`), `process embedding` (`--tracking-ref`,
  `--keypoint-ref`), `analysis run` (`--tracking-ref`) and `temporal-clustering`
  (`--embed-ref`), but `query` still takes an
  explicit `--embed-run <emb_NN>` (must be that id in every searched recording). Deferred:
  its embedding run also identifies the *exemplar* recordings, so the spec resolution
  is bespoke (planned).
- **Renderer preview**: preview the masked input image embedding will use.
- **The clip window's enlarged panes and their exports**: opening a clip's Masks or Keypoints
  pane large, and writing it as an mp4 or GIF, is a thing you do while watching the clip window
  it follows, so it lives there. The API is `detail.api.clip_views.export_clip_pane`, which a
  script can call for one clip at a time; a whole cluster at once is `analysis cluster-montage`.
- **Picking clips to export (pick mode on any clip grid)**: clicking clips into an export, in
  the order clicked, on INTERPRET, a cluster window or AUDIT, is a way of looking, forgotten when
  pick mode ends (nothing is written to the run). The CLI writes a montage of the node's clips
  in an order it sorts (`--clips`, `--sort`, and the grid's other choices: the layers,
  `--aligned`, `--context`); a hand-picked subset in a hand-picked order is exported from the
  grid's **Export ▾**. A script picks its own clips and calls
  `detail.api.clip_montage.ClusterMontage` with them.
- **Undo and redo of label changes (⌘Z / ⇧⌘Z)**: INTERPRET keeps a history of the label changes
  made on the page and in its cluster windows, in memory, so a slip can be taken back. The CLI's
  label commands are single explicit writes, so there is no history to step back through; to
  reverse one, run the opposite command (`--name-node` the old name, `--rename-label` it back). The undo
  itself is in the Python API: `TemporalClusteringRun.label_maps()` read before and after an
  edit, `detail.api.authored_run.label_change(before, after)`, and
  `TemporalClusteringRun.restore_labels(labels, node_labels)` to write either side back, by
  label id.
- **Exporting the clip window's view as shown**: *Video of what you see…*, *GIF of what you
  see…* and *This frame as PNG…* write whatever the clip window's ticks, orientation and context
  are at that moment, so they live there. The API is
  `detail.api.clip_views.export_clip_view` and `write_clip_frame`, with a
  `ClipViewSettings` saying what to draw; a whole cluster at once is `analysis cluster-montage`.
- **Filtering a cluster view by metadata, recording, region or time in session**: shading the
  cluster tree by the share of each cluster that matches a metadata value or range, a set of
  recordings, a region of the frame and a stretch of each recording, and greying the clips that do
  not, is a way of exploring that answers as you click. The arithmetic is in the Python API, so a
  script gets the same numbers: `detail.api.cluster_filter.Selection` holds the conditions (values
  or a range per metadata column, recordings, cells of the frame, and `session=(lo, hi)` as
  fractions of each recording's whole video; OR within a column, AND across),
  `cluster_filter.session_spans(experiment.dir, recording_ids)` reads each recording's first and
  last frame time for it, `NodeView.match_shares(selection, metadata, spans=spans)` gives every
  cluster's matching share (counting only the time inside a session range), and
  `cluster_filter.clip_matches(selection, clips, metadata, spans=spans)` marks a cluster's clips
  (`NodeView.clips(node)`; a clip is inside a session range when its midpoint is). `storage.metadata.infer_column_types` says which columns are numbers,
  dates, times of day or categories.
- **Arena / ROI / POI editing**: *drawing* the (time-varying) polygons and the landmark
  points (the GUI's ImageJ-style ROI/POI manager); the CLI is a poor polygon editor.
  `Recording.define_arena()` / `define_roi()` / `define_point()` / `set_rois()` /
  `set_points()` exist for programmatic/scripted creation. Carrying an existing spatial
  setup between recordings *is* in the CLI (`recording export-spatial` /
  `import-spatial` / `import-fov` / `import-calibration`).
  The **two-point distance-calibration pick** is likewise GUI-only, but the resulting
  scalar is settable via `recording set-calibration --pixels-per-mm` (or
  `Recording.set_pixels_per_mm()`), and an existing calibration carries over with
  `recording import-calibration`.
- **Marking keypoint example frames**: clicking each keypoint on a decoded frame, saying it
  is not visible, and seeing the skeleton drawn on the animal while you do it, is the GUI's
  labeling screen (**New example set…** or **Mark frames…** wherever a model needs examples). The CLI samples the frames
  (`keypoints examples sample`), adds a chosen one (`add-frame`) and imports a set prepared as
  a file (`keypoints examples import`, an `examples.yaml` with the images beside it), so a set
  can be scripted or shared; the marks themselves are clicks.
- **Scene geometry editor**: drawing a field of view, a calibration, regions and points
  onto a decoded frame (and seeing them against each other) is a picture, not a value the
  CLI can print. Same "poor polygon editor" reasoning as the arena/ROI/POI editing entry
  above. Every one of them can still be *carried* from the CLI (`recording import-fov` /
  `import-calibration` / `import-spatial`, and the `experiment propagate-*` commands), and
  `recording status` prints the counts (`arenas: N · rois: N · points: N`) so a script can
  confirm something landed.
- **An analysis run's phase rows**: the GUI run page lists every phase a module goes through
  and marks the one in flight. The CLI's `analysis run` progress bar names the current phase
  at the start of its message (`querying · …`) but not the ones still ahead: a row per phase is
  a layout the single terminal line it redraws in place does not have.

- **Query cutoff-by-eye and clip review**: `analysis query-calibrate` gives you the frequency
  curve as numbers and sets a cutoff, which is everything the CLI can honestly offer. Choosing
  a cutoff by *looking at the clips it returns*, scrubbing the matches at a candidate setting,
  watching them appear and disappear as you drag it, and auditing a blind sample to put a
  precision figure on the one you picked, is an iterate-until-it-looks-right loop with a video
  frame in it, so it is a GUI surface (see the guide). The **judging** is GUI-only; the number
  it buys is not, `analysis audit` reports the precision, its interval and its `n`, and
  `AuthoredRun.audit_precision()` returns the same in a script.
- **Cluster review**: `analysis
  temporal-interpret` prints the tree and set the labels. The **reliability audit** on a
  clustering vocabulary is the third choice a run records, and it is GUI-only for the same
  reason as Query's: judging whether a clip really is the thing you named the cluster means
  watching it. Its result reads back with `analysis audit`, like Query's, label by label.
  The API's finer label operations, `create_label`, `label_nodes` and `delete_label`, have no
  flags of their own: they are for scripts that build a vocabulary in steps, and
  `--name-node`, `--rename-label` and `--merge-label` already make every edit they can.
  The **labelled-data browser** (BROWSE: filtering the labelled instances by label, recording
  and metadata, the clickable ethogram, and playing them) is a GUI surface too; the instances
  themselves are a script's through `run.artifact().instances` and the export commands.
  *Looking at* what each cluster is: every clip of it playing at once, most typical, least
  typical or random, and the tree shaded by a filter, is the GUI's interpretation workspace
  (`analysis cluster-montage` writes the same clips as a video, GIFs or a CSV list). The CLI makes the decisions; the frames that inform them live in the
  GUI, like the other clip surfaces.
- **Reviewing a temporal clustering cluster**: watching every clip of a cluster, marking the
  ones that are not its label (or moving them to another label), putting them back, and
  finishing the review all start from watching the clips, so they are GUI surfaces (AUDIT's
  review pane). The API is `TemporalClusteringRun.take_out_clips`, `put_back_clips` and
  `taken_out` for the marks and `finish_review(node)` for the finish, which records every clip
  of the cluster at once. The figures are not GUI-only: on a temporal clustering run
  `analysis audit` prints each cluster's clip-review precision and each label's, then the
  earlier one-by-one audit under its own heading.
- **Behavior-query eval clip review**: `analysis query-eval` writes the metrics and the per-clip
  detail (`--predictions-out` / `--ground-truth-out`), but *watching* the TP/FP/FN clips and the
  held-out query (the scrubbable GT/TP/FP/FN timeline and the clip player in the recorded run's
  **Explore clips…** window) is a GUI surface (the recorded `query_eval` module, run from the GUI
  or `Experiment.run_analysis`). The CLI produces the numbers and the per-clip tables; the video
  review lives in the GUI, like the other clip surfaces.
- **The confidence slider on a detection result**: a prompt-test grid and a Quick segment
  result each carry a slider that re-draws and re-counts the run at a new confidence
  without re-running it: an iterate-until-it-looks-right control, so it lives in the
  [GUI](gui.md). The CLI's equivalent is `--min-conf` on `detection prompt-test` and
  `tools quick-segment`: the same view threshold, chosen up front instead of dragged.
- **Quick-segment overlay playback**: `tools quick-segment` runs the segmentation, saves the run,
  and (`--export`) renders the overlay clip to an mp4; *watching* the overlay play back is the GUI's
  Tools result view. The CLI covers the ends: run and export.
- **Cancelling a dataset, or a training run at once**: `process`, `analysis`, `distill` and
  `export` each take `cancel <id>`; `dataset` does not, and `train` has only `train stop`, which
  finishes the epoch in progress and keeps the best weights. Cancel either at once from the
  GUI's **Cancel** button or with `Experiment.cancel_dataset` / `cancel_model` in Python. This
  is a gap rather than a decision, and the two commands would be thin wrappers over methods
  that already exist.
- **Renaming a recording, and deleting a recording, a stream, an experiment, a processing run
  or an export**: each is done on the GUI page that shows it. In Python they are
  `Recording.set_name`, `Experiment.delete_recording`, `Recording.delete_stream`,
  `Experiment.delete`, `Experiment.delete_run` and `Experiment.delete_export`. Removing one
  marked frame from a keypoint example set is likewise the GUI's labeling screen, or
  `Experiment.remove_example_frame`. These are gaps rather than decisions; the other kinds of
  run and entry do have a CLI `delete` (`analysis`, `comparison`, `dataset`, `train`,
  `distill`, `tools`, `skeleton`, `keypoints examples`, and `model remove`).
