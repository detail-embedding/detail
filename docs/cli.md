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
sam3: ready=True reason=ok
```

`reason` is `ok` (installed and runnable), `not_installed` (fetch it with `model
install`), or one of `no_token` / `no_access` / `bad_token` / `offline`. Each
tells you which setup step to revisit, with a message and an action URL. Add
`--variant <name>` to check one size; without it you get the model's default.
`model check`/`install` aren't experiment state, so `--experiment` is optional (a
throwaway experiment is used if omitted).

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
PROJ=~/Documents/DetailExperiments/study        # the default location → also visible in the GUI

# 1. experiment + recording
detail experiment create study
#   created experiment: /Users/you/Documents/DetailExperiments/study
detail recording create --experiment $PROJ --name recording_001
#   created recording: recording_001

# 2. import a video part
detail recording add-video --experiment $PROJ --recording recording_001 --video clip.mp4
#   added video to recording_001: clip.mp4

# 3. detection: text-prompted. --every-n-frames 2 halves the cost by detecting
#    frames 0, 2, 4, … (omit it to detect every frame)
detail detection run --experiment $PROJ --recording recording_001 \
    --model sam3 --class mouse --prompt mouse --conf 0.1 --every-n-frames 2 --backend cuda
#   submitted detection: det_01
#   detection: det_01 (status=complete)

# 4. tracking: link detections into the subject track (track 0)
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker anchor_tracker --conf-threshold 0.1 --secondary-conf 0.05
#   submitted tracking: track_01
#   tracking: track_01 (status=complete)

# 5. embedding: embed the tracked subject each frame (DINOv3, gated)
detail embedding run --experiment $PROJ --recording recording_001 \
    --tracking track_01 --tracks 0 --backend cpu   # default model: dinov3_vit, vit_large
#   submitted embedding: emb_01
#   embedding: emb_01 (status=complete)

# 6. inspect
detail recording status --experiment $PROJ --recording recording_001
```

Each command blocks until its run finishes, streaming progress lines, then
prints the final status. Run ids are assigned per recording with a per-stage
prefix and counter: `det_01`, `track_01`, `emb_01` (a second run of the
same stage is `…_02`). Downstream commands name the upstream run by id
(`--detection det_01`, `--tracking track_01`), or, for a
**batch** whose recordings carry different ids for equivalent runs, by *kind* with the
`process`/`analysis` `--*-ref` selectors. A ref is either the experiment run that made the
upstream (`--detection-ref run_007`, the name a GUI picker row opens with) or one example run
(`--detection-ref <RECORDING:det_NN>`). Either way it names a set of settings, and each recording
reads its own run with those settings. A `run_NNN` whose recordings' runs do not share one set of
settings (a run recorded before a setting was stored) is refused; name one of its runs as
`RECORDING:run_id` instead.

> **Real subjects, real prompts.** Detection only writes rows where it finds the
> target. With zero detections the detection run still completes, but tracking fails
> with `zero tracks produced`. Use a matching prompt and lower `--conf` if needed.

## Command reference

Options below are the common ones; run `detail <group> <command> --help` for
the complete, current list. Every stage command takes `--experiment` and
`--recording`. Wherever a command takes a recording (`--recording`, `--search`, an
`--example` clip, the `RECORDING` in `RECORDING:track_NN`), give its name or its id. For every
run parameter's bounds, default, and description in one table, see the
[Parameters reference](reference/parameters.md).

### `experiment`
- `create <name> [--location <name> | --dir <parent>] [--camera-view <view> [--camera-view-note <text>]]`:
  create `<name>/` in an [experiment location](concepts.md#where-experiments-live-the-managed-workspace)
  (the default one unless `--location` names another) or in any `--dir <parent>`, which the GUI
  lists only if it is a location. `--camera-view` sets the default
  [camera view](concepts.md#camera-view) (`top_down`, `side`, `angled_down`, `other`); `other`
  needs `--camera-view-note`.
- `locations`: every experiment location on this machine, its folder, and `connected` or
  `not connected` (the drive is unplugged).
- `add-location <name> <folder>`: add `<folder>`, such as one on an external drive, as a location
  experiments can be created in and are listed from; with a name that exists, point it at its new
  folder.
- `remove-location <name>`: stop listing a location. Nothing on disk is moved or deleted.
- `status --experiment <dir>`: identity, whether it is archived, the default camera view, the
  recording list and per-recording run counts.
- `archive --experiment <dir> [--undo]`: hide the experiment from the GUI's **Experiments** list
  (`--undo` puts it back). Nothing in it changes, and every command still works on it.
- `set-camera-view --experiment <dir> <view> [--note <text>]`: set the default camera view every
  recording inherits; `unset` removes it. `--note` describes an `other` view and is refused with
  any other.
- `classes --experiment <dir>`: the experiment's detection classes, each as detected and, when
  renamed, `→` its [name](concepts.md#class-names).
- `rename-class --experiment <dir> <class> <name>`: call a detection class `<name>` on every
  screen. `<class>` is either of its names. `--reset` (in place of `<name>`) puts back the name
  it was detected with. A name another class already has is refused.
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

  The GUI's **Copy to other recordings…** button (beside any of the four setup sections
  on a recording's page) does all four from one dialog, showing which recordings will keep
  what they have before you commit.

### `recording`
- `create --experiment <dir> --name <id>`: e.g. `--name recording_001`. (Note:
  `create` uses `--name`; the other `recording` commands use `--recording <id>`.)
- `backfill-kinematics --experiment <dir> --recording <id> --embedding <emb_NN> [--overwrite]`:
  write the per-frame position and mask shape for an embedding run made before they were
  recorded, from the detection masks (no GPU, no re-embedding; minutes on a long recording). A
  run that has the position but not the shape is brought up to date; one that has both is
  skipped unless `--overwrite`.
- `backfill-motion --experiment <dir> --recording <id> --embedding <emb_NN> [--overwrite]`:
  measure the movement inside each animal's mask on every embedded frame (grooming, licking,
  scratching: what the outline cannot see). One forward read of every video part, no GPU;
  minutes to an hour on a long recording. The cluster profile then has **motion inside the
  mask** and **motion at the ends**. Already-measured parts are skipped unless `--overwrite`.
  There is no GUI button for it yet; the profile names this command when the measures are
  missing.
- `set-camera-view --experiment <dir> --recording <id> <view> [--note <text>]`: give one
  recording its own camera view, overriding the experiment default (for a recording made on a
  different rig). `default` removes the override so it inherits again. `recording status`
  shows the view and whether it is inherited.
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
  anything is wrong. Runs are gated on the same check, so a bad referenced video fails the
  run before it starts.
- `relink --experiment <dir> --recording <id> --part <N> --video <file>`: point a referenced
  part at a moved/renamed file. Accepted only if the file's content matches what was recorded
  (a mismatch is a changed video → remove and re-add). For a whole moved library, re-point the
  [video root](#video-root) instead. On an experiment made on another computer, relinking one
  file registers its video root here, so every recording under that root finds its video too.
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
  from an exported CSV. A name the recording already has is **refused by name** and
  reported; the ones that do not collide still arrive (regions and points are sets, so one
  collision does not refuse the rest). To change one, delete it here first: it is never
  overwritten, and no name is silently suffixed.

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
  misdescribe the region those runs actually filtered on.
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
- `status --experiment <dir> --recording <id>`: video parts, resolution/fps, spatial
  definitions (arenas / ROIs / points), streams, and run history:

  ```
  recording: recording_001
    video parts: 1
    resolution: 160x120 @ 30.0 fps
    arenas: 0  ·  rois: 0  ·  points: 0
    detection: det_01 status=complete
  ```

### `detection run`
`--model <id>` is any detection model: built-in (`sam3`), experiment-trained
(`model_NNN`), or imported (`model import` / `model list`). Built-ins also need
`--class` (the stored label) + `--prompt` (the search text), optional `--variant`
(if the model has size options); trained/imported models are closed-set and take neither.
Plus `--conf` (default `0.5`): a real filter here: a detection run stores only what
clears it, and changing your mind means another run (unlike `prompt-test`, which keeps
everything and lets you re-threshold the view). `--class-conf NAME=VALUE` (repeatable)
overrides `--conf` for one class at a time; a class with no override still uses `--conf`.
Plus `--batch-size <N>` (frames per GPU forward pass;
omit for the model default, `4` for YOLO and `1` for SAM3, bounded by GPU memory),
`--every-n-frames N` (process every Nth source frame, 1, the default, is every frame; 4
detects frames 0, 4, 8, … for roughly a quarter of the cost), and
[compute flags](#compute--multi-gpu). Writes `det_NN`.

`--teacher-window-s S` detects only the recording's first `S` seconds, for a keypoint teacher:
one short window of each camera or box is enough to teach a keypoint model, and DINOv3 on hours
of video is not. Tracking, `track merge` and keypoints on that run keep the same window (a
merge needs every track it merges to cover the same span). Nothing else accepts a teacher run:
embedding and location export refuse it and say so, because their results would quietly cover
only those first minutes. Run pickers label it
`first 5 min (teacher)`. Not on the GUI launch form: in the GUI, **Set up keypoints…** on the
Models page sets it for you.

`--every-n-frames` counts **source** frames, and rows keep that numbering, so a downstream
stage can only go coarser by a multiple of it: after detection at 2, tracking and embedding
may use 2, 4, 6, … Asking for 3 is refused rather than rounded: those frames were never
processed. `tracking run` and `embedding run` take the same flag; omit it there to keep
every frame the stage above produced.

### `detection prompt-test`
A fast QC loop *before* committing to a full `detection run`. For an **open-vocab**
model (`sam3`), compare candidate prompt phrasings with `--class <label>`
(applied to every candidate) and one or more `--prompt <text>` (repeatable, one
phrasing per candidate). For a **trained/local** model (`--model model_NN`, closed-set),
omit `--class`/`--prompt`; it previews its own classes, one result column each.
Narrow with `--show-class <name>` (repeatable; default all). Common flags: `--n`
(frames to sample, default `8`), `--min-conf` (default `0.5`), optional `--variant`,
`--seed`, and [compute flags](#compute--multi-gpu). `--start` / `--end` (seconds of
recording time) draw the sample only from that stretch; either one alone leaves the other
end open. A stretch with fewer frames than `--n` is refused and the error says how long the
recording is. Prints a
per-column detection summary; `--save-as <name>` also writes overlay JPGs + a
`test.json` under `detections/prompt_tests/<name>/`. Creates no run.

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
`0,1`, or subject names, e.g. `black mouse,white mouse`, for a merged run: see
[Merge tracks](#merge-tracks) below) or `--all-tracks` (embed every track in the run; mutually exclusive),
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
manage them. A frame with no heading of its own takes one interpolated across a gap no longer
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

**Pose in the embedding.** `--pose draw` paints the keypoint track run's points onto the image
the model sees, so one vector holds the mask and the pose together; `--pose append` writes the
pose as numbers (each point's place on the body, turned so the heading points right) after the
mask's vector, and needs `--heading`. `--pose none` (the default) is the mask alone. Either
needs `--keypoint-run` and points on every frame the run embeds. `--pose-weight` (append only;
1 gives the pose numbers equal say with the mask's vector, 0.5 half, 2 twice) sets the balance
when frames are compared, and `--pose-min-score` leaves out a point scoring below it (omit it
for the keypoint track run's own cut-off).


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
postures and picks the ones that differ (on an empty set, the `--typical` most typical ones first,
5 unless you say); `spread` takes
one frame per equal time slice; `heading` takes `-n` frames *per direction* the animal faces as
the camera sees it (toward, away, left, right, the diagonals, rearing; `--bucket` to choose some),
without a floor by default. `--floor-class` names a detection class that is the arena floor (a
hot plate) when your detector segments one: from a 45° or side view the floor is squashed in the
picture, so walking toward or away from the camera barely moves on screen, and the floor's
outline lets that movement be measured on the floor itself. Without it the toward and away
directions are filled less reliably; left and right are unaffected.
`varied` and `heading` read `--from-recordings` of the recordings, split evenly across the values
of a metadata column with `--spread-by`. A sample decodes frames, so it shows a progress bar;
`heading` reads each recording once, the first time, and keeps what it read. `heading` is for
a camera that looks across the arena: when every recording you name has the camera view
`top_down`, it is refused, since from above the animal never faces toward or away from the
camera. A recording with no camera view set is not refused (`recording set-camera-view`).

```bash
detail keypoints examples sample ex_01 --recording M09_CP1 --recording F13_HP2 \
  --strategy heading -n 3 --floor-class plate --spread-by acquisition_setup \
  --experiment my_experiment
detail keypoints examples reject ex_01 4 --bucket toward --experiment my_experiment
detail keypoints examples remove-frame ex_01 2 --experiment my_experiment
```

`reject` takes a frame out of the set and never samples it again: for one direction with
`--bucket`, or at all without it. `remove-frame` only takes it out. A set's
frames are copied in as PNGs; the marks are made in the GUI's labeling screen (GUI only, see
*Not in the CLI*), or arrive with a prepared file:

```bash
detail keypoints examples import shared/examples.yaml --experiment my_experiment
```

The file is an `examples.yaml` as the tool writes it (`schema_version` first, the skeleton, then
one entry per frame with its `image` path relative to the file and its `marks`), with the images
beside it; `examples export` writes exactly that. A bare skeleton file imports too, as an empty
set. Deleting a set a keypoint run lists is refused, so a run always finds its frames.

**Detect.** One run per selection of recordings:

- `keypoints detect --experiment <dir> --recording <id> [--recording <id> …] --model <method>`
  plus **one** of `--detection <det_NN>` (that exact run id, complete in every selected
  recording) or `--detection-ref <RECORDING:det_NN>` (derive the detection kind from one run
  and resolve each recording's own match). `--model` is a registered method, a keypoint student
  trained in this experiment, or a keypoint model from the machine's catalog (`model list`
  shows them with `role=keypoint`), looked up in that order; the run records which
  (`weights_source`: `trained` or `imported`). A method that works on each animal (`dinov3_match`)
  crops around every detection on the frame; a trained student looks at the whole frame.
  `--class <name>` limits a method that works on each detection to the detections of that
  class (for example `--class mouse` when the detector also finds the arena); a frame with no
  detection of it is skipped. Left out, every detection gets a pose. The class must be one the
  detection run found, and a method that looks at the whole frame refuses the option. The
  method `dinov3_match` learns from an example set, so it needs `--example-set <ex_NN>`. When
  the animals are told apart at detection and look different (a white and a black mouse as two
  classes), give each class its own set instead: `--class-examples Resident=ex_02
  --class-examples Intruder=ex_03`, once per class, in place of `--class` and `--example-set`.
  Each class's detections are matched against its own set, poses go only in the classes named,
  and every set must be on the same skeleton. The method's
  settings (`--decode`, `--side-rule`, `--crop-px`) can be left out, in which case they are
  resolved against the set's skeleton and recorded on the run. A trained student scores each
  whole pose, and `--min-pose-confidence` (default 0.25) is the lowest score it keeps; a pose
  below it is not stored, so changing it means running again. The patch method has no such
  score and refuses the option. `--every-n-frames` must be a multiple of the detection run's;
  `--batch-size` only changes speed. Writes `run_NNN`, and in each recording a `kd_NN` run
  under `keypoint_detect/` holding, per video part, one row per pose: its frame, the box it was
  placed in, the points, their scores and the method's candidates. No pose belongs to an
  animal yet. `process keypoint_detect` is the same command.
- `--backend remote --remote-target <name>` runs it on a registered cluster (see
  [Compute & multi-GPU](#compute--multi-gpu)).

Besides `dinov3_match`, `--model` takes methods that need no example set: the two DeepLabCut
SuperAnimal models, `superanimal_topviewmouse` (mice filmed from above) and
`superanimal_quadruped` (four-legged animals filmed from the side), each an optional download
(`model install`) whose weights are for academic, non-commercial use; a SLEAP model you imported
(`model import --keypoints`, under [`model`](#model)); and a student you distilled (below). Each
places the keypoints it was trained on, with their names, and takes no `--example-set` or
matching options. A SuperAnimal or SLEAP model works on each animal, like `dinov3_match`.

A keypoints run made before keypoint detect and track existed (`kp_NN`, from the old `keypoints
run`) still opens and exports with the keypoint targets, read as a track run. It cannot be
launched again.

**Track.** Give each pose to a tracked animal, then settle the points over time. It uses one CPU
core per recording, here or on a cluster, at about 200 animal-frames a second: about 4 minutes
for 13 minutes of two-animal video at 30 fps, most of it giving poses to animals by mask:

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
- Here, a run works on several recordings at once: one per core for half this machine's cores,
  at most 8 (eight recordings took 50 s instead of 200 s one at a time on a 16-core machine).
  `DETAIL_MAX_KEYPOINT_TRACK_PARALLELISM=N` sets the number instead.
- `--backend remote --remote-target <name>` runs each recording as its own job on a registered
  cluster, up to the target's concurrent jobs at once (see
  [Compute & multi-GPU](#compute--multi-gpu)). `--backend` is `cpu` (the default) or `remote`:
  there is no model, so there is no GPU to pick.
  `--method-param track_points=nose,neck_base,tail_base` tracks only those points (any of the
  detect run's point names, comma separated) and leaves every other point exactly as detected;
  a name the skeleton does not have is refused with the names it does have.
- `keypoints track-tune --experiment <dir> --example-set <ex_NN>` plus **one** of `--detect
  <kd_NN>` or `--detect-ref <RECORDING:kd_NN>`, the tracking run and matching as for `keypoints track`
  (`--tracking`, `--tracking-ref`, `--match-min-inside`, `--match-mask-grow`; left out, the
  one tracking run on the detect run's detection run), and optionally `--method-param
  NAME=VALUE` (the settings to start from) and `--seed`: tries 36 `temporal_decode` settings (transition scale,
  smoothing, despike) against the example set's labelled frames and recommends one. Use frames
  the detect run did not learn from: its own example set is refused. Tracking is re-run in
  memory on the videos holding those frames; nothing is written. The labelled recordings are
  split in two, one half picking the setting and the other checking it, and the table prints
  PCK@0.1 on both halves (all points, then the check half's left/right points and midline) beside
  detection alone and your current settings. Quote the check figures. A last row may leave some
  points as detected (`track_points`), where detection placed them better on the picking half.
  The last line is the `keypoints track` command that applies the recommendation.
- `keypoints track-preview --experiment <dir>` plus **one** of `--detect <kd_NN>` or
  `--detect-ref <RECORDING:kd_NN>`, and the tracking run, matching, `--method` and
  `--method-param` exactly as for `keypoints track`: tries your settings on a few seconds of a
  few recordings, without labels and without a run. `--recording` (up to three; left out, two
  of the recordings the settings cover) picks the recordings, `--start RECORDING=SECONDS` where
  each window starts (left out, the middle of the recording) and `--duration` how long every
  window is (30 s by default, at most 120). `--variant 'NAME=VALUE NAME=VALUE'` (up to two,
  needs `--method`) adds settings to try beside yours, e.g. `--variant 'transition_sigma_body=3
  smooth=false'`. Each window is assigned once and every setting tracked on it in memory, so a
  30-second window takes seconds rather than a full run's minutes. The table prints, for
  detection alone, your settings and each variant: the share of points placed, the median move
  from one frame to the next, jumps of more than half a body length in one frame per minute,
  left/right swaps per minute, how much the skeleton stretches, and **agreement**: of the points
  detection was most sure of, the share the setting keeps within a tenth of a body length.
  **These describe, they do not rank**: smoothing lowers the moves, jumps and swaps whether or
  not the points stay on the animal, so read each row against detection alone (the first; its
  agreement is 100% by definition) and watch the clips in the app. The last lines are the
  `keypoints track` command for each setting. Nothing is written.
  `--grid` (in place of `--variant`, `temporal_decode` only) searches every *Tune tracking*
  setting instead: six transition scales, smoothing off or at two windows, despike off or on,
  36 settings over the same windows in about the time of a preview. Settings no other setting
  beats on agreement, placed, jumps and swaps together are starred and listed first; that is a
  provisional rule, **not yet checked against labels**, and on footage where detection jitters
  more than a tenth of a body from frame to frame every setting's agreement sits close together,
  so the star separates little. Commands print for your settings and the starred ones.

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
frame within `--exclude-audit-s` of one, are never used. `--student-input` says what the student
sees: `whole_frame` (the default) trains on the whole video frame with one label per tracked
animal, so one pass per frame serves every animal; `crop` trains on a square crop around each
tracked animal, resized to `--student-crop-px`, for overlapping animals or a small animal in a
wide view. The student is run the way it was trained. The dataset also keeps what the teacher
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
  (left out, the method decides, as on a run). `--class-examples CLASS=SET`, repeated, tests one
  set per class in place of `--example-set`, as `keypoints detect` takes it; the held-out check
  then scores each class's frames against that class's other frames. Shows a progress bar
  through loading, labelling and the held-out check.
- `keypoints distill --experiment <dir> --name <name> --n-frames N --teacher patch|detect`.
  For `patch`: `--example-set ex_NN` (or one set per detection class, `--class-examples
  CLASS=SET` repeated, for animals told apart as classes that look different, labelled on a
  merged tracking run), and the recordings to label as `--recording <name>`
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
  `--patience`, `--train-backend` / `--train-device`. `--train-backend remote
  --train-remote-target <name>` trains on a registered cluster: the dataset is cut on this
  computer and goes up with the experiment, so the cluster needs no video. One bar folds
  `labeling → training → registering`; Ctrl-C detaches and `distill attach` rejoins. Writes `distill_NNN` with its
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
your experiments that copied it. While anything does, `edit --skeleton` and `delete` are refused
with that list; `save-as-new` keeps the edit as a new skeleton instead. A rename is always allowed.

### `keypoints setup`
Build a keypoint model for the whole experiment in one run: a teacher taught on one short
window of each acquisition setup (each camera or box), then a fast student distilled from it.
Fill in **acquisition_setup** for every recording first (see
[Recording information](#recording-information)).

```bash
detail keypoints setup plan --experiment my_experiment
detail keypoints setup start --experiment my_experiment --name "rat pose" --detector model_003 --class rat
```

`plan` prints the recording each setup will teach from (the first of that setup, and how much
of it is used) and its alternatives. Swap one with `--teacher "box 1=box1_day2"` on `start`
(repeatable).

`start` needs a distilled detector (`--detector`, a `model_NNN` from `distill run`) and which of
its classes gets keypoints (`--class`). It detects and tracks the first `--window-s` seconds of
each teacher recording (300 by default; a shorter recording is used whole), then **stops and
waits for you**: it prints the tracking runs to mark an example set on (see **Example sets**
under [`keypoints`](#keypoints); marking is done in the GUI) and the command that finishes it:

```bash
detail keypoints setup continue --experiment my_experiment --run distill_004 --example-set ex_01 --score-cutoff 0.4
```

`continue` runs the teacher (`--model`, `dinov3_match` by default) with that example set on
frames sampled from the teacher's tracks, then distills a student from them, as
`keypoints distill --teacher patch` does. `dinov3_match` has no default cut-off, so
`--score-cutoff` is required; `keypoints test --teacher patch` on the teacher's tracking runs
shows where to put it. An example set already marked on this animal can be given to `start`
with `--example-set` (and `--score-cutoff`): the run then goes straight through without
stopping.

**Two or more animals** that the detector tells apart as classes (a resident and an intruder):
repeat `--class`, once per animal. Each class is tracked on its own, the tracks are merged into
one run per teacher recording, and the run waits for **one example set per animal**, so animals
that look different are each matched against their own clicks. Finish it with one
`--class-examples` per class instead of `--example-set`; one student is trained for all of them:

```bash
detail keypoints setup start --experiment my_experiment --name "pair pose" --detector model_003 --class Resident --class Intruder
detail keypoints setup continue --experiment my_experiment --run distill_005 --class-examples Resident=ex_02 --class-examples Intruder=ex_03 --score-cutoff 0.4
```

`start` takes `--class-examples` as well, to go straight through.

`start` also takes the teacher's settings (`--model-variant`, `--decode`, `--side-rule`,
`--crop-px`) and the dataset and training options of `keypoints distill`, including
`--student-input`. Each part has its own compute: `--detect-backend` (and `--detect-device`,
`--detect-remote-target`), `--label-backend` / `--label-device` for the teacher, which runs on
this computer, and `--train-backend` (and `--train-device`, `--train-remote-target`). The run
survives the CLI closing; `distill list`, `status`, `attach` and `cancel` manage it like any
distill, and `status` reads `awaiting_references` while it waits.

### `process`
Experiment-level processing runs span a **recording selection** and record a `run_NNN`,
the unit a dataset is built from. (The recording-scoped `detection run` /
`tracking run` above do **not** create one.) `--recording` is repeatable.

**A launch shows its own progress before the run starts.** Those same commands read every
selected recording before they start anything, and each pass draws a bar naming the recording
it is on: `Resolving shared settings` (an embedding run that reads each recording for its
background colour), `Validating recordings`, then `Estimating disk space`, so a large selection
or a slow drive is visibly working. The run's own bar follows once it starts.

**Free space is checked before a run starts.** Every command that launches or retries a run
(`process detection`, `tracking`, `track-merge`, `embedding`, `keypoint_detect`, `retry`)
estimates what the run will write and refuses when that, plus 5 GB kept spare, does not fit
on the drive the experiment is on. The refusal names both numbers and the drive. After a
launch it prints `estimated output: … ; … free at launch`. `--ignore-free-space` launches
anyway. Remote runs are checked too, because their output is copied back into the
experiment. Typical sizes per hour of video: embedding about 0.4 GB per tracked animal (a
1024-wide model; it scales with the model's width and the number of tracks), detection about
20 MB, keypoints about 10 MB, tracking almost nothing. Under WSL, the drive checked is the
Windows drive holding WSL's disk whenever that has less room than WSL reports: an experiment
inside WSL lives in one file on that drive, so a full `C:` stops WSL itself.

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
  per-recording run ids it created. A detection run also gives each recording's
  `frames_with_detections=<with ≥1>/<processed>`.
- `process attach <run_NNN> --experiment <dir>`: stream a live run's progress (it keeps
  running with the CLI closed). Ctrl-C detaches; it does not cancel.
- `process retry <run_NNN> --experiment <dir>`: re-run the run's **incomplete** recordings
  into the same record. Every recording that isn't complete (failed, cancelled, or never
  launched) runs again under a fresh per-recording id; completed ones are left alone, and
  the run rolls up to `complete` once all of them are. A bar names each recording as it is
  re-checked; the command refuses if one no longer validates, naming each. A recording is re-run whole (there is no
  resume-from-mid-recording).
- `process cancel <run_NNN> --experiment <dir>`: stop a running run's coordinator and its
  in-flight recordings. Finished recordings keep their results; `process retry` picks up the
  rest.

`detection`, `tracking`, `track-merge`, `embedding`, `keypoint_detect` and `keypoint_track` have
`process` subcommands.

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
- A keypoint dataset (`dataset keypoints`) trains a YOLO-pose model: give a `-pose` base model
  (`yolo11n-pose.pt` and up). A base model that cannot learn the dataset's labels is refused.
- `list` / `status <model_NNN>` / `delete <model_NNN>`: inspect and remove trained
  models (`delete` confirms unless `--yes`). `list` shows each model's `role`: `detection`, or
  `keypoints` for a pose model. `status` prints a `trained with` line: the
  Ultralytics version and the settings asked for. The model's `model.yaml` also keeps every
  setting Ultralytics resolved (`trained_with.args`).
- `attach <model_NNN> --experiment <dir>`: stream a live training run's progress (it keeps
  running with the CLI closed). Ctrl-C detaches; it does not cancel.
- `promote <model_NNN> --experiment <dir> [--name <name>]`: copy a completed trained model into
  the machine-global model catalog so it's selectable in *any* experiment. Records its provenance
  (source experiment, dataset, base model, metrics); the original stays put. A keypoint student
  is promoted with its skeleton, which is also added to the skeleton library unless one with the
  same points, edges and pairs is there; it appears in `model list`, and
  `keypoints detect --model <id>` then finds it in any experiment.

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
  `--stage-videos` (needs a remote labeling or training target) copies every source recording's
  video to the cluster **in the background** while labeling and training run, so the detection run
  you start afterwards finds it there instead of uploading it first. Off by default: it can be 100+
  GB on the cluster's storage. It goes to the training target (else the labeling one); watch it with
  `detail remote staging-status`, stop it with `detail remote staging-stop`.
- `attach <distill_NNN>` / `resume <distill_NNN>` / `cancel <distill_NNN>`: rejoin a run, resume a
  failed/cancelled one (reuses a complete dataset, resumes a cancelled or crashed training), or stop a running one.
- `list` / `status <distill_NNN>` / `delete <distill_NNN>`: inspect and remove distill runs.
  `delete` removes only the run record: its dataset and trained model stay (delete those with
  `dataset delete` / `train delete`). `status` prints the run's `kind` (`detection`, or
  `keypoints` for one launched by [`keypoints distill`](#keypoints), whose third stage,
  `register`, it prints too; `keypoint_setup` for one launched by
  [`keypoints setup`](#keypoints-setup)).

### `locations export`
`--experiment`, `--recording`, `--tracking track_NN`, `--out <dir>`: a per-recording export
straight to `--out`, writing `tracks.csv` / `roi_relations.csv` / `point_relations.csv` /
`pair_relations.csv` + a `locations.yaml` manifest (per-track position relative to the
recording's ROIs/POIs, and the distance between every pair of animals, keyed by `t_ns`). No run
record. For a recorded, batchable version, see `analysis run` below.

- `--keypoints kt_NN` also writes `keypoints.csv`: every frame × animal × point of that
  keypoint track run, in normalized, pixel and mm units, with the point's score and whether it
  is at or above the cut-off. The run must come from the same tracking run.
- `--pair-edge-distance` also measures the nearest distance between two tracks' masks (0 when
  they touch). It compares the masks on every frame, so it is slower; off by default.

`pair_relations.csv` has one row per frame per pair of animals both tracked on it. A recording
with one animal gets the header only.

### `locations plot`
Draw a completed location-export analysis run: paths, occupancy, speed over time, or the path in
3-D.

```bash
detail locations plot --experiment ./study --run an_03 --out figs/paths
detail locations plot --experiment ./study --run an_03 --out figs/heat --kind heatmap --group-by genotype --format png --format pdf
```

`--run an_NN` and `--out <path without extension>` are required. `--recording` (repeatable)
narrows the plot; the default is every completed recording in the run. `--format png|svg|pdf`
(repeatable, default `png`; `html` writes one self-contained interactive page you can hover,
zoom and, in 3-D, turn, which opens in any browser offline) and `--theme light|dark`. The plot
options are `--kind
trace|heatmap|speed|trace3d|distance|distance_hist`, `--source mask|bbox|<keypoint>`, `--point
<keypoint>` (repeatable: several points on one path plot, one colour each), `--units mm|px|norm`
(default: mm when every plotted recording is calibrated, otherwise px), `--color-by
auto|time|speed|subject`, `--subject <label>` (repeatable; the default is every animal),
`--group-by <Recording Information column>`, `--split-by-subject` (also group by animal, or by
pair on a distance plot), `--pair "<label> – <label>"` (repeatable), `--distance-by
mask_centroid|bbox_centroid|mask_edge`, `--hist-bins`, `--bins`, `--time-bin-s`,
`--speed-clip-pct`, `--elev`, `--azim` and `--max-points`. `--help` lists each with its range and
default.

A keypoint `--source` or `--point` needs a run exported with `--keypoints`; a point under the
cut-off is left out of the path. `distance` is the distance between two tracks over time and
`distance_hist` how often they were at each distance, a line per pair (`mouse – object`). In a
group, each pair counts once. `--distance-by mask_edge` needs a run exported with
`--pair-edge-distance`.

Every animal in a recording is plotted, named by its tracking run's label (`white mouse`), or
`subject 0`, `subject 1`, … when the run has none. `--subject` takes the label and finds that
animal in each recording, whatever its track number there. In a group, each animal counts once.

```bash
detail locations plot --experiment ./study --run an_04 --out figs/speed --kind speed --group-by genotype --split-by-subject
detail locations plot --experiment ./study --run an_04 --out figs/apart --kind distance_hist --group-by genotype
```

### `locations montage`
Play a location-export run's recordings side by side, one tile per recording with every animal
in its colour, and write it as a GIF or MP4. One `--recording` is a preview of that recording.
Each tile is captioned with its recording's name; a name wider than the tile is cut in the middle
(`20251103_Ra…Shot1_ArenaE`), so the end that tells two recordings apart stays.

```bash
detail locations montage --experiment ./study --run an_04 --out montage.gif --start-s 60 --length-s 20
detail locations montage --experiment ./study --run an_04 --out last.png --last-frame --raw --keypoints --keypoint-trail
```

`--run an_NN` and `--out <file>` are required: `.gif` or `.mp4`, or `.png` with `--last-frame`.
An mp4 is H.264 unless `--codec mp4v` says otherwise ([Video format](#video-format-codec)).
`--recording` and `--subject` (both repeatable) narrow what plays; the default is every recording
and every animal. `--start-s` and `--length-s` set one window for every recording, all in sync;
`--full-length` plays each recording from its first to its last tracked frame instead, so they end
at different times. `--full-frame` shows the whole frame instead of cropping to the animals.
`--speed` (0.25 to 4) plays faster or slower than real time. The layers are `--raw`, `--mask` /
`--no-mask`, `--keypoints`, `--mask-trail` / `--no-mask-trail`, `--keypoint-trail` and
`--trail-point <name>` (repeatable). `--last-frame` writes the last picture with every trail
covering the whole clip, one picture of where each animal went. `--width` caps the picture's
width in pixels.

The montage reads the recordings' videos and tracking runs, so a recording deleted since the
export is left out and named.

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
anything ([below](#how-big-and-how-long)). The subject is whichever of `--recording` /
`--run` / `--analysis-run` you pass; pass none for the experiment itself.

Figure options, for every target that writes a figure (`instances`, `ethogram`, `report`):
`--theme light|dark`. Figures are light, for print, unless you pass `--theme dark` (for slides).

Clip options, for `--target clip` and `--target clips`: `--render raw|overlay|mask`
(repeatable), `--crop` / `--full-frame`, and `--size <px>`. `--target clip` also takes
`--frames START:END` (the inclusive frame range to write) plus `--track <id>` and
`--embed-run <emb_NN>`:

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
  one named kind it has. `--tracking-for <RECORDING:track_NN>`, repeatable once per
  recording (the recording by id or name), reads exactly that run there, whatever the other
  flags pick. It is for a recording whose tracking was redone beside a bad run:
  `--tracking track_01 --tracking-for "larva 26:track_02"` reads `track_02` there and
  `track_01` everywhere else, where `--tracking track_01` alone would read the bad run and
  naming both kinds with `--tracking-ref` is refused. With `--tracking-for`, `--tracking` /
  `--tracking-ref` may be left out, and the one kind the other recordings share is used. The
  run named must exist and be complete. `--keypoints <kt_NN>` / `--keypoints-ref
  <RECORDING:kt_NN>` add `keypoints.csv`, and `--keypoints-for <RECORDING:kt_NN>` does the
  same per recording; its poses must have gone to the tracking run read there. Runs a
  per-recording analysis module across the selection (writes `an_NN`). For
  `location_export` (the default), output lands in `analysis/<an_NN>/<recording_id>/` as
  the three CSVs. Same output as `locations export`, but recorded, with progress + batch.
- `query --experiment <dir> --search <id> [--search <id> …] --embed-run <emb_NN> --method <id> [--method-param <name>=<value> …] [--behavior <name> …] --example <[name=]recording:start-end> [--example … | --from-csv <clips.csv>] [--non-example <[name=]recording:start-end> …] [--track 0] [--split-gap <s>] [--min-match <s>] [--strict]`: search a cohort for behaviors you can name, from a few example clips each. One run searches **several behaviors**: declare names with `--behavior rear --behavior groom` and attach clips with a `name=` prefix (`--example rear=s1:0:12.4-13.1`); a bare `--example` makes one default behavior. Spans are **seconds** by default (`s1:0:12.4-13.1`, `recording[:track]:start-end`) or **frames** with an `f` suffix (`s1:0:1240f-1310f`). `--from-csv` reuses a prior clip CSV's rows as examples, which is also how you reuse a previous run's. Writes one opaque shard per (behavior × recording × subject) under `analysis/<an_NN>/search/`. **There is no threshold flag**: the search stores no cutoff, so choosing one afterwards costs nothing and never a second search. The run therefore finishes **open** rather than complete: it stays open until you close it. Turn the results into behavior instances with `analysis query-calibrate`, below. `--method` defaults to [`soft_knn`](modules/soft_knn.md), which is fast; its page says what is and is not known about how well it retrieves. `--min-match <s>` sets the shortest match the search returns, in seconds (default 0, no minimum). It is applied while matches are formed, so a brief spike cannot crowd out a longer match beside it, and changing it means searching again.

  The method's own parameters are set with `--method-param <name>=<value>`, repeatable,
  `--method-param k=30 --method-param duration_max_mult=3`. There is no generated
  option per parameter because `--method` is chosen at parse time; each method's page
  ([soft_knn](modules/soft_knn.md)) lists what it takes, with bounds and defaults. A name the method does not have is
  refused rather than ignored. `--non-example` marks a clip that is *not* the behavior, in the
  same spec as `--example`; only a method whose negatives are `optional` or `required` can use
  them, and a run that passes them to one that cannot is refused with the method's reason.
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
  behaviors as one behavior-stream CSV per searched recording. Each row carries
  `origin: predicted`, so re-importing one can never be read back as ground truth. A recording
  that matched nothing gets a header-only file: the zero is the finding.

  The artifact is pinned as it is exported and the snapshot id is printed, so a figure
  built from these CSVs can cite something that survives a later re-cut of the run.
- `temporal-clustering --experiment <dir> --recording <id> [--recording <id> …]
  [--embed-run <emb_NN> | --embed-ref <RECORDING:emb_NN> [--embed-ref …]] [--embed-for <RECORDING:emb_NN> …] [--track 0] [--method codebook_viterbi]
  [--detector binseg] [--min-bout <s>] [--band <s>] [--max-bouts <n>]
  [--fit-share <percent>] [--codewords <n>] [--min-run <s>] [--seed <n>]
  [--method-param <name>=<value> …] [--remote-target <name>]`: discover a vocabulary
  of **behaviors** pooled across the selected recordings. With
  `--method dtw_finch`, each trajectory is cut into bouts at the moments the pose changes, every
  pair of bouts is compared, and the comparison is grouped; the default, `codebook_viterbi`, is
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

  **A run redone beside a bad one.** `--embed-for <RECORDING:emb_NN>`, repeatable once per
  recording (the recording by id or name), reads exactly that run there, whatever
  `--embed-run` / `--embed-ref` pick for the rest. The run must be complete and comparable
  with the others.

  For `dtw_finch`, comparing every pair is what lets a slow and a fast performance of the same
  behavior match, and it is why a fit is expensive: the cost grows with the **square** of the bout count.
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

  `--seed <n>` is `codebook_viterbi`'s seed for its k-means grouping: the same seed on the same
  recordings gives the same vocabulary. Any other parameter the method declares (for example
  `window_s` or `max_gap_s`, or `dtw_finch`'s `pool` and `penalty`) is set with
  `--method-param <name>=<value>`, repeatable; the [codebook_viterbi](modules/codebook_viterbi.md)
  and [dtw_finch](modules/dtw_finch.md) pages list them with bounds and defaults. A parameter given
  both as its own option and as `--method-param` is refused.

  After the fit, the run prepares the evidence every cluster shows in the GUI, however small the
  cluster: counts, time per recording, positions, and the list of its clips with how typical each
  one is. Nothing is decoded from the video at this step. Time is measured from the recordings' own
  timestamps.

  The fit leaves the run **open**, not complete: a vocabulary is not a result until somebody has
  looked at it and labelled things. Interpret it with `temporal-interpret`.

  **On a cluster.** `--remote-target <name>` runs the fit on a registered cluster (`detail remote
  list`), which frees this computer for the hours a large study takes. Each recording's embedding
  run is sent to the cluster again first, so the fit reads what this computer has, and the
  recordings' videos are copied there (once; later runs reuse them). The
  run comes back here when the fit is done, open, as if it had run here. The work is CPU, so the
  job asks for one GPU, the fewest the cluster accepts, and that share of a node's CPUs and memory.

- `temporal-interpret --experiment <dir> --run <an_NN> [--tree] [--name-node <node>=<label> …]
  [--rename-label <old>=<new> …] [--merge-label <from>=<into> …]
  [--rebuild-cache] [--finalize] [--reopen]`: label the clusters you care
  about. The fit found a whole tree of clusters, from one holding everything down to the finest
  groups, and **a label goes on any node of that tree**. Every operation is quick: it rewrites the
  run's state and rebuilds the behavior output from a fit already on disk; nothing re-fits or re-compares. Each
  command ends with a `coverage:` line per recording: how much of the video was analysed, and how
  much of that is in a bout. A capped fit prints a `note:` line first, every time. Time tracking
  lost is in neither: a bout on both sides of a gap is two instances, never one spanning it. A
  temporal-clustering run whose results were written before that rule (2026-11-08) can still hold
  instances that span a gap, and so report more than 100%; its results are corrected the next time
  they are rewritten: any label change, or `analysis rebuild <an_NN>` (`--reopen` a finished run
  first).

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
  `scaling`. Both take names and are repeatable. Every edit on the command line is checked
  before any is written, and each reports its progress. The name `other` is refused: it is what
  the GUI's labels panel calls a label's own remainder.

  `--rebuild-cache` rebuilds every cluster's evidence from the fit, with a progress bar: for a run
  whose clips are reported out of date, or one prepared before every cluster listed its clips
  (such a run still opens; its clips just have no typicality order until it is rebuilt). It does
  not change any label and is allowed on a finished run. `--min-cluster-s` is gone: there is no
  size floor, and a run saved with one ignores it.

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

- `export-temporal-clustering --experiment <dir> --run <an_NN> --out <dir>`: write one
  behavior-stream CSV per fitted recording. Each row carries `origin: predicted`, so re-importing
  one can never read back as ground truth.

  The artifact is pinned as it is exported and the snapshot id is printed, so a figure
  built from these CSVs can cite something that survives a later re-cut of the run.

- `profile --experiment <dir> --run <an_NN> --node <node> [--compare rest|sibling|node]
  [--against <node>] [--explainer <id>] [--time-budget <s>] [--n-per-side <n>]
  [--margin <0-1>] [--keep-place] [--seed <n>] [--no-save]`: which measures set one node
  apart, the GUI's **Cluster profile** window as text. The first line is the verdict: balanced
  accuracy on held-out recordings beside the shuffled floor, and whether it clears the margin,
  or, for a cluster whose members all come from one recording, why there is no such score.
  Then what it was based on, what it skipped, each measure's direction with both medians, the
  plain rules, and the context. `--compare node --against <node>` compares it with one other
  cluster instead, to ask whether two clusters in different branches are the same behaviour;
  neither may hold the other. It also prints how well position and facing alone tell the two
  apart: when the body measures cannot and they can, the two are one behaviour seen in another
  place or direction, and `lookalikes` groups them. Where the animal is and which way it faces are measured but left
  out of the fit, so rearing toward the left and toward the back read as one behaviour;
  `--keep-place` puts them in. Kept with the run (`profiles/node_<id>/`, and a comparison with
  another cluster under it in `against_<other>/`) unless `--no-save`. The default explainer needs `shap`
  (`detail install-extra analysis`); without it the forest falls back to permutation
  importance and says so. Your labels are never read.
- `profile-all --experiment <dir> --run <an_NN> [--clusters floor|all|leaves|ids]
  [--node <node> …] [--n-per-side <n> | --use-time-budget [--time-budget <s>]]
  [--compare rest|sibling] [--keep-place] [--explainer <id>] [--margin <0-1>] [--seed <n>]
  [--redo] [--remote-target <name>] [--estimate]`: `profile` over many clusters of one run, as
  one recorded job you can leave running (overnight over a whole run, say). `--clusters floor`
  (the default) is every cluster with at least the run's minimum cluster time (30 seconds of
  members, or the minimum an older run recorded when it was fitted), largest first; `all` adds the nodes
  under the floor, `leaves` the tree's leaves, and `--node` (repeatable) names the ones you
  want. The root is never one. Before anything starts it prints how long it should take on this
  computer: each profile at its fixed count (or the median this run's own profiles took);
  `--estimate` prints that and stops. Each profile measures a fixed `--n-per-side`
  members (50) so it does not depend on this computer's speed; `--use-time-budget` measures for
  `--time-budget` seconds instead. `--margin` and `--seed` are `profile`'s, for every cluster.
  The job writes each cluster into its own `analysis/<an_NN>/` and places it beside the
  clustering run (`profiles/node_<id>/`), where `profile` and the GUI's INTERPRET read it.
  **Safe to stop:** a cluster already profiled with the same settings is skipped, so running
  the same command again does only what is left (`--redo` does every cluster again). One bar
  covers the whole job (clusters done, plus the running one's share, and the time left); each
  cluster's phase (`measuring`, then `fitting`) prints above it. A cluster that fails is listed
  with why, and the job goes on. Ctrl-C cancels and keeps the finished clusters. At the end it
  prints each cluster's status, and the time it took beside the estimate. The run is an
  ordinary analysis run: `analysis status`, `attach`, `cancel` and `delete` work on it, and its
  record lists every cluster. **On a cluster:** `--remote-target <name>` runs the whole job
  there instead, freeing this computer. The clustering run is re-sent first, so the cluster
  profiles what is on this computer now, and the profiles are placed beside it here when they
  come back.
- `find-lookalikes --experiment <dir> --run <an_NN> [--pairs all|labelled]
  [--n-per-side <n> | --use-time-budget [--time-budget <s>]] [--explainer <id>] [--margin <m>]
  [--seed <n>] [--redo] [--remote-target <name>] [--yes]`: look for clusters in different
  branches of a run that are the same behaviour, as one recorded job you can leave running. It
  compares pairs of the run's **branch tops** (the two halves of every split where both are at
  least the run's minimum cluster time), each with a `profile --compare node` of one against the
  other, with position and facing left out. Two clusters the body measures cannot tell apart
  **look alike**. `--pairs all` (the default) compares every pair, which takes many hours on a
  large run, so it first prints how many pairs it will compare and about how long, with the count
  for `labelled`, and asks (`--yes` skips the question). `--pairs labelled` compares each
  labelled cluster against every other, far fewer pairs; it needs at least one labelled cluster.
  Each comparison is saved beside the run, where
  `profile --compare node --against` would save it, and a pair already compared with the same
  settings (either way round, or by hand) is skipped: safe to stop and run again (`--redo`
  compares every pair again). One bar covers the whole job; at the end it prints how many pairs
  look alike and the groups. `--remote-target` runs the job on a registered cluster (`detail
  remote list`) instead of this computer, which is worth it on a large run: the comparisons
  are placed beside the run here once they are copied back.
- `lookalikes --experiment <dir> --run <an_NN> [--label-group <n> [--name <name>] [--include
  <node> …] [--leave-out <node> …] [--yes]]`: the run's **look-alike groups**. Pairs that look
  alike chain into groups: if A looks like B and B looks like C, all three are one. Each group
  lists its clusters (share of the run and your label), a proposed name (the label most of
  them already carry; with none, give one with `--name`), and why they are
  grouped: each pair with how well the body measures, and position and facing alone, told it
  apart. A label covers everything under its cluster that has no label of its own, and clusters
  sit inside one another, so a cluster can hold others that are a different behaviour (a
  rearing cluster holding a walking one). Under each cluster, `would also label` names the ones
  its label would take that are outside the group, with their share of the run; such a
  cluster is marked `[ ]`, the rest `[x]`. Every saved `profile --compare node` with position and
  facing left out counts, however it was made. `--label-group` gives group `n`'s `[x]` clusters
  one label, the proposed name or `--name`, adding each `--include` cluster and leaving out each
  `--leave-out` one: it prints what each cluster would do, including what it would also label (or
  why it is skipped: a cluster with another label of yours keeps it), and asks before writing
  (`--yes` does not ask). Labelling the swept clusters first takes them off the list. A label
  made this way is recorded as given to a look-alike group.
- `cluster-montage --experiment <dir> --run <an_NN> --node <node> [--out <file.mp4|.gif>]
  [--per-clip-gifs <dir>] [--csv <file.csv>] [--export-dir <dir>] [--clips N]
  [--sort typical|least-typical|recording|length|random]
  [--raw/--no-raw] [--mask/--no-mask] [--keypoints/--no-keypoints] [--mask-trail/--no-mask-trail]
  [--keypoint-trail/--no-keypoint-trail] [--trail-point <name> …] [--aligned/--as-recorded]
  [--context <s>] [--overlay] [--blind] [--max-width <px>] [--fps <n>]
  [--visual <look> …] [--write montage|clips|both] [--format mp4|gif …] [--mp4-width <px>]
  [--mp4-fps <n>] [--gif-width 640] [--gif-fps 12] [--codec h264|mp4v]`:
  export a node's clips from the terminal, for a node of a temporal-clustering run.
  It writes the same files a clip grid's **Export ▾** writes in the app. Give at least one of
  the four outputs:
  - `--out` is the montage itself: `--clips` clips (25 unless you say, any number up to the
    whole cluster) in `--sort` order (most typical first unless you say), each cropped to its
    animal and labelled `#rank · recording · start`, all starting together; a shorter clip holds
    its last frame until the longest ends. The rank is always the clip's typicality rank.
  - `--per-clip-gifs` writes one GIF per clip into a folder, named
    `rank_recording_m-ss.t.gif` (for example `3_mouse_07_0-42.1.gif`; `recording-C` with
    `--blind`).
  - `--csv` writes the clips as a list: a clip-query CSV (`analysis query --from-csv` reads it
    as example clips) with the clip's typicality rank, its recording's name and id, the subject,
    the start and end in nanoseconds and seconds, the node and the run after the standard
    columns, so the list (edited to the clips you want) seeds a Query's example clips.
  - `--export-dir` is **Export ▾ ▸ Export clips…**: each `--visual` in its own subfolder of the
    folder, in each `--format`, the montage file and/or one file per clip as `--write` says. A
    visual is `current` (the tick flags below), `raw`, `raw-outline`, `raw-outline-trail`,
    `raw-keypoints`, `raw-keypoints-trail`, `mask`, `mask-trail`, `keypoints`,
    `keypoints-trail` or `mask-keypoints-trails`; repeat `--visual` and `--format` for several.
    The defaults are the dialog's: `current`, both formats, `--write both`, the mp4 at full size
    (each clip at its crop's own size in the recording) and the recording's own frame rate unless
    `--mp4-width` / `--mp4-fps` say otherwise, the GIF at 640 pixels and 12 a second. The files
    are named as in the dialog: `montage.mp4`, `3_mouse_07_0-42.1.gif`, and so on, and
    `--trail-point` chooses every keypoint trail's points.

  The flags are the clip grid's **Show ▾** layers, with the same defaults: the mask and its
  trail, on white. `--raw` draws the footage, with the animal outlined; `--keypoints` adds the
  keypoints; a trail needs what it trails, and the mask trail follows `--mask` unless you set it.
  `--sort random` shuffles the cluster's clips, the same way every time for the same node, so a
  montage of 25 is a fair look at a large cluster rather than its centre alone. `--blind` labels
  recordings A, B, C. A GIF is 640 pixels wide at 12 frames a second unless `--max-width` /
  `--fps` say otherwise (each per-clip GIF too); an mp4 is 200-pixel tiles at the montage's own
  clock (the footage's rate, at most 15 a second). Every cluster has clips, however small. A run
  fitted before the node cache existed has none listed until it is built: the command says so
  and names `temporal-interpret --rebuild-cache`, which builds
  it.

  `--context 1` plays a second of footage either side of each clip in `--out` and
  `--per-clip-gifs` (**Show ▾**'s context on a clip grid); the labels and the `--csv` list still
  name each clip's own span.

  `--aligned` turns every tile so its animal faces up, by the heading its embedding run uses
  (**Aligned** on a clip grid): the run's keypoint track run and heading definition, or, for a
  run that was not aligned, the one heading definition saved for the experiment that fits its
  skeleton. A frame the embedding would not turn (a gap in the heading longer than its
  `max_gap_s`) is a grey tile marked *no heading*, never the frame unturned. A run aligned by
  `mirror` (a side view) is flipped as the run flips it rather than turned. `--aligned` applies
  to the tick flags and to every `--visual`; when no clip has a heading the command stops and
  says why. `--as-recorded` is the default.

  `--overlay` writes `--out` as one picture of the clips drawn over each other (**Export ▾ ▸
  Export overlay…** in the app): each clip's outline in its own colour, turned to face up and
  centred, with its keypoint skeleton on top and a key beside it naming each colour. It shows 12
  clips unless `--clips` says otherwise. Clips make an `.mp4` or a `.gif`, playing together.

  `--codec` chooses how every MP4 is encoded: `h264` (the default; plays in PowerPoint and every
  browser) or `mp4v`. Where this computer cannot encode H.264 the export writes `mp4v` and says
  so.

  ```bash
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --out rearing.gif
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --raw --keypoints --out grooming.gif
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --clips 40 --sort least-typical --per-clip-gifs odd_ones/ --csv odd_ones.csv
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --export-dir grooming/ --visual current --visual raw-outline --format gif
  detail analysis cluster-montage --experiment $PROJ --run an_07 --node 118 --aligned --out grooming_aligned.mp4
  ```

- `query-eval --experiment <dir> --embed-run <emb_NN> --behavior <label> [--behavior <label> …]
  [--search <recording> …] [--stream <name>] [--band 0.5] [--fusion min|mean] [--method dtw]
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
  recording. **Vary the embedding run** to compare embedding configurations (mask resolution, embed model,
  input-image constructor); vary `--band`/`--fusion`/`--method`
  for engine params.

  `--method` selects the scoring method, in two families. **Windowed matchers** (span-detection
  AP): `dtw` (default; banded subsequence DTW), `clip_pooled_mean`/`clip_pooled_max` (one pooled
  descriptor per clip) and `keyframe` (the most query-defining frame) return matches roughly the
  **example's length** and are scored by temporal overlap. `keyframe` picks its defining frame by
  `--keyframe-select medoid` (the most typical, the default), `distinctive` (the farthest from the
  background) or `rare` (the least like the rest of the database), and `--n-keyframes` (default 1)
  sets how many such frames it takes from the query: a match scores by holding a frame close to
  each. The **per-frame classifier** (per-frame
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
- `info <an_NN> --experiment <dir> [--export FILE.zip]`: prints every YAML file the run
  keeps about itself, as it is on disk: `analysis.yaml` (its record and every setting it was
  launched with), the module's own file (method settings, labels), then the rest
  (`machine.yaml`, `resources.yaml`). `--export` writes them to one zip instead. The GUI's
  **Meta-information…** window shows the same files.
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
  output and its current settings. The repair for a run whose results are **out of date**,
  a change that did not finish writing them, so the settings moved and the results did not
  (`status` says so, and the GUI offers **Rebuild results**). The expensive pass is untouched:
  this costs one pass over the outputs, not another fit or search. It is also the way to
  re-drive a run whose results write **failed**: the fit or search is still on disk, so the
  run does not have to be redone from the start.
  Rebuild also brings a clustering run written before 2026-11-08 up to the rule that no
  instance spans a stretch tracking lost (see `temporal-interpret`).

  A label or cutoff change writes the results and leaves the per-frame table for
  later, so it returns in seconds. `measures`, `--finalize` and the `export-*` commands bring that
  table up to date first, showing a progress bar only when there is something to rebuild.
- `delete <an_NN> --experiment <dir> [--yes]`: remove the run's `analysis/<an_NN>/`
  directory. Blocked only while the run is active (cancel it first).
- `duplicate <an_NN> --experiment <dir> [--name <text>]`: copy a run whole (its fit, tree,
  clips, labels, measures and profiles) to a new `an_NN` named `<name> (copy)`
  unless `--name`, leaving the original untouched; try something on the copy without risking
  the run. The copy's records name the copy. Refused while the run is going. A bar shows the
  copy; Ctrl-C stops it and removes what was copied.

### `measures`
Turn a behavior result into numbers: how many bouts, how much total time, a rate against observed
time, a mean bout duration. Any module that produces labeled behavior in time can be measured. The
full list, with what each one reports and what it needs, is the
[Measures reference](measures.md).

- `list`: every measure, with its name, what it answers, the unit its value carries, and the rule
  it uses at a window edge. `--measure` below takes the id in the first column, not the name.
- `export --experiment <dir> --run <an_NN> --out <dir> [--measure <name> …]
  [--windows whole|absolute|normalized|sliding] [--every <seconds>] [--parts N]
  [--length <seconds>] [--step <seconds>]`: measure the run and write three files into
  `<dir>`, `measures.csv` (one row per recording, subject, label, window and measure, with the
  observed time each rate was taken against), `frames.csv` (one row per frame: whether it was
  observed and what was active on it), and `measures.yaml` recording the windows and measures used.

  `--measure` defaults to every measure the source run can answer: one that needs something
  the run does not have (`transitions` needs non-overlapping behaviors; `unnamed_time` needs
  unnamed clusters) is skipped rather than reported as a column of zeros. `--windows` picks the
  window model: `whole` measures the
  full recording, `absolute` tiles it into fixed windows of `--every` seconds, `normalized` splits
  it into `--parts` equal fractions (for recordings of differing length), and `sliding` walks a
  `--length` window forward by `--step` for a time-resolved curve.

The instance table is not copied here: the run already writes `instances.csv` in
`analysis/<an_NN>/`.

### `comparison`
Group statistics across recording metadata: take a behavior result, split the recordings by a
column of `metadata.csv`, and report how the groups differ.

- `run --experiment <dir> --source-run <an_NN> [--snapshot <snap_NN>] --group-by <column> --aggregate-to <column|none>
  [--measure <name> …] [--recording <id> …] [--windows whole|absolute|normalized|sliding]
  [--every <seconds>] [--parts N] [--length <seconds>] [--step <seconds>] [--draws N] [--seed N]`: measure each unit, group them, and
  save the result to `comparisons/<cmp_NN>/`. Prints the `cmp_NN`. `--group-by`,
  `--aggregate-to`, `--measure` and `--recording` are all repeatable.

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
  the run does not have (`transitions` needs non-overlapping behaviors; `unnamed_time` needs
  unnamed clusters) is skipped rather than reported as a column of zeros. `--windows` picks the
  window model: `whole` measures the
  full recording, `absolute` tiles it into fixed windows of `--every` seconds, `normalized`
  splits it into `--parts` equal fractions (for recordings of differing length), and `sliding`
  walks a `--length` window forward by `--step` for a time-resolved comparison. `--draws` sets how
  many permutations the difference ranking's floor is drawn from.
- `export <cmp_NN> --experiment <dir> --out <dir> [--format pdf|svg|png …] [--theme light|dark]`: write the publication
  tree, `stats.csv` (long form, one row per observation), `tables/` (per measure: groups as rows
  with n / coverage / mean / median / sd), `figures/` (every observation as a point, plus the
  difference ranking with its chance floor), `summary.md`, and `manifest.json` with a SHA-256 per
  artifact. Warns if the source data has changed since the comparison was run. `--theme dark`
  draws the figures on a dark background for slides; the manifest records which theme was used.
- `list --experiment <dir>`: the experiment's comparisons, with the unit of analysis each used.
- `delete <cmp_NN> --experiment <dir> [--yes]`: remove the comparison and its exports. Confirms
  unless `--yes`.

Statistics stop at description: means, medians, spreads, and a permutation floor that says whether
a difference is bigger than shuffling the group labels would produce. There are no p-values,
`stats.csv` is the handoff to whatever you test with.

### `model`
Built-in (registry) model readiness/install, plus the **machine-global imported
model** catalog (shared across every experiment on the machine):

- `list`: every model, built-in and imported, with install status, role (`detection` or
  `keypoint`), and a keypoint model's skeleton. No network.
- `check <id> [--experiment <dir>]`: readiness reason + action URL. Consults the
  network only when the model isn't installed.
- `install <id> [--experiment <dir>]`: download weights into the global cache
  (idempotent; prints per-file progress). The only command that downloads. A model whose
  weights restrict use (the SuperAnimal keypoint models: academic, non-commercial) prints
  its license first; `list` shows it as `license=…`.
- `import <weights.pt> [--skeleton <preset|sk_NN|file.yaml>] [--family <family>] [--name <id>]
  [--made-by] [--description] [--notes] [--preview <image>]`: register an Ultralytics
  checkpoint into the machine-global catalog with optional metadata + a preview image. The
  checkpoint's task decides what it becomes, so leave `--family` out (when given, it must
  agree). A detection checkpoint becomes a detector, its class names read from the checkpoint;
  use it as `detect --model <id>`. A pose checkpoint becomes a keypoint model and needs
  `--skeleton`: a preset, a library skeleton or a skeleton YAML (see [`skeleton`](#skeleton)). It must have
  the model's number of points, and left/right pairs that match the ones the model swaps when
  it flips an image; a checkpoint that records no swap accepts only a skeleton without pairs.
  The check cannot catch a pair named the wrong way round (left and right swapped). The
  skeleton is added to the library unless an entry with the same skeleton is there, and the
  command prints the entry. Use the model as `keypoints detect --model <id>`. `--skeleton` is
  refused for a detection checkpoint.
- `import <export folder or model.onnx> --keypoints [--skeleton <yaml>] [--name <id>] …`:
  register a keypoint model exported from SLEAP (below), usable in any experiment as a
  `keypoints detect --model <id>`. The keypoint names and edges come from the export; which
  points pair as left and right is guessed from the names (`left_ear`, `right_ear`) and
  printed, so you can check it. `--skeleton` gives the pairs and edges instead, as a skeleton
  YAML (the export's names, in order). Needs the `keypoints_onnx` extra
  (`pip install -e ".[keypoints_onnx]"`). A run with `--backend cuda` needs that extra's
  GPU build of onnxruntime; with the CPU-only build the run is refused rather than run on
  the CPU, and the message says how to switch.
- `skeleton <id> [--yaml]`: an imported keypoint model's keypoints and left/right pairs;
  `--yaml` prints them as a file to edit.
- `set-skeleton <id> --skeleton <yaml>`: set its pairs and edges from that file. The names are the
  model's outputs and cannot change; runs already made keep the skeleton they recorded.
- `show <id>`: print one of your models' metadata + provenance (the detail page, as text),
  with its role, and for a keypoint model its skeleton: the library entry, points and pairs.
- `set-meta <id> [--made-by] [--description] [--notes] [--preview <image>]`: edit a
  model's metadata (only the flags you pass change).
- `export <id> <dest.zip>`: bundle a model (weights + `model.yaml` + preview) into a
  `.zip` to share it (shows progress).
- `restore <archive.zip>`: bring a model exported with `export` back into the catalog. A
  keypoint model keeps its own skeleton; it is added to the skeleton library unless an entry
  with the same skeleton is there ("name (2)" when the name is taken by a different one), and
  the command prints the entry it matched or added.
- `remove <id>`: delete an imported model.

**Exporting a SLEAP model.** DETAIL runs SLEAP models (SLEAP 1.5 and later, which train with
sleap-nn) exported to ONNX, without SLEAP installed. In the environment where SLEAP is
installed:

```
pip install "sleap-nn[export]"
sleap-nn export <model folder> -o <export folder> --format onnx
```

then `detail model import <export folder> --keypoints`. On Windows, run the export with
`set PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8 = "1"`) first; the exporter prints
characters a Windows console cannot show and stops. Three kinds of model import:

- **single-instance**: one animal per frame, found in the whole frame;
- **top-down**: export the centroid and centered-instance models together
  (`sleap-nn export <centroid folder> <centered-instance folder> --format onnx`); DETAIL keeps
  the animal whose centroid falls on the tracked subject;
- **centered-instance** on its own: the crop around DETAIL's tracked subject replaces SLEAP's
  centroid model. Only a model trained at input scale 1: sleap-nn exports a scaled one with
  its crop resize fixed at the training frame size, which misplaces every point, so the
  import refuses it and asks for the top-down export instead.

Bottom-up and multi-class models are refused at import, with the reason. An exported model is
not quite SLEAP's own predictions: the export finds each point on the model's confidence-map
grid (output stride ÷ input scale, often 4 px) without SLEAP's sub-pixel refinement, so a point
can sit a few pixels from where SLEAP itself would put it. DETAIL runs the export exactly as
sleap-nn's own ONNX runtime does, and size-matches each frame to the frame size the model was
trained on, as SLEAP does. A model trained in SLEAP 1.4 or earlier (TensorFlow) has to be
retrained in SLEAP 1.5+ before it can be exported.

### `modules`
List the registered processing modules and their dependency chain.

### `doctor`
Check this machine's setup: PyTorch / CUDA, the SAM 3 backend package, Hugging Face,
and model weights. It runs the checks in a fresh subprocess, so it reflects the
*current* install state (handy for debugging a setup over SSH). Each failing check
names the docs page that explains the fix; when a backend package is missing it also
prints the full copy-paste install sequence for this machine, with the real detected
paths, from opening a terminal to verifying (the Windows Triton line only on Windows).
Exits non-zero if any check failed. This is the headless twin of the GUI's
**Configuration ▸ Check readiness** button and **SAM 3 backend** page.

The *Analysis (optional)* group checks SHAP (for Cluster profile). It is optional: it warns,
never fails, and the output ends with the command that sets it up.

### `install-extra`
Install an optional extra's missing packages into the environment running detail:
`analysis` (SHAP for Cluster profile, and the clustering libraries). It uses `uv pip` when uv is installed, else
that environment's `pip`, installs only what is missing, and shows a progress bar; if the
installer fails, its output is printed. `--print-command` prints the command instead, to run
it yourself. The GUI's twin is **Install SHAP** on **Configuration ▸ Cluster profile**.

```bash
detail install-extra analysis
detail install-extra analysis --print-command
```

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

A tracking run's masks are kept per tracking and detection run, and per class for a tracking run
with a class filter, so such a run rebuilds its masks once, the first time they are needed after
this was added.

`recording add-video` builds the new part's preview after the import. `recording import`
does not; run `cache previews build` afterwards. The GUI builds them in the background.

### `hf-token <token>`
Validate and store a Hugging Face token in
`~/.config/detail/credentials.yaml`.

### `remote`
Machine-local registry of remote Slurm clusters (UW Tillicum, Hyak, a lab's own cluster)
DETAIL can submit jobs to (stored in `~/.config/detail/remote_targets.yaml`, never in an
experiment). DETAIL never stores or prompts for credentials: `<host>` should match a
`Host` entry in your own `~/.ssh/config` so keys, an agent, ControlMaster, and a cached Duo
session are already handled.

- `remote add <name> <host> --workdir <path> [--partition ...] [--account ...]
  [--conda-env ...] [--module ... (repeatable)] [--gpu-type ...] [--gpus-per-job N]
  [--cpus-per-task N] [--mem-gb N] [--walltime HH:MM:SS] [--max-concurrent-jobs N]
  [--cost-per-gpu-hour RATE]`:
  register (or re-point) cluster profile `<name>`. `--cost-per-gpu-hour` drives a live
  cost estimate on this target's runs (GPUs held × running hours × rate: `gpus_per_job`, or one for Find look-alikes
  and keypoint track, which are CPU work; running time is what the job spent on a GPU: time waiting in the Slurm queue is not billed); omit
  it to show no cost estimate. **`--walltime` defaults to `24:00:00`**, the longest a Tillicum job
  may run: Slurm kills a job at its walltime, and the old 4-hour default killed long detections.
  **`--max-concurrent-jobs` (default 8) is how many recordings of one run are on the cluster at
  once**, each its own Slurm job: `detail process detection --backend remote` over 80 recordings
  keeps 8 in flight instead of going one at a time. Re-running `remote add` for an existing name
  replaces the whole profile, so every option you leave out goes back to its default; the GUI's
  **Edit** changes one field and keeps the rest.
- `remote list`: list the registered profiles (name, host, scheduler, work directory, concurrent
  jobs, walltime).
- `remote remove <name>`: remove a profile.
- `remote test <name>`: check SSH reachability and that `sbatch` is on the remote `PATH`, then
  compare the DETAIL installed on the cluster with the build this copy is running. The second
  line says the cluster is the same build, or **behind** (with `remote update` below, and the
  `ssh … 'git -C … pull --ff-only'` it runs, to do it by hand), **ahead**, on a **different history**, or could not be
  compared (and why). A mismatch is a warning: the exit code is the connection's alone, so a
  script that checks reachability does not start failing. Nothing is fetched, and nothing on the
  cluster changes.
- `remote update <name>`: bring the DETAIL on the cluster up to date with this copy. It runs the
  same comparison and, only when the cluster is **behind**, asks (`-y` / `--yes` skips the
  question) and then runs `git pull --ff-only` in the cluster's DETAIL checkout, in the target's
  own environment, and compares again. It only ever moves that checkout forward: if the checkout
  has commits of its own, or an edit that the pull would overwrite, git refuses, the checkout is
  left exactly as it was, and git's own message is printed. Nothing on this machine changes and
  nothing is installed there: a changed `pyproject.toml` is reported, with the `pip install -e .`
  to run, not run for you. A run in flight on the cluster keeps importing modules as it goes, so
  update when none is running. **Ahead**, **different history** and **could not be compared** are
  not updated from here; the message says what to do instead (for *ahead*, update this copy). A
  pull can also finish without the two matching, when the cluster pulls from a remote that does
  not have this copy's newer commits: it says so. `--check` only compares, changes nothing, and
  exits 0 when the two are the same build and 1 when they are not. The exit code is 0 only when
  the cluster ends up on this build.
- `remote push-credentials <name>`: copy every credential this machine has configured to
  `<name>`'s own `~/.config/detail/credentials.yaml`, merging rather than overwriting
  whatever is already there. Hugging Face (`detail hf-token`) is the one credential this
  pushes today, needed for a **gated** model (SAM 3, DINOv3) to check access and download
  weights *on the cluster*: this machine's own token never reaches a remote job on its
  own, since the shepherd's readiness check and the worker both run there. A future
  credential type (a cloud provider's own access key) is forwarded the same way the day
  this machine manages one, with no change to this command.

`detection run`/`process detection`, `embedding run`/`process embedding`,
`keypoints detect`/`process keypoint_detect`, and
distillation's commands (`dataset label`, `train run`, `distill run`, and the training of
`keypoints distill` and `keypoints setup`) submit through
a registered profile (`--backend remote --remote-target <name>`, or the per-stage
`--label-remote-target` / `--train-remote-target` / `--detect-remote-target`; see
[Compute & multi-GPU](#compute--multi-gpu)). So do three analyses, with `--remote-target
<name>`: `analysis temporal-clustering`, `analysis
profile-all` and `analysis find-lookalikes`. Other stages don't take `--remote-target`
yet.
First-time setup, including the Windows-only steps:
[Connect a remote cluster](workflows/connect-a-remote-cluster.md).

The model does **not** need to be installed on this machine for a remote run: this machine
only submits the job, it never loads the model itself. The cluster's own install is checked
before the job is submitted (over ssh, in the profile's environment), so a model missing on
the cluster still fails clearly, with a message telling you to install it there, instead of
after it has queued and burned a GPU allocation.

**A gated model needs a Hugging Face token on the cluster, not just on this machine.**
Checking access to (or downloading) SAM 3 or DINOv3 needs an authenticated request, and
that request happens *on the cluster*: this machine's own token, set via `detail hf-token`,
never reaches it by itself. Run `detail remote push-credentials <name>` once per target
(or again after changing your token) before installing or running a gated model there for
the first time; a model whose weights are already cached on the cluster from an earlier
manual setup doesn't need this at all, which is why detection worked without it.

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
(default: auto, SAM 3 on an NVIDIA GPU). `--export out.mp4` renders the overlay clip (H.264,
or `--codec mp4v`) and needs `--consecutive`, since sampled frames are not contiguous.

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

### Video format: `--codec` {#video-format-codec}

Every command that writes an mp4 takes `--codec h264|mp4v`. **`h264`** (the default) is H.264,
High profile, 4:2:0: it plays in PowerPoint, QuickTime and every browser, and is a fraction
of the size. **`mp4v`** is MPEG-4 Part 2, the format DETAIL wrote before; PowerPoint on Windows
often refuses it. A machine that cannot encode H.264 writes `mp4v` instead and prints a `note:`
saying so, and an export folder's `export.yaml` records `video_codec`: the codec asked for and
the one written. The commands are `export run` (for `clip`, `clips`, `preview_video` and
`keypoints_video`), `analysis cluster-montage`, `locations montage` and
`tools quick-segment --export`.

### What each subject can produce

| Subject | Target | Writes |
| --- | --- | --- |
| experiment (no id flag) | `bundle` | one `.tar` of the whole experiment, importable back |
| | `metadata` | `metadata.csv`: one row per recording, one column per grouping variable |
| `--recording <id>` | `geometry` | field of view, distance calibration, every ROI and POI (the **scene geometry**) |
| | `streams` | one CSV per stream in DETAIL's aligned form, plus `streams.csv` |
| | `timestamps` | `frame_idx,t_ns` for every frame of every video part |
| | `video` | the video parts DETAIL imported a copy of |
| `--run <det_NN>` | `masks` | one COCO instance-segmentation JSON per recording |
| `--run <det_NN>` / `<trk_NN>` | `preview_frame` | one PNG of a single frame, in the style you pick |
| | `preview_video` | the whole recording, every part, in the style you pick, as one mp4 |
| `--run <trk_NN>` | `tracks` | one CSV per recording: a row per frame per track |
| `--run <emb_NN>` | `vectors` | `embeddings.npy` + `embeddings_index.csv` |
| `--run <kd_NN / kt_NN>` | `keypoints` | one CSV per recording: a row per frame per track per keypoint, with the score, whether it is visible at the cut-off, and what the track stage changed |
| | `keypoints_dlc` | DeepLabCut CSV per video part (and the same table as H5 with the `keypoints_export` extra), plus `timestamps.csv` |
| | `keypoints_sleap` | the SLEAP analysis HDF5 per video part, plus `timestamps.csv` |
| | `keypoints_nwb` | NWB with ndx-pose per video part |
| | `keypoints_video` | the footage with the points drawn on, one mp4 per video part, plus `keypoints.csv` |
| `--analysis-run <an_NN>` | `clips` | one mp4 per bout + `clips.csv` |
| | `instances` | `instances.csv`, `coverage.csv`, and the time-per-behavior and bout-duration figures |
| | `ethogram` | the ethogram as a picture, plus one `start_frame,end_frame,label` CSV per recording |
| | `frame_labels` | one CSV per recording and subject, one row per video frame (see [Per-frame behavior labels](#per-frame-behavior-labels)) |

`export list` shows **every** target the subject can have, and says why one cannot run,
"this recording has no streams imported yet" is something you can act on, where a missing row
would leave you unable to tell that from "DETAIL cannot export streams".

### Video that was referenced in place

`video` exports the parts DETAIL holds a copy of. A part added with **reference in place** is
skipped and the reason says so: that file is already yours, under the video root it was
registered from, and copying it back out would export nothing DETAIL made. A recording whose
parts are all referenced reports `video` as unavailable rather than writing a short export.

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
range you name with `--frames START:END`: the same export the **Export video…** button under
every clip player runs.

The full field of view is the default: a steady camera, where the surroundings are half the
answer. Crop to the subject when you want the pose at a constant scale for comparing bouts.

### A detection or tracking run's own preview, in seven styles

`--target preview_frame` (one PNG) and `--target preview_video` (the whole recording as one mp4)
draw a completed detection or tracking run the way its run-level preview draws it in the
app:
`--style` picks how: `mask_box_conf` (mask, box, confidence: the default), `mask_box`,
`mask_only`, `box_conf`, `box_only`, `raw_video` (no overlay), or `raw_mask` (each detection's
real pixels cut out onto a green background, for downstream keying). A tracking run's box always
carries its track id, with or without confidence, since the track id is why a tracking preview
exists.

`preview_frame` needs a frame (pass it as a one-frame `--frames START:START`), and both need
`--for-recording` with exactly one recording, since a run can span several and a picture is drawn
from one (naming two is refused rather than drawing the first):

```bash
detail export run --experiment ./study --run det_01 --target preview_frame \
    --for-recording recording_001 --frames 1200:1200 --style box_conf --out ~/exports
```

### Keypoints, for pose tools

A keypoint run (`kt_NN`, or a `kd_NN` for the raw placements) exports in the shapes other pose
tools read. Every one of them is written from the same rows, so a CSV and an NWB file of the
same export agree point for point. A track run's poses belong to tracked animals. A detect run's
do not: they export **unassigned**, with an empty `track_id` and each pose numbered within its
frame (`instance`), which DeepLabCut and SLEAP read as untracked individuals. `--source detect`
on a track run exports them the same way.

- `keypoints` is DETAIL's own long table, one CSV per recording: a row per frame per track per
  keypoint with `x_px, y_px, x_norm, y_norm, score, visible, changed`. `changed` is `0` as
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
recording's name, not its id.

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
detail export run --experiment ./study --run trk_01 --target tracks \
    --for-recording recording_004 --out ~/exports
```

Without it, **every** recording is exported, and for video that is the whole run: one overlay video per recording can add up to hundreds of gigabytes. A target that writes one file for the whole experiment (`metadata`, `bundle`) has no recordings to choose between, and `export list` shows which targets take `recordings` among their scopes. A recording the run does not cover is refused before anything is written.

### How big and how long

`export run` prints an estimate before it starts, from the same catalogue the app's dialog asks:

```bash
detail export run --experiment ./study --run trk_01 --target tracks \
    --for-recording recording_004 --out ~/exports --estimate
```

`--estimate` prints the total and one line per target with how the figure was reached, then stops; nothing is written and no folder is made. The sizes are counted from what is on disk where they can be (detections, tracks, keypoint rows), measured by writing a sample through the real writer for the keypoint CSVs, and measured on this machine for video, where a few dozen frames are timed (briefly) and scaled up. They are estimates: video seconds read high, and a total says **at least** when a target could not be sized (the performance report, whose size depends on what is installed, and the SLEAP and NWB keypoint files, which compress their arrays by an amount nothing tells you before they are written; such a row adds nothing to the free-space check). It also says how much room the drive under `--out` has. The keypoint video's drawing time is an approximation, taken from the engine on another branch.

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
reports blanks rather than guesses. **Nothing is sent anywhere**: it is a file, and sharing it is
your decision.

### The columns

`clips.csv`, `instance_id`, `source_label` (the cluster's own id), `label` (the name a person
gave it, blank if none), `label_path` (that name and every group it sits inside, outermost
first), `recording_id`, `subject`, `start_frame`, `end_frame`, `t_ns`,
`end_t_ns`, `dur_s`, and one `video_<render>` column per flavour. A blank video column means
that bout produced no clip: the track was not detected across it, or it fell outside the video
, and the row is kept because its times are still true.

`instances.csv`: the same identity columns plus `score`, `label_path`, and `cx_norm`,
`cy_norm`: where each bout happened, as a fraction of the frame. `label_path` is the whole chain
of names the bout sits under, outermost first (`not moving/resting`) where `label` is the last
element on its own. Total by `label` for the specific behavior, or by a prefix of `label_path` for
the group it belongs to; each bout appears once either way, so neither double-counts. A flat
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
does not stop it; come back with `export attach`. Ctrl-C while you are watching cancels it, like
every other `run` command. A short export is unchanged; it finishes before there is anything to
watch.

```bash
detail export run --experiment ./study --run trk_01 --target tracks --out ~/exports
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
index is not optional.

### Re-exporting

Every export makes a new folder, so re-exporting never overwrites the last one and never touches
anything else in the directory you pointed at.

## Tracking

Three trackers ship; all take `--conf-threshold`, `--arena`, and `--class`.

`--arena` takes the field of view's **name**, and that is what to pass when tracking more
than one recording: a name is the same field of view everywhere, while an arena *id* is
minted per recording, so the same id is a different polygon on the recording next to it.
Each recording resolves the name to its own arena, and one that has no field of view by that
name is refused by name rather than silently tracked whole-frame. An id is still accepted
for a single recording.

**Two animals, told apart by class instead of by tracking.** Every tracker here follows
*one* subject per recording, which is what the rest of the pipeline (embedding, query,
clustering) is built for. If a detection model already distinguishes two animals as two
classes (e.g. `black mouse` / `white mouse`), `--class` restricts one tracking run to a
single class's detections, so running it twice, once per class, on the *same* detection
run, produces two clean single-subject tracks instead of one tracker having to
disambiguate the animals itself:

```bash
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker anchor_tracker --class "black mouse"
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker anchor_tracker --class "white mouse"
```

Each run is an ordinary single-subject track downstream: embedding, query, clustering, and
comparison all read it the same way they'd read a one-animal recording. `--class` is checked
against the classes the detection run actually has; an unknown name is refused by name,
naming the classes it does have. Omit it to link every class's detections together, which is
right for a single-class detection run.


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

## Merge tracks

Combines 2+ single-subject tracking runs, typically two `--class`-filtered runs on the same
detection run, one per animal, into one multi-subject tracking run, so both animals' masks
show up in the same preview, the same location export, and the same embedding run. Each source
keeps its own `object_id` in the merged run (source order = `object_id` order); nothing is
re-tracked, the merge just relabels each source's already-clean rows.

Every source must share one detection run and carry a distinct `--class`: merging two runs
built from different detections, or two runs filtered to the same class, is refused.

```bash
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker anchor_tracker --class "black mouse"
# submitted tracking: track_01
detail tracking run --experiment $PROJ --recording recording_001 \
    --detection det_01 --tracker anchor_tracker --class "white mouse"
# submitted tracking: track_02

detail track-merge run --experiment $PROJ --recording recording_001 \
    --source track_01 --source track_02
# submitted track-merge: track_03
```

`process track-merge` runs this across several recordings at once, resolving each source by
spec rather than by a raw id shared across recordings: `--source-ref RECORDING:track_NN`
derives the class and tracker settings from that one named run and finds each selected
recording's own matching run, the same way `--detection-ref` works for `tracking`:

```bash
detail process track-merge --experiment $PROJ \
    --recording recording_001 --recording recording_002 \
    --source-ref recording_001:track_01 --source-ref recording_001:track_02
```

The merged run is an ordinary tracking run downstream: `export run --target preview_frame` /
`preview_video` and location export read it unmodified. For embedding, a merged run's
`object_id` is a *position* (source order), not an animal: the same integer can mean a
different animal in a recording whose merge happened to order its sources the other way.
So `embedding run --tracking track_03 --tracks` also takes the source classes by name,
`--tracks "black mouse"` or `--tracks "black mouse,white mouse"`, resolved to the right
`object_id` for *each* recording, rather than an integer you'd have to hope lines up:

```bash
detail embedding run --experiment $PROJ --recording recording_001 \
    --tracking track_03 --tracks "black mouse" --model dinov3_convnext_base
```

`process embedding --tracks` takes the same names across a batch.

Pooling two separately embedded animals into one clustering or query run is not supported;
that's a modeling decision, and is refused the same way pooling two tracks from one
recording always is.

## Compute & multi-GPU

Detection and embedding take:

- `--backend {cpu,cuda,mps}` (default `cpu`)
- `--device N`: CUDA device index, **repeatable** for multiple GPUs

```bash
# single GPU
detail detection run … --model sam3 --class mouse --prompt mouse --backend cuda --device 0
# two GPUs: work fans out across them
detail detection run … --backend cuda --device 0 --device 1
```

DINOv3 and a trained/imported YOLO run on `cpu`/`mps`/`cuda`; SAM 3 is officially CUDA-only.

**`detection run`/`process detection`, `embedding run`/`process embedding` and
`keypoints detect`/`process keypoint_detect` also accept
`--backend remote --remote-target <name>`** (§6): submits to a registered Slurm cluster
(`detail remote add/list`, e.g. Tillicum) instead of running here. The model still needs
to support `cuda` (the job runs as `cuda` on the cluster's GPUs) but this machine
doesn't: a laptop with no GPU, and no local install of the model at all (no weights
downloaded, no HF token configured), can run SAM 3 this way; only the cluster needs the
weights. `--device` is ignored with `--backend remote` (the cluster profile decides how
many GPUs the job gets). Tracking isn't wired to `--backend remote`: it's a CPU-only
algorithm over already-computed detections, with no compute backend to redirect.

**`keypoints track`/`process keypoint_track` also accept `--backend remote --remote-target
<name>`.** It is CPU work with no model, so there is nothing to install on the cluster. What
the cluster adds is more recordings side by side: here a run takes half this machine's cores (at
most 8), there each recording is its own Slurm job, up to the target's `--max-concurrent-jobs`. Each job asks for one GPU (the fewest
Tillicum accepts) with the target's CPUs and memory scaled down to match, so a target sized for
a multi-GPU model does not hold idle GPUs.

A **keypoint detect** run on a cluster needs the method's weights on the cluster only
(`dinov3_match` uses `dinov3_vit`'s, and DINOv3 is gated: `remote push-credentials`). Its
example set travels with the experiment, and so does a student distilled in it. A keypoint
model in this machine's model catalog (one you imported with `model import --keypoints`, or a
promoted student) lives outside the experiment, so the run uploads its folder to the cluster
before the readiness check, the way it does an imported detection model; an imported model
needs `onnxruntime` installed on the cluster.

**Distillation (`dataset label`, `train run`, `distill run`) is wired the same way.**
`dataset label` and `train run` take the same `--backend remote --remote-target <name>`;
`distill run` labels and trains as two separate submissions, so it takes them per stage:
`--label-backend remote --label-remote-target <name>` and/or `--train-backend remote
--train-remote-target <name>`, and either stage can go to a different cluster (or stay
local while the other goes remote). A GPU-less laptop can run the whole distill loop this
way: SAM 3 needs `cuda` to label and the trainer supports it too, so both stages resolve to
a cluster GPU. No other stage is wired to `--backend remote` yet.

**Background video upload** (`distill run --stage-videos`). A remote distill's labeling is sent only
its sampled frames, but the detection run after it reads every recording's video. With the flag, a
separate background process uploads the videos to the cluster while labeling and training run, and
keeps going after the distill ends:

```bash
detail remote staging-status --experiment $PROJ     # per target: status, files, GB, time left
detail remote staging-stop tillicum --experiment $PROJ   # omit the target to stop every upload
detail remote staging-start tillicum --experiment $PROJ  # start again, or resume a stopped one
```

There is no pause, but stop followed by start is one: a video already on the cluster is skipped
and a half-sent one continues where it stopped (rsync keeps the part it sent in a hidden
`.rsync-partial` folder beside the file, so a half-sent video never sits at the real path). `staging-start`
takes the recordings of the last upload to that target, or `--recording <id>` (repeatable).

A remote job that needs the videos while an upload is still running **waits for it** and shows its
progress, rather than uploading the same files a second time. If the upload failed or was stopped,
that job uploads whatever is missing itself; files that already arrived are kept.

**CLI default is `cpu`; the GUI and Python API auto-detect.** The CLI keeps an explicit `cpu`
default so a scripted or headless run is predictable; pass `--backend cuda` (or `mps`) to use an
accelerator. The GUI forms and the Python API instead default to the best backend present
(cuda > mps > cpu), narrowed to what the chosen model/trainer supports.

## Inspecting and troubleshooting runs

- `detail recording status` (above) lists every run and its status:
  `complete`, `failed`, `partial` (interrupted, resumable), or an active state.
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
- **`<path> is all zero bytes` (or `is empty`)** names a file the machine lost the same way: it
  shut down or lost power before the system had written the file to disk, and the file kept its
  size but not its contents. A run that sits on top of one (tracking on a detection run, say) is
  refused at launch, naming the recording and the file, instead of failing recording by recording
  once it starts. The data in that file is gone; re-run the stage that wrote it. If it came from a
  cluster, the cluster's copy is probably still intact (see
  [connect a remote cluster](workflows/connect-a-remote-cluster.md#troubleshooting)). A run is
  now recorded as complete only after its outputs are flushed to disk, which includes results
  copied back from a cluster, so this should only turn up for runs finished before that.
- Outputs live beside the inputs in the experiment tree
  (`detections/det_01/`, `tracking/track_01/`, `embeddings/emb_01/`); they are
  zarr stores read by the GUI and the Python API.
- **What Ctrl-C does depends on whose run it is.** On a `run` command it **cancels**: the
  worker is in that command's care, and walking away would leave it going with nothing watching
  it. On an `attach` command it **detaches**: you did not start that worker, and reattaching
  exists precisely so you can let go of it again. `distill run` is the one exception, launching
  a distill detaches you from it by design, because it is the longest thing this tool does.
- **Stopping a run you are not watching:** `<thing> cancel <id>` (`process`, `analysis`,
  `distill`, `export`), the GUI's **Cancel** button, or `Experiment.cancel_run` /
  `cancel_model` / `cancel_dataset` in Python. A dataset or training run has no `cancel`
  subcommand; see [Not in the CLI](#not-in-the-cli).

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
detail process list --experiment $PROJ          #   run_001  detection  complete

# 2. export a YOLO dataset (every 10th frame) -> dataset_001
detail dataset create --experiment $PROJ --name mouse_yolo \
    --source-run run_001 --every-n 10

# 3. train YOLO-s (segmentation) -> model_001  (uses CUDA when available)
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
lands in `models/model_001/run/weights/best.pt`. Training defaults to **CUDA when
the machine has it**; pass `--backend cpu` to force CPU. `train resume` continues
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

Include the heavy bits when you need them: `--include-video`, `--include-datasets`, `--gzip`. `--gzip` rarely pays: DETAIL's data files are compressed already, so it barely shrinks the archive and makes the export much slower. On import, `--replace` overwrites a same-id experiment and `--name <new>` lands a separate copy.

Both commands show a progress bar on stderr, relabelled as the work moves through
its phases: export goes `scanning` → `checking` → `hashing` → `packing`, import
goes `reading` → `extracting` → `verifying` → `placing`. A slim export reads every
file twice (once for the checksum, once to pack it), so a large experiment takes
minutes even without its video. If a file changes between those two reads (a run
still writing to the experiment), the export stops and names the file; let the
run finish and export again. The archive is written as `<out>.partial` and
renamed when it is complete, so a stopped export never leaves a file that looks
finished. A video-excluded import is
fully analysis-ready (embeddings, tracking re-tuning, metadata); the
frame-preview surfaces show "source video not in this export", and the clip
picker refuses to mark example clips until the video is on this machine.

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
recordings get a root of their own. Whatever is still not found is listed:

```
imported: /Users/you/Documents/DetailExperiments/study
video not found on this machine for 2 recording(s): s1, s2. Relink one file (`detail recording relink`) and every recording whose video is in the same folder finds it too.
```

### Importing event streams (MED-PC)

Attach an aligned operant-behavior stream to a recording:

```bash
detail recording add-stream --experiment $PROJ --recording recording_001 \
    --name medpc --importer medpc --data box15.txt --program program.MPC
#   added stream 'medpc' (medpc) to recording_001: 1234 rows
```

`--importer medpc` parses Med Associates `.txt` + `.MPC` into canonical events.
Aligning the stream to video time (an offset) and previewing it against the video
are done in the [GUI](gui.md#recording-detail).

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
  2.0,release
  ```

The importer validates strictly and names the problem: no recognized time column, more than
one time column, a column name repeated in the header, an empty or header-only file, a
`timeseries_csv` with no value columns or a non-numeric cell (or a `label` column, which means
you want `events_csv`), and an `events_csv` with no `label`, an empty label, or a non-integer
`code` / `frame`.

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
anchored to the video already, and passing an offset for one is refused rather than stored.

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
one unreadable file doesn't sink the batch.

One recording at a time takes `--preset` instead of `--importer` and the mapping flags:

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

Each behavior is recorded as **point** or **state** (BORIS's *Behavior type* column, else
inferred from durations) so the evaluation scores it correctly, and the annotation's **covered
behaviors** (the scored vocabulary) default to the labels present. Add
`--covered-behaviors "groom,rear"` to declare behaviors you *scored but never observed*; the
evaluation treats them as covered (a real absence, not "unscored") rather than skipping the recording.

**Ground truth vs predictions.** A stream also records where its labels came from. Anything you
import is ground truth unless you say otherwise; a file the tool exported (`analysis export-temporal-clustering`)
carries an `origin` column and imports as `predicted` on its own. Override either way with
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
resolve); then each file is imported against its own recording's timestamps. A file
that fails to parse is reported and skipped; the batch continues. Every file in one
`import-labels` run uses the same `--importer` (`behavior_csv` or `boris`).

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
the experiment, a duplicate id, an unknown importer, a missing file, or video parts
that disagree on fps/resolution all abort the import (nothing is created).
`--dry-run` runs the same validation and reports without importing.

### Recording information

Record what varies across your recordings (treatment, sex, day, …): one row per
recording, one column per variable. Every experiment has an `acquisition_setup` column:
which camera, box or rig each recording was made with, the same value for every recording
made the same way. Keypoint setup picks one recording per setup to learn from. It cannot be
deleted. From a CSV:

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
detail experiment set-metadata --experiment $PROJ --recording m27_day3 --set "acquisition_setup=box 1"
detail experiment delete-column --experiment $PROJ dose      # and its values everywhere
```

A new key in `--set` becomes a column on every recording (blank elsewhere): the
table is one shape, not one per recording.

### Naming recordings

`--recording` takes a **name or an id**. The name is yours (it starts as the video's
filename and you can change it); the id is the tool's (`recording_000`, assigned in
order, never reused). `recording status` prints both:

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
GUI is either a CLI flag or listed here, with the reason. This list is the
ledger of *deliberate* omissions: interactive, iterate-until-it-looks-right
tools that live in the [GUI](gui.md) (or the `scripts/smoke_*.py` examples),
not the CLI:

- **`--embed-ref` for `analysis query`**: the `--*-ref
  <RECORDING:run_id>` spec selector (choose the upstream by *kind* so a batch's
  recordings can carry different run ids) is wired for `process tracking`
  (`--detection-ref`), `process embedding` (`--tracking-ref`), `analysis run` (`--tracking-ref`) and `temporal-clustering` (`--embed-ref`), but `query` still takes an
  explicit `--embed-run <emb_NN>` (must be that id in every searched recording). Deferred:
  its embedding run also identifies the *exemplar* recordings, so the spec resolution
  is bespoke (planned).
- **Preview tracking's clips** (the Keypoint track form's **Preview tracking…**): the settings
  played side by side on the same frames. `keypoints track-preview` prints the same table, and
  `Experiment.preview_keypoint_tracking` returns every setting's points per frame for a script
  to draw; watching four clips in step is what the window is for, and a terminal cannot. The
  same holds for the grid search's clips (`--grid` prints the table and the front).
- **Measurements** (**Measure this run**, in a behavior scan's **Interpretation** section): the per-bout numbers
  (speed, area, appearance change). Neither the CLI nor the Python API has an equivalent today; the
  computation lives in `detail/gui/measurements.py` (`measure_run`), over the loaders in
  `detail/api/frame_slices.py`. A gap rather than a design choice: moving `measure_run` behind
  `api/` would make it reachable from both.

- **Skipping keypoint setup** (the Configuration checklist's **Skip…**): a choice about what
  the setup checklist nudges for, and the checklist is the GUI's. From Python,
  `detail.api.readiness.set_keypoints_skipped(True)` records it the same way.
- **Filtering recordings by a Recording-information column**: the GUI's recording pickers
  filter by name and by column (strain, dose, brain state). A CLI command names the recordings
  it runs over, and a script picks them from `Experiment.recording_metadata`
  (`{recording_id: {column: value}}`) before calling the API. A `--where Strain=CD1` selector is
  a possible addition, not built.
- **Clip checks** (the clip window's **Fits / Doesn't / Unclear**): an answer is given while
  watching a cluster's clip, which a terminal cannot show. The Python API records and reads them
  (`run.check_member(node, recording_id, subject, start_ns, end_ns, verdict)`,
  `run.member_checks(node)`), and they sit beside the run as `member_checks.csv`. Labelling a
  cluster, which the same window also does, is `--name-node` on
  `temporal-interpret`.
- **The clip window's enlarged Masks and Keypoints panes**, and their **Export video…** /
  **Export GIF…**: the pane is a view of one clip window, following its ticks as you change them,
  so a terminal has nothing to follow. From Python the same file is
  `detail.api.clip_pane_export.export_clip_pane(experiment_dir, recording_id=…, embed_run_id=…,
  track_id=…, first=…, last=…, pane="masks", settings=ClipViewSettings(…), path=…)`, one clip at
  a time. A whole cluster's clips at once is `analysis cluster-montage`.
- **What sets a cluster apart, in the clip window**: the window's measuring of a cluster's most
  typical clips against its animals' recordings. From Python it is
  `detail.api.clip_fingerprint.cluster_fingerprint(experiment, run, node, clip)`; there is no
  command, for the same reason a behavior scan's **Measure this run** has none (above).
- **Renderer preview**: preview the masked input image embedding will use.
- **Reviewing a heading direction**: the labeling screen's compass opens a window of one
  direction's candidates to look at, tick and turn down. Choosing among pictures is the point,
  so it is the GUI's; the CLI samples by direction (`examples sample --strategy heading
  --bucket`) and turns frames down (`examples reject --bucket`), and the API's
  `heading_pool` / `heading_candidates` / `add_heading_candidates` / `reject_candidate` are the
  window's own calls for a script.
- **Per-class view threshold in `detection prompt-test`**: the GUI's "Test prompts" /
  "Test detection" grid gives each class's column its own confidence spinbox, since the
  model already ran wide open and a threshold there only changes what's drawn/counted.
  The CLI's `--min-conf` stays a single value across every `--show-class` column: a
  one-shot batch command has no "move a slider and watch it re-render" to give a
  per-column control to.
- **Open SSH session** (Remote targets page): opens a terminal running `ssh <host>` to
  refresh an expired session. A CLI user is already at a terminal, so running `ssh <host>`
  directly needs no wrapper command.
- **Picking clips to export (pick mode on any clip grid)**: clicking clips into an export, in
  the order clicked, on INTERPRET, a cluster window or AUDIT, is a way of looking, forgotten when
  pick mode ends (nothing is written to the run). The CLI writes a montage of the node's clips
  in an order it sorts (`--clips`, `--sort`, and the grid's other choices: the layers,
  `--aligned`, `--context`); a hand-picked subset in a hand-picked order is exported from the
  grid's **Export ▾**. A script picks its own clips and calls
  `detail.api.clip_montage.ClusterMontage` with them. What a pick leads to is here:
  `cluster-montage` writes **Export clips…** (`--export-dir`), one GIF per clip
  (`--per-clip-gifs`) and the clip list (`--csv`), and `analysis query --from-csv` takes that
  list (edited to the clips you want) as a Query's example clips: the CLI's **New Query from N
  clips…**. From Python, `detail.api.clip_montage.query_examples(clips, name=…).config()` is the
  Query config those clips fill in.
- **Undo and redo of label changes (⌘Z / ⇧⌘Z)**: INTERPRET keeps a history of the label changes
  made on the page and in its cluster windows, in memory, so a slip can be taken back. The CLI's
  label commands are single explicit writes, so there is no history to step back through; to
  reverse one, run the opposite command (`--name-node` the old name, `--rename-label` it back). The undo
  itself is in the Python API: `AuthoredRun.label_maps()` read before and after an edit,
  `authored_run.label_change(before, after)`, and `AuthoredRun.restore_labels(labels,
  node_labels)` to write either side back, by label id.
- **Exporting the clip window's view as shown**: *Video of what you see…*, *GIF of what you
  see…* and *This frame as PNG…* write whatever the clip window's ticks, orientation and context
  are at that moment, so they live there. The API is
  `detail.api.clip_pane_export.export_clip_view` and `write_clip_frame`, with a
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
- **Per-cell metadata edits**: the cohort table's click-a-cell-and-type editing is the GUI's
  **Recording information** page (and `Experiment.set_metadata_table` for scripts). Whole columns
  *are* in the CLI, [`add-column` / `delete-column`](#recording-information) create and remove
  one, and [`import-metadata` / `export-metadata`](#recording-information) move values in and out,
  a CSV header adding any column it names. An exported template is where you edit values offline.
- **Cancelling a dataset or a training run**: `process`, `analysis`, `distill` and `export` each
  take `cancel <id>`; `dataset` and `train` do not. Stop one from the GUI's **Cancel** button or
  with `Experiment.cancel_dataset` / `cancel_model` in Python. This is a gap rather than a
  decision, and the two commands would be thin wrappers over methods that already exist.
