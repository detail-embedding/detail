# GUI guide

The desktop app does everything the [CLI](cli.md) does, plus tools that need a mouse: draw a field of view, test detection prompts, preview the embedding input image, browse results. New here? Read [Concepts](concepts.md) first.

!!! note

    Under active development. This guide covers the screens that exist today; planned flows are marked *planned*.

## Install & launch

```bash
uv pip install -e ".[gui]"      # GUI deps (PyQt6, pyqtgraph); add a backend, e.g. ".[gui,yolo]"
```

Launch with `detail-gui` (equivalently `python -m detail.gui`). There is no `detail gui` subcommand. On a machine that already has the tool installed, see [Launch it again later](getting-started.md#launch-it-again-later).

The app opens on **Experiments**, listing the experiments in your [managed workspace](concepts.md#where-experiments-live-the-managed-workspace) (`~/Documents/DetailExperiments` by default).

## First-run setup

Install model weights and store your token from the **Configuration** group. A trained or imported detection model needs neither.

1. **Configuration ▸ Hugging Face**: paste your token, click **Save token**. It validates and reports the account. The page links out to create an account, mint a read token, and request access: one link per access gate, so a single DINOv3 request covers every size.
2. **Model manager ▸ Hugging Face**: select `sam3`, click **Check access**, then **Install**.

The CLI equivalent of step 1:

```bash
detail hf-token hf_xxx
```

See [Hugging Face access](getting-started.md#hugging-face-access).

## The window

A **mode banner** across the top, the current **screen** below it. The banner is the scope switch: **Experiments**, **Active processes**, **Configuration**, **Model manager**, **Keypoints manager**, **Quick tools**, **Docs**. Most modes are a full-width screen whose items open by drilling in; an **Open →** swaps in the sub-screen. A left **rail** appears in two places: inside a project (open an experiment from Experiments and its sections list on the left) and in **Quick tools**, where each tool lists its saved runs. The rail folds like every side panel (below): drag its edge past its narrowest, or double-click the edge, and it shuts to a thin strip reading *Navigation*, which gives the screen its width on a laptop; click the strip or pull it back out to bring it back at the width it had. **Ctrl+\\** (**⌘\\** on a Mac) does either from anywhere.

**Side panels all fold the same way.** Wherever a list, a set of filters, a checklist or a summary sits beside the main content, the edge between them has a grip (a line with three dots). **Drag it** to make the panel wider or narrower. Drag it **past the panel's narrowest** and the panel snaps shut to a strip about 16 pixels wide with the panel's **name written down it** (*Navigation*, *Filters · 2 active*, *Labels*, *Instances*) and, at its middle where the grip's dots were, a **›** (a panel on the left) or **‹** (on the right) between two of the dots. **Click the strip** anywhere to open the panel, or **pull it outward**: the panel opens as soon as it leaves the edge and follows the pointer. Hovering the strip shows a pointing hand and says *Click or drag to show …*. Double-clicking the grip also toggles the panel, open or shut. **From the keyboard**, press Tab until the grip shows a highlighted outline, then Enter or Space hides the panel, and again shows it. If a panel is hidden while you are typing in it, the keyboard moves to its strip, so Enter brings it straight back. Clicking or dragging a grip never takes the keyboard from where you were working. It reopens at the width it last had. The main content and video players resize but never fold: they stop at their narrowest. Each panel's width and whether it is shut are kept for next time. At the right of the strip above the screen sit **Export…**, **Version History** and **Info**, Export appears whenever the screen you are on has data to write out, and Version History (see [below](#version-history)) is always there. The right end of the mode banner shows the build you are running.

```
[ Experiments ]  [ Active processes ]  [ Configuration ]  [ Model manager ]  [ Keypoints manager ]  [ Quick tools ]  [ Docs ]

Experiments     every experiment in the workspace: double-click a card to open
Active processes  everything in flight across every experiment, plus what failed
Configuration   machine-global setup checklist; each step opens its own screen:
                  GPU / CUDA             accelerator status + a CUDA "doctor"
                  Hugging Face access    token for gated model downloads
                  SAM 3 backend package  copy-paste commands to install the backend
                  SAM 3 weights          opens the Model manager
                  See it segment a clip  opens Quick tools
                  Recommended: distill a fast model   opens the Model manager
Model manager   every model on one page: global models, the open experiment's
                  trained models, and the Hugging Face foundation models;
                  distill or import a model from the top
Keypoints manager  the skeleton library: draw, import, export skeletons
Quick tools     one-off runs, no experiment (rail on the left):
                  Quick segment        run it, and browse its saved runs
Docs            open the full docs in your browser (offline)

Inside a project (rail on the left):
  Overview              the experiment itself: name, Export, Delete
  Recording information  one row per recording (CSV import)
  Recordings            add / open recordings
  Datasets              training datasets: build from a run, or auto-label
  Models                trained models, and the distill runs that make them
  Processing pipeline   launch + browse processing runs (detection/tracking/embedding)
  Analysis              launch + browse analysis runs
  Comparisons           compare behavior between groups
```

### Version History {#version-history}

The right end of the mode banner shows the build you are running, such as `v0.1.0 · 2026-09-20 · 033eef0`:
the date of the commit your copy is at, and that commit. The date is the quick answer to "is my copy
current?". A trailing `+` means the copy has local edits on top of that commit.

**Version History** opens a panel with the full build (commit, branch, date) and the list of what
changed, newest first, on `main`, one line per merged change, with its date. Nothing in it is kept
by hand: it is read from the git checkout you installed from, so a copy that was not installed from
a git clone shows the version number alone.

**Check for updates** in that panel compares your copy with the latest `main` on GitHub and says how
many changes you are behind. It is the only part that uses the network, and only when you click it.
To update, `git pull`, then reinstall as in [Getting started](getting-started.md). The same
facts are on the command line as [`detail version`](cli.md#version).

### Active processes {#active-processes}

**Active processes** lists everything in flight across every experiment (what it is, which experiment it belongs to, and when it started), plus anything that failed or was cancelled, which is the other thing you came looking for. Double-click a row to open that run's own page, where its bars are. Quick tools runs are not listed here: they run inside the window that started them, so watch one on its own page under **Quick tools**. Nothing is started from here; runs are launched from an experiment, and exports from **Export…** on any screen.

Three sections, because they answer different questions:

- **Runs** is processing, analysis, distill, datasets and models. It lists what is going and
  what failed or was cancelled; a finished run is on its experiment's own page. A failed or
  cancelled run stays listed until you delete it (or retry it to completion), however many times
  you have looked at it. The caption counts cancelled runs separately from failed and finished
  ones.
- **Exports** says what each export wrote and the folder it wrote to, and keeps your recent
  finished ones. An export's files land outside the experiment, so this is the only place that
  remembers where they went.
- **Imports** is video being copied in, plus your ten most recent finished imports. On a local
  disk an import is usually done before you get here.

A row names the recording; double-click it to open that recording. When a batch of imports
finishes, the strip under **Export…** names the recordings that came in and any that failed,
with a button to **Active processes**.

**Imports stop when you quit.** Runs do not: their workers are separate processes, which is why quitting offers to leave them going. An import is part of the app, so closing the window ends it and the recording is marked `failed` the next time you open that experiment; there is no resume. The quit prompt says which of the two you have in flight.

Open an experiment from **Experiments** to see its **overview** (identity, **Export…**, **Delete**, one summary row per section): it becomes the active target for new runs, and its sections list in the left rail under **Overview**, which is the overview screen itself and takes you back to it whenever you want it. Each collection screen is a table: **double-click a row** to open its detail or results.

Long operations never block the window: a run shows a progress bar and re-enables at a terminal status.

A **?** next to a field or section opens a short in-app explanation (from the bundled docs, so it works offline) with an **Open full docs ↗** link.

### Scrubbing frames

Every frame view (run viewers, input-image preview, field-of-view editor, stream preview) shares one scrubber, so the controls are the same wherever video appears.

**Every frame zooms.** Scroll over the picture to zoom in where you point, drag to pan, and press **Fit** (it appears in the corner once you have zoomed) to come back to the whole frame. The zoom stays put as you scrub or play, so you can follow one animal up close through a bout. Hold **Alt** while scrolling to scroll the page instead of zooming the picture.

- **Play** plays at the recording's frame rate and stops at the end. Press it again to start
  over. Touching the slider or a step button pauses. **Space** does the same.
- **Speed** runs from 0.25x to 4x, and the same speeds backward (**-0.25x** to **-4x**): a
  negative speed plays toward the start and stops on the first frame. Change it while playing
  to turn round. Video usually decodes more slowly backward, so expect the rate note below more
  often.
- **Prev frame** / **Next frame** step one frame. Hold either to speed up, to at most 3x the
  video's own speed; for longer distances drag the slider, use the keys below, or type a frame
  number.

Left of the slider is the **time** of the frame on screen, and right of it the time at the bar's
end (the video's length on a whole-recording view, the end of the window in a clip player), both
read from the recording's own frame timestamps. A view with no timestamps (an old
Quick segment run) shows no time rather than one worked out from the frame rate, which drifts on
video whose header rate is not its real one.

Some views draw on the slider itself: **orange brackets** mark a clip, with the footage around it
dimmed, and a **red tick** marks a point, such as a clip start you have marked and not yet
finished.

A clip player also has **Context: off · 1 s · 2 s · 5 s**, the footage shown either side of the
clip. The clip sits between the brackets, and playing starts at the left edge so you watch the
lead-in. A clip that fills nearly the whole bar (a cluster member that runs most of the video,
say) gets no brackets, since there is no context to set it apart from. Near the start or end of a video you get whatever footage there is. Each screen picks
its own starting amount: 1 s for a clip opened from AUDIT (a clustering run's or a behavior
scan's), whatever the montage's **Show ▾** context is set to for a clip opened from a clustering
run's INTERPRET (off until you change it), and off elsewhere. A change lasts until you close the
player.

**A clip in its own window** (double-click a clip, or a tile of the clustering page's montage)
opens in the **Advanced view**: untick **Advanced view** at its top for the plain one-picture
player, and tick it again to come back. **Blind mode** beside it names
recordings A, B, C… instead of by name, since a name can give away a treatment or group. The
window remembers both for next time. The same top row says how long the clip is and where it
starts in its recording (hover it for the whole line), and holds **◀ previous** / **next ▶**,
which step through the clips the window was opened with (a montage tile opens them in the
montage's order, and says so).

The clip window does not block the page that opened it: you can keep using the page while it is
open, and **double-clicking another clip shows it in the same window** (with that clip's
previous / next) rather than opening a second one. Each page has its own clip window, and it
closes when you leave the page.

The **As recorded** menu in the same top row (choose **Aligned**) turns the picture so the clip's animal faces
**up** in every frame: the footage, the masks and keypoints drawn on it and the two side panes
all turn together, about the middle of the animal's body. It needs the keypoints the embedding
read and a heading definition (see **Align pose** below); without them the menu is greyed and
says why on hover. A frame where the body's axis cannot be measured plays as recorded, with a
small *no axis* mark in its corner. The window remembers your choice.

**Export ▾** in the top row writes what the window shows. **Video of what you see…** and **GIF of
what you see…** write the picture as it is on screen (the footage alone in the plain view, the
footage with its two panes in the Advanced view; every tick; the orientation) over every frame on
the bar, so the clip's context is included. **This frame as PNG…** saves the frame under the
playhead. Each runs in the background; video and GIF show a progress bar with **Cancel**, and the
window says where the file went.

The Advanced view is laid out so you can see the behaviour: the picture takes the window's whole
width and most of its height, and everything else is a thin row under it. The window opens
nearly as wide as the screen and only as tall as the picture and its rows need, so no empty band
sits above the picture or above **Close**. When the window is too narrow for the controls under
the slider to fit on one line, the view ticks move to a second line together, with the frame
counter at its end. It shows three pictures
of each frame. The footage is on the left, large; beside it, the **Masks** pane shows every
tracked animal's outline on white (the clip's animal in orange, any other grey), and under that
the **Keypoints** pane shows the points on white (the clip's animal in colour, any other grey).
The side panes are never drawn over the footage. For that, the footage has its own two ticks
under the slider: **Masks** tints the clip's animal, and **Keypoints** draws its points. The
points are the keypoint run the embedding used (an embedding aligned by keypoints, or one that
drew or appended the pose). An embedding that used none has no points to show, so the Keypoints
pane says so and its tick is greyed. **Loop clip** plays the clip itself over and over, without
the context either side.

Under the slider, on the slider's own timeline, thin strips in small print show what the clip's
animal is doing, read from the embedding's kinematics: **speed** and **area** (its mask area, in
pixels, or in mm on a recording with a [distance calibration](concepts.md#distance-calibration)),
and under them a grey lane marking the frames where the animal has no tracked row. Each is
scaled to the window, and the number at the right end is the value at the playhead; hover a
strip for its value and unit there. The strips follow the embedding's own frame cadence, so a
run that embedded every third frame has a point every third frame. An embedding made before
kinematics were kept shows no strips and says to run `detail recording backfill-kinematics`.

**Trail** draws where the clip's animal has just been, over the last half second: each keypoint's
path, in that point's own colour (blue for a left point, vermillion for a right one, green on the
midline), fading into the past, on the footage and on the Keypoints pane. **Trail points** beside
it chooses which points leave a trail (the paws only, say); every point does until you untick
some. With **Align pose** on, the Keypoints pane's trails are each point's path *relative to the
body* (every past frame turned by its own body axis), so they show a paw stepping or the head
turning, not the animal crossing the arena; the footage shows that. A run with no keypoints has
no points to trail, so **Trail points** is greyed with the reason and **Trail** draws the centre
of the animal instead, in white.

**Mask trail** draws the clip's animal's masks from the last half second under its current one on
the Masks pane, fainter the older they are, so you can see how its outline moved and changed
shape. Only the clip's animal leaves one. The past masks are read from the detection run rather
than decoded from the video, so it costs playback little.

When the embedding read keypoints and a **heading** definition says which points are the front
and the back, three more strips show the body's shape: **length** (the front-to-back length,
where 1.00 is the animal's usual length, so a stretch reads above 1 and a hunch below), **bend**
(degrees at the body's middle, 0 for a straight body; it needs a point on the midline between the
front and the back, and is drawn either side of zero by the way it bends) and **turning** (degrees
per second; positive is clockwise on screen). **Align pose** redraws the Keypoints pane with the
clip's animal alone, centred, front up and at its usual size, so postures compare from clip to
clip whichever way the animals faced. The heading is the one the embedding was aligned by;
otherwise the experiment's one saved heading that fits the skeleton (make one with **Define…**
beside **Heading** on the embedding form). With none, or with several and an embedding aligned by
none of them, the three strips are left out and **Align pose** is greyed, with the reason on
hover.

**Click the Masks or Keypoints pane to open it large.** The pointer turns into a hand over either
pane, and hovering says **Click to enlarge**. The pane opens in a window of its own that you can
resize to the whole screen, drawn at the video's full resolution rather than the thumbnail made
bigger; clicking the pane again brings that window forward rather than opening a second. It has
its own **Play**, **Prev frame** / **Next frame**, slider, speed and **Loop**, and starts on the
frame the clip window was showing. It follows the clip window as you work: tick **Mask trail**,
**Trail**, **Trail points**, **Align pose** or **Context** there and the large pane changes with
it, and **◀ previous** / **next ▶** move it to the next clip. It closes with the clip window, and
stops playing when you close it. It draws from the masks and keypoints already stored for the
clip and never reads the video, and so do its exports; the one exception is a recording whose
record carries no frame size, where it decodes the footage to size the pane and says so at the
top. **Export video…** writes the pane as an mp4, every frame at full size. **Export GIF…** first
asks for the GIF's width (640 px unless you change it) and frame rate (12 fps), then where to
save it; the GIF keeps real time at any rate, so a lower rate skips frames rather than playing
slower. A GIF times its frames in hundredths of a second, so at a rate that does not divide 100
(12 fps, say) the frames take turns at 80 and 90 ms, and the whole GIF lasts as long as
the stretch it shows, to within a hundredth of a second. Both cover the frames on the window's
bar: the clip itself, or the clip with its context when **Context** is on, and the GIF dialog says which. A bar shows the export's progress, with
**Cancel**; the window then says where the file went.

When the video cannot decode at the speed you picked, the rate it reaches is shown beside the
speed menu (`72 fps`). On a narrow pane the row of controls wraps onto a second line rather than
shortening its labels.

The slider is coarse on long video, so use the keyboard:

- **← / →**: step one frame (hold to speed up, to at most 3x)
- **Shift+← / →** or **PageUp / PageDown**: big jump
- **Home / End**: jump to the ends
- **type a frame number** + **Enter**: go straight there

Two surfaces show frames and deliberately do not play:

- A **stream preview** for a recording with no video. There are no frames behind it, so the
  slider is a time cursor over the stream's span.
- A **Quick segment** run whose frames were sampled at random, scattered across the clip, where
  no speed would mean anything. A run sampled evenly, or read consecutively, does play, at the
  rate those frames represent.

## Configuration

Machine-global setup, shared across every experiment. The landing page is a **Get started** checklist (GPU / CUDA, Hugging Face access, the SAM 3 backend package, the SAM 3 weights, **See it segment a clip**, which opens [Quick segment](tools.md#quick-segment), and **Recommended: distill a fast model**), each with an **Open** button. On a machine with no NVIDIA GPU the two SAM 3 steps are marked as needing one, **Install a runnable detection model** is added after them, and **See it segment a clip** says it needs an NVIDIA GPU or an installed detection model, because Quick segment can only run SAM 3 on an NVIDIA GPU. It reads live state and marks each step ✓ when done: **See it segment a clip** once a Quick segment run has finished, and **Recommended: distill a fast model** once a distilled model is in the Model manager.

**Check readiness** re-runs the checks. It probes the environment in a fresh subprocess, so if you install something in a terminal while the app is open (a CUDA PyTorch wheel, the SAM 3 backend package), clicking it picks the change up **without restarting the app**. A newly installed SAM 3 backend is then usable straight away, Quick segment included. A reinstalled PyTorch is not: the running app keeps the PyTorch it started with, so restart DETAIL to use the new build. The same checks run headless as [`detail doctor`](cli.md#doctor). SAM 3 is only marked ready when its weights *and* its backend package (`import sam3`) are both present, so a run can't pass the check and then crash mid-run.

A step still to do carries a dismissable amber banner on its own screen, naming what is missing.

**Preview cache**, under the checklist, shows where this computer keeps its video previews and how much disk they use. Previews are small copies of each video (480×270 at most, 15 frames a second) and of each tracking run's masks, for clip montages to play from. They live outside every experiment, are never analysed and never exported. **Clear** asks once in place (the button turns into **Clear 1.4 GB** beside **Keep**); clearing deletes no experiment data, and the previews are rebuilt in the background when next needed. Expect tens of megabytes per hour of video, roughly twice that on Windows and Linux.

**Previews build in the background.** When a recording finishes importing, when you open an experiment with previews missing (a new computer, an imported archive, a cleared cache), and when a tracking run finishes while its page is open, the missing previews are queued. A bar at the right of the status bar shows the one in hand: which recording and part, whether it is the video or a tracking run's masks, how many are waiting, and the time left. It disappears when the queue is empty. One builds at a time; you can keep working. The same build is [`detail cache previews build`](cli.md#cache).

**Previews are built once per video on each computer, so keep them.** Building takes a while, so a long experiment can take hours, and clip montages wait until their recordings are built. Clearing is safe, but you wait for that again. Updating DETAIL keeps the previews you have; an update that needs to rebuild them says so before it starts, with what is rebuilt and roughly how long.

**GPU / CUDA** shows the detected accelerator. When CUDA is broken on an NVIDIA box, the *CUDA doctor* names the gap and gives a copyable fix command plus the driver and PyTorch links. It shows how it chose the build: the newest CUDA your driver supports, and the newest PyTorch build at or below it (CUDA 13.2 gives `cu132`). Under the command it reminds you to restart DETAIL once the install finishes. On a machine without an NVIDIA GPU the panel is greyed, not hidden.

**SAM 3 backend** gives the full copy-paste sequence to install SAM 3 into your environment, from opening a terminal to verifying: SAM 3 ships off PyPI, so it must be cloned and installed from source, and the detail package needs its `sam3` extra. Each OS gets one block, every command under a short `#` description; the block for the machine you're on uses the **real paths** the tool detects (your detail folder, your environment), the other OS's shows the relative forms and its own activation and Triton line. The block for your OS has an **Open Terminal** / **Open PowerShell** button (named for your platform's shell) next to **Copy** (it drops that terminal into your detail folder so you can paste the commands (greyed on Linux, where you open a terminal yourself). Run the sequence, then click **Check readiness** and the status flips without restarting the app. When the machine has no NVIDIA GPU, a note above the status says so) SAM 3 needs one to run, though the commands are still shown.

## Experiments

One card per experiment in your workspace: a still from the middle of its footage, the recording count, when it last changed, and how many runs of each stage have completed (`detect 2 · track 2 · embed 1`). An experiment with no video, or whose video is not on this machine, shows a placeholder instead of the still.

The cards appear at once; the still and the counts fill in a moment later. On a workspace held on a network share that moment is noticeable: the reading is the slow part, not the drawing.

- **New experiment**: name and optional description; created in the workspace.
- **Import experiment**: pick a `.tar` archive from [`experiment export`](cli.md#moving-experiments-between-machines) or the Export dialog's **Experiment bundle**. A same-id collision asks whether to replace, and first lists what replacing deletes that the archive does not bring back: video files (how many and their size) and datasets when the archive has none, export records, and earlier archives saved in the experiment folder. A bar shows how far the import has got (extracting, then verifying the archive against its checksum, then finding video used in place) and the time left, and **Cancel** stops it with nothing placed; an experiment it was going to replace is left as it was. Video used in place is looked for in the folder the exporting computer kept it in; recordings whose video still isn't found are named when the import finishes. A file at the recorded path that is a different video counts as not found.
- **Double-click a card** to open its project view.

## Experiment overview {#experiment}

The experiment's own screen, and the first one you see when you open it. **Overview** in the rail comes back to it.

Edit the experiment name and description, then **Save**.

- **Sections**: one row per section with a live count (recordings and how many have video, datasets and models (and how many are still being made), runs and how many are going or failed, analyses, comparisons) and an **Open →** into each.
- **Export experiment…**: opens the [Export dialog](#export) with **Experiment bundle** ticked: the whole experiment as one archive another DETAIL can import, with boxes for source video, training datasets and gzip. Clicking **Export** takes you to the export's page, which shows a bar, the phase it is in and the time left; the export runs in the background, so you can leave that page, and it stays listed under [Active processes](#active-processes). The archive is named after the experiment.
- **Delete experiment**: confirms first, then removes everything.

## Recordings

Lists the experiment's recordings and adds new ones.

- **Add recordings**: one row per recording, a name and an optional video file (a metadata-only recording is valid). A **Reference the video in place (don't copy it into the experiment)** checkbox uses the videos where they live instead of importing copies, for large libraries you can't duplicate ([Concepts](concepts.md#video-part)); the tool remembers each file's location automatically (nothing to configure). Imports run in the background; recordings appear at once and fill in as video imports. While they run, the status bar at the bottom of the window shows a progress bar, the recording importing now and its step (reading timestamps, copying), how many are done, and the estimated time left.
- **Import bundle**: bulk-import recordings (video + streams + grouping variables) from a manifest folder, the GUI form of [`recording import --bundle`](cli.md#bulk-importing-recordings). See the [Manifest reference](reference/manifest.md).
- **Import behavior labels**: batch-import ground-truth labels from a labels manifest, the GUI form of [`recording import-labels`](cli.md#importing-behavior-labels-ground-truth). Pick the manifest and importer (interval CSV or BORIS).

Open a recording (rail child or table row) for its detail screen.

### Recording detail

Every recording-level input lives here (run outputs are read-only). The top shows the recording's **name**, its **id**, where its video came from, and its **recording information**; below, the page splits into **DATA PIPELINE** (inputs that affect processing) and **ANALYSIS** (definitions used after the fact).

**DATA PIPELINE**

- **Video**: the imported part(s) and their on-disk location. **Add video part** opens the same reference-in-place checkbox as Add recordings. While it copies, **Cancel** reads **Stop**: **Stop**, Esc or closing the window stops the copy and removes what was copied, and the window closes once it has. A referenced video is not copied, so there is nothing to stop: it is added, and the page shows it. For a **referenced** video the location is the resolved external path; if it moved or changed, a status line flags it and **Locate video** points the part at its new file (accepted only if the content matches: a changed video means remove and re-add). For a **copied** video that is not on this computer (an experiment imported without its video), the status line says so and **Locate video** copies the original file back in. It offers the file the video was copied from when that is still on this computer, and accepts a file only if its frame times match the recorded part's exactly (the same number of frames, each at the same time) and its resolution matches; a video copied in by this version also has to have the same content. A bar with **Cancel** shows while it checks and copies, and a cancelled copy leaves nothing behind. You can move or rename the file while its recording page is open: the app lets go of a video a couple of seconds after it last showed a frame from it. On an experiment made on another computer, locating one video is enough: every recording whose videos are in the same folder finds them too. (Importing an experiment already looks where the exporting computer kept the video, so this is only needed when that folder isn't reachable from this computer under the same path.) A changed/missing referenced video also flags the runs below as possibly stale. **Move up** / **Move down** move the selected part one place, and are locked once anything has run against them. The parts table lists the parts in play order and does not sort when you click a column header, so the order you see is always the recording's order.
- **Scene geometry**: one section for the four things you draw on a frame.
    - The **field of view**, a polygon that [bounds detection and tracking](concepts.md#field-of-view).
    - The **distance calibration**: two points a known distance apart, giving
      [pixels-per-mm](concepts.md#distance-calibration), so analysis reports millimetres.
    - [Analysis-only](concepts.md#roi-poi) **regions** and **points of interest**.

    The page lists what each one holds and who reads it. Beside the list, a **geometry preview**
    shows the first frame with everything currently defined drawn on it in its own colour, or
    just the frame if nothing is defined yet, or a note if no video has been imported. Click it
    to see the same picture at full size. **Edit scene geometry…**, centered underneath, opens
    the editor.

    **One canvas, all four.** Whatever you are editing, the others stay on screen in their own colours, so a region is drawn against the field of view it sits inside rather than on a blank frame. Pick a definition in the list on the right to edit it (it takes its own colour and everything else greys out, so the frame says which one your drags will move. Its checkbox hides it; the **+ Field of view / + Distance calibration / + Regions of interest / + Points of interest** buttons above the list start a new one, and they are the only way to start one (clicking the frame draws nothing, which used to be an easy accident). Scroll over the frame to zoom in and out (or use the **Zoom in** / **Zoom out** buttons above it), and drag with the right mouse button to pan; **Fit** returns to the whole picture) useful for placing a vertex precisely on a high-resolution recording. A new definition starts on the frame showing, so you can draw it wherever you are. Scrub to another frame and **Set keyframe here** to make a shape follow a moving camera; a region or point can be limited to part of the video with **Active window**. The distance calibration has no keyframes: it is the same on every frame, so you draw and edit it on any frame. **Save** writes every kind and leaves the window open. It lights up after any edit (a shape, a name, the known distance, the active window) and greys out when there is nothing pending, which is how you know your edits are in. **Close** asks first if they are not. **Save** checks everything before it writes anything: two regions with the same name, say, are refused with nothing saved, so fix the name and save again. Select nothing and the dialog is just a picture of the whole setup, every kind in its own colour, the cheap way to check that a copied setup landed right. A region limited to part of the video is drawn only on the frames it applies to.

    A field of view a **tracking run** depends on is shown and drawn but cannot be edited or deleted: the run recorded that polygon. Delete the run first; the row names it.

    **Carrying a setup across recordings.** These live at the bottom of the editor, beside what they carry, and they all obey the same rules: pulling a setup in and pushing one out are the same operation, so a field of view behaves identically whether you import it here or copy it from the other recording. **Export as CSV…** writes the whole setup (field of view, calibration, regions and points) to one shareable file. **Import from…** pulls the other way, from another recording in the experiment or from a CSV, and is the same picker with the arrow reversed: the same list of individual definitions, blocked for the same reasons, reported the same way. A CSV brings whatever it holds, and the report lists only what was added and what was refused: a file with one region reports that region. **Copy to other recordings…** pushes this recording's setup onto some or all of the experiment's others, instead of pulling one in at a time: tick the individual field of view, calibration, regions and points you want, not whole categories, so every tick is a specific thing. Anything that cannot travel is listed with the reason and cannot be ticked: a name every chosen recording already has, or a region limited to part of the video. A distance calibration only goes to recordings whose video is the same shape as this one's. Another shape means another rig, which needs its own calibration, so those recordings are greyed out with the reason when the calibration is ticked. Calibrate one recording per rig and copy from each. Then tick which recordings to copy onto, listed by **name** (with the video file name beside it when that helps).

    **Nothing is ever overwritten, in either direction.** A field of view or a distance calibration only lands on a recording that has none, delete the existing one first. Regions and points are sets, so only the *names* already taken are refused and the rest arrive, which is what lets you pull one more zone in from a template. A recording that already has one of the ticked setups keeps it, and its row says so before you commit, greyed out if it already has *all* of them, still tickable if it can take the rest (a recording with a field of view but no regions still receives the regions). To change a setup that is already there, delete it on that recording first; if a tracking run is holding its field of view, the row names the runs you would have to delete.

    Two things about a setup don't travel between recordings, and the dialog says so before you click: **camera motion**, a field of view with keyframes is copied at its frame-0 position, since a pan belongs to the recording it was drawn on, and a **region with an active frame window**, which isn't copied at all, because those frame numbers mean something else on another recording. One "Copy to N" click runs it and reports everything that was left as it was, for every recording, each line naming the recording by name, and any shape whose camera motion was dropped. The equivalent CLI commands are `experiment propagate-fov` / `propagate-calibration` / `propagate-rois` / `propagate-points` (see [CLI](cli.md#experiment)).

**ANALYSIS**

- **Streams**: **Add stream** imports an aligned time-series. Choose the **data file** first and the dialog identifies it: MED-PC operant events (`.txt` + `.MPC`), your own CSV as a sensor trace (**Time-series CSV (sensor data)**) or timestamped events (**Events CSV**), or **behavior labels** from an interval CSV or a **BORIS** export. Change the importer above if the guess is wrong. A CSV must be saved as **CSV UTF-8** and comma-separated; one that is not is refused with a message saying what to change. A BORIS **aggregated** export (one row per bout, with *Start (s)* and *Stop (s)*) is identified as BORIS and refused, as in the CLI: export the observation again with **Export events > Tabular events**.

    The rest of the form follows from the importer. A CSV that isn't already shaped for DETAIL (an arbitrary export with its own column names) shows the file's **actual header**: say which column is time and what it holds (a frame index, seconds, raw `t_ns`, or a **UTC timestamp**, where the first row becomes `t=0`), then tick the columns to bring in. Ticked columns become **channels of one stream**, so a pose file with four channels is one stream you align once, not four you align four times.

    A MED-PC file that holds several boxes shows a **Session** list (subject, box and start time for each), and a BORIS export that holds several observations or subjects shows **Observation** and **Subject** lists. Pick the one this recording is: Import stays greyed until you do, and they are never merged. A file holding one of each shows the row greyed with what it holds. The choice belongs to this file, so **Save as…** does not keep it.

    **Alignment is part of importing.** Type the offset in seconds, or click **Sync to video…** to scrub to the frame where the stream starts and let it work the number out, either way nothing is written until you press Import. A stream on the video's own clock (behavior labels, or a CSV whose time column is a frame index) is anchored already, and the row is greyed to say so; for a CSV it follows the time column, so picking a different one re-checks it. The streams table refuses an offset for such a stream too. Change it later from the streams table: **Sync to video…**, or **Edit offset** to type a number. The table's **Rows** column fills in a moment after the page opens, because counting a long trace means reading all of it; a stream whose file is no longer in the recording folder shows **missing** there.

    **Reuse a mapping.** Once a CSV mapping works, **Save as…** names it and **Reuse a mapping** applies it to the next recording's own file, it pre-fills the time column, the kind and the ticked channels, and says so if that file is missing a column the setup names. The mapping stays chosen: pick a different file and it is applied to that file too. The list holds only the mappings saved under the importer selected above it, each shown with that importer's name, so pick the importer first; a mapping saved for one importer is not offered under another. A mapping's name may use letters, numbers, `_` and `-`, no spaces, because it becomes the stream's name on every recording you apply it to. Managing the saved set (delete, or move one to another experiment) is CLI work: `detail experiment list/delete/export/import-stream-preset`. To do a whole experiment at once without the GUI, see [`apply-stream-preset`](cli.md#reusing-a-mapping-across-recordings).

    **Preview** puts the video at the top, scrubbable, with the stream's timeline under it: one lane per label for an event stream, one lane of sample ticks for a continuous one. A program with more lanes than the window has room for scrolls; make the window taller to see more of them at once.

**Run history** (bottom): this recording's detection / tracking / embedding runs. Double-click a row for its results. Both GUI batch runs and per-recording CLI/API runs show up here.

**Delete recording** asks first, then removes the recording and everything it owns. It is refused while a run is still using the recording: a run going on it, or a processing, analysis, distill, dataset or export run that includes it. The message names each run; cancel it or let it finish, then delete. A finished processing run that included the recording keeps its other recordings and no longer lists this one.

## Recording information

One row per recording. The first two columns are always there: `recording_id`, then **file**: the video's name, read from whatever you imported (the full path is in the tooltip). After that, one column per thing that varies across them (treatment, sex, day). The table is view-only until you click **Edit**. `recording_id` and `file` are always read-only. In view mode, double-click a row to jump to that recording's page.

Edit mode is one transaction, staged in the table until you **Save** (or **Cancel** to discard):

- change cells directly (paste from a spreadsheet works),
- **Add column** adds a variable,
- **Delete column** removes one and its values.

You can also add a column from a single recording's page, columns are experiment-wide, so it appears (blank) on every recording.

After your own columns, a fixed set of **completion** checkboxes (Field of view, Distance calibration, ROI, POI, Streams, Detection, Tracking, Embedding) tick themselves in as you work on a recording. Hover one for the detail (how many field-of-view polygons are set, how many detection runs have completed). They're read-only and never exported: a glance at whether an experiment is finished, not stored data.

For bulk work, **Import CSV** merges a filled-in CSV and **Export CSV** writes a fill-in template, the GUI form of [`experiment import-metadata` / `export-metadata`](cli.md#recording-information). Both live in view mode. Save the sheet as **CSV UTF-8** and comma-separated (in Excel, Save As → CSV UTF-8 (Comma delimited)); a sheet saved with semicolons or in another encoding is refused with a message saying which.

**Import matches your sheet on `recording_id`, `name`, or the video file name**, whichever you have. Hover **Import CSV** for the list. A sheet made before you imported the videos usually has the file name: head that column `file`, `file_name`, `filename`, `video`, `video_file` or `video_name` (any case). A file name matches on its base name, in any case, with or without the extension, against any of a recording's video parts. If the sheet has none of those headings, the import asks which column holds the video file name. Pick it from the list of the sheet's columns, or cancel.

A key column only *matches* rows; it never renames anything (rename on the recording's page). Blank cells keep the existing value. A key that matches no recording, a file name two recordings share, or two rows for the same recording are all reported, and nothing is applied. **Export CSV** writes `recording_id`, `name` and `file` before your columns, so a filled-in template imports back on any of them.

### Names, ids, and where a video came from {#recording-identity}

Every recording has three things the tool keeps straight:

- **Name**, yours. It starts as the video's filename and you can change it whenever you like. It's what you see in the rail, the tables and the run forms, so it's worth making it something you'll recognise. Names have to be unique; renaming to one that's taken is refused.
- **Id** (`recording_000`): the tool's. Assigned in order, never reused, never changes. Run records, exports and the CLI speak it, so it's shown on the recording's page when you need to quote it.
- **Source**: the full path the footage came from. For a referenced video that's where it still lives, and it follows the file when you **Locate video** it. For a copied video it's where it was imported from: the tool holds its own copy, so the original can move or go away without affecting anything.

The point of the split: **never rename a video file to organise your data**, rename the recording instead. The file can move (relink it) and a recording can be built from several files, without either changing what identifies it.

### Notes {#notes}

Every page for one thing in an experiment has a **Notes** box: the experiment overview, a
recording, a processing or analysis run, a dataset, a model, a distill or export run, and a
comparison. It sits under the details and above the results. Write what the run was for or what
you noticed in it.

Notes start **folded**, so they take no room until you want them. The **Notes** button at the top
right of the window, beside **Export…**, **Version History** and **Info**, opens the box on the
page you are on and folds it again; it reads **Notes ●** when a note is saved there, so you can
tell without opening it. It opens the notes **only where you are**: every other page stays folded,
and a page opens folded every time you come to it. On an analysis run with a view open (Interpret,
say), the notes open directly under the view's top row; going back to the run's page shows them
folded there unless you opened them there too. On a page that has no notes the button is not
shown.

There is no Save button. The note is saved a moment after you stop typing, and again when you
click elsewhere or leave the page; the time of the last save shows beside the box. You can type
while a run is still going. Notes are kept in the experiment folder, so they are there the next
time you open it, and an export of that entry includes them as `notes.md`.

### Leaving a page with unsaved edits

If you've typed into a table or a form and navigate away without saving, you're asked: **Save**, **Discard**, or **Cancel** to stay where you are. Quitting asks too. A save that can't go through (a name another recording already has) tells you and keeps you on the page.

## Processing pipeline, launching and monitoring {#runs}

**Processing pipeline** is the experiment's run index and single launch point.

### Launching a run

**Launch a run:**

1. Pick a **module**, listed in pipeline order (**1. Detection** · **2. Tracking** · **3. Embedding**).
2. Select the **recording(s)**. Only recordings whose upstream dependency is satisfied are eligible. **Select all** / **Clear** toggle every eligible one.
3. Fill the module's **config** form, then run. The run fans out per recording.

Before the run starts, every selected recording is checked, and over many recordings that takes a
while. The line under the form says **Validating** and names the recording being checked with its
place in the selection, for example `Rat 12 (3/48)`. An embedding run reads each recording once
more beforehand for the settings the whole selection shares, such as its background colour
(**Resolving shared settings**).

While it checks, **Cancel** reads **Stop**. **Stop**, Esc or closing the window stops the check
before the run starts, and the window closes once it has. If the run had already been started,
it cannot be stopped from here: the window closes and the run's page opens, where **Cancel run**
stops it.

**Advanced Settings.** Every config form shows the parameters you are likely to change and
keeps the rest behind one **▶ Advanced Settings** toggle at the bottom, under a divider and
labelled with how many are inside. Click it and they all appear at once; a form with a lot of
knobs sub-divides them under headings, *Segmentation*, *Clustering*, *Tracker parameters*.
The window opens tall enough to show every setting outside **Advanced Settings**, and grows
when you pick a module with a longer form, up to the height of your screen.

When a hidden value differs from what the form opened with the header says so
(**▶ Advanced Settings (11 · 2 changed)**), and the section opens itself when **Start from**
seeds the form from a run that changed one, so nothing that made that run different stays
folded away.

**Start from.** Every launch form has one row under **Module**, and it is where a run begins:

```
Module:      [ Behavior clustering                        ▾ ]
Start from:  [ Defaults                                   ▾ ]
             │ Defaults
             ├── previous runs ─────────────────────────────
             │ an_17 · 1 recording · window_s=1.5 · 4 Sep
             └ an_09 · (no settings recorded)
```

- **Defaults**: where the form already is. Picking it back returns the form to them.
- **Previous runs**, newest first, each labelled with what it changed, so you can tell two
  apart without opening either.
- **Baselines**, named starting points for a particular rig or subject, listed above the runs
  when there are any. None ship yet: a baseline is a recommendation to everyone using the tool,
  so it waits on numbers measured rather than guessed.

Each pick replaces the last, rather than layering on it: the form is whatever you picked, and
nothing of the previous choice is left behind.

Picking one fills the form. If what it changed is hidden under **▶ Advanced Settings**, that
section opens itself and says how many changed: nothing that made a run different stays folded
away.

**When a setting cannot be restored**, a line under **Start from** says which and what the form
has instead. A choice this computer does not offer, such as a model that is not registered
here, stays on the form's default; a number outside what the form allows now is set to the
nearest value it does allow. For example `Confidence threshold: that run used 1.7 (outside what
this form allows); this form has 1.0.`

**What it does not carry** is your recordings or the run's upstream artifact: the embedding
run or the tracking run. Those are chosen for the run you are making, not copied
from one you pointed at, and a run id means something different in another experiment. Check
your recordings and the form offers the upstream runs they actually share.

A run has to have recorded its settings to be worth starting from. Runs made before the tool
recorded them are listed greyed out and say so rather than vanishing,
`detail analysis status <an_NN>` shows the same thing: no **settings** block means there is
nothing to carry. New runs record them.

### Keypoints

The keypoint stages place a **skeleton** on each animal: the points, the edges that join them and the
left/right pairs. Skeletons are drawn once, in the [Keypoints manager](#keypoints-manager), and
shared by every experiment.

A patch-matching keypoint model learns from an **example set**: a few frames of this experiment's
videos with a skeleton's points clicked on them. An example set has no page of its own. Wherever
you choose a model that needs one, the same three rows appear:

```
Skeleton:     [rat overhead ▾]
Example set:  [overhead rats ▾]  8 / 8 frames
              [New example set…]  [Mark frames…]  [Delete set…]
```

**Skeleton** lists the library, presets included. **Example set** lists this experiment's sets on
that skeleton, with how many of their frames are fully marked. Sets made before the library existed
are listed under the skeleton they match, or under *(no library skeleton)*. **New example set…**
opens a small window with the set's **Name** and **Skeleton** (opening on the skeleton chosen
above); **Create and mark frames** makes the set and opens the labeling screen on it. **Mark
frames…** opens the labeling screen on the chosen set. **Delete set…** asks first, and is refused
while a keypoint run uses the set. A set's skeleton can still change in the labeling screen until
its first mark; after that it is fixed, and a different skeleton is a new set.

The **labeling screen** (after **New example set…**, or **Mark frames…**) is a video
player with the set beside it. Choose the **Video** (every recording with a complete tracking run,
one entry per run) and the **Track**; the **Skeleton** beside them changes the set's skeleton
until the first point is marked, and is greyed after that. Find a frame by playing, scrubbing or
stepping; then click
the animal to place the keypoint named in the **Annotating** box. The first click on a frame adds
it to the set with the track's box; each click saves at once and moves on to the next unmarked
point. A placed point can also be dragged to move it (the same drag as the two editors); the
move saves when you let go and stays on that point rather than advancing. **Not visible (N)**
records that the point cannot be seen on this frame, **Clear
(Backspace)** unmarks it, **1-9** choose a point, **D** / **A** step through the set's frames,
which are also ticks on the slider. **Add frames** copies in frames the tool picks: **Varied
frames** reads every tracked frame's shape and chooses postures that differ (the typical ones
first), **Spread over time** takes one per equal time slice. The bar names the recording and
counts its frames as it reads them; **Cancel** under it (or closing the window) stops the draw
and adds nothing. **Remove frame** asks first when the
frame carries marks. The line at the bottom says when every keypoint has been placed visible on
at least one frame, which is what a run needs. The tool row offers **Point**; region and box
marks are reserved in the file format and greyed until something reads them.

Clicking marks is GUI-only. The CLI samples and lists frames and imports a set prepared as a
file; see the CLI guide's keypoints section.

**Keypoint detect** sits in the module list between Tracking and Embedding. Its form asks for
the **detection run** (one option per set of detection settings across the selected recordings),
the method and its size, and, for a method that learns from examples (`dinov3_match`), the
example set to learn from: the shared example-set picker (a skeleton from the
[Keypoints manager](#keypoints-manager), then this experiment's sets on it, with **New example
set…**, **Mark frames…** and **Delete set…**), shown only for a method that learns from examples.
The method list also offers this experiment's trained students, marked *trained in this
experiment*, and the keypoint models in the Model manager's **Global models**, marked *from the
model catalog*. A student needs no example set and no DINOv3 weights.
For a trained student the form shows **Minimum pose confidence** instead, opening on the
method's 0.25: a pose the student is less sure of is not stored. The decode, the left/right rule
and the crop size sit under Advanced and open on *auto*, which means the method resolves them
from the set's skeleton and records what it chose. There is no tracking run and no track list:
a detect run places points on every detected animal and leaves giving them to tracked animals
to keypoint track. **Start from** restores the method and its settings but asks again for the
detection run and the example set, which name things this experiment made.

**Test keypoints**, the form's last row, is the keypoints counterpart of detection's **Test
prompts**: it tries the method on a few tracked frames of the chosen detection run before a full
run, and writes nothing. It is greyed, with the reason, until the form has a method that learns
from an example set, an example set and a detection run with exactly one tracking run on it in
each selected recording. Its window has the same shape as Test prompts: choose
**Recordings** (all the selected ones, or one) and **Sample frames** (12 by default), then **Run
keypoint test**. Each frame shows the method's points, placed with the form's size, example set
and Advanced settings, so the test places them the way the run will. The **Score cut-off** slider
above the frames redraws without running again: a point is **filled** at or above the cut-off and
**hollow** below it, which is how a viewer or an export at that cut-off would hide it. A detect
run keeps every point and its score, so the cut-off here is only for looking. Below the frames,
the **Example set check** places each of your marked frames from the others and compares the
result with your own clicks, as a median error in pixels and as a share of body size; **Per
point** breaks it down by point. **Another sample** draws different frames, and double-click
a frame to see it full size.

A finished detect run opens in the detection viewer with every pose on the frame drawn,
coloured by role (two animals on a frame are two poses), and a line per video part: frames
walked, frames with at least one pose, and poses. A finished tracking run's page offers
**Run keypoint_detect →**, so the steps run in one line: detection, tracking, keypoint detect,
keypoint track. A finished detect run's page offers **Run keypoint_track →**.

**Keypoint track** asks for the detect run, the **tracking run** whose animals the poses are
given to, and a method. The tracking run opens on *The one on the detect run's detection run*:
the run finds it, and if that detection run has several finished tracking runs it refuses and
names them, so pick one from the list. **Match: share inside mask** (0.5) and **Match: mask grow**
(0.05 of body size) under Advanced decide when a pose belongs to an animal; both are starting
values, not measured ones. The method list opens on the detect run's own method, marked
*(default)*: `temporal_decode` for `dinov3_match`, `despike_smooth` for a trained student. Pick
the other and the form keeps your choice. The chosen method's settings appear under Advanced,
*Method settings*, and every one of them is part of the run's identity. It runs on the CPU in
seconds, so another tracking run, window or cut-off is a new run over the same detect run. Its
viewer is the tracking viewer with each animal's final points, a line per video part
saying how many poses were given to an animal and how many were not, a line per animal saying
how many of its frames were given a pose, a second loss row per track (frames the track had a
subject but no keypoints), a per-point coverage line, and an amber alert naming any point placed
on fewer than 80% of the track's frames with the one action that fixes it: mark more example
frames that show the point. A finished track run's page offers
**Run embedding →**, which opens the embedding form on the run's recordings with its
**Keypoint track run** already set to this run; choose **Egocentric alignment** yourself,
since turning or flipping depends on the camera.

**＋ Distill new model** with **Model type: Keypoints** turns the patch method, or finished
keypoint detect runs, into a fast **student**, a YOLO-pose model that places the same skeleton with no example set
and on any backend; see [Distilling a model](#distill). Each detect run in the teacher list says
which tracking run its poses go to (the one on its detection run); a ticked run whose detection
run has none, or several, holds **Distill** with the reason. To choose among several, use
`detail keypoints distill --tracking-run`. Once its run completes the student is
listed under **Models** with the source `distill_NNN`, and **Keypoint detect**'s method list
offers it beside `dinov3_match`, marked *trained in this experiment*. **Promote to global** on
its page copies it, with its skeleton, into the Model manager's **Global models**, and every
experiment's method list then offers it, marked *from the model catalog*. Export it there to use
it on another machine.

### Choosing the inputs

**Next stage in one click.** A completed detection run's page has **Run tracking →** (a tracking run has **Run keypoint_detect →** and **Run embedding →**, a keypoint detect run **Run keypoint_track →**); each opens the launch dialog on the next stage with this run's recordings checked. It carries no settings: the run you came from belongs to a *different* stage, so its parameters would mean nothing in the form you land on. Use **Start from** there if you want a previous tracking run's.

**Upstream picker.** A stage picks its input by [kind, not run id](concepts.md#detection-run): tracking's **Detection run** lists the detection kinds found across the selected recordings, each labelled with coverage (`N/M recordings`). Each recording uses its own matching run; any selected recording without it is dropped after you confirm. Recordings need not share a run id.

**Field-of-view picker.** Tracking's **Field of view** works the same way, and for the same reason: it lists the field-of-view **names** found across the selected recordings with their coverage, and each recording resolves the name to its own polygon. It used to list arena *ids*, which are minted per recording, so one entry meant a different region on each of them and every run finished looking fine. A recording that has no field of view by that name is dropped after you confirm, alongside any dropped for the detection run. Leave it on **(whole frame)** to track the full picture.

**Use every.** Each stage's form has one (the number of frames to step by) and beside
it, always, the rate that lands at:

```
Use every:   [ 4 ⌃⌄] frames    7.5 fps (source 30 fps)
Use every:   [ 1 ⌃⌄] frames    30 fps (source 30 fps)
```

Downstream it also says what share of its input that keeps, since `4` under a detection
run at 4 is *all* of the detected frames, not a quarter of them:

```
Use every:   [ 4 ⌃⌄] frames    7.5 fps (source 30 fps) · every detected frame
Use every:   [ 8 ⌃⌄] frames    3.7 fps (source 30 fps) · half the detected frames
```

It counts **source** frames everywhere, so a downstream stage can only go coarser by a
multiple of the one above it. The arrows step by that multiple and won't go below it, so
only a typed number can be out of range, and that says so (`⚠ the run above used every
4th frame, use 4, 8, 12, …`), rather than being quietly rounded.

Going coarser at *tracking* also makes linking harder: the subject moves further between
the frames being compared. Tracking's **Max missed frames** is in processed frames, so the form
shows what that is in seconds at the chosen rate.

A selection spanning different frame rates reads `7.5–15 fps (sources vary)`; with no
video imported yet, just `every 4th frame`. The source rate is the container's advertised
rate, so it can be off on variable-frame-rate footage: the frame count itself is exact.

**Picking a model.** Detection lists the built-in models, this project's trained models, and
your imported ones together. A model that can't run on this machine stays in the list, greyed,
with the reason on it: `sam3, needs CUDA` on a machine with no NVIDIA GPU, `: not installed`
when its weights aren't downloaded. The form opens on one that *can* run. If none can, it says
`No detection models available. Install models via Model manager.` and won't start a run.
Quick segment behaves the same way.

The embedding form is the exception: it always opens on the default model, `dinov3_vit` at
size `vit_large`, the same model the CLI and API use when you don't name one. If that size
isn't downloaded it stays selected, marked *not installed*, and the form won't start a run
until you install it or pick another model or size yourself. A different model or size
changes every vector a run writes, so the form never switches for you.

### Previewing before you commit

**Interactive iteration** (a button on the config form):

- **Detection**: **Test prompts** previews text-prompt candidates on sample frames; for a closed-set model (no prompt) it becomes **Test detection** and previews the model's own classes. In the form, **Text prompt** is what the model searches for; **Class name** is the label stored on each detection. The preview runs the model wide open and keeps every proposal, so its **Detection confidence threshold** slider re-draws the grid at any threshold without re-running, read the value you want off it, then set the form's **Confidence threshold** (which does filter what a run stores). A grid cell is a thumbnail: on a 1080p recording a mouse is a couple of dozen pixels across in one, which is not enough to judge a mask. **Double-click a cell to open that frame full size**, where you can scroll to zoom and drag to pan; it opens at the threshold the slider is on.
- **Embedding**: preview the masked input image and tune the renderer before running.

The embedding form's **Canvas scope** controls how the centered input images size their
canvas across several recordings. *Per recording* (the default) sizes each recording's
canvas to its own largest mask, so every subject fills the frame and body-size differences
are normalized away; *Shared across selection* uses one canvas for every selected recording,
keeping true relative size at the cost of a smaller subject filling less of the model input.
It is a no-op on a single recording, and runs made at different scopes can't be compared
with each other.

The max-contrast background has no such setting: a run over several recordings always picks
**one** color for all of them, from the animals' coloring pooled across the selection, so the
background can never tell the recordings apart. Embed every recording you want to compare in
the same run. Recordings embedded in separate runs can each get a different color, and a query
or clustering across them is then refused as *not comparable*.

**Preview input images** always renders each recording at its own canvas and background,
so it can show a different background color from the run's, and with *Shared across
selection* the subject slightly larger than the run will.

**Egocentric alignment** makes every animal face right before the model sees it, the way
Keypoint-MoSeq aligns: *Overhead camera: turn so the head points right*, or *Side camera: flip
so the animal faces right*. It needs a **Keypoint track run** on the same tracking run and a **Heading** definition, chosen from
the experiment's saved ones or made with **Define…** (tick the skeleton points whose mean is
the front and the ones whose mean is the back; a definition the run's skeleton cannot satisfy
is greyed with the missing points named). Under Advanced, *Fill heading gaps up to* says how
long a gap a frame's heading may be interpolated across (a frame in a longer gap is not
embedded and is counted as lost to heading), *Mirror hold* keeps a side view from flickering,
and *Facing-camera ratio* treats a foreshortened heading as undefined. The input-image
preview turns each frame the way the run will and shows no input where the run would embed
none. Alignment needs a cropped, true-scale input image, and an aligned run never compares
with an unaligned one.

The tracking config's hyperparameter fields adapt to the tracker you pick. See the [Tracking parameters](modules/tracking.md#tracker-parameters).

### While it runs

**Monitor.** While a run is active its page shows aggregate and per-recording progress; on completion, a results summary. On failure the per-recording table shows which recordings failed and why.

**Export… → Performance report** writes what the run cost: the GPU, driver, CPU, RAM and versions
of the machine it ran on; the wall clock per recording with frames, fps, peak RAM and peak GPU
memory; and how long each phase took (loading the model against processing, for instance). An
outlier row is the point, a recording that took three times its neighbours is a diagnosis, which a
single total hides. A distill run's page has the same export, with its labeling and training as
the rows. Frames and fps appear only for stages that count frames (detection, tracking,
embedding and the keypoint stages); a distill or an analysis shows its wall clock alone. It is a local file (nothing is sent anywhere), and it reads the run's own activity log,
so a run from before DETAIL kept one shows blanks rather than guesses.

Under the bar, while a run is going, is a row per **recording**: a bar each, the one in flight showing its stage and its own ETA, finished ones greyed with how long they took, failures in red with the reason, and a recording that was cancelled or stopped part-way (*cancelled*, *partial*) also in red, saying which. Past eight recordings the finished and the queued fold into counts (*7 done*, *12 queued*), so a 21-recording run is four rows rather than twenty-one, and the one that failed is not buried among them; click a count to open it. The rows stay in the order the run does them. Once the run finishes they give way to the **Results** list below, which carries each recording's produced run id and status. **Click a row to see that recording's results** underneath it: the video, the scatter, or (for a recording that did not finish) why. **Double-click** opens the recording itself. A long run scrolls inside the list, so the results underneath stay on screen.

Every run also writes `activity.csv` beside itself: a timestamped line per update, with both ETAs. Nothing in the app shows it; it is what the **Performance report** export reads and what to attach to a bug report.

**Retry.** A finished run with incomplete recordings shows **Retry N incomplete recordings**. It re-runs only those (failed, cancelled, or never started) into the same run, leaving completed recordings alone; the bar resumes where the run stands. A recording re-runs from the start. If a recording can no longer run (say its detection run was deleted), the retry stops and names each one. While it re-checks them, the status line names the recording it is on.

**Cancel.** An active run's page shows **Cancel run**. It stops the workers and marks the run `cancelled`, even a run left over from a previous session.

**Quitting with runs in flight.** Workers are detached and keep running if you close the window. If the open experiment has an active run, quitting prompts **Cancel runs & quit** / **Leave running & quit** / **Don't quit**. An **import** is the exception: it runs inside the app, so it stops either way and cannot be resumed; the prompt says so when one is in flight, and **Cancel runs & quit** marks it `cancelled` rather than leaving it to be found as `failed`.

### Reading the results

**The run's own details** (its module, config and what it ran from) fold behind **▶ Run details**, under a line naming the run's stage, status, date and cadence. The page is built to fit the window without scrolling: the video or the scatter takes the room the rest of the page does not, and shrinks with the window rather than pushing its transport off the bottom.

**Detection and tracking results.** A detection or tracking run's page plays the recording with each frame's boxes drawn on it (**Masks** adds the outlines), with a strip of detections per frame below, click the strip to jump there. A tracking run colours each track and adds a timeline of the frames where a subject was lost, on the same axis as the strip above it. With more than one subject, each row is labelled with its track. The frame [zooms and pans](#scrubbing-frames) like every other. Pick which recording to view from the list above the player.

**Embedding results.** An embedding run's results show three panes side by side: a 2-D PCA scatter colored by track, and, on clicking a point, that frame (titled **Frame x / last**, counted like the scrubbers) and the model's input image. Both images zoom and pan the same way. Large runs are subsampled. Needs `pyqtgraph` (the `gui` extra). An aligned run adds a loss row per track (frames the tracking run had a subject but no heading reached, so no embedding; click to jump), a line per track counting frames with their own heading, with an interpolated one, and dropped, and an amber alert when heading coverage is under 95% or the longest gap is over 2 s, with the two actions that fix it.

## Distilling a model {#distill}

**＋ Distill new model**, on an experiment's **Models** page and on the [Model manager](#models), makes either kind of model: **Model type** at the top of the dialog chooses a **Detector** (boxes and masks) or a **Keypoints** model, and the fields below it follow. Name and experiment are shared. For a detector it runs the model-fitting flow: generate a YOLO dataset, train a detection model, and apply it as an ordinary detection run. The trained model is usable in that experiment straight away, under **Models** in the rail; it joins the machine-wide **Global models** only when you [promote it](#models). Distillation pairs a **teacher** (a strong model that labels the frames, `sam3` by default) with a **student** (the fast YOLO trained on those labels), so the form's two model pickers are labeled that way.

**Models** in the rail lists the experiment's trained models, newest first. The **Source** column says where each came from: the distill run that trained it, or `trained` for one trained straight from a dataset. Double-click a row to open the model. Under it, **Distill runs** lists every distill run, including one that failed before it trained anything, which is why it is a table of its own; double-click a run to watch it, or to **Resume** a failed one. **Datasets** in the rail lists the training datasets. Double-click one to open it: its name, status, source run, frames, train/validation counts and classes, and your [notes](#notes). While it is being built the page shows **Cancel dataset**, which stops its worker; once it is not, **Delete dataset** removes its images and labels after asking, and cannot be undone.

Two buttons at the top of **Models** make a model:

- **＋ Distill new model** starts from nothing: it labels a frame sample, then trains on it, as one job.
- **Train from dataset…** trains on a dataset you already have, to retrain with different settings without labelling again, or to train on a dataset built from a detection run. It is greyed out until the experiment has a completed dataset.

Build a dataset two ways:

- from a completed detection or tracking run, or
- by sampling frames and labeling them with a prompt-capable model (the **Auto-label** dialog, no prior run needed). The **Teacher (labels the frames)** dropdown picks the model that labels (`sam3` by default), **Frame sampling** picks how frames are drawn (*stratified* spreads them across each timeline, *uniform* is random), and **Background share** caps how much of the dataset may be empty frames (where the model found none of the classes, kept with an empty label). **Test prompts** opens the same frame-grid preview detection uses.

Or do both at once: **＋ Distill new model** (on the **Models** page or the Model manager), with **Model type: Detector**, runs the whole thing as one durable job: label a frame sample with SAM 3, then train a YOLO detector on it. The form combines the labeling and training options (with the same **Test prompts** preview) and picks compute per stage (SAM 3 labeling needs CUDA). Under Advanced, **Background share** (default 0.1) keeps frames where SAM 3 found nothing, with an empty label, up to that share of the dataset, so the detector learns what an empty frame looks like. One progress bar folds `labeling → training`; under it, while a run is going, is a row per **stage** (Label frames and Train model) each a bar and its own status, because they are different-sized things: labelling is done or it isn't, training is at epoch N of the total. Once the training stage starts, its own **Training curve** and a **Stop and keep** button appear on this same page, below the stage rows (watching a distill run doesn't mean leaving the page to see whether it's still improving. The run survives closing the app, so reopen and it's still going, or **Resume distill** a failed one (it reuses a complete dataset and resumes a training that was cancelled or crashed). **Open trained model →** goes to the model's own page for everything the distill page doesn't manage) rename, weights, full lineage, and (once complete) **Promote to global**. The trained model is usable in this experiment at once.

**Model type: Keypoints** is the same shape for a keypoint student (DESIGN_PLAN §12 *Distill keypoints*). **Teacher** at the top says what labels the frames. **Patch method + example set**, the default, runs the patch method itself (`dinov3_match`) on the sampled frames, so no keypoint detect run is needed first: choose its **Method** and **Size** (a size not installed on this machine is greyed, and the form opens on one that is), the **Examples** (the same skeleton and example-set rows as the keypoint detect form, with **New example set…** and **Mark frames…**), and the **Tracking** whose boxes and masks it is given (every recording with that tracking is labelled; the line says how many). It runs on the **Label** compute under Advanced and needs the accelerator for any real number of frames. **Finished keypoint detect run** lists the experiment's finished keypoint detect runs instead, one line per run as `recording · kd_NN · method`, each naming the example set its run learned from and the tracking run its poses go to (`subjects from track_NN`, the one finished tracking run on the run's detection run), read while a busy overlay covers the form; tick one per recording (a recording with a single run is pre-ticked). A ticked run whose detection run has no finished tracking run, or several, holds **Distill** with the reason; `detail keypoints distill --tracking-run` chooses among several. Either way, name the student and set **Frames to label**: frames are drawn spread over time, only where every animal on the frame is tracked, so no animal in a training image goes unlabelled. Both teachers are raw: the patch method's labels are its placement on each frame alone, and a detect run's are its stored placements, with no smoothing over time. The example set's own clicks join them where the marked animal was alone on the frame, and audit frames (with everything within **Keep clear of audit frames** of one) are never used. **Score cut-off** (a point placed below it is taught as not visible) sits with the main fields, with a line saying what the teacher's score means and its range. When the teacher's method has a default the field reads *the teacher's default (0.5)*; the patch method has none, so it reads *required: set a value* and **Distill** stays greyed until you set one. **Test cut-off…** beside the field opens a window that runs the teacher on about a dozen frames drawn the way this distill would draw them, writing nothing: each frame shows its points **filled** where they would be taught as visible and **hollow** where they fall below the cut-off; double-click a frame to see it full size and zoomable. Drag **Score cut-off** to see the effect without running the teacher again; the line beside it counts the points that would be taught visible. For the patch method a **held-out check** sits below: each example frame is placed by the others and compared with your own clicks, as a median error in pixels and as a share of body size, at or above the cut-off and below it, with counts (fewer than 8 placements shows the count only). While it runs, the bar says which step it is on: reading the tracked frames, loading the method
(its weights, then your example frames), decoding frames, placing points (counted as the method
returns them) and the held-out check; **Cancel** stops it. **Another sample** draws different
frames; **Use this cut-off** writes the value into the field. Whenever **Distill** is greyed, the line beside it says why: an example set, a tracking run, a ticked detect run, a detect run with no single tracking run to give its poses to, or the cut-off. The training settings mirror **Train from dataset…**, with the **Student** a `-pose` checkpoint. The run has three stage rows, Label frames, Train model and **Register student**, the last filing the model under `role: keypoints` as `<name> distill_NNN` so **Keypoint detect** can offer it. A run that failed at register keeps its trained student; **Resume distill** only registers it. The **Distill runs** and **Models** tables' **Kind** column says `detector` or `keypoints`.

A trained model's detail page lists its lineage, classes, weights, and metrics (best-epoch and final mAP50), above them. **Trained with** says the Ultralytics version and the settings it was asked for (epochs, batch size, image size, patience); every other setting Ultralytics resolved is kept in the model's `model.yaml`. While a run is training (or after one has finished) a **Training curve** band plots mAP50 against epoch, marks the best epoch, and says so underneath: *Best so far: 0.832 mAP50 at epoch 31: no improvement in 7 epochs (of 60)*. The x axis spans the whole run from the first epoch, so the shape does not rescale under you as it fills in. A model imported from elsewhere has no curve, because it has no epochs here. Training runs on the best backend present (CUDA, else MPS, else CPU) unless you choose one. A training that was cancelled or crashed shows a **Resume training** button that continues from the last checkpoint.

**Stopping training early.** A run in progress shows two buttons, and they do different things. **Stop and keep** finishes the epoch in progress and then saves the best weights so far, use it when the curve has flattened and the model is good enough. The weights kept are the **best** epoch's, not the last one's, which is why the band marks the peak. It is not instant: training checks between epochs, so on real data expect a few minutes. The model then counts as **complete**, you chose to finish it, and its page says **Stopped early: at epoch N**. It has no **Resume training**: the run is finished, so there is nothing to resume; train a new model to train further. **Cancel model** stops the worker now and discards the run. A failed dataset or training run shows its recorded error. **Promote to global** copies a completed model into the machine-global catalog with its provenance (source experiment, dataset, base model, metrics), so any experiment can select it.

## Analysis

Experiment-level runs that turn derived data into analysis outputs under `analysis/<an_NN>/`. **New analysis** opens a launch dialog like a processing run: pick a **module**, check the eligible **recording(s)**, fill its config, and run. Each module's parameters are on its own page (linked below).

### An analysis run's page

Every analysis run opens on its **landing page**: the module, status and recordings, your
[notes](#notes), the run's snapshots, and **one card per view** of its results. Each card names
the view in bold as its headline, says under it what the view is for, and gives one line of status (*3 labels*, *2 recordings*). A
clustering run has four views (**Interpret**, **Audit**, **Browse**, **Measures**), a behavior
scan four (**Calibrate**, **Audit**, **Browse**, **Measures**), and a module with one result has a
single **Results** card. **Open output folder**, **Delete analysis** and **Start new analysis**
are on the landing page too.

**Click a card to open that view across the whole window.** The line at the top of the window
names where you are, for example *my_experiment › an_32 › Audit*: click the run name
(**an_32**) to go back to the landing page. Under it, a tab per view switches straight to another
without going back; a module with one result has no tabs. Run details and notes stay on the
landing page, so the view gets the window. While you read a [snapshot](#snapshots), a line under
the top line says **Read-only · *snapshot*** as well. The view you had open stays open when the page refreshes, for example when the run finishes.

While a run is still computing, its cards are greyed and say *Results appear when the run
finishes*; they open once it does.

### While a run is going

Its landing page shows a row per **phase** (the module's own steps, in order), so you can see which one is running and what is still ahead, the way a processing run's page shows a row per recording. The rows give way to the run's view cards once it finishes.

**A run that failed shows what went wrong, not results.** Its page carries the error the worker
recorded and **Delete analysis**; it does not open a workspace. That holds for the runs that stay
*open* while you interpret them too: an open run whose computing died is a failed run, and
opening one used to re-attempt the write that killed it.

**Results that are out of date.** Changing something on an open run, a label or a
cutoff, rewrites its results from what is already stored. If that rewrite does not finish (a
network share that stalls, a machine that sleeps), the run's page says the results describe an
earlier setting and offers **Rebuild results**, which re-derives them from the stored output and
your current settings. The expensive pass is untouched, so this is quick relative to the run; the
CLI equivalent is `analysis rebuild`. The change itself **was** saved: INTERPRET's message is
titled *Label saved, results not rebuilt*, and the error names the recording that could not be
read. Until the results are rebuilt, everything that reads them refuses and says so: measures,
a comparison, finishing the run, an export, a new audit batch and finishing a cluster's review.

**Labelling is quick; the per-frame table catches up later.** A label or a cutoff
commits in a few seconds however many recordings the run covers. The per-frame table that windowed
measures, **Finish**, exports and comparisons read is brought up to date the first time one of them
needs it (once for all the changes since, and only for the recordings those changes touched) with
its own progress bar.

### Location export

Writes per-track ROI/POI features (`tracks.csv`, `roi_relations.csv`, `point_relations.csv`). Tick a **tracking run** by kind (tick more than one when the recordings were tracked through different chains, as under [behavior clustering](#behavior-clustering)), and it writes each recording's CSVs to `analysis/<an_NN>/<recording_id>/`. **Rows per second** writes that many rows a second instead of every tracked frame, which is many times faster. The body columns measure the whole mask, tail and all. The detail page has **Open output folder**. For CSVs in a folder of your choosing, use the CLI `locations export --out <dir>`. See [Location export](modules/location_export.md) and [the concept](concepts.md#location-export).

### Behavior scan

Searches a cohort for behaviors you can already name, from a few example clips each, with no model. One run searches several behaviors at once. **New analysis… → Behavior scan**, check the recordings, pick the shared **embed run**, then press **Mark examples…**. The **Behaviors** table on the form is the summary: each behavior's examples, non-examples, how they spread over recordings and animals, and total marked time.

- **Mark examples…** opens the video to mark every behavior's clips in one pass. Under the playback controls, **Annotating: *name*** says which behavior the next clip goes to, with **example / non-example** and **Mark start** / **Mark end** beside it. After **Mark start** the button stays green and a red tick sits on the slider until you press **Mark end**. Marked clips are drawn on the slider in their behavior's colour, non-examples as an outline. Changing the recording or track drops a pending start and says so, and while the new track is read the Mark buttons say *loading…*.
- The right column lists the behaviors: click one to annotate it, and **Add**, **Rename** and **Remove** manage them. Each row has a bar toward three examples, which turns into a tick at three, and how many recordings and animals its examples come from. **Balance samples across** picks a recording metadata column, such as treatment, and the line under it names the first gap for the behavior you are annotating: all its examples from one recording, all of one animal, or a value of that column with none. The recording picker shows how many clips each recording already has. These are **advice, not refusals**: one clip works, and a genuinely brief behavior should be marked briefly, but more clips, longer ones and a wider sample measurably steady the result.
- A clip that is two behaviors at once is marked once under each: select the other behavior and mark the same span again. The marked clips are listed in the right column. Selecting one jumps the video to it, switching recording or track if it is on another. **Delete clip** or the Delete key removes it, and **Cancel** discards every change made in the dialog.
- The frame [zooms and pans](#scrubbing-frames), which is how you find the moment a paw leaves the floor. **Masks**, on by default as in every player, tints the subject's outline onto the frame so you can see the animal is being segmented the way you expect. It roughly doubles what a frame costs, so fast playback is slower with it on and the scrubber says so; untick it to play faster. When the subject's tracking has breaks, a **tracked** strip under the slider shades where it was seen (a fully tracked recording shows no strip; the status line gives the coverage either way), and a clip may cross a break in it: the example frames are pooled, so frames where the subject was not seen are skipped, and the status line names the break it crossed.
- The form shows an amber note when a behavior has fewer than three clips or a clip under 2 seconds.
- **Min match length** sets the shortest match the scan returns, in seconds; *off* returns every length. Matches shorter than it are never formed, so a brief flicker cannot hide a longer match next to it. It is part of the search, so changing it later means running the scan again.
- **Non-examples** are clips that are *not* the behavior; they join the labelled index alongside the drawn background, so a behavior you can point at the *wrong* examples of is scored more sharply.
- **Method** is in *Advanced settings* and defaults to `soft_knn`. Its page says what is and is not known about how well it retrieves. Without ground truth you cannot evaluate a matcher on your own data, so the default was chosen on measured cost instead.
- There is **no threshold** on this form. The search stores no cutoff, so changing your mind about one costs nothing and never a second search.

The run writes one opaque shard per behavior × recording × subject, and stays **open** rather than finishing complete. An open run is one you are still working on. Its views are the **calibration workspace**: **Calibrate**, **Audit**, **Browse** and **Measures**, each with the behavior selector at its top. The first three are described below in the order you use them: set a cutoff, look at what it kept, then measure how often it was right. **Measures** is the run's [measures](#measures) table.

- **Calibrate** is where the cutoff is chosen, and it is laid out the way the decision is made: what you are looking at on top, the ordering you are cutting in the middle, where it lands underneath. Top left, the **clip**, the whole video frame with the subject's mask tinted in, playing at real speed. Top right, the **curve** of how many instances each setting returns, with your cutoff marked, and under it the number box and **kept per recording**, a recording contributing far fewer than the rest is named, because a cutoff that keeps most of one animal's behavior and little of another's biases the very comparison you would run next.

    The middle is every match for the behavior in **one row, ordered by confidence, least confident on the left**, matching the direction of the axis of the curve above it. The divider standing between them *is* the cutoff: everything left of it is dropped, everything right is kept, and it says how many of each. Four things happen in that row and they are kept apart by where you do them: **drag the background** to scroll, **click a clip** to play it above, **double-click one** to open it in a window of its own, and **drag the divider**, the shaded band, or press its **◀ ▶** to move the cutoff. Dragging only looks; letting go commits. The steppers move one clip at a time, which is the finest step the evidence has, and they keep working once you have scrolled the divider out of sight. Clicking a clip also **marks** it, a heavy border, and the clip in the player. That is how you set the cutoff somewhere you have scrolled to: mark the clip you want to be the *least confident one kept*, then press **Cut here**, which names it and says what keeping from it leaves (`Cut here) keep from 0.7480 (108 kept)`. Two steps rather than one, deliberately: clicking a clip has exactly one meaning, so browsing the row can never move the line by accident, and the button says what it is about to do while it is still a button. Beside the row, **N dropped │ N kept**, **⤒ centre on cutoff**, which is greyed only when the row is short enough to need no scrolling, being able to *see* the divider is not the same as it being centred, and resizing the window moves it off centre without moving it out of sight, and **Cut here**, greyed until a clip is marked, and again once the line is already at it. The row is a **neighbourhood of the cutoff** (the 150 candidates either side) and nothing loads as you scroll it; to see past an edge you move the cutoff, which re-centres the row on the new line. That is deliberate: a query over a hundred hours of footage yields hundreds of thousands of candidates, and a row that tried to hold them all would be wider than Qt can paint. The ends say which end they are. **Drag the bar under the row** to make the frames bigger and fewer, or smaller and more; how many fit is otherwise simply how wide the pane is, so widening the window shows more.

    Below the clip row the page is four named sections (**Clips**, **Ethograms**, **Locations**, **Interpretation**): the same four, with the same names and the same fold and drag, on every behavior scan.

    **Ethograms** is one row per recording, where the kept instances actually fall across each video. It answers a question the row cannot: whether what you kept is spread through the session or bunched in the first two minutes. It is drawn when you open the behavior, and redraws when you **commit**, not as you drag, because it reads the instances the committed cutoff kept. Every row shares one time axis from 0:00, with a ruler above; scroll the wheel over the rows to zoom in around the pointer, drag to pan, and press **Whole** or **Esc** to zoom back out.

    **Locations** is where they fall: one square per recording, in experiment order, with a dot for every kept instance over the field of view. A cutoff that keeps a behavior only in one corner is a spatial bias nothing else on the screen shows. The row scrolls sideways and the bar under it makes the squares bigger; **click an instance to play it** beside them, the whole frame with the subject marked, and a click takes whatever is nearest so you never have to hit a dot exactly. The section is replaced by a note for a run whose embedding predates position capture; `detail recording backfill-kinematics` fills it in without re-embedding.
- **Browse** is the accepted set: every instance above the cutoff, with its recording and span, ordered by recording then time, chronology within an animal is what makes a systematic error visible. **Any column sorts**, so you can flip to confidence and back. Click a row to play it in the player beside the list's top rows; a long list scrolls in place. This is where a systematic error shows up, every match in one recording being the same wall-lean is invisible in a marginal band and obvious in a list.

A behavior with no cutoff is **absent** from the run's behavior output rather than present with zero instances: a named behavior with no rows means "we looked and it never happened", not "nobody has chosen a threshold yet". A point marked on the curve is one where the method stopped storing candidates: a curve that flattens because storage ran out looks exactly like one that flattens because the animal stopped, so it is marked rather than left to be read.

- **Audit** puts a number on how often the matches are actually the behavior, and it is where you spend a few minutes to earn a figure you can publish. The tab is split: reviewing on the left, what the reviewing bought on the right.

  **Start auditing**, in the left column, above the list it draws into, and reading *Review N more* once you have begun, where N is how many candidates are actually left, plays a sample one clip at a time, in random order and **without showing you the score**, knowing a clip scored 0.95 pulls the judgment toward accepting it, which inflates exactly the figure you are trying to measure. Answer **Yes**, **No** or **Unclear**, by button or by pressing **S**, **D** or **F**, home row under the left hand, so the right can stay on the mouse. The score appears after you decide, and a bar tracks the batch while the running total sits in the heading. That total counts this behavior's clips at its current cutoff (*52 of 68 instances audited*). Raise the cutoff past clips you have already judged and the heading says how many (*23 earlier decisions are on clips now below the cutoff*). Those answers are kept, and they count again if you lower it. **← and →** move back and forward through the batch, so a misclick on clip 3 is one keystroke to fix: a clip you have already judged shows what you said and takes a new answer, which supersedes the old one without discarding it. Going back does **not** show you the score again, a second look at 0.95 while you reconsider is the bias the blind review removes, and correcting a misclick does not need the number. When every clip in the batch has an answer the clip stops playing and a line beside the counter reports the batch, how many you reviewed and the yes/no/unclear tally. The clip stays where it is, so a misclick on the last one is still one keystroke to fix.

    The left column is how you choose what to look at. **WHAT TO REVIEW** offers *this batch*, *judged* (everything you have decided in this run, newest first) and *unclear*; click any row and that clip plays, so fixing a judgment from three batches ago is one click, not a walk. Switching between them changes what you walk through and records nothing: a clip you have not judged still counts as not judged. The list shows twenty clips and scrolls past that, so it stays beside the video rather than running the length of the window, and every row names its **recording** rather than the id underneath it. Resolving your unclears is the re-review worth doing: an unclear is dropped from the estimate, so each one you settle is a row the interval gets back. There is deliberately no yes/no scope, re-watching only your rejections moves precision in one direction, which is the figure the audit exists to measure honestly. Above the list, **Review N more** draws the next batch and names how many it will draw, and **Continue, N unjudged** offers exactly the clips you walked away from. Before your first judgement the draw button reads **Start auditing (N clips)**. When the pool is empty the button is greyed and says so. Twenty-five is a batch, not a limit, and each batch goes where the last left the sample thinnest. Leaving a batch unfinished costs nothing: the clips you did not judge are offered again next time, so switching behavior mid-review is free.

  The clip is the footage with the subject's mask tinted in, which is now what every query surface shows; there is no second pane holding the constructor's input image. The clip takes the whole width of its pane, with its transport directly under it and the three judgment buttons centred beneath that, one object, so a wider window makes the video bigger rather than moving the buttons away from it. It stops growing before it would push them off the bottom of the screen. On the right, three things that update as you judge. **BY BEHAVIOR** is every behavior's precision at once, **click a row to switch to it**, which is also how you see what is worth auditing next, since a behavior nobody has reviewed says so. **BY RECORDING** splits the current behavior per video (by **name**, the one you gave it), because one cutoff can behave differently in different animals and an overall figure hides that. **PRECISION BY CONFIDENCE** plots the rate against the score, with your cutoff marked. The shaded band is the 95% interval, so a part of the curve resting on two clips looks as uncertain as it is; a point drawn hollow rests on too few to estimate from. The strip underneath counts the clips you judged in each bin: a bar says how much of the curve there you have earned, and a grey circle on the baseline marks a bin that has candidates you have not looked at.

  Everything reads as three things together: the rate, the interval it rests on, and how many clips it came from. A row with too few judgments says so rather than showing a figure, because `1.00` from three clips and `1.00` from fifty look identical and mean different things. The sample is drawn across the **whole** confidence range and independent of your cutoff, so moving the cutoff afterwards re-reads precision from the clips you already judged, you never label twice for the same run. Each confidence bin gets about the same number of clips however many candidates it holds, so the figure weights each bin by how many of your cutoff's instances fall in it; it is not the plain share of the clips you said yes to. The same figure is what `detail analysis audit` prints. Clips that overlap one of your own example or non-example clips are never offered and never counted: they are what you told the search, so judging them would score your own answer. A line under the heading says when the figure needs a caveat: too few clips, bins nobody has judged yet, or clips drawn before this behavior's bins were recorded.

**Finish** says you are done; it does not lock the run. Browse still works afterwards, and **Reopen**, beside Finish, is one click if you want to change a cutoff or audit more clips. While the run is finished, the cutoff and the audit's buttons are greyed, and hovering one says to reopen.

The same cutoffs can be set from the CLI: `analysis query-calibrate --run <an_NN> --curve q01` prints the curve as numbers and `--cutoff q01=0.82` commits one.

The CLI `analysis query` runs the same search: the command and the run records keep the `query` name. See [Behavior scan](modules/query.md) and the [CLI guide](cli.md#analysis).

### Behavior clustering

Builds one vocabulary of *behaviors* pooled across the recordings you select, with no labels and nothing to point at ([how it works](modules/temporal_clustering.md)). A cluster is a stretch of movement and an instance is one performance of it, with a real start and end you can play. **New analysis… → Behavior clustering**, check the recordings, pick the **embedding run** and **track**, and **Run**. **Method** offers two ways of doing it, and the form shows only the chosen one's knobs. **codebook_viterbi** (the default) clusters a sample of frames into a codebook and decodes the sequence; its knobs are **Share of each recording**, **Codewords** and **Minimum run**. **dtw_finch** cuts the trajectory into bouts and compares every pair; the detector, minimum bout length, DTW band and **Max bouts** are its, under *Advanced settings*. The two tie on vocabulary quality and the bout path draws slightly better boundaries, but comparing every pair of bouts grows with the square of the study, so keep **dtw_finch** for short studies: the codebook's expensive step is bounded by a number you set, not by how much data you have. **Share of each recording** is how densely every video is sampled to build the codebook, the same in a large study as a small one. **Codewords** is the cost knob: the decode costs roughly the square of it on every frame you pooled, so lower it, not raise it, when a fit is slow.

**Embedding run** lists each set of embedding settings found across the checked recordings, with how many have a run of it (`N/M recordings`); the widest is ticked. Settings include everything upstream of the embedding, so recordings detected by two different models (one per strain, say) show up as two rows even when the embedding was made identically. **Tick both** to pool every recording into one vocabulary: each reads its run of the setting it has. A recording with a run of two ticked settings is refused in the form, and the runs must share an embedding model and input image, or the launch says they are not comparable.

Opening the run gives the [interpretation workspace](#interpretation-workspace) described below (INTERPRET with its tree, filters and a montage of every clip; AUDIT; BROWSE), and a behavior's members are bouts. Two things worth knowing:

- **A capped fit says so, and still labels everything.** Comparing every pair of bouts costs time growing with the *square* of their number, so **Max bouts** samples down the bouts the comparison sees. Every remaining bout is then matched to the nearest cluster, so the output has one instance per bout however low you set it: the cap bounds what the vocabulary is *learned from*, not what it labels. A line at the top of INTERPRET says which happened, since a capped fit is identical to a full one everywhere else on the screen. What you give up is that an assigned bout had no say in what the clusters are, which is why the tree and the member strip show the clustered ones only.
- **Coverage is partial on purpose.** Bouts don't tile a recording, stretches shorter than the minimum bout length are dropped, and gaps in tracking are not bouts. The CLI's coverage lines give the numbers: how much of the video was analysed, and how much of that is in a bout. Less than all of it is the honest answer, not a fault.

The same choices can be made from the CLI: `analysis temporal-interpret --run <an_NN> --tree`. See the [CLI guide](cli.md#analysis).


### The interpretation workspace {#interpretation-workspace}

A clustering fit finds a whole tree of clusters, from one cluster holding everything down to the finest groups, and then prepares the evidence you will look at for every one of them, however small, so that clicking around afterwards never waits.

The fit stays **open** rather than finishing complete: a vocabulary is not a result until somebody has looked at it. The run's views are the **interpretation workspace**: **INTERPRET**, **AUDIT** and **BROWSE**, and **MEASURES** (the run's [measures](#measures) table), each opened from its card on the run's page and switched between with the buttons in the row above it:

- **INTERPRET** is where you explore what the fit found and label the behaviors you care about. The **filters** sit in a column on the left. To their right, top to bottom: the **tree** with its key and **Find behaviors** box above it, the **label bar**, and a **montage** of every clip of the cluster you picked, all playing. Nothing on the page waits on a click: picking a cluster or a filter is instant, and the montage plays from small previews of your videos kept on this computer.

    **The tree.** Every block is a cluster, and its width is how much time it holds. The top block is everything; each row down shows what a block divides into, largest first. Where a cluster only sheds a small piece at a time, the pieces it sheds sit side by side in one row rather than one per row. Blocks too narrow to read are drawn as thin columns of their own colour at the end of their row, with no name on them, so a filter shades even the smallest cluster without zooming; each is still clickable, and hovering over any block names it. A thin dark line separates every block from the next, so two pale clusters side by side still read as two. Click a block to open it; the open cluster is ringed in dark and white, which shows on any colour, and a thin one is ringed round the outside. Scroll the wheel over the tree (or pinch on a trackpad) to zoom in around the pointer, and drag to pan; **Shift**+wheel, or the scrollbar beside the tree, moves down the levels. A zoomed tree keeps its zoom when you widen the window, so a wider window shows more of the run. Once you zoom, a scrollbar appears above the tree: it is the whole run, and its handle is the part you are looking at. **Double-click a block to label it** (a small editor opens on the block; Enter saves). Right-click a block for **Label…**, **Remove label**, **Open in window** and **Zoom to this** (on a snapshot or a finished run the first two are greyed and say why); **Whole tree** (or Esc) shows everything again, and **+** / **−** zoom from the buttons, toward the selected block. The arrow keys walk the tree as drawn: **↑** to the block above, **↓** to the largest block below, **←** **→** along the row. A block is coloured by its label: solid where you put the label, pale where a label above it covers it, grey where nothing does. Colour on the tree means one thing at a time: as soon as a filter shades the tree or a difference screen is ticked (below), the colours are theirs, and a labelled block shows a plain strip along its top and its name instead of its label's colour until you turn them off. Blue and yellow are kept for less and more: no label is ever given either, here or in BROWSE, the label check or an exported figure, so there are twelve label colours before they repeat. A small dot marks the clusters you have already opened. The tree opens tall enough to show **every level**. When that would leave the montage less than two rows of clips, the rows get shorter instead (names and shares then sit on one small line), so the whole tree is still on screen without scrolling. Drag the bar under the tree to set its height yourself, as short as 120px: your height is kept for next time, rows stay full height, and the tree scrolls. Drag it up to give the montage more of the page. In a window too short for the tree's height, the tree gives way first, down to 120px, so the montage keeps its rows and its whole scrollbar on screen; your height comes back when the window has room for it.

    **Find behaviors.** The box in the row above the tree, beside the zoom buttons, finds clusters by their number (the one a window's title and the label bar give, as in "Behavior #1321") or by a label you gave them. Type one number or one label and press **Enter** or **Find**; a new search replaces the last. A label is matched whole, ignoring capitals, so `climb` finds every cluster you labelled "Climb" but not "climbing". What you found is outlined in one colour, on the outside of each block so even a cluster too thin to read shows up, and never over its name; the chip beside the box shows the colour, and a label on several clusters says how many (`climb · 3`). The tree zooms to the smallest cluster that holds them all, so they are all on screen at once; a single cluster zooms right onto it, and then has no outline, since it is the whole tree you are looking at. Label another cluster while its label is searched and it is outlined too, without the tree moving. Click the chip to open its clips (a label's largest cluster), or **Clear** to end the search: the tree goes back to the view you had before you started. A number the run does not have says "not in this run", and a label no cluster has says so. Finding changes nothing in the run, and a re-fit ends the search, because the numbers then name different clusters.

    **The row above the tree** names each label in use with its colour (a plain square while a filter or a screen has the tree's colours) and how much time it holds, then the Find behaviors box (more numbers than fit wrap onto a second line), with the zoom buttons at its end; hover it for the tree's mouse and keyboard controls.

    **The column left of the tree has two tabs.** **Filters**, the one it opens on, asks *which clusters are made of what*; **Differences** asks *which clusters differ between conditions* (below). The tab that is open decides what the tree's colour means, and each tab's name counts what is on in it (*Filters · 2*, *Differences · 1*), so you can see the other one is in use without opening it. On the **Filters** tab, every change shades the tree at once, while you drag too. Drag the grip beside it to make it wider or narrower, down to 240px, and the location map shrinks or grows with it; drag it narrower still, or double-click the grip, and the column folds to a strip like every [side panel](#the-window). The strip reads *Filters*, or *Filters · 2 active* while two filters are on, so a filter is never on out of sight; **Reset all** is inside the open column. When the page reads its fields again, a filter that no longer has a line in the column is removed (a field that is gone or down to one value, values or recordings that no longer exist, or a range on a field that is now tick boxes), and the count and shading follow. Width and folded are kept for next time. A filter never hides anything. Instead, **every block is shaded by the share of its clips that match**, on a fixed scale from 0% to 100% that runs from dark to light in a dark window and light to dark in a light one, with no colour of its own, and a labelled block shows a plain strip along its top in place of its label's colour. The scale never stretches to fit the clusters: if every cluster is 43–55% one treatment, the tree is one even mid shade, which is the truth. The scale is always at the top of the Filters tab, *Clips that match* over a bar from 0% to 100%, with one line under it. With no filter on, the scale is greyed and the line says how many clips the run has (*2,781 clips*). With a filter on, a tall bar marks the cluster you have selected, with its share above it (*selected 2.3%*), and a thinner line marks the whole run (*run 23%*), so you can see at once whether the selected cluster has more or less than its fair share; the line says how many of the run's clips match (*1,367 of 2,781 clips match*), with **Reset all** at its right, which removes every filter on this tab and leaves a screen ticked on the Differences tab as it is. The scale and the line take the same space either way, so nothing below them moves when you start filtering. Hover the line for how fields combine: *A clip matches when it fits every field. Several ticked values in one field: any of them.* Hover a block for its exact share.

    **Differences** is the column's other tab: the difference screens saved over this run (a [comparison](#comparisons) with *Every cluster* as its measure), each with what it kept (*drug vs vehicle · paired   1 kept of 4 clearing*), and **New screen…**, which opens the comparison form already set to this run and its clusters; the screen it saves appears in the list, ticked, with this tab open. Tick a screen and, while this tab is open, the tree is coloured by it (below), and the clips of clusters that do not differ draw grey in the montage. It narrows the clips together with the Filters tab, so *differs under drug* and *in the drug sessions* together show the drug clips of the clusters that changed. A filter never re-runs the screen: with *first ten minutes* on you see the clusters that differ over whole sessions and which of their clips fall in those minutes, not the clusters that differ within them. One screen at a time: two rankings cannot shade one tree. A screen that cannot be read is greyed and cannot be ticked, and so is one with no floor or nothing to contrast (a screen saved before the form refused such shapes): with no floor nothing it ranked is shown to differ, so there is nothing for the tree to show. Why is written in full under the list (*Can't be ticked: habituation vs fentanyl_25 · paired (no floor: only one animal_id (3) has both …)*). A ticked screen stays ticked while the page reads its screens again, after a rebuild or a new screen, and is removed only if it is no longer listed.

    **Strictness.** Under the ticked screen a slider reads it looser or stricter than its own floor. The saved screen keeps a cluster when fewer than 5% of shuffled cohorts produce a larger top effect; slide right and clusters that would need, say, 20% count as kept too, slide left and only the surest stay. The line under the slider says the level and what is kept there (*p ≤ 0.2 (looser than its floor) · 6 kept of 12 clearing*), the tree colours what clears at that level, and the floor ticks on the scale move with it. Hover any block for its numbers: *family-wise p 0.2* (the share of shuffled cohorts whose largest effect reaches this cluster, the floor read as a number), *own p 0.03* (the same share against this cluster's own shuffles alone, so a cluster consistent in your animals but hidden by how many clusters were screened still says so) and *q 0.6* (the false-discovery rate over the family, the usual answer for many tests at once; it needs more animals than a family-wise floor does, so expect it to read high in a small cohort). This is for finding clusters worth a larger experiment: nothing in the saved screen changes, and a label you write at a loosened level says so (below). Because a screen ranks on the moderated t by default (*Rank on*, under *Difference ranking* when you launch it), loosening admits the clusters your animals agree on most, not the ones with the biggest numbers.

    **A label you write while a screen is ticked remembers the screen.** The screen chose the cluster you watched; the name is yours. If you later compare that label across the same column, the comparison says the screen chose the cluster it is scoring, so the effect does not read as an independent finding. Nothing is promoted or written for you, and a label you write with no screen ticked, or on a cluster the screen did not keep, records nothing. The label also remembers the strictness the screen was read at and the cluster's numbers: the label bar reads *· kept · family-wise p 0.02, own p 0.004, q 0.5* for a cluster the screen kept at its floor, and *· kept at p ≤ 0.2 · family-wise p 0.2, …* for one you labelled with the slider loosened, so a lead stays marked as a lead for as long as the label exists; the screen's page lists it *at p ≤ 0.2*, and a later comparison of the label says *kept at p ≤ 0.2* in its note.

    **The screen's scale** sits under the list while a screen is ticked: *Less ↔ more under drug (against vehicle)* over a bar that runs from **less** (blue) through **no difference** (grey) to **more** (yellow). The two ends differ in lightness as well as colour, so they stay apart for a colour-blind reader. Two ticks mark the floor either side of the middle, and a tall bar marks the cluster you have selected, reading its t with the effect after it (*selected t +8.8 (12.4×)*) on a screen ranked on the moderated t (the default), or the effect alone (*selected 12.4×*) on one ranked on the effect. Hover the bar for its units: *t −10 to t +10*, *÷20 to ×20*, or *ρ −1 to ρ +1* for a trend. A profile screen (three or more named values, such as sessions) reads *Differs across session* from **same** to **differs** with one floor tick and no colour, grey to white in a dark window: its effect is a size, because the direction belongs to each cluster's own pair of sessions, so the tall bar names the selected cluster's pair and direction (*selected t 6.2 (9.6× more under SA than REI)*) and the hover names every other's.

    **What the tree shows depends on the open tab.** With **Differences** open, the tree is coloured by the ticked screen, whatever is filtered: a cluster that cleared the floor takes the colour of its side, more or less, and everything inside the floor stays grey, so what you see coloured is exactly what the screen found. With no screen ticked, nothing is shaded. Hover a block for the screen's words on it, *more under drug than vehicle, 12.4× (7 of 10 ID in that direction) · kept · t +8.8 · family-wise p 0.02, own p 0.004, q 0.5*. With **Filters** open, the tree is shaded by the share of each cluster's clips that match the filters, and the clusters the ticked screen kept are **outlined** in the colour of their side, blue for less and yellow for more, with the screen's words added to their hover. The tree never mixes the two, and its colours only change meaning when you change tab.
    - **Location**, the first section, is a map of where the animal was over the whole run. Its line folds like a field's: click **▸ Location** to hide or show the map, and the line says what is picked, *any* or *1 region chosen* (squares touching side by side count as one region); folded or open is kept for next time. Drag to draw a box; it is drawn while you drag and stays afterwards with a handle on each corner. Drag a corner to resize the box, drag inside it to move it, or drag outside it to start a new one. A click without a drag adds or removes one square; hover the map for a reminder of all this. From the keyboard, Tab to the map, move the outlined square with the arrow keys, and press Space to add or remove it. While squares are picked, **Reset** at the right of the *Location* line removes them. A run that recorded no positions greys the map and says why.
    - **Time in session**, under Location, narrows to a stretch of each recording: click **▸ Time in session** to show a slider from **start** to **end**, and drag either handle (or Tab to a handle and use the arrow keys; Shift moves ten steps). Start is each recording's first frame and end its last, so *25%–50%* is the second quarter of every recording, however long each one is. The line under the slider says the range and how many clips fall in it (*12%–40% of each recording · 840 clips*), or *Whole session*; the folded line says *any* or the range, and **Reset** on it puts the whole session back. A clip on the edge of the range counts by its time: a cluster's shading counts only the part of its time inside the range, and a clip in the montage gets the match frame when its middle is inside, and a clip whose middle sits exactly where two ranges meet belongs to the later one, so *0%–50%* and *50%–100%* never both hold it. If a run clustered only part of each video, part of the range simply has no clips. Folded or open is kept for next time.
    - **Then one line per field**: **Video** (the run's recordings, searchable), then each column of your recording metadata. Each line reads *▸ name*, with how many values it has at the right, or the filter you set on it (*treatment_low only*, *2 of 3*). Click anywhere on a line to open its values under it, and again to close it; from the keyboard, Tab to a line and press Space or Enter. Lines open closed each time you open the page. While a field has a filter on, a small **Reset** at the right end of its line removes that filter. Fields with a filter on open and move to the top under *In use*, above a divider, with the rest below in their usual order; a field you are setting moves up once the pointer leaves it, so it never jumps between two clicks. With more than 8 fields, **Find a field** above the list narrows the lines to those whose name contains what you type.
    - A field's values depend on what the column holds. A column of clock times (*14:00:21*, for example a recording's start time of day) is read as a time of day and sorted by time. Categories, and numbers, dates or times with at most 12 different values, get a checklist: how many clips each value has, a **(no value)** entry, and a search box when there are more than 20 values. Untick what you do not want, or hover a value and click **only** to keep just that one (from the keyboard, **only** sits on the current row; Tab reaches it). Numbers, dates or times with more values get a range: bars of how many clips have each value, with a handle at each end of the range to drag along them; the range is drawn in colour and its from–to values are written under the bars.
    - A column that has the same value for every clip has nothing to choose, so it is a greyed line at the bottom of the list, for example *sex  F*; hover it for why. A column with one value and some blanks keeps its checklist, because **(no value)** is a real choice.

    **The montage**, under the tree, plays **every clip** of the cluster you opened, each cropped to its animal and on its own loop. The order button (**Most typical ▾**, at the right end of the label bar's first row) picks **Most typical** (the default), **Least typical** (where clips that do not belong collect) or **Random**; a cluster's random order is the same every time you open it. **Size** sets how large the tiles are: larger tiles mean fewer to a row, and the montage itself keeps its size. Only the tiles on screen play, so a cluster of two thousand clips scrolls as smoothly as one of twenty, and the clips you stop on are drawn first: a fast scroll does not leave the ones you passed to be drawn before them. **Hovering a tile pauses every other one** and shows a thin line under its picture that fills from the left as the clip plays; its tooltip gives its rank, recording, start and length, and, while a filter is on, each filter it meets or fails (*✓ animal_id A03  ✗ location*). **A clip that does not match the filters still plays, with its animal drawn grey.** A clip you removed on AUDIT stays in its cluster here, because INTERPRET shows what the clustering found: it is **faded**, with *not climb* along its bottom and a bold ✕ in its top corner (a moved clip shows *→ climb › left* and no ✕), and the label bar counts them (*48 clips · 6 removed*). Right-click it for **Put back in climb**. While a filter is on, every clip that matches has a frame in the match colour round its picture (with masks on or off), and a key at the right of the label bar's first row (beside the order button, **Show ▾**, **Size** and **Export ▾**) names that colour: *■ matches filters*. The montage has no controls row of its own, so its clips start right under the label bar. The clips are centred in the montage's width. **Every clip grid has the same toolbar**: **Most typical ▾ · Show ▾ · Size · Export ▾**, here at the right of the label bar's first row, over the clips in a cluster window, and over AUDIT's review grid (where the order is fixed, so there is no order button). **Show ▾** says what every tile draws: tick or untick **Video**, **Masks**, **Mask trail** and **Keypoints** (greyed, with the reason, when the embedding read no keypoints). With **Video** off the masks and keypoints are drawn on white; with **Masks** off the plain video plays, and a clip that does not match the filters is washed out instead of drawn grey. The last thing a tile still draws cannot be unticked. Under them, **As recorded** / **Aligned** turns every tile so the animal's body points up, frame by frame, from the keypoints the embedding read and a heading definition; on a run without them **Aligned** is greyed and says why, and a frame with no measurable axis plays as recorded with a small *no axis* mark. Last, **No context**, **1 s context**, **2 s context** or **5 s context**: that much video either side of each clip, so each tile plays the clip with its lead-in and follow-on. Each screen keeps what you choose.

    **Export ▾** writes the clips **as the grid shows them**: **Video of 182 clips…** and **GIF of 182 clips…** (every clip side by side, in the grid's order; the GIF asks for its width and frame rate first), **One file per clip…** (a GIF of each into a folder, numbered in order) and **CSV list of clips…** (a clip-query CSV that `detail analysis query --from-csv` reads as example clips, with each clip's rank, recording name, subject and own times; context is not part of a clip, so the list leaves it out). A greyed line at the bottom of the menu says what they draw: *Writes what the grid shows: video + masks, as recorded, no context*, following **Show ▾**. Each asks where to save and then writes in the background from the full video, not the previews, with a progress bar and **Cancel**; a line over the clips says what was written. Saving over a file that is already there asks first, whether the name was picked in the save dialog or **One file per clip…** finds its GIF names in the folder (it says how many). **Cancel** never costs an earlier copy: a file is replaced only once the new one is finished. The same exports are `detail analysis cluster-montage` from the command line.

    **Pick clips to export…**, at the top of **Export ▾**, chooses which clips and in what order. A yellow bar appears over the clips: *Pick mode · Click clips to pick them, in order · 3 picked*, with **Pick all**, **Clear**, **Export 3 ▾** and **Quit**. Nothing is picked to start. **Click a clip to pick it**: it shows its number, 1, 2, 3 in the order you click; click it again to un-pick it. Drag across clips, or click one and **Shift**-click another, to pick a run, in the grid's order; a drag that starts on a picked clip un-picks the run. **Pick all** picks every clip in the grid's order, **Clear** un-picks them all. **Export 3 ▾** (and **Export ▾** while you pick) writes the picked clips in pick order, the same four ways; with nothing picked it is greyed and says *Pick at least one clip.* **When the export finishes, pick mode closes**: the picks are cleared and a click goes back to what it did before (nothing here; on AUDIT, removing the clip). If you cancel the export, or it fails, you stay in pick mode with your picks as they were, to try again. Opening another cluster while picking ends pick mode and clears the picks, and the line over the clips says so; changing the order keeps them. **Quit**, or **Esc**, leaves pick mode without exporting. Opening another cluster also leaves it: the picks are cleared and a line over the clips says so (a new order keeps them). An export that finishes after you have quit and started picking again leaves your new picks alone. Picking changes nothing in the run: not the cluster, not AUDIT. Double-click a tile to open the clip, whole frame and from the full video, in the clip window, which steps through the rest in the montage's order with **◀ previous** / **next ▶**.

    **Open in window**, at the end of the label bar's last line (or **Open in window** on a cluster's right-click menu in the tree), opens the cluster in a **window of its own** beside the page, which stays fully usable. Open as many clusters as you like, each in its own window; opening a cluster that already has one brings that window forward. The windows close with the run's page.

    - **Each window has a number.** The first window you open is ①, the next ②, and so on (a closed window's number is the next one given out). The number leads the window's title (*① Behavior #1321 · climb · an_32*) and is drawn as a small dark disc in the corner of that cluster's block in the page's tree, so you can see which window shows which cluster. INTERPRET's own selection does not move when you click into a window or label from one; the page keeps showing the cluster you chose there.
    - **The window's top is the page's label bar**, for the window's cluster, laid out the same way. The first line is the window's number and the cluster (*① Behavior #1321*), its label chip, and at the right the montage toolbar (**Most typical ▾ · Show ▾ · Size · Export ▾**). The second gives the cluster's clips, time and share of the run at the left (hover for its separation and where it is concentrated) and the page's status at the right (*Labelled #1321 “climb” · Output up to date.*). The third is the label field with **Label**, **Remove**, **Labels ▾**, whose list opens with your recent labels, numbered, and **Undo (⌘Z)**. There is no **Open in window** here: this is the window. The clips start right under the bar.
    - **Labelling in the window** labels *that window's* cluster, and the page's tree, label bar and outlines update at once, as if you had labelled it on the page; the window's title follows the label. Type a name and press **Enter** to label; Enter anywhere else in the window does nothing (it never presses a button). Keys **1–9** in the window apply that recent label to the window's cluster. **Rename** and **Merge** from **Labels ▾** ask over the window, not over the page. On a snapshot or a finished run the label field and **Undo** are greyed and say why.
    - **The clips and their toolbar.** **Show ▾** and **Export ▾**, pick mode included, work exactly as on the page; in pick mode **Esc** leaves pick mode wherever you are in the window, and closes the window only when you are not picking. **Esc**, **Close** and the title bar's close all close it the same way: its number leaves the tree and its clips are let go. The window remembers what it shows, and its size and place. A minimised window, or one hidden behind others, stops playing until you can see it again.
    - Double-click a clip in the window to open it in the clip window.
    - Clips you removed or moved on AUDIT are faded with their tag here too.
    - **Re-fitting the run** renumbers its clusters, so every window closes and the page says so (*Closed 2 cluster windows: the run was re-fitted.*). Rebuilding the clips from the same fit, or the page re-reading the run, leaves the windows open, each where you had scrolled it. When a recording's masks finish building, the outlines appear in place, on the page and in every window: nothing scrolls back to the top and no clip restarts.

    **Undo.** ⌘Z on INTERPRET or in any of its windows undoes the last label change made in any of them: a label, a removed label, a rename, a merge, or a clip put back. Press it again to step further back; ⇧⌘Z redoes. **Undo (⌘Z)** on the label bar's last line, on the page and in each window, does the same, and its tooltip says what it will undo (*Undo: Labelled #1321 “climb”*). The undo is written like any label, in the background, and puts back the run's labels exactly, including a merged label's own identity, so its AUDIT answers and the clips removed from it or moved into it follow it back. It is greyed on a snapshot or a finished run, says *Nothing to undo* before your first change, and starts afresh when the run is re-fitted. ⌘Z while you are typing in the label field undoes your typing first.

    The label bar is three lines. The first is the cluster's name (*Behavior #41*) and its label as a large coloured chip (*not labelled* in grey when it has none, *inside “climb”* in an outline of that label's colour when a branch above it is labelled), with the montage's toolbar at its right end. The short line under it says what you need to decide whether the cluster deserves a label, how many clips, how much time and what share of the run it holds (*182 clips · 5.7 min · 9.6% of the run*), and at its right what your last label change did and whether it is saved (*Labelled #41 “climb” · Output up to date.*). In a window too narrow for both they end in *…*; hover the figures for the whole line, and for how different the cluster's two halves are against the run's usual split (*Separation 1.24 (run median 2.17).*). When the cluster is over-represented in one recording or metadata value by more than a random draw of the same size from your recordings would give, hovering the figures says so with the numbers (*Concentrated: animal_id A03 · 3.1× the run*), so a small cluster does not raise false alarms. The last line holds the label field and its buttons, **Undo (⌘Z)** and **Open in window**. Once AUDIT has reviewed every clip of the cluster, a line under the figures gives the figure (below). A cluster labelled while a difference screen was ticked gets a line under the figures too: *From screen cmp_01 (drug vs vehicle · paired): more under drug than vehicle, 12.4× · kept*, or *· inside kept cluster #1375* when you labelled a cluster inside the one the screen kept. It goes when the label goes, and undo and redo carry it with the label.

    The clips are exported from **Export ▾** on the montage's toolbar (above), not from **Export…** at the top right of the window, which writes the run's tables and has no section for the montage.

    **When previews are missing**, for example on a computer that has not built them yet, or after clearing the preview cache, the montage area alone shows a progress bar for this run's recordings, with the time left and **Cancel**; the tree, the labels and the filters work meanwhile, and the clips start playing when the previews are built. Building them moves this run's recordings to the front of the queue in the status bar. **Cancel** stops waiting for this run's recordings only; **Build previews** starts again. Where a recording's video is built but its animal outlines are not yet, the footage plays at once and the outlines appear when they are built.

    **The label bar**, between the tree and the clips, labels the cluster you opened: type a name in its large field and press **Enter** (or **Label**); **Remove** (or **U**) removes the cluster's own label, and **L** jumps to the field. As you type it suggests labels you have already used; choosing a suggestion (a click, or the arrow keys and **Enter**) only puts it in the field, and a second **Enter** labels. Inside a labelled branch it says so, and a label typed there becomes a sub-label. A click in the field just puts the cursor there. **Labels ▾ lists every label** with its colour, its time and its share: the most specific label wins, so a branch labelled "not moving" with a part labelled "resting" lists "not moving", then "resting" and **other** (the rest of "not moving") beneath it. **Click a label's name to put it in the field**, then press **Enter** (or **Label**) to apply it; nothing is labelled until you do. Hover a label for the rest: **go to** it, **rename** it (every cluster under it) or **merge** it into another label with the same parent label. A merge carries the clips you removed from the label or moved into it on AUDIT: they stay removed, or moved into the label it went into. **Labels ▾** opens with your **recent labels**, under *Recent (keys 1–9)*, each with its colour and a number. Choosing one there fills the field like any other label; pressing **1**–**9** (with no ⌘, Ctrl, Alt or Shift) while you are not typing in a field labels the open cluster with that label at once, the same as typing its name and pressing **Enter**. They lead with the labels you used last and then every other label by its time, up to nine; one the cluster may not join (a label's clusters share one parent label) is greyed in the list, and hovering it says why. The run's first label is typed; the recent labels appear once it has one. After each change the bar says what it did, at the right of the line under the cluster's name (*Labelled #41 “climb”*), then *Saving…* and *Output up to date.*

    **A label is one thing, however many clusters it has.** Adding a cluster to "rearing" makes it part of the same label, not a second label that happens to share a name: the two count as one behavior everywhere (their bouts join when they follow each other), and renaming the label renames all of them. Typing a name that already exists does the same and says so: the button reads **Add to “rearing”**. A label's clusters must sit under the same parent label, so a label that lives inside "climbing" is listed for a cluster outside it greyed, with the reason on hover. Names are unique, ignoring capitals. A label covers everything inside the cluster unless a cluster inside it has its own label, so you can label coarsely first and refine later, or the other way round. A label may not contain `/`, which separates the levels in the export. The tree and the label list update the moment you press **Enter**; the run's output catches up in the background, and a line beside the field says when it has. To see where your labels fall in each recording, and where the cluster you opened falls, use **BROWSE**.

    **Every cluster, every clip.** The evidence is prepared for every cluster, however small: its counts and shares, and the list of its clips with how typical each one is. A clip is one unbroken stretch at the finest level of the tree, inside the cluster, at least as long as the run's shortest instance. Stretches next to each other are never joined, so the whole run's block lists every clip, and a bout that crosses two smaller clusters plays as two clips. **Rebuild clips** prepares it again from the fit, with progress; it never refits. It appears at the top of the tab, beside the warning, when the tab says the clips are out of date, including on a run prepared before every cluster listed its clips: such a run opens as it was, and its clips have no typicality order until it is rebuilt. A run fitted before this screen existed opens to a single **Build the node cache** button. A build keeps going if you leave the page: come back and the tab shows its progress from where it is, and it never starts a second one for the same run.

- **AUDIT** is where you clean your labels and measure them. Clustering is never perfect: a cluster you labelled usually holds some clips that are something else. AUDIT has you watch **every clip** of each cluster, remove the ones that are wrong, and finish; the cluster's precision is then the exact share of its clips that really were the label, not an estimate from a sample. Whether a figure is good enough is yours to decide; AUDIT gives no verdicts.
    - **The list**, on the left, holds each label with its clusters under it. It opens wide enough to show every row whole; drag the grip between the list and the clips to widen or narrow it (down to 180 pixels), or past that to fold the list to a strip reading *Labels*, like every [side panel](#the-window); the width you leave, and whether it is folded, are kept the next time you open AUDIT. A row too long for the width you chose ends in *…*, and hovering it gives the whole of it. A label shows its colour, its name in bold and, once **every** cluster under it is reviewed (its sub-labels' clusters included), its figure on the right (*91%*: the clusters combined, each weighted by its number of clips); under that, *62 clips · 1 of 1 clusters reviewed*. A cluster shows *cluster #871* with its number of clips on the right; under that, a thin bar (how far down its clips you have scrolled) and *reviewing · 2 removed* or *not reviewed*. A reviewed cluster says **✓ Reviewed** in bold and *100% · 0 removed*, with no bar. Click a label (or move to it and press **Enter** or **Space**) to open its first cluster nobody has reviewed; click a cluster to open it. The **arrow keys** move through every row, labels included: a cluster opens as you reach it and the keyboard stays in the list, while passing a label opens nothing. Hovering a label says how many of its clips are removed, counting the same clusters as its *N clips* line: its own and its sub-labels'. **Right-click a cluster** to **Open in INTERPRET**. A run audited before clip review existed keeps that figure, never mixed into the new one: hovering the label gives it as *Earlier one-by-one audit: precision 0.88 (95% CI 0.70-0.96, n=25)*.
    - **The top row** names the cluster (*cluster #871*, its label), shows a thin bar with *reached 40 of 62* (how many clips have been on screen), and a large **✓ Finish review** button with *60 climb · 2 not climb* under it, counts that follow every click. Finish stays greyed (*Scroll to the last clip first*) until the last row of clips has been on screen, and while any clip shows *no preview* (*3 clips show no preview…*), since a clip nobody saw cannot be counted: build the previews, then finish. Then it turns blue, and **Enter** presses it. While **Pick clips to export…** is up, Finish is greyed (*Quit picking clips first*), so Enter does not finish the review. On a run whose clusters are ranked when first opened, the line under the top row says *ranking this cluster's clips…* until they arrive, as INTERPRET does; nothing can be removed or finished until then.
    - **Reviewing.** The clips play below in one scroll, **least typical first**, so the clips most likely to be wrong come first. **Click a clip to remove it** from the label: it fades where it is, with a bold **✕** in its top corner and *not climb* along its bottom, so nothing jumps. Click a removed clip again to put it back. **To remove a run, drag**: press on a clip and move to another, and every clip between them, in reading order, is tinted orange; let go to remove them all, as one step you can undo. A drag that starts on a removed clip puts the run back instead, tinted blue. Clips already in that state are skipped. **Esc** cancels a drag, and dragging near the top or bottom edge scrolls. A click that wobbles a little is still a click. **Or hold Shift**: the clips from the last one you clicked (ringed) to the one under the pointer are tinted, and **Shift-click** removes them. While a run is tinted, the line under the top row says in bold what letting go will do: *Let go to remove 4 clips · Esc cancels*, *Let go to put back 3 clips*, *Shift-click removes 7 clips*. **Right-click** a clip to **Move to** one of your other labels (it fades with *→ climb › left* and no ✕) or, on a removed or moved clip, to **Put back in climb**. **Double-click** a clip, or press **Space** over it, to open it whole frame and from the full video in the clip window. A double-click's first press is a click, so the double-click takes that removal back; Space right after a click opens the clip and keeps the removal. Hovering a clip shows a pointing hand, a faint ✕, and *Click: remove from climb · double-click: open*. Over the grid sits the montage toolbar every clip grid has, without the order button: **Show ▾** chooses what every clip draws, which way it faces and how much context it plays, as on INTERPRET (kept for AUDIT; greyed with the reason on a run whose embedding read no keypoints), **Size**, and **Export ▾**, which writes the cluster's clips as the grid shows them, in the order shown. **Pick clips to export…** works as on INTERPRET, and while its yellow bar is up a click **picks a clip for the export and never removes it**; no ✕ appears on hover. **Quit** or **Esc**, or an export of the picks that finishes, gives the click back to removing. You can pick and export from a reviewed cluster too.
    - Every change is saved as you make it, so leaving loses nothing: open another cluster and come back, and the clips you removed are still faded, before and after they reach the disk. A clip removed under another label, before the cluster was relabelled, does not count as removed here, the same rule Finish uses. Removing clips changes the run's output: the exported instances, the measures and BROWSE no longer cover those clips.
    - **Undo.** Under the top row a line says *Click a clip to remove it. Drag, or hold Shift, to remove a run.* until you change something, then what your last change did, *Removed 3 from climb*, *Put back 1 in climb* or *Moved 1 to climb › left*, with **Undo (⌘Z)**. Press ⌘Z again to step further back, as far as the start of this review. Finishing, or opening another cluster, starts the history afresh.
    - **Finishing** records every clip of the cluster, a removed or moved one as *not* the label and every other one as the label. The top row becomes a framed box: **✓ Reviewed**, the cluster's number, label and the date, the precision in large type, how many clips are the label and how many you removed, **Next cluster** (the next one nobody has reviewed) and **Reopen review**. The clips split in two: **Removed** first, then **Kept as** the label, each with its count. A reviewed cluster is read-only: clicking a clip changes nothing, and double-clicking still opens it. A finished cluster opened later shows the same; **Reopen review** puts the clips back in review order from the top so you can change what is removed, keeping the removals you made, and finishing again replaces the earlier figure. Until you do, the earlier figure is shown as out of date: *was* the earlier figure *· reviewing again* beside the cluster's name, and *was 97% · 2 removed* with a progress bar on its row in the list, even if you leave and come back. While a finish is being saved (*Saving the review…*) clicking a clip changes nothing. After you change which clusters a label holds, its figures start again; removing clips does not reset them. **Putting a clip back withdraws what the review said about it**: a reviewed cluster with 2 clips put back reads *6 yes · 2 remaining* rather than all reviewed, every other clip keeps its answer, and finishing again reviews the clips put back. This holds wherever the clip goes back, on AUDIT or with INTERPRET's **Put back in…**, and for an undo of a removal.
    - Moving to **AUDIT** from a labelled cluster on INTERPRET opens that cluster's label. Only labelled clips are reviewed: **unlabelled** time makes no claim to check.

    **The audit on INTERPRET.** A reviewed cluster gets a ring inside its block on the tree, in one colour: it says the cluster has been reviewed, not how well it did. Opening it puts an **AUDIT** line in the label bar, under the cluster's name: *precision 0.93 · all 60 clips reviewed*, and an **Act on it ▾** menu with **Move to…** (another label; one the cluster may not join is greyed, and hovering it says why), **Remove from the label**, and **Look at its halves** (zooms onto the cluster and opens its first half), for when you decide the figure calls for one. The rings follow your reviews and your label edits on INTERPRET.
- **BROWSE** is the dataset your labels make: every labelled instance, exactly as the run writes it, and it follows every label you add. Unlabelled time is left out.
    - **Filters**, on the left, narrow it: **Labels** (nested like the labels panel, with each label's count and time; unticking *rearing* leaves *supported* showing, because they are separate rows), **Recordings** (with a search box), and **Metadata** (pick a column, then tick its values). Each list is only as tall as its rows. The line at the top says how many instances and how much time pass the filters, and **Clear filters** puts everything back.
    - **The ethogram** shows the labelled time that passes, one strip per recording, coloured as on INTERPRET. Click a bout to select its row and play it. **The cluster you opened on INTERPRET is a thin lane under each strip**, labelled or not, so you can see where it falls and which label comes before and after it; click a mark in the lane to play that stretch. The lane is not drawn for the top block, which is all of the time. It is worked out when you open BROWSE, not on every click on INTERPRET, so clicking through clusters there stays quick. Up to six recordings, every strip shows, lane included; past six it scrolls, and a handle under it lets you show more strips at once. Every strip shares one time axis, starting at 0:00, so the same minute sits at the same place in every row. The wheel zooms around the pointer, drag pans, **Whole** or **Esc** zooms back out.
    - **The list** gives each instance's label, recording, start, duration and what the audit said about it (a subject column appears when a run has more than one). **Any column sorts**, and **selecting a row plays it** in the player beside it and outlines its bout on the ethogram; ↑ / ↓ steps through them. The list opens as wide as its columns, leaving the player a usable width; when that is not enough, the label column gives way, ending in *…*, and hovering a label gives it whole. Drag the grip to choose for yourself, or past the list's narrowest to fold it to a strip reading *Instances*. The list holds the first 2,000 instances that pass and says so; the ethogram always shows them all.
    - **Export selected** writes the rows you selected (Shift or Ctrl to select several), and **Export all** writes every instance that passes the filters, not only the ones the list has room for.
    - The player opens on the first instance, paused. Before anything is labelled the tab shows one line saying so; when the filters leave nothing, the ethogram, list and player give way to a line and a **Clear filters** button.

    This is where a systematic error shows up: one label whose every member is the same wall-lean is invisible on a tree, plausible in a list, and obvious the moment you watch three of them.

### Snapshots

On the run's landing page, above the view cards, **Snapshots** lists the frozen copies of this run's results: one per time a
number left it, whether by an export, a comparison, or pressing Finish. Each row carries the whole
story: when it was taken and why, how many labels it had, how many
instances it held, **the names themselves**, and **which comparison used it**. The names are there
because that is how you will actually recognise one: not "the third snapshot" but "the one with
rearing and grooming that the dose comparison read". Editing the run never changes a snapshot,
which is what lets you reopen a finished analysis without invalidating a number somebody has
already quoted. A run that has not sent anything anywhere shows no list at all.

**Double-click a snapshot to read it.** The page stays where it is and becomes that snapshot: a
banner on the landing page names which one you are on, and the view cards open the interpretation *as it was*,
the same tree, montage, labels and instance list, with the labels that were pinned. Everything
that would change it is greyed: labelling, rebuilding clips, Finish and Reopen, every AUDIT edit
(removing a clip by clicking it, **Undo**, **✓ Finish review**, **Reopen review**, and INTERPRET's **Put back in…**; every cluster's clips still play and can be selected, since watching changes nothing) and the audit line's **Move to…** and **Remove from the label**, and the run's own **Delete analysis**,
**Cancel analysis**, **Start new analysis** and **Open output folder**. **Measures** is not shown at all, because it reports on the live run. Everything that changes what you are *looking at* still works, open
any cluster, zoom the tree, play a clip. **Back to live run** returns you, and
double-clicking another row switches straight to it. **Behavior scan** runs work the same way: a
snapshot there is the calibration as it was, so the cutoffs are frozen and the row says how many
behaviors were calibrated and to what.

**Finish**, at the bottom of INTERPRET, says you are done; it does not lock the run. While a run is finished, labelling and the audit review are greyed, and hovering one says to reopen. Labels, removals and finished reviews (here and on AUDIT) are saved in the background, one at a time and in order. Leaving the page while they are still saving loses nothing: they finish behind you, and a page you open on the same run meanwhile reads it again once they have landed. The keypoint marks clicked in **Mark frames…** save the same way, so closing that window while they save loses nothing either. Quitting the app is the one exit that would lose them, so it asks first (*Changes still being saved*): **Wait, then quit** closes the window by itself once they are saved, **Quit now** quits without them, **Cancel** stays. Pressing **Finish** while labels are still saving waits for them to be saved and then finishes; from the press until the run is finished nothing can be labelled, on the page, in a cluster window, by a number key or by **Undo**, and hovering says why. **Rebuild clips** still works, because it changes what the page can show, not what the run says. **Reopen**, beside Finish, brings them back; the copy Finish pinned stays as it was. The same choices can be made from the CLI: `analysis temporal-interpret --run <an_NN> --tree` prints the clusters, coarsest first, with their time and labels, `--name-node <node_id>=<label>` labels one, and `--rebuild-cache` rebuilds the clips from the fit. *Not built yet:* applying a fitted vocabulary to a new recording, and maps in arena rather than camera coordinates. See the [CLI guide](cli.md#analysis).


### Measures

They appear on any analysis run that produced a behavior result: as the **Measures** view of a clustering or behavior-scan run, and on the run's page under its view cards for any other kind (not while you are reading a snapshot: they describe the live run). One row per recording, subject, behavior and measure: how many bouts, how much total time, a rate against the time the subject was actually observed, mean bout duration. Each row carries the unit its value is in, because the column mixes counts, durations, ratios and rates. The table fills its view, and the spare width is shared out across every column. Write the same table to CSV with `measures export` (see the [CLI guide](cli.md#measures)).

## Comparisons {#comparisons}

Does this differ between these groups? Pick a result to compare (a behavior result, or a location export for movement and body shape), say how the recordings split, and read the groups side by side. **New comparison** opens the launch form.

- **Result to compare**: the analysis run the comparison measures: one that produced a behavior result, or a finished location export. Only those are listed, labelled with the module that made them, how many recordings they cover, and when.
  {: #comparison-source }
- **Measure**: the run's **labelled behaviors**, or **every cluster** of a clustering run, named or not. The second is the screen for clusters nobody has named yet: each node of the tree with at least the candidate floor of member time (30 s, under *Difference ranking*) is ranked by how its share of time differs, with its rate (bouts per second of observed time) and mean bout length beside it, nested slices of one behavior collapsed to the best of them, and a one-line reason on every kept node. It is offered only for a clustering run with a current node cache. With it chosen, the four measures show ticked and fixed and **Read snapshot** is greyed: every cluster is measured by all four, from the node cache as it is now. Choose the labelled behaviors again and the measures you had ticked come back.
  {: #comparison-measure }
- **Movement and body shape**, the third choice under **Measure**, is offered for a location export: distance travelled, time moving, speed when moving, and the body's length, width, length-to-width ratio and area, one value for each recording and animal. A location export offers nothing else, and a behavior result does not offer it; the choice a run cannot answer is greyed, with the reason. Each of these measures is tested on its own, and the result lists them by that test's p, smallest first, with the false-discovery rate over the measures beside it. They are a handful of measures in different units, so they do not share one floor the way clusters do. The location export must have measured the body, on calibrated recordings; one made with thin parts removed, an option the export no longer has, is refused. Under **Advanced Settings**, **Locations** holds how the run is read: **Rows per second read** (5; **every row** reads them all), **Path smoothing** (1 s, so jitter is not counted as distance), **Moving above** (2 cm/s) and **Jump above** (7.5 body lengths a second: a faster step is a tracking jump and adds no distance).
  {: #comparison-locations }
- **Compare across**, which column of your recording information splits the recordings into groups, plus **subject** and **window_id**. Each column shows the values it holds and how many recordings carry each, counted over the recordings the chosen result covers, so you can see which ones split your cohort usefully before picking. Pick a result that covers only some recordings and the columns are counted again over those, with a line under the choices saying how many of the experiment's recordings it covers. A column holding only one value cannot define groups and is greyed out saying so. Choosing `window_id` compares early against late in the same way choosing `treatment` compares drug against vehicle.
  {: #comparison-groups }
- **One observation is**: what a single point in the result means. When one animal contributed several recordings, averaging each animal first and counting every recording separately give different answers, so the form asks, naming the column it infers identifies your animals, and pre-selecting averaging within it. Change the column if it guessed wrong. When no animal contributed more than one recording there is nothing to decide, and the form says so instead of asking.
  {: #one-observation }
- **Contrast these two**: with one column chosen, its values, to tick two of. Leave it alone and a column with two values contrasts them, a numeric column with more becomes a trend across them, and a column with more non-numeric values runs as a profile over all of them, each behavior reported by the two values it differs most between. Ticking two on a numeric column contrasts those two doses instead of the trend. A recording with no value in the column (blank, or *NA*) is not a group of its own: it is left out, and the summary under the choices says which.
  {: #comparison-levels }
- **The design line** under the choices says what will run before you run it: *between* animals when the column is the same for all of an animal's recordings (sex), *paired* within each animal when it changes between them (vehicle against drug), a *trend* within each animal over a numeric column (dose), a *profile* within each animal over three or more named values (session), each behavior scored by the pair of values it differs most between. You choose the columns; the design follows, and the floor each one ships is the one its design allows, so a pairing is never shuffled away. Some cohorts cannot have a floor at all, and the form can tell from the recordings alone: one animal holding both sessions of a paired contrast, one animal in every group, a single group. With **every cluster** chosen, Run is greyed and the line under the buttons says why and what would work (add a second animal with both sessions, or measure the labelled behaviors instead), since a screen without a floor keeps nothing. With the labelled behaviors, the summary warns that no floor will be computed and the comparison runs, reported as measured.
  {: #comparison-design }
- **Windows** (advanced): measure the whole recording, fixed tiles, equal fractions of each recording, or a sliding window for a time-resolved result. `window_id` then becomes a column you can compare across, which is how "early versus late" becomes the same operation as "drug versus vehicle".
  {: #comparison-windows }
- **Recordings** (advanced), which recordings take part. Left empty, every recording the chosen result covers is used. A recording it does not cover is greyed, marked *not in* the run, and cannot be picked.
  {: #comparison-recordings }
- **Read snapshot** (advanced), which pinned copy of the behavior result to measure. Left on **Current artifact**, the comparison pins whatever the run holds now and records the id it got, so its numbers stay reproducible after the run is reopened and re-cut. Pick a snapshot to compare the copy an earlier comparison read.
  {: #comparison-snapshot }
- **Floor draws** and **Seed** (advanced): how many shuffled-label draws the difference ranking's floor is built from, and the seed that makes it reproduce. More draws is a steadier floor and a longer run. A paired contrast over ten or fewer animals enumerates every sign flip instead, so its floor is exact and the seed does not move it. Beside them, for within-animal designs: **Presence (s)**, how much total time at a level counts an animal as having the behavior there; **Minimum units (share)**, the share of the cohort a behavior must be present in for its effect to be defined at all (half by default; below it the effect is undefined, never large); **Pseudo-count (s)**, the second of time added to both sides of a ratio so a behavior absent at one level has a finite one; **Suppress nested nodes**, which collapses a cluster screen's nested slices of one behavior to the best of them; and **Rank on**, what a within-animal ranking scores each behavior by. The default, the *moderated t*, is the mean per-animal effect over its spread across animals (plus the family's median spread, so exact agreement cannot own the floor): a behavior ranks by how far your animals agree on it, not by how large its numbers are, so a 2× change every animal shows outranks a tenfold swing that goes either way, and the floor stops belonging to the sparsest clusters. *Effect* ranks on the mean effect itself, as comparisons run before this choice existed did, and those open as they were run. The effect is shown either way; a between-animal design has only the effect to rank on.
  {: #comparison-floor }
- **Measures** (advanced), which of the [measures](measures.md) to compute. Left empty, every measure the chosen behavior result *can* answer is computed: a measure that turns out not to move is as much a result as one that does. A measure the run cannot answer is greyed out saying why: `Transitions` needs behaviors that cannot overlap, and `Time in unnamed clusters` needs a clustering run. A label with labels inside it counts only what they leave, so its rows read "<label> (other)" in the Measures tab and in every table and figure the comparison writes.
  {: #comparison-measures }

Below the columns, the form shows the groups your choice produces and how big each is, before you run anything. A group holding a single recording is flagged: it has no spread.

A saved comparison opens on three tabs. **Overview** leads with the groups it compared, how many recordings and animals stand behind them, the design that ran and why, why no permutation floor was computed when that applies, and the largest changes with their figures. When a label it scores was written from a cluster screen over the same column, the header says so (*rearing (cmp_01) was chosen from a difference screen over drug, so the effect here is not an independent finding*) and that label's reading in the table ends with *chosen from screen cmp_01 (cluster #1375 kept)*. A cluster screen's own page lists the labels written from it (*Labelled from this screen: rearing (#1375)*), as its run holds them now. The floor line says how the floor was made and what the ranking scores by (*ranked on the moderated t … s0 = 0.42*). For a within-animal design the table is the ranking itself: each row's effect, its *t* when the ranking is on it, how many animals went that way, how many had the behavior at all, whether it clears the floor, its p's (*p (family · own)*: *0.02 · own 0.004*, the share of shuffled cohorts whose largest effect reaches the row and the share of its own shuffles that do), and the one-line reading ("less under drug than vehicle, 2.1×; 8 of 10 animals in that direction"). A cluster screen lists the kept nodes first, the next ten below the floor after them. **Measures** is the full table: every measure in a fixed order, so the same row sits in the same place across comparisons. **Figures** is all of them as a grid; double-click one to open it full size. **New comparison** opens the form pre-filled from this one, for when you want the same source grouped a different way; **Export** writes the whole publication tree wherever you point it, in the figure formats you pick. A group of one reports no spread rather than a zero.

The figures draw themselves when you open the tab, and **Redraw figures** draws them again after an edit. Figures are drawn on white, for print; tick **Dark background (for slides)** beside the button, or in the Export dialog, for a dark version. Either way you get one figure per measurement, whatever the window model: **bars** across groups when there is one window, and a **curve** across windows when there are several, a sliding-window comparison is one line per group, not one plot per window. A bar is the group's mean with its s.e.m.; every animal (or recording, when that is what one observation is) is a dot on it, in the same place in every bar, and when the same animal appears in more than one group a line joins its dots, because two independent clouds of dots throw away the pairing that makes a crossover worth running. When the comparison found a difference, the **P value is on the plot**, as a bracket over the groups it compares (*P = 0.004*, *P < 0.001*), and nothing is written when it did not; a test against a numeric column is labelled with the column (*dose: P = 0.002*). The symbol says which P it is: *P_FWE* when the measure had to clear the floor shared by everything compared (a behavior result, a cluster screen), *P* when it was tested on its own (movement and body shape), and *P < 0.005* when the shuffles could not resolve lower. Groups are drawn in their natural order (doses as numbers, the value you named as the reference first, and in a sex × dose design the two sexes at one dose side by side) and coloured by what they are: the reference grey, the steps of a numeric column one hue from light to dark, two groups grey and blue, more than two named values distinct hues, and in a sex × dose design one hue per sex with the dose setting its lightness. The y axis names the behavior and its unit (*Rearing (% time)*), and the x axis names the column or columns the groups came from (*Sex | fentanyl_dose* over bars labelled *F|0*, *M|0*, …), which is also the legend's title on the composite. **Panels** is the same measurements as one figure, a row or grid at a double column's width with one legend and panel letters, for when three measures belong in one place. Every comparison with something to rank (two or more groups, or a within-animal design) gets the **difference ranking**, with the floor that says how big a difference chance would produce under its design: shuffling the group labels across animals for a between design, flipping each animal's own direction or shuffling its levels for a within one. Every figure carries what one point is. **Transitions** is the exception that is not drawn: it answers per ordered pair, which is a matrix rather than a series, so it is reported as rows in the Measures tab and in `stats.csv` instead of as one two-point plot per cell. Figures and tables are written beside the comparison, ready to use: the PDF has editable type, the SVG opens in Illustrator as plain objects (no clipping masks, each dot its own path, each P value one text), and the contrast is a `booktabs` LaTeX table that drops into a manuscript.

Beside the figures, every comparison writes the statistics a paper cites, when it runs and again when it exports: `significant.csv`, one row per result that cleared with the group means and s.e.m., the effect, the P it was judged by and which P that is, n and how many animals went that way; `ranking.csv`, the same for everything compared; `units.csv`, one row per animal and measurement with a column per group, the shape a spreadsheet or Prism takes; `unit_effects.csv`, each animal's own change; and `methods.txt`, a paragraph for a Methods section saying which test ran, what was shuffled, how many times or that every configuration was taken, which P is reported and at what level, written from the comparison itself so it cannot disagree with what ran. The Overview shows that paragraph under the floor line, ready to copy.

Statistics stop at description: the only p-values are the permutation floor's own (each ranked row's share of the shuffles that reach it, family-wise and on its own, and a false-discovery q over them), with no model behind them. `stats.csv` beside the comparison is one row per observation with its factors attached, which is what a mixed model in R or statsmodels wants as input.

## Model manager {#models}

**Model manager** (a banner tab) is the machine's whole model library on one page. It keeps distilled models from getting lost in a folder: each global model has a name, metadata, and a preview. Its three tables each show four rows and scroll for the rest: **Global models** newest first, **Hugging Face** with the suggested models first.

**Global models** are the models every experiment can use (ones you imported, ones you restored from a zip, and ones you promoted from an experiment) in a table: **Name · From project(s) · Made by · Created · Kind**. **Kind** is *detector* or *keypoints*. **Double-click a row** to open its detail. The detail shows a **preview** (one overlay frame of the model's own segmentation, **double-click it** to see it full size and zoom in), its kind, description, notes, and classes, and a **build history** (the source project(s), dataset, model size, and metrics it was distilled from). A keypoint model's detail also shows its **skeleton**: the points, the left/right pairs, and the [Keypoints manager](#keypoints-manager) entry it matches. The model always uses its own copy of the skeleton. From the detail page you can **Edit** the metadata, **Set preview**, **Export**, or **Delete**. **Export** is also a button below the table.

**Models in ‹experiment›** is the open experiment's trained models, the same table as its [Models](#distill) page. Double-click one to open it inside the experiment. With no experiment open it says so.

**Export** bundles a model (weights, metadata, and preview, and a keypoint model's skeleton) into a `.zip` so a lab can copy it to another machine. There, **Restore from zip** brings it back into **Global models**. A keypoint model's skeleton is added to the Keypoints manager, unless a skeleton with the same points, edges and pairs is already there; a line under the buttons says which entry it matched or was added as. An entry with the same name and a different skeleton is never changed: the new one is added as "name (2)". **Don't move or rename** model folders; use Export to copy one out.

**Hugging Face** holds the foundation models. The two suggested ones (`sam3` and DINOv3's `dinov3_vit` · `vit_large`, the default embedding model, whose weights the keypoint method `dinov3_match` also uses) are listed first, then whatever you have installed, then the rest. Which models are installed is checked when the page opens, and the installed rows move up when that finishes; the row you selected and the column you sorted on stay as they were. A model with several sizes gets **one row per size**, each its own download. Select a row to **Check access** (readiness plus an action URL) or **Install** (download its weights). Install is required before a run: runs never download, so an uninstalled model is greyed out in the run forms until you install it here. Store the token on **Configuration ▸ Hugging Face**.

Two actions sit at the top. **＋ Distill new model** makes a fast YOLO model of either kind (**Model type**: a detector labelled by SAM 3, or a keypoint model taught by the patch method or finished keypoint detect runs): it's usable in that experiment right away, under **Models** in the rail, where the **Kind** column tells them apart. **Import model** adds an external `.pt`, capturing a name, who made it, a description, notes, and an optional preview. Once you choose the file, the dialog reads it and shows its **Kind**: a detector, with the classes in the checkpoint, or a keypoint (pose) model. A file that is not a YOLO `.pt` checkpoint is named there in red, and **Import** stays greyed. A keypoint model needs the **Skeleton** its points follow: a preset, a skeleton from the Keypoints manager, or **From file…** for a skeleton YAML. The dialog checks it against the model and **Import** stays greyed until it fits: the same number of points, and left/right pairs that match the ones the model swaps when it flips an image. The check cannot catch a pair named the wrong way round (left and right swapped), because that swaps the same points. A model that records no left/right swap accepts only a skeleton without pairs.

A distilled model doesn't land in **Global models** on its own: a trained model stays experiment-scoped until you **promote** it, so you can try several distill attempts and only add the keeper to the shared library. From a distill run's detail page, **Open trained model →** takes you to the trained model's own page, where **Promote to global** copies it (weights, name, and preview) into **Global models**.

## Keypoints manager {#keypoints-manager}

**Keypoints manager** (a banner tab) is the machine's skeleton library. A skeleton is defined here
once and copied by whatever uses it, so the same skeleton serves every experiment.

The page is a table: the four built-in presets (**Quadruped overhead view (12 points)**,
**Quadruped overhead view (8 points)**, **Quadruped side view (5 points)** and **Midline (3
points)**) and then your own skeletons, **Name · Kind · Points · Used by**. **Used
by** counts the example sets, keypoint runs, datasets and trained models in your experiments that
copied it. A copy counts when it holds the same points, edges and pairs, whatever id it was given
on the machine where it was made, so an experiment from another computer does not lock an
unrelated skeleton that happens to share its id. Only experiments in the app's workspace are
searched; one created with `experiment create --dir` somewhere else is not counted, and its own
copies are unaffected by an edit here. **+ New skeleton** and **Import…** (read a skeleton file) sit above the table;
**Edit…**, **Export…** and **Delete** act on the selected row, and double-clicking a row is
**Edit…**.

**+ New skeleton** and **Edit…** open the skeleton editor in a window. A new skeleton starts on an
empty canvas; the skeleton is drawn on a plain background with no video frame behind it. Draw
with the scene geometry editor's gestures: click empty space to add a point, drag a point to move
it, right-click a point to remove it and its edges, scroll to zoom and right-drag to pan. Edges are
a gesture too: select a point, then Shift+click a second point to join them, and Shift+click it
again to unjoin. Under the canvas, the selected **Point**'s name, its **Pair with** partner, which
**Side** of the pair it is, and **Remove point**. A pair's left member draws in blue and its right
in vermillion, the midline in green. **Save** writes it and closes the window; **Cancel** with
unsaved changes asks first.

**A skeleton something uses keeps its points.** While anything in your experiments uses it, an
edit to its points is saved as a new skeleton: the editor's **Save** reads **Save as new**, and **Delete** is
greyed and says what uses it. Renaming it is still allowed. Delete what uses it and the skeleton
can be edited or deleted again. A preset always saves as new.

## Quick tools {#tools}

One-off runs that need no experiment ([Tools guide](tools.md)). The **Quick tools** screen lists each tool, and a left **rail** lists them too, expand a tool to see its saved runs, oldest first, and click one to open its result. Open a tool to launch a run and browse its saved runs.

**Quick segment** runs a segmentation model on raw clips so you can judge it on real footage ([Tools guide](tools.md)). **Add** one or more videos, pick **Sample evenly across each video** (the default (the opening frames of a clip often show nothing) or **Consecutive from the start**, add one or more **prompts**, then **Run**. A sampled run compares every prompt on the same frames: the **Compare** tab is the prompt × frame grid) **double-click any cell** to open that frame full size and zoom into it (and **Play** steps through them full size) and plays them, where the sampled positions are evenly spaced, at the speed that spacing really represents (`source fps ÷ the gap`), so a run sampled every 30th frame plays a minute of footage in two seconds rather than pretending to be real time. Both share one **Detection confidence threshold** slider that re-draws and re-counts the saved run at any threshold. A consecutive run plays the overlay clip and offers **Export mp4**; tick **whole video** to segment every frame with no limit: it confirms first, and **Stop** ends it while keeping what it finished. Runs are saved and managed like project data. SAM 3 needs an NVIDIA GPU; your own imported or distilled models run anywhere.

## Getting your data out {#export}

**Export…** sits at the right of the strip above the screen, next to **Info**. It appears
whenever what you are looking at has data to write out, and opens one dialog listing everything
*this* screen can produce, so a clustering result offers clips, instances, an ethogram and per-frame labels (one row per video frame, ready for a GLM; [the columns are explained here](cli.md#per-frame-behavior-labels)), a
tracking run offers its tracks, and a recording offers its arena geometry. Tick what you want; it
all goes into one folder.

**Save to** is where to put the export, not what to write into. **Creates** beneath it shows the
folder it will make (`study3_temporal_clustering_an_017_export_2026-09-10_1432/`), so two exports
never collide and one is still identifiable months later on somebody else's machine.

Every export writes **the data behind the artifact**, not just the artifact: clips come with a
`clips.csv` saying which bout each file is, so a figure can be rebuilt outside DETAIL rather than
only watched. Each folder also gets a `params.csv` of the settings that produced the run, your
[notes](#notes) on what you exported from as `notes.md` when it has any, and an `export.yaml`
manifest. Each row's tooltip names the files it writes.

Opened on part of a run (a cluster, the instances you selected, a clip's span), the dialog
shows an **Of:** choice between that part and *Everything in this run*. A row that cannot be
narrowed to the part you chose is greyed and says why, rather than writing everything under a
manifest that says it was narrowed: with a selection, *Per-frame behavior labels* reads
"cannot be narrowed to a selection". Choose *Everything in this run* and it comes back. On a
recording, **Clip** is greyed until there is a frame range to write: use **Export video…** in a
clip player. A whole-run report such as the performance report is written whole beside a
narrowed export, and `export.yaml` lists it under `whole`.

- **The clip you are watching.** A clip player has an **Export video…** button beside its
  Masks checkbox, on a clustering result, a behavior scan, a query, the audit tab and the spatial
  map. It writes the span on screen, in the flavour on screen: masks on draws the outline, masks
  off writes the plain footage. It opens the same dialog as **Export…** above, already pointed at
  that clip, so the destination and the folder naming work the way every other export does. The
  `clip.csv` beside it names the recording, subject and frame range, so the file can be put back
  where it came from. A clip opened in its own window has no **Export video…**: its **Export ▾**
  writes what the window shows straight to a file you choose (see
  [Scrubbing frames](#scrubbing-frames)).
- **Pictures as well as rows.** Anything with a plot exports the plot too, as PNG and SVG: the
  ethogram, time per behavior, the spread of bout durations. They are drawn on white for print;
  tick **Dark background (for slides)** in the dialog for a dark version.
- **A row you cannot tick says why.** Something this screen could export but has none of yet: no
  streams imported, nothing drawn on the arena, is shown greyed with the reason, rather than
  left out. Video that was **referenced in place** is greyed too: those files are already yours,
  under the video root they were registered from.
- **Clips** come in three flavours (raw, raw + mask overlay, and mask only) and you can tick
  more than one: they are written in a single decode pass, so three cost about what one does.
  The full field of view is the default; crop to the subject when you want the pose at a
  constant scale.
- **Select first to export less.** With a cluster selected on **Interpret**, or rows
  selected on **Browse**, an extra row appears offering that selection or the whole run. On a
  clustering run's **BROWSE**, **Export all** exports exactly what passes its filters.
- **Choose which recordings.** A processing run that covers several recordings (detection,
  tracking, embedding, keypoints) shows a **Recordings** list in the dialog, all ticked, with
  **All** and **None** beside it. Untick what you do not want: the tracks, masks, vectors, points
  and keypoint video and the other things written per recording are written for the ticked ones
  only. From a run's results screen, **Export…** opens with the recording you have selected
  in the table already chosen, so exporting what is on screen is the default and widening it is a
  click. The list is greyed (not hidden) while nothing ticked is written per recording, and a run
  of one recording has no list. An analysis run is narrowed by cluster or selection here;
  narrowing it by recording is [`--for-recording`](cli.md#exporting-part-of-a-run) on the command
  line.
- **How big, how long, and whether it fits.** Under the list the dialog says what ticking all this will write and how long it will take ("about 2.3 GB, about 4 min, 118 GB free there"), and re-asks a moment after each change. Hover it to see how each row was worked out. Video is timed on a few dozen frames of the real footage, so its seconds run high. If a row cannot be sized (the performance report, which depends on what is installed, and the SLEAP and NWB keypoint files, whose size depends on how well their arrays compress) the total says **at least**, and that row adds nothing to the free-space check. If the estimate is more than the drive under **Save to** has free, it says the export **will not fit** and **Export** is off until you tick less or pick another drive. (An export that writes one video per recording can run to hundreds of gigabytes; this is what stops it filling a drive.)
- **An export keeps going without you, and does not keep you.** Clicking **Export** closes the
  dialog and leaves you on the screen you were using; a line appears under the toolbar naming what
  started and where it is writing, with a button through to **Active processes**. Every export is a
  run: it survives closing the app and can be cancelled from its own screen, and it stays listed
  under **Exports** after it finishes, so you can find the folder later. One export is one run
  however many things you tick: they share a folder, and the run reports each separately, so one
  failing still leaves you the rest plus a note saying which failed and why.
- **The runs have their own data.** A tracking run exports **Tracks**: a row per frame per track,
  with the box and the mask centroid in fractions of the frame, in pixels, and in millimetres
  where the recording is calibrated. A detection run exports **Masks** as COCO
  instance-segmentation JSON. Neither needs an analysis first, and each writes one file per
  recording the run covered. A keypoint run (detect or track) offers five rows: **Keypoints**
  (DETAIL's long CSV), **Keypoints for DeepLabCut**, **Keypoints for SLEAP**, **Keypoints as
  NWB** and **Keypoints video** (the footage with the points drawn on). The dialog adds their
  shared knobs: **Points from** on a track run (**Tracking**, or **Detection, before
  tracking and smoothing** to see what tracking changed), the **Score threshold** (the method's default
  unless you move it), **Omit points below score threshold** (off, such a point keeps its
  position and is marked not visible; on, its position is written as empty), and for the
  video **Video: show masks and boxes**, which draws the tracking run's masks and boxes under
  the points. Every file in one export reads the same choices, so they agree point for point.

The same catalogue is on the command line as `detail export`, see the
[CLI guide](cli.md#getting-your-data-out) for the column meanings.

## Moving experiments in/out

- **Export** the whole experiment from **Export experiment…** on the experiment's overview page,
  or from **Export…** in the top bar with the experiment on screen and **Experiment bundle**
  ticked. Either way the app goes to the export's page, with a bar through each phase (finding
  the files, checking them, computing the checksum, packing) and the time left. It runs in the
  background and stays listed under **Active processes**. The archive is `<experiment name>.tar`,
  inside the export's own folder.
- **Let running jobs finish first.** The export reads every file twice, and a file that changes
  in between (a run still writing to the experiment) stops the export with a message naming the
  file, rather than producing an archive that would fail on import. Export records are not
  packed: they say where *this* machine wrote copies. Nor is an earlier archive saved inside the
  experiment folder. A folder that is a symlink is not followed, so nothing in it travels; the
  archive's `export.yaml` lists each one under `symlinked_folders_left_out`, and
  `experiment export` prints them.
- **Import** from **Experiments → Import experiment**: a bar over the page, with the time left
  and **Cancel**. Importing onto an experiment of the same name asks whether to replace it, and
  first lists what replacing deletes that the archive does not bring back (a slim archive's
  video files, with their count and size, and datasets; export records; earlier archives in the
  folder). Choose **No** and import a copy under another name if you want to keep them. Video
  deleted this way can be put back from the original files with **Locate video** on each
  recording's page.
- **Source video that would pack nothing says so.** With **Source video** ticked, the Export
  dialog says "No video will be packed" and why when no recording has video to pack here: their
  copied video is not on this computer (put it back with **Locate video**), or their video is
  used in place, which never travels in an archive.
- **Gzip is rarely worth ticking.** DETAIL's data files are compressed already, so it barely shrinks the bundle and makes the export much slower.
- **Video used in place is found again.** When a recording's video was used where it was (not
  copied), the archive notes which folder it was in on the computer that exported it. Importing
  looks in that same folder, so video on a server both computers reach is found without you doing
  anything. It also looks beside every folder it finds, so a study kept as one folder per day is
  found from any one of them, even when the two computers write the server's address
  differently. A folder counts only when a video in it matches the one the recording was made
  from, and a file at the recorded path that is a different video counts as not found.
  Any recording whose video still isn't found is listed by name when the import finishes; open
  one and use **Locate video** on its file, and every recording whose video is in the same folder
  finds it too.

An experiment exported without its source video still works for analysis; only the frame-preview panes need the video.

## CLI vs GUI

- **GUI-only:** drawing the field of view, testing prompts, previewing the renderer, the embedding result view, stream sync/preview, Quick-segment playback.
- **CLI-friendlier:** headless or batch processing, scripting, and multi-GPU runs on a server. See the [CLI guide](cli.md).
