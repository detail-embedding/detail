# Tools

Tools are one-off runs that need no experiment: point at a file, run a model, look at the
result. Each run is saved to a machine-global tools workspace
(`DetailExperiments/tools/`) and managed like project data: list, open, export, delete.

## Quick segment

Run a segmentation model on raw video clips and look at what it found. No import: the files
are read in place. Use it to judge a model on real footage before committing to a project,
and to get a result you can show someone.

Two modes:

- **Sample evenly across each video** (default) spreads the frames over the whole clip and
  runs **every prompt** on each one, so you can compare `mouse` against `mouse with tail`
  side by side. Sampling rather than reading the opening frames is the point: a still
  animal or a blank lead-in shows nothing in the first few seconds.
- **Consecutive from the start** reads straight through from frame 0. One clip, one prompt,
  and the only mode an overlay mp4 can be made from. Tick **whole video** to segment every
  frame with no limit; that can run for hours, so it asks first and you can **Stop** it.
  A stopped run keeps what it finished.

### GUI

Quick tools → Quick segment. **Add…** one or more videos, pick the mode and frame count,
add your prompts and choose the model, then **Run**. The line under the video
list estimates what the run costs (`frames × prompts = inferences`).

A sampled run opens on **Compare** (the prompt × frame grid; one prompt skips straight to
**Play**) and **Play** steps through the sampled frames at full size. A consecutive run
plays the overlay clip instead.

Every result carries a **Detection confidence threshold (slide to change)** box, above
the video and separate from the frame scrubber below it. The run kept every proposal the
model found, so moving it re-draws and re-counts what you already ran: drop it to see
faint detections the default hides, raise it to cut noise. The caption beside it says how
many of the stored detections the current setting keeps. An export bakes in whatever the
slider is showing, since an image can't be re-thresholded later.

**Export** (frames or mp4, by mode) asks where to put it and creates its own folder there,
named from the clip and the run (`openfield_01_quick-segment_run_003/`), holding the
overlay stills plus `test.json`, or the overlay clip under the same name. Two exports never
land on top of each other, and a folder still says what it is once it has left the app.

The Configuration checklist's **See it segment a clip** step opens this tool.

### CLI

```
detail tools quick-segment CLIP.mp4 [CLIP2.mp4 …] --prompt mouse [--prompt "mouse with tail"] [--frames 50] [--min-conf 0.25]
detail tools quick-segment CLIP.mp4 --consecutive --prompt mouse [--whole-video] [--export out.mp4]
```

Each `--prompt` is a **separate prompt**, compared side by side; put several classes in one
prompt by comma-separating them (`--prompt "mouse,cricket"`). `--sample` (the default)
spreads the frames across each clip; `--consecutive` reads from frame 0 and is what
`--export` needs. Left out, `--backend` takes the best backend the machine has that the model
supports (CUDA, then MPS, then CPU), so the default model, SAM 3, runs on CUDA with no flag.
With one of your own models (`--model <id>`), `--prompt` is still
required, but the model ignores its text and finds its own classes.

```
detail tools list            # saved runs
detail tools delete run_003  # remove one
```

### Requirements

Quick segment auto-picks **SAM 3**, which needs an NVIDIA GPU + CUDA (see
[getting started](getting-started.md)), unless you choose one of your own imported or
distilled models, which run anywhere and take no prompt.

## What's saved

Neither mode stores the video.

A **sampled** run saves `test.json` (what was sampled, and the detection count per frame
per prompt), one detection store per prompt, and each clip's frame-to-time map. No images:
the grid and the frame view draw overlays when you look at them, and **Export frames…** is
where pictures get made.

A **consecutive** run saves its **detections** plus the clip's `frame_idx → t_ns` map, and
re-renders the overlay from the source frames on the fly. The map is what keeps the overlay
honest: playback and export resolve exactly the frames the run segmented, instead of
deriving their own frame numbering and landing on a neighbour.

Both carry a `tool.yaml` manifest pointing at the source video (by absolute path; the file
isn't copied), and neither writes an image or a video of its own, so a run costs kilobytes
until you export. Deleting a run removes its folder.

## Adding a tool

Quick segment is the first tool; the framework is general. A new tool is one `Tool` in
`detail/tools/` (a pydantic config + a `run`) plus a GUI `ToolUI` in `detail/gui/tools/`, each
named in its module list (`TOOL_MODULES`, `TOOL_UI_MODULES`); it then appears in the Tools list
and sidebar with no shell change. Its CLI command is written by hand under `detail tools` in
`detail/cli.py`. A tool whose work can run
long takes the `stop_check` its `run` is handed and polls it, so the Stop button works.
