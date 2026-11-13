# Single animal

This guide takes you from a folder of videos to analyzed behavior, for experiments with **one
animal in each video**. It works the same whether the camera looks down on the animal or at it
from the side. Every chapter follows one example experiment, filmed from above, with three
recordings. [Side-view and isometric recordings](keypoints.md#side-view-and-isometric-recordings), in chapter 4, says
what changes for a camera beside the animal.

Everything happens in the DETAIL window. You do not type commands in this guide.

!!! note "Before you start"
    Install and configure DETAIL first: your [install page](../index.md#choose-your-setup), then
    [Configure DETAIL](../configure.md). The Configuration checklist shows **Setup complete**
    when you are ready. To send the heavy work to Tillicum, [set up Tillicum](../tillicum.md)
    too.

The first time a screen or dialog appears in the guide, its screenshot has numbered markers
on every button and field, and a **What everything on this screen does** box under it says
what each number does. Later chapters link back to it rather than repeat it.

## The chapters at a glance

Each box is one chapter. Click a box to open it. The first tag says where the chapter works in DETAIL: the button along the top, then the page in the list on the left. A **GPU** tag means the chapter's heavy work can run on your computer or on Tillicum.

<div class="guide-flow" markdown>

1. [Create an experiment and add recordings](create-an-experiment.md) <span class="where where-app">Experiments › Recordings</span> <span class="note">Bring your videos in and describe each recording.</span>
2. [Set up each recording](set-up-recordings.md) <span class="where where-app">Experiments › Recordings › a recording</span> <span class="note">Draw where the animal can be, and how big things are.</span>
3. [Detect and track](detect-and-track.md) <span class="where where-app">Experiments › Models</span> <span class="where where-app">Experiments › Processing pipeline</span> <span class="where where-gpu">GPU: your computer or Tillicum</span> <span class="note">Find the animal in every frame, and follow it.</span>
4. [Keypoints](keypoints.md) <span class="where where-app">Experiments › Models</span> <span class="where where-gpu">GPU: your computer or Tillicum</span> <span class="note">Place body points on the animal: nose, paws, tail base.</span>
5. [Embedding](embedding.md) <span class="where where-app">Experiments › Processing pipeline</span> <span class="where where-gpu">GPU: your computer or Tillicum</span> <span class="note">Describe what the animal looks like in each frame, for the analyses.</span>
6. [Location export](location-export.md) <span class="where where-app">Experiments › Analysis</span> <span class="note">Where the animal is on every frame, against your regions and points, as spreadsheets, plots and a side-by-side video.</span>
7. [Behavior scan](behavior-scan.md) <span class="where where-app">Experiments › Analysis</span> <span class="note">Find a behavior you can name, from a few examples.</span>
8. [Clustering](clustering.md) <span class="where where-app">Experiments › Analysis</span> <span class="note">Let DETAIL group frames into behaviors you have not named yet.</span>

</div>

Chapters 1 to 5 build on each other: do them in order. Chapter 6 needs only chapters 1 to 3,
so it can come as soon as tracking is done. Chapters 7 and 8 both start from chapter 5, so do
either or both.

??? info "About this guide: what each step makes"
    Each chapter makes a **run**: a saved result that the next chapter starts from. Detection
    finds the animal in each frame. Tracking joins those into one animal over time. Keypoints
    place body points on it. Embedding turns each frame into a list of numbers that describes
    the animal's posture, so that frames of the same behavior have similar numbers. Behavior
    scan and clustering both work from those numbers.

    Every run keeps the settings it was made with, so you can see later exactly how a result
    was made, and start a new run from the same settings.

## What you need

- **Your videos**, one animal in each. A few minutes of video is enough to follow the guide.
  Common formats such as `.mp4` and `.avi` work.
- **Time.** The chapters with the GPU tag run for a while: from minutes for a short video to
  hours for a large experiment. You can keep using DETAIL while they run.
- **For a side view:** the whole animal in the picture, from the side.
  **For a view from above:** the whole floor of the arena in the picture.

??? info "About this: closing DETAIL while something is running"
    Runs carry on without DETAIL: detection, tracking, distills and the rest each run as
    their own process, so you can close the window and open it again later to see how they
    stand. **Video imports do not**: they run inside DETAIL, so closing it stops them, and
    those recordings are marked **failed**. Leave DETAIL open until every import says
    **ready**.

    If you close DETAIL while something is running, it asks first, in a window called
    **Still in progress**:

    - **Cancel runs & quit** stops everything that is running, then closes.
    - **Leave running & quit** closes and leaves the runs going. When only imports are
      running it says **Quit anyway**, because they stop either way.
    - **Don't quit** keeps DETAIL open.

??? info "About this: what is different between the two views"
    Most of the guide is the same for both views. The demo is filmed from above; [Side-view and isometric
    recordings](keypoints.md#side-view-and-isometric-recordings), in chapter 4, says what changes for a
    camera beside the animal:

    - **The camera view setting.** You tell DETAIL which way the camera looks (chapter 1). Some
      measures only mean something from one view, such as how high the animal reaches, which
      only means something from the side.
    - **The keypoints.** A view from above and a side view see different body points, so they
      use different skeletons, and a side view picks its example frames by the direction the
      animal faces (chapter 4).
    - **What a behavior looks like.** Rearing, for example, is the animal standing up. From the
      side you see it stand; from above you see its outline shrink. Both can be found, from
      different evidence.

## The example experiment

| | |
|---|---|
| Camera | above the arena, looking down |
| Recordings | 3, one mouse in each |
| Length | 5 minutes each |

Your own experiment will have its own names and settings. The guide says which choices depend
on your experiment and which you can leave as they are.

## Next

[Create an experiment and add recordings](create-an-experiment.md).
