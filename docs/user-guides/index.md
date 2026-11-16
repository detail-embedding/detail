# User guides

A step-by-step path from a new computer to analyzed behavior, for experiments with **one
animal in each video**. You install DETAIL, set it up, then work through a full analysis in
the DETAIL desktop app, one chapter at a time. No programming is needed. You type (or paste) a
few commands during installation, and every one is given to you. After that, everything
happens in the DETAIL window.

Every chapter follows one example experiment, filmed from above. The guide works the same for
a camera beside the animal: [Side-view and isometric
recordings](single-animal/keypoints.md#side-view-and-isometric-recordings), in chapter 4, says
what changes.

!!! note "An NVIDIA GPU is required"
    DETAIL finds and follows animals with AI models that need an NVIDIA graphics card (GPU)
    in your computer. A computer with no NVIDIA GPU, such as any Mac and most laptops, cannot
    run them.

??? info "About GPUs"
    A GPU (graphics processing unit) is the chip that draws your screen. The AI models DETAIL
    uses run thousands of times faster on a GPU than on the main processor, but only on GPUs
    made by NVIDIA, because they are built on NVIDIA's CUDA software. Intel and AMD graphics,
    and Apple's chips, cannot run them.

## The path at a glance

First [install DETAIL](#choose-your-install-page), once per computer, then [configure
it](configure.md). The chapters come after that.

Each box is one chapter. Click a box to open it. The first tag says where the chapter works in
DETAIL: the button along the top, then the page in the list on the left. A **GPU** tag means
the chapter's heavy work runs on your computer's GPU, and takes a while.

<div class="guide-flow" markdown>

1. [Create an experiment and add recordings](single-animal/create-an-experiment.md) <span class="where where-app">Experiments › Recordings</span> <span class="note">Bring your videos in and describe each recording.</span>
2. [Set up each recording](single-animal/set-up-recordings.md) <span class="where where-app">Experiments › Recordings › a recording</span> <span class="note">Draw where the animal can be, and how big things are.</span>
3. [Detect and track](single-animal/detect-and-track.md) <span class="where where-app">Experiments › Models</span> <span class="where where-app">Experiments › Processing pipeline</span> <span class="where where-gpu">GPU</span> <span class="note">Find the animal in every frame, and follow it.</span>
4. [Keypoints](single-animal/keypoints.md) <span class="where where-app">Experiments › Models</span> <span class="where where-app">Experiments › Processing pipeline</span> <span class="where where-gpu">GPU</span> <span class="note">Place body points on the animal: nose, paws, tail base.</span>
5. [Embedding](single-animal/embedding.md) <span class="where where-app">Experiments › Processing pipeline</span> <span class="where where-gpu">GPU</span> <span class="note">Describe what the animal looks like in each frame, for the analyses.</span>
6. [Location export](single-animal/location-export.md) <span class="where where-app">Experiments › Analysis</span> <span class="note">Where the animal is on every frame, against your regions and points, as spreadsheets.</span>
7. [Behavior scan](single-animal/behavior-scan.md) <span class="where where-app">Experiments › Analysis</span> <span class="note">Find a behavior you can name, from a few examples.</span>
8. [Clustering](single-animal/clustering.md) <span class="where where-app">Experiments › Analysis</span> <span class="note">Let DETAIL group frames into behaviors you have not named yet.</span>


Chapters 1 to 5 build on each other: do them in order. Chapter 6 needs only chapters 1 to 3,
so it can come as soon as tracking is done. Chapters 7 and 8 both start from chapter 5, so do
either or both.

</div>

??? info "About this guide: what each step makes"
    Each chapter makes a **run**: a saved result that the next chapter starts from. Detection
    finds the animal in each frame. Tracking joins those into one animal over time. Keypoints
    place body points on it. Embedding turns each frame into a list of numbers that describes
    the animal's posture, so that frames of the same behavior have similar numbers. Behavior
    scan and clustering both work from those numbers.

    Every run keeps the settings it was made with, so you can see later exactly how a result
    was made, and start a new run from the same settings.

## Find a task

Every task is a step in one of the chapters. To do one thing without reading in order, find it
here.

| To | Go to |
|----|-------|
| Install DETAIL, or update it | [Install on Windows](install-windows.md), [Install on Linux](install-linux.md) |
| Launch DETAIL again later | [Windows](install-windows.md#launch-it-again-later), [Linux](install-linux.md#launch-it-again-later) |
| Get PyTorch to use the GPU | [Configure, step 1](configure.md#1-get-pytorch-to-use-the-gpu) |
| Set up a Hugging Face token | [Configure, step 2](configure.md#2-set-up-hugging-face-access) |
| Install SAM 3 and download the models | [Configure, steps 3 and 4](configure.md#3-install-the-sam-3-backend) |
| Create an experiment | [Chapter 1, step 1](single-animal/create-an-experiment.md#1-create-the-experiment) |
| Add videos as recordings | [Chapter 1, step 2](single-animal/create-an-experiment.md#2-add-the-recordings) |
| Describe each recording: box, animal, dose, date | [Chapter 1, step 3](single-animal/create-an-experiment.md#3-fill-in-recording-information) |
| Add a recording that is in several video files | [Chapter 1, step 4](single-animal/create-an-experiment.md#4-optional-a-recording-in-several-files) |
| Add many recordings at once, from a manifest file | [Chapter 1, step 5](single-animal/create-an-experiment.md#5-optional-many-recordings-at-once) |
| Draw the field of view | [Chapter 2, step 2](single-animal/set-up-recordings.md#2-draw-the-field-of-view) |
| Calibrate distance, for results in millimetres | [Chapter 2, step 3](single-animal/set-up-recordings.md#3-calibrate-distance) |
| Mark regions and points, such as a lever or a corner | [Chapter 2, step 4](single-animal/set-up-recordings.md#4-optional-mark-regions-and-points) |
| Copy a setup to other recordings | [Chapter 2, step 5](single-animal/set-up-recordings.md#5-set-up-the-other-recordings) |
| Add a stream: box events, a sensor log, hand-scored behavior | [Chapter 2, step 6](single-animal/set-up-recordings.md#6-optional-add-streams) |
| Line a stream up with the video | [Chapter 2, step 6](single-animal/set-up-recordings.md#line-the-stream-up-with-the-video) |
| Test a detection prompt | [Chapter 3, step 1](single-animal/detect-and-track.md#1-test-the-prompt) |
| Train a fast detector | [Chapter 3, step 2](single-animal/detect-and-track.md#2-distill-a-fast-detector) |
| Detect and track the animal | [Chapter 3, steps 3 and 4](single-animal/detect-and-track.md#3-detect-every-frame) |
| Save the tracks as files | [Chapter 3, step 4](single-animal/detect-and-track.md#optional-save-the-tracks) |
| Make your own skeleton | [Chapter 4, step 2](single-animal/keypoints.md#make-your-own-skeleton) |
| Place and track keypoints | [Chapter 4, steps 4 and 5](single-animal/keypoints.md#4-place-keypoints-on-every-frame) |
| Take keypoints to DeepLabCut, SLEAP or NWB | [Chapter 4, step 5](single-animal/keypoints.md#optional-take-the-keypoints-to-deeplabcut-sleap-or-nwb) |
| Work with a side-view recording | [Chapter 4](single-animal/keypoints.md#side-view-and-isometric-recordings) |
| Run an embedding | [Chapter 5](single-animal/embedding.md) |
| Export where the animal is, as spreadsheets | [Chapter 6](single-animal/location-export.md) |
| Find a behavior from a few examples | [Chapter 7](single-animal/behavior-scan.md) |
| Check a scan against hand-scored videos | [Chapter 7](single-animal/behavior-scan.md#if-you-have-hand-scored-videos) |
| Group behavior into clusters and label them | [Chapter 8](single-animal/clustering.md) |
| Save a cluster's clips | [Chapter 8, step 4](single-animal/clustering.md#save-the-clips) |

## Choose your install page

Every guide starts with the same installation. Find the row that matches your computer and
follow only that install page. Each page ends by telling you which page to read next.

| Your computer | Install page |
|---------------|--------------|
| Windows 10 or 11, with an NVIDIA GPU | [Install on Windows](install-windows.md) |
| Linux, with an NVIDIA GPU | [Install on Linux](install-linux.md) |

??? info "How to tell whether your computer has an NVIDIA GPU"
    - **Windows:** press ++ctrl+shift+esc++ to open Task Manager. Click **More details** if you
      see it, then the **Performance** tab. Each **GPU** entry on the left names its maker. You
      need one that says **NVIDIA**.
    - **Linux:** open a terminal and run `lspci | grep -i nvidia`. A line that names an NVIDIA
      device means you have one.

After the install page, go on to [Configure DETAIL](configure.md), then chapter 1,
[Create an experiment and add recordings](single-animal/create-an-experiment.md).

## Hardware and disk space

The computer you install DETAIL on needs:

| | Minimum | Recommended |
|---|---|---|
| NVIDIA GPU | 8 GB of GPU memory, such as a GeForce RTX 2070 SUPER or RTX 3060 | 16 GB of GPU memory or more, such as a GeForce RTX 4080 or RTX 4090 |
| Memory (RAM) | 16 GB | 32 GB or more |
| Processor | 4 cores | 8 cores or more |
| Free disk space for the installation | 25 GB | 50 GB, on an SSD |
| Operating system | Windows 10 or 11 (64-bit), or Ubuntu 22.04 or newer | the same |

The installation takes about 12 GB once it is done. The rest of the free space in the table
is room for the installer's temporary downloads and for the files your runs make. Your
recordings need space of their own on top of this.

| Part of the installation | Size |
|---|---|
| DETAIL itself | 0.3 GB |
| The Python environment, including PyTorch with GPU support | 7 GB |
| SAM 3, the detection model | 3.5 GB |
| DINOv3, the embedding model | 1.2 GB |
| Keypoint and tracking models | under 0.5 GB |

??? info "About these numbers"
    The sizes were measured on a Windows install. GPU memory matters most: a model that does
    not fit in it stops with an "out of memory" error, or on Windows slows to a crawl. More
    GPU memory lets DETAIL process more video frames at once, which is faster. The settings
    that decide how much memory a step uses, such as the **Batch size** for training a model,
    start at values that fit an 8 GB GPU. [Detect and
    track](single-animal/detect-and-track.md#batch-size-and-your-gpu) says how to raise them
    on a larger one. RAM matters when you analyze many long
    recordings together.

    To see how much memory your GPU has on Windows, open Task Manager's **Performance** tab
    and click the NVIDIA GPU: **Dedicated GPU memory** is the number to compare. On Linux,
    `nvidia-smi` shows it as the second number under **Memory-Usage**.

## What you need

- **A computer you can install software on.** Some steps ask for an administrator password.
- **About an hour** for installation. Most of it is waiting for downloads.
- **Free disk space** for DETAIL, the AI models, and your videos. Plan for tens of gigabytes
  before your videos.
- **Your GitHub account.** DETAIL downloads from GitHub. The install page signs your computer
  in to the account you read these guides with.
- **A Hugging Face account.** The AI models download from Hugging Face. [Configure
  DETAIL](configure.md) shows you how to make one.
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
    recordings](single-animal/keypoints.md#side-view-and-isometric-recordings), in chapter 4, says what changes for a
    camera beside the animal:

    - **The keypoints.** A view from above and a side view see different body points, so they
      use different skeletons (chapter 4).
    - **Turning the animal.** Before describing each frame, DETAIL turns the animal to face one
      way: from above it turns the head to point right, from the side it flips the animal to
      face right (chapter 5).
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

## How to read these guides

Each page is a list of numbered steps. Do them in order, and do not skip a step's check: it
tells you the step worked before you build on it.

- Grey boxes like the one below hold a command. Copy it with the button at the box's right
  edge, paste it into your terminal, and press ++enter++.

    ```
    echo hello
    ```

- The words before each command say where to type it, such as **In PowerShell** or **In
  the terminal**. When a step says **the same window**, use the window the previous
  step left you in: it depends on the folder or setting that step chose.
- **About this step** boxes explain what a step does and why. They are closed until you click
  them, and you can skip them.
- **If this doesn't work** boxes list what to try when a check fails.
- The first time a screen or dialog appears, its screenshot has numbered markers on every
  button and field, and a **What everything on this screen does** box under it says what each
  number does. Later chapters link back to it rather than repeat it.

??? info "New to the terminal?"
    A terminal is a window where you type commands instead of clicking. Windows calls its
    terminal **PowerShell**, and Linux usually calls it **Terminal**. The install pages tell you how to open it.

    The terminal shows a **prompt**, a short line ending in `>` or `$` that means it is ready.
    After you press ++enter++, wait for the prompt to come back before you type the next
    command. Some commands take minutes and print a lot of text while they work. That is
    normal.

    Red text or a line containing `error` usually means the command failed. Stop there and
    check that step's **If this doesn't work** box before going on.
