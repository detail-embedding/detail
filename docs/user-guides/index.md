# User guides

Step-by-step guides that take you from a new computer to analyzed behavior. Each one starts
with installing DETAIL, then works through a full analysis in the DETAIL desktop app. No
programming is needed. You will type (or paste) a few commands during installation, and every
one is given to you.

| Guide | For |
|-------|-----|
| [Single animal](single-animal/index.md) | One animal per video, filmed from above or from the side |

!!! note "An NVIDIA GPU is required"
    DETAIL finds and follows animals with AI models that need an NVIDIA graphics card (GPU).
    That GPU can be in your own computer, or on Tillicum, the University of Washington's GPU
    cluster. A computer with no NVIDIA GPU (any Mac, and most laptops) can use DETAIL only
    through Tillicum.

??? info "About GPUs and Tillicum"
    A GPU (graphics processing unit) is the chip that draws your screen. The AI models DETAIL
    uses run thousands of times faster on a GPU than on the main processor, but only on GPUs
    made by NVIDIA, because they are built on NVIDIA's CUDA software. Intel and AMD graphics,
    and Apple's chips, cannot run them.

    Tillicum is a shared computer cluster with many NVIDIA GPUs. DETAIL can send its heaviest
    work (finding the animal, its body points, and what it looks like in each frame) to
    Tillicum and bring the results back to your computer. You still use DETAIL on your own
    computer the whole time. Tillicum needs a UW account and a request for access.

## Choose your setup

Every guide starts with the same installation. Find the row that matches your computer and
follow only that install page. Each page ends by telling you which page to read next.

| Your computer | Heavy work runs on | Install page |
|---------------|--------------------|--------------|
| Windows, with an NVIDIA GPU | your computer | [Install on Windows](install-windows.md) |
| Windows, with an NVIDIA GPU, and you have Tillicum access | your computer or Tillicum | [Install on Windows with WSL](install-windows-wsl.md) |
| Linux, with an NVIDIA GPU | your computer or Tillicum | [Install on Linux](install-linux.md) |
| Mac, and you have Tillicum access | Tillicum | [Install on macOS](install-macos.md) |

??? info "How to tell whether your computer has an NVIDIA GPU"
    - **Windows:** press ++ctrl+shift+esc++ to open Task Manager. Click **More details** if you
      see it, then the **Performance** tab. Each **GPU** entry on the left names its maker. You
      need one that says **NVIDIA**.
    - **Linux:** open a terminal and run `lspci | grep -i nvidia`. A line that names an NVIDIA
      device means you have one.
    - **Mac:** Macs do not have NVIDIA GPUs. Use Tillicum.

After the install page, every setup continues the same way:

1. [Configure DETAIL](configure.md): set up the GPU, the detection model, and model downloads.
2. [Set up Tillicum](tillicum.md): only if your setup uses Tillicum.
3. The guide's own chapters, starting with [Single animal](single-animal/index.md).

## Hardware and disk space

The computer you install DETAIL on needs:

| | Minimum | Recommended |
|---|---|---|
| NVIDIA GPU (in your computer, or use Tillicum) | 8 GB of GPU memory, such as a GeForce RTX 2070 SUPER or RTX 3060 | 16 GB of GPU memory or more, such as a GeForce RTX 4080 or RTX 4090 |
| Memory (RAM) | 16 GB | 32 GB or more |
| Processor | 4 cores | 8 cores or more |
| Free disk space for the installation | 25 GB | 50 GB, on an SSD |
| Operating system | Windows 10 or 11 (64-bit), or Ubuntu 22.04 or newer | the same; a Mac only with Tillicum |

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

With Tillicum, the models are downloaded on Tillicum instead, into your project storage
there, so your computer needs less.

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
- **A GitHub account.** DETAIL is shared through GitHub. The install page shows you how to
  make one and get access.
- **A Hugging Face account.** The AI models download from Hugging Face. [Configure
  DETAIL](configure.md) shows you how to make one.
- **For Tillicum:** a UW NetID with Duo two-factor sign-in. Ask for cluster access early,
  because it can take a few days ([Set up Tillicum](tillicum.md)).

## How to read these guides

Each page is a list of numbered steps. Do them in order, and do not skip a step's check: it
tells you the step worked before you build on it.

- Grey boxes like the one below hold a command. Copy it with the button at the box's right
  edge, paste it into your terminal, and press ++enter++.

    ```
    echo hello
    ```

- The words before each command say where to type it, such as **In PowerShell** or **In
  Tillicum's terminal**. When a step says **the same window**, use the window the previous
  step left you in: it depends on the folder or setting that step chose.
- **About this step** boxes explain what a step does and why. They are closed until you click
  them, and you can skip them.
- **If this doesn't work** boxes list what to try when a check fails.

??? info "New to the terminal?"
    A terminal is a window where you type commands instead of clicking. Windows calls its
    terminal **PowerShell**, a Mac calls it **Terminal**, and Linux usually calls it
    **Terminal** too. The install pages tell you how to open it.

    The terminal shows a **prompt**, a short line ending in `>` or `$` that means it is ready.
    After you press ++enter++, wait for the prompt to come back before you type the next
    command. Some commands take minutes and print a lot of text while they work. That is
    normal.

    Red text or a line containing `error` usually means the command failed. Stop there and
    check that step's **If this doesn't work** box before going on.
