# Getting started

Requires Python ≥ 3.12 and a virtual environment
([uv](https://docs.astral.sh/uv/) in these examples).

A full setup runs through the sections below, in order:

1. [Install](#install) the repo and its Python dependencies into a virtual environment.
2. [GPU and CUDA](#gpu-and-cuda): install a CUDA build of torch (required for SAM3).
3. [SAM3 backend](#sam3-backend): install the SAM3 detector from source (NVIDIA GPU only).
4. [Hugging Face access](#hugging-face-access): get a token for the gated SAM3 and DINOv3 weights.
5. [Run](#run) the desktop app or CLI.

Already installed and just want to start the app? See
[Launch it again later](#launch-it-again-later).

Steps 2–4 are also a checklist in the app: install the `gui` extra, launch
`detail-gui`, and work through **Configuration**, which reads the live state of
this machine, writes the install commands with your own paths filled in, and
re-checks without a restart ([GUI guide](gui.md#configuration)). Reinstalling
PyTorch is the exception: restart DETAIL afterwards, because a running app keeps
the PyTorch it started with. `detail doctor` is the same checks headless.

## Install

Install uv, if you don't have it:

=== "macOS / Linux"

    ```
    brew install uv
    ```

=== "Windows"

    ```
    pipx install uv
    ```

Either OS can use the [standalone installer](https://docs.astral.sh/uv/getting-started/installation/)
instead.

Clone the repo and install the baseline extras.

=== "macOS / Linux"

    ```
    git clone https://github.com/detail-embedding/detail.git
    cd detail
    uv venv --seed
    source .venv/bin/activate
    uv pip install -e ".[yolo,torch,gui,analysis,sam3]"
    ```

=== "Windows"

    Windows blocks PowerShell scripts by default, and `.venv\Scripts\Activate` is
    one. Allow them for your user account first. You do this once, and it needs no
    administrator:

    ```
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```

    `RemoteSigned` runs scripts made on this machine and still blocks unsigned
    downloaded ones. `-Scope Process -ExecutionPolicy Bypass` also works but lasts
    only for the current window, so run it after you reopen PowerShell, not before.
    Only `-Scope LocalMachine` needs **Run as administrator**.

    ```
    git clone https://github.com/detail-embedding/detail.git
    cd detail
    uv venv --seed
    .venv\Scripts\Activate
    uv pip install -e ".[yolo,torch,gui,analysis,sam3]"
    ```

| Extra | Provides |
|-------|----------|
| `yolo` | YOLO training + trained/imported-model inference (CPU/MPS/CUDA) |
| `torch` | DINOv3 embedding (gated) |
| `gui` | desktop app |
| `analysis` | UMAP / HDBSCAN / scikit-learn (pose and behavior clustering); SciPy is installed by default |
| `sam3` | SAM3 backend ([below](#sam3-backend)) |
| `dev` | pytest + black (contributors) |

DINOv3 embedding weights are gated. Set up [Hugging Face access](#hugging-face-access)
before running detection or embedding.

These docs are also served locally. Run `mkdocs serve` and open
`http://127.0.0.1:8000` for the rendered site. The doc toolchain installs with the
package (no extra needed).

## GPU and CUDA

Update the [NVIDIA driver](https://www.nvidia.com/download/index.aspx) before you
install a CUDA build of torch. The driver caps which CUDA build can run, so an old
driver limits you to an old PyTorch.

Check whether torch already sees your GPU:

```
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `True`, nothing to do. If `False` and you have an NVIDIA GPU, find
the newest CUDA your driver supports. It is `CUDA Version` in the header of:

```
nvidia-smi
```

Install the newest PyTorch build at or below that version. A build's tag is the
version without the dot, but PyTorch publishes only some versions: CUDA 13.2
gives `cu132`, and CUDA 12.7 gives `cu126` because there is no `cu127`. The
**GPU / CUDA** page in the app works this out for you, or check the
[PyTorch selector](https://pytorch.org/get-started/locally/). Replace `cu126`
with your tag:

```
uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If DETAIL is open, restart it afterwards. A running app keeps the PyTorch it
started with.

## SAM3 backend

The baseline install already includes SAM3's runtime dependencies (the `sam3`
extra). SAM3 itself isn't on PyPI, so install it from source.

!!! warning "SAM3 requires an NVIDIA GPU"
    SAM3 runs on CUDA only; there is no CPU or Apple-silicon path. On other
    hardware you can still run a trained or imported YOLO detector (the `yolo`
    extra) on CPU/MPS. Complete the
    [GPU and CUDA](#gpu-and-cuda) step before this one, so `timm` doesn't leave
    you on a torch build that mismatches your driver. If CUDA breaks, see
    [Troubleshooting: CUDA after SAM3](#troubleshooting-cuda-after-sam3).

Clone SAM3 as a sibling directory and install it from source:

=== "macOS / Linux"

    ```
    git clone https://github.com/facebookresearch/sam3 ../sam3
    uv pip install -e ../sam3 --no-deps
    ```

=== "Windows"

    ```
    git clone https://github.com/facebookresearch/sam3 ..\sam3
    uv pip install -e ..\sam3 --no-deps
    ```

    Triton has no Windows wheel on PyPI. Install the port as well:

    ```
    uv pip install triton-windows
    ```

`--no-deps` keeps SAM3's `numpy<2` pin from downgrading numpy 2. SAM3's loader
imports the legacy `pkg_resources`, so cap setuptools:

```
uv pip install "setuptools<81"
```

Verify the backend:

```
python -c "import sam3.model_builder; print('sam3 OK')"
detail model check sam3
```

SAM3 weights are gated. Set up [Hugging Face access](#hugging-face-access).

## Hugging Face access

SAM3 and DINOv3 use gated weights from Hugging Face. You need a free
[Hugging Face account](https://huggingface.co/join) and a token that can read
gated repos.

1. Create a token in your
   [Hugging Face token settings](https://huggingface.co/settings/tokens):
   choose **Fine-grained**, then enable *"Read access to contents of all public
   gated repos you can access"* (without it the download returns HTTP 403).

2. Request access on each gated model's page. Open it while signed in and click
   **Agree and access repository** (or fill in the access-request form). Approval
   is usually instant. One request covers a whole family: every DINOv3 size sits
   behind the same agreement, so the link below is the only DINOv3 one you need.

    - [SAM3](https://huggingface.co/facebook/sam3)
    - [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

3. Make the token available to the tool by any of:

    ```
    detail hf-token hf_...
    ```

    ```
    export DETAIL_HF_TOKEN=hf_...
    ```

    or `~/.config/detail/credentials.yaml`.

4. Verify access. Readiness is cache-first and only hits the network when the
   model is absent locally:

    ```
    detail model check sam3
    detail model check dinov3_vit
    ```

Both should print `reason=ok`. `no_token` / `no_access` / `bad_token` indicate
which step to revisit; the message and action URL point to the exact fix.
`detail model install <id>` downloads the weights ahead of time. The same
checks are available from Python via `Experiment.check_model_access(<id>)`.

!!! note "Each model size is a separate download"
    Each DINOv3 size is its own set of weights. `detail model install dinov3_vit`
    fetches the default size (`vit_large`), which is what an embedding run uses
    when you don't pick a model, and what the keypoint method `dinov3_match` uses.
    For another size, name it: `detail model install dinov3_vit --variant
    vit_base`. In the GUI, the **Model manager** lists one row per size under
    **Hugging Face**; select the size you plan to run and click **Install**
    (one access request covers them all). A run checks the exact size it
    was configured with, so picking `vit_base` in a run while only
    `vit_large` is installed fails with "is not downloaded".

!!! note "Windows: model cache"
    Without symlink permission, Hugging Face stores each cached weight as a full
    duplicate copy, wasteful for the multi-GB SAM3 and DINOv3 files. Enable
    **Developer Mode** (Settings → Privacy & security → For developers) or run as
    admin so it can symlink instead. Add `uv pip install hf_xet` for faster
    downloads of Xet-backed repos. To silence the symlink warning once the above
    is set, `set HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

!!! note "Disk space for video previews"
    Each computer keeps small copies of your videos for clip montages to play from, in `~/.cache/detail/previews/`: tens of megabytes per hour of video, and roughly twice that on Windows and Linux, where the bundled video library cannot write H.264. They are built in the background after an import, are never part of an experiment, and can be cleared any time from Configuration or with `detail cache clear --previews`. Building them takes a while, once per computer, so clear them only when you need the space. Updating DETAIL keeps them; an update that needs to rebuild them says so first.
    To keep them elsewhere, set `DETAIL_PREVIEW_DIR` or add `preview_cache:` under
    `storage: default:` in `~/.config/detail/storage.yaml`.

## Run

Launch the desktop app:

```
detail-gui
```

The app opens on a list of your experiments, stored under `~/Documents/DetailExperiments`
by default.

Or use the CLI:

```
detail --help
```

## Launch it again later

Already installed? Go to the repo, activate the environment, launch.

=== "macOS / Linux"

    ```
    cd path/to/detail
    source .venv/bin/activate
    detail-gui
    ```

=== "Windows"

    ```
    cd path\to\detail
    .venv\Scripts\Activate
    detail-gui
    ```

Use `detail --help` instead of `detail-gui` for the CLI.

To pick up the latest changes first, `git pull` in the repo before activating. If
it pulled anything in, reinstall so new dependencies land:

```
uv pip install -e ".[yolo,torch,gui,analysis,sam3]"
```

## Troubleshooting: CUDA after SAM3

SAM3 doesn't pin `torch` directly but pulls `timm`, which can drag in a newer
torch wheel than your NVIDIA driver supports. Symptom: a `UserWarning: CUDA
initialization: The NVIDIA driver on your system is too old`, and
`torch.cuda.is_available()` is `False`.

Diagnose:

```
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

If torch's `+cuXXX` build is newer than the `CUDA Version` in `nvidia-smi`'s
header, update the [NVIDIA driver](https://www.nvidia.com/download/index.aspx).
If you can't, reinstall torch for your driver's CUDA as in
[GPU and CUDA](#gpu-and-cuda), then restart DETAIL.

## Next

- **[Create an experiment](workflows/create-a-experiment.md)**: the first workflow, and
  the start of the step-by-step path through recordings, streams, and metadata.
- **[Concepts](concepts.md)**: the pipeline and vocabulary.
- **[GUI guide](gui.md)** · **[CLI guide](cli.md)**: drive each surface.
