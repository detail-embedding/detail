# DETAIL

> **Under active development.** Please submit bugs and issues on the [issue page](https://github.com/detail-embedding/detail/issues).

A species-agnostic tool for behavioral analysis. Powered by open-source segmentation, tracking, and image embedding models.

**Documentation.** The docs site is the authoritative resource.

- [User guides](docs/user-guides/index.md): step by step, from installing DETAIL to analyzed behavior, with every screen explained (start here)
- [Getting started](docs/getting-started.md): install, GPU/CUDA, SAM3, and Hugging Face setup
- [Concepts](docs/concepts.md): tool design and data organization
- [CLI guide](docs/cli.md): headless operation via CLI
- [GUI guide](docs/gui.md): desktop app (Experiments, Configuration, Model manager, Quick tools)
- [Tools](docs/tools.md): one-off runs like Quick segment, no experiment needed

## Quick start

Requires Python ≥ 3.12. We recommend a virtual environment ([uv](https://docs.astral.sh/uv/) in these examples).

**Windows:** install uv, then close and reopen PowerShell so it finds `uv`.

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone DETAIL
```
git clone https://github.com/detail-embedding/detail.git
cd detail
uv venv --seed
```

Activate the environment.

**macOS / Linux**

```
source .venv/bin/activate
```

**Windows**

Windows blocks PowerShell scripts by default, and activation is a script. Allow them for your user account. You do this once, and it needs no administrator:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

```
.venv\Scripts\Activate
```

`-Scope Process -ExecutionPolicy Bypass` also works but lasts only for that PowerShell window. Only `-Scope LocalMachine` needs **Run as administrator**.

If you have an NVIDIA GPU, update the [NVIDIA driver](https://www.nvidia.com/download/index.aspx) first. The driver caps which CUDA build of PyTorch can run ([GPU and CUDA](docs/getting-started.md#gpu-and-cuda)).

Then install:

```
uv pip install -e ".[yolo,torch,gui,analysis,sam3]"
```

The `yolo` extra installs Ultralytics 8.4.x (`ultralytics>=8.4,<8.5`), the series the detector and keypoint trainers are tested against. Each trained model records the exact version it used.

Launch the desktop app:

```
detail-gui
```

Or use the CLI:

```
detail --help
```

## Launching it again later

Already installed? Go to the repo, activate the environment, launch.

**macOS / Linux**

```
cd path/to/detail
source .venv/bin/activate
detail-gui
```

**Windows**

```
cd path\to\detail
.venv\Scripts\Activate
detail-gui
```

To pick up the latest changes, `git pull` before activating, then reinstall with the `uv pip install -e` command above. `detail version --check` (or **Version History** in the GUI) tells you how far behind your copy is.

Browse the full docs locally as a rendered site (the tooling installs with the package):

```
mkdocs serve   # then open http://127.0.0.1:8000
```

The SAM3 backend, GPU/CUDA builds, and Hugging Face gated-model access have their own steps. See [Getting started](docs/getting-started.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setting up, running the tests, and opening a pull
request. In short:

- Install with `uv pip install -e ".[torch,gui,analysis,dev]"`, then run `pre-commit install`
  once. Commits then get formatted and checked.
- Work on a feature branch and open a pull request. Don't commit to `main` directly.
- Run `pytest -q -n auto` before opening a pull request.
