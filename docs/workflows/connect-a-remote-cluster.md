# Connect a remote cluster (Tillicum) from Windows

This guide connects a Windows install of DETAIL to a Slurm cluster, so detection jobs run there
instead of on your own machine. It uses UW Tillicum as the example; the steps are the same for
any SSH-reachable Slurm cluster ([Compute concepts](../concepts.md#models-backends-and-access)).

!!! note "macOS or Linux"
    Your system's SSH client already supports what DETAIL needs here. Skip to
    [Set up SSH so Duo only asks once](#5-set-up-ssh-so-duo-only-asks-once) and run those steps
    (and everything after) directly in your normal terminal, no WSL required.

!!! warning "Every `detail` command below runs on your machine, never inside an SSH session"
    `detail remote add/test`, `detail detection run`, and `detail process attach` all run in your
    **WSL terminal**. DETAIL drives the connection to the cluster; it never runs on the cluster
    itself. A command block below headed "on Tillicum" is the one exception: plain cluster setup
    (conda, installs), not a `detail` command.

The steps, in order:

1. [Request cluster access and storage](#1-request-cluster-access-and-storage): start this first, it can take days.
2. [Install WSL](#2-install-wsl) (Windows Subsystem for Linux).
3. [Open the WSL terminal](#3-open-the-wsl-terminal).
4. [Log in once to accept the cluster's host key](#4-log-in-once-to-accept-the-clusters-host-key).
5. [Set up SSH so Duo only asks once](#5-set-up-ssh-so-duo-only-asks-once).
6. [Install DETAIL inside WSL](#6-install-detail-inside-wsl).
7. [Set up the environment on the cluster](#7-set-up-the-environment-on-the-cluster).
8. [Register the cluster in DETAIL](#8-register-the-cluster-in-detail).
9. [Run a detection job on the cluster](#9-run-a-detection-job-on-the-cluster).

## Before you start

- DETAIL is already installed on this Windows machine ([Getting started](../getting-started.md)).
- You have a UW NetID with Duo two-factor enrolled.

## 1. Request cluster access and storage

Do this before anything else below. Provisioning is a human process on UW-IT's side and can take
a few days, so the rest of this guide's software setup can happen while you wait.

If you're joining a lab or team that already uses Tillicum, **check whether storage already
exists before asking UW-IT for any**: many accounts get a shared project allocation (often
1TB+) through their research group at no extra request. You can't check this until your account
exists and you can log in (step 4), so come back to this once you have access; the check itself
is one command:

```
groups                        # lists every group you belong to
ls -la /gpfs/projects/        # one directory per group with an allocation
```

If one of your groups has a matching directory there, that's your storage: skip straight to
step 6 for anything you'd otherwise have emailed UW-IT about. Only email
[help@uw.edu](mailto:help@uw.edu) for a new allocation if nothing turns up. Ask for both of the
following in one message, so you don't lose a round trip discovering the second one is missing:

- A Tillicum HPC account for your NetID.
- A scratch or project storage allocation (state how much data you expect to move per job; a few
  hundred GB is a reasonable starting ask).

Wait for confirmation that both are active before step 4.

## 2. Install WSL

Tillicum only accepts a password + Duo push for SSH, not a key. To avoid a fresh Duo prompt on
every single command, DETAIL keeps one SSH session open and reuses it (`ControlMaster`). Windows'
built-in SSH client doesn't reliably support that. WSL's Linux SSH client does, so the remote
setup runs inside WSL.

In **Command Prompt**, running as Administrator:

```
wsl --install
```

Restart your computer when it finishes.

After restarting, WSL opens an Ubuntu window automatically and asks you to create a username and
password. This is a new Linux account, separate from your Windows login; pick anything you'll
remember. If nothing opens automatically, continue to the next step.

## 3. Open the WSL terminal

From the Start menu, open **Ubuntu**. This is your WSL terminal: every command in the rest of
this guide up through step 6 runs here, not in Command Prompt or PowerShell.

## 4. Log in once to accept the cluster's host key

In the **WSL terminal**, replacing `sagolden` with your own NetID:

```
ssh sagolden@tillicum.hyak.uw.edu
```

- The first connection asks whether to trust the host's fingerprint. Type `yes`.
- Enter your UW NetID password when prompted (not shown as you type).
- Approve the Duo push on your phone.

A successful login lands you at a shell prompt on Tillicum. This is also when you can run step
1's storage check (`groups`, `ls -la /gpfs/projects/`) if you haven't yet. Then return to WSL:

```
exit
```

## 5. Set up SSH so Duo only asks once

Create `~/.ssh/config` in the **WSL terminal** (`nano ~/.ssh/config`, or any editor you have) with:

```
Host tillicum
    HostName tillicum.hyak.uw.edu
    User sagolden
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

Replace `sagolden` with your NetID. The last two lines make ssh send a small heartbeat every
minute and give up on the connection only after ten missed ones (about ten minutes of silence).
That keeps a quiet connection from being dropped by a router or firewall that forgets idle
sessions, and it makes a connection that really has died fail quickly instead of hanging. They
do not stop a connection from dropping (a laptop going to sleep still ends it), and a short
tolerance would turn a brief network blip into a dead session, which is why the count is
generous. They are a precaution: we have not confirmed what idle or session limits Tillicum
itself enforces. Create the socket directory the config points to:

```
mkdir -p ~/.ssh/sockets
```

Connect once (you'll hit the password + Duo prompt this one time):

```
ssh tillicum echo ok
```

That connection stays open in the background until it has gone unused for `ControlPersist`'s 4
hours. Every DETAIL command that talks to Tillicum reuses it instead of prompting again, and each
use restarts the 4 hours, so a run that is checking on its job every few seconds keeps it open for
as long as the run lasts. Confirm the shared session is up:

```
ssh -O check tillicum
```

## 6. Install DETAIL inside WSL

DETAIL's remote commands need to run through the SSH session above, so DETAIL itself needs to run
inside WSL too, not just in Windows. This is a separate install from your Windows one: WSL has
its own filesystem and Python.

Check your Python version in the **WSL terminal**:

```
python3 --version
```

DETAIL needs Python 3.12. If you have it, skip to the venv step below. If not (Ubuntu on a newer
WSL image can default to a newer Python that PyTorch has no wheels for yet), install it from the
deadsnakes PPA:

```
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv
```

Create the environment and clone the repo. Use `python3.12` in place of `python3` below if you
just installed it. Do this inside WSL's own home directory (`~`), not under `/mnt/c/...`: the
Windows filesystem mounted into WSL doesn't support the file permissions a Python install needs.

```
python3 -m venv ~/detail-venv
source ~/detail-venv/bin/activate
```

Install the GitHub CLI and sign in, to clone the (private) repo:

```
sudo apt install -y gh
gh auth login --hostname github.com --git-protocol https --web
```

Follow the printed one-time code and URL to finish sign-in in a browser, then clone and install:

```
gh repo clone mlurbur/detail ~/detail
cd ~/detail
pip install -e ".[yolo,torch,gui,analysis,sam3]"
```

!!! note "Local GPU and local model install not required for remote-only use"
    This WSL environment only drives the connection to Tillicum; the actual detection job runs on
    the cluster's GPUs, not yours, and a remote job's `validate()` no longer requires the model
    to be installed locally either. If `detail doctor` reports no local CUDA device or a missing
    SAM3 install, that's fine for remote-only runs. Installing SAM3 here anyway is still useful if
    you also want to preview prompts, run local jobs, or use `detail doctor`'s full green check.

Optional: SAM3 isn't on PyPI, so install it from source (same two-step install as any machine, see
[Getting started: SAM3 backend](../getting-started.md#sam3-backend)):

```
git clone https://github.com/facebookresearch/sam3 ~/sam3
pip install -e ~/sam3 --no-deps
pip install "setuptools<81"
```

Optional: sign in for Hugging Face access and install the detection model (skip if you only
plan to submit remote jobs from this machine):

```
detail hf-token hf_...
detail model install sam3
```

Confirm the install:

```
detail doctor
```

## 7. Set up the environment on the cluster

The remote job needs its own DETAIL install on Tillicum, in a conda environment its Slurm job
activates. Everything in this step runs **on Tillicum**, not in WSL. SSH in first (this reuses
the shared session from step 5, no Duo prompt):

```
ssh tillicum
```

### Point conda at your project storage, before creating anything

Do this *before* your first `conda create`, otherwise conda silently puts environments and every
package they download under your home directory's small quota, not the project storage from
step 1, and a multi-GB install like torch can exceed it (`Disk quota exceeded`, even though the
project allocation sitting right next to it is nearly empty).

Using the project directory you found in step 1 (`<project_dir>` below, e.g.
`/gpfs/projects/goldeninference/sagolden`, a personal subfolder so you don't collide with
labmates' files in the shared group directory):

```
mkdir -p <project_dir>/conda-envs
mkdir -p <project_dir>/conda-pkgs
echo "envs_dirs:" >> ~/.condarc
echo "  - <project_dir>/conda-envs" >> ~/.condarc
echo "pkgs_dirs:" >> ~/.condarc
echo "  - <project_dir>/conda-pkgs" >> ~/.condarc
```

Confirm it took:

```
cat ~/.condarc
```

### Find the real partition and GPU names

Every cluster names its partitions differently, don't guess:

```
sinfo -o "%P %G %l"
```

Note the partition name (the one marked `*` is the default) and its GPU type. You'll need both in
step 8.

### Create the environment and install DETAIL

```
module load conda/Miniforge3-25.3.1-3
conda create -y -n detail-env python=3.12
conda activate detail-env
```

Sign in to `gh` again here (a separate machine, separate login) and install:

```
gh auth login --hostname github.com --git-protocol https --web
gh repo clone mlurbur/detail ~/detail
cd ~/detail
pip install -e ".[yolo,torch,gui,analysis,sam3]"
```

Clone and install SAM3 here too, the same two-step install as WSL (step 6): it's a completely
separate machine, so nothing from your WSL install carries over:

```
git clone https://github.com/facebookresearch/sam3 <project_dir>/sam3
pip install -e <project_dir>/sam3 --no-deps
pip install "setuptools<81"
```

Sign in for Hugging Face access and install the model here as well: the shepherd syncs your
experiment data and any *imported or distilled* detection model a run uses (see below), but never
the downloaded built-in model cache (SAM 3, DINOv3):

```
detail hf-token hf_...
detail model install sam3
```

!!! note "After today, you don't need to SSH in again just to update this"
    Once the cluster is registered (step 8), `detail remote push-credentials <name>` (run
    from WSL, on your own machine) copies your Hugging Face token here for you. Useful the
    next time you rotate the token, or add a gated model (DINOv3) you didn't set up during
    this first pass.

Confirm everything:

```
detail doctor
```

!!! note "Login nodes have no GPU"
    `CUDA (GPU): no CUDA device` here is expected: you're still on the login node. Installing and
    downloading weights is fine here; only the actual job runs on a GPU node. If you want to
    verify the torch build works before submitting a real job, request a short interactive
    session with the partition name from above:

    ```
    srun --partition=<partition> --gpus=1 --time=0:10:00 --pty bash
    conda activate detail-env
    python3 -c "import torch; print(torch.cuda.is_available())"
    exit
    ```

    `False` **on the login node** means nothing; only trust this check from inside an `srun`
    session on an actual GPU node. If it's `False` there too, see
    [Troubleshooting: CUDA after SAM3](../getting-started.md#troubleshooting-cuda-after-sam3):
    reinstall `torch`/`torchvision` from a single pinned `--index-url` (never combine
    `--index-url` and `--extra-index-url` to fix a version mismatch: pip picks the newer of the
    two regardless of which is "primary", which can silently reinstall a build your driver can't
    run).

Return to WSL:

```
exit
```

## 8. Register the cluster in DETAIL

=== "GUI"

    Launch the app from the **WSL terminal** (not Windows):

    ```
    detail-gui
    ```

    On a fresh WSL Ubuntu, install the X11 display libraries once first (the command is in
    [the GUI guide](../gui.md#wsl-display)): without them the app runs on Wayland, where
    opening a drop-down inside some dialogs has crashed it, and it says so in the terminal.

    Go to **Configuration → Remote targets → Add target** and fill in:

    | Field | Value |
    |-------|-------|
    | Name | a short id, e.g. `tillicum` |
    | Host | `tillicum` (must match the `Host` alias in `~/.ssh/config` from step 5) |
    | Remote work directory | your project directory from step 7, e.g. `<project_dir>/detail_runs` |
    | Partition | the real name from step 7's `sinfo`, e.g. `gpu-h200` |
    | Conda environment | `detail-env` |
    | Modules to load | `conda/Miniforge3-25.3.1-3` (or whatever `module load` you used in step 7) |
    | GPU type / GPUs per job | your job's requirements |
    | Walltime | leave at `24:00:00`, Tillicum's maximum (Slurm kills a job at this limit) |
    | Concurrent jobs | leave at 8 (see [Running many recordings at once](#running-many-recordings-at-once)) |

    Click **Test connection** to confirm DETAIL can reach the cluster and find `sbatch`. It also
    compares the DETAIL installed on the cluster with the one you are running here, and says if the
    cluster is behind and what to run to update it. Do this again whenever you update DETAIL on your
    own machine.

=== "CLI"

    ```
    detail remote add tillicum tillicum \
      --workdir <project_dir>/detail_runs \
      --conda-env detail-env \
      --partition gpu-h200 \
      --module conda/Miniforge3-25.3.1-3
    ```

    ```
    detail remote test tillicum
    ```

**Both `--conda-env` and `--module` are required together**: without the module, the sbatch
script tries to run `conda activate` before `conda` is even on `PATH` (`conda: command not
found`); without `--conda-env`, nothing activates the environment at all (`python: command not
found`), even with the right module loaded.

See the [GUI guide](../gui.md#remote-targets) and [CLI guide](../cli.md#remote) for every field.

## 9. Run a detection job on the cluster

First, find the recording id you want to run against:

```
detail experiment status --experiment <path/to/your/experiment>
```

=== "GUI"

    Open a detection run's launch dialog and set **Compute → Backend** to **Remote**, then pick
    your registered target. Everything else is the same as a local run.

=== "CLI"

    ```
    detail detection run --experiment <path> --recording <recording_id> \
      --backend remote --remote-target tillicum \
      --model sam3 --class mouse --prompt mouse --conf 0.1
    ```

    `--class`/`--prompt` are required for a promptable model like SAM3, swap `mouse` for
    whatever you're actually detecting.

The run streams progress the same way a local run does: `submitted detection: det_NN`, then a
live progress bar once the Slurm job starts, loads the model, and begins the frame loop. DETAIL
syncs the recording up, submits the job, polls it, and syncs the results back down when it
finishes. If your terminal returns to a prompt without streaming, reattach:

```
detail process attach det_NN --experiment <path>
```

Ctrl-C detaches the terminal only; the job keeps running on Tillicum either way.

You can keep working in the experiment while a remote job runs. When it finishes, DETAIL copies back only that job's own run folder (its results and logs), not the whole experiment, so a recording you renamed, another run that was going, or a run you deleted in the meantime is left as you left it.

Going the other way, when a job starts DETAIL sends your experiment's files to the cluster again, but a run folder the cluster already has is never overwritten: a job that is running or has just finished there owns its folder, and your computer's copy of it is older. Only run folders the cluster does not have yet (this job's own, for one) are sent. If the cluster still has a folder under this job's run name from a run you deleted earlier, that old folder is removed first, so the new run never starts inside the old one's results. This needs `rsync` on the machine DETAIL runs on, which WSL has; without it, a plain copy is used and starting a second job while another is running is not safe.

**Before the Slurm job starts, the status line names what DETAIL is doing.** It reads in turn:
uploading experiment files, uploading videos (with how many of how many files and gigabytes are
across, and an estimate of the time left), uploading the model (only for an imported or distilled
detection model, below), checking the model on the cluster, submitting the job, and waiting in the
queue. The video upload is the long one when your videos are referenced in
place rather than copied into the experiment: a library of 80 recordings was about 116 GB and took
around an hour. Labeling (and so distilling) avoids that cost: DETAIL picks the roughly 1,000
frames to label on your machine, extracts them as images, and uploads only those, a few gigabytes
instead of the whole library. The bar stays at 0 through these steps, because nothing has been
detected or labeled yet; the message and the time estimate are what move. The videos are copied once into a
cache on the cluster, so a later run skips every file already there. If a run seems stuck on one of
these steps, the same lines are in `shepherd.log` in the run's folder, with a time on each.

**Upload the videos while the distill runs.** The detection run that follows a distill reads every
recording, so it still has the whole library to send. Add `--stage-videos` to `detail distill run`
(or tick **Upload videos in the background** in the Distill dialog) and DETAIL copies them in a
separate background process while labeling and training run, and keeps going after the distill
ends. It is off by default because it can be 100+ GB on the cluster's storage. A later run that
needs the videos waits for an upload still in progress, showing its progress, instead of sending
the same files twice, and uploads whatever is missing if that upload failed or was stopped.

```bash
detail remote staging-status --experiment <path>
detail remote staging-stop --experiment <path>
detail remote staging-start tillicum --experiment <path>
```

There is no pause, but stop then start is one: finished videos are skipped and a half-sent one
continues. In the app, these are **Background uploads…** on the experiment's **Models** page
(**Stop upload** and **Resume upload**).

## Running many recordings at once

A run over many recordings (`detail process detection --backend remote` on 80 of them) submits
each recording as its own Slurm job. **By default 8 are on the cluster at once**, then the next
starts as each one finishes; set the target's **Concurrent jobs** (`remote add
--max-concurrent-jobs`) to change it. The progress bar then counts recordings (finished plus the
fractions of the ones in flight), the run page shows a row for every recording in flight, and
the CLI prints each recording's phase as it changes. The GPUs a run uses are concurrent jobs
times GPUs per job, and Slurm queues any job the cluster has no room for.

**Why 8 and not more.** The limit is the connection, not the cluster. Every job in flight keeps
ssh sessions open through your one shared connection, and Tillicum allows **10 sessions per
connection** (measured by opening twelve at once: ten ran and two were refused with `Session open
refused by peer`; the host's `sshd_config` is not readable without root, so it cannot be looked
up). Past 10, new sessions are refused. Eight jobs leave two: the background video upload takes
one, and one is left for your own commands, **Test connection** or a status check.

**Do not open many sessions yourself while a run is going.** A refused session during a poll is
read as the cluster being unreachable. A run on a current DETAIL waits and retries; one started
on a version before the fix for that (the run's own `shepherd.log` shows an old traceback) records
the recording as failed while its job carries on. That is how the twelve-session measurement above
failed a recording in a live run.

**It needs `rsync` on the machine DETAIL runs on** (WSL has it). Concurrent jobs rely on the
upload leaving a running job's folder on the cluster alone, which only rsync can do; without it
DETAIL runs one recording at a time and says so in the run's log.

**The walltime applies to each job, not the run.** A recording that needs more than the target's
walltime is killed by Slurm and recorded failed; raise the walltime (up to the cluster's limit)
rather than the job count if that is what you see.

## Profile clusters on the cluster

A **Profile clusters** job can run on the cluster too, so a night over a whole clustering run
uses the cluster's nodes instead of your computer:

```bash
detail analysis profile-all --experiment my_experiment --run an_10 --remote-target tillicum
```

(in the app: **Run on** in the Profile clusters dialog). The clustering run is sent to the
cluster first, exactly as it is on your computer; the profiles come back into the run when the
job ends.

## Troubleshooting

**The run says `cannot reach tillicum over ssh since 14:32 …`**: the connection to the cluster
dropped (the computer slept, the network changed, or the shared session ended). **The job on
Tillicum is not affected and keeps running there.** DETAIL waits, asking again with longer and
longer gaps (up to a minute), and carries on from where it was as soon as you log in again: run
`ssh tillicum echo ok` once in the WSL terminal, or use **Open SSH session** on the Remote targets
page. The progress bar keeps its last numbers and only the message changes. If the job finishes
while you are away, DETAIL waits for the connection before copying the results back, so open the
session and they arrive. It gives up after a week and says the job may still be running; the
computer has to stay on for it to wait at all (see below). Before a long run, `ssh -O check
tillicum` confirms the shared session is up.

**The run says `the ssh connection to tillicum is full …`**: you are logged in and the
connection works, but it is carrying as many commands as the cluster allows at once (ten). You
don't need to do anything: DETAIL keeps its own jobs to eight sessions between them and carries on
when one frees up. If it keeps saying this, close other terminals, file transfers or editor
windows that use the same connection.

**The run says `failed` after the computer slept or restarted**: the helper that watches the job
runs on your computer, so it stops when the computer does, and the run is then recorded as
failed even though the job on Tillicum may still be going. Check with `ssh tillicum "squeue
--me"`: the quotes matter, because `ssh tillicum "squeue -u $USER"` expands `$USER` on *your*
computer, where it is a different name from your cluster login, and lists the wrong user's jobs.
`ssh tillicum "sacct -j <job id> -o JobID,State,Elapsed,ExitCode -P"` gives one job's real state;
the job id is `remote_job_id` in the run folder's `job.json`. The results, when it finishes, are
under the target's work directory on the cluster; copy them back with `rsync`. Retrying the
failed recording submits a new job rather than picking this one up.

**A run copied back from Tillicum later says `<file> is all zero bytes`**: the computer shut down
(or WSL was stopped) shortly after the results arrived, before it had written them to disk, so
the local copy kept its file sizes and lost its contents. The copy on Tillicum is separate and is
usually still fine. Check one of the file names from the message there (`ssh tillicum "head -c 200
<remote work directory>/<experiment>/recordings/<recording>/detections/<run>/part_001.zarr/zarr.json"`
should print JSON), then copy the run folder back in the WSL terminal. **`--checksum` is needed**,
because the damaged files have the right size and date and a plain `rsync -a` would skip them, and
`--exclude job.json` keeps the run's local record:

```
rsync -a --checksum -s --exclude job.json \
  "tillicum:<remote work directory>/<experiment>/recordings/<recording>/detections/<run>/" \
  "<experiment folder>/recordings/<recording>/detections/<run>/"
```

Add `--dry-run -i` first to see what it would copy. Results are now flushed to disk before a run
is recorded as complete, so this applies to runs that finished before that change.

**`your ssh session to tillicum has expired …`** (ssh's own words for it are `Permission
denied (gssapi-keyex,gssapi-with-mic,keyboard-interactive)`, still shown below DETAIL's
message) from `detail remote test`, **Test** on the Remote targets page, `push-credentials`,
or a run submission: step 5's shared SSH session (`ControlPersist 4h`) has timed out: every
DETAIL command talks to Tillicum non-interactively, so an expired session fails fast like this
instead of prompting. A remote file's own `Permission denied` (no method list in brackets) is
a different problem and keeps its own message.
Refresh it with one real, interactive login: either run `ssh tillicum echo ok` again
(a password + Duo prompt), or click **Open SSH session** on the Remote targets page,
which opens a terminal running `ssh tillicum` for you. DETAIL never sees the password
or the Duo approval either way. Retry the command that failed once you're logged in.

**`the shared ssh connection to tillicum is not responding`**: step 5's shared connection is
still open but has stopped answering, usually after the computer slept or the network changed.
Left alone, every command waits on it forever and a run sits on "uploading experiment files"
with no error, so DETAIL checks it before each command (a check of the shared connection takes
milliseconds when it is healthy) and stops with this message after 15 seconds. The jobs already
on Tillicum are not affected. Close the stale connection and open a fresh one in the WSL
terminal, then Retry the recordings that failed:

```
pkill -f 'ssh: .*\[mux\]'
ssh tillicum true
```

The second command asks for the password and Duo again. **Open SSH session** does not fix this
one: it joins the stale connection instead of replacing it. The `ServerAliveInterval` lines in
step 5's config make a connection that has really died close on its own instead of lingering, so
check that your own `~/.ssh/config` has them (`ssh -G tillicum | grep -i serveralive`).

**`Session open refused by peer`** (inside an `ssh … exited 255` message): the shared SSH
connection is full. OpenSSH lets one connection carry ten commands at once, and a run with
several recordings in flight (`--max-concurrent-jobs`, default 8) shares one. DETAIL waits
and tries a refused command again (up to a few minutes, backing off), so you usually never see
it; if a run still ends with it, the connection stayed full for that whole time. Retry the
recordings that failed, or lower **Concurrent jobs** on the target. It is **not** the
`Permission denied` of an expired session above: that has no "refused by peer" line, fails at
once, and needs you to log in.

**A detection or keypoint model you imported or distilled** (the ones in the Model manager's
library, not built-in SAM 3 or `dinov3_match`) lives on your machine, outside the experiment, so DETAIL **uploads it for you**
before the check: it asks the cluster where its imported models live, copies the model's folder
there (a few tens of megabytes; one already there costs only a comparison), then runs the
readiness check. A model trained *inside* the experiment travels with the experiment upload. If the
model is not in this machine's library, the run fails at once with `the model … is not in this
machine's model library`; if the cluster cannot say where its models live, with `could not
find where … keeps its imported models`. Before this existed, every recording of a run failed with
`could not check model readiness … no model.yaml in …/imported/<id>`; copying the folder by hand
(`rsync -a ~/.cache/detail/models/imported/<id>/ <host>:.cache/detail/models/imported/<id>/`)
fixed that and is no longer needed. A readiness failure now shows its cause on its first line
instead of the cluster's login banner.

**`… the DETAIL installed on 'tillicum' is older than this app and has no detail.…`**: the copy
of DETAIL **on the cluster** is older than the one you are running here, so a module (or a name in
one) this version imports is missing there. It shows on the first line of every recording that
tried to start, whichever step hit it (checking the model, finding where imported models live,
registering a video folder), because each of them runs DETAIL on the cluster. Older versions showed
the same problem as `could not check model readiness … No module named 'detail.…'` under the login
banner. The cluster runs its own checkout (`~/detail` on Tillicum, usually the `remote-detail`
branch): update it with `git pull --ff-only` there, in the target's conda environment, whenever you
update DETAIL on your own machine, then run again (**Resume** takes up the recordings that failed).
A plain `No module named 'detail'` is a different problem: DETAIL is not installed in the
environment the target activates, so check the target's activation line. **Test connection** (or
`detail remote test`) compares the cluster's build with yours and prints this same update command
before you start a run, so run it after updating DETAIL here.

**`mkdir: cannot create directory '/gscratch': Permission denied`**: the remote work directory
is set to another cluster's scratch convention. Use the exact path from step 7 (on Tillicum,
typically under `/gpfs/projects/...`), not a copied example.

**The Slurm job fails immediately with `conda: command not found`**: the target's **Modules to
load** field is missing the conda module (step 7's `module load ...` line): add it with
`--module` (repeatable) or the GUI's "Modules to load" field.

**The Slurm job fails immediately with `python: command not found`**: the target's **Conda
environment** field is blank, so the submitted script never activates one. Fill it in (step 8).

**`Disk quota exceeded` installing packages on the cluster**: almost always means conda placed
the environment under your home directory instead of a project allocation. Check
`conda config --show envs_dirs pkgs_dirs`: if it lists paths under your home directory first,
your `~/.condarc` (step 7) either wasn't written or was written after the environment already
existed. Fix the `.condarc`, then `conda env remove -n <name>` and recreate it. Only ask UW-IT for
more space if you've confirmed you're actually using your project allocation and it's genuinely
full (`df -h <project_dir>`).

**A remote job fails with `ModuleNotFoundError: No module named 'sam3'` or a missing-weights
error**: SAM3 and its weights are machine-global, not part of the experiment the shepherd
syncs. Install SAM3 and the model directly on the cluster too (step 7): this is a separate,
fully independent install from WSL's.

**A remote job fails with a Hugging Face authorization/access error** (a gated model like
SAM 3 or DINOv3 that has never been installed on the cluster before): the cluster needs its
*own* Hugging Face token to check access and download weights; this machine's token
doesn't reach it on its own. Run `detail remote push-credentials <name>` from WSL, then
retry (or install the model there directly, as in step 7).

## Next

- [Concepts: Models, backends, and access](../concepts.md#models-backends-and-access): how a remote run fits the pipeline.
- [GUI guide: Remote targets](../gui.md#remote-targets) · [CLI guide: `remote`](../cli.md#remote).
