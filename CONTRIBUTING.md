# Contributing to DETAIL

Thanks for helping. This page covers setting up, running the tests, and opening a pull request.
The rules the code itself follows (schemas, storage, the hard rules) are in
[`CLAUDE.md`](CLAUDE.md), and [`DESIGN_PLAN.md`](DESIGN_PLAN.md) is the specification behind them.

## Set up

DETAIL uses [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer. Install uv, clone, and make an
environment as the [README](README.md) describes, then install the package with the extras the
tests need:

```
uv pip install -e ".[torch,gui,analysis,dev]"
```

CI tests a different set: CPU-only torch, `.[dev,gui,keypoints_export]` and a few packages from the
other extras (`.github/actions/setup/action.yml` lists them). Add `keypoints_export` to run the
SLEAP, DeepLabCut H5 and NWB export tests; without its packages they skip. A CPU-only torch is
enough; no test needs a GPU. On Linux, Qt also needs a few system libraries (`libegl1 libgl1
libxkbcommon0 libdbus-1-3` on Debian and Ubuntu).

Then install the git hooks, once per clone:

```
pre-commit install
```

This sets up one check: **at each commit**, black and a few quick checks run on the files you
are committing, in about a second.

**Nothing runs at push.** A pre-push hook used to run the whole suite, and it gated the wrong
event — a branch is pushed many times and merged once, so it charged minutes repeatedly for a
single decision. The suite belongs at merge time: run it yourself before merging, and do not
count on CI to stop a merge. It takes a few minutes:

```
pytest -q -n auto --dist loadfile
```

If `pre-commit install` refuses because `core.hooksPath` is set, you have the older hook setup.
Run `git config --unset core.hooksPath`, then install again.

## Run the tests

**The full suite:**

```
pytest -q -n auto --dist loadfile
```

`-n auto` runs the tests in parallel, one worker per CPU core, and `--dist loadfile` keeps each
file's tests on one worker, so expensive shared fixtures are built once. The suite takes minutes, longer on a machine with fewer cores or one that is also doing other work. Without `-n auto` the same tests run one at a time and take several times as long.

**While you work, run only what your change touches:**

```
pytest -q $(python scripts/select_tests.py)
```

`scripts/select_tests.py` picks the test files that match the files you changed, plus every
guard test. When it cannot narrow honestly, it prints nothing, and the command runs the whole
suite. It is for speed while you edit; run the full suite before you open a pull request.

**By tier.** Every test is marked with what it costs. The cost mark comes from the directory it
lives in, with per-file exceptions; the `guard` mark comes from a list of files. Both are in
`tests/tiers.py`:

| marker | what it means | run it with |
| --- | --- | --- |
| `unit` | in-process, no Qt, no real workload | `pytest -q -m unit` |
| `gui` | needs a Qt application | `pytest -q -m gui` |
| `slow` | starts real worker processes, runs a real pipeline, or encodes video | `pytest -q -m slow` |
| `guard` | a check on the code itself, such as a source scan or a registry sweep; also `unit` or `gui`, never `slow` | `pytest -q -m guard` |

**The GUI tests need no display.** They set `QT_QPA_PLATFORM=offscreen` themselves, so they run
over SSH and in CI as they do on a desktop.

**Warnings fail the run.** The suite runs with `filterwarnings = error`. If your change makes a
warning appear, fix its cause rather than silencing it.

## Sign the contributor license agreement

Before your first pull request can be merged, read the
[DETAIL Contributor License Agreement](CLA.md) and agree to it. You keep the copyright in what
you write. You give the project the right to distribute your work, including under a different
license in future versions: DETAIL's first release is AGPL-3.0, and later versions are planned
under a non-commercial license.

To agree, add a line to [`.github/cla-signatures.md`](.github/cla-signatures.md) in your first
pull request, in a commit you make yourself: your GitHub username, full name, the date, and the
CLA version. The **CLA** check on the pull request fails until that line is there, and names the
username it looked for. It also fails if the line is added in someone else's commit, or if the
pull request adds a line for another person or changes one already recorded: each person signs
for themselves.

If your employer or school has rights in what you write, make sure it allows you to contribute
on these terms before you sign.

## Open a pull request

1. Branch from `main`, and push the branch before your first commit, so that a later `git push`
   goes to your branch and not to `main`:

   ```
   git checkout main && git pull
   git checkout -b short-descriptive-name
   git push -u origin HEAD
   ```

2. Make the change. If it changes anything a user sees or types, update the matching page under
   `docs/` in the same pull request. The table in
   [`ship-a-change`](.claude/skills/ship-a-change/SKILL.md) §5 lists which page goes with which
   change.
3. Run `pytest -q -n auto --dist loadfile` and read the last line (the pass count), not just the
   exit code.
4. Open the pull request. Say what changed, which command you ran and its result line, and, if
   you fixed a bug, what test now fails without your fix.

A fix is expected to cover the kind of bug, not only the one instance you hit. The
[`fix-the-class`](.claude/skills/fix-the-class/SKILL.md) procedure describes how, and
[`docs/recorded_classes.md`](docs/recorded_classes.md) lists the kinds already guarded against.

## Public and private work

All work happens on feature branches and merges into `main` of the public repository,
`github.com/detail-embedding/detail`, when it is ready.

A branch pushed to the public repository is visible to everyone. A feature that must stay
secret until it is ready keeps its branch in the private repository instead. That is the
private repository's only purpose.

### Step 1: which repository is this?

Run `git remote get-url origin`.

| `origin` is | you are in |
| --- | --- |
| `github.com/detail-embedding/detail` | the **public repository** |
| anything else, and `git remote get-url upstream` is `github.com/detail-embedding/detail` | the **private repository** |
| neither | unknown: stop and ask |

### Rules in both repositories

- A feature branch holds one piece of work: one feature, one fix, or one docs change.
- Start it from `main`. Merge `main` into it regularly.
- Unfinished work stays on its feature branch until it is ready. Do not merge it early and hide
  it; `surfaces.HIDDEN` is not for work in progress.
- Never commit credentials or unpublished data.

### Rules in the public repository

- Use it for everything that can be seen while in progress, including every bug fix.
- A feature branch merges into `main` through a pull request.
- Never name work from the private repository in a commit message, pull request or issue.

### Rules in the private repository

- Use it only for a feature that must stay secret until it is ready.
- `main` here is an exact copy of the public repository's `main`. Never commit to, merge into,
  or push to it. Update it only by fast-forwarding from `upstream/main`.
- A feature branch here never merges into `main` here. When it is ready, it moves to the public
  repository (below).
- A feature that needs another secret feature starts from that feature's branch. They then move
  to the public repository in that order.
- `integration` is a scratch branch for testing several secret features together: `main` plus
  those feature branches. Rebuild it from `main` when a feature changes. Never commit to it
  directly, start a branch from it, or push it to the public repository.
- Never push to the public repository from here, except as below.

### Moving a finished feature from the private repository to the public repository

Only a maintainer does this. An agent does it only on an explicit instruction.

1. Merge `main` into the feature branch. CI must be green.
2. Give the branch a neutral name. The name becomes public.
3. Read the full diff against `upstream/main`. Anything pushed to the public repository is
   permanent.
4. Push that one branch to the public repository and open a pull request there.
5. A second maintainer approves the pull request.
6. After it merges: fast-forward `main` in the private repository, delete the feature branch
   there, and rebuild `integration`.

## Report a bug

Use the bug report form under **Issues**. It asks for the output of `detail doctor`, so run that
first. [`docs/bug_reports.md`](docs/bug_reports.md) explains what to capture before it is lost,
starting with the terminal output. Filing from a terminal, where `gh` cannot fill in the form,
use [`.github/bug_report_body.md`](.github/bug_report_body.md) instead.

Questions go to **Discussions**, not Issues. Report a security problem privately, through
**Security ▸ Report a vulnerability**, never in a public issue.
