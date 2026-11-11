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

That is the set CI tests against. A CPU-only torch is enough; no test needs a GPU. On Linux, Qt
also needs a few system libraries (`libegl1 libgl1 libxkbcommon0 libdbus-1-3` on Debian and
Ubuntu).

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
pytest -q -n auto
```

`-n auto` runs the tests in parallel, one worker per CPU core. The suite takes minutes, longer on a machine with fewer cores or one that is also doing other work. Without `-n auto` the same tests run one at a time and take several times as long.

**While you work, run only what your change touches:**

```
pytest -q $(python scripts/select_tests.py)
```

`scripts/select_tests.py` picks the test files that match the files you changed, plus every
guard test. When it cannot narrow honestly, it prints nothing, and the command runs the whole
suite. It is for speed while you edit; run the full suite before you open a pull request.

**By tier.** Every test is marked with what it costs. The mark comes from the directory it lives
in (`tests/tiers.py` holds the table):

| marker | what it means | run it with |
| --- | --- | --- |
| `unit` | in-process, no Qt, no real workload | `pytest -q -m unit` |
| `gui` | needs a Qt application | `pytest -q -m gui` |
| `slow` | starts real worker processes, runs a real pipeline, or encodes video | `pytest -q -m slow` |
| `guard` | a check on the code itself, such as a source scan or a registry sweep; can be any of the three above | `pytest -q -m guard` |

**The GUI tests need no display.** They set `QT_QPA_PLATFORM=offscreen` themselves, so they run
over SSH and in CI as they do on a desktop.

**Warnings fail the run.** The suite runs with `filterwarnings = error`. If your change makes a
warning appear, fix its cause rather than silencing it.

## Open a pull request

1. Branch from `main`, and push the branch before your first commit, so that a later `git push`
   goes to your branch and not to `main`:

   ```
   git checkout main && git pull
   git checkout -b short-descriptive-name
   git push -u origin HEAD
   ```

2. Make the change. If it changes anything a user sees or types, update the matching page under
   `docs/` in the same pull request. `CLAUDE.md` lists which page goes with which change.
3. Run `pytest -q -n auto` and read the last line (the pass count), not just the exit code.
4. Open the pull request. Say what changed, which command you ran and its result line, and, if
   you fixed a bug, what test now fails without your fix.

A fix is expected to cover the kind of bug, not only the one instance you hit. The
[`fix-the-class`](.claude/skills/fix-the-class/SKILL.md) procedure describes how, and
[`docs/recorded_classes.md`](docs/recorded_classes.md) lists the kinds already guarded against.

## Report a bug

Use the bug report form under **Issues**. [`docs/bug_reports.md`](docs/bug_reports.md) explains
what to capture before it is lost, starting with the terminal output.
