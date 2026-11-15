---
name: ship-a-change
description: Take a change from a branch to a merged PR — branching, the upstream trap, which tests to run when, the docs that must move with it, and what the PR body has to state. Use when starting a branch, running tests before a push, opening a PR, or deciding whether a change is ready to merge.
---

# Shipping a change

## 1. Branch, and fix the upstream immediately

```
git checkout main && git pull
git checkout -b descriptive-branch-name
git push -u origin HEAD
```

**Run `git push -u origin HEAD` before any commits.** Creating a branch off `origin/main` sets its
upstream to `origin/main`, so a later bare `git push` can go **to main directly**. Verify:

```
git config branch.<branchname>.merge   # must show refs/heads/<branchname>
```

Short, lowercase, hyphen-separated names (`add-csv-importer`, `fix-zarr-chunk-size`).

**In a worktree, branch off `origin/main` explicitly** (`git checkout -b name origin/main`) — `main`
is checked out elsewhere. And **fetch first**: `origin/main` as a local ref is only as fresh as your
last fetch, so branching right after merging another PR silently gives you the previous state. That
happened in this repo's own doc-debt series, and the symptom was a whole merged PR missing from the
new branch.

## 2. Check what moved under you

**At session start, and before each push:**

```
git fetch origin main && git log --oneline HEAD..origin/main
```

A branch open across ten upstream commits will meet a rework of its own module at merge time, when
the cost of understanding it is highest — and the conflict will look textual while being a design
decision. If a commit there touches a file you are editing, read it before continuing.

## 3. Run the right tests at the right time

```
pytest -q $(python scripts/select_tests.py)                        # the edit loop: this change + every guard
pytest -q $(python scripts/select_tests.py --no-guards)            # tighter, one file
pytest -q $(python scripts/select_tests.py --against origin/main)  # before pushing
pytest -q -n auto --dist loadfile                                  # the gate (through the lock on a shared machine)
```

`select_tests.py` prints nothing when the honest answer is "everything", so each line degrades to a
full run by itself. `--explain` shows which changed files it read.

**It is a convenience, not a gate.** It may narrow wrongly. Never cite a selected run as evidence a
change is green — **say which command you ran.**

**On a shared machine the full run goes through the machine lock:** `python scripts/with_suite_lock.py --who <you> -- pytest -q -n auto --dist loadfile`. Several agents may share one machine, and a full suite takes every core for minutes. Two at once, or one beside someone's timing run, cost real time: a benchmark thrown away, and a tier's timing published several times too high. Coordinating by message worked with two sessions and failed with three. `scripts/with_suite_lock.py` takes one lock per machine, and if someone else holds it, waits and prints who and since when. `--who` is the name others see; use your session's. Selected runs don't need it: they're small, and two side by side cost nothing. The lock is `/tmp/detail-suite.lock` on macOS, Linux and WSL and `%ProgramData%\detail-suite.lock` on Windows, the same for every checkout and branch (`tests/scripts/test_machine_dir.py` pins it). **A WSL run and a Windows run do not queue behind each other**: file locks don't cross that boundary both ways, so before a gate check the other side's `detail-suite.lock.holder` by hand.

**Run the full suite the `run-tests` way** — in the background, with its progress on the status line in a terminal and posted to the conversation when the person may be watching from the Claude app — not minutes of silence.

**Pass `-n auto --dist loadfile` for the full run, and give it an explicit 10-minute budget.**
The suite is single-threaded without `-n`, and `--dist loadfile` keeps a file's tests on one worker
so the expensive pipeline-fit fixtures are built once rather than once per worker. A default two-minute command budget kills it part-way, which looks
exactly like a hang. It also stops after 10 failures (`--maxfail=10`); `--maxfail=0` lifts that
when you are triaging a wide failure.

**Say which box a timing came from — they do not travel.** A memory-bound machine slows down when concurrent runs fill its memory, and a run that is quick alone can take several times as long; a CPU-bound machine is limited by its cores, and serial is several times slower than `-n auto --dist loadfile`. Either way, run what the change touches while you work, and run everything once, at the end.

## 4. Before opening the PR

1. `git fetch origin main && git merge origin/main`
2. `black detail/ tests/`
3. `pytest -q -n auto --dist loadfile` in full, and **read the "N passed" line** — not the exit code, not a `tail` chained to
   a merge command
4. Update user docs if the change touched the CLI, GUI, public API, install/extras or the pipeline
   model — see below
5. `git push origin HEAD && gh pr create --fill` — `--fill` takes the title from your commits or
   branch name; make it the user-facing summary (§6)

### Right before merging: verify against the `main` you are merging into

**The full local run is the gate.** Do not count on CI to stop a merge, and there is **deliberately no pre-push hook** — one runs the suite on every push, which gates the wrong event: a branch is pushed many times (on creation, per commit, after merging `main` in) and merged once. The suite belongs at merge, so these four steps are the gate:

1. `git fetch origin main && git log --oneline HEAD..origin/main`
2. **Non-empty? Merge `main` in, run the gate again, then check again.** Do not skip this because the
   file lists do not overlap — both breaks described below had no textual conflict.
3. Run the gate — `pytest -q -n auto --dist loadfile`, through the lock on a shared machine (§3) — and
   **read the "N passed" line** — not the exit code, and not a `tail` piped into a merge command.
4. Put that command and its result line in the PR body. Where a PR carries no checks, that is the
   only evidence the branch was tested, and a reviewer has nothing else to go on.

If `main` moves again while step 3 runs, re-verify what the incoming commits can *reach* rather than
re-running everything forever — `main` moving faster than the gate runs is normal with several
sessions, and step 2 taken literally never terminates.

A branch that was green when you tested it is not green *now* if `main` moved since. Two green
branches can merge into a red `main` with no textual conflict: git sees no overlap, and the break is
in what one branch assumes about the other. It happened twice in one day:

- Two PRs landed tests that called a helper a third PR had just turned into a fixture.
- One PR renamed a clustering parameter, `fit_rows` to `fit_share`, minutes before another merged a
  fixture still passing `fit_rows`. Every test using that fixture then errored at setup until a
  follow-up PR fixed it.

Nothing else can be counted on to run the suite on the combined result. So, immediately
before `gh pr merge`:

1. `git fetch origin && git log --oneline HEAD..origin/main`. If this prints nothing, the run you
   already have is against this `main`; merge.
2. Otherwise `git merge origin/main`, run `pytest -q -n auto --dist loadfile` again, read the
   "N passed" line, push, and go back to step 1. `main` can move while the suite runs.

The suite takes minutes, which is cheap next to a red `main` that every other open branch
then inherits.

`gh pr merge --delete-branch` silently checks out main, so verify the current branch before the next
push.

## 5. The docs that move with the change

This tool is under active development, so a change that alters anything a user sees or types updates
the doc **in the same change**:

| changed | update |
|---|---|
| a CLI command, option or default (`detail/cli.py`) | `docs/cli.md`, and the README quick-start if its example changed |
| a GUI screen, flow or launch step (`detail/gui/`) | `docs/gui.md` |
| the public API a user scripts against (`detail/api/`) | the guide that shows it |
| install, extras or backend requirements (`pyproject.toml`) | README install |
| a new stage / tracker / model / importer, or the pipeline mental model or on-disk layout | `docs/concepts.md` |

Ground examples in commands and screens that **exist today**; mark not-yet-built features *planned*
rather than documenting them as working. Voice and authoring rules are `docs/docs_design.md` §5 —
read it before writing user docs. Adding a page is the `add-a-docs-page` skill.

**If a change makes a doc wrong, fixing the doc is part of that change, not a follow-up.**

Surface parity — a capability reaching the API, CLI *and* GUI — is the `wire-every-surface` skill.

## 6. What the PR body has to state

There is no PR template, so write a body that answers:

- **what changed**, and what class of bug it fixes — `fix-the-class` for the shape of that
- **which command you ran** and its actual result line
- **what the guard is**, and that you watched it fail — `guard-that-can-fail`
- **where the fix stops**: every swept site marked fixed / deliberately-not / hidden, with reasons
- **for a GUI change, the launch command and what to click** — `gui-surface-change` §6, the item
  most often left out

### The title is what Version History shows — when the PR is squash-merged

Nothing is appended for Version History, and no version is bumped. The GUI panel and
`detail version --history` read the first-parent log of `main` (`core/buildinfo.py::history`): one
line per merge, and each line is that commit's **subject**. A squash merge (`gh pr merge --squash`)
makes the subject the PR title plus `(#N)`, so the title *is* the one-line summary a user reads: say
what the change does for them, not how it was built. GitHub's default merge commit makes the subject
`Merge pull request #N from owner/branch` and puts the title on its third line, where the panel does
not look, so users read the branch name instead.
Pick the merge style knowing which line ships.

## 7. The gates, and when there are none

- **CI** (`.github/workflows/ci.yml`) runs `black --check` and the suite as four tier jobs — `guard` and `unit` always, `gui` and `slow` when the PR touches source they cover; `full` does **not** run on `pull_request`, so a failure only it sees is invisible to every PR.
- **Only `.github/workflows/docs.yml` runs `mkdocs build --strict`, so when CI is not running nothing does.**
  The suite covers *part* of what strict catches and not the rest, so know which half you
  are missing: `tests/test_docs_nav.py` enforces the nav/`exclude_docs` split structurally, and
  `test_docs_access.py` does build the site — but through `docs_access.ensure_site_built`, which runs
  `mkdocs build` **without** `--strict`, so a warning that strict would fail on passes. Run it by
  hand before merging anything under `docs/` or `mkdocs.yml` — and **in a worktree you must set
  `PYTHONPATH`**:

  ```
  PYTHONPATH=$PWD mkdocs build --strict
  ```

  **Without it, a worktree's build is green and wrong.** `mkdocs` is a console script, so cwd is not
  on `sys.path` — `import detail` falls through to the editable install's finder, which resolves to
  the **main checkout**, whatever branch that happens to have out. The `mkdocs-gen-files` scripts
  render the method and module reference pages from those imported config models, so the build
  validates someone else's tree: a config field you just added is simply absent from the generated
  page and nothing complains. Measured — a worktree build came back green while missing
  the field that was the whole point of the change. `pytest` escapes this via `pythonpath = ["."]`
  in `pyproject.toml`; `mkdocs` has no equivalent. From the main checkout the plain command is
  correct, which is what makes the worktree case easy to miss.
- **`pre-commit install`, once per clone**, installs the *commit* stage only: black and a few quick
  checks over staged files, about a second. It writes the shared git dir, so it covers every worktree
  of the clone, and git deliberately ignores repository-supplied hooks, so a fresh clone still needs
  that command — `tests/test_pre_commit_config.py` fails when it is missing. If it refuses because
  `core.hooksPath` is set from the older arrangement, `git config --unset core.hooksPath` first.
- **No hook runs the suite.** `tests/test_pre_commit_config.py` fails if a pre-push stage reappears in the config. The four steps above are the gate.
- **Check which gate is actually live.** When there is no CI and no hook, running the suite by hand *is* the gate, and the PR body should say so.
- A local run on macOS does **not** substitute for CI's Linux + CPU-only coverage.
