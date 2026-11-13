---
name: run-tests
description: Run tests so the person watching sees their progress and result — a live bar on the status line in a terminal, bars posted in the conversation when they may be watching from the Claude app, and the result posted the moment a run ends. Use whenever you run pytest, from one file to the full suite.
---

# Running tests so they can be watched

A foreground `pytest` shows nothing until it ends, and a suite that takes minutes looks exactly like a
hang. This applies to **every** test run, not only the full suite: a one-minute selected run is
just as opaque, and every run's result has to reach the person before you act on it. A
PreToolUse hook (`scripts/pytest_reminder_hook.py`) reminds you of this skill on any Bash command
that runs pytest; it never blocks anything. Every run already records its progress: `scripts/suite_progress_plugin.py` is on every run
through `addopts`, so there is nothing to add to the command. What this skill adds is getting
that progress **to the person**, who may be in the Claude app or in a terminal.

| where they watch | what they see | how it gets there |
|---|---|---|
| **terminal** | a live bar under the prompt, redrawn every second, and your result message | `statusLine` in `.claude/settings.json`, nothing per run |
| **Claude app** | a bar message from you at each 10%, each failure, and the result | a Monitor on the run; **you post each line** |

The status line is the better display — it updates every second — so in a terminal you do not
post progress: the Monitor's raw lines are already on screen there, and a repost of each is a
duplicate. The app draws no status line and shows no Monitor lines, so **there the messages you
post are the only progress the person sees.**

**Which one.** Nothing states where the person is watching. Remote Control is what lets a session
be watched from the app, and while it is connected Claude Code sets
`CLAUDE_CODE_BRIDGE_SESSION_ID` (observed, not documented — check it each time rather than trusting
it forever):

```
echo "${CLAUDE_CODE_BRIDGE_SESSION_ID:+app}"
```

`app` → do steps 2 and 3 below. Empty → terminal only: skip them, and go from step 1 to step 4.
If the person tells you where they are watching, that overrides this.

## The procedure

**Every run, whatever its size: post its result as its own message, first, before any
analysis.** The Monitor's final line (`✓ 7,736 passed in 4m04s`, `✗ 4 failed, 7,704 passed in
4m37s`) or pytest's own summary line — then pytest's "N passed" line, then investigate. A run
that goes straight from progress (or from silence) to a diagnosis leaves the person not knowing
it finished.

Then pick by how long the run will take:

**A quick run** (one file, a few tests — seconds): run it in the foreground. Nothing else to do;
in a terminal the status line shows it anyway.

**Anything longer** (a selected run with guards, a directory, the full suite):

1. **Start it in the background** (`run_in_background: true`, a timeout that covers the run).
   On a shared machine a full suite also goes through the suite lock (`ship-a-change` §3):

   ```
   python scripts/with_suite_lock.py --who <you> -- pytest -q -n auto --dist loadfile
   ```

2. **(app) Immediately arm a Monitor** on it, from the same checkout, with the maximum
   `timeout_ms`:

   ```
   python scripts/suite_progress.py watch
   ```

   It follows the run started in this checkout in the last 10 seconds (pass `--since <epoch>` if
   the watch starts later than that), prints `waiting for pytest to start` while someone else
   holds the lock, then a line per 10% — **at most one every 15 seconds** (`--every`), so a run of
   a few seconds posts only its failures and result — a `✗ <test id>` line for each failure as it
   happens, and the result, and exits, so the Monitor ends with the run.

3. **(app) Post every Monitor line as a one-line message**, the bar in inline code, adding
   nothing:

   `pytest [████████░░░░░░░░░░░░] 40% 3,106/7,736 · ETA 1m33s · no failures`

   A `✗` line: post the test id, and say you will look at it when the run ends — or start
   reading now, if it is in code you are working on.

4. **Post the result** as above, when the Monitor's final line or the background task's
   completion arrives.

5. **Do not poll** with sleeps or repeated reads. The Monitor's lines and the task's completion
   notice are the whole signal.

Too many messages is its own failure: leave `--step` at 10 and `--every` at 15 unless the person
asks for more.

## A crashed or stalled worker

`--max-worker-restart=0` is in `addopts`: `[gwN] node down: Not properly terminated` is a red run
naming the test it crashed on, not something to wait out. Under xdist's default the replacement
re-ran the crashed file, and gates stalled holding the machine lock. `scripts/suite_watchdog`
also ends a run that makes no progress for `suite_stall_timeout` seconds, printing what it was
waiting on. A crash in `tests/gui` is usually a Qt object freed while in use: every GUI worker
is rooted until it finishes, a GUI test that leaves one running fails at teardown, and each
test's orphaned windows are collected at its teardown. `tests/test_xdist_crash_policy.py` has the
measurement.

## If nothing shows

- **The watcher waits forever:** it matches the run by checkout root and start time. A run in
  another worktree needs `--root <that worktree>`; a watch started late needs `--since`.
- **No status line in a terminal:** it runs through `scripts/run_python.sh`, which uses the
  first Python that works (the repo venv, then `python3`, `python`, `py -3`) and is silent when
  none does. So a blank line usually means no Python on the path, or a personal `statusLine` in
  `.claude/settings.local.json` overriding the repo's. On Windows it needs Git Bash, which Claude
  Code uses for Bash when installed; without it there is no Bash tool and the hook never fires.

## The run log: which tests fail, and how often

Every finished run is also appended, one line each, to `.git/detail-test-runs.jsonl` in the
clone, shared by all its worktrees (`scripts/flaky_report.py` says why it lives there). It
records every failed test id with the load average and the number of other pytest runs going
when it failed. Nothing to do per run.

When a test fails and passes on a re-run, do not guess at a cause from one sighting — read how
often it has failed before:

```
python scripts/flaky_report.py                      # this clone
python scripts/flaky_report.py a.jsonl b.jsonl      # several machines' logs, combined
```

`failed / runs` counts only runs that selected the test, and is a lower bound (see the module).
