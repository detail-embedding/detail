---
name: file-a-bug-report
description: Capture the evidence before it expires, then write a report a fixer who cannot reach the machine can act on. Use when filing a bug report, triaging one from another machine, or when a user describes a failure you cannot reproduce.
---

# Filing a bug report

**A report from a machine the fixer cannot reach is the only record of that machine's state.**
Observations are load-bearing and unrepeatable; inferences are disposable, because the fixer can
re-check every claim about our code and can never re-observe the box.

**Collect ruthlessly, conclude tentatively.**

`docs/bug_reports.md` is the full spec and has a worked example. `gh issue create --template
"Bug report"` is the form: `--template` takes the `name:` in the front matter of
`.github/ISSUE_TEMPLATE/bug_report.md`, not the file name.

## 1. Capture, before you write a sentence

Some of it is expiring while you type. **Save the terminal scrollback to a file first, before
anything else, before closing any window** — any output the tool does not persist lives in one window
and dies with it.

Then, what the run's own folder **does** keep — attach each that exists:

- **`activity.csv`** — a timestamped line per update: recording, stage, progress, both ETAs. **This is
  the one that separates *slow* from *stuck*:** if its last line is two hours old, the run stopped
  reporting then, and that timestamp is the first fact a fixer needs.
- **`worker.log`** — the stage backend's own stdout and stderr, including the line it stopped at.
  `executor/local.py` redirects both into it via `rundir.open_log`.
- **`coordinator.log`** — the same for a multi-recording run's coordinator. A single-recording run has
  none; it still has the other two.

Then, while the failure is still on screen:

- **The process tree**, with pids, parentage and start times
- **Two CPU-time samples, several seconds apart, with timestamps.** One sample is an impression; two
  are a measurement. "0% CPU" is not a finding — "8490.65s → 8490.65s across 5s" is
- **Memory, per process.** A working set unchanged *to the byte* across samples is evidence; "seems
  idle" is not
- **`nvidia-smi`** — VRAM, utilization, temperature, power — for anything touching a GPU
- **File mtimes and last-known-good state**: the last row of a results file, the last artifact written
- **The OS event log** — sleep, a driver reset, a thermal event
- **`detail doctor` verbatim, in a fresh subprocess.** Do not summarize it

**Ask the human while they are standing there.** *"Were you at the machine at 07:00?"* is free now and
unrecoverable a day later.

## 2. Falsify your own mechanism first

The repo is on the reporting machine too. One report's *"accumulates over epoch boundaries"* was one
file read from false, and the fixer chased it, because a mechanism is the checkable part of a
hypothesis.

- Cite `file:line`
- **Pin the third-party version you read** — the fixer's venv is not yours
- Report the config the run **resolved** (`run/args.yaml`, `job.json`), not the one it was given

## 3. Name the discriminator

Before filing: say what you ruled out and how, then name the **discriminator** — the one cheap command
that separates the hypotheses still standing.

**It is the highest-value line in the report and the one most often missing.** The GPU-hang report had
none, and sleep / a GPU driver reset / a blocked console all fit every observation it contained.

## 4. Two rules that are easy to skip

**One failure per issue, cross-linked.** The report that motivated this spec did it, and it is why it
was actionable.

**A missing log is its own bug — file it too.** If diagnosis needed a process table because the output
that would have answered it was never written anywhere, that is a class and it is ours.

## 5. Platform

CI is Linux and the dev box is macOS, so **a Windows report is often the only signal that platform
sends** — an unfixable one costs a platform's worth of coverage, not one bug. Mark the OS, the Python,
and whether the workspace is on a network share; several recorded classes are invisible on a local
disk and immediate over SMB.

## 6. Mark observation and inference separately

The fixer cannot tell from the page which kind they are reading, and will spend a round trip finding
out. Say "I observed" and "I think", and keep them in different sentences.
