# DETAIL — Bug Reports From Another Machine

`docs/bug_reports.md` is the authoritative specification for **how a problem observed on one
machine is written down so it can be fixed from another**. It stands alongside
`DESIGN_PLAN.md` (schemas and pipeline behavior) and `docs/docs_design.md` (the documentation
system) as an agent-facing source of truth: dense and rationale-heavy, written for the Claude
agents and human contributors who *file and diagnose* reports, not for end users. Read it
before filing an issue from a machine the fixer cannot reach.

## Sections

1. Purpose, scope & relationship to other docs
2. The asymmetry every rule follows from
3. Capture before you write
4. Writing the report
5. Ruling out, and the discriminator
6. Worked example — #279 / #280
7. Decision log

---

## 1. Purpose, scope & relationship to other docs

**What this doc is.** The rules for filing a report the primary checkout can act on, when the
failure happened somewhere the fixer has no shell. CI is Linux-only and the dev box is macOS
(`CLAUDE.md`, *CI runs Linux + CPU*), so a report from a Windows box is frequently the **only
signal that platform ever sends** — and Windows is untested by CI besides (`CLAUDE.md`,
*Windows is untested by CI*). A report that arrives unfixable costs a platform's worth of
coverage, not one bug.

**What this doc is not.** It does not govern how a *user* writes an issue. Users are served by
`.github/ISSUE_TEMPLATE/bug_report.md`, which GitHub puts in front of them; nothing here
should ever be quoted at someone reporting a problem. It also does not govern *fixing* — that
is `CLAUDE.md`'s "Fixing a bug means fixing its class", which begins where a good report ends.

**Relationship to `CLAUDE.md`.** This doc is the single full home for these rules, so they
cannot diverge between two files. `CLAUDE.md`'s *Filing a bug report* section keeps only the
operative checklist and a pointer here. When the two disagree, **this doc wins** and
`CLAUDE.md` is trimmed to match.

**Relationship to `docs/known-issues.md`.** That page is user-facing: symptom → when it bites
→ workaround, for problems we know about and have not fixed. A report that resolves into a
durable limitation earns an entry there. The two are not substitutes — one is evidence, the
other is advice.

---

## 2. The asymmetry every rule follows from

A report from a machine the fixer cannot reach **is the only record of that machine's state**.

That produces an asymmetry which decides everything else in this doc:

- **Observations are load-bearing and unrepeatable.** A process tree, a CPU-time sample, a
  file mtime, a scrollback buffer. The fixer can never go back and collect these. If they are
  not in the report, they are gone.
- **Inferences are disposable.** A root-cause hypothesis, a mechanism, a reading of our
  source. The fixer can re-derive every one of these, and *will*, because they have the repo
  and a debugger and you have neither their context nor their afternoon.

So the report's value is concentrated almost entirely in its observations, and a report that
spends its effort on a confident diagnosis while letting an artifact expire has optimized the
wrong half. Write for the asymmetry: **collect ruthlessly, conclude tentatively.**

The corollary is that inferences are not merely optional — an inference stated with the
confidence of an observation is actively harmful, because the fixer cannot tell from the page
which kind it is reading and will spend a round trip finding out. Mark them.

---

## 3. Capture before you write

**Do this before composing a single sentence.** Some of it is expiring while you type.

**The rule: any output the tool does not persist lives in one window and dies with it.** Save
the whole scrollback to a file first, before any other step, before closing anything.

What the run *does* keep, in its own folder — attach each that exists:

- `activity.csv` — a timestamped line per update: recording, stage, progress, and both ETAs.
  This is the one that separates *slow* from *stuck*: if its last line is two hours old, the run
  stopped reporting then, and that timestamp is the first fact a fixer needs.
- `worker.log` — the stage backend's own stdout and stderr, including the line it stopped at.
- `coordinator.log` — the same for a multi-recording run's coordinator.

**This section used to say the tool "discards all worker stdout/stderr".** It does not, and has
not for some time — `executor/local.py` redirects both into those files. The instruction that
followed sent reporters to their terminal scrollback for something already on disk, and would
have had them close a window holding the only copy of the rest.

Then, while the failure is still on screen:

- **The process tree**, with pids, parentage, and start times.
- **Two CPU-time samples, several seconds apart, with timestamps.** One sample is an
  impression; two are a measurement. "0% CPU" is not a finding — "8490.65s → 8490.65s across
  5s" is.
- **Memory**, per process. A working set unchanged *to the byte* across samples is evidence;
  "seems idle" is not.
- **`nvidia-smi`** — VRAM, utilization, temperature, power — for anything touching a GPU.
- **File mtimes and last-known-good state**: the last row of a results file, the last
  checkpoint written, the run's `job.json`.
- **The OS event log**, for a failure with no exception. On Windows this is where sleep/wake
  (Kernel-Power 42 / 107), power transitions (Power-Troubleshooter 1) and display-driver
  resets (4101) are recorded, and it is routinely the only place a hardware-level cause is
  visible at all.

### Collection commands

**Windows (PowerShell).**

```powershell
Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Select-Object ProcessId,ParentProcessId,CreationDate,WorkingSetSize,CommandLine | Format-List
Get-Process -Id <pid> | Select-Object Id,CPU,WorkingSet64,StartTime   # run twice, ~5s apart
nvidia-smi
Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddDays(-3)} | Where-Object { $_.ProviderName -match 'Kernel-Power|Power-Troubleshooter' -or $_.Id -eq 4101 } | Sort-Object TimeCreated | Select-Object TimeCreated,ProviderName,Id | Format-Table -AutoSize
powercfg /a
```

**macOS / Linux.**

```sh
ps -ww -eo pid,ppid,lstart,rss,etime,time,command | grep -i python
ps -o pid,time,rss -p <pid>   # run twice, ~5s apart
nvidia-smi
log show --last 3h --predicate 'eventMessage CONTAINS "sleep"'   # macOS
journalctl --since '-3h' -p warning                              # Linux
```

**Every platform.** `detail doctor` — the machine's install state, run in a fresh subprocess
so it reflects the current install rather than the app's memory of it (`detail/cli.py`, and
`docs/cli.md`). Paste it verbatim; do not summarize it.

### Ask the human while they are standing there

A machine-local agent has something the fixer will never have again: **a person next to the
machine.** "Were you logged in around 07:00?", "was the terminal window still open?", "did you
kill it with the app or with `taskkill`?" are free to ask now and unrecoverable a day later.
Ask them, and record the answers in the report as answers — not as assumptions.

---

## 4. Writing the report

Use `.github/ISSUE_TEMPLATE/bug_report.md`. Its headings are the required sections;
`gh issue create --template bug_report.md` gets them from an agent's side.

**Separate observation from inference, visibly.** Evidence goes in a section of measurements
with units and timestamps. The hypothesis goes in its own section, and says in its own words
whether it was confirmed and how. "Root cause (hypothesis — not confirmed via debugger
attach)" is exactly right; a heading that reads "Root cause" over an unverified guess is not.

**Report the config the run resolved, not the one it was given.** A stage's effective
parameters are frequently not the ones the user set — defaults fill in, a backend applies its
own, `validate()` writes values back. Attach `run/args.yaml` and `job.json`, not the numbers
from the launch form.

**Falsify your own mechanism before filing.** The repo is on the reporting machine too. If the
hypothesis names a mechanism in our code or a dependency's — "it deadlocks at epoch
boundaries", "the cache is re-read per frame" — go read that code and confirm the mechanism
exists before asserting it. A conclusion can survive a wrong mechanism, but the fixer cannot
tell which part to trust and will chase the mechanism first, because it is the checkable part.

**Cite `file:line`, and pin the version you read.** The fixer's virtualenv is not yours. A
claim about a third-party library is only checkable if the report says which version it was
read from; the same function differs across two minor releases and the fixer will silently
read the wrong one.

**A missing log is its own bug — file it too.** If diagnosing this required reading a process
table because the tool discards its own output, that is a defect in the tool and it belongs in
the tracker as its own issue. Per `CLAUDE.md`'s "Fixing a bug means fixing its class", the
missing log *is* the class; the failure you were chasing is one instance of what it hides.

**One failure per issue, cross-linked.** A second symptom found while investigating the first
gets its own issue that links back. Two failures in one thread cannot be closed independently
and the second is invariably the one that gets lost.

---

## 5. Ruling out, and the discriminator

**A hypothesis with no eliminated alternatives is a guess.** Name the alternatives you
considered, and for each, the evidence that kills it — or say plainly that you could not
eliminate it. "No open TCP connections for either pid, which argues against a stalled network
call" is worth more than a paragraph of argument for the favoured theory, because it is the
only part the fixer cannot reproduce.

**Then name the discriminator: the one command or observation that would separate the
hypotheses still standing.** This is the highest-value line in a report and it is the one most
often missing. It converts a remote diagnosis from a conversation into a single round trip.

Write it in the form *if it is A, X will be true; if it is B, Y will be*. It only counts if
running it is cheap and the two outcomes genuinely differ — a "test" that both hypotheses
predict is not one.

This matters most where the signature is degenerate. Distinct causes routinely produce
byte-identical symptoms — a machine that slept, a GPU driver reset, and a blocked console all
present as *every process alive, 0% CPU, VRAM resident, no exception, status still running*.
Nothing about the symptom will ever separate those. Only a discriminating test will, and the
person who can run it is the one filing the report.

---

## 6. Worked example — #279 / #280

Issues #279 and #280 (Ultralytics training deadlock on Windows, September 2026). Both were
filed by an agent on the reporting machine and both were above the normal standard. Keeping
the example is worthwhile precisely because it was a *good* report — the gaps below are the
ones that survive doing everything else right.

**What it got right, and what to copy:**

- An environment block with OS, GPU, driver, CUDA, trainer, base model, resolved
  hyperparameters, and the job's identity.
- Two CPU-time samples five seconds apart, quoted to the centisecond. That single measurement
  is what made "deadlock" a finding rather than an impression.
- `nvidia-smi` with utilization, temperature and power — which pre-emptively ruled out thermal
  throttling.
- File mtimes tying the last checkpoint to the last results row to the minute.
- The hypothesis explicitly labelled unconfirmed.
- One piece of negative evidence, with its reasoning ("no open TCP connections … argues
  against a stalled network call").
- A secondary finding kept separate from the primary one, and a follow-up filed as its own
  cross-linked issue rather than appended to the first.
- A workaround, with the exact command.

**The three gaps, each of which cost a round trip:**

1. **The decaying artifact was not captured.** The app had been launched from a terminal, and
   Ultralytics' own log — which would have named where training stopped — existed only in that
   window's scrollback. It was never saved. No later analysis could recover it, and the root
   cause is still unknown. This is why §3 says capture first and compose second.
2. **A checkable mechanism was asserted without checking it.** The report attributed the
   deadlock to worker IPC state "accumulating over epoch boundaries". Ultralytics'
   `InfiniteDataLoader` constructs its workers once and reuses them for the entire run
   (`ultralytics/data/build.py`, 8.4.70), so there are no epoch boundaries in the worker
   lifecycle for anything to accumulate across. The *conclusion* — a DataLoader deadlock —
   survived and is still the leading theory; the stated reasoning did not, and the fixer spent
   a round trip on it because a mechanism is the part of a hypothesis that looks checkable.
3. **No discriminator.** Sleep, a GPU driver reset (TDR), and a blocked console were all
   consistent with every observation in the report. Separating them took four exchanges and a
   question for the reporter — *were you at the machine that morning?* — that would have cost
   nothing to ask while he was standing at it.

---

## Decision log

Settled decisions (most recent first). Reverse or amend here; don't re-litigate in prose.

| Decision | Rationale |
| --- | --- |
| This doc is the full home for the reporting rules; `CLAUDE.md` keeps a pointer + checklist | one home so the rules can't diverge across two files (§1); mirrors `docs_design.md`'s arrangement |
| Blank issues stay enabled — no `ISSUE_TEMPLATE/config.yml` | the template is a checklist for whoever diagnoses, not a gate; forcing it onto a two-line docs typo is friction that gets routed around (§4) |
| The #279 worked example stays in full, in this doc rather than in `CLAUDE.md` | the gaps only teach with their evidence attached, and `CLAUDE.md` is loaded every session and is for operative rules, not case history (§6) |
| Two CPU-time samples are required, not encouraged | one sample cannot distinguish "idle" from "between writes"; this is the single measurement that made #279 diagnosable at all (§3) |
| Reports name a discriminating test rather than only a hypothesis | the degenerate-signature case is common and no amount of symptom detail resolves it; only the reporter can cheaply run the test (§5) |
