---
name: fix-the-class
description: Fix a bug's class, not its symptom — name the class, sweep for siblings, fix the mechanism, say where the fix stops, ship a guard, record it. Use when fixing a bug, triaging a defect report, acting on a review finding, or deciding whether a fix is done.
---

# Fixing a bug means fixing its class

A fix that removes the symptom in front of you and leaves its siblings in place is not done. Every
bug is an instance of something — name that something, then go looking for the rest.

**The repo's own history is the argument.** `query_eval` learned three lessons (a derived `params`,
a resolved default written back, a headline metric shipped beside its floor) and `cluster_eval`,
a module since removed, re-broke all three. A PR fixed an ETA that fed nominal stage weights into a time extrapolation, and
shipped while two identical instances sat in `cli.py`. Two more PRs gave analysis runs a reattach
that processing runs had had all along. Every one was found by a user months later, and every one
was a `grep` away at the time of the original fix.

## The six steps, in order (3.5 is a half-step)

### 1. Name the class

One sentence describing the defect **without naming the file it was found in**: *"a nominal
quantity fed into a computation that needs a measured one"*, not *"the distill ETA is wrong"*. If
you cannot write that sentence, you do not understand the bug yet.

### 2. Derive a searchable signature and sweep

A real class has a `grep` — a symbol, a call shape, a registry to enumerate. Search the **whole
repo**, not the module you started in.

`docs/recorded_classes.md` is the list of signatures somebody has already derived. Read it: if the
shape you just named is there, you are looking at a regression with a history, not a new bug, and
the row names the guard that was supposed to stop it.

### 3. Fix the mechanism, not the site

Put it in the shared contract (`executor/runners.py`, `core/`, the base class) so the next member
inherits it. Where N siblings share a shape, drive them from one table — `Experiment._JOB_RUN_KINDS`
is the pattern — instead of N parallel implementations that will drift.

### 3.5. Say where the fix stops

The sweep tells you where the shape appears; it does not tell you where the **reason** appears. For
every site the sweep found, record **fixed / deliberately not / hidden**, and give the reason for
each "deliberately not". A shared shape is not a shared reason, and a rule applied uniformly breaks
the one caller whose constraints differ. Two things that look like one mistake often answer
different questions.

Worked cases, including a class fix that was written and then **withdrawn**, are in
`docs/how_we_work.md` R3/R4.

A sweep that hits a capability hidden by `detail/surfaces.py` §25 has found a deliberate non-fix —
hidden is not dead, and its guards stay. Say so here.

**Measured over 65 PRs that followed this procedure: 48 did step 3.5, under 17 different
headings.** There is no shared shape for it, which is part of why it reads as done when it is not.

### 4. Ship a guard — a fix without one is incomplete

This class of bug is invisible to behavioural tests, because nothing is wrong at the line where it
is written. Match the guard to the shape:

| shape | guard | examples |
|---|---|---|
| registry / capability parity | a test data-driven over the registry, so a new member is covered the day it is added | `tests/configs/test_record_completeness.py`, `tests/api/test_job_reattach.py`, `tests/gui/test_model_availability.py` |
| a pattern that must not appear in source | a fail-closed source scan | `tests/storage/test_path_serialization.py`, `tests/models/test_no_implicit_network.py`, `tests/gui/test_ui_thread_blocking.py` |
| a wrong value the user reads | assert the number, not that the call returned | `tests/api/test_distill.py::test_training_eta_tracks_the_real_remaining_time` |

**Then make sure the guard can fail.** That is its own skill — invoke `guard-that-can-fail`, which
covers the negative case, the revert test, and the fixture traps. It is the step most often skipped
and the one that most often ships something green and empty.

**And account for what it costs.** A guard is required; a guard that recomputes something is not.
Coverage for the package must not drop (compare `--cov=detail` on `main` and on your branch) —
which is what makes *consolidating* existing tests legal rather than reading as deletion — and a
module starts at most one real executor run (`run_analysis` / `run_processing`) unless
`tests/test_real_run_budget.py` says otherwise with the number. Build an expensive artifact once in
a shared fixture and copy the result per test; see `tests/conftest.py::_temporal_fit_template`. Its
pose-clustering predecessor, removed with that module, replaced 28 identical fits worth 12 of the
suite's 36 minutes. CLAUDE.md's *What a new test costs* is the rule; neither half is a wall-clock budget, because
the defect is "done again" rather than "slow" (`docs/how_we_work.md` R10).

### 5. Retire a guard whose class has become impossible

A guard whose mechanism is gone — the surface deleted, the shape replaced by one that cannot express
the bug — is deleted in the change that made it impossible, and its row comes out of
`docs/recorded_classes.md`. A guard that can no longer fail is not protection, it is a green light
with nothing behind it, and it costs wall-clock on every run to say nothing.

The repo holds `FILE_TIER`, `UNIT_EXEMPTIONS` and `_CONFIRMATIONS` to exactly this standard — an
entry that excuses nothing fails — and holds *tests* to it only in part:
`tests/test_guard_negative_cases.py` requires a guard-tier source scan to carry a negative case.
Nothing checks that any other guard can still fail.

Note which way this cuts: the answer to a vacuous guard is to delete it **or** to give it a negative
case that proves it still bites. Never to leave it green and unexamined.

**Measured: 2 of 65 PRs did this step.** It is the least-performed step in the procedure.

### 6. Record the class

Write it in the **guard's module docstring** — the class, how it hid, the rule. That is the
canonical write-up, because it cannot drift from the thing enforcing it, where a second copy in
prose goes stale. `tests/gui/test_frame_source.py` is the model.

Then add **one line** to `docs/recorded_classes.md`: a rule and a guard path. Not the write-up.

**Measured: 20 of 65 PRs did this step**, and the most recent row added before this skill existed
cited a guard file with no module docstring at all — the write-up existing only in the index, which
is the inverse of the rule.

## Why the steps with an artifact get done

Step 4 (ship a guard) was done in **63 of 65** PRs. Steps 5 and 6 were done in 2 and 20. The
difference is that step 4 leaves a file in the diff and the others leave a sentence. So: when you
do steps 3.5, 5 and 6, leave something a reader can see — a verdict column with one row per swept
site, a deleted guard, a docstring.

And beware the failure that shape invites, from this repo's own notes: *"A rule that describes an
artifact ('say where the fix stops') is satisfied by producing the artifact: an agent writes the
sentence without doing the thinking, and it reads as diligence."*

## Read these

- `docs/recorded_classes.md` — every class already paid for, each with its guard
- `docs/how_we_work.md` — R1–R17, the method and the worked cases, including the withdrawn fix
- `CLAUDE.md` — the rule itself, and the hard rules a fix must not break
