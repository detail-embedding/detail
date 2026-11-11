---
name: Bug report
about: Something went wrong. Fill this in from the machine it happened on.
title: ''
labels: ''
assignees: ''
---

<!--
If you are filing this from the machine the problem happened on, capture the evidence
BEFORE writing anything — some of it is expiring right now. Any output the tool does not
persist lives in one window and dies with it, so if the app was launched from a terminal,
save that window's ENTIRE scrollback to a file first.

The run's own folder keeps more than the scrollback does. Attach each of these that exists:
`activity.csv` (a timestamped line per update — the one that separates *slow* from *stuck*),
`worker.log` (the stage backend's own stdout and stderr, including the line it stopped at),
and `coordinator.log` for a multi-recording run. A single-recording run has no coordinator
log; it still has the other two.

Agents: docs/bug_reports.md is the full spec. Delete any section that genuinely does not
apply; do not delete one because you did not get to it — say what you could not collect.
-->

## Summary

<!-- What happened, what you expected instead, and what state it is in now. -->

## Environment

<!-- Paste `detail doctor` verbatim. Do not summarize it. -->

```
```

- **OS / version:**
- **GPU / driver / CUDA:** <!-- if relevant -->
- **DETAIL commit:**
- **How it was launched:** <!-- GUI, CLI, which terminal app -->

## Reproduction

<!-- The steps, or the command. If it is intermittent, say how often and out of how many. -->

## Evidence

<!--
Measurements, with units and timestamps. Not characterizations.
"8490.65s -> 8490.65s across 5s" is evidence; "0% CPU" is an impression.
For a hang: process tree with pids, TWO CPU-time samples seconds apart, per-process memory,
nvidia-smi, file mtimes, the last row of any results file, job.json.
-->

## What I ruled out, and how

<!--
Each alternative you considered and the evidence that kills it. Say plainly where you could
not eliminate one — that is useful too. A hypothesis with no eliminated alternatives is a
guess.
-->

## Hypothesis

<!--
State whether this is CONFIRMED or NOT CONFIRMED, and by what.
If it names a mechanism in our code or a dependency's, go read that code and confirm the
mechanism exists before asserting it. Cite file:line, and pin the version you read from —
the fixer's virtualenv is not yours.
-->

## The discriminating test

<!--
The one cheap command or observation that would separate the hypotheses still standing.
Write it as: if it is A, X will be true; if it is B, Y will be.
This is the highest-value line in the report. Degenerate signatures are common — a machine
that slept, a GPU driver reset and a blocked console all present identically — and nothing
about the symptom will ever separate them.
-->

## Captured artifacts

<!--
What you saved and where: scrollback file, event-log dump, run/args.yaml, job.json, logs.
Attach them. Note anything that was already gone by the time you looked.
-->

## Workaround

<!-- If you found one, the exact commands. If data was lost, say so. -->
