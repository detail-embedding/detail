<!--
A bug report body for `gh issue create --body-file`, for agents and contributors filing from
a terminal: `gh` cannot fill in an issue form. Its `###` headings are the bug form's fields
(.github/ISSUE_TEMPLATE/bug_report.yml), in the same order, so a report reads the same however
it was filed. tests/test_issue_forms.py keeps the two in step.

    gh issue create --type Bug --label bug --title "<what broke, where>" --body-file report.md

docs/bug_reports.md is the full spec. Capture the evidence BEFORE writing anything: some of it
is expiring right now. Delete a section only if it genuinely does not apply; do not delete one
because you did not get to it — say what you could not collect.
-->

### What happened, and what did you expect?

<!-- What happened, what you expected instead, and what state it is in now. -->

### Steps or command to reproduce

<!-- The steps, or the command. If it is intermittent, say how often and out of how many. -->

### How did you run it?

<!-- GUI (the Detail app), CLI (a detail command in a terminal), or Python script or notebook. -->

### detail doctor output

```text
```

### Run-folder logs

<!--
Attach each that exists: `activity.csv` (a timestamped line per update — the one that separates
*slow* from *stuck*), `worker.log` (the stage backend's own stdout and stderr, including the
line it stopped at), and `coordinator.log` for a multi-recording run.
-->

### Terminal output

<!-- The whole scrollback, saved to a file before the window was closed. Attach it. -->

### What you've ruled out, or a guess at the cause

#### Evidence

<!--
Measurements, with units and timestamps. Not characterizations.
"8490.65s -> 8490.65s across 5s" is evidence; "0% CPU" is an impression.
For a hang: process tree with pids, TWO CPU-time samples seconds apart, per-process memory,
nvidia-smi, file mtimes, the last row of any results file, job.json.
-->

#### What I ruled out, and how

<!--
Each alternative you considered and the evidence that kills it. Say plainly where you could
not eliminate one. A hypothesis with no eliminated alternatives is a guess.
-->

#### Hypothesis

<!--
CONFIRMED or NOT CONFIRMED, and by what. If it names a mechanism in our code or a
dependency's, read that code and confirm the mechanism exists before asserting it. Cite
file:line, and pin the version you read from.
-->

#### The discriminating test

<!--
The one cheap command or observation that would separate the hypotheses still standing:
if it is A, X will be true; if it is B, Y will be.
-->

#### Captured artifacts

<!-- What you saved and where. Note anything that was already gone by the time you looked. -->

### Workaround

<!-- If you found one, the exact commands. If data was lost, say so. -->

### Before you submit

- [x] I searched open and closed issues for this problem
