---
name: schema-edge-tests
description: Derive test cases from the edges of the schema and the workflow, not from typical scenarios. Use when writing tests for a stage that consumes a zarr, a run record, or anything another stage produced.
---

# Test the edges, not the happy path

Derive a stage's cases from the **documented edge properties of the schema it consumes**, not only
from typical scenarios. Happy-path fixtures hide real bugs, and they hide them by passing.

## Schema edges

- The detection schema states *"frames with zero detections have no rows"* (`DESIGN_PLAN.md` §9) — so
  any stage that reads it must be tested with **zero-detection frames** (absent from the zarr), as
  well as one-vs-many detections per frame, multi-part input, and an **empty part** (zero rows). A
  tracker that only ages tracks on frames that *have* rows is wrong, and a fixture where every frame
  has a detection will never catch it.
- For any per-part zarr consumer: the **empty store**, a **single row**, and the **resume boundary** —
  not just a populous store.
- Every zarr output test asserts column names, dtypes **and chunk sizes** explicitly. The pattern is
  in `DESIGN_PLAN.md` Tenet 16.
- Every validation check has a test that triggers it and asserts the specific exception type and a
  meaningful fragment of the message.

## Workflow edges, which is the half that gets missed

Which recordings play which role is a shape the fixtures have to cover, and until recently none did. Every fixture built a clusterer whose fit recordings **were** its assigned recordings, so `pooled_index`'s assumption that the two sets match went untested — and failed on every held-out fit, the project's primary evaluation protocol.

**Most clusterers in practice have an assignment outside their fit set: the untested shape was the normal one.**

So when a stage consumes something produced elsewhere, test **fit ≠ applied**, multi-recording, and
single-recording — not just the simplest topology. `tests/analysis/test_held_out_diagnostics.py` is
the model.

## Assert content, not absence of an exception

For a bug class whose signature is *plausible output instead of a crash*, a smoke test that only
checks the call returns is a regression test for the crash you already fixed.

The first attempt at the held-out fix above made `diagnose` run to completion while scoring only the
fit recording — every held-out bout silently unlabeled, every percentage diluted, nothing raised.

Assert the numbers a user reads: that the held-out recording's bouts **carry labels**, that a metric
clears its floor, that a table's scope matches what it was computed over.

## When a scenario test only looks like coverage

If reverting the fix does not make the test fail, the test is not covering the behaviour. Prefer
writing the test against the schema edge **first**, watching it fail, then fixing.
`guard-that-can-fail` is the procedure for that, including the trap where a revert silently fails to
apply and reports "passed".

## Fixtures

Use the shared fixtures in `tests/conftest.py`, documented with their code in `DESIGN_PLAN.md`
Tenet 16 — `experiment_dir`, `recording_dir`, `sample_detection_zarr`, `sample_assignment_zarr`.
Never define equivalent setup inline, and never mock a storage primitive: integration tests use real
temporary directories via `tmp_path`.
