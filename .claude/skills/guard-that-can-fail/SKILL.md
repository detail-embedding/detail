---
name: guard-that-can-fail
description: Make a test actually able to fail — add the negative case, revert the fix and watch it break, delete the feature, and check the fixture reaches the code path. Use when writing or reviewing any test meant to prevent a class of bug: a source scan, a registry sweep, a structural or anti-drift check.
---

# A guard that can fail

A guard exists because the bug it covers is **invisible to behavioural tests** — nothing is wrong at
the line where it is written. That is exactly the condition under which a guard can be green for the
wrong reason and nobody finds out.

**Measured in this repo: 34 PR bodies contain the word "vacuous", and three admit a guard that
passed against its own reverted fix.** One PR wrote a guard, discovered it could not fail, and
deleted it — *"A decorative test is worse than none."* To see how many source scans still lack a
negative case, count the entries in `SCANS_WITHOUT_A_NEGATIVE_CASE`
(`tests/test_guard_negative_cases.py`).

Before believing a guard, **name the mutation it should fail on, then run it.**

## 1. Ship the negative case

A scan that matches nothing today passes vacuously forever. Add the case that proves it still bites,
named `test_the_*_would_catch_*`:

```python
def test_the_save_button_scan_would_catch_one():
    ...
```

Factor the predicate into a **pure function** so the negative case can hand it a literal — the
`tests/test_docs_nav.py::_undeclared` idiom. A negative case that has to build a whole fixture to
plant a violation usually ends up testing the fixture.

`tests/test_guard_negative_cases.py` enforces this by ratchet: `SCANS_WITHOUT_A_NEGATIVE_CASE` is the
exemption list and its `CEILING` is **lowered when a scan grows a negative case, never raised**.
What it checks, exactly:

- **Which files:** only files in `tests/tiers.py::GUARD_FILES` whose source contains `ast.parse` or
  `rglob("*.py")`. A scan outside the guard tier is not checked at all.
- **Which names count:** a test function whose name contains `would_catch`, `would_fail`,
  `would_flag`, `would_notice`, `catches`, `rejects`, `detects`, `counterexample`, `negative` or
  `not_vacuous` (`_NEGATIVE_CASE`). Nothing else counts: `would_be_caught` and
  `would_actually_catch` do not, so a few entries in the exemption list do have a negative case,
  under a name the ratchet misses.

**Feed it the spellings a person would actually write.** Measured 2026-09-11: of 23 source-scanning guard files,
11 had no negative case — including `tests/storage/test_path_serialization.py`, the one CLAUDE.md
calls "the idiom" and four other guards were copied from. Fed real spellings, it caught
`str(path.relative_to(base))` and missed an f-string, a `%`-format, a call in the receiver, and a
`str(` wrapped onto two lines. Nobody had ever watched it reject anything.

## 2. Revert the fix and watch the guard fail

If it still passes, it is not covering the behaviour.

**A revert that silently failed to apply reports "passed", which reads identically to a covered
guard** — an anchor `black` has since reflowed, a patch that raised before it wrote. So:

1. assert the revert **changed the file**
2. watch the test fail
3. restore, and watch it pass

Two of the three states are not enough.

## 3. Delete the feature, not just the fix

Reverting a fix catches a regression; it does not catch a suite that never tested the feature at all.
Remove the feature's **mechanism** — the two overrides, the whole class — and re-run.

Four GUI suites stayed green with `SnapshotViewMixin` gutted, because no fixture made the frozen
thing differ from the live one, and a fixture that never diverges cannot tell a working freeze from a
broken one. In the same change, forcing `openable = False` — which makes the feature unreachable in
the app — **left 58 tests green**. Two different deletions, two different measurements, one finding.

The check is a number: **report how many tests the deletion broke. Zero is a finding, not a pass.**
`docs/how_we_work.md` R11 is this rule with both commit quotes.

## 4. The commonest reason a guard cannot fail is its fixture

A test that never reaches the code path proves nothing about it:

- a caption check ran on a one-group cohort, which draws neither a paired figure nor a ranking, and
  passed with `_caption` deleted from both
- a label check ran on a fixture with no record to label from, so it asserted the **fallback**
- a parametrisation over config models iterated **zero times** for two of them

**Cut the fixture to one of everything the stage loops over, not just one unit.** A cadence test
pinned the run count at one and then used three exemplar clips, so `0 < f < 1` was satisfied by the
exemplar count alone and the one-clip case emitted nothing until the search ended. Parametrize over the inner
count rather than picking a comfortable one.

**Test the workflow edges, not only the schema ones.** Every fixture built a clusterer whose fit recordings *were* its assigned recordings, so the assumption that the two sets match went untested — and failed on every held-out fit, the project's primary evaluation protocol. Most clusterers in practice have an assignment outside their fit set: the untested shape was the normal one.

## 5. Three idioms that pass against the bug they guard

- **`isVisible()` where you mean "deliberately hidden"** — use `isHidden()`, which is independent of
  whether an ancestor was ever shown.
- **Containment in a rendered string** — `"5 more" in "Review 25 more"` is True, and `"9 yes" in
  "19 yes"` is too. Compare the whole string.
- **Draining on a value that is already truthy** — `_drain(qapp, lambda: bool(tab._judged))` returns
  instantly on the *previous* read, so the assertion runs against pre-refresh state. Clear the value
  first, or wait on a count that must increase.

## 6. Three ways a guard passes without guarding

- **Diverge before the surface reads, not after.** A widget reads its state once at construction, so
  changing the data afterwards proves nothing — the values it holds were read while the two agreed.
- **Scope a guard to its own subject.** A heap sweep for cleared closures passed alone and failed in
  the full suite by tripping on other tests' residue. Take a weak reference to the object under test
  rather than walking `gc.get_objects()`.
- **Key an allowlist by identity, not by label.** A walk over every enabled control keyed on button
  text let an unlabelled one through: its label is set later, so before that it matches `""`. Filter
  on *visibility* and the same walk sees nothing at all, because a never-shown widget reports nothing
  visible — vacuous, and green.

## 7. Assert the property, not a proxy for it

"An interval band was drawn" passes on a band of constant width; "the score is not on screen" passes
when it is rendered at a different number of decimals; "the video is not playing" passes in a fixture
whose clips never decode. `len(notice) > 20` passed on the one page that named no action, by one
character.

Name the property that makes the fix a fix, then assert **that**.

**Assert content, not absence of an exception.** For a bug class whose signature is *plausible output
instead of a crash*, a smoke test that only checks the call returns is a regression test for the crash
you already fixed. One attempted fix made `diagnose` run to completion while scoring only the fit
recording — every held-out bout silently unlabeled, every percentage diluted, nothing raised. Assert
the numbers a user reads.

## 8. A widget that was never shown is not in the state you are asserting about

An unlaid-out `QTableView` reports the default `35.0` axis width whatever `setWidth` was called with;
a pyqtgraph `ViewBox` sits at its 0–1 default whatever the data is; `isVisible()` is False for every
child of a widget nobody called `show()` on. Three guards in one PR passed against their own reverted
fix on exactly this.

If the property under test is geometry, visibility or layout: `resize()` + `show()` +
`processEvents()` first — and if a **resize** is what triggers the bug, resize again after the draw.

## Where the guard goes

Match the guard to the shape. Then decide whether the file belongs in the **guard tier**: the files
listed in `tests/tiers.py::GUARD_FILES`, which CI's `guard` job (`pytest -m guard`) runs on every
PR and `scripts/select_tests.py` adds to every edit-loop run.

- **A file gets into the guard tier only by being added to `GUARD_FILES` by hand.** Its cost tier
  (`unit` / `gui` / `slow`) comes from its directory automatically.
- **Nothing checks this for you.** `DIRECTORY_TIER` covers all of `tests/`, so `test_test_tiers.py`
  passes whether or not you add the file. It fails only if a `GUARD_FILES` entry resolves to the
  `slow` tier, because guards run at every stage of the gate.
- **Outside the guard tier**, a test still protects its class, but runs only when its own tier or
  the change selects it, and the negative-case ratchet above does not see it.

The Guard column of `docs/recorded_classes.md` names the test that protects each class, and that
test may sit in any tier.

The class this guard covers is recorded in its **module docstring** (the canonical write-up) plus one
line in `docs/recorded_classes.md`. That is step 6 of the `fix-the-class` skill, which is where to go
for the procedure around this one.
