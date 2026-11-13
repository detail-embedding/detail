# How work is done here

**Agent-facing. Not a user page** — it is in `mkdocs.yml`'s `exclude_docs`.

`CLAUDE.md` is the rulebook: what must be true of the code. This is the method: how to decide
*which* rule applies, *how far* a fix should reach, *what* a claim has to carry, and when the answer
is to build nothing at all.

**Where this came from.** Every quote below is from a commit message written while this project
was developed in a private repository. That history is not published, so the commit hashes are
left out and a quote cannot be looked up from here. Nothing here is invented advice.

**How it is organised, and why the split matters.**

- **Part I — Rules.** Each has a trigger, a procedure, and a **check that can come out false**. If
  you cannot fail the check, you have not followed the rule.
- **Part II — Judgment.** Worked cases only. These are decisions with no procedure behind them.
  They are deliberately not written as rules, because a rule that describes an *artifact* — "say
  where the fix stops" — is satisfied by producing the artifact. You would write the sentence
  without doing the thinking, and it would read as diligence. Read the cases; ask the same
  questions; expect to get some wrong.

Triggers and checks marked **[inferred]** are not sourced from the corpus — they are one reading of
the worked cases. Treat them as a starting point and correct them.

---

# Part I — Rules

## R1. Find the seam before you build

**Trigger:** you are about to add a module, stage, widget or helper.
**Procedure:** find what already does the closest thing. Ask whether the gap is a missing feature or
a missing *connection* between two things that exist.
**Check:** the change says which existing machinery it reused, or why none fit.

> *"Almost none of the fix needed building: prompt_test already runs N prompts x M sampled frames …
> The work was a seam, not a feature."*
> *"The plan called for a third seam, merge_segments … So the seam dissolves into CLUSTER_RULES and
> the machinery is one registry entry instead of a new pipeline stage."*

*Without this:* a parallel pipeline stage beside the one that already does it.

## R2. Fix at the seam every caller passes through, then sweep and count

**Trigger:** you have a defect at a named call site.
**Procedure:** put the fix where every caller passes through it, not at the site. Derive a
searchable signature. Sweep `detail/`, `tests/` **and** every other code tree. Count what you find.
**Check:** the change states the number of sites found and their disposition. A number larger than
the reported one is the normal outcome.

> *"The class is *a hazard diagnosed at one call site and mitigated there, rather than at the seam
> every call site passes through* … it was complete on the day it was written and incomplete the
> moment a second module wrote a zarr."*
> *"The sweep for that signature found it live at ten sites across eight modules, not the two in the
> report."*
> *"dataset_export was MISSED by phase 2 — now populates eta_s/elapsed_s like its sibling
> labeling."*

*Without this:* the third occurrence, fixed the same local way.

## R3. Say what the fix deliberately does not change

**Trigger:** R2's sweep returned more than one site.
**Procedure:** for each site, decide *fixed* / *deliberately not* / *hidden*, and give the reason for
every "deliberately not". A shared shape is not a shared reason.
**Check:** every swept site appears in the change with a disposition, and any allowlist the guard
carries is empty or every entry states its reason.

> *"Deliberately **not** listed: `frequency_curve(` / `candidates_at(` / `score_distribution(` …
> listing them would mean allow-listing them straight back."*
> *"The rename is symbol-targeted, not text-targeted, because "comparison" is also an ordinary
> English word here… Those are untouched."*

**Counter-case — a class fix withdrawn after it was written:**
> *"I had extended it to retry_run and resume_distill as a class fix; those are coordinator-backed…
> The extension bought them nothing they did not have and added the race."*

*Without this:* a correct rule applied uniformly, breaking the one caller whose constraints differ.

## R4. Two things that look like one mistake may not be

**Trigger:** you are about to unify two code paths, two helpers, or two error cases.
**Procedure:** state the *question* each one answers. If the questions differ, they are different
things that happen to share a shape.
**Check:** the change names the distinction, or says explicitly that the two questions are the same
one. **[inferred check]**

> *"The drift had four distinct shapes, worth naming because they are not the same mistake:"*
> *"**Eligibility and coverage are different reasons to disable a row.**"*
> *"Allocation needs a different question from listing — what has been claimed rather than what is a
> clusterer"*

*Without this:* one helper answers two questions and is wrong for one of them.

## R5. Reject the obvious fix in writing

**Trigger:** you have a candidate fix that looks cheap.
**Procedure:** before writing it, check whether it is already present and failing, and name the
mechanism that makes it insufficient — not the observation that it did not work.
**Check:** the change contains a sentence of the form "X was already in place and is not the answer,
because Y".

> *"Retrying was already in place and is not the answer: core/atomic has retried os.replace 20x at
> 10ms since well before the report… What was missing was any contract for what the caller does
> then."*
> *"huggingface_hub binds that env var into a module constant at import time, so the write was a
> silent no-op — and moving it earlier cannot help, since readiness() imports huggingface_hub (via
> _require_hf) before load() ever runs."*
> *"The default was not the bug. Three mechanisms each failed to stop it:"*

*Without this:* the plausible fix ships beside the real cause, and the guard stays dead.

## R6. Every claim carries units, a baseline, and the machine

**Trigger:** you are about to write "faster", "cleaner", "rare", "most", or "should be equivalent".
**Procedure:** replace it with a measurement: the number, its unit, what it was before, and where it
was measured.
**Check:** no adjective in the change describes a quantity the code could report.

*Without this:* an unfalsifiable claim that nobody can regress against.

## R7. Name the evidence that does not exist

**Trigger:** you are about to state a cause, a default, or a limit.
**Procedure:** separate what you measured from what you assumed. For the assumed part, say what
would settle it and where that evidence would live.
**Check:** the change distinguishes measured from unmeasured, and every unmeasured claim names its
missing study.

> *"This does **not** have a proven link to the intermittent exit-134 aborts seen on CI. It removes a
> measured hazard of the same class; whether that is what has been hanging the runs is unproven."*
> *"Not done here: the proposals-per-frame measurement. It needs CUDA and real footage, and this
> machine has neither."*
> *"**The floor is a policy choice, not a measurement**, and the comment says so. The literature
> supports *having* one and gives no principled value; a constant that looks derived invites being
> read as one."*
> *"Nothing here has been run on our data — candidate directions with mechanisms, not results."*

*Without this:* a plausible mechanism is narrated as the cause and the next reader stops looking.

## R8. Refuse rather than round, guess, or footnote

**Trigger:** an input is ambiguous, unreachable, or means two things.
**Procedure:** raise, and name the reachable values. Do not pick one silently, and do not proceed
with a caveat in the output.
**Check:** the ambiguous case has a test asserting it raises, with the message fragment.

> *"Asking for 3 raises and names the reachable values rather than rounding — those frames were never
> processed, so rounding would silently analyse a different set than the one requested."*
> *"with several streams and no name it now raises. One stream still resolves silently — ambiguity is
> the trigger, not the absence of a name"*

**The related half — bound a fallback by its reason, never its symptom:**
> *"The fallback is gated on the rule rather than on the file being absent, so it cannot mask a real
> missing artifact: a bout-path clusterer whose cache was deleted still raises the original error,
> which is a case worth reporting."*

*Without this:* the fix silences a second, genuine failure.

## R9. Validate every guard against the unfixed tree

**Trigger:** you wrote a test for a fix. Always.
**Procedure:** revert the fix. **Assert the revert actually applied.** Run the guard. Watch it fail.
Restore. Watch it pass. Three states, not two.
**Check:** the change states how many tests failed against the reverted tree.

> *"Baseline before the fix: 7 failed, 2 passed, and the 2 were exactly those two."*
> *"Reverted and watched 8 of 10 fail, then restored."*
> *"Each of these was confirmed by applying the mutation, watching the test, and restoring -- all
> three states, since a revert that silently fails to apply reports 'passed' too."*

**When the guard still passes, the fixture is wrong — not the guard:**
> *"16 guards, each validated by reverting its fix and watching it fail. Two of them reported vacuous
> first… Both fixtures fixed rather than the guards weakened."*
> *"A guard whose subject nothing currently exercises is a guard that passes forever."*
> *"A guard that cannot fail is worse than none, because it reads as coverage."*

*Without this:* green tests protecting nothing.

## R10. Assert the wrong content, not that the call returned

**Trigger:** the defect produced plausible output rather than an exception.
**Procedure:** assert the value a user reads. Name the property that makes the fix a fix, and assert
that.
**Check:** the assertion would fail on the old code for the *right* reason, not because it raised.

> *"The tests assert content — a held-out bout carries its label, the digest names the recording it
> scored — rather than that the call did not raise. An exception was never the failure mode here, so
> a smoke test that only checks for one would have passed throughout."*
> *"So the guard asserts the match set, not the costs. A guard on the curve would fail on numerical
> noise while missing the failure it exists to catch."*
> *"Guarded by counting the reads rather than timing them: a wall-clock assertion on the machine that
> also runs the suite is a flake, and the defect is not 'slow' but 'done again'."*

*Without this:* a regression test for the crash you already fixed.

## R11. Delete the feature, not just the fix

**Trigger:** you believe a feature is covered by tests.
**Procedure:** remove the feature's *mechanism* — the overrides, the whole class, the flag that makes
it reachable — and re-run. Reverting a fix catches a regression; it does not catch a suite that never
tested the feature at all.
**Check:** the change reports how many tests the deletion broke. Zero is a finding, not a pass.

> *"Both mixin overrides could be replaced with `super()` — the feature entirely removed — and the
> GUI suites stayed green."*
> *"forcing `openable = False`, which makes the feature unreachable in the app, used to leave 58 tests
> green."*

*Without this:* you prove a regression is gone without noticing the feature was never covered.

The operative form of this rule, with the other ways a guard passes without guarding, is the
`guard-that-can-fail` skill. The two quotes above are two different deletions in one commit — the
mixin overrides, and the reachability flag — so they are two measurements, not one restated twice.

## R12. Ship the positive control

**Trigger:** a change is gated on a flag, a config value, or a code path selection.
**Procedure:** test both sides. Show that the *same* input under the *other* setting produces the
other outcome.
**Check:** the test suite contains the negative arm, not only the new behaviour.

> *"Tests include a positive control -- the same unlabeled recording under the default still lands in
> `skipped` with no rows -- so the flag is proven to be what does the work."*

*Without this:* the test passes for a reason unrelated to your change.

## R13. Never add a production hook to make a test possible

**Trigger:** you want to reach into production code to observe something.
**Procedure:** find a seam that already exists, or test at a different level, or accept a static
guard instead.
**Check:** no symbol exists in `detail/` whose only caller is a test.

> *"An earlier attempt added a hook to production code purely to make this testable; that is the wrong
> trade and it was thrown away."*
> *"The guard is static — field names and source text … A guard that needed all that would not get
> written, and an unwritten guard protects nothing."*

*Without this:* production shape distorted by test convenience.

## R14. Verify through the real command, on real data, at real scale

**Trigger:** you are about to report something as working.
**Procedure:** run the path a user runs. Use data the size a real experiment has. If it is a screen,
open it and look at the picture.
**Check:** the change names what was run, on what, and what it found — findings from the real run are
the normal outcome.

> *"Two gaps found by running the re-embed, not by testing it."*
> *"Only walking the CLI caught it, which is why the guard is a real fit through the module rather
> than an assertion about the source."*
> *"I should not have generalised a layout from an 800-cell test; benchmarking on realistic data is a
> rule this repo already records."*

*Without this:* "the tests pass" reported as "it works".

## R15. Separate pre-existing from introduced

**Trigger:** something fails that you did not expect.
**Procedure:** verify against `main` without your branch. Say which it is. Fix the pre-existing thing
your change *depends on*; leave someone else's half-landed feature alone.
**Check:** every failure mentioned is labelled pre-existing or introduced, with how that was
established.

> *"Also fixes a pre-existing data-corruption bug this work depends on … Verified against main before
> changing it."*
> *"Unrelated and pre-existing: the GUI suite flakes about one run in four on a leaked
> `BackgroundCall`… Verified on `main` without this branch."*
> *"Listed rather than wired, because wiring someone else's half-landed feature from a merge commit is
> not this PR's business."*

*Without this:* a red suite blamed on the branch, or unrelated breakage folded in silently.

## R16. Correct your own claim in the artifact that carries it

**Trigger:** you find that something you previously wrote is wrong.
**Procedure:** correct it where it lives — the docstring, the design section, the comment — not in a
later message. Say plainly that it was wrong.
**Check:** the correction is in the file a future reader will find, and the change says so.

> *"The fix was right and is revert-checked; the explanation beside it was a plausible story that
> happened to be false, which is worse in a comment than in a commit because it is what the next
> reader believes."*
> *"A ported docstring claimed a monotone ramp scores ~0 … My own test falsified it: a ramp scores
> 0.75 … Corrected the docstring, not the test."*

**When code and spec disagree, decide which is wrong — often the code:**
> *"DESIGN_PLAN §5 needed no change … The spec was right and the code drifted."*
> *"`DESIGN_PLAN.md` §20 is amended in this change rather than quietly broken, and the sentence names
> what would have to change if the sweep later separates them"*

*Without this:* a wrong claim keeps steering decisions after the code is right.

## R17. What a change writes down

**Trigger:** you are writing a commit message or PR body.
**Procedure:** the body carries the class (not the site), the obvious fix you rejected and why, the
sweep count, what you deliberately did not change, the numbers with units, the guards and how many
failed against the reverted tree, and anything you got wrong on the way.
**Check:** a reader who was not there can tell what would have to change for the conclusion to be
wrong.

This is not ceremony. It is why this document could be written at all: the reasoning was recorded at
the moment it happened, in the artifact that carries the change.

> *"Two layout rules this package already had, applied at the one site that reported each and never
> swept. The sweep is the change; the two fixes fall out of it."*

---

# Part II — Judgment

No procedures here. These are decisions the corpus shows being made, with the reasoning that was
given at the time. Read them for the shape of the question, not for a rule to apply.

## J1. When to withhold something that works

Roughly twenty times, a working capability was built and not shipped. The reason is never "it is not
finished" — it is that shipping it would assert something nobody measured.

> *"**Restore is deliberately not built.**… real but narrow, and not worth an affordance whose value
> cannot be shown."*
> *"A baseline is a recommendation to everyone using the tool, so `Single subject` was one setting's
> worth of advice with no measurement behind it. Out… until there are numbers worth recommending."*
> *"Two mechanisms I wrote and then removed rather than ship unguardable:"*
> *"the guard written for it passed against its own revert… Reverted rather than shipped
> unfalsifiable."*
> *"**No version tree.** Lineage is stored (`parent`) and never made a graph to navigate: 2 of 12
> analysts opened such a feature in the one controlled study, and VisTrails — built around exactly
> that — was abandoned in 2018."*
> *"Experimental: reachable only by constructing the model directly, deliberately NOT wired into the
> embedding config / CLI / GUI (not yet a user-facing capability)."*

The counterweight, so this does not become a reason to hoard: a capability with **no caller** is not
withheld, it is dead, and it gets deleted.

> *"`query_eval_run` is deleted … zero callers and zero tests … Adding a test would only freeze an
> unused function;"*

## J2. How wide is this class, really

R2 says sweep; R3 says bound it. Neither tells you where the line is. These are the calls being made.

> *"Two copies could already drift; temporal clustering would have made three, which is where this
> stops being tolerable."*
> *"`audit.csv` is the second consumer, which is what justifies extracting now rather than making a
> third copy."*
> *"Those are recorded as extraction targets triggered by the second implementation, not by a date."*
> *"The fix is at the stream, not the ~250 non-ASCII literals in the package. Which of them reaches
> stdout is not a property of any one of them,"*
> *"`names` is a property onto the current one, which is what kept the change small — `name_of`,
> `named_count`, the manifest… are all untouched."*

## J3. When a default has no right answer

> *"`aggregate_to` has **no default**… The CLI refuses rather than picking"*
> *"a default would be a guess wearing a number"*
> *"That is still true of **accuracy** — the matcher sweep has never run — and no longer true of
> everything… So the default is chosen on **cost**, every surface says so, and the accuracy question
> stays open"*
> *"no conversion can preserve a frame-count default for everyone — holding 60 Hz fixed necessarily
> changes 15 fps, which is the bug itself; this picks the side our data is on."*

## J4. When the real fix is a design question

> *"The measure is the thing to fix, and a real `from -> to` key does not fit the current `fans_out`
> contract… so it is its own change rather than something to bolt onto a figure."*
> *"what a transition *matrix* should look like is an open §24 question and 400 wrong answers is not a
> way of answering it."*
> *"Parked until the placement is decided, so it need not be re-hashed."*

## J5. Which direction to guess in

When you cannot know, the two mistakes rarely cost the same.

> *"the two mistakes do not cost the same: guessing 'ours' refuses a retry somebody can repeat,
> guessing 'not ours' starts a second process on a device the first one still holds."*
> *"an omission is an AttributeError rather than a blank screen."*
> *"A bound the caller hits beats a silently degenerate labeling; the sweep is a test rather than a
> comment."*

## J6. When the method itself lies to you

The rules above can all be followed and still produce a false result. These are the recorded cases.

**The revert harness can lie in the safe-looking direction.**
> *"Python invalidates a cached `.pyc` on (mtime, size), so successive writes inside one mtime tick
> look unchanged and iteration 3 ran iteration 2's bytecode — reporting a real guard as vacuous… a
> revert harness that rewrites one module repeatedly can lie in the safe-looking direction."*

**Your reference implementation can carry the same bug.**
> *"`_brute_optimum` in the test omitted the same relaxation, so the reference and the DP agreed with
> each other and both were wrong."*

**What you tested and what you shipped can differ.**
> *"the run was green against a *working tree* that contained the fixes, while the *commit* did not.
> Scoping `git add` by path… turned "what I tested" and "what I shipped" into two different things."*

**A guard's own "guard the guard" test can go stale.**
> *"The identity guard's 'guard the guard' test named `temporal_stack_s`, so it would have passed
> while testing nothing once the field was gone; re-pointed at a live field."*

**A documented rule can be believed, quoted, and absent.**
> *"The GUI style block has said 'persist main window size + sidebar split via QSettings' for a long
> time, and `saveGeometry`… appeared nowhere in detail/gui/. Believed, quoted in review, and absent —
> which is worse than a rule that was never written, because nobody goes looking."*

---

## What this document does not cover

**Reviewing someone else's change.** The corpus is about making and verifying changes, not about
reading them. If you are reviewing, the rules above give you the questions to ask — where does this
stop, what was swept, how many failed against the reverted tree — but the standard for accepting work
is not recorded anywhere and is not inferable from commit messages.
