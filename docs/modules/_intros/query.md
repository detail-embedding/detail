Behavior scan is the app's name for the `query` module (`analysis query` on the CLI). Point at
a few example clips of a behavior you can already name, and it finds every stretch
that resembles them across the recordings you select. You then pick a confidence cutoff by
looking at what each setting returns.

It works **without ground truth**, and that shapes it. You can review what came back, so
precision is something you can measure; you cannot know what was missed without labeling
everything, which is the work this exists to avoid. So it is built around calibrate-then-audit
rather than around a score.

A run searches **several behaviors at once**: a cohort is usually quantified on more than one,
and searching each separately would reload every trajectory once per behavior. Each behavior
carries its own example clips and, afterwards, its own cutoff. The match parameters are
run-level: instances found under different settings are not comparable, and one run emits one
artifact. If two behaviors genuinely need different settings, launch two runs.

**The search stores no cutoff.** Choosing one is not part of searching, so changing your mind
costs nothing and never a second search.

The **method** sits in advanced settings on purpose: without ground truth you cannot evaluate one matcher against another on your own data, so it is not a casual dropdown. One method is available, **soft k-NN**, and it is the default. It scores each frame against all your clips at once, so more examples cost it almost nothing; it sizes each match from the score's own shape rather than from the example; and it needs a background sample to score against. A search is quick. Its page says what is and is not known about it.
