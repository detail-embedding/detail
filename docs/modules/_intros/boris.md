<!-- Prepended (prose only, no heading) to the generated modules/boris.md page. -->
Imports a BORIS tabular-events export as behavior labels. A recording is one animal in one
session, so an export holding several observations or several subjects is refused until the
parameters below pick one of each; the refusal lists what the export holds. Bouts from
different observations or animals are never merged onto one track.

A behavior scored both as START/STOP bouts and as POINT events is refused, naming the rows of
each kind: a behavior is one kind or the other, and evaluation scores the two differently.
