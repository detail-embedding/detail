<!-- Prepended (prose only, no heading) to the generated modules/track_merge.md page. -->
Track-merge combines two or more single-subject tracking runs (typically one per animal,
each already filtered to its own detection class) into one multi-subject tracking run.
Nothing is re-tracked: each source's already-clean rows are relabeled with their own subject
id. The result is an ordinary tracking run, so the existing preview, location export and
embedding all read it unmodified.
