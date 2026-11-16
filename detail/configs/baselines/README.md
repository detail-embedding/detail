# Baselines

Named starting points offered by every launch form's **Start from** row, beside *Defaults* and
the settings of previous runs. One `.yaml` per baseline, in this directory.

**None ship today.** The mechanism is here and guarded; the files arrive when there are numbers
worth recommending, which means measured on real recordings rather than guessed. A baseline is
a recommendation to every user of the tool, so an invented one is worse than none.

```yaml
schema_version: 1
name: Mouse open field
description: >-
  What rig and subject this is for, in a sentence.
settings:
  detection: {conf_threshold: 0.25}
  tracking:  {tracker_id: iou_single_tracker}
  embedding: {constructor: silhouette_centered}
```

Keyed by `module_id`, so one baseline can span a pipeline and each form offers it only for the
modules it names.

## Rules, enforced at load

`configs/baselines.py` refuses a file that breaks any of these, so a wrong baseline is a build
error rather than something that waits in a dropdown until somebody picks it:

- every key exists on that module's config model;
- every value is inside the bounds the model declares;
- **no key is a `selection`** — a field that names an artifact (`embed_run_id`,
  `source_recording_ids`, `tracking_run_id`). That is the rule the whole idea rests on: a baseline holds settings only,
  so it means the same thing in another experiment on another machine. Embedding a data
  reference is what makes presets rot elsewhere — ilastik's `.ilp`, a SLEAP config that cannot
  find its `models/` folder.

`tests/configs/test_baselines.py` checks each of these, and sweeps this directory so a file
added here is validated the day it lands.
