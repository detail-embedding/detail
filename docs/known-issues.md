# Known issues

Current limitations and edge behaviors to expect. Each notes when it bites and the workaround; the
run still completes normally in every case below.

## Older results on recordings with several video files keep wrong frame numbers

A query or clustering run made on a recording of several video files, before Detail numbered
frames across the whole recording, stored frame numbers that restart at 0 in each file. Its times
are right. Exporting such a run (clips, `instances.csv`, `clips.csv`, the
ethogram CSV) uses the stored numbers, so a bout in a later file is listed and cut as if it were in
the first.

- **Workaround:** rebuild the run's results once, with
  `detail analysis rebuild <an_NN> --experiment <dir>`, then export again. Any change to the run's
  labels or cutoff rebuilds them too.

## A moved experiment can't resume an in-progress training

Training bakes machine-local paths into its checkpoint. `train resume` recovers a run on the **same
machine**, but an experiment archived and moved to another machine (the cross-machine flow in the CLI
guide) can't resume a *partially-trained* model. A dataset moved *within* a machine resumes fine.

- **Workaround:** finish training before `experiment export`, or export with `--include-datasets`
  (a plain export leaves datasets out) and retrain the model from its dataset on the new machine.
