# Known issues

Current limitations and edge behaviors to expect. Each notes when it bites and the workaround; the
run still completes normally in every case below.

## A moved experiment can't resume an in-progress training

Training bakes machine-local paths into its checkpoint. `train resume` recovers a run on the **same
machine**, but an experiment archived and moved to another machine (the cross-machine flow in the CLI
guide) can't resume a *partially-trained* model. A dataset moved *within* a machine resumes fine.

- **Workaround:** finish training before `experiment export`, or export with `--include-datasets`
  (a plain export leaves datasets out) and retrain the model from its dataset on the new machine.
