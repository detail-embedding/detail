"""A §22 clustering fit on a remote target (§6 "Remote analysis").

Pose and temporal clustering share everything the contract asks of a remote-capable module, so
it lives here once rather than once per module:

- **What it reads, refreshed.** Each recording's own embedding run (``embed_run_for``), sent
  again over the cluster's copy (``refreshed_inputs``). The upload never rewrites a run directory
  the cluster already has, so an embedding run that went up while it was still being written —
  under another remote job of the same experiment — would keep its first ``zarr.json`` there for
  good, and the fit would read a shorter store than this machine has, with nothing failing. The
  mirror is exact and costs a file list when nothing changed.
- **What it writes.** Only its own run directory: the fit, the §19 artifact, the node cache and
  its thumbnails, the history. That is what the copy-back returns, so there is no
  ``after_copy_back``.
- **The video.** The node cache decodes a thumbnail per pictured member, so the job's recordings'
  videos are staged on the cluster; the shepherd does that already for any job naming
  ``source_recording_ids``.
- **The node.** One GPU, the fewest Tillicum accepts: the fit, the projection and the decode
  are CPU work, and the node's CPUs and memory scale with the GPUs asked for.

Not refreshed, deliberately: the detection runs behind each embedding run. They only place the
thumbnails' boxes, and a detection store is hundreds of thousands of chunk files whose listing
alone would cost more than the refresh is worth.
"""

from __future__ import annotations

from pathlib import Path

from detail.storage import experiment as experiment_store


def compute_dict(compute) -> dict | None:
    """``compute`` as written to ``job.json``: the shepherd sets a ``ComputeTarget`` before it
    writes the cluster's envelope, and a job's own ``to_dict`` must write that as its dict.
    """
    return compute.to_dict() if hasattr(compute, "to_dict") else compute


class RemoteClusteringFit:
    """The §6 "Remote analysis" hooks of a clustering fit; mixed into its ``AnalysisModule``."""

    remote_capable = True

    def remote_resources(self, job) -> dict:
        """One GPU, the fewest a job may ask for: the work is CPU (see the module docstring)."""
        return {"gpus": 1}

    def refreshed_inputs(self, experiment_root, job) -> list[Path]:
        """Each recording's own embedding run, the fit's input."""
        root = Path(experiment_root)
        return sorted(
            {
                experiment_store.recording_path(root, recording_id)
                / "embeddings"
                / job.embed_run_for(recording_id)
                for recording_id in job.source_recording_ids
            }
        )
