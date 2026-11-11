"""``TemporalClusteringRun`` — the handle a temporal-clustering run is interpreted through (§22).

The API boundary for the authored half (``api/pose_clustering.py`` was its sibling until pose
clustering was removed). The fit is
computed once and immutable; everything here is either a read of it or an edit of the
interpretation on top of it, and every edit funnels through ``_edit`` so the finalize check, the
history entry and the reprojection cannot be forgotten by one of them.

The lifecycle itself is ``AuthoredRun``'s and is not written again here — this supplies the four
hooks and the reads the interpretation surface needs.
"""

from pathlib import Path

from detail.api import authored_run as authored
from detail.api.authored_run import (
    AuthoredRun,
    LabelOperations,
    SnapshotInterpretation,
    SnapshotViewMixin,
    label_interpretation,
)
from detail.api.clip_edits import ClipEditOperations
from detail.api.clip_review import ClipReviewOperations
from detail.api.clustering_nodes import NodeCacheMixin
from detail.api.experiment import Experiment
from detail.configs.temporal_clustering import TemporalClusteringConfig
from detail.core import run_selection
from detail.pipeline import temporal_clustering as temporal_pipeline
from detail.storage import behavior_output, temporal_runs
from detail.storage import snapshots as snapshots_store


class TemporalClusteringRun(
    ClipReviewOperations,
    ClipEditOperations,
    LabelOperations,
    NodeCacheMixin,
    AuthoredRun,
):
    """Handle for a temporal-clustering run.

    ``inner`` (the executor handle) is present while the fit is in flight and ``None`` for a run
    reached later to interpret; every interpretation operation works either way.
    """

    #: The launch config.
    config_model = TemporalClusteringConfig

    def __init__(self, inner, experiment: Experiment, run_id: str):
        super().__init__(inner, experiment, run_id)
        # The loaded fit, warmed on first use. On the *handle* rather than module-level: a handle
        # is what an interpretation surface holds open while someone works, and a process-wide
        # cache would outlive the run it describes. It matters more here than for pose — the fit
        # is a dense K×K matrix, so re-reading it per interaction is both slow and large.
        self._fit = None

    kind = "temporal-clustering run"
    nested_labels = True

    #: ``AuthoredRun``'s writers and this run's own: the clip edits (through ``_edit``) and the
    #: clip review (through ``_record``), each refused on a snapshot view.
    _WRITE_METHODS = (
        *AuthoredRun._WRITE_METHODS,
        "take_out_clips",
        "put_back_clips",
        "finish_review",
    )

    # -- what AuthoredRun needs from this module ------------------------------------------

    snapshot_view_class = None  # set below, once the view class exists

    def snapshot_interpretation(self, record: dict) -> SnapshotInterpretation:
        """A snapshot of this run is the labels it carried (§22)."""
        return label_interpretation(record)

    def _read_state(self) -> temporal_runs.TemporalClusteringState:
        return temporal_runs.read_temporal_state(self._run_dir())

    def _write_state(self, state) -> None:
        temporal_runs.write_temporal_state(self._run_dir(), state)

    def _log(self, entry: dict) -> None:
        temporal_pipeline.append_history(self._run_dir(), entry)

    def _project_frames(self, *, progress=None) -> int:
        return temporal_pipeline.project_frames(
            self._experiment.dir, self._run_dir(), progress=progress
        )

    def _project(self, *, progress=None, reuse: bool = False) -> None:
        temporal_pipeline.project_artifact(
            self._experiment.dir,
            self._run_dir(),
            progress=progress,
            fit=self._fit,
            reuse=reuse,
        )

    # --- reads ---------------------------------------------------------------------------

    def state(self) -> temporal_runs.TemporalClusteringState:
        """The whole authored state: labels, the cache floor, and the fit's resolved inputs."""
        self._require_fit()
        return self._read_state()

    def data_sources(self) -> dict:
        """``{source_recording_ids, embed_run_id, embed_run_ids, track_id, method}``.

        ``embed_run_ids`` is each recording's own embedding run, which is what a reader
        must use (``core.run_selection.run_for_recording``); ``embed_run_id`` is set only
        when every recording's agrees.
        """
        state = self.state()
        return {
            "source_recording_ids": list(state.source_recording_ids),
            "embed_run_id": state.embed_run_id,
            "embed_run_ids": run_selection.runs_by_recording(
                state, state.source_recording_ids
            ),
            "track_id": state.track_id,
            "method": state.method,
        }

    def subsample_note(self) -> str | None:
        """What the fit left out, in a sentence, or ``None`` if it compared every bout.

        §22 requires the interpretation surface to state a subsample. A fit capped by
        ``max_segments`` looks exactly like a full one — same clusters, same table, same
        confident-looking scatter — so this is the only thing standing between a user and a
        vocabulary quietly built on a fraction of their data.
        """
        return self._method().subsample_note(self._load_fit())

    def instances(self) -> list:
        """Every instance the current interpretation yields.

        Reads the projected artifact rather than recomputing: the artifact is what every other
        reader sees, so a browse list built from a second computation could disagree with it.
        """
        return list(self.artifact().instances)

    def artifact(self) -> behavior_output.BehaviorOutput:
        """The §19 behavior output this run currently projects."""
        return behavior_output.read_behavior_output(self._run_dir())

    # --- edits ---------------------------------------------------------------------------

    def _tree(self):
        """This fit's merge tree — the id space names live in."""
        return self._method().hierarchy_of(self._load_fit())

    def _reresolve(self, state) -> dict:
        """Recompute the label resolution; see ``authored_run.reresolve_labels``."""
        return authored.reresolve_labels(
            state, self._tree(), self._load_fit().labels_at
        )

    def hierarchy(self):
        """This fit's merge tree — the :class:`ClusterTree` a surface draws (§22).

        Asked of the handle rather than reached for past it, which is the layering §22 states
        ("the module never looks inside a fit"). ``node_view`` is what a surface reads a node
        through; this is the bare tree it indexes.
        """
        return self._tree()

    def export(self, out_dir: Path | str, *, progress=None) -> Path:
        """Write one per-recording behavior-stream CSV into ``out_dir``.

        §19's deliberate, user-initiated re-ingest: the artifact is never auto-written into a
        recording's ``streams/``. Each row carries ``origin: predicted``, so re-importing one can
        never read back as ground truth.

        The artifact is **pinned first** and the CSVs are written from the pinned copy (§21), so
        an exported file always corresponds to a snapshot still on disk: reopening the run and
        re-cutting it afterwards cannot leave the export describing something unfindable. Pinning
        brings deferred frame views current, the one long step, reported through ``progress``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pinned = self.pin(reason="export", progress=progress)
        output = self.artifact() if pinned is None else self.snapshot_artifact(pinned)
        for recording_id in self.state().source_recording_ids:
            behavior_output.write_stream_csv(
                out_dir / f"{recording_id}.csv", output, recording_id
            )
        return out_dir

    # --- internals -----------------------------------------------------------------------

    def _state_file(self) -> Path:
        return temporal_runs.temporal_yaml_path(self._run_dir())

    def _fit_path(self) -> Path:
        return temporal_runs.fit_path(self._run_dir(), self._method().shard_suffix)

    def _member_resolver(self, method, fit):
        return temporal_pipeline.member_resolver(method, fit)

    def _require_fit(self) -> None:
        if not temporal_runs.has_temporal_state(self._run_dir()):
            raise FileNotFoundError(
                f"analysis run {self.id} has no temporal-clustering state; it is either a "
                "different kind of run or its fit did not finish"
            )

    def _method(self):
        return temporal_pipeline.resolve_method(self.state().method)

    def _load_fit(self):
        if self._fit is None:
            method = temporal_pipeline.resolve_method(
                temporal_runs.read_temporal_state(self._run_dir()).method
            )
            self._fit = method.load(
                temporal_runs.fit_path(self._run_dir(), method.shard_suffix)
            )
        return self._fit


class TemporalClusteringSnapshotView(SnapshotViewMixin, TemporalClusteringRun):
    """One pinned snapshot of a temporal-clustering run, read as the interpretation it was.

    The class is the run's own, so the scatter, the ethograms, the member strip and the instance
    list are computed by exactly the code that computes them live — a second implementation is
    what would let a frozen result and a live one drift apart while both look right.
    """

    _state_class = temporal_runs.TemporalClusteringState


TemporalClusteringRun.snapshot_view_class = TemporalClusteringSnapshotView
