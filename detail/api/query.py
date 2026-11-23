"""``QueryRun`` — the API seam (Tenet 2) for a §20 cohort-search run.

A Query run has two halves and they have opposite costs. **Search** is computed, expensive
and immutable: a pure function of (exemplars, method, params, embeddings), stored as opaque
per-unit shards. **Calibration** is authored, cheap and instant: every operation here is a
rewrite of ``query.yaml`` followed by a reprojection of the §19 artifact from search results
already on disk. Nothing recomputes, which is the point — a change of mind about a cutoff must
not cost another search.

The run's own lifecycle status (``open`` while the user calibrates, ``complete`` once they say
they are done) lives in ``analysis.yaml`` and is authoritative; the inner executor handle's
status reflects only the search job. The two are orthogonal (§21), which is why a run that
searched perfectly reads ``open`` rather than ``complete``.
"""

from pathlib import Path

from detail.api import exporters
from detail.api.authored_run import (
    AuthoredRun,
    SnapshotInterpretation,
    SnapshotViewMixin,
)
from detail.api.experiment import Experiment
from detail.api.query_review import QueryReviewOperations
from detail.core.errors import UserInputError
from detail.pipeline import query as query_pipeline
from detail.storage import analysis_runs as analysis_store
from detail.storage import behavior_output
from detail.storage import query_runs
from detail.storage import recording as recording_store
from detail.storage import snapshots as snapshots_store


class QueryRun(QueryReviewOperations, AuthoredRun):
    """Handle for a Query run. ``inner`` (the executor handle) is present while the search is
    in flight and ``None`` for a run reached later to calibrate; every calibration operation
    works either way."""

    #: ``AuthoredRun``'s writers and this run's own: the review's removals (through ``_edit``)
    #: and its census (through ``_record``), each refused on a snapshot view.
    _WRITE_METHODS = (
        *AuthoredRun._WRITE_METHODS,
        "take_out_clips",
        "put_back_clips",
        "finish_review",
    )

    def __init__(self, inner, experiment: Experiment, run_id: str):
        super().__init__(inner, experiment, run_id)
        # Loaded search results per behavior, warmed by `load_results`. On the *handle*
        # rather than module-level: a handle is what a calibration surface holds open while
        # someone works, and a process-wide cache would outlive the run it describes.
        self._results: dict[str, dict[tuple[str, int], object]] = {}
        # Each behavior's candidates at no cutoff (the audit's population), and each searched
        # unit's timeline they are placed on. Same lifetime as `_results`, for the same reason:
        # every audit figure on a surface reads them, and they move only when the search does.
        self._pools: dict[str, list] = {}
        self._timelines: dict[tuple[str, int], tuple] = {}

    kind = "query run"

    # -- what AuthoredRun needs from this module ------------------------------------------

    def snapshot_interpretation(self, record: dict) -> SnapshotInterpretation:
        """A snapshot of this run is its behaviors and the cutoff each was calibrated to (§20).

        A cutoff is what a Query run authors, the way a granularity is what a clustering run
        authors — "which cutoff did I ship?" is the question a pinned result has to answer. A
        behavior with `cutoff: None` has been searched and not calibrated, which is why the count
        of calibrated ones is its own column rather than folded into the total.
        """
        state = record.get("state") or {}
        behaviors = list(state.get("behaviors") or [])
        names = {
            str(b.get("id", i)): str(b.get("name") or "")
            for i, b in enumerate(behaviors)
            if b.get("name")
        }
        calibrated = [b for b in behaviors if b.get("cutoff") is not None]
        cutoffs = ", ".join(f"{float(b['cutoff']):.2f}" for b in calibrated[:4])
        vocabulary = snapshots_store.vocabulary(names)
        return SnapshotInterpretation(
            columns=("Behaviors", "Calibrated", "Cutoffs", "Vocabulary"),
            cells=(
                str(len(behaviors)),
                str(len(calibrated)),
                cutoffs or "—",
                vocabulary,
            ),
            headline=(
                f"{len(behaviors)} behaviors · {len(calibrated)} calibrated · {vocabulary}"
            ),
        )

    def _refuse_if_incomplete(self, op: str) -> None:
        """A search that did not reach every unit refuses to be finalized or exported, naming them.

        Two ways a unit goes unsearched: the search was cancelled before it (``partial``) or one
        behavior's search raised on it (``open``, with the behavior in ``behaviors_failed``).
        Either way the unit has no shard, and an export would write it as a header-only CSV —
        "searched, found nothing" — which is the one thing it is not.
        """
        record = analysis_store.read_analysis_yaml(self._run_dir())
        status = record.get("status", "open")
        failed = record.get("behaviors_failed") or {}
        if status not in self._INCOMPLETE_STATUSES and not failed:
            return
        missing = query_pipeline.missing_units(self._run_dir())
        detail = "; ".join(
            f"{behavior_id} on {', '.join(recordings)}"
            for behavior_id, recordings in sorted(missing.items())
        )
        reasons = "; ".join(f"{b}: {why}" for b, why in sorted(failed.items()))
        raise ValueError(
            f"{self.kind} {self.id} did not search every recording it was asked to "
            f"(status {status!r}), so it will not {op}. Missing: "
            f"{detail or 'units whose search failed'}"
            f"{f' ({reasons})' if reasons else ''}. Calibrating what was searched still works; "
            "re-run the search to cover the rest."
        )

    def _read_state(self) -> query_runs.QueryState:
        return query_runs.read_query_state(self._run_dir())

    def _write_state(self, state) -> None:
        query_runs.write_query_state(self._run_dir(), state)

    def _method(self):
        """The query method that wrote this run's search.

        Raises:
            UserInputError: this version no longer has it (``docs/removed_capabilities.md``),
                naming the method. Its search cannot be read — not calibrated, reviewed or
                audited — and saying so beats the bare registry error, which reached the CLI as a
                traceback.
        """
        from detail.core.registries import QUERY_METHODS
        from detail.query_methods import import_query_methods

        method_id = self.state().method
        import_query_methods()
        if method_id not in QUERY_METHODS:
            raise UserInputError(
                f"query run {self.id!r} was searched with the {method_id!r} method, which this "
                "version no longer has, so its search cannot be calibrated, reviewed or "
                "audited. The run record, its behavior output and its output folder are kept"
            )
        return query_pipeline.resolve_method(method_id)

    def _edit(self, op: str, mutate, *, reproject: bool = True, progress=None):
        """``AuthoredRun._edit``, refused first when the run's method is gone: every query edit
        reprojects through it, and finding that out inside the reprojection left the edit saved
        over results that were never rebuilt (#24 — a cutoff, a rename, a clip put back).
        """
        self._method()
        return super()._edit(op, mutate, reproject=reproject, progress=progress)

    def _log(self, entry: dict) -> None:
        query_pipeline.append_history(self._run_dir(), entry)

    def _project_frames(self, *, progress=None) -> int:
        return query_pipeline.project_frames(
            self._experiment.dir, self._run_dir(), progress=progress
        )

    def _project(self, *, progress=None, reuse: bool = False) -> None:
        # Handed whatever results this handle has already warmed, so a surface that loaded them
        # to draw a curve does not pay for them again on every commit.
        query_pipeline.project_artifact(
            self._experiment.dir,
            self._run_dir(),
            progress=progress,
            results=self._results or None,
            reuse=reuse,
        )

    # --- authored state (reads) ----------------------------------------------------------

    def state(self) -> query_runs.QueryState:
        """The whole authored state — behaviors, method, searched recordings.

        Through ``_read_state`` rather than reading the file, because that is the seam a snapshot
        view replaces (§21). Reading the run directory here bypassed it, so a pinned calibration
        rendered the *live* cutoffs and names under a banner saying "Read-only" — the frozen
        instance list beside them making it look internally consistent.
        """
        return self._read_state()

    def behaviors(self) -> list[query_runs.Behavior]:
        """The run's behaviors, each with its id, name, example clips and cutoff."""
        return self.state().behaviors

    def data_sources(self) -> dict:
        """``{search_recording_ids, embed_run_id, track_id, method}`` for this run."""
        state = self.state()
        return {
            "search_recording_ids": list(state.search_recording_ids),
            "embed_run_id": state.embed_run_id,
            "track_id": state.track_id,
            "method": state.method,
        }

    def load_results(self, behavior_id: str, progress=None) -> dict:
        """Load this behavior's search results and extract their candidates, once.

        Returns ``{(recording_id, subject): SearchResult}`` for the units whose search
        completed; a unit whose shard is absent (its search failed or was interrupted) is
        simply not in the map, which is what lets the rest still be calibrated.

        **The extraction is forced here, not left to the first reader.** A method caches its
        candidate list on the result object, so whoever touches it first pays for the whole
        thing — and that is the one genuinely sized step in calibration. Doing it behind a
        named call with a ``progress`` callback is what lets a surface show a bar instead of
        freezing on whichever read happened to come first.

        ``progress(stage, fraction, detail)`` matches the synchronous long-op shape the CLI's
        ``_StageProgress`` and the GUI's ``run_progress`` both drive.
        """
        if behavior_id in self._results:
            return self._results[behavior_id]
        state = self.state()
        state.behavior(behavior_id)  # raises, naming the ids that exist
        method = self._method()
        recordings = list(state.search_recording_ids)
        loaded: dict[tuple[str, int], object] = {}
        for i, recording_id in enumerate(recordings):
            if progress is not None:
                progress("loading", i / max(len(recordings), 1), recording_id)
            path = query_runs.shard_path(
                self._run_dir(),
                behavior_id,
                recording_id,
                state.track_id,
                method.shard_suffix,
            )
            if not path.is_file():
                continue
            result = method.load(path)
            # Force the extraction while the caller is watching a bar, rather than on
            # whichever read touches it first.
            method.instances_at(result, 0.0, state.method_params)
            loaded[(recording_id, state.track_id)] = result
        if progress is not None:
            progress("loading", 1.0, "")
        self._results[behavior_id] = loaded
        return loaded

    def candidates_at(self, behavior_id: str, cutoff: float) -> dict:
        """``{(recording_id, subject): [QueryCandidate]}`` at ``cutoff`` — **without**
        committing it.

        What a calibration surface shows while the user is still deciding: the matches a
        setting would yield, read off the warmed results rather than by reprojecting. Keeping
        this on the handle is what stops a view from having to know where a shard lives or how
        a method is resolved.
        """
        state = self.state()
        method = self._method()
        return {
            unit: method.instances_at(result, float(cutoff), state.method_params)
            for unit, result in self.load_results(behavior_id).items()
        }

    def score_distribution(self, behavior_id: str, *, bins: int = 20) -> dict:
        """``{recording_id: (counts, edges)}`` over every candidate's confidence.

        §20 requires the per-recording distributions to be **always visible**, because a
        single cutoff that keeps most of one animal's behavior and little of another's biases
        the result along the very axis being compared — and a pooled curve cannot show it.
        Histograms of the candidates, not of the instances at some cutoff: the point is to see
        where each recording's mass sits before choosing one.

        Every searched recording gets an entry, including one whose search returned nothing —
        an absent row would read as "not searched", which is the distinction coverage exists
        to preserve.
        """
        import numpy as np

        edges = np.linspace(0.0, 1.0, bins + 1)
        out: dict[str, tuple] = {}
        for recording_id in self.state().search_recording_ids:
            out[recording_id] = (np.zeros(bins, dtype=np.int64), edges)
        for (recording_id, _subject), candidates in self.candidates_at(
            behavior_id, 0.0
        ).items():
            scores = np.array([c.confidence for c in candidates], dtype=float)
            counts, _ = np.histogram(scores, bins=edges)
            out[recording_id] = (counts, edges)
        return out

    def recording_duration_ns(self, recording_id: str) -> int:
        """Full video duration of a searched recording, in nanoseconds.

        The x-axis extent a timeline is drawn against, so a mark's position reflects where in
        the whole recording it falls rather than where it falls among the matches.
        """
        from detail.storage import experiment as experiment_store
        from detail.storage import recording as recording_store

        data = recording_store.read_recording(
            experiment_store.recording_path(self._experiment.dir, recording_id)
        )
        return int(float(data["video"]["total_duration_s"]) * 1e9)

    def frequency_curve(
        self, behavior_id: str, *, steps: int = 41, progress=None
    ) -> list[dict]:
        """How many instances each cutoff returns — the calibration view's whole basis.

        Without ground truth there is no score to optimize, so choosing a cutoff means looking
        at what each setting returns. This is that, as data: ``[{cutoff, n_instances, capped},
        ...]`` over ``steps`` evenly spaced cutoffs in ``[0, 1]``.

        **Instance count is not guaranteed monotonic** in the cutoff (§20): a method that
        re-forms extents can split one instance into two as the cutoff tightens. So this
        reports what it measures and no caller may assume a shape.

        ``capped`` says the method's own candidate limit bound that point. §20 requires it:
        "a curve that flattens because storage ran out looks exactly like a curve that
        flattens because the animal stopped."

        Cheap because ``instances_at`` is a slice of one cached extraction rather than a fresh
        search — before that it was quadratic per point and a slider drag cost minutes.

        Reads the results ``load_results`` warmed, so building the curve a second time (a
        second behavior, a redraw, a reopened view) costs no disk and no extraction.
        """
        state = self.state()
        method = self._method()
        cap = method.candidate_cap(state.method_params)
        results = list(self.load_results(behavior_id, progress=progress).values())

        curve: list[dict] = []
        for i in range(steps):
            cutoff = i / (steps - 1) if steps > 1 else 0.0
            counts = [
                len(method.instances_at(result, cutoff, state.method_params))
                for result in results
            ]
            curve.append(
                {
                    "cutoff": cutoff,
                    "n_instances": sum(counts),
                    "capped": cap is not None and any(n >= cap for n in counts),
                }
            )
        return curve

    def artifact(self) -> behavior_output.BehaviorOutput:
        """The §19 behavior artifact this run currently states.

        Raises:
            FileNotFoundError: If the search never completed, so there is nothing to state.
        """
        return behavior_output.read_behavior_output(self._run_dir())

    def _audit_pool(self, label: str | None):
        """Every candidate the search found, **at no cutoff at all** (§19/§20).

        Not ``instances.csv``. The artifact holds only what the current cutoff kept
        (``pipeline/query.py``'s projection), so sampling it would make the audit a function of
        a decision the user is still making: every adjustment would invalidate part of the
        evidence, and the flow would degenerate into set-cutoff, audit, re-set, re-audit.

        Enumerating at zero instead is what buys the property the audit exists for — **one
        audit serves every cutoff**. It is affordable because a cutoff-free set is small (583
        candidates for soft-kNN, and 716 for the since-removed dtw, on a 30-minute recording) and
        because ``instance_id`` is a content hash of the span, so a decision recorded here still
        attaches to its instance after the cutoff moves and the artifact is rebuilt.
        """
        from detail.api.run_preview import load_track_trajectory
        from detail.storage import experiment as experiment_store

        state = self.state()
        method = self._method()
        pool: list = []
        for behavior in state.behaviors:
            if label is not None and behavior.id != label:
                continue
            if behavior.id in self._pools:
                pool.extend(self._pools[behavior.id])
                continue
            mine: list = []
            for (recording_id, subject), result in self.load_results(
                behavior.id
            ).items():
                if (recording_id, subject) not in self._timelines:
                    recording_dir = experiment_store.recording_path(
                        self._experiment.dir, recording_id
                    )
                    self._require_searched_data(recording_dir, recording_id, state)
                    _emb, frame_idx, t_ns = load_track_trajectory(
                        recording_dir, state.embed_run_id, subject
                    )
                    self._timelines[(recording_id, subject)] = (t_ns, frame_idx)
                t_ns, frame_idx = self._timelines[(recording_id, subject)]
                mine.extend(
                    query_pipeline.instances_from(
                        method.instances_at(result, 0.0, state.method_params),
                        recording_id=recording_id,
                        subject=subject,
                        behavior_id=behavior.id,
                        t_ns=t_ns,
                        frame_idx=frame_idx,
                    )
                )
            self._pools[behavior.id] = mine
            pool.extend(mine)
        return pool

    def _require_searched_data(
        self, recording_dir: Path, recording_id: str, state
    ) -> None:
        """Refuse, naming it, a recording the search read whose data has since been deleted.

        The candidates are re-read from the recording's embedding, so a recording deleted
        after the search (or its embedding run) leaves nothing to read them from.

        Raises:
            UserInputError: the recording, or its embedding run, is no longer on disk.
        """
        what = None
        if not recording_dir.is_dir():
            what = f"recording {recording_id!r} it searched was deleted"
        elif not (recording_dir / "embeddings" / state.embed_run_id).is_dir():
            name = recording_store.read_recording(recording_dir).get(
                "name", recording_id
            )
            what = (
                f"embedding run {state.embed_run_id!r} of recording {name!r}, which it "
                "searched, was deleted"
            )
        if what is not None:
            raise UserInputError(
                f"query run {self.id!r} cannot read its candidates: the {what}. Run the "
                "query again over the recordings that are left"
            )

    def _audit_examples(self, rows) -> set[str]:
        """The candidates the user's own example or non-example clips overlap (§19).

        Those clips *defined* the behavior, so a candidate on top of one is the method finding
        its own input: judging it scores the answer key, and it is near-certain to be judged a
        yes. They stay in the run's counts and the export, marked ``is_example``; the audit
        never draws them and never counts a judgment of one.
        """
        return query_runs.example_instance_ids(self.state().behaviors, rows)

    def _audit_edges(self, labels) -> dict:
        """``{behavior id: edges}`` — each behavior's audit bins, from its candidates at no cutoff.

        Read from ``audit.yaml`` where the behavior has recorded bins (the earlier one-by-one
        audit recorded them at its first draw), else derived — the same edges, because they are
        a pure function of the candidates and the search results never change.
        """
        from detail.storage import behavior_audit as audit

        labels = sorted(str(label) for label in labels)
        stored = audit.read_bins(self._audit_dir())
        edges = {
            label: (
                stored[label]
                if label in stored
                else audit.bin_edges(
                    [
                        row.score
                        for row in behavior_output.bouts(self._audit_pool(label))
                        if row.score is not None
                    ]
                )
            )
            for label in labels
        }
        return edges

    def _audit_rows(self, label: str | None, cutoff: float | None):
        """Every behavior's candidates at no cutoff that its cutoff keeps — ``cutoff`` when one is
        given, else the behavior's own — and nothing left for storage to cut.

        Not the artifact. The estimate weights each score bin by how many instances the cutoff
        keeps in it, and a bin's instances below the committed cutoff are not in the artifact at
        all, so a cutoff lower than the committed one had no population to weight. For both
        query methods the candidates a cutoff keeps *are* the artifact's instances at it
        (``instances_at`` is a filter of one extraction), so nothing else moves. A behavior with
        no cutoff and none given is absent, as it is from the artifact.
        """
        from detail.storage import behavior_audit as audit

        direction = audit.score_direction(self.artifact().manifest)
        rows: list = []
        for behavior in self.state().behaviors:
            if label is not None and behavior.id != label:
                continue
            effective = behavior.cutoff if cutoff is None else float(cutoff)
            if effective is None:
                continue
            rows.extend(
                row
                for row in self._audit_pool(behavior.id)
                if audit.retained(row.score, effective, direction)
            )
        return rows, None

    # --- authored state (writes) ---------------------------------------------------------

    def set_cutoff(
        self, behavior_id: str, cutoff: float | None, *, progress=None
    ) -> "QueryRun":
        """Set (or clear, with ``None``) one behavior's confidence cutoff, and reproject.

        A cutoff of ``None`` is not zero. An uncalibrated behavior is **absent** from the
        artifact rather than present with no rows, because §19 reads a vocabulary entry with
        no rows inside coverage as a real zero — "we looked and it never happened" — which is
        a different claim from "nobody has chosen a threshold yet".

        Reprojects, which reads a shard and rewrites a frame-view zarr per searched unit —
        real work on a cohort, so ``progress(stage, fraction, detail)`` is driven throughout
        and every caller is expected to render it.

        Raises:
            ValueError: If the run is finalized, or the cutoff is outside ``[0, 1]``.
            KeyError: If no behavior carries that id.
        """
        if cutoff is not None and not 0.0 <= float(cutoff) <= 1.0:
            raise ValueError(
                f"cutoff must be a confidence in [0, 1], got {cutoff!r}. Confidence is "
                "canonical across query methods (§20), so this axis is method-independent."
            )

        def mutate(state: query_runs.QueryState) -> dict:
            behavior = state.behavior(behavior_id)
            behavior.cutoff = None if cutoff is None else float(cutoff)
            return {"behavior_id": behavior_id, "cutoff": behavior.cutoff}

        return self._edit("set_cutoff", mutate, progress=progress)

    def name_behavior(
        self, behavior_id: str, name: str, *, progress=None
    ) -> "QueryRun":
        """Rename one behavior, and reproject.

        The **id** never moves (§19 hashes it into every ``instance_id``, which audit decisions
        are keyed by), so a rename re-points the label and leaves every recorded judgment
        attached to what it was made about.

        Raises:
            ValueError: If the run is finalized, the name is blank, or another behavior in
                this run already carries it.
            KeyError: If no behavior carries that id.
        """
        cleaned = str(name).strip()
        if not cleaned:
            raise ValueError("a behavior name cannot be blank")

        def mutate(state: query_runs.QueryState) -> dict:
            behavior = state.behavior(behavior_id)
            clash = [
                b for b in state.behaviors if b.id != behavior_id and b.name == cleaned
            ]
            if clash:
                raise ValueError(
                    f"another behavior in this run is already called {cleaned!r} "
                    f"({clash[0].id}); names are what the artifact's vocabulary carries, so "
                    "two behaviors sharing one would pool into a single label"
                )
            behavior.name = cleaned
            return {"behavior_id": behavior_id, "name": cleaned}

        return self._edit("name_behavior", mutate, progress=progress)

    def export(self, out_dir: Path | str, *, progress=None) -> Path:
        """Write one per-recording behavior-stream CSV into a new folder under ``out_dir``;
        return that folder. Each export gets its own, so a second one never replaces the first.

        §19's deliberate, user-initiated re-ingest: the artifact is never auto-written into a
        recording's ``streams/``, because keeping the prediction/ground-truth boundary
        structural is what makes the leakage rule enforceable rather than remembered. Each row
        carries ``origin: predicted``, so re-importing one can never read back as ground truth.

        One CSV per searched recording, including recordings with no instances — an empty file
        is the honest statement for a recording that was searched and matched nothing.

        Raises:
            FileNotFoundError: If the run carries no artifact yet.
            ValueError: If the search did not reach every unit (cancelled, or a behavior's
                search failed on one) — the message names them; re-run the search.
            UserInputError: If no behavior has a cutoff yet (nothing is calibrated).

        The artifact is **pinned first** and the CSVs are written from the pinned copy (§21), so
        an exported file always corresponds to a snapshot still on disk: reopening the run and
        re-cutting it afterwards cannot leave the export describing something unfindable. Pinning
        brings deferred frame views current, the one long step, reported through ``progress``.
        """
        self._refuse_if_incomplete("be exported")
        if not any(b.cutoff is not None for b in self.state().behaviors):
            # A header-only file reads as "searched and matched nothing" (`docs/cli.md`), and
            # here nothing was calibrated: no behavior has a cutoff to cut the search at.
            raise UserInputError(
                "nothing is calibrated yet, so there is nothing to export: set a cutoff for "
                "at least one behavior (`detail analysis query-calibrate --cutoff q01=0.8`, "
                "or Calibrate in the app) first"
            )
        pinned = self.pin(reason="export", progress=progress)
        output = self.artifact() if pinned is None else self.snapshot_artifact(pinned)
        out_dir = exporters.new_export_folder(
            exporters.ExportSubject("analysis_run", self._experiment.dir, self.id),
            out_dir,
        )
        for recording_id in self.state().search_recording_ids:
            behavior_output.write_stream_csv(
                out_dir / f"{recording_id}.csv", output, recording_id
            )
        return out_dir


class QuerySnapshotView(SnapshotViewMixin, QueryRun):
    """One pinned snapshot of a Query run, read as the calibration it was (§21).

    A Query snapshot's interpretation is its behaviors and their cutoffs — "which cutoff did I
    ship?" is the reproducibility question a pinned search has to answer, and it is as much an
    interpretation as a granularity is. This was left undeclared at first on the grounds that a
    strip of clips is not a vocabulary; the cutoffs are the part that judgment missed.
    """

    _state_class = query_runs.QueryState


QueryRun.snapshot_view_class = QuerySnapshotView
