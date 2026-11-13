"""The node-cache half of a clustering run handle, shared by pose and temporal (§22).

Both modules' handles read their nodes the same way and rebuild the cache the same way; only
where the fit lives and how a ranked member is placed differ, and those two come from the module
(``_fit_path``, ``_member_resolver``). One mixin rather than two copies, for the reason
``authored_run`` gives: the sequence is the point, and a copy is where a step gets dropped.
"""

from pathlib import Path

from detail.core.utils import subject_key


class NodeCacheMixin:
    """``node_view()`` and ``rebuild_node_cache()`` for a clustering run handle."""

    _node_view = None
    #: The CLI command that interprets (and rebuilds) this kind of run, for messages.
    _interpret_command = "temporal-interpret"

    def _fit_path(self) -> Path:
        raise NotImplementedError

    def _member_resolver(self, method, fit):
        raise NotImplementedError

    def check_member(
        self,
        node: int,
        recording_id: str,
        subject: int,
        start_ns: int,
        end_ns: int,
        verdict: str,
    ):
        """Record whether one clip of ``node`` fits it: ``verdict`` is ``"fits"``,
        ``"doesnt"`` or ``"unclear"`` (§22 *Clip checks*). Appends; a later answer for the same
        clip supersedes this one. Evidence about the clustering, never an input to it.

        Raises:
            ValueError: an unknown ``verdict``, or the run is finalized (as the audit is
                refused; ``reopen()`` first).
            ReadOnlyRunError: the run is a read-only view.
        """
        from detail.storage import member_checks

        if verdict not in member_checks.VERDICTS:
            raise ValueError(
                f"verdict must be one of {member_checks.VERDICTS}, not {verdict!r}"
            )
        key = member_checks.ClipKey(
            int(node),
            str(recording_id),
            subject_key(subject),
            int(start_ns),
            int(end_ns),
        )
        self._record(
            "check_member",
            {
                "node": key.node,
                "recording_id": key.recording_id,
                "subject": key.subject,
                "start_ns": key.start_ns,
                "end_ns": key.end_ns,
                "verdict": verdict,
            },
        )
        member_checks.append_check(self._run_dir(), key, verdict)
        return self

    def member_checks(self, node: int):
        """``node``'s clip checks: counts, the latest answer per clip, and the estimate with
        its 95% interval (``storage.member_checks.CheckSummary``)."""
        from detail.storage import member_checks

        return member_checks.summary(self._run_dir(), int(node))

    def has_node_cache(self) -> bool:
        from detail.storage import node_cache as store

        return store.has_node_cache(self._run_dir())

    def node_names(self) -> dict[str, str]:
        """``{node id: label name}`` to show, legacy names included, without writing anything.

        A run written before labels lived on nodes keeps them in ``names_by_granularity`` until
        its first edit folds them. Reading only ``node_names`` opened such a run unlabelled while
        BROWSE still showed the old names. This folds them on a copy (loading the fit only when
        there is anything to fold), so every surface shows them from the first open.
        """
        import copy

        from detail.api import authored_run as authored

        state = self.state()
        if not state.names_by_granularity:
            return {k: v for k, v in state.node_names.items() if v}
        folded = copy.deepcopy(state)
        authored.fold_legacy_names(folded, self._tree(), self._load_fit().labels_at)
        return {k: v for k, v in folded.node_names.items() if v}

    def node_view(self):
        """This run's nodes, read once from its node cache and kept on the handle.

        Raises ``FileNotFoundError`` for a run with no cache (one fitted before it existed);
        a surface offers :meth:`rebuild_node_cache` for that, rather than a blank screen.
        """
        from detail.api.node_view import NodeView

        if self._node_view is None:
            from detail.api.node_cache import current_subject, fit_digest

            view = NodeView.open(
                self._run_dir(),
                self._tree(),
                recording_names=self._experiment.recording_names(),
            )
            # §22: a cache built from another fit is reported, never silently read as this one's.
            try:
                view.stale = view.stale_reason(
                    {
                        **fit_digest(self._fit_path()),
                        **current_subject(self.state().track_id),
                    }
                )
            except FileNotFoundError:
                view.stale = None
            self._node_view = view
            self._clip_edit_stamp = None
        self._sync_clip_edits(self._node_view)
        return self._node_view

    def _state_file(self) -> Path | None:
        """The authored state file whose clip edits the view follows, or ``None`` for a run whose
        state has none (a saved pose-clustering run: the module is removed and took no clip edits).
        """
        return None

    def _sync_clip_edits(self, view) -> None:
        """Hand the view the state's clip edits when they may have changed since it last took
        them: the state file's time and size for a live run, once for a snapshot (frozen).
        """
        path = self._state_file()
        if path is None:
            return
        if not self.editable:
            stamp = ("snapshot",)
        else:
            try:
                stat = path.stat()
            except FileNotFoundError:
                return
            stamp = (stat.st_mtime_ns, stat.st_size)
        if getattr(self, "_clip_edit_stamp", None) == stamp:
            return
        state = self._read_state()
        view.set_clip_edits(getattr(state, "clip_edits", None) or [], state.labels)
        self._clip_edit_stamp = stamp

    def clips(self, node: int, *, progress=None):
        """Every clip of ``node`` (``NodeView.clips``), with its typicality filled in.

        From the node cache alone for a method whose typicality sums over leaf runs. A method
        ranked on demand (``NodeView.ranks_on_demand``: a medoid order) is ranked here on first
        ask, from the fit, by its own ``rank_members``, and kept on the handle per node, so the
        next ask is free; that path is slow on a large node and belongs off the UI thread.
        ``progress(stage, fraction, detail)`` reports it; a cached answer reports nothing.

        Raises:
            FileNotFoundError: the run has no node cache (one fitted before the cache existed),
                naming the rebuild that makes one.
        """
        node = int(node)
        try:
            view = self.node_view()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{self.id} has no node cache, so its clusters list no clips yet. Build it "
                f"with `detail analysis {self._interpret_command} --run {self.id} "
                "--rebuild-cache`, or Rebuild clips on its INTERPRET tab."
            ) from exc
        clips = view.clips(node)
        if not view.ranks_on_demand or not len(clips):
            return clips
        memo = self.__dict__.setdefault("_clip_ranks", {})
        if node not in memo:
            memo[node] = self._rank_on_demand(view, node, clips, progress)
        return clips.with_typicality(memo[node])

    def _rank_on_demand(self, view, node: int, clips, progress=None):
        """``(C,)`` typicality for ``clips`` from ``rank_members``, matched by where each
        ranked member starts; a clip the ranking does not reach (a bout assigned after the fit,
        never compared) stays ``NaN``."""
        import numpy as np

        def report(detail: str, fraction: float) -> None:
            if progress is not None:
                progress("ranking clips", fraction, detail)

        report("loading the fit", 0.0)
        method, fit = self._method(), self._load_fit()
        report("ranking members", 0.5)
        ranked = method.rank_members(fit, view.leaves_under(node))
        resolve = self._member_resolver(method, fit)
        index = {r: i for i, r in enumerate(view.recording_ids)}
        score: dict[tuple[int, int, int], float] = {}
        for instance, distance in ranked:
            where = resolve(instance)
            if where is None:
                continue
            recording_id, subject, start, _end = where
            key = (index.get(recording_id, -1), subject_key(subject), int(start))
            score.setdefault(key, float(distance))
        report("done", 1.0)
        return np.array(
            [
                score.get((int(r), int(s), int(t)), np.nan)
                for r, s, t in zip(clips.recording, clips.subject, clips.start_ns)
            ],
            dtype=np.float64,
        )

    def typical_members(
        self, node: int, limit: int | None = None, *, progress=None
    ) -> list[dict]:
        """``node``'s clips as rows, most typical first (``ClipTable.rows``), at most ``limit``.

        What the clip window and the montage take. Every node with members has clips, however
        small (§22, *Preview clips for every node*). ``progress`` reports a method ranked on
        demand (``clips``).

        Rows are ``{recording_id, subject, start_t_ns, end_t_ns, frame_ns, jpeg, cx, cy,
        typicality}``; ``jpeg`` is ``b""``, since nothing is decoded here.
        """
        clips = self.clips(int(node), progress=progress)
        order = clips.order("typical")
        if limit is not None:
            order = order[: int(limit)]
        return clips.rows(order)

    def _audit_clusters(self, rows) -> dict | None:
        """``{instance_id: node}`` — the labelled cluster holding most of each instance (§19).

        From the node cache, because an instance reports a *label* and the clusters of one label
        fuse: which cluster it came from is a question about its time, which the cache's leaf runs
        answer. ``None`` without a cache, so the audit falls back to drawing by label and says
        per-cluster figures need one; an instance the cache cannot place is left out.
        """
        from detail.storage import behavior_output

        if not self.has_node_cache():
            return None
        try:
            view = self.node_view()
        except FileNotFoundError:
            return None
        labelled = {int(node) for node in self.state().node_labels}
        if not labelled:
            return {}
        bouts = behavior_output.bouts(list(rows))
        owners = view.cluster_of_spans(
            [(r.recording_id, r.subject, r.t_ns, r.end_t_ns) for r in bouts], labelled
        )
        return {r.instance_id: o for r, o in zip(bouts, owners) if o is not None}

    def _clip_trajectories(self, method, fit, progress=None):
        """The fit's own trajectories, for a method whose clips are scored from every member's
        vector (``ClipScoring.kind == "centroid"``), or ``None`` when the method needs none.
        A module's handle overrides this where its method needs them."""
        return None

    def rebuild_node_cache(self, *, progress=None):
        """Rebuild the node cache from this run's fit (a stale cache, or one from before clips).

        Not an authored edit: it changes what the views can show, never what the run says, so it
        is allowed on a finished run. It is refused on a snapshot view, which writes nothing. It
        holds the run's edit lock throughout.
        """
        from detail.api import authored_run
        from detail.api import node_cache

        self._refuse_if_read_only("rebuild node cache")
        with authored_run._edit_lock(self._run_dir()):
            method, fit = self._method(), self._load_fit()
            state = self._read_state()
            node_cache.write_for_run(
                self._experiment.dir,
                self._run_dir(),
                method=method,
                fit=fit,
                fit_path=self._fit_path(),
                params=dict(state.method_params),
                embed_run_for=state.embed_run_for,
                subject=state.track_id,
                trajectories=self._clip_trajectories(method, fit, progress),
                progress=progress,
            )
            self._log({"op": "rebuild node cache"})
        self._node_view = None
        self.__dict__.pop("_clip_ranks", None)
        return self
