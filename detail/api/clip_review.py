"""Precision of a clustering run's labels, from reviewing every clip of each cluster (§19/§22).

The API half of ``storage/clip_review.py``. **The population is the clustering's**: a label's clips
are the leaf runs of the clusters the labels give it (``NodeView.clips``, the montage clips), each
counted under its own cluster — its deepest labelled node.

**A cluster is reviewed whole.** The reviewer watches every clip of one cluster, takes out the
ones that are not its label (``take_out_clips``, which writes as it goes) and then calls
:meth:`finish_review`, which writes a census: ``no`` for each clip taken out of the label or moved
to another, ``yes`` for every other clip, in one append. The cluster's precision is then exact —
a count, ``remaining == 0``, which ``storage/audit_figures`` reads as "all audited". **Why a
census and not a sample:** the user decided every clip is reviewed (review round 11); a random draw
of 25 left most clips unseen, and a clip the reviewer has not watched cannot be taken out.

- **One label** (``path``) covers its own clusters and every cluster under it: a parent's figure
  combines its own clips with its sub-labels' by size (``behavior_audit.stratified_estimate``,
  each cluster a stratum of its clip count). With every cluster reviewed it is exact.
- **A label whose clusters changed resets.** An answer counts only when it was written at or after
  its cluster's label's ``membership_changed_at``, so choosing clusters by how they scored cannot
  raise the score the choice was made on. A rename is not a change, and neither is a take-out;
  an undo puts back the time the label had before the edit it undoes (``restore_labels``).
- **An answer belongs to the label it was given as.** It counts only for a cluster whose label
  path is the one it was written under (``label_path``), so a cluster moved to a newly made
  label starts unreviewed rather than showing the old label's census.

Read from ``_audit_dir()``, so a snapshot reports its own frozen review; written to ``_run_dir()``
and refused on a snapshot or a finalized run, as every audit write is.
"""

import numpy as np

from detail.core.utils import subject_key
from detail.storage import clip_edits as clip_edits_store
from detail.storage import clip_review as review_store


class ClipReviewOperations:
    """``finish_review`` / ``review_precision`` / ``review_by_cluster``.

    The host supplies ``state``, ``node_view``, ``_record``, ``_run_dir`` and ``_audit_dir``
    (``TemporalClusteringRun``).
    """

    # -- the population -------------------------------------------------------------------

    def _review_clusters(self, path: str | None = None) -> list[dict]:
        """``[{node, label, path, clips}]`` per labelled cluster at or under ``path`` (every one
        for ``None``): the clips whose deepest labelled node is that cluster, as ``ClipRef``s.

        Raises:
            FileNotFoundError: the run has no node cache, so its clips are not listed.
        """
        from detail.storage import behavior_audit as audit

        state = self.state()
        view = self.node_view()
        named = {
            int(n) for n, label in state.node_labels.items() if label in state.labels
        }
        groups = np.asarray(view.tree.leaf_groups(named), dtype=np.int64)
        out = []
        for node in sorted(named):
            label = state.node_labels[str(node)]
            node_path = state.resolved_paths.get(label) or state.labels[label]
            if path is not None and not audit.under(node_path, path):
                continue
            table = view.clips(node)
            own = groups[table.leaf] == node if len(table) else np.zeros(0, bool)
            table = table.take(np.flatnonzero(own))
            clips = [
                clip_edits_store.ClipRef(
                    table.recording_ids[int(r)],
                    subject_key(s),
                    int(t0),
                    int(t1),
                )
                for r, s, t0, t1 in zip(
                    table.recording, table.subject, table.start_ns, table.end_ns
                )
            ]
            out.append(
                {"node": node, "label": label, "path": node_path, "clips": clips}
            )
        return out

    def _review_since(self, label: str) -> str | None:
        return (self.state().membership_changed_at or {}).get(label)

    def _current_answers(
        self, clusters: list[dict]
    ) -> dict[tuple, review_store.ReviewRow]:
        """``{clip key: answer}`` for the clusters' clips, each answer given **as its cluster's
        label path** and made since that label last changed.

        The path matters as much as the time: an answer says "this clip is (not) climb", and
        read under another label it says nothing. Matched on the clip alone, a cluster moved to
        a label made for it showed the old label's census as the new one's precision — a new
        label has no ``membership_changed_at`` to set the old answers aside. A rename or merge
        re-points the answers at the new path (``LabelOperations._carry_answers``), so it
        keeps them.
        """
        rows = review_store.read_review(self._audit_dir())
        latest = review_store.latest_by_path(rows)
        out = {}
        for cluster in clusters:
            since = self._review_since(cluster["label"])
            for clip in cluster["clips"]:
                row = review_store.answer_for(latest, clip.key, cluster["path"])
                if row is not None and (since is None or row.drawn_at >= since):
                    out[clip.key] = row
        return out

    @staticmethod
    def _one_cluster(clusters: list[dict], node: int | None) -> list[dict]:
        if node is None:
            return clusters
        picked = [c for c in clusters if c["node"] == int(node)]
        if not picked:
            raise ValueError(
                f"node {node} is not a labelled cluster in this scope, so it has no clips to "
                "review"
            )
        return picked

    # -- finishing a cluster ----------------------------------------------------------------

    def _not_its_label(self, cluster: dict) -> set[tuple]:
        """The keys of ``cluster``'s clips taken out of its label or moved to another — every
        edit that still applies: made from the cluster's label or one above it."""
        from detail.storage import behavior_audit as audit

        state = self.state()
        keys = {clip.key for clip in cluster["clips"]}
        out = set()
        for edit in clip_edits_store.latest(state.clip_edits):
            key = clip_edits_store.clip_ref(edit).key
            label = edit.get("from_label")
            if key not in keys or label not in state.labels:
                continue
            source = state.resolved_paths.get(label) or state.labels[label]
            if audit.under(cluster["path"], source):
                out.add(key)
        return out

    def finish_review(self, node: int) -> dict:
        """Record that every clip of cluster ``node`` has been reviewed, and return its figure.

        Writes one answer per clip in a single append: ``no`` for each clip taken out of the
        cluster's label or moved to another (``take_out_clips``, made before this call), ``yes``
        for every other one. A later call writes a new census that supersedes this one.

        Returns the cluster's :meth:`review_by_cluster` row: ``estimate`` is exact
        (``remaining == 0``).

        Raises:
            FileNotFoundError: no node cache.
            ValueError: ``node`` is not a labelled cluster; a finalized run.
            ReadOnlyRunError: a snapshot view.
        """
        (cluster,) = self._one_cluster(self._review_clusters(), node)
        out = self._not_its_label(cluster)
        # Logged once the node is known to be a cluster to finish, so a refused call leaves no
        # history line behind; the read-only and finalize refusals still come before any write.
        self._record("finish_review", {"node": int(node)})
        review_store.append_answers(
            self._run_dir(),
            [(clip, clip.key not in out) for clip in cluster["clips"]],
            node=cluster["node"],
            label_path=cluster["path"],
        )
        return next(r for r in self.review_by_cluster() if r["node"] == int(node))

    # -- reading the figure ---------------------------------------------------------------

    def review_precision(self, path: str, node: int | None = None) -> dict | None:
        """The precision the clip review supports for ``path`` (or one cluster), or ``None``
        when nothing in scope has been judged since its label last changed.

        The ``behavior_audit`` estimate form — ``precision``, ``interval``, ``n``,
        ``n_confirmed``, ``below_min_n``, ``unjudged_share``, ``by_stratum`` — plus ``size``
        (clips in scope), ``remaining`` (not yet judged) and ``since`` (the label's last change,
        when ``path`` names one label). Reads ``storage/audit_figures`` as it is.
        """
        from detail.storage import behavior_audit as audit
        from detail.storage import run_labels
        from detail.storage.behavior_output import PATH_SEPARATOR

        clusters = self._one_cluster(self._review_clusters(path), node)
        answers = self._current_answers(clusters)
        if not answers:
            return None
        groups = {
            c["node"]: (
                len(c["clips"]),
                [answers[k.key].yes for k in c["clips"] if k.key in answers],
            )
            for c in clusters
        }
        size = sum(len(c["clips"]) for c in clusters)
        estimate = audit.stratified_estimate(groups)
        state = self.state()
        label = run_labels.find(state.labels, path.split(PATH_SEPARATOR)[-1])
        estimate.update(
            {
                "size": size,
                "remaining": size - len(answers),
                "since": self._review_since(label) if label else None,
            }
        )
        return estimate

    def review_by_cluster(self, path: str | None = None) -> list[dict]:
        """Each labelled cluster's review figure, at or under ``path`` (every one for ``None``).

        ``[{node, label, path, size, share, reviewed_at, estimate}]``: ``label`` the label's name,
        ``path`` its path, ``size`` the cluster's clips, ``share`` its part of its label's clips,
        ``reviewed_at`` when its latest answer counted here was written (ISO-8601 UTC; ``None``
        before any), ``estimate``
        the ``behavior_audit.estimate`` form plus ``size`` and ``remaining``. Judgments count from
        the label's last change, as the label's figure does.
        """
        from detail.storage import behavior_audit as audit

        state = self.state()
        clusters = self._review_clusters(path)
        answers = self._current_answers(clusters)
        label_size: dict[str, int] = {}
        for c in clusters:
            label_size[c["label"]] = label_size.get(c["label"], 0) + len(c["clips"])
        out = []
        for c in clusters:
            rows = [answers[k.key] for k in c["clips"] if k.key in answers]
            judged = [row.yes for row in rows]
            out.append(
                {
                    "node": c["node"],
                    "label": state.labels.get(c["label"], ""),
                    "path": c["path"],
                    "size": len(c["clips"]),
                    "share": len(c["clips"]) / max(label_size[c["label"]], 1),
                    "reviewed_at": max((row.drawn_at for row in rows), default=None),
                    "estimate": {
                        **audit.estimate(judged),
                        "size": len(c["clips"]),
                        "remaining": len(c["clips"]) - len(judged),
                    },
                }
            )
        return out
