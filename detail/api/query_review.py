"""A query run's audit: every clip the cutoff keeps is reviewed, one recording at a time (§20).

The query half of §22's clip review. A reviewer opens one recording of one behavior, watches every
candidate the behavior's cutoff keeps there, removes the wrong ones (:meth:`take_out_clips`, which
writes as it goes) and finishes (:meth:`finish_review`), which writes a census to
``clip_review.csv``: ``no`` for each removed clip, ``yes`` for every other one, in one append, under
the behavior's id as its label path. The recording's precision at that cutoff is then a count.

**The population is the cutoff's.** A behavior's candidates are the method's, at no cutoff
(``QueryRun._audit_pool``); its cutoff keeps the ones at or above it. Candidates overlapping the
behavior's own example or non-example clips are the method finding its input and are never
reviewed or counted (``_audit_examples``).

**A figure is exact at a cutoff when every candidate at or above it has an answer.** Nothing stores
how far a review reached: a recording's ``reviewed_floor`` is derived from the log, the lowest score
at or above which every candidate has a current answer. Lowering the cutoff below it leaves the new
candidates unreviewed, and the figure there is refused rather than read as a sample — the answered
clips are the top of the recording's candidates, not a sample of them, and read as one they
flatter the method. ``at_floor`` gives the exact figure over the reviewed range instead.

**A removal edits the output, never an input** (``Behavior.removed``; ``pipeline/query.py``'s
``project_artifact`` leaves the clip out at every cutoff). Nothing is moved between behaviors: each
is its own search under its own cutoff. Putting a clip back withdraws its answer under this
behavior only (``clip_review.withdraw(label_path=)``), since one span can be a candidate of two.

**Blind.** :meth:`review_clips` hands a surface the clips in a seeded random order and without their
scores, so a review pane has no score to show and no rank to read one from.

Read from ``_audit_dir()``, so a snapshot reports its own frozen review; written to ``_run_dir()``
and refused on a snapshot or a finalized run, as every audit write is.
"""

import hashlib
import random
from typing import Self

from detail.core.errors import UserInputError
from detail.storage import analysis_runs as analysis_store
from detail.storage import clip_edits as clip_edits_store
from detail.storage import clip_review as review_store
from detail.storage import query_runs


class QueryReviewOperations:
    """``review_clips`` / ``take_out_clips`` / ``put_back_clips`` / ``taken_out`` /
    ``finish_review`` / ``review_precision`` / ``review_by_recording``.

    The host supplies ``state``, ``_read_state``, ``_audit_pool``, ``_audit_examples``,
    ``artifact``, ``_edit``, ``_record``, ``_refuse_if_read_only``, ``_run_dir``, ``_audit_dir``
    and ``_experiment`` (``QueryRun``).
    """

    # -- resolving what the caller named ----------------------------------------------------

    @staticmethod
    def _review_behavior(state, behavior: str) -> query_runs.Behavior:
        """The behavior with this id or name.

        Raises:
            ValueError: no behavior carries it, naming the ones that exist.
        """
        for b in state.behaviors:
            if b.id == behavior or b.name == behavior:
                return b
        known = ", ".join(f"{b.id} ({b.name})" for b in state.behaviors)
        raise ValueError(f"no behavior {behavior!r} in this run; it has {known}")

    def _review_recording(self, state, recording: str) -> str:
        """The searched recording with this id or name.

        Raises:
            ValueError: the run did not search it, naming the ones it did.
        """
        names = self._experiment.recording_names()
        for rid in state.search_recording_ids:
            if recording in (rid, names.get(rid)):
                return rid
        known = ", ".join(names.get(r, r) for r in state.search_recording_ids)
        raise ValueError(
            f"this run did not search a recording {recording!r}; it searched {known}"
        )

    # -- the population -----------------------------------------------------------------------

    def _review_candidates(self, behavior_id: str) -> dict[str, list]:
        """``{recording_id: [candidate rows]}``: the behavior's candidates at no cutoff, minus
        the ones its example and non-example clips overlap, highest score first. Every searched
        recording has an entry."""
        pool = self._audit_pool(behavior_id)
        examples = self._audit_examples(pool)
        out: dict[str, list] = {rid: [] for rid in self.state().search_recording_ids}
        for row in pool:
            if row.instance_id not in examples:
                out.setdefault(row.recording_id, []).append(row)
        for rows in out.values():
            rows.sort(key=lambda row: (-(row.score or 0.0), row.t_ns))
        return out

    def _cutoff_of(self, behavior: query_runs.Behavior, cutoff: float | None) -> float:
        if cutoff is not None:
            return float(cutoff)
        if behavior.cutoff is None:
            raise UserInputError(
                f"{behavior.name!r} has no cutoff yet, so there is nothing to review: set a "
                "cutoff on CALIBRATE first"
            )
        return float(behavior.cutoff)

    def _kept(self, rows: list, cutoff: float) -> list:
        from detail.storage import behavior_audit as audit

        direction = audit.score_direction(self.artifact().manifest)
        return [row for row in rows if audit.retained(row.score, cutoff, direction)]

    def _review_answers(self, behavior_id: str) -> dict:
        """``{clip key: answer row}`` — each clip's current answer as this behavior."""
        latest = review_store.latest_by_path(
            review_store.read_review(self._audit_dir())
        )
        out = {}
        for key, path in latest:
            if path != behavior_id:
                continue
            row = review_store.answer_for(latest, key, behavior_id)
            if row is not None:
                out[key] = row
        return out

    # -- what a surface shows -----------------------------------------------------------------

    def review_clips(self, behavior: str, recording: str) -> list[dict]:
        """Every clip to review for ``behavior`` in ``recording`` at the behavior's cutoff, in
        the order to show them: ``[{recording_id, subject, t_ns, end_t_ns}]``.

        **No score, and a random order** (§20): seeing a clip's score, or its rank, pulls a
        judgment toward it. The order is seeded by the behavior and the recording, so reopening
        shows the same order. Clips the behavior's own examples overlap are left out.

        Raises:
            ValueError: an unknown behavior or recording.
            UserInputError: the behavior has no cutoff yet.
        """
        state = self.state()
        b = self._review_behavior(state, behavior)
        rid = self._review_recording(state, recording)
        rows = self._kept(self._review_candidates(b.id)[rid], self._cutoff_of(b, None))
        clips = sorted(
            (
                {
                    "recording_id": row.recording_id,
                    "subject": int(row.subject or 0),
                    "t_ns": int(row.t_ns),
                    "end_t_ns": int(row.end_t_ns),
                }
                for row in rows
            ),
            key=lambda clip: (clip["t_ns"], clip["end_t_ns"]),
        )
        seed = hashlib.sha256(f"{b.id}\0{rid}".encode()).digest()
        random.Random(int.from_bytes(seed[:8], "big")).shuffle(clips)
        return clips

    # -- removing and putting back ------------------------------------------------------------

    def take_out_clips(
        self, clips, from_label: str, to_label: str | None = None, *, progress=None
    ) -> Self:
        """Remove ``clips`` from behavior ``from_label`` (an id or a name), and reproject.

        Each clip is a ``ClipRef``, a clip dict or ``(recording_id, subject, t_ns, end_t_ns)``
        and must be one of the behavior's candidates (at any cutoff). The artifact leaves a
        removed clip out at every cutoff; the search, the examples and the non-examples are
        untouched. A clip already removed is left as it is.

        Raises:
            ValueError: no clips, an unknown behavior, a ``to_label``, or a finalized run.
            UserInputError: a clip that is not a candidate of the behavior, or that its own
                example or non-example clips overlap (a ``ValueError``).
            ReadOnlyRunError: a snapshot view.
        """
        from detail.core.utils import utc_timestamp

        refs = clip_edits_store.clip_refs(clips)
        if not refs:
            raise ValueError("take_out_clips needs at least one clip")
        self._refuse_if_read_only("remove clips from a behavior")
        b = self._review_behavior(self.state(), from_label)
        if to_label is not None:
            raise ValueError(
                f"a clip of {b.name!r} can be removed, not moved to {to_label!r}: each "
                "behavior is its own search under its own cutoff, so a moved clip would be an "
                "instance the other behavior's search never found"
            )
        self._refuse_clips_not_candidates(b, refs)

        def mutate(state) -> dict:
            behavior = state.behavior(b.id)
            have = behavior.removed_keys()
            at = utc_timestamp()
            added = [ref for ref in refs if ref.key not in have]
            behavior.removed.extend(
                {
                    "recording_id": ref.recording_id,
                    "subject": ref.subject,
                    "t_ns": ref.t_ns,
                    "end_t_ns": ref.end_t_ns,
                    "at": at,
                }
                for ref in added
            )
            return {"clips": len(added), "behavior_id": b.id}

        self._edit("take clips out", mutate, progress=progress)
        return self

    def _refuse_clips_not_candidates(self, behavior, refs) -> None:
        pool = self._audit_pool(behavior.id)
        examples = self._audit_examples(pool)
        candidates = {query_runs.clip_key(row): row for row in pool}
        for ref in refs:
            row = candidates.get(ref.key)
            if row is None:
                raise UserInputError(
                    f"the clip {ref.recording_id} subject {ref.subject} "
                    f"{ref.t_ns}-{ref.end_t_ns} ns is not a candidate of {behavior.name!r}; "
                    "only a clip its search found can be removed from it"
                )
            if row.instance_id in examples:
                raise UserInputError(
                    f"the clip {ref.recording_id} {ref.t_ns}-{ref.end_t_ns} ns overlaps one "
                    f"of {behavior.name!r}'s own example clips, which are never reviewed"
                )

    def put_back_clips(self, clips, from_label: str, *, progress=None) -> Self:
        """Undo :meth:`take_out_clips` for ``clips`` in behavior ``from_label``.

        A clip that was not removed is ignored; when none was, nothing is written. Each clip's
        answer as this behavior is withdrawn, so its recording reads it as not yet reviewed
        until the review is finished again; an answer as another behavior is untouched.

        Raises:
            ValueError: an unknown behavior, or a finalized run.
            ReadOnlyRunError: a snapshot view.
        """
        refs = clip_edits_store.clip_refs(clips)
        self._refuse_if_read_only("put clips back")
        b = self._review_behavior(self._read_state(), from_label)
        keys = {ref.key for ref in refs}
        if not keys & b.removed_keys():
            self._refuse_if_finalized("put clips back")
            return self

        def mutate(state) -> dict:
            behavior = state.behavior(b.id)
            back = keys & behavior.removed_keys()
            behavior.removed = [
                r for r in behavior.removed if query_runs.removal_key(r) not in back
            ]
            # The census said "no" because the clip was out; once it is back that answer no
            # longer describes it (#24, as on a clustering run).
            review_store.withdraw(self._run_dir(), back, label_path=b.id)
            return {"clips": len(back), "behavior_id": b.id}

        self._edit("put clips back", mutate, progress=progress)
        return self

    def taken_out(self, label: str | None = None) -> list[dict]:
        """The clips removed, one per clip and behavior: ``{recording_id, subject, t_ns,
        end_t_ns, at, behavior_id, behavior_name}``. ``label`` (an id or a name) keeps one
        behavior's.

        Raises:
            ValueError: an unknown behavior.
        """
        state = self.state()
        behaviors = (
            [self._review_behavior(state, label)]
            if label is not None
            else state.behaviors
        )
        return [
            {**r, "behavior_id": b.id, "behavior_name": b.name}
            for b in behaviors
            for r in b.removed
        ]

    # -- finishing a recording ----------------------------------------------------------------

    def finish_review(self, behavior: str, recording: str) -> dict:
        """Record that every clip of ``behavior`` in ``recording`` at its cutoff has been
        reviewed, and return the recording's :meth:`review_by_recording` row.

        Writes one answer per clip in a single append: ``no`` for each removed clip, ``yes`` for
        every other one. A later call writes a new census that supersedes this one.

        Raises:
            ValueError: an unknown behavior or recording; a finalized run.
            UserInputError: the behavior has no cutoff yet.
            ResultsOutOfDateError: the results are out of date after a failed rebuild.
            ReadOnlyRunError: a snapshot view.
        """
        self._refuse_if_read_only("finish_review")
        # A census written while the results are known to be out of date would record answers
        # about an output the rebuild has not caught up with (#24, as on a clustering run).
        analysis_store.refuse_if_stale(self._run_dir())
        state = self._read_state()
        b = self._review_behavior(state, behavior)
        rid = self._review_recording(state, recording)
        rows = self._kept(self._review_candidates(b.id)[rid], self._cutoff_of(b, None))
        removed = b.removed_keys()
        self._record("finish_review", {"behavior_id": b.id, "recording_id": rid})
        review_store.append_answers(
            self._run_dir(),
            [
                (query_runs.clip_key(row), query_runs.clip_key(row) not in removed)
                for row in rows
            ],
            node=None,
            label_path=b.id,
        )
        return next(
            r for r in self.review_by_recording(b.id) if r["recording_id"] == rid
        )

    # -- reading the figure -------------------------------------------------------------------

    @staticmethod
    def _floor(rows: list, answers: dict) -> float | None:
        """The lowest score at or above which every candidate in ``rows`` has an answer;
        ``None`` when the top one has none. Ties at a score are one step: the floor never sits
        on a score some unanswered candidate shares. A query's score is a confidence, higher
        better (§20), which is the order this walks."""
        unanswered = [
            row.score for row in rows if query_runs.clip_key(row) not in answers
        ]
        top_unanswered = max(unanswered) if unanswered else None
        answered = [
            row.score
            for row in rows
            if top_unanswered is None or row.score > top_unanswered
        ]
        return min(answered) if answered else None

    def _recording_figure(self, rows: list, answers: dict, cutoff: float) -> dict:
        from detail.storage import behavior_audit as audit

        kept = self._kept(rows, cutoff)
        judged = [
            answers[query_runs.clip_key(row)]
            for row in kept
            if query_runs.clip_key(row) in answers
        ]
        return {
            **audit.estimate([row.yes for row in judged]),
            "size": len(kept),
            "remaining": len(kept) - len(judged),
            "reviewed_at": max((row.drawn_at for row in judged), default=None),
        }

    def review_by_recording(
        self, behavior: str, *, cutoff: float | None = None
    ) -> list[dict]:
        """Each searched recording's review of ``behavior`` at ``cutoff`` (the behavior's own
        when none is given), in search order.

        ``[{recording_id, name, size, removed, examples, exact, reviewed_floor, reviewed_at,
        estimate, at_floor}]``: ``size`` the clips to review at the cutoff, ``removed`` how many
        of them are removed, ``examples`` how many the behavior's own clips overlap (left out),
        ``exact`` whether every clip at the cutoff has an answer (a recording with none is),
        ``estimate`` the ``behavior_audit.estimate`` form plus ``size``, ``remaining`` and
        ``reviewed_at``. ``at_floor`` is the exact figure at ``reviewed_floor`` when the cutoff
        sits below it, else ``None``.

        Raises:
            ValueError: an unknown behavior.
            UserInputError: the behavior has no cutoff and none is given.
        """
        state = self.state()
        b = self._review_behavior(state, behavior)
        at = self._cutoff_of(b, cutoff)
        names = self._experiment.recording_names()
        answers = self._review_answers(b.id)
        removed = b.removed_keys()
        pool = self._audit_pool(b.id)
        example_ids = self._audit_examples(pool)
        examples = self._kept(
            [row for row in pool if row.instance_id in example_ids], at
        )
        out = []
        for rid, rows in self._review_candidates(b.id).items():
            figure = self._recording_figure(rows, answers, at)
            floor = self._floor(rows, answers)
            exact = figure["remaining"] == 0
            out.append(
                {
                    "recording_id": rid,
                    "name": names.get(rid, rid),
                    "size": figure["size"],
                    "removed": sum(
                        1
                        for row in self._kept(rows, at)
                        if query_runs.clip_key(row) in removed
                    ),
                    "examples": sum(1 for row in examples if row.recording_id == rid),
                    "exact": exact,
                    "reviewed_floor": floor,
                    "reviewed_at": figure["reviewed_at"],
                    "estimate": figure,
                    "at_floor": (
                        None
                        if exact or floor is None
                        else self._recording_figure(rows, answers, floor)
                    ),
                }
            )
        return out

    def review_precision(
        self,
        behavior: str,
        recording: str | None = None,
        *,
        cutoff: float | None = None,
    ) -> dict | None:
        """The precision the review supports for ``behavior`` at ``cutoff`` (its own when none is
        given), over every recording or one; ``None`` when nothing in scope has an answer.

        The ``behavior_audit`` estimate form, combined across recordings by
        ``stratified_estimate`` (each recording a stratum of its clip count), plus ``size``,
        ``remaining``, ``exact``, ``reviewed_floor`` (the lowest cutoff at which every recording
        in scope is exact) and ``at_floor`` (the figure there, when the cutoff sits below it).
        **Not exact means no figure**: ``precision`` is ``None`` with a ``reason``, because the
        answered clips are the top of the candidates, not a sample of them.

        Raises:
            ValueError: an unknown behavior or recording.
            UserInputError: the behavior has no cutoff and none is given.
        """
        rid = (
            None
            if recording is None
            else self._review_recording(self.state(), recording)
        )

        def scope(at: float | None) -> list[dict]:
            return [
                r
                for r in self.review_by_recording(behavior, cutoff=at)
                if rid is None or r["recording_id"] == rid
            ]

        rows = scope(cutoff)
        if not any(r["estimate"]["n"] or r["at_floor"] for r in rows):
            return None
        floors = [r["reviewed_floor"] for r in rows if r["size"] or r["at_floor"]]
        floor = None if None in floors else max(floors, default=None)
        out = self._combined(rows)
        exact = out["remaining"] == 0
        if not exact:
            out["precision"] = None
            out["reason"] = (
                f"{out['remaining']} of {out['size']} clips at this cutoff are not reviewed"
            )
        out["exact"] = exact
        out["reviewed_floor"] = floor
        out["at_floor"] = (
            None if exact or floor is None else self._combined(scope(floor))
        )
        return out

    @staticmethod
    def _combined(rows: list[dict]) -> dict:
        """The recordings' figures as one, each a stratum of its clip count."""
        from detail.storage import behavior_audit as audit

        groups = {
            r["recording_id"]: (
                r["estimate"]["size"],
                [True] * r["estimate"]["n_confirmed"]
                + [False] * (r["estimate"]["n"] - r["estimate"]["n_confirmed"]),
            )
            for r in rows
        }
        estimate = audit.stratified_estimate(groups)
        estimate.update(
            {
                "size": sum(r["estimate"]["size"] for r in rows),
                "remaining": sum(r["estimate"]["remaining"] for r in rows),
            }
        )
        return estimate
