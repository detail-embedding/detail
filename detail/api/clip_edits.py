"""Taking clips out of a label on a temporal clustering run, and what that does to its output (§22).

The edits themselves are stored fit-free (``storage/clip_edits.py``); this is the half that needs
the fit: turning each edited clip into the members it covers, and the run operations that write
the edits through ``_edit`` (the edit lock, the snapshot and finalized refusals, the history line,
the reprojection).

**The derivation.** Every member inside an edited clip's span — same recording and subject, its
start at or after the clip's, its end at or before the clip's inclusive end — gets the group of the
label it was moved to, or ``-1`` for taken out, *before* the method cuts its runs
(``TemporalClusterMethod.instances_for``'s ``member_groups``). So a clip taken out of the middle of
an instance splits it in two, a piece left shorter than the method's minimum length is dropped as
any short run is, and every export, measure and frame view, all of which read ``instances.csv``,
follow.

**An edit applies only while it still means what it said.** It names the label the clip was taken
out of; when the clip's label is no longer that label or one nested under it (its cluster was
relabelled, or the label deleted), the edit is left in the state and has no effect.
"""

from typing import Self

import numpy as np

from detail.core.protocols import KEEP_GROUP
from detail.storage import clip_edits as clip_edits_store
from detail.storage.clip_edits import ClipRef
from detail.storage.clip_edits import clip_ref

__all__ = ["ClipRef", "clip_ref", "ClipEditOperations", "member_groups"]


def _path_of(state, label: str) -> str:
    return state.resolved_paths.get(label) or state.labels.get(label, label)


def _under(path: str, target: str) -> bool:
    from detail.storage import behavior_output as bo

    return path == target or path.startswith(target + bo.PATH_SEPARATOR)


def member_groups(method, fit, state, leaf_groups: np.ndarray) -> np.ndarray | None:
    """``(M,)`` the per-member override ``instances_for`` takes, or ``None`` with no edit in force.

    ``leaf_groups`` is ``label_groups``' partition (groups numbered in ``run_labels.order``), so a
    member's current label is read off it, and a moved clip gets its target label's group.
    """
    from detail.storage import run_labels

    edits = clip_edits_store.latest(getattr(state, "clip_edits", None) or [])
    if not edits:
        return None
    order = run_labels.order(state.labels)
    group_of = {label: i for i, label in enumerate(order)}
    places = method.member_places(fit)
    current = np.asarray(leaf_groups, dtype=np.int64)[
        np.asarray(method.member_leaves(fit), dtype=np.int64)
    ]
    recording = np.asarray(places.recording_index, dtype=np.int64)
    subject = np.asarray(places.subject, dtype=np.int64)
    start = np.asarray(places.start_ns, dtype=np.int64)
    end = np.asarray(places.end_ns, dtype=np.int64)
    sorted_members = np.lexsort((start, subject, recording))
    units: dict[tuple[int, int], tuple[int, int]] = {}
    begin = 0
    for i in range(1, len(sorted_members) + 1):
        if i == len(sorted_members) or (
            recording[sorted_members[i]] != recording[sorted_members[begin]]
            or subject[sorted_members[i]] != subject[sorted_members[begin]]
        ):
            if begin < len(sorted_members):
                first = sorted_members[begin]
                units[(int(recording[first]), int(subject[first]))] = (begin, i)
            begin = i
    sorted_start = start[sorted_members]
    index = {rid: i for i, rid in enumerate(places.recording_ids)}
    out = np.full(len(current), KEEP_GROUP, dtype=np.int64)
    applied = False
    for edit in edits:
        if edit["from_label"] not in state.labels:
            continue
        bounds = units.get((index.get(edit["recording_id"], -1), int(edit["subject"])))
        if bounds is None:
            continue
        lo, hi = bounds
        a = lo + int(np.searchsorted(sorted_start[lo:hi], edit["t_ns"], side="left"))
        b = lo + int(
            np.searchsorted(sorted_start[lo:hi], edit["end_t_ns"], side="right")
        )
        members = sorted_members[a:b]
        members = members[end[members] <= edit["end_t_ns"]]
        from_path = _path_of(state, edit["from_label"])
        target = group_of.get(edit["to_label"] or "", -1)
        for member in members:
            group = int(current[member])
            if group < 0 or not _under(_path_of(state, order[group]), from_path):
                continue
            out[member] = target
            applied = True
    return out if applied else None


class ClipEditOperations:
    """``take_out_clips`` / ``put_back_clips`` / ``taken_out`` for a temporal clustering run.

    The host supplies ``_edit``, ``_read_state``, ``state``, ``_refuse_if_read_only`` and
    ``_refuse_if_finalized`` (``AuthoredRun``).
    """

    def _resolve_label(self, state, label: str, what: str) -> str:
        """A label id from an id or a name.

        Raises:
            ValueError: no label by that id or name.
        """
        from detail.storage import run_labels

        if label in state.labels:
            return label
        found = run_labels.find(state.labels, str(label))
        if found is None:
            known = ", ".join(sorted(state.labels.values())) or "none yet"
            raise ValueError(
                f"{what} {label!r} is not a label of this run (labels: {known})"
            )
        return found

    def take_out_clips(
        self, clips, from_label: str, to_label: str | None = None, *, progress=None
    ) -> Self:
        """Take ``clips`` out of ``from_label``: unlabelled, or moved to ``to_label``.

        ``clips`` is one clip or a list, each a :class:`ClipRef`, a ``ClipTable.rows()`` row, or
        ``(recording_id, subject, t_ns, end_t_ns)`` (:func:`clip_ref`). ``from_label`` and
        ``to_label`` are label ids or names. The clip stays in its cluster; only the output
        changes, and an earlier edit of the same clip is replaced. Reprojects, so
        ``instances.csv`` stops covering the clips (``progress(stage, fraction, detail)``).

        Raises:
            ValueError: no clips, an unknown label, ``to_label`` equal to ``from_label``, or a
                finalized run.
            ReadOnlyRunError: a snapshot view.
        """
        from detail.core.utils import utc_timestamp

        refs = clip_edits_store.clip_refs(clips)
        if not refs:
            raise ValueError("take_out_clips needs at least one clip")
        self._refuse_if_read_only("remove clips from a label")

        def mutate(state) -> dict:
            source = self._resolve_label(state, from_label, "from_label")
            target = (
                None
                if to_label is None
                else self._resolve_label(state, to_label, "to_label")
            )
            if target == source:
                raise ValueError(
                    f"to_label is {state.labels[source]!r}, the label the clips are removed from"
                )
            state.clip_edits = clip_edits_store.taken_out(
                state.clip_edits, refs, source, target, utc_timestamp()
            )
            return {"clips": len(refs), "from_label": source, "to_label": target}

        self._edit("take clips out", mutate, progress=progress)
        self._clip_edit_stamp = None
        return self

    def put_back_clips(self, clips, *, progress=None) -> Self:
        """Undo :meth:`take_out_clips` for ``clips``: each reports its clustering's label again.

        A clip with no edit is ignored; when none of ``clips`` has one nothing is written.

        Raises:
            ValueError: a finalized run.
            ReadOnlyRunError: a snapshot view.
        """
        refs = clip_edits_store.clip_refs(clips)
        self._refuse_if_read_only("put clips back")
        keys = {ref.key for ref in refs}
        current = clip_edits_store.latest(self._read_state().clip_edits)
        if not any(clip_ref(edit).key in keys for edit in current):
            self._refuse_if_finalized("put clips back")
            return self

        def mutate(state) -> dict:
            state.clip_edits, removed = clip_edits_store.put_back(
                state.clip_edits, refs
            )
            return {"clips": removed}

        self._edit("put clips back", mutate, progress=progress)
        self._clip_edit_stamp = None
        return self

    def taken_out(self, label: str | None = None) -> list[dict]:
        """The clips taken out, one per clip: ``{recording_id, subject, t_ns, end_t_ns,
        from_label, to_label, at, from_name, to_name}``, label ids and their names.

        ``label`` (an id or a name) keeps the clips taken out of that label. Every stored edit is
        listed, including one whose clip's label has since changed, which no longer applies
        (module docstring).
        """
        state = self.state()
        wanted = None if label is None else self._resolve_label(state, label, "label")
        return [
            {
                **edit,
                "from_name": state.labels.get(edit["from_label"], ""),
                "to_name": (
                    state.labels.get(edit["to_label"], "") if edit["to_label"] else None
                ),
            }
            for edit in clip_edits_store.latest(state.clip_edits)
            if wanted is None or edit["from_label"] == wanted
        ]
