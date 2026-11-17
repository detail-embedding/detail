"""Clips a user took out of a label, or moved to another, on a temporal clustering run (§22).

Clustering is never perfect, and a user who has watched a clip and knows it is wrong must not be
made to export it. So a clip can be **taken out** of the label the clustering gave it (it reports
as unlabelled) or **moved** to another label. The fit is not touched: the clip stays in its
cluster, and the edit is an override applied when the instances are derived.

**A clip is one leaf run** — the montage clip, ``(recording_id, subject, t_ns, end_t_ns)`` — which
the node cache keeps stable because the fit never changes. ``end_t_ns`` is the clip's last
member's own time, inclusive (§19). :class:`ClipRef` is that address; :func:`clip_ref` builds one
from a ``ClipTable.rows()`` row (``start_t_ns``), a stored edit or review row (``t_ns``), or a
4-tuple.

The stored list (``TemporalClusteringState.clip_edits``) holds one entry per clip,
``{recording_id, subject, t_ns, end_t_ns, from_label, to_label, at}``: ``from_label`` the label id
it was taken out of, ``to_label`` the label id it was moved to or ``None`` for taken out, ``at``
when (ISO-8601 UTC). The latest entry for a clip wins; putting a clip back removes its entry.

Fit-free and pure: the derivation that needs the tree lives in ``api/clip_edits.py``.
"""

from dataclasses import dataclass

EDIT_FIELDS = (
    "recording_id",
    "subject",
    "t_ns",
    "end_t_ns",
    "from_label",
    "to_label",
    "at",
)


@dataclass(frozen=True)
class ClipRef:
    """One clip: one leaf run of one subject in one recording, ``end_t_ns`` inclusive."""

    recording_id: str
    subject: int
    t_ns: int
    end_t_ns: int

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.recording_id, self.subject, self.t_ns, self.end_t_ns)


def clip_ref(clip) -> ClipRef:
    """A :class:`ClipRef` from a ``ClipRef``, a ``ClipTable.rows()`` row, a stored row, or a tuple.

    Raises:
        TypeError: ``clip`` is none of these.
        KeyError: a dict missing the recording, subject or either end.
    """
    if isinstance(clip, ClipRef):
        return clip
    if isinstance(clip, dict):
        start = clip["start_t_ns"] if "start_t_ns" in clip else clip["t_ns"]
        return ClipRef(
            str(clip["recording_id"]),
            int(clip.get("subject") or 0),
            int(start),
            int(clip["end_t_ns"]),
        )
    if isinstance(clip, (tuple, list)) and len(clip) == 4:
        recording_id, subject, start, end = clip
        return ClipRef(str(recording_id), int(subject), int(start), int(end))
    raise TypeError(
        f"a clip is a ClipRef, a clip row dict or (recording_id, subject, t_ns, end_t_ns), "
        f"not {type(clip).__name__}"
    )


def clip_refs(clips) -> list[ClipRef]:
    """Each of ``clips`` as a :class:`ClipRef`, duplicates dropped, order kept.

    A single clip (a ``ClipRef`` or a row dict) is accepted as a list of one.
    """
    if isinstance(clips, (ClipRef, dict)):
        clips = [clips]
    seen: dict[tuple, ClipRef] = {}
    for clip in clips:
        ref = clip_ref(clip)
        seen.setdefault(ref.key, ref)
    return list(seen.values())


def normalize(edits) -> list[dict]:
    """The stored list as written: every field present, ids as ``str``, times as ``int``."""
    out = []
    for edit in edits or []:
        to_label = edit.get("to_label")
        out.append(
            {
                "recording_id": str(edit["recording_id"]),
                "subject": int(edit.get("subject") or 0),
                "t_ns": int(edit["t_ns"]),
                "end_t_ns": int(edit["end_t_ns"]),
                "from_label": str(edit["from_label"]),
                "to_label": None if to_label in (None, "") else str(to_label),
                "at": str(edit.get("at") or ""),
            }
        )
    return out


def latest(edits) -> list[dict]:
    """One entry per clip, the latest winning, in the order the winners were written."""
    by_clip: dict[tuple, dict] = {}
    for edit in normalize(edits):
        key = clip_ref(edit).key
        by_clip.pop(key, None)
        by_clip[key] = edit
    return list(by_clip.values())


def taken_out(
    edits, clips, from_label: str, to_label: str | None, at: str
) -> list[dict]:
    """``edits`` with ``clips`` taken out of ``from_label`` (to ``to_label``), replacing any
    earlier entry for the same clip."""
    refs = clip_refs(clips)
    keys = {ref.key for ref in refs}
    kept = [e for e in latest(edits) if clip_ref(e).key not in keys]
    return kept + [
        {
            "recording_id": ref.recording_id,
            "subject": ref.subject,
            "t_ns": ref.t_ns,
            "end_t_ns": ref.end_t_ns,
            "from_label": str(from_label),
            "to_label": None if to_label is None else str(to_label),
            "at": at,
        }
        for ref in refs
    ]


def put_back(edits, clips) -> tuple[list[dict], int]:
    """``(edits without clips' entries, how many entries that removed)``."""
    keys = {ref.key for ref in clip_refs(clips)}
    current = latest(edits)
    kept = [e for e in current if clip_ref(e).key not in keys]
    return kept, len(current) - len(kept)


def _key_text(edit) -> str:
    recording_id, subject, t_ns, end_t_ns = clip_ref(edit).key
    return f"{recording_id}|{subject}|{t_ns}|{end_t_ns}"


def by_key(edits) -> dict[str, dict]:
    """``{clip key: edit}`` for the latest edit of each clip — a map an undo can diff and put
    back entry by entry (``AuthoredRun.label_maps``). The key is text so it survives JSON.
    """
    return {_key_text(edit): dict(edit) for edit in latest(edits)}


def relabel(edits, source: str, into: str) -> tuple[list[dict], int]:
    """``(edits with label id source read as into, how many edits that changed)``: what a merge
    of ``source`` into ``into`` does to them. An edit left moving a clip from ``into`` to
    ``into`` says nothing any more, and is dropped."""
    out, changed = [], 0
    for edit in latest(edits):
        new = dict(edit)
        if new["from_label"] == source:
            new["from_label"] = into
        if new["to_label"] == source:
            new["to_label"] = into
        if new != edit:
            changed += 1
        if new["to_label"] is not None and new["to_label"] == new["from_label"]:
            continue
        out.append(new)
    return out, changed


def restore(edits, wanted: dict[str, dict | None]) -> list[dict]:
    """``edits`` with each clip in ``wanted`` (keyed as :func:`by_key`) set to its edit there, or
    its edit removed for ``None``; every other clip's edit as it is."""
    current = by_key(edits)
    for key, edit in wanted.items():
        current.pop(key, None)
        if edit is not None:
            current[key] = dict(edit)
    return normalize(list(current.values()))
