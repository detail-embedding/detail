"""A subject read back from data goes through ``core.utils.subject_key``, never a bare ``int``.

**The class.** A subject — a behavior row's ``subject``, a clip spec's ``track_id``, a
clustering member's ``subject`` — was once either a track's int **or** a joint (§14) run's
``"joint"``, the pair. Code written when only one animal could reach it cast the value with
``int(...)``, and every such cast either raised on a pair or, worse, never saw the pair at all
because something upstream had already turned it into ``0``: a clustering fit's ``int64``
``subject`` column held ``0`` for a joint run, and every card, thumbnail and profile of a pair
clustering showed track 0, one animal of the pair, with no error anywhere.

**Now.** The single-animal build cut the pair subject, so a subject is a plain track id. The
scan stays because ``subject_key`` is still the one reading of a stored subject: ``None`` or
``""`` (a stream with no subject column) is track ``0``, a digit string or ``np.integer`` is
its int, and anything else raises with a message naming the value. A bare ``int(...)``
raises a ``TypeError`` on the ``None`` a stream's ``or 0`` read as track 0; keeping the reads
in one function keeps them the same.

This scan finds ``int(...)`` applied to a subject: an expression naming ``subject`` /
``subjects`` / ``run_subject`` / ``thumb_subject``, or a ``track_id`` **read from data** (an
attribute, a ``["track_id"]`` or a ``.get("track_id")``). A bare ``int(track_id)`` parameter is
not scanned: it is the most common spelling of a keypoint track id, which is always an int, so
this is a lower bound on the class, not its edge.
"""

import ast
import pathlib
import textwrap

import pytest

import detail
from detail.core.utils import subject_key

_SUBJECT_NAMES = {"subject", "subjects", "run_subject", "thumb_subject", "row_subjects"}

#: Files allowed ``int()`` on a subject, with how many and why. Exact: a new cast in one of
#: these fails as surely as one anywhere else, and an entry that no longer excuses anything
#: fails too, so the list cannot outlive its reasons.
ALLOWED = {
    # The int64 codes themselves: decoded on the way out, never handed out as codes.
    "api/node_view.py": (
        1,
        "the int64 codes a unit index is built on, decoded on the way out",
    ),
    # The cluster methods' bookkeeping: handed subject_key's int, stored and compared as given.
    "cluster_methods/codebook_viterbi.py": (3, "the method's int64 bookkeeping"),
    "cluster_methods/dtw_finch.py": (2, "the method's int64 bookkeeping"),
    # The slice's own subject, already read with subject_key by its caller.
    "api/frame_slices.py": (3, "the slice's subject, already a subject_key int"),
    # A detection frame's track ids are object ids, ints by §13 — never a pair.
    "core/dataclasses.py": (2, "a detection frame's object ids (§13)"),
    # Keypoints: a keypoint track's subject is an int by §26, never a pair.
    "core/examples.py": (6, "a keypoint example set's track id (§26)"),
    "pipeline/headings.py": (3, "a heading's keypoint track id (§26)"),
    "pipeline/keypoint_dataset.py": (2, "a keypoint example frame's track id (§26)"),
    "pipeline/keypoint_track_tune.py": (1, "a keypoint example frame's track id (§26)"),
    "pipeline/subject_frames.py": (2, "an example-set candidate's track id (§26)"),
    # Hidden by surfaces.HIDDEN (§25): kept guarded and untouched, not generalised into.
    "analysis/keypoint_audit.py": (2, "hidden: keypoint_audit"),
    "storage/clip_query_csv.py": (1, "hidden: behavior_query"),
}


def _names_a_subject(node: ast.AST) -> bool:
    """Whether the expression ``node`` reads a subject, by the forms in the module docstring."""
    for part in ast.walk(node):
        if isinstance(part, ast.Name) and part.id in _SUBJECT_NAMES:
            return True
        if isinstance(part, ast.Attribute) and (
            part.attr in _SUBJECT_NAMES or part.attr == "track_id"
        ):
            return True
        if isinstance(part, ast.Constant) and part.value in (
            *_SUBJECT_NAMES,
            "track_id",
        ):
            return True
    return False


def _casts(source: str) -> list[int]:
    """Line numbers of every ``int(<a subject>)`` in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and node.args
            and _names_a_subject(node.args[0])
        ):
            found.append(node.lineno)
    return found


def _offenders(root: pathlib.Path) -> dict[str, list[int]]:
    out = {}
    for py in sorted(root.rglob("*.py")):
        lines = _casts(py.read_text(encoding="utf-8-sig"))
        if lines:
            out[py.relative_to(root).as_posix()] = lines
    return out


def test_no_subject_is_cast_to_int_outside_the_reasoned_list():
    root = pathlib.Path(detail.__file__).parent
    found = _offenders(root)
    unexpected = {
        path: lines
        for path, lines in found.items()
        if len(lines) != ALLOWED.get(path, (0, ""))[0]
    }
    stale = sorted(path for path in ALLOWED if path not in found)
    assert not unexpected, (
        "int() on a subject — read it with core.utils.subject_key, which reads a "
        f"missing subject as track 0 and refuses what is no track id: {unexpected}"
    )
    assert not stale, f"ALLOWED excuses files with no such cast any more: {stale}"


@pytest.mark.parametrize(
    "source",
    [
        "subject = 0 if row.subject is None else int(row.subject)",
        "key = (r, int(spec.get('track_id', 0)))",
        "x = int(member['subject'])",
        "x = int(clip.track_id)",
        "x = int(\n    row.subject or 0\n)",
        "x = int(t.run_subject[r])",
    ],
)
def test_the_scan_sees_every_spelling_the_bug_took(source):
    """Each of these is a site this PR fixed, in the form it was found in."""
    assert _casts(textwrap.dedent(source)), source


@pytest.mark.parametrize(
    "source",
    [
        "x = subject_key(row.subject)",
        "x = int(track_id)",  # a parameter: a keypoint track id, out of the scan's reach
        "x = int(start_ns)",
    ],
)
def test_the_scan_leaves_the_fixed_forms_alone(source):
    assert not _casts(source), source


@pytest.mark.parametrize(
    "value, key",
    [(0, 0), (3, 3), ("2", 2), (None, 0), ("", 0)],
)
def test_subject_key_reads_every_stored_form(value, key):
    assert subject_key(value) == key


def test_subject_key_refuses_what_is_no_subject():
    with pytest.raises(ValueError, match="track"):
        subject_key("pair")
