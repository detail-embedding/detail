"""Guard: a ground-truth consumer never enumerates behavior streams unfiltered.

**The class:** a query for "the labels" that cannot tell a human's labels from a method's.
`streams_by_role("behavior")` returns both, because a re-ingested prediction is a
behavior-role stream in every structural sense — that is the whole point of the round trip.
So every caller that means *ground truth* has to filter on `origin`, and the ones that mean
*every stream* have to say so deliberately.

This is invisible to behavioral tests: a new consumer built on the unfiltered call works
perfectly, right up until someone imports an ethogram onto a recording it reads. Then it
scores a method against its own output and reports a good number. Nothing raises.

So the rule is enforced on the source instead: `ground_truth_streams()` is the ground-truth
accessor, and any other site listing behavior streams needs an entry here saying why it wants
all of them.
"""

import pathlib
import re

import detail

_PKG = pathlib.Path(detail.__file__).parent

# Matches a behavior-role stream listing on one line, in either spelling: the Recording
# method (`recording.streams_by_role("behavior")`) or the storage function
# (`streams_store.streams_by_role(recording_dir, "behavior")`).
_LISTING = re.compile(r"streams_by_role\([^)]*[\"']behavior[\"']\s*\)")

# (file relative to detail/, why this site wants every behavior stream, prediction included).
# A ground-truth consumer does NOT belong here — it belongs on ground_truth_streams().
ALLOWED: dict[str, str] = {
    "analysis/behavior_labels.py": "defines ground_truth_streams and resolves an explicitly "
    "named stream, so it is the thing doing the filtering",
    "pipeline/query_eval.py": "builds the 'available stream names' list for a typo error "
    "message; a predicted name that exists should say so, then be refused by the resolver",
    "analysis/provenance.py": "filters inline on is_predicted() to hash only ground-truth "
    "labels; the listing is the input to that filter",
}


def _listing_sites() -> set[str]:
    found = set()
    for path in _PKG.rglob("*.py"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.lstrip().startswith("#"):
                continue
            if _LISTING.search(line):
                found.add(path.relative_to(_PKG).as_posix())
    return found


def test_no_unexplained_consumer_lists_behavior_streams_unfiltered():
    offenders = sorted(_listing_sites() - set(ALLOWED))
    assert not offenders, (
        "these files list behavior streams without filtering on origin, so a re-ingested "
        "prediction reads as ground truth. Use behavior_labels.ground_truth_streams(), or "
        f"add an ALLOWED entry saying why every stream is wanted: {offenders}"
    )


def test_the_allow_list_has_no_stale_entries():
    """A stale entry is the same bug one level up — a claim nothing checks."""
    stale = sorted(set(ALLOWED) - _listing_sites())
    assert not stale, f"ALLOWED entries that no longer list behavior streams: {stale}"


def test_the_scan_would_actually_catch_one():
    """Guard the guard: a source scan that matches nothing passes vacuously forever."""
    assert _LISTING.search('    for s in recording.streams_by_role("behavior"):')
    assert _LISTING.search(
        '    streams_store.streams_by_role(recording_dir, "behavior")'
    )
    # and does not fire on the filtered accessor, or on another role
    assert not _LISTING.search("    for entry in ground_truth_streams(recording):")
    assert not _LISTING.search('    recording.streams_by_role("signal")')
