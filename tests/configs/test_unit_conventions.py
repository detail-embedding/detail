"""A temporal window must be declared in seconds, never counted in frames or rows.

The recurring bug this guards: a parameter whose meaning depends on context the system never
checks. A window of "8 frames" is 0.53 s on a 15 fps recording and 0.14 s on a 60 Hz one, so the
same config produces a different analysis on every recording and nothing anywhere says so. It has
bitten four times — ASOT's radius as a fraction of the sequence, the temporal stack in frames, and
``novelty_half`` / ``min_size``, which kept their frame counts while ``window_s`` and ``min_len_s``
beside them were converted to seconds for exactly this reason.

None of those is visible to a behavioural test: nothing is wrong at the line where the value is
written, and the fixture that would catch it has to run at a different cadence than the one the
default was tuned at. So this is a metadata scan over the declared config fields, in the spirit of
``tests/storage/test_path_serialization.py``.

Read through ``describe_config``, the same source the CLI help and the docs page read, so the
guard cannot drift from what a user is shown.
"""

import re

import pytest

from detail.configs.describe import describe_config, iter_config_models

# A field naming frames/rows as *its own unit* — "window half-width, in embedded frames",
# "Minimum rows between two matches" — rather than counting things ("total frames to sample",
# "frames per GPU forward pass"). The distinction is the one that matters: a count of items is
# the same count at any cadence; a window is a different duration at every cadence.
_WINDOW_PHRASES = re.compile(
    r"\b(in|of|between|apart|width|window|gap|tolerance|separation|spacing|margin)\b[^.]{0,40}"
    r"\b(frames?|rows?)\b|\b(frames?|rows?)\b[^.]{0,30}\b(between|apart|either side|window)\b",
    re.I,
)

# Fields that name frames/rows and are NOT durations, with why. Each entry is a claim that the
# value means the same thing at any cadence; if that stops being true, move it to seconds.
_NOT_A_DURATION = {
    "n_frames": "a count of frames to sample, not a span of time",
    "fit_share": "a percentage of each recording's frames to sample, not a span of time",
    "batch_size": "frames per forward pass — a memory/throughput knob",
    "val_split": "a fraction of the labeled set",
    "n_keyframes": "how many defining frames to require, not how far apart",
    "n_negatives": "how many background frames to sample",
    "k": "how many neighbours vote",
    "feat_stride": "a step in rows of the feature matrix, which is that matrix's own index space",
    "every_n_frames": "a decimation ratio over the source frames; it *defines* a run's cadence "
    "rather than depending on it, and downstream stages compose with it as a multiple",
    "resize": "pixels, not time",
    "max_results": "a result count",
    "n_exemplars": "a count of exemplars",
    "min_contour_area": "pixels",
    "max_segments": "a cost bound on the O(bouts^2) comparison",
    "folds": "cross-validation folds",
    "seed": "an RNG seed",
}

# Cadence-dependent windows still counted in frames/rows. Empty is the goal: every entry is a knob
# that means a different duration on every recording. Add a reason, or convert it to seconds.
#
# These three are real instances of the bug, deferred rather than dismissed. Each needs its own
# seconds->frames conversion point in code that does no unit conversion today, which is a larger
# change than the segmentation knobs (those already sit beside window_s/min_len_s/band_s and reuse
# their conversion). tracked as planned work.
_CADENCE_EXEMPT: dict[str, str] = {}


def _fields():
    for module_id, model in iter_config_models():
        for field in describe_config(model):
            yield module_id, field


def test_no_temporal_window_is_counted_in_frames_or_rows():
    offenders = []
    for module_id, field in _fields():
        name, help_text, unit = field["name"], field["help"] or "", field["unit"]
        if name.endswith("_s") or name in _NOT_A_DURATION or name in _CADENCE_EXEMPT:
            continue
        if unit in ("frames", "rows") or _WINDOW_PHRASES.search(help_text):
            offenders.append(f"{module_id}.{name} (unit={unit!r}): {help_text[:70]}")
    assert not offenders, (
        "a temporal window must be declared in seconds and converted where the cadence is known "
        "(clip_query.band_rows), or it means a different duration on every recording:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


def test_every_seconds_field_declares_its_unit():
    """The name and the metadata must agree — the docs page and CLI help render the unit."""
    missing = [
        f"{module_id}.{field['name']}"
        for module_id, field in _fields()
        if field["name"].endswith("_s") and field["unit"] != "s"
    ]
    assert (
        not missing
    ), f"fields named _s but not declaring unit 's': {sorted(set(missing))}"


@pytest.mark.parametrize("allow_list", [_NOT_A_DURATION, _CADENCE_EXEMPT])
def test_the_allow_lists_do_not_excuse_fields_that_no_longer_exist(allow_list):
    """A stale entry silently re-permits the bug for a *new* field of the same name."""
    live = {field["name"] for _module_id, field in _fields()}
    stale = sorted(set(allow_list) - live)
    assert not stale, f"allow-list entries for fields that no longer exist: {stale}"


def test_the_scan_actually_recognises_a_frame_counted_window():
    """Guard the guard: if the phrase match stops working, everything passes vacuously."""
    assert _WINDOW_PHRASES.search(
        "novelty comparison window half-width, in embedded frames."
    )
    assert _WINDOW_PHRASES.search("Minimum rows between two matches on one track")
    assert _WINDOW_PHRASES.search("Minimum embedded frames between successive cuts")
    # ...and does not fire on a count of things
    assert not _WINDOW_PHRASES.search(
        "Total frames to sample and label, allocated per recording"
    )
    assert not _WINDOW_PHRASES.search("Frames per GPU forward pass.")
