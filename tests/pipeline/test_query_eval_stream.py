"""Selecting which ground-truth behavior stream query-eval scores against (§16).

A recording may carry more than one ``behavior``-role stream (e.g. an early hand-scored
subset alongside a full ethogram). Eligibility reads the stream's *covered* vocabulary, so
without a selector the first stream used to win and every label belonging to the others read
as unscored — the recording was then skipped silently, which looks like missing annotation
rather than the wrong stream being read.

``resolve_behavior_stream`` now raises instead, so that particular wrong answer is no longer
reachable. Which stream is "first" depends on the order they were added, and a run that
scores a different ground truth after an unrelated import is not one anybody can check.
"""

import numpy as np
import pytest

from detail.api.recording import Recording
from detail.pipeline.query_eval import QueryEvalModule, evaluate_query
from detail.storage import recording as recording_store

N = 240
PERIOD = 20_000_000
FIRST = [(20, 34), (60, 74), (100, 114), (140, 154)]
SECOND = [(30, 44), (80, 94), (130, 144), (180, 194)]


def _trajectory():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(N, 8)).astype(np.float32)
    for s, e in FIRST:
        emb[s : e + 1] = np.array([4.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    for s, e in SECOND:
        emb[s : e + 1] = np.array([0, 4.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    return emb, np.arange(N, dtype=np.int32), np.arange(N, dtype=np.int64) * PERIOD


@pytest.fixture
def two_stream_recording(experiment_dir, recording_dir, make_embedding_run, tmp_path):
    """One recording with two behavior streams: `early` (added first) and `full`."""
    emb, frame_idx, t_ns = _trajectory()
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, emb, frame_idx, t_ns)
    rec = Recording(experiment_dir, "recording_001")
    early = tmp_path / "early.csv"
    early.write_text(
        "\n".join(
            ["start_frame,end_frame,label"] + [f"{s},{e},climb" for s, e in FIRST]
        )
        + "\n",
        encoding="utf-8",
    )
    rec.add_stream("early", raw_path=early, importer="behavior_csv")
    full = tmp_path / "full.csv"
    full.write_text(
        "\n".join(
            ["start_frame,end_frame,label"] + [f"{s},{e},groom" for s, e in SECOND]
        )
        + "\n",
        encoding="utf-8",
    )
    rec.add_stream("full", raw_path=full, importer="behavior_csv")
    return experiment_dir


def _by_behavior(experiment_dir, config):
    return {r.behavior: r for r in evaluate_query(experiment_dir, config)}


def test_without_a_selector_the_ambiguity_is_an_error_not_a_silent_skip(
    two_stream_recording,
):
    """The failure this file's docstring describes is now impossible to get silently.

    Before: the first stream won, every label belonging to the others read as unscored, and
    the recording was skipped with `"no eligible recordings"` — which looks like missing
    annotation rather than the wrong stream being read. A run could report a clean number for
    one behavior and nothing at all for another, for no reason visible anywhere.
    """
    from detail.analysis.behavior_labels import AmbiguousBehaviorStreamError

    with pytest.raises(AmbiguousBehaviorStreamError, match="early.*full|full.*early"):
        _by_behavior(
            two_stream_recording,
            {"embed_run_id": "emb_01", "behaviors": ["climb", "groom"], "band_s": 0.2},
        )


def test_selecting_the_second_stream_scores_its_behaviors(two_stream_recording):
    results = _by_behavior(
        two_stream_recording,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "stream_name": "full",
            "band_s": 0.2,
        },
    )
    assert results["groom"].skipped is None
    assert results["groom"].metrics.n_gt == len(SECOND) - 1  # one held out as exemplar


def test_each_stream_scores_its_own_behaviors_when_named(two_stream_recording):
    """Naming is the only way in now, so each stream must be reachable and score its own."""
    early = _by_behavior(
        two_stream_recording,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["climb"],
            "stream_name": "early",
            "band_s": 0.2,
        },
    )
    full = _by_behavior(
        two_stream_recording,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "stream_name": "full",
            "band_s": 0.2,
        },
    )
    assert early["climb"].metrics is not None
    assert full["groom"].metrics is not None


def test_intervals_come_from_the_selected_stream_not_the_first(two_stream_recording):
    """Eligibility and interval loading must read the *same* stream — reading the
    vocabulary from one and the intervals from another silently scores the wrong spans.

    Each behavior exists in exactly one stream, so a behavior is scorable under its own
    stream and skipped under the other, and its ``n_gt`` matches that stream's count.
    """
    base = {"embed_run_id": "emb_01", "band_s": 0.2}
    under_full = _by_behavior(
        two_stream_recording,
        {**base, "behaviors": ["climb", "groom"], "stream_name": "full"},
    )
    under_early = _by_behavior(
        two_stream_recording,
        {**base, "behaviors": ["climb", "groom"], "stream_name": "early"},
    )
    assert under_full["groom"].metrics.n_gt == len(SECOND) - 1
    assert under_full["climb"].skipped == "no eligible recordings"
    assert under_early["climb"].metrics.n_gt == len(FIRST) - 1
    assert under_early["groom"].skipped == "no eligible recordings"


def test_unknown_stream_name_names_the_available_streams(two_stream_recording):
    """A typo must not read as 'nothing is annotated' — that points at the labels."""
    with pytest.raises(ValueError, match="no selected recording has a behavior stream"):
        QueryEvalModule().validate(
            two_stream_recording,
            {
                "embed_run_id": "emb_01",
                "behaviors": ["groom"],
                "stream_name": "ful",  # typo
                "track_id": 0,
            },
        )


def test_unknown_stream_error_lists_what_is_available(two_stream_recording):
    with pytest.raises(ValueError, match="early.*full|full.*early"):
        QueryEvalModule().validate(
            two_stream_recording,
            {
                "embed_run_id": "emb_01",
                "behaviors": ["groom"],
                "stream_name": "nope",
                "track_id": 0,
            },
        )


def test_validate_accepts_the_selected_stream(two_stream_recording):
    out = QueryEvalModule().validate(
        two_stream_recording,
        {
            "embed_run_id": "emb_01",
            "behaviors": ["groom"],
            "stream_name": "full",
            "track_id": 0,
        },
    )
    assert out["source_recording_ids"] == ["recording_001"]


def test_cli_exposes_the_stream_option():
    from detail.cli import analysis_query_eval

    opt = next(p for p in analysis_query_eval.params if p.name == "stream_name")
    assert opt.default is None  # omitted = the first behavior stream, as before
