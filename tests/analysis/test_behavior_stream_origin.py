"""A predicted behavior stream is never read as ground truth (§19).

**The class:** a derived artifact re-ingested through the same door as its source, with
nothing on it recording which one it is. It is invisible to behavioral tests because every
step works — the export is correct, the import is correct, the eval runs green — and the only
symptom is a number that is better than it should be.

The instance this guards, from the behavior clusterers (since removed): a clusterer's ethogram
exported as `start_frame,end_frame,label` and re-imported through `behavior_csv`, the same
importer a hand-scored CSV uses. Before `origin`, the resulting stream was byte-indistinguishable
from human annotation — same role, same shape, same clock, same importer, and
`covered_behaviors`/`behavior_types` derived from the predicted rows themselves. So
`Recording.is_annotated` flipped true, the recording passed query-eval's `requires: ["embedding",
"behavior"]` gate, `is_annotated_for` returned true for every predicted label, and `cluster_eval`
would score — or `auto_names` name — a clusterer against its own output. Nothing failed anywhere.

**The rule:** a label may score a method, never build one. `origin` makes the
prediction/ground-truth boundary structural rather than something a reader has to remember to
check, which is what makes that rule enforceable at all.

Legacy streams carry no `origin` and read as `ground_truth`, so nothing already on disk
changes meaning.
"""

import numpy as np
import pytest

from detail.analysis import behavior_labels
from detail.analysis import provenance
from detail.api.recording import Recording
from detail.storage import recording as recording_store
from detail.storage import streams as streams_store

PERIOD = 33_333_333
GT = "start_frame,end_frame,label\n0,9,grooming\n20,29,rearing\n"
PREDICTED = (
    "start_frame,end_frame,label,origin\n"
    "0,9,grooming,predicted\n"
    "20,29,rearing,predicted\n"
)


@pytest.fixture
def recording(recording_dir):
    """A recording with a real timestamp map — behavior_csv is on the video_frames clock,
    so the import layer resolves frames against it."""
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.arange(60, dtype=np.int64) * PERIOD,
    )
    return Recording(recording_dir.parent.parent, recording_dir.name)


def _add(recording, tmp_path, name, text, **kw):
    path = tmp_path / f"{name}.csv"
    path.write_text(text, encoding="utf-8")
    recording.add_stream(name, raw_path=path, importer="behavior_csv", **kw)
    return next(s for s in recording.streams if s.get("name") == name)


# --------------------------------------------------------------------------- #
# the manifest field
# --------------------------------------------------------------------------- #
def test_a_stream_with_no_origin_reads_as_ground_truth():
    # Every behavior stream written before this field existed. Nothing on disk changes.
    assert streams_store.stream_origin({"name": "behavior"}) == "ground_truth"
    assert streams_store.is_predicted({"name": "behavior"}) is False


def test_a_hand_written_csv_imports_as_ground_truth(recording, tmp_path):
    entry = _add(recording, tmp_path, "truth", GT)
    assert streams_store.stream_origin(entry) == "ground_truth"


def test_the_files_own_origin_column_is_honored(recording, tmp_path):
    # The tool's own exporters write this column, so a prediction identifies itself rather
    # than depending on whoever imports it remembering what it was.
    entry = _add(recording, tmp_path, "pred", PREDICTED)
    assert streams_store.is_predicted(entry)


def test_an_explicit_origin_beats_the_column(recording, tmp_path):
    # So a user can mark a prediction that came from somewhere else, or correct a file.
    entry = _add(recording, tmp_path, "pred", GT, origin="predicted")
    assert streams_store.is_predicted(entry)


def test_a_bad_origin_is_refused(recording, tmp_path):
    with pytest.raises(ValueError, match="origin must be one of"):
        _add(recording, tmp_path, "x", GT, origin="probably_true")


def test_a_file_declaring_two_origins_is_refused(recording, tmp_path):
    mixed = (
        "start_frame,end_frame,label,origin\n"
        "0,9,grooming,predicted\n"
        "20,29,rearing,ground_truth\n"
    )
    with pytest.raises(ValueError, match="more than one origin"):
        _add(recording, tmp_path, "x", mixed)


def test_origin_on_a_stream_with_no_labels_is_refused(recording, tmp_path):
    # A signal stream has no labels, so "where the labels came from" means nothing on it.
    path = tmp_path / "s.csv"
    path.write_text("t_ns,value\n0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only applies to a labeled"):
        recording.add_stream(
            "sig", raw_path=path, importer="timeseries_csv", origin="predicted"
        )


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #
def test_a_predicted_stream_does_not_annotate_the_recording(recording, tmp_path):
    _add(recording, tmp_path, "pred", PREDICTED)
    # It would previously have gated this recording into an eval — against itself.
    assert recording.is_annotated is False
    assert recording.is_annotated_for("grooming") is False
    assert recording.is_annotated_for("grooming", "pred") is False


def test_a_ground_truth_stream_still_annotates(recording, tmp_path):
    _add(recording, tmp_path, "truth", GT)
    assert recording.is_annotated is True
    assert recording.is_annotated_for("grooming") is True


def test_naming_a_predicted_stream_for_scoring_raises(recording, tmp_path):
    _add(recording, tmp_path, "pred", PREDICTED)
    with pytest.raises(behavior_labels.PredictedBehaviorStreamError, match="predicted"):
        behavior_labels.resolve_behavior_stream(recording, "pred")
    with pytest.raises(behavior_labels.PredictedBehaviorStreamError):
        behavior_labels.recording_behavior_intervals(recording, stream_name="pred")


def test_an_unnamed_read_resolves_past_a_prediction(recording, tmp_path):
    """A prediction beside the truth must not make the truth ambiguous, nor be picked.

    Both failure modes are live: picking it scores a method against itself, and raising
    ambiguity would break every existing unnamed reader the moment an ethogram is imported.
    """
    _add(recording, tmp_path, "truth", GT)
    _add(recording, tmp_path, "pred", PREDICTED)
    assert behavior_labels.resolve_behavior_stream(recording, None) == "truth"
    assert recording.covered_behaviors() == ["grooming", "rearing"]


def test_a_recording_carrying_only_a_prediction_reads_as_unannotated(
    recording, tmp_path
):
    # The case the old ambiguity check could never catch: with one stream there was nothing
    # to be ambiguous about, so the prediction was simply used.
    _add(recording, tmp_path, "pred", PREDICTED)
    assert behavior_labels.resolve_behavior_stream(recording, None) is None
    assert behavior_labels.recording_behavior_intervals(recording) == []


def test_two_ground_truth_streams_still_raise_ambiguity(recording, tmp_path):
    _add(recording, tmp_path, "truth_a", GT)
    _add(recording, tmp_path, "truth_b", GT)
    with pytest.raises(behavior_labels.AmbiguousBehaviorStreamError):
        behavior_labels.resolve_behavior_stream(recording, None)


def test_a_caller_that_wants_the_prediction_can_ask_for_it(recording, tmp_path):
    # Rendering a prediction, or comparing it against truth, is legitimate — what is not is
    # treating it as truth by default.
    _add(recording, tmp_path, "pred", PREDICTED)
    name = behavior_labels.resolve_behavior_stream(
        recording, "pred", allow_predicted=True
    )
    assert name == "pred"
    intervals = behavior_labels.recording_behavior_intervals(
        recording, stream_name=name, allow_predicted=True
    )
    assert [label for _s, _e, label in intervals] == ["grooming", "rearing"]


def test_ground_truth_streams_filters(recording, tmp_path):
    _add(recording, tmp_path, "truth", GT)
    _add(recording, tmp_path, "pred", PREDICTED)
    names = [s["name"] for s in behavior_labels.ground_truth_streams(recording)]
    assert names == ["truth"]


# --------------------------------------------------------------------------- #
# drift
# --------------------------------------------------------------------------- #
def test_importing_a_prediction_does_not_report_the_labels_as_changed(
    recording, tmp_path
):
    """`labels_hash` answers "did the ground truth change" — a prediction is not an answer.

    Folding one in would report drift on every result built from that recording, and drift is
    only useful while it means what it says.
    """
    _add(recording, tmp_path, "truth", GT)
    before = provenance.labels_hash(recording.dir)
    _add(recording, tmp_path, "pred", PREDICTED)
    assert provenance.labels_hash(recording.dir) == before
