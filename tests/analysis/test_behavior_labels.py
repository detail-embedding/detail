"""Shared behavior-labels primitive — pure interval→frame logic + the real GT stream path.

The pure tests pin the multi-class generalization (state, point, overlap, fixed vocabulary); the
recording tests exercise the same ingest path the eval uses — a ``behavior_csv`` stream added via
``Recording.add_stream``, then read back as a per-frame label array.
"""

import numpy as np

from detail.analysis import behavior_labels as bl
from detail.api.recording import Recording
from detail.storage import recording as recording_store

PERIOD = 33_333_333  # ns per frame at 30 fps


# --- pure frames_to_labels ----------------------------------------------------------- #
def test_state_point_and_unlabeled():
    t = np.arange(10, dtype=np.int64) * 100
    intervals = [(200, 400, "walk"), (700, 700, "poke")]
    labels, id_to_label = bl.frames_to_labels(
        t, intervals, behavior_types={"walk": "state", "poke": "point"}
    )
    assert id_to_label == {0: "poke", 1: "walk"}  # ids follow sorted label names
    walk, poke = 1, 0
    assert list(labels) == [-1, -1, walk, walk, walk, -1, -1, poke, -1, -1]


def test_point_window_widens_the_hit():
    t = np.arange(10, dtype=np.int64) * 100
    labels, _ = bl.frames_to_labels(
        t, [(500, 500, "poke")], behavior_types={"poke": "point"}, point_window_ns=100
    )
    assert list(labels) == [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1]  # t in [400, 600]


def test_overlap_first_interval_wins():
    t = np.arange(5, dtype=np.int64)
    labels, id_to_label = bl.frames_to_labels(t, [(1, 3, "a"), (2, 4, "b")])
    assert id_to_label == {0: "a", 1: "b"}
    # frames 2/3 are claimed by "a" (earlier in stream order), only frame 4 is "b"
    assert list(labels) == [-1, 0, 0, 0, 1]


def test_fixed_label_to_id_and_unknown_label_skipped():
    t = np.arange(4, dtype=np.int64)
    labels, id_to_label = bl.frames_to_labels(
        t, [(0, 1, "walk"), (2, 3, "other")], label_to_id={"walk": 5, "rear": 6}
    )
    # "walk" uses its fixed id 5; "other" is not in the map, so those frames stay unlabeled
    assert list(labels) == [5, 5, -1, -1]
    assert id_to_label == {5: "walk", 6: "rear"}


# --- the real behavior stream path --------------------------------------------------- #
def _recording_with_labels(experiment_dir, recording_dir, tmp_path, lines) -> Recording:
    t_ns = np.arange(60, dtype=np.int64) * PERIOD
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    csv = tmp_path / "labels.csv"
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Recording(experiment_dir, "recording_001").add_stream(
        "behavior", raw_path=csv, importer="behavior_csv"
    )
    return Recording(experiment_dir, "recording_001")


def test_recording_frame_labels_over_a_real_stream(
    experiment_dir, recording_dir, tmp_path
):
    lines = ["start_frame,end_frame,label", "10,20,groom", "40,45,groom", "30,30,rear"]
    recording = _recording_with_labels(experiment_dir, recording_dir, tmp_path, lines)

    intervals = bl.recording_behavior_intervals(recording)
    assert len(intervals) == 3 and {iv[2] for iv in intervals} == {"groom", "rear"}

    fl = bl.recording_frame_labels(recording)
    assert fl.labels.shape == (60,)
    groom = next(i for i, n in fl.id_to_label.items() if n == "groom")
    rear = next(i for i, n in fl.id_to_label.items() if n == "rear")
    assert (fl.labels[10:21] == groom).all()
    assert (fl.labels[40:46] == groom).all()
    assert fl.labels[30] == rear  # point behavior labels exactly its frame
    assert (fl.labels[0:10] == -1).all()  # frames the annotation did not cover


def test_no_behavior_stream_is_empty(experiment_dir, recording_dir):
    assert (
        bl.recording_behavior_intervals(Recording(experiment_dir, "recording_001"))
        == []
    )


def test_behaviors_filter_restricts(experiment_dir, recording_dir, tmp_path):
    lines = ["start_frame,end_frame,label", "10,20,groom", "30,30,rear"]
    recording = _recording_with_labels(experiment_dir, recording_dir, tmp_path, lines)
    intervals = bl.recording_behavior_intervals(recording, behaviors=["groom"])
    assert {iv[2] for iv in intervals} == {"groom"}
