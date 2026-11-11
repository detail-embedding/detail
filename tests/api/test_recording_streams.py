"""Recording stream API: add_stream, alignment, frame_t_ns (Section 3)."""

import numpy as np
import pytest

from detail.api.recording import Recording
from detail.storage import recording as recording_store

DATA_TXT = """\
Start Date: 02/16/26
Subject: TEST1
Box: 4
Start Time: 14:00:38
MSN: Test_MSN
G:       3.000
H:      10.000
E:
     0:       19.000       11.000      100.000
T:
     0:        0.000        1.500       10.000
"""

PROGRAM_MPC = """\
\\  Event stamps ( array E )
\\   11  = Houselight on
\\   19  = Timeout ends and/or drug available at START
\\   100 = Recording termination
\\
\\  Variables
"""


def _src_files(tmp_path):
    src = tmp_path / "_src"
    src.mkdir(exist_ok=True)
    data = src / "subject.txt"
    program = src / "task.MPC"
    data.write_text(DATA_TXT, encoding="utf-8")
    program.write_text(PROGRAM_MPC, encoding="utf-8")
    return data, program


def _recording(experiment_dir) -> Recording:
    return Recording(experiment_dir, "recording_001")


# -- add_stream end-to-end -------------------------------------------------- #
def test_add_stream_end_to_end(experiment_dir, recording_dir, tmp_path):
    data, program = _src_files(tmp_path)
    sess = _recording(experiment_dir)
    sess.add_stream(
        "medpc", raw_path=data, importer="medpc", aux_paths={"program": program}
    )

    streams = sess.streams
    assert len(streams) == 1
    entry = streams[0]
    assert entry["name"] == "medpc"
    assert entry["type"] == "events"
    assert entry["columns"] == ["t_ns", "code", "label"]
    assert entry["alignment"] == {"offset_ns": 0, "active": "offset"}

    rows = sess.stream_events("medpc")
    assert rows == [
        (0, 19, "Timeout ends and/or drug available at START"),
        (1_500_000_000, 11, "Houselight on"),
        (10_000_000_000, 100, "Recording termination"),
    ]

    # raw source files copied verbatim; canonical written; paths are recording-relative
    assert (recording_dir / "streams/medpc/raw/subject.txt").is_file()
    assert (recording_dir / "streams/medpc/raw/task.MPC").is_file()
    assert (recording_dir / "streams/medpc/canonical/part_001.csv").is_file()
    blob = (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    assert "/_src/" not in blob and str(tmp_path) not in blob


def test_add_stream_duplicate_name_rejected(experiment_dir, recording_dir, tmp_path):
    data, program = _src_files(tmp_path)
    sess = _recording(experiment_dir)
    sess.add_stream(
        "medpc", raw_path=data, importer="medpc", aux_paths={"program": program}
    )
    with pytest.raises(ValueError, match="already exists"):
        sess.add_stream(
            "medpc", raw_path=data, importer="medpc", aux_paths={"program": program}
        )


def test_add_stream_unknown_importer(experiment_dir, recording_dir, tmp_path):
    data, _ = _src_files(tmp_path)
    with pytest.raises(ValueError, match="unknown importer"):
        _recording(experiment_dir).add_stream("s", raw_path=data, importer="nope")


def test_add_stream_missing_aux_rejected(experiment_dir, recording_dir, tmp_path):
    data, _ = _src_files(tmp_path)
    with pytest.raises(ValueError, match="requires aux"):
        _recording(experiment_dir).add_stream("s", raw_path=data, importer="medpc")


def test_add_stream_missing_data_file(experiment_dir, recording_dir, tmp_path):
    _, program = _src_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        _recording(experiment_dir).add_stream(
            "s",
            raw_path=tmp_path / "nope.txt",
            importer="medpc",
            aux_paths={"program": program},
        )


def test_add_stream_missing_aux_file(experiment_dir, recording_dir, tmp_path):
    data, _ = _src_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        _recording(experiment_dir).add_stream(
            "s",
            raw_path=data,
            importer="medpc",
            aux_paths={"program": tmp_path / "nope.MPC"},
        )


def test_add_stream_invalid_name(experiment_dir, recording_dir, tmp_path):
    data, program = _src_files(tmp_path)
    with pytest.raises(ValueError):
        _recording(experiment_dir).add_stream(
            "bad name", raw_path=data, importer="medpc", aux_paths={"program": program}
        )


# -- delete + offset + dependents ------------------------------------------- #
def test_delete_stream(experiment_dir, recording_dir, tmp_path):
    data, program = _src_files(tmp_path)
    sess = _recording(experiment_dir)
    sess.add_stream(
        "medpc", raw_path=data, importer="medpc", aux_paths={"program": program}
    )
    sess.delete_stream("medpc")
    assert sess.streams == []
    assert not (recording_dir / "streams/medpc").exists()


def test_set_stream_offset_and_dependents(experiment_dir, recording_dir, tmp_path):
    data, program = _src_files(tmp_path)
    sess = _recording(experiment_dir)
    sess.add_stream(
        "medpc", raw_path=data, importer="medpc", aux_paths={"program": program}
    )
    sess.set_stream_offset("medpc", -5_000_000_000)
    assert sess.streams[0]["alignment"]["offset_ns"] == -5_000_000_000
    assert sess.stream_dependents("medpc") == []


# -- frame_t_ns (global frame → video t_ns) --------------------------------- #
def test_frame_t_ns_across_parts(experiment_dir, recording_dir):
    ts_dir = recording_dir / "video/timestamps/generated"
    recording_store.write_timestamps_csv(
        ts_dir / "part_001.csv", np.array([0, 100, 200, 300, 400], dtype=np.int64)
    )
    recording_store.write_timestamps_csv(
        ts_dir / "part_002.csv", np.array([500, 600, 700], dtype=np.int64)
    )
    data = recording_store.read_recording(recording_dir)
    data["video"]["files"].append(
        {
            "index": 1,
            "file": "video/part_002.mp4",
            "timestamps": {
                "generated": "video/timestamps/generated/part_002.csv",
                "user": None,
                "active": "generated",
            },
        }
    )
    recording_store.write_recording(recording_dir, data)

    sess = _recording(experiment_dir)
    assert sess.frame_t_ns(0) == 0
    assert sess.frame_t_ns(4) == 400  # last frame of part 1
    assert sess.frame_t_ns(5) == 500  # first frame of part 2
    assert sess.frame_t_ns(7) == 700
    with pytest.raises(IndexError):
        sess.frame_t_ns(8)
    with pytest.raises(IndexError):
        sess.frame_t_ns(-1)


def test_frame_timestamps_concatenates_parts(experiment_dir, recording_dir):
    ts_dir = recording_dir / "video/timestamps/generated"
    recording_store.write_timestamps_csv(
        ts_dir / "part_001.csv", np.array([0, 100, 200], dtype=np.int64)
    )
    sess = _recording(experiment_dir)
    assert sess.frame_timestamps().tolist() == [0, 100, 200]


def test_frame_timestamps_empty_without_video(experiment_dir, recording_dir):
    # the fixture declares a part with no timestamp CSV; remove the video section
    data = recording_store.read_recording(recording_dir)
    data.pop("video", None)
    recording_store.write_recording(recording_dir, data)
    assert _recording(experiment_dir).frame_timestamps().size == 0


# -- StreamPreview accessor ------------------------------------------------- #
def test_stream_preview(experiment_dir, recording_dir, tmp_path):
    data, program = _src_files(tmp_path)
    sess = _recording(experiment_dir)
    sess.add_stream(
        "medpc", raw_path=data, importer="medpc", aux_paths={"program": program}
    )
    preview = sess.stream_preview("medpc")
    assert preview.type == "events"
    assert preview.offset_ns == 0
    # distinct labels, first-seen order
    assert preview.labels() == [
        "Timeout ends and/or drug available at START",
        "Houselight on",
        "Recording termination",
    ]
    # stream-local events
    assert preview.events() == [
        (0, "Timeout ends and/or drug available at START"),
        (1_500_000_000, "Houselight on"),
        (10_000_000_000, "Recording termination"),
    ]
    # aligned events apply the offset
    sess.set_stream_offset("medpc", 2_000_000_000)
    aligned = sess.stream_preview("medpc").events(aligned=True)
    assert aligned[0] == (2_000_000_000, "Timeout ends and/or drug available at START")


def test_stream_preview_unknown_raises(experiment_dir, recording_dir):
    with pytest.raises(ValueError, match="no stream"):
        _recording(experiment_dir).stream_preview("nope")


# -- behavior labels (frame-indexed interval stream, Section 3) -------------- #
def _write_timestamps(recording_dir):
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv",
        np.array([0, 100, 200, 300, 400], dtype=np.int64),
    )


def _behavior_csv(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("start_frame,end_frame,label,subject\n1,3,groom,1\n", encoding="utf-8")
    return p


def test_add_behavior_stream_resolves_frames(experiment_dir, recording_dir, tmp_path):
    _write_timestamps(recording_dir)
    sess = _recording(experiment_dir)
    assert sess.is_annotated is False

    sess.add_stream(
        "behavior", raw_path=_behavior_csv(tmp_path), importer="behavior_csv"
    )

    entry = sess.streams[0]
    assert entry["role"] == "behavior"
    assert entry["shape"] == "interval"
    assert entry["clock"] == "video_frames"
    assert entry["alignment"] == {"offset_ns": 0, "active": "offset"}
    assert entry["dtypes"]["subject"] == "int32"

    # frame -> t_ns resolved against the active map; frames kept alongside
    rows = sess.stream_events("behavior")
    assert rows == [(100, 300, "groom", 1, 1, 3)]

    assert sess.is_annotated is True
    assert [s["name"] for s in sess.streams_by_role("behavior")] == ["behavior"]


def test_add_behavior_without_video_raises(experiment_dir, recording_dir, tmp_path):
    data = recording_store.read_recording(recording_dir)
    data.pop("video", None)
    recording_store.write_recording(recording_dir, data)
    with pytest.raises(ValueError, match="no video to anchor"):
        _recording(experiment_dir).add_stream(
            "behavior", raw_path=_behavior_csv(tmp_path), importer="behavior_csv"
        )


def test_add_behavior_frame_past_video_raises(experiment_dir, recording_dir, tmp_path):
    _write_timestamps(recording_dir)  # 5 frames
    p = tmp_path / "labels.csv"
    p.write_text(
        "start_frame,end_frame,label\n1,9,groom\n", encoding="utf-8"
    )  # frame 9 > 4
    with pytest.raises(ValueError, match="past the video"):
        _recording(experiment_dir).add_stream(
            "behavior", raw_path=p, importer="behavior_csv"
        )


def test_covered_behaviors_override_and_helpers(
    experiment_dir, recording_dir, tmp_path
):
    _write_timestamps(recording_dir)
    sess = _recording(experiment_dir)
    p = tmp_path / "labels.csv"
    p.write_text("start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8")
    # 'rear' was scored but never occurred -> declared, still covered
    sess.add_stream(
        "behavior",
        raw_path=p,
        importer="behavior_csv",
        covered_behaviors=["groom", "rear"],
    )
    assert set(sess.covered_behaviors()) == {"groom", "rear"}
    assert sess.is_annotated_for("groom") and sess.is_annotated_for("rear")
    assert not sess.is_annotated_for("climb")
    assert sess.behavior_types()["groom"] == "state"


def test_covered_behaviors_must_cover_present(experiment_dir, recording_dir, tmp_path):
    _write_timestamps(recording_dir)
    p = tmp_path / "labels.csv"
    p.write_text("start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must include every labeled behavior"):
        _recording(experiment_dir).add_stream(
            "behavior",
            raw_path=p,
            importer="behavior_csv",
            covered_behaviors=["rear"],
        )


# -- generic CSV streams (timeseries_csv / events_csv, Section 3) ------------ #
def test_add_timeseries_csv_stream_end_to_end(experiment_dir, recording_dir, tmp_path):
    p = tmp_path / "signal.csv"
    p.write_text("seconds,bpm,temp\n0.0,72,36.5\n0.5,73,36.6\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    # a seconds-clocked continuous stream needs no video to anchor
    sess.add_stream("hr", raw_path=p, importer="timeseries_csv")

    entry = sess.streams[0]
    assert entry["type"] == "sampled"
    assert entry["shape"] == "continuous"
    assert entry["clock"] == "stream_seconds"
    assert entry["columns"] == ["t_ns", "bpm", "temp"]
    assert entry["dtypes"]["bpm"] == "float"
    assert entry["sample_rate_hz"] == 2.0
    # canonical floats round-trip through the typed read
    assert sess.stream_events("hr") == [(0, 72.0, 36.5), (500_000_000, 73.0, 36.6)]


def test_add_timeseries_frame_clock_resolves_against_map(
    experiment_dir, recording_dir, tmp_path
):
    _write_timestamps(recording_dir)  # frames 0..4 -> t_ns 0,100,...,400
    p = tmp_path / "signal.csv"
    p.write_text("frame,bpm\n1,72\n3,73\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    sess.add_stream("hr", raw_path=p, importer="timeseries_csv")

    entry = sess.streams[0]
    assert entry["clock"] == "video_frames"
    assert "sample_rate_hz" not in entry  # not resolvable at import for a frame clock
    # frame -> t_ns resolved from the active map; the frame column rides alongside
    assert sess.stream_events("hr") == [(100, 1, 72.0), (300, 3, 73.0)]


def test_add_events_csv_stream_end_to_end(experiment_dir, recording_dir, tmp_path):
    p = tmp_path / "events.csv"
    p.write_text("seconds,code,label\n1.0,5,press\n2.0,,release\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    sess.add_stream("ev", raw_path=p, importer="events_csv")

    entry = sess.streams[0]
    assert entry["type"] == "events"
    assert entry["shape"] == "event"
    assert entry["clock"] == "stream_seconds"
    assert entry["columns"] == ["t_ns", "code", "label"]
    assert sess.stream_events("ev") == [
        (1_000_000_000, 5, "press"),
        (2_000_000_000, None, "release"),
    ]


# -- explicit column mapping + alignment at import (Section 3) --------------- #


def _pose_csv(tmp_path):
    """A CSV shaped the way a source tool shaped it: an unrecognized time-column name."""
    p = tmp_path / "pose.csv"
    p.write_text("t_sec,nose_x,nose_y,tail_x\n0,1,2,3\n0.1,4,5,6\n", encoding="utf-8")
    return p


def test_mapped_csv_makes_one_stream_carrying_the_chosen_channels(
    experiment_dir, recording_dir, tmp_path
):
    """One CSV, one stream, only the named columns.

    The count is the point: a stream per column would multiply the per-stream alignment
    offset and store a verbatim copy of the source file per column.
    """
    sess = _recording(experiment_dir)
    sess.add_stream(
        "pose",
        raw_path=_pose_csv(tmp_path),
        importer="timeseries_csv",
        config={
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [
                {"column": "nose_x", "name": "nose_x"},
                {"column": "nose_y", "name": "ny"},
            ],
        },
    )

    assert len(sess.streams) == 1
    entry = sess.streams[0]
    assert entry["columns"] == ["t_ns", "nose_x", "ny"]
    assert entry["dtypes"] == {"t_ns": "int64", "nose_x": "float", "ny": "float"}
    assert entry["clock"] == "stream_seconds"
    assert sess.stream_events("pose") == [(0, 1.0, 2.0), (100_000_000, 4.0, 5.0)]
    # tail_x was not asked for, so it is not in the stream
    assert "tail_x" not in entry["columns"]
    raw_files = list((recording_dir / "streams/pose/raw").iterdir())
    assert len(raw_files) == 1, raw_files


def test_mapped_csv_records_the_mapping_it_resolved(
    experiment_dir, recording_dir, tmp_path
):
    """``auto`` is resolved at import and the resolved value written back (Tenet 9).

    A record saying ``auto`` cannot be reproduced once the source file is gone.
    """
    p = tmp_path / "s.csv"
    p.write_text("seconds,v\n0,1\n1,2\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    sess.add_stream("s", raw_path=p, importer="timeseries_csv", config={})

    metadata = sess.streams[0]["metadata"]
    assert metadata["time_column"] == "seconds"
    assert metadata["time_kind"] == "seconds"  # not "auto"
    assert metadata["channels"] == ["v"]


def test_no_config_imports_exactly_as_before(experiment_dir, recording_dir, tmp_path):
    """The sniffing path is untouched, so a stream written before configs existed reads."""
    p = tmp_path / "s.csv"
    p.write_text("seconds,a,b\n0,1,2\n1,3,4\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    sess.add_stream("s", raw_path=p, importer="timeseries_csv")

    entry = sess.streams[0]
    assert entry["columns"] == ["t_ns", "a", "b"]
    assert sess.stream_events("s") == [(0, 1.0, 2.0), (1_000_000_000, 3.0, 4.0)]


def test_utc_time_kind_anchors_at_the_first_row(
    experiment_dir, recording_dir, tmp_path
):
    p = tmp_path / "sensor.csv"
    p.write_text(
        "Timestamp,temp\n2026-09-04T14:00:38Z,36.5\n2026-09-04T14:00:39Z,36.6\n",
        encoding="utf-8",
    )
    sess = _recording(experiment_dir)
    sess.add_stream(
        "sensor",
        raw_path=p,
        importer="timeseries_csv",
        config={"time_column": "Timestamp", "time_kind": "utc"},
    )

    entry = sess.streams[0]
    assert entry["clock"] == "stream_seconds"
    assert entry["metadata"]["utc_anchor_iso"] == "2026-09-04T14:00:38+00:00"
    assert sess.stream_events("sensor") == [(0, 36.5), (1_000_000_000, 36.6)]


def test_offset_ns_is_set_at_import(experiment_dir, recording_dir, tmp_path):
    sess = _recording(experiment_dir)
    sess.add_stream(
        "pose",
        raw_path=_pose_csv(tmp_path),
        importer="timeseries_csv",
        config={"time_column": "t_sec", "time_kind": "seconds"},
        offset_ns=3_480_000_000,
    )

    assert sess.streams[0]["alignment"] == {
        "offset_ns": 3_480_000_000,
        "active": "offset",
    }


def test_offset_ns_refused_on_a_video_anchored_stream(
    experiment_dir, recording_dir, tmp_path
):
    """A ``video_frames`` stream is anchored at import, so an offset would be ignored on
    read. Refused rather than stored, or the manifest carries a number nothing applies.
    """
    _write_timestamps(recording_dir)
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="already the video's own"):
        sess.add_stream(
            "behavior",
            raw_path=_behavior_csv(tmp_path),
            importer="behavior_csv",
            offset_ns=5,
        )


def test_offset_refusal_reads_the_resolved_clock_not_the_declared_one(
    experiment_dir, recording_dir, tmp_path
):
    """``timeseries_csv`` declares ``stream_seconds`` but reports ``video_frames`` for a
    frame-kind mapping. Checking the declared clock would accept an offset read ignores.
    """
    _write_timestamps(recording_dir)
    p = tmp_path / "f.csv"
    p.write_text("idx,v\n1,10\n3,30\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="already the video's own"):
        sess.add_stream(
            "f",
            raw_path=p,
            importer="timeseries_csv",
            config={"time_column": "idx", "time_kind": "frame"},
            offset_ns=5,
        )


def test_duplicate_header_names_are_refused(experiment_dir, recording_dir, tmp_path):
    """``csv.DictReader`` collapses two same-named columns to whichever came last, so the
    first one's data is silently gone. Duplicate headers are ordinary in the wild (every
    DeepLabCut export repeats the scorer name), so this refuses rather than guesses."""
    p = tmp_path / "dup.csv"
    p.write_text("seconds,x,x\n0,1,2\n", encoding="utf-8")
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="names \\['x'\\] more than once"):
        sess.add_stream("d", raw_path=p, importer="timeseries_csv")


@pytest.mark.parametrize("reserved", ["t_ns", "frame"])
def test_channel_named_after_a_reserved_column_is_refused(
    experiment_dir, recording_dir, tmp_path, reserved
):
    """A channel called ``t_ns`` writes a second ``t_ns`` column into the canonical CSV and
    overwrites the int64 time anchor's dtype with ``float`` (Tenet 8). The manifest stays
    well-formed and the stream is silently unreadable."""
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="reserved by the canonical stream schema"):
        sess.add_stream(
            "p",
            raw_path=_pose_csv(tmp_path),
            importer="timeseries_csv",
            config={
                "time_column": "t_sec",
                "time_kind": "seconds",
                "channels": [{"column": "nose_x", "name": reserved}],
            },
        )


@pytest.mark.parametrize(
    "channels, match",
    [
        (
            [{"column": "nose_x", "name": "q"}, {"column": "nose_y", "name": "q"}],
            "channel name 'q' is listed twice",
        ),
        (
            [{"column": "nose_x", "name": "a"}, {"column": "nose_x", "name": "b"}],
            "source column 'nose_x' is listed twice",
        ),
    ],
)
def test_repeated_channels_are_refused(
    experiment_dir, recording_dir, tmp_path, channels, match
):
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match=match):
        sess.add_stream(
            "p",
            raw_path=_pose_csv(tmp_path),
            importer="timeseries_csv",
            config={
                "time_column": "t_sec",
                "time_kind": "seconds",
                "channels": channels,
            },
        )


def test_auto_time_kind_refuses_an_unreadable_column_name(
    experiment_dir, recording_dir, tmp_path
):
    """Raise rather than guess: ``t_sec`` could be seconds or a frame index, and picking one
    silently makes the same command mean different things on different files."""
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="cannot tell what column 't_sec' holds"):
        sess.add_stream(
            "p",
            raw_path=_pose_csv(tmp_path),
            importer="timeseries_csv",
            config={"time_column": "t_sec"},
        )


def test_config_for_an_importer_that_declares_none_is_refused(
    experiment_dir, recording_dir, tmp_path
):
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="takes no config"):
        sess.add_stream(
            "b",
            raw_path=_behavior_csv(tmp_path),
            importer="behavior_csv",
            config={"time_column": "x"},
        )


def test_unknown_column_in_the_mapping_is_refused(
    experiment_dir, recording_dir, tmp_path
):
    sess = _recording(experiment_dir)
    with pytest.raises(ValueError, match="not in the CSV header"):
        sess.add_stream(
            "p",
            raw_path=_pose_csv(tmp_path),
            importer="timeseries_csv",
            config={
                "time_column": "t_sec",
                "time_kind": "seconds",
                "channels": [{"column": "nope", "name": "n"}],
            },
        )


# -- saved presets, applied across recordings (Section 3) -------------------- #


def _experiment_with_recordings(tmp_path, *names):
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path / "proj")
    for name in names:
        proj.create_recording(name)
    return proj


def _pose_for(tmp_path, rid, columns="t_sec,nose_x,nose_y,extra"):
    p = tmp_path / f"pose_{rid}.csv"
    body = ",".join("1" for _ in columns.split(",")[1:])
    p.write_text(f"{columns}\n0,{body}\n0.1,{body}\n", encoding="utf-8")
    return p


PRESET_CONFIG = {
    "time_column": "t_sec",
    "time_kind": "seconds",
    "channels": [
        {"column": "nose_x", "name": "nose_x"},
        {"column": "nose_y", "name": "nose_y"},
    ],
}


def test_a_preset_saved_on_one_recording_applies_to_another(tmp_path):
    """The whole point of a preset, and the shape nothing tested before.

    Every earlier fixture saved and applied on the *same* recording, so the case that
    matters — a mapping worked out once and reused on the next sixty — was never exercised.
    """
    proj = _experiment_with_recordings(tmp_path, "r1", "r2")
    r1, r2 = proj.resolve_recording("r1"), proj.resolve_recording("r2")
    r1.add_stream(
        "pose",
        raw_path=_pose_for(tmp_path, "r1"),
        importer="timeseries_csv",
        config=PRESET_CONFIG,
    )
    r1.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)

    r2.add_stream_from_preset("pose", _pose_for(tmp_path, "r2"), "dlc")

    assert r2.streams[0]["columns"] == r1.streams[0]["columns"]
    assert r2.streams[0]["metadata"]["time_column"] == "t_sec"


def test_a_preset_applies_to_a_csv_whose_columns_are_reordered(tmp_path):
    """A preset names columns, not positions — the next recording's export may differ."""
    proj = _experiment_with_recordings(tmp_path, "r1")
    proj.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)
    reordered = tmp_path / "odd.csv"
    reordered.write_text(
        "nose_y,extra,t_sec,nose_x\n2,9,0,1\n5,9,0.1,4\n", encoding="utf-8"
    )

    proj.resolve_recording("r1").add_stream_from_preset("pose", reordered, "dlc")

    entry = proj.resolve_recording("r1").streams[0]
    assert entry["columns"] == [
        "t_ns",
        "nose_x",
        "nose_y",
    ]  # the preset's order, not the file's
    assert proj.resolve_recording("r1").stream_events("pose") == [
        (0, 1.0, 2.0),
        (100_000_000, 4.0, 5.0),
    ]


def test_the_experiment_and_recording_share_one_saved_set(tmp_path):
    proj = _experiment_with_recordings(tmp_path, "r1")
    proj.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)

    assert [p["name"] for p in proj.resolve_recording("r1").list_stream_presets()] == [
        "dlc"
    ]


def test_apply_stream_preset_takes_an_offset_per_recording(tmp_path):
    """Alignment is per recording even when the mapping is not: each video and logger
    pair has its own skew, so one number for all of them is one recording's answer."""
    proj = _experiment_with_recordings(tmp_path, "r1", "r2", "r3")
    proj.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)

    skipped = proj.apply_stream_preset(
        "dlc",
        [
            ("r1", _pose_for(tmp_path, "r1"), 3_480_000_000),
            ("r2", _pose_for(tmp_path, "r2"), 3_512_000_000),
            ("r3", _pose_for(tmp_path, "r3"), None),
        ],
        name="pose",
    )

    assert skipped == []
    offsets = [
        proj.resolve_recording(r).streams[0]["alignment"]["offset_ns"]
        for r in ("r1", "r2", "r3")
    ]
    assert offsets == [3_480_000_000, 3_512_000_000, 0]


def test_apply_stream_preset_skips_a_bad_file_and_keeps_going(tmp_path):
    """One unreadable file must not sink a sixty-recording batch."""
    proj = _experiment_with_recordings(tmp_path, "r1", "r2")
    proj.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)
    short = tmp_path / "short.csv"
    short.write_text("t_sec,nose_x\n0,1\n", encoding="utf-8")  # no nose_y

    skipped = proj.apply_stream_preset(
        "dlc",
        [("r1", _pose_for(tmp_path, "r1"), None), ("r2", short, None)],
        name="pose",
    )

    assert [ref for ref, _ in skipped] == ["r2"]
    assert "nose_y" in skipped[0][1]
    assert proj.resolve_recording("r1").streams  # the good one still landed
    assert not proj.resolve_recording("r2").streams


def test_apply_stream_preset_reports_progress_per_recording(tmp_path):
    """A batch over a whole experiment is a long op, so it reports (the long-op rule)."""
    proj = _experiment_with_recordings(tmp_path, "r1", "r2")
    proj.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)
    seen = []

    proj.apply_stream_preset(
        "dlc",
        [
            ("r1", _pose_for(tmp_path, "r1"), None),
            ("r2", _pose_for(tmp_path, "r2"), None),
        ],
        name="pose",
        progress=lambda i, n, ref, frac, status: seen.append((i, n, ref, frac, status)),
    )

    assert [(i, n, ref, status) for i, n, ref, _f, status in seen] == [
        (0, 2, "r1", "importing"),
        (0, 2, "r1", "done"),
        (1, 2, "r2", "importing"),
        (1, 2, "r2", "done"),
    ]


def _give_video_file(proj, rid, path):
    data = recording_store.read_recording(proj.get_recording(rid).dir)
    data["video"] = {"files": [{"index": 0, "origin_name": path}]}
    recording_store.write_recording(proj.get_recording(rid).dir, data)


def test_apply_stream_preset_keys_rows_on_the_video_file_name(tmp_path):
    """#330, the batch half: a logger table made on the rig names the video it ran
    beside, not an id the tool minted later. Same rule as a metadata import — base name,
    any case, extension optional — and a name shared across folders is refused per row
    rather than guessed, so the rest of the batch still lands."""
    proj = _experiment_with_recordings(tmp_path, "r1", "r2", "r3", "r4")
    _give_video_file(proj, "r1", "/rig1/m27_day3.mp4")
    _give_video_file(proj, "r2", "/rig1/m27_day4.mp4")
    _give_video_file(proj, "r3", "/rig1/dup.mp4")
    _give_video_file(proj, "r4", "/rig2/dup.mp4")
    proj.save_stream_preset("dlc", "timeseries_csv", PRESET_CONFIG)

    skipped = proj.apply_stream_preset(
        "dlc",
        [
            ("M27_DAY3.MP4", _pose_for(tmp_path, "a"), 1_000_000_000),
            ("m27_day4", _pose_for(tmp_path, "b"), None),
            ("dup", _pose_for(tmp_path, "c"), None),
            ("ghost.mp4", _pose_for(tmp_path, "d"), None),
        ],
        name="pose",
        by_video=True,
    )

    assert skipped == [
        ("dup", "video file 'dup' matches several recordings: 'r3', 'r4'"),
        ("ghost.mp4", "unknown video file 'ghost.mp4' — not in this experiment"),
    ]
    assert proj.get_recording("r1").streams[0]["alignment"]["offset_ns"] == 1e9
    assert proj.get_recording("r2").streams
    assert not proj.get_recording("r3").streams
    assert not proj.get_recording("r4").streams


def test_an_unknown_preset_imports_nothing(tmp_path):
    """Checked before the first recording is touched (Tenet 10), not on the way past it."""
    proj = _experiment_with_recordings(tmp_path, "r1")

    with pytest.raises(ValueError, match="no stream preset"):
        proj.apply_stream_preset(
            "nope", [("r1", _pose_for(tmp_path, "r1"), None)], name="pose"
        )

    assert not proj.resolve_recording("r1").streams


def test_a_presets_own_offset_is_used_when_none_is_given(tmp_path):
    proj = _experiment_with_recordings(tmp_path, "r1", "r2")
    proj.save_stream_preset(
        "rig", "timeseries_csv", PRESET_CONFIG, offset_ns=1_000_000_000
    )

    proj.resolve_recording("r1").add_stream_from_preset(
        "pose", _pose_for(tmp_path, "r1"), "rig"
    )
    proj.resolve_recording("r2").add_stream_from_preset(
        "pose", _pose_for(tmp_path, "r2"), "rig", offset_ns=7
    )

    assert proj.resolve_recording("r1").streams[0]["alignment"]["offset_ns"] == 1e9
    assert proj.resolve_recording("r2").streams[0]["alignment"]["offset_ns"] == 7
