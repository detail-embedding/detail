"""`frame_slices.from_tracking` over a recording of more than one video part.

**The class: a cumulative part offset added to a timestamp map that already carries it.** A
part's active timestamp CSV is written recording-global (`Recording.add_video_part` stores
`local_t_ns + offset_ns`, Tenet 8: time is nanoseconds from the start of the *first* part), but
the loader added the earlier parts' elapsed time again, so every row after part 1 sat a whole
earlier-parts-length too late. `between()` then searched the right span in the wrong place and
returned nothing, and a measurement of a bout in part 2 or later came out empty or NaN with no
error: the number was plausible, the rows were just somewhere else.

How it hid: no test called the real loader (`measurements._load_slice` is monkeypatched in
`tests/gui/test_measurements.py`), and every fixture was a single part, where the offset is zero
and the bug has nothing to add.

The fixture writes part 2's timestamps the way the import does. A fixture whose second part
restarted at zero would pass the old code and encode the opposite of the contract.

**The sibling class: a part-local frame number compared with a recording-wide one.** An
embedding store's `frame_idx` is the detection part's own and restarts at zero in every video
part, while a slice's frames are recording-wide. `_embeddings_for` merged every part's local numbers
into one list and joined it against the slice, so for a bout after part 1 it found nothing (the
appearance change came out NaN) or found another part's frame with the same number (the wrong
embedding, with no error). The same single-part fixtures hid it. `_series_parts` now carries each
part's offset.
"""

import numpy as np
import pytest
import yaml

from detail.api.frame_slices import between, from_tracking
from detail.api.recording import Recording
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage.zarr_io import (
    AssignmentPartWriter,
    DetectionPartWriter,
    EmbeddingPartWriter,
    KinematicsPartWriter,
    kinematics_part_path,
)

PERIOD = 100_000_000  # 0.1 s per frame
FRAMES = 10  # per part
PRESENT = (3, 4)  # part-local frames the track is detected in, in every part


def _box(animal, part):
    """Animal 0 sits still; animal 1 comes 10 px closer in every part (centres on a line, y=10).

    So the distance between them is 60 px in part 1, 50 in part 2, 40 in part 3: a number that
    changes with the part, which is what lets a mix-up of parts show.
    """
    if animal == 0:
        return (0.0, 0.0, 0.2, 0.2)
    return (0.6 - 0.1 * part, 0.0, 0.8 - 0.1 * part, 0.2)


def _vector(part, frame):
    """An embedding that says which recording-wide frame it belongs to."""
    return np.array([part * FRAMES + frame + 1, 1.0, 0.0, 0.0], dtype=np.float32)


def _write_embeddings(sdir, stem, part, t_ns, series, *, sidecar):
    store = sdir / "embeddings" / "emb_01" / f"track_{series}" / f"{stem}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    with EmbeddingPartWriter(store, output_dim=4) as writer:
        for frame in PRESENT:
            writer.add_row(frame, int(t_ns[frame]), 0, _vector(part, frame))
    if sidecar:
        # The union of the two boxes, and a footprint that differs by part.
        with KinematicsPartWriter(kinematics_part_path(store)) as kin:
            for frame in PRESENT:
                kin.add_row(
                    frame,
                    int(t_ns[frame]),
                    0.4,
                    0.1,
                    0.05 + 0.01 * part,
                    np.array([0.0, 0.0, 0.8 - 0.1 * part, 0.2], dtype=np.float32),
                )


def _recording(tmp_path, n_parts, *, animals=(0,), embedded=None, sidecar=None):
    """`n_parts` parts of FRAMES frames each, `animals` seen at PRESENT in every part.

    ``embedded`` writes an embedding series beside it: ``"0"`` for an ordinary track's.
    """
    proj_dir = experiment_store.create_experiment(tmp_path, "p")
    recording_store.create_recording(proj_dir, "s1")
    recording = Recording(proj_dir, "s1")
    sdir = recording.dir
    ddir = sdir / "detections" / "det_01"
    adir = sdir / "tracking" / "track_01" / "assignments"
    ddir.mkdir(parents=True)
    adir.mkdir(parents=True)

    files = []
    for part in range(n_parts):
        stem = f"part_{part + 1:03d}"
        # Recording-global, as the import writes it: part k starts where part k-1 ended.
        t_ns = (part * FRAMES + np.arange(FRAMES, dtype=np.int64)) * PERIOD
        recording_store.write_timestamps_csv(
            sdir / "video" / "timestamps" / "generated" / f"{stem}.csv", t_ns
        )
        files.append(
            {
                "index": part,
                "file": f"video/{stem}.mp4",
                "timestamps": {
                    "generated": f"video/timestamps/generated/{stem}.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        )
        with DetectionPartWriter(ddir / f"{stem}.zarr") as writer:
            for frame in PRESENT:
                for animal in animals:
                    writer.add_row(
                        frame,
                        int(t_ns[frame]),
                        animal,
                        "mouse",
                        0.9,
                        _box(animal, part),
                        "",
                    )
        with AssignmentPartWriter(adir / f"{stem}.zarr") as writer:
            for _ in PRESENT:
                for animal in animals:
                    writer.add_object_id(animal)
        if embedded is not None:
            _write_embeddings(
                sdir,
                stem,
                part,
                t_ns,
                embedded,
                sidecar=bool(sidecar),
            )

    (sdir / "tracking" / "track_01" / "track.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "track_01",
                "detection_run_id": "det_01",
                "class_name": None,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data = recording_store.read_recording(sdir)
    data["video"] = {
        "storage": "default",
        "fps_container": 10.0,
        "fps_actual": 10.0,
        "fps_source": "container",
        "resolution": [100, 100],
        "total_duration_s": n_parts * FRAMES * PERIOD / 1e9,
        "files": files,
    }
    data.setdefault("detection_runs", []).append(
        {"id": "det_01", "status": "complete", "downsampling_run_id": None}
    )
    data.setdefault("tracking_runs", []).append(
        {"id": "track_01", "status": "complete", "detection_run_id": "det_01"}
    )
    recording_store.write_recording(sdir, data)
    return recording


def _global_t_ns(n_parts):
    return [
        (part * FRAMES + frame) * PERIOD for part in range(n_parts) for frame in PRESENT
    ]


@pytest.mark.parametrize("n_parts", [1, 2, 3])
def test_each_row_keeps_its_recording_global_time(tmp_path, n_parts):
    """The time on a row is the time in the part's own map, which is already global."""
    recording = _recording(tmp_path, n_parts)
    slice_ = from_tracking(recording.dir, tracking_run_id="track_01", track_id=0)
    assert slice_.t_ns.tolist() == _global_t_ns(n_parts)
    assert slice_.frame_idx.tolist() == [
        part * FRAMES + frame for part in range(n_parts) for frame in PRESENT
    ]


def test_a_bout_in_a_later_part_is_found_where_it_happened(tmp_path):
    """The user-visible half: a span in part 2, stated in recording time, has rows."""
    recording = _recording(tmp_path, 2)
    slice_ = from_tracking(recording.dir, tracking_run_id="track_01", track_id=0)
    start, end = (FRAMES + PRESENT[0]) * PERIOD, (FRAMES + PRESENT[1]) * PERIOD
    bout = between(slice_, start, end)
    assert len(bout) == len(PRESENT)
    assert bout.t_ns.tolist() == [start, end]


# --- the sibling: part-local frame numbers compared with recording-wide ones ------------- #
# A store's `frame_idx` restarts at zero in every video part, while a slice's frames are
# recording-wide. The embeddings join compared the two directly: a row of part 2 found nothing (a
# NaN appearance change) or found another part's frame with the same number (a wrong embedding).
WIDE_FRAMES = [part * FRAMES + frame for part in range(2) for frame in PRESENT]


def test_embeddings_follow_the_part_they_were_made_in(tmp_path):
    recording = _recording(tmp_path, 2, embedded="0")
    slice_ = from_tracking(
        recording.dir, tracking_run_id="track_01", track_id=0, embed_run_id="emb_01"
    )
    assert slice_.embeddings is not None
    assert slice_.embeddings[:, 0].tolist() == [f + 1 for f in WIDE_FRAMES]


# --- one animal from its sidecar, and the chooser ---------------------------------------- #
def test_one_animals_sidecar_gives_mask_centroid_and_area_on_the_wide_scale(tmp_path):
    from detail.api.frame_slices import from_kinematics

    recording = _recording(tmp_path, 2, embedded="0", sidecar=True)
    slice_ = from_kinematics(
        recording.dir, embed_run_id="emb_01", track_id=0, every_n_frames=3
    )
    assert slice_.frame_idx.tolist() == WIDE_FRAMES
    assert slice_.t_ns.tolist() == _global_t_ns(2)
    assert slice_.stride == 3, "the embedding run's cadence"
    assert slice_.centroid.ravel().tolist() == pytest.approx([40, 10] * 4)
    assert slice_.area.tolist() == pytest.approx([500, 500, 600, 600])
    assert slice_.embeddings[:, 0].tolist() == [f + 1 for f in WIDE_FRAMES]


def test_no_sidecar_is_none_not_an_error(tmp_path):
    from detail.api.frame_slices import from_kinematics

    recording = _recording(tmp_path, 1, embedded="0", sidecar=False)
    assert from_kinematics(recording.dir, embed_run_id="emb_01", track_id=0) is None


@pytest.mark.parametrize("sidecar", [True, False])
def test_track_slice_prefers_the_sidecar_and_falls_back_to_the_boxes(tmp_path, sidecar):
    from detail.api.frame_slices import track_slice

    recording = _recording(tmp_path, 2, embedded="0", sidecar=sidecar)
    record = {"id": "emb_01", "tracking_run_id": "track_01", "every_n_frames": 1}
    slice_ = track_slice(recording, record, 0)
    assert slice_.t_ns.tolist() == _global_t_ns(2)
    assert (slice_.centroid is not None) == sidecar
    assert slice_.bbox is not None
    assert track_slice(recording, {"id": "emb_01"}, 0) is None


def test_between_cuts_every_per_row_field_and_carries_the_rest():
    """By inspection, not a list: a field added to `FrameSlice` reaches every bout cut."""
    import dataclasses

    from detail.analysis.descriptors import FrameSlice

    n = 10
    rows = np.arange(n)
    whole = FrameSlice(
        frame_idx=rows,
        t_ns=rows.astype(np.int64) * 10,
        centroid=np.zeros((n, 2)),
        area=np.ones(n),
        bbox=np.zeros((n, 4)),
        embeddings=np.zeros((n, 3)),
        labels=rows,
        shape={"top": rows.astype(float), "bottom": rows.astype(float) + 1},
        body_length=42.0,
        camera_view="side",
        stride=2,
    )
    cut = between(whole, 20, 50)
    assert len(cut) == 4
    for f in dataclasses.fields(FrameSlice):
        value = getattr(cut, f.name)
        if isinstance(value, np.ndarray):
            assert value.shape[0] == 4, f.name
    assert cut.shape["top"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert (cut.body_length, cut.camera_view, cut.stride) == (42.0, "side", 2)
