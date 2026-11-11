"""`api.frame_slices.from_kinematics`: the second loader the module's docstring promised.

What it has to get right that `from_tracking` already does: pixels not normalized units, one
global frame axis across parts (offsets from each part's timestamp map, never from the sidecar's
own rows), the run's true row stride. What is its own: the sidecar's clock, and refusing to
invent pixels when the record carries no frame size.
"""

import numpy as np
import pytest
import yaml

from detail.api import frame_slices
from detail.pipeline import kinematics_backfill
from detail.storage import recording as recording_store
from detail.storage.zarr_io import KinematicsPartWriter, kinematics_part_path

from tests.experiments import write_embedding_run

EMB = "emb_01"


def _sidecar_run(recording_dir, frames, t_ns, *, cx, cy):
    """An embedding run over ``frames`` whose sidecar says the subject was at ``(cx, cy)``."""
    return write_embedding_run(
        recording_dir,
        np.zeros((len(frames), 4), np.float32),
        np.asarray(frames),
        np.asarray(t_ns),
        tracking_run_id="track_01",  # the recording-level id `processed_recording` writes
        cx=cx,
        cy=cy,
    )


def _set_embedding_cadence(run_dir, every_n_frames: int) -> None:
    path = run_dir / "embedding.yaml"
    meta = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    meta["every_n_frames"] = every_n_frames
    path.write_text(yaml.dump(meta, sort_keys=False), encoding="utf-8")


def _set_tracking_cadence(recording_dir, every_n_frames: int) -> None:
    data = recording_store.read_recording(recording_dir)
    data["tracking_runs"][0]["every_n_frames"] = every_n_frames
    recording_store.write_recording(recording_dir, data)


def _embedding_part(run_dir, name, frames, t_ns):
    """The embedding part a sidecar sits beside: the loader finds sidecars through them, so a
    sidecar with no part is not a sidecar at all."""
    from detail.storage.zarr_io import EmbeddingPartWriter

    with EmbeddingPartWriter(
        run_dir / "track_0" / f"{name}.zarr", output_dim=4
    ) as writer:
        for frame, t in zip(frames, t_ns):
            writer.add_row(int(frame), int(t), 0, np.zeros(4, np.float32))


def test_a_run_without_a_sidecar_is_none_not_an_error(processed_recording):
    sdir = processed_recording.recording.dir
    write_embedding_run(
        sdir, np.zeros((3, 4), np.float32), [0, 1, 2], [0, 1000, 2000],
        tracking_run_id="track_01",
    )  # fmt: skip
    assert frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0) is None


def test_the_sidecar_is_read_in_pixels_on_the_recordings_frame(processed_recording):
    """40x40 recording, so px == norm * 40: centroid, area and box all land in pixels, the
    clock is the sidecar's own, and both index columns are int64."""
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], [0, 2000, 4000], cx=[0.25, 0.5, 0.75], cy=[0.5] * 3)

    got = frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0)

    assert got is not None
    np.testing.assert_array_equal(got.frame_idx, [0, 2, 4])
    np.testing.assert_array_equal(got.t_ns, [0, 2000, 4000])
    assert got.frame_idx.dtype == np.int64 and got.t_ns.dtype == np.int64
    np.testing.assert_allclose(got.centroid, [[10, 20], [20, 20], [30, 20]])
    np.testing.assert_allclose(got.area, [0.05 * 40 * 40] * 3)  # the helper writes 0.05
    np.testing.assert_allclose(got.bbox, [[0, 0, 40, 40]] * 3)
    assert got.stride == 1
    assert got.embeddings is None and got.labels is None
    assert len(got) == 3


def test_the_stride_is_the_rows_true_step_in_video_frames(processed_recording):
    """The embed pass keeps ``frame % every_n_frames == 0`` on top of the tracking cadence, so
    rows sit at multiples of both: lcm(4, 3) = 12, and either alone is itself."""
    sdir = processed_recording.recording.dir
    run_dir = _sidecar_run(sdir, [0], [0], cx=[0.5], cy=[0.5])

    _set_tracking_cadence(sdir, 4)
    assert frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0).stride == 4
    _set_embedding_cadence(run_dir, 3)
    assert frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0).stride == 12
    _set_tracking_cadence(sdir, 1)
    assert frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0).stride == 3


def test_frame_idx_is_made_global_from_each_parts_timestamp_map(processed_recording):
    """Part 2's rows continue part 1's frame axis by part 1's *frame count* (5, from its
    timestamp map), not by its sidecar's last row (2). The sidecar's clock restarts with the
    part, and is kept as recorded."""
    sdir = processed_recording.recording.dir
    run_dir = _sidecar_run(sdir, [0, 2], [0, 2000], cx=[0.5, 0.5], cy=[0.5, 0.5])

    second = "video/timestamps/generated/part_002.csv"
    recording_store.write_timestamps_csv(
        sdir / second, np.arange(5, dtype=np.int64) * 1000
    )
    data = recording_store.read_recording(sdir)
    data["video"]["files"].append(
        {
            "index": 1,
            "file": "video/part_002.mp4",
            "timestamps": {"generated": second, "user": None, "active": "generated"},
        }
    )
    recording_store.write_recording(sdir, data)
    _embedding_part(run_dir, "part_002", [0, 2], [0, 2000])
    with KinematicsPartWriter(
        kinematics_part_path(run_dir / "track_0" / "part_002.zarr")
    ) as kin:
        for frame, t in ((0, 0), (2, 2000)):
            kin.add_row(frame, t, 0.5, 0.5, 0.05, [0.0, 0.0, 1.0, 1.0])

    got = frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0)

    np.testing.assert_array_equal(got.frame_idx, [0, 2, 5, 7])
    np.testing.assert_array_equal(got.t_ns, [0, 2000, 0, 2000])


def test_a_backfilled_sidecar_agrees_with_the_detection_store(processed_recording):
    """The same track read by both loaders: identical frames and boxes (the union box, in
    pixels), and a position within a pixel — the mask's centre of mass against the box's
    centre, on a filled square."""
    it = processed_recording
    moving = [
        (f, 0, (0.1 + 0.05 * f, 0.1, 0.5 + 0.05 * f, 0.5), it.square(4, 20, 4 + 2 * f, 20 + 2 * f))
        for f in range(5)
    ]  # fmt: skip
    recording = it.add_recording("recording_002", detections=moving, object_ids=[0] * 5)
    sdir = recording.dir
    write_embedding_run(
        sdir, np.zeros((5, 4), np.float32), list(range(5)), np.arange(5) * 1000,
        tracking_run_id="track_01",
    )  # fmt: skip
    assert kinematics_backfill.backfill_run(sdir, EMB) == 1

    from_store = frame_slices.from_tracking(
        sdir, tracking_run_id="track_01", track_id=0
    )
    from_sidecar = frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0)

    np.testing.assert_array_equal(from_sidecar.frame_idx, from_store.frame_idx)
    np.testing.assert_allclose(from_sidecar.bbox, from_store.bbox, atol=1e-6)
    np.testing.assert_allclose(from_sidecar.position, from_store.position, atol=1.0)
    assert from_sidecar.area is not None and (from_sidecar.area > 0).all()
    assert (
        from_store.area is None
    )  # the store loader leaves the mass-centre inputs unset


def test_a_record_without_a_frame_size_is_refused_rather_than_guessed(
    processed_recording,
):
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0], [0], cx=[0.5], cy=[0.5])
    data = recording_store.read_recording(sdir)
    del data["video"]["resolution"]
    recording_store.write_recording(sdir, data)

    with pytest.raises(ValueError, match="video.resolution"):
        frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0)


def test_a_sidecar_for_a_part_the_recording_does_not_list_is_refused(
    processed_recording,
):
    sdir = processed_recording.recording.dir
    run_dir = _sidecar_run(sdir, [0], [0], cx=[0.5], cy=[0.5])
    _embedding_part(run_dir, "part_009", [0], [0])
    with KinematicsPartWriter(
        kinematics_part_path(run_dir / "track_0" / "part_009.zarr")
    ) as kin:
        kin.add_row(0, 0, 0.5, 0.5, 0.05, [0.0, 0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="part_009"):
        frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0)


def test_between_cuts_a_clip_out_by_the_sidecars_clock(processed_recording):
    sdir = processed_recording.recording.dir
    _sidecar_run(sdir, [0, 2, 4], [0, 2000, 4000], cx=[0.2, 0.5, 0.8], cy=[0.5] * 3)
    whole = frame_slices.from_kinematics(sdir, embed_run_id=EMB, track_id=0)

    clip = frame_slices.between(whole, 2000, 4000)

    np.testing.assert_array_equal(clip.frame_idx, [2, 4])
    np.testing.assert_allclose(clip.centroid[:, 0], [20, 32])
    assert clip.stride == whole.stride
