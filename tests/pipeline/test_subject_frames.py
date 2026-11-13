"""Subject frames and example-frame sampling (§26): the arithmetic, then the two strategies over
a real recording with a tracking run and a video."""

import numpy as np
import pytest

from detail.core.skeleton import Skeleton
from detail.pipeline import subject_frames as sf
from detail.storage import example_sets
from detail.storage import metadata as metadata_store
from tests import experiments

SK = Skeleton(names=("nose", "tail_base"), edges=(("nose", "tail_base"),))


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
def test_allocation_is_proportional_to_the_candidates():
    assert sf._allocate([100, 300], 4) == [1, 3]


def test_allocation_gives_each_source_one_while_n_allows():
    assert sf._allocate([1000, 1], 2) == [1, 1]
    assert sf._allocate([5, 0, 5], 2) == [1, 0, 1]


def test_allocation_never_exceeds_what_a_source_has():
    assert sf._allocate([2, 10], 12) == [2, 10]
    assert sum(sf._allocate([2, 10], 50)) == 12
    assert sf._allocate([0, 0], 3) == [0, 0]


def _cands(n, part_id="part_001"):
    return [
        {"part_id": part_id, "frame_idx": i, "t_ns": i * 1000, "track_id": 0}
        for i in range(n)
    ]


def test_spread_takes_one_frame_from_each_time_bin():
    picks = sf._spread(_cands(100), 4, np.random.default_rng(0))
    idx = [p["frame_idx"] for p in picks]
    assert len(idx) == 4
    for lo, hi, i in zip((0, 25, 50, 75), (25, 50, 75, 100), idx):
        assert lo <= i < hi


def test_spreading_more_frames_than_exist_takes_each_once():
    picks = sf._spread(_cands(3), 10, np.random.default_rng(0))
    assert sorted(p["frame_idx"] for p in picks) == [0, 1, 2]


def test_choose_recordings_is_stratified_and_seeded():
    ids = [f"recording_{i:03d}" for i in range(40)]
    picked = sf.choose_recordings(ids, 4, seed=1)
    assert len(picked) == 4
    for lo, hi, rec in zip((0, 10, 20, 30), (10, 20, 30, 40), picked):
        assert lo <= ids.index(rec) < hi
    assert sf.choose_recordings(ids, 4, seed=1) == picked
    assert sf.choose_recordings(ids, 40) == ids
    assert sf.choose_recordings(ids, 99) == ids
    with pytest.raises(ValueError, match="1 or more"):
        sf.choose_recordings(ids, 0)


def _setups(sizes: dict[str, int]) -> tuple[list[str], dict[str, str]]:
    ids, groups = [], {}
    for value, n in sizes.items():
        for _ in range(n):
            rid = f"recording_{len(ids):03d}"
            ids.append(rid)
            groups[rid] = value
    return ids, groups


@pytest.mark.parametrize(
    ("sizes", "k", "want"),
    [
        ({"1": 10, "2": 8, "3": 10, "4": 8}, 8, {"1": 2, "2": 2, "3": 2, "4": 2}),
        # a value with too few gives its unused share to the others
        ({"1": 10, "2": 1, "3": 10}, 9, {"1": 4, "2": 1, "3": 4}),
        # a remainder goes to the values with the most recordings left
        ({"1": 4, "2": 9, "3": 4}, 7, {"1": 2, "2": 3, "3": 2}),
    ],
)
def test_choose_recordings_splits_evenly_across_metadata_values(sizes, k, want):
    ids, groups = _setups(sizes)
    picked = sf.choose_recordings(ids, k, seed=3, groups=groups)
    got = {value: sum(groups[r] == value for r in picked) for value in sizes}
    assert got == want
    assert picked == [r for r in ids if r in picked], "not in list order"
    assert sf.choose_recordings(ids, k, seed=3, groups=groups) == picked


def test_choose_recordings_by_value_ignores_list_position():
    # imported in one block per setup: by position alone, 4 of 40 would miss setups entirely
    ids, groups = _setups({"A": 34, "B": 2, "C": 2, "D": 2})
    by_position = sf.choose_recordings(ids, 4, seed=0)
    by_value = sf.choose_recordings(ids, 4, seed=0, groups=groups)
    assert {groups[r] for r in by_position} != {"A", "B", "C", "D"}
    assert {groups[r] for r in by_value} == {"A", "B", "C", "D"}


def test_a_recording_with_no_value_is_its_own_group():
    ids, groups = _setups({"A": 6})
    ids.append("recording_blank")
    picked = sf.choose_recordings(ids, 2, seed=0, groups=groups)
    assert "recording_blank" in picked


def test_spreading_by_an_unknown_column_names_the_columns(tmp_path):
    metadata_store.upsert_row(tmp_path, "recording_000", {"setup": "1"})
    assert sf.recording_groups(tmp_path, "setup") == {"recording_000": "1"}
    with pytest.raises(ValueError, match=r"no metadata column 'box'.*'setup'"):
        sf.recording_groups(tmp_path, "box")


def test_strategy_ids_come_from_the_one_table():
    assert sf.STRATEGIES == tuple(s.id for s in sf.STRATEGY_TABLE)
    assert len(set(sf.STRATEGIES)) == len(sf.STRATEGIES)
    assert sf.DEFAULT_STRATEGY in sf.STRATEGIES


def test_choose_takes_the_typical_rows_then_farthest_first():
    feats = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    weights = np.ones(5)
    typical, varied = sf.choose(feats, weights, np.empty((0, 1)), 1, 2)
    # frame-weighted mean is 4.8; row 2 (2.0) is nearest it
    assert typical == [2]
    # farthest from the typical pick first (11.0), then farthest from both (0.0)
    assert varied == [4, 0]


def test_choose_without_typical_starts_farthest_from_the_mean():
    feats = np.array([[0.0], [1.0], [9.0]])
    typical, varied = sf.choose(feats, np.ones(3), np.empty((0, 1)), 0, 1)
    assert typical == []
    assert varied == [2]


def test_choose_counts_anchors_as_already_in_the_set():
    feats = np.array([[0.0], [5.0], [10.0]])
    anchors = np.array([[10.0]])
    _typical, varied = sf.choose(feats, np.ones(3), anchors, 0, 1)
    assert varied == [0]


def test_kmeans_separates_two_clusters():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    labels, centres = sf.kmeans(x, 2, np.random.default_rng(0))
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1
    assert labels[0] != labels[20]
    reps = sf.representatives(x, 2, np.random.default_rng(0))
    assert sorted(size for _, size in reps) == [20, 20]


def test_subject_frames_unions_one_tracks_boxes_in_a_frame():
    det = {
        "frame_idx": np.array([0, 0, 2, 3]),
        "t_ns": np.array([0, 0, 2000, 3000]),
        "bbox": np.array(
            [
                [0.1, 0.1, 0.3, 0.3],
                [0.2, 0.2, 0.5, 0.6],
                [0.0, 0.0, 0.1, 0.1],
                [0.5, 0.5, 0.9, 0.9],
            ]
        ),
    }
    object_id = np.array([0, 0, 1, 0])
    rows = sf.subject_frames(det, object_id, 0, 1)
    assert [r["frame_idx"] for r in rows] == [0, 3]
    assert rows[0]["bbox"] == (0.1, 0.1, 0.5, 0.6)
    assert rows[0]["rows"] == [0, 1]
    assert [r["frame_idx"] for r in sf.subject_frames(det, object_id, 0, 3)] == [0, 3]


def test_shape_row_tells_a_bar_from_a_square():
    square = np.zeros((40, 40), bool)
    square[10:30, 10:30] = True
    bar = np.zeros((40, 40), bool)
    bar[18:22, 4:36] = True
    s = sf.shape_row(square, (0.25, 0.25, 0.75, 0.75))
    b = sf.shape_row(bar, (0.1, 0.45, 0.9, 0.55))
    assert s[1] < 0.05  # a square has no long axis
    assert b[1] > 0.8  # a bar is elongated, along x (cos 2a = 1)
    assert b[2] > 0.7
    assert sf.shape_row(None, (0.1, 0.1, 0.5, 0.3)).shape == (6,)


# --------------------------------------------------------------------------- #
# Over a real recording
# --------------------------------------------------------------------------- #
N_FRAMES = 5


@pytest.fixture
def tracked(processed_recording):
    """A second recording with one track on frames 0..4, a different posture each frame, and a
    real 40x40 video to decode from."""
    built = processed_recording
    detections, object_ids = [], []
    for frame in range(N_FRAMES):
        c0 = 4 + frame * 2
        mask = built.square(10, 30, c0, c0 + 8 + frame * 4)
        bbox = (c0 / 40, 10 / 40, (c0 + 8 + frame * 4) / 40, 30 / 40)
        detections.append((frame, 0, bbox, mask))
        object_ids.append(0)
    recording = built.add_recording("recording_002", detections, object_ids)
    experiments.write_video(recording.dir / "video" / "part_001.mp4", N_FRAMES)
    built.tracked_recording = recording
    built.sources = {"recording_002": "track_01"}
    return built


def test_tracked_frames_are_keyed_by_part_id(tracked):
    rows = sf.tracked_frames(tracked.tracked_recording.dir, "track_01")
    assert [r["frame_idx"] for r in rows] == list(range(N_FRAMES))
    assert {r["part_id"] for r in rows} == {"part_001"}
    assert rows[0]["track_id"] == 0
    assert rows[1]["bbox"] == pytest.approx((6 / 40, 10 / 40, 18 / 40, 30 / 40))
    assert sf.tracked_frames(tracked.tracked_recording.dir, "track_01", [7]) == []
    with pytest.raises(ValueError, match="no tracking run 'track_09'"):
        sf.tracked_frames(tracked.tracked_recording.dir, "track_09")


def test_one_frames_lookup_agrees_with_the_whole_scan(tracked):
    """`tracked_frame` reads one part's frame column and that frame's rows; whatever it
    returns has to be the row the whole-run scan returns, for every tracked frame."""
    rec = tracked.tracked_recording.dir
    rows = sf.tracked_frames(rec, "track_01")
    assert rows
    for row in rows:
        one = sf.tracked_frame(
            rec, "track_01", row["part_id"], row["frame_idx"], row["track_id"]
        )
        assert one == row
    first = rows[0]
    assert sf.tracked_frame(rec, "track_01", first["part_id"], 10**6, 0) is None
    assert sf.tracked_frame(rec, "track_01", "part_999", first["frame_idx"], 0) is None
    assert (
        sf.tracked_frame(rec, "track_01", first["part_id"], first["frame_idx"], 7)
        is None
    ), "another track has no subject there"
    with pytest.raises(ValueError, match="no tracking run 'track_09'"):
        sf.tracked_frame(rec, "track_09", first["part_id"], 0, 0)


def test_spread_sampling_copies_frames_with_their_boxes(tracked):
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)

    frames = sf.sample_frames(
        tracked.experiment.dir, created.id, tracked.sources, 3, strategy="spread"
    )

    assert len(frames) == 3
    assert len({f.key() for f in frames}) == 3
    for frame in frames:
        assert frame.sampled_by == "spread"
        assert frame.part_id == "part_001"
        assert frame.recording_id == "recording_002"
        assert frame.t_ns == frame.frame_idx * 1000
        c0 = 4 + frame.frame_idx * 2
        assert frame.bbox == pytest.approx(
            (c0 / 40, 0.25, (c0 + 8 + frame.frame_idx * 4) / 40, 0.75)
        )
        rgb = example_sets.load_image(tracked.experiment.dir, created.id, frame.image)
        assert rgb.shape == (40, 40, 3)
        assert abs(float(rgb.mean()) - (20 + frame.frame_idx * 45)) < 12


def test_sampling_never_draws_a_frame_the_set_holds(tracked):
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    first = sf.sample_frames(
        tracked.experiment.dir, created.id, tracked.sources, 3, strategy="spread"
    )
    created.frames.extend(first)
    example_sets.write_set(tracked.experiment.dir, created)

    rest = sf.sample_frames(
        tracked.experiment.dir, created.id, tracked.sources, 5, strategy="spread"
    )

    assert len(rest) == N_FRAMES - 3
    assert {f.key() for f in rest}.isdisjoint({f.key() for f in first})
    created.frames.extend(rest)
    example_sets.write_set(tracked.experiment.dir, created)
    with pytest.raises(ValueError, match="no tracked frames left"):
        sf.sample_frames(
            tracked.experiment.dir, created.id, tracked.sources, 1, strategy="spread"
        )


@pytest.mark.parametrize("strategy", ["spread", "varied"])
def test_a_frame_refused_for_every_bucket_is_never_drawn(
    tracked, monkeypatch, strategy
):
    monkeypatch.setattr(sf, "installed_backbone", lambda: None)
    root = tracked.experiment.dir
    created = example_sets.create_set(root, "rats", SK)
    everything = sf.sample_frames(
        root, created.id, tracked.sources, N_FRAMES, strategy="spread"
    )
    keys = [f.key() for f in everything]
    created.reject(keys[0])  # every bucket
    created.reject(
        keys[1], "toward"
    )  # one bucket: a bucketless strategy may still draw it
    example_sets.write_set(root, created)

    drawn = sf.sample_frames(
        root, created.id, tracked.sources, N_FRAMES, strategy=strategy, n_typical=0
    )

    drawn_keys = {f.key() for f in drawn}
    assert keys[0] not in drawn_keys
    assert keys[1] in drawn_keys


def test_sampling_rejects_a_bad_request(tracked):
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    root = tracked.experiment.dir
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        sf.sample_frames(root, created.id, tracked.sources, 1, strategy="random")
    with pytest.raises(ValueError, match="at least one frame"):
        sf.sample_frames(root, created.id, tracked.sources, 0, strategy="spread")
    with pytest.raises(ValueError, match="at least one recording"):
        sf.sample_frames(root, created.id, {}, 1, strategy="spread")
    with pytest.raises(ValueError, match="n_typical"):
        sf.sample_frames(root, created.id, tracked.sources, 1, n_typical=-1)
    with pytest.raises(ValueError, match="from_recordings"):
        sf.sample_frames(root, created.id, tracked.sources, 1, from_recordings=0)


def test_pick_frame_adds_the_navigated_frame_once(tracked):
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    root = tracked.experiment.dir

    frame = sf.pick_frame(
        root, created.id, "recording_002", "track_01", "part_001", 2, 0
    )

    assert frame.frame_idx == 2 and frame.t_ns == 2000 and frame.sampled_by is None
    assert (example_sets.set_dir(root, created.id) / frame.image).is_file()
    created.frames.append(frame)
    example_sets.write_set(root, created)
    with pytest.raises(ValueError, match="already in example set"):
        sf.pick_frame(root, created.id, "recording_002", "track_01", "part_001", 2, 0)
    with pytest.raises(ValueError, match="no subject on frame 2"):
        sf.pick_frame(root, created.id, "recording_002", "track_01", "part_001", 2, 5)
    with pytest.raises(ValueError, match="no subject on frame 40"):
        sf.pick_frame(root, created.id, "recording_002", "track_01", "part_001", 40, 0)


def test_varied_without_a_backbone_picks_typical_then_varied_from_the_shapes(
    tracked, monkeypatch
):
    monkeypatch.setattr(sf, "installed_backbone", lambda: None)
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    stages = []

    frames = sf.sample_frames(
        tracked.experiment.dir,
        created.id,
        tracked.sources,
        2,
        n_typical=1,
        progress=lambda stage, fraction, detail: stages.append(stage),
    )

    assert [f.sampled_by for f in frames] == [
        "typical:shape",
        "varied:shape",
        "varied:shape",
    ]
    assert len({f.key() for f in frames}) == 3
    assert ["reading shapes", "grouping", "copying frames"] == [
        s for i, s in enumerate(stages) if s not in stages[:i]
    ]


def test_a_second_varied_pass_adds_no_typical_frames(tracked, monkeypatch):
    monkeypatch.setattr(sf, "installed_backbone", lambda: None)
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    first = sf.sample_frames(
        tracked.experiment.dir, created.id, tracked.sources, 1, n_typical=1
    )
    created.frames.extend(first)
    example_sets.write_set(tracked.experiment.dir, created)

    second = sf.sample_frames(
        tracked.experiment.dir, created.id, tracked.sources, 2, n_typical=5
    )

    assert [f.sampled_by for f in second] == ["varied:shape", "varied:shape"]
    assert {f.key() for f in second}.isdisjoint({f.key() for f in first})


def test_varied_with_a_backbone_embeds_every_representative_and_anchor(
    tracked, monkeypatch
):
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    anchor = sf.pick_frame(
        tracked.experiment.dir,
        created.id,
        "recording_002",
        "track_01",
        "part_001",
        0,
        0,
    )
    created.frames.append(anchor)
    example_sets.write_set(tracked.experiment.dir, created)
    seen = {}

    def fake_embed(images, bboxes, variant, device, progress=None):
        seen["n"] = len(images)
        seen["variant"] = variant
        seen["shapes"] = {img.shape for img in images}
        # a vector per subject from its box, so "far apart" is well defined
        return np.array([[b[0], b[2]] for b in bboxes], dtype=np.float64)

    monkeypatch.setattr(sf, "installed_backbone", lambda: "vit_base")
    monkeypatch.setattr(sf, "embed_subjects", fake_embed)

    frames = sf.sample_frames(
        tracked.experiment.dir, created.id, tracked.sources, 2, device="cpu"
    )

    # four fresh representatives (frames 1..4) plus the one anchor frame reached the model
    assert seen["n"] == 5
    assert seen["variant"] == "vit_base"
    assert seen["shapes"] == {(40, 40, 3)}
    assert [f.sampled_by for f in frames] == ["varied:dinov3_vit_base"] * 2
    assert anchor.key() not in {f.key() for f in frames}


def test_a_missing_video_is_refused_before_any_pick(tracked, monkeypatch):
    monkeypatch.setattr(sf, "installed_backbone", lambda: None)
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    (tracked.tracked_recording.dir / "video" / "part_001.mp4").unlink()

    with pytest.raises(FileNotFoundError, match="video not present"):
        sf.sample_frames(tracked.experiment.dir, created.id, tracked.sources, 1)
    assert not (
        example_sets.set_dir(tracked.experiment.dir, created.id) / "frames"
    ).exists()


def test_subject_crop_is_square_and_padded_off_the_frame():
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[:, :, 1] = 200
    crop = sf.subject_crop(rgb, (0.0, 0.0, 0.5, 0.5))
    assert crop.shape[0] == crop.shape[1]
    assert crop[0, 0, 1] == 114  # padding where the crop runs off the top-left
    assert crop[-1, -1, 1] == 200


# --- the shape pass, fast and the same (reported from the app: minutes per Add frames) ------ #
def _reference_shape_row(mask, bbox):
    """`shape_row` as it was: the hull and box from every pixel (1.54ms a row on real masks)."""
    import cv2

    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    c0, r0 = max(int(x0 * w) - 1, 0), max(int(y0 * h) - 1, 0)
    c1, r1 = min(int(np.ceil(x1 * w)) + 1, w), min(int(np.ceil(y1 * h)) + 1, h)
    crop = mask[r0:r1, c0:c1].astype(np.uint8)
    if not crop.any():
        crop = mask.astype(np.uint8)
    m = cv2.moments(crop, binaryImage=True)
    area = float(m["m00"])
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / area
    lo, hi = np.linalg.eigvalsh(cov)
    e = 1.0 - float(np.sqrt(max(lo, 0.0) / hi)) if hi > 0 else 0.0
    angle2 = float(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    hull_area = float(cv2.contourArea(cv2.convexHull(cv2.findNonZero(crop))))
    ys, xs = np.nonzero(crop)
    box_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return np.array(
        [
            np.log(area),
            e,
            e * np.cos(angle2),
            e * np.sin(angle2),
            min(area / hull_area, 1.0) if hull_area > 0 else 1.0,
            area / box_area,
        ]
    )


def _shapes():
    import cv2

    blob = np.zeros((60, 80), np.uint8)
    cv2.ellipse(blob, (40, 30), (25, 9), 30, 0, 360, 1, -1)
    ring = blob.copy()
    cv2.ellipse(ring, (40, 30), (10, 4), 30, 0, 360, 0, -1)  # a hole
    pair = np.zeros((60, 80), np.uint8)
    pair[10:20, 5:25] = 1
    pair[35:55, 50:70] = 1  # two pieces of one subject
    dot = np.zeros((60, 80), np.uint8)
    dot[30, 40] = 1
    line = np.zeros((60, 80), np.uint8)
    line[20, 10:70] = 1
    rng = np.random.default_rng(0)
    speckle = (rng.random((60, 80)) > 0.7).astype(np.uint8)
    return {
        "blob": blob,
        "ring": ring,
        "pair": pair,
        "dot": dot,
        "line": line,
        "speckle": speckle,
    }


@pytest.mark.parametrize("name", list(_shapes()))
def test_the_fast_shape_row_is_the_old_one_exactly(name):
    """The hull from the outer outlines, not every pixel: the same numbers on holes, several
    pieces, a single pixel, a line and noise (and on 3,000 real 600x800 masks, all identical).
    """
    mask = _shapes()[name].astype(bool)
    bbox = (0.0, 0.0, 1.0, 1.0)
    np.testing.assert_array_equal(
        sf.shape_row(mask, bbox), _reference_shape_row(mask, bbox)
    )


def test_tracked_rows_read_the_masks_they_use_and_no_others(tracked):
    """The mask strings stay on disk until a row asks (1.65s of a 2.2s part read on real data,
    for ~12 masks in Test cut-off), and what a row gets is what the whole read gave."""
    from detail.storage.zarr_io import DetectionPartReader

    rows, size = sf._tracked_rows(tracked.tracked_recording.dir, "track_01", None)
    assert isinstance(rows[0]["det"]["mask_rle"], sf._LazyColumn)
    det_path = next(
        (tracked.tracked_recording.dir / "detections").glob("*/part_001.zarr")
    )
    whole = DetectionPartReader(det_path).read()
    for row in rows:
        lazy = sf._subject_mask(row, size)
        eager = sf.subject_mask(whole, row["rows"], size)
        np.testing.assert_array_equal(lazy, eager)


def test_reading_shapes_moves_inside_one_recording(tracked, monkeypatch):
    """The bar moved only when a recording finished, seconds apart on real data: with one
    recording it sat at 0 for the whole pass. The inner pair is asserted on one recording.
    """
    from detail.core import progress as core_progress

    monkeypatch.setattr(core_progress, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setattr(sf, "installed_backbone", lambda: None)
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    events = []
    sf.sample_frames(
        tracked.experiment.dir,
        created.id,
        tracked.sources,
        1,
        n_typical=1,
        progress=lambda stage, fraction, detail: events.append(
            (stage, fraction, detail)
        ),
    )
    shapes = [(f, d) for s, f, d in events if s == "reading shapes"]
    inner = [f for f, d in shapes if 0 < f < 1]
    assert inner and inner == sorted(inner), shapes
    assert any(
        "frames" in d for _f, d in shapes
    ), "names the frame count, not only the recording"


def test_a_cancelled_draw_leaves_no_pictures_and_no_frames(tracked, monkeypatch):
    """Cancel arrives as `OperationCancelled` from the progress callback (the GUI's Cancel or a
    closed window). The pictures this draw had copied are removed; the set is untouched.
    """
    from detail.core.progress import OperationCancelled

    monkeypatch.setattr(sf, "installed_backbone", lambda: None)
    created = example_sets.create_set(tracked.experiment.dir, "rats", SK)
    frames_dir = example_sets.set_dir(tracked.experiment.dir, created.id) / "frames"

    def cancel_mid_copy(stage, fraction, detail):
        if stage == "copying frames" and fraction > 0:
            raise OperationCancelled(stage)

    with pytest.raises(OperationCancelled):
        sf.sample_frames(
            tracked.experiment.dir,
            created.id,
            tracked.sources,
            2,
            n_typical=1,
            progress=cancel_mid_copy,
        )
    assert not list(frames_dir.glob("*.png")), "a copied picture was left behind"
    assert example_sets.read_set(tracked.experiment.dir, created.id).frames == []
