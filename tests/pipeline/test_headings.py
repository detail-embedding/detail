"""Camera-relative headings for the heading sampler (§26 *Headings*): the rules on synthetic
bodies and tails, then the cache and the strategy over a real recording, a real tracking run and
a real video. The DINOv3 forward pass is the one thing faked — an external model call.
"""

import math

import numpy as np
import pytest
import yaml

from detail.core.examples import HEADING_BUCKETS
from detail.pipeline import headings as hd
from detail.pipeline import subject_frames as sf
from detail.storage import example_sets
from detail.storage import zarr_store
from tests.pipeline.test_subject_frames import N_FRAMES, SK, tracked  # noqa: F401


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def _body(e, axis, tail_dx=np.nan, tail_dy=np.nan, reach=1.0, area=400.0, tail_px=60.0):
    return {"e": e, "axis": axis, "area": area, "tail_dx": tail_dx, "tail_dy": tail_dy,
            "tail_px": tail_px, "tail_reach": reach}  # fmt: skip


def _rows(bodies, x=None):
    n = len(bodies)
    return hd.TrackRows(
        frame_idx=np.arange(n),
        x=np.full(n, 320.0) if x is None else np.asarray(x, float),
        y_floor=np.full(n, 400.0),
        height=np.full(n, 20.0),
        area=np.full(n, 500.0),
        top=np.full(n, 0.5),
        body=bodies,
    )


def test_the_head_points_away_from_the_tail_base():
    side_on = [_body(0.8, 0.0, tail_dx=12.0, tail_dy=2.0)] * 12  # tail to the right
    end_on = [
        _body(
            0.1, 90.0, tail_dx=0.0, tail_dy=-8.0, reach=1.0
        ),  # long tail above: behind
        _body(
            0.1, 90.0, tail_dx=0.0, tail_dy=8.0, reach=1.0
        ),  # long tail below: in front
        _body(0.1, 90.0, tail_dx=0.0, tail_dy=8.0, reach=0.3),  # short leftover: paws
    ]
    rows = _rows(side_on + end_on)
    out = hd.tail_headings(rows, np.ones(len(rows.body), bool))
    assert all(out[i] == 180.0 for i in range(12)), "tail on the right: facing left"
    assert out[12] == 90.0, "tail above an end-on body: facing the camera"
    assert out[13] == -90.0, "a long tail below: facing away"
    assert 14 not in out, "paws below a mouse facing the camera say nothing"


def test_a_tail_too_small_or_off_the_floor_says_nothing():
    rows = _rows(
        [_body(0.8, 0.0, tail_dx=12.0, tail_px=1.0), _body(0.8, 0.0, tail_dx=12.0)]
    )
    out = hd.tail_headings(rows, np.array([True, False]))
    assert out == {}


def test_the_tail_vetoes_walking_and_toward_away_walking_needs_it():
    walking = {
        0: (0.0, "lateral"),
        1: (0.0, "lateral"),
        2: (90.0, "radial"),
        3: (90.0, "radial"),
    }
    tails = {0: 0.0, 1: 180.0, 3: 90.0}
    kept, conflicts = hd.fuse(walking, tails)
    assert kept == {0: 0.0, 3: 90.0}
    assert conflicts == {1}, "walking right with the tail on the right: dropped"


def test_steady_sideways_walking_is_its_heading():
    n = 30
    bodies = [_body(0.8, 0.0, area=400.0)] * n
    x = 320.0 + 4.0 * (np.arange(n) - n / 2)  # rightwards through the image centre
    out = hd.walking_headings(_rows(bodies, x=x), None, 30.0, 640, np.ones(n, bool))
    assert out, "no walking found"
    assert all(abs(phi) < 10 and kind == "lateral" for phi, kind in out.values())


def test_the_heading_is_relative_to_the_line_of_sight():
    """Walking rightwards on the right of the image is walking slightly away from the camera
    (negative: toward is +90), by the bearing to the animal."""
    n = 30
    x = 560.0 + 4.0 * np.arange(n)
    out = hd.walking_headings(
        _rows([_body(0.8, 0.0)] * n, x=x), None, 30.0, 640, np.ones(n, bool)
    )
    for i, (phi, _kind) in out.items():
        beta = math.degrees(math.atan((x[i] - 320) / (hd.FOCAL_FRAC * 640)))
        assert phi == pytest.approx(-beta, abs=1.0)


def test_an_end_on_body_cannot_walk_sideways():
    n = 30
    rows = _rows([_body(0.05, 90.0)] * n, x=320.0 + 4.0 * np.arange(n))
    assert hd.walking_headings(rows, None, 30.0, 640, np.ones(n, bool)) == {}


def test_buckets_are_45_degree_bins_from_facing_right():
    assert [hd.bin_of(a) for a in (0, 44, 46, 90, 180, -90, -135)] == [
        0,
        1,
        1,
        2,
        4,
        6,
        5,
    ]
    assert HEADING_BUCKETS[2] == "toward" and HEADING_BUCKETS[6] == "away"


# --------------------------------------------------------------------------- #
# The cache and the strategy, over a real recording
# --------------------------------------------------------------------------- #
class FakeBackbone:
    """``patch_grids`` of a DINOv3 ViT, deterministic per image."""

    weight = "vit_large"

    def patch_grids(self, images):
        import torch

        out = []
        for im in images:
            g = torch.Generator().manual_seed(
                int(np.asarray(im, np.int64).sum()) % 2**31
            )
            grid = torch.randn((1024, 14, 14), generator=g)
            out.append(torch.nn.functional.normalize(grid, dim=0))
        return out


def _compute(tracked):  # noqa: F811
    root = tracked.experiment.dir
    return hd.compute_headings(root, "recording_002", "track_01", None, FakeBackbone())


def test_the_cache_has_the_spec_schema(tracked):  # noqa: F811
    out = _compute(tracked)
    meta = yaml.safe_load((out / hd.META_FILENAME).read_text(encoding="utf-8-sig"))
    assert list(meta)[0] == "schema_version" and meta["schema_version"] == 1
    assert meta["method"] == {"version": hd.METHOD_VERSION, "constants": hd.CONSTANTS}
    assert meta["floor"] is None and meta["parts"] == ["part_001"]
    root = zarr_store.open_group(out / "part_001.zarr", mode="r")
    assert root.attrs["schema_version"] == 1
    expect = {
        "frame_idx": "int32", "t_ns": "int64", "track_id": "int32", "walking_deg": "float32",
        "tail_deg": "float32", "conflict": "bool", "rearing": "bool", "pickable": "bool",
        "bbox": "float32", "feature_row": "int32", "features": "float16",
        "features_mirror": "float16",
    }  # fmt: skip
    assert set(root.array_keys()) == set(expect)
    for name, dtype in expect.items():
        assert str(root[name].dtype) == dtype, name
    assert root["frame_idx"].chunks == (hd.ROW_CHUNK,)
    assert root["bbox"].shape == (N_FRAMES, 4)
    assert root["features"].chunks == (
        hd.FEATURE_CHUNK,
        hd.POOL * hd.POOL * hd.PROJ_DIM,
    )
    embedded = int((root["feature_row"][:] >= 0).sum())
    assert root["features"].shape == (embedded, 1024)
    assert root["features_mirror"].shape == root["features"].shape
    assert hd.cache_matches(tracked.tracked_recording.dir, "track_01", None)
    assert not hd.cache_matches(tracked.tracked_recording.dir, "track_01", {"cx": 1.0})


def test_the_pass_holds_one_batch_of_pictures_at_a_time(
    tracked, monkeypatch
):  # noqa: F811
    """Tenet 11: crops are embedded as they are decoded, in bounded batches — a pass that
    gathered every crop first would hold a whole video's frames on a long recording."""
    monkeypatch.setattr(hd, "EMBED_BATCH", 1)  # batches of EMBED_BATCH * 4 = 4 crops
    monkeypatch.setattr(hd, "EMBED_EVERY", 1)  # all 5 frames: more crops than one batch
    made, embedded, waiting, sizes = [0], [0], [], []
    real_crop = sf.subject_crop

    def counting_crop(rgb, bbox):
        made[0] += 1
        return real_crop(rgb, bbox)

    class Counting(FakeBackbone):
        def patch_grids(self, images):
            sizes.append(len(images))
            if len(sizes) % 2:  # a batch's crops (its mirror is the second call)
                waiting.append(made[0] - embedded[0])
                embedded[0] += len(images)
            return super().patch_grids(images)

    monkeypatch.setattr(sf, "subject_crop", counting_crop)
    hd.compute_headings(
        tracked.experiment.dir, "recording_002", "track_01", None, Counting()
    )
    assert sizes and max(sizes) <= 4
    assert (
        max(waiting) <= 4
    ), f"{max(waiting)} crops waited for one batch: all were gathered first"
    root = zarr_store.open_group(
        hd.headings_dir(tracked.tracked_recording.dir, "track_01") / "part_001.zarr",
        mode="r",
    )
    assert sum(sizes) == 2 * root["features"].shape[0], "each crop and its mirror, once"


def test_a_cached_recording_loads_and_pools(tracked):  # noqa: F811
    _compute(tracked)
    cache = hd.load_headings(tracked.experiment.dir, "recording_002", "track_01")
    assert len(cache.frame_idx) == N_FRAMES
    assert cache.key(0) == ("recording_002", "part_001", int(cache.t_ns[0]), 0)
    pool = hd.heading_pool([cache])
    assert len(pool.row) == int(((cache.feature_row >= 0) & cache.pickable).sum())
    # one recording: no other recording to borrow similar frames from, so nothing is guessed
    assert set(pool.source.tolist()) <= {-1, 0, 1, 2}


def test_the_heading_strategy_samples_per_bucket_and_skips_rejections(
    tracked, monkeypatch
):  # noqa: F811
    monkeypatch.setattr(hd, "_load_backbone", lambda device: FakeBackbone())
    monkeypatch.setattr(hd, "_release", lambda model: None)
    root = tracked.experiment.dir
    created = example_sets.create_set(root, "rats", SK)
    first = sf.sample_frames(root, created.id, tracked.sources, 1, strategy="heading")
    assert first and all(f.sampled_by.startswith("heading:") for f in first)
    bucket = first[0].sampled_by.split(":")[1]
    assert bucket in HEADING_BUCKETS
    created.reject(first[0].key(), bucket)
    example_sets.write_set(root, created)

    again = sf.sample_frames(
        root,
        created.id,
        tracked.sources,
        N_FRAMES,
        strategy="heading",
        buckets=[bucket],
    )
    assert first[0].key() not in {f.key() for f in again}, "rejected for this bucket"


def test_the_heading_options_are_refused_where_they_do_not_apply(tracked):  # noqa: F811
    root = tracked.experiment.dir
    created = example_sets.create_set(root, "rats", SK)
    with pytest.raises(ValueError, match="heading strategy's"):
        sf.sample_frames(
            root, created.id, tracked.sources, 1, strategy="spread", buckets=["toward"]
        )
    with pytest.raises(ValueError, match="unknown heading buckets"):
        sf.sample_frames(
            root, created.id, tracked.sources, 1, strategy="heading", buckets=["up"]
        )
    with pytest.raises(ValueError, match="no chosen recording has a 'plate' detection"):
        hd.resolve_floors(root, tracked.sources, "plate", None)


def test_coverage_counts_the_sets_frames_per_bucket():
    from detail.core.examples import ExampleFrame, ExampleSet

    def frame(sampled_by, t):
        return ExampleFrame("frames/x.png", "r", "track_01", 0, "part_001", t, t, (0, 0, 1, 1),
                            sampled_by=sampled_by)  # fmt: skip

    s = ExampleSet("ex_01", "x", "example", "", SK, frames=[
        frame("heading:toward:tail", 1), frame("heading:toward:walking", 2),
        frame(None, 3), frame("varied:shape", 4)])  # fmt: skip
    got = hd.coverage(None, s)
    assert got["toward"] == {"in_set": 2, "marked": 0}
    assert got["hand-picked"]["in_set"] == 1 and got["other"]["in_set"] == 1
    assert sum(v["in_set"] for v in got.values()) == 4
    assert not math.isnan(0.0)
