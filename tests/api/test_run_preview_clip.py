"""Tests for the clip-query loader + comparability + ClipQueryPreview
(detail/api/run_preview.py)."""

import numpy as np
import pytest

from detail.api import run_preview
from detail.storage import experiment as experiment_store


def _recording(experiment_dir, sid):
    path = experiment_store.recording_path(experiment_dir, sid)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _traj(rng, n, dim=8):
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    frame_idx = np.arange(n, dtype=np.int32)
    t_ns = (np.arange(n) * 33_000_000).astype(np.int64)
    return emb, frame_idx, t_ns


def test_load_track_trajectory_shapes_and_dtypes(experiment_dir, make_embedding_run):
    s = _recording(experiment_dir, "s1")
    emb, frame_idx, t_ns = _traj(np.random.default_rng(0), 50)
    make_embedding_run(s, emb, frame_idx, t_ns)
    e, f, t = run_preview.load_track_trajectory(s, "emb_01", 0)
    assert e.shape == (50, 8) and e.dtype == np.float32
    assert f.shape == (50,) and t.dtype == np.int64
    np.testing.assert_allclose(e, emb, atol=1e-6)
    np.testing.assert_array_equal(f, frame_idx)


def test_load_track_trajectory_missing_track_raises(experiment_dir, make_embedding_run):
    s = _recording(experiment_dir, "s1")
    make_embedding_run(s, *_traj(np.random.default_rng(1), 10))
    with pytest.raises(ValueError, match="no embedded rows"):
        run_preview.load_track_trajectory(s, "emb_01", 5)


@pytest.mark.parametrize("load", ["load_track_trajectory", "load_track_timeline"])
def test_a_missing_track_names_its_recording(experiment_dir, make_embedding_run, load):
    """A failed rebuild over a cohort said only "track 0", leaving the user to guess which
    recording had gone (#24). The name, not the folder id, since that is what the user knows.
    """
    from detail.storage import recording as recording_store

    s = recording_store.create_recording(
        experiment_dir, "recording_001", name="mouse A"
    )
    make_embedding_run(s, *_traj(np.random.default_rng(1), 10))
    with pytest.raises(ValueError) as info:
        getattr(run_preview, load)(s, "emb_01", 5)
    assert str(info.value) == (
        "no embedded rows for track 5 of recording 'mouse A' in embedding run 'emb_01'"
    )


def test_validate_comparability_ok(experiment_dir, make_embedding_run):
    s1, s2 = _recording(experiment_dir, "s1"), _recording(experiment_dir, "s2")
    make_embedding_run(s1, *_traj(np.random.default_rng(2), 20))
    make_embedding_run(s2, *_traj(np.random.default_rng(3), 20))
    common = run_preview.validate_comparability([(s1, "emb_01"), (s2, "emb_01")])
    assert common["model_id"] == "dinov3_vit"


def test_validate_comparability_mismatched_model_raises(
    experiment_dir, make_embedding_run
):
    s1, s2 = _recording(experiment_dir, "s1"), _recording(experiment_dir, "s2")
    make_embedding_run(s1, *_traj(np.random.default_rng(4), 20))
    make_embedding_run(s2, *_traj(np.random.default_rng(5), 20), model_variant="large")
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability([(s1, "emb_01"), (s2, "emb_01")])


def test_validate_comparability_incomplete_raises(experiment_dir, make_embedding_run):
    s1 = _recording(experiment_dir, "s1")
    make_embedding_run(s1, *_traj(np.random.default_rng(6), 20), status="running")
    with pytest.raises(ValueError, match="not complete"):
        run_preview.validate_comparability([(s1, "emb_01")])


def test_missing_run_raises(experiment_dir):
    s1 = _recording(experiment_dir, "s1")
    with pytest.raises(ValueError, match="not found"):
        run_preview.validate_comparability([(s1, "emb_99")])


def test_preview_load_databases(experiment_dir, make_embedding_run):
    s1, s2 = _recording(experiment_dir, "s1"), _recording(experiment_dir, "s2")
    make_embedding_run(s1, *_traj(np.random.default_rng(7), 30))
    make_embedding_run(s2, *_traj(np.random.default_rng(8), 40))
    preview = run_preview.ClipQueryPreview(experiment_dir)
    dbs = preview.load_databases([("s1", 0), ("s2", 0)], "emb_01")
    assert set(dbs) == {("s1", 0), ("s2", 0)}
    assert dbs[("s2", 0)][0].shape == (40, 8)


def test_global_to_part():
    assert run_preview._global_to_part([10, 5], 0) == (0, 0)
    assert run_preview._global_to_part([10, 5], 9) == (0, 9)
    assert run_preview._global_to_part([10, 5], 10) == (1, 0)
    assert run_preview._global_to_part([10, 5], 14) == (1, 4)
    with pytest.raises(IndexError):
        run_preview._global_to_part([10, 5], 15)
