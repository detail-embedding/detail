"""Regenerating ``kinematics/`` for a run that predates it (§14).

The test that matters here is **backfill == inline**: a run whose sibling was written during the
embed pass and one whose sibling was reconstructed afterwards must be byte-identical. Two paths
that produce "a plausible position" independently will drift, and nothing downstream could tell
which one it was looking at.

The other guard is the part-relative one. The backfill reads one part's detections at a time and
never forms an index across parts, because `det_row` restarts at zero in every part — a
flatten-then-match version resolves one part's rows against another's, silently.
"""

import numpy as np
import pytest

from detail.api.run_preview import load_track_kinematics
from detail.pipeline import kinematics_backfill
from detail.storage.zarr_io import PartZarrReader, kinematics_part_path

from tests.pipeline.test_embedding import (  # noqa: F401 — fixtures/helpers
    _job,
    _registered_embedder,
    _setup,
    _setup_multi_part,
)
from detail.pipeline import embedding


def _run_embedding(recording_dir):
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    with _registered_embedder():
        embedding.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)
    return run_dir


def _read_all(part):
    return {k: np.asarray(v) for k, v in PartZarrReader(part).read().items()}


def test_backfill_reproduces_what_the_embed_pass_wrote(tmp_path):
    """The contract that lets the two paths coexist.

    Written as a comparison rather than as two lists of expected numbers on purpose: the claim is
    that they *agree*, and a test that hardcodes the answer twice can pass while both are wrong.
    """
    recording_dir = _setup(tmp_path, object_ids=[0, 0, -1, 0])
    run_dir = _run_embedding(recording_dir)
    part = run_dir / "track_0" / "part_001.zarr"

    inline = _read_all(kinematics_part_path(part))
    assert (
        len(inline["frame_idx"]) == 3
    ), "fixture wrote nothing; the comparison is vacuous"

    kinematics_backfill.backfill_run(recording_dir, "emb_01", overwrite=True)
    rebuilt = _read_all(kinematics_part_path(part))

    assert set(rebuilt) == set(inline)
    for name in inline:
        np.testing.assert_array_equal(rebuilt[name], inline[name], err_msg=name)


def test_backfill_writes_a_sibling_for_a_run_that_has_none(tmp_path):
    import shutil

    recording_dir = _setup(tmp_path, object_ids=[0, 0, -1, 0])
    run_dir = _run_embedding(recording_dir)
    part = run_dir / "track_0" / "part_001.zarr"
    shutil.rmtree(kinematics_part_path(part))  # a run made before kinematics existed

    assert load_track_kinematics(recording_dir, "emb_01", 0) is None
    assert kinematics_backfill.missing_parts(recording_dir, "emb_01") == [part]

    stages = []
    written = kinematics_backfill.backfill_run(
        recording_dir, "emb_01", progress=lambda *a: stages.append(a)
    )
    assert written == 1
    assert stages and stages[-1][1] == 1.0, "a minutes-long op must report progress"

    got = load_track_kinematics(recording_dir, "emb_01", 0)
    assert got is not None
    assert got["frame_idx"].tolist() == [0, 1, 3]


def test_an_already_backfilled_part_is_skipped(tmp_path):
    """So an interrupted backfill resumes rather than redoing every part."""
    recording_dir = _setup(tmp_path, object_ids=[0, 0, -1, 0])
    _run_embedding(recording_dir)
    assert kinematics_backfill.missing_parts(recording_dir, "emb_01") == []
    assert kinematics_backfill.backfill_run(recording_dir, "emb_01") == 0


def test_each_part_is_backfilled_from_its_own_detections(tmp_path):
    """`det_row` restarts per part, so a flatten-then-match backfill reads the wrong rows.

    Asserted as alignment: each part's kinematics has that part's frames, in that part's order.
    """
    import shutil

    recording_dir = _setup_multi_part(tmp_path, [[0, 0, 0], [0, 0]])
    run_dir = recording_dir / "embeddings" / "emb_01"
    run_dir.mkdir(parents=True)
    with _registered_embedder():
        embedding.run(_job(), run_dir, progress_cb=lambda e: None, stop_event=None)

    for part in sorted((run_dir / "track_0").glob("part_*.zarr")):
        shutil.rmtree(kinematics_part_path(part))
    assert kinematics_backfill.backfill_run(recording_dir, "emb_01") == 2

    for part in sorted((run_dir / "track_0").glob("part_*.zarr")):
        emb = _read_all(part)
        kin = _read_all(kinematics_part_path(part))
        np.testing.assert_array_equal(kin["frame_idx"], emb["frame_idx"])
        np.testing.assert_array_equal(kin["t_ns"], emb["t_ns"])


def test_backfilling_an_unknown_run_says_so(tmp_path):
    recording_dir = _setup(tmp_path, object_ids=[0, 0])
    with pytest.raises(ValueError, match="emb_99"):
        kinematics_backfill.backfill_run(recording_dir, "emb_99")


def test_a_fragmented_frame_reports_the_union_not_the_representative(tmp_path):
    """The finding that put geometry in the embed pass rather than downstream of `det_row`.

    Two detections of one subject on one frame — the detector split the animal — are OR-merged
    into one silhouette, and `det_row` records only the highest-confidence of them. Geometry
    taken from that row describes the larger fragment and calls it the animal.

    The fixture has to give the two rows **different boxes**, or the union and the
    representative are the same number and this passes without testing anything.
    """
    recording_dir = _setup(
        tmp_path,
        object_ids=[0, 0],
        frames=[0, 0],  # both detections on frame 0
        boxes=[(0.10, 0.20, 0.30, 0.40), (0.50, 0.60, 0.80, 0.90)],
    )
    run_dir = _run_embedding(recording_dir)
    part = run_dir / "track_0" / "part_001.zarr"

    inline = _read_all(kinematics_part_path(part))
    assert (
        len(inline["frame_idx"]) == 1
    ), "the two rows must merge into one embedded frame"
    assert inline["bbox_norm"][0].tolist() == pytest.approx([0.10, 0.20, 0.80, 0.90])
    # Neither detection's own box — which is the whole point.
    assert inline["bbox_norm"][0].tolist() != pytest.approx([0.10, 0.20, 0.30, 0.40])

    kinematics_backfill.backfill_run(recording_dir, "emb_01", overwrite=True)
    rebuilt = _read_all(kinematics_part_path(part))
    for name in inline:
        np.testing.assert_array_equal(rebuilt[name], inline[name], err_msg=name)
