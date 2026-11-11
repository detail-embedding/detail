"""A distance calibration gives one scale or refuses — never a plausible wrong one.

**The class: a documented assumption with no check behind it.** ``pixels_per_mm_at``
said in its docstring that its scalar branch assumed an aspect-preserving resize, and
nothing asserted it. One ``pixels_per_mm`` is isotropic, so across an aspect change it is
wrong in some direction by construction — yet the function returned a number, the copy
paths (``copy_calibration``, the CSV import, the Copy-to-other-recordings dialog)
checked nothing, and the value flowed into the ``_mm`` columns of the location export
with a ``pixels_per_mm`` field in the run record that made it look audited.

**How it hid.** The two ways to enter a calibration — a GUI two-point pick and a typed
scalar — were read by two branches with different physical models: the pick recomputed
from its normalized points (both axis ratios), the scalar rescaled by the width ratio
alone. They agree exactly on an aspect-preserving resize, which is the only case the
earlier guard exercised, and on a *horizontal* pick at any resolution, since a
horizontal ruler only ever sees the width ratio. A vertical pick copied from a landscape
video onto a portrait one read a very different scale one way than the other.

**The rule.** One formula for both entry paths, and a refusal
(``calibration_aspect_refusal``) wherever the video's shape differs from the one the
calibration was measured on — at the copy, and again at the read, because the video can
be replaced after the calibration is set. Parametrised over pick direction: a
horizontal-only fixture passes while the bug is live.
"""

import pytest

from detail.storage import recording as recording_store
from detail.storage import spatial

REF = (1920, 1080)
DIST_MM = 200.0
PICKS = {
    "horizontal": [[0.25, 0.5], [0.75, 0.5]],
    "vertical": [[0.5, 0.25], [0.5, 0.75]],
    "diagonal": [[0.25, 0.25], [0.75, 0.75]],
}
SAME_SHAPE = [(960, 540), (3840, 2160), (1920, 1080), (854, 480)]
OTHER_SHAPE = [(1280, 1024), (1080, 1920), (640, 480)]


def _both_entries(points):
    """The same physical calibration entered as a two-point pick and as a scalar."""
    ppm = spatial._pixel_distance(points, *REF) / DIST_MM
    pick = {
        "points": points,
        "distance_mm": DIST_MM,
        "pixels_per_mm": ppm,
        "ref_resolution": list(REF),
    }
    scalar = {
        "points": None,
        "distance_mm": None,
        "pixels_per_mm": ppm,
        "ref_resolution": list(REF),
    }
    return pick, scalar


@pytest.mark.parametrize("target", SAME_SHAPE)
@pytest.mark.parametrize("direction", sorted(PICKS))
def test_pick_and_scalar_read_the_same_scale_on_a_uniform_resize(direction, target):
    points = PICKS[direction]
    pick, scalar = _both_entries(points)
    expected = spatial._pixel_distance(points, *target) / DIST_MM
    assert spatial.pixels_per_mm_at(pick, *target) == pytest.approx(expected, rel=1e-3)
    assert spatial.pixels_per_mm_at(scalar, *target) == pytest.approx(
        expected, rel=1e-3
    )


@pytest.mark.parametrize("target", OTHER_SHAPE)
@pytest.mark.parametrize("direction", sorted(PICKS))
def test_both_entry_paths_refuse_a_video_of_another_shape(direction, target):
    for block in _both_entries(PICKS[direction]):
        with pytest.raises(ValueError, match="different video shape"):
            spatial.pixels_per_mm_at(block, *target)


def test_refusal_names_both_resolutions_and_the_way_out():
    _pick, scalar = _both_entries(PICKS["vertical"])
    refusal = spatial.calibration_aspect_refusal(scalar, 1280, 1024)
    assert "1920x1080" in refusal and "1280x1024" in refusal
    assert "calibrate this recording on its own video" in refusal
    assert spatial.calibration_aspect_refusal(scalar, 960, 540) is None


@pytest.mark.parametrize("direction", sorted(PICKS))
def test_copy_is_refused_onto_a_video_of_another_shape(
    experiment_dir, recording_dir, direction
):
    spatial.write_calibration_from_points(recording_dir, PICKS[direction], DIST_MM, REF)
    dest = experiment_dir / "recordings" / "dest"
    dest.mkdir(parents=True)
    recording_store.write_recording(dest, {"schema_version": 1, "id": "dest"})

    with pytest.raises(ValueError, match="different video shape"):
        spatial.copy_calibration(recording_dir, dest, (1280, 1024))
    assert spatial.read_calibration(dest) is None

    block = spatial.copy_calibration(recording_dir, dest, (960, 540))
    assert block["ref_resolution"] == list(REF)
