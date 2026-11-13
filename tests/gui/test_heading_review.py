"""The heading sampler's compass and review window (§26 *Headings*), over the fixture experiment
with a real video: candidates are decoded from it, and every add and reject goes through the API
to the set on disk."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPointF  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402

from detail.core.skeleton import Skeleton  # noqa: E402
from detail.gui import heading_review as mod  # noqa: E402
from detail.gui import keypoint_labeling  # noqa: E402
from tests.gui import subjects  # noqa: E402
from tests.gui.conftest import drain_busy_calls, drain_until  # noqa: E402

SK = Skeleton(names=("nose", "tail_base"), edges=(("nose", "tail_base"),))


@pytest.fixture
def proj(tmp_path):
    return subjects.an_experiment_with_completed_runs(tmp_path)


@pytest.fixture
def pool(proj):
    return subjects.a_heading_pool(proj.get_recording("recording_001"))


def _open(qapp, proj, pool, bucket="toward"):
    set_id = proj.create_example_set("rats", SK).id
    dialog = mod.HeadingReviewDialog(proj, set_id, pool, bucket)
    dialog.resize(900, 700)
    dialog.show()
    qapp.processEvents()
    drain_busy_calls(qapp)
    return dialog, set_id


def test_the_requirement_sweep_builds_the_window(qapp, proj):
    """The window is one of the dialogs `test_requirements` walks, built from its providers —
    not exempted, so every GUI rule reaches it."""
    assert mod.HeadingReviewDialog in subjects.buildable_dialogs()
    built = subjects.build_dialog(
        mod.HeadingReviewDialog, proj, proj.get_recording("recording_001")
    )
    assert built is not None and built.bucket == "toward"


def test_the_window_shows_a_directions_candidates_decoded(qapp, proj, pool):
    dialog, _ = _open(qapp, proj, pool)
    assert dialog.bucket == "toward"
    assert len(dialog.candidates()) == 3, "every fixture frame faces the camera"
    assert all(t.picture.pixmap() is not None and not t.picture.pixmap().isNull()
               for t in dialog._candidate_tiles), "a candidate was never decoded"  # fmt: skip
    assert not dialog._add.isEnabled() and "Tick" in dialog._add.toolTip()


def test_a_double_click_opens_the_whole_frame_zoomable(qapp, proj, pool, monkeypatch):
    """The tile is a thumbnail; judging a paw needs the frame (test_frame_panes' ALLOWED)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    from detail.gui import full_frame

    opened = []
    monkeypatch.setattr(
        full_frame.FullFrameDialog, "open", lambda self: opened.append(self)
    )
    dialog, _ = _open(qapp, proj, pool)
    QTest.mouseDClick(dialog._candidate_tiles[0].picture, Qt.MouseButton.LeftButton)
    assert opened, "no full frame opened"
    drain_busy_calls(qapp)
    assert opened[0].windowTitle().startswith("recording_001 f")


def test_ticked_candidates_are_added_with_their_heading(qapp, proj, pool):
    dialog, set_id = _open(qapp, proj, pool)
    changed = []
    dialog.changed.connect(lambda: changed.append(1))
    dialog._candidate_tiles[0].check.setChecked(True)
    assert dialog._add.isEnabled() and dialog._add.text() == "Add 1 selected"
    dialog._add.click()
    drain_busy_calls(qapp)
    stored = proj.example_set(set_id)
    assert len(stored.frames) == 1
    assert stored.frames[0].sampled_by == "heading:toward:walking"
    assert (
        changed and len(dialog.candidates()) == 2
    ), "an added frame is not offered again"


def test_not_this_direction_is_remembered_for_the_direction(qapp, proj, pool):
    dialog, set_id = _open(qapp, proj, pool)
    key = tuple(dialog.candidates()[0]["key"])
    dialog._candidate_tiles[0].buttons["Not this direction"].click()
    drain_busy_calls(qapp)
    stored = proj.example_set(set_id)
    assert stored.is_rejected(key, "toward") and not stored.is_rejected(key, "away")
    assert key not in {tuple(c["key"]) for c in dialog.candidates()}


def test_reject_swaps_in_the_next_best_guess(qapp, proj, pool, monkeypatch):
    dialog, set_id = _open(qapp, proj, pool)
    first = dialog.candidates()[0]
    proj.add_heading_candidates(set_id, [first], "toward")
    dialog.show_bucket("toward")
    drain_busy_calls(qapp)
    proj.set_example_marks(set_id, 0, [{"target": "nose", "x": 0.2, "y": 0.2}])
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.Yes)  # fmt: skip
    dialog._reject(0)
    drain_busy_calls(qapp)
    stored = proj.example_set(set_id)
    assert asked, "a marked frame is rejected only after asking"
    assert stored.is_rejected(tuple(first["key"]), "toward")
    assert len(stored.frames) == 1 and stored.frames[0].key() != tuple(first["key"])


def test_the_compass_maps_a_click_to_a_direction(qapp):
    compass = mod.HeadingCompass()
    compass.resize(330, 220)
    compass.show()
    qapp.processEvents()
    seen = []
    assert compass.cursor().shape().name != "PointingHandCursor"
    compass.clicked.connect(seen.append)
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    for bucket, b in (("right", 0), ("toward", 2), ("left", 4), ("away", 6)):
        where = compass._bin_center(b).toPoint()
        QTest.mouseClick(compass, Qt.MouseButton.LeftButton, pos=where)
        assert seen[-1] == bucket
    assert compass.bucket_at(QPointF(-50, -50)) is None
    assert compass.cursor().shape().name == "PointingHandCursor"


def test_the_labeling_screen_opens_the_window_from_its_compass(
    qapp, proj, pool, monkeypatch
):
    set_id = proj.create_example_set("rats", SK).id
    dialog = keypoint_labeling.KeypointLabelingDialog(proj, set_id)
    dialog.show()
    qapp.processEvents()
    drain_busy_calls(qapp)
    drain_busy_calls(qapp)
    assert drain_until(qapp, lambda: dialog._raw is not None)
    calls = []

    def heading_pool(sources, **options):
        calls.append((sources, options))
        return pool

    monkeypatch.setattr(proj, "heading_pool", heading_pool)
    opened = []
    monkeypatch.setattr(
        mod.HeadingReviewDialog, "open", lambda self: opened.append(self)
    )
    dialog._compass.clicked.emit("toward")
    drain_busy_calls(qapp)
    assert calls and calls[0][1]["floor_class"] is None
    assert opened and opened[0].bucket == "toward"
    dialog._compass.clicked.emit("left")
    drain_busy_calls(qapp)
    assert len(calls) == 1, "the pool is held while the subset does not change"
    dialog._n_recordings.setValue(dialog._n_recordings.value())
    dialog._floor.setCurrentIndex(0)
    assert dialog._compass.coverage()["toward"] == {"in_set": 0, "marked": 0}
