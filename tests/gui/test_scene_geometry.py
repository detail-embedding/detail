"""Guards for the one spatial workspace (§10).

Four page sections, five dialogs and four editing idioms became one dialog over one canvas.
The classes recorded here are the ones that merge could re-break, and each is invisible to a
test that only checks the dialog builds:

- **A capability with no way in, that also deletes what it cannot express.** The multi-region
  manager was the only route to a recording's ROIs, and it read `active` nowhere and wrote it
  nowhere — so opening it and pressing *Save all* turned ``frames 100–900`` into ``whole
  video``, on every region in the recording, including ones the user never touched. The
  single-definition editor that *could* edit an active window (`RoiDialog`) had no call site
  in the app at all. Nothing failed either way: the dialog rendered, the save succeeded, and
  the region went on applying to footage the user had excluded.

- **Not touching something is not a write.** A calibration set from the CLI's
  ``--pixels-per-mm`` has no points, so a canvas cannot show it. A combined save that
  validates it like a fresh pick refuses ("needs exactly two points") and blocks a save about
  regions; one that reads its empty canvas as a deletion drops it silently. Both are wrong,
  and both are reachable only through the *combined* save — neither existed while each kind
  had its own dialog.

- **A shared canvas has to be told what it is drawing.** `max_points` was a constructor
  argument when one dialog meant one kind. With four kinds on one canvas, a stale max lets a
  click put a third vertex on a two-point calibration.

- **A control that reports pending edits is told by every edit, and only by edits** (#35
  finding 10). Save greys out when nothing is pending, and the docs call that the sign your
  edits are in. Rename, the known distance, the whole-video box and the active-window bounds
  wrote into a definition and never told Save, so it stayed grey; two tests hid it by calling
  `_sync()` by hand after the edit. The other way round, a scrub wrote the canvas back at its
  6-decimal precision over a vertex stored with more, and lit Save with nothing changed. Every
  handler that writes a definition ends in `_edited()` (or `_sync()`), checked over the class's
  source with negative cases; each control is driven through its own signal; and "changed" is
  compared at the canvas's precision.

- **An edit lands on the frame being shown** (#35 finding 11). A new field of view or
  calibration got its first keyframe at frame 0 whatever frame was showing, so the canvas was
  read-only there, and **Set keyframe here** then made two keyframes, which locks the point
  count. A new definition now starts on the frame showing, and the calibration (stored with no
  frame) is edited on any frame. Clicked with the mouse on frame 500 here.

- **A commit of several parts checks everything its writes refuse before the first write** (#35
  finding 13). The pre-check missed a name used twice, so the field of view was stored and then
  `set_rois` refused. It now asks the storage layer's own validators
  (`Recording.check_spatial_defs`, `check_distance_calibration`) for every kind; asserted on
  what is stored, not on the message.
"""

import ast
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402

from PyQt6.QtWidgets import QDialog  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.arena import SPATIAL_KINDS, _BLANK_FRAME as _BLANK  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402


def _square(dx=0.0):
    return [(0.2 + dx, 0.2), (0.8 + dx, 0.2), (0.8 + dx, 0.8), (0.2 + dx, 0.8)]


def _kf(dx=0.0):
    return [(0, _square(dx))]


def _recording(tmp_path, name="s1", *, resolution=(1920, 1080)):
    exp = Experiment.create("p", tmp_path)
    rec = exp.create_recording(name)
    data = recording_store.read_recording(rec.dir)
    data["video"] = {"resolution": list(resolution), "files": [], "storage": "default"}
    recording_store.write_recording(rec.dir, data)
    return exp, rec


def _workspace(rec):
    from detail.gui.arena import SceneGeometryDialog

    return SceneGeometryDialog(rec)


def _row_of(dialog, index):
    """The list row showing `_regions[index]` — by the id the item carries, never by
    position, since the per-kind headings already offset it."""
    for row in range(dialog._list.count()):
        if dialog._list.item(row).data(Qt.ItemDataRole.UserRole) == index:
            return row
    raise AssertionError(f"no row for region {index}")


def _index_of(dialog, name):
    return next(i for i, r in enumerate(dialog._regions) if r["name"] == name)


def _rename(dialog, name):
    """Type a name the way a person does: the field's own signal, never a hand `_sync()`.
    Calling `_sync()` by hand is what hid #35 finding 10 — the handler never did."""
    dialog._name.setText(name)
    dialog._name.editingFinished.emit()


# --- the active window survives the trip through the editor ------------------ #
def test_an_active_window_survives_a_load_and_save(qapp, tmp_path):
    """The data-loss fix, asserted as the interval rather than as "save returned".

    Reverting it (dropping `active` from the load or the save) leaves every one of these
    regions well-formed and the dialog perfectly functional — the window just silently
    becomes "whole video", which is a wider claim than the user made and produces analysis
    rows they never asked for.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    rec.define_roi("post-injection", _kf(0.05), active=(100, 900))

    dialog = _workspace(rec)
    dialog._save_all()

    windows = {r["name"]: r.get("active") for r in rec.rois}
    assert windows["post-injection"] == {"start": 100, "end": 900}
    assert windows["nest"] is None


def test_an_active_window_can_be_set_and_cleared(qapp, tmp_path):
    """And it is *editable*, which closes a surface-parity gap: `define_roi(active=...)`
    is reachable from the API and the CLI and had no GUI at all."""
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())

    dialog = _workspace(rec)
    dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, "nest")))
    assert dialog._whole_video.isChecked()

    dialog._whole_video.setChecked(False)
    dialog._on_scrub(120)
    dialog._set_active_bound(start=True)
    dialog._on_scrub(400)
    dialog._set_active_bound(start=False)
    dialog._save_all()
    assert rec.rois[0]["active"] == {"start": 120, "end": 400}

    dialog = _workspace(rec)
    dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, "nest")))
    dialog._whole_video.setChecked(True)
    dialog._save_all()
    assert rec.rois[0]["active"] is None


# --- untouched kinds are not rewritten --------------------------------------- #
def test_a_scalar_calibration_survives_a_save_about_something_else(
    qapp, tmp_path, monkeypatch
):
    """A calibration with no points must neither block the save nor be deleted by it.

    `set-calibration --pixels-per-mm` stores a scalar and no points, so there is nothing for
    a canvas to draw. Validating it like a fresh two-point pick refuses a save that was
    about a region; treating its empty canvas as a deletion drops a calibration nobody
    touched. Untouched means untouched.
    """
    from PyQt6.QtWidgets import QMessageBox

    _exp, rec = _recording(tmp_path)
    rec.set_pixels_per_mm(4.0)
    rec.define_roi("nest", _kf())
    assert rec.distance_calibration["points"] is None

    dialog = _workspace(rec)
    refused = []
    dialog._refuse = lambda title, message: refused.append(message)
    errors = []
    with monkeypatch.context() as patch:
        patch.setattr(
            QMessageBox,
            "critical",
            staticmethod(lambda *a, **k: errors.append(a[-1])),
        )
        dialog._save_all()

    # The save has to *succeed*, not merely leave things alone. Asserting only that the
    # calibration is intact passes against the bug too — a save that blows up on the
    # untouched calibration also changes nothing, while showing the user an error about a
    # value they never touched and dropping the edit they came for.
    assert not refused and not errors, f"save refused: {refused or errors}"
    assert not dialog._save.isEnabled(), "a clean save leaves nothing pending"
    assert rec.distance_calibration is not None
    assert rec.pixels_per_mm == 4.0
    assert [r["name"] for r in rec.rois] == ["nest"]


def test_a_locked_field_of_view_does_not_block_saving_a_region(qapp, tmp_path):
    """Editing arenas means clearing and redefining them, which the API refuses while a
    tracking run holds one. A save about a *region* must not reach that refusal — the arena
    write is skipped entirely when the arenas are untouched."""
    _exp, rec = _recording(tmp_path)
    arena_id = rec.define_arena("in use", _kf())
    data = recording_store.read_recording(rec.dir)
    data["tracking_runs"] = [{"id": "trk_003", "arena_ids": [arena_id]}]
    recording_store.write_recording(rec.dir, data)
    rec.define_roi("nest", _kf())

    dialog = _workspace(rec)
    index = _index_of(dialog, "nest")
    dialog._list.setCurrentRow(_row_of(dialog, index))
    dialog._regions[index]["name"] = "renamed"
    dialog._save_all()

    assert [r["name"] for r in rec.rois] == ["renamed"]
    assert [a["name"] for a in rec.arenas] == ["in use"]  # untouched, and not refused


def test_a_locked_field_of_view_cannot_be_edited(qapp, tmp_path):
    _exp, rec = _recording(tmp_path)
    arena_id = rec.define_arena("in use", _kf())
    data = recording_store.read_recording(rec.dir)
    data["tracking_runs"] = [{"id": "trk_003", "arena_ids": [arena_id]}]
    recording_store.write_recording(rec.dir, data)

    dialog = _workspace(rec)
    index = _index_of(dialog, "in use")
    assert "trk_003" in dialog._regions[index]["locked"]
    dialog._list.setCurrentRow(_row_of(dialog, index))
    assert not dialog._name.isEnabled()
    assert not dialog._delete.isEnabled()
    assert dialog._delete.toolTip()  # says why, rather than reading as broken
    for button in (dialog._set_kf, dialog._del_kf, dialog._undo, dialog._clear):
        assert not button.isEnabled()


# --- one canvas, four kinds --------------------------------------------------- #
def test_selecting_a_kind_tells_the_canvas_how_many_points_it_takes(qapp, tmp_path):
    """`max_points` was a constructor argument when a dialog meant one kind.

    On a shared canvas a stale max is a click adding a third vertex to a calibration, or a
    point of interest growing a second. Asserted as the pair — a polygon kind then a point
    kind — because either alone passes with the value hardcoded the other way.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    rec.define_point("lever", [(0, [(0.5, 0.5)])])
    rec.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)

    dialog = _workspace(rec)
    for name, expected in (("nest", None), ("lever", 1), ("Distance calibration", 2)):
        dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, name)))
        assert dialog._canvas._max_points == expected, name


def test_nothing_selected_draws_every_definition(qapp, tmp_path):
    """Select nothing and the workspace *is* the old read-only preview.

    That is why the separate preview dialog could be deleted rather than kept in step: it
    existed only because the four editors could not show each other.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_arena("rig", _kf())
    rec.define_roi("nest", _kf(0.02))
    rec.define_point("lever", [(0, [(0.5, 0.5)])])
    rec.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)

    dialog = _workspace(rec)
    assert dialog._selected is None
    shapes = dialog._context_shapes()
    assert len(shapes) == 4, "one per kind, all drawn together"
    assert sum(1 for _pts, _c, is_point, _v in shapes if is_point) == 1  # the POI
    colors = {c.name() for _pts, c, _p, _v in shapes}
    assert len(colors) == 4, f"two kinds share a colour: {colors}"


def test_hiding_one_definition_removes_it_from_the_frame(qapp, tmp_path):
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    rec.define_roi("feeder", _kf(0.05))

    dialog = _workspace(rec)
    assert len(dialog._context_shapes()) == 2
    row = _row_of(dialog, _index_of(dialog, "feeder"))
    dialog._list.item(row).setCheckState(Qt.CheckState.Unchecked)
    assert len(dialog._context_shapes()) == 1


def test_a_kinds_first_definition_keeps_that_kinds_colour(qapp, tmp_path):
    """A kind's colour is the same claim on the page's swatch and on the frame, so it may
    not move because a region was added above it — which is what colouring by position in
    the flat list did."""
    _exp, rec = _recording(tmp_path)
    rec.define_arena("rig", _kf())
    before = _workspace(rec)._region_color(0).name()

    rec.define_roi("nest", _kf())
    rec.define_point("lever", [(0, [(0.5, 0.5)])])
    after = _workspace(rec)
    assert after._region_color(_index_of(after, "rig")).name() == before


def test_a_second_definition_never_borrows_another_kinds_colour(qapp, tmp_path):
    """Found by looking at the render: the second region drew in the calibration's blue, so
    a zone and a measurement were the same colour on one frame."""
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    rec.define_roi("feeder", _kf(0.05))
    rec.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)

    dialog = _workspace(rec)
    colors = [c.name() for _pts, c, _p, _v in dialog._context_shapes()]
    assert len(set(colors)) == len(colors), f"a colour is reused: {colors}"


# --- the whole set is one transaction ---------------------------------------- #
def test_an_invalid_definition_refuses_before_anything_is_written(qapp, tmp_path):
    """Validate every kind, then commit — a partial save leaves the recording in a state
    neither the user nor the dialog describes."""
    from PyQt6.QtWidgets import QMessageBox

    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    rec.define_point("lever", [(0, [(0.5, 0.5)])])

    dialog = _workspace(rec)
    warned = []
    dialog._refuse = lambda title, message: warned.append(message)
    # a region with too few vertices to be a polygon
    dialog._regions[_index_of(dialog, "nest")]["keyframes"] = {0: [(0.1, 0.1)]}
    dialog._save_all()

    assert warned and "needs at least 3 points" in warned[0]
    assert [r["name"] for r in rec.rois] == ["nest"]  # unchanged on disk
    assert [p["name"] for p in rec.points] == ["lever"]  # and so is the kind after it
    assert QMessageBox  # imported for the real path; the stub above keeps it silent


# --- colour says which one you are editing ----------------------------------- #
def test_selecting_one_greys_the_others_and_colours_it(qapp, tmp_path):
    """Colour means "this is the one you are editing", grey means context.

    Two halves that only make sense together, and each passes alone with the other
    hardcoded: the canvas drew *everything* editable in a fixed red, so the swatch beside a
    definition and the outline on the frame disagreed until you saved; and the siblings
    stayed at full colour, so nothing on the frame said which one your drags would move.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_arena("rig", _kf())
    rec.define_roi("nest", _kf(0.02))

    dialog = _workspace(rec)
    # nothing selected: the read-only overview, every kind in its own colour
    assert dialog._canvas._dim_context is False

    index = _index_of(dialog, "nest")
    dialog._list.setCurrentRow(_row_of(dialog, index))
    assert dialog._canvas._dim_context is True
    assert dialog._canvas._edit_color.name() == dialog._region_color(index).name()

    dialog._list.setCurrentRow(-1)
    assert dialog._canvas._dim_context is False


def test_context_does_not_draw_a_region_outside_its_active_window(qapp, tmp_path):
    """The overview has to agree with what analysis will do.

    `roi_at_frame` returns nothing outside the window, and the preview this dialog absorbed
    went through it. Interpolating keyframes alone drew the region at every frame — a
    picture that contradicts the measurement, on the one property no picture can otherwise
    show. The *selected* definition still draws everywhere, or setting a window would hide
    the thing you are editing.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_roi("timed", _kf(), active=(3000, 5000))

    dialog = _workspace(rec)
    assert dialog._selected is None
    assert rec.roi_at_frame(rec.rois[0]["id"], 0) is None
    assert dialog._context_shapes() == [], "drawn at a frame it does not apply to"

    dialog._on_scrub(4000)
    assert len(dialog._context_shapes()) == 1

    dialog._on_scrub(0)
    dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, "timed")))
    assert dialog._ghost_for(dialog._regions[0], 0), "the edited one stays visible"


def test_a_recording_with_no_video_says_so_and_takes_no_clicks(qapp, tmp_path):
    """`_locate` indexes an empty part list, the scrubber swallows the error, and the
    canvas keeps its blank placeholder — still clickable. You could draw a polygon against
    a black rectangle and save it (tests/gui/test_frame_source.py's recorded class)."""
    exp = Experiment.create("p", tmp_path)
    rec = exp.create_recording("s1")  # no video part
    rec.define_roi("zone", _kf())

    dialog = _workspace(rec)
    assert dialog._canvas.is_unavailable()
    assert not dialog._scrubber.isEnabled()


def test_one_add_button_per_kind_and_no_menu(qapp, tmp_path):
    """A `QToolButton` stretched to the panel width put its small label in the middle and
    Qt's menu indicator at the far right edge — it read as a dropdown, not a button. With
    four kinds there is nothing to collapse into a menu, and naming them costs a click
    less."""
    _exp, rec = _recording(tmp_path)
    dialog = _workspace(rec)
    assert set(dialog._add_buttons) == {k.id for k in SPATIAL_KINDS}
    for kind in SPATIAL_KINDS:
        button = dialog._add_buttons[kind.id]
        assert button.text() == f"+ {kind.label}"
        assert button.menu() is None, "no menu to render an indicator for"
    before = len(dialog._regions)
    dialog._add_buttons["roi"].click()
    assert len(dialog._regions) == before + 1
    assert dialog._regions[-1]["kind"] == "roi"


def test_save_keeps_the_window_open_and_tracks_pending_edits(qapp, tmp_path):
    """Save used to close, which fought the actions beside it: Copy/Import/Export refuse
    while edits are pending, so saving to unblock them shut the dialog and you had to
    reopen it. Save greying out is also the dialog's only sign that edits are pending.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())

    dialog = _workspace(rec)
    # Shown, because "did it close" is a visibility question and a dialog nobody showed
    # reports hidden either way — which is how this test first passed against the bug.
    dialog.show()
    qapp.processEvents()
    assert not dialog._save.isEnabled()  # nothing pending on open
    assert dialog._save.toolTip() == "No changes to save"

    index = _index_of(dialog, "nest")
    dialog._list.setCurrentRow(_row_of(dialog, index))
    _rename(dialog, "renamed")
    assert dialog._save.isEnabled()

    dialog._save_all()
    qapp.processEvents()
    assert dialog.isVisible(), "Save must not close the window"
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert not dialog._save.isEnabled()  # saved, so nothing pending again
    assert [r["name"] for r in rec.rois] == ["renamed"]


def test_closing_with_pending_edits_asks_first(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    dialog = _workspace(rec)
    dialog.show()
    qapp.processEvents()
    dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, "nest")))
    _rename(dialog, "renamed")

    asked = []
    with monkeypatch.context() as patch:
        patch.setattr(
            QMessageBox,
            "question",
            staticmethod(
                lambda *a, **k: (asked.append(a[-1]), QMessageBox.StandardButton.No)[1]
            ),
        )
        # Click the button, not `_close()` — the wiring is what this guards. Calling the
        # method directly passes even with the button connected straight to `reject`,
        # which is exactly the revert it is supposed to catch.
        dialog._close_button.click()
    assert asked, "closing over unsaved edits discarded them silently"
    assert "Discard" in asked[0]
    qapp.processEvents()
    assert dialog.isVisible(), "answering No must leave the dialog open"
    assert [r["name"] for r in rec.rois] == ["nest"]  # nothing written


def test_dragging_a_vertex_lights_up_save(qapp, tmp_path):
    """An edit has to reach Save, and the canvas is the only thing that knows one happened.

    `_capture_current` writes the canvas back into the region model, and it used to run
    only when something *else* made the dialog re-read — a scrub, or picking another
    definition. So moving a vertex left Save greyed out and scrubbing a frame lit it up,
    which reads as the frame having been the edit. The canvas now says when it changed.
    """
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())

    dialog = _workspace(rec)
    dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, "nest")))
    assert not dialog._save.isEnabled()

    points = list(dialog._canvas.normalized_polygon())
    points[0] = (points[0][0] + 0.05, points[0][1])
    dialog._canvas.set_polygon(points, editable=True)
    dialog._canvas.edited.emit()

    assert dialog._save.isEnabled(), "a canvas edit never reached Save"
    dialog._save_all()
    assert not dialog._save.isEnabled()
    assert rec.roi_keyframes(rec.rois[0]["id"])[0][1][0][0] == pytest.approx(
        points[0][0], abs=1e-6
    )


def test_the_canvas_reports_every_kind_of_edit(qapp, tmp_path):
    """Each mutating path emits, not just the one the fix was written against."""
    from detail.gui.arena import ArenaCanvas

    canvas = ArenaCanvas(_BLANK, points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)])
    seen = []
    canvas.edited.connect(lambda: seen.append(1))

    canvas.undo()
    assert len(seen) == 1
    canvas.clear()
    assert len(seen) == 2


# --- every edit reaches Save (#35 finding 10) -------------------------------- #
# Save greys out when nothing is pending, and the docs call that the sign your edits are
# in. Four handlers wrote into a definition and never told Save — rename, the known
# distance, the whole-video box and the active-window bounds — and two tests hid it by
# calling `_sync()` by hand after the edit. Each case below drives the control's own
# signal on a freshly opened dialog, so the only thing that can light Save is the handler.
def _everything(tmp_path):
    _exp, rec = _recording(tmp_path)
    rec.define_roi("nest", _kf())
    rec.define_roi("feeder", [(0, _square()), (300, _square(0.05))])
    rec.define_roi("late", _kf(0.1), active=(0, 900))
    rec.define_point("lever", [(0, [(0.5, 0.5)])])
    rec.set_distance_calibration([(0.1, 0.1), (0.5, 0.1)], 10.0)
    return rec


def _pick(dialog, name):
    dialog._list.setCurrentRow(_row_of(dialog, _index_of(dialog, name)))


def _scrub(dialog, frame):
    """Move the scrubber, whose `frame_changed` is what reaches the dialog."""
    dialog._scrubber.set_total(1000)
    dialog._scrubber.set_frame(frame, decode=False)


_EDITS = {
    "rename": ("nest", lambda d: _rename(d, "burrow")),
    "known distance": ("Distance calibration", lambda d: d._distance.setValue(25.0)),
    "distance unit": ("Distance calibration", lambda d: d._unit.setCurrentText("cm")),
    "whole-video box": ("nest", lambda d: d._whole_video.setChecked(False)),
    "set active start": (
        "late",
        lambda d: (_scrub(d, 100), d._set_start.click()),
    ),
    "set active end": ("late", lambda d: (_scrub(d, 500), d._set_end.click())),
    "set keyframe here": ("nest", lambda d: (_scrub(d, 200), d._set_kf.click())),
    "delete keyframe": ("feeder", lambda d: (_scrub(d, 300), d._del_kf.click())),
    "undo point": ("nest", lambda d: d._undo.click()),
    "clear points": ("nest", lambda d: d._clear.click()),
    "delete": ("nest", lambda d: d._delete.click()),
    "canvas drag": (
        "nest",
        lambda d: (
            d._canvas.set_polygon(_square(0.05), editable=True),
            d._canvas.edited.emit(),
        ),
    ),
}


@pytest.mark.parametrize("control", sorted(_EDITS))
def test_every_edit_control_lights_save(qapp, tmp_path, control):
    rec = _everything(tmp_path)
    dialog = _workspace(rec)
    name, act = _EDITS[control]
    _pick(dialog, name)
    assert not dialog._save.isEnabled(), "Save was live before any edit"
    act(dialog)
    assert dialog._dirty(), f"{control} did not change anything — the case is vacuous"
    assert dialog._save.isEnabled(), f"{control} changed a definition; Save stayed grey"


@pytest.mark.parametrize("name", ["nest", "feeder", "late", "lever"])
def test_looking_is_not_an_edit(qapp, tmp_path, name):
    """Selecting a definition and scrubbing writes the canvas back into it. Its vertices
    came from the API with more decimals than the canvas keeps (``0.2 + 0.1``), so that
    write-back used to read as an edit and light Save with nothing changed — which is also
    what made the active-window cases above pass with their own fix removed."""
    rec = _everything(tmp_path)
    dialog = _workspace(rec)
    _pick(dialog, name)
    for frame in (100, 300, 0):
        _scrub(dialog, frame)
    assert dialog._dirty() == [] and not dialog._save.isEnabled()


@pytest.mark.parametrize("kind_id", [k.id for k in SPATIAL_KINDS])
def test_every_add_button_lights_save(qapp, tmp_path, kind_id):
    _exp, rec = _recording(tmp_path)
    dialog = _workspace(rec)
    assert not dialog._save.isEnabled()
    dialog._add_buttons[kind_id].click()
    assert dialog._save.isEnabled()


@pytest.mark.parametrize("kind_id", [k.id for k in SPATIAL_KINDS])
def test_the_name_field_shows_the_definition_selected(qapp, tmp_path, kind_id):
    """Adding selects the new definition with its handler suppressed, so the Name field
    stayed blank (or kept the one selected before) although the definition had a name.
    """
    rec = _everything(tmp_path) if kind_id != "calibration" else _recording(tmp_path)[1]
    dialog = _workspace(rec)
    if kind_id != "calibration":
        _pick(dialog, "nest")
    dialog._add_buttons[kind_id].click()
    assert dialog._name.text() == dialog._current()["name"]
    dialog._delete.click()
    assert dialog._name.text() == "", "the Name field kept a deleted definition's name"


# --- the rule, over the class's source --------------------------------------- #
# The cases above cover today's controls. A handler added tomorrow is covered by this: a
# method of `SceneGeometryDialog` that writes into a definition must call `_edited()` or
# `_sync()` (which ends in the same Save refresh). `_capture_current` is the one helper,
# named with its reason.
_DEFINITION_KEYS = {"name", "keyframes", "active", "distance_mm", "visible"}
_HELPERS = {
    "_capture_current": "writes the canvas back; every caller ends in _edited or _sync",
}


def _is_self_attr(node, attr):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _region_names(fn) -> set[str]:
    """Names bound to a definition in ``fn``: from `self._current()`, an index into
    `self._regions`, a loop over it, or a dict handed to `self._regions.append`."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            if isinstance(target, ast.Name) and (
                (isinstance(value, ast.Call) and _is_self_attr(value.func, "_current"))
                or (
                    isinstance(value, ast.Subscript)
                    and _is_self_attr(value.value, "_regions")
                )
            ):
                names.add(target.id)
        elif isinstance(node, ast.For) and _is_self_attr(node.iter, "_regions"):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and _is_self_attr(node.func.value, "_regions")
        ):
            names |= {a.id for a in node.args if isinstance(a, ast.Name)}
    return names


def _writes_a_definition(target, names) -> bool:
    if not isinstance(target, ast.Subscript):
        return False
    node = target
    while isinstance(node, ast.Subscript):
        node = node.value
    return _is_self_attr(node, "_regions") or (
        isinstance(node, ast.Name) and node.id in names
    )


def _unsynced_edits(source: str, cls: str = "SceneGeometryDialog") -> list[str]:
    """Methods of ``cls`` that write into a definition and never tell Save."""
    klass = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.ClassDef) and n.name == cls
    )
    found = []
    for fn in klass.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name in _HELPERS:
            continue
        names = _region_names(fn)
        targets = []
        for node in ast.walk(fn):
            if isinstance(node, (ast.Assign, ast.Delete)):
                targets += node.targets
            elif isinstance(node, ast.AugAssign):
                targets.append(node.target)
        if not any(_writes_a_definition(t, names) for t in targets):
            continue
        calls = {
            node.func.attr
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        }
        if not calls & {"_edited", "_sync"}:
            found.append(fn.name)
    return found


def test_every_handler_that_writes_a_definition_tells_save():
    import detail.gui.arena as arena

    source = Path(arena.__file__).read_text(encoding="utf-8")
    assert _unsynced_edits(source) == []
    for helper in _HELPERS:  # an exemption that excuses nothing is stale
        assert f"def {helper}(" in source, f"{helper} is exempt but no longer exists"


@pytest.mark.parametrize(
    "body",
    [
        'reg = self._current()\n        reg["name"] = "x"',
        'self._regions[0]["active"] = None',
        'for reg in self._regions:\n            reg["visible"] = False',
        'reg = self._regions[i]\n        del reg["keyframes"][3]',
        'new = {"kind": "roi"}\n        new["distance_mm"] = 1.0\n'
        "        self._regions.append(new)",
    ],
)
def test_the_save_scan_would_catch_an_unsynced_handler(body):
    source = f"class SceneGeometryDialog:\n    def _on_x(self, i=0):\n        {body}\n"
    assert _unsynced_edits(source) == ["_on_x"]
    fixed = source + "        self._edited()\n"
    assert _unsynced_edits(fixed) == []


def test_the_save_scan_ignores_a_dict_that_is_not_a_definition():
    source = (
        "class SceneGeometryDialog:\n"
        "    def _save_all(self):\n"
        "        entry = {}\n"
        '        entry["active"] = None\n'
    )
    assert _unsynced_edits(source) == []


# --- a new definition is drawable on the frame showing (#35 finding 11) ------- #
@pytest.mark.parametrize("kind_id, clicks", [("arena", 3), ("calibration", 2)])
def test_a_definition_started_on_a_later_frame_can_be_drawn_there(
    qapp, tmp_path, kind_id, clicks
):
    """A field of view or calibration started on frame 500 took no clicks.

    Its first keyframe went to frame 0, so frame 500 was "interpolated" and the canvas
    read-only; **Set keyframe here** then made two keyframes, which locks the point count,
    so clicks still did nothing. Clicked here with the mouse, on a shown canvas.
    """
    from PyQt6.QtCore import QPoint
    from PyQt6.QtTest import QTest

    _exp, rec = _recording(tmp_path)
    dialog = _workspace(rec)
    dialog.resize(1040, 740)
    dialog.show()
    qapp.processEvents()
    dialog._canvas.set_image(_BLANK)  # a frame on screen, as with real footage
    _scrub(dialog, 500)
    dialog._add_buttons[kind_id].click()
    qapp.processEvents()

    rect = dialog._canvas._image_rect()
    spots = [(0.2, 0.2), (0.8, 0.3), (0.5, 0.8)][:clicks]
    for fx, fy in spots:
        at = QPoint(
            round(rect.x() + fx * rect.width()), round(rect.y() + fy * rect.height())
        )
        QTest.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=at)
    drawn = dialog._canvas.normalized_polygon()
    assert len(drawn) == clicks, f"{clicks} clicks on frame 500 placed {len(drawn)}"
    assert dialog._save.isEnabled()

    if kind_id == "calibration":
        dialog._distance.setValue(40.0)
    dialog._save_all()
    if kind_id == "arena":
        ((frame, points),) = rec.arena_keyframes(rec.arenas[0]["id"])
        assert frame == 500 and len(points) == 3
    else:
        cal = rec.distance_calibration
        assert len(cal["points"]) == 2 and cal["distance_mm"] == 40.0


def test_a_calibration_has_no_keyframes_to_set(qapp, tmp_path):
    """The calibration is stored with no frame, so the editor offers no keyframe on it."""
    _exp, rec = _recording(tmp_path)
    rec.set_distance_calibration([(0.1, 0.1), (0.5, 0.1)], 10.0)
    dialog = _workspace(rec)
    _pick(dialog, "Distance calibration")
    _scrub(dialog, 700)
    assert not dialog._set_kf.isEnabled() and not dialog._del_kf.isEnabled()
    assert "same on every frame" in dialog._set_kf.toolTip()
    assert dialog._undo.isEnabled(), "the calibration is not editable on frame 700"


# --- Save is one transaction (#35 finding 13) --------------------------------- #
def _two_named(rec, kind):
    if kind == "arena":
        rec.define_arena("cage", _kf())
        rec.define_arena("cage 2", _kf(0.05))
    else:
        rec.define_arena("cage", _kf())
    if kind == "roi":
        rec.define_roi("a", _kf())
        rec.define_roi("b", _kf(0.05))
    if kind == "point":
        rec.define_point("a", [(0, [(0.5, 0.5)])])
        rec.define_point("b", [(0, [(0.6, 0.6)])])


@pytest.mark.parametrize("kind", ["arena", "roi", "point"])
def test_a_name_used_twice_is_refused_before_anything_is_written(
    qapp, tmp_path, monkeypatch, kind
):
    """Every refusal a write can raise is checked before the first write.

    The pre-check did not look for a name used twice, so the field of view was written and
    then `set_rois` refused — the comment promised one transaction. Asserted on what is
    stored: the edited field of view must be unchanged on disk.
    """
    from PyQt6.QtWidgets import QMessageBox

    _exp, rec = _recording(tmp_path)
    _two_named(rec, kind)
    stored_fov = rec.arena_keyframes(rec.arenas[0]["id"])
    stored = {k: [e["name"] for e in getattr(rec, f"{k}s")] for k in ("roi", "point")}

    dialog = _workspace(rec)
    told = []
    dialog._refuse = lambda _title, message: told.append(message)
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: told.append(a[2]))
    )
    fov = next(r for r in dialog._regions if r["kind"] == "arena")
    fov["keyframes"] = {0: [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]}
    names = [r for r in dialog._regions if r["kind"] == kind]
    names[1]["name"] = names[0]["name"]
    dialog._save_all()

    assert told and "named" in told[0], told
    assert rec.arena_keyframes(rec.arenas[0]["id"]) == stored_fov
    assert {
        k: [e["name"] for e in getattr(rec, f"{k}s")] for k in ("roi", "point")
    } == stored


def test_a_calibration_the_write_would_refuse_is_refused_first(
    qapp, tmp_path, monkeypatch
):
    """The calibration is written last, so its refusal (two points on one spot) used to
    arrive after the field of view and every region were already stored."""
    from PyQt6.QtWidgets import QMessageBox

    _exp, rec = _recording(tmp_path)
    rec.define_arena("cage", _kf())
    rec.set_distance_calibration([(0.1, 0.1), (0.5, 0.1)], 10.0)
    stored_fov = rec.arena_keyframes(rec.arenas[0]["id"])

    dialog = _workspace(rec)
    told = []
    dialog._refuse = lambda _title, message: told.append(message)
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: told.append(a[2]))
    )
    next(r for r in dialog._regions if r["kind"] == "arena")["keyframes"] = {
        0: [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
    }
    cal = next(r for r in dialog._regions if r["kind"] == "calibration")
    cal["keyframes"] = {0: [(0.4, 0.4), (0.4, 0.4)]}
    dialog._save_all()

    assert told and "distinct" in told[0], told
    assert rec.arena_keyframes(rec.arenas[0]["id"]) == stored_fov
