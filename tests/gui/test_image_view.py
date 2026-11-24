"""The shared image view: zoom that a user can reach, and undo.

**The classes this guards**, recorded in `CLAUDE.md` and `docs/recorded_classes.md`:

- *"A binding is only real on the platform it can be produced on."* The 3-D scatter shipped pan
  bound to middle-drag and Ctrl+left-drag; a MacBook has no middle button, and macOS turns
  Ctrl+left into a *right* button before Qt sees it. Documented, implemented, unreachable. So the
  pan here is plain left-drag, and that is asserted rather than assumed.
- *"A widget whose size hint is a product of the size it was given."* A `QLabel` takes its hint
  from its pixmap, and this one's pixmap is scaled to the pane — the picture grew for seconds
  after opening a clip. `tests/gui/test_audit_playback.py` holds that assertion on the page's
  hint; what is added here is that **zooming** must not reintroduce it.
- *"A container that scrolls but whose children swallow the wheel."* `PageScroll` routes the
  wheel to the page unless a widget claims `OWNS_WHEEL`. A zoom-on-wheel view is exactly such a
  claim — and a view that is *not* zoomable must not make it, or it becomes a hole the page
  cannot be scrolled through.
- *"A picture scaled in a unit other than the pixels it is shown in."* The pane scaled each
  picture to its logical size, and a display scaled to 125% enlarged it again when painting, so
  every frame and plot here was soft. It scales to ``devicePixelRatioF()`` and tags the pixmap.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

from detail.gui.image_view import ImageView  # noqa: E402
from detail.gui.scroll import OWNS_WHEEL  # noqa: E402


def _shown(view, qapp, size=(400, 300)):
    view.resize(*size)
    view.show()
    qapp.processEvents()
    return view


def _patterned(width=640, height=480):
    """A pixmap whose regions differ, so a pan can be *seen*.

    A bare `QPixmap(w, h)` is uninitialised and uniform, so cropping two different windows out of
    it gives the same image — the pan test passed against a broken pan and failed against a
    working one. The recorded "a fixture that never diverges cannot tell a working freeze from a
    broken one" class, in its simplest possible form.
    """
    from PyQt6.QtGui import QColor, QPainter

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(20, 20, 20))
    painter = QPainter(pixmap)
    for i in range(0, width, 40):
        painter.fillRect(i, 0, 20, height, QColor((i * 7) % 255, 90, 200))
    painter.end()
    return pixmap


def _wheel(view, notches=1, modifiers=Qt.KeyboardModifier.NoModifier):
    """A wheel event over the middle of the view."""
    from PyQt6.QtGui import QWheelEvent

    centre = QPointF(view.width() / 2, view.height() / 2)
    return QWheelEvent(
        centre,
        view.mapToGlobal(centre.toPoint()).toPointF(),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


# --------------------------------------------------------------------------- #
# the wheel, and who gets it
# --------------------------------------------------------------------------- #
def test_a_zoomable_view_claims_the_wheel_and_a_plain_one_does_not(qapp):
    """The pair. Either half alone passes with the rule hardcoded the other way.

    A zoomable view that does not claim `OWNS_WHEEL` never zooms — the page scrolls out from
    under the gesture. A plain one that *does* claim it is a hole the page cannot scroll through,
    which is the recorded `test_scroll.py` class.
    """
    assert bool(ImageView(zoomable=True).property(OWNS_WHEEL)) is True
    assert bool(ImageView(zoomable=False).property(OWNS_WHEEL)) is False


def test_the_wheel_zooms_when_zoomable(qapp):
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    view.wheelEvent(_wheel(view, notches=1))
    assert view.zoom > 1.0


def test_the_wheel_does_nothing_on_a_plain_view(qapp):
    """And it must *ignore* the event, not swallow it — an accepted event never reaches the
    page, so a plain view that ate the wheel would stop the page scrolling over it."""
    view = _shown(ImageView(zoomable=False), qapp)
    view.set_frame(_patterned())
    event = _wheel(view)
    view.wheelEvent(event)
    assert view.zoom == 1.0
    assert not event.isAccepted()


def test_alt_wheel_is_left_for_the_page(qapp):
    """Alt is `PageScroll`'s documented escape hatch and has to survive the opt-in."""
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    event = _wheel(view, modifiers=Qt.KeyboardModifier.AltModifier)
    view.wheelEvent(event)
    assert view.zoom == 1.0
    assert not event.isAccepted()


# --------------------------------------------------------------------------- #
# the way back
# --------------------------------------------------------------------------- #
def test_a_zoomed_view_offers_fit_and_an_unzoomed_one_does_not(qapp):
    """A view that can be moved out of frame needs a way back — and a Fit button over a picture
    that is already fitted is a control that does nothing."""
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    from PyQt6.QtWidgets import QPushButton

    fit = view.findChild(QPushButton)
    assert fit is not None, "a zoomable view ships no Fit control"
    assert fit.isHidden(), "Fit is offered before anything is zoomed"

    view.wheelEvent(_wheel(view, notches=2))
    qapp.processEvents()
    assert not fit.isHidden(), "zoomed in, but no way back is shown"

    fit.click()
    assert view.zoom == 1.0
    assert fit.isHidden()


def test_a_plain_view_has_no_fit_control(qapp):
    from PyQt6.QtWidgets import QPushButton

    assert ImageView(zoomable=False).findChild(QPushButton) is None


# --------------------------------------------------------------------------- #
# pan, on a gesture this laptop can make
# --------------------------------------------------------------------------- #
def test_pan_is_plain_left_drag(qapp):
    """Not middle-drag, not Ctrl+left-drag. Both are unreachable on a MacBook trackpad — the
    recorded `test_scatter_navigation` class, where the capability existed and could not be used.
    """
    from PyQt6.QtGui import QMouseEvent

    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    view.wheelEvent(_wheel(view, notches=3))  # pan only means something once zoomed
    qapp.processEvents()

    def mouse(kind, pos):
        return QMouseEvent(
            kind,
            QPointF(*pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    from PyQt6.QtCore import QEvent

    before = view.pixmap().toImage()
    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, (200, 150)))
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, (260, 150)))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, (260, 150)))
    qapp.processEvents()

    assert (
        view.pixmap().toImage() != before
    ), "a left-drag over a zoomed view did not pan it"


def test_a_drag_is_not_a_click(qapp):
    """`clicked` drives play/pause on three surfaces. A pan that also toggled playback would
    make the picture jump every time somebody moved it."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QMouseEvent

    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    view.wheelEvent(_wheel(view, notches=3))
    clicks = []
    view.clicked.connect(lambda: clicks.append(1))

    def mouse(kind, pos):
        return QMouseEvent(
            kind,
            QPointF(*pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, (200, 150)))
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, (280, 150)))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, (280, 150)))
    assert clicks == [], "a drag emitted clicked, so panning toggles playback"

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, (200, 150)))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, (200, 150)))
    assert clicks == [1], "a plain click stopped reaching the player"


def _release(view, pos):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QMouseEvent

    def mouse(kind):
        return QMouseEvent(
            kind,
            QPointF(*pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease))


def test_a_click_reports_where_on_the_frame_it_landed(qapp):
    """A 640x480 frame fitted into 400x300 fills it exactly, so a widget point maps straight
    across; a click in the margin of a letterboxed frame lands on no picture (§26's labeling
    screen places a point where ``clicked_at`` says)."""
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    where = []
    view.clicked_at.connect(lambda u, v: where.append((u, v)))
    clicks = []
    view.clicked.connect(lambda: clicks.append(1))
    _release(view, (100, 75))
    assert where[-1] == pytest.approx((0.25, 0.25), abs=0.01)
    assert clicks == [1], "clicked_at must ride beside clicked, not replace it"

    tall = _shown(ImageView(zoomable=False), qapp)
    tall.set_frame(_patterned(300, 600))  # fitted to 150x300, centred in 400 wide
    assert tall.image_position(QPointF(200, 150)) == pytest.approx((0.5, 0.5), abs=0.01)
    assert (
        tall.image_position(QPointF(20, 150)) is None
    )  # the margin is not the picture


def test_a_click_maps_through_zoom_and_pan(qapp):
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    centre = QPointF(200, 150)
    view.wheelEvent(_wheel(view, notches=4))  # zoom about the centre keeps it fixed
    assert view.zoom > 2
    assert view.image_position(centre) == pytest.approx((0.5, 0.5), abs=0.01)
    # a point half the pane right of centre is a quarter of the *zoomed* picture away
    right = view.image_position(QPointF(300, 150))
    assert right[0] == pytest.approx(0.5 + 100 / (400 * view.zoom), abs=0.01)


def test_the_picture_is_drawn_at_the_screen_s_pixels_and_maps_clicks_as_before(
    qapp, monkeypatch
):
    """On a display scaled to 125%, a picture scaled to the pane's logical size was enlarged a
    second time when painted, so every frame and plot shown here was soft. A 400×300 pane is
    500×375 of the screen's pixels: the pixmap shown is those, tagged with the ratio. Zoomed and
    panned, the crop drawn and the frame position a click maps to still agree.

    What the pane hands the label is read, not ``label.pixmap()``: a ``QLabel`` gives back its
    picture re-scaled to the screen it is on, and this test's offscreen screen is 100%.
    """
    from PyQt6.QtCore import QEvent, QSize
    from PyQt6.QtGui import QMouseEvent

    view = ImageView(zoomable=True)
    monkeypatch.setattr(view, "devicePixelRatioF", lambda: 1.25)
    handed = []
    real_set = view.setPixmap

    def set_pixmap(pixmap):
        handed.append(pixmap)
        real_set(pixmap)

    monkeypatch.setattr(view, "setPixmap", set_pixmap)
    _shown(view, qapp)
    frame = _patterned(1600, 1200)
    view.set_frame(frame)
    shown = handed[-1]
    assert (shown.width(), shown.height(), shown.devicePixelRatio()) == (500, 375, 1.25)

    view._set_zoom(2.0, about=None)

    def mouse(kind, x):
        return QMouseEvent(
            kind,
            QPointF(x, 150),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, 200))
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove, 240))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, 240))
    qapp.processEvents()

    # Twice the 500×375 fit is 1000×750; centred, then dragged 40 pane pixels (50 of the
    # screen's) right, the crop starts at x = 250 - 50.
    grown = frame.scaled(
        QSize(1000, 750),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    shown = handed[-1]
    assert (shown.width(), shown.height(), shown.devicePixelRatio()) == (500, 375, 1.25)
    assert shown.toImage() == grown.copy(200, 187, 500, 375).toImage()
    assert view.image_position(QPointF(200, 150)) == pytest.approx(
        ((200 + 250) / 1000, (187 + 187.5) / 750), abs=0.005
    )


def test_no_frame_no_position(qapp):
    view = _shown(ImageView(zoomable=True), qapp)
    assert view.image_position(QPointF(10, 10)) is None
    where = []
    view.clicked_at.connect(lambda u, v: where.append((u, v)))
    _release(view, (10, 10))
    assert where == []


# --------------------------------------------------------------------------- #
# zoom must not reach the layout
# --------------------------------------------------------------------------- #
def test_zooming_does_not_change_what_the_page_asks_for(qapp):
    """The growth loop, one level on from `test_audit_playback`.

    That test proved a *frame arriving* leaves the page's hint alone. This proves **zooming**
    does too — the pixmap gets bigger, and the pixmap is what a `QLabel` takes its hint from, so
    this is exactly where the loop would come back.
    """
    view = ImageView(zoomable=True)
    page = QWidget()
    box = QVBoxLayout(page)
    box.addWidget(view, 1)
    box.addWidget(QLabel("the controls row under the video"))
    page.resize(900, 600)
    page.show()
    QApplication.processEvents()
    view.set_frame(_patterned(1920, 1080))
    QApplication.processEvents()
    before = page.sizeHint().height()

    for _ in range(5):
        view.wheelEvent(_wheel(view, notches=1))
        QApplication.processEvents()

    assert view.zoom > 1.0, "the fixture never zoomed, so this asserts nothing"
    assert page.sizeHint().height() == before, (
        f"the page's hint went {before} -> {page.sizeHint().height()} because the view was "
        "zoomed; zoom changes what is drawn, never what is asked for"
    )


def test_zoom_is_bounded(qapp):
    """Fit is the floor: a picture smaller than the pane it already fits is not "zoomed out"."""
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    for _ in range(40):
        view.wheelEvent(_wheel(view, notches=-1))
    assert view.zoom == 1.0
    for _ in range(80):
        view.wheelEvent(_wheel(view, notches=1))
    assert view.zoom <= 16.0


@pytest.mark.parametrize(
    "blank",
    [
        lambda view: view.setText("(video not on this machine)"),
        lambda view: view.clear(),
        lambda view: view.clear_frame("(video not on this machine)"),
    ],
    ids=["setText", "clear", "clear_frame"],
)
def test_a_message_is_not_painted_over_by_the_last_frame(qapp, blank):
    """A pane that cannot show a frame must stop showing the last one it had.

    `QLabel`'s own `setText` / `clear` blank the label and leave this view's held frame alone,
    so the next resize or wheel notch — both of which redraw from the held frame — put the old
    picture back over the message. Hosts that swapped a `QLabel` for this view call both.
    """
    view = _shown(ImageView(zoomable=True), qapp)
    view.set_frame(_patterned())
    blank(view)
    view.resize(420, 320)
    view.wheelEvent(_wheel(view, notches=1))
    qapp.processEvents()
    assert view.pixmap_size() is None
    assert view.pixmap() is None or view.pixmap().isNull()


def test_the_picture_block_will_not_be_squeezed_below_its_controls(qapp):
    """`PictureWithControls` manages its children by hand, so a hint is not a floor.

    Given less height than it asked for, `_reflow` holds the picture at its own minimum and
    positions the controls *below* that — past the end of the widget, where Qt clips them. The
    stream preview shipped exactly that: a 15px band of video and no transport, because a
    sibling took a fixed 320px inside a scrolling dialog body.
    """
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    from detail.gui.image_view import PictureWithControls

    picture = ImageView(zoomable=True)
    picture.setMinimumHeight(260)
    controls = QLabel("the transport row")
    block = PictureWithControls(picture, controls)
    floor = block.minimumSizeHint().height()
    assert floor >= 260 + controls.sizeHint().height(), (
        "the block's floor does not cover its picture's minimum plus its controls, so a host "
        "can squeeze the controls out of the widget"
    )

    host = QWidget()
    box = QVBoxLayout(host)
    box.addWidget(block)
    host.resize(400, 80)  # far less than the block needs
    host.show()
    qapp.processEvents()
    assert (
        controls.geometry().bottom() <= block.height()
    ), "the controls are clipped away"


# --------------------------------------------------------------------------- #
# the next surface, not just today's
# --------------------------------------------------------------------------- #
def test_every_embedder_is_named_by_some_test():
    """Discovered, not listed — `tests/gui/test_spatial_map.py` is the template.

    A shared widget grows capabilities while somebody works on *one* of its surfaces, and the
    others go on rendering a plausible wrong picture. Four hosts embed this view today; the point
    of discovering them is that the fifth is covered the day it is added rather than the day
    someone remembers this file.
    """
    import pathlib

    gui = pathlib.Path(__file__).resolve().parents[2] / "detail" / "gui"
    tests = pathlib.Path(__file__).resolve().parent
    embedders = {
        py.stem
        for py in gui.rglob("*.py")
        if "ImageView(" in py.read_text(encoding="utf-8-sig")
        and py.stem != "image_view"
    }
    covered = " ".join(
        py.read_text(encoding="utf-8-sig") for py in tests.rglob("test_*.py")
    )

    # Guard the guard: a discovery that finds nothing passes vacuously forever, and the whole
    # point is that it keeps finding the *next* one.
    assert len(embedders) >= 3, f"the sweep stopped finding embedders: {embedders}"

    missing = sorted(stem for stem in embedders if stem not in covered)
    assert not missing, (
        f"these embed the shared image view and no test names them: {missing}. A capability "
        f"added here reaches the surface being worked on and no other unless something checks."
    )


@pytest.mark.parametrize("top", [False, True], ids=["centred", "top"])
def test_the_picture_block_centres_unless_told_to_sit_at_the_top(qapp, top):
    """``top=True`` is for a player *beside* a list (Browse): centred in a pane as tall as the
    list's workspace, the video sat hundreds of pixels below the list's first rows. Either
    way the controls stay directly under the picture."""
    from detail.gui.image_view import PictureWithControls

    picture = ImageView(zoomable=True)
    controls = QLabel("the transport row")
    block = PictureWithControls(picture, controls, top=top)
    block.resize(400, 900)  # a 16:9 picture 400 wide is 225 tall: lots of room to spare
    block.show()
    qapp.processEvents()
    try:
        spare = 900 - picture.height() - controls.height()
        assert picture.y() == (0 if top else spare // 2)
        assert controls.y() == picture.geometry().bottom() + 1
    finally:
        block.close()


def test_the_controls_span_the_picture_not_the_widened_window(qapp):
    """A 4:3 frame in a window widened past it draws narrow, held back by the height. Its
    controls used to take the window's whole width and run past both edges of the frame (#628,
    Mark examples widened); now they are as wide as the picture draws, centred under it — and
    never narrower than their own one-line width."""
    from PyQt6.QtGui import QPixmap

    from detail.gui.image_view import PictureWithControls
    from detail.gui.requirements import (
        a_players_controls_span_its_picture_not_the_window,
    )
    from detail.gui.scrubber import FrameScrubber

    picture = ImageView(zoomable=True)
    picture.set_frame(QPixmap(1600, 1200))
    scrubber = FrameScrubber(playable=True)
    block = PictureWithControls(picture, scrubber)
    block.set_aspect(picture.pixmap_size())
    block.resize(2000, 900)
    block.show()
    qapp.processEvents()
    try:
        drawn = round(picture.height() * 4 / 3)
        assert drawn < 1500, "the setup must leave the picture height-limited"
        assert scrubber.width() == max(drawn, scrubber.sizeHint().width())
        assert scrubber.geometry().center().x() in range(999, 1002)  # centred under it
        a_players_controls_span_its_picture_not_the_window(scrubber)
    finally:
        block.close()


def test_the_block_regrows_its_controls_when_a_strip_appears_later(qapp):
    """The player's "tracked" strip appears after the first layout, once a track's breaks are
    known. The block places its children by hand, so it used to keep the transport at its old
    height — the button row clipped (Mark examples at 1100x740, #628 follow-up)."""
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QLabel

    from detail.gui.image_view import PictureWithControls
    from detail.gui.requirements import a_players_controls_get_the_height_they_need
    from detail.gui.scrubber import FrameScrubber

    picture = ImageView(zoomable=True)
    picture.set_frame(QPixmap(160, 90))
    scrubber = FrameScrubber(playable=True)
    strip = QLabel("tracked")
    strip.setMinimumHeight(20)
    strip.hide()
    scrubber.set_underlay(strip)
    block = PictureWithControls(picture, scrubber)
    block.resize(700, 600)
    block.show()
    qapp.processEvents()
    try:
        before = scrubber.height()
        strip.show()  # after layout, with no resize of the block
        for _ in range(5):
            qapp.processEvents()
        assert scrubber.height() > before
        a_players_controls_get_the_height_they_need(scrubber)
    finally:
        block.close()


def test_the_block_reserves_the_height_a_wrapping_control_row_needs(qapp):
    """Nothing here is managed by a ``QLayout``, so the block is the only thing that can give
    the transport enough height — and it used to read ``controls.sizeHint().height()``, which
    is the row's *one-line* height. A row that wrapped drew its second line outside the
    geometry it was handed, where it was clipped away: every control present, wired, invisible.

    Driven with a real ``FrameScrubber``, whose row wraps (`gui/flow.py`), rather than a stand-in
    — the point is that the two widgets agree, and a stand-in can be made to agree with anything.
    """
    from detail.gui.image_view import PictureWithControls
    from detail.gui.scrubber import FrameScrubber

    scrubber = FrameScrubber(playable=True)
    scrubber.set_total(479)
    narrow = 380
    wrapped = scrubber.heightForWidth(narrow)
    assert (
        wrapped > scrubber.sizeHint().height()
    ), f"the row does not wrap at {narrow}px, so this cannot fail"

    block = PictureWithControls(ImageView(zoomable=True), scrubber, top=True)
    block.resize(narrow, 700)
    block.show()
    QApplication.processEvents()
    try:
        assert (
            scrubber.height() >= wrapped
        ), f"the transport got {scrubber.height()}px for {wrapped}px of wrapped row"
        assert (
            scrubber.geometry().bottom() < block.height()
        ), "the transport's last line is outside the block that holds it"
    finally:
        block.close()
