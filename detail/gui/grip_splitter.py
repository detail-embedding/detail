"""Panes side by side: a divider you can see, side panels that fold to a labelled strip, and
both remembered.

**Every horizontal split in the app is one of these** (`requirements.a_split_is_built_as_a_grip_splitter`).
Before, there were eighteen: five `GripSplitter`s and thirteen plain `QSplitter`s, and three
different answers to "how do I get this panel out of the way" — the navigation's ‹ / › button,
the Interpret filters' "◂ Hide filters" / "Filters ▸", and, on six splitters, dragging a pane to
nothing, which left no trace of where it had gone (user, 2026-11-09). One behaviour now:

- **Drag the grip** to resize. A *side* panel (a list, the navigation, filters, a checklist, a
  stats column) dragged below its minimum **snaps shut** to a strip about 16px wide, with the
  panel's name written down it — never to zero. Main content and video players resize but never
  fold: they stop at their minimum.
- **The strip is its own way back.** At its centre, exactly where an open grip's dots sit, it
  shows an expand mark (› for a panel on the left, ‹ on the right) in the middle dot's place, a
  dot above and below it. **Click**
  anywhere on the strip to open the panel at the width it last had, or **pull the strip
  outward**: the panel opens as soon as it leaves the strip and follows the pointer. A pointing
  hand and "Click or drag to show <name>" say so on hover. **Double-click** the grip still
  toggles, open or shut.
- **The keyboard reaches it too.** A foldable panel's grip is in the Tab order; with it focused
  (a ring in the palette's focus colour), **Enter** or **Space** shuts the panel or opens it. A
  strip is otherwise unreachable without a mouse — only the navigation had a shortcut (Ctrl+\\).
  When a panel holding the keyboard focus shuts, the focus moves to its strip rather than staying
  on a widget nobody can see, so the next Enter opens it again.
- **Width and open/shut are remembered** per splitter in ``QSettings`` under ``panes/<key>/``.

Why the name on the strip: `docs/gui_design.md`'s "Sections, not panes" quotes Krita removing
per-pane collapse because *"a collapsed pane is an unlabelled sliver with no affordance."* The
strip is labelled, so it reads as a panel you put away rather than a gap.

Each splitter declares which of its panes fold (:meth:`GripSplitter.set_foldable`), and only the
first or last pane can: the strip is the grip beside it, and a middle pane has two.

The house style is zero stylesheets (``requirements.the_package_writes_no_stylesheets``), so the
handle paints itself: a hairline along the divider and three dots at its centre in the chart
theme's recessive colours — and, shut, the expand mark and the name. The mark is drawn by the
handle, not a button beside the panel: ``requirements.a_side_panel_has_no_hide_button`` forbids
the button and allows the strip (user, 2026-11-09: "keep the icon to expand in the same location
as the handles").
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from detail.gui import plot_theme
from detail.gui.qt import (
    QApplication,
    QByteArray,
    QColor,
    QFontMetrics,
    QPainter,
    QPalette,
    QPen,
    QPointF,
    QRectF,
    QSettings,
    QSize,
    QSplitter,
    QSplitterHandle,
    QWidget,
    Qt,
    Signal,
)

#: Thick enough to aim at; thin enough not to read as a gutter.
HANDLE_PX = 9
#: The shut strip's width at the least: the name is written down it, one line of small text.
STRIP_PX = 16
#: No pane is dragged narrower than this, whatever its own minimum says — a pane with no
#: minimum of its own could otherwise be dragged to a sliver while "not foldable".
PANE_FLOOR_PX = 60
#: The expand mark on a shut strip, pointing the way the panel opens. Text, not an icon file:
#: the app ships no icons (``requirements.a_button_carries_no_icon``) and paints its own grips.
EXPAND_LEFT_PANEL = "›"
EXPAND_RIGHT_PANEL = "‹"
#: The grip's three dots, along the divider from its centre.
_DOT_STEPS = (-6, 0, 6)
#: Between the mark and the dot on either side of it.
_MARK_GAP = 3
#: The mark's size against the app's font: the strip is ~16px wide, and a › at text size is a
#: 5px tick that does not read as something to press.
_MARK_SCALE = 1.6
#: The name stops this far from the strip's top and from the mark.
_NAME_MARGIN = 6
#: Shorter than this, the name is not drawn: a two-letter stub reads as noise, and the tooltip
#: still names the panel.
_NAME_MIN_PX = 24
#: Keys that shut or open a focused grip's panel.
TOGGLE_KEYS = (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
#: Where a splitter's state is kept: ``panes/<key>/sizes``, ``panes/<key>/open``, ``panes/<key>/shut``.
SETTINGS_GROUP = "panes"


def _ints(value) -> list[int]:
    """A comma-joined list of ints as `QSettings` hands it back (a string, or a list in INI)."""
    if value is None:
        return []
    parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for part in parts:
        try:
            out.append(int(part))
        except (TypeError, ValueError):
            return []
    return out


def _legacy_sizes(state) -> list[int]:
    """The pane sizes in a ``QSplitter.saveState`` blob — read directly, because
    ``restoreState`` re-applies the old splitter's collapsible flags along with them."""
    from detail.gui.qt import QDataStream, QIODevice

    if not isinstance(state, QByteArray) or state.isEmpty():
        return []
    stream = QDataStream(state, QIODevice.OpenModeFlag.ReadOnly)
    marker, version = stream.readInt32(), stream.readInt32()
    if marker != 0xFF or version != 1:
        return []
    count = stream.readUInt32()
    if count > 64:
        return []
    sizes = [stream.readInt32() for _ in range(count)]
    return sizes if stream.status() == QDataStream.Status.Ok else []


def _joined(values: Iterable[int]) -> str:
    return ",".join(str(int(v)) for v in values)


class _GripHandle(QSplitterHandle):
    """The divider: drags, folds on a drag past its pane's minimum, toggles on a double-click.

    Shut, it is the panel's strip, and the strip is its own way back: a click anywhere on it
    opens the panel at its last width, and a pull outward opens it under the pointer.
    """

    def __init__(self, orientation: Qt.Orientation, parent: "GripSplitter") -> None:
        super().__init__(orientation, parent)
        #: Where in the handle the press landed, along the split; ``None`` when no button is down.
        self._press: int | None = None
        #: Where in the splitter the press landed, for telling a click from a drag.
        self._press_at = 0
        self._dragging = False
        #: The gesture began on a shut strip: it is opening the panel, not resizing it.
        self._opening = False
        #: When a click last opened the strip — the second click of a double-click is the same
        #: gesture, not a request to shut it again.
        self._clicked_open_at: float | None = None

    def _pick(self, point) -> int:
        horizontal = self.orientation() == Qt.Orientation.Horizontal
        return int(point.x() if horizontal else point.y())

    def _shut_pane(self) -> int | None:
        """The pane this handle is the strip of while it is shut, else ``None``."""
        splitter = self.splitter()
        pane = splitter.governed_pane(splitter.indexOf(self))
        return pane if pane is not None and splitter.is_folded(pane) else None

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        hint = super().sizeHint()
        if self._shut_pane() is None:
            return hint
        strip = self.splitter().strip_width()
        if self.orientation() == Qt.Orientation.Horizontal:
            return QSize(strip, hint.height())
        return QSize(hint.width(), strip)

    # -- the gesture -------------------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._press = self._pick(event.position())
        self._press_at = self._pick(self.mapToParent(event.position().toPoint()))
        self._opening = self._shut_pane() is not None
        # An open grip drags from the first pixel, as it always has; a strip waits out the
        # platform's drag distance so a click on it is a click.
        self._dragging = not self._opening

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._press is None:
            return
        along = self._pick(self.mapToParent(event.position().toPoint()))
        if not self._dragging:
            if abs(along - self._press_at) < QApplication.startDragDistance():
                return
            self._dragging = True
        splitter = self.splitter()
        # The strip is wider than the grip it opens into: centre the grip under the pointer
        # rather than carry an offset into the strip that the open grip does not have.
        offset = splitter.handleWidth() // 2 if self._opening else self._press
        splitter.drag_handle(
            splitter.indexOf(self), along - offset, opening=self._opening
        )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        clicked = self._press is not None and not self._dragging
        self._press = None
        self._dragging = False
        if not clicked or event.button() != Qt.MouseButton.LeftButton:
            return
        pane = self._shut_pane()
        if pane is not None:
            self._clicked_open_at = time.monotonic()
            self.splitter().set_folded(pane, False)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        splitter = self.splitter()
        pane = splitter.governed_pane(splitter.indexOf(self))
        if pane is None or event.key() not in TOGGLE_KEYS:
            super().keyPressEvent(event)
            return
        splitter.toggle(pane)
        event.accept()

    def focusInEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().focusOutEvent(event)
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._press = None
        if event.button() != Qt.MouseButton.LeftButton:
            return
        opened_at, self._clicked_open_at = self._clicked_open_at, None
        if (
            opened_at is not None
            and (time.monotonic() - opened_at) * 1000
            <= QApplication.doubleClickInterval()
        ):
            return
        splitter = self.splitter()
        pane = splitter.governed_pane(splitter.indexOf(self))
        if pane is not None:
            splitter.toggle(pane)

    # -- the picture -------------------------------------------------------------------- #
    def mark(self) -> str:
        """The expand mark a shut strip shows — › opens a panel on the left, ‹ one on the right."""
        pane = self._shut_pane()
        if pane is None or self.orientation() != Qt.Orientation.Horizontal:
            return ""
        return EXPAND_LEFT_PANEL if pane == 0 else EXPAND_RIGHT_PANEL

    def _mark_font(self):
        font = self.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * _MARK_SCALE)
        return font

    def _mark_origin(self) -> tuple[QPointF, QRectF]:
        """The baseline point the mark is drawn from, and the glyph's own ink rect there —
        centred on the strip by the glyph's ink, not the font's line box, whose descent would
        put the visible mark above the centre."""
        ink = QFontMetrics(self._mark_font()).tightBoundingRect(self.mark())
        origin = QPointF(
            self.width() / 2 - (ink.left() + ink.width() / 2),
            self.height() / 2 - (ink.top() + ink.height() / 2),
        )
        return origin, QRectF(ink).translated(origin)

    def mark_rect(self) -> QRectF:
        """Where the expand mark's ink is: centred on the strip, on the spot an open grip has its
        middle dot. Empty when there is no mark."""
        if not self.mark():
            return QRectF()
        return self._mark_origin()[1]

    def dot_centres(self) -> list[QPointF]:
        """The grip dots: three at the centre of an open grip; on a shut strip, the mark takes
        the middle one's place and the other two sit just above and below it."""
        rect = self.rect()
        centre = QPointF(rect.width() / 2, rect.height() / 2)
        if self.orientation() != Qt.Orientation.Horizontal:
            return [QPointF(centre.x() + step, centre.y()) for step in _DOT_STEPS]
        mark = self.mark_rect()
        if mark.isEmpty():
            return [QPointF(centre.x(), centre.y() + step) for step in _DOT_STEPS]
        reach = mark.height() / 2 + _MARK_GAP + 1.6
        return [
            QPointF(centre.x(), centre.y() - reach),
            QPointF(centre.x(), centre.y() + reach),
        ]

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        theme = plot_theme.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        horizontal = self.orientation() == Qt.Orientation.Horizontal
        shut = self._shut_pane()
        painter.setPen(QPen(QColor(theme.grid), 1))
        if shut is not None:
            # Shut, the strip is a panel's edge: a hairline on the side the content is.
            x = rect.width() - 1 if shut == 0 else 0
            if horizontal:
                painter.drawLine(QPointF(x + 0.5, 0), QPointF(x + 0.5, rect.height()))
            else:
                y = rect.height() - 1 if shut == 0 else 0
                painter.drawLine(QPointF(0, y + 0.5), QPointF(rect.width(), y + 0.5))
        else:
            centre = QPointF(rect.center())
            if horizontal:
                painter.drawLine(
                    QPointF(centre.x(), 0), QPointF(centre.x(), rect.height())
                )
            else:
                painter.drawLine(
                    QPointF(0, centre.y()), QPointF(rect.width(), centre.y())
                )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.text_muted))
        for dot in self.dot_centres():
            painter.drawEllipse(dot, 1.6, 1.6)
        mark = self.mark()
        if mark:
            painter.setPen(QColor(theme.text))
            painter.setFont(self._mark_font())
            painter.drawText(self._mark_origin()[0], mark)
            painter.setFont(self.font())
            self._paint_name(painter, self.splitter().pane_name(shut), theme)
        if self.hasFocus():
            # The focus ring: where Enter or Space would act. Inset so the handle's own bounds
            # hold all of it, on an open grip as thin as `HANDLE_PX`.
            ring = QPen(self.palette().color(QPalette.ColorRole.Highlight), 2)
            painter.setPen(ring)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(rect).adjusted(1, 1, -1, -1))
        painter.end()

    def _paint_name(self, painter: QPainter, name: str, theme) -> None:
        """The panel's name, written down the strip from its top (a quarter turn clockwise),
        stopping short of the dots round the mark."""
        top = _NAME_MARGIN
        length = int(self.dot_centres()[0].y()) - _NAME_MARGIN - top
        if length < _NAME_MIN_PX or not name:
            return
        metrics = self.fontMetrics()
        text = metrics.elidedText(name, Qt.TextElideMode.ElideRight, length)
        painter.save()
        painter.setPen(QColor(theme.text))
        painter.translate(self.width() / 2, top)
        painter.rotate(90)
        painter.drawText(
            QRectF(0, -self.width() / 2, length, self.width()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()


class GripSplitter(QSplitter):
    """``QSplitter`` with visible grips, side panels that fold to a labelled strip, and memory.

    ``key`` names where the split is remembered (``panes/<key>/``). Mark the side panels with
    :meth:`set_foldable` once the panes are added, then call :meth:`remember`.
    """

    #: ``(pane, shut)`` whenever a pane folds or opens, by any gesture.
    fold_changed = Signal(int, bool)

    def __init__(self, orientation: Qt.Orientation, *, key: str, parent=None):
        super().__init__(orientation, parent)
        if not key:
            raise ValueError(
                "a GripSplitter needs a key: its width and open/shut are remembered"
            )
        self.setHandleWidth(HANDLE_PX)
        self.setChildrenCollapsible(False)
        self._key = key
        #: Foldable panes and the name each shows on its strip.
        self._names: dict[int, str] = {}
        self._folded: set[int] = set()
        #: The width each foldable pane had when it was last open, for reopening.
        self._open: dict[int, int] = {}
        self._remembering = False
        self.splitterMoved.connect(self._on_moved)

    def createHandle(self) -> QSplitterHandle:  # noqa: N802 — Qt's name
        return _GripHandle(self.orientation(), self)

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802 — Qt's name
        self.insertWidget(self.count(), widget)

    def insertWidget(
        self, index: int, widget: QWidget
    ) -> None:  # noqa: N802 — Qt's name
        """Every pane is set not collapsible, explicitly: Qt's per-pane default is a third
        state that ``isCollapsible`` reports as *true* whatever ``childrenCollapsible`` says.
        """
        super().insertWidget(index, widget)
        self.setCollapsible(self.indexOf(widget), False)

    @property
    def key(self) -> str:
        return self._key

    # -- which panes fold -------------------------------------------------------------- #
    def set_foldable(self, pane: int, name: str) -> None:
        """Let ``pane`` (the first or the last) fold to a strip reading ``name``."""
        if pane not in (0, self.count() - 1) or self.count() < 2:
            raise ValueError(
                f"pane {pane} of {self.count()}: only the first or last pane folds — the strip "
                "is the grip beside it, and a middle pane has two"
            )
        self._names[pane] = name
        self._sync_tooltips()

    def foldable_panes(self) -> dict[int, str]:
        return dict(self._names)

    def pane_name(self, pane: int) -> str:
        return self._names.get(pane, "")

    def set_pane_name(self, pane: int, name: str) -> None:
        """Rename ``pane``'s strip — e.g. "Filters · 2 active"."""
        if pane not in self._names or self._names[pane] == name:
            return
        self._names[pane] = name
        self._sync_tooltips()
        handle = self.handle(self._handle_of(pane))
        if handle is not None:
            handle.update()

    def governed_pane(self, handle_index: int) -> int | None:
        """The foldable pane handle ``handle_index`` opens and shuts, if any."""
        if handle_index == 1 and 0 in self._names:
            return 0
        last = self.count() - 1
        if handle_index == last and last in self._names:
            return last
        return None

    def _handle_of(self, pane: int) -> int:
        return 1 if pane == 0 else pane

    def strip_width(self) -> int:
        return max(STRIP_PX, self.fontMetrics().height() + 2)

    def minimum_of(self, pane: int) -> int:
        """The narrowest ``pane`` may be dragged open — its own minimum, never under the floor."""
        widget = self.widget(pane)
        horizontal = self.orientation() == Qt.Orientation.Horizontal
        own = max(
            widget.minimumWidth() if horizontal else widget.minimumHeight(),
            (
                widget.minimumSizeHint().width()
                if horizontal
                else widget.minimumSizeHint().height()
            ),
        )
        return max(own, PANE_FLOOR_PX)

    # -- shut and open ------------------------------------------------------------------ #
    def is_folded(self, pane: int) -> bool:
        return pane in self._folded

    def toggle(self, pane: int) -> None:
        self.set_folded(pane, not self.is_folded(pane))

    def set_folded(self, pane: int, folded: bool, *, width: int | None = None) -> None:
        """Shut ``pane`` to its strip, or open it at ``width`` (else the width it last had)."""
        if pane not in self._names:
            raise ValueError(f"pane {pane} does not fold")
        folded = bool(folded)
        if folded == self.is_folded(pane):
            return
        sizes = self.sizes()
        neighbour = 1 if pane == 0 else pane - 1
        handle = self.handle(self._handle_of(pane))
        focus = QApplication.focusWidget()
        holds_focus = focus is not None and (
            focus is self.widget(pane) or self.widget(pane).isAncestorOf(focus)
        )
        if folded:
            # Only a laid-out width is the user's: before the first show it is Qt's guess.
            if sizes[pane] > 0 and self.isVisible():
                self._open[pane] = sizes[pane]
            self._folded.add(pane)
            self.setCollapsible(pane, True)
            handle.updateGeometry()
            self.refresh()
            freed = sizes[pane]
            sizes[pane] = 0
            sizes[neighbour] += freed
            self.setSizes(sizes)
        else:
            self._folded.discard(pane)
            handle.updateGeometry()
            self.refresh()
            self.setCollapsible(pane, False)
            want = width if width is not None else self._open.get(pane, 0)
            want = max(want, self.minimum_of(pane))
            sizes = self.sizes()
            total = sum(sizes)
            sizes[neighbour] = max(sizes[neighbour] - want, 1)
            sizes[pane] = want
            if total > 0:
                self.setSizes(sizes)
            else:
                self.setSizes([want if i == pane else s for i, s in enumerate(sizes)])
            self._open[pane] = want
        handle.update()
        self._sync_tooltips()
        if folded and holds_focus:
            # Left where it was, the focus sits on a widget nobody can see and keys go nowhere.
            handle.setFocus(Qt.FocusReason.OtherFocusReason)
        self._save()
        self.fold_changed.emit(pane, folded)

    def _sync_tooltips(self) -> None:
        """Each foldable pane's grip says what it does now, and points the way it does it: a
        hand on a strip (a click opens it), the resize arrows on an open grip."""
        horizontal = self.orientation() == Qt.Orientation.Horizontal
        resize = (
            Qt.CursorShape.SplitHCursor if horizontal else Qt.CursorShape.SplitVCursor
        )
        for pane, name in self._names.items():
            handle = self.handle(self._handle_of(pane))
            if handle is None:
                continue
            shut = self.is_folded(pane)
            handle.setToolTip(
                f"Click or drag to show {name}"
                if shut
                else f"Drag to resize; drag narrower or double-click to hide {name}"
            )
            handle.setCursor(Qt.CursorShape.PointingHandCursor if shut else resize)
            # Tab reaches it; a click does not take the focus from the pane being worked in.
            handle.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            handle.setAccessibleName(f"{'Show' if shut else 'Hide'} {name}")

    # -- the drag ----------------------------------------------------------------------- #
    def drag_handle(self, index: int, pos: int, *, opening: bool = False) -> None:
        """Move handle ``index`` to ``pos``: resize, or fold a side panel dragged past its
        minimum, or open a shut one dragged back out.

        ``opening`` is a drag that began on a shut strip. That one opens the panel as soon as
        it is pulled off the strip — at its minimum, then following the pointer — rather than
        only once the pointer has travelled the panel's whole minimum width with nothing on
        screen moving, which reads as a strip that cannot be dragged. Pulled back onto the
        strip in the same drag, the panel shuts again.
        """
        horizontal = self.orientation() == Qt.Orientation.Horizontal
        area = self.contentsRect()
        start = area.left() if horizontal else area.top()
        end = (area.right() if horizontal else area.bottom()) + 1
        pane = self.governed_pane(index)
        if pane is not None:
            would = pos - start if pane == 0 else end - pos - self.handleWidth()
            if opening:
                if would <= self.strip_width() // 2:
                    if not self.is_folded(pane):
                        self.set_folded(pane, True)
                    return
                if self.is_folded(pane):
                    self.set_folded(pane, False, width=would)
                    return
            elif would < self.minimum_of(pane):
                if not self.is_folded(pane):
                    self.set_folded(pane, True)
                return
            if self.is_folded(pane):
                self.set_folded(pane, False, width=would)
                return
        self.moveSplitter(self._clamped(index, pos), index)

    def _clamped(self, index: int, pos: int) -> int:
        """``pos`` held so neither pane beside handle ``index`` goes under its minimum."""
        horizontal = self.orientation() == Qt.Orientation.Horizontal

        def edge(rect, far: bool) -> int:
            if horizontal:
                return rect.right() + 1 if far else rect.left()
            return rect.bottom() + 1 if far else rect.top()

        before = index - 1
        left = (
            edge(self.contentsRect(), False)
            if before == 0
            else edge(self.handle(before).geometry(), True)
        )
        right = (
            edge(self.contentsRect(), True)
            if index + 1 >= self.count()
            else edge(self.handle(index + 1).geometry(), False)
        )
        low = left + self.minimum_of(before)
        high = right - self.handleWidth() - self.minimum_of(index)
        return max(low, min(pos, high)) if low <= high else pos

    def _on_moved(self, *_args) -> None:
        sizes = self.sizes()
        for pane in self._names:
            if not self.is_folded(pane) and sizes[pane] > 0:
                self._open[pane] = sizes[pane]
        self._save()

    # -- memory ------------------------------------------------------------------------- #
    def _setting(self, name: str) -> str:
        return f"{SETTINGS_GROUP}/{self._key}/{name}"

    def remember(
        self,
        *,
        legacy_state: str | None = None,
        legacy_shut: Iterable[int] = (),
        legacy_open: dict[int, int] | None = None,
        legacy_keys: Iterable[str] = (),
    ) -> bool:
        """Restore the saved split (call once the panes are added and marked), and save every
        change from now on.

        Returns whether a saved split was restored, so the caller keeps its default otherwise.

        The ``legacy_*`` arguments carry a split saved before the one scheme existed — a
        ``saveState`` blob (``splitters/<key>`` by default), panes it had hidden, widths it
        remembered — into ``panes/<key>/`` once, after which ``legacy_keys`` are removed.
        """
        settings = QSettings()
        sizes = _ints(settings.value(self._setting("sizes")))
        restored = False
        if len(sizes) == self.count() and self._usable(sizes):
            self._open.update(
                {
                    i: w
                    for i, w in enumerate(_ints(settings.value(self._setting("open"))))
                    if i in self._names and w > 0
                }
            )
            self.setSizes(sizes)
            for pane in _ints(settings.value(self._setting("shut"))):
                if pane in self._names:
                    self.set_folded(pane, True)
            restored = True
        else:
            restored = self._adopt_legacy(
                settings,
                legacy_state or f"splitters/{self._key}",
                set(legacy_shut),
                dict(legacy_open or {}),
            )
            if restored:
                self._remembering = True
                self._save()
        for key in (*legacy_keys, legacy_state or f"splitters/{self._key}"):
            settings.remove(key)
        self._remembering = True
        return restored

    def _usable(self, sizes: list[int]) -> bool:
        """A saved split restores only if every pane that cannot fold is open in it."""
        return all(s > 0 or i in self._names for i, s in enumerate(sizes))

    def _adopt_legacy(self, settings, state_key, shut, open_widths) -> bool:
        sizes = _legacy_sizes(settings.value(state_key))
        restored = len(sizes) == self.count() and self._usable(sizes)
        if restored:
            # The old splitters could drag any pane to nothing. A side panel left at zero is
            # shut; a main pane left at zero means the blob is not one to trust (`_usable`).
            shut |= {i for i, s in enumerate(sizes) if s == 0}
            self.setSizes([s if s > 0 else 1 for s in sizes])
        for pane, width in open_widths.items():
            if pane not in self._names or width <= 0:
                continue
            self._open[pane] = width
            restored = True
            if pane in shut:
                continue
            sizes = self.sizes()
            total = sum(sizes) or width + 1000 * (self.count() - 1)
            rest = [i for i in range(self.count()) if i != pane]
            share = max(total - width, 1) // len(rest)
            self.setSizes([width if i == pane else share for i in range(self.count())])
        for pane in shut:
            if pane in self._names:
                self.set_folded(pane, True)
                restored = True
        return bool(restored)

    def _save(self) -> None:
        if not self._remembering:
            return
        settings = QSettings()
        settings.setValue(self._setting("sizes"), _joined(self.sizes()))
        settings.setValue(
            self._setting("open"),
            _joined(self._open.get(i, 0) for i in range(self.count())),
        )
        settings.setValue(self._setting("shut"), _joined(sorted(self._folded)))
