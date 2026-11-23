"""A row of controls that wraps instead of being squeezed.

The class this exists for: **a control row inside a player is handed the video's width, and
nothing enforces its minimum.** ``image_view.PictureWithControls`` positions its two children
with ``setGeometry`` rather than through a ``QLayout`` (its own docstring says why — a layout
stretch recomputed from a constraint that layout's own child just changed is reentrant), so the
transport gets exactly the block's width whatever it asked for. Measured on ``FrameScrubber``:
the row wants 932 px at the interface font's smallest size and 1144 px at 12 pt, and cannot go
below 878 px without Qt eliding the labels. At a 357 px pane "Prev frame" rendered as "ev fram".

Eliding is the worst of the available answers because it is silent and it degrades the *labels*
— the one part of a button that says what it does. This package has no icons to fall back on
("Icons: none. Text labels only"), and hiding controls behind an overflow trades a legibility
problem for a discoverability one. Wrapping keeps every control, keeps every label whole, and
costs one row of height only while the pane is too narrow to do better.

**The trailing group.** ``addStretch`` in a horizontal row means "push what follows to the right
edge", and one caller uses it that way — the frame readout sits right while the buttons sit
left. A flow layout has no single right edge to push to, so the spacer is kept as a real item
and read as a *marker*: everything after it is right-aligned on whatever line it lands on. On a
row that fits, that is the same picture the ``QHBoxLayout`` drew, which is what makes this a
drop-in replacement rather than a redesign of eight surfaces.

**The trailing group never wraps alone.** When the row is short by less than the trailing
group's width, the group is all that wraps — the frame counter on a line by itself, under a full
line of buttons, reading as a stray (measured in the clip window at 1324px). So when the group
would start a line holding nothing else, the last control before it goes down with it: two
lines either way, and the second one reads as a row.
"""

from detail.gui.qt import QLayout, QPoint, QRect, QSize, QSizePolicy, QWidget, Qt


class FlowLayout(QLayout):
    """A horizontal row that wraps onto further lines when it runs out of width.

    Supports the subset of ``QHBoxLayout`` this package's control rows use — ``addWidget``,
    ``insertWidget``, ``addStretch``, ``indexOf``, ``itemAt``, ``count`` — so a row can move to
    it without its host changing.
    """

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list = []
        self._space = spacing
        self.setSpacing(spacing)

    # -- QLayout plumbing ----------------------------------------------------- #
    def addItem(self, item) -> None:  # noqa: N802 - Qt virtual
        self._items.append(item)

    def insertWidget(self, index: int, widget: QWidget) -> None:  # noqa: N802
        """``QBoxLayout.insertWidget``'s shape, which ``FrameScrubber.add_control`` calls.

        ``QLayout`` has no insert of its own: ``addWidget`` goes through the ``addItem``
        virtual above, so the item lands at the end and is moved into place here. Doing the
        reparenting through ``addWidget`` rather than by hand is what keeps a host widget
        owned and shown the way a layout-managed child is.
        """
        self.addWidget(widget)
        self._items.insert(index, self._items.pop())

    def addStretch(self, _stretch: int = 0) -> None:  # noqa: N802
        """Mark the start of the trailing, right-aligned group. See the module docstring."""
        from detail.gui.qt import QSpacerItem

        self.addItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        )

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt virtual
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt virtual
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt virtual
        """Nothing. A row that wraps asks for the width it is given, never for more."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt virtual
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt virtual
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt virtual
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        """One line of everything — what the row would like, and what it gets when it fits."""
        left, top, right, bottom = self.getContentsMargins()
        width = height = 0
        widgets = [i for i in self._items if i.spacerItem() is None]
        for item in widgets:
            hint = item.sizeHint()
            width += hint.width()
            height = max(height, hint.height())
        width += self._space * max(len(widgets) - 1, 0)
        return QSize(width + left + right, height + top + bottom)

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt virtual
        """The **widest single control**, not the whole row.

        This is the point of the class: the row's floor is one item, because everything else
        can go on another line. A floor of the whole row is what left no answer but eliding.
        """
        left, top, right, bottom = self.getContentsMargins()
        width = height = 0
        for item in self._items:
            if item.spacerItem() is not None:
                continue
            size = item.minimumSize()
            width = max(width, size.width())
            height = max(height, size.height())
        return QSize(width + left + right, height + top + bottom)

    # -- the arrangement ------------------------------------------------------ #
    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        """Place every item, or just measure. Returns the height the row needs.

        One pass, from this rect's width alone — never from anything already drawn, so
        repeated calls settle rather than chase each other (the same rule
        ``PictureWithControls._reflow`` states). A second pass only when the first left the
        trailing group alone on its line (see the module docstring), and it is kept only if
        it fixed that.
        """
        left, top, right, bottom = self.getContentsMargins()
        area = rect.adjusted(left, top, -right, -bottom)
        placed, trailing_at, end = self._place(area, None)
        orphan = self._orphaned_trailing(placed, trailing_at)
        if orphan is not None:
            again, _at, again_end = self._place(area, orphan)
            if self._orphaned_trailing(again, trailing_at) is None and (
                again[orphan][2] == again[trailing_at][2]
            ):
                placed, end = again, again_end
        self._right_align_trailing(placed, trailing_at, area)
        if apply:
            for item, geometry, _line in placed:
                item.setGeometry(geometry)
        return end - rect.y() + bottom

    def _place(self, area: QRect, break_before: int | None):
        """``([item, rect, line], ...)``, where the trailing group starts, and the bottom —
        flowing left to right, a new line wherever the next item does not fit or at
        ``break_before`` (an index into the placed items)."""
        x, y = area.x(), area.y()
        line_height = 0
        line = 0
        placed: list = []
        trailing_at: int | None = None
        for item in self._items:
            if item.spacerItem() is not None:
                trailing_at = len(placed)
                continue
            hint = item.sizeHint()
            forced = break_before is not None and len(placed) == break_before
            if placed and (forced or x + hint.width() > area.right() + 1):
                x = area.x()
                y += line_height + self._space
                line_height = 0
                line += 1
            placed.append([item, QRect(QPoint(x, y), hint), line])
            x += hint.width() + self._space
            line_height = max(line_height, hint.height())
        return placed, trailing_at, y + line_height

    @staticmethod
    def _orphaned_trailing(placed: list, trailing_at: int | None) -> int | None:
        """The index of the control to carry down when the trailing group starts a line of
        its own after a line it did not fit on; ``None`` when it does not, or when that line
        holds one control only (carrying it would empty a line)."""
        if trailing_at is None or not 0 < trailing_at < len(placed):
            return None
        before, first = placed[trailing_at - 1], placed[trailing_at]
        if first[2] == before[2]:
            return None
        if sum(1 for p in placed[:trailing_at] if p[2] == before[2]) < 2:
            return None
        return trailing_at - 1

    @staticmethod
    def _right_align_trailing(
        placed: list, trailing_at: int | None, area: QRect
    ) -> None:
        """Push the group after the stretch to the right edge of the line it ended on.

        Only that line: a trailing group that wrapped has items above which are already where
        the flow put them, and dragging those right would open a hole in the middle of a line
        that is full.
        """
        if trailing_at is None or trailing_at >= len(placed):
            return
        last_line = placed[-1][2]
        group = [p for p in placed[trailing_at:] if p[2] == last_line]
        if not group:
            return
        shift = area.right() - group[-1][1].right()
        if shift <= 0:
            return
        for entry in placed[trailing_at:]:
            if entry[2] == last_line:
                entry[1] = entry[1].translated(shift, 0)
