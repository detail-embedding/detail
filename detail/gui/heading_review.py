"""The heading sampler's two surfaces (§26 *Headings*): the coverage compass on the labeling
screen, and the per-direction review window it opens.

The compass draws, per camera-relative direction, how many of the example set's frames face
that way and how many of those are marked, with the camera at the bottom; a direction short of
frames is drawn as a gap. Clicking one opens :class:`HeadingReviewDialog` on it: the direction's
frames in the set above its ranked candidates. **Reject** turns an in-set frame down for this
direction (it goes to the set's ``rejected``) and puts the next candidate in its place; **Not
this direction** turns a candidate down; ticked candidates are copied in.

Everything slow — decoding a candidate's frame, reading a set frame's picture, copying frames
in — runs off the UI thread (``gui/busy.py``). The pool the window ranks from was built before
it opened and is held in memory, so paging and rejecting cost a ranking, not a heading pass.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.core.examples import HEADING_BINS, HEADING_BUCKETS
from detail.core.visualize import draw_bbox
from detail.gui.busy import run_busy
from detail.gui.dialog import Dialog
from detail.gui.full_frame import to_pixmap
from detail.gui.sizing import emphasis_font, set_available

#: How the window names a bucket.
BUCKET_LABELS = {
    "right": "Right",
    "right-toward": "Right, toward",
    "toward": "Toward",
    "left-toward": "Left, toward",
    "left": "Left",
    "left-away": "Left, away",
    "away": "Away",
    "right-away": "Right, away",
    "rearing": "Rearing",
    "unclear": "Unclear",
}
#: How a candidate's heading was known, as a tile says it.
SOURCE_LABELS = {
    "walking+tail": "walking",
    "walking": "walking",
    "tail": "tail",
    "similar": "similar frames",
    None: "",
}
#: A direction with fewer frames than this in the set is drawn as a gap.
ENOUGH = 3
PAGE = 8
TILE_PX = 150


class HeadingCompass(QWidget):
    """Eight directions around a plate seen from above, the camera at the bottom; beside it
    rearing, unclear and the frames with no heading. ``clicked(bucket)`` when one is pressed —
    the hand cursor is shown only while something is connected (a cue derived from the
    wiring)."""

    clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self._coverage: dict = {}
        self._selected: str | None = None
        self._compact = compact
        size = 64 if compact else 220
        self.setMinimumSize(size + (0 if compact else 110), size)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setToolTip(
            "Example frames per direction the animal faces, as marked / in the set; "
            "the camera is at the bottom. Click a direction to review and add frames there."
        )

    def connectNotify(self, signal) -> None:  # noqa: N802 — Qt's name
        super().connectNotify(signal)
        if self.receivers(self.clicked) > 0:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_coverage(self, coverage: dict) -> None:
        self._coverage = dict(coverage)
        self.update()

    def set_selected(self, bucket: str | None) -> None:
        self._selected = bucket
        self.update()

    def coverage(self) -> dict:
        return dict(self._coverage)

    # -- geometry ------------------------------------------------------------ #
    def _dial(self) -> tuple[QPointF, float]:
        side = min(self.height(), self.width() if self._compact else self.height())
        # room below for the camera mark and its label
        r = side / 2 - (8 if self._compact else 40)
        return QPointF(side / 2, self.height() / 2 - (2 if self._compact else 10)), max(
            r, 8.0
        )

    def _bin_center(self, b: int) -> QPointF:
        c, r = self._dial()
        a = math.radians(b * 45.0)  # 0 = right, +90 = toward the camera (down)
        return QPointF(c.x() + r * math.cos(a), c.y() + r * math.sin(a))

    def _side_rects(self) -> dict[str, QRectF]:
        if self._compact:
            return {}
        x = self.height() + 6
        return {
            name: QRectF(x, 30 + 26 * k, self.width() - x - 4, 22)
            for k, name in enumerate(("rearing", "unclear", "hand-picked"))
        }

    def bucket_at(self, point: QPointF) -> str | None:
        """The bucket under ``point``, or None."""
        for b in range(8):
            center = self._bin_center(b)
            reach = 10 if self._compact else 28
            if math.hypot(point.x() - center.x(), point.y() - center.y()) <= reach:
                return HEADING_BINS[b]
        for name, rect in self._side_rects().items():
            if rect.contains(point) and name in HEADING_BUCKETS:
                return name
        return None

    # -- painting ------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        text = pal.windowText().color()
        c, r = self._dial()
        p.setPen(QPen(pal.mid().color(), 1))
        p.drawEllipse(c, r, r)
        p.setBrush(pal.mid().color())
        below = 6 if self._compact else 20  # clear of the toward box
        p.drawRect(QRectF(c.x() - 7, c.y() + r + below, 14, 6))  # the camera
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF() - (2 if self._compact else 0), 6))
        p.setFont(font)
        if not self._compact:
            p.setPen(pal.placeholderText().color())
            p.drawText(
                QRectF(c.x() - 40, c.y() + r + below + 6, 80, 14),
                Qt.AlignmentFlag.AlignCenter,
                "camera",
            )
            # the four main directions, just inside the ring next to their boxes
            inset, w, h = 30, 40, 14
            flags = Qt.AlignmentFlag
            for rect, align, label in (
                (
                    QRectF(c.x() + r - inset - w, c.y() - h / 2, w, h),
                    flags.AlignRight,
                    "right",
                ),
                (
                    QRectF(c.x() - r + inset, c.y() - h / 2, w, h),
                    flags.AlignLeft,
                    "left",
                ),
                (
                    QRectF(c.x() - w, c.y() - r + 18, 2 * w, h),
                    flags.AlignHCenter,
                    "away",
                ),
                (
                    QRectF(c.x() - w, c.y() + r - 18 - h, 2 * w, h),
                    flags.AlignHCenter,
                    "toward",
                ),
            ):
                p.drawText(rect, align | flags.AlignVCenter, label)
        for b, name in enumerate(HEADING_BINS):
            center = self._bin_center(b)
            cov = self._coverage.get(name, {"in_set": 0, "marked": 0})
            gap = cov["in_set"] < ENOUGH
            fill = QColor("#E1F5EE") if not gap else QColor("#FAEEDA")
            edge = QColor("#0F6E56") if not gap else QColor("#854F0B")
            if name == self._selected:
                edge = pal.highlight().color()
            p.setPen(QPen(edge, 2 if name == self._selected else 1))
            p.setBrush(fill)
            half = 8 if self._compact else 27
            p.drawRoundedRect(
                QRectF(
                    center.x() - half, center.y() - half * 0.6, 2 * half, half * 1.2
                ),
                4,
                4,
            )
            if not self._compact:
                p.setPen(QColor("#04342C") if not gap else QColor("#412402"))
                p.drawText(
                    QRectF(
                        center.x() - half, center.y() - half * 0.6, 2 * half, half * 1.2
                    ),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{cov['marked']}/{cov['in_set']}",
                )
        p.setPen(text)
        for name, rect in self._side_rects().items():
            cov = self._coverage.get(name, {"in_set": 0, "marked": 0})
            label = BUCKET_LABELS.get(name, "Hand-picked")
            p.drawText(
                rect,
                Qt.AlignmentFlag.AlignVCenter,
                f"{label}  {cov['marked']}/{cov['in_set']}",
            )
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        bucket = self.bucket_at(event.position())
        if bucket is not None and self.receivers(self.clicked) > 0:
            self.clicked.emit(bucket)
        super().mousePressEvent(event)


def _crop(rgb: np.ndarray, bbox) -> np.ndarray:
    from detail.pipeline.subject_frames import subject_crop

    return cv2.resize(
        subject_crop(rgb, bbox), (TILE_PX, TILE_PX), interpolation=cv2.INTER_AREA
    )


def _candidate_pictures(experiment, candidates: list[dict]) -> list[np.ndarray | None]:
    """Off the UI thread: each candidate's subject crop, decoded exactly."""
    from detail.pipeline.subject_frames import _decode

    out = []
    for cand in candidates:
        rgb = _decode(experiment.dir, cand["recording_id"], cand)
        out.append(None if rgb is None else _crop(rgb, cand["bbox"]))
    return out


def _set_pictures(
    experiment, set_id: str, rows: list[tuple[int, tuple]]
) -> list[np.ndarray]:
    """Off the UI thread: each set frame's subject crop, from its stored picture."""
    return [_crop(experiment.example_frame_image(set_id, i), bbox) for i, bbox in rows]


def _full_candidate(experiment, cand: dict) -> np.ndarray:
    """Off the UI thread (FullFrameDialog renders there): a candidate's whole frame, its box."""
    from detail.pipeline.subject_frames import _decode

    rgb = _decode(experiment.dir, cand["recording_id"], cand)
    if rgb is None:
        raise ValueError("the frame could not be decoded exactly")
    return draw_bbox(rgb, cand["bbox"], (255, 220, 60))


def _full_set_frame(experiment, set_id: str, index: int, bbox) -> np.ndarray:
    """Off the UI thread: a set frame's stored picture, whole, with its box."""
    return draw_bbox(
        experiment.example_frame_image(set_id, index), bbox, (255, 220, 60)
    )


class _Thumb(QLabel):
    """A tile's crop: small, for telling frames apart; a double click opens the whole frame,
    zoomable, in a window of its own — the answer to wanting a better look at a paw."""

    def __init__(self, render, title: str) -> None:
        super().__init__("Decoding frame…")
        self._render, self._title = render, title
        self.setToolTip("Double-click to open the whole frame.")

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt's name
        from detail.gui.full_frame import FullFrameDialog

        dialog = FullFrameDialog(self._render, title=self._title, parent=self.window())
        dialog.open()


class _Tile(QFrame):
    """One frame: its crop, where it came from, and its actions."""

    def __init__(
        self,
        caption: str,
        tag: str,
        actions: list[tuple[str, object]],
        render,
        checkable=False,
    ):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(TILE_PX + 30)
        column = QVBoxLayout(self)
        column.setContentsMargins(4, 4, 4, 4)
        self.picture = _Thumb(render, caption.lstrip("✓· "))
        self.picture.setFixedSize(TILE_PX, TILE_PX)
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.picture)
        top = QHBoxLayout()
        self.check = QCheckBox(caption) if checkable else None
        top.addWidget(self.check if checkable else QLabel(caption), 1)
        if tag:
            top.addWidget(QLabel(tag))
        column.addLayout(top)
        row = QHBoxLayout()
        self.buttons = {}
        for label, slot in actions:
            button = QPushButton(label)
            button.setAutoDefault(False)
            button.clicked.connect(slot)
            row.addWidget(button)
            self.buttons[label] = button
        row.addStretch(1)
        column.addLayout(row)

    def show_picture(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self.picture.setText("(could not decode)")
        else:
            self.picture.setPixmap(to_pixmap(rgb))


class HeadingReviewDialog(Dialog):
    """One direction's frames in the set above its ranked candidates (§26 *Headings*).

    ``changed`` when the set was modified (a reject, an add); ``open_frame(index)`` asks the
    labeling screen to go to an in-set frame.
    """

    changed = pyqtSignal()
    open_frame = pyqtSignal(int)

    def __init__(
        self, experiment, set_id: str, heading_pool, bucket: str, parent=None
    ) -> None:
        super().__init__(parent)
        if bucket not in HEADING_BUCKETS:
            raise ValueError(
                f"unknown heading bucket {bucket!r}; expected one of {HEADING_BUCKETS}"
            )
        self._experiment = experiment
        self._set_id = set_id
        self._pool = heading_pool
        self._bucket = bucket
        self._shown = PAGE
        self.setWindowTitle("Review a direction")
        body = self.body()
        head = QHBoxLayout()
        self._compass = HeadingCompass(compact=True)
        self._compass.clicked.connect(self.show_bucket)
        head.addWidget(self._compass)
        titles = QVBoxLayout()
        self._title = QLabel()
        self._title.setFont(emphasis_font(self._title, 2))
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        titles.addWidget(self._title)
        titles.addWidget(self._summary)
        head.addLayout(titles, 1)
        # ◀ ▶ step; ‹ › are a side panel's fold mark (`a_side_panel_has_no_hide_button`).
        self._prev = QPushButton("◀ Previous")
        self._prev.setToolTip("The previous direction round the compass")
        self._next = QPushButton("Next ▶")
        self._next.setToolTip("The next direction round the compass")
        for button, step in ((self._prev, -1), (self._next, +1)):
            button.setAutoDefault(False)
            button.clicked.connect(lambda _=False, s=step: self._step(s))
            head.addWidget(button)
        body.addLayout(head)
        body.addWidget(
            QLabel(
                "In the set — Reject swaps in the next best guess and remembers the reject"
            )
        )
        self._in_set = QGridLayout()
        self._in_set.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        body.addLayout(self._in_set)
        body.addWidget(QLabel("Candidates, best first"))
        self._candidates = QGridLayout()
        self._candidates.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        body.addLayout(self._candidates)
        buttons = QDialogButtonBox()
        self._more = buttons.addButton(
            "Show more", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._more.clicked.connect(self._show_more)
        self._add = buttons.addButton(
            "Add selected", QDialogButtonBox.ButtonRole.ApplyRole
        )
        self._add.clicked.connect(self._add_selected)
        close = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        close.clicked.connect(self.accept)
        for button in buttons.buttons():
            button.setAutoDefault(False)
        self.set_buttons(buttons)
        self._candidate_rows: list[dict] = []
        self._candidate_tiles: list[_Tile] = []
        self.show_bucket(bucket)

    # -- public -------------------------------------------------------------- #
    @property
    def bucket(self) -> str:
        return self._bucket

    def candidates(self) -> list[dict]:
        return list(self._candidate_rows)

    def show_bucket(self, bucket: str) -> None:
        self._bucket = bucket
        self._shown = PAGE
        self._refresh()

    # -- building ------------------------------------------------------------ #
    def _step(self, step: int) -> None:
        order = HEADING_BUCKETS
        self.show_bucket(order[(order.index(self._bucket) + step) % len(order)])

    def _show_more(self) -> None:
        self._shown += PAGE
        self._refresh()

    def _refresh(self) -> None:
        experiment, set_id = self._experiment, self._set_id
        current = experiment.example_set(set_id)
        coverage = experiment.heading_coverage(set_id, self._pool)
        self._compass.set_coverage(coverage)
        self._compass.set_selected(self._bucket)
        cov = coverage[self._bucket]
        self._candidate_rows = experiment.heading_candidates(
            set_id, self._pool, self._bucket, self._shown
        )
        self._title.setText(BUCKET_LABELS[self._bucket])
        self._summary.setText(
            f"{cov['marked']} marked of {cov['in_set']} in the set · "
            f"{len(self._candidate_rows)} candidates shown"
        )
        from detail.pipeline.headings import frame_buckets

        buckets = frame_buckets(self._pool, current)
        in_set = [
            (i, f) for i, f in enumerate(current.frames) if buckets[i] == self._bucket
        ]
        self._fill_in_set(current, in_set)
        self._fill_candidates()
        set_available(self._add, False, "Tick candidates to add them.")
        set_available(
            self._more,
            len(self._candidate_rows) >= self._shown,
            "Every candidate in this direction is shown.",
        )

    def _clear(self, grid: QGridLayout) -> None:
        while grid.count():
            item = grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _fill_in_set(self, current, in_set) -> None:
        self._clear(self._in_set)
        names = self._experiment.recording_names()
        tiles = []
        for k, (i, frame) in enumerate(in_set):
            marked = "✓" if current.labeled(i) else "·"
            source = (
                frame.sampled_by.split(":")[2]
                if frame.sampled_by.count(":") >= 2
                else ""
            )
            tile = _Tile(
                f"{marked} {names.get(frame.recording_id, frame.recording_id)} f{frame.frame_idx}",
                SOURCE_LABELS.get(source, source),
                [
                    ("Open", lambda _=False, i=i: self._open(i)),
                    ("Reject", lambda _=False, i=i: self._reject(i)),
                ],
                lambda e=self._experiment, s=self._set_id, i=i, b=frame.bbox: (
                    _full_set_frame(e, s, i, b)
                ),
            )
            self._in_set.addWidget(tile, k // 4, k % 4)
            tiles.append(tile)
        if not in_set:
            self._in_set.addWidget(
                QLabel(
                    "No frames in this direction yet: tick candidates below to add some."
                ),
                0,
                0,
            )
            return
        rows = [(i, frame.bbox) for i, frame in in_set]
        experiment, set_id = self._experiment, self._set_id
        run_busy(
            self,
            lambda: _set_pictures(experiment, set_id, rows),
            lambda pictures: [t.show_picture(p) for t, p in zip(tiles, pictures)],
            label="Reading frames…",
            supersedes="heading_review_in_set",
        )

    def _fill_candidates(self) -> None:
        self._clear(self._candidates)
        names = self._experiment.recording_names()
        self._candidate_tiles = []
        for k, cand in enumerate(self._candidate_rows):
            tile = _Tile(
                f"{names.get(cand['recording_id'], cand['recording_id'])} f{cand['frame_idx']}",
                SOURCE_LABELS.get(cand["source"], ""),
                [("Not this direction", lambda _=False, k=k: self._not_this(k))],
                lambda e=self._experiment, c=cand: _full_candidate(e, c),
                checkable=True,
            )
            tile.check.toggled.connect(self._selection_changed)
            self._candidates.addWidget(tile, k // 4, k % 4)
            self._candidate_tiles.append(tile)
        if not self._candidate_rows:
            self._candidates.addWidget(
                QLabel("No candidates left in this direction."), 0, 0
            )
            return
        experiment, rows, tiles = (
            self._experiment,
            list(self._candidate_rows),
            list(self._candidate_tiles),
        )
        run_busy(
            self,
            lambda: _candidate_pictures(experiment, rows),
            lambda pictures: [t.show_picture(p) for t, p in zip(tiles, pictures)],
            label="Decoding frames…",
            supersedes="heading_review_candidates",
        )

    # -- actions ------------------------------------------------------------- #
    def selected(self) -> list[dict]:
        return [
            c
            for c, t in zip(self._candidate_rows, self._candidate_tiles)
            if t.check.isChecked()
        ]

    def _selection_changed(self) -> None:
        n = len(self.selected())
        self._add.setText(f"Add {n} selected" if n else "Add selected")
        set_available(self._add, n > 0, "Tick candidates to add them.")

    def _add_selected(self, *, chosen: list[dict] | None = None) -> None:
        chosen = self.selected() if chosen is None else chosen
        if not chosen:
            return
        experiment, set_id, bucket = self._experiment, self._set_id, self._bucket
        set_available(self._add, False, "Adding frames…")
        run_busy(
            self,
            lambda: experiment.add_heading_candidates(set_id, chosen, bucket),
            self._after_change,
            label="Adding frames…",
        )

    def _not_this(self, k: int) -> None:
        cand = self._candidate_rows[k]
        self._experiment.reject_candidate(self._set_id, cand["key"], self._bucket)
        self._after_change(None)

    def _reject(self, index: int) -> None:
        """Turn an in-set frame down for this direction and put the next candidate in its
        place. A frame with marks asks first: the marks go with it."""
        current = self._experiment.example_set(self._set_id)
        if current.frames[index].marks:
            answer = QMessageBox.question(
                self,
                "Reject frame",
                "This frame has marks on it; rejecting it removes them. Reject it?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._experiment.reject_example_frame(self._set_id, index, self._bucket)
        replacement = self._experiment.heading_candidates(
            self._set_id, self._pool, self._bucket, 1
        )
        if replacement:
            self._add_selected(chosen=replacement)
        else:
            self._after_change(None)

    def _open(self, index: int) -> None:
        self.open_frame.emit(index)

    def _after_change(self, _result) -> None:
        self.changed.emit()
        self._refresh()
