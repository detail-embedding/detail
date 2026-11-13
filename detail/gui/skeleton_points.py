"""A choice of a skeleton's points (§26): one checkbox per point, for a setting declared with
``ui: skeleton_points`` (``track_points`` on the keypoint track methods).

It knows nothing of a body plan: the names and the left/right pairs come from whatever skeleton
the form hands it, in column order. Every point starts ticked; :meth:`value` is ``None`` while
they all are (the setting's "every point"), else the ticked names. A value set before the
skeleton is known (a restored config) is held, reported back as it was set, and applied when the
skeleton arrives; a choice made on one skeleton carries to the next for the names they share.

``wording`` is what the ticks are *for*, which only the note under them says: ``"track"`` (the
keypoint track form's points to track, with its note on a split left/right pair) or
``"trail"`` (the clip window's points that leave a trail, where a split pair means nothing).
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QGridLayout, QLabel, QVBoxLayout, QWidget

_COLUMNS = 4
_NO_SKELETON = (
    "Choose a keypoint detect run to list its points; until then, every point."
)
_UNSET = object()
#: The note under the ticks, by what they are for: no skeleton yet, none ticked, every point
#: ticked, some ticked — and whether a left/right pair split by the ticks is worth a note.
_WORDING = {
    "track": {
        "empty": _NO_SKELETON,
        "none": "Tick at least one point to track.",
        "every": "Every point is tracked.",
        "some": "Unticked points keep the detect run's placement, untouched.",
        "split": True,
    },
    "trail": {
        "empty": "No keypoints in this clip, so no point leaves a trail.",
        "none": "Tick at least one point to draw its trail.",
        "every": "Every point leaves a trail.",
        "some": "Only the ticked points leave a trail.",
        "split": False,
    },
}


class SkeletonPointsPicker(QWidget):
    """Tick the points to include; ``changed`` fires on every tick."""

    changed = pyqtSignal()

    def __init__(
        self, parent: QWidget | None = None, *, wording: str = "track"
    ) -> None:
        super().__init__(parent)
        if wording not in _WORDING:
            raise ValueError(f"wording={wording!r} is not one of {sorted(_WORDING)}")
        self._wording = _WORDING[wording]
        self._names: list[str] = []
        self._pairs: list[tuple[str, str]] = []
        self._boxes: dict[str, QCheckBox] = {}
        self._pending = _UNSET  # a value set before the skeleton was known
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        host = QWidget()
        self._grid = QGridLayout(host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(host)
        self._note = QLabel(self._wording["empty"])
        self._note.setWordWrap(True)
        layout.addWidget(self._note)

    # -- the skeleton ------------------------------------------------------------- #
    def set_skeleton(self, names, pairs=()) -> None:
        """Show ``names`` (column order), with ``pairs`` for the split-pair note."""
        if self._pending is not _UNSET:
            wanted = None if self._pending is None else set(self._pending)
            self._pending = _UNSET
        else:
            unticked = {n for n, box in self._boxes.items() if not box.isChecked()}
            wanted = {str(n) for n in names} - unticked
        for box in self._boxes.values():
            self._grid.removeWidget(box)
            box.deleteLater()
        self._names = [str(n) for n in names]
        self._pairs = [(str(a), str(b)) for a, b in pairs]
        self._boxes = {}
        for i, name in enumerate(self._names):
            box = QCheckBox(name)
            box.setChecked(wanted is None or name in wanted)
            box.toggled.connect(self._ticked)
            self._grid.addWidget(box, i // _COLUMNS, i % _COLUMNS)
            self._boxes[name] = box
        self._sync_note()

    def names(self) -> list[str]:
        return list(self._names)

    # -- the value ---------------------------------------------------------------- #
    def value(self) -> list[str] | None:
        """``None`` when every point is ticked, else the ticked names in column order
        (possibly empty, which the form refuses to launch). Before a skeleton: what was set.
        """
        if not self._boxes:
            return None if self._pending is _UNSET else self._pending
        ticked = [n for n in self._names if self._boxes[n].isChecked()]
        return None if len(ticked) == len(self._names) else ticked

    def set_value(self, value: list[str] | None) -> None:
        """Tick exactly ``value`` (``None``: every point)."""
        if not self._boxes:
            self._pending = None if value is None else list(value)
            return
        wanted = None if value is None else set(value)
        for name, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(wanted is None or name in wanted)
            box.blockSignals(False)
        self._sync_note()
        self.changed.emit()

    def _ticked(self, _checked: bool) -> None:
        self._sync_note()
        self.changed.emit()

    def _sync_note(self) -> None:
        words = self._wording
        if not self._boxes:
            self._note.setText(words["empty"])
            return
        ticked = {n for n, box in self._boxes.items() if box.isChecked()}
        split = [f"{a} / {b}" for a, b in self._pairs if (a in ticked) != (b in ticked)]
        if not ticked:
            text = words["none"]
        elif split and words["split"]:
            text = (
                "Not swapped left from right, since only one of each is tracked: "
                + "; ".join(split)
                + "."
            )
        elif len(ticked) == len(self._boxes):
            text = words["every"]
        else:
            text = words["some"]
        self._note.setText(text)
