"""One example set per detection class (§26 *One example set per detection class*).

For animals told apart at detection that look different, a patch method matches each class's
detections against its own example set. This widget is the per-class form of the shared
``ExampleSetPicker``: one block per class, the class heading its own picker. It is embedded
wherever such a map is chosen — the keypoint detect form, the distill dialog's patch teacher,
**Set up keypoints…** and a setup run's *Continue setup* step — so the four read and write the
map the same way (``class_example_sets``).

``checkable`` (default) gives each class a tick, for a surface where choosing the classes *is*
part of the map (the detect form, the distill dialog); without it every class is in, for a
surface whose classes were chosen already (a setup's animals). ``sources`` is handed to every
picker, as the setup page hands its single picker the teacher tracks.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from detail.core import class_names
from detail.gui.example_set_picker import ExampleSetPicker


class ClassExampleSets(QWidget):
    """One block per detection class: (with ``checkable``) a tick for whether that class is
    in the map, and that class's own example-set picker. Blocks are rebuilt when the classes
    change; a class's tick and set survive a rebuild that keeps it.
    """

    changed = pyqtSignal()

    def __init__(
        self, parent=None, *, experiment=None, checkable: bool = True, sources=None
    ) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._experiment = experiment
        self._checkable = checkable
        self._sources = sources
        self._rows: dict[str, tuple[QCheckBox, ExampleSetPicker]] = {}
        #: ``{stored class: name}`` — what each block is headed with (§1 *Class names*).
        self._labels: dict[str, str] = {}
        self._headings: list[QLabel] = []
        #: The classes a restored config ticked, waiting for the classes to be read.
        self._pending: set[str] | None = None

    def set_experiment(self, experiment) -> None:
        self._experiment = experiment
        for _tick, picker in self._rows.values():
            picker.set_experiment(experiment)

    def set_classes(self, names: list[str], *, labels: dict[str, str]) -> None:
        """One block per class in ``names`` (stored names, which :meth:`value` keys on),
        each headed by what the experiment calls it: ``labels`` is ``{stored: name}``,
        partial is fine (an absent class is shown by its stored name)."""
        self._labels = dict(labels)
        keep = {
            name: (tick.isChecked(), picker.set_id())
            for name, (tick, picker) in self._rows.items()
        }
        for tick, picker in self._rows.values():
            tick.deleteLater()
            picker.deleteLater()
        for heading in self._headings:
            heading.deleteLater()
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.layout() is not None:
                item.layout().deleteLater()
        self._rows = {}
        self._headings = []
        for name in names:
            # The class heads its own picker (skeleton, set, buttons), indented under it,
            # so each block reads as that class's.
            row = QHBoxLayout()
            row.addSpacing(24)
            shown = self._label(name)
            tick = QCheckBox(shown)
            tick.setToolTip(
                f"Place poses in the {shown} detections, matched against the example "
                "set below it"
            )
            picker = ExampleSetPicker(self._experiment, sources=self._sources)
            picker.setToolTip(
                f"The example set the {shown} detections are matched against"
            )
            checked, set_id = keep.get(name, (True, None))
            if self._pending is not None:
                checked = name in self._pending
            if not self._checkable:
                checked = True
            tick.setChecked(checked)
            if set_id:
                picker.select(set_id)
            picker.setEnabled(checked)
            tick.toggled.connect(picker.setEnabled)
            tick.toggled.connect(lambda _on: self.changed.emit())
            picker.changed.connect(self.changed.emit)
            row.addWidget(picker, 1)
            if self._checkable:
                self._layout.addWidget(tick)
            else:
                tick.hide()  # every class is in; the heading names it
                heading = QLabel(f"{shown}:")
                self._headings.append(heading)
                self._layout.addWidget(heading)
            self._layout.addLayout(row)
            self._rows[name] = (tick, picker)
        if names:
            self._pending = None
        self.changed.emit()

    def classes(self) -> list[str]:
        return list(self._rows)

    def _label(self, name: str) -> str:
        return class_names.label(name, self._labels)

    def value(self) -> dict[str, str]:
        """``{class: set}`` for every ticked class with a set chosen."""
        return {
            name: picker.set_id()
            for name, (tick, picker) in self._rows.items()
            if tick.isChecked() and picker.set_id() is not None
        }

    def problem(self) -> str | None:
        """Why the map is not runnable yet, or None."""
        ticked = [n for n, (tick, _p) in self._rows.items() if tick.isChecked()]
        if len(ticked) < 2:
            return "Tick two classes or more, each with its own example set"
        missing = [n for n in ticked if self._rows[n][1].set_id() is None]
        if missing:
            return (
                "Choose an example set for "
                f"{', '.join(self._label(n) for n in missing)}"
            )
        return None

    def load_classes(self, names) -> None:
        """Tick a previous run's classes (a setting); their sets are selections and are
        asked again, as the single *Examples* picker's is. Applied to the rows when the
        classes are read, if they have not been yet."""
        self._pending = set(names)
        if self._rows:
            for name, (tick, _picker) in self._rows.items():
                tick.setChecked(name in self._pending)
            self._pending = None
