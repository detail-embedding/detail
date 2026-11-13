"""The clip window's cluster sections, under the labelling panel: **what sets this cluster
apart**, and **compare** — the two questions the clips alone answer slowly.

**What sets it apart** draws the cluster page's own measurement strips (``plots.DescriptorStrip``)
for the cluster's few most distinctive measurements, each with this clip's value as a line, so
"is this one typical?" is read off the strip rather than guessed (``api.clip_fingerprint``,
measured off the UI thread with its progress shown, the clip's line moved without re-measuring).

**Compare** loops a clip of the cluster's sibling in the tree beside the labelling — most
labelling decisions are "what makes these two different" — and opens the cluster's nine most
typical clips as a grid of stills, any of which the window can switch to.

Everything here reaches the run through the opener's ``ClusterLabelling``; nothing reads disk on
the UI thread.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.gui.dialog import Dialog


class ClipFingerprintPanel(QWidget):
    """The cluster's most distinctive measurements, the clip on screen marked on each."""

    def __init__(self, context, clip, parent=None) -> None:
        super().__init__(parent)
        self._context = context
        self._clip = clip
        self._names: list[str] = []
        self._strips: dict = {}
        self._generation = 0
        column = QVBoxLayout(self)
        column.setContentsMargins(6, 0, 0, 0)
        column.addWidget(QLabel("<b>What sets this cluster apart</b>"))
        self._status = QLabel("Measuring the cluster…")
        self._status.setObjectName("clip_fingerprint_status")
        self._status.setWordWrap(True)
        column.addWidget(self._status)
        self._rows = QVBoxLayout()
        column.addLayout(self._rows)
        self._caption = QLabel("")
        self._caption.setWordWrap(True)
        column.addWidget(self._caption)
        self._measure()

    def _measure(self) -> None:
        from detail.gui.workers import StageProgressWorker

        clip = self._clip
        worker = StageProgressWorker(
            lambda emit: self._context.fingerprint(clip, emit),  # off-thread
            receiver=self,
        )
        worker.tick.connect(lambda _permille, status: self._status.setText(status))
        worker.done.connect(self._show)
        worker.failed.connect(
            lambda exc: self._status.setText(f"Could not measure the cluster: {exc}")
        )
        # Held: unlike `BackgroundCall`, this worker does not root itself, and a QThread
        # collected while it runs takes the process down with it.
        self._worker = worker
        worker.start()

    def _show(self, fingerprint) -> None:
        from detail.gui.plots import DescriptorStrip

        if not fingerprint.rows:
            self._status.setText(
                "Nothing measurable sets this cluster apart from its animals' recordings."
            )
            return
        self._status.setText("")
        self._names = [row.name for row in fingerprint.rows]
        for row in fingerprint.rows:
            strip = DescriptorStrip(row.label, row.unit)
            strip.setObjectName(f"clip_fingerprint_{row.name}")
            strip.show_values(row.baseline, row.cluster)
            strip.mark_value(row.clip)
            strip.setToolTip(_direction(row))
            self._strips[row.name] = strip
            self._rows.addWidget(strip)
        self._caption.setText(
            f"Grey: {fingerprint.n_baseline} stretches of the same animals' recordings. "
            f"Dots: this cluster's {fingerprint.n_members} most typical clips, from "
            f"{fingerprint.n_units} recordings. Blue line: the clip on screen. Each measured "
            "over at least half a second."
        )

    def set_clip(self, clip) -> None:
        """Move the blue line to ``clip``, reading only its own track (already cached)."""
        from detail.gui.workers import BackgroundCall

        self._clip = clip
        if not self._names:
            return
        self._generation += 1
        generation = self._generation
        names = list(self._names)
        call = BackgroundCall(
            lambda: self._context.clip_measures(clip, names),  # off-thread
            receiver=self,
        )
        call.done.connect(lambda values: self._mark(generation, values))
        call.start()

    def _mark(self, generation: int, values: dict) -> None:
        if generation != self._generation:
            return
        for name, strip in self._strips.items():
            strip.mark_value(values.get(name))

    def strips(self) -> dict:
        return dict(self._strips)


def _direction(row) -> str:
    side = "higher" if row.delta > 0 else "lower"
    return (
        f"{row.label}: this cluster is {side} than its animals' recordings "
        f"(Cliff's delta {row.delta:+.2f}; its median is above {row.percentile:.0%} of them)."
    )


class ClipComparePanel(QWidget):
    """The sibling cluster, looping; and the cluster's nine most typical clips as a grid.
    ``open_clip`` fires with a clip picked in the grid, for the window to show."""

    open_clip = pyqtSignal(object)

    def __init__(self, context, parent=None) -> None:
        super().__init__(parent)
        self._context = context
        self._sibling = context.sibling()
        self._index = 0
        column = QVBoxLayout(self)
        column.setContentsMargins(6, 0, 0, 0)
        column.addWidget(QLabel("<b>Compare</b>"))
        self._player = None
        if self._sibling is None or not self._sibling[1]:
            column.addWidget(
                QLabel("This cluster has no sibling with clips to compare.")
            )
        else:
            from detail.gui.behavior_query import _ClipPlayer  # local: import cycle

            title, clips = self._sibling
            self._title = QLabel("")
            self._title.setWordWrap(True)
            column.addWidget(self._title)
            self._player = _ClipPlayer(
                context.experiment(), autoplay=True, loop=True, context_s=0
            )
            self._player.setObjectName("clip_sibling_player")
            # A reference beside the clip being judged, not a second clip to study: capped so
            # the column's other sections stay in view.
            from detail.gui.sizing import lines

            self._player.setMaximumHeight(lines(16))
            column.addWidget(self._player)
            another = QPushButton("Another clip of it")
            another.setObjectName("clip_sibling_next")
            another.setToolTip(f"Show the next most typical clip of {title}.")
            another.clicked.connect(lambda: self._show_sibling(1))
            column.addWidget(another)
            self._show_sibling(0)
        row = QHBoxLayout()
        grid = QPushButton("Grid of 9 clips…")
        grid.setObjectName("clip_grid_open")
        grid.setToolTip(
            "This cluster's nine most typical clips side by side; click one to watch it here."
        )
        grid.clicked.connect(self._open_grid)
        row.addWidget(grid)
        # Offered only where the page wired one: the cue follows the capability.
        open_profile = getattr(context, "open_profile", None)
        if callable(open_profile):
            profile = QPushButton("Cluster profile…")
            profile.setObjectName("clip_profile_open")
            profile.setToolTip(
                "Which measures set this cluster apart, checked on recordings the model has "
                "not seen, with plain rules and context."
            )
            profile.clicked.connect(lambda: open_profile(self.window()))
            row.addWidget(profile)
        row.addStretch(1)
        column.addLayout(row)

    def _show_sibling(self, by: int) -> None:
        title, clips = self._sibling
        self._index = (self._index + by) % len(clips)
        self._title.setText(
            f"Its sibling, {title}: clip {self._index + 1} of {len(clips)}, looping"
        )
        self._player.set_clip(clips[self._index])

    def sibling_player(self):
        return self._player

    def _open_grid(self) -> None:
        dialog = MemberGridDialog(self._context, self)
        dialog.picked.connect(self.open_clip)
        dialog.exec()


class MemberGridDialog(Dialog):
    """The cluster's nine most typical clips as stills; clicking one picks it."""

    picked = pyqtSignal(object)

    def __init__(self, context, parent=None) -> None:
        super().__init__(parent)
        from detail.gui.clip_grid import ClipGrid

        self.setWindowTitle(f"{context.title()}: most typical clips")
        layout = self.body()
        self.grid = ClipGrid(context.experiment(), n_cells=9, columns=3)
        self.grid.clip_clicked.connect(self._pick)
        layout.addWidget(self.grid)
        self.grid.show_clips(
            context.member_clips(9), empty_text="This cluster has no clips listed."
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

    def _pick(self, clip) -> None:
        self.picked.emit(clip)
        self.accept()
