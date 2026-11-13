"""The training curve: mAP50 per epoch, and the best score so far.

**The question this answers, which the page could not.** A user wants to stop a training run
when the metrics look good. The model page showed one number — the final mAP50 — and a bar counting
epochs. Neither says whether the score is *still climbing*: a run that has been improving for an
hour and a run that plateaued forty epochs ago render identically, so the only way to decide was
to let it finish, which is the decision being avoided.

**Why a curve and a sentence, and not a verdict.** Three shapes were on the table: a plain
curve, a curve plus "best so far: 0.847 at epoch 31", and a curve plus a plateau hint ("no
improvement in 12 epochs"). The middle one, because the first makes you read a peak off pixels
and the third is the tool guessing — a plateau that resolves two epochs later reads as a bug,
and the judgment about a particular dataset is the user's. The sentence states a fact; the shape
of the curve is the argument.

The data is `Experiment.model_metric_history`, which reads the trainer's own on-disk record.
Three consequences, all wanted: the curve is whole on a page opened after the run started, a
model that trained overnight has its curve in the morning, and this widget holds no accumulated
state that could disagree with the file.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from detail.core.training_metrics import best_epoch, best_line
from detail.gui import plot_theme
from detail.gui.collapse import CollapsibleBand
from detail.gui.plots import bound_axes
from detail.gui.sizing import lines

#: How tall the plot is, in lines of the application font (`gui/sizing.py`'s `em`).
#:
#: Not a pixel constant: a number picked on this screen is correct on this screen, and it cannot
#: follow a user who has asked for larger text — the box stays put while the words grow. Ranged
#: rather than fixed, so a squeezed pane shrinks the data area and keeps the axis (a
#: `setFixedHeight` plot crops, and what crops is the x-axis and its labels).
PLOT_LINES = 14

#: mAP is a fraction. Saying so stops the view scrolling to coordinates the quantity cannot take
#: and then labelling them — `bound_axes`' whole job.
MAP_DOMAIN = (0.0, 1.0)


class TrainingCurve(QWidget):
    """A mAP50-per-epoch plot with the best-so-far line under it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self._plot = plot_theme.themed_plot()
        # Three numbers, three jobs — the same split `fit_table_height` documents, because a
        # pyqtgraph `PlotWidget` has the same problem a `QTableView` does at the other extreme:
        # its sizeHint is a constant **600x480** and its policy is `Expanding`, so left alone it
        # asks for 480px and then takes the page's slack on top. Measured: 455px of plot on a
        # 900px window, which pushed *Stop and keep* — the button the curve exists to inform —
        # below the fold.
        #
        # The **minimum** is the floor it may be squeezed to; the **hint** is what it draws at
        # (assigned on the instance, which is what PyQt's virtual dispatch reads); `Preferred`
        # stops it competing with the page's trailing stretch. No maximum, and no
        # `setFixedHeight`: one number acting as floor and ceiling crops the x-axis, and the
        # bottom of a plot is its axis.
        height = lines(PLOT_LINES)
        self._plot.setMinimumHeight(height)
        self._plot.sizeHint = lambda: QSize(600, height)
        self._plot.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._plot.getAxis("left").setLabel("mAP50")
        self._plot.getAxis("bottom").setLabel("epoch")
        box.addWidget(self._plot)

        self._caption = QLabel("(no completed epoch yet)")
        self._caption.setWordWrap(True)
        box.addWidget(self._caption)
        self._history: list[dict] = []

    def set_history(
        self, history: list[dict], *, total_epochs: int | None = None
    ) -> None:
        """Redraw from a history. Idempotent — the same history twice draws one curve.

        Rebuilt rather than appended to, and the caption is **set** rather than added to: a
        caption composed by appending to the widget's own text prints twice on a second delivery
        for one selection, and no test that re-selects ever sees it (this package has had that
        one).
        """
        self._history = list(history)
        self._plot.clear()
        points = [
            (row["epoch"], row["box_map50"])
            for row in self._history
            if row.get("box_map50") is not None and row.get("epoch") is not None
        ]
        self._caption.setText(best_line(self._history, total_epochs=total_epochs))
        if not points:
            bound_axes(self._plot, x=(0, max(total_epochs or 1, 1)), y=MAP_DOMAIN)
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        theme = plot_theme.current()
        self._plot.plot(xs, ys, pen=plot_theme.pen(theme.identity(0)))
        best = best_epoch(self._history)
        if best is not None:
            # The peak is marked on the curve as well as named in the caption, because the
            # sentence says *where* it is and the eye still has to find it. An ink ring, not a
            # second hue: it annotates the curve rather than being another series on it.
            self._plot.plot(
                [best[0]],
                [best[1]],
                pen=None,
                symbol="o",
                symbolSize=plot_theme.MARKS.marker_size + 3,
                symbolBrush=plot_theme.brush(theme.surface),
                symbolPen=plot_theme.pen(theme.summary, width=2),
            )
        # The x domain is the run, not the epochs drawn so far — otherwise the curve rescales
        # under the reader on every tick and a rising line looks flat.
        span = max(max(xs), total_epochs or 0, 1)
        bound_axes(self._plot, x=(0, span), y=MAP_DOMAIN)
        # After the items, not before: a range set before them did not survive a splitter drag.
        self._plot.setXRange(0, span)
        self._plot.setYRange(0, 1)
        self._plot.getViewBox().disableAutoRange()


class TrainingCurveBand(CollapsibleBand):
    """The curve as a foldable section, remembered per user.

    Open by default while training is the thing on screen: a page that opens having hidden its
    own evidence is worse than one that scrolls.
    """

    def __init__(self, parent=None) -> None:
        self.curve = TrainingCurve()
        super().__init__("Training curve", self.curve, key="model/curve", open_=True)

    def set_history(
        self, history: list[dict], *, total_epochs: int | None = None
    ) -> None:
        self.curve.set_history(history, total_epochs=total_epochs)
