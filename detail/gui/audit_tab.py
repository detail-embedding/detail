"""The §19 reliability audit, on screen — blind review, and the precision it buys.

The one quality signal these modules have. Without it a run says how many instances a cutoff
returns and nothing about whether they are right, and a behavior the method works well on looks
exactly like one it does not.

**The query run's AUDIT only.** It asks "how precise is this behavior at this cutoff", one flat
behavior at a time, scoped by the workspace's own selector. Clustering's audit is
``gui/label_check.py``: its labels nest and are judged per label and per cluster, which is a
different screen.

**Review happens here, not in a dialog.** The first version put it in a popup, which hid the
statistics behind the thing that changes them and never said which behavior was being judged.
Judging is the work; the numbers are why. They belong on one screen.

**Blind, and structurally so.** The score is not built into the review pane at all until a
decision is recorded — not built and hidden. Seeing 0.95 pulls a judgment toward accepting and
inflates precision by exactly the quantity the audit exists to measure, it fails silently because
a biased estimate is still a plausible number, and a hidden widget is one ``setVisible(True)``
away from bringing it back.

**A number that is not supported is not shown.** Below the reporting floor a row says how many
were judged and refuses to estimate. 1.00 from n=3 renders identically to 1.00 from n=50, and the
whole point of the audit is to stop an unaudited-looking number reading like an audited one.

**Every figure is the run's** (``audit_precision``, ``audit_precision_by_recording``,
``audit_precision_by_score_bin``) — the number ``detail analysis audit`` prints, weighted by score
bin and leaving out clips the user's own examples overlap. This tab used to compute its own from the
storage layer, which is how it could print a different number from the CLI for one audit.

A leaf: it imports ``behavior_query`` for the player and the storage layer for the audit log,
and nothing from ``gui.pages`` or ``gui.modules``. That is also why the tables here are built by
hand rather than through ``pages._table`` — importing a hub from a leaf is how both of this
package's shipped import cycles formed (``tests/gui/test_import_cycles.py``).
"""

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QKeySequence,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import numpy as np
import pyqtgraph as pg

from detail.core import run_selection
from detail.gui.behavior_query import ClipSpec, _ClipPlayer
from detail.gui.busy import run_busy
from detail.gui import plot_theme
from detail.gui.grip_splitter import GripSplitter
from detail.gui.plots import bound_axes
from detail.gui.sizing import emphasis_font, lines
from detail.gui.status_color import MUTED, set_label_color
from detail.gui.tables import (
    BROWSE_ROWS,
    fill_width,
    fit_table_height,
    give_slack_to,
    list_table,
)
from detail.storage import audit_figures
from detail.storage import behavior_audit as audit
from detail.storage import behavior_output as bo

#: What one sitting costs. A batch, never a cap — "review more" is always one click away, and
#: the measured reason is the interval: one batch is directional (n~17 above a typical cutoff,
#: 0.66-0.97 at an observed 0.90) and two are reportable (n~35, 0.78-0.97).
BATCH = audit.DEFAULT_BUDGET

#: The walks the sidebar offers, in the order they are shown. `this batch` exists only while
#: one does; `unclear` is greyed with its count rather than hidden when empty, because the
#: count is the point — an unclear is dropped from the estimate, so each one settled is a row
#: the estimate gets back.
SCOPES = (
    ("batch", "this batch"),
    ("judged", "judged"),
    ("unclear", "unclear"),
)

#: The judgment keys. A 25-clip pass should be 25 keystrokes, not 25 trips to the mouse.
#: Deliberately not ``Space``: the player toggles play on a click, and a key that both plays and
#: judges is a key that judges by mistake.
#: Home row under the left hand, in the order the buttons sit, so the right hand can stay on
#: the mouse. `Y`/`N`/`U` read better and type worse: they are three separate reaches, and a
#: 25-clip pass is 25 of them.
KEYS = (("Yes", "S", True), ("No", "D", False), ("Unclear", "F", None))

#: Height of the precision panel, as a **range** rather than a fixed number. The reason it was
#: pinned still holds — a panel that resizes as numbers arrive moves everything under it — but a
#: fixed height is a floor and a ceiling at once, so a pane shorter than the number cannot squeeze
#: it and crops it instead, taking the axis with it. A range holds still until the window is
#: genuinely too short, and then shrinks rather than being cut.
#:
#: In lines of the interface font (`sizing.lines`), so scaled text gets a taller panel instead of
#: bigger labels in the same box.
#: Stored as line *counts*, not pixels: `sizing.lines` reads the application font, and there is
#: no application at import time. Resolved where the widget is built, which also means a size
#: follows the font in force when the panel is actually made rather than when the module loaded.
_AXIS_PX = 58  #: Left-axis width, shared so both plots' areas start at the same x.
_CURVE_MIN_LINES = 7
_CURVE_MAX_LINES = 12
#: The judged-count strip under the curve. Short on purpose — it answers "where did I look",
#: which is a shape, not a quantity anyone reads a value off.
#:
#: Short, but not *this* short: pyqtgraph's axis chrome is a constant ~40px, so at three lines
#: (48px) the strip had 8px of drawing in it and its rotated "judged" label was clipped against
#: the frame. Five lines leaves 40px of bars, which is what a shape needs to be a shape.
_STRIP_MIN_LINES = 5
_STRIP_MAX_LINES = 7

#: How tall the judging block — the picture and its transport — may grow.
#:
#: A **ceiling**, not a size: `_block_height` asks the picture what it needs at the width it has
#: and this is only where that stops. `PictureWithControls` derives the picture's height from the
#: width it is handed, so without a ceiling a review pane 1122px wide drew the clip 815px tall and
#: pushed the three buttons that act on it 860px below its top edge, against the bottom of a
#: 1080px screen. Nothing was hidden and nothing failed; the video and the decision about it had
#: simply stopped being one object.
#:
#: 34 lines is ~560px at the interface font. Chosen against the rows below it rather than against
#: the clip: the block plus the line, the question, the judgment row and the nav row is ~730px,
#: which fits the content area of a maximized window on a 1080p display with the transport, the
#: question and all three buttons on screen. Past the ceiling the picture is drawn at its own
#: aspect ratio inside a wider block, which is the same letterboxing any player does.
#:
#: In lines rather than pixels so scaled text gets a taller block (`sizing.lines`), and stored as
#: a count because there is no application at import time.
_VIDEO_MAX_LINES = 34


def format_precision(estimate: dict | None) -> str:
    """One estimate as the line a person reads, including the two ways it declines to be one.

    ``None`` means nothing has been audited; ``below_min_n`` means something has and it is not
    enough. Those are different states and they read differently — "not audited" is work
    outstanding, "4 judged, too few" is work done that does not support a number.
    """
    if estimate is None or not estimate.get("n"):
        return "not audited"
    if estimate.get("reason"):
        # A third state, and it must come before the numeric branch: a coverage refusal
        # carries `precision: None` while `below_min_n` is False, so this used to fall through
        # and raise `TypeError` formatting None — on the UI thread, where the excepthook
        # reports and carries on, leaving the tab half-drawn rather than erroring.
        pct = round(100 * (estimate.get("coverage") or 0))
        return f"— {pct}% of the audit still applies, too little to report"
    if estimate.get("below_min_n"):
        return f"— {estimate['n']} judged, too few to estimate"
    low, high = estimate["interval"]
    return f"{estimate['precision']:.2f}  ({low:.2f}–{high:.2f}, n={estimate['n']})"


def _fix_axes(plot) -> None:
    """Both plots' areas aligned.

    The left axis is given a fixed width, or the two plots' plotting areas start at different x
    and a bar no longer sits under its own point. (The SI-prefix rewrite is off on every themed
    plot already — ``plot_theme.apply``.)
    """
    plot.getAxis("left").setWidth(_AXIS_PX)
    # The plot's ground ran flush to the pane on every side. A margin inside the widget keeps
    # the frame off the panel edge and the two plots off each other.
    plot.setContentsMargins(2, 6, 8, 2)


class _VideoBlock(QStackedWidget):
    """The clip and its transport, as tall as the picture needs at the width it is given.

    A widget rather than a `sizeHint` assigned on a stack, and that is the third mechanism
    tried here — the first two are worth knowing because both *looked* right:

    - **A constant height** (a `sizeHint` returning the ceiling) reserves the same block
      whatever the width. `PictureWithControls` draws the picture at its own aspect ratio
      inside whatever it is handed, so on a wide pane the picture came up short and `top=True`
      gathered the difference into one gap, directly under the transport where the buttons go.
    - **A `sizeHint` that reads the widget's own width** is right on paper and stale in
      practice: `QWidgetItemV2` caches an item's hint and nothing invalidates it when only the
      *width* changes, so the layout kept answering from whatever width it last saw. Measured
      in the query workspace: Python read 466px, the layout laid the block out at 531, and the
      71px difference was the gap — in the one workspace wide enough to expose it.

    Qt's own `heightForWidth` is the third, and it does not work **here**: `QWidgetItem` asks
    a widget's *layout* for it whenever the widget has one, and a `QStackedWidget` always has
    one (`QStackedLayout`, which answers for whichever page is current and knows nothing of
    this rule). The override was never consulted — measured, the block took its 360px minimum
    and the clip shrank instead of the gap closing.

    So the hint carries the rule, and the widget **invalidates it itself** when its width
    changes. That is the one thing the stale version was missing, and a `resizeEvent` is where
    a widget is told. Height cannot feed back into width in a vertical layout, so this settles
    after one extra pass rather than oscillating.
    """

    def __init__(self, height_for_width, ceiling: int) -> None:
        super().__init__()
        self._height_for_width = height_for_width
        self._ceiling = ceiling
        self.setMaximumHeight(ceiling)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt virtual
        width = self.width()
        height = (
            self._ceiling
            if width <= 0
            else min(self._ceiling, self._height_for_width(width))
        )
        return QSize(super().sizeHint().width(), height)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt virtual
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            # `QWidgetItemV2` caches an item's hint and invalidates it on `invalidate()`, which
            # nothing calls for a width-only change — so without this the layout keeps sizing
            # the block from whatever width it last saw.
            self.updateGeometry()


class AuditTab(QWidget):
    """Review on the left, what the reviewing bought on the right."""

    changed = pyqtSignal()
    #: A row clicked in the by-label table. The tab is a leaf and does not know the workspace,
    #: so it asks rather than reaching for the behavior selector itself.
    behavior_requested = pyqtSignal(str)

    def __init__(self, experiment, run) -> None:
        super().__init__()
        self._experiment = experiment
        self._run = run
        #: Whether the run is finished, read off-thread with the rest of the audit. ``None``
        #: until that read lands, which counts as read-only: live before it, a finished run's
        #: Start was a button the API refuses.
        self._finished: bool | None = None
        self._label: str | None = None
        self._queue: list = []
        self._index = 0
        # This session's answers, so a revisited clip can show what you said. The log is the
        # record; this is only what the buttons need to mark.
        self._decisions: dict = {}
        #: The clips drawn by the last batch, kept so `this batch` stays a scope you can
        #: return to after wandering off into what you judged earlier.
        self._batch: list = []
        #: `(instance, said)` for every clip this run has a standing decision on.
        self._judged: list = []
        self._pending = 0
        self._remaining = 0
        #: The last completed batch's tally, shown beside the clip counter until a walk starts.
        self._tally = ""
        #: The list and the walk drive each other, so each must not re-drive the other.
        self._syncing = False
        #: Which clip the player is already showing. `set_clip` restarts autoplay, so
        #: re-setting the clip a refresh happens to land on undoes a pause and re-decodes a
        #: frame that is already on screen.
        self._shown: str | None = None
        #: `_audit_pool` re-enumerates candidates at cutoff zero, and does not change while a
        #: run is authored — so it is read once per behavior rather than on every refresh.
        self._pool_cache: tuple | None = None
        #: ``{recording_id: name}``. A recording's id is the tool's and its name is the user's
        #: (§11), and this surface showed three ids: the clip rows, the line under the video and
        #: the by-recording table. Read with the artifact in `_compute`, because the map is one
        #: read per recording and this runs on the UI thread otherwise.
        self._names: dict[str, str] = {}
        #: `_vocabulary()`'s answer, read by `_compute` and kept by `_render` — beside
        #: `_pool_cache`, and for the same reason: a read of the run, not of the slice on screen.
        self._vocab: dict[str, str] | None = None

        root = QVBoxLayout(self)
        self._headline = QLabel()
        self._headline.setFont(emphasis_font(self._headline))
        root.addWidget(self._headline)
        # What the number is, beside the number. "Precision" alone is read as accuracy by
        # exactly the person this tab exists for, and the missing half is not an oversight:
        # recall would need the instances the run *didn't* find, which is a different and much
        # larger sample than the 25 clips this asks for. Saying so is cheaper than being
        # misread, and stops "0.9 precision" being quoted as "90% of the rearing".
        self._what = QLabel(
            "Of the instances above your cutoff, the share that really are the behavior. "
            "It does not say how many the run missed — that would be recall, which this "
            "audit deliberately does not estimate."
        )
        self._what.setWordWrap(True)
        set_label_color(self._what, MUTED)
        root.addWidget(self._what)
        #: What qualifies the headline figure — the shared wording the CLI prints beside it
        #: (`audit_figures.notes`): too few audited, bins not all audited yet, clips drawn before
        #: the bins were recorded. Hidden when there is nothing to say.
        self._qualifiers = QLabel()
        self._qualifiers.setWordWrap(True)
        set_label_color(self._qualifiers, MUTED)
        self._qualifiers.setVisible(False)
        root.addWidget(self._qualifiers)

        # Two modes, not one screen with everything empty in it. Before anything is reviewed
        # there is exactly one useful action, and the three-column workspace around it is four
        # regions describing their own absence: a clip list with nothing in it, a prompt adrift
        # in a large void, two tables reading "not audited", and a pair of empty axes. So the
        # workspace arrives with the data it is for.
        self._body = QStackedWidget()
        self._body.addWidget(self._build_intro())

        # The clip list and the summary fold by their grips to strips reading "Clips" and
        # "Summary"; the clip being judged does not.
        split = GripSplitter(Qt.Orientation.Horizontal, key="query_audit")
        split.addWidget(self._build_sidebar())
        split.addWidget(self._build_review())
        split.addWidget(self._build_stats())
        split.set_foldable(0, "Clips")
        split.set_foldable(2, "Summary")
        # The panel column is a **summary**, and it opened at 520px of which most was the
        # precision-by-confidence plot — ten bins and two short tables holding a pane wider than
        # the clip being judged. Narrower here, and still draggable; the walk list takes some of
        # what it gives up, because a recording's *name* is longer than the id it replaced.
        restored = split.remember()
        if not restored:
            split.setSizes([300, 700, 400])
        # **Every pane gives ground, in proportion.** A splitter takes a deficit the same way it
        # hands out a surplus — by stretch factor — so `0 / 1 / 0`, which was here to send extra
        # width to the clip, also sent the *whole* shortfall to it. On a 1397px window the two
        # side columns kept their full width and the clip's pane was left with 357px: a 268px
        # video, and a transport row squeezed until "Prev frame" read "ev fram". The clip still
        # gets the largest share at every size; what it no longer does is absorb the loss alone.
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 4)
        split.setStretchFactor(2, 2)
        self._body.addWidget(split)
        # The stack is the page's fill element, so no trailing stretch (GUI style).
        root.addWidget(self._body, 1)
        self.reload()

    # -- construction --------------------------------------------------------- #
    def _build_intro(self) -> QWidget:
        """What the page is before anything has been reviewed: one sentence and one button.

        Precision is the only thing this tab produces and it needs a sample, so until there is
        one there is nothing to show and one thing to do. Saying how long it takes is the part
        that matters — the reason people do not start is not that the button is hard to find.
        """
        pane = QWidget()
        column = QVBoxLayout(pane)
        # **No leading stretch.** Centring it vertically was the intent — one sentence and one
        # button in a large pane — and it is wrong here for a reason that only shows up in the
        # app: a tab inside a `QTabWidget` is as tall as the *tallest* sibling tab, and this
        # workspace's other tabs are tall. Centred in that, the only action on the page sat
        # hundreds of pixels down, below the fold of every workspace that embeds this — pose,
        # temporal and query alike. `gui_design` §3 top-justifies a pane for exactly this, and
        # `assert_top_justified` cannot see it: a leading stretch absorbs the slack correctly,
        # so no label spreads and the check it was written for passes.
        self._intro_headline = QLabel()
        self._intro_headline.setFont(emphasis_font(self._intro_headline))
        self._intro_headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._intro_headline)

        self._intro_body = QLabel()
        self._intro_body.setWordWrap(True)
        self._intro_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._intro_body)
        column.addSpacing(12)

        row = QHBoxLayout()
        row.addStretch()
        self._intro_start = QPushButton()
        # Disabled until `_refresh_sidebar` has read the run and decided. A button that is live
        # before anything has been read is live on a snapshot too, which is what the walk over
        # enabled controls found — and it found it by *label*, so an unlabelled one is the case
        # that slips through (`tests/gui/test_query_workspace.py`).
        self._intro_start.setEnabled(False)
        self._intro_start.clicked.connect(lambda: self._start_batch())
        row.addWidget(self._intro_start)
        row.addStretch()
        column.addLayout(row)

        self._intro_note = QLabel()
        self._intro_note.setWordWrap(True)
        self._intro_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_label_color(self._intro_note, MUTED)
        column.addWidget(self._intro_note)
        column.addStretch()
        return pane

    def _sync_mode(self) -> None:
        """Show the workspace once there is something in it, and not before.

        Keyed on "has anything ever been judged, or is a batch drawn" rather than on the
        current scope: filtering to ``unclear (0)`` after an audit is an empty *view*, not an
        empty tab, and it keeps the workspace and says so in the review pane as it always did.
        """
        started = bool(self._judged) or bool(self._queue)
        self._body.setCurrentIndex(1 if started else 0)
        self._headline.setVisible(started)

    def _build_sidebar(self) -> QWidget:
        """What to walk, and how to get more of it.

        One concept, in one column. Before this the tab had two — a batch you walked and a
        summary you returned to — so opening an earlier judgment replaced the whole pane and
        left no way back but pressing `→` seventy-five times. Here a scope sets the walk and a
        row sets the position, the buttons that draw more sit above them, and there is nothing
        to escape from because you never left.
        """
        pane = QWidget()
        column = QVBoxLayout(pane)
        column.setSpacing(4)
        self._start = QPushButton()
        self._start.clicked.connect(lambda: self._start_batch())
        if not self._editable():
            self._start.setEnabled(False)
            self._start.setToolTip(self._read_only_reason())
        column.addWidget(self._start)
        self._continue = QPushButton()
        self._continue.clicked.connect(self._continue_batch)
        if not self._editable():
            self._continue.setEnabled(False)
            self._continue.setToolTip(self._read_only_reason())
        column.addWidget(self._continue)
        self._exhausted = QLabel()
        self._exhausted.setWordWrap(True)
        column.addWidget(self._exhausted)

        column.addSpacing(12)
        column.addWidget(QLabel("WHAT TO REVIEW"))
        self._scopes = {}
        for scope, text in SCOPES:
            button = QRadioButton(text)
            button.toggled.connect(
                lambda checked, s=scope: self._on_scope(s) if checked else None
            )
            column.addWidget(button)
            self._scopes[scope] = button
        # Guarded: `toggled` fires synchronously, and `_on_scope` reaches for the list and
        # the player — neither of which is built yet. PyQt reports an exception escaping a
        # slot and carries on, so without this the tab still *constructs*, with a sidebar
        # that was never filled and a traceback nobody reads.
        self._syncing = True
        try:
            self._scopes["judged"].setChecked(True)
        finally:
            self._syncing = False

        self._list_model = QStandardItemModel()
        # Which column takes the slack is set after each fill, not here — see `_fill_list`.
        self._list = list_table()
        self._list.setModel(self._list_model)
        # Selection, not double-click: the row does not *open* a page, it says which clip to
        # look at — the same job `_BrowseTab` does beside its own player, and the same idiom,
        # so a single click and the keyboard both work.
        self._list.selectionModel().currentRowChanged.connect(self._on_row)
        # Sized to `BROWSE_ROWS` and then scrolling, at the **top** of the pane with the rest
        # left empty — the shape both Browse tabs settled on, for the same reason. A splitter
        # hands each child its whole height whatever it prefers, so this filled: twelve judged
        # clips drew as twenty-eight rows of grid down an 856px column while the player beside
        # it centred its video against the middle of that. Capping the list's *maximum* is the
        # wrong fix and was tried there — a splitter is only as tall as its shortest-capped
        # child, so the whole row sinks to the bottom of the tab.
        #
        # Not `tables.top_of_pane`: this pane is a column of its own with the draw buttons and
        # the scope radios above the list, so the stretch goes at the end of that column rather
        # than into a wrapper around the table.
        #
        # Fitted **here**, on the empty table, as well as on every refill: the hint is a method
        # assigned on the instance and sip stops looking for a Python override once C++ has
        # asked, so a table first sized after it has been laid out never takes the size.
        fit_table_height(self._list, max_rows=BROWSE_ROWS, min_rows=0)
        column.addWidget(self._list)
        column.addStretch(1)
        return pane

    def _build_review(self) -> QWidget:
        """The clip and the decision about it, packed at the top of the pane.

        The clip takes the **whole** width of its pane; the rows under it are centred on that
        width. A capped, centred column was tried and rejected — a video that stops short of
        the room it has been given reads as a video that could not fill it.
        """
        pane = QWidget()
        column = QVBoxLayout(pane)
        self._bar = QProgressBar()
        self._bar.setRange(0, BATCH)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)  # a rule, not a widget: it reports, it does not ask
        column.addWidget(self._bar)
        self._progress = QLabel()
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._progress)

        # The prompt occupies the same space the video will, so an audit with nothing in it
        # yet reads as "nothing here" rather than as a broken pane. A stack rather than an
        # overlay so neither has to know the other's geometry.
        self._stack = _VideoBlock(self._picture_height, lines(_VIDEO_MAX_LINES))
        self._prompt = QLabel()
        self._prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prompt.setWordWrap(True)
        placeholder = QWidget()
        prompt_column = QVBoxLayout(placeholder)
        prompt_column.addStretch()
        prompt_column.addWidget(self._prompt)
        prompt_column.addStretch()
        # Not looping. A clip that restarts on its own keeps pulling the eye back while you
        # are deciding, and the decision here is a single yes/no about what you just saw —
        # the replay is the ▶ button, if you want one. The cluster strip settled on the
        # same answer for the same reason.
        # `top=True`: the picture sits at the top of whatever the block is given rather than
        # centred in it, so the transport is directly under the frame at every pane height.
        self._player = _ClipPlayer(
            self._experiment, autoplay=True, loop=False, context_s=1, top=True
        )
        self._stack.addWidget(placeholder)
        self._stack.addWidget(self._player)
        # What the block reserves, and it is a **question about its own width**, not a constant.
        #
        # A stretch is not the answer: `QBoxLayout` splits slack by stretch factor, so a stack
        # sharing one with the trailing spacer took half of whatever was going and the block
        # grew 42px for every 700px of window. A constant is not the answer either, and that
        # was the first fix here — `PictureWithControls` draws the picture at its own aspect
        # ratio inside whatever it is handed, so a block 495px tall around a picture that only
        # needs 268 leaves 227px of nothing between the video and the buttons that judge it.
        # Measured on a 357px-wide pane; reported as the buttons dropping down on resize.
        #
        # So: the height the picture needs at the width this block actually has, capped. Height
        # from width is not a feedback loop — a vertical layout takes the width from the parent
        # and never from this hint — which is the same reasoning `PictureWithControls._reflow`
        # already runs on. Assigned before the stack is ever laid out: sip stops looking for a
        # Python override of a C++ virtual once C++ has asked and found none, so a hint set
        # later is invisible to the layout (`tables.fit_table_height` records the same).
        column.addWidget(self._stack)
        self._where = QLabel()
        self._where.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._where)
        self._question = QLabel()
        question_font = self._question.font()
        question_font.setBold(True)
        self._question.setFont(question_font)
        self._question.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self._question)

        # Centred under the clip they are about. Left-aligned they sat at the far edge of a
        # pane the video no longer reaches, which reads as two unrelated things on one screen.
        self._button_row = QHBoxLayout()
        self._button_row.addStretch()
        self._buttons: list[QPushButton] = []
        self._shortcuts: list[QShortcut] = []
        for text, key, value in KEYS:
            # The key is in the label as literal text, never as an `&` mnemonic: Qt eats the
            # ampersand and renders an underline instead (the recorded button-text rule).
            button = QPushButton(f"{text} ({key})")
            button.clicked.connect(lambda _c=False, v=value: self._decide(v))
            self._button_row.addWidget(button)
            self._buttons.append(button)
            # A real `QShortcut` rather than `button.setShortcut`, which is always
            # `WindowShortcut` with no way to narrow it — so `S` pressed on Calibrate, or typed
            # into a filter box, would record a human judgment about a clip nobody is looking
            # at. That is this surface's worst failure arriving by accident, so the shortcut is
            # scoped to the tab and its children.
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda v=value: self._decide(v))
            self._shortcuts.append(shortcut)
        self._button_row.addStretch()
        # Dead from the start on a snapshot, not once the off-thread load lands: a QPushButton is
        # enabled when it is built, so until `_render` ran these sat live on a read-only screen.
        # The window is small and it is exactly the window somebody clicking fast is in.
        if not self._editable():
            for button in self._buttons:
                button.setEnabled(False)
                button.setToolTip(self._read_only_reason())
        holder = QWidget()
        holder.setLayout(self._button_row)
        column.addWidget(holder)

        self._last = QLabel()
        self._last.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._last.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self._last)

        nav = QHBoxLayout()
        self._back = QPushButton("◀ back (←)")
        self._back.clicked.connect(lambda: self._step(-1))
        self._forward = QPushButton("next (→) ▶")
        self._forward.clicked.connect(lambda: self._step(1))
        nav.addWidget(self._back)
        nav.addStretch()
        nav.addWidget(self._forward)
        nav_holder = QWidget()
        nav_holder.setLayout(nav)
        column.addWidget(nav_holder)
        # The block packs at the top (`gui_design` §3): the stack above is the fill child and
        # this is where its surplus goes once it has hit its cap, rather than into the labels.
        column.addStretch(1)
        for key, delta in (("Left", -1), ("Right", 1)):
            move = QShortcut(QKeySequence(key), self)
            move.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            move.activated.connect(lambda d=delta: self._step(d))
            self._shortcuts.append(move)
        return pane

    def _picture_height(self, width: int) -> int:
        """How tall the clip and its transport need to be at ``width`` — asked of the player,
        which is the thing that knows the frame's aspect ratio and how far its row has wrapped.

        Passed to `_VideoBlock` as a function rather than read from it, so the block has no
        opinion about what it is showing and this tab has none about how Qt asks.
        """
        return self._player.height_for_width(width)

    def _build_stats(self) -> QWidget:
        pane = QWidget()
        column = QVBoxLayout(pane)
        # Tight, so a heading reads as belonging to the table under it; the group gaps below
        # do the separating. Equal spacing everywhere left each heading equidistant between
        # two tables and belonging to neither.
        column.setSpacing(4)
        self._build_label_table(column)

        column.addSpacing(12)
        self._by_recording_heading = QLabel()
        column.addWidget(self._by_recording_heading)
        self._by_recording_model = QStandardItemModel()
        self._by_recording = list_table()
        self._by_recording.setModel(self._by_recording_model)
        column.addWidget(self._by_recording)

        # Precision against confidence — the panel the Calibrate overlay should have been. It
        # was drawn there as `precision x max_instances` on the *instances* axis, with no second
        # axis and no legend, so no value could be read off it. Here it owns its axes.
        column.addSpacing(14)
        self._curve_heading = QLabel("PRECISION BY CONFIDENCE")
        column.addWidget(self._curve_heading)
        column.addSpacing(4)
        self._plot = plot_theme.themed_plot()
        self._plot.setMinimumHeight(lines(_CURVE_MIN_LINES))
        self._plot.setMaximumHeight(lines(_CURVE_MAX_LINES))
        self._plot.setLabel("left", "precision")
        self._plot.setYRange(0.0, 1.0, padding=0.05)
        self._plot.getAxis("bottom").setStyle(
            showValues=False
        )  # the strip carries the axis
        _fix_axes(self._plot)
        bound_axes(self._plot, x=(0.0, 1.0), y=(0.0, 1.0))
        # Ten bins in 150px: there is nothing here to explore, and the right-click menu's
        # "Export…" / "View All" is noise on a summary. The other plots keep their zoom.
        self._plot.setMouseEnabled(False, False)
        self._plot.setMenuEnabled(False)
        column.addWidget(self._plot)

        # How many judged clips each part of the curve rests on. The band above says how sure
        # the rate is; this says where the looking actually happened, which a tight band on a
        # bin nobody sampled cannot tell you.
        self._strip = plot_theme.themed_plot()
        self._strip.setMinimumHeight(lines(_STRIP_MIN_LINES))
        self._strip.setMaximumHeight(lines(_STRIP_MAX_LINES))
        self._strip.setLabel("bottom", "confidence")
        self._strip.setLabel("left", "judged")
        self._strip.setXLink(self._plot)  # one x, so a bar sits under its own point
        _fix_axes(self._strip)
        bound_axes(self._strip, x=(0.0, 1.0), y=(0.0, None))
        self._strip.setMouseEnabled(False, False)
        self._strip.setMenuEnabled(False)
        column.addWidget(self._strip)
        column.addStretch()
        return pane

    def _build_label_table(self, column) -> None:
        """The by-behavior table, which navigates."""
        self._by_label_heading = QLabel()
        column.addWidget(self._by_label_heading)
        self._by_label_model = QStandardItemModel()
        self._by_label = list_table()
        self._by_label.setModel(self._by_label_model)
        self._by_label.clicked.connect(self._on_label_row)
        column.addWidget(self._by_label)

    def _draw_curve(self, points: list) -> None:
        """The per-bin precision, or nothing at all.

        Absent before an audit, because a flat line at zero reads as "precision is low" rather
        than "nobody has looked".
        """
        self._plot.clear()
        self._strip.clear()
        has_points = bool(points)
        for widget in (self._plot, self._strip, self._curve_heading):
            widget.setVisible(has_points)
        if not has_points:
            return
        x = np.array(
            [(b["score_low"] + b["score_high"]) / 2.0 for b in points], dtype=float
        )
        y = np.array([b["precision"] for b in points], dtype=float)
        theme = plot_theme.current()
        series = theme.identity(0)
        self._draw_interval(points, x)
        judged = np.array([b["n"] for b in points], dtype=float)
        # Two marker classes, the same distinction `format_precision` makes in words in the
        # tables: a rate resting on 3 clips is not the same claim as one resting on 30, and
        # drawing them as identical dots is how the plot said it was.
        for solid in (True, False):
            keep = [
                i
                for i, b in enumerate(points)
                if b["n"] and bool(b["below_min_n"]) is not solid
            ]
            if not keep:
                continue
            self._plot.plot(
                x[keep],
                y[keep],
                pen=None,
                symbol="o",
                symbolSize=plot_theme.MARKS.marker_size,
                # Hollow means "below the reporting floor"; its hole is the surface, not a
                # transparency the curve would show through.
                symbolBrush=plot_theme.brush(series if solid else theme.surface),
                symbolPen=plot_theme.pen(series, width=1.5),
            )
        line = [i for i, b in enumerate(points) if b["n"]]
        if len(line) > 1:
            self._plot.plot(x[line], y[line], pen=plot_theme.pen(series))
        self._draw_strip(points, x, judged)
        # Confidence is a [0, 1] quantity, so the axis says [0, 1] whatever the bins happen
        # to cover. Cropped to the data it read as though the range began at 0.28, which
        # pinned a cutoff of 0.28 against the left edge and made "where in the range am I"
        # unanswerable — and being under 1.0 it also picked up an SI prefix, relabelling
        # confidence "300 (x0.001)".
        #
        # The cause is that **x auto-range was never turned off**: construction sets only
        # `setYRange`, so x stayed on auto from the start and every relayout re-fitted it to
        # whatever the bins covered. (Measured on pyqtgraph 0.14.0: `setXRange` disables it
        # permanently, and `plot()`, `addItem` and `clear()` do *not* turn it back on — an
        # earlier version of this comment claimed they did, and was wrong.)
        for widget in (self._plot, self._strip):
            widget.setXRange(0.0, 1.0, padding=0.02)
            widget.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
        self._plot.setYRange(0.0, 1.0, padding=0.05)
        cutoff = self._cutoff(self._label)
        if cutoff is not None:
            # The cutoff in the colour every cutoff has, so it is recognisably the line dragged
            # on the calibration curve — here only shown, not moved.
            self._plot.addItem(
                pg.InfiniteLine(
                    pos=float(cutoff),
                    angle=90,
                    movable=False,
                    pen=plot_theme.pen(theme.handle, width=1.5, dashed=True),
                )
            )

    def _draw_interval(self, points: list, x) -> None:
        """The 95% interval as a band around the curve.

        Not a new quantity — it is the Wilson interval `_estimate` already returns beside
        every rate, drawn where the claim is made. Without it a bin resting on two clips and
        one resting on thirty are the same dot at the same height, which is the failure
        `_estimate`'s own docstring names: "1.00 from n=3 renders identically to 1.00 from
        n=50 and supports nothing".
        """
        keep = [i for i, b in enumerate(points) if b["n"] and b.get("interval")]
        if len(keep) < 2:
            return
        lows = np.array([points[i]["interval"][0] for i in keep], dtype=float)
        highs = np.array([points[i]["interval"][1] for i in keep], dtype=float)
        band = pg.FillBetweenItem(
            pg.PlotDataItem(x[keep], lows),
            pg.PlotDataItem(x[keep], highs),
            brush=plot_theme.brush(
                plot_theme.current().identity(0), plot_theme.fill_alpha()
            ),
        )
        self._plot.addItem(band)

    def _draw_strip(self, points: list, x, judged) -> None:
        """How many judged clips back each bin, and a tick where none do.

        A bin nobody has judged has ``precision: None``, which reaches the curve as a NaN and
        renders as a gap — indistinguishable from nothing being there at all. The tick says
        which it is: candidates exist here, and you have not looked at any of them.
        """
        theme = plot_theme.current()
        width = float(x[1] - x[0]) * 0.8 if len(x) > 1 else 0.08
        top = float(max(judged.max(), 1.0))
        self._strip.setYRange(0.0, top, padding=0.1)
        # Two ticks in 64px. Left to itself pyqtgraph draws 0/10/20/30 into a strip barely
        # taller than the labels, and they overlap into a smear.
        self._strip.getAxis("left").setTicks([[(0, "0"), (top, f"{int(top)}")]])
        drawn = [i for i, value in enumerate(judged) if value]
        if drawn:
            self._strip.addItem(
                pg.BarGraphItem(
                    x=x[drawn],
                    height=judged[drawn],
                    width=width,
                    brush=plot_theme.brush(theme.identity(0)),
                    pen=None,
                )
            )
        empty = [i for i, b in enumerate(points) if not b["n"] and b.get("n_instances")]
        if empty:
            self._strip.plot(
                x[empty],
                np.zeros(len(empty)),
                pen=None,
                symbol="o",
                symbolSize=plot_theme.MARKS.marker_size,
                symbolBrush=plot_theme.brush(theme.surface),
                symbolPen=plot_theme.pen(theme.text_muted, width=1.5),
            )

    # -- scoping -------------------------------------------------------------- #
    def set_behavior(self, label: str | None) -> None:
        """Scope to one label, mirroring the other tabs' selector hook.

        A batch in progress is **not** discarded — its unjudged clips return to the pool and
        are offered first next time (``AuthoredRun.audit_sample``), so switching costs nothing
        and needs no confirmation. What it must not do is carry on: §19 is explicit that
        alternating "is this rearing?" with "is this grooming?" corrupts the estimate quietly.
        """
        self._label = label
        self._shown = None
        self._queue = []
        self._batch = []
        self._index = 0
        self._decisions = {}
        self._tally = ""
        self._pool_cache = None  # a different behavior is a different pool
        self.reload()

    def _on_label_row(self, index) -> None:
        item = self._by_label_model.item(index.row(), 0)
        label = item.data(Qt.ItemDataRole.UserRole) if item else None
        if label and label != self._label:
            self.behavior_requested.emit(label)
            self.set_behavior(label)

    # -- reading (off the UI thread) ------------------------------------------ #
    def reload(self) -> None:
        """Re-read the artifact and the audit, **off the UI thread**.

        `instances.csv` grows with the cohort, and this now reads every label's instances
        rather than one, so a read in a widget method freezes the window for however long it
        takes (``tests/gui/test_ui_thread_blocking.py``).
        """
        run_busy(
            self,
            self._compute,
            self._render,
            label="Reading the audit…",
            supersedes="audit",  # a behavior switch or a cutoff change reloads mid-read
        )

    def _labels(self) -> list[str]:
        state = getattr(self._run, "state", None)
        if state is not None and hasattr(state(), "behaviors"):
            return [b.id for b in state().behaviors]
        try:
            rows = self._run.artifact().instances  # off-thread: `_compute`
        except FileNotFoundError:
            return []
        return sorted({row.source_label for row in rows})

    def _pool(self) -> list:
        """The audit pool for this behavior, cached — off-thread, via `_compute`.

        **Not `artifact().instances`.** `_audit_pool` enumerates candidates at cutoff zero on
        purpose (`api/query.py`), so that one audit serves every cutoff: `instance_id` is a
        content hash, so a decision still attaches to its span after the cutoff moves and the
        artifact is rebuilt. Reading the artifact here would undo that — raise the cutoff and
        the judgments you made below it vanish from the only surface that shows them, with
        nothing failing and the count beside them agreeing with itself.

        It re-runs the method's candidate enumeration, so it is read once per behavior rather
        than on every refresh; nothing about it moves while a run is being authored.

        **Without the candidates the user's own clips overlap** (`_audit_examples`): the draw never
        offers them and no estimate counts them, so counting them here left "remaining" stuck
        above zero with nothing left to draw, and listed judgments the figure does not use.
        """
        if self._pool_cache is None or self._pool_cache[0] != self._label:
            pool = list(self._run._audit_pool(self._label))
            examples = self._run._audit_examples(pool)
            self._pool_cache = (
                self._label,
                [row for row in pool if row.instance_id not in examples],
            )
        return self._pool_cache[1]

    def _review_rows(self, pool: list) -> tuple[list, int, int]:
        """``(judged, pending, remaining)`` for this label — off-thread, from the log.

        Judged means a standing decision, unclear included (`AuditRow.decided`); a drawn but
        unanswered clip is *pending*, and one never offered counts toward *remaining*, which
        is what the draw button can honestly promise. Newest first, because "I think I got
        that one wrong" is what sends someone to the list.
        """
        decisions = {
            row.instance_id: row
            for row in audit.latest_decisions(self._run._audit_dir())
            if row.decided
        }
        offered = {row.instance_id for row in audit.read_audit(self._run._audit_dir())}
        rows = [r for r in bo.bouts(pool) if r.instance_id in decisions]
        rows.sort(key=lambda r: decisions[r.instance_id].drawn_at or "", reverse=True)
        every = bo.bouts(pool)
        pending = sum(
            1
            for r in every
            if r.instance_id in offered and r.instance_id not in decisions
        )
        remaining = sum(1 for r in every if r.instance_id not in decisions)
        return (
            [(r, decisions[r.instance_id].human_label) for r in rows],
            pending,
            remaining,
        )

    def _compute(self) -> dict:
        """Off-thread: every read and every estimate, for the label *and* for the table."""
        # Both are reads of the run rather than of the slice on screen, and both are wanted by
        # `_render` on the UI thread — the names are one `read_recording` per recording.
        # Returned, not assigned: an older read finishing after a newer one must not win.
        vocab = self._vocabulary()
        names = self._experiment.recording_names()  # off-thread: `_compute`
        # Asked only of a live run: a snapshot is read-only whatever its run says now.
        finished = self._run.editable and self._run.status == "complete"
        try:
            everything = list(self._run.artifact().instances)  # off-thread: `_compute`
        except FileNotFoundError:
            return {
                "labels": [],
                "mine": 0,
                "recording_names": names,
                "vocab": vocab,
                "finished": finished,
            }
        run_dir = self._run._audit_dir()
        # Every figure comes from the run's API — the same `audit_precision` the CLI prints and a
        # script reads — so the three surfaces cannot disagree about one audit. This tab used to
        # call the storage layer itself, with the artifact's instances, which is how its headline
        # stayed an unweighted share after the estimate learned to weight by score bin.
        by_label = []
        for label in self._labels():
            if not any(r.source_label == label for r in everything):
                continue
            by_label.append(
                {
                    "label": label,
                    "name": self._display_name(label),
                    **(
                        self._run.audit_precision(  # off-thread: `_compute`
                            label=label, cutoff=self._cutoff(label)
                        )
                        or {}
                    ),
                }
            )
        mine = [
            r
            for r in everything
            if self._label is None or r.source_label == self._label
        ]
        judged_rows, pending, remaining = self._review_rows(self._pool())
        # The headline's count and its total have to be drawn from one set: this scope at its
        # current cutoff. Counted over the whole log, raising a cutoff shrank the total under an
        # unchanged count ("75 of 68"), and a second behavior's work landed in the first's line.
        current = {r.instance_id for r in bo.bouts(mine)}
        counted = {
            r.instance_id
            for r in audit.latest_decisions(run_dir)
            if r.human_label is not None
        }
        # Only behaviors with a cutoff: one without is absent from the artifact rather than cut
        # by anything, so its decisions are not "below the cutoff".
        pool = bo.bouts(self._pool())
        cut = {
            label
            for label in {r.source_label for r in pool}
            if self._cutoff(label) is not None
        }
        pooled = {r.instance_id for r in pool if r.source_label in cut}
        return {
            "labels": by_label,
            "recording_names": names,
            "vocab": vocab,
            "finished": finished,
            "mine": len(current),
            "judged_rows": judged_rows,
            "pending": pending,
            "remaining": remaining,
            "estimate": self._run.audit_precision(  # off-thread: `_compute`
                label=self._label, cutoff=self._cutoff(self._label)
            ),
            "by_recording": self._run.audit_precision_by_recording(
                label=self._label, cutoff=self._cutoff(self._label)
            ),
            "judged": len(counted & current),
            "below_cutoff": len((counted & pooled) - current),
            "bins": [
                b
                for b in (
                    (
                        self._run.audit_precision_by_score_bin(
                            label=self._label, cutoff=self._cutoff(self._label)
                        )
                        if self._label is not None
                        else None
                    )
                    or []
                )
                if b["precision"] is not None
            ],
        }

    def _vocabulary(self) -> dict[str, str]:
        """``{behavior id: name}``. The id (`q01`) is the tool's and is shown nowhere."""
        reader = getattr(self._run, "state", None)
        if reader is None:
            return {}
        behaviors = getattr(reader(), "behaviors", None)
        if behaviors is None:
            return {}
        return {b.id: b.name for b in behaviors}

    def _display_name(self, label: str) -> str:
        """What ``label`` is called on screen — its name, or the label when it has none."""
        if self._vocab is None:
            self._vocab = self._vocabulary()
        return self._vocab.get(label) or label

    def _recording_name(self, recording_id: str) -> str:
        """The recording's name, or its id if it has none left (deleted since the run)."""
        return self._names.get(recording_id, recording_id)

    def _cutoff(self, label: str | None) -> float | None:
        """The cutoff precision is reported at — ``None`` for every behavior at once."""
        state = getattr(self._run, "state", None)
        if state is None or label is None:
            return None
        try:
            return state().behavior(label).cutoff
        except (KeyError, AttributeError):
            return None

    # -- rendering ------------------------------------------------------------ #
    def _render(self, data: dict) -> None:
        if "vocab" in data:
            self._vocab = data["vocab"]
        # Before every enable below, each of which asks `_editable`.
        self._finished = bool(data.get("finished"))
        # Here, ahead of the early return for an empty run: set only on the full path, a run
        # with nothing to audit went on saying "Still reading this run…" after it had been read.
        for button in (self._start, self._continue, self._intro_start):
            button.setToolTip(self._read_only_reason())
        # Before the first `_display_name` below, which is the headline's own subject.
        self._names = data.get("recording_names") or {}
        name = self._display_name(self._label) if self._label else "all behaviors"
        if not data.get("mine"):
            self._headline.setText(f"{name} — nothing to audit yet")
            self._qualifiers.setVisible(False)
            self._intro_headline.setText(f"Nothing to audit in {name} yet")
            self._intro_body.setText(
                "Auditing samples the instances this run found. There are none to sample."
            )
            # Hidden, not greyed: its text is set only once there is something to draw, so with
            # nothing to audit it was an empty box with no word on it.
            self._intro_start.setVisible(False)
            self._intro_start.setEnabled(False)
            self._intro_note.setText("")
            self._start.setEnabled(False)
            self._set_buttons(False)
            self._sync_mode()
            return
        # The batch bar resets every 25, so the running total lives here, where it is always
        # on screen. Decisions the cutoff now excludes are named beside it, or raising the cutoff
        # reads as work lost.
        below = data.get("below_cutoff") or 0
        self._headline.setText(
            f"{name} — precision {format_precision(data.get('estimate'))}"
            f"     ·     {data['judged']} of {data['mine']} instances audited"
            + (
                f" · {below} earlier decision{'s are' if below != 1 else ' is'} on "
                f"clip{'s' if below != 1 else ''} now below the cutoff"
                if below
                else ""
            )
        )
        notes = audit_figures.notes(data.get("estimate"))
        self._qualifiers.setText(" · ".join(notes))
        self._qualifiers.setVisible(bool(notes))
        self._start.setEnabled(self._editable())
        self._intro_start.setVisible(True)

        # "all behaviors" is a scope, not a name, so it reads badly in a sentence about what
        # something *is*: "how many of these really are all behaviors" asks nothing. Named only
        # when a single label is selected.
        subject = self._display_name(self._label) if self._label else None
        self._intro_headline.setText(
            f"{data['mine']:,} instances"
            + (f" of {subject}" if subject else "")
            + ", none reviewed yet"
        )
        self._intro_body.setText(
            "Precision is how many of these are really "
            + (subject if subject else "the behavior they were assigned")
            + ", and it needs a sample.\n"
            f"Reviewing {min(BATCH, data['mine'])} clips takes a couple of minutes and gives "
            "you a figure with its interval."
        )

        self._by_label_heading.setText("BY BEHAVIOR")
        # No id column: a behavior's `q01` is shown nowhere else, so it would be noise.
        self._fill(
            self._by_label,
            self._by_label_model,
            ["Behavior", "precision"],
            [
                ((row["name"], format_precision(row or None)), row["label"])
                for row in data["labels"]
            ],
        )
        self._by_recording_heading.setText(f"BY RECORDING · {name}")
        self._fill(
            self._by_recording,
            self._by_recording_model,
            ["Recording", "precision"],
            [
                (
                    (self._recording_name(row["recording_id"]), format_precision(row)),
                    None,
                )
                for row in (data.get("by_recording") or [])
            ],
        )
        self._draw_curve(data.get("bins") or [])
        self._judged = data.get("judged_rows") or []
        self._pending = int(data.get("pending") or 0)
        self._remaining = int(data.get("remaining") or 0)
        self._decisions.update({row.instance_id: said for row, said in self._judged})
        self._refresh_sidebar()
        if not self._queue:
            # Nothing drawn — walk what has been judged, or say there is nothing at all.
            self._queue = [row for row, _said in self._scope_rows()]
            self._index = 0
        if self._queue:
            self._show_current()
        else:
            self._show_empty()

    @staticmethod
    def _fill(
        view, model, headers, rows, *, fit: bool = True, max_rows: int | None = None
    ) -> None:
        """Rebuild ``model`` from ``(cells, payload)`` rows; the payload rides column 0.

        ``fit`` is for the two summaries beside the review: their **height** hugs their rows,
        which is what stops two rows drawing above ~700px of empty grid with the confidence
        panel pushed off the bottom, and their **width** fills the column they sit in.

        ``max_rows`` is the review list's half — the same width rule (it is its column's
        content, `gui_design` §3's named exception) with a *height* that stops at that many
        rows and scrolls. Without it the list took the whole pane; `_build_sidebar` has the
        measurement.

        Every table here therefore fills its width and bounds its height, which is the two
        halves of `gui_design` §3 rather than a choice between them.
        """
        model.clear()
        model.setHorizontalHeaderLabels(headers)
        for cells, payload in rows:
            items = [QStandardItem(str(cell)) for cell in cells]
            if payload is not None:
                items[0].setData(payload, Qt.ItemDataRole.UserRole)
            model.appendRow(items)
        if fit:
            fit_table_height(view)
            # **Fill**, not hug. `fit_table_width` caps a table to its columns, which is right
            # when the leftover would be dead grid *inside* the table — but these two sit alone
            # in a 400px column, so capping them to ~200px of content left the other 200 blank
            # beside them and the confidence plot below filling the whole width. Reported as
            # "the tables don't fill the pane".
            fill_width(view)
        elif max_rows is not None:
            # `min_rows=0`: the list sits beside a player in a splitter, which decides the
            # share; a floor here only raises the whole pane's minimum.
            fit_table_height(view, max_rows=max_rows, min_rows=0)

    #: Why the audit controls are dead when they are, on a snapshot (§21).
    FROZEN = (
        "This is a snapshot — the audit as it stood when the result left the run. "
        "Go back to the live run to judge more clips."
    )
    #: Why they are dead on a finished run — the other reason, undone by Reopen instead.
    FINISHED = "This run is finished. Reopen it to judge more clips."

    def _editable(self) -> bool:
        """Whether this handle may record decisions.

        A snapshot carries its own copy of ``audit.csv``, so the estimate and the judgments behind
        it are worth *reading* — "what precision did I report when I published this?" is the whole
        point of a pinned result. Adding to them is what cannot happen: `label_audit` and
        `audit_sample` write to the **run's** directory, so a decision recorded from a snapshot
        would land on the live run rather than on the copy being read.

        A finished run refuses them too (`_record`'s finalize check), so asking only
        ``editable`` left Start live on every finished run, as a button the API refuses.
        """
        return self._run.editable and self._finished is False

    def _read_only_reason(self) -> str:
        """The tooltip for a dead control: which of the three reasons it is, or ``""``."""
        if not self._run.editable:
            return self.FROZEN
        if self._finished is None:
            return "Still reading this run…"
        return self.FINISHED if self._finished else ""

    def _set_buttons(self, enabled: bool) -> None:
        enabled = enabled and self._editable()
        for button in self._buttons:
            button.setEnabled(enabled)
            button.setToolTip(self._read_only_reason())

    def _show_empty(self) -> None:
        """The pane with nothing to show at all — no batch, and nothing judged.

        The only state left for this page. It used to carry three, because finishing a batch
        put you on a summary you then had to leave; now the walk never ends, so there is
        nowhere to return from.
        """
        self._player.pause()
        self._bar.setValue(0)
        self._progress.setText("")
        self._where.setText("")
        self._question.setText("")
        self._prompt.setText(
            "No clips reviewed yet.\n\nPrecision needs a sample. Reviewing "
            f"{BATCH} clips takes a couple of minutes\nand gives you a figure with its "
            "interval."
        )
        self._stack.setCurrentIndex(0)
        # Safe from here too: `_sync_mode` keys on whether anything has ever been judged or is
        # drawn, so an empty *scope* after an audit stays on the workspace and only a genuinely
        # untouched tab goes back to the intro.
        self._sync_mode()
        self._set_buttons(False)
        self._back.setEnabled(False)
        self._forward.setEnabled(False)

    def _refresh_sidebar(self) -> None:
        """The two draw buttons, and why one of them cannot be pressed."""
        offer = min(BATCH, self._remaining)
        self._start.setText(
            f"Start auditing ({offer} clips)"
            if not self._judged
            else f"Review {offer} more"
        )
        # Greyed with the reason on the page, rather than staying live and failing into a
        # label under the video: it used to stay enabled with the pool exhausted, still
        # promising 25, and clicking it did nothing anyone would notice.
        #
        # `_editable` here as well as in `_render`: this is the *second* place that enables the
        # button, and gating only the first left it live on a snapshot — the enable site you did
        # not think of is the one that decides.
        self._start.setEnabled(bool(self._remaining) and self._editable())
        self._continue.setText(f"Continue — {self._pending} unjudged")
        self._continue.setVisible(bool(self._pending))
        # Enabled, not just labelled: it was given text, visibility and a tooltip and never
        # `setEnabled`, so on a snapshot it stayed live and clicking it did nothing — the third
        # enable site in this file, after the two the comments above already describe.
        self._continue.setEnabled(self._editable())
        self._exhausted.setText(
            "" if self._remaining else "Every candidate has been reviewed."
        )
        self._intro_start.setText(f"Start auditing ({offer} clips)")
        self._intro_start.setEnabled(self._start.isEnabled())
        self._intro_start.setToolTip(self._start.toolTip())
        self._intro_note.setText(
            "" if self._remaining else "Every candidate has been reviewed."
        )
        self._fill_list()
        self._sync_mode()

    def _batch_tally(self) -> str:
        """What the batch you just finished was, beside the clip counter.

        It used to own a whole pane. A line is the right weight: the precision in the heading
        is the result, and this is the sanity check on your own pass — 24 yes and 1 no reads
        very differently from 18/5/2, and a precision figure alone cannot show you that.
        """
        said = [self._decisions.get(row.instance_id) for row in self._batch]
        return (
            f"✓ batch done — {len(said)} reviewed · {said.count(True)} yes · "
            f"{said.count(False)} no · {said.count(None)} unclear"
        )

    def _scope(self) -> str:
        for scope, button in self._scopes.items():
            if button.isChecked():
                return scope
        return "judged"

    def _scope_rows(self, scope: str | None = None) -> list:
        """`(instance, said)` for one scope — what the list shows and what `← / →` walk."""
        scope = scope or self._scope()
        if scope == "batch":
            return [(row, self._decisions.get(row.instance_id)) for row in self._batch]
        if scope == "unclear":
            return [pair for pair in self._judged if pair[1] is None]
        return list(self._judged)

    def _fill_list(self) -> None:
        """The list, and the count on each scope that says what is in it."""
        counts = {scope: len(self._scope_rows(scope)) for scope, _t in SCOPES}
        for scope, text in SCOPES:
            button = self._scopes[scope]
            button.setText(f"{text} ({counts[scope]})")
            # Greyed, never hidden: a scope that vanishes takes its count with it.
            button.setEnabled(bool(counts[scope]))
        if not counts[self._scope()]:
            self._scopes["judged"].setChecked(True)
        # No score column. These rows are re-judgeable from here, and 0.71 beside a clip you
        # are about to reconsider is the bias `_reveal` already withholds on a revision.
        self._fill(
            self._list,
            self._list_model,
            ["Clip", "Said"],
            [
                (
                    (
                        f"{self._recording_name(row.recording_id)} · "
                        f"{row.t_ns / 1e9:.1f} s",
                        # A clip drawn and not yet judged said nothing; "unclear" is an answer.
                        (
                            self._said(row)
                            if row.instance_id in self._decisions
                            else "—"
                        ),
                    ),
                    row.instance_id,
                )
                for row, said in self._scope_rows()
            ],
            fit=False,
            max_rows=BROWSE_ROWS,
        )
        # The **clip** column takes the slack, not the last one. `Said` holds one of three
        # words and never needs more than "unclear"; the clip needs a recording's name and a
        # timestamp, and at the width a header cell gave it — 100px, measured, against the
        # 336px "Fish tank A · 123.4 s" asks for — it elided them both. A stretch on the last
        # section put the spare 145px on the narrow column and left the wide one truncated,
        # which is the shape `cluster_workspace` already solved this way for its name column.
        #
        # **Here rather than where the table is built**, and re-applied on every fill: `_fill`
        # clears the model, and clearing removes the columns — so a width set at construction
        # (when the model has no columns at all) is applied to nothing, and one set after the
        # first fill is discarded by the second. `give_slack_to`, not `Stretch`: a stretched
        # clip column could not be dragged, and dragging `Said` wider grew it leftward.
        self._list.resizeColumnToContents(1)
        give_slack_to(self._list, 0)
        self._sync_selection()

    def _on_scope(self, _scope: str) -> None:
        """A scope change is a change of walk, so the queue follows it."""
        if self._syncing:
            return
        rows = self._scope_rows()
        self._queue = [row for row, _said in rows]
        self._decisions.update({row.instance_id: said for row, said in rows})
        self._index = 0
        self._fill_list()
        self._show_current()

    def _on_row(self, current, _previous) -> None:
        """A row selected is a position in the walk — it looks, and never records."""
        if self._syncing or not current.isValid():
            return
        item = self._list_model.item(current.row(), 0)
        instance_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        position = next(
            (i for i, row in enumerate(self._queue) if row.instance_id == instance_id),
            None,
        )
        if position is None:
            return
        self._index = position
        self._show_current()

    def _sync_selection(self) -> None:
        """Walk → list. Guarded, or selecting a row would re-drive the walk that moved it."""
        instance = self._current()
        if instance is None:
            return
        for r in range(self._list_model.rowCount()):
            item = self._list_model.item(r, 0)
            if (
                item is not None
                and item.data(Qt.ItemDataRole.UserRole) == instance.instance_id
            ):
                self._syncing = True
                try:
                    self._list.selectRow(r)
                finally:
                    self._syncing = False
                return

    # -- the review loop ------------------------------------------------------ #
    def _continue_batch(self) -> None:
        """Just the clips drawn but never judged. Same pending-first draw, smaller budget."""
        self._start_batch(budget=self._pending)

    def _start_batch(self, budget: int | None = None) -> None:
        if not self._editable():
            return
        drawn = self._run.audit_sample(budget=budget or BATCH, label=self._label)
        if not drawn:
            self._refresh_sidebar()  # says why, where the button that failed is
            return
        self._batch = list(drawn)
        self._queue = list(drawn)
        self._index = 0
        self._tally = ""  # the tally describes a finished batch, not this one
        self._syncing = True
        try:
            self._scopes["batch"].setChecked(True)
        finally:
            self._syncing = False
        self._fill_list()
        # `WidgetWithChildrenShortcut` fires only while this widget or a child has focus, so a
        # batch that started from the button would otherwise need a second click before the
        # keys did anything.
        self.setFocus()
        self._show_current()

    def _current(self):
        return self._queue[self._index] if self._index < len(self._queue) else None

    def _step(self, delta: int) -> None:
        """Move within the batch without judging — how a misclick gets corrected.

        Bounded rather than wrapping: arriving back at clip 1 after the last one reads as a
        reset, and a reviewer who has lost their place cannot tell the two apart.
        """
        if not self._queue:
            return
        self._index = max(0, min(len(self._queue) - 1, self._index + delta))
        self._show_current()

    def _decision_for(self, instance) -> bool | None:
        """This reviewer's standing answer for a clip, or ``None`` if they have not said."""
        return self._decisions.get(instance.instance_id)

    def _said(self, instance) -> str:
        """What the reviewer said about a clip: yes, no or unclear."""
        return {True: "yes", False: "no", None: "unclear"}[
            self._decisions.get(instance.instance_id)
        ]

    def _show_current(self) -> None:
        instance = self._current()
        if instance is None:
            self._show_empty()
            return
        # The **outer** mode as well as the inner one. `_start_batch` ends here, and switching
        # only the review pane's stack left the page still showing the intro: the batch was
        # drawn, the clip was loaded, and pressing "Start auditing" looked like it did nothing.
        # Every path that puts something on screen has to say so at both levels.
        self._sync_mode()
        self._sync_selection()
        self._stack.setCurrentIndex(1)
        self._set_buttons(True)
        self._bar.setRange(0, len(self._queue))
        self._bar.setValue(
            sum(1 for row in self._queue if row.instance_id in self._decisions)
        )
        self._back.setEnabled(self._index > 0)
        self._forward.setEnabled(self._index < len(self._queue) - 1)

        name = self._display_name(self._label) if self._label else "behavior"
        answered = self._decision_for(instance)
        said = self._said(instance)
        self._progress.setText(
            f"clip {self._index + 1} of {len(self._queue)}"
            + (
                ""
                if instance.instance_id not in self._decisions
                else f"  ·  you said {said} — press to change"
            )
            + (f"      {self._tally}" if self._tally else "")
        )
        self._question.setText(f"Is this {name}?")
        # The answer is marked on the button; the **score is not shown**, even on a clip whose
        # score was revealed when it was first judged. A revision made while looking at 0.95 is
        # the bias the blind review exists to remove, and correcting a misclick does not need
        # the number.
        for button, (_text, _key, value) in zip(self._buttons, KEYS):
            button.setDefault(
                instance.instance_id in self._decisions and value == answered
            )
        self._where.setText(
            f"{self._recording_name(instance.recording_id)} · "
            f"{instance.t_ns / 1e9:.1f} s · "
            f"{(instance.end_t_ns - instance.t_ns) / 1e9:.1f} s long"
        )
        if instance.instance_id == self._shown:
            # Already on screen. A refresh must not restart it: `set_clip` autoplays, so
            # re-setting the same clip put the loop back after the batch-end pause and
            # re-decoded a frame nobody asked for.
            return
        self._shown = instance.instance_id
        self._player.set_clip(
            ClipSpec(
                recording_id=instance.recording_id,
                track_id=int(instance.subject or 0),
                embed_run_id=run_selection.run_for_recording(
                    self._run.data_sources(), instance.recording_id
                )
                or "",
                start_t_ns=int(instance.t_ns),
                end_t_ns=int(instance.end_t_ns),
                label=name,
                # Deliberately not `instance.score`. `ClipSpec` carries one and a caption may
                # render it; the reviewer must not see it before deciding.
                score=None,
            )
        )

    def _decide(self, value: bool | None) -> None:
        """Record one judgment: yes, no, or ``None`` for unclear."""
        instance = self._current()
        if instance is None or not self._buttons or not self._buttons[0].isEnabled():
            return
        if not self._editable():
            return
        revision = instance.instance_id in self._decisions
        self._run.label_audit(instance.instance_id, value)
        self._decisions[instance.instance_id] = value
        self._reveal(instance, value, revision=revision)
        # A revision stays put so the reviewer can see it took; a first answer advances, which
        # is what makes a 25-clip pass three keys and no navigation.
        if not revision and self._index < len(self._queue) - 1:
            self._index += 1
        # **Every** clip answered, not "the last one was answered": `→` walks past clips
        # without judging them, so a batch can end on clip 25 with clip 12 still blank.
        finished = bool(self._batch) and all(
            row.instance_id in self._decisions for row in self._batch
        )
        if not revision and finished:
            self._tally = self._batch_tally()
            # A looping clip beside the numbers is motion competing with the result.
            self._player.pause()
        self._show_current()
        # The list is redrawn from `_decisions`, which costs nothing; the panels are a full
        # re-read of the artifact and the pool, so they wait for the end of the batch — or for
        # a revision, which is one deliberate act whose effect on the number you want to see.
        # Recomputing on every keystroke put a busy overlay between the reviewer and the
        # next clip, and made a 25-clip pass 25 reads of everything.
        self._fill_list()
        if finished or revision:
            self.reload()
        self.changed.emit()

    def _reveal(self, instance, value: bool | None, *, revision: bool = False) -> None:
        """Say what was just recorded, and show the score only the first time.

        A revision gets the answer and **no number**: the score was revealed once already when
        the clip was first judged, and putting it back on screen while someone reconsiders is
        the bias the blind review exists to remove. Correcting a misclick does not need it.
        """
        said = self._said(instance)
        if revision:
            self._last.setText(f"changed to {said}")
            return
        score = "" if instance.score is None else f"{instance.score:.3f} · "
        self._last.setText(f"last: {score}you said {said}")
