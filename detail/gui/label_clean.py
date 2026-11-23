"""AUDIT's review pane: every clip of one cluster, to remove the wrong ones and finish (§19, §22).

**One cluster at a time, every clip of it** (review round 11, the user's decision: "we force people
to review all clips"): one continuous scroll of every clip, all playing, **least typical first** —
the clips most likely to be wrong come first. The order is fixed: an order switch here would only
be a way to see fewer of the clips that matter.

**Everything that is about the cluster sits in the top row** (review round 12, 3A): "cluster #871",
its label, a thin progress bar with "reached 40 of 62" (``LiveMontage.seen``), and **✓ Finish
review** — a large bold default button (round 14, 2A: Enter presses it), "60 climb · 2 not climb"
under it, disabled with "Scroll to the last clip first" until the last row of tiles has been on
screen. Nothing sits under the scroll, so nothing has to be scrolled to.

**A click removes** (review round 13, 2B — it replaced round 11's selection): a plain click on a
clip removes it from the label at once ("not climb": faded, a bold ✕, the tag), and a click on a
removed clip puts it back. **A run** (round 14, 1C) is a **drag** — every clip from the one pressed
to the one under the mouse, removed on letting go as one step, or put back when the drag started
on a removed clip — or a **Shift-click**, the run from the last clicked clip. Both show what they
will do before they do it (``LiveMontage.preview``): the run tinted on the grid and, in bold on
the line under the top row, "Let go to remove 4 clips · Esc cancels" / "Let go to put back 3
clips" / "Shift-click removes 7 clips". Clips already in that state are skipped. **Right-click** offers *Move to <label>* for every other
label and, on a removed or moved clip, *Put back in climb*. A **double-click** (or Space) opens the
clip window; its first press was a click, so the pane retracts that click (``undo_requested``)
when a real double-click — a ``MouseButtonDblClick`` the pane watches on the grid — on the same
clip follows within the double-click interval. Space after a click opens the clip and keeps the
removal: it is a key, not the second half of a click. A change is
written at once (the page's ordered off-thread queue), so leaving loses nothing; the clip stays
where it was and the scroll does not move.

**While the review is being saved** ("Saving the review…") and **while a method that ranks on
demand ranks the clips** ("ranking this cluster's clips…"), a click changes nothing: the line under
the top row says why, and Finish is greyed with it. **A reopened review** shows its earlier figure
beside the title as out of date — "was 97% · reviewing again" (``set_was``) — until it is finished
again.

**Undo** (round 12, 1A): under the top row, a line says what the last change did — "Removed 3
from climb", "Put back 1 in climb", "Moved 1 to climb › left" — with **Undo (⌘Z)**; ⌘Z steps
back through every change of this review. Before any change the line says how to start. The
history is the page's (it knows each clip's earlier mark); this pane only shows it.

**A finished cluster** (round 13, 1B): the top row and the undo line give way to a framed box —
"✓ Reviewed", the cluster, its label and the day, the precision large, "57 of 60 are climb · 3
removed", **Next cluster: #871** and **Reopen review** — and the page splits the grid into
"Removed (3)" and "Kept as climb (57)". The grid is read-only until Reopen review: no click
removes, and no hover cue says one would; a double-click still opens a clip.

**The montage toolbar** is every clip grid's (2026-11-09, option A): **Show ▾ · Size · Export ▾**
over the grid — the order menu hidden, since the order is fixed — with **Show ▾**'s context and
**Export ▾**'s pick mode (``montage_export.py``). Pick mode lays a tinted bar over the grid and
takes the clicks for itself: a click then picks a clip for the export, never removes it from the
label, and no ✕ cue is drawn; Finish is greyed, so Enter does not finish the review either. **Quit** or Esc — or an export of the picks that finishes — gives
the click back to removing. Picks touch
nothing here — not the removals, not the review — so a reviewed, read-only cluster can still be
picked from and exported.

The pane is a widget the page fills; every read and write is the page's (``label_check``).
"""

from __future__ import annotations

from dataclasses import dataclass

import time

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.gui.live_montage import LiveMontage, MontageClip, context_span
from detail.gui.sizing import emphasis_font, set_available, small_font
from detail.gui.status_color import MUTED, READY, set_button_color, set_label_color
from detail.storage import behavior_audit as audit
from detail.storage import behavior_output as bo

#: A clip loops at least this long, so a one-frame clip is still watchable.
MIN_LOOP_NS = 400_000_000
#: The review grid's orientation (As recorded / Aligned), kept for AUDIT.
ALIGNED_KEY = "label_check/review_aligned"
#: What **Show ▾** draws on the review grid (``show_menu``), kept for AUDIT.
LOOK_KEY = "label_check/review_look"
#: The context **Show ▾** plays either side of each review clip, kept for AUDIT.
CONTEXT_KEY = "label_check/review_context"
#: What Finish says while the scroll has not reached the end.
SCROLL_FIRST = "Scroll to the last clip first."
#: The line under the top row before any change.
CLICK_HINT = "Click a clip to remove it. Drag, or hold Shift, to remove a run."
#: The line under the top row while pick mode has the grid's clicks.
PICKING_HINT = (
    "Picking clips to export: a click picks a clip, it removes nothing. Quit or Esc to "
    "remove clips again."
)
#: How much darker than the palette's accent Finish is filled with. The style lightens a
#: button's colour into its gradient: the macOS accent ``#0a60ff`` as given rendered ``#8fb7fe``,
#: 2.0:1 under white text; darker(190) renders ``#4070c6``, 4.8:1 (measured, round 14).
FINISH_ACCENT_DARKER = 190
#: Why Undo is off with no change to undo.
NOTHING_TO_UNDO = "Nothing to undo"
#: Why Finish is greyed while pick mode has the grid's clicks.
PICKING_FINISH = "Quit picking clips first: Finish saves the review, not the picks."
#: Why nothing can be marked while a finish is being written.
SAVING = "Saving the review…"
#: Why nothing can be marked while a method that ranks on demand ranks the clips.
RANKING = "ranking this cluster's clips…"
#: Why a finished cluster's clips do not respond to a click.
REVIEWED_READ_ONLY = "This cluster is reviewed. Press Reopen review to change it."


def reviewed_read_only(noun: str) -> str:
    """Why a finished scope's clips do not respond to a click: :data:`REVIEWED_READ_ONLY` for
    a ``noun`` other than a cluster (a query run reviews a recording)."""
    return REVIEWED_READ_ONLY.replace("cluster", noun)


def clips_text(n: int) -> str:
    return f"{n:,} clip" if n == 1 else f"{n:,} clips"


def no_preview_text(n: int) -> str:
    """Why Finish is off while ``n`` clips show *no preview*: Finish counts every clip not
    removed as the label, and a clip whose picture never showed was not seen."""
    shows = "shows" if n == 1 else "show"
    return (
        f"{clips_text(n)} {shows} no preview, and a clip nobody saw cannot be counted. "
        "Build the previews, then finish."
    )


def waiting_text(n: int) -> str:
    """Why Finish is off while ``n`` clips' pictures are still on their way."""
    return f"Waiting for {clips_text(n)} to show."


def preview_text(preview) -> str:
    """What the line under the top row says while a run preview is up (``LiveMontage.preview``):
    what letting go, or the Shift-click, will do."""
    n = len(preview.indices)
    if preview.kind == "drag" and preview.put_back:
        if not n:
            return "Nothing removed in this run to put back · Esc cancels"
        return f"Let go to put back {clips_text(n)}"
    if preview.kind == "drag":
        if not n:
            return "Every clip in this run is removed already · Esc cancels"
        return f"Let go to remove {clips_text(n)} · Esc cancels"
    if not n:
        return "Shift-click removes nothing here: these are removed already"
    return f"Shift-click removes {clips_text(n)}"


def display(path: str | None) -> str:
    """A label path as people read it: ``climb › left``."""
    return (path or "").replace(bo.PATH_SEPARATOR, audit.PATH_DISPLAY)


def clip_key(recording_id: str, subject, start_ns, end_ns) -> tuple[str, int, int, int]:
    """What names a clip: one leaf run of one subject, ``(recording, subject, start, end)``."""
    return (str(recording_id), int(subject or 0), int(start_ns), int(end_ns))


def edit_key(edit) -> tuple[str, int, int, int]:
    """:func:`clip_key` of a take-out record (a dict or an object with the same names)."""
    get = edit.get if isinstance(edit, dict) else lambda k: getattr(edit, k, None)
    return clip_key(get("recording_id"), get("subject"), get("t_ns"), get("end_t_ns"))


def _edit_field(edit, name: str):
    return edit.get(name) if isinstance(edit, dict) else getattr(edit, name, None)


def edit_label(edit, side: str) -> str | None:
    """The label an edit names on ``side`` (``"from"`` / ``"to"``), as a path: ``from_path``
    where the page added it, else the run's ``from_name``, else the stored id."""
    for field in (f"{side}_path", f"{side}_name", f"{side}_label"):
        value = _edit_field(edit, field)
        if value:
            return str(value)
    return None


def reopen_tip(precision: str, removed: int, noun: str = "cluster") -> str:
    """Reopen review's tooltip: what reopening changes, and that the removals survive it."""
    figure = f"update {precision}" if precision else "update the figure"
    if removed <= 0:
        kept = "Nothing was removed."
    elif removed == 1:
        kept = "Your 1 removal is kept."
    else:
        kept = f"Your {removed:,} removals are kept."
    return f"Change this {noun}'s removals, then finish again to {figure}. {kept}"


def tag_for(edit) -> str:
    """ "not climb" for a clip removed, "→ climb › left" for one moved to another label."""
    to = edit_label(edit, "to")
    if to:
        return f"→ {display(to)}"
    return f"not {display(edit_label(edit, 'from'))}"


def crossed_for(edit) -> bool:
    """Whether a tile wears the bold ✕: a clip removed from its label, not moved to another."""
    return edit is not None and not edit_label(edit, "to")


def day_of(when: str | None) -> str:
    """ "8 Nov" for an ISO-8601 time, in local time; ``""`` for none or one that does not
    parse."""
    if not when:
        return ""
    try:
        from datetime import datetime

        stamp = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    except ValueError:
        return ""
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone()  # stored in UTC; the day the reader lived through
    return f"{stamp.day} {stamp.strftime('%b')}"


def note_for(edit) -> str:
    """The faded tile's first tooltip line: "Removed from climb in review, 8 Nov"."""
    day = day_of(_edit_field(edit, "at"))
    day = f", {day}" if day else ""
    source = display(edit_label(edit, "from"))
    to = edit_label(edit, "to")
    if to:
        return f"Moved from {source} to {display(to)} in review{day}"
    return f"Removed from {source} in review{day}"


@dataclass
class ScopeClips:
    """The clips of one scope (a label, or one of its clusters), with what each tile needs."""

    rows: list[
        dict
    ]  # ``ClipTable.rows()``: recording_id, subject, start_t_ns, end_t_ns, …
    typicality: np.ndarray
    names: dict[str, str]

    def least_typical_first(self) -> np.ndarray:
        """Indices, least typical first — the clips most likely to be wrong first; unknown
        typicality last, ties in time order (``ClipTable.order``'s convention)."""
        key = -np.asarray(self.typicality, dtype=np.float64)
        key = np.where(np.isfinite(key), key, np.inf)
        return np.argsort(key, kind="stable")

    def key(self, index: int) -> tuple[str, int, int, int]:
        row = self.rows[index]
        return clip_key(
            row["recording_id"], row["subject"], row["start_t_ns"], row["end_t_ns"]
        )

    def caption(self, index: int) -> str:
        row = self.rows[index]
        name = self.names.get(row["recording_id"], row["recording_id"])
        seconds = int(row["start_t_ns"]) // 1_000_000_000
        return f"{name} {seconds // 60}:{seconds % 60:02d}"


def tiles_for(
    clips: ScopeClips, order, edits: dict, context_s: float = 0.0
) -> list[MontageClip]:
    """One tile per clip in ``order``, looping through ``context_s`` either side; a clip with a
    removal record in ``edits`` is faded."""
    tiles = []
    for rank, i in enumerate(order, start=1):
        row = clips.rows[int(i)]
        edit = edits.get(clips.key(int(i)))
        low, high = context_span(row["start_t_ns"], row["end_t_ns"], context_s)
        tiles.append(
            MontageClip(
                max(high - low, MIN_LOOP_NS),
                rank,
                clips.caption(int(i)),
                tag=tag_for(edit) if edit is not None else None,
                faded=edit is not None,
                note=note_for(edit) if edit is not None else "",
                crossed=crossed_for(edit),
            )
        )
    return tiles


def _shortcut(widget: QWidget, key: str, slot) -> QShortcut:
    shortcut = QShortcut(QKeySequence(key), widget)
    shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    shortcut.activated.connect(slot)
    return shortcut


class ReviewPane(QWidget):
    """Every clip of one cluster, to remove the wrong ones and finish (module docstring)."""

    #: ``(display indices, to_label)``: remove these from the label, or move them to
    #: ``to_label`` (a label path).
    take_out_requested = pyqtSignal(list, object)
    #: Display indices to put back.
    put_back_requested = pyqtSignal(list)
    undo_requested = pyqtSignal()
    finish_requested = pyqtSignal()
    next_requested = pyqtSignal()
    again_requested = pyqtSignal()

    def __init__(
        self, parent=None, *, export, noun: str = "cluster", keys: str = "label_check"
    ) -> None:
        """``export`` is the page's ``montage_export.ExportSource``: the clips **Export ▾**
        writes. ``noun`` is what one review covers — a ``"cluster"`` on a clustering run, a
        ``"recording"`` of a behavior on a query run (§20) — and ``keys`` the ``QSettings`` group
        the grid's orientation, layers and context are kept under, so each page keeps its own.
        """
        super().__init__(parent)
        self._noun = noun
        self._keys = keys
        self._label = ""
        self._reason = "Still reading this run…"
        self._targets: list[str] = []
        self._finished = False
        #: How many clips the shown review counted as not the label (`show_result`).
        self._removed = 0
        self._saving = False
        #: Why the clips are not here yet (``RANKING``), or ``""``.
        self._waiting = ""
        #: Whether the last mouse event on the grid was a double-click: what tells a real
        #: double-click's ``clip_activated`` from Space's.
        self._double_click = False
        self._next_node: int | None = None
        self._can_undo = False
        #: ``(index, when)`` of the last plain click — what a double-click on it retracts.
        self._last_click: tuple[int, float] | None = None
        #: What the last change did, for the line under the top row; ``None`` before any.
        self._undo_line: str | None = None

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        self.top = QWidget()
        top = QHBoxLayout(self.top)
        top.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("")
        self.title.setFont(emphasis_font(self.title, steps=2))
        top.addWidget(self.title)
        top.addSpacing(10)
        self.label_name = QLabel("")
        self.label_name.setFont(emphasis_font(self.label_name))
        top.addWidget(self.label_name)
        top.addSpacing(10)
        #: "was 97% · reviewing again" on a reopened review (`set_was`).
        self.was = QLabel("")
        set_label_color(self.was, MUTED)
        self.was.setVisible(False)
        top.addWidget(self.was)
        top.addSpacing(6)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(120, 6)
        self.progress.setToolTip(
            "How far down the clips you have scrolled. Finish opens once the last row has "
            "been on screen."
        )
        top.addWidget(self.progress)
        top.addSpacing(6)
        self.reached = QLabel("")
        set_label_color(self.reached, MUTED)
        top.addWidget(self.reached)
        top.addStretch(1)
        finish = QVBoxLayout()
        finish.setContentsMargins(0, 0, 0, 0)
        finish.setSpacing(1)
        self.finish = QPushButton("✓ Finish review")
        self.finish.setFont(emphasis_font(self.finish, steps=2))
        self.finish.setDefault(True)
        self.finish.setAutoDefault(True)
        self.finish.setMinimumHeight(self.finish.sizeHint().height() + 8)
        self.finish.clicked.connect(self._finish)
        finish.addWidget(self.finish)
        #: "56 climb · 4 not climb", under Finish: what finishing will record.
        self.finish_counts = QLabel("")
        self.finish_counts.setFont(small_font(self.finish_counts))
        set_label_color(self.finish_counts, MUTED)
        finish.addWidget(self.finish_counts, 0, Qt.AlignmentFlag.AlignHCenter)
        top.addLayout(finish)
        column.addWidget(self.top)

        self.box = self._build_box()
        column.addWidget(self.box)

        self.undo_row = QWidget()
        undo = QHBoxLayout(self.undo_row)
        undo.setContentsMargins(0, 0, 0, 0)
        self.undo_text = QLabel("")
        set_label_color(self.undo_text, MUTED)
        self._plain_font = self.undo_text.font()
        undo.addWidget(self.undo_text)
        self.undo = QPushButton("Undo (⌘Z)")
        self.undo.clicked.connect(self._undo)
        undo.addWidget(self.undo)
        undo.addStretch(1)
        self.undo_row.setFixedHeight(self.undo.sizeHint().height())
        # Its text changes with every click, and a row asking for its full width grew the window.
        self.undo_row.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        column.addWidget(self.undo_row)

        self.montage = LiveMontage(
            aligned_key=ALIGNED_KEY.replace("label_check", keys, 1),
            look_key=LOOK_KEY.replace("label_check", keys, 1),
            context_key=CONTEXT_KEY.replace("label_check", keys, 1),
            export=export,
        )
        # The order is fixed: least typical first.
        self.montage.order_button.setVisible(False)
        self.montage.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.montage.canvas.installEventFilter(self)
        self.montage.clip_clicked.connect(self._on_click)
        self.montage.run_marked.connect(self._on_run)
        self.montage.preview_changed.connect(self._sync)
        self.montage.clip_activated.connect(self._on_activated)
        self.montage.clip_context_requested.connect(self._on_menu)
        self.montage.seen_changed.connect(lambda _n: self._sync())
        self.montage.pictures_changed.connect(self._sync)
        self.montage.exports.picking_changed.connect(lambda _on: self._sync())
        column.addWidget(self.montage, 1)

        self._keys = [
            _shortcut(self, "Ctrl+Z", self._undo),
            _shortcut(self, "Return", self._finish),
            _shortcut(self, "Enter", self._finish),
        ]
        self._sync()

    def _build_box(self) -> QFrame:
        """The finished cluster's box: "✓ Reviewed", what was reviewed, the precision large,
        the counts, Next cluster and Reopen review."""
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        box.setFrameShadow(QFrame.Shadow.Plain)
        row = QHBoxLayout(box)
        row.setContentsMargins(10, 6, 10, 6)
        self.reviewed = QLabel("✓ Reviewed")
        self.reviewed.setFont(emphasis_font(self.reviewed, steps=2))
        set_label_color(self.reviewed, READY)
        row.addWidget(self.reviewed)
        row.addSpacing(12)
        self.what = QLabel("")
        row.addWidget(self.what)
        self.box_label = QLabel("")
        self.box_label.setFont(emphasis_font(self.box_label, steps=0))
        row.addWidget(self.box_label)
        self.when = QLabel("")
        row.addWidget(self.when)
        row.addSpacing(16)
        self.precision = QLabel("")
        self.precision.setFont(emphasis_font(self.precision, steps=4))
        self.precision.setToolTip(
            f"Precision: the share of this {self._noun}'s clips that are its label, counted "
            "over every clip."
        )
        row.addWidget(self.precision)
        row.addSpacing(8)
        self.result = QLabel("")
        row.addWidget(self.result)
        row.addStretch(1)
        self.next = QPushButton(f"Next {self._noun}")
        self.next.clicked.connect(self.next_requested)
        row.addWidget(self.next)
        # "Reopen", not "again" or "start over": the removals are kept, and only finishing
        # again replaces the figure (review round 15, 2A).
        self.again = QPushButton("Reopen review")
        self.again.clicked.connect(self.again_requested)
        row.addWidget(self.again)
        return box

    # -- inputs ------------------------------------------------------------------------- #
    def show_cluster(self, node: int, label: str, colour=None) -> None:
        """A cluster to review: ``cluster #871`` of label path ``label``."""
        self.show_scope(f"cluster #{int(node)}", label, colour)

    def show_scope(self, title: str, label: str, colour=None) -> None:
        """A scope to review — ``title`` (*cluster #871*, a recording's name) — of label path
        ``label``."""
        self._label = label
        self.title.setText(title)
        self.what.setText(title)
        self.label_name.setText(display(label))
        self.box_label.setText(display(label))
        if colour is not None:
            set_label_color(self.label_name, colour)
            set_label_color(self.box_label, colour)
        self.montage.clear_anchor()
        self._last_click = None
        self._sync()

    def set_targets(self, targets: list[str]) -> None:
        """The other label paths a clip may be moved to, most clips first."""
        self._targets = list(targets)
        self._sync()

    def set_editable(self, reason: str) -> None:
        """``""`` where clips can be removed and a review finished; else why not (a snapshot,
        a finished run, a method that keeps no clip edits)."""
        self._reason = reason
        self._sync()

    def set_was(self, text: str) -> None:
        """An earlier figure that no longer stands ("was 97% · reviewing again"), or ``""``."""
        self.was.setText(text)
        self.was.setVisible(bool(text))

    def set_waiting(self, text: str) -> None:
        """Why the clips are not here yet (``RANKING``), or ``""`` once they are."""
        self._waiting = text
        self._sync()

    def set_undo(self, text: str | None) -> None:
        """What the last change did ("Removed 3 from climb"), or ``None`` with none to undo."""
        self._can_undo = bool(text)
        self._undo_line = text
        self._sync()

    def start_review(self) -> None:
        """Back to reviewing: the box hidden, Finish waiting for the scroll's end."""
        self._finished = False
        self._saving = False
        self.montage.clear_anchor()
        self._sync()

    def show_saving(self) -> None:
        self._saving = True
        self._sync()

    def show_result(
        self,
        precision: str,
        summary: str,
        next_node: int | str | None,
        *,
        removed: int,
        when: str = "",
    ) -> None:
        """The cluster is reviewed: ``precision`` ("95%") large, ``summary`` ("57 of 60 are
        climb · 3 removed"), the day it was reviewed, and — when one is left — the next
        cluster's button. ``removed`` is how many clips the review counted as not the label,
        which Reopen review's tooltip promises to keep."""
        self._finished = True
        self._saving = False
        self._removed = int(removed)
        self.precision.setText(precision)
        self.result.setText(summary)
        self.when.setText(f"· {when}" if when else "")
        if next_node is None:
            self.next.setText(f"Next {self._noun}")
        elif isinstance(next_node, int):
            self.next.setText(f"Next {self._noun}: #{next_node}")
        else:
            self.next.setText(f"Next {self._noun}: {next_node}")
        self._next_node = next_node
        self._sync()

    def refresh(self) -> None:
        """Re-read the tiles' marks — after the page changed them."""
        self._sync()

    def is_finished(self) -> bool:
        return self._finished

    def reached_end(self) -> bool:
        clips = len(self.montage.clips())
        return clips > 0 and self.montage.seen() >= clips

    def can_mark(self) -> bool:
        """Whether a click on a clip changes it now: an editable run, a cluster in review, no
        finish being saved and the clips ranked."""
        return (
            not self._reason
            and not self._finished
            and not self._saving
            and not self._waiting
        )

    def _busy_reason(self) -> str:
        """Why nothing may change for now, or ``""``: a finish being saved, clips ranking."""
        return SAVING if self._saving else self._waiting

    # -- state -------------------------------------------------------------------------- #
    def _faded(self, index: int) -> bool:
        clips = self.montage.clips()
        return 0 <= index < len(clips) and clips[index].faded

    def _hint(self, index: int) -> str:
        name = display(self._label)
        if self._faded(index):
            return f"Click: put back in {name} · double-click: open"
        return f"Click: remove from {name} · double-click: open"

    def _sync(self) -> None:
        name = display(self._label)
        reason = self._reason
        finished = self._finished
        self.montage.set_click_hint(self._hint if self.can_mark() else None)
        self.top.setVisible(not finished)
        self.undo_row.setVisible(not finished)
        self.box.setVisible(finished)
        # A run dragged in pick mode picks clips for an export: its bar says so, never this line.
        preview = self.montage.preview() if self.montage.click_layer() is None else None
        if preview is not None:
            # In the page's ink, not the hint's grey: it says what letting go is about to do.
            self.undo_text.setFont(emphasis_font(self.undo_text, steps=0))
            ink = QApplication.palette().color(QPalette.ColorRole.WindowText)
            set_label_color(self.undo_text, ink)
            self.undo_text.setText(preview_text(preview))
        else:
            self.undo_text.setFont(self._plain_font)
            set_label_color(self.undo_text, MUTED)
            picking = self.montage.click_layer() is not None
            self.undo_text.setText(
                PICKING_HINT
                if picking
                else self._busy_reason() or self._undo_line or reason or CLICK_HINT
            )
        busy = self._busy_reason()
        set_available(
            self.undo,
            not reason and not busy and self._can_undo,
            reason or busy or NOTHING_TO_UNDO,
        )
        if self.undo.isEnabled():
            self.undo.setToolTip(
                "Reverse the last change. ⌘Z again steps further back."
            )

        total = len(self.montage.clips())
        marked = sum(1 for c in self.montage.clips() if c.faded)
        seen = min(self.montage.seen(), total)
        self.next.setVisible(finished and self._next_node is not None)
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(seen)
        self.reached.setText(f"reached {seen:,} of {total:,}")
        self.finish_counts.setText(
            f"{total - marked:,} {name} · {marked:,} not {name}" if total else ""
        )
        if busy:
            set_available(self.finish, False, busy)
        elif reason:
            set_available(self.finish, False, reason)
        elif self.montage.click_layer() is not None:
            # Pick mode has the grid: Enter there is not a step of the review.
            set_available(self.finish, False, PICKING_FINISH)
        elif not total:
            set_available(self.finish, False, f"This {self._noun} has no clips.")
        else:
            waiting, failed = self.montage.unpictured()
            if failed:
                # Scrolling on would not help: say it first.
                set_available(self.finish, False, no_preview_text(failed))
            elif not self.reached_end():
                set_available(self.finish, False, SCROLL_FIRST)
            else:
                set_available(self.finish, not waiting, waiting_text(waiting))
            if self.finish.isEnabled():
                self.finish.setToolTip(
                    f"Save the review: every clip not removed counts as {name}, every removed "
                    f"or moved clip as not. Gives this {self._noun}'s exact precision."
                )
        # The app's style draws a default button with only a coloured border; filled with the
        # palette's accent once it can be pressed, it reads as the step to take. Accent, not
        # Highlight: on macOS Highlight is the pale selection blue, which white text washes out.
        accent = QApplication.palette().color(QPalette.ColorRole.Accent)
        fill = accent.darker(FINISH_ACCENT_DARKER)
        set_button_color(self.finish, fill if self.finish.isEnabled() else None)
        set_available(self.again, not reason, reason)
        if not reason:
            self.again.setToolTip(
                reopen_tip(self.precision.text(), self._removed, self._noun)
            )
        if self.next.isVisible():
            self.next.setToolTip("Open the next cluster nobody has reviewed yet.")

    # -- clicks ------------------------------------------------------------------------- #
    def _on_click(self, index: int, _shift: bool) -> None:
        """A click removes the clip (or puts a removed one back). A Shift-click with an anchor
        arrives as a run (:meth:`_on_run`); one without is a plain click."""
        clips = self.montage.clips()
        if not self.can_mark() or not 0 <= index < len(clips):
            return
        self._last_click = (index, time.monotonic())
        if clips[index].faded:
            self.put_back_requested.emit([index])
        else:
            self.take_out_requested.emit([index], None)

    def _on_run(self, indices: list, put_back: bool) -> None:
        """A drag let go, or a Shift-click: mark the run the preview showed, as one step."""
        if not self.can_mark() or not indices:
            return
        self._last_click = None
        if put_back:
            self.put_back_requested.emit(list(indices))
        else:
            self.take_out_requested.emit(list(indices), None)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt virtual
        """Note whether the grid's last mouse press was a double-click's second one, so
        :meth:`_on_activated` can tell a double-click from Space."""
        if watched is self.montage.canvas:
            kind = event.type()
            if kind == QEvent.Type.MouseButtonDblClick:
                self._double_click = True
            elif kind in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress):
                self._double_click = False
        return super().eventFilter(watched, event)

    def _on_activated(self, index: int) -> None:
        """A double-click's first press already marked the clip: retract that click. Only a
        real double-click: Space opens the clip and leaves the click before it standing.
        """
        last, self._last_click = self._last_click, None
        double, self._double_click = self._double_click, False
        if not double or last is None or last[0] != index or not self.can_mark():
            return
        interval = QApplication.doubleClickInterval() / 1000
        if time.monotonic() - last[1] <= interval and self._can_undo:
            self.undo_requested.emit()

    def _on_menu(self, index: int, where: QPoint) -> None:
        """Right-click a clip: move it to another label, or put it back."""
        menu = self.clip_menu(index)
        if menu is not None:
            from detail.gui.menus import open_once

            open_once(menu, where, wait=False)

    def clip_menu(self, index: int) -> QMenu | None:
        """The menu a right-click on clip ``index`` opens: *Move to <label>* for every other
        label and, on a removed or moved clip, *Put back in <label>*. Its entries are greyed
        with the reason when nothing may change."""
        if not 0 <= index < len(self.montage.clips()):
            return None
        name = display(self._label)
        tag = self.montage.clips()[index].tag
        menu = QMenu(self)
        actions = []
        for path in self._targets:
            if tag == f"→ {display(path)}":
                continue  # already moved there
            actions.append(
                menu.addAction(
                    f"Move to {display(path)}",
                    lambda p=path, i=index: self._move([i], p),
                )
            )
        if self._faded(index):
            if actions:
                menu.addSeparator()
            actions.append(
                menu.addAction(
                    f"Put back in {name}", lambda i=index: self._put_back([i])
                )
            )
        if not actions:
            return None
        if not self.can_mark():
            why = self._reason or reviewed_read_only(self._noun)
            for action in actions:
                action.setEnabled(False)
                action.setToolTip(why)
            menu.setToolTipsVisible(True)
        return menu

    def _move(self, indices: list, to_label: str) -> None:
        if self.can_mark() and indices:
            self._last_click = None
            self.take_out_requested.emit(list(indices), to_label)

    def _put_back(self, indices: list) -> None:
        if self.can_mark() and indices:
            self._last_click = None
            self.put_back_requested.emit(list(indices))

    def _undo(self) -> None:
        if self.can_mark() and self._can_undo:
            self._last_click = None
            self.undo_requested.emit()

    def _finish(self) -> None:
        if self.finish.isEnabled() and self.finish.isVisible():
            self.finish_requested.emit()
