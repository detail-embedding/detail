"""The Notes box: the user's own notes on the entry a page shows, saved as they type (§15).

One widget, placed by every entry page — the experiment overview, a recording, a processing or
analysis run, a dataset, a model, a distill or export run, a comparison
— below the entry's details and above its results. `tests/gui/test_notes.py` builds each of those
pages and fails on one without it, so a page added later is held to the same rule.

**Saved without a button.** A note is written a moment after typing stops, when the box loses
focus, and when it is hidden — which is what happens to it when the page is navigated away from,
rebuilt, or the window closes. There is nothing to forget to press, so there is nothing for
`editing.EditableSurface` to ask about either.

**One box per page, kept across rebuilds.** A run page rebuilds its body as recordings finish, and
a box rebuilt with it would take the cursor away from someone typing mid-sentence. `place_notes`
moves the page's one box to where the rebuilt body wants it instead.

**Opened from the shell, only where you are.** The box has no fold button of its own: the shell's
top-right bar carries **Notes** (``Notes ●`` when the page's entry has a saved note) beside
Export… / Version History / Info, bound to whichever page is showing (`app.MainWindow`, found
through :func:`notes_hub` and ``findChild``), and hidden on a page with no notes. **Folded on
every page and every view until opened there, and nothing remembers it**: opening the notes on
one page leaves every other page folded, and a page opened again (or in the next session) opens
folded. The user asked for exactly this in review round 4 ("I don't want the notes visibly to
propagate across pages"); the app-wide ``bands/notes`` setting it replaces opened a run's
landing page with its notes showing because they had been opened on INTERPRET. Folded by
default, because the window is the scarce thing on every page that carries one.

**A place, not a page.** A page that shows its result full-window (an analysis run with a view
open) moves its one box under the view's row while the view is open, so the button never opens a
box on a page that is hidden (`AnalysisRunDetailPage._place_notes_box`). The box is open or
folded *per place* (:meth:`NotesBox.set_place`: the landing page, or each view by name), so
opening it in INTERPRET and going back shows the landing page folded unless it was opened there.
Within one page the places are remembered while the page lives — a rebuild keeps them, since it
keeps the box.

Folded, a box is hidden outright and costs its page nothing — which is what a processing run's
results page needs to fit the window unscrolled (`tests/gui/test_run_page_fits.py`).
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.gui.lifetime import alive, cleared
from detail.gui.sizing import emphasis_font, lines
from detail.gui.status_color import ERROR, MUTED, set_label_color

#: How long typing may pause before the note is written, ms.
AUTOSAVE_MS = 800

#: How many lines of text the box shows before it scrolls.
LINES_SHOWN = 4

#: Grey text in an empty box. Short on purpose: it is the only thing saying what the box is
#: for, so it has to be read at a glance (asked for in these words, 2026-11-03).
PLACEHOLDER = "Add notes here (optional)"

#: After the header when a saved note exists, so a folded box says there is something in it.
HAS_NOTE_MARK = "\u25cf"


def button_text(has_note: bool) -> str:
    """The shell's Notes button: ``Notes``, or ``Notes ●`` when the page's entry has a note."""
    return "Notes" + (f" {HAS_NOTE_MARK}" if has_note else "")


class _Hub(QObject):
    """Announces every Notes box as it is built, so the shell can bind its button to a box a
    page builds after it was shown (a run page's body arrives off the UI thread)."""

    created = pyqtSignal(object)


_HUB: _Hub | None = None


def notes_hub() -> _Hub:
    """The one hub (made on first use, never at import: it is a QObject)."""
    global _HUB
    if _HUB is None:
        _HUB = _Hub()
    return _HUB


_TOOLTIP = (
    "Anything worth remembering about this: what it was for, what you noticed. "
    "Saved as you type, and kept with the experiment."
)


class NotesBox(QWidget):
    """An editable note on one entry, read from and autosaved to its ``notes.md``."""

    #: A save landed: whether a note exists may have changed (the ``●``).
    saved = pyqtSignal()
    #: Opened or folded — by the shell's button, or by moving to a place where it is the other.
    open_changed = pyqtSignal(bool)

    def __init__(self, experiment, kind: str, entry_id: str = "") -> None:
        """Folded; the shell's Notes button opens it, for the place it is in (:meth:`set_place`)."""
        super().__init__()
        self._experiment = experiment
        self.kind = kind
        self.entry_id = entry_id
        # Its own height and no more: on a page whose layout has slack above its trailing
        # stretch, a Preferred box took a share and pushed its title away from the text.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("")
        set_label_color(self.status, MUTED)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(PLACEHOLDER)
        self.editor.setToolTip(_TOOLTIP)
        self.editor.setFixedHeight(lines(LINES_SHOWN + 1))
        header = QHBoxLayout()
        self.title = QLabel("Notes")
        self.title.setFont(emphasis_font(self.title, steps=0))
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.status)
        layout.addLayout(header)
        layout.addWidget(self.editor)

        self._saved_text = ""
        try:
            self._saved_text = experiment.notes(kind, entry_id)
        except (OSError, ValueError) as exc:
            self._show_error(f"Could not read notes: {exc}")
        self.editor.setPlainText(self._saved_text)
        #: Where the box is shown (``""``: its page; ``"view:<name>"``: a full-window view) and the
        #: places it has been opened in. Never persisted: a new page opens folded.
        self._place = ""
        self._open_at: set[str] = set()
        # Only ever hidden here: shown before it has a parent, a widget becomes a window of its
        # own. Laid out on a shown page, a box nobody hid shows with it.
        self.setVisible(False)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(AUTOSAVE_MS)
        self._timer.timeout.connect(self.save)
        self.editor.textChanged.connect(self._on_changed)
        self._focus_out = _FocusOutSaves(self)
        self.editor.installEventFilter(self._focus_out)
        notes_hub().created.emit(self)

    def text(self) -> str:
        return self.editor.toPlainText()

    def has_unsaved(self) -> bool:
        return self.text() != self._saved_text

    def has_note(self) -> bool:
        """Whether a note is saved for this entry — what the ``●`` says."""
        return bool(self._saved_text.strip())

    def is_open(self) -> bool:
        return not self.isHidden()

    def set_open(self, open_: bool) -> None:
        """Show the box or fold it away entirely — here, in its current place, and nowhere else."""
        if open_:
            self._open_at.add(self._place)
        else:
            self._open_at.discard(self._place)
        self._apply()

    def place(self) -> str:
        return self._place

    def set_place(self, place: str) -> None:
        """The box now shows in ``place``: open if it was opened there, else folded. Called
        after a move, which re-parents the box and so hides it."""
        self._place = place
        self._apply()

    def _apply(self) -> None:
        open_ = self._place in self._open_at
        # Never shown without a parent: it would become a window of its own.
        if self.parentWidget() is not None or not open_:
            self.setVisible(open_)
        self.open_changed.emit(open_)

    def _on_changed(self) -> None:
        if self.has_unsaved():
            self.status.setText("Editing…")
            set_label_color(self.status, MUTED)
            self._timer.start()

    def save(self) -> None:
        """Write the note now if it changed. A failure is shown on the box, never raised: this
        runs from a timer and from hide, where an exception would have nowhere to go."""
        self._timer.stop()
        if not self.has_unsaved():
            return
        text = self.text()
        try:
            self._experiment.set_notes(self.kind, self.entry_id, text)
        except FileNotFoundError:
            self._show_error("Not saved: this entry no longer exists.")
            return
        except (OSError, ValueError) as exc:
            self._show_error(f"Not saved: {exc}")
            return
        self._saved_text = text
        self.status.setText(f"Saved {datetime.now():%H:%M}")
        set_label_color(self.status, MUTED)
        self.saved.emit()

    def _show_error(self, message: str) -> None:
        self.status.setText(message)
        set_label_color(self.status, ERROR)

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt override
        self.save()
        super().hideEvent(event)


class _FocusOutSaves(QObject):
    """Saves the box when its editor loses focus — clicking anywhere else on the page."""

    def __init__(self, box: NotesBox) -> None:
        super().__init__(box)
        self._box = box

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt override
        if cleared(self):
            return False
        if event.type() == QEvent.Type.FocusOut:
            self._box.save()
        return False


def place_notes(owner, body, experiment, kind: str, entry_id: str = "") -> NotesBox:
    """Add ``owner``'s one Notes box to the end of ``body``, building it the first time.

    A page that rebuilds its body calls this from the rebuild, at the point the box belongs;
    the box it built before is moved there rather than replaced, so its text, cursor and focus
    survive. The caller must not discard the box with the rest of the body.
    """
    box = getattr(owner, "_notes_box", None)
    if box is None or not alive(box):
        box = NotesBox(experiment, kind, entry_id)
        owner._notes_box = box
    else:
        body.removeWidget(box)
    body.addWidget(box)
    return box
