"""**Export ▾** and pick mode: one export for every clip grid (§22, 2026-11-09, option A).

INTERPRET's montage, the pop-out cluster window and AUDIT's review grid each carry the same
toolbar — **Most typical ▾ · Show ▾ · Size · Export ▾** — because ``live_montage.LiveMontage``
builds it, this module's :class:`MontageExport` included. A page cannot have a montage without it,
and has nothing to add to it: it hands the montage an :class:`ExportSource` (the clips the grid
shows, in grid order) and the rest is here, once.

**Export ▾** writes what the grid shows — **Video of N clips…**, **GIF of N clips…**, **One file
per clip…**, **CSV list of clips…** — every clip, in grid order, drawn with **Show ▾**'s layers,
orientation and context (``show_menu.layers_of``, ``api.clip_montage.with_context``). A greyed
line under the entries says so in words ("Writes what the grid shows: video + masks, as recorded,
no context"), so the menu cannot promise one picture and write another. The CSV lists each clip's
own span: context is footage around a clip, not part of it. Every write runs off the UI thread
behind a bar with Cancel (``animation_export.run_export`` / ``busy.run_progress``), and the line
over the grid says what was written.

**Pick clips to export…** enters **pick mode**: a bar over the grid (:class:`PickBar`, about 30px,
shown only while picking) — "Pick mode · Click clips to pick them, in order · 3 picked [Pick all]
[Clear] [Export 3 ▾] [Quit]". It starts with **nothing** picked (user decision, 2026-11-09): a
click picks a clip and numbers it — 1, 2, 3, in the order clicked, the number painted in its
corner — and a second click un-picks it; a drag or a Shift-click picks a run, in grid order onto
the end (a drag that starts on a picked clip un-picks its run) — the montage's own run mechanism
(``LiveMontage.set_click_layer``: the same press, drag, preview and Shift as AUDIT's removals, with
this module deciding what a click means). **Pick all** picks every clip in grid order; **Clear**
un-picks them all. While picking, **Export ▾** and the bar's **Export N ▾** write the picks, in
pick order, and are greyed with "Pick at least one clip." while there are none. **An export of
the picks that finishes leaves pick mode** (user, 2026-11-09): the picks are what it was for. One
cancelled or failed leaves it as it was, picks and all, to try again. **Quit** or Esc leaves pick
mode without exporting — the button said *Done* until the same review, which read as if it would
finish the export. Either way the grid's own click is back unchanged (AUDIT: a click removes from
the label; INTERPRET: nothing), because the page's hint and connections were never touched. Picks are a
way of looking: they change nothing in the run — not the cluster, not AUDIT — and a reviewed AUDIT
cluster, read-only for removals, can still be picked from, since picking only exports.

A pick is held as its clip (recording, subject, start, end), not its place on the grid, so a new
order or a re-marked tile keeps it.

**Under the quick writers** (the user's choice, 2026-11-10, porting remote-detail's montage
window): **Export clips…** opens ``export_clips.MontageExportDialog`` — several looks at once,
each in its own subfolder, MP4 and/or GIF each at its own width and rate, a montage file and/or
one file per clip, with a preview and a size and time estimate — on the grid's clips, or the
picks while picking. **Export overlay…** writes the clips drawn over each other in one picture,
each turned to its heading and in its own colour (``ClusterMontage.export_overlay``): an MP4 or
GIF, or a PNG when every clip is one frame (a pose cluster's postures); the picks while
picking, else the grid's first :data:`~detail.api.clip_montage.OVERLAY_CLIPS`. **New Query from
N clips…** opens a behaviour Query's launch dialog with the clips as its examples, on the
embedding the clustering read (``api.clip_montage.query_examples``); nothing runs until Run is
pressed there. Every MP4 asks its codec — the save dialog's file types for **Video of N clips…**
and the overlay, the dialog's own picker in **Export clips…** — and says so when this machine
wrote another (``video_codec.with_codec_note``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QKeySequence, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
)

from detail.gui.sizing import emphasis_font, set_available

#: ``(kind, menu text)`` — what **Export ▾** writes, in menu order; ``{n}`` is the clip count.
KINDS: tuple[tuple[str, str], ...] = (
    ("video", "Video of {n} clips…"),
    ("gif", "GIF of {n} clips…"),
    ("files", "One file per clip…"),
    ("list", "CSV list of clips…"),
)
#: ``(kind, menu text)`` — the entries under the quick writers, after a separator.
MORE_KINDS: tuple[tuple[str, str], ...] = (
    ("clips", "Export clips…"),
    ("overlay", "Export overlay…"),
    ("query", "New Query from {n} clips…"),
)
_TIPS = {
    "video": "One video of the clips side by side, as the grid shows them.",
    "gif": "One GIF of the clips side by side, as the grid shows them.",
    "files": "A GIF of each clip, into a folder, named by its place and recording.",
    "list": (
        "The clips as a CSV list, one row each, in this order — each clip's own span (context "
        "is footage around a clip, not part of it). `analysis query --from-csv` reads it."
    ),
    "clips": (
        "Write the clips as videos or GIFs in as many looks as you tick: a montage of them, "
        "one file each, or both — with a preview and an estimate of the size and time."
    ),
    "overlay": (
        "The clips drawn over each other in one picture, each turned to face up by its "
        "heading and in its own colour, with a key: an MP4 or GIF of them playing together, "
        "or a PNG when every clip is one frame. Its look is its own, not Show ▾'s."
    ),
    "query": (
        "Open a new behaviour Query with these clips as its examples, on the embedding the "
        "clustering read. You review it and press Run."
    ),
}
#: Why **Export overlay…** waits: the overlay turns every clip by its heading.
OVERLAY_NEEDS_HEADING = (
    "The overlay turns every clip to face up by its heading, and this run cannot: "
)
PICK_TEXT = "Pick clips to export…"
PICK_TIP = (
    "Choose which clips to export, and in what order: click clips to pick them, numbered in "
    "the order clicked. Changes nothing in the run."
)
NOTHING_PICKED = "Pick at least one clip."
EVERY_CLIP_PICKED = "Every clip is picked, in the grid's order."
#: The pick bar's way out. Not "Done", which read as if it would export (user, 2026-11-09).
QUIT_TEXT = "Quit"
QUIT_TIP = "Leave pick mode without exporting (Esc)"
#: How strongly the pick bar is tinted with the theme's orange over the page (0–1).
BAR_TINT = 0.28


@dataclass(frozen=True)
class ExportSource:
    """What a page's grid exports. Every field is asked when an export happens, never kept.

    ``clips``: the grid's clips as ``api.clip_montage.MontageClip``, in grid order, one per tile,
    each its own span (the context is widened here) and ranked by its place on the grid.
    ``experiment_dir``: where the source video is read from. ``origin``: where the clips came
    from, for the CSV list and a Query's name (``api.clip_montage.MontageOrigin``). ``name``:
    the stem a suggested file name starts with ("climb"). ``empty``: why there is nothing to
    export while the grid is empty ("Open a cluster in the tree to export its clips").
    ``experiment``: the open experiment, which **New Query from N clips…** launches in.
    """

    clips: Callable[[], Sequence]
    experiment_dir: Callable[[], Path]
    origin: Callable[[], object]
    name: Callable[[], str]
    empty: Callable[[], str]
    experiment: Callable[[], object]


def describe(look, context_s: float, *, picking: bool) -> str:
    """``Writes what the grid shows: video + masks, as recorded, no context``."""
    from detail.gui.show_menu import context_words, describe_look

    shown = describe_look(look).replace(" · ", ", ")
    line = f"Writes what the grid shows: {shown}, {context_words(context_s)}"
    return f"{line}, in pick order" if picking else line


def pick_preview_text(preview) -> str:
    """What the pick bar says while a run preview is up: always about the export, never the
    label (``live_montage.RunPreview``)."""
    n = len(preview.indices)
    clips = f"{n:,} clip{'s' if n != 1 else ''}"
    verb = "picks" if preview.put_back else "un-picks"
    if preview.kind == "shift":
        return f"Shift-click {verb} {clips}"
    return f"Let go to {verb[:-1]} {clips} · Esc cancels"


def written_text(path) -> str:
    """What the line over the grid says once ``path`` is written."""
    from detail.gui.animation_export import written_note

    return written_note(Path(path))


def clip_key(clip) -> tuple:
    """A clip as a pick holds it: what it is, never where it sits on the grid."""
    from detail.core.utils import subject_key

    return (
        str(clip.recording_id),
        subject_key(clip.subject),
        int(clip.start_t_ns),
        int(clip.end_t_ns),
    )


def as_series_clip(clip):
    """``clip`` (an ``api.clip_montage.MontageClip``) with its subject a plain ``int``: a page
    that read it from an ``int64`` column hands an ``np.int64``."""
    import dataclasses

    import numpy as np

    from detail.core.utils import subject_key

    if isinstance(clip.subject, np.integer):
        return dataclasses.replace(clip, subject=subject_key(clip.subject))
    return clip


def query_prefill(clips, origin, title: str) -> dict:
    """What **New Query from N clips…** opens the Query launch with: ``seed`` (the behaviour,
    named for the cluster, and ``clips`` as its examples), ``recordings`` to check,
    ``selection`` (the embedding) and the ``note`` saying what is left to choose.

    The clips are the user's own — what they watched and chose — so a method given them is
    given what a user could give it ("Labels score a method; they never build one").

    Raises:
        ValueError: no clips.
    """
    from detail.api import clip_montage as montage_api

    clips = list(clips)
    examples = montage_api.query_examples(clips, name=origin.behavior_name)
    behavior = {
        "name": examples.behavior_name,
        "examples": [dict(c) for c in examples.examples],
        "negatives": [],
    }
    params: dict = {"behaviors": [behavior], "track_id": examples.track_id}
    left = [
        "add any other recordings to search",
        "check the method and its settings",
        "press Run",
    ]
    if examples.embed_run_id is None:
        embedding = (
            "read from several embedding runs "
            f"({', '.join(examples.embed_run_ids)}) — a Query searches one"
        )
        left.insert(0, "choose the embedding run")
    else:
        embedding = f"on {examples.embed_run_id}, the embedding the clustering read"
    note = (
        f"From the clips of {title}: {len(clips)} example clips of "
        f"'{examples.behavior_name}' {embedding}, with their "
        f"{len(examples.recording_ids)} recording(s) checked. Still to do: "
        + "; ".join(left)
        + "."
    )
    return {
        "seed": {"module_id": "query", "params": params},
        "recordings": list(examples.recording_ids),
        "selection": (
            {"embed_run_id": examples.embed_run_id} if examples.embed_run_id else {}
        ),
        "note": note,
    }


@dataclass(frozen=True)
class ExportJob:
    """What an export writes: ``clips`` widened by the context, in order; ``labels`` naming
    each by its own span; ``plain`` the clips as they are (the CSV's); the ``layers``.
    """

    clips: list
    labels: list[str]
    plain: list
    layers: object
    context_s: float
    picks: bool
    #: The pick session the picks were made in (``None`` for every clip): only that session
    #: ends when the export finishes.
    session: int | None = None


class ExportMenu(QPushButton):
    """**Export ▾** (the toolbar's) or **Export N ▾** (the pick bar's): the four writers, and on
    the toolbar's, **Pick clips to export…** and the greyed line saying what is written.
    """

    #: A writer was chosen: ``"video"``, ``"gif"``, ``"files"`` or ``"list"``.
    requested = pyqtSignal(str)
    #: **Pick clips to export…** was chosen.
    pick_wanted = pyqtSignal()

    def __init__(self, *, offers_pick: bool, parent=None) -> None:
        super().__init__("Export ▾", parent)
        self.setObjectName("montage_export")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._offers_pick = offers_pick
        self.menu = QMenu(self)
        self.menu.setToolTipsVisible(True)
        self.pick_action: QAction | None = None
        if offers_pick:
            self.pick_action = self.menu.addAction(PICK_TEXT)
            self.pick_action.setToolTip(PICK_TIP)
            self.pick_action.triggered.connect(lambda _c=False: self.pick_wanted.emit())
            self.menu.addSeparator()
        #: The writers, by kind (:data:`KINDS`, then :data:`MORE_KINDS`).
        self.actions_by_kind: dict[str, QAction] = {}
        for group in (KINDS, MORE_KINDS):
            for kind, _text in group:
                action = self.menu.addAction("")
                action.triggered.connect(
                    lambda _c=False, k=kind: self.requested.emit(k)
                )
                self.actions_by_kind[kind] = action
            self.menu.addSeparator()
        #: What a write draws, in words — greyed: a line to read, not to click.
        self.describe_action = self.menu.addAction("")
        self.describe_action.setEnabled(False)
        self.clicked.connect(self.open_menu)

    def open_menu(self) -> None:
        self.menu.exec(self.mapToGlobal(QPoint(0, self.height())))

    def sync(
        self,
        count: int,
        description: str,
        reason: str | None,
        *,
        overlay_reason: str | None,
    ) -> None:
        """``count`` clips would be written, as ``description`` says; ``reason``: why nothing
        can be (each writer greyed with it), or ``None``; ``overlay_reason``: why
        **Export overlay…** cannot, past that."""
        if not self._offers_pick:
            self.setText(f"Export {count} ▾")
        for kind, text in (*KINDS, *MORE_KINDS):
            action = self.actions_by_kind[kind]
            why = reason or (overlay_reason if kind == "overlay" else None)
            action.setText(text.format(n=f"{count:,}"))
            action.setEnabled(why is None)
            action.setToolTip(_TIPS[kind] if why is None else why)
        self.describe_action.setText(description)
        if not self._offers_pick:
            set_available(self, reason is None, reason or "")
            if reason is None:
                self.setToolTip(f"Export the {count:,} picked clips, in pick order")
        else:
            self.setToolTip("Export the clips as the grid shows them")


class PickBar(QFrame):
    """The bar over the grid while picking: "Pick mode · Click clips to drop or re-pick · 6
    picked  [Pick all] [Clear] [Export 6 ▾] [Quit]", tinted so it reads as a mode."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pick_bar")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 2, 6, 2)
        self.title = QLabel("Pick mode")
        self.title.setFont(emphasis_font(self.title, steps=0))
        row.addWidget(self.title)
        self.text = QLabel("")
        # It changes with every click, and a label asking for its full width grew the window.
        self.text.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row.addWidget(self.text, 1)
        self.pick_all = QPushButton("Pick all")
        self.pick_all.setToolTip("Pick every clip, in the grid's order.")
        self.clear = QPushButton("Clear")
        self.clear.setToolTip("Un-pick every clip, to start the order again.")
        self.export = ExportMenu(offers_pick=False)
        self.quit = QPushButton(QUIT_TEXT)
        self.quit.setToolTip(QUIT_TIP)
        for button in (self.pick_all, self.clear, self.export, self.quit):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(button)
        self._tint()

    def changeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        from PyQt6.QtCore import QEvent

        if event.type() == QEvent.Type.PaletteChange and not getattr(
            self, "_tinting", False
        ):
            self._tint()
        super().changeEvent(event)

    def _tint(self) -> None:
        """The theme's orange laid over the page's window colour, in either theme."""
        from detail.gui import plot_theme

        self._tinting = True
        try:
            base = QApplication.palette().color(QPalette.ColorRole.Window)
            orange = QColor(plot_theme.current().hue("orange"))
            mixed = QColor(
                round(base.red() + (orange.red() - base.red()) * BAR_TINT),
                round(base.green() + (orange.green() - base.green()) * BAR_TINT),
                round(base.blue() + (orange.blue() - base.blue()) * BAR_TINT),
            )
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Window, mixed)
            self.setPalette(palette)
            self.setAutoFillBackground(True)
        finally:
            self._tinting = False


class MontageExport(QObject):
    """A montage's **Export ▾**, its pick mode and its writers (module docstring)."""

    #: Pick mode was entered (``True``) or left.
    picking_changed = pyqtSignal(bool)

    def __init__(self, montage, source: ExportSource) -> None:
        super().__init__(montage)
        self._montage = montage
        self._source = source
        self._picking = False
        #: The picked clips' keys, in pick order, while picking.
        self._picks: list[tuple] = []
        #: Counts pick sessions, so an export finishing ends only the one it was made in.
        self._session = 0
        #: The grid's clips (keys) when picking started: a grid of other clips ends it.
        self._picked_from: frozenset = frozenset()
        #: ``(keys, {key: display index})`` of the grid as last read; ``None`` when stale.
        self._grid: tuple[list[tuple], dict[tuple, int]] | None = None
        #: ``(index, when, picks before)`` of the last plain click — what a double-click on
        #: the same clip takes back, so opening a clip does not pick or un-pick it.
        self._last_click: tuple[int, float, list] | None = None
        #: The open **Export clips…** dialog, kept in step with the grid while it is open.
        self._dialog = None

        self.button = ExportMenu(offers_pick=True)
        self.button.requested.connect(self.export)
        self.button.pick_wanted.connect(self.start_picking)
        self.button.menu.aboutToShow.connect(self.sync)
        montage.add_control(self.button)

        self.pick_bar = PickBar()
        self.pick_bar.setVisible(False)
        self.pick_bar.pick_all.clicked.connect(self.pick_all)
        self.pick_bar.clear.clicked.connect(self.clear)
        self.pick_bar.quit.clicked.connect(self.stop_picking)
        self.pick_bar.export.requested.connect(self.export)
        self.pick_bar.export.menu.aboutToShow.connect(self.sync)
        montage.above_grid.addWidget(self.pick_bar)
        #: What the last export wrote.
        self.written = QLabel("")
        self.written.setVisible(False)
        self.written.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        montage.above_grid.addWidget(self.written)

        self._escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), montage)
        self._escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._escape.setEnabled(False)
        self._escape.activated.connect(self._on_escape)

        montage.clips_changed.connect(self._grid_changed)
        montage.clip_activated.connect(self._on_activated)
        montage.preview_changed.connect(self.sync)
        for signal in (
            montage.context_changed,
            montage.aligned_changed,
            montage.show_button.layer_toggled,
            montage.look_changed,
        ):
            signal.connect(lambda *_a: self.sync())
        self.sync()

    # -- what the grid holds --------------------------------------------------------------- #
    def _clips(self) -> list:
        clips = [as_series_clip(c) for c in self._source.clips()]
        # Asked between a page's new order and its new tiles, the two can disagree for a moment:
        # nothing is exported from a grid that is not the one on screen.
        if len(clips) != len(self._montage.clips()):
            return []
        return clips

    def _keys(self) -> tuple[list[tuple], dict[tuple, int]]:
        if self._grid is None:
            keys = [clip_key(c) for c in self._clips()]
            self._grid = keys, {k: i for i, k in enumerate(keys)}
        return self._grid

    def _grid_changed(self) -> None:
        """New tiles. While picking: the same clips re-ordered or re-marked keep their picks;
        an empty grid (a cluster still being ranked) waits; **other clips** — another cluster
        on INTERPRET — end pick mode and say so, the picks cleared. Filtered silently, picks
        of one cluster were dropped and the next cluster's clips picked into the same export.
        """
        self._grid = None
        if self._picking:
            keys, place = self._keys()
            if not self._picked_from:
                self._picked_from = frozenset(keys)  # picking began on an empty grid
            elif keys and frozenset(keys) != self._picked_from:
                self.stop_picking()
                self._say("Pick mode ended: the grid shows other clips now.")
                return
            self._picks = [k for k in self._picks if k in place]
        self.sync()

    # -- pick mode ------------------------------------------------------------------------ #
    def is_picking(self) -> bool:
        return self._picking

    def picks(self) -> list[int]:
        """The picked clips as display indices, in pick order (empty when not picking)."""
        _keys, place = self._keys()
        return [place[k] for k in self._picks if k in place]

    def start_picking(self) -> None:
        """Pick mode, nothing picked yet: each click picks the next clip of the export."""
        from detail.gui.live_montage import ClickLayer

        self._picks = []
        self._picking = True
        self._session += 1
        self._picked_from = frozenset(self._keys()[0])
        self._last_click = None
        self.pick_bar.setVisible(True)
        self._escape.setEnabled(True)
        self._montage.set_click_layer(
            ClickLayer(
                hint=self._hint,
                is_off=lambda i: not self._is_picked(i),
                clicked=self._on_click,
                run=self._on_run,
                dropped=lambda i: False,
                number=self._number,
                shift_puts_back=True,
            )
        )
        self._montage.canvas.setFocus()
        self.sync()
        self.picking_changed.emit(True)

    def stop_picking(self) -> None:
        """Leave pick mode: the picks are forgotten and the grid's own click is back."""
        if not self._picking:
            return
        self._picking = False
        self._picks = []
        self._last_click = None
        self.pick_bar.setVisible(False)
        self._escape.setEnabled(False)
        self._montage.set_click_layer(None)
        self.sync()
        self.picking_changed.emit(False)

    def pick_all(self) -> None:
        """Every clip, in the grid's order."""
        keys, _place = self._keys()
        self._picks = list(keys)
        self._marks_changed()

    def clear(self) -> None:
        self._picks = []
        self._montage.clear_anchor()
        self._marks_changed()

    def _on_escape(self) -> None:
        if not self._montage._cancel_drag():
            self.stop_picking()

    def _is_picked(self, index: int) -> bool:
        keys, _place = self._keys()
        return 0 <= index < len(keys) and keys[index] in self._picks

    def _number(self, index: int) -> int | None:
        if not self._picking or not self._is_picked(index):
            return None
        keys, _place = self._keys()
        return self._picks.index(keys[index]) + 1

    def _hint(self, index: int) -> str:
        if self._is_picked(index):
            number = self._number(index)
            return f"Click: un-pick this clip (number {number}) · double-click: open"
        return f"Click: pick this clip, as number {len(self._picks) + 1} · double-click: open"

    def _on_click(self, index: int) -> None:
        keys, _place = self._keys()
        if not 0 <= index < len(keys):
            return
        self._last_click = (index, time.monotonic(), list(self._picks))
        key = keys[index]
        if key in self._picks:
            self._picks.remove(key)
        else:
            self._picks.append(key)
        self._marks_changed()

    def _on_run(self, indices: list, put_back: bool) -> None:
        """A drag or Shift run: pick the dropped ones (in grid order, onto the end) or drop the
        picked ones."""
        keys, _place = self._keys()
        chosen = [keys[i] for i in indices if 0 <= i < len(keys)]
        self._last_click = None
        if put_back:
            self._picks.extend(k for k in chosen if k not in self._picks)
        else:
            gone = set(chosen)
            self._picks = [k for k in self._picks if k not in gone]
        self._marks_changed()

    def _on_activated(self, index: int) -> None:
        """A double-click's first press was a click: take it back, so opening a clip leaves its
        pick as it was."""
        last, self._last_click = self._last_click, None
        if not self._picking or last is None or last[0] != index:
            return
        if time.monotonic() - last[1] <= QApplication.doubleClickInterval() / 1000:
            self._picks = last[2]
            self._marks_changed()

    def _marks_changed(self) -> None:
        self._montage.canvas.update()
        self.sync()

    # -- what an export writes ------------------------------------------------------------ #
    def job(self) -> ExportJob:
        """What an export would write now: every clip in grid order, or — while picking — the
        picks in pick order; drawn as **Show ▾** says, widened by its context."""
        from detail.api import clip_montage as montage_api
        from detail.gui.show_menu import layers_of

        clips = self._clips()
        if self._picking:
            _keys, place = self._keys()
            clips = [clips[place[k]] for k in self._picks if k in place]
        context = self._montage.context_s()
        return ExportJob(
            clips=montage_api.with_context(clips, context),
            labels=montage_api.montage_labels(clips, blind=False),
            plain=list(clips),
            layers=layers_of(self._montage.look()),
            context_s=context,
            picks=self._picking,
            session=self._session if self._picking else None,
        )

    def reason(self) -> str | None:
        """Why nothing can be exported now, or ``None``."""
        if not self._clips():
            return self._source.empty() or "There are no clips to export."
        if self._picking and not self._picks:
            return NOTHING_PICKED
        return None

    def overlay_reason(self) -> str | None:
        """Why **Export overlay…** cannot run now (past :meth:`reason`), or ``None``."""
        note = self._montage.alignment_note()
        if note is None:
            return None
        return OVERLAY_NEEDS_HEADING + note[0].lower() + note[1:]

    def sync(self) -> None:
        """Both menus and the pick bar say what an export would write now; an open **Export
        clips…** dialog follows."""
        reason = self.reason()
        overlay = self.overlay_reason()
        count = len(self._picks) if self._picking else len(self._clips())
        line = describe(
            self._montage.look(), self._montage.context_s(), picking=self._picking
        )
        self.button.sync(count, line, reason, overlay_reason=overlay)
        if self.button.pick_action is not None:
            empty = not self._clips()
            self.button.pick_action.setEnabled(not empty and not self._picking)
            self.button.pick_action.setToolTip(
                (self._source.empty() or "There are no clips to pick from.")
                if empty
                else (
                    "Picking now: the bar over the grid." if self._picking else PICK_TIP
                )
            )
        bar = self.pick_bar
        bar.export.sync(count, line, reason, overlay_reason=overlay)
        bar.text.setText(
            pick_preview_text(self._montage.preview())
            if self._picking and self._montage.preview() is not None
            else f"Click clips to pick them, in order · {len(self._picks):,} picked"
        )
        bar.text.setToolTip(bar.text.text())
        every = len(self._picks) == len(self._keys()[0])
        set_available(bar.pick_all, not every, EVERY_CLIP_PICKED)
        if not every:
            bar.pick_all.setToolTip("Pick every clip, in the grid's order.")
        set_available(bar.clear, bool(self._picks), "Nothing is picked.")
        if self._picks:
            bar.clear.setToolTip("Un-pick every clip, to start the order again.")
        if self._dialog is not None:
            self._dialog.refresh()

    def export(self, kind: str) -> None:
        """Ask where, then write ``kind`` (:data:`KINDS`) of the current job."""
        from detail.gui import animation_export as ax

        job = self.job()
        if self.reason() is not None or not job.clips:
            return
        owner, n = self._montage, len(job.clips)
        what = "picks" if job.picks else "montage"
        shown = describe(self._montage.look(), job.context_s, picking=job.picks)
        shown = shown.split(": ", 1)[1]
        if kind == "clips":
            self.open_dialog()
        elif kind == "overlay":
            self.export_overlay()
        elif kind == "query":
            self.use_as_query_examples()
        elif kind == "video":
            answer = ax.ask_mp4_path(owner, "Export video", self._file(what, ax.MP4))
            if answer is not None:
                self.write_montage(job, answer[0], codec=answer[1])
        elif kind == "gif":
            answer = ax.ask_gif_settings(
                owner,
                f"{n:,} clips side by side ({shown}), for one pass of the "
                "longest; each clip loops in it.",
            )
            if answer is None:
                return
            path = ax.ask_save_path(
                owner, "Export GIF", self._file(what, ax.GIF), ax.GIF
            )
            if path is not None:
                self.write_montage(job, path, max_width=answer[0], fps=answer[1])
        elif kind == "files":
            answer = ax.ask_gif_settings(
                owner,
                f"One GIF for each of the {n:,} clips ({shown}), named by "
                "its place and recording.",
            )
            if answer is None:
                return
            folder = QFileDialog.getExistingDirectory(
                owner, "Export the clips as GIFs into…"
            )
            if folder:
                self.write_files(job, folder, max_width=answer[0], fps=answer[1])
        elif kind == "list":
            chosen, _filter = QFileDialog.getSaveFileName(
                owner,
                "Export the clips as a CSV list",
                self._file("clips", "csv"),
                "CSV list (*.csv)",
            )
            if chosen:
                self.write_list(job, chosen)

    def _file(self, what: str, kind: str) -> str:
        from detail.gui.animation_export import safe_name

        return f"{safe_name(self._source.name())}_{what}.{kind}"

    def write_montage(
        self, job: ExportJob, path, *, max_width=None, fps=None, codec=None
    ):
        """``job`` as one montage at ``path`` (``.mp4`` / ``.gif``), off the UI thread behind
        a bar with Cancel; an mp4 in ``codec`` (``core.video_codecs``; the default when
        ``None``), the line over the grid saying so when this machine wrote another."""
        from detail.api import clip_montage as montage_api
        from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
        from detail.gui import animation_export
        from detail.gui.video_codec import with_codec_note

        experiment_dir = self._source.experiment_dir()
        codec = codec or DEFAULT_VIDEO_CODEC

        def work(progress):
            montage = montage_api.ClusterMontage(experiment_dir, job.clips)
            try:
                return montage.export(  # off-thread (run_progress)
                    path,
                    job.layers,
                    job.labels,
                    max_width=max_width,
                    fps=fps,
                    progress=progress,
                    codec=codec,
                )
            finally:
                montage.close()

        return animation_export.run_export(
            self._montage,
            work,
            on_written=lambda written: self._finished(
                job, with_codec_note(written_text(written), written, codec)
            ),
            label=f"Exporting {len(job.clips):,} clips…",
        )

    def write_files(self, job: ExportJob, folder, *, max_width: int, fps: float):
        """A GIF per clip of ``job`` into ``folder``, numbered in order, off the UI thread
        behind a bar with Cancel."""
        from detail.api import clip_montage as montage_api
        from detail.gui.busy import run_progress

        experiment_dir = self._source.experiment_dir()
        stems = [
            f"{n:02d}_{montage_api.clip_file_stem(c)}"
            for n, c in enumerate(job.plain, 1)
        ]

        def work(progress):
            montage = montage_api.ClusterMontage(experiment_dir, job.clips)
            try:
                return montage.export_clip_gifs(  # off-thread (run_progress)
                    Path(folder),
                    job.layers,
                    job.labels,
                    stems,
                    max_width=max_width,
                    fps=fps,
                    progress=progress,
                )
            finally:
                montage.close()

        def done(paths) -> None:
            self._finished(job, f"Wrote {len(paths):,} GIFs to {folder}")

        return run_progress(
            self._montage,
            work,
            done,
            label=f"Writing {len(job.clips):,} GIFs…",
            on_cancel=lambda: None,
        )

    def write_list(self, job: ExportJob, path):
        """``job``'s clips, each its own span, as a CSV list at ``path``, off the UI thread."""
        from detail.api import clip_montage as montage_api
        from detail.gui import animation_export

        path = Path(path)
        if path.suffix.lower() != ".csv":
            path = path.with_name(f"{path.name}.csv")
        experiment_dir, origin = self._source.experiment_dir(), self._source.origin()

        def work(progress):
            montage = montage_api.ClusterMontage(experiment_dir, job.plain)
            try:
                return montage.export_clip_list(  # off-thread (run_progress)
                    path, origin, progress=progress
                )
            finally:
                montage.close()

        return animation_export.run_export(
            self._montage,
            work,
            on_written=lambda written: self._finished(job, written_text(written)),
            label="Writing the clip list…",
        )

    def _finished(self, job: ExportJob, text: str) -> None:
        """An export finished: say what it wrote, and — when it wrote the picks — leave pick
        mode, since that is what the picks were for. Only success reaches here
        (``busy.run_progress``): a cancel or a failure leaves pick mode and its picks as they
        were."""
        self._say(text)
        # Only the session the picks were made in: a slow export finishing after Quit and a new
        # pick session would otherwise end the new one and throw its picks away.
        if job.picks and self._picking and job.session == self._session:
            self.stop_picking()

    def _say(self, text: str) -> None:
        self.written.setText(text)
        self.written.setToolTip(text)
        self.written.setVisible(True)

    # -- Export overlay… ------------------------------------------------------------------ #
    def overlay_clips(self, job: ExportJob) -> tuple[list, list[str]]:
        """``(clips, labels)`` an overlay of ``job`` draws: every pick, or the grid's first
        :data:`~detail.api.clip_montage.OVERLAY_CLIPS` (one picture of two thousand outlines
        is no picture)."""
        from detail.api.clip_montage import OVERLAY_CLIPS

        if job.picks:
            return list(job.clips), list(job.labels)
        return list(job.clips[:OVERLAY_CLIPS]), list(job.labels[:OVERLAY_CLIPS])

    def export_overlay(self, path=None, codec: str | None = None):
        """**Export overlay…**: where to write it — an MP4 in the codec its file type names,
        or a GIF; a PNG when every clip is one frame — then the export behind a bar with
        Cancel. ``path`` (and ``codec``) skip the dialog, for a test."""
        from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
        from detail.gui.animation_export import safe_name
        from detail.gui.video_codec import codec_of_filter, mp4_filters

        job = self.job()
        if self.reason() is not None or self.overlay_reason() is not None:
            return None
        clips, labels = self.overlay_clips(job)
        if not clips:
            return None
        still = all(int(c.start_t_ns) == int(c.end_t_ns) for c in clips)
        if path is None:
            kinds = (
                ["PNG image (*.png)"]
                if still
                else [*mp4_filters(), "GIF animation (*.gif)"]
            )
            chosen_path, chosen = QFileDialog.getSaveFileName(
                self._montage,
                "Export overlay",
                f"{safe_name(self._source.name())}_overlay",
                ";;".join(kinds),
            )
            if not chosen_path:
                return None
            path = Path(chosen_path)
            suffix = ".png" if still else (".gif" if "*.gif" in chosen else ".mp4")
            if path.suffix.lower() != suffix:
                path = path.with_name(f"{path.name}{suffix}")
            codec = codec_of_filter(chosen)
        return self.write_overlay(
            job, clips, labels, Path(path), codec=codec or DEFAULT_VIDEO_CODEC
        )

    def write_overlay(self, job: ExportJob, clips, labels, path: Path, *, codec: str):
        """The overlay of ``clips`` at ``path``, off the UI thread behind a bar with Cancel."""
        from detail.api import clip_montage as montage_api
        from detail.gui import animation_export
        from detail.gui.video_codec import with_codec_note

        experiment_dir = self._source.experiment_dir()

        def work(progress):
            montage = montage_api.ClusterMontage(experiment_dir, clips)
            try:
                return montage.export_overlay(  # off-thread (run_progress)
                    path, labels, progress=progress, codec=codec
                )
            finally:
                montage.close()

        return animation_export.run_export(
            self._montage,
            work,
            on_written=lambda written: self._finished(
                job, with_codec_note(written_text(written), written, codec)
            ),
            label="Exporting the overlay…",
        )

    # -- New Query from N clips… ---------------------------------------------------------- #
    def query_prefill(self) -> dict:
        """What **New Query from N clips…** opens the Query launch with (:func:`query_prefill`)
        — the picks while picking, else every clip; each its own span, never its context.
        """
        return query_prefill(
            self.job().plain, self._source.origin(), self._source.name()
        )

    def use_as_query_examples(self):
        """Open a new behaviour Query's launch dialog with the clips as its examples. Nothing
        runs until the user presses Run there. Returns the dialog (closed), for a test.
        """
        from detail.gui.dialog import Dialog
        from detail.gui.launch import LaunchAnalysisDialog

        if self.reason() is not None:
            return None
        prefill = self.query_prefill()
        dialog = LaunchAnalysisDialog(
            self._source.experiment(),
            self._montage.window(),
            seed=prefill["seed"],
            recordings=prefill["recordings"],
            selection=prefill["selection"],
            note=prefill["note"],
        )
        if dialog.exec() == Dialog.DialogCode.Accepted and dialog.run_handle:
            self._say(
                f"Launched {dialog.run_handle.id}: it is under Analysis while it runs."
            )
        return dialog

    # -- Export clips… (export_clips.MontageExportDialog reads these, live) -------------- #
    def open_dialog(self):
        """Open **Export clips…** on the grid's clips — preset to the picks while picking —
        or bring the open one forward."""
        from detail.gui.export_clips import MontageExportDialog

        dialog = self._dialog
        if dialog is None:
            dialog = MontageExportDialog(self, selected_only=self._picking)
            dialog.finished.connect(lambda _code: self._forget_dialog(dialog))
            self._dialog = dialog
        else:
            dialog.set_selected_only(self._picking)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _forget_dialog(self, dialog) -> None:
        if self._dialog is dialog:
            self._dialog = None

    def montage(self):
        return self._montage

    def members(self) -> list:
        """Every clip of the grid, in its order, each its own span."""
        return self._clips()

    def shown_clips(self) -> list:
        return self._clips()

    def selected_clips(self) -> list:
        """The picks, in pick order, while picking; else none."""
        if not self._picking:
            return []
        clips = self._clips()
        _keys, place = self._keys()
        return [clips[place[k]] for k in self._picks if k in place]

    def pick_session(self) -> int | None:
        """The pick session now, or ``None`` when not picking — what an export of the picks
        hands back to :meth:`note_written`, so only its own session ends."""
        return self._session if self._picking else None

    def context_s(self) -> float:
        return self._montage.context_s()

    def layers(self):
        """What the tiles draw now, as an export's layers (``show_menu.layers_of``)."""
        from detail.gui.show_menu import layers_of

        return layers_of(self._montage.look())

    def trail_skeleton(self):
        """``(names, pairs)`` of the keypoints, once a tile has shown them, else ``None``."""
        return self._montage.skeleton()

    def trail_names(self) -> list[str] | None:
        points = self._montage.look().trail_points
        return None if points is None else list(points)

    def title_text(self) -> str:
        return str(self._source.name())

    def origin(self):
        return self._source.origin()

    def experiment_dir(self) -> Path:
        return Path(self._source.experiment_dir())

    def note_written(self, text: str, *, session: int | None = None) -> None:
        """What an export from the dialog wrote, on the line over the grid; an export of the
        picks made in ``session`` ends that pick session, as a quick one does."""
        self._say(text)
        if session is not None and self._picking and session == self._session:
            self.stop_picking()
