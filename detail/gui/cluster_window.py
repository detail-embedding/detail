"""One cluster's clips in a window of their own, beside INTERPRET (§22, the 11-9 meeting).

**Open in window** — on INTERPRET's label bar and on a cluster's right-click menu in the
tree — opens the cluster in a **non-modal** window: on top, INTERPRET's own label bar for **its**
cluster, and under it the clips playing in a live montage (``live_montage.LiveMontage``, the
page's own widget). INTERPRET stays fully usable beside it.

**One window per cluster** (:class:`ClusterWindows`, owned by the page): opening a cluster that has
a window raises it rather than opening a second; several clusters may be open at once, each in its
own. The windows are the page's children, so they close with it — and with the run, which takes
the page with it. Each window has a **number** — the lowest one free when it opened — shown in its
title ("① Behavior #1321 · climb") and on its cluster's block in INTERPRET's tree, so a window and
the cluster it shows can be matched at a glance; the marks follow windows as they open and close.
A reload that leaves the tree as it was re-reads each window's clips; a re-fit, which renumbers the
tree, closes every window and the page says so ("Closed 2 cluster windows: the run was
re-fitted"), since a node id would name another cluster now (``node_workspace.tree_identity``).

**The window's top is INTERPRET's label bar** (user, 2026-11-09): the same ``label_panel.LabelBar``,
built by the same function as the page's (``node_workspace.build_label_bar``) and filled by the
same ones, so the two cannot drift. Line 1: the window's number and the cluster ("① Behavior
#1395"), its label chip, and the montage toolbar at the right (**Most typical ▾ · Show ▾ · Size ·
Export ▾**). Line 2: the figures (clips · time · share of the run; the separation and the
concentration in their tooltip) at the left, the status at the right. Line 3: the label field,
**Label**, **Remove**, **Labels ▾** (opening with the page's recent labels, numbered, each greyed
with its reason) and **Undo (⌘Z)**. Only the page's *Open in window* is left off — this is the
window. Nothing sits between the bar and the grid but the pick bar while picking.
Nothing is labelled here: every request goes to the page's own write queue (``_label`` and its
neighbours, told which window asked), so the tree, the label bar and the rings refresh as from the
page, the window's bar and title follow from the page's refresh, and a prompt (rename, merge, a
refused name) opens over this window. The status is the page's — what the last write did and
whether it is saved, "Output up to date." included — and **Undo (⌘Z)** is one history for the page
and all its windows (``label_history.py``), so ⌘Z here undoes the last label change wherever it
was made. Keys 1–9 in a window apply that recent label to the window's cluster.
**INTERPRET stays put** (D1 B, 2026-11-09): neither focusing a window nor labelling from one moves
the page's selection — the number on the tree says where the window's cluster is. A snapshot or a
finished run greys the field, and Undo, with the page's reason.

**Enter never clicks a button here.** A ``QDialog`` makes every push button auto-default and the
first of them the default, so Enter in the label field — after the field had labelled the cluster
and enabled **Remove** — clicked Remove, and the label was gone as soon as it was written. The
``gui/dialog.Dialog`` base strips default and auto-default from every button of a dialog shown
non-modally (``requirements.a_non_modal_window_has_no_default_button``): Enter in the field
submits the field, and Enter anywhere else does nothing.

**The montage toolbar** is every clip grid's (2026-11-09, option A): **Most typical ▾ · Show ▾ ·
Size · Export ▾**, the montage's own (``live_montage.LiveMontage``), so the window and INTERPRET
and AUDIT cannot differ; here, as on INTERPRET, it sits on the label bar's first line. **Show ▾** holds the layers, the orientation and the
context, kept for every cluster window under the window's keys. **Export ▾** writes every clip as
the window shows it, or — **Pick clips to export…** — enters pick mode, a bar over the grid while
picking: a click picks or un-picks a clip, numbered in export order; an export that finishes
leaves it, and **Quit** or Esc leaves it without exporting (``montage_export.py``). Picks change nothing in the run. The
permanent Show row, Context box, picks bar and export bar the window had until then are gone:
the grid has their height.
"""

from __future__ import annotations

import numpy as np
from detail.gui.qt import (
    QByteArray,
    QDialogButtonBox,
    QLineEdit,
    QPoint,
    QSettings,
    QSize,
    Qt,
    is_deleted,
)

from detail.api.clip_tiles import TileLook
from detail.gui.dialog import Dialog
from detail.gui.live_montage import LiveMontage, MontageClip, context_span
from detail.gui.sizing import fit_to_screen, lines, set_available

#: The window's orientation (As recorded / Aligned), kept for every cluster window.
ALIGNED_KEY = "interpret/window_aligned"
#: Where a window's size and place are kept (``QWidget.saveGeometry``), shared by every one.
GEOMETRY_KEY = "interpret/cluster_window_geometry"
#: What the window shows, kept between windows: ``TileLook`` fields and the context.
LOOK_KEY = "interpret/cluster_window_look"
CONTEXT_KEY = "interpret/cluster_window_context"
#: The opening size, in lines: about 960 × 740 at 16 px a line, a comfortable share of a
#: 1440 × 900 laptop screen with INTERPRET still visible beside it.
_WIDE = 60
_TALL = 46
_SCREEN_SHARE = 0.85
#: How far each further window opens from the last, so two do not sit exactly on top.
_CASCADE_PX = 32
#: The shortest a tile loops for: one proxy frame (``node_workspace._MIN_LOOP_NS``).
_MIN_LOOP_NS = 1_000_000_000 // 15


def window_mark(number: int) -> str:
    """A window's number as its title and the tree show it: ① … ⑳, then (21)."""
    if 1 <= number <= 20:
        return chr(0x2460 + number - 1)
    return f"({number})"


class ClusterWindow(Dialog):
    """One cluster's label bar (INTERPRET's) and clips, in a non-modal window."""

    def __init__(self, interpret, node: int, parent=None, *, number: int = 1) -> None:
        """``interpret`` is the INTERPRET tab that opened it: its view, labels and write
        queue. ``number`` is the window's mark, in its title and on the tree."""
        super().__init__(parent)
        page = interpret
        self._page = page
        self._node = int(node)
        self.number = int(number)
        self._clips = None
        self._order = np.zeros(0, dtype=np.int64)
        self._catalog = None
        self._clip_reader = None
        self._request = 0
        #: The figures' clip count, or what the clips are waiting on.
        self._clip_text = ""
        #: The cluster's figures (``node_workspace.cluster_facts``), read once per load.
        self._facts = None
        layout = self.body()

        from detail.gui.montage_export import ExportSource
        from detail.gui.node_workspace import build_label_bar, undo_keys

        # A window opened before any look was kept starts without the trail: half a laptop
        # screen of trails read as noise (the window's first opening ticks).
        fresh = not isinstance(QSettings().value(LOOK_KEY), str)
        self.montage = LiveMontage(
            aligned_key=ALIGNED_KEY,
            look_key=LOOK_KEY,
            context_key=CONTEXT_KEY,
            export=ExportSource(
                clips=self.export_clips,
                experiment_dir=lambda: page._experiment.dir,
                origin=self._origin,
                name=self._export_stem,
                empty=lambda: "This cluster's clips are still being read.",
            ),
        )
        if fresh:
            self.montage.set_look(TileLook(mask_trail=False))
        self.montage.set_alignment(page._align_note)
        self.montage.set_points(page._points_note)
        self.montage.setMinimumHeight(lines(18))
        self.montage.order_changed.connect(self._on_order)
        self.montage.context_changed.connect(lambda _s: self._on_context())
        self.montage.clip_activated.connect(self._open_clip)
        self.montage.set_hover_text(self._hover_text)
        self.montage.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        #: INTERPRET's label bar, built by the page's own function — told which window asks,
        #: so its prompts open over it and INTERPRET's selection stays — with no *Open in
        #: window* (this is the window).
        self.bar = build_label_bar(page, self.montage, origin=self)
        page._label_bar.status_changed.connect(self.bar.show_status)
        self.bar.show_status(page._label_bar.status.text())
        self._undo_keys = undo_keys(self, page.undo, page.redo)
        layout.addWidget(self.bar)
        layout.addWidget(self.montage, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        self.set_buttons(buttons)
        self.setModal(False)
        # On `finished`, not in closeEvent: Esc (and every other way to the dialog's `done`)
        # closes a QDialog without a close event, which left the window on the tree's marks
        # and held in the page's list, its montage and tile cache with it.
        self.finished.connect(self._closed)
        self.resize(
            fit_to_screen(
                QSize(lines(_WIDE), lines(_TALL)),
                share=_SCREEN_SHARE,
                screen=self.screen(),
            )
        )
        self._restore_geometry()
        self.refresh_labels()
        self.load_clips()

    # -- what it is a window of ------------------------------------------------ #
    @property
    def node(self) -> int:
        return self._node

    def display_order(self) -> list[int]:
        """The clips, as indices into the cluster's clip table, in the window's order."""
        return [int(i) for i in self._order]

    def look(self) -> TileLook:
        """What the tiles show now, the orientation included."""
        return self.montage.look()

    def context_s(self) -> float:
        return self.montage.context_s()

    # -- geometry -------------------------------------------------------------- #
    def _restore_geometry(self) -> None:
        stored = QSettings().value(GEOMETRY_KEY)
        if isinstance(stored, QByteArray):
            self.restoreGeometry(stored)

    def cascade(self, step: int) -> None:
        """Open ``step`` windows along from the remembered place, so windows do not stack."""
        if step:
            self.move(self.pos() + QPoint(_CASCADE_PX * step, _CASCADE_PX * step))

    def _closed(self, _result: int = 0) -> None:
        """The window closed — by Close, the title bar, Esc or ``close_all``: keep its place,
        take its number off the tree and let go of its clips."""
        QSettings().setValue(GEOMETRY_KEY, self.saveGeometry())
        self._page._windows.forget(self)
        self.montage.set_source(None)

    # -- keys ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        key = event.key()
        if key == Qt.Key.Key_Escape and self.montage.exports.is_picking():
            # Esc leaves pick mode wherever the focus is; it closes the window only when not
            # picking (the montage's own Esc covers focus in the grid).
            self.montage.exports.stop_picking()
            event.accept()
            return
        if (
            Qt.Key.Key_1 <= key <= Qt.Key.Key_9
            and not isinstance(self.focusWidget(), QLineEdit)
            and not event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        ):
            self.bar.quick.press(key - Qt.Key.Key_1)
            event.accept()
            return
        super().keyPressEvent(event)

    # -- labels: filled by the page -------------------------------------------- #
    def refresh_labels(self) -> None:
        """Fill the label bar and the title from the page's labels and editability, by the
        page's own functions (``node_workspace.show_cluster_name`` and its neighbours).
        """
        from detail.gui.node_workspace import show_cluster_name, show_label_state

        page = self._page
        node = self._node
        bar = self.bar
        show_cluster_name(
            page, bar, node, f"{window_mark(self.number)} {page._cluster_name(node)}"
        )
        self._show_figures()
        rows = page._label_rows()
        field = bar.field
        field.set_labels(rows, page._colour_of_root)
        show_label_state(page, bar, node)
        reason = page._edit_reason() or "The run is still loading."
        field.set_editable(page._can_edit, reason)
        bar.quick.set_editable(page._can_edit, reason)
        bar.set_quick_labels(page._quick_labels(rows))
        bar.show_status(page._label_bar.status.text())
        own, inherited, _names, _choices = page._label_field_state(node)
        label = own or (f"inside “{inherited}”" if inherited else "not labelled")
        self.setWindowTitle(
            f"{window_mark(self.number)} {page._cluster_name(node)} · {label} · "
            f"{page._run.id}"
        )

    # -- the clips -------------------------------------------------------------- #
    def _ensure_catalog(self) -> None:
        if self._catalog is None:
            from detail.api import clip_tiles

            page = self._page
            self._catalog = clip_tiles.TileCatalog(
                page._experiment, page._tracking, page._embeds
            )

    def load_clips(self, *, keep_scroll: bool = False) -> None:
        """The cluster's clips: from the page's cache in memory, or ranked off the UI thread
        for a method that ranks on demand.

        ``keep_scroll``: the same clips read again (a reload of the same tree) — the tiles,
        the scroll, the clocks and the pictures stay while they are read."""
        page = self._page
        view = page._view
        if view is None:
            return
        self._ensure_catalog()
        self._request += 1
        request = self._request
        self._facts = None
        if not view.ranks_on_demand:
            self._show_clips(view.clips(self._node), keep_scroll=keep_scroll)
            return
        from detail.gui.workers import BackgroundCall

        run, node = page._run, self._node
        if not keep_scroll:
            # The window's clips go with its tiles, or the order and context switches draw
            # the old ones back while the new ones rank (`node_workspace._clear_clips`).
            self._clips = None
            self._order = np.zeros(0, dtype=np.int64)
            self.montage.set_clips([])
        # Said where INTERPRET says it: in the figures, not on a line over the grid.
        self._say(f"ranking this {page._terms.unit}'s clips…")
        reader = BackgroundCall(lambda: run.clips(node), receiver=self)
        self._clip_reader = reader
        reader.done.connect(
            lambda clips: request == self._request
            and self._show_clips(clips, keep_scroll=keep_scroll)
        )
        reader.failed.connect(
            lambda exc: request == self._request
            and self._say(f"the clips could not be ranked: {exc}")
        )
        reader.finished.connect(reader.deleteLater)
        reader.start()

    def _say(self, text: str) -> None:
        """What the clips are waiting on, at the head of the figures."""
        self._clip_text = text
        self._show_figures()

    def _show_figures(self) -> None:
        from detail.gui.node_workspace import cluster_facts, show_figures

        page = self._page
        if page._view is None:
            return
        if self._facts is None:
            self._facts = cluster_facts(page, self._node)
        show_figures(self.bar, self._clip_text, self._facts)

    def _show_clip_count(self) -> None:
        from detail.gui.node_workspace import clip_count_text

        self._say(clip_count_text(self._page, self._clips))

    def _show_clips(self, clips, *, keep_scroll: bool = False) -> None:
        self._clips = clips
        self._order = clips.order(self.montage.order(), seed=self._node)
        self._set_source(keep_pictures=keep_scroll)
        self._show_tiles(keep_scroll=keep_scroll)
        self._show_clip_count()

    def previews_changed(self) -> None:
        """The run's previews or masks were (re)built: the tiles read their pictures afresh,
        in place — order, scroll and clocks kept (``node_workspace._previews_arrived``).
        """
        self._catalog = None
        self._ensure_catalog()
        if self._clips is not None:
            self._set_source(keep_pictures=True)

    def _tile_clips(self) -> list:
        """The montage's clips in the window's order, widened by the context."""
        from detail.api import clip_tiles

        clips, context = self._clips, self.montage.context_s()
        out = []
        for i in self._order:
            rid = clips.recording_ids[int(clips.recording[i])]
            start, end = context_span(clips.start_ns[i], clips.end_ns[i], context)
            out.append(clip_tiles.TileClip(rid, int(clips.subject[i]), start, end))
        return out

    def _set_source(self, *, keep_pictures: bool = False) -> None:
        from detail.api import clip_tiles

        if self._catalog is not None and self._clips is not None:
            self.montage.set_source(
                clip_tiles.ClipTiles(self._catalog, self._tile_clips()),
                keep_pictures=keep_pictures,
            )

    def _show_tiles(self, *, keep_scroll: bool) -> None:
        """The tiles, with the AUDIT removals faded and tagged (as on INTERPRET)."""
        from detail.gui.label_clean import crossed_for, note_for, tag_for
        from detail.gui.node_workspace import _clock

        clips = self._clips
        if clips is None:
            return
        page = self._page
        names = page._view.recording_names
        context = self.montage.context_s()
        tiles = []
        for rank, i in enumerate(self._order, start=1):
            rid = clips.recording_ids[int(clips.recording[i])]
            start = int(clips.start_ns[i])
            low, high = context_span(start, clips.end_ns[i], context)
            edit = page._taken.get(page._clip_key_of(clips, int(i)))
            tiles.append(
                MontageClip(
                    max(high - low, _MIN_LOOP_NS),
                    rank,
                    f"{names.get(rid, rid)} {_clock(start)}",
                    tag=tag_for(edit) if edit is not None else None,
                    faded=edit is not None,
                    note=note_for(edit) if edit is not None else "",
                    crossed=crossed_for(edit),
                )
            )
        from detail.gui.node_workspace import marks_changed

        if not keep_scroll or marks_changed(self.montage.clips(), tiles):
            self.montage.set_clips(tiles, keep_scroll=keep_scroll)

    def refresh_marks(self) -> None:
        """The AUDIT removals changed: re-mark the tiles and recount them; nothing moves."""
        self._show_tiles(keep_scroll=True)
        if self._clips is not None:
            self._show_clip_count()

    def _on_order(self, key: str) -> None:
        self.montage.set_order(key)
        if self._clips is None:
            return
        self._order = self._clips.order(key, seed=self._node)
        self._set_source()
        self._show_tiles(keep_scroll=False)

    def _on_context(self) -> None:
        """**Show ▾**'s context changed: every tile plays its clip widened by it."""
        if self._clips is None:
            return
        self._set_source()
        self._show_tiles(keep_scroll=True)

    # -- the export -------------------------------------------------------------- #
    def export_clips(self) -> list:
        """The window's clips in its order as ``api.clip_montage.MontageClip`` — what
        **Export ▾** writes (``montage_export.ExportSource``), each its own span, ranked by
        its place in the order."""
        from detail.api import clip_montage as montage_api

        clips, page = self._clips, self._page
        if clips is None or page._view is None:
            return []
        names = page._view.recording_names
        out = []
        for rank, i in enumerate(self._order, start=1):
            rid = clips.recording_ids[int(clips.recording[i])]
            out.append(
                montage_api.MontageClip(
                    rank=rank,
                    recording_id=rid,
                    recording_name=names.get(rid, rid),
                    subject=int(clips.subject[i]),
                    embed_run_id=page._state.embed_run_for(rid),
                    start_t_ns=int(clips.start_ns[i]),
                    end_t_ns=int(clips.end_ns[i]),
                )
            )
        return out

    def _export_stem(self) -> str:
        page = self._page
        return page._names.get(str(self._node)) or f"{page._terms.unit}_{self._node}"

    def _origin(self):
        from detail.api.clip_montage import MontageOrigin

        page = self._page
        return MontageOrigin(
            run_id=str(page._run.id),
            node=self._node,
            name=page._names.get(str(self._node)),
            unit=page._terms.unit,
        )

    # -- opening one clip --------------------------------------------------------- #
    def _hover_text(self, index: int) -> str:
        clips = self._clips
        if clips is None or not 0 <= index < len(self._order):
            return ""
        from detail.gui.node_workspace import _clock

        i = int(self._order[index])
        rid = clips.recording_ids[int(clips.recording[i])]
        name = self._page._view.recording_names.get(rid, rid)
        start, end = int(clips.start_ns[i]), int(clips.end_ns[i])
        text = f"#{index + 1} · {name} · {_clock(start)} · {(end - start) / 1e9:.1f} s"
        exports = self.montage.exports
        if exports.is_picking():
            picks = exports.picks()
            text += (
                f" · picked #{picks.index(index) + 1}"
                if index in picks
                else " · not picked"
            )
        return text

    def _open_clip(self, index: int) -> None:
        """A tile double-clicked: the clip window, with previous / next through this
        window's order."""
        from detail.gui import clip_popup
        from detail.gui.node_workspace import _ORDER_NOTES, clip_sequence

        if self._clips is None or not 0 <= index < len(self._order):
            return
        sequence = clip_sequence(self._page, self._clips, self._order)
        clip_popup.open_clip_window(
            self,
            self._page._experiment,
            sequence[index],
            sequence=sequence,
            index=index,
            context_s=self.montage.context_s(),
            order_note=dict(_ORDER_NOTES).get(self.montage.order(), ""),
        )


class ClusterWindows:
    """The page's cluster windows, one per cluster, keyed by node (the page is one run), each
    numbered — the lowest number free when it opened — and marked with it on the page's tree.
    """

    def __init__(self, page) -> None:
        self._page = page
        self._open: dict[int, ClusterWindow] = {}

    def windows(self) -> list[ClusterWindow]:
        self._prune()
        return list(self._open.values())

    def get(self, node: int) -> ClusterWindow | None:
        self._prune()
        return self._open.get(int(node))

    def marks(self) -> dict[int, int]:
        """``{node: window number}`` for every open window."""
        return {node: window.number for node, window in self._open.items()}

    def open(self, node: int) -> ClusterWindow:
        """Open ``node``'s window, or raise the one already open for it."""
        existing = self.get(node)
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return existing
        taken = {window.number for window in self._open.values()}
        number = next(n for n in range(1, len(taken) + 2) if n not in taken)
        window = ClusterWindow(self._page, int(node), parent=self._page, number=number)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.cascade(len(self._open))
        self._open[int(node)] = window
        self._sync_marks()
        self._page._sync_undo()
        window.show()
        return window

    def forget(self, window: ClusterWindow) -> None:
        for node, held in list(self._open.items()):
            if held is window:
                del self._open[node]
        self._sync_marks()

    def refresh(self) -> None:
        """The labels or the editability changed on the page."""
        for window in self.windows():
            window.refresh_labels()

    def refresh_marks(self) -> None:
        """AUDIT's removals changed."""
        for window in self.windows():
            window.refresh_marks()

    def reloaded(self) -> None:
        """The page re-read the run and its tree is as it was: each window re-reads its
        clips. (A tree that changed closes them all instead: ``close_all``.)"""
        page = self._page
        for window in self.windows():
            window._catalog = None
            window._facts = None
            window.montage.set_alignment(page._align_note)
            window.montage.set_points(page._points_note)
            window.refresh_labels()
            window.load_clips(keep_scroll=True)

    def previews_changed(self) -> None:
        """The page's previews were (re)built: each window reads its pictures afresh, in
        place, so a clip that had no picture plays now and nothing else moves."""
        for window in self.windows():
            window.previews_changed()

    def close_all(self) -> int:
        """Close every window; how many there were."""
        windows = self.windows()
        for window in windows:
            window.close()
        return len(windows)

    def _sync_marks(self) -> None:
        self._prune()
        icicle = getattr(self._page, "_icicle", None)
        if icicle is not None:
            icicle.set_window_marks(self.marks())

    def _prune(self) -> None:
        for node, window in list(self._open.items()):
            if is_deleted(window):
                del self._open[node]
