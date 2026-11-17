"""INTERPRET for a clustering run: pick a cluster, watch every clip of it, label it (§22).

The page a user lands on to label what a fit found. On the left, **the filter dock**
(``cluster_filters.FilterDock``: two tabs — *Filters*, with the match key and the count line
at its top, then Location, Time in session, Video and a line per metadata column; and
*Differences*, with the saved screens, the ticked screen's scale and its strictness), which
folds to a strip. The open tab decides what the tree's colour means (``_shade_tree``). On the
right, top to bottom:

- **The tree** (``NodeIcicle``), under the labels in use with their time
  (``match_key.TreeKeyRow``) and the zoom buttons. It opens at the height that shows **every
  level**; when that would leave the montage under two rows of tiles, the rows shrink instead of
  scrolling (``_fit_tree``). A drag on its handle is kept and wins from then on. Either way it
  gives height back, down to ``TREE_MIN_PX``, before the montage drops below its two rows, and
  the montage's own floor is one row — so the column always fits the window. A filter never
  hides a cluster: with the Filters tab open every block is shaded by the share of its
  clips that match, on a fixed 0–100% scale (the ramp is the key at the top of that tab), the
  clusters a ticked screen kept are framed in the colour of their side, and a block's label
  colour moves to a strip along its top; with the Differences tab open the blocks take the
  ticked screen's statistic instead. "n of N clips match" and Reset all are in the dock, under the match key.
- **The label bar** (``label_panel.LabelBar``, review round 9, option A), between the tree and
  the clips, in three lines (laptop layout and user decision A, 2026-11-09). Line one: the open
  cluster's name and its label as a coloured chip, and at the right end the montage's toolbar
  (the match key while a filter is on, the order, **Show ▾**, Size, **Export ▾**). Line two: its
  figures — clips, time and share of the run, shortened with "…" to the room left; their
  tooltip has the whole line, the separation against the run's median, and where the cluster is
  concentrated when it beats its floor — and at its right what the last write did and whether it
  is saved. Once AUDIT has reviewed every clip of it, the audit line with what to do. Line three:
  the **label field** (type a name and press Enter; **Labels ▾** lists every label with its
  colour and time, and Rename / Merge from there) with **Label**, **Remove** and **Labels ▾** —
  which opens with the run's recent labels, numbered, keys 1–9 with the tree in focus — then
  **Undo (⌘Z)** and **Open in window**.
- **The montage** (``live_montage.LiveMontage``): every clip of the selected cluster, all
  playing, most typical first (or least typical, or random). Hovering a tile pauses the rest; a
  clip outside the filters still plays, its animal drawn grey. A clip removed from its label on
  AUDIT stays where the clustering put it, **faded** with a "not climb" tag and a bold ✕ (or,
  moved, an "→ climb › left" tag) — this page shows the raw clustering — and right-clicking it
  offers **Put back in climb**, written through the same queue as a label. The label bar counts
  them ("48 clips · 6 removed"). It has no controls row of its own: its toolbar — the
  order menu, **Show ▾** (Video, Masks, Mask trail, Keypoints, As recorded / Aligned, the
  context), Size, **Export ▾** and, while a filter is on, the match key — is on the label bar's
  first row, and the grid starts right under it. **Export ▾** writes every clip the grid shows, in
  its order and as it shows them, or the clips picked in pick mode (``montage_export.py``, the
  same on every clip grid); the shell's Export… lists no montage exports of its own.
- **Cluster windows** (``cluster_window.py``): *Open in window* (or a cluster's right-click
  menu) opens the cluster in a non-modal window of its own — one per cluster — whose top is this
  page's label bar, built by the same function (:func:`build_label_bar`) and filled by the same
  ones (:func:`show_cluster_name`, :func:`show_figures`, :func:`show_label_state`), writing
  through this page's queue; only *Open in window* is left off.

**Nothing here waits on a click.** Opening the page reads the node cache once, off the UI
thread; after that, selecting a cluster and filtering are arithmetic over arrays in memory
(``api/node_view.py``, ``api/cluster_filter.py``), and the montage paints from this machine's
video previews (``api/clip_tiles.py``) on its own worker. A label shows at once and is written in
the background, one at a time and in order (``_label`` / ``_write_next``).

**When previews are missing** the montage area alone is blocked, with a progress bar, the time
left and Cancel, waiting only on this run's recordings (``gui/preview_jobs.py`` builds them, and
moves them to the front of its queue); the tree, the labels and the filters stay usable.
"""

import dataclasses
import math
import time

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCompleter,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.api import cluster_filter
from detail.core.progress import remaining_seconds
from detail.gui import node_cache_jobs, plot_theme, preview_jobs
from detail.gui.audit_table import audited_colour
from detail.gui.busy import run_busy, run_progress
from detail.gui.cluster_filters import (
    Column,
    FilterBar,
    MAP_PX,
    RecordingChoice,
    NodeEffect,
    ScreenChoice,
    ScreenLevel,
    DIFFERENCES_TAB,
    effect_shades,
    breakdown,
    kept_marks,
    numbers_text,
    panel_name,
)
from detail.gui.cluster_find import FindClustersBar, common_cluster
from detail.gui.grip_splitter import GripSplitter
from detail.gui.label_panel import (
    LabelBar,
    LabelField,
    row_path,
    label_choices,
    label_paths,
    label_rows,
    merge_targets,
    root_slots,
)
from detail.gui.live_montage import (
    ALIGN_CHECKING,
    LiveMontage,
    MontageClip,
    context_span,
)
from detail.gui.match_key import TreeKeyRow
from detail.gui.match_key import one_sided
from detail.gui.node_audit import NodeAuditBlock
from detail.gui.node_icicle import TreePane, format_time, row_px
from detail.gui.sizing import ResizeHandle, lines, set_available
from detail.gui.workers import BackgroundCall
from detail.storage.behavior_output import PATH_SEPARATOR

#: Where a tree height the user **dragged** to is kept between pages and sessions. Until they
#: drag, the tree takes the height that shows every level. Renamed from
#: ``interpret/tree_height`` so a height an earlier build remembered (it remembered a default,
#: not only a drag) does not cut the tree off at two rows.
TREE_HEIGHT_KEY = "interpret/tree_height_dragged"
#: Interpret's montage orientation (As recorded / Aligned), kept for the page.
ALIGNED_KEY = "interpret/montage_aligned"
#: The context **Show ▾** plays either side of each of Interpret's clips, kept for the page.
CONTEXT_KEY = "interpret/montage_context"
#: What **Show ▾** draws on Interpret's montage (``show_menu``); the older app-wide Show masks
#: setting (``live_montage.MASKS_KEY``) is read once to start from.
LOOK_KEY = "interpret/montage_look"
#: Tile rows the montage keeps when the tree takes the height for every level.
MONTAGE_MIN_ROWS = 2
#: Where the filter dock's width and open/shut are kept (``panes/interpret_filters/``).
FILTER_SPLIT_KEY = "interpret_filters"
#: The keys the dock used before every side panel folded one way: carried over once, removed.
_LEGACY_FILTER_WIDTH_KEY = "splitters/interpret_filters_width"
_LEGACY_FILTER_FOLDED_KEY = "splitters/interpret_filters_folded"
#: How to use the tree: the key row's tooltip.
TREE_KEYS_TEXT = (
    "Click a block to watch its clips · wheel to zoom, drag to pan, Shift+wheel to scroll "
    "down the levels · double-click a block to label it · ↑ ↓ ← → walk the tree · L label · "
    "U remove label · 1–9 a recent label"
)
#: The tree never drags shorter than this: a few rows of blocks.
TREE_MIN_PX = 120
#: The shortest a tile loops for: one proxy frame, so a one-frame clip still has a clock.
_MIN_LOOP_NS = 1_000_000_000 // 15


def tree_identity(view) -> tuple | None:
    """What makes a node id mean one cluster: the fit the clips were built from (its file and
    size, ``NodeView.built_from``) and the tree's merges, or ``None`` with no tree. A re-fit
    changes it; rebuilding the clips from the same fit, or re-reading the run, does not.
    """
    if view is None:
        return None
    import hashlib

    tree = view.tree
    merges = np.ascontiguousarray(np.asarray(tree.children, dtype=np.int64))
    built = view.built_from
    return (
        built.get("fit"),
        built.get("fit_size"),
        int(tree.n_leaves),
        hashlib.sha1(merges.tobytes()).hexdigest(),
    )


def marks_changed(shown, tiles) -> bool:
    """Whether ``tiles`` draw anything ``shown`` (a montage's tiles now) does not — a fade, a
    tag, a pick — leaving out the filter match, which the page sets on its own.

    A label write re-reads AUDIT's removals and re-marks every montage open on the run: this
    page's and each cluster window's. Re-setting tiles that have not changed relaid and
    repainted every one of them, in every window, for a label written in another
    (2026-11-09); the tiles are now handed over only when a mark changed."""
    from dataclasses import replace

    if len(shown) != len(tiles):
        return True
    return [replace(t, matching=True) for t in shown] != [
        replace(t, matching=True) for t in tiles
    ]


def undo_keys(widget: QWidget, undo, redo) -> list:
    """⌘Z and ⇧⌘Z on ``widget`` and everything in it: INTERPRET's page and each cluster window
    bind the page's one history to them."""
    out = []
    for keys, slot in (
        (QKeySequence.StandardKey.Undo, undo),
        (QKeySequence.StandardKey.Redo, redo),
    ):
        shortcut = QShortcut(keys, widget)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(slot)
        out.append(shortcut)
    return out


def build_label_bar(page, montage, *, origin=None) -> LabelBar:
    """The label bar for ``page``'s labels, ``montage``'s toolbar at its first row's right.

    INTERPRET's bar and every cluster window's top are built here, so the two cannot differ;
    the page adds *Open in window* to its own afterwards. Every request goes to the page's write
    queue; ``origin`` is the cluster window asking (its prompts open over it, and INTERPRET's
    selection stays), ``None`` for the page.
    """
    bar = LabelBar(page._terms)
    field = bar.field
    field.label_requested.connect(lambda node, name: page._label(node, name, origin))
    field.rename_wanted.connect(lambda name: page._rename_label(name, origin))
    field.merge_wanted.connect(lambda name: page._merge_label(name, origin))
    field.label_picked.connect(page._select_label)
    bar.quick.label_requested.connect(
        lambda node, name: page._label(node, name, origin)
    )
    bar.undo.clicked.connect(page.undo)
    # The montage's toolbar sits at the right of the bar's first row, not on a row of its own
    # above the grid (laptop layout, 2026-11-09, decision 1 C): 31px of every window went to it.
    bar.add_trailing_widget(montage.detach_controls())
    return bar


def cluster_facts(page, node: int) -> tuple[list[str], str, str]:
    """``(figures, separation, concentration)`` for ``node``: the time and share on the bar's
    figures line, and the two sentences its tooltip adds (2026-11-09: on the line they were
    what got cut at a laptop's width). The concentration is said only when the cluster is more
    concentrated than a random draw of its size reaches (``NodeView.distribution``'s floor).
    """
    view = page._view
    summary = view.summary(node)
    parts = [format_time(summary.time_ns), f"{100 * summary.share:.1f}% of the run"]
    separation = ""
    if np.isfinite(summary.separation):
        median = view.separation_median()
        separation = (
            f"Separation {summary.separation:.2f} (run median {median:.2f})."
            if np.isfinite(median)
            else f"Separation {summary.separation:.2f}."
        )
    best = None
    for dimension, groups in page._groups.items():
        found = view.distribution(node, groups)
        if found.flagged and (best is None or found.ratio / found.floor > best[0]):
            best = (found.ratio / found.floor, dimension, found)
    concentration = ""
    if best is not None and int(node) != int(view.root):
        _lift, dimension, found = best
        top = found.values.index(found.top)
        concentration = (
            f"Concentrated: {dimension} {found.top} · {found.ratio:.1f}× the run. "
            f"{100 * found.node[top]:.0f}% of this {page._terms.unit}'s time "
            f"is {dimension} {found.top}, against "
            f"{100 * found.run[top]:.0f}% of the run's: "
            f"{found.ratio:.1f}×, where a random draw of its size reaches "
            f"{found.floor:.1f}× (95th percentile)."
        )
    return parts, separation, concentration


def show_cluster_name(page, bar: LabelBar, node: int, name: str) -> None:
    """The bar's first line: ``name`` and ``node``'s label chip — its own, or a parent's
    drawn ``inside “label”`` — and, under the figures, where its own label came from when a
    difference screen chose the cluster (:func:`origin_line`)."""
    own = page._names.get(str(node), "")
    owner = page._owner(node)
    bar.show_cluster(
        name,
        own or (page._names[str(owner)] if owner is not None else ""),
        page._colour_of(node) if owner is not None else None,
        inherited=not own and owner is not None,
    )
    bar.set_origin(origin_line(page._origins.get(str(node))) if own else "")


def origin_line(origin: dict | None) -> str:
    """The label bar's line for a label a difference screen chose (``run_labels.LabelOrigin``
    as stored): *From screen cmp_01 (drug vs vehicle · paired): more under drug than vehicle,
    12.4× · kept · family-wise p 0.02, own p <0.01, q 0.40* -- *· inside kept cluster #1375* /
    *· above kept cluster #1375* when the labelled cluster is not the kept one itself, *kept at
    p ≤ 0.2* when the screen was read looser than its floor (§22). ``""`` for no origin.
    """
    from detail.analysis.comparison_report import kept_at_text

    if not origin:
        return ""
    comparison = origin.get("comparison", "?")
    screen = origin.get("screen", "")
    head = f"From screen {comparison}" + (f" ({screen})" if screen else "")
    relation = origin.get("relation", "")
    kept = origin.get("kept", "?")
    if relation == "under":
        where = f"inside kept cluster #{kept}"
    elif relation == "above":
        where = f"above kept cluster #{kept}"
    else:
        where = "kept"
    where += kept_at_text(origin)
    text = origin.get("text", "")
    line = f"{head}: {text} · {where}" if text else f"{head} · {where}"
    numbers = numbers_text(
        *(_stored_number(origin.get(k)) for k in ("family_p", "own_p", "q"))
    )
    return f"{line} · {numbers}" if numbers else line


def _stored_number(text) -> float | None:
    """A number as the state stores it (a string), or ``None`` for a blank."""
    if text is None or str(text).strip() == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def show_figures(bar: LabelBar, clip_text: str, facts) -> None:
    """The bar's figures — ``clip_text`` ("48 clips · 6 removed"), then the time and share —
    with the separation and the concentration in their tooltip (:func:`cluster_facts`).
    """
    parts, separation, concentration = facts
    bar.set_figures(" · ".join(([clip_text] if clip_text else []) + list(parts)))
    bar.figures.set_extra_tooltip(
        "\n".join(t for t in (separation, concentration) if t)
    )


def show_label_state(page, bar: LabelBar, node: int) -> None:
    """The bar's label field and recent labels, for ``node``."""
    own, inherited, names, choices = page._label_field_state(node)
    bar.field.show_node(node, own, inherited, names, choices)
    bar.quick.show_node(node, own, choices)


def clip_count_text(page, clips) -> str:
    """ "48 clips · 6 removed": ``clips``' count, and how many AUDIT took out of the label."""
    n = len(clips)
    text = f"{n:,} clip{'s' if n != 1 else ''}"
    out = (
        sum(1 for i in range(n) if page._clip_key_of(clips, i) in page._taken)
        if page._taken
        else 0
    )
    if out:
        text += f" · {out:,} removed"
    return text


class InlineLabelEditor(QFrame):
    """A one-line label editor that opens where the user double-clicked. Enter saves; Esc and
    clicking away cancel."""

    submitted = pyqtSignal(str)

    def __init__(self, current: str, known: list[str], unit: str, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        box = QHBoxLayout(self)
        box.setContentsMargins(4, 4, 4, 4)
        box.addWidget(QLabel("Label:"))
        self.edit = QLineEdit(current)
        self.edit.setPlaceholderText(f"name this {unit}")
        completer = QCompleter(known, self.edit)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.edit.setCompleter(completer)
        self.edit.setMinimumWidth(lines(16))
        self.edit.returnPressed.connect(self._submit)
        box.addWidget(self.edit)
        # Typing a name that exists joins that label (§22) — said here, before Enter, so it
        # never reads as making a second label of the same name.
        self._known = {name.casefold(): name for name in known}
        self.hint = QLabel("")
        box.addWidget(self.hint)
        self.edit.textChanged.connect(self._sync_hint)

    def _sync_hint(self, text: str) -> None:
        match = self._known.get(text.strip().casefold())
        self.hint.setText(f"adds to “{match}”" if match else "")

    def focus(self) -> None:
        self.edit.setFocus()
        self.edit.selectAll()

    def _submit(self) -> None:
        name = self.edit.text().strip()
        self.close()
        if name:
            self.submitted.emit(name)


def _clock(t_ns: int) -> str:
    seconds = int(t_ns // 1_000_000_000)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class _PreviewWait:
    """This page's wait on the preview queue: which of the run's recordings are still missing,
    and the bar's fraction over them alone."""

    def __init__(self, queue, recording_ids, run_recordings=()) -> None:
        self.queue = queue
        self.waiting = list(recording_ids)
        # The build in hand finishes before any recording waited for starts (one build at a
        # time), so when it is one of this run's it is counted too, whatever it still lacks: a
        # page reopened after its proxy landed and while its masks build read it as missing
        # only masks, and called this run's own build "the build in hand" at 0 of 1, its bar
        # empty, beside a status bar at 60% for the same build (2026-11-10).
        if queue.current in set(run_recordings) and queue.current not in self.waiting:
            self.waiting.insert(0, queue.current)
        self.total = len(self.waiting)
        self.started = time.monotonic()
        self.failed: list[str] = []
        # The time left is measured from where the wait found the queue, not from 0: joined
        # part-way through a build, the whole fraction so far was not bought by this clock.
        self._from = self.fraction()

    def fraction(self) -> float:
        done = self.total - len(self.waiting)
        in_hand = (
            self.queue.permille / 1000.0 if self.queue.current in self.waiting else 0.0
        )
        return (done + in_hand) / max(self.total, 1)

    def message(self) -> str:
        done = self.total - len(self.waiting)
        head = (
            f"Building video previews for this run: {done} of {self.total} recordings"
        )
        if self.queue.current in self.waiting:
            return f"{head} · {self.queue.phase or 'starting'}"
        if self.queue.current is not None:
            return f"{head} · waiting for the build in hand to finish"
        return head

    def eta(self) -> float | None:
        span = 1.0 - self._from
        if span <= 0:
            return None
        return remaining_seconds(
            (self.fraction() - self._from) / span, time.monotonic() - self.started
        )


def _node_effects(ranking, record: dict) -> dict:
    """Every candidate cluster's row of a screen, by node: effect, clears, kept, sentence —
    through `comparison_report.suggestion`, the same words the Comparisons page prints.
    """
    from detail.analysis import comparison_report, node_measures

    kept = {item.measure_key for item in ranking.kept_items()}
    out = {}
    for item in ranking.contrasts:
        node = node_measures.node_of(item.measure_key[1])
        if node is None:
            continue
        out[node] = NodeEffect(
            effect=getattr(item, "effect", None),
            clears=ranking.clears_floor(item),
            kept=item.measure_key in kept,
            text=comparison_report.suggestion(item, ranking, record),
            pair=getattr(item, "pair", None),
            family_p=getattr(item, "family_p", None),
            own_p=getattr(item, "own_p", None),
            q=getattr(item, "q", None),
            statistic=getattr(item, "statistic", None),
        )
    return out


def _screen_levels(ranking, view) -> tuple[ScreenLevel, ...]:
    """The screen read at each strictness the dock's slider offers (§22): the floor at that
    level, the clusters kept there and their leaves — all lookups over the one ranking, so
    the slider never re-ranks. Empty for a screen with no floor."""
    from detail.analysis import compare
    from detail.api import comparison as comparison_api

    if ranking.floor is None:
        return ()
    out = []
    for alpha in compare.STRICTNESS_LEVELS:
        kept = comparison_api.kept_nodes_of(ranking, alpha)
        out.append(
            ScreenLevel(
                alpha=alpha,
                floor=ranking.floor_at(alpha),
                kept=tuple(int(n) for n in kept),
                leaves=view.leaves_under_nodes(kept),
                clearing=len(ranking.clearing_at(alpha)),
            )
        )
    return tuple(out)


def _effect_caption(ranking, record: dict) -> str:
    """The effect axis's words: ``Less ↔ more under drug``, ``Less ↔ more with dose``, or
    ``Differs across session`` for a profile, whose axis is a size (§22)."""
    if ranking.profile:
        across = (
            record.get("design_column")
            or ", ".join(record.get("group_by") or [])
            or "the groups"
        )
        return f"Differs across {across}"
    if ranking.design == "trend":
        return f"Less ↔ more with {record.get('design_column') or 'the column'}"
    levels = ranking.levels or ()
    if len(levels) == 2:
        return f"Less ↔ more under {levels[1]} (against {levels[0]})"
    return "Less ↔ more"


class NodeInterpretTab(QWidget):
    """The INTERPRET tab for a pose or temporal clustering run."""

    #: A label landed in the output: AUDIT and BROWSE re-read it.
    changed = pyqtSignal()
    #: Finish or Reopen landed.
    status_changed = pyqtSignal()
    #: The selected cluster for BROWSE's ethogram lane: ``(view, node, name, colour)``, or
    #: ``None`` for the root (all of the time, so a lane would say nothing).
    lane_changed = pyqtSignal(object)

    def __init__(self, run, experiment, terms) -> None:
        super().__init__()
        self._run = run
        self._experiment = experiment
        self._terms = terms
        self._view = None
        self._state = None
        self._names: dict[str, str] = {}
        self._selected: int | None = None
        #: Label edits waiting to be written, in order — each a call on the run.
        self._pending: list = []
        self._writer: BackgroundCall | None = None
        self._can_edit = False
        #: **Finish** was clicked: nothing labels until the run is finished (or it failed), and
        #: the finalize waits for the writes still queued (``_finalize_when_saved``).
        self._finishing = False
        self._finalizing = False
        self._inline_editor = None
        #: AUDIT's per-cluster rows, by node — which clusters the tree rings as audited.
        self._audited: dict[int, dict] = {}
        self._audit_reader: BackgroundCall | None = None
        self._audit_stale = False
        #: Clips removed from a label in review, by clip key (``label_clean.clip_key``): faded
        #: on the montage, never hidden — INTERPRET shows what the clustering gave each cluster.
        self._taken: dict[tuple, dict] = {}
        #: Bumped by every full reload, so a read started before one is dropped.
        self._generation = 0
        #: ``{recording_id: {column: value}}`` and each column's inferred type.
        self._metadata: dict[str, dict[str, str]] = {}
        self._column_types: dict[str, str] = {}
        #: ``{recording_id: (first_ns, last_ns)}``: each recording's whole video, what Time in
        #: session is a fraction of (``cluster_filter.session_spans``, read off the UI thread).
        self._spans: dict[str, tuple[int, int]] = {}
        self._groups: dict[str, dict[str, str]] = {}
        self._selection = cluster_filter.Selection()
        self._shares: np.ndarray | None = None
        #: The ticked screen, when its effects are known: what the tree is coloured by
        #: while the Differences tab is open, and whose kept clusters it frames on Filters.
        self._effect_screen: ScreenChoice | None = None
        #: The selected cluster's clips, in the montage's order, and each condition's pass.
        self._clips = None
        self._order = np.zeros(0, dtype=np.int64)
        self._passed: dict[tuple, np.ndarray] = {}
        self._clip_reader: BackgroundCall | None = None
        self._clip_request = 0
        self._tracking: dict[str, str] = {}
        self._catalog = None
        self._wait: _PreviewWait | None = None
        self._local_queue = None
        self._root_clips = None
        self._header_parts: list[str] = []
        #: The open cluster's clip count once its clips are in ("214 clips"), or what is pending.
        self._clip_text = ""
        #: Labels used on this page, most recent first: the bar's quick buttons lead with them.
        self._recent: list[str] = []
        self._missing_previews: list[str] = []
        self._missing_masks: list[str] = []
        self._mask_queue = None
        #: ``{recording_id: embed_run_id}``, and whether those embeddings' keypoints can be
        #: drawn / align the clips (``None``) or why not — read with the run, off the thread.
        self._embeds: dict[str, str] = {}
        self._points_note: str | None = ALIGN_CHECKING
        self._align_note: str | None = ALIGN_CHECKING
        from detail.gui.cluster_window import ClusterWindows

        #: The run's cluster windows, one per cluster (``cluster_window.py``).
        self._windows = ClusterWindows(self)
        from detail.gui.label_history import LabelHistory

        #: Every label change made here or in a window, for ⌘Z (``label_history.py``).
        self._history = LabelHistory()
        #: What the loaded tree is (``tree_identity``): a reload that changes it closes every
        #: window and clears the history, since a node id no longer names the same cluster.
        self._tree_key = None

        layout = QVBoxLayout(self)

        # §22: clips built from another fit are said to be, with the way to fix them. The one
        # row above the workspace that comes and goes: shown only while the clips are stale.
        self._stale_row = QWidget()
        stale_row = QHBoxLayout(self._stale_row)
        stale_row.setContentsMargins(0, 0, 0, 0)
        self._stale = QLabel("")
        self._stale.setWordWrap(True)
        stale_row.addWidget(self._stale, 1)
        self._rebuild = QPushButton("Rebuild clips")
        self._rebuild.clicked.connect(self._rebuild_cache)
        stale_row.addWidget(self._rebuild)
        self._stale_row.setVisible(False)
        layout.addWidget(self._stale_row)

        # §22: a capped fit is identical to a full one everywhere else on this screen, so if it is
        # not said here it is not said anywhere. Hidden, not blank, for a fit that compared all.
        self._subsample = QLabel("")
        self._subsample.setWordWrap(True)
        self._subsample.setVisible(False)
        layout.addWidget(self._subsample)

        self._missing = QWidget()
        missing = QVBoxLayout(self._missing)
        missing.addWidget(
            QLabel(
                f"This run was fitted before {terms.unit} clips were cached, so it has no "
                "node cache yet. Building it reads each recording once and can take a few "
                "minutes."
            )
        )
        build = QPushButton("Build the node cache")
        self._build = build
        build.clicked.connect(self._rebuild_cache)
        missing.addWidget(build)
        missing.addStretch()
        self._missing.setVisible(False)
        layout.addWidget(self._missing)

        # A build in flight, whether this page started it or an earlier one did (it outlives the
        # page: `gui/node_cache_jobs.py`). Shown in place of the tree, which is being rebuilt.
        self._building = QWidget()
        building = QVBoxLayout(self._building)
        self._build_status = QLabel("")
        self._build_status.setWordWrap(True)
        building.addWidget(self._build_status)
        self._build_bar = QProgressBar()
        self._build_bar.setRange(0, 1000)
        building.addWidget(self._build_bar)
        building.addWidget(
            QLabel(
                "This keeps going if you leave the page; come back to it at any time."
            )
        )
        building.addStretch()
        self._building.setVisible(False)
        layout.addWidget(self._building)

        # [filter dock | the tree, the label bar and the montage], side by side. The dock folds
        # by its grip to a strip reading "Filters · n active"; its width and whether it is shut
        # are kept between pages and sessions.
        self._split = GripSplitter(Qt.Orientation.Horizontal, key=FILTER_SPLIT_KEY)
        main = QWidget()
        rows = QVBoxLayout(main)
        rows.setContentsMargins(0, 0, 0, 0)

        undo_keys(self, self.undo, self.redo)
        # Not a widget: the filters' state, whose dock is what goes on screen.
        self._filters = FilterBar(self)
        self._filters.selection_changed.connect(self._on_selection)
        self._filters.screen_requested.connect(self._run_screen)
        self._filters.tab_changed.connect(lambda _tab: self._shade_tree())
        #: The saved difference screens being read off the UI thread, and the id of a screen
        #: just run from the dock, to tick when its leaves land.
        self._screens_reader: BackgroundCall | None = None
        self._screens_stale = False
        self._tick_screen: str | None = None
        self._dock = self._filters.dock
        self._split.addWidget(self._dock)
        self._split.addWidget(main)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.set_foldable(0, panel_name(cluster_filter.Selection()))

        tree_pane = QWidget()
        tree_box = QVBoxLayout(tree_pane)
        tree_box.setContentsMargins(0, 0, 0, 0)
        # Row 1: the label swatches, Find clusters, then the zoom buttons. How to use the tree
        # is the row's tooltip. Find sits in the row's spare width rather than a row of its
        # own (user, 2026-11-16: ~580px of it went unused at 1440 wide), and wraps onto a
        # second line only while more ids are listed than fit.
        key_row = QWidget()
        key_line = QHBoxLayout(key_row)
        key_line.setContentsMargins(0, 0, 0, 0)
        self._key_row = TreeKeyRow()
        key_line.addWidget(self._key_row)
        # Ids typed here are ringed in the tree, and it zooms to the smallest cluster holding
        # them all (`gui/cluster_find.py`). It takes the row's spare width, so the buttons keep
        # their own (`requirements.a_button_in_a_row_keeps_its_own_width`).
        #: The tree's view before Find first zoomed it (`NodeIcicle.view_state`): what an
        #: emptied list goes back to. ``None`` while nothing is being found.
        self._before_find = None
        self._find = FindClustersBar(unit=terms.unit, units=terms.units)
        self._find.found_changed.connect(self._show_found)
        self._find.rings_changed.connect(self._icicle_rings)
        self._find.picked.connect(self.select)
        key_line.addWidget(self._find, 1)
        self._zoom_in = QPushButton("+")
        self._zoom_in.setToolTip("Zoom in (or scroll the wheel over the tree)")
        self._zoom_in.clicked.connect(lambda: self._icicle.zoom_toward_selection(0.5))
        self._zoom_out = QPushButton("−")
        self._zoom_out.clicked.connect(lambda: self._icicle.zoom_by(2.0))
        self._fit = QPushButton("Whole tree")
        self._fit.clicked.connect(lambda: self._icicle.fit())
        for button in (self._zoom_out, self._zoom_in, self._fit):
            key_line.addWidget(button)
        key_row.setToolTip(TREE_KEYS_TEXT)
        tree_box.addWidget(key_row)
        self._tree = TreePane()
        self._icicle = self._tree.icicle
        self._icicle.node_picked.connect(self.select)
        self._icicle.label_wanted.connect(self._edit_label_inline)
        self._icicle.unlabel_wanted.connect(lambda node: self._label(node, ""))
        self._icicle.zoomed.connect(lambda _node: self._sync_zoom_buttons())
        self._icicle.window_wanted.connect(self.open_window)
        tree_box.addWidget(self._tree, 1)
        self._tree_holder = tree_pane
        dragged = self._remembered_tree_height()
        #: Whether the user has dragged the tree's height; until then it fits every level.
        self._tree_dragged = dragged is not None
        self._icicle.set_fit_rows(not self._tree_dragged)
        self._tree_holder.setFixedHeight(dragged or lines(16))
        # The tree drags to trade height with the montage, and does not fold: the page is the
        # tree and its clips, and a fold header above it was one more control to read (user
        # decision 2026-11-07). A handle, never a vertical splitter (`test_workspace_parity`).
        rows.addWidget(tree_pane)
        #: The drag handle under the tree.
        self._tree_handle = ResizeHandle()
        self._tree_handle.dragged.connect(self._resize_tree)
        rows.addWidget(self._tree_handle)

        from detail.gui.montage_export import ExportSource

        self._montage = LiveMontage(
            aligned_key=ALIGNED_KEY,
            look_key=LOOK_KEY,
            context_key=CONTEXT_KEY,
            export=ExportSource(
                clips=self.export_clips,
                experiment_dir=lambda: self._experiment.dir,
                origin=self._origin,
                name=self._export_stem,
                empty=lambda: f"Open a {self._terms.unit} in the tree to export its clips",
            ),
        )
        # A cluster window's top is built by the same call (`build_label_bar`).
        self._label_bar = build_label_bar(self, self._montage)
        self._label_field = self._label_bar.field
        rows.addWidget(self._label_bar)
        # Its floor is one row of tiles, not the rows `_fit_tree` aims for: a floor the column
        # cannot honour is what pushed the grid's bottom out of the window (see `_fit_tree`).
        self._montage.setMinimumHeight(self._montage.height_for_rows(1))
        self._montage.order_changed.connect(self._on_order)
        self._montage.context_changed.connect(lambda _s: self._on_context())
        self._montage.clip_activated.connect(self._play_clip)
        self._montage.clip_context_requested.connect(self._on_clip_menu)
        self._montage.set_hover_text(self._hover_text)
        self._open_window = QPushButton("Open in window")
        self._open_window.setToolTip(
            "This cluster's clips in a window of their own, beside this page: label it there, "
            "and choose what the tiles show and what to export, as here."
        )
        self._open_window.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self._open_window.clicked.connect(lambda: self.open_window())
        set_available(self._open_window, False, "The run is still loading.")
        self._label_bar.add_action_widget(self._open_window)
        #: Where the open cluster is concentrated, said in the figures' tooltip only — a chip
        #: of its own on row 1 left the figures 56px at 1440×847 (user, 2026-11-09).
        self.concentration = ""
        #: The open cluster's separation against the run's median, for the figures' tooltip.
        self._separation = ""
        self._audit_block = NodeAuditBlock()
        self._audit_block.move_wanted.connect(self._label)
        self._audit_block.remove_wanted.connect(lambda node: self._label(node, ""))
        self._audit_block.halves_wanted.connect(self._look_at_halves)
        self._label_bar.lines.addWidget(self._audit_block)
        self._preview_note = QWidget()
        note_row = QHBoxLayout(self._preview_note)
        note_row.setContentsMargins(0, 0, 0, 0)
        self._preview_text = QLabel("")
        self._preview_text.setWordWrap(True)
        note_row.addWidget(self._preview_text, 1)
        self._build_previews = QPushButton("Build previews")
        self._build_previews.clicked.connect(self._wait_for_previews)
        note_row.addWidget(self._build_previews)
        self._preview_note.setVisible(False)
        self._montage.header_lines.addWidget(self._preview_note)
        #: What happened to the page's windows (a re-fit closed them); an export says what it
        #: wrote on the montage's own line (``montage_export``).
        self._written = QLabel("")
        self._written.setVisible(False)
        self._montage.header_lines.addWidget(self._written)
        rows.addWidget(self._montage, 1)
        self._main = main
        self._fit_pending = False
        main.installEventFilter(self)
        self._label_bar.installEventFilter(self)
        layout.addWidget(self._split, 1)
        self._restore_dock()
        self._sync_zoom_buttons()

        bottom = QHBoxLayout()
        bottom.addStretch()
        self._finish = QPushButton("Finish")
        self._finish.clicked.connect(self._finish_run)
        bottom.addWidget(self._finish)
        self._reopen = QPushButton("Reopen")
        self._reopen.clicked.connect(self._reopen_run)
        bottom.addWidget(self._reopen)
        layout.addLayout(bottom)

        # Hidden until loaded: shown under the loading overlay, the empty tree and montage
        # read as a broken page for the second the load takes.
        self._split.setVisible(False)
        self._job = None
        self._apply_editability()
        self.reload()

    # --- loading -------------------------------------------------------------------------

    def reload(self) -> None:
        """Read the state, the node cache, the metadata and what previews are missing, off the
        UI thread.

        **A build in flight is checked first, synchronously** (a dict lookup). Checked only
        after the off-thread load, a page reopened mid-build showed the empty workspace under a
        "Loading…" overlay for a second and only then switched to the build's progress, which
        reads as the build having vanished and come back.
        """
        self._generation += 1
        job = node_cache_jobs.running(self._run._run_dir())
        if job is not None:
            self._attach(job)
            return
        run, experiment = self._run, self._experiment

        def work():
            from detail.api import clip_tiles
            from detail.storage import metadata as metadata_store

            state = run.state()
            labels = run.node_names()
            try:
                view = run.node_view()
            except FileNotFoundError:
                view = None
            metadata = {
                rid: {k: str(v or "") for k, v in row.items() if k != "recording_id"}
                for rid, row in experiment.recording_metadata.items()
            }
            note = getattr(run, "subsample_note", None)
            loaded = {
                "state": state,
                "view": view,
                "labels": labels,
                "metadata": metadata,
                "types": metadata_store.infer_column_types(metadata),
                "subsample": note() if callable(note) else None,
                "audited": {},
                "taken": {},
                "tracking": {},
                "missing": {"proxy": [], "masks": []},
            }
            if view is None:
                return loaded
            loaded["audited"], loaded["taken"] = self._read_audit()
            embeds = {}
            for rid in view.recording_ids:
                try:
                    embeds[rid] = state.embed_run_for(rid)
                except (KeyError, ValueError):
                    continue
            tracking = clip_tiles.tracking_runs(  # off-thread (run_busy)
                experiment.dir, embeds
            )
            loaded["tracking"] = tracking
            loaded["missing"] = clip_tiles.missing(  # off-thread (run_busy)
                experiment, tracking, view.recording_ids
            )
            from detail.api.clip_views import pose_notes

            loaded["embeds"] = embeds
            loaded["pose"] = pose_notes(experiment.dir, embeds)  # off-thread (run_busy)
            loaded["root_clips"] = view.clips(view.root)
            loaded["spans"] = cluster_filter.session_spans(  # off-thread (run_busy)
                experiment.dir, view.recording_ids
            )
            return loaded

        run_busy(
            self,
            work,
            self._apply_loaded,
            label=f"Loading the {self._terms.units}…",
            supersedes="tree",
        )

    def _read_audit(self) -> tuple[dict[int, dict], dict[tuple, dict]]:
        """``(per-cluster rows by node, taken-out clips by key)``, off the UI thread; none before
        any output.

        A run with clip review (temporal) reads its reviewed clusters (``review_by_cluster``,
        kept only where every clip is reviewed — AUDIT shows no figure for a cluster before
        that) and its take-outs; the hidden pose method keeps neither and reads its
        one-by-one audit.
        """
        from detail.storage import audit_figures
        from detail.gui.label_check import read_taken

        run = self._run
        try:
            if not hasattr(run, "review_by_cluster"):
                rows = run.audit_by_cluster()  # off-thread: `reload`, `refresh_audit`
                return {int(row["node"]): row for row in rows}, {}
            rows = run.review_by_cluster()  # off-thread
            taken = read_taken(run)  # off-thread
        except FileNotFoundError:
            return {}, {}
        return {
            int(row["node"]): row
            for row in rows
            if audit_figures.complete(row.get("estimate"))
        }, taken

    def refresh_audit(self) -> None:
        """Re-read the audit without reloading the page: after AUDIT judged something, and after
        a label edit here, which moves clusters in and out of what was audited."""
        if self._view is None:
            return
        if self._audit_reader is not None:
            self._audit_stale = True
            return
        self._audit_stale = False
        generation = self._generation
        reader = BackgroundCall(self._read_audit, receiver=self)
        self._audit_reader = reader

        def done(read) -> None:
            self._audit_reader = None
            # Else a newer full reload has read the audit already.
            if generation == self._generation:
                self._audited, self._taken = read
                self._refresh_outlines()
                if self._selected is not None:
                    self._show_audit_block(self._selected)
                self._mark_taken()
            if self._audit_stale:
                self.refresh_audit()

        def failed(_exc) -> None:
            self._audit_reader = None

        reader.done.connect(done)
        reader.failed.connect(failed)
        reader.finished.connect(reader.deleteLater)
        reader.start()

    def _refresh_outlines(self) -> None:
        """Ring every cluster AUDIT has judged clips of, in one colour: the ring says "audited",
        never how well — that reading is the person's, from the figure in the header."""
        ring = audited_colour()
        self._icicle.set_outlines(
            {
                node: ring
                for node, row in self._audited.items()
                if (row.get("estimate") or {}).get("n")
            }
        )

    def _apply_loaded(self, loaded) -> None:
        self._audited = loaded["audited"]
        self._taken = loaded.get("taken") or {}
        subsample = loaded["subsample"]
        self._subsample.setText(subsample or "")
        self._subsample.setVisible(bool(subsample))
        self._state = loaded["state"]
        view = loaded["view"]
        self._view = view
        self._names = dict(
            loaded["labels"]
        )  # legacy names folded on read (`node_names`)
        self._origins = {
            str(k): dict(v) for k, v in (self._state.label_origins or {}).items()
        }
        self._metadata = loaded["metadata"]
        self._column_types = loaded["types"]
        self._spans = dict(loaded.get("spans") or {})
        self._apply_editability()
        job = node_cache_jobs.running(self._run._run_dir())
        if job is not None:
            self._attach(job)
            return
        self._building.setVisible(False)
        has_cache = view is not None
        self._missing.setVisible(not has_cache)
        self._split.setVisible(has_cache)
        if not has_cache:
            self._reloaded_tree(None)
            return
        stale = getattr(view, "stale", None)
        self._stale.setText(
            f"⚠ These clips are out of date: {stale}. Press Rebuild clips to rebuild them "
            "from this run's fit; nothing is refitted."
            if stale
            else ""
        )
        self._stale.setVisible(bool(stale))
        self._stale_row.setVisible(bool(stale))
        self._root_clips = loaded["root_clips"]
        self._groups = self._dimension_groups(view)
        self._icicle.set_view(view)
        self._find.set_lookup(self._nodes_found_by)
        self._schedule_tree_fit()
        self._fill_filters(view)
        self._read_screens()
        self._tracking = loaded["tracking"]
        self._embeds = dict(loaded.get("embeds") or {})
        self._points_note, self._align_note = loaded.get("pose", (None, None))
        self._montage.set_alignment(self._align_note)
        self._montage.set_points(self._points_note)
        set_available(self._open_window, True, "")
        self._open_window.setToolTip(
            "This cluster's clips in a window of their own, beside this page: label it there, "
            "and choose what the tiles show and what to export, as here."
        )
        self._new_catalog()
        self._refresh_labels()
        self._on_selection(self._filters.selection(), reselect=False)
        self.select(self._selected if self._selected is not None else view.root)
        self._sync_previews(loaded["missing"])
        self._reloaded_tree(view)

    def _reloaded_tree(self, view) -> None:
        """The run was re-read. Its tree unchanged (:func:`tree_identity`), each window re-reads
        its clips and the ⌘Z history stands. Changed — a re-fit — every window closes, saying
        so, and the history goes: a node id names a different cluster now, and a window or an
        undo kept across that would show or write the wrong one."""
        key = tree_identity(view)
        same = key is not None and key == self._tree_key
        first = self._tree_key is None
        self._tree_key = key
        if same or (first and view is not None):
            self._windows.reloaded()
            return
        self._history.clear()
        self._sync_undo()
        if self._find.entry() is not None:
            # The view kept before finding names a node of the old fit: drop it, not restore it.
            self._before_find = None
            self._find.clear_all()
            self._icicle.fit()
        closed = self._windows.close_all()
        if closed:
            self._written.setText(
                f"Closed {closed} cluster window{'s' if closed != 1 else ''}: the run was "
                "re-fitted."
            )
            self._written.setVisible(True)

    def _dimension_groups(self, view) -> dict[str, dict[str, str]]:
        """``{dimension: {recording_id: value}}`` — every metadata column, and the recording —
        what a concentration is measured over."""
        out: dict[str, dict[str, str]] = {}
        for rid, row in self._metadata.items():
            for column, value in row.items():
                out.setdefault(column, {})[rid] = value
        out["recording"] = {
            r: view.recording_names.get(r, r) for r in view.recording_ids
        }
        return out

    def _fill_filters(self, view) -> None:
        """The filter bar's columns, recordings and density, counted in clips."""
        clips = self._root_clips
        per_recording = np.bincount(
            np.asarray(clips.recording, dtype=np.int64),
            minlength=len(view.recording_ids),
        )
        count = dict(zip(view.recording_ids, per_recording.tolist()))
        columns = []
        for name in sorted({c for row in self._metadata.values() for c in row}):
            counts: dict[str, int] = {}
            for rid in view.recording_ids:
                value = (self._metadata.get(rid, {}).get(name) or "").strip()
                counts[value] = counts.get(value, 0) + count.get(rid, 0)
            columns.append(
                Column(name, self._column_types.get(name, "category"), counts)
            )
        self._filters.set_columns(columns)
        self._filters.set_recordings(
            [
                RecordingChoice(rid, view.recording_names.get(rid, rid), count[rid])
                for rid in view.recording_ids
            ]
        )
        self._filters.set_density(view.positions(view.root))
        self._filters.set_location_available(
            view.has_positions(),
            "This run recorded no positions, so there is no place in the frame to pick.",
        )
        self._filters.set_session_available(
            bool(self._spans),
            "None of this run's recordings has a timestamp map, so there is no session to "
            "measure a time in.",
        )

    # --- the Differences line: saved screens, and running one ------------------------------

    def _read_screens(self) -> None:
        """List this run's saved difference screens in the dock, their kept clusters as leaves
        (§22, §24 *The node source*), read off the UI thread: each screen is re-ranked from its
        stored observations through the one record → ranking mapping, seconds over a large
        family. The line says it is reading until they land."""
        view = self._view
        if view is None:
            return
        if self._screens_reader is not None:
            # Read again when the one in flight lands, as `refresh_audit` does: a screen just
            # run from the dock must be listed, not lost behind an earlier read.
            self._screens_stale = True
            return
        self._screens_stale = False
        self._filters.set_screens([], loading=True)
        experiment, run_id = self._experiment, str(self._run.id)
        generation = self._generation

        def work():
            from detail.api import comparison as comparison_api

            out = []
            for comp in comparison_api.node_screens(experiment, run_id):  # off-thread
                record = comp.record()
                label = comparison_api.screen_label(record)
                try:
                    ranking = comp.ranking()  # off-thread: seconds over a large family
                except (OSError, ValueError, KeyError) as exc:
                    out.append(
                        ScreenChoice(comp.id, label, f"cannot be read: {exc}", None)
                    )
                    continue
                if ranking is None:
                    out.append(
                        ScreenChoice(comp.id, label, "nothing to rank", frozenset())
                    )
                    continue
                kept = comparison_api.kept_nodes_of(ranking)
                leaves = view.leaves_under_nodes(kept)
                if ranking.floor is None:
                    note = f"no floor: {ranking.no_floor_reason}"
                else:
                    note = f"{len(kept)} kept of {len(ranking.clearing())} clearing"
                out.append(
                    ScreenChoice(
                        comp.id,
                        label,
                        note,
                        leaves,
                        effects=_node_effects(ranking, record),
                        floor=ranking.floor,
                        # The key's axis: a profile's effect is a size (§22), whatever the
                        # record calls the design that produced it.
                        design="profile" if ranking.profile else ranking.design,
                        caption=_effect_caption(ranking, record),
                        kept=tuple(int(n) for n in kept),
                        column=str(record.get("design_column") or ""),
                        levels=_screen_levels(ranking, view),
                        # What the floor is in the unit of (§24 *The statistic*): the key's
                        # axis and the blocks follow it.
                        statistic=ranking.statistic,
                    )
                )
            return out

        reader = BackgroundCall(work, receiver=self)
        self._screens_reader = reader

        def done(entries) -> None:
            self._screens_reader = None
            if generation != self._generation:
                return
            self._filters.set_screens(entries)
            wanted = self._tick_screen
            choice = next((e for e in entries if e.key == wanted), None)
            if choice is not None and choice.leaves is not None:
                self._tick_screen = None
                # A screen just saved from **New screen…** is ticked and shown on its tab.
                self._filters.set_open_tab(DIFFERENCES_TAB)
                self._filters.set_selection(
                    dataclasses.replace(
                        self._filters.selection(),
                        leaves=choice.leaves,
                        leaves_source=choice.key,
                    )
                )
            if self._screens_stale:
                self._read_screens()

        def failed(exc) -> None:
            self._screens_reader = None
            self._filters.set_screens([])
            QMessageBox.warning(self, "Difference screens", str(exc))
            if self._screens_stale:
                self._read_screens()

        reader.done.connect(done)
        reader.failed.connect(failed)
        reader.finished.connect(reader.deleteLater)
        reader.start()

    def _run_screen(self) -> None:
        """**Run a new screen…**: the comparison launch dialog, seeded to this run and the
        cluster source, so the dock lists the screen it saves and ticks it."""
        from detail.gui.comparison import ComparisonLaunchDialog

        dialog = ComparisonLaunchDialog(
            self._experiment,
            self,
            seed={"source_run_id": str(self._run.id), "source": "nodes"},
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.comparison is None:
            return
        self._tick_screen = dialog.comparison.id
        self._read_screens()

    def _remembered_tree_height(self) -> int | None:
        """The height the user dragged the tree to, or ``None`` when they never have."""
        stored = QSettings().value(TREE_HEIGHT_KEY)
        try:
            return max(int(stored), TREE_MIN_PX)
        except (TypeError, ValueError):
            return None

    def _resize_tree(self, delta: int) -> None:
        """The tree's handle: trade tree height for montage height, kept for next time. From
        the first drag on, rows keep their full height and the tree scrolls."""
        height = max(self._tree_holder.height() + int(delta), TREE_MIN_PX)
        if self._main.isVisible():
            # Never past what leaves the montage one row: the column would overflow the window.
            height = min(
                height, max(self._tree_room(self._montage_floor()), TREE_MIN_PX)
            )
        self._tree_dragged = True
        self._icicle.set_fit_rows(False)
        self._tree_holder.setFixedHeight(height)
        QSettings().setValue(TREE_HEIGHT_KEY, height)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        # The bar grows a line (the AUDIT line) without the column changing size, and that
        # height comes out of the tree's room as much as a window resize does.
        # The event's type first: the hides of a teardown reach this filter after the page's
        # Python half is gone (``live_montage.LiveMontage._sync_clock``), and only a resize
        # needs the page's widgets.
        if event.type() == QEvent.Type.Resize and (
            watched is self._main or watched is self._label_bar
        ):
            self._schedule_tree_fit()
        return super().eventFilter(watched, event)

    def _schedule_tree_fit(self) -> None:
        """Fit the tree after the layout has run: a resize event arrives before it does."""
        if not self._fit_pending:
            self._fit_pending = True
            QTimer.singleShot(0, self._fit_tree)

    def _visible_main_height(self) -> int:
        """The tree-and-montage column's height, cut to what the window shows of it.

        Inside a page that scrolls (`scroll.PageScroll`), the column can be laid out taller than
        the window; sized to that, the tree would push the montage off screen.
        """
        height = self._main.height()
        scroll = self._main.parentWidget()
        while scroll is not None and not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        if scroll is not None and scroll.widget() is not None:
            top = self._main.mapTo(scroll.widget(), QPoint(0, 0)).y()
            shown = scroll.viewport().height() - top
            if shown > 0:
                height = min(height, shown)
        return height

    def _others_height(self) -> int:
        """What the column's rows other than the tree and the montage want: their size hints,
        the spacing between the shown rows and the column's margins.

        **From hints, never from the heights they were given.** It used to be the column's
        height less the tree's and the montage's — but once the column overflows, those heights
        are the overflow, and the room computed from them was 30px where 140 was real, so the
        tree never gave up a pixel and the grid's bottom went out of the window (1440×847).
        """
        layout = self._main.layout()
        shown, total = 0, 0
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget is None or widget.isHidden():
                continue
            shown += 1
            if widget in (self._tree_holder, self._montage):
                continue
            # `minimumHeight` too: a fixed-height row (the 6px handle) has no size hint.
            total += max(
                widget.sizeHint().height(),
                widget.minimumSizeHint().height(),
                widget.minimumHeight(),
            )
        margins = layout.contentsMargins()
        return (
            total
            + max(shown - 1, 0) * max(layout.spacing(), 0)
            + margins.top()
            + margins.bottom()
        )

    def _montage_floor(self) -> int:
        return self._montage.height_for_rows(1)

    def _tree_room(self, montage: int) -> int:
        """The tallest the tree can be while the montage keeps ``montage`` px."""
        return self._visible_main_height() - self._others_height() - montage

    def _fit_tree(self) -> None:
        """Size the tree so the column fits the window, the montage keeping
        ``MONTAGE_MIN_ROWS`` rows of tiles while the tree can give them up.

        Until the user drags it, the tree asks for every level (rows shrinking past that,
        `set_fit_rows`); dragged, it asks for the height they left it at. Either way it gives
        height back down to ``TREE_MIN_PX`` before the montage drops below its rows, and the
        montage's own floor is one row — so on a laptop window the grid shrinks to what is left
        and its scrollbar stays whole, rather than its bottom going under the window's edge.
        """
        self._fit_pending = False
        if self._view is None or not self._main.isVisible():
            return
        self._montage.setMinimumHeight(self._montage_floor())
        icicle, holder = self._icicle, self._tree_holder
        if self._tree_dragged:
            wanted = self._remembered_tree_height() or holder.height()
        else:
            levels = icicle.whole_tree_levels()
            if levels <= 0:
                return
            chrome = holder.height() - icicle.height()
            wanted = chrome + levels * row_px()
        room = self._tree_room(self._montage.height_for_rows(MONTAGE_MIN_ROWS))
        height = max(min(wanted, room), TREE_MIN_PX)
        if height != holder.height():
            holder.setFixedHeight(height)

    def _nodes_found_by(self, entry) -> list[int]:
        """The clusters a Find entry names, largest first: an id the tree has, or every
        cluster whose own label is ``entry`` (ignoring case)."""
        view = self._view
        if view is None:
            return []
        times = view.node_times()
        if isinstance(entry, int):
            return [entry] if 0 <= entry < len(times) else []
        key = entry.casefold()
        nodes = [
            int(k) for k, name in self._names.items() if (name or "").casefold() == key
        ]
        return sorted(nodes, key=lambda node: -float(times[node]))

    def _icicle_rings(self, found: dict) -> None:
        self._icicle.set_found(found)

    def _show_found(self, found: dict) -> None:
        """Find clusters changed: ring its ids, and zoom to the smallest cluster holding all of
        them. The view before the first zoom is kept, and an emptied list (**Clear**, or the
        last id's ×) takes the rings off and shows that view again (user, 2026-11-16).
        """
        self._icicle.set_found(found)
        if self._view is None:
            return
        if not found:
            if self._before_find is not None:
                self._icicle.restore_view(self._before_find)
                self._before_find = None
            return
        if self._before_find is None:
            self._before_find = self._icicle.view_state()
        self._icicle.zoom_to(common_cluster(self._view, list(found)))

    def _sync_zoom_buttons(self) -> None:
        zoomed = self._icicle.is_zoomed()
        set_available(self._zoom_out, zoomed, "The whole tree is shown already")
        set_available(self._fit, zoomed, "The whole tree is shown already")
        if zoomed:
            self._zoom_out.setToolTip("Zoom out")
            self._fit.setToolTip("Show the whole tree (Esc)")

    # --- the filter dock -------------------------------------------------------------------

    def _restore_dock(self) -> None:
        """The dock's remembered width, and shut or open as it was left."""
        settings = QSettings()
        try:
            legacy_width = int(settings.value(_LEGACY_FILTER_WIDTH_KEY))
        except (TypeError, ValueError):
            legacy_width = 0
        restored = self._split.remember(
            legacy_shut=(
                {0}
                if settings.value(_LEGACY_FILTER_FOLDED_KEY, False, type=bool)
                else set()
            ),
            legacy_open={0: legacy_width} if legacy_width else None,
            legacy_keys=(_LEGACY_FILTER_WIDTH_KEY, _LEGACY_FILTER_FOLDED_KEY),
        )
        if not restored:
            # Until the user drags it: the width it had when its floor was the map's 320 (it
            # drags down to `cluster_filters.DOCK_MIN_PX` now, and the map follows).
            self._split.setSizes([MAP_PX + lines(2), lines(80)])

    # --- labels ----------------------------------------------------------------------------

    def _slot_of_root(self, root: str) -> int:
        return root_slots(label_paths(self._view, self._names).values()).get(root, 0)

    def _colour_of_root(self, root: str) -> QColor:
        return plot_theme.identity(self._slot_of_root(root))

    def _colour_of(self, node: int) -> QColor:
        owner = self._owner(node)
        if owner is None:
            return QColor(plot_theme.current().neutral)
        path = label_paths(self._view, self._names).get(str(owner), "")
        return self._colour_of_root(path.split(PATH_SEPARATOR)[0])

    def _owner(self, node: int) -> int | None:
        """``node`` itself if labelled, else its nearest labelled ancestor, else ``None``."""
        if self._names.get(str(node)):
            return int(node)
        for ancestor in reversed(self._view.ancestors(node)):
            if self._names.get(str(ancestor)):
                return int(ancestor)
        return None

    def _refresh_labels(self) -> None:
        view = self._view
        fills: dict[int, tuple[QColor, bool]] = {}
        for node in range(view.tree.n_nodes):
            owner = self._owner(node)
            if owner is not None:
                fills[node] = (self._colour_of(node), owner == node)
        self._icicle.set_labels(fills, {int(k): v for k, v in self._names.items()})
        self._refresh_outlines()
        rows = label_rows(view, self._names)
        self._label_field.set_labels(rows, self._colour_of_root)
        self._label_bar.set_quick_labels(self._quick_labels(rows))
        self._key_row.swatches.set_labels(
            [
                (text, self._colour_of_root(root), time_ns / 1e9)
                for text, depth, root, time_ns in rows
                if depth == 0 and root
            ]
        )
        if self._selected is not None:
            self._show_label_field(self._selected)
            self._show_header(self._selected)
            self._emit_lane(self._selected)
        self._windows.refresh()
        # A label found by name may now be on more clusters, or fewer: re-ring, do not zoom.
        # Bands keep off the labels' colours, which fill the blocks they ring.
        self._find.set_reserved(
            range(len(root_slots(label_paths(view, self._names).values())))
        )
        self._find.refresh()

    # --- selection -----------------------------------------------------------------------

    def select(self, node: int) -> None:
        if self._view is None or node is None:
            return
        self._selected = int(node)
        self._clip_text = ""
        self._icicle.set_selected(self._selected)
        self._show_label_field(self._selected)
        self._show_header(self._selected)
        self._show_audit_block(self._selected)
        self._show_key()
        self._emit_lane(self._selected)
        self._load_clips(self._selected)

    def _emit_lane(self, node: int) -> None:
        view = self._view
        if view is None or int(node) == int(view.root):
            self.lane_changed.emit(None)
            return
        name = self._names.get(str(node)) or f"{self._terms.unit} #{node}"
        colour = (
            self._colour_of(node)
            if self._owner(node) is not None
            else QColor(plot_theme.current().text)
        )
        self.lane_changed.emit((view, int(node), name, colour))

    def _quick_labels(self, rows) -> list[tuple[str, QColor]]:
        """``[(name, colour)]`` for the bar's quick buttons: labels used on this page, most
        recent first, then every other label by its time, largest first (``rows`` is
        :func:`label_rows`' order)."""
        paths = label_paths(self._view, self._names)
        root_of: dict[str, str] = {}
        for node, path in paths.items():
            root_of.setdefault(self._names[node], path.split(PATH_SEPARATOR)[0])
        by_time = [
            row_path(text, depth, root).split(PATH_SEPARATOR)[-1]
            for text, depth, root, _t in rows
            if root and text != "other"
        ]
        ordered = list(dict.fromkeys([*self._recent, *by_time]))
        return [
            (name, self._colour_of_root(root_of[name]))
            for name in ordered
            if name in root_of
        ]

    def _cluster_name(self, node: int) -> str:
        """ "Behavior #1384": what the bar and its confirmations call a cluster."""
        return f"{self._terms.unit.capitalize()} #{int(node)}"

    def _show_header(self, node: int) -> None:
        """The label bar's first line and its figures (:func:`cluster_facts`)."""
        parts, self._separation, self.concentration = cluster_facts(self, node)
        self._header_parts = parts
        show_cluster_name(self, self._label_bar, node, self._cluster_name(node))
        self._show_figures()

    def _show_figures(self) -> None:
        show_figures(
            self._label_bar,
            self._clip_text,
            (self._header_parts, self._separation, self.concentration),
        )

    def _show_audit_block(self, node: int) -> None:
        self._audit_block.show_node(
            node,
            self._audited.get(int(node)),
            label_choices(self._view, self._names, node),
            has_halves=bool(self._view.children(node)),
            editable_reason=self._edit_reason(),
        )

    def _look_at_halves(self, node: int) -> None:
        """Zoom onto the cluster so its two halves fill the tree, and open the first."""
        self._icicle.zoom_to(node)
        children = list(self._view.children(node))
        if children:
            self.select(int(children[0]))

    def _label_field_state(self, node: int) -> tuple:
        """``(own, inherited, names, choices)``: what a label field shows for ``node`` — this
        page's bar, and a cluster window's row."""
        own = self._names.get(str(node), "")
        owner = self._owner(node)
        inherited = (
            self._names.get(str(owner), "")
            if owner is not None and owner != node
            else ""
        )
        choices = label_choices(self._view, self._names, node)
        return own, inherited, sorted({n for n in self._names.values() if n}), choices

    def _label_rows(self):
        return label_rows(self._view, self._names)

    def _show_label_field(self, node: int) -> None:
        show_label_state(self, self._label_bar, node)

    def _edit_label_inline(self, node: int, where) -> None:
        """A small editor at ``where`` (global), for a double-click or the tree's menu."""
        if not self._can_edit or self._view is None:
            return
        self.select(node)
        editor = InlineLabelEditor(
            self._names.get(str(node), ""),
            sorted({n for n in self._names.values() if n}),
            self._terms.unit,
            self,
        )
        editor.submitted.connect(lambda name, n=int(node): self._label(n, name))
        editor.move(where)
        editor.show()
        editor.focus()
        self._inline_editor = editor

    def selected_label_path(self) -> str | None:
        """The label path covering the selected node (its own, or the nearest labelled branch
        above it), or ``None`` — what AUDIT opens on when you move to it from here."""
        if self._view is None or self._selected is None:
            return None
        owner = self._owner(self._selected)
        if owner is None:
            return None
        return label_paths(self._view, self._names).get(str(owner))

    def _select_label(self, root: str) -> None:
        paths = label_paths(self._view, self._names)
        for node, path in sorted(paths.items(), key=lambda item: len(item[1])):
            if path.split(PATH_SEPARATOR)[0] == root:
                self.select(int(node))
                return

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if (
            self._view is None
            or self._selected is None
            or isinstance(self.focusWidget(), QLineEdit)
        ):
            super().keyPressEvent(event)
            return
        node = self._selected
        key = event.key()
        # A key that labels takes no modifier (the keypad's is fine), as in a cluster window:
        # ⌘3 is the app's, not "label with recent label 3".
        bare = not event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        # The rows as drawn, not the raw merges: a chain of sliver merges is one row on screen,
        # and walking it one merge per key press would be ten presses through the same cluster.
        parent = self._icicle.shown_parent(node)
        children = self._icicle.shown_children(node)
        if key == Qt.Key.Key_Up and parent is not None:
            self.select(parent)
        elif key == Qt.Key.Key_Down and children:
            times = self._view.node_times()
            self.select(max(children, key=lambda c: times[c]))
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and parent is not None:
            siblings = self._icicle.shown_children(parent)
            if node in siblings:
                index = siblings.index(node) + (-1 if key == Qt.Key.Key_Left else 1)
                if 0 <= index < len(siblings):
                    self.select(siblings[index])
        elif key == Qt.Key.Key_L and bare:
            self._label_field.focus()
        elif key == Qt.Key.Key_U and bare and self._names.get(str(node)):
            self._label(node, "")
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_9 and bare:
            # The recent label with that number; a greyed one (not allowed here) does nothing.
            self._label_bar.quick.press(key - Qt.Key.Key_1)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # --- filters ---------------------------------------------------------------------------

    def _on_selection(self, selection, reselect: bool = True) -> None:
        """Shade the tree by each cluster's matching share, and grey the montage's misses."""
        self._selection = selection
        self._split.set_pane_name(0, panel_name(selection))
        view = self._view
        if view is None:
            return
        if selection.is_empty:
            self._shares = None
            self._effect_screen = None
            self._filters.set_match_count(None, len(self._root_clips))
        else:
            # The Filters tab's shade is the share of clips that match its own fields: the
            # ticked screen is the other tab's, and frames its kept clusters there instead
            # of narrowing the share to them (§22).
            fields = selection.without(cluster_filter.CLUSTERS)
            try:
                self._shares = (
                    None
                    if fields.is_empty
                    else view.match_shares(
                        fields, self._metadata, self._column_types, self._spans
                    )
                )
            except ValueError as exc:
                QMessageBox.information(self, "Filter not applied", str(exc))
                self._filters.set_selection(cluster_filter.Selection())
                return
            self._effect_screen = self._ticked_screen(selection)
            matched = cluster_filter.clip_matches(
                selection,
                self._root_clips,
                self._metadata,
                self._column_types,
                self._spans,
            )
            self._filters.set_match_count(int(matched.sum()), len(self._root_clips))
        self._filters.set_session_count(
            None
            if selection.session is None
            else int(
                cluster_filter.clip_matches(
                    selection.only(cluster_filter.SESSION),
                    self._root_clips,
                    spans=self._spans,
                ).sum()
            )
        )
        self._shade_tree()
        if reselect:
            self._apply_matching()

    def _shade_tree(self) -> None:
        """Colour the tree by what the open tab is about (§22), and fill both scales.

        **Differences** open: each block by the ticked screen's statistic, grey inside the
        floor, at the strictness the dock is set to; nothing shaded with no screen ticked.
        **Filters** open: each block by the share of its clips that match the fields, and
        the clusters the ticked screen kept framed in the colour of their effect. The tree
        never blends the two, and never changes what its colour means without a tab having
        been clicked: with one scale for both, ticking a field beside a screen swapped the
        blue and orange for a share with nothing said (user, 2026-11-16).
        """
        view = self._view
        if view is None:
            return
        screen = self._effect_screen
        alpha = self._selection.alpha if self._selection is not None else None
        n_nodes = int(view.tree.n_nodes)
        if self._filters.open_tab() == DIFFERENCES_TAB:
            self._icicle.set_kept({})
            if screen is None:
                self._icicle.set_shades(None)
            else:
                positions, texts = effect_shades(screen, n_nodes, alpha=alpha)
                self._icicle.set_shades(
                    positions,
                    diverging=True,
                    one_sided=one_sided(screen.design),
                    texts=texts,
                )
        else:
            self._icicle.set_shades(
                None
                if self._shares is None
                else {node: float(share) for node, share in enumerate(self._shares)}
            )
            if screen is None:
                self._icicle.set_kept({})
            else:
                marks, texts = kept_marks(screen, n_nodes, alpha=alpha)
                colour = (
                    plot_theme.size_colour
                    if one_sided(screen.design)
                    else plot_theme.effect_colour
                )
                # The pole's own colour, whatever the size: a frame says kept and which
                # way, and a weak effect's shade of it was too dull to find on a grey fill.
                self._icicle.set_kept(
                    {
                        node: colour(math.copysign(1.0, value))
                        for node, value in marks.items()
                        if value
                    },
                    texts,
                )
        self._show_key()

    def _ticked_screen(self, selection) -> ScreenChoice | None:
        """The ticked screen, when its effects are known."""
        if selection.leaves is None:
            return None
        choice = self._filters.screen(selection.leaves_source)
        if choice is None or not choice.effects:
            return None
        return choice

    def _show_key(self) -> None:
        """Fill the two scales: the ticked screen's on the Differences tab, the clips that
        match on the Filters tab. Each is its own widget, so neither changes its units.
        """
        key = self._filters.dock.key
        screens = self._filters.screens
        screen = self._effect_screen
        screens.set_scale_shown(screen is not None and self._view is not None)
        if screen is not None and self._view is not None:
            alpha = self._selection.alpha if self._selection is not None else None
            level = screen.level(alpha)
            selected = (
                int(self._selected)
                if self._selected is not None and self._selected != self._view.root
                else None
            )
            row = screen.effects.get(selected) if selected is not None else None
            clears = selected is not None and screen.clears(selected, alpha)
            screens.key.set_effect(
                screen.caption,
                screen.design,
                floor=screen.floor if level is None else level.floor,
                selected_effect=None if row is None or not clears else row.effect,
                selected_pair=None if row is None or not clears else row.pair,
                statistic=screen.statistic,
                selected_statistic=(
                    None if row is None or not clears else row.statistic
                ),
            )
        if self._shares is None or self._view is None:
            key.set_filter(False)
            return

        def share(node: int | None) -> float | None:
            if node is None or not np.isfinite(self._shares[node]):
                return None
            return float(self._shares[node])

        key.set_filter(
            True,
            selected_share=(
                share(self._selected) if self._selected != self._view.root else None
            ),
            run_share=share(self._view.root),
        )

    def _apply_matching(self) -> None:
        """Each of the montage's clips meets the filters or is drawn grey; the per-condition
        ✓ / ✗ for the hover come from the same rule (``clip_matches`` on one condition).
        """
        clips = self._clips
        if clips is None or not len(clips):
            self._passed = {}
            return
        selection = self._selection
        self._passed = {
            key: cluster_filter.clip_matches(
                selection.only(key),
                clips,
                self._metadata,
                self._column_types,
                self._spans,
            )
            for key in selection.keys()
        }
        every = np.ones(len(clips), dtype=bool)
        for flags in self._passed.values():
            every &= flags
        self._montage.set_matching(
            every[self._order].tolist(), filtered=not selection.is_empty
        )

    def _hover_text(self, index: int) -> str:
        clips = self._clips
        if clips is None or not (0 <= index < len(self._order)):
            return ""
        i = int(self._order[index])
        rid = clips.recording_ids[int(clips.recording[i])]
        name = self._view.recording_names.get(rid, rid)
        start, end = int(clips.start_ns[i]), int(clips.end_ns[i])
        lines_ = [
            f"#{index + 1} · {name} · {_clock(start)} · {(end - start) / 1e9:.1f} s",
        ]
        if self._passed:
            lines_.append(
                breakdown(
                    self._selection,
                    {key: bool(flags[i]) for key, flags in self._passed.items()},
                    self._metadata.get(rid, {}),
                )
            )
        if np.isfinite(clips.cx[i]):
            lines_.append(
                f"where: {100 * clips.cx[i]:.0f}% across, {100 * clips.cy[i]:.0f}% down"
            )
        lines_.append("Double-click to open it in the clip window")
        return "\n".join(lines_)

    # --- the montage -----------------------------------------------------------------------

    def _load_clips(self, node: int) -> None:
        """The node's clips: from the cache in memory, or — for a method ranked on demand
        (``NodeView.ranks_on_demand``) — ranked off the UI thread on first ask."""
        self._clip_request += 1
        request = self._clip_request
        view = self._view
        if not view.ranks_on_demand:
            self._show_clips(node, view.clips(node))
            return
        run = self._run
        self._clear_clips()
        self._clip_text = f"ranking this {self._terms.unit}'s clips…"
        self._show_figures()
        reader = BackgroundCall(lambda: run.clips(node), receiver=self)
        self._clip_reader = reader

        def done(clips) -> None:
            if request == self._clip_request:
                self._show_header(node)
                self._show_clips(node, clips)

        reader.done.connect(done)
        reader.failed.connect(
            lambda exc: request == self._clip_request
            and QMessageBox.critical(self, "Clips not ranked", str(exc))
        )
        reader.finished.connect(reader.deleteLater)
        reader.start()

    def _clear_clips(self) -> None:
        """Empty the montage *and* the page's clips together while a node's clips rank.

        Every control that re-shows or re-marks the montage reads ``_clips`` / ``_order``, so
        emptying only the montage left them on the previous node's clips: a filter set in the
        wait handed its flags to a montage holding none (``N flags for 0 clips``), and the order
        or context switch drew the previous node's tiles back.
        """
        self._clips = None
        self._order = np.zeros(0, dtype=np.int64)
        self._passed = {}
        self._montage.set_clips([])

    def _show_clips(self, node: int, clips, *, keep_scroll: bool = False) -> None:
        self._clips = clips
        self._order = clips.order(self._montage.order(), seed=int(node))
        names = self._view.recording_names
        self._montage.set_clips(self._tiles(names), keep_scroll=keep_scroll)
        self._show_clip_count()
        self._set_source()
        self._apply_matching()

    def _set_source(self, *, keep_pictures: bool = False) -> None:
        """The montage's pictures, from the page's catalog, for its clips in its order."""
        from detail.api import clip_tiles

        clips = self._clips
        if self._catalog is None or clips is None:
            return
        context = self._montage.context_s()
        rows = []
        for i in self._order:
            rid = clips.recording_ids[int(clips.recording[i])]
            start, end = context_span(clips.start_ns[i], clips.end_ns[i], context)
            rows.append(clip_tiles.TileClip(rid, int(clips.subject[i]), start, end))
        self._montage.set_source(
            clip_tiles.ClipTiles(self._catalog, rows), keep_pictures=keep_pictures
        )

    def _clip_key(self, i: int) -> tuple:
        return self._clip_key_of(self._clips, i)

    def _clip_key_of(self, clips, i: int) -> tuple:
        """Clip ``i`` of ``clips``' key in AUDIT's removals (``label_clean.clip_key``)."""
        from detail.gui.label_clean import clip_key

        return clip_key(
            clips.recording_ids[int(clips.recording[i])],
            clips.subject[i],
            clips.start_ns[i],
            clips.end_ns[i],
        )

    def _tiles(self, names) -> list[MontageClip]:
        """The montage's tiles in its order; a clip removed in review is faded and tagged."""
        from detail.gui.label_clean import crossed_for, note_for, tag_for

        clips = self._clips
        context = self._montage.context_s()
        tiles = []
        for rank, i in enumerate(self._order, start=1):
            rid = clips.recording_ids[int(clips.recording[i])]
            start, end = context_span(clips.start_ns[i], clips.end_ns[i], context)
            edit = self._taken.get(self._clip_key(int(i)))
            tiles.append(
                MontageClip(
                    max(end - start, _MIN_LOOP_NS),
                    rank,
                    f"{names.get(rid, rid)} {_clock(start)}",
                    tag=tag_for(edit) if edit is not None else None,
                    faded=edit is not None,
                    note=(
                        f"{note_for(edit)}. Right-click to put it back."
                        if edit is not None
                        else ""
                    ),
                    crossed=crossed_for(edit),
                )
            )
        return tiles

    def _show_clip_count(self) -> None:
        """ "48 clips · 6 removed" in the label bar's figures."""
        self._clip_text = clip_count_text(self, self._clips)
        self._show_figures()

    def _mark_taken(self) -> None:
        """Re-mark the montage's faded clips after the removals changed; nothing else moves.
        The windows first: each holds its own clips, whatever this page's grid is doing.
        """
        self._windows.refresh_marks()
        if self._clips is None or self._view is None or not len(self._order):
            return
        if len(self._order) != len(self._montage.clips()):
            return
        tiles = self._tiles(self._view.recording_names)
        if marks_changed(self._montage.clips(), tiles):
            self._montage.set_clips(tiles, keep_scroll=True)
            self._apply_matching()
            self._show_clip_count()

    def _on_clip_menu(self, index: int, where) -> None:
        """Right-click a faded clip: put it back in the label it was removed from."""
        from detail.gui.label_clean import display, edit_label

        clips = self._clips
        if clips is None or not 0 <= index < len(self._order):
            return
        key = self._clip_key(int(self._order[index]))
        edit = self._taken.get(key)
        if edit is None:
            return
        menu = QMenu(self)
        source = edit_label(edit, "from")
        action = menu.addAction(
            f"Put back in {display(source)}", lambda k=key: self._put_back(k)
        )
        action.setEnabled(self._can_edit)
        if not self._can_edit:
            action.setToolTip(self._edit_reason())
            menu.setToolTipsVisible(True)
        from detail.gui.menus import open_once

        open_once(menu, where, wait=False)

    def _put_back(self, key: tuple) -> None:
        """Fade lifted at once; written in the background, in order, with the labels."""
        if not self._can_edit or key not in self._taken:
            return
        from detail.gui.label_clean import display, edit_label
        from detail.gui.label_history import Change

        edit = self._taken[key]
        ref = {
            "recording_id": key[0],
            "subject": key[1],
            "start_t_ns": key[2],
            "end_t_ns": key[3],
        }
        run = self._run

        def show(after: bool) -> None:
            if after:
                self._taken.pop(key, None)
            else:
                self._taken[key] = edit
            self._mark_taken()

        def put(after: bool) -> None:
            if after:
                run.put_back_clips([ref])
            else:
                run.take_out_clips([ref], edit["from_label"], edit.get("to_label"))

        self._record(
            Change(
                f"Put the clip back in {display(edit_label(edit, 'from'))}", show, put
            ),
            (lambda: put(True), None),
        )

    def _on_order(self, key: str) -> None:
        self._montage.set_order(key)
        if self._selected is not None and self._clips is not None:
            self._show_clips(self._selected, self._clips)

    def _on_context(self) -> None:
        """**Show ▾**'s context changed: every tile plays its clip widened by it, the scroll
        kept."""
        if self._selected is not None and self._clips is not None:
            self._show_clips(self._selected, self._clips, keep_scroll=True)

    def _new_catalog(self) -> None:
        from detail.api import clip_tiles

        self._catalog = clip_tiles.TileCatalog(
            self._experiment, self._tracking, self._embeds
        )

    def _previews_arrived(self) -> None:
        """A recording's previews or masks were built: every montage on the run — this page's
        and each window's — reads its pictures afresh, in place. The tiles, their order, the
        scroll and every clock stay; each tile keeps its picture until the new one lands. (It
        re-set the tiles, so a recording's masks finishing sent every window back to its top
        and re-ranked a method that ranks on demand.)"""
        self._new_catalog()
        if self._selected is not None and self._clips is not None:
            self._set_source(keep_pictures=True)
        self._windows.previews_changed()

    def _play_clip(self, index: int) -> None:
        """A tile double-clicked: the clip in the clip window, from the source video, with
        previous / next through the montage's order."""
        from detail.gui import clip_popup

        clips = self._clips
        if clips is None or not (0 <= index < len(self._order)):
            return
        sequence = clip_sequence(self, clips, self._order)
        clip_popup.open_clip_window(
            self,
            self._experiment,
            sequence[index],
            sequence=sequence,
            index=index,
            context_s=self._montage.context_s(),
            order_note=dict(_ORDER_NOTES).get(self._montage.order(), ""),
        )

    # --- previews --------------------------------------------------------------------------

    def _queue(self):
        """The application's preview queue when it serves this experiment, else this page's."""
        found = preview_jobs.current()
        if found is not None and found.experiment is not None:
            if found.experiment.dir == self._experiment.dir:
                return found
        if self._local_queue is None:
            self._local_queue = preview_jobs.PreviewQueue(self._experiment)
        return self._local_queue

    def _sync_previews(self, missing: dict) -> None:
        """Block the montage on the run's missing proxies; play footage where only masks are
        missing, and put the outlines on as each recording's masks land."""
        self._missing_previews = list(missing["proxy"])
        self._missing_masks = list(missing["masks"])
        self._end_wait()
        self._preview_note.setVisible(False)
        if self._missing_masks:
            queue = self._queue()
            if self._mask_queue is not queue:
                if self._mask_queue is not None:
                    self._mask_queue.recording_done.disconnect(self._on_masks_done)
                queue.recording_done.connect(self._on_masks_done)
                self._mask_queue = queue
            queue.prioritise(self._missing_masks)
            self._note_previews(
                f"{len(self._missing_masks)} of this run's recordings play without the "
                "animal's outline until their masks are built; they appear as each finishes."
            )
        if self._missing_previews:
            self._wait_for_previews()

    def _on_masks_done(self, recording_id: str) -> None:
        if recording_id not in self._missing_masks:
            return
        self._missing_masks.remove(recording_id)
        self._previews_arrived()
        if not self._missing_masks and self._wait is None:
            self._preview_note.setVisible(False)

    def _wait_for_previews(self) -> None:
        missing = list(self._missing_previews)
        if not missing:
            return
        queue = self._queue()
        if self._wait is not None and self._wait.queue is queue:
            return
        view = self._view
        self._wait = _PreviewWait(
            queue, missing, view.recording_ids if view is not None else ()
        )
        queue.progress.connect(self._on_preview_progress)
        queue.recording_done.connect(self._on_preview_done)
        queue.failed.connect(self._on_preview_failed)
        self._preview_note.setVisible(False)
        queue.prioritise(missing)
        self._show_wait()

    def _show_wait(self) -> None:
        wait = self._wait
        if wait is None:
            return
        self._montage.set_blocked(
            wait.message(), wait.fraction(), wait.eta(), self._cancel_previews
        )

    def _on_preview_progress(self, _permille: int, _status: str) -> None:
        self._show_wait()

    def _on_preview_done(self, recording_id: str) -> None:
        wait = self._wait
        if wait is None or recording_id not in wait.waiting:
            return
        wait.waiting.remove(recording_id)
        self._missing_previews = [
            r for r in self._missing_previews if r != recording_id
        ]
        if wait.waiting:
            self._show_wait()
            return
        self._end_wait()
        self._previews_arrived()
        if wait.failed:
            self._note_previews(
                f"Previews could not be built for {len(wait.failed)} recording(s); their "
                "tiles show no picture."
            )

    def _on_preview_failed(self, recording_id: str, _exc) -> None:
        """A build that failed is waited for no longer; the note after the wait says so."""
        wait = self._wait
        if wait is None or recording_id not in wait.waiting:
            return
        wait.failed.append(recording_id)
        self._on_preview_done(recording_id)

    def _cancel_previews(self) -> None:
        """Cancel: stop waiting for this run's recordings (and their builds), nobody else's."""
        wait = self._wait
        if wait is None:
            return
        wait.queue.withdraw(wait.waiting)
        self._end_wait()
        self._note_previews(
            f"Previews are not built for {len(self._missing_previews)} of this run's "
            "recordings, so their tiles show no picture."
        )

    def _note_previews(self, text: str) -> None:
        self._preview_text.setText(text)
        self._preview_note.setVisible(True)
        self._build_previews.setVisible(bool(self._missing_previews))

    def _end_wait(self) -> None:
        wait, self._wait = self._wait, None
        if wait is not None:
            for signal, slot in (
                (wait.queue.progress, self._on_preview_progress),
                (wait.queue.recording_done, self._on_preview_done),
                (wait.queue.failed, self._on_preview_failed),
            ):
                try:
                    signal.disconnect(slot)
                except TypeError:
                    pass  # never connected, or already gone
        self._montage.clear_blocked()

    # --- cluster windows -------------------------------------------------------------------

    def open_window(self, node: int | None = None):
        """Open ``node`` (default: the selected cluster) in a window of its own, or raise its
        window when it has one (``cluster_window.ClusterWindows``). Returns it."""
        node = self._selected if node is None else node
        if self._view is None or node is None:
            return None
        return self._windows.open(int(node))

    def windows(self):
        """The run's cluster windows (``cluster_window.ClusterWindows``)."""
        return self._windows

    # --- exports ---------------------------------------------------------------------------

    def export_clips(self) -> list:
        """The montage's clips in its order as ``api.clip_montage.MontageClip`` — what
        **Export ▾** writes (``montage_export.ExportSource``): every clip, each its own span,
        ranked by its place in the order. The exports read the source video, not the previews.
        """
        from detail.api import clip_montage as montage_api

        clips = self._clips
        if clips is None or not len(clips) or self._view is None:
            return []
        names = self._view.recording_names
        out = []
        for rank, i in enumerate(self._order, start=1):
            rid = clips.recording_ids[int(clips.recording[i])]
            out.append(
                montage_api.MontageClip(
                    rank=rank,
                    recording_id=rid,
                    recording_name=names.get(rid, rid),
                    subject=int(clips.subject[i]),
                    embed_run_id=self._state.embed_run_for(rid),
                    start_t_ns=int(clips.start_ns[i]),
                    end_t_ns=int(clips.end_ns[i]),
                )
            )
        return out

    def _export_stem(self) -> str:
        """What an export's suggested file name starts with: the cluster's label, or its id."""
        return (
            self._names.get(str(self._selected))
            or f"{self._terms.unit}_{self._selected}"
        )

    def _origin(self):
        from detail.api.clip_montage import MontageOrigin

        return MontageOrigin(
            run_id=str(self._run.id),
            node=int(self._selected),
            name=self._names.get(str(self._selected)),
            unit=self._terms.unit,
        )

    # --- labelling -----------------------------------------------------------------------

    def _label(self, node: int, name: str, origin=None) -> None:
        """Show the label at once; write it in the background, in order.

        ``origin`` is the cluster window that asked, or ``None`` for this page: a prompt opens
        over the window that asked, and only a label written from this page selects its cluster
        here — a window's label leaves INTERPRET where it was (D1 B, 2026-11-09).
        """
        if not self._can_edit:
            return
        name = name.strip()
        if PATH_SEPARATOR in name:
            QMessageBox.warning(
                origin or self,
                "Label not saved",
                f"A label may not contain {PATH_SEPARATOR!r}: it separates the levels of a "
                "nested label in the exported output. Use another character.",
            )
            return
        names, recent = dict(self._names), list(self._recent)
        origins = dict(self._origins)
        found = self._screen_origin(node) if name else None
        if name:
            names[str(int(node))] = name
            recent = [name, *(n for n in recent if n != name)]
            text = f"Labelled #{int(node)} “{name}”"
            if found is not None:
                origins[str(int(node))] = found.to_dict()
        else:
            names.pop(str(int(node)), None)
            origins.pop(str(int(node)), None)
            text = f"Removed the label from #{int(node)}"
        self._change(
            text,
            names,
            recent,
            lambda n=int(node), m=name, o=found: self._run.name_nodes(
                {n: m}, origins={n: o}
            ),
            origins=origins,
        )
        if origin is None:
            self.select(node)

    def _screen_origin(self, node: int):
        """Where a label written now would come from: the ticked screen, when it points at
        ``node`` (``run_labels.label_origin``), else ``None``. The screen need not be the
        only filter on -- it is the one that chose the cluster either way."""
        from detail.storage import run_labels

        selection = self._selection
        if selection is None or selection.leaves_source is None or self._view is None:
            return None
        choice = self._filters.screen(selection.leaves_source)
        if choice is None:
            return None
        # At the strictness the screen is read at: a label chosen at p ≤ 0.3 says so (§22).
        kept_texts = choice.kept_texts(selection.alpha)
        if not kept_texts:
            return None
        return run_labels.label_origin(
            int(node),
            comparison=choice.key,
            kept_texts=kept_texts,
            column=choice.column,
            screen=choice.label,
            ancestors_of=self._view.ancestors,
            alpha=selection.alpha,
            numbers=choice.kept_numbers(selection.alpha),
        )

    def _label_id(self, name: str) -> str:
        """The run's id for the label called ``name`` — read when the write runs, off the UI
        thread, so a rename queued behind another edit finds the label as that edit left it.
        """
        from detail.storage import run_labels

        label = run_labels.find(self._run.state().labels, name)
        if label is None:
            raise KeyError(f"there is no label called “{name}” in this run any more")
        return label

    def _rename_label(self, old: str, origin=None) -> None:
        """Rename a label — every cluster under it — after asking for the new name, over the
        window that asked."""
        if not self._can_edit:
            return
        parent = origin or self
        new, ok = QInputDialog.getText(
            parent, "Rename label", f"Rename “{old}” to:", text=old
        )
        new = new.strip()
        if not ok or not new or new == old:
            return
        clash = next(
            (n for n in self._names.values() if n.casefold() == new.casefold()), None
        )
        if PATH_SEPARATOR in new or (clash and clash.casefold() != old.casefold()):
            QMessageBox.warning(
                parent,
                "Label not renamed",
                (
                    f"There is already a label called “{clash}”. To put these clusters under "
                    "it, use Merge into another instead."
                    if clash
                    else f"A label may not contain {PATH_SEPARATOR!r}: it separates the levels "
                    "of a nested label in the exported output."
                ),
            )
            return
        self._change(
            f"Renamed “{old}” to “{new}”",
            {k: (new if v == old else v) for k, v in self._names.items()},
            [new if n == old else n for n in self._recent],
            lambda: self._run.rename_label(self._label_id(old), new),
        )

    def _merge_label(self, source: str, origin=None) -> None:
        """Move every cluster of ``source`` to a label chosen from those it may join, asking
        over the window that asked."""
        if not self._can_edit:
            return
        parent = origin or self
        targets = merge_targets(self._view, self._names, source)
        if not targets:
            QMessageBox.information(
                parent,
                "Nothing to merge into",
                f"No label shares “{source}”'s parent label, and a label's clusters must share "
                "one. Label the branch that holds them instead.",
            )
            return
        into, ok = QInputDialog.getItem(
            parent,
            "Merge label",
            f"Move every cluster of “{source}” into:",
            targets,
            0,
            False,
        )
        if not ok or not into:
            return
        self._change(
            f"Merged “{source}” into “{into}”",
            {k: (into if v == source else v) for k, v in self._names.items()},
            [n for n in self._recent if n != source],
            lambda: self._run.merge_labels(
                self._label_id(source), self._label_id(into)
            ),
        )

    # --- one history of label changes: ⌘Z here and in every cluster window ------------------

    def _change(
        self,
        text: str,
        names: dict,
        recent: list,
        write,
        *,
        origins: dict | None = None,
    ) -> None:
        """Show a label change at once, record it for ⌘Z, and queue its write — which reads the
        run's label maps either side of itself, so an undo can put them back by id. ``origins``
        is the page's copy of where each label came from after the change; left out, it keeps
        what it has."""
        from detail.api.authored_run import label_change
        from detail.gui.label_history import Change, restore_side

        sides = {
            False: (dict(self._names), list(self._recent), dict(self._origins)),
            True: (
                dict(names),
                list(recent),
                dict(self._origins if origins is None else origins),
            ),
        }
        written: list = [None]
        run = self._run

        def show(after: bool) -> None:
            self._names, self._recent, self._origins = (
                dict(sides[after][0]),
                list(sides[after][1]),
                dict(sides[after][2]),
            )
            self._refresh_labels()

        def put(after: bool) -> None:
            labels, nodes, stamps, origins_side = restore_side(written[0], after)
            if labels or nodes or origins_side:
                run.restore_labels(labels, nodes, stamps, origins_side)

        def work():
            before = run.label_maps()
            write()
            return label_change(before, run.label_maps())

        def landed(result) -> None:
            written[0] = result

        self._record(Change(text, show, put), (work, landed))

    def _record(self, change, first_write) -> None:
        """Record ``change`` for ⌘Z, show it, and queue ``first_write`` (``(work, landed)``)."""
        self._history.record(change)
        self._label_bar.confirm(change.text)
        change.show(True)
        self._pending.append(first_write)
        self._write_next()
        self._sync_undo()

    def undo(self) -> None:
        """Undo the last label change, made here or in any of this run's windows (⌘Z)."""
        if self._can_edit and self._history.can_undo():
            self._step(self._history.undo(), after=False)

    def redo(self) -> None:
        """Make again the last change :meth:`undo` took back (⇧⌘Z)."""
        if self._can_edit and self._history.can_redo():
            self._step(self._history.redo(), after=True)

    def _step(self, change, *, after: bool) -> None:
        self._label_bar.confirm(f"{'Redid' if after else 'Undid'}: {change.text}")
        change.show(after)
        self._pending.append((lambda: change.write(after), None))
        self._write_next()
        self._sync_undo()

    def _sync_undo(self) -> None:
        from detail.gui.label_history import show_undo

        reason = (
            "" if self._can_edit else self._edit_reason() or "The run is still loading."
        )
        show_undo(self._label_bar.undo, self._history, reason)
        for window in self._windows.windows():
            show_undo(window.bar.undo, self._history, reason)

    def _write_next(self) -> None:
        if self._writer is not None or not self._pending:
            # Nothing left to start: leave "Output up to date." standing (blanking it here, as
            # this did, wiped it the instant the last write's `done` had set it).
            if self._pending:
                self._label_bar.set_saving("Saving…")
            return
        write, landed = self._pending.pop(0)
        self._label_bar.set_saving("Updating the output…")
        writer = BackgroundCall(write, receiver=self)
        self._writer = writer

        def done(result) -> None:
            self._writer = None
            if landed is not None:
                landed(result)
            if not self._pending:
                self._label_bar.set_saving("Output up to date.")
                self.changed.emit()
            self._write_next()
            self._finalize_when_saved()

        def failed(exc) -> None:
            self._writer = None
            self._pending.clear()
            # What the history recorded may no longer be what the run holds.
            self._history.clear()
            self._label_bar.set_saving("")
            # A finish waiting on this write does not go ahead over a label that was lost.
            self._finishing = False
            self._apply_editability()
            QMessageBox.critical(self, "Label not saved", str(exc))
            self.reload()

        writer.done.connect(done)
        writer.failed.connect(failed)
        writer.finished.connect(writer.deleteLater)
        writer.start()

    # --- rebuilding, and the lifecycle -----------------------------------------------------

    def _rebuild_cache(self) -> None:
        """Start a build for this run, or follow the one already running for it.

        Through ``node_cache_jobs`` rather than a worker owned by this tab: a build takes minutes,
        and leaving the page must neither lose track of it nor let a second one start.
        """
        run = self._run

        def work(emit):
            run.rebuild_node_cache(progress=emit)

        self._attach(node_cache_jobs.start(run._run_dir(), work))

    def _attach(self, job) -> None:
        """Show ``job``'s progress from where it is, and reload when it ends."""
        if job is self._job:
            return  # already following it: a second set of connections would reload twice
        self._job = job
        self._missing.setVisible(False)
        self._split.setVisible(False)
        self._building.setVisible(True)
        self._on_build_progress(job.permille, job.status)
        for widget in (self._rebuild, self._build):
            widget.setEnabled(False)
        job.progress.connect(self._on_build_progress)
        job.finished.connect(self._on_build_finished)
        job.failed.connect(self._on_build_failed)

    def _on_build_progress(self, permille: int, status: str) -> None:
        self._build_bar.setValue(int(permille))
        self._build_status.setText(
            f"Building the {self._terms.unit} clips: {status} ({permille / 10:.0f}%)"
        )

    def _on_build_finished(self) -> None:
        self._job = None
        self._building.setVisible(False)
        self.reload()

    def _on_build_failed(self, exc) -> None:
        self._job = None
        self._building.setVisible(False)
        QMessageBox.critical(self, "Building the clips failed", str(exc))
        self.reload()

    def _edit_reason(self) -> str:
        """Why this run's labels cannot change here, or ``""`` where they can."""
        if not getattr(self._run, "editable", False):
            return (
                "This is a snapshot of the run as it was, frozen by design (§21). Go back to "
                "the live run to change it."
            )
        if self._finishing:
            return (
                "The run is being finished: its labels cannot change until that is done. "
                "Reopen it afterwards to change them."
            )
        if self._state is not None and self._run.status == "complete":
            return "This run is finished. Reopen it to change it."
        return ""

    def _apply_editability(self) -> None:
        editable = bool(getattr(self._run, "editable", False))
        loaded = self._state is not None
        finished = loaded and self._run.status == "complete"
        self._can_edit = loaded and editable and not finished and not self._finishing
        why = self._edit_reason()
        set_available(
            self._finish,
            self._can_edit,
            why or "The run is still loading.",
        )
        set_available(
            self._reopen,
            loaded and editable and finished,
            why if not editable else "The run is open already.",
        )
        self._rebuild.setEnabled(loaded and editable)
        self._build.setEnabled(loaded and editable)
        # A snapshot reads its run's saved screens; running one is the live run's to do, like
        # a rebuild — not because a screen changes the snapshot, but because every control that
        # writes is dead on a frozen view, and this one writes a comparison.
        set_available(
            self._filters.screens.run_button,
            loaded and editable,
            (
                "The run is still loading."
                if not loaded
                else "Open the live run to run a difference screen; a snapshot lists the "
                "run's saved screens only."
            ),
        )
        dead = why or "The run is still loading."
        self._label_field.set_editable(self._can_edit, dead)
        self._label_bar.quick.set_editable(self._can_edit, dead)
        self._icicle.set_edit_reason("" if self._can_edit else dead)
        self._sync_undo()
        self._rebuild.setToolTip(
            "" if editable else "A snapshot cannot be rebuilt; open the live run."
        )
        self._windows.refresh()

    def _finish_run(self) -> None:
        """**Finish**: from the click, nothing labels — here, in a window, by key or by ⌘Z —
        and the finalize starts once the label writes still queued have landed. Run beside the
        queue, it read frame views a write had just made out of date and failed, leaving the
        run open."""
        if not self._can_edit:
            return
        self._finishing = True
        self._apply_editability()
        self._finalize_when_saved()

    def _finalize_when_saved(self) -> None:
        if not self._finishing or self._finalizing:
            return
        if self._writer is not None or self._pending:
            self._label_bar.set_saving("Saving the labels, then finishing the run…")
            return
        self._finalizing = True
        run = self._run

        def work(emit):
            return run.finalize(progress=emit)

        def ended() -> None:
            self._finishing = self._finalizing = False
            self._apply_editability()

        def done(_state) -> None:
            ended()
            self.status_changed.emit()

        def failed(exc) -> None:
            ended()
            QMessageBox.critical(self, "The run was not finished", str(exc))

        run_progress(self, work, done, label="Finishing…", on_error=failed)

    def _reopen_run(self) -> None:
        self._run.reopen()
        self._apply_editability()
        self.status_changed.emit()

    def export_scope(self):
        """The selected node's label, or ``None`` for the whole run."""
        from detail.api.exporters import ExportScope

        if self._selected is None or not self._names.get(str(self._selected)):
            return None
        return ExportScope(cluster=str(self._selected))

    # --- what tests and the docs read --------------------------------------------------------

    def montage(self) -> LiveMontage:
        return self._montage

    def filters(self) -> FilterBar:
        return self._filters

    def label_field(self) -> LabelField:
        return self._label_field

    def label_bar(self) -> LabelBar:
        return self._label_bar

    def montage_order(self) -> list[int]:
        """The selected cluster's clips, as indices into its ``ClipTable``, in montage order."""
        return [int(i) for i in self._order]


def clip_sequence(tab, clips, order) -> list:
    """The clip window's previous / next sequence: ``clips`` in ``order``, each a
    ``ClipSpec`` named by its place and recording."""
    from detail.gui.behavior_query import ClipSpec

    names = tab._view.recording_names
    sequence = []
    for rank, i in enumerate(order, start=1):
        rid = clips.recording_ids[int(clips.recording[i])]
        sequence.append(
            ClipSpec(
                rid,
                int(clips.subject[i]),
                tab._state.embed_run_for(rid),
                int(clips.start_ns[i]),
                int(clips.end_ns[i]),
                f"#{rank} · {names.get(rid, rid)}",
            )
        )
    return sequence


_ORDER_NOTES = (
    ("typical", "most typical first"),
    ("atypical", "least typical first"),
    ("random", "in random order"),
)
