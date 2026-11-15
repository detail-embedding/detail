"""GUI shell: a top mode banner + a tree rail (project / Quick tools) + a top context label.

Navigation is two-level (docs/gui_design.md §3.1). The **banner** (``_ModeBanner``) is the
global scope switch — a flat, exclusive row of modes (``_MODES``): **Experiments**, **Active
processes**, **Configuration**, **Model manager**, **Keypoints manager**, **Quick tools**,
**Docs**. Every mode but Quick tools renders **full-width with no sidebar**; sub-navigation
there is **drill-in** — the landing shows the mode's items and an ``Open →`` (or a row) swaps
in a full-width sub-page (Configuration's GPU / CUDA, SAM 3 backend and Hugging Face steps; a
Model manager model's detail page).

The left **rail** (a ``QTreeWidget``) appears in two scopes, one at a time (``self._rail``):
the **project view**, reached by opening an experiment from Experiments, listing an
**Overview** node and then that experiment's sections (``gui/sections.py``) — Recording
information / Recordings / Datasets / Models / Processing pipeline / Analysis / Comparisons;
and **Quick tools**, listing each registered tool with its saved
runs beneath. Both are folders that expand to their own items, and both follow one rule:
clicking a folder toggles it (and shows its page), clicking a leaf opens its page. A
structural change refreshes only the rail on screen (``_refresh_active_experiment`` /
``_refresh_tools_rail``), never a full rebuild, so expansion survives. The top bar shows the
current context as read-only identity — no Back button, since the banner (and the rail)
always offer a way anywhere. The shell talks to the backend only through ``Experiment`` /
``Recording`` (Tenet 2).

A project section's **children are read when the section opens**, off the UI thread
(``_fill_section``). Building them with the tree cost one record read per recording and per
run — on the UI thread, for children nobody can see until a folder is expanded — which is
seconds of frozen window on a workspace held on a network share (#316). So the rebuild after
a structural change touches no disk, and the re-expand that follows it re-reads exactly the
folders the user had open. Both the toggle and the ▸ therefore come from the node's
declaration (``_FOLDER``), never from ``childCount()``, which a closed section has none of.
"""

import contextlib
import html

from PyQt6.QtCore import QByteArray, QSettings, QSize, Qt, QThreadPool, QTimer
from PyQt6.QtGui import QCursor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detail import help as help_mod
from detail import surfaces
from detail.api import active as api_active
from detail.api import exporters as api_exporters
from detail.api import run_spec
from detail.api import tools as api_tools
from detail.api.experiment import Experiment
from detail.api.exporters import ExportSubject
from detail.api.readiness import ReadinessSignals
from detail.core import buildinfo
from detail.gui.active_processes import ActiveProcessesPage
from detail.gui import comparison as gui_comparison
from detail.gui import export as gui_export
from detail.gui import pages as gui_pages
from detail.gui import preview_jobs
from detail.gui import readiness_guide
from detail.gui import sections as gui_sections
from detail.gui import notes as gui_notes
from detail.gui import selectable
from detail.gui.editing import confirm_leave
from detail.gui.grip_splitter import GripSplitter
from detail.gui.help_ui import HelpPopover
from detail.gui.import_queue import ImportQueue
from detail.gui.lifetime import alive
from detail.gui.scroll import PageScroll
from detail.gui.status_color import ERROR, set_label_color
from detail.gui.version_history import BuildLabel, VersionHistoryPopover
from detail.gui.workers import BackgroundCall, import_progress_fraction
from detail.storage import read_cache
from detail.storage import recording as recording_store

_ROLE = Qt.ItemDataRole.UserRole
#: Whether a rail node is a **folder** — a node whose click toggles expansion as well as
#: navigating. Declared when the node is built, never inferred from ``childCount()``: a
#: project section's children are read when it opens, so a closed folder has none, and a
#: predicate that asks the *children* whether the node is a folder answers "no" and the
#: section never opens. The tools rail sets it from its own records, so ``_on_tree_click``
#: keeps one rule for both rails.
_FOLDER = Qt.ItemDataRole.UserRole + 1

_DISMISSED_KEY = "guidance/dismissed"  # QSettings: comma-joined dismissed step kinds
#: The window's size and position, and where the rail divider sits. Persisted because
#: `docs/gui_design.md` has claimed for a long time that they are and they were not — and
#: because `_discard_content` rebuilds the page on every navigation, so anything the user
#: drags inside one is gone the moment they click elsewhere. These two survive it.
_GEOMETRY_KEY = "window/geometry"
#: Where the rail's width and open/shut are kept (``panes/navigation/``, `grip_splitter`).
NAV_SPLIT_KEY = "navigation"
#: The keys the rail used before every side panel folded one way (2026-11-09): carried into
#: ``panes/navigation/`` once, then removed.
_LEGACY_SPLIT_KEY = "window/rail_split"
_LEGACY_NAV_HIDDEN_KEY = "window/nav_hidden"
_LEGACY_NAV_WIDTH_KEY = "window/nav_width"
#: The rail's widest; also its width when it is shown again with no remembered split.
_RAIL_PX = 260
#: The rail's narrowest open; dragged under it, the rail shuts to its strip.
_RAIL_MIN_PX = 160


def _load_dismissed() -> set[str]:
    raw = QSettings().value(_DISMISSED_KEY, "")
    return {kind for kind in str(raw).split(",") if kind}


def _save_dismissed(dismissed: set[str]) -> None:
    QSettings().setValue(_DISMISSED_KEY, ",".join(sorted(dismissed)))


#: What the window opens at on a screen big enough to hold it.
_PREFERRED_SIZE = (1400, 900)


def _opening_size() -> QSize:
    """The preferred opening size, capped to the display that has to show it.

    ``resize()`` does not fail on a size the screen cannot hold — it succeeds, and the window
    hangs off the bottom of the display where the title bar can no longer reach it. Measured on
    a 13" MacBook: ``availableGeometry`` is 1470x833 and the hardcoded 900 asked for 67px that
    do not exist.

    ``availableGeometry`` rather than ``geometry``: it already excludes the menu bar and the
    Dock, which is the space a window may actually occupy.
    """
    width, height = _PREFERRED_SIZE
    screen = QApplication.primaryScreen()
    if screen is None:  # offscreen platforms in CI have no screen to consult
        return QSize(width, height)
    available = screen.availableGeometry()
    return QSize(min(width, available.width()), min(height, available.height()))


def _named(names: list[str]) -> str:
    """Up to three names, then how many more — a batch of sixty is not one sentence."""
    shown = ", ".join(names[:3])
    return shown if len(names) <= 3 else f"{shown} and {len(names) - 3} more"


def _import_notice_text(results: list[tuple[str, bool, str]]) -> str:
    """What the strip under **Export…** says when a batch of imports has finished.

    ``results`` is ``[(recording name, ok, error), ...]``; an empty batch says nothing.

    It replaces two status-bar messages — *"Imported recording_001"* for eight seconds, then
    *"Imports complete"* for five — along the bottom edge of the window, where the user who had
    just imported a 533 MB video did not see either. On a local disk an import is usually over
    within a second, so this is often the only sign one happened, which is why it names the
    recordings as the user named them (§11) rather than by id.

    Module-level and pure, like `_quit_message`, so the sentence is asserted without a window.
    """
    if not results:
        return ""
    done = [name for name, ok, _ in results if ok]
    failed = [(name, error) for name, ok, error in results if not ok]
    parts = []
    if len(done) == 1:
        parts.append(f"Imported {done[0]}.")
    elif done:
        parts.append(f"Imported {len(done)} recordings: {_named(done)}.")
    if failed:
        listed = "; ".join(
            f"{name} — {error}" if error else name for name, error in failed
        )
        parts.append(f"{len(failed)} failed: {listed}.")
    parts.append(f"See {'it' if len(results) == 1 else 'them'} under Active processes.")
    return " ".join(parts)


def _anything_survives_quit(active: dict[str, list[str]]) -> bool:
    """Whether any of the work in flight would still be going after the window closes.

    False means the *"Leave running"* button has nothing to leave running, so it stops being
    offered under that name. Off the table rather than a check for the import kind, like the
    sentence beside it.
    """
    return any(
        active.get(spec.kind) for spec in api_active.RUN_KINDS if spec.survives_quit
    )


def _quit_message(active: dict[str, list[str]]) -> str:
    """What the quit prompt says about the work in flight, given ``{kind: [ids]}``.

    It used to say one thing — *"They keep running in the background if you quit without
    cancelling them"* — which was true of every family on the day it was written and became
    false the moment one of them was an **import**: an import is a thread in this process
    driven by a queue that cannot resume, so quitting ends it, and the next time you open the
    experiment it reads ``failed``. A prompt that describes the outcome wrongly is worse than
    one that does not appear, because the user makes a decision on it.

    Which families survive is read off ``RunKind.survives_quit``, not a check for the import
    kind, so the next in-process family is described correctly the day it is added.

    Module-level and pure: this is the sentence worth asserting, and a test should not have to
    build a window and a message box to read it.
    """
    ephemeral = sum(
        len(active.get(spec.kind, []))
        for spec in api_active.RUN_KINDS
        if not spec.survives_quit
    )
    total = sum(len(ids) for ids in active.values())
    detached = total - ephemeral
    if not ephemeral:
        return (
            f"{total} run(s) are still in progress. They keep running in the "
            "background if you quit without cancelling them."
        )
    if not detached:
        return (
            f"{ephemeral} import(s) are still in progress. Imports stop when you quit "
            "and cannot be resumed — the recordings will be marked failed."
        )
    return (
        f"{total} process(es) are still in progress. {detached} run(s) keep going in the "
        f"background if you quit; {ephemeral} import(s) stop and cannot be resumed."
    )


_MODES = ("home", "runs", "config", "models", "keypoints", "tools", "docs")
_MODE_LABELS = {
    "home": "Experiments",
    # Second, beside Experiments, because "is my run still going" is asked far more often than
    # anything in the three modes after it — and it is the question people were leaving the app
    # to answer.
    #
    # **Processes, not runs.** A run is one family of the things in flight here; an export is
    # another, and a tool run will be a third. Naming the mode after one of its sections is how
    # the others end up looking like they belong somewhere else. The mode *key* stays `"runs"`:
    # it is internal, and `kind == "runs"` elsewhere means the project rail's Processing
    # pipeline, which is a different thing that must not be renamed with it.
    "runs": "Active processes",
    "config": "Configuration",
    "models": "Model manager",
    # Skeletons are machine-wide like models, so they sit beside them (§26).
    "keypoints": "Keypoints manager",
    "tools": "Quick tools",
    "docs": "Docs",
}
# Declared once in ``gui/sections.py`` — three copies of this list had already drifted
# (the rail said "Processing pipeline" where the overview said "Runs" for one ``kind``).
_EXPERIMENT_SECTION_KINDS = gui_sections.SECTION_KINDS


class _ModeBanner(QFrame):
    """The top mode banner — an exclusive row of buttons (greyscale, no stylesheet, no
    icons) that switches the whole workspace. It reads as a distinct bar above the content:
    a slightly tinted background (derived from the theme palette, so it holds in light *and*
    dark) plus a sunken bottom edge. This is the global scope switch; clicking a mode is the
    only place scope changes."""

    def __init__(self, on_switch) -> None:
        super().__init__()
        # A distinct bar, palette-only (no QSS): tint the Window role a step off the content
        # background — darker on a light theme, lighter on a dark one — so it stays greyscale
        # and legible either way. A separate sunken divider (added under it in the central
        # layout) draws the boundary line against the content.
        self.setAutoFillBackground(True)
        pal = self.palette()
        window = pal.color(QPalette.ColorRole.Window)
        tint = window.darker(122) if window.lightness() > 128 else window.lighter(145)
        pal.setColor(QPalette.ColorRole.Window, tint)
        self.setPalette(pal)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for mode in _MODES:
            button = QPushButton(_MODE_LABELS[mode])
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, m=mode: on_switch(m))
            self._group.addButton(button)
            self._buttons[mode] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self._layout = layout

    def add_trailing(self, widget: QWidget) -> None:
        """Pin ``widget`` to the bar's right end, past the mode buttons' stretch."""
        self._layout.addWidget(widget)

    def button(self, mode: str) -> QPushButton | None:
        return self._buttons.get(mode)

    def set_active(self, mode: str) -> None:
        button = self._buttons.get(mode)
        if button is not None:
            button.setChecked(True)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DETAIL")
        # Size and rail position are restored at the end of __init__, once the splitter they
        # apply to exists — see `_restore_layout`, which falls back to `_opening_size`.
        self._experiment: Experiment | None = None
        self._import_queue: ImportQueue | None = None
        #: Names by recording id, learned when an import starts. The queue drops its spec before
        #: it reports an item done, so this is where the name survives to be announced.
        self._import_names: dict[str, str] = {}
        #: ``(name, ok, error)`` per finished import in the batch still running.
        self._import_results: list[tuple[str, bool, str]] = []
        self._readiness: ReadinessSignals | None = None
        self._dismissed: set[str] = _load_dismissed()
        self._readiness_signals = None  # kept alive until the off-thread probe emits

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.itemClicked.connect(self._on_tree_click)
        # A project section reads its children when it opens (#316). Every route to open one
        # ends in `setExpanded(True)`, and Qt emits this for a childless item too, so one
        # connection covers the rail click, the post-refresh re-expand and `_go_to_section`.
        self._tree.itemExpanded.connect(self._fill_section)
        #: Bumped by every project-rail rebuild; a fill delivered against an older one is
        #: dropped. `{kind: generation}` records which sections that tree has already read.
        self._rail_gen = 0
        self._rail_filled: dict[str, int] = {}

        self._context = QLabel("")
        self._context.setContentsMargins(8, 4, 8, 4)
        font = self._context.font()
        font.setBold(True)
        self._context.setFont(font)
        # Selectable like every label, and its one link (a page's way back from a view it has
        # open, `_render_context`) clickable by mouse and keyboard.
        self._context.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        self._context.linkActivated.connect(self._on_context_link)
        self._context.linkHovered.connect(self._on_context_link_hovered)
        #: The path `_show` was given; `_render_context` may append the page's open view.
        self._context_base = ""

        # Which build is running, at the right end of the mode banner and always visible: its
        # date is how a user tells at a glance whether their copy is current. Read from git
        # (`core/buildinfo`), off the UI thread since it is several subprocess calls. Not in
        # the strip beside Info, where a label that is always there pushes Export off its edge.
        self._build: buildinfo.BuildInfo | None = None
        self._build_label = BuildLabel()
        self._version_history = QPushButton("Version History")
        self._version_history.setCursor(Qt.CursorShape.PointingHandCursor)
        self._version_history.setToolTip(
            "What changed in recent versions, and check for updates"
        )
        self._version_history.clicked.connect(self._open_version_history)
        build_worker = BackgroundCall(buildinfo.build_info, receiver=self)  # off-thread
        build_worker.done.connect(self._show_build)
        build_worker.start()

        # A page-top `?` beside the context label: help for the current screen. Its topic is
        # the navigated node's kind (PAGE_DOCS); hidden for kinds with no doc.
        self._page_help_kind: str | None = None
        self._page_help = QPushButton("Info")
        self._page_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self._page_help.setToolTip("Help for this page")
        self._page_help.clicked.connect(self._open_page_help)
        self._page_help.hide()

        # The one Export control (#336), beside `Info` in the strip that stays put while the
        # page scrolls. Its subject comes from whichever opener showed the page, so a result
        # page has export the day it becomes navigable and no page hand-lists anything.
        self._export = gui_export.ExportButton()
        self._export.launched.connect(self._on_export_launched)
        # The answer to "what happened to my export" sits under the button that started it,
        # rather than replacing the screen the user was on. `_show` clears it.
        self._export_notice = gui_export.ExportNotice()
        self._export_notice.show_processes.connect(lambda: self._switch_mode("runs"))

        # The page's notes, opened and folded from here rather than from a header on each page:
        # bound to the page on screen (`_bind_page_notes`), hidden on a page with no notes. It
        # opens and folds the notes only where you are; every page and view starts folded.
        self._notes_button = QPushButton(gui_notes.button_text(False))
        self._notes_button.setCheckable(True)
        self._notes_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._notes_button.setToolTip(
            "Show or hide your notes here (● when there is a note). Every page opens with "
            "them folded."
        )
        self._notes_button.clicked.connect(self._toggle_page_notes)
        self._notes_button.hide()
        self._notes_box = None
        gui_notes.notes_hub().created.connect(self._on_notes_created)

        self._content = QStackedWidget()
        # The page scrolls rather than forcing the window taller than the screen. The stack
        # stays the thing pages are added to and removed from (`_show` / `_discard_content`
        # are unchanged); only what holds it changed.
        self._scroll = PageScroll()
        self._scroll.setWidget(self._content)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 8, 0)
        top_layout.addWidget(self._context, 1)
        top_layout.addWidget(self._notes_button)
        top_layout.addWidget(self._export)
        top_layout.addWidget(self._version_history)
        top_layout.addWidget(self._page_help)
        right_layout.addWidget(top_bar)
        right_layout.addWidget(self._export_notice)
        # The context bar stays put; only the page below it scrolls. A header that scrolled
        # away would take the "which run am I looking at" line with it.
        right_layout.addWidget(self._scroll, 1)

        split = GripSplitter(Qt.Orientation.Horizontal, key=NAV_SPLIT_KEY)
        # A stable object name is what makes the saved state re-attachable. Veusz is the warning
        # here: its docks have none, so restore is unreliable and Qt says so at runtime.
        split.setObjectName("rail_split")
        split.addWidget(self._build_rail_box())
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        # The rail folds like every side panel: drag it narrower than its minimum, or
        # double-click its grip, and it shuts to a strip reading "Navigation" (Ctrl+\ too).
        split.set_foldable(0, "Navigation")
        self._split = split

        # The top banner switches scope (Experiments / Configuration / Model manager / Quick
        # tools / Docs); the tree below is the contextual rail for the active mode. A sunken
        # divider under the banner sets it off from the content below.
        self._mode = "home"
        # Which rail the one shared tree currently holds — ``"project"`` | ``"tools"`` |
        # ``None`` (no rail). A refresh helper must check this rather than "is the tree
        # visible": with two rails, a background import finishing while Quick tools is on
        # screen would otherwise rebuild the *project* rail into the tools tree.
        self._rail: str | None = None
        self._banner = _ModeBanner(self._switch_mode)
        self._banner.add_trailing(self._build_label)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self._banner)
        central_layout.addWidget(divider)
        central_layout.addWidget(split, 1)
        self.setCentralWidget(central)

        # The import queue's bar (§1.3). In the status bar rather than on a page because the
        # queue is app-level: the user keeps working while it drains, on whatever page they
        # like. Permanent, so a transient status message cannot cover it.
        self._import_bar = QProgressBar()
        self._import_bar.setRange(0, 1000)
        self._import_bar.setTextVisible(False)  # the message beside it says the rest
        self._import_bar.setMaximumWidth(200)
        self._import_bar.hide()
        self.statusBar().addPermanentWidget(self._import_bar)
        # The preview queue's line and bar (§3 *Video preview cache*), beside the import bar
        # for the same reason: app-level background work that outlives any page. Its own label
        # rather than the status message, which the import queue owns.
        self._preview_label = QLabel("")
        self._preview_label.hide()
        self._preview_label_palette = self._preview_label.palette()
        self.statusBar().addPermanentWidget(self._preview_label)
        self._preview_bar = QProgressBar()
        self._preview_bar.setRange(0, 1000)
        self._preview_bar.setTextVisible(False)
        self._preview_bar.setMaximumWidth(160)
        self._preview_bar.hide()
        self.statusBar().addPermanentWidget(self._preview_bar)
        self._preview_queue: preview_jobs.PreviewQueue | None = None
        self._preview_failures: list[str] = []

        self._refresh_readiness()
        self._restore_layout()  # after the splitter exists, and before anything is shown
        self._switch_mode("home")  # builds the rail + shows the experiments list

    # -- banner mode → full-width page (rail: project + Quick tools) --------- #
    def _open_active_run(self, run) -> None:
        """Open the run a row names, in the experiment it belongs to.

        Switching experiment first is the whole job: a row in this list may be in a project that
        is not the open one, and opening its page without that would show a run detail against
        the wrong experiment — the page reads its record from whichever experiment is current.

        Routing rather than rendering: everything the row summarises is already on that page,
        including the live bar and the activity log.
        """
        from detail.api.experiment import Experiment

        if not self._may_leave():
            return
        experiment = Experiment(run.experiment_dir)
        self._open_experiment(experiment)
        opener = self._active_run_openers().get(run.kind)
        if opener is not None:
            opener(run.run_id)

    def _active_run_openers(self) -> dict:
        """Which page shows each kind of run the Active-runs listing can show.

        Hand-written because the destination is a *method*, which `api.active.RUN_KINDS` has no
        business knowing — so `tests/gui/test_active_run_openers.py` holds it to that table
        instead. It carried two of the six kinds: a dataset, model, clusterer or distill row drew
        the pointing hand and the trailing `›` that `navigable=True` promises, and double-clicking
        it did nothing. Every one of these openers already existed; only the mapping was missing.
        """
        return {
            "run": self._open_run,
            "analysis_run": self._open_analysis_run,
            "distill": self._open_distill_run,
            "dataset": self._open_dataset,
            "model": self._open_model,
            "export": self._open_export_run,
            # The recording being imported *is* this row's own page — it is the thing being
            # made, and once the import fails it is where you delete it. Sending the click to
            # the Recordings table instead would discard the id the row is carrying and leave
            # you to find the same recording by eye.
            "import": self._open_recording,
        }

    def _switch_mode(self, mode: str) -> None:
        """Switch the banner scope. Most modes render **full-width with no sidebar**;
        Quick tools carries its own rail (its tools, each expanding to its saved runs), and
        the project view carries the active experiment's (``_open_experiment``).
        Experiments = the experiments list; Configuration = the Get-started checklist (each step
        drills into a full-width sub-page); Model manager = its landing page; Keypoints manager = the skeleton library
        (§26); Quick tools = its
        landing page + rail; Docs = the docs page. Sub-navigation is drill-in elsewhere.
        """
        if not self._may_leave():
            self._banner.set_active(self._mode)  # undo the button's own selection
            return
        self._mode = mode
        self._banner.set_active(mode)
        self._tree.clear()
        self._rail = None
        self._show_rail(False)  # rebuilt below by the modes that carry a rail
        if mode == "home":
            self._show_experiments_home()
            self._set_page_help("experiments_group")
        elif mode == "runs":
            self._show(
                ActiveProcessesPage(on_open=self._open_active_run), "Active processes"
            )
            # Every other branch here sets the page help and this one did not, so arriving from
            # Configuration left the Info button up and pointing at the Configuration doc.
            self._set_page_help("active_processes")
        elif mode == "config":
            self._show(
                gui_pages.ConfigurationPage(self._go_to_node, self._on_readiness),
                "Configuration",
            )
            self._set_page_help("config_group")
        elif mode == "models":
            self._open_model_manager()
            self._set_page_help("model_manager_group")
        elif mode == "keypoints":
            self._open_keypoints_manager()
            self._set_page_help("keypoints_manager_group")
        elif mode == "tools":
            self._build_tools_rail()  # the rail is up on the landing too, not just a tool
            self._show(gui_pages.ToolsPage(self._go_to_tool), "Quick tools")
            self._set_page_help("tools_group")
        elif mode == "docs":
            self._open_documentation()
            self._set_page_help("documentation")

    def _item(self, label: str, role: tuple, folder: bool = False) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        item.setData(0, _ROLE, role)
        item.setData(0, _FOLDER, folder)
        if folder:
            # Qt's default draws the ▸ only when the item *has* children, and a project
            # section's are not read until it opens — so without this the arrow is gone from
            # every section until you happen to click one, which is the affordance that says
            # it opens at all. Say it from the declaration instead, like the toggle; the fill
            # hands the decision back to Qt (`_apply_section_children`) once the contents are
            # known, so a section that turns out to be empty drops its arrow.
            item.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
            )
        return item

    # -- the project rail (the only place a sidebar appears) ----------------- #
    def _experiment_section_items(self, experiment) -> list[QTreeWidgetItem]:
        """The rail nodes for one experiment, from the declared section list
        (``gui/sections.py``) filtered to what this build shows (§25).

        **Folders only — no disk.** A section's children are read when the section opens
        (:meth:`_fill_section`), not here: this ran on the UI thread and cost one record read
        per recording and per run to build children that are invisible until a folder is
        expanded (#316). Whether a node is a folder comes from the *declaration*
        (``section.children``), so it is known without asking the disk.

        Every node carries its experiment so a click activates the right one.

        **Overview leads, and it is not a section.** The experiment's own page had no rail
        node at all (#340/#331): it was shown once on the way in and was unreachable from
        then on, taking *Export experiment*, *Delete experiment* and the name edit with it.
        It is added here rather than to ``gui/sections.py`` because that list is also what
        ``ExperimentOverviewPage`` renders its summary rows from — a ``Section`` there would
        give the overview a row that opens itself, and the page's tests derive both sides of
        their comparison from the same list, so it would pass. Here rather than in
        ``_build_project_rail`` for the opposite reason: that method rebuilds purely from
        this one, so a node added there is dropped by the next refresh.
        """
        return [
            self._item("Overview", ("experiment", experiment)),
            *(
                self._item(
                    section.title,
                    (section.kind, experiment),
                    folder=section.children is not None,
                )
                for section in gui_sections.sections()
            ),
        ]

    def _build_project_rail(self, experiment) -> None:
        """Show the left rail for the project view — the active experiment's sections at the
        top level, their children left to arrive when a section opens.

        Bumping the generation here (never in the fill) is what makes a rebuild supersede
        every request made against the tree it replaced — the same split Home draws between
        ``_refresh`` and ``_load_facts``.
        """
        self._tree.clear()
        self._rail_gen += 1
        self._rail_filled.clear()
        for item in self._experiment_section_items(experiment):
            self._tree.addTopLevelItem(item)
        self._rail = "project"
        self._show_rail(True)

    def _fill_section(self, item: QTreeWidgetItem) -> None:
        """Read one project-rail section's children off the UI thread, once it is open.

        Every way a section opens lands here — a rail click, ``_reexpand_top_roles`` after a
        refresh, an overview page's ``Open →`` — because they all end in ``setExpanded(True)``
        and Qt emits ``itemExpanded`` for a childless item too. So the fill needs no second
        trigger beside the toggle rule, and a section added later inherits it.

        The read is O(recordings) / O(runs) small-file reads: seconds on a network workspace,
        which is why it is off the UI thread (CLAUDE.md) rather than merely cheap here.
        """
        if self._rail != "project" or self._experiment is None:
            return  # the tools rail expands its own nodes in `_build_tools_rail`
        kind = item.data(0, _ROLE)[0]
        section = next(
            (s for s in gui_sections.sections() if s.kind == kind and s.children), None
        )
        if section is None:
            return
        generation = self._rail_gen
        if self._rail_filled.get(kind) == generation:
            return  # already read for the tree on screen — an expand/collapse/expand is free
        self._rail_filled[kind] = generation
        experiment = self._experiment
        worker = BackgroundCall(
            lambda: section.child_items(experiment), receiver=self
        )  # off-thread
        worker.done.connect(
            lambda items, g=generation, k=kind, ck=section.child_kind: (
                self._apply_section_children(g, k, ck, items)
            )
        )
        worker.failed.connect(
            lambda _exc, k=kind: self._rail_filled.pop(k, None)
        )  # the folder stays empty; opening it again retries
        worker.start()

    def _apply_section_children(
        self, generation: int, kind: str, child_kind: str | None, items
    ) -> None:
        # A rebuild replaced the tree this result was read for — its nodes are gone and the
        # section will be re-read when it re-opens. The window going away is the other way
        # these nodes disappear, and a generation cannot see it (nothing runs on the way
        # out), so that half is the worker's `receiver=self` (`GuiWorker`).
        if generation != self._rail_gen:
            return
        node = self._top_node(
            kind
        )  # re-found, never a held pointer into a cleared tree
        if node is None:
            return
        node.takeChildren()  # idempotent: a second delivery replaces, never appends
        node.addChildren(
            [self._item(label, (child_kind, child_id)) for label, child_id in items]
        )
        # The contents are known now, so Qt can go back to drawing the ▸ only when there is
        # something behind it — a section that really is empty stops advertising rows.
        node.setChildIndicatorPolicy(
            QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicatorWhenChildless
        )

    # -- the Quick tools rail (§9) ------------------------------------------- #
    def _tool_rail_items(self) -> list[QTreeWidgetItem]:
        """The rail nodes for Quick tools — one folder per registered tool, its saved runs
        beneath it oldest first (store order, like every project-rail section). One sweep of
        the tools workspace builds the whole rail, not one listing per tool."""
        by_tool: dict[str, list[dict]] = {}
        for record in api_tools.list_tool_runs():
            by_tool.setdefault(record["tool_id"], []).append(record)
        items = []
        for info in api_tools.list_tools():
            records = by_tool.get(info.tool_id, [])
            # Built eagerly (one sweep covers every tool), so a tool with no saved runs is a
            # leaf, exactly as before — the flag only spells out what `childCount()` said.
            tool = self._item(info.label, ("tool", info.tool_id), folder=bool(records))
            for record in records:
                tool.addChild(
                    self._item(
                        f"{record['id']}  ·  {record.get('status')}",
                        ("tool_run", (info.tool_id, record["id"])),
                    )
                )
            items.append(tool)
        return items

    def _build_tools_rail(self) -> None:
        """Show the left rail for Quick tools — the tools at the top level, each expanding to
        its saved runs. Folders open by default: with one tool a collapsed rail shows nothing
        useful (a later collapse survives ``_refresh_tools_rail``)."""
        self._tree.clear()
        for item in self._tool_rail_items():
            self._tree.addTopLevelItem(item)
            item.setExpanded(True)
        self._rail = "tools"
        self._show_rail(True)

    # -- targeted tree refresh (never a full rebuild — expansion survives) ---- #
    def _top_node(self, kind: str) -> QTreeWidgetItem | None:
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.data(0, _ROLE)[0] == kind:
                return item
        return None

    def _refresh_experiments_list(self) -> None:
        """The experiments list is its own page (not the rail) — re-read it if shown."""
        self._refresh_shown()

    def _refresh_active_experiment(self) -> None:
        """Re-read the active experiment's rail sections in place after a structural change
        (run/recording/analysis added or deleted), preserving which folders were open. A
        no-op unless the project rail is currently shown."""
        # ``self._rail``, not "is the tree visible": Quick tools has a rail of its own now,
        # and a background import finishing while it is on screen must not overwrite it.
        if self._experiment is None or self._rail != "project":
            return
        open_roles = self._expanded_top_roles()
        self._build_project_rail(self._experiment)
        self._reexpand_top_roles(open_roles)

    def _refresh_tools_rail(self) -> None:
        """Re-read the Quick tools rail in place after a run is launched or deleted,
        preserving which tools were open. A no-op unless that rail is currently shown.
        """
        if self._rail != "tools":
            return
        open_roles = self._expanded_top_roles()
        self._build_tools_rail()
        self._reexpand_top_roles(open_roles)

    def _expanded_top_roles(self) -> set[tuple]:
        """Which top-level rail nodes are open, by role — what a rebuild restores from. Both
        rails re-use the identical role for a surviving node (the same ``Experiment`` object,
        the same tool id), so the role is a stable key across the rebuild."""
        return {
            self._tree.topLevelItem(i).data(0, _ROLE)
            for i in range(self._tree.topLevelItemCount())
            if self._tree.topLevelItem(i).isExpanded()
        }

    def _reexpand_top_roles(self, roles: set[tuple]) -> None:
        """Re-open the top-level nodes a rebuild replaced, and close the rest — so a folder
        the user collapsed stays collapsed even when the builder opens it by default."""
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            item.setExpanded(item.data(0, _ROLE) in roles)

    def _select_rail_node(self, role: tuple) -> None:
        """Move the rail selection onto the node for ``role`` (a top-level item or one of its
        children), opening its parent. Every path into a page — a rail click, a landing
        button, a table double-click, the jump after a run — goes through the opener that
        calls this, so the rail always shows where you are."""
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.data(0, _ROLE) == role:
                self._tree.setCurrentItem(item)
                return
            for j in range(item.childCount()):
                child = item.child(j)
                if child.data(0, _ROLE) == role:
                    item.setExpanded(True)
                    self._tree.setCurrentItem(child)
                    return

    def _refresh_training(self) -> None:
        """A dataset, model or distill run was added or removed: re-read the rail's Datasets
        and Models folders, and the page on screen if it lists them (the Models page, the
        Model manager's experiment table)."""
        self._refresh_active_experiment()
        self._refresh_shown()

    def _on_tree_click(self, item: QTreeWidgetItem, _column: int) -> None:
        # One rule for every project-rail node: a click **always shows the clicked node's
        # page**, and a folder (a section with children) also **toggles** its expansion.
        # (Previously a click on an already-open folder only collapsed it and left the prior
        # screen up — inconsistent.) The toggle intent is captured before ``_navigate`` and
        # enforced after, so collapsing still sticks; ``_navigate`` only says *what to show*,
        # never how to expand — a new section kind gets this for free.
        #
        # Folder-ness is the node's **declaration** (``_FOLDER``), not ``childCount()``: a
        # project section's children are read when it opens, so a closed folder has none and
        # asking the children would answer "not a folder" and never open it (#316).
        if not self._may_leave():
            return
        folder = bool(item.data(0, _FOLDER))
        want_expanded = not (folder and item.isExpanded())
        self._navigate(item)
        if folder:
            item.setExpanded(want_expanded)

    def _may_leave(self) -> bool:
        """Ask the page on screen about unsaved edits. ``False`` means stay put.

        Every navigation entry point calls this *first* — before building the next page
        or moving the rail selection — so answering Cancel genuinely leaves you where you
        were. A page that holds nothing (almost all of them) passes straight through.
        """
        current = self._content.widget(0) if self._content.count() else None
        return confirm_leave(self, current)

    def _set_page_help(self, kind: str | None) -> None:
        """Point the page-top `?` at the current screen's doc (hidden if the kind has none)."""
        self._page_help_kind = kind
        self._page_help.setVisible(kind in help_mod.PAGE_DOCS)

    def _open_page_help(self) -> None:
        topic = help_mod.PAGE_DOCS.get(self._page_help_kind)
        if topic is None:
            return
        self._page_help_popover = HelpPopover(
            topic, self._page_help
        )  # keep a reference
        self._page_help_popover.show_at(self._page_help)

    def _show_build(self, info: buildinfo.BuildInfo) -> None:
        self._build = info
        self._build_label.set_build(info)

    def _open_version_history(self) -> None:
        self._version_history_popover = VersionHistoryPopover(
            self._version_history, self._build
        )  # keep a reference
        self._version_history_popover.show_at(self._version_history)

    def _navigate(self, item: QTreeWidgetItem) -> None:
        """Run a rail node's navigation action — show its page (and, in a project, activate
        its experiment). One dispatcher for both rails: the active project's sections and
        Quick tools' tools + saved runs; kinds are distinct across the two, so a node knows
        its own destination. Expansion is applied uniformly by ``_on_tree_click``, never here.
        """
        kind, payload = item.data(0, _ROLE)
        # The opener is the toggle point, not the rail (§25). The rail is one of several doors
        # into a section — a landing button, a table double-click and the jump after a run are
        # the others — and they all funnel through here, so a hidden section is unreachable by
        # every route rather than merely unlisted by one.
        if not surfaces.is_surfaced(kind):
            return
        self._set_page_help(kind)
        if kind == "experiment":
            self._ensure_active(payload)
            self._open_experiment_overview()
        elif kind == "metadata":
            self._ensure_active(payload)
            self._show(
                gui_pages.MetadataPage(self._experiment, on_open=self._open_recording),
                self._experiment.name,
            )
        elif kind == "recordings":
            self._ensure_active(payload)
            self._show(self._recordings_page(), self._experiment.name)
        elif kind == "recording":
            experiment, recording_id = payload
            self._ensure_active(experiment)
            self._open_recording(recording_id)
        elif kind == "datasets":
            self._ensure_active(payload)
            self._open_datasets()
        elif kind == "dataset":
            experiment, dataset_id = payload
            self._ensure_active(experiment)
            self._open_dataset(dataset_id)
        elif kind == "models":
            self._ensure_active(payload)
            self._open_models()
        elif kind == "model":
            experiment, model_id = payload
            self._ensure_active(experiment)
            self._open_model(model_id)
        elif kind == "runs":
            self._ensure_active(payload)
            self._show(
                gui_pages.ExperimentRunsPage(
                    self._experiment, self._open_run, self._on_run_launched
                ),
                self._experiment.name,
            )
        elif kind == "run":
            experiment, run_id = payload
            self._ensure_active(experiment)
            self._open_run(run_id)
        elif kind == "analysis":
            self._ensure_active(payload)
            self._open_analysis()
        elif kind == "analysis_run":
            experiment, run_id = payload
            self._ensure_active(experiment)
            self._open_analysis_run(run_id)
        elif kind == "comparisons":
            self._ensure_active(payload)
            self._open_comparisons()
        elif kind == "comparison":
            experiment, comparison_id = payload
            self._ensure_active(experiment)
            self._open_comparison(comparison_id)
        elif kind == "tool":
            self._open_tool(payload)
        elif kind == "tool_run":
            self._open_tool_run(*payload)

    # -- navigation targets -------------------------------------------------- #
    def _open_comparisons(self) -> None:
        self._show(
            gui_comparison.ComparisonsPage(
                self._experiment, self._open_comparison, self._on_comparison_created
            ),
            self._experiment.name,
        )

    def _open_comparison(self, comparison_id: str) -> None:
        self._show(
            gui_comparison.ComparisonDetailPage(
                self._experiment,
                comparison_id,
                self._on_comparison_deleted,
                self._on_comparison_created,
            ),
            self._experiment.name,
        )
        self._set_page_export("comparison", comparison_id)

    def _on_comparison_created(self, comparison_id: str) -> None:
        self._refresh_active_experiment()  # the new one belongs in the rail
        self._open_comparison(comparison_id)

    def _on_comparison_deleted(self) -> None:
        self._refresh_active_experiment()  # the comparison is gone from Comparisons
        self._open_comparisons()

    def _open_model_manager(self) -> None:
        """The Model manager mode — every model on one page: Global models (imported or
        promoted), the active experiment's trained models, and the Hugging Face foundation
        models, Distill + Import on top."""
        self._show(
            gui_pages.ModelManagerPage(
                self._experiment,
                self._open_global_model,
                self._open_experiment_model,
                self._on_distill_launched,
                banner=self._banner_for("global_models"),
                on_banner_dismiss=lambda: self._dismiss_step("global_models"),
                on_readiness_changed=self._on_readiness_changed,
            ),
            "Model manager",
        )

    def _open_experiment_model(self, model_id: str) -> None:
        """A trained model from the Model manager's experiment table. Enters the project
        first, the way an Active-processes row does, so the page arrives with its rail and
        a way back to the rest of the experiment."""
        if not self._may_leave() or self._experiment is None:
            return
        self._open_experiment(self._experiment)
        self._open_model(model_id)

    def _open_global_model(self, model_id: str) -> None:
        """One of your models' detail (metadata + provenance + preview) — a drill-in."""
        self._show(
            gui_pages.GlobalModelDetailPage(
                model_id,
                on_deleted=lambda: self._switch_mode("models"),
                on_changed=self._refresh_shown,
            ),
            f"Model manager  ›  {model_id}",
        )

    def _banner_for(self, kind: str) -> str | None:
        """Banner text for a Configuration page, or None — shown only when that page's
        step is a guiding, incomplete, un-dismissed one (readiness_guide)."""
        if self._readiness is None or kind in self._dismissed:
            return None
        step = readiness_guide.guides_step(kind, self._readiness)
        # "Still to do", not "Next step": the sidebar ● marks the single next step, but a
        # page banners any guiding step that's incomplete (which may not be first).
        return f"Still to do — {step.label}." if step is not None else None

    def _open_documentation(self) -> None:
        # The full docs site is built locally on first open and served over file://
        # (detail/docs_access.py); nothing is published yet.
        self._show(gui_pages.DocumentationPage(), "Documentation")

    def _go_to_node(self, kind: str) -> None:
        """Navigate to a machine-global destination from a Configuration checklist button — a
        full-width drill-in (no rail). GPU / CUDA and Hugging Face are Configuration
        sub-pages (breadcrumb back via the banner); Install (Model installs) and Distill open
        the Model manager mode; docs opens the Docs mode."""
        if not self._may_leave():
            return
        if kind == "gpu_cuda":
            self._show(gui_pages.GpuCudaPage(), "Configuration  ›  GPU / CUDA")
            self._set_page_help("gpu_cuda")
        elif kind == "sam3_backend":
            self._show(
                gui_pages.Sam3BackendPage(
                    on_readiness_changed=self._on_readiness_changed,
                ),
                "Configuration  ›  SAM 3 backend",
            )
            self._set_page_help("sam3_backend")
        elif kind == "hugging_face":
            self._show(
                gui_pages.HuggingFacePage(
                    banner=self._banner_for("hugging_face"),
                    on_banner_dismiss=lambda: self._dismiss_step("hugging_face"),
                    on_readiness_changed=self._on_readiness_changed,
                ),
                "Configuration  ›  Hugging Face",
            )
            self._set_page_help("hugging_face")
        elif kind in ("global_models", "distill_model"):
            # Both Configuration checklist steps land on the Model manager: it hosts the
            # Hugging Face install table and the Distill button on one page.
            self._switch_mode("models")
        elif kind == "quick_segment":
            # The first-run "wow": point the checklist at the tool itself, so seeing a
            # clip segmented is one click from setup rather than a dead step.
            self._switch_mode("tools")
        elif kind == "documentation":
            self._switch_mode("docs")

    def _go_to_section(self, kind: str) -> None:
        """Open a section from an overview page's Open → button — every section lives in
        the project rail, so this selects its node and shows it."""
        if not surfaces.is_surfaced(kind) or kind not in _EXPERIMENT_SECTION_KINDS:
            return
        if not self._may_leave():
            return
        node = self._top_node(kind)  # the project rail (shown while a project is open)
        if node is not None:
            self._tree.setCurrentItem(node)
            node.setExpanded(True)
            self._navigate(node)

    # -- tools (§9) ---------------------------------------------------------- #
    def _tool_label(self, tool_id: str) -> str:
        for info in api_tools.list_tools():
            if info.tool_id == tool_id:
                return info.label
        return tool_id

    def _open_tool(self, tool_id: str) -> None:
        """Show one tool's launch page and put the rail on its node — every way in (a rail
        click, the landing's Open button) lands here, so the selection can't drift."""
        self._select_rail_node(("tool", tool_id))
        self._show(
            gui_pages.ToolRunPage(
                tool_id,
                on_launched=lambda run_id: self._on_tool_run_launched(tool_id, run_id),
                on_open_run=lambda run_id: self._open_tool_run(tool_id, run_id),
            ),
            f"Quick tools  ›  {self._tool_label(tool_id)}",
        )

    def _open_tool_run(self, tool_id: str, run_id: str) -> None:
        self._select_rail_node(("tool_run", (tool_id, run_id)))
        self._show(
            gui_pages.ToolRunDetailPage(
                tool_id,
                run_id,
                on_deleted=lambda: self._on_tool_run_deleted(tool_id),
            ),
            f"Quick tools  ›  {self._tool_label(tool_id)}  ›  {run_id}",
        )

    def _go_to_tool(self, tool_id: str) -> None:
        """Open a tool's run page from the Quick tools landing."""
        if not self._may_leave():
            return
        self._open_tool(tool_id)

    def _on_tool_run_launched(self, tool_id: str, run_id: str) -> None:
        """A run finished — list it in the rail, then open its result."""
        self._refresh_tools_rail()
        self._open_tool_run(tool_id, run_id)

    def _on_tool_run_deleted(self, tool_id: str) -> None:
        """A run is gone — drop it from the rail and fall back to its tool's page (the rail
        keeps that tool in view, so bouncing out to the landing would lose your place).
        """
        self._refresh_tools_rail()
        self._open_tool(tool_id)

    # -- setup-readiness guidance (§1) --------------------------------------- #
    def _refresh_readiness(self) -> None:
        """Recompute the readiness signals off the UI thread (a fresh subprocess run of the
        checks, so a mid-session install is reflected without restart), then drop dismissals
        for steps now done."""
        self._readiness_signals = gui_pages._CallSignals()
        self._readiness_signals.done.connect(self._on_readiness)
        QThreadPool.globalInstance().start(
            gui_pages._CallTask(gui_pages._fresh_signals, self._readiness_signals)
        )

    def _on_readiness(self, signals: ReadinessSignals) -> None:
        self._readiness = signals
        done = {
            step.kind for step in readiness_guide.readiness_steps(signals) if step.done
        }
        if self._dismissed & done:
            # a done step needs no dismissal — drop it so a later regression re-nags
            self._dismissed -= done
            _save_dismissed(self._dismissed)
        # There is no next-step marker: this used to mark a sidebar node, and banner modes are
        # full-width with no rail to mark. `_ReadinessPanel._row` draws ✓ or ○ per step and
        # nothing singles the next one out, so the nudge is the page banner above and only that.
        # (This comment claimed the ● had moved to the checklist; it had not, and `docs/gui.md`
        # documented the marker for as long.)

    def _on_readiness_changed(self) -> None:
        """A Configuration action flipped a signal (token saved, model installed) —
        recompute so the marker (and any banner) reflect it."""
        self._refresh_readiness()

    def _dismiss_step(self, kind: str) -> None:
        self._dismissed.add(kind)
        _save_dismissed(self._dismissed)

    def _recordings_page(self) -> QWidget:
        return gui_pages.ExperimentRecordingsPage(
            self._experiment,
            self._open_recording,
            self._refresh_active_experiment,
            on_import=self._submit_import,
            pending_source=(
                self._import_queue.source_for if self._import_queue else None
            ),
            cancel_imports=self._import_queue.cancel if self._import_queue else None,
        )

    # -- import queue (§3, non-blocking) ------------------------------------- #
    def _submit_import(self, specs, part_kwargs) -> None:
        if self._import_queue is not None:
            self._import_queue.submit(specs, part_kwargs)  # shells created here
            self._refresh_active_experiment()  # queued + empty recordings appear at once
            self._refresh_shown()

    def _refresh_shown(self) -> None:
        """Re-read the current page if it reflects import state (e.g. the recordings
        list shows queued rows + their source)."""
        widget = self._content.currentWidget()
        if widget is not None and hasattr(widget, "refresh"):
            widget.refresh()

    def _on_import_started(self, recording_id: str) -> None:
        self._import_names[recording_id] = self._recording_name(recording_id)
        self._refresh_shown()  # queued → importing in the table

    def _on_import_progress(self, event) -> None:
        """Draw the queue's folded ``ImportProgress`` — through the one tested mapping."""
        fraction, status = import_progress_fraction(event)
        self._import_bar.setValue(round(fraction * 1000))
        self._import_bar.show()
        self.statusBar().showMessage(status)

    def _on_import_item_done(self, recording_id: str, ok: bool, error: str) -> None:
        name = self._import_names.pop(recording_id, None) or self._recording_name(
            recording_id
        )
        self._import_results.append((name, ok, error))
        if ok and self._preview_queue is not None:
            self._preview_queue.queue([recording_id])  # its proxy, in the background
        self._refresh_active_experiment()  # the recording appears/updates in the sidebar
        self._refresh_shown()
        # No clear here: the queue follows this with a progress tick that drops the finished
        # recording from the line, and clearing first would blank it between recordings.

    def _on_import_idle(self) -> None:
        # The progress line has no timeout. Left alone it goes on saying "Importing …" about a
        # batch that has finished, which is what the completion notice used to overwrite.
        self.statusBar().clearMessage()
        self._import_bar.hide()
        self._refresh_shown()
        results, self._import_results = self._import_results, []
        message = _import_notice_text(results)
        if message:
            self._export_notice.say(message)

    def _recording_name(self, recording_id: str) -> str:
        """The name the user gave a recording, or its id when the record cannot be read."""
        if self._experiment is None:
            return recording_id
        with contextlib.suppress(FileNotFoundError, KeyError, ValueError):
            return self._experiment.get_recording(recording_id).name
        return recording_id

    def _show_experiments_home(self) -> None:
        self._show(gui_pages.HomePage(self._open_experiment), "All experiments")

    def _ensure_active(self, experiment: Experiment) -> None:
        """Make ``experiment`` the active one — the target of Distill, the import
        queue, and run actions — if it isn't already. Clicking inside a not-yet-active
        experiment's folder switches to it without disturbing the tree's expansion."""
        if self._experiment is not None and self._experiment.dir == experiment.dir:
            return
        self._activate_experiment(experiment)
        self._refresh_shown()

    def _activate_experiment(self, experiment: Experiment) -> None:
        # Switching experiments drops the per-experiment read caches (the "hard reload" the
        # cache docstrings describe) so a long recording across many experiments doesn't
        # accumulate their entries. Keys are absolute paths, so this bounds growth —
        # it never affects correctness.
        read_cache.clear_read_cache()
        recording_store.clear_recording_cache()
        run_spec.clear_legacy_cache()
        self._experiment = experiment
        experiment.reconcile_imports()  # crashed/closed imports → failed (no resume, §3)
        experiment.reconcile_runs()  # orphaned runs → real on-disk status (§15)
        experiment.reconcile_job_runs()  # same backstop for every single-job run kind
        # A fresh queue has nothing in flight. Whatever the old one last drew is not this
        # queue's to finish, and if the old one is gone its `idle` will never come to clear it.
        self._import_bar.hide()
        self.statusBar().clearMessage()
        self._import_queue = ImportQueue(experiment)
        self._import_queue.item_started.connect(self._on_import_started)
        self._import_queue.progress.connect(self._on_import_progress)
        self._import_queue.item_done.connect(self._on_import_item_done)
        self._import_queue.idle.connect(self._on_import_idle)
        # Previews: one queue per open experiment, started on everything missing — a new
        # machine, an archive import, a cleared cache. The missing check runs in the worker.
        self._preview_label.hide()
        self._preview_bar.hide()
        self._preview_queue = preview_jobs.install(experiment)
        self._preview_queue.progress.connect(self._on_preview_progress)
        self._preview_queue.idle.connect(self._on_preview_idle)
        self._preview_queue.failed.connect(self._on_preview_failed)
        self._preview_failures = []
        self._preview_queue.queue()

    def _on_preview_progress(self, permille: int, status: str) -> None:
        if (
            self._preview_bar.isHidden()
        ):  # a new run of builds: the old failures are history
            self._preview_failures = []
            self._preview_label.setToolTip("")
            self._preview_label.setPalette(self._preview_label_palette)
        self._preview_bar.setValue(permille)
        self._preview_bar.show()
        self._preview_label.setText(status)
        self._preview_label.show()

    def _on_preview_failed(self, recording_id: str, why: str) -> None:
        self._preview_failures.append(f"{self._recording_name(recording_id)}: {why}")

    def _on_preview_idle(self) -> None:
        """The queue drained. Failures stay on the line — with each one in its tooltip —
        until the next build starts: a build that failed quietly is a montage that never
        plays, with nothing to say why."""
        self._preview_bar.hide()
        if not self._preview_failures:
            self._preview_label.hide()
            return
        count = len(self._preview_failures)
        self._preview_label.setText(
            f"Previews: {count} build{'s' if count != 1 else ''} failed (hover for why)"
        )
        self._preview_label.setToolTip("\n".join(self._preview_failures))
        set_label_color(self._preview_label, ERROR)
        self._preview_label.show()

    def _open_experiment(self, experiment: Experiment) -> None:
        """Enter the **project view** — make the experiment active, show the left rail with
        its sections (the only place a rail appears), and show the experiment **overview** (a
        summary of its sections, not the Recordings list, §3.1). Reached from the Experiments list.
        The banner stays on Experiments; the breadcrumb + banner button exit back to the list.
        """
        if not self._may_leave():
            return
        self._ensure_active(experiment)
        self._mode = "home"
        self._banner.set_active("home")
        self._build_project_rail(
            experiment
        )  # the project's sections; the rail becomes visible
        # This door bypasses `_navigate`, which is what sets the page help from the node's
        # kind — so it says its own. Without this the overview inherited whatever kind was
        # last set (`experiments_group` from boot), and its `?` opened the experiments
        # *list* rather than the experiment: a help button that answers a different
        # question, which is worse than one that is missing.
        self._set_page_help("experiment")
        self._open_experiment_overview()

    def _open_experiment_overview(self) -> None:
        """Show the open experiment's overview — the page the rail's **Overview** node and
        the way in from Experiments both land on.

        One opener, two doors, so a rail click and the jump in from Home cannot disagree
        about what is on screen (§25: the opener is the toggle point). It does not ask
        ``_may_leave`` — every route here has already asked, ``_on_tree_click`` on the rail
        side and ``_open_experiment`` on the other — and it does not set the page help,
        because ``_navigate`` sets that from the node's kind before it dispatches.

        ``_set_page_export`` comes **after** ``_show``: ``_show`` clears the subject as its
        fail-closed reset, so a subject set before it is erased.
        """
        self._select_rail_node(("experiment", self._experiment))
        self._show(
            gui_pages.ExperimentOverviewPage(
                self._experiment,
                self._go_to_section,
                on_deleted=self._on_experiment_deleted,
                on_export_launched=self._on_export_launched,
            ),
            f"Experiments  ›  {self._experiment.name}",
        )
        self._set_page_export("experiment")

    def _on_experiment_deleted(self) -> None:
        """The active experiment's directory is gone.

        Drop the page that reads it *before* navigating. ``_switch_mode`` asks the page on
        screen about unsaved edits first, and this one's read-through properties
        (``experiment.name``) now raise ``FileNotFoundError`` — which used to escape and
        leave the window unable to navigate anywhere at all.
        """
        self._experiment = None  # the deleted experiment is gone
        self._discard_content()  # its page reads a config.yaml that no longer exists
        self._switch_mode("home")  # back to the experiments list (rail hidden)

    def _open_recording(self, recording_id: str) -> None:
        self._show(
            gui_pages.RecordingDetailPage(
                self._experiment.get_recording(recording_id),
                on_open_run=self._open_run,
                on_deleted=self._on_recording_deleted,
                experiment=self._experiment,
                on_open_recording_run=self._open_recording_run,
            ),
            f"{self._experiment.name}  ›  {recording_id}",
        )
        self._set_page_export("recording", recording_id)

    def _open_recording_run(
        self, recording_id: str, module_id: str, run_id: str
    ) -> None:
        self._show(
            gui_pages.RecordingRunDetailPage(
                self._experiment, recording_id, module_id, run_id
            ),
            f"{self._experiment.name}  ›  {recording_id}  ›  {run_id}",
        )

    def _on_recording_deleted(self) -> None:
        self._refresh_active_experiment()  # the recording is gone from tree + list
        self._show(self._recordings_page(), self._experiment.name)

    def _open_run(self, run_id: str) -> None:
        # Reattach to a still-running run (survived an app restart) so it shows a live
        # bar; a finished run reattaches to nothing and shows its static record.
        self._show(
            gui_pages.RunDetailPage(
                self._experiment,
                run_id,
                live=self._experiment.attach_active_run(run_id),
                on_deleted=self._on_run_deleted,
                on_open_recording=self._open_recording,
                on_run_launched=self._on_run_launched,
            ),
            f"{self._experiment.name}  ›  {run_id}",
        )
        self._set_page_export("run", run_id)

    def _on_run_launched(self, handle) -> None:
        if handle is None:
            return
        self._refresh_active_experiment()  # the new run appears under Runs
        self._show(
            gui_pages.RunDetailPage(
                self._experiment,
                handle.id,
                live=handle,
                on_deleted=self._on_run_deleted,
                on_open_recording=self._open_recording,
                on_run_launched=self._on_run_launched,
            ),
            f"{self._experiment.name}  ›  {handle.id}",
        )

    def _on_run_deleted(self) -> None:
        self._refresh_active_experiment()  # the run is gone from the Runs list
        self._show(
            gui_pages.ExperimentRunsPage(
                self._experiment, self._open_run, self._on_run_launched
            ),
            self._experiment.name,
        )

    # -- training (datasets + models, §12) ----------------------------------- #
    def _open_datasets(self) -> None:
        self._show(
            gui_pages.DatasetsPage(
                self._experiment, self._open_dataset, self._on_dataset_launched
            ),
            self._experiment.name,
        )

    def _open_keypoints_manager(self) -> None:
        """The Keypoints manager mode (§26): the machine-wide skeleton library."""
        from detail.gui.keypoints_manager import KeypointsManagerPage

        self._show(KeypointsManagerPage(), "Keypoints manager")

    def _open_models(self) -> None:
        """The experiment's trained models and the distill runs that make them — the page
        the rail's Models node opens, and where a deleted model or distill run returns to.
        """
        self._show(
            gui_pages.ModelsPage(
                self._experiment,
                self._open_model,
                self._on_model_launched,
                on_open_distill_run=self._open_distill_run,
                on_distill_launched=self._on_distill_launched,
            ),
            self._experiment.name,
        )

    def _open_dataset(self, dataset_id: str) -> None:
        self._show(
            gui_pages.DatasetDetailPage(
                self._experiment,
                dataset_id,
                live=self._experiment.attach_active_dataset_run(dataset_id),
                on_deleted=self._on_dataset_deleted,
            ),
            f"{self._experiment.name}  ›  {dataset_id}",
        )

    def _open_model(self, model_id: str) -> None:
        self._show(
            gui_pages.ModelDetailPage(
                self._experiment,
                model_id,
                live=self._experiment.attach_active_model_run(model_id),
                on_deleted=self._on_model_deleted,
            ),
            f"{self._experiment.name}  ›  {model_id}",
        )

    def _on_dataset_launched(self, handle) -> None:
        if handle is None:
            return
        self._refresh_training()  # the new dataset appears under Datasets
        self._show(
            gui_pages.DatasetDetailPage(
                self._experiment,
                handle.id,
                live=handle,
                on_deleted=self._on_dataset_deleted,
            ),
            f"{self._experiment.name}  ›  {handle.id}",
        )

    def _on_model_launched(self, handle) -> None:
        if handle is None:
            return
        self._refresh_training()  # the new model appears under Models
        self._show(
            gui_pages.ModelDetailPage(
                self._experiment,
                handle.id,
                live=handle,
                on_deleted=self._on_model_deleted,
            ),
            f"{self._experiment.name}  ›  {handle.id}",
        )

    # -- export runs (#354) --------------------------------------------------- #
    def _on_export_launched(self, handle) -> None:
        """Say it started and stay put — unless it exports the whole experiment.

        An export of a run, a recording or a figure does **not** do what `_on_run_launched`
        does. Navigating to the thing you just started is right for a run — starting it is why
        you opened the app — and wrong for an export, which is an errand off a screen you are in
        the middle of using. Taking that screen away to show a bar costs more than the bar is
        worth, and the run page is one click away in the notice either way.

        **An export of the experiment itself is not an errand**, so it goes to its bar. Packing
        a whole experiment is minutes, nothing on the screen it was started from is in use, and
        the user asked for exactly this after a bundle export they had to go looking for (#916).
        Unsaved edits on the current page are asked about first, as for any navigation; keeping
        them keeps the notice.

        The description comes from the record rather than the form, so the notice names what was
        actually written down — the same `describe_run` the Active-processes row uses, which is
        the one-formatter rule: a fact a person reads has one spelling.
        """
        if handle is None or self._experiment is None:
            return
        what, where = "", ""
        record = None
        try:
            record = self._experiment.export_record(handle.id)
            what, where = api_exporters.describe_run(record)
        except Exception:  # noqa: BLE001
            # A notice must never be why a launch looks failed. The run is submitted and going
            # by the time this is called, so there is nothing left here that can fail.
            what = "your export"
        whole_experiment = (record or {}).get("subject_kind") == "experiment"
        if whole_experiment and self._may_leave():
            self._open_export_run(handle.id, live=handle)
            return
        self._export_notice.announce(what, where)

    def _open_export_run(self, run_id: str, live=None) -> None:
        """An export run's own page, reached from Active processes or straight from a submit.

        ``live`` is the launch handle when we just started it — the bar is then up immediately
        instead of after the reattach below has rescanned. Arriving any other way reattaches,
        which is what makes a run started in a previous session still show a live bar.

        Deliberately does **not** call `_set_page_export`: an export run has nothing to export,
        and copying that line from `_open_run` would offer a control that can produce nothing.
        `_show` clears it on every page swap, so the omission is the whole implementation.
        """
        self._show(
            gui_pages.ExportRunDetailPage(
                self._experiment,
                run_id,
                live=live or self._experiment.attach_active_export_run(run_id),
            ),
            f"{self._experiment.name}  ›  {run_id}",
        )

    # -- distill runs (§12) -------------------------------------------------- #
    def _open_distill_run(self, run_id: str) -> None:
        self._show(
            gui_pages.DistillRunDetail(
                self._experiment,
                run_id,
                live=self._experiment.attach_active_distill_run(run_id),
                on_deleted=self._on_distill_deleted,
                on_open_model=self._open_model,
            ),
            f"{self._experiment.name}  ›  {run_id}",
        )
        self._set_page_export("distill", run_id)

    def _on_distill_launched(self, handle) -> None:
        if handle is None:
            return
        # A distill can be launched for any experiment from the machine-global Distill
        # surface — switch to the run's experiment so its Distill subtree follows it.
        self._ensure_active(handle.experiment)
        self._refresh_training()  # the new run appears on the Models page
        self._show(
            gui_pages.DistillRunDetail(
                handle.experiment,
                handle.id,
                live=handle,
                on_deleted=self._on_distill_deleted,
                on_open_model=self._open_model,
            ),
            f"{handle.experiment.name}  ›  {handle.id}",
        )
        self._set_page_export("distill", handle.id)

    def _on_distill_deleted(self) -> None:
        self._refresh_training()  # the run is gone from the Models page
        self._open_models()

    # -- analysis runs (§16) ------------------------------------------------- #
    def _open_analysis(self) -> None:
        self._show(
            gui_pages.ExperimentAnalysisPage(
                self._experiment, self._open_analysis_run, self._on_analysis_launched
            ),
            self._experiment.name,
        )

    def _open_analysis_run(self, run_id: str) -> None:
        self._show(
            gui_pages.AnalysisRunDetailPage(
                self._experiment,
                run_id,
                # Reattach a still-running analysis (§16), the way _open_run does for a
                # processing run: navigating away destroys the page and its progress
                # worker, so without this the bar never comes back and never completes.
                live=self._experiment.attach_active_analysis_run(run_id),
                on_deleted=self._on_analysis_deleted,
                on_new=self._new_analysis,
            ),
            f"{self._experiment.name}  ›  {run_id}",
        )
        self._set_page_export("analysis_run", run_id)

    def _new_analysis(self) -> None:
        """Start an unseeded analysis run — the same dialog the Analysis index opens.

        Distinct from :meth:`_refine_analysis`, which pre-loads *this* run's config. Finishing
        one analysis is where you decide what to do next, and that is usually a different
        module rather than a variation on this one.
        """
        from detail.gui.launch import LaunchAnalysisDialog

        dialog = LaunchAnalysisDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._on_analysis_launched(dialog.run_handle)

    def _on_analysis_launched(self, handle) -> None:
        if handle is None:
            return
        self._refresh_active_experiment()  # the new run appears under Analysis
        self._show(
            gui_pages.AnalysisRunDetailPage(
                self._experiment,
                handle.id,
                live=handle,
                on_deleted=self._on_analysis_deleted,
                on_new=self._new_analysis,
            ),
            f"{self._experiment.name}  ›  {handle.id}",
        )

    def _on_analysis_deleted(self) -> None:
        self._refresh_active_experiment()  # the run is gone from Analysis
        self._open_analysis()

    def _on_dataset_deleted(self) -> None:
        self._refresh_training()  # the dataset is gone from Datasets
        self._open_datasets()

    def _on_model_deleted(self) -> None:
        self._refresh_training()  # the model is gone from Models
        self._open_models()

    # -- content swap -------------------------------------------------------- #
    def _discard_content(self) -> None:
        """Drop the page on screen without putting another one up.

        Its own swap path (``_show``) uses this, and so does a delete: a page whose data
        has just been removed must not survive to be asked about unsaved edits by the
        next navigation (``_may_leave`` asks ``_content.widget(0)``).
        """
        self._bind_page_notes(None)
        while self._content.count():
            old = self._content.widget(0)
            self._content.removeWidget(old)
            old.deleteLater()

    def _show(self, widget: QWidget, context: str) -> None:
        self._discard_content()
        self._content.addWidget(widget)
        self._content.setCurrentWidget(widget)
        # Fail closed: every page swap clears the Export control, and the opener that knows the
        # subject sets it back. A page that shows nothing exportable therefore shows no control,
        # rather than inheriting the last screen's subject and offering to export it.
        self._export.set_subject(None)
        # The notice belongs to the screen it was raised on: you have navigated, which is what
        # it was telling you to do, so it has said its piece.
        self._export_notice.hide()
        # Every page's text becomes selectable here, which is why a page's text can be copied
        # and a dialog's cannot: a dialog never passes through the shell. `selectable.py` says
        # what happened to the application-wide filter that was supposed to close that gap.
        selectable.make_text_selectable(widget)
        self._context_base = context
        changed = getattr(widget, "context_view_changed", None)
        if changed is not None:
            changed.connect(self._render_context)
        self._render_context()
        self._bind_page_notes(widget.findChild(gui_notes.NotesBox))

    # -- the path line ------------------------------------------------------- #
    def _context_page(self):
        """The page on screen when it can open views of its own (`context_view`), else None."""
        page = self._content.currentWidget()
        if page is None or not alive(page) or not hasattr(page, "context_view"):
            return None
        return page

    def _render_context(self) -> None:
        """The path line: ``_show``'s text, or with a page view open, ``… › <page> › <view>``
        where ``<page>`` links back to the page's landing (an analysis run's views, §16).
        """
        page = self._context_page()
        view = page.context_view() if page is not None else None
        if not view:
            self._context.setTextFormat(Qt.TextFormat.PlainText)
            self._context.setText(self._context_base)
            return
        head, sep, last = self._context_base.rpartition("  \u203a  ")
        arrow = "&nbsp;&nbsp;\u203a&nbsp;&nbsp;"
        self._context.setTextFormat(Qt.TextFormat.RichText)
        self._context.setText(
            (html.escape(head) + arrow if sep else "")
            + f'<a href="landing">{html.escape(last)}</a>'
            + arrow
            + html.escape(view)
        )

    def _on_context_link(self, _href: str) -> None:
        page = self._context_page()
        if page is not None:
            page.close_view()

    def _on_context_link_hovered(self, href: str) -> None:
        page = self._context_page()
        tip = getattr(page, "BACK_TIP", "") if page is not None else ""
        if href and tip:
            QToolTip.showText(QCursor.pos(), tip, self._context)
        else:
            QToolTip.hideText()

    # -- the page's notes ------------------------------------------------------ #
    def _bind_page_notes(self, box) -> None:
        """Point the Notes button at ``box`` (the page's one Notes box), or hide it for none."""
        old = self._notes_box
        if old is not None and alive(old):
            for signal in (old.saved, old.open_changed):
                try:
                    signal.disconnect(self._sync_notes_button)
                except TypeError:
                    pass
        self._notes_box = box
        if box is not None:
            box.saved.connect(self._sync_notes_button)
            box.open_changed.connect(self._sync_notes_button)
        self._sync_notes_button()

    def _on_notes_created(self, box) -> None:
        """A box built after its page was shown (a body that loads off the UI thread): bind it
        once it is in the page, which happens after its constructor returns."""
        QTimer.singleShot(0, lambda: self._adopt_notes(box))

    def _adopt_notes(self, box) -> None:
        page = self._content.currentWidget()
        if (
            alive(box)
            and page is not None
            and page.isAncestorOf(box)
            and box is not self._notes_box
        ):
            self._bind_page_notes(box)

    def _sync_notes_button(self, *_args) -> None:
        box = self._notes_box
        if box is None or not alive(box):
            self._notes_button.hide()
            return
        self._notes_button.setText(gui_notes.button_text(box.has_note()))
        self._notes_button.setChecked(box.is_open())
        self._notes_button.show()

    def _toggle_page_notes(self) -> None:
        box = self._notes_box
        if box is not None and alive(box):
            box.set_open(not box.is_open())
        self._sync_notes_button()

    def _set_page_export(self, kind: str, subject_id: str = "") -> None:
        """Point the Export control at what this page is showing.

        Called by the one opener per destination — the same place `_select_rail_node` is
        called from, and for the same reason: every route into a page goes through it, so a
        table double-click and a rail click cannot disagree about what is on screen.
        """
        if self._experiment is None:
            return
        self._export.set_subject(
            ExportSubject(kind, self._experiment.dir, subject_id),
            self._content.widget(0) if self._content.count() else None,
        )

    # -- window + rail geometry, across sessions ----------------------------- #
    # -- the rail: shown per mode, folded to its strip by the user ----------- #
    def _build_rail_box(self) -> QWidget:
        """The rail's column: the tree, in a pane that folds to a strip reading "Navigation".

        Folded the way every side panel is (`gui/grip_splitter.py`): dragged narrower than
        :data:`_RAIL_MIN_PX`, or its grip double-clicked, and it shuts; dragged back out or
        double-clicked again, it opens at the width it had. 260px of a 1440px laptop window went
        to a list that is nearly empty inside a run (laptop layout, 2026-11-09). Ctrl+\\ (⌘\\ on
        a Mac) does the same from anywhere — the keyboard's way to it, since a grip takes no
        focus.
        """
        from PyQt6.QtGui import QKeySequence, QShortcut

        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._tree, 1)
        box.setMinimumWidth(_RAIL_MIN_PX)
        box.setMaximumWidth(_RAIL_PX)
        self._rail_box = box
        shortcut = QShortcut(QKeySequence("Ctrl+\\"), self)
        shortcut.activated.connect(
            lambda: self._rail_box.isVisibleTo(self) and self._split.toggle(0)
        )
        return box

    def _show_rail(self, shown: bool) -> None:
        """Whether this mode carries a rail at all (the tree's own flag says so)."""
        self._tree.setVisible(shown)
        self._rail_box.setVisible(shown)

    def nav_hidden(self) -> bool:
        """Whether the rail is shut to its strip."""
        return self._split.is_folded(0)

    def _save_layout(self) -> None:
        """Remember the window's size (the rail's split remembers itself, `grip_splitter`)."""
        QSettings().setValue(_GEOMETRY_KEY, self.saveGeometry())

    def _restore_layout(self) -> None:
        """Put the window back where it was, or fall back to a screen-bounded default.

        ``restoreGeometry`` is used rather than a remembered width and height because it
        *"checks if the restored geometry is outside the available screen geometry, and modifies
        it as appropriate"* — the clamping `_opening_size` has to do by hand, done properly, and
        the case that matters when a laptop comes back without the monitor it was last docked to.

        Its return value is checked. OBS is the model: a stale or truncated blob restores nothing
        and silently leaves the window at whatever size it happened to have, so a false here means
        fall back to the default rather than trust it.
        """
        settings = QSettings()
        geometry = settings.value(_GEOMETRY_KEY)
        if not (isinstance(geometry, QByteArray) and self.restoreGeometry(geometry)):
            self.resize(_opening_size())
        try:
            legacy_width = int(settings.value(_LEGACY_NAV_WIDTH_KEY, 0))
        except (TypeError, ValueError):
            legacy_width = 0
        if not self._split.remember(
            legacy_state=_LEGACY_SPLIT_KEY,
            legacy_shut=(
                {0}
                if settings.value(_LEGACY_NAV_HIDDEN_KEY, False, type=bool)
                else set()
            ),
            legacy_open={0: legacy_width} if legacy_width else None,
            legacy_keys=(_LEGACY_NAV_HIDDEN_KEY, _LEGACY_NAV_WIDTH_KEY),
        ):
            self._split.setSizes([_RAIL_PX, 940])

    # -- quit (don't silently orphan in-flight workers) ---------------------- #
    def closeEvent(self, event) -> None:
        """Prompt before quitting with runs in flight (they outlive the GUI as
        detached worker processes — §15). Cancel & quit kills them first.

        Unsaved edits get the same courtesy first: quitting is another way to leave the
        page you are on, and losing a filled-in table to it would be the same bug.
        """
        if not self._may_leave():
            event.ignore()
            return
        # Saved before the run prompt, not after: "Cancel & quit" and "quit anyway" both end
        # here, and a layout that only persisted on the quietest of the three exits is one a
        # user would find had reverted at random.
        self._save_layout()
        active = self._active_runs()
        if not active:
            self._stop_previews()
            super().closeEvent(event)
            return
        choice = self._confirm_quit(active)
        if choice == "dont":
            event.ignore()
            return
        if choice == "cancel_quit":
            self._cancel_all_active(active)
        self._stop_previews()
        event.accept()

    def _stop_previews(self) -> None:
        """End the preview build in hand on quit; a half-built part is discarded and is
        built again next time the experiment is opened."""
        if self._preview_queue is not None:
            self._preview_queue.shutdown()

    def _confirm_quit(self, active: dict[str, list[str]]) -> str:
        """Ask how to handle in-flight runs on quit. Returns ``"cancel_quit"`` |
        ``"leave"`` | ``"dont"`` (the default; also when the dialog is dismissed)."""
        box = QMessageBox(self)
        # "Runs in progress" and "Leave running" were both true of every family until one of
        # them stopped being a run that keeps running. The body text is not the only thing in a
        # dialog a user reads — a button is the thing they act on, and offering to "leave
        # running" something that is about to stop is the same false promise one line down.
        box.setWindowTitle("Still in progress")
        box.setText(_quit_message(active))
        cancel_quit = box.addButton(
            "Cancel runs && quit", QMessageBox.ButtonRole.AcceptRole
        )
        leave = box.addButton(
            (
                "Leave running && quit"
                if _anything_survives_quit(active)
                else "Quit anyway"
            ),
            QMessageBox.ButtonRole.DestructiveRole,
        )
        dont_quit = box.addButton("Don't quit", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(dont_quit)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_quit:
            return "cancel_quit"
        if clicked is leave:
            return "leave"
        return "dont"

    def _active_runs(self) -> dict[str, list[str]]:
        """Ids of the open experiment's runs whose status is still active, keyed by run kind.

        Off `api.active.RUN_KINDS` rather than its own list of families, which is what it was:
        four of the six, missing **clusterer** and **distill**. So closing the window during a
        distill run — hours of GPU training, the longest thing this app does — accepted the
        close without a word, and `_cancel_all_active` could not have reached it either.
        """
        if self._experiment is None:
            return {}
        return api_active.active_ids(self._experiment)

    def _cancel_all_active(self, active: dict[str, list[str]]) -> None:
        """Cancel every active run (blocking — we're quitting, so the workers must
        die before the process exits). A run finishing on its own mid-quit is fine
        (cancel is a no-op once terminal).

        The import **queue** is signalled first, before any record is touched. `cancel_import`
        records a decision and cannot reach a worker thread; the queue's event is the thing
        that actually stops a copy in flight. Marking first would be a status written before
        the work that justifies it — the record would say ``cancelled`` while the copy ran on.
        """
        if self._import_queue is not None and any(
            active.get(spec.kind)
            for spec in api_active.RUN_KINDS
            if not spec.survives_quit
        ):
            self._import_queue.cancel()
        for spec in api_active.RUN_KINDS:
            cancel = getattr(self._experiment, spec.cancel)
            for run_id in active.get(spec.kind, []):
                with contextlib.suppress(FileNotFoundError, ValueError):
                    cancel(run_id)
