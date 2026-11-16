"""Content-area pages for the GUI shell (slice 1).

Each page is a thin view over ``Experiment`` / ``Recording`` (Tenet 2) — reads are
wired to the real API; heavier actions (launch a run, draw an arena, install a
model) are present as disabled affordances and land in later slices. As pages
grow dialogs/widgets they move into the ``pages/`` package per docs/gui_design.md.
"""

import contextlib
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import (
    QEvent,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPalette,
    QPixmap,
    QShortcut,
    QStandardItem,
    QStandardItemModel,
)
from PyQt6.QtWidgets import (
    QBoxLayout,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QRadioButton,
    QStackedWidget,
    QTabBar,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detail import docs_access
from detail import surfaces
from detail.gui import sizing
from detail.gui.camera_view import CameraViewPicker, inherit_label
from detail.gui.dialog import Dialog
from detail.gui import experiment_locations
from detail.gui.full_frame import EnlargeableThumb
from detail.gui.collapse import CollapsibleBand
from detail.gui.recording_filter import RecordingFilter
from detail.gui.lifetime import alive, cleared, discard
from detail.gui.sizing import (
    emphasis_font,
    clamp_width,
    can_grow,
    expands_vertically,
    fit_list_height,
    has_slack as _has_slack,
    lines,
    set_available,
)
from detail.gui.tables import fill_width, fit_table_height as _fit_table_height
from detail.gui.tables import nav_cell
from detail.gui.tables import (
    enable_sorting,
    give_slack_to,
    make_columns_resizable,
    spread_slack,
)
from detail.gui.tables import refit_slack
from detail.gui.tables import refill_table_widget, row_id, stamp_row_id
from detail.gui.tables import SortCell
from detail.gui.tables import MIN_VISIBLE_ROWS
from detail.gui import requirements
from detail.api import models as api_models
from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.gui.spin import PLAIN, FloatSpin, IntSpin
from detail.api import skeletons as api_skeletons
from detail.api.spatial_categories import CATEGORIES_BY_KEY, SPATIAL_CATEGORIES
from detail.gui.arena import SPATIAL_KINDS
from detail.storage import locations as locations_store
from detail.storage import metadata as metadata_store
from detail.storage import read_cache
from detail.storage import spatial as spatial_store
from detail.api import checks as api_checks
from detail.api import exporters
from detail.api import tools as api_tools
from detail.core import buildinfo
from detail.core import camera_view as camera_view_mod
from detail.core import run_cost
from detail.core.progress import format_duration
from detail.api.experiment import Experiment
from detail.api.experiment import RESUMABLE_TRAINING_STATUSES
from detail.api.readiness import signals_from
from detail.core.dataclasses import (
    AWAITING_REFERENCES,
    ACTIVE_STATUSES,
    AUTHORED_STATUSES,
    OUTPUT_STATUSES,
    CheckStatus,
)
from detail.gui import cuda_doctor
from detail.gui import open_external
from detail.gui import modules as _gui_modules  # noqa: F401 — registers module UIs
from detail.api import readiness as readiness_api
from detail.gui import preview_jobs
from detail.gui import readiness_guide
from detail.gui.modules.registry import ANALYSIS_GUI_MODULES, GUI_MODULES
from detail.gui.result_views import ResultView, result_views
from detail.gui.notes import NotesBox, place_notes
from detail.gui.run_activity import LiveRunPanel
from detail.gui.scroll import PageScroll
from detail.gui.tools.registry import TOOL_GUIS, import_tool_uis
from detail.gui import sections as gui_sections
from detail.gui.busy import run_busy, run_progress
from detail.gui.snapshots import snapshot_columns, snapshot_rows
from detail.gui.status_color import ERROR, MUTED, READY, WARN, set_label_color
from detail.gui.workers import (
    BackgroundCall,
    FitProgressWorker,
    InstallModelWorker,
    RunProgressWorker,
    StageProgressWorker,
    StageStatusLine,
)
from detail.storage import analysis_runs as analysis_runs_store
from detail.gui import run_units
from detail.storage import clocks, credentials, read_cache
from detail.gui.sizing import mono_font
from detail.gui.space_prompt import launch_anyway


def _mono() -> QFont:
    return mono_font()


MONO = _mono()

#: Left indent on the scene-geometry kind blocks, marking them as sub-sections.
_SCENE_INDENT = 14


def _heading(text: str) -> QLabel:
    lbl = QLabel(text)
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


def _divider() -> QFrame:
    """A plain horizontal rule, for separating a constructive action from a destructive one."""
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setFrameShadow(QFrame.Shadow.Plain)
    return rule


def _band_caption(text: str) -> QWidget:
    """A top-level group caption (bold, CAPS title) over a full-width rule — the
    divider grouping the recording page into 'DATA PIPELINE' vs 'ANALYSIS' bands.
    One level above the per-section ``_heading``s (which the rule sets it apart
    from)."""
    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setContentsMargins(0, 10, 0, 0)
    box.setSpacing(2)
    box.addWidget(_heading(text))
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setFrameShadow(QFrame.Shadow.Plain)
    box.addWidget(rule)
    return holder


# The status palette and `set_label_color` live in `gui/status_color.py`, a leaf module, and are
# imported above. They used to be defined right here — *after* the `detail.gui.modules` import,
# which reaches `confidence.py`, which read them back out of this module. See that file.


def _part_location(recording_dir, part: dict) -> str:
    """A video part's **absolute** on-disk location for display.

    A reference-in-place part resolves its ``{root, rel}`` through the video-root registry to
    the real external path (with a ``(referenced)`` tag); a copied part shows the managed file's
    absolute path. If the referenced root isn't registered on this machine, show the
    ``root/rel`` reference and say so — the user needs to register/relink it.
    """
    from detail.storage import recording as recording_store
    from detail.storage import video_roots

    source = part.get("source")
    if source is None:
        return str(Path(recording_dir) / part["file"])
    try:
        resolved = recording_store.resolve_part_video_path(recording_dir, part)
    except video_roots.MissingVideoRootError:
        return (
            f"{source['root']}/{source['rel']}  "
            f"(video root '{source['root']}' not registered on this machine)"
        )
    return f"{resolved}  (referenced)"


def add_body_widget(body, widget) -> None:
    """Add ``widget`` to a page body, giving it room if it asks for room.

    The class this fixes: **a page adds every child with stretch 0, so a child that declares it
    wants to expand can never get any.** Pages top-justify and end with `addStretch()`, which is
    right for the common case of small stacked rows — but it means a fill element (a splitter, a
    canvas, a 3-D scatter) collapses to its size hint while the stretch below it takes the whole
    window. Nothing fails: the page renders, the widget is there, it is just letterbox-thin with
    empty space beneath it.

    The GUI style rule already names the *opposite* failure ("a page with small content, no fill
    child, and no trailing stretch spreads down the page"). This is its mirror, and it had no
    rule and no guard, so it recurred.

    So the widget's own `sizePolicy` decides, rather than each page hand-tuning: a vertical
    policy carrying the Expanding flag gets stretch 1, everything else keeps stretch 0 and the
    top-justified behaviour every existing page relies on.
    """
    body.addWidget(widget, 1 if expands_vertically(widget) else 0)


def give_slack_to_body(layout, index: int, fills: bool) -> None:
    """The page's trailing spacer at ``index`` takes the slack, unless the body has a fill child.

    The second half of :func:`add_body_widget`. Giving the fill child stretch 1 is not enough
    while the page's own trailing ``addStretch(1)`` is still there: two items at stretch 1 split
    the window between them, and the results sat in the top half of the page with empty space
    below (a clustering run's INTERPRET, measured at 500 of 1,150 px). **Taken out, not set to
    stretch 0**: a spacer is *expanding*, so at factor 0 it still competes — measured on the run
    page with the spacer left in at 0, the results pane got 42% of the window.
    """
    item = layout.itemAt(index)
    spacer = item is not None and item.spacerItem() is not None
    if fills and spacer:
        layout.takeAt(index)
    elif not fills and not spacer:
        layout.insertStretch(index, 1)


def _make_selectable(widget: QWidget) -> None:
    """Enable mouse text-selection on every QLabel under ``widget``. ``MainWindow._show``
    covers a page's labels at show time, but rows built later (an off-thread readiness /
    CUDA probe returning) aren't there yet — call this after such an async render so
    users can still copy that text."""
    for label in widget.findChildren(QLabel):
        label.setTextInteractionFlags(
            label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
        )


def _instructions(text: str) -> QLabel:
    """A word-wrapped on-screen instruction band (guidance above, interface below —
    docs_design §5). Plain: no stylesheet, no icons."""
    label = QLabel(text)
    label.setWordWrap(True)
    return label


def _source_label(recording) -> str:
    """Where a recording's footage came from, for a one-line table cell.

    The first part's absolute path, plus ``+N more`` when a recording spans several
    files — the per-part detail belongs on the recording's own page, not in a cell.
    """
    sources = [s for s in recording.video_sources if s]
    if not sources:
        return ""
    extra = f"   +{len(sources) - 1} more" if len(sources) > 1 else ""
    return f"{sources[0]}{extra}"


class _Movable:
    """One definition a picker can offer, and whether it can actually go.

    Both the push (Copy) and the pull (Import) list *definitions* now, not categories. A
    category tick is a bet: you tick "Regions of interest" and find out afterwards which
    of them the destination refused. A definition tick is a specific thing, so whatever
    cannot go can be disabled with its reason *before* the click — which is the difference
    between a dialog that reports failures and one that never offers them.
    """

    __slots__ = ("kind", "name", "blocked", "flattens")

    def __init__(self, kind, name: str, blocked: str | None, flattens: bool):
        self.kind = kind
        self.name = name
        self.blocked = blocked  # why it cannot go, or None
        self.flattens = flattens  # its keyframe motion will be dropped

    @property
    def label(self) -> str:
        text = self.name
        if self.flattens:
            text += "  ·  camera motion not copied"
        if self.blocked:
            text += f"  —  {self.blocked}"
        return text


def _movables(source, blocked_for) -> list[_Movable]:
    """Every definition on ``source``, in kind order, each with its verdict.

    ``blocked_for(kind, name) -> str | None`` is the only thing the two directions
    disagree about: Import asks whether *this* recording already holds the name, Copy asks
    whether any chosen target could still take it.
    """
    from detail.gui.arena import SPATIAL_KINDS
    from detail.storage import spatial as spatial_store

    out: list[_Movable] = []
    with read_cache.pinned():
        for kind in SPATIAL_KINDS:
            if kind.id == "calibration":
                if source.distance_calibration is None:
                    continue
                out.append(
                    _Movable(
                        kind, "Distance calibration", blocked_for(kind, None), False
                    )
                )
                continue
            raw = spatial_store.read_defs(source.dir, kind.id)
            _ok, flattened, refused = spatial_store.prepare_for_arrival(raw)
            reasons = dict(refused)
            for entry in raw:
                name = entry["name"]
                out.append(
                    _Movable(
                        kind,
                        name,
                        reasons.get(name) or blocked_for(kind, name),
                        name in flattened,
                    )
                )
    return out


def _fill_movables(listwidget, movables: list[_Movable]) -> None:
    """Render ``movables`` as a checkable list under bold, non-selectable kind headings —
    the same shape as the scene-geometry editor's own Definitions list, so the two read as
    one idea. A blocked row is present, unticked and disabled, with its reason: hiding it
    would leave the user wondering where their region went."""
    # A blocked row carries its reason, which is longer than any sensible dialog is wide.
    # Left to scroll horizontally the bar eats a row of height and the reason is still
    # unreadable; elided, the row stays scannable and the tooltip carries the whole thing.
    listwidget.setTextElideMode(Qt.TextElideMode.ElideRight)
    listwidget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    listwidget.blockSignals(True)
    listwidget.clear()
    last_kind = None
    for i, m in enumerate(movables):
        if m.kind is not last_kind:
            header = QListWidgetItem(m.kind.label)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, None)
            listwidget.addItem(header)
            last_kind = m.kind
        item = QListWidgetItem(m.label)
        item.setData(Qt.ItemDataRole.UserRole, i)
        if m.blocked:
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setToolTip(m.blocked)
        else:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(
                "Camera motion is not copied — the frame-0 position is."
                if m.flattens
                else ""
            )
        listwidget.addItem(item)
    listwidget.blockSignals(False)


def _checked_movables(listwidget, movables: list[_Movable]) -> list[_Movable]:
    return [
        movables[listwidget.item(row).data(Qt.ItemDataRole.UserRole)]
        for row in range(listwidget.count())
        if listwidget.item(row).data(Qt.ItemDataRole.UserRole) is not None
        and listwidget.item(row).checkState() == Qt.CheckState.Checked
    ]


def _names_by_kind(chosen: list[_Movable]) -> dict[str, list[str]]:
    by_kind: dict[str, list[str]] = {}
    for m in chosen:
        by_kind.setdefault(m.kind.id if m.kind.id != "arena" else "fov", []).append(
            m.name
        )
    return by_kind


def _recording_display_name(recording) -> str:
    """How a recording is labelled in a cross-recording picker.

    The **name** leads, because it is the only thing guaranteed to tell two recordings
    apart: names are unique by construction (§11) and ids are the tool's, not the user's.
    This used to lead with the video's file name on the reasoning that it is what a person
    recognises — true until a rig writes every session to the same file name, or several
    recordings reference one clip, at which point the picker offers eight rows reading
    ``rig_camera.mp4`` and the choice cannot be made at all.

    The file name still comes along when it adds something, since it is often how somebody
    knows which session this is.
    """
    name = recording.name
    for source in recording.video_sources:
        if source:
            stem = Path(source).name
            return name if stem == name else f"{name}  ·  {stem}"
    return name


def _recording_rows(experiment):
    """``(ids, rows, an import is live)`` for the recordings table. Qt-free, runs OFF the UI
    thread — it is one ``recording.yaml`` read per recording (mtime-cached).

    Pinned, because a row asks the same file for eight different fields and every
    ``Recording`` property is its own read: 9.0 round trips per recording without it, 2.0
    with (the floor — one listing probe, one read). See ``read_cache.pinned``.

    The **Source** column is left as what disk says; the caller overlays the import queue's
    in-memory answer, which it alone can see.
    """
    with read_cache.pinned():
        recordings = experiment.recordings  # off-thread
        active = any(s.import_status in ("queued", "importing") for s in recordings)
        rows = [
            (
                s.name,
                s.import_status,
                _source_label(s),
                len(s.video_parts),
                len(s.streams),
                len(s.arenas),
                camera_view_cell(s.camera_view),
                len(s.detection_runs),
            )
            for s in recordings
        ]
        return [s.id for s in recordings], rows, active


#: The Import CSV button's hover text — how a sheet's rows find their recordings, built
#: from the one list of headers the import recognises so the two cannot drift.
#: Line breaks are explicit: Qt word-wraps only a rich-text tooltip, so a plain one this
#: long would otherwise render as a single line the width of the screen.
_IMPORT_CSV_TOOLTIP = (
    "Merge a filled-in CSV into this table.\n\n"
    "Each row is matched to a recording by one key column:\n"
    "  •  recording_id\n"
    "  •  name\n"
    "  •  the video file name, in a column headed\n"
    "     " + ", ".join(metadata_store.VIDEO_KEY_HEADERS) + " (any case)\n\n"
    "A file name matches on its base name, any case, extension optional.\n"
    "No such column? You'll be asked which one holds the video file name.\n\n"
    "Blank cells keep their value. If any row matches no recording,\n"
    "nothing is applied."
)


def _file_name_cell(recording) -> tuple[str, str]:
    """The recording's file name for the metadata table's second column (OQ-TBL-3/4):
    base name of the first video part in the cell, full path(s) in the tooltip — the
    same "base name on screen, full path on hover" split as ``_source_label``. A
    recording with no video yet (metadata entered ahead of import) shows blank."""
    sources = [s for s in recording.video_sources if s]
    if not sources:
        return "", ""
    extra = f"  +{len(sources) - 1} more" if len(sources) > 1 else ""
    return f"{Path(sources[0]).name}{extra}", "\n".join(sources)


def camera_view_cell(resolved: camera_view_mod.CameraView) -> str:
    """A camera view as a table cell: ``Side view (default)`` when the recording inherits
    it, so a recording that sets its own is the one without the suffix."""
    if resolved.source == camera_view_mod.SOURCE_EXPERIMENT:
        return f"{resolved.label} (default)"
    return resolved.label


def _camera_view_completion(recording) -> tuple[bool, str]:
    resolved = recording.camera_view
    if not resolved.is_set:
        return False, "No camera view set, here or as the experiment default"
    return True, camera_view_cell(resolved)


def _fov_completion(recording) -> tuple[bool, str]:
    n = len(recording.arenas)
    if n == 0:
        return False, "No field of view set"
    return True, f"{n} field of view{'s' if n != 1 else ''} set"


def _calibration_completion(recording) -> tuple[bool, str]:
    cal = recording.distance_calibration
    if cal is None:
        return False, "Not calibrated"
    if recording.calibration_refusal() is not None:
        return False, "Calibrated on a video of a different shape — recalibrate"
    ppm = cal.get("pixels_per_mm")
    detail = f" — {ppm:.3f} px/mm" if isinstance(ppm, (int, float)) else ""
    return True, f"Calibrated{detail}"


def _named_entries_completion(
    entries: list[dict], singular: str, plural: str, verb: str
) -> tuple[bool, str]:
    """Shared shape for ROI / POI / stream completion cells: a count plus the names
    set, or a plain "none yet" when there are none."""
    n = len(entries)
    if n == 0:
        return False, f"No {plural} {verb}"
    names = ", ".join(e.get("name", "") for e in entries if e.get("name"))
    count = f"{n} {singular if n == 1 else plural} {verb}"
    return True, f"{count}: {names}" if names else count


def _run_completion(runs: list[dict], noun: str) -> tuple[bool, str]:
    """Detection / tracking / embedding completion: checked once at least one run has
    **completed** — an attempt that errored or is still running doesn't count."""
    n_total = len(runs)
    if n_total == 0:
        return False, f"No {noun} runs"
    n_complete = sum(1 for r in runs if r.get("status") == "complete")
    plural = "run" if n_total == 1 else "runs"
    if n_complete == 0:
        return False, f"{n_total} {noun} {plural}, none complete"
    return True, f"{n_complete}/{n_total} {noun} {plural} complete"


#: The header tooltip on the one column every experiment has (§11).
_ACQUISITION_SETUP_TIP = (
    "Which camera, box or rig this recording was made with — the same value for every "
    "recording made the same way. Keypoint setup picks one recording per setup to learn "
    "from. Every experiment has this column; it can't be deleted."
)


#: The completion columns on the metadata table (§ "Recording information"), in the
#: order they're shown — a header and a ``Recording -> (checked, tooltip)`` function,
#: each read off the same ``recording.yaml`` already loaded for the row (cheap under
#: ``read_cache.pinned()``, same as ``_recording_rows`` above).
_COMPLETION_COLUMNS: tuple[tuple[str, object], ...] = (
    ("Camera view", _camera_view_completion),
    ("Field of view", _fov_completion),
    ("Distance calibration", _calibration_completion),
    ("ROI", lambda r: _named_entries_completion(r.rois, "ROI", "ROIs", "set")),
    ("POI", lambda r: _named_entries_completion(r.points, "POI", "POIs", "set")),
    (
        "Streams",
        lambda r: _named_entries_completion(r.streams, "stream", "streams", "added"),
    ),
    ("Detection", lambda r: _run_completion(r.detection_runs, "detection")),
    ("Tracking", lambda r: _run_completion(r.tracking_runs, "tracking")),
    ("Embedding", lambda r: _run_completion(r.embedding_runs, "embedding")),
)


def _analysis_rows(experiment):
    """``(ids, rows)`` for the analysis-runs index. Qt-free, runs OFF the UI thread — one
    ``analysis.yaml`` per run (mtime-cached, and pinned so a run read twice costs once).
    """
    with read_cache.pinned():
        refs = experiment.analysis_runs  # off-thread
        return (
            [r.id for r in refs],
            [(r.id, r.module_id, ", ".join(r.recording_ids), r.status) for r in refs],
        )


def _recording_labels(experiment, recording_ids) -> str:
    """Names for the recording ids a run record stores.

    Records key on the id (permanent), users read the name (§11). An id with no name
    falls back to itself — a recording deleted since the run still has to render.
    """
    names = experiment.recording_names()
    return ", ".join(names.get(rid, rid) for rid in recording_ids) or "—"


def _status_label() -> QLabel:
    """A page-top status line — bold + a size larger, for the green 'ready' / result
    messages users scan first (they belong at the top of the page)."""
    label = QLabel("")
    label.setWordWrap(True)
    label.setFont(emphasis_font(label, steps=2))
    return label


def surfaced_model_rows() -> list:
    """The Model manager's installable rows, less any model this build hides (§25).

    Filtered here, where the catalog is enumerated for a user, not in ``api.models``: the API
    never consults the table, and an install check by id still answers for a hidden model.
    """
    return [r for r in api_models.list_model_variants() if surfaces.is_surfaced(r.id)]


def _link_button(text: str, url: str) -> QPushButton:
    """A push button that opens ``url`` in the user's browser."""
    button = QPushButton(text)
    button.clicked.connect(lambda: open_external.open_url(button, url))
    return button


class _ReadinessBanner(QFrame):
    """A dismissible amber banner nudging the current setup step (§1): a line of text +
    a '×' that calls ``on_dismiss`` and hides the banner. Plain — no stylesheet."""

    def __init__(self, text: str, on_dismiss):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self._on_dismiss = on_dismiss
        box = QHBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        set_label_color(label, WARN)
        box.addWidget(label, 1)
        close = QPushButton("×")
        close.setFixedWidth(24)
        close.setToolTip("Dismiss")
        close.clicked.connect(self._dismiss)
        box.addWidget(close)

    def _dismiss(self) -> None:
        if self._on_dismiss is not None:
            self._on_dismiss()
        self.hide()


_LEVEL_COLORS = {"ready": READY, "warn": WARN, "error": ERROR}


def _check_row(label: str, ok: bool, detail: str) -> QWidget:
    """A CUDA-doctor diagnostics row: a ✓/✗ glyph over ``label: detail``."""
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    glyph = QLabel("✓" if ok else "✗")
    set_label_color(glyph, READY if ok else WARN)
    box.addWidget(glyph)
    box.addWidget(QLabel(f"{label}: {detail}"))
    box.addStretch()
    return row


def _command_row(command: str) -> QWidget:
    """A read-only monospace copy-paste command with a Copy button (the setup-doctor fix
    row, shared by the GPU/CUDA and SAM 3 backend pages)."""
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    field = QLineEdit(command)
    field.setReadOnly(True)
    field.setFont(MONO)
    box.addWidget(field, 1)
    copy = QPushButton("Copy")
    copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(command))
    box.addWidget(copy)
    return row


def _link_row(links: list[tuple[str, str]]) -> QWidget:
    """A left-aligned row of external-link buttons."""
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    for text, url in links:
        box.addWidget(_link_button(text, url))
    box.addStretch()
    return row


HF_JOIN_URL = "https://huggingface.co/join"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"

_MODEL_DISPLAY = {
    "sam3": "SAM 3",
    "dinov3_convnext_base": "DINOv3 (ConvNeXt)",
    "dinov3_vit": "DINOv3 (ViT)",
    # The shared DINOv3 access gate (hf_gate), not a model: one agreement covers every
    # backbone and size, so the request link names the family.
    "dinov3": "DINOv3",
}


def _model_display(model_id: str) -> str:
    """A friendly name for a model id in user-facing labels."""
    return _MODEL_DISPLAY.get(model_id, model_id)


def _cell(value) -> QStandardItem:
    """One table cell: displays its text, sorts on its **value**.

    `QStandardItem` compares display strings, so a column of counts sorts 1, 11, 12, 2 and a
    column of seconds puts `10.0` before `9.0`. `SortCell` has existed for this since the two
    result workspaces needed it — the shared builder every index table goes through simply never
    used it, so turning sorting on made every numeric column in the app visibly wrong at once.

    A number keeps its own ordering; anything else sorts as text, which is what it is. `bool` is
    excluded deliberately: it is an `int` in Python and a column of them reads as Yes/No, where
    alphabetical is the ordering a reader expects.

    A caller that pre-formats a number into a string (`f"{x:.3f}"`) is still sorting text, and
    this cannot see that — it never receives the number. `tests/gui/test_table_sorting.py` names
    the columns that do it.
    """
    if isinstance(value, QStandardItem):
        return (
            value  # a caller that needs both a format and a key builds the cell itself
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return QStandardItem(str(value))
    return SortCell(str(value), value)


def _table(
    headers: list[str],
    rows: list[tuple],
    navigable: bool = False,
    ids: list | None = None,
) -> QTableView:
    """Read-only table sized to its rows, columns filling the width, sortable and
    resizable (``docs/gui_design.md`` §3).

    ``navigable=True`` marks a table whose rows **open a detail/result page** on
    double-click: it shows a pointing-hand cursor over the rows and appends a trailing
    "›" column — the shared cue that a row is clickable (``docs/gui_design.md`` §3).
    Pass it only where a ``doubleClicked`` → open is wired; a **selection-only** table
    (row-select just enables buttons) or a **static** display must NOT set it, or the
    cue would mislead. The extra column is inert — open handlers key off :func:`row_id`.

    ``ids``, parallel to ``rows``, stamps each row's identity onto its first cell
    (:func:`stamp_row_id`) so a click handler can recover it via :func:`row_id`
    regardless of how the user has sorted the view — reading ``index.row()`` into a
    separately-held list is only correct while nothing can reorder the rows.
    """
    n_data = len(headers)
    all_headers = [*headers, ""] if navigable else list(headers)
    table = QTableView()
    model = QStandardItemModel(len(rows), len(all_headers))
    model.setHorizontalHeaderLabels(all_headers)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            item = _cell(value)
            item.setFont(MONO)
            if c == 0 and ids is not None:
                stamp_row_id(item, ids[r])
            model.setItem(r, c, item)
        if navigable:
            chevron = (
                nav_cell()
            )  # one definition of the cue, shared with `gui/tables.py`
            chevron.setFont(MONO)
            model.setItem(r, n_data, chevron)
    table.setModel(model)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().setVisible(False)
    make_columns_resizable(table)
    # Qt's `wordWrap` defaults to **true**, and nothing in this package has ever asked for it:
    # every cell here is an id, a status, a timestamp or a path. What it costs is ragged row
    # heights — and `_fit_table_height` sizes the view from one uniform row, so a single wrapped
    # cell leaves the table short by exactly the rows it grew, drawing an empty strip under the
    # last one. Found by rendering the Active-processes exports table, whose destination column
    # is a real path; invisible to every test, because the rows are all there and all correct.
    table.setWordWrap(False)
    if headers:
        # The last *data* column takes the slack; a navigable "›" column stays narrow.
        give_slack_to(table, n_data - 1)
    if navigable:
        table.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
    enable_sorting(table)  # after the model has its rows
    _fit_table_height(table)
    return table


class IndexTable(QWidget):
    """A navigable index table that says so when it has nothing in it.

    **The class this exists for.** A table with zero rows still renders its column headers, and
    ``_index_caption`` still appends "double-click a row to open" — so an empty collection page
    reads as a broken table inviting you to click a row that is not there. Twelve index pages did
    this; a user opening a fresh experiment's Comparisons section met it first. The idiom for the
    fix already existed everywhere *else* in the app — a parenthesised lowercase label that
    replaces the widget (``MeasuresPanel`` and many more) —
    and the CLI has had the wording for eight of these exact collections since it was written
    (``cli.py``'s ``(no analysis runs)``, ``(no comparisons)``, …).

    A widget rather than a parameter on :func:`_table`, for two reasons. ``_table`` returns a
    ``QTableView`` that nine call sites immediately use *as a view* (``doubleClicked.connect``,
    ``selectionModel``, ``horizontalHeader``), so it cannot return a label instead. And five
    pages rebuild in place as rows come and go, so a fix applied only at construction would not
    survive deleting the last row — which is precisely when the empty state matters.
    """

    def __init__(
        self,
        headers: list[str],
        *,
        empty_text: str,
        navigable: bool = True,
        max_rows: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._headers = headers
        self._navigable = navigable
        # `None` is the display's own budget (`tables.max_visible_rows`). A number is for a
        # table that shares its page with others — the Model manager stacks three — where
        # one long list pushing the rest off screen is worse than it scrolling in place.
        self._max_rows = max_rows
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self._notice = QLabel(empty_text)
        self._notice.setWordWrap(True)
        self._view = _table(headers, [], navigable=navigable)
        box.addWidget(self._view)
        box.addWidget(self._notice)
        self.set_rows([])

    @property
    def view(self) -> QTableView:
        """The underlying view — for ``doubleClicked``, selection, and column tweaks.

        Handed back rather than wrapped: a page that needs to stretch one column or read
        ``currentIndex`` should not have to route through this class to do it.
        """
        return self._view

    def set_empty_text(self, text: str) -> None:
        """Change what zero rows *says*.

        For a page that reads its rows off the UI thread: while the read is in flight the
        table has no rows without meaning it, so it says it is reading, and only once the rows
        land may it claim the collection is empty. An empty state is a claim about the data.
        """
        self._notice.setText(text)

    def set_cell(self, key, column: int, value) -> None:
        """Replace one cell of the row stamped ``key`` — for a column read off the UI thread.

        In place rather than through :meth:`set_rows`, which rebuilds the model and with it
        the selection a page is showing results for.
        """
        model = self._view.model()
        for r in range(model.rowCount()):
            if row_id(model.index(r, 0)) == key:
                item = _cell(value)
                item.setFont(MONO)
                model.setItem(r, column, item)
        self._view.resizeColumnsToContents()
        refit_slack(self._view)

    def set_rows(self, rows: list[tuple], *, ids: list | None = None) -> None:
        """Show ``rows``, or the notice when there are none.

        ``ids``, parallel to ``rows``, stamps each row's identity (:func:`stamp_row_id`)
        so a click/selection handler can recover it via :func:`row_id` regardless of how
        the user has sorted the view \u2014 see :func:`_table`.
        """
        model = self._view.model()
        model.removeRows(0, model.rowCount())
        n_data = len(self._headers)
        for r, row in enumerate(rows):
            model.insertRow(r)
            for c, value in enumerate(row):
                item = _cell(value)
                item.setFont(MONO)
                if c == 0 and ids is not None:
                    stamp_row_id(item, ids[r])
                model.setItem(r, c, item)
            if self._navigable:
                chevron = QStandardItem("\u203a")
                chevron.setFont(MONO)
                chevron.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                model.setItem(r, n_data, chevron)
        # Column widths were fit to an empty (or differently-filled) table at construction
        # or the last refresh; `Interactive` columns don't recompute themselves on a model
        # change the way the old `ResizeToContents` mode did.
        self._view.resizeColumnsToContents()
        refit_slack(self._view)
        # A live-refreshing page (a run in progress) can call this repeatedly while the
        # user has an active sort; `QStandardItemModel` only resorts on an explicit
        # `sort()` call, so freshly-inserted rows would otherwise sit in insertion order
        # under a header still showing last click's arrow.
        if self._view.isSortingEnabled():
            header = self._view.horizontalHeader()
            model.sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
        _fit_table_height(
            self._view,
            max_rows=self._max_rows,
            min_rows=(
                MIN_VISIBLE_ROWS
                if self._max_rows is None
                else min(MIN_VISIBLE_ROWS, self._max_rows)
            ),
        )
        self._view.setVisible(bool(rows))
        self._notice.setVisible(not rows)


#: The label and count columns of an overview row, wide enough for the longest of each
#: ("Recording information", "6 columns · 2 rows") so the buttons line up down the page.
SECTION_LABEL_WIDTH = 180
SECTION_COUNT_WIDTH = 170


def _section_row(label: str, on_open) -> tuple[QWidget, QLabel]:
    """One row of an overview dashboard (§3.1): a section label, an updatable count/status
    label, and an **Open →** button. Returns the row widget and the count label so an
    off-thread summary can fill the count in after the row is shown. ``on_open`` is a
    0-arg callable (the caller binds the section)."""
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    name = QLabel(label)
    name.setMinimumWidth(SECTION_LABEL_WIDTH)
    box.addWidget(name)
    count = QLabel("…")
    count.setMinimumWidth(SECTION_COUNT_WIDTH)
    set_label_color(count, MUTED)
    box.addWidget(count)
    button = QPushButton("Open →")
    button.clicked.connect(lambda: on_open())
    box.addWidget(button)
    # The slack goes **after** the button, not into the count. Giving the count the stretch put
    # every `Open →` against the right edge of whatever it was handed — on a wide window that is a
    # thousand pixels from the label it opens, and the pair stops reading as one row. Three
    # left-packed columns instead: the buttons still line up with each other, which is what makes
    # them a column, but they line up next to their labels rather than next to the window.
    box.addStretch()
    return row, count


def _short_dt(iso: str | None) -> str:
    """Compact ``YYYY-MM-DD HH:MM`` from an ISO-8601 ``created_at`` (no seconds), or ''."""
    if not iso:
        return ""
    return iso[:16].replace("T", " ")


def _caption_text(summary: str, *, rows: int | None = None) -> str:
    """The caption's **text** — one rule, so a page that fills its table later can restate it.

    Split out of :func:`_index_caption` rather than duplicated at the call site: a second copy
    of "when does the hint appear" is exactly how the two halves of the empty-index bug came to
    disagree in the first place.
    """
    if rows == 0:
        return summary
    return f"{summary}  ·  double-click a row to open"


def _index_caption(summary: str, *, rows: int | None = None) -> QLabel:
    """The one-line caption above a navigable index table (§3.1): a summary + the shared
    'double-click a row to open' click hint.

    ``rows=0`` drops the hint. It is half of the empty-index bug :class:`IndexTable` fixes —
    "0 dataset(s) · double-click a row to open" tells the reader to click a row that does not
    exist, and it is composed here, before the table is even built.
    """
    return _instructions(_caption_text(summary, rows=rows))


def _confirm_and_cancel(widget, title: str, message: str, fn, on_done) -> None:
    """Confirm, then run a kill-capable cancel off the UI thread.

    The cancel waits out a grace period and may force-kill a worker, so it must
    not block the UI thread. ``on_done`` rebuilds the page (now terminal); the
    worker ref is parked on ``widget`` so the QThread isn't garbage-collected
    before it finishes.
    """
    confirm = QMessageBox.question(widget, title, message)
    if confirm != QMessageBox.StandardButton.Yes:
        return
    worker = BackgroundCall(fn, receiver=widget)
    worker.done.connect(lambda _result: on_done())

    def _failed(exc) -> None:
        QMessageBox.critical(widget, title, str(exc))
        on_done()

    worker.failed.connect(_failed)
    widget._cancel_worker = worker
    worker.start()


def _fresh_signals():
    """Recompute readiness from a **fresh subprocess** run of the checks (§1) — so a
    mid-session ``pip install`` is reflected on re-check without restarting the app."""
    return signals_from(api_checks.run_checks())


class _ReadinessPanel(QFrame):
    """The setup-readiness checklist (§1): recomputes the signals off the UI thread (in a
    fresh subprocess, so a mid-session install shows on re-check) and shows a row per step,
    each with an Open button that navigates via ``on_navigate``. A **Check readiness** button
    re-runs it. Steps + ordering come from ``readiness_guide`` (shared with the page banners,
    so surfaces never drift); ``on_computed(signals)`` lets the shell reuse the same compute
    for its banner state instead of spawning its own."""

    def __init__(self, on_navigate, on_computed=None):
        super().__init__()
        self._on_navigate = on_navigate
        self._on_computed = on_computed
        self._panel_signals = None
        self._keypoints_skipped = False
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        box = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(_heading("Get started"), 1)
        self._recheck = QPushButton("Check readiness")
        self._recheck.setToolTip(
            "Re-run the checks — after installing something, this picks it up without "
            "restarting the app."
        )
        self._recheck.clicked.connect(lambda: self._compute())
        header.addWidget(self._recheck)
        box.addLayout(header)
        self._rows = QVBoxLayout()
        box.addLayout(self._rows)
        self._status = QLabel("Checking setup…")
        set_label_color(self._status, MUTED)
        self._rows.addWidget(self._status)
        self._compute()

    def _compute(self) -> None:
        # The probe spawns a subprocess + stats caches — off the UI thread so the page
        # stays responsive ("Checking setup…" until it returns).
        self._recheck.setEnabled(False)
        self._panel_signals = _CallSignals()
        self._panel_signals.done.connect(self._render)
        self._panel_signals.failed.connect(self._on_error)
        QThreadPool.globalInstance().start(
            _CallTask(_fresh_signals, self._panel_signals)
        )

    def _render(self, signals) -> None:
        self._recheck.setEnabled(True)
        self._clear_rows()
        for row in self._build_rows(signals):
            self._rows.addWidget(row)
        _make_selectable(self)  # the rows were built after MainWindow._show ran
        if self._on_computed is not None:
            self._on_computed(signals)  # keep the shell's banner state in sync

    def _clear_rows(self) -> None:
        while self._rows.count():
            item = self._rows.takeAt(0)
            discard(item.widget())

    def _on_error(self, exc) -> None:
        # run_checks is resilient, but the probe is off-thread: a failure must
        # show an error, not leave "Checking setup…" up forever (looks hung).
        self._recheck.setEnabled(True)
        self._clear_rows()
        message = QLabel(f"Couldn't check setup: {exc}")
        message.setWordWrap(True)
        set_label_color(message, ERROR)
        self._rows.addWidget(message)
        _make_selectable(self)

    def _build_rows(self, s) -> list[QWidget]:
        self._keypoints_skipped = s.keypoints_skipped
        rows = [self._row(step) for step in readiness_guide.readiness_steps(s)]
        rows.append(self._footer(readiness_guide.is_setup_complete(s)))
        return rows

    def _row(self, step) -> QWidget:
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        glyph = QLabel("✓" if step.done else "○")
        set_label_color(glyph, READY if step.done else (WARN if step.guides else MUTED))
        box.addWidget(glyph)
        box.addWidget(QLabel(step.label))
        box.addStretch()
        if step.kind == readiness_guide.KEYPOINT_STEP:
            box.addWidget(self._keypoint_choice(skipped=self._keypoints_skipped))
        button = QPushButton("Open")
        button.setEnabled(step.kind is not None)  # None → not yet built (coming soon)
        if step.kind is not None:
            button.setObjectName(step.kind)
            button.clicked.connect(lambda: self._on_navigate(step.kind))
        box.addWidget(button)
        return row

    def _keypoint_choice(self, *, skipped: bool) -> QPushButton:
        """Skip keypoint setup (with what that gives up), or take the skip back."""
        if skipped:
            button = QPushButton("Undo skip")
            button.setToolTip("Nudge for keypoint setup again")
            button.clicked.connect(lambda: self._set_keypoints_skipped(False))
        else:
            button = QPushButton("Skip…")
            button.setToolTip(
                "Go without a keypoint model; Analysis loses alignment and pose"
            )
            button.clicked.connect(self._skip_keypoints)
        return button

    def _skip_keypoints(self) -> None:
        costs = "\n".join(
            f"  • {cost}" for cost in readiness_guide.KEYPOINTS_SKIPPED_COSTS
        )
        confirm = QMessageBox.question(
            self,
            "Skip keypoints",
            "Without a keypoint model, Analysis cannot offer:\n\n"
            f"{costs}\n\n"
            "You can set keypoints up later from an experiment's Models page. Skip for now?",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._set_keypoints_skipped(True)

    def _set_keypoints_skipped(self, skipped: bool) -> None:
        readiness_api.set_keypoints_skipped(skipped)
        self._compute()

    def _footer(self, complete) -> QWidget:
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        glyph = QLabel("✓" if complete else "→")
        set_label_color(glyph, READY if complete else MUTED)
        box.addWidget(glyph)
        message = "Setup complete. You're ready to run." if complete else "New here?"
        box.addWidget(QLabel(message))
        box.addStretch()
        button = QPushButton("Set up your first experiment ↗")
        button.setObjectName("documentation")
        button.clicked.connect(lambda: self._on_navigate("documentation"))
        box.addWidget(button)
        return row


class _RemoteTargetsSummary(QFrame):
    """A non-checklist Configuration section: how many remote (Slurm) cluster profiles
    are registered, with an Open button to manage them (§6). Kept separate from the
    Get-started checklist below — zero registered targets is a normal setup, not an
    incomplete one, so it has no ✓/○ to earn."""

    def __init__(self, on_navigate):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        box = QHBoxLayout(self)
        box.addWidget(_heading("Remote targets"), 1)
        self._count = QLabel("")
        set_label_color(self._count, MUTED)
        box.addWidget(self._count)
        button = QPushButton("Open")
        button.clicked.connect(lambda: on_navigate("remote_targets"))
        box.addWidget(button)
        self._refresh()

    def _refresh(self) -> None:
        n = len(api_remote.list_remote_targets())
        self._count.setText(
            "no clusters registered yet" if n == 0 else f"{n} registered"
        )


class ConfigurationPage(QWidget):
    """Configuration landing — machine-global setup at a glance (§1). The 'Get started'
    checklist lives here (not on Home); each item opens its own Configuration page."""

    def __init__(self, on_navigate, on_computed=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Configuration"))
        layout.addWidget(
            _instructions("Setup checklist. Complete all items for full functionality.")
        )
        layout.addWidget(_ReadinessPanel(on_navigate, on_computed))
        layout.addWidget(_RemoteTargetsSummary(on_navigate))
        layout.addWidget(PreviewCachePanel())
        layout.addStretch()


class PreviewCachePanel(QWidget):
    """Configuration's "Preview cache" line (§3 *Video preview cache*): where this machine
    keeps its display-only video previews, how much disk they use, and **Clear**.

    Clearing deletes no experiment data and every preview rebuilds in the background when it
    is next needed, so it asks in place — a second click on the same row — rather than in a
    dialog (the confirmation rule in ``requirements.py`` is for what cannot be undone). The
    size walks the cache folder, so it is measured off the UI thread.
    """

    def __init__(self) -> None:
        super().__init__()
        from detail.api import previews

        self._worker = None
        self._size = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.addWidget(_heading("Preview cache"))
        layout.addWidget(
            _instructions(
                "Small copies of each video that clip montages play from. Built in the "
                "background after an import; clearing them deletes no experiment data, and "
                "they rebuild when next needed."
            )
        )
        row = QHBoxLayout()
        self._location = QLabel(str(previews.preview_cache_dir()))
        self._location.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(self._location, 1)
        self._size_label = QLabel("Measuring…")
        set_label_color(self._size_label, MUTED)
        row.addWidget(self._size_label)
        self._clear = QPushButton("Clear")
        self._clear.clicked.connect(self._on_clear)
        self._clear.setEnabled(False)
        self._clear.setToolTip("Measuring the cache…")
        row.addWidget(self._clear)
        self._keep = QPushButton("Keep")
        self._keep.clicked.connect(self._reset)
        self._keep.hide()
        row.addWidget(self._keep)
        layout.addLayout(row)
        self._measure()

    def _measure(self) -> None:
        from detail.api import previews

        self._size_label.setText("Measuring…")
        self._clear.setEnabled(False)
        self._clear.setToolTip("Measuring the cache…")
        self._worker = BackgroundCall(previews.preview_cache_size, receiver=self)
        self._worker.done.connect(self._show_size)
        self._worker.failed.connect(self._show_error)
        self._worker.start()

    def _show_size(self, size: int) -> None:
        from detail.api.exporters import format_size

        self._size = int(size)
        self._size_label.setText(format_size(self._size))
        self._reset()

    def _show_error(self, exc) -> None:
        self._size_label.setText(f"Couldn't measure: {exc}")
        set_label_color(self._size_label, ERROR)

    def _reset(self) -> None:
        self._clear.setText("Clear")
        self._clear.setEnabled(bool(self._size))
        self._clear.setToolTip(
            "Delete every preview on this machine; they rebuild when next needed."
            if self._size
            else "The preview cache is empty: nothing to clear."
        )
        self._keep.hide()

    def _on_clear(self) -> None:
        from detail.api import previews
        from detail.api.exporters import format_size

        if self._keep.isHidden():
            self._clear.setText(f"Clear {format_size(self._size)}")
            self._keep.show()
            return
        queue = preview_jobs.current()
        if queue is not None:
            queue.cancel()  # a build in flight would write into what is being cleared
        self._keep.hide()
        self._clear.setEnabled(False)
        self._clear.setToolTip("Clearing the cache…")
        self._size_label.setText("Clearing… (a build in progress stops first)")
        self._worker = BackgroundCall(previews.clear_preview_cache, receiver=self)
        self._worker.done.connect(lambda _freed: self._measure())
        self._worker.failed.connect(self._show_error)
        self._worker.start()


class GpuCudaPage(QWidget):
    """Configuration ▸ GPU / CUDA — accelerator status + a CUDA "doctor" (§1).

    Detects the accelerator off the UI thread and, when CUDA isn't working on an NVIDIA
    box, names the exact gap (a CPU-only torch wheel, a driver mismatch) and shows a
    machine-specific fix command to copy. On a non-NVIDIA machine the CUDA panel is
    greyed, not hidden."""

    def __init__(self):
        super().__init__()
        self._scan_signals = None
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("GPU / CUDA"))
        self._status = _status_label()  # the detected line: top, bold + larger
        self._status.setText("Detecting accelerator…")
        set_label_color(self._status, MUTED)
        layout.addWidget(self._status)
        layout.addWidget(
            _instructions(
                "For full tool functionality, DETAIL should be used on a machine with at "
                "least one NVIDIA GPU and CUDA. Some frontier models, such as SAM3, will "
                "not function. And other models will run too slow to be practical."
            )
        )
        self._doctor = QFrame()
        self._doctor.setFrameShape(QFrame.Shape.StyledPanel)
        self._doctor.setFrameShadow(QFrame.Shadow.Plain)
        self._doctor_box = QVBoxLayout(self._doctor)
        layout.addWidget(self._doctor)
        layout.addStretch()
        self._scan()

    def _scan(self) -> None:
        # probe imports torch + runs nvidia-smi — off the UI thread so the page opens
        # immediately (shows "Detecting…" until it returns).
        self._scan_signals = _CallSignals()
        self._scan_signals.done.connect(self._render)
        self._scan_signals.failed.connect(self._on_error)
        QThreadPool.globalInstance().start(
            _CallTask(cuda_doctor.probe, self._scan_signals)
        )

    def _on_error(self, exc) -> None:
        # The probe imports torch + runs nvidia-smi off-thread; a failure must surface,
        # not leave "Detecting accelerator…" up forever (looks hung).
        self._status.setText(f"Couldn't detect the accelerator: {exc}")
        set_label_color(self._status, ERROR)
        self._doctor.setEnabled(False)

    def _render(self, result) -> None:
        diag = cuda_doctor.diagnose(result)
        self._status.setText(diag.headline)
        set_label_color(self._status, _LEVEL_COLORS[diag.level])
        while self._doctor_box.count():
            item = self._doctor_box.takeAt(0)
            discard(item.widget())
        self._doctor_box.addWidget(_heading("CUDA setup"))
        for label, ok, detail in diag.checks:
            self._doctor_box.addWidget(_check_row(label, ok, detail))
        if diag.problem:
            problem = QLabel(diag.problem)
            problem.setWordWrap(True)
            set_label_color(problem, WARN)
            self._doctor_box.addWidget(problem)
        if diag.fix_reason:
            reason = QLabel(diag.fix_reason)
            reason.setWordWrap(True)
            self._doctor_box.addWidget(reason)
        if diag.fix_command:
            self._doctor_box.addWidget(self._fix_row(diag.fix_command))
            restart = QLabel(cuda_doctor.RESTART_AFTER_FIX)
            restart.setWordWrap(True)
            set_label_color(restart, WARN)
            self._doctor_box.addWidget(restart)
        if diag.links:
            self._doctor_box.addWidget(_link_row(diag.links))
        # Greyed only when there's nothing to act on (a non-NVIDIA box). A fixable
        # problem — e.g. PyTorch not installed — stays active so the fix is copyable,
        # even though that machine isn't "CUDA-compatible".
        self._doctor.setEnabled(diag.compatible or diag.fix_command is not None)
        _make_selectable(self._doctor)  # rows were built after MainWindow._show ran

    def _fix_row(self, command: str) -> QWidget:
        return _command_row(command)


_SAM3_NO_NVIDIA = "SAM 3 requires an NVIDIA GPU. None detected on this machine."
_SAM3_TORCH_CANT_USE_GPU = (
    "SAM 3 requires an NVIDIA GPU that PyTorch can use. This machine has one, but "
    "PyTorch can't use it yet: Configuration ▸ GPU / CUDA shows the fix."
)


class Sam3BackendPage(QWidget):
    """Configuration ▸ SAM 3 backend — the full copy-paste sequence to install SAM 3 into
    this environment (§1).

    SAM 3 ships off PyPI, so it must be cloned and installed from source. The commands come
    from ``api_checks`` and use the paths the running tool actually knows — the detail repo
    and the active environment — so the sequence for *this* machine's OS is ready to paste
    end to end (open a terminal → activate → install → clone → verify). The other OS's
    sequence is shown for reference with the relative forms. An **Open a terminal here**
    button drops a terminal into the repo folder. The backend-importable status is checked
    off the UI thread in a fresh subprocess (so it reflects a just-run install); **Check
    readiness** re-runs it and a flip notifies the shell so the checklist and banners
    update."""

    def __init__(self, on_readiness_changed=None):
        super().__init__()
        self._on_readiness_changed = on_readiness_changed
        self._scan_signals = None
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("SAM 3 backend"))
        # The GPU warning precedes the status line — SAM 3 is CUDA-only, so this frames the
        # status below it. Shown immediately on a Mac (never has CUDA), then from the check.
        self._cant_run = QLabel(_SAM3_NO_NVIDIA)
        self._cant_run.setWordWrap(True)
        set_label_color(self._cant_run, WARN)
        self._cant_run.setVisible(sys.platform == "darwin")
        layout.addWidget(self._cant_run)
        header = QHBoxLayout()
        self._status = _status_label()  # the importable verdict: top, bold + larger
        self._status.setText("Checking the SAM 3 backend…")
        set_label_color(self._status, MUTED)
        header.addWidget(self._status, 1)
        self._recheck = QPushButton("Check readiness")
        self._recheck.setToolTip(
            "Re-run the check — after installing the backend, this picks it up without "
            "restarting the app."
        )
        self._recheck.clicked.connect(lambda: self._scan())
        header.addWidget(self._recheck)
        layout.addLayout(header)
        layout.addWidget(
            _instructions(
                "Run the following commands to install SAM 3 from GitHub. Click "
                '"Check readiness" to verify installation.'
            )
        )
        for script in api_checks.setup_scripts():
            layout.addWidget(self._script_block(script))
        layout.addStretch()
        self._scan()

    def _terminal_button(self) -> QPushButton:
        repo = api_checks.detail_repo_root()
        argv = api_checks.terminal_launch_argv(str(repo)) if repo is not None else None
        button = QPushButton(f"Open {api_checks.terminal_name()}")
        if argv is None:
            button.setEnabled(False)  # grey-out, don't hide (no repo, or Linux)
            hint = (
                f"Open a terminal in {repo} yourself."
                if repo is not None
                else "Editable install not found; run these in your detail folder."
            )
            button.setToolTip(hint)
        else:
            button.setToolTip(f"Open a terminal in {repo}")
            button.clicked.connect(lambda: self._open_terminal(argv))
        return button

    def _open_terminal(self, argv: list[str]) -> None:
        # Popen returns immediately (the terminal runs independently), so it's safe on the
        # UI thread. We only open the terminal in the repo folder — never auto-run installs.
        try:
            subprocess.Popen(argv)
        except OSError as exc:
            QMessageBox.critical(
                self, "Open terminal", f"Couldn't open a terminal:\n{exc}"
            )

    def _script_block(self, script) -> QWidget:
        # Both OSes render identically: one read-only block (each command under a `#`
        # description) plus a single Copy button.
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Plain)
        box = QVBoxLayout(frame)
        title = (
            f"{script.os_label}  (this machine)"
            if script.is_current
            else script.os_label
        )
        box.addWidget(_heading(title))
        body = script.as_script()
        text = QPlainTextEdit(body)
        text.setReadOnly(True)
        text.setFont(MONO)
        text.setFixedHeight(18 * (body.count("\n") + 2))
        box.addWidget(text)
        row = QHBoxLayout()
        row.addStretch()
        if script.is_current:
            # Open a terminal in the repo folder, right next to Copy — the run-it flow.
            row.addWidget(self._terminal_button())
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(body))
        row.addWidget(copy)
        box.addLayout(row)
        return frame

    def _scan(self) -> None:
        # run_checks spawns a subprocess (import sam3) — off the UI thread so the page opens
        # immediately (shows "Checking…" until it returns).
        self._recheck.setEnabled(False)
        self._scan_signals = _CallSignals()
        self._scan_signals.done.connect(self._render)
        self._scan_signals.failed.connect(self._on_error)
        QThreadPool.globalInstance().start(
            _CallTask(api_checks.run_checks, self._scan_signals)
        )

    def _render(self, results) -> None:
        self._recheck.setEnabled(True)
        by_id = {r.check_id: r for r in results}
        result = by_id.get("sam3_backend")
        if result is not None and result.status == CheckStatus.OK:
            self._status.setText("SAM 3 backend package is importable. You're set.")
            set_label_color(self._status, READY)
        else:
            detail = result.detail if result is not None else "status unavailable"
            self._status.setText(f"SAM 3 backend not importable — {detail}")
            set_label_color(self._status, WARN)
        # SAM 3 needs CUDA; show the "can't run here" note whenever there's no CUDA device,
        # saying which gap it is: no NVIDIA GPU, or one PyTorch can't use yet.
        cuda = by_id.get("cuda")
        nvidia = by_id.get("nvidia_gpu")
        self._cant_run.setText(
            _SAM3_TORCH_CANT_USE_GPU
            if nvidia is not None and nvidia.status == CheckStatus.OK
            else _SAM3_NO_NVIDIA
        )
        self._cant_run.setVisible(cuda is None or cuda.status != CheckStatus.OK)
        if self._on_readiness_changed is not None:
            self._on_readiness_changed()

    def _on_error(self, exc) -> None:
        # The check runs off-thread; a failure must surface, not leave "Checking…" up
        # forever (looks hung).
        self._recheck.setEnabled(True)
        self._status.setText(f"Couldn't check the SAM 3 backend: {exc}")
        set_label_color(self._status, ERROR)


class HuggingFacePage(QWidget):
    """Configuration ▸ Hugging Face — token entry that gates the gated downloads (§1).

    Token validation hits the network (``whoami``), so it runs off the UI thread."""

    def __init__(self, banner=None, on_banner_dismiss=None, on_readiness_changed=None):
        super().__init__()
        self._worker = None
        self._banner = None
        self._on_readiness_changed = on_readiness_changed
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Hugging Face"))
        if banner is not None:
            self._banner = _ReadinessBanner(banner, on_banner_dismiss)
            layout.addWidget(self._banner)

        self._status = _status_label()  # at the top of the page, bold + larger
        layout.addWidget(self._status)

        layout.addWidget(
            _instructions(
                "Some model weights must be downloaded from gated repositories on "
                "Hugging Face. To use these models, create a free account and request "
                "access to model weights. Approval is automatic and takes a few minutes."
            )
        )

        layout.addWidget(_band_caption("Account & access"))
        links = QHBoxLayout()
        links.addWidget(_link_button("Create account ↗", HF_JOIN_URL))
        links.addWidget(_link_button("Create token ↗", HF_TOKEN_URL))
        links.addStretch()
        layout.addLayout(links)
        access = QHBoxLayout()
        access.addWidget(QLabel("Request access:"))
        for model_id, url in api_models.gated_model_pages():
            access.addWidget(_link_button(f"{_model_display(model_id)} ↗", url))
        access.addStretch()
        layout.addLayout(access)

        layout.addWidget(_band_caption("Token"))
        form = QFormLayout()
        self._token = QLineEdit()
        self._token.setEchoMode(QLineEdit.EchoMode.Password)
        self._token.setPlaceholderText("hf_…")
        form.addRow("Token:", self._token)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self._save_btn = QPushButton("Save token")
        self._save_btn.clicked.connect(self._save_token)
        actions.addWidget(self._save_btn)
        self._remove_btn = QPushButton("Remove token")
        self._remove_btn.clicked.connect(self._remove_token)
        actions.addWidget(self._remove_btn)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addWidget(
            _instructions(
                "A saved token is reused for every gated download. Paste a different "
                "token and Save to switch accounts."
            )
        )
        layout.addStretch()
        self._refresh_status()

    def _refresh_status(self) -> None:
        if credentials.get_hf_token() is not None:
            self._status.setText("✓ Token saved — gated model weights can download.")
            set_label_color(self._status, READY)
            self._remove_btn.setEnabled(True)
        else:
            self._status.setText(
                "No token saved — add one to download gated model weights."
            )
            set_label_color(self._status, WARN)
            sizing.set_available(
                self._remove_btn, False, "no token saved — there is nothing to remove"
            )

    def _save_token(self) -> None:
        token = self._token.text().strip()
        if not token:
            QMessageBox.warning(self, "Hugging Face", "Paste a token first.")
            return
        self._save_btn.setEnabled(False)
        self._status.setText("Checking token…")
        set_label_color(self._status, MUTED)
        self._worker = BackgroundCall(
            lambda: credentials.set_hf_token(token), receiver=self
        )
        self._worker.done.connect(self._on_valid)
        self._worker.failed.connect(self._on_invalid)
        self._worker.start()

    def _on_valid(self, account) -> None:
        self._save_btn.setEnabled(True)
        self._token.clear()
        self._status.setText(
            f"✓ Token saved (account: {account}). Gated downloads enabled."
        )
        set_label_color(self._status, READY)
        self._remove_btn.setEnabled(True)
        if self._banner is not None:
            self._banner.hide()
        if self._on_readiness_changed is not None:
            self._on_readiness_changed()

    def _on_invalid(self, exc) -> None:
        self._save_btn.setEnabled(True)
        self._status.setText(str(exc))
        set_label_color(self._status, ERROR)

    def _remove_token(self) -> None:
        credentials.clear_hf_token()
        if credentials.get_hf_token() is not None:
            # a token from DETAIL_HF_TOKEN survives — the GUI can't unset an env var
            self._status.setText(
                "A token from the DETAIL_HF_TOKEN environment variable is still in "
                "use — unset it in your shell to remove it."
            )
            set_label_color(self._status, WARN)
            sizing.set_available(
                self._remove_btn,
                False,
                "the token comes from DETAIL_HF_TOKEN — unset it in your shell",
            )
        else:
            self._refresh_status()
        if self._on_readiness_changed is not None:
            self._on_readiness_changed()


class _RemoteTargetDialog(Dialog):
    """Add or edit a remote (Slurm) target profile (§6). ``existing`` is ``None`` to add
    a new target; a target's name is its identity, so editing can't change it — remove
    and re-add to rename one."""

    def __init__(self, existing: RemoteTargetConfig | None = None, parent=None):
        super().__init__(parent)
        self._existing = existing
        self.setWindowTitle("Edit target" if existing else "Add remote target")
        layout = self.body()
        form = QFormLayout()

        self._name = QLineEdit(existing.name if existing else "")
        if existing is not None:
            self._name.setEnabled(False)
            self._name.setToolTip(
                "A target's name is its identity — remove and re-add to rename it."
            )
        form.addRow("Name:", self._name)

        self._host = QLineEdit(existing.host if existing else "")
        self._host.setPlaceholderText("tillicum")
        self._host.setToolTip(
            "Should match a Host alias in your own ~/.ssh/config, so keys, an agent, "
            "ControlMaster, and a cached Duo session are already handled."
        )
        form.addRow("SSH host:", self._host)

        self._workdir = QLineEdit(existing.remote_workdir if existing else "")
        self._workdir.setPlaceholderText("/gscratch/scrubbed/<netid>/detail_runs")
        form.addRow("Remote work directory:", self._workdir)

        self._partition = QLineEdit(existing.partition or "" if existing else "")
        form.addRow("Partition:", self._partition)

        self._account = QLineEdit(existing.account or "" if existing else "")
        form.addRow("Account:", self._account)

        self._conda_env = QLineEdit(existing.conda_env or "" if existing else "")
        form.addRow("Conda environment:", self._conda_env)

        self._modules = QLineEdit(", ".join(existing.modules) if existing else "")
        self._modules.setPlaceholderText("cuda/12.4, gcc/13.2")
        form.addRow("Modules to load:", self._modules)

        self._gpu_type = QLineEdit(existing.gpu_type or "" if existing else "")
        form.addRow("GPU type:", self._gpu_type)

        self._gpus_per_job = IntSpin()
        self._gpus_per_job.setRange(1, 8)
        self._gpus_per_job.setValue(existing.gpus_per_job if existing else 1)
        form.addRow("GPUs per job:", self._gpus_per_job)

        self._cpus_per_task = IntSpin()
        self._cpus_per_task.setRange(1, 128)
        self._cpus_per_task.setValue(existing.cpus_per_task if existing else 4)
        form.addRow("CPUs per task:", self._cpus_per_task)

        self._mem_gb = IntSpin()
        self._mem_gb.setRange(1, 2048)
        self._mem_gb.setValue(existing.mem_gb if existing else 32)
        form.addRow("Memory (GB):", self._mem_gb)

        walltime_field = RemoteTargetConfig.model_fields["walltime"]
        self._walltime = QLineEdit(
            existing.walltime if existing else walltime_field.default
        )
        self._walltime.setToolTip(walltime_field.description)
        form.addRow("Walltime:", self._walltime)

        jobs_field = RemoteTargetConfig.model_fields["max_concurrent_jobs"]
        self._max_concurrent_jobs = IntSpin()
        self._max_concurrent_jobs.setRange(1, 64)
        self._max_concurrent_jobs.setValue(
            existing.max_concurrent_jobs if existing else jobs_field.default
        )
        self._max_concurrent_jobs.setToolTip(jobs_field.description)
        form.addRow("Concurrent jobs:", self._max_concurrent_jobs)

        self._cost_per_gpu_hour = FloatSpin(PLAIN)
        self._cost_per_gpu_hour.setRange(0, 1000)
        self._cost_per_gpu_hour.setSpecialValueText("not set")
        self._cost_per_gpu_hour.setValue(
            (existing.cost_per_gpu_hour if existing else None) or 0
        )
        self._cost_per_gpu_hour.setToolTip(
            "Drives a live cost estimate on this target's runs "
            '(GPUs the job held × running hours × this rate). Leave at 0 ("not set") '
            "to show no cost estimate."
        )
        # A spin box's horizontal policy is Fixed by default, unlike the QLineEdit fields
        # above it in this same form — so of every row here, this is the one that would not
        # grow if the dialog were resized wider. Expanding (+ a doubled floor over its natural
        # hint) matches the other rows' behavior rather than leaving this one row inert.
        self._cost_per_gpu_hour.setFrame(True)
        self._cost_per_gpu_hour.setMinimumWidth(
            self._cost_per_gpu_hour.sizeHint().width() * 2
        )
        self._cost_per_gpu_hour.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        form.addRow("Cost per GPU-hour (USD):", self._cost_per_gpu_hour)

        layout.addLayout(form)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Save" if existing else "Add"
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def values(self) -> RemoteTargetConfig:
        """The entered profile.

        Raises:
            ValueError: (``pydantic.ValidationError``) on a bad name/scheduler/bound —
                the same checks ``detail remote add`` enforces.
        """
        modules = [m.strip() for m in self._modules.text().split(",") if m.strip()]
        return RemoteTargetConfig(
            name=self._name.text().strip(),
            host=self._host.text().strip(),
            remote_workdir=self._workdir.text().strip(),
            partition=self._partition.text().strip() or None,
            account=self._account.text().strip() or None,
            conda_env=self._conda_env.text().strip() or None,
            modules=modules,
            gpu_type=self._gpu_type.text().strip() or None,
            gpus_per_job=self._gpus_per_job.value(),
            cpus_per_task=self._cpus_per_task.value(),
            mem_gb=self._mem_gb.value(),
            walltime=self._walltime.text().strip()
            or RemoteTargetConfig.model_fields["walltime"].default,
            max_concurrent_jobs=self._max_concurrent_jobs.value(),
            cost_per_gpu_hour=self._cost_per_gpu_hour.value() or None,
        )


class RemoteTargetsPage(QWidget):
    """Configuration ▸ Remote targets — machine-global Slurm cluster profiles (§6) a
    detection run can submit to instead of running locally (``detail remote`` is the CLI
    twin). Not part of the Get-started checklist: zero registered targets is a normal
    setup, not an incomplete one, so this is its own Configuration destination rather
    than a checklist step."""

    _COLUMNS = [
        "Name",
        "Host",
        "Scheduler",
        "Remote workdir",
        "GPUs/job",
        "Concurrent jobs",
        "Walltime",
        "Cost/GPU-hr",
    ]

    def __init__(self):
        super().__init__()
        self._table = None
        self._targets: list[RemoteTargetConfig] = []
        self._test_worker = None
        self._push_creds_worker = None
        self._check_worker = None
        self._update_worker = None
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Remote targets"))
        layout.addWidget(
            _instructions(
                "Machine-global Slurm clusters (UW Tillicum, Hyak, a lab's own box) "
                "detection can submit to. A target's host should match a Host alias in "
                "your own ~/.ssh/config — DETAIL never stores or prompts for a credential."
            )
        )

        # Add adds a row, so it goes above the table (same reasoning as GlobalModelsPage's
        # Import); Edit/Remove/Test act on the row you selected, so they stay under it.
        creates = QHBoxLayout()
        self._add_btn = QPushButton("Add target")
        self._add_btn.clicked.connect(self._add_target)
        creates.addWidget(self._add_btn)
        creates.addStretch()
        layout.addLayout(creates)

        self._table_holder = QVBoxLayout()
        layout.addLayout(self._table_holder)

        actions = QHBoxLayout()
        self._edit_btn = QPushButton("Edit")
        self._edit_btn.clicked.connect(self._edit_target)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_target)
        self._test_btn = QPushButton("Test connection")
        self._test_btn.clicked.connect(self._test_target)
        self._push_creds_btn = QPushButton("Push credentials")
        self._push_creds_btn.clicked.connect(self._push_credentials)
        self._ssh_session_btn = QPushButton("Open SSH session")
        self._ssh_session_btn.setToolTip(
            "Opens a real terminal running `ssh <host>` — refreshes an expired "
            "SSH/Duo session without using Test connection or Push credentials, "
            "which fail fast rather than prompting. DETAIL never sees your "
            "password or Duo approval; this only opens the terminal ssh runs in."
        )
        self._ssh_session_btn.clicked.connect(self._open_ssh_session)
        self._update_btn = QPushButton("Update DETAIL…")
        self._update_btn.setToolTip(
            "Compares the DETAIL installed on the selected target with this app, and when "
            "it is behind, updates it with `git pull --ff-only` in its checkout once you "
            "confirm. It only moves that checkout forward (git refuses and changes nothing "
            "if the checkout has commits or edits of its own), and installs nothing. Do it "
            "when no run is going there."
        )
        self._update_btn.clicked.connect(self._update_target)
        # Two rows by meaning, and by width: a single row of six buttons made this page's minimum
        # width 1006px against 818px with five, which clips or scrolls on a laptop window. The
        # first acts on the entry in the table, the second on the cluster it names.
        for button in (self._edit_btn, self._remove_btn):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        cluster_actions = QHBoxLayout()
        for button in (
            self._test_btn,
            self._push_creds_btn,
            self._ssh_session_btn,
            self._update_btn,
        ):
            cluster_actions.addWidget(button)
        cluster_actions.addStretch()
        layout.addLayout(cluster_actions)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        self._targets = api_remote.list_remote_targets()
        display = [
            (
                t.name,
                t.host,
                t.scheduler,
                t.remote_workdir,
                str(t.gpus_per_job),
                str(t.max_concurrent_jobs),
                t.walltime,
                f"${t.cost_per_gpu_hour:.2f}" if t.cost_per_gpu_hour else "—",
            )
            for t in self._targets
        ]
        if self._table is not None:
            self._table.deleteLater()
        self._table = _table(
            self._COLUMNS, display, ids=[t.name for t in self._targets]
        )
        self._table.selectionModel().currentRowChanged.connect(self._on_selection)
        self._table_holder.addWidget(self._table)
        self._update_actions()

    def _selected(self) -> RemoteTargetConfig | None:
        if self._table is None:
            return None
        name = row_id(self._table.currentIndex())
        if name is None:
            return None
        return next((t for t in self._targets if t.name == name), None)

    def _on_selection(self, *_args) -> None:
        self._update_actions()

    def _update_actions(self) -> None:
        row = self._selected()
        for button in (
            self._edit_btn,
            self._remove_btn,
            self._test_btn,
            self._push_creds_btn,
            self._update_btn,
        ):
            sizing.set_available(button, row is not None, "Select a target first")
        if not api_checks.ssh_terminal_supported():
            # Grey out, don't hide: no reliable terminal-launch command exists on
            # plain Linux (outside WSL) — the same gap Sam3BackendPage's terminal
            # button leaves for the user to work around themselves.
            sizing.set_available(
                self._ssh_session_btn,
                False,
                "No terminal launcher on this machine — run `ssh <host>` yourself",
            )
        else:
            sizing.set_available(
                self._ssh_session_btn, row is not None, "Select a target first"
            )

    def _add_target(self) -> None:
        dialog = _RemoteTargetDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            config = dialog.values()
        except ValueError as e:
            QMessageBox.critical(self, "Add target", str(e))
            return
        api_remote.add_remote_target(config)
        self._rebuild_table()

    def _edit_target(self) -> None:
        row = self._selected()
        if row is None:
            return
        dialog = _RemoteTargetDialog(existing=row, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            config = dialog.values()
        except ValueError as e:
            QMessageBox.critical(self, "Edit target", str(e))
            return
        api_remote.add_remote_target(config)  # re-points the same name
        self._rebuild_table()

    def _remove_target(self) -> None:
        row = self._selected()
        if row is None:
            return
        if (
            QMessageBox.question(
                self, "Remove target", f"Remove remote target {row.name!r}?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        api_remote.remove_remote_target(row.name)
        self._rebuild_table()

    def _test_target(self) -> None:
        row = self._selected()
        if row is None:
            return
        sizing.set_available(self._test_btn, False, "Testing…")
        self._status.setText("Testing connection…")
        set_label_color(self._status, MUTED)
        self._test_worker = BackgroundCall(
            lambda: api_remote.test_remote_target(row.name), receiver=self
        )
        self._test_worker.done.connect(self._on_test_done)
        self._test_worker.failed.connect(self._on_test_failed)
        self._test_worker.start()

    def _on_test_done(self, status) -> None:
        self._update_actions()
        build = status.build
        # Reachable but not running this app's build is a warning with a fix, not a pass.
        self._status.setText(
            status.detail if build is None else f"{status.detail}\n{build.describe()}"
        )
        matches = build is None or build.matches
        set_label_color(self._status, READY if status.ok and matches else WARN)

    def _on_test_failed(self, exc) -> None:
        self._update_actions()
        self._status.setText(str(exc))
        set_label_color(self._status, ERROR)

    def _open_ssh_session(self) -> None:
        """Open a real terminal running an interactive ``ssh <host>`` login for the
        selected target — refreshes an expired SSH/Duo session without DETAIL ever
        seeing the password or the Duo approval (§6): this only opens the terminal
        ``ssh`` itself runs in, exactly as if the user had typed the command."""
        row = self._selected()
        if row is None:
            return
        argv = api_checks.ssh_refresh_argv(row.host)
        if argv is None:
            return  # button is greyed out in this case; nothing to do
        try:
            subprocess.Popen(argv)
        except OSError as exc:
            QMessageBox.critical(
                self, "Open SSH session", f"Couldn't open a terminal:\n{exc}"
            )

    def _update_target(self) -> None:
        """Compare the selected target's DETAIL with this app; offer to update it if behind.

        Two steps with a question between them, because the second one changes something on the
        user's cluster account: the check (read-only), then, only for a target that is *behind*
        and only after a yes, ``git pull --ff-only`` there (``api.remote.update_remote_build``,
        which ``detail remote update`` also calls)."""
        row = self._selected()
        if row is None:
            return
        sizing.set_available(self._update_btn, False, "Checking…")
        self._status.setText(f"Checking DETAIL on {row.name!r}…")
        set_label_color(self._status, MUTED)
        self._check_worker = BackgroundCall(
            lambda: api_remote.check_remote_build(row.name), receiver=self
        )
        self._check_worker.done.connect(self._on_update_checked)
        self._check_worker.failed.connect(self._on_update_failed)
        self._check_worker.start()

    def _on_update_checked(self, build) -> None:
        self._update_actions()
        self._status.setText(build.describe())
        set_label_color(self._status, READY if build.matches else WARN)
        if build.state != buildinfo.BEHIND:
            return  # the same build needs nothing; the others say what to do instead
        if (
            QMessageBox.question(
                self,
                "Update DETAIL",
                f"DETAIL on {build.target!r} is {build.commits or 'some'} commit(s) behind "
                f"this app ({build.remote_commit} → {build.local_commit}).\n\n"
                f"Run `git pull --ff-only` in {build.remote_root} on {build.host}?\n\n"
                "It only moves that checkout forward, and installs nothing. Update when no "
                "run is going there: a run in flight keeps importing modules as it goes.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._start_update(build)

    def _start_update(self, build) -> None:
        sizing.set_available(self._update_btn, False, "Updating…")
        self._status.setText(f"Updating DETAIL on {build.target!r}…")
        set_label_color(self._status, MUTED)
        worker = StageProgressWorker(
            lambda emit: api_remote.update_remote_build(
                build.target, before=build, progress=emit
            ),
            receiver=self,
        )
        self._update_worker = worker
        worker.tick.connect(self._on_update_tick)
        worker.done.connect(self._on_update_done)
        worker.failed.connect(self._on_update_failed)
        worker.start()

    def _on_update_tick(self, _permille: int, status: str) -> None:
        self._status.setText(status)

    def _on_update_done(self, result) -> None:
        self._update_actions()
        self._status.setText(
            result.detail
            if result.ok or not result.output
            else f"{result.detail}\n{result.output}"
        )
        set_label_color(self._status, READY if result.ok else WARN)

    def _on_update_failed(self, exc) -> None:
        self._update_actions()
        self._status.setText(str(exc))
        set_label_color(self._status, ERROR)

    def _push_credentials(self) -> None:
        """Copy every credential this machine has configured
        (``api.remote.CREDENTIAL_SPECS`` — Hugging Face today, a cloud provider's
        own access key the day this machine manages one) to the selected target
        (§6): a gated model (SAM 3, DINOv3) needs a Hugging Face token *on the
        cluster* to check access and download weights there, and nothing forwards
        this machine's own token there on its own — a remote job's shepherd and
        worker both run on the cluster and never see this machine's credentials
        file."""
        row = self._selected()
        if row is None:
            return
        sizing.set_available(self._push_creds_btn, False, "Pushing…")
        self._status.setText("Pushing credentials…")
        set_label_color(self._status, MUTED)
        self._push_creds_worker = BackgroundCall(
            lambda: api_remote.push_credentials(row.name), receiver=self
        )
        self._push_creds_worker.done.connect(self._on_push_credentials_done)
        self._push_creds_worker.failed.connect(self._on_push_credentials_failed)
        self._push_creds_worker.start()

    def _on_push_credentials_done(self, status) -> None:
        self._update_actions()
        self._status.setText(status.detail)
        set_label_color(self._status, READY if status.ok else WARN)

    def _on_push_credentials_failed(self, exc) -> None:
        self._update_actions()
        self._status.setText(str(exc))
        set_label_color(self._status, ERROR)


class ToolsPage(QWidget):
    """Tools landing (§9) — one-off runs that need no experiment, grouped by category.
    Each tool opens its own run page (``on_open``); a new tool appears here with no shell
    change (it self-registers)."""

    def __init__(self, on_open):
        super().__init__()
        import_tool_uis()
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Quick tools"))
        layout.addWidget(
            _instructions(
                "Quick, one-off runs that don't need an experiment. Pick a tool to start."
            )
        )
        by_category: dict[str, list] = {}
        for info in api_tools.list_tools():
            by_category.setdefault(info.category, []).append(info)
        for category in sorted(by_category):
            layout.addWidget(_band_caption(category))
            for info in by_category[category]:
                row = QWidget()
                box = QHBoxLayout(row)
                box.setContentsMargins(0, 0, 0, 0)
                label = QLabel(f"{info.label} — {info.description}")
                label.setWordWrap(True)
                box.addWidget(label, 1)
                button = QPushButton("Open")
                button.clicked.connect(
                    lambda _checked=False, tid=info.tool_id: on_open(tid)
                )
                box.addWidget(button)
                layout.addWidget(row)
        layout.addStretch()


class ToolRunPage(QWidget):
    """Launch one tool (§9): its config form + Run (off-thread, with a progress bar).

    On completion the new run id is handed to ``on_launched`` (the shell adds it to the
    sidebar and opens its result). The interactive over-cap confirm lives here (a button
    on the config form's page), the one-launch-surface pattern."""

    def __init__(self, tool_id: str, on_launched, on_open_run=None):
        super().__init__()
        import_tool_uis()
        self._tool_id = tool_id
        self._on_launched = on_launched
        self._on_open_run = on_open_run
        self._worker = None
        info = next(t for t in api_tools.list_tools() if t.tool_id == tool_id)
        layout = QVBoxLayout(self)
        layout.addWidget(_heading(info.label))
        layout.addWidget(_instructions(f"{info.description} No experiment needed."))
        self._config = TOOL_GUIS[tool_id].config_widget()
        layout.addWidget(self._config)
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        # A tool run can be unbounded (Quick segment over a whole recording), so it must
        # be stoppable — polled, so it finishes the frame it is on and keeps its output.
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setVisible(False)
        run_row.addWidget(self._stop_btn)
        run_row.addStretch()
        run_holder = QWidget()
        run_holder.setLayout(run_row)
        layout.addWidget(run_holder)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._add_saved_runs(layout)
        # Factor **1**, not `addStretch()`'s default of 0. At 0 the spacer merely has an
        # expanding policy and competes with every `Preferred` widget above it — measured here,
        # three 15px form labels laid out at 38px. The same reason `Dialog._top_justify` passes 1.
        layout.addStretch(1)

    def _add_saved_runs(self, layout) -> None:
        """A table of this tool's saved runs below the launch form — the folder shows its
        contents (§3.1). Double-click opens a run's result via ``on_open_run``."""
        runs = api_tools.list_tool_runs(self._tool_id)
        if not runs:
            return
        layout.addWidget(_band_caption("Saved runs"))
        layout.addWidget(_index_caption(f"{len(runs)} run(s)"))
        self._run_ids = [r["id"] for r in runs]
        rows = [
            (r["id"], r.get("status"), (r.get("result") or {}).get("summary", ""))
            for r in runs
        ]
        table = _table(
            ["Run", "Status", "Summary"], rows, navigable=True, ids=self._run_ids
        )
        if self._on_open_run is not None:
            table.doubleClicked.connect(lambda idx: self._on_open_run(row_id(idx)))
        layout.addWidget(table)

    def _run(self) -> None:
        error = self._config.validation_error()
        if error:
            QMessageBox.warning(self, "Run", error)
            return
        over = getattr(self._config, "over_cap_message", lambda: None)()
        if over is not None:
            confirm = QMessageBox.question(self, "Long video", over)
            if confirm != QMessageBox.StandardButton.Yes:
                return
        config = self._config.config()
        self._run_btn.setEnabled(False)
        self._stop_btn.setVisible(True)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setMaximum(1000)
        self._progress.setValue(0)
        worker = StageProgressWorker(
            lambda progress: api_tools.run_tool(
                self._tool_id,
                config,
                progress=progress,
                stop_check=worker.stop_requested,
            ),
            receiver=self,
        )
        self._worker = worker
        self._worker.tick.connect(self._on_tick)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._stop_btn.setEnabled(False)
            self._status.setText("stopping — finishing the current frame…")

    def _reset_controls(self) -> None:
        self._progress.setVisible(False)
        self._stop_btn.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText("")

    def _on_tick(self, permille: int, status: str) -> None:
        self._progress.setValue(permille)
        self._status.setText(status)

    def _on_done(self, record: dict) -> None:
        self._reset_controls()
        self._on_launched(record["id"])

    def _on_failed(self, exc) -> None:
        self._reset_controls()
        QMessageBox.critical(self, "Run failed", str(exc))


class ToolRunDetailPage(QWidget):
    """One saved tool run (§9): its result view (playback + export) + Delete."""

    def __init__(self, tool_id: str, run_id: str, on_deleted):
        super().__init__()
        import_tool_uis()
        self._run_id = run_id
        self._on_deleted = on_deleted
        layout = QVBoxLayout(self)
        try:
            manifest = api_tools.read_tool_run(run_id)
        except FileNotFoundError:
            layout.addWidget(_heading("Run not found"))
            layout.addStretch()
            return
        layout.addWidget(_heading(f"{manifest.get('label', tool_id)}  ·  {run_id}"))
        status = manifest.get("status")
        if status == "complete":
            layout.addWidget(TOOL_GUIS[tool_id].result_widget(manifest))
        elif status == "failed":
            error = QLabel(f"Run failed: {manifest.get('error', 'unknown error')}")
            error.setWordWrap(True)
            set_label_color(error, ERROR)
            layout.addWidget(error)
        else:
            layout.addWidget(QLabel(f"Status: {status}"))
        delete = QPushButton("Delete run")
        delete.clicked.connect(self._delete)
        layout.addWidget(delete)
        layout.addStretch()

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self, "Delete run", "Delete this tool run? This cannot be undone."
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            api_tools.delete_tool_run(self._run_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))
            return
        self._on_deleted()


class DocumentationPage(QWidget):
    """Documentation — opens the offline-built docs site in the browser.

    The full site (user guides, manual, concepts, CLI/API reference) is built into a per-user
    cache on first open and served over ``file://`` — it works with no network. The
    user guides come first: they are where someone new starts.
    """

    #: The page **User guides ↗** opens: the guides' overview, as a nav path without extension.
    GUIDES_PAGE = "user-guides/index"

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Documentation"))
        layout.addWidget(
            _instructions(
                "Step-by-step user guides, from installing DETAIL to analyzed behavior, "
                "and the full DETAIL documentation — the manual, concepts, and the CLI and "
                "API reference. Both open in your browser and work offline. The first "
                "open builds the site (a few seconds); later opens are instant."
            )
        )
        self._guides = QPushButton("User guides ↗")
        self._guides.clicked.connect(lambda: self._open_docs(self.GUIDES_PAGE))
        self._open = QPushButton("Open full docs ↗")
        self._open.clicked.connect(lambda: self._open_docs("index"))
        for button in (self._guides, self._open):
            row = QHBoxLayout()
            row.addWidget(button)
            row.addStretch()
            layout.addLayout(row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.hide()
        layout.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()
        self._worker: StageProgressWorker | None = None

    def _open_docs(self, page: str = "index") -> None:
        if not docs_access.needs_build():
            open_external.open_url(self, docs_access.local_url(page))
            return
        self._set_buttons_enabled(False)
        self._progress.setValue(0)
        self._progress.show()
        self._status.setText("Building documentation…")

        def work(emit):
            docs_access.ensure_site_built(progress=emit)  # off-thread (has its own bar)
            return docs_access.local_url(page)

        self._worker = StageProgressWorker(work, receiver=self)
        self._worker.tick.connect(self._on_tick)
        self._worker.done.connect(self._on_built)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_tick(self, permille: int, status: str) -> None:
        self._progress.setValue(permille)
        self._status.setText(status)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._guides.setEnabled(enabled)
        self._open.setEnabled(enabled)

    def _on_built(self, url: object) -> None:
        self._progress.hide()
        self._status.setText("")
        self._set_buttons_enabled(True)
        open_external.open_url(self, str(url))

    def _on_failed(self, exc: object) -> None:
        self._progress.hide()
        self._status.setText("")
        self._set_buttons_enabled(True)
        QMessageBox.critical(
            self, "Documentation", f"Could not build the documentation:\n{exc}"
        )


# --------------------------------------------------------------------------- #
# Home (Home context)
# --------------------------------------------------------------------------- #
class NewExperimentDialog(Dialog):
    """Experiment-level info for a new experiment — name required, description optional —
    and the experiment location (§2) to create it in, the default folder unless chosen.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New experiment")
        layout = self.body()
        form = QFormLayout()
        self._name = QLineEdit()
        form.addRow("Name:", self._name)
        self._desc = QLineEdit()
        form.addRow("Description (optional):", self._desc)
        # The experiment-wide default every recording inherits (§10); optional here,
        # changeable later on the Recordings page.
        self._camera_view = CameraViewPicker(unset_label="Not set")
        form.addRow("Camera view (optional):", self._camera_view)
        self._location = experiment_locations.LocationPicker()
        form.addRow("Save in:", self._location)
        layout.addLayout(form)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        box.accepted.connect(self._on_accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _on_accept(self) -> None:
        if not self._name.text().strip():
            QMessageBox.warning(self, "New experiment", "Name is required.")
            return
        try:
            camera_view_mod.validate(*self._camera_view.value())
        except ValueError as exc:
            QMessageBox.warning(self, "New experiment", str(exc))
            return
        self.accept()

    def values(self) -> tuple[str, str]:
        return self._name.text().strip(), self._desc.text().strip()

    def camera_view(self) -> tuple[str | None, str | None]:
        """The chosen default camera view and its note — ``(None, None)`` when not set."""
        return self._camera_view.value()

    def location(self) -> str:
        """The experiment location to create it in."""
        return self._location.location()


# Stages shown on an experiment card, in pipeline order. Kept short deliberately: the
# card answers "how far has this one got", not "what exactly ran".
_CARD_STAGES = (("detection", "detect"), ("tracking", "track"), ("embedding", "embed"))


def _card_facts(experiment) -> dict:
    """Everything a card says beyond its name, in one pass, off the UI thread.

    Qt-free and returning plain data, so ``HomePage`` can compute it for the whole
    workspace in a ``BackgroundCall`` — the same treatment ``preview_image`` already gets,
    for the same reason. Every read here scales with the experiment: the recording listing,
    the run records, three stats for the mtime.

    Recording-scoped CLI runs (``detection run`` ...) carry no ``run_NNN`` and so are not
    counted; those commands are being retired in favour of ``process <stage>``,
    after which the stage counts are the whole picture.
    """
    counts = {module: 0 for module, _ in _CARD_STAGES}
    for run in experiment.processing_runs:  # off-thread (see the docstring)
        if run.get("status") == "complete" and run.get("module_id") in counts:
            counts[run["module_id"]] += 1
    return {
        # recording_ids, not recordings: the count needs the listing, not a record read
        # per recording.
        "n_recordings": len(experiment.recording_ids),
        "counts": counts,
        "modified_at": experiment.modified_at,
    }


class _ExperimentCard(QFrame):
    """One experiment in the Experiments list: a still from its footage beside its
    identity and how far the pipeline has got.

    A card rather than a table row because the front door's job is recognition — every
    row of the old table looked alike, and the one thing that identifies an experiment
    (what the footage looks like) was absent. Double-click opens it, as the table did.
    """

    _THUMB_W, _THUMB_H = 160, 96
    # Every card is the same height whatever it holds, so the list scans as a list: the
    # eye tracks one column of names rather than hunting rows of varying size. The still
    # sets it; the text (name, description, counts, stages) fits inside.
    _CARD_H = _THUMB_H + 16

    def __init__(self, experiment, on_open, on_archive, *, location: str | None = None):
        super().__init__()
        self._experiment = experiment
        self._on_open = on_open
        # A real box, and a panel fill behind it: StyledPanel alone renders as a hairline
        # under Fusion, so the rows read as page background with text floating on it.
        # Shape + palette rather than a stylesheet (GUI style), and Base-vs-Window is a
        # palette pair, so the contrast survives a dark theme.
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(1)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(
            QPalette.ColorRole.Window, palette.color(QPalette.ColorRole.Base)
        )
        self.setPalette(palette)
        self.setCursor(Qt.CursorShape.PointingHandCursor)  # this row opens something
        self.setFixedHeight(self._CARD_H)

        row = QHBoxLayout(self)
        self._thumb = QLabel("…")
        self._thumb.setFixedSize(self._THUMB_W, self._THUMB_H)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._thumb)

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        name = QLabel(experiment.name)
        font = name.font()
        font.setBold(True)
        name.setFont(font)
        if location is None:
            text.addWidget(name)
        else:
            name_row = QHBoxLayout()
            name_row.addWidget(name)
            where = QLabel(
                "· Default folder"
                if location == locations_store.DEFAULT_LOCATION
                else f"· {location}"
            )
            set_label_color(where, MUTED)
            name_row.addWidget(where)
            name_row.addStretch()
            text.addLayout(name_row)
        self._description = None
        if experiment.description:
            # One line, elided: a long description must not push the card taller than its
            # neighbours or force the list to scroll sideways. The full text is the tooltip.
            self._description = QLabel(experiment.description)
            self._description.setToolTip(experiment.description)
            self._description.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            text.addWidget(self._description)
        # The counts and the mtime are a directory listing and a run record per run, which
        # is seconds on a workspace held on a network share. They arrive via `set_facts`.
        self._summary = QLabel("…")  # the thumbnail's placeholder, for the same reason
        text.addWidget(self._summary)
        self._stages = QLabel("")
        text.addWidget(self._stages)
        text.addStretch()
        row.addLayout(text, 1)

        # Archive sits on the card it acts on, and reads the state back: an archived card
        # (shown only under Show archived) says so and offers the way back.
        side = self._side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.addStretch()
        archived = experiment.archived
        if archived:
            badge = QLabel("Archived")
            set_label_color(badge, MUTED)
            side.addWidget(badge, alignment=Qt.AlignmentFlag.AlignRight)
        self.archive_button = QPushButton("Unarchive" if archived else "Archive")
        self.archive_button.setToolTip(
            "Put this experiment back on the list."
            if archived
            else "Hide this experiment from the list. Nothing in it is changed or "
            "deleted; Show archived brings it back."
        )
        self.archive_button.clicked.connect(
            lambda: on_archive(experiment, not archived)
        )
        side.addWidget(self.archive_button, alignment=Qt.AlignmentFlag.AlignRight)
        row.addLayout(side)
        self._elide_description()  # measures the button column, so after it exists
        for label in self.findChildren(QLabel):
            label.installEventFilter(self)

    def set_facts(self, facts: dict) -> None:
        """Show what :func:`_card_facts` found (called on the UI thread with its result)."""
        n = facts["n_recordings"]
        modified = (
            datetime.fromtimestamp(facts["modified_at"]).strftime("%Y-%m-%d %H:%M")
            if facts["modified_at"]
            else "—"
        )
        self._summary.setText(
            f"{n} recording{'' if n == 1 else 's'} · modified {modified}"
        )
        counts = facts["counts"]
        if sum(counts.values()) == 0:
            # "nothing yet" reads quieter than a count
            self._stages.setText("no runs yet")
            set_label_color(self._stages, MUTED)
            return
        self._stages.setText(
            "  ·  ".join(f"{label} {counts[module]}" for module, label in _CARD_STAGES)
        )

    def set_preview(self, path) -> None:
        """Show the cached still, or say why there isn't one (never an empty box)."""
        if path is None:
            self._thumb.setText("no video")
            set_label_color(self._thumb, MUTED)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._thumb.setText("no video")
            return
        self._thumb.setPixmap(
            pixmap.scaled(
                self._THUMB_W,
                self._THUMB_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide_description()

    def _elide_description(self) -> None:
        """Fit the description to one line at the card's current width.

        Called at build time as well as on resize: Qt sends no resize event to a widget
        that has never been shown, so relying on the event alone leaves the full text in
        place until something displays it.
        """
        if self._description is None:
            return
        # Measured from the card, not from the label: inside resizeEvent the child has not
        # been re-laid-out yet, so its own width is still the previous one.
        row = self.layout()
        margins = row.contentsMargins()
        available = (
            self.width()
            - self._THUMB_W
            - self._side.sizeHint().width()
            - margins.left()
            - margins.right()
            - 2 * row.spacing()
        )
        self._description.setText(
            self._description.fontMetrics().elidedText(
                self._experiment.description,
                Qt.TextElideMode.ElideRight,
                max(available, 1),
            )
        )

    def mouseDoubleClickEvent(self, event) -> None:
        self._on_open(self._experiment)
        super().mouseDoubleClickEvent(event)

    def eventFilter(self, watched, event) -> bool:
        """Open on a double-click anywhere on the card, over a label as well as beside one.

        Page text is selectable (``_make_selectable``, applied by ``MainWindow._show``),
        and a selectable QLabel *with text* starts a selection anywhere in its area and
        swallows the double-click. That made the behavior differ by card: the thumbnail
        label holding a pixmap has no text, so the click fell through and the experiment
        opened, while the same label holding the "no video" placeholder selected the
        nearest character instead — the same spot, live on one card and dead on another.

        Only the double-click is intercepted, so single-click and drag still select text.
        """
        if cleared(self):
            return False
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self._on_open(self._experiment)
            return True
        return super().eventFilter(watched, event)


class HomePage(QWidget):
    """The workspace home: one card per experiment. ``on_open(Experiment)`` opens one.
    (The setup-readiness checklist lives on the Configuration page, not here.)"""

    def __init__(self, on_open):
        super().__init__()
        self._on_open = on_open
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(_heading("Experiments"))
        self._layout.addWidget(_instructions("Double-click an experiment to open it."))
        # Names a location whose drive is unplugged, so its experiments being absent from the
        # list reads as "not here right now" rather than "gone".
        self._not_connected = _instructions("")
        self._not_connected.setVisible(False)
        self._layout.addWidget(self._not_connected)
        # The cards keep their own scroll even though the content area now scrolls too: it is
        # what keeps **New experiment** anchored below them rather than letting a long
        # workspace list push the one action off the bottom (ParaView and 3D Slicer both hold
        # their commit row outside the scrolled region for the same reason). The two never
        # fight — the page only scrolls once this has run out, and `PageScroll` leaves the
        # wheel to an inner area that still has somewhere to go.
        inner = QWidget()
        self._holder = QVBoxLayout(inner)
        self._holder.setContentsMargins(0, 0, 0, 0)
        self._holder.addStretch()  # cards pack at the top, space falls to the bottom
        self._scroll = PageScroll()
        self._scroll.setWidget(inner)
        self._layout.addWidget(self._scroll, 1)
        self._cards: list[_ExperimentCard] = []
        self._preview_worker = None
        self._facts_worker = None
        # One counter for both deliveries: a refresh is the single event that invalidates
        # either, and two counters would let one of them go stale on its own.
        self._preview_gen = 0
        self._show_archived = QCheckBox("Show archived")
        self._show_archived.toggled.connect(lambda _on: self._refresh())
        self._refresh()

        row = QHBoxLayout()
        new_btn = QPushButton("New experiment")
        new_btn.clicked.connect(self._new_experiment)
        row.addWidget(new_btn)
        self._import_btn = QPushButton("Import experiment")
        self._import_btn.clicked.connect(self._import_experiment)
        row.addWidget(self._import_btn)
        locations_btn = QPushButton("Experiment locations…")
        locations_btn.setToolTip(
            "The folders experiments are kept in — add one on an external drive"
        )
        locations_btn.clicked.connect(self._manage_locations)
        row.addWidget(locations_btn)
        row.addStretch()
        row.addWidget(self._show_archived)
        self._layout.addLayout(row)

    def _refresh(self) -> None:
        found = locations_store.experiment_locations()
        notice = experiment_locations.not_connected_notice(found)
        self._not_connected.setText(notice)
        self._not_connected.setVisible(bool(notice))
        workspace = Experiment.list_all()
        n_archived = sum(1 for e in workspace if e.archived)
        self._show_archived.setText(f"Show archived ({n_archived})")
        self._experiments = (
            workspace
            if self._show_archived.isChecked()
            else [e for e in workspace if not e.archived]
        )
        # Bumped here, not in `_load_previews`: it has to cover the cards going away, and
        # a refresh that finds no experiments returns before starting a worker at all.
        self._preview_gen += 1
        for card in self._cards:
            discard(card)
        self._cards = []
        for experiment in self._experiments:
            # Which location a card is in only says something once there is more than one.
            location = (
                locations_store.location_of(experiment.dir, among=found)
                if len(found) > 1
                else None
            )
            card = _ExperimentCard(
                experiment, self._on_open, self._set_archived, location=location
            )
            # Before the trailing stretch, so cards stay top-packed as the list grows.
            self._holder.insertWidget(self._holder.count() - 1, card)
            self._cards.append(card)
        self._load_facts()
        self._load_previews()

    def _load_facts(self) -> None:
        """Fill the counts in off the UI thread.

        A card states how many recordings an experiment has and how far its pipeline has
        got, and both are disk: a recording listing, a record per processing run, three
        stats for the mtime. Built inline that took seconds for a **single** card on
        a workspace held on an SMB share — long enough that Windows retitles the window
        "not responding" — and it scales with the number of experiments. Same treatment as
        the stills below, for the same reason: cards render immediately and the numbers
        arrive when they arrive.
        """
        experiments = list(self._experiments)
        if not experiments:
            return
        generation = self._preview_gen
        self._facts_worker = BackgroundCall(
            lambda: [_card_facts(e) for e in experiments], receiver=self
        )
        self._facts_worker.done.connect(
            lambda facts, g=generation: self._apply_facts(g, facts)
        )
        self._facts_worker.failed.connect(
            lambda _exc: None
        )  # cards keep their placeholder
        self._facts_worker.start()

    def _apply_facts(self, generation: int, facts: object) -> None:
        if generation != self._preview_gen:  # see _apply_previews
            return
        for card, card_fact in zip(self._cards, facts):
            card.set_facts(card_fact)

    def _load_previews(self) -> None:
        """Fill the stills in off the UI thread.

        The first listing of a workspace decodes a frame per experiment; after that each
        is a file read. Either way Home must not decode video while someone is looking at
        it, so cards render immediately and the pictures arrive when they arrive.
        """
        experiments = list(self._experiments)
        if not experiments:
            return
        generation = self._preview_gen
        self._preview_worker = BackgroundCall(
            lambda: [e.preview_image() for e in experiments], receiver=self
        )
        self._preview_worker.done.connect(
            lambda paths, g=generation: self._apply_previews(g, paths)
        )
        self._preview_worker.failed.connect(lambda _exc: None)  # cards keep their text
        self._preview_worker.start()

    def _apply_previews(self, generation: int, paths: object) -> None:
        # A *refresh* replaced the cards this result was computed for. The page closing is
        # the other way these cards go away, and a generation cannot see it — nothing runs
        # here on the way out — so that half is the worker's `receiver=self` (`GuiWorker`).
        if generation != self._preview_gen:
            return
        for card, path in zip(self._cards, paths):
            card.set_preview(path)

    def _set_archived(self, experiment: Experiment, archived: bool) -> None:
        try:
            if archived:
                experiment.archive()
            else:
                experiment.unarchive()
        except (OSError, ValueError) as e:
            title = "Archive experiment" if archived else "Unarchive experiment"
            QMessageBox.critical(self, title, str(e))
            return
        self._refresh()

    def _new_experiment(self) -> None:
        dialog = NewExperimentDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, description = dialog.values()
        try:
            experiment = Experiment.create(name, location=dialog.location())
        except (FileExistsError, FileNotFoundError, ValueError) as e:
            QMessageBox.critical(self, "New experiment", str(e))
            return
        if description:
            experiment.set_metadata(description=description)
        view, note = dialog.camera_view()
        if view is not None:
            experiment.set_camera_view(view, note)  # validated by the dialog's accept
        self._on_open(experiment)

    def _import_experiment(self) -> None:
        archive, _ = QFileDialog.getOpenFileName(
            self,
            "Import experiment",
            "",
            "DETAIL export (*.tar);;All files (*)",
        )
        if not archive:
            return
        location = locations_store.DEFAULT_LOCATION
        connected = [
            loc for loc in locations_store.experiment_locations() if loc.connected
        ]
        if len(connected) > 1:
            chooser = experiment_locations.ChooseLocationDialog(self)
            if chooser.exec() != QDialog.DialogCode.Accepted:
                return
            location = chooser.location()
        self._run_import(archive, replace=False, location=location)

    def _manage_locations(self) -> None:
        dialog = experiment_locations.ExperimentLocationsDialog(self)
        dialog.exec()
        if dialog.changed:
            self._refresh()

    def _run_import(self, archive: str, *, replace: bool, location: str) -> None:
        """Import off the UI thread, behind a bar that says how far it has got.

        A slim archive of a real experiment is tens of thousands of files, and a full-video one
        many GB. The button used to read "Importing…" and nothing else, which is
        indistinguishable from a hang — and an import given up on mid-extract leaves its staging
        folder behind (#916). Cancellable, because a cancel at any report leaves nothing placed:
        the import stages to a temporary folder and stops reporting before it replaces anything.
        """
        from detail.gui.busy import run_progress

        self._import_btn.setEnabled(False)
        run_progress(
            self,
            lambda emit: _import_and_find_video(archive, replace, location, emit),
            self._on_import_done,
            label="Importing experiment…",
            on_error=lambda exc: self._on_import_failed(archive, exc, location),
            on_cancel=self._reset_import_btn,
            # The import's fraction is weighed by measured seconds (`storage/export._PassCost`).
            eta=True,
        )

    def _reset_import_btn(self) -> None:
        self._import_btn.setEnabled(True)

    def _on_import_done(self, imported: tuple[Experiment, list[str]]) -> None:
        experiment, missing = imported
        self._reset_import_btn()
        self._on_open(experiment)
        if missing:
            shown = ", ".join(missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            QMessageBox.information(
                self,
                "Import experiment",
                f"Video for {len(missing)} recording(s) was not found on this computer: "
                f"{shown}{more}.\n\nOpen one of them and use Locate video on its file. "
                "Every recording whose video is in the same folder will find it too.",
            )

    def _on_import_failed(self, archive: str, exc: Exception, location: str) -> None:
        self._reset_import_btn()
        if isinstance(exc, FileExistsError):
            confirm = QMessageBox.question(
                self,
                "Import experiment",
                f"{exc}\n\nReplace the existing experiment with the imported one?",
            )
            if confirm == QMessageBox.StandardButton.Yes:
                self._run_import(archive, replace=True, location=location)
            return
        QMessageBox.critical(self, "Import experiment", str(exc))


def _import_and_find_video(
    archive: str, replace: bool, location: str, progress
) -> tuple[Experiment, list[str]]:
    """Import, then name the recordings whose referenced video is still not on this machine.

    In the import's worker because the check stats every referenced part, on a server share.
    """
    experiment = Experiment.import_archive(
        archive, replace=replace, location=location, progress=progress
    )
    names = experiment.recording_names()
    return experiment, [names.get(r, r) for r in experiment.referenced_video_missing()]


# --------------------------------------------------------------------------- #
# Experiment-context pages
# --------------------------------------------------------------------------- #
class _ReferenceControls(QWidget):
    """A single "reference in place" checkbox (Section 2) — no location picker.

    When on, the video is used where it lives (not copied); the tool remembers the file's
    folder automatically, so the user picks a *video*, never a location. ``reference_params()``
    returns the ``add_video_part`` kwargs (``storage``) or ``None`` when off."""

    toggled = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self._on = QCheckBox(
            "Reference the video in place (don't copy it into the experiment)"
        )
        self._on.setToolTip(
            "Use the video where it lives instead of importing a copy — for large libraries "
            "you can't duplicate. A moved or changed file is detected before any run uses it."
        )
        self._on.toggled.connect(self.toggled.emit)
        box.addWidget(self._on)

    def reference_params(self) -> dict | None:
        return {"storage": "reference"} if self._on.isChecked() else None


class AddRecordingsDialog(Dialog):
    """Add one or many recordings in one window: each row is a recording (name +
    video). Populate from selected video files or a folder, edit the names, then
    Create — video import runs off the UI thread."""

    def __init__(self, experiment: Experiment, on_submit, parent=None):
        super().__init__(parent)
        self._experiment = experiment
        self._on_submit = on_submit  # (specs, part_kwargs) → the app import queue
        self._meta_cols = experiment.metadata_columns
        self.setWindowTitle("Add recordings")
        layout = self.body()
        layout.addWidget(
            QLabel(
                "Each row is a recording. Video is optional (a metadata-only recording "
                "is valid). Imports run in the background after you click Create."
            )
        )

        self._table = QTableWidget(0, 2 + len(self._meta_cols))
        self._table.setObjectName("add_recordings_staging")
        # Genuinely an editor: this is the staging sheet where names and metadata are typed
        # before the recordings are created. Declared, so the exception is on the record rather
        # than inferred from the edit triggers (`requirements.EDITOR_PROPERTY`).
        self._table.setProperty(requirements.EDITOR_PROPERTY, True)
        fill_width(self._table)
        self._table.setHorizontalHeaderLabels(
            ["Recording name", "Video file", *self._meta_cols]
        )
        make_columns_resizable(self._table)
        self._table.horizontalHeader().setStretchLastSection(True)
        # Not sortable: rows here are recordings-to-be-created, in the order the user is
        # entering them, not records to rank — a header-click reorder mid-entry would be
        # actively confusing, and nothing external maps a row's *position* to an identity
        # (every read is `self._table.item(r, c)` at Create time), so there is no
        # correctness hazard either way.
        layout.addWidget(self._table, 1)

        btns = QHBoxLayout()
        add_videos = QPushButton("Add videos")
        add_videos.clicked.connect(self._add_videos)
        add_empty = QPushButton("Add empty recording")
        add_empty.clicked.connect(lambda: self._append_row("", ""))
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        btns.addWidget(add_videos)
        btns.addWidget(add_empty)
        btns.addWidget(remove)
        btns.addStretch()
        layout.addLayout(btns)

        self._reference = _ReferenceControls()
        layout.addWidget(self._reference)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        box.accepted.connect(self._create)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _add_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        for f in files:
            self._append_row(Path(f).stem, f)

    def _append_row(self, name: str, video: str) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(name))
        video_item = QTableWidgetItem(video)
        video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(r, 1, video_item)
        for c in range(2, 2 + len(self._meta_cols)):
            self._table.setItem(r, c, QTableWidgetItem(""))

    def _remove_selected(self) -> None:
        for r in sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True):
            self._table.removeRow(r)

    def _specs(self) -> list[tuple[str, str, dict]]:
        out = []
        for r in range(self._table.rowCount()):
            name = self._table.item(r, 0).text().strip()
            video = (self._table.item(r, 1).text() or "").strip()
            meta = {}
            for i, col in enumerate(self._meta_cols):
                cell = self._table.item(r, 2 + i)
                meta[col] = cell.text() if cell is not None else ""
            out.append((name, video, meta))
        return out

    def _create(self) -> None:
        rows = self._specs()
        if not rows:
            QMessageBox.warning(self, "Add recordings", "Add at least one recording.")
            return
        names = [n for n, _, _ in rows]
        if any(not n for n in names):
            QMessageBox.warning(self, "Add recordings", "Every recording needs a name.")
            return
        if len(set(names)) != len(names):
            QMessageBox.warning(
                self, "Add recordings", "Recording names must be unique."
            )
            return
        # Against existing *names*, not ids — the id is the tool's and opaque now. The
        # API would suffix a clash silently; here the user can see the row and fix it,
        # so say so instead.
        clash = sorted(set(names) & set(self._experiment.recording_names().values()))
        if clash:
            QMessageBox.warning(
                self,
                "Add recordings",
                f"These names are already used: {', '.join(clash)}",
            )
            return
        missing = [v for _, v, _ in rows if v and not Path(v).is_file()]
        if missing:
            QMessageBox.warning(
                self, "Add recordings", f"Video not found: {missing[0]}"
            )
            return
        from detail.api.imports import RecordingImportSpec

        # Names, not ids: the tool mints the id, the name is the user's (§11).
        specs = [
            RecordingImportSpec(
                name=name, metadata=meta, video_parts=[Path(video)] if video else []
            )
            for name, video, meta in rows
        ]
        # Non-blocking: hand the batch to the app import queue and return. The second arg
        # is the per-part add_video_part kwargs.
        self._on_submit(specs, self._reference.reference_params())
        self.accept()


def _experiment_summary(experiment) -> dict[str, str]:
    """Per-section summary strings for the experiment overview (§3.1). Runs OFF the UI
    thread — it reads O(recordings)/O(runs) yaml (mtime-cached). Qt-free.

    Pinned: one pass over every section of the experiment is exactly the burst
    ``read_cache.pinned`` exists for, and a summary read from one moment in time is what a
    dashboard should show anyway.
    """
    with read_cache.pinned():
        recordings = experiment.recordings  # off-thread (see the docstring)
        with_video = sum(1 for r in recordings if r.video_parts)
        empty = len(recordings) - with_video
        runs = experiment.processing_runs  # off-thread
        running = sum(1 for r in runs if r.get("status") == "running")
        failed = sum(1 for r in runs if r.get("status") == "failed")
        run_bits = [f"{len(runs)} run(s)"]
        if running:
            run_bits.append(f"{running} running")
        if failed:
            run_bits.append(f"{failed} failed")
        rec_text = f"{len(recordings)} · {with_video} with video"
        if empty:
            rec_text += f", {empty} empty"
        return {
            "metadata": f"{len(experiment.metadata_columns)} columns · "
            f"{len(experiment.recording_metadata)} rows",
            "recordings": rec_text,
            "datasets": _training_summary(experiment.datasets),  # off-thread
            "models": _training_summary(experiment.models),  # off-thread
            "runs": " · ".join(run_bits),
            "analysis": f"{len(experiment.analysis_runs)} run(s)",  # off-thread
            "comparisons": f"{len(experiment.comparisons)}",
        }


def _training_summary(refs) -> str:
    """``N`` for a Datasets / Models overview row, plus how many are still being made."""
    active = sum(1 for ref in refs if ref.status in ACTIVE_STATUSES)
    return f"{len(refs)} · {active} in progress" if active else f"{len(refs)}"


class ExperimentOverviewPage(QWidget):
    """Experiment overview (§3.1) — the experiment's landing page. Its identity + settings
    (name, description, folder, per-stage defaults, Export) sit at the top (this absorbs the
    former **Data** tab), then a summary row per section (Recording information / Recordings / Runs /
    Analysis) with a live count filled off the UI thread and an
    **Open →** that navigates via ``on_open_section(kind)``, and **Delete experiment** at the
    bottom. It never duplicates a sub-page's table."""

    # Declared once in ``gui/sections.py`` and filtered to what this build shows (§25).
    # This list used to be a third copy, and said "Runs" where the rail said
    # "Processing pipeline" for the same section.
    @property
    def _SECTIONS(self):
        return [(s.kind, s.title) for s in gui_sections.sections()]

    def __init__(
        self, experiment, on_open_section, on_deleted=None, on_export_launched=None
    ):
        super().__init__()
        self._experiment = experiment
        self._on_deleted = on_deleted
        #: Receives the export run the Export dialog submitted — the shell's
        #: `_on_export_launched`, which says it started and where it is writing.
        self._on_export_launched = on_export_launched
        self._summary_signals = None
        self._counts: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Experiment overview"))

        # Identity + settings (the former Data tab).
        form = QFormLayout()
        self._name = QLineEdit(experiment.name)
        form.addRow("Name:", self._name)
        self._desc = QLineEdit(experiment.description)
        form.addRow("Description:", self._desc)
        folder = QLabel(str(experiment.dir))
        folder.setFont(MONO)
        form.addRow("Folder:", folder)
        # The layout's own margin is what put these labels 9px right of the heading above
        # them; every other element on this page starts at the page margin.
        form.setContentsMargins(0, 0, 0, 0)
        form_box = QWidget()
        form_box.setLayout(form)
        layout.addWidget(clamp_width(form_box))
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self._export_btn = QPushButton("Export experiment…")
        self._export_btn.setToolTip(
            "Pack the whole experiment into one archive another DETAIL can import. It runs in "
            "the background and is listed under Active processes."
        )
        self._export_btn.clicked.connect(self._export)
        buttons.addWidget(self._export_btn)
        save = QPushButton("Save")
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        # The slack goes after both buttons, never between them — `_section_row` states the
        # rule and this row broke it, leaving Save 760px from Export on a wide window.
        buttons.addStretch()
        button_box = QWidget()
        button_box.setLayout(buttons)
        layout.addWidget(clamp_width(button_box))
        self._notes_box = NotesBox(experiment, "experiment")
        layout.addWidget(clamp_width(self._notes_box))
        # Section summary rows. The caption and its instruction are clamped like the rows
        # they head, or the rule under "Sections" runs 318px past the content it captions.
        layout.addWidget(clamp_width(_band_caption("Sections")))
        layout.addWidget(
            clamp_width(_instructions("Open a section to view or launch its work."))
        )
        for kind, label in self._SECTIONS:
            row, count = _section_row(label, lambda k=kind: on_open_section(k))
            self._counts[kind] = count
            layout.addWidget(clamp_width(row))

        layout.addStretch()
        delete = QPushButton("Delete experiment")
        delete.clicked.connect(self._delete_experiment)
        layout.addWidget(delete, alignment=Qt.AlignmentFlag.AlignLeft)
        self._load_summary()

    # -- off-thread section counts (never O(recordings) on the UI thread) ----- #
    def _load_summary(self) -> None:
        self._summary_signals = _CallSignals()
        self._summary_signals.done.connect(self._render_summary)
        self._summary_signals.failed.connect(self._on_summary_error)
        QThreadPool.globalInstance().start(
            _CallTask(
                lambda: _experiment_summary(self._experiment), self._summary_signals
            )
        )

    def _render_summary(self, summary: dict) -> None:
        for kind, label in self._counts.items():
            if kind in summary:
                label.setText(summary[kind])

    def _on_summary_error(self, _exc) -> None:
        for label in self._counts.values():
            if label.text() == "…":
                label.setText("—")

    # -- identity / export / delete (absorbed from the Data tab) -------------- #
    def _save(self) -> None:
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Experiment", "Name is required.")
            return
        self._experiment.set_metadata(name=name, description=self._desc.text())
        QMessageBox.information(self, "Experiment", "Saved.")

    # -- unsaved edits (gui/editing.py) --------------------------------------
    def unsaved_summary(self) -> str | None:
        name = self._name.text().strip()
        if name not in ("", self._experiment.name) or (
            self._desc.text() != self._experiment.description
        ):
            return "This experiment's name or description has unsaved changes."
        return None

    def save_edits(self) -> None:
        name = self._name.text().strip()
        if not name:
            raise ValueError("the experiment needs a name")
        self._experiment.set_metadata(name=name, description=self._desc.text())

    def discard_edits(self) -> None:
        self._name.setText(self._experiment.name)
        self._desc.setText(self._experiment.description)

    def _export(self) -> None:
        """Open the one Export dialog with the experiment bundle ticked.

        A second *trigger* for it, beside the top bar's Export…, not a second implementation
        (`gui/export.py::open_export`). This button used to run `Experiment.export` in a
        `BackgroundCall` behind a dialog of its own, so the whole feedback for a 17 GB pack
        was the button reading "Exporting…" — nothing in Active processes, no bar, and gone
        if the window closed (#916). As an export run it is listed, reports its phases, and
        outlives the window.
        """
        from detail.api.exporters import ExportSubject
        from detail.gui.export import open_export

        handle = open_export(
            self, ExportSubject("experiment", self._experiment.dir), only=("bundle",)
        )
        if handle is not None and self._on_export_launched is not None:
            self._on_export_launched(handle)

    def _delete_experiment(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete experiment",
            f"Delete experiment {self._experiment.name} and everything in it — all "
            "recordings, runs, and imported video? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._experiment.delete()
        if self._on_deleted is not None:
            self._on_deleted()


class ModelManagerPage(QWidget):
    """Every model this machine can run, on one page (§5/§12): **Global models** (imported,
    or promoted from an experiment — with metadata + a preview), the **active experiment's**
    trained models, and the **Hugging Face** foundation models; Distill + Import at the top.

    A banner mode because the library is machine-global. The experiment table is the one
    experiment-scoped part: a model trained in an experiment is usable there at once and only
    joins Global models when promoted, so without it the models a user just made are the ones
    this page did not show. Each table holds :data:`TABLE_ROWS` and scrolls past that, so the
    three fit one screen however long any of them grows.
    """

    _YOUR_COLUMNS = ["Name", "From project(s)", "Made by", "Created", "Kind"]
    TABLE_ROWS = 4

    def __init__(
        self,
        experiment,
        on_open_model,
        on_open_experiment_model,
        on_distill_launched,
        banner=None,
        on_banner_dismiss=None,
        on_readiness_changed=None,
    ):
        super().__init__()
        self._experiment = experiment  # the active one (may be None)
        self._on_open_model = on_open_model
        self._on_open_experiment_model = on_open_experiment_model
        self._on_distill_launched = on_distill_launched
        self._your = None
        self._import_worker = None
        self._models = []
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Model manager"))
        layout.addWidget(
            _instructions(
                "One model library for the machine — it spans every experiment."
            )
        )

        actions = QHBoxLayout()
        distill_btn = QPushButton("＋ Distill new model")
        distill_btn.clicked.connect(self._distill)
        import_btn = QPushButton("Import model")
        import_btn.clicked.connect(self._import_model)
        restore_btn = QPushButton("Restore from zip")
        restore_btn.setToolTip("Bring back a model saved with Export (a .zip)")
        restore_btn.clicked.connect(self._restore_model)
        actions.addWidget(distill_btn)
        actions.addWidget(import_btn)
        actions.addWidget(restore_btn)
        actions.addStretch()
        layout.addLayout(actions)
        # What the last restore brought back, and which skeleton library entry a keypoint
        # model's skeleton matched or was added as. Empty until a restore finishes.
        self._restored = QLabel()
        self._restored.setWordWrap(True)
        self._restored.hide()
        layout.addWidget(self._restored)

        layout.addWidget(_heading("Global models"))
        self._your_holder = QVBoxLayout()
        layout.addLayout(self._your_holder)
        your_actions = QHBoxLayout()
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._export_selected)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete_selected)
        your_actions.addWidget(self._export_btn)
        your_actions.addWidget(self._delete_btn)
        your_actions.addStretch()
        layout.addLayout(your_actions)

        layout.addWidget(
            _heading(
                f"Models in {experiment.name}"
                if experiment is not None
                else "Experiment models"
            )
        )
        self._experiment_holder = QVBoxLayout()
        layout.addLayout(self._experiment_holder)
        self._experiment_table = None

        layout.addWidget(_heading("Hugging Face"))
        self._hf = GlobalModelsPage(
            banner=banner,
            on_banner_dismiss=on_banner_dismiss,
            on_readiness_changed=on_readiness_changed,
            builtin_only=True,
            embedded=True,
            max_rows=self.TABLE_ROWS,
        )
        layout.addWidget(self._hf)

        note = QLabel(
            "⚠ Models live here — don't move or rename them. Use Export to copy one out."
        )
        note.setWordWrap(True)
        set_label_color(note, WARN)
        layout.addWidget(note)

        layout.addStretch()
        self._rebuild_your_models()
        self._rebuild_experiment_models()

    def refresh(self) -> None:
        """Re-read every table (called after an import/distill/delete)."""
        self._rebuild_your_models()
        self._rebuild_experiment_models()
        self._hf._rebuild_table()

    def _rebuild_experiment_models(self) -> None:
        if self._experiment_table is not None:
            self._experiment_table.deleteLater()
        if self._experiment is None:
            # A note in the house idiom, not an empty table: there is no experiment to be
            # empty — and its models are there, one click away, once one is open.
            widget = QLabel("(open an experiment to see the models trained in it)")
            set_label_color(widget, MUTED)
        else:
            ids, rows = experiment_model_rows(self._experiment)
            widget = IndexTable(
                EXPERIMENT_MODEL_COLUMNS,
                empty_text=(
                    "(no models trained in this experiment yet — ＋ Distill new model "
                    "trains one)"
                ),
                max_rows=self.TABLE_ROWS,
            )
            widget.set_rows(rows, ids=ids)
            widget.view.doubleClicked.connect(
                lambda idx: self._on_open_experiment_model(row_id(idx))
            )
        self._experiment_table = widget
        self._experiment_holder.addWidget(widget)

    def _rebuild_your_models(self) -> None:
        # Newest first, so the rows the table holds before it scrolls are the recent ones.
        self._models = sorted(
            api_models.imported_models(role=None),
            key=lambda m: m.created_at or "",
            reverse=True,
        )
        rows = []
        for m in self._models:
            projects = (
                ", ".join(
                    s.get("name") or s.get("id") or "?" for s in m.source_experiments
                )
                or "—"
            )
            rows.append(
                (
                    m.name or m.id,
                    projects,
                    m.made_by or "—",
                    _short_dt(m.created_at) or "—",
                    model_kind(m.role),
                )
            )
        if self._your is not None:
            self._your.deleteLater()
        self._your = IndexTable(
            self._YOUR_COLUMNS,
            empty_text=(
                "(no global models yet — Import model adds one from a file; a model "
                "trained in an experiment joins when you promote it)"
            ),
            max_rows=self.TABLE_ROWS,
        )
        # Row identity is the model's own index in `self._models` — stamped rather than
        # trusted from view position, since `self._models` stays in fetch order while the
        # view can be sorted by any column.
        self._your.set_rows(rows, ids=list(range(len(self._models))))
        self._your.view.doubleClicked.connect(self._open_row)
        self._your.view.selectionModel().currentRowChanged.connect(
            self._update_your_actions
        )
        self._your_holder.addWidget(self._your)
        self._update_your_actions()

    def _selected_model(self):
        if self._your is None:
            return None
        key = row_id(self._your.view.currentIndex())
        return self._models[key] if key is not None else None

    def _update_your_actions(self, *_args) -> None:
        has = self._selected_model() is not None
        sizing.set_available(self._export_btn, has, "Select a model to export")
        sizing.set_available(self._delete_btn, has, "Select a model to delete")

    def _open_row(self, index) -> None:
        key = row_id(index)
        if key is not None:
            self._on_open_model(self._models[key].id)

    def _distill(self) -> None:
        from detail.gui.fit import DistillDialog

        dialog = DistillDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._on_distill_launched(dialog.run_handle)

    def _import_model(self) -> None:
        dialog = ImportModelDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._import_worker = run_busy(
            self,
            import_selected(dialog),
            lambda _mid: self.refresh(),
            label="Importing model…",
        )

    def _restore_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Restore model", "", "Zip archive (*.zip);;All files (*)"
        )
        if not path:
            return
        self._import_worker = run_busy(
            self,
            lambda: restore_model_archive(path),
            self._on_restored,
            label="Restoring model…",
        )

    def _on_restored(self, restored: dict) -> None:
        self.refresh()
        self._restored.setText(restored_summary(restored))
        self._restored.show()

    def _export_selected(self) -> None:
        model = self._selected_model()
        if model is None:
            return
        _export_model_dialog(self, model.id)

    def _delete_selected(self) -> None:
        model = self._selected_model()
        if model is None:
            return
        if (
            QMessageBox.question(self, "Delete model", f"Delete model {model.id!r}?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        api_models.remove_imported_model(model.id)
        self.refresh()


def _export_model_dialog(widget: QWidget, model_id: str) -> None:
    """Shared 'Export…' flow (row action + detail page): pick a .zip, export off-thread."""
    dest, _ = QFileDialog.getSaveFileName(
        widget, "Export model", f"{model_id}.zip", "Zip archive (*.zip)"
    )
    if not dest:
        return
    run_progress(
        widget,
        lambda emit: api_models.export_model(model_id, dest, progress=emit),
        lambda _r: QMessageBox.information(
            widget, "Export model", f"Exported to {dest}"
        ),
        label="Exporting model…",
    )


def restore_model_archive(zip_path) -> dict:
    """Off the UI thread: restore an exported model's ``.zip`` into the catalog and say what
    came back, as ``{"ref": ImportedModelRef, "skeleton_added": bool | None}``.

    ``skeleton_added`` is None for a detector. For a keypoint model it says whether the
    skeleton was new to this machine's library (True) or matched an entry or preset already
    there (False) — read by comparing the library before and after, since the restore names
    the entry (``skeleton_ref``) but not which of the two happened.
    """
    before = {entry["id"] for entry in api_skeletons.skeletons()}
    model_id = api_models.import_model_archive(zip_path)  # off-thread: run_busy only
    ref = api_models.imported_model(model_id)
    added = None
    if ref.skeleton_ref:
        added = ref.skeleton_ref.get("id") not in before
    return {"ref": ref, "skeleton_added": added}


def restored_summary(restored: dict) -> str:
    """One sentence on a finished restore: the model, its kind, and for a keypoint model the
    Keypoints manager entry its skeleton matched or was added as."""
    ref = restored["ref"]
    kind = "keypoint model" if model_kind(ref.role) == "keypoints" else "detector"
    text = f"Restored {ref.name or ref.id} ({ref.id}), a {kind}."
    entry = ref.skeleton_ref or {}
    if entry:
        where = f"{entry.get('name')} ({entry.get('id')})"
        if restored["skeleton_added"]:
            text += f" Its skeleton was added to the Keypoints manager as {where}."
        else:
            text += f" Its skeleton matches {where}, already in the Keypoints manager."
    return text


def read_checkpoint(weights) -> dict:
    """Off the UI thread: what importing ``weights`` would make, for the import dialog.

    ``{"task", "classes", "k", "flip_idx", "skeletons"}`` — the checkpoint's task decides
    (§12 *Keypoint models in the catalog*); for a pose checkpoint ``k`` / ``flip_idx`` are what
    a skeleton is checked against and ``skeletons`` the presets and library to pick from. The
    same two readers ``import_model`` uses, so the dialog cannot judge a file differently from
    the import it leads to.

    Raises:
        ValueError: The file is not a checkpoint those readers understand.
    """
    from detail.api import models as api_models

    out = {**api_models.inspect_checkpoint(Path(weights)), "skeletons": []}
    if out["k"] is not None:
        out["skeletons"] = api_skeletons.skeletons()
    return out


class ImportModelDialog(Dialog):
    """Import a model into the catalog *with metadata* (§12): pick what kind of file and the
    file, then name it and add who-made-it / description / notes / an optional preview image.
    Replaces the bare file-picker so a model isn't imported nameless and metadata-less.

    Two kinds of file. A **model checkpoint** (``.pt``): its task decides what it imports as,
    so there is no family to type (§12 *Keypoint models in the catalog*): a typed family is how
    a pose ``.pt`` once became a detector that failed when built. Once the file is read, a pose
    checkpoint shows the **skeleton step** — pick the skeleton its points follow, and see the
    check against the model's point count and left/right pairs — and Import waits until that
    check passes. A detection checkpoint has no such step. A **keypoint model exported to
    ONNX** (§26 — a sleap-nn export) carries its keypoint names, and their roles are edited on
    the model's page afterwards, so it has neither step.
    """

    #: The skeleton picker's last row: choose a skeleton YAML instead of a library entry.
    FROM_FILE = "__from_file__"
    #: What can be imported: a checkpoint (detection or pose, read from the file) or an ONNX
    #: keypoint export.
    _FORMATS = (
        ("detection", "Model checkpoint (.pt)"),
        ("keypoints", "Keypoint model (SLEAP export, .onnx)"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import model")
        self._weights: Path | None = None
        self._preview: Path | None = None
        self._checkpoint: dict | None = None  # read_checkpoint's answer for _weights
        self._skeleton_file: Path | None = None
        self._fits = False
        self._read_worker = None
        layout = self.body()
        form = QFormLayout()

        self._format = QComboBox()
        self._format.setObjectName("import_format")
        for value, label in self._FORMATS:
            self._format.addItem(label, value)
        self._format.currentIndexChanged.connect(self._format_changed)
        form.addRow("File", self._format)

        self._weights_label = QLabel("(no file chosen)")
        self._pick = QPushButton("Choose .pt")
        self._pick.clicked.connect(self._pick_weights)
        form.addRow("Weights", _row(self._weights_label, self._pick))
        self._kind = QLabel("(choose a file to see what it is)")
        set_label_color(self._kind, MUTED)
        self._kind.setWordWrap(True)
        form.addRow("Kind", self._kind)

        # The skeleton step: shown only for a pose checkpoint.
        self._skeleton = QComboBox()
        self._skeleton.currentIndexChanged.connect(self._skeleton_chosen)
        self._skeleton_label = QLabel("Skeleton")
        form.addRow(self._skeleton_label, self._skeleton)
        self._check = QLabel()
        self._check.setWordWrap(True)
        self._check_label = QLabel("")
        form.addRow(self._check_label, self._check)
        for widget in (
            self._skeleton,
            self._skeleton_label,
            self._check,
            self._check_label,
        ):
            widget.hide()

        self._name = QLineEdit()
        form.addRow("Name", self._name)
        self._made_by = QLineEdit()
        form.addRow("Made by", self._made_by)
        self._description = QLineEdit()
        form.addRow("Description", self._description)
        self._notes = QPlainTextEdit()
        self._notes.setFixedHeight(56)
        form.addRow("Notes", self._notes)

        self._preview_label = QLabel("(none)")
        ppick = QPushButton("Choose image")
        ppick.clicked.connect(self._pick_preview)
        form.addRow("Preview", _row(self._preview_label, ppick))
        layout.addLayout(form)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Import")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)
        self._sync_ok()

    # -- what kind of file ---------------------------------------------------------- #
    def kind(self) -> str:
        """``"keypoints"`` for an ONNX keypoint export, ``"detection"`` for a checkpoint
        (which may itself be a pose model: the checkpoint says, not this)."""
        return self._format.currentData()

    def _format_changed(self) -> None:
        """A file picked for the other kind does not carry over."""
        onnx = self.kind() == "keypoints"
        self._pick.setText("Choose .onnx" if onnx else "Choose .pt")
        self._weights = None
        self._checkpoint = None
        self._weights_label.setText("(no file chosen)")
        self._kind.setText("(choose a file to see what it is)")
        set_label_color(self._kind, MUTED)
        self._show_skeleton_step(False)
        self._sync_ok()

    # -- the weights, and what they are ------------------------------------------ #
    def _pick_weights(self) -> None:
        if self.kind() == "keypoints":
            # The exporter writes export_metadata.json (and, for a downscaling model,
            # model.onnx.data) beside the .onnx; the import reads them from there.
            path, _ = QFileDialog.getOpenFileName(
                self, "Keypoint model", "", "ONNX model (*.onnx);;All files (*)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Model weights", "", "Model weights (*.pt);;All files (*)"
            )
        if path:
            self.set_weights(Path(path))

    def set_weights(self, weights: Path) -> None:
        """Take ``weights`` as the file to import and read what it is off the UI thread
        (the file picker's continuation; a test calls it directly). An ONNX export is read
        by the import itself, so there is nothing to read here."""
        self._weights = Path(weights)
        self._checkpoint = None
        self._weights_label.setText(self._weights.name)
        if not self._name.text():
            self._name.setText(self._weights.stem)
        self._show_skeleton_step(False)
        if self.kind() == "keypoints":
            self._kind.setText(
                "Keypoints · a SLEAP export: its keypoint names come from the export, and "
                "their left/right roles are set on the model's page after import."
            )
            self._kind.setPalette(self._weights_label.palette())  # plain text
            self._sync_ok()
            return
        self._kind.setText("Reading the checkpoint…")
        set_label_color(self._kind, MUTED)
        self._sync_ok()
        self._read_worker = run_busy(
            self,
            lambda: read_checkpoint(weights),
            self._on_checkpoint_read,
            label="Reading the checkpoint…",
            on_error=self._on_checkpoint_failed,
            supersedes="checkpoint",
        )

    def _on_checkpoint_read(self, checkpoint: dict) -> None:
        self._checkpoint = checkpoint
        if checkpoint["k"] is None:
            classes = ", ".join(checkpoint["classes"]) or "none"
            self._kind.setText(f"Detector ({checkpoint['task']}) · classes: {classes}")
            self._kind.setPalette(self._weights_label.palette())  # plain text
            self._show_skeleton_step(False)
        else:
            pairs = checkpoint["flip_idx"]
            swapped = (
                "no left/right flip recorded"
                if pairs is None
                else f"{sum(1 for i, j in enumerate(pairs) if i < j)} left/right pair(s)"
            )
            self._kind.setText(
                f"Keypoints · places {checkpoint['k']} points, {swapped}. Pick the "
                "skeleton its points follow."
            )
            self._kind.setPalette(self._weights_label.palette())  # plain text
            self._fill_skeletons(checkpoint["skeletons"])
            self._show_skeleton_step(True)
        self._sync_ok()

    def _on_checkpoint_failed(self, exc: BaseException) -> None:
        self._checkpoint = None
        self._kind.setText(f"Cannot import this file: {exc}")
        set_label_color(self._kind, ERROR)
        self._show_skeleton_step(False)
        self._sync_ok()

    # -- the skeleton step -------------------------------------------------------- #
    def _show_skeleton_step(self, shown: bool) -> None:
        for widget in (
            self._skeleton,
            self._skeleton_label,
            self._check,
            self._check_label,
        ):
            widget.setVisible(shown)
        if not shown:
            self._fits = False

    def _fill_skeletons(self, entries: list[dict]) -> None:
        self._skeleton.blockSignals(True)
        self._skeleton.clear()
        self._skeleton.addItem("(choose a skeleton)", None)
        k = self._checkpoint["k"] if self._checkpoint else None
        for entry in entries:
            n = len(entry["skeleton"])
            label = f"{entry['name']} ({entry['id']})"
            if f"{n} points" not in entry["name"]:  # presets already say so
                label += f" · {n} points"
            self._skeleton.addItem(label, entry["id"])
            if n != k:  # offered, so the check can say why it does not fit
                self._skeleton.setItemData(
                    self._skeleton.count() - 1,
                    f"this model places {k} points",
                    Qt.ItemDataRole.ToolTipRole,
                )
        self._skeleton.addItem("From file…", self.FROM_FILE)
        self._skeleton.blockSignals(False)
        self._skeleton_file = None
        self._check.setText("")
        self._fits = False

    def _skeleton_chosen(self, *_args) -> None:
        source = self._skeleton.currentData()
        if source == self.FROM_FILE:
            path, _ = QFileDialog.getOpenFileName(
                self, "Skeleton file", "", "Skeleton (*.yaml *.yml);;All files (*)"
            )
            if not path:
                self._skeleton.setCurrentIndex(0)
                return
            self.set_skeleton_file(Path(path))
            return
        self._skeleton_file = None
        self._run_check()

    def set_skeleton_file(self, path: Path) -> None:
        """Use the skeleton in a YAML file (the *From file…* row's continuation)."""
        self._skeleton_file = Path(path)
        index = self._skeleton.findData(self.FROM_FILE)
        self._skeleton.blockSignals(True)
        self._skeleton.setItemText(index, f"From file: {self._skeleton_file.name}")
        self._skeleton.setCurrentIndex(index)
        self._skeleton.blockSignals(False)
        self._run_check()

    def skeleton_source(self) -> str | None:
        """What ``import_model(skeleton=)`` is handed: a preset or library id, a YAML
        path, or None (a detector, or nothing chosen yet)."""
        if self._skeleton.isHidden():
            return None
        if self._skeleton.currentData() == self.FROM_FILE:
            return None if self._skeleton_file is None else str(self._skeleton_file)
        return self._skeleton.currentData()

    def _run_check(self) -> None:
        """The import's own check, run here so Import is offered only for a skeleton it
        will accept (``core.skeleton.check_against_model``)."""
        from detail.core.skeleton import check_against_model

        self._fits = False
        source = self.skeleton_source()
        if source is None or self._checkpoint is None:
            self._check.setText("")
            self._sync_ok()
            return
        try:
            skeleton = api_skeletons.resolve_skeleton(source)
            check_against_model(
                skeleton, self._checkpoint["k"], self._checkpoint["flip_idx"]
            )
        except (ValueError, KeyError, TypeError) as exc:
            self._check.setText(f"✗ {exc}")
            set_label_color(self._check, ERROR)
        else:
            self._fits = True
            pairs = len(skeleton.pairs)
            self._check.setText(
                f"✓ Fits: {len(skeleton)} points, and its {pairs} left/right pair(s) "
                "match the model's. The check cannot tell a pair named the wrong way "
                "round (left and right swapped)."
            )
            set_label_color(self._check, READY)
        self._sync_ok()

    def _sync_ok(self) -> None:
        if self._weights is None:
            reason = "Choose a weights file first"
        elif self.kind() == "keypoints":
            reason = ""  # the export is read by the import itself
        elif self._checkpoint is None:
            reason = "Waiting for the checkpoint to be read"
        elif self._checkpoint["k"] is not None and not self._fits:
            reason = "Choose a skeleton that fits this keypoint model"
        else:
            reason = ""
        sizing.set_available(self._ok, not reason, reason)

    def _pick_preview(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Preview image", "", "Images (*.png *.jpg *.jpeg);;All files (*)"
        )
        if path:
            self._preview = Path(path)
            self._preview_label.setText(self._preview.name)

    def values(self) -> tuple[Path, dict]:
        """(weights_path, kwargs) for ``import_selected``'s import call."""
        skeleton = (
            {} if self.kind() == "keypoints" else {"skeleton": self.skeleton_source()}
        )
        return self._weights, {
            **skeleton,
            "name": self._name.text().strip() or None,
            "made_by": self._made_by.text().strip() or None,
            "description": self._description.text().strip() or None,
            "notes": self._notes.toPlainText().strip() or None,
            "preview": self._preview,
        }


def import_selected(dialog: "ImportModelDialog"):
    """The import an accepted ``ImportModelDialog`` asks for, as a callable for a worker —
    the one place either Import button turns the dialog into an API call."""
    path, meta = dialog.values()
    if dialog.kind() == "keypoints":
        return lambda: api_models.import_keypoint_model(path, **meta)
    return lambda: api_models.import_model(path, **meta)


class _EditModelMetaDialog(Dialog):
    """Edit a model's made-by / description / notes (the detail-page Edit action)."""

    def __init__(self, ref, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit model")
        layout = self.body()
        form = QFormLayout()
        self._made_by = QLineEdit(ref.made_by)
        form.addRow("Made by", self._made_by)
        self._description = QLineEdit(ref.description)
        form.addRow("Description", self._description)
        self._notes = QPlainTextEdit(ref.notes)
        self._notes.setFixedHeight(56)
        form.addRow("Notes", self._notes)
        layout.addLayout(form)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Save")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def values(self) -> dict:
        return {
            "made_by": self._made_by.text().strip(),
            "description": self._description.text().strip(),
            "notes": self._notes.toPlainText().strip(),
        }


class GlobalModelDetailPage(QWidget):
    """A machine-global model's detail (§12): preview + metadata + build-history provenance —
    including, for a distilled model, the dataset it was trained on (teacher model, frame
    counts, per-class counts) and its training curve — with Edit / Set preview… / Export… /
    Delete. Reads the catalog (not an experiment), so it renders a distilled/imported model no
    experiment owns. The dataset/curve data is a copy captured at promote time (§12's
    ``distilled.py``), not read live from the source experiment, so it still renders after that
    experiment is deleted. ``experiment`` is the open one, or None: the keypoint roles editor
    borrows a frame from it to draw over."""

    def __init__(self, model_id: str, on_deleted=None, on_changed=None, *, experiment):
        super().__init__()
        self._model_id = model_id
        self._experiment = experiment
        self._on_deleted = on_deleted
        self._on_changed = on_changed
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(_heading(f"Model: {model_id}"))
        self._body = QVBoxLayout()
        self._layout.addLayout(self._body)
        self._layout.addStretch()
        self._body_widgets = []
        self._build()

    def _add(self, widget) -> None:
        add_body_widget(self._body, widget)
        self._body_widgets.append(widget)

    def _build(self) -> None:
        for widget in self._body_widgets:
            discard(widget)
        self._body_widgets = []
        try:
            ref = api_models.imported_model(self._model_id)
        except FileNotFoundError:
            self._add(QLabel("(model not found)"))
            return

        top = QHBoxLayout()
        preview = EnlargeableThumb()
        preview.setMinimumSize(240, 160)
        pix = (
            QPixmap(str(ref.preview_path))
            if ref.preview_path is not None and Path(ref.preview_path).is_file()
            else None
        )
        if pix is not None and not pix.isNull():
            preview.setPixmap(
                pix.scaledToWidth(360, Qt.TransformationMode.SmoothTransformation)
            )
            # 360px of a distilled model's own segmentation is the surface this was asked
            # for: whether the mask is right is not a question a thumbnail answers. The
            # file is re-read off the UI thread when the enlarged view asks for it.
            path = str(ref.preview_path)
            preview.set_render(
                lambda p=path: QImage(p),
                title=f"{ref.name or self._model_id} — preview",
            )
        else:
            preview.setText("(no preview)")
            set_label_color(preview, MUTED)
        top.addWidget(preview)

        form = QFormLayout()
        form.addRow("Name", QLabel(ref.name or "—"))
        form.addRow("Kind", QLabel(model_kind(ref.role)))
        form.addRow("Made by", QLabel(ref.made_by or "—"))
        form.addRow("Description", _wrap_label(ref.description or "—"))
        form.addRow("Notes", _wrap_label(ref.notes or "—"))
        form.addRow("Created", QLabel(_short_dt(ref.created_at) or "—"))
        if ref.role == "keypoints":
            if ref.details.get("model_type"):  # an ONNX export says what it is
                form.addRow("Format", QLabel(_keypoint_kind(ref.details)))
            form.addRow("Keypoints", _wrap_label(", ".join(ref.classes) or "—"))
        else:
            form.addRow("Classes", QLabel(", ".join(ref.classes) or "—"))
        form_widget = QWidget()
        form_widget.setLayout(form)
        top.addWidget(form_widget, 1)
        top_widget = QWidget()
        top_widget.setLayout(top)
        self._add(top_widget)

        if ref.skeleton is not None:
            self._add(_heading("Skeleton"))
            self._add(_skeleton_form(ref.skeleton, ref.skeleton_ref))

        prov = ref.provenance or {}
        if ref.source_experiments or prov:
            self._add(_heading("Build history"))
            projects = (
                ", ".join(
                    s.get("name") or s.get("id") or "?" for s in ref.source_experiments
                )
                or "—"
            )
            bh = QFormLayout()
            bh.addRow("From project(s)", QLabel(projects))
            bh.addRow("Dataset", QLabel(str(prov.get("source_dataset_id") or "—")))
            bh.addRow("Model size", QLabel(str(prov.get("base_model") or "—")))
            bh.addRow("Training framework", QLabel(str(prov.get("trainer_id") or "—")))
            bh.addRow("Metrics", QLabel(str(prov.get("metrics") or {})))
            dataset = prov.get("dataset") or {}
            if dataset:
                n_train = dataset.get("n_train") or 0
                n_val = dataset.get("n_val") or 0
                requested = dataset.get("n_frames")
                frames_text = f"{n_train + n_val} used"
                if requested:
                    frames_text += f" of {requested} requested"
                bh.addRow("Teacher model", QLabel(str(dataset.get("model") or "—")))
                bh.addRow("Frames labeled", QLabel(frames_text))
                bh.addRow("Train / val split", QLabel(f"{n_train} / {n_val}"))
                per_class = dataset.get("per_class") or {}
                if per_class:
                    bh.addRow("Per-class frames", QLabel(str(per_class)))
            bh_widget = QWidget()
            bh_widget.setLayout(bh)
            self._add(bh_widget)

            history = prov.get("training_history") or []
            if history:
                from detail.gui.training_curve import TrainingCurveBand

                band = TrainingCurveBand()
                band.set_history(history)
                self._add(band)

        from detail.models import onnx_keypoints

        onnx_import = onnx_keypoints.is_imported(ref.id)
        if onnx_import and ref.skeleton is not None and not ref.skeleton.pairs:
            self._add(
                _wrap_label(
                    "No left/right pairs: set them with Edit roles so a mirrored "
                    "animal's sides are told apart."
                )
            )

        actions = QHBoxLayout()
        for label, slot in (
            ("Edit", self._edit),
            *((("Edit roles", self._edit_roles),) if onnx_import else ()),
            ("Set preview", self._set_preview),
            ("Export", self._export),
            ("Delete", self._delete),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            actions.addWidget(button)
        actions.addStretch()
        actions_widget = QWidget()
        actions_widget.setLayout(actions)
        self._add(actions_widget)

    def _notify_changed(self) -> None:
        self._build()
        if self._on_changed is not None:
            self._on_changed()

    def _edit(self) -> None:
        try:
            ref = api_models.imported_model(self._model_id)
        except FileNotFoundError:
            return
        dialog = _EditModelMetaDialog(ref, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        api_models.update_model_metadata(self._model_id, **dialog.values())
        self._notify_changed()

    def _edit_roles(self) -> None:
        """The left/right pairs and edges of an imported keypoints model; the names are its
        outputs and stay (§26), which the API refuses to change with the reason."""
        dialog = KeypointRolesDialog(
            api_models.keypoint_model_skeleton(self._model_id),
            experiment=self._experiment,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            api_models.set_keypoint_model_skeleton(self._model_id, dialog.skeleton())
        except ValueError as exc:
            QMessageBox.critical(self, "Keypoint roles", str(exc))
            return
        self._notify_changed()

    def _set_preview(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Preview image", "", "Images (*.png *.jpg *.jpeg);;All files (*)"
        )
        if not path:
            return
        api_models.set_model_preview(self._model_id, path)
        self._notify_changed()

    def _export(self) -> None:
        _export_model_dialog(self, self._model_id)

    def _delete(self) -> None:
        if (
            QMessageBox.question(
                self, "Delete model", f"Delete model {self._model_id!r}?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        api_models.remove_imported_model(self._model_id)
        if self._on_deleted is not None:
            self._on_deleted()


_KEYPOINT_KINDS = {
    "single_instance": "single-instance (whole frame)",
    "centered_instance": "centered-instance (a crop around the tracked animal)",
    "topdown": "top-down (finds the animals, keeps the tracked one)",
}


def _keypoint_kind(details: dict) -> str:
    """``SLEAP top-down (finds the animals, keeps the tracked one) · RGB · sleap-nn 0.3.3``."""
    kind = _KEYPOINT_KINDS.get(
        details.get("model_type"), str(details.get("model_type"))
    )
    channels = {1: "greyscale", 3: "RGB"}.get(details.get("channels"), "?")
    bits = [f"SLEAP {kind}", channels]
    if details.get("crop_size"):
        bits.append("crop {}x{}".format(*details["crop_size"]))
    if details.get("exporter_version"):
        bits.append(f"sleap-nn {details['exporter_version']}")
    return " · ".join(bits)


class KeypointRolesDialog(Dialog):
    """Edit an imported keypoints model's pairs and edges on the skeleton editor (§26). The
    keypoints are the model's outputs: renaming, adding or removing one is refused on save.
    """

    def __init__(self, skeleton, *, experiment, parent=None) -> None:
        from detail.gui.skeleton_editor import SkeletonEditor

        super().__init__(parent)
        self.setWindowTitle("Keypoint roles")
        body = self.body()
        note = QLabel(
            "Set which points pair as left and right, and the edges between them. The "
            "keypoint names are the model's outputs and stay as they are."
        )
        note.setWordWrap(True)
        body.addWidget(note)
        self._editor = SkeletonEditor(experiment=experiment)
        self._editor.load(skeleton)
        body.addWidget(self._editor, 1)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def skeleton(self):
        return self._editor.skeleton()


def _skeleton_form(skeleton, skeleton_ref: dict | None) -> QWidget:
    """A keypoint model's skeleton as rows: its points, its left/right pairs, and the
    Keypoints manager entry it was matched or added to. The model is built from its own
    copy (§12 *Keypoint models in the catalog*), so this shows that copy, not the entry.
    """
    pairs = ", ".join(f"{a} ↔ {b}" for a, b in skeleton.pairs) or "none"
    entry = skeleton_ref or {}
    form = QFormLayout()
    form.addRow("Points", _wrap_label(", ".join(skeleton.names)))
    form.addRow("Left/right pairs", _wrap_label(pairs))
    form.addRow(
        "In the Keypoints manager",
        QLabel(f"{entry.get('name')} ({entry.get('id')})" if entry else "—"),
    )
    widget = QWidget()
    widget.setLayout(form)
    return widget


def _row(*widgets) -> QWidget:
    """A horizontal row of widgets as a single form field (the first stretches)."""
    box = QHBoxLayout()
    box.setContentsMargins(0, 0, 0, 0)
    for i, widget in enumerate(widgets):
        box.addWidget(widget, 1 if i == 0 else 0)
    holder = QWidget()
    holder.setLayout(box)
    return holder


def _wrap_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


class ExperimentRecordingsPage(QWidget):
    """Recording list (real); add recordings; open one. Calls ``on_open(recording_id)``."""

    def __init__(
        self,
        experiment: Experiment,
        on_open,
        on_changed=None,
        on_import=None,
        pending_source=None,
        cancel_imports=None,
    ):
        super().__init__()
        self._experiment = experiment
        self._on_open = on_open
        self._on_changed = on_changed  # notify the shell to refresh the nav tree
        self._on_import = on_import  # submit import specs to the app queue (§3)
        self._pending_source = pending_source  # sid -> queued source path (in-memory)
        self._cancel_imports = cancel_imports
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(_heading("Recordings"))
        self._layout.addLayout(self._camera_view_row())
        self._table = None
        self._ids: list[str] = []
        #: Bumped by every rebuild; rows delivered against an older one are dropped.
        self._table_gen = 0
        self._rows_worker = None
        #: Whether a fill is in flight — i.e. whether the table's zero rows mean anything yet.
        self._reading = False
        # The three ways to add a recording go **above** the table, because the table is the
        # thing that grows: on a sixty-recording experiment a button under it is a button you
        # scroll to. Cancel imports stays with them — it is the same row's own undo, meaningless
        # anywhere else, and it is hidden unless an import is actually running.
        row = QHBoxLayout()
        add = QPushButton("Add recordings")
        add.clicked.connect(self._add_recordings)
        import_bundle = QPushButton("Import bundle")
        import_bundle.clicked.connect(self._import_bundle)
        import_labels = QPushButton("Import behavior labels")
        import_labels.clicked.connect(self._import_behavior_labels)
        row.addWidget(add)
        row.addWidget(import_bundle)
        row.addWidget(import_labels)
        self._cancel_btn = QPushButton("Cancel imports")
        self._cancel_btn.clicked.connect(self._cancel)
        self._cancel_btn.setVisible(cancel_imports is not None)
        sizing.set_available(
            self._cancel_btn,
            getattr(self, "_import_active", False),
            "No imports are running",
        )
        row.addWidget(self._cancel_btn)
        row.addStretch()
        self._layout.addLayout(row)

        self._table_holder = QVBoxLayout()
        self._layout.addLayout(self._table_holder)
        self._rebuild_table()
        self._layout.addStretch()

    def _camera_view_row(self) -> QHBoxLayout:
        """The experiment's default camera view (§10), which every recording inherits unless
        its Scene geometry sets its own. Saved the moment it changes, like a rename."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera view (experiment default):"))
        self._camera_view = CameraViewPicker(unset_label="Not set")
        # Read off the UI thread like the table beside it (`test_page_read_cost`): the picker
        # waits, greyed, until it knows what it is showing.
        self._camera_view.setEnabled(False)
        self._camera_view.changed.connect(self._on_camera_view_changed)
        experiment = self._experiment
        self._camera_view_call = BackgroundCall(
            lambda: experiment.camera_view, receiver=self  # off-thread
        )
        self._camera_view_call.done.connect(self._show_camera_view)
        self._camera_view_call.failed.connect(lambda _exc: None)
        self._camera_view_call.start()
        row.addWidget(self._camera_view, 1)
        hint = QLabel("A recording from another rig sets its own in Scene geometry.")
        set_label_color(hint, MUTED)
        row.addWidget(hint)
        class_names = QPushButton("Class names…")
        class_names.setToolTip(
            "Name each detection class as this experiment calls it (Resident, Intruder…)"
        )
        class_names.clicked.connect(self._edit_class_names)
        row.addWidget(class_names)
        return row

    def _edit_class_names(self) -> None:
        """Read the class roster off the UI thread (it reads every run record), then edit
        it (§1 *Class names*). Nothing on this page shows a class; every surface that does
        reads the names when it is built."""
        from detail.gui.class_names_dialog import ClassNamesDialog

        experiment = self._experiment

        def edit(names) -> None:
            ClassNamesDialog(experiment, names, parent=self).exec()

        self._class_names_call = run_busy(
            self,
            experiment.class_names,
            edit,
            label="Reading classes…",
            supersedes="class names",
        )

    def _show_camera_view(self, default) -> None:
        self._camera_view.set_value(default.view, default.note)
        self._camera_view.setEnabled(True)

    def _on_camera_view_changed(self) -> None:
        try:
            self._experiment.set_camera_view(*self._camera_view.value())
        except ValueError as exc:
            QMessageBox.warning(self, "Camera view", str(exc))
            default = self._experiment.camera_view
            self._camera_view.set_value(default.view, default.note)
            return
        # Every inheriting row's Camera view cell just changed with it.
        self._rebuild_table()

    #: Column headers, here rather than inside the build so the empty table has them before
    #: the rows arrive.
    _HEADERS = (
        "Recording",
        "Status",
        "Source",
        "Parts",
        "Streams",
        "Field of view",
        "Camera view",
        "Detections",
    )
    #: What a table with no rows says — and the two are **not** interchangeable. Reading the
    #: rows takes a moment (a record per recording, on a share a second or more), and for that
    #: moment the table has zero rows without meaning it: showing the settled notice there
    #: tells someone with sixty recordings that they have none, under a button offering to
    #: import some. An empty state is a claim about the data, so it may only be shown once the
    #: data has been read.
    _EMPTY_TEXT = (
        "(no recordings yet — Add recordings imports videos into this experiment)"
    )
    _READING_TEXT = "(reading recordings…)"

    def _rebuild_table(self) -> None:
        """Put the table up now; fill it from disk off the UI thread.

        A row costs a read of that recording's ``recording.yaml`` — seconds of frozen
        window over a large experiment on a network share, and still over a second with
        the reads collapsed (``read_cache.pinned()``). So the
        rows are computed in a worker and the table is built when they land.
        """
        self._table_gen += 1
        generation = self._table_gen
        self._reading = True
        if self._table is None:
            # First build: the page's shape, and a notice saying why it is still blank —
            # never the settled empty state, which would be a false claim (see _EMPTY_TEXT).
            self._install_table([], self._READING_TEXT)
        experiment = self._experiment
        self._rows_worker = BackgroundCall(
            lambda: _recording_rows(experiment), receiver=self  # off-thread
        )
        self._rows_worker.done.connect(
            lambda result, g=generation: self._apply_rows(g, result)
        )
        self._rows_worker.failed.connect(
            lambda _exc: None
        )  # the table keeps what it has
        self._rows_worker.start()

    def _apply_rows(self, generation: int, result) -> None:
        # A later rebuild owns the table now (the generation); the page going away is the
        # other way these widgets vanish and nothing runs on the way out, which is the
        # worker's `receiver=self` (`GuiWorker`).
        if generation != self._table_gen:
            return
        ids, rows, active = result
        # The pending source comes from the **in-memory** import queue, so it is read here
        # rather than in the worker — and it is the truer answer while an import is live,
        # since the part is not written yet (TRIAL_NOTES #3).
        if self._pending_source is not None:
            rows = [
                (row[0], row[1], self._pending_source(sid) or row[2], *row[3:])
                for sid, row in zip(ids, rows)
            ]
        self._ids = ids
        self._reading = False  # the table's zero rows now *mean* zero recordings
        self._install_table(rows, self._EMPTY_TEXT)
        self._import_active = active
        if getattr(self, "_cancel_btn", None) is not None:
            self._cancel_btn.setVisible(self._cancel_imports is not None)
            self._cancel_btn.setEnabled(active)

    def _install_table(self, rows, empty_text: str) -> None:
        table = IndexTable(list(self._HEADERS), empty_text=empty_text)
        # `self._ids` is already in sync with `rows` by the time this runs — either both are
        # still empty (the initial "(reading recordings…)" build) or `_apply_rows` just set
        # both together.
        table.set_rows(rows, ids=self._ids)
        table.view.doubleClicked.connect(self._open_row)
        if self._table is not None:
            self._table.deleteLater()
        self._table_holder.addWidget(table)
        self._table = table

    def refresh(self) -> None:
        """Re-read the recordings table (the shell calls this on import events)."""
        self._rebuild_table()

    def _cancel(self) -> None:
        if self._cancel_imports is not None:
            self._cancel_imports()
            self._rebuild_table()

    def _open_row(self, index) -> None:
        key = row_id(index)
        if key is not None:
            self._on_open(key)

    def _add_recordings(self) -> None:
        on_submit = self._on_import or self._import_synchronously
        if (
            AddRecordingsDialog(self._experiment, on_submit, self).exec()
            == QDialog.DialogCode.Accepted
        ):
            self._rebuild_table()
            if self._on_changed is not None:
                self._on_changed()

    def _import_synchronously(self, specs, part_kwargs) -> None:
        """Fallback when no app import queue is wired (e.g. tests) — blocks."""
        from detail.api.imports import import_specs

        import_specs(self._experiment, specs, part_kwargs=part_kwargs)

    def _import_behavior_labels(self) -> None:
        """Batch-import behavior ground-truth labels onto existing recordings (§3)."""
        from detail.gui.streams import ImportBehaviorLabelsDialog

        if (
            ImportBehaviorLabelsDialog(self._experiment, self).exec()
            == QDialog.DialogCode.Accepted
        ):
            self._rebuild_table()
            if self._on_changed is not None:
                self._on_changed()

    def _import_bundle(self) -> None:
        """Import from a YAML bundle folder: validate the whole manifest, confirm,
        then hand the specs to the import queue (§3)."""
        from detail.api.imports import build_specs
        from detail.storage import manifest as manifest_store

        folder = QFileDialog.getExistingDirectory(self, "Select import bundle folder")
        if not folder:
            return
        try:
            manifest = manifest_store.load_manifest(folder)
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.critical(self, "Import bundle", str(exc))
            return
        existing = [s.id for s in self._experiment.recordings]
        errors, warnings = manifest_store.validate_manifest(folder, manifest, existing)
        if errors:
            QMessageBox.critical(
                self,
                "Import bundle",
                "Cannot import — fix these and retry:\n\n" + "\n".join(errors[:20]),
            )
            return
        specs = build_specs(folder, manifest)
        summary = f"Import {len(specs)} recording(s) from this bundle?"
        if warnings:
            summary += "\n\nWarnings:\n" + "\n".join(warnings[:10])
        if (
            QMessageBox.question(self, "Import bundle", summary)
            != QMessageBox.StandardButton.Yes
        ):
            return
        (self._on_import or self._import_synchronously)(specs, None)
        self._rebuild_table()


class ExperimentRunsPage(QWidget):
    """Processing-runs index. Open a run → ``on_open(run_id)``; launch a new run
    → ``on_launched(AttachedRun)``."""

    def __init__(self, experiment: Experiment, on_open, on_launched=None):
        super().__init__()
        self._experiment = experiment
        self._on_open = on_open
        self._on_launched = on_launched
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Processing pipeline"))
        runs = experiment.processing_runs
        running = sum(1 for r in runs if r.get("status") == "running")
        failed = sum(1 for r in runs if r.get("status") == "failed")
        bits = [f"{len(runs)} run(s)"]
        if running:
            bits.append(f"{running} running")
        if failed:
            bits.append(f"{failed} failed")
        layout.addWidget(_index_caption(" · ".join(bits), rows=len(runs)))
        rows = [
            (
                r["id"],
                r.get("module_id"),
                run_cost.compute_label(r.get("compute")),
                len(r.get("recording_ids", [])),
                r.get("status"),
                _short_dt(r.get("created_at")),
            )
            for r in runs
        ]
        self._ids = [r["id"] for r in runs]
        new_run = QPushButton("New run")
        new_run.clicked.connect(self._launch)
        layout.addWidget(new_run, alignment=Qt.AlignmentFlag.AlignLeft)
        table = IndexTable(
            ["Run", "Module", "Where", "Recordings", "Status", "Started"],
            empty_text="(no processing runs yet — New run starts one)",
        )
        table.set_rows(rows, ids=self._ids)
        table.view.doubleClicked.connect(lambda idx: on_open(row_id(idx)))
        layout.addWidget(table)
        layout.addStretch()

    def _launch(self) -> None:
        from detail.gui.launch import LaunchRunDialog

        dialog = LaunchRunDialog(self._experiment, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_launched:
            self._on_launched(dialog.run_handle)


class _CallSignals(QObject):
    done = pyqtSignal(object)  # the callable's return value — on the UI thread
    failed = pyqtSignal(object)  # the exception it raised — same thread


class _CallTask(QRunnable):
    """Run a 0-arg callable on the global thread pool and report back on the UI thread:
    ``done`` with the return value, or ``failed`` with the exception it raised — so a
    consumer can render an error state instead of hanging on its "checking…" placeholder
    forever (a bare probe that raises used to vanish here silently). Delivery is a queued
    cross-thread signal; if the receiving widget is torn down first Qt drops the queued
    slot call, and the ``RuntimeError`` guard covers the signals object itself being
    deleted (widget gone / app closing) — so neither path crashes the pool (the same
    idiom as the embedding-preview loader)."""

    def __init__(self, fn, signals: "_CallSignals") -> None:
        super().__init__()
        self._fn = fn
        self._signals = signals

    def run(self) -> None:
        try:
            result = self._fn()
        except (
            Exception
        ) as exc:  # noqa: BLE001 — boundary: report it, never crash the pool
            self._emit("failed", exc)
            return
        self._emit("done", result)

    def _emit(self, signal_name: str, payload) -> None:
        # The signal *lookup* itself must be inside the guard: if the widget (and its
        # _CallSignals) was torn down while the task ran, accessing the bound signal on
        # the deleted C++ object raises RuntimeError — catch it, don't crash the pool.
        try:
            getattr(self._signals, signal_name).emit(payload)
        except RuntimeError:
            pass  # the signals object was torn down (widget gone / app closing)


class GlobalModelsPage(QWidget):
    """All machine-global models — built-in (registries) + imported weights — in one
    catalog (§5). Experiment-independent: reached from the top-level Models node.
    Per-experiment *trained* models live under each experiment's "Trained models"."""

    # One row per (model, variant): install state is per variant — each DINOv3 size is
    # its own repo and its own download — so a single row per model reported the default
    # variant's state for every size, and Install only ever fetched the default.
    _COLUMNS = [
        "Id",
        "Size / variant",
        "Role",
        "Architecture",
        "Source",
        "Status",
        "Classes",
    ]

    def __init__(
        self,
        banner=None,
        on_banner_dismiss=None,
        on_readiness_changed=None,
        *,
        builtin_only=False,
        embedded=False,
        max_rows=None,
    ):
        super().__init__()
        # ``builtin_only`` restricts the table to the Hugging Face / foundation models —
        # imported/distilled models live in the Model manager's "Global models" table, so
        # Import/Remove are hidden here. ``embedded`` drops the page's own heading + trailing
        # stretch so it can slot under a subheading inside ModelManagerPage.
        self._builtin_only = builtin_only
        self._max_rows = (
            max_rows  # see `IndexTable` — a table sharing a page with others
        )
        self._import_worker = None
        self._check_worker = None
        self._install_worker = None
        self._scan_signals = None
        self._installed = None
        self._scan_failed = False
        self._table = None
        self._banner = None
        self._on_readiness_changed = on_readiness_changed
        layout = QVBoxLayout(self)
        if not embedded:
            layout.addWidget(_heading("Models"))
        if banner is not None:
            self._banner = _ReadinessBanner(banner, on_banner_dismiss)
            layout.addWidget(self._banner)
        if not embedded:
            caption = QLabel(
                "Machine-global models, shared across every experiment. Built-in models "
                "download on demand (Install); imported weights are added below. "
                "Per-experiment trained models live under each experiment's Trained models."
            )
            caption.setWordWrap(True)
            layout.addWidget(caption)

        # Import **adds a row**, so it goes above the table: a create action below a list is
        # one you have to scroll past the list to reach, and the list is the thing that grows.
        # Check access / Install / Remove act on the row you have *selected*, so they stay under
        # it — they read as operating on what is above them, which they are.
        self._import_btn = QPushButton("Import model")
        self._import_btn.clicked.connect(self._import_model)
        if not builtin_only:
            creates = QHBoxLayout()
            creates.addWidget(self._import_btn)
            creates.addStretch()
            layout.addLayout(creates)

        self._table_holder = QVBoxLayout()
        layout.addLayout(self._table_holder)

        actions = QHBoxLayout()
        self._check_btn = QPushButton("Check access")
        self._check_btn.clicked.connect(self._check_access)
        self._install_btn = QPushButton("Install")
        self._install_btn.clicked.connect(self._install)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_imported)
        buttons = [self._check_btn, self._install_btn]
        if not builtin_only:  # import/remove of your models is the manager's job now
            buttons += [self._remove_btn]
        for button in buttons:
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._install_status = QLabel("")  # current file + MB (parity with the CLI)
        self._install_status.setWordWrap(True)
        self._install_status.setVisible(False)
        layout.addWidget(self._install_status)
        # The selected model's license, when its weights restrict use (SuperAnimal, §26): said
        # where Install is pressed, as the CLI says it before it downloads.
        self._license_note = QLabel("")
        self._license_note.setWordWrap(True)
        self._license_note.setVisible(False)
        layout.addWidget(self._license_note)
        if not embedded:
            layout.addStretch()
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        self._rows = surfaced_model_rows()
        if self._builtin_only:
            self._rows = [r for r in self._rows if r.source == "built-in"]
        self._installed = None  # unknown until the off-thread cache scan returns
        self._scan_failed = False
        self._render_table()
        self._scan_installed()

    def _render_table(self) -> None:
        # Suggested first, then installed, then the rest — a stable sort, so the catalog's own
        # order holds inside each band. Installed is unknown until the cache scan lands, so the
        # installed rows move up once when it does; the suggested ones never move. Sorting
        # `self._rows` itself keeps each row's stamped index pointing at the right model.
        suggested = {pair: i for i, pair in enumerate(api_models.suggested_models())}
        installed = self._installed or {}
        self._rows.sort(
            key=lambda r: (
                suggested.get((r.id, r.variant), len(suggested)),
                not installed.get((r.id, r.variant), False),
            )
        )

        def status(key: tuple[str, str | None]) -> str:
            if self._scan_failed:
                return "unknown"  # the cache scan raised — don't hang on "checking…"
            if self._installed is None:
                return "checking…"
            return "installed" if self._installed.get(key) else "not installed"

        display = [
            (
                r.id,
                r.variant or "—",
                " · ".join(r.capabilities) or r.role,
                r.architecture,
                r.source,
                status((r.id, r.variant)),
                ", ".join(r.classes) or "—",
            )
            for r in self._rows
        ]
        if self._table is not None:
            self._table.deleteLater()
        self._table = _table(self._COLUMNS, display, ids=list(range(len(self._rows))))
        if self._max_rows is not None:
            _fit_table_height(
                self._table,
                max_rows=self._max_rows,
                min_rows=min(MIN_VISIBLE_ROWS, self._max_rows),
            )
        self._table.selectionModel().currentRowChanged.connect(self._on_selection)
        self._table_holder.addWidget(self._table)
        self._update_actions()

    def _scan_installed(self) -> None:
        # is_installed stats the HF / Ultralytics caches per model — off the UI thread
        # so the page opens immediately (it shows "checking…" until the scan returns).
        rows = list(self._rows)
        self._scan_signals = _CallSignals()
        self._scan_signals.done.connect(self._on_installed_scanned)
        self._scan_signals.failed.connect(self._on_scan_failed)
        QThreadPool.globalInstance().start(
            _CallTask(
                lambda: {
                    (r.id, r.variant): api_models.is_installed(r.id, r.variant)
                    for r in rows
                },
                self._scan_signals,
            )
        )

    def _on_installed_scanned(self, installed: dict) -> None:
        self._installed = installed
        self._render_table()  # re-render with the real statuses (no re-scan)

    def _on_scan_failed(self, exc) -> None:
        # A per-model is_installed() raised (a broken cache, a probe error). Show
        # "unknown" rather than a permanent "checking…"; Install stays off (_installed is
        # still None) so we never offer to reinstall a model we couldn't verify.
        self._scan_failed = True
        self._render_table()

    def _selected(self):
        if self._table is None:
            return None
        key = row_id(self._table.currentIndex())
        return self._rows[key] if key is not None else None

    def _on_selection(self, *_args) -> None:
        self._update_actions()

    def _update_actions(self) -> None:
        row = self._selected()
        terms = row.license if row is not None else None
        self._license_note.setText(
            f"License: {terms.name}. {terms.summary} {terms.url}" if terms else ""
        )
        self._license_note.setVisible(terms is not None)
        built_in = row is not None and row.source == "built-in"
        # While the scan is in flight (_installed is None) treat models as installed
        # so Install isn't offered before we know — it re-enables on _on_installed_scanned.
        installed = (
            (self._installed or {}).get((row.id, row.variant), True) if row else True
        )
        sizing.set_available(
            self._check_btn, built_in, "Select a built-in model to check its access"
        )
        sizing.set_available(
            self._install_btn,
            built_in and not installed,
            "Already installed" if installed else "Select a built-in model to install",
        )
        sizing.set_available(
            self._remove_btn,
            row is not None and row.source == "imported",
            "Select a model you imported — built-in models are removed from disk elsewhere",
        )

    # --- check access (off-thread; one-shot) -------------------------------- #
    def _check_access(self) -> None:
        row = self._selected()
        if row is None:
            return
        sizing.set_available(self._check_btn, False, "Checking access…")
        worker = BackgroundCall(
            lambda: api_models.model_readiness(row.id, row.variant), receiver=self
        )
        worker.done.connect(self._on_readiness)
        worker.failed.connect(self._on_check_failed)
        self._check_worker = worker  # keep a ref so the thread isn't collected
        worker.start()

    def _on_readiness(self, readiness) -> None:
        self._check_worker = None
        self._update_actions()
        if readiness.ready:
            QMessageBox.information(self, "Check access", "Ready to use.")
            return
        if readiness.reason == "not_installed":
            # Access is fine — the weights just aren't downloaded yet. That is the
            # normal state for a fresh gated model, so it is information, not a
            # warning: route the user to Install rather than reading as a failure.
            QMessageBox.information(
                self, "Check access", "Access granted. Click Install to download."
            )
            return
        message = readiness.message or readiness.reason
        if readiness.action_url:
            message += f"\n\n{readiness.action_url}"
        QMessageBox.warning(self, "Check access", message)

    def _on_check_failed(self, exc: Exception) -> None:
        self._check_worker = None
        self._update_actions()
        QMessageBox.critical(self, "Check access", str(exc))

    # --- install (off-thread; streams InstallProgress) ---------------------- #
    def _install(self) -> None:
        row = self._selected()
        if row is None:
            return
        sizing.set_available(self._install_btn, False, "Installing…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate until the total is known
        self._install_status.setVisible(True)
        self._install_status.setText("starting download…")
        worker = InstallModelWorker(row.id, row.variant, receiver=self)
        worker.progress.connect(self._on_install_progress)
        worker.finished_install.connect(self._on_install_done)
        self._install_worker = worker  # keep a ref so the thread isn't collected
        worker.start()

    def _on_install_progress(self, done: int, total: int, file: str) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)
        parts = []
        if file:
            parts.append(file)
        if total > 0:
            parts.append(
                f"{done / 1e6:.1f} / {total / 1e6:.1f} MB ({100 * done / total:.0f}%)"
            )
        self._install_status.setText("  ·  ".join(parts) or "downloading…")

    def _on_install_done(self, error: str) -> None:
        self._install_worker = None
        self._progress.setVisible(False)
        self._install_status.setVisible(False)
        self._progress.setRange(0, 100)
        if error:
            QMessageBox.critical(self, "Install model", error)
        else:
            if self._banner is not None:
                self._banner.hide()
            if self._on_readiness_changed is not None:
                self._on_readiness_changed()
        self._rebuild_table()  # refresh the Status column

    # --- import / remove (machine-global catalog) --------------------------- #
    def _import_model(self) -> None:
        # The Model manager's dialog, not a bare .pt picker: one Import, both kinds (§26).
        dialog = ImportModelDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._import_btn.setEnabled(False)
        self._import_btn.setText("Importing…")
        worker = BackgroundCall(import_selected(dialog), receiver=self)
        worker.done.connect(self._on_import_done)
        worker.failed.connect(self._on_import_failed)
        self._import_worker = worker  # keep a ref so the thread isn't collected
        worker.start()

    def _reset_import_btn(self) -> None:
        self._import_btn.setText("Import model")
        self._import_btn.setEnabled(True)
        self._import_worker = None

    def _on_import_done(self, _model_id) -> None:
        self._reset_import_btn()
        self._rebuild_table()

    def _on_import_failed(self, exc: Exception) -> None:
        self._reset_import_btn()
        QMessageBox.critical(self, "Import model", str(exc))

    def _remove_imported(self) -> None:
        row = self._selected()
        if row is None or row.source != "imported":
            return
        if (
            QMessageBox.question(
                self, "Remove model", f"Remove imported model {row.id!r}?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        api_models.remove_imported_model(row.id)
        self._rebuild_table()


class _ImportSpatialDialog(Dialog):
    """Pull part of another recording's scene geometry onto this one — from a recording in
    the experiment (default) or an exported CSV (§10).

    **The mirror of `CopyToRecordingsDialog`**, and deliberately the same shape: pick what
    to bring, see what this recording will keep, commit once. It used to offer only regions
    and points, with the field of view and the calibration reachable through two other
    buttons that no longer exist — so the pull direction covered half of what the push did
    and the halves did not match. Both now read the same category table and the same
    arrival rules, which is the whole point: a setup crossing between two recordings should
    not care which end you started from.

    A category is offered only when the **source** has one and this recording can take it.
    Offering a tick that will certainly be refused is a control whose only outcome is a
    line in a report, and the report reads as a failure.
    """

    def __init__(self, recording, experiment=None, parent=None):
        super().__init__(parent)
        self._recording = recording
        self._experiment = experiment
        self.setWindowTitle(f"Import scene geometry — {recording.name}")
        self._others = (
            [r for r in experiment.recordings if r.id != recording.id]
            if experiment is not None
            else []
        )

        layout = self.body()
        layout.addWidget(_heading("From"))
        self._from_rec = QRadioButton("A recording in this experiment")
        self._rec_combo = QComboBox()
        with read_cache.pinned():
            for r in self._others:
                self._rec_combo.addItem(_recording_display_name(r), r.id)
        self._from_csv = QRadioButton("An exported CSV file")
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._from_rec.setEnabled(bool(self._others))
        self._from_rec.setChecked(bool(self._others))
        self._from_csv.setChecked(not self._others)
        self._from_rec.toggled.connect(self._on_source_changed)
        self._rec_combo.currentIndexChanged.connect(self._refresh_items)
        layout.addWidget(self._from_rec)
        layout.addWidget(self._rec_combo)
        layout.addWidget(self._from_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self._path, 1)
        csv_row.addWidget(browse)
        csv_holder = QWidget()
        csv_holder.setLayout(csv_row)
        layout.addWidget(csv_holder)

        layout.addWidget(_heading("Bring"))
        self._items = QListWidget()
        self._items.itemChanged.connect(self._sync_count)
        layout.addWidget(self._items, 1)
        self._movables: list[_Movable] = []

        layout.addWidget(
            _instructions(
                "Nothing on this recording is overwritten. Anything that cannot come "
                "across is listed with the reason and cannot be ticked."
            )
        )

        self._box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = self._box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Import")
        self._box.accepted.connect(self.accept)
        self._box.rejected.connect(self.reject)
        self.set_buttons(self._box)  # pinned outside the scroll

        self._on_source_changed()

    # -- source ---------------------------------------------------------------- #
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose spatial CSV", "", "CSV (*.csv);;All files (*)"
        )
        if path:
            self._path.setText(path)
            self._refresh_items()

    def _on_source_changed(self) -> None:
        from_rec = self._from_rec.isChecked()
        self._rec_combo.setEnabled(from_rec)
        self._path.setEnabled(not from_rec)
        self._refresh_items()

    def _source_recording(self):
        if not self._from_rec.isChecked() or self._experiment is None:
            return None
        rid = self._rec_combo.currentData()
        return self._experiment.get_recording(rid) if rid else None

    # -- what can actually come across ----------------------------------------- #
    def _blocked_for(self, kind, name):
        """Why this definition cannot land here.

        Kind-grained kinds are refused whole — this recording either has a field of view or
        it does not. Set-grained ones are refused by *name*, which is the whole reason the
        picker lists definitions: "nest" may be taken while "feeder" is free, and a
        category tick could not say so.
        """
        if kind.collision_grain == "kind":
            cat = CATEGORIES_BY_KEY[kind.id if kind.id != "arena" else "fov"]
            occupied = cat.occupied(self._recording)
            if occupied:
                return f"this recording {occupied}"
            source = self._source_recording()
            return cat.unfit(source, self._recording) if source else None
        taken = {e["name"] for e in getattr(self._recording, kind.id + "s")}
        return (
            "this recording already has one with this name" if name in taken else None
        )

    def _refresh_items(self) -> None:
        source = self._source_recording()
        self._movables = _movables(source, self._blocked_for) if source else []
        _fill_movables(self._items, self._movables)
        self._sync_count()

    def _sync_count(self) -> None:
        chosen = _checked_movables(self._items, self._movables)
        from_csv = self._from_csv.isChecked() and bool(self._path.text())
        set_available(
            self._ok,
            bool(chosen) or from_csv,
            (
                "Choose a recording or a CSV file"
                if self._source_recording() is None and not from_csv
                else "Tick at least one thing to bring"
            ),
        )

    def do_import(self) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Bring exactly what is ticked.

        Returns ``(added, flattened, refused)`` — the triple every arrival path reports, so
        one renderer serves the pull and the push.
        """
        source = self._source_recording()
        if (
            source is None
        ):  # a CSV carries the whole setup; each kind takes its own part
            return self._import_csv()
        added, flattened, refused = [], [], []
        for key, names in _names_by_kind(
            _checked_movables(self._items, self._movables)
        ).items():
            try:
                a, f, r = CATEGORIES_BY_KEY[key].give(source, self._recording, names)
            except (FileNotFoundError, ValueError) as exc:
                refused.append((CATEGORIES_BY_KEY[key].label, str(exc)))
                continue
            added += a
            flattened += f
            refused += r
        return added, flattened, refused

    def _import_csv(self) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        path = self._path.text()
        added, flattened, refused = [], [], []
        for kind_id, call in (
            ("arena", lambda: self._recording.import_field_of_view_csv(path)),
            ("roi", lambda: self._recording.import_spatial_csv(path, kinds=("roi",))),
            (
                "point",
                lambda: self._recording.import_spatial_csv(path, kinds=("point",)),
            ),
        ):
            try:
                a, f, r = call()
            except (FileNotFoundError, ValueError) as exc:
                refused.append((kind_id, str(exc)))
                continue
            added += a
            flattened += f
            refused += r
        try:
            self._recording.import_calibration_csv(path)
            added.append("distance calibration")
        except (FileNotFoundError, ValueError) as exc:
            refused.append(("distance calibration", str(exc)))
        return added, flattened, refused

    # -- what can actually be brought ------------------------------------------ #

    # -- running ---------------------------------------------------------------- #


class AddVideoPartDialog(Dialog):
    """Add a video part to a recording, running the import off the UI thread with a
    progress bar (the copy + timestamp generation is a long op — a bare synchronous call
    froze the window)."""

    def __init__(self, recording, parent=None):
        super().__init__(parent)
        self._recording = recording
        self.setWindowTitle("Add video part")
        layout = self.body()
        pick = QHBoxLayout()
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("Choose a video file…")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        pick.addWidget(self._path, 1)
        pick.addWidget(browse)
        layout.addLayout(pick)
        self._reference = _ReferenceControls()
        layout.addWidget(self._reference)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        layout.addWidget(self._status)
        layout.addStretch()
        self._box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = self._box.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Add part")
        self._box.accepted.connect(self._start)
        self._box.rejected.connect(self.reject)
        self.set_buttons(self._box)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add video part",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self._path.setText(path)

    def _start(self) -> None:
        path = self._path.text()
        if not path:
            QMessageBox.warning(self, "Add video part", "Choose a video file.")
            return
        params = self._reference.reference_params() or {}
        self._set_busy(True)

        def work(progress):
            self._recording.add_video_part(path, progress=progress, **params)

        self._worker = StageProgressWorker(work, receiver=self)
        self._worker.tick.connect(self._on_tick)
        self._worker.done.connect(lambda _r: self.accept())
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_tick(self, permille: int, status: str) -> None:
        self._progress.setValue(permille)
        self._status.setText(status)

    def _on_failed(self, exc) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Add video part", str(exc))

    def _set_busy(self, busy: bool) -> None:
        self._ok.setEnabled(not busy)
        self._path.setEnabled(not busy)
        self._reference.setEnabled(not busy)
        self._progress.setVisible(busy)
        self._status.setVisible(busy)


class RelinkVideoDialog(Dialog):
    """Point a referenced video part at a moved/renamed file (Section 2), verified by
    fingerprint. Synchronous + fast (a head/tail hash + a yaml write) — no worker thread,
    mirroring the import-field-of-view dialog; the caller runs ``do_relink()`` in a
    try/except and surfaces errors via ``QMessageBox``."""

    def __init__(self, recording, parent=None):
        super().__init__(parent)
        from detail.storage import recording as recording_store

        self._recording = recording
        self.setWindowTitle("Locate video")
        layout = self.body()
        layout.addWidget(
            _instructions(
                "A referenced video moved or was renamed. Point this part at the file's "
                "new location — it's accepted only if the content matches what was "
                "recorded (a changed video means remove and re-add the recording)."
            )
        )
        form = QFormLayout()
        self._part = QComboBox()
        for part in recording.video_parts:
            if part.get("source") is not None:  # only reference parts can be relinked
                self._part.addItem(recording_store.part_stem(part), part["index"])
        form.addRow("Part:", self._part)
        pick = QWidget()
        pick_row = QHBoxLayout(pick)
        pick_row.setContentsMargins(0, 0, 0, 0)
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("Choose the moved video…")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        pick_row.addWidget(self._path, 1)
        pick_row.addWidget(browse)
        form.addRow("New location:", pick)
        layout.addLayout(form)
        layout.addStretch()
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Relink")
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        self.set_buttons(box)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate video", "", "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        if path:
            self._path.setText(path)

    def _on_ok(self) -> None:
        if not self._path.text():
            QMessageBox.warning(self, "Locate video", "Choose the moved video file.")
            return
        if self._part.count() == 0:
            QMessageBox.warning(
                self, "Locate video", "This recording has no referenced parts."
            )
            return
        self.accept()

    def do_relink(self) -> str | None:
        """Relink the selected part (synchronous). Raises the ``relink_video_part`` errors.

        Returns the video-root alias the relink registered, or ``None``.
        """
        return self._recording.relink_video_part(
            self._part.currentData(), self._path.text()
        )


class CopyToRecordingsDialog(Dialog):
    """Push one recording's spatial setup onto a chosen set of the experiment's others
    in one action (Section 10, issue #342) — the batch form of pulling one setting
    into every recording one at a time, the other way around.

    **It never replaces.** A target that already holds a checked category keeps what it
    has, and says so on its own row *before* the copy is offered. The operation has no
    undo and spans many recordings at once, so it is additive by construction; correcting
    an existing setup is a deliberate delete on the recording that needs it. Blocking is
    per category, not per recording — a target with a field of view but no regions still
    receives the regions, and only a target where *every* checked category is taken is
    disabled outright.

    The four categories, what counts as "already has one", and how a copy is handed over
    all come from ``api.spatial_categories``; this dialog renders that table and owns no
    opinion about it. `run` is handed a `plan` read on the UI thread — a worker must not
    touch a widget, and this dialog is parented to the page, so navigating away mid-copy
    would otherwise pull its C++ side out from under a running thread.
    """

    @classmethod
    def has_anything_to_copy(cls, recording) -> bool:
        """Whether at least one category has data on ``recording`` — gates the button
        that opens this dialog (nothing to open it for otherwise)."""
        with read_cache.pinned():
            return any(c.has_data(recording) for c in SPATIAL_CATEGORIES)

    def __init__(self, recording, experiment, parent=None):
        super().__init__(parent)
        self._recording = recording
        self._experiment = experiment
        self._others = [r for r in experiment.recordings if r.id != recording.id]
        #: Target ids the *user* turned off, kept apart from the ones a full block turns
        #: off. Without the split, toggling a category rebuilds every row's check state
        #: and silently rearms a selection somebody had narrowed on purpose — on an
        #: action that writes to every checked recording.
        self._unchecked: set[str] = set()
        self.setWindowTitle(f"Copy to other recordings from {recording.name}")

        layout = self.body()
        layout.addWidget(_heading("Copy"))
        self._items = QListWidget()
        self._items.itemChanged.connect(self._on_item_ticked)
        layout.addWidget(self._items, 1)
        self._movables: list[_Movable] = []

        layout.addWidget(_heading("To"))
        all_row = QWidget()
        all_row_layout = QHBoxLayout(all_row)
        all_row_layout.setContentsMargins(0, 0, 0, 0)
        self._all = QCheckBox(
            f"All {len(self._others)} other recording"
            f"{'' if len(self._others) == 1 else 's'}"
        )
        self._all.setChecked(True)
        self._all.toggled.connect(self._toggle_all)
        all_row_layout.addWidget(self._all)
        all_row_layout.addStretch()
        uncheck_all = QPushButton("Uncheck all")
        uncheck_all.clicked.connect(lambda: self._all.setChecked(False))
        all_row_layout.addWidget(uncheck_all)
        layout.addWidget(all_row)

        self._targets = QListWidget()
        for r in self._others:
            item = QListWidgetItem(_recording_display_name(r))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(Qt.ItemDataRole.UserRole, r.id)
            self._targets.addItem(item)
        self._targets.itemChanged.connect(self._on_item_changed)
        # A row carries the reason it is blocked, which is longer than the list is wide.
        # Left to scroll horizontally the bar eats a row of height (measured: 6 targets
        # rendered as 4 + a scrollbar), and the reason is in the tooltip anyway.
        self._targets.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._targets.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # The list is what this dialog is *for*, so it takes the slack rather than being
        # capped to its rows with the remainder left dead at the bottom — `finish_dialog`
        # is then the no-op it is meant to be on a dialog whose content fills.
        layout.addWidget(
            RecordingFilter(self._targets, experiment, select_buttons=True)
        )
        layout.addWidget(self._targets, 1)

        layout.addWidget(
            _instructions(
                "A recording that already has one keeps it — nothing here is "
                "overwritten. Copied setups are a starting point; open each recording "
                "afterward to check it."
            )
        )

        self._box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = self._box.button(QDialogButtonBox.StandardButton.Ok)
        self._box.accepted.connect(self.accept)
        self._box.rejected.connect(self.reject)
        self.set_buttons(self._box)  # pinned outside the scroll

        # After the target list exists: what a definition is judged against is the set of
        # chosen targets, so the two refresh together.
        self._refresh_items()
        self._refresh_targets()

    # -- what can actually be copied ----------------------------------------- #

    # -- what can actually be copied ------------------------------------------ #
    def _blocked_for(self, kind, name):
        """Why this definition cannot go to *any* chosen target.

        Per target it may be refused and still worth offering — "nest" can be taken on
        three recordings and free on four. It is only withdrawn when no checked target
        could take it, which is when a tick could produce nothing but refusals.
        """
        targets = [
            r for r in self._others if r.id not in self._unchecked
        ] or self._others
        if kind.collision_grain == "kind":
            cat = CATEGORIES_BY_KEY[kind.id if kind.id != "arena" else "fov"]
            empty = [r for r in targets if cat.occupied(r) is None]
            if not empty:
                return "every chosen recording already has one"
            if not any(cat.unfit(self._recording, r) is None for r in empty):
                return (
                    "different video shape — every chosen recording without one is "
                    "another rig, which needs its own calibration"
                )
            return None
        free = [
            r
            for r in targets
            if name not in {e["name"] for e in getattr(r, kind.id + "s")}
        ]
        if not free:
            return "every chosen recording already has one with this name"
        return None

    def _refresh_items(self) -> None:
        self._movables = _movables(self._recording, self._blocked_for)
        _fill_movables(self._items, self._movables)

    def _on_item_ticked(self, _item) -> None:
        self._refresh_targets()

    # -- selection ----------------------------------------------------------- #
    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Record the user's own intent for this row, so a later category toggle can put
        it back rather than overwriting it with a fresh default."""
        rid = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self._unchecked.discard(rid)
        else:
            self._unchecked.add(rid)
        self._sync_count()

    def _toggle_all(self, checked: bool) -> None:
        self._unchecked = set() if checked else {r.id for r in self._others}
        self._refresh_targets()

    def _refresh_targets(self) -> None:
        """Redraw every target row for what is currently ticked.

        A row is disabled only when *every* ticked definition is already on it — there is
        then nothing the copy could do. A row that is merely partly blocked stays checkable
        and says what it will skip, because the rest still travels.
        """
        chosen = _checked_movables(self._items, self._movables)
        self._targets.blockSignals(True)
        with read_cache.pinned():
            for i, r in enumerate(self._others):
                item = self._targets.item(i)
                blocked, unfit = self._blocked_on(r, chosen)
                name = _recording_display_name(r)
                dead = bool(chosen) and len(blocked) + len(unfit) == len(chosen)
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEnabled
                    if dead
                    else item.flags() | Qt.ItemFlag.ItemIsEnabled
                )
                if blocked or unfit:
                    # The row names what is left behind in a few words — the full
                    # reason is a sentence, and elided it loses the part that matters.
                    held = [f"already has: {', '.join(blocked)}"] if blocked else []
                    short = held + (
                        [f"different video shape: {', '.join(lb for lb, _ in unfit)}"]
                        if unfit
                        else []
                    )
                    full = held + [f"{label}: {reason}" for label, reason in unfit]
                    item.setText(f"{name}  —  {'; '.join(short)}")
                    item.setToolTip(
                        "\n".join(full)
                        + "\nThese are left as they are; anything else ticked still "
                        "arrives."
                    )
                else:
                    item.setText(name)
                    item.setToolTip("")
                item.setCheckState(
                    Qt.CheckState.Unchecked
                    if dead or r.id in self._unchecked
                    else Qt.CheckState.Checked
                )
        self._targets.blockSignals(False)
        self._sync_count()

    def _blocked_on(
        self, target, chosen: list[_Movable]
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """``(held, unfit)``: the ticked definitions ``target`` already holds, and the
        ``(label, reason)`` of those that do not fit it (a calibration onto another rig's
        video shape)."""
        held: list[str] = []
        unfit: list[tuple[str, str]] = []
        for m in chosen:
            if m.kind.collision_grain == "kind":
                cat = CATEGORIES_BY_KEY[m.kind.id if m.kind.id != "arena" else "fov"]
                if cat.occupied(target) is not None:
                    held.append(m.kind.label)
                elif (reason := cat.unfit(self._recording, target)) is not None:
                    unfit.append((m.kind.label, reason))
            elif m.name in {e["name"] for e in getattr(target, m.kind.id + "s")}:
                held.append(m.name)
        return held, unfit

    def _checked_target_ids(self) -> list[str]:
        return [
            self._targets.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._targets.count())
            if (self._targets.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
            and self._targets.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _sync_count(self) -> None:
        n = len(self._checked_target_ids())
        chosen = _checked_movables(self._items, self._movables)
        self._ok.setText(f"Copy to {n}" if n else "Copy")
        set_available(
            self._ok,
            bool(n and chosen),
            (
                "Tick at least one thing to copy"
                if not chosen
                else "Tick at least one recording to copy to"
            ),
        )

    # -- running ------------------------------------------------------------- #
    def plan(self) -> tuple[dict[str, list[str]], list[str]]:
        """``({category key: [definition names]}, target ids)`` read on the UI thread.

        `run` executes in a worker, and a worker may not touch a Qt widget: this dialog is
        parented to the page that opened it, so a navigation mid-copy destroys its C++
        side while the thread is still reading check states off it.
        """
        return (
            _names_by_kind(_checked_movables(self._items, self._movables)),
            self._checked_target_ids(),
        )

    def run(
        self,
        plan: tuple[dict[str, list[str]], list[str]],
        progress: Callable[[str, float, str], None] | None,
    ) -> list[tuple[str, str, str]]:
        """Copy every planned category onto every planned target, off the UI thread.

        Categories run in a fixed order, each getting an even slice of ``progress``.
        Returns ``(category, subject, note)`` triples for everything the user should be
        told — a recording that kept what it had, and a definition that could not travel.
        """
        by_kind, target_ids = plan
        keys = list(by_kind)
        total = len(keys)
        notes: list[tuple[str, str, str]] = []
        for index, key in enumerate(keys):
            cat = CATEGORIES_BY_KEY[key]

            def sub_progress(stage, fraction, detail, index=index, label=cat.label):
                if progress is not None:
                    progress(label, (index + fraction) / total, detail)

            result = self._experiment.propagate_spatial(
                key,
                self._recording.id,
                targets=target_ids,
                names=by_kind[key],
                progress=sub_progress,
            )
            notes.extend((cat.label, ref, err) for ref, err in result.skipped)
            notes.extend((cat.label, name, reason) for name, reason in result.refused)
        return notes


# Detail views (content area, reached by drilling in)
# --------------------------------------------------------------------------- #
class RecordingDetailPage(QWidget):
    """One recording: video parts, field of view, run history (read-only for slice 1)."""

    def __init__(
        self,
        recording,
        on_open_run=None,
        on_deleted=None,
        experiment=None,
        on_open_recording_run=None,
    ):
        super().__init__()
        self._recording = recording
        self._on_open_run = on_open_run
        self._on_open_recording_run = on_open_recording_run
        self._on_deleted = on_deleted
        self._experiment = experiment
        layout = QVBoxLayout(self)
        self._title = _heading(f"Recording: {recording.name}")
        layout.addWidget(self._title)

        form = QFormLayout()
        self._name = QLineEdit(recording.name)
        form.addRow("Name:", self._name)
        # The id is the tool's and never changes; it is shown because run records, the
        # CLI and exports still speak it, so you need somewhere to read it off.
        recording_id = QLabel(recording.id)
        recording_id.setFont(MONO)
        form.addRow("Id:", recording_id)
        sources = [s for s in recording.video_sources if s]
        if sources:
            source = QLabel("\n".join(sources))
            source.setWordWrap(True)
            source.setFont(MONO)
            form.addRow("Source:" if len(sources) == 1 else "Sources:", source)
        layout.addLayout(form)
        save_row = QHBoxLayout()
        save = QPushButton("Save name")
        save.clicked.connect(self._save_name)
        save_row.addStretch()
        save_row.addWidget(save)
        layout.addLayout(save_row)

        if experiment is not None:
            self._notes_box = NotesBox(experiment, "recording", recording.id)
            layout.addWidget(self._notes_box)
            self._build_metadata(layout)

        layout.addWidget(_band_caption("DATA PIPELINE"))

        self._parts_heading = _heading("")
        layout.addWidget(self._parts_heading)
        self._video_status_lbl = QLabel(
            ""
        )  # referenced-video integrity, filled off-thread
        self._video_status_lbl.setWordWrap(True)
        self._video_status_lbl.setVisible(False)
        layout.addWidget(self._video_status_lbl)
        # Above the parts table, like every other create action: the list is what grows, so a
        # button under it is one you scroll past the list to reach.
        add_row = QHBoxLayout()
        add_part = QPushButton("Add video part")
        add_part.clicked.connect(self._add_part)
        self._locate_btn = QPushButton("Locate video")
        self._locate_btn.clicked.connect(self._locate_video)
        # Shown only for referenced parts (a copy-only recording never surfaces the concept),
        # and enabled only when a referenced video is actually missing — the off-thread scan
        # (_on_video) flips it. Relinking a present video does nothing, so it stays greyed.
        self._locate_btn.setVisible(
            any(p.get("source") is not None for p in recording.video_parts)
        )
        self._locate_btn.setEnabled(False)
        self._locate_btn.setToolTip("No referenced video needs locating.")
        self._move_up = QPushButton("Move up")
        self._move_up.clicked.connect(lambda: self._move_part(-1))
        self._move_down = QPushButton("Move down")
        self._move_down.clicked.connect(lambda: self._move_part(1))
        n_runs = (
            len(recording.detection_runs)
            + len(recording.tracking_runs)
            + len(recording.embedding_runs)
        )
        if n_runs:  # order is fixed once anything has been computed against it
            for button in (self._move_up, self._move_down):
                button.setEnabled(False)
                button.setToolTip("Reorder before running any processing")
        add_row.addWidget(add_part)
        add_row.addWidget(self._locate_btn)
        add_row.addWidget(self._move_up)
        add_row.addWidget(self._move_down)
        add_row.addStretch()
        layout.addLayout(add_row)
        self._parts_holder = QVBoxLayout()
        layout.addLayout(self._parts_holder)
        self._parts_table = None
        self._video_signals = None
        self._rebuild_parts()
        self._scan_video()

        # One section for all four spatial kinds. It sits under DATA PIPELINE because the
        # field of view is a processing input; the per-kind scope note carries what the
        # old split across DATA PIPELINE / ANALYSIS was saying, which was a fact about the
        # kind rather than about which heading its widget happened to land under.
        self._spatial_holder = QVBoxLayout()
        layout.addLayout(self._spatial_holder)
        self._spatial_widgets = []
        self._rebuild_scene_geometry()

        layout.addWidget(_band_caption("ANALYSIS"))

        self._streams_holder = QVBoxLayout()
        layout.addLayout(self._streams_holder)
        self._stream_widgets = []
        self._streams_table = None
        self._stream_names = []
        self._sync_btn: QPushButton | None = None
        self._edit_offset_btn: QPushButton | None = None
        self._rebuild_streams()

        # Run history sits at the very bottom: it only becomes relevant once a
        # recording has runs, so it stays out of the way until then.
        self._build_run_history(layout)

        layout.addStretch()

        if self._experiment is not None:
            delete = QPushButton("Delete recording")
            delete.clicked.connect(self._delete_recording)
            layout.addWidget(delete, alignment=Qt.AlignmentFlag.AlignLeft)

    def _build_run_history(self, layout) -> None:
        """Detection/tracking/embedding runs (pipeline outputs) — pinned to the very
        bottom of the page (it only matters once runs exist)."""
        layout.addWidget(_heading("Run history"))
        recording = self._recording
        records = (
            [("detection", r) for r in recording.detection_runs]
            + [("tracking", r) for r in recording.tracking_runs]
            + [("embedding", r) for r in recording.embedding_runs]
        )
        # Shown by the off-thread video scan (_on_video) when the referenced video changed
        # or went missing: every run below was computed against it, so results may be stale.
        self._runs_stale_lbl = QLabel(
            "These runs were computed on this video, which has since changed or gone "
            "missing — their results may be stale."
        )
        self._runs_stale_lbl.setWordWrap(True)
        set_label_color(self._runs_stale_lbl, WARN)
        self._runs_stale_lbl.setVisible(False)
        self._has_runs = bool(records)
        layout.addWidget(self._runs_stale_lbl)
        history = [
            (
                stage,
                r["id"],
                r.get("processing_run_id") or "—",
                run_cost.compute_label(r.get("compute")),
                r.get("status"),
            )
            for stage, r in records
        ]
        # double-click a row to open its result view: a experiment-backed run opens the
        # experiment run that produced it (run_NNN); a per-recording-only run (CLI/API,
        # no experiment record) opens a recording-scoped result view directly.
        self._run_targets = [r.get("processing_run_id") for _, r in records]
        self._run_rows = [(stage, r["id"]) for stage, r in records]
        runs_table = IndexTable(
            ["Stage", "Id", "From run", "Where", "Status"],
            empty_text="(nothing has been run on this recording yet)",
        )
        # Identity is the row's original index into `records` — both `_run_targets` and
        # `_run_rows` are parallel to it, so one stamped key resolves either.
        runs_table.set_rows(history, ids=list(range(len(records))))
        runs_table.view.doubleClicked.connect(self._open_run_row)
        layout.addWidget(runs_table)

    def _open_run_row(self, index) -> None:
        key = row_id(index)
        if key is None:
            return
        target = self._run_targets[key]
        if target and self._on_open_run is not None:
            self._on_open_run(target)
        elif self._on_open_recording_run is not None:
            stage, run_id = self._run_rows[key]
            self._on_open_recording_run(self._recording.id, stage, run_id)

    def _delete_recording(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete recording",
            f"Delete recording {self._recording.id} and all its runs, field-of-view "
            "regions, and imported video? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._experiment.delete_recording(self._recording.id)
        if self._on_deleted is not None:
            self._on_deleted()

    def _save_name(self) -> None:
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Recording", "Name is required.")
            return
        try:
            self._recording.set_name(name)
        except ValueError as exc:  # a name another recording already holds
            QMessageBox.warning(self, "Recording", str(exc))
            return
        self._title.setText(f"Recording: {name}")
        QMessageBox.information(self, "Recording", "Saved.")

    def _build_metadata(self, layout) -> None:
        """This recording's grouping-variable row (§11), editable inline. Columns are
        managed experiment-wide; adding one here adds it everywhere."""
        layout.addWidget(_heading("Recording information"))
        self._meta_holder = QVBoxLayout()
        layout.addLayout(self._meta_holder)
        self._meta_widgets: list = []
        self._rebuild_metadata()

    def _rebuild_metadata(self) -> None:
        """The fields for this recording's row, rebuilt when a column is added.

        Columns are experiment-wide, so adding one here adds it everywhere (blank on the
        other recordings) — the table is one shape, not one per recording.
        """
        for widget in self._meta_widgets:
            discard(widget)
        self._meta_widgets = []
        cols = self._experiment.metadata_columns
        values = self._experiment.recording_metadata.get(self._recording.id, {})
        self._meta_edits = {}
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        if not cols:
            box.addWidget(
                _instructions(
                    "No information columns yet. Add one here (or on the Recording "
                    "information page) to record what varies across your recordings — "
                    "treatment, day, animal id."
                )
            )
        else:
            form = QFormLayout()
            for col in cols:
                edit = QLineEdit(values.get(col, ""))
                self._meta_edits[col] = edit
                form.addRow(f"{col}:", edit)
            box.addLayout(form)
        buttons = QHBoxLayout()
        add_column = QPushButton("Add column")
        add_column.clicked.connect(self._add_metadata_column)
        buttons.addWidget(add_column)
        buttons.addStretch()
        if cols:
            save = QPushButton("Save metadata")
            save.clicked.connect(self._save_metadata)
            buttons.addWidget(save)
        box.addLayout(buttons)
        self._meta_holder.addWidget(holder)
        self._meta_widgets.append(holder)

    def _add_metadata_column(self) -> None:
        existing = self._experiment.metadata_columns
        hint = (
            "  Existing: " + ", ".join(existing) if existing else ""
        )  # discourage near-duplicates (treatment vs Treatment)
        name, ok = QInputDialog.getText(self, "Add column", f"Column name:{hint}")
        name = name.strip()
        if not ok or not name:
            return
        try:
            self._experiment.add_metadata_column(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Add column", str(exc))
            return
        self._rebuild_metadata()

    def _save_metadata(self) -> None:
        self._experiment.set_recording_metadata(
            self._recording.id, {c: e.text() for c, e in self._meta_edits.items()}
        )
        QMessageBox.information(self, "Recording", "Metadata saved.")

    # -- unsaved edits (gui/editing.py) --------------------------------------
    def unsaved_summary(self) -> str | None:
        """The name field and the metadata form both have a Save button people miss."""
        pending = []
        if self._name.text().strip() not in ("", self._recording.name):
            pending.append("the name")
        stored = (
            self._experiment.recording_metadata.get(self._recording.id, {})
            if self._experiment is not None
            else {}
        )
        changed = sum(
            1
            for col, edit in getattr(self, "_meta_edits", {}).items()
            if edit.text() != stored.get(col, "")
        )
        if changed:
            pending.append(
                f"{changed} recording-information field{'s' if changed != 1 else ''}"
            )
        if not pending:
            return None
        return (
            f"{self._recording.name} has unsaved changes to "
            + " and ".join(pending)
            + "."
        )

    def save_edits(self) -> None:
        if self._name.text().strip() not in ("", self._recording.name):
            self._recording.set_name(self._name.text().strip())  # raises on a duplicate
            self._title.setText(f"Recording: {self._recording.name}")
        if getattr(self, "_meta_edits", None) and self._experiment is not None:
            self._experiment.set_recording_metadata(
                self._recording.id, {c: e.text() for c, e in self._meta_edits.items()}
            )

    def discard_edits(self) -> None:
        self._name.setText(self._recording.name)
        stored = (
            self._experiment.recording_metadata.get(self._recording.id, {})
            if self._experiment is not None
            else {}
        )
        for col, edit in getattr(self, "_meta_edits", {}).items():
            edit.setText(stored.get(col, ""))

    # -- preview spatial setup (§10, issue #342 §C) --------------------------- #
    # "Apply-to-all is a starting point, not a claim of correctness" — this is the
    # cheap way to see which of the recordings it was pushed onto came out wrong.
    # One read-only dialog covers all four kinds, reached the same way as Copy.

    def _rebuild_scene_geometry(self) -> None:
        for widget in self._spatial_widgets:
            discard(widget)
        self._spatial_widgets = []

        def add(widget) -> None:
            self._spatial_holder.addWidget(widget)
            self._spatial_widgets.append(widget)

        add(_heading("Scene geometry"))

        from detail.gui.arena import GeometryPreview

        # Left: the four kinds, as text. Right: a picture of them, and the one button
        # that edits any of it — reading the picture is what tells you whether you need
        # the button at all. The pin covers this synchronous burst only; the preview reads
        # its shapes in the same background call that decodes its frame.
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        left = QVBoxLayout()
        right = QVBoxLayout()

        with read_cache.pinned():
            view_label = QLabel(f"Camera view: {self._recording.camera_view.described}")
            view_label.setContentsMargins(_SCENE_INDENT, 0, 0, 0)
            left.addWidget(view_label)
            # The four kinds are sub-sections of this one, and the style guide forbids a
            # custom point size, so the hierarchy is carried by an indent — otherwise the
            # section heading and its four kind headings are five identical bold lines.
            for kind, entries in self._scene_entries().items():
                for widget in self._scene_kind_widgets(kind, entries):
                    widget.setContentsMargins(_SCENE_INDENT, 0, 0, 0)
                    left.addWidget(widget)

            right.addWidget(
                _heading("Geometry preview"), alignment=Qt.AlignmentFlag.AlignHCenter
            )
            right.addWidget(
                GeometryPreview(self._recording),
                alignment=Qt.AlignmentFlag.AlignHCenter,
            )
            open_button = QPushButton("Edit scene geometry…")
            open_button.clicked.connect(self._open_scene_geometry)
            set_available(
                open_button,
                bool(self._recording.video_parts),
                "Import a video part first",
            )
            right.addWidget(open_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        columns.addLayout(left, 3)
        columns.addLayout(right, 2)
        holder = QWidget()
        holder.setLayout(columns)
        add(holder)

    def _scene_entries(self) -> dict:
        """``{SpatialKind: [entry, ...]}`` in the table's order.

        The calibration is not a list on the recording, so it is wrapped in one here rather
        than given its own branch in the renderer — the point of the table is that this
        page holds no per-kind special case.
        """
        cal = self._recording.distance_calibration
        by_id = {
            "arena": self._recording.arenas,
            "calibration": [cal] if cal is not None else [],
            "roi": self._recording.rois,
            "point": self._recording.points,
        }
        return {kind: by_id[kind.id] for kind in SPATIAL_KINDS}

    def _scene_kind_widgets(self, kind, entries: list) -> list:
        """A kind's block: heading with its count and scope, its description, its rows.

        The description is the sentence each of the four sections used to carry; losing it
        was the cost of merging them and it is not worth paying. The scope note
        ("used by detection & tracking" / "analysis only") is what the DATA PIPELINE /
        ANALYSIS split said, which is a fact about the kind rather than about which heading
        its widget landed under — which is why it survives the merge and the split did not.
        """
        widgets = []
        title_holder = QWidget()
        title_row = QHBoxLayout(title_holder)
        title_row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(
            kind.label if kind.singleton else f"{kind.label} ({len(entries)})"
        )
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        title_row.addWidget(label)
        title_row.addStretch()
        scope = QLabel(f"— {kind.scope}")
        set_label_color(scope, MUTED)
        title_row.addWidget(scope)
        widgets.append(title_holder)
        widgets.append(_instructions(kind.description))

        if not entries:
            widgets.append(_instructions("(none yet)"))
            return widgets
        for i, entry in enumerate(entries):
            row_holder = QWidget()
            row = QHBoxLayout(row_holder)
            row.setContentsMargins(0, 0, 0, 0)
            swatch = QLabel("■")
            set_label_color(swatch, self._scene_color(kind, i))
            row.addWidget(swatch)
            detail = QLabel(self._scene_entry_text(kind, entry))
            detail.setFont(MONO)
            row.addWidget(detail, 1)
            widgets.append(row_holder)
        return widgets

    def _scene_color(self, kind, index: int) -> QColor:
        """The same allocation the editor and the geometry preview use, so a swatch
        here, the outline in the editor, and the static preview all draw one claim."""
        from detail.gui.arena import region_color

        return region_color(kind, index)

    def _scene_entry_text(self, kind, entry) -> str:
        if kind.id == "calibration":
            refusal = self._recording.calibration_refusal()
            if refusal is not None:
                return f"unusable: {refusal}"
            text = f"{self._recording.pixels_per_mm:.4g} px/mm"
            if entry.get("distance_mm"):
                text += f"  ·  {entry['distance_mm']:.0f} mm reference"
            return text
        shape = "point" if kind.max_points == 1 else f"{entry['n_vertices']} verts"
        active = entry.get("active")
        window = (
            f"frames {active['start']}–{active['end']}" if active else "whole video"
        )
        return f"{entry['id']} · {entry['name']} · {shape} · {window}"

    def _open_scene_geometry(self) -> None:
        if not self._arena_video_available():
            return
        from detail.gui.arena import SceneGeometryDialog

        SceneGeometryDialog(self._recording, self._experiment, parent=self).exec()
        self._rebuild_scene_geometry()

    def _arena_video_available(self) -> bool:
        """Arena drawing needs the source video as a backdrop — absent in a
        video-excluded export."""
        if self._recording.video_available:
            return True
        QMessageBox.information(
            self,
            "Arena",
            "Drawing an arena needs the source video, which isn't in this export "
            "(it's still on the source machine).",
        )
        return False

    def _rebuild_streams(self) -> None:
        for widget in self._stream_widgets:
            discard(widget)
        self._stream_widgets = []
        self._streams_table = None
        streams = self._recording.streams
        self._stream_names = [s["name"] for s in streams]
        heading = _heading(f"Streams ({len(streams)})")
        self._streams_holder.addWidget(heading)
        self._stream_widgets.append(heading)

        self._sync_btn = None
        self._edit_offset_btn = None
        if streams:
            rows = []
            for s in streams:
                n_events = len(self._recording.stream_events(s["name"]))
                offset_s = (s.get("alignment") or {}).get("offset_ns", 0) / 1e9
                kind = s.get("role") or s.get("shape") or s.get("type", "")
                rows.append(
                    (
                        s["name"],
                        kind,
                        n_events,
                        # Formatted to 3dp *and* keyed on the number: as a bare string this
                        # column sorted "10.000" before "9.000".
                        SortCell(f"{offset_s:.3f}", offset_s),
                    )
                )
            table = _table(["Name", "Type", "Rows", "Offset (s)"], rows)
            self._streams_holder.addWidget(table)
            self._stream_widgets.append(table)
            self._streams_table = table

        row_holder = QWidget()
        row = QHBoxLayout(row_holder)
        row.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("Add stream")
        add.clicked.connect(self._add_stream)
        row.addWidget(add)
        if streams:
            preview = QPushButton("Preview")
            preview.clicked.connect(self._preview_stream)
            self._sync_btn = QPushButton("Sync to video")
            self._sync_btn.clicked.connect(self._sync_stream)
            self._edit_offset_btn = QPushButton("Edit offset")
            self._edit_offset_btn.clicked.connect(self._edit_offset)
            delete = QPushButton("Delete stream")
            delete.clicked.connect(self._delete_stream)
            for button in (preview, self._sync_btn, self._edit_offset_btn, delete):
                row.addWidget(button)
            # Alignment applies only to an external-clock stream; a video-anchored
            # stream (behavior labels) is already aligned, so gate Sync/Edit-offset
            # on the selected stream's clock rather than showing them unconditionally.
            if self._streams_table is not None:
                self._streams_table.selectionModel().currentRowChanged.connect(
                    lambda *_: self._update_alignment_buttons()
                )
                if len(self._stream_names) == 1:
                    self._streams_table.selectRow(0)
            self._update_alignment_buttons()
        row.addStretch()
        self._streams_holder.addWidget(row_holder)
        self._stream_widgets.append(row_holder)

    def _current_stream_name(self) -> str | None:
        """The selected stream (or the sole stream), without prompting — for gating."""
        if self._streams_table is None:
            return None
        idx = self._streams_table.currentIndex().row()
        if 0 <= idx < len(self._stream_names):
            return self._stream_names[idx]
        if len(self._stream_names) == 1:
            return self._stream_names[0]
        return None

    def _update_alignment_buttons(self) -> None:
        """Enable Sync/Edit-offset only for a stream whose clock needs alignment.

        A ``video_frames`` / ``video_seconds`` stream (behavior labels) is already on
        the video clock, so both controls are disabled with a reason; Sync also needs
        a video part to scrub. Runs on selection change and after a rebuild.
        """
        if self._sync_btn is None or self._edit_offset_btn is None:
            return
        name = self._current_stream_name()
        manifest = next(
            (s for s in self._recording.streams if s.get("name") == name), None
        )
        clock = (manifest or {}).get("clock", "stream_seconds")
        aligns = manifest is not None and clocks.needs_alignment(clock)
        has_video = bool(self._recording.video_parts)
        if not aligns:
            reason = "This stream is on the video clock — no alignment needed."
            self._sync_btn.setEnabled(False)
            self._edit_offset_btn.setEnabled(False)
            self._sync_btn.setToolTip(reason if manifest is not None else "")
            self._edit_offset_btn.setToolTip(reason if manifest is not None else "")
            return
        self._edit_offset_btn.setEnabled(True)
        self._edit_offset_btn.setToolTip("")
        self._sync_btn.setEnabled(has_video)
        self._sync_btn.setToolTip("" if has_video else "Import a video part first")

    def _selected_stream(self) -> str | None:
        if self._streams_table is None:
            return None
        idx = self._streams_table.currentIndex().row()
        if 0 <= idx < len(self._stream_names):
            return self._stream_names[idx]
        if len(self._stream_names) == 1:  # single stream — no need to select first
            return self._stream_names[0]
        QMessageBox.information(self, "Streams", "Select a stream first.")
        return None

    def _add_stream(self) -> None:
        from detail.gui.streams import AddStreamDialog

        dialog = AddStreamDialog(self._recording, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._rebuild_streams()

    def _preview_stream(self) -> None:
        from detail.gui.streams import StreamPreviewDialog

        name = self._selected_stream()
        if name is None:
            return
        StreamPreviewDialog(self._recording, name, parent=self).exec()

    def _sync_stream(self) -> None:
        from detail.gui.streams import SyncStreamDialog

        name = self._selected_stream()
        if name is None:
            return
        dialog = SyncStreamDialog(self._recording, name, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._rebuild_streams()

    def _edit_offset(self) -> None:
        name = self._selected_stream()
        if name is None:
            return
        current = next((s for s in self._recording.streams if s["name"] == name), {})
        current_s = (current.get("alignment") or {}).get("offset_ns", 0) / 1e9
        value, ok = QInputDialog.getDouble(
            self, "Edit offset", "Offset (seconds):", current_s, -1e6, 1e6, 3
        )
        if not ok:
            return
        self._recording.set_stream_offset(name, round(value * 1e9))
        self._rebuild_streams()

    def _delete_stream(self) -> None:
        name = self._selected_stream()
        if name is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete stream",
            f"Delete stream {name!r} and its imported + canonical files? "
            "This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._recording.delete_stream(name)
        self._rebuild_streams()

    def _rebuild_parts(self) -> None:
        parts = self._recording.video_parts
        self._parts_heading.setText(f"Video parts ({len(parts)})")
        rows = [(p["index"], _part_location(self._recording.dir, p)) for p in parts]
        table = _table(["#", "Location"], rows)
        if self._parts_table is not None:
            self._parts_table.deleteLater()
        self._parts_holder.addWidget(table)
        self._parts_table = table

    def _scan_video(self) -> None:
        # verify_videos stats + head/tail-hashes this recording's parts — off the UI thread.
        self._video_signals = _CallSignals()
        self._video_signals.done.connect(self._on_video)
        self._video_signals.failed.connect(lambda _e: None)  # best-effort
        verify = self._recording.verify_videos
        QThreadPool.globalInstance().start(_CallTask(verify, self._video_signals))

    def _on_video(self, issues) -> None:
        stale = bool(issues) and getattr(self, "_has_runs", False)
        if getattr(self, "_runs_stale_lbl", None) is not None:
            self._runs_stale_lbl.setVisible(stale)
        kinds = {i.kind for i in issues}
        # Locate only helps a *missing* (moved/renamed) referenced file — relink refuses a
        # changed file, and a present one needs nothing. Enable it only when it can act.
        missing = bool(kinds & {"missing_file", "missing_root"})
        self._locate_btn.setEnabled(missing)
        self._locate_btn.setToolTip(
            "Point the moved video at its new file."
            if missing
            else "No referenced video needs locating."
        )
        if not issues:
            self._video_status_lbl.setVisible(False)
            return
        if missing:
            msg, color = (
                "A referenced video is missing — Locate it, or re-point its video root.",
                ERROR,
            )
        else:
            msg, color = (
                "A referenced video changed since it was added — remove and re-add the "
                "recording to use the new file.",
                WARN,
            )
        self._video_status_lbl.setText(msg)
        set_label_color(self._video_status_lbl, color)
        self._video_status_lbl.setVisible(True)

    def _locate_video(self) -> None:
        dialog = RelinkVideoDialog(self._recording, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            registered = dialog.do_relink()
        except (FileNotFoundError, IndexError, ValueError) as exc:
            QMessageBox.critical(self, "Locate video", str(exc))
            return
        self._parts_changed()
        self._scan_video()
        if registered is not None:
            QMessageBox.information(
                self,
                "Locate video",
                f"The folder holding this video is now remembered on this computer as "
                f"'{registered}'. Other recordings whose videos are in it will find "
                "them too.",
            )

    def _parts_changed(self) -> None:
        """Rebuild everything on this page that is derived from the video parts.

        The one entry point for an action that adds, relinks or reorders a part. The scene
        geometry section reads the parts too — its preview draws part 0's first frame and
        its editor button is greyed until a part exists — and rebuilding only the parts
        table left it saying "No video imported" under a table listing one.
        """
        self._rebuild_parts()
        self._rebuild_scene_geometry()

    def _add_part(self) -> None:
        dialog = AddVideoPartDialog(self._recording, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._parts_changed()
            self._locate_btn.setVisible(
                any(p.get("source") is not None for p in self._recording.video_parts)
            )
            self._locate_btn.setEnabled(
                False
            )  # re-enabled by the rescan if a video is missing
            self._scan_video()

    def _move_part(self, delta: int) -> None:
        row = self._parts_table.currentIndex().row()
        n = len(self._recording.video_parts)
        if row < 0 or not (0 <= row + delta < n):
            return
        order = list(range(n))
        order[row], order[row + delta] = order[row + delta], order[row]
        try:
            self._recording.reorder_video_parts(order)
        except ValueError as e:
            QMessageBox.warning(self, "Reorder video parts", str(e))
            return
        self._parts_changed()
        self._parts_table.selectRow(row + delta)


def _run_cadence(experiment, record: dict) -> tuple[str, str]:
    """``(frames, lineage)`` for a processing run's detail header.

    ``frames`` is the run's own cadence with the rate it lands at; ``lineage`` is the
    chain it was built on, each ancestor carrying its own cadence — so where the thinning
    happened is visible without opening three pages. Both are best-effort: a run whose
    recording has since been deleted just shows less.
    """
    from detail.api import run_spec
    from detail.gui.modules._every_n import source_rate, summary

    every_n = int((record.get("config") or {}).get("every_n_frames") or 1)
    per = record.get("per_recording") or {}
    recordings = []
    for sid in per:
        try:
            recordings.append(experiment.get_recording(sid))
        except (FileNotFoundError, ValueError):
            continue
    frames = summary(every_n, source_rate(recordings))

    lineage = ""
    module_id = record.get("module_id")
    first = next(iter(per.items()), None)
    if first is not None and recordings and module_id:
        sid, info = first
        run_id = info.get("run_id")
        recording = next((r for r in recordings if r.id == sid), None)
        if recording is not None and run_id:
            try:
                # A run record stores the model *id*; the user reads the name (§11). Each
                # ancestor leads with the experiment run that made it.
                lineage = run_spec.run_label(
                    recording,
                    module_id,
                    run_id,
                    experiment.model_names(),
                    class_names=experiment.class_names(),
                )
            except (KeyError, ValueError, FileNotFoundError):
                lineage = ""
    return frames, lineage


def _remote_cost_rate(compute: dict | None) -> float | None:
    """``$/hour`` for a still-running remote job's live cost estimate, or ``None``
    for a local job or a remote target with no configured rate (§ run location,
    duration, cost). A target that has since been renamed/removed is the same as
    "no rate" — a live estimate is a nicety, not a fact the page can insist on."""
    if not compute or compute.get("backend") != "remote":
        return None
    with contextlib.suppress(Exception):
        target = api_remote.get_remote_target(compute.get("remote_target"))
        if target.cost_per_gpu_hour is not None:
            return target.gpus_per_job * target.cost_per_gpu_hour
    return None


class RunResults(QWidget):
    """A run's recordings, and the results of the one selected — one pane, one list.

    **The class: two lists of the same recordings.** The page carried a *Per-recording* table
    (recording, produced run id, status) and, under it, the module's result view, which opened
    with a **dropdown of the same recordings** to choose between. Two controls for one choice,
    the table's own rows inert beside it, and together they pushed the video below the fold of
    a full-screen window. The table is the picker now: select a row and its results appear
    underneath, which is also what makes the rest of the page fit without scrolling.

    A row that did not complete says so *here*, in the results space, rather than in an Error
    column that was mostly empty and always as wide as its worst row.

    Double-click still opens the recording itself, so the row means the same thing it did.
    """

    #: Rows before the list scrolls. The results underneath are the reason: this is a picker
    #: with a preview, not a table with a picture under it, so the list takes what it needs to
    #: be usable and the video or scatter keeps the rest of the window.
    ROWS = 5

    def __init__(self, experiment, record: dict, ui, on_open_recording=None) -> None:
        super().__init__()
        self._experiment = experiment
        self._record = record
        self._ui = ui
        self._on_open_recording = on_open_recording
        self._module_id = record.get("module_id")
        self._per = record.get("per_recording") or {}
        self._recording_ids = list(self._per)
        self._selected_id: str | None = None
        self._shown = None  # the widget in the results space, replaced on every pick
        self._whole_run = None  # a module with no per-recording view builds this once
        # Expanding, so `add_body_widget` hands this the page's slack: the results space is
        # what absorbs a resize, and the video inside it is what rescales.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        names = experiment.recording_names()
        # Detection's per-recording question is how much of each recording it found something
        # on — asked across a run of many recordings, it is a column, not a pick per row.
        coverage = self._module_id == "detection"
        self._table = IndexTable(
            ["Recording", "Run id", "Status"]
            + (["Frames with detections"] if coverage else []),
            empty_text="(this run recorded no per-recording results)",
            max_rows=self.ROWS,
        )
        self._table.set_rows(
            [
                (
                    names.get(sid, sid),  # the name is the handle (§11)
                    info.get("run_id"),
                    info.get("status"),
                )
                + (
                    ("…" if info.get("status") == "complete" else "",)
                    if coverage
                    else ()
                )
                for sid, info in self._per.items()
            ],
            ids=self._recording_ids,
        )
        if coverage:
            self._read_coverage()
        self._table.view.doubleClicked.connect(self._open_recording_row)
        # Selection, not `clicked`: the keyboard moves it too, and a list you can only drive
        # with a mouse is one half the people using it cannot.
        self._table.view.selectionModel().currentRowChanged.connect(self._on_selected)
        layout.addWidget(self._table)
        self._results = QVBoxLayout()
        layout.addLayout(self._results, 1)
        self._select_first()

    def sizeHint(self) -> QSize:
        """The smallest useful version of itself — this pane is *given* height, not asking.

        A `QTableView`'s hint plus a video's own dimensions is how a page comes to prefer more
        height than the window it is shown in, and the shell then scrolls it: measured at 866px
        of page in a 700px content area, with the transport below the bottom edge. The pane is
        the page's fill child, so what it asks for is its floor and what it *gets* is the rest
        of the window.
        """
        return QSize(super().sizeHint().width(), self.minimumSizeHint().height())

    #: The frames column's index: after Recording, Run id and Status.
    _COVERAGE_COLUMN = 3

    def _read_coverage(self) -> None:
        """Fill the frames column off the UI thread — each count reads a run's store."""
        experiment = self._experiment
        complete = {
            sid: str(info.get("run_id"))
            for sid, info in self._per.items()
            if info.get("status") == "complete"
        }

        def read() -> dict:
            counts = {}
            for sid, run_id in complete.items():
                try:
                    recording = experiment.get_recording(sid)
                    counts[sid] = recording.detection_coverage(run_id)  # off-thread
                except (FileNotFoundError, ValueError):
                    counts[sid] = None  # the recording or its run is gone
            return counts

        self._coverage_call = BackgroundCall(read, receiver=self)
        self._coverage_call.done.connect(self._on_coverage)
        # A read that failed outright says so per row rather than leaving "…" forever.
        self._coverage_call.failed.connect(
            lambda _exc: self._on_coverage(dict.fromkeys(complete))
        )
        self._coverage_call.start()

    def _on_coverage(self, counts: dict) -> None:
        for sid, pair in counts.items():
            if pair is None:
                self._table.set_cell(sid, self._COVERAGE_COLUMN, "(unreadable)")
                continue
            with_detections, processed = pair
            # Sorts on the fraction: 900/1,000 is the better recording than 1,000/10,000.
            self._table.set_cell(
                sid,
                self._COVERAGE_COLUMN,
                SortCell(
                    f"{with_detections:,}/{processed:,}",
                    with_detections / processed if processed else 0.0,
                ),
            )

    def _select_first(self) -> None:
        """Open on a recording rather than on an empty space, preferring one that completed."""
        order = sorted(
            range(len(self._recording_ids)),
            key=lambda row: self._per[self._recording_ids[row]].get("status")
            != "complete",
        )
        if order:
            self._table.view.selectRow(order[0])

    def _on_selected(self, current, _previous=None) -> None:
        recording_id = row_id(current) if current is not None else None
        if recording_id is None:
            return
        self._selected_id = recording_id
        self._show(recording_id)

    def export_scope(self):
        """The recording on screen, as the scope the top bar's **Export…** opens on.

        What a cluster workspace does for its cluster: the page says which part of its subject
        the user is looking at, and the dialog starts there. Without it, **Export…** from the
        page of a large run offered the whole run, and a keypoints video export of it
        wrote every recording when one was wanted. The dialog's *Recordings* list is where the
        scope is widened again.
        """
        from detail.api.exporters import ExportScope

        recording_id = self._selected_id
        if recording_id is None or len(self._recording_ids) < 2:
            return None  # nothing to narrow: one recording is the whole run
        return ExportScope(recording_ids=(recording_id,))

    def _open_recording_row(self, index) -> None:
        recording_id = row_id(index)
        if recording_id is not None and self._on_open_recording is not None:
            self._on_open_recording(recording_id)

    def _show(self, recording_id: str) -> None:
        if self._shown is not None:
            self._shown.setParent(None)
            self._shown.deleteLater()
            self._shown = None
        self._shown = self._widget_for(recording_id)
        self._results.addWidget(self._shown, 1)

    def _widget_for(self, recording_id: str) -> QWidget:
        info = self._per.get(recording_id) or {}
        status = info.get("status")
        if status != "complete":
            return self._failure_label(recording_id, info)
        recording = self._recording(recording_id)
        if self._ui is None:
            from detail.gui.keypoint_run_preview import legacy_preview_widget

            legacy = (
                legacy_preview_widget(
                    recording, str(self._module_id), str(info.get("run_id"))
                )
                if recording is not None
                else None
            )
            return legacy or QLabel("(this module has no result view)")
        if recording is None:
            return QLabel(f"({recording_id} is no longer in this experiment)")
        widget = self._ui.preview_widget(
            recording,
            info.get("run_id"),
            experiment_dir=self._experiment.dir,
            parent_run_id=self._record.get(
                "id"
            ),  # the experiment's run, what an export is of
        )
        if widget is not None:
            return widget
        # A module whose results are the run's, not each recording's. Built once:
        # picking another row shows the same thing, because it *is* the same thing.
        if self._whole_run is None:
            self._whole_run = self._ui.result_widget(self._experiment, self._record)
        return self._whole_run

    def _recording(self, recording_id: str):
        """The recording, or ``None`` when the experiment no longer has it.

        A run record outlives its recordings: one deleted since the run leaves a row whose
        results cannot be read. This runs from a selection slot, where an exception is
        swallowed by the app's excepthook and the space simply stays blank.
        """
        try:
            return self._experiment.get_recording(recording_id)
        except (FileNotFoundError, ValueError, KeyError):
            return None

    def _failure_label(self, recording_id: str, info: dict) -> QWidget:
        """Why this recording has no results — the half of the Error column worth keeping.

        The stage record carries the status; the reason is in the run dir's envelope, which is
        what `_stage_run_error` reads. A recording that never started has neither, and says
        that rather than leaving the space blank.
        """
        status = info.get("status") or "not run"
        reason = info.get("error")
        recording = self._recording(recording_id) if info.get("run_id") else None
        if not reason and recording is not None:
            reason = _stage_run_error(recording, self._module_id, info["run_id"])
        label = QLabel(
            f"This recording's run is {status}."
            + (f"\n\n{reason}" if reason else "")
            + ("" if reason else "\n\nNo reason was recorded.")
        )
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        return label


def _next_stages(module_id: str | None) -> list[str]:
    """The per-recording stages a finished ``module_id`` run offers next, that this build
    offers, in pipeline order: each module's ``offered_after`` when it declares one, else those
    whose upstream it is and those that read it as a second input (``aux_refs``, §6.5) —
    embedding reads a keypoint track run that way to align (§14), so a finished track run
    offers *Run embedding →* as every other stage offers its next step.

    A batch stage is left out: it consumes a *set* of runs picked on its own form, not
    the one run this page is about. Hidden surfaces (§25) are left out too.
    """
    from detail import surfaces
    from detail.executor.runners import PROCESSING_MODULES
    from detail.gui.launch import _ordered_module_ids

    # A merged-tracks run *is* a tracking run to everything downstream (it writes the same
    # `tracking/track_NN` output), so it is offered whatever tracking is offered.
    if module_id == "track_merge":
        module_id = "tracking"
    return _ordered_module_ids(
        [
            m.module_id
            for m in PROCESSING_MODULES.values()
            if module_id in _offered_after(m)
            and not m.batch
            and surfaces.is_surfaced(m.module_id)
        ]
    )


def _offered_after(module) -> set[str]:
    """The stages whose finished run offers ``module`` next: its declaration, else its
    upstream and every auxiliary input."""
    declared = getattr(module, "offered_after", None)
    if declared is not None:
        return set(declared)
    return {module.upstream} | _aux_modules(module)


def _aux_modules(module) -> set[str]:
    """The modules ``module`` reads as auxiliary inputs (§6.5 ``aux_refs``)."""
    return {aux for aux, _field in getattr(module, "aux_refs", ()) or ()}


class RunDetailPage(QWidget):
    """One processing run: config + per-recording breakdown + (when complete) the
    module's result view. With a live ``AttachedRun`` handle (a fresh launch or a
    reattach after reopen), shows progress and rebuilds when the run finishes."""

    def __init__(
        self,
        experiment: Experiment,
        run_id: str,
        live=None,
        on_deleted=None,
        on_open_recording=None,
        on_run_launched=None,
    ):
        super().__init__()
        self._experiment = experiment
        self._run_id = run_id
        self._on_deleted = on_deleted
        self._on_open_recording = on_open_recording
        self._on_run_launched = on_run_launched
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(_heading(f"Run: {run_id}"))
        from detail.storage import runs as runs_store

        self._panel = LiveRunPanel()
        self._panel.set_live(live is not None)
        self._layout.addWidget(self._panel)
        self._body = QVBoxLayout()
        self._layout.addLayout(self._body)
        # Factor 1, not the default 0. A 0-factor spacer is merely *expanding*, so it competes
        # with every Preferred widget above it and the form takes the slack instead — laying its
        # single-line rows out 60px apart against 21 compact. A spread form then redistributes
        # whenever anything above it changes, which is how a panel that updates twice a second
        # reaches the bottom of the page.
        self._layout.addStretch(1)
        #: Where that spacer is, so a build with a fill child in it can hand the slack over
        #: instead: two items at stretch 1 split the window between them, which left the
        #: results space at half the height it should have had.
        self._stretch_index = self._layout.count() - 1
        self._body_widgets = []
        self._live_progress = None
        self._recordings_done = -1  # forces the first event to re-read the record
        self._build()
        if live is not None:
            self._attach_worker(live)

    def _attach_worker(self, handle) -> None:
        """Stream `handle` into this page — the one place both attach paths wire it.

        Opening on a live run and Retry (`_start_live`) each wired the worker by hand, and the
        copies drifted: Retry never connected `progress`, so a retried run's per-recording rows
        sat still under a moving bar.
        """
        self._worker = RunProgressWorker(handle, receiver=self)
        self._worker.tick.connect(self._on_tick)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_run.connect(self._on_done)
        self._worker.start()

    def _add(self, widget) -> None:
        add_body_widget(self._body, widget)
        self._body_widgets.append(widget)

    def _give_slack_to_the_body(self, fills: bool) -> None:
        """The trailing spacer takes the slack, unless something in the body wants it."""
        give_slack_to_body(self._layout, self._stretch_index, fills)

    def _build(self) -> None:
        for widget in self._body_widgets:
            discard(widget)
        self._body_widgets = []
        record = next(
            (r for r in self._experiment.processing_runs if r["id"] == self._run_id),
            None,
        )
        if record is None:
            self._add(QLabel("(run record not found)"))
            return
        # The band's fold follows the *record*, not whoever opened this page — see
        # LiveRunPanel.set_active for the pair of facts that were being read off one flag.
        self._panel.set_active(record.get("status") in ACTIVE_STATUSES)
        self._refresh_units(record)

        form_holder = QWidget()
        form = QFormLayout(form_holder)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Module:", QLabel(str(record.get("module_id"))))
        form.addRow("Where:", QLabel(run_cost.compute_label(record.get("compute"))))
        form.addRow("Status:", QLabel(str(record.get("status"))))
        form.addRow("Created:", QLabel(str(record.get("created_at"))))
        duration_s = record.get("duration_s")
        if duration_s is not None:
            form.addRow("Duration:", QLabel(format_duration(duration_s)))
        # Cost: a persisted final value once terminal; a live-updating estimate while
        # still running, ticking off the same elapsed_s the progress bar already
        # gets — no extra I/O per tick (§ run location, duration, cost).
        self._cost_label = None
        self._cost_rate_per_hour = None
        cost = record.get("total_estimated_cost_usd")
        if cost is not None:
            form.addRow("Cost:", QLabel(f"${cost:.2f}"))
        elif record.get("status") in ACTIVE_STATUSES:
            self._cost_rate_per_hour = _remote_cost_rate(record.get("compute"))
            if self._cost_rate_per_hour is not None:
                self._cost_label = QLabel("$0.00")
                form.addRow("Cost (running):", self._cost_label)
        frames, lineage = _run_cadence(self._experiment, record)
        if frames:
            form.addRow("Frames:", QLabel(frames))
        if lineage:
            form.addRow("From:", QLabel(lineage))
        cfg = QPlainTextEdit(str(record.get("config", {})))
        cfg.setReadOnly(True)
        cfg.setFixedHeight(lines(3))
        form.addRow("Config:", cfg)
        # What the run *is*, in a line, with the rest folded behind it. Measured: those six rows
        # are 188px of a content area this window gives a page 643 of, and the results below
        # them could then only be reached by scrolling — which is the report this answers. What
        # stays out is what tells two runs apart at a glance; the lineage and the config are
        # reference, read once, and one click away. The fold is remembered.
        summary = QLabel(
            "  ·  ".join(
                str(part)
                for part in (
                    record.get("module_id"),
                    record.get("status"),
                    (record.get("created_at") or "")[:16].replace("T", " "),
                    frames,
                )
                if part
            )
        )
        summary.setWordWrap(True)
        self._add(summary)
        details = CollapsibleBand(
            "Run details", form_holder, key="run_detail.details", open_=False
        )
        self._add(details)
        # Under Run details, opened from the shell's Notes button; folded, it costs this page
        # nothing, which it needs to fit the window unscrolled (`test_run_page_fits`).
        place_notes(self, self._body, self._experiment, "run", self._run_id)

        per = record.get("per_recording", {})
        # The live rows above and this list are the same claim about the same recordings, so
        # they take turns rather than both being on screen: while the run goes, what is
        # *happening*; once it is over, what came *out* — which is this list's own half, since
        # it carries each recording's produced run id, its results, and a way to open it.
        #
        # Scoped to the results. The first version of this skipped the *rest of the method* on
        # a live run, which also took Cancel run with it — the one control a running run needs.
        results = None
        if record.get("status") not in ACTIVE_STATUSES:
            self._add(_heading("Results"))
            ui = GUI_MODULES.get(record.get("module_id"))
            results = RunResults(
                self._experiment,
                record,
                ui,
                on_open_recording=self._on_open_recording,
            )
            # A module's own tabs beside the results (the keypoint track run's Audit, §26);
            # none leaves the results as the page's fill child with no tab bar over them.
            tabs = ui.run_tabs(self._experiment, record) if ui is not None else []
            if tabs:
                from PyQt6.QtWidgets import QTabWidget

                holder = QTabWidget()
                holder.setSizePolicy(
                    QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
                )
                holder.addTab(results, "Results")
                for title, widget in tabs:
                    holder.addTab(widget, title)
                results = holder
            self._add(results)
        self._give_slack_to_the_body(results is not None)

        # Next pipeline stages: every stage that reads this one — tracking after detection,
        # embedding and keypoint detect after tracking — one click each, pre-seeded with this
        # run's recordings. Derived from the registry: a hand map named one stage per run and
        # so never offered the second stage to read tracking (§26).
        if record.get("status") == "complete":
            for next_stage in _next_stages(record.get("module_id")):
                run_next = QPushButton(f"Run {next_stage} →")
                run_next.clicked.connect(
                    lambda _=False, m=next_stage: self._run_next_stage(m)
                )
                self._add(run_next)

        if record.get("status") in ACTIVE_STATUSES:
            cancel = QPushButton("Cancel run")
            cancel.clicked.connect(self._cancel)
            self._add(cancel)
        else:
            # Every recording that isn't complete — failed, cancelled, or never launched
            # (absent from per_recording entirely, hence the scan over recording_ids). The
            # statuses are already in the record, so the count costs no extra read.
            pending = [
                sid
                for sid in record.get("recording_ids") or []
                if per.get(sid, {}).get("status") != "complete"
            ]
            if pending:
                plural = "s" if len(pending) != 1 else ""
                self._retry_btn = QPushButton(
                    f"Retry {len(pending)} incomplete recording{plural}"
                )
                self._retry_btn.clicked.connect(lambda: self._retry())
                self._add(self._retry_btn)
            delete = QPushButton("Delete run")
            delete.clicked.connect(self._delete)
            self._add(delete)

    def _retry(self, ignore_free_space: bool = False) -> None:
        """Re-run the incomplete recordings, off the UI thread.

        Not shaped like ``_FitDetailPage._resume``, which calls its API inline: retry
        validates every incomplete recording and spawns a coordinator — O(recordings) disk
        work, the freeze CLAUDE.md forbids on the UI thread. No confirm: it destroys
        nothing, and the count is in the label — unless the re-run would not fit on the
        drive, which asks (``space_prompt``, #743).
        """
        self._retry_btn.setEnabled(False)
        self._status.setText("Validating recordings…")
        self._status.setVisible(True)
        worker = BackgroundCall(
            lambda progress: self._experiment.retry_run(
                self._run_id, ignore_free_space=ignore_free_space, progress=progress
            ),
            receiver=self,
            reports_progress=True,
        )
        status_line = StageStatusLine()
        worker.tick.connect(
            lambda stage, fraction, detail: self._status.setText(
                status_line(stage, fraction, detail)
            )
        )
        worker.done.connect(self._start_live)

        def _failed(exc) -> None:
            self._retry_btn.setEnabled(True)
            self._status.setVisible(False)
            self._status.setText("")
            if launch_anyway(self, "Retry run", exc):
                self._retry(ignore_free_space=True)
                return
            message = (
                str(exc)
                if isinstance(exc, (ValueError, FileNotFoundError))
                else f"{type(exc).__name__}: {exc}"
            )
            QMessageBox.critical(self, "Retry run", message)

        worker.failed.connect(_failed)
        self._retry_worker = worker  # keep a ref so the thread isn't collected
        worker.start()

    def _start_live(self, handle) -> None:
        """Attach a freshly-relaunched run to this already-open page (retry).

        Order matters: ``_build`` re-reads the record, so ``retry_run`` must already
        have flipped it to ``running`` — which it does before returning the handle —
        or the rebuilt page still shows Retry instead of Cancel. Rebinding ``_worker``
        is safe because this is only reachable from a terminal state, where the
        previous worker has already emitted ``finished_run`` and exited.
        """
        self._panel.set_live(True)
        self._build()
        self._attach_worker(handle)

    def _cancel(self) -> None:
        _confirm_and_cancel(
            self,
            "Cancel run",
            f"Cancel run {self._run_id}? Its in-flight workers will be stopped.",
            lambda: self._experiment.cancel_run(self._run_id),
            self._build,
        )

    def _run_next_stage(self, next_module: str) -> None:
        from detail.gui.launch import LaunchRunDialog

        record = next(
            (r for r in self._experiment.processing_runs if r["id"] == self._run_id), {}
        )
        seed = {
            "module_id": next_module,
            "recording_ids": record.get("recording_ids") or [],
        }
        # A stage that reads this run as a *second* input (embedding reading a keypoint track
        # run, §14) has its own upstream picker too; name this run's settings so the form opens
        # on them rather than on whichever group happens to cover the most recordings. Nothing
        # else is guessed: the alignment mode depends on the camera (Tenet 9).
        from detail.executor.runners import PROCESSING_MODULES

        module_id = record.get("module_id")
        reader = PROCESSING_MODULES.get(next_module)
        if reader is not None and module_id in _aux_modules(reader):
            spec = self._this_runs_spec(record)
            if spec is not None:
                seed["upstream_specs"] = {module_id: spec}
        dialog = LaunchRunDialog(self._experiment, parent=self, seed=seed)
        if (
            dialog.exec() == QDialog.DialogCode.Accepted
            and dialog.run_handle is not None
            and self._on_run_launched is not None
        ):
            self._on_run_launched(dialog.run_handle)

    def _this_runs_spec(self, record: dict) -> dict | None:
        """This run's settings as a run spec (§6.5), read from the first recording it
        completed; ``None`` when it completed none."""
        from detail.api import run_spec

        for recording_id, info in (record.get("per_recording") or {}).items():
            stage_run = info.get("run_id")
            if info.get("status") != "complete" or not stage_run:
                continue
            recording = self._experiment.get_recording(recording_id)
            return run_spec.spec_from_run(
                recording, str(record.get("module_id")), str(stage_run)
            )
        return None

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete run",
            f"Delete run {self._run_id} and its outputs? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_run(self._run_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete run", str(exc))
            return
        if self._on_deleted is not None:
            self._on_deleted()

    #: The bar and the status line live on the panel now. Kept as properties because the pages
    #: and their tests reach for them by these names, and because a run page asking its panel for
    #: its bar reads worse than a page that simply has one.
    @property
    def _progress(self):
        return self._panel.progress

    @property
    def _status(self):
        return self._panel.status

    def _on_tick(self, permille: int, status: str) -> None:
        self._panel.tick(permille, status)

    def _on_progress(self, event) -> None:
        """Redraw the unit rows from the live event.

        Re-reads the run record **only when a recording boundary is crossed**, which is the only
        moment new per-recording facts exist. Re-reading on every event would be a disk read
        twice a second on the UI thread for the life of the run — the cost model this package
        has already been bitten by, and invisible on a local disk.
        """
        self._live_progress = event
        done = getattr(event, "recordings_done", None)
        if done != self._recordings_done:
            self._recordings_done = done
            self._record_cache = self._find_record()
        self._refresh_units(getattr(self, "_record_cache", None))
        if self._cost_label is not None and self._cost_rate_per_hour is not None:
            elapsed_s = getattr(event, "elapsed_s", None)
            if elapsed_s is not None:
                self._cost_label.setText(
                    f"${elapsed_s / 3600.0 * self._cost_rate_per_hour:.2f}"
                )

    def _find_record(self):
        return next(
            (r for r in self._experiment.processing_runs if r["id"] == self._run_id),
            None,
        )

    def _refresh_units(self, record) -> None:
        """Draw the per-recording rows — **while the run is going, and not after**.

        The Per-recording table below says the same thing for a finished run, and better: it
        carries each recording's produced run id and a way to open it, which a progress row has
        no business growing. Showing both at once would be two surfaces answering one question,
        which is this package's most-repeated class — so they take turns, and each has the half
        of the run's life it is right for. Live: what is happening. Finished: what came out.
        """
        if record is None:
            return
        self._record_cache = record
        if record.get("status") not in ACTIVE_STATUSES:
            self._panel.set_units([])  # the Per-recording table takes over below
            return
        self._panel.set_units(
            run_units.processing_units(
                record,
                self._live_progress,
                names=self._experiment.recording_names(),
            ),
            heading="Recordings",
        )

    def _on_done(self) -> None:
        self._panel.set_live(False)
        self._live_progress = None  # the run is over; no row is in flight
        self._queue_previews(self._find_record())
        self._build()

    def _queue_previews(self, record) -> None:
        """A finished tracking run has masks to shrink for the previews (§3): queue its
        recordings on the app's preview queue. Nothing to do for any other module."""
        if not record or record.get("module_id") != "tracking":
            return
        if record.get("status") != "complete":
            return
        preview_jobs.queue_recordings(
            self._experiment, list(record.get("recording_ids") or [])
        )


def _stage_run_error(recording, module_id: str, run_id: str) -> str | None:
    """The failure reason for a per-recording stage run, read from its run dir's
    job.json — the recording.yaml stage record carries only the status, not the
    reason. Returns None if the module/run dir/envelope is absent."""
    from detail.executor import rundir
    from detail.executor.runners import PROCESSING_MODULES

    module = PROCESSING_MODULES.get(module_id)
    if module is None:
        return None
    run_dir = Path(recording.dir) / module.output_path(run_id)
    try:
        return rundir.read_envelope(run_dir).get("error")
    except FileNotFoundError:
        return None


class RecordingRunDetailPage(QWidget):
    """One per-recording stage run (det_01 / track_01 / emb_01) opened directly from a
    recording's Run history — for runs launched per recording via the CLI/API, which
    carry no experiment run record. It asks the module for that one recording's results
    (`ModuleUI.preview_widget`), which is what the run-detail page's results list builds too.
    """

    def __init__(
        self, experiment: Experiment, recording_id: str, module_id: str, run_id: str
    ):
        super().__init__()
        layout = QVBoxLayout(self)
        recording = experiment.get_recording(recording_id)
        layout.addWidget(_heading(f"{module_id}: {run_id}  ·  {recording.name}"))
        runs = getattr(recording, f"{module_id}_runs", None)
        if runs is None:
            # A module this build no longer has (the stage before §26 wrote ``keypoint_runs``):
            # its runs are still listed in recording.yaml.
            from detail.storage import recording as recording_store

            legacy_key = {"keypoints": "keypoint_runs"}.get(
                module_id, f"{module_id}_runs"
            )
            runs = recording_store.read_recording(recording.dir).get(legacy_key, [])
        record = {r["id"]: r for r in runs}.get(run_id)
        status = record.get("status") if record else None
        if record is not None:
            layout.addWidget(
                QLabel(f"Where: {run_cost.compute_label(record.get('compute'))}")
            )
        layout.addWidget(QLabel(f"Status: {status or '—'}"))
        if record is not None:
            duration_s = record.get("duration_s")
            if duration_s is not None:
                layout.addWidget(QLabel(f"Duration: {format_duration(duration_s)}"))
            cost = record.get("estimated_cost_usd")
            if cost is not None:
                layout.addWidget(QLabel(f"Cost: ${cost:.2f}"))
        if record is None:
            layout.addWidget(QLabel("(run not found)"))
            layout.addStretch()
            return
        error_record = record
        if status != "complete":  # the reason lives in the run dir's job.json
            reason = _stage_run_error(recording, module_id, run_id)
            if reason:
                error_record = {**record, "error": reason}
        error = _FitDetailPage._error_label(error_record)
        if error is not None:
            layout.addWidget(error)
        ui = GUI_MODULES.get(module_id)
        if status != "complete":
            preview = None
        elif ui is not None:
            # No experiment-level run behind this one (it was launched per recording), so
            # there is nothing an export could be of: the viewer leaves Export out.
            preview = ui.preview_widget(
                recording, run_id, experiment_dir=experiment.dir, parent_run_id=None
            )
        else:
            from detail.gui.keypoint_run_preview import legacy_preview_widget

            preview = legacy_preview_widget(recording, module_id, run_id)
        if preview is None:
            layout.addStretch()
        else:
            layout.addWidget(_heading("Results"))
            layout.addWidget(preview, 1)


# --------------------------------------------------------------------------- #
# Training (fit/train family, §12) — experiment-scoped datasets + models
# --------------------------------------------------------------------------- #
class DatasetsPage(QWidget):
    """Datasets index. Open one → ``on_open(dataset_id)``; generate a new dataset
    → ``on_launched(DatasetRun)``."""

    def __init__(self, experiment: Experiment, on_open, on_launched=None):
        super().__init__()
        self._experiment = experiment
        self._on_launched = on_launched
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Datasets"))
        refs = experiment.datasets
        layout.addWidget(_index_caption(f"{len(refs)} dataset(s)", rows=len(refs)))
        rows = [(d.id, d.name, d.status, d.n_train, d.n_val) for d in refs]
        self._ids = [d.id for d in refs]
        table = IndexTable(
            ["Id", "Name", "Status", "Train", "Val"],
            empty_text="(no datasets yet — Generate dataset builds one from a recording's labels)",
        )
        table.set_rows(rows, ids=self._ids)
        table.view.doubleClicked.connect(lambda idx: on_open(row_id(idx)))
        layout.addWidget(table)
        new = QPushButton("Generate dataset")
        new.clicked.connect(self._generate)
        layout.addWidget(new, alignment=Qt.AlignmentFlag.AlignLeft)
        label = QPushButton("Auto-label")
        label.clicked.connect(self._label)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

    def _generate(self) -> None:
        from detail.gui.fit import GenerateDatasetDialog

        dialog = GenerateDatasetDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_launched:
            self._on_launched(dialog.run_handle)

    def _label(self) -> None:
        from detail.gui.fit import GenerateLabelingDialog

        dialog = GenerateLabelingDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_launched:
            self._on_launched(dialog.run_handle)


#: An experiment's trained-models table — one declaration for the Models page and the Model
#: manager's experiment table, so the two views of the same models cannot drift apart.
EXPERIMENT_MODEL_COLUMNS = [
    "Id",
    "Name",
    "Kind",
    "Status",
    "Source",
    "Framework",
    "Dataset",
]


def model_kind(role: str | None) -> str:
    """What a Kind column says about a model or a distill run: the words *Distill new model*
    offers (``detector`` / ``keypoints``), not the stored role — ``keypoints`` on an
    experiment's model, ``keypoint`` (its ``MODEL_REGISTRIES`` role) on a catalog row.
    """
    return "keypoints" if role in ("keypoints", "keypoint") else "detector"


EXPERIMENT_MODELS_EMPTY = (
    "(no trained models yet — ＋ Distill new model labels frames and trains one; "
    "Train from dataset… trains on a dataset you already have)"
)


def experiment_model_rows(experiment) -> tuple[list[str], list[tuple]]:
    """``(ids, rows)`` for an experiment's trained models, **newest first**.

    ``Source`` is the distill run that trained the model, or ``trained`` for one trained from
    a dataset directly. It is read off the distill records — each names the model its train
    stage produced — rather than stored on the model, so no model schema carries it.
    """
    made_by = {}
    for record in experiment.distill_runs:
        train = (record.get("stages") or {}).get("train") or {}
        if train.get("model_id"):
            made_by[train["model_id"]] = record.get("id")
    refs = list(reversed(experiment.models))  # the store lists oldest first
    rows = [
        (
            m.id,
            m.name,
            model_kind(m.role),
            m.status,
            made_by.get(m.id, "trained"),
            m.trainer_id,
            m.dataset_id,
        )
        for m in refs
    ]
    return [m.id for m in refs], rows


class ModelsPage(QWidget):
    """An experiment's trained models, and the distill runs that make them (§12).

    Two tables. **Models** is what can be used — a detection run picks from it. **Distill
    runs** is how most of them were made, and stays its own table rather than rows of the
    first: a run that failed while labelling produced no model, and it is where **Resume**
    lives. Open a model → ``on_open(model_id)``; open a run → ``on_open_distill_run(run_id)``;
    a launch from either button → ``on_launched`` / ``on_distill_launched``.
    """

    def __init__(
        self,
        experiment: Experiment,
        on_open,
        on_launched=None,
        *,
        on_open_distill_run=None,
        on_distill_launched=None,
        banner: str | None = None,
        on_banner_dismiss=None,
    ):
        super().__init__()
        self._experiment = experiment
        self._on_open = on_open
        self._on_launched = on_launched
        self._on_open_distill_run = on_open_distill_run
        self._on_distill_launched = on_distill_launched
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Models"))
        if banner is not None:  # keypoint setup still to do (§26, #717)
            layout.addWidget(_ReadinessBanner(banner, on_banner_dismiss))

        actions = QHBoxLayout()
        # One way to make any model (§12): the dialog's Model type chooses a detector or a
        # keypoint model, as the Model manager's button does.
        distill = QPushButton("＋ Distill new model")
        distill.clicked.connect(self._distill)
        actions.addWidget(distill)
        setup_keypoints = QPushButton("Set up keypoints…")
        setup_keypoints.setToolTip(
            "Teach a keypoint model from a few minutes of each camera or box, then train a "
            "fast one on it: the keypoint model alignment and pose use (§26)"
        )
        setup_keypoints.clicked.connect(self._setup_keypoints)
        actions.addWidget(setup_keypoints)
        self._train_btn = QPushButton("Train from dataset…")
        self._train_btn.clicked.connect(self._train)
        actions.addWidget(self._train_btn)
        uploads = QPushButton("Background uploads…")
        uploads.setToolTip(
            "Watch, or stop, the videos a distill is copying to a cluster in the background."
        )
        uploads.clicked.connect(self._background_uploads)
        actions.addWidget(uploads)
        actions.addStretch()
        layout.addLayout(actions)

        self._body = QVBoxLayout()
        layout.addLayout(self._body)
        layout.addStretch()
        self._widgets: list[QWidget] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-read both tables (the shell calls this after a launch or a delete)."""
        for widget in self._widgets:
            discard(widget)
        self._widgets = []
        experiment = self._experiment

        ids, rows = experiment_model_rows(experiment)
        self._add(_index_caption(f"{len(rows)} model(s)", rows=len(rows)))
        models = IndexTable(
            EXPERIMENT_MODEL_COLUMNS, empty_text=EXPERIMENT_MODELS_EMPTY
        )
        models.set_rows(rows, ids=ids)
        models.view.doubleClicked.connect(lambda idx: self._on_open(row_id(idx)))
        self._add(models)

        self._add(_band_caption("Distill runs"))
        runs = list(reversed(experiment.distill_runs))  # newest first, like the models
        self._add(_index_caption(f"{len(runs)} distill run(s)", rows=len(runs)))
        run_rows = []
        for record in runs:
            stages = record.get("stages") or {}
            run_rows.append(
                (
                    record.get("id"),
                    record.get("name", ""),
                    model_kind(record.get("kind")),
                    record.get("status"),
                    (stages.get("label") or {}).get("status", "—"),
                    (stages.get("train") or {}).get("status", "—"),
                )
            )
        # The `›` cue is derived from the wiring: a run table nobody can open does not
        # advertise that it opens.
        navigable = self._on_open_distill_run is not None
        table = IndexTable(
            ["Id", "Name", "Kind", "Status", "Label", "Train"],
            empty_text=("(no distill runs yet — ＋ Distill new model starts one)"),
            navigable=navigable,
        )
        table.set_rows(run_rows, ids=[r.get("id") for r in runs])
        if navigable:
            table.view.doubleClicked.connect(
                lambda idx: self._on_open_distill_run(row_id(idx))
            )
        self._add(table)

        has_dataset = any(d.status == "complete" for d in experiment.datasets)
        sizing.set_available(
            self._train_btn,
            has_dataset,
            "Needs a completed dataset — make one on the Datasets page, or ＋ Distill new model",
        )

    def _add(self, widget: QWidget) -> None:
        self._body.addWidget(widget)
        self._widgets.append(widget)

    def _distill(self) -> None:
        from detail.gui.fit import DistillDialog

        dialog = DistillDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_distill_launched:
            self._on_distill_launched(dialog.run_handle)

    def _setup_keypoints(self) -> None:
        from detail.gui.keypoint_setup import KeypointSetupDialog

        dialog = KeypointSetupDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_distill_launched:
            self._on_distill_launched(dialog.run_handle)

    def _train(self) -> None:
        from detail.gui.fit import TrainModelDialog

        dialog = TrainModelDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_launched:
            self._on_launched(dialog.run_handle)

    def _background_uploads(self) -> None:
        from detail.gui.staging import BackgroundUploadsDialog

        BackgroundUploadsDialog(self._experiment, parent=self).exec()


class _FitDetailPage(QWidget):
    """Shared scaffold for a dataset/model detail view: heading + (when launched
    live) a progress bar fed by a ``FitProgressWorker``, rebuilt on completion.
    """

    def __init__(self, title: str, live=None):
        super().__init__()
        self._layout = QVBoxLayout(self._content_widget())
        self._layout.addWidget(_heading(title))
        self._panel = LiveRunPanel()
        self._panel.set_live(live is not None)
        self._layout.addWidget(self._panel)
        self._body = QVBoxLayout()
        self._layout.addLayout(self._body)
        # Factor 1, not the default 0. A 0-factor spacer is merely *expanding*, so it competes
        # with every Preferred widget above it and the form takes the slack instead — laying its
        # single-line rows out 60px apart against 21 compact. A spread form then redistributes
        # whenever anything above it changes, which is how a panel that updates twice a second
        # reaches the bottom of the page.
        self._layout.addStretch(1)
        self._stretch_index = self._layout.count() - 1
        self._body_widgets = []
        self._curve = None
        self._curve_model: str | None = None
        self._curve_total: int | None = None
        self._curve_epoch_key = None
        self._build()
        if live is not None:
            self._attach_worker(live)

    def _content_widget(self) -> QWidget:
        """The widget the heading, live panel and body are laid out in: the page itself here.

        `AnalysisRunDetailPage` answers a landing page inside a stack, so the same scaffold
        becomes the page you land on, with a full-window view host beside it.
        """
        return self

    def _attach_worker(self, handle) -> None:
        """Stream `handle` into this page — the one place both attach paths wire it, for the
        reason `RunDetailPage._attach_worker` gives."""
        # A new attempt (Resume) counts epochs afresh; the last one seen belongs to the old run
        # and would swallow the first redraw of this one if it happened to match.
        self._curve_epoch_key = None
        self._worker = FitProgressWorker(handle, receiver=self)
        self._worker.tick.connect(self._on_tick)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_run.connect(self._on_done)
        self._worker.start()

    def _add(self, widget) -> None:
        add_body_widget(self._body, widget)
        self._body_widgets.append(widget)
        if expands_vertically(widget):
            give_slack_to_body(self._layout, self._stretch_index, True)

    def _build(self) -> None:
        for widget in self._body_widgets:
            discard(widget)
        self._body_widgets = []
        give_slack_to_body(self._layout, self._stretch_index, False)
        self._curve = None  # its widget just went with the rest; _render re-adds it
        self._curve_model = None
        try:
            record = self._record()
        except FileNotFoundError:
            self._add(QLabel("(record not found)"))
            return
        self._render(record)
        # The band's fold follows the *record*, not whoever opened this page — this page shows
        # its stop/cancel buttons off the same status, so reading the two from different places
        # is how a running job gets a folded, empty Activity band (LiveRunPanel.set_active).
        self._panel.set_active(record.get("status") in ACTIVE_STATUSES)

    @property
    def _progress(self):
        return self._panel.progress

    @property
    def _status(self):
        return self._panel.status

    def _on_tick(self, permille: int, status: str) -> None:
        self._panel.tick(permille, status)

    def _on_progress(self, event) -> None:
        """Hook for a subclass that draws unit rows from the structured event.

        A no-op here: most `_FitDetailPage` subclasses (dataset, model) have one bar and no
        parts to break it into — `DistillRunDetail` is the one that overrides this, the same
        way `RunDetailPage._on_progress` reads `RunProgressWorker.progress`.
        """

    def _add_training_curve_for(self, model_id: str, hyperparams: dict) -> None:
        """Add `model_id`'s training curve, and remember it so a finished epoch redraws it.

        With no history yet nothing is added, but the model is still remembered: the first
        finished epoch then rebuilds the page once, which adds the curve.
        """
        self._curve_model = model_id
        self._curve_total = hyperparams.get("epochs")
        history = self._experiment.model_metric_history(model_id)
        self._curve = _training_curve_widget(history, hyperparams)
        if self._curve is not None:
            self._add(self._curve)

    def _epoch_finished(self, key) -> None:
        """Redraw the curve when `key` — whatever a subclass's event says about epochs — moves.

        The curve was drawn when the page was built and not again until the run ended, so the
        question it is on the page to answer, *is it still improving*, showed the moment the
        page opened for the whole of a training run — and a distill page, built as training
        began, showed no curve at all. Redrawn in place (`set_history` does not stack) rather
        than by rebuilding, which would tear down every widget on the page once an epoch.
        Ultralytics writes the epoch's ``results.csv`` row before the callback that reports it.
        """
        if key is None or key == self._curve_epoch_key or self._curve_model is None:
            return
        self._curve_epoch_key = key
        try:
            history = self._experiment.model_metric_history(self._curve_model)
        except FileNotFoundError:
            return
        if self._curve is not None:
            self._curve.set_history(history, total_epochs=self._curve_total)
        elif history:
            self._build()

    def _on_done(self, error: str = "") -> None:
        self._panel.set_live(False)
        self._build()  # the rebuilt record now carries the error (shown by _render)
        if error:
            QMessageBox.critical(self, "Run failed", error)

    def _start_live(self, handle) -> None:
        """Attach a freshly-launched run to this already-open page — show the progress bar and
        stream it (used when a run is started from the detail page itself, e.g. resume).
        """
        self._panel.set_live(True)
        self._build()  # reflect the now-running status
        self._attach_worker(handle)

    @staticmethod
    def _error_label(record: dict) -> QLabel | None:
        """A red, word-wrapped label with a failed run's recorded error, or None."""
        if record.get("status") != "failed" or not record.get("error"):
            return None
        label = QLabel(str(record["error"]))
        label.setWordWrap(True)
        palette = label.palette()
        palette.setColor(QPalette.ColorRole.WindowText, ERROR)
        label.setPalette(palette)
        return label

    def _record(self) -> dict:
        raise NotImplementedError

    def _render(self, record: dict) -> None:
        raise NotImplementedError

    def _add_notes(self, kind: str, entry_id: str) -> None:
        """The entry's Notes box, below its details and above its results (`gui/notes.py`)."""
        place_notes(self, self._body, self._experiment, kind, entry_id)


class DatasetDetailPage(_FitDetailPage):
    """One training dataset: source lineage, frame selection, class list, counts."""

    def __init__(
        self, experiment: Experiment, dataset_id: str, live=None, on_deleted=None
    ):
        self._experiment = experiment
        self._dataset_id = dataset_id
        self._on_deleted = on_deleted
        super().__init__(f"Dataset: {dataset_id}", live=live)

    def _record(self) -> dict:
        return self._experiment.dataset_record(self._dataset_id)

    def _render(self, record: dict) -> None:
        holder = QWidget()
        form = QFormLayout(holder)
        source = record.get("source", {}) or {}
        form.addRow("Name:", QLabel(str(record.get("name", ""))))
        form.addRow("Status:", QLabel(str(record.get("status"))))
        error = self._error_label(record)
        if error is not None:
            form.addRow("Error:", error)
        form.addRow("Where:", QLabel(run_cost.compute_label(record.get("compute"))))
        duration_s = record.get("duration_s")
        if duration_s is not None:
            form.addRow("Duration:", QLabel(format_duration(duration_s)))
        cost = record.get("estimated_cost_usd")
        if cost is not None:
            form.addRow("Cost:", QLabel(f"${cost:.2f}"))
        form.addRow("Source run:", QLabel(str(source.get("run_id", "—"))))
        form.addRow("Frames:", QLabel(str(source.get("frame_selection", "—"))))
        form.addRow(
            "Train / Val:",
            QLabel(f"{record.get('n_train', 0)} / {record.get('n_val', 0)}"),
        )
        classes = [c.get("name") for c in record.get("classes", [])]
        form.addRow("Classes:", QLabel(", ".join(classes) if classes else "—"))
        self._add(holder)
        self._add_notes("dataset", self._dataset_id)
        if record.get("status") in ACTIVE_STATUSES:
            cancel = QPushButton("Cancel dataset")
            cancel.clicked.connect(self._cancel)
            self._add(cancel)
        else:
            delete = QPushButton("Delete dataset")
            delete.clicked.connect(self._delete)
            self._add(delete)

    def _cancel(self) -> None:
        _confirm_and_cancel(
            self,
            "Cancel dataset",
            f"Cancel dataset {self._dataset_id}? Its worker will be stopped.",
            lambda: self._experiment.cancel_dataset(self._dataset_id),
            self._build,
        )

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete dataset",
            f"Delete dataset {self._dataset_id} and its images/labels? "
            "This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_dataset(self._dataset_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete dataset", str(exc))
            return
        if self._on_deleted is not None:
            self._on_deleted()


def _training_curve_widget(history: list[dict], hyperparams: dict) -> QWidget | None:
    """The training curve for one model's history — the one widget `ModelDetailPage` and
    `DistillRunDetail` both embed, built here once so watching a distill run and watching its
    model draw the same picture rather than two independently-written ones (this package's
    most-repeated class). `None` when there is no history yet — an imported model has no
    epochs at all, and a training that hasn't finished one has none either — a plot of nothing
    is the "renders the frame of the thing it does not have" failure.
    """
    if not history:
        return None
    from detail.gui.training_curve import TrainingCurveBand

    band = TrainingCurveBand()
    band.set_history(history, total_epochs=hyperparams.get("epochs"))
    return band


class ExportRunDetailPage(_FitDetailPage):
    """One export run: where it went, which targets it covered, and how each fared (#354).

    **One request is one run**, so the interesting content is the per-target breakdown — a
    `partial` run is one that wrote some things and not others, and a single red word for the
    whole export says none of what a user needs. The outcomes table is therefore the page.

    No result widget and no `GUI_MODULES` entry: an export is not a module, `ExportDialog` is its
    launch surface, and registering one would put it in the processing launch picker.
    """

    def __init__(self, experiment: Experiment, run_id: str, live=None, on_deleted=None):
        self._experiment = experiment
        self._run_id = run_id
        self._on_deleted = on_deleted
        super().__init__(f"Export: {run_id}", live=live)

    def _record(self) -> dict:
        return self._experiment.export_record(self._run_id)

    def _render(self, record: dict) -> None:
        holder = QWidget()
        form = QFormLayout(holder)
        form.addRow("Status:", QLabel(str(record.get("status"))))
        error = self._error_label(record)
        if error is not None:
            form.addRow("Error:", error)
        # `subject_id` already carries the kind ("run_004"), so prefixing `subject_kind` reads
        # "run run_004". The id is what you would search for; the kind is not news.
        form.addRow("From:", QLabel(str(record.get("subject_id") or "—")))
        form.addRow(
            "Targets:", QLabel(", ".join(record.get("target_ids") or []) or "—")
        )
        self._add(holder)
        self._add_notes("export_run", self._run_id)

        # Composed here from the record's `{root, rel}` + folder name, the way a referenced
        # video's path is: the record holds no machine path (Tenet 6), and "where are my files"
        # is the one question this page exists to answer. The bare folder name is not an answer.
        #
        # **Outside the form**, because a wrapped `QLabel` in a `QFormLayout` reports a
        # one-line height and the row clips the rest — a path is the one field here long enough
        # to wrap, and it was losing its first and last lines. Full width, where it has room.
        destination = exporters.destination_of(record)
        # Tense follows the status. "Wrote to" over a running export is a claim about work that
        # has not happened — the same shape as a status written before the thing that justifies
        # it, in one word.
        running = record.get("status") in ACTIVE_STATUSES
        # One row per target, which is what makes a multi-target export legible while it runs —
        # the aggregate bar gives every target an equal share, so the rows are where the truth
        # is. `heading=` is a parameter on `set_units` for exactly this: an export's units are
        # targets, not recordings.
        self._panel.set_units(run_units.export_units(record), heading="targets")
        self._add(_heading("Writing to" if running else "Wrote to"))
        where = QLabel(str(destination or record.get("folder", "—")))
        where.setFont(mono_font())
        where.setWordWrap(True)
        self._add(where)

        outcomes = record.get("outcomes") or []
        if outcomes:
            self._add(_heading("What each target did"))
            failed = any(o.get("status") == "failed" for o in outcomes)
            # "why" only when something went wrong: on a clean run it is a wide empty stripe,
            # and a column blank in the ordinary case teaches people to ignore it in the case
            # it exists for.
            headers = ["target", "status", "files"] + (["why"] if failed else [])
            rows = [
                (
                    str(o.get("target", "")),
                    str(o.get("status", "")),
                    str(len(o.get("files") or [])),
                    *([str(o.get("error", ""))] if failed else []),
                )
                for o in outcomes
            ]
            # Not navigable: a target row opens nothing, and the cue is a claim (CLAUDE.md).
            self._add(_table(headers, rows))
        elif record.get("files"):
            form.addRow("Wrote:", QLabel(f"{len(record['files'])} file(s)"))

        if destination is not None and destination.is_dir():
            open_folder = QPushButton("Open folder")
            open_folder.clicked.connect(
                lambda: open_external.open_local_path(self, destination)
            )
            self._add(open_folder)

        if record.get("status") in ACTIVE_STATUSES:
            cancel = QPushButton("Cancel export")
            cancel.clicked.connect(self._cancel)
            self._add(cancel)
        else:
            delete = QPushButton("Delete export record")
            delete.clicked.connect(self._delete)
            self._add(delete)

    def _cancel(self) -> None:
        _confirm_and_cancel(
            self,
            "Cancel export",
            f"Cancel export {self._run_id}? Whatever it has already written is kept.",
            lambda: self._experiment.cancel_export(self._run_id),
            self._build,
        )

    def _delete(self) -> None:
        # Names what is *not* deleted: the exported files are in the user's own folder, and a
        # confirmation that let someone believe otherwise would be the wrong kind of reassuring.
        confirm = QMessageBox.question(
            self,
            "Delete export record",
            f"Delete the record of export {self._run_id}? The exported files themselves are "
            "in the folder you chose and are not touched.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_export(self._run_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete export record", str(exc))
            return
        if self._on_deleted is not None:
            self._on_deleted()


class ModelDetailPage(_FitDetailPage):
    """One trained model: lineage, class list, weight paths, final metrics."""

    def __init__(
        self, experiment: Experiment, model_id: str, live=None, on_deleted=None
    ):
        self._experiment = experiment
        self._model_id = model_id
        self._on_deleted = on_deleted
        super().__init__(f"Model: {model_id}", live=live)

    def _record(self) -> dict:
        return self._experiment.model_record(self._model_id)

    def _render(self, record: dict) -> None:
        holder = QWidget()
        form = QFormLayout(holder)
        weights = record.get("weights", {}) or {}
        metrics = record.get("metrics", {}) or {}
        classes = [c.get("name") for c in record.get("classes", [])]
        form.addRow("Name:", QLabel(str(record.get("name", ""))))
        form.addRow("Status:", QLabel(str(record.get("status"))))
        if record.get("stopped_at_epoch") is not None:
            # A finished model that did not run its course says so — a 5-epoch model and a
            # 100-epoch one otherwise read the same.
            form.addRow(
                "Stopped early:",
                QLabel(f"at epoch {record['stopped_at_epoch']} (Stop and keep)"),
            )
        error = self._error_label(record)
        if error is not None:
            form.addRow("Error:", error)
        form.addRow("Where:", QLabel(run_cost.compute_label(record.get("compute"))))
        duration_s = record.get("duration_s")
        if duration_s is not None:
            form.addRow("Duration:", QLabel(format_duration(duration_s)))
        cost = record.get("estimated_cost_usd")
        if cost is not None:
            form.addRow("Cost:", QLabel(f"${cost:.2f}"))
        form.addRow("Training framework:", QLabel(str(record.get("trainer_id", ""))))
        form.addRow("Dataset:", QLabel(str(record.get("dataset_id", ""))))
        form.addRow("Family:", QLabel(str(record.get("model_family", "—"))))
        form.addRow("Model size:", QLabel(str(record.get("base_model", "—"))))
        trained_with = QLabel(api_models.training_summary(record) or "—")
        trained_with.setWordWrap(True)
        form.addRow("Trained with:", trained_with)
        form.addRow("Classes:", QLabel(", ".join(classes) if classes else "—"))
        form.addRow("Weights:", QLabel(str(weights.get("best", "—"))))
        form.addRow(
            "Metrics:",
            QLabel("  ".join(f"{k}={v}" for k, v in metrics.items()) or "—"),
        )
        self._add_training_curve(record)
        self._add(holder)
        self._add_notes("model", self._model_id)
        if record.get("status") in ACTIVE_STATUSES:
            # Two buttons, because there are two things someone wants and they are not the same
            # thing. "The metrics look good, that will do" keeps the model; "this is wrong, make
            # it stop" does not. One button called Cancel offered only the second and did the
            # first badly — the kill lands mid-epoch, so the weights are on disk and the record
            # never learns about them.
            stop = QPushButton("Stop and keep")
            stop.setToolTip(
                "Finish the epoch in progress, then save the best weights so far. "
                "The run may take a few minutes to reach that point."
            )
            stop.clicked.connect(self._stop)
            self._add(stop)
            cancel = QPushButton("Cancel model")
            cancel.setToolTip("Stop now and discard this training run.")
            cancel.clicked.connect(self._cancel)
            self._add(cancel)
        else:
            if record.get("status") in RESUMABLE_TRAINING_STATUSES:
                resume = QPushButton("Resume training")
                resume.clicked.connect(self._resume)
                self._add(resume)
            if record.get("status") == "complete":
                promote = QPushButton("Promote to global")
                promote.clicked.connect(self._promote)
                self._add(promote)
            delete = QPushButton("Delete model")
            delete.clicked.connect(self._delete)
            self._add(delete)

    def _cancel(self) -> None:
        _confirm_and_cancel(
            self,
            "Cancel model",
            f"Cancel model {self._model_id}? Its training worker will be stopped and this "
            "run discarded. To keep the weights it has reached, use Stop and keep.",
            lambda: self._experiment.cancel_model(self._model_id),
            self._build,
        )

    def _add_training_curve(self, record: dict) -> None:
        """The curve, when there is a history to draw and a reason to look at one.

        Shown for a run in progress (the decision it exists for) and for a finished one (how the
        score got where it did), and skipped entirely when the history is empty — a plot of
        nothing is the "renders the frame of the thing it does not have" failure, and an imported
        model has no epochs at all rather than zero of them.
        """
        self._add_training_curve_for(self._model_id, record.get("hyperparams") or {})

    def _on_progress(self, event) -> None:
        self._epoch_finished(getattr(event, "epoch", None))

    def _stop(self) -> None:
        """Ask training to stop at its next epoch and keep what it has.

        No confirmation dialog: it is not destructive, and it is reversible in the only sense
        that matters — Resume training picks the run back up.

        The status line says what is happening, because the answer is *not* immediate: the flag
        is seen at the next epoch boundary, which is minutes on real data. A button that goes
        quiet for four minutes is the exact reading this whole issue is about, so the wait is
        announced rather than left to be inferred.
        """
        self._experiment.stop_model(self._model_id)
        self._panel.status.setText(
            "Stopping after this epoch — the best weights so far will be saved."
        )
        self._panel.status.setVisible(True)

    def _promote(self) -> None:
        """Copy this trained model into the machine-global catalog (usable in any
        experiment) with its provenance — off the UI thread (§12)."""
        current = str(self._experiment.model_record(self._model_id).get("name", ""))
        name, ok = QInputDialog.getText(
            self,
            "Promote to global",
            "Name for the global model (usable in any experiment):",
            text=current,
        )
        if not ok:
            return
        run_busy(
            self,
            lambda: self._experiment.promote_model(
                self._model_id, name=name.strip() or None
            ),
            lambda gid: QMessageBox.information(
                self, "Promote to global", f"Registered global model: {gid}"
            ),
            label="Promoting…",
            on_error=lambda exc: QMessageBox.critical(
                self, "Promote to global", str(exc)
            ),
        )

    def _resume(self) -> None:
        try:
            run = self._experiment.resume_training(self._model_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Resume training", str(exc))
            return
        self._start_live(run)  # stream the resumed run on this page

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete model",
            f"Delete model {self._model_id} and its weights? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_model(self._model_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete model", str(exc))
            return
        if self._on_deleted is not None:
            self._on_deleted()


def _stage_suffix(stage: dict) -> str:
    """`` · remote: tillicum · 12m30s · $1.50`` appended to a distill stage's line —
    or ``""`` for a stage with none of these yet (still pending, or a legacy run
    from before this field existed). § run location, duration, cost."""
    bits = []
    if stage.get("compute"):
        bits.append(run_cost.compute_label(stage["compute"]))
    if stage.get("duration_s") is not None:
        bits.append(format_duration(stage["duration_s"]))
    if stage.get("estimated_cost_usd") is not None:
        bits.append(f"${stage['estimated_cost_usd']:.2f}")
    return "  ·  " + "  ·  ".join(bits) if bits else ""


class DistillRunDetail(_FitDetailPage):
    """One distill run (§12): the two stages (label → train) + their produced dataset /
    model, with a live progress bar when launched. Resume a failed/cancelled run; delete
    prunes only the run record (its dataset + model stay)."""

    _LAG_REREAD_S = 1.0

    def __init__(
        self,
        experiment: Experiment,
        run_id: str,
        live=None,
        on_deleted=None,
        on_open_model=None,
    ):
        self._experiment = experiment
        self._run_id = run_id
        self._on_deleted = on_deleted
        self._on_open_model = on_open_model
        self._phase: str | None = None
        self._live_progress = None
        self._record_cache: dict | None = None
        self._last_lag_reread = float("-inf")
        super().__init__(f"Distill: {run_id}", live=live)

    def _record(self) -> dict:
        return self._experiment.distill_run_record(self._run_id)

    def _on_tick(self, permille: int, status: str) -> None:
        """Track the stage rows to the run, not to the moment the page opened.

        The body is built once and (in the base) rebuilt only on terminal status, so the
        stage rows kept showing the launch-time snapshot — "Label: pending · dataset —" —
        for the whole run, which on a 90-minute distill is the entire time anyone looks at
        it. The coordinator rewrites the record as each stage starts and finishes, so
        rebuild when the phase changes: twice a run, not once a tick.
        """
        super()._on_tick(permille, status)
        phase = status.split("(")[0].strip()
        if phase != self._phase:
            self._phase = phase
            self._build()

    def _on_progress(self, event) -> None:
        """Move the live stage's row without rebuilding the page.

        `_on_tick` already rebuilds the whole form at each phase change (twice a run);
        redrawing it on every progress tick — several times a second — is the widget-teardown
        cost this package has already measured (`unit_rows.py`'s docstring). The row updates in
        place instead, off the record `_render` cached at the last rebuild — that record's
        stage statuses don't change mid-phase, only the live one's fraction does.

        **Except inside the coordinator's announce-then-launch gap.** It emits a phase before
        it writes that stage ``running`` (with the stage's dataset/model id) — seconds later
        for labelling — and a rebuild landing in the gap cached a record whose stage is still
        ``pending``. Trusted until the next phase change, that hid the curve, Stop and keep and
        the link to the model for the entire training stage. So while the cache lags the live
        event, re-read — at most once per `_LAG_REREAD_S`, because the stage write is
        best-effort and may not land until the next one.
        """
        self._live_progress = event
        if (
            self._record_cache is not None
            and run_units.distill_record_lags(self._record_cache, event)
            and time.monotonic() - self._last_lag_reread >= self._LAG_REREAD_S
        ):
            self._last_lag_reread = time.monotonic()
            self._build()  # _render re-caches the record and redraws the rows
            return
        self._refresh_units(self._record_cache)
        if getattr(event, "phase", None) == "training":
            # The coordinator reports training as epoch / total_epochs, so this moves once an
            # epoch — the distill page's "another epoch is done".
            self._epoch_finished(getattr(event, "inner_fraction", None))

    def _refresh_units(self, record: dict | None) -> None:
        """Draw the stage rows while the run is going; hidden once it's over.

        The Label:/Train: fields below already say the finished state — terminal status, the
        dataset/model id produced — which is better than a row for it. Live: what stage is
        going and how far. Finished: what came out. Same rule as `RunDetailPage._refresh_units`.
        """
        if record is None:
            return
        # A keypoint setup waiting for its reference clicks is not over: its rows say which
        # stages are done and which one is waiting on the person reading this page.
        if (
            record.get("status") not in ACTIVE_STATUSES
            and record.get("status") != AWAITING_REFERENCES
        ):
            self._panel.set_units([])
            return
        self._panel.set_units(
            run_units.distill_units(record, self._live_progress), heading="Stages"
        )

    def _render(self, record: dict) -> None:
        self._record_cache = record
        self._refresh_units(record)
        holder = QWidget()
        form = QFormLayout(holder)
        stages = record.get("stages") or {}
        label = stages.get("label") or {}
        train = stages.get("train") or {}
        form.addRow("Name:", QLabel(str(record.get("name", ""))))
        form.addRow("Status:", QLabel(str(record.get("status"))))
        error = self._error_label(record)
        if error is not None:
            form.addRow("Error:", error)
        duration_s = record.get("duration_s")
        if duration_s is not None:
            form.addRow("Duration:", QLabel(format_duration(duration_s)))
        total_cost = record.get("total_estimated_cost_usd")
        if total_cost is not None:
            form.addRow("Total cost:", QLabel(f"${total_cost:.2f}"))
        form.addRow(
            "Label:",
            QLabel(
                f"{label.get('status', '—')}  ·  dataset {label.get('dataset_id') or '—'}"
                f"{_stage_suffix(label)}"
            ),
        )
        model_id = train.get("model_id")
        # The stage records the model id; show the name the user gave it too (§11).
        model_name = self._experiment.model_names().get(model_id) if model_id else None
        shown = (
            f"{model_name} · {model_id}"
            if model_name and model_name != model_id
            else (model_id or "—")
        )
        form.addRow(
            "Train:",
            QLabel(
                f"{train.get('status', '—')}  ·  model {shown}{_stage_suffix(train)}"
            ),
        )
        # A run started without the opt-in has nothing to show, and neither does one recorded
        # before the option existed (no `config` key at all).
        config = record.get("config") or {}
        upload_target = (
            config.get("stage_video_target") if config.get("stage_videos") else None
        )
        if upload_target:
            from detail.gui.staging import UploadStatusLabel

            form.addRow("Videos:", UploadStatusLabel(self._experiment, upload_target))
        if "register" in stages:
            # A keypoint distill's third stage (§12 *Distill keypoints*): the student joins the
            # catalog under `role: keypoints`, so the keypoint method picker can offer it.
            register = stages.get("register") or {}
            form.addRow(
                "Register:",
                QLabel(
                    f"{register.get('status', '—')}  ·  "
                    + (
                        "student in the keypoint method picker"
                        if register.get("status") == "complete"
                        else "role keypoints, in this experiment's models"
                    )
                ),
            )
        self._add(holder)
        self._add_notes("distill", self._run_id)
        # "running", not ACTIVE_STATUSES: a *pending* training has no epochs and no
        # curve, so a link offering to show one would open an empty page.
        training_started = train.get("status") in ("complete", "running")
        if model_id and training_started:
            # Below the distill status, not on a page you have to navigate to: the curve and
            # Stop and keep are what someone watching a distill run actually wants, and having
            # to leave this page to see whether training is stuck was the complaint. Shares
            # `_training_curve_widget` and `Experiment.stop_model` with `ModelDetailPage`
            # verbatim — the same widget and the same call, not a second implementation of
            # either (this package's most-repeated class is two pages answering one question
            # with two mechanisms).
            self._add_training_status(model_id)
        if self._on_open_model is not None and model_id and training_started:
            # The rest of the model page — rename, Promote to global…, weights, full lineage —
            # this run detail only manages the run record and (above) the model's live
            # training, not full model management.
            open_model = QPushButton(
                "Open trained model →"
                if train.get("status") == "complete"
                else "Open training →"
            )
            open_model.clicked.connect(lambda: self._on_open_model(model_id))
            self._add(open_model)
        if upload_target:
            uploads = QPushButton("Background uploads…")
            uploads.setToolTip(
                "Watch, stop or resume the videos being copied to the cluster."
            )
            uploads.clicked.connect(self._background_uploads)
            self._add(uploads)
        if record.get("status") == AWAITING_REFERENCES:
            self._add_examples_step(record)
            cancel = QPushButton("Cancel setup")
            cancel.clicked.connect(self._cancel)
            self._add(cancel)
            delete = QPushButton("Delete distill run")
            delete.clicked.connect(self._delete)
            self._add(delete)
        elif record.get("status") in ACTIVE_STATUSES:
            cancel = QPushButton("Cancel distill")
            cancel.clicked.connect(self._cancel)
            self._add(cancel)
        else:
            if record.get("status") in ("failed", "cancelled"):
                resume = QPushButton("Resume distill")
                resume.clicked.connect(self._resume)
                self._add(resume)
            delete = QPushButton("Delete distill run")
            delete.clicked.connect(self._delete)
            self._add(delete)

    def _background_uploads(self) -> None:
        from detail.gui.staging import BackgroundUploadsDialog

        BackgroundUploadsDialog(self._experiment, parent=self).exec()

    # --- a keypoint setup waiting for its example set (§26, #717) ------------------- #
    def _add_examples_step(self, record: dict) -> None:
        """The step only a person can do: mark an example set on the teacher's tracks (or
        choose one already marked on this animal), then continue the run with it."""
        from detail.api.experiment import setup_class_names, setup_teacher_tracks
        from detail.configs.keypoint_dataset import KeypointDatasetConfig
        from detail.gui._pydantic_form import optional_spin
        from detail.gui.class_example_sets import ClassExampleSets
        from detail.gui.example_set_picker import ExampleSetPicker
        from detail.gui.keypoint_setup import method_default_cutoff

        # With several animals the teacher tracks are the merged ones, and each class gets
        # its own set (§26 *One example set per detection class*).
        tracks = setup_teacher_tracks(record)
        animals = setup_class_names(record)
        names = self._experiment.recording_names()
        what = (
            f"one example set per animal ({', '.join(animals)}), each a few frames with "
            "every keypoint placed on that animal"
            if animals
            else "an example set on them (a few frames with every keypoint placed)"
        )
        note = QLabel(
            f"The teacher recordings are tracked. Mark {what}, or choose sets already "
            "marked, then continue: the teacher labels frames with them and the fast model "
            "trains. Teacher recordings: "
            + ", ".join(names.get(r, r) for r in tracks)
            + "."
        )
        note.setWordWrap(True)
        self._add(note)
        holder = QWidget()
        form = QFormLayout(holder)
        self._setup_animals = animals
        self._examples = ExampleSetPicker(self._experiment, sources=tracks)
        self._class_sets = ClassExampleSets(
            experiment=self._experiment, checkable=False, sources=tracks
        )
        self._class_sets.set_classes(animals, labels=self._experiment.renamed_classes())
        if animals:
            self._examples.hide()
            form.addRow(self._class_sets)
        else:
            self._class_sets.hide()
            form.addRow(self._examples)
        method_id = (record.get("config") or {}).get("keypoints_model_id") or ""
        self._setup_default_cutoff = method_default_cutoff(self._experiment, method_id)
        self._setup_method = method_id
        self._setup_cutoff = optional_spin(
            KeypointDatasetConfig,
            "score_cutoff",
            (
                f"the method's default ({self._setup_default_cutoff:g})"
                if self._setup_default_cutoff is not None
                else "required: set a value"
            ),
        )
        given = ((record.get("config") or {}).get("keypoint_dataset") or {}).get(
            "score_cutoff"
        )
        if given is not None and given != self._setup_default_cutoff:
            self._setup_cutoff.setValue(given)
        self._setup_test = QPushButton("Test cut-off…")
        self._setup_test.setToolTip(
            "See the teacher's points on a sample of the teacher recordings, slide the "
            "cut-off, and check it against your example clicks in pixels."
        )
        self._setup_test.clicked.connect(self._test_setup_cutoff)
        cutoff_row = QWidget()
        cutoff_layout = QHBoxLayout(cutoff_row)
        cutoff_layout.setContentsMargins(0, 0, 0, 0)
        cutoff_layout.addWidget(self._setup_cutoff, 1)
        cutoff_layout.addWidget(self._setup_test)
        form.addRow("Score cut-off:", cutoff_row)
        self._setup_config = dict(record.get("config") or {})
        self._setup_tracks = dict(tracks)
        self._add(holder)
        self._continue = QPushButton("Continue setup")
        self._continue.clicked.connect(self._continue_setup)
        self._examples.changed.connect(self._sync_continue)
        self._class_sets.changed.connect(self._sync_continue)
        self._setup_cutoff.valueChanged.connect(lambda _v: self._sync_continue())
        self._sync_continue()
        self._add(self._continue)

    def _setup_sets(self) -> dict:
        """The sets *Continue setup* and *Test cut-off…* hand on: one set, or one per
        animal for a setup of several."""
        if self._setup_animals:
            return {"class_example_sets": self._class_sets.value()}
        return {"example_set": self._examples.set_id()}

    def _setup_sets_missing(self) -> str | None:
        if self._setup_animals:
            missing = [
                name
                for name in self._setup_animals
                if name not in self._class_sets.value()
            ]
            return (
                f"Choose an example set for {', '.join(missing)}" if missing else None
            )
        if self._examples.set_id() is None:
            return "Choose an example set, or make one with New example set…"
        return None

    def _sync_continue(self) -> None:
        from detail.gui._pydantic_form import optional_value

        missing = self._setup_sets_missing()
        if missing is not None:
            reason = missing
        elif (
            self._setup_default_cutoff is None
            and optional_value(self._setup_cutoff) is None
        ):
            reason = f"Set a score cut-off: {self._setup_method} has no default"
        else:
            reason = None
        sizing.set_available(self._continue, reason is None, reason or "")
        sizing.set_available(self._setup_test, missing is None, missing or "")

    def _test_setup_cutoff(self) -> None:
        """*Test cut-off…* on the teacher tracks with the chosen set, as the setup will
        teach; *Use this cut-off* writes the value back."""
        from detail.core.dataclasses import ComputeTarget
        from detail.gui._pydantic_form import optional_value
        from detail.gui.keypoint_test import KeypointTestDialog

        cfg = self._setup_config
        compute = cfg.get("keypoints_compute")
        teacher = {
            **self._setup_sets(),
            "tracking": dict(self._setup_tracks),
            "model_id": cfg.get("keypoints_model_id"),
            "model_variant": cfg.get("keypoints_model_variant"),
            "label_compute": ComputeTarget.from_dict(compute) if compute else None,
        }
        cutoff = optional_value(self._setup_cutoff)
        dialog = KeypointTestDialog(
            self._experiment,
            teacher,
            cutoff=cutoff if cutoff is not None else self._setup_default_cutoff,
            parent=self,
        )
        if dialog.exec() and dialog.chosen_cutoff() is not None:
            self._setup_cutoff.setValue(dialog.chosen_cutoff())

    def _continue_setup(self) -> None:
        from detail.gui._pydantic_form import optional_value

        try:
            run = self._experiment.continue_keypoint_setup(
                self._run_id,
                score_cutoff=optional_value(self._setup_cutoff),
                **self._setup_sets(),
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            QMessageBox.warning(self, "Continue setup", str(exc).strip("'"))
            return
        self._start_live(run)  # stream the rest of the run on this page

    def _add_training_status(self, model_id: str) -> None:
        """The trained model's own curve + Stop and keep, inline.

        One `model_record` read for both: the curve's `total_epochs` and whether the model is
        still running are two facts off the same file, and fetching it twice on every rebuild
        is the redundant-read shape this package has been bitten by before.

        The stage names a `model_id` the moment it starts the run (`_set_stage("train",
        "running", model_id=...)`), before this page has any occasion to read it — a
        `FileNotFoundError` here means the model this stage points at isn't on disk (yet, or a
        deleted one), and there is nothing to show rather than a crashed page.
        """
        try:
            model_record = self._experiment.model_record(model_id)
        except FileNotFoundError:
            return
        self._add_training_curve_for(model_id, model_record.get("hyperparams") or {})
        if model_record.get("status") in ACTIVE_STATUSES:
            stop = QPushButton("Stop and keep")
            stop.setToolTip(
                "Finish the epoch in progress, then save the best weights so far. "
                "The run may take a few minutes to reach that point."
            )
            stop.clicked.connect(lambda _=False, mid=model_id: self._stop_training(mid))
            self._add(stop)

    def _stop_training(self, model_id: str) -> None:
        """Ask training to stop at its next epoch and keep what it has — the same call
        `ModelDetailPage._stop` makes, offered here too now that the curve is inline.

        No confirmation dialog: not destructive. `stop_model` finishes the run early rather
        than leaving it partial, so the Train: stage above completes normally at the next
        epoch boundary — there is nothing to resume afterward.
        """
        self._experiment.stop_model(model_id)
        self._panel.status.setText(
            "Stopping after this epoch — the best weights so far will be saved."
        )
        self._panel.status.setVisible(True)

    def _cancel(self) -> None:
        _confirm_and_cancel(
            self,
            "Cancel distill",
            f"Cancel distill run {self._run_id}? Its coordinator and stage worker "
            "will be stopped.",
            lambda: self._experiment.cancel_distill(self._run_id),
            self._build,
        )

    def _resume(self) -> None:
        try:
            run = self._experiment.resume_distill(self._run_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Resume distill", str(exc))
            return
        self._start_live(run)  # stream the resumed run on this page

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete distill run",
            f"Delete distill run {self._run_id}? Its dataset and trained model are kept "
            "(delete those separately).",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_distill(self._run_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete distill run", str(exc))
            return
        if self._on_deleted is not None:
            self._on_deleted()


# --------------------------------------------------------------------------- #
# Analysis runs (§16) — index + detail
# --------------------------------------------------------------------------- #
class ExperimentAnalysisPage(QWidget):
    """Analysis-runs index. Open one → ``on_open(run_id)``; launch a new run →
    ``on_launched(AnalysisRun)``."""

    #: Two notices, and the same rule the recordings table follows: an empty state is a claim
    #: about the data, so it waits until the data has been read (see ``_EMPTY_TEXT`` there).
    _EMPTY_TEXT = "(no analysis runs yet — New analysis starts one)"
    _READING_TEXT = "(reading analysis runs…)"

    def __init__(self, experiment: Experiment, on_open, on_launched=None):
        super().__init__()
        self._experiment = experiment
        self._on_launched = on_launched
        self._on_open = on_open
        self._ids: list[str] = []
        self._reading = True
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Analysis"))
        # One record read per analysis run — 81 round trips over 40 of them, ~1 s of frozen
        # window on a workspace held on a network share. The page goes up now and the rows
        # arrive when they arrive.
        # Blank while reading, not "reading…": the table's notice already says that, and the
        # two stacked read as a page that lost its place.
        self._caption = _index_caption("", rows=0)
        layout.addWidget(self._caption)
        new = QPushButton("New analysis")
        new.clicked.connect(self._new)
        layout.addWidget(new, alignment=Qt.AlignmentFlag.AlignLeft)
        self._table = IndexTable(
            ["Id", "Module", "Recordings", "Status"], empty_text=self._READING_TEXT
        )
        self._table.view.doubleClicked.connect(self._open_row)
        layout.addWidget(self._table)
        layout.addStretch()
        self._rows_worker = BackgroundCall(
            lambda: _analysis_rows(experiment), receiver=self  # off-thread
        )
        self._rows_worker.done.connect(self._apply_rows)
        self._rows_worker.failed.connect(lambda _exc: None)  # the notice stays as it is
        self._rows_worker.start()

    def _apply_rows(self, result) -> None:
        ids, rows = result
        self._ids = ids
        self._reading = False
        self._table.set_empty_text(self._EMPTY_TEXT)
        self._table.set_rows(rows, ids=self._ids)
        self._caption.setText(
            _caption_text(f"{len(rows)} analysis run(s)", rows=len(rows))
        )

    def _open_row(self, index) -> None:
        key = row_id(index)
        if key is not None:
            self._on_open(key)

    def _new(self) -> None:
        from detail.gui.launch import LaunchAnalysisDialog

        dialog = LaunchAnalysisDialog(self._experiment, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and self._on_launched:
            self._on_launched(dialog.run_handle)


def _is_mutable_module(module_id: str | None) -> bool:
    """Whether this analysis module's runs are **authored** (§21) rather than computed.

    Read from the backend ``AnalysisModule``, not the GUI module UI, because the same fact
    decides three different things — whether a run renders while still open, whether Refine
    applies, and (for the API and CLI) whether a run can be edited in place. One source, so
    they cannot disagree.

    A module id with no registered module is not mutable: an unknown module's run is a record
    the shell can still show, and guessing "authored" for it would render a workspace nothing
    can drive.

    The registry is populated explicitly rather than relied on. `ANALYSIS_MODULES` fills when
    the pipeline modules are imported, and this page can be reached from the rail without any
    launch dialog having done that — leaving every module reading as not-authored, so an open
    run would show no results and Refine would appear on one that cannot be refined. Both
    silent. The import is idempotent and the modules are already loaded by the time a run
    detail page opens in practice.
    """
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    module = ANALYSIS_MODULES.get(module_id or "")
    return bool(getattr(module, "mutable", False))


def _analysis_phases(module_id: str | None) -> tuple[str, ...]:
    """The module's declared phase list (§16, issue #393), or ``()`` for an unknown module.

    Same registry-population concern as ``_is_mutable_module`` just above (and the same fix):
    this page can be reached without any launch dialog having imported the pipeline modules.
    """
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    module = ANALYSIS_MODULES.get(module_id or "")
    return tuple(getattr(module, "phases", ()) or ())


class MeasuresPanel(QWidget):
    """The whole-recording measures of one analysis run's behavior artifact (§23).

    One panel for every producer that emits the artifact, instead of three that drift. The
    authored workspaces carry it as a MEASURES tab — it is per *artifact*, not per behavior or
    label, so it gets a tab of its own rather than sitting under AUDIT, where it read as part of
    the review — and the run page shows it directly for a producer with no workspace.

    Values carry mixed units — nanoseconds, counts, ratios, per-second rates — so the unit
    comes from the registry and gets its own column. **Durations are shown in seconds**: the
    artifact stores nanoseconds (Tenet 8) and a mean bout rendered raw reads `2e+07`, which is
    a number nobody can act on. The CSV export keeps nanoseconds, which is the machine's format;
    this is the screen's.

    Reading it means a frame view per unit, so it runs off the UI thread.
    """

    def __init__(self, experiment, run_id: str) -> None:
        super().__init__()
        self._experiment = experiment
        self._run_id = run_id
        self._started = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # The CSV the CLI's `measures export` writes, from the page that shows the numbers:
        # the table was readable here and only copyable from a terminal. Above the table, so
        # the table runs to the bottom of the tab.
        export_row = QHBoxLayout()
        self._export = QPushButton("Export CSV…")
        self._export.clicked.connect(self._choose_export_folder)
        set_available(self._export, False, "Measuring…")
        export_row.addWidget(self._export)
        self._exported = QLabel("")
        self._exported.setWordWrap(True)
        export_row.addWidget(self._exported, 1)
        layout.addLayout(export_row)
        self._body = QLabel("Measuring…")
        layout.addWidget(self._body)

    #: Shown while **Export CSV…** is available; ``set_available`` clears a tooltip on enable.
    _EXPORT_HELP = (
        "Save these measures as measures.csv (one row per recording, label and measure; "
        "durations in nanoseconds), with frames.csv and measures.yaml, into a folder you "
        "choose."
    )

    def _offer_export(self) -> None:
        set_available(self._export, True, "")
        self._export.setToolTip(self._EXPORT_HELP)

    def _choose_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Export measures to", str(self._experiment.dir)
        )
        if folder:
            self.export_to(Path(folder))

    def export_to(self, folder: Path) -> None:
        """Write this run's measures into ``folder / <run id>_measures``, off the UI thread.

        A subfolder per run, so exporting a second run into the same folder never overwrites
        the first's ``measures.csv``.
        """
        out = Path(folder) / f"{self._run_id}_measures"
        experiment, run_id = self._experiment, self._run_id
        set_available(self._export, False, "Exporting…")
        self._exported.setText("")
        run_progress(
            self,
            lambda emit: experiment.measures(run_id).export(out, progress=emit),
            self._exported_into,
            label="Exporting measures…",
            on_error=self._export_failed,
        )

    def _exported_into(self, out) -> None:
        self._offer_export()
        self._exported.setText(
            f"Wrote measures.csv, frames.csv and measures.yaml to {out}"
        )

    def _export_failed(self, exc: BaseException) -> None:
        self._offer_export()
        self._exported.setText(f"Could not export: {exc}")

    def showEvent(self, event) -> None:
        """Measure when the panel is first shown, not when it is constructed.

        Constructing a page must not start disk work: the run-detail page is built in tests
        (and could be built speculatively) without ever being displayed, and a read left
        running against a directory the caller then removes is a race with no owner. Deferring
        to the first show also means a page nobody looks at costs nothing.
        """
        super().showEvent(event)
        if self._started:
            return
        self._started = True
        run_progress(self, self._compute, self._show_rows, label="Measuring…")

    def _compute(self, emit):
        # The *one* number-rendering rule, shared with the comparison report and its exports:
        # a duplicate here printed `0.317 fraction` on this page and `31.7 %` on the comparison
        # page for the same measure, because the copy predated the fraction branch. Imported in
        # the method so opening the GUI does not pull the report layer in.
        from detail.analysis import comparison_report
        from detail.analysis.measures import MEASURES

        # `emit` reports the one long step: frame views an edit deferred, rebuilt before reading.
        rows = self._experiment.measures(self._run_id).table(progress=emit)
        return [
            (
                row["recording_id"],
                str(row["subject"]),
                row["label"] or "(unnamed)",
                MEASURES[row["measure"]].label,
                *comparison_report.shown_value(
                    row["value"], MEASURES[row["measure"]].unit
                ),
                str(row["n"]),
            )
            for row in rows
            # A pairwise measure keys on an ordered pair; showing only the from-label here
            # would print the same label several times with different numbers.
            if not row["label_to"]
        ]

    def _show_rows(self, rows) -> None:
        layout = self.layout()
        layout.removeWidget(self._body)
        self._body.deleteLater()
        if not rows:
            self._body = QLabel(
                "(this run's artifact holds no measurable instances yet)"
            )
        else:
            self._body = _table(
                ["Recording", "Subject", "Label", "Measure", "Value", "Unit", "n"], rows
            )
            # `_table` gives the slack to its last data column, a one-digit count; handed to
            # Label instead it was 678px at 1440 wide, the numbers a gulf away from their names.
            # Every column gets an equal share.
            spread_slack(self._body)
            # The tab's whole content, so it is as tall as its rows up to the tab — not the
            # display's row budget, which on a 1440x847 window stopped it at y=700 with 118px
            # of tab empty under it.
            _fit_table_height(self._body, max_rows=len(rows))
        layout.addWidget(self._body)  # under the export row
        if rows:
            self._offer_export()
        else:
            set_available(self._export, False, "Nothing to export yet.")


#: Why a view card is greyed while its run is still computing.
VIEW_WAITS_FOR_RUN = "Results appear when the run finishes."

#: Why a view card is greyed when the stored results no longer match the run.
VIEW_WAITS_FOR_REBUILD = (
    "These results are out of date. Rebuild them (above) to open a view."
)


class ViewCard(QPushButton):
    """A button whose face is a headline and two plain lines: ``Interpret ›`` in bold, two sizes
    up, then what the view is for and its status line.

    No stylesheet (`requirements.py`): the headline is a ``QLabel`` with an emphasis ``QFont``,
    laid out inside the button with the other two. The labels let every mouse event through, so
    the whole card is one click target, and the button keeps focus, Space / Enter and its
    disabled look (a disabled button greys its children). ``text()`` is the three lines, so
    everything that reads a button by its text — the requirement checks, tests — still can.
    """

    def __init__(self, headline: str, purpose: str, status: str, parent=None) -> None:
        super().__init__(parent)
        self._lines = (headline, purpose, status)
        self.setAccessibleName(headline)
        self.setAccessibleDescription(" ".join(x for x in (purpose, status) if x))
        box = QVBoxLayout(self)
        margin = lines(0.6)
        box.setContentsMargins(margin, margin, margin, margin)
        box.setSpacing(2)
        self.headline = QLabel(headline)
        self.headline.setFont(emphasis_font(self.headline, steps=2))
        self.purpose = QLabel(purpose)
        self.purpose.setWordWrap(True)
        # Always present, blank where there is no status, so the cards in a row match.
        self.status = QLabel(status or " ")
        for label in (self.headline, self.purpose, self.status):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            box.addWidget(label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def text(self) -> str:
        return "\n".join(self._lines)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        return self.layout().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        return self.layout().minimumSize()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 — Qt's name
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 — Qt's name
        return self.layout().totalHeightForWidth(width)


class AnalysisRunDetailPage(_FitDetailPage):
    """One analysis run (§16): a **landing page**, and its result views full window.

    The landing page is the run's facts, Notes, the live phase rows while it runs, its snapshots,
    and one card per result view (`registry.result_views`) saying what the view is for and one
    status line. A card opens its view to fill the window under one row of tabs, one per view,
    to switch straight to another. Every analysis module gets this from this one class — a
    module with several views builds a `MultiViewWorkspace`, any other result widget is one view
    called "Results", which gets no tab: there is nothing to switch to, and the path line already
    names it.

    The way back is the shell's path line, not a button in the row: with a view open it reads
    ``<experiment> › <run_id> › <view>``, and ``<run_id>`` is a link to the landing page. The page
    says what to append through `context_view` / `context_view_changed` and the shell calls
    `close_view` when the link is clicked (`MainWindow._render_context`). A ``‹ <run_id>``
    button sat at the head of the row and read as a fifth view (user, review round 15).

    The views are built **eagerly**, when the page is, exactly as the inline result was: the
    landing page only shows or hides the host. So a hand-off inside a workspace (AUDIT's "Open in
    INTERPRET") still works, and moves the tabs and the path line through ``view_changed``.

    The link is an in-page mode switch, like snapshot mode's "Back to live run": the shell keeps
    no history (`docs/gui_design.md` §3.1). Esc is not bound to it, because AUDIT already uses
    Esc for its own way back.
    """

    #: The open view changed, or the page went back to its landing page: the shell redraws its
    #: path line from `context_view`.
    context_view_changed = pyqtSignal()

    #: The link's tooltip on the shell's path line while a view is open.
    BACK_TIP = "Back to this run's page: its details, notes and the list of views."

    def __init__(
        self,
        experiment: Experiment,
        run_id: str,
        live=None,
        on_deleted=None,
        on_new=None,
        on_open=None,
    ):
        self._experiment = experiment
        self._run_id = run_id
        self._on_deleted = on_deleted
        self._on_new = on_new
        #: ``on_open(run_id)`` shows another analysis run — the copy *Duplicate* just made.
        self._on_open = on_open
        # Which snapshot this page is showing, or None for the live run (§21).
        #
        # An attribute rather than something the click handler holds, because `_build()` re-renders
        # a shown page — a refresh from anywhere would otherwise drop the page silently back to the
        # live run: same layout, different numbers, no event. That is this feature's own failure
        # mode arriving through a side door, so the mode is page state and a test pins it.
        self._viewing: str | None = None
        # The most recent live event and the module's declared phase list (§16, issue #393),
        # so `_on_progress` can redraw the phase rows on every tick without re-reading the
        # record — the record's `module_id` does not change while a run is going, so
        # `_refresh_units` resolves `_phases` once, at build time, rather than every tick.
        self._live_progress = None
        self._phases: tuple[str, ...] = ()
        # Which result view fills the window, or None for the landing page. Page state for the
        # reason `_viewing` is: a rebuild (a recording finishing, a snapshot opened) must not
        # throw the user back to the landing page.
        self._open_view: str | None = None
        self._result: QWidget | None = None
        self._views: list[ResultView] = []
        #: One tab per view, or None when there is one view (or no result) to show.
        self._view_tabs: QTabBar | None = None
        #: The landing page's cards by view name, this build.
        self._cards: dict[str, QPushButton] = {}
        super().__init__(f"Analysis: {run_id}", live=live)

    def _content_widget(self) -> QWidget:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)
        self._landing = QWidget()
        self._view_host = QWidget()
        host = QVBoxLayout(self._view_host)
        # The window is the view's: the side margins match the landing page, the top and bottom
        # are spent on the view instead of on gaps around one row.
        margins = host.contentsMargins()
        host.setContentsMargins(margins.left(), 4, margins.right(), 0)
        host.setSpacing(4)
        self._view_row = QHBoxLayout()
        host.addLayout(self._view_row)
        # The page's one Notes box comes here while a view is open (`_place_notes_box`): the
        # shell's Notes button is bound to it, and on the hidden landing page it opened unseen.
        self._view_notes = QVBoxLayout()
        self._view_notes.setContentsMargins(0, 0, 0, 0)
        host.addLayout(self._view_notes)
        self._view_slot = QVBoxLayout()
        self._view_slot.setContentsMargins(0, 0, 0, 0)
        host.addLayout(self._view_slot, 1)
        self._stack.addWidget(self._landing)
        self._stack.addWidget(self._view_host)
        return self._landing

    def _record(self) -> dict:
        # One `job.json` read before the record is believed. The experiment-wide sweep runs
        # when the experiment is opened, which is too early for a run that fails *during* the
        # session — and this page is where an optimistic record turns into an offer to
        # interpret a run that died. See `core/dataclasses.py::reconciled_status`.
        self._experiment.reconcile_analysis_run(self._run_id)
        record = self._experiment.analysis_record(self._run_id)
        # An authored run writes `open` before its job ends (§21). While the job is still
        # working, show it as running: building the workspace then reads an artifact that is
        # not there yet. Display only; the record on disk is the runner's to write.
        if record.get("status") not in ACTIVE_STATUSES and (
            self._experiment.analysis_job_active(self._run_id)
        ):
            record = {**record, "status": "running"}
        return record

    def _on_progress(self, event) -> None:
        """Redraw the phase rows from the live event — no record re-read (`_refresh_units`
        already resolved `_phases` for this build; see `__init__`), the same cost concern
        `RunDetailPage._on_progress` names for the recording rows.
        """
        self._live_progress = event
        self._panel.set_units(
            run_units.analysis_units(event, self._phases), heading="Phases"
        )

    def _refresh_units(self, record) -> None:
        """Draw the phase rows — while the run is going, and not after.

        Once a run is over, "Results" above says what it produced and a phase row has nothing
        left to add — the same "two surfaces, one claim" reason `RunDetailPage` hides its rows
        for a finished run rather than showing them beside the Per-recording table.
        """
        self._phases = _analysis_phases(record.get("module_id"))
        if record.get("status") not in ACTIVE_STATUSES:
            self._panel.set_units([])
            return
        self._panel.set_units(
            run_units.analysis_units(self._live_progress, self._phases),
            heading="Phases",
        )

    def _has_artifact(self) -> bool:
        """Whether this run wrote the §19 behavior artifact — the thing §23 measures."""
        from detail.storage import behavior_output

        return behavior_output.has_behavior_output(
            self._experiment.analysis_run_dir(self._run_id)
        )

    def _add_snapshots(self, module_id) -> None:
        """What has left this run, and what used it (§21).

        Shown beside the status it qualifies rather than inside the workspace: a snapshot is a
        property of the *run*, not of the interpretation being authored, and the workspace already
        carries the tabs for that.

        A run with none shows nothing at all — an empty table would suggest something is missing
        when in fact nothing has left yet.

        **Navigable only where a module can render one.** Double-clicking a row puts the page in
        snapshot mode — the same workspace, driven by the run as it was, with everything that would
        change the snapshot greyed out. A module that returns no ``snapshot_widget`` lists its
        snapshots without the cue, so a chevron never promises a view that does not exist.
        """
        run = self._experiment.authored_run(self._run_id)
        if run is None:
            return
        try:
            rows = snapshot_rows(self._experiment, run)
        except (FileNotFoundError, ValueError):
            return
        if not rows:
            return
        records = run.snapshots()
        self._snapshot_ids = [str(r.get("snapshot_id")) for r in records]
        openable = self._snapshot_ui(module_id) is not None
        self._add(_heading("Snapshots"))
        self._add(
            QLabel(
                "Frozen copies of this run's results, saved whenever a number left it — an "
                "export, a comparison, or finishing. Editing the run never changes one, which is "
                "what lets you reopen a finished analysis without invalidating a number somebody "
                "already quoted."
                + (" Double-click one to see what it was." if openable else "")
            )
        )
        table = _table(
            snapshot_columns(run, records),
            rows,
            navigable=openable,
            ids=self._snapshot_ids,
        )
        if openable:
            table.doubleClicked.connect(self._open_snapshot_row)
        # Bold the row being shown, so "which one am I on" is answered where the list is as well as
        # in the banner.
        if self._viewing in self._snapshot_ids:
            row = self._snapshot_ids.index(self._viewing)
            model = table.model()
            for column in range(model.columnCount()):
                item = model.item(row, column)
                if item is not None:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
        self._add(table)

    def _snapshot_ui(self, module_id):
        """The module UI, if it can render a snapshot read-only — else ``None``."""
        ui = ANALYSIS_GUI_MODULES.get(module_id)
        if ui is None:
            return None
        run = self._experiment.authored_run(self._run_id)
        if run is None or getattr(run, "snapshot_view_class", None) is None:
            return None
        return ui

    def _open_snapshot_row(self, index) -> None:
        snapshot_id = row_id(index)
        if snapshot_id is None:
            return
        run = self._experiment.authored_run(self._run_id)
        try:
            run.view(snapshot_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.information(self, "Snapshot", str(exc))
            return
        self._viewing = snapshot_id
        self._build()

    def _back_to_live(self) -> None:
        self._viewing = None
        self._build()

    def _add_snapshot_banner(self, record: dict) -> None:
        """Say what is on screen, and offer the one way out.

        Loud on purpose. The failure this whole feature exists to prevent is reading a number off a
        frozen copy believing it is current, and a subtle label is how that happens.
        """
        run = self._experiment.authored_run(self._run_id)
        snapshot = run.snapshot(self._viewing)
        headline = run.snapshot_interpretation(snapshot).headline
        taken = str(snapshot.get("taken_at", ""))[:16].replace("T", " ")

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.Box)
        # A band that is plainly not the page. The shade is derived from the current Window colour
        # rather than hardcoded, so it separates from the background in a light theme and in a dark
        # one; a fixed grey would vanish into one of them.
        frame.setAutoFillBackground(True)
        palette = frame.palette()
        background = palette.color(QPalette.ColorRole.Window)
        palette.setColor(
            QPalette.ColorRole.Window,
            (
                background.darker(112)
                if background.lightness() > 127
                else background.lighter(145)
            ),
        )
        frame.setPalette(palette)

        row = QHBoxLayout(frame)
        text = QLabel(
            f"Viewing {self._viewing} — pinned {taken} for {snapshot.get('reason')}. Read-only."
            + (f"\n{headline}" if headline else "")
        )
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = text.font()
        font.setBold(True)
        text.setFont(font)
        # Amber, through the shared status palette: "you are not looking at the live run" is the
        # warning state, not decoration, which is the only thing colour is spent on here.
        set_label_color(text, WARN)

        back = QPushButton("Back to live run")
        back.clicked.connect(self._back_to_live)
        # Balanced by the button's own width on the left, so the text sits in the *middle* of the
        # banner rather than in the middle of what the button leaves over.
        row.addSpacing(back.sizeHint().width())
        row.addWidget(text, 1)
        row.addWidget(back)
        self._add(frame)

    def _open_meta_information(self) -> None:
        """Read the run's YAML files off the UI thread, then show them (`gui/run_info`)."""
        run_busy(
            self,
            lambda: self._experiment.analysis_info(self._run_id),
            self._show_meta_information,
            label="Reading meta-information…",
            supersedes="meta-information",
        )

    def _show_meta_information(self, info) -> None:
        from detail.gui.run_info import RunInfoWindow

        RunInfoWindow(info, self).exec()

    def _render(self, record: dict) -> None:
        holder = QWidget()
        form = QFormLayout(holder)
        form.addRow("Module:", QLabel(str(record.get("module_id", ""))))
        if record.get("name"):
            form.addRow("Name:", QLabel(str(record["name"])))
        # In snapshot mode this describes the run, not what is on screen, so it says which.
        form.addRow(
            "Status:",
            QLabel(
                str(record.get("status"))
                + (" (of the live run)" if self._viewing else "")
            ),
        )
        error = self._error_label(record)
        if error is not None:
            form.addRow("Error:", error)
        form.addRow(
            "Recordings:",
            QLabel(
                _recording_labels(self._experiment, record.get("recording_ids", []))
            ),
        )
        self._add(holder)
        meta = QPushButton("Meta-information…")
        meta.setToolTip(
            "Every file this run keeps about itself (its record and settings, the module's "
            "own settings, where it ran) in one window, with Export."
        )
        meta.clicked.connect(self._open_meta_information)
        # Its own size at the left, not the page's width: a full-width bar read as a divider.
        meta_row = QWidget()
        row = QHBoxLayout(meta_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(meta)
        row.addStretch(1)
        self._add(meta_row)
        self._add_notes("analysis_run", self._run_id)
        self._refresh_units(record)
        # A "partial" run (some recordings failed/cancelled) still wrote CSVs for the
        # recordings that finished — its artifacts list only those — so show its output like
        # a complete run rather than offering only Delete.
        #
        # An **authored** run (§21) also has output while it is still being worked on: the
        # statistics are what the user is deciding against. Which statuses those are is the
        # *module's* to say, so this asks the flag rather than listing them — a hardcoded
        # tuple here goes stale the day a module grows a lifecycle, and nothing fails when it
        # does. `mutable` lives on the AnalysisModule so the API and CLI read the same fact.
        module_id = record.get("module_id")
        status = record.get("status")
        has_output = status in OUTPUT_STATUSES or (
            _is_mutable_module(module_id) and status in AUTHORED_STATUSES
        )
        # An artifact its own run says does not match its state is not output. §19 calls the
        # frame view a materialized view and says regenerate rather than reconcile, so the
        # page offers the regeneration instead of the numbers — showing them would be showing
        # one setting's results under another setting's heading.
        stale = has_output and self._add_stale_notice(record)
        if stale:
            has_output = False
        if self._viewing:
            self._add_snapshot_banner(record)
        self._add_snapshots(module_id)
        ui = ANALYSIS_GUI_MODULES.get(module_id)
        self._measures_in_result = False  # set by the result built below, every rebuild
        frozen_ui = self._snapshot_ui(module_id) if self._viewing else None
        result = None
        if frozen_ui is not None:
            result = frozen_ui.snapshot_widget(self._experiment, record, self._viewing)
        elif ui is not None and has_output:
            result = ui.result_widget(self._experiment, record)
            # A workspace that carries Measures as a view of its own says so by holding one.
            self._measures_in_result = bool(
                result is not None and result.findChildren(MeasuresPanel)
            )
        self._install_result(result, ui)
        if result is not None:
            self._add_view_cards(
                f"Results as of {self._viewing}" if self._viewing else "Results"
            )
        elif ui is not None and record.get("status") in ACTIVE_STATUSES:
            self._add_planned_cards(ui, VIEW_WAITS_FOR_RUN)
        elif ui is not None and stale:
            self._add_planned_cards(ui, VIEW_WAITS_FOR_REBUILD)
        # §23's numbers live in the workspaces now, as a MEASURES tab (`ClusterWorkspace`, the
        # query workspace) — a section here sat under every tab, and under AUDIT it read as part
        # of the review. Shown here only when the result built holds no panel of its own, so a
        # producer without a workspace (or a run whose workspace did not build) keeps its numbers.
        #
        # **Hidden** in snapshot mode rather than greyed: it reads the live run's artifact, and live
        # numbers under a snapshot banner are the exact confusion this feature exists to prevent.
        # "Grey out, don't hide" governs a capability that does not apply — not data that is wrong.
        if (
            has_output
            and self._has_artifact()
            and not self._viewing
            and not self._measures_in_result
        ):
            self._add(_heading("Measures"))
            self._add(MeasuresPanel(self._experiment, self._run_id))
        if has_output:
            open_folder = QPushButton("Open output folder")
            open_folder.clicked.connect(self._open_folder)
            self._add(self._live_only(open_folder))
        if record.get("status") in ACTIVE_STATUSES:
            cancel = QPushButton("Cancel analysis")
            cancel.clicked.connect(self._cancel)
            self._add(self._live_only(cancel))
        else:
            # An exact copy to try something on, the original untouched (§16).
            duplicate = QPushButton("Duplicate analysis…")
            duplicate.setToolTip(
                "Copy this run whole — fit, clusters, clips, labels and profiles — to "
                "a new run, leaving this one as it is."
            )
            duplicate.clicked.connect(lambda: self._duplicate())
            self._add(self._live_only(duplicate))
            delete = QPushButton("Delete analysis")
            delete.clicked.connect(self._delete)
            self._add(self._live_only(delete))
        # The end of a run is where you decide what to do next, and the answer is usually
        # "another one". Unlike Refine this seeds nothing — a *different* module is the common
        # case — so it needs no §25 gate: the launch dialog only offers modules this build
        # surfaces. Separated by a rule so it does not sit flush against Delete.
        if self._on_new is not None:
            self._add(_divider())
            new_run = QPushButton("Start new analysis")
            new_run.clicked.connect(self._on_new)
            self._add(new_run)

    # -- landing page and full-window views ---------------------------------- #
    def _build(self) -> None:
        # The last build's result goes with the rest of the body; until this build installs
        # one, there is no view to keep open.
        self._result = None
        self._views = []
        self._view_tabs = None
        self._cards = {}
        # The Notes box back on the landing page first, where the rebuild places it.
        self._place_notes_box(in_view=False)
        super()._build()
        self._sync_mode()

    def _install_result(self, result: QWidget | None, ui) -> None:
        """Put this build's result widget in the view host, under its row of view tabs."""
        self._result = result
        self._views = result_views(result, getattr(ui, "purpose", ""))
        self._view_tabs = None
        if result is None:
            return
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        if len(self._views) > 1:
            # Tabs, not buttons: the views are one choice of what fills the window, and a row of
            # plain buttons read as actions — the way back among them read as a fifth view.
            tabs = QTabBar()
            tabs.setExpanding(False)
            tabs.setDrawBase(False)
            for view in self._views:
                index = tabs.addTab(view.name)
                tabs.setTabToolTip(index, view.purpose)
            tabs.currentChanged.connect(self._on_tab_changed)
            layout.addWidget(tabs)
            self._view_tabs = tabs
        if self._viewing:
            # The banner is on the landing page; inside a view this is what says it is frozen.
            frozen = QLabel(f"Read-only \u00b7 {self._viewing}")
            set_label_color(frozen, WARN)
            layout.addSpacing(12)
            layout.addWidget(frozen)
        layout.addStretch(1)
        _make_selectable(row)  # built on every rebuild, after MainWindow._show ran
        # One view and a live run: nothing to switch to and nothing to say, so no row at all.
        row.setVisible(self._view_tabs is not None or bool(self._viewing))
        self._view_row.addWidget(row)
        self._view_slot.addWidget(result, 1)
        # Discarded with the rest of the body on the next build.
        self._body_widgets += [row, result]
        changed = getattr(result, "view_changed", None)
        if changed is not None:
            changed.connect(self._on_view_changed)

    def _select_tab(self, name: str) -> None:
        """Mark ``name`` as the current tab without it counting as a click."""
        tabs = self._view_tabs
        if tabs is None or name not in self.view_names():
            return
        blocked = tabs.blockSignals(True)
        tabs.setCurrentIndex(self.view_names().index(name))
        tabs.blockSignals(blocked)

    def _on_tab_changed(self, index: int) -> None:
        names = self.view_names()
        if 0 <= index < len(names) and self._view_is_open():
            self._show_view(names[index])

    def context_view(self) -> str | None:
        """What the shell's path line appends after the run id: the open view, or ``None``."""
        return self.open_view()

    def close_view(self) -> None:
        """The shell's path-line link: back to the landing page."""
        self._show_landing()

    def _add_view_cards(self, heading: str) -> None:
        """One card per view: its name, what it is for, and its status line. A card opens it."""
        self._add(_heading(heading))
        cards = []
        for view in self._views:
            status = view.status_text()
            card = self._card(view.name, view.purpose, status)
            card.clicked.connect(lambda _checked, name=view.name: self._show_view(name))
            cards.append(card)
            self._cards[view.name] = card
        self._add(self._card_grid(cards))

    def _add_planned_cards(self, ui, reason: str) -> None:
        """The cards the run will have, greyed with why — there is no result to open yet."""
        self._add(_heading("Results"))
        cards = []
        for name, purpose in ui.planned_views():
            card = self._card(name, purpose, "")
            set_available(card, False, reason)
            cards.append(card)
            self._cards[name] = card
        self._add(self._card_grid(cards))

    @staticmethod
    def _card(name: str, purpose: str, status: str) -> QPushButton:
        """A view card: the view's name as a bold headline, its purpose and status under it."""
        card = ViewCard(f"{name} \u203a", purpose, status)
        card.setToolTip(f"Open {name} to fill the window.")
        return card

    @staticmethod
    def _card_grid(cards: list[QPushButton]) -> QWidget:
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        for index, card in enumerate(cards):
            grid.addWidget(card, index // 2, index % 2)
        return holder

    def view_names(self) -> list[str]:
        return [view.name for view in self._views]

    def open_view(self) -> str | None:
        """The view filling the window, or ``None`` on the landing page."""
        return self._open_view if self._view_is_open() else None

    def _view_is_open(self) -> bool:
        return self._result is not None and self._open_view in self.view_names()

    def _show_view(self, name: str) -> None:
        self._open_view = name
        self._sync_mode()

    def _show_landing(self) -> None:
        self._open_view = None
        self._sync_mode()

    def _on_view_changed(self, name: str) -> None:
        """A hand-off inside the workspace moved the view: the tabs and the path line follow."""
        if self._view_is_open() and name in self.view_names():
            self._open_view = name
            self._select_tab(name)
            self._place_notes_box(in_view=True)
            self.context_view_changed.emit()

    def _sync_mode(self) -> None:
        """Show the landing page or the open view, whichever the page state says."""
        if not self._view_is_open():
            self._open_view = None
        open_ = self._open_view is not None
        if open_:
            show = getattr(self._result, "show_view", None)
            if show is not None:
                show(self._open_view)
            self._select_tab(self._open_view)
        front, back = (
            (self._view_host, self._landing)
            if open_
            else (self._landing, self._view_host)
        )
        # A stack is as tall as its tallest page unless the hidden one is Ignored, and the
        # landing page under a full-window view must not make the shell scroll (or the reverse).
        back.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        front.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._place_notes_box(in_view=open_)
        self._stack.setCurrentWidget(front)
        self.updateGeometry()
        self.context_view_changed.emit()

    def _place_notes_box(self, in_view: bool) -> None:
        """Move the page's one Notes box under the view row, or back to its place on the
        landing page. One box, moved rather than copied, so its text and autosave are the same
        wherever it is shown; open or folded is per place (`NotesBox.set_place`: the landing
        page, or each view), so opening it in a view leaves the landing page folded."""
        box = getattr(self, "_notes_box", None)
        if box is None or not alive(box):
            return
        index = self._body.indexOf(box)
        if index >= 0:
            self._notes_index = index
        if in_view and index >= 0:
            self._body.removeWidget(box)
            self._view_notes.addWidget(box)
        elif not in_view and index < 0:
            self._view_notes.removeWidget(box)
            self._body.insertWidget(
                getattr(self, "_notes_index", self._body.count()), box
            )
        # A move re-parents the box, which hides it; it shows again only if it was opened in
        # the place it is now in.
        box.set_place(f"view:{self._open_view}" if in_view else "")

    def _view_in_front(self) -> bool:
        stack = getattr(self, "_stack", None)
        return stack is not None and stack.currentWidget() is self._view_host

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        """With a view open, ask for nothing: the shell's `PageScroll` then gives the page the
        viewport exactly (`gui/scroll.py`), and the view fills the window instead of the page
        scrolling to the view's preferred height. The landing page asks as any page does.
        """
        if self._view_in_front():
            return QSize(super().sizeHint().width(), 0)
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        """Nothing either, with a view open — a full-window view is held to the window.

        The scroll takes the larger of the two hints, so a floor here would bring the scrolling
        back through the other door. A view whose own floor is taller than the window is
        squeezed rather than scrolled; that is the decision (review round 1, 2026-11-07), and
        the place to win the room back is the view's own rows.
        """
        if self._view_in_front():
            return QSize(super().minimumSizeHint().width(), 0)
        return super().minimumSizeHint()

    def export_scope(self):
        """The view in front answers; the landing page exports the whole run."""
        from detail.gui.export import scope_of

        if not self._view_is_open():
            return None
        return scope_of(self._result)

    def _add_stale_notice(self, record: dict) -> bool:
        """Say that the results do not match the settings, and offer the rebuild. True if shown.

        Reached when a reprojection did not finish — a transient write on a network share, a
        kill. The state moved, the artifact did not, and until this existed nothing anywhere
        said so: the run went on rendering the previous granularity's instances under the new
        granularity's heading.
        """
        from detail.storage import analysis_runs as analysis_store

        if self._viewing or record.get(analysis_store.STALE_KEY) is not True:
            return False
        text = QLabel(
            "These results are out of date — the last change to this run did not finish "
            "writing them, so what is stored describes an earlier setting. Rebuild them to "
            "see the run as it is now."
        )
        text.setWordWrap(True)
        set_label_color(text, WARN)
        self._add(text)
        rebuild = QPushButton("Rebuild results")
        rebuild.clicked.connect(self._rebuild_artifact)
        self._add(self._live_only(rebuild))
        return True

    def _rebuild_artifact(self) -> None:
        """Re-derive the §19 artifact from the run's own compute output and current state.

        Off the UI thread with a real bar: this is a shard read and a frame-view zarr per
        analyzed unit — a minute over sixty recordings on a share, which is exactly the length
        of silence that reads as a hang.
        """
        run = self._experiment.authored_run(self._run_id)
        if run is None:
            return

        def work(emit):
            run.reproject(progress=emit)  # off-thread

        run_progress(
            self,
            work,
            lambda _result: self._build(),
            label="Rebuilding results…",
        )

    def _live_only(self, button):
        """Grey a button that acts on the *run* while a snapshot is being read.

        Greyed rather than removed: these are capabilities that do not apply in this mode, which is
        what the house rule covers — unlike the Measures panel, which is hidden because its numbers
        would be wrong rather than unavailable.
        """
        if self._viewing:
            button.setEnabled(False)
            button.setToolTip(
                f"Acts on the run itself. Go back to the live run to use it — you are reading "
                f"{self._viewing}."
            )
        return button

    def _open_folder(self) -> None:
        out_dir = self._experiment.analysis_run_dir(self._run_id)
        open_external.open_local_path(self, out_dir)

    def _cancel(self) -> None:
        _confirm_and_cancel(
            self,
            "Cancel analysis",
            f"Cancel analysis {self._run_id}? Its worker will be stopped.",
            lambda: self._experiment.cancel_analysis(self._run_id),
            self._build,
        )

    def _duplicate(self, name: str | None = None) -> None:
        """Copy the run under a new id, off the UI thread with a bar (a run can be large),
        then open the copy. ``name`` is asked for when not given."""
        if name is None:
            record = self._experiment.analysis_record(self._run_id)
            suggested = f"{record.get('name') or self._run_id} (copy)"
            name, accepted = QInputDialog.getText(
                self, "Duplicate analysis", "Name of the copy:", text=suggested
            )
            if not accepted:
                return
        experiment, run_id = self._experiment, self._run_id

        def work(emit):
            return experiment.duplicate_analysis(
                run_id, name=name.strip() or None, progress=emit
            )  # off-thread

        def done(new_id) -> None:
            if self._on_open is not None:
                self._on_open(new_id)
            else:
                QMessageBox.information(
                    self, "Duplicate analysis", f"{run_id} was copied to {new_id}."
                )

        run_progress(
            self,
            work,
            done,
            label="Copying the run…",
            on_error=lambda exc: QMessageBox.critical(
                self, "Duplicate analysis", str(exc)
            ),
            on_cancel=lambda: None,
        )

    def _delete(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete analysis",
            f"Delete analysis {self._run_id} and its outputs? This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._experiment.delete_analysis(self._run_id)
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Delete analysis", str(exc))
            return
        if self._on_deleted is not None:
            self._on_deleted()


# --------------------------------------------------------------------------- #
# Grouping variables (§11) — the cohort metadata table
# --------------------------------------------------------------------------- #
class MetadataPage(QWidget):
    """Experiment metadata.csv as a cohort table: recording id, then the file each
    recording's footage came from (derived, read-only — OQ-TBL-3), then user-defined
    columns (§11), then a read-only **completion** column per thing a recording can be
    worked on — field of view, distance calibration, ROI, POI, streams, detection,
    tracking, embedding — a checkbox that fills in once that recording has it, with the
    full count in its tooltip. **View-only by default** — click *Edit* to change cells
    and add/delete columns (copy/paste supported); the file and completion columns are
    never editable, in either mode. Edit mode is a **transaction**: every change is
    staged in the table and nothing touches disk until *Save*; *Cancel* discards all of
    it. Import/Export CSV are whole-table file operations and live in view mode.
    ``recording_id`` is always read-only. **Double-click a row (view mode only) to open
    that recording** — the design note's item D — via ``on_open(recording_id)``."""

    #: Fixed columns before the user's own (recording_id, file) — everything from here
    #: to :meth:`_completion_start` is a user metadata column.
    _FILE_COL = 1
    _USER_COL_START = 2

    def __init__(self, experiment: Experiment, on_open=None):
        super().__init__()
        self._experiment = experiment
        self._on_open = on_open  # optional: callers that don't navigate pass nothing
        self._editing = False
        self._cols: list[str] = []
        #: ``(checked, tooltip)`` per completion column, per row — computed once per
        #: rebuild so a row's eight boxes come from the one `recording.yaml` read that
        #: built the row, rather than eight more.
        self._completion_cells: list[list[tuple[bool, str]]] = []
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Recording information"))
        self._hint = _instructions("")
        layout.addWidget(self._hint)
        self._table = QTableWidget(0, 0)
        # `_rebuild` overrides this per-column once the completion columns exist (a
        # stretched checkbox column is mostly dead grid) — kept here too so the table
        # still fills its pane before the first build, and for `test_table_width.py`'s
        # per-constructor scan.
        fill_width(self._table)
        make_columns_resizable(self._table)
        enable_sorting(self._table)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        if self._on_open is not None:
            # The shared "this row opens something" cue (`_table(navigable=True)`) is a
            # `QTableView` helper; this page is a `QTableWidget` for its edit machinery,
            # so the cursor is set directly rather than pulling in that helper.
            self._table.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
            self._table.doubleClicked.connect(self._open_row)
        # This is the one table in the app that becomes an editor. Declared, because the
        # conformance walk only ever sees it in view mode — see `requirements.EDITOR_PROPERTY`.
        self._table.setProperty(requirements.EDITOR_PROPERTY, True)
        # Clicking a row highlights the row, as it does everywhere else in the app.
        # `_enter_edit` swaps to cell selection, because in edit mode the table is a
        # grid you copy and paste single cells in, not a list you pick an item from.
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # Stretch: the table is the page, and fills the window down to its last row
        # (`tables.fit_table_height`'s `fill` says why it must not overrun it).
        layout.addWidget(self._table, 1)
        QShortcut(QKeySequence.StandardKey.Copy, self._table, activated=self._copy)
        QShortcut(QKeySequence.StandardKey.Paste, self._table, activated=self._paste)

        row = QHBoxLayout()
        self._edit_btn = QPushButton("Edit")
        self._edit_btn.clicked.connect(self._enter_edit)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel)
        # Adding/removing a column is a structural edit — staged inside edit mode, not a
        # peer of Edit; Import/Export are whole-table file ops that stay in view mode.
        self._add_col_btn = QPushButton("Add column")
        self._add_col_btn.clicked.connect(self._add_column)
        self._delete_col_btn = QPushButton("Delete column")
        self._delete_col_btn.clicked.connect(self._delete_column)
        self._import_btn = QPushButton("Import CSV")
        self._import_btn.setToolTip(_IMPORT_CSV_TOOLTIP)
        self._import_btn.clicked.connect(self._import_csv)
        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setToolTip(
            "Write this table to a CSV to fill in offline: recording_id, name and file, "
            "then every column. Import it back keyed on any of the three."
        )
        self._export_btn.clicked.connect(self._export_csv)
        for button in (
            self._edit_btn,
            self._save_btn,
            self._cancel_btn,
            self._add_col_btn,
            self._delete_col_btn,
            self._import_btn,
            self._export_btn,
        ):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        self._rebuild()  # first: the view-mode hint below reads the column list
        self._leave_edit()  # sets view-mode hint + button visibility

    def _completion_start(self) -> int:
        """Column index of the first completion column — after recording_id, file, and
        however many user columns are staged right now."""
        return self._USER_COL_START + len(self._cols)

    def _rebuild(self) -> None:
        # One read per recording, pinned: the row wants several ``Recording`` fields
        # (video sources, arenas, calibration, rois, points, streams, three run lists)
        # and every one of those is its own ``recording.yaml`` read without the pin —
        # see ``_recording_rows`` above, and ``tests/gui/test_page_read_cost.py``.
        with read_cache.pinned():
            recordings = self._experiment.recordings
            self._recordings = [s.id for s in recordings]
            file_cells = [_file_name_cell(s) for s in recordings]
            self._completion_cells = [
                [fn(s) for _label, fn in _COMPLETION_COLUMNS] for s in recordings
            ]
        self._cols = self._experiment.metadata_columns
        table = self._experiment.recording_metadata

        def _fill() -> None:
            self._table.clear()
            completion_start = self._completion_start()
            self._table.setColumnCount(completion_start + len(_COMPLETION_COLUMNS))
            self._table.setHorizontalHeaderLabels(
                [
                    "recording_id",
                    "file",
                    *self._cols,
                    *[label for label, _fn in _COMPLETION_COLUMNS],
                ]
            )
            for i, name in enumerate(self._cols):
                if name == metadata_store.ACQUISITION_SETUP_KEY:
                    self._table.horizontalHeaderItem(
                        i + self._USER_COL_START
                    ).setToolTip(_ACQUISITION_SETUP_TIP)
            self._table.setRowCount(len(self._recordings))
            for r, sid in enumerate(self._recordings):
                key = QTableWidgetItem(sid)
                key.setFlags(key.flags() & ~Qt.ItemFlag.ItemIsEditable)
                key.setFont(MONO)
                self._table.setItem(r, 0, key)

                file_name, file_tooltip = file_cells[r]
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if file_tooltip:
                    file_item.setToolTip(file_tooltip)
                self._table.setItem(r, 1, file_item)

                values = table.get(sid, {})
                for c, col in enumerate(self._cols, start=self._USER_COL_START):
                    self._table.setItem(r, c, QTableWidgetItem(values.get(col, "")))

                for i, (checked, tooltip) in enumerate(self._completion_cells[r]):
                    item = QTableWidgetItem()
                    # Not `ItemIsUserCheckable`. The indicator draws from the check state
                    # alone — verified — so dropping the flag makes the cell genuinely
                    # inert instead of a control that toggles under the cursor and is put
                    # back by a handler. A checkbox that undoes your click is worse than
                    # one that never moves: it looks like the app disagreed with you and
                    # says nothing about why. These columns report what the recording has.
                    item.setFlags(
                        item.flags()
                        & ~Qt.ItemFlag.ItemIsUserCheckable
                        & ~Qt.ItemFlag.ItemIsEditable
                    )
                    item.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
                    item.setToolTip(tooltip)
                    self._table.setItem(r, completion_start + i, item)

        # `QTableWidget` resorts itself after every `setItem` while sorting is enabled —
        # see `tables.refill_table_widget` — which would otherwise scramble which cells
        # land in which row as this loop writes one row's columns at a time. The file
        # and completion columns made each row *longer*, so there is more of a row to
        # lose halfway through writing it.
        refill_table_widget(self._table, _fill)
        self._table.resizeColumnsToContents()
        # The slack goes to `file`, not the last section and not the last user column. A
        # checkbox column given it is mostly dead grid; a grouping variable is usually a word or
        # a digit, so a `Cohort` of "1" and "11" rendered 400px wide with the checkboxes crushed
        # beside it. `file` holds a video name, is the one column here whose content is long,
        # and is the one that gets truncated when something else eats the width. After the
        # resize to contents, which would otherwise shrink it back to its own text.
        give_slack_to(self._table, self._FILE_COL)
        _fit_table_height(self._table, fill=True)

    def _open_row(self, index) -> None:
        """View mode only — in edit mode a double-click means "edit this cell", not
        "leave the page" (``_paste`` draws the same line for clipboard paste).

        Identity comes from the row's own ``recording_id`` cell, never from
        ``self._recordings[index.row()]``: this table sorts, so a widget row's position
        stops agreeing with the build order the moment a user clicks a header, and the
        double-click would open a different recording than the one under the cursor.
        ``_save`` was given this treatment when sorting landed; the open path was not.
        """
        if self._editing or self._on_open is None:
            return
        recording_id = self._row_recording_id(index.row())
        if recording_id:
            self._on_open(recording_id)

    def _row_recording_id(self, r: int) -> str:
        """The recording column 0 of widget row ``r`` actually holds right now.

        Column 0 is the stable per-row key (read-only, never blank) regardless of how the
        table is currently sorted — never assume widget row ``r`` is ``self._recordings[r]``,
        which is only true in the freshly-rebuilt, unsorted table.
        """
        item = self._table.item(r, 0)
        return item.text() if item is not None else ""

    # -- view / edit mode ----------------------------------------------------
    def _enter_edit(self) -> None:
        self._editing = True
        # Sorting off while cells are editable. A `QTableWidget` resorts itself the moment an
        # item's data changes, so typing into a cell of the sorted column moves that row away
        # mid-edit and the next keystrokes land in whatever slid into its place. Measured:
        # [alpha, bravo, charlie] sorted, edit row 0 to "zulu", result [bravo, charlie, zulu].
        # `refill_table_widget` covers the bulk-repopulate path; this is the one a person meets.
        self._table.setSortingEnabled(False)
        # Cell selection while editing: `_copy` reads `selectedRanges()`, and a whole-row
        # range would take `recording_id`, `file` and all eight completion boxes with it.
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.AnyKeyPressed
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        # edit tools appear; view-mode file ops step aside until Save/Cancel
        self._edit_btn.setVisible(False)
        self._import_btn.setVisible(False)
        self._export_btn.setVisible(False)
        self._save_btn.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._add_col_btn.setVisible(True)
        self._delete_col_btn.setVisible(True)
        self._update_delete_enabled()
        self._hint.setText(
            "Editing — double-click a cell to change it; copy/paste supported. "
            "Add or delete columns. Save to write, Cancel to discard."
        )

    def _leave_edit(self) -> None:
        self._editing = False
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Give sorting back: the mode has to restore what it took, or one trip through Edit
        # leaves the table permanently unsortable and nothing says why.
        enable_sorting(self._table)
        self._edit_btn.setVisible(True)
        self._import_btn.setVisible(True)
        self._export_btn.setVisible(True)
        self._save_btn.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._add_col_btn.setVisible(False)  # add/delete columns are edit-mode only
        self._delete_col_btn.setVisible(False)
        open_hint = (
            " Double-click a row to open that recording." if self._on_open else ""
        )
        self._hint.setText(
            (
                "One row per recording, one column per thing that varies across them. "
                "View-only — click Edit to change cells; Import/Export CSV for bulk work."
                if self._user_cols()
                else "One row per recording, one column per thing that varies across them "
                "(treatment, day, animal id). Click Edit to add your first column, or "
                "Import CSV to bring in a sheet you already have."
            )
            + open_hint
        )

    def _user_cols(self) -> list[str]:
        """The columns a person added — every one but the reserved ones, which stay."""
        return [c for c in self._cols if c not in metadata_store.RESERVED_COLUMNS]

    def _update_delete_enabled(self) -> None:
        self._delete_col_btn.setEnabled(bool(self._user_cols()))

    def _save(self) -> None:
        """Commit the whole edited table in one write — staged column adds/deletes and
        cell edits together — then return to view mode.

        Keyed on each widget row's own ``recording_id`` cell (:func:`_row_recording_id`),
        not on ``self._recordings[r]``: the table may be sorted by any column, and
        ``self._recordings`` stays in the order it was last read from the store.
        """
        rows = {
            # Both halves. The key is the row's own `recording_id` cell, because the table
            # may be sorted and `self._recordings` is in store order — saving by position
            # writes to the wrong recording. The offset is `_USER_COL_START`, because the
            # user's columns no longer begin at 1: `file` sits in front of them.
            self._row_recording_id(r): {
                col: self._cell_text(r, c)
                for c, col in enumerate(self._cols, start=self._USER_COL_START)
            }
            for r in range(self._table.rowCount())
        }
        self._experiment.set_metadata_table(self._cols, rows)
        self._leave_edit()
        self._rebuild()

    def _cancel(self) -> None:
        self._leave_edit()
        self._rebuild()  # discard all staged edits by re-reading the store

    # -- unsaved edits (gui/editing.py) --------------------------------------
    def unsaved_summary(self) -> str | None:
        """What Save would write that the file doesn't already hold.

        Compared against the store rather than tracked with a dirty flag: entering edit
        mode and changing nothing is not unsaved work, and a cell edited back to its
        original value isn't either.
        """
        if not self._editing:
            return None
        stored = self._experiment.recording_metadata
        changed = sum(
            1
            for r in range(self._table.rowCount())
            for c, col in enumerate(self._cols, start=self._USER_COL_START)
            if self._cell_text(r, c)
            != (stored.get(self._row_recording_id(r), {}).get(col, ""))
        )
        columns = set(self._cols) ^ set(self._experiment.metadata_columns)
        if not changed and not columns:
            return None
        parts = []
        if changed:
            parts.append(f"{changed} changed cell{'s' if changed != 1 else ''}")
        if columns:
            parts.append(
                f"{len(columns)} column change{'s' if len(columns) != 1 else ''}"
            )
        return "Recording information has " + " and ".join(parts) + "."

    def save_edits(self) -> None:
        self._save()

    def discard_edits(self) -> None:
        self._cancel()

    # -- copy / paste --------------------------------------------------------
    def _cell_text(self, r: int, c: int) -> str:
        item = self._table.item(r, c)
        return item.text() if item is not None else ""

    def _copy(self) -> None:
        """Selected cells → clipboard as tab/newline TSV (works in either mode).

        In **view mode** the selection is a whole row (that is the highlight the rest of the
        app gives a clicked row), so a range would carry `recording_id`, `file` and all eight
        completion boxes. The cell under the cursor is what a person means there. Block copy
        is an edit-mode activity, and edit mode selects cells.
        """
        ranges = (
            self._table.selectedRanges()
            if self._table.selectionBehavior()
            == QAbstractItemView.SelectionBehavior.SelectItems
            else []
        )
        if ranges:
            rng = ranges[0]
            rows = range(rng.topRow(), rng.bottomRow() + 1)
            cols = range(rng.leftColumn(), rng.rightColumn() + 1)
        else:
            current = self._table.currentItem()
            if current is None:
                return
            rows = range(current.row(), current.row() + 1)
            cols = range(current.column(), current.column() + 1)
        lines = ["\t".join(self._cell_text(r, c) for c in cols) for r in rows]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _paste(self) -> None:
        """Clipboard TSV → cells from the current cell (edit mode only). A single
        copied value fills the whole selection (fill-down); ``recording_id`` and
        out-of-range overflow are skipped."""
        if not self._editing:
            return
        text = QGuiApplication.clipboard().text()
        current = self._table.currentItem()
        if not text or current is None:
            return
        grid = [line.split("\t") for line in text.rstrip("\n").split("\n")]
        ranges = self._table.selectedRanges()
        if len(grid) == 1 and len(grid[0]) == 1 and ranges:
            value = grid[0][0]
            rng = ranges[0]
            for r in range(rng.topRow(), rng.bottomRow() + 1):
                for c in range(rng.leftColumn(), rng.rightColumn() + 1):
                    self._set_cell(r, c, value)
            return
        for i, line in enumerate(grid):
            for j, value in enumerate(line):
                self._set_cell(current.row() + i, current.column() + j, value)

    def _set_cell(self, r: int, c: int, value: str) -> None:
        if (
            c < self._USER_COL_START
            or c >= self._completion_start()
            or r < 0
            or r >= self._table.rowCount()
        ):
            # recording_id (0), file (1), and the completion columns are never
            # editable — never mind out-of-range paste overflow.
            return
        item = self._table.item(r, c)
        if item is None:
            item = QTableWidgetItem()
            self._table.setItem(r, c, item)
        item.setText(value)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export metadata CSV", "metadata.csv", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        run_busy(
            self,
            lambda: self._experiment.export_metadata_csv(path),
            lambda _r: QMessageBox.information(
                self, "Export metadata", f"Wrote {path}"
            ),
            label="Exporting…",
        )

    def _add_column(self) -> None:
        """Add a column to the table, staged — committed on Save, discarded on Cancel."""
        name, ok = QInputDialog.getText(self, "Add column", "Column name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name == "recording_id" or name in self._cols:
            QMessageBox.warning(
                self, "Add column", f"Column '{name}' already exists or is reserved."
            )
            return
        self._cols.append(name)
        # Insert right before the completion columns (now one further right, since
        # `_completion_start` reads `self._cols`) rather than appending at the very
        # end, which would land the new column after them.
        col = self._completion_start() - 1
        self._table.insertColumn(col)
        self._table.setHorizontalHeaderItem(col, QTableWidgetItem(name))
        for r in range(self._table.rowCount()):
            self._table.setItem(r, col, QTableWidgetItem(""))
        self._update_delete_enabled()
        _fit_table_height(self._table)

    def _delete_column(self) -> None:
        """Remove a column from the table, staged — committed on Save, restored on Cancel."""
        deletable = self._user_cols()
        if not deletable:
            return
        name, ok = QInputDialog.getItem(
            self, "Delete column", "Column to delete:", deletable, 0, False
        )
        if not ok or not name:
            return
        idx = self._cols.index(name)
        self._table.removeColumn(idx + self._USER_COL_START)
        self._cols.pop(idx)
        self._update_delete_enabled()
        _fit_table_height(self._table)

    def _import_csv(self) -> None:
        """Bulk-apply grouping variables from a CSV (§11): merge-only, skip blanks,
        reject unknown recordings — on any error nothing is applied.

        A sheet with no recognised key column is not refused: the person is asked which
        of its columns holds each recording's video file name (#330), because that is the
        key a sheet made before import has, under whatever heading its author gave it.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import metadata CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            key_column, header = self._experiment.metadata_csv_key(path)
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.critical(self, "Import metadata", f"Cannot import — {exc}")
            return
        if key_column is None:
            key_column, ok = QInputDialog.getItem(
                self,
                "Import metadata",
                "This sheet has no recording_id, name or video file name column.\n"
                "Which column holds each recording's video file name?\n"
                "(Matched on the base name, any case, extension optional.)",
                header,
                0,
                False,
            )
            if not ok or not key_column:
                return
        errors, warnings, applied = self._experiment.import_metadata_csv(
            path, key_column=key_column
        )
        if errors:
            QMessageBox.critical(
                self,
                "Import metadata",
                "Cannot import — fix these and retry:\n\n" + "\n".join(errors[:20]),
            )
            return
        self._rebuild()
        message = f"Updated {applied} recording(s)."
        if warnings:
            message += "\n\n" + "\n".join(warnings)
        QMessageBox.information(self, "Import metadata", message)
