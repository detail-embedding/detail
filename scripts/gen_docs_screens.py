"""Render GUI screens to committed PNGs for the user guides.

``docs/docs_design.md`` §6: GUI images are auto-rendered, not hand-captured. This
generator opens the real GUI offscreen and grabs each documented screen via
``QWidget.grab()``. Rerun a target after a GUI change to refresh its PNGs; the docs
build just embeds them (no Qt at build time).

    python scripts/gen_docs_screens.py user-guides   # install + Configure DETAIL
    python scripts/gen_docs_screens.py more-ways     # chapter 1's optional steps
    python scripts/gen_docs_screens.py single-animal <chapter> <demo experiment folder>
                                                     # one chapter of the user guide

``user-guides`` and ``more-ways`` need nothing outside the repository. The chapter targets
read the guide's demo experiment, whose videos are too large to commit.

Offscreen Qt needs no display: ``QT_QPA_PLATFORM=offscreen`` is set before any Qt
import, exactly as ``tests/gui/conftest.py`` does.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any Qt import
if sys.platform == "win32":
    # Offscreen Qt on Windows finds no fonts of its own and draws every glyph as a box.
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import json
import shutil
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QPoint, QRect, QSettings, QSize, QThreadPool
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QStandardItemModel
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableView,
    QWidget,
)

from detail.api import checks as api_checks
from detail.api import models as api_models
from detail.api import previews
from detail.api.experiment import Experiment
from detail.api.readiness import ReadinessSignals
from detail.checks import commands
from detail.core.dataclasses import CheckResult, CheckStatus, ComputeTarget
from detail.gui import app as app_module
from detail.gui import arena
from detail.gui import cuda_doctor, pages, streams
from detail.gui.__main__ import configure_application
from detail.gui.app import MainWindow
from detail.storage import locations

_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "docs" / "assets" / "sample-data"
_GUIDES_OUT = _ROOT / "docs" / "assets" / "user-guides"
_WIN_SIZE = (1040, 680)
_HIGHLIGHT = QColor("#D55E00")  # vermillion callout — a doc mark, not a UI status color

# A committed image never shows this machine's paths. The Configuration screens print the
# detail folder they run from; these stand in for it.
_DISPLAY_REPO = Path(
    r"C:\Users\yourname\detail" if sys.platform == "win32" else "/home/yourname/detail"
)
# And for the folder an experiment lives in, which its overview prints.
_DISPLAY_WORKSPACE = Path(
    r"C:\Users\yourname\Documents\DetailExperiments"
    if sys.platform == "win32"
    else "/home/yourname/Documents/DetailExperiments"
)


def _noop(*args: object, **kwargs: object) -> None:
    return None


#: How many numbered markers each written image carries, keyed by its path under
#: ``_GUIDES_OUT``. ``tests/test_user_guide_legends.py`` checks each image's legend against
#: it: a legend whose lines run out of step with the markers is wrong from that line on.
_CALLOUTS_FILE = _GUIDES_OUT / "callouts.json"

_marker_counts: dict[int, int] = {}  # id(pixmap) -> markers drawn on it

_callouts: dict[str, int] = {}


def _write(pixmap: QPixmap, rel_path: str, out_dir: Path = _GUIDES_OUT) -> None:
    out = out_dir / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(out))
    if out_dir == _GUIDES_OUT:
        _callouts[rel_path] = _marker_counts.pop(id(pixmap), 0)
    print(f"wrote {out.relative_to(_ROOT)}")


def _save_callouts() -> None:
    """Merge this run's marker counts into the committed record; an image drawn with none
    drops out of it."""
    known = {}
    if _CALLOUTS_FILE.exists():
        known = json.loads(_CALLOUTS_FILE.read_text(encoding="utf-8-sig"))
    known.update(_callouts)
    known = {path: n for path, n in sorted(known.items()) if n}
    _CALLOUTS_FILE.write_text(json.dumps(known, indent=2) + "\n", encoding="utf-8")


def _find_button(root: QWidget, text: str) -> QPushButton:
    def norm(label: str) -> str:
        return label.replace("&", "").rstrip("…").strip().lower()

    # Visible only: a page swapped out of the window can outlive the swap as a hidden child,
    # and its same-labelled button would be boxed in the wrong place.
    for button in root.findChildren(QPushButton):
        if button.isVisible() and norm(button.text()) == norm(text):
            return button
    raise LookupError(f"no button labelled {text!r}")


def _box(pixmap: QPixmap, rect: QRect) -> None:
    painter = QPainter(pixmap)
    painter.setPen(QPen(_HIGHLIGHT, 3))
    painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 5, 5)
    painter.end()


def _settle() -> None:
    """Let the off-UI-thread probes the pages start finish and paint their results."""
    for _ in range(3):
        QThreadPool.globalInstance().waitForDone(60_000)
        QApplication.processEvents()


def _box_widget(window: QWidget, pixmap: QPixmap, target: QWidget) -> None:
    _box(pixmap, QRect(target.mapTo(window, QPoint(0, 0)), target.size()))


def _rail_item(window: MainWindow, kind: str):
    """The project rail's node for ``kind`` (``recordings``, ``metadata``, ...)."""
    tree = window._tree
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        data = item.data(0, app_module._ROLE)
        if data and data[0] == kind:
            return item
        stack.extend(item.child(i) for i in range(item.childCount()))
    raise LookupError(f"no rail node of kind {kind!r}")


def _box_rail(window: MainWindow, pixmap: QPixmap, kind: str) -> None:
    tree = window._tree
    rect = tree.visualItemRect(_rail_item(window, kind))
    _box(pixmap, QRect(tree.viewport().mapTo(window, rect.topLeft()), rect.size()))


def _go_to_rail(window: MainWindow, kind: str) -> None:
    item = _rail_item(window, kind)
    window._tree.setCurrentItem(
        item
    )  # what a click does, so the rail marks the page shown
    window._navigate(item)
    _settle()


_CONTROL_TYPES = (
    QAbstractButton,
    QAbstractSlider,
    QLineEdit,
    QComboBox,
    QAbstractSpinBox,
    QAbstractItemView,
    QPlainTextEdit,
)


def _controls(root: QWidget) -> list[QWidget]:
    """Every visible thing a reader can click or type into under ``root``."""
    found = []
    for widget in root.findChildren(QWidget):
        if not isinstance(widget, _CONTROL_TYPES) or not widget.isVisible():
            continue
        if (
            isinstance(widget, QHeaderView)
            or widget.objectName() == "qt_tableview_cornerbutton"
        ):
            continue
        found.append(widget)
    return found


def _number(
    pixmap: QPixmap,
    base: QWidget,
    marks: list,
    *,
    covers: QWidget | None = None,
) -> None:
    """Draw a numbered marker on each of ``marks``, in order, for the page's legend.

    A mark is a widget, a list of widgets (one number for a group, such as a column of
    **Open →** buttons), a ``QRect`` in ``base``'s coordinates (a rail node), or a ``QPoint``
    there (an explicit spot, for a picture whose corners sit beside other controls). With
    ``covers``, every visible control under that widget must sit inside a mark, so a button
    the app gains later fails the render instead of going undescribed in the guide.
    """
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont(painter.font())
    font.setBold(True)
    font.setPointSize(8)
    painter.setFont(font)
    radius = 9
    numbered: list[QWidget] = []
    for n, mark in enumerate(marks, start=1):
        # Placed where a label's text is not: left of a rail node's text, and on a control's
        # top-right corner (for a button, the end of its natural width, which is where a
        # checkbox's own text ends rather than the far edge of a stretched row).
        if isinstance(mark, QPoint):
            centre = QPoint(mark)
        elif isinstance(mark, QRect):
            centre = QPoint(mark.left() - radius - 1, mark.center().y())
        else:
            group = mark if isinstance(mark, list) else [mark]
            numbered.extend(group)
            widget = group[0]
            rect = QRect(widget.mapTo(base, QPoint(0, 0)), widget.size())
            right = rect.right()
            if isinstance(widget, QAbstractButton):
                right = rect.left() + min(rect.width(), widget.sizeHint().width())
            centre = QPoint(right - 1, rect.top() + 1)
        centre.setX(max(radius, min(centre.x(), pixmap.width() - radius - 1)))
        centre.setY(max(radius, min(centre.y(), pixmap.height() - radius - 1)))
        painter.setPen(QPen(QColor("white"), 1.5))
        painter.setBrush(_HIGHLIGHT)
        painter.drawEllipse(centre, radius, radius)
        circle = QRect(centre - QPoint(radius, radius), QSize(2 * radius, 2 * radius))
        painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, str(n))
    painter.end()
    _marker_counts[id(pixmap)] = _marker_counts.get(id(pixmap), 0) + len(marks)
    if covers is None:
        return
    missing = [
        widget
        for widget in _controls(covers)
        if not any(widget is mark or mark.isAncestorOf(widget) for mark in numbered)
    ]
    if missing:
        names = ", ".join(
            f"{type(w).__name__} {getattr(w, 'text', lambda: '')()!r}" for w in missing
        )
        raise LookupError(f"controls with no number in the legend: {names}")


def _rail_rect(window: MainWindow, kind: str) -> QRect:
    tree = window._tree
    rect = tree.visualItemRect(_rail_item(window, kind))
    return QRect(tree.viewport().mapTo(window, rect.topLeft()), rect.size())


def _page(window: MainWindow) -> QWidget:
    """The screen under the window's chrome: what a page's legend describes."""
    return window._content.widget(0)


def _mask_paths(root: QWidget, real: dict[str, str]) -> None:
    """Replace each real local path prefix with its display stand-in, in every label, line
    edit and table cell under ``root``: a committed image never shows this machine's paths.
    """

    def masked(text: str) -> str:
        for prefix, shown in real.items():
            text = text.replace(prefix, shown)
        return text

    for label in root.findChildren(QLabel):
        label.setText(masked(label.text()))
    for edit in root.findChildren(QLineEdit):
        edit.setText(masked(edit.text()))
    for view in root.findChildren(QTableView):
        model = view.model()
        if not isinstance(model, QStandardItemModel):
            continue
        for row in range(model.rowCount()):
            for col in range(model.columnCount()):
                item = model.item(row, col)
                if item is not None and item.text():
                    item.setText(masked(item.text()))


def _pump(seconds: float) -> None:
    """Process events for ``seconds``: for a view that decodes frames with no signal to
    wait on from outside."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.05)
    _settle()


def _wait_until(ready, seconds: float = 60.0) -> None:
    """Pump events until ``ready()`` holds: a frame decoded on a worker thread lands late."""
    deadline = time.monotonic() + seconds
    while not ready():
        if time.monotonic() > deadline:
            raise TimeoutError("the screen never finished loading")
        QApplication.processEvents()
        time.sleep(0.05)
    _settle()


def _normalized(points: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(x / 800, y / 600) for x, y in points]


def _in_reading_order(root: QWidget, base: QWidget) -> list[QWidget]:
    """Every control under ``root``, top to bottom and left to right on screen.

    A field's **?** help buttons are left out: they are one legend line of their own, the
    group :func:`_help_buttons` returns. So is a control inside another one, such as a spin
    box's own text field: the reader sees one field, so it gets one number.
    """

    def position(widget):
        p = widget.mapTo(base, QPoint(0, 0))
        return (p.y() // 12, p.x())  # one row per line of controls

    found = [w for w in _controls(root) if not _is_help(w)]
    outermost = [
        w for w in found if not any(o is not w and o.isAncestorOf(w) for o in found)
    ]
    return sorted(outermost, key=position)


def _is_help(widget: QWidget) -> bool:
    return isinstance(widget, QAbstractButton) and widget.text() == "?"


def _help_buttons(root: QWidget) -> list[QWidget]:
    return [w for w in _controls(root) if _is_help(w)]


def _describe(widget: QWidget) -> str:
    for attr in ("text", "currentText", "placeholderText"):
        if hasattr(widget, attr):
            text = getattr(widget, attr)()
            if text:
                return f"{type(widget).__name__} {text!r}"
    return type(widget).__name__


def _number_screen(shot, base, root, rel_path, *, skip=(), extra=()):
    """Number every control under ``root`` in reading order (the **?** buttons last, as one),
    check the legend will cover the screen, write the image, and print what each number is.

    ``skip`` leaves out controls described elsewhere (the window's own banner); ``extra``
    adds marks ``root`` does not hold, such as the window's Export… button.
    """
    marks = [
        w
        for w in _in_reading_order(root, base)
        if not any(w is s or s.isAncestorOf(w) for s in skip)
    ]
    marks = [*extra, *marks]
    helps = _help_buttons(root)
    if helps:
        marks.append(helps)
    _number(shot, base, marks, covers=root if not skip else None)
    _write(shot, rel_path, _GUIDES_OUT)
    print(f"--- legend for {rel_path}")
    for n, mark in enumerate(marks, start=1):
        if isinstance(mark, list):
            print(f"  {n}. group of {len(mark)}: {_describe(mark[0])}")
        else:
            print(f"  {n}. {_describe(mark)}")


def _open_advanced(dialog: QWidget) -> QPushButton:
    toggle = next(
        b
        for b in dialog.findChildren(QPushButton)
        if b.isVisible() and "Advanced Settings" in b.text()
    )
    toggle.click()
    QApplication.processEvents()
    return toggle


def _below(widgets, base, top: int, bottom: int | None = None):
    def y(w):
        return w.mapTo(base, QPoint(0, 0)).y()

    return [w for w in widgets if y(w) > top and (bottom is None or y(w) < bottom)]


def _signals(*, gpu: bool, done: bool) -> ReadinessSignals:
    """The checklist on a machine with an NVIDIA GPU.

    ``gpu`` is whether PyTorch can use it: a first Windows install has the CPU-only
    PyTorch, so its GPU row is unchecked until the GPU / CUDA step is done. ``done`` is
    steps 2 to 4 complete.
    """
    return ReadinessSignals(
        gpu=gpu,
        hf_token=done,
        sam3=done,
        sam3_backend=done,
        sam3_weights=done,
        detection_model=done,
        has_experiment=False,
        quick_segment_run=False,
        distilled_model=False,
    )


def _fresh_machine() -> None:
    """Stand in for a just-installed Windows box, never the machine running this script.

    The Configuration screens read live state: this computer's GPU and PyTorch build, its
    token, its model cache, and the folder DETAIL runs from. Every one of those would
    otherwise land in a committed image.
    """
    pages._fresh_signals = lambda: _signals(gpu=False, done=False)
    # A CPU-only PyTorch beside a working driver: what a first Windows install has.
    cuda_doctor.probe = lambda: cuda_doctor.ProbeResult(
        cuda=False,
        mps=False,
        cuda_devices=[],
        torch_installed=True,
        torch_cuda_build=None,
        driver_version="566.36",
        driver_cuda="12.7",
    )
    api_checks.run_checks = lambda **kwargs: [
        CheckResult("cuda", CheckStatus.FAIL),
        CheckResult("sam3_backend", CheckStatus.FAIL, "No module named 'sam3'"),
    ]
    commands.detail_repo_root = lambda: _DISPLAY_REPO
    commands.active_env_dir = lambda: _DISPLAY_REPO / ".venv"
    api_checks.detail_repo_root = commands.detail_repo_root
    pages.credentials.get_hf_token = lambda: None
    api_models.is_installed = lambda *args, **kwargs: False
    api_models.imported_models = lambda **kwargs: []
    # The Preview cache line under the checklist prints this machine's home folder and size.
    previews.preview_cache_dir = lambda: _DISPLAY_REPO.parent / ".cache/detail/previews"
    previews.preview_cache_size = lambda: 0


def _render_more_ways(parent: Path) -> None:
    """Chapter 1's optional steps: a recording in several files, and Import bundle.

    Both come from a small experiment built here from the committed sample clips, so this
    target needs no demo folder. Each image boxes the one button its step clicks and carries
    no numbered markers: the pages themselves are numbered where the guide first shows them.
    """
    out = "single-animal/create"
    # Resolved, because a stored source path is: on macOS the temp folder is behind a symlink,
    # and an unresolved prefix would leave part of the real path in the picture.
    parent = parent.resolve()
    # The clips under the names the chapter's manifest uses, in a folder shown as the reader's.
    clips = parent / "clips"
    clips.mkdir()
    names = {
        "boxA_am.mp4": "box1_morning.mp4",
        "boxA_pm.mp4": "box1_afternoon.mp4",
        "boxB.mp4": "box2.mp4",
    }
    for sample, name in names.items():
        shutil.copyfile(_SAMPLE / sample, clips / name)
    videos = (
        r"C:\Users\yourname\Videos"
        if sys.platform == "win32"
        else "/home/yourname/Videos"
    )
    workspace = parent / "experiments"
    workspace.mkdir()
    locations.experiments_root = lambda: workspace
    experiment = Experiment.create("Top-down demo", parent_dir=workspace)
    box1 = experiment.add_recording("box1")
    box1.add_video_part(clips / "box1_morning.mp4")
    box1.add_video_part(clips / "box1_afternoon.mp4")
    experiment.add_recording("box2").add_video_part(clips / "box2.mp4")
    # The clips folder first: it sits inside ``parent``, and a prefix is replaced in order.
    shown = {str(clips): videos, str(workspace): str(_DISPLAY_WORKSPACE)}

    window = MainWindow()
    window._build_label.hide()
    window.resize(*_WIN_SIZE)
    window.show()
    window._open_experiment(experiment)
    _go_to_rail(window, "recordings")
    _settle()
    _mask_paths(window, shown)
    shot = window.grab()
    _box_widget(window, shot, _find_button(_page(window), "Import bundle"))
    _write(shot, f"{out}/import-bundle.png")

    window.resize(
        _WIN_SIZE[0], 940
    )  # the recording page is tall: show the Video part whole
    window._open_recording(box1.id)
    _settle()
    _mask_paths(window, shown)
    shot = window.grab()
    _box_widget(window, shot, _find_button(_page(window), "Add video part"))
    _write(shot, f"{out}/add-video-part.png")
    window.close()


def _render_user_guides(parent: Path) -> None:
    """The screens the install pages and Configure DETAIL show, from an empty workspace."""
    home = MainWindow()
    home._build_label.hide()  # this checkout's commit, which no reader is running
    home.resize(*_WIN_SIZE)
    home.show()
    _settle()
    _write(home.grab(), "install/first-launch.png", _GUIDES_OUT)
    home.close()

    _fresh_machine()
    window = MainWindow()
    window._build_label.hide()
    window.resize(*_WIN_SIZE)
    window.show()
    _settle()

    window._switch_mode("config")
    _settle()
    shot = window.grab()
    _box_widget(window, shot, window._banner.button("config"))
    _box_widget(window, shot, window.findChild(QPushButton, "gpu_cuda"))
    _write(shot, "configure/checklist-fresh.png", _GUIDES_OUT)

    def sub_page(kind: str, rel_path: str, button: str, size=_WIN_SIZE) -> None:
        window.resize(*size)
        window._go_to_node(kind)
        _settle()
        shot = window.grab()
        _box_widget(window, shot, _find_button(window, button))
        _write(shot, rel_path, _GUIDES_OUT)

    sub_page("gpu_cuda", "configure/gpu-cuda.png", "Copy")

    # After the PyTorch reinstall and a restart, the checklist sees the GPU.
    pages._fresh_signals = lambda: _signals(gpu=True, done=False)
    api_checks.run_checks = lambda **kwargs: [
        CheckResult("cuda", CheckStatus.OK),
        CheckResult("sam3_backend", CheckStatus.FAIL, "No module named 'sam3'"),
    ]
    window._switch_mode("config")
    _settle()
    shot = window.grab()
    _box_widget(window, shot, window.findChild(QPushButton, "hugging_face"))
    _write(shot, "configure/checklist-gpu.png", _GUIDES_OUT)

    sub_page("hugging_face", "configure/hugging-face.png", "Save token")
    sub_page("sam3_backend", "configure/sam3-backend.png", "Copy", size=(1040, 900))

    window.resize(*_WIN_SIZE)
    window._switch_mode("models")
    _settle()
    hf = window.findChild(pages.GlobalModelsPage)
    table = hf.findChild(QTableView)
    for row in range(table.model().rowCount()):
        if table.model().index(row, 0).data() == "sam3":
            table.selectRow(row)
    QApplication.processEvents()
    shot = window.grab()
    _box_widget(window, shot, _find_button(hf, "Install"))
    _write(shot, "configure/model-manager.png", _GUIDES_OUT)

    pages._fresh_signals = lambda: _signals(gpu=True, done=True)
    window._switch_mode("config")
    _settle()
    _write(window.grab(), "configure/checklist-done.png", _GUIDES_OUT)
    window.close()


def _demo_window(demo_dir: Path, size=_WIN_SIZE) -> MainWindow:
    """The app window opened on the demo experiment, as a reader would have it."""
    locations.experiments_root = lambda: demo_dir.parent
    window = MainWindow()
    window._build_label.hide()
    window.resize(*size)
    window.show()
    window._open_experiment(Experiment(demo_dir))
    _settle()
    return window


def _render_create(demo_dir: Path) -> None:
    """Chapter 1: create an experiment and add recordings.

    The steps before any video exists are drawn from a fresh experiment in a throwaway
    folder; the ones after are drawn from the guide's demo experiment, ``demo_dir``, which
    was built by following the same steps (its videos are too large to commit, so it lives
    outside the repository).
    """
    out = "single-animal/create"
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        locations.experiments_root = lambda: parent
        home = MainWindow()
        home._build_label.hide()
        home.resize(*_WIN_SIZE)
        home.show()
        _settle()
        shot = home.grab()
        _box_widget(home, shot, _find_button(home, "New experiment"))
        banner = [
            "Experiments",
            "Active processes",
            "Configuration",
            "Model manager",
            "Keypoints manager",
            "Quick tools",
            "Docs",
            "Version History",
            "Info",
            "New experiment",
            "Import experiment",
        ]
        _number(shot, home, [_find_button(home, b) for b in banner], covers=home)
        _write(shot, f"{out}/experiments-empty.png", _GUIDES_OUT)

        dialog = pages.NewExperimentDialog()
        dialog._name.setText("Top-down demo")
        dialog._desc.setText("Fentanyl self-administration, camera above the box")
        dialog.resize(560, dialog.sizeHint().height())
        dialog.show()
        QApplication.processEvents()
        shot = dialog.grab()
        _box_widget(dialog, shot, _find_button(dialog, "Create"))
        _number(
            shot,
            dialog,
            [
                dialog._name,
                dialog._desc,
                _find_button(dialog, "Create"),
                _find_button(dialog, "Cancel"),
            ],
            covers=dialog,
        )
        _write(shot, f"{out}/new-experiment.png", _GUIDES_OUT)
        dialog.close()

        fresh = Experiment.create("Top-down demo", parent_dir=parent)
        fresh.set_metadata(
            description="Fentanyl self-administration, camera above the box"
        )
        home._open_experiment(fresh)
        _settle()
        _mask_paths(home, {str(parent): str(_DISPLAY_WORKSPACE)})
        shot = home.grab()
        _box_rail(home, shot, "recordings")
        overview = _page(home)
        name, description = [
            e for e in overview.findChildren(QLineEdit) if e.isVisible()
        ][:2]
        sections = [
            b
            for b in overview.findChildren(QPushButton)
            if b.isVisible() and b.text().startswith("Open")
        ]
        rail = [
            "experiment",
            "metadata",
            "recordings",
            "datasets",
            "models",
            "runs",
            "analysis",
            "comparisons",
        ]
        _number(
            shot,
            home,
            [
                *[_rail_rect(home, kind) for kind in rail],
                home._notes_button,
                home._export,
                name,
                description,
                _find_button(overview, "Export experiment"),
                _find_button(overview, "Save"),
                sections,
                _find_button(overview, "Delete experiment"),
            ],
            covers=overview,
        )
        _write(shot, f"{out}/overview.png", _GUIDES_OUT)

        _go_to_rail(home, "recordings")
        shot = home.grab()
        _box_widget(home, shot, _find_button(home, "Add recordings"))
        recordings = _page(home)
        _number(
            shot,
            home,
            [
                _find_button(recordings, "Add recordings"),
                _find_button(recordings, "Import bundle"),
                _find_button(recordings, "Import behavior labels"),
                _find_button(recordings, "Cancel imports"),
            ],
            covers=recordings,
        )
        _write(shot, f"{out}/recordings-empty.png", _GUIDES_OUT)

        add = pages.AddRecordingsDialog(fresh, _noop)
        videos = (
            r"C:\Users\yourname\Videos"
            if sys.platform == "win32"
            else "/home/yourname/Videos"
        )
        for n in (1, 2, 3):
            add._append_row(f"box{n}", str(Path(videos) / f"box{n}.mp4"))
        add.resize(700, 340)
        add._table.setColumnWidth(1, 360)
        add.show()
        QApplication.processEvents()
        shot = add.grab()
        _box_widget(add, shot, _find_button(add, "Create"))
        _number(
            shot,
            add,
            [
                add._table,
                _find_button(add, "Add videos"),
                _find_button(add, "Add empty recording"),
                _find_button(add, "Remove selected"),
                add._reference._on,
                _find_button(add, "Create"),
                _find_button(add, "Cancel"),
            ],
            covers=add,
        )
        _write(shot, f"{out}/add-recordings.png", _GUIDES_OUT)
        add.close()
        home.close()

    window = _demo_window(demo_dir)
    _go_to_rail(window, "recordings")
    _mask_paths(window, {str(demo_dir.parent.parent / "top-down"): videos})
    shot = window.grab()
    _number(shot, window, [_page(window).findChild(QTableView)])
    _write(shot, f"{out}/recordings-added.png", _GUIDES_OUT)

    _go_to_rail(window, "metadata")
    info = _page(window)
    shot = window.grab()
    _box_widget(window, shot, _find_button(info, "Edit"))
    _number(
        shot,
        window,
        [
            info._table,
            _find_button(info, "Edit"),
            _find_button(info, "Import CSV"),
            _find_button(info, "Export CSV"),
        ],
        covers=info,
    )
    _write(shot, f"{out}/recording-information-view.png", _GUIDES_OUT)

    info._enter_edit()
    QApplication.processEvents()
    shot = window.grab()
    _box_widget(window, shot, _find_button(info, "Add column"))
    _box_widget(window, shot, _find_button(info, "Save"))
    _number(
        shot,
        window,
        [
            info._table,
            _find_button(info, "Save"),
            _find_button(info, "Cancel"),
            _find_button(info, "Add column"),
            _find_button(info, "Delete column"),
        ],
        covers=info,
    )
    _write(shot, f"{out}/recording-information.png", _GUIDES_OUT)
    info._cancel()
    window.close()


_DEMO_FOV = [(105, 55), (665, 55), (665, 598), (105, 598)]
_DEMO_CALIBRATION = [(120, 340), (650, 340)]
_DEMO_ROI = [(560, 60), (660, 60), (660, 590), (560, 590)]


def _draw_region(dialog, kind: str, points, name: str | None = None) -> None:
    """Do what a reader does in the editor: + the kind, then click its points."""
    dialog._add_region(kind)
    region = dialog._regions[dialog._selected]
    frame = next(iter(region["keyframes"]))
    region["keyframes"][frame] = _normalized(points)
    if name is not None:
        dialog._name.setText(name)
        dialog._on_rename()
    dialog._sync()
    dialog._sync_dirty()
    QApplication.processEvents()


#: Made up for the guide: a controller whose clock started 2.5 s after box1's video,
#: switching a light every 30 s and sounding a tone between. Not real data.
_DEMO_EVENTS = "seconds,label\n" + "".join(
    f"{t:.3f},{label}\n"
    for t, label in sorted(
        [(0.0, "session_start"), (290.0, "session_end")]
        + [(30.0 + 60 * i, "light_on") for i in range(5)]
        + [(60.0 + 60 * i, "light_off") for i in range(4)]
        + [(45.0 + 60 * i, "tone") for i in range(4)]
    )
)
_DEMO_EVENTS_START_FRAME = 75  # 2.5 s at 30 fps


def _demo_sensor_csv() -> str:
    """A made-up sensor log with its own column names, one row a second."""
    rows = ["Time (s),Temperature (C),Light (lux)"]
    for t in range(290):
        light = 310.0 if (t - 30) % 60 < 30 and t >= 30 else 12.0
        rows.append(f"{t},{21.4 + 0.002 * t:.3f},{light:.1f}")
    return "\n".join(rows) + "\n"


def _render_streams(demo_dir: Path) -> None:
    """Chapter 2's Streams section: adding a stream to box1, syncing it, previewing it.

    The demo has no data beside its videos, so the streams are made up for the guide
    (``_DEMO_EVENTS``, ``_demo_sensor_csv``), and drawn on a fresh experiment holding box1's
    video, referenced in place, so the demo itself is left without them.
    """
    out = "single-animal/set-up"
    win = sys.platform == "win32"
    clips = demo_dir.parent.parent / "top-down"
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        locations.experiments_root = lambda: parent
        data = parent / "data"
        data.mkdir()
        events = data / "box1_events.csv"
        events.write_text(_DEMO_EVENTS, encoding="utf-8")
        sensor = data / "box1_sensor.csv"
        sensor.write_text(_demo_sensor_csv(), encoding="utf-8")
        shown = {
            str(data): (
                r"C:\Users\yourname\Documents\BoxData"
                if win
                else "/home/yourname/Documents/BoxData"
            ),
            str(clips): (
                r"C:\Users\yourname\Videos" if win else "/home/yourname/Videos"
            ),
            str(parent): str(_DISPLAY_WORKSPACE),
        }

        fresh = Experiment.create("Top-down demo", parent_dir=parent)
        box1 = fresh.add_recording("box1")
        box1.add_video_part(clips / "box1.mp4", storage="reference")

        window = MainWindow()
        window._build_label.hide()
        window.resize(1040, 1340)
        window.show()
        window._open_experiment(fresh)
        _settle()
        window._tree.setCurrentItem(_rail_item(window, "recordings"))
        window._open_recording(box1.id)
        page = _page(window)

        dialog = streams.AddStreamDialog(box1, parent=window)
        dialog.resize(640, 330)
        dialog.show()
        dialog._data.set_path(str(events))
        dialog._name.setText("box_events")
        _settle()
        _mask_paths(dialog, shown)
        shot = dialog.grab()
        _number_screen(shot, dialog, dialog, f"{out}/add-stream.png")
        dialog._data._field.setText(
            str(events)
        )  # the masked path is only for the picture

        columns, rows = dialog._parse_rows()
        sync = streams.SyncStreamDialog.for_rows(box1, columns, rows, parent=dialog)
        landed: list[int] = []  # each frame decodes off the UI thread
        sync._scrubber.rendered.connect(lambda frame, _payload: landed.append(frame))
        sync.resize(1000, 820)
        sync.show()
        _wait_until(lambda: bool(landed) and not sync._decode_busy.isVisible())
        sync._scrubber.set_frame(_DEMO_EVENTS_START_FRAME)
        _wait_until(
            lambda: landed[-1] == _DEMO_EVENTS_START_FRAME
            and not sync._decode_busy.isVisible()
        )
        _settle()
        shot = sync.grab()
        _number_screen(shot, sync, sync, f"{out}/sync-stream.png")
        dialog._offset.setValue(sync._computed_offset() / 1e9)
        QDialog.reject(sync)
        _settle()
        dialog._import()
        _settle()
        page._rebuild_streams()
        _settle()
        _mask_paths(window, shown)
        shot = window.grab()
        _number(
            shot,
            window,
            [
                page._streams_table,
                _find_button(page, "Add stream"),
                _find_button(page, "Preview"),
                _find_button(page, "Sync to video"),
                _find_button(page, "Edit offset"),
                _find_button(page, "Delete stream"),
            ],
        )
        _write(shot, f"{out}/recording-streams.png", _GUIDES_OUT)

        preview = streams.StreamPreviewDialog(box1, "box_events", parent=window)
        preview.resize(1000, 860)
        preview.show()
        _wait_until(lambda: hasattr(preview, "_scrubber"))
        shown_frames: list[int] = []
        preview._scrubber.rendered.connect(
            lambda frame, _payload: shown_frames.append(frame)
        )
        light_on = (
            _DEMO_EVENTS_START_FRAME + 30 * 30
        )  # the first light_on, on the video
        preview._scrubber.set_frame(light_on)
        _wait_until(
            lambda: bool(shown_frames)
            and shown_frames[-1] == light_on
            and not preview._decode_busy.isVisible()
        )
        _settle()
        shot = preview.grab()
        marks = _in_reading_order(preview, preview)
        marks.insert(len(marks) - 1, preview._strip)  # clickable, before Close
        _number(shot, preview, marks, covers=preview)
        _write(shot, f"{out}/stream-preview.png", _GUIDES_OUT)
        QDialog.reject(preview)

        mapping = streams.AddStreamDialog(box1, parent=window)
        mapping.resize(640, 700)
        mapping.show()
        mapping._data.set_path(str(sensor))
        _settle()
        band = mapping._band
        band._time_kind.setCurrentIndex(band._time_kind.findData("seconds"))
        for box in band._table.findChildren(QCheckBox):
            box.setChecked(True)
        mapping._name.setText("box_sensor")
        _settle()
        _mask_paths(mapping, shown)
        shot = mapping.grab()
        _number_screen(shot, mapping, mapping, f"{out}/add-stream-mapping.png")
        QDialog.reject(mapping)
        QDialog.reject(dialog)
        window.close()


def _render_set_up_recordings(parent: Path, demo_dir: Path) -> None:
    """Chapter 2 of the user guide: a recording's page and its scene geometry.

    The before-and-while screens come from a fresh experiment in ``parent`` holding box1's
    video, referenced in place; the copy dialog and the finished page come from the demo
    experiment, whose geometry was saved by the same steps.
    """
    out = "single-animal/set-up"
    videos = (
        r"C:\Users\yourname\Videos"
        if sys.platform == "win32"
        else "/home/yourname/Videos"
    )
    clips = demo_dir.parent.parent / "top-down"
    shown = {str(clips): videos, str(parent): str(_DISPLAY_WORKSPACE)}

    fresh = Experiment.create("Top-down demo", parent_dir=parent)
    box1 = fresh.add_recording("box1")
    box1.add_video_part(clips / "box1.mp4", storage="reference")
    demo_values = Experiment(demo_dir).recording_metadata["recording_000"]
    for column in demo_values:
        if column not in fresh.metadata_columns:
            fresh.add_metadata_column(column)
    fresh.set_recording_metadata(box1.id, demo_values)

    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 1340)
    window.show()
    window._open_experiment(fresh)
    _settle()
    window._tree.setCurrentItem(_rail_item(window, "recordings"))
    window._open_recording(box1.id)
    page = _page(window)
    preview = page.findChild(arena.GeometryPreview)
    _wait_until(lambda: preview._openable)
    _mask_paths(window, shown)
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "Edit scene geometry"))
    fields = [
        e for e in page.findChildren(QLineEdit) if e.isVisible() and e is not page._name
    ]
    tables = [t for t in page.findChildren(QTableView) if t.isVisible()]
    _number(
        shot,
        window,
        [
            page._name,
            _find_button(page, "Save name"),
            fields,
            _find_button(page, "Add column"),
            _find_button(page, "Save metadata"),
            _find_button(page, "Add video part"),
            _find_button(page, "Locate video"),
            _find_button(page, "Move up"),
            _find_button(page, "Move down"),
            tables[0],
            preview,
            _find_button(page, "Edit scene geometry"),
            _find_button(page, "Add stream"),
            *tables[1:],
            _find_button(page, "Delete recording"),
        ],
        covers=page,
    )
    _write(shot, f"{out}/recording-page.png", _GUIDES_OUT)

    editor = arena.SceneGeometryDialog(box1, fresh, parent=window)
    landed: list[int] = []  # the first frame decodes off the UI thread
    editor._scrubber.rendered.connect(lambda frame, _payload: landed.append(frame))
    editor.show()
    editor.resize(1040, 900)
    _wait_until(lambda: bool(landed) and not editor._decode_busy.isVisible())
    shot = editor.grab()
    scrubber = editor._scrubber
    _number(
        shot,
        editor,
        [
            editor._zoom_out_btn,
            editor._zoom_in_btn,
            editor._zoom_fit_btn,
            editor._canvas.mapTo(editor, QPoint(16, 16)),
            *editor._add_buttons.values(),
            editor._list,
            editor._delete,
            scrubber.findChild(QAbstractSlider),
            _find_button(scrubber, "Play"),
            scrubber.findChild(QComboBox),
            _find_button(scrubber, "Prev frame"),
            _find_button(scrubber, "Next frame"),
            scrubber.findChild(QLineEdit),
            editor._set_kf,
            editor._del_kf,
            editor._undo,
            editor._clear,
            editor._name,
            editor._copy,
            editor._import,
            editor._export,
            editor._save,
            editor._close_button,
        ],
        covers=editor,
    )
    _write(shot, f"{out}/editor-empty.png", _GUIDES_OUT)

    _draw_region(editor, "arena", _DEMO_FOV)
    shot = editor.grab()
    _box_widget(editor, shot, editor._save)
    _write(shot, f"{out}/editor-field-of-view.png", _GUIDES_OUT)

    _draw_region(editor, "calibration", _DEMO_CALIBRATION)
    editor._distance.setValue(200.0)
    QApplication.processEvents()
    shot = editor.grab()
    _box_widget(editor, shot, editor._save)
    _number(shot, editor, [editor._distance, editor._unit])
    _write(shot, f"{out}/editor-calibration.png", _GUIDES_OUT)

    _draw_region(editor, "roi", _DEMO_ROI, name="Right wall")
    shot = editor.grab()
    _number(shot, editor, [editor._whole_video, editor._set_start, editor._set_end])
    _write(shot, f"{out}/editor-region.png", _GUIDES_OUT)
    editor.reject()
    window.close()

    locations.experiments_root = lambda: demo_dir.parent
    demo = Experiment(demo_dir)
    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 1340)
    window.show()
    window._open_experiment(demo)
    _settle()
    window._tree.setCurrentItem(_rail_item(window, "recordings"))
    window._open_recording("recording_000")
    page = _page(window)
    preview = page.findChild(arena.GeometryPreview)
    _wait_until(lambda: preview._openable)
    _mask_paths(
        window, {str(demo_dir.parent): str(_DISPLAY_WORKSPACE), str(clips): videos}
    )
    _write(window.grab(), f"{out}/recording-page-done.png", _GUIDES_OUT)
    window.close()

    # Copying is drawn where it has somewhere to go: the fresh experiment, with box1 set up
    # and its two neighbours not yet.
    box1.define_arena("field of view 1", [(0, _normalized(_DEMO_FOV))])
    box1.set_distance_calibration(_normalized(_DEMO_CALIBRATION), 200.0)
    for n in (2, 3):
        other = fresh.add_recording(f"box{n}")
        other.add_video_part(clips / f"box{n}.mp4", storage="reference")
        values = Experiment(demo_dir).recording_metadata[f"recording_00{n - 1}"]
        fresh.set_recording_metadata(other.id, values)
    copy = pages.CopyToRecordingsDialog(box1, fresh)
    copy.resize(560, 720)
    copy.show()
    _settle()
    shot = copy.grab()

    def position(widget):
        p = widget.mapTo(copy, QPoint(0, 0))
        return (p.y() // 12, p.x())  # one row per line of controls, left to right

    _number(shot, copy, sorted(_controls(copy), key=position), covers=copy)
    _write(shot, f"{out}/copy-dialog.png", _GUIDES_OUT)
    copy.reject()


def _render_set_up(demo_dir: Path) -> None:
    """Chapter 2: the recording page, the scene geometry editor, copying, and streams."""
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        locations.experiments_root = lambda: parent
        _render_set_up_recordings(parent, demo_dir)
    _render_streams(demo_dir)


def _runs_of(demo_dir: Path, module: str) -> list[str]:
    """The demo's processing runs of ``module``, oldest first."""
    from detail.storage import runs as runs_store

    found = []
    for run_id in sorted(runs_store.discover_runs(demo_dir)):
        if runs_store.read_run(demo_dir, run_id).get("module_id") == module:
            found.append(run_id)
    return found


def _recording_run_id(demo_dir: Path, run_id: str, recording_id: str) -> str:
    """The per-recording id a processing run gave ``recording_id`` (``det_01``, …)."""
    from detail.storage import runs as runs_store

    record = runs_store.read_run(demo_dir, run_id)
    entry = record.get("per_recording", {}).get(recording_id)
    if not entry:
        raise LookupError(f"{run_id} has no run on {recording_id}")
    return entry["run_id"]


#: The recording single-frame pictures are drawn from: box 3, whose mouse is the most often
#: clearly stretched out. Box 1's sits against the wall, curled, for much of its video.
_PICTURE_RECORDING = "recording_002"


def _elongation(
    demo_dir: Path, recording_id: str, frames=None, every: int = 10
) -> dict[int, tuple[float, int]]:
    """``{frame_idx: (elongation, area_px)}``: the outline's spread along its long axis over
    its spread along the short one, from the demo's newest detection masks.

    A picture of one frame shows a stretched-out animal, never a curled one, because a curled
    or grooming animal hides the body's line the points and outlines are about. The ratio of
    the mask's two spreads tells the two apart at any angle, where the box's aspect does not.
    ``frames`` limits it to those frames; otherwise every ``every``-th frame is measured.
    """
    import cv2
    import numpy as np

    from detail.core.utils import decode_mask
    from detail.executor import rundir
    from detail.storage import zarr_store

    det_id = _recording_run_id(
        demo_dir, _runs_of(demo_dir, "detection")[-1], recording_id
    )
    det_dir = demo_dir / "recordings" / recording_id / "detections" / det_id
    size = rundir.mask_resolution(det_dir, "part_001")
    group = zarr_store.open_group(det_dir / "part_001.zarr", mode="r")
    frame_idx = np.asarray(group["frame_idx"][:])
    wanted = (
        sorted({int(f) for f in frames})
        if frames is not None
        else list(range(0, int(frame_idx.max()) + 1, every))
    )
    out: dict[int, tuple[float, int]] = {}
    for row in np.flatnonzero(np.isin(frame_idx, wanted)):
        mask = decode_mask(str(group["mask_rle"][row]), size)
        area = int(mask.sum())
        if area < 50:
            continue
        # The body alone: an opening strips the tail, a thin line that would make a curled
        # animal with its tail held out read as a long one.
        k = max(3, int(0.25 * area**0.5)) | 1
        body = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)
        )
        ys, xs = np.nonzero(body)
        if xs.size < 50:
            continue
        spread = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float)))
        ratio = float(np.sqrt(spread[-1] / max(spread[0], 1e-9)))
        out[int(frame_idx[row])] = (ratio, area)
    return out


def _most_elongated(demo_dir: Path, recording_id: str, frames=None) -> int:
    """The frame of ``recording_id`` (among ``frames``, if given) whose animal is clearly
    stretched out: see :func:`_elongation`.

    Not the single most elongated outline, which is a mask glitch or a tail held straight
    out from a curled body as often as a long animal: among outlines of about the usual
    area, the most elongated below the top 3% of the recording's elongations.
    """
    import numpy as np

    every = _elongation(demo_dir, recording_id)
    found = every if frames is None else _elongation(demo_dir, recording_id, frames)
    if not found:
        raise LookupError(f"no outlined frame to choose on {recording_id}")
    ratios = np.array([r for r, _ in every.values()])
    areas = np.array([a for _, a in every.values()])
    ceiling = float(np.percentile(ratios, 97))
    usual = float(np.median(areas))
    fitting = {
        f: r for f, (r, a) in found.items() if r <= ceiling and 0.8 <= a / usual <= 1.25
    }
    pool = fitting or {f: r for f, (r, _) in found.items()}
    return max(pool, key=pool.get)


def _well_placed_frame(demo_dir: Path, recording_id: str, kt_id: str) -> int:
    """A frame of ``recording_id`` to show keypoints on: an elongated mouse whose points sit
    where they belong.

    Among frames where the head–tail line is near its usual full length, every point is
    confident and the midpoint sits on the line's middle (a straight, stretched body), the
    one with the most elongated outline (:func:`_elongation`). A frame chosen by the outline
    alone can show the tail point partway up the back.
    """
    import numpy as np

    from detail.storage import zarr_store

    track = zarr_store.open_group(
        demo_dir
        / "recordings"
        / recording_id
        / "keypoint_track"
        / kt_id
        / "track_0"
        / "part_001.zarr",
        mode="r",
    )
    frame_idx = np.asarray(track["frame_idx"][:])
    points = np.asarray(track["points"][:, :, :2])
    scores = np.asarray(track["point_scores"][:])
    length = np.linalg.norm(points[:, 0] - points[:, -1], axis=1)
    usual = float(np.nanpercentile(length, 95))
    middle = (points[:, 0] + points[:, -1]) / 2
    off_line = np.linalg.norm(points[:, len(points[0]) // 2] - middle, axis=1)
    placed = (
        (length >= 0.85 * usual)
        & (length <= 1.1 * usual)
        & (scores.min(axis=1) >= 0.8)
        & (off_line <= 0.05 * np.maximum(length, 1e-9))
    )
    candidates = frame_idx[placed][::3]
    if candidates.size == 0:
        return _most_elongated(demo_dir, recording_id)
    return _most_elongated(demo_dir, recording_id, candidates)


def _choose_item(combo: QComboBox, text: str) -> None:
    """Pick the first entry of ``combo`` whose label contains ``text``."""
    items = [combo.itemText(i) for i in range(combo.count())]
    matches = [i for i, item in enumerate(items) if text in item]
    if not matches:
        # The run pickers list settings, not run ids: the only real entry is the run.
        real = [i for i, item in enumerate(items) if not item.startswith("(")]
        if len(real) != 1:
            raise LookupError(f"no entry containing {text!r} in {items}")
        matches = real
    combo.setCurrentIndex(matches[0])


def _render_detect(demo_dir: Path, part: str = "all") -> None:
    """Chapter 3: distill a detector, detect, track.

    The before screens (an empty Models page, the Distill dialog, its prompt test) come from
    a fresh experiment holding the demo's three videos, referenced in place; the run pages
    and launch dialogs come from the demo experiment, whose runs were made by these steps.
    """
    from detail.gui.fit import DistillDialog
    from detail.gui.launch import LaunchRunDialog
    from detail.gui.preview import PromptTestDialog

    out = "single-animal/detect"
    clips = demo_dir.parent.parent / "top-down"
    demo = Experiment(demo_dir)
    if part == "runs":
        _render_detect_runs(demo_dir, demo)
        return

    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        locations.experiments_root = lambda: parent
        fresh = Experiment.create("Top-down demo", parent_dir=parent)
        for column in demo.metadata_columns:
            fresh.add_metadata_column(column)
        for n in (1, 2, 3):
            recording = fresh.add_recording(f"box{n}")
            recording.add_video_part(clips / f"box{n}.mp4", storage="reference")
            fresh.set_recording_metadata(
                recording.id, demo.recording_metadata[f"recording_00{n - 1}"]
            )

        window = MainWindow()
        window._build_label.hide()
        window.resize(*_WIN_SIZE)
        window.show()
        window._open_experiment(fresh)
        _settle()
        _go_to_rail(window, "models")
        page = _page(window)
        shot = window.grab()
        _box_widget(window, shot, _find_button(page, "＋ Distill new model"))
        _number_screen(shot, window, page, f"{out}/models-empty.png")

        dialog = DistillDialog(fresh, parent=window)
        dialog._name.setText("mouse detector")
        dialog._class_rows[0][0].setText("mouse")
        dialog._class_rows[0][1].setText("mouse")
        dialog._n_frames.setValue(500)
        dialog.show()
        _settle()
        shot = dialog.grab()
        _box_widget(dialog, shot, _find_button(dialog, "Distill"))
        _number_screen(shot, dialog, dialog, f"{out}/distill-dialog.png")

        toggle = _open_advanced(dialog)
        dialog._hparams["batch_size"].setValue(8)
        _settle()
        # The dialog scrolls; bring Advanced Settings to its top so every row shows.
        area = dialog.findChild(QScrollArea)
        area.verticalScrollBar().setValue(
            toggle.mapTo(area.widget(), QPoint(0, 0)).y() - 6
        )
        _settle()
        shot = dialog.grab()
        top = toggle.mapTo(dialog, QPoint(0, 0)).y()
        bottom = min(
            dialog._class_rows[0][0].mapTo(dialog, QPoint(0, 0)).y(),
            _find_button(dialog, "Distill").mapTo(dialog, QPoint(0, 0)).y(),
            area.viewport().mapTo(dialog, QPoint(0, area.viewport().height())).y(),
        )
        advanced = _below(_in_reading_order(dialog, dialog), dialog, top, bottom)
        helps = _below(_help_buttons(dialog), dialog, top, bottom)
        _box_widget(dialog, shot, dialog._hparams["batch_size"])
        _number(shot, dialog, [*advanced, helps] if helps else advanced)
        _write(shot, f"{out}/distill-advanced.png", _GUIDES_OUT)
        print(f"--- legend for {out}/distill-advanced.png")
        for n, w in enumerate(advanced, start=1):
            print(f"  {n}. {_describe(w)}")
        dialog.reject()
        if part == "dialogs":
            window.close()
            return
        test = PromptTestDialog(
            list(fresh.recordings),
            model_id="sam3",
            prompt_text="mouse",
            compute=ComputeTarget(
                backend="cuda",
                device_indices=[0],
                cloud_provider=None,
                cloud_instance=None,
            ),
            parent=window,
        )
        test.resize(1000, 900)
        test.show()
        _settle()
        # The dialog reports a failure in a modal box no one can answer offscreen.
        failures: list[str] = []
        test._on_error = lambda message: failures.append(message)
        test._worker = None
        test._start()
        test._worker.failed.disconnect()
        test._worker.failed.connect(test._on_error)
        _wait_until(lambda: test._results_widget is not None or failures, seconds=900)
        if failures:
            raise RuntimeError(f"prompt test failed: {failures[0]}")
        _settle()
        shot = test.grab()
        _number_screen(shot, test, test, f"{out}/prompt-test.png")
        test.reject()
        window.close()
    if part == "all":
        _render_detect_runs(demo_dir, demo)


def _render_detect_runs(demo_dir: Path, demo: Experiment) -> None:
    """Steps 2 to 4 from the demo's own runs: the distill, detection and tracking."""
    from detail.gui.launch import LaunchRunDialog

    out = "single-animal/detect"
    model_id = sorted(m.id for m in demo.models)[0]
    detection_run = _runs_of(demo_dir, "detection")[-1]
    tracking_run = _runs_of(demo_dir, "tracking")[-1]
    distill_run = sorted(p.name for p in (demo_dir / "distill").glob("distill_*"))[0]
    window = _demo_window(demo_dir, size=(1040, 900))
    _go_to_rail(window, "models")
    window._open_distill_run(distill_run)
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/distill-run.png")

    _go_to_rail(window, "runs")
    page = _page(window)
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "New run"))
    _number_screen(shot, window, page, f"{out}/pipeline.png")

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    launch._select_all_recordings()
    _choose_item(launch._config._model, model_id)
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, _find_button(launch, "Run"))
    _number_screen(shot, launch, launch, f"{out}/launch-detection.png")
    toggle = _open_advanced(launch)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    shot = launch.grab()
    top = toggle.mapTo(launch, QPoint(0, 0)).y()
    advanced = [
        w
        for w in _below(_in_reading_order(launch, launch), launch, top)
        if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
    ]
    _number(shot, launch, advanced)
    _write(shot, f"{out}/launch-detection-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/launch-detection-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")
    launch.reject()

    window._open_run(detection_run)
    _settle()
    _pump(3)
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/detection-run.png")

    det_id = _recording_run_id(demo_dir, detection_run, "recording_000")
    window._open_recording_run("recording_000", "detection", det_id)
    _pump(10)
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/detection-result.png")

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("tracking"))
    _settle()
    launch._select_all_recordings()
    _settle()
    form = launch._config
    _wait_until(
        lambda: form._arena.count() > 1
    )  # the fields of view resolve off-thread
    _choose_item(form._arena, "field of view 1")
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, form._arena)
    _number_screen(shot, launch, launch, f"{out}/launch-tracking.png")
    toggle = _open_advanced(launch)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    shot = launch.grab()
    top = toggle.mapTo(launch, QPoint(0, 0)).y()
    advanced = [
        w
        for w in _below(_in_reading_order(launch, launch), launch, top)
        if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
    ]
    _number(shot, launch, advanced)
    _write(shot, f"{out}/launch-tracking-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/launch-tracking-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")
    launch.reject()

    window._open_run(tracking_run)
    _settle()
    _pump(3)
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/tracking-run.png")

    track_id = _recording_run_id(demo_dir, tracking_run, "recording_000")
    window._open_recording_run("recording_000", "tracking", track_id)
    _pump(10)
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/tracking-result.png")
    window.close()


def _render_keypoints(demo_dir: Path) -> None:
    """Chapter 4: a keypoint model distilled from an example set, then keypoint detect and
    keypoint track.

    Drawn from the demo experiment: its example set ``ex_01`` and its keypoint distill (the
    newest distill run) were made by these steps.
    """
    from detail.gui import scrubber as scrubber_module
    from detail.gui.example_set_picker import ExampleSetPicker, NewExampleSetDialog
    from detail.gui.fit import DistillDialog
    from detail.gui.keypoint_labeling import KeypointLabelingDialog
    from detail.gui.keypoint_test import KeypointTestDialog
    from detail.gui.launch import LaunchRunDialog

    out = "single-animal/keypoints"
    demo = Experiment(demo_dir)
    set_id = demo.example_sets()[0].id
    tracking_run = _runs_of(demo_dir, "tracking")[-1]
    detection_run = _runs_of(demo_dir, "detection")[-1]
    kt_run = _runs_of(demo_dir, "keypoint_track")[-1]
    tracks = {
        rid: _recording_run_id(demo_dir, tracking_run, rid)
        for rid in ("recording_000", "recording_001", "recording_002")
    }
    keypoint_model = sorted(m.id for m in demo.models)[-1]
    keypoint_distill = sorted(p.name for p in (demo_dir / "distill").glob("distill_*"))[
        -1
    ]

    window = _demo_window(demo_dir, size=(1040, 900))
    _go_to_rail(window, "models")
    page = _page(window)
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "＋ Distill new model"))
    _number_screen(shot, window, page, f"{out}/models-page.png")

    dialog = DistillDialog(demo, parent=window, model_type="keypoints")
    dialog._name.setText("mouse keypoints")
    form = dialog.keypoint_form()
    dialog.show()
    _pump(3)
    _settle()
    form._examples.select(set_id)
    form._fields["n_frames"].setValue(300)
    _pump(2)
    _settle()
    dialog.resize(dialog.width(), 900)
    _settle()
    shot = dialog.grab()
    _box_widget(dialog, shot, _find_button(dialog, "Distill"))
    _number_screen(shot, dialog, dialog, f"{out}/distill-keypoints.png")

    toggle = _open_advanced(dialog)
    for name, widget in form._hparams.items():
        if name == "batch_size":
            widget.setValue(8)
    _settle()
    area = dialog.findChild(QScrollArea)
    if area is not None:
        area.verticalScrollBar().setValue(
            toggle.mapTo(area.widget(), QPoint(0, 0)).y() - 6
        )
    _settle()
    shot = dialog.grab()
    top = toggle.mapTo(dialog, QPoint(0, 0)).y()
    bottom = _find_button(dialog, "Distill").mapTo(dialog, QPoint(0, 0)).y()
    if area is not None:
        bottom = min(
            bottom,
            area.viewport().mapTo(dialog, QPoint(0, area.viewport().height())).y(),
        )
    advanced = _below(_in_reading_order(dialog, dialog), dialog, top, bottom)
    helps = _below(_help_buttons(dialog), dialog, top, bottom)
    _number(shot, dialog, [*advanced, helps] if helps else advanced)
    _write(shot, f"{out}/distill-keypoints-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/distill-keypoints-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")

    picker = ExampleSetPicker(demo)
    new_set = NewExampleSetDialog(
        picker._entries, picker.skeleton_id(), "mouse overhead", parent=window
    )
    new_set.show()
    _settle()
    shot = new_set.grab()
    _number_screen(shot, new_set, new_set, f"{out}/new-example-set.png")
    new_set.reject()

    labeling = KeypointLabelingDialog(demo, set_id, sources=tracks, parent=window)
    labeling.resize(1200, 860)
    labeling.show()
    _pump(8)
    # A marked frame, so the skeleton shows, and the one whose mouse is most stretched out.
    frames = demo.example_set(set_id).frames
    labeling._go_to(
        max(
            range(len(frames)),
            key=lambda i: _elongation(
                demo_dir, frames[i].recording_id, [frames[i].frame_idx]
            ).get(frames[i].frame_idx, (0.0, 0))[0],
        )
    )
    _pump(6)
    shot = labeling.grab()
    _number_screen(shot, labeling, labeling, f"{out}/labeling.png")
    labeling.reject()

    test = KeypointTestDialog(demo, form.test_teacher(), cutoff=0.75, parent=window)
    test.resize(1100, 900)
    test.show()
    _wait_until(lambda: test._result is not None, seconds=900)
    _pump(3)
    shot = test.grab()
    _number_screen(shot, test, test, f"{out}/test-cutoff.png")
    test.reject()
    QDialog.reject(dialog)

    window._open_distill_run(keypoint_distill)
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/distill-run.png")

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("keypoint_detect"))
    _settle()
    launch._select_all_recordings()
    _pump(3)
    form = launch._config
    _choose_item(form._detection, detection_run)
    _choose_item(form._method, keypoint_model)
    _pump(2)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, form._method)
    _number_screen(shot, launch, launch, f"{out}/launch-keypoint-detect.png")
    launch.reject()

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("keypoint_track"))
    _settle()
    launch._select_all_recordings()
    _pump(3)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    shot = launch.grab()
    _number_screen(shot, launch, launch, f"{out}/launch-keypoint-track.png")
    toggle = _open_advanced(launch)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1600))
    _settle()
    shot = launch.grab()
    top = toggle.mapTo(launch, QPoint(0, 0)).y()
    advanced = [
        w
        for w in _below(_in_reading_order(launch, launch), launch, top)
        if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
    ]
    _number(shot, launch, advanced)
    _write(shot, f"{out}/launch-keypoint-track-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/launch-keypoint-track-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")
    launch.reject()

    kt_id = _recording_run_id(demo_dir, kt_run, _PICTURE_RECORDING)
    window._open_recording_run(_PICTURE_RECORDING, "keypoint_track", kt_id)
    _pump(12)
    page = _page(window)
    page.findChild(scrubber_module.FrameScrubber).set_frame(
        _well_placed_frame(demo_dir, _PICTURE_RECORDING, kt_id)
    )
    _pump(12)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/keypoint-result.png")
    _save_callouts()  # before close: a page with a live player can hang it
    window.close()


def _render_new_skeleton(demo_dir: Path) -> None:
    """Chapter 4's *Optional: make your own skeleton*: the Keypoints manager and a 5-point
    skeleton (the midline with the ears) drawn in the editor. Nothing is saved.

    The library is shown with its presets only, so this machine's own skeletons stay out of
    the picture.
    """
    from detail.api import skeletons as skeleton_library
    from detail.core.skeleton import Skeleton
    from detail.gui.keypoints_manager import SkeletonEditorDialog

    out = "single-animal/keypoints"
    presets = [e for e in skeleton_library.skeletons() if e["builtin"]]
    skeleton_library.skeletons = lambda: list(presets)
    skeleton_library.dependents_by_skeleton = lambda: {}

    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 700)
    window.show()
    window._switch_mode("keypoints")
    _pump(3)
    _settle()
    page = _page(window)
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "+ New skeleton"))
    _number_screen(shot, window, page, f"{out}/keypoints-manager.png")

    dialog = SkeletonEditorDialog(None, [], {e["name"] for e in presets}, parent=window)
    dialog._name.setText("mouse 5-point")
    # Seen from above with the nose up the screen, the animal's left ear is on the left.
    dialog.editor().load(
        Skeleton(
            names=("nose", "left_ear", "right_ear", "back", "tail_base"),
            edges=(
                ("nose", "left_ear"),
                ("nose", "right_ear"),
                ("nose", "back"),
                ("back", "tail_base"),
            ),
            pairs=(("left_ear", "right_ear"),),
            layout=((0.5, 0.14), (0.38, 0.27), (0.62, 0.27), (0.5, 0.52), (0.5, 0.86)),
        )
    )
    editor = dialog.editor()
    editor._selected = 1  # left_ear: its pair and side show in the row under the canvas
    editor._show()
    dialog._sync()
    dialog.resize(760, 820)
    dialog.show()
    _settle()
    shot = dialog.grab()
    _number_screen(shot, dialog, dialog, f"{out}/new-skeleton.png")
    # Closed past its own reject, which would ask about the unsaved skeleton in a modal box.
    QDialog.reject(dialog)
    window.close()


def _render_keypoints_chapter(demo_dir: Path) -> None:
    _render_new_skeleton(demo_dir)
    _render_keypoints(demo_dir)


def _analyses_of(demo_dir: Path, module: str) -> list[str]:
    """The demo's analysis runs of ``module``, oldest first."""
    import yaml

    found = []
    for run_dir in sorted((demo_dir / "analysis").glob("an_*")):
        record = yaml.safe_load(
            (run_dir / "analysis.yaml").read_text(encoding="utf-8-sig")
        )
        if record.get("module_id") == module:
            found.append(run_dir.name)
    return found


def _render_embedding(demo_dir: Path) -> None:
    """Chapter 5: embed the tracked animal, turned by its keypoints."""
    from detail.gui.launch import LaunchRunDialog
    from detail.gui.modules._heading import HeadingDefinitionDialog
    from detail.gui.run_preview import EmbeddingRunPreview

    out = "single-animal/embedding"
    demo = Experiment(demo_dir)
    tracking_run = _runs_of(demo_dir, "tracking")[-1]
    kt_run = _runs_of(demo_dir, "keypoint_track")[-1]
    embedding_run = _runs_of(demo_dir, "embedding")[-1]
    window = _demo_window(demo_dir, size=(1040, 900))

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("embedding"))
    _settle()
    launch._select_all_recordings()
    _pump(3)
    form = launch._config
    _choose_item(form._tracking, tracking_run)
    _pump(2)
    form._align.setCurrentIndex(form._align.findData("rotate"))
    _pump(2)
    _choose_item(form._keypoint, kt_run)
    _pump(2)
    form._heading.set_name("body_axis")
    _pump(2)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, form._heading)
    _number_screen(shot, launch, launch, f"{out}/launch-embedding.png")
    toggle = _open_advanced(launch)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1600))
    _settle()
    shot = launch.grab()
    top = toggle.mapTo(launch, QPoint(0, 0)).y()
    advanced = [
        w
        for w in _below(_in_reading_order(launch, launch), launch, top)
        if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
    ]
    _number(shot, launch, advanced)
    _write(shot, f"{out}/launch-embedding-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/launch-embedding-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")

    names = form._heading._skeleton_names
    heading = HeadingDefinitionDialog(demo, skeleton_names=names, parent=launch)
    heading._name.setText("body_axis")
    for widget, ticked in (
        (heading._front_list, ("head",)),
        (heading._back_list, ("tail",)),
    ):
        if widget is None:
            continue
        for i in range(widget.count()):
            if widget.item(i).text() in ticked:
                widget.item(i).setCheckState(Qt.CheckState.Checked)
    heading.show()
    heading.resize(heading.width(), max(heading.height(), 640))
    _settle()
    shot = heading.grab()
    _number_screen(shot, heading, heading, f"{out}/define-heading.png")
    heading.reject()
    launch.reject()

    window.resize(1040, 1180)  # the cloud under the table: Delete run stays in view
    _go_to_rail(window, "runs")
    window._open_run(embedding_run)
    _settle()
    page = _page(window)
    # Box 3, whose mouse is the clearest stretched out; the point clicked is the frame nearest
    # its most elongated outline (see _most_elongated).
    results = page.findChild(pages.RunResults)
    results._table.view.selectRow(results._recording_ids.index(_PICTURE_RECORDING))
    _pump(3)
    preview = page.findChild(EmbeddingRunPreview)
    _wait_until(lambda: preview._preview is not None, seconds=300)
    _pump(2)
    points = preview._scatter.points()
    target = _most_elongated(demo_dir, _PICTURE_RECORDING)
    nearest = min(points, key=lambda pt: abs(preview._locators[pt.data()][2] - target))
    preview._on_clicked(None, [nearest])
    _pump(15)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/embedding-run.png")
    _save_callouts()
    window.close()


def _render_location_export(demo_dir: Path) -> None:
    """Chapter 6: Location export over the demo's tracking, with each box's *Right wall*
    region and *Port* point. The launch form is drawn seeded from the demo's run, as *Start
    from* would fill it."""
    import yaml

    from detail.gui.launch import LaunchAnalysisDialog

    out = "single-animal/location-export"
    demo = Experiment(demo_dir)
    run_id = _analyses_of(demo_dir, "location_export")[-1]
    window = _demo_window(demo_dir, size=(1040, 900))

    record = yaml.safe_load(
        (demo_dir / "analysis" / run_id / "analysis.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    launch = LaunchAnalysisDialog(demo, parent=window, seed=record)
    launch.show()
    _settle()
    launch._select_all_recordings()
    _pump(3)
    runs = launch._config._tracking.widget
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1200))
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, runs)
    _number_screen(shot, launch, launch, f"{out}/launch-location-export.png")
    launch.reject()

    _go_to_rail(window, "analysis")
    window._open_analysis_run(run_id)
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/location-export-run.png")

    page._show_view("Files")
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/location-export-files.png")
    _save_callouts()
    window.close()


def _render_behavior_scan(demo_dir: Path) -> None:
    """Chapter 7: a behavior scan for rearing and grooming.

    The demo's scan holds the example clips chosen for the guide and the cut-offs chosen for
    it (0.5 each); the launch form is drawn seeded from it, as *Start from* would fill it.
    """
    import yaml

    from detail.gui.behavior_query import ClipExampleDialog
    from detail.gui.launch import LaunchAnalysisDialog

    out = "single-animal/behavior-scan"
    demo = Experiment(demo_dir)
    run_id = _analyses_of(demo_dir, "query")[-1]
    window = _demo_window(demo_dir, size=(1040, 900))
    _go_to_rail(window, "analysis")
    page = _page(window)
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "New analysis"))
    _number_screen(shot, window, page, f"{out}/analysis-page.png")

    record = yaml.safe_load(
        (demo_dir / "analysis" / run_id / "analysis.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    launch = LaunchAnalysisDialog(demo, parent=window, seed=record)
    launch.show()
    _settle()
    launch._select_all_recordings()
    _pump(3)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    form = launch._config
    shot = launch.grab()
    _box_widget(launch, shot, form._mark_examples)
    _number_screen(shot, launch, launch, f"{out}/launch-behavior-scan.png")
    toggle = _open_advanced(launch)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1600))
    _settle()
    shot = launch.grab()
    top = toggle.mapTo(launch, QPoint(0, 0)).y()
    advanced = [
        w
        for w in _below(_in_reading_order(launch, launch), launch, top)
        if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
    ]
    _number(shot, launch, advanced)
    _write(shot, f"{out}/launch-behavior-scan-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/launch-behavior-scan-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")

    marking = ClipExampleDialog(
        demo,
        form._recordings,
        form._embed.currentData(),
        parent=launch,
        max_gap_ns=form._split_gap_ns(),
        require_contiguous=False,
        behaviors=form._behaviors,
        active=0,
        negatives_reason=form._negatives_reason(),
    )
    marking.resize(1200, 860)
    marking.show()
    _pump(8)
    table = marking._clip_table
    table.setCurrentIndex(table.model().index(0, 2))
    _pump(10)
    shot = marking.grab()
    _number_screen(shot, marking, marking, f"{out}/mark-examples.png")
    marking.reject()
    launch.reject()

    window._open_analysis_run(run_id)
    _pump(5)
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/scan-run.png")

    from detail.gui.modules.analysis_query import _QueryWorkspace

    # The views are drawn on their own, as tall as the tallest (Calibrate) so the whole of
    # it is laid out; a shorter view is capped to what it holds, Finish and Reopen at its
    # foot.
    window.resize(1240, 2000)
    for name, height in (
        ("Calibrate", None),
        ("Audit", 520),
        ("Browse", 860),
        ("Measures", 760),
    ):
        page._show_view(name)
        _pump(10)
        _settle()
        workspace = _page(window).findChild(_QueryWorkspace)
        workspace.setMaximumHeight(height or 16_777_215)
        _pump(5)
        _settle()
        if name == "Browse":  # the first instance playing beside the list
            table = workspace._browse.findChild(QTableView)
            table.setCurrentIndex(table.model().index(0, 0))
            _pump(10)
        shot = workspace.grab()
        _number_screen(shot, workspace, workspace, f"{out}/{name.lower()}.png")
    _save_callouts()
    window.close()


def _render_run_export(demo_dir: Path) -> None:
    """The top bar's **Export…** dialog on a processing run, as the run's page opens it:
    chapter 3's on the tracking run, chapter 4's on the keypoint track run. Nothing is
    exported: the dialog is drawn, then closed."""
    from detail.gui import export as gui_export
    from detail.gui.export import ExportDialog

    saved = (
        r"C:\Users\yourname\Documents\Exports"
        if sys.platform == "win32"
        else "/home/yourname/Documents/Exports"
    )
    window = _demo_window(demo_dir, size=(1240, 900))
    shots = (
        (
            _runs_of(demo_dir, "tracking")[-1],
            "single-animal/detect/export-tracks.png",
            {"tracks"},
        ),
        (
            _runs_of(demo_dir, "keypoint_track")[-1],
            "single-animal/keypoints/export-keypoints.png",
            {"keypoints_dlc"},
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        for run_id, rel, ticked in shots:
            window._open_run(run_id)
            _settle()
            page = _page(window)
            subject = window._export._subject
            dialog = ExportDialog(
                subject,
                gui_export.exporters.targets_for(subject),
                gui_export.scope_of(page),
                window,
            )
            for target_id, check in dialog._include.items():
                if check.isEnabled():
                    check.setChecked(target_id in ticked)
            dialog._out.setText(tmp)
            dialog.resize(760, 1040)
            dialog.show()
            _settle()
            _wait_until(
                lambda: dialog._estimate_label.text() not in ("", "Estimating…"),
                seconds=120,
            )
            _settle()
            # Masked without a re-estimate: a changed folder restarts it, and the picture
            # would show "Estimating…".
            dialog._out.blockSignals(True)
            _mask_paths(dialog, {tmp: saved})
            # As tall as its form, so nothing scrolls and nothing is left blank.
            scroll = dialog.findChild(QScrollArea)
            if scroll is not None:
                dialog.resize(
                    760,
                    dialog.height()
                    - scroll.viewport().height()
                    + scroll.widget().sizeHint().height(),
                )
            _settle()
            shot = dialog.grab()
            # The rows this kind of run cannot write are one mark: they say why themselves.
            greyed = [c for c in dialog._include.values() if not c.isEnabled()]
            marks = []
            for w in _in_reading_order(dialog, dialog):
                if w in greyed:
                    if w is greyed[0]:
                        marks.append(greyed)
                else:
                    marks.append(w)
            _number(shot, dialog, marks, covers=dialog)
            _write(shot, rel, _GUIDES_OUT)
            print(f"--- legend for {rel}")
            for n, mark in enumerate(marks, start=1):
                first = mark[0] if isinstance(mark, list) else mark
                print(f"  {n}. {_describe(first)}")
            dialog.reject()
    window.close()


def _cluster_window(demo_dir: Path, size=(1240, 1000)):
    """The demo's clustering run, opened on its landing page: (window, run id, page)."""
    run_id = _analyses_of(demo_dir, "temporal_clustering")[-1]
    window = _demo_window(demo_dir, size=size)
    _go_to_rail(window, "analysis")
    window._open_analysis_run(run_id)
    _pump(5)
    _settle()
    return window, run_id, _page(window)


def _render_clustering(demo_dir: Path, node: str = "", part: str = "all") -> None:
    """Chapter 8: a behavior clustering of the demo, its Interpret view, a cluster's
    window and clips, labelling, Audit, Browse and Measures."""
    import yaml

    from detail.gui.cluster_workspace import ClusterWorkspace
    from detail.gui.launch import LaunchAnalysisDialog

    out = "single-animal/clustering"
    demo = Experiment(demo_dir)
    window, run_id, page = _cluster_window(demo_dir)

    if part in ("all", "launch", "launch-only"):
        record = yaml.safe_load(
            (demo_dir / "analysis" / run_id / "analysis.yaml").read_text(
                encoding="utf-8-sig"
            )
        )
        launch = LaunchAnalysisDialog(demo, parent=window, seed=record)
        launch.show()
        _settle()
        launch._select_all_recordings()
        _pump(3)
        launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
        _settle()
        shot = launch.grab()
        _number_screen(shot, launch, launch, f"{out}/launch-temporal_clustering.png")
        toggle = _open_advanced(launch)
        launch.resize(launch.width(), min(launch.sizeHint().height(), 1800))
        _settle()
        # The dialog scrolls; bring Advanced Settings to its top so every row shows.
        area = launch.findChild(QScrollArea)
        area.verticalScrollBar().setValue(
            toggle.mapTo(area.widget(), QPoint(0, 0)).y() - 6
        )
        _settle()
        shot = launch.grab()
        top = toggle.mapTo(launch, QPoint(0, 0)).y()
        bottom = area.viewport().mapTo(launch, QPoint(0, area.viewport().height())).y()
        advanced = [
            w
            for w in _below(_in_reading_order(launch, launch), launch, top, bottom)
            if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
        ]
        _number(shot, launch, advanced)
        _write(shot, f"{out}/launch-temporal_clustering-advanced.png", _GUIDES_OUT)
        print(f"--- legend for {out}/launch-temporal_clustering-advanced.png")
        for n, w in enumerate(advanced, start=1):
            print(f"  {n}. {_describe(w)}")
        launch.reject()
    if part == "launch-only":
        window.close()
        return

    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/clustering-run.png")

    page._show_view("Interpret")
    _pump(20)
    _settle()
    workspace = _page(window).findChild(ClusterWorkspace)
    interpret = workspace._interpret
    if node:
        interpret.select(int(node))
        _pump(25)
        _settle()
    shot = window.grab()
    _number_screen(shot, window, interpret, f"{out}/interpret.png")
    _save_callouts()
    window.close()


#: The demo's labels for chapter 8, put on by eye from each cluster's montage: the guide's own,
#: not careful scoring. Cluster ids are the demo clustering's.
_DEMO_LABELS = {20: "walking", 3: "staying in place"}
_DEMO_CLUSTER = 20


def _render_clustering_views(demo_dir: Path) -> None:
    """Chapter 8 after the launch: the run's page, Interpret before and after labelling,
    the filters, a cluster's window, a clip's window, Audit, Browse and Measures.

    It starts by clearing the demo's two labels, draws the screens a reader sees before
    labelling, then puts the labels back on, as chapter 8's step 5 does, and draws the rest.
    The demo is left labelled either way.
    """
    from detail.api.cluster_filter import Selection
    from detail.gui.clip_popup import ClipPopup
    from detail.gui.cluster_workspace import ClusterWorkspace

    out = "single-animal/clustering"
    demo = Experiment(demo_dir)
    run_id = _analyses_of(demo_dir, "temporal_clustering")[-1]
    run = demo.temporal_clustering_run(run_id)
    run.name_nodes({node: "" for node in _DEMO_LABELS})

    window, run_id, page = _cluster_window(demo_dir)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/clustering-run.png")

    def interpret_view():
        page = _page(window)
        page._show_view("Interpret")
        _pump(20)
        _settle()
        return _page(window).findChild(ClusterWorkspace)

    workspace = interpret_view()
    interpret = workspace._interpret
    interpret.select(_DEMO_CLUSTER)
    _pump(25)
    _settle()
    shot = window.grab()
    _number_screen(shot, window, interpret, f"{out}/interpret.png")

    interpret._filters.set_selection(Selection(values={"drug": frozenset({"on"})}))
    _pump(20)
    _settle()
    shot = window.grab()
    _box_widget(window, shot, interpret._dock)
    _write(shot, f"{out}/filters.png", _GUIDES_OUT)
    interpret._filters.set_selection(Selection())
    _pump(10)

    cluster = interpret.open_window(_DEMO_CLUSTER)
    cluster.resize(1000, 820)
    cluster.show()
    _pump(25)
    _settle()
    shot = cluster.grab()
    _number_screen(shot, cluster, cluster, f"{out}/cluster-window.png")
    cluster.close()
    _pump(3)

    interpret._play_clip(0)
    _pump(15)
    popup = next(
        w
        for w in QApplication.topLevelWidgets()
        if isinstance(w, ClipPopup) and w.isVisible()
    )
    popup.resize(1000, 860)
    _pump(15)
    _settle()
    shot = popup.grab()
    _number_screen(shot, popup, popup, f"{out}/clip-window.png")
    popup.close()
    _pump(3)

    run.name_nodes(dict(_DEMO_LABELS))
    interpret.reload()
    interpret.changed.emit()  # Audit and Browse re-read, as after a label typed in the app
    _pump(20)
    _settle()
    interpret.select(_DEMO_CLUSTER)
    _pump(25)
    _settle()
    shot = window.grab()
    _box_widget(window, shot, interpret._label_bar)
    _write(shot, f"{out}/labelled.png", _GUIDES_OUT)

    page = _page(window)
    page._show_view("Audit")
    _pump(15)
    _settle()
    workspace = _page(window).findChild(ClusterWorkspace)
    workspace._audit.open_label(_DEMO_LABELS[_DEMO_CLUSTER])
    _pump(25)
    _settle()
    shot = window.grab()
    _number_screen(shot, window, workspace._audit, f"{out}/audit.png")

    for name in ("Browse", "Measures"):
        page = _page(window)
        page._show_view(name)
        _pump(15)
        _settle()
        workspace = _page(window).findChild(ClusterWorkspace)
        view = workspace._browse if name == "Browse" else _page(window)
        if name == "Browse":  # a walking bout in the player: a stretched-out mouse
            table = workspace._browse.findChild(QTableView)
            model = table.model()
            walking = next(
                r
                for r in range(model.rowCount())
                if model.index(r, 0).data() == _DEMO_LABELS[_DEMO_CLUSTER]
            )
            table.setCurrentIndex(model.index(walking, 0))
            _pump(15)
        shot = window.grab()
        _number_screen(shot, window, view, f"{out}/{name.lower()}.png")
    _save_callouts()
    window.close()


def _render_clustering_chapter(demo_dir: Path) -> None:
    _render_clustering(demo_dir, part="launch-only")
    _render_clustering_views(demo_dir)


def main() -> None:
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    # The windows below remember their size and splits through QSettings when they close;
    # keep that in a throwaway store, never the reader's own DETAIL settings.
    settings_dir = tempfile.mkdtemp(prefix="gen_docs_settings_")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir
    )
    # A run in flight in the demo would make closing a window ask "Still in progress",
    # a modal no one can answer offscreen.
    MainWindow._active_runs = lambda self: []
    if sys.platform == "win32":
        # The fonts a real Windows session would resolve; offscreen Qt picks neither itself.
        app.setFont(QFont("Segoe UI", 9))
        QFont.insertSubstitution("Monospace", "Consolas")
    if sys.argv[1:] in (["user-guides"], ["more-ways"]):
        render = (
            _render_user_guides if sys.argv[1] == "user-guides" else _render_more_ways
        )
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            locations.experiments_root = (
                lambda: parent
            )  # the home screen lists only this
            render(parent)
        return
    if sys.argv[1:2] == [
        "single-animal"
    ]:  # one chapter: its name, then the demo folder
        chapter, demo_dir, *rest = sys.argv[2:]
        _SINGLE_ANIMAL[chapter](Path(demo_dir), *rest)
        _save_callouts()
        return
    raise SystemExit(__doc__)


#: The user guide's chapters, each rendered from the demo experiment's folder.
_SINGLE_ANIMAL: dict = {
    "clustering": _render_clustering_chapter,
    "clustering-explore": _render_clustering,
    "run-export": _render_run_export,
    "behavior-scan": _render_behavior_scan,
    "location-export": _render_location_export,
    "embedding": _render_embedding,
    "skeleton": _render_new_skeleton,
    "keypoints": _render_keypoints_chapter,
    "detect": _render_detect,
    "set-up": _render_set_up,
    "create": _render_create,
}


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # A video page's decode thread can outlive its window and keep the interpreter from
    # exiting; every image and the marker counts are written by now.
    os._exit(0)
