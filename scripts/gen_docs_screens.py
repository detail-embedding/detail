"""Render GUI screens to committed PNGs for the user guides.

``docs/docs_design.md`` §6: GUI images are auto-rendered, not hand-captured. This
generator opens the real GUI offscreen and grabs each documented screen via
``QWidget.grab()``. Rerun a target after a GUI change to refresh its PNGs; the docs
build just embeds them (no Qt at build time).

    python scripts/gen_docs_screens.py user-guides   # install, Configure and Tillicum
    python scripts/gen_docs_screens.py more-ways     # chapter 1's two optional steps,
                                                     # from the committed sample clips
    python scripts/gen_docs_screens.py single-animal <demo experiment folder>
    python scripts/gen_docs_screens.py keypoints <demo experiment folder>
                                                     # chapter 4 alone
                                                     # the chapters' screens
    python scripts/gen_docs_screens.py clustering <demo experiment folder> <run> <node>
                                                     # chapter 8: the run's page, Interpret,
                                                     # a cluster window, the clip window,
                                                     # Audit's review, Browse and Measures

A clustering target draws on a copy of the demo (``_demo_copy``): its clips are rebuilt if
the run predates the node cache, its previews are built into this computer's preview cache
(``DETAIL_PREVIEW_DIR`` moves it), and Audit's review is written to the copy only.

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

from PyQt6.QtCore import Qt, QPoint, QRect, QSize, QThreadPool
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
from detail.api import remote as api_remote
from detail.api import run_spec
from detail.api.experiment import Experiment
from detail.api.readiness import ReadinessSignals
from detail.api.remote import RemoteTargetStatus
from detail.checks import commands
from detail.configs.remote import RemoteTargetConfig
from detail.core import buildinfo
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.gui import app as app_module
from detail.gui import arena
from detail.gui import cuda_doctor, pages, streams
from detail.gui import experiment_locations
from detail.gui.__main__ import configure_application
from detail.gui.app import MainWindow
from detail.gui.camera_view import CameraViewPicker
from detail.gui.preview import PromptTestDialog
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


def _signals(*, gpu: bool, done: bool) -> ReadinessSignals:
    """The checklist on a machine with an NVIDIA GPU.

    ``gpu`` is whether PyTorch can use it: a first Windows install has the CPU-only
    PyTorch, so its GPU row says PyTorch can't use the GPU yet until the GPU / CUDA step is
    done. ``done`` is steps 2 to 4 complete.
    """
    return ReadinessSignals(
        gpu=gpu,
        nvidia_gpu=True,
        hf_token=done,
        sam3=done,
        sam3_backend=done,
        sam3_weights=done,
        detection_model=done,
        has_experiment=False,
        distilled_detector=False,
        keypoint_student=False,
        keypoints_skipped=False,
        shap=True,
    )


def _fresh_machine() -> None:
    """Stand in for a just-installed Windows box, never the machine running this script.

    The Configuration screens read live state: this computer's GPU and PyTorch build, its
    token, its model cache, its registered clusters, and the folder DETAIL runs from. Every
    one of those would otherwise land in a committed image.
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
        CheckResult("cuda", CheckStatus.OK),
        CheckResult("nvidia_gpu", CheckStatus.OK),
        CheckResult("sam3_backend", CheckStatus.FAIL, "No module named 'sam3'"),
    ]
    commands.detail_repo_root = lambda: _DISPLAY_REPO
    commands.active_env_dir = lambda: _DISPLAY_REPO / ".venv"
    api_checks.detail_repo_root = commands.detail_repo_root
    pages.credentials.get_hf_token = lambda: None
    api_models.is_installed = lambda *args, **kwargs: False
    api_models.imported_models = lambda **kwargs: []
    api_remote.list_remote_targets = lambda: []
    # The preview cache line prints where this computer keeps its previews: the default
    # place in a stand-in home, which holds none yet.
    os.environ[locations.PREVIEW_DIR_ENV] = str(
        _DISPLAY_REPO.parent / ".cache" / "detail" / "previews"
    )


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
    """The screens the User guides (docs/user-guides/) show, from an empty workspace."""
    home = MainWindow()
    home.resize(*_WIN_SIZE)
    home.show()
    QApplication.processEvents()
    _write(home.grab(), "install/first-launch.png", _GUIDES_OUT)
    home.close()

    _fresh_machine()
    window = MainWindow()
    window._build_label.hide()  # this checkout's commit, which no reader is running
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

    _render_tillicum(window)
    window.close()


def _render_tillicum(window: MainWindow) -> None:
    """The Set up Tillicum screens: the target as the guide fills it in, and a passing test."""
    target = RemoteTargetConfig(
        name="tillicum",
        host="tillicum",
        remote_workdir="/gpfs/projects/yourgroup/yournetid/detail_runs",
        partition="gpu-h200",
        conda_env="detail-env",
        modules=["conda/Miniforge3-25.3.1-3"],
    )
    dialog = pages._RemoteTargetDialog(parent=window)
    dialog._name.setText(target.name)
    dialog._host.setText(target.host)
    dialog._workdir.setText(target.remote_workdir)
    dialog._partition.setText(target.partition)
    dialog._conda_env.setText(target.conda_env)
    dialog._modules.setText(", ".join(target.modules))
    dialog.resize(620, dialog.sizeHint().height())
    dialog.show()
    QApplication.processEvents()
    shot = dialog.grab()
    _box_widget(dialog, shot, _find_button(dialog, "Add"))
    _write(shot, "tillicum/add-target.png", _GUIDES_OUT)
    dialog.close()

    api_remote.list_remote_targets = lambda: [target]
    window.resize(*_WIN_SIZE)
    window._go_to_node("remote_targets")
    _settle()
    page = window.findChild(pages.RemoteTargetsPage)
    page._table.selectRow(0)
    # The status a passing Test connection writes, through the page's own handler.
    page._on_test_done(
        RemoteTargetStatus(
            ok=True,
            detail="connected to 'tillicum'; sbatch found",
            build=buildinfo.BuildComparison(
                state=buildinfo.SAME,
                target="tillicum",
                host="tillicum",
                local_commit="53c87e9e",
                remote_commit="53c87e9e",
            ),
        )
    )
    QApplication.processEvents()
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "Test connection"))
    _write(shot, "tillicum/remote-targets.png", _GUIDES_OUT)


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


def _render_create_screens(parent: Path, out: str) -> MainWindow:
    """Chapter 1 step 1: the empty Experiments screen, New experiment, Add location and
    Experiment locations.

    ``parent`` is the throwaway folder the experiments root already points at; storage.yaml
    is pointed there too, so the only location is the default one and the location this adds
    for the last picture never reaches the real machine config. Returns the window, still open
    on the Experiments screen.
    """
    locations.CONFIG_DIR = parent / "config"
    locations.STORAGE_PATH = locations.CONFIG_DIR / "storage.yaml"
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
        "Experiment locations",
    ]
    marks = [_find_button(home, b) for b in banner] + [_page(home)._show_archived]
    _number(shot, home, marks, covers=home)
    _write(shot, f"{out}/experiments-empty.png", _GUIDES_OUT)

    dialog = pages.NewExperimentDialog()
    dialog._name.setText("Top-down demo")
    dialog._desc.setText("Fentanyl self-administration, camera above the box")
    dialog._camera_view.set_value("top_down")
    # The throwaway root is a temp path; show the folder a reader actually has.
    dialog._location._combo.setItemText(
        0,
        experiment_locations.location_label(
            locations.ExperimentLocation("default", _DISPLAY_WORKSPACE, True)
        ),
    )
    dialog.resize(620, dialog.sizeHint().height())
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
            dialog._camera_view._combo,
            dialog._location._combo,
            _find_button(dialog, "Add location"),
            _find_button(dialog, "Create"),
            _find_button(dialog, "Cancel"),
        ],
        covers=dialog,
    )
    _write(shot, f"{out}/new-experiment.png", _GUIDES_OUT)
    dialog.close()

    adding = experiment_locations.AddLocationDialog()
    adding._folder.setText(
        r"E:\DetailExperiments"
        if sys.platform == "win32"
        else "/media/yourname/LabDrive/DetailExperiments"
    )
    adding._name.setText("Lab drive")
    adding.resize(620, adding.sizeHint().height())
    adding.show()
    QApplication.processEvents()
    shot = adding.grab()
    _box_widget(adding, shot, _find_button(adding, "Add"))
    _number(
        shot,
        adding,
        [
            adding._folder,
            _find_button(adding, "Choose"),
            adding._name,
            _find_button(adding, "Add"),
            _find_button(adding, "Cancel"),
        ],
        covers=adding,
    )
    _write(shot, f"{out}/add-location.png", _GUIDES_OUT)
    adding.close()

    drive = parent / "drive"
    drive.mkdir()
    locations.add_location("Lab drive", drive)
    listing = experiment_locations.ExperimentLocationsDialog()
    _mask_paths(
        listing,
        {
            drive.resolve().as_posix(): "E:/DetailExperiments",
            parent.as_posix(): _DISPLAY_WORKSPACE.as_posix(),
        },
    )
    listing.resize(720, listing.sizeHint().height())
    listing.show()
    QApplication.processEvents()
    shot = listing.grab()
    _number(
        shot,
        listing,
        [
            _find_button(listing, "Add location"),
            _find_button(listing, "Change folder"),
            _find_button(listing, "Remove"),
            _find_button(listing, "Close"),
        ],
        covers=listing,
    )
    _write(shot, f"{out}/experiment-locations.png", _GUIDES_OUT)
    listing.close()
    locations.remove_location("Lab drive")
    return home


def _render_single_animal(parent: Path, demo_dir: Path) -> None:
    """Chapter 1 of the Single animal guide: create an experiment and add recordings.

    The steps before any video exists are drawn from a fresh experiment in ``parent``; the
    ones after are drawn from the guide's demo experiment, ``demo_dir``, which was built by
    following the same steps (its videos are too large to commit, so it lives outside the
    repository).
    """
    out = "single-animal/create"
    home = _render_create_screens(parent, out)

    fresh = Experiment.create("Top-down demo", parent_dir=parent)
    fresh.set_metadata(description="Fentanyl self-administration, camera above the box")
    fresh.set_camera_view("top_down")
    home._open_experiment(fresh)
    _settle()
    _mask_paths(home, {str(parent): str(_DISPLAY_WORKSPACE)})
    shot = home.grab()
    _box_rail(home, shot, "recordings")
    overview = _page(home)
    name, description = [e for e in overview.findChildren(QLineEdit) if e.isVisible()][
        :2
    ]
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
            recordings.findChild(CameraViewPicker)._combo,
            _find_button(recordings, "Class names"),
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
        add._table.item(n - 1, 2).setText(f"box {n}")
    add.resize(760, 360)
    add._table.setColumnWidth(1, 300)
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

    locations.experiments_root = lambda: demo_dir.parent
    window = MainWindow()
    window._build_label.hide()
    window.resize(*_WIN_SIZE)
    window.show()
    window._open_experiment(Experiment(demo_dir))
    _settle()
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


# The demo boxes' floors, in source pixels of the 800x600 clips: the field of view drawn
# around each, and the two points 200 mm apart (left wall to right wall) that calibrate it.
_DEMO_FOV = [(105, 55), (665, 55), (665, 598), (105, 598)]
_DEMO_CALIBRATION = [(120, 340), (650, 340)]
_DEMO_ROI = [(560, 60), (660, 60), (660, 590), (560, 590)]


# A frame of box1 with the mouse whole on the floor, and a square crop around it (left, top,
# right, bottom, source pixels): the picture chapter 4 draws a skeleton over.
_BACKDROP_FRAME = 1900
_BACKDROP_CROP = (470, 60, 770, 360)


def _normalized(points: list[tuple[int, int]]) -> list[tuple[float, float]]:
    return [(x / 800, y / 600) for x, y in points]


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
        fresh.set_camera_view("top_down")
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
        mapping.resize(640, 520)
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
    """Chapter 2 of the Single animal guide: a recording's page and its scene geometry.

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
    fresh.set_camera_view("top_down")
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
            editor._camera_view._combo,
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
    copy.show()
    _settle()
    shot = copy.grab()

    def position(widget):
        p = widget.mapTo(copy, QPoint(0, 0))
        return (p.y() // 12, p.x())  # one row per line of controls, left to right

    _number(shot, copy, sorted(_controls(copy), key=position), covers=copy)
    _write(shot, f"{out}/copy-dialog.png", _GUIDES_OUT)
    copy.reject()


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


def _number_screen(shot, base, root, rel_path, *, skip=(), extra=(), also=()):
    """Number every control under ``root`` in reading order (the **?** buttons last, as one),
    check the legend will cover the screen, write the image, and print what each number is.

    ``skip`` leaves out controls described elsewhere (the window's own banner); ``extra``
    adds marks ``root`` does not hold, such as the window's Export… button; ``also`` numbers
    a region that is not a control (a tree, a grid of clips) in its place in reading order,
    as one mark with whatever controls it holds, such as its scrollbar.
    """

    def position(widget):
        p = widget.mapTo(base, QPoint(0, 0))
        return (p.y() // 12, p.x())

    marks = [
        w
        for w in _in_reading_order(root, base)
        if not any(w is s or s.isAncestorOf(w) for s in skip)
        and not any(a.isAncestorOf(w) for a in also)
    ]
    marks = [*extra, *sorted([*marks, *also], key=position)]
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


def _stand_in_tillicum() -> None:
    """A registered Tillicum target, so each compute row offers **Remote** as it does on a
    computer set up for it (this one renders without one)."""
    target = RemoteTargetConfig(
        name="tillicum",
        host="tillicum",
        remote_workdir="/gpfs/projects/yourgroup/yournetid/detail_runs",
        partition="gpu-h200",
        conda_env="detail-env",
        modules=["conda/Miniforge3-25.3.1-3"],
    )
    api_remote.list_remote_targets = lambda: [target]
    from detail.gui import compute_target as compute_module

    if hasattr(compute_module, "api_remote"):
        compute_module.api_remote.list_remote_targets = lambda: [target]


def _render_detect_and_track(parent: Path, demo_dir: Path) -> None:
    """Chapter 3 of the Single animal guide: distill a detector, detect, track.

    The before screens (an empty Models page, the Distill dialog, its prompt test) come from
    a fresh experiment holding the demo's three videos, referenced in place; the run pages
    and launch dialogs come from the demo experiment, whose runs were made by these steps.
    """
    from detail.gui.fit import DistillDialog
    from detail.gui.launch import LaunchRunDialog

    out = "single-animal/detect"
    clips = demo_dir.parent.parent / "top-down"
    Experiment.imported_models = property(
        lambda self: []
    )  # this machine's own: not shown
    _stand_in_tillicum()

    fresh = Experiment.create("Top-down demo", parent_dir=parent)
    fresh.set_camera_view("top_down")
    demo = Experiment(demo_dir)
    for column in demo.metadata_columns:
        if column not in fresh.metadata_columns:
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
    dialog.show()
    _settle()
    shot = dialog.grab()
    _box_widget(dialog, shot, _find_button(dialog, "Distill"))
    _number_screen(shot, dialog, dialog, f"{out}/distill-dialog.png")

    toggle = _open_advanced(dialog)
    dialog.resize(dialog.width(), min(dialog.sizeHint().height(), 1400))
    _settle()
    shot = dialog.grab()
    top = toggle.mapTo(dialog, QPoint(0, 0)).y()
    bottom = dialog._class_rows[0][0].mapTo(dialog, QPoint(0, 0)).y()
    advanced = _below(_in_reading_order(dialog, dialog), dialog, top, bottom)
    helps = _below(_help_buttons(dialog), dialog, top, bottom)
    _number(shot, dialog, [*advanced, helps] if helps else advanced)
    _write(shot, f"{out}/distill-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/distill-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")

    # The same section set for Tillicum: both stages remote, a batch for an H200.
    for row in (dialog._compute, dialog._train_compute):
        row.backend.setCurrentText("remote")
    QApplication.processEvents()
    for name, widget in dialog._hparams.items():
        if name == "batch_size":
            widget.setValue(64)
    dialog._stage_videos.setChecked(True)
    _settle()
    shot = dialog.grab()
    changed = [
        dialog._compute.backend,
        dialog._compute.remote_target,
        dialog._train_compute.backend,
        dialog._train_compute.remote_target,
        dialog._hparams["batch_size"],
        dialog._stage_videos,
    ]
    for widget in changed:
        if widget is not None and widget.isVisible():
            _box_widget(dialog, shot, widget)
    _write(shot, f"{out}/distill-advanced-tillicum.png", _GUIDES_OUT)
    dialog.reject()

    test = PromptTestDialog(
        list(fresh.recordings),
        model_id="sam3",
        prompt_text="mouse",
        compute=None,
        parent=window,
    )
    test.resize(1000, 900)
    test.show()
    _settle()
    test._start()
    _wait_until(lambda: test._results_widget is not None, seconds=600)
    _settle()
    shot = test.grab()
    _number_screen(shot, test, test, f"{out}/prompt-test.png")
    test.reject()
    window.close()

    locations.experiments_root = lambda: demo_dir.parent
    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 900)
    window.show()
    window._open_experiment(demo)
    _settle()
    _go_to_rail(window, "models")
    window._open_distill_run("distill_002")
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
    form = launch._config
    form._model.setCurrentIndex(
        next(
            i
            for i in range(form._model.count())
            if "model_003" in form._model.itemText(i)
        )
    )
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, _find_button(launch, "Run"))
    _number_screen(shot, launch, launch, f"{out}/launch-detection.png")
    toggle = _open_advanced(launch)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
    _settle()
    shot = launch.grab()
    top = toggle.mapTo(launch, QPoint(0, 0)).y()
    advanced = _below(_in_reading_order(launch, launch), launch, top)
    advanced = (
        [w for w in advanced if w.text() not in ("Run", "Cancel")]
        if False
        else [
            w
            for w in advanced
            if not (isinstance(w, QPushButton) and w.text() in ("Run", "Cancel"))
        ]
    )
    _number(shot, launch, advanced)
    _write(shot, f"{out}/launch-detection-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/launch-detection-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")
    launch.reject()

    window._open_run("run_001")
    _settle()
    page = _page(window)
    # The Frames with detections column is read off the UI thread; "…" until it lands.
    results = page.findChild(pages.RunResults)
    model = results._table.view.model()
    _wait_until(
        lambda: all(
            model.index(r, pages.RunResults._COVERAGE_COLUMN).data() != "…"
            for r in range(model.rowCount())
        )
    )
    shot = window.grab()
    # The chosen recording's detections under the table are one number: its player is the
    # recording page's, which detection-result.png numbers control by control.
    preview = results._shown
    marks = [w for w in _in_reading_order(page, window) if not preview.isAncestorOf(w)]
    marks.insert(marks.index(results._table.view) + 1, preview)
    _number(shot, window, marks, covers=page)
    _write(shot, f"{out}/detection-run.png", _GUIDES_OUT)

    window._open_recording_run("recording_000", "detection", "det_01")
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
    form._arena.setCurrentIndex(
        next(
            i
            for i in range(form._arena.count())
            if "Field of view" in form._arena.itemText(i)
        )
    )
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

    window._open_run("run_002")
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/tracking-run.png")

    window._open_recording_run("recording_000", "tracking", "track_01")
    _pump(10)
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/tracking-result.png")
    window.close()


def _render_keypoints(demo_dir: Path) -> None:
    """Chapter 4 of the Single animal guide: set up keypoints, then detect and track them.

    Everything is drawn from the demo experiment. Its setup ``distill_003`` ran to the end; a
    second setup, ``distill_004``, was started only to be drawn waiting for its example set.
    """
    from detail.gui.example_set_picker import ExampleSetPicker, NewExampleSetDialog
    from detail.gui.keypoint_labeling import KeypointLabelingDialog
    from detail.gui.keypoint_setup import KeypointSetupDialog
    from detail.gui.keypoint_test import KeypointTestDialog
    from detail.gui.launch import LaunchRunDialog

    out = "single-animal/keypoints"
    Experiment.imported_models = property(
        lambda self: []
    )  # this machine's own: not shown
    _stand_in_tillicum()
    locations.experiments_root = lambda: demo_dir.parent
    demo = Experiment(demo_dir)

    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 900)
    window.show()
    window._open_experiment(demo)
    _settle()
    _go_to_rail(window, "models")
    page = _page(window)
    shot = window.grab()
    _box_widget(window, shot, _find_button(page, "Set up keypoints"))
    _write(shot, f"{out}/models-page.png", _GUIDES_OUT)

    setup = KeypointSetupDialog(demo, parent=window)
    setup.show()
    _pump(3)
    setup._name.setText("mouse keypoints")
    _settle()
    setup.resize(setup.width(), min(setup.sizeHint().height(), 1400))
    _settle()
    shot = setup.grab()
    _box_widget(setup, shot, setup._start)
    _number_screen(shot, setup, setup, f"{out}/setup-dialog.png")
    toggle = _open_advanced(setup)
    setup.resize(setup.width(), min(setup.sizeHint().height(), 1600))
    _settle()
    shot = setup.grab()
    top = toggle.mapTo(setup, QPoint(0, 0)).y()
    advanced = [
        w
        for w in _below(_in_reading_order(setup, setup), setup, top)
        if not (isinstance(w, QPushButton) and w.text() in ("Start", "Cancel"))
    ]
    _number(shot, setup, advanced)
    _write(shot, f"{out}/setup-advanced.png", _GUIDES_OUT)
    print(f"--- legend for {out}/setup-advanced.png")
    for n, w in enumerate(advanced, start=1):
        print(f"  {n}. {_describe(w)}")
    setup.reject()

    # The page a setup shows while it waits for an example set exists only while one does:
    # start a second setup without an example set, let it pause, render, then cancel it.
    waiting = [
        r["id"] for r in demo.distill_runs if r.get("status") == "awaiting_references"
    ]
    if waiting:
        window._open_distill_run(waiting[0])
        _settle()
        page = _page(window)
        page._examples.select("ex_01")
        page._setup_cutoff.setValue(0.75)
        _settle()
        shot = window.grab()
        _number_screen(shot, window, page, f"{out}/setup-waiting.png")
    else:
        print(
            f"skipped {out}/setup-waiting.png: no keypoint setup is waiting for examples"
        )

    picker = ExampleSetPicker(demo)
    new_set = NewExampleSetDialog(
        picker._entries, picker.skeleton_id(), "mouse overhead", parent=window
    )
    new_set.show()
    _settle()
    shot = new_set.grab()
    _number_screen(shot, new_set, new_set, f"{out}/new-example-set.png")
    new_set.reject()

    # The finished setup's teacher tracks: what its example set was marked on.
    done = demo.distill_run_record("distill_003")
    tracks = dict(done["stages"]["track"]["run_ids"])
    labeling = KeypointLabelingDialog(demo, "ex_01", sources=tracks, parent=window)
    labeling.resize(1200, 860)
    labeling.show()
    _pump(8)
    labeling._go_to(5)  # a marked frame, so the skeleton shows
    _pump(6)
    shot = labeling.grab()
    _number_screen(shot, labeling, labeling, f"{out}/labeling.png")
    labeling.reject()

    cfg = done["config"]
    from detail.core.dataclasses import ComputeTarget

    compute = cfg.get("keypoints_compute")
    teacher = {
        "example_set": "ex_01",
        "tracking": tracks,
        "model_id": cfg.get("keypoints_model_id"),
        "model_variant": cfg.get("keypoints_model_variant"),
        "label_compute": ComputeTarget.from_dict(compute) if compute else None,
    }
    test = KeypointTestDialog(demo, teacher, cutoff=0.75, parent=window)
    test.resize(1100, 900)
    test.show()
    _wait_until(lambda: test._result is not None, seconds=900)
    _pump(3)
    shot = test.grab()
    _number_screen(shot, test, test, f"{out}/test-cutoff.png")
    test.reject()

    window._open_distill_run("distill_003")
    _settle()
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/setup-done.png")

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("keypoint_detect"))
    _settle()
    launch._select_all_recordings()
    _pump(3)
    form = launch._config
    form._detection.setCurrentIndex(
        next(
            i
            for i in range(form._detection.count())
            if "run_001" in form._detection.itemText(i)
        )
    )
    form._method.setCurrentIndex(
        next(
            i
            for i in range(form._method.count())
            if "model_004" in form._method.itemText(i)
        )
    )
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

    window._open_recording_run("recording_000", "keypoint_track", "kt_01")
    _pump(12)
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/keypoint-result.png")
    _save_callouts()  # before close: a page with a live player can hang it
    window.close()


def _render_new_skeleton(demo_dir: Path) -> None:
    """Chapter 4's *Make your own skeleton*: the Keypoints manager, a 6-point skeleton drawn
    in the editor, and the picker that lends it a frame of the demo to draw over. Nothing is
    saved: the editor is drawn filled and then closed.

    The library is shown with its presets only, so this machine's own skeletons stay out of
    the picture.
    """
    from detail.api import skeletons as skeleton_library
    from detail.core.skeleton import Skeleton
    from detail.gui.keypoints_manager import SkeletonEditorDialog
    from detail.gui.skeleton_backdrop import FramePickerDialog

    out = "single-animal/keypoints"
    demo = Experiment(demo_dir)
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

    dialog = SkeletonEditorDialog(
        None, [], {e["name"] for e in presets}, experiment=demo, parent=window
    )
    dialog._name.setText("mouse 6-point")
    # Seen from above with the nose up the screen, the animal's left ear is on the left.
    dialog.editor().load(
        Skeleton(
            names=("nose", "left_ear", "right_ear", "neck", "back", "tail_base"),
            edges=(
                ("nose", "left_ear"),
                ("nose", "right_ear"),
                ("nose", "neck"),
                ("neck", "back"),
                ("back", "tail_base"),
            ),
            pairs=(("left_ear", "right_ear"),),
            layout=(
                (0.5, 0.14),
                (0.38, 0.27),
                (0.62, 0.27),
                (0.5, 0.38),
                (0.5, 0.6),
                (0.5, 0.86),
            ),
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

    picker = FramePickerDialog(demo, parent=dialog)
    picker.resize(760, 760)
    picker.show()
    _wait_until(lambda: picker._shown is not None)
    picker.scrubber().set_frame(_BACKDROP_FRAME, decode=True)
    _wait_until(lambda: picker._shown_frame == _BACKDROP_FRAME)
    width, height = picker.canvas().frame_size()
    left, top, right, bottom = _BACKDROP_CROP
    picker.canvas().set_crop(
        (left / width, top / height, right / width, bottom / height)
    )
    _settle()
    shot = picker.grab()
    _number_screen(shot, picker, picker, f"{out}/skeleton-backdrop.png")
    picker.reject()
    # Closed past its own reject, which would ask about the unsaved skeleton in a modal box.
    QDialog.reject(dialog)

    side = next(e for e in presets if e["id"] == "quadruped_side_5")
    dialog = SkeletonEditorDialog(
        side, [], {e["name"] for e in presets}, experiment=demo, parent=window
    )
    dialog.resize(760, 820)
    dialog.show()
    _settle()
    shot = dialog.grab()
    _number_screen(shot, dialog, dialog, f"{out}/side-skeleton.png")
    QDialog.reject(dialog)
    window.close()


def _render_location_export(demo_dir: Path, run_id: str = "an_06") -> None:
    """Chapter 6 of the Single animal guide: Location export over the demo's tracking and its
    keypoint tracking, with each box's *Right wall* region and *Port* point; then the run's
    **Plots** tab and **Montage…**. The launch form is drawn seeded from the demo's
    ``run_id``, as *Start from* would fill it."""
    import yaml

    from detail.gui.launch import LaunchAnalysisDialog

    out = "single-animal/location-export"
    Experiment.imported_models = property(
        lambda self: []
    )  # this machine's own: not shown
    _stand_in_tillicum()
    locations.experiments_root = lambda: demo_dir.parent
    demo = Experiment(demo_dir)
    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 900)
    window.show()
    window._open_experiment(demo)
    _settle()

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
    # The full tracking, as the chapter has the reader tick it: on a tie the picker opens on
    # the newest run, here the keypoint setup's five-minute teacher tracking.
    runs = launch._config._tracking.widget
    _tick_only(runs, "run_002")
    _pump(2)
    keypoints = launch._config._keypoints.combo
    _choose_containing(keypoints, "run_006")
    _pump(2)
    launch.resize(launch.width(), min(launch.sizeHint().height(), 1200))
    _settle()
    shot = launch.grab()
    _box_widget(launch, shot, runs)
    _box_widget(launch, shot, keypoints)
    _number_screen(shot, launch, launch, f"{out}/launch-location-export.png")
    launch.reject()

    _render_run_landing(window, run_id, "Results", f"{out}/run-page.png")
    page, panel = _location_panel(window, run_id)
    _location_tabs(page).setCurrentIndex(1)  # Files
    window.resize(1040, 760)
    _settle()
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/location-export-run.png")

    _location_tabs(page).setCurrentIndex(0)  # Plots
    window.resize(1240, 1000)
    _wait_until(lambda: panel._plots is not None, 300)
    _plot_view(panel, kind="trace", points=["nose", "tail_base"])
    _number_screen(panel.grab(), panel, panel, f"{out}/plots-trace.png")
    _plot_view(panel, kind="heatmap", points=[], source="mask", group_by="drug")
    _plot_figure(panel, f"{out}/plots-heatmap-by-drug.png")
    _plot_view(panel, kind="trace3d", group_by=None)
    _plot_figure(panel, f"{out}/plots-trace3d.png")

    dialog = _location_montage(window, panel, raw=True, keypoints=True, mask_trail=True)
    _number_screen(dialog.grab(), dialog, dialog, f"{out}/montage.png")
    _last_tracing_frame(panel, dialog, f"{out}/last-tracing-frame.png")
    dialog.reject()
    _save_callouts()
    window.close()


def _tick_only(runs, run_id: str) -> None:
    """Tick the run picker's row for ``run_id`` and untick the others."""
    for i in range(runs.count()):
        item = runs.item(i)
        item.setCheckState(
            Qt.CheckState.Checked if run_id in item.text() else Qt.CheckState.Unchecked
        )


def _choose_containing(combo: QComboBox, text: str) -> None:
    index = next(i for i in range(combo.count()) if text in combo.itemText(i))
    combo.setCurrentIndex(index)


def _location_panel(window: MainWindow, run_id: str):
    """A Location export run's page on its **Results** view, and the **Plots** panel on it."""
    from detail.gui.location_plots import LocationPlotsPanel

    page = _open_view(window, run_id, "Results", size=(1240, 1000))
    (panel,) = page.findChildren(LocationPlotsPanel)
    return page, panel


def _location_tabs(page: QWidget):
    from PyQt6.QtWidgets import QTabWidget

    return next(t for t in page.findChildren(QTabWidget) if t.tabText(0) == "Plots")


def _plot_view(panel, **values) -> None:
    """Set the **Plots** controls the way a reader would, then wait for the picture."""
    from detail.configs.location_plots import LocationPlotView
    from detail.gui._pydantic_form import set_any_field_value

    for name, value in values.items():
        widget = panel._widgets[name]
        if name in ("group_by", "source"):
            widget.setCurrentIndex(widget.findData(value))
        elif name in ("points", "subjects", "pairs"):
            for i in range(widget.count()):
                item = widget.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if item.text() in value
                    else Qt.CheckState.Unchecked
                )
        else:
            set_any_field_value(LocationPlotView, name, widget, value)
    _pump(2)
    _wait_until(lambda: panel._picture.pixmap_size() is not None, 300)
    _pump(2)


def _plot_figure(panel, rel_path: str) -> None:
    """The **Plots** tab's current plot as its own picture, the way **Export image…** draws
    it, with no markers and none of the pane around it."""
    from detail.report import plot_render

    spec = panel._plots.build(panel.view(), panel.selected_recordings())
    out = _GUIDES_OUT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(plot_render.render_png(spec, theme="light"))
    print(f"wrote {out.relative_to(_ROOT)}")


def _save_figure(pixmap: QPixmap, rel_path: str) -> None:
    """A picture with no markers: written beside the screens, never into ``callouts.json``
    (the legend check would ask for a legend it has no reason to carry)."""
    out = _GUIDES_OUT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(out))
    print(f"wrote {out.relative_to(_ROOT)}")


def _location_montage(window: MainWindow, panel, **ticks):
    """**Montage…** over the panel's ticked recordings and animals, stopped two thirds of
    the way through the window, with the layers in ``ticks`` set."""
    from detail.gui.location_montage import LocationMontageDialog

    dialog = LocationMontageDialog(
        panel._experiment,
        panel._plots,
        panel.selected_recordings(),
        panel._chosen_animals(),
        parent=window,
    )
    dialog.resize(1000, 900)
    dialog.show()
    _wait_until(lambda: dialog._built is not None, 600)
    for name, value in ticks.items():
        dialog._widgets[name].setChecked(value)
    scrubber = dialog._scrubber
    scrubber.pause()
    scrubber.set_frame(dialog._built.montage.n_steps * 2 // 3)
    _pump(4)
    return dialog


def _last_tracing_frame(panel, dialog, rel_path: str) -> None:
    """The montage's **Last tracing frame…** picture, as the button writes it."""
    panel._plots.export_montage(
        dialog.view(),
        _GUIDES_OUT / rel_path,
        panel.selected_recordings(),
        panel._chosen_animals(),
        last_frame=True,
    )
    print(f"wrote {(_GUIDES_OUT / rel_path).relative_to(_ROOT)}")


def _render_input_images(demo_dir: Path, frame: int = 1900) -> None:
    """Chapter 5's *Input image* choices: one demo frame through each surfaced constructor,
    turned as the demo's embedding run ``emb_01`` turns it (the path *Preview input images*
    takes), beside the original frame. A figure, not a screen: it carries no markers.
    """
    import cv2
    import numpy as np
    import yaml

    from detail.pipeline.embedding import _alignment_settings, largest_turned_mask_px
    from detail.transforms.constructors import CONSTRUCTORS, with_alignment

    demo = Experiment(demo_dir)
    recording = demo.get_recording("recording_000")
    meta = yaml.safe_load(
        (recording.dir / "embeddings" / "emb_01" / "embedding.yaml").read_text(
            encoding="utf-8-sig"
        )
    )
    run_id, tracks = meta["tracking_run_id"], [0]
    alignment = {
        key: meta[key]
        for key in (
            "align",
            "keypoint_run_id",
            "heading_definition",
            "heading_cutoff",
            "max_gap_s",
            "mirror_hold_s",
            "facing_camera_ratio",
        )
    }
    alignment.update(pose="none", pose_min_score=meta["heading_cutoff"])
    canvas = largest_turned_mask_px(
        recording.dir, run_id, tracks, _alignment_settings(alignment)
    )
    side = 260

    def tile(image: np.ndarray, label: str) -> np.ndarray:
        h, w = image.shape[:2]
        scale = side / max(h, w)
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
        out = np.full((side + 34, side, 3), 255, dtype=np.uint8)
        y0, x0 = (side - image.shape[0]) // 2, (side - image.shape[1]) // 2
        out[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]] = image
        cv2.putText(
            out,
            label,
            (6, side + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        return out

    tiles = []
    original = None
    for key, label in (
        ("silhouette_centered", "Silhouette"),
        ("masked_color_centered", "Color"),
        ("masked_color_centered_contrast", "Color on max-contrast bg"),
        ("full_frame", "Full frame"),
    ):
        aligned = (
            CONSTRUCTORS[key].crops_to_subject
            if hasattr(CONSTRUCTORS[key], "crops_to_subject")
            else key != "full_frame"
        )
        spec = recording.resolve_constructor(
            key,
            run_id,
            tracks,
            model_id=meta["model_id"],
            canvas_px=canvas if aligned else None,
        )
        if aligned:
            spec = with_alignment(spec, alignment["align"])
        preview = recording.input_image_preview(
            run_id, spec, alignment=alignment if aligned else None
        )
        _actual, rgb, image = preview.input_at(0, frame, 0)
        if original is None:
            original = tile(rgb, "Original frame")
        tiles.append(tile(image, label))
    # Three to a row, so each tile stays readable on a phone.
    cells = [original, *tiles]
    blank = np.full_like(original, 255)
    while len(cells) % 3:
        cells.append(blank)
    gap = np.full((side + 34, 10, 3), 255, dtype=np.uint8)
    rows = []
    for i in range(0, len(cells), 3):
        rows.append(np.hstack([cells[i], gap, cells[i + 1], gap, cells[i + 2]]))
    figure = np.vstack(rows)
    path = _GUIDES_OUT / "single-animal" / "embedding" / "input-images.png"
    cv2.imwrite(str(path), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
    print(f"wrote {path}")


def _track_form(experiment: Experiment, detect_run: str | None, method: str | None):
    """A keypoint track launch dialog over every recording, on ``detect_run`` (the first
    offered when None) and ``method`` (the form's default when None)."""
    from detail.gui.launch import LaunchRunDialog

    launch = LaunchRunDialog(experiment)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("keypoint_track"))
    _settle()
    launch._select_all_recordings()
    _pump(3)
    form = launch._config
    if detect_run is not None:
        combo = form._picker.combo
        try:
            _wait_until(lambda: combo.count() > 0, seconds=300)
            _pick(combo, detect_run)
        except (TimeoutError, StopIteration):
            found = [combo.itemText(i) for i in range(combo.count())]
            raise SystemExit(f"no detect run {detect_run!r}; the list holds {found}")
        _pump(2)
    if method is not None:
        form._method.setCurrentIndex(form._method.findData(method))
        _pump(2)
    return launch, form


def _render_track_preview(demo_dir: Path) -> None:
    """Chapter 5's *Preview tracking*: the demo's despike_smooth settings and two variants on
    short windows, played side by side. Nothing is written."""
    from detail.gui.keypoint_track_preview import KeypointTrackPreviewDialog

    out = "single-animal/keypoints"
    locations.experiments_root = lambda: demo_dir.parent
    demo = Experiment(demo_dir)
    launch, form = _track_form(demo, "run_005", None)
    preview = KeypointTrackPreviewDialog(
        demo, form.config(), form._preview_recordings(), launch
    )
    preview.resize(1200, 1500)
    preview.show()
    _settle()
    preview.run_preview()
    _wait_until(lambda: preview._result is not None, seconds=900)
    _pump(15)
    _settle()
    # The window keeps itself within a screen and scrolls; its contents are drawn whole.
    inner = preview.findChild(QScrollArea).widget()
    inner.resize(inner.width(), inner.sizeHint().height())
    _pump(5)
    _settle()
    shot = inner.grab()
    _number_screen(shot, inner, inner, f"{out}/track-preview.png")
    QDialog.reject(preview)
    QDialog.reject(launch)


def _render_track_tune(
    exp_dir: Path, track_run: str, detect_run: str, set_name: str
) -> None:
    """Chapter 5's *Tune tracking*: temporal_decode settings scored against a held-out
    example set, on the settings ``track_run`` was launched with, which is what the track
    form hands the window. Read only: the settings are scored and none is taken."""
    import yaml

    from detail.gui.keypoint_track_tune import KeypointTrackTuneDialog

    out = "single-animal/keypoints"
    experiment = Experiment(exp_dir)
    record = yaml.safe_load(
        (exp_dir / "runs" / track_run / "run.yaml").read_text(encoding="utf-8-sig")
    )
    current = record["config"]
    # The detect kind as the picker resolves it now, from the run it names (``run_NNN``):
    # what the form hands the window. The spec frozen in an older record can lag it.
    detect_spec = run_spec.spec_from_experiment_run(
        experiment, "keypoint_detect", detect_run
    )
    tune = KeypointTrackTuneDialog(
        experiment,
        detect_spec,
        current["config"],
        None,
        tracking_spec=current.get("tracking_spec"),
        match={
            "min_inside": current["match_min_inside"],
            "mask_grow": current["match_mask_grow"],
        },
    )
    tune.resize(1100, 900)
    tune.show()
    _wait_until(lambda: tune._sets_loaded, seconds=600)
    _pick(tune._sets, set_name)
    _pump(2)
    tune.score()
    # A failure lands in the summary line, not in a result: stop on it rather than wait.
    _wait_until(
        lambda: tune._result is not None
        or tune._summary.text().startswith(("Could not", "Stopped")),
        seconds=3500,
    )
    if tune._result is None:
        raise SystemExit(tune._summary.text())
    # Wide enough to name each setting, with the recommendation (selected) in view.
    tune.resize(1150, 1000)
    _pump(3)
    picked = tune._table.selectionModel().selectedRows()
    if picked:
        tune._table.scrollTo(picked[0])
    _pump(3)
    _settle()
    shot = tune.grab()
    _number_screen(shot, tune, tune, f"{out}/track-tune.png")
    QDialog.reject(tune)


def _render_side_frame_picker(
    exp_dir: Path, set_id: str, floor: str, bucket: str = "toward"
) -> None:
    """Chapter 4's *Side-view and isometric recordings*: the marking window on **By heading**
    with its compass, and the review window one direction opens.

    Drawn from an angled-view experiment's own example set, which is only read: the review
    window is opened the way a click on the compass opens it, and closed untouched.
    """
    from detail.gui.heading_review import HeadingReviewDialog
    from detail.gui.keypoint_labeling import KeypointLabelingDialog

    out = "single-animal/keypoints"
    experiment = Experiment(exp_dir)
    labeling = KeypointLabelingDialog(experiment, set_id)
    labeling.resize(1200, 900)
    labeling.show()
    _wait_until(lambda: bool(labeling._offered), seconds=300)
    _pump(5)
    strategy = labeling._strategy
    strategy.setCurrentIndex(strategy.findData("heading"))
    try:
        _wait_until(lambda: labeling._floor.count() > 1, seconds=120)
        _pick(labeling._floor, floor)
    except (TimeoutError, StopIteration):
        found = [labeling._floor.itemText(i) for i in range(labeling._floor.count())]
        raise SystemExit(f"no floor class {floor!r}; the list holds {found}")
    labeling._n_recordings.setValue(min(6, labeling._n_recordings.maximum()))
    _pump(2)
    if labeling._sample_tracking.count():  # several tracking kinds: take the first
        labeling._sample_tracking.setCurrentIndex(
            next(
                i
                for i in range(labeling._sample_tracking.count())
                if labeling._sample_tracking.itemData(i) is not None
            )
        )
        _pump(2)
    sources = labeling.sample_sources()
    print(
        f"heading pool from {len(sources)} recordings, {labeling._n_recordings.value()} read"
    )
    if not sources:
        raise SystemExit("the marking window offers no recordings to sample by heading")
    labeling._review_direction(bucket)
    _wait_until(lambda: labeling.findChild(HeadingReviewDialog) is not None, 3600)
    review = labeling.findChild(HeadingReviewDialog)
    review.resize(1100, 900)
    _pump(20)
    _settle()
    shot = review.grab()

    # Every tile carries the same buttons: the first tile of each kind is numbered for all.
    def first(kind, text=None):
        return next(
            w
            for w in review.findChildren(kind)
            if w.isVisible() and (text is None or w.text() == text)
        )

    marks = [
        first(QPushButton, "‹"),
        first(QPushButton, "›"),
        first(QPushButton, "Open"),
        first(QPushButton, "Reject"),
        first(QCheckBox),
        first(QPushButton, "Not this direction"),
        first(QPushButton, "Show more"),
        first(QPushButton, "Close"),
        first(QPushButton, "Add selected"),
    ]
    _number(shot, review, marks)
    _write(shot, f"{out}/side-direction-review.png", _GUIDES_OUT)
    QDialog.reject(review)
    review.setParent(None)
    review.deleteLater()
    _pump(2)

    labeling.resize(1200, 1260)  # the side panel whole, unscrolled
    labeling._go_to(
        0
    )  # an example frame, as a click on it opens: the animal and its marks
    _pump(15)
    _settle()
    shot = labeling.grab()
    _number_screen(shot, labeling, labeling, f"{out}/side-frame-picker.png")
    QDialog.reject(labeling)


def _pick(combo: QComboBox, text: str) -> None:
    """Choose the first item of ``combo`` whose text contains ``text``."""
    combo.setCurrentIndex(
        next(i for i in range(combo.count()) if text in combo.itemText(i))
    )


def _render_embedding(demo_dir: Path) -> None:
    """Chapter 5 of the Single animal guide: embed the tracked, turned animal."""
    from detail.gui.launch import LaunchRunDialog
    from detail.gui.modules._heading import HeadingDefinitionDialog

    out = "single-animal/embedding"
    Experiment.imported_models = property(
        lambda self: []
    )  # this machine's own: not shown
    _stand_in_tillicum()
    locations.experiments_root = lambda: demo_dir.parent
    demo = Experiment(demo_dir)

    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 900)
    window.show()
    window._open_experiment(demo)
    _settle()

    launch = LaunchRunDialog(demo, parent=window)
    launch.show()
    _settle()
    module = launch._module_combo
    module.setCurrentIndex(module.findData("embedding"))
    _settle()
    launch._select_all_recordings()
    _pump(3)
    form = launch._config
    _pick(form._tracking, "run_002")
    _pump(2)
    form._align.setCurrentIndex(form._align.findData("rotate"))
    _pump(2)
    _pick(form._keypoint, "run_006")
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
        (heading._front_list, ("nose", "neck_base")),
        (heading._back_list, ("tail_base",)),
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

    def loaded_cloud(page: QWidget) -> None:
        """Wait for the PCA cloud, then click one point so the frame and input show."""
        from detail.gui.run_preview import EmbeddingRunPreview

        preview = page.findChild(EmbeddingRunPreview)
        _wait_until(lambda: preview._preview is not None, seconds=300)
        _pump(2)
        points = preview._scatter.points()
        preview._on_clicked(None, [points[len(points) // 3]])
        _pump(15)

    window.resize(1040, 1180)  # the cloud under the table: Delete run stays in view
    window._open_run("run_009")
    _settle()
    page = _page(window)
    loaded_cloud(page)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/embedding-run.png")
    _save_callouts()
    window.close()


def _render_behavior_scan(demo_dir: Path) -> None:
    """Chapter 6 of the Single animal guide: a behavior scan for rearing and grooming.

    The demo's ``an_01`` holds the examples marked for the guide and the cut-offs chosen for it
    (0.5 each); the launch form is drawn seeded from it, as *Start from* would fill it.
    """
    import yaml

    from detail.gui.behavior_query import ClipExampleDialog
    from detail.gui.launch import LaunchAnalysisDialog
    from detail.gui.modules.analysis_query import _QueryWorkspace

    out = "single-animal/behavior-scan"
    Experiment.imported_models = property(
        lambda self: []
    )  # this machine's own: not shown
    _stand_in_tillicum()
    locations.experiments_root = lambda: demo_dir.parent
    demo = Experiment(demo_dir)

    window = MainWindow()
    window._build_label.hide()
    window.resize(1040, 900)
    window.show()
    window._open_experiment(demo)
    _settle()
    _go_to_rail(window, "analysis")
    page = _page(window)
    shot = window.grab()
    _number_screen(shot, window, page, f"{out}/analysis-page.png")

    record = yaml.safe_load(
        (demo_dir / "analysis" / "an_01" / "analysis.yaml").read_text(
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

    _render_run_landing(window, "an_01", "Calibrate", f"{out}/run-page.png")

    # Each view full window, as its card opens it, drawn without the window around it: the
    # behavior selector is the view's own top row. Calibrate holds four sections, so the
    # window is tall enough that none of them scrolls.
    for name, height in (
        ("Audit", 520),
        ("Browse", 820),
        ("Measures", 760),
        ("Calibrate", None),
    ):
        page = _open_view(window, "an_01", name, size=(1240, 2000))
        workspace = page.findChild(_QueryWorkspace)
        # The views share one tab widget, as tall as Calibrate: a shorter view is cut to
        # what it holds, Finish and Reopen still at its foot.
        workspace.setMaximumHeight(height or 16_777_215)
        _pump(30)
        _settle()
        if name == "Calibrate":  # a kept clip in the player, as a click on it would
            strip = workspace._calibrate._grid
            shown = sorted(strip._cells)
            pick = shown[len(shown) // 2 + 1]
            strip._emit_clicked(strip._all[pick], pick)
            _pump(10)
        if name == "Browse":  # the first instance playing beside the list
            table = workspace._browse.findChild(QTableView)
            table.setCurrentIndex(table.model().index(0, 0))
            _pump(10)
        shot = workspace.grab()
        _number_screen(shot, workspace, workspace, f"{out}/{name.lower()}.png")
    _save_callouts()
    window.close()


def _demo_copy(demo_dir: Path):
    """A throwaway copy of ``demo_dir`` beside a copy of nothing else: what a target that
    writes (a label, a review, a rebuilt clip cache) draws on, so the demo keeps no trace.

    A context manager yielding the copy's folder. A frame reader in a worker thread can still
    hold a video when the copy is removed, so cleanup errors are ignored.
    """
    import contextlib
    import shutil

    @contextlib.contextmanager
    def copied():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            copy = Path(tmp) / demo_dir.name
            shutil.copytree(demo_dir, copy)
            yield copy

    return copied()


def _clustering_handle(demo: Experiment, run_id: str):
    return demo.temporal_clustering_run(run_id)


def _ready_clips(demo: Experiment, run_id: str) -> None:
    """Make a clustering run's clips playable, as **Rebuild clips** and the background preview
    builder would: a run whose node cache predates this version is rebuilt, and every missing
    preview (each video's proxy and its tracking masks, kept on this computer) is built.
    """
    handle = _clustering_handle(demo, run_id)
    try:
        stale = handle.node_view().stale
    except FileNotFoundError:
        stale = "missing"
    if stale:
        print(f"rebuilding {run_id}'s clips: {stale}")
        handle.rebuild_node_cache()
    if demo.missing_previews():
        print("building previews…")
        report = demo.build_previews()
        print(f"built previews: {report}")


def _open_view(window: MainWindow, run_id: str, view: str | None, size=(1280, 1000)):
    """An analysis run's page, on ``view`` (a card's name) or on its landing page (``None``),
    in a window of ``size``. Returns the page."""
    window.resize(*size)
    if not isinstance(_page(window), pages.AnalysisRunDetailPage) or (
        _page(window)._run_id != run_id
    ):
        window._open_analysis_run(run_id)
        _pump(5)
        _settle()
    page = _page(window)
    if view is None:
        page.close_view()
    else:
        page._show_view(view)
    _pump(10)
    _settle()
    return page


def _render_run_landing(window: MainWindow, run_id: str, card: str, rel: str) -> None:
    """An analysis run's landing page, every control numbered and ``card`` boxed: the step
    that says which card to click."""
    page = _open_view(window, run_id, None, size=(1040, 700))
    shot = window.grab()
    _box_widget(window, shot, page._cards[card])
    _number_screen(shot, window, page, rel)


def _view_tabs_mark(page) -> list:
    """The row of view tabs over a full-window view, as one mark: it is not a control the
    numbering finds by itself."""
    return [page._view_tabs] if page._view_tabs is not None else []


def _menu_under(shot: QPixmap, base: QWidget, button: QWidget, menu) -> QPixmap:
    """``shot`` with ``menu`` drawn open under ``button``, where clicking the button opens it:
    an offscreen menu is a window of its own, so it is grabbed and laid on the picture.
    """
    menu.aboutToShow.emit()  # what fills or greys its entries as it opens
    menu.popup(QPoint(0, 0))
    _pump(1)
    picture = menu.grab()
    menu.close()
    composed = QPixmap(shot)
    painter = QPainter(composed)
    where = button.mapTo(base, QPoint(0, button.height()))
    x = max(0, min(where.x(), composed.width() - picture.width()))
    painter.drawPixmap(x, where.y(), picture)
    painter.end()
    return composed


def _render_cluster_audit(demo_dir: Path, run_id: str) -> None:
    """Chapter 8's **Audit** view on ``run_id``, on a copy of the demo: the first label opened
    on its first cluster, with two of its clips removed, as a reviewer clicks them. Nothing
    is finished, and the demo keeps no trace of the review."""
    with _demo_copy(demo_dir) as copy:
        locations.experiments_root = lambda: copy.parent
        demo = Experiment(copy)
        _ready_clips(demo, run_id)
        window = MainWindow()
        window._build_label.hide()
        window.show()
        window._open_experiment(demo)
        _settle()
        _draw_audit(window, run_id, "single-animal/clustering")
        _save_callouts()
        window.close()


def _draw_audit(window: MainWindow, run_id: str, out: str) -> None:
    from detail.gui.cluster_workspace import ClusterWorkspace

    page = _open_view(window, run_id, "Audit")
    tab = page.findChild(ClusterWorkspace)._audit
    _wait_until(lambda: tab._loaded, seconds=120)
    first = tab._list.topLevelItem(0)
    tab._on_chosen(first)
    _wait_until(
        lambda: tab._open is not None and len(tab._open.get("order", [])) > 0,
        seconds=300,
    )
    _pump(10)
    pane = tab.pane
    for index in (1, 4):  # two clips judged not the label, as a click marks them
        pane._on_click(index, False)
        _pump(2)
    _wait_until(lambda: not tab.writes_pending(), seconds=120)
    _pump(30)
    _settle()
    shot = window.grab()
    _number_screen(
        shot,
        window,
        page,
        f"{out}/audit.png",
        extra=_view_tabs_mark(page),
        also=[pane.montage.grid],
    )


def _render_run_export(demo_dir: Path) -> None:
    """The top bar's **Export…** dialog on a processing run, as the run's page opens it:
    chapter 3's on the tracking run, chapter 4's on the keypoint track run. Nothing is
    exported: the dialog is drawn, then closed."""
    from detail.gui import export as gui_export
    from detail.gui.export import ExportDialog

    win = sys.platform == "win32"
    saved = (
        r"C:\Users\yourname\Documents\Exports"
        if win
        else "/home/yourname/Documents/Exports"
    )
    window = MainWindow()
    window._build_label.hide()
    window.resize(1240, 900)
    window.show()
    window._open_experiment(Experiment(demo_dir))
    _settle()
    shots = (
        ("run_002", "single-animal/detect/export-tracks.png", {"tracks"}),
        ("run_006", "single-animal/keypoints/export-keypoints.png", {"keypoints_dlc"}),
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


def _render_lookalikes(exp_dir: Path, run_id: str) -> None:
    """Chapter 8's look-alike window, on a clustering run whose Find look-alike clusters job
    found groups, opened as INTERPRET's *Look-alike behaviors…* opens it, on its first group.
    Read only: nothing is labelled."""
    from detail.gui.cluster_workspace import ClusterWorkspace

    out = "single-animal/clustering"
    experiment = Experiment(exp_dir)
    window = MainWindow()
    window._build_label.hide()
    window.resize(1240, 1400)
    window.show()
    window._open_experiment(experiment)
    _settle()
    window._open_analysis_run(run_id)
    _pump(5)
    _settle()
    workspace = _page(window).findChild(ClusterWorkspace)
    _pump(20)
    workspace._interpret._open_lookalikes()
    looks = workspace._interpret._lookalikes_window
    _wait_until(
        lambda: looks._groups or "Reading" not in looks._status.text(), seconds=300
    )
    looks.resize(1150, 1080)  # the label row under the reasons, in view
    if looks._tree.topLevelItemCount():
        first = looks._tree.topLevelItem(0)
        looks._tree.setCurrentItem(first)
        first.setExpanded(True)
    _pump(5)
    _settle()
    _number_screen(looks.grab(), looks, looks, f"{out}/lookalikes.png")
    QDialog.reject(looks)
    window.close()


def _render_clustering(
    demo_dir: Path, run_id: str, node: int, launch_only: bool = False
) -> None:
    """Chapter 8 of the Single animal guide: a clustering of the demo, ``run_id`` (behavior
    clustering, ``an_04``), on a copy of the demo.

    ``node`` is the cluster drawn open on **Interpret**, in its own window, in the clip window
    and profiled; ``launch_only`` draws the launch form alone. The copy's clips are made
    playable first (`_ready_clips`), and its **Audit** gets a review (`_draw_audit`), so the
    demo itself is never written.
    """
    from detail.gui.launch import LaunchAnalysisDialog

    out = "single-animal/clustering"
    Experiment.imported_models = property(
        lambda self: []
    )  # this machine's own: not shown
    _stand_in_tillicum()
    with _demo_copy(demo_dir) as copy:
        locations.experiments_root = lambda: copy.parent
        demo = Experiment(copy)
        record = demo.analysis_record(run_id)

        window = MainWindow()
        window._build_label.hide()
        window.resize(1040, 900)
        window.show()
        window._open_experiment(demo)
        _settle()

        launch = LaunchAnalysisDialog(demo, parent=window, seed=record)
        launch.show()
        _settle()
        launch._select_all_recordings()
        _pump(3)
        launch.resize(launch.width(), min(launch.sizeHint().height(), 1400))
        _settle()
        shot = launch.grab()
        _number_screen(shot, launch, launch, f"{out}/launch-{record['module_id']}.png")
        launch.reject()
        if launch_only:
            window.close()
            return

        _ready_clips(demo, run_id)
        _render_run_landing(window, run_id, "Interpret", f"{out}/run-page.png")
        _draw_interpret(window, demo, run_id, node, out)
        _draw_audit(window, run_id, out)
        _draw_browse_measures(window, run_id, out)
        _save_callouts()
        window.close()


def _draw_interpret(
    window: MainWindow, demo: Experiment, run_id: str, node: int, out: str
) -> None:
    """**Interpret** with ``node`` open and its montage playing; then **Cluster tools ▾** and
    **Export ▾** drawn open, the tree's right-click menu, the cluster in its own window, a
    clip in the clip window, and the windows and jobs **Cluster tools ▾** opens."""
    from detail.gui.cluster_profile import ClusterProfileDialog
    from detail.gui.cluster_workspace import ClusterWorkspace
    from detail.gui.launch import LaunchAnalysisDialog
    from detail.gui.live_montage import LiveMontage
    from detail.gui.montage_export import ExportMenu
    from detail.gui.node_icicle import NodeIcicle

    page = _open_view(window, run_id, "Interpret", size=(1280, 1000))
    interpret = page.findChild(ClusterWorkspace)._interpret
    _wait_until(lambda: interpret._view is not None, seconds=300)
    interpret.select(node)
    _wait_until(lambda: interpret._clips is not None, seconds=300)
    _pump(40)
    _settle()
    shot = window.grab()
    _number_screen(
        shot,
        window,
        page,
        f"{out}/interpret.png",
        extra=_view_tabs_mark(page),
        also=_interpret_regions(interpret),
    )

    # **Cluster tools ▾** and **Export ▾** open: a menu has no controls to number.
    plain = window.grab()
    tools = interpret._tools
    _save_figure(
        _menu_under(plain, window, tools, interpret.cluster_tools_menu()),
        f"{out}/cluster-tools.png",
    )
    # The toolbar's, not the pick bar's (hidden until picking, and with no pick entry).
    export = next(m for m in interpret.findChildren(ExportMenu) if m.isVisible())
    _save_figure(
        _menu_under(plain, window, export, export.menu), f"{out}/export-menu.png"
    )

    tree = interpret.findChild(NodeIcicle)
    menu = tree.context_menu(node)
    menu.popup(QPoint(0, 0))
    _pump(1)
    _save_figure(menu.grab(), f"{out}/right-click-menu.png")
    menu.close()

    # The cluster in a window of its own, beside the page.
    cluster_window = interpret.open_window(node)
    cluster_window.resize(1000, 760)
    _pump(40)
    _settle()
    _number_screen(
        cluster_window.grab(),
        cluster_window,
        cluster_window,
        f"{out}/cluster-window.png",
        also=[cluster_window.findChild(LiveMontage).grid],
    )
    cluster_window.close()

    # A tile double-clicked: the clip window, beside the page.
    interpret._play_clip(0)
    _wait_until(lambda: getattr(interpret, "_clip_window", None) is not None, 120)
    popup = interpret._clip_window
    popup.resize(1200, 900)
    _pump(30)
    _settle()
    _number_screen(popup.grab(), popup, popup, f"{out}/clip-preview.png")
    popup.close()

    handle = _clustering_handle(demo, run_id)
    profile = ClusterProfileDialog(demo, handle, node, title=run_id, parent=window)
    profile.resize(1050, 1900)  # the context under the chart, unscrolled
    profile.show()
    _settle()
    if getattr(profile, "_profile", None) is None:
        profile._recompute()
        _wait_until(lambda: getattr(profile, "_profile", None) is not None, 600)
    _pump(5)
    _settle()
    # The dialog keeps itself within a screen and scrolls; its contents are drawn whole.
    inner = profile.findChild(QScrollArea).widget()
    inner.resize(inner.width(), inner.sizeHint().height())
    _settle()
    _number_screen(inner.grab(), inner, inner, f"{out}/cluster-profile.png")
    profile.reject()

    # Profile clusters, as **Cluster tools ▾** opens it, set up to run on Tillicum: drawn,
    # never launched.
    batch = LaunchAnalysisDialog(
        demo,
        window,
        seed={"module_id": "profile_describe", "params": {}},
        selection={"source_run_id": run_id},
    )
    batch.show()
    _pump(5)
    _settle()
    form = batch._config
    form._where.setCurrentIndex(form._where.findData("tillicum"))
    _pump(2)
    batch.resize(batch.width(), min(batch.sizeHint().height(), 1500))
    _settle()
    _number_screen(batch.grab(), batch, batch, f"{out}/profile-describe.png")
    QDialog.reject(batch)

    # The Find look-alike clusters job's launch form, as **Cluster tools ▾** opens it. The
    # look-alike window is drawn from a run that has groups (``lookalikes`` target): the
    # demo's clusters all tell apart.
    find = LaunchAnalysisDialog(
        demo,
        window,
        seed={"module_id": "cluster_lookalikes", "params": {}},
        selection={"source_run_id": run_id},
    )
    find.show()
    _pump(5)
    _settle()
    find.resize(find.width(), min(find.sizeHint().height(), 1300))
    _settle()
    _number_screen(find.grab(), find, find, f"{out}/find-lookalikes.png")
    QDialog.reject(find)


def _interpret_regions(interpret) -> list:
    """**Interpret**'s three regions that are not controls, each one mark: the filters, the
    tree, and the montage of the open cluster's clips."""
    return [interpret._dock, interpret._tree, interpret.montage().grid]


def _draw_browse_measures(window: MainWindow, run_id: str, out: str) -> None:
    """**Browse**, with the first bout playing, and **Measures**, each full window."""
    for name in ("Browse", "Measures"):
        page = _open_view(window, run_id, name, size=(1280, 900))
        if name == "Browse":
            table = next(
                t
                for t in page.findChildren(QTableView)
                if t.isVisible() and t.model() is not None and t.model().rowCount()
            )
            table.setCurrentIndex(table.model().index(0, 0))
        _pump(15)
        _settle()
        shot = window.grab()
        _number_screen(
            shot,
            window,
            page,
            f"{out}/{name.lower()}.png",
            extra=_view_tabs_mark(page),
        )


def main() -> None:
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    if sys.platform == "win32":
        # The fonts a real Windows session would resolve; offscreen Qt picks neither itself.
        app.setFont(QFont("Segoe UI", 9))
        QFont.insertSubstitution("Monospace", "Consolas")
    if sys.argv[1:2] == ["keypoints"]:  # chapter 4 alone, from the demo folder
        _render_keypoints(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:2] == ["streams"]:  # chapter 2's Streams section, the demo folder
        _render_streams(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:2] == ["location-export"]:  # the demo folder [, run id]
        _render_location_export(Path(sys.argv[2]), *sys.argv[3:4])
        _save_callouts()
        return
    if sys.argv[1:2] == ["input-images"]:  # the demo folder
        _render_input_images(Path(sys.argv[2]))
        return
    if sys.argv[1:2] == ["track-preview"]:  # the demo folder
        _render_track_preview(Path(sys.argv[2]))
        _save_callouts()
        return
    # experiment, track run, the detect run it read, held-out set name
    if sys.argv[1:2] == ["track-tune"]:
        _render_track_tune(Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
        _save_callouts()
        return
    if sys.argv[1:] == ["create"]:  # chapter 1 step 1 alone
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            locations.experiments_root = lambda: parent
            _render_create_screens(parent, "single-animal/create").close()
        _save_callouts()
        return
    if sys.argv[1:2] == ["run-export"]:  # the demo folder
        _render_run_export(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:2] == ["cluster-audit"]:  # demo folder, clustering run
        _render_cluster_audit(Path(sys.argv[2]), sys.argv[3])
        _save_callouts()
        return
    if sys.argv[1:2] == ["lookalikes"]:  # experiment folder, clustering run
        _render_lookalikes(Path(sys.argv[2]), sys.argv[3])
        _save_callouts()
        return
    # experiment folder, example set, floor class
    if sys.argv[1:2] == ["side-frame-picker"]:
        _render_side_frame_picker(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
        _save_callouts()
        return
    if sys.argv[1:2] == ["new-skeleton"]:  # chapter 4's own-skeleton subsection
        _render_new_skeleton(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:2] == ["embedding"]:  # chapter 5 alone, from the demo folder
        _render_embedding(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:2] == ["behavior-scan"]:  # chapter 6 alone, from the demo folder
        _render_behavior_scan(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:2] == [
        "clustering"
    ]:  # chapter 7: demo folder, then the cluster to open [, launch: the form alone]
        _render_clustering(
            Path(sys.argv[2]),
            sys.argv[3],
            int(sys.argv[4]),
            launch_only=sys.argv[5:6] == ["launch"],
        )
        _save_callouts()
        return
    if sys.argv[1:2] == ["single-animal"]:
        if len(sys.argv) != 3:
            raise SystemExit(
                "usage: gen_docs_screens.py single-animal <demo experiment folder>"
            )
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            locations.experiments_root = lambda: parent
            _render_single_animal(parent, Path(sys.argv[2]))
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            locations.experiments_root = lambda: parent
            _render_set_up_recordings(parent, Path(sys.argv[2]))
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            locations.experiments_root = lambda: parent
            _render_detect_and_track(parent, Path(sys.argv[2]))
        _render_keypoints(Path(sys.argv[2]))
        _save_callouts()
        return
    if sys.argv[1:] == ["user-guides"]:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            locations.experiments_root = lambda: parent
            _render_user_guides(parent)
        return
    if sys.argv[1:] == ["more-ways"]:
        with tempfile.TemporaryDirectory() as tmp:
            _render_more_ways(Path(tmp))
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # A video page's decode thread can outlive its window and keep the interpreter from
    # exiting; every image and the marker counts are written by now.
    os._exit(0)
