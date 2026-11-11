"""Render GUI screens to committed PNGs for the Workflows tutorials.

``docs/docs_design.md`` §6: GUI images are auto-rendered, not hand-captured. This
generator builds a small real experiment from the committed sample data
(``docs/assets/sample-data/``) and grabs each documented screen offscreen via
``QWidget.grab()``. Rerun it after a GUI change to refresh the PNGs; the docs
build just embeds them (no Qt at build time). This is the "first documented
screen" harness §6 defers, landed as committed PNGs (no golden-compare yet).

Two kinds of image per navigating step (§6):

- a **full-window** capture of the screen the step names, with the control the
  user clicks marked by a drawn highlight box (``_save_window``), and
- the **dialog or panel** that opens, cropped tight (``_save``).

    python scripts/gen_docs_screens.py

Offscreen Qt needs no display: ``QT_QPA_PLATFORM=offscreen`` is set before any Qt
import, exactly as ``tests/gui/conftest.py`` does.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before any Qt import

import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from detail.api.experiment import Experiment
from detail.gui import pages, streams
from detail.gui.__main__ import configure_application
from detail.gui.app import MainWindow
from detail.storage import locations

_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE = _ROOT / "docs" / "assets" / "sample-data"
_OUT = _ROOT / "docs" / "assets" / "workflows"
_WIN_SIZE = (1040, 680)
_HIGHLIGHT = QColor("#D55E00")  # vermillion callout — a doc mark, not a UI status color

# The dialogs echo whatever path the user picked. In the harness we set those
# display-only fields to a neutral path instead of the real absolute location of
# the committed sample files, so a screenshot never bakes in a local home dir.
_DISPLAY_DIR = "/data/mouse-study"


def _noop(*args: object, **kwargs: object) -> None:
    return None


def _build_experiment(parent_dir: Path) -> Experiment:
    """A real on-disk experiment loaded from the committed sample clips + stream."""
    experiment = Experiment.create("operant-study", parent_dir=parent_dir)
    box_a = experiment.create_recording("boxA")
    box_a.add_video_part(_SAMPLE / "boxA_am.mp4")
    box_a.add_video_part(_SAMPLE / "boxA_pm.mp4")
    box_a.add_stream(
        "medpc",
        raw_path=_SAMPLE / "PX22.txt",
        importer="medpc",
        aux_paths={"program": _SAMPLE / "Golden_Liq_SelfAdmin.MPC"},
    )
    box_b = experiment.create_recording("boxB")
    box_b.add_video_part(_SAMPLE / "boxB.mp4")
    experiment.import_metadata_csv(_SAMPLE / "metadata.csv")
    return experiment


def _write(pixmap: QPixmap, rel_path: str) -> None:
    out = _OUT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(out))
    print(f"wrote {out.relative_to(_ROOT)}")


def _save(widget: QWidget, rel_path: str, size: tuple[int, int] | None = None) -> None:
    """Grab a standalone dialog or panel, cropped to itself."""
    if size is not None:
        widget.resize(*size)
    widget.show()
    QApplication.processEvents()
    _write(widget.grab(), rel_path)
    widget.close()


def _find_button(root: QWidget, text: str) -> QPushButton:
    def norm(label: str) -> str:
        return label.replace("&", "").rstrip("…").strip().lower()

    for button in root.findChildren(QPushButton):
        if norm(button.text()) == norm(text):
            return button
    raise LookupError(f"no button labelled {text!r}")


def _box(pixmap: QPixmap, rect: QRect) -> None:
    painter = QPainter(pixmap)
    painter.setPen(QPen(_HIGHLIGHT, 3))
    painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), 5, 5)
    painter.end()


def _save_window(window: MainWindow, rel_path: str, button: str) -> None:
    """Grab the whole app window and draw a box around the named button."""
    window.show()
    QApplication.processEvents()
    pixmap = window.grab()
    target = _find_button(window, button)
    _box(pixmap, QRect(target.mapTo(window, QPoint(0, 0)), target.size()))
    _write(pixmap, rel_path)


def _save_tree_window(window: MainWindow, rel_path: str, node: str) -> None:
    """Grab the whole app window and draw a box around a named sidebar tree node."""
    window.show()
    QApplication.processEvents()
    pixmap = window.grab()
    tree = window._tree
    items = tree.findItems(
        node, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive
    )
    if not items:
        raise LookupError(f"no sidebar node labelled {node!r}")
    rect = tree.visualItemRect(items[0])
    top_left = tree.viewport().mapTo(window, rect.topLeft())
    _box(pixmap, QRect(top_left, rect.size()))
    _write(pixmap, rel_path)


def _fill_stream_dialog(dialog: streams.AddStreamDialog) -> None:
    index = dialog._importer.findData("medpc")
    dialog._importer.setCurrentIndex(index)  # rebuilds the aux pickers
    dialog._name.setText("medpc")
    dialog._data._field.setText(f"{_DISPLAY_DIR}/PX22.txt")
    dialog._aux_pickers["program"]._field.setText(
        f"{_DISPLAY_DIR}/Golden_Liq_SelfAdmin.MPC"
    )


def _render_dialogs(experiment: Experiment) -> None:
    """The tight dialog / panel captures."""
    new_experiment = pages.NewExperimentDialog()
    new_experiment._name.setText("operant-study")
    new_experiment._desc.setText("Open-field study, two arenas")
    _save(
        new_experiment, "create-a-experiment/new-experiment-dialog.png", size=(480, 120)
    )

    add_recordings = pages.AddRecordingsDialog(experiment, _noop)
    add_recordings._append_row("boxB", f"{_DISPLAY_DIR}/boxB.mp4")
    _save(add_recordings, "add-a-recording/add-recordings-dialog.png", size=(640, 300))

    add_stream = streams.AddStreamDialog(experiment.get_recording("boxA"))
    _fill_stream_dialog(add_stream)
    _save(add_stream, "add-stream-data/add-stream-dialog.png")

    metadata = pages.MetadataPage(experiment)
    _save(metadata, "add-stream-data/metadata-page.png", size=(560, 260))


def _render_windows(experiment: Experiment) -> None:
    """The full-window captures, each highlighting the button the step clicks."""
    experiment_view = MainWindow()
    experiment_view.resize(*_WIN_SIZE)
    experiment_view._open_experiment(experiment)
    _save_window(
        experiment_view, "add-a-recording/experiment-recordings.png", "Add recordings"
    )
    _save_window(
        experiment_view,
        "import-recordings-from-yaml/experiment-recordings.png",
        "Import bundle",
    )
    _save_tree_window(experiment_view, "add-stream-data/metadata-node.png", "Metadata")
    experiment_view._open_recording("boxA")
    experiment_view.resize(
        _WIN_SIZE[0], 940
    )  # the recording page is tall — show it whole
    QApplication.processEvents()
    _save_window(
        experiment_view,
        "add-a-multi-part-recording/recording-full.png",
        "Add video part",
    )
    _save_window(experiment_view, "add-stream-data/recording-full.png", "Add stream")
    experiment_view.close()


def main() -> None:
    app = QApplication.instance() or QApplication([])
    configure_application(app)
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        locations.experiments_root = lambda: parent  # the home screen lists only this

        home = MainWindow()  # empty experiments home, before any experiment exists
        home.resize(*_WIN_SIZE)
        _save_window(home, "create-a-experiment/experiments-home.png", "New experiment")
        home.close()

        experiment = _build_experiment(parent)
        _render_dialogs(experiment)
        _render_windows(experiment)


if __name__ == "__main__":
    main()
