"""The GUI's status palette, and the one way to apply it to a label.

A **leaf**: it imports nothing from ``detail.gui`` and nothing from ``detail`` at all, which is
the whole point. These four colours and ``set_label_color`` used to live in ``pages.py``, near
the top of a 5000-line module that imports ``detail.gui.modules`` at line 78 to register the
module UIs. That import reaches ``modules/analysis_query.py`` -> ``gui/confidence.py``, which
imported the colours straight back out of ``pages`` — a cycle, and one that only bites in a
particular order: ``pages`` defines the colours *after* that line, so anything that imported
``pages`` first got a partially-initialized module and

    ImportError: cannot import name '_MUTED' from partially initialized module

``pytest tests/gui/test_pages.py`` on its own was 31 failures, all of them this. The full suite
passed, because something else always imported first and the names existed by the time
``confidence`` asked. A cycle that depends on import order is invisible until the day someone
runs one file.

Colour is reserved for genuine status (``docs/gui_design.md``): green ready, amber warning,
orange-red error, grey transient. There is no stylesheet anywhere in the GUI, so the colour goes
on through the widget's palette.
"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QAbstractButton, QLabel

READY = QColor(40, 140, 40)  # green — ready
WARN = QColor(170, 110, 0)  # amber — warning / setup incomplete
ERROR = QColor(225, 105, 45)  # orange-red — error (dark red reads badly on a dark bg)
MUTED = QColor(120, 120, 120)  # grey — transient / neutral


def set_label_color(label: QLabel, color: QColor) -> None:
    """Set a status label's text color via the palette (no stylesheet — GUI style)."""
    palette = label.palette()
    palette.setColor(QPalette.ColorRole.WindowText, color)
    label.setPalette(palette)


def set_button_color(button: QAbstractButton, color: QColor | None) -> None:
    """Fill a button with a status colour to say a mode is **on**, or ``None`` to put it back.

    For a state the person started and has not finished — a clip start marked and no end yet —
    which a status line elsewhere on the dialog does not say where the next press goes. White text
    on the fill, since every status colour here is mid-dark. Restored from the application's
    palette rather than a saved copy, so a theme switch while it was lit is not undone.
    """
    if color is None:
        button.setPalette(QPalette())
        button.setAutoFillBackground(False)
        return
    palette = button.palette()
    palette.setColor(QPalette.ColorRole.Button, color)
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("white"))
    button.setPalette(palette)
    button.setAutoFillBackground(True)
