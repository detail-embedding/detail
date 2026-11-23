"""The colour INTERPRET rings a reviewed cluster in (§19, §22).

AUDIT's table of labels and its cell delegates lived here until review round 11 replaced the table
with a list of clusters to review (``gui/label_check.py``). What is left is the one colour that
says *reviewed* on INTERPRET's tree.

**No colour means good or bad.** Whether a precision is good enough is the reader's call (§19): the
ring says a cluster has been reviewed, never how well it scored.
"""

from detail.gui import plot_theme
from detail.gui.qt import QColor


def audited_colour(theme=None) -> QColor:
    """The one colour a reviewed cluster is ringed in on INTERPRET's tree."""
    theme = theme or plot_theme.current()
    return QColor(theme.text)
