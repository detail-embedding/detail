"""The result-view protocol: what an analysis run's page lists as cards and opens full window.

Re-exported by `gui/modules/registry.py`, whose contract it is part of (§16, `docs/gui_design.md`
§3.2). It lives here, outside the ``modules`` package, because `gui/cluster_workspace.py` builds
on `MultiViewWorkspace` and is itself imported by module UIs while that package is importing —
a cycle a sibling module does not have.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QTabWidget, QWidget

#: What a result widget with no views of its own is called on the run page: one view.
RESULTS_VIEW = "Results"

#: The purpose a module's single view states when the module declares none.
DEFAULT_PURPOSE = "What this run produced."


@dataclass(frozen=True)
class ResultView:
    """One full-window view of an analysis run's results (§16 run page, `gui_design.md` §3.2).

    The run page opens on a landing page with a card per view; a card opens its view to fill the
    window, and a row of tabs over the view switches straight to another. ``status`` is the
    card's one status line, called when the landing page is built — so it must be cheap: one
    small file read at most, never a read per recording.
    """

    name: str
    purpose: str
    widget: QWidget
    status: Callable[[], str] | None = None

    def status_text(self) -> str:
        """The status line, or ``""`` when there is none or it cannot be read just now."""
        if self.status is None:
            return ""
        try:
            return self.status()
        except (OSError, ValueError, KeyError) as exc:
            return f"(could not read: {exc})"


def result_views(widget: QWidget | None, purpose: str = "") -> list[ResultView]:
    """The views ``widget`` offers: its own list if it declares ``views()``, else one "Results".

    One function asked by the one page class, so every analysis module gets the landing page
    and the full-window views without doing anything — a module with several views says so by
    building a `MultiViewWorkspace`, and every other result widget is one view.
    """
    if widget is None:
        return []
    declared = getattr(widget, "views", None)
    if callable(declared):
        return list(declared())
    return [ResultView(RESULTS_VIEW, purpose or DEFAULT_PURPOSE, widget)]


class MultiViewWorkspace(QWidget):
    """A result widget made of several views over one ``QTabWidget`` whose tab bar is hidden.

    The tabs are still how the workspace holds its views — the hand-offs between them
    (AUDIT's "Open in INTERPRET") still call ``setCurrentWidget`` — but the *switch* is the run
    page's row of view tabs, not a tab bar inside the view: two rows of tabs for one choice
    is what the full-window view exists to remove. ``view_changed`` is how a hand-off inside
    the workspace reaches that row.

    One base for the three workspaces that have views (clustering, behavior scan, annotation),
    so the protocol has one implementation rather than three that drift.
    """

    #: The name of the view now in front, whoever moved it.
    view_changed = pyqtSignal(str)

    def _init_views(
        self,
        tabs: QTabWidget,
        specs: list[tuple[str, str, QWidget, Callable[[], str] | None]],
    ) -> None:
        """Adopt ``tabs`` as the view holder: one ``(name, purpose, widget, status)`` per view,
        each already added to ``tabs``, in order."""
        self._view_tabs = tabs
        self._view_specs = list(specs)
        tabs.tabBar().setVisible(False)
        # No tab bar, so no frame drawn for one either: the view meets the page edge to edge.
        tabs.setDocumentMode(True)
        tabs.currentChanged.connect(self._emit_view_changed)

    def _emit_view_changed(self, _index: int) -> None:
        self.view_changed.emit(self.current_view())

    def views(self) -> list[ResultView]:
        return [
            ResultView(name, purpose, widget, status)
            for name, purpose, widget, status in self._view_specs
        ]

    def show_view(self, name: str) -> None:
        """Bring the view called ``name`` to the front.

        Raises:
            KeyError: If no view has that name.
        """
        for spec_name, _purpose, widget, _status in self._view_specs:
            if spec_name == name:
                self._view_tabs.setCurrentWidget(widget)
                return
        raise KeyError(
            f"no view called {name!r}; this workspace has {self.view_names()}"
        )

    def current_view(self) -> str:
        current = self._view_tabs.currentWidget()
        for name, _purpose, widget, _status in self._view_specs:
            if widget is current:
                return name
        return ""

    def view_names(self) -> list[str]:
        return [spec[0] for spec in self._view_specs]
