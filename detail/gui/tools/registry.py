"""GUI-side tool-UI registry (§27) — the Qt half of a ``Tool``, mirroring
``gui/modules/registry.py``.

Each ``Tool`` contributes a bespoke UI: a config form (its run page) and a result view
(a completed run's output). Registered by ``tool_id`` and paired with the core ``TOOLS``
entry. The shell stays generic — ``ToolsPage`` lists tools, ``ToolRunPage`` hosts the
form and runs it off-thread, ``ToolRunDetailPage`` hosts the result view. Adding a tool's
UI is a file here + one ``register_tool_ui`` call — no shell change. ``result_widget``
lives here, never on the core ``Tool`` protocol — that would pull Qt into ``core/``
(Tenet 3).
"""

import importlib
from abc import ABC, abstractmethod

from detail.gui.modules.registry import ConfigWidget
from detail.gui.qt import QWidget

TOOL_UI_MODULES: tuple[str, ...] = ("detail.gui.tools.quick_segment",)


class ToolUI(ABC):
    tool_id: str  # pairs with the core TOOLS entry

    @abstractmethod
    def config_widget(self) -> ConfigWidget:
        """A fresh config form for the tool's run page."""

    @abstractmethod
    def result_widget(self, manifest: dict) -> QWidget:
        """A view of one completed run's output, for the run-detail page."""


TOOL_GUIS: dict[str, ToolUI] = {}


def register_tool_ui(cls: type) -> type:
    """Register a ToolUI (instantiated once), keyed by its ``tool_id``."""
    instance = cls()
    TOOL_GUIS[instance.tool_id] = instance
    return cls


def import_tool_uis() -> None:
    """Import every tool-UI module so its ``register_tool_ui`` runs and populates
    ``TOOL_GUIS``."""
    for name in TOOL_UI_MODULES:
        importlib.import_module(name)
