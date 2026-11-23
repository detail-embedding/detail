"""GUI-side importer-UI registry (Section 3) — the Qt half of a ``StreamImporter``.

An importer that declares a ``config_model`` needs a way to collect it, and for the CSV one
the *choices* come from the file being imported rather than from the model, so the form
cannot be generated. Each such importer contributes a ``ConfigBand``: the widget the
``AddStreamDialog`` shows under the importer picker, which reads a chosen file and returns
that importer's ``config`` dict.

Registered by ``importer_id`` and paired with the core ``STREAM_IMPORTERS`` entry, mirroring
``gui/tools/registry.py`` and ``gui/modules/registry.py``. The dialog stays generic: it hosts
whatever band the picked importer contributes and shows nothing where an importer contributes
none, so an importer with no parameters (the interval and events CSVs) needs no file here at
all. MED-PC and BORIS contribute a picker (``selection.py``): their files can hold several
sessions, observations or subjects, and which ones is read from the file.

The band lives here, never on the core ``StreamImporter`` protocol — that would pull Qt into
``core/`` (Tenet 3).
"""

import importlib
from abc import ABC, abstractmethod

from detail.gui.qt import QWidget, Signal

IMPORTER_UI_MODULES: tuple[str, ...] = (
    "detail.gui.importers.csv_columns",
    "detail.gui.importers.selection",
)


class ConfigBand(QWidget):
    """One importer's config widget, shown under the picker in the Add-stream dialog.

    ``changed`` tells the dialog to re-check whether the form can be submitted; ``config()``
    returns the importer's ``config`` dict (or ``None`` for "use the importer's defaults").
    ``problem()`` is the one sentence to show and to refuse the import with, or ``None``.
    """

    changed = Signal()

    def set_source(self, path: str) -> None:
        """Called when the user picks a data file. Read it here, not in ``__init__``."""

    def config(self) -> dict | None:
        return None

    def preset_config(self) -> dict | None:
        """The part of :meth:`config` a saved preset keeps — all of it, by default.

        A band whose config is a choice about *this file* (which session, which subject)
        returns ``None``: the same choice on the next recording's file would pick another
        animal's data, which is the opposite of what a preset is for.
        """
        return self.config()

    def set_config(self, config: dict) -> None:
        """Pre-fill from a saved preset, against whatever file is currently loaded.

        A preset names columns; this file may not have all of them. Apply what matches and
        report the rest through :meth:`problem` or a notice — never fail silently, and never
        leave a column selected that is not there.
        """

    def problem(self) -> str | None:
        return None

    def suggested_name(self) -> str:
        """A stream name suggestion from the file, or ``""`` to leave the dialog's own."""
        return ""


class ImporterUI(ABC):
    importer_id: str  # pairs with the core STREAM_IMPORTERS entry

    @abstractmethod
    def config_band(self, parent=None) -> ConfigBand:
        """A fresh config band for the Add-stream dialog."""


IMPORTER_GUIS: dict[str, ImporterUI] = {}


def register_importer_ui(cls: type) -> type:
    """Register an ImporterUI (instantiated once), keyed by its ``importer_id``."""
    instance = cls()
    IMPORTER_GUIS[instance.importer_id] = instance
    return cls


def import_importer_uis() -> None:
    """Import every importer-UI module so its ``register_importer_ui`` runs and populates
    ``IMPORTER_GUIS``."""
    for name in IMPORTER_UI_MODULES:
        importlib.import_module(name)
