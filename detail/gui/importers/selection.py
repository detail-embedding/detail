"""Pickers for files that hold several sessions or subjects: MED-PC and BORIS (Section 3).

A MED-PC data file can hold every box run that day, and a BORIS export several observations and
subjects. A recording is one animal in one session, so the Add-stream dialog asks which one,
from what the file holds, before importing. The choices come from the core importer modules
(``medpc.list_sessions``, ``boris.list_selection``), which read the file the way the import
does, so the picker and the refusal it prevents always describe the same things.

A file holding one of each shows the row greyed with what it holds: there is nothing to choose,
and the band contributes no config, so the import is exactly what it was before this existed.

The choice is about this file, so a saved preset leaves it out (``preset_config``).
"""

from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout

from detail.configs.stream_select import BorisSelectionConfig, MedpcSessionConfig
from detail.gui._pydantic_form import field_meta
from detail.gui.importers.registry import ConfigBand, register_importer_ui
from detail.gui.status_color import MUTED, set_label_color
from detail.importers import boris, medpc

_CHOOSE = "— choose one —"


def _fill(combo: QComboBox, items: list[tuple[str, object]], only: str) -> None:
    """Fill ``combo`` with ``(text, data)`` items: a placeholder first when there is a choice,
    or the one item greyed when there is not."""
    combo.blockSignals(True)
    combo.clear()
    if len(items) > 1:
        combo.addItem(_CHOOSE, None)
        for text, data in items:
            combo.addItem(text, data)
        combo.setEnabled(True)
    else:
        combo.addItem(items[0][0] if items else only, None)
        combo.setEnabled(False)
    combo.blockSignals(False)


class MedpcSessionBand(ConfigBand):
    """One row: which session (``Start Date:`` block) of the data file to import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: str | None = None
        self._sessions: list[dict] = []
        self._error: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self._session = QComboBox()
        self._session.setToolTip(field_meta(MedpcSessionConfig, "session")["help"])
        self._session.currentIndexChanged.connect(lambda _i: self.changed.emit())
        form.addRow("Session:", self._session)
        layout.addLayout(form)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        set_label_color(self._hint, MUTED)
        layout.addWidget(self._hint)
        layout.addStretch(1)
        _fill(self._session, [], "(choose a data file above)")

    def set_source(self, path: str) -> None:
        if path == self._source:
            return
        self._source = path
        self._error = None
        try:
            self._sessions = medpc.list_sessions(path)
        except (OSError, ValueError) as exc:
            self._sessions, self._error = [], str(exc)
        _fill(
            self._session,
            [(s["label"], s["session"]) for s in self._sessions],
            "(no session found)",
        )
        self._hint.setText(
            f"This file holds {len(self._sessions)} sessions, one per box or run. A "
            "recording is one of them; they are never merged."
            if len(self._sessions) > 1
            else ""
        )
        self.changed.emit()

    def config(self) -> dict | None:
        chosen = self._session.currentData()
        return None if chosen is None else {"session": int(chosen)}

    def preset_config(self) -> dict | None:
        return None

    def problem(self) -> str | None:
        if self._error:
            return f"Could not read this MED-PC file: {self._error}"
        if len(self._sessions) > 1 and self._session.currentData() is None:
            return (
                f"This file holds {len(self._sessions)} sessions; choose the one this "
                "recording is."
            )
        return None


class BorisSelectionBand(ConfigBand):
    """Two rows: which observation, then which subject of it, to import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: str | None = None
        self._observations: list[str] = []
        self._subjects: list[str] = []
        self._error: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self._observation = QComboBox()
        self._observation.setToolTip(
            field_meta(BorisSelectionConfig, "observation")["help"]
        )
        self._observation.currentIndexChanged.connect(self._on_observation)
        form.addRow("Observation:", self._observation)
        self._subject = QComboBox()
        self._subject.setToolTip(field_meta(BorisSelectionConfig, "subject")["help"])
        self._subject.currentIndexChanged.connect(lambda _i: self.changed.emit())
        form.addRow("Subject:", self._subject)
        layout.addLayout(form)
        self._hint = QLabel()
        self._hint.setWordWrap(True)
        set_label_color(self._hint, MUTED)
        layout.addWidget(self._hint)
        layout.addStretch(1)
        _fill(self._observation, [], "(choose a data file above)")
        _fill(self._subject, [], "(choose a data file above)")

    def set_source(self, path: str) -> None:
        if path == self._source:
            return
        self._source = path
        self._error = None
        try:
            self._observations = boris.list_selection(path)["observations"]
        except (OSError, ValueError) as exc:
            self._observations, self._error = [], str(exc)
        _fill(
            self._observation,
            [(o, o) for o in self._observations],
            "(no Observation id column)",
        )
        self._on_observation()

    def _on_observation(self, *_args) -> None:
        """Re-list the subjects for the observation now chosen, then re-check the form."""
        self._subjects = []
        observation = self._chosen_observation()
        if self._source and not self._error:
            if observation is not None or len(self._observations) <= 1:
                try:
                    self._subjects = boris.list_selection(self._source, observation)[
                        "subjects"
                    ]
                except (OSError, ValueError) as exc:
                    self._error = str(exc)
        _fill(
            self._subject,
            [(s or "(no subject)", s) for s in self._subjects],
            "(choose an observation first)" if self._observations else "(no rows)",
        )
        parts = []
        if len(self._observations) > 1:
            parts.append(f"{len(self._observations)} observations")
        if len(self._subjects) > 1:
            parts.append(f"{len(self._subjects)} subjects")
        self._hint.setText(
            f"This export holds {' and '.join(parts)}. A recording is one animal in one "
            "session; they are never merged."
            if parts
            else ""
        )
        self.changed.emit()

    def _chosen_observation(self) -> str | None:
        if len(self._observations) == 1:
            return self._observations[0]
        return self._observation.currentData()

    def config(self) -> dict | None:
        out = {}
        if len(self._observations) > 1 and self._observation.currentData() is not None:
            out["observation"] = self._observation.currentData()
        if len(self._subjects) > 1 and self._subject.currentData() is not None:
            out["subject"] = self._subject.currentData()
        return out or None

    def preset_config(self) -> dict | None:
        return None

    def problem(self) -> str | None:
        if self._error:
            return f"Could not read this BORIS export: {self._error}"
        if len(self._observations) > 1 and self._observation.currentData() is None:
            return (
                f"This export holds {len(self._observations)} observations; choose the "
                "one this recording is."
            )
        if len(self._subjects) > 1 and self._subject.currentData() is None:
            return (
                f"This export scores {len(self._subjects)} subjects; choose the one this "
                "stream is for."
            )
        return None


@register_importer_ui
class MedpcSessionUI:
    importer_id = "medpc"

    def config_band(self, parent=None) -> ConfigBand:
        return MedpcSessionBand(parent)


@register_importer_ui
class BorisSelectionUI:
    importer_id = "boris"

    def config_band(self, parent=None) -> ConfigBand:
        return BorisSelectionBand(parent)
