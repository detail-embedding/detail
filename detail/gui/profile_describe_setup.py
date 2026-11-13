"""Configuration ▸ Cluster profile — the optional analysis setup (§1, §22).

One thing makes the Cluster profile window more useful, and it is not needed to run anything:

- **SHAP**, which Cluster profile's default explainer uses. Without it the forest's importances
  are permutation importance, and the window says so. **Install SHAP** installs the ``analysis``
  extra's missing packages into the interpreter running this app (``api/extras.py``), streaming
  the installer's output into a bar with Cancel; a failure shows that output. The same install
  is offered as a command to paste, for a machine where the button cannot be used.

The SHAP status is probed off the UI thread, through the same **fresh-subprocess** check run as
the rest of Configuration (so an install made in a terminal shows on *Check again*). The page is
reached from the Get-started checklist and, as :class:`ProfileDescribeSetupDialog`, from the
profile window wherever it says SHAP is missing.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.api import checks as api_checks
from detail.api import extras
from detail.core.dataclasses import CheckStatus
from detail.gui.busy import run_progress
from detail.gui.dialog import Dialog
from detail.gui.lifetime import discard
from detail.gui.pages import (
    _command_row,
    _heading,
    _instructions,
    _make_selectable,
    _status_label,
)
from detail.gui.sizing import mono_font, set_available
from detail.gui.status_color import ERROR, MUTED, READY, WARN, set_label_color
from detail.gui.workers import BackgroundCall

#: What SHAP changes in Cluster profile, said where the install is offered.
SHAP_EXPLAINED = (
    "Cluster profile fits a random forest to a cluster's measures against a comparison set, "
    "and reports which measures it leaned on. With SHAP, each measure's weight is its mean "
    "|SHAP| value, and its direction comes from how the measure and its SHAP value move "
    "together, row by row. Without SHAP, the weights are permutation importance (how much "
    "held-out accuracy drops when the measure is shuffled), and the direction comes from the "
    "two medians. Both are scored on held-out recordings against the same shuffled floor."
)


class ShapSetup(QFrame):
    """SHAP: installed or not (with its version), what it changes, Install SHAP, and the
    command to paste instead. ``on_changed()`` runs after a re-check."""

    def __init__(self, on_changed=None, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Plain)
        self._on_changed = on_changed
        self._installed = False
        column = QVBoxLayout(self)
        column.addWidget(_heading("SHAP, for Cluster profile"))
        self.status = _status_label()
        self.status.setText("Checking SHAP…")
        set_label_color(self.status, MUTED)
        column.addWidget(self.status)
        column.addWidget(_instructions(SHAP_EXPLAINED))

        actions = QHBoxLayout()
        self.install = QPushButton("Install SHAP")
        self.install.clicked.connect(self._install)
        set_available(self.install, False, "Checking whether SHAP is installed…")
        actions.addWidget(self.install)
        self.recheck = QPushButton("Check again")
        self.recheck.setToolTip(
            "Re-run the check in a fresh process, so an install made in a terminal shows"
        )
        self.recheck.clicked.connect(lambda: self.check())
        actions.addWidget(self.recheck)
        actions.addStretch(1)
        column.addLayout(actions)
        self.what = QLabel("")
        self.what.setWordWrap(True)
        set_label_color(self.what, MUTED)
        column.addWidget(self.what)

        self.paste = QLabel("Or paste this into a terminal:")
        column.addWidget(self.paste)
        self._command_holder = QVBoxLayout()
        column.addLayout(self._command_holder)
        self._command: QWidget | None = None

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(mono_font())
        self.log.setMinimumHeight(140)
        self.log.setVisible(False)  # the installer's output, shown when it failed
        column.addWidget(self.log)
        self.check()

    # -- status (a fresh-subprocess probe: off the UI thread) -------------------- #
    def check(self) -> None:
        set_available(self.recheck, False, "Checking…")
        self._check_call = BackgroundCall(_shap_state, receiver=self)  # off-thread
        self._check_call.done.connect(self._show)
        self._check_call.failed.connect(self._failed_check)
        self._check_call.start()

    def _show(self, state) -> None:
        result, extra, command, version = state
        set_available(self.recheck, True, "")
        self._installed = result is not None and result.status == CheckStatus.OK
        if self._installed:
            self.status.setText(
                f"SHAP {version or ''} is installed. Cluster profile explains with SHAP."
            )
            set_label_color(self.status, READY)
            set_available(self.install, False, "SHAP is installed")
        else:
            self.status.setText(
                "SHAP is not installed. Cluster profile uses permutation importance instead."
            )
            set_label_color(self.status, WARN)
            set_available(self.install, True, "")
        if self._command is not None:
            discard(self._command)
            self._command = None
        if extra.missing:
            self.what.setText(
                f"Installs the analysis extra's missing packages "
                f"({', '.join(extra.missing)}) into the environment running this app: "
                f"{sys.executable}"
            )
            self._command = _command_row(command)
            self._command_holder.addWidget(self._command)
        else:
            self.what.setText(
                f"The analysis extra is complete in the environment running this app: "
                f"{sys.executable}"
            )
        self.paste.setVisible(bool(extra.missing))
        _make_selectable(self)
        if self._on_changed is not None:
            self._on_changed()

    def _failed_check(self, exc) -> None:
        set_available(self.recheck, True, "")
        self.status.setText(f"Couldn't check SHAP: {exc}")
        set_label_color(self.status, ERROR)

    # -- install (a subprocess, streamed: off the UI thread) --------------------- #
    def _install(self) -> None:
        self.log.setVisible(False)
        run_progress(
            self,
            lambda emit: extras.install_extra("analysis", emit),  # off-thread
            self._installed_ok,
            label="Installing SHAP…",
            on_error=self._install_failed,
            on_cancel=self._install_cancelled,
            supersedes="install shap",
        )

    def _installed_ok(self, _status) -> None:
        self.status.setText("Installed. Checking again…")
        set_label_color(self.status, READY)
        self.check()

    def _install_failed(self, exc) -> None:
        self.status.setText("The install failed. The installer's output is below.")
        set_label_color(self.status, ERROR)
        self.log.setPlainText(str(exc))
        self.log.setVisible(True)

    def _install_cancelled(self) -> None:
        self.status.setText(
            "Install cancelled. Check again to see what, if anything, was installed."
        )
        set_label_color(self.status, WARN)


def _shap_state():
    """The ``shap`` check's result from a fresh interpreter, the analysis extra's status, the
    command to paste, and SHAP's version. Off the UI thread: a subprocess, and disk reads.
    """
    results = {r.check_id: r for r in api_checks.run_checks()}
    return (
        results.get("shap"),
        extras.status("analysis"),  # off-thread (BackgroundCall)
        extras.command_text("analysis"),  # off-thread
        extras.package_version("shap"),
    )


class ProfileDescribePage(QWidget):
    """Configuration ▸ Cluster profile. ``on_readiness_changed()`` tells the shell a check
    finished, so the checklist's step follows without a restart."""

    def __init__(self, on_readiness_changed=None, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(_heading("Cluster profile"))
        layout.addWidget(
            _instructions(
                "Optional. SHAP makes the Cluster profile window more useful, and it is not "
                "needed to run anything: without it the profile says it used permutation "
                "importance."
            )
        )
        self.shap = ShapSetup(on_changed=on_readiness_changed)
        layout.addWidget(self.shap)
        layout.addStretch(1)


class ProfileDescribeSetupDialog(Dialog):
    """The same page, opened from the profile window where it says SHAP is missing."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cluster profile setup")
        self.page = ProfileDescribePage()
        self.body().addWidget(self.page)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
