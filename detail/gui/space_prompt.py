"""The questions a launch asks before it runs: one for a refused-for-space launch (#743), one
for a launch that costs more than a person would guess (:func:`ask_before_launch`).

``Experiment.run_processing`` / ``retry_run`` refuse a run whose estimated output does not fit
where the experiment is, raising ``InsufficientSpaceError``. Every GUI surface that launches
one — the launch dialog and the run page's Retry — asks the same thing through here, so the
two cannot word it differently or default it differently.

A module whose config can be far more costly than it looks says so through its form's
``ConfigWidget.launch_question`` (Find look-alike clusters with every pair: many hours of
compute); the launch dialog asks it here, with the same default — the choice that keeps going
takes a deliberate click.
"""

from dataclasses import dataclass

from PyQt6.QtWidgets import QMessageBox, QWidget

from detail.core.disk_space import InsufficientSpaceError


@dataclass(frozen=True)
class LaunchQuestion:
    """What a costly launch asks: ``text`` what it costs and the cheaper way, ``go`` the
    button that launches anyway (a verb for the choice, e.g. "Run every pair")."""

    text: str
    go: str


def ask_before_launch(parent: QWidget, title: str, question: LaunchQuestion) -> bool:
    """``True`` when the user chose ``question.go``. Defaults to Cancel: the costly choice
    is the one that takes a deliberate click, as :func:`launch_anyway`'s is."""
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(question.text)
    go = box.addButton(question.go, QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel)
    box.exec()
    return box.clickedButton() is go


def launch_anyway(parent: QWidget, title: str, exc: BaseException) -> bool:
    """``True`` when ``exc`` is a space refusal and the user chose to launch anyway.

    Defaults to No: a drive filled to its last megabyte is what crashed WSL in #743, so the
    choice that keeps going is the one that takes a deliberate click.
    """
    if not isinstance(exc, InsufficientSpaceError):
        return False
    answer = QMessageBox.question(
        parent,
        title,
        f"{exc.estimate.describe()}.\n\nIf the drive fills while the run is writing, the run "
        "fails, and on WSL the whole Linux environment can stop with it.\n\nLaunch anyway?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes
