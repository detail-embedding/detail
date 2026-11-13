"""The question a launch refused for space asks (#743), and that every GUI launcher asks it.

``space_prompt.launch_anyway`` is the one wording and the one default (No) for the launch
dialog and the run page's Retry. The prompt itself is driven through a patched
``QMessageBox.question`` (a modal box cannot be clicked in a test); the wiring is a source
scan, data-driven over every GUI call that launches or retries a processing run, so a third
launcher added later is held to it.
"""

import re
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QMessageBox, QWidget

from detail.core.disk_space import InsufficientSpaceError, SpaceEstimate
from detail.gui import space_prompt

GB = 1024**3
_REFUSAL = InsufficientSpaceError(
    SpaceEstimate(bytes=90 * GB, free_bytes=20 * GB, where="C:", per_recording={})
)


@pytest.mark.parametrize(
    "answer, expected",
    [(QMessageBox.StandardButton.Yes, True), (QMessageBox.StandardButton.No, False)],
)
def test_a_space_refusal_asks_and_follows_the_answer(
    qapp, monkeypatch, answer, expected
):
    asked = {}

    def _question(parent, title, text, buttons, default):
        asked.update(title=title, text=text, default=default)
        return answer

    monkeypatch.setattr(QMessageBox, "question", _question)
    assert space_prompt.launch_anyway(QWidget(), "Launch run", _REFUSAL) is expected
    assert "90.0 GB" in asked["text"] and "20.0 GB" in asked["text"]
    assert asked["default"] == QMessageBox.StandardButton.No


def test_any_other_failure_is_not_a_question(qapp, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: pytest.fail("asked about a non-space error")
    )
    assert space_prompt.launch_anyway(QWidget(), "t", ValueError("bad config")) is False


_LAUNCH = re.compile(r"\.(run_processing|retry_run)\(")


def test_every_gui_launcher_offers_to_launch_anyway():
    gui = Path(space_prompt.__file__).resolve().parent
    launchers = sorted(
        path.relative_to(gui).as_posix()
        for path in gui.rglob("*.py")
        if _LAUNCH.search(path.read_text(encoding="utf-8-sig"))
    )
    assert launchers == ["launch.py", "pages.py"], launchers
    for rel in launchers:
        source = (gui / rel).read_text(encoding="utf-8-sig")
        assert "launch_anyway" in source and "ignore_free_space=True" in source, rel
