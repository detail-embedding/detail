"""Saving an export never writes over a file nobody was asked about (#29, findings 14 and 15).

**The class: a write whose target is not known to be free.** A save dialog asks before
replacing the name it returns. On a platform whose dialog does not add the suffix itself (Qt's
own dialog, on Linux), typing ``climb`` returns ``climb`` — which does not exist, so nothing was
asked — and the suffix added afterwards wrote straight over ``climb.mp4``. The montage video and
GIF, the montage's CSV list and the clip window's PNG all did this. A folder picker never asks
about the files inside it, so **One file per clip…** into a folder that already held the last
export's GIFs replaced them unasked.

**The rule.** A suffix is added in one place, ``animation_export.with_suffix_confirmed``, which
asks when the name it makes exists; a per-clip export checks its names in the folder before it
starts (``clip_montage.existing_gifs``). The scan below fails on a suffix appended anywhere
else under ``detail/gui``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from detail.api.animation import GIF
from detail.gui import animation_export as ax
from detail.gui.qt import QFileDialog, QMessageBox

_APPENDED_SUFFIX = re.compile(r"\.with_name\(\s*f[\"']\{[^}]*\.name\}\.")


def _appends_a_suffix(source: str) -> bool:
    return bool(_APPENDED_SUFFIX.search(source))


@pytest.fixture
def asked(monkeypatch):
    """Every replace question asked, answered with ``answer[0]``."""
    seen: list[str] = []
    answer = [QMessageBox.StandardButton.No]

    def question(_parent, _title, text, *_a, **_k):
        seen.append(text)
        return answer[0]

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return seen, answer


def _typed(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), ""))
    )


def test_a_suffix_added_after_the_dialog_asks_before_replacing(
    qapp, tmp_path, monkeypatch, asked
):
    seen, answer = asked
    (tmp_path / "climb.mp4").write_bytes(b"the earlier video")
    _typed(monkeypatch, tmp_path / "climb")

    assert ax.ask_save_path(None, "Export video", "x.mp4", ax.MP4) is None
    assert len(seen) == 1 and "climb.mp4" in seen[0]

    answer[0] = QMessageBox.StandardButton.Yes
    assert ax.ask_save_path(None, "Export video", "x.mp4", ax.MP4) == (
        tmp_path / "climb.mp4"
    )


def test_no_question_when_the_name_is_free_or_the_dialog_already_asked(
    qapp, tmp_path, monkeypatch, asked
):
    seen, _answer = asked
    _typed(monkeypatch, tmp_path / "climb")
    assert ax.ask_save_path(None, "t", "x.gif", GIF) == tmp_path / "climb.gif"
    (tmp_path / "climb.gif").write_bytes(b"x")
    # The name the dialog returned is the one it asked about.
    _typed(monkeypatch, tmp_path / "climb.gif")
    assert ax.ask_save_path(None, "t", "x.gif", GIF) == tmp_path / "climb.gif"
    assert seen == []


@pytest.mark.parametrize("kind, pattern", [("csv", "CSV list (*.csv)"), ("png", "")])
def test_a_list_and_a_picture_ask_the_same_way(
    qapp, tmp_path, monkeypatch, asked, kind, pattern
):
    seen, _answer = asked
    (tmp_path / f"clips.{kind}").write_bytes(b"x")
    _typed(monkeypatch, tmp_path / "clips")
    assert ax.ask_save_path(None, "t", "x", kind, pattern or "PNG (*.png)") is None
    assert len(seen) == 1


def test_the_replace_question_counts_and_names_the_files(qapp, asked):
    seen, _answer = asked
    ax.confirm_replace(None, "t", [f"{i:02d}.gif" for i in range(5)])
    assert seen == [
        "5 files with this name already exist there (00.gif, 01.gif, 02.gif and 2 more). "
        "Replace them?"
    ]


def test_no_suffix_is_appended_after_a_dialog_anywhere_else_in_the_gui():
    package = Path(ax.__file__).parent
    offenders = [
        p.relative_to(package).as_posix()
        for p in sorted(package.rglob("*.py"))
        if p.name != "animation_export.py"
        and _appends_a_suffix(p.read_text(encoding="utf-8-sig"))
    ]
    assert not offenders, (
        f"{offenders} add a suffix to a chosen path; use "
        "animation_export.ask_save_path / with_suffix_confirmed, which asks when the "
        "name that makes exists"
    )


def test_the_suffix_scan_would_catch_one():
    assert _appends_a_suffix('path = path.with_name(f"{path.name}.csv")')
    assert _appends_a_suffix("p = p.with_name(f'{chosen.name}.{kind}')")
    assert not _appends_a_suffix('tmp = final.with_name(f"{final.stem}.tmp")')
