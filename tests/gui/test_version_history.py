"""The Version History popover: what it says about the build, the history, and an update check.

Git is stubbed at ``detail.core.buildinfo``'s functions — the reads themselves are tested against
real repositories in ``tests/core/test_buildinfo.py``; this is what the popover makes of them.
"""

import pytest

from detail.core import buildinfo
from detail.gui import version_history
from tests.gui.conftest import drain_until

_INFO = buildinfo.BuildInfo(
    version="0.1.0",
    commit="abc1234",
    date="2026-09-20",
    commit_count=832,
    branch="main",
)


@pytest.fixture
def anchor(qapp):
    from detail.gui.qt import QPushButton

    button = QPushButton("Version History")
    button.show()
    yield button
    button.close()


def _popover(qapp, anchor, monkeypatch, *, history, check, info=_INFO):
    monkeypatch.setattr(buildinfo, "build_info", lambda: _INFO)
    monkeypatch.setattr(buildinfo, "history", lambda: history)
    monkeypatch.setattr(buildinfo, "check_for_updates", check)
    popover = version_history.VersionHistoryPopover(anchor, info)
    popover.show_at(anchor)
    return popover


def _row_texts(popover) -> list[str]:
    layout = popover._body_layout
    return [
        layout.itemAt(i).widget().text()
        for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
    ]


def test_history_rows_are_listed_in_the_order_git_gives_them(qapp, anchor, monkeypatch):
    entries = [
        buildinfo.HistoryEntry("c3", "2026-09-20", "newest (#3)"),
        buildinfo.HistoryEntry("c2", "2026-09-19", "middle (#2)"),
        buildinfo.HistoryEntry("c1", "2026-09-18", "oldest (#1)"),
    ]
    popover = _popover(qapp, anchor, monkeypatch, history=entries, check=lambda: None)
    assert drain_until(qapp, lambda: len(_row_texts(popover)) == 3)
    assert _row_texts(popover) == [
        "2026-09-20   newest (#3)",
        "2026-09-19   middle (#2)",
        "2026-09-18   oldest (#1)",
    ]
    popover.close()


def test_a_failed_update_check_is_shown_inline_and_can_be_retried(
    qapp, anchor, monkeypatch
):
    # Offline is the normal way this fails; it gets a sentence in the popover, not a modal
    # error box over a panel the user opened to read.
    def offline():
        raise buildinfo.UpdateCheckError("Could not fetch origin/main — offline.")

    popover = _popover(qapp, anchor, monkeypatch, history=[], check=offline)
    popover._check.click()
    assert drain_until(qapp, lambda: "offline" in popover._update_status.text())
    assert popover._check.isEnabled()
    popover.close()


def test_outside_a_git_checkout_the_popover_says_so(qapp, anchor, monkeypatch):
    bare = buildinfo.BuildInfo(version="0.1.0")
    popover = _popover(
        qapp, anchor, monkeypatch, history=[], check=lambda: None, info=bare
    )
    assert popover._build.text() == bare.describe()
    assert drain_until(qapp, lambda: len(_row_texts(popover)) == 1)
    assert "not a git checkout" in _row_texts(popover)[0]
    popover.close()


def test_opened_before_the_build_is_read_it_reads_the_build_itself(
    qapp, anchor, monkeypatch
):
    popover = _popover(
        qapp, anchor, monkeypatch, history=[], check=lambda: None, info=None
    )
    assert drain_until(qapp, lambda: popover._build.text() == _INFO.describe())
    popover.close()


def test_the_build_label_shortens_rather_than_widen_its_row(qapp):
    info = buildinfo.BuildInfo(
        version="0.1.0", commit="abc1234", date="2026-09-20", local_changes=True
    )
    full, version_and_date, date = info.labels()
    label = version_history.BuildLabel()
    label.set_build(info)
    assert label.minimumSizeHint().width() == 0
    full_width = label.sizeHint().width()
    height = label.sizeHint().height()
    metrics = label.fontMetrics()
    label.show()  # a hidden widget's resize event is held until it is shown

    label.resize(full_width, height)
    assert label.text() == full
    label.resize(metrics.horizontalAdvance(version_and_date), height)
    assert label.text() == version_and_date
    # The date is the "is my copy current?" part, so it is the last whole part standing.
    label.resize(metrics.horizontalAdvance(date), height)
    assert label.text() == date
    label.resize(metrics.horizontalAdvance(date) // 2, height)
    assert label.text().endswith("…")

    assert label.build_text() == full
    assert full in label.toolTip()
    # The hint comes from the full text, so shortening cannot ratchet the row narrower.
    assert label.sizeHint().width() == full_width
    label.close()
