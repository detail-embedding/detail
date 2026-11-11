"""``detail --version`` and ``detail version``: the CLI twin of the GUI's Version History panel.

Git is stubbed at ``detail.core.buildinfo``'s functions — the reads are tested against real
repositories in ``tests/core/test_buildinfo.py``. What this pins is that the CLI prints the same
sentences the GUI shows (both call ``describe()``) and turns a failed check into a clean error.
"""

import pytest
from click.testing import CliRunner

from detail.cli import main
from detail.core import buildinfo

_INFO = buildinfo.BuildInfo(
    version="0.1.0",
    commit="abc1234",
    date="2026-09-20",
    commit_count=832,
    branch="main",
)
_HISTORY = [
    buildinfo.HistoryEntry("abc1234", "2026-09-20", "gui: newest (#3)"),
    buildinfo.HistoryEntry("def5678", "2026-09-19", "pipeline: older (#2)"),
]


@pytest.fixture
def stubbed(monkeypatch):
    fetches = []

    def check():
        fetches.append(True)
        return buildinfo.UpdateStatus(behind=3, ahead=0, upstream_date="2026-09-22")

    monkeypatch.setattr(buildinfo, "build_info", lambda: _INFO)
    monkeypatch.setattr(buildinfo, "history", lambda limit=200: _HISTORY[:limit])
    monkeypatch.setattr(buildinfo, "check_for_updates", check)
    return fetches


def test_version_flag_prints_the_build_label(stubbed):
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == _INFO.label()


def test_version_prints_the_build_and_touches_no_network_by_default(stubbed):
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == _INFO.describe()
    assert stubbed == []


def test_version_history_lists_the_newest_n_changes(stubbed):
    result = CliRunner().invoke(main, ["version", "--history", "2"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1:] == [
        "  2026-09-20  abc1234  gui: newest (#3)",
        "  2026-09-19  def5678  pipeline: older (#2)",
    ]


def test_version_check_says_how_far_behind(stubbed):
    result = CliRunner().invoke(main, ["version", "--check"])
    assert result.exit_code == 0, result.output
    assert stubbed == [True]
    assert result.stdout.splitlines()[-1] == (
        "3 commits behind origin/main (latest 2026-09-22). "
        "To update: git pull, then reinstall."
    )
    assert "Fetching origin/main" in result.stderr


def test_a_failed_check_is_a_clean_error_not_a_traceback(stubbed, monkeypatch):
    def offline():
        raise buildinfo.UpdateCheckError("Could not fetch origin/main — offline.")

    monkeypatch.setattr(buildinfo, "check_for_updates", offline)
    result = CliRunner().invoke(main, ["version", "--check"])
    assert result.exit_code == 1
    assert "Could not fetch origin/main" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
