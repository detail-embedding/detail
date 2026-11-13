"""The CLI's half of Cluster profile setup: `install-extra`, and what `doctor` says about the
optional step.

Surface parity with Configuration ▸ Cluster profile: every action that page offers is reachable
here, against the same API (``api/extras``). Nothing here installs anything real — the
installer is a stand-in.
"""

from __future__ import annotations

from click.testing import CliRunner

from detail.api import checks as api_checks
from detail.api import extras
from detail.cli import main
from detail.core.dataclasses import CheckResult, CheckStatus


def test_install_extra_prints_the_command_without_running_it(monkeypatch):
    monkeypatch.setattr(extras, "run_installer", lambda *a, **k: 1 / 0)
    res = CliRunner().invoke(main, ["install-extra", "analysis", "--print-command"])
    assert res.exit_code == 0, res.output
    assert "install" in res.output


def test_install_extra_runs_the_api_install_with_a_bar(monkeypatch):
    calls = []
    states = iter(
        [
            extras.ExtraStatus("analysis", ("shap",), ("shap",)),
        ]
    )
    monkeypatch.setattr(extras, "status", lambda extra: next(states))

    def install(extra, progress):
        progress("installing", 0.5, "Installing collected packages: shap")
        calls.append(extra)
        return extras.ExtraStatus(extra, ("shap",), ())

    monkeypatch.setattr(extras, "install_extra", install)
    res = CliRunner().invoke(main, ["install-extra", "analysis"])
    assert res.exit_code == 0, res.output
    assert calls == ["analysis"] and "installed shap" in res.output


def test_install_extra_shows_the_installers_output_when_it_fails(monkeypatch):
    monkeypatch.setattr(
        extras,
        "status",
        lambda extra: extras.ExtraStatus(extra, ("shap",), ("shap",)),
    )

    def install(extra, progress):
        raise extras.InstallError("ERROR: No matching distribution found for shap")

    monkeypatch.setattr(extras, "install_extra", install)
    res = CliRunner().invoke(main, ["install-extra", "analysis"])
    assert res.exit_code != 0 and "No matching distribution" in res.output


def test_doctor_reports_the_optional_step_and_still_exits_zero(monkeypatch):
    infos = api_checks.list_checks()
    results = [CheckResult(i.check_id, CheckStatus.OK, "") for i in infos]
    results = [r for r in results if r.check_id != "shap"] + [
        CheckResult("shap", CheckStatus.WARN, "not installed"),
    ]
    monkeypatch.setattr(api_checks, "run_checks", lambda **k: results)
    res = CliRunner().invoke(main, ["doctor"])
    assert res.exit_code == 0, res.output
    assert "Analysis (optional)" in res.output
    assert "SHAP (Cluster profile)" in res.output
    assert "Optional (Cluster profile):" in res.output
    assert "detail install-extra analysis" in res.output


def test_install_extra_offers_only_the_analysis_extra():
    res = CliRunner().invoke(main, ["install-extra", "describe"])
    assert res.exit_code != 0 and "'analysis'" in res.output
