"""The pytest reminder hook adds context and never blocks.

**The class: a hook that guesses and refuses.** Whether a command is a long run cannot be read off
its text, and a PreToolUse hook in the repo's ``.claude/settings.json`` runs for every session on
every machine. A refusal that guessed wrong would stall someone else's gate. So the hook may only
add context: ``test_it_never_decides_anything`` pins that it emits no permission decision for any
input, and ``test_the_settings_run_this_script_only_for_pytest`` pins the wiring — the settings
file names this script, under Bash, filtered to commands containing ``pytest``.

Proven to fire in a live session when it was added (#590): the reminder arrived as hook context on
a command containing ``pytest``, and a command without it never started the hook.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scripts import pytest_reminder_hook as hook

REPO = Path(__file__).resolve().parents[2]


def _payload(command: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command}}


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q tests/scripts",
        ".venv/bin/python -m pytest -q -n auto --dist loadfile",
        "python scripts/with_suite_lock.py --who me -- pytest -q -n auto",
        "timeout 600 python -m pytest -q $(python scripts/select_tests.py)",
    ],
)
def test_a_pytest_command_gets_the_reminder(command):
    out = hook.reminder_for(_payload(command))
    assert out["hookSpecificOutput"]["additionalContext"] == hook.REMINDER
    assert "run-tests" in hook.REMINDER


@pytest.mark.parametrize(
    "command", ["ls", "git status", "python scripts/pytest_reminder_hook.py", ""]
)
def test_other_commands_get_nothing(command):
    assert hook.reminder_for(_payload(command)) is None


@pytest.mark.parametrize(
    "payload",
    [_payload("pytest -q"), _payload("ls"), {}, {"tool_input": None}],
)
def test_it_never_decides_anything(payload):
    out = hook.reminder_for(payload) or {}
    specific = out.get("hookSpecificOutput", {})
    assert "permissionDecision" not in specific
    assert "decision" not in out and out.get("continue", True) is True


def test_unreadable_input_prints_nothing_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert hook.main() == 0
    assert capsys.readouterr().out == ""


def test_the_settings_run_this_script_only_for_pytest():
    settings = json.loads(
        (REPO / ".claude" / "settings.json").read_text(encoding="utf-8-sig")
    )
    (entry,) = settings["hooks"]["PreToolUse"]
    assert entry["matcher"] == "Bash"
    (command,) = entry["hooks"]
    assert command["if"] == "Bash(*pytest*)"
    script = command["command"].split()[-1].strip('"')
    assert script.startswith("${CLAUDE_PROJECT_DIR:-.}/"), script
    relative = script.removeprefix("${CLAUDE_PROJECT_DIR:-.}/")
    assert (REPO / relative).resolve() == Path(hook.__file__).resolve()
