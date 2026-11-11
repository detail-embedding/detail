"""A Claude Code PreToolUse hook: when a Bash command runs pytest, remind Claude of ``run-tests``.

A reminder, not a block. Whether a command is "a long run" cannot be read reliably off its text
(a wrapper, a selected file list, ``-n auto`` or not), and a block that guesses wrong stalls
every session in the repo, on every machine, including other people's gates. A reminder that
guesses wrong costs a sentence. So this never refuses anything: it adds context and exits 0.

Configured in ``.claude/settings.json`` with ``"if": "Bash(*pytest*)"``, so Claude Code does not
start it for other commands. Standard library only; any unreadable input prints nothing.
"""

from __future__ import annotations

import json
import re
import sys

REMINDER = (
    "This command runs pytest — follow the run-tests skill. Post the result as its own "
    "message before any analysis. If the run will take more than a few seconds and the session "
    "may be watched from the Claude app (CLAUDE_CODE_BRIDGE_SESSION_ID set), run it in the "
    "background and relay `python scripts/suite_progress.py watch` through a Monitor."
)

_PYTEST = re.compile(r"\bpytest\b")


def reminder_for(payload: dict) -> dict | None:
    """The hook output for one tool call, or ``None`` when the command does not run pytest."""
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not _PYTEST.search(command):
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": REMINDER,
        }
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    output = reminder_for(payload) if isinstance(payload, dict) else None
    if output is not None:
        print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
