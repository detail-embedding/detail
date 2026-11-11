#!/bin/sh
# Run a stdlib-only script with the first Python that actually works, on Linux, macOS and Windows.
#
# Claude Code runs the repo's statusLine and hook commands through a shell, and no one interpreter
# name exists everywhere: `python3` on Linux and macOS; `python` or `py` on Windows, where
# `python3` is usually a Microsoft Store stub that prints an error and exits non-zero. So each
# candidate is tried with a trivial `-c` first, and the stub fails that and is skipped. The repo's
# own venv comes first (bin/ on POSIX, Scripts/ on Windows), then the names on PATH.
#
# If none works, exit 0 with no output: a missing interpreter must not turn into a hook error on
# every pytest command, or a status line full of error text.
#
# Usage: sh scripts/run_python.sh <script.py> [args...]
# LF line endings are pinned in .gitattributes: a CRLF checkout of this file does not run.

root="${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"

# Whether an interpreter runs at all and is Python 3.9+. Its words are the arguments, so a path
# with spaces (C:/Users/First Last/...) stays one word.
works() {
    "$@" -c "import sys; sys.exit(sys.version_info < (3, 9))" >/dev/null 2>&1 </dev/null
}

for py in "$root/.venv/bin/python" "$root/.venv/Scripts/python.exe"; do
    if [ -f "$py" ] && works "$py"; then
        exec "$py" "$@"
    fi
done
for py in python3 python; do
    if works "$py"; then
        exec "$py" "$@"
    fi
done
# The Windows launcher, told to pick a Python 3.
if works py -3; then
    exec py -3 "$@"
fi
exit 0
