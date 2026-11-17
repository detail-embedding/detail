"""The error an ordinary input mistake raises, so every surface can show it as one line.

A user who hands DETAIL a YAML with a missing bracket, a CSV saved in the wrong encoding or
a file that is not an archive has made an everyday mistake, not found a bug. The reader that
notices raises :class:`UserInputError` with a message that names the file (and the line,
row or field) and says what to do. The CLI's root group turns it into ``Error: <message>``
and exit 1 (``detail/cli.py::_ConsoleSafeGroup.invoke``); the GUI's import dialogs already
catch ``ValueError`` and show the message.

It is a ``ValueError`` so every existing ``except (FileNotFoundError, ValueError)`` keeps
catching it. The CLI root catches **this type, not ``ValueError``**: a ``ValueError`` from a
real programming mistake still prints its traceback, where it can be reported and fixed.
"""


class UserInputError(ValueError):
    """An ordinary input mistake: the message names the thing and says how to fix it."""
