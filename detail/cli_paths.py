"""The one path type every CLI path option and argument uses.

A path a user types may start with ``~``. A shell expands it only when it is unquoted and at
the start of a word, so ``--dir "~/Documents/DetailExperiments"`` (the form the docs show), a
``--dir=~/x`` in some shells, or a path from a script reaches the command as a literal ``~``.
``click.Path`` keeps it, and ``experiment create`` then made a folder named ``~`` in the
current directory, which the GUI never shows (#31).

:class:`UserPath` is ``click.Path`` with ``~`` expanded first, so ``exists=True`` checks the
real place. ``tests/test_cli_paths.py`` fails on a ``click.Path(`` anywhere else in
``detail/``.
"""

from __future__ import annotations

import os

import click


class UserPath(click.Path):
    """``click.Path`` that expands a leading ``~`` (and ``~user``) before checking the path."""

    def convert(self, value, param, ctx):
        if isinstance(value, (str, os.PathLike)):
            value = os.path.expanduser(value)
        return super().convert(value, param, ctx)
