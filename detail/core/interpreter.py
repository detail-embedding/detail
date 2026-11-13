"""How DETAIL starts a Python interpreter: never with its working directory on ``sys.path``.

``python -m x``, ``python -`` and ``python -c`` put the directory they start in first on
``sys.path``. DETAIL starts interpreters in directories it does not choose: a Slurm job runs where
``sbatch`` was called, which over ssh is ``$HOME``, and a local worker inherits whatever folder
the user launched the app from. A source checkout sitting in that folder (``~/detail``,
``~/sam3``) has no ``__init__.py`` at its root, so Python takes it for an empty namespace package
of that name and an editable install's finder never gets to answer. ``-P`` (``safe_path``, Python
3.11+) leaves the working directory off and every other entry, ``PYTHONPATH``'s included, alone.

``tests/core/test_interpreter.py`` is the guard; its module docstring is the write-up.
"""

from __future__ import annotations

import sys

#: Don't put the working directory (``-m``) or ``''`` (``-`` / ``-c``) first on ``sys.path``.
SAFE_PATH_FLAG = "-P"


def module_argv(module: str, *args: str) -> list[str]:
    """The argv that runs ``module`` with this environment's interpreter, wherever it starts."""
    return [sys.executable, SAFE_PATH_FLAG, "-m", module, *args]


def remote_python_command(rest: str) -> str:
    """Shell text that runs the activated environment's ``python`` with ``rest`` (``-`` or
    ``-m <module> ...``), wherever the remote shell happens to be."""
    return f"python {SAFE_PATH_FLAG} {rest}"
