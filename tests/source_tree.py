"""Stop any test from writing into the repository tree, at the write.

``tests/conftest.py`` installs :func:`install` once per test process, before collection, and
fails a test that :func:`take` reports a write for. The class it guards, how it hid and the rule
are in ``tests/test_source_tree_writes.py``'s docstring; this module is the mechanism.

It is a ``sys.addaudithook``: CPython raises an audit event for every ``open`` and every
``os.mkdir`` / ``os.remove`` / ``os.rename`` / ... a Python process makes, so the hook sees the
write *before it happens* and from any test, fixture or thread in the process — including a
file created and deleted inside one test, which leaves nothing a before/after comparison of the
tree could see. It raises, so the write never lands and no other test sees it, and it records,
so a test that swallows the exception (``except Exception``) still fails at teardown.

What it cannot see: a write made by a child process (a spawned worker, a ``subprocess.run``),
and one made by C code that does not go through Python's file functions.
"""

from __future__ import annotations

import os
import pathlib
import sys
import types

#: Write-shaped audit events, and which positional arguments are paths they create, change or
#: remove. ``open`` is filtered by its flags in :func:`written_paths`; ``os.rename`` changes
#: both of its paths; ``os.symlink`` / ``os.link`` write only the new name.
_PATH_ARGS: dict[str, tuple[int, ...]] = {
    "open": (0,),
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.symlink": (1,),
    "os.link": (1,),
    "os.truncate": (0,),
    "os.utime": (0,),
    "shutil.rmtree": (0,),
}

#: The ``dir_fd`` argument of each event that takes one. A relative path with a real ``dir_fd``
#: is relative to that directory, not to the working directory, and cannot be resolved here;
#: ``shutil.rmtree`` is the one caller in practice, and its own event names the top directory.
_DIR_FD_ARG = {
    "os.mkdir": 2,
    "os.rmdir": 1,
    "os.remove": 1,
    "os.rename": 2,
    "os.symlink": 2,
    "os.link": 2,
    "os.utime": 3,
}

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

#: Names under the repository root that tools own and a test run may write, each with its writer.
#: A path is allowed when any of its components is one of these; everything else is the tree.
ALLOWED_PARTS = frozenset(
    {
        # The interpreter's bytecode cache, written on import.
        "__pycache__",
        # pytest's cache provider (``--lf``, ``--nf``), written at session end. It builds the
        # directory as ``pytest-cache-files-*`` beside it and renames it into place
        # (``_ALLOWED_PREFIXES``).
        ".pytest_cache",
        # git's directory: ``scripts/suite_progress_plugin.py`` appends the run log
        # (``.git/detail-test-runs.jsonl``), shared by every worktree of the clone.
        ".git",
    }
)


#: Tool-owned names that carry a random suffix.
_ALLOWED_PREFIXES = ("pytest-cache-files-",)


class SourceTreeWriteError(AssertionError):
    """A test tried to write into the repository tree.

    An ``AssertionError`` rather than an ``OSError``, so code under test that handles a failed
    write (``except OSError``) cannot turn the guard into a quiet fallback.
    """


def written_paths(event: str, args: tuple) -> list[str]:
    """The paths an audit event would create, change or remove; empty for anything else."""
    indices = _PATH_ARGS.get(event)
    if indices is None:
        return []
    if event == "open":
        _path, mode, flags = args
        if isinstance(flags, int) and flags >= 0:
            if not flags & _WRITE_FLAGS:
                return []
        elif not (isinstance(mode, str) and any(c in mode for c in "wax+")):
            return []
    fd_index = _DIR_FD_ARG.get(event)
    relative_to_fd = (
        fd_index is not None
        and len(args) > fd_index
        and args[fd_index] not in (None, -1)
    )
    out = []
    for index in indices:
        path = args[index]
        if not isinstance(path, (str, bytes, os.PathLike)):
            continue  # a file descriptor
        path = os.fsdecode(path)
        if relative_to_fd and not os.path.isabs(path):
            continue
        out.append(path)
    return out


def in_tree(
    path: str, root: str | os.PathLike, *, flavour: types.ModuleType = os.path
) -> bool:
    """Whether ``path`` is inside ``root`` and not under a tool-owned name.

    Both sides are compared in one spelling, ``normcase(abspath(...))``, because the two arrive
    spelled by different rules. The root is pytest's rootdir, which keeps the working
    directory's spelling, and a Windows shell can start in ``c:\\...``. The paths written come
    from ``Path.resolve()``, which gives the disk's ``C:\\...``. Compared as raw strings, no
    write was ever in the tree, and the guard refused nothing for the whole run (#66).
    ``abspath`` on the root too: a rootless ``/r`` gains the current drive on Windows, as the
    written path does. ``flavour`` is the path module, so a test can check Windows rules
    (``ntpath``) on any platform.
    """
    full = flavour.normcase(flavour.abspath(path))
    base = flavour.normcase(flavour.abspath(root))
    if full != base and not full.startswith(base + flavour.sep):
        return False
    if full == base:
        return False  # ``mkdir(parents=True, exist_ok=True)`` passes through the root itself
    parts = full[len(base) + 1 :].split(flavour.sep)
    return not any(
        part in ALLOWED_PARTS or part.startswith(_ALLOWED_PREFIXES) for part in parts
    )


_root: list[pathlib.Path] = []
_writes: list[tuple[str, str]] = []


def _hook(event: str, args: tuple) -> None:
    if event not in _PATH_ARGS or not _root:
        return
    for path in written_paths(event, args):
        if in_tree(path, _root[0]):
            _writes.append((event, path))
            raise SourceTreeWriteError(
                f"a test tried to write into the repository tree: {event} {path!r}. Tests "
                "write under tmp_path; a fake module or package goes in tmp_path and the code "
                "under test is pointed at it. See tests/test_source_tree_writes.py."
            )


def install(root: pathlib.Path) -> None:
    """Guard ``root`` for the rest of this process. An audit hook cannot be removed, so once."""
    if _root:
        return
    _root.append(pathlib.Path(os.path.abspath(root)))
    sys.addaudithook(_hook)


def take() -> list[tuple[str, str]]:
    """The writes refused since the last call, and forget them."""
    out = list(_writes)
    _writes.clear()
    return out
