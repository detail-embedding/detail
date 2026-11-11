"""A stored relative path is forward-slash on every OS (Tenet 6).

**The class.** ``str(p.relative_to(base))`` yields backslashes on Windows and forward slashes
everywhere else, so a value-equality test passes on Linux and on the dev box while the path
written into a experiment file is non-portable. Nothing is wrong at the line where it is
written, and CI is Linux-only — so a source scan is the only thing that can see it.

**Why this file was rewritten.** It is the scan CLAUDE.md points at as "the
``test_path_serialization.py`` idiom", copied by four other guards — and it was a line-by-line
regex with **no negative case**, so nobody had ever watched it catch anything. Fed the forms a
person would actually write, it missed four:

===========================================  ==================
``str(path.relative_to(base))``              caught
``str(Path(p).relative_to(base))``           **missed** — a call in the receiver
``str(p.resolve().relative_to(base))``       **missed** — same
``f"{path.relative_to(base)}"``              **missed** — and it is the modern spelling
``"%s" % path.relative_to(base)``            **missed**
a ``str(`` split across two lines            **missed** — the scan read one line at a time
===========================================  ==================

Every one of those produces exactly the bug this file exists to stop. The f-string is the
one that matters: it is what somebody writes today, and ``__format__`` falls through to
``__str__``, so it is the same backslash on the same platform.

Read from the parse tree instead, for the reason ``gui/requirements.py`` gives: a text search
cannot tell a call from a docstring explaining why nobody makes it, and it cannot see a call
that wrapped onto a second line. And the negative case is now the point of the file — a scan
whose pattern quietly stops matching is indistinguishable from a package that never offends.
"""

import ast
import pathlib
import textwrap

import detail


def _stringifies_a_relative_to(node: ast.AST) -> bool:
    """Whether ``node`` turns a ``relative_to`` result into text.

    Three spellings, one bug: ``str(...)``, an f-string placeholder, and ``%``-formatting.
    ``.as_posix()`` is none of them — it returns the portable string directly, which is the
    whole fix — and a bare ``relative_to`` whose result is used as a *path* is fine too
    (``storage/export.py`` takes ``.parts`` off it, then calls ``as_posix``).
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (node.func.attr == "relative_to")
    )


def _offenders(root: pathlib.Path) -> list[str]:
    """Every place under ``root`` that renders a ``relative_to`` result as text."""
    found = []
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            hit = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "str"
                and node.args
                and _stringifies_a_relative_to(node.args[0])
            ):
                hit = node
            elif isinstance(node, ast.FormattedValue) and _stringifies_a_relative_to(
                node.value
            ):
                hit = node
            elif (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mod)
                and _stringifies_a_relative_to(node.right)
            ):
                hit = node
            if hit is not None:
                found.append(f"{py.name}:{hit.lineno}")
    return found


def test_no_backslash_prone_path_serialization():
    package = pathlib.Path(detail.__file__).parent
    offenders = _offenders(package)
    assert not offenders, (
        "use .as_posix() instead of rendering a relative_to() result as text — str(), an "
        "f-string and %-formatting all give backslashes on Windows (Tenet 6):\n"
        + "\n".join(offenders)
    )


def test_the_scan_would_catch_every_spelling_of_the_bug(tmp_path):
    """Guard the guard, and the reason this file exists.

    Each of these is the same defect, and the regex this replaced caught only the first. A scan
    with no negative case cannot tell "the package is clean" from "the pattern stopped
    matching", and this one had been the template for four other guards while nobody had seen
    it reject anything.
    """
    spellings = {
        "plain": "    return str(path.relative_to(base))",
        "call_in_receiver": "    return str(Path(p).relative_to(base))",
        "method_in_receiver": "    return str(p.resolve().relative_to(base))",
        "fstring": '    return f"{path.relative_to(base)}"',
        "percent": '    return "%s" % path.relative_to(base)',
        "wrapped_over_two_lines": "    return str(\n        path.relative_to(base)\n    )",
    }
    for name, body in spellings.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "offender.py").write_text(
            f"def build(path, base, p, Path):\n{body}\n", encoding="utf-8"
        )
        assert _offenders(directory), f"the scan misses the {name!r} spelling"


def test_the_scan_passes_what_is_actually_correct(tmp_path):
    """And the other half: the fix, and the two shapes in `detail/` that only look like it.

    A rule that also rejects the correct spelling gets worked around rather than followed.
    `storage/video_roots.py` calls `relative_to` for a containment check and throws the result
    away; `storage/export.py` keeps it as a *path*, reads `.parts`, and calls `as_posix` at the
    end. Neither renders it as text, and neither is a bug.
    """
    (tmp_path / "fine.py").write_text(
        textwrap.dedent("""
            def the_fix(path, base):
                return path.relative_to(base).as_posix()


            def containment_check(path, root):
                try:
                    path.resolve().relative_to(root)
                    return True
                except ValueError:
                    return False


            def kept_as_a_path(path, base):
                rel = path.relative_to(base)
                if rel.parts and rel.parts[0] == "datasets":
                    return None
                return rel.as_posix()
            """),
        encoding="utf-8",
    )
    assert not _offenders(tmp_path)
