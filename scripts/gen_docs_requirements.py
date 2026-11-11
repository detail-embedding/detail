"""mkdocs-gen-files — generate the GUI Requirements reference from the registry.

The point of generating it: **a rule that cannot be enforced cannot be written down.** Every
sentence on this page is the first line of a function in ``detail/gui/requirements.py`` whose
body is the check, so the guidelines and the tests cannot say different things. That failure is
a recorded class here — the style block claimed the window geometry and rail split were
persisted while no Qt geometry call existed anywhere in the package — and prose is where it
always begins.

The page also prints what is **not** enforced yet (``NOT_YET_SWEPT``), so a rule mid-migration
reads as mid-migration rather than as settled fact. That list is a ratchet, checked by
``tests/gui/test_requirements.py``.

Reading only names and docstrings, never calling a predicate, is what lets this run in the docs
build: `requirements.py` imports no Qt at module scope and the docs toolchain does not install
the `gui` extra.
"""

import mkdocs_gen_files

from detail.gui import requirements as reqs

_PAGE = "reference/gui-requirements.md"

_LEAD = """# GUI requirements

Every structural rule the app's widgets are checked against, generated from
`detail/gui/requirements.py`. Each is a function whose docstring is the sentence below and whose
body is the check, so this page cannot drift from what is enforced.

The reasoning behind these — why the rules exist, what each one cost when it was missing — lives
in `docs/gui_design.md`. This page is the list.
"""


#: A line under a kind's heading where the kind is not self-explanatory. "Button" needs none;
#: "Package" does, because it is the one subject that is not a widget.
_KIND_NOTES = {
    "package": (
        "Rules about what the source tree *never* does. These cannot be checked by walking "
        "widgets — an icon on a list item is not a `QWidget` — so the subject is `detail/gui/` "
        "itself, read as a parse tree rather than as text."
    ),
}


def _rows() -> str:
    out = []
    for kind in reqs.kinds():
        out.append(f"\n## {kind.capitalize()}\n")
        if kind in _KIND_NOTES:
            out.append(f"{_KIND_NOTES[kind]}\n")
        for rule in reqs.for_kind(kind):
            out.append(f"- **{rule.statement}**")
            # A rule that holds on pages and not yet on dialogs must read that way here, or the
            # page claims more than the check delivers — which is the drift this file exists to
            # prevent. The containers it does *not* cover carry a `NOT_YET_SWEPT` entry below.
            if rule.containers is not None:
                out.append(f"    - swept on: {', '.join(rule.containers)}")
            if rule.exempt:
                out.append(
                    f"    - {len(rule.exempt)} exempt: "
                    + "; ".join(f"`{k}` ({v})" for k, v in sorted(rule.exempt.items()))
                )
    return "\n".join(out)


def _backlog() -> str:
    if not reqs.NOT_YET_SWEPT:
        return "\nEvery written-down rule is enforced over every subject.\n"
    lines = [
        "\n## Written down, not yet enforced everywhere\n",
        "These are in the guidelines and are **not** checked over every subject — a spot-check on",
        "one or two instances, a source scan, or nothing at all. The list only shrinks.\n",
    ]
    for rule, why in sorted(reqs.NOT_YET_SWEPT.items()):
        lines.append(f"- **{rule}** — {why}")
    return "\n".join(lines)


with mkdocs_gen_files.open(_PAGE, "w") as handle:
    handle.write(_LEAD)
    handle.write(_rows())
    handle.write("\n")
    handle.write(_backlog())
    handle.write("\n")
