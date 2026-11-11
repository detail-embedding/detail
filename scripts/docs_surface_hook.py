"""mkdocs hook — drop prose for capabilities this build hides (§25).

A shim. The marker grammar and the strip live in :mod:`detail.surfaces`, beside the table that
decides what is hidden, because the site build is only one of three readers that must agree
about a marker: the guard (``tests/test_surfaces.py``) and the in-app help panel
(``detail.help``) read the same source files and would otherwise publish what the site drops.

Marking is for prose that teaches a hidden capability as though it were available. It is *not*
for prose that documents an absence — ``docs/cli.md``'s "Not in the CLI" entries explain what the
CLI leaves out and why, which is what CLAUDE.md requires ("the undocumented gap is the bug, not
the omission itself").
"""

from detail.surfaces import strip_hidden


def on_page_markdown(markdown: str, page=None, **_kwargs) -> str:
    """A page with no markers is returned untouched, which is nearly all of them — including
    every page ``gen-files`` generates, since those render from registries and carry no prose
    of ours. Stripping here rather than later also drops the block's headings from ``page.toc``
    and keeps them out of the search index, which is built from the rendered content."""
    where = getattr(getattr(page, "file", None), "src_uri", None) or "<page>"
    return strip_hidden(markdown, where=where)
