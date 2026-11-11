"""Fail-closed guard: a decoded video frame is drawn into a view that zooms, not a bare label.

**The class.** *A capability that is a property of a shared widget reaches only the surfaces that
were rewritten to use it; the rest keep drawing a plausible, un-zoomable picture, and nothing
fails.* #389 built `gui/image_view.py` after zoom was asked for on every image the app shows,
and wired the four surfaces in front of it. Five months later Detection Results, Tracking Results
and Behavior Query's Edit clips were still `QLabel.setPixmap(_to_pixmap(rgb).scaled(pane.size()))`
— and so were the model-input crop scrubber, both embedding panes and both stream previews. Nine
surfaces. Every one built in a test, every one green, every one rendering a frame nobody could
lean into.

**Why no behavioural test sees it.** The widget *works*: it shows the frame, at the size it was
given. The missing thing is a gesture, and the sweep that could have noticed — the `video` kind in
`detail/gui/requirements.py` — finds a surface's picture by looking for an `ImageView`, so a pane
that is a plain `QLabel` is not a failing picture but **no picture at all**. An absent subject is
the quietest way for a guard to be green, which is `tests/gui/subjects.py`'s own lesson.

So this scans instead, for the shape the bug is written in: a **frame** (anything derived from
``_to_pixmap``, this package's ndarray→pixmap converter, or an image file loaded off disk) handed
to ``setPixmap``. The seams that legitimately do that are `image_view.py`, which *is* the zoomable
view, and `full_frame.py`, which hosts one. Everything else must be on :data:`ALLOWED` with the
reason it is not a pane somebody needs to zoom — a thumbnail that opens a player, a cell that
opens the frame full size, a plot.

An entry is a claim, so :func:`test_no_stale_allowances` fails on one that no longer matches
anything — the `test_dialog_space` ratchet, where nine exemptions went on excusing dialogs that
had become buildable and nothing said so.
"""

import ast
from pathlib import Path

import detail

_GUI = Path(detail.__file__).parent / "gui"

#: The modules that legitimately put a frame into a widget: the zoomable view itself, and the
#: dialog that hosts one. A third entry here would mean a second way to show a frame.
_SEAMS = ("image_view.py", "full_frame.py")

#: ``file.py:Owner`` -> why this frame is not drawn into a zoomable view. Each is a deliberate
#: non-fix, in the `fix-the-class` sense: the sweep reached it and stopped for a stated reason.
ALLOWED: dict[str, str] = {
    # -- thumbnails: small by design, and each opens something bigger ------------------- #
    "clip_grid.py:_Cell.set_image": (
        "a strip cell — a single click marks it and loads it into the player beside the row, "
        "and a double click opens the clip in a window of its own"
    ),
    "cluster_strip.py:_Thumb._paint": (
        "a cluster member's crop — a click plays it in place, a double click opens the whole "
        "frame in a player; the crop is for telling members apart, not for studying one"
    ),
    "quick_preview.py:QuickPreview._paint": (
        "a strided sample of a clip in a 132px square — double-click opens it in a real player, "
        "which is the answer to wanting a better look at it"
    ),
    "pages.py:_ExperimentCard.set_preview": (
        "an experiment card's still, an identifier rather than something to read; "
        "double-clicking the card opens the experiment"
    ),
    # -- cells that open the frame full size (`full_frame.EnlargeableThumb`) ------------- #
    "preview.py:_MatrixCell.show_overlay": (
        "a prompt-test grid cell; double-clicking it opens that frame full size and zoomable"
    ),
    "tools/quick_segment.py:_SampledMatrix._fill": (
        "a comparison grid cell; double-clicking it opens that frame full size and zoomable"
    ),
    "keypoint_test.py:KeypointTestDialog._redraw": (
        "a Test cut-off frame card; double-clicking it opens that frame full size and "
        "zoomable, drawn at the cut-off the grid shows"
    ),
    "pages.py:GlobalModelDetailPage._build": (
        "the model preview; double-clicking it opens the image full size and zoomable"
    ),
    # -- plots, which are a separate decision -------------------------------------------- #
    "comparison.py:_FigureThumb.__init__": (
        "a comparison figure is a plot headed for a manuscript, and opens in the OS viewer "
        "(#397); zoom on plots is OQ-VIEW-7, deferred by the user"
    ),
}


def _owners(tree: ast.AST) -> dict:
    """Every node -> the ``Class.method`` enclosing it, innermost winning."""
    out: dict = {}

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                for inner in ast.walk(child):
                    out[inner] = name
                walk(child, name + ".")

    walk(tree, "")
    return out


def _is_frame(node: ast.AST) -> bool:
    """Whether this expression produces a *picture of something*, rather than a swatch.

    Two spellings, because both are how a frame reaches a label in this package:
    ``_to_pixmap(rgb)`` (the ndarray converter every frame surface uses) and
    ``QPixmap(str(path))`` (an image read off disk). A ``QPixmap(14, 14)`` colour swatch is
    neither, and must not be caught — it is not something anybody zooms.
    """
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        # The public converter too (`full_frame.to_pixmap`, however it is imported): the Test
        # cut-off window drew its frame cards through it into bare labels, and a scan that
        # knew only the private spelling passed it.
        name = getattr(inner.func, "id", None) or getattr(inner.func, "attr", None)
        if name in ("_to_pixmap", "to_pixmap"):
            return True
        if not isinstance(inner.func, ast.Name):
            continue
        if inner.func.id == "QPixmap" and any(
            isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "str"
            for arg in inner.args
        ):
            return True
    return False


def frame_pane_offenders(source: str, rel: str) -> list[str]:
    """``rel:Owner`` for every frame drawn into a widget in ``source``.

    A pure function over text, so the negative cases below can hand it a literal rather than
    build a widget — `tests/test_docs_nav.py::_undeclared` is the idiom, and a negative case
    that needs a fixture ends up testing the fixture.

    **One level of local assignment is followed.** ``pixmap = QPixmap(str(path))`` on one line
    and ``label.setPixmap(pixmap.scaledToWidth(360))`` on the next is the same bug written over
    two statements, and the first version of this scan missed exactly that shape in
    ``comparison.py`` and ``pages.py`` — the `test_path_serialization` lesson, where a guard
    four others were copied from turned out to catch one spelling of five.
    """
    tree = ast.parse(source)
    owners = _owners(tree)
    holds_a_frame: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_frame(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    holds_a_frame.add(target.id)
    out = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setPixmap"
            and node.args
        ):
            continue
        arg = node.args[0]
        named = any(
            isinstance(inner, ast.Name) and inner.id in holds_a_frame
            for inner in ast.walk(arg)
        )
        if _is_frame(arg) or named:
            out.append(f"{rel}:{owners.get(node, '<module>')}")
    return out


def _scan() -> list[str]:
    found = []
    for path in sorted(_GUI.rglob("*.py")):
        if path.name in _SEAMS:
            continue
        rel = path.relative_to(_GUI).as_posix()
        found.extend(frame_pane_offenders(path.read_text(encoding="utf-8-sig"), rel))
    return found


def test_a_frame_is_drawn_into_a_view_that_zooms():
    """Every frame surface either uses `ImageView` or says why it does not.

    The nine surfaces this was written against are all converted; what it holds now is the
    *next* one, which is the only reason a guard exists.
    """
    offenders = sorted(set(_scan()) - set(ALLOWED))
    assert not offenders, (
        f"{offenders} draw a decoded frame into a plain widget. A frame a user has to judge "
        f"something by — a mask, a paw, the moment a bout starts — is drawn into "
        f"`image_view.ImageView`, which zooms, pans and fits. If this one is a thumbnail that "
        f"opens something bigger, or a plot, add it to ALLOWED with that reason."
    )


def test_no_stale_allowances():
    """An entry that excuses nothing is one nobody will question — `test_dialog_space`'s class."""
    stale = sorted(set(ALLOWED) - set(_scan()))
    assert not stale, (
        f"these are allowed to draw a frame into a plain widget and no longer do: {stale}. "
        f"Delete the entries."
    )


def test_the_scan_would_catch_the_surfaces_it_was_written_for():
    """The negative case, in the spellings the nine offending surfaces were actually written in.

    Fed only one of these, a scan passes on the other four and nobody finds out: this is the
    `test_path_serialization` finding, where the guard CLAUDE.md calls "the idiom" caught one
    spelling and missed an f-string, a `%`-format, a call in the receiver and a wrapped line.
    """
    written = "\n".join(
        [
            # run_preview.py, before this change
            "class P:\n"
            "    def _display(self, payload):\n"
            "        pixmap = _to_pixmap(payload).scaled(\n"
            "            self._image.size(),\n"
            "            Qt.AspectRatioMode.KeepAspectRatio,\n"
            "        )\n"
            "        self._image.setPixmap(pixmap)\n",
            # behavior_query.py, before this change — built inline, one statement
            "class Q:\n"
            "    def _display(self, rgb):\n"
            "        self._src.setPixmap(_to_pixmap(rgb).scaled(400, 300))\n",
            # streams.py — a different scaling call, wrapped over lines
            "class R:\n"
            "    def _display(self, image):\n"
            "        self._image.setPixmap(\n"
            "            _to_pixmap(image).scaledToWidth(640)\n"
            "        )\n",
            # pages.py — a file loaded off disk, through a local
            "class S:\n"
            "    def _build(self):\n"
            "        pix = QPixmap(str(ref.preview_path))\n"
            "        preview.setPixmap(pix.scaledToWidth(360))\n",
            # a bare module-level one, no class to hide in
            "label.setPixmap(_to_pixmap(rgb))\n",
        ]
    )
    caught = sorted(frame_pane_offenders(written, "x.py"))
    assert caught == [
        "x.py:<module>",
        "x.py:P._display",
        "x.py:Q._display",
        "x.py:R._display",
        "x.py:S._build",
    ], caught


def test_the_scan_does_not_catch_what_is_already_right():
    """False positives matter as much: a scan that flags the fix teaches people to silence it.

    A swatch is not a frame, a cleared pane is not a frame, and a view being *handed* a frame
    through `set_frame` is the shape this guard exists to push surfaces towards.
    """
    fine = "\n".join(
        [
            "self._view.set_frame(_to_pixmap(rgb))",  # the fix
            "self._image.clear_frame('(no video)')",
            "swatch.setPixmap(QPixmap(14, 14))",  # a colour square
            "self._image.setPixmap(QPixmap())",  # clearing the cell
            "thumb.setPixmap(pixmap)",  # a pixmap from somewhere this scan cannot see
        ]
    )
    assert frame_pane_offenders(fine, "x.py") == []
