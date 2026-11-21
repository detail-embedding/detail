"""Rewrite a matplotlib SVG so it opens in Illustrator as something a person can edit.

matplotlib's SVG draws correctly everywhere, but it is hard to work on in Illustrator:

    clipping   every artist inside an axes is clipped to it, and each clip arrives as a clipping
               mask around the object. Removed. Nothing the comparison figures draw outside its
               axes is meant to be hidden (a dot at zero is a whole dot); render the result
               before relying on that for a new panel.
    markers    dots and tick marks are <use> references to one shared shape. Written out as
               ordinary paths, so each is a plain object.
    math text  a label with an italic, a subscript or a superscript is written one character per
               <tspan>, each at its own x, and arrives as loose characters. Rebuilt as a single
               <text> whose runs differ only in style and baseline shift. matplotlib also lifts
               such a label by the part of a point it rounds its height up by, so it sits above
               the plain labels beside it; the rebuilt text is put back on the baseline.
    spaces     every <text> keeps its spaces as written (xml:space="preserve").
    fonts      the quoted font-family list is reduced to its first family, as an attribute.
    styles     style="..." becomes attributes, and the document-wide stylesheet (round joins) is
               written onto each stroked shape, so nothing depends on CSS being read.
    opacity    a stroke-only shape's stroke-opacity becomes the object's opacity.
    groups     a group around a single object is dropped and the object takes its name; a text
               object carries no name, so Illustrator lists it by its content.

What Illustrator does whatever the SVG says (measured by scripting Illustrator 2027): it makes
one point-text object per style run and left-justifies it, so a label such as "dose: P = 0.002"
arrives as three grouped text objects. Everything above is what a file can control.

Ported from the paper's ``fig_group_differences/svg_edit.py`` (detail#76), where each step was
checked in Illustrator. ``figures.save_figure(..., editable_svg=True)`` runs it on every SVG a
comparison writes; other exporters keep matplotlib's own SVG, which a figure with a pattern
fill or a gradient still needs (this rewrite refuses a file whose references it cannot inline).
"""

import collections
import copy
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from matplotlib import font_manager
from matplotlib.ft2font import LoadFlags

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
XML = "http://www.w3.org/XML/1998/namespace"
HREF = f"{{{XLINK}}}href"
#: ids matplotlib gives a text's wrapper group; dropped, so the text is listed by its content
AUTO_TEXT_ID = re.compile(r"^text_\d+$")
PURE_TRANSLATE = re.compile(r"^translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*\)$")
NO_ROTATION = re.compile(r"^rotate\(-?0(\.0+)?[ ,)]")


class SvgRewriteError(ValueError):
    """The SVG holds something this rewrite cannot turn into plain objects."""


def _tag(name: str) -> str:
    return f"{{{SVG}}}{name}"


def _fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _remove(root, unwanted) -> int:
    count = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if unwanted(child):
                parent.remove(child)
                count += 1
    return count


def _styles_to_attributes(root) -> None:
    for el in root.iter():
        style = el.attrib.pop("style", None)
        if style is None:
            continue
        for part in style.split(";"):
            key, found, value = part.partition(":")
            if not found:
                continue
            key, value = key.strip(), value.strip()
            if key == "font":
                raise SvgRewriteError(
                    f"a font shorthand this rewrite does not read: {value!r}"
                )
            if key == "font-family":
                value = value.split(",")[0].strip().strip("'\"")
            if key == "font-size":
                value = value.removesuffix("px")
            el.set(key, value)


def _write_out_stylesheet(root) -> None:
    """matplotlib's ``*{stroke-linejoin: round; stroke-linecap: butt}``, per stroked shape."""
    for el in root.iter():
        if el.get("stroke", "none") != "none":
            el.attrib.setdefault("stroke-linejoin", "round")


def _simplify_opacity(root) -> None:
    for el in root.iter():
        stroke, fill = el.get("stroke-opacity"), el.get("fill-opacity")
        if stroke is None or "opacity" in el.attrib:
            continue
        if el.get("fill") == "none" or stroke == fill:
            el.set("opacity", stroke)
            del el.attrib["stroke-opacity"]
            el.attrib.pop("fill-opacity", None)


def _inline_uses(root) -> int:
    """Each <use> becomes a copy of the shape it names, moved to the use's x and y."""
    by_id = {el.get("id"): el for el in root.iter() if el.get("id")}
    count = 0
    for parent in list(root.iter()):
        for index, child in enumerate(list(parent)):
            if child.tag != _tag("use"):
                continue
            shape = copy.deepcopy(by_id[child.get(HREF).lstrip("#")])
            del shape.attrib["id"]
            for key, value in child.attrib.items():
                if key not in (HREF, "x", "y", "transform"):
                    shape.set(key, value)
            move = f"translate({child.get('x', '0')} {child.get('y', '0')})"
            shape.set(
                "transform", " ".join(filter(None, [child.get("transform"), move]))
            )
            shape.tail = child.tail
            parent[index] = shape
            count += 1
    return count


def _advance(char: str, size: float, style: str, weight: str, family: str) -> float:
    """How far ``char`` moves the pen, in the SVG's units (points)."""
    bold = weight not in ("400", "normal")
    prop = font_manager.FontProperties(
        family=family, style=style, weight="bold" if bold else "normal"
    )
    font = font_manager.get_font(font_manager.findfont(prop))
    font.set_size(size, 72)
    return (
        font.load_char(ord(char), flags=LoadFlags.NO_HINTING).linearHoriAdvance
        / 65536.0
    )


def _one_text(group, spans, centred: bool):
    """One <text> for a math-text group: its characters in reading order, as style runs."""
    chars = []
    for span in spans:
        xs, ys, text = (
            span.get("x").split(),
            span.get("y").split(),
            (span.text or " ").replace("\xa0", " "),
        )
        if len(xs) != len(text):
            raise SvgRewriteError(
                f"a math-text span with {len(xs)} positions for {text!r}"
            )
        ys = ys * len(xs) if len(ys) == 1 else ys
        look = (
            float(span.get("font-size")),
            span.get("font-style", "normal"),
            span.get("font-weight", "400"),
        )
        chars += [(float(x), float(y), ch, *look) for x, y, ch in zip(xs, ys, text)]
    chars.sort(key=lambda c: c[0])
    family = spans[0].get("font-family")
    # the body size; a subscript or superscript is smaller
    size = max(c[3] for c in chars)
    baseline = collections.Counter(c[1] for c in chars if c[3] == size).most_common(1)[
        0
    ][0]
    runs: list[tuple[tuple, list[str]]] = []
    for _, y, char, char_size, style, weight in chars:
        look = (char_size, style, weight, round(baseline - y, 3))
        if runs and runs[-1][0] == look:
            runs[-1][1].append(char)
        else:
            runs.append((look, [char]))
    x, last = chars[0][0], chars[-1]
    text = ET.Element(
        _tag("text"), {k: v for k, v in group.attrib.items() if k != "transform"}
    )
    if centred:
        x += 0.5 * (last[0] + _advance(last[2], last[3], last[4], last[5], family) - x)
        text.set("text-anchor", "middle")
    # the group's origin is the true baseline
    y, transform = 0.0, group.get("transform", "")
    moved = PURE_TRANSLATE.match(transform)
    if moved:
        x, y, transform = x + float(moved[1]), y + float(moved[2]), ""
    text.set("x", _fmt(x))
    text.set("y", _fmt(y))
    if transform:
        text.set("transform", transform)
    text.set("font-family", family)
    text.set("font-size", _fmt(size))
    for (run_size, style, weight, shift), letters in runs:
        span = ET.SubElement(text, _tag("tspan"))
        span.text = "".join(letters)
        if run_size != size:
            span.set("font-size", _fmt(run_size))
        if style != "normal":
            span.set("font-style", style)
        if weight != "400":
            span.set("font-weight", weight)
        if shift:
            span.set("baseline-shift", _fmt(shift))
    return text


def _rebuild_math_text(root, centred: set[str]) -> int:
    found = []
    for parent in root.iter():
        kids = list(parent)
        for index, child in enumerate(kids):
            if (
                child.tag != _tag("g")
                or len(child) != 1
                or child[0].tag != _tag("text")
            ):
                continue
            spans = list(child[0])
            if not spans or any(
                s.tag != _tag("tspan") or "x" not in s.attrib for s in spans
            ):
                continue
            before = kids[index - 1] if index else None
            source = (
                html.unescape(before.text.strip())
                if before is not None and before.tag is ET.Comment
                else None
            )
            found.append((parent, index, child, spans, source))
    for parent, index, group, spans, source in found:
        text = _one_text(group, spans, centred=source in centred)
        text.tail = group.tail
        parent[index] = text
    return len(found)


def _tidy_text(root) -> None:
    for el in root.iter(_tag("text")):
        el.set(f"{{{XML}}}space", "preserve")
        if NO_ROTATION.match(el.get("transform", "")):
            del el.attrib["transform"]


def _unwrap_single_groups(root) -> None:
    """``<g id="name"><path/></g>`` becomes ``<path id="name"/>``; an empty group goes."""
    changed = True
    while changed:
        changed = False
        for parent in list(root.iter()):
            for index, child in enumerate(list(parent)):
                if (
                    child.tag != _tag("g")
                    or set(child.attrib) - {"id"}
                    or len(child) != 1
                ):
                    continue
                only, name = child[0], child.get("id")
                if only.tag == _tag("text") and name and AUTO_TEXT_ID.match(name):
                    name = None
                if name:
                    only.set("id", name)
                elif only.tag == _tag("text"):
                    only.attrib.pop("id", None)
                only.tail = child.tail
                parent[index] = only
                changed = True
        changed |= bool(_remove(root, lambda el: el.tag == _tag("g") and len(el) == 0))


def clean(source: Path, dest: Path, centred: set[str] = frozenset()) -> dict:
    """Write ``dest``, the Illustrator-friendly rewrite of matplotlib's ``source``.

    ``centred`` is the set of math-text strings drawn centred on their anchor, which stay
    centred when edited. Returns counts of what was rewritten.

    Raises:
        SvgRewriteError: the file holds a reference (a pattern, a gradient) this rewrite
            cannot inline, or a text it cannot rebuild.
    """
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", XLINK)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source, parser=parser)
    root = tree.getroot()
    _styles_to_attributes(root)
    _write_out_stylesheet(root)
    _simplify_opacity(root)
    counts = {"markers": _inline_uses(root)}
    counts["clips"] = sum(1 for el in root.iter() if el.attrib.pop("clip-path", None))
    counts["math_texts"] = _rebuild_math_text(root, set(centred))
    _remove(
        root,
        lambda el: el.tag is ET.Comment or el.tag in (_tag("defs"), _tag("metadata")),
    )
    _tidy_text(root)
    _unwrap_single_groups(root)
    left = [el.tag for el in root.iter() if el.tag in (_tag("use"), _tag("clipPath"))]
    refs = [v for el in root.iter() for v in el.attrib.values() if "url(#" in v]
    if left or refs:
        raise SvgRewriteError(
            f"the rewrite left references it removed the targets of: {left + refs}"
        )
    counts["texts"] = sum(1 for _ in root.iter(_tag("text")))
    counts["paths"] = sum(1 for _ in root.iter(_tag("path")))
    counts["images"] = sum(1 for _ in root.iter(_tag("image")))
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    return counts
