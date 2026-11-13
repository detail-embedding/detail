"""A user guide's numbered screenshot and its legend must count the same.

The class: a legend written in step with its image's markers until one line describes two
of them (``2. **Zoom out**, 3. **Zoom in** and 4. **Fit**``). Markdown numbers list items,
not the numbers written in them, so that line renders as 2 and every line after it sits one
marker early: the Scene geometry editor showed 27 markers over 24 descriptions, each wrong
from **Zoom in** on, and nothing about the page looked broken.

``scripts/gen_docs_screens.py`` records how many markers it drew on each image in
``docs/assets/user-guides/callouts.json``. This checks every user-guide image listed there
against the legend box that follows it on its page: one list item per marker, numbered
1 to N in order.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "docs" / "user-guides"
CALLOUTS = ROOT / "docs" / "assets" / "user-guides" / "callouts.json"

_IMAGE = re.compile(r"\]\((?:\.\./)+assets/user-guides/(?P<path>[^)]+\.png)\)")
_LEGEND = re.compile(r'^\?\?\? info "What ')
_ITEM = re.compile(r"^    (\d+)\. ")


def legend_numbers(lines: list[str], image_line: int) -> list[int] | None:
    """The list numbers of the legend box after ``image_line``, or None if it has none.

    The box is the first ``??? info "What …"`` before the next image or heading; its items
    are the four-space-indented numbered lines inside it.
    """
    for i in range(image_line + 1, len(lines)):
        line = lines[i]
        if _IMAGE.search(line) or line.startswith("#"):
            return None
        if _LEGEND.match(line):
            numbers = []
            for body in lines[i + 1 :]:
                if body and not body.startswith(" "):
                    break
                match = _ITEM.match(body)
                if match:
                    numbers.append(int(match.group(1)))
            return numbers
    return None


def _legends() -> dict[str, list[int] | None]:
    found = {}
    for page in sorted(GUIDES.rglob("*.md")):
        lines = page.read_text(encoding="utf-8-sig").splitlines()
        for n, line in enumerate(lines):
            match = _IMAGE.search(line)
            if match:
                found[match.group("path")] = legend_numbers(lines, n)
    return found


def _counts() -> dict[str, int]:
    return json.loads(CALLOUTS.read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("image", sorted(_counts()))
def test_every_marker_has_its_own_legend_line(image):
    legends = _legends()
    assert image in legends, f"{image} has markers but no page shows it"
    numbers = legends[image]
    assert numbers is not None, f"{image} has markers but no legend box follows it"
    expected = list(range(1, _counts()[image] + 1))
    assert numbers == expected, (
        f"{image}: {len(expected)} markers but the legend numbers {numbers}. Give each "
        "marker its own list item, in order: a line that names two markers shifts every "
        "line after it."
    )


def test_the_check_catches_a_line_that_names_two_markers():
    combined = [
        "![A dialog.](../assets/user-guides/x.png)",
        "",
        '??? info "What everything on this screen does"',
        "    1. **Name**: what it is called.",
        "    2. **Zoom out** and 3. **Zoom in**: zoom the frame.",
        "    4. **Close**: closes it.",
    ]
    assert legend_numbers(combined, 0) == [1, 2, 4]
    assert legend_numbers(combined, 0) != [1, 2, 3, 4]
