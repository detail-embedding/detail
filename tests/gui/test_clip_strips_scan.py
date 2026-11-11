"""Guard: the clip window's strips name no curve.

**The class.** A shared surface summarising a module-shaped fact by hardcoding one module's
shape (``docs/recorded_classes.md``). The port this replaced kept ``_SHORT`` and ``_SIGNED``
tables of curve names inside the strips widget, so a curve added to ``api.clip_curves`` was
invisible until somebody edited the widget — and the two tables already disagreed. The rule:
a row is a ``Curve``, whose name, unit and range rule travel with it, and ``clip_strips.py``
spells none of their names.

The scan is a lower bound, as every source scan is: it looks for the names the API produces
today, quoted. A new curve is covered the day it is added to the list below, and
``test_the_names_are_the_ones_the_api_produces`` fails when a listed name stops being one.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
STRIPS = ROOT / "detail" / "gui" / "clip_strips.py"
#: Where a curve's name is spelled once, as data the strips are fed.
API = (
    ROOT / "detail" / "api" / "clip_curves.py",
    ROOT / "detail" / "analysis" / "body_axis.py",
)
#: Every curve name ``read_clip_curves`` can produce.
NAMES = ("speed", "area", "length", "bend", "turning")


def _violations(source: str) -> list[str]:
    """The curve names ``source`` spells as a string literal."""
    return [name for name in NAMES if re.search(rf"""["']{name}["']""", source)]


def test_the_names_are_the_ones_the_api_produces():
    api = "\n".join(path.read_text(encoding="utf-8-sig") for path in API)
    missing = [name for name in NAMES if not re.search(rf"""["']{name}["']""", api)]
    assert missing == [], f"not a curve the API produces any more: {missing}"


def test_the_strips_spell_no_curve_name():
    assert _violations(STRIPS.read_text(encoding="utf-8-sig")) == []


def test_the_scan_would_catch_a_hand_listed_name():
    assert _violations('_SHORT = {"speed": "spd", "area": "A"}') == ["speed", "area"]
    assert _violations("_SIGNED = frozenset({'bend'})") == ["bend"]
    assert _violations("# speed is a word in a comment") == []
