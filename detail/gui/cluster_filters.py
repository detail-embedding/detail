"""The cluster page's filters: a column docked beside the tree.

One selection, set from the dock. A selection never *hides* anything — the page shades the tree by
each cluster's matching share and draws non-matching clips grey — so this module only says *what*
is selected:

- **The dock** (:class:`FilterDock`, ``bar.dock``), top to bottom:

  - **The key and the count line, always** (``dock.top_box``), fixed above the sections so they
    stay in view while the sections scroll, and the same height with or without a filter, so
    nothing below moves when the first filter goes on (user, review round 8, option B). With no
    filter the key (`gui/match_key.MatchKey`) is greyed with no marks and the line says the
    run's clip count ("2,781 clips"); with one, the key carries its marks and the line says
    "n of N clips match" (which the page computes and hands back,
    :meth:`FilterBar.set_match_count`; how fields combine is its tooltip) with Reset all at its
    right. No pills: each field in use already says its filter on its own line.
  - **Location** — the run's position density as a region picker (`gui/density_map.py`): draw a
    box, move it, resize it from its corners, or click squares (the map's tooltip says so).
    It is a line that folds like a field's (laptop layout, 2026-11-09, decision 2 C): "▸ Location"
    with "any" or "1 region chosen" at its right and, while a region is picked, its Reset; open
    or folded is remembered (:data:`LOCATION_OPEN_KEY`). The map fills the dock's width, between
    :data:`MAP_MIN_PX` and :data:`MAP_MAX_PX`, so a narrower dock is a smaller map rather than a
    map cut off.
  - **Time in session** — a line folding like Location's, "▸ Time in session" with "any" or
    "12%–40%" at its right and its Reset while a range is set; open, a plain two-handle slider
    over each recording's whole video (:class:`SessionRange`: the :class:`RangeHistogram` with
    ``bars=False``), "start · middle · end" under it and a line with the range and the clips it
    holds. Open or folded is remembered (:data:`SESSION_OPEN_KEY`), as Location's is.
  - **Differences** — a line folding like Location's: the run's saved difference screens
    (§22, §24 ``source: nodes``), one tickable at a time, each with what it kept ("drug vs
    vehicle · paired   1 kept of 4"), and **New screen…**, which the page answers with the
    comparison launch dialog seeded to this run. Ticking one is the :data:`CLUSTERS` condition:
    the leaves under the clusters it kept, handed in by the page (:class:`ScreenChoice`), so the
    tree shades a kept cluster 100%, its parent by the time under kept children, the rest grey.
    Under the list, while a screen is ticked, **Strictness**: a slider over the family-wise
    levels a screen can be read at (``compare.STRICTNESS_LEVELS``, the screen's own 5% floor
    among them), with the kept set at each level handed in by the page (:class:`ScreenLevel`).
    Looser shows more clusters, each with its p on hover; it is a view, and what the saved
    screen kept does not move (``Selection.alpha``).
  - **One line per field** (:class:`FieldRow`) — Video, then each metadata column: "▸ name",
    and at the right its value count or the filter set on it ("drug_A only", "2 of 3"),
    then, while it has a filter, a small Reset that removes it. The line is not a button: a
    click anywhere on it, or Space / Enter when it has the focus, opens its values under it;
    the pointer over it tints it. Open or shut lasts the page's session and is not
    remembered. Fields with a filter on open and sort to the top under "In use", then a
    divider, then the rest in their usual order (a filter set *inside* a field re-sorts when
    the pointer leaves it, so the line does not move between two clicks). Over
    :data:`FIND_FIELD_OVER` fields, "Find a field (N)" above the list filters the lines by
    name. A field's values are a :class:`Checklist` for categories, and for a number, date or
    time-of-day column with at most :data:`MAX_TICK_VALUES` distinct values (count per value,
    a "(no value)" entry, an "only" on the hovered row, type-ahead over
    :data:`SEARCH_OVER` values); a :class:`RangeHistogram` — two handles dragged along the bars
    of clips per value — for a number or date column with more.
  - **A column with a single value** is a greyed line at the bottom (:class:`ConstantRow`),
    "animal  F3", its tooltip saying there is nothing to filter. A column
    with one value *and* blanks keeps its checklist, because "(no value)" is a real choice.
    A list of 18 fields, every one open, was about 1,400 px of scrolling (user, review
    round 5).

- **Folded** — dragged narrower than :data:`DOCK_MIN_PX`, or its grip double-clicked — the dock
  is a strip reading "Filters", or "Filters · 2 active" while two filters are on
  (:func:`panel_name`), so an active filter is never out of sight; Reset all stays inside the
  open dock. The fold is the splitter's (`gui/grip_splitter.py`), the same on every side panel.

- **The controller** (:class:`FilterBar`): not a widget. It owns the selection and builds
  ``bar.dock``, the one thing a page lays out; the count line and Reset all are in the dock.

**OR within a column, AND across columns** (and video and location are a column each). Every
change emits at once — a drag while it moves, at most every 50 ms (`gui/throttle.py`) — so the
tree shades under the hand.

The widget reads nothing: columns, values and counts are handed in (:meth:`FilterBar.set_columns`,
:meth:`FilterBar.set_recordings`, :meth:`FilterBar.set_density`) and it emits a plain
:class:`Selection`.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import datetime as dt

from PyQt6.QtCore import QEvent, QObject, QRectF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from detail.analysis.compare import FLOOR_ALPHA, STRICTNESS_LEVELS
from detail.api.cluster_filter import CLUSTERS, LOCATION, SESSION, VIDEO, Selection
from detail.gui import plot_theme
from detail.gui.density_map import RegionPicker
from detail.gui.match_key import MatchKey, effect_extent, effect_position, one_sided
from detail.gui.selectable import make_text_selectable
from detail.gui.sizing import (
    ElidedLabel,
    emphasis_font,
    fit_list_height,
    lines,
    set_available,
    small_font,
)
from detail.gui.throttle import Throttle
from detail.storage import metadata as metadata_store

CATEGORY = "category"
NUMBER = "number"
DATE = "date"
#: A clock time (``14:00:21``): sorted and ranged by seconds after midnight, like a number.
TIME = "time"
KINDS = (CATEGORY, NUMBER, DATE, TIME)
#: The value a recording with an empty cell has. Listed as "(no value)", never dropped.
NO_VALUE = ""
NO_VALUE_TEXT = "(no value)"
#: A checklist longer than this gets a type-ahead field.
SEARCH_OVER = 20
#: A number or date column with at most this many distinct values gets tick boxes, not a range:
#: on an_32 a range over three values drew one or two full-width bars and said nothing.
MAX_TICK_VALUES = 12
#: A histogram draws one bar per distinct value up to this many, then equal-width bins.
_MAX_BARS = 40
#: More fields than this (Video, the columns, and the one-value ones) and the list gets a
#: "Find a field" box above it.
FIND_FIELD_OVER = 8
#: The caption over the fields with a filter on, which sort to the top of the list.
IN_USE_TEXT = "In use"
OTHER_FIELDS_TEXT = "Other fields"
#: How the fields combine: the tooltip of the "n of N clips match" line.
MATCH_RULE_TEXT = "A clip matches when it fits every field. Several ticked values in one field: any of them."
#: Condition keys (the API's). A metadata column's key is ``("meta", column)``.
CONDITION_KEYS = (VIDEO, LOCATION, SESSION, CLUSTERS)
#: What the Differences line says with no screen saved, while reading them, and when ticked.
NO_SCREENS_TEXT = "No difference screen saved for this run yet."
SCREENS_LOADING_TEXT = "Reading the saved screens…"
SCREENS_HINT_TEXT = (
    "One at a time: the tree shades by the clusters the ticked screen kept, a kept cluster "
    "100%, its parent by the time under kept children."
)
RUN_SCREEN_TEXT = "New screen…"
#: The Strictness slider under the screens: its caption and what its tooltip says.
STRICTNESS_TEXT = "Strictness"
STRICTNESS_RULE_TEXT = (
    "How sure a cluster must be to show: the share of shuffled cohorts whose largest effect "
    "reaches it (its family-wise p). The saved screen keeps at p ≤ 0.05; slide right to see "
    "clusters that would need more animals to clear that, each with its p on hover. This "
    "changes the view, not the saved screen."
)
#: The region picker's width before the dock has been laid out; from then on it fills the
#: dock's width (:meth:`FilterBar._fit_map`), between these two.
MAP_PX = 320
MAP_MIN_PX = 160
MAP_MAX_PX = 480
#: The narrowest the open dock drags to (laptop layout, 2026-11-09, decision 2 C). It was the map's
#: 320 plus a margin, 350 of a 1136px content width on a laptop.
DOCK_MIN_PX = 240
#: How much of a field's summary ("50%–100%", "2 recordings") survives before it elides, in
#: characters: more than the name beside it keeps, and a constant, so no value widens the line.
SUMMARY_MIN_CHARS = 9
#: Whether the Location line is open, kept between pages and sessions.
LOCATION_OPEN_KEY = "interpret/filters_location_open"
#: Whether the Time in session line is open, kept like Location's.
SESSION_OPEN_KEY = "interpret/filters_session_open"
#: The Time in session slider's steps over a recording: one per percent.
SESSION_STEPS = 100
#: How Time in session reads a recording and an edge clip: its range line's tooltip.
SESSION_RULE_TEXT = (
    "Start is each recording's first frame and end its last, so a range is the same part of "
    "every recording whatever its length. A cluster's shading counts only the time inside the "
    "range; a clip matches when its middle is inside."
)


# --------------------------------------------------------------------------- #
# plain data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Column:
    """One metadata column: its name, its type, and the clips per value (``""`` = no value)."""

    name: str
    kind: str
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"column kind must be one of {KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class NodeEffect:
    """One cluster's row of a screen: its effect (``None`` where undefined), whether it clears
    the floor, whether it was kept after nesting, the screen's sentence for it, and ``pair``,
    the two levels the effect is between (``(a, b)``, *b against a*) — what a profile's key
    names at the selected cluster's mark, since its one-sided axis cannot (§22). ``family_p``,
    ``own_p`` and ``q`` are the item's numbers (``compare.Effect``), for the hover and for a
    label's origin; ``None`` for an undefined effect, or a row built without them."""

    effect: float | None
    clears: bool
    kept: bool
    text: str
    pair: tuple[str, str] | None = None
    family_p: float | None = None
    own_p: float | None = None
    q: float | None = None


@dataclass(frozen=True)
class ScreenLevel:
    """A screen read at one strictness (§22): the family-wise level ``alpha``, the floor at
    it in the effect's unit (``None`` where the screen has no floor), the clusters kept there
    in rank order, their leaves — the :data:`CLUSTERS` condition at that level — and how many
    clusters clear that floor before nesting is suppressed."""

    alpha: float
    floor: float | None
    kept: tuple[int, ...]
    leaves: frozenset
    clearing: int = 0


@dataclass(frozen=True)
class ScreenChoice:
    """A saved difference screen the Differences line offers.

    ``key`` is the comparison id, ``label`` how it reads ("drug vs vehicle · paired"),
    ``note`` what it kept ("1 kept of 4 clearing") or why it cannot be ticked, and ``leaves`` the
    leaves under its kept clusters — ``None`` while the page is still reading it, or when it
    cannot be read, which greys the row with the note as its tooltip. ``effects`` is every
    candidate cluster's row, ``floor`` the screen's floor in the effect's unit and ``design``
    its design, for the effect mode (:func:`effect_shades`); ``caption`` is the axis's words
    ("Less ↔ more under drug"). ``kept`` is the clusters it kept, in rank order, and ``column``
    the metadata column it contrasted: what a label written while the screen is ticked records
    as its origin (``run_labels.LabelOrigin``, §22).

    ``levels`` is the screen read at each strictness the slider offers
    (:class:`ScreenLevel`); :meth:`level` finds one, and the screen's own fields stand in for
    its own floor when none were handed in.
    """

    key: str
    label: str
    note: str
    leaves: frozenset | None
    effects: Mapping[int, NodeEffect] = field(default_factory=dict)
    floor: float | None = None
    design: str = ""
    caption: str = ""
    kept: tuple[int, ...] = ()
    column: str = ""
    levels: tuple[ScreenLevel, ...] = ()

    def level(self, alpha: float | None) -> ScreenLevel | None:
        """The screen at the strictness ``alpha`` (``None``: its own floor), or ``None`` when
        it was not read at that level."""
        wanted = FLOOR_ALPHA if alpha is None else float(alpha)
        for level in self.levels:
            if abs(level.alpha - wanted) < 1e-9:
                return level
        if abs(wanted - FLOOR_ALPHA) < 1e-9:
            return ScreenLevel(
                FLOOR_ALPHA,
                self.floor,
                tuple(self.kept),
                self.leaves if self.leaves is not None else frozenset(),
                sum(1 for row in self.effects.values() if row.clears),
            )
        return None

    def clears(self, node: int, alpha: float | None) -> bool:
        """Whether ``node`` clears the floor at ``alpha``: the row's own verdict at the
        screen's own floor; at another level its family-wise p against the level, as the API
        reads it (``Ranking.clears_at``), or its size against that level's floor for a row
        built without a p."""
        row = self.effects.get(int(node))
        if row is None or row.effect is None or row.effect != row.effect:
            return False
        if alpha is None or abs(float(alpha) - FLOOR_ALPHA) < 1e-9:
            return row.clears
        if row.family_p is not None:
            return row.family_p <= float(alpha)
        level = self.level(alpha)
        if level is None or level.floor is None:
            return False
        return abs(float(row.effect)) > level.floor

    def kept_texts(self, alpha: float | None = None) -> dict[int, str]:
        """``{kept cluster: its sentence}`` at the strictness ``alpha`` -- what an origin is
        built from. Empty when the screen was not read at that level."""
        level = self.level(alpha)
        return {
            int(node): (self.effects[node].text if node in self.effects else "")
            for node in (level.kept if level is not None else ())
        }

    def kept_numbers(self, alpha: float | None = None) -> dict[int, dict]:
        """``{kept cluster: {"family_p", "own_p", "q"}}`` at ``alpha``, for the origin."""
        return {
            node: {
                "family_p": row.family_p,
                "own_p": row.own_p,
                "q": row.q,
            }
            for node in self.kept_texts(alpha)
            if (row := self.effects.get(node)) is not None
        }


def numbers_text(family_p: float | None, own_p: float | None, q: float | None) -> str:
    """``family-wise p 0.02, own p <0.01, q 0.40`` — the numbers a hover and a label's origin
    line carry, in one wording; ``""`` when there are none."""
    from detail.analysis.comparison_report import p_text

    parts = []
    if family_p is not None:
        parts.append(f"family-wise p {p_text(family_p)}")
    if own_p is not None:
        parts.append(f"own p {p_text(own_p)}")
    if q is not None:
        parts.append(f"q {p_text(q)}")
    return ", ".join(parts)


def effect_shades(
    choice: ScreenChoice, n_nodes: int, alpha: float | None = None
) -> tuple[dict[int, float], dict[int, str]]:
    """``(positions, texts)`` for the tree in the effect mode: every node's position on the
    −1..+1 axis (NaN where it is not a candidate or its effect is undefined, 0 inside the floor)
    and the hover line for it. The one map from a screen to the blocks, through
    :func:`match_key.effect_position`, which the key uses for its own mark. At the strictness
    ``alpha`` the floor and the kept set are that level's (:meth:`ScreenChoice.level`).
    """
    extent = effect_extent(choice.design)
    half = one_sided(choice.design)
    at_floor = alpha is None or abs(alpha - FLOOR_ALPHA) < 1e-9
    level = None if at_floor else choice.level(alpha)
    kept = set(level.kept) if level is not None else set()
    at = "" if at_floor else f" at p ≤ {alpha:g}"
    positions: dict[int, float] = {}
    texts: dict[int, str] = {}
    for node in range(int(n_nodes)):
        row = choice.effects.get(node)
        if row is None:
            positions[node] = float("nan")
            continue
        clears = choice.clears(node, alpha)
        positions[node] = effect_position(row.effect, clears, extent, one_sided=half)
        text = row.text
        # At the screen's own floor the row's verdict is the screen's; at another level the
        # kept set is that level's.
        if row.kept if at_floor else node in kept:
            text += f" · kept{at}"
        elif clears:
            text += f" · clears the floor{at}, nested under a kept cluster"
        elif row.effect is not None:
            text += f" · inside the floor{at}"
        numbers = numbers_text(row.family_p, row.own_p, row.q)
        if numbers and row.effect is not None:
            text += f" · {numbers}"
        texts[node] = text
    return positions, texts


@dataclass(frozen=True)
class RecordingChoice:
    """A recording the Video section lists: its id (emitted), its name (shown), its clips."""

    recording_id: str
    name: str
    count: int


def _as_date(text: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(text).strip()[:10])
    except ValueError:
        return None


def _as_number(text: str) -> float | None:
    try:
        return float(str(text).strip())
    except ValueError:
        return None


def _parse(column: Column, text: str):
    """``text`` as the column's type (a number, a date), or ``None`` when it does not parse or
    the column is a category."""
    if column.kind == NUMBER:
        return _as_number(text)
    if column.kind == DATE:
        return _as_date(text)
    if column.kind == TIME:
        return metadata_store.parse_time(text)
    return None


def distinct_values(column: Column) -> set:
    """The column's distinct values, blanks not counted, **as parsed**: "1" and "1.0" are one
    number. A value that does not parse counts as itself."""
    out = set()
    for text in column.counts:
        if text == NO_VALUE:
            continue
        parsed = _parse(column, text)
        out.add(text if parsed is None else parsed)
    return out


def single_value(column: Column) -> str | None:
    """The one value every clip has (``""`` when every cell is empty), or ``None`` when there
    is a choice to make. Values no clip has do not count; a blank beside a value does.
    """
    present = {k for k, c in column.counts.items() if c > 0} or set(column.counts)
    if NO_VALUE in present:
        return NO_VALUE if present == {NO_VALUE} else None
    probe = Column(column.name, column.kind, {k: 1 for k in present})
    if len(distinct_values(probe)) != 1:
        return None
    return sorted(present)[0]


def section_kind(column: Column) -> str:
    """Which widget the column's section is: ``"same"`` (onto the one line), ``"checklist"``
    or ``"range"``."""
    if single_value(column) is not None:
        return "same"
    if column.kind == CATEGORY or len(distinct_values(column)) <= MAX_TICK_VALUES:
        return "checklist"
    return "range"


def condition(selection: Selection, column: str):
    """Column ``column``'s condition: its picked values, its ``(lo, hi)`` range, or ``None``."""
    if column in selection.values:
        return selection.values[column]
    return selection.ranges.get(column)


def _is_values(value) -> bool:
    """A condition's shape decides what it is: a set is values, a tuple is a range. The column's
    type does not — a number column with a few values is filtered by ticking them."""
    return isinstance(value, (set, frozenset))


def with_condition(selection: Selection, column: Column | str, value) -> Selection:
    """``selection`` with ``column``'s condition set to ``value`` (``None`` removes it): a
    ``frozenset`` of values, or a ``(lo, hi)`` range."""
    name = column if isinstance(column, str) else column.name
    values = {k: v for k, v in selection.values.items() if k != name}
    ranges = {k: v for k, v in selection.ranges.items() if k != name}
    if value is not None:
        if _is_values(value):
            values[name] = frozenset(value)
        elif isinstance(value, tuple) and len(value) == 2:
            ranges[name] = tuple(value)
        else:
            raise TypeError(
                f"a condition is a set of values or a (lo, hi) tuple, got {value!r}"
            )
    return replace(selection, values=values, ranges=ranges)


def breakdown(
    selection: Selection, passed: Mapping[tuple, bool], meta: Mapping[str, str]
) -> str:
    """``✓ animal_id A03  ✗ location`` — each active filter a clip meets or fails.

    ``passed[key]`` is whether the clip meets condition ``key`` alone, which the page asks of the
    API's own rule (``cluster_filter.clip_matches(selection.only(key), …)``), so the hover text
    and the grey tile can never disagree. ``meta`` is the clip's recording's metadata row.
    """
    parts = []
    for key in selection.keys():
        mark = "✓" if passed.get(key) else "✗"
        if key == VIDEO:
            parts.append(f"{mark} video")
        elif key == LOCATION:
            parts.append(f"{mark} location")
        elif key == SESSION:
            parts.append(f"{mark} time in session")
        elif key == CLUSTERS:
            parts.append(f"{mark} differs")
        else:
            value = (meta.get(key[1]) or NO_VALUE).strip() or NO_VALUE_TEXT
            parts.append(f"{mark} {key[1]} {value}")
    return "  ".join(parts)


def screen_summary(selection: Selection, screens: Mapping[str, ScreenChoice]) -> str:
    """What the Differences line says at its right: ``none``, or the ticked screen's label,
    with the strictness after it when the screen is read away from its own floor."""
    if selection.leaves is None:
        return "none"
    source = selection.leaves_source or ""
    choice = screens.get(source)
    text = choice.label if choice is not None else source or "a screen"
    if selection.alpha is not None and abs(selection.alpha - FLOOR_ALPHA) >= 1e-9:
        text += f" · p ≤ {selection.alpha:g}"
    return text


def _number_text(value: float) -> str:
    return f"{value:g}"


def _date_text(text: str) -> str:
    day = _as_date(text)
    return f"{day:%b} {day.day}" if day else str(text)


def field_summary(
    selection: Selection,
    key: tuple,
    columns: Mapping[str, Column],
    recording_names: Mapping[str, str],
) -> str:
    """What a field's row says at its right: ``"3 values"`` with no filter on it, or the filter —
    ``"drug_A only"``, ``"2 of 3"``, ``"none"``, ``"25–30"``."""
    if key == VIDEO:
        picked = selection.recording_ids
        total = len(recording_names)
        if picked is None:
            return f"{total} recording{'s' if total != 1 else ''}"
        named = sorted(recording_names.get(r, r) for r in picked)
    else:
        name = key[1]
        value = condition(selection, name)
        total = len(columns[name].counts) if name in columns else 0
        if value is None:
            return f"{total} value{'s' if total != 1 else ''}"
        if not _is_values(value):
            if isinstance(value[0], str):
                return f"{_date_text(value[0])} – {_date_text(value[1])}"
            return f"{_number_text(value[0])}–{_number_text(value[1])}"
        named = sorted((v or NO_VALUE_TEXT) for v in value)
    if not named:
        return "none"
    if len(named) == 1:
        return f"{named[0]} only"
    return f"{len(named)} of {total}"


def constant_tooltip(name: str, value: str, actions: int) -> str:
    """Why a one-value field cannot be opened."""
    return (
        f"All {actions:,} clips have {name} {value or NO_VALUE_TEXT}, so there is nothing "
        "to filter."
    )


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def location_summary(cells) -> str:
    """What the Location line says at its right: ``"any"``, or how many separate regions the
    picked squares make ("1 region chosen") — squares touching side by side are one region.
    """
    cells = set(cells or ())
    if not cells:
        return "any"
    regions = 0
    while cells:
        regions += 1
        stack = [cells.pop()]
        while stack:
            x, y = stack.pop()
            for near in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if near in cells:
                    cells.remove(near)
                    stack.append(near)
    return f"{regions} region{'s' if regions != 1 else ''} chosen"


def _heading(text: str, detail: str = "") -> QLabel:
    """A section's heading: its name, and a short ``detail`` (a count) when one helps. Never
    the column's type: "date · date" and "time · categories" read as noise."""
    label = QLabel(f"{text}  · {detail}" if detail else text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setFont(emphasis_font(label, steps=0))
    label.setWordWrap(True)
    return label


class Checklist(QWidget):
    """Checkable entries with counts, an "only" on the row under the pointer, and a
    type-ahead when the list is long.

    ``changed`` carries the checked keys as a ``frozenset``, or ``None`` when every entry is
    checked — "all of them" is no condition.

    **No All / None.** "None" made a filter nothing matches, which is never what a filter is
    for; "All" was the way back. The way back is the field's Reset, in its line
    (:class:`FieldRow`), shown only while there is something to reset. To narrow to one value,
    **only** sits on the row the
    pointer is over — or, from the keyboard, the current row: Tab from the list reaches it.
    One button moved between rows rather than one per row, so a list of 200 recordings builds
    one widget, not 200.
    """

    changed = pyqtSignal(object)

    def __init__(
        self,
        entries: Sequence[tuple[str, str, int]],
        *,
        noun: str,
        searchable: bool | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._keys = [key for key, _text, _count in entries]
        self._texts = {key: text for key, text, _count in entries}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        long_list = len(entries) > SEARCH_OVER
        if searchable is None:
            searchable = long_list
        self.search: QLineEdit | None = None
        if searchable:
            self.search = QLineEdit()
            self.search.setPlaceholderText(f"Search {noun}…")
            self.search.setObjectName("filter_search")
            self.search.setClearButtonEnabled(True)
            self.search.textChanged.connect(self._filter)
            layout.addWidget(self.search)
        self.list = QListWidget()
        for key, text, count in entries:
            item = QListWidgetItem(f"{text}   {count:,}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(f"{text}: {count:,} clips")
            self.list.addItem(item)
        self.list.itemChanged.connect(self._on_item)
        fit_list_height(self.list, max_rows=8)
        layout.addWidget(self.list)
        if searchable and long_list:
            hint = QLabel(f"{len(entries)} values, so this list can be searched.")
            hint.setWordWrap(True)
            layout.addWidget(hint)
        #: Ticks just the row it sits on; moved to the hovered row, or the current one.
        self.only_button = QPushButton("only", self.list.viewport())
        self.only_button.setVisible(False)
        self.only_button.clicked.connect(self._only_clicked)
        self._only_key: str | None = None
        self.list.setMouseTracking(True)
        self.list.itemEntered.connect(self._show_only)
        self.list.currentItemChanged.connect(
            lambda item, _old: self._show_only(item) if item is not None else None
        )
        self.list.verticalScrollBar().valueChanged.connect(self._hide_only)
        self.list.viewport().installEventFilter(self)
        self._quiet = False

    def value(self) -> frozenset | None:
        picked = frozenset(
            self.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        )
        return None if len(picked) == len(self._keys) else picked

    def set_value(self, picked: frozenset | None) -> None:
        """Check exactly ``picked`` (every entry for ``None``) without emitting."""
        self._quiet = True
        for i in range(self.list.count()):
            item = self.list.item(i)
            on = picked is None or item.data(Qt.ItemDataRole.UserRole) in picked
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._quiet = False

    def only(self, key: str) -> None:
        """Tick ``key`` alone and emit — what the row's **only** does."""
        self.set_value(frozenset({key}))
        self.changed.emit(self.value())

    def only_key(self) -> str | None:
        """The row **only** sits on now, or ``None`` while it is hidden."""
        return self._only_key if self.only_button.isVisible() else None

    def keys(self) -> list[str]:
        """Every entry's key, in the order listed."""
        return list(self._keys)

    def visible_keys(self) -> list[str]:
        return [
            self.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list.count())
            if not self.list.item(i).isHidden()
        ]

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().casefold())

    def _show_only(self, item) -> None:
        if item is None or item.isHidden():
            self._hide_only()
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        self._only_key = key
        self.only_button.setToolTip(f"Tick only {self._texts.get(key, key)}")
        rect = self.list.visualItemRect(item)
        hint = self.only_button.sizeHint()
        height = min(hint.height(), rect.height())
        self.only_button.setGeometry(
            rect.right() - hint.width() - 2,
            rect.top() + (rect.height() - height) // 2,
            hint.width(),
            height,
        )
        self.only_button.setVisible(True)
        self.only_button.raise_()

    def _hide_only(self, *_args) -> None:
        if not self.only_button.hasFocus():
            self.only_button.setVisible(False)

    def _only_clicked(self) -> None:
        if self._only_key is not None:
            self.only(self._only_key)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        # The pointer leaving the list takes "only" with it, unless it went onto the button —
        # or the keyboard is on the list, where "only" belongs to the current row.
        if event.type() == QEvent.Type.Leave and watched is self.list.viewport():
            if not self.only_button.underMouse():
                current = self.list.currentItem()
                if self.list.hasFocus() and current is not None:
                    self._show_only(current)
                else:
                    self._hide_only()
        return super().eventFilter(watched, event)

    def _on_item(self, _item) -> None:
        if not self._quiet:
            self.changed.emit(self.value())


def _bins(
    values: Sequence[float], counts: Sequence[int]
) -> tuple[list[int], list[int]]:
    """``(bar counts, bar index per value)``: a bar per value, or equal-width bins past 40."""
    if len(values) <= _MAX_BARS:
        return list(counts), list(range(len(values)))
    lo, hi = values[0], values[-1]
    step = (hi - lo) / _MAX_BARS or 1.0
    where = [min(int((v - lo) / step), _MAX_BARS - 1) for v in values]
    bars = [0] * _MAX_BARS
    for index, count in zip(where, counts):
        bars[index] += count
    return bars, where


class Histogram(QWidget):
    """Bars of clips per value (or per bin), the ones inside the picked span drawn in colour."""

    def __init__(self, counts: Sequence[int], parent=None) -> None:
        super().__init__(parent)
        self._counts = list(counts)
        self._span = (0, len(self._counts) - 1)
        self.setFixedHeight(lines(2))
        self.setMinimumWidth(lines(10))

    def set_span(self, first: int, last: int) -> None:
        self._span = (first, last)
        self.update()

    def _bar_width(self) -> float:
        return self.width() / max(len(self._counts), 1)

    def _bars_height(self) -> float:
        return float(self.height())

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        if not self._counts:
            return
        theme = plot_theme.current()
        painter = QPainter(self)
        self._paint_bars(painter, theme)
        self._paint_over(painter, theme)
        painter.end()

    def _paint_bars(self, painter: QPainter, theme) -> None:
        peak = max(max(self._counts), 1)
        width = self._bar_width()
        tall = self._bars_height()
        inside = plot_theme.match_colour(0.7, theme)
        outside = QColor(theme.neutral)
        for i, count in enumerate(self._counts):
            height = 2 + (tall - 2) * count / peak
            colour = inside if self._span[0] <= i <= self._span[1] else outside
            painter.fillRect(
                QRectF(i * width + 1, tall - height, max(width - 2, 1), height),
                colour,
            )

    def _paint_over(self, painter: QPainter, theme) -> None:
        """What a subclass draws over the bars."""


class RangeHistogram(Histogram):
    """The bars, with a handle at each end of the picked span to drag along them.

    Replaces a from/to pair of sliders under the bars, which on a real run read as one or two
    full-width blue bars and told the user nothing: the handles *are* on the bars, so where the
    span starts and ends is where the hand is. A press moves the nearer handle to it; dragging
    carries it. ``span_changed`` carries the value indices ``(first, last)`` while dragging (at
    most every 50 ms) and once more on release.

    **From the keyboard**: Tab reaches the left handle, Tab again the right one (Shift+Tab back);
    Left / Right move the focused handle one value, with Shift ten. The focused handle is drawn in
    the palette's highlight colour.

    ``bars=False`` draws a plain track instead of the bars, one step per value with no binning:
    a plain two-handle range slider (Time in session, which has no count per moment to show). Its
    track is :data:`TRACK_PX` thick and each handle a knob :data:`KNOB_W_PX` wide and a line of
    text tall, set in from the ends by half a knob so neither is cut at 0% or 100% — a 3px track
    with 8px squares read as too small to grab (user, 2026-11-09).
    """

    span_changed = pyqtSignal(object)

    #: The grip's half-width at the bottom of a handle, in pixels.
    GRIP_PX = 4
    #: Values a Shift+arrow moves a handle.
    BIG_STEP = 10
    #: Without bars: the track's thickness and a knob's width, in pixels.
    TRACK_PX = 6
    KNOB_W_PX = 12

    def __init__(
        self,
        positions: Sequence[float],
        counts: Sequence[int],
        parent=None,
        *,
        bars: bool = True,
    ):
        self._bars = bool(bars)
        if self._bars:
            shown, self._bar_of = _bins(list(positions), list(counts))
        else:
            shown, self._bar_of = [0] * len(positions), list(range(len(positions)))
        super().__init__(shown, parent)
        self._n = len(self._bar_of)
        self._first, self._last = 0, max(self._n - 1, 0)
        self._grab: str | None = None
        #: The handle the arrow keys move.
        self._key_handle = "first"
        self._live = Throttle(self.span_changed.emit)
        self.setFixedHeight(lines(2.6) if self._bars else self._knob_height() + 6)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Drag a handle along the bars to narrow the range"
            if self._bars
            else "Drag a handle to narrow the range (or Tab to it and use the arrow keys)"
        )
        self._sync_bars()

    @staticmethod
    def _knob_height() -> int:
        """A knob's height without bars: a line of text."""
        return max(lines(1.1), 16)

    def _inset(self) -> float:
        """How far the track is set in from each end: half a knob without bars, else none."""
        return 0.0 if self._bars else self.KNOB_W_PX / 2

    def _bar_width(self) -> float:
        if self._bars:
            return super()._bar_width()
        return (self.width() - 2 * self._inset()) / max(len(self._counts), 1)

    def knob_rects(self) -> list[QRectF]:
        """Without bars, the two knobs' rectangles (first, last); with bars, none."""
        if self._bars:
            return []
        tall = self._knob_height()
        top = (self.height() - tall) / 2
        return [
            QRectF(x - self.KNOB_W_PX / 2, top, self.KNOB_W_PX, tall)
            for x in self.handle_x()
        ]

    def key_handle(self) -> str:
        """``"first"`` or ``"last"``: the handle the arrow keys move."""
        return self._key_handle

    def focusInEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.reason() == Qt.FocusReason.BacktabFocusReason:
            self._key_handle = "last"
        elif event.reason() == Qt.FocusReason.TabFocusReason:
            self._key_handle = "first"
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().focusOutEvent(event)
        self.update()

    def focusNextPrevChild(self, next_: bool) -> bool:  # noqa: N802 — Qt's name
        """Tab steps from the left handle to the right one before leaving, Shift+Tab back."""
        if next_ and self._key_handle == "first":
            self._key_handle = "last"
            self.update()
            return True
        if not next_ and self._key_handle == "last":
            self._key_handle = "first"
            self.update()
            return True
        return super().focusNextPrevChild(next_)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        key = event.key()
        if key not in (Qt.Key.Key_Left, Qt.Key.Key_Right) or not self._n:
            super().keyPressEvent(event)
            return
        step = (
            self.BIG_STEP
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            else 1
        )
        step = -step if key == Qt.Key.Key_Left else step
        first, last = self._first, self._last
        if self._key_handle == "first":
            first = min(max(first + step, 0), last)
        else:
            last = max(min(last + step, self._n - 1), first)
        if (first, last) != self.indices():
            self.set_indices(first, last)
            self.span_changed.emit(self.indices())
        event.accept()

    def indices(self) -> tuple[int, int]:
        """``(first, last)``: the value indices the span covers, inclusive."""
        return self._first, self._last

    def set_indices(self, first: int, last: int) -> None:
        """Show the span ``first..last`` without emitting."""
        last = min(max(last, 0), max(self._n - 1, 0))
        first = min(max(first, 0), last)
        self._first, self._last = first, last
        self._sync_bars()

    def _sync_bars(self) -> None:
        if self._n:
            self.set_span(self._bar_of[self._first], self._bar_of[self._last])

    def _bars_height(self) -> float:
        return float(self.height() - 2 * self.GRIP_PX)

    def _paint_bars(self, painter: QPainter, theme) -> None:
        if self._bars:
            super()._paint_bars(painter, theme)
            return
        # A plain track: the whole length in the neutral colour, the picked span over it in the
        # match colour.
        thick = self.TRACK_PX
        top = (self.height() - thick) / 2
        inset = self._inset()
        left, right = self.handle_x()
        painter.fillRect(
            QRectF(inset, top, self.width() - 2 * inset, thick), QColor(theme.neutral)
        )
        painter.fillRect(
            QRectF(left, top, max(right - left, 1), thick),
            plot_theme.match_colour(0.7, theme),
        )

    def handle_x(self) -> tuple[float, float]:
        """The two handles' x in pixels: the left edge of the first bar in the span and the
        right edge of the last."""
        width = self._bar_width()
        inset = self._inset()
        return inset + self._span[0] * width, inset + (self._span[1] + 1) * width

    def _paint_over(self, painter: QPainter, theme) -> None:
        tall = self._bars_height()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        focused = self.palette().color(QPalette.ColorRole.Highlight)
        if not self._bars:
            for which, rect in zip(("first", "last"), self.knob_rects()):
                keyed = self.hasFocus() and which == self._key_handle
                painter.setPen(QPen(QColor(theme.surface), 1))
                painter.setBrush(focused if keyed else QColor(theme.text))
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            return
        for which, x in zip(("first", "last"), self.handle_x()):
            x = min(max(x, 1.0), self.width() - 1.0)
            ink = (
                focused
                if self.hasFocus() and which == self._key_handle
                else QColor(theme.text)
            )
            painter.setPen(QPen(ink, 2))
            painter.drawLine(int(round(x)), 0, int(round(x)), int(tall))
            painter.setPen(QPen(QColor(theme.surface), 1))
            painter.setBrush(ink)
            painter.drawRect(
                QRectF(x - self.GRIP_PX, tall - 1, 2 * self.GRIP_PX, 2 * self.GRIP_PX)
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _bar_at(self, x: float) -> int:
        return min(max(int(x // self._bar_width()), 0), len(self._counts) - 1)

    def _moved_to(self, x: float) -> tuple[int, int]:
        """The span with the grabbed handle at ``x``, never past the other one.

        On the bars a handle takes the bar under the hand. Without them it goes to the nearest
        step boundary, so a handle let go at 40% of the track reads 40%."""
        if not self._bars:
            along = (x - self._inset()) / self._bar_width()
            edge = min(max(int(round(along)), 0), self._n)
            if self._grab == "first":
                return min(edge, self._last), self._last
            return self._first, max(edge - 1, self._first)
        bar = self._bar_at(x)
        if self._grab == "first":
            first = min(bisect.bisect_left(self._bar_of, bar), self._n - 1)
            return min(first, self._last), self._last
        last = max(bisect.bisect_right(self._bar_of, bar) - 1, 0)
        return self._first, max(last, self._first)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.button() != Qt.MouseButton.LeftButton or not self._n:
            return
        x = event.position().x()
        left, right = self.handle_x()
        self._grab = "first" if abs(x - left) <= abs(x - right) else "last"
        self._key_handle = self._grab
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._live.start(self.indices())
        self._drag_to(x)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._grab is not None:
            self._drag_to(event.position().x())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if self._grab is None:
            return
        self._drag_to(event.position().x(), live=False)
        self._grab = None
        self._live.flush(self.indices())

    def _drag_to(self, x: float, live: bool = True) -> None:
        span = self._moved_to(x)
        if span != self.indices():
            self.set_indices(*span)
        if live:
            self._live.offer(self.indices())


class _RangeSection(QWidget):
    """A range over a column's sorted distinct values: the bars with their handles, the picked
    from–to under them, and how many clips a range leaves out for having no value.

    ``changed`` carries ``(lo, hi)`` inclusive, or ``None`` for the full span.
    """

    changed = pyqtSignal(object)

    def __init__(
        self,
        keys: Sequence,
        counts: Sequence[int],
        positions: Sequence[float],
        text: Callable[[object], str],
        missing: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._keys = list(keys)
        self._text = text
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.histogram = RangeHistogram(positions, counts)
        self.histogram.span_changed.connect(self._on_span)
        layout.addWidget(self.histogram)
        self.span_label = ElidedLabel("")
        layout.addWidget(self.span_label)
        if missing:
            note = QLabel(f"{missing:,} clips have no value; a range leaves them out.")
            note.setWordWrap(True)
            layout.addWidget(note)
        self._sync()

    def _emitted(self, key):
        return key

    def value(self):
        if not self._keys:
            return None
        a, b = self.histogram.indices()
        if a == 0 and b == len(self._keys) - 1:
            return None
        return (self._emitted(self._keys[a]), self._emitted(self._keys[b]))

    def _span_of(self, low, high) -> tuple[int, int]:
        a = min(bisect.bisect_left(self._keys, low), len(self._keys) - 1)
        b = max(bisect.bisect_right(self._keys, high) - 1, 0)
        return a, b

    def _parsed_span(self, span):
        return span

    def set_value(self, span) -> None:
        """Show ``span`` (the full one for ``None``) without emitting."""
        if not self._keys:
            return
        if span is None:
            self.histogram.set_indices(0, len(self._keys) - 1)
        else:
            self.histogram.set_indices(*self._span_of(*self._parsed_span(span)))
        self._sync()

    def _on_span(self, _span) -> None:
        self._sync()
        self.changed.emit(self.value())

    def _sync(self) -> None:
        if not self._keys:
            self.span_label.setText("No values to pick from.")
            return
        a, b = self.histogram.indices()
        whole = a == 0 and b == len(self._keys) - 1
        self.span_label.setText(
            f"{self._text(self._keys[a])} – {self._text(self._keys[b])}"
            + ("   (every value)" if whole else "")
        )


class NumberRange(_RangeSection):
    """A range over a number column's distinct values (``"1"`` and ``"1.0"`` are one).

    ``changed`` carries ``(lo, hi)`` floats inclusive, or ``None`` for the full span.
    """

    def __init__(self, column: Column, parent=None) -> None:
        parsed: dict[float, int] = {}
        missing = 0
        for text, count in column.counts.items():
            number = _as_number(text)
            if number is None:
                missing += count
            else:
                parsed[number] = parsed.get(number, 0) + count
        values = sorted(parsed)
        super().__init__(
            values, [parsed[v] for v in values], values, _number_text, missing, parent
        )
        self.values = values


class DateRange(_RangeSection):
    """A range over a date column's days, on bars of clips per day.

    ``changed`` carries ``(lo, hi)`` ISO dates inclusive, or ``None`` for the full span.
    """

    def __init__(self, column: Column, parent=None) -> None:
        per_day: dict[dt.date, int] = {}
        missing = 0
        for text, count in column.counts.items():
            day = _as_date(text)
            if day is None:
                missing += count
            else:
                per_day[day] = per_day.get(day, 0) + count
        days = sorted(per_day)
        super().__init__(
            days,
            [per_day[d] for d in days],
            [float(d.toordinal()) for d in days],
            lambda day: day.isoformat(),
            missing,
            parent,
        )
        self.days = days

    def _emitted(self, key):
        return key.isoformat()

    def _parsed_span(self, span):
        low, high = (_as_date(bound) for bound in span)
        return (low or self.days[0], high or self.days[-1])


class TimeRange(_RangeSection):
    """A range over a time-of-day column's clock times, on bars of clips per time.

    ``changed`` carries ``(lo, hi)`` as the column's own time strings, inclusive, or ``None``.
    """

    def __init__(self, column: Column, parent=None) -> None:
        per_time: dict[float, int] = {}
        shown: dict[float, str] = {}
        missing = 0
        for text, count in column.counts.items():
            seconds = metadata_store.parse_time(text)
            if seconds is None:
                missing += count
            else:
                per_time[seconds] = per_time.get(seconds, 0) + count
                shown.setdefault(seconds, text.strip())
        times = sorted(per_time)
        self._shown = shown
        super().__init__(
            times,
            [per_time[t] for t in times],
            times,
            lambda seconds: shown[seconds],
            missing,
            parent,
        )
        self.times = times

    def _emitted(self, key):
        return self._shown[key]

    def _parsed_span(self, span):
        low, high = (metadata_store.parse_time(str(bound)) for bound in span)
        return (
            self.times[0] if low is None else low,
            self.times[-1] if high is None else high,
        )


def session_text(session) -> str:
    """A time-in-session range as a person reads it: ``"12%–40%"``."""
    lo, hi = session
    return f"{100 * lo:.0f}%–{100 * hi:.0f}%"


def session_summary(session) -> str:
    """What the Time in session line says at its right: ``"any"`` or ``"12%–40%"``."""
    return "any" if session is None else session_text(session)


class SessionRange(QWidget):
    """Time in session: a plain two-handle slider over each recording's whole video, start to
    end, in :data:`SESSION_STEPS` steps — the bar-less :class:`RangeHistogram` — with "start ·
    middle · end" under it and a line saying the range and how many clips it holds ("12%–40% of
    each recording · 840 clips", or "Whole session").

    ``changed`` carries ``(lo, hi)`` fractions, or ``None`` for the whole session. The clip count
    is the page's (:meth:`set_count`), since only it has the clips.
    """

    changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        steps = SESSION_STEPS
        self.slider = RangeHistogram(
            [i / steps for i in range(steps)], [0] * steps, bars=False
        )
        self.slider.span_changed.connect(self._on_span)
        self._slider_tip = self.slider.toolTip()
        layout.addWidget(self.slider)
        ticks = QHBoxLayout()
        ticks.setContentsMargins(0, 0, 0, 0)
        self.ticks = [QLabel(text) for text in ("start", "middle", "end")]
        for label, align in zip(
            self.ticks,
            (
                Qt.AlignmentFlag.AlignLeft,
                Qt.AlignmentFlag.AlignHCenter,
                Qt.AlignmentFlag.AlignRight,
            ),
        ):
            label.setFont(small_font(label))
            label.setAlignment(align)
            ticks.addWidget(label, 1)
        layout.addLayout(ticks)
        self.span_label = QLabel("")
        self.span_label.setWordWrap(True)
        self.span_label.setToolTip(SESSION_RULE_TEXT)
        layout.addWidget(self.span_label)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setVisible(False)
        layout.addWidget(self.note)
        self._count: int | None = None
        self._sync()

    def value(self) -> tuple[float, float] | None:
        first, last = self.slider.indices()
        if first == 0 and last == SESSION_STEPS - 1:
            return None
        return (first / SESSION_STEPS, (last + 1) / SESSION_STEPS)

    def set_value(self, session) -> None:
        """Show ``session`` (the whole one for ``None``) without emitting."""
        if session is None:
            self.slider.set_indices(0, SESSION_STEPS - 1)
        else:
            lo, hi = session
            first = int(round(lo * SESSION_STEPS))
            last = max(int(round(hi * SESSION_STEPS)) - 1, first)
            self.slider.set_indices(first, last)
        self._sync()

    def set_count(self, clips: int | None) -> None:
        """How many of the run's clips the range holds (by midpoint); ``None`` says nothing."""
        self._count = clips
        self._sync()

    def set_available(self, available: bool, reason: str) -> None:
        """Grey the slider, with ``reason`` under it, when no recording has a timestamp map."""
        set_available(self.slider, available, reason)
        if available:
            self.slider.setToolTip(self._slider_tip)
        self.note.setText("" if available else reason)
        self.note.setVisible(not available)

    def _on_span(self, _span) -> None:
        self._sync()
        self.changed.emit(self.value())

    def _sync(self) -> None:
        session = self.value()
        if session is None:
            self.span_label.setText("Whole session")
            return
        line = f"{session_text(session)} of each recording"
        if self._count is not None:
            line += f" · {self._count:,} clip{'s' if self._count != 1 else ''}"
        self.span_label.setText(line)


def _value_entries(column: Column) -> list[tuple[str, str, int]]:
    """A checklist's entries: ``(raw value, shown text, clips)``, in the column's own order —
    numbers and dates by value, categories by name — with "(no value)" last."""

    def order(item):
        text = item[0]
        parsed = _parse(column, text)
        return (
            text == NO_VALUE,
            parsed is None,
            parsed if parsed is not None else 0,
            text,
        )

    return [
        (value, value or NO_VALUE_TEXT, count)
        for value, count in sorted(column.counts.items(), key=order)
    ]


def _section_for(column: Column) -> QWidget:
    """The widget for a column that has a choice to make (``section_kind`` is not ``"same"``)."""
    if section_kind(column) == "range":
        if column.kind == NUMBER:
            return NumberRange(column)
        if column.kind == TIME:
            return TimeRange(column)
        return DateRange(column)
    return Checklist(_value_entries(column), noun=column.name)


def _field_rule() -> QFrame:
    """The thin line above each field: one field's edge, so the names do not run together."""
    rule = QFrame()
    rule.setFrameShape(QFrame.Shape.HLine)
    rule.setFrameShadow(QFrame.Shadow.Plain)
    rule.setEnabled(False)
    return rule


def _field_name_font() -> QFont:
    font = QFont(QApplication.font("QLabel"))
    font.setBold(True)
    return font


def _hover_tint(palette: QPalette) -> QColor:
    """The window colour a shade towards the text: lighter on a dark theme, darker on a light
    one — a faint "this line answers the pointer" that is not a button's frame."""
    base = palette.color(QPalette.ColorRole.Window)
    return base.lighter(125) if base.lightness() < 128 else base.darker(106)


class _FieldHead(QWidget):
    """A field's line: "▸ name" in bold, its summary and (while filtered) its Reset at the right.

    Not a button. A field name drawn as an auto-raise tool button rose into a framed button
    under the pointer, and a column of them read as "too button-y and busy" (user, review round
    8). So the line is plain text that answers like a row: a click anywhere on it toggles, the
    pointer over it tints it, and it takes the focus — Space or Enter toggles, and the focused
    line is outlined in the palette's highlight colour.
    """

    toggled = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hovered = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.toggled.emit()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.toggled.emit()
            return
        super().keyPressEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def is_hovered(self) -> bool:
        return self._hovered

    def tooltip_text(self) -> str:
        """The line's tooltip: what it does, after the whole text of any word cut to fit (the
        words take no mouse events, so their tooltips are the line's)."""
        cut = [
            label.text()
            for label in self.findChildren(ElidedLabel)
            if label.isVisibleTo(self) and label.is_elided()
        ]
        return "\n".join([*cut, self.toolTip()] if cut else [self.toolTip()])

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            QToolTip.showText(event.globalPos(), self.tooltip_text(), self)
            return True
        return super().event(event)

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt's name
        if not (self._hovered or self.hasFocus()):
            return
        painter = QPainter(self)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._hovered:
            painter.fillRect(rect, _hover_tint(self.palette()))
        if self.hasFocus():
            painter.setPen(
                QPen(self.palette().color(QPalette.ColorRole.Highlight), 1.5)
            )
            painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.end()


class FieldRow(QWidget):
    """One field of the dock on one line — "▸ name", and at the right its value count or the
    filter set on it, and a small Reset while it has one — with its values under it while open.
    A click anywhere on the line (or Space / Enter on it) opens or closes it. ``left`` says the
    pointer has gone, which is when the bar re-sorts a field whose filter changed under the hand
    (rather than moving it mid-click). ``reset_button`` is wired by the bar, which owns the
    selection.
    """

    left = pyqtSignal()

    def __init__(self, name: str, section: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.name = name
        self.section = section
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(_field_rule())
        self.head = _FieldHead()
        self.head.setToolTip(f"Show or hide the values of {name}")
        self.head.toggled.connect(lambda: self.set_open(not self.is_open()))
        head = QHBoxLayout(self.head)
        head.setContentsMargins(lines(0.25), 1, 2, 1)
        self.title = ElidedLabel("")
        self.title.setFont(_field_name_font())
        # A summary ("50%–100%") holds more of itself than the name beside it before eliding.
        self.summary = ElidedLabel("", min_chars=SUMMARY_MIN_CHARS)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFont(small_font(self.reset_button))
        # Sized to its word: a style's minimum push-button width made it as wide as Reset all.
        metrics = self.reset_button.fontMetrics()
        self.reset_button.setFixedSize(
            metrics.horizontalAdvance("Reset") + lines(1), metrics.height() + 6
        )
        self.reset_button.setToolTip(f"Remove the filter on {name}")
        self.reset_button.setVisible(False)
        for label in (self.title, self.summary):
            label.setTextFormat(Qt.TextFormat.PlainText)
            # The line takes the click, so a press on the words toggles it like one anywhere else.
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Name and summary elide rather than widen the line, so no value can push the panel
        # wider than itself; the Reset sits inside the same width.
        self.summary.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.summary)
        head.addWidget(self.reset_button)
        # The line is as tall with its Reset as without, so showing it moves nothing below.
        self.head.setMinimumHeight(self.reset_button.sizeHint().height() + 2)
        layout.addWidget(self.head)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(lines(1.2), 0, 0, lines(0.4))
        body.addWidget(section)
        layout.addWidget(self.body)
        self.set_open(False)

    def is_open(self) -> bool:
        return not self.body.isHidden()

    def set_open(self, open_: bool) -> None:
        self.body.setVisible(bool(open_))
        self.title.setText(("▾ " if open_ else "▸ ") + self.name)

    def set_filtered(self, filtered: bool) -> None:
        """Show the line's Reset while the field has a filter on it, and only then."""
        self.reset_button.setVisible(bool(filtered))

    def summary_text(self) -> str:
        return self.summary.text()

    def leaveEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().leaveEvent(event)
        self.left.emit()


class ConstantRow(QWidget):
    """A field with one value for every clip: its name and that value, greyed and not
    openable, with the reason in its tooltip."""

    def __init__(self, name: str, value: str, actions: int, parent=None) -> None:
        super().__init__(parent)
        self.name = name
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(_field_rule())
        row = QHBoxLayout()
        row.setContentsMargins(lines(0.4), 2, 0, 2)
        outer.addLayout(row)
        self.name_label = ElidedLabel(name)
        self.name_label.setFont(_field_name_font())
        self.value_label = ElidedLabel(
            value or NO_VALUE_TEXT, min_chars=SUMMARY_MIN_CHARS
        )
        for label in (self.name_label, self.value_label):
            label.setTextFormat(Qt.TextFormat.PlainText)
        row.addWidget(self.name_label)
        row.addStretch(1)
        row.addWidget(self.value_label)
        self.setToolTip(constant_tooltip(name, value, actions))
        self.setEnabled(False)
        self._mute_labels()

    def changeEvent(self, event) -> None:  # noqa: N802 — Qt's name
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._mute_labels()

    def _mute_labels(self) -> None:
        """Grey the words with the colour a disabled plain label uses.

        A label made selectable (as every label here is, `selectable.make_text_selectable`)
        draws through a text document, which takes the palette's *Text* role, not WindowText —
        so a palette that greys disabled WindowText and leaves disabled Text alone (a dark
        palette built by role, as the laptop render's was) drew these one-value lines as bright
        as a live field while the light theme greyed them. Re-read on a palette change, so a
        theme switch follows.
        """
        muted = self.palette().color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText
        )
        for label in (self.name_label, self.value_label):
            palette = label.palette()
            if (
                palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
                != muted
            ):
                palette.setColor(
                    QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, muted
                )
                label.setPalette(palette)

    def texts(self) -> tuple[str, str]:
        return self.name_label.text(), self.value_label.text()


# --------------------------------------------------------------------------- #
# the dock and the bar
# --------------------------------------------------------------------------- #
class ScreenList(QWidget):
    """The Differences line's body: the run's saved difference screens, one tickable at a time,
    and **New screen…**.

    ``changed`` carries the :class:`ScreenChoice` ticked, or ``None`` when the tick came off.
    Ticking one unticks the other — two screens cannot shade one tree — which is a radio in
    all but its look; a tick box is what every other list in the dock has, and the note under
    the list says the rule. ``run_requested`` is the button: the widget reads nothing and runs
    nothing, the page answers it.

    **Strictness**, under the list while a screen is ticked: a slider over
    ``compare.STRICTNESS_LEVELS`` with a line saying the level and what the screen keeps there
    ("p ≤ 0.2 · 6 kept of 12 clearing"). ``alpha_changed`` carries the level, ``None`` at the
    screen's own floor. Hidden for a screen with no floor, which has no levels to read at.
    """

    changed = pyqtSignal(object)
    alpha_changed = pyqtSignal(object)
    run_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.list = QListWidget()
        self.list.setObjectName("screen_list")
        self.list.itemChanged.connect(self._on_item)
        layout.addWidget(self.list)
        self.strictness_box = QWidget()
        box = QVBoxLayout(self.strictness_box)
        box.setContentsMargins(0, 4, 0, 0)
        box.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.strictness_caption = QLabel(STRICTNESS_TEXT)
        self.strictness_caption.setToolTip(STRICTNESS_RULE_TEXT)
        head.addWidget(self.strictness_caption)
        head.addStretch(1)
        box.addLayout(head)
        self.strictness = QSlider(Qt.Orientation.Horizontal)
        self.strictness.setObjectName("screen_strictness")
        self.strictness.setRange(0, len(STRICTNESS_LEVELS) - 1)
        self.strictness.setPageStep(1)
        self.strictness.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strictness.setTickInterval(1)
        self.strictness.setValue(STRICTNESS_LEVELS.index(FLOOR_ALPHA))
        self.strictness.setToolTip(STRICTNESS_RULE_TEXT)
        self.strictness.valueChanged.connect(self._on_strictness)
        box.addWidget(self.strictness)
        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        self.strictness_ends = [
            QLabel(f"p ≤ {STRICTNESS_LEVELS[0]:g}"),
            QLabel("stricter ← → looser"),
            QLabel(f"p ≤ {STRICTNESS_LEVELS[-1]:g}"),
        ]
        for label, align in zip(
            self.strictness_ends,
            (
                Qt.AlignmentFlag.AlignLeft,
                Qt.AlignmentFlag.AlignHCenter,
                Qt.AlignmentFlag.AlignRight,
            ),
        ):
            label.setFont(small_font(label))
            label.setAlignment(align)
            ends.addWidget(label, 1)
        box.addLayout(ends)
        self.strictness_label = QLabel("")
        self.strictness_label.setWordWrap(True)
        self.strictness_label.setToolTip(STRICTNESS_RULE_TEXT)
        box.addWidget(self.strictness_label)
        layout.addWidget(self.strictness_box)
        self.run_button = QPushButton(RUN_SCREEN_TEXT)
        self.run_button.setToolTip(
            "Run a difference screen over every cluster of this run: which clusters differ "
            "across a column of your recording information, with the floor they must clear."
        )
        self.run_button.clicked.connect(self.run_requested)
        layout.addWidget(self.run_button)
        self._choices: dict[str, ScreenChoice] = {}
        self._quiet = False
        self.set_entries([])

    def set_entries(
        self, entries: Sequence[ScreenChoice], *, loading: bool = False
    ) -> None:
        """The screens to offer; the one ticked stays ticked when it is still offered."""
        picked = self.value()
        self._quiet = True
        self.list.clear()
        self._choices = {}
        for choice in entries:
            text = f"{choice.label}   {choice.note}" if choice.note else choice.label
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, choice.key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            if choice.leaves is None:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(choice.note)
            else:
                item.setToolTip(
                    f"{choice.label}: {choice.note}. Tick it to shade the tree by the "
                    f"clusters it kept."
                )
            self.list.addItem(item)
            self._choices[choice.key] = choice
        self.list.setVisible(bool(entries))
        if loading:
            self.note.setText(SCREENS_LOADING_TEXT)
        elif entries:
            self.note.setText(SCREENS_HINT_TEXT)
        else:
            self.note.setText(NO_SCREENS_TEXT)
        fit_list_height(self.list, max_rows=6)
        self._quiet = False
        choice = self._choices.get(picked or "")
        if choice is not None and choice.leaves is not None:
            self.set_value(picked)
        else:
            self._sync_strictness()

    def value(self) -> str | None:
        """The ticked screen's key, or ``None``."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                return item.data(Qt.ItemDataRole.UserRole)
        return None

    def set_value(self, key: str | None) -> None:
        """Tick exactly ``key`` (nothing for ``None``) without emitting."""
        self._quiet = True
        for i in range(self.list.count()):
            item = self.list.item(i)
            on = key is not None and item.data(Qt.ItemDataRole.UserRole) == key
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._quiet = False
        self._sync_strictness()

    def alpha(self) -> float | None:
        """The strictness the slider is at: a level of ``compare.STRICTNESS_LEVELS``, or
        ``None`` at the screen's own floor."""
        level = STRICTNESS_LEVELS[self.strictness.value()]
        return None if abs(level - FLOOR_ALPHA) < 1e-9 else level

    def set_alpha(self, alpha: float | None) -> None:
        """Put the slider at ``alpha`` (the screen's own floor for ``None``) without emitting.
        A level the slider does not offer puts it at the floor."""
        wanted = FLOOR_ALPHA if alpha is None else float(alpha)
        index = STRICTNESS_LEVELS.index(FLOOR_ALPHA)
        for i, level in enumerate(STRICTNESS_LEVELS):
            if abs(level - wanted) < 1e-9:
                index = i
        self._quiet = True
        self.strictness.setValue(index)
        self._quiet = False
        self._sync_strictness()

    def _sync_strictness(self) -> None:
        """Show the slider while a screen with levels is ticked, and say what it keeps at the
        level the slider is at."""
        choice = self._choices.get(self.value() or "")
        level = choice.level(self.alpha()) if choice is not None else None
        shown = (
            choice is not None
            and choice.leaves is not None
            and choice.floor is not None
        )
        self.strictness_box.setVisible(shown)
        if not shown or level is None:
            self.strictness_label.setText("")
            return
        at = (
            "the screen's own floor"
            if self.alpha() is None
            else "looser than its floor"
        )
        if level.alpha < FLOOR_ALPHA:
            at = "stricter than its floor"
        self.strictness_label.setText(
            f"p ≤ {level.alpha:g} ({at}) · {len(level.kept)} kept of "
            f"{level.clearing} clearing"
        )

    def _on_strictness(self, _value: int) -> None:
        self._sync_strictness()
        if self._quiet:
            return
        self.alpha_changed.emit(self.alpha())

    def _on_item(self, item) -> None:
        if self._quiet:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() != Qt.CheckState.Checked:
            self._sync_strictness()
            self.changed.emit(None)
            return
        self.set_value(key)
        self.changed.emit(self._choices.get(key))


def _holder() -> QWidget:
    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setContentsMargins(0, 0, 0, 0)
    return holder


def panel_name(selection: Selection) -> str:
    """What the dock's strip reads while it is shut: "Filters", or "Filters · 2 active"."""
    active = len(selection.keys())
    return f"Filters · {active} active" if active else "Filters"


class FilterDock(QWidget):
    """The filter column beside the tree: Location, then one line per field (Video and each
    metadata column), scrolling as one. It folds by its splitter's grip, like every side panel
    (`gui/grip_splitter.py`) — there is no hide button of its own."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Filters")
        title.setFont(emphasis_font(title, steps=0))
        body.addWidget(title)
        #: The key and the count line, above the sections and outside their scroll (the key
        #: reads what the sections set, so it stays in view while they are scrolled). Always
        #: shown and always the same height — greyed with the run's clip count while nothing is
        #: filtered — so the fields never move when the first filter goes on or the last comes
        #: off (review round 8, option B; round 7 hid it and the column jumped).
        self.top_box = QWidget()
        top_box = QVBoxLayout(self.top_box)
        top_box.setContentsMargins(0, 0, 0, lines(0.4))
        self.key = MatchKey()
        top_box.addWidget(self.key)
        #: "2,781 clips", or "n of N clips match" and Reset all (filled by the bar).
        self.count_row = QHBoxLayout()
        self.count_row.setContentsMargins(0, 0, 0, 0)
        top_box.addLayout(self.count_row)
        body.addWidget(self.top_box)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sections = QWidget()
        self._column = QVBoxLayout(self.sections)
        self._column.setContentsMargins(0, 0, lines(0.5), 0)
        # No spacing between the Location line and the field lines: Location is one of them,
        # and a gap after it made its line read taller than the rest (user, 2026-11-09).
        self._column.setSpacing(0)
        self.location_box = _holder()
        # Location and Time in session are two lines of the list, spaced like the fields.
        self.location_box.layout().setSpacing(0)
        self._column.addWidget(self.location_box)
        #: "Find a field (N)", shown over more than :data:`FIND_FIELD_OVER` fields — with room
        #: above and below, which goes with it when it is hidden.
        self.find_box = QWidget()
        find = QVBoxLayout(self.find_box)
        find.setContentsMargins(0, lines(0.3), 0, lines(0.3))
        self.find_field = QLineEdit()
        self.find_field.setObjectName("field_search")
        self.find_field.setClearButtonEnabled(True)
        find.addWidget(self.find_field)
        self.find_box.setVisible(False)
        self._column.addWidget(self.find_box)
        #: The field lines, arranged by the bar: in use, a divider, the rest, the one-value ones.
        self.fields_box = _holder()
        self.fields_box.layout().setSpacing(0)
        self._column.addWidget(self.fields_box)
        self._column.addStretch(1)
        self.scroll.setWidget(self.sections)
        body.addWidget(self.scroll, 1)
        outer.addWidget(self.body, 1)

        self.body.setMinimumWidth(DOCK_MIN_PX)


class FilterBar(QObject):
    """The filters' state and wiring: a page lays out ``bar.dock`` (the filters with the key, the
    count line and Reset all at the dock's top) and listens to ``selection_changed``. Nothing of
    it is drawn outside the dock — the row above the tree it once held went with the label field,
    which moved into the label bar (review round 9).

    ``selection_changed`` carries a :class:`Selection` on every change, whichever gesture made it
    (a dock section, a field's Reset, Reset all, or :meth:`set_selection`).
    """

    selection_changed = pyqtSignal(object)
    #: **Run a new screen…** on the Differences line: the page opens the launch dialog.
    screen_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selection = Selection()
        #: The saved screens offered, by id; ``None`` until the page has listed them, so a
        #: selection naming one is kept until there is a list to check it against.
        self._screens: dict[str, ScreenChoice] | None = None
        self._columns: dict[str, Column] = {}
        self._recordings: list[RecordingChoice] = []
        self._names: dict[str, str] = {}
        self._meta_sections: dict[str, QWidget] = {}
        self._same: dict[str, str] = {}
        self._video_list: Checklist | None = None
        #: Every openable field's line, keyed by its condition key, in the usual order.
        self._rows: dict[tuple, FieldRow] = {}
        self._video_row: FieldRow | None = None
        self._meta_rows: list[FieldRow] = []
        self._constant_rows: list[ConstantRow] = []
        #: The fields that were in use at the last arrangement: one newly in use opens.
        self._in_use: tuple = ()
        #: A field's filter changed under the pointer; re-sort when the pointer leaves it.
        self._arrange_pending = False
        self.dock = FilterDock()
        self.dock.find_field.textChanged.connect(lambda _t: self._arrange())
        fields = self.dock.fields_box.layout()
        self.in_use_label = QLabel(IN_USE_TEXT)
        # Each field draws its own line, so the break after the fields in use is a caption.
        self.in_use_divider = QLabel(OTHER_FIELDS_TEXT)
        self.no_columns = QLabel("This experiment has no metadata columns.")
        self.no_columns.setWordWrap(True)
        for widget in (self.in_use_label, self.in_use_divider, self.no_columns):
            fields.addWidget(widget)

        count_row = self.dock.count_row
        self.count_label = ElidedLabel("")
        self.count_label.set_extra_tooltip(MATCH_RULE_TEXT)
        count_row.addWidget(self.count_label, 1)
        self.clear_button = QPushButton("Reset all")
        self.clear_button.setToolTip("Remove every filter")
        self.clear_button.clicked.connect(self.clear)
        # Hidden with no filter but keeping its place, so the line is one height either way.
        policy = self.clear_button.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.clear_button.setSizePolicy(policy)
        count_row.addWidget(self.clear_button)

        location = self.dock.location_box.layout()
        self.region = RegionPicker(width=MAP_PX, reset_row=False)
        self.region.region_changed.connect(self._on_region)
        self.location_note = self.region.note
        #: The Location line: folds like a field's, its Reset in the line.
        self.location_row = FieldRow("Location", self.region)
        self.location_row.head.setToolTip(
            "Where the animal is: show or hide the map of where the clips were"
        )
        # The map uses the dock's whole width rather than a field's indent.
        self.location_row.body.layout().setContentsMargins(0, 0, 0, lines(0.4))
        self.location_row.reset_button.setToolTip("Remove the picked squares")
        self.location_row.reset_button.clicked.connect(lambda: self.remove(LOCATION))
        self.location_row.head.toggled.connect(self._remember_location_open)
        self.location_row.set_open(
            QSettings().value(LOCATION_OPEN_KEY, True, type=bool)
        )
        #: The line's name, where the old heading was.
        self.location_heading = self.location_row.title
        location.addWidget(self.location_row)
        #: Time in session: a range over each recording's whole video, folding like Location.
        self.session = SessionRange()
        self.session.changed.connect(self._on_session)
        self.session_row = FieldRow("Time in session", self.session)
        self.session_row.head.setToolTip(
            "When in the recording: show or hide a range from each recording's start to its end"
        )
        self.session_row.reset_button.setToolTip("Use the whole session")
        self.session_row.reset_button.clicked.connect(lambda: self.remove(SESSION))
        self.session_row.head.toggled.connect(self._remember_session_open)
        self.session_row.set_open(QSettings().value(SESSION_OPEN_KEY, False, type=bool))
        location.addWidget(self.session_row)
        #: Differences: the run's saved difference screens, folding like Location's line.
        self.screens = ScreenList()
        self.screens.changed.connect(self._on_screen)
        self.screens.alpha_changed.connect(self._on_alpha)
        self.screens.run_requested.connect(self.screen_requested)
        self.screens_row = FieldRow("Differences", self.screens)
        self.screens_row.head.setToolTip(
            "Which clusters differ: show or hide this run's saved difference screens"
        )
        self.screens_row.reset_button.setToolTip("Untick the screen")
        self.screens_row.reset_button.clicked.connect(lambda: self.remove(CLUSTERS))
        location.addWidget(self.screens_row)
        self.dock.scroll.viewport().installEventFilter(self)

        self.set_columns([])
        self.set_recordings([])
        self._total: int | None = None
        self._matching: int | None = None
        self._refresh()

    # -- the map's width and the Location line ------------------------------ #
    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        dock = getattr(self, "dock", None)
        if (
            event.type() == QEvent.Type.Resize
            and dock is not None
            and watched is dock.scroll.viewport()
        ):
            self._fit_map()
        return super().eventFilter(watched, event)

    def _fit_map(self) -> None:
        """The map fills the dock's width, between :data:`MAP_MIN_PX` and :data:`MAP_MAX_PX`."""
        room = (
            self.dock.scroll.viewport().width()
            - self.dock.sections.layout().contentsMargins().right()
            - self.dock.sections.layout().contentsMargins().left()
        )
        self.region.map.set_side(min(max(room, MAP_MIN_PX), MAP_MAX_PX))

    def _remember_location_open(self) -> None:
        QSettings().setValue(LOCATION_OPEN_KEY, self.location_row.is_open())

    def _remember_session_open(self) -> None:
        QSettings().setValue(SESSION_OPEN_KEY, self.session_row.is_open())

    # -- inputs ------------------------------------------------------------- #
    def set_columns(self, columns: Sequence[Column]) -> None:
        """The metadata columns, each with its inferred type and clips per value."""
        self._columns = {c.name: c for c in columns}
        self._meta_sections = {}
        self._same = {}
        for old in (*self._meta_rows, *self._constant_rows):
            old.setParent(None)
            old.deleteLater()
        self._meta_rows, self._constant_rows = [], []
        for column in columns:
            if section_kind(column) == "same":
                value = single_value(column)
                self._same[column.name] = value
                self._constant_rows.append(
                    ConstantRow(column.name, value, sum(column.counts.values()))
                )
                continue
            section = _section_for(column)
            section.changed.connect(
                lambda value, name=column.name: self._on_meta(name, value)
            )
            self._meta_sections[column.name] = section
            self._meta_rows.append(self._field_row(column.name, section))
        self.no_columns.setVisible(not columns)
        self._rebuild_rows()
        self._keep_showable()

    def constant_rows(self) -> list[ConstantRow]:
        """The greyed lines of the fields with one value, in column order."""
        return list(self._constant_rows)

    def field_rows(self) -> list[FieldRow]:
        """The openable field lines as arranged now, top to bottom, hidden ones included."""
        layout = self.dock.fields_box.layout()
        widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
        return [w for w in widgets if isinstance(w, FieldRow)]

    def field_row(self, name: str) -> FieldRow:
        """The line of field ``name`` (``"Video"`` for the recordings, ``"Location"`` for the
        map, ``"Time in session"`` for the session range)."""
        if name == "Location":
            return self.location_row
        if name == "Time in session":
            return self.session_row
        if name == "Differences":
            return self.screens_row
        if name == "Video" and self._video_row is not None:
            return self._video_row
        return self._rows[("meta", name)]

    def set_screens(
        self, screens: Sequence[ScreenChoice], *, loading: bool = False
    ) -> None:
        """The run's saved difference screens, newest first — what the Differences line
        offers. ``loading`` says the page is still reading them: the offer is then unknown,
        not empty, so the screens already known stay known and a ticked one stays applied
        until the new list lands to check it against."""
        if not loading:
            self._screens = {s.key: s for s in screens}
        self.screens.set_entries(screens, loading=loading)
        self._keep_showable()
        self._refresh()

    def screen(self, key: str | None) -> ScreenChoice | None:
        """The offered screen ``key`` names, or ``None``."""
        return None if key is None or self._screens is None else self._screens.get(key)

    def set_recordings(self, recordings: Sequence[RecordingChoice]) -> None:
        """The run's recordings, with clips per recording."""
        self._recordings = list(recordings)
        self._names = {r.recording_id: r.name for r in recordings}
        if self._video_row is not None:
            self._video_row.setParent(None)
            self._video_row.deleteLater()
        self._video_list = Checklist(
            [(r.recording_id, r.name, r.count) for r in recordings],
            noun="recordings",
            searchable=True,
        )
        self._video_list.changed.connect(self._on_video)
        self._video_row = self._field_row("Video", self._video_list)
        self._rebuild_rows()
        self._keep_showable()

    def _showable(self, selection: Selection) -> Selection:
        """``selection`` less every condition the dock has no line to show and reset: a column
        gone or down to one value (a greyed line), a range on a column now ticked or ticks on
        one now a range, ticked values the column no longer has, recordings the run no longer
        has. Each would stay applied, and counted in "Filters · N active", out of sight.
        """
        values = {}
        for name, picked in selection.values.items():
            section = self._meta_sections.get(name)
            if isinstance(section, Checklist):
                kept = frozenset(picked) & frozenset(section.keys())
                if kept:
                    values[name] = kept
        ranges = {
            name: span
            for name, span in selection.ranges.items()
            if name in self._meta_sections
            and not isinstance(self._meta_sections[name], Checklist)
        }
        leaves, source, alpha = (
            selection.leaves,
            selection.leaves_source,
            selection.alpha,
        )
        if self._screens is not None and leaves is not None:
            offered = self._screens.get(source or "")
            level = offered.level(alpha) if offered is not None else None
            if offered is None or offered.leaves is None:
                leaves, source, alpha = None, None, None
            elif level is None or (alpha is not None and offered.floor is None):
                # The screen was not read at that strictness (no floor, no levels): its own.
                leaves, alpha = offered.leaves, None
            else:
                # The screen's leaves as listed now, at the strictness in force: a re-read
                # lists them over the view the page has now, which a rebuild may have re-cut.
                leaves = level.leaves
        recordings = selection.recording_ids
        if recordings is not None:
            recordings = frozenset(recordings) & {
                r.recording_id for r in self._recordings
            }
            recordings = recordings or None
        return replace(
            selection,
            values=values,
            ranges=ranges,
            recording_ids=recordings,
            leaves=leaves,
            leaves_source=source,
            alpha=alpha,
        )

    def _keep_showable(self) -> None:
        """After a refill: drop what the dock can no longer show, and tell the page."""
        showable = self._showable(self._selection)
        if showable != self._selection:
            self.set_selection(showable)
        else:
            self._sync_sections()
            self._refresh()

    def _field_row(self, name: str, section: QWidget) -> FieldRow:
        row = FieldRow(name, section)
        row.left.connect(self._on_row_left)
        key = (
            VIDEO if name == "Video" and section is self._video_list else ("meta", name)
        )
        row.reset_button.clicked.connect(lambda _c=False, k=key: self.remove(k))
        return row

    def _rebuild_rows(self) -> None:
        """The field lines changed: key them, arrange them, show the selection in them."""
        self._rows = {}
        if self._video_row is not None:
            self._rows[VIDEO] = self._video_row
        for row in self._meta_rows:
            self._rows[("meta", row.name)] = row
        self._in_use = ()
        total = len(self._rows) + len(self._constant_rows)
        self.dock.find_field.setPlaceholderText(f"Find a field ({total})")
        self.dock.find_box.setVisible(total > FIND_FIELD_OVER)
        if total <= FIND_FIELD_OVER:
            self.dock.find_field.clear()
        make_text_selectable(self.dock.sections)
        self._sync_sections()
        self._refresh()
        self._arrange()

    def set_density(self, grid) -> None:
        """The whole run's position counts, ``(16, 16)`` indexed ``[by, bx]``."""
        self.region.set_density(grid)

    def set_location_available(self, available: bool, reason: str) -> None:
        """Grey the location picker, with ``reason`` said under its heading, when the run has
        no positions to pick from."""
        self.region.set_available(available, reason)

    def set_session_available(self, available: bool, reason: str) -> None:
        """Grey the Time in session slider, with ``reason`` under it, when no recording of the
        run has a timestamp map to measure its session by."""
        self.session.set_available(available, reason)

    def set_session_count(self, clips: int | None) -> None:
        """How many of the run's clips the session range holds, for its line."""
        self.session.set_count(clips)

    def set_match_count(self, matching: int | None, total: int) -> None:
        """``matching`` of ``total`` clips match (``None`` while nothing is filtered)."""
        self._matching, self._total = matching, total
        self._refresh()

    # -- the selection ------------------------------------------------------ #
    def selection(self) -> Selection:
        return self._selection

    def set_selection(self, selection: Selection) -> None:
        """Show ``selection`` in every section and line; emits when it is a change."""
        if selection == self._selection:
            return
        self._selection = selection
        self._sync_sections()
        self._refresh()
        self._arrange()
        self.selection_changed.emit(selection)

    def clear(self) -> None:
        """What Reset all does."""
        self.set_selection(Selection())

    def remove(self, key: tuple) -> None:
        """What a field's Reset does: drop condition ``key`` and nothing else."""
        self.set_selection(self._selection.without(key))

    # -- internals ---------------------------------------------------------- #
    def _changed(self, selection: Selection) -> None:
        if selection == self._selection:
            return
        self._selection = selection
        self._refresh()
        # Set from inside a field: re-sorting now would move the line from under the pointer
        # between two clicks, so it waits until the pointer leaves the field.
        self._arrange_pending = True
        self.selection_changed.emit(selection)

    def _on_row_left(self) -> None:
        if self._arrange_pending:
            self._arrange()

    def _arrange(self) -> None:
        """Lay the field lines out: those with a filter on under "In use" (a field newly in use
        opens), a divider, the rest in their usual order, then the one-value lines; "Find a
        field" hides the lines whose name it does not contain."""
        self._arrange_pending = False
        keys = set(self._selection.keys())
        in_use = tuple(k for k in self._rows if k in keys)
        for key in in_use:
            if key not in self._in_use:
                self._rows[key].set_open(True)
        self._in_use = in_use
        rest = [row for key, row in self._rows.items() if key not in keys]
        needle = self.dock.find_field.text().strip().casefold()

        def shown(row) -> bool:
            return not needle or needle in row.name.casefold()

        layout = self.dock.fields_box.layout()
        while layout.count():
            layout.takeAt(0)
        layout.addWidget(self.in_use_label)
        for key in in_use:
            layout.addWidget(self._rows[key])
        layout.addWidget(self.in_use_divider)
        for row in (*rest, *self._constant_rows):
            layout.addWidget(row)
        layout.addWidget(self.no_columns)
        for row in (*self._rows.values(), *self._constant_rows):
            row.setVisible(shown(row))
        any_in_use = any(shown(self._rows[k]) for k in in_use)
        self.in_use_label.setVisible(any_in_use)
        self.in_use_divider.setVisible(
            any_in_use and any(shown(r) for r in (*rest, *self._constant_rows))
        )

    def _on_meta(self, name: str, value) -> None:
        self._changed(with_condition(self._selection, name, value))

    def _on_video(self, value) -> None:
        self._changed(replace(self._selection, recording_ids=value))

    def _on_region(self, region) -> None:
        self._changed(replace(self._selection, cells=frozenset(region) or None))

    def _on_session(self, session) -> None:
        self._changed(replace(self._selection, session=session))

    def _on_screen(self, choice) -> None:
        if choice is None or choice.leaves is None:
            self._changed(self._selection.without(CLUSTERS))
            return
        self._changed(self._at_strictness(choice, self.screens.alpha()))

    def _on_alpha(self, alpha) -> None:
        choice = (
            self._screens.get(self.screens.value() or "") if self._screens else None
        )
        if choice is None or choice.leaves is None:
            return
        self._changed(self._at_strictness(choice, alpha))

    def _at_strictness(self, choice: ScreenChoice, alpha: float | None) -> Selection:
        """The selection with ``choice`` ticked at the strictness ``alpha``: that level's
        leaves, or the screen's own where it was not read at that level."""
        level = choice.level(alpha)
        if level is None or (alpha is not None and choice.floor is None):
            return replace(
                self._selection,
                leaves=choice.leaves,
                leaves_source=choice.key,
                alpha=None,
            )
        return replace(
            self._selection, leaves=level.leaves, leaves_source=choice.key, alpha=alpha
        )

    def _sync_sections(self) -> None:
        for name, section in self._meta_sections.items():
            value = condition(self._selection, name)
            wants_values = isinstance(section, Checklist)
            if value is not None and _is_values(value) != wants_values:
                value = None  # a condition of the other shape cannot be shown here
            section.set_value(value)
        if self._video_list is not None:
            self._video_list.set_value(self._selection.recording_ids)
        self.region.blockSignals(True)
        self.region.set_region(self._selection.cells or frozenset())
        self.region.blockSignals(False)
        self.session.set_value(self._selection.session)
        self.screens.set_value(self._selection.leaves_source)
        self.screens.set_alpha(self._selection.alpha)
        if self._selection.leaves is not None:
            self.screens_row.set_open(True)

    def _refresh(self) -> None:
        if not hasattr(self, "_total"):
            return
        selection = self._selection
        self.clear_button.setVisible(not selection.is_empty)
        filtering = not selection.is_empty and self._matching is not None
        if self._total is None:
            line = ""
        elif filtering:
            line = f"{self._matching:,} of {self._total:,} clips match"
        else:
            line = f"{self._total:,} clips"
        self.count_label.setText(line)
        keys = set(selection.keys())
        for key, row in self._rows.items():
            row.summary.setText(
                field_summary(selection, key, self._columns, self._names)
            )
            row.set_filtered(key in keys)
        self.location_row.summary.setText(location_summary(selection.cells))
        self.location_row.set_filtered(bool(selection.cells))
        self.session_row.summary.setText(session_summary(selection.session))
        self.session_row.set_filtered(selection.session is not None)
        self.screens_row.summary.setText(screen_summary(selection, self._screens or {}))
        self.screens_row.set_filtered(selection.leaves is not None)
