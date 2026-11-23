"""Structural requirements every GUI element must satisfy — the list, in one place.

`docs/gui_design.md` §3.1 holds the reasoning for ~45 rules, and `tests/gui/` holds 88 files
enforcing some of them. What did not exist is a **list**: you learned
the rules by writing a page and failing twelve guards one at a time, a new *kind* of element had
nothing to be added to, and prose could drift from enforcement. That last one is a recorded class
in its own right — *"a documented rule that was never implemented… believed, quoted, and
absent"* — and an inventory (#349) found it is the normal state rather than the exception:

    swept, over every subject ......... 16
    spot-checked, one or two instances . 13
    satisfied but unenforced ...........  4
    nothing .............................11

"No `&` in button text" reads as enforced and checks three buttons of roughly two hundred.

**One definition feeds three things.** A requirement is a function; its docstring is the sentence
printed into the guidelines, its body is the check, and its name is what a failure reports. They
cannot disagree, because there is only one of them.

**No Qt at module scope.** `scripts/gen_docs_requirements.py` imports this to print the rules, and
the docs build does not install the `gui` extra. Import Qt inside a predicate, or under
``TYPE_CHECKING`` — the same seam `core/protocols.py` uses to name `torch.Tensor` without pulling
in torch.

**Two rules about rules**, both enforced at registration rather than by habit:

- *Every requirement ships a counterexample* — something it must reject. A guard that cannot fail
  passes forever, and this repo has shipped several: ``len(notice) > 20`` satisfied by a
  twenty-one character string, a parametrisation that iterated zero times, four GUI suites green
  with `SnapshotViewMixin` gutted. The counterexample is the cheapest possible proof that a
  predicate has teeth, so it is a condition of existing rather than a good intention.
- *A requirement states one thing.* If the docstring needs an "and", it is two requirements —
  otherwise a failure cannot say which half broke.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field


class RequirementError(AssertionError):
    """A subject does not satisfy a requirement. Carries the rule so the grid can name it."""


@dataclass(frozen=True)
class Requirement:
    """One structural rule: what it applies to, what it says, and how to check it."""

    name: str
    kinds: tuple[str, ...]
    statement: str
    check: Callable[[object], None]
    counterexample: Callable[[], object]
    #: Subjects excused from this rule, ``{subject label: why}``. An exemption is a claim, so it
    #: carries a reason; a stale one fails, the same contract every allowlist here is held to.
    exempt: dict[str, str] = field(default_factory=dict)
    #: The `NOT_YET_SWEPT` entry this rule retires, spelled exactly as that dict keys it.
    #: Migrating a rule is two edits in two files, so it is exactly the kind of pair that
    #: half-happens: the entry stays behind, goes on reading like real remaining work, and the next
    #: person re-implements a rule that is already enforced. Naming the retired entry here keeps
    #: the pair checkable.
    #:
    #: This said "worded as CLAUDE.md words it" while **none of the twelve values appeared in
    #: CLAUDE.md at all** — a claim about a second file that nothing could check, since the only
    #: consumers are a dict-membership test and a size assertion. The keys are self-describing rule
    #: labels; the prose home is `docs/gui_design.md`.
    retires: str | None = None
    #: Which containers the rule holds in — ``("page",)``, ``("dialog",)``, ``("package",)`` — or
    #: ``None`` for all of them. A rule that is true on pages and not yet on dialogs is a real
    #: state, and this is where it is said out loud: the alternative is one exemption per subject,
    #: which buries the shape of the gap in a hundred identical lines. A scoped rule earns a
    #: `NOT_YET_SWEPT` entry for the containers it does not cover.
    containers: tuple[str, ...] | None = None

    def applies_to(self, kind: str, container: str | None = None) -> bool:
        if kind not in self.kinds:
            return False
        return (
            self.containers is None or container is None or container in self.containers
        )


REQUIREMENTS: list[Requirement] = []


def requirement(
    *,
    kinds: Iterable[str],
    counterexample: Callable[[], object],
    exempt: dict[str, str] | None = None,
    retires: str | None = None,
    containers: tuple[str, ...] | None = None,
):
    """Register a structural requirement.

    ``counterexample`` builds a subject the check must **reject**. It is required, and the grid
    runs it: a requirement whose counterexample passes the check is a requirement that cannot
    fail, and is rejected rather than quietly kept.

    The statement printed into the guidelines is the docstring's first line, so it is written for
    a reader rather than for a test — imperative, one sentence, no "should".
    """

    def register(func: Callable[[object], None]) -> Callable[[object], None]:
        doc = (func.__doc__ or "").strip()
        if not doc:
            raise ValueError(
                f"{func.__name__} has no docstring — its first line is the guideline text"
            )
        statement = doc.split("\n", 1)[0].strip()
        REQUIREMENTS.append(
            Requirement(
                name=func.__name__,
                kinds=tuple(kinds),
                statement=statement,
                check=func,
                counterexample=counterexample,
                exempt=dict(exempt or {}),
                retires=retires,
                containers=containers,
            )
        )
        return func

    return register


def for_kind(kind: str) -> list[Requirement]:
    """Every requirement that applies to ``kind``, in declaration order."""
    return [r for r in REQUIREMENTS if r.applies_to(kind)]


def kinds() -> list[str]:
    """Every kind any requirement mentions, sorted — what the guidelines are grouped by."""
    return sorted({k for r in REQUIREMENTS for k in r.kinds})


# --------------------------------------------------------------------------- #
# button
# --------------------------------------------------------------------------- #
def _a_button_with_an_ampersand():
    from PyQt6.QtWidgets import QPushButton

    return QPushButton("Validate & save")


def _an_unlabelled_button():
    from PyQt6.QtWidgets import QPushButton

    return QPushButton("")


def _a_button_wearing_an_icon():
    from PyQt6.QtGui import QIcon, QPixmap
    from PyQt6.QtWidgets import QPushButton

    button = QPushButton("Install")
    pixmap = QPixmap(16, 16)
    pixmap.fill()
    button.setIcon(QIcon(pixmap))
    return button


@requirement(kinds=("button",), counterexample=_a_button_with_an_ampersand)
def a_button_never_shows_a_bare_ampersand(button) -> None:
    """Button text escapes `&` as `&&` — Qt reads a single one as a keyboard mnemonic.

    "Validate & save" renders as "Validate _save": the ampersand vanishes and the next letter is
    underlined. The rule has been in the guidelines for a long while and was checked on three
    buttons of the audit tab, which is roughly 1.5% of the buttons in the app.
    """
    text = button.text()
    assert "&" not in text.replace("&&", ""), (
        f"{text!r} contains a bare & — Qt will eat it and underline the next letter. "
        "Reword, or escape it as &&."
    )


@requirement(kinds=("button",), counterexample=_an_unlabelled_button)
def a_button_says_what_it_does(button) -> None:
    """A button carries text — icons are not used here, so an empty one is unreachable.

    This is also the shape that once let a real bug through an allowlist: a walk over enabled
    controls keyed on button text let an unlabelled one past, because its label is set later and
    until then it matches the empty string.
    """
    assert button.text().strip(), (
        f"a {type(button).__name__} has no text; with no icon set there is nothing on screen "
        "for the user to read"
    )


@requirement(kinds=("button",), counterexample=_a_button_wearing_an_icon)
def a_button_carries_no_icon(button) -> None:
    """A button is text, never an icon — this app sets no icon sets, SVG or themes.

    The guideline has held since the GUI was written and nothing enforced it: the walk finds
    **0 of 126** buttons wearing one, so this converts a rule that was true by everyone's care
    into a rule that stays true.
    """
    assert button.icon().isNull(), (
        f"{button.text()!r} has an icon set; the house style is text labels only, so an icon "
        "here is either a style break or a rule that needs rewriting"
    )


def _a_disabled_button_with_no_reason():
    from PyQt6.QtWidgets import QPushButton

    button = QPushButton("Install")
    button.setEnabled(False)
    return button


@requirement(
    kinds=("button",),
    counterexample=_a_disabled_button_with_no_reason,
    retires="Grey out, don't hide — and say why",
)
def a_disabled_button_says_why(button) -> None:
    """A greyed-out button explains itself in its tooltip, so it reads as not-yet, not broken.

    "Grey out, don't hide" was only half a rule. The app did both halves at once: seven buttons
    said "Import a video part first" and twelve said nothing at all, for the *same* kind of
    disablement — a step you have not done yet. A control that is present and inert with no
    reason reads as a bug in the tool.

    ``sizing.set_available`` is what keeps this true where the state is recomputed: every one of
    the twelve lived in a handler that flipped ``setEnabled`` on each selection change and had no
    matching place to flip the reason.
    """
    if button.isEnabled():
        return
    assert (button.toolTip() or "").strip(), (
        f"{button.text()!r} is disabled with nothing saying why — use sizing.set_available so "
        "the reason travels with the state"
    )


def _row_holding(button):
    """The horizontal layout ``button`` sits in directly, and its index there; ``(None, -1)``."""
    from PyQt6.QtWidgets import QBoxLayout

    parent = button.parentWidget()
    if parent is None or parent.layout() is None:
        return None, -1
    pending = [parent.layout()]
    while pending:
        layout = pending.pop()
        index = layout.indexOf(button)
        if index >= 0:
            horizontal = isinstance(layout, QBoxLayout) and layout.direction() in (
                QBoxLayout.Direction.LeftToRight,
                QBoxLayout.Direction.RightToLeft,
            )
            return (layout, index) if horizontal else (None, -1)
        for i in range(layout.count()):
            child = layout.itemAt(i).layout()
            if child is not None:
                pending.append(child)
    return None, -1


#: Wider than this is "as wide as it is given" — Qt's QWIDGETSIZE_MAX, less any rounding.
_UNCAPPED_PX = 10_000


def _takes_spare_width(layout, index) -> bool:
    """Whether item ``index`` of ``layout`` would take the row's spare width before a button:
    a stretch, or a widget or nested row that expands — and has no width cap to stop at.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QPushButton

    item = layout.itemAt(index)
    widget = item.widget()
    if widget is not None and (
        not widget.isVisibleTo(widget.window()) or isinstance(widget, QPushButton)
    ):
        return False
    grows = layout.stretch(index) > 0 or bool(
        item.expandingDirections() & Qt.Orientation.Horizontal
    )
    return grows and item.maximumSize().width() >= _UNCAPPED_PX


def _a_row_of_buttons_with_nothing_to_take_the_spare_width():
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

    row = QWidget()
    line = QHBoxLayout(row)
    key = QLabel("key")
    # The tree's row: its one stretched item capped, so the stretch went to the buttons.
    key.setMaximumWidth(200)
    key.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    line.addWidget(key, 1)
    button = QPushButton("Whole tree")
    line.addWidget(button)
    # The button holds its row: a child does not keep its parent alive.
    button._row = row
    return button


@requirement(
    kinds=("button",),
    counterexample=_a_row_of_buttons_with_nothing_to_take_the_spare_width,
)
def a_button_in_a_row_keeps_its_own_width(button) -> None:
    """A button in a horizontal row stays as wide as its text: the row gives its spare width to a stretch, never to the buttons.

    A ``QPushButton`` grows horizontally by default (its policy is ``Minimum``), so a row whose
    only stretched item is capped, or that has none, hands every spare pixel to its buttons. On a
    laptop the tree's "−", "+" and "Whole tree" were ~420px each, a row of three filling the
    pane (user screenshot, 2026-11-09). The fix is a stretch where the row should give — usually
    before right-aligned buttons — or a button that cannot grow (a ``Fixed`` policy), where a
    stretch would make the row itself greedy in the row it sits in. A button meant to span says
    so with a stretch factor of its own.
    """
    from PyQt6.QtWidgets import QSizePolicy

    if not button.isVisibleTo(button.window()) or button.maximumWidth() < _UNCAPPED_PX:
        return
    grow = QSizePolicy.PolicyFlag.GrowFlag.value
    if not button.sizePolicy().horizontalPolicy().value & grow:
        return
    layout, index = _row_holding(button)
    if layout is None:
        return
    # A stretch factor on the button itself is a stated span (the forms' Advanced toggle, a
    # pair of half-width marking buttons), not a row that forgot to give its width elsewhere.
    if layout.stretch(index) > 0 or any(
        _takes_spare_width(layout, i) for i in range(layout.count()) if i != index
    ):
        return
    assert False, (
        f"{button.text()!r} sits in a row with nothing to take its spare width, so it grows "
        "to fill it — add a stretch where the row should give (`addStretch(1)`), or stop the "
        "button growing"
    )


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
#: A table that is genuinely an editor sets this property on itself. One table does
#: (`MetadataPage`'s), and before this existed nothing in the app said so — the read-only rule
#: simply asserted something untrue and passed because the walk never enters edit mode.
EDITOR_PROPERTY = "detail_table_is_an_editor"


def _an_editable_table():
    from PyQt6.QtWidgets import QTableView

    return QTableView()  # Qt's default allows editing on double-click


def _a_table_that_opens_nothing_but_looks_like_it_does():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTableView

    table = QTableView()
    table.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
    return table


def _a_navigable_table_with_no_cue():
    from PyQt6.QtWidgets import QTableView

    table = QTableView()
    table.doubleClicked.connect(lambda _index: None)
    return table


def _opens_something(table) -> bool:
    """Whether a row of ``table`` opens a page — read off the **wiring**, never a flag beside it.

    A cue and a connection declared in parallel live in different files and one of them is
    eventually wrong; CLAUDE.md states this for shared widgets — "a cue is derived from the
    wiring, never set beside it".

    **The subtracted 1 is Qt's, not ours.** A bare ``QTableWidget`` already reports one
    ``doubleClicked`` receiver: its own internal relay that re-emits ``itemDoubleClicked``.
    Measured, on an empty widget with nothing connected to it. Counting that as "the app wired an
    opener" made this predicate a proxy rather than the property — it read `MetadataPage`'s
    *editable grid* as a navigable table and demanded a cue that would have been a lie.
    """
    from PyQt6.QtWidgets import QTableWidget

    if not isinstance(table, QTableWidget):
        return table.receivers(table.doubleClicked) > 0
    own = table.receivers(table.doubleClicked) - 1  # less Qt's itemDoubleClicked relay
    return own + table.receivers(table.itemDoubleClicked) > 0


@requirement(kinds=("table",), counterexample=_an_editable_table)
def a_table_is_read_only_unless_it_declares_itself_an_editor(table) -> None:
    """A table displays unless it says otherwise — `EDITOR_PROPERTY` is how it says so.

    Qt's default opens an editor on double-click, which on a navigable table competes with the
    gesture that opens the row.

    **This rule used to assert that flatly, and it was false.** `MetadataPage` is an editor: its
    `_enter_edit` flips `setEditTriggers` at runtime. The rule passed anyway, because the walk
    only ever builds a page and looks — it never enters edit mode. So the guidelines said "no
    table in this app is an editor" while one was, in a state the guard could not reach. That is
    the fixture-that-never-diverges shape, and the fix is not to widen the rule but to make the
    exception *declare itself*: an editor sets the property, and the pair of transition tests in
    `tests/gui/test_table_sorting.py` covers what it does in the mode this cannot see.

    Declaring is deliberately a property on the widget rather than an allowlist of class names.
    Two dialogs in this app already shared one name (`ExportDialog`), which silently merged every
    exemption keyed by `__name__` — a recorded class here.
    """
    from PyQt6.QtWidgets import QAbstractItemView

    if table.property(EDITOR_PROPERTY):
        return
    assert table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers, (
        f"{table.editTriggers()} — a table is read-only by default, and on a navigable one an "
        f"edit trigger fights the double-click that opens the row. If it is genuinely an "
        f"editor, set the {EDITOR_PROPERTY!r} property so the exception is on the record"
    )


@requirement(kinds=("table",), counterexample=_a_navigable_table_with_no_cue)
def a_table_whose_rows_open_something_says_so(table) -> None:
    """A table whose rows open a page shows a pointing hand over them.

    Half of a pair; the other half is that a table which opens *nothing* must not show the cue.
    Either alone passes with the rule hardcoded the other way, which is why CLAUDE.md asks for
    the pair to be tested rather than the half in front of you.
    """
    from PyQt6.QtCore import Qt

    if not _opens_something(table):
        return
    assert table.viewport().cursor().shape() == Qt.CursorShape.PointingHandCursor, (
        "rows open a page on double-click but the viewport shows the ordinary cursor, so "
        "nothing on screen says the row is clickable"
    )


@requirement(
    kinds=("table",), counterexample=_a_table_that_opens_nothing_but_looks_like_it_does
)
def a_table_that_opens_nothing_shows_no_cue(table) -> None:
    """A table with no row action shows the ordinary cursor — the hand is a promise.

    A selection-only table (a row-select that just enables a button) that shows the pointing
    hand invites a double-click which does nothing.
    """
    from PyQt6.QtCore import Qt

    if _opens_something(table):
        return
    assert table.viewport().cursor().shape() != Qt.CursorShape.PointingHandCursor, (
        "the viewport shows a pointing hand but no doubleClicked receiver is wired, so the cue "
        "promises an action that does not happen"
    )


# --------------------------------------------------------------------------- #
# dialog
# --------------------------------------------------------------------------- #
def _a_dialog_that_cannot_scroll():
    """A dialog with rows straight in its own layout — the shape this rule exists to reject."""
    from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout

    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("a row"))
    layout.addWidget(QLabel("another row"))
    return dialog


@requirement(kinds=("dialog",), counterexample=_a_dialog_that_cannot_scroll)
def a_dialog_scrolls_its_content(dialog) -> None:
    """A dialog's content sits in one scroll area, with its buttons pinned outside it.

    **Reported from the app**: *"windows open with no scroll bar, hiding content."* The dialog it
    was reported on had eight items in its top-level layout and **one** of them scrolled — a
    `PageScroll` around the module's config form. The other seven could not, so they set a floor
    Qt clamps ``resize()`` to: asked for 300px that dialog returned **323**, and on a large
    enough cohort the floor passes the screen and the window grows past the display with its
    buttons under the Dock.

    **Every simpler version of this check passes on the broken dialog.** "Has a scroll area" —
    it had one. "Fits on screen" — measured across all 26 constructible dialogs, *zero* have a
    floor above the 833px available height, because the fixture is an empty experiment; the rule
    would be green forever and red on real data. So the check is structural: the top-level
    layout holds the scrolled body and the buttons, and nothing else that could grow beside them.

    `gui/dialog.py::Dialog.body` is where the rows go and `set_buttons` is where the box goes.
    """
    from PyQt6.QtWidgets import QDialogButtonBox

    from detail.gui.scroll import PageScroll

    layout = dialog.layout()
    assert layout is not None, f"{type(dialog).__name__} has no layout at all"
    stray = []
    scrolls = 0
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if isinstance(widget, PageScroll):
            scrolls += 1
            continue
        if isinstance(widget, QDialogButtonBox):
            continue
        stray.append(type(widget).__name__ if widget is not None else "a nested layout")
    assert scrolls == 1 and not stray, (
        f"{type(dialog).__name__}'s top-level layout holds {stray or 'no scroll area'} beside "
        f"{scrolls} scroll(s) — those cannot scroll, so they set a floor the window cannot go "
        f"below. Put the rows in `Dialog.body()` and the box through `Dialog.set_buttons()`."
    )


def _a_window_whose_first_button_is_remove():
    """A non-modal dialog as Qt leaves it: its push buttons auto-default, the first of them —
    here **Remove** — what Enter clicks."""
    from PyQt6.QtWidgets import QDialog, QLineEdit, QPushButton, QVBoxLayout

    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLineEdit())
    layout.addWidget(QPushButton("Remove"))
    dialog.setModal(False)
    return dialog


@requirement(kinds=("dialog",), counterexample=_a_window_whose_first_button_is_remove)
def a_non_modal_window_has_no_default_button(dialog) -> None:
    """A dialog shown non-modally has no default or auto-default button: Enter in it clicks
    nothing.

    ``QDialog`` makes every push button auto-default and the first of them in the focus chain
    the default, which Enter clicks from any widget that passes Return on — a line edit does,
    after acting on it. In a modal OK / Cancel dialog that is Enter-is-OK, and wanted. In a
    window that stays open beside its page there is no OK: the cluster window's first button was
    **Remove**, so Enter in its label field wrote the label and then removed it (2026-11-09).

    Checked on the shown dialog, not its source: the default is chosen when the dialog is shown,
    and a button that takes focus later becomes it then. `gui/dialog.Dialog` takes both off
    every button of a dialog that is not modal, at show and whenever its content is refreshed.
    Every dialog the sweep builds is shown non-modally, so this holds the base's policy on all of
    them; a modal one, opened with ``exec()``, keeps Qt's Enter-is-OK.
    """
    from PyQt6.QtWidgets import QPushButton

    if dialog.isModal():
        return
    armed = [
        f"{button.text()!r} ({'default' if button.isDefault() else 'auto-default'})"
        for button in dialog.findChildren(QPushButton)
        if button.isDefault() or button.autoDefault()
    ]
    assert not armed, (
        f"{type(dialog).__name__} is open beside its page with {armed}: Enter in it clicks "
        f"that button. Inherit `gui/dialog.Dialog`, which takes default and auto-default off "
        f"every button of a dialog that is not modal."
    )


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #
def _a_label_nobody_can_copy():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel

    label = QLabel("CUDA is unavailable: torch was built without it")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
    return label


@requirement(
    kinds=("label",),
    counterexample=_a_label_nobody_can_copy,
    retires="Selectable text: a dialog's labels",
)
def a_label_can_be_selected_and_copied(label) -> None:
    """Text on screen can be selected, so it can be copied into a bug report.

    Qt labels are not selectable by default. Pages had this from `MainWindow._show`; dialogs
    never did, because a dialog does not pass through the shell — so ~100 labels across ~30
    dialogs could not be copied, and a dialog is where the copyable text usually is, because
    that is where the error message appears.

    **Two mechanisms were wrong before the third was right**, which is the part worth keeping:

    - `sizing.finish_dialog` looked like the shared dialog seam and reached **8 of 44** dialogs.
      A policy living where a surface must *remember* to visit is a policy that gets missed. It
      has since been deleted, every caller having moved to `Dialog.set_buttons`.
    - An application-wide event filter reached everything and **segfaulted**, about one suite run
      in two. `gui/selectable.py` carries the stack; the short version is that an application
      filter is offered posted events bound for destroyed objects, and sip dies building the
      Python wrapper upstream of the filter's own first line.
    - `gui/dialog.py::Dialog` — a base whose `showEvent` applies the policy to *itself*. A dialog
      is alive when it is being shown, so none of the filter's exposure exists.

    A label holding a pixmap or nothing is skipped: there is no text to select.
    """
    from PyQt6.QtCore import Qt

    if not (label.text() or "").strip():
        return
    pixmap = label.pixmap()
    if pixmap is not None and not pixmap.isNull():
        return
    assert (
        label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    ), f"{label.text()[:60]!r} cannot be selected, so nobody can copy it"


# --------------------------------------------------------------------------- #
# The rules that are not swept yet
# --------------------------------------------------------------------------- #
#: Rules that exist in prose and are **not** enforced over every subject, with why. Seeded from
#: the #349 inventory; the guard fails when this **grows**, so the list is a shrinking backlog
#: rather than a place to put things. This is the same shape as
#: `tests/gui/test_ui_thread_blocking.py`'s ALLOWED, which has been emptied of three entries
#: this way.
#:
#: Keyed by the rule as CLAUDE.md words it, so the entry can be found from either end.
#:
#: Guidelines rules that **are** checked, by a guard that is not a requirement in this registry.
#: They were sitting in :data:`NOT_YET_SWEPT` reading as outstanding work.
#:
#: A rule does not have to live here to be enforced. A registry requirement runs over walked
#: *widgets*; some rules are about a registry, an API, or a click sequence, and the right check
#: for those is a sweep or a behavioural test. What was wrong was the bookkeeping: the backlog
#: said fifteen rules were unchecked when four of them had a guard, so "how much is left" was
#: an over-count and the ceiling it ratchets was measuring the wrong thing.
#:
#: Each entry names the test that covers it, and `test_covered_elsewhere_still_covers` fails on
#: a path that has stopped existing — the same claim-with-a-reason shape as every other table
#: here, because an entry nobody checks is how the dialog exemptions rotted nine deep.
COVERED_ELSEWHERE: dict[str, str] = {
    "Tables for lists: sortable headers": (
        "tests/gui/test_table_sorting.py::test_every_table_is_sortable_unless_named_in_sort_exempt"
        " — a sweep over every table the fixture can build, with its own exemption list. The "
        "entry here was blocked on PR #348, which closed; the work landed in #364"
    ),
    "Long operations never block the GUI thread": (
        "tests/gui/test_ui_thread_blocking.py::test_no_unwrapped_slow_seam_outside_allowlist"
        " — a fail-closed source scan over the known-slow seams, with two negative cases"
    ),
    "The rail toggle is one uniform rule": (
        "tests/gui/test_smoke_construct.py::test_click_toggle_is_uniform_across_folders"
        " — drives the real `_on_tree_click` over folders and nested sections"
    ),
    "A recording's id is the tool's, its name is the user's": (
        "tests/api/test_recording_identity.py — an API rule, and Tenet 2 makes the API the "
        "boundary, so a widget walk is the wrong place to ask"
    ),
}

#: :data:`BACKLOG_CEILING` is a **ratchet**: it is lowered as rules migrate and never raised. A
#: PR that adds an entry is a PR that wrote a rule into the guidelines without a check, and has
#: to say so out loud by editing this number.
NOT_YET_SWEPT: dict[str, str] = {
    # --- app-wide; source scans rather than widget checks (PR 5) ------------- #
    # --- table (PR 2) -------------------------------------------------------- #
    "Navigable tables signal a clickable row": (
        "half done: the pointing-hand half of the cue is swept by "
        "`a_table_whose_rows_open_something_says_so` and its pair. The trailing `\u203a` column "
        "is checked at four call sites and nowhere sweeps it"
    ),
    "A table beside a player is selection-driven, not navigable": "spot-checked",
    # --- page and dialog (PR 3) --------------------------------------------- #
    "Pages top-justify": "test_page_body_space.py + layout_checks.py cover several, not all",
    "A widget declares the floor its own contents need": "spot-checked",
    "A folder's page summarises its contents": "needs a predicate for 'summary, not a duplicate'",
    # --- the small kinds (PR 4) --------------------------------------------- #
    "Forms: QFormLayout, tab/enter/esc": "no predicate yet",
    # --- shell / navigation (PR 3) ------------------------------------------ #
    "One launch surface per module": (
        "a registry sweep, not a widget rule — and genuinely unwritten. `test_module_config_widget_builds` proves every module *has* a config widget; nothing asserts it has only one "
        "surface, which is what the rule says"
    ),
    "Credential surfaces offer a way out": "one surface exists; a rule needs a second",
}

#: The ratchet under :data:`NOT_YET_SWEPT`. Lowered when a rule migrates; never raised.
#: **A literal, not ``len(NOT_YET_SWEPT)``.** Derived from the dict it caps, the assertion
#: ``len(NOT_YET_SWEPT) <= BACKLOG_CEILING`` is true by construction and can never fail — a
#: ratchet with no teeth, in the file that exists to give rules teeth. Lower it when a rule
#: migrates; raising it is the visible act of writing a guideline with no check behind it.
BACKLOG_CEILING = 8


# --------------------------------------------------------------------------- #
# package — the rules that are about what the source never does
# --------------------------------------------------------------------------- #
#: Files excused from *every* source scan below, and why. Only `requirements.py`: its job is to
#: *build* the violations every rule must reject. Nothing else belongs here — a file excused for
#: one rule's reason and listed here is excused from every rule written after it, which is how
#: `label_panel.py`, `arena.py` and `__main__.py` sat outside the stylesheet, animation, dialog,
#: playback and split rules for an icon-only reason (review, 2026-11-09). A file with a reason
#: that belongs to one rule goes in that rule's own table (`_ICON_EXEMPT`).
_SCAN_EXEMPT: dict[str, str] = {
    "requirements.py": "holds the counterexamples — it constructs what the rules reject",
}

#: Files excused from `the_package_uses_no_icon_sets` only, and why.
_ICON_EXEMPT: dict[str, str] = {
    "__main__.py": "sets the application's Dock/taskbar icon, which is not a widget icon set",
    "arena.py": "draws a generated colour swatch beside each arena (CLAUDE.md, Icons)",
    "label_panel.py": "draws each label's generated colour swatch in the label list — the "
    "colour that label has on the tree, which its name alone cannot say",
}

#: Qt's animation and effect machinery, by name. "No animations" and "no shadows" are two of the
#: oldest lines in the style guide and nothing has ever checked them; the package uses none of
#: these today, so this holds a rule that is true by everyone's care rather than by construction.
_ANIMATION_NAMES = frozenset(
    {
        "QPropertyAnimation",
        "QVariantAnimation",
        "QAbstractAnimation",
        "QParallelAnimationGroup",
        "QSequentialAnimationGroup",
        "QTimeLine",
        "QEasingCurve",
        "QGraphicsDropShadowEffect",
        "QGraphicsBlurEffect",
        "QGraphicsOpacityEffect",
        "setGraphicsEffect",
    }
)


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
#: Words that start the label of a button which **adds an item** to a list.
#:
#: A prefix test rather than a registry, because the thing being detected is what the button
#: says to the person reading it. A button that acts on the row you have *selected* — Install,
#: Remove, Check access — is not one of these and correctly lives under the table, where it
#: reads as operating on what is above it.
CREATE_WORDS = ("New", "Add", "Import", "Distill")

#: Trailing glyphs that mark a button as navigation rather than an action: `›` opens another
#: page, `↗` opens a browser. A "Distill runs ›" link — the Model manager had one — starts with
#: a create word and creates nothing.
NAVIGATION_GLYPHS = ("\u203a", "\u2197")


def _create_buttons(page) -> list:
    from PyQt6.QtWidgets import QPushButton

    out = []
    for button in page.findChildren(QPushButton):
        text = (button.text() or "").strip()
        if not text or not button.isVisibleTo(page):
            continue
        if text.endswith(NAVIGATION_GLYPHS):
            continue
        if text.lstrip("\uff0b ").startswith(CREATE_WORDS):
            out.append(button)
    return out


def _lists(page) -> list:
    from PyQt6.QtWidgets import QTableView, QTableWidget

    return [
        t for t in page.findChildren((QTableView, QTableWidget)) if t.isVisibleTo(page)
    ]


def _a_page_with_its_create_action_below_the_list():
    from PyQt6.QtWidgets import QPushButton, QTableWidget, QVBoxLayout, QWidget

    from detail.gui import tables

    page = QWidget()
    box = QVBoxLayout(page)
    table = QTableWidget(4, 2)
    # Conformant in every way but the one under test: a counterexample that breaks a *second*
    # rule fails that rule's sweep instead of proving anything about this one.
    tables.fill_width(table)
    box.addWidget(table)
    box.addWidget(QPushButton("New run"))
    page.resize(400, 300)
    page.show()
    page.layout().activate()
    return page


@requirement(
    kinds=("page",),
    counterexample=_a_page_with_its_create_action_below_the_list,
    exempt={
        "MetadataPage.<MetadataPage>": (
            "Import CSV is one button in a mode toolbar — Edit / Save / Cancel / Add column / "
            "Delete column / Import / Export — that operates on the table as a whole rather "
            "than adding a row to it. The recordings are already there; the CSV fills their "
            "columns. Lifting Import out of that row would separate it from Export, which is "
            "not a create action by any reading, and leave a toolbar split across the table "
            "for one word in one label."
        ),
    },
)
def a_page_puts_its_create_action_above_the_list_it_fills(page) -> None:
    """Put a page's New / Add / Import action above the list it fills, never below it.

    A list is the thing on a page that grows. Put the button that adds to it underneath and the
    button moves further away with every item added — until, on the page where it matters most,
    you scroll to reach the control you came for. Reported from a real experiment: "as the run
    list gets long it gets buried and you have to scroll to find this."

    Buttons that act on the **selected** row are a different thing and stay below the table,
    where they read as operating on what is above them.
    """
    tables = _lists(page)
    if not tables:
        return
    if page.layout() is not None:
        # Geometry is its previous value until the layout runs, and a page built for a walk has
        # had no reason to run one. Activating is idempotent and changes no state a sibling rule
        # could read differently.
        page.layout().activate()
    placed = [(t.mapTo(page, t.rect().topLeft()).y(), t) for t in tables]
    for button in _create_buttons(page):
        y = button.mapTo(page, button.rect().topLeft()).y()
        table_y, _ = min(placed, key=lambda pair: abs(pair[0] - y))
        assert y < table_y, (
            f"{button.text()!r} sits {y - table_y}px below the list it fills — put it above, "
            "or the list growing pushes it off the screen"
        )


def _names_used(package, *, exempt=()) -> dict[str, set[str]]:
    """Every identifier each source file *uses*, by file — from the AST, never the text.

    A text search cannot tell `setStyleSheet` in a call from `setStyleSheet` in the docstring
    explaining why nobody calls it, so a rule enforced by `grep` fails the moment somebody writes
    it down. Reading names and attributes out of the parse tree skips strings and comments by
    construction.
    """
    import ast

    used: dict[str, set[str]] = {}
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT or source.name in exempt:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.rsplit(".", 1)[-1])
        used[source.name] = names
    return used


def _a_package_that(
    line: str, *, fusion: bool = True, extra: str = "", entry: str | None = None
):
    """A one-file package containing ``line`` — the smallest thing a source rule must reject.

    ``extra`` is for a rule about a *statement* rather than an expression — a class definition,
    say — which cannot be planted inside the function body ``line`` goes into. ``entry`` replaces
    the entry point's body, for the rule about what the entry point does.
    """
    import pathlib
    import tempfile

    directory = pathlib.Path(tempfile.mkdtemp())
    body = "    apply_style(app)\n" if fusion else "    pass\n"
    if entry is not None:
        body = f"    {entry}\n"
    (directory / "__main__.py").write_text(
        f"def configure(app):\n{body}", encoding="utf-8"
    )
    (directory / "offender.py").write_text(
        f"{extra}def build(widget):\n    {line}\n", encoding="utf-8"
    )
    return directory


def _flat_buttons(package) -> list[str]:
    """``file:line`` of every ``setFlat(...)`` call not passed a literal ``False`` — from the AST."""
    import ast

    found = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setFlat"
                and not (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value is False
                )
            ):
                found.append(f"{source.relative_to(package).as_posix()}:{node.lineno}")
    return found


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that("widget.setFlat(True)"),
)
def a_button_looks_like_a_button(package) -> None:
    """A button has the frame every other button has — nothing in the package calls `setFlat`.

    A flat button reads as text. One shipped as the only way into the full measurements list,
    and the report was "it took me a while to find it and it doesn't look like any of the other
    buttons". The check sat in that panel's own test, and a walk over built widgets cannot reach
    it either — the panel is a module result view, not a page or a dialog. So the rule is about
    what the package never does, and is read from the source.
    """
    found = _flat_buttons(package)
    assert not found, (
        f"setFlat at {found}: a flat button reads as text rather than as a control, and every "
        "other button in the app has a frame"
    )


#: Re-runnable loads that keep their own guard against an older result, and what that guard is.
#: Keyed by ``file:Class.method`` like :data:`_CONFIRMATIONS`.
_SUPERSEDED_ELSEWHERE: dict[str, str] = {
    "measurements.py:MeasurementsPanel.select": (
        "its own `_generation`, which an edit or a selection bumps even when no read starts — "
        "a wider meaning than `supersedes`, which only a newer call moves"
    ),
}

#: Method names that mean "read it again": the shape a second call arrives in while the first is
#: still reading.
_RELOAD_NAMES = ("reload", "_reload", "select", "set_experiment")
_RELOAD_PREFIXES = ("refresh", "_refresh", "_load", "load_")


def _unsuperseded_reloads(package) -> list[str]:
    """``file:Class.method`` of each reload that starts a busy call without ``supersedes=``."""
    import ast

    found = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT or source.name == "busy.py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        scopes = [(None, node) for node in tree.body]
        scopes += [
            (cls.name, node)
            for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for node in cls.body
        ]
        for owner, fn in scopes:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name not in _RELOAD_NAMES and not fn.name.startswith(
                _RELOAD_PREFIXES
            ):
                continue
            for node in ast.walk(fn):
                name = getattr(getattr(node, "func", None), "id", None) or getattr(
                    getattr(node, "func", None), "attr", None
                )
                if (
                    isinstance(node, ast.Call)
                    and name in ("run_busy", "run_progress")
                    and not any(k.arg == "supersedes" for k in node.keywords)
                ):
                    where = f"{owner}.{fn.name}" if owner else fn.name
                    key = f"{source.relative_to(package).as_posix()}:{where}"
                    if key not in _SUPERSEDED_ELSEWHERE:
                        found.append(key)
    return found


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        "pass",
        extra=(
            "class Tab:\n"
            "    def reload(self):\n"
            "        run_busy(self, read, show)\n\n\n"
        ),
    ),
)
def a_reload_supersedes_its_last_call(package) -> None:
    """A load that can be asked for again names what it loads, so the newest request wins.

    Two overlapping busy calls deliver in the order their threads *finish*, so the slower, older
    read lands last and decides the screen. AUDIT's `reload` overlapped up to five times while a
    behavior was chosen and a cutoff set, and a stale "nothing to audit" disabled Start after the
    cutoff had made batches available — 13 of 86 full runs of one test, and a real wrong screen
    for anyone switching quickly. ``run_busy(..., supersedes=name)`` drops every result but the
    newest for that name on that widget; a method named like a reload either passes one or is in
    :data:`_SUPERSEDED_ELSEWHERE` with the guard it keeps instead.

    ``set_experiment`` is a reload too: the distill form read its teachers through one without a
    name, from an experiment combo outside its own overlay, so switching experiments mid-read could
    leave the first experiment's keypoint runs listed under the second.
    """
    found = _unsuperseded_reloads(package)
    assert not found, (
        f"{found} start a busy call that a later call can overtake: pass `supersedes=` naming "
        "what is loaded, or list the method in `_SUPERSEDED_ELSEWHERE` with its own guard"
    )


#: The GUI's thread classes and the two helpers that start one. A call to any of them starts work
#: whose result is delivered back to a widget.
_WORKER_STARTS = frozenset(
    {
        "BackgroundCall",
        "StageProgressWorker",
        "FitProgressWorker",
        "RunProgressWorker",
        "InstallModelWorker",
        "run_busy",
        "run_progress",
    }
)

#: Work whose result accepts a dialog without ``hold_open_while``, and why that is safe. Keyed by
#: ``file:Class.method`` like :data:`_CONFIRMATIONS`. Empty: every site found holds its dialog.
_CLOSES_ON_ITS_OWN: dict[str, str] = {}


def _accepts(node) -> bool:
    """Whether ``node``'s body calls ``.accept()`` on something — closing a dialog as done."""
    import ast

    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "accept"
        for n in ast.walk(node)
    )


def _callback_accepts(callback, scope, methods: dict) -> bool:
    """Whether a result callback — a lambda, a nested def, or ``self.method`` — accepts a dialog."""
    import ast

    if isinstance(callback, ast.Lambda):
        return _accepts(callback)
    if isinstance(callback, ast.Name):
        return any(
            isinstance(n, ast.FunctionDef) and n.name == callback.id and _accepts(n)
            for n in ast.walk(scope)
        )
    if (
        isinstance(callback, ast.Attribute)
        and isinstance(callback.value, ast.Name)
        and callback.value.id == "self"
    ):
        if callback.attr == "accept":
            return True
        method = methods.get(callback.attr)
        return method is not None and _accepts(method)
    return False


def _unheld(scope, methods: dict) -> bool:
    """Whether ``scope`` starts work whose result accepts a dialog, and does not hold it open."""
    import ast

    starts = held = False
    callbacks = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in _WORKER_STARTS:
            starts = True
            if name in ("run_busy", "run_progress"):
                callbacks += node.args[2:3]
                callbacks += [k.value for k in node.keywords if k.arg == "on_error"]
                held |= name == "run_progress" and any(
                    k.arg == "on_cancel" for k in node.keywords
                )
        elif (
            name == "connect"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr in ("done", "failed")
        ):
            callbacks += node.args[:1]
        elif name == "hold_open_while":
            held = True
    return (
        starts
        and not held
        and any(_callback_accepts(cb, scope, methods) for cb in callbacks)
    )


def _dialogs_closing_on_running_work(package) -> list[str]:
    """``file:Class.method`` (or ``file:function``) of each start of work that accepts a dialog
    when it finishes, where the dialog is not held open while it runs."""
    import ast

    found = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        rel = source.relative_to(package).as_posix()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and _unheld(node, {}):
                found.append(f"{rel}:{node.name}")
            elif isinstance(node, ast.ClassDef):
                methods = {
                    m.name: m for m in node.body if isinstance(m, ast.FunctionDef)
                }
                for method in methods.values():
                    if _unheld(method, methods):
                        found.append(f"{rel}:{node.name}.{method.name}")
    return [key for key in found if key not in _CLOSES_ON_ITS_OWN]


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        "pass",
        extra=(
            "class AddPartDialog(Dialog):\n"
            "    def _start(self):\n"
            "        worker = StageProgressWorker(self._copy, receiver=self)\n"
            "        worker.done.connect(lambda _r: self.accept())\n"
            "        worker.start()\n\n\n"
        ),
    ),
)
def a_dialog_that_runs_work_owns_its_close(package) -> None:
    """A dialog whose answer comes from background work stays open until that work has answered.

    Esc, the window's close button and a Cancel wired to ``reject`` close a dialog as cancelled,
    and the work it started goes on regardless. The launch dialogs closed during validation and
    the run started anyway, with the page told nothing had; Add video part closed on Cancel and
    added the part behind a page still listing the old ones; the export dialog, the behavior-label
    import and the skeleton save had the same shape (#35). Work whose result *accepts* the dialog
    is the shape: the dialog's answer depends on it, so closing first answers wrongly. Pass the
    worker to ``Dialog.hold_open_while`` — a close request then stops the work or says why it
    cannot — or start it with ``run_progress(..., on_cancel=...)``, which cancels it with the
    window; or list the site in :data:`_CLOSES_ON_ITS_OWN` with the reason it is safe.
    """
    found = _dialogs_closing_on_running_work(package)
    assert not found, (
        f"{found} accept a dialog when background work finishes, and nothing stops Esc or the "
        "window's close from closing it first as cancelled: hold it with "
        "`hold_open_while(worker, on_close_request)`"
    )


#: Palette writes that pass a literal colour, and why that one is not a status colour. Keyed by
#: ``file:Class.method`` like :data:`_CONFIRMATIONS`, for the same reason.
_LITERAL_PALETTE_COLOURS: dict[str, str] = {}


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        "widget.palette().setColor(QPalette.ColorRole.WindowText, QColor(200, 150, 0))"
    ),
    retires="Colour: greyscale base",
)
def a_palette_colour_is_declared_rather_than_spelled_out(package) -> None:
    """A widget's palette takes its colour from ``gui/status_color.py``, not from an RGB triple.

    The guideline is "greyscale base; colour reserved for genuine status", and the package has
    one module holding the four status colours. What nothing checked was whether a *palette*
    write went through it — and five did not, in the way this always fails:

    - ``fit.py`` set ``QColor(40, 140, 40)`` / ``QColor(225, 105, 45)``, byte-identical copies of
      ``READY`` and ``ERROR``, with a comment naming the private constants they were copied from
      before those moved into the status module.
    - **Three** call sites set ``QColor(200, 150, 0)`` and called it amber, while the declared
      ``WARN`` is ``QColor(170, 110, 0)``. The app had two ambers, and which one a warning got
      depended on which screen it was on.

    Nothing was broken, no test could fail, and the two ambers were never going to converge on
    their own — a copied colour has no way to learn that the original changed. That is the
    "shared object nobody shares" class, in the one place where the drift is visible to a user
    and invisible to a reviewer, because ``(200, 150, 0)`` and ``(170, 110, 0)`` read the same
    in a diff.

    **A literal is what is flagged, not colour itself.** Deriving from the palette
    (``palette.color(Base)``, ``background.darker(112)``) is how a surface tints itself against
    whatever theme it is in, and passing a colour *in* — a track's hue, a judgement's colour — is
    how the Okabe-Ito palette reaches a widget. Both stay. What cannot stay is an RGB triple
    typed at the call site, because that is the one spelling that cannot be changed in one place.
    """
    import ast

    offenders = []
    for source in sorted(package.rglob("*.py")):
        if source.name in ("requirements.py", "status_color.py"):
            continue
        rel = source.relative_to(package).as_posix()
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        owner = _owners(tree)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setColor"
            ):
                continue
            literal = any(
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Name)
                and arg.func.id == "QColor"
                and all(isinstance(a, ast.Constant) for a in arg.args)
                and arg.args
                for arg in node.args
            )
            if not literal:
                continue
            key = f"{rel}:{owner.get(node, '<module>')}"
            if key not in _LITERAL_PALETTE_COLOURS:
                offenders.append(key)
    assert not sorted(set(offenders)), (
        f"{sorted(set(offenders))} set a palette colour from an RGB triple. Use a name from "
        f"`gui/status_color.py` so the app has one amber rather than two, derive it from the "
        f"palette, or take it as a parameter. If the colour is genuinely not a status, declare "
        f"it in `_LITERAL_PALETTE_COLOURS` with the reason."
    )


#: The one module allowed to build and colour a pyqtgraph plot by hand: the adapter every other
#: plot is built through.
_PLOT_THEME_MODULE = "plot_theme.py"
#: pyqtgraph's own styling entry points. Any of them outside the adapter is a second copy of the
#: chart design — a background, a pen or a brush that does not follow the theme.
_PLOT_STYLING_CALLS = frozenset({"PlotWidget", "mkPen", "mkBrush"})


def _plot_styling_offenders(package) -> list[str]:
    """``file:line call`` for every plot styled outside ``plot_theme`` — the rule's whole test."""
    import ast

    offenders = []
    for source in sorted(package.rglob("*.py")):
        if source.name in ("requirements.py", _PLOT_THEME_MODULE):
            continue
        rel = source.relative_to(package).as_posix()
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            # `setBackground` only with a literal: a table item's `setBackground(QBrush(...))`
            # is a cell, not a plot, and a literal is exactly what `setBackground("w")` was.
            literal_background = (
                name == "setBackground"
                and node.args
                and isinstance(node.args[0], (ast.Constant, ast.Tuple))
            )
            if name in _PLOT_STYLING_CALLS or literal_background:
                offenders.append(f"{rel}:{node.lineno} {name}")
    return offenders


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        'plot = pg.PlotWidget(); plot.setBackground("w")'
    ),
)
def a_plot_takes_its_colours_from_the_chart_theme(package) -> None:
    """Build a plot with `plot_theme.themed_plot` and colour its marks by role, never by literal.

    The rule this replaces was "a plot is on white": `setBackground("w")` at eleven construction
    sites, each with a comment saying every other plot did the same. In a dark window that was a
    row of lit panels with grey tick labels on them, and the eleven copies could only be changed
    one at a time — the same "shared object nobody shares" shape as the two ambers above, spelled
    in pyqtgraph. Pens had drifted the same way: the calibration curve drew ``(0, 114, 178)``
    while two sites beside it took ``PALETTE_OKABE_ITO[3]`` as "the same blue", which a reorder
    of the palette would have made three colours without failing anything.

    ``themed_plot`` sets the surface, the axis ink, outward ticks and the SI-prefix suppression,
    and re-themes on a light/dark switch; ``plot_theme.pen`` / ``brush`` take a theme colour by
    role (``theme.identity(i)``, ``theme.neutral``, ``theme.handle``). The design itself is
    ``core/chart_theme.py``.
    """
    offenders = _plot_styling_offenders(package)
    assert not offenders, (
        f"{offenders} build or colour a plot outside `gui/plot_theme.py`, so it will not follow "
        "the chart theme. Build it with `plot_theme.themed_plot()` and colour it with "
        "`plot_theme.pen` / `plot_theme.brush` and a theme role."
    )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        'label.setFont(QFont("Monospace"))', extra="from PyQt6.QtGui import QFont\n\n\n"
    ),
    retires="Typography: system defaults, monospace for ids",
)
def typography_comes_from_the_shared_helpers(package) -> None:
    """The monospace face and every size step come from ``gui/sizing.py``, not from a call site.

    Two halves of one guideline — *"system defaults; ``QFont("Monospace")`` with
    ``StyleHint.Monospace`` for ids and paths; no custom point sizes except a page's status
    line"* — and both had drifted the same way, because the helper lived somewhere a leaf could
    not reach.

    - **Five copies of a three-line function.** ``sizing.mono_font`` is the canonical one and its
      own docstring already said *"``pages.MONO`` and two local copies predate this leaf"*. By
      the time this rule was written there were four, in ``pages``, ``export``,
      ``behavior_query`` and ``modules/analysis_query``. All faithful — which is the trap: five
      correct copies are indistinguishable from one shared function until the day one changes.
    - **Three spellings of "a heading".** The guidelines said *bold + one size larger*,
      ``pages._status_label`` did **+2**, and ``audit_tab`` hand-rolled **+1** twice. The helper
      was in ``pages.py``, the 5000-line hub that imports the module registry, so ``audit_tab``
      — a leaf — could not call it. Identical to ``finish_dialog``, which reached 8 of 44 dialogs
      for the same reason and became a base class.

    A size step is still a *choice* (a page's status line and a panel's intro heading are
    legitimately different weights) — ``emphasis_font(widget, steps=...)`` takes it as a
    parameter. What the rule rejects is deciding it privately at a call site, which is the one
    spelling that cannot be changed in one place.
    """
    offenders = {
        f: sorted(names & {"setPointSize"})
        for f, names in _names_used(package).items()
        if "setPointSize" in names and f != "sizing.py"
    }
    assert not offenders, (
        f"{offenders} set a point size directly. Use `sizing.emphasis_font(widget, steps=...)` "
        f"so a heading is one gesture rather than a number each caller picks — the guidelines, "
        f"`_status_label` and `audit_tab` said one size larger, two larger and one larger."
    )
    # `_names_used` reads *identifiers*, and the face is a **string literal** — so a check
    # written against it matched nothing at all, in the counterexample and in the package alike.
    # Caught by `test_every_requirement_rejects_its_counterexample`, which is the whole reason a
    # rule ships with one.
    import ast

    mono = set()
    for source in sorted(package.rglob("*.py")):
        if source.name in ("requirements.py", "sizing.py"):
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "QFont"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "Monospace"
            ):
                mono.add(source.relative_to(package).as_posix())
    assert not sorted(mono), (
        f"{sorted(mono)} name the Monospace face directly. Call `sizing.mono_font()` — five "
        f"copies of those three lines existed, all correct, which is exactly why nothing failed."
    )


def _an_image_view_that_cannot_be_unzoomed():
    """A zoomable view whose Fit control has been taken away."""
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.image_view import ImageView

    view = ImageView(zoomable=True)
    for button in view.findChildren(QPushButton):
        button.setParent(None)
    return view


def _an_image_view_that_zooms_without_claiming_the_wheel():
    """A zoomable view that never set `OWNS_WHEEL` — the shape this rule rejects."""
    from detail.gui.image_view import ImageView
    from detail.gui.scroll import OWNS_WHEEL

    view = ImageView(zoomable=True)
    view.setProperty(OWNS_WHEEL, False)
    return view


@requirement(
    kinds=("image",),
    counterexample=_an_image_view_that_zooms_without_claiming_the_wheel,
    retires="Images: QLabel + QPixmap, no zoom/pan",
)
def an_image_view_that_zooms_owns_the_wheel(view) -> None:
    """A view that zooms on the wheel claims the wheel; one that does not, does not.

    The pair, because either half alone passes with the rule hardcoded the other way.
    `PageScroll._page_takes_wheel` walks up from the event's widget looking for `OWNS_WHEEL`, so
    a zoomable view without it never zooms — the page scrolls out from under the gesture — and a
    *plain* view that claims it becomes a hole the page cannot be scrolled through, which is the
    recorded `tests/gui/test_scroll.py` class.

    **This rule replaces one that was deleted rather than swept.** The guideline said *"Images:
    QLabel + QPixmap for static display. No zoom/pan in v1"*, and #337 made it false: zoom was
    added to every image in the tool, at six surfaces at once. A guard enforcing
    the old rule would have been a guard against the feature. That is the retirement rule
    (CLAUDE.md step 6) in its first use — a rule whose class has become impossible goes, and
    what replaces it is what is checked now.
    """
    from detail.gui.scroll import OWNS_WHEEL

    claims = bool(view.property(OWNS_WHEEL))
    if view.zoomable:
        assert claims, (
            "this view zooms on the wheel but does not claim OWNS_WHEEL, so `PageScroll` takes "
            "the wheel first and the page scrolls instead of the picture zooming"
        )
    else:
        assert not claims, (
            "this view does not zoom but claims OWNS_WHEEL, so it is a hole the page cannot be "
            "scrolled through — the wheel goes nowhere over it"
        )


@requirement(kinds=("image",), counterexample=_an_image_view_that_cannot_be_unzoomed)
def a_zoomable_image_view_offers_a_way_back(view) -> None:
    """A view that can be moved out of frame ships a visible way back.

    `tests/gui/test_scatter_navigation.py` records the 3-D scatter, which could be rotated off
    screen with no way to recover it — a capability that was documented, implemented and
    unreachable. The same applies to any view a user can pan: the gesture that leaves is not
    enough on its own.

    The control belongs to the **view**, not to each host, for the reason
    `tests/gui/test_spatial_map.py` records: a capability added to a shared widget as something
    the embedder wires reaches the surface being worked on and no other. Four hosts today.
    """
    from PyQt6.QtWidgets import QPushButton

    if not view.zoomable:
        return
    assert view.findChild(QPushButton) is not None, (
        "this view zooms and pans but ships no Fit control, so a picture dragged off centre "
        "cannot be recovered"
    )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that('widget.setStyleSheet("color: red")'),
    retires="Stylesheets: zero QSS",
)
def the_package_writes_no_stylesheets(package) -> None:
    """The GUI uses zero QSS — shapes, palette colours and widget options instead.

    "Stylesheets: none. Zero QSS" has been in the style guide since the GUI was written, and the
    package genuinely has none. What it did not have is anything stopping the first one: a
    stylesheet is the cheapest way to make one screen look different, which is exactly why a
    house style that bans them has to be checked rather than remembered.
    """
    offenders = [
        f for f, names in _names_used(package).items() if "setStyleSheet" in names
    ]
    assert not offenders, (
        f"{', '.join(offenders)} calls setStyleSheet; the house style is zero QSS — use a "
        "QFrame shape, a palette colour, or a built-in widget option"
    )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that('button.setIcon(QIcon("save.svg"))'),
    retires="Icons: none, outside buttons",
)
def the_package_uses_no_icon_sets(package) -> None:
    """No icon sets, SVG or themes — a control is named by its text.

    The widget walk checks the buttons it can reach; this reaches the ones it cannot. An icon on a
    `QListWidgetItem` or a tree node is not a `QWidget`, and no walk over widgets will ever see
    it, so the two rules are the same house style checked from opposite ends.

    Three uses are legitimate and named in `_ICON_EXEMPT` — this rule's own table, not
    `_SCAN_EXEMPT`, so the excuse reaches no other rule: the application's Dock/taskbar icon,
    which is not a control, and the arena and label lists' generated colour swatches, which carry
    the same information as the colour they stand for.
    """
    offenders = [
        f
        for f, names in _names_used(package, exempt=_ICON_EXEMPT).items()
        if "setIcon" in names or "QIcon" in names
    ]
    assert not offenders, (
        f"{', '.join(offenders)} sets an icon; the house style is text labels only. If the icon "
        "carries information no text can (a colour swatch), say so in _ICON_EXEMPT"
    )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that('QPropertyAnimation(widget, b"geometry")'),
    retires="No rounded corners, shadows, or animations",
)
def the_package_animates_nothing(package) -> None:
    """Nothing animates and nothing casts a shadow — the GUI is flat and instant.

    An animation is the one style break that also costs correctness: a widget mid-transition is
    not in the state a test asserts about, and this suite's recorded classes already hold three
    ways an unlaid-out widget lies about itself. Flat and instant keeps "what the code says" and
    "what is on screen" the same thing.
    """
    offenders = {
        f: sorted(names & _ANIMATION_NAMES)
        for f, names in _names_used(package).items()
        if names & _ANIMATION_NAMES
    }
    assert not offenders, (
        f"animation or effect machinery in {offenders}; the house style is no animations, no "
        "shadows, no rounded corners"
    )


#: Every `QMessageBox.question` in the package, and the effect the user cannot take back if they
#: say Yes. **A declared inventory rather than a text match on the message.** The rule is about
#: what the prompt *guards*, and the only machine-readable evidence of that is at the call site,
#: not in the wording — a rule keyed on the words would go green on a reworded message and red on
#: an innocent one.
#:
#: Keyed by `file:Class.method`, not by the bare method name: eleven of these are called `_delete`
#: on eleven different pages, and an allowlist keyed by label rather than identity is a recorded
#: class in this repo.
#:
#: The house rule said *"only for destructive actions (delete a run, overwrite a experiment)"* and
#: was written in the GUI's first mockup commit, before any of these existed. Four of them are not
#: deletes and are right anyway — they guard an effect the user cannot see afterwards, which is
#: the same reason a delete is guarded. They are listed here with what that effect is, so the
#: exception is arguable rather than invisible; the one genuine nag is marked as such.
_CONFIRMATIONS: dict[str, str] = {
    # --- deletes: the case the rule was written for -------------------------- #
    "comparison.py:ComparisonDetailPage._delete": "deletes a comparison and everything it wrote",
    "pages.py:AnalysisRunDetailPage._delete": "deletes an analysis run and its outputs",
    "pages.py:DatasetDetailPage._delete": "deletes a dataset and its images/labels",
    "pages.py:DistillRunDetail._delete": "deletes a distill run",
    "example_set_picker.py:ExampleSetPicker._delete_set": "deletes an example set, its frames and marks",
    "keypoints_manager.py:KeypointsManagerPage._delete_selected": "deletes a library skeleton nothing uses",
    "keypoints_manager.py:SkeletonEditorDialog.reject": "closing the skeleton editor with unsaved edits: Discard or Cancel",
    # A frame with marks on it: the clicks are gone with it (the PNG stays for any run that
    # lists it, which is why the delete is of the row, not the file).
    "keypoint_labeling.py:KeypointLabelingDialog._remove": "drops a marked example frame and its marks",
    # The exported files are in the user's own folder and are *not* touched — which is exactly
    # why this asks: "delete export" reads like it removes them, and the prompt is where that is
    # corrected before the click rather than discovered after it.
    "pages.py:ExportRunDetailPage._delete": "forgets an export run's record",
    "pages.py:ExperimentOverviewPage._delete_experiment": "deletes an experiment and everything in it",
    "pages.py:GlobalModelDetailPage._delete": "deletes a model and its weights",
    "pages.py:GlobalModelsPage._remove_imported": "removes an imported model from the catalog",
    "pages.py:ModelDetailPage._delete": "deletes a model and its weights",
    "pages.py:ModelManagerPage._delete_selected": "deletes a model and its weights",
    "pages.py:RecordingDetailPage._delete_recording": "deletes a recording and all its runs",
    "pages.py:RecordingDetailPage._delete_stream": "deletes a stream and its files",
    "pages.py:RunDetailPage._delete": "deletes a run and its outputs",
    "pages.py:ToolRunDetailPage._delete": "deletes a tool run",
    # --- overwrites: a delete wearing a different verb ------------------------ #
    "pages.py:HomePage._on_import_failed": "replaces an existing experiment with an imported one",
    "streams.py:AddStreamDialog._save_preset": "replaces a saved setup of the same name",
    "editing.py:confirm_leave": "discards unsaved edits the page is holding",
    "arena.py:SceneGeometryDialog._close": "discards unsaved scene-geometry edits",
    "pages.py:_confirm_and_cancel": "kills a running worker; its text comes from the caller",
    "animation_export.py:confirm_replace": (
        "writes an export over files of the same name already in the folder (#29); asked "
        "where the save dialog could not ask itself"
    ),
    # --- not deletes, and right anyway: an effect with nowhere else to appear -- #
    "launch.py:_covered_subset": (
        "runs on fewer recordings than the user selected. Silently narrowing a run is the "
        "irreversible part — nothing downstream says which ones were dropped"
    ),
    "pages.py:ExperimentRecordingsPage._import_bundle": (
        "imports a whole bundle, and is the only surface that shows the import warnings"
    ),
    "pages.py:ToolRunPage._run": (
        "starts an unbounded segmentation that 'can take hours'; the cost is invisible until "
        "it is already being paid"
    ),
}

#: Dialogs deliberately opened non-modally, and why. A run config is modal so the window it
#: launches from cannot be touched mid-launch; a *report* is not a launch.
_NON_MODAL_DIALOGS: dict[str, str] = {
    "clip_pane_window.py": (
        "the enlarged Masks / Keypoints pane (`open_pane_window`) is a companion view of the "
        "clip window, not a config: it follows that window's ticks live, so the clip window has "
        "to stay usable beside it; it is the clip window's child and closes with it"
    ),
    "clip_popup.py": (
        "the clip window (`open_clip_window`) is for watching a clip while the page that opened "
        "it stays in use — the next double-click on that page shows its clip in the same "
        "window, which a modal window would make impossible (11-9 meeting); it is the page's "
        "child and closes with it"
    ),
    "cluster_window.py": (
        "a cluster's pop-out window (`ClusterWindows.open`) sits beside INTERPRET while the "
        "tree is used to open more of them, one per cluster, and labels through the page's own "
        "queue; it is the page's child and closes with it (11-9 meeting)"
    ),
}


def _DIALOG_CLASS_NAMES(package) -> set:
    """Every class in the package that subclasses something ending in ``Dialog``.

    Collected package-wide because a dialog is defined in one module and opened from another.
    """
    import ast

    names = set()
    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(b, ast.Name) and b.id.endswith("Dialog") for b in node.bases
            ):
                names.add(node.name)
    return names


def _owners(tree) -> dict:
    """Every AST node -> the ``Class.method`` that encloses it.

    Built top-down rather than by searching upward, so a nested definition gets the full path
    and two same-named methods on different classes stay distinguishable.
    """
    import ast

    out: dict = {}

    def walk(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                for inner in ast.walk(child):
                    out[inner] = name
                walk(child, f"{name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _calls_named(package, attr: str, on: str | None = None) -> dict:
    """``file:Class.method`` -> count, for every call to ``attr`` (optionally on ``on``).

    From the AST, for the reason :func:`_names_used` gives: the docstring explaining why nobody
    calls a thing must not read as a call to it.

    Only `requirements.py` is skipped, because its job is to construct the violations rules
    reject. `_SCAN_EXEMPT` once excused icon-only files here too, and hid a real confirmation in
    `arena.py` from the confirmation rule — an exemption inherited by a rule it was never written
    for. That table now holds `requirements.py` alone, and the icon excuses live in
    `_ICON_EXEMPT`.
    """
    import ast

    found: dict = {}
    for source in sorted(package.rglob("*.py")):
        if source.name == "requirements.py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        owner = _owners(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == attr):
                continue
            if on is not None and not (
                isinstance(func.value, ast.Name) and func.value.id == on
            ):
                continue
            rel = source.relative_to(package).as_posix()
            key = f"{rel}:{owner.get(node, '<module>')}"
            found[key] = found.get(key, 0) + 1
    return found


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        'QMessageBox.question(widget, "Rename", "Rename this?")'
    ),
    retires="Confirmation dialogs only for destructive actions",
)
def a_confirmation_guards_something_the_user_cannot_undo(package) -> None:
    """A Yes/No prompt guards an effect the user cannot take back — and says which, here.

    Qt makes a confirmation the cheapest thing to add, and every one added to an action that is
    cheap and reversible teaches people to dismiss the ones that are not. That is the whole cost,
    and it is paid on the *destructive* prompts, not the harmless one.

    **Checked as a declared inventory, not by matching the message text.** What a prompt guards
    lives at the call site; the wording is free text that a rule keyed on it would follow into
    nonsense — green on a reworded delete, red on an honest rename. So every call is declared
    with its effect, a new one fails until somebody writes down what it protects, and the
    declarations are readable as the list of every question this app asks.

    The rule it replaces said "only for destructive actions" and was written in the GUI's first
    mockup commit, before a single one of these call sites existed. Four turned out not to be
    deletes and to be right regardless — they guard an effect with nowhere else to appear, which
    is the same reason a delete is guarded rather than a different one.
    """
    sites = _calls_named(package, "question", on="QMessageBox")
    undeclared = sorted(set(sites) - set(_CONFIRMATIONS))
    assert not undeclared, (
        f"undeclared confirmation(s): {undeclared}. A Yes/No prompt is only for an effect the "
        f"user cannot take back — add it to `_CONFIRMATIONS` with what that effect is, or drop "
        f"the prompt. A confirmation on something cheap and reversible teaches people to dismiss "
        f"the ones that are not."
    )
    stale = sorted(set(_CONFIRMATIONS) - set(sites))
    assert not stale, (
        f"`_CONFIRMATIONS` names call site(s) that no longer exist: {stale}. A declaration that "
        f"declares nothing is the shape that let nine dialog exemptions rot."
    )


@requirement(
    kinds=("package",),
    # Constructs a real dialog, because the rule resolves by construction rather than by the
    # name of the variable: a bare `dialog.show()` is no longer a violation and must not be
    # offered as one, or the counterexample stops proving the rule can fail.
    counterexample=lambda: _a_package_that(
        "d = MyDialog()\n    d.show()",
        extra="class MyDialog(Dialog):\n    pass\n\n\n",
    ),
    retires="Run config is a modal dialog; edits are inline",
)
def a_dialog_opens_modally(package) -> None:
    """A dialog is opened with ``exec()``. "Do something" stays distinct from "edit something".

    `QDialog.show()` opens it non-modally, so for a *config* dialog the window it was launched
    from stays live underneath: the run can be started from a form whose experiment changed
    behind it, and a second copy of the same dialog can be opened on top of the first.

    **Resolved by construction, not by variable name.** The first version of this matched
    ``.show()`` on anything called ``dialog``, which is a rule about spelling — it would miss
    ``d.show()`` and ``self._explore.show()`` and pass forever. This one collects the classes in
    the package that actually subclass a dialog, then flags ``show()`` on a value one of them was
    constructed into. Narrower to fool, and it fails on the rename that the name-matching version
    was blind to.

    The package already did this everywhere, which is the point of writing it down: the rule sat
    in the style guide with nothing checking it, and the one deliberate exception carried its
    reason in a code comment where no sweep could see it.
    """
    import ast

    # Once, before the loop. Computed inside it this re-parsed the whole package per file —
    # quadratic, and it tripped the suite's 120s per-test watchdog on a package this size.
    package_dialogs = _DIALOG_CLASS_NAMES(package)
    offenders = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        rel = source.relative_to(package).as_posix()
        if rel in _NON_MODAL_DIALOGS:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        dialog_classes = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(b, ast.Name) and b.id.endswith("Dialog") for b in node.bases
            )
        }
        # A dialog built in one module is opened in another, so the names a *call* can refer to
        # are the package's dialogs, not this file's.
        known = package_dialogs | dialog_classes
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            built = set()
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in known
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            built.add(target.id)
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                ):
                    continue
                if node.func.attr != "show":
                    continue
                receiver = node.func.value
                opened_here = (
                    isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Name)
                    and receiver.func.id in known
                )
                if opened_here or (
                    isinstance(receiver, ast.Name) and receiver.id in built
                ):
                    offenders.append(f"{rel}:{func.name}")
    assert not sorted(set(offenders)), (
        f"{sorted(set(offenders))} open a dialog with `show()`. A dialog is modal — `exec()` — so "
        f"the surface it launched from cannot change underneath it. If it is genuinely a report "
        f"rather than a config, declare it in `_NON_MODAL_DIALOGS` with the reason."
    )


def _a_scrollbar_with_arrows():
    """A bar under plain Fusion — the style the tool's replaces — which draws both arrows."""
    from PyQt6.QtWidgets import QScrollBar, QStyleFactory

    bar = QScrollBar()
    bar._plain = QStyleFactory.create("Fusion")  # the style is not owned by the bar
    bar.setStyle(bar._plain)
    bar.resize(18, 200)
    return bar


@requirement(kinds=("scrollbar",), counterexample=_a_scrollbar_with_arrows)
def a_scrollbar_has_no_arrow_buttons(bar) -> None:
    """A scrollbar has no arrow buttons: one look for every bar in the tool, the handle over the
    whole track.

    Checked on where the style puts the buttons rather than on a picture: an empty rect is what
    makes Fusion draw none, and it is the same answer for a page's bar, a table's, the cluster
    tree's and a zoom overview's. A bar given its own style — or an app started without
    `apply_style` — has them back, and fails here.
    """
    from PyQt6.QtWidgets import QStyle, QStyleOptionSlider

    # From the bar's public state: `initStyleOption` is protected, and most bars in a page are
    # a scroll area's own, created by Qt rather than Python, where PyQt will not call it.
    option = QStyleOptionSlider()
    option.initFrom(bar)
    option.orientation = bar.orientation()
    option.minimum, option.maximum = bar.minimum(), bar.maximum()
    option.sliderPosition = option.sliderValue = bar.value()
    option.pageStep, option.singleStep = bar.pageStep(), bar.singleStep()
    for part in (
        QStyle.SubControl.SC_ScrollBarAddLine,
        QStyle.SubControl.SC_ScrollBarSubLine,
    ):
        rect = bar.style().subControlRect(
            QStyle.ComplexControl.CC_ScrollBar, option, part, bar
        )
        assert rect.isEmpty(), (
            f"this scrollbar lays out an arrow button ({part.name}) — it is not under the "
            "tool's style (gui/style.py); every bar is one look, with no arrows"
        )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that("pass", entry='app.setStyle("Fusion")'),
    retires="Qt style is Fusion",
)
def the_app_installs_its_own_style(package) -> None:
    """The application installs the tool's style (`gui/style.py`): Fusion, one scrollbar look.

    The only rule here that asserts a **presence**, and deliberately so. An absence rule is
    unfalsifiable when the thing was never there — §25's doc anchors are the recorded case, where
    a typo'd anchor satisfied "this string must be gone" forever, and read correctly in review.
    "The style is installed" fails the day the line is deleted, which is the failure worth having:
    without it the app takes each platform's native styling, and every layout number measured in
    this suite stops describing what the user sees.

    `apply_style`, not `setStyle("Fusion")`: plain Fusion draws a scrollbar's arrow buttons, and
    the tool's scrollbars — page, tables, the tree's levels and the zoom overviews — are one look
    without them only because one style lays them out (`a_scrollbar_has_no_arrow_buttons`).
    """
    import ast

    entry = package / "__main__.py"
    assert (
        entry.is_file()
    ), f"{package} has no __main__.py to configure the application in"
    tree = ast.parse(entry.read_text(encoding="utf-8-sig"))
    installed = any(
        isinstance(node, ast.Call)
        and (
            getattr(node.func, "id", None) == "apply_style"
            or getattr(node.func, "attr", None) == "apply_style"
        )
        for node in ast.walk(tree)
    )
    assert installed, (
        f"{entry.name} never calls apply_style(app) (gui/style.py); without it the app takes "
        "plain Fusion or each platform's native styling, with arrow buttons on its scrollbars"
    )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        "pass", extra="class Ask(QDialog):\n    pass\n"
    ),
)
def a_dialog_inherits_the_shared_base(package) -> None:
    """A dialog subclasses `gui.dialog.Dialog`, never `QDialog` — that is where policy lives.

    The widget rules sweep every dialog the walk can *build*, and 28 of 44 need context it does
    not have, so a new dialog that skipped the base could sit uncovered indefinitely. This half
    needs no fixture: it reads the class statement.

    It is also what makes the base worth having. A base nobody is required to inherit is a base
    that covers the dialogs written before the rule and none of the ones written after — which is
    exactly how `sizing.finish_dialog` ended up reaching 8 of 44.
    """
    import ast

    offenders = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                # `Dialog` itself is the one class that may: it *is* the base.
                if (
                    isinstance(base, ast.Name)
                    and base.id == "QDialog"
                    and node.name != "Dialog"
                ):
                    offenders.append(f"{source.name}::{node.name}")
    assert not offenders, (
        f"{offenders} subclass QDialog directly; inherit `detail.gui.dialog.Dialog` so the "
        "dialog-wide policies (selectable text) apply without each dialog remembering them"
    )


#: Every class in `detail/gui/` that steps frames on a timer, and why it may. Asserted as an
#: **equality**, not a ceiling: a new player fails for not being listed, a deleted one fails for a
#: stale entry. "At most four" would be nearly vacuous with four present, and says nothing about
#: *which* four.
_PLAYERS: dict[str, str] = {
    "FrameScrubber": "the one place. Every other frame surface asks this one",
    "_ClipPlayer": (
        "**not playback.** Its loop is gone — the scrubber plays now. What is left is the "
        "edge-walk timer, which slides the *window* the scrubber spans when the playhead is held "
        "against either end, and moves the frame only as a consequence of the window moving "
        "underneath it. The scan cannot tell those apart from the call it makes, and narrowing "
        "it to try would be a predicate tuned to one caller; the honest answer is an entry here "
        "saying which kind of timer this is"
    ),
    "QuickPreview": (
        "**permanent** (and was `_PlayingCell` until it moved somewhere both strips could "
        "reach it). It indexes a list of already-decoded frames rather than calling a "
        "decoder, and its interval is coupled to a stride, so a bout longer than 120 frames "
        "plays over its real wall-clock duration at reduced temporal resolution. A "
        "decoder-backed player has nowhere to put either. Folding it would be the recorded "
        "mistake of imposing one implementation's constraint on all of them"
    ),
    "LiveMontage": (
        "**permanent**, for the reason `QuickPreview` is: a grid of every clip of a cluster, all "
        "playing at once, each on its own clock. `FrameScrubber` is one decoder and one timer "
        "per surface; forty tiles would be forty of each. One timer advances every visible "
        "tile's offset and asks a preview reader for a picture drawn at tile size off the UI "
        "thread — no decoder, no speed menu, no step controls, by design"
    ),
}

#: What a slot has to do for its timer to count as playback: move which frame is shown. Measured
#: against the package rather than guessed — every class here that owns a `QTimer` at all was
#: listed, and these four calls separate the four players from the six debounce, reveal,
#: select-all and hold timers exactly.
_ADVANCES_A_FRAME = ("set_frame", "_apply(", "_show_frame")


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        "pass",
        extra=(
            "class Player:\n"
            "    def __init__(self):\n"
            "        self._t = QTimer(self)\n"
            "        self._t.timeout.connect(self._advance)\n"
            "    def _advance(self):\n"
            "        self._scrubber.set_frame(1)\n"
        ),
    ),
)
def playback_lives_in_one_place(package) -> None:
    """Only a listed class steps frames on a timer; every other surface asks the frame widget.

    Twelve surfaces showed video and none of them could play it, while three classes each carried
    a private copy of the same loop. The copies were not the problem in themselves — they are
    *why* the twelve went without, because a player that lives in its host reaches that host and
    nothing else.

    Detection is structural rather than by name, since a name-based scan is satisfied by a
    comment: a class counts when it connects a ``timeout`` to one of its own methods and that
    method moves which frame is shown. That last clause is what separates a player from the
    debounce, reveal and hold timers the package is full of — six of them, all of which this
    rule must not claim.
    """
    import ast

    found: dict[str, str] = {}
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bodies = {
                sub.name: ast.unparse(sub)
                for sub in node.body
                if isinstance(sub, ast.FunctionDef)
            }
            for call in ast.walk(node):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "connect"
                    and isinstance(call.func.value, ast.Attribute)
                    and call.func.value.attr == "timeout"
                    and call.args
                ):
                    continue
                slot = call.args[0]
                name = slot.attr if isinstance(slot, ast.Attribute) else None
                body = bodies.get(name or "", "")
                if any(token in body for token in _ADVANCES_A_FRAME):
                    found[node.name] = source.name

    unlisted = sorted(set(found) - set(_PLAYERS))
    stale = sorted(set(_PLAYERS) - set(found))
    assert not unlisted, (
        f"{unlisted} step frames on their own timer and are not in _PLAYERS. A player that "
        "lives in its host reaches that host and nothing else — drive FrameScrubber's instead, "
        "or add an entry saying why this one cannot"
    )
    assert not stale, (
        f"_PLAYERS names {stale}, which no longer step frames on a timer — delete the entries "
        "so the list keeps meaning what it says"
    )


#: The classes allowed to zoom on the wheel without the shared 1-D zoom (`gui/zoom.py`), each
#: with why. A class that zooms on the wheel and is not here uses a `ZoomWindow`.
_ZOOMERS: dict[str, str] = {
    "ImageView": (
        "**2-D.** A picture zooms about a point in both axes and pans in both; a window over one "
        "span is the wrong shape, and its overview would be a box, not a bar"
    ),
    "ArenaCanvas": (
        "**2-D**, for the same reason as `ImageView`: the field-of-view canvas zooms a frame, "
        "not a span of time or of a tree"
    ),
}


def _zooms_on_the_wheel(node) -> bool:
    """A class whose wheel handler reads the wheel and zooms, or that takes a pinch."""
    import ast

    for sub in node.body:
        if isinstance(sub, ast.FunctionDef) and sub.name == "wheelEvent":
            body = ast.unparse(sub)
            if "angleDelta" in body and "zoom" in body.lower():
                return True
    return "ZoomNativeGesture" in ast.unparse(node)


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        "pass",
        extra=(
            "class Timeline:\n"
            "    def wheelEvent(self, event):\n"
            "        self._zoom_by(0.8 ** (event.angleDelta().y() / 120))\n"
        ),
    ),
)
def zoom_lives_in_one_place(package) -> None:
    """A widget that zooms on the wheel uses the shared zoom (`gui/zoom.py`), or is listed.

    The cluster tree and the ethograms each carried a copy — the window, the wheel and pinch
    mapping, and a hand-drawn overview band that looked like no other control in the app. Two
    copies were how the overviews came to differ from every scrollbar beside them; the shared
    `ZoomWindow` and its `OverviewBar` (a real scrollbar) are how the next zoomable widget gets
    the same gestures and the same bar without writing either.

    Structural, like `playback_lives_in_one_place`: a class counts when its ``wheelEvent`` reads
    ``angleDelta`` and zooms, or it handles a pinch; it passes when it holds a `ZoomWindow` or
    uses the shared gestures.
    """
    import ast

    found: dict[str, bool] = {}
    for source in sorted(package.rglob("*.py")):
        # Only the counterexamples' home and the shared zoom itself are skipped.
        if source.name in ("requirements.py", "zoom.py"):
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _zooms_on_the_wheel(node):
                text = ast.unparse(node)
                found[node.name] = any(
                    token in text
                    for token in ("ZoomWindow", "zoom_wheel", "zoom_pinch")
                )
    own = sorted(name for name, shared in found.items() if not shared)
    unlisted = sorted(set(own) - set(_ZOOMERS))
    stale = sorted(set(_ZOOMERS) - set(own))
    assert not unlisted, (
        f"{unlisted} zoom on the wheel with their own zoom code. Hold a ZoomWindow and use "
        "zoom_wheel / zoom_pinch / OverviewBar (gui/zoom.py), or add an entry to _ZOOMERS "
        "saying why this one cannot"
    )
    assert not stale, (
        f"_ZOOMERS names {stale}, which no longer zoom on their own — delete the entries so "
        "the list keeps meaning what it says"
    )


# --------------------------------------------------------------------------- #
# video — what a frame surface owes whoever is watching it
# --------------------------------------------------------------------------- #
def _a_playable_surface_with_no_play_control():
    from detail.gui.scrubber import FrameScrubber

    scrubber = FrameScrubber(playable=True)
    scrubber._play_btn.setParent(None)
    scrubber._play_btn = None
    return scrubber


@requirement(
    kinds=("video",),
    counterexample=_a_playable_surface_with_no_play_control,
    retires="A video player's rules",
)
def a_playable_video_surface_starts_and_stops_from_one_control(video) -> None:
    """A video surface that plays does it from one control that both starts and stops.

    Twelve surfaces showed frames and not one could play them. What a person does in front of
    video is press play, and its absence read as the tool being broken rather than as something
    not yet built: a user reaching for play and finding none took it as a fault.

    One control rather than two, because the gesture that stops playback is the same gesture that
    started it and the pointer should not have to move between them — which matters at the rate
    someone scoring behavior actually presses it.

    Scoped to a **playable** surface, which is a decision each host makes and states at its
    call site — ``_SampledFrames`` indexes a sampled list with no rate behind it, and the stream
    preview passes ``playable=self._has_video``. Both are refusals rather than omissions, and
    ``playable`` having no default is what keeps them written down.
    """
    if not video.is_playable():
        return
    button = video.play_button()
    assert button is not None, (
        "built playable with no play control — pass `playable=False` if this surface genuinely "
        "cannot play, and say so at the call site"
    )
    assert (
        button.isCheckable()
    ), f"{button.text()!r} does not toggle, so stopping is a different target from starting"


def _a_surface_with_one_speed():
    from detail.gui.scrubber import FrameScrubber

    scrubber = FrameScrubber(playable=True)
    scrubber._speed_box.clear()
    scrubber._speed_box.addItem("1x", 1.0)
    return scrubber


@requirement(
    kinds=("video",),
    counterexample=_a_surface_with_one_speed,
)
def a_playable_video_surface_reaches_both_sides_of_one_times(video) -> None:
    """A video surface that plays offers speeds both slower and faster than 1x.

    Behavior is read at both ends. A fast bout — a flail, a rear, a bite — is not resolvable at
    1x and has to be slowed; a quiet stretch between bouts is skipped at speed rather than
    scrubbed through. A player pinned at 1x makes the first impossible and the second tedious, so
    the menu is not a refinement of playback, it is half of what playback is for.

    Stated as "both sides" rather than as a list: the exact set is a genuine choice
    (``PLAYBACK_SPEEDS``), but a menu that only slows, or only hurries, is a different capability
    wearing the same control.
    """
    if not video.is_playable():
        return
    speeds = [s for s in video.playback_speeds() if s is not None]
    assert any(s < 1.0 for s in speeds) and any(s > 1.0 for s in speeds), (
        f"the speed menu offers {sorted(speeds)}, which does not reach both sides of 1x — a "
        "player that can only slow down, or only speed up, covers half of what it is for"
    )


def _a_surface_that_cannot_step():
    from detail.gui.scrubber import FrameScrubber

    scrubber = FrameScrubber(playable=True)
    # Taken out of the control row while the attribute still holds it — the shape a rewrite
    # actually produces, and the one an attribute-based check cannot see.
    scrubber._controls.removeWidget(scrubber._prev)
    scrubber._prev.setParent(None)
    return scrubber


@requirement(kinds=("video",), counterexample=_a_surface_that_cannot_step)
def a_video_surface_steps_one_frame_in_both_directions(video) -> None:
    """Every video surface moves one frame back and one frame forward, however it plays.

    Playing and stepping answer different questions and neither substitutes for the other.
    Stepping is how a boundary is found — which frame a bout starts on, where a paw leaves the
    floor — and it was kept explicitly when playback arrived, because playing is no use for finding
    a behavior's edges. What this rule exists against is a rewrite treating the
    step buttons as the thing play replaces.

    It holds on **every** video surface, playable or not. A surface that cannot play is one where
    stepping is the only way through it, so it needs the pair more, not less.
    """
    buttons = video.step_buttons()
    assert len(buttons) == 2, (
        f"this surface has {len(buttons)} frame-step control(s); back one frame and forward one "
        "frame are both needed, and play replaces neither"
    )


#: How far a player's controls may sit below its own picture before this rule calls it a gap
#: rather than ordinary layout spacing. `PictureWithControls` puts nothing between the two but
#: the layout's own item spacing (a handful of px under Fusion); #396 measured the bug at 225px.
_MAX_PICTURE_CONTROLS_GAP = 24


def _picture_paired_with(scrubber):
    """The picture this scrubber controls, climbing from its parent — or ``(None, None)``.

    Not every ``video`` surface has a picture beside it: an arena dialog's keyframe scrubber
    drives a slider over nothing painted, and quick segment's sampled view can refuse to play at
    all. Neither widget needs to know about the other; the nearest ancestor holding an
    `ImageView` is the container both share, which is also the frame this rule measures the gap
    in.
    """
    from detail.gui.image_view import ImageView

    parent = scrubber.parentWidget()
    while parent is not None:
        pictures = parent.findChildren(ImageView)
        if pictures:
            return parent, pictures[0]
        parent = parent.parentWidget()
    return None, None


def _picture_drawn_bottom(picture, parent) -> int:
    """Where the picture's own image actually ends, mapped into ``parent``'s coordinates.

    Not the label's bounding box: `ImageView` keeps the pixmap's aspect ratio and centres it
    (`AlignCenter`), so a label stretched taller than its content draws the picture in the
    middle of itself with empty space above and below — measuring the *widget's* bottom edge
    would call that empty space part of the picture and never see the gap it creates. A label
    with nothing drawn yet has no picture to measure; its own bottom is the best signal there
    is.
    """
    from PyQt6.QtCore import QPoint

    pixmap = picture.pixmap()
    local_y = (
        picture.height()
        if pixmap is None or pixmap.isNull()
        else ((picture.height() + pixmap.height()) // 2)
    )
    return picture.mapTo(parent, QPoint(0, local_y)).y()


def _a_player_with_its_controls_far_below_its_picture():
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    from detail.gui.image_view import ImageView
    from detail.gui.scrubber import FrameScrubber

    host = QWidget()
    box = QVBoxLayout(host)
    picture = ImageView(zoomable=True)
    # The bug itself: a layout stretch on the picture, so it claims the whole pane — a 16:9
    # frame in a box this tall letterboxes hard, which is what makes the gap this rule exists
    # to catch.
    picture.set_frame(QPixmap(160, 90))
    box.addWidget(picture, 1)
    scrubber = FrameScrubber(playable=True)
    box.addWidget(scrubber)
    host.resize(
        400, 900
    )  # deliberately over-tall — the shape this rule exists to catch
    host.show()
    host.layout().activate()
    # `host` has no Python reference once this function returns except this one — and Qt
    # deletes a parentless widget's C++ side along with it, taking `scrubber` down too
    # (`RuntimeError: wrapped C/C++ object ... has been deleted`, this repo's recorded Qt
    # lifetime class). Keeping it reachable from the widget this function actually returns
    # is what keeps it alive for as long as the check needs it.
    scrubber._host = host
    return scrubber


@requirement(
    kinds=("video",),
    counterexample=_a_player_with_its_controls_far_below_its_picture,
)
def a_players_controls_sit_under_its_picture(video) -> None:
    """A player's controls sit directly under its picture, not at the bottom of whatever pane
    it was handed.

    A 16:9 clip in a tall pane letterboxes: the picture draws its own aspect-correct height and
    leaves the rest empty. Handing that empty remainder to the picture — the ordinary layout
    instinct, the same one that gives a table a stretch factor so it fills its panel — stretches
    the *label* to fill it instead, and its controls end up pinned to the bottom of the pane,
    however far that is from the picture itself. #396 measured it: 267px of dead space around a
    280px-tall picture, and the controls 225px below the picture's own bottom edge.

    Scoped to a video surface that has a picture paired with it at all — a bare scrubber with
    nothing painted beside it has no gap to measure.
    """
    parent, picture = _picture_paired_with(video)
    if picture is None:
        return
    if parent.layout() is not None:
        # Geometry is its previous value until the layout runs, and a widget built for a walk
        # has had no reason to run one. Activating is idempotent and changes no state a sibling
        # rule could read differently.
        parent.layout().activate()
    picture_bottom = _picture_drawn_bottom(picture, parent)
    controls_top = video.mapTo(parent, video.rect().topLeft()).y()
    gap = controls_top - picture_bottom
    assert 0 <= gap <= _MAX_PICTURE_CONTROLS_GAP, (
        f"the controls sit {gap}px from the picture's own bottom edge — a player's controls "
        "belong directly under its picture, not at the bottom of whatever pane it was handed"
    )


#: How far a transport may run past the picture's drawn edges before it reads as belonging to
#: the window rather than to the frame — a few pixels of rounding and frame border.
_MAX_CONTROLS_OVERHANG = 8


def _a_player_with_controls_wider_than_its_picture():
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QVBoxLayout, QWidget

    from detail.gui.image_view import ImageView
    from detail.gui.scrubber import FrameScrubber

    host = QWidget()
    box = QVBoxLayout(host)
    picture = ImageView(zoomable=True)
    # A 4:3 frame in a wide, short window: the picture is held back by the height and draws
    # narrow, and the scrubber under it is handed the window's whole width.
    picture.set_frame(QPixmap(160, 120))
    box.addWidget(picture, 1)
    scrubber = FrameScrubber(playable=True)
    box.addWidget(scrubber)
    host.resize(1600, 420)
    host.show()
    host.layout().activate()
    scrubber._host = (
        host  # keep the parentless host alive (see the counterexample above)
    )
    return scrubber


@requirement(
    kinds=("video",),
    counterexample=_a_player_with_controls_wider_than_its_picture,
)
def a_players_controls_span_its_picture_not_the_window(video) -> None:
    """A player's controls are as wide as its picture draws, not as wide as the window.

    The sideways half of `a_players_controls_sit_under_its_picture`. A picture keeps its aspect
    ratio, so in a window widened past it the frame is limited by the *height* and draws in the
    middle, while a transport handed the pane's width runs out past both edges — the slider's
    ends are no longer the clip's ends. Seen on Mark examples in a widened window (#628), and a
    property of the shared `PictureWithControls`, so every player had it.

    The transport may be wider than a small picture: never narrower than its own one-line width,
    because squeezing it would wrap the row instead. A surface whose picture has not drawn a
    frame yet has no width to measure.
    """
    parent, picture = _picture_paired_with(video)
    if picture is None:
        return
    if parent.layout() is not None:
        parent.layout().activate()
    size = picture.pixmap_size()
    if (
        size is None
        or size.width() <= 0
        or picture.width() <= 0
        or picture.height() <= 0
    ):
        return
    scale = min(picture.width() / size.width(), picture.height() / size.height())
    drawn = round(size.width() * scale)
    allowed = max(drawn, video.sizeHint().width()) + _MAX_CONTROLS_OVERHANG
    assert video.width() <= allowed, (
        f"the controls are {video.width()}px wide under a picture drawn {drawn}px wide — a "
        "player's controls span its picture, not the window it sits in"
    )


def _a_player_whose_controls_are_cut_short():
    from PyQt6.QtWidgets import QWidget

    from detail.gui.scrubber import FrameScrubber

    # Positioned by hand, the way `PictureWithControls` does it, and given less height than the
    # transport needs: the button row draws outside the scrubber and is clipped away.
    host = QWidget()
    host.resize(700, 400)
    scrubber = FrameScrubber(playable=True)
    scrubber.setParent(host)
    scrubber.setGeometry(0, 0, 700, 20)
    host.show()
    scrubber._host = (
        host  # keep the parentless host alive (see the counterexamples above)
    )
    return scrubber


@requirement(
    kinds=("video",),
    counterexample=_a_player_whose_controls_are_cut_short,
)
def a_players_controls_get_the_height_they_need(video) -> None:
    """A player's controls are never shorter than they need at their width — clipped is never
    an answer; a window too small for them scrolls.

    Seen on Mark examples (#628 follow-up): the "tracked" strip appears once a track's breaks
    are known, after the first layout, and the transport grew a row. `PictureWithControls`
    places its children by hand and only re-measured on its own resize, so the transport kept
    its old height and the Play / Prev frame row was cut in half — with no scrollbar, because
    the dialog had been told the old, smaller height too.

    Measured only on a scrubber that has been shown: an unshown widget has never been laid
    out, and its height is not a fact about anything.
    """
    if not video.isVisible():
        return
    need = video.heightForWidth(video.width())
    assert video.height() >= need, (
        f"the controls are {video.height()}px tall and need {need}px at {video.width()}px "
        "wide — the rows past that are drawn outside the widget and clipped"
    )


def _a_package_building_a_numeric_cell_by_hand():
    return _a_package_that(
        "pass",
        extra='def build(rows):\n    return [QStandardItem(f"{r.score:.3f}") for r in rows]\n',
    )


#: Cells built with ``str(<name>)`` whose value is genuinely text, keyed ``file:Class.method``
#: like :data:`_CONFIRMATIONS`, with the reason. A key that matches no site fails
#: (``tests/gui/test_requirements.py``), so this cannot outlive the code it excuses.
_TEXT_CELLS_FROM_STR: dict[str, str] = {
    "pages.py:_cell": (
        "the non-number branch of the shared builder: a number goes to `SortCell` on the next "
        "line, so whatever reaches `str()` here is already text"
    ),
}


def _numeric_text_cells(package) -> tuple[list[str], set[str]]:
    """Every cell built as text from what is probably a number, and the exemptions that matched.

    Pure over a directory, so the guard's own tests can hand it one-line packages
    (``test_the_numeric_cell_rule_would_catch_*``). The spellings, each found shipped:

    - an f-string with a numeric format spec (``f"{x:.3f}"``);
    - ``str()`` around a counting call (``str(len(rows))``);
    - ``str()`` around a bare name or attribute (``str(m.n_folds)``, and ``str(x)`` inside a
      comprehension over numbers — the fold table of query eval, a module since removed);
    - a conditional expression with ``str()`` in a branch (``"" if rank is None else
      str(rank)``, that module's Rank column).

    The last two cannot tell a number from an id held in a variable, so a ``str(<name>)`` cell
    that really is text is declared in :data:`_TEXT_CELLS_FROM_STR` with its reason.
    """
    import ast
    import re

    # `ast.unparse` of a format spec returns the f-string literal (`"f'.3f'"`) — no colon,
    # which is what made the first version of this rule match nothing and pass vacuously on
    # the whole package. Its counterexample is what caught that.
    numeric = re.compile(r"\.\d*[fged%]|,")

    def is_str_call(node) -> bool:
        return (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "str"
            and bool(node.args)
        )

    offenders: list[str] = []
    excused: set[str] = set()
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        rel = source.relative_to(package).as_posix()
        text = source.read_text(encoding="utf-8-sig")
        lines = text.split("\n")
        tree = ast.parse(text)
        owner = _owners(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in {"QStandardItem", "QTableWidgetItem"} or not node.args:
                continue
            arg = node.args[0]
            looks_numeric = may_be_text = False
            if isinstance(arg, ast.JoinedStr):
                looks_numeric = any(
                    isinstance(v, ast.FormattedValue)
                    and v.format_spec is not None
                    and numeric.search(ast.unparse(v.format_spec))
                    for v in arg.values
                )
            elif is_str_call(arg):
                inner = arg.args[0]
                if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") in {
                    "len",
                    "int",
                    "float",
                    "round",
                    "sum",
                }:
                    looks_numeric = True
                elif isinstance(inner, (ast.Name, ast.Attribute, ast.Subscript)):
                    looks_numeric = may_be_text = True
            elif isinstance(arg, ast.IfExp) and (
                is_str_call(arg.body) or is_str_call(arg.orelse)
            ):
                looks_numeric = may_be_text = True
            if not looks_numeric:
                continue
            key = f"{rel}:{owner.get(node, '<module>')}"
            if may_be_text and key in _TEXT_CELLS_FROM_STR:
                excused.add(key)
                continue
            offenders.append(
                f"{rel}:{node.lineno}  {lines[node.lineno - 1].strip()[:70]}"
            )
    return offenders, excused


@requirement(
    kinds=("package",),
    counterexample=_a_package_building_a_numeric_cell_by_hand,
)
def a_numeric_cell_sorts_on_its_value(package) -> None:
    """A cell built from a number uses `num_cell`, so its column sorts numerically.

    Every table in the package is sortable now, and `QStandardItem` compares **display text** —
    so a score column orders `0.9` above `0.12` and a count column runs 1, 11, 12, 2. The
    formatted string is the only thing a plain cell keeps; the number it came from is gone by
    the time Qt compares anything.

    **This is the half a call-site audit misses.** `pages._cell` gives the two shared builders
    numeric ordering, and it looked complete — but twenty-one cells were built by hand in the
    analysis result views and the query workspaces, where no builder is involved at all. An
    audit over builder *calls* found one site; this scan, over the cell constructor itself,
    found twenty-one.

    **And the scan had its own gap** (#35): it read only an f-string with a numeric spec and
    `str()` around a counting call, so `str(m.n_folds)`, `str(x)` in a comprehension and
    `"" if rank is None else str(rank)` all passed it, and the Rank, Folds and per-fold count
    columns of query eval (a module since removed) sorted 1, 10, 11, 2. Those spellings are
    read now; a `str(<name>)` cell that is really text is declared in
    :data:`_TEXT_CELLS_FROM_STR`.
    """
    offenders, _excused = _numeric_text_cells(package)
    assert not offenders, (
        "these cells are built from a number and will sort as text — use `tables.num_cell`, "
        "which keeps the formatted string and orders on the value (or, for a `str(<name>)` "
        "that is really text, declare it in `_TEXT_CELLS_FROM_STR` with the reason):\n  "
        + "\n  ".join(offenders)
    )


def _literal_done_steps(package) -> list[str]:
    """``file:line`` of every checklist ``Step(...)`` whose ``done`` is a literal ``True`` or
    ``False``. Pure over a directory, so its test can hand it a one-line package."""
    import ast

    offenders = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name != "Step":
                continue
            done = node.args[2] if len(node.args) > 2 else None
            for keyword in node.keywords:
                if keyword.arg == "done":
                    done = keyword.value
            if isinstance(done, ast.Constant) and isinstance(done.value, bool):
                offenders.append(
                    f"{source.relative_to(package).as_posix()}:{node.lineno}"
                )
    return offenders


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        'Step("quick_segment", "See it segment a clip", False, guides=False)'
    ),
)
def a_checklist_step_reads_its_state(package) -> None:
    """A setup checklist step's tick is read from a signal, never written as a literal.

    The Configuration checklist's last two rows, "See it segment a clip" and "Recommended:
    distill a fast model", were built with ``done=False``, so neither could ever show ✓ while
    `docs/gui.md` said the list "reads live state and marks each step ✓ when done" (#35). A
    literal in the ``done`` slot is the whole defect: nothing is wrong at the line, and the
    checklist renders. Read the state the step asks for (a `ReadinessSignals` field) instead.
    """
    offenders = _literal_done_steps(package)
    assert not offenders, (
        f"{offenders} build a checklist step whose tick is a literal, so it can never change: "
        "derive `done` from a `ReadinessSignals` field"
    )


def _a_table_nobody_sized():
    from PyQt6.QtGui import QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QTableView

    view = QTableView()
    model = QStandardItemModel(40, 1)
    for r in range(40):
        model.setItem(r, 0, QStandardItem(f"row {r}"))
    view.setModel(model)
    return view  # never passed to `fit_table_height`


@requirement(kinds=("table",), counterexample=_a_table_nobody_sized)
def a_table_is_sized_to_its_rows_or_declared_to_fill(table) -> None:
    """A table's height comes from `tables.fit_table_height`, unless it is meant to fill.

    Two failures, one rule. A table nobody sizes keeps Qt's default `Expanding` policy and its
    constant 192px hint, so it does both wrong things at once: an **empty** one draws a blank
    box the height of the pane (measured: 429px of white grid on the model manager's Hugging
    Face table, with no rows in it at all), and a **full** one shows five rows however many it
    holds. Neither fails anything — the page renders, the widget exists.

    The exception is real and is why this is not simply "call the helper": a table that is a
    page's or dialog's primary content is *added with a stretch factor* and should fill. That is
    the declaration, and it is read from the layout rather than from a list of class names,
    because the layout is where the intent already lives.

    Was written down and unenforced for a long time — `test_table_height.py` checked the helper,
    not the tables — which is exactly how two pages ended up never calling it.
    """
    from PyQt6.QtWidgets import QSizePolicy

    parent = table.parentWidget()
    layout = parent.layout() if parent is not None else None
    if layout is not None:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is not None and item.widget() is table:
                stretch = getattr(layout, "stretch", lambda _i: 0)(i)
                if stretch > 0:
                    return  # declared to fill, by the layout that fills it
                break

    vertical = table.sizePolicy().verticalPolicy()
    assert vertical != QSizePolicy.Policy.Expanding, (
        "this table still has Qt's default Expanding policy, so nobody sized it: empty it "
        "draws a blank box the height of the pane, and full it shows five rows whatever it "
        "holds. Call `tables.fit_table_height`, or add it with a stretch factor if it is "
        "meant to fill"
    )


# --------------------------------------------------------------------------- #
# splitter — panes side by side (`gui/grip_splitter.py`)
# --------------------------------------------------------------------------- #
def _calls_to(package, name: str) -> dict[str, list]:
    """``{file: [ast.Call, ...]}`` for every call to ``name`` (bare or as an attribute)."""
    import ast

    found: dict[str, list] = {}
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None
            )
            if called == name:
                found.setdefault(source.relative_to(package).as_posix(), []).append(
                    node
                )
    return found


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that("QSplitter(Qt.Orientation.Horizontal)"),
)
def a_split_is_built_as_a_grip_splitter(package) -> None:
    """Panes side by side are a `GripSplitter` — no bare `QSplitter` is built anywhere in the GUI.

    Eighteen splits, five of them `GripSplitter` and thirteen plain `QSplitter`s, gave three
    answers to "put this panel away": a ‹ / › button on the navigation, "◂ Hide filters" on
    Interpret, and on six splitters dragging a pane to nothing, with no trace of where it went —
    the navigation among them (user, 2026-11-09). The widget walk sees the splits it can build;
    this sees the ones built lazily, inside a result view or on a click.
    """
    offenders = [
        f"{f}:{node.lineno}"
        for f, nodes in _calls_to(package, "QSplitter").items()
        for node in nodes
    ]
    assert not offenders, (
        f"bare QSplitter built at {', '.join(offenders)}: use `grip_splitter.GripSplitter` "
        "with a key, and `set_foldable` for a side panel"
    )


@requirement(
    kinds=("package",),
    counterexample=lambda: _a_package_that(
        'GripSplitter(Qt.Orientation.Horizontal, key="x")'
    ),
)
def a_split_remembers_where_it_was_left(package) -> None:
    """Every `GripSplitter` is restored from and saved to its key — each file calls `remember()` once per split it builds.

    The page is rebuilt on every navigation (`MainWindow._show` discards it), so a split that is
    not remembered is a width the user sets again every time they come back. The key is required
    by the constructor; `remember()` is what reads it, and a split that never calls it keeps
    nothing.
    """
    short = _unremembered_splits(package)
    assert not short, (
        f"a GripSplitter that is never remembered: {', '.join(short)} — call "
        "`split.remember()` on that splitter once its panes are added and marked foldable"
    )


def _unremembered_splits(package) -> list[str]:
    """``file:line`` of each `GripSplitter` built with no ``.remember()`` called on *it*.

    Matched by the name the splitter is held under, not counted per file: a count let any
    ``remember(...)`` stand in — `clip_popup.py`'s module-level ``remember(key, on)`` for its
    toggles, or a second split's call covering the first. A local name (``split``) is matched
    within the function that built it, since two functions in one file each name theirs
    ``split``; an attribute (``self._split``) anywhere in the file, since a page restores its
    split in a later method, and a name it is handed on to (``self._split = split``) counts. A
    splitter built and never bound to a name cannot be remembered.
    """
    import ast

    out = []
    for source in sorted(package.rglob("*.py")):
        if source.name in _SCAN_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        rel = source.relative_to(package).as_posix()

        def receivers(scope) -> set[str]:
            return {
                ast.unparse(node.func.value)
                for node in ast.walk(scope)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "remember"
            }

        in_file = receivers(tree)
        functions = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and _called_name(node) == "GripSplitter"
            ):
                continue
            owner = min(
                (
                    f
                    for f in functions
                    if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)
                ),
                key=lambda f: (f.end_lineno or f.lineno) - f.lineno,
                default=tree,
            )
            targets = [
                t
                for parent in ast.walk(owner)
                if isinstance(parent, ast.Assign) and parent.value is node
                for t in parent.targets
            ]
            # ``self._split = split`` hands the splitter on under a second name.
            names = {ast.unparse(t) for t in targets}
            targets += [
                t
                for parent in ast.walk(owner)
                if isinstance(parent, ast.Assign)
                and isinstance(parent.value, ast.Name)
                and parent.value.id in names
                for t in parent.targets
            ]
            ok = any(
                ast.unparse(t)
                in (in_file if isinstance(t, ast.Attribute) else receivers(owner))
                for t in targets
            )
            if not ok:
                out.append(f"{rel}:{node.lineno}")
    return out


def _called_name(call) -> str | None:
    import ast

    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _a_vertical_splitter():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QSplitter, QWidget

    split = QSplitter(Qt.Orientation.Vertical)
    split.addWidget(QWidget())
    split.addWidget(QWidget())
    return split


def _a_splitter_that_drags_a_pane_to_nothing():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QSplitter, QWidget

    split = QSplitter(Qt.Orientation.Horizontal)
    split.addWidget(QWidget())
    split.addWidget(QWidget())
    return split


@requirement(kinds=("splitter",), counterexample=_a_vertical_splitter)
def a_split_puts_panes_side_by_side(splitter) -> None:
    """A splitter divides panes side by side; a vertical stack is `CollapsibleBand`s with drag bars.

    The rule `tests/gui/test_workspace_parity.py` wrote down — "vertical sections are
    `CollapsibleBand` + drag handle; horizontal panes are `QSplitter`" — was swept over two
    workspaces, and the explorer of query eval (a module since removed) kept a vertical
    `QSplitter` (timeline over the panes) that neither of them could see. Swept here over every
    page and dialog the corpus builds.
    """
    from PyQt6.QtCore import Qt

    assert splitter.orientation() == Qt.Orientation.Horizontal, (
        f"a vertical {type(splitter).__name__}: a stack of sections is CollapsibleBands "
        "(`gui/collapse.py`), which fold by a titled header"
    )


@requirement(
    kinds=("splitter",), counterexample=_a_splitter_that_drags_a_pane_to_nothing
)
def a_pane_never_drags_to_nothing(splitter) -> None:
    """No pane drags to zero width: it stops at its minimum, or — a side panel — shuts to a strip with its name on it.

    A plain `QSplitter` lets any pane be dragged to nothing, and the pane is then gone with no
    trace: the navigation, Browse's player, the syllable tables (user, 2026-11-09).
    `GripSplitter` holds every pane at its minimum unless the splitter declared it foldable
    (`set_foldable`, with the name the strip shows), and a shut pane's strip is its grip, never
    zero wide.
    """
    from detail.gui.grip_splitter import GripSplitter

    assert isinstance(splitter, GripSplitter), (
        f"a {type(splitter).__name__} lets a pane be dragged to nothing — build a "
        "`grip_splitter.GripSplitter`"
    )
    foldable = splitter.foldable_panes()
    for pane in range(splitter.count()):
        if pane in foldable:
            assert foldable[pane].strip(), f"pane {pane} folds to a strip with no name"
            assert splitter.isCollapsible(pane) == splitter.is_folded(
                pane
            ), f"pane {pane} ({foldable[pane]}) collapses while it is open"
        else:
            assert not splitter.isCollapsible(
                pane
            ), f"pane {pane} is not a side panel and can still be dragged to nothing"


#: Glyphs that, alone on a button or beside "Hide"/"Show", mean "put a panel away / bring it back".
_PANE_CHEVRONS = "‹›«»◂▸"
#: Words that name a side pane on a show/hide button.
_PANE_WORDS = (
    "filter",
    "navigation",
    "nav",
    "panel",
    "pane",
    "sidebar",
    "rail",
    "list",
    "checklist",
)


def _hides_a_pane(text: str) -> bool:
    """Whether button text reads as hiding or showing a side pane — pure, so the negative cases
    can hand it a literal."""
    import re

    bare = text.replace("&", "").strip()
    if not bare:
        return False
    if all(ch in _PANE_CHEVRONS or ch.isspace() for ch in bare):
        return True
    words = re.findall(r"[a-z]+", bare.lower())
    if words[:1] == ["hide"]:
        return True
    has_chevron = any(ch in _PANE_CHEVRONS for ch in bare)
    names_pane = any(w.startswith(p) for w in words for p in _PANE_WORDS)
    toggles = bool({"hide", "show"} & set(words))
    # "Filters ▸" is a pane's name and a chevron and nothing else; "▸ filter_settings run" is a
    # fold header that happens to contain a pane word, so the name alone must be short.
    return has_chevron and (toggles or (names_pane and len(words) <= 2))


def _a_hide_filters_button():
    from PyQt6.QtWidgets import QPushButton

    return QPushButton("◂ Hide filters")


@requirement(kinds=("button",), counterexample=_a_hide_filters_button)
def a_side_panel_has_no_hide_button(button) -> None:
    """No separate button hides or shows a side panel — the panel's grip shuts it, and the strip it leaves is clicked or pulled to open it.

    The navigation had ‹ / ›, Interpret's filters had "◂ Hide filters" and "Filters ▸", every
    other panel had nothing: three gestures for one job, learned per panel. The grip folds every
    side panel (`gui/grip_splitter.py`), and Ctrl+\\ keeps the keyboard's way to the navigation.

    **The strip's own mark is allowed, and is not a button.** A shut panel's strip shows › or ‹
    at its centre, where an open grip's dots are, and a click anywhere on the strip opens the
    panel (user, 2026-11-09: "keep the icon to expand in the same location as the handles").
    That target is the splitter handle itself, painted by it — so it sits where the panel is
    and nowhere else, and there is still one gesture per panel. This rule walks push buttons,
    which the handle is not; the same ‹ on a `QPushButton` beside the panel is still refused,
    and `tests/gui/test_grip_splitter.py` pins both halves.

    Read from the button's text, so it can be fooled: a toggle labelled "Less", "Collapse" or
    "×" passes, and so would a chevron button that pages through something rather than hiding
    a pane — `_PANE_CHEVRONS` holds only ‹ › « » ◂ ▸ because ◀ ▶ are play and step here. It
    walks `QPushButton`s (the corpus's "button" kind), so a `QToolButton` toggle is not seen.
    """
    assert not _hides_a_pane(button.text()), (
        f"{button.text()!r} reads as a button that hides or shows a panel; a side panel folds "
        "by its splitter's grip, and its strip opens it — `GripSplitter.set_foldable`"
    )
