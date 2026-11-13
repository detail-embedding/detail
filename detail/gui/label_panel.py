"""Labelling on INTERPRET: the label bar, and the label arithmetic every surface shares (§22).

The **label bar** (:class:`LabelBar`) sits between the tree and the montage (review round 9,
option A), in three lines. The first names the open cluster and shows its label as a coloured
chip, with the controls the page seats at its right (the montage's toolbar). The second carries
its figures at the left (shortened with "…" to the room left, the whole line in a tooltip) and at
the right what the last write did and whether it is saved. The third starts with the **label
field** (:class:`LabelField`): type a name and press Enter
(or **Label**) to label the selected cluster; a name that exists joins that label, and the button
says so before the click. A click in the field only puts the cursor there (user, 2026-11-09: it
used to open the list, which took the first click). **Labels ▾** lists every label with its colour
and its time (most specific label winning); **choosing a label's name anywhere only fills the
field** — the list's entries and a type-ahead suggestion alike — and Enter or **Label** writes it.
Each entry's submenu is where that label is gone to, renamed or merged; a label the cluster may
not join is greyed with the reason. The list opens with the recent labels (:class:`QuickLabels`),
numbered for their keys 1–9 — a key labels at once, the entry fills — and greyed with the same
reason; after **Labels ▾** come **Undo** and then the page's actions on the cluster
(INTERPRET's *Open in window*).

The functions are the rules: which label covers a node (:func:`label_paths`), how labels nest into
rows (:func:`path_rows`, shared with BROWSE so "other" means one thing), which labels a node may
join (:func:`label_choices`) and which a label may merge into (:func:`merge_targets`). All of it is
computed from the node cache in memory, so it follows a label the instant it is typed.
"""

from PyQt6.QtCore import QEvent, QObject, QPoint, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from detail.core.progress import format_duration
from detail.gui import plot_theme
from detail.gui.sizing import (
    ElidedLabel,
    emphasis_font,
    larger_font,
    lines,
    set_available,
)
from detail.storage.behavior_output import PATH_SEPARATOR

UNLABELLED = "unlabelled"
#: How many recent labels the bar offers as buttons: one per number key, 1–9.
QUICK_LABELS = 9
#: The undo button's text on the label bar and in a cluster window (AUDIT's, ``label_clean``).
UNDO_TEXT = "Undo (⌘Z)"


def label_paths(view, node_names: dict) -> dict[str, str]:
    """``{node id: "outer/inner"}`` for every labelled node, from the tree's own ancestry."""
    out = {}
    for node, name in node_names.items():
        if not name:
            continue
        chain = [
            node_names[str(a)]
            for a in view.ancestors(int(node))
            if node_names.get(str(a))
        ]
        out[str(node)] = PATH_SEPARATOR.join([*chain, name])
    return out


def path_rows(by_path: dict[str, int]) -> list[tuple[str, int, str, int]]:
    """``[(text, depth, root label, amount)]`` for ``{label path: amount}``, largest group first.

    A root label's row is its whole group's amount. Beneath it, one row per more specific path,
    and an *other* row when some sits on the root label itself beside sub-labels. Grouped by the
    outermost name, so two nodes that share a name share a row. INTERPRET's labels panel and
    BROWSE's label filter both lay their labels out with this, so "other" means one thing.
    """
    groups: dict[str, dict[str, int]] = {}
    for path, amount in by_path.items():
        root = path.split(PATH_SEPARATOR)[0]
        groups.setdefault(root, {})[path] = amount
    rows: list[tuple[str, int, str, int]] = []
    for root, parts in sorted(groups.items(), key=lambda item: -sum(item[1].values())):
        rows.append((root, 0, root, sum(parts.values())))
        if len(parts) > 1 or root not in parts:
            for path, amount in sorted(parts.items(), key=lambda item: -item[1]):
                text = "other" if path == root else path.split(PATH_SEPARATOR, 1)[1]
                rows.append((text, 1, root, amount))
    return rows


def row_path(text: str, depth: int, root: str) -> str:
    """The label path a :func:`path_rows` row stands for — a root's "other" is the root itself."""
    if depth == 0 or text == "other":
        return root
    return f"{root}{PATH_SEPARATOR}{text}"


def root_slots(paths) -> dict[str, int]:
    """``{root label: colour slot}`` — roots in name order, so a label keeps its colour on every
    surface that colours by it (INTERPRET's tree and labels panel, BROWSE)."""
    roots = sorted({p.split(PATH_SEPARATOR)[0] for p in paths if p})
    return {root: index for index, root in enumerate(roots)}


def label_rows(view, node_names: dict) -> list[tuple[str, int, str, int]]:
    """``[(text, depth, root label, time_ns)]`` per label, largest first, unlabelled last
    (:func:`path_rows`): what the label list and the swatches above the tree show."""
    times = view.label_time(node_names)
    paths = label_paths(view, node_names)
    by_path: dict[str, int] = {}
    for node, time in times.items():
        if node == UNLABELLED:
            continue
        path = paths.get(node, node)
        by_path[path] = by_path.get(path, 0) + time
    rows = path_rows(by_path)
    rows.append((UNLABELLED, 0, "", times.get(UNLABELLED, 0)))
    return rows


def _parent_name(view, node: int, node_names: dict) -> str:
    """The label on the nearest labelled branch strictly above ``node``, or ``""``."""
    for ancestor in reversed(view.ancestors(int(node))):
        name = node_names.get(str(ancestor))
        if name:
            return name
    return ""


def label_choices(
    view, node_names: dict, node: int
) -> list[tuple[str, int, str | None]]:
    """``[(label, clusters, why not | None)]`` — the existing labels ``node`` could join.

    Most-used first. A label the node cannot join carries the reason, from the rule the run holds
    (``authored_run.check_labels``): a label's clusters share one parent label, and a label is never
    inside itself. Answered here, before the click, so a choice that would be refused is greyed with
    its reason rather than offered and then rejected.
    """
    members: dict[str, list[int]] = {}
    for other, name in node_names.items():
        if name:
            members.setdefault(name, []).append(int(other))
    own = node_names.get(str(node), "")
    mine = _parent_name(view, node, node_names)
    above = {node_names.get(str(a)) for a in view.ancestors(int(node))} - {None, ""}
    out = []
    for name in sorted(members, key=lambda n: (-len(members[n]), n.casefold())):
        if name == own:
            continue
        clusters = members[name]
        theirs = _parent_name(view, clusters[0], node_names)
        if name in above:
            reason = f"Already inside “{name}”."
        elif any(int(node) in view.ancestors(c) for c in clusters):
            reason = f"A cluster of “{name}” sits inside this one."
        elif theirs != mine:
            where = f"inside “{theirs}”" if theirs else "at the top level"
            here = f"inside “{mine}”" if mine else "at the top level"
            reason = (
                f"“{name}” sits {where} and this sits {here}: a label's clusters share one "
                "parent label."
            )
        else:
            reason = None
        out.append((name, len(clusters), reason))
    return out


def merge_targets(view, node_names: dict, label: str) -> list[str]:
    """The labels ``label`` could be merged into: those under the same parent label, neither
    holding the other — the moves that leave every label with one parent."""
    members: dict[str, list[int]] = {}
    for node, name in node_names.items():
        if name:
            members.setdefault(name, []).append(int(node))
    if label not in members:
        return []
    parent = _parent_name(view, members[label][0], node_names)
    mine = set(members[label])
    out = []
    for name, nodes in members.items():
        if name == label or _parent_name(view, nodes[0], node_names) != parent:
            continue
        nested = any(set(view.ancestors(n)) & mine for n in nodes) or any(
            set(view.ancestors(n)) & set(nodes) for n in mine
        )
        if not nested:
            out.append(name)
    return sorted(out, key=str.casefold)


_FIELD_TIP = (
    "Type a name and press Enter. Labels ▾ lists every label; choosing one puts it "
    "here. L focuses here; double-click a block to label it in place."
)


def _swatch(colour: QColor) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(colour))
    return QIcon(pixmap)


class _LabelsMenu(QMenu):
    """**Labels ▾**: a click on a label's own row fills the field, as its submenu's first entry
    does; hovering the row still opens the submenu (go to, rename, merge). A plain QMenu gives a
    row with a submenu nothing to do on a click, so the name — the thing a person clicks — was
    the one part of the list that did nothing (user, 2026-11-09)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        #: A label row's path (its action's ``data``; two sub-labels may share a name) → the
        #: entry a click on the row triggers.
        self.fills: dict[str, object] = {}

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt's name
        action = self.actionAt(event.position().toPoint())
        fill = self.fills.get(action.data()) if action is not None else None
        if fill is not None and fill.isEnabled():
            action.menu().hide()
            self.hide()
            fill.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _ChooseSuggestion(QObject):
    """On the type-ahead's popup: Enter calls ``chosen()``, which says whether it took the key.

    An object of its own rather than the field's filter, so the popup's events while the field
    is torn down never reach a half-deleted field: only an Enter key does."""

    def __init__(self, parent, chosen) -> None:
        super().__init__(parent)
        self._chosen = chosen

    def eventFilter(self, _watched, event) -> bool:  # noqa: N802 — Qt's name
        return (
            event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and self._chosen()
        )


class LabelField(QWidget):
    """``[ name…               ] [Label] [Remove] [Labels ▾]`` — labelling the selected cluster.

    The label bar's second line. The page hands it the cluster (:meth:`show_node`) and the
    labels (:meth:`set_labels`); it emits what the user asked for and changes nothing itself, so
    the page's one write queue is the only path to the run. **Only Enter and Label write**: a
    label chosen from **Labels ▾** or from the type-ahead is put in the field (:meth:`fill`) for
    one of them to apply — one rule for every click on a name (user, 2026-11-09, after a numbered
    entry wrote at once while a listed one did not).
    """

    #: ``(node, name)``: label ``node``; ``""`` removes its own label.
    label_requested = pyqtSignal(int, str)
    #: A label's name, from its entry in the list.
    rename_wanted = pyqtSignal(str)
    merge_wanted = pyqtSignal(str)
    #: Go to a label: its root name (the page selects its widest cluster).
    label_picked = pyqtSignal(str)

    def __init__(self, terms, parent=None, *, tip=None, caption_tip: str = "") -> None:
        """``tip(name, key)``: a live button's tooltip, where the buttons do something other
        than label the open cluster (AUDIT's cleaning view moves clips with them)."""
        super().__init__(parent)
        self._terms = terms
        self._tip = tip
        self._dead_reason = ""
        self._node: int | None = None
        self._own = ""
        self._can_edit = False
        self._existing: list[str] = []
        self._rows: list[tuple[str, int, str, int]] = []
        self._colour_of = lambda _root: QColor(plot_theme.current().neutral)
        self._choices: dict[str, str | None] = {}
        #: The recent labels the menu leads with (:meth:`set_recent`).
        self._recent: QuickLabels | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setFont(larger_font(self.edit))
        self.edit.setMinimumWidth(lines(14))
        self.edit.setPlaceholderText(f"Label this {terms.unit}")
        self.edit.returnPressed.connect(self.submit)
        self.edit.textChanged.connect(self._sync_button)
        self._completer = QCompleter([])
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.edit.setCompleter(self._completer)
        # Enter on a highlighted suggestion fills the field; Qt's completer would also pass the
        # key on to the field and write it, unlike a click on the same suggestion.
        self._suggestions = _ChooseSuggestion(self._completer, self._suggestion_chosen)
        self._completer.popup().installEventFilter(self._suggestions)
        self.edit.installEventFilter(self)
        row.addWidget(self.edit, 1)
        self.label_button = QPushButton("Label")
        self.label_button.setFont(emphasis_font(self.label_button, steps=1))
        # The style's default-button look (the accent fill on macOS) is what makes it the one
        # action on the line, without a stylesheet; Enter in the field submits it too.
        self.label_button.setDefault(True)
        self.label_button.setAutoDefault(True)
        self.label_button.clicked.connect(self.submit)
        row.addWidget(self.label_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(
            lambda: self._node is not None and self.label_requested.emit(self._node, "")
        )
        row.addWidget(self.remove_button)
        # A QPushButton like its neighbours, not a QToolButton: macOS gives that class a smaller
        # default font and height, so "Labels ▾" read as smaller than Label and Remove.
        self.list_button = QPushButton()
        self.list_button.setText("Labels ▾")
        self.list_button.setToolTip(
            "Your recent labels first, numbered — press 1–9 with the tree in focus to label "
            "with one at once — then every label, with its colour and time. Choosing one puts "
            "it in the field; hover a label to go to, rename or merge it"
        )
        self.list_button.clicked.connect(self.show_list)
        row.addWidget(self.list_button)
        self._sync_enabled()

    # -- inputs ------------------------------------------------------------- #
    def set_recent(self, recent: "QuickLabels") -> None:
        """Lead the labels menu with ``recent``'s numbered entries (keys 1–9)."""
        self._recent = recent

    def set_labels(self, rows, colour_of) -> None:
        """Every label as :func:`label_rows` gives them; ``colour_of(root)`` is its colour."""
        self._rows = [r for r in rows if r[2]]
        self._colour_of = colour_of

    def show_node(
        self, node: int, own: str, inherited: str, names: list[str], choices
    ) -> None:
        """Point the field at ``node``: ``own`` is its label, ``inherited`` its nearest labelled
        parent's, ``choices`` the labels it could join (:func:`label_choices`)."""
        self._node = int(node)
        self._own = own
        self.edit.setText(own)
        self.edit.setPlaceholderText(
            f"Sub-label inside “{inherited}”"
            if inherited and not own
            else f"Label this {self._terms.unit}"
        )
        self._existing = list(names)
        self._completer.model().setStringList(self._existing)
        self._choices = {name: reason for name, _n, reason in choices or []}
        self._sync_button()
        self._sync_enabled()

    def set_editable(self, can_edit: bool, reason: str = "") -> None:
        """``reason``: why nothing can be labelled when ``can_edit`` is false, said on the field
        and its buttons."""
        self._can_edit = bool(can_edit)
        self._dead_reason = reason
        self._sync_enabled()

    # -- actions ------------------------------------------------------------ #
    def submit(self) -> None:
        if self._node is None or not self._can_edit:
            return
        name = self.edit.text().strip()
        if name:
            self.label_requested.emit(self._node, name)

    def focus(self) -> None:
        if self._can_edit:
            self.edit.setFocus()
            self.edit.selectAll()

    def fill(self, name: str) -> None:
        """Put ``name`` in the field, selected, for Enter or **Label** to apply; writes nothing."""
        if self._node is None or not self._can_edit:
            return
        self.edit.setText(name)
        # While the menu is open this makes the field its window's focus: where the cursor is
        # when the menu closes, so Enter applies the name (not Labels ▾, which had it).
        self.edit.setFocus(Qt.FocusReason.PopupFocusReason)
        self.edit.selectAll()

    def labels_menu(self) -> QMenu:
        """The list of every label: colour, name, time. A click on a name fills the field; its
        submenu goes to, renames or merges it."""
        menu = _LabelsMenu(self)
        menu.setToolTipsVisible(True)
        recent = self._recent.actions() if self._recent is not None else []
        if recent:
            heading = menu.addAction("Recent (keys 1–9)")
            heading.setEnabled(False)
            for key in recent:
                # Its own entry, not the one the key triggers: the key labels at once and the
                # entry fills. Greyed, and saying why, exactly when its key does nothing.
                name = key.data()
                entry = menu.addAction(
                    key.icon(), key.text(), lambda n=name: self.fill(n)
                )
                entry.setEnabled(key.isEnabled())
                entry.setToolTip(
                    f"Put “{name}” in the field; Enter labels with it "
                    f"(key {key.text().split()[0]} labels at once)"
                    if key.isEnabled()
                    else key.toolTip()
                )
            menu.addSeparator()
        if not self._rows:
            empty = menu.addAction("No labels yet. Type a name to make the first.")
            empty.setEnabled(False)
            return menu
        total = sum(t for _x, depth, _r, t in self._rows if depth == 0) or 1
        for text, depth, root, time in self._rows:
            path = row_path(text, depth, root)
            name = path.split(PATH_SEPARATOR)[-1]
            shown = ("    " * depth) + text
            sub = menu.addMenu(
                _swatch(self._colour_of(root)),
                f"{shown} · {format_duration(time / 1e9)} · {100 * time / total:.0f}%",
            )
            sub.setToolTipsVisible(True)
            if text == "other":
                note = sub.addAction(f"Time inside “{root}” no sub-label covers")
                note.setEnabled(False)
                continue
            join = sub.addAction(f"Label as “{name}”", lambda n=name: self.fill(n))
            reason = self._choices.get(name, "")
            if name == self._own:
                join.setEnabled(False)
                join.setToolTip(f"This {self._terms.unit} is “{name}” already.")
            elif not self._can_edit or self._node is None:
                join.setEnabled(False)
            elif reason:
                join.setEnabled(False)
                join.setToolTip(reason)
            sub.menuAction().setData(path)
            menu.fills[path] = join
            sub.menuAction().setToolTip(
                join.toolTip()
                if not join.isEnabled() and join.toolTip()
                else f"Click to put “{name}” in the field; hover to go to, rename or merge it"
            )
            sub.addAction(f"Go to “{name}”", lambda r=root: self.label_picked.emit(r))
            sub.addSeparator()
            for verb, signal in (
                ("Rename", self.rename_wanted),
                ("Merge into another", self.merge_wanted),
            ):
                action = sub.addAction(
                    f"{verb} “{name}”…", lambda n=name, s=signal: s.emit(n)
                )
                action.setEnabled(self._can_edit)
        return menu

    def show_list(self) -> None:
        from detail.gui.menus import open_once

        open_once(
            self.labels_menu(),
            self.edit.mapToGlobal(QPoint(0, self.edit.height())),
            wait=True,
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 — Qt's name
        if (
            watched is self.edit
            and event.type() == QEvent.Type.ShortcutOverride
            and self._passes_on(event)
        ):
            # Left unaccepted, so the page's (or window's) ⌘Z shortcut gets it.
            return True
        return super().eventFilter(watched, event)

    def _suggestion_chosen(self) -> bool:
        """Enter on a highlighted suggestion: fill the field with it and stop the key there."""
        popup = self._completer.popup()
        index = popup.currentIndex()
        if not popup.isVisible() or not index.isValid():
            return False
        popup.hide()
        self.edit.setText(index.data())
        self.edit.selectAll()
        return True

    def _passes_on(self, event) -> bool:
        """⌘Z / ⇧⌘Z the field has no use for: a line edit claims both even with nothing of its
        own to undo, which after Enter — focus still in the field — left ⌘Z doing nothing at all
        where it should undo the label just written."""
        if event.matches(QKeySequence.StandardKey.Undo):
            return not self.edit.isUndoAvailable()
        if event.matches(QKeySequence.StandardKey.Redo):
            return not self.edit.isRedoAvailable()
        return False

    # -- internals ---------------------------------------------------------- #
    def _sync_button(self) -> None:
        """ "Add to ‘X’" when the typed name is an existing label, "Label" otherwise — so typing a
        name that exists says it joins that label rather than looking like a new one (§22).
        """
        typed = self.edit.text().strip().casefold()
        match = next((n for n in self._existing if n.casefold() == typed), None)
        if match and match != self._own:
            self.label_button.setText(f"Add to “{match}”")
        else:
            self.label_button.setText("Label")

    def _sync_enabled(self) -> None:
        if not self._can_edit:
            why = self._dead_reason or "This run cannot be edited here."
        elif self._node is None:
            why = f"Pick a {self._terms.unit} in the tree first."
        else:
            why = ""
        set_available(self.edit, not why, why)
        if not why:
            self.edit.setToolTip(_FIELD_TIP)
        set_available(self.label_button, not why, why)
        set_available(
            self.remove_button,
            not why and bool(self._own),
            why or f"This {self._terms.unit} has no label of its own.",
        )


def _swatch_pixmap(colour: QColor, px: int) -> QPixmap:
    pixmap = QPixmap(px, px)
    pixmap.fill(QColor(colour))
    return pixmap


class LabelChip(QWidget):
    """The open cluster's label, large, in its colour: filled for its own label, outlined for a
    parent's ("inside “climb”"), muted text alone for "not labelled"."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFont(emphasis_font(self, steps=2))
        self._text = ""
        self._colour: QColor | None = None
        self._filled = False
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_label(self, text: str, colour: QColor | None, *, filled: bool) -> None:
        """``colour`` ``None`` draws ``text`` muted with no chip (the "not labelled" state)."""
        self._text, self._colour, self._filled = text, colour, bool(filled)
        self.setAccessibleName(text)
        self.setToolTip(text)
        self.updateGeometry()
        self.update()

    def text(self) -> str:
        return self._text

    def is_filled(self) -> bool:
        return self._colour is not None and self._filled

    def colour(self) -> QColor | None:
        return self._colour

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt's name
        metrics = self.fontMetrics()
        pad = metrics.height() // 2 if self._colour is not None else 0
        return QSize(
            metrics.horizontalAdvance(self._text) + 2 * pad + 2,
            metrics.height() + metrics.height() // 2,
        )

    def paintEvent(self, _event) -> None:  # noqa: N802 — Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        palette = self.palette()
        if self._colour is None:
            painter.setPen(
                palette.color(
                    QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText
                )
            )
            painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter, self._text)
            return
        radius = rect.height() / 4
        if self._filled:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._colour)
            painter.drawRoundedRect(rect, radius, radius)
            light = self._colour.lightnessF() > 0.6
            painter.setPen(
                QColor(Qt.GlobalColor.black if light else Qt.GlobalColor.white)
            )
        else:
            painter.setPen(QPen(self._colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)
            painter.setPen(palette.color(QPalette.ColorRole.WindowText))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._text)


class QuickLabels(QObject):
    """The run's recent labels, ``1 climb`` … ``9 …`` — keys 1–9, and the top of **Labels ▾**.

    The label's number key (which the page routes to :meth:`press`) asks to put the open cluster
    under that label: the same request as typing its name and pressing Enter, so it joins the
    label, and a label the cluster may not join (the parent rule, ``label_choices``) is greyed
    with the reason. **Labels ▾** shows the entries under their numbers, but its entries fill
    the field instead (:meth:`LabelField.labels_menu`): a key is a shortcut, a click is a choice.

    **No button of its own** (user, 2026-11-09). It was **Recent ▾**, one menu beside
    **Labels ▾** — two menus of labels side by side, the second a subset of the first. The
    entries now lead the Labels ▾ menu, numbered (:meth:`LabelField.set_recent`), so the keys
    stay discoverable where every label already is, and the field's row has the room back.
    :attr:`menu` holds the same entries on their own, for a surface that wants them alone.
    """

    #: ``(node, name)``: put ``node`` under the label ``name``.
    label_requested = pyqtSignal(int, str)

    def __init__(self, terms, parent=None, *, tip=None) -> None:
        """``tip(name, key)``: a live entry's tooltip, where the entries do something other
        than label the open cluster."""
        super().__init__(parent)
        self._terms = terms
        self._tip = tip
        self._dead_reason = ""
        self._node: int | None = None
        self._own = ""
        self._can_edit = False
        self._choices: dict[str, str | None] = {}
        self._names: list[str] = []
        self._colours: list[QColor] = []
        #: The recent labels, most recent first: entry ``i`` is key ``i + 1``.
        self._menu = QMenu()
        self._menu.setToolTipsVisible(True)
        self._actions: list = []

    def set_labels(self, labels: list[tuple[str, QColor]]) -> None:
        """``[(name, colour)]``, most recent first; only the first :data:`QUICK_LABELS` show."""
        labels = list(labels)[:QUICK_LABELS]
        names = [name for name, _c in labels]
        colours = [QColor(c) for _n, c in labels]
        if names != self._names or [c.rgb() for c in colours] != [
            c.rgb() for c in self._colours
        ]:
            self._names, self._colours = names, colours
            self._rebuild()
        self._sync_enabled()

    def show_node(self, node: int, own: str, choices) -> None:
        self._node = int(node)
        self._own = own
        self._choices = {name: reason for name, _n, reason in choices or []}
        self._sync_enabled()

    def set_editable(self, can_edit: bool, reason: str = "") -> None:
        """``reason``: why the entries are dead when ``can_edit`` is false, said on each."""
        self._can_edit = bool(can_edit)
        self._dead_reason = reason
        self._sync_enabled()

    def names(self) -> list[str]:
        return list(self._names)

    def action(self, index: int):
        """Recent label ``index``'s menu entry (key ``index + 1``)."""
        return self._actions[index]

    def press(self, index: int) -> bool:
        """Key ``index + 1``: request that label if its entry is live; whether it was."""
        if not 0 <= index < len(self._actions):
            return False
        action = self._actions[index]
        if not action.isEnabled():
            return False
        action.trigger()
        return True

    def actions(self) -> list:
        """The entries, key order — what **Labels ▾** leads with."""
        return list(self._actions)

    def _rebuild(self) -> None:
        self._menu.clear()
        self._actions = []
        px = QFontMetrics(QApplication.font()).height() - 2
        for index, (name, colour) in enumerate(zip(self._names, self._colours)):
            # A menu entry carries the label's colour the way the Labels ▾ list does; the
            # no-icon rule is about buttons (`requirements.a_button_carries_no_icon`).
            action = self._menu.addAction(
                QIcon(_swatch_pixmap(colour, px)), f"{index + 1}  {name}"
            )
            action.setData(name)
            action.triggered.connect(
                lambda _c=False, n=name: self._node is not None
                and self.label_requested.emit(self._node, n)
            )
            self._actions.append(action)

    def _sync_enabled(self) -> None:
        unit = self._terms.unit
        for index, (name, action) in enumerate(zip(self._names, self._actions)):
            if not self._can_edit:
                why = self._dead_reason or "This run cannot be edited here."
            elif self._node is None:
                why = f"Pick a {unit} in the tree first."
            elif name == self._own:
                why = f"This {unit} is “{name}” already."
            else:
                why = self._choices.get(name) or ""
            action.setEnabled(not why)
            action.setToolTip(
                why
                or (
                    self._tip(name, index + 1)
                    if self._tip is not None
                    else f"Add this {unit} to “{name}” (key {index + 1})"
                )
            )


class LabelBar(QFrame):
    """The label bar between the tree and the montage (review round 9, option A), in three
    lines (laptop layout, 2026-11-09; the figures' own line, user decision A the same day). A
    cluster window's top is this same bar, built by the same function
    (``node_workspace.build_label_bar``), its name led by the window's number and without the
    page's *Open in window*:

    ``Behavior #1384  [ climb ]                 [the toolbar the page adds]``
    ``214 clips · 3.1 min · 4.2% of the run          Labelled #1384 “climb” · Output up to date``
    ``[ name…                 ] [Label] [Remove] [Labels ▾]  [Undo (⌘Z)] [the page's actions]``

    The figures and the status share a short line under the title, the figures at its left and
    the status at its right: on the title's line beside INTERPRET's montage toolbar (with
    **Export ▾**) the figures kept 179 of the 217px they needed at 1440×847, and the status 80 of
    262 on the field's. Both still shorten with "…" if a window is narrower yet, saying the whole
    line in their tooltip (:class:`sizing.ElidedLabel`). The trailing controls
    (:meth:`add_trailing_widget`, INTERPRET's montage toolbar) sit at the title line's right end. The page fills it (:meth:`show_cluster`, :meth:`set_figures`) and owns every write; the
    bar holds :attr:`field` (the label field) and :attr:`quick` (the recent labels).
    """

    #: The status line's text, each time it changes: a cluster window shows the same line.
    status_changed = pyqtSignal(str)

    def __init__(self, terms, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        box = QVBoxLayout(self)
        box.setContentsMargins(lines(0.5), lines(0.3), lines(0.5), lines(0.3))
        box.setSpacing(lines(0.25))
        first = QHBoxLayout()
        self.name = QLabel("")
        self.name.setFont(emphasis_font(self.name, steps=3))
        first.addWidget(self.name)
        first.addSpacing(8)
        self.chip = LabelChip()
        first.addWidget(self.chip)
        first.addStretch(1)
        #: Where :meth:`add_trailing_widget` puts the page's controls, at the row's right.
        self._trailing = QHBoxLayout()
        self._trailing.setContentsMargins(0, 0, 0, 0)
        first.addLayout(self._trailing)
        box.addLayout(first)
        #: The figures' and the status's line, under the title.
        self.info = QWidget()
        info = QHBoxLayout(self.info)
        info.setContentsMargins(0, 0, 0, 0)
        #: Shortened with "…" only if the line cannot hold it, the whole line in its tooltip.
        self.figures = ElidedLabel("")
        self.figures.setTextFormat(Qt.TextFormat.PlainText)
        info.addWidget(self.figures)
        info.addStretch(1)
        self.status = ElidedLabel("")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        info.addWidget(self.status)
        box.addWidget(self.info)
        #: Where the cluster's label came from, when a difference screen chose it
        #: (:meth:`set_origin`): *From screen cmp_01 (drug vs vehicle · paired): more under
        #: drug than vehicle, 12.4× · kept*. Hidden for a label the user found on their own.
        self.origin = ElidedLabel("")
        self.origin.setTextFormat(Qt.TextFormat.PlainText)
        self.origin.hide()
        box.addWidget(self.origin)
        #: Lines the page puts under the name (the AUDIT line).
        self.lines = QVBoxLayout()
        self.lines.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self.lines)
        second = QHBoxLayout()
        second.setContentsMargins(0, 0, 0, 0)
        self.field = LabelField(terms)
        second.addWidget(self.field, 1)
        #: The recent labels: keys 1–9, and the top of **Labels ▾** (no button of their own).
        self.quick = QuickLabels(terms, self)
        self.field.set_recent(self.quick)
        #: Undoes the last label change made here or in a cluster window (``label_history``);
        #: the page fills it and owns the history.
        self.undo = QPushButton(UNDO_TEXT)
        second.addWidget(self.undo)
        #: Where :meth:`add_action_widget` puts the page's actions on the cluster.
        self._action_row = QHBoxLayout()
        self._action_row.setContentsMargins(0, 0, 0, 0)
        second.addLayout(self._action_row)
        box.addLayout(second)
        self._confirmation = ""
        self._saving = ""
        self._sync_status()

    def add_trailing_widget(self, widget: QWidget) -> None:
        """Put ``widget`` at the right end of the first line (INTERPRET's montage controls)."""
        self._trailing.addSpacing(lines(0.8))
        self._trailing.addWidget(widget)

    def add_action_widget(self, widget: QWidget) -> None:
        """Put ``widget`` on the field's line after **Undo** — an action on the open cluster
        (INTERPRET's *Open in window*). The first line has no room for one on a laptop: with
        it there, every montage control was cut below its own width (1440×847)."""
        self._action_row.addWidget(widget)

    def show_cluster(
        self, name: str, label: str, colour: QColor | None, *, inherited: bool
    ) -> None:
        """The first line's name and chip: ``label`` ``""`` reads "not labelled"; ``inherited``
        draws a parent's label as ``inside “label”``, outlined."""
        self.name.setText(name)
        if not label:
            self.chip.set_label("not labelled", None, filled=False)
        elif inherited:
            self.chip.set_label(f"inside “{label}”", colour, filled=False)
        else:
            self.chip.set_label(label, colour, filled=True)

    def set_figures(self, text: str) -> None:
        self.figures.setText(text)

    def set_origin(self, text: str) -> None:
        """The line under the figures saying which screen chose this cluster; ``""`` hides it."""
        self.origin.setText(text)
        self.origin.setVisible(bool(text))

    def confirm(self, text: str) -> None:
        """What the last write did — "Labelled #1384 “climb”" — ahead of whether it is saved."""
        self._confirmation = text
        self._sync_status()

    def set_saving(self, text: str) -> None:
        """ "Saving…", "Updating the output…", "Output up to date."."""
        self._saving = text
        self._sync_status()

    def set_quick_labels(self, labels: list[tuple[str, QColor]]) -> None:
        self.quick.set_labels(labels)

    def show_status(self, text: str) -> None:
        """Show another bar's status line as it stands — a cluster window's bar follows the
        page's (:attr:`status_changed`), which owns every write."""
        self.status.setText(text)
        self.status.setVisible(bool(text))

    def _sync_status(self) -> None:
        self.status.setText(
            " · ".join(t for t in (self._confirmation, self._saving) if t)
        )
        self.status.setVisible(bool(self.status.text()))
        self.status_changed.emit(self.status.text())
