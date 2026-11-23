"""The label field on INTERPRET (``label_panel.LabelField``), at the size a real vocabulary has.

The fixture run carries a label or two; a run a person has worked through carries dozens, nested.
So the list's rows, their times and their actions are exercised here against labels handed in
directly, not through a fit.
"""

import pytest

from detail.gui.cluster_workspace import POSE_TERMS
from detail.gui.label_panel import LabelField
from detail.gui.qt import QColor, QMenu, QPoint, QTest, Qt

N = 30


@pytest.fixture
def field(qapp):
    field = LabelField(POSE_TERMS)
    rows = [(f"label_{i:02d}", 0, f"label_{i:02d}", (N - i) * 10**9) for i in range(N)]
    rows.insert(1, ("inner", 1, "label_00", 4 * 10**9))
    rows.insert(2, ("other", 1, "label_00", 26 * 10**9))
    field.set_labels(rows, lambda root: QColor("#3a78c2"))
    field.set_editable(True)
    names = [f"label_{i:02d}" for i in range(N)] + ["inner"]
    field.show_node(
        7,
        "",
        "",
        names,
        [(n, 1, None) for n in names[1:]] + [("label_00", 1, "Already inside it.")],
    )
    seen: list = []
    field.label_requested.connect(lambda node, name: seen.append((node, name)))
    field.seen = seen
    return field


def test_every_label_is_listed_with_its_time_and_share(field):
    texts = [a.text() for a in field.labels_menu().actions()]
    assert len(texts) == N + 2
    assert texts[0].startswith("label_00 · 30s · ")
    assert texts[1].startswith("    inner · ") and texts[2].startswith("    other · ")


def test_a_label_the_cluster_cannot_join_is_greyed_with_why(field):
    menu = field.labels_menu().actions()[0].menu()
    join = menu.actions()[0]
    assert not join.isEnabled() and join.toolTip() == "Already inside it."
    open_one = field.labels_menu().actions()[5].menu().actions()[0]
    assert open_one.isEnabled()
    open_one.trigger()
    assert field.edit.text() == "label_03" and field.seen == [], "filled, not written"


def test_other_is_a_note_not_a_label(field):
    other = field.labels_menu().actions()[2].menu().actions()
    assert len(other) == 1 and not other[0].isEnabled()


def test_typing_an_existing_name_says_it_joins_it(field):
    field.edit.setText("LABEL_03")
    assert field.label_button.text() == "Add to “label_03”"
    field.edit.setText("new one")
    assert field.label_button.text() == "Label"
    field.submit()
    assert field.seen == [(7, "new one")]


def test_nothing_is_live_on_a_run_that_cannot_change(field):
    field.set_editable(False)
    assert not field.edit.isEnabled() and field.edit.toolTip()
    menu = field.labels_menu().actions()[3].menu()
    enabled = {a.text(): a.isEnabled() for a in menu.actions() if a.text()}
    assert not any(v for k, v in enabled.items() if not k.startswith("Go to"))
    assert enabled["Go to “label_01”"] is True


def _shown(qapp, field):
    field.resize(600, 40)
    field.show()
    field.activateWindow()
    qapp.processEvents()


def _click_entry(qapp, menu, action):
    """A real click on ``action`` in ``menu``, popped up the way **Labels ▾** pops it."""
    menu.popup(QPoint(0, 0))
    qapp.processEvents()
    QTest.mouseClick(
        menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(action).center()
    )
    qapp.processEvents()


def test_the_first_click_on_the_field_types_with_no_list(qapp, field, monkeypatch):
    """The user's report: the first click on the empty field opened every label, which took
    the click, so typing needed a second one."""
    opened = []
    monkeypatch.setattr(QMenu, "exec", lambda *_a, **_k: opened.append(True))
    monkeypatch.setattr(QMenu, "popup", lambda *_a, **_k: opened.append(True))
    _shown(qapp, field)
    try:
        field.edit.setText("")
        QTest.mouseClick(field.edit, Qt.MouseButton.LeftButton, pos=QPoint(4, 4))
        qapp.processEvents()
        assert opened == [], "a click in the field opened a list"
        QTest.keyClicks(field.edit, "rear")
        assert field.edit.text() == "rear"
        assert not field._completer.popup().isVisible() or field.edit.hasFocus()
        assert field.seen == []
    finally:
        field.hide()


def test_choosing_a_listed_label_fills_the_field_and_enter_writes_it(qapp, field):
    _shown(qapp, field)
    try:
        menu = field.labels_menu()
        row = menu.actions()[5]
        assert row.menu() is not None and row.data() == "label_03"
        _click_entry(qapp, menu, row)
        assert (
            field.edit.text() == "label_03" and field.seen == []
        ), "filled, not written"
        assert field.edit.selectedText() == "label_03"
        assert not menu.isVisible()
        QTest.keyClick(field.edit, Qt.Key.Key_Return)
        assert field.seen == [(7, "label_03")]
    finally:
        field.hide()


def test_a_label_the_cluster_cannot_join_fills_nothing(qapp, field):
    _shown(qapp, field)
    try:
        menu = field.labels_menu()
        _click_entry(qapp, menu, menu.actions()[0])
        menu.hide()
        assert field.edit.text() == "" and field.seen == []
        assert menu.actions()[0].toolTip() == "Already inside it."
    finally:
        field.hide()


def test_a_numbered_entry_fills_the_field_and_its_key_still_labels(qapp, field):
    from detail.gui.label_panel import QuickLabels

    quick = QuickLabels(POSE_TERMS, field)
    quick.set_labels([("label_04", QColor("#3a78c2")), ("label_05", QColor("red"))])
    quick.set_editable(True)
    quick.show_node(7, "", [("label_04", 1, None), ("label_05", 1, None)])
    keyed: list = []
    quick.label_requested.connect(lambda node, name: keyed.append((node, name)))
    field.set_recent(quick)
    _shown(qapp, field)
    try:
        menu = field.labels_menu()
        entries = menu.actions()
        assert entries[0].text() == "Recent (keys 1–9)"
        assert [a.text() for a in entries[1:3]] == ["1  label_04", "2  label_05"]
        assert entries[1] is not quick.action(0), "the menu's entry is not the key's"
        _click_entry(qapp, menu, entries[2])
        assert field.edit.text() == "label_05"
        assert field.seen == [] and keyed == [], "the entry wrote"
        assert quick.press(0) and keyed == [(7, "label_04")], "the key labels at once"
    finally:
        field.hide()


def test_enter_on_a_suggestion_fills_the_field_and_a_second_enter_writes(qapp, field):
    _shown(qapp, field)
    try:
        field.edit.setText("")
        QTest.mouseClick(field.edit, Qt.MouseButton.LeftButton, pos=QPoint(4, 4))
        QTest.keyClicks(field.edit, "inne")
        popup = field._completer.popup()
        qapp.processEvents()
        assert popup.isVisible(), "no suggestion to choose"
        QTest.keyClick(popup, Qt.Key.Key_Down)
        QTest.keyClick(popup, Qt.Key.Key_Return)
        qapp.processEvents()
        assert field.edit.text() == "inner" and field.seen == [], "the suggestion wrote"
        QTest.keyClick(field.edit, Qt.Key.Key_Return)
        assert field.seen == [(7, "inner")]
    finally:
        field.hide()


def test_a_choice_from_the_labels_button_leaves_the_cursor_in_the_field(qapp, field):
    """Through the real button: the menu runs modally and hands focus back to the button as it
    closes, so Enter after a choice would do nothing unless the field takes the focus back.
    """
    from detail.gui.qt import QTimer

    _shown(qapp, field)
    chosen: list = []

    def choose():
        menu = qapp.activePopupWidget()
        row = next(a for a in menu.actions() if a.data() == "label_03")
        QTest.mouseClick(
            menu, Qt.MouseButton.LeftButton, pos=menu.actionGeometry(row).center()
        )
        chosen.append(True)

    try:
        field.list_button.setFocus()
        QTimer.singleShot(50, choose)
        QTest.mouseClick(field.list_button, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert chosen and field.edit.text() == "label_03" and field.seen == []
        # The offscreen platform leaves no window active once a popup closes; a real one
        # reactivates the field's window, which gives the focus to its focus widget.
        field.activateWindow()
        qapp.processEvents()
        assert field.edit.hasFocus(), "the cursor is not in the field"
        QTest.keyClick(field.edit, Qt.Key.Key_Return)
        assert field.seen == [(7, "label_03")]
    finally:
        field.hide()


def test_opening_labels_menu_leaves_no_menu_behind(field, qapp):
    """Review: every click on **Labels ▾** built a menu parented to the field and never deleted
    it, so a session of labelling grew one menu (and ~30 submenus) per click. Opened by a real
    click on the button; each time the menu closes on Esc, as a person dismisses it."""
    from detail.gui.qt import QApplication, QCoreApplication, QEvent, QTimer

    field.show()
    qapp.processEvents()

    def dismiss():
        menu = QApplication.activePopupWidget()
        assert menu is not None, "Labels ▾ opened no menu"
        QTest.keyClick(menu, Qt.Key.Key_Escape)

    before = len(field.findChildren(QMenu))
    for _ in range(3):
        QTimer.singleShot(50, dismiss)
        QTest.mouseClick(field.list_button, Qt.MouseButton.LeftButton)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        qapp.processEvents()
    assert len(field.findChildren(QMenu)) == before
