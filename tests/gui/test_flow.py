"""``FlowLayout`` — a control row that wraps instead of being squeezed.

The class it exists for is measured in its own module docstring: **a control row inside a
player is handed the video's width, and nothing enforces its minimum.**
``PictureWithControls`` positions its children with ``setGeometry`` rather than through a
``QLayout``, so the row gets the block's width whatever it asked for — and Qt's answer to a
button narrower than its text is to elide the *label*, silently. Reported from the app as
"Prev frame" rendering as "ev fram".

**The trailing group never wraps alone** (recorded class): a row short by less than the
readout's width used to wrap the readout and nothing else — the clip window's frame counter on
a line of its own under a full line of buttons at 1324px. The last control goes down with it.

Every test here measures a laid-out widget rather than a policy. A layout that is never given
a geometry has placed nothing, and an unshown widget's children all report zero — which is the
state in which every one of these assertions would pass against a broken layout.
"""

from detail.gui.flow import FlowLayout
from detail.gui.qt import QApplication, QLabel, QPushButton, QWidget


def _row(qapp, width, *, labels=("alpha", "bravo", "charlie", "delta"), trailing=None):
    """A host widget ``width`` wide holding ``labels`` as buttons, laid out and shown."""
    host = QWidget()
    layout = FlowLayout(host)
    buttons = [QPushButton(text) for text in labels]
    for button in buttons:
        layout.addWidget(button)
    tail = None
    if trailing is not None:
        layout.addStretch()
        tail = QLabel(trailing)
        layout.addWidget(tail)
    host.resize(width, layout.heightForWidth(width))
    host.show()
    QApplication.processEvents()
    return host, buttons, tail


def test_a_row_that_fits_stays_on_one_line(qapp):
    host, buttons, _ = _row(qapp, 1200)
    try:
        assert len({b.y() for b in buttons}) == 1, [b.geometry() for b in buttons]
    finally:
        host.hide()


def test_a_row_that_does_not_fit_wraps_rather_than_squeezing_its_labels(qapp):
    """The whole point. Both halves are asserted, because either alone is satisfiable by a
    layout that is simply wrong in the other direction: a row that wraps but still squeezes,
    and a row that never squeezes because it never lays anything out."""
    widest = max(QPushButton(t).sizeHint().width() for t in ("alpha", "charlie"))
    host, buttons, _ = _row(qapp, widest + 20)
    try:
        assert len({b.y() for b in buttons}) > 1, "the row did not wrap"
        squeezed = [
            (b.text(), b.width(), b.sizeHint().width())
            for b in buttons
            if b.width() < b.sizeHint().width()
        ]
        assert not squeezed, f"labels elided instead of wrapping: {squeezed}"
    finally:
        host.hide()


def test_the_trailing_group_sits_at_the_right_edge_of_its_line(qapp):
    """``addStretch`` means "push what follows to the right", and one caller uses it that way
    — the frame readout sits right while the buttons sit left. A flow layout has no single
    right edge, so the spacer is kept as a marker; on a row that fits, the picture it draws is
    the one ``QHBoxLayout`` drew, which is what makes this a drop-in replacement."""
    host, buttons, tail = _row(qapp, 1200, trailing="245 / 479")
    try:
        assert tail.y() == buttons[0].y(), "the readout left the buttons' line"
        # The content area's right edge, not the widget's: the layout keeps its margins, and a
        # test that ignores them is measuring the margin rather than the alignment.
        _l, _t, right_margin, _b = host.layout().getContentsMargins()
        edge = host.width() - right_margin
        assert tail.geometry().right() >= edge - 2, (
            f"the readout ends at {tail.geometry().right()}, not at the content edge {edge} "
            f"of a {host.width()}px row — it was not pushed right"
        )
        assert (
            tail.x() > buttons[-1].geometry().right()
        ), "the readout overlaps the buttons"
    finally:
        host.hide()


def test_the_floor_is_one_control_not_the_whole_row(qapp):
    """A row whose minimum is its own full width is a row with no answer but eliding — which
    is exactly the state ``FrameScrubber`` was in (878px floor, handed 357px)."""
    host = QWidget()
    layout = FlowLayout(host)
    buttons = [QPushButton(t) for t in ("alpha", "bravo", "charlie", "delta")]
    for button in buttons:
        layout.addWidget(button)
    widest = max(b.minimumSizeHint().width() for b in buttons)
    total = sum(b.sizeHint().width() for b in buttons)
    left, _t, right, _b = layout.getContentsMargins()
    floor = layout.minimumSize().width()
    assert (
        floor <= widest + left + right
    ), f"floor {floor} against a widest item of {widest}"
    assert floor < total / 2, f"floor {floor} is most of the row's {total}px"


def test_the_height_it_promises_is_the_height_it_takes(qapp):
    """``heightForWidth`` is what a host reserves space with, and ``setGeometry`` is what
    actually places the items. A layout whose two answers disagree clips its own last line.
    """
    for width in (1200, 700, 400, 240):
        host, buttons, _ = _row(qapp, width)
        try:
            promised = host.layout().heightForWidth(width)
            used = max(b.geometry().bottom() for b in buttons) + 1
            assert used <= promised, f"at {width}px: placed {used}px into {promised}px"
        finally:
            host.hide()


def test_a_widget_inserted_mid_row_lands_where_it_was_asked_for(qapp):
    """``FrameScrubber.add_control`` inserts a host's checkbox after the step buttons, so the
    insert has to respect the index rather than appending — ``QLayout`` has no insert of its
    own and ``addWidget`` goes through ``addItem``, which appends."""
    host = QWidget()
    layout = FlowLayout(host)
    first, last = QPushButton("first"), QPushButton("last")
    layout.addWidget(first)
    layout.addWidget(last)
    inserted = QPushButton("inserted")
    layout.insertWidget(1, inserted)
    assert [layout.itemAt(i).widget() for i in range(layout.count())] == [
        first,
        inserted,
        last,
    ]
    assert inserted.parentWidget() is host, "an inserted widget was never reparented"


def test_the_trailing_group_never_wraps_onto_a_line_by_itself(qapp):
    """A row short by less than the readout's width wrapped the readout and nothing else — the
    clip window's frame counter alone under a full line of buttons at 1324px. The last control
    goes down with it, and the readout still ends that line on the right."""
    labels = ("alpha", "bravo", "charlie", "delta")
    probe, probe_buttons, probe_tail = _row(
        qapp, 2000, labels=labels, trailing="245 / 479"
    )
    one_line = probe.layout().sizeHint().width()
    probe.hide()
    _l, _t, right, _b = probe.layout().getContentsMargins()
    host, buttons, tail = _row(
        qapp,
        one_line - probe_tail.sizeHint().width() // 2,
        labels=labels,
        trailing="245 / 479",
    )
    try:
        assert (
            len({b.y() for b in buttons}) == 2
        ), "the row must wrap for the check to bite"
        assert buttons[0].y() != tail.y()
        assert buttons[-1].y() == tail.y(), "the readout is alone on its line"
        assert tail.geometry().right() >= host.width() - right - 2
        assert host.height() >= tail.geometry().bottom()
    finally:
        host.hide()


def test_a_trailing_group_after_a_lone_control_is_not_carried(qapp):
    """Carrying the only control on a line would empty it: then the readout wraps as before."""
    widest = QPushButton("alpha").sizeHint().width()
    host, buttons, tail = _row(
        qapp, widest + 4, labels=("alpha",), trailing="245 / 479"
    )
    try:
        assert tail.y() > buttons[0].y()
    finally:
        host.hide()
