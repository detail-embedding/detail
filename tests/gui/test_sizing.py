"""``gui/sizing.py``'s helpers: the fine-print font the strips use, and the elided label.

**The class: a text label cut by its layout mid-letter, with nothing to say so.** A ``QLabel``
shorter than its text is clipped, and the label bar's figures read "…of the run · sepa" into the
chip beside them. ``sizing.ElidedLabel`` shortens with "…" and says the whole line in its
tooltip, while ``text()`` stays whole.

**Its sibling: a value's text setting a panel's minimum width** inside a scroll area that never
scrolls sideways. Each filter a user picked raised the filters column's minimum until it grew
past the panel and cut its right edge, so an elided label's floor is a constant, never its text.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel  # noqa: E402

from detail.gui.sizing import small_font  # noqa: E402


def test_small_font_is_smaller_than_the_widgets_own(qapp):
    label = QLabel("fine print")
    assert small_font(label).pointSizeF() < label.font().pointSizeF()


def test_an_elided_label_shortens_with_an_ellipsis_and_says_the_whole_line(qapp):
    """The label bar's figures were clipped mid-letter ("…of the run · sepa") with no way to
    read the rest: a cut line ends in "…" and its tooltip is the whole line."""
    from detail.gui.sizing import ElidedLabel

    line = "182 clips · 5.7 min · 9.6% of the run · separation 1.24 (run median 0.80)"
    label = ElidedLabel(line)
    label.resize(label.sizeHint())
    label.show()
    qapp.processEvents()
    try:
        assert not label.is_elided() and label.toolTip() == ""
        label.resize(120, label.height())
        qapp.processEvents()
        assert label.text() == line, "text() is always the whole line"
        assert label.is_elided() and label.shown_text().endswith("…")
        assert label.toolTip() == line
        assert label.minimumSizeHint().width() < 120 < label.sizeHint().width()
        label.set_extra_tooltip("why it matters")
        assert label.toolTip() == f"{line}\nwhy it matters"
    finally:
        label.close()


def test_an_elided_labels_floor_is_a_constant_never_its_text(qapp):
    """Its minimum width is a few characters whatever the line, so no value can widen the row it
    sits in (the filters panel, 2026-11-09); ``min_chars`` raises that floor for a label its row
    should cut last, and it is still a constant."""
    from detail.gui.sizing import ElidedLabel

    short, long_ = ElidedLabel("x" * 40), ElidedLabel("x" * 400)
    assert short.minimumSizeHint().width() == long_.minimumSizeHint().width()
    held = ElidedLabel("x" * 400, min_chars=9)
    assert long_.minimumSizeHint().width() < held.minimumSizeHint().width()
    assert (
        held.minimumSizeHint().width()
        == ElidedLabel("y" * 40, min_chars=9).minimumSizeHint().width()
    )
    assert (
        ElidedLabel("ab", min_chars=9).minimumSizeHint().width()
        <= held.minimumSizeHint().width()
    )
