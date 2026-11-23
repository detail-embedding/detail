"""A band of a result view that folds away, and remembers that it was folded.

The class it addresses: **a workspace whose parts all insist on their space at once.** The
temporal-clustering INTERPRET tab stacks a 3-D scatter, a cluster table, a strip of frames and a
row of ethograms; each is worth its height while you are using it and none of them is worth it
while you are using another. Their floors sum to 887px against the 739px a 13" laptop can show,
so on that machine the tab could not be laid out at all — and lowering any one floor only makes
that part unusable in exchange.

Folding is the only move that changes the sum rather than redistributing it. A hidden widget is
outside its layout entirely, so its floor and its preferred height both leave with it: measured on
this tab, folding the ethogram band takes the floor from 843px to 672px.

**Sections fold by their header; panes fold to a labelled strip.** Krita shipped per-pane
collapse and then removed it — *"even Krita power users were not using these options. New users to
Krita seemed unable to grasp this functionality altogether."* — because a collapsed pane was an
unlabelled sliver with no affordance. A collapsed *section* keeps its title in place, so it reads
as a thing you closed and can open: every band here keeps a titled header whether it is open or
shut. A side *pane* folds by its splitter's grip to a strip with its name written down it
(`gui/grip_splitter.py`), which answers the same objection from the other side.

Both bands open by default. The frames are how a syllable is recognised and named, so a screen
that opens without them is a screen that has hidden the evidence; the page scrolls (`gui/scroll.py`)
rather than the bands starting shut.
"""

from detail.gui.qt import (
    QApplication,
    QPushButton,
    QSettings,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    Signal,
)
from detail.gui.sizing import ResizeHandle

#: Where a band's open/shut state is kept, per band key.
_SETTINGS_PREFIX = "bands/"


class CollapsibleBand(QWidget):
    """A titled header over a widget that can be folded away.

    ``key`` is what the state is remembered under, so a band keeps its position across
    navigations — the page is rebuilt on every one (``MainWindow._show`` discards it), and a fold
    that reset each time would be a control nobody could rely on.

    Deliberately not ``forms.AdvancedSection``, which shares the look and none of the job: that one
    holds ``QFormLayout`` rows in named groups, tracks which of them differ from the values the
    form opened with, and titles itself from that count. This holds one arbitrary widget. The two
    have the same header glyph on purpose and no shared implementation, because the alternative is
    a base class with two subclasses that have nothing in common but a triangle.
    """

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        key: str,
        open_: bool = True,
        on_resize=None,
    ):
        """``on_resize`` makes the band draggable as well as foldable.

        Given a callable, the band grows a `ResizeHandle` under its content and hands it the
        pixel delta of each drag. What a drag *means* stays with the content — the clip strip
        trades how many members fit against how big they are, the map trades how many recordings
        fit against how well a corner bias reads — so the band carries the gesture and never the
        size.

        A band without one is a section you can only fold. That was every band but two, and a
        band you can drag beside a band you cannot is what got reported: `ResizeHandle`'s own
        docstring has said so since it was written, on a screen that had exactly that.
        """
        super().__init__()
        self._title = title
        self._key = key
        self._content = content

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        # A plain framed button, matching `forms.AdvancedSection` — and, like it, deliberately not
        # a `QToolButton`: macOS registers a smaller class-specific default font for that class,
        # so the header came out visibly smaller than the labels beside it.
        self._header = QPushButton()
        self._header.setFont(QApplication.font("QLabel"))
        self._header.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self._header.clicked.connect(self._on_clicked)
        box.addWidget(self._header)
        box.addWidget(content)
        #: Folded, the handle goes with the content: a drag bar under a section with nothing in
        #: it resizes nothing, and `test_the_frames_caption_folds_with_its_frames` asserts that
        #: folding takes the band's parts off screen rather than leaving one behind.
        self._handle: ResizeHandle | None = None
        #: Whether dragging would change anything right now (`set_resizable`).
        self._resizable = True
        if on_resize is not None:
            self._handle = ResizeHandle()
            self._handle.dragged.connect(on_resize)
            box.addWidget(self._handle)
        self.set_open(self._remembered(default=open_), remember=False)

    # -- state ------------------------------------------------------------------------ #
    def is_open(self) -> bool:
        return self._content.isVisibleTo(self)

    def set_open(self, open_: bool, *, remember: bool = True) -> None:
        self._content.setVisible(open_)
        if self._handle is not None:
            self._handle.setVisible(open_ and self._resizable)
        self._header.setText(f"{'▼' if open_ else '▶'}  {self._title}")
        if remember:
            QSettings().setValue(
                f"{_SETTINGS_PREFIX}{self._key}", "1" if open_ else "0"
            )

    def set_resizable(self, enabled: bool) -> None:
        """Show the drag handle only while dragging it would change something.

        A band whose content already shows all of itself — an ethogram with every strip on
        screen — has nothing to give a drag, and a handle that moves nothing reads as broken.
        """
        self._resizable = enabled
        if self._handle is not None:
            self._handle.setVisible(enabled and self.is_open())

    def _remembered(self, *, default: bool) -> bool:
        stored = QSettings().value(f"{_SETTINGS_PREFIX}{self._key}", None)
        return default if stored is None else str(stored) == "1"

    def _on_clicked(self) -> None:
        self.set_open(not self.is_open())
        self.toggled.emit(self.is_open())
