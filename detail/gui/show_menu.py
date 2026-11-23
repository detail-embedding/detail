"""What a montage's tiles show — one definition of the layers, and the **Show ▾** menu.

The layers a tile can draw — **Video**, **Masks**, **Mask trail**, **Keypoints** — and the
orientation (**As recorded** / **Aligned**) are a ``clip_tiles.TileLook``. Every surface that
offers them reads :data:`LAYERS` for their names and tips and :func:`layer_states` for which may
change and why not, so the cluster window's Show row and INTERPRET's and AUDIT's **Show ▾** say
the same thing the same way. What an export writes from a look is :func:`layers_of`, and how it
is said is :func:`describe_look`, so the bar above the export buttons cannot drift from the
picture.

**Show ▾** (:class:`ShowButton`, 2026-11-09) replaced INTERPRET's *Show masks* check box and its
*As recorded ▾* menu with one button: the four layers, checkable, then the two orientations as a
radio pair, then the **context** either side of each clip (none, 1 s, 2 s, 5 s — the cluster
window's Context box until "one montage toolbar everywhere", 2026-11-09). A layer that cannot
change is greyed with the reason — the last one a tile draws, a trail without its masks,
keypoints the embedding never read — and **Aligned** with why the run cannot align. Every clip
grid carries the same button (``live_montage.LiveMontage`` builds it), and each page keeps its
choice under its own keys (:func:`remembered_look`, the montage's ``context_key``).
"""

from __future__ import annotations

from dataclasses import replace

from detail.api.clip_tiles import TileLook
from detail.gui.qt import (
    QAction,
    QActionGroup,
    QMenu,
    QPoint,
    QPushButton,
    QSettings,
    QSizePolicy,
    Signal,
)

#: ``(TileLook field, label, tooltip)`` — every layer a tile can draw, in menu order.
LAYERS: tuple[tuple[str, str, str], ...] = (
    ("footage", "Video", "The video. Off: the masks and keypoints on white."),
    ("masks", "Masks", "The animal's mask: outlined on the video, filled on white."),
    (
        "mask_trail",
        "Mask trail",
        "Where the animal's outline was over the last half second, fading.",
    ),
    ("keypoints", "Keypoints", "The keypoint run the embedding read."),
)
#: Why the last layer a tile draws cannot be unticked.
LAST_LAYER = "A tile has to show something."
#: Why the trail is greyed with the masks off.
TRAIL_NEEDS_MASKS = "The trail follows the masks: tick Masks."
#: What a look is said as, in an export's line: ``video + masks``.
_WORDS = {
    "footage": "video",
    "masks": "masks",
    "mask_trail": "mask trail",
    "keypoints": "keypoints",
}
_DRAWN = ("footage", "masks", "keypoints")
#: The context choices, in seconds of footage either side of each clip (the clip player's).
CONTEXT_CHOICES_S = (0.0, 1.0, 2.0, 5.0)


def context_words(seconds: float) -> str:
    """``no context`` / ``1 s context``: a context as an export's line says it."""
    return f"{seconds:g} s context" if seconds else "no context"


def layer_states(
    look: TileLook, points_note: str | None
) -> dict[str, tuple[bool, str]]:
    """``{field: (can change, why not)}`` for each layer under ``look``.

    The last of video / masks / keypoints a tile draws cannot be unticked; the trail needs the
    masks; keypoints need a run that has them (``points_note`` says why not) — though ticked ones
    can still be unticked, so a remembered tick is never stuck on.
    """
    drawn = [key for key in _DRAWN if getattr(look, key)]
    out: dict[str, tuple[bool, str]] = {}
    for key, _label, _tip in LAYERS:
        if key == "keypoints" and points_note is not None:
            out[key] = (bool(look.keypoints), points_note)
        elif key == "mask_trail":
            out[key] = (bool(look.masks), TRAIL_NEEDS_MASKS)
        else:
            out[key] = (drawn != [key], LAST_LAYER)
    return out


def describe_look(look: TileLook) -> str:
    """``video + masks · aligned``: what tiles drawn with ``look`` show, in words."""
    shown = [
        _WORDS[key]
        for key, _label, _tip in LAYERS
        if getattr(look, key) and (key != "mask_trail" or look.masks)
    ]
    return f"{' + '.join(shown)} · {'aligned' if look.aligned else 'as recorded'}"


def layers_of(look: TileLook):
    """An export's layers (``api.clip_montage.MontageLayers``) for exactly what ``look`` shows."""
    from detail.api.clip_montage import MontageLayers

    return MontageLayers(
        raw=look.footage,
        mask=look.masks,
        keypoints=look.keypoints,
        mask_trail=look.masks and look.mask_trail,
        aligned=look.aligned,
    )


def remembered_look(key: str, *, masks_key: str | None = None) -> TileLook:
    """The look kept under ``key`` (``remember_look``), the orientation aside.

    ``masks_key``: the older *Show masks* setting to start from when ``key`` holds nothing yet
    — read once, so a page that had masks off opens with them off.
    """
    stored = QSettings().value(key)
    if isinstance(stored, str):
        names = set(stored.split(",")) if stored else set()
        try:
            return TileLook(
                **{field: field in names for field, _l, _t in LAYERS}, aligned=False
            )
        except ValueError:
            pass
    if masks_key is not None:
        masks = str(QSettings().value(masks_key, "1")) not in ("0", "false")
        return TileLook(masks=masks, mask_trail=masks)
    return TileLook()


def remember_look(key: str, look: TileLook) -> None:
    QSettings().setValue(
        key, ",".join(field for field, _l, _t in LAYERS if getattr(look, field))
    )


class ShowButton(QPushButton):
    """**Show ▾**: the layers, checkable, then **As recorded** / **Aligned** (module docstring).

    The owner keeps the look and the orientation and pushes them in (:meth:`sync`); the button
    only says what was asked for.
    """

    #: ``(field, on)``: a layer ticked or unticked.
    layer_toggled = Signal(str, bool)
    #: The orientation chosen: whether **Aligned**.
    aligned_wanted = Signal(bool)
    #: The context chosen, in seconds either side of each clip.
    context_wanted = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__("Show ▾", parent)
        self.setObjectName("montage_show")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolTip(
            "What the tiles show, which way they face, and the footage either side"
        )
        self.menu = QMenu(self)
        self.menu.setToolTipsVisible(True)
        #: The layer entries, by ``TileLook`` field.
        self.layer_actions: dict[str, QAction] = {}
        for key, label, tip in LAYERS:
            action = self.menu.addAction(label)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda on, k=key: self.layer_toggled.emit(k, bool(on))
            )
            self.layer_actions[key] = action
        self.menu.addSeparator()
        group = QActionGroup(self)
        group.setExclusive(True)
        #: ``{False: As recorded, True: Aligned}``.
        self.orientation_actions: dict[bool, QAction] = {}
        for aligned, label in ((False, "As recorded"), (True, "Aligned")):
            action = self.menu.addAction(label)
            action.setCheckable(True)
            group.addAction(action)
            action.triggered.connect(
                lambda _c=False, a=aligned: self.aligned_wanted.emit(a)
            )
            self.orientation_actions[aligned] = action
        self.menu.addSeparator()
        context = QActionGroup(self)
        context.setExclusive(True)
        #: ``{seconds: action}`` — the context either side of each clip.
        self.context_actions: dict[float, QAction] = {}
        for seconds in CONTEXT_CHOICES_S:
            label = "No context" if not seconds else f"{seconds:g} s context"
            action = self.menu.addAction(label)
            action.setCheckable(True)
            action.setToolTip(
                "Video either side of each clip: every tile plays the clip and its context, "
                "and an export writes it."
            )
            context.addAction(action)
            action.triggered.connect(
                lambda _c=False, s=seconds: self.context_wanted.emit(s)
            )
            self.context_actions[seconds] = action
        self.clicked.connect(self.open_menu)

    def open_menu(self) -> None:
        self.menu.exec(self.mapToGlobal(QPoint(0, self.height())))

    def sync(
        self,
        look: TileLook,
        *,
        aligned: bool,
        align_note: str | None,
        align_tip: str,
        points_note: str | None,
        context_s: float,
    ) -> None:
        """Show ``look``, the orientation and the context; grey what cannot change, with
        why."""
        states = layer_states(look, points_note)
        for key, label, tip in LAYERS:
            action = self.layer_actions[key]
            action.setChecked(bool(getattr(look, key)))
            enabled, why = states[key]
            action.setEnabled(enabled)
            action.setToolTip(tip if enabled else why)
        self.orientation_actions[bool(aligned)].setChecked(True)
        align = self.orientation_actions[True]
        align.setEnabled(align_note is None)
        align.setToolTip(align_tip if align_note is None else align_note)
        self.context_actions[float(context_s)].setChecked(True)


def toggled(look: TileLook, key: str, on: bool) -> TileLook:
    """``look`` with layer ``key`` ticked or unticked."""
    return replace(look, **{key: bool(on)})
