"""Whether two marks can be told apart with the hue taken away — the identity guards' measure.

A mark is compared the way a reader without colour vision sees it: as **luminance**, and by its
**texture** rather than its overall lightness. Lightness alone is not counted as a second
channel. The label colours were chosen for hue, so how light each one is is an accident, and two
of them sit 15 levels apart out of 255 (slot 0's green, slot 8's mauve): near enough to "differ"
in a pixel comparison and not to a reader.

Texture is each pixel's sign against the mark's own mean — lighter, darker or neither — so a fill
in any colour reads as "no texture", and one pattern in two colours reads as one texture.
Two marks are compared at their **best alignment**, because a pattern is anchored to the widget
it is drawn on: the same pattern at two places on a strip starts at two phases, and a comparison
that ignored that would call one identity two.
"""

import numpy as np

from detail.gui.qt import QImage

#: How far, out of 255, a pixel must sit from its mark's mean lightness to be part of a texture —
#: above the edge noise antialiasing leaves, below the weakest stroke a pattern draws (white on
#: the lightest label colour, 80 levels).
CONTRAST = 24.0

#: The share of pixels whose texture must differ, at the best alignment, for two marks to read
#: as two.
APART = 0.10


def luminance(image) -> np.ndarray:
    """``image`` (a ``QImage`` or ``QPixmap``) as a float array of luminance, 0-255."""
    if not isinstance(image, QImage):
        image = image.toImage()
    grey = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width, height, stride = grey.width(), grey.height(), grey.bytesPerLine()
    # PySide6's `constBits()` is a buffer of the image's own size; no sizing step.
    rows = np.frombuffer(grey.constBits(), np.uint8)[: stride * height]
    rows = rows.reshape(height, stride)
    return rows[:, :width].astype(float)


def texture(grey: np.ndarray) -> np.ndarray:
    """Each pixel's sign against the mark's mean: +1 lighter, -1 darker, 0 neither."""
    centred = grey - grey.mean()
    return np.where(centred > CONTRAST, 1, np.where(centred < -CONTRAST, -1, 0))


def difference(a: np.ndarray, b: np.ndarray, *, shift: int = 0) -> float:
    """The share of pixels whose texture differs, at the alignment that makes it smallest.

    ``shift`` bounds the search, in pixels each way; 0 compares the two as drawn.
    """
    ta, tb = texture(a), texture(b)
    height = min(ta.shape[0], tb.shape[0])
    width = min(ta.shape[1], tb.shape[1])
    ta, tb = ta[:height, :width], tb[:height, :width]
    best = 1.0
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            # Overlap of `ta` with `tb` moved by (dx, dy); never less than half of either.
            ya, yb = max(dy, 0), max(-dy, 0)
            xa, xb = max(dx, 0), max(-dx, 0)
            h, w = height - abs(dy), width - abs(dx)
            if h < height / 2 or w < width / 2:
                continue
            here = ta[ya : ya + h, xa : xa + w] != tb[yb : yb + h, xb : xb + w]
            best = min(best, float(here.mean()))
    return best


def apart(a: np.ndarray, b: np.ndarray, *, shift: int = 0) -> bool:
    """Whether two marks read as two in greyscale, by texture and not by lightness alone."""
    return difference(a, b, shift=shift) >= APART


def pairs_together(marks: dict, *, shift: int = 0) -> list[tuple]:
    """Every pair of ``{name: luminance}`` that does **not** read apart, with its difference."""
    names = list(marks)
    return [
        (one, other, round(difference(marks[one], marks[other], shift=shift), 3))
        for i, one in enumerate(names)
        for other in names[i + 1 :]
        if not apart(marks[one], marks[other], shift=shift)
    ]
