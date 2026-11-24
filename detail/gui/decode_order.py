"""Which thumbnails to decode first: the ones somebody is looking at.

A strip of clips holds far more cells than fit on screen, and each one costs a video decode. Fire
them all at once, in index order, and the pictures a person is actually looking at are queued
behind everything to their left — on a large behavior, seconds before the visible row
filled and more churn behind it, which also starves the player's own decode because they share
the global pool.

**One rule, written for two strips.** The query CALIBRATE's confidence strip grew this ordering
while somebody was looking at that page, and `ClusterClipStrip` (clustering) never got it and fired
every clip at once — the recorded class *two surfaces answering the same question with two
mechanisms* — so the ordering moved here, as a function of geometry. The confidence strip is gone
(CALIBRATE plays the shared montage, 2026-11); the cluster strip still asks.

Kept Qt-light on purpose: it reads `x()` / `width()` off whatever it is handed, so it is testable
against plain stand-ins and the strips stay responsible for their own layout and caching.
"""

#: How many decodes to queue in one pass. A screenful and its margins, not the whole strip: the
#: rest arrives as the user scrolls. Sized above a full row so a wide window still fills in one go.
DECODE_BUDGET = 24


def visible_first(
    cells: dict,
    *,
    viewport_left: int,
    viewport_width: int,
    cell_px: int,
    centre: int = 0,
) -> list[int]:
    """The indices in ``cells`` worth decoding now, nearest ``centre`` first.

    ``cells`` maps index to widget. The result is bounded to what is on screen plus a margin
    either side, then ordered by distance from ``centre`` — the index the view is centred on, or
    about to be.

    **The caller must have laid the row out first.** A cell that Qt has not positioned reports
    ``width() == 0`` at an ``x()`` a few pixels from its neighbours, so the on-screen test admits
    *every* cell and the bound silently does nothing. That failure is invisible — the pictures
    still arrive, just in the wrong order — which is why the check below is on the viewport rather
    than on the cells: a viewport narrower than one cell means the geometry has not settled, and
    the honest answer then is to order without filtering rather than to filter on nonsense.
    """
    if not cells:
        return []
    margin = cell_px * 2  # a screen's worth either side, so a scroll finds them warm
    usable = viewport_width > cell_px and any(c.width() for c in cells.values())

    wanted = []
    for index, cell in cells.items():
        if usable and (
            cell.x() + cell.width() < viewport_left - margin
            or cell.x() > viewport_left + viewport_width + margin
        ):
            continue
        wanted.append(index)
    return sorted(wanted, key=lambda index: abs(index - centre))


def centre_index(cells: dict, *, viewport_left: int, viewport_width: int) -> int:
    """The index nearest the middle of the viewport, or 0 before there is a layout to ask.

    Zero rather than a guess: every strip here scrolls to the left when its contents change, so
    before the first layout the left edge *is* where the user is about to be looking.
    """
    if not cells:
        return 0
    middle = viewport_left + viewport_width // 2
    positioned = {i: c for i, c in cells.items() if c.width()}
    if not positioned:
        return 0
    return min(
        positioned,
        key=lambda i: abs(positioned[i].x() + positioned[i].width() // 2 - middle),
    )
