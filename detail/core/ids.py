"""Ordering for generated ids.

Every id the tool mints is ``<stem>_<n>`` zero-padded to a fixed width — ``run_007``,
``an_04``, ``clusterer_012``. Sorting those as strings is right only while the number stays
inside the pad: at the boundary ``an_100`` sorts before ``an_99`` and ``run_1000`` before
``run_999``, so every listing built on ``sorted()`` silently reorders itself once an
experiment grows past it.

An experiment with a few hundred analysis runs (2-digit pad) sorts ``an_100`` and up
into the middle, and the Analysis table, the project rail and
``detail analysis list`` all end at ``an_99``.

Ids are opaque handles. This module is the one place allowed to read the number back out of
one, and only to order them.
"""

import re
from collections.abc import Iterable

_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")


def numeric_id_key(name: str) -> tuple[str, int, str]:
    """Sort key that orders a generated id by its number, not its text.

    Returns ``(stem, number, name)``. A name with no trailing number sorts by its text with
    a number of ``-1``, so it lands before the numbered ids of the same stem and the order
    stays **total and deterministic** whatever is in the directory — a listing must never
    depend on what happens to be sitting beside it.

    >>> sorted(["an_9", "an_10", "an_100", "an_795"], key=numeric_id_key)
    ['an_9', 'an_10', 'an_100', 'an_795']
    """
    match = _TRAILING_NUMBER.match(name)
    if match is None:
        return (name, -1, name)
    return (match.group(1), int(match.group(2)), name)


def sorted_by_id(names: Iterable[str]) -> list[str]:
    """Ids in numeric order — the one way ``discover_*`` is allowed to sort.

    A single named call rather than ``sorted(..., key=numeric_id_key)`` at eight sites, so
    the guard that keeps the ninth allocator honest is a scan for one symbol.
    """
    return sorted(names, key=numeric_id_key)
