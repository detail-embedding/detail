"""Keypoint skeletons (§26 "Skeletons"): names are labels, structure is declared.

A keypoint's name is arbitrary — nothing in DETAIL derives meaning from it at run time, and
nothing guesses a role from it. The only structure read is declared beside the names, and all
of it is the user's: the **edges** that join points (drawn, and read by the skeleton decode) and
the left/right **symmetry pairs** among them. This is SLEAP's skeleton (nodes, edges,
symmetries); DeepLabCut stores less. There is no body axis and no side on a skeleton: the
heading an embedding aligns to is a *heading definition* chosen per embedding run (§14), and
which image side "left" falls on is learned from the example clicks, per pair. The **midline**
is simply the points in no pair.

The position of a name in ``names`` is the column order of every stored array. A skeleton is
copied into every example set and every run that uses it, and a run counts its skeleton by the
**digest** of the copy it holds, never by a template's name.

The optional **layout** is where the skeleton editor drew each point, as fractions of its
canvas. It is kept so the editor reopens a skeleton as it was drawn and nothing else reads it,
which is why it is outside the digest: two skeletons that differ only in where they were drawn
mean the same thing to a run.

Pure data, no backend: imported by models, storage and every surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

Layout = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Skeleton:
    """The keypoints in column order, the edges between them, and the left/right pairs."""

    names: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()  # (left name, right name)
    # Where the editor drew each point, per column, as (x, y) fractions of its canvas; None
    # for a skeleton that was never drawn. Outside equality and the digest (see the module
    # docstring): it is how the skeleton looks in the editor, not what it means.
    layout: Layout | None = field(default=None, compare=False)
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))
        object.__setattr__(self, "edges", tuple(tuple(e) for e in self.edges))
        object.__setattr__(self, "pairs", tuple(tuple(p) for p in self.pairs))
        if self.layout is not None:
            object.__setattr__(
                self,
                "layout",
                tuple((float(x), float(y)) for x, y in self.layout),
            )
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(self.names)})

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Column of ``name``; raises ``KeyError`` naming the skeleton's points."""
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                f"no keypoint {name!r} in this skeleton; it has {list(self.names)}"
            ) from None

    def pair_indices(self) -> list[tuple[int, int]]:
        """``(left column, right column)`` for every declared pair, in declaration order."""
        return [(self.index(left), self.index(right)) for left, right in self.pairs]

    def midline_names(self) -> tuple[str, ...]:
        """The points in no pair, in column order — the midline the decode reads."""
        paired = {name for pair in self.pairs for name in pair}
        return tuple(n for n in self.names if n not in paired)

    def validate(self) -> None:
        """Raise ``ValueError`` naming the first inconsistency (Tenet 17)."""
        if not self.names:
            raise ValueError("a skeleton needs at least one keypoint")
        seen: set[str] = set()
        for name in self.names:
            if not name or not name.strip():
                raise ValueError("keypoint names must be non-empty")
            if name in seen:
                raise ValueError(f"keypoint {name!r} appears twice")
            seen.add(name)
        in_pair: set[str] = set()
        for left, right in self.pairs:
            for end in (left, right):
                if end not in self._index:
                    raise ValueError(
                        f"pair ({left!r}, {right!r}) names unknown keypoint {end!r}"
                    )
            if left == right:
                raise ValueError(
                    f"pair ({left!r}, {right!r}) pairs a keypoint with itself"
                )
            for end in (left, right):
                if end in in_pair:
                    raise ValueError(f"keypoint {end!r} is in two pairs")
                in_pair.add(end)
        seen_edges: set[frozenset[str]] = set()
        for a, b in self.edges:
            for end in (a, b):
                if end not in self._index:
                    raise ValueError(
                        f"edge ({a!r}, {b!r}) names unknown keypoint {end!r}"
                    )
            if a == b:
                raise ValueError(f"edge ({a!r}, {b!r}) joins a keypoint to itself")
            key = frozenset((a, b))
            if key in seen_edges:
                raise ValueError(f"edge ({a!r}, {b!r}) is listed twice")
            seen_edges.add(key)
        if self.layout is not None:
            if len(self.layout) != len(self.names):
                raise ValueError(
                    f"the layout places {len(self.layout)} point(s) but the skeleton has "
                    f"{len(self.names)}"
                )
            for name, (x, y) in zip(self.names, self.layout):
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError(
                        f"the layout puts {name!r} at ({x}, {y}), outside the canvas (0..1)"
                    )

    def digest(self) -> str:
        """A content hash of names, edges and pairs — a run's skeleton identity (§26). The
        layout is not in it: where a skeleton was drawn changes nothing a run means."""
        structure = {
            "names": list(self.names),
            "edges": [list(e) for e in self.edges],
            "pairs": [list(p) for p in self.pairs],
        }
        blob = json.dumps(structure, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        out = {
            "names": list(self.names),
            "edges": [list(e) for e in self.edges],
            "pairs": [list(p) for p in self.pairs],
        }
        if self.layout is not None:
            out["layout"] = [[round(x, 4), round(y, 4)] for x, y in self.layout]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        """A skeleton from its mapping (``to_dict``'s shape, or a hand-written file's).

        Each part is checked for shape before it is read: ``names: nose`` is one word, not
        the points ``n, o, s, e``, and ``pairs: [ab]`` is not the pair ``(a, b)`` (#31).

        Raises:
            ValueError: ``names`` is not a list of names, or ``edges`` / ``pairs`` /
                ``layout`` is not a list of two-item lists.
        """
        layout = d.get("layout")
        return cls(
            names=tuple(_names(d.get("names"))),
            edges=tuple(_twos(d.get("edges"), "edges", "two point names")),
            pairs=tuple(_twos(d.get("pairs"), "pairs", "two point names")),
            layout=(
                tuple(
                    (float(x), float(y))
                    for x, y in _twos(layout, "layout", "an x and a y", numbers=True)
                )
                if layout is not None
                else None
            ),
        )

    def with_layout(self, layout: Layout | None) -> "Skeleton":
        """The same skeleton drawn differently."""
        return Skeleton(
            names=self.names, edges=self.edges, pairs=self.pairs, layout=layout
        )


def _names(value) -> list[str]:
    """``names`` as a list of point names; a lone word or a mapping is refused."""
    if not isinstance(value, (list, tuple)):
        found = "nothing" if value is None else repr(value)
        raise ValueError(
            f"a skeleton's 'names' must be a list of point names (e.g. [nose, tail]), "
            f"not {found}"
        )
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(
                f"a skeleton's 'names' must be a list of point names; {item!r} is not one"
            )
    return [str(n) for n in value]


def _twos(value, what: str, items: str, *, numbers: bool = False) -> list[tuple]:
    """``value`` as a list of two-item tuples (an edge, a pair, a layout position)."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"a skeleton's {what!r} must be a list of [{items}] lists, not {value!r}"
        )
    kinds = (int, float) if numbers else (str, int)
    out = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(isinstance(v, bool) or not isinstance(v, kinds) for v in item)
        ):
            raise ValueError(
                f"a skeleton's {what!r} must be a list of [{items}] lists; {item!r} is not one"
            )
        out.append(tuple(item) if numbers else tuple(str(v) for v in item))
    return out


def flip_index(skeleton: Skeleton) -> list[int]:
    """Ultralytics' ``flip_idx``: the column each point becomes under a left-right flip — its
    pair partner, or itself for a midline point."""
    out = list(range(len(skeleton.names)))
    for left, right in skeleton.pair_indices():
        out[left], out[right] = right, left
    return out


def pairs_from_flip_index(flip_idx: list[int]) -> set[frozenset[int]]:
    """The left/right column pairs a ``flip_idx`` swaps; a point mapped to itself is in none.

    Raises:
        ValueError: ``flip_idx`` is not a pairing — an index out of range, or a point that
            does not map back to the one that maps to it.
    """
    k = len(flip_idx)
    pairs: set[frozenset[int]] = set()
    for i, j in enumerate(flip_idx):
        if not 0 <= j < k:
            raise ValueError(
                f"flip_idx {list(flip_idx)} maps point {i} to {j}, outside 0..{k - 1}"
            )
        if flip_idx[j] != i:
            raise ValueError(
                f"flip_idx {list(flip_idx)} maps point {i} to {j} but {j} to {flip_idx[j]}; "
                "a flip must swap points in pairs"
            )
        if i != j:
            pairs.add(frozenset((i, j)))
    return pairs


def check_against_model(skeleton: Skeleton, k: int, flip_idx: list[int] | None) -> None:
    """Refuse a skeleton that does not describe a pose model with ``k`` points and this
    ``flip_idx`` (§12 *Keypoint models in the catalog*).

    The check cannot see a pair named the wrong way round: (left, right) and (right, left)
    give the same ``flip_idx``.

    Raises:
        ValueError: An invalid skeleton; a different number of points; pairs when the model
            has no ``flip_idx`` to check them against; or pairs that differ from the ones the
            model's ``flip_idx`` swaps.
    """
    skeleton.validate()
    if len(skeleton) != k:
        raise ValueError(
            f"the model places {k} points but the skeleton has {len(skeleton)} "
            f"({', '.join(skeleton.names)}); pick a skeleton with {k} points"
        )
    if flip_idx is None:
        if skeleton.pairs:
            raise ValueError(
                "the model has no flip_idx, so its left/right pairs cannot be checked against "
                f"the skeleton's {len(skeleton.pairs)} pair(s); use a skeleton without pairs, "
                "or a model trained with flip_idx"
            )
        return
    if len(flip_idx) != k:
        raise ValueError(
            f"the model's flip_idx has {len(flip_idx)} entries for {k} points; the "
            "checkpoint is inconsistent"
        )
    model_pairs = pairs_from_flip_index(flip_idx)
    skeleton_pairs = pairs_from_flip_index(flip_index(skeleton))
    if model_pairs != skeleton_pairs:

        def named(pairs: set[frozenset[int]]) -> str:
            shown = sorted(tuple(sorted(p)) for p in pairs)
            return (
                ", ".join(
                    f"({skeleton.names[a]}, {skeleton.names[b]})" for a, b in shown
                )
                or "none"
            )

        raise ValueError(
            f"the model swaps {named(model_pairs)} under a left-right flip, but the "
            f"skeleton pairs {named(skeleton_pairs)}; the skeleton's left/right pairs must "
            "be the points the model swaps"
        )


def default_layout(skeleton: Skeleton) -> Layout:
    """Where to draw a skeleton that was never drawn: the structure, not the animal.

    Midline points down the middle in column order (the head, first in the list, at the top),
    each pair's left member to the left of its right member at the left member's row. This
    is the schematic the first editor drew, kept as the starting layout a skeleton without
    one shows once; the editor saves whatever the user makes of it.
    """
    names = skeleton.names
    if not names:
        return ()
    paired: dict[str, str] = {}
    for left, right in skeleton.pairs:
        paired[left] = "left"
        paired[right] = "right"
    row_of = {name: i for i, name in enumerate(names)}
    for left, right in skeleton.pairs:
        if left in row_of and right in row_of:
            row_of[right] = row_of[left]
    rows = sorted(set(row_of.values()))
    y_of = {r: 0.1 + 0.8 * j / max(len(rows) - 1, 1) for j, r in enumerate(rows)}
    x_of = {"midline": 0.5, "left": 0.3, "right": 0.7}
    return tuple(
        (x_of[paired.get(name, "midline")], round(y_of[row_of[name]], 4))
        for name in names
    )


# --------------------------------------------------------------------------- #
# Presets — a starting skeleton for the labeling screen
# --------------------------------------------------------------------------- #
#: ``id -> (label, skeleton)``: the two quadruped skeletons the research measured (overhead and
#: side view) and Quick pose's simpler overhead
#: and three-point midline presets. Named for the body plan and view, not a
#: species (renamed 2026-10-09 from ``rodent_*`` / ``elongate``, which no longer resolve).
#: Each is ordered head to tail, the order a person
#: clicks in, and ships a layout so the editor opens on a picture of the animal rather than
#: a column. A preset is copied into an example set on use (§26 "Skeleton templates").
PRESETS: dict[str, tuple[str, Skeleton]] = {
    "quadruped_overhead_12": (
        "Quadruped overhead view (12 points)",
        Skeleton(
            names=(
                "nose",
                "left_ear",
                "right_ear",
                "neck_base",
                "left_shoulder",
                "right_shoulder",
                "mid_back",
                "left_hip",
                "right_hip",
                "tail_base",
                "tail_mid",
                "tail_tip",
            ),
            edges=(
                ("nose", "neck_base"),
                ("left_ear", "neck_base"),
                ("right_ear", "neck_base"),
                ("neck_base", "left_shoulder"),
                ("neck_base", "right_shoulder"),
                ("neck_base", "mid_back"),
                ("mid_back", "tail_base"),
                ("tail_base", "left_hip"),
                ("tail_base", "right_hip"),
                ("tail_base", "tail_mid"),
                ("tail_mid", "tail_tip"),
            ),
            pairs=(
                ("left_ear", "right_ear"),
                ("left_shoulder", "right_shoulder"),
                ("left_hip", "right_hip"),
            ),
            layout=(
                (0.5, 0.08),
                (0.41, 0.17),
                (0.59, 0.17),
                (0.5, 0.27),
                (0.38, 0.36),
                (0.62, 0.36),
                (0.5, 0.48),
                (0.4, 0.6),
                (0.6, 0.6),
                (0.5, 0.68),
                (0.5, 0.81),
                (0.5, 0.94),
            ),
        ),
    ),
    "quadruped_side_5": (
        "Quadruped side view (5 points)",
        Skeleton(
            names=("nose", "left_ear", "right_ear", "mid_back", "tail_base"),
            edges=(
                ("nose", "left_ear"),
                ("nose", "right_ear"),
                ("nose", "mid_back"),
                ("mid_back", "tail_base"),
            ),
            pairs=(("left_ear", "right_ear"),),
            layout=(
                (0.12, 0.5),
                (0.3, 0.3),
                (0.36, 0.38),
                (0.56, 0.36),
                (0.88, 0.5),
            ),
        ),
    ),
    "quadruped_overhead_8": (
        "Quadruped overhead view (8 points)",
        Skeleton(
            names=(
                "nose",
                "left_ear",
                "right_ear",
                "neck",
                "spine_mid",
                "tail_base",
                "tail_mid",
                "tail_tip",
            ),
            edges=(
                ("nose", "neck"),
                ("left_ear", "neck"),
                ("right_ear", "neck"),
                ("neck", "spine_mid"),
                ("spine_mid", "tail_base"),
                ("tail_base", "tail_mid"),
                ("tail_mid", "tail_tip"),
            ),
            pairs=(("left_ear", "right_ear"),),
            layout=(
                (0.5, 0.08),
                (0.41, 0.18),
                (0.59, 0.18),
                (0.5, 0.28),
                (0.5, 0.47),
                (0.5, 0.65),
                (0.5, 0.8),
                (0.5, 0.94),
            ),
        ),
    ),
    "midline_3": (
        "Midline (3 points)",
        Skeleton(
            names=("head", "midpoint", "tail"),
            edges=(("head", "midpoint"), ("midpoint", "tail")),
            layout=((0.15, 0.5), (0.5, 0.5), (0.85, 0.5)),
        ),
    ),
}
