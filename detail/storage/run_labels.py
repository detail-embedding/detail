"""The labels of a clustering run — stored things that clusters point at (§22).

A label used to be a string on each node it was given to, so two clusters named "rearing" were two
strings that happened to match. Nothing said they were one label: the box that took the second
offered the first as a suggestion and merged them silently, a rename reached one cluster at a time,
and the surfaces disagreed about whether "rearing" here and "rearing" there were the same thing.

Now a label has an id. ``labels`` maps the id to its name and ``node_labels`` maps a node of the
merge tree to the id it carries. A cluster joins a label by pointing at it; renaming the label
renames every cluster under it; merging moves them all. Nesting still comes from the tree — a
label's parent is the label on the nearest labelled branch above its clusters — and
``api/authored_run.check_labels`` holds the one rule that makes that well defined: every cluster of
a label sits under the same parent label.

A label written while a difference screen was ticked carries where it came from
(``label_origins``, node -> :class:`LabelOrigin`): the saved screen, the cluster it kept and the
column it contrasted, so a comparison of that label over the same column can say the cluster was
chosen by the contrast it is now scored on (§22, §24).

Fit-free on purpose. A state is read without its fit (a snapshot, the CLI, the manifest writer), so
everything here works on the maps alone; what needs the tree -- the relation of a node to a kept
cluster -- takes the tree's ``ancestors_of`` as an argument.
"""

import dataclasses
import re

#: A label's id: ``L`` and a number, never reused within a run. Valid as a ``source_label`` and as a
#: frame-view array name (``storage/frame_view.py``'s ``^[A-Za-z0-9_-]+$``), and never a bare
#: number, so it cannot be read as a node id from a run written before labels had ids.
LABEL_ID = re.compile(r"^L([0-9]+)$")


def mint(labels: dict[str, str]) -> str:
    """The next unused id — one past the largest, so an id deleted once is never handed out again
    while a larger one survives."""
    numbers = [int(m.group(1)) for key in labels if (m := LABEL_ID.match(key))]
    return f"L{max(numbers, default=0) + 1}"


def order(labels: dict[str, str]) -> list[str]:
    """Label ids in the order they were made (``L2`` before ``L10``)."""
    return sorted(labels, key=lambda key: (_number(key), key))


def _number(key: str) -> int:
    match = LABEL_ID.match(key)
    return int(match.group(1)) if match else 0


def find(labels: dict[str, str], name: str) -> str | None:
    """The id of the label called ``name``, ignoring case and surrounding space, or ``None``.

    Case-insensitive because names are unique that way: "Rearing" typed beside "rearing" is the same
    behavior misspelt, and two labels told apart only by a capital is a vocabulary nobody can read.
    """
    wanted = name.strip().casefold()
    for key, existing in labels.items():
        if existing.strip().casefold() == wanted:
            return key
    return None


def node_names(labels: dict[str, str], node_labels: dict[str, str]) -> dict[str, str]:
    """``{node: name}`` — the per-node view every reader that only needs a name keeps using."""
    return {
        node: labels[label]
        for node, label in node_labels.items()
        if label in labels and labels[label]
    }


def from_node_names(names: dict) -> tuple[dict[str, str], dict[str, str]]:
    """``(labels, node_labels)`` for a state written before labels had ids.

    One label per distinct name, in the order its first node appears (node id order), so the same
    file always mints the same ids. Fit-free, so it cannot tell a name used under two different
    parents from one used under one; the next edit, which has the tree, splits those
    (``authored_run.reresolve_labels``). A snapshot is never edited, and it does not need to be:
    its paths come from the artifact it froze.
    """
    labels: dict[str, str] = {}
    node_labels: dict[str, str] = {}
    for node in sorted(names, key=lambda key: int(key) if str(key).isdigit() else 0):
        name = str(names[node] or "").strip()
        if not name:
            continue
        label = find(labels, name)
        if label is None:
            label = mint(labels)
            labels[label] = name
        node_labels[str(node)] = label
    return labels, node_labels


def read(data: dict) -> tuple[dict[str, str], dict[str, str]]:
    """``(labels, node_labels)`` from a stored state, whichever generation wrote it."""
    if "labels" in data or "node_labels" in data:
        labels = {str(k): str(v) for k, v in (data.get("labels") or {}).items()}
        node_labels = {
            str(k): str(v)
            for k, v in (data.get("node_labels") or {}).items()
            if str(v) in labels
        }
        return labels, node_labels
    return from_node_names(data.get("node_names") or {})


#: How a labelled node stands to the cluster a difference screen kept (§22): it *is* the kept
#: cluster, sits *under* one, or is *above* one.
KEPT = "kept"
UNDER = "under"
ABOVE = "above"
RELATIONS = (KEPT, UNDER, ABOVE)


@dataclasses.dataclass(frozen=True)
class LabelOrigin:
    """Where a node's label came from, when a difference screen pointed at the cluster (§22).

    A label is the user's, written from the clips; what the screen did was choose the cluster
    to watch. That choice has to travel with the label, because a comparison of the label across
    the same column later would be scoring a cluster the same contrast selected -- a number that
    looks like an independent finding and is not (CLAUDE.md, *labels score a method; they never
    build one*). Nothing fails without this record; the comparison just cannot say so.

    ``comparison`` is the saved screen's id (``cmp_NN``), ``kept`` the cluster it kept that the
    labelled node stands to as ``relation`` (:data:`RELATIONS`), ``column`` the metadata column
    the screen contrasted (what a later comparison is checked against), ``screen`` how the screen
    read when the label was written ("drug vs vehicle · paired") and ``text`` the screen's
    sentence for the kept cluster. The last two are copies, so a state reads without the
    comparison it names -- which may since have been deleted.

    **The strictness the screen was read at travels with the label.** A screen is read at a
    family-wise level (§22): its own 5% floor, or looser when the user slid the strictness to see
    more. ``alpha`` is that level (``""`` for an origin written before strictness existed, which
    is the floor), and ``family_p`` / ``own_p`` / ``q`` are the kept cluster's numbers when the
    label was written -- so a label chosen at p ≤ 0.3 says so for as long as it exists, and a
    reader can tell a finding from a lead. Every field is a string, as the state stores it.
    """

    comparison: str
    kept: str
    relation: str
    column: str
    screen: str = ""
    text: str = ""
    alpha: str = ""
    family_p: str = ""
    own_p: str = ""
    q: str = ""

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "LabelOrigin | None":
        """The origin a stored dict holds, or ``None`` for one missing what it needs."""
        if not isinstance(data, dict):
            return None
        fields = {k: str(data.get(k) or "") for k in ("comparison", "kept", "relation")}
        if not all(fields.values()) or fields["relation"] not in RELATIONS:
            return None
        return cls(
            **fields,
            **{
                k: str(data.get(k) or "")
                for k in ("column", "screen", "text", "alpha", "family_p", "own_p", "q")
            },
        )


def read_origins(data: dict) -> dict[str, dict[str, str]]:
    """``{node: origin}`` from a stored state -- only the origins that read back whole."""
    out: dict[str, dict[str, str]] = {}
    for node, stored in (data.get("label_origins") or {}).items():
        origin = LabelOrigin.from_dict(stored)
        if origin is not None:
            out[str(node)] = origin.to_dict()
    return out


def screen_relation(node: int, kept, ancestors_of) -> str | None:
    """How ``node`` stands to the clusters a screen ``kept``: :data:`KEPT`, :data:`UNDER`,
    :data:`ABOVE`, or ``None`` when the screen did not point at it. ``ancestors_of(node)`` is
    the tree's root-first path above a node (``NodeView.ancestors``)."""
    node = int(node)
    kept_nodes = {int(k) for k in kept}
    if node in kept_nodes:
        return KEPT
    if kept_nodes & set(ancestors_of(node)):
        return UNDER
    if any(node in ancestors_of(k) for k in kept_nodes):
        return ABOVE
    return None


def _number_text(value) -> str:
    """A p, q or level as the state stores it: ``"0.0039"``, ``""`` for none."""
    return "" if value is None else f"{float(value):.6g}"


def label_origin(
    node: int,
    *,
    comparison: str,
    kept_texts: dict,
    column: str,
    screen: str,
    ancestors_of,
    alpha: float | None = None,
    numbers: dict | None = None,
) -> LabelOrigin | None:
    """The origin a label on ``node`` carries from the screen ``comparison``, or ``None`` when the
    screen did not point at the node. ``kept_texts`` is ``{kept node: its sentence}`` at the
    strictness ``alpha`` the screen was read at (``None``: its own floor); the one the node
    stands to is recorded -- the node itself, the kept ancestor, or the kept node beneath it (the
    nearest, for a node above several) -- with its numbers from ``numbers``,
    ``{kept node: {"family_p", "own_p", "q"}}``, where given."""
    relation = screen_relation(node, kept_texts, ancestors_of)
    if relation is None:
        return None
    node = int(node)
    kept_nodes = {int(k): k for k in kept_texts}
    if relation == KEPT:
        chosen = node
    elif relation == UNDER:
        chosen = next(a for a in reversed(ancestors_of(node)) if a in kept_nodes)
    else:
        below = [k for k in kept_nodes if node in ancestors_of(k)]
        chosen = min(below, key=lambda k: len(ancestors_of(k)))
    found = (numbers or {}).get(kept_nodes[chosen]) or {}
    return LabelOrigin(
        comparison=str(comparison),
        kept=str(chosen),
        relation=relation,
        column=str(column),
        screen=str(screen),
        text=str(kept_texts[kept_nodes[chosen]] or ""),
        alpha=_number_text(alpha),
        family_p=_number_text(found.get("family_p")),
        own_p=_number_text(found.get("own_p")),
        q=_number_text(found.get("q")),
    )


def screened(data: dict, columns) -> dict[str, list[dict]]:
    """``{label name: [origin + node, ...]}`` for the labels of a stored state whose clusters a
    difference screen over one of ``columns`` chose -- what a labels comparison over those
    columns records as not independent of its own contrast (§24)."""
    wanted = {str(c) for c in columns}
    labels, node_labels = read(data)
    names = {str(k): str(v) for k, v in (data.get("resolved_names") or {}).items()}
    out: dict[str, list[dict]] = {}
    for node, origin in read_origins(data).items():
        if origin["column"] not in wanted or node not in node_labels:
            continue
        name = names.get(node) or labels.get(node_labels[node], "")
        if name:
            out.setdefault(name, []).append({"node": node, **origin})
    return out
