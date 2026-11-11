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

Fit-free on purpose. A state is read without its fit (a snapshot, the CLI, the manifest writer), so
everything here works on the two maps alone; what needs the tree lives with the tree.
"""

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
