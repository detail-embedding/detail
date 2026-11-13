"""``AuthoredRun`` — the lifecycle every §21 authored analysis shares.

An authored run's compute finishes and then *stays open* while a person works on it: Query's
cutoffs, and temporal clustering's (§22) granularity and names — pose clustering's too, until it
was removed. The compute half differs completely between them; this half does not.

What lives here is the shape §21 calls **an operation** — refuse if finalized, mutate the state,
write it, log it, reproject the artifact — plus the status/finalize/reopen lifecycle around it.
It is one class rather than one copy per module because the whole point of the sequence is that
no step can be forgotten by one of them, and by the time a third module wanted it there were
already two copies drifting apart.

A subclass supplies four things — where its state lives, how to write it, how to log, and how to
rebuild its artifact — and inherits the rest.
"""

import dataclasses
import threading
from pathlib import Path
from typing import Self

from detail.analysis import provenance
from detail.api.experiment import AnalysisRun, Experiment
from detail.core.dataclasses import reconciled_status
from detail.storage import analysis_runs as analysis_store
from detail.storage import behavior_output
from detail.storage import dirscan
from detail.storage import frame_view
from detail.storage import snapshots as snapshots_store


@dataclasses.dataclass(frozen=True)
class SnapshotInterpretation:
    """What a snapshot *was*, in the module's own terms.

    The surfaces that show a snapshot — the run page's table, its banner, ``analysis snapshots`` —
    used to read `granularity` and `names` straight out of the record, which is clustering's shape
    and nobody else's. A calibrated Query run with two named behaviors rendered as
    ``Clusters — · Named 0 · unnamed``: not missing information but *wrong* information, on a run
    that was fully interpreted. Nothing failed, because a blank cell is a plausible cell.

    So the module answers instead of the surface assuming. ``columns`` and ``cells`` are the same
    length and go in the table; ``headline`` is the one-line form a banner shows. One method builds
    all three from one read of the state, so a module cannot describe itself two ways.
    """

    columns: tuple[str, ...]
    cells: tuple[str, ...]
    headline: str


def label_interpretation(record: dict) -> SnapshotInterpretation:
    """A clustering snapshot summarized: how many labels it carried, and what they were (§22).

    Shared because the pose and temporal versions are identical. A snapshot is frozen and has no
    fit behind it, so it answers from what it stored, newest generation first: the resolution,
    then the per-granularity map, then the flat map that predates both. A snapshot taken before
    §22 dropped the granularity also says which setting it was taken at, because that is part of
    what it froze.
    """
    state = record.get("state") or {}
    names = state.get("resolved_names")
    if names is None:
        names = (state.get("names_by_granularity") or {}).get(
            str(state.get("granularity"))
        )
    if names is None:
        names = state.get("names") or {}
    names = {str(k): str(v) for k, v in names.items() if v}
    vocabulary = snapshots_store.vocabulary(names)
    labels = len(set(names.values()))
    headline = f"{labels} label{'s' if labels != 1 else ''} · {vocabulary}"
    if state.get("granularity") is not None:
        headline += f" · taken at {state['granularity']} clusters"
    return SnapshotInterpretation(
        columns=("Labels", "Vocabulary"),
        cells=(str(labels), vocabulary),
        headline=headline,
    )


def cluster_nodes(tree, labels, granularity: int):
    """Each cluster id at ``granularity`` mapped to the tree node it *is*.

    The join between the two id spaces, and it is one scatter: every item carries the node its leaf
    sits at when the tree is cut, and a cluster is exactly the items sharing one such node, so
    reading the node off any member is well defined. ``parent_map`` is the same trick one level
    down, and this reuses it rather than re-deriving it.

    Written for both clustering modules (pose clustering is since removed; its saved runs still
    read through this) because the join is a property of the tree, not of what the method
    clustered — ``labels`` and ``leaf_of_row`` share an index space for all of them (sampled rows
    for the pose methods, clustered bouts for ``dtw_finch``).
    """
    import numpy as np

    from detail.analysis import segmentation

    per_item = tree.leaf_nodes_at(int(granularity))[tree.leaf_of_row]
    return segmentation.parent_map(np.asarray(labels), per_item)


def name_path(tree, node: int, node_names: dict) -> list[str]:
    """The names on ``node`` and every named branch above it, **outermost first**.

    Nested naming, which is the whole point of keeping a tree rather than a list: "not moving" on a
    high branch and "resting" on one inside it are both true of the same bout, and the second does
    not replace the first. ``label`` stays the innermost — so every reader that knows only about
    ``label`` is unaffected — and this is the ancestry it sits in.

    A name is never *overwritten* by naming a branch above it. That is the property that makes the
    hierarchy usable: you can name coarsely first and refine, or refine first and generalise, and
    end up at the same place.
    """
    out: list[str] = []
    current = int(node)
    parent = tree.parent
    while current >= 0:
        name = (node_names or {}).get(str(current)) or ""
        if name:
            out.append(name)
        current = int(parent[current])
    out.reverse()
    return out


#: ``source_label`` of an instance under no labelled node (§22). One bucket: without a
#: granularity there is no honest cluster id to give unlabelled time, and leaf-sized instances
#: would bury the labelled ones. ``label`` stays empty, which is what §23's ``unnamed_time`` reads.
UNLABELLED = "unlabelled"


def source_label_of(group: int, state) -> str:
    """The artifact's ``source_label`` for an ``instances_for`` group: the **label id** (§22).

    ``labelled_instances`` hands the method one group per label, numbered in label order, so the
    group is an index into ``run_labels.order(state.labels)``. Keyed by label rather than by node so
    the clusters of one label are one behavior in the output — and so a rename, which changes no
    id, moves no instance.
    """
    from detail.storage import run_labels

    if int(group) < 0:
        return UNLABELLED
    return run_labels.order(state.labels)[int(group)]


def ancestors(tree, node: int) -> list[int]:
    """The path from the root down to ``node``'s parent, root first (``NodeView.ancestors``
    without the cache) -- what a label's origin is placed with (``run_labels.screen_relation``).
    """
    parent = tree.parent
    out = []
    current = int(parent[int(node)])
    while current >= 0:
        out.append(current)
        current = int(parent[current])
    return list(reversed(out))


def parent_label(tree, node: int, node_labels: dict) -> str | None:
    """The label on the nearest labelled branch strictly **above** ``node``, or ``None``."""
    current = int(tree.parent[int(node)])
    while current >= 0:
        label = node_labels.get(str(current))
        if label:
            return label
        current = int(tree.parent[current])
    return None


def check_labels(labels: dict, node_labels: dict, tree) -> None:
    """Refuse a set of labels the output could not report unambiguously (§22).

    Three rules, each named in the message with the clusters that break it:

    - **A name is unique** (ignoring case) and holds no path separator. Two labels told apart by a
      capital, or a name that reads as two levels in an exported ``label_path``, cannot be read.
    - **All clusters of a label sit under one parent label.** Nesting comes from the tree, so a
      label's path is the labels above its clusters; clusters under two different parents would
      give one label two paths, and its precision, its export and its row in every table would
      each have to pick one.
    - **A label is not inside itself.** A cluster under a branch already carrying its label adds
      nothing, and would write ``rearing/rearing``.

    Raises:
        ValueError: with the rule broken and the clusters that break it.
    """
    from detail.storage.behavior_output import PATH_SEPARATOR

    seen: dict[str, str] = {}
    for label, name in labels.items():
        if not name.strip():
            raise ValueError(f"label {label} has no name")
        if PATH_SEPARATOR in name:
            raise ValueError(
                f"a name may not contain {PATH_SEPARATOR!r}, which separates the levels of an "
                f"exported label_path: {name!r}"
            )
        key = name.strip().casefold()
        if key == "other":
            raise ValueError(
                "'other' is the name the labels panel gives a label's own remainder (its "
                "clusters outside every sub-label), so a label called that would be two rows "
                "with one name — choose another name"
            )
        if key in seen:
            raise ValueError(
                f"there is already a label called {labels[seen[key]]!r} — add the cluster to it "
                "instead of making a second one"
            )
        seen[key] = label
    parents: dict[str, dict] = {}
    for node, label in node_labels.items():
        if label not in labels:
            raise ValueError(
                f"cluster #{node} points at a label that does not exist ({label})"
            )
        above = parent_label(tree, int(node), node_labels)
        if above == label:
            raise ValueError(
                f"cluster #{node} is already inside a branch labelled {labels[label]!r}, so "
                "labelling it again adds nothing"
            )
        parents.setdefault(label, {}).setdefault(above, []).append(int(node))
    for label, by_parent in parents.items():
        if len(by_parent) > 1:
            where = "; ".join(
                f"under {labels[p]!r}" if p else "at the top level"
                for p in sorted(by_parent, key=lambda p: (p is not None, p or ""))
            )
            clusters = ", ".join(
                f"#{n}" for nodes in by_parent.values() for n in sorted(nodes)
            )
            raise ValueError(
                f"the clusters of {labels[label]!r} ({clusters}) would sit {where}. A label's "
                "clusters must share one parent label — give the others a label of their own, "
                "or label the branch that holds them all"
            )


def membership_changes(before: dict, after: dict, tree) -> set[str]:
    """The labels whose set of clusters an edit changed, **and every label above them**.

    Above them too, because a parent label's audit covers its sub-labels' instances: moving a
    cluster out of "climb corner" changes what "climb and rear" is made of as well. A rename moves
    no cluster and changes nothing here. A label that no longer exists is left out — there is no
    figure of its to reset — and so is one the edit created: it has no audit to set aside, and
    counting it said "edited after the audit" for every label a run ever made.
    """

    def members(mapping: dict) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for node, label in mapping.items():
            out.setdefault(label, set()).add(node)
        return out

    old, new = members(before), members(after)
    # A label that did not exist before has no audit for an edit to set aside.
    changed = {label for label in set(old) & set(new) if old[label] != new[label]}
    # Walk up from every cluster whose label changed, removed ones included: a cluster that lost
    # its own label now reports under the label above it, whose clusters did not change at all.
    moved = {
        node for node in set(before) | set(after) if before.get(node) != after.get(node)
    }
    for node in moved:
        above = parent_label(tree, int(node), after)
        while above is not None:
            if above in old:
                changed.add(above)
            anchor = next(iter(new.get(above, ())), None)
            above = None if anchor is None else parent_label(tree, int(anchor), after)
    return {label for label in changed if label in new}


def _node_paths(state) -> dict[str, str]:
    """``{node: label path}`` for every labelled cluster, from the state's resolved paths."""
    return {
        node: state.resolved_paths.get(label, "")
        for node, label in state.node_labels.items()
        if state.resolved_paths.get(label)
    }


def path_moves(
    before: dict[str, str], after: dict[str, str], nodes: set[str] | None = None
) -> dict[str, str]:
    """``{old path: new path}`` for the label paths an edit renamed or merged away.

    Read off the clusters: a cluster whose path changed says where its old path went. A path whose
    clusters went to two different places (a split) is left out — there is no one place to carry
    its answers to. ``nodes`` limits which clusters speak (``None``: all): only those that kept
    their label, so a cluster *moved* to another label does not take the answers about its old
    one with it (``LabelOperations._label_edit``).
    """
    seen: dict[str, set[str]] = {}
    for node, old in before.items():
        if nodes is not None and node not in nodes:
            continue
        new = after.get(node)
        if new and new != old:
            seen.setdefault(old, set()).add(new)
    return {old: next(iter(new)) for old, new in seen.items() if len(new) == 1}


def carried_path(path: str, moved: dict[str, str]) -> str | None:
    """``path`` with its longest moved prefix replaced, or ``None`` when nothing above it moved."""
    from detail.storage.behavior_output import PATH_SEPARATOR

    best = None
    for old in moved:
        if (path == old or path.startswith(old + PATH_SEPARATOR)) and (
            best is None or len(old) > len(best)
        ):
            best = old
    return None if best is None else moved[best] + path[len(best) :]


def label_resolution(tree, labels: dict, node_labels: dict) -> tuple[dict, dict]:
    """``({label: name}, {label: "outer/inner"})`` for every label that has a cluster (§22).

    Keyed by label id because that is what ``instances.csv`` stores as ``source_label``. A label's
    path is the same from any of its clusters — ``check_labels`` holds that — so it is read off the
    first. The two maps are built from one pass so a name and its path cannot come to disagree.
    """
    from detail.storage import run_labels
    from detail.storage.behavior_output import PATH_SEPARATOR

    names_by_node = run_labels.node_names(labels, node_labels)
    first: dict[str, int] = {}
    for node, label in sorted(node_labels.items(), key=lambda kv: int(kv[0])):
        first.setdefault(label, int(node))
    names = {label: labels[label] for label in first}
    paths = {
        label: PATH_SEPARATOR.join(name_path(tree, node, names_by_node))
        for label, node in first.items()
    }
    return names, paths


def fold_legacy_names(state, tree, labels_at) -> int:
    """Move a pre-node state's names (``names_by_granularity``) onto the nodes they were given at,
    as labels. Returns how many names moved.

    Pure over ``state``: :func:`reresolve_labels` calls it on every edit, and
    ``NodeCacheMixin.node_labels`` calls it on a *copy* so a run written before labels lived on
    nodes shows its names on open, not only after its first edit. A name that already exists joins
    that label (``run_labels.find``), as it would typed today.
    """
    from detail.storage import run_labels

    moved = 0
    for setting, names in state.names_by_granularity.items():
        if setting in (None, "None") or not names:
            continue
        nodes = cluster_nodes(tree, labels_at(int(setting)), int(setting))
        for cluster_id, name in names.items():
            if name and int(cluster_id) < len(nodes):
                label = run_labels.find(state.labels, str(name))
                if label is None:
                    label = run_labels.mint(state.labels)
                    state.labels[label] = str(name)
                state.node_labels[str(int(nodes[int(cluster_id)]))] = label
                moved += 1
    state.names_by_granularity = {}
    return moved


def reresolve_labels(state, tree, labels_at) -> dict:
    """Recompute ``resolved_names`` / ``resolved_paths``, folding older states first.

    Called by every edit of both clustering modules, because the resolution is derived and a derived
    value with more than one writer is the thing that goes quietly stale. Returns what the fold
    changed for the edit's ``history.jsonl`` entry — ``folded`` (in words) and ``migrated_names``
    (how many per-granularity names moved onto nodes) — empty once a state is current.

    Two older shapes fold here, because both need the tree:

    - ``names_by_granularity`` (cluster id → name per setting) goes onto the nodes it was given at
      (``labels_at`` is the fit's own).
    - A state from before labels had ids reads as one label per distinct name
      (``run_labels.from_node_names``), which the parent rule can break: one name used under two
      different parents becomes "name" and "name (2)", and a cluster labelled the same as a branch
      above it loses the redundant label.
    """
    from detail.storage import run_labels

    notes: list[str] = []
    moved = (
        fold_legacy_names(state, tree, labels_at) if state.names_by_granularity else 0
    )
    for node, label in sorted(state.node_labels.items(), key=lambda kv: int(kv[0])):
        if parent_label(tree, int(node), state.node_labels) == label:
            del state.node_labels[node]
            notes.append(
                f"#{node} was inside a branch already labelled {state.labels[label]!r}"
            )
    by_parent: dict[str, dict] = {}
    for node, label in sorted(state.node_labels.items(), key=lambda kv: int(kv[0])):
        above = parent_label(tree, int(node), state.node_labels)
        by_parent.setdefault(label, {}).setdefault(above, []).append(node)
    for label, groups in by_parent.items():
        for extra, nodes in list(groups.items())[1:]:
            base, k = state.labels[label], 2
            while run_labels.find(state.labels, f"{base} ({k})") is not None:
                k += 1
            split = run_labels.mint(state.labels)
            state.labels[split] = f"{base} ({k})"
            for node in nodes:
                state.node_labels[node] = split
            notes.append(
                f"{base!r} sat under two parents; the clusters under "
                f"{state.labels.get(extra, 'the top level') if extra else 'the top level'} "
                f"became {state.labels[split]!r}"
            )
    state.resolved_names, state.resolved_paths = label_resolution(
        tree, state.labels, state.node_labels
    )
    info: dict = {}
    if notes:
        info["folded"] = notes
    if moved:
        info["migrated_names"] = moved
    return info


def label_groups(tree, state):
    """Each leaf's label as an index into ``run_labels.order(state.labels)``, ``-1`` for none.

    The partition ``instances_for`` turns into instances. One group per **label**, not per labelled
    node, so the clusters of one label are one behavior: two of them alternating frame to frame make
    one bout, not a run of short ones that inflate the count.
    """
    import numpy as np

    from detail.storage import run_labels

    index = {label: i for i, label in enumerate(run_labels.order(state.labels))}
    deepest = np.asarray(tree.leaf_groups(state.named_nodes()), dtype=np.int64)
    lut = np.full(int(tree.n_nodes), -1, dtype=np.int64)
    for node, label in state.node_labels.items():
        if label in index:
            lut[int(node)] = index[label]
    return np.where(deepest >= 0, lut[np.clip(deepest, 0, None)], -1)


def labelled_instances(fit, method, state) -> list:
    """The instances the current labels yield: each member under the label of its deepest
    labelled node, the clusters of one label as one group.

    The tree resolves the labels to a partition of its leaves (``label_groups``), the method turns
    that partition into instances, and runs that end up sharing a label are fused. The method is
    handed group numbers and never a name (``instances_for``).

    Clips a user took out of a label, or moved, override their members' groups before the method
    cuts runs (``api/clip_edits.py``). Only a temporal state carries them; the pose state, hidden,
    has none, so its method is never handed the override.
    """
    from detail.api import clip_edits

    tree = method.hierarchy_of(fit)
    groups = label_groups(tree, state)
    override = clip_edits.member_groups(method, fit, state, groups)
    if override is None:
        found = method.instances_for(fit, groups, dict(state.method_params))
    else:
        found = method.instances_for(
            fit, groups, dict(state.method_params), member_groups=override
        )
    return fuse_runs(fit, method, found)


def fuse_runs(fit, method, instances: list) -> list:
    """Join consecutive instances of one group within one unit, where the method allows it.

    A sub-floor flicker between two runs of one group was dropped by the method, and the fused
    span covers it, which is right once both sides report the same node. Only where the method
    declares ``instances_are_runs``: joining two bouts a segmenter cut apart would be a claim it
    did not make.
    """
    if not getattr(method, "instances_are_runs", True):
        return list(instances)
    recording_index = getattr(fit, "row_recording_index", None)
    if recording_index is None:
        recording_index = fit.recording_index
    subjects = getattr(fit, "row_subjects", None)
    if subjects is None:
        subjects = fit.subjects

    def same_unit(a, b) -> bool:
        left, right = int(a.end_row), int(b.start_row)
        return (
            recording_index[left] == recording_index[right]
            and subjects[left] == subjects[right]
        )

    fused: list = []
    for instance in instances:
        last = fused[-1] if fused else None
        if (
            last is not None
            and last.cluster_id == instance.cluster_id
            and same_unit(last, instance)
        ):
            fused[-1] = type(instance)(
                last.start_row, instance.end_row, instance.cluster_id
            )
        else:
            fused.append(instance)
    return fused


def coverage_lines(rows: list[dict], names: dict[str, str], inside: str) -> list[str]:
    """:meth:`AuthoredRun.coverage_summary` as one line per recording, for the GUI and the CLI.

    One formatter for both surfaces, so the workspace and ``*-interpret`` cannot word the same
    two numbers differently. ``inside`` is the module's noun ("in a posture", "in a bout"). A
    recording with no timestamp map says "an unknown share" rather than a number built from a
    guessed length.
    """
    return [
        f"{names.get(row['recording_id'], row['recording_id'])}: "
        f"{coverage_fragment(row, inside)}"
        for row in rows
    ]


def coverage_fragment(row: dict, inside: str) -> str:
    """One recording's two numbers: ``analysed 99.7% of the video · in a posture 91.7% of that``."""
    video, analysed = row["video_ns"], row["analysed_ns"]
    of_video = f"{100 * analysed / video:.1f}%" if video else "an unknown share"
    labelled = f"{100 * row['in_instance_ns'] / analysed:.1f}%" if analysed else "none"
    return f"analysed {of_video} of the video · {inside} {labelled} of that"


class LabelOperations:
    """The label edits a clustering run takes (§22), shared so pose and temporal cannot drift.

    A label is a stored thing clusters point at (``storage/run_labels.py``). Every edit here goes
    through ``_label_edit``: fold an older state, apply the change, then refuse it if the labels it
    leaves could not be reported unambiguously (``check_labels``) — before anything is written.
    Folding first matters: a state from before labels had ids is repaired by the fold, while an
    *edit* that would break the rule is refused, never quietly repaired.

    The host supplies ``_edit``, ``_tree`` and ``_reresolve``.
    """

    def _label_edit(
        self,
        op: str,
        change,
        *,
        progress=None,
        stamps: dict | None = None,
        merge: bool = False,
    ) -> Self:
        """Apply ``change`` to the label state and stamp the labels whose clusters it changed.

        ``stamps`` (``{label id: membership_changed_at or None}``) overrides those stamps: an
        undo puts the label's clusters back *and* the time they last changed, so a review made
        before a misclick and its ⌘Z still counts (``restore_labels``).

        Recorded answers follow a path that moved only where the cluster **kept its label** — a
        rename, or a label added above — or where the edit is a ``merge`` (the user saying two
        labels are one behavior). A cluster given to another label is a different claim, and
        an answer about the old label says nothing about it: carried along, a cluster moved to
        a newly made label read the old label's whole review as its own.

        A label's origin (``label_origins``) is the node's own label's: whatever the edit, a node
        left without a label of its own loses it here, so no origin outlives the label it
        explains.
        """
        moved_paths: dict[str, str] = {}

        def mutate(state) -> dict:
            paths_before = _node_paths(state)
            ids_before = dict(state.node_labels)
            folded = self._reresolve(state)
            before = dict(state.node_labels)
            entry = change(state) or {}
            for node in [n for n in state.label_origins if n not in state.node_labels]:
                del state.label_origins[node]
            tree = self._tree()
            check_labels(state.labels, state.node_labels, tree)
            state.resolved_names, state.resolved_paths = label_resolution(
                tree, state.labels, state.node_labels
            )
            changed = membership_changes(before, state.node_labels, tree)
            if changed:
                from detail.core.utils import utc_timestamp

                now = utc_timestamp()
                for label in changed:
                    state.membership_changed_at[label] = now
                entry["membership_changed"] = sorted(changed)
            for label, stamp in (stamps or {}).items():
                if stamp is None:
                    state.membership_changed_at.pop(label, None)
                else:
                    state.membership_changed_at[label] = str(stamp)
            for label in list(state.membership_changed_at):
                if label not in state.labels:
                    del state.membership_changed_at[label]
            kept = None
            if not merge:
                kept = {
                    node
                    for node, label in ids_before.items()
                    if state.node_labels.get(node) == label
                }
            moved_paths.update(path_moves(paths_before, _node_paths(state), kept))
            entry.update(folded or {})
            return entry

        result = self._edit(op, mutate, progress=progress)
        if moved_paths:
            self._carry_answers(moved_paths)
            self._carry_review(moved_paths)
        return result

    def _carry_review(self, moved: dict[str, str]) -> None:
        """Re-point the clip review's answers (``clip_review.csv``) at the paths their labels
        now have, as :meth:`_carry_answers` does for ``audit.csv``.

        A review answer counts only under the label path it was given as
        (``ClipReviewOperations._current_answers``), so without this a rename would leave every
        reviewed cluster unreviewed. Each carried answer is a new row with the original's
        ``drawn_at``; one the new path already has a later answer for is left alone.
        """
        from detail.storage import clip_review as review_store

        if not review_store.review_path(self._run_dir()).is_file():
            return
        latest = review_store.latest_by_path(review_store.read_review(self._run_dir()))
        carried = []
        for (key, path), (_position, row) in latest.items():
            new = carried_path(path, moved) if path else None
            if new is None or new == path:
                continue
            there = latest.get((key, new))
            if there is not None and there[1].drawn_at >= row.drawn_at:
                continue
            carried.append(dataclasses.replace(row, label_path=new))
        review_store.append_rows(self._run_dir(), carried)

    def _carry_answers(self, moved: dict[str, str]) -> None:
        """Re-point recorded answers at the paths their labels now have (§19).

        An answer is stored as a label **path** of names. A rename or a merge changes no instance,
        so the decisions still join their clips, but an answer naming the old path would then match
        nothing: every "yes" read as "no" and precision fell toward zero. So each such answer gets
        a superseding row naming the new path, with the original row's ``drawn_at`` — the log stays
        append-only and "judged since an edit" is unaffected.
        """
        from detail.storage import behavior_audit as audit

        for row in audit.latest_decisions(self._run_dir()):
            if not row.answer or row.answer == audit.NOTHING_CELL:
                continue
            ruled = row.answer.startswith(audit.NOTHING_CELL)
            path = row.answer[len(audit.NOTHING_CELL) :] if ruled else row.answer
            new = carried_path(path, moved)
            if new is None or new == path:
                continue
            audit.append_audit(
                self._run_dir(),
                row.instance_id,
                score=row.score,
                human_label=row.human_label,
                drawn_at=row.drawn_at,
                decided=row.decided,
                answer=audit.ruled_out(new) if ruled else new,
            )

    def name_nodes(
        self, names: dict, *, origins: dict | None = None, progress=None
    ) -> Self:
        """Label nodes **by name**: an existing name adds the node to that label, a new name makes
        a label, and an empty name removes the node's own label (§22).

        ``origins`` (``{node: run_labels.LabelOrigin}``) records, per node given a name, the
        difference screen that pointed at its cluster; it travels with the node's own label and
        goes when that label does. A node with no entry keeps the origin it had, so renaming a
        cluster in place does not forget where it was found.

        The door scripts and the CLI use. A label here covers every member beneath the node unless
        a node inside carries a more specific one, and labels nest rather than replace: a label on a
        branch above an already labelled one leaves the inner one where it is, so a vocabulary can
        be built coarse-first or fine-first and land in the same place. A label left with no
        cluster by this call goes with it, as the per-node name it replaces always did.

        Raises:
            ValueError: the result breaks a rule ``check_labels`` holds — a name with the path
                separator, or a label whose clusters would sit under two parent labels.
        """
        from detail.storage import run_labels

        wanted = {str(int(k)): str(v).strip() for k, v in names.items()}
        wanted_origins = _origins_as_dicts(origins)

        def change(state) -> dict:
            emptied = set()
            for node, name in wanted.items():
                if not name:
                    old = state.node_labels.pop(node, None)
                    if old:
                        emptied.add(old)
                    continue
                if node in wanted_origins:
                    state.label_origins[node] = wanted_origins[node]
                label = run_labels.find(state.labels, name)
                current = state.node_labels.get(node)
                alone = current is not None and [
                    n for n, lab in state.node_labels.items() if lab == current
                ] == [node]
                if label is None and alone:
                    # The node is its label's only cluster and the name is new: that is a rename,
                    # and a rename keeps the label's id — so no instance id moves and recorded
                    # audit judgments keep joining, as renaming a node always did.
                    state.labels[current] = name
                    continue
                if label is None:
                    label = run_labels.mint(state.labels)
                    state.labels[label] = name
                state.node_labels[node] = label
                if (
                    current
                    and current != label
                    and current not in state.node_labels.values()
                ):
                    emptied.add(current)
            for label in emptied - set(state.node_labels.values()):
                state.labels.pop(label, None)
            entry = {"node_names": wanted}
            if wanted_origins:
                entry["origins"] = wanted_origins
            return entry

        return self._label_edit("name nodes", change, progress=progress)

    def create_label(self, name: str, *, progress=None) -> str:
        """Make a label with no clusters yet, and return its id."""
        from detail.storage import run_labels

        made: list[str] = []

        def change(state) -> dict:
            label = run_labels.mint(state.labels)
            state.labels[label] = name.strip()
            made.append(label)
            return {"label": label, "name": name.strip()}

        self._label_edit("create label", change, progress=progress)
        return made[0]

    def label_nodes(
        self, assignments: dict, *, origins: dict | None = None, progress=None
    ) -> Self:
        """Point nodes at labels by id — ``{node: label_id}``, or ``None`` to remove a node's own
        label. What "Add to an existing label" does. ``origins`` as :meth:`name_nodes` takes it.

        Raises:
            KeyError: a label id that does not exist.
            ValueError: the result breaks a rule ``check_labels`` holds.
        """
        wanted = {
            str(int(k)): (None if v is None else str(v)) for k, v in assignments.items()
        }
        wanted_origins = _origins_as_dicts(origins)

        def change(state) -> dict:
            for node, label in wanted.items():
                if label is None:
                    state.node_labels.pop(node, None)
                elif label not in state.labels:
                    raise KeyError(f"no label {label} in this run")
                else:
                    state.node_labels[node] = label
                    if node in wanted_origins:
                        state.label_origins[node] = wanted_origins[node]
            entry = {"node_labels": wanted}
            if wanted_origins:
                entry["origins"] = wanted_origins
            return entry

        return self._label_edit("label nodes", change, progress=progress)

    def rename_label(self, label: str, name: str, *, progress=None) -> Self:
        """Rename a label — every cluster under it at once. No instance id moves: ids carry the
        label's id, never its name, so recorded audit judgments keep joining."""

        def change(state) -> dict:
            if label not in state.labels:
                raise KeyError(f"no label {label} in this run")
            old = state.labels[label]
            state.labels[label] = name.strip()
            return {"label": label, "from": old, "to": name.strip()}

        return self._label_edit("rename label", change, progress=progress)

    def merge_labels(self, source: str, into: str, *, progress=None) -> Self:
        """Move every cluster of ``source`` to ``into`` and remove ``source``.

        A cluster of ``source`` that sits inside a branch labelled ``into`` already reports
        ``into``, so its own label is dropped rather than refused as redundant.
        """

        def change(state) -> dict:
            for label in (source, into):
                if label not in state.labels:
                    raise KeyError(f"no label {label} in this run")
            if source == into:
                raise ValueError("a label cannot be merged into itself")
            tree = self._tree()
            moved = [
                node for node, label in state.node_labels.items() if label == source
            ]
            for node in moved:
                state.node_labels[node] = into
            for node in moved:
                if parent_label(tree, int(node), state.node_labels) == into:
                    del state.node_labels[node]
            names = (state.labels[source], state.labels[into])
            del state.labels[source]
            return {
                "from": names[0],
                "into": names[1],
                "clusters": sorted(moved, key=int),
            }

        return self._label_edit("merge labels", change, progress=progress, merge=True)

    def delete_label(self, label: str, *, progress=None) -> Self:
        """Remove a label and take it off every cluster that carried it."""

        def change(state) -> dict:
            if label not in state.labels:
                raise KeyError(f"no label {label} in this run")
            name = state.labels.pop(label)
            for node in [n for n, lab in state.node_labels.items() if lab == label]:
                del state.node_labels[node]
            return {"label": label, "name": name}

        return self._label_edit("delete label", change, progress=progress)

    def ancestors_of(self, node: int) -> list[int]:
        """The path from the root down to ``node``'s parent, root first, from the fit's tree
        -- what places a label's origin against a screen's kept clusters
        (``api.comparison.label_origin``) with no node cache in hand."""
        return ancestors(self._tree(), node)

    def label_maps(
        self,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, dict]]:
        """``(labels, node_labels, membership_changed_at, label_origins)`` as stored: each label
        id's name, each node's label id, when each label's clusters last changed, and where each
        labelled node's label came from.

        What :func:`label_change` compares, before and after an edit, so the edit can be put back
        exactly (:meth:`restore_labels`) — label ids included, which a write by name cannot do,
        and the change times, so an undo does not read as a new edit of the label.
        """
        state = self.state()
        return (
            dict(state.labels),
            dict(state.node_labels),
            dict(state.membership_changed_at or {}),
            {k: dict(v) for k, v in (state.label_origins or {}).items()},
        )

    def restore_labels(
        self,
        labels: dict[str, str | None],
        node_labels: dict[str, str | None],
        membership_changed_at: dict[str, str | None] | None = None,
        label_origins: dict[str, dict | None] | None = None,
        *,
        progress=None,
    ) -> Self:
        """Set these label ids' names and these nodes' label ids back to the values given —
        ``None`` removes one — and leave every other label and node as it is.

        The inverse of an edit, from :func:`label_change`: INTERPRET's undo writes the *before*
        side and its redo the *after* side. By id, not by name, so undoing a merge brings the
        merged label back under its own id and the audit answers recorded against it join it
        again; :meth:`name_nodes` would mint a new one.

        ``membership_changed_at`` puts back when each label's clusters last changed
        (``None`` removes the stamp). Without it a restore is an edit like any other and stamps
        every label it changes *now* — so undoing a misclick set aside a whole completed review
        of the label, whose clusters were by then exactly what was reviewed.
        ``label_origins`` puts back where each node's label came from the same way, so a redo
        of a label written from a screen does not forget the screen.

        Raises:
            ValueError: the result breaks a rule ``check_labels`` holds.
        """
        wanted_labels = {str(k): v for k, v in labels.items()}
        wanted_nodes = {str(int(k)): v for k, v in node_labels.items()}
        stamps = {str(k): v for k, v in (membership_changed_at or {}).items()}
        wanted_origins = {str(k): v for k, v in (label_origins or {}).items()}

        def change(state) -> dict:
            for label, name in wanted_labels.items():
                if name is None:
                    state.labels.pop(label, None)
                else:
                    state.labels[label] = str(name)
            for node, label in wanted_nodes.items():
                if label is None:
                    state.node_labels.pop(node, None)
                else:
                    state.node_labels[node] = str(label)
            for node, label in list(state.node_labels.items()):
                if label not in state.labels:
                    del state.node_labels[node]
            for node, origin in wanted_origins.items():
                if origin is None:
                    state.label_origins.pop(node, None)
                else:
                    state.label_origins[node] = dict(origin)
            return {"labels": wanted_labels, "node_labels": wanted_nodes}

        return self._label_edit(
            "restore labels", change, progress=progress, stamps=stamps
        )


def label_change(before: tuple[dict, ...], after: tuple[dict, ...]) -> tuple[dict, ...]:
    """One ``{key: (before, after)}`` per map of :meth:`AuthoredRun.label_maps` -- names,
    nodes' labels, membership change times, label origins -- for what differs between two
    reads; ``None`` where one side has no entry."""
    return tuple(
        {
            key: (old.get(key), new.get(key))
            for key in sorted(set(old) | set(new))
            if old.get(key) != new.get(key)
        }
        for old, new in zip(before, after)
    )


def _origins_as_dicts(origins: dict | None) -> dict[str, dict]:
    """``{node: origin dict}`` from what a caller hands ``name_nodes`` -- ``LabelOrigin``
    values or their dicts -- dropping ``None`` entries."""
    from detail.storage import run_labels

    out: dict[str, dict] = {}
    for node, origin in (origins or {}).items():
        if origin is None:
            continue
        if isinstance(origin, run_labels.LabelOrigin):
            out[str(int(node))] = origin.to_dict()
        else:
            parsed = run_labels.LabelOrigin.from_dict(dict(origin))
            if parsed is None:
                raise ValueError(f"node {node}: not a label origin: {origin!r}")
            out[str(int(node))] = parsed.to_dict()
    return out


class ReadOnlyRunError(RuntimeError):
    """Raised when a write is attempted through a snapshot view (§21)."""


#: One lock per run directory, so two authored edits on the same run cannot interleave.
#:
#: §21 says authored operations are **strictly sequential** — "the user acts between operations, so
#: two at once is a bug rather than a concurrency problem to solve". That was true of the *compute*
#: operations it was written about, and quietly false of state edits: every one runs off the UI
#: thread through ``gui/busy.py``, so dragging the granularity slider twice, or naming while a
#: granularity change is still in flight, runs two ``_edit`` calls at once.
#:
#: Both halves of an edit break under that. The state rewrite is read-modify-write, so the second
#: writer overwrites what the first just did; and ``_project`` rewrites the frame views, where two
#: writers race on one zarr store — observed twice in five full-suite runs as
#: ``an array exists ... at path 't_ns'`` and as a vanished ``zarr.<uuid>.partial``, both from one
#: call ``rmtree``-ing the staging directory the other was writing into.
#:
#: A lock rather than a refusal: the second edit is a real thing the user asked for, and making it
#: wait costs nothing (it is already off the UI thread) where raising would surface as an error box
#: for an ordinary double-click. Keyed by directory rather than held on the instance because a
#: handle is rebuilt per call — ``experiment.temporal_clustering_run(id)`` returns a new object each
#: time, so an instance attribute would lock nothing.
_EDIT_LOCKS: dict[str, threading.Lock] = {}
_EDIT_LOCKS_GUARD = threading.Lock()


def _edit_lock(run_dir) -> threading.Lock:
    """The lock for one run directory, created on first use."""
    key = str(run_dir)
    with _EDIT_LOCKS_GUARD:
        return _EDIT_LOCKS.setdefault(key, threading.Lock())


def _as_label(source_label: str) -> int | str:
    """A §19 ``source_label`` as the key its own producer uses.

    ``int`` for a cluster id, the string itself for anything else. Not a coercion with a
    fallback so much as a statement that the label's *type* belongs to the producer: a surface
    that wants an int has one, and a surface whose labels are names is not silently emptied.
    """
    try:
        return int(source_label)
    except ValueError:
        return source_label


class AuthoredRun(AnalysisRun):
    """An analysis run whose result is authored in place after its compute finishes (§21).

    ``inner`` (the executor handle) is present while the compute is in flight and ``None`` for a
    run reached later to work on; every authored operation works either way.
    """

    #: What the run is called in a refusal message ("query run an_03 is finalized…").
    kind: str = "run"
    #: Whether this run's labels nest (§22: labels on tree nodes), so the audit addresses a label
    #: by its **path** and one judgment answers every level. A query's behaviors are flat ids.
    nested_labels: bool = False

    def __init__(self, inner, experiment: Experiment, run_id: str):
        super().__init__(inner, experiment, run_id)

    # — what a subclass supplies ---------------------------------------------------------

    def _read_state(self):
        """The run's authored state, as its own dataclass."""
        raise NotImplementedError

    def _write_state(self, state) -> None:
        """Rewrite the authored state atomically."""
        raise NotImplementedError

    def _log(self, entry: dict) -> None:
        """Append one operation to the run's history — provenance, never authoritative (§21)."""
        raise NotImplementedError

    def _project(self, *, progress=None, reuse: bool = False) -> None:
        """Rebuild the §19 artifact from the compute's stored output and the current state.

        ``reuse`` is an edit's projection (``behavior_artifact.write_artifact``): the intervals
        now, the frame views deferred to :meth:`bring_frame_views_current`.
        """
        raise NotImplementedError

    def _project_frames(self, *, progress=None) -> int:
        """Rebuild the frame views an edit deferred; the producer's ``project_frames``."""
        raise NotImplementedError

    # — lifecycle ------------------------------------------------------------------------

    def _run_dir(self) -> Path:
        return analysis_store.analysis_dir(self._experiment.dir, self.id)

    @property
    def status(self) -> str:
        """The run's own status — ``open`` while it is being worked on, ``complete`` once
        finished.

        Read from the record rather than the executor handle, which only ever describes the
        *compute* job: a run whose compute finished perfectly is ``open``, because nobody has
        said they are done with it.
        """
        return analysis_store.read_analysis_yaml(self._run_dir()).get("status", "open")

    #: Whether this handle may write. ``True`` for a run, ``False`` for a snapshot view.
    #:
    #: Deliberately **not** ``status``. A finalized run is still editable — §21 calls finalize a
    #: statement rather than a lock and ``reopen()`` is one call away — while a snapshot view never
    #: is, whatever the run underneath it says. Two different reasons a control is dead, which is
    #: why a surface that conflates them tells a user who finished a run yesterday the same thing
    #: it tells one who is looking at a frozen copy.
    editable = True

    @property
    def error(self) -> str | None:
        return self._inner.error if self._inner is not None else None

    def progress(self):
        return self._inner.progress() if self._inner is not None else iter(())

    def wait(self):
        # Found by the scan in `tests/api/test_snapshot_view.py`, not by reasoning about it: this
        # reconciles a crashed compute by writing the run's status, which is a write like any
        # other. Waiting on a frozen copy is meaningless anyway — the compute it would wait for
        # belongs to the run.
        self._refuse_if_read_only("wait")
        if self._inner is None:
            return self
        self._inner.wait()
        # An "open" record is the runner's own statement (§21) — right up until the job it
        # was written from failed. `reconciled_status` draws that line: an `open` run whose
        # envelope says `complete` stays open, and one whose envelope says `failed` does not.
        # The runner writes `open` *before* projecting the artifact, so this is not a corner.
        run_dir = self._run_dir()
        status = reconciled_status(
            analysis_store.read_analysis_yaml(run_dir).get("status"), self._inner.status
        )
        if status is not None:
            analysis_store.set_analysis_status(run_dir, status, self._inner.error)
        return self

    def cancel(self) -> None:
        # A view built from a *launch* handle carries a live ``_inner``, so an unguarded cancel
        # would kill the run's compute from a screen opened only to read it. Inert on a reattach
        # handle (``_inner is None``), which is the only reason this was not a live bug.
        self._refuse_if_read_only("cancel")
        if self._inner is not None:
            self._inner.cancel()

    def finalize(self, *, progress=None):
        """Mark the run finished — the user's statement that they are done, and pin the artifact.

        The artifact is unchanged: it already says what the state says. What finalize adds is a
        **snapshot** of it (§21), because "I am done" is the moment a result is most likely to be
        cited, and ``reopen()`` is one call away — a statement that can move out from under a
        number somebody has already read is not a statement.

        Long when edits have deferred frame views (see :meth:`pin`), so it reports through
        ``progress(stage, fraction, detail)``.
        """
        self._refuse_if_read_only("finalize")
        self.pin(reason="finish", progress=progress)
        analysis_store.set_analysis_status(self._run_dir(), "complete")
        self._log({"op": "finalize"})
        return self

    def reopen(self):
        """Move a finalized run back to ``open`` so it can be worked on again.

        The counterpart to ``finalize`` being a statement rather than a lock (§21) — without it
        the soft lock would be a hard one.
        """
        self._refuse_if_read_only("reopen")
        analysis_store.set_analysis_status(self._run_dir(), "open")
        self._log({"op": "reopen"})
        return self

    # — snapshots (§21) ------------------------------------------------------------------

    def pin(self, *, reason: str, progress=None) -> str | None:
        """Freeze the working artifact and return the snapshot id.

        Returns the *existing* snapshot when the artifact has not changed since the last pin, so
        exporting the same run twice yields one snapshot rather than two, and ``None`` when the
        run has no artifact to pin (a fit whose granularity is unset — §21).

        Brings the frame views current first: a snapshot copies them and is never rewritten, so
        this is the last moment they can be. That rebuild is the long part, and the only one that
        reports through ``progress``.
        """
        self._refuse_if_read_only("pin")
        self.bring_frame_views_current(progress=progress)
        return snapshots_store.pin(
            self._run_dir(),
            reason=reason,
            state=self._snapshot_state(),
            code_sha=provenance.code_version(),
        )

    def _snapshot_state(self) -> dict:
        """The authored decisions, as plain data for the snapshot record.

        A subclass whose state is not a dataclass overrides this; every one today is.
        """
        state = self._read_state()
        if dataclasses.is_dataclass(state):
            return dataclasses.asdict(state)
        return dict(getattr(state, "__dict__", {}))

    def snapshots(self) -> list[dict]:
        """Every pinned copy of this run's artifact, oldest first."""
        run_dir = self._run_dir()
        return dirscan.read_each(
            snapshots_store.discover_snapshots(run_dir),
            lambda sid: snapshots_store.read_snapshot(run_dir, sid),
        )

    def snapshot(self, snapshot_id: str) -> dict:
        """One snapshot's record — what it was taken for, and the state behind it."""
        return snapshots_store.read_snapshot(self._run_dir(), snapshot_id)

    #: The module's :class:`SnapshotViewMixin` subclass, or ``None`` where reading a snapshot back
    #: as an interpretation does not mean anything. ``None`` is a declaration, not an omission: it
    #: is what tells a surface to show the snapshot rows without a cue that opens them. Every
    #: module that authors an interpretation has one — Query included, whose cutoffs are as much an
    #: interpretation as a granularity is.
    snapshot_view_class = None

    def view(self, snapshot_id: str):
        """This run as it was when ``snapshot_id`` was pinned — read-only (§21).

        Everything that reads the run reads the frozen state and the frozen artifact instead; every
        write refuses. Returns ``None`` when the module declares no view class.

        Raises:
            FileNotFoundError: If the run has no such snapshot.
            ValueError: If the snapshot recorded no state. Snapshots pinned before a comparison
                pinned through the run handle carry ``state: {}``, so there is no granularity and
                no vocabulary to render — a fact worth saying plainly, since the alternative is a
                ``KeyError`` from the state dataclass.
        """
        if self.snapshot_view_class is None:
            return None
        record = self.snapshot(snapshot_id)
        if not (record.get("state") or {}):
            raise ValueError(
                f"snapshot {snapshot_id} of {self.id} recorded no settings, so there is no "
                "interpretation to show. It was pinned before snapshots carried the state behind "
                "them; its frozen instances are still readable with snapshot_artifact()."
            )
        return self.snapshot_view_class(self, snapshot_id, record)

    def snapshot_interpretation(self, record: dict) -> SnapshotInterpretation:
        """Summarize one snapshot record in this module's own terms.

        The default says nothing rather than guessing: a module that has not declared how it
        interprets a snapshot shows the columns every snapshot has (when, why, how many instances,
        what used it) and no invented ones. Overriding is how a module earns its own.
        """
        return SnapshotInterpretation(columns=(), cells=(), headline="")

    def instance_placements(self) -> dict | None:
        """``{cluster_id: {recording_id: [placement, ...]}}`` — where each instance was, **and
        which instance it was**, or ``None`` when this run has no positions at all.

        A placement is a dict with ``instance_id``, ``cx``, ``cy``, ``subject``, ``t_ns`` and
        ``end_t_ns`` — enough for a surface to turn a clicked dot back into the thing it stands
        for. :meth:`instance_positions` is the coordinates-only view of this, derived rather
        than read a second time.

        Identity is carried rather than recomputed on purpose. ``instance_id`` is a hash of
        ``(recording_id, subject, source_label, t_ns, end_t_ns)``, and for a **pose** run the
        artifact's ``end_t_ns`` is the end of the row *run* while ``member_clips`` reports the
        one row, ``end_t_ns == t_ns``. (It used to extend the row by its sampling period, as a
        preview "with duration" — which a player reading both ends inclusive turned into the
        row plus the next frame, and a thumbnail at the span's middle into the next frame
        alone.) Rebuilding the hash from a clip would therefore miss for pose and match for
        temporal, silently and only on one module.
        """
        output = self.artifact()
        if output.features is None:
            return None
        out: dict[int | str, dict[str, list]] = {}
        for row in behavior_output.bouts(output.instances):
            position = output.features.get(row.instance_id)
            if position is None or position[0] is None or position[1] is None:
                continue
            # The producer's label, as an `int` where it is one. Clustering labels its groups
            # `0`, `1`, `2` and its surfaces index a colour palette with them; Query labels its
            # behaviors `q01`, `q02` (`storage/query_runs.py::behavior_id`). This coerced
            # unconditionally and dropped what would not parse, under a comment claiming the row
            # belonged to another producer — so **every** query run reported no placements at
            # all, and `instance_positions` with it. A key that cannot hold one producer's
            # labels is not a filter, it is a hole.
            cluster = _as_label(row.source_label)
            out.setdefault(cluster, {}).setdefault(row.recording_id, []).append(
                {
                    "instance_id": row.instance_id,
                    "cx": float(position[0]),
                    "cy": float(position[1]),
                    "subject": int(row.subject),
                    "t_ns": int(row.t_ns),
                    "end_t_ns": int(row.end_t_ns),
                }
            )
        return out

    def instance_positions(self) -> dict | None:
        """``{cluster_id: {recording_id: [(cx_norm, cy_norm), ...]}}`` — where each instance was,
        or ``None`` when this run has no positions to report at all.

        On the base rather than on one run class, because every §19 producer writes the same
        artifact and the surfaces that read it are shared. It lived on ``TemporalClusteringRun``
        alone, and ``ClusterWorkspace`` — which pose and behavior clustering both use — reached
        for it with ``getattr(run, "instance_positions", None)``. A run type that never
        implemented it was then indistinguishable from a run that had no positions, and both
        silently hid the map: pose clustering shipped without the spatial map the other
        clustering pages had, and nothing anywhere said so.

        ``None`` is the actionable state — the run predates §14's ``kinematics/`` and
        ``recording backfill-kinematics`` fills it in. ``{}`` means there were positions but
        none landed in a cluster, which needs no action and no sentence.

        Grouped here rather than in the widget because it is a fact about this run's artifact,
        not about how a panel is laid out (§22: the method answers, the surface renders).
        """
        placements = self.instance_placements()
        if placements is None:
            return None
        return {
            cluster: {
                recording_id: [(p["cx"], p["cy"]) for p in placements_here]
                for recording_id, placements_here in by_recording.items()
            }
            for cluster, by_recording in placements.items()
        }

    def snapshot_artifact(self, snapshot_id: str):
        """The frozen §19 artifact a snapshot holds.

        Read from the snapshot rather than re-derived from its recorded state: re-deriving under
        later code is exactly what makes an old result unreproducible (§21).
        """
        return behavior_output.read_behavior_output(
            snapshots_store.snapshot_dir(self._run_dir(), snapshot_id)
        )

    def coverage_summary(self) -> list[dict]:
        """Per recording: how much of the video was analysed, and how much of that is labelled.

        ``{recording_id, video_ns, analysed_ns, in_instance_ns}`` — two ratios a user reads
        together, because one number cannot say *which* step lost a frame: ``analysed / video``
        is tracking and embedding, ``in_instance / analysed`` is this run. Read from
        :meth:`artifact`, so a snapshot view reports its frozen numbers.

        ``video_ns`` is ``None`` when the recording has no timestamp map. Instance time uses
        :func:`behavior_output.instance_ns`, the rule the cluster table and measures use, so the
        three agree.
        """
        from detail.storage import read_cache

        output = self.artifact()
        periods = behavior_output.period_by_recording(output.coverage)
        analysed: dict[str, int] = {}
        for row in output.coverage:
            analysed[row.recording_id] = analysed.get(row.recording_id, 0) + int(
                row.observed_ns
            )
        inside: dict[str, int] = {}
        for row in behavior_output.bouts(output.instances):
            inside[row.recording_id] = inside.get(
                row.recording_id, 0
            ) + behavior_output.instance_ns(row, periods)
        with read_cache.pinned():
            return [
                {
                    "recording_id": recording_id,
                    "video_ns": self._video_ns(recording_id),
                    "analysed_ns": analysed_ns,
                    "in_instance_ns": inside.get(recording_id, 0),
                }
                for recording_id, analysed_ns in analysed.items()
            ]

    def _video_ns(self, recording_id: str) -> int | None:
        """A recording's video length, from its timestamp maps (Tenet 8), not ``total_duration_s``."""
        from detail.storage import experiment as experiment_store
        from detail.storage import recording as recording_store
        from detail.storage import video as video_store

        recording_dir = experiment_store.recording_path(
            self._experiment.dir, recording_id
        )
        try:
            data = recording_store.read_recording(recording_dir)
            parts = data["video"]["files"]
            last = recording_store.load_active_timestamps_cached(
                recording_dir, data, len(parts) - 1
            )
        except (FileNotFoundError, KeyError, IndexError, ValueError):
            return None
        t_ns = last.values
        if not len(t_ns):
            return None
        return int(t_ns[0]) + video_store.part_duration_ns(t_ns, 0.0)

    @property
    def artifact_stale(self) -> bool:
        """Whether this run's §19 artifact is known not to match its authored state.

        Set while a reprojection is in flight and cleared when it lands, so a reprojection that
        dies — a transient share write, a kill — leaves the disagreement *recorded* instead of
        silent. :meth:`reproject` is the way back.
        """
        return analysis_store.artifact_stale(self._run_dir())

    def _set_stale(self, stale: bool) -> None:
        """Mark (or clear) the artifact-vs-state disagreement.

        The mark goes down **before** the work and never in an ``except``: a SIGKILL mid-write
        leaves exactly the disagreement an exception does, and nothing catches that.
        """
        self._refuse_if_read_only("mark its artifact stale")
        analysis_store.set_artifact_stale(self._run_dir(), stale)

    def reproject(self, *, progress=None):
        """Rebuild the §19 artifact from the stored compute output and the current state.

        Every edit already does this; it is public for the case where the artifact and the state
        disagree — a crash between the two writes — since §19's rule for a materialized view is
        to regenerate rather than to reconcile. Long (a shard read and a frame-view zarr per
        analyzed unit), so it reports through ``progress(stage, fraction, detail)``.
        """
        self._refuse_if_read_only("reproject")
        with _edit_lock(self._run_dir()):
            self._set_stale(True)
            self._project(progress=progress)
            self._set_stale(False)
        return self

    def bring_frame_views_current(self, *, progress=None) -> int:
        """Rebuild the frame views the last edits deferred; return how many were rebuilt.

        An edit writes the intervals and leaves the per-frame views, which are most of a
        reprojection's cost and read by none of the surfaces an edit refreshes (#325). Anything
        that reads them — measures, a pin, an export — calls this first, and a reader that did not
        gets ``FrameViewsPendingError`` rather than a stale view. Free when nothing is pending;
        otherwise one zarr per recording the edits actually moved, reported through
        ``progress(stage, fraction, detail)``.

        Under the edit lock, so it cannot interleave with an edit that is moving the intervals
        under it.
        """
        self._refuse_if_read_only("bring_frame_views_current")
        # The free case asks nothing of the producer — not its state, not the lock — so every
        # reader can call this unconditionally, whatever state the run is otherwise in.
        if not frame_view.is_pending(self._run_dir()):
            return 0
        with _edit_lock(self._run_dir()):
            return self._project_frames(progress=progress)

    # — the operation --------------------------------------------------------------------

    def _edit(self, op: str, mutate, *, reproject: bool = True, progress=None):
        """Apply one authored edit: rewrite the state, log it, reproject if it changed what the
        instances are.

        Every edit funnels through here so the finalize check, the history entry and the
        reprojection cannot be forgotten by one of them — and, now, so that two of them cannot
        run at once (see :data:`_EDIT_LOCKS`). The read-modify-write and the reprojection are
        both inside the lock, because both break when they interleave.
        """
        self._refuse_if_read_only(op)
        with _edit_lock(self._run_dir()):
            self._refuse_if_finalized(op)
            state = self._read_state()
            entry = mutate(state)
            # Read before the mark below goes down. A clear flag is the statement that the frame
            # views on disk agree with `instances.csv`, which is the only thing that makes keeping
            # an unchanged unit's view safe; a run left stale by a killed edit is rebuilt whole.
            reuse = reproject and not self.artifact_stale
            if reproject:
                # Marked before the state is written, because from here until the reprojection
                # lands the two genuinely disagree. The old order — state, history, then the
                # work that makes them true — left a run saying `granularity: 138` over an
                # artifact saying `null`, with a history line asserting it had worked and no
                # surface able to tell. A durable commit ordered before the operation that
                # justifies it needs, at minimum, to record that it is not justified yet.
                self._set_stale(True)
            self._write_state(state)
            self._log({"op": op, **(entry or {})})
            if reproject:
                self._project(progress=progress, reuse=reuse)
                self._set_stale(False)
        return self

    #: Every method on this class that writes to the run directory.
    #:
    #: Declared rather than inferred so a read-only view can refuse all of them at once, and so
    #: ``tests/api/test_snapshot_view.py`` can scan this module for a writer that was added without
    #: joining the list. ``finalize`` and ``pin`` are the ones that make the list necessary: neither
    #: goes through ``_edit``, and both write to ``self._run_dir()`` — the *live* run — so an
    #: unguarded ``finalize()`` on a view would finish the run somebody was only looking at.
    _WRITE_METHODS = (
        "_edit",
        "reproject",
        "bring_frame_views_current",
        "pin",
        "finalize",
        "reopen",
        "wait",
        "_record",
        "audit_sample",
        "label_audit",
        "cancel",
        "_write_state",
        "_set_stale",
        "rebuild_node_cache",
    )

    def _audit_dir(self) -> Path:
        """Where the §19 reliability audit's decisions are **read** from.

        A seam, because a snapshot carries its own copy of ``audit.csv`` and the panel that reports
        precision has to read *that* one. Reading by run directory mixed a frozen numerator with
        live decisions: a snapshot whose own file said ``precision 1.00 (n=3)`` reported
        ``0.50 (n=6)`` after three more clips were judged on the live run — a number that was never
        true of anything.

        Writes are deliberately **not** routed through here. ``audit_sample`` and ``label_audit``
        keep addressing ``_run_dir()`` and keep being refused on a view; a snapshot's audit is
        evidence to read, never to add to.
        """
        return self._run_dir()

    def _refuse_if_read_only(self, op: str) -> None:
        """A snapshot view refuses every write, and names the way back.

        The whole guarantee is that what left a run cannot be changed by looking at it. A view
        addresses the *run's* directory for everything except its state and its artifact, so a
        write that slipped through would land on the live run rather than on the frozen copy --
        the failure would not even be local to the thing being looked at.
        """
        if not self.editable:
            raise ReadOnlyRunError(
                f"{self.kind} {self.id} is being read as it was at "
                f"{getattr(self, '_snapshot_id', 'a snapshot')}, so it will not {op}. Snapshots "
                "are frozen by design (§21); work on the run itself instead."
            )

    # -- the reliability audit (§19) ------------------------------------------------------

    def _audit_pool(self, label: str | None):
        """The instances an audit may sample from — the artifact's, by default.

        A subclass overrides this when its artifact is a *projection* of something larger.
        ``QueryRun`` does: ``instances.csv`` holds only what the current cutoff kept, so
        sampling it would make the sample a function of a decision the user is still making,
        and every adjustment would invalidate part of the evidence.
        """
        rows = list(self.artifact().instances)
        return [r for r in rows if label is None or r.source_label == label]

    def _audit_examples(self, rows) -> set[str]:
        """The ``instance_id``s among ``rows`` that the user's own clips overlap — never drawn and
        never counted (§19: an instance that fed the method cannot also score it).

        Empty here: a clustering run is given no clips. ``QueryRun`` overrides it.
        """
        return set()

    def _audit_edges(self, labels, *, fix_seed: int | None = None) -> dict | None:
        """``{source_label: edges}`` — where each label's scores are cut into audit bins.

        ``None`` here: a clustering run's instances carry no score, so there is nothing to bin.
        ``QueryRun`` overrides it, and with ``fix_seed`` records them at a label's first draw.
        """
        return None

    def _audit_rows(self, label: str | None, cutoff: float | None):
        """``(rows, cutoff)`` an estimate is over: the instances it may count, and the cutoff the
        storage layer still has to apply to them.

        The artifact's instances here, cut at ``cutoff``. ``QueryRun`` hands back its candidates
        at no cutoff already cut at each behavior's own, because a query estimate is weighted
        by score bin and the bins are sized from the candidates, not from the artifact.
        """
        rows = [
            row
            for row in self.artifact().instances
            if label is None or row.source_label == label
        ]
        return rows, cutoff

    def audit_sample(
        self,
        *,
        label: str | None = None,
        path: str | None = None,
        budget: int | None = None,
        seed: int = 0,
        own_only: bool = False,
        node: int | None = None,
    ) -> list:
        """Draw a batch to review and record it as drawn, returning the instances.

        ``budget`` is a **batch**, not a cap: more is always one more call away. ``label``
        scopes to one behavior — §19 reviews one at a time, because alternating "is this
        rearing?" with "is this grooming?" corrupts the estimate quietly — and ``None`` spans
        the artifact, which is what a clustering vocabulary wants.

        **Pending first, then a fresh draw.** A batch is recorded as drawn before it is judged
        (``human_label`` is tri-state precisely so "drawn, not yet judged" is a state), so
        anyone who walks away mid-batch leaves rows behind. Excluding *every* logged row from
        the next draw — which is what this used to do — made those rows unreachable for the
        life of the run: they were never judged and never offered again, so the sample was
        quietly smaller than the pool could have given and nothing failed. Offering them back
        first also makes leaving a batch free, which is what lets the surface switch behavior
        mid-review without a confirmation dialog: nothing is lost, so there is nothing to warn
        about.

        ``path`` scopes a nested vocabulary to one label **and everything under it** — "rearing"
        draws from its own instances and from "rearing/supported" — which is what lets a single
        batch score both levels (:func:`~detail.storage.behavior_audit.verdict`). An instance
        with no label is never drawn: the unlabelled remainder makes no claim to confirm.
        """
        from detail.storage import behavior_audit as audit
        from detail.storage import behavior_output as bo

        budget = audit.DEFAULT_BUDGET if budget is None else budget
        # Before any write. `_record` carries the finalize check, so drawing first would append
        # rows to a finalized run and *then* refuse — the check has to gate the operation, not
        # report on it.
        self._record(
            "audit_sample",
            {
                "label": label,
                "path": path,
                "budget": budget,
                "seed": seed,
                "own_only": own_only,
                "node": node,
            },
        )
        pool = self._audit_pool(label)
        if self.nested_labels:
            pool = self._audit_scope(pool, path, own_only=own_only)
        # The user's own clips are never offered: they told the method what the behavior is,
        # so judging them scores the answer key (§19).
        examples = self._audit_examples(pool)
        pool = [row for row in pool if row.instance_id not in examples]
        clusters = self._audit_clusters(pool)
        if node is not None:
            # One cluster on its own: what a reviewer asks for by choosing a cluster's row.
            pool = [r for r in pool if (clusters or {}).get(r.instance_id) == node]
        # `decided`, not `human_label is not None`: an **unclear** is a decision that carries
        # no evidence either way, so a metric drops it — but offering it again as though it had
        # never been seen means an ambiguous clip comes back forever.
        # A label edited after its audit counts only judgments since (`audit_precision`), so the
        # clips judged before are drawable again — set aside and never re-offered, a fully audited
        # label could not be audited further. One cluster on its own keeps every judgment, as its
        # figure does.
        since = None if node is not None else self._label_changed_at(path)
        decided = {
            row.instance_id
            for row in audit.latest_decisions(self._run_dir())
            if row.decided and (since is None or (row.drawn_at or "") >= since)
        }
        offered = {
            row.instance_id
            for row in audit.read_audit(self._run_dir())
            if since is None or (row.drawn_at or "") >= since
        }
        # Drawn but never judged, and still in this label's pool. Ordered by the pool so a
        # resumed batch reads the way the first one did.
        pending = [
            row
            for row in bo.bouts(pool)
            if row.instance_id in offered and row.instance_id not in decided
        ]
        drawn = pending[:budget]
        if len(drawn) < budget:
            fresh = audit.draw_sample(
                pool,
                stratum_of=clusters,
                budget=budget - len(drawn),
                seed=seed,
                exclude_instance_ids=decided | offered,
                edges=self._audit_edges(
                    {str(row.source_label) for row in pool}, fix_seed=seed
                ),
            )
            for row in fresh:
                audit.append_audit(self._run_dir(), row.instance_id, score=row.score)
            drawn = [*drawn, *fresh]
        return drawn

    def label_audit(
        self, instance_id: str, is_match: bool | None, *, answer: str = ""
    ) -> Self:
        """Record one human decision. Appends — an earlier decision is superseded, not erased.

        ``is_match`` is whether the instance is its own (deepest) label; ``None`` is unclear.
        ``answer`` is the level that *is* true when the label nests — ``"rearing"`` for
        "rearing, not supported", or :data:`~detail.storage.behavior_audit.NOTHING_CELL` for
        "none of these". :func:`~detail.storage.behavior_audit.answer_options` lists the
        ``(text, answer, is_match)`` triples a check of one label offers, and
        :func:`~detail.storage.behavior_audit.human_label_for` derives ``is_match`` from an
        answer and the clip's own path, so a caller never has to derive the pair by hand.
        """
        from detail.storage import behavior_audit as audit

        self._record(
            "label_audit",
            {"instance_id": instance_id, "label": is_match, "answer": answer},
        )
        audit.append_audit(
            self._run_dir(),
            instance_id,
            human_label=is_match,
            decided=True,
            answer=answer,
        )
        return self

    @staticmethod
    def _audit_scope(rows, path: str | None, *, own_only: bool = False) -> list:
        """The labelled rows an audit draws from, narrowed to ``path`` and what nests under it.

        For a run whose labels nest. A query's candidates are enumerated before any name is
        attached (``QueryRun._audit_pool``), so "has a label" would empty its pool.
        """
        from detail.storage import behavior_audit as audit

        return [
            row
            for row in rows
            if row.label
            and (path is None or audit.under(row.label_path or row.label, path))
            and (not own_only or (row.label_path or row.label) == path)
        ]

    def audit_labels(self) -> list[str]:
        """Every label path the audit can address — each label present, and each above it.

        A parent appears even when no instance reports it directly (every member sits under a
        sub-label), because its precision is still a question: it is scored over everything
        under it. Ordered so a parent precedes its sub-labels, which is how a surface nests them.
        """
        from detail.storage import behavior_output as bo

        paths: set[str] = set()
        for row in self.artifact().instances:
            if not row.label:
                continue
            parts = (row.label_path or row.label).split(bo.PATH_SEPARATOR)
            for depth in range(1, len(parts) + 1):
                paths.add(bo.PATH_SEPARATOR.join(parts[:depth]))
        return sorted(paths, key=lambda p: p.split(bo.PATH_SEPARATOR))

    def audit_precision(
        self,
        *,
        label: str | None = None,
        path: str | None = None,
        cutoff: float | None = None,
        own_only: bool = False,
    ) -> dict | None:
        """The precision the audit supports, with its interval, its ``n`` and its coverage.

        The **read half** of the audit. Everything above records a judgment; without this
        nothing could read what the judging bought, so the number existed only inside the GUI,
        which reached past the API into storage to get it. A script could spend an afternoon
        auditing and not retrieve the figure that afternoon produced.

        ``None`` when nothing has been audited — the caller says "no reliability estimate" and
        never substitutes one. A returned ``precision`` of ``None`` with a ``reason`` is the
        other refusal: the audit no longer covers enough of this run to speak for it
        (:func:`~detail.storage.behavior_audit.decision_coverage`).

        ``path`` asks about one label of a nested vocabulary, over every instance under it. Where
        the run's clusters are known (a clustering run with a node cache) the estimate is
        **stratified** by cluster — each cluster's precision weighted by its share of the label —
        because the draw gives undecided clusters more than their share. A label whose clusters
        changed after it was audited counts only judgments made since (``membership_changed_at``);
        the result says so in ``since``.

        Read from ``_audit_dir()``, so a snapshot reports its own frozen audit rather than the
        live run's — the mix that seam exists to prevent.
        """
        from detail.storage import behavior_audit as audit

        artifact = self.artifact()
        rows, storage_cutoff = self._audit_rows(label, cutoff)
        if own_only and path:
            # The label's own clusters only — its "other": instances reporting as plain *path*.
            rows = [row for row in rows if (row.label_path or row.label) == path]
        examples = self._audit_examples(rows)
        rows = [row for row in rows if row.instance_id not in examples]
        clusters = self._audit_clusters(self._audit_scope(rows, path)) if path else None
        since = self._label_changed_at(path)
        estimate = audit.precision_at_cutoff(
            self._audit_dir(),
            rows,
            cutoff=storage_cutoff,
            manifest=artifact.manifest,
            universe=self._audit_pool(None),
            target_path=path,
            stratum_of=clusters,
            since=since,
            edges=self._audit_edges({str(row.source_label) for row in rows}),
        )
        if estimate is not None:
            estimate["cutoff"] = cutoff
            estimate["since"] = since
            if path:
                estimate.update(self._audit_left(self._audit_scope(rows, path), since))
        return estimate

    def _audit_left(self, rows, since: str | None) -> dict:
        """``{"size", "remaining"}``: the scope's instances, and how many are still unjudged.

        A judgment from before ``since`` is set aside here exactly as the draw sets it aside, so
        "nothing left" means what :meth:`audit_sample` will find.
        """
        from detail.storage import behavior_audit as audit

        decided = {
            row.instance_id
            for row in audit.latest_decisions(self._audit_dir())
            if row.decided and (since is None or (row.drawn_at or "") >= since)
        }
        bouts = list(behavior_output.bouts(rows))
        return {
            "size": len(bouts),
            "remaining": sum(1 for row in bouts if row.instance_id not in decided),
        }

    def _label_changed_at(self, path: str | None) -> str | None:
        """When the clusters of the label at ``path`` last changed, or ``None``."""
        if not path or not self.nested_labels:
            return None
        from detail.storage import run_labels

        state = self.state()
        label = run_labels.find(
            state.labels, path.split(behavior_output.PATH_SEPARATOR)[-1]
        )
        return (getattr(state, "membership_changed_at", {}) or {}).get(label)

    def _audit_clusters(self, rows) -> dict | None:
        """``{instance_id: node}`` — the labelled cluster holding most of each instance — or
        ``None`` where the run has no clusters to attribute to (a query, or a clustering run
        without its node cache)."""
        return None

    def audit_by_cluster(
        self,
        *,
        path: str | None = None,
        own_only: bool = False,
    ) -> list[dict]:
        """Each labelled cluster's precision with its 95% confidence interval (§19, clusters).

        ``[{node, label, path, size, share, estimate, answers}]`` — the question a user asked by
        picking the cluster: is it as pure as they thought? The audit reports the interval; the
        answer is the reader's. ``path`` is the label path the cluster carries; ``size`` its
        instances; ``share`` its part of its label's instances; ``answers`` counts what its judged
        clips were
        answered as, in the words of the buttons (``answer_text``), which is what says *what* an
        impure cluster is instead. The ``path`` argument limits it to clusters under one label;
        ``None`` gives every labelled cluster, which INTERPRET draws on its tree.

        Every judgment a cluster has counts here, before or after a label edit: the figure is about
        the cluster, and the cluster did not change. Empty where clusters are not known.
        """
        from detail.storage import behavior_audit as audit
        from detail.storage import run_labels

        rows = [row for row in self.artifact().instances if row.label]
        if path:
            rows = self._audit_scope(rows, path, own_only=own_only)
        clusters = self._audit_clusters(rows)
        if clusters is None:
            return []
        state = self.state()
        names = run_labels.node_names(state.labels, state.node_labels)
        decisions = {
            r.instance_id: r for r in audit.latest_decisions(self._audit_dir())
        }
        sizes: dict = {}
        decided: dict = {}
        judged: dict = {}
        answers: dict = {}
        label_size: dict = {}
        for row in behavior_output.bouts(rows):
            node = clusters.get(row.instance_id)
            if node is None:
                continue
            own = row.label_path or row.label
            sizes[node] = sizes.get(node, 0) + 1
            label_size[names.get(str(node), "")] = (
                label_size.get(names.get(str(node), ""), 0) + 1
            )
            decision = decisions.get(row.instance_id)
            if decision is None or not decision.decided:
                continue
            decided[node] = decided.get(node, 0) + 1
            node_path = state.resolved_paths.get(
                state.node_labels.get(str(node), ""), own
            )
            value = audit.verdict(decision, own, node_path)
            if value is not None:
                judged.setdefault(node, []).append(value)
            text = audit.answer_text(decision, own)
            answers.setdefault(node, {})
            answers[node][text] = answers[node].get(text, 0) + 1
        out = []
        for node in sorted(sizes, key=lambda n: (names.get(str(n), ""), -sizes[n])):
            label = names.get(str(node), "")
            out.append(
                {
                    "node": int(node),
                    "label": label,
                    "path": state.resolved_paths.get(
                        state.node_labels.get(str(node), ""), ""
                    ),
                    "size": sizes[node],
                    "share": sizes[node] / max(label_size.get(label, 1), 1),
                    "estimate": {
                        **audit.estimate(judged.get(node, [])),
                        "size": sizes[node],
                        "remaining": sizes[node] - decided.get(node, 0),
                    },
                    "answers": answers.get(node, {}),
                }
            )
        return out

    def audit_precision_by_recording(
        self,
        *,
        label: str | None = None,
        path: str | None = None,
        cutoff: float | None = None,
        own_only: bool = False,
    ) -> list[dict] | None:
        """The same estimate split per recording — where one cutoff behaves differently in
        different animals, which an overall figure hides completely (§20)."""
        from detail.storage import behavior_audit as audit

        artifact = self.artifact()
        rows, storage_cutoff = self._audit_rows(label, cutoff)
        examples = self._audit_examples(rows)
        rows = [
            row
            for row in rows
            if row.instance_id not in examples
            and (not (own_only and path) or (row.label_path or row.label) == path)
        ]
        return audit.precision_by_recording(
            self._audit_dir(),
            rows,
            cutoff=storage_cutoff,
            manifest=artifact.manifest,
            universe=self._audit_pool(None),
            target_path=path,
            edges=self._audit_edges({str(row.source_label) for row in rows}),
        )

    def audit_precision_by_score_bin(
        self, *, label: str, cutoff: float | None = None
    ) -> list[dict] | None:
        """One behavior's precision in each of its audit's score bins — the confidence curve.

        Over the instances ``cutoff`` keeps (the behavior's own when ``None``), in the bins its
        audit was drawn in, so a point is the share of exactly the clips that bin's draw gave it.
        ``None`` when nothing is audited, or when the run's instances carry no score to bin.
        """
        from detail.storage import behavior_audit as audit

        artifact = self.artifact()
        rows, storage_cutoff = self._audit_rows(label, cutoff)
        examples = self._audit_examples(rows)
        direction = audit.score_direction(artifact.manifest)
        rows = [
            row
            for row in rows
            if row.instance_id not in examples
            and audit.retained(row.score, storage_cutoff, direction)
        ]
        edges = (self._audit_edges({label}) or {}).get(label)
        if edges is None:
            return None
        return audit.precision_by_score_bin(
            self._audit_dir(),
            rows,
            manifest=artifact.manifest,
            universe=self._audit_pool(None),
            edges=edges,
        )

    def audit_decisions(self) -> list:
        """Every standing decision — one row per instance, the most recent judgment about it.

        ``AuditRow.decided`` separates "judged unclear" from "never looked at"; both carry
        ``human_label`` of ``None``.
        """
        from detail.storage import behavior_audit as audit

        return audit.latest_decisions(self._audit_dir())

    def _record(self, op: str, entry: dict) -> None:
        """Refuse if finalized, then log — for an operation that writes no state.

        The audit writes to ``audit.csv``, not to the run's state, and reprojects nothing: a
        decision is evidence *about* the instances rather than a change to them. So it cannot
        go through ``_edit``, which rewrites state and reprojects. It still has to make the
        same finalize check, and this is the one place that policy lives for those operations.

        **Auditing a finalized run is refused today**, and that is a decision rather than an
        inevitability: nothing about the instances, the cutoff or the export moves when someone
        labels a clip, so a case exists for allowing it. It is refused because precision is a
        number a reader sees, and a finalized run whose reported precision shifts underneath
        them is the thing ``_refuse_if_finalized`` exists to prevent. ``reopen()`` is one call.
        If that trade is revisited, this method is the only place it has to change.
        """
        self._refuse_if_read_only(op)
        self._refuse_if_finalized(op)
        self._log({"op": op, **entry})

    def _refuse_if_finalized(self, op: str) -> None:
        """A finalized run refuses an edit and names the way back.

        §21 calls finalize "a statement by the user that they are done, not a lock", and this is
        not one: ``reopen()`` is a single call away. But a statement that can be silently
        overwritten is not a statement, so the run says no once rather than quietly moving under
        a result somebody has already read.
        """
        if self.status == "complete":
            raise ValueError(
                f"{self.kind} {self.id} is finalized, so it will not {op}. Call reopen() "
                "first if you meant to change it."
            )


class SnapshotViewMixin:
    """The run as it was when a snapshot was pinned: same class, frozen inputs, no writes.

    **Two overrides carry the whole thing.** Every read on an authored clustering run opens with
    ``state = self.state()`` and ``fit = self._load_fit()`` — ``clusters``, ``projection``,
    ``occupancy``, ``ethogram_rows``, ``member_clips``, ``representatives``, all of them. So
    replacing where the *state* comes from and where the *artifact* comes from replaces every
    number on the screen, and nothing else has to know a snapshot exists. That is also why this is
    a subclass rather than a wrapper: a delegating wrapper hands back the run's own bound methods,
    whose ``self.state()`` is still the run's.

    **The fit is deliberately read from the run, not the snapshot.** ``fit.npz`` has one writer --
    the module's run job — and a re-fit mints a new ``an_NN``, so a run's fit never changes under
    a snapshot. Copying it into every snapshot would multiply the largest file in the run
    directory to freeze something that cannot move.

    A subclass supplies ``_state_class``, the dataclass its module's state is stored as.
    """

    #: The module's state dataclass; it must expose ``from_dict``.
    _state_class = None

    editable = False

    #: Instance attributes that must **not** come across from the live run.
    #:
    #: ``editable`` because it is the single switch behind every refusal: it is a *class* attribute
    #: here, so an instance-level one on the run would shadow it and silently make a read-only view
    #: writable. The caches because they are mutable objects — copied by reference, a view and its
    #: run would hand each other each other's values, and nothing would fail.
    _CACHES = (
        "_fit",
        "_projections",
        "_instances",
        "_results",
        "_pools",
        "_timelines",
        "_clip_ranks",
    )
    #: ``_node_view`` too: a view takes its own state's clip edits into it
    #: (``NodeCacheMixin._sync_clip_edits``), so sharing the run's would show one's edits on the
    #: other.
    _NOT_INHERITED = ("editable", "_node_view", "_clip_edit_stamp", *_CACHES)

    def __init__(self, run, snapshot_id: str, record: dict) -> None:
        self.__dict__.update(run.__dict__)
        # `editable` back to the class attribute, and the caches to fresh containers of whatever
        # shape the run keeps — never the run's own objects.
        for name in self._NOT_INHERITED:
            self.__dict__.pop(name, None)
        for name in self._CACHES:
            if name in run.__dict__:
                self.__dict__[name] = (
                    {} if isinstance(run.__dict__[name], dict) else None
                )
        self._snapshot_id = str(snapshot_id)
        self._snapshot_record = dict(record)
        self._snapshot_state_data = dict(record.get("state") or {})

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    def snapshot_record(self) -> dict:
        """The ``snapshot.yaml`` behind this view — when it was taken, why, and its code sha."""
        return dict(self._snapshot_record)

    def _read_state(self):
        return self._state_class.from_dict(self._snapshot_state_data)

    def _write_state(self, state) -> None:
        self._refuse_if_read_only("write its state")

    def _audit_dir(self):
        """The snapshot's own ``audit.csv``, not the run's."""
        return snapshots_store.snapshot_dir(self._run_dir(), self._snapshot_id)

    def artifact(self):
        """The frozen §19 artifact, read rather than re-derived.

        Re-deriving it from the recorded state under later code is exactly what makes an old
        result unreproducible (§21) — and it would also hide a snapshot whose state no longer
        reproduces its own instances, which is a thing worth being able to notice.
        """
        return behavior_output.read_behavior_output(
            snapshots_store.snapshot_dir(self._run_dir(), self._snapshot_id)
        )
