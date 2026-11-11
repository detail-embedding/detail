"""``temporal_clustering.yaml`` — the authored state of a temporal-clustering run (§21/§22).

The sibling of ``storage/pose_runs.py``, and the same two halves. **The fit** is computed,
expensive and immutable — here that is the O(bouts²) DTW matrix, stored as an opaque shard under
``fit/``. **Interpretation** is authored, cheap and instant: the labels a user puts on nodes of the
fit's merge tree (§22). This file is the authored half.

Deliberately a separate module rather than a shared one parameterized by kind. The two states are
near-identical today and are free to diverge — a temporal run may come to carry a decode or an
alignment rule that means nothing to a posture — and §22's whole position is that the two modules
resemble each other closely enough that sharing is the trap.

**The cluster id comes from the method and is never derived from a name.** It is the artifact's
``source_label``, and §19 hashes it into every ``instance_id``. A name is a thing a user renames; an
id that moved when they did would silently re-point every recorded judgment at a different cluster.

**Many nodes may share one label.** A fit holds several variants of the same behavior in
different branches, and labelling each the same is how they become one behavior.

Written through ``storage/state_yaml.py``: atomic rewrite, ``schema_version`` first.
"""

from dataclasses import dataclass, field
from pathlib import Path

from detail.configs.clustering_legacy import drop_legacy_floor
from detail.core import run_selection
from detail.storage import clip_edits as clip_edits_store
from detail.storage import run_labels
from detail.storage import state_yaml

TEMPORAL_FILENAME = "temporal_clustering.yaml"
FIT_DIRNAME = "fit"
HISTORY_FILENAME = "history.jsonl"


def temporal_yaml_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / TEMPORAL_FILENAME


def fit_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / FIT_DIRNAME


def fit_path(run_dir: Path | str, suffix: str) -> Path:
    """Where the method's serialized fit lives.

    ``suffix`` is the method's declared ``shard_suffix``, not a guess: a serializer may append its
    own (``np.savez_compressed`` adds ``.npz``), and a caller that built the path without it would
    write one file and read another.
    """
    return fit_dir(run_dir) / f"fit{suffix}"


@dataclass
class TemporalClusteringState:
    """The whole authored state of one temporal-clustering run.

    ``granularity`` is a legacy field: the cut a run written before §22's node-centric model was
    read at. Nothing sets it now; a legacy run keeps it so its old names can be folded onto the
    nodes they were given at.

    **A label is a stored thing, and nodes of the merge tree point at it** (``labels``,
    ``node_labels``; ``storage/run_labels.py``). A member reports the label on its **deepest
    labelled ancestor** — so "immobility" given once on a coarse node reaches every member beneath
    it, and "rearing left" given deeper coexists with it rather than contradicting it. Several
    clusters joining one label is one label, not several matching strings: renaming it renames all
    of them, and the output treats their members as one behavior.

    ``resolved_names`` / ``resolved_paths`` are that resolution **precomputed**, keyed by label id.
    Derived, with one writer — every edit recomputes them — and kept because resolving needs the
    tree, while a snapshot record, this module and the artifact manifest all have to answer "what
    is this called" with no fit in hand. A state written before labels had ids keeps its per-node
    names and reads as one label per distinct name.

    ``names_by_granularity`` is the previous model, read-only. A state written under it keeps
    reporting its names, and the next edit folds them onto nodes and clears it.

    A cluster absent from the map is **unnamed**, which §22 distinguishes from noise: unnamed is
    work outstanding, noise is a method saying it put the bout nowhere. ``dtw_finch`` has no
    noise, so every cluster here is one or the other.
    """

    #: The single id when every recording's run shares it, else ``None``. Read a
    #: recording's run through :meth:`embed_run_for`, never this directly.
    embed_run_id: str | None
    track_id: int
    method: str
    method_params: dict = field(default_factory=dict)
    source_recording_ids: list[str] = field(default_factory=list)
    granularity: int | None = None
    #: label id -> name: the labels themselves (``storage/run_labels.py``). The authority, with
    #: ``node_labels``; a name is unique within the run, ignoring case.
    labels: dict[str, str] = field(default_factory=dict)
    #: node id -> label id: which label each labelled node of the merge tree carries.
    node_labels: dict[str, str] = field(default_factory=dict)
    #: label id -> when its clusters last changed (ISO-8601 UTC). A label edited after it was
    #: audited reports its precision from judgments made since (§19), so choosing clusters by
    #: how they scored cannot raise the score the choice was made on. A rename is not a change.
    membership_changed_at: dict[str, str] = field(default_factory=dict)
    #: label id -> name, for the labels that reach the output. Derived from ``labels`` and
    #: ``node_labels`` with the fit's tree, keyed the way ``instances.csv`` stores ``source_label``.
    resolved_names: dict[str, str] = field(default_factory=dict)
    #: label id -> ``"outer/inner"``. The **ancestry** of that name, which
    #: ``instances.csv`` writes as ``label_path`` so an exported bundle carries the hierarchy
    #: without the tree beside it. Derived alongside ``resolved_names`` and never separately: two
    #: derivations of one resolution is how a name and its path come to disagree.
    resolved_paths: dict[str, str] = field(default_factory=dict)
    #: The previous model, read-only, folded onto nodes by the next edit.
    names_by_granularity: dict[str, dict[str, str]] = field(default_factory=dict)
    #: recording id -> the embedding run the fit read there (§6.5). Empty on a run from
    #: before it existed, whose ``embed_run_id`` meant that id in every recording.
    embed_run_ids: dict[str, str] = field(default_factory=dict)
    #: Clips taken out of a label, or moved to another, after watching them
    #: (``storage/clip_edits.py``): ``[{recording_id, subject, t_ns, end_t_ns, from_label,
    #: to_label, at}]``, one per clip, the latest winning. An override on the derivation, never
    #: on the fit: the clip stays in its cluster.
    clip_edits: list[dict] = field(default_factory=list)

    def embed_run_for(self, recording_id: str) -> str:
        """``recording_id``'s own embedding run — ``embed_run_ids`` when recorded, else the
        single ``embed_run_id`` a run from before the map meant in every recording."""
        return run_selection.require_run_for_recording(self, recording_id)

    @property
    def names(self) -> dict[str, str]:
        """The labels in effect, keyed by labelled node id (the artifact's ``source_label``).

        The same shape every fit-free caller has always read — ``name_of``, ``named_count``, the
        artifact manifest — now answered from the resolution rather than from a per-setting map.
        """
        return self.resolved_names

    def name_of(self, cluster_id: int | str) -> str:
        """The user's name for a cluster, or ``""`` if nothing at or above it is named."""
        return self.names.get(str(cluster_id), "")

    def named_count(self) -> int:
        """How many nodes carry a label."""
        return len([name for name in self.names.values() if name])

    @property
    def node_names(self) -> dict[str, str]:
        """``{node: name}`` — the per-node view, derived from the labels. Read-only: a label is
        given, renamed or merged through the run's label operations, never by editing this.
        """
        return run_labels.node_names(self.labels, self.node_labels)

    def named_nodes(self) -> set[int]:
        """The node ids carrying a label — what ``ClusterTree.deepest_named`` resolves against."""
        return {int(node) for node in self.node_names}

    @staticmethod
    def _names_by_granularity(data: dict) -> dict[str, dict[str, str]]:
        """Read the per-granularity map, migrating a state written before it existed.

        A legacy state holds one flat ``names`` map belonging to whatever ``granularity`` it was
        last set to, so that is where it goes. Runs already on disk keep their names rather than
        appearing to have lost them the first time they are opened.
        """
        stored = data.get("names_by_granularity")
        if isinstance(stored, dict):
            return {
                str(g): {str(k): str(v) for k, v in (names or {}).items()}
                for g, names in stored.items()
            }
        legacy = {str(k): str(v) for k, v in (data.get("names") or {}).items()}
        if not legacy:
            return {}
        return {str(data.get("granularity")): legacy}

    @staticmethod
    def _resolved_names(data: dict) -> dict[str, str]:
        """The names in effect, migrating a state written before names moved onto nodes.

        Three generations, newest first: ``resolved_names``, then the per-granularity map at
        whatever setting the state is on, then the flat map that predates both. Fit-free by
        construction — ``from_dict`` has no tree — so a legacy state reports the names it always
        did, and folding them onto nodes waits for an edit, which does have one.
        """
        stored = data.get("resolved_names")
        # `if stored`, not `isinstance(..., dict)`: an **empty** map is what a state written before
        # names moved onto nodes round-trips as, and treating "present" as "authoritative" let it
        # shadow the legacy names the state still carries. A legitimately nameless state resolves
        # to nothing down either branch, so falling through costs nothing.
        if stored:
            return {str(k): str(v) for k, v in stored.items()}
        by_granularity = TemporalClusteringState._names_by_granularity(data)
        return dict(by_granularity.get(str(data.get("granularity"))) or {})

    def to_dict(self) -> dict:
        return {
            "source_recording_ids": list(self.source_recording_ids),
            "embed_run_id": self.embed_run_id,
            "embed_run_ids": dict(self.embed_run_ids),
            "track_id": self.track_id,
            "method": self.method,
            "method_params": dict(self.method_params),
            "granularity": self.granularity,
            # Keys coerced to str throughout: a cluster id is an int in the method and a string
            # in the CSV, and a granularity is an int in the widget, and YAML would otherwise
            # round-trip them as ints on one path and strings on the other — the shape
            # `behavior_output.names_map` already normalizes for the same reason.
            "labels": {str(k): str(v) for k, v in self.labels.items()},
            "node_labels": {str(k): str(v) for k, v in self.node_labels.items()},
            "membership_changed_at": {
                str(k): str(v) for k, v in self.membership_changed_at.items()
            },
            "resolved_names": {str(k): str(v) for k, v in self.resolved_names.items()},
            "resolved_paths": {str(k): str(v) for k, v in self.resolved_paths.items()},
            "names_by_granularity": {
                str(g): {str(k): str(v) for k, v in names.items()}
                for g, names in self.names_by_granularity.items()
            },
            "clip_edits": clip_edits_store.latest(self.clip_edits),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalClusteringState":
        data = drop_legacy_floor(data)  # a run saved with a size floor
        # Whichever generation wrote it: ids and pointers, or the per-node names before them.
        labels, node_labels = run_labels.read(data)
        return cls(
            embed_run_id=data.get("embed_run_id") or None,
            embed_run_ids={
                str(k): str(v) for k, v in (data.get("embed_run_ids") or {}).items()
            },
            track_id=int(data.get("track_id", 0)),
            method=str(data["method"]),
            method_params=dict(data.get("method_params") or {}),
            source_recording_ids=list(data.get("source_recording_ids") or []),
            granularity=(
                None if data.get("granularity") is None else int(data["granularity"])
            ),
            labels=labels,
            node_labels=node_labels,
            membership_changed_at={
                str(k): str(v)
                for k, v in (data.get("membership_changed_at") or {}).items()
            },
            resolved_names=cls._resolved_names(data),
            resolved_paths={
                str(k): str(v) for k, v in (data.get("resolved_paths") or {}).items()
            },
            names_by_granularity=(
                {} if node_labels else cls._names_by_granularity(data)
            ),
            clip_edits=clip_edits_store.latest(data.get("clip_edits") or []),
        )


def write_temporal_state(run_dir: Path | str, state: TemporalClusteringState) -> Path:
    """Rewrite the authored state atomically."""
    return state_yaml.write_state_yaml(temporal_yaml_path(run_dir), state.to_dict())


def read_temporal_state(run_dir: Path | str) -> TemporalClusteringState:
    """Read the authored state.

    Raises:
        FileNotFoundError: If the run has none — it was never fitted, or is not a
            temporal-clustering run.
    """
    return TemporalClusteringState.from_dict(read_state_dict(run_dir))


def read_state_dict(run_dir: Path | str) -> dict:
    """The raw record, for a reader that wants a field without the dataclass."""
    return state_yaml.read_state_yaml(temporal_yaml_path(run_dir))


def has_temporal_state(run_dir: Path | str) -> bool:
    return temporal_yaml_path(run_dir).exists()
