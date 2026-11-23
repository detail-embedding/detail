"""``query.yaml`` — the authored state of a Query run (§20/§21).

A Query run has two halves. **Search** is computed, expensive and immutable: a pure function of
(exemplars, method, params, embeddings), stored as opaque per-unit shards under ``search/``.
**Calibration** is authored, cheap and instant: the cutoff, the behavior names, the audit
labels. This file is the authored half.

What lives here, and what deliberately does not:

- **Here**: the behaviors (id, name, example clips, non-example clips), the method and its
  resolved params, and each behavior's cutoff.
- **Not here**: the lifecycle status, which lives in ``analysis.yaml`` and nowhere else; the
  human audit decisions, which are appended to ``audit.csv`` rather than rewritten; and any
  count, which is derived from those two. §21's one-writer invariant — dual writers on one fact
  drift silently, because both files stay well-formed while disagreeing.

The **behavior id is minted here and never derived from the name** (``q01``, ``q02``, …). It is
the artifact's ``source_label``, and §19 hashes it into every ``instance_id``, which audit
decisions are keyed by. A name is a thing a user renames; an id that moved when they did would
silently re-point every recorded judgment at a different behavior.

Written through ``storage/state_yaml.py``: atomic rewrite, ``schema_version`` first.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only — this module stays numpy-free at import
    import numpy as np

from detail.core import atomic
from detail.storage import state_yaml

QUERY_FILENAME = "query.yaml"
SEARCH_DIRNAME = "search"
HISTORY_FILENAME = "history.jsonl"
BACKGROUND_FILENAME = "background.npz"


def query_yaml_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / QUERY_FILENAME


def search_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / SEARCH_DIRNAME


def background_path(run_dir: Path | str) -> Path:
    """Where a run's drawn background sample lives (§20).

    Not in ``query.yaml`` and not in the job envelope: a few thousand rows at 768-d is tens of
    megabytes, which Tenet 4 keeps out of ``job.json``, and it is *computed*, not authored —
    ``query.yaml`` is the authored half.
    """
    return Path(run_dir) / BACKGROUND_FILENAME


@dataclass(frozen=True)
class Background:
    """One run's background sample: the vectors, and where each row came from (§20).

    §20 requires a draw to be recorded "identically (strategy, scope, size, seed, **and the
    rows drawn**)". The rows are what makes it auditable rather than merely reproducible: a
    seed reproduces a draw only against an unchanged trajectory, so the addresses are what
    lets a reader check *which* frames calibrated a confidence months later.

    ``vectors`` is what the method receives. The three parallel address columns are the same
    length and are never consulted by the search — they exist for the record.
    """

    vectors: "np.ndarray"  # (N, D) float32
    recording_ids: list[str]
    subjects: list[int]
    rows: list[int]
    strategy: str
    scope: str
    size: int
    seed: int

    def __len__(self) -> int:
        return len(self.vectors)


def write_background(run_dir: Path | str, sample: Background) -> Path:
    """Write the drawn sample, compressed. Atomic — the runner may be re-entered on resume."""
    import numpy as np

    path = background_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            vectors=sample.vectors,
            recording_ids=np.array(sample.recording_ids, dtype="str"),
            subjects=np.array(sample.subjects, dtype=np.int64),
            rows=np.array(sample.rows, dtype=np.int64),
            strategy=np.array(sample.strategy, dtype="str"),
            scope=np.array(sample.scope, dtype="str"),
            size=np.int64(sample.size),
            seed=np.int64(sample.seed),
        )
    atomic.replace(tmp, path)
    return path


def read_background(run_dir: Path | str) -> Background | None:
    """The stored sample, or ``None`` if this run has none (its method asked for none)."""
    import numpy as np

    path = background_path(run_dir)
    if not path.is_file():
        return None
    with np.load(path) as data:
        return Background(
            vectors=data["vectors"],
            recording_ids=[str(x) for x in data["recording_ids"]],
            subjects=[int(x) for x in data["subjects"]],
            rows=[int(x) for x in data["rows"]],
            strategy=str(data["strategy"]),
            scope=str(data["scope"]),
            size=int(data["size"]),
            seed=int(data["seed"]),
        )


def behavior_id(index: int) -> str:
    """``q01`` for the first behavior, ``q02`` for the second, …

    Zero-padded to two digits and minted from position, matching the ``source_label`` shape
    §19's manifest example uses. Never derived from the name — see the module docstring.
    """
    return f"q{index + 1:02d}"


@dataclass
class Behavior:
    """One behavior a run searches for: its id, its name, and the clips that define it.

    ``cutoff`` is the authored half and is ``None`` until the user sets one — which is not the
    same as zero. A run whose cutoff is unset has been searched but not calibrated, and its
    artifact says so by carrying no instances for that behavior rather than every candidate the
    method could find.

    ``removed`` is the clips a reviewer took out of this behavior (§20 *The audit*): one
    ``{recording_id, subject, t_ns, end_t_ns, at}`` per clip, ``at`` ISO-8601 UTC. It edits the
    output, never an input: the search, the examples and the non-examples are untouched, and the
    artifact leaves these candidates out at every cutoff. Kept per behavior because one span can be
    a candidate of two behaviors. A state written before the field reads as an empty list.
    """

    id: str
    name: str
    examples: list[dict] = field(default_factory=list)
    negatives: list[dict] = field(default_factory=list)
    cutoff: float | None = None
    removed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "examples": [dict(e) for e in self.examples],
            "negatives": [dict(n) for n in self.negatives],
            "cutoff": self.cutoff,
            "removed": [dict(r) for r in self.removed],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Behavior":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            examples=[dict(e) for e in (data.get("examples") or [])],
            negatives=[dict(n) for n in (data.get("negatives") or [])],
            cutoff=(None if data.get("cutoff") is None else float(data["cutoff"])),
            removed=[dict(r) for r in (data.get("removed") or [])],
        )

    def removed_keys(self) -> set[tuple[str, int, int, int]]:
        """``(recording_id, subject, t_ns, end_t_ns)`` of every removed clip — ``ClipRef.key``."""
        return {removal_key(r) for r in self.removed}


def removal_key(entry: dict) -> tuple[str, int, int, int]:
    """The ``ClipRef.key`` of one stored removal (``Behavior.removed``)."""
    return (
        str(entry["recording_id"]),
        int(entry.get("subject") or 0),
        int(entry["t_ns"]),
        int(entry["end_t_ns"]),
    )


@dataclass
class QueryState:
    """The whole authored state of one Query run."""

    embed_run_id: str
    track_id: int
    method: str
    method_params: dict = field(default_factory=dict)
    behaviors: list[Behavior] = field(default_factory=list)
    search_recording_ids: list[str] = field(default_factory=list)

    def behavior(self, behavior_id: str) -> Behavior:
        """The behavior with this id.

        Raises:
            KeyError: If no behavior carries it, naming the ones that do.
        """
        for b in self.behaviors:
            if b.id == behavior_id:
                return b
        raise KeyError(
            f"no behavior {behavior_id!r} in this run; it has "
            f"{[b.id for b in self.behaviors]}"
        )

    def to_dict(self) -> dict:
        return {
            "search_recording_ids": list(self.search_recording_ids),
            "embed_run_id": self.embed_run_id,
            "track_id": self.track_id,
            "method": self.method,
            "method_params": dict(self.method_params),
            "behaviors": [b.to_dict() for b in self.behaviors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QueryState":
        return cls(
            embed_run_id=str(data["embed_run_id"]),
            track_id=int(data.get("track_id", 0)),
            method=str(data["method"]),
            method_params=dict(data.get("method_params") or {}),
            behaviors=[Behavior.from_dict(b) for b in (data.get("behaviors") or [])],
            search_recording_ids=list(data.get("search_recording_ids") or []),
        )


def clip_key(row) -> tuple[str, int, int, int]:
    """``(recording_id, subject, t_ns, end_t_ns)`` of an instance row — the ``ClipRef.key`` a
    removal (``Behavior.removed``) and a review answer (``clip_review.csv``) are keyed by.
    """
    return (
        str(row.recording_id),
        int(row.subject or 0),
        int(row.t_ns),
        int(row.end_t_ns),
    )


def example_instance_ids(behaviors, instances) -> set[str]:
    """The ``instance_id``s of ``instances`` that overlap one of their own behavior's clips.

    A behavior's example and non-example clips are what the user *told* the method; a candidate
    on top of one is the method finding its own input. §19 makes such an instance ineligible
    for the audit — judging it scores the answer key — and this is the one place that decides
    which instances those are, for the draw, every estimate and the ``is_example`` column alike.

    An instance is matched against **its own** behavior's clips (``source_label`` is the
    behavior id), on the same recording and subject, by any shared time
    (:func:`~detail.storage.behavior_output.spans_overlap`).
    """
    from detail.storage import behavior_output

    clips: dict[tuple[str, str, int], list[tuple[int, int]]] = {}
    for behavior in behaviors:
        for spec in [*behavior.examples, *behavior.negatives]:
            key = (
                str(behavior.id),
                str(spec["recording_id"]),
                int(spec.get("track_id", 0)),
            )
            clips.setdefault(key, []).append(
                (int(spec["start_t_ns"]), int(spec["end_t_ns"]))
            )
    out: set[str] = set()
    for row in instances:
        subject = 0 if row.subject is None else int(row.subject)
        spans = clips.get((str(row.source_label), str(row.recording_id), subject), ())
        if any(
            behavior_output.spans_overlap((row.t_ns, row.end_t_ns), span)
            for span in spans
        ):
            out.add(row.instance_id)
    return out


def write_query_state(run_dir: Path | str, state: QueryState) -> Path:
    """Write ``query.yaml`` atomically. No status, no counters (§21)."""
    return state_yaml.write_state_yaml(query_yaml_path(run_dir), state.to_dict())


def read_query_state(run_dir: Path | str) -> QueryState:
    """Read ``query.yaml``.

    Raises:
        FileNotFoundError: If the run carries no authored state.
    """
    return QueryState.from_dict(state_yaml.read_state_yaml(query_yaml_path(run_dir)))


def has_query_state(run_dir: Path | str) -> bool:
    return query_yaml_path(run_dir).is_file()


def shard_path(
    run_dir: Path | str,
    behavior_id: str,
    recording_id: str,
    subject: int,
    suffix: str,
    chunk: int | None = None,
) -> Path:
    """``search/<behavior>/<recording>__<subject>[__chunk_KK]<ext>`` (§20's storage tree).

    One shard per search unit, which is what resume keys off: a completed shard is never
    recomputed. The extension is the **method's** — it owns serialization, so the module
    stores whatever it hands back without looking inside.
    """
    stem = f"{recording_id}__{subject}"
    if chunk is not None:
        stem = f"{stem}__chunk_{chunk:02d}"
    return search_dir(run_dir) / behavior_id / f"{stem}{suffix}"
