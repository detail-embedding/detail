"""Pinned copies of an authored run's §19 artifact: ``analysis/<run_id>/snapshots/<id>/`` (§21).

An authored run's artifact is **reprojected on every edit** — that is what makes granularity and
naming cheap enough to iterate on. The cost is that ``run_id`` names a mutable slot rather than a
result: a comparison that read ``an_15`` last week and one that reads it today can disagree, with
nothing anywhere failing. ``reopen()`` is a supported workflow (§21 calls finalize "a statement,
not a lock"), so this is not a case to forbid; it is one to record.

A **snapshot** is a frozen copy of the artifact, taken when a number *leaves* — an export, a
comparison, a finalize. Editing never takes one. Nor does displaying: the BROWSE tables and the
CLI status read the working artifact through ``artifact()`` and must stay free.

**Minted by difference, not by action.** ``pin`` hashes the artifact and returns the existing
snapshot when nothing has changed, so exporting the same run twice yields one snapshot rather than
two. That is what keeps a directory of pinned results from becoming the ``final_v2_REAL_final``
pile that copy-on-every-action produces.

**The artifact is copied, not re-derived.** Storing only the authored state and rebuilding on
demand would be far smaller, and it would mean a later code change silently yields different
numbers for an old result — the exact failure this exists to prevent, and one this repo has already
shipped twice (the label-leakage fix, the DTW pair-count fix). What is *not* copied is the
expensive pass: the fit, the distance matrix, the zarrs all stay where they are, which is where the
size actually lives.

**``snapshot.yaml`` is written last**, so it is the commit point: a directory without one is a
half-written snapshot and discovery skips it, the same rule ``storage/comparisons.py`` uses.

Layering: this module takes ``code_sha`` and ``state`` as arguments rather than reaching for
``analysis/provenance.py``, so ``storage`` keeps its one-way dependency (the note in
``storage/integrity.py`` states the same rule for fingerprints).
"""

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.storage import behavior_output
from detail.storage import clip_review
from detail.storage import dirscan
from detail.storage import frame_view
from detail.storage.integrity import sha256_file

SCHEMA_VERSION = 1
SNAPSHOTS_DIRNAME = "snapshots"
SNAPSHOT_FILENAME = "snapshot.yaml"

#: Why a snapshot was taken. Not free text: a snapshot nobody can explain is clutter, and the
#: reason is what makes a list of them readable.
#:
#: **Which** comparison used one is deliberately *not* stored here. An export has no stable
#: identifier to record (its path is outside the experiment, and an absolute path in an experiment
#: file is a hard-rule violation), and a comparison id does not exist until after the pin. So the
#: link is derived by scanning ``comparisons/`` for ``source_snapshot`` — a stored copy of a fact
#: checkable from the store is the second writer §21 exists to prevent.
REASONS = frozenset({"comparison", "export", "finish"})

#: Copied into a snapshot. The audit comes along because its decisions are evidence about *this*
#: artifact state — precision computed against a later reprojection is a different claim (§19).
_COPIED = (
    behavior_output.INSTANCES_FILENAME,
    behavior_output.COVERAGE_FILENAME,
    behavior_output.MANIFEST_FILENAME,
    behavior_output.AUDIT_FILENAME,
    behavior_output.AUDIT_BINS_FILENAME,
    clip_review.REVIEW_FILENAME,
)

#: The §19 frame view, copied as a directory so a snapshot can be **measured** and not merely read.
#: It is §23's per-window denominator and it is *not* regenerable from the artifact:
#: ``pipeline/behavior_artifact.py::write_frame_views`` needs ``analyzed_t_ns``, which comes from
#: the fit rather than from ``instances.csv``. It is small beside a fit that is never
#: copied, so self-containment is affordable. Absent for clustering runs,
#: which decline the frame view (§22) so a windowed measure refuses rather than dividing by the
#: wrong denominator.
_COPIED_DIRS = (behavior_output.FRAMES_DIRNAME,)

#: Hashed to decide whether the artifact changed. The audit and the frame view are deliberately
#: **excluded**: a human decision does not change what the instances are, and the frame view is a
#: materialized view of them (§19 — "if the two ever disagree, regenerate"), so hashing either adds
#: nothing and risks minting a snapshot for a rewrite that changed no instance.
_HASHED = (
    behavior_output.INSTANCES_FILENAME,
    behavior_output.COVERAGE_FILENAME,
    behavior_output.MANIFEST_FILENAME,
)


def snapshots_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / SNAPSHOTS_DIRNAME


def snapshot_dir(run_dir: Path | str, snapshot_id: str) -> Path:
    return snapshots_dir(run_dir) / snapshot_id


def discover_snapshots(run_dir: Path | str) -> list[str]:
    """Sorted ``snap_NN`` ids, from the folders that actually carry a record.

    A directory without ``snapshot.yaml`` is a half-written snapshot, not a snapshot — skipped
    rather than surfaced as an id that fails to read.
    """
    return dirscan.discover_ids(snapshots_dir(run_dir), SNAPSHOT_FILENAME)


def next_snapshot_id(run_dir: Path | str) -> str:
    """Run-monotonic ``snap_NN``, one past the highest existing.

    Scan-based like every other allocator in this package, so there is no counter to drift from
    what is on disk.
    """
    nums = [
        int(sid.split("_", 1)[1])
        for sid in discover_snapshots(run_dir)
        if sid.startswith("snap_") and sid.split("_", 1)[1].isdigit()
    ]
    return f"snap_{max(nums, default=0) + 1:02d}"


def artifact_hash(run_dir: Path | str) -> str | None:
    """A content hash of the working artifact, or ``None`` when there is no artifact.

    Composed from the per-file hashes rather than hashing a concatenation, the way
    ``provenance.labels_hash`` already composes: the file names go into the digest too, so adding
    a file changes the hash even if its content matches an existing one.
    """
    rdir = Path(run_dir)
    if not behavior_output.has_behavior_output(rdir):
        return None
    digest = hashlib.sha256()
    for name in _HASHED:
        path = rdir / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def latest_snapshot(run_dir: Path | str) -> dict | None:
    """The most recent snapshot's record, or ``None`` when the run has none."""
    ids = discover_snapshots(run_dir)
    if not ids:
        return None
    return read_snapshot(run_dir, ids[-1])


def read_snapshot(run_dir: Path | str, snapshot_id: str) -> dict:
    path = snapshot_dir(run_dir, snapshot_id) / SNAPSHOT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {SNAPSHOT_FILENAME} in {path.parent}")
    record = yaml_load.safe_load(atomic.read_text(path))
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version is {record.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    return record


def pin(
    run_dir: Path | str,
    *,
    reason: str,
    state: dict | None = None,
    code_sha: str | None = None,
) -> str | None:
    """Freeze the working artifact, or return the existing snapshot that already matches it.

    Args:
        reason: One of :data:`REASONS` — why the artifact is leaving.
        state: The authored decisions verbatim, so a reader can see what produced these numbers
            without re-deriving them.
        code_sha: The code version, because re-deriving under different code is what makes an old
            result unreproducible.

    Returns:
        The snapshot id, or ``None`` when the run has no artifact to pin. A run whose granularity
        is unset has none (§21), and finalizing it is legal — so this is a real state, not an
        error.

    Raises:
        ValueError: If ``reason`` is not one of :data:`REASONS`.
        FrameViewsPendingError: An edit left the frame views out of date. A snapshot copies them
            verbatim and is never rewritten, so pinning now would freeze a stale view for good.
    """
    if reason not in REASONS:
        raise ValueError(
            f"snapshot reason must be one of {sorted(REASONS)}, got {reason!r}"
        )
    rdir = Path(run_dir)
    frame_view.refuse_if_pending(rdir)
    content_hash = artifact_hash(rdir)
    if content_hash is None:
        return None

    previous = latest_snapshot(rdir)
    if previous is not None and previous.get("content_hash") == content_hash:
        # Mint by difference: the artifact has not moved since the last pin, so this *is* that
        # snapshot. Writing a second one would name the same numbers twice.
        return str(previous["snapshot_id"])

    snapshot_id = next_snapshot_id(rdir)
    sdir = snapshot_dir(rdir, snapshot_id)
    sdir.mkdir(parents=True, exist_ok=True)
    for name in _COPIED:
        source = rdir / name
        if source.is_file():
            shutil.copyfile(source, sdir / name)
    for name in _COPIED_DIRS:
        source = rdir / name
        if source.is_dir():
            shutil.copytree(source, sdir / name, dirs_exist_ok=True)

    record = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "parent": None if previous is None else previous.get("snapshot_id"),
        "content_hash": content_hash,
        "code_sha": code_sha,
        "state": dict(state or {}),
    }
    path = sdir / SNAPSHOT_FILENAME
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(record, handle, sort_keys=False)
    # Written last, and atomically: the record is the commit point, so a crash mid-copy leaves a
    # directory `discover_snapshots` ignores rather than a snapshot that lies about its contents.
    atomic.replace(tmp, path)
    return snapshot_id


MAX_VOCABULARY = 4


def vocabulary(names: dict[str, str], limit: int = MAX_VOCABULARY) -> str:
    """The names themselves, deduplicated and in order, cut with a count past ``limit``.

    Deduplicated because §22 makes many-to-one naming the normal case — a vocabulary that repeated
    "rearing" four times would be noise. This is what a person actually remembers a snapshot by, so
    both surfaces show it rather than a version number.
    """
    seen: list[str] = []
    for name in names.values():
        if name not in seen:
            seen.append(name)
    if not seen:
        return "unnamed"
    if len(seen) <= limit:
        return ", ".join(seen)
    return ", ".join(seen[:limit]) + f" +{len(seen) - limit} more"
