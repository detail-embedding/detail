"""Publication-grade provenance capture for a comparison.

Records what a result traces back to and lets drift be detected: the run lineage
(embedding -> tracking -> detection -> video) per member, a content hash of the ground-truth
labels, the source-video fingerprints, the §19 artifact that was read, the code version, and the
environment. ``drift`` compares a
recorded capture against a fresh one and names what changed (labels edited, video swapped, code
moved) so a stale result never re-exports silently.

Everything is best-effort and never raises for a missing optional (a git-less checkout, an
unresolved weights revision): provenance annotates, it does not gate the run.
"""

import hashlib
import platform
import sys
from pathlib import Path

from detail.core import buildinfo
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage import streams as streams_store
from detail.storage.integrity import (
    file_fingerprint,
    sha256_file,
)  # used below + re-exported


def environment() -> dict:
    """The export-time environment: Python, platform, and key package versions (no backend load)."""
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import numpy

        env["numpy"] = numpy.__version__
    except Exception:  # numpy absent is not a provenance failure
        pass
    try:
        from importlib.metadata import version

        env["detail"] = version("detail")
    except Exception:
        pass
    return env


def code_version() -> str | None:
    """Git SHA of the detail source tree, or ``None`` outside a git checkout."""
    return buildinfo.head_sha()


def labels_hash(recording_dir: Path | str) -> str:
    """Content hash of every ground-truth behavior stream's canonical CSV.

    Hashing all behavior streams (not just the queried behaviors) means any label edit changes the
    hash, which is what ``drift`` needs. **Predicted** streams are excluded (§19): re-ingesting a
    clustering run's ethogram would otherwise report the ground truth as having changed, and drift
    is only useful while it means what it says.
    """
    parts: list[str] = []
    for entry in sorted(
        (
            e
            for e in streams_store.streams_by_role(recording_dir, "behavior")
            if not streams_store.is_predicted(e)
        ),
        key=lambda e: e.get("name", ""),
    ):
        csv = (
            streams_store.stream_dir(recording_dir, entry["name"])
            / streams_store.CANONICAL_DIRNAME
            / streams_store.CANONICAL_FILENAME
        )
        if csv.is_file():
            parts.append(sha256_file(csv))
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _find(items, run_id: str | None) -> dict | None:
    for item in items or []:
        if item.get("id") == run_id:
            return item
    return None


def embedding_lineage(recording_record: dict, embed_run_id: str) -> dict:
    """Walk embedding -> tracking -> detection -> video in a ``recording.yaml`` record.

    Raises:
        ValueError: the embedding run is not in the recording record.
    """
    emb = _find(recording_record.get("embedding_runs"), embed_run_id)
    if emb is None:
        raise ValueError(f"embedding run {embed_run_id!r} not in recording record")
    track = _find(recording_record.get("tracking_runs"), emb.get("tracking_run_id"))
    det = _find(
        recording_record.get("detection_runs"),
        (track or {}).get("detection_run_id"),
    )
    video = recording_record.get("video", {})
    return {
        "embedding": {
            "id": emb["id"],
            "model_id": emb.get("model_id"),
            "model_variant": emb.get("model_variant"),
            "constructor": emb.get("constructor"),
            "readout": emb.get("readout"),
            "input_size": emb.get("input_size"),
        },
        "tracking": (
            None
            if track is None
            else {
                "id": track["id"],
                "tracker_id": track.get("tracker_id"),
                "detection_run_id": track.get("detection_run_id"),
            }
        ),
        "detection": (
            None
            if det is None
            else {
                "id": det["id"],
                "model_id": det.get("model_id"),
                "weights_source": det.get("weights_source"),
            }
        ),
        "video": [f.get("file") for f in video.get("files", [])],
    }


def capture(
    experiment_dir: Path | str,
    *,
    recording_ids: list[str],
    members: list[tuple[str, str]],
    source: dict | None = None,
) -> dict:
    """Full provenance for a comparison: code, environment, and per-recording lineage + hashes.

    ``members`` is ``[(label, embed_run_id), ...]``. No timestamp here (the comparison record owns
    ``created_at``), so the capture is deterministic given fixed inputs.

    ``source`` identifies the §19 artifact that was read — ``{run_id, snapshot_id, content_hash}``.
    Passed in rather than derived here so this stays the dumb recorder it is; the caller is the only
    one that knows *which* directory it measured. Omitted by callers that have no single source
    artifact (the removed benchmarks spanned many), and absent from every record written before it
    existed, which :func:`drift` treats as "nothing to say" rather than as a change.
    """
    recordings: dict = {}
    for sid in recording_ids:
        sdir = experiment_store.recording_path(experiment_dir, sid)
        record = recording_store.read_recording(sdir)
        recordings[sid] = {
            "labels_hash": labels_hash(sdir),
            "videos": {
                (f.get("file") or recording_store.part_stem(f)): file_fingerprint(
                    recording_store.resolve_part_video_path(sdir, f)
                )
                for f in record.get("video", {}).get("files", [])
            },
            "members": {
                label: embedding_lineage(record, embed_run_id)
                for label, embed_run_id in members
            },
        }
    captured = {
        "code_sha": code_version(),
        "environment": environment(),
        "recordings": recordings,
    }
    if source:
        captured["source"] = dict(source)
    return captured


def drift(recorded: dict, current: dict) -> list[str]:
    """Human-readable list of provenance items that changed (empty means no drift).

    Every item means "re-running now would not reproduce this", **not** "the stored numbers are
    wrong" — a moved code SHA does not corrupt a `stats.csv` that is already computed, and neither
    does a re-cut source analysis, whose numbers stay reachable through the snapshot they were
    measured from (§21). The wording of each line has to carry that, because a warning read as
    "your result is broken" invites someone to throw away a perfectly good one.
    """
    changes: list[str] = []
    recorded_source = recorded.get("source") or {}
    current_source = current.get("source") or {}
    if (
        recorded_source.get("content_hash")
        and current_source.get("content_hash")
        and recorded_source["content_hash"] != current_source["content_hash"]
    ):
        # Named with the snapshot it was measured from, so a reader can tell the saved result is
        # still exactly reproducible and only the *current* interpretation has moved.
        pinned = recorded_source.get("snapshot_id")
        run = recorded_source.get("run_id", "the source run")
        changes.append(
            f"source analysis {run} changed since this ran"
            + (
                f" (these numbers are {pinned}'s; a re-run would differ)"
                if pinned
                else ""
            )
        )
    if recorded.get("code_sha") != current.get("code_sha"):
        changes.append(
            f"code changed ({recorded.get('code_sha')} -> {current.get('code_sha')})"
        )
    for sid, rec in recorded.get("recordings", {}).items():
        cur = current.get("recordings", {}).get(sid, {})
        if rec.get("labels_hash") != cur.get("labels_hash"):
            changes.append(f"labels changed for recording {sid}")
        for video, fingerprint in rec.get("videos", {}).items():
            if cur.get("videos", {}).get(video) != fingerprint:
                changes.append(f"video {video} changed for recording {sid}")
    return changes
