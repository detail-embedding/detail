"""Content fingerprints for files (Section 2) — the identity of a referenced video.

``file_fingerprint`` is a cheap size + head/tail hash that catches every realistic change to a
multi-GB video (re-encode, truncation, replacement) without reading the whole file;
``sha256_file`` is the full hash, for small inputs or an explicit deep verify. Storage-layer and
dependency-free (``hashlib`` + ``pathlib`` only), so the run path, recordings, and provenance can
all share one identity notion without a ``storage -> analysis`` import.
"""

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path | str) -> str:
    """Full SHA-256 of a file (small inputs like label CSVs, or an explicit deep verify)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path | str) -> dict:
    """A cheap fingerprint for large files (videos): size + mtime + head/tail hash.

    Full content hashing of a multi-GB video is avoided by design; the head/tail hash plus size
    catches any realistic change (re-encode, truncation, replacement). The change decision keys on
    ``size`` + ``sha256_ends``; ``mtime_ns`` is advisory (a harmless re-copy changes it).
    """
    p = Path(path)
    stat = p.stat()
    digest = hashlib.sha256()
    with open(p, "rb") as handle:
        digest.update(handle.read(_CHUNK))
        if stat.st_size > _CHUNK:
            handle.seek(max(0, stat.st_size - _CHUNK))
            digest.update(handle.read(_CHUNK))
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256_ends": digest.hexdigest(),
    }
