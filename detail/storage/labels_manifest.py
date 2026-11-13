"""Behavior-label batch manifest — map existing recordings to label files (Section 3).

A labels manifest is a small YAML that adds behavior ground-truth labels to recordings
that already have video:

    schema_version: 1
    labels:
      recording_001: labels/s1.csv
      recording_002: labels/s2.csv

Paths resolve relative to the manifest's folder (or absolute). Validation is upfront
and exhaustive (Tenet 10): every recording must exist and every file must resolve before
anything is imported. Distinct from the bundle manifest (``storage/manifest.py``),
which *creates* recordings; this one *adds to* existing ones.
"""

from pathlib import Path


from detail.core import yaml_load

SCHEMA_VERSION = 1


def load_labels_manifest(path: Path | str) -> dict:
    """Parse a labels manifest YAML.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If it is not a ``schema_version: 1`` mapping with a ``labels`` map.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labels manifest not found: {path}")
    with open(path, encoding="utf-8-sig") as f:
        data = yaml_load.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("labels"), dict):
        raise ValueError("labels manifest must be a mapping with a 'labels' mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"labels manifest schema_version must be {SCHEMA_VERSION}, got "
            f"{data.get('schema_version')!r}"
        )
    return data


def resolve_labels_manifest(
    manifest_path: Path | str, manifest: dict, existing_ids: list[str]
) -> list[tuple[str, Path, list[str] | None]]:
    """Resolve + validate a labels manifest into ``[(recording_id, file, covered), ...]``.

    Each ``labels`` value is either a filename, or a mapping ``{file,
    covered_behaviors}`` that also declares the recording's scored-behavior vocabulary
    (§3). Every problem is collected and reported together (Tenet 10): a recording id
    not in the experiment, a value missing its ``file``, or a file that does not resolve.

    Raises:
        ValueError: If any recording is unknown, malformed, or any file is missing.
    """
    manifest_dir = Path(manifest_path).parent
    known = set(existing_ids)
    mapping: list[tuple[str, Path, list[str] | None]] = []
    errors: list[str] = []
    for sid, entry in manifest["labels"].items():
        if isinstance(entry, dict):
            file_str = entry.get("file")
            covered = entry.get("covered_behaviors")
        else:
            file_str, covered = entry, None
        if sid not in known:
            errors.append(f"recording {sid!r} is not in the experiment")
        if not file_str:
            errors.append(f"labels[{sid!r}] needs a file (a filename or {{file: ...}})")
            continue
        resolved = manifest_dir / file_str
        if not resolved.is_file():
            errors.append(f"label file for {sid!r} not found: {resolved}")
        mapping.append((sid, resolved, covered))
    if errors:
        raise ValueError("labels manifest is invalid:\n  " + "\n  ".join(errors))
    return mapping
