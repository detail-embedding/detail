"""Referenced-video integrity checks (Section 2) — the shared truth for the pre-run gate.

A reference-in-place video lives in user-owned storage, so the tool re-earns trust on every read
by comparing the file against what it recorded when the video was accepted: the file must still be
where the ``{root, rel}`` reference resolves, and its content fingerprint must still match. This
module is the single source both the hard pre-run gate (``module.validate``) and the surfaces
(``recording verify``, the per-recording check) call — one place that decides "missing" vs
"changed" so they never disagree.

Copied parts (the tool owns the file) are checked for existence only. Reference parts add the
cheap ``file_fingerprint`` (size + head/tail hash) comparison; a mismatch is a *changed* video,
whose remedy is remove/re-add (a moved-but-unchanged file is a relink, not a change).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from detail.storage import integrity
from detail.storage import recording as recording_store
from detail.storage import video_roots


@dataclass(frozen=True)
class VideoIssue:
    """One thing wrong with a recording's video source.

    ``kind`` is ``missing_root`` (the reference's video-root alias isn't registered on this
    machine), ``missing_file`` (the resolved path doesn't exist), or ``changed_fingerprint`` (a
    referenced file's content differs from what was recorded).
    """

    kind: str
    part_id: str
    detail: str


def validate_source_videos(
    recording_dir: Path | str,
    recording_data: dict,
) -> list[VideoIssue]:
    """Every integrity problem with the parts a run would read (empty means all good).

    Resolves the same parts a stage reads via ``video_files`` + ``resolve_part_video_path``,
    so it checks exactly the files the worker will open. Cheap enough (a head/tail hash per
    referenced part) to run on every launch.
    """
    issues: list[VideoIssue] = []
    files = recording_store.video_files(recording_data)
    for part in files:
        stem = recording_store.part_stem(part)
        try:
            # Unverified: the comparison below is this gate's own, uncached and reported as
            # `changed_fingerprint` rather than as a missing root.
            path = recording_store.resolve_part_video_path(
                recording_dir, part, verify=False
            )
        except video_roots.MissingVideoRootError as exc:
            issues.append(VideoIssue("missing_root", stem, str(exc)))
            continue
        if not path.is_file():
            issues.append(
                VideoIssue("missing_file", stem, f"video not found at {path}")
            )
            continue
        recorded = part.get("fingerprint")
        if part.get("source") is not None and recorded is not None:
            current = integrity.file_fingerprint(path)
            if not integrity.same_content(current, recorded):
                issues.append(
                    VideoIssue(
                        "changed_fingerprint",
                        stem,
                        f"content changed since it was added ({path}) — remove and "
                        "re-add the recording to use the new file",
                    )
                )
    return issues


def raise_on_video_issues(issues: list[VideoIssue]) -> None:
    """Turn integrity issues into a single actionable ``ValueError`` for the pre-run gate."""
    if issues:
        detail = "\n  ".join(f"part {i.part_id}: {i.detail}" for i in issues)
        raise ValueError(f"video input is not usable:\n  {detail}")
