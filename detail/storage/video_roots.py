"""Machine-local registry of external video roots (Section 2).

Reference-in-place recordings store ``{root: <alias>, rel: <path>}`` in recording.yaml; the alias
resolves to an absolute directory recorded in ``~/.config/detail/storage.yaml``. So the experiment
file holds no machine path (Tenet 6 stays intact) and moving a whole video library is a single
re-point of the alias.

**The mechanism now lives in ``storage/roots.py``**, keyed by the ``video_roots`` section of that
file; this module is the video-shaped door onto it, and every caller and the ``video-root`` CLI
command are unchanged. It was generalised on its second user — an export run's destination is the
same problem (#354): a directory outside the experiment that an experiment file may not name.
"""

from pathlib import Path

from detail.storage import roots

#: The ``storage.yaml`` section this door owns. Sits alongside the existing ``storage.<alias>``
#: shape ``locations.py`` reads, without disturbing it.
VIDEO_ROOTS_KEY = "video_roots"


class MissingVideoRootError(roots.MissingRootError):
    """A referenced video-root alias is not registered on this machine.

    A subclass rather than an alias so the existing ``except MissingVideoRootError`` sites — the
    run gate and the integrity check — keep catching exactly what they caught before.
    """


class VideoNotRecordedError(MissingVideoRootError, FileNotFoundError):
    """The file a referenced part resolves to is not the video that was recorded for it.

    Its fingerprint differs: a different video at the same path, such as a root alias this
    machine already uses for another folder that holds a file of the same name, or a root
    re-pointed with ``video-root set``. Reading it shows another animal's pixels against this
    recording's timestamps, so it is treated as not reachable here, the same as a missing root
    or a missing file. Both bases are deliberate: every ``except MissingVideoRootError``
    presence gate (`Recording.video_available`) and every ``except FileNotFoundError`` frame
    source already degrades to "video not here" on them, so none of them reads the wrong file.
    """


def list_video_roots() -> dict[str, Path]:
    """Every registered alias -> absolute directory, from storage.yaml."""
    return roots.list_roots(VIDEO_ROOTS_KEY)


def video_root(alias: str) -> Path | None:
    """The directory an alias resolves to, or ``None`` if it isn't registered."""
    return roots.root(VIDEO_ROOTS_KEY, alias)


def register_video_root(alias: str, path: Path | str) -> None:
    """Register (or re-point) ``alias`` to ``path``, merged into storage.yaml atomically.

    Raises if the alias is empty or the path is not an existing directory — a typo or an
    unmounted drive should fail loudly at registration, not silently at run time.
    """
    roots.register_root(VIDEO_ROOTS_KEY, alias, path)


def resolve_video_ref(ref: dict) -> Path:
    """Resolve a recording's ``{root, rel}`` reference to an absolute path.

    Raises ``MissingVideoRootError`` when the alias isn't registered (the run gate and the
    integrity check turn this into an actionable "register the root / relink" message). Does not
    check that the file exists — that is the caller's presence check.
    """
    alias = ref.get("root")
    rel = ref.get("rel")
    if not alias or rel is None:
        raise ValueError(f"invalid video reference {ref!r}")
    # Resolved through this module's own `video_root`, not `roots.resolve_ref`: that keeps the
    # lookup seam callers already patch, and keeps the message naming the video-root command.
    found = video_root(alias)
    if found is None:
        raise MissingVideoRootError(
            f"video root {alias!r} is not registered on this machine; register it with "
            f"`detail video-root set {alias} <path>`"
        )
    return found / rel


def resolve_or_register_root(video_path: Path | str) -> str:
    """A video-root alias that contains ``video_path``, auto-registering the video's folder if
    none does — so a reference import needs no root from the user (Section 2).

    Reuses the **deepest** existing root that already holds the file (videos from the same tree
    then share a location); otherwise registers the video's parent directory under an auto alias.
    The alias is an internal detail — the user picks a video, not a location; this keeps the
    ``{root, rel}`` model (the seam a remote backend plugs into) without any location UI.
    """
    return roots.resolve_or_register(VIDEO_ROOTS_KEY, video_path)
