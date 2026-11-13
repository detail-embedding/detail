"""Free space where a path really lives, and the estimate a processing run is checked against.

**The WSL trap (#743).** Inside WSL, an experiment under the distribution's own filesystem lives
in an ext4 virtual disk (``ext4.vhdx``) that is one file on a Windows drive. ``statvfs`` on the
path reports the *virtual* disk's free space — its own ceiling, 1 TB by default — so on a box
whose ``C:`` had 98 GB free it said 789 GB, and a remote embedding batch filled ``C:`` to 17 MB
and crashed WSL with the run's coordinator inside it. The space that bounds the write is the
smaller of the two: the virtual disk's, and the Windows drive's that holds it.

Everything here is a pure read: no network, no writes. Asking Windows where the virtual disk is
costs one ``powershell.exe`` call (about a second), made once per process and only on WSL, and
only for a path on the distribution's own filesystem (a ``/mnt/<drive>`` path is the Windows
drive itself, and ``statvfs`` already answers for it).
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The one wording of ``--ignore-free-space``, for every CLI command that launches a run.
IGNORE_FREE_SPACE_HELP = (
    "Launch even when the run's estimated output does not fit in the free space where the "
    "experiment is (it is refused otherwise). The estimate is printed either way."
)

#: Room left on top of a run's own estimate. The estimate is of the run's output alone; the OS,
#: WSL's virtual disk and everything else writing to the drive need space too, and a drive run
#: to its last megabyte is what crashed WSL in #743.
HEADROOM_BYTES = 5 * 1024**3


def format_size(n_bytes: float) -> str:
    """``n_bytes`` for a person: ``"1.2 GB"``, ``"340 MB"``, ``"12 KB"``."""
    for unit, scale in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {unit}"
    return f"{max(n_bytes, 0) / 1024:.0f} KB"


@dataclass(frozen=True)
class SpaceEstimate:
    """What a run will write, and the room it has (#743).

    ``where`` names the drive the free space was read from — the Windows drive behind a WSL
    virtual disk, when that is the tighter one — so a refusal says which drive to clear.
    """

    bytes: int
    free_bytes: int
    where: str
    per_recording: dict[str, int]

    @property
    def fits(self) -> bool:
        return self.bytes + HEADROOM_BYTES <= self.free_bytes

    def describe(self) -> str:
        return (
            f"this run will write about {format_size(self.bytes)}; "
            f"{format_size(self.free_bytes)} is free on {self.where}"
        )


class InsufficientSpaceError(ValueError):
    """A run's estimated output does not fit where the experiment is (#743)."""

    def __init__(self, estimate: SpaceEstimate) -> None:
        self.estimate = estimate
        super().__init__(
            f"{estimate.describe()}, and {format_size(HEADROOM_BYTES)} is kept spare. Free "
            "space there, run fewer recordings at once, or launch anyway (CLI "
            "--ignore-free-space)"
        )


def free_space(path: Path | str) -> tuple[int, str]:
    """``(free_bytes, where)`` for the drive that really bounds a write under ``path``.

    ``path`` need not exist yet: the nearest existing parent is asked.
    """
    path = _nearest_existing(Path(path))
    free = shutil.disk_usage(path).free
    where = str(path)
    host = _wsl_host_drive(path)
    if host is not None:
        host_free = shutil.disk_usage(host[0]).free
        if host_free < free:
            free, where = host_free, f"{host[1]} (the Windows drive holding WSL's disk)"
    return free, where


def frames_to_process(recording_data: dict, every_n_frames: int | None) -> int:
    """The frames a stage reads in one recording: the timestamp-map count recorded at import
    (``video.files[].frame_count``, Tenet 8), thinned by its stride."""
    files = (recording_data.get("video") or {}).get("files") or []
    total = sum(int(f.get("frame_count") or 0) for f in files)
    return -(-total // max(int(every_n_frames or 1), 1))


def dir_bytes(path: Path | str) -> int:
    """Bytes of every file under ``path`` (0 when it does not exist): what a stage that
    rewrites an existing run, rather than a video, will write about as much of."""
    path = Path(path)
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _nearest_existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _is_wsl() -> bool:
    try:
        return (
            "microsoft"
            in Path("/proc/version")
            .read_text(encoding="utf-8-sig", errors="replace")
            .lower()
        )
    except OSError:
        return False


def _wsl_host_drive(path: Path) -> tuple[str, str] | None:
    """``(mount, label)`` of the Windows drive holding WSL's virtual disk, e.g.
    ``("/mnt/c", "C:")`` — when ``path`` lives on that virtual disk, else ``None``."""
    if os.name != "posix" or not _is_wsl():
        return None
    try:
        on_distro_disk = os.stat(path).st_dev == os.stat("/").st_dev
    except OSError:
        return None
    if not on_distro_disk:
        return None
    letter = _vhdx_drive_letter()
    if letter is None:
        return None
    mount = f"/mnt/{letter.lower()}"
    return (mount, f"{letter.upper()}:") if Path(mount).is_dir() else None


@functools.lru_cache(maxsize=1)
def _vhdx_drive_letter() -> str | None:
    """The drive letter of this distribution's ``BasePath`` (where its ``ext4.vhdx`` is), read
    from the per-user ``Lxss`` registry key. ``None`` when Windows cannot be asked."""
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        return None
    script = (
        "Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | "
        "ForEach-Object { Get-ItemProperty $_.PSPath } | "
        f"Where-Object {{ $_.DistributionName -eq '{distro}' }} | "
        "ForEach-Object { $_.BasePath }"
    )
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return drive_letter(out)


def drive_letter(windows_path: str) -> str | None:
    """``"C"`` for ``C:\\…`` or ``\\\\?\\C:\\…``; ``None`` for anything else (a UNC share, an
    empty answer, more than one line)."""
    base = windows_path.strip().removeprefix("\\\\?\\")
    if "\n" in base or len(base) < 2 or base[1] != ":" or not base[0].isalpha():
        return None
    return base[0]
