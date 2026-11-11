"""The most memory a job used — the column a timing table cannot answer on its own.

A run that takes an hour on one machine and fails on another usually fails on memory, and the
question a reader of a performance report asks next is *"how close to the limit was it?"*.
Nothing recorded that, so this samples it while the job runs.

**RAM is the whole process tree, summed.** The job's own process is not where all of it lives:
Ultralytics training starts DataLoader worker processes, and a multi-GPU detection or embedding
run spawns one pool process per GPU. So every sample adds this process's resident memory to its
children's, and the peak is the largest such sum.

**GPU memory is read only from a torch this job already imported.** Importing torch costs
seconds and would break the lazy-import rule for every CPU-only job, so a job that never loaded
torch reports no GPU figure — blank, which reads as "not measured", never 0, which would read as
a measurement. CUDA keeps its own peak counters, read once at the end. MPS has no peak counter,
so it is sampled alongside RAM. A multi-GPU run does its GPU work in spawned children whose
counters this process cannot see, so it too reports blank rather than a misleading number.

Every probe is guarded: a sampler that raises would fail the job it is measuring.
"""

from __future__ import annotations

import sys
import threading

#: Seconds between samples. A peak shorter than this can be missed; one second is fine-grained
#: against jobs that run for minutes to hours, and the cost of a sample is a few syscalls.
SAMPLE_INTERVAL_S = 1.0


def _tree_rss_bytes(proc) -> int:
    """Resident memory of ``proc`` plus every descendant, or 0 if none could be read."""
    import psutil

    total = 0
    try:
        members = [proc, *proc.children(recursive=True)]
    except (psutil.Error, OSError):
        members = [proc]
    for member in members:
        try:
            total += member.memory_info().rss
        except (psutil.Error, OSError):
            continue  # a child that exited between listing and reading
    return total


def _mps_bytes() -> int | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.backends.mps.is_available():
            return None
        return int(torch.mps.driver_allocated_memory())
    except Exception:  # noqa: BLE001 — a probe must never fail the job
        return None


def _cuda_peaks() -> tuple[int, int] | None:
    """``(reserved, allocated)`` peak bytes summed over devices, if this process used CUDA."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        devices = range(torch.cuda.device_count())
        reserved = sum(torch.cuda.max_memory_reserved(d) for d in devices)
        allocated = sum(torch.cuda.max_memory_allocated(d) for d in devices)
    except Exception:  # noqa: BLE001
        return None
    return int(reserved), int(allocated)


class PeakSampler:
    """Samples peak memory on a daemon thread between :meth:`start` and :meth:`stop`."""

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = None
        self.peak_rss_bytes = 0
        self.peak_mps_bytes: int | None = None

    def start(self) -> "PeakSampler":
        try:
            import psutil

            self._proc = psutil.Process()
        except (
            Exception
        ):  # noqa: BLE001 — no psutil means no RAM figure, not a failed job
            self._proc = None
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def _sample(self) -> None:
        if self._proc is not None:
            try:
                rss = _tree_rss_bytes(self._proc)
            except Exception:  # noqa: BLE001
                rss = 0
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        mps = _mps_bytes()
        if mps is not None:
            self.peak_mps_bytes = max(self.peak_mps_bytes or 0, mps)

    def stop(self) -> dict:
        """Stop sampling and return the peaks. Blank (``None``) means not measured."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self._interval_s)
        self._sample()
        cuda = _cuda_peaks()
        if cuda is not None:
            backend, reserved, allocated = "cuda", cuda[0], cuda[1]
        elif self.peak_mps_bytes:
            backend, reserved, allocated = "mps", self.peak_mps_bytes, None
        else:
            backend, reserved, allocated = None, None, None
        return {
            "peak_ram_bytes": self.peak_rss_bytes or None,
            "gpu_backend": backend,
            "peak_gpu_reserved_bytes": reserved,
            "peak_gpu_allocated_bytes": allocated,
            "sample_interval_s": self._interval_s,
        }
