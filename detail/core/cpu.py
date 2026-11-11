"""How many threads a stage that waits on reads should run: one rule, written once.

**The class this closes.** ``max(1, min(8, (os.cpu_count() or 4) - 1))`` was the embedding
worker's render pool and the kinematics backfill's decode pool, each spelled out in its own
module, and the clip montage arrived with a third answer to the same question — ``WORKERS = 6``,
a constant that was right on the box it was measured on and nowhere else. CLAUDE.md: two copies
of a shape can be tolerated; the third is when it becomes one mechanism. This is that mechanism,
and ``tests/core/test_cpu.py`` keeps ``os.cpu_count()`` out of everywhere but here and the
few sites that answer a *different* question (listed there with their reasons).

The rule itself: every core but one, so the thread that feeds the pool and the UI thread that
waits on it keep a core; never more than ``cap``, because past eight the share the data is read
from, not the CPU, is what runs out; never fewer than one.

**Cores are the ones this process may run on**, not the machine's. On a SLURM node
``os.cpu_count()`` is the whole node while the job was allocated a few of its cores, and a pool
sized from the node oversubscribes the allocation. ``os.sched_getaffinity`` answers for the
process where the OS has it (Linux); elsewhere the machine count is the best there is.

**``share`` splits those cores among sibling pools.** A multi-GPU run starts one worker process per
GPU, each with its own pool, and each asking for every core but one would put ``share`` times the
threads on the machine.
"""

from __future__ import annotations

import os

#: The most threads any read-bound pool asks for by default.
DEFAULT_CAP = 8


def usable_cores() -> int | None:
    """Cores this process may be scheduled on, or ``None`` when the OS will not say."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0)) or None
    return os.cpu_count()


def thread_workers(cap: int = DEFAULT_CAP, share: int = 1) -> int:
    """Threads for one of ``share`` sibling pools whose work is mostly waiting on reads: this
    pool's part of the usable cores, less one each for the threads that feed and wait on it, in
    ``[1, cap]``. Cores unknown count as four."""
    share = max(1, int(share))
    cores = usable_cores() or 4
    return max(1, min(int(cap), (cores - share) // share))
