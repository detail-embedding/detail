"""``core/cpu.py``: the one rule for how many threads a read-bound pool runs, and the scan that
keeps it the one.

**The class.** ``max(1, min(8, (os.cpu_count() or 4) - 1))`` was spelled out in the embedding
worker and in the kinematics backfill, and the clip montage arrived with a third answer to the
same question (``WORKERS = 6``, a constant right on one box). Two copies can be tolerated; the
third becomes one mechanism (``CLAUDE.md``, *Sizing a change*). The scan below fails on an
``os.cpu_count()`` anywhere under ``detail/`` but ``core/cpu.py`` and the sites in ``ALLOWED``,
each of which answers a *different* question and says so. An entry is a claim: one whose site
has stopped calling ``os.cpu_count()`` fails rather than sitting here excusing nothing.
"""

import pathlib
import re

import pytest

from detail.core import cpu

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "detail"
RULE = PACKAGE / "core" / "cpu.py"
_CALL = re.compile(r"\bos\.cpu_count\(\)")

#: Sites that size something other than a read-bound thread pool, with why each is not the rule.
ALLOWED: dict[str, str] = {
    # Reporting the machine's core count on a performance report, not sizing anything.
    "detail/core/hostinfo.py": "the machine block reports the count",
    # The transcode pool: decoders self-thread, so a floor of four, env-overridable.
    "detail/core/utils.py": "max_transcode_parallelism: capped at 4, DETAIL_MAX_TRANSCODE_PARALLELISM",
}


def _violations(root: pathlib.Path) -> list[str]:
    """Every file under ``root`` (relative paths) that calls ``os.cpu_count()`` and is neither
    the rule nor allowed."""
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if path == RULE or rel in ALLOWED:
            continue
        if _CALL.search(path.read_text(encoding="utf-8-sig")):
            found.append(rel)
    return found


def test_the_rule_is_cores_minus_one_within_one_and_the_cap(monkeypatch):
    for cores, want in ((1, 1), (2, 1), (4, 3), (9, 8), (72, 8), (None, 3)):
        monkeypatch.setattr(cpu, "usable_cores", lambda c=cores: c)
        assert cpu.thread_workers() == want, cores
    monkeypatch.setattr(cpu, "usable_cores", lambda: 72)
    assert cpu.thread_workers(cap=4) == 4


def test_cores_are_the_processes_allocation_not_the_machines(monkeypatch):
    """A SLURM job given 4 cores of a 64-core node sizes its pool from the 4."""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False
    )
    assert cpu.usable_cores() == 4
    assert cpu.thread_workers() == 3


def test_without_affinity_the_machine_count_is_used(monkeypatch):
    """Windows and macOS have no sched_getaffinity: the machine count stands."""
    import os

    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    assert cpu.usable_cores() == 12


@pytest.mark.parametrize(
    "cores, share, want",
    [(32, 1, 8), (32, 3, 8), (16, 3, 4), (8, 3, 1), (4, 4, 1), (72, 2, 8)],
)
def test_sibling_pools_split_the_cores(monkeypatch, cores, share, want):
    """One pool per GPU worker: each gets its part, less one for the thread that feeds it."""
    monkeypatch.setattr(cpu, "usable_cores", lambda: cores)
    assert cpu.thread_workers(share=share) == want


def test_the_two_pools_that_had_their_own_copy_use_the_rule():
    from detail.pipeline import _embedding_worker, kinematics_backfill

    assert _embedding_worker._render_workers() == cpu.thread_workers()
    assert kinematics_backfill._decode_workers() == cpu.thread_workers()


def test_no_module_but_the_rule_counts_cores_itself():
    assert _violations(PACKAGE) == []


def test_every_allowed_site_still_counts_cores():
    """An entry is a claim about a site that exists and still calls ``os.cpu_count()``."""
    stale = [
        rel
        for rel in ALLOWED
        if not (ROOT / rel).exists()
        or not _CALL.search((ROOT / rel).read_text(encoding="utf-8-sig"))
    ]
    assert stale == [], f"allowed sites that no longer count cores: {stale}"


def test_the_scan_would_catch_a_third_copy(tmp_path):
    planted = tmp_path / "detail" / "api" / "clip_montage.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        "import os\nWORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))\n",
        encoding="utf-8",
    )
    found = [
        p.relative_to(tmp_path).as_posix()
        for p in sorted((tmp_path / "detail").rglob("*.py"))
        if _CALL.search(p.read_text(encoding="utf-8-sig"))
    ]
    assert found == ["detail/api/clip_montage.py"]
    assert _CALL.search("n = os.cpu_count() or 1")
    assert not _CALL.search("n = cpu.thread_workers()")


@pytest.mark.parametrize("bad", [0, -1])
def test_a_cap_below_one_still_gives_one_thread(bad):
    assert cpu.thread_workers(cap=bad) == 1
