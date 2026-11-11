"""Peak memory a job used, sampled across its whole process tree.

The sampler exists because a job's own process is not where all of its memory lives: training
starts DataLoader workers and a multi-GPU run spawns one process per GPU. So the test that
matters puts the memory in a *child*, where a sampler reading only its own process sees nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil

import detail
from detail.core import resource_peaks

#: The checkout under test, first on the child's path — otherwise a worktree's child would
#: import whichever checkout the editable install points at.
_REPO = str(Path(detail.__file__).resolve().parents[1])

_BLOCK_MB = 200


def test_memory_held_by_a_child_process_is_counted():
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time
                block = bytearray({_BLOCK_MB} * 2**20)
                block[::4096] = b"x" * len(block[::4096])  # touch every page
                print("ready", flush=True)
                time.sleep(30)
            """),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        own = psutil.Process().memory_info().rss
        sampler = resource_peaks.PeakSampler(interval_s=0.05).start()
        time.sleep(0.2)
        peaks = sampler.stop()
    finally:
        child.kill()
        child.wait()
        child.stdout.close()
    # Above this process's own memory by the child's block: a sampler that read only itself
    # would pass a bare ">= block" check whenever pytest alone outgrew the block.
    assert peaks["peak_ram_bytes"] >= own + 0.9 * _BLOCK_MB * 2**20, (own, peaks)


def test_a_job_that_never_loaded_torch_reports_no_gpu_figure():
    """Blank, not 0: a zero reads as a measurement. Run in a fresh interpreter so a torch some
    other test imported cannot make this pass or fail for the wrong reason — and so the same run
    proves the sampler and the run-time machine snapshot never import torch themselves.
    """
    code = textwrap.dedent("""
        import sys
        from detail.core import hostinfo, resource_peaks
        peaks = resource_peaks.PeakSampler(interval_s=0.05).start().stop()
        facts = hostinfo.collect(load_torch=False)
        assert "torch" not in sys.modules, "torch was imported"
        assert peaks["gpu_backend"] is None, peaks
        assert peaks["peak_gpu_reserved_bytes"] is None, peaks
        assert facts["torch"] == hostinfo.NOT_LOADED, facts
        print("ok")
    """)
    env = {
        **os.environ,
        "PYTHONPATH": _REPO + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
