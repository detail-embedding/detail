"""The shared read counter charges a measurement only for reads under its own root.

``tests/round_trips.py`` holds the write-up: a process-wide count charged a page for another
test's leftover progress poller, failing ``test_page_read_cost`` in 5 of 86 full runs.
"""

from __future__ import annotations

import os
import threading
import time

from tests.round_trips import count_stats, counting


def test_a_stray_thread_reading_elsewhere_is_not_counted(tmp_path):
    mine, elsewhere = tmp_path / "mine", tmp_path / "elsewhere"
    mine.mkdir()
    elsewhere.mkdir()
    (mine / "a.yaml").write_text("a: 1", encoding="utf-8")
    stop = threading.Event()

    def poller():  # what a page left open on a "running" run does, every 0.25 s
        while not stop.is_set():
            os.stat(elsewhere)
            time.sleep(0.001)

    stray = threading.Thread(target=poller, daemon=True)
    stray.start()
    try:
        with counting(mine) as seen:
            time.sleep(0.05)  # the stray gets plenty of reads in
            os.stat(mine / "a.yaml")
            # `pathlib` opens through `io.open`, which is what the measured seams read with.
            (mine / "a.yaml").read_text(encoding="utf-8-sig")
            list(os.scandir(mine))
    finally:
        stop.set()
        stray.join(1)
    assert dict(seen) == {"stat": 1, "open": 1, "scandir": 1}


def test_a_read_on_a_worker_thread_under_the_root_still_counts(tmp_path):
    """Off the UI thread is not free: the count is by path, never by thread."""
    target = tmp_path / "a.yaml"
    target.write_text("a: 1", encoding="utf-8")

    def read_off_thread():
        worker = threading.Thread(target=lambda: os.stat(target))
        worker.start()
        worker.join()

    _, stats = count_stats(tmp_path, read_off_thread)
    assert stats == 1
