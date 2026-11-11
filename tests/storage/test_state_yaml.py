"""``storage/state_yaml.py`` — the authored-run state record (pose / query / temporal / annotation).

The read goes through ``core.atomic.read_text`` because the write goes through
``core.atomic.replace``: on Windows a reader that opens the file while it is being replaced
gets a transient sharing-violation ``PermissionError`` (#317, seen on this read under
full-suite load). ``tests/pipeline/test_atomic_write_seam.py`` pins that routing by source
scan; the test here pins it by call, and ``tests/core/test_atomic.py`` pins the retry.
"""

from pathlib import Path

import pytest

from detail.core import atomic
from detail.storage import state_yaml


def test_round_trip_puts_schema_version_first(tmp_path):
    path = tmp_path / "run" / "pose_clustering.yaml"
    state_yaml.write_state_yaml(path, {"status": "draft", "schema_version": 99, "k": 3})

    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == "schema_version: 1"
    assert state_yaml.read_state_yaml(path) == {
        "schema_version": 1,
        "status": "draft",
        "k": 3,
    }
    assert not (path.parent / (path.name + ".tmp")).exists()


def test_read_of_an_absent_record_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no pose_clustering.yaml in"):
        state_yaml.read_state_yaml(tmp_path / "pose_clustering.yaml")


def test_read_goes_through_the_retry(tmp_path, monkeypatch):
    # The retry itself is `tests/core/test_atomic.py`'s; what this pins is that the read
    # reaches it, rather than a bare `open` a concurrent `atomic.replace` can deny.
    path = tmp_path / "pose_clustering.yaml"
    state_yaml.write_state_yaml(path, {"status": "draft"})
    real_retry = atomic.retry_on_lock
    calls = []

    def spy(operation):
        calls.append(operation)
        return real_retry(operation)

    monkeypatch.setattr(atomic, "retry_on_lock", spy)

    assert state_yaml.read_state_yaml(path)["status"] == "draft"
    assert len(calls) == 1, "read_state_yaml bypassed core.atomic's retry"
