import pytest

from detail.core import atomic
from tests.module_clock import patch_module_time


def test_retry_on_lock_succeeds_after_transient_error(monkeypatch):
    patch_module_time(monkeypatch, atomic, sleep=lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "locked")
        return "ok"

    assert atomic.retry_on_lock(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_on_lock_reraises_after_exhausting(monkeypatch):
    patch_module_time(monkeypatch, atomic, sleep=lambda _s: None)
    calls = {"n": 0}

    def always_locked():
        calls["n"] += 1
        raise PermissionError(13, "locked")

    with pytest.raises(PermissionError):
        atomic.retry_on_lock(always_locked)
    assert calls["n"] == atomic.RETRY_ATTEMPTS


def test_retry_on_lock_does_not_swallow_other_errors(monkeypatch):
    patch_module_time(monkeypatch, atomic, sleep=lambda _s: None)

    def boom():
        raise ValueError("not a lock")

    with pytest.raises(ValueError, match="not a lock"):
        atomic.retry_on_lock(boom)


def test_replace_moves_file(tmp_path):
    src = tmp_path / "a.tmp"
    dst = tmp_path / "a.txt"
    src.write_text("hello", encoding="utf-8")
    atomic.replace(src, dst)
    assert dst.read_text(encoding="utf-8-sig") == "hello"
    assert not src.exists()
