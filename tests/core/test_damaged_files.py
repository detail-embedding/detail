"""A file the OS lost reads as "damaged, here, and why", not as a decoder error with no path.

**The class: a persisted file torn by an unclean shutdown, read as ordinary content.** After a
shutdown a filesystem can keep a new file's name and length while its bytes were still in the
cache, so it reads back as that many zeros (or as nothing). The parsers then fail in their own
words. PyYAML and ``json`` sniff leading NULs as UTF-32, so what a user saw was

    UnicodeDecodeError: 'utf-32-be' codec can't decode byte 0x00 in position 92: truncated data

on one recording of a tracking run, naming no file and no cause. The file was a 93-byte
``zarr.json`` in a detection run copied back from a cluster (Oct 2026, issue #695); 418 of its
chunk files were zeros too, and the cluster's own copy was intact.

The repo had met half of this before: ``atomic.replace`` flushes before it renames because an
interrupted distill run once left three records all-NUL. That protects the *writers that go
through it*. Three things were still open, and one guard each:

- **the read** names the file and says what happened (this module, and
  ``tests/storage/test_zarr_damaged.py`` for zarr), as an error that the readers which already
  tolerate a torn record still catch;
- **the write**: a run is not recorded complete until its outputs are flushed
  (``tests/executor/test_complete_is_durable.py``), including the copy-back that lost them;
- **the launch**: a damaged upstream run is refused up front, by name
  (``tests/api/test_damaged_upstream.py``).

Where the fix stops (``fix-the-class`` step 3.5): an *empty* text file is returned as ``""``,
because a writer may have meant it; a zero-filled zarr *chunk* is not flagged, because an array
of zeros is a legitimate chunk and only the codec can tell. Text readers that bypass
``atomic.read_text`` (a bare ``yaml.safe_load(path.read_text())``) are not covered; the
canonical seam is.
"""

import json
import os

import pytest
import yaml

from detail.core import atomic
from detail.storage import dirscan

LOST = b"\x00" * 93


def test_an_all_nul_file_is_named_with_its_path_and_cause(tmp_path):
    path = tmp_path / "recording.yaml"
    path.write_bytes(LOST)

    with pytest.raises(atomic.DamagedFileError) as caught:
        atomic.read_text(path)

    message = str(caught.value)
    assert str(path) in message  # the file, which the decoder error never gave
    assert "all zero bytes" in message and "93" in message
    assert "shut down" in message  # the cause, in words
    assert "Re-run" in message  # and what to do


def test_a_file_that_has_real_content_is_untouched(tmp_path):
    path = tmp_path / "recording.yaml"
    path.write_text("schema_version: 1\nname: café\n", encoding="utf-8")
    assert atomic.read_text(path) == "schema_version: 1\nname: café\n"


def test_a_file_with_text_and_a_stray_nul_is_not_called_damaged(tmp_path):
    """Only a file with *nothing but* NULs is what a lost write leaves."""
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc\x00def")
    assert atomic.read_text(path) == "abc\x00def"


def test_an_empty_file_reads_as_empty_text(tmp_path):
    """A writer may have meant it. Empty is a zarr-store verdict, not a text one."""
    path = tmp_path / "marker.txt"
    path.write_bytes(b"")
    assert atomic.read_text(path) == ""


def test_a_zeroed_job_envelope_is_named_not_a_json_decode_error(tmp_path):
    """The JSON side of the seam: every run's ``job.json`` is read through it."""
    from detail.executor import rundir

    (tmp_path / "job.json").write_bytes(LOST)
    with pytest.raises(atomic.DamagedFileError, match="job.json"):
        rundir.read_envelope(tmp_path)


def test_the_error_is_what_the_tolerant_readers_already_catch():
    """The torn-record handlers predate this and catch the parsers' own errors. A new type that
    slipped past them would turn a listing that degraded into one that crashes, so it is both
    an ``OSError`` and a ``ValueError`` (as ``io.UnsupportedOperation`` is), and every handler
    in the repo is checked against it, not against my reading of them."""
    error = atomic.DamagedFileError("x")
    assert isinstance(error, OSError) and isinstance(error, ValueError)
    assert isinstance(error, dirscan.UNREADABLE_RECORD_ERRORS)
    assert not isinstance(error, yaml.YAMLError)  # which is why OSError matters
    for handler in (
        (OSError, yaml.YAMLError),  # storage/benchmarks.py, gui/comparison.py
        (FileNotFoundError, ValueError, yaml.YAMLError),  # api/experiment.py
    ):
        assert isinstance(error, handler)


def test_the_listing_scanner_skips_a_zeroed_record_instead_of_crashing(tmp_path):
    """End to end through ``dirscan``: one zero-filled record among healthy ones."""
    good = tmp_path / "recordings" / "recording_001"
    bad = tmp_path / "recordings" / "recording_002"
    for folder in (good, bad):
        folder.mkdir(parents=True)
    (good / "recording.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (bad / "recording.yaml").write_bytes(LOST)

    def read(folder):
        return yaml.safe_load(atomic.read_text(folder / "recording.yaml"))

    results = []
    for folder in (good, bad):
        try:
            results.append(read(folder))
        except dirscan.UNREADABLE_RECORD_ERRORS:
            results.append(None)
    assert results == [{"schema_version": 1}, None]


# --------------------------------------------------------------------------- #
# flush_tree: what "complete" has to mean
# --------------------------------------------------------------------------- #
@pytest.fixture
def fsynced(monkeypatch):
    """Which paths were flushed. ``fsync`` takes a descriptor, so the path is resolved from it
    at call time; a real file is flushed for real, so the call also proves it can be."""
    seen = []
    real = os.fsync

    def spy(fd):
        try:
            seen.append(os.readlink(f"/proc/self/fd/{fd}"))
        except OSError:  # not Linux: count the call, the path is not recoverable
            seen.append(None)
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    return seen


def _tree(root):
    for relative in ("a.json", "z/b.bin", "z/deep/c.bin"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data " + relative.encode())
    return 3


def test_flush_tree_flushes_every_file_and_says_how_many(tmp_path, fsynced):
    expected = _tree(tmp_path)

    assert atomic.flush_tree(tmp_path) == expected
    # at least a flush per file; directories add more where the platform has them
    assert len(fsynced) >= expected
    if os.name != "nt" and all(fsynced):
        names = {os.path.basename(p) for p in fsynced}
        assert {"a.json", "b.bin", "c.bin"} <= names


def test_flush_tree_leaves_the_data_as_it_found_it(tmp_path):
    _tree(tmp_path)
    atomic.flush_tree(tmp_path)
    assert (tmp_path / "z" / "deep" / "c.bin").read_bytes() == b"data z/deep/c.bin"


def test_a_file_replaced_while_walking_is_skipped_not_fatal(tmp_path, monkeypatch):
    """A progress thread rewriting ``job.json`` swaps a temp file in and out under the walk.
    The replacement was flushed by ``replace`` before the rename, so a vanished one is fine.
    """
    _tree(tmp_path)
    real = atomic._flush_to_disk

    def vanish(path):
        if str(path).endswith("b.bin"):
            raise FileNotFoundError(path)
        real(path)

    monkeypatch.setattr(atomic, "_flush_to_disk", vanish)
    assert atomic.flush_tree(tmp_path) == 2  # the other two


def test_flush_tree_of_a_missing_directory_flushes_nothing(tmp_path):
    assert atomic.flush_tree(tmp_path / "nope") == 0


def test_a_json_envelope_that_is_merely_malformed_is_still_a_json_error(tmp_path):
    """Only the zero-filled file is rewritten into the new error. Genuine garbage is the
    parser's to report."""
    path = tmp_path / "job.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(atomic.read_text(path))
