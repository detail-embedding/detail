"""Content fingerprints (Section 2): the cheap head/tail fingerprint and the full hash."""

import hashlib

from detail.storage import integrity


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "a.bin"
    payload = b"hello world" * 1000
    path.write_bytes(payload)
    assert integrity.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_file_fingerprint_reports_size_and_ends_hash(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(b"abc")
    fp = integrity.file_fingerprint(path)
    assert fp["size"] == 3
    assert "mtime_ns" in fp
    # For a file smaller than the chunk, sha256_ends is the whole-file hash.
    assert fp["sha256_ends"] == hashlib.sha256(b"abc").hexdigest()


def test_file_fingerprint_hashes_head_and_tail_of_a_large_file(tmp_path):
    # A file larger than one chunk hashes the first + last MiB, not the middle.
    chunk = integrity._CHUNK
    head = b"H" * chunk
    middle = b"M" * chunk
    tail = b"T" * chunk
    path = tmp_path / "big.bin"
    path.write_bytes(head + middle + tail)
    expected = hashlib.sha256()
    expected.update(head)
    expected.update(tail)
    fp = integrity.file_fingerprint(path)
    assert fp["size"] == 3 * chunk
    assert fp["sha256_ends"] == expected.hexdigest()


def test_fingerprint_changes_when_content_changes(tmp_path):
    path = tmp_path / "v.bin"
    path.write_bytes(b"x" * 10)
    before = integrity.file_fingerprint(path)
    path.write_bytes(b"x" * 10 + b"y")  # append one byte
    after = integrity.file_fingerprint(path)
    assert after["size"] != before["size"]
    assert after["sha256_ends"] != before["sha256_ends"]


def test_provenance_reexports_the_storage_helpers():
    # The functions moved to storage; provenance still exposes them for its existing callers.
    from detail.analysis import provenance

    assert provenance.file_fingerprint is integrity.file_fingerprint
    assert provenance.sha256_file is integrity.sha256_file
