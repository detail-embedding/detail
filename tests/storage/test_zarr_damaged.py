"""A zarr the OS lost is named at the store, whichever array or group is read.

The class and the incident are written up in ``tests/core/test_damaged_files.py``; this is its
zarr half. A detection run copied back from a cluster had a zero-filled 93-byte group
``zarr.json`` (and 418 zero or empty chunks). zarr handed it to ``json.loads`` as bytes, which
sniffs leading NULs as UTF-32, so the failure was ``'utf-32-be' codec can't decode byte 0x00 in
position 92: truncated data`` (position 92 is the last of 93 bytes) with no path.

Why it is on the **store** (``zarr_store.RetryingLocalStore``): every zarr this package opens
goes through ``zarr_store.open_group``, and every read of one goes through the store's ``get`` or
``get_sync``. One check there names the file for the group, for each array, for a chunk and for
anything written next year, where a check at the call that happened to fail names one.

What is deliberately not caught: a zero-filled *chunk*. An array of zeros is a legitimate chunk
and the store cannot tell the two apart without decoding it; the codec reports that one. The
group and array metadata, which are JSON that cannot be all zeros, and any *empty* file, which
zarr never writes, are caught.

``check_intact`` is the cheap, metadata-only form of the same question, for a launch over many
recordings (``tests/api/test_damaged_upstream.py``).
"""

import os

import pytest

from detail.core import atomic
from detail.storage import zarr_store
from detail.storage.zarr_io import DetectionPartReader, DetectionPartWriter


@pytest.fixture
def part(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as writer:
        for frame in range(40):
            writer.add_row(frame, frame * 33, 0, "mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
    return path


def _lose(path):
    """What an unclean shutdown leaves: the same length, all zeros."""
    path.write_bytes(b"\x00" * path.stat().st_size)


def _chunk(part):
    return next(
        p
        for p in sorted((part / "bbox").rglob("*"))
        if p.is_file() and p.name != "zarr.json"
    )


def test_a_healthy_part_reads_and_passes_the_check(part):
    zarr_store.check_intact(part.parent)
    assert DetectionPartReader(part).column("bbox").shape[0] == 40


def test_a_zeroed_group_is_named_where_it_is_opened(part):
    _lose(part / "zarr.json")

    with pytest.raises(atomic.DamagedFileError) as caught:
        zarr_store.open_group(part, mode="r")

    assert str(part / "zarr.json") in str(caught.value)
    assert "all zero bytes" in str(caught.value)


def test_it_is_not_the_utf32_error_it_used_to_be(part):
    """The report, reproduced: the group's metadata at its real length, zeroed, read the way the
    failing worker read it (the production reader)."""
    group_json = part / "zarr.json"
    _lose(group_json)
    assert group_json.stat().st_size == 93  # position 92 of 93, as in the report

    with pytest.raises(atomic.DamagedFileError):
        DetectionPartReader(part)
    try:
        DetectionPartReader(part)
    except UnicodeDecodeError:
        pytest.fail("a decoder error that names no file, as before")
    except atomic.DamagedFileError:
        pass


def test_a_zeroed_array_is_named_when_the_array_is_read(part):
    _lose(part / "bbox" / "zarr.json")
    group = zarr_store.open_group(part, mode="r")  # the group itself is fine

    with pytest.raises(atomic.DamagedFileError) as caught:
        group["bbox"]

    assert str(part / "bbox" / "zarr.json") in str(caught.value)


def test_an_empty_chunk_is_named_when_it_is_read(part):
    chunk = _chunk(part)
    chunk.write_bytes(b"")
    group = zarr_store.open_group(part, mode="r")

    with pytest.raises(atomic.DamagedFileError) as caught:
        group["bbox"][:]

    assert str(chunk) in str(caught.value)
    assert "empty" in str(caught.value)


def test_a_chunk_that_was_never_written_is_still_just_missing(tmp_path):
    """Not damage: zarr skips a chunk that holds only the fill value, and a missing key reads as
    the fill value. An empty *file* is the only thing flagged."""
    path = tmp_path / "sparse.zarr"
    group = zarr_store.open_group(path, mode="w")
    group.create_array("x", shape=(10,), chunks=(5,), dtype="int32", fill_value=7)
    reopened = zarr_store.open_group(path, mode="r")
    assert list(reopened["x"][:]) == [7] * 10


# --------------------------------------------------------------------------- #
# check_intact: the metadata-only form, for a launch
# --------------------------------------------------------------------------- #
def test_check_intact_names_a_zeroed_group_without_opening_anything(part):
    _lose(part / "zarr.json")
    with pytest.raises(atomic.DamagedFileError, match="zarr.json"):
        zarr_store.check_intact(part.parent)


def test_check_intact_names_a_zeroed_array_and_an_empty_one(part):
    _lose(part / "frame_idx" / "zarr.json")
    with pytest.raises(atomic.DamagedFileError, match="frame_idx"):
        zarr_store.check_intact(part.parent)

    (part / "frame_idx" / "zarr.json").write_bytes(b"")
    with pytest.raises(atomic.DamagedFileError, match="empty"):
        zarr_store.check_intact(part.parent)


def test_check_intact_reads_metadata_only(part):
    """A zeroed chunk is the codec's to report. Walking every chunk of every recording of a launch
    is exactly the cost this check exists to avoid."""
    _lose(_chunk(part))
    zarr_store.check_intact(part.parent)  # does not raise


def test_check_intact_ignores_a_zarr_json_outside_a_zarr_directory(tmp_path):
    """Only a ``*.zarr`` store's metadata is this function's business."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "zarr.json").write_bytes(b"\x00" * 10)
    zarr_store.check_intact(tmp_path)


def test_check_intact_of_a_directory_that_is_not_there_is_fine(tmp_path):
    zarr_store.check_intact(tmp_path / "nothing-here")


def test_a_store_can_be_checked_by_its_own_path(part):
    """``root`` may be the ``*.zarr`` folder itself, not only something above one."""
    zarr_store.check_intact(part)
    _lose(part / "zarr.json")
    with pytest.raises(atomic.DamagedFileError, match="zarr.json"):
        zarr_store.check_intact(part)


def test_a_zarr_folder_above_the_root_does_not_make_the_root_a_store(tmp_path):
    """The check looks at the path *below* ``root``, not at every folder in the string it was
    handed: a ``data.zarr`` somewhere up the tree says nothing about ``runs`` inside it.
    """
    root = tmp_path / "data.zarr" / "runs"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "zarr.json").write_bytes(b"\x00" * 10)
    zarr_store.check_intact(root)
    assert os.path.isdir(root)


def test_both_read_entry_points_of_the_store_name_a_lost_file(part):
    """``get`` (zarr's async path, which a group open and an array read take) and ``get_sync``
    are separate overrides. The first is covered through the reader above; the second is called
    directly, since nothing reached through ``open_group`` today is guaranteed to use it.
    """
    import asyncio

    _lose(part / "zarr.json")
    store = zarr_store.RetryingLocalStore(part, read_only=True)

    with pytest.raises(atomic.DamagedFileError, match="zarr.json"):
        store.get_sync("zarr.json")
    with pytest.raises(atomic.DamagedFileError, match="zarr.json"):
        asyncio.run(store.get("zarr.json"))
    # and a partial read is not a whole-file claim, so it is left alone
    from zarr.abc.store import RangeByteRequest

    assert store.get_sync("zarr.json", byte_range=RangeByteRequest(0, 4)) is not None
