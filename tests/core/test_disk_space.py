"""Free space where a write really lands (#743) — and the WSL trap.

Inside WSL, ``statvfs`` on a path under the distribution's own filesystem answers for its
ext4 virtual disk, whose ceiling (1 TB) is not the space it has: the disk is one file on a
Windows drive. On the machine that found this, it reported 789 GB free while ``C:`` had 98 GB,
and later 17 MB. The guard pins that the tighter of the two is what a run is checked against.
"""

from collections import namedtuple

import pytest

from detail.core import disk_space

_Usage = namedtuple("_Usage", "total used free")
GB = 1024**3


def test_format_size():
    assert disk_space.format_size(1.5 * GB) == "1.5 GB"
    assert disk_space.format_size(340 * 1024**2) == "340.0 MB"
    assert disk_space.format_size(12 * 1024) == "12 KB"
    assert disk_space.format_size(2 * 1024**4) == "2.0 TB"


def test_frames_to_process_sums_parts_and_rounds_the_stride_up():
    data = {"video": {"files": [{"frame_count": 1000}, {"frame_count": 501}]}}
    assert disk_space.frames_to_process(data, None) == 1501
    assert disk_space.frames_to_process(data, 2) == 751  # frames 0, 2, …, 1500
    assert disk_space.frames_to_process({}, 3) == 0


def test_an_estimate_must_leave_the_headroom_free():
    room = disk_space.HEADROOM_BYTES
    fits = disk_space.SpaceEstimate(
        bytes=10, free_bytes=10 + room, where="x", per_recording={}
    )
    tight = disk_space.SpaceEstimate(
        bytes=11, free_bytes=10 + room, where="x", per_recording={}
    )
    assert fits.fits and not tight.fits


def test_the_windows_drive_behind_wsl_bounds_the_write_when_it_is_tighter(
    monkeypatch, tmp_path
):
    usage = {
        str(tmp_path): _Usage(1024 * GB, 235 * GB, 789 * GB),
        "/mnt/c": _Usage(931 * GB, 833 * GB, 98 * GB),
    }
    monkeypatch.setattr(disk_space.shutil, "disk_usage", lambda p: usage[str(p)])
    monkeypatch.setattr(disk_space, "_wsl_host_drive", lambda _p: ("/mnt/c", "C:"))

    free, where = disk_space.free_space(tmp_path)

    assert free == 98 * GB
    assert where.startswith("C:")


def test_the_path_s_own_free_space_wins_when_it_is_the_tighter(monkeypatch, tmp_path):
    usage = {
        str(tmp_path): _Usage(1024 * GB, 1000 * GB, 24 * GB),
        "/mnt/c": _Usage(931 * GB, 100 * GB, 831 * GB),
    }
    monkeypatch.setattr(disk_space.shutil, "disk_usage", lambda p: usage[str(p)])
    monkeypatch.setattr(disk_space, "_wsl_host_drive", lambda _p: ("/mnt/c", "C:"))

    assert disk_space.free_space(tmp_path) == (24 * GB, str(tmp_path))


def test_a_path_that_does_not_exist_yet_asks_its_nearest_parent(monkeypatch, tmp_path):
    # Off WSL on purpose: on a WSL box the real lookup answers with the Windows drive, which
    # is the point of the module and would make this test about the machine it runs on.
    monkeypatch.setattr(disk_space, "_wsl_host_drive", lambda _p: None)
    free, where = disk_space.free_space(tmp_path / "not" / "yet")
    assert free > 0 and where == str(tmp_path)


@pytest.mark.parametrize(
    "answer, letter",
    [
        ("C:\\Users\\samgo\\AppData\\Local\\wsl\\{2159524b}", "C"),
        ("\\\\?\\D:\\WSL\\Ubuntu\r\n", "D"),
        ("\\\\server\\share\\wsl", None),
        ("", None),
        ("C:\\a\nD:\\b", None),
    ],
)
def test_the_drive_letter_is_read_from_the_registry_answer(answer, letter):
    assert disk_space.drive_letter(answer) == letter


def test_off_wsl_there_is_no_host_drive(monkeypatch, tmp_path):
    monkeypatch.setattr(disk_space, "_is_wsl", lambda: False)
    assert disk_space._wsl_host_drive(tmp_path) is None
