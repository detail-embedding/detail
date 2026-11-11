"""Machine-local video-root registry (Section 2) — storage.yaml's first writer."""

import pytest
import yaml

from detail.storage import locations, video_roots
from tests.concurrency import join_all, rendezvous


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Point storage.yaml at a throwaway location so tests never touch ~/.config."""
    config_dir = tmp_path / "config"
    monkeypatch.setattr(locations, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(locations, "STORAGE_PATH", config_dir / "storage.yaml")
    return config_dir


def test_register_and_resolve(isolated_storage, tmp_path):
    root = tmp_path / "nas" / "videos"
    root.mkdir(parents=True)
    video_roots.register_video_root("lab_nas", root)
    assert video_roots.video_root("lab_nas") == root
    assert video_roots.list_video_roots() == {"lab_nas": root}
    resolved = video_roots.resolve_video_ref({"root": "lab_nas", "rel": "day3/m27.mp4"})
    assert resolved == root / "day3" / "m27.mp4"


def test_resolve_unknown_root_raises_missing(isolated_storage):
    with pytest.raises(video_roots.MissingVideoRootError) as exc:
        video_roots.resolve_video_ref({"root": "nope", "rel": "a.mp4"})
    assert "nope" in str(exc.value)


def test_resolve_rejects_a_malformed_reference(isolated_storage):
    with pytest.raises(ValueError):
        video_roots.resolve_video_ref({"root": "lab_nas"})  # no rel


def test_register_rejects_a_missing_directory(isolated_storage, tmp_path):
    with pytest.raises(NotADirectoryError):
        video_roots.register_video_root("gone", tmp_path / "does-not-exist")


def test_register_rejects_an_empty_alias(isolated_storage, tmp_path):
    with pytest.raises(ValueError):
        video_roots.register_video_root("", tmp_path)


def test_repoint_updates_the_alias(isolated_storage, tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    video_roots.register_video_root("lab_nas", first)
    video_roots.register_video_root("lab_nas", second)  # re-point (bulk move)
    assert video_roots.video_root("lab_nas") == second


def test_resolve_or_register_auto_registers_the_folder(isolated_storage, tmp_path):
    vids = tmp_path / "data" / "day3"
    vids.mkdir(parents=True)
    video = vids / "m27.mp4"
    video.write_bytes(b"x")
    alias = video_roots.resolve_or_register_root(video)
    # a location for the video's folder now exists and resolves the file back
    assert video_roots.video_root(alias) == vids.resolve()
    assert (
        video_roots.resolve_video_ref({"root": alias, "rel": "m27.mp4"})
        == vids / "m27.mp4"
    )


def test_resolve_or_register_reuses_a_containing_root(isolated_storage, tmp_path):
    lib = tmp_path / "data"
    (lib / "day3").mkdir(parents=True)
    video = lib / "day3" / "m27.mp4"
    video.write_bytes(b"x")
    video_roots.register_video_root("lib", lib)  # a broad root already covers it
    alias = video_roots.resolve_or_register_root(video)
    assert alias == "lib"  # reused, not a second auto root
    assert set(video_roots.list_video_roots()) == {"lib"}


def test_auto_alias_dedupes_same_basename(isolated_storage, tmp_path):
    a = tmp_path / "x" / "day1"
    b = tmp_path / "y" / "day1"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "v.mp4").write_bytes(b"1")
    (b / "v.mp4").write_bytes(b"2")
    alias_a = video_roots.resolve_or_register_root(a / "v.mp4")
    alias_b = video_roots.resolve_or_register_root(b / "v.mp4")
    assert alias_a == "day1" and alias_b == "day1_2"  # distinct folders, deduped names


def test_write_preserves_other_storage_keys(isolated_storage, tmp_path):
    # storage.yaml may already hold the managed-workspace root; registering a video root
    # must not clobber it.
    isolated_storage.mkdir(parents=True, exist_ok=True)
    locations.STORAGE_PATH.write_text(
        yaml.safe_dump({"storage": {"default": {"root": "/tmp/exp"}}}), encoding="utf-8"
    )
    root = tmp_path / "vids"
    root.mkdir()
    video_roots.register_video_root("lab_nas", root)
    data = yaml.safe_load(locations.STORAGE_PATH.read_text(encoding="utf-8-sig"))
    assert data["storage"]["default"]["root"] == "/tmp/exp"
    assert data["video_roots"]["lab_nas"] == root.resolve().as_posix()


def test_concurrent_registrations_all_survive(isolated_storage, tmp_path):
    """Importing a batch runs the parts in parallel *processes*, and each registers its
    folder. A plain read-modify-write loses whichever alias landed first — and a lost
    alias means a referenced part's {root, rel} resolves to nothing, so the run gate
    refuses a video sitting right where it always was.

    Threads stand in for the workers here: same interleaving of read/merge/write, no
    subprocess machinery.
    """
    import threading

    folders = []
    for i in range(8):
        folder = tmp_path / f"rig{i}"
        folder.mkdir()
        folders.append(folder)

    barrier = threading.Barrier(len(folders))

    def register(index):
        rendezvous(barrier)  # maximise the overlap
        video_roots.register_video_root(f"rig{index}", folders[index])

    threads = [
        threading.Thread(target=register, args=(i,)) for i in range(len(folders))
    ]
    for t in threads:
        t.start()
    join_all(threads)

    registered = video_roots.list_video_roots()
    assert sorted(registered) == [f"rig{i}" for i in range(len(folders))]


def test_a_failed_write_leaves_no_temp_file(isolated_storage, tmp_path, monkeypatch):
    """The temp name is unique per writer (a shared one raced), so a crash between write
    and replace would otherwise litter the config dir with orphans."""
    from detail.storage import roots

    folder = tmp_path / "rig"
    folder.mkdir()
    # Patched on `roots`, which is where the write now lives — `video_roots` is the video-shaped
    # door onto it. Point of use, as the mocking rule says; the point moved.
    monkeypatch.setattr(
        roots.atomic, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom"))
    )
    with pytest.raises(OSError):
        video_roots.register_video_root("rig", folder)
    assert not list(isolated_storage.glob("*.tmp"))
