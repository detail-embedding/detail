"""One edit of a machine-config file writes back to the file it read (``storage/machine_file.py``).

The class: **a read-modify-write that looks its target path up twice.** ``storage.yaml``,
``credentials.yaml`` and ``remote_targets.yaml`` are each named by a module global, and each
writer read through the global, changed the mapping, and wrote through the global again. Tests
move those globals — every test points the machine config at a throwaway directory and puts it
back — so a write that straggles past the fixture lands somewhere else than its read came from.

It happened on 2026-11-06. A video-import worker thread outlived its test on ``main`` (which has
no session-wide config floor), read the test's ``storage.yaml``, and wrote it over the
developer's real one after the path was restored: every real video root on the machine was
replaced by one test alias. The demo experiments behind the user guides stopped opening with
``MissingVideoRootError`` a day later, and nothing in the run that did it failed.

The guard moves the module's global **during the read** — the moment a straggler meets — and
asserts the file it moved to is untouched while the file read from received the edit.
Data-driven over every writer, so one added later is covered by adding its row.
"""

from __future__ import annotations

import pytest
import yaml

from detail.core import yaml_load
from detail.storage import credentials, locations, remote_targets, roots


def _register_root(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    roots.register_root("video_roots", "new", folder)


def _clear_token(_tmp_path):
    credentials.clear_hf_token()


def _remove_target(_tmp_path):
    remote_targets.remove_remote_target("old")


#: (module, the global naming its file, what the file read from holds, the edit)
WRITERS = {
    "storage.yaml": (
        locations,
        "STORAGE_PATH",
        {"video_roots": {"old": "C:/videos"}},
        _register_root,
    ),
    "credentials.yaml": (
        credentials,
        "CREDENTIALS_PATH",
        {"huggingface": "hf_test", "other": "kept"},
        _clear_token,
    ),
    "remote_targets.yaml": (
        remote_targets,
        "REMOTE_TARGETS_PATH",
        {"schema_version": 1, "targets": {"old": {"host": "h"}, "keep": {"host": "k"}}},
        _remove_target,
    ),
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_an_edit_writes_back_to_the_file_it_read(writer, tmp_path, monkeypatch):
    module, attr, seeded, edit = WRITERS[writer]
    read_from = tmp_path / "test" / writer
    elsewhere = tmp_path / "real" / writer
    read_from.parent.mkdir()
    elsewhere.parent.mkdir()
    read_from.write_text(yaml.safe_dump(seeded), encoding="utf-8")
    real = yaml.safe_dump({"video_roots": {"two-animal": "C:/two"}, "huggingface": "x"})
    elsewhere.write_text(real, encoding="utf-8")
    monkeypatch.setattr(module, attr, read_from)
    if module is locations:
        monkeypatch.setattr(locations, "CONFIG_DIR", read_from.parent)

    parse = yaml_load.safe_load

    def parse_then_move(text, *args, **kwargs):
        # The fixture restoring the real path while the edit is between its read and write.
        monkeypatch.setattr(module, attr, elsewhere)
        return parse(text, *args, **kwargs)

    monkeypatch.setattr(yaml_load, "safe_load", parse_then_move)
    edit(tmp_path)

    assert (
        elsewhere.read_text(encoding="utf-8-sig") == real
    ), f"{writer}: an edit that read one file overwrote another"
    assert (
        yaml.safe_load(read_from.read_text(encoding="utf-8-sig")) != seeded
    ), f"{writer}: the edit never reached the file it read"
