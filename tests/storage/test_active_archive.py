"""Which finished runs Active Processes has dismissed: the experiment-scoped store."""

import yaml

from detail.storage import active_archive


def test_no_file_archives_nothing(tmp_path):
    assert active_archive.archived_ids(tmp_path) == {}


def test_archive_then_read_back(tmp_path):
    active_archive.archive(tmp_path, "distill", "distill_002")

    assert active_archive.archived_ids(tmp_path) == {"distill": {"distill_002"}}


def test_archiving_is_idempotent(tmp_path):
    active_archive.archive(tmp_path, "distill", "distill_002")
    active_archive.archive(tmp_path, "distill", "distill_002")

    assert active_archive.archived_ids(tmp_path) == {"distill": {"distill_002"}}


def test_archiving_accumulates_within_a_kind(tmp_path):
    active_archive.archive(tmp_path, "model", "model_001")
    active_archive.archive(tmp_path, "model", "model_002")

    assert active_archive.archived_ids(tmp_path) == {
        "model": {"model_001", "model_002"}
    }


def test_archiving_keeps_kinds_separate(tmp_path):
    active_archive.archive(tmp_path, "distill", "distill_002")
    active_archive.archive(tmp_path, "model", "model_001")

    assert active_archive.archived_ids(tmp_path) == {
        "distill": {"distill_002"},
        "model": {"model_001"},
    }


def test_schema_version_is_the_first_key(tmp_path):
    active_archive.archive(tmp_path, "distill", "distill_002")

    text = (tmp_path / active_archive._FILENAME).read_text(encoding="utf-8-sig")
    assert text.startswith("schema_version: 1")
    assert yaml.safe_load(text)["schema_version"] == 1
