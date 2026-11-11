"""dataset.yaml / model.yaml / track.yaml are written atomically (tmp + os.replace).

A worker subprocess rewrites these while the GUI reads them, so a plain truncating ``open("w")``
would let a reader parse an empty file — the atomic-write hard rule. A crash mid-serialize must
leave the previous file intact (a truncating write would leave an empty one)."""

from unittest.mock import patch

import pytest

from detail.storage import datasets as datasets_store
from detail.storage import models as models_store
from detail.trackers import _common as tracker_common


def test_write_dataset_yaml_atomic_keeps_original_on_crash(tmp_path):
    ddir = tmp_path / "dataset_001"
    ddir.mkdir()
    datasets_store.write_dataset_yaml(ddir, {"id": "dataset_001", "status": "complete"})
    with patch("detail.storage.datasets.yaml.dump", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            datasets_store.write_dataset_yaml(
                ddir, {"id": "dataset_001", "status": "running"}
            )
    assert datasets_store.read_dataset_yaml(ddir)["status"] == "complete"


def test_write_model_yaml_atomic_keeps_original_on_crash(tmp_path):
    mdir = tmp_path / "model_001"
    mdir.mkdir()
    models_store.write_model_yaml(mdir, {"id": "model_001", "status": "complete"})
    with patch("detail.storage.models.yaml.dump", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            models_store.write_model_yaml(
                mdir, {"id": "model_001", "status": "running"}
            )
    assert models_store.read_model_yaml(mdir)["status"] == "complete"


def test_write_track_yaml_atomic_keeps_original_on_crash(tmp_path):
    path = tmp_path / "track.yaml"
    tracker_common.write_track_yaml(path, {"schema_version": 1, "status": "complete"})
    with patch("detail.trackers._common.yaml.dump", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            tracker_common.write_track_yaml(
                path, {"schema_version": 1, "status": "running"}
            )
    assert (
        path.read_text(encoding="utf-8-sig") == "schema_version: 1\nstatus: complete\n"
    )
