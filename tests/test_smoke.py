import numpy as np
import zarr

import detail


def test_package_version():
    assert detail.__version__ == "0.1.0"


def test_experiment_dir_fixture(experiment_dir):
    assert (experiment_dir / "config.yaml").exists()


def test_detection_zarr_schema(sample_detection_zarr):
    root = zarr.open_group(str(sample_detection_zarr), mode="r")
    assert set(root.array_keys()) == {
        "frame_idx",
        "t_ns",
        "detection_idx",
        "class_name",
        "confidence",
        "bbox",
        "mask_rle",
    }
    assert root["frame_idx"].dtype == np.int32
    assert root["t_ns"].dtype == np.int64
    assert root["confidence"].dtype == np.float32
    assert root["bbox"].shape[1] == 4
    assert root["frame_idx"].chunks == (10000,)
    assert root["mask_rle"].chunks == (1000,)
