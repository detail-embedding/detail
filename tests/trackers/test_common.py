import numpy as np
import pytest

from detail.trackers import _common


def _data():
    return {
        "frame_idx": np.array([0, 0, 1, 1], dtype=np.int32),
        "class_name": np.array(
            ["black mouse", "white mouse", "black mouse", "white mouse"]
        ),
        "confidence": np.array([0.9, 0.8, 0.9, 0.8], dtype=np.float32),
    }


def test_filter_by_class_keeps_only_the_requested_class():
    filtered = _common.filter_by_class(_data(), "white mouse")
    assert filtered["class_name"].tolist() == ["white mouse", "white mouse"]
    assert filtered["confidence"].tolist() == pytest.approx([0.8, 0.8])
    assert filtered["frame_idx"].tolist() == [0, 1]


def test_filter_by_class_none_is_a_true_passthrough():
    data = _data()
    filtered = _common.filter_by_class(data, None)
    assert filtered is data


def test_filter_by_class_unknown_class_yields_empty_arrays():
    filtered = _common.filter_by_class(_data(), "orange mouse")
    assert filtered["class_name"].tolist() == []
    assert filtered["frame_idx"].tolist() == []
