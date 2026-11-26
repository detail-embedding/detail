"""Recording visualization recipes and export settings refuse what cannot be drawn (§22)."""

import pydantic
import pytest

from detail.configs.recording_visualization import (
    MAX_COLOURED,
    RecordingVisualizationConfig,
    RecordingVisualizationExport,
    speed_timing,
)


def test_a_recipe_defaults_to_the_full_recording_named_with_location():
    r = RecordingVisualizationConfig(recording_id="recording_001")
    assert (r.start_s, r.end_s, r.clusters, r.location) == (None, None, "named", True)


@pytest.mark.parametrize(
    "fields, fragment",
    [
        ({"start_s": 10.0}, "give both start_s and end_s"),
        ({"start_s": 10.0, "end_s": 5.0}, "must be after start_s"),
        ({"clusters": "cut"}, "needs cut_height"),
        (
            {
                "clusters": "cut",
                "cut_height": 0.3,
                "cut_included": list(range(1, MAX_COLOURED + 2)),
            },
            "at most 8",
        ),
        ({"clusters": "cut", "cut_height": 0.3, "cut_included": [0]}, "start at 1"),
        ({"cut_height": 0.3}, "belong to clusters 'cut'"),
    ],
)
def test_an_impossible_recipe_is_refused(fields, fragment):
    with pytest.raises(pydantic.ValidationError, match=fragment):
        RecordingVisualizationConfig(recording_id="recording_001", **fields)


def test_export_height_keeps_16_by_9_in_even_pixels():
    assert RecordingVisualizationExport().height == 1080
    assert RecordingVisualizationExport(width=1280).height == 720
    assert RecordingVisualizationExport(width=1000).height % 2 == 0


def test_export_needs_a_background():
    with pytest.raises(pydantic.ValidationError):
        RecordingVisualizationExport(themes=[])


@pytest.mark.parametrize(
    "speed, source, expected",
    [
        ("real-time", 100.0, (50.0, 1.0)),
        ("real-time", 30.0, (30.0, 1.0)),
        ("every-frame", 100.0, (100.0, 1.0)),
        ("half-speed", 100.0, (50.0, 0.5)),
    ],
)
def test_speed_is_relative_to_the_source_rate(speed, source, expected):
    assert speed_timing(speed, source) == expected
