"""The comparison config refuses the two shapes that render as plausible charts (§24)."""

import pydantic
import pytest

from detail.configs.comparison import ComparisonConfig, WindowSpec

_NS = 1_000_000_000


def test_aggregate_to_has_no_default():
    """Pooling recordings and averaging within animal are both legitimate and answer different
    questions, so the tool does not choose. Omitting it is an error, not a default."""
    with pytest.raises(pydantic.ValidationError):
        ComparisonConfig(source_run_id="an_01", group_by=["drug"])


def test_an_empty_path_is_a_choice_not_an_absence():
    cfg = ComparisonConfig(source_run_id="an_01", group_by=["drug"], aggregate_to=[])
    assert cfg.aggregate_to == []


def test_a_column_cannot_both_group_and_aggregate():
    """Every group would hold exactly one observation, and the comparison would render as a
    scatter of singletons with no spread anywhere."""
    with pytest.raises(
        pydantic.ValidationError, match="both group_by and aggregate_to"
    ):
        ComparisonConfig(
            source_run_id="an_01", group_by=["animal"], aggregate_to=["animal"]
        )


def test_group_by_cannot_be_empty():
    with pytest.raises(pydantic.ValidationError):
        ComparisonConfig(source_run_id="an_01", group_by=[], aggregate_to=[])


@pytest.mark.parametrize(
    "spec, expected",
    [
        ({"kind": "whole"}, 1),
        ({"kind": "absolute", "every_s": 10.0}, 3),
        ({"kind": "normalized", "parts": 4}, 4),
        ({"kind": "sliding", "length_s": 10.0, "step_s": 5.0}, 5),
    ],
)
def test_window_specs_resolve_against_a_recording_span(spec, expected):
    assert len(WindowSpec(**spec).windows_for(0, 25 * _NS)) == expected


def test_window_durations_are_seconds_and_convert_to_nanoseconds():
    """Declared in seconds because the same *count* is a different duration on every recording —
    the unit convention `tests/configs/test_unit_conventions.py` enforces."""
    first = WindowSpec(kind="absolute", every_s=10.0).windows_for(0, 25 * _NS)[0]
    assert first.end_t_ns - first.start_t_ns == 10 * _NS


def test_the_record_carries_the_resolved_config():
    cfg = ComparisonConfig(
        source_run_id="an_01", group_by=["drug"], aggregate_to=["animal"], seed=7
    )
    record = cfg.as_record()
    assert record["aggregate_to"] == ["animal"]
    assert record["seed"] == 7
    assert record["windows"]["kind"] == "whole"


# --------------------------------------------------------------------------- #
# the design is resolved, never configured (§24 *Designs*)
# --------------------------------------------------------------------------- #


def test_there_is_no_design_field_the_design_is_resolved():
    """Decision 2026-11-07: the user chooses the variables and the tool resolves the design —
    between, paired or trend — from what the column does within the unit. A `design` field would
    be an override nobody could state the meaning of; `levels` is the one lever."""
    assert "design" not in ComparisonConfig.model_fields
    assert "levels" in ComparisonConfig.model_fields


def test_levels_need_one_column_and_two_distinct_values():
    with pytest.raises(pydantic.ValidationError, match="one column"):
        ComparisonConfig(
            source_run_id="an_01",
            group_by=["drug", "sex"],
            aggregate_to=[],
            levels=["veh", "fent"],
        )
    for levels in (["veh"], ["veh", "veh"], ["a", "b", "c"]):
        with pytest.raises(pydantic.ValidationError, match="two distinct"):
            ComparisonConfig(
                source_run_id="an_01", group_by=["drug"], aggregate_to=[], levels=levels
            )


def test_the_record_carries_every_field_including_the_new_ones():
    cfg = ComparisonConfig(
        source_run_id="an_01",
        group_by=["drug"],
        aggregate_to=["animal"],
        source="nodes",
        levels=["veh", "fent"],
        min_units_fraction=0.75,
        presence_s=2.0,
        pseudo_count_s=0.5,
        suppress_nesting=False,
    )
    record = cfg.as_record()
    assert set(ComparisonConfig.model_fields) <= set(record), sorted(
        set(ComparisonConfig.model_fields) - set(record)
    )
    assert record["source"] == "nodes"
    assert record["levels"] == ["veh", "fent"]
    assert record["min_units_fraction"] == 0.75
    assert record["presence_s"] == 2.0 and record["pseudo_count_s"] == 0.5
    assert record["suppress_nesting"] is False


def test_min_units_fraction_is_a_share_of_the_cohort():
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(pydantic.ValidationError):
            ComparisonConfig(
                source_run_id="an_01",
                group_by=["drug"],
                aggregate_to=[],
                min_units_fraction=bad,
            )
    assert ComparisonConfig.model_fields["min_units_fraction"].default == 0.5
