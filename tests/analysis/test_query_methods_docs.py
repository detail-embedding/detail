"""Anti-drift for the Querying methods doc page (scripts/gen_docs_querying_methods.py).

The page's method table and figure lanes come from ``analysis.query_methods_docs``, keyed off
``QueryEvalConfig.method``. These tests pin that the config, the implemented matchers, and the
described methods name the same set, so adding a method without documenting it fails here rather
than shipping a silent gap or an unrendered lane.
"""

import pathlib

from detail.analysis import retrieval
from detail.analysis.query_methods_docs import (
    MATCH_LENGTH_EXAMPLE,
    MATCH_LENGTH_PER_FRAME,
    describe_query_controls,
    describe_query_methods,
    query_method_ids,
)
from detail.configs.query_eval import _PERFRAME_METHODS, QueryEvalConfig
from detail.core.registries import QUERY_METHODS
from detail.query_methods import import_query_methods


def test_config_methods_match_the_implementation():
    # The windowed matchers (retrieval curve fns) plus the eval-only per-frame methods;
    # frame_span is gone (its top_percent cutoff was the wrong lever it was removed for).
    implemented = set(retrieval._CURVE_FNS) | set(_PERFRAME_METHODS)
    assert set(query_method_ids()) == implemented
    assert "frame_span" not in query_method_ids()


def test_every_registered_query_method_is_a_known_method_id():
    """§20's `QUERY_METHODS` is a different registry at a different level — method objects,
    not curve fns — but a method id it invents that no surface knows would be unreachable.

    Data-driven over the registry, so a method added there is covered the day it lands.
    """
    import_query_methods()
    unknown = set(QUERY_METHODS) - set(query_method_ids())
    assert not unknown, (
        f"query methods registered under ids no config offers: {sorted(unknown)}. "
        "Either add the id to the method Literal or register it under an existing one."
    )


def test_describe_covers_every_method_in_order():
    described = [doc.method_id for doc in describe_query_methods()]
    assert described == query_method_ids()
    assert all(doc.summary and doc.description for doc in describe_query_methods())


def test_no_method_is_variable_length():
    # The per-frame methods produce no spans; the windowed matchers stay near the example
    # length. frame_span was the only variable-length method and it is gone.
    variable = {
        doc.method_id for doc in describe_query_methods() if doc.variable_length
    }
    assert variable == set()


def test_a_per_frame_method_is_not_described_as_example_length():
    """The page's intro says a per-frame method's matches are not tied to the example's length,
    and the table under it said "≈ example" for soft_knn. Each kind of method gets its own cell.
    """
    by_id = {doc.method_id: doc for doc in describe_query_methods()}
    assert _PERFRAME_METHODS, "no per-frame method left to check"
    for method_id in _PERFRAME_METHODS:
        assert by_id[method_id].match_length == MATCH_LENGTH_PER_FRAME, method_id
    for method_id in retrieval._CURVE_FNS:
        assert by_id[method_id].match_length == MATCH_LENGTH_EXAMPLE, method_id


def test_the_generator_renders_the_match_length_it_is_given():
    """The cell comes from `QueryMethodDoc.match_length`, not a second copy in the script, which
    is where the "≈ example" for every non-variable method was decided."""
    script = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "gen_docs_querying_methods.py"
    ).read_text(encoding="utf-8-sig")
    assert "method.match_length" in script
    assert MATCH_LENGTH_EXAMPLE not in script


def test_controls_are_real_config_fields_with_defaults():
    controls = describe_query_controls()
    for control in controls:
        assert control.name in QueryEvalConfig.model_fields, control.name
        assert control.default.startswith("`"), control.name  # a rendered default value
    # the default is read from the config, not hard-coded on the page
    band = next(c for c in controls if c.name == "band_s")
    assert band.default == f"`{QueryEvalConfig.model_fields['band_s'].default}` s"


def test_curve_fns_and_perframe_methods_are_disjoint():
    """A method must be handled by exactly one branch of evaluate_query. A method in BOTH
    _CURVE_FNS and _PERFRAME_METHODS would silently take the per-frame path."""
    from detail.configs.query_eval import _PERFRAME_METHODS

    assert set(retrieval._CURVE_FNS).isdisjoint(_PERFRAME_METHODS)
