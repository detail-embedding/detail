"""Registry parity over QUERY_METHODS (§20), data-driven so a new method is covered the
day it is added.

The declarations here are read *before* anything runs — validate() enforces the exemplar
bounds, the module reads `sampling` to decide whether to draw a background, and it reads
`chunk_overlap_rows` to decide whether a long trajectory can be split. A method that gets one
of them wrong fails at launch for a user, not in CI, which is what these check.
"""

import numpy as np
import pytest
from pydantic import BaseModel

from detail.core.dataclasses import ExemplarBounds, NegativeSupport, SamplingRequest
from detail.core.registries import QUERY_METHODS
from detail.query_methods import import_query_methods

import_query_methods()

METHOD_IDS = sorted(QUERY_METHODS)


def _exemplars(*lengths, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, dim)).astype(np.float32) for n in lengths]


@pytest.fixture(params=METHOD_IDS)
def method(request):
    return QUERY_METHODS[request.param]()


def test_at_least_one_method_is_registered():
    """Guard the guard: every test below is parametrized over the registry, so an empty
    registry would make the whole file pass while checking nothing."""
    assert METHOD_IDS


def test_every_declared_module_actually_registers_a_method():
    """A module listed in ``QUERY_METHOD_MODULES`` that registers nothing is invisible here.

    ``register_query_method`` takes the id — ``@register_query_method("soft_knn")`` — so writing
    the bare ``@register_query_method`` binds the decorator *factory* to the class name and
    registers nothing at all. The module imports, the class is defined, every test below
    parametrizes over a registry that simply does not contain it, and the whole file passes
    while the method is unreachable. That happened while soft-kNN was being written, and
    nothing caught it until a script tried to instantiate the method.
    """
    import importlib

    from detail.query_methods import QUERY_METHOD_MODULES

    registered = {cls.__module__ for cls in QUERY_METHODS.values()}
    missing = [
        name
        for name in QUERY_METHOD_MODULES
        if importlib.import_module(name).__name__ not in registered
    ]
    assert not missing, (
        f"these query-method modules register nothing: {missing}. The decorator takes the "
        'method id — @register_query_method("<id>") — and a bare @register_query_method '
        "silently registers no method."
    )


def test_method_id_matches_its_registry_key():
    for method_id, cls in QUERY_METHODS.items():
        assert cls().method_id == method_id


def test_declares_a_config_model(method):
    assert issubclass(method.config_model, BaseModel)
    method.config_model()  # every parameter has a default, so a bare launch is possible


def test_declares_its_exemplar_bounds(method):
    bounds = method.exemplars
    assert isinstance(bounds, ExemplarBounds)
    assert bounds.min_clips >= 1
    assert bounds.min_total_rows >= 1


def test_exemplar_bounds_are_satisfiable(method):
    """A method declaring `max_clips=2` and `min_total_rows=10_000` accepts nothing, and
    finds out only when a user tries."""
    bounds = method.exemplars
    if bounds.max_clips is not None:
        assert bounds.min_clips <= bounds.max_clips
    if bounds.max_total_rows is not None:
        assert bounds.min_total_rows <= bounds.max_total_rows


def test_declares_negative_support(method):
    negatives = method.negatives
    assert isinstance(negatives, NegativeSupport)
    assert negatives.support in ("unusable", "optional", "required")


def test_declares_its_sampling_request(method):
    assert method.sampling is None or isinstance(method.sampling, SamplingRequest)
    if method.sampling is not None:
        assert method.sampling.size > 0
        assert method.sampling.scope in ("subject", "recording", "cohort")


def _prepare(method, exemplars):
    """``prepare`` the way the module does — with a background when the method asks for one.

    A method declaring a ``SamplingRequest`` is entitled to assume the module drew it
    (``pipeline/query.py::draw_background``), and refusing to score without one is correct:
    scoring against the exemplars alone would return a different, uncalibrated number that
    still looks like a confidence. So the parity tests below have to honour the declaration
    they already assert exists, rather than calling every method the same way.
    """
    sample = None
    if method.sampling is not None:
        rng = np.random.default_rng(5)
        dim = exemplars[0].shape[1]
        sample = rng.normal(size=(method.sampling.size, dim)).astype(np.float32)
    return method.prepare(exemplars, {}, sample=sample)


def test_answers_chunk_overlap_rows(method):
    state = _prepare(method, _exemplars(12, 9))
    overlap = method.chunk_overlap_rows(state, {})
    assert overlap is None or overlap >= 0


def test_confidence_is_canonical(method):
    """`[0, 1]` and **higher better**, for every method — which is what lets the cutoff axis,
    the audit stratification and the artifact's score column stay method-independent.

    Asserted on a *planted* trajectory so the candidate list cannot be empty: over pure noise
    a method that returns nothing satisfies "every confidence is in range" while saying
    nothing at all, and the direction half was never checked.
    """
    rng = np.random.default_rng(0)
    trajectory = rng.normal(size=(120, 8)).astype(np.float32)
    exemplars = _exemplars(10)
    trajectory[40:50] = exemplars[0]  # something to actually find
    state = _prepare(method, exemplars)
    result = method.search(state, trajectory, [(0, 120)], {})
    candidates = method.instances_at(result, 0.0, {})
    assert candidates, "nothing found, so nothing about confidence was tested"
    for candidate in candidates:
        assert 0.0 <= candidate.confidence <= 1.0
    # The native score must **agree** with the confidence ranking, in whichever direction the
    # method's own number runs: the removed `dtw`'s was a cosine distance (lower better) and
    # soft-kNN's is the vote itself (higher better, and identical to the confidence). Only
    # `confidence` is canonical — §20 makes it so precisely because a native score is the
    # method's own — so requiring a distance here would be requiring one method's convention
    # of everyone. What must not happen is the two disagreeing, which is the score-direction
    # hazard: an artifact column ordered against the ranking the method produced.
    scores = [
        c.score for c in sorted(candidates, key=lambda c: c.confidence, reverse=True)
    ]
    assert scores == sorted(scores) or scores == sorted(scores, reverse=True), (
        f"{method.method_id}: the native score does not order consistently with the "
        f"confidence it was returned beside: {scores}"
    )


def test_a_method_declares_its_shard_suffix(method):
    """§20 gives the shard extension to the method. Declared rather than implied: a writer
    whose serializer appends its own suffix would write one file and read another.

    Read as an attribute, not via `getattr(..., "")`: the default made a method that declares
    nothing pass this *and* round-trip through a file with no extension, hiding exactly the
    failure the docstring describes.
    """
    assert isinstance(method.shard_suffix, str)
    assert method.shard_suffix.startswith("."), method.shard_suffix


def test_the_protocol_gives_shard_suffix_no_default_to_inherit():
    """Guard the guard: with one method registered, the check above cannot see a method that
    forgets to declare a suffix — so assert the *mechanism* that makes forgetting impossible.

    A default on the ABC is what made the original guard vacuous: `getattr(m, "shard_suffix",
    "")` and an inherited `""` agreed, a method declaring nothing passed, and its shard
    round-tripped through a file with no extension. Both halves had to go.
    """
    from detail.core.protocols import QueryMethod

    assert "shard_suffix" in QueryMethod.__annotations__
    assert not hasattr(
        QueryMethod, "shard_suffix"
    ), "shard_suffix has a default on the ABC, so a method declaring none inherits it"


def test_serialization_round_trips(method, tmp_path):
    rng = np.random.default_rng(1)
    trajectory = rng.normal(size=(120, 8)).astype(np.float32)
    state = _prepare(method, _exemplars(10))
    result = method.search(state, trajectory, [(0, 120)], {})

    path = tmp_path / f"shard_{method.method_id}{method.shard_suffix}"
    method.serialize(result, path)
    assert path.is_file(), "serialize wrote somewhere other than the declared path"
    back = method.load(path)
    assert method.instances_at(back, 0.0, {}) == method.instances_at(result, 0.0, {})
