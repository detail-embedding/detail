"""``resolve_method_params`` drops a retired parameter and still refuses an unknown one.

Old runs saved ``projection_points`` / ``projection_layout`` in their ``method_params`` (the 2-D
layout the clustering screen no longer draws, §22), and ``seed`` for methods that only used it to
seed that layout. "Start from a previous run" feeds those back
through here, so they must resolve; a typo must still be refused, because that refusal is the
whole reason this function exists.
"""

import pytest

from detail.cluster_methods import (
    import_pose_cluster_methods,
    import_temporal_cluster_methods,
)
from detail.configs.method_params import RETIRED_PARAMS, resolve_method_params
from detail.core.registries import POSE_CLUSTER_METHODS, TEMPORAL_CLUSTER_METHODS

import_pose_cluster_methods()
import_temporal_cluster_methods()

METHODS = {**POSE_CLUSTER_METHODS, **TEMPORAL_CLUSTER_METHODS}


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_an_old_runs_retired_params_resolve_and_are_dropped(method_id):
    method = METHODS[method_id]
    declared = set(method.config_model.model_fields)
    old = {"projection_points": 50000, "projection_layout": "umap", "seed": 7}
    resolved = resolve_method_params(method, old)
    assert not (RETIRED_PARAMS - declared) & set(
        resolved
    ), "undeclared retired names dropped"
    kept = {k: v for k, v in old.items() if k in declared}
    assert resolved == resolve_method_params(method, kept)


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_a_genuinely_unknown_key_is_still_refused(method_id):
    with pytest.raises(ValueError, match="has no parameter"):
        resolve_method_params(
            METHODS[method_id], {"projection_layout": "pca", "durationmin": 2}
        )


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_a_retired_name_a_method_still_declares_keeps_its_meaning(method_id):
    """``seed`` is retired for dtw_finch only (and was for finch, until pose clustering was
    removed): codebook_viterbi still takes one, and dropping it there would silently reseed
    its fits."""
    method = METHODS[method_id]
    if "seed" in method.config_model.model_fields:
        assert resolve_method_params(method, {"seed": 5})["seed"] == 5
    assert not {"projection_points", "projection_layout"} & set(
        method.config_model.model_fields
    )
