"""The cluster-method contract that outlives any one registry's members.

This file swept every registered **pose**-cluster method against the §22 contract (config model,
noise flag, shard suffix, granularity, round trip, purity of a fit). Pose clustering and its two
methods were removed (``docs/removed_capabilities.md``, section 12); its registry stays, empty, and
the per-method sweep went with its members — restore it from the tag with them. What is left holds
for the protocol and for every method still registered, pose or temporal.
"""

import pytest

from detail.cluster_methods import import_pose_cluster_methods
from detail.core.registries import POSE_CLUSTER_METHODS

import_pose_cluster_methods()


def test_the_protocol_gives_shard_suffix_no_default_to_inherit():
    """A method that forgets to declare one must fail this suite rather than silently inherit ""."""
    from detail.core.protocols import PoseClusterMethod

    assert "shard_suffix" not in vars(PoseClusterMethod)


def _every_method():
    """Pose and temporal methods alike: the rule is about the seam, not which module it serves."""
    from detail.cluster_methods import import_temporal_cluster_methods
    from detail.core.registries import TEMPORAL_CLUSTER_METHODS

    import_temporal_cluster_methods()
    return {**POSE_CLUSTER_METHODS, **TEMPORAL_CLUSTER_METHODS}


@pytest.mark.parametrize("method_id", sorted(_every_method()))
def test_no_label_can_reach_a_method(method_id):
    """The leakage rule made structural rather than conventional (§22).

    `tests/pipeline/test_label_leakage.py` permutes a ground-truth stream and asserts predictions
    do not move. That shape does not apply here, because there is nothing to permute: a pose method
    is handed trajectories and config and has no parameter through which ground truth could enter.
    So the guard is that this stays true — a future method (or a future signature change) that
    accepts a stream, a label array or an annotation would make the leak *expressible*, and the
    rule's whole point is that it should not be.

    Named parameters rather than a source scan because the protocol is the seam: anything a caller
    could pass has to appear here.
    """
    import inspect

    forbidden = ("label", "stream", "truth", "annotation", "behavior", "ground")
    # `instances_for` is the redesign's seam (§22): it takes a partition of the method's own tree
    # as node ids, never a name, and the parameter is named for what it is so this scan can say so.
    for name in (
        "fit",
        "instances_at",
        "instances_for",
        "member_leaves",
        "rank_members",
        "node_separations",
    ):
        signature = inspect.signature(getattr(_every_method()[method_id], name))
        for parameter in signature.parameters:
            assert not any(word in parameter.lower() for word in forbidden), (
                f"{method_id}.{name} takes {parameter!r}, through which a label could reach the "
                f"method — see CLAUDE.md, 'Labels score a method; they never build one'"
            )
