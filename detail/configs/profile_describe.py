"""Profile clusters — a cluster profile over many nodes of a clustering run, as one recorded
analysis job (§22 "Profile clusters"). Its module id is ``profile_describe``, the id it has had
since it also ran a model's description of each cluster, so the runs made then still open.

The single source for each parameter's type, bounds, default, help and label: the GUI form and
CLI ``--help`` read this metadata (DESIGN_PLAN §6.5). It **is** the per-cluster config it runs —
``ClusterProfileConfig``, inherited rather than restated, so a field it gains reaches this job,
its record and both surfaces by name — plus the things only a batch has: which run, which of
its clusters, and whether finished clusters are skipped.

Where the job runs (``compute``) is a ``ComputeTarget`` and stays outside this model, as every
module's compute input does: this machine, or a remote target (§6 "Remote analysis").

A config saved before the description step was cut may still carry its settings
(``RETIRED_FIELDS``); they are dropped on load (pydantic ignores an unknown key, and the
module's ``validate`` drops them by name), so an old run's record and "start from" still read.
"""

from typing import Literal

from pydantic import Field

from detail.analysis import explainers
from detail.configs.cluster_profile import COMPARE_REST
from detail.configs.cluster_profile import ClusterProfileConfig

#: ``clusters``' choices. ``floor``: the nodes whose member time is at or above the run's
#: ``min_cluster_s`` (the floor its clips are built at). ``all``: every node with members.
#: ``leaves``: the tree's leaves with members. ``ids``: the nodes named in ``node_ids``.
CLUSTERS_FLOOR = "floor"
CLUSTERS_ALL = "all"
CLUSTERS_LEAVES = "leaves"
CLUSTERS_IDS = "ids"
CLUSTER_CHOICES = (CLUSTERS_FLOOR, CLUSTERS_ALL, CLUSTERS_LEAVES, CLUSTERS_IDS)

#: Members measured per side of each profile, by default: a fixed count rather than the time
#: budget, so a profile is the same numbers on a slower or a remote machine. 50 is about what
#: the 30 s budget measured on the 2026-10-28 workstation (34, 44 and 73 per side on the
#: three real profiles of clusters large enough to fill it: an_11 nodes 12 and 19, an_10
#: node 877).
DEFAULT_PER_SIDE = 50


#: Settings an older run of this job carried for the description step it no longer has.
RETIRED_FIELDS = (
    "backend",
    "n_clips",
    "batch_size",
    "blind",
    "use_setup",
    "use_behaviours",
    "allow_external",
    "model",
    "vision_model",
    "num_ctx",
)


class ProfileDescribeConfig(ClusterProfileConfig):
    """One job: profile each chosen node of one clustering run."""

    source_run_id: str = Field(
        ...,
        description="The temporal-clustering run whose clusters to profile (e.g. an_10). "
        "Each cluster's profile is saved beside that run, where its Interpret tab shows it.",
        json_schema_extra={"scope": "selection", "label": "Clustering run"},
    )
    clusters: Literal[CLUSTER_CHOICES] = Field(
        default=CLUSTERS_FLOOR,
        description="Which clusters. 'floor': the ones with at least the run's minimum cluster time (the "
        "floor its clips are built at). 'all': every node with members. 'leaves': the "
        "tree's leaves. 'ids': the nodes listed below. The root is never one: it has nothing "
        "to be compared against.",
        json_schema_extra={"label": "Clusters"},
    )
    node_ids: list[int] = Field(
        default_factory=list,
        description="The nodes to run when Clusters is 'ids', as the tree numbers them.",
        json_schema_extra={"label": "Node ids", "advanced": True},
    )
    skip_existing: bool = Field(
        default=True,
        description="Skip a cluster whose profile was already made with these settings. "
        "Off: every cluster is profiled again (the old profile is replaced).",
        json_schema_extra={"label": "Skip finished clusters"},
    )
    # Restated only where the batch differs from one cluster: every cluster against one named
    # other cluster is not a batch question, so 'node' (and its `against`) is one cluster's.
    compare: Literal["rest", "sibling"] = Field(
        default=COMPARE_REST,
        description="What each cluster is told apart from. 'rest' is every other member of "
        "the run: what makes it unusual overall. 'sibling' is the cluster it was merged with "
        "in the tree: what splits the two.",
        json_schema_extra={"label": "Compare against"},
    )
    n_per_side: int | None = Field(
        default=DEFAULT_PER_SIDE,
        ge=explainers.MIN_ROWS_PER_SIDE,
        le=20_000,
        description="Measure exactly this many members of each cluster and as many of the "
        "comparison, so a profile does not depend on how fast this computer reads tracks. "
        "Unset: as many as fit in the time budget instead (members per side).",
        json_schema_extra={"label": "Fixed count per side"},
    )


#: The fields each per-cluster call takes, read from the profile config rather than listed: a
#: field it gains is passed through by name.
PROFILE_FIELDS = tuple(ClusterProfileConfig.model_fields)
