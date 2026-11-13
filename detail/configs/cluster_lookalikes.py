"""Find look-alike clusters — which clusters in different branches of a clustering run's tree
are one behaviour, as one recorded analysis job (§22 "Look-alike clusters").

The single source for each parameter's type, bounds, default, help and label: the GUI form and
CLI ``--help`` read this metadata (DESIGN_PLAN §6.5). Each pair it tests is a cluster profile of
one cluster against the other (``compare: node``) with position and facing left out, so the
profile's own settings that make sense for a pair are carried by name (``PROFILE_FIELDS``); the
ones a pair sets itself — which clusters, and that place is left out — are not settings here.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from detail.analysis import explainers
from detail.configs.cluster_profile import ClusterProfileConfig

#: Members measured per side of each pair, by default: a fixed count, as Profile clusters
#: uses, so a pair is the same numbers on any machine.
DEFAULT_PER_SIDE = 50

#: ``pairs``' choices. ``all``: every pair of branch tops. ``labelled``: each branch top a label
#: covers against every other branch top (``api/cluster_lookalikes.labelled_pairs``).
PAIRS_ALL = "all"
PAIRS_LABELLED = "labelled"
PAIR_CHOICES = (PAIRS_ALL, PAIRS_LABELLED)

#: The profile settings each pair sets itself rather than taking from the job.
SET_PER_PAIR = ("compare", "against", "leave_out_place")
#: Settings a saved config may still carry from before suggested names were cut, when pairs of
#: branch tops were chosen by how alike their suggested names were: dropped on load (pydantic
#: ignores an unknown key), so an old run's "start from" and its record still read.
RETIRED_FIELDS = ("min_name_overlap",)
#: ``pairs`` values a saved config may still carry, and what each now reads as: ``names`` (pairs
#: whose suggested names shared words) went with suggested names, and reads as the default.
RETIRED_PAIRS = {"names": PAIRS_ALL}


class ClusterLookalikesConfig(BaseModel):
    """One job: compare pairs of one clustering run's branch tops, body measures only."""

    source_run_id: str = Field(
        ...,
        description="The temporal-clustering run whose clusters to compare (e.g. "
        "an_17). Each pair's comparison is saved beside that run, where its Interpret tab "
        "groups them.",
        json_schema_extra={"scope": "selection", "label": "Clustering run"},
    )
    pairs: Literal[PAIR_CHOICES] = Field(
        default=PAIRS_ALL,
        description="Which pairs of branch tops (the two halves of every split where both "
        "are at least the run's minimum cluster time) to compare. 'all': every pair, which "
        "is compute intensive (many hours on a large run). 'labelled': each branch top a "
        "label covers against every other branch top, far fewer pairs; it finds unlabelled "
        "clusters that look like ones already named. Two clusters under one label are not "
        "compared.",
        json_schema_extra={"label": "Pairs to compare"},
    )
    skip_existing: bool = Field(
        default=True,
        description="Skip a pair already compared with these settings (in either order), "
        "including one compared by hand in the Cluster profile window. Off: every pair is "
        "compared again.",
        json_schema_extra={"label": "Skip compared pairs"},
    )
    explainer: Literal[tuple(explainers.EXPLAINERS)] = Field(
        default=explainers.DEFAULT_EXPLAINER,
        description=ClusterProfileConfig.model_fields["explainer"].description,
        json_schema_extra={"label": "Explainer"},
    )
    n_per_side: int | None = Field(
        default=DEFAULT_PER_SIDE,
        ge=explainers.MIN_ROWS_PER_SIDE,
        le=20_000,
        description="Measure exactly this many members of each cluster of a pair, so a "
        "comparison does not depend on how fast this computer reads tracks. Unset: as many "
        "as fit in the time budget instead (members per side).",
        json_schema_extra={"label": "Fixed count per side"},
    )
    time_budget_s: float = Field(
        default=30.0,
        gt=0,
        le=1800,
        description=ClusterProfileConfig.model_fields["time_budget_s"].description,
        json_schema_extra={"label": "Time budget", "unit": "s", "advanced": True},
    )
    margin: float = Field(
        default=explainers.DEFAULT_MARGIN,
        ge=0.0,
        le=0.5,
        description="How far held-out balanced accuracy must sit above the shuffled floor "
        "for body measures to tell a pair apart (0-1). A pair below it looks alike.",
        json_schema_extra={"label": "Separation margin"},
    )
    seed: int = Field(
        default=0,
        ge=0,
        description=ClusterProfileConfig.model_fields["seed"].description,
        json_schema_extra={"label": "Seed", "advanced": True},
    )

    @field_validator("pairs", mode="before")
    @classmethod
    def _a_retired_choice_reads_as_its_replacement(cls, value):
        """An older run's ``pairs: names`` (from "start from" or its record) still loads."""
        return RETIRED_PAIRS.get(value, value) if isinstance(value, str) else value


#: The profile settings the job passes to every pair, by name: every profile field but the
#: ones a pair sets itself (``tests/configs/test_cluster_lookalikes.py`` holds the split).
PROFILE_FIELDS = tuple(
    name for name in ClusterProfileConfig.model_fields if name not in SET_PER_PAIR
)
