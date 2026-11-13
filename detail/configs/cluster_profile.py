"""Cluster profile config — what sets one cluster apart, as measures, with how far to trust it.

A profile samples one clustering node's members and a comparison set, measures both with the
registered descriptors (``analysis/descriptors.py``), and fits an explainer
(``analysis/explainers.py``) scored on held-out recordings beside a shuffled floor. Not a
registry module: it is asked for from the Interpretation tab, the clip window and the montage
(GUI), ``Experiment.profile_cluster`` (API) and ``analysis profile`` (CLI), and it writes a
sibling folder under the clustering run rather than a run of its own.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from detail.analysis import explainers

COMPARE_REST = "rest"
COMPARE_SIBLING = "sibling"
COMPARE_NODE = "node"


class ClusterProfileConfig(BaseModel):
    """One profile of one node."""

    compare: Literal["rest", "sibling", "node"] = Field(
        default=COMPARE_REST,
        description="What the cluster is told apart from. 'rest' is every other member of "
        "the run: what makes it unusual overall. 'sibling' is the cluster it was merged with "
        "in the tree: what splits the two, usually the more useful question for naming. "
        "'node' is any other cluster you name (against): whether two clusters from "
        "different branches are the same behaviour.",
        json_schema_extra={"label": "Compare against"},
    )
    against: int | None = Field(
        default=None,
        ge=0,
        description="The other cluster, by node id, when comparing against one cluster. "
        "Neither may hold the other: a cluster inside this one is part of it, and one this "
        "cluster sits inside is best asked as its sibling.",
        json_schema_extra={"label": "Other cluster"},
    )
    leave_out_place: bool = Field(
        default=True,
        description="Measure where the animal is and which way it faces, but leave those "
        "out of the fit, so the answer is about what the body does. Rearing toward the left "
        "and toward the back then read as one behaviour.",
        json_schema_extra={"label": "Leave out position and facing"},
    )
    explainer: Literal[tuple(explainers.EXPLAINERS)] = Field(
        default=explainers.DEFAULT_EXPLAINER,
        description="How the measures are weighed. The default, a random forest explained "
        "with SHAP, falls back to permutation importance when shap is not installed, and the "
        "result says which ran.",
        json_schema_extra={"label": "Explainer"},
    )
    time_budget_s: float = Field(
        default=30.0,
        gt=0,
        le=1800,
        description="Measure as many members as fit in about this many seconds. Ignored "
        "when a fixed count is set.",
        json_schema_extra={"label": "Time budget", "unit": "s"},
    )
    n_per_side: int | None = Field(
        default=None,
        ge=explainers.MIN_ROWS_PER_SIDE,
        le=20_000,
        description="Measure exactly this many members of the cluster and as many of the "
        "comparison, instead of a time budget (members per side).",
        json_schema_extra={"label": "Fixed count per side", "advanced": True},
    )
    margin: float = Field(
        default=explainers.DEFAULT_MARGIN,
        ge=0.0,
        le=0.5,
        description="How far held-out balanced accuracy must sit above the shuffled floor for "
        "the measures to count as separating the cluster (0-1, balanced accuracy). Below it, "
        "the explanation is shown folded, as not trustworthy.",
        json_schema_extra={"label": "Separation margin"},
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Seeds the member draw and the fit, so the same profile is the same "
        "numbers tomorrow.",
        json_schema_extra={"label": "Seed", "advanced": True},
    )

    @model_validator(mode="after")
    def _against_goes_with_node(self):
        if self.compare == COMPARE_NODE and self.against is None:
            raise ValueError(
                "compare 'node' needs the other cluster: set against to its node id"
            )
        if self.compare != COMPARE_NODE and self.against is not None:
            raise ValueError(
                f"against names another cluster, which only compare 'node' uses; "
                f"compare is {self.compare!r}"
            )
        return self
