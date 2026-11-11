"""Comparison config (§24) — the single source for each parameter's type, bounds, help and label.

A comparison filters recordings, groups them by columns of ``metadata.csv`` (§11), and summarizes
§23 measures over one §19 behavior artifact. Descriptive only: no tests, no corrections, no models
(§24) — the tidy ``stats.csv`` is the handoff to a mixed model elsewhere.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from detail.configs.windows import WindowSpec


class ComparisonConfig(BaseModel):
    """One comparison over one behavior artifact."""

    source_run_id: str = Field(
        ...,
        description="The analysis run whose behavior output is compared (§19).",
        json_schema_extra={"scope": "selection", "label": "Source run"},
    )
    source_snapshot: str | None = Field(
        default=None,
        description="Which pinned copy of that run's artifact to compare (§21). Left unset, the "
        "comparison pins the run's current artifact and records the id it got, so its numbers "
        "stay reproducible after the run is reopened and re-cut.",
        json_schema_extra={"label": "Source snapshot"},
    )
    group_by: list[str] = Field(
        ...,
        min_length=1,
        description="Columns to compare across: any metadata.csv column, plus subject or window_id.",
        json_schema_extra={"label": "Group by"},
    )
    aggregate_to: list[str] = Field(
        ...,
        description="Columns to average through before grouping: the unit of analysis. Empty "
        "means each recording is an observation; ['animal'] means each animal is. Both are "
        "legitimate and they answer different questions, so there is no default.",
        json_schema_extra={"label": "Aggregate to"},
    )
    recordings: list[str] | None = Field(
        None,
        description="Restrict to these recordings; unset compares every recording in the artifact.",
        json_schema_extra={"label": "Recordings"},
    )
    measures: list[str] | None = Field(
        None,
        description="Which §23 measures to compute; unset computes every registered measure.",
        json_schema_extra={"advanced": True, "label": "Measures"},
    )
    windows: WindowSpec = Field(
        default_factory=WindowSpec,
        description="How to slice each recording in time: the whole span, fixed durations, "
        "equal fractions, or a sliding window. window_id then becomes a column to compare "
        "across, which is how early-versus-late becomes the same operation as drug-versus-"
        "vehicle.",
        json_schema_extra={"advanced": "Windows", "label": "Windows"},
    )
    draws: int = Field(
        200,
        ge=1,
        description="Permutation draws behind the difference ranking's floor.",
        json_schema_extra={"advanced": True, "label": "Floor draws", "ui_max": 100_000},
    )
    seed: int = Field(
        0,
        ge=0,
        description="Seed for the permutation floor, so a ranking reproduces.",
        json_schema_extra={"advanced": True, "label": "Seed", "ui_max": 2**31 - 1},
    )

    @model_validator(mode="after")
    def _check_disjoint(self) -> "ComparisonConfig":
        """A column cannot both define the groups and be averaged through.

        Grouping by ``animal`` while aggregating to ``animal`` asks for one observation per animal
        *and* one group per animal — every group would hold exactly one point, and the comparison
        would render as a scatter of singletons with no spread anywhere. Refused rather than drawn.
        """
        overlap = sorted(set(self.group_by) & set(self.aggregate_to))
        if overlap:
            raise ValueError(
                f"{overlap} appears in both group_by and aggregate_to: every group would hold "
                f"exactly one observation. Group by it, or aggregate to it — not both."
            )
        return self

    def as_record(self) -> dict[str, Any]:
        """The fields worth persisting, resolved — what the record carries (Tenet 9)."""
        return {
            "source_run_id": self.source_run_id,
            "source_snapshot": self.source_snapshot,
            "group_by": list(self.group_by),
            "aggregate_to": list(self.aggregate_to),
            "recordings": list(self.recordings) if self.recordings else None,
            "measures": list(self.measures) if self.measures else None,
            "windows": self.windows.model_dump(),
            "draws": self.draws,
            "seed": self.seed,
        }
