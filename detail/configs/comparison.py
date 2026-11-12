"""Comparison config (§24) — the single source for each parameter's type, bounds, help and label.

A comparison filters recordings, groups them by columns of ``metadata.csv`` (§11), and summarizes
§23 measures over one §19 behavior artifact — or, with ``source: nodes``, over every candidate
node of a clustering run's node cache (§24 *The node source*). Descriptive only: no tests, no
corrections, no models (§24) — the tidy ``stats.csv`` is the handoff to a mixed model elsewhere.

**There is no ``design`` field.** The design — between units, paired within them, or a trend —
is resolved from what the chosen column does within the unit when the comparison runs
(``compare.resolve_design``), and written to the record with the sentence every surface prints.
The user chooses the variables; naming two ``levels`` is the one lever that changes what runs.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from detail.configs.windows import WindowSpec


class ComparisonConfig(BaseModel):
    """One comparison over one behavior artifact, or over one clustering run's nodes."""

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
    source: Literal["labels", "nodes"] = Field(
        "labels",
        description="What is measured: the run's labelled behaviors (its §19 artifact), or every "
        "candidate node of a clustering run's node cache, labelled or not: the difference screen "
        "over clusters nobody has named yet. Nodes need a clustering run with a current node cache.",
        json_schema_extra={"label": "Measure"},
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
    levels: list[str] | None = Field(
        None,
        description="Two values of the one group_by column to contrast, read as the second "
        "against the first. Unset, a column with two values contrasts them, a numeric column "
        "with more is a trend, and a column with more non-numeric values asks for these.",
        json_schema_extra={"label": "Contrast levels"},
    )
    recordings: list[str] | None = Field(
        None,
        description="Restrict to these recordings; unset compares every recording in the artifact.",
        json_schema_extra={"label": "Recordings"},
    )
    measures: list[str] | None = Field(
        None,
        description="Which §23 measures to compute; unset computes every registered measure. "
        "The node source always measures share of time, rate, total time and mean duration.",
        json_schema_extra={"advanced": True, "label": "Measures"},
    )
    windows: WindowSpec = Field(
        default_factory=WindowSpec,
        description="How to slice each recording in time: the whole span, fixed durations, "
        "equal fractions, or a sliding window. window_id then becomes a column to compare "
        "across, which is how early-versus-late becomes the same operation as drug-versus-"
        "vehicle. The node source measures the whole recording.",
        json_schema_extra={"advanced": "Windows", "label": "Windows"},
    )
    draws: int = Field(
        200,
        ge=1,
        description="Permutation draws behind the difference ranking's floor. A paired contrast "
        "over ten or fewer units enumerates every sign flip instead, and ignores this.",
        json_schema_extra={"advanced": True, "label": "Floor draws", "ui_max": 100_000},
    )
    seed: int = Field(
        0,
        ge=0,
        description="Seed for the permutation floor, so a ranking reproduces.",
        json_schema_extra={"advanced": True, "label": "Seed", "ui_max": 2**31 - 1},
    )
    presence_s: float = Field(
        5.0,
        ge=0.0,
        description="Seconds of total time at a level for a unit to count as having the "
        "behavior present there. Within-unit designs count present units toward the minimum, "
        "and a paired contrast reports how many units are present at each level.",
        json_schema_extra={"advanced": True, "label": "Presence (s)", "unit": "s"},
    )
    pseudo_count_s: float = Field(
        1.0,
        ge=0.0,
        description="Seconds of time added to both sides of a within-unit ratio, in the "
        "measure's own unit, so a node absent at one level has a finite ratio rather than "
        "an infinite one that would own the floor.",
        json_schema_extra={"advanced": True, "label": "Pseudo-count (s)", "unit": "s"},
    )
    min_units_fraction: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description="The share of the cohort's units a behavior must be present in for its "
        "within-unit effect to be defined. Below it the effect is undefined, never large: a node "
        "three of eight animals ever showed cannot clear a floor on their behalf.",
        json_schema_extra={"advanced": True, "label": "Minimum units (share)"},
    )
    candidate_floor_s: float = Field(
        30.0,
        ge=0.0,
        description="Seconds of member time a cluster needs, over the whole run, to be a "
        "candidate of the node source. Below it a cluster is too small to be judged from its "
        "clips, and every candidate widens the family the floor is taken over.",
        json_schema_extra={
            "advanced": True,
            "label": "Candidate floor (s)",
            "unit": "s",
        },
    )
    suppress_nesting: bool = Field(
        True,
        description="Over a hierarchical source, keep the best clearing node and drop its "
        "ancestors and descendants, repeating down the ranking, so one behavior is reported "
        "once rather than as every slice of itself. Ignored for a flat source.",
        json_schema_extra={"advanced": True, "label": "Suppress nested nodes"},
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

    @model_validator(mode="after")
    def _check_levels(self) -> "ComparisonConfig":
        """Two distinct levels of exactly one column, or none.

        Levels name values of *a* column; with two grouping columns a level is ambiguous, and one
        level is not a contrast. Whether the values exist is the data's question, answered when
        the comparison runs.
        """
        if self.levels is None:
            return self
        if len(self.group_by) != 1:
            raise ValueError(
                f"levels {self.levels} name values of one column, and {len(self.group_by)} "
                f"columns define the groups. Group by one column to contrast two of its levels."
            )
        if len(self.levels) != 2 or self.levels[0] == self.levels[1]:
            raise ValueError(
                f"levels must be two distinct values of {self.group_by[0]!r}, got {self.levels}"
            )
        return self

    def as_record(self) -> dict[str, Any]:
        """The fields worth persisting, resolved — what the record carries (Tenet 9)."""
        return {
            "source_run_id": self.source_run_id,
            "source_snapshot": self.source_snapshot,
            "source": self.source,
            "group_by": list(self.group_by),
            "aggregate_to": list(self.aggregate_to),
            "levels": list(self.levels) if self.levels else None,
            "recordings": list(self.recordings) if self.recordings else None,
            "measures": list(self.measures) if self.measures else None,
            "windows": self.windows.model_dump(),
            "draws": self.draws,
            "seed": self.seed,
            "presence_s": self.presence_s,
            "pseudo_count_s": self.pseudo_count_s,
            "min_units_fraction": self.min_units_fraction,
            "candidate_floor_s": self.candidate_floor_s,
            "suppress_nesting": self.suppress_nesting,
        }
