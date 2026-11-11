"""Behavior-query eval config (§16) — the single source for each parameter's type,
bounds, default, help, and label (CLI help + docs read them).

Scores how well the behavior-query engine retrieves a known behavior against the
`behavior`-role ground-truth streams (§3): held-out exemplars seed the query, matches
are scored against the remaining ground-truth intervals. References (recordings, the embed
run, the behaviors) are checked against the experiment in ``validate``; the scalars carry
bounds. Time knobs are **seconds** (converted to ns in the runner)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ``max_gap_s`` was the original name of the trajectory-continuity split. It was renamed to
# ``trajectory_split_gap_s`` to say which gap it means. Saved runs + sweeps carry the old key.
_LEGACY_KEYS = {"max_gap_s": "trajectory_split_gap_s"}

# Per-frame scoring methods (§16): score every frame, rank, integrate a per-frame Average
# Precision — a frame is positive iff it falls in a scored ground-truth interval (state) or
# within ±point_window_s of a scored point. No span/IoU matching, no proposal cutoff. These
# are eval-only (not windowed retrieval matchers), so ``evaluate_query`` branches on this set.
_PERFRAME_METHODS = frozenset({"soft_knn"})


class QueryEvalConfig(BaseModel):
    source_recording_ids: list[str] = Field(
        default_factory=list,
        description="Recordings to search (default: every recording scored for the behavior).",
        json_schema_extra={"scope": "selection", "label": "Recordings"},
    )
    embed_run_id: str = Field(
        ...,
        description="Embedding run to query (emb_NN): the embedding configuration being evaluated.",
        json_schema_extra={"scope": "selection", "label": "Embedding run"},
    )
    behaviors: list[str] = Field(
        default_factory=list,
        description="Behavior label(s) to evaluate; one metrics row per behavior.",
        json_schema_extra={"label": "Behaviors"},
    )
    stream_name: str | None = Field(
        None,
        description="Ground-truth behavior stream to score against, by name. Omit when every "
        "recording carries exactly one. A recording carrying several stops the run and lists "
        "their names, rather than picking whichever is first. The other streams' labels would "
        "read as unscored and drop that recording.",
        json_schema_extra={"label": "Ground-truth stream"},
    )
    track_id: int = Field(
        0,
        ge=0,
        description="Subject track id to search.",
        json_schema_extra={"label": "Track"},
    )
    fusion: Literal["min", "mean"] = Field(
        "min",
        description="Exemplar fusion: min (resemble any one) or mean (resemble all).",
        json_schema_extra={"label": "Fusion"},
    )
    method: Literal[
        "dtw",
        "clip_pooled_mean",
        "clip_pooled_max",
        "keyframe",
        "soft_knn",
    ] = Field(
        "dtw",
        description="Scoring method. Windowed matchers (span-detection AP): dtw (banded "
        "subsequence DTW), clip_pooled_mean/max (one pooled descriptor per clip), keyframe "
        "(the most query-defining frame). Per-frame classifier (per-frame AP): soft_knn "
        "(softmax-weighted vote over the k nearest labeled frames + sampled negatives).",
        json_schema_extra={"label": "Method"},
    )
    keyframe_select: Literal["medoid", "distinctive", "rare"] = Field(
        "medoid",
        description="For method=keyframe: pick defining frame(s) by medoid (typical), "
        "distinctive (farthest from background), or rare (least db-similar).",
        json_schema_extra={"advanced": "Matcher", "label": "Keyframe select"},
    )
    n_keyframes: int = Field(
        1,
        ge=1,
        description="For method=keyframe: number of query-defining frames to require.",
        json_schema_extra={"advanced": "Matcher", "label": "Keyframes"},
    )
    band_s: float = Field(
        0.5,
        ge=0.0,
        description="Warp/duration tolerance in seconds (0 = rigid fixed length). Windowed "
        "matchers only (dtw / clip_pooled / keyframe).",
        json_schema_extra={"label": "Band", "unit": "s"},
    )
    k: int = Field(
        20,
        ge=1,
        description="For method=soft_knn: number of nearest labeled frames voting on each "
        "frame's score.",
        json_schema_extra={"advanced": "Matcher", "label": "k (neighbors)"},
    )
    tau: float = Field(
        0.07,
        gt=0.0,
        description="For method=soft_knn: softmax temperature on cosine similarity (lower = "
        "sharper weighting of the closest neighbors).",
        json_schema_extra={"advanced": "Matcher", "label": "Temperature"},
    )
    n_negatives: int = Field(
        500,
        ge=1,
        description="For method=soft_knn: background frames sampled as the negative class in "
        "the neighbor index (class-balances the rare positive frames).",
        json_schema_extra={"advanced": "Matcher", "label": "Negatives"},
    )
    negative_pool: Literal["unlabeled", "labeled_clean"] = Field(
        "unlabeled",
        description="For method=soft_knn: where the negative class is sampled from. `unlabeled` "
        "draws from everything not held out, target behavior included, which is what a user has "
        "at query time. `labeled_clean` draws only from frames known not to be the behavior, "
        "which needs the ground truth and so measures a score no user can reproduce; it is kept "
        "to re-read runs recorded before this existed.",
        json_schema_extra={"advanced": "Matcher", "label": "Negative pool"},
    )
    trajectory_split_gap_s: float | None = Field(
        None,
        ge=0.0,
        description="A break in tracking longer than this (seconds) splits the trajectory "
        "into separate stretches to search, and an exemplar must sit inside one. "
        "0 = never split. Default: ~4x the subject's frame period.",
        json_schema_extra={
            "advanced": "Scoring",
            "label": "Trajectory split gap",
            "unit": "s",
            "ui_max": 3600.0,
            "ui_step": 0.5,
        },
    )
    max_results: int | None = Field(
        None,
        ge=1,
        description="Cap on retrieved candidates (default: unbounded, needed for honest recall).",
        json_schema_extra={
            "advanced": "Scoring",
            "label": "Max results",
            "ui_max": 1_000_000,
        },
    )
    n_exemplars: int = Field(
        1,
        ge=1,
        description="Ground-truth instances held out as query exemplars (excluded from scoring).",
        json_schema_extra={"label": "Exemplars"},
    )
    n_holdout: int | None = Field(
        None,
        ge=1,
        description="Instances removed from the scored set (default: n_exemplars). Set it to the "
        "largest n_exemplars in a sweep so every point is scored against the same instances; "
        "otherwise the test set shrinks as the query grows and the curve measures its own "
        "denominator rather than the effect of the exemplars.",
        json_schema_extra={"advanced": "Scoring", "label": "Holdout"},
    )
    folds: int = Field(
        1,
        ge=0,
        description=(
            "Cross-validation folds: 1 = a single seeded split; k = k folds, each a "
            "distinct n_exemplars-subset, with metrics aggregated mean±std; 0 = "
            "leave-one-out (every instance once; requires n_exemplars=1)."
        ),
        json_schema_extra={"label": "Folds"},
    )
    seed: int = Field(
        0,
        ge=0,
        description="Seed for held-out exemplar selection (fix it across compared runs).",
        json_schema_extra={"advanced": "Scoring", "label": "Seed"},
    )
    iou_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="State behaviors: min temporal IoU for a predicted span to count as a hit.",
        json_schema_extra={"advanced": "Scoring", "label": "IoU threshold"},
    )
    point_window_s: float = Field(
        0.5,
        ge=0.0,
        description="Point behaviors: +/- tolerance in seconds for a predicted match to hit a point.",
        json_schema_extra={"advanced": "Scoring", "label": "Point window", "unit": "s"},
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for old, new in _LEGACY_KEYS.items():
            if old in out and out.get(new) is None:
                out[new] = out[old]
            out.pop(old, None)
        return out

    @model_validator(mode="after")
    def _leave_one_out_needs_single_exemplar(self) -> "QueryEvalConfig":
        # LOO (folds=0) uses each instance as one query, so it is only defined for
        # n_exemplars=1. Enforced here (the single source), so every surface — the
        # synchronous CLI, the recorded module's validate(), and the GUI — rejects it.
        if self.folds == 0 and self.n_exemplars != 1:
            raise ValueError(
                "leave-one-out (folds=0) uses each instance as one query, so it requires "
                f"n_exemplars=1 (got {self.n_exemplars})"
            )
        return self

    @model_validator(mode="after")
    def _holdout_covers_the_exemplars(self) -> "QueryEvalConfig":
        # The holdout is the set removed from scoring; the exemplars are the prefix of it used as
        # the query. A holdout smaller than the query would put an exemplar back in the scored
        # set, where the model has already seen it.
        if self.n_holdout is not None and self.n_holdout < self.n_exemplars:
            raise ValueError(
                f"n_holdout={self.n_holdout} is smaller than n_exemplars={self.n_exemplars}; the "
                "holdout is what leaves the scored set and the exemplars are drawn from it, so it "
                "has to be at least as large"
            )
        if self.n_holdout is not None and self.folds == 0:
            raise ValueError(
                "leave-one-out (folds=0) already means every instance takes a turn as the query, "
                "which a separate holdout contradicts — use folds=1 with a seed per replicate, or "
                "drop n_holdout"
            )
        return self
