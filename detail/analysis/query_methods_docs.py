"""Single source for the Querying methods doc page (§7).

Per querying method: a one-line summary, a short description of how it works, and whether its
match length is tied to the example. The method set itself comes from ``QueryEvalConfig.method``
(the config Literal), so adding a method there forces an entry here (``describe_query_methods``
raises until one exists), and the anti-drift test pins the config set against the implemented
matchers.
"""

from dataclasses import dataclass
from typing import get_args

from detail.configs.describe import describe_config
from detail.configs.query_eval import QueryEvalConfig

# One user-facing line per method id. A method with no entry raises in describe_query_methods.
_SUMMARIES = {
    "dtw": "aligns the example, allowing it to run faster or slower",
    "clip_pooled_mean": "compares one average descriptor per window",
    "clip_pooled_max": "compares one peak descriptor per window",
    "keyframe": "finds windows that contain the example's defining frame(s)",
    "soft_knn": "votes each frame's score from its nearest labeled frames",
}
# A short description of how each method works. Kept plain and em-dash-free (it renders on the page).
_DESCRIPTIONS = {
    "dtw": (
        "Builds a cost table of every example frame against every recording frame and finds the "
        "lowest-cost alignment of the whole example ending at each position, with a free start and "
        "end. The band bounds how far the alignment may drift, so a match runs about the example's "
        "length. It is the one method that tolerates the behavior running faster or slower."
    ),
    "clip_pooled_mean": (
        "Collapses the example to one descriptor, the average of its frame vectors, and compares "
        "it to the average of each candidate window. Frame order inside the window does not "
        "matter, which makes it fast and coarse."
    ),
    "clip_pooled_max": (
        "Like the average pooling, but each descriptor is the per-feature peak across frames. It "
        "asks whether a window peaks the way the example does, rather than whether it averages the "
        "same."
    ),
    "keyframe": (
        "Reduces the example to its most defining frame(s), chosen by `keyframe_select` (the "
        "typical, the most distinctive, or the rarest pose). A window scores well when it contains "
        "a frame close to each keyframe, so it fits a behavior that a signature pose defines."
    ),
    "soft_knn": (
        "Scores every recording frame by a softmax-weighted vote over its `k` nearest labeled "
        "frames, where the labeled set is the example frames plus sampled background frames. The "
        "temperature `tau` sharpens the vote toward the closest neighbors. It ranks all frames and "
        "integrates a threshold-free per-frame Average Precision (a frame counts as positive when "
        "it falls inside a labeled interval), calibrating similarity against the background so rare "
        "positive frames still stand out. This treats retrieval as classification."
    ),
}
# Methods whose returned span is not tied to the example's length. The per-frame methods
# produce no spans at all, and the windowed matchers stay near the example length.
_VARIABLE_LENGTH: set[str] = set()

# The controls surfaced on the page: (QueryEvalConfig field, which methods it affects, what it
# sets). The default value and unit come from the config field, not from here.
_CONTROLS = [
    (
        "fusion",
        "every method",
        "resemble any one example (`min`) or all of them (`mean`)",
    ),
    (
        "band_s",
        "windowed matchers (`dtw`, `clip_pooled`, `keyframe`)",
        "warp and duration tolerance; `0` is rigid",
    ),
    (
        "keyframe_select",
        "`keyframe`",
        "pick the typical, most distinctive, or rarest pose",
    ),
    ("n_keyframes", "`keyframe`", "how many defining frames to require"),
    ("k", "`soft_knn`", "nearest labeled frames voting on each frame"),
    ("tau", "`soft_knn`", "softmax temperature; lower sharpens the vote"),
    ("n_negatives", "`soft_knn`", "background frames sampled as the negative class"),
]


@dataclass(frozen=True)
class QueryMethodDoc:
    """One querying method described for the docs page."""

    method_id: str
    summary: str
    description: str
    variable_length: bool  # True: match length grows with the behavior, not the example


@dataclass(frozen=True)
class QueryControlDoc:
    """One querying control described for the docs page (default read from the config)."""

    name: str
    default: str  # formatted with unit, e.g. "`0.5` s"
    applies_to: str
    sets: str


def query_method_ids() -> list[str]:
    """The querying method ids, in config order (the single source is the config Literal)."""
    return list(get_args(QueryEvalConfig.model_fields["method"].annotation))


def describe_query_methods() -> list[QueryMethodDoc]:
    """Per-method doc metadata, derived from the config method set."""
    docs: list[QueryMethodDoc] = []
    for method_id in query_method_ids():
        if method_id not in _SUMMARIES or method_id not in _DESCRIPTIONS:
            raise ValueError(
                f"querying method {method_id!r} is missing a summary or description in "
                "analysis.query_methods_docs"
            )
        docs.append(
            QueryMethodDoc(
                method_id=method_id,
                summary=_SUMMARIES[method_id],
                description=_DESCRIPTIONS[method_id],
                variable_length=method_id in _VARIABLE_LENGTH,
            )
        )
    return docs


def _format_default(field: dict) -> str:
    value = field["default"]
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # 30.0 -> 30
    unit = f" {field['unit']}" if field["unit"] else ""
    return f"`{value}`{unit}"


def describe_query_controls() -> list[QueryControlDoc]:
    """Per-control doc metadata; the default and unit are read from QueryEvalConfig."""
    fields = {field["name"]: field for field in describe_config(QueryEvalConfig)}
    docs: list[QueryControlDoc] = []
    for name, applies_to, sets in _CONTROLS:
        if name not in fields:
            raise ValueError(
                f"query control {name!r} is not a QueryEvalConfig field "
                "(analysis.query_methods_docs._CONTROLS is out of sync with the config)"
            )
        docs.append(
            QueryControlDoc(
                name=name,
                default=_format_default(fields[name]),
                applies_to=applies_to,
                sets=sets,
            )
        )
    return docs
