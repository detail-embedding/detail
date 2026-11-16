"""Behavior-query eval metrics CSV (§16) — one row per (embedding run, config, behavior).

A self-describing tidy table a shell sweep accumulates: each ``analysis query-eval`` call
appends its rows so configs/embedding runs sit side by side for comparison. Written
atomically (tmp + ``os.replace``); appending is read-existing + append + atomic rewrite
(the house grain — there is no in-place append helper, and this stays safe against a
concurrent reader). Mirrors ``storage/clip_query_csv.py``.
"""

import csv
import io
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from detail.core import atomic

COLUMNS = [
    "embed_run_id",
    "behavior",
    "behavior_type",
    "n_recordings",
    "fusion",
    "band_s",
    "iou_threshold",
    "point_window_s",
    "n_exemplars",
    "seed",
    "trajectory_split_gap_s",
    "max_results",
    "n_gt",
    "n_pred",
    "tp",
    "fp",
    "fn",
    "ap",
    "precision",
    "recall",
    "best_f1",
    "best_f1_threshold",
    # Cross-validation aggregate (§16): appended last + optional on read so an older
    # sweep CSV (pre-folds) still reads. For folds=1: n_folds=1, every *_std=0, and the
    # point estimates equal the single split — identical to the pre-folds headline.
    "n_folds",
    "ap_std",
    "precision_std",
    "recall_std",
    "best_f1_std",
    # The matcher, and each headline next to its null (§16) — a sweep row that doesn't
    # name its matcher can't be compared, and one without its null can't be believed.
    "method",
    "recall_null",
    "recall_lift",
    "ap_null",
    "ap_lift",
    # How many instances left the scored set. Equal to ``n_exemplars`` unless a sweep pinned it so
    # every point on an n_exemplars curve is scored against the same instances — the pair is what
    # tells a reader whether the curve measures its own denominator.
    "n_holdout",
    # Where soft_knn drew its negatives. Without it two runs are indistinguishable afterwards,
    # and the difference is worth up to 0.4 AP.
    "negative_pool",
]

# Optional columns — absent in an older CSV; the reader fills the dataclass default.
_METRIC_OPTIONAL = {
    "n_folds",
    "ap_std",
    "precision_std",
    "recall_std",
    "best_f1_std",
    "method",
    "recall_null",
    "recall_lift",
    "ap_null",
    "ap_lift",
    "trajectory_split_gap_s",
    "n_holdout",
    "negative_pool",
}

# Columns renamed since a CSV may have been written: old header → current field.
_METRIC_LEGACY = {"max_gap_s": "trajectory_split_gap_s"}


@dataclass
class MetricRow:
    embed_run_id: str
    behavior: str
    behavior_type: str
    n_recordings: int
    fusion: str
    band_s: float
    iou_threshold: float
    point_window_s: float
    n_exemplars: int
    seed: int
    trajectory_split_gap_s: float | None
    max_results: int | None
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ap: float
    precision: float
    recall: float
    best_f1: float
    best_f1_threshold: float
    n_folds: int = 1
    ap_std: float = 0.0
    precision_std: float = 0.0
    recall_std: float = 0.0
    best_f1_std: float = 0.0
    method: str = "dtw"
    # Every method writes its null and lift: the span path shifts its spans in time, and the
    # per-frame path circularly shifts each subject's scores (``analysis.query_eval``'s
    # ``perframe_null``).
    # ``None`` is a row with no overlap entry at its configured level, or one read from a CSV
    # written before these columns existed; it is an empty cell, not a misleading 0.0.
    recall_null: float | None = None
    recall_lift: float | None = None
    ap_null: float | None = None
    ap_lift: float | None = None
    # Instances removed from the scored set. Equal to ``n_exemplars`` unless a sweep pinned it so
    # that every point on an n_exemplars curve is scored against the same instances; the pair is
    # what tells a reader whether the curve measures its own denominator.
    n_holdout: int | None = None
    # A row written before the pool was selectable came from the clean one.
    negative_pool: str = "labeled_clean"


FOLD_COLUMNS = [
    "behavior",
    "fold",
    "n_gt",
    "n_pred",
    "tp",
    "fp",
    "fn",
    "ap",
    "precision",
    "recall",
    "best_f1",
    "best_f1_threshold",
]


@dataclass
class FoldMetricRow:
    """One cross-validation fold's metrics for a behavior (§16) — the per-fold detail
    behind the aggregate ``metrics.csv`` row."""

    behavior: str
    fold: int
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    ap: float
    precision: float
    recall: float
    best_f1: float
    best_f1_threshold: float


PREDICTION_COLUMNS = [
    "behavior",
    "recording_id",
    "track_id",
    "rank",
    "outcome",
    "score",
    "start_t_ns",
    "end_t_ns",
    "iou",
    "offset_ns",
    "matched_gt_start_t_ns",
    "matched_gt_end_t_ns",
    "fold",
]


@dataclass
class PredictionRow:
    """One retrieved candidate and how it scored against ground truth (§16).

    ``outcome`` is ``"tp"`` or ``"fp"``. ``iou`` is set for a **state** hit,
    ``offset_ns`` for a **point** hit (the other is empty); ``matched_gt_*`` name the
    consumed GT span on a TP. Rank is 1-based within the fold; ``fold`` is the
    cross-validation fold (0 for a single-split run). All behaviors + folds share one
    ``predictions.csv``, filtered by ``behavior`` / ``fold``."""

    behavior: str
    recording_id: str
    track_id: int
    rank: int
    outcome: str
    score: float
    start_t_ns: int
    end_t_ns: int
    iou: float | None
    offset_ns: int | None
    matched_gt_start_t_ns: int | None
    matched_gt_end_t_ns: int | None
    fold: int = 0


GROUND_TRUTH_COLUMNS = [
    "behavior",
    "recording_id",
    "track_id",
    "start_t_ns",
    "end_t_ns",
    "role",
    "outcome",
    "fold",
]


@dataclass
class GroundTruthRow:
    """One ground-truth interval and its fate in a fold (§16).

    ``role`` is ``"exemplar"`` (held out to seed the query, not scored) or
    ``"scored"``; ``outcome`` is ``"matched"`` / ``"fn"`` for a scored interval and
    empty for an exemplar. ``fold`` is the cross-validation fold (0 for a single
    split) — the exemplar / scored split differs per fold."""

    behavior: str
    recording_id: str
    track_id: int
    start_t_ns: int
    end_t_ns: int
    role: str
    outcome: str
    fold: int = 0


def _cell(value) -> str:
    return "" if value is None else str(value)


def _conv(value: str, annotation):
    """Coerce a CSV cell by the dataclass field annotation; ``""`` → ``None`` only for
    an ``X | None`` field (a plain ``str`` field keeps the empty string)."""
    ann = str(annotation)
    if value == "":
        return None if "None" in ann else value
    if "int" in ann:
        return int(value)
    if "float" in ann:
        return float(value)
    return value


def _write_rows(path: Path | str, columns: list[str], rows: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            data = asdict(row)
            writer.writerow([_cell(data[c]) for c in columns])
    atomic.replace(tmp, path)


def _read_rows(
    path: Path | str,
    row_cls,
    columns: list[str],
    required: set[str] | None = None,
    legacy: dict[str, str] | None = None,
) -> list:
    """Read rows, tolerating **absent optional columns** (a col not in ``required``):
    an older CSV is a prefix of the current schema, so a missing optional column falls
    back to the dataclass default rather than failing. ``legacy`` maps a **renamed**
    column's old header to its current field, so a CSV written before the rename keeps
    its value instead of silently reverting to the default."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no csv at {path}")
    typed = {f.name: f.type for f in fields(row_cls)}
    need = set(columns) if required is None else required
    aliases = dict(legacy or {})
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        present = set(reader.fieldnames or [])
        # A renamed column present under its old header satisfies the requirement.
        renamed = {new: old for old, new in aliases.items() if old in present}
        missing = need - present - set(renamed)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        use = [
            c for c in columns if c in present or c in renamed
        ]  # absent optional → dataclass default
        return [
            row_cls(**{c: _conv(r[renamed.get(c, c)], typed[c]) for c in use})
            for r in reader
        ]


OVERLAP_COLUMNS = [
    "behavior",
    "behavior_type",
    "iou",
    "n_gt",
    "n_pred",
    "tp",
    "recall",
    "recall_null",
    "recall_lift",
    "precision",
    "ap",
    "ap_null",
    "ap_lift",
]


@dataclass
class OverlapRow:
    """One behavior's scores at one overlap level, each headline beside its null (§16).

    ``metrics.csv`` carries the headline at the single configured ``iou_threshold``; this
    is the shape of the result *across* strictness levels, which is where a matcher that
    finds the right moments but the wrong extent shows up. ``recall_null`` is the same
    predictions slid to the wrong places, ``ap_null`` the same outcomes in random order;
    a value that doesn't clear its null isn't real. For a **point** behavior there is one
    row, at its ``point_window_s`` tolerance (``iou`` is not meaningful)."""

    behavior: str
    behavior_type: str
    iou: float
    n_gt: int
    n_pred: int
    tp: int
    recall: float
    recall_null: float
    recall_lift: float
    precision: float
    ap: float
    ap_null: float
    ap_lift: float


def write_overlap_csv(path: Path | str, rows: list[OverlapRow]) -> None:
    """Write the per-overlap-level breakdown atomically (header = ``OVERLAP_COLUMNS``)."""
    _write_rows(path, OVERLAP_COLUMNS, rows)


def read_overlap_csv(path: Path | str) -> list[OverlapRow]:
    """Read ``overlap.csv`` into ``OverlapRow``s."""
    return _read_rows(path, OverlapRow, OVERLAP_COLUMNS)


def write_predictions_csv(path: Path | str, rows: list[PredictionRow]) -> None:
    """Write per-clip prediction outcomes atomically (header = ``PREDICTION_COLUMNS``)."""
    _write_rows(path, PREDICTION_COLUMNS, rows)


def read_predictions_csv(path: Path | str) -> list[PredictionRow]:
    """Read ``predictions.csv`` into ``PredictionRow``s (raises if a column is missing)."""
    return _read_rows(
        path, PredictionRow, PREDICTION_COLUMNS, set(PREDICTION_COLUMNS) - {"fold"}
    )


def write_ground_truth_csv(path: Path | str, rows: list[GroundTruthRow]) -> None:
    """Write ground-truth outcomes atomically (header = ``GROUND_TRUTH_COLUMNS``)."""
    _write_rows(path, GROUND_TRUTH_COLUMNS, rows)


def read_ground_truth_csv(path: Path | str) -> list[GroundTruthRow]:
    """Read ``ground_truth.csv`` into ``GroundTruthRow``s (raises if a column is missing)."""
    return _read_rows(
        path, GroundTruthRow, GROUND_TRUTH_COLUMNS, set(GROUND_TRUTH_COLUMNS) - {"fold"}
    )


def write_folds_csv(path: Path | str, rows: list[FoldMetricRow]) -> None:
    """Write per-fold metrics atomically (header = ``FOLD_COLUMNS``)."""
    _write_rows(path, FOLD_COLUMNS, rows)


def read_folds_csv(path: Path | str) -> list[FoldMetricRow]:
    """Read ``folds.csv`` into ``FoldMetricRow``s."""
    return _read_rows(path, FoldMetricRow, FOLD_COLUMNS)


def read_pr_points(path: Path | str) -> list[tuple[float, float]]:
    """Read ``(recall, precision)`` PR-curve points written by ``write_pr_points``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no pr points at {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [(float(r["recall"]), float(r["precision"])) for r in reader]


def write_metrics_csv(path: Path | str, rows: list[MetricRow]) -> None:
    """Write the metrics table atomically (header row = ``COLUMNS``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            data = asdict(row)
            writer.writerow([_cell(data[c]) for c in COLUMNS])
    atomic.replace(tmp, path)


def read_metrics_csv(path: Path | str) -> list[MetricRow]:
    """Read the metrics table into ``MetricRow``s (the cross-validation, matcher and null
    columns are optional, and the renamed gap column still reads under its old header, so
    an older sweep CSV still loads)."""
    return _read_rows(
        path,
        MetricRow,
        COLUMNS,
        set(COLUMNS) - _METRIC_OPTIONAL,
        legacy=_METRIC_LEGACY,
    )


def append_metrics(path: Path | str, rows: list[MetricRow]) -> None:
    """Append rows to the metrics CSV (creating it with a header if absent)."""
    path = Path(path)
    existing = read_metrics_csv(path) if path.is_file() else []
    write_metrics_csv(path, existing + list(rows))


def write_pr_points(path: Path | str, points: list[tuple[float, float]]) -> None:
    """Write ``(recall, precision)`` PR-curve points for one eval to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recall", "precision"])
        writer.writerows(points)
    atomic.replace(tmp, path)


def write_similarity(path: Path | str, recording_id: str, t_ns, score) -> None:
    """Persist a behavior's fold-0 per-frame similarity curve atomically (npz: t_ns, score).

    The similarity figure reads this; it is the display recording's per-frame similarity to the
    fixed split's exemplars (higher = more example-like), aligned to the recording by ``t_ns``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        t_ns=np.asarray(t_ns),
        score=np.asarray(score, dtype=np.float32),
        recording_id=np.str_(recording_id),
    )
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_bytes(buffer.getvalue())
    atomic.replace(tmp, path)
