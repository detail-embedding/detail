"""Cutting a pose trajectory into bouts, and comparing a bout to a medoid.

Holds the numeric steps every bout-based clustering shares: turning a config's detector knobs
into the detector's params (``detector_params``, ``detector_params_rows``), segmenting one
recording's trajectory into bouts (``segment_recording``), stacking several recordings into one
pooled row space (``pool_recordings``), and the banded DTW distance from a bout to a medoid
(``bout_medoid_distance``).

Temporal clustering (``cluster_methods/dtw_finch.py``) runs these steps. They were shared with the
removed clusterer and syllable-discovery modules, each of which used to hold a part of them, and
they live here so a caller depends on this module rather than on another caller.

Not in ``analysis/segmentation.py``: that module imports nothing from ``detail`` at module level,
and ``segment_recording`` needs the cadence helpers in ``analysis/clip_query.py``.
"""

import numpy as np

from detail.analysis import clip_query as clip_core
from detail.analysis import segmentation

SEC_TO_NS = 1_000_000_000
# Omitting max_gap splits a subject's trajectory at any single dropped frame (a normal step is 1× the
# embed cadence, a one-frame drop is 2×) — the same aggressive default the removed syllable
# discovery used.
GAP_FACTOR = 1.5

# Detector knobs that are **durations**, carried as nanoseconds from here and converted to
# feature-matrix indices at segmentation time, where the subject's cadence is known. The detector
# itself still takes rows: it works on the feature matrix and has no time axis. Same path
# ``window_s`` / ``min_len_s`` / ``band_s`` already take (seconds -> ns in the config resolution,
# ns -> rows in segment_recording) — a raw frame count here would mean a different window on every
# recording, which is exactly the bug tests/configs/test_unit_conventions.py guards.
_DURATION_PARAMS = ("half_ns", "min_size_ns", "smooth_half_ns", "nms_window_ns")
# ns key -> the row-valued keyword the detector actually accepts.
_DURATION_TO_ROWS = {
    "half_ns": "half",
    "min_size_ns": "min_size",
    "smooth_half_ns": "smooth_half",
    "nms_window_ns": "nms_window",
}


def detector_params(model: object) -> dict:
    """The per-detector params the pure detector consumes, assembled from the model's flat knobs
    (the unused ones for a given detector are simply not passed).

    Durations leave here in **nanoseconds** (``*_ns``); ``detector_params_rows`` turns them into
    feature indices once the trajectory's cadence is known.

    ``model`` is any config carrying these attributes, read by name: ``detector``, ``penalty``,
    ``percentile``, ``novelty_half_s``, ``abd_smooth_half_s``, ``abd_nms_window_s`` and
    ``min_size_s``.
    """
    ns = lambda seconds: int(round(float(seconds) * SEC_TO_NS))  # noqa: E731
    if model.detector == "binseg":
        return {"penalty": model.penalty, "min_size_ns": ns(model.min_size_s)}
    if model.detector == "novelty":
        return {
            "half_ns": ns(model.novelty_half_s),
            "percentile": model.percentile,
            "min_size_ns": ns(model.min_size_s),
        }
    if model.detector == "abd":
        return {
            "smooth_half_ns": ns(model.abd_smooth_half_s),
            "nms_window_ns": ns(model.abd_nms_window_s),
            "min_size_ns": ns(model.min_size_s),
        }
    return {  # speed
        "percentile": model.percentile,
        "min_size_ns": ns(model.min_size_s),
    }


def detector_params_rows(params: dict, t_ns, feat_stride: int = 1) -> dict:
    """Resolve a detector's duration params to feature-matrix indices for one subject.

    ``t_ns`` is the subject's timestamps, so the same seconds becomes a different index count at a
    different cadence — the whole point. Indices are in *feature* space, which steps by
    ``feat_stride``, and every duration resolves to at least 1 so a short window never rounds away
    to nothing. A params dict carrying no ``*_ns`` key (a record written before the conversion)
    passes through unchanged.
    """
    out = {k: v for k, v in params.items() if k not in _DURATION_PARAMS}
    for ns_key, row_key in _DURATION_TO_ROWS.items():
        if ns_key in params:
            rows = clip_core.band_rows(t_ns, int(params[ns_key]))
            out[row_key] = max(1, int(round(rows / max(1, int(feat_stride)))))
        elif row_key in params:
            out[row_key] = params[row_key]  # legacy record: already rows
    return out


def segment_recording(emb, frame_idx, t_ns, params: dict):
    """Segment one recording's pose trajectory into bouts, reusing the discovery pipeline.

    Returns ``(bouts, band, resolved)`` — ``bouts`` are inclusive ``(r0, r1)`` trajectory-row spans,
    ``band`` the DTW band in rows at this subject's cadence, ``resolved`` the seconds→rows conversion.
    """
    period = clip_core.nominal_period_ns(t_ns)
    max_gap = (
        params["max_gap_ns"]
        if params.get("max_gap_ns") is not None
        else int(round(GAP_FACTOR * period))
    )
    runs = clip_core.segment_runs(t_ns, max_gap)
    window_rows = max(1, clip_core.band_rows(t_ns, params["window_ns"]))
    min_len_rows = max(1, clip_core.band_rows(t_ns, params["min_len_ns"]))
    band = clip_core.band_rows(t_ns, params["band_ns"])
    X, row_of_X, xruns = segmentation.build_features(
        emb,
        runs,
        window=window_rows,
        pool=params["pool"],
        feat_stride=params["feat_stride"],
    )
    # Durations arrive as nanoseconds and become feature indices here, where this subject's
    # cadence is known — the same seconds is a different index count on a differently sampled
    # recording, which is the point.
    detector_params = detector_params_rows(
        params["detector_params"], t_ns, params.get("feat_stride", 1)
    )
    cuts = segmentation.detect_transitions(
        X, xruns, params["detector"], detector_params
    )
    bouts = segmentation.segment_bouts(cuts, row_of_X, xruns, min_len_rows)
    resolved = {
        "cadence_fps": (SEC_TO_NS / period if period > 0 else 0.0),
        "window_rows": window_rows,
        "min_len_rows": min_len_rows,
        "band_rows": band,
        "max_gap_ns": max_gap,
        "n_runs": len(runs),
    }
    # The score the detector cut on, kept for diagnostics (why a cut landed where it did).
    # Feature-index space; ``row_of_X`` maps it back to trajectory rows, and thence to t_ns.
    # None for a detector that scores no curve (binseg). Nothing in the fit reads it.
    curve = segmentation.transition_curve(X, xruns, params["detector"], detector_params)
    trace = (
        None
        if curve is None
        else {
            "values": np.asarray(curve.values, dtype=np.float32),
            "row_of_X": np.asarray(row_of_X, dtype=np.int64),
            "cuts": np.asarray(cuts, dtype=np.int64),
            "threshold": float(curve.threshold),
            "offset": int(curve.offset),
        }
    )
    return bouts, band, resolved, trace


def pool_recordings(
    recording_data: list[dict],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple]]:
    """Concatenate per-recording trajectories into one pooled space.

    Returns the stacked ``(rows, D)`` embeddings, each bout's span **in pooled row coordinates**,
    and a parallel ``origin`` row per bout carrying its source recording + frame/``t_ns`` bounds.
    Deterministic given the same recordings and segmentation params, which is what lets a
    re-cluster rebuild the identical pooled layout and check it against a cached matrix.
    """
    pooled_blocks, pooled_bouts, origin = [], [], []
    offset = 0
    for sd in recording_data:
        emb = np.asarray(sd["emb"])
        pooled_blocks.append(emb)
        for r0, r1 in sd["bouts"]:
            pooled_bouts.append((offset + r0, offset + r1))
            origin.append(
                (
                    sd["recording_id"],
                    int(sd["frame_idx"][r0]),
                    int(sd["frame_idx"][r1]),
                    int(sd["t_ns"][r0]),
                    int(sd["t_ns"][r1]),
                )
            )
        offset += emb.shape[0]
    pooled_emb = (
        np.vstack(pooled_blocks)
        if pooled_blocks
        else np.zeros((0, 0), dtype=np.float32)
    )
    return pooled_emb, pooled_bouts, origin


def bout_medoid_distance(bout_emb, medoid_emb, band: int) -> float:
    """Banded fixed-endpoint DTW distance between a bout and a medoid (the fit's exact metric)."""
    from detail.analysis import _dtw_numba
    from detail.analysis.segmentation import _normalize

    xi = _normalize(np.asarray(bout_emb, dtype=np.float64))
    xj = _normalize(np.asarray(medoid_emb, dtype=np.float64))
    cost = 1.0 - xi @ xj.T
    effective_band = max(
        band, abs(len(xi) - len(xj))
    )  # keep unequal-length bouts reachable
    return float(_dtw_numba.dtw_dp(cost, effective_band))
