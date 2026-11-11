"""Soft k-NN query method (§20) — per-frame scoring, instances formed from the peaks.

It pools every exemplar frame into one labelled index, so five clips cost about what
one does, where a per-exemplar matcher pays one pass per clip.

**The scoring is a port; the segmentation is not.** `pipeline/query_eval.py` scores frames and
folds them into a per-frame Average Precision, which needs no spans at all — its `_bool_runs`
exists to make figures readable and says so. §20 needs instances, so how a per-frame score
becomes one had to be decided, and it was decided by comparing the alternatives:

- **The raw score flickers.** A run above any cutoff lasts a frame or two, so
  thresholding the raw score is not a viable way to form anything.
- **Smoothing the score beats patching the mask.** Bridging gaps and dropping short
  runs wanted a different setting per behavior; a moving average wanted the same window
  everywhere. Smoothing also happens once here, where a bridge rule would have to re-run
  for every cutoff the user drags through.
- **The hard part is extent, not retrieval.** The method finds an occurrence far more
  often than it gets its edges right. Of three extent rules at matched candidate counts,
  growing each match to **half its own peak** sized matches best, ahead of
  threshold-plus-min-run and a fixed exemplar-length window. A fixed window is the best
  *finder* and the worst *sizer*, which is why it flatters behaviours of consistent
  duration.

**Instance count is monotonic in the cutoff**, which `instances_at` does not require and the
calibration page is far easier to reason about with. It falls out of forming candidates *before*
any cutoff exists: raising the cutoff only removes them, it never re-forms an extent. It also
makes `instances_at` a filter over a few hundred tuples rather than a pass over the trajectory.

**No candidate cap.** Suppression at half the exemplar length already bounds the count to about
`rows / separation` — roughly 1,200 on a thirty-minute recording — so every local maximum is
kept and the curve is complete down to zero. §20's warning that "a curve that flattens because
storage ran out looks exactly like a curve that flattens because the animal stopped" is answered
by there being nothing to truncate.

**A FINCH-style merge was tried here and rejected.** Grouping the above-floor runs by nearest
neighbour and taking connected components over-merges by construction — components chain, giving
spans many times longer than the true occurrences — and TW-FINCH's temporal
term cannot prevent it, because for a fixed row it is a constant factor across every candidate
and the 1-NN is identical under any scaling of it. FINCH partitions; retrieval selects.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from detail.analysis import retrieval
from detail.configs.query_soft_knn import SoftKnnQueryConfig
from detail.core.dataclasses import (
    ExemplarBounds,
    NegativeSupport,
    QueryCandidate,
    SamplingRequest,
)
from detail.core.protocols import QueryMethod
from detail.core.registries import register_query_method

_log = logging.getLogger(__name__)

#: Rows scored per pass. Named here so a cadence test can force more than one block over a
#: small fixture — the block loop is where this method's progress comes from, so a fixture
#: that fits in one block would make the guard pass while reporting nothing.
SCORE_BLOCK = retrieval._SOFT_KNN_BLOCK

#: What a shard holds. Named once so `serialize` and `load` cannot drift apart.
_SHARD_ARRAYS = ("score", "start", "end", "peak", "row_offset")


@dataclass
class SoftKnnState:
    """What ``prepare`` derives once per behavior: the labelled index and the row scales."""

    labeled: np.ndarray  # (L, D), L2-normalized; the positives come first
    n_positive: int
    smooth_rows: int
    separation_rows: int
    max_extent_rows: int

    @property
    def context_rows(self) -> int:
        """Rows of neighbouring trajectory a candidate's peak needs to resolve fully.

        Its extent may grow ``max_extent_rows`` either way, and the smoothed value it grows
        against is itself an average over ``smooth_rows``. A chunk that owns a peak must hold
        this much on both sides of it — see ``chunk_overlap_rows``.
        """
        return int(self.max_extent_rows + self.smooth_rows)


@dataclass
class SoftKnnResult:
    """This method's ``SearchResult``: the smoothed curve, and the candidates formed from it.

    Both, rather than one or the other. The candidates are what a cutoff filters, and they are
    tiny. The curve is ~4 bytes a row — 430 KB for a thirty-minute subject — and it is what a
    calibration surface can draw under the clips to show *why* a cutoff returns what it does;
    recomputing it would mean re-running the search.
    """

    score: np.ndarray  # (M,) smoothed, float32
    start: np.ndarray  # (C,) inclusive, trajectory-global
    end: np.ndarray  # (C,) inclusive, trajectory-global
    peak: np.ndarray  # (C,) the candidate's confidence
    row_offset: int

    @property
    def n_rows(self) -> int:
        return len(self.score)


def _smooth(score: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over ``window`` rows, edge-padded.

    Row ``j`` averages rows ``j - window // 2`` to ``j + (window - 1) // 2``: symmetric for an odd
    window, and one row heavier on the past for an even one, which has no centre row.
    """
    if window < 2 or len(score) == 0:
        return score
    padded = np.pad(score, window, mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    first = window - window // 2
    return averaged[first : first + len(score)].astype(np.float32)


def _candidates(
    score: np.ndarray, state: SoftKnnState, fraction: float, min_rows: int = 0
):
    """Peaks with non-maximum suppression, each grown to ``fraction`` of its own height.

    ``score`` is **one continuous run**, not the whole trajectory, and that is what keeps a
    match from spanning a tracking gap (§20's ``runs`` contract). A gap is a jump in *time*,
    not in row index — ``clip_query.segment_runs`` splits on the ``t_ns`` step and its runs
    are row-contiguous — so over the whole array there is no boundary here to stop the growth
    below, and a match spans minutes of unrecorded footage while looking like any other.

    The ``score <= 0`` skip is the second, narrower half: rows in *no* run keep a score of 0,
    their threshold would be ``0 * fraction == 0``, and every neighbour clears it, so one
    seeded there would grow until the length cap.

    Taken highest-first, so a strong match is never suppressed by a weaker neighbour — the
    same ordering property the removed `dtw`'s extraction relied on, and what makes a stricter
    cutoff a subset of a looser one rather than a different answer.

    ``taken`` does double duty: it enforces the separation *and* stops one candidate growing
    through another, so extents never overlap and no row belongs to two instances.

    A span shorter than ``min_rows`` is rejected **here, before it can suppress anything**:
    it is neither kept nor marked ``taken``, so a lower neighbour may still grow through it.
    Filtering after formation would lose that neighbour — a brief spike beside a longer, lower
    bump suppresses the bump, then is itself dropped, and the stretch returns nothing. Its rows
    go into ``no_seed`` instead, so they are not each re-tried as a seed. A kept span reports
    the highest score inside it, which is its seed's unless it grew through a rejected spike.
    """
    order = np.argsort(-score, kind="stable")
    taken = np.zeros(len(score), dtype=bool)
    no_seed = np.zeros(len(score), dtype=bool)
    spans: list[tuple[int, int, float]] = []
    reach = max(state.separation_rows, 1)
    cap = max(state.max_extent_rows, 1)
    for row in order:
        if taken[row] or no_seed[row] or score[row] <= 0.0:
            continue
        threshold = score[row] * fraction
        lo = hi = int(row)
        while lo > 0 and hi - lo + 1 < cap and score[lo - 1] >= threshold:
            if taken[lo - 1]:
                break
            lo -= 1
        while hi < len(score) - 1 and hi - lo + 1 < cap:
            if taken[hi + 1] or score[hi + 1] < threshold:
                break
            hi += 1
        if hi - lo + 1 < min_rows:
            no_seed[lo : hi + 1] = True
            continue
        taken[max(0, lo - reach) : min(len(score), hi + reach + 1)] = True
        spans.append((lo, hi, float(score[lo : hi + 1].max())))
    spans.sort(key=lambda s: -s[2])
    return spans


@register_query_method("soft_knn")
class SoftKnnQueryMethod(QueryMethod):
    """Score every frame by a soft k-NN vote, then form instances from the score's peaks."""

    method_id = "soft_knn"
    config_model = SoftKnnQueryConfig
    shard_suffix = ".npz"
    score_meaning = (
        "confidence that the stretch is an instance of the behavior: [0, 1], higher is "
        "better. The stretch's **peak** per-frame score, where a frame's score is a "
        "softmax-weighted vote over its k nearest labelled frames — the example frames "
        "against a background drawn from the rest of the recordings. It is the peak rather "
        "than the mean of the stretch because the cutoff selects peaks, so this is the only "
        "reading under which 'confidence at or above the cutoff' describes what was kept. "
        "Calibrated against the background, not against precision: only an audit can say "
        "what a given value means."
    )
    # The first method that can use them. Non-examples join the labelled index alongside the
    # drawn background, so a user who knows what the behavior is *not* can say so.
    negatives = NegativeSupport(support="optional")
    # Cohort scope, deliberately: a background drawn per recording calibrates confidence to
    # that recording, so one cutoff would mean different things in different animals (§20).
    sampling = SamplingRequest(size=500, scope="cohort", strategy="uniform")
    exemplars = ExemplarBounds(
        min_clips=1,
        # Cost is nearly flat in the clip count — every exemplar frame joins one index, and
        # the labelled set is dominated by the background — so this is a sanity bound rather
        # than a cost one.
        max_clips=100,
        min_total_rows=2,
        max_total_rows=None,
        # `prepare` vstacks every clip into one index and reads nothing but its length back,
        # so a clip that crosses a tracking gap contributes exactly the frames a clip that
        # does not would. Refusing it was the removed `dtw`'s path premise applied to a method with
        # no path — and it was the commonest way a user's marked clip was rejected.
        requires_contiguous=False,
    )

    def prepare(
        self, exemplars, params, *, negatives=None, sample=None
    ) -> SoftKnnState:
        """Pool the labelled index and derive the row scales from the exemplars."""
        cfg = self._params(params)
        positives = [np.asarray(e, dtype=np.float32) for e in exemplars if len(e)]
        if not positives:
            raise ValueError("soft_knn needs at least one non-empty example clip")
        blocks = list(positives)
        blocks.extend(
            np.asarray(n, dtype=np.float32) for n in (negatives or ()) if len(n)
        )
        n_positive = sum(len(p) for p in positives)
        if sample is not None and len(sample):
            blocks.append(np.asarray(sample, dtype=np.float32))
        elif self.sampling is not None:
            # The module draws it; arriving without one means the run was assembled wrong,
            # and scoring against the exemplars alone would silently return a different
            # (uncalibrated) number rather than fail.
            raise ValueError(
                "soft_knn scores against a background and none was drawn; the module draws "
                "one per run before searching (§20)"
            )
        # Lengths in rows are derived from the exemplars, once, and travel in the state —
        # the same shape the removed `dtw` method's band rows had. A user sets multiples; the rows follow
        # whatever cadence this recording was embedded at.
        #
        # Rows, not wall-clock: a clip spanning a tracking gap holds fewer rows than its
        # duration implies, and rows are the right unit anyway because the trajectory these
        # scale — the smoothing width, the separation, the extent cap — is also only the
        # observed rows. The gap is absent from both sides, so they stay commensurate.
        median_rows = int(np.median([len(p) for p in positives]))
        return SoftKnnState(
            labeled=retrieval._l2_normalize(np.vstack(blocks)),
            n_positive=n_positive,
            smooth_rows=max(2, round(median_rows * cfg.smoothing_mult)),
            separation_rows=max(1, median_rows // 2),
            max_extent_rows=max(1, round(median_rows * cfg.duration_max_mult)),
        )

    def search(
        self,
        state,
        trajectory,
        runs,
        params,
        *,
        row_offset=0,
        progress_cb=None,
        min_rows=0,
    ) -> SoftKnnResult:
        """Score this chunk, smooth it, and form its candidates. **No cutoff is applied.**

        Scored and smoothed **per continuous run**, never across one: a tracking gap is not a
        moment where the animal held still, and averaging over it would invent a score for
        rows nobody observed. Rows outside every run keep a score of 0, so nothing is ever
        found in a gap.
        """
        cfg = self._params(params)
        emb = np.asarray(trajectory, dtype=np.float32)
        spans = list(runs) if runs else [(0, len(emb))]
        score = np.zeros(len(emb), dtype=np.float32)
        total = sum(max(hi - lo, 0) for lo, hi in spans) or 1
        done = 0
        for lo, hi in spans:
            if hi <= lo:
                continue
            width = hi - lo
            raw = retrieval.soft_knn_frame_scores(
                emb[lo:hi],
                state.labeled[: state.n_positive],
                state.labeled[state.n_positive :],
                k=cfg.k,
                tau=cfg.tau,
                block=SCORE_BLOCK,
                progress_cb=(
                    None
                    if progress_cb is None
                    else lambda f, _d=done, _w=width: progress_cb((_d + f * _w) / total)
                ),
            )
            score[lo:hi] = _smooth(raw, state.smooth_rows)
            done += width
        if progress_cb is not None:
            progress_cb(1.0)
        # Per run, so a candidate can never grow across a gap — see `_candidates`, where a
        # zero-scored gap row otherwise seeds a match that grows until the length cap.
        spans_found = [
            (lo + a, lo + b, peak)
            for lo, hi in spans
            if hi > lo
            for a, b, peak in _candidates(
                score[lo:hi], state, cfg.extent_fraction, min_rows
            )
        ]
        spans_found.sort(key=lambda s: -s[2])
        return SoftKnnResult(
            score=score,
            start=np.array([s for s, _e, _p in spans_found], dtype=np.int64)
            + row_offset,
            end=np.array([e for _s, e, _p in spans_found], dtype=np.int64) + row_offset,
            peak=np.array([p for _s, _e, p in spans_found], dtype=np.float32),
            row_offset=row_offset,
        )

    def instances_at(self, result, cutoff: float, params) -> list:
        """The candidates whose peak clears ``cutoff`` — a filter, not a search.

        O(candidates) and **monotonic**: extents were fixed before any cutoff existed, so
        raising it can only remove instances. §20 permits a method to re-form extents and
        warns that the count may not fall; this one never does, which is what lets the
        calibration curve be read as a straightforward frequency curve.
        """
        keep = np.nonzero(result.peak >= cutoff)[0]
        return [
            QueryCandidate(
                start_row=int(result.start[i]),
                end_row=int(result.end[i]),
                confidence=float(result.peak[i]),
                # The native score *is* the confidence here — a k-NN vote is already a
                # calibrated-looking [0, 1] — so there is no second number to carry.
                score=float(result.peak[i]),
                # Every exemplar's frames are pooled into one index, so no instance is
                # attributable to a particular clip. 0 rather than a guess.
                example_idx=0,
            )
            for i in keep
        ]

    def merge(self, chunk_results, state, params=None) -> SoftKnnResult:
        """Stitch the chunks, each owning the candidates whose **peak** it resolved fully.

        A peak needs ``state.context_rows`` on both sides to grow to its true extent, so a
        chunk owns ``[offset + C, offset + n - C)`` — extended to the trajectory's own ends
        for the first and last chunk, which have no neighbour to borrow from. Because
        ``chunk_overlap_rows`` declares ``2 * C``, consecutive owned ranges abut exactly: the
        module advances a chunk by ``target - overlap``, so the next chunk's interior starts
        precisely where this one's ended. Declaring ``C`` instead would leave a ``C``-row band
        owned by nobody, and every match peaking inside it would vanish.
        """
        ordered = sorted(chunk_results, key=lambda r: r.row_offset)
        if not ordered:
            raise ValueError("merge needs at least one chunk result")
        context = state.context_rows
        score_parts, starts, ends, peaks = [], [], [], []
        cursor = ordered[0].row_offset
        for index, chunk in enumerate(ordered):
            first, last = index == 0, index == len(ordered) - 1
            lo = chunk.row_offset if first else chunk.row_offset + context
            hi = chunk.row_offset + chunk.n_rows - (0 if last else context)
            if lo != cursor:
                raise ValueError(
                    f"chunk results do not tile: rows [{cursor}, {lo}) are owned by no "
                    "chunk. Chunks must overlap by the method's declared overlap."
                )
            score_parts.append(
                chunk.score[lo - chunk.row_offset : hi - chunk.row_offset]
            )
            owned = (chunk.start + (chunk.end - chunk.start) // 2 >= lo) & (
                chunk.start + (chunk.end - chunk.start) // 2 < hi
            )
            starts.append(chunk.start[owned])
            ends.append(chunk.end[owned])
            peaks.append(chunk.peak[owned])
            cursor = hi
        order = np.argsort(-np.concatenate(peaks), kind="stable")
        return SoftKnnResult(
            score=np.concatenate(score_parts),
            start=np.concatenate(starts)[order],
            end=np.concatenate(ends)[order],
            peak=np.concatenate(peaks)[order],
            row_offset=ordered[0].row_offset,
        )

    def chunk_overlap_rows(self, state, params) -> int:
        """Twice the context a peak needs, which is what makes the owned ranges tile.

        The module advances chunks by ``target - overlap``. With ``2 * context`` each chunk's
        safe interior is exactly one stride wide and consecutive interiors abut; with
        ``context`` they would leave a gap the width of the context between them, and a match
        peaking there would be found by neither chunk. See ``merge``.
        """
        return 2 * state.context_rows

    def candidate_cap(self, params) -> int | None:
        """Nothing truncates: suppression already bounds the count. See the module docstring."""
        return None

    def serialize(self, result, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:  # exactly this path, no appended suffix
            np.savez_compressed(
                handle,
                score=result.score,
                start=result.start,
                end=result.end,
                peak=result.peak,
                row_offset=np.int64(result.row_offset),
            )

    def load(self, path: Path) -> SoftKnnResult:
        with np.load(path) as data:
            missing = set(_SHARD_ARRAYS) - set(data.files)
            if missing:
                raise ValueError(
                    f"soft_knn shard {path} missing arrays {sorted(missing)} "
                    "(not a soft_knn shard, or regenerate it with the current tool)"
                )
            return SoftKnnResult(
                score=data["score"],
                start=data["start"],
                end=data["end"],
                peak=data["peak"],
                row_offset=int(data["row_offset"]),
            )

    @staticmethod
    def _params(params) -> SoftKnnQueryConfig:
        """Accept either a validated model or the raw dict the job carries."""
        if isinstance(params, SoftKnnQueryConfig):
            return params
        return SoftKnnQueryConfig.model_validate(params or {})
