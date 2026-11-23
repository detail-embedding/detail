"""The §19 reliability audit — ``audit.csv`` beside a behavior output.

Auditing is **optional**. A user may set a cutoff and take the instances without reviewing
any of them, and the artifact is complete and valid without this file. What is not optional
is that the *absence* of an audit reads as absence: every function here returns ``None``
rather than a default, a proxy, or a plausible-looking zero, because an unaudited count
presented like an audited one is the bug the audit exists to prevent.

Two things are deliberately not stored, and both are the same discipline:

- **The estimate is never a stored number.** Precision at cutoff 0.15 says nothing about
  precision at 0.25, but a stored ``precision: 0.88`` renders identically after the cutoff
  moves. Derived from the sample, it recomputes over whatever audited instances the current
  cutoff retains, with an honest ``n``, and cannot drift from the thing it describes.
- **Decisions are appended, never rewritten.** Human judgment is primary data: re-auditing a
  larger sample later adds rows and supersedes earlier ones without discarding them, and a
  crash costs at most the row being written (``storage/append_log.py``).

Sampling is stratified by **score bin, not top-N**. A user reviewing the highest-ranked
matches is reviewing the easiest ones, and precision comes out optimistic. Drawing across the bins
estimates precision as a function of score instead, and precision at a cutoff is then **each
bin's precision weighted by how many of the cutoff's instances it holds** (:func:`stratified_estimate`).
The weighting is not optional: the draw gives every bin the same share of clips however many
instances each holds, so a plain proportion over the judged clips over-counts the thin, high-scoring
bins. Measured on a skewed score distribution, the plain proportion read 0.62 where the truth was
0.40, and its interval covered the truth 17% of the time.

**The bins are fixed per label**, from the label's candidates at no cutoff, and recorded in
``audit.yaml`` at the first draw, so the draw and the estimate cut the scores at the same
places. The draw is gone — the review of every clip replaced it (§19, §20) — and these readers
are what an audit already on disk is read with.

**Every function here counts bouts, not rows.** ``instances.csv`` is one row per
``(bout, label)``, so a cluster the user named twice appears twice over one span carrying one
``instance_id`` and one human decision. Counting rows therefore counts that decision twice: on
a two-bout artifact where one bout has two names, precision reported 0.667 where the truth was
0.5, and a stratified draw spent a whole bin's budget showing the reviewer the same clip twice. Nothing about the arithmetic looks wrong
while it happens — which is exactly why §19 introduced ``instance_id``. So each entry point
collapses through ``behavior_output.bouts`` first, and a new one must too.

(The spec calls these score *bands*. In this codebase ``band_s`` / ``band_ns`` / ``band_rows``
already mean the DTW warp tolerance, so the code says ``bin``.)
"""

import numpy as np

from pathlib import Path

from detail.core.utils import utc_timestamp
from detail.storage import behavior_output as bo
from detail.storage import state_yaml
from detail.storage.append_log import AppendLog

AUDIT_COLUMNS = ["instance_id", "score", "human_label", "drawn_at", "answer"]

#: What the ``answer`` cell holds for "none of these" — the reviewer said the instance is not even
#: the outermost label it carries. A label path is never the bare separator (``name_nodes``
#: refuses a name containing it), so this cannot collide with a real answer, and it has to differ
#: from the empty cell, which means "no answer recorded" (a draw, an unclear, a legacy row).
NOTHING_CELL = bo.PATH_SEPARATOR


def ruled_out(path: str) -> str:
    """The ``answer`` cell for "not *path*", saying nothing about the labels above it.

    The separator and then the path: a label path never starts with the separator, and
    :data:`NOTHING_CELL` is the same form with nothing after it (not even the outermost label).
    It is what a check of a sub-label records for "no": asked "is this *climb front wall*?", a
    reviewer who says no has not said whether it is *climb and rear*.
    """
    return NOTHING_CELL + path


def _ruled(answer: str) -> str | None:
    """The path a :func:`ruled_out` answer rules out, ``""`` for "none of these", else ``None``."""
    return answer[len(NOTHING_CELL) :] if answer.startswith(NOTHING_CELL) else None


#: How much of a stored audit must still apply to the instances being scored before a reliability
#: number is reported at all.
#:
#: **A policy choice, not a measurement.** The literature supports *having* a floor — condensed-list
#: scoring reports the coverage drop and declines below one — and gives no principled value for it.
#: Said plainly here because a constant that looks derived invites being read as one, which is the
#: "nominal quantity fed into a computation that needs a measured one" mistake.
MIN_COVERAGE = 0.8

# A default that keeps each bin big enough to mean something on the sample sizes a person
# will actually label (tens, not thousands). Callers override it.
DEFAULT_BINS = 10


class AuditRow:
    """One human decision about one instance.

    ``human_label`` is tri-state: ``True`` (the instance really is the behavior), ``False``
    (it is not), or ``None`` — *either* not yet judged or judged unclear. Readers must test
    ``is True`` / ``is False`` and never truthiness, or an undecided instance counts as a
    rejection and precision reads low for a sample nobody finished.

    ``decided`` separates the two ``None``s. A metric wants ``human_label is not None`` (an
    unclear carries no evidence either way and is dropped); anything asking *has this been
    looked at* wants ``decided``, or an unclear is offered again as though unseen.
    """

    __slots__ = ("instance_id", "score", "human_label", "drawn_at", "decided", "answer")

    def __init__(
        self,
        instance_id: str,
        score: float | None,
        human_label: bool | None,
        drawn_at: str,
        decided: bool | None = None,
        answer: str = "",
    ):
        self.instance_id = instance_id
        self.score = score
        self.human_label = human_label
        self.drawn_at = drawn_at
        # Inferred, not defaulted to False: a True/False label is self-evidently a decision,
        # so only an unclear has to say so, and a caller that forgets cannot silently
        # downgrade a recorded judgment to "never looked at".
        self.decided = (human_label is not None) if decided is None else decided
        #: The label path the reviewer said is true, :data:`NOTHING_CELL` for "none of these",
        #: :func:`ruled_out` for "not this one" (nothing said above it), or ``""`` when no path
        #: was recorded — see :func:`verdict` for how each is read.
        self.answer = answer

    def __eq__(self, other) -> bool:
        if not isinstance(other, AuditRow):
            return NotImplemented
        return self._key() == other._key()

    def __repr__(self) -> str:
        return (
            f"AuditRow(instance_id={self.instance_id!r}, score={self.score!r}, "
            f"human_label={self.human_label!r}, drawn_at={self.drawn_at!r}, "
            f"decided={self.decided!r}, answer={self.answer!r})"
        )

    def _key(self) -> tuple:
        return (
            self.instance_id,
            self.score,
            self.human_label,
            self.drawn_at,
            self.decided,
            self.answer,
        )


def _log(run_dir: Path | str) -> AppendLog:
    return AppendLog(Path(run_dir) / bo.AUDIT_FILENAME, AUDIT_COLUMNS)


#: What the ``human_label`` cell holds for a decision of "unclear".
#:
#: An empty cell means **no decision yet** and this means **decided, and not sure**. They were
#: the same value until this constant existed, and nothing failed: the estimate is right either
#: way, because it drops both. What it cost is everything else — an unclear was re-offered by
#: the one-by-one draw as though it had never been seen, so an ambiguous clip could be handed back
#: forever, and no surface could list a reviewer's unclears to resolve them.
#:
#: Old logs wrote an empty cell for both, so an unclear recorded before this reads as
#: undecided, which is exactly its behaviour then. Nothing to migrate; a widening of meaning,
#: not of schema.
UNCLEAR_CELL = "unclear"


def _bool_cell(value: bool | None, *, decided: bool) -> str:
    if not decided:
        return ""
    return UNCLEAR_CELL if value is None else ("true" if value else "false")


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "" or value.strip().lower() == UNCLEAR_CELL:
        return None
    return value.strip().lower() == "true"


def _parse_decided(value: str | None) -> bool:
    """Whether a row records a *decision*, unclear included, rather than only a draw."""
    return bool(value is not None and value != "")


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def append_audit(
    run_dir: Path | str,
    instance_id: str,
    *,
    score: float | None = None,
    human_label: bool | None = None,
    drawn_at: str | None = None,
    decided: bool | None = None,
    answer: str = "",
) -> AuditRow:
    """Record one row, appended durably. Returns the row written.

    ``decided`` marks a **human decision** rather than a draw. Left unset it is inferred from
    ``human_label``, because a ``True``/``False`` is self-evidently one — so the only caller
    that has to say it is the one recording an **unclear**, whose ``None`` would otherwise be
    indistinguishable from "not looked at yet" (see :data:`UNCLEAR_CELL`). Inferring rather
    than defaulting to ``False`` matters: a caller that passes a label and forgets the flag
    would silently write a blank cell and lose the decision.

    ``answer`` is the label path the reviewer picked when the instance's label is nested
    (``"rearing"`` for "rearing, not supported"), or :data:`NOTHING_CELL`. The caller still
    writes ``human_label`` — whether the answer is the instance's own label — so every reader
    that knows nothing of paths reads exactly what it read before the column existed.
    """
    row = AuditRow(
        instance_id=str(instance_id),
        score=score,
        human_label=human_label,
        drawn_at=drawn_at or utc_timestamp(),
        decided=decided,  # `AuditRow` owns the inference; two copies would drift
        answer=answer,
    )
    _log(run_dir).append(
        [
            row.instance_id,
            "" if row.score is None else row.score,
            _bool_cell(row.human_label, decided=row.decided),
            row.drawn_at,
            row.answer,
        ]
    )
    return row


def read_audit(run_dir: Path | str) -> list[AuditRow]:
    """Every recorded decision, in the order it was made — the full trail, duplicates kept."""
    return [
        AuditRow(
            instance_id=row.get("instance_id") or "",
            score=_parse_float(row.get("score")),
            human_label=_parse_bool(row.get("human_label")),
            drawn_at=row.get("drawn_at") or "",
            decided=_parse_decided(row.get("human_label")),
            answer=row.get("answer") or "",
        )
        for row in _log(run_dir).rows()
    ]


def under(path: str, target: str) -> bool:
    """Whether ``path`` is ``target`` or a label nested inside it."""
    return path == target or path.startswith(target + bo.PATH_SEPARATOR)


def verdict(row: AuditRow, own_path: str, target: str) -> bool | None:
    """Whether a judgment says an instance labelled ``own_path`` really is ``target``.

    ``target`` is ``own_path`` or one of its ancestors — the audit scores a label over every
    instance under it. One judgment answers every level at once, which is the point of
    recording *which* level was true rather than a yes to the deepest:

    - an answer at or under ``target`` → yes ("rearing, not supported" is still rearing);
    - an answer outside it, or "none of these" → no;
    - "not *X*" (:func:`ruled_out`) → no for ``target`` at or under *X*, and ``None`` above it:
      ruling out a sub-label says nothing about its parent;
    - unclear or undecided → ``None``, dropped like any unclear.

    **A row written before the answer column existed** is a yes/no to ``own_path`` alone. Its
    yes is an answer of ``own_path``; its no rules out ``own_path`` and says nothing about the
    ancestors — "not supported rearing" may still be rearing — so it counts only when ``target``
    is ``own_path`` and is dropped for every coarser label rather than guessed.
    """
    if row.human_label is None and not row.answer:
        return None
    ruled = _ruled(row.answer or "")
    if ruled is not None:
        return False if not ruled or under(target, ruled) else None
    if row.answer:
        return under(row.answer, target)
    if row.human_label is True:
        return under(own_path, target)
    return False if own_path == target else None


#: How a label path reads on screen: "rearing › supported". The stored separator is a slash,
#: which reads as a file path.
PATH_DISPLAY = " › "


def answer_choices(own_path: str) -> list[tuple[str, str, bool]]:
    """``[(text, answer, is_match)]`` — what a reviewer may say about an instance, deepest first.

    One choice per level of the instance's label, then "none of these". For ``rearing/supported``:
    "rearing › supported", "rearing, not supported", "not rearing". Each carries the ``answer``
    to record and the ``is_match`` (whether it is the instance's own label) that every reader
    without paths still reads.

    **A flat label keeps the plain yes/no**, with an empty ``answer`` — exactly the row the audit
    wrote before answers existed, so a query run's ``audit.csv`` does not change shape in meaning.
    Unclear is the caller's to add: it records no answer at all.
    """
    parts = own_path.split(bo.PATH_SEPARATOR)
    if len(parts) == 1:
        return [(own_path, "", True), (f"not {own_path}", "", False)]
    out = [(PATH_DISPLAY.join(parts), own_path, True)]
    for depth in range(len(parts) - 1, 0, -1):
        kept = PATH_DISPLAY.join(parts[:depth])
        out.append(
            (
                f"{kept}, not {parts[depth]}",
                bo.PATH_SEPARATOR.join(parts[:depth]),
                False,
            )
        )
    out.append((f"not {parts[0]}", NOTHING_CELL, False))
    return out


def answer_options(scope: str, branch) -> list[tuple[str, str, bool]]:
    """``[(text, answer, is_match)]`` — the answers a whole check offers, the same for every clip.

    Built from the **scope**, never from a clip: one answer per label in the scope's branch
    (the scope itself reads "*scope*, other" when it has sub-labels), then "not *scope*".
    Building the buttons from each clip's own label made S mean "yes" on one clip and a
    sub-label on the next — a keyboard that re-maps itself under a reviewer's fingers. A fixed
    list also no longer says which label a clip was drawn from, so the answer is the
    reviewer's, not a confirmation of the label shown.

    **Only the scope is asked about**, never the labels above it. Checking a sub-label once also
    offered "*parent*, not *sub-label*" and "not *parent*": two ways to say no that differed only
    about a label the reviewer had not chosen to check. "not *scope*" is :func:`ruled_out`, which
    :func:`verdict` reads as no for the scope and as nothing for its parent; a yes still counts
    for the parent, since a *climb front wall* is a *climb and rear*. A root-level scope's "not"
    is :data:`NOTHING_CELL`, the same claim, and a root-level label with no sub-labels keeps the
    flat yes/no of :func:`answer_choices`, so a query run's ``audit.csv`` does not change.

    Answers are label paths, so :func:`verdict` reads them unchanged: "climb corner" given for a
    clip drawn from "climb front wall" is a miss for front wall and a hit for "climb and rear".
    ``is_match`` is relative to a clip whose own label is the scope; a caller recording a
    judgment derives ``human_label`` from the clip's own path (:func:`human_label_for`).
    """
    labels = sorted(
        {path for path in branch if under(path, scope)} | {scope},
        key=lambda path: path.split(bo.PATH_SEPARATOR),
    )
    parts = scope.split(bo.PATH_SEPARATOR)
    if labels == [scope] and len(parts) == 1:
        return answer_choices(scope)
    if labels == [scope]:
        out = [(parts[-1], scope, True)]
    else:
        out = [
            (
                PATH_DISPLAY.join(path.split(bo.PATH_SEPARATOR))
                + (", other" if path == scope else ""),
                path,
                path == scope,
            )
            for path in labels
        ]
    nothing = NOTHING_CELL if len(parts) == 1 else ruled_out(scope)
    out.append((f"not {parts[-1]}", nothing, False))
    return out


def human_label_for(answer: str, is_match: bool | None, own_path: str) -> bool | None:
    """Whether an answer from :func:`answer_options` says the clip *is* its own label — what
    ``human_label`` records for readers that know nothing of paths. An answer naming a path is
    a yes exactly when it is the clip's own; the flat yes/no (empty answer) keeps its own.
    """
    if is_match is None:
        return None
    ruled = _ruled(answer)
    if ruled is not None:
        return False if not ruled or under(own_path, ruled) else None
    if answer:
        return answer == own_path
    return is_match


def answer_text(row: AuditRow, own_path: str) -> str:
    """What a standing decision said, in the words the reviewer chose it by."""
    if not row.decided:
        return ""
    if row.human_label is None and not row.answer:
        return "unclear"
    for text, answer, is_match in answer_choices(own_path):
        if (row.answer or "") == answer and (answer or row.human_label is is_match):
            return text
    ruled = _ruled(row.answer or "")
    if ruled == "":
        return f"not {own_path.split(bo.PATH_SEPARATOR)[0]}"
    if ruled:
        return f"not {ruled.rsplit(bo.PATH_SEPARATOR, 1)[-1]}"
    if row.answer and ruled is None:
        # A label the clip does not carry, from a check's fixed list: a sibling sub-label.
        return PATH_DISPLAY.join(row.answer.split(bo.PATH_SEPARATOR))
    return "yes" if row.human_label else "no"


def _judgment(decision: AuditRow, row, target: str | None) -> bool | None:
    """The yes/no a decision contributes to an estimate — about ``target`` when one is given."""
    if target is None:
        return decision.human_label
    return verdict(decision, row.label_path or row.label, target)


def _scoped(instances, target: str | None) -> list:
    """The bouts an estimate is over: every bout, or those under ``target``."""
    rows = bo.bouts(instances)
    if target is None:
        return rows
    return [row for row in rows if under(row.label_path or row.label, target)]


def latest_decisions(run_dir: Path | str) -> list[AuditRow]:
    """One row per instance — the most recent decision about each.

    Re-auditing appends rather than edits, so an instance a user revisited has several rows
    and the last one is what they think now. Order follows each instance's *first* appearance,
    so a re-decision does not reshuffle the list under a reader.
    """
    latest: dict[str, AuditRow] = {}
    order: list[str] = []
    for row in read_audit(run_dir):
        if row.instance_id not in latest:
            order.append(row.instance_id)
        latest[row.instance_id] = row
    return [latest[key] for key in order]


def decision_coverage(
    run_dir: Path | str, universe, *, since: str | None = None
) -> dict:
    """How much of the stored audit still applies to this run.

    ``universe`` is the run's **whole current instance set** — every label, no cutoff — not the
    slice being estimated. That distinction is the whole meaning of the number: this asks
    whether a judged instance still *exists*, never whether it is currently *selected*. Handed a
    slice it counts ordinary selection as loss, and every caller has a slice to hand: scoping to
    one behaviour put two-behaviour runs at 50% coverage, and raising a cutoff drove a run to
    8%, both refusing with "the vocabulary changed" when nothing had.

    ``{"n_decisions", "n_applicable", "coverage"}`` over **judged** decisions only — a drawn but
    unjudged row is not evidence, so counting it would understate coverage.

    This exists because the alternative is the failure IR's test-collection literature spent two
    decades replacing. An audit is keyed by ``instance_id``, a content hash of the bout *as
    produced*; re-cut the vocabulary and the bouts change, so their ids change, so decisions drop
    out of the intersection with nothing failing. That is not merely lossy: whether an instance
    survives a re-cut is **not independent of what the audit measures** — clusters merge because
    their contents resemble each other, which correlates with whether they were judged right — so
    the survivors are a biased subset and the estimate flatters the method (Zobel, SIGIR 1998, on
    scoring unjudged documents as absent).

    ``coverage`` is ``None`` when nothing has been judged, which is not the same as 0.0: no
    evidence differs from evidence that no longer applies.
    """
    judged = [
        row
        for row in latest_decisions(run_dir)
        if row.human_label is not None and _after(row, since)
    ]
    if not judged:
        return {"n_decisions": 0, "n_applicable": 0, "coverage": None}
    current = {row.instance_id for row in bo.bouts(universe)}
    applicable = sum(1 for row in judged if row.instance_id in current)
    return {
        "n_decisions": len(judged),
        "n_applicable": applicable,
        "coverage": applicable / len(judged),
    }


def _coverage_refusal(coverage: dict, min_coverage: float) -> str | None:
    """Why a reliability number is withheld, or ``None`` when it may be reported."""
    value = coverage["coverage"]
    if value is None or value >= min_coverage:
        return None
    return (
        f"only {coverage['n_applicable']} of {coverage['n_decisions']} audited instances still "
        f"exist ({value:.0%} coverage, floor {min_coverage:.0%}) — the vocabulary changed since "
        "this was audited, and the instances that survived are not a random sample of it. Audit "
        "more, or read the number from the snapshot it was audited against."
    )


def has_audit(run_dir: Path | str) -> bool:
    """Whether an audit accompanies this artifact — what the manifest's ``audited`` reports."""
    return (Path(run_dir) / bo.AUDIT_FILENAME).is_file()


def retained(score: float | None, cutoff: float | None, direction: str) -> bool:
    """Whether an instance at ``score`` survives ``cutoff``.

    ``direction`` is the artifact's declared ``score.direction``, and reading it rather than
    assuming is the entire reason §19 makes it non-optional: a score may be a *distance* (the
    default convention, where "above the cutoff" means ``score <= cutoff``) or a similarity.
    Assuming the wrong one silently inverts precision and reports a confident wrong number.
    """
    if cutoff is None or score is None:
        return True
    return score <= cutoff if direction == "lower_is_better" else score >= cutoff


def score_direction(manifest: dict) -> str:
    """The artifact's declared score direction (default: the distance convention)."""
    return (manifest.get("score") or {}).get("direction") or "lower_is_better"


def bin_edges(scores, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """``n_bins + 1`` equal-width edges spanning the observed scores.

    Equal-width over the *observed* range rather than a fixed 0–1: a method's scores may
    occupy any interval, and binning over an interval it never reaches puts every instance in
    one bin, which is the top-N sample this is meant to avoid.
    """
    values = np.asarray([s for s in scores if s is not None], dtype=np.float64)
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def _bin_of(score: float, edges: np.ndarray) -> int:
    """Which bin a score falls in; the top edge belongs to the last bin."""
    idx = int(np.searchsorted(edges, score, side="right") - 1)
    return min(max(idx, 0), len(edges) - 2)


def read_bins(run_dir: Path | str) -> dict[str, np.ndarray]:
    """``{source_label: edges}`` recorded in ``audit.yaml``, or ``{}`` when none are.

    A label with no entry was never drawn from since bins were recorded; its caller derives its
    edges the way the draw's first recording did (they are a pure function of the label's
    candidates).
    """
    path = Path(run_dir) / bo.AUDIT_BINS_FILENAME
    if not path.is_file():
        return {}
    record = state_yaml.read_state_yaml(path)
    return {
        str(label): np.asarray(entry["edges"], dtype=np.float64)
        for label, entry in (record.get("bins") or {}).items()
    }


def bins_fixed_at(run_dir: Path | str) -> dict[str, str]:
    """``{source_label: when}`` — when each label's bins were recorded (ISO-8601 UTC)."""
    path = Path(run_dir) / bo.AUDIT_BINS_FILENAME
    if not path.is_file():
        return {}
    record = state_yaml.read_state_yaml(path)
    return {
        str(label): str(entry.get("fixed_at") or "")
        for label, entry in (record.get("bins") or {}).items()
    }


def _label_edges(rows, edges: dict | None, n_bins: int = DEFAULT_BINS) -> dict:
    """``{source_label: edges}`` for every label among ``rows`` — given, or derived from them."""
    by_label: dict[str, list] = {}
    for row in rows:
        if row.score is not None:
            by_label.setdefault(str(row.source_label), []).append(row.score)
    given = edges or {}
    return {
        label: (
            np.asarray(given[label], dtype=np.float64)
            if label in given
            else bin_edges(scores, n_bins)
        )
        for label, scores in by_label.items()
    }


def _score_stratum(row, edges: dict) -> tuple:
    """The ``(source_label, bin)`` a scored bout counts toward."""
    label = str(row.source_label)
    return (label, _bin_of(row.score, edges[label]))


#: Judged instances below which a precision is reported as unknown rather than as a number.
#: An audit is designed around tens of labels, so a stratum can easily hold three — and 1.00
#: from n=3 renders identically to 1.00 from n=50 while supporting nothing. Same discipline as
#: an unaudited run reading as unaudited: the honest output here is a refusal, not a figure.
MIN_N_FOR_ESTIMATE = 8


def wilson_interval(
    n_confirmed: int, n: int, z: float = 1.96
) -> tuple[float, float] | None:
    """A 95% Wilson score interval for a proportion, or ``None`` for ``n == 0``.

    Wilson rather than the textbook normal approximation, because this module is designed
    around samples of tens and the normal interval misbehaves badly there — at 9 of 10 it runs
    past 1.0, and at 10 of 10 it collapses to zero width, claiming certainty from ten
    observations. Both are wrong in the direction that flatters the result, and neither looks
    wrong on screen.
    """
    if n <= 0:
        return None
    return _wilson(n_confirmed / n, n, z)


def _wilson(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """The Wilson interval for a proportion ``p`` observed over ``n`` (which may be an
    *effective* n, as a stratified estimate has)."""
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _estimate(
    judged: list,
    *,
    min_n: int = MIN_N_FOR_ESTIMATE,
    coverage: dict | None = None,
    refusal: str | None = None,
) -> dict:
    """The reported form of a set of decisions: ``n``, the rate, and its interval.

    The rate is **always** returned when anything was judged, and so is the interval. Hiding
    the rate below ``min_n`` was the first attempt and it is the wrong layer: it destroys a
    number a caller may legitimately want (a test asserting a rate, an export) to enforce
    something that is really about rendering.

    ``below_min_n`` is that rendering signal instead. A surface shows "4 judged — too few to
    estimate" rather than `1.00`, because 1.00 from n=3 renders identically to 1.00 from n=50
    and supports nothing. The interval says the same thing quantitatively — n=3 at 1.00 is
    0.44–1.00 — which is why it travels beside every rate here rather than being offered as an
    option a caller can forget to ask for.
    """
    confirmed = sum(1 for value in judged if value is True)
    return {
        "n": len(judged),
        "n_confirmed": confirmed,
        # Two different reasons a number should not be trusted, and they compose rather than
        # compete. `below_min_n` is *precision of the estimate* — too few decisions to say
        # much, so the rate is still returned and a surface renders words instead. `refusal`
        # is *coverage of the population* (#245): decisions drawn against a vocabulary that
        # no longer exists are not a sample of this one, so there is no rate to report and
        # the caller gets `None` with a reason.
        "precision": (
            None if refusal else (confirmed / len(judged)) if judged else None
        ),
        "interval": wilson_interval(confirmed, len(judged)),
        "interval_method": "wilson95" if judged else None,
        "below_min_n": bool(judged) and len(judged) < min_n,
        **(coverage or {}),
        "reason": refusal,
    }


def estimate(judged: list) -> dict:
    """The reported form of a list of yes/no judgments — ``n``, the rate and its interval.

    The same form every precision here returns, for a caller that has grouped judgments its own
    way (by cluster, say) rather than by cutoff, recording or score.
    """
    return _estimate(judged)


#: How narrow a 95% confidence interval is aimed at by default: ±0.10. A sample-size goal, not a
#: judgment of the label — survey practice uses ±0.05 or ±0.10, and ±0.10 takes 30–60 clips near
#: a precision of 0.9 where ±0.05 takes several hundred.
DEFAULT_GOAL_WIDTH = 0.20


def _after(row: AuditRow, since: str | None) -> bool:
    """Whether a decision was made at or after ``since`` (ISO-8601 UTC, which sorts as text)."""
    return since is None or (row.drawn_at or "") >= since


def stratified_estimate(
    groups: dict,
    *,
    min_n: int = MIN_N_FOR_ESTIMATE,
    coverage: dict | None = None,
    refusal: str | None = None,
) -> dict:
    """Precision over strata of known size: ``groups`` is ``{key: (N_c, [judgments])}``.

    **Why strata.** A label's clusters can each be audited on their own, so a small cluster may
    hold more judged clips than its share. Pooling those judgments would tilt the label's figure
    toward whichever clusters were audited most. Weighting each cluster's
    precision by its share of the label's instances, ``Σ w_c·p_c``, puts that right — the standard
    stratified estimate, valid for any allocation and for judgments drawn before strata existed.

    Strata nobody has judged yet are left out and the rest re-weighted, and ``unjudged_share`` says
    how much of the label that leaves uncovered; a surface calls the figure provisional while it is
    above zero, because "0.90 over half the label" is a different claim from "0.90".

    The interval is Wilson's on the **effective n**, ``p(1−p)/Var``, with each stratum's variance
    taken at ``(x+2)/(n+4)`` so a cluster judged 10 of 10 does not claim zero spread. With one
    stratum it is exactly :func:`_estimate`, so a label of one cluster reads as it always did.
    """
    judged = {
        key: (int(size), [v for v in values if v is not None])
        for key, (size, values) in groups.items()
    }
    by_stratum = {
        key: _estimate(values, min_n=min_n) for key, (_n, values) in judged.items()
    }
    used = {
        key: (size, values)
        for key, (size, values) in judged.items()
        if values and size > 0
    }
    total = sum(size for size, _v in judged.values())
    covered = sum(size for size, _v in used.values())
    unjudged = (1.0 - covered / total) if total else 0.0
    if len(used) <= 1:
        values = next(iter(used.values()))[1] if used else []
        return {
            **_estimate(values, min_n=min_n, coverage=coverage, refusal=refusal),
            "unjudged_share": unjudged,
            "by_stratum": by_stratum,
        }
    p = 0.0
    p_tilde = 0.0
    variance = 0.0
    n = confirmed = 0
    for size, values in used.values():
        weight = size / covered
        yes = sum(1 for v in values if v is True)
        p += weight * yes / len(values)
        shrunk = (yes + 2) / (len(values) + 4)
        p_tilde += weight * shrunk
        variance += weight * weight * shrunk * (1 - shrunk) / len(values)
        n += len(values)
        confirmed += yes
    effective_n = max(p_tilde * (1 - p_tilde) / variance, 1.0) if variance > 0 else n
    return {
        "n": n,
        "n_confirmed": confirmed,
        "precision": None if refusal else p,
        "interval": _wilson(p, effective_n),
        "n_effective": effective_n,
        "interval_method": "wilson95-stratified",
        "below_min_n": n < min_n,
        **(coverage or {}),
        "reason": refusal,
        "unjudged_share": unjudged,
        "by_stratum": by_stratum,
    }


def ci_width(estimate: dict | None) -> float | None:
    """How wide an estimate's 95% confidence interval is, or ``None`` without one."""
    interval = (estimate or {}).get("interval")
    return None if not interval else float(interval[1] - interval[0])


def _bin_groups(rows, decisions: dict, target: str | None, edges: dict) -> dict:
    """``{(source_label, bin): (N, [judgments])}`` — the score strata :func:`stratified_estimate`
    weights, each sized by how many of ``rows`` it holds.

    ``rows`` are what the cutoff retains, so a bin the cutoff cuts through counts only its
    retained part. A scoreless bout (none in a query run) is a stratum of its own rather than
    dropped, so the sizes still add up to every retained bout.
    """
    groups: dict = {}
    for row in bo.bouts(rows):
        key = (
            _score_stratum(row, edges)
            if row.score is not None
            else (str(row.source_label), None)
        )
        size, values = groups.get(key, (0, []))
        if row.instance_id in decisions:
            values = values + [_judgment(decisions[row.instance_id], row, target)]
        groups[key] = (size + 1, values)
    return groups


def _drawn_before(run_dir: Path | str, rows, decisions: dict) -> int:
    """How many judged ``rows`` were first drawn before their label's bins were recorded.

    Those clips were drawn under whatever binning the draw used then — across every behavior at
    once, for a draw with no behavior chosen — and are weighted by the recorded per-label bins
    now (post-stratified). The figure is still a stratified estimate; this is what says part of
    it was not drawn in the bins it is weighted by.
    """
    fixed = bins_fixed_at(run_dir)
    first: dict[str, str] = {}
    for row in read_audit(run_dir):
        first.setdefault(row.instance_id, row.drawn_at or "")
    return sum(
        1
        for row in bo.bouts(rows)
        if row.score is not None
        and row.instance_id in decisions
        and decisions[row.instance_id].human_label is not None
        # No recorded bins at all sorts after every timestamp, so every draw is "before".
        and first.get(row.instance_id, "") < fixed.get(str(row.source_label), "~")
    )


def precision_by_recording(
    run_dir: Path | str,
    instances,
    *,
    cutoff: float | None = None,
    manifest: dict | None = None,
    min_coverage: float = MIN_COVERAGE,
    universe=None,
    target_path: str | None = None,
    edges: dict | None = None,
) -> list[dict] | None:
    """Precision within each recording, each with its own ``n`` and interval.

    ``target_path`` scores a nested label over every instance under it (see
    :func:`precision_at_cutoff`), and a scored artifact is weighted by score bin within each
    recording, exactly as the headline is (``instances`` and ``edges`` as there).

    The sample is **not** stratified by recording — it is drawn across score bins and falls
    across recordings in proportion to where the instances are. That is a deliberate trade:
    balancing the draw would double the labels for a fixed number of bins, and the budget is a
    person's attention. So this reports what the proportional draw happened to give, and every
    row carries ``below_min_n`` so a surface can refuse rather than print a rate from three
    decisions.

    Reported anyway, rather than left out, because it is the one place a per-recording
    calibration difference becomes visible: a background drawn across the cohort means one
    cutoff can behave differently in different animals (§20), and an overall precision hides
    that completely.
    """
    if not has_audit(run_dir):
        return None
    decisions = {row.instance_id: row for row in latest_decisions(run_dir)}
    if not decisions:
        return None
    direction = score_direction(manifest or {})
    scoped = list(_scoped(instances, target_path))
    label_edges = _label_edges(scoped, edges)
    kept: dict[str, list] = {}
    for row in scoped:
        if retained(row.score, cutoff, direction):
            kept.setdefault(row.recording_id, []).append(row)
    # The same coverage the headline reports, so a per-recording row cannot print a rate the
    # figure above it is refusing to print. It is a property of the run, so it is computed once.
    coverage = decision_coverage(run_dir, instances if universe is None else universe)
    refusal = _coverage_refusal(coverage, min_coverage)
    out = []
    for recording_id in sorted(kept):
        rows = kept[recording_id]
        if any(row.score is not None for row in rows):
            figure = stratified_estimate(
                _bin_groups(rows, decisions, target_path, label_edges),
                coverage=coverage,
                refusal=refusal,
            )
        else:
            figure = _estimate(
                [
                    value
                    for row in rows
                    if row.instance_id in decisions
                    for value in [
                        _judgment(decisions[row.instance_id], row, target_path)
                    ]
                    if value is not None
                ],
                coverage=coverage,
                refusal=refusal,
            )
        out.append({"recording_id": recording_id, "n_instances": len(rows), **figure})
    return out


def precision_at_cutoff(
    run_dir: Path | str,
    instances,
    *,
    cutoff: float | None,
    manifest: dict | None = None,
    min_coverage: float = MIN_COVERAGE,
    universe=None,
    target_path: str | None = None,
    stratum_of: dict | None = None,
    since: str | None = None,
    edges: dict | None = None,
) -> dict | None:
    """Precision over the audited instances the cutoff **retains**, with its ``n``.

    **A scored artifact (a query run) is weighted by score bin**: each ``(label, bin)``'s
    precision counts by how many of the cutoff's instances that bin holds
    (:func:`stratified_estimate`), and the interval is Wilson's on the effective n. The draw
    spreads clips evenly across bins of very different sizes, so the plain share of judged clips
    estimates nothing in particular; on a skewed score distribution it read 0.62 where the truth
    was 0.40. ``instances`` must be the label's **candidates at no cutoff** — the population the
    draw was made from — and ``edges`` (``{source_label: edges}``) the bins it was drawn in
    (``audit.yaml``, :func:`read_bins`); without ``edges`` they are derived from ``instances``,
    which is the same
    thing when ``instances`` is that population. ``n_drawn_before_bins`` counts the judgments
    drawn before their label's bins were recorded, which are weighted by those bins all the same.

    ``stratum_of`` (instance → cluster) makes it a **stratified** estimate over the label's
    clusters, weighted by each one's share of the scoped instances (:func:`stratified_estimate`).
    ``since`` counts only decisions made at or after it — a label whose clusters changed after it
    was audited reports from fresh judgments, so dropping the clusters that scored badly cannot
    raise the figure the dropping was based on. ``n_before`` counts the judgments it set aside,
    so the reader can say how many there were instead of "not audited".

    ``target_path`` asks about a **label** rather than about each instance's own: the estimate
    is over every instance whose ``label_path`` is the target or nested inside it, and each
    judgment is read through :func:`verdict`, so "rearing" gets an exact precision from the same
    judgments that score "rearing/supported". ``None`` reads ``human_label`` as it always did.

    ``None`` when nothing has been audited, and ``n: 0`` with ``precision: None`` when the
    cutoff retains nothing anyone judged — a rate over zero decisions is not a rate.

    Which instances a cutoff retains depends on the artifact's declared ``score.direction``
    (§19), read from ``manifest`` rather than assumed. See :func:`retained`.

    Every return carries ``n_decisions`` / ``n_applicable`` / ``coverage``, and withholds its
    headline number with a ``reason`` when coverage falls below ``min_coverage``: an audit drawn
    against one vocabulary is not a random sample of a different one (see
    :func:`decision_coverage`).
    """
    if not has_audit(run_dir):
        return None
    decisions = {row.instance_id: row for row in latest_decisions(run_dir)}
    if not decisions:
        return None
    direction = score_direction(manifest or {})
    every = decisions
    decisions = {k: v for k, v in decisions.items() if _after(v, since)}
    candidates = list(_scoped(instances, target_path))
    scoped = [row for row in candidates if retained(row.score, cutoff, direction)]
    # What `since` set aside, so a reader can say "0 judged since, 78 before" rather than a
    # bare "not audited yet" beside a label that has been audited.
    n_before = sum(
        1
        for row in scoped
        if row.instance_id in every
        and row.instance_id not in decisions
        and _judgment(every[row.instance_id], row, target_path) is not None
    )
    coverage = decision_coverage(
        run_dir, instances if universe is None else universe, since=since
    )
    refusal = _coverage_refusal(coverage, min_coverage)
    head = {"cutoff": cutoff, "direction": direction, "n_before": n_before}
    if stratum_of is not None:
        groups: dict = {}
        for row in scoped:
            key = stratum_of.get(row.instance_id, row.label_path or row.source_label)
            size, values = groups.get(key, (0, []))
            if row.instance_id in decisions:
                values = values + [
                    _judgment(decisions[row.instance_id], row, target_path)
                ]
            groups[key] = (size + 1, values)
        return {
            **head,
            **stratified_estimate(groups, coverage=coverage, refusal=refusal),
        }
    if any(row.score is not None for row in candidates):
        groups = _bin_groups(
            scoped, decisions, target_path, _label_edges(candidates, edges)
        )
        return {
            **head,
            "n_drawn_before_bins": _drawn_before(run_dir, scoped, decisions),
            **stratified_estimate(groups, coverage=coverage, refusal=refusal),
        }
    judged = [
        value
        for row in scoped
        if row.instance_id in decisions
        for value in [_judgment(decisions[row.instance_id], row, target_path)]
        if value is not None
    ]
    return {**head, **_estimate(judged, coverage=coverage, refusal=refusal)}
