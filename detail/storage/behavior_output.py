"""The §19 behavior output — the shared analysis artifact.

Three files under ``analysis/<an_NN>/``: ``instances.csv`` (labeled spans of behavior in
time), ``coverage.csv`` (what was actually analyzed, so an absent label reads as a real zero
rather than a gap), and ``behavior_output.yaml`` (everything that varies between producers).
Every analysis module that produces labeled behavior writes these, so the layers above them —
descriptive measures, frame-level export, ethograms, comparison across recording metadata —
are written once instead of once per method.

**One writer, many producers.** A module hands ``write_behavior_output`` its pieces; a module
that hand-rolls its own writer is the bug. That is not tidiness: the double-count hazard and
the ``instance_id`` contract below only hold while there is a single implementation of them.

The interval schema is deliberately the §3 ``behavior``-role stream's, so anything that reads
a stream reads a behavior output. Both ends are **inclusive** — a state behavior labels a
frame with ``start <= t <= end`` (``analysis/behavior_labels.py::frames_to_labels``), and
``end_frame`` matches.

Three contracts, each enforced here rather than remembered by producers:

- **``instance_id`` is derived from bout identity, never from row position.** ``audit.csv``
  keys human decisions by it and §21 reprojects this artifact on every cutoff drag or rename,
  so a row counter would silently re-point every stored decision at a different bout after
  the first reprojection.
- **Rows are one-per-label, not one-per-bout.** A cluster carrying two names emits two rows
  over the same span (the schema stays flat and stream-compatible), so summing durations
  double counts and nothing about the arithmetic looks wrong when it does. Counting distinct
  ``instance_id`` is how a measure says it must not.
- **An instance outside ``coverage.csv`` is refused.** Coverage is the only thing separating
  "we looked and it never happened" from "this recording was not in the run", and an instance
  in an uncovered unit means the coverage table is lying about what was analyzed.
"""

import csv
import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from detail.core import atomic
from detail.core.ids import sorted_by_id
from detail.storage import clip_review
from detail.storage import dirscan
from detail.storage import experiment as experiment_store

SCHEMA_VERSION = 1

INSTANCES_FILENAME = "instances.csv"
COVERAGE_FILENAME = "coverage.csv"
MANIFEST_FILENAME = "behavior_output.yaml"
# Written by ``storage/behavior_audit.py``; its presence, or ``clip_review.csv``'s, is what
# ``audited`` reports.
AUDIT_FILENAME = "audit.csv"
# The score-bin edges each label's audit is drawn and weighted by, recorded at its first draw
# (``behavior_audit.fix_bins``).
AUDIT_BINS_FILENAME = "audit.yaml"
# Written by ``storage/frame_view.py`` — one zarr group per (recording_id, subject).
FRAMES_DIRNAME = "frames"
# What we *measured about* each instance, as opposed to what an instance *is* (§19).
# A separate file rather than more columns on `instances.csv`: that one is a contract other
# producers and readers are written against, while this list is expected to grow (position now;
# speed, area and aspect with the per-cluster descriptors). Joined on `instance_id`, which is a
# content hash of the bout's identity and so survives the reprojection that rewrites both.
FEATURES_FILENAME = "instance_features.csv"
FEATURE_COLUMNS = ["instance_id", "cx_norm", "cy_norm"]
#: Whether one of the user's own example or non-example clips overlaps the instance (§19) — a
#: query run's, so the audit's exclusion is visible to anyone reading the artifact or its export.
EXAMPLE_COLUMN = "is_example"

#: The separator between the levels of a ``label_path``. A name containing it would make the
#: path ambiguous, so ``name_nodes`` refuses one; a legacy name that already contains it still
#: writes, and its path is the one place that ambiguity shows.
PATH_SEPARATOR = "/"

INSTANCE_COLUMNS = [
    "recording_id",
    "subject",
    "t_ns",
    "end_t_ns",
    "start_frame",
    "end_frame",
    "label",
    "label_path",
    "source_label",
    "instance_id",
    "score",
]

COVERAGE_COLUMNS = [
    "recording_id",
    "subject",
    "start_t_ns",
    "end_t_ns",
    "observed_ns",
    "n_observed",
]

# What a reader must find to call the file a behavior output at all. The rest fills from
# defaults so a file written by an older schema still reads (the clip_query_csv idiom).
_INSTANCE_REQUIRED = {"recording_id", "t_ns", "end_t_ns", "source_label"}
_COVERAGE_REQUIRED = {"recording_id", "start_t_ns", "end_t_ns", "observed_ns"}

ORIGINS = frozenset({"predicted", "ground_truth"})
LABEL_SPACES = frozenset({"pooled", "per_recording"})
SCORE_DIRECTIONS = frozenset({"lower_is_better", "higher_is_better"})
OVERLAP_MODES = frozenset({"allowed", "exclusive"})
BEHAVIOR_TYPES = frozenset({"state", "point"})

# Older fits say "global" for what §19 calls "pooled". Accepted on read
# and normalized on write, so the two vocabularies cannot drift apart in stored files.
_LABEL_SPACE_SYNONYMS = {"global": "pooled"}

_MISSING_FRAME = -1  # a file that did not record the readability columns


class BehaviorOutputError(ValueError):
    """The pieces handed to the writer do not describe a readable artifact."""


def instance_id_for(
    recording_id: str,
    subject: int | None,
    source_label: str,
    t_ns: int,
    end_t_ns: int,
) -> str:
    """A bout's stable id: a content hash of what makes it that bout.

    Deliberately **not** a row counter. The artifact is regenerated whenever a user drags a
    cutoff or names a cluster (§21) and ``audit.csv`` stores human decisions against these
    ids, so an id that moved with row position would silently re-point every decision at a
    different bout on the next reprojection, with nothing anywhere failing.
    """
    payload = "\x00".join(
        [
            str(recording_id),
            "" if subject is None else str(int(subject)),
            str(source_label),
            str(int(t_ns)),
            str(int(end_t_ns)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class BehaviorInstance:
    """One labeled span. Producers supply one per **bout**; the writer fans it out per label.

    ``label`` is empty until a human names the thing that produced it — a discovered cluster
    has an id before it has a name and may never get one, which is a column rather than a
    mode. ``source_label`` (the producer's own id: ``q01``, cluster ``7``) is always present,
    which is what keeps the naming auditable.

    ``instance_id`` derives from the identity fields when not supplied — and **not** from the
    label, so naming, renaming or nesting a branch never moves an id and the audit decisions
    recorded against it keep joining. A value read back from a file is kept as-is, so decisions
    recorded against an artifact written under an older id scheme still join too.
    """

    recording_id: str
    subject: int | None
    t_ns: int
    end_t_ns: int
    start_frame: int
    end_frame: int
    source_label: str
    label: str = ""
    #: The named branches this span sits under, outermost first, joined by ``PATH_SEPARATOR`` —
    #: ``"not moving/resting"``. ``label`` is the **last** element, so every reader that knows
    #: only about ``label`` is unaffected and a roll-up to a coarser behavior is a prefix match.
    #:
    #: Empty where the producer has no hierarchy (a query result, a ground-truth import), which is
    #: not a missing value: a flat vocabulary's path *is* its label, and `_expand` fills it in.
    label_path: str = ""
    score: float | None = None
    instance_id: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id:
            self.instance_id = instance_id_for(
                self.recording_id,
                self.subject,
                self.source_label,
                self.t_ns,
                self.end_t_ns,
            )

    @property
    def unit(self) -> tuple[str, int | None]:
        """The ``(recording_id, subject)`` this instance must be covered by."""
        return (self.recording_id, self.subject)


@dataclass
class CoverageRow:
    """One analyzed ``(recording_id, subject)`` and the time it actually observed.

    ``observed_ns`` is the denominator every rate ships with: video duration is not recording
    duration is not tracked time, and comparing a 20-minute recording with a 60-minute one on
    raw counts is meaningless — while comparing them on a rate is *still* wrong if one lost
    30% of its frames to tracking.
    """

    recording_id: str
    subject: int | None
    start_t_ns: int
    end_t_ns: int
    observed_ns: int
    n_observed: int

    @property
    def unit(self) -> tuple[str, int | None]:
        return (self.recording_id, self.subject)


@dataclass
class BehaviorOutput:
    """A read artifact: its manifest, its instance rows, its coverage rows, and what was
    measured about each instance.

    ``features`` maps ``instance_id`` to ``(cx_norm, cy_norm)``. It lives on the artifact rather
    than behind a separate accessor because it *is* part of the artifact: every surface that reads
    instances can then ask where they happened without knowing there is a second file.

    **``None`` and ``{}`` are different answers, and a surface must be able to act on the
    difference.** ``None`` means the file was never written — the run predates §14's
    ``kinematics/`` and ``recording backfill-kinematics`` fills it in. ``{}`` means features were
    computed and there was nothing to place. Only the first is actionable, and collapsing the two
    is how a user ends up adjusting a setting that was never the problem: the same class as
    ``human_label=None`` meaning both "judged unclear" and "never looked at"
    (``tests/storage/test_behavior_audit.py``).
    """

    manifest: dict
    instances: list[BehaviorInstance]
    coverage: list[CoverageRow]
    features: dict | None = None
    #: The instances the user's own clips overlap (``instance_features.csv``'s ``is_example``),
    #: or ``None`` where that was never recorded — a producer given no clips, or a run written
    #: before the column existed.
    example_ids: set | None = None

    @property
    def bout_count(self) -> int:
        """Distinct **bouts** — not rows. What a duration sum must be built on, since a
        multi-label cluster contributes one row per label over the same span."""
        return len({row.instance_id for row in self.instances})

    @property
    def audited(self) -> bool:
        return bool(self.manifest.get("audited"))


def bouts(instances: list[BehaviorInstance]) -> list[BehaviorInstance]:
    """One entry per underlying **bout**, deduplicated by ``instance_id``, first row winning.

    ``instances.csv`` is one row per ``(bout, label)``, so a cluster the user named twice
    appears twice over one span. Anything that counts, samples, or measures has to collapse
    those rows first or it double counts — the failure ``instance_id`` exists to prevent. It
    lives here, shared, rather than in each consumer, because the consumers that got it wrong
    were the ones that reimplemented it.

    Order is the input's, first appearance winning: ``_expand`` promises the producer's order
    survives a write, so collapsing must not reorder. A caller wanting time order sorts the
    result — which is why dedup and sorting are separate here.
    """
    seen: dict[str, BehaviorInstance] = {}
    for row in instances:
        seen.setdefault(row.instance_id, row)
    return list(seen.values())


def period_by_recording(coverage: list[CoverageRow]) -> dict[str, int]:
    """Each analyzed recording's row period, from its coverage row: observed time over rows.

    The fallback ``analysis/measures.py::_period_ns`` uses when no frame view is read, and the one
    every time sum here uses, so the table, the coverage line and a measure agree on what an
    instance lasts.
    """
    out: dict[str, int] = {}
    for row in coverage:
        out.setdefault(
            row.recording_id,
            max(int(row.observed_ns) // max(int(row.n_observed), 1), 1),
        )
    return out


def instance_ns(row: BehaviorInstance, periods: dict[str, int]) -> int:
    """How long one instance lasts: its inclusive span plus the one period its last row covers.

    ``end_t_ns`` is the last row's timestamp, so without the period a one-row instance lasts zero
    — the rule ``measures.measured_span`` states for the same reason.
    """
    return max(int(row.end_t_ns) - int(row.t_ns), 0) + int(
        periods.get(row.recording_id, 0)
    )


def contains_span(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    """Whether the inclusive span ``inner`` lies wholly inside the inclusive span ``outer``."""
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two inclusive spans share any time at all."""
    return a[0] <= b[1] and b[0] <= a[1]


def behavior_output_artifacts() -> list[dict]:
    """The ``{path, kind, label}`` entries an ``AnalysisModule.output_manifest`` declares for
    this artifact.

    Derived here rather than hand-listed in each producing module, for the reason
    ``runners.job_params`` is derived: a hand-written list is complete until the day the
    artifact grows a file, and nothing fails when it stops being. It grew one — the frame view
    (§19) — which is the case this existed for.
    """
    return [
        {
            "path": INSTANCES_FILENAME,
            "kind": "instances",
            "label": "Behavior instances",
        },
        {"path": COVERAGE_FILENAME, "kind": "coverage", "label": "Coverage"},
        {
            "path": MANIFEST_FILENAME,
            "kind": "behavior_output",
            "label": "Behavior output",
        },
        {"path": FRAMES_DIRNAME, "kind": "frames", "label": "Per-frame view"},
        {
            "path": FEATURES_FILENAME,
            "kind": "instance_features",
            "label": "Instance features",
        },
    ]


def _names_map(manifest: dict, key: str = "names") -> dict[str, list[str]]:
    """``{source_label: [label, ...]}`` with keys and values coerced to ``str``.

    A cluster's ``source_label`` is an integer id in the producer and a string in the CSV, and
    YAML round-trips it as whichever it was written as. Coercing here means the lookup cannot
    miss on ``7`` vs ``"7"``. A scalar value is accepted as a one-element list.

    ``key`` selects ``names`` or the parallel ``label_paths``, which has the same shape and is
    read the same way — one coercion, not two that drift.
    """
    raw = manifest.get(key) or {}
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        labels = [value] if isinstance(value, str) else list(value or [])
        out[str(key)] = [str(label) for label in labels]
    return out


def _resolved_manifest(run_dir: Path, manifest: dict) -> dict:
    """The manifest as it will be stored: normalized, with the derived fields filled in.

    ``audited`` is **derived** from whether a review sits beside the artifact — ``audit.csv``
    (the one-by-one audit) or ``clip_review.csv`` (the review of every clip, §19/§20/§22) — never
    accepted from the producer, the same discipline as the derived ``params`` rule. A stored
    copy of a fact that is checkable from the store is a second writer on one fact. Reading
    ``audit.csv`` alone reported a run reviewed clip by clip as never audited.
    """
    resolved = {k: v for k, v in manifest.items() if k != "schema_version"}
    space = resolved.get("label_space")
    if isinstance(space, str):
        resolved["label_space"] = _LABEL_SPACE_SYNONYMS.get(space, space)
    resolved["audited"] = (
        run_dir / AUDIT_FILENAME
    ).is_file() or clip_review.review_path(run_dir).is_file()
    return resolved


def _validate(
    instances: list[BehaviorInstance],
    coverage: list[CoverageRow],
    manifest: dict,
) -> None:
    """Refuse anything that would store a well-formed artifact making a false claim.

    Everything is checked before a byte is written (Tenet 10). Each check exists because the
    failure it prevents is silent: the file stays parseable and the number it produces stays
    plausible.
    """
    producer = manifest.get("producer") or {}
    missing = [k for k in ("module_id", "run_id") if not producer.get(k)]
    if missing:
        raise BehaviorOutputError(
            f"behavior output manifest needs producer.{'/producer.'.join(missing)} — an "
            "artifact that cannot name the run that produced it is not reproducible from "
            "its own metadata (Tenet 9)"
        )

    origin = manifest.get("origin")
    if origin not in ORIGINS:
        raise BehaviorOutputError(
            f"behavior output origin must be one of {sorted(ORIGINS)}, got {origin!r}. "
            "It is what keeps a prediction from being read as ground truth."
        )

    space = manifest.get("label_space")
    if space not in LABEL_SPACES:
        raise BehaviorOutputError(
            f"behavior output label_space must be one of {sorted(LABEL_SPACES)}, got "
            f"{space!r}. A comparison that pools a per-recording label space across animals "
            "renders as a tidy chart and means nothing."
        )

    overlap = manifest.get("overlap")
    if overlap is not None and overlap not in OVERLAP_MODES:
        raise BehaviorOutputError(
            f"behavior output overlap must be one of {sorted(OVERLAP_MODES)}, got {overlap!r}"
        )

    for label, kind in (manifest.get("behavior_types") or {}).items():
        if kind not in BEHAVIOR_TYPES:
            raise BehaviorOutputError(
                f"behavior_types[{label!r}] must be one of {sorted(BEHAVIOR_TYPES)}, got "
                f"{kind!r}"
            )

    # Keyed by unit *carrying the row*, because the instance checks below need the covered
    # span and not merely the fact that the unit was analyzed.
    seen_units: dict[tuple[str, int | None], CoverageRow] = {}
    for row in coverage:
        if row.unit in seen_units:
            raise BehaviorOutputError(
                f"coverage.csv has more than one row for {row.unit} — it is one row per "
                "analyzed (recording_id, subject)"
            )
        seen_units[row.unit] = row
        if row.end_t_ns < row.start_t_ns:
            raise BehaviorOutputError(
                f"coverage row {row.unit} ends before it starts "
                f"({row.start_t_ns} > {row.end_t_ns})"
            )
        if row.observed_ns < 0 or row.n_observed < 0:
            raise BehaviorOutputError(
                f"coverage row {row.unit} has a negative observed_ns/n_observed"
            )
        covered_ns = row.end_t_ns - row.start_t_ns
        if row.observed_ns > covered_ns:
            raise BehaviorOutputError(
                f"coverage row {row.unit} observed {row.observed_ns} ns of a span only "
                f"{covered_ns} ns long. observed_ns is the time actually seen *inside* "
                "[start_t_ns, end_t_ns], and it is the denominator every rate ships with — "
                "one larger than its own span silently deflates every rate built on it."
            )

    vocabulary = set(manifest.get("vocabulary") or [])
    names = _names_map(manifest)
    declared = {label for labels in names.values() for label in labels}
    declared |= {row.label for row in instances if row.label}
    unknown = sorted(declared - vocabulary)
    if unknown:
        raise BehaviorOutputError(
            f"labels {unknown} are not in the manifest vocabulary {sorted(vocabulary)} — a "
            "label outside the vocabulary cannot be read as a real zero where it is absent"
        )

    scored = [row for row in instances if row.score is not None]
    if scored:
        direction = (manifest.get("score") or {}).get("direction")
        if direction not in SCORE_DIRECTIONS:
            raise BehaviorOutputError(
                "a behavior output carrying scores must declare score.direction as one of "
                f"{sorted(SCORE_DIRECTIONS)}, got {direction!r}. Inverting it silently "
                "reverses every ranking built on this artifact, with no error."
            )

    for row in instances:
        if row.end_t_ns < row.t_ns:
            raise BehaviorOutputError(
                f"instance {row.instance_id} ends before it starts "
                f"({row.t_ns} > {row.end_t_ns}); both ends are inclusive"
            )
        if row.unit not in seen_units:
            raise BehaviorOutputError(
                f"instance {row.instance_id} is in {row.unit}, which has no coverage.csv "
                "row. Coverage is what separates a real zero from a recording that was "
                "never analyzed, so an uncovered instance makes it lie."
            )
        covered = seen_units[row.unit]
        if not contains_span(
            (covered.start_t_ns, covered.end_t_ns), (row.t_ns, row.end_t_ns)
        ):
            raise BehaviorOutputError(
                f"instance {row.instance_id} spans [{row.t_ns}, {row.end_t_ns}] but "
                f"{row.unit} was only analyzed over [{covered.start_t_ns}, "
                f"{covered.end_t_ns}]. Being *in* a covered unit is not the same as being "
                "inside the covered span, and every rate over this unit divides by a "
                "denominator that does not contain its own numerator."
            )

    if overlap == "exclusive":
        # The manifest's own claim, checked against the rows that make it. §23 reads
        # `exclusive` as "at most one label is active at a time" and refuses `transitions`
        # without it, so a stored `exclusive` that the rows contradict does not fail — it
        # produces a transition table built on an undefined "next label".
        by_unit: dict[tuple[str, int | None], list[BehaviorInstance]] = {}
        for row in bouts(instances):
            by_unit.setdefault(row.unit, []).append(row)
        for unit, rows in by_unit.items():
            ordered = sorted(rows, key=lambda r: (r.t_ns, r.end_t_ns))
            for earlier, later in zip(ordered, ordered[1:]):
                if spans_overlap(
                    (earlier.t_ns, earlier.end_t_ns), (later.t_ns, later.end_t_ns)
                ):
                    raise BehaviorOutputError(
                        f"manifest declares overlap: exclusive, but in {unit} instance "
                        f"{earlier.instance_id} [{earlier.t_ns}, {earlier.end_t_ns}] "
                        f"overlaps {later.instance_id} [{later.t_ns}, {later.end_t_ns}]. "
                        "Declare overlap: allowed, or separate the spans."
                    )


def _expand(
    instances: list[BehaviorInstance],
    names: dict[str, list[str]],
    paths: dict[str, list[str]] | None = None,
) -> list[BehaviorInstance]:
    """One row per ``(bout, label)``, sharing the bout's ``instance_id``.

    A cluster the user named twice — "rearing" and "near wall" — emits two rows over the same
    span rather than one row with a multi-valued cell, so the schema stays flat and a measure
    over "rearing" picks it up with no special case. Producer order is preserved: this artifact
    is rewritten on every authored change, and reordering it would churn the diff for nothing.

    A bout whose ``source_label`` has no entry in ``names`` passes through unchanged, keeping
    whatever ``label`` the producer set — which is how a ground-truth artifact, whose labels
    are known and whose naming map is empty, keeps them.

    ``paths`` carries the same shape as ``names`` — one entry per label, in the same order — and
    supplies each row's ``label_path``. A producer with no hierarchy supplies none, and the path
    **falls back to the label itself**: a flat vocabulary's path is one level deep, and leaving the
    column empty would make "this producer has no hierarchy" and "this bout is unlabelled" the same
    cell. Those are different states and one of them is a finding.

    **Idempotent**, because §21 reprojects this artifact on every authored change and a
    reprojection is a read of the expanded rows followed by a write. Fanning an already-fanned
    row out again gives ``labels x labels`` rows for one bout, and nothing downstream notices:
    ``bout_count`` counts distinct ``instance_id`` and still reports one.
    """
    paths = paths or {}
    out: list[BehaviorInstance] = []
    emitted: set[tuple[str, str]] = set()
    for row in instances:
        labels = names.get(row.source_label)
        here = paths.get(row.source_label) or []
        if labels:
            expanded_rows = [
                replace(
                    row,
                    label=label,
                    label_path=(here[i] if i < len(here) else "") or label,
                )
                for i, label in enumerate(labels)
            ]
        else:
            expanded_rows = [replace(row, label_path=row.label_path or row.label)]
        for expanded in expanded_rows:
            key = (expanded.instance_id, expanded.label)
            if key in emitted:
                continue
            emitted.add(key)
            out.append(expanded)
    return out


def _write_csv(path: Path, columns: list[str], rows: list[list]) -> None:
    """Write one CSV atomically. ``csv.writer``, never manual joins — a behavior label may
    contain a comma (§3)."""
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    atomic.replace(tmp, path)


def write_instance_features(
    run_dir: Path | str, positions: dict, *, example_ids: set | None = None
) -> Path:
    """Write ``instance_features.csv`` — one row per instance, keyed by ``instance_id``.

    ``positions`` is ``{instance_id: (cx_norm, cy_norm)}``. A NaN pair is written as **empty
    cells**, not as the string ``nan``: an empty cell is what every other nullable column in this
    artifact uses for "not known", and it reads back as ``None`` rather than as a float that
    compares unequal to itself in whatever consumes it.

    ``example_ids`` adds the ``is_example`` column (``true`` / ``false``): the instances one of
    the user's own example or non-example clips overlaps, which the audit never draws or counts
    (§19). ``None`` — a producer given no clips — writes no such column, which reads back as
    "not known" rather than as "none of them".

    Rows are sorted by ``instance_id`` so a reprojection that changes nothing rewrites a
    byte-identical file (§18.5's re-export rule, one artifact over).
    """
    run_dir = Path(run_dir)
    rows = []
    for instance_id in sorted(positions):
        cx, cy = positions[instance_id]
        finite = cx == cx and cy == cy  # NaN is the only value unequal to itself
        row = [
            instance_id,
            _cell(cx if finite else None),
            _cell(cy if finite else None),
        ]
        if example_ids is not None:
            row.append("true" if instance_id in example_ids else "false")
        rows.append(row)
    path = run_dir / FEATURES_FILENAME
    columns = FEATURE_COLUMNS + ([EXAMPLE_COLUMN] if example_ids is not None else [])
    _write_csv(path, columns, rows)
    return path


def read_example_ids(run_dir: Path | str) -> set[str] | None:
    """The ``instance_id``s marked ``is_example``, or ``None`` when no file or no such column.

    ``None`` and an empty set differ the way :func:`read_instance_features`' two answers do: no
    column means nobody recorded which instances the user's clips overlap; an empty set means
    someone did and none do.
    """
    path = Path(run_dir) / FEATURES_FILENAME
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if EXAMPLE_COLUMN not in (reader.fieldnames or []):
            return None
        return {
            row["instance_id"]
            for row in reader
            if (row.get(EXAMPLE_COLUMN) or "").strip().lower() == "true"
        }


def read_instance_features(run_dir: Path | str) -> dict | None:
    """``{instance_id: (cx_norm, cy_norm)}``, or ``None`` when the file was never written.

    **Absent is a state, not an error** — and not the same state as empty. An artifact written
    before features existed has no file at all, and a caller renders that ("this run predates
    position capture", fixed by ``recording backfill-kinematics``) rather than failing. A file
    that exists with no rows means the producer looked and found nothing to place, which is not
    actionable and needs a different sentence.

    Returning ``{}`` for both is what this signature used to do, and it made the two
    indistinguishable at every reader — the recorded class in
    ``tests/storage/test_behavior_audit.py``: two states must not share one encoded value.
    """
    path = Path(run_dir) / FEATURES_FILENAME
    if not path.is_file():
        return None
    out: dict[str, tuple[float | None, float | None]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["instance_id"]] = (
                _opt_float(row.get("cx_norm")),
                _opt_float(row.get("cy_norm")),
            )
    return out


def _cell(value) -> str:
    """A nullable value as a CSV cell — ``None`` is an empty cell, never a fabricated 0."""
    return "" if value is None else str(value)


def _instance_values(row: BehaviorInstance) -> list:
    """One CSV row, **derived from ``INSTANCE_COLUMNS``** rather than hand-listed beside it.

    The hand-written version was complete on the day it was written and silently incomplete the
    moment a field was added: the header would carry ``label_path`` and every row would be one
    cell short, which `csv.writer` writes without complaint. Same class as a hand-written
    ``record["params"]`` — the list and the thing it describes have to be one object.
    """
    return [_cell(getattr(row, column)) for column in INSTANCE_COLUMNS]


def _coverage_values(row: CoverageRow) -> list:
    return [
        row.recording_id,
        _cell(row.subject),
        row.start_t_ns,
        row.end_t_ns,
        row.observed_ns,
        row.n_observed,
    ]


def write_behavior_output(
    run_dir: Path | str,
    *,
    instances: list[BehaviorInstance],
    coverage: list[CoverageRow],
    manifest: dict,
) -> Path:
    """Write the three files into ``run_dir``; return it.

    ``instances`` is one entry per **bout**; the writer fans it out per label from the
    manifest's ``names`` map, and takes each row's ``label_path`` from the parallel
    ``label_paths``. Everything is validated before anything is written.

    ``behavior_output.yaml`` is written **last**, because discovery scans for it: until it
    lands the run is simply not a behavior output, so a reader never sees half an artifact.
    Each file is a tmp + ``os.replace``, so a concurrent reader never sees a torn one.

    Raises:
        BehaviorOutputError: the manifest is incomplete or a row contradicts it.
    """
    run_dir = Path(run_dir)
    resolved = _resolved_manifest(run_dir, manifest)
    _validate(instances, coverage, resolved)
    rows = _expand(
        instances, _names_map(resolved), _names_map(resolved, key="label_paths")
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / INSTANCES_FILENAME,
        INSTANCE_COLUMNS,
        [_instance_values(row) for row in rows],
    )
    _write_csv(
        run_dir / COVERAGE_FILENAME,
        COVERAGE_COLUMNS,
        [_coverage_values(row) for row in coverage],
    )

    ordered = {"schema_version": SCHEMA_VERSION}
    ordered.update(resolved)
    tmp = run_dir / (MANIFEST_FILENAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(ordered, f, sort_keys=False)
    atomic.replace(tmp, run_dir / MANIFEST_FILENAME)
    return run_dir


def _opt_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _int_or(value: str | None, default: int) -> int:
    """An int cell with a fallback for an absent column.

    Not ``_opt_int(...) or default``: frame 0 and an ``n_observed`` of 0 are both real values,
    and ``0 or -1`` is ``-1``.
    """
    parsed = _opt_int(value)
    return default if parsed is None else parsed


def _opt_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_rows(path: Path, required: set[str]) -> list[dict]:
    """Read a CSV into dicts, refusing a file that is not this table at all.

    Missing *optional* columns are tolerated (they fill a default); a missing required column
    means the file is something else, and saying so beats returning plausible rows.

    Raises:
        ValueError: a required column is absent.
    """
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"behavior output CSV {path} missing columns {sorted(missing)} "
                "(not a behavior output, or regenerate it with the current tool)"
            )
        return list(reader)


def read_behavior_output(run_dir: Path | str) -> BehaviorOutput:
    """Read the artifact at ``run_dir``.

    Tolerant by design: a file missing an optional column reads with that column's default,
    so an artifact written by an older schema still loads. ``label_space`` is normalized
    through the ``global`` → ``pooled`` synonym on the way out.

    Raises:
        FileNotFoundError: there is no ``behavior_output.yaml`` here.
        ValueError: a CSV is missing a required column.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_FILENAME} in {run_dir}")
    manifest = yaml.safe_load(atomic.read_text(manifest_path)) or {}
    space = manifest.get("label_space")
    if isinstance(space, str):
        manifest["label_space"] = _LABEL_SPACE_SYNONYMS.get(space, space)

    instances = [
        BehaviorInstance(
            recording_id=row["recording_id"],
            subject=_opt_int(row.get("subject")),
            t_ns=int(row["t_ns"]),
            end_t_ns=int(row["end_t_ns"]),
            start_frame=_int_or(row.get("start_frame"), _MISSING_FRAME),
            end_frame=_int_or(row.get("end_frame"), _MISSING_FRAME),
            source_label=row["source_label"],
            label=row.get("label") or "",
            label_path=row.get("label_path") or "",
            score=_opt_float(row.get("score")),
            instance_id=row.get("instance_id") or "",
        )
        for row in _read_rows(run_dir / INSTANCES_FILENAME, _INSTANCE_REQUIRED)
    ]
    coverage = [
        CoverageRow(
            recording_id=row["recording_id"],
            subject=_opt_int(row.get("subject")),
            start_t_ns=int(row["start_t_ns"]),
            end_t_ns=int(row["end_t_ns"]),
            observed_ns=int(row["observed_ns"]),
            n_observed=_int_or(row.get("n_observed"), 0),
        )
        for row in _read_rows(run_dir / COVERAGE_FILENAME, _COVERAGE_REQUIRED)
    ]
    return BehaviorOutput(
        manifest=manifest,
        instances=instances,
        coverage=coverage,
        features=read_instance_features(run_dir),
        example_ids=read_example_ids(run_dir),
    )


STREAM_COLUMNS = ["start_frame", "end_frame", "label", "subject", "origin"]


def write_stream_csv(
    path: Path | str,
    output: BehaviorOutput,
    recording_id: str,
    *,
    instances: list | None = None,
    keep_unnamed: bool = False,
) -> Path:
    """Project one recording's instances into a CSV the ``behavior_csv`` importer reads.

    This is the deliberate, user-initiated re-ingest §19 describes: the artifact is *not*
    auto-written into a recording's ``streams/``, because keeping the prediction/ground-truth
    boundary structural is what makes the leakage rule enforceable rather than remembered.

    The ``origin`` column carries the artifact's own ``origin`` through, so a predicted
    artifact re-ingests as a stream that declares itself and can never be read back as ground
    truth. Rows are filtered to one recording, since a stream belongs to one recording while
    the artifact spans the run's whole selection. Unlabeled rows are dropped by default: a
    behavior stream's ``label`` is required, and an unnamed cluster has nothing to call itself.

    **The one writer of this file shape.** The catalogue ``ethogram`` export target writes
    through here too, passing ``instances`` (its scope-narrowed rows) and ``keep_unnamed``
    (an unnamed cluster is written under its producer's own id, ``source_label``, because the
    ethogram is a picture of everything that happened). A second writer of the same header
    without ``origin`` is how a method's output once re-imported as ground truth.
    """
    path = Path(path)
    origin = output.manifest.get("origin") or "predicted"
    source = output.instances if instances is None else instances
    rows = []
    for row in source:
        label = row.label or (row.source_label if keep_unnamed else "")
        if row.recording_id == recording_id and label:
            rows.append(
                [row.start_frame, row.end_frame, label, _cell(row.subject), origin]
            )
    _write_csv(path, STREAM_COLUMNS, rows)
    return path


def covered_recording_ids(run_dir: Path | str) -> list[str]:
    """The recordings the artifact at ``run_dir`` covers, in ``coverage.csv``'s order.

    Read from ``coverage.csv`` alone: a surface describing what a comparison will compare asks
    this before the run, and the instance table grows with the run.

    Raises:
        ValueError: ``coverage.csv`` is missing a required column.
    """
    rows = _read_rows(Path(run_dir) / COVERAGE_FILENAME, _COVERAGE_REQUIRED)
    return list(dict.fromkeys(row["recording_id"] for row in rows))


def has_behavior_output(run_dir: Path | str) -> bool:
    """Whether an analysis run directory carries the artifact."""
    return (Path(run_dir) / MANIFEST_FILENAME).is_file()


def discover_behavior_outputs(experiment_dir: Path | str) -> list[str]:
    """Ids of the analysis runs carrying a behavior output, in numeric id order.

    By scanning, the way runs and datasets are already discovered — no central
    mutable list. A run whose manifest is absent is simply not one: an interrupted run cannot
    honestly state coverage over recordings it did not finish, so the artifact appears at
    ``complete`` or not at all.
    """
    root = experiment_store.analysis_dir(experiment_dir)
    return sorted_by_id(
        path.name for path in dirscan.child_dirs(root) if has_behavior_output(path)
    )
