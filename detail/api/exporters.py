"""One catalogue of what can be exported, read by every surface (#336).

`docs/export_design.md` is the spec. The rule it exists to satisfy: any analysis you show
should be exportable — and the sharp half, so should the **data required to recreate** a video
file, not just the video.

**Why a catalogue and not a widget protocol.** An earlier design proposed an `export_targets()` that a
result widget implements. Export is the one feature that has to work identically headless — the
destination is a GLM or a collaborator's machine, not a click — and a widget-declared contract
means writing every exporter twice. So the catalogue is here, Qt-free, and both the CLI and the
GUI read it. A widget contributes only **scope** (this cluster / this selection / all).

**An export is a subject, a target, and a scope.** A subject is not always a run: an experiment
(bundle, metadata), a recording (geometry), a run (everything else). Upstream stages produce runs
too, so detection / tracking / embedding need no new mechanism.

**Every target lists the data files it writes beside its artifact**, or declares that its
artifact *is* the data (a CSV export is its own input). That is the half that separates this from
a Save-PNG button, and it is enforced by `tests/api/test_export_registry.py` rather than by
convention — it is the rule most likely to decay.

The declaration is a **list of filenames**, not a sentence about them: it is what a surface shows
someone deciding whether to click, and a filename is checkable against what the export actually
wrote. A prose description of the same thing is neither.

**The scoping rule: export what DETAIL made, not what you handed it.** Video and streams arrived
from the user and they still have them. Tracks, masks, embeddings and analysis results did not
exist before we computed them, and have no other way out. This is what keeps the list short
enough to stay meaningful.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: What a target may be narrowed to. `None` scope always means the whole subject.
SCOPE_KINDS = ("cluster", "selection", "frames", "recordings")


class ExportError(ValueError):
    """The subject, target or scope handed to an export does not describe something writable."""


@dataclass(frozen=True)
class ExportSubject:
    """What an export exports *from* — an experiment, a recording, or a run.

    ``kind`` is the navigation kind the GUI already routes on (`experiment`, `recording`,
    `run`, `analysis_run`), which is what lets the shell place the control off the navigated
    node rather than off a hand-kept list.
    """

    kind: str
    experiment_dir: Path
    id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_dir", Path(self.experiment_dir))


@dataclass(frozen=True)
class ExportScope:
    """How much of a subject to write. Every field ``None``/empty means "all of it".

    ``cluster`` is a source label (a cluster id as the artifact spells it); ``instance_ids`` is
    an explicit list, which is what "export a selection" is — the same call with the rows named.

    ``frames`` is an inclusive, recording-**global** frame range. It is a scope rather than an
    option because it answers this class's own question — how much of the subject to write — and
    the two places that matter read it through :meth:`describe`: the manifest's ``scope`` line,
    and the "nothing to write" sentence. A range hidden in ``**options`` would leave both saying
    "everything" about a ten-second clip.
    """

    cluster: str | None = None
    instance_ids: tuple[str, ...] = ()
    frames: tuple[int, int] | None = None
    recording_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalised on the way in: this arrives from a GUI spin box, a CLI string and a YAML
        # record (where a tuple comes back as a list), and a scope that compares unequal to an
        # identical one is a bug that only shows up in a record round trip.
        if self.frames is not None:
            lo, hi = (int(v) for v in self.frames)
            object.__setattr__(self, "frames", (min(lo, hi), max(lo, hi)))
        # Ids, in the order given, each once: a list that names a recording twice must not
        # export it twice, and `("a", "b")` must compare equal to `["a", "b"]` read back from YAML.
        object.__setattr__(
            self,
            "recording_ids",
            tuple(dict.fromkeys(str(r) for r in self.recording_ids)),
        )

    def is_all(self) -> bool:
        return (
            self.cluster is None
            and not self.instance_ids
            and self.frames is None
            and not self.recording_ids
        )

    def describe(self) -> str:
        parts = []
        if self.instance_ids:
            parts.append(f"{len(self.instance_ids)} selected")
        elif self.cluster is not None:
            parts.append(f"cluster {self.cluster}")
        elif self.frames is not None:
            parts.append(f"frames {self.frames[0]}-{self.frames[1]}")
        if self.recording_ids:
            parts.append(
                f"recording {self.recording_ids[0]}"
                if len(self.recording_ids) == 1
                else f"{len(self.recording_ids)} recordings"
            )
        return ", ".join(parts) or "everything"

    def includes(self, recording_id: str) -> bool:
        """Whether ``recording_id`` is inside this scope — every recording when none is named."""
        return not self.recording_ids or str(recording_id) in self.recording_ids


@dataclass(frozen=True)
class Estimate:
    """What one target expects to write and how long it will take — an estimate, said as one.

    Either number may be ``None``, which means *not knowable here* and is shown as such. A
    guessed zero would read as "free", and a guessed figure on a target whose cost depends on
    something unreadable (a video that is not on this machine) would read as a measurement.
    ``basis`` is the one sentence that says how the figure was reached, shown on hover and by
    `export run --estimate`: an estimate nobody can check is a number they stop trusting the
    first time it is wrong.
    """

    bytes: int | None
    seconds: float | None
    basis: str = ""


@dataclass(frozen=True)
class ExportEstimate:
    """The estimate for a whole export: one entry per target, and the room there is for it."""

    per_target: tuple[tuple[str, Estimate], ...]
    free_bytes: int | None = None

    @property
    def complete(self) -> bool:
        """Whether every target could be estimated. A total over some of them is a floor."""
        return all(
            e.bytes is not None and e.seconds is not None for _, e in self.per_target
        )

    @property
    def bytes(self) -> int | None:
        known = [e.bytes for _, e in self.per_target if e.bytes is not None]
        return sum(known) if known else None

    @property
    def seconds(self) -> float | None:
        known = [e.seconds for _, e in self.per_target if e.seconds is not None]
        return sum(known) if known else None

    @property
    def fits(self) -> bool | None:
        """Whether the estimate fits the destination drive; ``None`` when either is unknown."""
        if self.bytes is None or self.free_bytes is None:
            return None
        return self.bytes <= self.free_bytes


@dataclass(frozen=True)
class ExportTarget:
    """One thing a subject can produce, and the data behind it.

    ``data_files`` lists the files written beside the artifact that carry the rows, coordinates
    or masks needed to rebuild it elsewhere — the filenames, so a surface can show them and a
    test can check them against what the export actually wrote. A target whose artifact *is* its
    own data (a CSV re-encoding) sets ``artifact_is_data`` instead. One or the other, never
    neither: that is the registry sweep's rule.

    ``run`` takes ``(subject, dest, scope, progress)`` and returns the paths it wrote, relative
    to ``dest``. It never creates ``dest`` or writes the sidecars; `write_export` owns those, so
    every target gets the manifest, the params sidecar and the cleanup for free.

    ``estimate`` takes ``(subject, scope, options)`` and returns an :class:`Estimate`: how much
    this target will write and how long it will take, for that scope. It must not write anything
    and should be cheap (a few file reads; a video target times a few dozen frames), because the
    Export dialog asks on every change. The registry sweep requires one on every shipped target.

    ``scopes`` lists what the target may be narrowed to (`SCOPE_KINDS`). ``"recordings"`` in it
    is a promise that ``run`` writes only the recordings in ``scope.recording_ids`` and that
    ``estimate`` prices only those; the sweep runs it on a two-recording fixture to check.
    """

    id: str
    label: str
    subject_kind: str
    artifact: str
    run: Callable[..., Sequence[Path]]
    data_files: tuple[str, ...] = ()
    artifact_is_data: bool = False
    #: Pictures this target draws. Separate from ``data_files`` because a figure is an
    #: *artifact*, not the data behind one — a target that shipped only PNGs would satisfy a
    #: combined list while being exactly the Save-PNG button this rule exists to prevent.
    figure_files: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    portable: bool = False
    #: Renders this target accepts, when it writes clips. Empty for everything else.
    renders: tuple[str, ...] = field(default_factory=tuple)
    estimate: Callable[..., Estimate] | None = None

    def included_files(self) -> tuple[str, ...]:
        """Every file this target puts in the folder besides the artifact itself.

        `write_export` adds the last two to every export, so they are listed here rather than
        by each target — a surface shows one list and it is the whole list. The subject's
        ``notes.md`` is not among them: it is the user's file, written only when they wrote one,
        and the same whichever target was picked.
        """
        return (*self.figure_files, *self.data_files, PARAMS_NAME, MANIFEST_NAME)


_TARGETS: list[ExportTarget] = []


def register(target: ExportTarget) -> ExportTarget:
    """Add a target to the catalogue. Raises on a duplicate ``(subject_kind, id)``."""
    for existing in _TARGETS:
        if existing.id == target.id and existing.subject_kind == target.subject_kind:
            raise ValueError(
                f"export target {target.id!r} is already registered for "
                f"subject kind {target.subject_kind!r}"
            )
    if bool(target.data_files) == bool(target.artifact_is_data):
        raise ValueError(
            f"export target {target.id!r} must declare exactly one of `data_files` (the files "
            f"carrying the data behind the artifact) or `artifact_is_data` (the artifact is "
            f"itself the data)"
        )
    _TARGETS.append(target)
    return target


def all_targets() -> tuple[ExportTarget, ...]:
    """Every registered target, hidden capabilities included — for the registry sweep."""
    _load()
    return tuple(_TARGETS)


def targets_for(subject: ExportSubject) -> list[ExportTarget]:
    """Every target this subject *kind* declares, in catalogue order — available or not.

    Unfiltered on purpose, which is the opposite of what this did at first. Hiding a target
    whose inputs are absent leaves a user unable to tell "this tool cannot export my streams"
    from "this recording has no streams yet", and the second is the one they can act on. The
    house rule is **grey out, don't hide** (gui_design.md §3.1): a surface shows the row and
    says why it is off. Ask :func:`unavailable_reason` for that sentence.

    Deliberately **not** filtered by `detail.surfaces` either. That toggle governs *launching,
    not reading* (§25) — "a completed run of a hidden module still renders its result" — and an
    export is a read. Gating here would make a finished run's data unreachable the day its
    module was hidden, which is the opposite of what the toggle promises.
    """
    _load()
    return [t for t in _TARGETS if t.subject_kind == subject.kind]


def unavailable_reason(target: ExportTarget, subject: ExportSubject) -> str | None:
    """Why ``target`` cannot run for ``subject`` right now, or ``None`` when it can.

    A sentence a person can act on — "this recording has no streams yet", not "unavailable".
    It is what a greyed-out row says, so it has to name the missing thing rather than the
    failure.
    """
    _load()
    from detail.api import export_targets

    return export_targets.unavailable_reason(target, subject)


#: ``{subject kind: (what it is called, the command that lists them)}`` for a subject held in a
#: store by id. A recording is checked by its directory.
_SUBJECT_LISTINGS = {
    "recording": ("recording", "detail experiment status"),
    "run": ("processing run", "detail process list"),
    "analysis_run": ("analysis run", "detail analysis list"),
    "distill": ("distill run", "detail distill list"),
}


def check_subject(subject: ExportSubject) -> None:
    """Refuse a subject whose id names nothing in its experiment.

    Without this a mistyped id listed every target as unavailable for a reason that was not the
    reason ("this recording has no video parts yet", "this run's record cannot be read"), and the
    user went looking for a problem with a recording that does not exist (#33).

    Raises:
        UserInputError: No recording or run with that id.
    """
    from detail.api import export_targets
    from detail.core.errors import UserInputError

    if subject.kind not in _SUBJECT_LISTINGS:
        return
    if subject.kind == "recording":
        where = export_targets._recording_dir(subject) / "recording.yaml"
        exists = where.is_file()
    else:
        found = export_targets._run_dir_for(subject)
        exists = found is not None and found.is_dir()
    if not exists:
        noun, listing = _SUBJECT_LISTINGS[subject.kind]
        raise UserInputError(
            f"no {noun} {subject.id!r} in this experiment; `{listing}` shows them"
        )


def available_targets(subject: ExportSubject) -> list[ExportTarget]:
    """Just the ones that can run — what decides whether the control appears at all."""
    return [t for t in targets_for(subject) if unavailable_reason(t, subject) is None]


def recordings_of(subject: ExportSubject) -> list[str]:
    """The recording ids a subject spans, in id order — what a recordings scope chooses from.

    A run's are the recordings it completed; an analysis run's are the ones its instances come
    from. An experiment, a recording and a comparison have none to choose between, which is the
    empty list: the Export dialog reads that as "no recordings control".
    """
    _load()
    from detail.api import export_targets

    return export_targets.recordings_of(subject)


def free_bytes_at(path: Path | str) -> int | None:
    """Free space on the drive ``path`` is (or would be) on, or ``None`` if it cannot be read.

    The nearest folder that exists, because the destination is usually one the export is about
    to create. A cosmetic question must not be why an export cannot start, so an unreadable
    drive is "unknown", not an error.
    """
    import shutil

    probe = Path(path).expanduser()
    for candidate in (probe, *probe.parents):
        if candidate.exists():
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                return None
    return None


def estimate(
    subject: ExportSubject,
    targets: Sequence[ExportTarget] | ExportTarget,
    *,
    scope: ExportScope | None = None,
    parent: Path | str | None = None,
    **options,
) -> ExportEstimate:
    """How big an export will be and how long it will take, before anything is written.

    Asks each target for its own :class:`Estimate` for ``scope`` and ``options`` (the same ones
    `write_export` would be given) and, when ``parent`` is given, how much room the drive it
    names has. Nothing is written or created.

    A target that has no estimator, or whose estimator fails, comes back as an estimate with no
    numbers and the reason in its ``basis``: the same rule as the rest of the catalogue, that one
    target's trouble is reported beside it and does not take the answer for the others with it.
    """
    wanted = [targets] if isinstance(targets, ExportTarget) else list(targets)
    scope = scope or ExportScope()
    # The destination rides in the options as ``destination``: the estimators that copy files
    # price them at that drive's measured speed, and nothing else reads it.
    asked = (
        dict(options) if parent is None else {**options, "destination": Path(parent)}
    )
    per_target = []
    for target in wanted:
        if target.estimate is None:
            per_target.append(
                (target.id, Estimate(None, None, "this target has no estimate"))
            )
            continue
        # The `try` holds the estimator and nothing else; the drive probe below is not part of
        # the question a target is answering.
        try:
            answer = target.estimate(subject, scope, dict(asked))
        except Exception as exc:  # noqa: BLE001 — per-target isolation boundary
            answer = Estimate(
                None, None, f"could not estimate: {type(exc).__name__}: {exc}"
            )
        per_target.append((target.id, answer))
    return ExportEstimate(
        tuple(per_target),
        free_bytes=None if parent is None else free_bytes_at(parent),
    )


def format_size(nbytes: int | float | None) -> str:
    """``3.7 GB`` — one formatter for a size, because two surfaces printing it two ways is the
    recorded class where the same fact reads differently in the dialog and the terminal.
    """
    if nbytes is None:
        return "unknown"
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"


def _duration(seconds: float | None) -> str:
    """``4m12s``, from the repo's one duration formatter (`core.progress`)."""
    from detail.core.progress import format_duration

    return "unknown" if seconds is None else format_duration(seconds)


def describe_one(estimate: Estimate) -> str:
    """One target's figures as a phrase: ``3.7 GB, 4m12s`` or why there are none."""
    if estimate.bytes is None and estimate.seconds is None:
        return f"cannot be estimated: {estimate.basis}"
    return f"{format_size(estimate.bytes)}, {_duration(estimate.seconds)}"


def describe_estimate(result: ExportEstimate) -> str:
    """The one-line summary both surfaces show: ``about 3.7 GB, about 4m12s``.

    When some target could not be estimated the totals are a floor and the line says so, rather
    than presenting the sum of what was known as the whole.
    """
    if result.bytes is None and result.seconds is None:
        return "cannot be estimated"
    prefix = "about" if result.complete else "at least"
    return f"{prefix} {format_size(result.bytes)}, {prefix} {_duration(result.seconds)}"


def target(subject_kind: str, target_id: str) -> ExportTarget:
    """One target by id, or raise naming what is available."""
    _load()
    for candidate in _TARGETS:
        if candidate.subject_kind == subject_kind and candidate.id == target_id:
            return candidate
    known = sorted(t.id for t in _TARGETS if t.subject_kind == subject_kind)
    raise ExportError(
        f"no export target {target_id!r} for a {subject_kind}; "
        f"available: {', '.join(known) or '(none)'}"
    )


_loaded = False


def _load() -> None:
    """Import the target definitions once, so registration is a side effect of use, not import.

    Keeps this module free of the heavy readers its targets need (zarr, cv2), which is what
    lets the CLI's `--help` and the GUI's menu build without paying for them.
    """
    global _loaded
    if not _loaded:
        _loaded = True
        from detail.api import export_targets  # noqa: F401  (registers on import)


# --------------------------------------------------------------------------- #
# The one writer
# --------------------------------------------------------------------------- #

MANIFEST_NAME = "export.yaml"
PARAMS_NAME = "params.csv"
#: The subject's own notes (`storage/notes.py`), copied in when it has any.
NOTES_NAME = "notes.md"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TargetOutcome:
    """How one target of an export fared.

    An export is one thing a user asked for and several targets doing it, so "did it work" has
    one answer per target and the run's own status is a roll-up of them. Kept per target because
    a run that wrote the instances and failed the clips has something useful to say, and one red
    word for the whole export says none of it.
    """

    target_id: str
    status: str  # "complete" | "failed"
    files: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "target": self.target_id,
            "status": self.status,
            "files": list(self.files),
            "error": self.error,
        }


@dataclass(frozen=True)
class ExportResult:
    """What one export wrote — one folder, however many targets went into it."""

    dest: Path
    target_ids: tuple[str, ...]
    files: tuple[str, ...]
    empty_reason: str = ""
    outcomes: tuple[TargetOutcome, ...] = ()

    @property
    def is_empty(self) -> bool:
        return bool(self.empty_reason)

    @property
    def status(self) -> str:
        """``complete`` / ``partial`` / ``failed`` — the roll-up of the per-target outcomes.

        ``partial`` is the interesting one: some targets wrote and some did not, which is a real
        state a user can act on (re-run the one that failed) and which a bare success/failure
        cannot express.
        """
        failed = [o for o in self.outcomes if o.status == "failed"]
        if not failed:
            return "complete"
        return "failed" if len(failed) == len(self.outcomes) else "partial"

    @property
    def failures(self) -> tuple[TargetOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status == "failed")


def destination_of(record: dict) -> Path | None:
    """Where an export run's files actually landed, composed from its record.

    The record holds ``{root, rel}`` plus a folder name and **never a path** (Tenet 6), so the
    absolute location is composed at display time — the way a reference-in-place video's is.
    ``None`` when the alias is no longer registered on this machine, which a surface renders as
    the folder name: still true, just not findable from here.

    One function because both surfaces ask: "where are my files" is the last question of every
    export, and two spellings of the answer is the class where the same fact prints two ways.
    """
    from detail.pipeline.export_run import EXPORT_ROOTS_KEY
    from detail.storage import roots

    try:
        parent = roots.resolve_ref(EXPORT_ROOTS_KEY, record.get("destination") or {})
    except (roots.MissingRootError, ValueError):
        return None
    return parent / str(record.get("folder", ""))


def describe_run(record: dict) -> tuple[str, str]:
    """What an export run's row says: **what** it wrote, and **where it went**.

    Every other run family names a registered module, and the shared listing read `module_id`
    and fell back to the family's own label. An export names no module, so every cell of a
    column headed *exporting* read the literal string "export" — a column that renders, sorts
    and says nothing. The catalogue knows the target's label; the listing should ask it.

    The destination is the second half because it is the question a *finished* export exists to
    answer. The files are outside the experiment, so "where are they" cannot be derived from the
    run id the way every other output can. `destination_of` returns ``None`` when the alias is
    no longer registered on this machine — the folder name is still true then, just not findable
    from here, so it is what a surface shows rather than a blank.

    Returns:
        ``(what, where)``. Unknown target ids pass through as their id rather than raising: a
        record written by a build that had a target this one does not is still a real export,
        and refusing to describe it would take the whole listing down with it.
    """
    kind = str(record.get("subject_kind", ""))
    labels = []
    for target_id in record.get("target_ids") or []:
        try:
            labels.append(target(kind, str(target_id)).label)
        except ExportError:
            labels.append(str(target_id))
    dest = destination_of(record)
    where = _readable(dest) if dest is not None else str(record.get("folder", ""))
    return ", ".join(labels), where


def _readable(path: Path) -> str:
    """A destination under the user's home written ``~/…``.

    Here rather than in each surface because both the Active-processes row and the shell's
    "export started" notice print this one fact, and a fact a person reads has exactly one
    formatter — the notice printed ``/Users/<you>/Downloads/…`` beside a table printing
    ``~/Downloads/…`` for a day, which is the recorded class in miniature. ``Path.home()`` can
    raise on a machine with no resolvable home; a cosmetic abbreviation must not be why a
    listing fails.

    ``as_posix()`` rather than ``str()`` — the Windows-backslash rule, and here it is the
    *display* argument as much as the guard's: the ``~/`` prefix is a POSIX shorthand, so half a
    string in one convention and half in the other would be worse than either. A path that falls
    outside the home directory is not abbreviated and keeps the platform's own separators, which
    is what somebody pasting it into a file manager needs.
    """
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except (ValueError, OSError, RuntimeError):
        return str(path)


def folder_name(subject: ExportSubject, when=None) -> str:
    """``cohort_temporal_clustering_an_01_export_2026-09-10_1432`` — the folder one export creates.

    Every export makes its **own** folder rather than writing loose files into one you chose.
    Three things follow, and they are why this is not a cosmetic choice:

    - two exports cannot collide, so there is no cleanup step and no manifest-driven deletion
      of files in a directory that also holds yours;
    - the folder says what it is months later, on a machine that is not yours, which is where
      these end up;
    - a failed export leaves a folder that is visibly its own, rather than a partial mess mixed
      into somebody's Desktop.

    The name carries what you would need to find it again: the experiment, what produced it, the
    run, and when.
    """
    from datetime import datetime

    from detail.core.utils import safe_path_part

    when = when or datetime.now()
    parts = [experiment_name(subject)]
    module = _module_name(subject)
    if module:
        parts.append(module)
    if subject.id:
        parts.append(subject.id)
    parts += ["export", when.strftime("%Y-%m-%d_%H%M%S")]
    return safe_path_part("_".join(p for p in parts if p), allow_dot=False)


def _unique_dir(parent: Path, name: str) -> Path:
    """``parent/name``, or ``parent/name_2`` if that exists — never an existing folder.

    Seconds are not enough on their own: two CSV exports of the same run can land inside one,
    and an export that silently wrote into the previous export's folder would mix two results
    and leave one manifest describing both. The whole reason a folder is minted per export is
    that nothing it writes can land on top of something else.
    """
    candidate = parent / name
    index = 2
    while candidate.exists():
        candidate = parent / f"{name}_{index}"
        index += 1
    return candidate


def experiment_name(subject: ExportSubject) -> str:
    """The experiment's own name, as the user gave it — what an export is named after.

    Read from ``name``: it used to read ``id``, a key no ``config.yaml`` has, so every export
    fell back to the folder, and the experiment bundle to the literal ``experiment.tar``. The
    folder is the tool's id; the name is the user's, and it is what they will look for.
    """
    from detail.storage import experiment as experiment_store

    try:
        config = experiment_store.read_config(subject.experiment_dir)
    except (FileNotFoundError, ValueError):
        return subject.experiment_dir.name
    return str(config.get("name") or subject.experiment_dir.name)


def _module_name(subject: ExportSubject) -> str:
    """What produced the subject — the module for a run, the kind otherwise."""
    if subject.kind == "recording":
        return "recording"
    if subject.kind == "experiment":
        return ""
    from detail.api import export_targets

    record = export_targets.run_record(subject) or {}
    return str(record.get("module_id") or subject.kind)


def write_export(
    subject: ExportSubject,
    targets: Sequence[ExportTarget] | ExportTarget,
    parent: Path | str,
    *,
    scope: ExportScope | None = None,
    progress=None,
    stop_check=None,
    folder: str = "",
    **options,
) -> ExportResult:
    """Run every target into one **new** folder under ``parent``. Returns what it wrote.

    ``parent`` is where you want the export to land; the folder inside it is named by
    :func:`folder_name`, so nothing you already have there is touched or overwritten.

    ``folder`` overrides that name. The export **run** mints it at submit so the surface can say
    where the files are going before the worker starts, and so a retry cannot quietly land
    somewhere else; a direct caller leaves it empty and gets the generated name.

    ``stop_check`` is polled between targets and handed to any target that accepts one. A
    stopped export keeps what it finished — that is the whole reason it is checked between
    targets rather than the run being killed — and reports ``partial``.

    Three things happen around the targets' own work, so no target has to remember them:

    - **`params.csv`** — the run record's own ``params``, which is already derived from the job
      via ``runners.job_params``. Never hand-listed here: a hand-written dict is complete until
      the day a field is added, and nothing fails when it stops being.
    - **`notes.md`** — the subject's notes as the user wrote them, when it has any: what a run
      was for travels with what it produced.
    - **`export.yaml`** — schema_version first, then what was written.

    An export that produces nothing still writes the manifest, saying why, rather than leaving a
    folder that looks like a failure.

    Args:
        subject: what to export from.
        targets: one target or several; all must belong to ``subject.kind``.
        parent: the directory to create the export folder in.
        scope: how much of the subject; ``None`` means all of it.
        progress: a ``(stage, fraction, detail)`` callback — decoding clips is long.
        **options: target-specific knobs (``renders``, ``crop``, ``size``, ``max_per_cluster``,
            and ``theme`` — ``"light"`` or ``"dark"`` — for every target that draws a figure).
            Each target takes what it recognises and ignores the rest.

    Raises:
        ExportError: if no target was asked for, or one does not belong to this subject kind.
    """
    wanted = [targets] if isinstance(targets, ExportTarget) else list(targets)
    if not wanted:
        raise ExportError("ask for at least one export target")
    for target in wanted:
        if target.subject_kind != subject.kind:
            raise ExportError(
                f"export target {target.id!r} is for a {target.subject_kind}, "
                f"not a {subject.kind}"
            )
        reason = unavailable_reason(target, subject)
        if reason is not None:
            raise ExportError(f"cannot export {target.label.lower()}: {reason}")
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    dest = _unique_dir(parent, folder or folder_name(subject))
    dest.mkdir()

    scope = scope or ExportScope()
    written: list[Path] = []
    outcomes: list[TargetOutcome] = []
    for index, target in enumerate(wanted):
        # Between targets, not inside one: a half-written target is worse than an unstarted one,
        # and the point of stopping gracefully is to keep what already landed.
        if stop_check is not None and stop_check():
            outcomes.append(TargetOutcome(target.id, "failed", error="cancelled"))
            continue
        # **One target's failure is not the export's.** Without this, target 3 of 8 raising means
        # targets 4-8 never run and no manifest is written at all — tolerable when you are
        # watching a traceback, and the difference between "7 of 8 done, clips failed because a
        # video moved" and an empty folder when you come back to it an hour later.
        #
        # The `try` holds `target.run` and nothing else: the sidecars, the manifest and the
        # progress emit below are bookkeeping *about* this work, and an isolation boundary may
        # not span them (CLAUDE.md).
        try:
            produced = list(
                target.run(
                    subject,
                    dest,
                    scope,
                    _slice_progress(progress, index, len(wanted)),
                    **options,
                )
            )
        except Exception as exc:  # noqa: BLE001 — per-target isolation boundary
            outcomes.append(
                TargetOutcome(target.id, "failed", error=f"{type(exc).__name__}: {exc}")
            )
            continue
        written += produced
        outcomes.append(
            TargetOutcome(
                target.id,
                "complete",
                files=tuple(sorted(_relative(dest, p) for p in produced)),
            )
        )
    relative = tuple(sorted(_relative(dest, p) for p in written))

    params = _params_for(subject)
    if params:
        _write_params(dest / PARAMS_NAME, params)
        relative = tuple(sorted((*relative, PARAMS_NAME)))

    notes = _notes_for(subject)
    if notes:
        with open(dest / NOTES_NAME, "w", encoding="utf-8", newline="") as handle:
            handle.write(notes)
        relative = tuple(sorted((*relative, NOTES_NAME)))

    # A failed target is not an empty export: the folder holds a manifest saying what failed and
    # why, which is the thing worth having. "Nothing to write" stays reserved for the case where
    # every target ran fine and had nothing to say.
    empty_reason = (
        ""
        if relative or any(o.status == "failed" for o in outcomes)
        else _empty_reason(wanted, scope)
    )
    _write_manifest(dest, subject, wanted, scope, relative, empty_reason, outcomes)
    return ExportResult(
        dest, tuple(t.id for t in wanted), relative, empty_reason, tuple(outcomes)
    )


def _slice_progress(progress, index: int, total: int):
    """Fold one target's 0..1 into its share of the whole export's bar.

    Without this a three-target export runs its bar to 100% three times, which reads as
    finished-then-not — the aggregate-fold rule (DESIGN_PLAN §6), one surface over.
    """
    if progress is None:
        return None

    def emit(stage: str, fraction: float, detail: str = "") -> None:
        progress(stage, (index + max(0.0, min(1.0, fraction))) / total, detail)

    return emit


def _empty_reason(targets: Sequence[ExportTarget], scope: ExportScope) -> str:
    names = ", ".join(t.label for t in targets)
    return f"{names} had nothing to write for {scope.describe()}."


def _relative(dest: Path, path: Path) -> str:
    """A written path as posix, relative to ``dest``.

    ``.as_posix()`` rather than ``str()`` — `str()` gives backslashes on Windows and forward
    slashes on Linux, so a value-equality test cannot catch the difference
    (`tests/storage/test_path_serialization.py`).
    """
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    return path.relative_to(dest).as_posix()


def read_manifest(dest: Path | str) -> dict | None:
    """The manifest a previous export left in ``dest``, or ``None`` if there is none."""
    import yaml

    from detail.core import atomic

    path = Path(dest) / MANIFEST_NAME
    if not path.exists():
        return None
    loaded = yaml.safe_load(atomic.read_text(path))
    return loaded if isinstance(loaded, dict) else None


def _params_for(subject: ExportSubject) -> dict:
    """The subject's run parameters, read from the record that already derived them.

    Never re-derived and never hand-listed: `runners.job_params` built ``record["params"]`` at
    run time, and re-deriving here would be a second implementation of the same dict — the
    `test_record_completeness` class, one surface over. An experiment or a recording has no
    run parameters, so it ships no sidecar.
    """
    if subject.kind not in ("run", "analysis_run"):
        return {}
    from detail.api import export_targets

    record = export_targets.run_record(subject)
    params = record.get("params") if record else None
    return dict(params) if isinstance(params, dict) else {}


def _notes_for(subject: ExportSubject) -> str:
    """The subject's notes, or ``""``.

    Every export subject kind is a notes kind by name (`tests/api/test_notes.py` holds the
    catalogue to that), so an unknown kind raises rather than quietly exporting no notes.
    """
    from detail.storage import notes as notes_store

    try:
        directory = notes_store.entry_dir(
            subject.experiment_dir, subject.kind, subject.id
        )
    except FileNotFoundError:
        return ""
    return notes_store.read_notes(directory)


def _write_params(path: Path, params: dict) -> None:
    """``key,value`` — the settings that produced the run, beside what it produced."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key in sorted(params):
            writer.writerow([key, _cell(params[key])])


def _cell(value) -> str:
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def _write_manifest(
    dest: Path,
    subject: ExportSubject,
    targets: Sequence[ExportTarget],
    scope: ExportScope,
    files: tuple[str, ...],
    empty_reason: str,
    outcomes: Sequence["TargetOutcome"] = (),
) -> None:
    """Write ``export.yaml`` — schema_version first, then what this export wrote."""
    from datetime import datetime, timezone

    import yaml

    from detail.core import atomic

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "targets": [t.id for t in targets],
        "subject_kind": subject.kind,
        "subject_id": subject.id,
        "scope": scope.describe(),
        "artifacts": [t.artifact for t in targets],
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": list(files),
    }
    # Only when something went wrong. A per-target block on every export would be noise on the
    # common one, and its absence is then readable as "all of them worked".
    if any(o.status == "failed" for o in outcomes):
        manifest["status"] = (
            "failed" if all(o.status == "failed" for o in outcomes) else "partial"
        )
        manifest["outcomes"] = [o.to_dict() for o in outcomes]
    if empty_reason:
        manifest["empty"] = empty_reason
    tmp = dest / f"{MANIFEST_NAME}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        yaml.dump(manifest, handle, sort_keys=False)
    atomic.replace(tmp, dest / MANIFEST_NAME)
