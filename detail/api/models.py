"""Experiment-independent machine-global model catalog (Section 5).

Built-in models (the ``MODELS`` / ``EMBEDDING_MODELS`` registries) and
machine-global imported weights are not experiment state — readiness, install, and
the imported catalog are all shared across experiments. This is the experiment-free
surface the GUI's top-level **Models** tab uses; ``Experiment``'s model methods
delegate here. Backend imports stay lazy (Tenet 13): importing this module pulls
in no torch / ultralytics / sam3.
"""

import inspect
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from detail.api import skeletons as skeletons_api
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core.dataclasses import InstallProgress, ModelReadiness, TextCue
from detail.core.errors import UserInputError
from detail.core.protocols import PatchFeatures
from detail.core.registries import EMBEDDING_MODELS, MODEL_REGISTRIES, MODELS
from detail.core.skeleton import Skeleton
from detail.executor.runners import import_runner_modules
from detail.models import imported as imported_store
from detail.storage import models as models_store
from detail.storage import skeleton_library
from detail.storage.user_yaml import parse_user_yaml
from detail.storage.user_yaml import require_mapping


@dataclass
class ImportedModelRef:
    """Index entry for a machine-global imported/distilled model, with the catalog
    metadata the model manager renders (§12)."""

    id: str
    name: str
    model_family: str
    classes: list[str]
    source: str = (
        "imported"  # "imported" (dropped-in .pt) | "distilled" (promoted, §12)
    )
    provenance: dict | None = None  # distilled: source experiment + dataset + metrics
    created_at: str | None = None
    made_by: str = ""
    description: str = ""
    notes: str = ""
    preview_path: Path | None = None  # absolute path to the preview image, if any
    # source projects for the "from project(s)" column — a list (legacy single-source
    # records are wrapped into one entry by ``_source_experiments``)
    source_experiments: list[dict] = field(default_factory=list)
    # What the model is (§12 *Keypoint models in the catalog*): "detection" or "keypoint",
    # from the record's task. A keypoint model also carries the skeleton it is built with and
    # the library entry that skeleton was matched or added to.
    role: str = "detection"
    task: str | None = None
    skeleton: Skeleton | None = None
    skeleton_ref: dict | None = None


@dataclass
class ToolModel:
    """A model a tool (Quick segment) can run — SAM 3 or one of your models (§27)."""

    id: str
    name: str
    prompt_capable: (
        bool  # True → the tool shows a text-prompt field; False → closed-set
    )


@dataclass
class GlobalModelRow:
    """One row of the machine-global model catalog (built-in or imported).

    ``variants`` lists every size the model offers; ``variant`` names the single one
    a row stands for and is set only by ``list_model_variants`` (the install-status
    surfaces), since install state is per variant.
    """

    id: str
    role: str  # a MODEL_REGISTRIES role: "detection" | "embedding" (| "keypoint", §26)
    architecture: str  # yolo / sam / convnext, or the imported model's family
    source: str  # "built-in" | "imported"
    variants: list[str]
    classes: list[str]  # fixed classes (imported); empty for open-vocab/embedding
    gated: bool
    input_size: int | None = (
        None  # square px an embedding model consumes; None for detection
    )
    output_dim: int | None = None  # embedding-vector length; None for detection
    provenance: dict | None = None  # distilled models: where they came from (§12)
    variant: str | None = None  # the one size this row stands for (list_model_variants)
    # What the row's weights are read for: its role, plus "patch features" when the model
    # declares PatchFeatures (§14) — one download, listed once, read for both uses.
    capabilities: list[str] = field(default_factory=list)


@dataclass
class EmbeddingModelDoc:
    """One embedding model described from code, for the Embedding models docs page."""

    model_id: str
    architecture: str
    input_size: int | None
    output_dim: int | None
    default_variant: str | None
    variant_sizes: list[
        tuple[str, int]
    ]  # (variant, embedding size); empty for single-size models
    backends: list[str]
    gated: bool
    description: str  # first line of the model class docstring


def _ensure_registered() -> None:
    """Populate the registries — they start empty until backend modules import."""
    import_runner_modules(PIPELINE_RUNNERS)


def describe_embedding_models() -> list[EmbeddingModelDoc]:
    """Per-model metadata for the Embedding models page, derived from ``EMBEDDING_MODELS``.

    Reads static class attributes only (no instantiation, no backend, no network), so a new
    embedding model self-documents. Per-variant embedding sizes come from the model's optional
    ``VARIANT_OUTPUT_DIMS`` (single-size models leave it empty).
    """
    _ensure_registered()
    docs: list[EmbeddingModelDoc] = []
    for mid in sorted(EMBEDDING_MODELS):
        cls = EMBEDDING_MODELS[mid]
        doc = (inspect.getdoc(cls) or "").split("\n", 1)[0].strip()
        docs.append(
            EmbeddingModelDoc(
                model_id=mid,
                architecture=getattr(cls, "architecture", "—"),
                input_size=getattr(cls, "input_size", None),
                output_dim=getattr(cls, "output_dim", None),
                default_variant=cls.default_variant(),
                variant_sizes=list(getattr(cls, "VARIANT_OUTPUT_DIMS", {}).items()),
                backends=list(getattr(cls, "supported_backends", [])),
                gated=bool(getattr(cls, "gated", False)),
                description=doc,
            )
        )
    return docs


def all_models() -> dict[str, tuple[str, type]]:
    """Every built-in model by id, as ``(role, class)`` — the one lookup across the
    model registries (§5 "Model registry").

    The registries stay separate because each runner dispatches on its own; this view is
    how a *surface* resolves an id without knowing the role. The hand-written chain it
    replaced (``MODELS.get(id) or EMBEDDING_MODELS.get(id)``, four copies) answered a
    duplicate id with the detection model silently; now a duplicate is refused at
    registration and the view has one answer per id.
    """
    _ensure_registered()
    return {
        model_id: (role, cls)
        for role, registry in MODEL_REGISTRIES
        for model_id, cls in registry.items()
    }


def _model(model_id: str, variant: str | None = None):
    """Construct a built-in model at ``variant`` (its default when None).

    The variant is load-bearing here, not cosmetic: a varianted model resolves a
    different weight source per size (DINOv3's sizes are four separate HF repos), so
    installing / checking without it answers for the default variant while the run
    asks for the one the user picked (§5).
    """
    models = all_models()
    if model_id not in models:
        raise ValueError(f"unknown model_id {model_id!r}; available: {sorted(models)}")
    _role, cls = models[model_id]
    return cls(variant)


def supported_backends(model_id: str) -> list[str]:
    """The backends a built-in model accepts (cuda/mps/cpu), so a default compute
    lands on a device the model can actually run. Falls back to all three for a
    model not in the built-in registries (imported/trained loaders resolve by
    family, and support the full set)."""
    entry = all_models().get(model_id)
    if entry is None:
        return ["cuda", "mps", "cpu"]
    return list(getattr(entry[1], "supported_backends", ["cuda", "mps", "cpu"]))


def list_models() -> list[GlobalModelRow]:
    """All machine-global models — built-in (every role in ``MODEL_REGISTRIES``) + imported."""
    _ensure_registered()
    rows: list[GlobalModelRow] = []
    for role, registry in MODEL_REGISTRIES:
        for mid in sorted(registry):
            cls = registry[mid]
            if getattr(cls, "shares_weights_with", None):
                # Nothing of its own to install: the backbone's row carries the capability
                # (§14), so one download is listed once. The id still resolves (all_models).
                continue
            capabilities = [role]
            if issubclass(cls, PatchFeatures):
                capabilities.append("patch features")
            rows.append(
                GlobalModelRow(
                    id=mid,
                    role=role,
                    architecture=getattr(cls, "architecture", "—"),
                    source="built-in",
                    variants=cls.variants(),
                    classes=[],  # built-ins: open-vocab / embedding — no fixed classes
                    gated=getattr(cls, "gated", False),
                    input_size=getattr(cls, "input_size", None),
                    output_dim=getattr(cls, "output_dim", None),
                    capabilities=capabilities,
                )
            )
    for rec in imported_store.list_records():
        role = _catalog_role(rec)
        rows.append(
            GlobalModelRow(
                id=rec["id"],
                role=role,
                architecture=rec.get("model_family", "—"),
                source=rec.get("source", "imported"),  # "imported" | "distilled" (§12)
                variants=[],
                classes=[c["name"] for c in rec.get("classes", [])],
                gated=False,
                provenance=rec.get("provenance"),
                capabilities=[role],
            )
        )
    return rows


def _catalog_role(record: dict) -> str:
    """A catalog record's ``MODEL_REGISTRIES`` role, from its task: a pose model is a
    ``keypoint`` model, anything else a ``detection`` one (§12)."""
    return "keypoint" if imported_store.is_pose_record(record) else "detection"


def suggested_models() -> tuple[tuple[str, str | None], ...]:
    """The foundation models a new setup is pointed at first, as ``(model_id, variant)``.

    SAM 3 segments, and the default embedding model at its default size embeds — read from
    ``EmbeddingConfig`` and the model's ``default_variant()``, never written out here, so the
    model a user is told to install is the one a run with no ``--model`` uses. (A literal
    ``("dinov3_vit", "vit_base")`` sat here while the default size was elsewhere.) The same
    ``dinov3_vit`` weights serve ``dinov3_match``, so one install covers embedding and
    keypoints. The Model manager lists these above everything else. Declared here, not in the
    GUI, so the CLI can say the same thing; ``tests/gui/test_model_manager.py`` fails if a pair
    stops naming a real row.
    """
    from detail.configs.embedding import EmbeddingConfig

    model_id = EmbeddingConfig.model_fields["model_id"].default
    _role, cls = all_models()[model_id]
    return (("sam3", None), (model_id, cls.default_variant()))


def list_model_variants() -> list[GlobalModelRow]:
    """The catalog expanded to one row per installable unit — a (model, variant) pair.

    Install state is per variant (each DINOv3 size is its own HF repo and its own
    download), so any surface that reports installed / offers Install must list at
    this granularity. A model with no size axis contributes its single row with
    ``variant=None``. ``list_models`` stays one-row-per-model for the docs tables.
    """
    rows: list[GlobalModelRow] = []
    for row in list_models():
        if not row.variants:
            rows.append(row)
            continue
        rows.extend(replace(row, variant=name) for name in row.variants)
    return rows


def gated_model_pages() -> list[tuple[str, str]]:
    """(key, Hugging Face repo page URL) — one entry per **access gate** a user must
    accept before gated weights will download.

    The key is the model's ``hf_gate`` when several models share one agreement, else the
    model id. Meta gates the whole DINOv3 collection together, so its backbones and sizes
    collapse to a single request link: sending someone to the same form twice reads as two
    outstanding steps when only one exists.
    """
    from detail.models.hf_access import gate_url

    pages: list[tuple[str, str]] = []
    seen: set[str] = set()
    for model_id, (_role, cls) in sorted(all_models().items()):
        if not getattr(cls, "gated", False):
            continue
        key = getattr(cls, "hf_gate", None) or model_id
        if key in seen:
            continue
        # Some models (DINOv3) resolve their repo per-instance from the size variant,
        # so read it off an instance rather than the class.
        repo = getattr(cls(), "hf_repo", None)
        if repo:
            seen.add(key)
            pages.append((key, gate_url(repo)))
    return pages


def prompt_capable_models() -> list[str]:
    """Built-in detection models that accept a text prompt (``TextCue``) — the ones a
    labeling run can use (``sam3``). Closed-set trained/imported models are
    excluded (empty ``supported_cue_types``, and not in ``MODELS``)."""
    _ensure_registered()
    return sorted(
        mid for mid, cls in MODELS.items() if TextCue in cls.supported_cue_types
    )


def model_readiness(model_id: str, variant: str | None = None) -> ModelReadiness:
    """Proactive readiness for a model — drives the 'check access' UI. Imported
    models are on disk by definition (no repo/token), so they are always ready;
    only built-ins consult their cache/credentials.

    ``variant`` selects the size to answer for (the model's default when omitted)."""
    _ensure_registered()
    if model_id in imported_store.discover():
        return ModelReadiness(
            ready=True, reason="ok", message="", action_url=None, repo_id=None
        )
    return _model(model_id, variant).readiness()


def install_model(
    model_id: str,
    progress_cb: Callable[[InstallProgress], None] | None = None,
    *,
    variant: str | None = None,
) -> None:
    """Ensure a built-in model's weights are present; raises ModelNotReadyError if blocked.

    ``variant`` picks the size to download. Each variant is a separate download, so
    installing one leaves the others absent — installing the default does not make a
    run configured for another size runnable."""
    _model(model_id, variant).install(progress_cb)


def is_installed(model_id: str, variant: str | None = None) -> bool:
    """Whether a model's weights are on this machine — a cheap, **no-network** check,
    and the authoritative one: a run's pre-flight gates on it, because ``load()`` will
    not download. Imported models are on disk by definition; every built-in answers
    for itself via ``DetectionModel/EmbeddingModel.is_installed`` (§5).

    Answers for ``variant`` (the model's default when omitted) — the run gates on the
    variant it was configured with, so a variant-blind answer here is a false green."""
    _ensure_registered()
    if model_id in imported_store.discover():
        return True
    return _model(model_id, variant).is_installed()


def imported_models() -> list[ImportedModelRef]:
    """The machine-global imported/distilled models (shared across all experiments) —
    'your models' in the manager, with their catalog metadata."""
    return [_imported_ref(rec) for rec in imported_store.list_records()]


def imported_model(model_id: str) -> ImportedModelRef:
    """One imported/distilled model's full metadata (FileNotFoundError if absent)."""
    if model_id not in imported_store.discover():
        raise FileNotFoundError(f"no imported model {model_id!r}")
    return _imported_ref(imported_store.read(model_id))


def _imported_ref(rec: dict) -> ImportedModelRef:
    mid = rec["id"]
    preview = rec.get("preview")
    skeleton = rec.get("skeleton")
    return ImportedModelRef(
        id=mid,
        name=rec.get("name", ""),
        model_family=rec.get("model_family", ""),
        classes=[c["name"] for c in rec.get("classes", [])],
        source=rec.get("source", "imported"),
        provenance=rec.get("provenance"),
        created_at=rec.get("created_at"),
        made_by=rec.get("made_by", ""),
        description=rec.get("description", ""),
        notes=rec.get("notes", ""),
        preview_path=(imported_store.model_dir(mid) / preview) if preview else None,
        source_experiments=_source_experiments(rec.get("provenance")),
        role=_catalog_role(rec),
        task=rec.get("task"),
        skeleton=Skeleton.from_dict(skeleton) if skeleton else None,
        skeleton_ref=rec.get("skeleton_ref"),
    )


def training_summary(record: dict) -> str | None:
    """One line on what trained a model, from its ``model.yaml`` (or a promoted model's
    ``provenance``): the backend and its version, then the settings asked for —
    ``"ultralytics 8.4.70 · epochs 100 · batch_size 16 · imgsz 640 · patience 50"``.

    The one rendering the GUI model page and ``train status`` both show. None for a record
    with neither (an imported model, or one trained before either was recorded).
    """
    trained_with = record.get("trained_with") or {}
    hyperparams = record.get("hyperparams") or {}
    parts = []
    if trained_with.get("backend"):
        version = trained_with.get("version") or "(version unknown)"
        parts.append(f"{trained_with['backend']} {version}")
    parts.extend(f"{key} {value}" for key, value in hyperparams.items())
    return " · ".join(parts) or None


def _source_experiments(provenance: dict | None) -> list[dict]:
    """Source projects for the 'from project(s)' column — always a list. Accepts the
    **legacy single-source** shape (``source_experiment_name``/``_id``) and wraps it into a
    one-element list, so models promoted before the schema change still render."""
    if not provenance:
        return []
    if provenance.get("source_experiments"):
        return list(provenance["source_experiments"])
    name = provenance.get("source_experiment_name")
    if name:
        return [{"name": name, "id": provenance.get("source_experiment_id")}]
    return []


def inspect_checkpoint(weights_path: Path | str) -> dict:
    """What importing ``weights_path`` would make, read with the same readers ``import_model``
    uses (§12 *Keypoint models in the catalog*), so a surface cannot judge a file differently
    from the import it leads to.

    Returns:
        ``{"task", "classes", "k", "flip_idx"}`` — ``k`` (points) and ``flip_idx`` are set only
        for a pose checkpoint; they are what a skeleton is checked against.

    Raises:
        FileNotFoundError: ``weights_path`` does not exist.
        UserInputError: The file is not a YOLO ``.pt`` checkpoint
            (``models/_ultralytics.read_checkpoint``, the one reader both use).
        ValueError: A checkpoint with no class names, or a pose one with no ``kpt_shape``.
    """
    from detail.core.registries import TRAINED_LOADERS
    from detail.models.yolo_pose import TrainedYoloPose

    _ensure_registered()
    path = Path(weights_path)
    classes, task = TRAINED_LOADERS[imported_store._PROBE_FAMILY].introspect(path)
    out = {"task": task, "classes": list(classes), "k": None, "flip_idx": None}
    if task == imported_store.POSE_TASK:
        out["k"], out["flip_idx"] = TrainedYoloPose.introspect(path)
    return out


def import_model(
    weights_path: Path | str,
    *,
    family: str | None = None,
    skeleton=None,
    name: str | None = None,
    made_by: str | None = None,
    description: str | None = None,
    notes: str | None = None,
    preview: Path | str | None = None,
) -> str:
    """Import an external Ultralytics checkpoint into the machine-global catalog (usable in
    any experiment). Class names and the task are read from the checkpoint; the task decides
    whether it lands as a detection or a keypoint model (§12 *Keypoint models in the
    catalog*). ``family`` is optional and, when given, is checked against the task.

    A pose checkpoint needs ``skeleton`` — a preset id, a library id, a skeleton YAML, a
    ``Skeleton`` or its dict — with the checkpoint's number of points and left/right pairs
    matching its ``flip_idx``; it is added to the skeleton library unless one with its digest
    is there. ``made_by`` / ``description`` / ``notes`` / ``preview`` are captured so the
    model isn't 'lost in a folder' (§12). Returns the assigned id.

    Raises:
        FileNotFoundError: ``weights_path`` (or ``preview``) does not exist.
        UserInputError: ``weights_path`` is not a YOLO ``.pt`` checkpoint.
        ValueError: Unknown family or one that disagrees with the task, an unknown skeleton,
            a pose checkpoint without a fitting skeleton, a skeleton for a detection
            checkpoint, a class-less checkpoint, or an id collision.
    """
    _ensure_registered()  # populate TRAINED_LOADERS + MODELS for the family loader
    resolved = None if skeleton is None else skeletons_api.resolve_skeleton(skeleton)
    skeleton_name = (
        None if skeleton is None else skeletons_api.skeleton_source_name(skeleton)
    )
    return imported_store.import_model(
        weights_path,
        family=family,
        skeleton=resolved,
        skeleton_name=skeleton_name,
        name=name,
        made_by=made_by,
        description=description,
        notes=notes,
        preview=preview,
    )


def update_model_metadata(
    model_id: str,
    *,
    made_by: str | None = None,
    description: str | None = None,
    notes: str | None = None,
) -> None:
    """Edit an imported/distilled model's user metadata in place — only the fields passed
    are changed (a ``None`` leaves that field untouched). Raises if the model is absent.
    """
    mdir = _catalog_dir_or_raise(model_id)
    rec = models_store.read_model_yaml(mdir)
    if made_by is not None:
        rec["made_by"] = made_by
    if description is not None:
        rec["description"] = description
    if notes is not None:
        rec["notes"] = notes
    models_store.write_model_yaml(mdir, rec)


def set_model_preview(model_id: str, image_path: Path | str) -> None:
    """Set or replace an imported/distilled model's preview image (raises if absent)."""
    mdir = _catalog_dir_or_raise(model_id)
    rel = imported_store.set_preview(mdir, image_path)
    rec = models_store.read_model_yaml(mdir)
    rec["preview"] = rel
    models_store.write_model_yaml(mdir, rec)


def export_model(
    model_id: str,
    dest_zip: Path | str,
    progress: Callable[[str, float, str], None] | None = None,
) -> Path:
    """Zip a model's catalog directory (weights + ``model.yaml`` + preview) to ``dest_zip``
    so a lab can copy it to another machine and restore it with ``import_model_archive`` —
    metadata + provenance + preview travel with it. Reports progress per file (a
    weights file can be tens of MB — the long-op rule). Returns the written path."""
    src = _catalog_dir_or_raise(model_id)
    dest_zip = Path(dest_zip)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    total = len(files) or 1
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, p in enumerate(files):
            if progress is not None:
                progress("exporting", i / total, p.name)
            zf.write(p, p.relative_to(src.parent).as_posix())  # keep the <id>/ prefix
    if progress is not None:
        progress("exporting", 1.0, "done")
    return dest_zip


def import_model_archive(zip_path: Path | str) -> str:
    """Restore a model exported by ``export_model`` (a zip of its ``<id>/`` catalog dir),
    bringing back its weights + metadata + preview. Returns the assigned id.

    A keypoint model's skeleton travels in its ``model.yaml`` and the model keeps that copy
    whatever this machine's library holds. The skeleton is checked, then added to the library
    unless an entry or preset with its digest exists ("name (2)" when the name is taken), and
    the record's ``skeleton_ref`` is rewritten to name that entry on this machine — read it
    back with :func:`imported_model` to say which entry was matched or added.

    Raises:
        FileNotFoundError: ``zip_path`` does not exist.
        ValueError: The archive is not a single model dir, has an unsafe path, collides
            with an existing/built-in id, has no ``model.yaml``, or holds a keypoint model
            whose skeleton is missing, invalid or does not match its recorded digest.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(f"archive not found: {zip_path}")
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise UserInputError(
            f"{zip_path.name} is not a model export (a .zip made by `detail model "
            "export`); to add a .pt checkpoint use `detail model import`"
        ) from None
    with zf:
        names = [n for n in zf.namelist() if n.strip("/") and not _is_finder_litter(n)]
        for member in names:
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"unsafe path in archive: {member}")
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1 or not any("/" in n.strip("/") for n in names):
            raise UserInputError(_one_model_folder_message(zip_path, names, tops))
        model_id = tops.pop()
        imported_store._validate_id(model_id)  # reject built-in / reserved-shape ids
        if imported_store.model_dir(model_id).exists():
            raise ValueError(
                f"a model {model_id!r} already exists; remove it before restoring"
            )
        catalog = imported_store.catalog_dir()
        catalog.mkdir(parents=True, exist_ok=True)
        # Built aside and moved into place only once it checks out, so a refused archive
        # leaves nothing in the catalog.
        staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=catalog))
        try:
            zf.extractall(staging, members=names)
            record = _restored_record(staging / model_id, zip_path)
            skeleton = (
                _restored_skeleton(record)
                if imported_store.is_pose_record(record)
                else None
            )
            dest = imported_store.model_dir(model_id)
            (staging / model_id).rename(dest)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    if skeleton is not None:
        record["skeleton_ref"] = _library_entry_for(record, skeleton)
        models_store.write_model_yaml(dest, record)
    return model_id


def _is_finder_litter(member: str) -> bool:
    """A zip entry macOS adds and nothing reads: ``__MACOSX/`` resource forks, ``.DS_Store``."""
    parts = member.strip("/").split("/")
    return parts[0] == "__MACOSX" or parts[-1] == ".DS_Store"


def _one_model_folder_message(zip_path: Path, names: list[str], tops: set[str]) -> str:
    """Why an archive is not one model folder, and what to zip instead."""
    loose = sorted(n for n in names if "/" not in n.strip("/"))
    if models_store.MODEL_FILENAME in loose:
        return (
            f"{zip_path.name} holds a model folder's contents ({models_store.MODEL_FILENAME} "
            "is at the top of the zip); zip the model folder itself, so the zip holds one "
            "folder with model.yaml inside it — or use `detail model export`"
        )
    if not tops:
        return f"{zip_path.name} is empty; it should hold one model folder"
    return (
        f"{zip_path.name} must hold exactly one model folder, and holds "
        f"{', '.join(sorted(tops))}; zip the model folder itself — or use `detail model "
        "export`"
    )


def _restored_record(model_dir: Path, zip_path: Path) -> dict:
    """The ``model.yaml`` of a model being restored, checked to be a usable model.

    It is a record (a mapping) whose ``id`` is its folder's name, with the weights it names
    beside it. Checking only that ``model.yaml`` existed, a zip without weights restored and
    was listed "installed", and one whose folder had been renamed restored under the new
    name while its record kept the old id, so two models answered to one id (#31).

    Raises:
        UserInputError: No ``model.yaml``, or one that fails a check above.
    """
    path = model_dir / models_store.MODEL_FILENAME
    if not path.is_file():
        raise UserInputError(
            f"{zip_path.name} holds no {model_dir.name}/{models_store.MODEL_FILENAME}; it "
            "is not a model export (a .zip made by `detail model export`)"
        )
    source = f"{zip_path.name}: {model_dir.name}/{models_store.MODEL_FILENAME}"
    text = path.read_bytes()
    record = require_mapping(parse_user_yaml(text, source), source, "a model record")
    if record.get("id") != model_dir.name:
        raise UserInputError(
            f"{source} is model {record.get('id')!r}, but its folder in the archive is "
            f"{model_dir.name!r}; a model is restored under its own id, so restore the zip "
            "as `detail model export` wrote it"
        )
    best = (record.get("weights") or {}).get("best")
    weights = model_dir / str(best) if best else None
    if (
        weights is None
        or ".." in Path(str(best)).parts
        or Path(str(best)).is_absolute()
        or not weights.is_file()
    ):
        raise UserInputError(
            f"{zip_path.name} holds no weights for model {model_dir.name!r} (its "
            f"model.yaml names {best!r}); export it again with `detail model export`"
        )
    return record


def _restored_skeleton(record: dict) -> Skeleton:
    """The skeleton a restored keypoint model carries, checked (raises ``ValueError``)."""
    body = record.get("skeleton")
    if not body:
        raise ValueError(
            f"keypoint model {record.get('id')!r} has no skeleton in its model.yaml; the "
            "archive cannot be restored"
        )
    skeleton = Skeleton.from_dict(body)
    skeleton.validate()
    recorded = record.get("skeleton_digest")
    if recorded and recorded != skeleton.digest():
        raise ValueError(
            f"keypoint model {record.get('id')!r}'s skeleton does not match its recorded "
            f"digest ({skeleton.digest()} vs {recorded}); the archive was edited or damaged"
        )
    return skeleton


def _library_entry_for(record: dict, skeleton: Skeleton) -> dict:
    """``{id, name, digest}`` of the library entry a restored keypoint model's skeleton
    matched or was added as, named after the entry it came from on the exporting machine.
    """
    name = (record.get("skeleton_ref") or {}).get("name") or (
        f"{record.get('name') or record.get('id')} skeleton"
    )
    return skeleton_library.ensure_in_library(skeleton, name)


def remove_imported_model(model_id: str) -> None:
    """Delete a machine-global imported model (raises if absent).

    Also deletes a catalog folder an import cut short left behind (no ``model.yaml``,
    :func:`imported.unfinished`): it is no model, but it blocks its name, and nothing else
    could delete it. Only a name in either listing is deleted, so an id is never a path.
    """
    if (
        model_id not in imported_store.discover()
        and model_id not in imported_store.unfinished()
    ):
        raise FileNotFoundError(f"no imported model {model_id!r}")
    imported_store.remove(model_id)


def tool_models() -> list[ToolModel]:
    """Models a tool (Quick segment §27) can run: prompt-capable built-ins (SAM 3) plus every
    imported/distilled *detection* model (a keypoint model is not one). Closed-set models are ``prompt_capable=False`` so the tool can
    hide its prompt field and run detection directly (also unblocks the no-GPU wow)."""
    out = [
        ToolModel(id=mid, name=mid, prompt_capable=True)
        for mid in prompt_capable_models()
    ]
    for ref in imported_models():
        if ref.role != "detection":
            continue  # a keypoint model segments nothing (§12)
        out.append(ToolModel(id=ref.id, name=ref.name or ref.id, prompt_capable=False))
    return out


def _catalog_dir_or_raise(model_id: str) -> Path:
    """The catalog dir for an imported/distilled model, or FileNotFoundError."""
    if model_id not in imported_store.discover():
        raise FileNotFoundError(f"no imported model {model_id!r}")
    return imported_store.model_dir(model_id)
