"""Bundle-manifest schema, loading + validation for bulk recording import (Section 3).

A *bundle* is a folder holding a YAML ``manifest`` plus the video/stream files it
references (paths resolve relative to the folder, or absolute). The manifest is a
list of recordings, each with optional ordered video parts, streams, and grouping
variables.

The **schema is a single source**: the pydantic models below (``Manifest`` /
``RecordingSpec`` / ``StreamSpec``) both validate a manifest *and* feed the generated
Manifest reference page (``describe_manifest`` → ``scripts/gen_docs_manifest.py``),
so the docs cannot drift from what the code accepts. Validation is two-phase and
**exhaustive** (Tenet 10): the models report every structural problem at once, then
a procedural pass reports every cross-file problem (missing files, mismatched video
parts, importer-specific inputs) that a schema can't express. Building the
executable specs lives in ``api/imports`` (this layer stays storage-only); the
source paths it references are read-only inputs, never persisted (Tenet 6).
"""

import re
import typing
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from detail.core.registries import STREAM_IMPORTERS
from detail.importers import import_stream_importers
from detail.storage import video as video_store

MANIFEST_NAMES = ("manifest.yaml", "manifest.yml")
SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


# --- schema (single source for validation + the Manifest reference) --------- #
class StreamSpec(BaseModel):
    """One aligned time-series stream to import onto the recording."""

    name: str = Field(
        description="Stream name, path-safe ([A-Za-z0-9_-]) and unique within the recording."
    )
    importer: str = Field(
        description="A registered importer id (e.g. `medpc`); see the stream inputs table below."
    )
    raw: str = Field(
        description="The primary data file, relative to the bundle folder."
    )
    aux: dict[str, str] = Field(
        default_factory=dict,
        description="Importer-specific extra inputs, keyed by input name (e.g. MED-PC's `program`).",
    )
    config: dict = Field(
        default_factory=dict,
        description="The importer's own parameters, matching its config model (e.g. a CSV's "
        "`time_column` / `time_kind` / `channels`). Omit for an importer that takes none.",
    )
    offset_ns: int = Field(
        default=0,
        description="Alignment offset applied to the stream clock, in nanoseconds.",
    )

    @field_validator("name")
    @classmethod
    def _path_safe(cls, value: str) -> str:
        if not _SAFE_NAME.match(value):
            raise ValueError("stream needs a path-safe 'name' ([A-Za-z0-9_-])")
        return value


class RecordingSpec(BaseModel):
    """One recording to create, with its optional video parts, streams, and metadata."""

    id: str = Field(
        description="Recording id: [A-Za-z0-9_-], unique in the manifest, not already in the experiment."
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Grouping variables (any keys) written to the experiment metadata.",
    )
    video: list[str] = Field(
        default_factory=list,
        description="Ordered video parts, relative paths; multiple parts must share fps + resolution.",
    )
    streams: list[StreamSpec] = Field(
        default_factory=list, description="Aligned time-series streams."
    )

    @field_validator("id")
    @classmethod
    def _path_safe(cls, value: str) -> str:
        if not _SAFE_NAME.match(value):
            raise ValueError("id must match [A-Za-z0-9_-]")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _stringify(cls, value: object) -> object:
        # Grouping variables are strings; a YAML scalar like `box: 4` is coerced so
        # a numeric value is accepted rather than rejected (matches the old leniency).
        if isinstance(value, dict):
            return {k: str(v) for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _unique_stream_names(self) -> "RecordingSpec":
        seen: set[str] = set()
        for stream in self.streams:
            if stream.name in seen:
                raise ValueError(f"duplicate stream name {stream.name!r}")
            seen.add(stream.name)
        return self


class Manifest(BaseModel):
    """A bundle manifest: the ``schema_version`` and the list of recordings to import."""

    schema_version: int = Field(
        default=SCHEMA_VERSION, description="Manifest schema version (must be 1)."
    )
    recordings: list[RecordingSpec] = Field(
        description="The recordings to create; at least one."
    )

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version {value!r} (expected {SCHEMA_VERSION})"
            )
        return value

    @model_validator(mode="after")
    def _non_empty_unique_ids(self) -> "Manifest":
        if not self.recordings:
            raise ValueError("manifest has no recordings")
        seen: set[str] = set()
        for recording in self.recordings:
            if recording.id in seen:
                raise ValueError(f"duplicate id {recording.id!r} in the manifest")
            seen.add(recording.id)
        return self


# --- loading ---------------------------------------------------------------- #
def find_manifest(bundle_dir: Path | str) -> Path | None:
    bundle = Path(bundle_dir)
    for name in MANIFEST_NAMES:
        if (bundle / name).is_file():
            return bundle / name
    return None


def load_manifest(bundle_dir: Path | str) -> dict:
    """Parse the bundle's manifest YAML into a raw mapping.

    Raises:
        FileNotFoundError: If no manifest file is present.
        ValueError: If it does not parse to a mapping with a ``recordings`` list.
    """
    path = find_manifest(bundle_dir)
    if path is None:
        raise FileNotFoundError(
            f"no manifest ({' or '.join(MANIFEST_NAMES)}) in {bundle_dir}"
        )
    with open(path, encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("recordings"), list):
        raise ValueError("manifest must be a mapping with a 'recordings' list")
    return data


def resolve(bundle_dir: Path | str, path_str: str) -> Path:
    """Resolve a manifest path: relative under the bundle, or absolute as-is."""
    return Path(bundle_dir) / path_str


# --- validation ------------------------------------------------------------- #
def validate_manifest(
    bundle_dir: Path | str, manifest: dict, existing_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Validate a loaded manifest against the bundle + the experiment's existing
    recordings. Returns ``(errors, warnings)`` — empty ``errors`` means safe to
    import. Structural problems (from the schema) are reported first and, when the
    structure is sound, every cross-file problem is reported together."""
    try:
        model = Manifest.model_validate(manifest)
    except ValidationError as exc:
        return _format_errors(exc, manifest), []
    return _validate_files(Path(bundle_dir), model, existing_ids)


def _format_errors(exc: ValidationError, manifest: dict) -> list[str]:
    """Turn a pydantic ``ValidationError`` into friendly, recording-scoped strings."""
    return [f"{_prefix(err['loc'], manifest)}{_message(err)}" for err in exc.errors()]


def _prefix(loc: tuple, manifest: dict) -> str:
    if len(loc) >= 2 and loc[0] == "recordings" and isinstance(loc[1], int):
        recordings = manifest.get("recordings") or []
        sid = recordings[loc[1]].get("id") if loc[1] < len(recordings) else None
        head = (
            f"recording {sid!r}"
            if isinstance(sid, str) and sid
            else f"recording[{loc[1]}]"
        )
        if len(loc) >= 4 and loc[2] == "streams" and isinstance(loc[3], int):
            head += f" stream[{loc[3]}]"
        return f"{head}: "
    return ""


def _message(err: dict) -> str:
    kind = err["type"]
    field = err["loc"][-1] if err["loc"] else ""
    if kind == "value_error":
        return err["msg"].removeprefix("Value error, ")
    if kind == "missing":
        return f"missing {field!r}"
    if kind in ("int_type", "int_parsing"):
        return f"{field} must be an integer"
    if kind == "list_type":
        return f"{field} must be a list"
    if kind == "dict_type":
        return f"{field} must be a mapping"
    if kind == "string_type":
        return f"{field} must be a string"
    return err["msg"]


def _validate_files(
    bundle: Path, model: Manifest, existing_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Cross-file checks a schema can't express: file existence, homogeneous video
    parts, importer-specific inputs, and collisions with existing recordings."""
    errors: list[str] = []
    warnings: list[str] = []
    import_stream_importers()  # populate STREAM_IMPORTERS
    existing = set(existing_ids)
    sources: dict[str, str] = {}  # resolved source path -> recording id (reuse warning)

    for recording in model.recordings:
        where = f"recording {recording.id!r}"
        if recording.id in existing:
            errors.append(f"{where}: a recording with this id already exists")
        _validate_video(bundle, recording, where, sources, errors, warnings)
        _validate_streams(bundle, recording, where, errors)
    return errors, warnings


def _validate_video(bundle, recording, where, sources, errors, warnings) -> None:
    probes = []
    for part in recording.video:
        full = resolve(bundle, part)
        if not full.is_file():
            errors.append(f"{where}: video not found: {part}")
            continue
        if str(full) in sources:
            warnings.append(
                f"{where}: video {part} also used by {sources[str(full)]!r}"
            )
        sources[str(full)] = recording.id
        try:
            probes.append(video_store.probe_video(full))
        except Exception as exc:  # noqa: BLE001 — report, don't crash validation
            errors.append(f"{where}: cannot open video {part}: {exc}")
    if len(probes) > 1:  # parts of one recording must share fps + resolution
        res0, fps0 = probes[0]["resolution"], round(probes[0]["fps"], 3)
        if any(
            pr["resolution"] != res0 or round(pr["fps"], 3) != fps0 for pr in probes[1:]
        ):
            errors.append(f"{where}: video parts must share fps + resolution")


def _validate_stream_config(importer_cls, stream, where, errors) -> None:
    """Check a stream's ``config`` against its importer's model, collecting the problem.

    Part of the upfront pass (Tenet 10) so a mistyped mapping in a 60-recording bundle is
    reported with every other problem, rather than failing that one recording an hour in.
    """
    if not stream.config:
        return
    model = getattr(importer_cls, "config_model", None)
    if model is None:
        errors.append(
            f"{where}: stream {stream.name!r} importer {stream.importer!r} takes no "
            f"config, but one was given ({sorted(stream.config)})"
        )
        return
    try:
        model.model_validate(stream.config)
    except ValidationError as exc:
        errors.append(f"{where}: stream {stream.name!r} invalid config: {exc}")


def _validate_streams(bundle, recording, where, errors) -> None:
    for stream in recording.streams:
        importer_cls = STREAM_IMPORTERS.get(stream.importer)
        if importer_cls is None:
            errors.append(
                f"{where}: stream {stream.name!r} unknown importer {stream.importer!r}"
            )
        if not resolve(bundle, stream.raw).is_file():
            errors.append(
                f"{where}: stream {stream.name!r} raw not found: {stream.raw}"
            )
        if importer_cls is None:
            continue
        _validate_stream_config(importer_cls, stream, where, errors)
        for spec in importer_cls().aux_inputs:
            if spec.key not in stream.aux:
                errors.append(
                    f"{where}: stream {stream.name!r} missing aux {spec.key!r}"
                )
                continue
            path = resolve(bundle, stream.aux[spec.key])
            if not path.is_file():
                errors.append(
                    f"{where}: stream {stream.name!r} aux {spec.key!r} not found: {stream.aux[spec.key]}"
                )
            elif spec.extensions and path.suffix.lower() not in spec.extensions:
                errors.append(
                    f"{where}: stream {stream.name!r} aux {spec.key!r} must be {spec.extensions}"
                )


# --- docs (single source: the models above) --------------------------------- #
_MANIFEST_MODELS = (Manifest, RecordingSpec, StreamSpec)


def _type_label(annotation: object) -> str:
    origin = typing.get_origin(annotation)
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if origin is list:
        (inner,) = typing.get_args(annotation)
        if inner is str:
            return "list of strings"
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return f"list of {inner.__name__.removesuffix('Spec').lower()}"
        return "list"
    if origin is dict:
        return "mapping"
    return str(annotation)


def describe_manifest() -> list[dict]:
    """The manifest schema, read from the pydantic models — the single source the
    generated Manifest reference renders from. One entry per model, each with its
    fields' name / type / required / default / help."""
    sections = []
    for model in _MANIFEST_MODELS:
        fields = []
        for name, info in model.model_fields.items():
            required = info.is_required()
            default = None if required else info.get_default(call_default_factory=True)
            fields.append(
                {
                    "name": name,
                    "type": _type_label(info.annotation),
                    "required": required,
                    "default": default,
                    "help": info.description or "",
                }
            )
        sections.append(
            {
                "model": model.__name__,
                "summary": (model.__doc__ or "").strip(),
                "fields": fields,
            }
        )
    return sections
