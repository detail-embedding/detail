"""Imported keypoint models, run through ONNX (§26 Phase 4).

A user's own pose model, trained elsewhere and exported to ONNX, imported into the machine-global
catalog (``models/imported.py``, role ``keypoints``, family ``onnx_keypoints``) and run with
``onnxruntime`` — never with the tool that trained it installed.

**What is supported is what an exporter actually emits.** v1 reads sleap-nn's ONNX exports
(SLEAP 1.5+). Those graphs take a uint8 image ``(B, C, H, W)`` named ``image`` and bake in the
normalization, the model's input scaling and the peak finding — not sleap-nn's sub-pixel
refinement, and not its padding to the network's stride, so a point can sit a few pixels from
where SLEAP itself would put it — and output keypoint coordinates already in input-image pixels: ``peaks (B, K, 2)`` x, y, NaN below the export's peak
threshold, and ``peak_vals (B, K)``, 0 there. The metadata (node names, edges, model type,
channels, crop size) is embedded in the file when the exporter had ``onnx`` installed, and is
always written beside it as ``export_metadata.json``. Three model types fit a stage that already
knows where the subject is:

- ``single_instance`` — the whole frame in, one point per keypoint out.
- ``centered_instance`` — a ``crop_size`` crop centred on the subject's box in. DETAIL's
  tracking stands in for SLEAP's centroid model.
- ``topdown`` — the centroid + centred-instance bundle, whole frame in, up to ``max_instances``
  animals out; the one whose centroid falls on the tracked subject's box is kept.

Bottom-up and multi-class exports need grouping or identity DETAIL does not do here, and are
refused at import with that reason, as is an ONNX file that is not a sleap-nn export.

Ported from the pre-§26 keypoints stage onto the ``InstanceFrame`` protocol: every preset places
one subject per tracked box (``_per_subject.place_subjects``), one point per keypoint, its score
the export's peak value.

``onnxruntime`` is imported only inside functions (Tenet 13): importing this module pulls in no
backend. A run asked for CUDA runs on CUDA or is refused: onnxruntime itself drops to the CPU
when it cannot start CUDA, which left a run recorded as CUDA running on the CPU.
"""

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from detail.core import atomic
from detail.core import yaml_load
from detail.core.dataclasses import FrameResult, ModelReadiness
from detail.core.installed import is_importable
from detail.core.keypoints import ScoreSpec, InstanceFrame
from detail.core.protocols import KeypointModel
from detail.core.skeleton import Skeleton
from detail.core.utils import utc_timestamp
from detail.models import imported as imported_store
from detail.models._per_subject import place_subjects
from detail.storage import models as models_store

FAMILY = "onnx_keypoints"
WEIGHTS_DIR = "weights"
FORMAT_SLEAP = "sleap_nn"
PRESETS = ("single_instance", "centered_instance", "topdown")
_SLEAP_METADATA_KEY = "sleap_nn_metadata"
_TRAINING_CONFIG_KEY = "training_config"
_SIDECAR = "export_metadata.json"
#: Outputs each preset reads, by the names sleap-nn's export gives them.
_OUTPUTS = {
    "single_instance": ("peaks", "peak_vals"),
    "centered_instance": ("peaks", "peak_vals"),
    "topdown": ("centroids", "peaks", "peak_vals", "instance_valid"),
}
_REFUSED = {
    "bottomup": "a bottom-up model finds every animal's parts and groups them afterwards, "
    "which its ONNX export leaves to Python; export a single-instance or top-down model",
    "centroid": "a centroid model places one point per animal, not a skeleton",
    "multi_class_bottomup": "multi-class models assign identities, which DETAIL's tracking "
    "already does; export the matching single-instance or top-down model",
    "multi_class_topdown": "multi-class models assign identities, which DETAIL's tracking "
    "already does; export the matching single-instance or top-down model",
}


class OnnxBackendMissingError(ImportError):
    """``onnxruntime`` is not installed (the ``keypoints_onnx`` extra)."""


class OnnxCudaUnavailableError(ValueError):
    """A run asked for CUDA and the installed onnxruntime cannot run there."""


def backend_available() -> bool:
    """Whether onnxruntime is installed — a disk check, it imports nothing."""
    return is_importable("onnxruntime")


def _ort():
    try:
        import onnxruntime
    except ImportError as exc:
        raise OnnxBackendMissingError(
            "imported keypoint models run with onnxruntime; install it with "
            "`pip install -e '.[keypoints_onnx]'`"
        ) from exc
    return onnxruntime


def _cuda_index(device: str) -> int:
    """The first index of ``resolve_device``'s CUDA string ("0", "1", "0,1")."""
    return int(device.split(",")[0])


# --------------------------------------------------------------------------- #
# Reading an export
# --------------------------------------------------------------------------- #
def _resolve_paths(path: Path) -> tuple[Path, Path | None]:
    """``(model.onnx, export_metadata.json or None)`` from a file or an export folder."""
    path = Path(path)
    if path.is_dir():
        onnx_path = path / "model.onnx"
        if not onnx_path.is_file():
            candidates = sorted(path.glob("*.onnx"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"{path} holds {len(candidates)} .onnx files; pass the one to import"
                )
            onnx_path = candidates[0]
    else:
        onnx_path = path
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    sidecar = onnx_path.parent / _SIDECAR
    return onnx_path, (sidecar if sidecar.is_file() else None)


def read_export(path: Path | str) -> dict:
    """What an export declares, checked against what the graph actually takes and returns.

    ``path`` is the ``.onnx`` file or the folder sleap-nn wrote it to. Returns
    ``{"onnx": Path, "metadata": dict, "preset": str, "input": {...}, "outputs": [...]}``.

    Raises:
        FileNotFoundError: no ONNX file at ``path``.
        ValueError: not a sleap-nn export, a model type DETAIL cannot use, or a graph whose
            input / outputs disagree with what that model type exports.
        OnnxBackendMissingError: onnxruntime is not installed.
    """
    onnx_path, sidecar = _resolve_paths(Path(path))
    ort = _ort()
    try:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # onnxruntime raises its own exception types
        raise ValueError(
            f"{onnx_path.name} is not a readable ONNX model: {exc}"
        ) from exc
    raw = session.get_modelmeta().custom_metadata_map.get(_SLEAP_METADATA_KEY)
    if raw is not None:
        metadata = json.loads(raw)
    elif sidecar is not None:
        metadata = json.loads(atomic.read_text(sidecar))
    else:
        raise ValueError(
            f"{onnx_path.name} carries no sleap-nn metadata and has no {_SIDECAR} beside "
            "it. DETAIL imports sleap-nn exports (`sleap-nn export <model> --format onnx`); "
            "pass the export folder, or keep export_metadata.json next to the .onnx"
        )
    model_type = str(metadata.get("model_type") or "")
    if model_type in _REFUSED:
        raise ValueError(f"cannot import a {model_type} model: {_REFUSED[model_type]}")
    if model_type not in PRESETS:
        raise ValueError(
            f"unknown sleap-nn model type {model_type!r}; DETAIL imports {list(PRESETS)}"
        )
    inputs = session.get_inputs()
    if len(inputs) != 1 or len(inputs[0].shape) != 4:
        raise ValueError(
            f"expected one 4-D image input, got {[(i.name, i.shape) for i in inputs]}"
        )
    image = inputs[0]
    if image.type != "tensor(uint8)":
        raise ValueError(f"expected a uint8 image input, got {image.type}")
    channels = int(metadata.get("input_channels") or 0)
    declared = image.shape[1]
    if channels not in (1, 3) or (isinstance(declared, int) and declared != channels):
        raise ValueError(
            f"the export declares {channels} input channel(s) and the graph takes "
            f"{declared}; DETAIL feeds 1 (greyscale) or 3 (RGB)"
        )
    outputs = [o.name for o in session.get_outputs()]
    missing = [name for name in _OUTPUTS[model_type] if name not in outputs]
    if missing:
        raise ValueError(
            f"a {model_type} export should output {list(_OUTPUTS[model_type])}; this "
            f"graph has {outputs} (missing {missing})"
        )
    names = [str(n) for n in metadata.get("node_names") or []]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"the export's node names are missing or repeated: {names}")
    crop = metadata.get("crop_size")
    if model_type == "centered_instance" and not crop:
        raise ValueError("a centered_instance export must declare its crop_size")
    if (
        model_type == "centered_instance"
        and float(metadata.get("input_scale") or 1) != 1
    ):
        # Measured on sleap-nn 0.3.3: the standalone centred-instance export is traced at the
        # training frame size, so its internal resize is frozen there — a 256 px crop came back
        # with peaks out to 728 px. The top-down bundle resizes its crops correctly.
        raise ValueError(
            "this centered_instance model was trained at input scale "
            f"{metadata.get('input_scale')}, and sleap-nn exports such a model with its crop "
            "resize fixed to the training frame size, so it misplaces every point on a "
            "crop. Export the top-down bundle instead (`sleap-nn export <centroid> "
            "<centered_instance> --format onnx`)"
        )
    return {
        "onnx": onnx_path,
        "metadata": metadata,
        "preset": model_type,
        "input": {
            "name": image.name,
            "channels": channels,
            "crop_size": [int(v) for v in crop] if crop else None,
            "frame_size": _frame_size(session, onnx_path, model_type),
        },
        "outputs": outputs,
    }


def _frame_size(session, onnx_path: Path, model_type: str) -> list[int] | None:
    """``[height, width]`` sleap-nn size-matches every whole frame to before its model — the
    training config's ``max_height`` / ``max_width`` — or None when it declares neither.

    That is also the size the export was traced at, and a graph from the legacy exporter
    keeps its whole-frame resize fixed at the traced size (the top-down bundle), so feeding
    frames matched to it is what keeps those graphs right, as well as what sleap-nn does.
    Read from the training config embedded in the file, else the one exported beside it; a
    bundle's whole-frame stage is its centroid model.
    """
    if model_type == "centered_instance":
        return None  # fed crops, never whole frames
    raw = session.get_modelmeta().custom_metadata_map.get(_TRAINING_CONFIG_KEY)
    config = yaml_load.safe_load(raw) if raw else None
    if config is None:
        for name in ("training_config.yaml", "training_config_centroid.yaml"):
            sidecar = onnx_path.parent / name
            if sidecar.is_file():
                config = yaml_load.safe_load(atomic.read_text(sidecar))
                break
    if (
        isinstance(config, dict)
        and "centroid" in config
        and "data_config" not in config
    ):
        config = config["centroid"]
        config = yaml_load.safe_load(config) if isinstance(config, str) else config
    try:
        pre = config["data_config"]["preprocessing"]
    except (TypeError, KeyError):
        return None
    height, width = pre.get("max_height"), pre.get("max_width")
    return [int(height), int(width)] if height and width else None


def export_skeleton(metadata: dict) -> Skeleton:
    """The export's skeleton with its left/right pairs inferred from the names (§26): a
    starting value the user can always change; nothing is ever read from a name at run time.
    """
    names = [str(n) for n in metadata["node_names"]]
    edges = [(names[int(a)], names[int(b)]) for a, b in metadata.get("edge_inds") or []]
    return Skeleton(names=tuple(names), edges=tuple(edges), pairs=_infer_pairs(names))


def _infer_pairs(names: list[str]) -> tuple[tuple[str, str], ...]:
    """``(left, right)`` for each name whose ``left`` / ``right`` token has a partner: ``left_ear``
    with ``right_ear``, ``ear_l`` with ``ear_r``, ``LeftPaw`` with ``RightPaw``."""
    present = set(names)
    swaps = (
        ("left", "right"),
        ("Left", "Right"),
        ("LEFT", "RIGHT"),
        ("_l", "_r"),
        ("l_", "r_"),
        ("_L", "_R"),
        ("L_", "R_"),
    )
    pairs = []
    for name in names:
        for left, right in swaps:
            if left not in name:
                continue
            partner = name.replace(left, right, 1)
            if partner != name and partner in present:
                pairs.append((name, partner))
                break
    return tuple(pairs)


# --------------------------------------------------------------------------- #
# The catalog
# --------------------------------------------------------------------------- #
def import_model(
    path: Path | str,
    *,
    name: str | None = None,
    skeleton: Skeleton | None = None,
    made_by: str | None = None,
    description: str | None = None,
    notes: str | None = None,
    preview: Path | str | None = None,
) -> str:
    """Copy a sleap-nn ONNX export into the catalog as an imported keypoints model.

    ``skeleton`` overrides the roles inferred from the export's names; its names must be the
    export's, in its order (they are the output columns). Returns the assigned id.

    Raises:
        FileNotFoundError / ValueError / OnnxBackendMissingError: see ``read_export``; also
            a skeleton whose names differ from the export's, or an id collision.
    """
    export = read_export(path)
    metadata = export["metadata"]
    inferred = export_skeleton(metadata)
    if skeleton is None:
        skeleton = inferred
    elif list(skeleton.names) != list(inferred.names):
        raise ValueError(
            f"the skeleton's keypoints {list(skeleton.names)} must be the export's, in "
            f"order: {list(inferred.names)}"
        )
    skeleton.validate()
    onnx_path = export["onnx"]
    model_id = imported_store._slugify(
        name or metadata.get("model_name") or onnx_path.parent.name or onnx_path.stem
    )
    imported_store._validate_id(model_id)
    mdir = imported_store.model_dir(model_id)
    if mdir.exists():
        raise ValueError(
            f"imported model {model_id!r} already exists; remove it or pass a "
            "different name"
        )
    weights = _copy_weights(onnx_path, mdir / WEIGHTS_DIR)
    preview_rel = (
        imported_store.set_preview(mdir, preview) if preview is not None else None
    )
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": name or model_id,
            "source": "imported",
            "role": "keypoints",
            "model_family": FAMILY,
            "format": FORMAT_SLEAP,
            "preset": export["preset"],
            "weights": {
                "onnx": f"{WEIGHTS_DIR}/{weights[0].name}",
                "data": [f"{WEIGHTS_DIR}/{path.name}" for path in weights[1:]],
            },
            "sha256": _sha256(weights),
            "input": export["input"],
            "peak_threshold": metadata.get("peak_threshold"),
            "max_instances": metadata.get("max_instances"),
            "anchor_part": metadata.get("anchor_part"),
            "skeleton": skeleton.to_dict(),
            # What the exporter said about itself — never its checkpoint_path, an absolute
            # path on the machine that trained it.
            "export": {
                key: metadata.get(key)
                for key in (
                    "sleap_nn_version",
                    "model_type",
                    "model_name",
                    "backbone",
                    "input_scale",
                    "output_stride",
                    "export_timestamp",
                )
            },
            "status": "complete",
            "created_at": utc_timestamp(),
            "made_by": made_by or "",
            "description": description or "",
            "notes": notes or "",
            "preview": preview_rel,
        },
    )
    return model_id


def set_skeleton(model_id: str, skeleton: Skeleton) -> None:
    """Replace an imported keypoints model's roles (its left/right pairs and edges).

    The names cannot change — they are the model's output columns. A run already made keeps
    the skeleton it recorded in its own ``keypoints.yaml``.

    Raises:
        FileNotFoundError: no such imported keypoints model.
        ValueError: different names, or a skeleton that fails validation.
    """
    record = read_record(model_id)
    current = Skeleton.from_dict(record["skeleton"])
    if list(skeleton.names) != list(current.names):
        raise ValueError(
            f"the keypoints are the model's outputs and cannot change: expected "
            f"{list(current.names)}, got {list(skeleton.names)}"
        )
    skeleton.validate()
    record["skeleton"] = skeleton.to_dict()
    models_store.write_model_yaml(imported_store.model_dir(model_id), record)


def read_record(model_id: str) -> dict:
    """An imported keypoints model's ``model.yaml``.

    Raises:
        FileNotFoundError: no imported model ``model_id``, or it is not a keypoints model.
    """
    if model_id not in imported_store.discover(role=None):
        raise FileNotFoundError(f"no imported model {model_id!r}")
    record = imported_store.read(model_id)
    if imported_store.role_of(record) != "keypoints":
        raise FileNotFoundError(
            f"imported model {model_id!r} is a {imported_store.role_of(record)} model, "
            "not a keypoints model"
        )
    return record


def is_imported(model_id: str) -> bool:
    """Whether ``model_id`` names an imported ONNX keypoints model.

    The family is checked, not just the role: a YOLO-pose student shares the catalog and the
    role (``models/yolo_pose.py``), and loading one as an ONNX graph would fail at run time.
    A record written before families were told apart here is an ONNX import.
    """
    if model_id not in imported_store.discover(role="keypoints"):
        return False
    return imported_store.read(model_id).get("model_family", FAMILY) == FAMILY


def _copy_weights(onnx_path: Path, dest: Path) -> list[Path]:
    """Copy the graph and its external weights, keeping their names, and prove the copy loads.

    The torch exporter sleap-nn falls back to for a downscaling model writes the weights
    beside the graph as ``<name>.data``. sleap-nn then re-saves the graph whole when it embeds
    its metadata, leaving that file unused — but only when ``onnx`` was installed at export;
    otherwise the graph still names it, and both must travel under their own names. Loading
    the copy is what makes a piece left behind fail the import, not a run hours later.

    Raises:
        ValueError: the copied model does not load (a weights file it names is missing).
    """
    dest.mkdir(parents=True)
    files = [onnx_path]
    data = onnx_path.with_name(onnx_path.name + ".data")
    if data.is_file():
        files.append(data)
    copied = []
    for path in files:
        shutil.copy2(path, dest / path.name)
        copied.append(dest / path.name)
    try:
        _ort().InferenceSession(str(copied[0]), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises its own exception types
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise ValueError(
            f"the copied model does not load ({exc}); if the export keeps its weights in "
            "another file beside the .onnx, keep that file next to it and import again"
        ) from exc
    return copied


def _sha256(paths: list[Path]) -> str:
    """One digest over the graph and its weights, in order — what makes a re-import under the
    same id distinguishable in a run's record."""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class ImportedKeypointModel(KeypointModel):
    """An imported keypoints model from the catalog, run with onnxruntime (§26)."""

    supported_backends = ["cpu", "cuda"]
    architecture = FAMILY
    per_detection = True  # top-down: one instance per detection's box
    n_candidates = 1
    # The export's peak threshold already left the faintest points unplaced; what is placed
    # is kept, and a cut-off is the user's to set where the points are read.
    score_spec = ScoreSpec("heatmap peak", 0.0, 1.0, None)
    default_track_method = "despike_smooth"

    def __init__(self, model_id: str) -> None:
        super().__init__(None)
        self.model_id = model_id
        self._record = read_record(model_id)
        self._skeleton = Skeleton.from_dict(self._record["skeleton"])
        self.preset = self._record["preset"]
        self._input = dict(self._record["input"])
        self._path = (
            imported_store.model_dir(model_id) / self._record["weights"]["onnx"]
        )
        self._session = None

    def skeleton(self) -> Skeleton:
        return self._skeleton

    def is_installed(self) -> bool:
        return self._path.is_file()

    def readiness(self) -> ModelReadiness:
        """Ready when the imported file is in the catalog and onnxruntime is installed —
        both local, so a run that would fail at ``load`` is refused at validate instead.
        """
        if not self.is_installed():
            return ModelReadiness(
                ready=False,
                reason="not_installed",
                message=f"imported model {self.model_id!r} has lost its weights file; "
                "remove it and import the export again",
                action_url=None,
                repo_id=None,
            )
        if not backend_available():
            return ModelReadiness(
                ready=False,
                reason="missing_dependency",
                message="imported keypoint models run with onnxruntime; install it with "
                "`pip install -e '.[keypoints_onnx]'`",
                action_url=None,
                repo_id=None,
            )
        return ModelReadiness(
            ready=True, reason="ok", message="", action_url=None, repo_id=None
        )

    def check_backend(self, backend: str) -> None:
        """Refuse CUDA when the installed onnxruntime is the CPU-only build."""
        if backend != "cuda":
            return
        ort = _ort()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise OnnxCudaUnavailableError(
                f"this run asked for CUDA, but the installed onnxruntime {ort.__version__} "
                "is the CPU-only build. Uninstall `onnxruntime` and install the "
                "`keypoints_onnx` extra again, which brings the GPU build "
                "(`onnxruntime-gpu`), or run on the CPU"
            )

    def load(self, device: str) -> None:
        """Open the ONNX session on ``device``: the CPU, or the CUDA device it names."""
        ort = _ort()
        if device == "cpu":
            self._session = ort.InferenceSession(
                str(self._path), providers=["CPUExecutionProvider"]
            )
            return
        self.check_backend("cuda")
        # onnxruntime-gpu loads cuDNN at the first convolution and finds it among the DLLs
        # torch has loaded; without this a session reports CUDA and fails on its first frame.
        import torch

        session = ort.InferenceSession(
            str(self._path),
            providers=[("CUDAExecutionProvider", {"device_id": _cuda_index(device)})],
        )
        if session.get_providers()[0] != "CUDAExecutionProvider":
            built_for = getattr(ort, "cuda_version", "")
            raise OnnxCudaUnavailableError(
                f"onnxruntime-gpu {ort.__version__} could not start CUDA. It needs the "
                f"CUDA {built_for} and cuDNN 9 libraries, which a PyTorch build for the "
                f"same CUDA major version provides; PyTorch here is {torch.__version__}. "
                "Run on the CPU, or install a matching PyTorch"
            )
        self._session = session

    def resolved(self) -> dict:
        """What ran: the imported file's content hash and how it was read (Tenet 9) — a
        re-import under the same id would otherwise be indistinguishable in the record.
        """
        return {
            "imported": {
                "format": self._record.get("format"),
                "preset": self.preset,
                "sha256": self._record.get("sha256"),
                "peak_threshold": self._record.get("peak_threshold"),
                "sleap_nn_version": (self._record.get("export") or {}).get(
                    "sleap_nn_version"
                ),
            }
        }

    def predict_batch(self, frames: list[InstanceFrame]) -> list[FrameResult]:
        if self._session is None:
            raise RuntimeError("call load() before predict_batch()")
        return place_subjects(frames, len(self._skeleton), self._place)

    def _place(self, subjects) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each ``(frame, box)``'s points by the export's own preset."""
        crops = [_Subject(frame, bbox) for frame, bbox in subjects]
        if self.preset == "centered_instance":
            return self._centered(crops)
        return self._whole_frames(crops)

    # -- presets ------------------------------------------------------------ #
    def _image(self, rgb: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` uint8 in the channels the model was trained on."""
        if self._input["channels"] == 1:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[None]
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def _run(self, batch: np.ndarray) -> dict:
        names = [o.name for o in self._session.get_outputs()]
        values = self._session.run(None, {self._input["name"]: batch})
        return dict(zip(names, values))

    def _whole_frames(self, crops) -> list[tuple[np.ndarray, np.ndarray]]:
        """single_instance and topdown: whole frames in, frame pixels out. Frames of one size
        go through in one call, size-matched first the way sleap-nn's own predictor does.
        """
        results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(crops)
        by_shape: dict[tuple, list[int]] = {}
        for i, crop in enumerate(crops):
            by_shape.setdefault(crop.frame.shape[:2], []).append(i)
        for (h, w), indices in by_shape.items():
            matched = [self._size_match(crops[i].frame) for i in indices]
            ratio = matched[0][1]
            batch = np.stack([self._image(image) for image, _ratio in matched])
            out = self._run(batch)
            for row, i in enumerate(indices):
                if self.preset == "topdown":
                    peaks, vals = self._pick_instance(
                        out, row, crops[i].bbox, (w, h), ratio
                    )
                else:
                    peaks, vals = out["peaks"][row], out["peak_vals"][row]
                results[i] = _result(np.asarray(peaks) / ratio, vals, (w, h))
        return results

    def _size_match(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """``(image, ratio)``: ``rgb`` scaled by ``ratio`` to fit the training frame size and
        zero-padded at the bottom and right to it — sleap-nn's ``apply_sizematcher``. A frame
        already that size, or a model that declares none, passes through at ratio 1."""
        size = self._input.get("frame_size")
        h, w = rgb.shape[:2]
        if not size or (h, w) == tuple(size):
            return rgb, 1.0
        max_h, max_w = size
        ratio = min(max_h / h, max_w / w)
        target_h, target_w = int(round(h * ratio)), int(round(w * ratio))
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(rgb, (target_w, target_h), interpolation=interp)
        padded = np.zeros((max_h, max_w, rgb.shape[2]), np.uint8)
        padded[:target_h, :target_w] = scaled
        return padded, ratio

    def _pick_instance(self, out: dict, row: int, bbox, frame_wh, ratio: float = 1.0):
        """The top-down instance on the tracked subject: valid, centroid inside its box,
        nearest the box centre. None found leaves every point unplaced. ``ratio`` is the
        size-match's, so the centroids come back to frame pixels before the box is asked.
        """
        w, h = frame_wh
        x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        centroids = np.asarray(out["centroids"][row]) / ratio
        valid = np.asarray(out["instance_valid"][row], dtype=bool)
        inside = (
            valid
            & (centroids[:, 0] >= x1)
            & (centroids[:, 0] <= x2)
            & (centroids[:, 1] >= y1)
            & (centroids[:, 1] <= y2)
        )
        k = len(self._skeleton)
        if not inside.any():
            return np.full((k, 2), np.nan, np.float32), np.zeros(k, np.float32)
        candidates = np.flatnonzero(inside)
        best = candidates[
            np.argmin(np.linalg.norm(centroids[candidates] - centre, axis=1))
        ]
        return out["peaks"][row][best], out["peak_vals"][row][best]

    def _centered(self, crops) -> list[tuple[np.ndarray, np.ndarray]]:
        """centered_instance: a ``crop_size`` crop centred on the subject's box, zero-padded
        past the frame's edge as SLEAP pads; peaks come back in crop pixels and are moved by
        the crop's corner."""
        crop_h, crop_w = self._input["crop_size"]
        images, corners = [], []
        for crop in crops:
            h, w = crop.frame.shape[:2]
            x1, y1, x2, y2 = crop.bbox
            cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
            left = int(round(cx - crop_w / 2))
            top = int(round(cy - crop_h / 2))
            image = self._image(crop.frame)
            padded = np.zeros((image.shape[0], crop_h, crop_w), np.uint8)
            sx0, sy0 = max(left, 0), max(top, 0)
            sx1, sy1 = min(left + crop_w, w), min(top + crop_h, h)
            if sx1 > sx0 and sy1 > sy0:
                padded[:, sy0 - top : sy1 - top, sx0 - left : sx1 - left] = image[
                    :, sy0:sy1, sx0:sx1
                ]
            images.append(padded)
            corners.append((left, top, w, h))
        out = self._run(np.stack(images))
        results = []
        for row, (left, top, w, h) in enumerate(corners):
            peaks = out["peaks"][row] + np.array([left, top], np.float32)
            results.append(_result(peaks, out["peak_vals"][row], (w, h)))
        return results


class _Subject:
    """One subject handed to a preset: the whole frame and its tracked box (normalized xyxy)."""

    __slots__ = ("frame", "bbox")

    def __init__(self, frame: np.ndarray, bbox) -> None:
        self.frame = frame
        self.bbox = bbox


def _result(
    peaks: np.ndarray, vals: np.ndarray, frame_wh
) -> tuple[np.ndarray, np.ndarray]:
    """Frame pixels → normalized; an unplaced point (NaN) carries a NaN score, not 0."""
    w, h = frame_wh
    xy = (np.asarray(peaks, np.float64) / np.array([w, h], np.float64)).astype(
        np.float32
    )
    placed = np.isfinite(xy).all(axis=1)
    score = np.where(placed, np.asarray(vals, np.float32), np.nan).astype(np.float32)
    xy[~placed] = np.nan
    return xy, score
