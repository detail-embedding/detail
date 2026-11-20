"""The one seam that imports ultralytics (Section 5).

Ultralytics phones home by default. It resolves ``ONLINE = is_online()`` at import
time (``utils/__init__.py``) — a live DNS probe of one.one.one.one / dns.google —
and snapshots that constant into a module-level ``events = Events()`` singleton
whose ``enabled`` is decided once, in ``Events.__init__``. That one constant gates,
together: the Google-Analytics POST on every predict/train/val, the PyPI
version check on train, and ``check_requirements``' auto-pip-install.

Because all of it is decided at *import* time, ``SETTINGS["sync"] = False``
afterwards is a no-op (and would mutate the user's global ultralytics settings.json
besides) — the same class of bug as ``HF_HUB_CACHE`` in ``hf_access``. The only
lever that works is the env var, set before the first import anywhere in the
process, which is why every ultralytics import in this package goes through here.

Weight downloads still work: the only lazy ``is_online()`` sits inside a download
exception handler, where it makes a first-attempt failure raise instead of retry.
"""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from detail.core.errors import UserInputError
from detail.core.visualize import bgr_to_rgb

if TYPE_CHECKING:
    from ultralytics import YOLO

#: What ``YOLO(path)`` raised on a file that is not an Ultralytics checkpoint, as observed with
#: Ultralytics 8.4.215 (#33 finding 7): ``TypeError`` for an empty or cut-short file, a text or
#: other non-PyTorch file renamed ``.pt``, and a PyTorch file holding something other than a
#: model; ``AttributeError`` for a PyTorch file holding a list; ``RuntimeError`` for a zip that is
#: not a PyTorch archive. Named, not ``Exception``: anything else is a bug and keeps its traceback.
_NOT_A_CHECKPOINT = (TypeError, AttributeError, RuntimeError)


def import_ultralytics():
    """Import ultralytics with its network side-effects disabled, and return it."""
    # Must be the string "true": the check is
    # `str(os.getenv("YOLO_OFFLINE", "")).lower() == "true"` — "1" is silently ignored.
    os.environ.setdefault("YOLO_OFFLINE", "true")
    import ultralytics

    return ultralytics


def read_checkpoint(weights_path: Path | str) -> "YOLO":
    """The Ultralytics model in a ``.pt`` checkpoint a user hands in, for reading what it is.

    The one reader for a checkpoint picked for import (``model import``, the GUI's Import
    model): a file that is not one is refused as a :class:`UserInputError` naming it, never as
    whatever Ultralytics raised (#33 finding 7). The suffix is checked first, so another format
    Ultralytics can open (an exported ``model.onnx``) is not stored as ``weights/best.pt``,
    and a missing path is refused before Ultralytics could take its name for one to download.

    Raises:
        FileNotFoundError: ``weights_path`` does not exist.
        UserInputError: The file is not a ``.pt`` file, is empty, or Ultralytics cannot read
            it as a checkpoint.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.suffix.lower() != ".pt":
        raise UserInputError(
            f"{path.name} is not a YOLO .pt checkpoint; import the .pt file an Ultralytics "
            "training wrote (its weights/best.pt)"
        )
    if path.stat().st_size == 0:
        raise UserInputError(
            f"{path.name} is empty, so it is not a YOLO .pt checkpoint; copy the file again"
        )
    import_ultralytics()
    from ultralytics import YOLO

    try:
        return YOLO(str(path))
    except _NOT_A_CHECKPOINT as exc:
        raise UserInputError(
            f"{path.name} is not a YOLO .pt checkpoint Ultralytics can read (cut short, or "
            "another kind of file). Import the weights/best.pt an Ultralytics training "
            "wrote"
        ) from exc


class SplitPredict:
    """Ultralytics' ``predict`` on a list of frames, cut at its own stage boundaries (§7
    *Execution*): a CPU half that may run on another thread, and a device half.

    ``Model.predict`` runs ``preprocess`` → ``inference`` → ``postprocess`` back to back under
    one lock, so a caller cannot letterbox the next batch while the GPU runs this one. These
    are the same stage methods with the same arguments, called separately:

    - :meth:`prepare` is the numpy part of ``BasePredictor.preprocess`` (``pre_transform``'s
      letterbox, BGR→RGB, stack, BCHW, contiguous), stopping before the device copy.
    - :meth:`run` is the rest of ``preprocess`` (device copy, dtype, ``/255``), then
      ``inference`` and ``postprocess``, under ``torch.inference_mode`` as
      ``stream_inference`` runs them.

    The predictor is built and warmed once by a public ``predict`` call on the first batch's
    first frame, whose output is discarded: the same ``setup_model`` / ``setup_source`` /
    warmup the first ``predict`` of a serial run performs. Each later batch is set up as
    ``setup_source`` would for a list of numpy frames: one batch, ``image{i}.jpg`` paths.

    A batch must be split the way ``predict`` would have received it: letterbox padding
    depends on whether every frame in the batch has one shape. ``tests/models/test_trained_yolo.py``
    compares this against ``predict`` byte for byte, so an Ultralytics release that moves a
    stage fails there rather than in a run.
    """

    def __init__(self, model, **predict_kwargs) -> None:
        self._model = model
        self._kwargs = dict(predict_kwargs)
        self._predictor = None
        self._lock = threading.Lock()

    def predictor(self, frames: list):
        """The warmed predictor, built by one public ``predict`` call the first time."""
        with self._lock:
            if self._predictor is None:
                self._model.predict(frames[:1], **self._kwargs)
                self._predictor = self._model.predictor
            return self._predictor

    def prepare(self, frames: list):
        """The CPU half: a uint8 BCHW tensor still on the host."""
        import torch

        predictor = self.predictor(frames)
        # Ultralytics reverses the stacked batch's channels with numpy; cv2's swap is the
        # same permutation, bit for bit, without the strided copy (core/visualize.py).
        im = np.stack(
            [
                bgr_to_rgb(x) if x.shape[-1] == 3 else x
                for x in predictor.pre_transform(frames)
            ]
        )
        im = np.ascontiguousarray(im.transpose((0, 3, 1, 2)))
        return torch.from_numpy(im)

    def run(self, prepared, frames: list) -> list:
        """The device half: one ``Results`` per frame, as ``predict`` returns them."""
        import torch

        predictor = self._predictor
        with torch.inference_mode():
            im = prepared.to(predictor.device)
            im = im.half() if predictor.model.fp16 else im.float()
            im /= 255
            preds = predictor.inference(im)
            predictor.batch = (
                [f"image{i}.jpg" for i in range(len(frames))],
                frames,
                [""] * len(frames),
            )
            return predictor.postprocess(preds, im, frames)
