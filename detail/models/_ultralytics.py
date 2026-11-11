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

import numpy as np

from detail.core.visualize import bgr_to_rgb


def import_ultralytics():
    """Import ultralytics with its network side-effects disabled, and return it."""
    # Must be the string "true": the check is
    # `str(os.getenv("YOLO_OFFLINE", "")).lower() == "true"` — "1" is silently ignored.
    os.environ.setdefault("YOLO_OFFLINE", "true")
    import ultralytics

    return ultralytics


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
