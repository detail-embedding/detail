"""The "which device do I run on" pair of rows, in one place.

A backend combo and a CUDA device number appear on every form that launches real work —
detection, embedding, quick segment, and each stage of the Distill flow. Nine copies had
drifted apart: five repeated the same one-line "grey the device box unless the backend is
cuda" slot, two repeated the combo rebuild verbatim, and one forgot the greying entirely, so
its device number stayed live on a Mac.

Not a QWidget: it adds its two rows to the caller's ``QFormLayout`` so the labels keep the
form's own label column. Build it, ``add_to`` a form, then read ``compute_target()``.
"""

from __future__ import annotations

from detail.core.dataclasses import ComputeTarget
from detail.core.utils import available_backends, best_supported_backend
from detail.gui.help_ui import help_button, label_with
from detail.gui.qt import QComboBox, QFormLayout
from detail.gui.spin import IntSpin

_ALL_BACKENDS = ["cuda", "mps", "cpu"]
_MAX_DEVICE_INDEX = 8
_NONE_AVAILABLE = "(none available)"


class ComputeTargetRow:
    """A backend combo plus its CUDA device number.

    Starts on the best backend this machine has. Call ``set_backends`` when the choice
    narrows to what a selected model supports; leave it alone for a form that offers all
    three.
    """

    def __init__(self) -> None:
        self.backend = QComboBox()
        self.device = IntSpin()
        self.device.setRange(0, _MAX_DEVICE_INDEX)
        self.backend.currentTextChanged.connect(self._sync_device_enabled)
        self.set_backends(_ALL_BACKENDS)

    def add_to(self, form: QFormLayout, prefix: str = "") -> None:
        """Add the two rows. ``prefix`` distinguishes several targets on one form (the
        Distill dialog labels its stages "Label" and "Train")."""
        backend_label = f"{prefix} backend:" if prefix else "Backend:"
        device_label = f"{prefix} CUDA device:" if prefix else "CUDA device:"
        form.addRow(
            label_with(backend_label, help_button("compute-target")), self.backend
        )
        form.addRow(device_label, self.device)

    def set_backends(self, backends: list[str]) -> None:
        """Offer the ones ``backends`` and this machine have in common, fastest selected.

        The intersection is the point. Offering a component's own list unfiltered put
        ``cuda`` in front of a Mac user for a cuda-only model, and Qt made it stick:
        ``best_supported_backend(["cuda"])`` falls back to ``"cpu"``, and
        ``setCurrentText("cpu")`` on a combo holding only ``"cuda"`` is a no-op, so the
        form sat on a device that does not exist here with its index box live.

        Nothing in common leaves a disabled placeholder — the caller is expected to have
        blocked the run already, by the same reasoning that left nothing to offer.
        """
        usable = [b for b in backends if b in available_backends()]
        self.backend.blockSignals(True)
        self.backend.clear()
        if usable:
            self.backend.addItems(usable)
            self.backend.setCurrentText(best_supported_backend(usable))
        else:
            self.backend.addItem(_NONE_AVAILABLE)
        self.backend.setEnabled(bool(usable))
        self.backend.blockSignals(False)
        self._sync_device_enabled()

    def current_backend(self) -> str:
        return self.backend.currentText() if self.backend.isEnabled() else ""

    def compute_target(self) -> ComputeTarget:
        backend = self.current_backend()
        return ComputeTarget(
            backend,
            [self.device.value()] if backend == "cuda" else None,
            None,
            None,
        )

    def _sync_device_enabled(self) -> None:
        # Greyed rather than hidden, per the GUI style rule.
        self.device.setEnabled(self.backend.currentText() == "cuda")
