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

from PyQt6.QtWidgets import QComboBox, QFormLayout

from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.core.dataclasses import ComputeTarget
from detail.core.utils import available_backends, best_supported_backend
from detail.gui.help_ui import help_button, label_with
from detail.gui.spin import IntSpin

_ALL_BACKENDS = ["cuda", "mps", "cpu"]
_REMOTE_BACKEND = "remote"
_MAX_DEVICE_INDEX = 8
_NONE_AVAILABLE = "(none available)"


def remote_targets_for(
    backends: list[str], allow_remote: bool, *, no_model: bool = False
) -> list[RemoteTargetConfig]:
    """Registered remote targets usable for a model needing ``backends``.

    Independent of this *machine's* own hardware (unlike ``available_backends``) — a
    registered Slurm cluster is exactly what a GPU-less laptop needs for a cuda-only
    model — but still gated on the caller opting in (``allow_remote``) and the model
    itself being cuda-capable (§6: Slurm targets are GPU nodes). ``no_model`` is the
    other way in: CPU work that loads no model (keypoint track) runs on a GPU node's CPU as
    well as on this machine's. The one place this condition is written;
    `ComputeTargetRow.set_backends` and the model picker
    (`gui/model_availability.py::apply_availability`) both read it rather than each
    restating "cuda in backends and allow_remote" on its own.
    """
    if not allow_remote or ("cuda" not in backends and not no_model):
        return []
    return api_remote.list_remote_targets()


class ComputeTargetRow:
    """A backend combo plus its CUDA device number.

    Starts on the best backend this machine has. Call ``set_backends`` when the choice
    narrows to what a selected model supports; leave it alone for a form that offers all
    three.

    ``allow_remote`` is a required-decision, not a default: whether *this* form's run
    can be routed to a registered Slurm cluster (§6) is a fact about the module it
    launches, not something every embedder of this shared widget should silently start
    offering. Leave it ``False`` until the run's executor path actually supports it.

    ``cpu_only`` is for a stage that loads no model: it offers this machine's CPU and, with
    ``allow_remote``, every registered cluster, whose nodes have CPUs whatever else they hold.
    """

    def __init__(self, allow_remote: bool = False, cpu_only: bool = False) -> None:
        self._allow_remote = allow_remote
        self._cpu_only = cpu_only
        self.backend = QComboBox()
        self.device = IntSpin()
        self.device.setRange(0, _MAX_DEVICE_INDEX)
        self.remote_target = QComboBox() if allow_remote else None
        self.backend.currentTextChanged.connect(self._sync_device_enabled)
        self.set_backends(["cpu"] if cpu_only else _ALL_BACKENDS)

    def add_to(self, form: QFormLayout, prefix: str = "") -> None:
        """Add the rows. ``prefix`` distinguishes several targets on one form (the
        Distill dialog labels its stages "Label" and "Train")."""
        backend_label = f"{prefix} backend:" if prefix else "Backend:"
        device_label = f"{prefix} CUDA device:" if prefix else "CUDA device:"
        form.addRow(
            label_with(backend_label, help_button("compute-target")), self.backend
        )
        if not self._cpu_only:  # no state of a CPU-only row ever picks a device
            form.addRow(device_label, self.device)
        if self.remote_target is not None:
            remote_label = f"{prefix} remote target:" if prefix else "Remote target:"
            form.addRow(
                label_with(remote_label, help_button("compute-target")),
                self.remote_target,
            )

    def set_backends(self, backends: list[str]) -> None:
        """Offer the ones ``backends`` and this machine have in common, fastest selected.

        The intersection is the point. Offering a component's own list unfiltered put
        ``cuda`` in front of a Mac user for a cuda-only model, and Qt made it stick:
        ``best_supported_backend(["cuda"])`` falls back to ``"cpu"``, and
        ``setCurrentText("cpu")`` on a combo holding only ``"cuda"`` is a no-op, so the
        form sat on a device that does not exist here with its index box live.

        Nothing in common leaves a disabled placeholder — the caller is expected to have
        blocked the run already, by the same reasoning that left nothing to offer.

        A registered remote target is offered independent of this *machine's* hardware
        (unlike the intersection above) whenever the module supports ``cuda`` — a
        cuda-only model is exactly the case a GPU-less laptop needs a cluster for.
        """
        usable = [b for b in backends if b in available_backends()]
        remote_targets = remote_targets_for(
            backends, self._allow_remote, no_model=self._cpu_only
        )
        self.backend.blockSignals(True)
        self.backend.clear()
        if usable:
            self.backend.addItems(usable)
        if remote_targets:
            self.backend.addItem(_REMOTE_BACKEND)
        if usable:
            self.backend.setCurrentText(best_supported_backend(usable))
        elif remote_targets:
            self.backend.setCurrentText(_REMOTE_BACKEND)
        else:
            self.backend.addItem(_NONE_AVAILABLE)
        self.backend.setEnabled(bool(usable or remote_targets))
        self.backend.blockSignals(False)
        if self.remote_target is not None:
            self.remote_target.clear()
            for target in remote_targets:
                self.remote_target.addItem(target.name)
        self._sync_device_enabled()

    def current_backend(self) -> str:
        return self.backend.currentText() if self.backend.isEnabled() else ""

    def compute_target(self) -> ComputeTarget:
        backend = self.current_backend()
        if backend == _REMOTE_BACKEND:
            remote_target = (
                self.remote_target.currentText() if self.remote_target else ""
            )
            return ComputeTarget(
                backend, None, None, None, remote_target=remote_target or None
            )
        return ComputeTarget(
            backend,
            [self.device.value()] if backend == "cuda" else None,
            None,
            None,
        )

    def _sync_device_enabled(self) -> None:
        # Greyed rather than hidden, per the GUI style rule.
        self.device.setEnabled(self.backend.currentText() == "cuda")
        if self.remote_target is not None:
            self.remote_target.setEnabled(self.backend.currentText() == _REMOTE_BACKEND)
