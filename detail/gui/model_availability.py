"""Whether a model can run *on this machine*, and how a picker says so.

A picker that lists every registered model lets you choose one that cannot run here, and the
run then fails after the launch dialog has closed — a CUDA-only model on a Mac died on
``requested cuda device(s) [0] not available``, having shown ``Backend: cuda`` and an enabled
device box the whole time. The GUI rule is *grey out, don't hide* (`CLAUDE.md`): the row stays
visible with the reason attached, so it reads as "not here" rather than "not supported".

Two reasons, **backend first**: on a Mac, downloading SAM 3 would not help, so "needs CUDA" is
the useful thing to say even when the weights are also absent.

**Never ask ``readiness()`` here.** It consults the network when a model is absent
(``models/hf_access``), and this runs on the UI thread on every model change. The
authoritative disk check is ``api.models.is_installed`` — no network, and the same answer the
run's pre-flight gates on.

**A third way out, for a form that offers it:** a cuda-only model with no local CUDA is not
"needs CUDA" when the caller passes ``allow_remote=True`` and a Slurm cluster is registered
(§6) — the run can go there instead, the same reasoning `ComputeTargetRow.set_backends` uses
to offer a "remote" backend row (`gui/compute_target.py::remote_targets_for`, the one place
that condition is written). Neither local install state nor local hardware then blocks the
row: a remote job's own readiness is checked on the cluster before it is submitted
(`executor/remote_shepherd.py`), never here. Leave ``allow_remote`` at its default ``False``
for a form whose module isn't remote-wired — a cuda-only model there still needs its reason.

Call ``apply_availability`` once after building or rebuilding a picker's list, never from a
selection handler: it moves the selection to something runnable, which fires the form's own
sync handlers.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette, QStandardItemModel
from PyQt6.QtWidgets import QLabel

from detail.core.utils import available_backends
from detail.gui.compute_target import remote_targets_for
from detail.gui.status_color import WARN

# Appended to a row's label. Split on the separator to recover the base label, so re-marking
# a combo (a rebuild, a machine that gained a GPU) never stacks suffixes.
_SEPARATOR = "  — "
_NO_MODEL_ROW = "(no model available)"

_BACKEND_NAMES = {"cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}


def no_models_message(role: str) -> str:
    """The one sentence a form shows when nothing in its picker can run here."""
    return f"No {role} models available. Install models via Model manager."


def availability_note() -> QLabel:
    """The amber line a form shows when its picker has nothing runnable. Starts hidden."""
    label = QLabel("")
    label.setWordWrap(True)
    palette = label.palette()
    palette.setColor(QPalette.ColorRole.WindowText, WARN)
    label.setPalette(palette)
    label.setVisible(False)
    return label


def show_note(label: QLabel, message: str | None) -> None:
    """Show ``message`` on an ``availability_note`` label, or hide it for ``None``."""
    label.setText(message or "")
    label.setVisible(bool(message))


def _base_label(text: str) -> str:
    return text.split(_SEPARATOR)[0]


def _installed(model_id: str) -> bool:
    """Whether any of the model's weights are on disk.

    A model with sizes counts as installed if *any* size is — the variant row carries the
    per-size state, and opening on a model because one size is missing would hide the three
    that are there. An id the catalog does not own (an experiment-trained model) raises, and
    is installed by definition: its weights live in the experiment.
    """
    from detail.api import models as api_models

    try:
        variants = _variants(model_id)
        if not variants:
            return api_models.is_installed(model_id)
        return any(api_models.is_installed(model_id, name) for name in variants)
    except (KeyError, ValueError):
        return True


def _variants(model_id: str) -> list[str]:
    """The sizes a model has. A model that shares another's weights (`dinov3_match` on
    `dinov3_vit`) has no catalog row of its own, so its sizes come from its class: without
    this it read as size-less, was checked at its default size only, and a machine with
    `vit_small` installed but not `vit_large` was told no keypoint method was available.
    """
    from detail.api import models as api_models

    return api_models.model_variants(model_id)


def unavailable_reason(model_id: str, *, allow_remote: bool = False) -> str | None:
    """Why ``model_id`` cannot run here, or ``None`` when it can.

    ``allow_remote`` mirrors ``ComputeTargetRow(allow_remote=...)`` on the same form: when
    set, and a registered Slurm cluster can take this model (``remote_targets_for``), neither
    reason below applies — the run does not need this machine's backend or its disk.
    """
    from detail.api import models as api_models

    supported = api_models.supported_backends(model_id)
    # A catalog model (an imported or distilled one) qualifies too: the run uploads it.
    if remote_targets_for(supported, allow_remote):
        return None
    if not [b for b in supported if b in available_backends()]:
        names = " or ".join(_BACKEND_NAMES.get(b, b.upper()) for b in supported)
        return f"needs {names}"
    if not _installed(model_id):
        shared = _shares_weights_with(model_id)
        if shared:
            # This model has no row of its own in the Model manager, so "not installed"
            # names nothing anyone can install: name the row and the sizes it needs.
            sizes = " or ".join(_variants(model_id)) or "its weights"
            return f"needs {shared} {sizes} installed"
        return "not installed"
    from detail.models import onnx_keypoints

    if onnx_keypoints.is_imported(model_id) and not onnx_keypoints.backend_available():
        return "needs onnxruntime"  # an imported keypoint model's runtime (§26)
    return None


def _shares_weights_with(model_id: str) -> str | None:
    from detail.api import models as api_models

    entry = api_models.all_models().get(model_id)
    return getattr(entry[1], "shares_weights_with", None) if entry else None


def _tooltip(reason: str) -> str:
    if reason == "not installed":
        return "Not downloaded on this machine — install it from the Model manager."
    if reason == "needs onnxruntime":
        return "Imported keypoint models run with onnxruntime: pip install -e '.[keypoints_onnx]'"
    if reason.startswith("needs ") and reason.endswith(" installed"):
        return (
            f"This method reuses another model's weights; it {reason[:-len(' installed')]} "
            "— install that size on the Model manager."
        )
    return "This machine has no device this model can run on."


def selected_model_id(combo) -> str | None:
    """The chosen model's id — the only way to read it back.

    A label carries its reason as a suffix, so ``currentText()`` is a *label*, not an id;
    reading it as one shipped ``'sam3  — not installed'`` as a ``model_id``.
    """
    data = combo.currentData()
    if data is not None:
        return data["id"] if isinstance(data, dict) else data
    text = combo.currentText()
    return None if text == _NO_MODEL_ROW else _base_label(text)


def apply_availability(combo, role: str, *, allow_remote: bool = False) -> str | None:
    """Mark every row, open on one that can run, and report when none can.

    ``allow_remote`` is the same required-decision as ``ComputeTargetRow(allow_remote=...)``
    on the same form (`gui/compute_target.py`) — pass it only for a module whose executor
    path is actually remote-wired (detection, today), so a cuda-only model there is offered
    when a registered cluster can take it instead of the reason it would otherwise show.

    Returns ``None`` when something is runnable, else the sentence the form shows and
    ``validation_error`` refuses the run with.
    """
    item_model = combo.model()
    if not isinstance(item_model, QStandardItemModel):
        return None  # a combo whose view has no item flags — styling is a hint, not the gate
    for index in reversed(
        range(combo.count())
    ):  # drop a placeholder from a previous pass
        if combo.itemText(index) == _NO_MODEL_ROW:
            combo.removeItem(index)
    available = []
    for index in range(combo.count()):
        item = item_model.item(index)
        model_id = _model_id_at(combo, index)
        if item is None or model_id is None:
            continue
        reason = unavailable_reason(model_id, allow_remote=allow_remote)
        combo.setItemText(index, _base_label(combo.itemText(index)))
        item.setEnabled(reason is None)
        if reason is None:
            available.append(index)
            item.setData(None, Qt.ItemDataRole.ToolTipRole)
        else:
            combo.setItemText(index, combo.itemText(index) + _SEPARATOR + reason)
            item.setData(_tooltip(reason), Qt.ItemDataRole.ToolTipRole)
    if not available:
        # A greyed row sitting in a closed combo reads as *selected*. Say nothing is.
        combo.insertItem(0, _NO_MODEL_ROW)
        placeholder = item_model.item(0)
        if placeholder is not None:
            placeholder.setEnabled(False)
        combo.setCurrentIndex(0)
        return no_models_message(role)
    if combo.currentIndex() not in available:
        combo.setCurrentIndex(available[0])
    return None


def _model_id_at(combo, index: int) -> str | None:
    data = combo.itemData(index)
    if data is not None:
        return data["id"] if isinstance(data, dict) else data
    return _base_label(combo.itemText(index))


def mark_variants(combo, model_id: str) -> None:
    """Disable the sizes of ``model_id`` whose weights are absent, and label them."""
    from detail.api import models as api_models

    item_model = combo.model()
    if not isinstance(item_model, QStandardItemModel):
        return
    for index in range(combo.count()):
        variant = combo.itemData(index)
        item = item_model.item(index)
        if variant is None or item is None:
            continue
        try:
            installed = api_models.is_installed(model_id, variant)
        except (KeyError, ValueError):  # an id the catalog no longer knows
            continue
        item.setEnabled(installed)
        combo.setItemText(index, _base_label(combo.itemText(index)))
        if not installed:
            combo.setItemText(
                index, combo.itemText(index) + _SEPARATOR + "not installed"
            )
            item.setData(_tooltip("not installed"), Qt.ItemDataRole.ToolTipRole)


def fill_variants(
    combo, model_id: str, variants, default: str | None, *, keep_default: bool = False
) -> None:
    """Fill a size picker with ``variants`` of ``model_id``, sizes not installed greyed and
    labelled, opened on ``default`` when it is installed and otherwise on the first size that
    is (a run against a greyed size fails at validate).

    One place for what four forms (detection, embedding, keypoint detect, keypoint distill)
    each wrote out; the copy that was missing it opened on ``vit_large`` on a machine with only
    ``vit_base`` and failed inside *Test cut-off…*. Leaves an empty ``variants`` empty: what a
    size-less model shows instead is the caller's.

    ``keep_default`` leaves ``default`` selected, greyed and labelled *not installed*, instead
    of moving to an installed size. For a form whose default size is part of what a run *is*
    (embedding: the size changes every vector), quietly running a different size is worse than
    a launch the form refuses with a reason; the caller must then refuse an unavailable size in
    its ``validation_error`` (``variant_available``).
    """
    combo.clear()
    for name in variants:
        combo.addItem(name, name)
    if not variants:
        return
    if default is not None:
        index = combo.findData(default)
        if index >= 0:
            combo.setCurrentIndex(index)
    mark_variants(combo, model_id)
    if not keep_default and not variant_available(combo):
        runnable = first_available_index(combo)
        if runnable is not None:
            combo.setCurrentIndex(runnable)


def variant_available(combo) -> bool:
    """Whether the selected size can run here: :func:`mark_variants` greys one that cannot."""
    item_model = combo.model()
    if combo.currentIndex() < 0 or not isinstance(item_model, QStandardItemModel):
        return True
    item = item_model.item(combo.currentIndex())
    return item is None or item.isEnabled()


def first_available_index(combo) -> int | None:
    """The first entry that can run, or ``None`` — so a picker can open on something
    runnable instead of on a disabled row the user has to notice and move off."""
    item_model = combo.model()
    if not isinstance(item_model, QStandardItemModel):
        return None
    for index in range(combo.count()):
        item = item_model.item(index)
        if item is not None and item.isEnabled():
            return index
    return None
