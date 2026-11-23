"""A model you can pick but never run is worse than one that is greyed out.

The pickers listed every registered model with no check at all, so on a Mac the detection
form opened on a CUDA-only SAM 3 with `Backend: cuda` and a live device box, and the run
died on `requested cuda device(s) [0] not available` — after the launch dialog had closed.
The GUI rule is grey out, don't hide: the row stays, with the reason on it.

Every test here states the machine it is about (``real_machine`` opts out of the
suite-wide "everything installed, every backend present" default in ``conftest``).
"""

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core.utils import available_backends
from detail.executor.runners import import_runner_modules
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.model_availability import (
    apply_availability,
    fill_variants,
    first_available_index,
    mark_variants,
    no_models_message,
    selected_model_id,
    unavailable_reason,
    variant_available,
)
from detail.gui.modules.registry import GUI_MODULES
from detail.gui.qt import QComboBox
from detail.gui.tools.registry import TOOL_GUIS, import_tool_uis


@pytest.fixture
def mac(monkeypatch, real_machine):
    """Apple silicon: mps, no CUDA. Everything downloadable is downloaded."""
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: True)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)


@pytest.fixture
def gpu_box(monkeypatch, real_machine):
    """A CUDA machine with nothing downloaded yet."""
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: False)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: False)


def _combo(*entries):
    combo = QComboBox()
    for name in entries:
        combo.addItem(name, name)
    return combo


def _rows(combo):
    return [
        (combo.itemText(i), combo.model().item(i).isEnabled())
        for i in range(combo.count())
    ]


# --------------------------------------------------------------------------- #
# available_backends — the machine, not a component's wish list
# --------------------------------------------------------------------------- #
def test_available_backends_reports_this_machine(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    assert available_backends() == ["mps", "cpu"]
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    assert available_backends() == ["cuda", "mps", "cpu"]


def test_the_backend_row_never_offers_a_device_this_machine_lacks(qapp, mac):
    """`best_supported_backend(["cuda"])` returns "cpu" — a backend not on the list it was
    given — and `setCurrentText("cpu")` past a cuda-only combo is a Qt no-op. So the row
    sat on cuda. It now shows what the machine and the model have in common, or nothing.
    """
    row = ComputeTargetRow()
    row.set_backends(["cuda", "mps", "cpu"])
    assert "cuda" not in [row.backend.itemText(i) for i in range(row.backend.count())]
    assert row.backend.currentText() == "mps"

    row.set_backends(["cuda"])  # a cuda-only model here
    assert row.backend.currentText() != "cuda"
    assert not row.backend.isEnabled()
    assert not row.device.isEnabled()
    assert row.compute_target().device_indices is None


# --------------------------------------------------------------------------- #
# The two reasons, in order
# --------------------------------------------------------------------------- #
def test_a_model_needing_an_absent_backend_says_so(qapp, mac):
    assert unavailable_reason("sam3") == "needs CUDA"  # sam3 is cuda-only


def test_the_backend_reason_wins_over_the_download_one(qapp, monkeypatch, real_machine):
    """Both are true for SAM 3 on a fresh Mac. Only one is actionable there — downloading
    it would still leave nothing to run it on."""
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: False)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    assert unavailable_reason("sam3") == "needs CUDA"


def test_a_missing_download_says_so_where_the_hardware_is_fine(qapp, gpu_box):
    assert unavailable_reason("sam3") == "not installed"


def test_a_model_counts_as_installed_when_any_of_its_sizes_is(
    qapp, monkeypatch, real_machine
):
    """Its sizes are separate downloads; the variant row carries which. Calling the whole
    model absent because one size is missing would hide the three that are there."""
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr(
        "detail.api.models.is_installed",
        lambda model_id, variant=None: variant == "convnext_large",
    )
    assert unavailable_reason("dinov3_convnext_base") is None


# --------------------------------------------------------------------------- #
# apply_availability
# --------------------------------------------------------------------------- #
def test_it_marks_greys_and_opens_on_something_runnable(
    qapp, monkeypatch, real_machine
):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr(
        "detail.api.models.is_installed", lambda model_id, variant=None: model_id != "a"
    )
    monkeypatch.setattr("detail.api.models.supported_backends", lambda _id: ["cuda"])
    combo = _combo("a", "b")
    assert apply_availability(combo, "detection") is None
    assert _rows(combo) == [("a  — not installed", False), ("b", True)]
    assert selected_model_id(combo) == "b"  # not the disabled row it opened on


def test_nothing_runnable_gives_a_message_and_a_row_that_admits_it(qapp, mac):
    """A greyed row sitting in a closed combo reads as *selected*."""
    combo = _combo("sam3")
    message = apply_availability(combo, "detection")
    assert message == no_models_message("detection")
    assert message == "No detection models available. Install models via Model manager."
    assert combo.currentText() == "(no model available)"
    assert selected_model_id(combo) is None


def test_marking_is_idempotent(qapp, mac):
    """Pickers are re-marked on every rebuild; suffixes and placeholders must not stack."""
    combo = _combo("sam3")
    for _ in range(3):
        apply_availability(combo, "detection")
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "(no model available)",
        "sam3  — needs CUDA",
    ]


def test_a_model_that_appears_drops_its_label(qapp, monkeypatch, real_machine):
    """Install it in another window, come back, and the row is live again — the marker is
    derived each time, never latched."""
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    installed = {"value": False}
    monkeypatch.setattr(
        "detail.api.models.is_installed", lambda *a, **k: installed["value"]
    )
    combo = _combo("sam3")
    assert apply_availability(combo, "detection") is not None
    installed["value"] = True
    assert apply_availability(combo, "detection") is None
    assert _rows(combo) == [("sam3", True)]


# --------------------------------------------------------------------------- #
# Every picker, one rule
# --------------------------------------------------------------------------- #
def _pickers():
    import_runner_modules(PIPELINE_RUNNERS)
    import_tool_uis()
    return {
        "detection": GUI_MODULES["detection"],
        "embedding": GUI_MODULES["embedding"],
        "quick_segment": TOOL_GUIS["quick_segment"],
    }


@pytest.mark.parametrize("form_id", sorted(_pickers()))
def test_no_picker_ships_its_placeholder_as_a_model_id(qapp, form_id, gpu_box):
    """Nothing runs here, so every picker sits on its ``(no model available)`` row. A form
    reading ``currentText()`` as an id would hand that string to the runner — the reason
    the id lives in ``itemData`` and is read back through one accessor.

    The embedding form is the one picker that stays on its configured default instead (a
    different model changes every vector, so it never moves for you): it reads back that
    model's real id, and the run is refused by ``validation_error``, never by a bogus id.
    """
    from detail.configs.embedding import EmbeddingConfig

    widget = _pickers()[form_id].config_widget()
    expected = (
        EmbeddingConfig.model_fields["model_id"].default
        if form_id == "embedding"
        else None
    )
    assert selected_model_id(widget._model) == expected


@pytest.mark.parametrize("form_id", sorted(_pickers()))
def test_every_picker_reports_a_real_catalog_id(
    qapp, form_id, monkeypatch, real_machine
):
    """And where something *is* runnable, the id is the model's, not its decorated label."""
    from detail.api import models as api_models

    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: True)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    widget = _pickers()[form_id].config_widget()
    known = {row.id for row in api_models.list_models()}
    assert selected_model_id(widget._model) in known


def test_the_config_a_run_receives_carries_the_bare_id(qapp, monkeypatch, real_machine):
    """``config()`` is what reaches the runner, so the accessor has to be what it uses —
    with a size suffixed too, since the variant row is marked the same way."""
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr(
        "detail.api.models.is_installed",
        lambda model_id, variant=None: variant != "vit_large",
    )
    widget = GUI_MODULES["embedding"].config_widget()
    assert (
        "—" in widget._variant.currentText()
    )  # the default size, marked not installed
    config = widget.config()
    assert config["model_id"] == "dinov3_vit"
    assert config["model_variant"] == "vit_large"


@pytest.mark.parametrize("form_id", sorted(_pickers()))
def test_every_picker_refuses_the_run_when_nothing_can_run(qapp, form_id, gpu_box):
    widget = _pickers()[form_id].config_widget()
    assert widget.validation_error() is not None
    assert "Install models via Model manager" in widget.validation_error()


@pytest.mark.parametrize("form_id", sorted(_pickers()))
def test_every_picker_shows_its_note_only_when_blocked(qapp, form_id, gpu_box):
    widget = _pickers()[form_id].config_widget()
    assert widget._no_models.text().endswith("Install models via Model manager.")


# --------------------------------------------------------------------------- #
# Variant rows (the half that already existed)
# --------------------------------------------------------------------------- #
def test_an_absent_size_is_disabled_and_says_why(qapp, monkeypatch, real_machine):
    monkeypatch.setattr(
        "detail.api.models.is_installed",
        lambda model_id, variant=None: variant == "convnext_base",
    )
    combo = _combo("convnext_tiny", "convnext_base")
    mark_variants(combo, "dinov3_convnext_base")
    assert _rows(combo) == [
        ("convnext_tiny  — not installed", False),
        ("convnext_base", True),  # the installed one is left alone
    ]


def test_the_check_never_consults_the_network(qapp, monkeypatch, real_machine):
    """``readiness()`` reaches the network when a model is absent, and this runs on the UI
    thread on every model change. Only the disk check belongs here."""
    called = []
    monkeypatch.setattr(
        "detail.api.models.is_installed", lambda *a, **k: called.append(a) or True
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("readiness() must not be called from the picker")

    monkeypatch.setattr("detail.models.hf_access.readiness", _boom)
    mark_variants(_combo("convnext_base"), "dinov3_convnext_base")
    assert called


def test_variant_relabelling_is_idempotent(qapp, monkeypatch, real_machine):
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: False)
    combo = _combo("convnext_tiny")
    for _ in range(3):
        mark_variants(combo, "dinov3_convnext_base")
    assert combo.itemText(0) == "convnext_tiny  — not installed"


def test_first_available_index_finds_a_runnable_row(qapp, monkeypatch, real_machine):
    monkeypatch.setattr(
        "detail.api.models.is_installed",
        lambda model_id, variant=None: variant == "convnext_large",
    )
    combo = _combo("convnext_tiny", "convnext_base", "convnext_large")
    mark_variants(combo, "dinov3_convnext_base")
    assert first_available_index(combo) == 2


def test_a_size_picker_opens_on_a_runnable_size_unless_told_to_keep_its_default(
    qapp, monkeypatch, real_machine
):
    """The declared default wins — unless it isn't downloaded. The detection and keypoint
    forms then open on a size that is (a disabled row is one the user has to notice); the
    embedding form passes ``keep_default`` and stays on it, because a different size changes
    every vector (its own guard: tests/gui/test_form_defaults_match_config.py)."""
    monkeypatch.setattr(
        "detail.api.models.is_installed",
        lambda model_id, variant=None: variant != "vit_large",
    )
    sizes = ["vit_small", "vit_base", "vit_large"]
    moved = QComboBox()
    fill_variants(moved, "dinov3_vit", sizes, "vit_large")
    assert moved.currentData() != "vit_large"
    assert moved.model().item(moved.currentIndex()).isEnabled()
    kept = QComboBox()
    fill_variants(kept, "dinov3_vit", sizes, "vit_large", keep_default=True)
    assert kept.currentData() == "vit_large"
    assert not variant_available(kept)
