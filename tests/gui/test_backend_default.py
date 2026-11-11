"""Compute forms default to the best available backend, not a hardcoded cuda."""

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import GUI_MODULES  # noqa: E402


# Detection's backend now tracks the selected model's supported_backends (its default
# model, SAM3, is cuda-only), so its combo is covered in test_detection_config.py.
@pytest.mark.parametrize("module_id", ["embedding"])
def test_compute_form_defaults_to_best_available_backend(
    qapp, tmp_path, monkeypatch, module_id
):
    # No NVIDIA GPU but an Apple accelerator present: the form must default to mps,
    # never the old hardcoded "cuda" (which fails on a machine without an NVIDIA GPU).
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    widget = GUI_MODULES[module_id].config_widget()
    widget.on_experiment_set(Experiment.create("p", tmp_path))
    assert widget._compute.backend.currentText() == "mps"
