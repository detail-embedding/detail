"""The location export's launch form: every measurement option, from the config model.

**The class:** a config field the form does not offer. The form's ``config()`` is what a GUI
run is launched with, so an option missing from it is silently the default for every run
started from the app, and the record it writes is well-formed (``wire-every-surface``). The
round-trip guard cannot see it: that guard perturbs what the form *emits*, and a form that
emits nothing for a field has nothing to perturb. This one starts from the model's fields.
"""

from PyQt6.QtWidgets import QCheckBox

from detail.configs.location_export import LocationExportConfig, option_names
from detail.gui.modules.analysis_location import (
    EVERY_FRAME_TEXT,
    _LocationExportConfig,
)


def _options(form) -> dict:
    return {name: form.config()[name] for name in option_names()}


def test_the_form_offers_every_option_of_the_config_at_its_default(qapp):
    form = _LocationExportConfig()
    defaults = LocationExportConfig()
    assert set(option_names()) <= set(form.config())
    assert _options(form) == {name: getattr(defaults, name) for name in option_names()}
    for name in option_names():
        described = LocationExportConfig.model_fields[name].description
        assert form._options[name].toolTip() == described, name


def test_the_form_gives_back_every_option_it_is_handed(qapp):
    form = _LocationExportConfig()
    wanted = {"sample_fps": 5.0, "remove_thin_parts": True, "thin_part_scale": 0.2}
    assert set(wanted) == set(option_names()), "a new option needs a value in this test"
    form.load_config(wanted)
    assert _options(form) == wanted
    # and back to unset: None is a value the form must be able to show, not skip
    form.load_config({"sample_fps": None, "remove_thin_parts": False})
    assert _options(form) == {**wanted, "sample_fps": None, "remove_thin_parts": False}


def test_an_unset_rate_reads_as_every_frame(qapp):
    form = _LocationExportConfig()
    assert form._options["sample_fps"].text() == EVERY_FRAME_TEXT
    form.load_config({"sample_fps": 5.0})
    assert form._options["sample_fps"].text() != EVERY_FRAME_TEXT


def test_the_scale_can_be_set_only_while_thin_parts_are_removed(qapp):
    form = _LocationExportConfig()
    thin, scale = form._options["remove_thin_parts"], form._options["thin_part_scale"]
    assert isinstance(thin, QCheckBox)
    assert not thin.isChecked() and not scale.isEnabled()
    thin.setChecked(True)
    assert scale.isEnabled()
    thin.setChecked(False)
    assert not scale.isEnabled()
    # a disabled box still reports its value: the record names the scale either way
    assert form.config()["thin_part_scale"] == LocationExportConfig().thin_part_scale
