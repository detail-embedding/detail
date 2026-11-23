"""The location export's launch form: every measurement option, from the config model.

**The class:** a config field the form does not offer. The form's ``config()`` is what a GUI
run is launched with, so an option missing from it is silently the default for every run
started from the app, and the record it writes is well-formed (``wire-every-surface``). The
round-trip guard cannot see it: that guard perturbs what the form *emits*, and a form that
emits nothing for a field has nothing to perturb. This one starts from the model's fields.
"""

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
    wanted = {"sample_fps": 5.0}
    assert set(wanted) == set(option_names()), "a new option needs a value in this test"
    form.load_config(wanted)
    assert _options(form) == wanted
    # and back to unset: None is a value the form must be able to show, not skip
    form.load_config({"sample_fps": None})
    assert _options(form) == {"sample_fps": None}


def test_an_unset_rate_reads_as_every_frame(qapp):
    form = _LocationExportConfig()
    assert form._options["sample_fps"].text() == EVERY_FRAME_TEXT
    form.load_config({"sample_fps": 5.0})
    assert form._options["sample_fps"].text() != EVERY_FRAME_TEXT


def test_a_config_from_before_thin_parts_were_removed_restores_without_them(qapp):
    # an older run's record still carries the removed options; the form shows what it has
    form = _LocationExportConfig()
    form.load_config(
        {"sample_fps": 2.0, "remove_thin_parts": True, "thin_part_scale": 0.2}
    )
    assert _options(form) == {"sample_fps": 2.0}
    assert "remove_thin_parts" not in form.config()
