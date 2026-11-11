"""A registry member that grows a configurable dimension needs a config channel.

**The class.** When a registry's members become configurable and the protocol has nowhere to
carry the configuration, the next feature does not add the channel — it goes *around the
registry*. And what it loses is not the feature, which works fine from the surface it was
built for, but every other surface the registry feeds. That loss is silent: nothing raises,
no test fails, and the gap is only visible to someone who tries to reach the capability from
somewhere else.

The instance: `StreamImporter.import_stream(data_path, out_dir, aux_paths)` had no `config`
argument, so explicit CSV column mapping was written as a free function outside
`STREAM_IMPORTERS`, writing an `importer: csv_column_mapping` value that resolved to nothing
in the registry. Three surfaces went with it, all silently:

- the GUI picker enumerates `STREAM_IMPORTERS`, so the path needed a *second* button;
- `recording add-stream --importer` could not name it;
- `storage/manifest.py::_validate_streams` rejects an unregistered importer id, so the
  bundle manifest — the only bulk path — could not express it at all. A user importing 60
  recordings could not use the feature on any of them.

**The rule.** A registry member's parameters live in a pydantic `config_model` and reach it
through the protocol, the way `Tool` and `ProcessingModule` already do. Then one registration
gives every surface the capability at once, and `configs/describe.py::iter_config_models`
carries its parameters into the docs and the config guards for free.

**What is checked here.** That the channel exists and is honoured end to end: every importer
accepts `config`, an importer declaring a `config_model` is reachable through it, one that
declares none refuses a config rather than ignoring it, and the parameters land in the
generated reference. `tests/configs/test_describe.py` covers the registry-walk half (it fails
when a registry grows a config model the reference never sees), and
`tests/api/test_recording_streams.py` covers what the mapping produces.
"""

import inspect

import pytest

from detail.configs.describe import iter_config_models
from detail.core.protocols import StreamImporter
from detail.core.registries import STREAM_IMPORTERS
from detail.importers import import_stream_importers


@pytest.fixture(autouse=True)
def _registry():
    import_stream_importers()
    assert STREAM_IMPORTERS, "no importers registered; this file would pass vacuously"


def test_every_importer_accepts_a_config_argument():
    """The channel is on the protocol, so a caller never has to ask which importer takes one.

    An importer whose signature lacks it raises `TypeError` from `add_stream`, which is a
    crash rather than the silent bypass this file is about, but it is the same boundary.
    """
    missing = [
        importer_id
        for importer_id, cls in STREAM_IMPORTERS.items()
        if "config" not in inspect.signature(cls.import_stream).parameters
    ]
    assert not missing, f"importers whose import_stream takes no config: {missing}"


def test_the_protocol_declares_the_channel():
    """Guard the guard: the check above is vacuous if the base class stops declaring it."""
    assert "config" in inspect.signature(StreamImporter.import_stream).parameters
    assert hasattr(StreamImporter, "config_model")


def test_at_least_one_importer_declares_a_config_model():
    """Guard the guard, the other half: a sweep over importers that all declare `None`
    proves nothing about the channel actually working."""
    declared = [
        importer_id
        for importer_id, cls in STREAM_IMPORTERS.items()
        if getattr(cls, "config_model", None) is not None
    ]
    assert declared, "no importer declares a config_model, so the channel is untested"


def test_importer_parameters_reach_the_generated_reference():
    """The payoff for staying inside the registry: the docs, `detail help` and every config
    guard read `iter_config_models`, so an importer's parameters are covered by all of them
    the day its model exists."""
    reachable = {module_id for module_id, _ in iter_config_models()}
    declared = {
        importer_id
        for importer_id, cls in STREAM_IMPORTERS.items()
        if getattr(cls, "config_model", None) is not None
    }
    assert declared <= reachable, sorted(declared - reachable)


def test_an_importer_with_no_config_model_refuses_a_config(
    experiment_dir, recording_dir, tmp_path
):
    """Refused, not ignored. Silently dropping it is how a mapping that was never applied
    still produces a well-formed stream, with the wrong columns and no error."""
    from detail.api.recording import Recording

    labels = tmp_path / "labels.csv"
    labels.write_text("start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="takes no config"):
        Recording(experiment_dir, "recording_001").add_stream(
            "b", raw_path=labels, importer="behavior_csv", config={"time_column": "x"}
        )
