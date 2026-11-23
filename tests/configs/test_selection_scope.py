"""Every config field that names an artifact inside one experiment is declared a *selection*.

**The class.** A data reference left at the default ``scope``. ``describe_config`` reads a
field's ``scope`` from its ``json_schema_extra`` and defaults an undeclared one to ``"setting"``,
so a run id nobody annotated is, to every reader, a reusable knob. Four things read it:

- ``configs/baselines.py::_check`` refuses a shipped baseline that carries a selection. It is
  the rule baselines rest on: a baseline means the same thing in another experiment only if it
  holds no ids, because ``track_01`` there is somebody else's tracking run, or nothing.
- ``configs/describe.py::changed_from_defaults`` leaves selections out of the one-line summary
  **Start from** shows for a previous run, since they are never carried to the new one.
- ``gui/modules/registry.py::ConfigWidget.load_config`` restores settings only.
- ``detail analysis status`` marks a selection ``[selection]`` in a run's settings block.

**How it hid.** The analysis modules declared their run ids (``embed_run_id``,
``source_recording_ids``) and the processing modules never did. Nothing failed, because each
reader had a second reason to be right: no baseline ships yet, and every processing form
emits a batch *spec* (``detection_spec``), never the id, and re-asks for it in its own
``load_config``. ``tests/configs/test_baselines.py`` proved the refusal with ``embed_run_id``,
a field that was declared, so it passed while ``baselines._check("t", "embedding",
{"tracking_run_id": "track_01"})`` was accepted. So was every other processing run id, the
example set ``keypoint_detect`` learns from, the per-recording ``arena_ids``, and a
comparison's snapshot and recording list. Only ``location_export.tracking_run_id`` was refused.
A documentation review found it by reading the baselines README against the code.

**The rule.** A field whose name is an identifier (``*_id``, ``*_ids``, ``*_snapshot``,
``recordings``), or whose help cites an id the tool mints (``det_01``, ``kt_NN``), is declared
``"scope": "selection"``. The exceptions are in ``NOT_AN_ARTIFACT``, each with its reason. An
entry is a claim: one that stops excusing a field fails. Keyed by module, so a new module's
``model_id`` is classified by whoever adds it.
"""

import re

from detail.configs.baselines import BaselineError, _check
from detail.configs.describe import describe_config, iter_config_models

#: A field name that reads as a pointer to something rather than a value.
_IDENTIFIER_NAME = re.compile(r"(?:_ids?|_snapshot|^recordings)$")
#: An id the tool mints inside an experiment, as a help text cites one: ``det_01``, ``ex_NN``.
_MINTED_ID = re.compile(r"\b[a-z]+_(?:\d{2,}|NN)\b")

_REGISTERED_MODEL = "a registered model id, which is the same model in every experiment"
_MODEL_OR_TRAINED = (
    "usually a registered model id, which travels; it can also name a model trained in this "
    "experiment, which does not. Left a setting because `selection` would also drop the model "
    "from Start from's run summary, where it is ranked first on purpose; refusing a trained "
    "id in a baseline needs a check on the value, not the scope"
)
_TRACK = (
    "names a track inside a tracking run, and a single-subject run is always track 0. One "
    "classification for both modules: behavior clustering's form restores it through "
    "the base `load_config`, so `selection` would stop Start from carrying it there"
)
_TRACKER = "a registered tracker, the same code in every experiment"

#: ``(module_id, field)`` pairs that look like an identifier and are deliberately a setting.
NOT_AN_ARTIFACT: dict[tuple[str, str], str] = {
    ("tracking", "tracker_id"): _TRACKER,
    ("embedding", "model_id"): _REGISTERED_MODEL,
    ("labeling", "model_id"): _REGISTERED_MODEL,
    ("detection", "model_id"): _MODEL_OR_TRAINED,
    ("keypoint_detect", "model_id"): _MODEL_OR_TRAINED,
    ("query", "track_id"): _TRACK,
    ("temporal_clustering", "track_id"): _TRACK,
}


def _names_an_artifact(field: dict) -> bool:
    """Whether a ``describe_config`` field looks like a pointer to something in an experiment."""
    return bool(
        _IDENTIFIER_NAME.search(field["name"])
        or _MINTED_ID.search(field.get("help") or "")
    )


def _misdeclared(
    models: list[tuple[str, list[dict]]], exempt: dict[tuple[str, str], str]
) -> list[str]:
    """``module.field`` for each artifact-naming field not declared a selection nor exempt."""
    return [
        f"{module_id}.{field['name']}"
        for module_id, fields in models
        for field in fields
        if _names_an_artifact(field)
        and field["scope"] != "selection"
        and (module_id, field["name"]) not in exempt
    ]


def _described() -> list[tuple[str, list[dict]]]:
    return [
        (module_id, describe_config(model)) for module_id, model in iter_config_models()
    ]


def _artifact_fields() -> list[tuple[str, dict]]:
    return [
        (module_id, field)
        for module_id, fields in _described()
        for field in fields
        if _names_an_artifact(field)
        and (module_id, field["name"]) not in NOT_AN_ARTIFACT
    ]


def test_every_field_that_names_an_artifact_is_a_selection():
    wrong = _misdeclared(_described(), NOT_AN_ARTIFACT)
    assert not wrong, (
        f"these fields name an artifact inside one experiment but are not declared "
        f'`"scope": "selection"`: {", ".join(wrong)}. A shipped baseline could carry them, '
        "Start from's summary shows them as if they were carried, and a form that restores "
        "settings through the base `load_config` would restore them. Declare the scope, or "
        "add the field to NOT_AN_ARTIFACT with the reason it travels."
    )


def test_a_baseline_refuses_every_field_that_names_an_artifact():
    """The property the README promises, asked of the loader rather than of the declaration."""
    accepted = []
    for module_id, field in _artifact_fields():
        value = [] if field["type"] == "array" else "x_01"
        try:
            _check("Probe", module_id, {field["name"]: value})
        except BaselineError as exc:
            if "selection" in str(exc):
                continue
        accepted.append(f"{module_id}.{field['name']}")
    assert (
        not accepted
    ), f"a baseline naming any of these is not refused as a selection: {', '.join(accepted)}"


def test_every_exemption_still_excuses_a_field():
    """An entry for a field that was renamed, removed or declared a selection excuses nothing."""
    fields = {
        (module_id, field["name"]): field
        for module_id, fields in _described()
        for field in fields
    }
    stale = []
    for key in NOT_AN_ARTIFACT:
        field = fields.get(key)
        if field is None:
            stale.append(f"{key[0]}.{key[1]} (no such field)")
        elif not _names_an_artifact(field):
            stale.append(f"{key[0]}.{key[1]} (no longer looks like an identifier)")
        elif field["scope"] == "selection":
            stale.append(f"{key[0]}.{key[1]} (now declared a selection)")
    assert not stale, "stale NOT_AN_ARTIFACT entries: " + ", ".join(stale)


def test_the_sweep_would_catch_a_run_id_declared_a_setting():
    """Guard the guard, on the shapes that slipped through and the ones a person would write."""
    reported = {
        "name": "tracking_run_id",
        "help": "Tracking run whose tracks to embed (e.g. track_01).",
        "scope": "setting",
    }
    by_help_only = {
        "name": "upstream",
        "help": "The detection run to read (e.g. det_01).",
        "scope": "setting",
    }
    a_snapshot = {"name": "source_snapshot", "help": "", "scope": "setting"}
    a_recording_list = {"name": "recordings", "help": "", "scope": "setting"}
    exempt_elsewhere = {"name": "model_id", "help": "", "scope": "setting"}
    declared = {"name": "embed_run_id", "help": "(e.g. emb_01)", "scope": "selection"}
    a_knob = {"name": "batch_size", "help": "Frames per pass.", "scope": "setting"}
    models = [
        (
            "probe",
            [
                reported,
                by_help_only,
                a_snapshot,
                a_recording_list,
                exempt_elsewhere,
                declared,
                a_knob,
            ],
        )
    ]
    assert _misdeclared(models, NOT_AN_ARTIFACT) == [
        "probe.tracking_run_id",
        "probe.upstream",
        "probe.source_snapshot",
        "probe.recordings",
        "probe.model_id",
    ]


def test_the_sweep_reaches_the_fields_it_guards():
    """A walk over nothing passes. The fields the review found are among those swept."""
    swept = {f"{module_id}.{field['name']}" for module_id, field in _artifact_fields()}
    assert len(swept) >= 16, sorted(swept)
    assert {
        "embedding.tracking_run_id",
        "tracking.detection_run_id",
        "keypoint_detect.example_set_id",
        "keypoint_track.detect_run_id",
        "location_export.tracking_run_id",
    } <= swept
