"""A ``None`` for a setting with a default means that default, at every entry a run's config takes.

**The class: "use the default" spelled two ways, and only one of them accepted.** Internal
callers pass an optional keyword straight on into a stage's config, so "the default" arrives as
``None`` (keypoint setup's ``tracker_id: str | None = None``). The config models declare most
defaulted settings as a plain type (``tracker_id: str = DEFAULT_TRACKER``): pydantic fills the
default for an omitted key and refuses ``None``. Tracking's ``validate()`` ran the model before
its own ``or DEFAULT_TRACKER`` fallback, so **every keypoint setup, one animal or two, failed at
its tracking stage** with ``tracker_id: Input should be a valid string``. 232 defaulted fields
across 45 config models refused ``None`` the same way.

**How it hid.** Every form fills the field and every test of a default omits the key, the
spelling that worked. The setup's own tests replaced the tracking run with a stand-in, so its
config never met the model; they now check it (``tests/api/test_keypoint_setup.py``,
``_as_the_module_reads_it``). The GUI's one-animal setup kept working only because the main
checkout carried an uncommitted reorder of tracking's ``validate()``.

**The rule: resolve ``None`` at the entry, not in each module.** ``configs.defaults.
none_means_default`` replaces a ``None`` with the field's default wherever the field has one and
cannot hold ``None`` — there ``None`` was only ever an error, so nothing that validates today
changes — and writes the default into the dict, so the run record names it (Tenet 9). Every
function that hands a config to a module's ``validate`` / ``validate_batch`` calls it first;
the scan below enforces that, and a new entry fails it until it does or says why not.

Swept, by entry: ``Experiment._validate_processing`` (``run_processing``),
``Experiment._validate_retry_recordings`` (retry), ``Experiment.run_analysis``,
``Recording.run`` and ``api/keypoints._covers`` fixed. The three hidden entries this once
exempted (``run_annotation``, ``clip_query.query_clips``, ``discover_syllables``) were removed
with their capabilities (``docs/removed_capabilities.md``). Deliberately not: the modules' own
``config.get(x) or <default>`` fallbacks stay; each matches its model's default, and they still
serve a direct ``module.validate`` call with the key omitted. Keypoint setup also resolves its
``tracker_id`` at start, so its record names the tracker rather than ``null``.
"""

from __future__ import annotations

import ast
import typing
from pathlib import Path

import pydantic
import pytest
from pydantic_core import PydanticUndefined

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.configs.defaults import none_means_default
from detail.configs.tracking import DEFAULT_TRACKER
from detail.executor.runners import (
    ANALYSIS_MODULES,
    PROCESSING_MODULES,
    import_runner_modules,
)
from tests.api.test_experiment import _video_recording

REPO = Path(__file__).resolve().parents[2]
_VALIDATORS = {"validate", "validate_batch"}
#: (file, function) → why it hands a config to ``validate`` without ``none_means_default``.
EXEMPT: dict[tuple[str, str], str] = {}


def entries_without_it(source: str) -> list[str]:
    """Names of functions in ``source`` that call ``module.validate`` / ``validate_batch``
    without calling ``none_means_default``. Matches the receiver by name (``module``), which is
    what every entry in the package calls it; the non-vacuous test pins how many it finds.
    """
    missing = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        validates = any(
            isinstance(c.func, ast.Attribute)
            and c.func.attr in _VALIDATORS
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "module"
            for c in calls
        )
        resolves = any(
            (isinstance(c.func, ast.Name) and c.func.id == "none_means_default")
            or (
                isinstance(c.func, ast.Attribute)
                and c.func.attr == "none_means_default"
            )
            for c in calls
        )
        if validates and not resolves:
            missing.append(node.name)
    return missing


def _entries_in_package() -> tuple[list[tuple[str, str]], int]:
    missing, found = [], 0
    for path in sorted((REPO / "detail").rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        if "module.validate" not in source:
            continue
        rel = path.relative_to(REPO).as_posix()
        found += source.count("module.validate(") + source.count(
            "module.validate_batch("
        )
        missing += [(rel, name) for name in entries_without_it(source)]
    return missing, found


def test_every_config_entry_resolves_none_first():
    missing, _ = _entries_in_package()
    unexcused = [site for site in missing if site not in EXEMPT]
    assert not unexcused, (
        "these hand a config to a module's validate without "
        f"configs.defaults.none_means_default: {unexcused}"
    )


def test_every_exemption_still_excuses_something():
    missing, _ = _entries_in_package()
    stale = [site for site in EXEMPT if site not in missing]
    assert not stale, f"exempt entries that no longer need it: {stale}"


def test_scan_is_not_vacuous():
    _, found = _entries_in_package()
    assert found >= 7, f"only {found} module.validate calls found; the scan lost them"


@pytest.mark.parametrize(
    "source",
    [
        "def run(module, config):\n    return module.validate(d, config)\n",
        "def run(module, config):\n    return module.validate_batch(d, ids, config)\n",
        "class E:\n    def run(self, module, c):\n        x = module.validate(d, s, c)\n",
        # resolving in another function does not cover this one
        "def a(c):\n    return none_means_default(m, c)\n"
        "def b(module, c):\n    return module.validate(d, c)\n",
    ],
    ids=["validate", "validate_batch", "method", "other-function"],
)
def test_scan_catches_an_entry_without_it(source):
    assert entries_without_it(source)


@pytest.mark.parametrize(
    "source",
    [
        "def run(module, c):\n"
        "    return module.validate(d, none_means_default(module.config_model, c))\n",
        "def run(module, c):\n"
        "    c = defaults.none_means_default(module.config_model, c)\n"
        "    return module.validate_batch(d, ids, c)\n",
        "def run(module, c):\n    return module.build_job(c)\n",
    ],
    ids=["inline", "attribute", "no-validate"],
)
def test_scan_accepts_an_entry_with_it(source):
    assert entries_without_it(source) == []


def _accepts(field, value) -> None:
    annotation = field.annotation
    if field.metadata:
        annotation = typing.Annotated[(annotation, *field.metadata)]
    pydantic.TypeAdapter(annotation).validate_python(value)


def _modules():
    import_runner_modules(PIPELINE_RUNNERS)
    return [
        pytest.param(module.config_model, id=f"{kind}:{module_id}")
        for kind, registry in (
            ("processing", PROCESSING_MODULES),
            ("analysis", ANALYSIS_MODULES),
        )
        for module_id, module in sorted(registry.items())
        if module.config_model is not None
    ]


@pytest.mark.parametrize("model", _modules())
def test_none_becomes_the_default_in_every_module(model):
    every_none = {name: None for name in model.model_fields}
    resolved = none_means_default(model, every_none)
    for name, field in model.model_fields.items():
        if field.is_required():
            assert resolved[name] is None, f"{name} is required: no default to give"
            continue
        _accepts(field, resolved[name])
        if resolved[name] is not None:
            default = field.get_default(call_default_factory=True)
            assert resolved[name] == default, name


def test_a_none_a_field_can_hold_is_kept():
    resolved = none_means_default(
        PROCESSING_MODULES["tracking"].config_model,
        {"tracker_id": None, "arena_name": None, "class_name": None},
    )
    assert resolved == {
        "tracker_id": DEFAULT_TRACKER,
        "arena_name": None,
        "class_name": None,
    }


def test_an_omitted_setting_is_left_to_the_model():
    assert none_means_default(PROCESSING_MODULES["tracking"].config_model, {}) == {}


def test_run_processing_takes_a_none_tracker_as_the_default(tmp_path):
    import_runner_modules(PIPELINE_RUNNERS)
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    recorded, validated = proj._validate_processing(
        PROCESSING_MODULES["tracking"],
        ["recording_001"],
        {"detection_run_id": "det_01", "tracker_id": None},
    )
    assert recorded["tracker_id"] == DEFAULT_TRACKER  # what the run record holds
    assert validated["recording_001"]["tracker_id"] == DEFAULT_TRACKER


def test_retry_takes_a_none_tracker_as_the_default(tmp_path):
    import_runner_modules(PIPELINE_RUNNERS)
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    errors, validated = proj._validate_retry_recordings(
        PROCESSING_MODULES["tracking"],
        ["recording_001"],
        {"detection_run_id": "det_01", "tracker_id": None},
    )
    assert errors == {}
    assert validated["recording_001"]["tracker_id"] == DEFAULT_TRACKER


def test_a_recording_run_takes_a_none_tracker_as_the_default(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    _video_recording(proj, "recording_001")
    # Past the model, the next check is the detection run: reaching it shows None passed.
    with pytest.raises(ValueError, match="detection run 'det_09' not found"):
        proj.get_recording("recording_001").run(
            "tracking", {"detection_run_id": "det_09", "tracker_id": None}
        )


def test_run_analysis_takes_a_none_method_as_the_default(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    with pytest.raises(ValueError, match="at least one recording to search"):
        proj.run_analysis(
            "query",
            {"method": None, "embed_run_id": "emb_01", "source_recording_ids": []},
        )


def test_the_default_is_what_the_field_declares():
    """The substitution is the model's own default, never one invented here."""
    model = PROCESSING_MODULES["tracking"].config_model
    field = model.model_fields["tracker_id"]
    assert field.default is not PydanticUndefined
    assert none_means_default(model, {"tracker_id": None})["tracker_id"] == (
        field.default
    )
