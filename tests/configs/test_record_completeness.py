"""A knob that changed a result must reach the record that claims to describe it.

Tenet 9: *"A run must be fully reproducible from its own metadata alone."* The tenet was
written, was read, and did not prevent any of this — because nothing checked it. Analysis
modules (§16) arrived after it and never inherited the discipline that processing runs have.

The concrete failures this pins:

- ``cluster_eval`` hand-listed six keys into ``record["params"]``. ``stream_name`` was added
  later and never joined them, so **every one of the 230 analysis runs on disk omits the
  ground-truth vocabulary it scored against** — including runs that passed one explicitly.
- ``query_eval`` had the same bug in ``build_job`` once ("the bug that made a recorded run
  ignore its matcher") and fixed it by carrying fields *by name* instead of by hand. That fix
  never propagated one file over. This test is what makes it propagate.

Both checks are static — they read source and field names, never call ``validate()`` and need
no experiment on disk. That matters: ``validate()`` has two different signatures across the two
registries, ``derive`` (since removed) had none at all, and each needs a fully populated
experiment to get past its first line. A guard that needed all that would not be written, and an
unwritten guard protects nothing.

The allow-lists are claims, not exemptions: each entry asserts the field cannot change a
result. They are checked against the live field set, so a stale name cannot excuse a new field
(the idiom from ``tests/test_cli_cluster_parity.py``).
"""

import inspect
from dataclasses import fields as dataclass_fields

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.executor.runners import (
    ANALYSIS_MODULES,
    PROCESSING_MODULES,
    import_runner_modules,
)

import_runner_modules(PIPELINE_RUNNERS)


# A config field reaches the job under its own name, or under a renaming the codebase applies
# systematically. Two such conventions exist and are recognized here rather than needing an
# allow-list entry per field:
#
#   `x_s` -> `x_ns`   a temporal knob declared in seconds and converted at the boundary that
#                     knows the cadence (the convention PR #141 established)
#   `x_id` -> `x_ids` a single id resolved to the per-recording map it stands for
#
# Anything else needs an entry below, which is a *claim* that the field cannot change a result.
def _reaches(field: str, job_fields: set[str]) -> bool:
    if field in job_fields:
        return True
    if field.endswith("_s") and f"{field[:-2]}_ns" in job_fields:
        return True
    if field.endswith("_id") and f"{field}s" in job_fields:
        return True
    return False


_NOT_PART_OF_THE_RESULT: dict[str, dict[str, str]] = {
    "query_eval": {
        "source_recording_ids": "resolved to the eligible set; recorded as recording_ids",
    },
    "location_export": {
        "source_recording_ids": "recorded as the record's recording_ids",
    },
}


def _module_ids(registry):
    return sorted(registry)


@pytest.mark.parametrize("module_id", _module_ids(ANALYSIS_MODULES))
def test_every_config_field_reaches_the_job(module_id):
    """A field the job drops cannot reach the record, the worker, or a re-run."""
    module = ANALYSIS_MODULES[module_id]
    model = getattr(module, "config_model", None)
    job_class = getattr(module, "job_class", None)
    if model is None or job_class is None:
        pytest.skip(f"{module_id} declares no config model or job class")

    job_fields = {f.name for f in dataclass_fields(job_class)}
    exempt = set(_NOT_PART_OF_THE_RESULT.get(module_id, {}))
    missing = sorted(
        name
        for name in model.model_fields
        if name not in exempt and not _reaches(name, job_fields)
    )
    assert not missing, (
        f"{module_id}: these config fields never reach {job_class.__name__}, so a run using "
        f"them cannot be reproduced from its record: {missing}"
    )


@pytest.mark.parametrize("module_id", _module_ids(ANALYSIS_MODULES))
def test_the_record_params_are_derived_from_the_job_not_hand_listed(module_id):
    """A hand-written ``params`` dict drifts from the job the moment a field is added.

    It is not enough for the fields to line up *today* — a literal that happens to be complete
    is one config parameter away from being silently incomplete, and nothing fails when it
    becomes so. So this asserts the *mechanism*: params comes from the job.

    **And that a record has params at all**, which this asked about for a long time without
    checking. A module writing none of them *skipped* here — the guard tested only *how* a
    module writes params, never *whether* it does, so the one state it most needed to catch was
    the one it read as "nothing to say". Four of nine analysis modules were in it, including
    both clustering modules: `an_17`, a real behavior-clustering run, records its recordings and
    its artifacts and not one number it was launched with. Nothing fails — the record is
    well-formed, the run completes, the result opens — and Tenet 9 ("fully reproducible from its
    own metadata alone") is simply not true of it. It is also what the GUI's Refine reads, so
    those runs re-opened at defaults with no way to tell.
    """
    module = ANALYSIS_MODULES[module_id]
    source = inspect.getsource(type(module))
    assert 'record["params"]' in source, (
        f"{module_id} never writes record['params'], so its runs record nothing they were "
        "launched with — not reproducible from their own metadata (Tenet 9), and nothing for "
        "Refine to restore. Write `record['params'] = job_params(job)` where the record is "
        "written."
    )

    assignment = source.split('record["params"]')[1].split("\n\n")[0]
    derived = (
        "job_params(" in assignment
        or "to_dict()" in assignment
        or "_job_config(" in assignment
    )
    assert derived, (
        f"{module_id} hand-writes record['params']. Derive it from the job — "
        "`runners.job_params(job)` — a hand-listed dict is how cluster_eval lost "
        "`stream_name` for every run ever recorded."
    )


@pytest.mark.parametrize("module_id", sorted(_NOT_PART_OF_THE_RESULT))
def test_the_allow_lists_do_not_excuse_fields_that_no_longer_exist(module_id):
    module = ANALYSIS_MODULES.get(module_id)
    assert (
        module is not None
    ), f"allow-list names a module that no longer exists: {module_id}"
    model = getattr(module, "config_model", None)
    if model is None:
        pytest.skip(f"{module_id} declares no config model")
    stale = sorted(set(_NOT_PART_OF_THE_RESULT[module_id]) - set(model.model_fields))
    assert (
        not stale
    ), f"{module_id}: allow-list entries for fields that no longer exist: {stale}"


def test_the_registries_are_populated():
    """A data-driven guard over an empty registry passes by testing nothing."""
    assert len(ANALYSIS_MODULES) >= 4, ANALYSIS_MODULES
    assert len(PROCESSING_MODULES) >= 4, PROCESSING_MODULES


@pytest.mark.parametrize("module_id", _module_ids(ANALYSIS_MODULES))
def test_every_module_declares_its_phases(module_id):
    """Issue #393: a run page can only show what phase it is in — and what is still ahead — if
    the module says what its phases are ahead of time. Nothing else can: the phase names exist
    only as the ``message`` on a live progress event, so a module that declares none leaves the
    GUI unable to draw anything but the current line of text, forever.

    ``phases`` is required (no default on ``AnalysisModule``) for the same reason
    ``record["params"]`` above is checked for existing at all, not just for being derived
    correctly: a defaulted-empty field reads identically whether a module has one phase or the
    author simply never got to it, and nothing fails either way.
    """
    module = ANALYSIS_MODULES[module_id]
    phases = getattr(module, "phases", None)
    assert phases, (
        f"{module_id} declares no phases — a run page cannot show which phase it is in or what "
        "is still ahead (§16, issue #393). Add `phases = (...)`, in order, naming the steps the "
        "runner's own progress messages already advance through."
    )
    assert all(
        isinstance(p, str) and p for p in phases
    ), f"{module_id}.phases contains a non-string or empty entry: {phases}"


def test_the_phases_guard_would_catch_a_module_with_none_declared(monkeypatch):
    """Guard the guard: a module with an empty ``phases`` must fail, not skip — the state four
    of nine modules were already in for `record["params"]` before that omission got its own test
    above (`test_the_guard_would_catch_a_module_that_records_no_params_at_all`)."""
    from detail.pipeline import temporal_clustering as tc

    monkeypatch.setattr(tc.TemporalClusteringModule, "phases", ())
    with pytest.raises(AssertionError, match="declares no phases"):
        test_every_module_declares_its_phases(tc.TemporalClusteringModule.module_id)


def test_the_guard_would_catch_a_dropped_field(monkeypatch):
    """Guard the guard: with `stream_name` gone from the job, the check must fail — that is
    the exact shape of the bug it exists for."""
    from detail.pipeline.query_eval import QueryEvalJob

    kept = tuple(f for f in dataclass_fields(QueryEvalJob) if f.name != "stream_name")
    monkeypatch.setattr(
        "tests.configs.test_record_completeness.dataclass_fields",
        lambda cls: kept if cls is QueryEvalJob else dataclass_fields(cls),
    )
    with pytest.raises(AssertionError, match="stream_name"):
        test_every_config_field_reaches_the_job("query_eval")


def test_the_guard_would_catch_a_module_that_records_no_params_at_all(monkeypatch):
    """Guard the guard, for the state it used to *skip*.

    This is the more valuable half. A module writing a hand-listed dict is caught by the
    assertion below it; a module writing nothing was read as "nothing to check" and passed
    silently, which is how four of nine got here. So the empty case gets its own mutation:
    hide the assignment from the source the check reads, and the check must fail.
    """
    from detail.pipeline import temporal_clustering as tc

    real = inspect.getsource

    def without_params(obj):
        return real(obj).replace('record["params"] = job_params(job)', "pass")

    monkeypatch.setattr(
        "tests.configs.test_record_completeness.inspect.getsource", without_params
    )
    with pytest.raises(AssertionError, match="never writes"):
        test_the_record_params_are_derived_from_the_job_not_hand_listed(
            tc.TemporalClusteringModule.module_id
        )


# --- processing modules: the config must reach the *manifest entry*, not just the run ---- #
#
# A processing run writes its recipe twice: into its own output directory, and into the
# recording's `embedding_runs` / `detection_runs` manifest entry via `extra_record_fields`.
# Only the second is what a comparison reads (`api/comparison._embedding_config`), so a knob
# recorded in the first and dropped from the second is invisible exactly where it decides
# whether two runs are comparable — `canvas_scope` / `bg_color_scope` were written to
# `embedding.yaml` and omitted here, which silently disarmed the held-constant check for
# them. Same lesson as the analysis-module guard above, one registry over.

_NOT_RECIPE_IDENTITY: dict[str, dict[str, str]] = {
    "embedding": {
        "all_tracks": "resolved into the recorded track_ids",
        "batch_size": "throughput only — does not change the stored vector",
    },
    "keypoint_detect": {
        "batch_size": "throughput only — does not change a placed point",
    },
    "keypoint_track": {
        "match_min_inside": "recorded inside match (min_inside)",
        "match_mask_grow": "recorded inside match (mask_grow)",
    },
}


def _synthetic_config(model, module_id: str) -> dict:
    """A config dict with every model field at its default, plus the non-model keys the
    runner reads. Static enough to need no experiment on disk (the idiom above)."""
    config = {name: field.default for name, field in model.model_fields.items()}
    config.update({"track_ids": [0], "every_n_frames": 1})
    for name in ("tracking_run_id", "detection_run_id", "model_id"):
        config.setdefault(name, "x")
    return config


@pytest.mark.parametrize("module_id", _module_ids(PROCESSING_MODULES))
def test_every_config_field_reaches_the_recording_manifest(module_id):
    module = PROCESSING_MODULES[module_id]
    model = getattr(module, "config_model", None)
    if model is None or not hasattr(module, "extra_record_fields"):
        pytest.skip(f"{module_id} declares no config model or extra_record_fields")

    recorded = set(module.extra_record_fields(_synthetic_config(model, module_id)))
    exempt = set(_NOT_RECIPE_IDENTITY.get(module_id, {}))
    missing = sorted(
        name
        for name in model.model_fields
        if name not in exempt and name not in recorded
    )
    assert not missing, (
        f"{module_id}: these config fields never reach the recording manifest entry, so a "
        f"comparison cannot tell two runs apart on them: {missing}"
    )


@pytest.mark.parametrize("module_id", sorted(_NOT_RECIPE_IDENTITY))
def test_the_processing_allow_lists_are_not_stale(module_id):
    module = PROCESSING_MODULES.get(module_id)
    assert module is not None, f"allow-list names a module that is gone: {module_id}"
    model = getattr(module, "config_model", None)
    if model is None:
        pytest.skip(f"{module_id} declares no config model")
    stale = sorted(set(_NOT_RECIPE_IDENTITY[module_id]) - set(model.model_fields))
    assert not stale, f"{module_id}: allow-list entries for gone fields: {stale}"


def test_the_manifest_guard_would_catch_a_dropped_scope(monkeypatch):
    """Guard the guard: drop `canvas_scope` back out of the record and the check must fail."""
    module = PROCESSING_MODULES["embedding"]
    original = module.extra_record_fields
    monkeypatch.setattr(
        type(module),
        "extra_record_fields",
        lambda self, config: {
            k: v for k, v in original(config).items() if k != "canvas_scope"
        },
    )
    with pytest.raises(AssertionError, match="canvas_scope"):
        test_every_config_field_reaches_the_recording_manifest("embedding")
