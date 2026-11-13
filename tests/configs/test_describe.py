"""Tests for the parameter-docs generator (detail/configs/describe.py) — the single source
behind the Parameters reference page (§7)."""

from detail.configs.describe import (
    describe_config,
    iter_config_models,
    render_module_markdown,
    render_parameters_markdown,
    surfaced_config_models,
)


def _live_registries() -> dict[str, dict]:
    """Every registry dict in ``core/registries.py``, discovered rather than listed.

    Discovered on purpose: a hand-written list is the same failure one level up. A registry
    added to that module joins this walk the day it exists, without anyone remembering to.
    """
    from detail.core import registries

    return {
        name: value
        for name, value in vars(registries).items()
        if name.isupper() and isinstance(value, dict)
    }


def test_every_registry_declaring_config_models_reaches_the_parameter_reference():
    """A registry `iter_config_models` does not walk is undocumented *and* unguarded at once.

    `iter_config_models` is the one enumeration point for a run parameter: `detail help`, the
    generated module pages and Parameters table, the frames-as-a-unit scan, the advanced-form
    guard and the help-anchor map all read it. So a family it misses escapes all of them
    together — which is how `dtw`'s parameters reached a shipped default without ever meeting
    the unit-convention scan, and how the YOLO trainer's stayed out of the docs entirely.

    Both were found by review rather than by a test, because nothing anywhere compared the
    registries against the walk. This is that comparison.
    """
    assert not _unreachable_config_models(), (
        "these declare a config model that the parameter reference never sees, so their "
        "parameters are undocumented and every config guard skips them: "
        + ", ".join(sorted(_unreachable_config_models()))
    )


def _unreachable_config_models() -> list[str]:
    reachable = {module_id for module_id, _ in iter_config_models()}
    return [
        f"{registry_name}[{entry_id!r}]"
        for registry_name, registry in _live_registries().items()
        for entry_id, entry in sorted(registry.items())
        if getattr(entry, "config_model", None) is not None
        and entry_id not in reachable
    ]


def test_the_registry_parity_scan_would_catch_one():
    """Guard the guard, both halves.

    A walk over registries that happen to hold nothing passes vacuously forever, and a scan
    whose comparison is wrong reports nothing however much is missing. So: assert the
    registries are populated and really do declare config models, then put an unreachable one
    into a live registry and assert the scan *names* it.
    """
    populated = {
        name: registry for name, registry in _live_registries().items() if registry
    }
    assert (
        len(populated) >= 6
    ), f"registries look unpopulated: {sorted(_live_registries())}"
    declared = [
        entry_id
        for registry in populated.values()
        for entry_id, entry in registry.items()
        if getattr(entry, "config_model", None) is not None
    ]
    assert declared, "no registry entry declares a config model; the scan tests nothing"

    class _Unreachable:
        config_model = object()

    from detail.core import registries

    # TRACKERS is walked by the scan but not by `iter_config_models`, so an entry here is
    # exactly the shape the guard exists for. The registry fixture restores it after.
    registries.TRACKERS["not_in_the_reference"] = _Unreachable
    caught = _unreachable_config_models()
    assert any("not_in_the_reference" in entry for entry in caught), caught


def test_iter_covers_the_migrated_modules():
    ids = {module_id for module_id, _ in iter_config_models()}
    assert {
        "location_export",
        "detection",
        "embedding",
        "tracking",
        "labeling",  # §12 dataset source — not a module, listed explicitly
    } <= ids


def test_raw_runner_parameters_reach_the_reference():
    """A runner outside the module registries must still render here.

    Otherwise its parameters live only in hand-written CLI prose, which is exactly the drift the
    generated page exists to prevent.

    The example is ``comparison`` rather than ``cluster_fit``: the clusterer chain was hidden
    (§25) when this was written, and has since been removed; a hidden capability is the one
    thing that must *not* render here, so using it as the example asserted the opposite of the
    rule the moment its surfacing was declared correctly. ``comparison`` is the same shape — its
    own store, its own CLI group, no module registry — and is surfaced.
    """
    md = render_parameters_markdown()
    assert "## Comparison" in md
    # a knob that only exists on the comparison config, so it cannot come from another module
    assert "`aggregate_to`" in md and "`draws`" in md


def test_describe_config_shape():
    from detail.configs.detection import DetectionConfig

    rows = describe_config(DetectionConfig)
    every_n = next(r for r in rows if r["name"] == "every_n_frames")
    assert every_n["label"] == "Use every"
    assert every_n["type"] == "integer"
    assert every_n["minimum"] == 1
    assert every_n["default"] == 1
    assert not every_n["required"]


def test_an_optional_literal_keeps_its_choices():
    """``Literal[...] | None`` puts the enum inside ``anyOf``; reading only the top-level
    ``enum`` reported no choices, so the docs table, the CLI and the form lost the options.
    """
    from typing import Literal

    from pydantic import BaseModel, Field

    class Knob(BaseModel):
        plain: Literal["a", "b"] = Field(
            "a", description="d", json_schema_extra={"label": "P"}
        )
        maybe: Literal["x", "y"] | None = Field(
            None, description="d", json_schema_extra={"label": "M"}
        )

    rows = {r["name"]: r for r in describe_config(Knob)}
    assert rows["plain"]["choices"] == ["a", "b"] and not rows["plain"]["optional"]
    assert rows["maybe"]["choices"] == ["x", "y"] and rows["maybe"]["optional"]


def test_a_number_that_may_also_be_one_word_is_not_a_choice():
    """``int | Literal["auto"]`` is a free number that may also be one word. Reading the
    ``const`` branch alone makes it a one-item choice: the form becomes a combo of ``"auto"``,
    and a saved value of 3 or ``"auto"`` can no longer load.
    """
    from typing import Literal

    from pydantic import BaseModel, Field

    class Knob(BaseModel):
        series: int | Literal["auto"] = Field(
            0, description="d", json_schema_extra={"label": "S"}
        )
        either: Literal["a"] | Literal["b"] | None = Field(
            None, description="d", json_schema_extra={"label": "E"}
        )

    rows = {r["name"]: r for r in describe_config(Knob)}
    assert rows["series"]["choices"] is None
    assert rows["either"]["choices"] == ["a", "b"]


def test_every_documented_param_has_help_and_label():
    # The drift guard: a config-model field with no description or no label renders a blank cell
    # in the reference (and a missing GUI tooltip / CLI help). Fail loudly with the offenders.
    missing = []
    for module_id, model_cls in iter_config_models():
        for row in describe_config(model_cls):
            if not row["help"]:
                missing.append(f"{module_id}.{row['name']}: no description")
            if row["label"] == row["name"]:
                missing.append(f"{module_id}.{row['name']}: no label")
    assert not missing, missing


def test_render_includes_each_module():
    """The page covers every module this build *surfaces* — and only those.

    The set is `surfaced_config_models`, not `iter_config_models`: this is the user-facing
    parameter reference, one of §25's docs enumeration points, so a module the build hides must
    be absent from it. The inverse half is asserted too, or the filter would be guarded
    vacuously the moment nothing is hidden.
    """
    md = render_parameters_markdown()
    assert md.startswith("# Parameters")

    def heading(module_id):
        return f"## {module_id.replace('_', ' ').capitalize()}"

    # Whole lines, never `in md`: one id being a prefix of another is enough to break the
    # containment form, and `dtw` is a prefix of `dtw_finch`. Read as a substring, a hidden
    # `## Dtw` (the query method, hidden until it was removed) is "present" because the
    # surfaced `## Dtw finch` contains it — so the absence half of this test passed only while no
    # hidden id prefixed a surfaced one.
    headings = {line for line in md.splitlines() if line.startswith("## ")}

    surfaced = {module_id for module_id, _ in surfaced_config_models()}
    for module_id in surfaced:
        assert heading(module_id) in headings
    for module_id, _ in iter_config_models():
        if module_id not in surfaced:
            assert (
                heading(module_id) not in headings
            ), f"{module_id} is hidden by §25 but the parameter reference still documents it"
    # a known parameter row renders with its bound
    assert "Confidence threshold (`conf_threshold`)" in md
    assert "0–1" in md


def test_field_descriptions_use_no_em_dash():
    # Field descriptions are the single source rendered to the module pages, CLI --help, and
    # GUI tooltips. The house voice bans em dashes (docs_design.md §5); a comma / colon / two
    # sentences instead. This guard fails CI on a regression.
    offenders = [
        f"{module_id}.{p['name']}"
        for module_id, model_cls in iter_config_models()
        for p in describe_config(model_cls)
        if "—" in (p["help"] or "")
    ]
    assert not offenders, "field descriptions with an em dash: " + ", ".join(offenders)


def test_module_page_anchors_every_field_by_name():
    # Each module page gives every field an explicit `### <label> {#<name>}` heading, so the
    # in-app help can deep-link a field as modules/<id>#<name> with no slug math.
    missing = []
    for module_id, model_cls in iter_config_models():
        md = render_module_markdown(module_id, model_cls)
        for p in describe_config(model_cls):
            if f"### {p['label']} {{#{p['name']}}}" not in md:
                missing.append(f"{module_id}.{p['name']}")
    assert not missing, "fields with no anchored heading: " + ", ".join(missing)


def test_module_intro_is_prepended_under_the_title():
    md = render_module_markdown(
        "detection",
        dict(iter_config_models())["detection"],
        intro="A one-line intro.",
    )
    title_at = md.index("# Detection")
    intro_at = md.index("A one-line intro.")
    params_at = md.index("## Parameters")
    assert title_at < intro_at < params_at


def test_every_tracker_hyperparam_is_documented_and_anchored():
    # Tracker knobs are not pydantic fields; their label + help live in
    # configs/tracker_params.py and the tracking page emits an anchor per knob. Every
    # knob any tracker exposes must have a doc entry that resolves, so a field ? on a
    # dynamically-built tracker row deep-links its definition.
    from detail import help
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.configs.tracker_params import TRACKER_HYPERPARAM_DOCS
    from detail.core.registries import TRACKERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    missing = []
    for tracker_id, cls in TRACKERS.items():
        for name in getattr(cls, "HYPERPARAMS_SCHEMA", {}):
            if name not in TRACKER_HYPERPARAM_DOCS or not help.anchor_exists(
                "modules/tracking", name
            ):
                missing.append(f"{tracker_id}.{name}")
    assert not missing, "tracker knobs with no resolving doc entry: " + ", ".join(
        missing
    )
