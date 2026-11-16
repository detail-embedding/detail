"""describe_config — read a module's pydantic config model into per-parameter docs (§7).

The one place the generated parameter reference (`docs/reference/parameters.md`, built by
`scripts/gen_docs_parameters.py`) renders from; the GUI tooltips and CLI ``--help`` read the same field
metadata (via `gui/_pydantic_form.py` and `Model.model_fields`). Pydantic-only — the
registry walk imports the pipeline lazily so importing this module stays cheap."""

from pydantic import BaseModel

from detail.configs.comparison import ComparisonConfig
from detail.configs.keypoint_track import DespikeSmoothConfig, TemporalDecodeConfig
from detail.configs.keypoint_dataset import KeypointDatasetConfig
from detail.configs.labeling import LabelingConfig

# Runners registered with ``register_runner`` rather than as a Processing/Analysis module — the §12
# dataset sources (labeling, keypoint_dataset) — and a few other configs with no module registry
# still declare config models, but the walk below cannot find them. List them here so their
# parameters reach the reference; a hand-written parameter table in the CLI docs is the bug this
# avoids. (query_eval is a registered §16 AnalysisModule, so the registry walk covers it.)
#
# The third element is **the surface that launches it**, and it is required. A module registry
# member is reached by its own id, so §25 hiding it is enough; these have no registry, so their
# only way in is somebody else's command group or GUI section — and `surfaced_config_models`
# asked only whether the *module id* was hidden. The removed clusterer fit and apply
# (`docs/removed_capabilities.md`) were launched solely by the hidden `clusterer` CLI group and
# `clusterers` GUI section, and neither id was named in §25's entry: so both answered
# `is_surfaced` with True and published a parameter page each to the user site for a capability
# nobody could reach. Derived from the way in, rather than left to be remembered a second time.
_UNREGISTERED_CONFIG_MODELS: list[tuple[str, type[BaseModel], str]] = [
    ("labeling", LabelingConfig, "distill"),
    # The keypoint twin of labeling (§12 *Distill keypoints*): the label stage of a keypoint
    # distill, launched by `keypoints distill` / *Distill new model* (Keypoints), with no module registry.
    ("keypoint_dataset", KeypointDatasetConfig, "distill"),
    # A comparison (§24) is not an AnalysisModule -- it has its own store and its own CLI group --
    # so nothing walked it, and its nine parameters reached neither the generated reference nor
    # any of the guards that read this list.
    ("comparison", ComparisonConfig, "comparison"),
    # The keypoint track *methods* (§26 "Track methods"): each has its own config model, every
    # field of it identity, reached through `keypoint_track`'s `method` — a parameter page per
    # method, so the GUI's `?` on a method knob has a page.
    ("temporal_decode", TemporalDecodeConfig, "keypoint_track"),
    ("despike_smooth", DespikeSmoothConfig, "keypoint_track"),
]

#: ``module_id -> the surface that launches it``, for the modules with no registry of their own.
LAUNCHED_BY: dict[str, str] = {
    module_id: surface for module_id, _model, surface in _UNREGISTERED_CONFIG_MODELS
}


def _nonnull_type(prop: dict) -> str:
    if prop.get("type"):
        return prop["type"]
    for branch in prop.get("anyOf", []):  # X | None → the non-null branch's type
        if branch.get("type") and branch["type"] != "null":
            return branch["type"]
    return ""


def _bounds(prop: dict) -> tuple:
    if "minimum" in prop or "maximum" in prop:
        return prop.get("minimum"), prop.get("maximum")
    for branch in prop.get("anyOf", []):
        if branch.get("type") in ("integer", "number"):
            return branch.get("minimum"), branch.get("maximum")
    return None, None


def _exclusive_bounds(prop: dict) -> tuple:
    """``gt`` / ``lt`` bounds, which land in the schema under their own keys.

    Reported separately from the inclusive pair rather than folded into it: a spin box has
    to step *inside* an exclusive bound (``val_split`` is ``lt=1``, so the box stops at 0.9)
    and the docs have to render it as ``< 1``, not ``1``."""
    if "exclusiveMinimum" in prop or "exclusiveMaximum" in prop:
        return prop.get("exclusiveMinimum"), prop.get("exclusiveMaximum")
    for branch in prop.get("anyOf", []):
        if branch.get("type") in ("integer", "number"):
            return branch.get("exclusiveMinimum"), branch.get("exclusiveMaximum")
    return None, None


def _choices(prop: dict) -> list | None:
    """The values a ``Literal[...]`` field offers, or None for a free field.

    Three JSON-schema spellings of one shape: ``enum`` (two or more values), ``const`` (a
    ``Literal`` with one value — pydantic 2 emits no ``enum`` for it), and either of those
    inside the ``anyOf`` a ``Literal[...] | None`` produces. Reading only the top-level
    ``enum`` lost the options of the other two, so the docs table, the CLI and the form saw
    a free string.
    """
    if "enum" in prop:
        return list(prop["enum"])
    if "const" in prop:
        return [prop["const"]]
    for branch in prop.get("anyOf", []):
        if "enum" in branch:
            return list(branch["enum"])
        if "const" in branch:
            return [branch["const"]]
    return None


def _optional(prop: dict) -> bool:
    """Whether the field accepts ``None`` — an ``X | None`` with a null branch."""
    return any(branch.get("type") == "null" for branch in prop.get("anyOf", []))


def describe_config(model_cls: type[BaseModel]) -> list[dict]:
    """One dict per field: name, label, help, type, required, optional, default, minimum,
    maximum, unit, ui, ui_max, advanced, scope.

    ``ui`` / ``ui_max`` are the presentation hints the GUI reads (`gui/spin.py` kinds, and a
    spin ceiling for a field the model leaves unbounded). ``advanced`` is the third: ``True``
    puts the field under the form's collapsible "Advanced settings"
    (`gui/forms.py::ConfigForm`), and a string names its own section, so a form with a lot of
    knobs can split them ("Segmentation" / "Clustering"). All three are presentation only —
    they never change what the model enforces, so the CLI and validator are unaffected.

    ``scope`` is not presentation: it says whether a field is a **setting** (reusable) or a
    **selection** (names an artifact in this experiment, so it never travels between runs).
    """
    schema = model_cls.model_json_schema()
    required = set(schema.get("required", []))
    params = []
    for name, prop in schema.get("properties", {}).items():
        lo, hi = _bounds(prop)
        exclusive_lo, exclusive_hi = _exclusive_bounds(prop)
        params.append(
            {
                "name": name,
                "label": prop.get("label", name),
                "help": prop.get("description", ""),
                "type": _nonnull_type(prop),
                "required": name in required,
                "optional": _optional(prop),
                "default": prop.get("default"),
                "minimum": lo,
                "maximum": hi,
                "exclusive_minimum": exclusive_lo,
                "exclusive_maximum": exclusive_hi,
                "unit": prop.get("unit"),
                "ui": prop.get("ui"),
                "ui_max": prop.get("ui_max"),
                "ui_step": prop.get("ui_step"),
                "advanced": prop.get("advanced", False),
                # Whether this field is a *setting* (a knob, reusable anywhere) or a
                # *selection* (it names a specific artifact in this experiment). A run config
                # is both wearing one dict, and only the first half can be carried from one
                # run to the next: `embed_run_id` reused across experiments is meaningless, or
                # — worse — names something real and different. Declared on the field so a new
                # one is classified by whoever adds it; a denylist in the consumer is the drift
                # bug `job_params` and `test_record_completeness` already exist to prevent.
                "scope": prop.get("scope", "setting"),
                # A `Literal[...]`'s allowed values, or None for a field that is not a
                # choice. Read off the schema's `enum` rather than `get_args` on the
                # annotation, so this is the same single source as every other key here —
                # a form calling `get_args` on a model field itself (the query-eval form's
                # `_FUSIONS`) is the retyping this table exists to remove.
                "choices": _choices(prop),
            }
        )
    return params


#: How a run's summary reads when nothing was changed. Not an empty string: a blank line beside
#: two described ones reads as a failure to describe, rather than as "this one is the default".
NO_CHANGES = "defaults"


def changed_from_defaults(model_cls: type[BaseModel], config: dict) -> dict:
    """The **settings** in ``config`` whose value differs from the model's own default.

    What distinguishes one previous run from another, and therefore what a picker shows.
    Derived rather than written, because a hand-listed per-module summary is complete on the day
    it is written and silently stale afterwards — the shape ``modules/_common.run_label``'s
    ``if module == "detection" ... elif "tracking"`` chain already has, and that CLAUDE.md
    records as a class ("a shared surface summarizing a module-shaped fact by hardcoding one
    module's shape").

    **Selections are excluded.** They are not carried from one run to the next, so they cannot
    tell two starting points apart — and naming one invites the reader to expect it to come
    along.

    **The whole config is walked, not just the model's fields.** A clustering run's real knobs —
    the detector, the window, the penalty — live in ``method_params``, which is deliberately
    *not* a model field (``configs/method_params.py``: each method owns and validates its own
    set). A summary built from ``model_fields`` alone would report every temporal-clustering run
    as ``defaults``, which is exactly the case the picker exists to tell apart.
    """
    fields = {f["name"]: f for f in describe_config(model_cls)}
    found: list[tuple[int, str, object]] = []
    for name, value in config.items():
        field = fields.get(name)
        if field is not None and field["scope"] != "setting":
            continue
        if isinstance(value, dict):
            found.extend(_changed_nested(name, value, config))
            continue
        if field is None:
            continue  # a non-model scalar has no default to differ from
        if not _same_setting(value, field["default"]):
            found.append((_rank(field), name, value))
    # Ordered, because a caller shows the first few and the rest as a count. Alphabetical put
    # the scope knobs (`canvas_scope`, then also `bg_color_scope`) first and hid `model_id` /
    # `model_variant` behind "(+4 more)" — four embedding runs on four different models, four
    # identical labels. The field's own `advanced` flag is the ranking: a knob the form
    # shows up front is a knob the user chose deliberately, which is exactly what tells two
    # runs apart.
    found.sort(key=lambda item: (item[0], item[1]))
    return {name: value for _rank_, name, value in found}


def _rank(field: dict) -> int:
    """0 for a field the form shows up front, 1 for one under Advanced settings."""
    return 1 if field["advanced"] else 0


def _same_setting(value, default) -> bool:
    """Whether a stored value is the default, treating every "nothing" as one thing.

    A form that emits ``[]`` for a field defaulting to ``None`` has not changed it, and
    ``arena_ids=0 item(s)`` in a summary of what makes a run distinctive is worse than silence:
    it spends the line on the absence of a choice.
    """
    empty = (None, [], {}, "", ())
    if value in empty and default in empty:
        return True
    return value == default


def _changed_nested(
    name: str, value: dict, config: dict
) -> list[tuple[int, str, object]]:
    """A dict-valued config key's knobs that differ from the defaults of whatever owns them.

    Ranked like the model's own fields, from the owning model's ``advanced`` flags where it has
    them. A tracker declares a plain ``{name: (type, default)}`` schema with no such flag, so
    its knobs rank up front — which is right: a tracker has few, and they are the reason one
    tracking run differs from another.
    """
    defaults, ranks = _owner_defaults(value, config)
    if defaults is None:
        # Nothing in this config claims these keys, so there is nothing to compare them
        # against — and the dicts in this position are the ones a picker does not carry
        # anyway: `compute` is a property of the machine, `detection_spec` names an upstream
        # run. Reporting `compute=4 set` spends the line on something the pick will not
        # restore, which is worse than silence.
        return []
    return [
        (ranks.get(key, 0), key, item)
        for key, item in value.items()
        if not _same_setting(item, defaults[key])
    ]


def _owner_defaults(params: dict, config: dict) -> tuple[dict | None, dict]:
    """``{param: default}`` of the registered thing whose parameter set is ``params``' keys.

    **Discovered, not declared**, because there is nowhere to declare it: ``method_params`` and
    ``hyperparams`` are not model fields, so they have no ``json_schema_extra`` to carry a
    pointer, and a table here keyed by field name is the hardcoding this module avoids
    elsewhere. Instead every string in the same config is tried as a registry id, and an owner
    counts only if it declares **every** key in ``params`` — a method's parameter list is
    specific enough that covering all of them identifies it.

    Fails to ``None`` rather than raising: an old run naming a method since removed is ordinary,
    and a label that cannot be built is better empty than raised into a dropdown.
    """
    if not params:
        return None, {}
    for value in config.values():
        if not isinstance(value, str):
            continue
        defaults, ranks = _registered_defaults(value)
        if defaults is not None and set(params) <= set(defaults):
            return defaults, ranks
    return None, {}


def _registered_defaults(thing_id: str) -> tuple[dict | None, dict]:
    """``({param: default}, {param: rank})`` for a registered method or tracker.

    ``(None, {})`` when the id names nothing registered.
    """
    from detail.cluster_methods import (
        import_pose_cluster_methods,
        import_temporal_cluster_methods,
    )
    from detail.core.registries import (
        POSE_CLUSTER_METHODS,
        QUERY_METHODS,
        TEMPORAL_CLUSTER_METHODS,
        TRACKERS,
    )
    from detail.query_methods import import_query_methods

    import_pose_cluster_methods()
    import_temporal_cluster_methods()
    import_query_methods()
    # Trackers register on import of their own modules; there is no `import_*` helper,
    # and `pipeline.tracking` is the module that pulls them all in.
    import detail.pipeline.tracking  # noqa: F401

    for registry in (
        TEMPORAL_CLUSTER_METHODS,
        POSE_CLUSTER_METHODS,
        QUERY_METHODS,
        TRACKERS,
    ):
        owner = registry.get(thing_id)
        if owner is None:
            continue
        model_cls = getattr(owner, "config_model", None)
        if model_cls is not None:
            described = describe_config(model_cls)
            return (
                {f["name"]: f["default"] for f in described},
                {f["name"]: _rank(f) for f in described},
            )
        # A tracker declares `{name: (type, default)}` rather than a pydantic model.
        schema = getattr(owner, "HYPERPARAMS_SCHEMA", None)
        if schema:
            return {name: default for name, (_type, default) in schema.items()}, {}
    return None, {}


def _short(value) -> str:
    """One value, short enough for a dropdown line."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return f"{len(value)} item(s)"
    text = str(value)
    return text if len(text) <= 24 else text[:21] + "…"


def summarize_settings(
    model_cls: type[BaseModel], config: dict, *, limit: int = 3
) -> str:
    """``"window_s=1.5 · penalty=3 (+2 more)"`` — what makes this config worth picking.

    ``limit`` caps the line rather than the information: the count of what was dropped stays,
    so a run with many changes never looks like a run with three.
    """
    changed = changed_from_defaults(model_cls, config)
    if not changed:
        return NO_CHANGES
    # `changed` is already ordered most-distinctive first; re-sorting here would undo that.
    names = list(changed)
    shown = " · ".join(f"{name}={_short(changed[name])}" for name in names[:limit])
    extra = len(names) - limit
    return f"{shown} (+{extra} more)" if extra > 0 else shown


def iter_config_models() -> list[tuple[str, type[BaseModel]]]:
    """`(module_id, config_model)` for every registered thing that declares one — processing
    then analysis modules, the runners outside those registries (§12 dataset sources, a
    comparison, the keypoint track methods), the §27 tools, the §20 query methods, and the §12 trainers. Each group
    sorted by id, so the output is stable. Imports what it needs to populate the registries.

    **Every registry whose members can declare a `config_model` is walked here.** This is the
    one enumeration point for a parameter: `detail help`, the generated docs pages, the
    unit-convention scan, the advanced-form guard and the help-anchor map all read it, so a
    registry left out is a family of parameters undocumented *and* unguarded at once — which
    is how `min_separation_rows` reached a default without ever meeting the frames-as-a-unit
    scan. `tests/configs/test_describe.py` fails when a registry grows a config model this
    does not reach.
    """
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.core.registries import (
        POSE_CLUSTER_METHODS,
        QUERY_METHODS,
        STREAM_IMPORTERS,
        TEMPORAL_CLUSTER_METHODS,
        TOOLS,
        TRAINERS,
    )
    from detail.executor.runners import (
        ANALYSIS_MODULES,
        PROCESSING_MODULES,
        import_runner_modules,
    )
    from detail.cluster_methods import (
        import_pose_cluster_methods,
        import_temporal_cluster_methods,
    )
    from detail.importers import import_stream_importers
    from detail.query_methods import import_query_methods
    from detail.tools import import_tools
    from detail.trainers import import_trainers

    import_runner_modules(PIPELINE_RUNNERS)
    import_tools()
    import_query_methods()
    import_pose_cluster_methods()
    import_temporal_cluster_methods()
    import_trainers()
    import_stream_importers()
    pairs: list[tuple[str, type[BaseModel]]] = []
    for registry in (PROCESSING_MODULES, ANALYSIS_MODULES):
        for module_id, module in sorted(registry.items()):
            if getattr(module, "config_model", None) is not None:
                pairs.append((module_id, module.config_model))
    pairs.extend(
        (module_id, model) for module_id, model, _surface in _UNREGISTERED_CONFIG_MODELS
    )
    for registry in (
        TOOLS,
        QUERY_METHODS,
        POSE_CLUSTER_METHODS,
        TEMPORAL_CLUSTER_METHODS,
        TRAINERS,
        STREAM_IMPORTERS,
    ):
        for entry_id, entry in sorted(registry.items()):
            if getattr(entry, "config_model", None) is not None:
                pairs.append((entry_id, entry.config_model))
    return pairs


def _num(x):
    return int(x) if isinstance(x, float) and x.is_integer() else x


def _range_str(p: dict) -> str:
    """``0–1``, ``0–<1``, ``≥ 2``, ``< 1`` … — an exclusive bound keeps its ``<`` / ``>`` so
    the page never reads as if the endpoint were allowed."""
    lo = _num(p["minimum"]) if p["minimum"] is not None else None
    hi = _num(p["maximum"]) if p["maximum"] is not None else None
    exclusive_lo = (
        _num(p["exclusive_minimum"]) if p["exclusive_minimum"] is not None else None
    )
    exclusive_hi = (
        _num(p["exclusive_maximum"]) if p["exclusive_maximum"] is not None else None
    )
    low = (
        f"{lo}"
        if lo is not None
        else (f">{exclusive_lo}" if exclusive_lo is not None else None)
    )
    high = (
        f"{hi}"
        if hi is not None
        else (f"<{exclusive_hi}" if exclusive_hi is not None else None)
    )
    if low is not None and high is not None:
        return f"{low}–{high}"
    if low is not None:
        return f"≥ {lo}" if lo is not None else f"> {exclusive_lo}"
    if high is not None:
        return f"≤ {hi}" if hi is not None else f"< {exclusive_hi}"
    return "—"


def _default_str(p: dict) -> str:
    if p["required"]:
        return "*required*"
    return "—" if p["default"] is None else f"`{p['default']}`"


#: Modules whose display name is not their id with the underscores taken out. **This is the
#: one source for a module's display name** — the GUI's ``ModuleUI.label`` reads it too
#: (``gui/modules/registry.py``), so the launch picker and the docs page cannot disagree.
#:
#: It lives here rather than on the GUI class because this module imports no Qt, so the
#: dependency runs the safe direction: the GUI may read ``detail.configs``, never the reverse.
#: An id that mangles correctly needs no entry; ``location_export`` → "Location export" already.
_DISPLAY_NAMES = {
    "query": "Behavior scan",
    "query_eval": "Behavior-query eval",
    "temporal_clustering": "Behavior clustering",
    # The importer's own `display_name`, which is what the picker shows. Left to the
    # fallback the page would be titled "Timeseries csv" while the app says "Time-series
    # CSV (sensor data)" — the exact drift this table exists to close.
    "timeseries_csv": "Time-series CSV (sensor data)",
}


def module_display_name(module_id: str) -> str:
    """A module's display name — the launch picker's label and its docs page title.

    Falls back to the id with underscores removed, which is right for most modules. The
    fallback is why this went unnoticed: the docs page for ``query`` was titled "Query" while
    the app called it "Behavior scan", and nothing compared the two —
    ``tests/test_docs_modules_nav.py`` matches nav entries to pages by *path*.
    """
    return _DISPLAY_NAMES.get(module_id) or module_id.replace("_", " ").capitalize()


def render_module_markdown(
    module_id: str, model_cls: type[BaseModel], intro: str = ""
) -> str:
    """One module's page: an H1 + optional intro + a ``### <label> {#<name>}`` per field.

    The explicit ``{#<name>}`` anchor is the field's own name, so the in-app help deep-links a
    field with no slug math (``modules/<id>#<name>``). ``intro`` is prose (no heading) placed
    under the title.
    """
    lines = [f"# {module_display_name(module_id)}", ""]
    if intro:
        lines += [intro.strip(), ""]
    lines += ["## Parameters", ""]
    for p in describe_config(model_cls):
        lines.append(f"### {p['label']} {{#{p['name']}}}")
        lines.append("")
        meta = [f"`{p['name']}`"]
        if p["type"]:
            meta.append(p["type"])
        if p["required"]:
            meta.append("required")
        elif p["default"] is not None:
            meta.append(f"default `{p['default']}`")
        rng = _range_str(p)
        if rng != "—":
            meta.append(f"range {rng}")
        if p["unit"]:
            meta.append(f"unit {p['unit']}")
        lines += [" · ".join(meta), ""]
        if p["help"]:
            lines += [p["help"], ""]
    lines += _extra_field_markdown(module_id)
    return "\n".join(lines)


def _extra_field_markdown(module_id: str) -> list[str]:
    """Anchored doc entries for a module's non-pydantic fields — today the tracker
    hyperparameters, one ``### <label> {#<name>}`` per knob any tracker exposes, so a
    field ``?`` on a dynamically-built tracker row deep-links its definition."""
    if module_id != "tracking":
        return []
    from detail.configs.tracker_params import TRACKER_HYPERPARAM_DOCS

    lines = [
        "## Tracker parameters",
        "",
        "Tuning knobs for the chosen tracker. Which ones appear depends on the "
        "tracker you select.",
        "",
    ]
    for name, param in TRACKER_HYPERPARAM_DOCS.items():
        meta = f"`{name}` · range {_num(param.minimum)}–{_num(param.maximum)}"
        lines += [f"### {param.label} {{#{name}}}", "", meta, "", param.help, ""]
    return lines


def surfaced_config_models() -> list[tuple[str, type[BaseModel]]]:
    """:func:`iter_config_models` filtered to the modules this build shows (§25).

    **For the user-facing docs only.** ``iter_config_models`` stays unfiltered because every other
    consumer needs the whole set: ``detail help``, the unit-convention scan, the advanced-form
    guard and the help-anchor map all read it, and a hidden module's parameters must keep meeting
    those checks — hidden code that stops being guarded is hidden code that rots, and switching it
    back on becomes a surprise rather than a toggle.

    What must *not* happen is the user guide documenting a capability this build does not offer,
    which is why the docs are one of §25's enumeration points.

    **A module id is not always its own way in.** The registry members are reached by id, so
    asking about the id is the whole question for them. The handful with no registry
    (:data:`LAUNCHED_BY`) are reached only through somebody else's command group or GUI section,
    and asking about the id alone answered True for the (since removed) ``cluster_fit`` and
    ``cluster_apply`` while both of their doors were shut — two parameter pages published for a
    capability no user could open. So the launch surface is asked about too.
    """
    from detail import surfaces

    return [
        (module_id, model)
        for module_id, model in iter_config_models()
        if surfaces.is_surfaced(module_id)
        and surfaces.is_surfaced(LAUNCHED_BY.get(module_id, module_id))
    ]


def render_parameters_markdown() -> str:
    """The full Parameters reference page — one section + table per module."""
    lines = [
        "# Parameters",
        "",
        "Every run parameter, generated from each module's config model. This is the one home for "
        "a parameter's bounds and help; the GUI tooltips and CLI `--help` read the same source.",
        "",
    ]
    for module_id, model_cls in surfaced_config_models():
        lines.append(f"## {module_id.replace('_', ' ').capitalize()}")
        lines.append("")
        lines.append("| Parameter | Type | Default | Range | Description |")
        lines.append("| --- | --- | --- | --- | --- |")
        for p in describe_config(model_cls):
            help_txt = p["help"] + (f" ({p['unit']})" if p["unit"] else "")
            lines.append(
                f"| {p['label']} (`{p['name']}`) | {p['type'] or '—'} | {_default_str(p)} | "
                f"{_range_str(p)} | {help_txt} |"
            )
        lines.append("")
    return "\n".join(lines)
