"""A config form can read back what it was given (§16 Refine, and settings reuse).

The class: **a form that accepts a configuration and silently ignores it.**
``ConfigWidget.load_config`` was a *default no-op* on the base class, so a module that never
implemented it still satisfied the protocol, still built, still launched — and quietly opened at
defaults while a button above it said "Refine (new run seeded from this)". Four analysis modules
shipped exactly that; five more had an implementation written against the run *record* while
``config()`` emitted a different shape, so they restored some knobs and dropped others.

Nothing fails when they do. The dialog opens, the form is valid, the run completes, and the user
gets a run configured differently from the one they asked to refine. The existing Refine tests
assert the *button appears*, which is true either way.

The rule: ``config()`` and ``load_config()`` are inverses **over the fields that are settings**.
A *selection* field (``embed_run_id``, ``source_recording_ids``, ``clusterer_id``) is deliberately
not round-tripped — it names an artifact the dialog re-asks for, and carrying one is what kills
this class of feature elsewhere (ilastik's ``.ilp``, a SLEAP config that cannot find its
``models/`` folder on another machine).

Three ways this guard can pass without guarding, all of which it did while being written:

1. **No perturbation.** Read, load what you just read, read again — passes against a no-op for
   the plainest possible reason: nothing touched the widgets. The first draft did this and
   reported every broken module as passing. So the config handed back is a *different* one.
2. **Skipping on a missing method.** Requiring both ``config`` and ``load_config`` before
   testing let ``ComparisonLaunchDialog`` sit outside the guard while emitting a full nine-key
   config — the defect *is* having one and not the other, so that is now an assertion.
3. **Perturbing only the top level.** The benchmark member forms (since removed) nested
   everything under ``base_fit`` / ``base_eval``, so a flat walk changed nothing and skipped.
   The walk recurses.

Data-driven over the same registries ``test_advanced_form.py`` walks, so a module added tomorrow
is covered without anyone remembering this file.
"""

import math

import pytest

from detail.configs.describe import describe_config, iter_config_models

from tests.gui.test_advanced_form import _DIALOGS, _build_dialog, _registry_forms

#: Errors a form raises when asked for a config it has no data for (no video picked, no
#: experiment set). Not a failure of the round trip — the case is skipped and says so.
_UNCONFIGURABLE = (
    NotImplementedError,
    AttributeError,
    ValueError,
    KeyError,
    IndexError,
)

#: Keys that name an artifact rather than a setting, for the surfaces with no config model to
#: read ``scope`` from (the launch dialogs; the benchmark member forms too, until they were
#: removed). Not the source of truth: a new *module* field is classified on the pydantic field
#: itself, never here.
_SELECTION_KEYS = frozenset(
    {
        "source_recording_ids",
        "recordings",
        "embed_run_id",
        "clusterer_id",
        "source_run_id",
        "source_snapshot",
        "tracking_run_id",
    }
)

#: Fields a form deliberately does not render, so ``config()`` reports a constant and there is
#: nothing to restore. An entry is a **claim**, checked: the field must come back *unchanged*
#: after a load, so an exemption that stops being true fails rather than hiding a real gap.
EXEMPT: dict[str, dict[str, str]] = {
    "keypoint_detect": {
        "min_pose_confidence": (
            "shown only for a method that reports a pose confidence (§26 *Visibility*); "
            "the form opens on dinov3_match, which reports none and whose validate refuses "
            "the threshold, so config() sends None there. The round trip and the falsy 0.0 "
            "for a method that reports one are test_keypoint_detect_gui.py's "
            "test_min_pose_confidence_shows_only_for_a_method_that_reports_one"
        ),
    },
}


def _same(a, b) -> bool:
    """Equal, allowing for a spin box quantizing a float to its own decimals.

    ``0.07 + 0.5`` is ``0.5700000000000001`` in binary floating point and a two-decimal box
    shows ``0.57``. Comparing exactly would fail every float knob for a reason that has nothing
    to do with the round trip.
    """
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    return a == b


def _fields(module_id: str) -> dict[str, dict]:
    model = dict(iter_config_models()).get(module_id)
    return {f["name"]: f for f in describe_config(model)} if model is not None else {}


def _config_of(widget):
    try:
        return widget.config()
    except _UNCONFIGURABLE:
        return None


def _from_model(field: dict, value):
    """A different valid value for a field the config model describes."""
    if field["choices"]:
        others = [c for c in field["choices"] if c != value]
        return others[0] if others else None
    if field["type"] == "boolean":
        return not bool(value)
    if field["type"] not in ("integer", "number") or value is None:
        return None
    low = field["minimum"] if field["minimum"] is not None else 0
    high = field["maximum"] if field["maximum"] is not None else value + 1
    step = 1 if field["type"] == "integer" else 0.5
    candidate = value + step
    if candidate > high:
        candidate = value - step
    if candidate < low or candidate == value:
        return None
    return int(candidate) if field["type"] == "integer" else candidate


#: A tenth of a second, in nanoseconds. A ``*_ns`` key is rendered by a *seconds* box, so a
#: perturbation has to be large enough to survive that box's two decimals — a nanosecond is not,
#: and a zero-defaulted duration cannot be halved.
_NS_STEP = 100_000_000


def _from_value(name: str, value):
    """A different valid value for a key with no model behind it, or ``None`` for no idea.

    Only bools and numbers. A string or a list has no generically-safe "other value" —
    inventing one risks failing a form for rejecting nonsense rather than for ignoring its
    config, which is how a guard becomes the thing people learn to work around.

    Numbers are **halved**, not nudged, and that matters twice. A nudge of +1 on a duration
    stored in nanoseconds is a *nanosecond*, which is far below the resolution of the
    two-decimal seconds box that renders it — the form holds 0.5 s either way and the guard
    reports a defect that is really its own rounding. And a nudge upward walks a bounded knob
    into its ceiling (`extent_fraction` defaults to 0.999 of a maximum 0.999), where the widget
    clamps and, again, the guard blames the form. Halving moves by a visible amount and lands
    inside any range whose floor is at or below half the default, which every knob here has.
    Floats are rounded to two decimals, the coarsest box resolution in the package.
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        if name.endswith("_ns"):
            return value // 2 if value >= 2 * _NS_STEP else value + _NS_STEP
        return value // 2 if value >= 2 else value + 1
    if isinstance(value, float):
        halved = round(value / 2, 2)
        return halved if 0 < halved < value else None
    return None


def _perturb(
    config: dict, fields: dict[str, dict], path: tuple = (), nested_model=None
) -> tuple[dict, dict]:
    """``config`` with every settable setting moved, and ``{path: new value}`` for each.

    Recurses into nested dicts (``base_fit`` / ``base_eval`` / ``method_params`` / ``windows``),
    because the benchmark member forms (since removed) put *everything* one level down: a
    flat walk found nothing to change in them, skipped, and left the implementations it was
    written for untested.
    """
    out = dict(config)
    changed: dict[tuple, object] = {}
    for name, value in config.items():
        if name in _SELECTION_KEYS:
            continue
        if isinstance(value, dict):
            # Nested method / fit knobs. The module's own model does not describe them (they
            # belong to the method). A form that knows the method's model names it
            # (`nested_config_model(key, config)`), so its bounds are respected; otherwise
            # they are perturbed by value. Halving a knob whose floor sits above half its
            # default (`smooth_min_points`: 5, floor 3) asks for an invalid value the widget
            # rightly clamps, and the guard then blames the form.
            model = nested_model(name, config) if nested_model is not None else None
            sub_fields = (
                {f["name"]: f for f in describe_config(model)}
                if model is not None
                else {}
            )
            sub_out, sub_changed = _perturb(value, sub_fields, path + (name,))
            if sub_changed:
                out[name] = sub_out
                changed.update(sub_changed)
            continue
        field = fields.get(name)
        other = _from_model(field, value) if field else _from_value(name, value)
        if other is not None:
            out[name] = other
            changed[path + (name,)] = other
    return out, changed


def _at(config, path: tuple):
    for key in path:
        if not isinstance(config, dict):
            return None
        config = config.get(key)
    return config


@pytest.mark.parametrize("module_id", sorted(_registry_forms()))
def test_a_form_reads_back_the_settings_it_was_given(qapp, module_id):
    widget = _registry_forms()[module_id].config_widget()
    before = _config_of(widget)
    if before is None:
        pytest.skip(f"{module_id} cannot produce a config without an experiment")

    settings = {n: f for n, f in _fields(module_id).items() if f["scope"] == "setting"}
    target, changed = _perturb(
        before, settings, nested_model=getattr(widget, "nested_config_model", None)
    )
    if not changed:
        pytest.skip(
            f"{module_id} has no perturbable setting — every field it emits names an "
            "artifact, or is a string this guard will not invent a value for"
        )

    widget.load_config(target)
    after = _config_of(widget)
    assert after is not None

    exempt = EXEMPT.get(module_id, {})
    for path in sorted(changed):
        name = path[-1]
        if name in exempt:
            assert _same(_at(after, path), _at(before, path)), (
                f"{module_id}.{name} is exempt as a value the form does not render, but it "
                "moved — the exemption is stale: delete it and restore the field."
            )
            continue
        assert _same(_at(after, path), changed[path]), (
            f"{module_id}.{'.'.join(path)} did not survive load_config — the form accepted a "
            f"config and kept its own value ({_at(after, path)!r} rather than "
            f"{changed[path]!r}). That is invisible in use: the form opens at the default, "
            "the run is valid, and nothing fails."
        )


@pytest.mark.parametrize("name", _DIALOGS)
def test_a_launch_dialog_reads_back_the_settings_it_was_given(qapp, name, tmp_path):
    """The launch surfaces outside the module registries, same rule.

    These have no ``module_id`` to look a model up by, so every key is perturbed by value and
    ``_SELECTION_KEYS`` stands in for the ``scope`` declaration.
    """
    widget = _build_dialog(name, tmp_path)
    if not hasattr(widget, "config"):
        pytest.skip(f"{name} has no config() — nothing to round-trip")
    # Deliberately NOT skipped when `load_config` is missing: a dialog that can *produce* a
    # config and cannot take one back is the defect, not an exclusion. Skipping on both is how
    # `ComparisonLaunchDialog` sat outside this guard while emitting a full nine-key config.
    assert hasattr(widget, "load_config"), (
        f"{name} produces a config but cannot read one back — Refine and the settings picker "
        "would open it at defaults with nothing failing."
    )
    before = _config_of(widget)
    if not before:
        pytest.skip(f"{name} cannot produce a config in isolation")

    target, changed = _perturb(before, {})
    if not changed:
        pytest.skip(f"{name} emits no perturbable value")

    widget.load_config(target)
    after = _config_of(widget)
    assert after is not None
    for path in sorted(changed):
        assert _same(_at(after, path), changed[path]), (
            f"{name}.{'.'.join(path)} did not survive load_config "
            f"({_at(after, path)!r} rather than {changed[path]!r})"
        )


#: Modules whose `build_job` is callable from a form's own config with nothing else — no
#: experiment, no disk.
_JOB_BUILDABLE = {
    "temporal_clustering",
    "query",
}

#: Modules whose `validate()` **transforms** the config before `build_job` sees it, so the form's
#: own output cannot build a job. Each entry names the transform, and
#: `test_the_validate_only_list_is_not_stale` checks the claim by calling `build_job` and
#: requiring it to refuse — an entry that stops being true drops a module out of the
#: end-to-end round trip silently, which is how a list like this rots.
_NEEDS_VALIDATE: dict[str, str] = {}


@pytest.mark.parametrize("module_id", sorted(_NEEDS_VALIDATE))
def test_the_validate_only_list_is_not_stale(module_id):
    from detail.executor.runners import ANALYSIS_MODULES

    widget = _registry_forms()[module_id].config_widget()
    config = {
        "source_recording_ids": [],
        "search_recording_ids": [],
        **(_config_of(widget) or {}),
    }
    with pytest.raises((KeyError, TypeError, ValueError)):
        ANALYSIS_MODULES[module_id].build_job("an_test", config)


def _falsy_target(field: dict):
    """The falsy value this field legally accepts and does not default to, or ``None``.

    ``0`` for a number whose range reaches it, ``False`` for a bool defaulting True, ``[]`` for
    a non-empty list default.
    """
    if field["type"] == "boolean":
        return False if field["default"] is True else None
    if field["type"] == "array":
        return [] if field["default"] else None
    if field["type"] not in ("integer", "number") or field["default"] == 0:
        return None
    low, exclusive = field["minimum"], field["exclusive_minimum"]
    if (low is None or low <= 0) and (exclusive is None or exclusive < 0):
        return 0 if field["type"] == "integer" else 0.0
    return None


@pytest.mark.parametrize("module_id", sorted(_registry_forms()))
def test_a_form_reads_back_a_setting_whose_value_is_falsy(qapp, module_id):
    """The value a truthiness check drops: ``0``, ``False``, ``[]``.

    A distinct class from the test above, and invisible to it. ``load_config`` implementations
    are written as ``if config.get("max_gap_ns"): ...`` — which is right for an enum string,
    where no legal value is falsy, and silently wrong for a number whose range reaches zero. A
    gap of ``0`` means *never split*; a ``require_ground_truth`` of ``False`` means *score the
    unlabeled recordings too*. Both are deliberate choices, and both are read as "the key isn't
    there" and quietly replaced by the default.

    It is the same shape as the bug that prompted this test: picking **Defaults** in the
    Start-from row did nothing, because the handler skipped an item whose payload was falsy.
    A control that applies some of its values and ignores the rest, where the ignored one is
    the one meaning "none".
    """
    widget = _registry_forms()[module_id].config_widget()
    before = _config_of(widget)
    if before is None:
        pytest.skip(f"{module_id} cannot produce a config without an experiment")

    exempt = EXEMPT.get(module_id, {})
    fields = _fields(module_id)
    target = dict(before)
    changed = {}
    for key in before:
        # A duration is declared in seconds and emitted in nanoseconds — the one renaming this
        # codebase applies systematically ("Durations are stored in nanoseconds; the form's
        # field is seconds"). Following it here is what reaches `query` (and reached
        # `clip_annotation`, until it was removed), which would otherwise skip this sweep entirely
        # and is exactly where the pattern lives.
        field = fields.get(key) or (
            fields.get(f"{key[:-3]}_s") if key.endswith("_ns") else None
        )
        if field is None or field["scope"] != "setting" or key in exempt:
            continue
        falsy = _falsy_target(field)
        if falsy is not None and not _same(before[key], falsy):
            target[key] = changed[key] = falsy
    if not changed:
        pytest.skip(f"{module_id} has no setting that legally takes a falsy value")

    widget.load_config(target)
    after = _config_of(widget)
    assert after is not None
    for name in sorted(changed):
        assert _same(after.get(name), changed[name]), (
            f"{module_id}.{name} was set to {changed[name]!r} and came back "
            f"{after.get(name)!r} — a legal value dropped for being falsy. `if config.get(x)` "
            "is right for a choice and wrong for a number whose range reaches zero."
        )


def test_some_module_actually_has_a_falsy_settable_field():
    """Guard the guard: the sweep above is worthless if every module skips it."""
    covered = [
        module_id
        for module_id in _registry_forms()
        if any(_falsy_target(f) is not None for f in _fields(module_id).values())
    ]
    assert len(covered) >= 3, covered


@pytest.mark.parametrize("module_id", sorted(_JOB_BUILDABLE))
def test_a_form_reads_back_what_the_run_record_actually_stores(qapp, module_id):
    """The whole path, not just the widgets: form → job → record params → form.

    The test above proves a form is its own inverse. It cannot see the step in between, where a
    runner converts what the form emitted into what it stores — `syllable_discovery` (since
    removed) wrote `window_ns` for a form field named `window_s`, so its `load_config` was
    reading keys the form never produced and restoring nothing. That is invisible to a widget-only
    round trip and to every behavioural test, because the run is correct; only *re-opening* it is
    wrong.

    So this goes through `build_job` and `runners.job_params` — the real writers of
    `record["params"]` — and `launch.seed_config`, the real reader. A key the job renames on the
    way out shows up here as a knob that came back at its default.
    """
    from detail.executor.runners import ANALYSIS_MODULES, job_params
    from detail.gui.launch import seed_config

    widget = _registry_forms()[module_id].config_widget()
    before = _config_of(widget)
    assert before is not None, f"{module_id} produced no config to build a job from"

    module = ANALYSIS_MODULES[module_id]
    settings = {n: f for n, f in _fields(module_id).items() if f["scope"] == "setting"}
    target, changed = _perturb(before, settings)
    assert changed, f"{module_id} has no perturbable setting"

    # The form's own config is what the launch dialog hands the runner, so the job is built
    # from `target` exactly as a real run would build it.
    # The recording lists come from the launch dialog, not the form, so they are supplied here
    # the way it supplies them. They are selections and take no part in the assertions.
    dialog_supplied = {"source_recording_ids": [], "search_recording_ids": []}
    job = module.build_job("an_test", {**dialog_supplied, **before, **target})
    params = job_params(job)
    widget.load_config(seed_config({"module_id": module_id, "params": params}))
    after = _config_of(widget)

    exempt = EXEMPT.get(module_id, {})
    checked = [path for path in changed if path[-1] not in exempt]
    absent = sorted(".".join(p) for p in checked if _at(params, p) is None)
    assert not absent, f"{module_id}: absent from record params: {absent}"
    for path in sorted(checked):
        assert _same(_at(after, path), changed[path]), (
            f"{module_id}.{'.'.join(path)}: the form was set to {changed[path]!r}, the record "
            f"stored {_at(params, path)!r}, and re-opening gives {_at(after, path)!r}. The "
            "form and the runner disagree about this parameter's name, unit or value — "
            "invisible in a run, and wrong every time somebody refines one."
        )


def test_every_analysis_module_records_the_settings_a_form_can_set():
    """A module whose record omits a knob the form sets cannot be refined, however good its
    `load_config` is. Data-driven, so this is the list that has to shrink, not a fixed one.
    """
    import inspect

    from detail.executor.runners import ANALYSIS_MODULES

    silent = sorted(
        module_id
        for module_id, module in ANALYSIS_MODULES.items()
        if 'record["params"]' not in inspect.getsource(type(module))
    )
    assert not silent, (
        f"these analysis modules record no params at all: {silent}. Their runs cannot be "
        "reproduced from their own metadata and re-open at defaults."
    )


def test_every_declared_scope_is_one_of_the_two_it_can_be():
    """A misspelt ``scope`` is silently "not a setting", which excuses a field from the guard.

    ``describe_config`` defaults an undeclared field to ``"setting"``, so a typo — ``"selecton"``
    — does not read as a default; it reads as *neither*, and this file's ``scope == "setting"``
    filter then quietly stops perturbing that knob. The form would go untested for exactly the
    field somebody had just thought hard enough about to annotate.
    """
    wrong = [
        f"{module_id}.{field['name']} = {field['scope']!r}"
        for module_id, model in iter_config_models()
        for field in describe_config(model)
        if field["scope"] not in ("setting", "selection")
    ]
    assert not wrong, f"unknown scope: {', '.join(wrong)}"


def test_the_guard_would_catch_a_form_that_ignores_its_config(qapp):
    """Guard the guard: a no-op ``load_config`` must not satisfy the assertions above.

    Written against the same shape the real tests use — read, perturb, load, read — because the
    version *without* the perturbation passed on a no-op, which is how the flaw in the first
    draft of this file was found. The nested case is checked too: the flat walk that preceded
    ``_perturb`` skipped both benchmark member forms.
    """

    class _Ignores:
        def __init__(self):
            self._value = {"knob": 1, "base_fit": {"nested": 2}}

        def config(self):
            return {
                "knob": self._value["knob"],
                "base_fit": dict(self._value["base_fit"]),
            }

        def load_config(self, config):
            pass  # the defect, in one line

    widget = _Ignores()
    before = widget.config()
    target, changed = _perturb(before, {})
    assert set(changed) == {("knob",), ("base_fit", "nested")}, (
        "the perturbation missed the nested key, so a form that ignores only its nested "
        "config would skip rather than fail"
    )
    widget.load_config(target)
    after = widget.config()
    assert any(
        not _same(_at(after, p), changed[p]) for p in changed
    ), "a no-op load_config was not detected, so the assertions above cannot fail"


def test_the_query_eval_form_round_trips_with_behaviors_present(qapp):
    """#35 finding 3. Every other case here builds the form with no recordings, so its
    `behaviors` is `[]` and the line that read them never ran: `config()` emits names while
    `load_config` read each entry as a run record's dict (`b.get("skipped")`), and raised
    `AttributeError` the first time a behavior was listed. One shape now, both ways."""
    from PyQt6.QtCore import Qt

    from detail.gui.modules.analysis_query_eval import _QueryEvalConfig

    vocabulary = ([], {None: (["dig", "groom", "rear"], {})})
    form = _QueryEvalConfig()
    form.apply_recordings([], vocabulary)
    form._behaviors.item(0).setCheckState(Qt.CheckState.Unchecked)  # dig
    written = form.config()
    assert written["behaviors"] == ["groom", "rear"], "the fixture must name a subset"

    form.load_config(written)  # raised AttributeError
    assert form.config()["behaviors"] == ["groom", "rear"]

    fresh = _QueryEvalConfig()  # the Start from order: load first, vocabulary after
    fresh.load_config(written)
    fresh.apply_recordings([], vocabulary)
    assert fresh.config()["behaviors"] == ["groom", "rear"]
