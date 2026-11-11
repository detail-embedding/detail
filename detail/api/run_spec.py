"""Run identity — spec-based upstream selection with full lineage (§6.5).

A **run spec** is the identity of a processing run: its own output-affecting
settings (``ProcessingModule.identity_fields``) plus, recursively, the spec of the
upstream run it was computed from (``ProcessingModule.lineage_ref``). Two runs with
the same spec are interchangeable inputs downstream, regardless of which recording
they live in or which per-recording ordinal (``det_NN``) they were assigned.

This replaces "match on the per-recording run id" — a per-recording monotonic counter
that is ``det_01`` for the first run in *every* recording no matter what it was
(``api/recording.py::_next_run_id``). Matching on that counter both false-negatives
(the same batch is a different ordinal per recording) and false-merges (different runs
share an ordinal). Matching on the spec fixes both, and lets a batch resolve each
recording's own matching run.

A spec is a plain JSON-native structure so it round-trips through the stored run
record::

    {"module": "tracking",
     "identity": {"tracker_id": "anchor_tracker", "arena_ids": [], "hyperparams": {...}},
     "parent": {"module": "detection", "identity": {...}, "parent": {"root": True}}}

``freeze_spec`` turns it into a hashable, normalized key for equivalence (float
rounding + dict/list canonicalization, stable across a JSON round-trip). Everything
here is Qt-free and imports its backends lazily.
"""

from pathlib import Path

# A lineage that bottoms out at the recording's video (detection has no upstream run).
ROOT_SPEC = {"root": True}

# job.json → recovered identity, memoized per run dir (immutable once complete).
_LEGACY_CACHE: dict[str, dict] = {}


def _processing_module(module_id: str):
    """The registered ``ProcessingModule`` for ``module_id`` (importing the runner
    modules on first miss so callers need not pre-import)."""
    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules

    if module_id not in PROCESSING_MODULES:
        from detail.api._runner_modules import PIPELINE_RUNNERS

        import_runner_modules(PIPELINE_RUNNERS)
    try:
        return PROCESSING_MODULES[module_id]
    except KeyError:
        raise ValueError(f"unknown processing module {module_id!r}")


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #
def _freeze(value):
    """Hashable, normalized form of a JSON-native value — the equivalence primitive.

    Floats are rounded (float-repr noise, and JSON round-trip, must not split two
    UI-representable values); dicts become sorted ``(key, frozen)`` tuples; lists and
    tuples become tuples. The result compares equal before and after a JSON round-trip.
    """
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def freeze_spec(spec: dict) -> tuple:
    """Hashable equivalence key for a spec — usable as a dict key or for ``==``."""
    return _freeze(spec)


def _is_root(spec) -> bool:
    return spec is None or (isinstance(spec, dict) and spec.get("root") is True)


# --------------------------------------------------------------------------- #
# Building a spec from a run record
# --------------------------------------------------------------------------- #
def spec_of(
    recording_data: dict, module_id: str, record: dict, recording_dir=None
) -> dict:
    """The full-lineage spec of one run ``record`` in ``recording_data`` (§6.5).

    ``recording_dir`` enables the legacy fallback: for a record written before an
    identity field was recorded (``prompt_fingerprint`` / ``hyperparams``), the value
    is recovered from that run's ``job.json``; unrecoverable → a per-run sentinel that
    never merges. New records need no fallback and pass ``recording_dir=None`` freely.
    """
    module = _processing_module(module_id)
    identity = {}
    defaults = module.identity_defaults
    for field in module.identity_fields:
        if field in record:
            identity[field] = record[field]
        elif field in defaults:
            # The field's absence has one honest reading (see ``identity_defaults``), so a
            # record written before it existed compares equal to one that states it.
            identity[field] = defaults[field]
        else:
            identity[field] = _recover_field(module, field, record, recording_dir)
    return {
        "module": module_id,
        "identity": identity,
        "parent": _parent_spec(module, recording_data, record, recording_dir),
    }


def _parent_spec(module, recording_data: dict, record: dict, recording_dir) -> dict:
    """The spec of ``record``'s lineage parent, or a root/missing sentinel."""
    if module.lineage_ref is None:
        return dict(ROOT_SPEC)
    up_mod, ref_field = module.lineage_ref
    up_id = record.get(ref_field)
    if up_id is None:  # the upstream run is optional and was not used
        return dict(ROOT_SPEC)
    up_module = _processing_module(up_mod)
    up_rec = next(
        (r for r in recording_data.get(up_module.runs_key, []) if r.get("id") == up_id),
        None,
    )
    if up_rec is None:
        return {"missing": f"{up_mod}:{up_id}"}
    return spec_of(recording_data, up_mod, up_rec, recording_dir)


def _recover_field(module, field: str, record: dict, recording_dir) -> object:
    """A legacy record's missing identity value — from ``job.json`` if reachable, else
    a per-run sentinel (a safe false-split; never a false-merge)."""
    if recording_dir is not None:
        recovered = _legacy_identity(module, record, recording_dir)
        if field in recovered:
            return recovered[field]
    # The sentinel must be unique per *recording*, not just per run id: two recordings each
    # hold a first-of-its-kind run at the same ordinal (``det_01``), so keying on the id
    # alone would collapse two genuinely-different legacy runs into one spec across
    # recordings — a false-merge, the dangerous direction. Include the recording so it can
    # only ever false-*split* (safe). Not persisted (only legacy on-the-fly grouping
    # reaches this), so no absolute path lands in a experiment file.
    recording_key = Path(recording_dir).name if recording_dir is not None else "?"
    return f"__unresolved__:{field}:{recording_key}:{record.get('id')}"


def _legacy_identity(module, record: dict, recording_dir) -> dict:
    """Identity fields recovered from a run's ``job.json`` (best-effort, memoized)."""
    output = record.get("output")
    if not output:
        return {}
    run_dir = Path(recording_dir) / output
    cache_key = str(run_dir)
    if cache_key not in _LEGACY_CACHE:
        from detail.executor import rundir

        try:
            job = rundir.read_job(run_dir)
            recovered = module.recompute_identity_from_job(job)
        except (FileNotFoundError, KeyError, ValueError, OSError, TypeError):
            recovered = {}
        _LEGACY_CACHE[cache_key] = recovered
    return _LEGACY_CACHE[cache_key]


def clear_legacy_cache() -> None:
    """Drop the memoized ``job.json`` legacy-identity recoveries (tests; a experiment
    switch). Keys are unique run-dir paths, so this only bounds growth over a
    long-lived process — it never changes a result."""
    _LEGACY_CACHE.clear()


# --------------------------------------------------------------------------- #
# Labels (always carry lineage so differing parents are visible)
# --------------------------------------------------------------------------- #
def spec_label(spec: dict, model_names: dict | None = None) -> str:
    """A human label for the picker — this run's own settings, plus ``· from <parent>``
    when it has a lineage parent (so "same tracker, different detection" is visible).

    ``model_names`` is an optional ``{model_id: name}`` map (``Experiment.model_names``).
    Records key on the model id, which is permanent, but a user reads the name they gave
    it (§11), so a mapped id renders ``name · id``. Passing nothing renders bare ids, which
    keeps this module free of any experiment or Qt dependency.
    """
    own = _own_label(spec.get("module", ""), spec.get("identity", {}), model_names)
    parent = spec.get("parent")
    if isinstance(parent, dict) and "module" in parent:
        return f"{own} · from {spec_label(parent, model_names)}"
    return own


def every_n_label(every_n_frames) -> str:
    """`` · every Nth frame`` for a thinned run, empty for every-frame (the common case).

    Appended to every stage's picker label: a run's cadence decides what a downstream
    stage may ask for, so it has to be visible where the upstream run is chosen.
    """
    stride = int(every_n_frames or 1)
    return f" · every {_ordinal(stride)} frame" if stride > 1 else ""


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _own_label(module_id: str, identity: dict, model_names: dict | None = None) -> str:
    if module_id == "detection":
        return _model_conf_label(
            identity.get("model_id"),
            identity.get("model_variant"),
            identity.get("conf_threshold"),
            identity.get("weights_source"),
            identity.get("prompt_classes"),
            model_names,
        ) + every_n_label(identity.get("every_n_frames"))
    if module_id == "tracking":
        hp = identity.get("hyperparams")
        hp = hp if isinstance(hp, dict) else {}
        conf = hp.get("conf_threshold")
        arenas = identity.get("arena_ids") or []
        label = str(identity.get("tracker_id", "tracker"))
        if isinstance(conf, (int, float)):
            label += f" @ conf {float(conf):.2f}"
        if isinstance(arenas, list) and arenas:
            label += f" · arena {','.join(str(a) for a in arenas)}"
        return label + every_n_label(identity.get("every_n_frames"))
    if module_id == "embedding":
        label = _model_variant_label(
            identity.get("model_id"), identity.get("model_variant"), model_names
        )
        constructor = identity.get("constructor")
        if constructor:
            label += f" · {constructor}"
        return label + every_n_label(identity.get("every_n_frames"))
    return module_id


def _model_variant_label(model_id, variant, model_names: dict | None = None) -> str:
    """``name · model_id[/variant]`` when the id has a name, else the bare id.

    Only an id present in the map gets a name, which is what we want: a built-in model's id
    *is* its name (``sam3``), and a run whose model was deleted since still has to render.
    The "don't print it twice when they match" rule matches the detection picker's.
    """
    name = (model_names or {}).get(model_id)
    label = f"{name} · {model_id}" if name and name != model_id else str(model_id)
    if variant:
        label += f"/{variant}"
    return label


def _model_conf_label(
    model_id, variant, conf, weights_source, classes=None, model_names=None
) -> str:
    label = _model_variant_label(model_id, variant, model_names)
    if isinstance(conf, (int, float)):
        label += f" @ {float(conf):.2f}"
    if weights_source and weights_source != "pretrained":
        label += f" ({weights_source})"
    if isinstance(classes, list) and classes:  # the prompt's target classes
        label += f" · {', '.join(str(c) for c in classes)}"
    return label


# --------------------------------------------------------------------------- #
# Grouping across recordings + per-recording resolution
# --------------------------------------------------------------------------- #
def _recording_run_data(recording) -> dict:
    """The per-module run lists a spec needs, read from a Recording's own record
    properties (``recording.detection_runs`` etc., named by ``runs_key``) — works for
    the live Recording and a lightweight stand-in alike, no recording.yaml re-read."""
    from detail.executor.runners import PROCESSING_MODULES, import_runner_modules

    if not PROCESSING_MODULES:
        from detail.api._runner_modules import PIPELINE_RUNNERS

        import_runner_modules(PIPELINE_RUNNERS)
    return {
        module.runs_key: list(getattr(recording, module.runs_key, []) or [])
        for module in PROCESSING_MODULES.values()
    }


def group_specs(
    recordings, module_id: str, model_names: dict | None = None
) -> list[dict]:
    """Distinct ``module_id`` specs across ``recordings`` — ``[{spec, label, recordings}]``
    where ``recordings`` is the ids of those covered by each spec (newest run wins on a
    same-spec tie within a recording). Does *not* require a spec common to all recordings;
    that is what lets a batch cover recordings whose runs were split across launches.
    """
    entries = (
        (sess.id, _recording_run_data(sess), getattr(sess, "dir", None))
        for sess in recordings
    )
    return _group(entries, module_id, model_names)


def _group(entries, module_id: str, model_names: dict | None = None) -> list[dict]:
    """:func:`group_specs` over ``(recording_id, recording_data, recording_dir)`` triples —
    the shape both a GUI holding Recordings and a ``validate()`` holding only the experiment
    directory can produce. Each group also carries ``runs``, ``{recording_id: run_id}``.
    """
    module = _processing_module(module_id)
    groups: dict[tuple, dict] = {}
    for recording_id, data, recording_dir in entries:
        newest: dict[tuple, tuple] = {}  # key -> (spec, created_at, run_id)
        for record in data.get(module.runs_key, []):
            if record.get("status") != "complete":
                continue
            spec = spec_of(data, module_id, record, recording_dir)
            key = freeze_spec(spec)
            created = record.get("created_at") or ""
            if key not in newest or created > newest[key][1]:
                newest[key] = (spec, created, record.get("id"))
        for key, (spec, _created, run_id) in newest.items():
            group = groups.setdefault(key, {"spec": spec, "runs": {}})
            group["runs"][recording_id] = run_id
    result = [
        {
            "spec": g["spec"],
            "label": spec_label(g["spec"], model_names),
            "recordings": list(g["runs"]),
            "runs": dict(g["runs"]),
        }
        for g in groups.values()
    ]
    result.sort(key=lambda g: g["label"])
    return result


def resolve_spec_in_recording(
    recording_data: dict, module_id: str, spec: dict, recording_dir=None
) -> str | None:
    """This recording's own complete ``module_id`` run id matching ``spec`` (newest on a
    tie), or ``None`` if the recording has no matching run."""
    module = _processing_module(module_id)
    target = freeze_spec(spec)
    best_id = None
    best_created = None
    for record in recording_data.get(module.runs_key, []):
        if record.get("status") != "complete":
            continue
        if (
            freeze_spec(spec_of(recording_data, module_id, record, recording_dir))
            == target
        ):
            created = record.get("created_at") or ""
            if best_created is None or created > best_created:
                best_created, best_id = created, record.get("id")
    return best_id


def resolve_spec_for_recording(recording, module_id: str, spec: dict) -> str | None:
    """Resolve ``spec`` to ``recording``'s own run id — the GUI convenience for callers
    holding a Recording (reads its record properties) rather than a recording_data dict.
    """
    return resolve_spec_in_recording(
        _recording_run_data(recording), module_id, spec, getattr(recording, "dir", None)
    )


def spec_from_run(recording, module_id: str, run_id: str) -> dict:
    """The spec of a *named* run in ``recording`` — for the CLI ``--*-ref RECORDING:run_id``
    selectors, which derive a spec from one concrete run and resolve each recording's own
    match.

    Raises:
        ValueError: The recording has no such run.
    """
    data = _recording_run_data(recording)
    module = _processing_module(module_id)
    record = next(
        (r for r in data.get(module.runs_key, []) if r.get("id") == run_id), None
    )
    if record is None:
        run_ref = getattr(recording, "id", recording)
        raise ValueError(f"recording {run_ref!r} has no {module_id} run {run_id!r}")
    return spec_of(data, module_id, record, getattr(recording, "dir", None))


def resolve_analysis_selection(
    module, experiment_dir, recording_ids: list[str], config: dict
) -> dict[str, str]:
    """``{recording_id: run_id}`` — the upstream run an analysis reads in each recording.

    Driven by the module's ``selects = (upstream_module, ref_field)`` declaration, so every
    analysis resolves its upstream the same way. Four inputs, in order:

    - ``<upstream>_spec`` (the GUI picker, the CLI's ``--*-ref``): each recording's own run
      with those settings, whatever id it was given.
    - ``<upstream>_specs``, a list (the picker with several groups ticked, ``--*-ref``
      repeated): each recording's own run from the one listed set of settings it has. For
      recordings whose upstream was made through different chains, e.g. two detection models
      for two strains, so no single spec covers them all. A recording with a run of none,
      or of more than one, is refused rather than resolved by list order (Tenet 9). Pooled
      embedding runs must also pass ``run_preview.validate_comparability``, because a spec
      equal across recordings no longer vouches for the recipe.
    - a single ``ref_field`` id: that id in every recording — the literal, older form.
    - neither: the one set of settings every selected recording has a complete run of.
      Refused when there is none, and when there are several (Tenet 9: a result must not
      depend on which sorted first).

    Settings rather than the shared per-recording id, because the id is a counter a retry or
    a separate launch moves: the same settings sit under ``emb_01`` in one recording and
    ``emb_02`` in the next, and different settings can share an ``emb_01``.

    Raises:
        ValueError: A recording has no complete matching run, or the default is ambiguous.
    """
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    up_mod, ref_field = module.selects
    runs_key = _processing_module(up_mod).runs_key
    entries = []
    for sid in recording_ids:
        recording_dir = experiment_store.recording_path(experiment_dir, sid)
        entries.append(
            (sid, recording_store.read_recording(recording_dir), recording_dir)
        )

    spec = config.get(f"{up_mod}_spec")
    specs = config.get(f"{up_mod}_specs")
    if spec is not None and specs:
        raise ValueError(
            f"pass {up_mod}_spec or {up_mod}_specs, not both — {up_mod}_specs already "
            "names every set of settings to read"
        )
    if specs:
        return _resolve_across_specs(entries, up_mod, specs)
    if spec is not None:
        mapping = {}
        for sid, data, recording_dir in entries:
            found = resolve_spec_in_recording(data, up_mod, spec, recording_dir)
            if found is None:
                raise ValueError(
                    f"recording {sid!r} has no complete {up_mod} run with the selected "
                    f"settings ({spec_label(spec)}); run {up_mod} on it, or leave it out"
                )
            mapping[sid] = found
        return mapping
    run_id = config.get(ref_field)
    if run_id:
        for sid, data, _recording_dir in entries:
            by_id = {r.get("id"): r for r in data.get(runs_key, [])}
            record = by_id.get(run_id)
            if record is None:
                raise ValueError(f"recording {sid!r} has no {up_mod} run {run_id!r}")
            if record.get("status") != "complete":
                raise ValueError(
                    f"{up_mod} run {run_id!r} in recording {sid!r} is "
                    f"{record.get('status')!r}, not complete"
                )
        return {sid: run_id for sid in recording_ids}
    return _default_selection(_group(entries, up_mod), up_mod, recording_ids)


def _resolve_across_specs(entries, up_mod: str, specs: list[dict]) -> dict[str, str]:
    """``{recording_id: run_id}`` from several sets of settings, one per recording.

    Each recording must have a complete run of exactly one listed spec. Two would leave the
    choice to list order, and none means the recording was selected without anything to read.

    Raises:
        ValueError: A recording matches none or several of ``specs``, or the pooled
            embedding runs are not comparable.
    """
    unique: dict[tuple, dict] = {}
    for spec in specs:
        unique.setdefault(freeze_spec(spec), spec)
    mapping: dict[str, str] = {}
    for sid, data, recording_dir in entries:
        found = [
            (spec, run_id)
            for spec in unique.values()
            if (run_id := resolve_spec_in_recording(data, up_mod, spec, recording_dir))
        ]
        if not found:
            labels = "; ".join(spec_label(s) for s in unique.values())
            raise ValueError(
                f"recording {sid!r} has no complete {up_mod} run with any of the selected "
                f"settings ({labels}); run {up_mod} on it, or leave it out"
            )
        if len(found) > 1:
            labels = "; ".join(f"{spec_label(s)} ({run_id})" for s, run_id in found)
            raise ValueError(
                f"recording {sid!r} has a run of more than one of the selected {up_mod} "
                f"settings — {labels}. Select only one of them, so the run read there "
                "does not depend on list order"
            )
        mapping[sid] = found[0][1]
    if up_mod == "embedding" and len(unique) > 1:
        from detail.api.run_preview import validate_comparability

        validate_comparability(
            [(recording_dir, mapping[sid]) for sid, _data, recording_dir in entries]
        )
    return mapping


def _default_selection(
    groups: list[dict], up_mod: str, recording_ids: list[str]
) -> dict[str, str]:
    """The one group covering every recording, or an error saying what to do instead."""
    total = len(recording_ids)
    full = [g for g in groups if len(g["recordings"]) == total]
    if len(full) == 1:
        return dict(full[0]["runs"])
    if not full:
        if not groups:
            raise ValueError(
                f"the selected recordings have no completed {up_mod} run; "
                f"run {up_mod} on them first"
            )
        best = max(groups, key=lambda g: len(g["recordings"]))
        missing = [sid for sid in recording_ids if sid not in best["runs"]]
        more = " …" if len(missing) > 5 else ""
        raise ValueError(
            f"no {up_mod} settings cover every selected recording. The closest, "
            f"{best['label']!r}, covers {len(best['recordings'])} of {total}; missing "
            f"{missing[:5]}{more}. Run {up_mod} on those with the same settings, or "
            "leave them out"
        )
    options = "; ".join(
        "{!r} (e.g. {}:{})".format(g["label"], *next(iter(g["runs"].items())))
        for g in full
    )
    raise ValueError(
        f"the selected recordings share {len(full)} different {up_mod} settings — "
        f"{options}. Choose one explicitly rather than letting the result depend on "
        "which sorted first"
    )


def resolve_into_config(
    module, recording_dir, recording_data: dict, config: dict
) -> dict:
    """Resolve a batch ``<upstream>_spec`` in ``config`` to this recording's own upstream
    run id, in place (§6.5). A no-op when the module has no lineage parent or the config
    carries an explicit id. Pops the spec key so downstream ``validate`` sees only the
    resolved ref field.

    Raises:
        ValueError: The recording has no complete upstream run matching the spec.
    """
    refs = []
    if module.lineage_ref is not None:
        refs.append(module.lineage_ref)
    # §6.5 auxiliary inputs (embedding's keypoint track run, §14): a batch run passes
    # ``<aux>_spec`` and each recording resolves its own match, exactly as the upstream.
    refs.extend(getattr(module, "aux_refs", ()) or ())
    for up_mod, ref_field in refs:
        spec_field = f"{up_mod}_spec"
        if spec_field not in config:
            continue
        spec = config.pop(spec_field)
        if _is_root(spec):
            config[ref_field] = None
            continue
        resolved = resolve_spec_in_recording(
            recording_data, up_mod, spec, recording_dir
        )
        if resolved is None:
            raise ValueError(
                f"no complete {up_mod} run matching the selected spec in recording "
                f"{Path(recording_dir).name!r}"
            )
        config[ref_field] = resolved
    return config
