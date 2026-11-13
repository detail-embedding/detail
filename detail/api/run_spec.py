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

A run that read an auxiliary input (``ProcessingModule.aux_refs``, §26 — an aligned
embedding's keypoints run) also carries ``"aux": {"keypoints": <that run's spec>}``; a run that
read none has no ``"aux"`` key, so every spec written before aux inputs existed is unchanged.

``freeze_spec`` turns it into a hashable, normalized key for equivalence (float
rounding + dict/list canonicalization, stable across a JSON round-trip). Everything
here is Qt-free and imports its backends lazily.
"""

from pathlib import Path

from detail.core.class_names import label as class_label
from detail.pipeline import teacher_window

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
    spec = {
        "module": module_id,
        "identity": identity,
        "parent": _parent_spec(module, recording_data, record, recording_dir),
    }
    aux = _aux_specs(module, recording_data, record, recording_dir)
    if aux:
        spec["aux"] = aux
    return spec


def _aux_specs(module, recording_data: dict, record: dict, recording_dir) -> dict:
    """``{aux_module: spec}`` for each auxiliary input ``record`` names (§26)."""
    out = {}
    for aux_mod, ref_field in module.aux_refs:
        aux_id = record.get(ref_field)
        if aux_id is None:
            continue
        aux_module = _processing_module(aux_mod)
        aux_rec = next(
            (
                r
                for r in recording_data.get(aux_module.runs_key, [])
                if r.get("id") == aux_id
            ),
            None,
        )
        out[aux_mod] = (
            {"missing": f"{aux_mod}:{aux_id}"}
            if aux_rec is None
            else spec_of(recording_data, aux_mod, aux_rec, recording_dir)
        )
    return out


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
def spec_label(
    spec: dict,
    model_names: dict | None = None,
    *,
    class_names: dict | None = None,
    provenance: dict | None = None,
) -> str:
    """A human label for the picker — this run's own settings, plus ``· from <parent>``
    when it has a lineage parent (so "same tracker, different detection" is visible).

    ``model_names`` is an optional ``{model_id: name}`` map (``Experiment.model_names``).
    Records key on the model id, which is permanent, but a user reads the name they gave
    it (§11), so a mapped id renders ``name · id``. ``class_names`` is the experiment's
    ``{stored class: name}`` (§1 *Class names*), so a class reads as the user calls it.
    ``provenance`` (:func:`provenance_of`, merged across recordings by :func:`group_specs`)
    leads each level with the experiment run(s) that made it and adds what the records show
    beside the identity, such as per-class thresholds. Passing nothing renders bare ids and
    stored names, which keeps this module free of any experiment or Qt dependency.
    """
    names = class_names or {}
    prov = provenance or {}
    own = _own_label(
        spec.get("module", ""),
        spec.get("identity", {}),
        model_names,
        names,
        prov.get("shown") or {},
    )
    runs = runs_label(prov.get("runs") or [])
    if runs:
        own = f"{runs} · {own}"
    for aux_mod, aux in (spec.get("aux") or {}).items():
        if isinstance(aux, dict) and "module" in aux:
            aux_prov = (prov.get("aux") or {}).get(aux_mod) or {}
            aux_label = _own_label(
                aux["module"],
                aux.get("identity", {}),
                model_names,
                names,
                aux_prov.get("shown") or {},
            )
            aux_runs = runs_label(aux_prov.get("runs") or [])
            if aux_runs:
                aux_label = f"{aux_runs} · {aux_label}"
            own += f" · {aux['module']} {aux_label}"
    parent = spec.get("parent")
    if isinstance(parent, dict) and "module" in parent:
        parent_label = spec_label(
            parent, model_names, class_names=names, provenance=prov.get("parent")
        )
        return f"{own} · from {parent_label}"
    return own


#: How many experiment runs a label names before it says ``+N more``.
_RUNS_SHOWN = 3


def runs_label(runs: list[str]) -> str:
    """``run_001 + run_004`` — the experiment runs a label reads, ``+N more`` past three."""
    shown = " + ".join(runs[:_RUNS_SHOWN])
    hidden = len(runs) - _RUNS_SHOWN
    return f"{shown} + {hidden} more" if hidden > 0 else shown


#: Record keys a label shows that are not identity fields themselves — the identity holds
#: only their digest (``class_conf_thresholds_fingerprint``, ``class_example_digests``), which
#: tells two runs apart but cannot be read.
_SHOWN_FIELDS = {
    "detection": ("class_conf_thresholds",),
    "keypoint_detect": ("example_set_id", "class_example_sets"),
}


def provenance_of(recording_data: dict, module_id: str, record: dict) -> dict:
    """What a label shows about ``record`` beside its settings, down its lineage.

    ``{"runs": [name], "shown": {...}, "parent": {...}, "aux": {module: {...}}}``, mirroring
    the spec's shape. The name is the experiment run that launched the record
    (``processing_run_id``, ``run_NNN``), which is what a user knows a run by; a record made
    outside a processing run has only its per-recording id, so that id stands in.
    """
    module = _processing_module(module_id)
    prov = {
        "runs": [record.get("processing_run_id") or str(record.get("id", "?"))],
        "shown": {
            f: record[f]
            for f in _SHOWN_FIELDS.get(module_id, ())
            if record.get(f) not in (None, {}, "")
        },
    }
    refs = [module.lineage_ref] if module.lineage_ref else []
    for is_aux, (up_mod, ref_field) in [(False, r) for r in refs] + [
        (True, r) for r in module.aux_refs
    ]:
        up_id = record.get(ref_field)
        if up_id is None:
            continue
        up_module = _processing_module(up_mod)
        up_rec = next(
            (
                r
                for r in recording_data.get(up_module.runs_key, [])
                if r.get("id") == up_id
            ),
            None,
        )
        if up_rec is None:
            continue
        up_prov = provenance_of(recording_data, up_mod, up_rec)
        if is_aux:
            prov.setdefault("aux", {})[up_mod] = up_prov
        else:
            prov["parent"] = up_prov
    return prov


def merge_provenance(provs: list[dict]) -> dict:
    """One provenance for a group of same-settings runs across recordings: every run name
    any of them carries, sorted, at every level. ``shown`` is the first one's, since the
    settings it stands beside are the same by construction."""
    provs = [p for p in provs if p]
    if not provs:
        return {}
    merged = {
        "runs": sorted({name for p in provs for name in p.get("runs", [])}),
        "shown": dict(provs[0].get("shown") or {}),
    }
    parents = [p["parent"] for p in provs if p.get("parent")]
    if parents:
        merged["parent"] = merge_provenance(parents)
    aux_mods = sorted({m for p in provs for m in (p.get("aux") or {})})
    if aux_mods:
        merged["aux"] = {
            m: merge_provenance([(p.get("aux") or {}).get(m) or {} for p in provs])
            for m in aux_mods
        }
    return merged


def run_label(
    recording,
    module_id: str,
    run_id: str,
    model_names: dict | None = None,
    *,
    class_names: dict | None = None,
) -> str:
    """The label of one named run in ``recording``, experiment runs included at every level —
    a run page's lineage line.

    Raises:
        ValueError: The recording has no such run.
    """
    data = _recording_run_data(recording)
    record = _named_record(data, module_id, run_id, recording)
    return spec_label(
        spec_of(data, module_id, record, getattr(recording, "dir", None)),
        model_names,
        class_names=class_names,
        provenance=provenance_of(data, module_id, record),
    )


def every_n_label(every_n_frames) -> str:
    """`` · every Nth frame`` for a thinned run, empty for every-frame (the common case).

    Appended to every stage's picker label: a run's cadence decides what a downstream
    stage may ask for, so it has to be visible where the upstream run is chosen.
    """
    # A legacy record's unrecoverable value is a per-run sentinel string (``_recover_field``):
    # its cadence is unknown, so it is not shown rather than guessed.
    if not isinstance(every_n_frames, (int, float, type(None))):
        return ""
    stride = int(every_n_frames or 1)
    return f" · every {_ordinal(stride)} frame" if stride > 1 else ""


def teacher_label(teacher_window_s) -> str:
    """`` · first 5 min (teacher)`` for a keypoint teacher run (§26), empty otherwise.

    On the detection label, so every run downstream of it carries it too (``from …``): a
    teacher run is offered in pickers that will refuse it, and this is how a person sees why.
    """
    if teacher_window_s is None:
        return ""
    return f" · {teacher_window.window_label(float(teacher_window_s))} (teacher)"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _own_label(
    module_id: str,
    identity: dict,
    model_names: dict | None = None,
    class_names: dict | None = None,
    shown: dict | None = None,
) -> str:
    names = class_names or {}
    shown = shown or {}
    if module_id == "detection":
        classes = identity.get("prompt_classes")
        return (
            _model_conf_label(
                identity.get("model_id"),
                identity.get("model_variant"),
                identity.get("conf_threshold"),
                identity.get("weights_source"),
                (
                    [class_label(str(c), names) for c in classes]
                    if isinstance(classes, list)
                    else classes
                ),
                model_names,
            )
            + _class_values_label(shown.get("class_conf_thresholds"), names, "{:.2f}")
            + every_n_label(identity.get("every_n_frames"))
            + teacher_label(identity.get(teacher_window.WINDOW_KEY))
        )
    if module_id == "tracking":
        hp = identity.get("hyperparams")
        hp = hp if isinstance(hp, dict) else {}
        conf = hp.get("conf_threshold")
        arenas = identity.get("arena_ids") or []
        class_name = identity.get("class_name")
        label = str(identity.get("tracker_id", "tracker"))
        if class_name:
            label += f" · {class_label(str(class_name), names)}"
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
        align = identity.get("align")
        if align and align != "none":
            label += f" · aligned ({align})"
        return label + every_n_label(identity.get("every_n_frames"))
    if module_id == "keypoints":
        label = _model_variant_label(
            identity.get("model_id"), identity.get("model_variant"), model_names
        )
        if identity.get("reference_digest"):
            label += f" · refs {str(identity['reference_digest'])[:6]}"
        return label + every_n_label(identity.get("every_n_frames"))
    if module_id == "keypoint_detect":
        label = _model_variant_label(
            identity.get("model_id"), identity.get("model_variant"), model_names
        )
        if identity.get("class_name"):
            label += f" · {class_label(str(identity['class_name']), names)}"
        if shown.get("example_set_id"):
            label += f" · examples {shown['example_set_id']}"
        label += _class_values_label(
            shown.get("class_example_sets"), names, "{}", lead=" · examples "
        )
        return label + every_n_label(identity.get("every_n_frames"))
    if module_id == "keypoint_track":
        label = str(identity.get("method") or module_id)
        config = identity.get("config")
        config = config if isinstance(config, dict) else {}
        label += "".join(
            f" · {step}" for step in ("despike", "smooth") if config.get(step)
        )
        points = config.get("track_points")
        if isinstance(points, list):
            label += f" · {len(points)} points"
        return label
    return module_id


def _class_values_label(
    values, class_names: dict, fmt: str, *, lead: str = " · "
) -> str:
    """`` · Intruder 0.20, Resident 0.50`` — a per-class map, each class by its name (§1);
    empty for no map."""
    if not isinstance(values, dict) or not values:
        return ""
    parts = (
        f"{class_label(str(c), class_names)} {fmt.format(v)}"
        for c, v in sorted(values.items(), key=lambda kv: str(kv[0]))
    )
    return lead + ", ".join(parts)


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
    recordings,
    module_id: str,
    model_names: dict | None = None,
    *,
    class_names: dict | None = None,
) -> list[dict]:
    """Distinct ``module_id`` specs across ``recordings`` — ``[{spec, label, recordings}]``
    where ``recordings`` is the ids of those covered by each spec (newest run wins on a
    same-spec tie within a recording). Does *not* require a spec common to all recordings;
    that is what lets a batch cover recordings whose runs were split across launches.
    ``class_names`` labels classes as :func:`spec_label` does.
    """
    entries = (
        (sess.id, _recording_run_data(sess), getattr(sess, "dir", None))
        for sess in recordings
    )
    return _group(entries, module_id, model_names, class_names)


def _group(
    entries,
    module_id: str,
    model_names: dict | None = None,
    class_names: dict | None = None,
) -> list[dict]:
    """:func:`group_specs` over ``(recording_id, recording_data, recording_dir)`` triples —
    the shape both a GUI holding Recordings and a ``validate()`` holding only the experiment
    directory can produce. Each group also carries ``runs``, ``{recording_id: run_id}``, and
    ``provenance``, the experiment runs those per-recording runs came from (its label leads
    with them: two groups can have settings that differ only where a label cannot show), and
    ``created_at``, its newest run's, which a picker breaks a coverage tie by.
    """
    module = _processing_module(module_id)
    groups: dict[tuple, dict] = {}
    for recording_id, data, recording_dir in entries:
        newest: dict[tuple, tuple] = {}  # key -> (spec, created_at, record)
        for record in data.get(module.runs_key, []):
            if record.get("status") != "complete":
                continue
            spec = spec_of(data, module_id, record, recording_dir)
            key = freeze_spec(spec)
            created = record.get("created_at") or ""
            if key not in newest or created > newest[key][1]:
                newest[key] = (spec, created, record)
        for key, (spec, created, record) in newest.items():
            group = groups.setdefault(
                key, {"spec": spec, "runs": {}, "provs": [], "created_at": ""}
            )
            group["runs"][recording_id] = record.get("id")
            group["provs"].append(provenance_of(data, module_id, record))
            group["created_at"] = max(group["created_at"], created)
    result = []
    for g in groups.values():
        provenance = merge_provenance(g["provs"])
        result.append(
            {
                "spec": g["spec"],
                "label": spec_label(
                    g["spec"],
                    model_names,
                    class_names=class_names,
                    provenance=provenance,
                ),
                "recordings": list(g["runs"]),
                "runs": dict(g["runs"]),
                "provenance": provenance,
                "created_at": g["created_at"],
            }
        )
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
    record = _named_record(data, module_id, run_id, recording)
    return spec_of(data, module_id, record, getattr(recording, "dir", None))


def _named_record(data: dict, module_id: str, run_id: str, recording) -> dict:
    module = _processing_module(module_id)
    record = next(
        (r for r in data.get(module.runs_key, []) if r.get("id") == run_id), None
    )
    if record is None:
        run_ref = getattr(recording, "id", recording)
        raise ValueError(f"recording {run_ref!r} has no {module_id} run {run_id!r}")
    return record


def spec_from_experiment_run(experiment, module_id: str, run_id: str) -> dict:
    """The ``module_id`` settings an experiment run (``run_NNN``) made — the run named the way
    a picker row names it, for the CLI ``--*-ref run_NNN`` selectors.

    The spec is read from the run's per-recording runs, and each recording then resolves its
    own run with those settings, the same as a picker row: a recording whose newest run with
    the same settings came from a later launch reads that one.

    Raises:
        ValueError: The run made no complete ``module_id`` run, or its recordings' runs do
            not share one set of settings (a run recorded before a setting was stored, whose
            value cannot be recovered, differs per recording; name one with
            ``RECORDING:run_id`` instead).
        FileNotFoundError: The experiment has no such run.
    """
    from detail.storage import runs as runs_store

    record = runs_store.read_run(experiment.dir, run_id)
    specs: dict[tuple, dict] = {}
    for sid, entry in (record.get("per_recording") or {}).items():
        if entry.get("status") != "complete" or not entry.get("run_id"):
            continue
        recording = experiment.get_recording(sid)
        data = _recording_run_data(recording)
        made = next(
            (
                r
                for r in data.get(_processing_module(module_id).runs_key, [])
                if r.get("id") == entry["run_id"]
            ),
            None,
        )
        if made is None:
            continue
        spec = spec_of(data, module_id, made, getattr(recording, "dir", None))
        specs.setdefault(freeze_spec(spec), spec)
    if not specs:
        raise ValueError(
            f"run {run_id!r} made no complete {module_id} run; pick a {module_id} run "
            "(the run list shows each run's stage)"
        )
    if len(specs) > 1:
        raise ValueError(
            f"run {run_id!r}'s {module_id} runs do not share one set of settings across "
            "its recordings; name one of them as RECORDING:run_id instead"
        )
    return next(iter(specs.values()))


def resolve_analysis_selection(
    module, experiment_dir, recording_ids: list[str], config: dict
) -> dict[str, str]:
    """``{recording_id: run_id}`` — the upstream run an analysis reads in each recording.

    Driven by the module's ``selects = (upstream_module, ref_field)`` declaration, so every
    analysis resolves its upstream the same way. ``<upstream>_overrides``, a
    ``{recording_id: run_id}`` map, comes first: each named recording reads exactly that run
    (:func:`resolve_overrides`), and the rest resolve from four inputs, in order:

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
        ValueError: A recording has no complete matching run, the default is ambiguous, or
            an override names a recording outside the selection or a run that is missing or
            incomplete.
    """
    from detail.storage import experiment as experiment_store
    from detail.storage import recording as recording_store

    up_mod, ref_field = module.selects
    entries = []
    for sid in recording_ids:
        recording_dir = experiment_store.recording_path(experiment_dir, sid)
        entries.append(
            (sid, recording_store.read_recording(recording_dir), recording_dir)
        )
    overrides = resolve_overrides(
        entries, up_mod, config.get(f"{up_mod}_overrides") or {}
    )
    rest = [entry for entry in entries if entry[0] not in overrides]
    mapping = (
        _resolve_selection(rest, up_mod, ref_field, config, [e[0] for e in rest])
        if rest
        else {}
    )
    mapping.update(overrides)
    if overrides and up_mod == "embedding" and len(mapping) > 1:
        from detail.api.run_preview import validate_comparability

        validate_comparability(
            [(recording_dir, mapping[sid]) for sid, _data, recording_dir in entries]
        )
    return {sid: mapping[sid] for sid in recording_ids}


def resolve_overrides(entries, up_mod: str, overrides: dict) -> dict[str, str]:
    """``overrides`` checked against ``(recording_id, recording_data, recording_dir)``
    ``entries``: every key a selected recording, every run present and complete in it (§6.5
    *per-recording overrides*). An explicit choice, so nothing here falls back.

    Raises:
        ValueError: A key is not a selected recording, or its run is missing or not
            complete.
    """
    data_by_id = {sid: data for sid, data, _recording_dir in entries}
    runs_key = _processing_module(up_mod).runs_key
    resolved = {}
    for sid, run_id in overrides.items():
        if sid not in data_by_id:
            raise ValueError(
                f"the {up_mod} run chosen for recording {sid!r} names a recording that is "
                "not selected; select it, or drop its choice"
            )
        record = {r.get("id"): r for r in data_by_id[sid].get(runs_key, [])}.get(run_id)
        if record is None:
            raise ValueError(
                f"recording {sid!r} has no {up_mod} run {run_id!r} (chosen for it by name)"
            )
        if record.get("status") != "complete":
            raise ValueError(
                f"{up_mod} run {run_id!r} chosen for recording {sid!r} is "
                f"{record.get('status')!r}, not complete"
            )
        resolved[sid] = str(run_id)
    return resolved


def _resolve_selection(
    entries, up_mod: str, ref_field: str, config: dict, recording_ids: list[str]
) -> dict[str, str]:
    """:func:`resolve_analysis_selection`'s four inputs over ``entries`` — the recordings no
    override names."""
    runs_key = _processing_module(up_mod).runs_key
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


def _resolve_aux(
    recording_dir, recording_data: dict, aux_mod: str, spec: dict, parent_id
) -> str | None:
    """This recording's ``aux_mod`` run matching ``spec`` and descending from the lineage
    parent run ``parent_id`` (§26: the chain stays linear), newest on a tie; else None.
    """
    aux_module = _processing_module(aux_mod)
    parent_field = aux_module.lineage_ref[1] if aux_module.lineage_ref else None
    narrowed = dict(recording_data)
    narrowed[aux_module.runs_key] = [
        r
        for r in recording_data.get(aux_module.runs_key, [])
        if parent_field is None or r.get(parent_field) == parent_id
    ]
    return resolve_spec_in_recording(narrowed, aux_mod, spec, recording_dir)
