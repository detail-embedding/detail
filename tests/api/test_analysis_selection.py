"""An analysis reads each recording's own upstream run, matched on settings — never on its id.

**The class.** A per-recording run id (``emb_01``, ``track_02``) is a counter
(``api/recording.py::_next_run_id``): max + 1 of whatever that recording already holds, failed
runs included. So it drifts between recordings for reasons that have nothing to do with settings —
a retry, a deleted run, a recording processed in a later launch — and it collides across runs that
have nothing in common. Any consumer that picks an upstream run by intersecting ids across a
selection is wrong in both directions at once:

- **false negative**: the same settings under ``emb_01`` in one recording and ``emb_02`` in the next
  share no id, so the selection is refused with every setting identical;
- **false merge**: two unrelated runs that both happen to be ``emb_01`` are offered as one choice,
  and a result silently mixes them.

**How it hid.** Every fixture made one run per recording, so every id was ``emb_01`` and the
intersection was always the right answer. Issue #547 found it on an experiment where one
recording had been retried inside the batch (``emb_02`` after a failed ``emb_01``) and one
had been added later: pose clustering refused every recording, while
``run_spec.group_specs`` over the same recordings reported one embedding group. The one
retry alone was enough — the rest, all from one launch, would have been refused too.

**The rule.** An analysis names its upstream in ``AnalysisModule.selects``; its ``validate()``
resolves the selection through ``AnalysisModule.resolve_selection``
(``run_spec.resolve_analysis_selection``) to a ``{recording_id: run_id}`` map, and every reader
asks ``core.run_selection.run_for_recording`` which run a recording used. A GUI picker offers
settings with their coverage (``gui.modules._common.UpstreamSpecPicker``), not shared ids.

**The guard**, data-driven over ``ANALYSIS_MODULES`` so a new module is covered the day it lands:

1. every surfaced module that ``selects`` an upstream is either exercised here or named in
   ``NOT_YET_BY_SETTINGS`` with its reason — a claim checked against the live registry;
2. each exercised module resolves the issue's cohort (same settings, different ids) to each
   recording's own run, by default and from an explicit spec;
3. and refuses the false merge (different settings, one shared id) instead of treating it as one;
4. a source scan over the GUI for the id-intersection helpers, with an allow-list whose every
   entry must still be a caller, and a negative case;
5. a *list* of settings (``<upstream>_specs`` — two strains, each detected by its own model)
   resolves each recording to its run of the one listed setting it has, refuses a recording
   with none or several, and refuses pooled embeddings whose recipes differ — the check equal
   specs used to make implicitly.

Hidden capabilities (§25) keep the old helper deliberately; the scan's allow-list says so per file.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from detail import surfaces
from detail.api import run_spec
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.core import run_selection
from detail.executor.runners import (
    ANALYSIS_MODULES,
    PROCESSING_MODULES,
    import_runner_modules,
)
from detail.storage import recording as recording_store

import_runner_modules(PIPELINE_RUNNERS)

REPO = Path(__file__).resolve().parents[2]

#: The minimal config each exercised module needs beyond its selection. Keyed by module id.
EXERCISED: dict[str, dict] = {
    "location_export": {},
    "temporal_clustering": {"method": "dtw_finch", "track_id": 0},
}

#: Surfaced modules that still pick their upstream by id. A claim: each must be registered,
#: surfaced and selecting something, or the entry is stale.
NOT_YET_BY_SETTINGS: dict[str, str] = {
    "query": "#547 PR2 — its embedding run also names the exemplar recordings, so its "
    "resolution covers search and exemplar recordings alike",
}

#: Prefix of the upstream's run ids, for building the cohort.
_PREFIX = {"detection": "det", "tracking": "track", "embedding": "emb"}


def _surfaced_selecting() -> dict[str, tuple[str, str]]:
    return {
        module_id: module.selects
        for module_id, module in ANALYSIS_MODULES.items()
        if module.selects is not None and surfaces.is_surfaced(module_id)
    }


def test_every_surfaced_selecting_module_is_exercised_or_named():
    selecting = _surfaced_selecting()
    unaccounted = sorted(set(selecting) - set(EXERCISED) - set(NOT_YET_BY_SETTINGS))
    assert not unaccounted, (
        f"{unaccounted} select an upstream run and are neither exercised by this guard nor "
        "named in NOT_YET_BY_SETTINGS with a reason"
    )


def test_every_claim_names_a_live_surfaced_selecting_module():
    """``NOT_YET_BY_SETTINGS`` excuses a surfaced module, so it must name one. ``EXERCISED`` may
    also name a hidden module (§25): hidden is not dead, and dropping its coverage here would let
    its selection rot until the day it is switched back on."""
    selecting = _surfaced_selecting()
    registered_selecting = {
        module_id
        for module_id, module in ANALYSIS_MODULES.items()
        if module.selects is not None
    }
    stale = sorted(
        [m for m in EXERCISED if m not in registered_selecting]
        + [m for m in NOT_YET_BY_SETTINGS if m not in selecting]
    )
    assert not stale, f"{stale} are not live modules that select an upstream run"
    assert not set(EXERCISED) & set(NOT_YET_BY_SETTINGS)


# --------------------------------------------------------------------------- #
# The cohort
# --------------------------------------------------------------------------- #
def _recording(experiment_dir: Path, recording_id: str) -> Path:
    rec = experiment_dir / "recordings" / recording_id
    for d in ("video/timestamps/generated", "detections", "tracking", "embeddings"):
        (rec / d).mkdir(parents=True, exist_ok=True)
    (rec / "recording.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": recording_id,
                "name": recording_id,
                "created_at": "2026-01-01T00:00:00Z",
                "video": {"resolution": [640, 480], "files": []},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return rec


def _chain(rec: Path, make_embedding_run, suffix: str, settings: str, status: str):
    """det_/track_/emb_<suffix> — one full lineage, its settings named by ``settings``.

    The one knob that differs is the tracker's, so it changes the tracking spec *and*, through
    lineage, the embedding spec: a false merge at either level uses the same cohort. ``M`` is
    ``B``'s chain embedded with another model — a recipe that must never pool with ``A``'s.
    """
    max_missed, model_id = _SETTINGS[settings]
    data = recording_store.read_recording(rec)
    det = PROCESSING_MODULES["detection"].run_record(
        f"det_{suffix}", {"model_id": "sam3", "prompt": None}, status
    )
    track = PROCESSING_MODULES["tracking"].run_record(
        f"track_{suffix}",
        {
            "detection_run_id": f"det_{suffix}",
            "tracker_id": "anchor_tracker",
            "hyperparams": {"max_missed": max_missed},
        },
        status,
    )
    for record in (det, track):
        record["created_at"] = f"2026-01-0{suffix[-1]}T00:00:00Z"
    data.setdefault("detection_runs", []).append(det)
    data.setdefault("tracking_runs", []).append(track)
    recording_store.write_recording(rec, data)

    rng = np.random.default_rng(int(suffix))
    n = 40
    make_embedding_run(
        rec,
        rng.normal(size=(n, 4)).astype(np.float32),
        np.arange(n, dtype=np.int64),
        np.arange(n, dtype=np.int64) * 17_000_000,
        embed_run_id=f"emb_{suffix}",
        tracking_run_id=f"track_{suffix}",
        status=status,
        model_id=model_id,
    )


#: settings name -> (the tracker's max_missed, the embedding model).
_SETTINGS = {
    "A": (5, "dinov3_convnext_base"),
    "B": (9, "dinov3_convnext_base"),
    "M": (9, "dinov3_vits16"),
}


@pytest.fixture
def cohort(experiment_dir, make_embedding_run):
    """Build recordings from ``{recording_id: [(suffix, settings, status), ...]}``."""

    def _build(layout: dict) -> Path:
        for recording_id, runs in layout.items():
            rec = _recording(experiment_dir, recording_id)
            for suffix, settings, status in runs:
                _chain(rec, make_embedding_run, suffix, settings, status)
        return experiment_dir

    return _build


#: The issue's shape: the same settings, but one recording's first attempt failed and the retry
#: minted the next id.
RETRIED = {
    "recording_001": [("01", "A", "complete")],
    "recording_002": [("01", "A", "failed"), ("02", "A", "complete")],
}

#: The false merge: one shared id, different settings.
COLLIDING = {
    "recording_001": [("01", "A", "complete")],
    "recording_002": [("01", "B", "complete")],
}


def _config(module_id: str, **selection) -> dict:
    return {
        "source_recording_ids": ["recording_001", "recording_002"],
        **EXERCISED[module_id],
        **selection,
    }


def _selection(module_id: str, validated: dict) -> dict:
    _upstream, ref_field = ANALYSIS_MODULES[module_id].selects
    return run_selection.runs_by_recording(
        validated, validated["source_recording_ids"], ref_field
    )


@pytest.mark.parametrize("module_id", sorted(EXERCISED))
def test_the_same_settings_under_different_ids_resolve_to_each_recordings_own_run(
    cohort, module_id
):
    """The discriminating test from #547: this raised "share no completed embedding run"."""
    experiment_dir = cohort(RETRIED)
    module = ANALYSIS_MODULES[module_id]
    upstream, ref_field = module.selects
    prefix = _PREFIX[upstream]

    validated = module.validate(experiment_dir, _config(module_id))

    assert _selection(module_id, validated) == {
        "recording_001": f"{prefix}_01",
        "recording_002": f"{prefix}_02",
    }
    # The ids disagree, so there is no single id to state — and the job carries the map.
    assert validated[ref_field] is None
    job = module.build_job("an_01", validated)
    assert run_selection.run_for_recording(job, "recording_002", ref_field) == (
        f"{prefix}_02"
    )


@pytest.mark.parametrize("module_id", sorted(EXERCISED))
def test_a_spec_named_from_one_recording_resolves_in_every_recording(cohort, module_id):
    """``--embed-ref recording_001:emb_01`` and the GUI picker take this path."""
    experiment_dir = cohort(RETRIED)
    module = ANALYSIS_MODULES[module_id]
    upstream, _ref_field = module.selects
    prefix = _PREFIX[upstream]
    spec = _spec(experiment_dir, "recording_001", upstream, f"{prefix}_01")

    validated = module.validate(
        experiment_dir, _config(module_id, **{f"{upstream}_spec": spec})
    )

    assert _selection(module_id, validated) == {
        "recording_001": f"{prefix}_01",
        "recording_002": f"{prefix}_02",
    }


@pytest.mark.parametrize("module_id", sorted(EXERCISED))
def test_different_settings_sharing_an_id_are_not_one_choice(cohort, module_id):
    """The false merge. Matching on the shared ``*_01`` would have resolved this silently."""
    experiment_dir = cohort(COLLIDING)
    module = ANALYSIS_MODULES[module_id]
    upstream, _ref_field = module.selects
    prefix = _PREFIX[upstream]

    with pytest.raises(ValueError, match="cover every selected recording"):
        module.validate(experiment_dir, _config(module_id))

    spec = _spec(experiment_dir, "recording_001", upstream, f"{prefix}_01")
    with pytest.raises(ValueError, match="'recording_002' has no complete"):
        module.validate(
            experiment_dir, _config(module_id, **{f"{upstream}_spec": spec})
        )


@pytest.mark.parametrize("upstream", ["tracking", "embedding"])
def test_the_picker_groups_offer_settings_not_ids(cohort, upstream):
    """What the GUI offers: one choice covering both when only the ids differ, two when only
    the settings do."""
    retried = _groups(cohort(RETRIED), upstream)
    assert [sorted(g["recordings"]) for g in retried] == [
        ["recording_001", "recording_002"]
    ]


@pytest.mark.parametrize("upstream", ["tracking", "embedding"])
def test_the_picker_groups_split_a_shared_id_with_different_settings(cohort, upstream):
    colliding = _groups(cohort(COLLIDING), upstream)
    assert sorted(len(g["recordings"]) for g in colliding) == [1, 1]


def test_an_explicit_id_is_that_id_in_every_recording(cohort):
    """The literal form still means what it says — and refuses a recording whose run of that
    id did not complete, rather than reading a failed one."""
    experiment_dir = cohort(RETRIED)
    module = ANALYSIS_MODULES["temporal_clustering"]
    with pytest.raises(ValueError, match="'failed', not complete"):
        module.validate(
            experiment_dir, _config("temporal_clustering", embed_run_id="emb_01")
        )


def test_several_full_coverage_settings_are_refused_not_sorted(cohort):
    """Tenet 9: the default must not depend on which set of settings sorts first."""
    experiment_dir = cohort(
        {
            "recording_001": [("01", "A", "complete"), ("02", "B", "complete")],
            "recording_002": [("01", "A", "complete"), ("02", "B", "complete")],
        }
    )
    with pytest.raises(ValueError, match="share 2 different embedding settings"):
        ANALYSIS_MODULES["temporal_clustering"].validate(
            experiment_dir, _config("temporal_clustering")
        )


# --------------------------------------------------------------------------- #
# Several settings: recordings whose upstream came through different chains
# --------------------------------------------------------------------------- #
#: Two strains, each detected by its own model: no one set of settings covers both, and the
#: second recording's run of its settings is a retry, so the ids disagree as well.
SPLIT = {
    "recording_001": [("01", "A", "complete")],
    "recording_002": [("01", "B", "failed"), ("02", "B", "complete")],
}


def _specs(experiment_dir, upstream: str, *refs: tuple[str, str]) -> list[dict]:
    return [_spec(experiment_dir, rid, upstream, run_id) for rid, run_id in refs]


@pytest.mark.parametrize("module_id", sorted(EXERCISED))
def test_several_settings_pool_recordings_no_one_setting_covers(cohort, module_id):
    """The case the list exists for: refused by default, resolved per recording when each
    recording's settings are named."""
    experiment_dir = cohort(SPLIT)
    module = ANALYSIS_MODULES[module_id]
    upstream, ref_field = module.selects
    prefix = _PREFIX[upstream]
    with pytest.raises(ValueError, match="cover every selected recording"):
        module.validate(experiment_dir, _config(module_id))

    specs = _specs(
        experiment_dir,
        upstream,
        ("recording_001", f"{prefix}_01"),
        ("recording_002", f"{prefix}_02"),
    )
    validated = module.validate(
        experiment_dir, _config(module_id, **{f"{upstream}_specs": specs})
    )

    expected = {"recording_001": f"{prefix}_01", "recording_002": f"{prefix}_02"}
    assert _selection(module_id, validated) == expected
    job = module.build_job("an_01", validated)
    assert run_selection.run_for_recording(job, "recording_002", ref_field) == (
        f"{prefix}_02"
    )


@pytest.mark.parametrize("module_id", sorted(EXERCISED))
def test_a_recording_with_runs_of_two_listed_settings_is_refused(cohort, module_id):
    """Tenet 9: list order must not pick which run a recording reads."""
    experiment_dir = cohort(
        {
            "recording_001": [("01", "A", "complete"), ("02", "B", "complete")],
            "recording_002": [("01", "B", "complete")],
        }
    )
    module = ANALYSIS_MODULES[module_id]
    upstream, _ref_field = module.selects
    prefix = _PREFIX[upstream]
    specs = _specs(
        experiment_dir,
        upstream,
        ("recording_001", f"{prefix}_01"),
        ("recording_002", f"{prefix}_01"),
    )
    with pytest.raises(ValueError, match="'recording_001' has a run of more than one"):
        module.validate(
            experiment_dir, _config(module_id, **{f"{upstream}_specs": specs})
        )


@pytest.mark.parametrize("module_id", sorted(EXERCISED))
def test_a_recording_with_none_of_the_listed_settings_is_refused(cohort, module_id):
    """Listing the same settings twice is one setting, and it still leaves recording_002
    with nothing to read."""
    experiment_dir = cohort(SPLIT)
    module = ANALYSIS_MODULES[module_id]
    upstream, _ref_field = module.selects
    prefix = _PREFIX[upstream]
    specs = _specs(
        experiment_dir,
        upstream,
        ("recording_001", f"{prefix}_01"),
        ("recording_001", f"{prefix}_01"),
    )
    with pytest.raises(ValueError, match="'recording_002' has no complete .* any of"):
        module.validate(
            experiment_dir, _config(module_id, **{f"{upstream}_specs": specs})
        )


def test_one_spec_and_a_list_together_are_refused(cohort):
    experiment_dir = cohort(SPLIT)
    spec = _spec(experiment_dir, "recording_001", "embedding", "emb_01")
    with pytest.raises(ValueError, match="not both"):
        ANALYSIS_MODULES["temporal_clustering"].validate(
            experiment_dir,
            _config(
                "temporal_clustering", embedding_spec=spec, embedding_specs=[spec, spec]
            ),
        )


@pytest.mark.parametrize(
    "module_id",
    sorted(m for m in EXERCISED if ANALYSIS_MODULES[m].selects[0] == "embedding"),
)
def test_pooled_embeddings_must_share_a_recipe(cohort, module_id):
    """Equal specs used to vouch for the recipe; a list of specs does not, so the resolver
    checks it. Neither clustering module calls the comparability check itself, so without this
    each would pool vectors from two different models into one vocabulary — measured: with the
    resolver's check removed, both validate the mixed-model cohort and launch."""
    experiment_dir = cohort(
        {
            "recording_001": [("01", "A", "complete")],
            "recording_002": [("01", "M", "complete")],
        }
    )
    specs = _specs(
        experiment_dir,
        "embedding",
        ("recording_001", "emb_01"),
        ("recording_002", "emb_01"),
    )
    with pytest.raises(ValueError, match="not comparable"):
        ANALYSIS_MODULES[module_id].validate(
            experiment_dir, _config(module_id, embedding_specs=specs)
        )


def _spec(experiment_dir, recording_id, module_id, run_id) -> dict:
    rec = experiment_dir / "recordings" / recording_id
    data = recording_store.read_recording(rec)
    record = next(
        r for r in data[PROCESSING_MODULES[module_id].runs_key] if r["id"] == run_id
    )
    return run_spec.spec_of(data, module_id, record, rec)


def _groups(experiment_dir, module_id) -> list[dict]:
    recordings = []
    for recording_id in ("recording_001", "recording_002"):
        rec = experiment_dir / "recordings" / recording_id
        data = recording_store.read_recording(rec)
        recordings.append(SimpleNamespace(**{**data, "id": recording_id, "dir": rec}))
    return run_spec.group_specs(recordings, module_id)


# --------------------------------------------------------------------------- #
# The source scan: no surfaced picker intersects ids
# --------------------------------------------------------------------------- #
_INTERSECTS_IDS = re.compile(r"(?<!def )\bcommon_complete_run_(?:ids|items)\s*\(")

#: Files that still offer shared ids, each with why. Every entry must still be a caller.
ALLOWED_ID_PICKERS: dict[str, str] = {
    "detail/gui/modules/_common.py": "the helpers' own definitions",
    "detail/gui/modules/analysis_query.py": "#547 PR2",
}


def _intersects_ids(source: str) -> bool:
    return bool(_INTERSECTS_IDS.search(source))


def _gui_sources() -> dict[str, str]:
    return {
        path.relative_to(REPO).as_posix(): path.read_text(encoding="utf-8-sig")
        for path in sorted((REPO / "detail" / "gui").rglob("*.py"))
    }


def test_no_other_gui_surface_picks_an_upstream_by_shared_id():
    offenders = sorted(
        rel
        for rel, source in _gui_sources().items()
        if _intersects_ids(source) and rel not in ALLOWED_ID_PICKERS
    )
    assert not offenders, (
        f"{offenders} offer upstream runs by the ids every selected recording shares; offer "
        "settings with their coverage instead (gui.modules._common.UpstreamSpecPicker)"
    )


def test_every_allowed_id_picker_is_still_one():
    sources = _gui_sources()
    stale = sorted(
        rel for rel in ALLOWED_ID_PICKERS if not _intersects_ids(sources.get(rel, ""))
    )
    assert (
        not stale
    ), f"{stale} no longer intersect ids; remove them from the allow-list"


def test_the_id_picker_scan_would_catch_one():
    assert _intersects_ids("ids = common_complete_run_ids(recordings, 'embedding')")
    assert _intersects_ids("items = common_complete_run_items(\n    recs, 'embedding')")
    assert _intersects_ids(
        "from x import y; y = _common.common_complete_run_items (r, m)"
    )
    assert not _intersects_ids("def common_complete_run_ids(recordings, module):")
    assert not _intersects_ids("spec_groups(recordings, 'embedding')")


# --------------------------------------------------------------------------- #
# A run recorded before the map
# --------------------------------------------------------------------------- #
def _legacy_state() -> dict:
    """A state as written before ``embed_run_ids`` existed: the one id, meant everywhere."""
    return {
        "embed_run_id": "emb_01",
        "track_id": 0,
        "method": "finch",
        "source_recording_ids": ["recording_001", "recording_002"],
    }


@pytest.mark.parametrize("module_id", ["temporal_clustering"])
def test_a_run_from_before_the_map_reads_its_one_id_in_every_recording(module_id):
    from detail.storage import temporal_runs

    state_class = {
        "temporal_clustering": temporal_runs.TemporalClusteringState,
    }[module_id]
    state = state_class.from_dict(_legacy_state())
    job = ANALYSIS_MODULES[module_id].job_class.from_dict(
        {**_legacy_state(), "job_id": "an_01"}
    )

    for source in (state, job, _legacy_state()):
        assert run_selection.runs_by_recording(
            source, ["recording_001", "recording_002"]
        ) == {"recording_001": "emb_01", "recording_002": "emb_01"}
    assert state.embed_run_for("recording_002") == "emb_01"
    assert job.embed_run_for("recording_002") == "emb_01"


def test_the_map_wins_over_the_single_id_when_both_are_recorded():
    source = {"embed_run_id": "emb_01", "embed_run_ids": {"recording_002": "emb_02"}}
    assert run_selection.run_for_recording(source, "recording_002") == "emb_02"
    # A recording outside the map is outside the run — not quietly given the single id.
    assert run_selection.run_for_recording(source, "recording_009") is None
