"""Batch run mode (§6.5) — a module that runs once over the whole subset, not per recording.

Exercised through a tiny fake batch module, so the mode stays pinned now that derive (the
only real batch module) has been removed. The key contract: RunOrchestrator calls ``run_batch``
exactly once (never the per-recording path), and the run finalizes over the whole subset.
"""

import pytest

from detail.api.experiment import Experiment, RunOrchestrator
from detail.core.dataclasses import _JOB_TAGS, _JOB_TYPES
from detail.executor.runners import (
    JOB_RUNNERS,
    PROCESSING_MODULES,
    BatchProcessingModule,
    _BatchNoJob,
    register_processing_module,
)
from detail.storage import runs as runs_store

# Fake batch modules the tests register share this id prefix so the fixture can drop
# exactly them — not the real modules that ``run_processing`` registers during a test.
_FAKE_PREFIX = "_fake_batch"


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """Unregister only the *fake* batch modules a test registered (the registries are
    process globals; a leak would collide with the next run's ids). Real modules that
    ``import_runner_modules`` registers during a test must stay."""
    yield
    for module_id in [m for m in PROCESSING_MODULES if m.startswith(_FAKE_PREFIX)]:
        PROCESSING_MODULES.pop(module_id, None)
        JOB_RUNNERS.pop(module_id, None)
        _JOB_TYPES.pop(module_id, None)
    # _BatchNoJob's job class is shared by every batch module (derive was one until it was
    # removed), so a fake registration overwrote its _JOB_TAGS entry; re-point it at whatever
    # real batch module still owns it (or drop it if none remain) so the registry stays sound.
    owners = [mid for mid, cls in _JOB_TYPES.items() if cls is _BatchNoJob]
    if owners:
        _JOB_TAGS[_BatchNoJob] = owners[0]
    else:
        _JOB_TAGS.pop(_BatchNoJob, None)


def _register_fake_batch(module_id, calls, *, fail=False):
    class _FakeBatch(BatchProcessingModule):
        id_prefix = "emb"
        runs_key = "embedding_runs"
        upstream = "embedding"

        # A batch module writes one output per recording, so it must say where — the
        # per-recording *job* seams are dead for it, this one is not (delete_run needs it).
        def output_path(self, run_id):
            return f"embeddings/{run_id}/"

        def estimate_output_bytes(self, recording_dir, recording_data, config):
            return 0

        def validate_batch(self, experiment_root, recording_ids, config):
            calls.append(("validate_batch", tuple(recording_ids)))
            return {**config, "normalized": True}

        def run_batch(self, experiment, recording_ids, config, emit_progress, cancel):
            calls.append(("run_batch", tuple(recording_ids), dict(config)))
            emit_progress("deriving", 0.5, recording_ids[0])  # a mid-run tick
            if fail:
                raise ValueError("boom")
            return "emb_09"

    _FakeBatch.module_id = module_id
    return register_processing_module(_FakeBatch)


def _seed_run(proj, run_id, module_id, recording_ids, config):
    from detail.core.utils import utc_timestamp

    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": module_id,
            "recording_ids": list(recording_ids),
            "config": config,
            "per_recording": {},
            "status": "running",
            "created_at": utc_timestamp(),
        },
    )


def test_batch_module_runs_once_over_the_subset(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    for sid in ("s1", "s2", "s3"):
        proj.create_recording(sid)
    calls: list = []
    _register_fake_batch("_fake_batch_ok", calls)
    _seed_run(proj, "run_001", "_fake_batch_ok", ["s1", "s2", "s3"], {"transform": "x"})

    events: list = []
    RunOrchestrator(
        proj,
        "run_001",
        "_fake_batch_ok",
        ["s1", "s2", "s3"],
        {"transform": "x"},
        events.append,
        "",
    ).run()

    # run_batch fired exactly once, over the whole subset — never the per-recording path.
    batch_calls = [c for c in calls if c[0] == "run_batch"]
    assert len(batch_calls) == 1
    assert batch_calls[0][1] == ("s1", "s2", "s3")
    record = runs_store.read_run(proj.dir, "run_001")
    assert record["status"] == "complete"
    # every recording shares the produced run id
    assert {v["run_id"] for v in record["per_recording"].values()} == {"emb_09"}
    # the mid-run tick moved the aggregate bar (an intermediate recordings_done)
    assert any(0 < e.recordings_done < e.recordings_total for e in events)


def test_batch_progress_moves_within_a_single_recording(tmp_path):
    """A single-recording batch must show smooth movement, not sit at 0 until the end —
    the within-run fraction rides `inner`, so the GUI fold advances. Regression: the
    bar was frozen at 0% for a one-recording derive (inner was None)."""
    from detail.gui.workers import processing_progress_fraction

    proj = Experiment.create("proj", tmp_path)
    proj.create_recording("s1")

    def _register_ticking(module_id, calls):
        class _Ticking(BatchProcessingModule):
            id_prefix = "emb"
            runs_key = "embedding_runs"
            upstream = "embedding"

            def output_path(self, run_id):
                return f"embeddings/{run_id}/"

            def estimate_output_bytes(self, recording_dir, recording_data, config):
                return 0

            def validate_batch(self, experiment_root, recording_ids, config):
                return dict(config)

            def run_batch(
                self, experiment, recording_ids, config, emit_progress, cancel
            ):
                emit_progress("loading", 0.0, "reading")
                emit_progress("fitting", 0.1, "fitting")
                for r in range(0, 101, 20):
                    emit_progress("writing", 0.15 + 0.85 * (r / 100), recording_ids[0])
                return "emb_09"

        _Ticking.module_id = module_id
        return register_processing_module(_Ticking)

    _register_ticking("_fake_batch_tick", [])
    _seed_run(proj, "run_003", "_fake_batch_tick", ["s1"], {})
    events: list = []
    RunOrchestrator(
        proj, "run_003", "_fake_batch_tick", ["s1"], {}, events.append, ""
    ).run()

    fracs = [processing_progress_fraction(e)[0] for e in events]
    assert fracs == sorted(fracs)  # monotonic
    assert any(0.0 < f < 1.0 for f in fracs)  # moves within the single recording
    assert max(fracs) == 1.0
    # the phase label rides through (loading → fitting → writing)
    messages = {e.inner.message for e in events if e.inner is not None}
    assert {"loading", "fitting", "writing"} <= messages
    # and the label never overshoots "(1/1)"
    labels = [processing_progress_fraction(e)[1] for e in events]
    assert not any("(2/1)" in lab for lab in labels)


def test_batch_failure_fails_the_whole_run(tmp_path):
    proj = Experiment.create("proj", tmp_path)
    for sid in ("s1", "s2"):
        proj.create_recording(sid)
    calls: list = []
    _register_fake_batch("_fake_batch_fail", calls, fail=True)
    _seed_run(proj, "run_002", "_fake_batch_fail", ["s1", "s2"], {})
    RunOrchestrator(
        proj, "run_002", "_fake_batch_fail", ["s1", "s2"], {}, lambda p: None, ""
    ).run()
    record = runs_store.read_run(proj.dir, "run_002")
    assert record["status"] == "failed"
    assert all(v["status"] == "failed" for v in record["per_recording"].values())
    assert "boom" in record["per_recording"]["s1"]["error"]


def test_run_processing_validates_batch_once(tmp_path, monkeypatch):
    proj = Experiment.create("proj", tmp_path)
    for sid in ("s1", "s2"):
        proj.create_recording(sid)
    calls: list = []
    _register_fake_batch("_fake_batch_validate", calls)
    # Stub the coordinator spawn (don't leave a detached subprocess); run_processing
    # validates up front, and we assert that branch fired once over the whole subset,
    # not per-recording.
    monkeypatch.setattr("detail.api.experiment.spawn_coordinator", lambda *a, **k: 4321)
    proj.run_processing("_fake_batch_validate", ["s1", "s2"], {"transform": "x"})
    validate_calls = [c for c in calls if c[0] == "validate_batch"]
    assert len(validate_calls) == 1
    assert validate_calls[0][1] == ("s1", "s2")
