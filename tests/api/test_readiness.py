"""Setup-readiness signals for the Home onboarding panel (§1).

The signals are now *derived from the live checker framework*: ``readiness_signals`` runs the
checks and ``signals_from`` maps their results to the five booleans, plus two cheap workspace
probes (imported catalog, experiment listing). These tests pin the check results + probes so
the mapping is deterministic on any machine/CI, plus one real-wiring smoke test.
"""

import pytest

from detail.api.readiness import ReadinessSignals, readiness_signals, signals_from
from detail.core.dataclasses import CheckResult, CheckStatus


def _res(check_id, ok):
    return CheckResult(check_id, CheckStatus.OK if ok else CheckStatus.WARN, "")


def _pin(
    monkeypatch,
    *,
    gpu,
    hf,
    sam3_weights,
    sam3_backend,
    imported,
    experiments,
    tool_runs=(),
):
    results = [
        _res("cuda", gpu),
        _res("hf_token", hf),
        _res("sam3_weights", sam3_weights),
        _res("sam3_backend", sam3_backend),
    ]
    monkeypatch.setattr("detail.api.readiness.run_checks", lambda **k: results)
    # ``imported`` is catalog ids (detection models) or records (to give one a task).
    records = [r if isinstance(r, dict) else {"id": r} for r in imported]
    monkeypatch.setattr(
        "detail.api.readiness.imported_store.discover",
        lambda: [r["id"] for r in records],
    )
    monkeypatch.setattr(
        "detail.api.readiness.imported_store.list_records", lambda: records
    )
    monkeypatch.setattr(
        "detail.api.readiness.locations.list_experiment_dirs", lambda: experiments
    )
    monkeypatch.setattr(
        "detail.api.readiness.tool_runs.list_tool_runs",
        lambda tool_id=None: [r for r in tool_runs if r["tool_id"] == tool_id],
    )


def test_readiness_signals_real_call_returns_booleans():
    # No mocks: exercise the real wiring (run_checks in-process) on whatever the host state
    # is, asserting shape/types rather than host-specific values.
    signals = readiness_signals()
    assert isinstance(signals, ReadinessSignals)
    for value in vars(signals).values():
        assert isinstance(value, bool)


def test_readiness_signals_all_false_on_a_fresh_machine(monkeypatch):
    _pin(
        monkeypatch,
        gpu=False,
        hf=False,
        sam3_weights=False,
        sam3_backend=False,
        imported=[],
        experiments=[],
    )
    assert readiness_signals() == ReadinessSignals(
        gpu=False,
        hf_token=False,
        sam3=False,
        sam3_backend=False,
        sam3_weights=False,
        detection_model=False,
        has_experiment=False,
        quick_segment_run=False,
        distilled_model=False,
    )


def test_readiness_signals_all_true_when_fully_set_up(monkeypatch):
    _pin(
        monkeypatch,
        gpu=True,
        hf=True,
        sam3_weights=True,
        sam3_backend=True,
        imported=[{"id": "m1", "source": "distilled"}],
        experiments=["exp1"],
        tool_runs=[{"tool_id": "quick_segment", "status": "complete"}],
    )
    assert readiness_signals() == ReadinessSignals(
        gpu=True,
        hf_token=True,
        sam3=True,
        sam3_backend=True,
        sam3_weights=True,
        detection_model=True,
        has_experiment=True,
        quick_segment_run=True,
        distilled_model=True,
    )


def test_sam3_needs_both_weights_and_backend(monkeypatch):
    # The honesty fix: weights alone must not report SAM 3 ready — the sam3 package must
    # also import, or a run would crash at `import sam3`.
    _pin(
        monkeypatch,
        gpu=True,
        hf=True,
        sam3_weights=True,
        sam3_backend=False,
        imported=[],
        experiments=[],
    )
    assert readiness_signals().sam3 is False
    _pin(
        monkeypatch,
        gpu=True,
        hf=True,
        sam3_weights=False,
        sam3_backend=True,
        imported=[],
        experiments=[],
    )
    assert readiness_signals().sam3 is False
    _pin(
        monkeypatch,
        gpu=True,
        hf=True,
        sam3_weights=True,
        sam3_backend=True,
        imported=[],
        experiments=[],
    )
    assert readiness_signals().sam3 is True


def test_sam3_sub_signals_are_exposed_and_sam3_is_their_and(monkeypatch):
    # The Configuration checklist splits SAM 3 into backend (its own page) and weights
    # (Model manager), so each half is exposed and `sam3` stays their AND.
    results = [
        _res("sam3_backend", True),
        _res("sam3_weights", False),
    ]
    signals = signals_from(results)
    assert signals.sam3_backend is True
    assert signals.sam3_weights is False
    assert signals.sam3 is False


def test_detection_model_true_via_imported_when_sam3_absent(monkeypatch):
    _pin(
        monkeypatch,
        gpu=False,
        hf=False,
        sam3_weights=False,
        sam3_backend=False,
        imported=["m1"],
        experiments=[],
    )
    signals = readiness_signals()
    assert signals.sam3 is False and signals.detection_model is True


def test_a_keypoint_model_alone_is_not_a_detection_model(monkeypatch):
    # A pose model in the catalog detects nothing (§12 *Keypoint models in the catalog*):
    # the checklist must not tell a user with only one that detection is ready.
    _pin(
        monkeypatch,
        gpu=False,
        hf=False,
        sam3_weights=False,
        sam3_backend=False,
        imported=[{"id": "rat_pose", "task": "pose"}],
        experiments=[],
    )
    assert readiness_signals().detection_model is False


def test_sam3_alone_is_a_usable_detector_only_with_a_gpu(monkeypatch):
    common = dict(
        hf=True, sam3_weights=True, sam3_backend=True, imported=[], experiments=[]
    )
    _pin(monkeypatch, gpu=False, **common)
    assert readiness_signals().detection_model is False
    _pin(monkeypatch, gpu=True, **common)
    assert readiness_signals().detection_model is True


def test_an_unknown_sam3_probe_degrades_instead_of_raising():
    # A check that couldn't answer (UNKNOWN, e.g. the subprocess timed out) is not OK, so
    # the signal degrades to False rather than propagating.
    results = [CheckResult("sam3_weights", CheckStatus.UNKNOWN, "probe unavailable")]
    assert signals_from(results).sam3 is False


def test_gpu_and_hf_token_map_to_their_checks(monkeypatch):
    _pin(
        monkeypatch,
        gpu=True,
        hf=False,
        sam3_weights=False,
        sam3_backend=False,
        imported=[],
        experiments=[],
    )
    signals = readiness_signals()
    assert signals.gpu is True and signals.hf_token is False


# --- the checklist's last two steps read real state (#35) ---------------------------------


def test_a_completed_quick_segment_run_ticks_see_it_segment_a_clip():
    """Through the real tool-run store: the step is done once a run has *completed*. A run
    that failed or was stopped before it showed anything has not shown a clip segmented.
    """
    from detail.storage import tool_runs

    assert signals_from([]).quick_segment_run is False
    tool_runs.write_tool_run(
        "quick_segment", "run_001", {"id": "run_001", "status": "failed"}
    )
    assert signals_from([]).quick_segment_run is False
    tool_runs.write_tool_run(
        "quick_segment", "run_002", {"id": "run_002", "status": "complete"}
    )
    assert signals_from([]).quick_segment_run is True


@pytest.mark.parametrize(
    "record, distilled",
    [
        ({"id": "m1", "source": "imported"}, False),
        ({"id": "m1", "source": "distilled"}, True),
        # a distilled *keypoint* model detects nothing, and the step is about a detector
        ({"id": "k1", "source": "distilled", "task": "pose"}, False),
    ],
)
def test_only_a_distilled_detector_ticks_the_distill_step(
    monkeypatch, record, distilled
):
    _pin(
        monkeypatch,
        gpu=False,
        hf=False,
        sam3_weights=False,
        sam3_backend=False,
        imported=[record],
        experiments=[],
    )
    assert readiness_signals().distilled_model is distilled
