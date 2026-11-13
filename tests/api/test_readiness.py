"""Setup-readiness signals for the Home onboarding panel (§1).

The signals are now *derived from the live checker framework*: ``readiness_signals`` runs the
checks and ``signals_from`` maps their results to the five booleans, plus two cheap workspace
probes (imported catalog, experiment listing). These tests pin the check results + probes so
the mapping is deterministic on any machine/CI, plus one real-wiring smoke test.
"""

from detail.api.readiness import ReadinessSignals, readiness_signals, signals_from
from detail.core.dataclasses import CheckResult, CheckStatus
from detail.models.imported import role_of


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
    records=(),
    nvidia=False,
):
    results = [
        _res("cuda", gpu),
        _res("nvidia_gpu", nvidia),
        _res("hf_token", hf),
        _res("sam3_weights", sam3_weights),
        _res("sam3_backend", sam3_backend),
    ]
    monkeypatch.setattr("detail.api.readiness.run_checks", lambda **k: results)
    # ``imported`` is catalog ids (detection models) or records (to give one a task);
    # ``records`` are catalog records with a role and a source. Each is filtered by the real
    # ``role_of``, so a pose record without a role is a keypoints one, as on disk.
    records = [*records, *(r if isinstance(r, dict) else {"id": r} for r in imported)]

    def of_role(role):
        return [r for r in records if role is None or role_of(r) == role]

    monkeypatch.setattr(
        "detail.api.readiness.imported_store.discover",
        lambda role: [r.get("id") for r in of_role(role)],
    )
    monkeypatch.setattr(
        "detail.api.readiness.locations.list_experiment_dirs", lambda: experiments
    )
    monkeypatch.setattr("detail.api.readiness.imported_store.list_records", of_role)


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
        nvidia_gpu=False,
        hf_token=False,
        sam3=False,
        sam3_backend=False,
        sam3_weights=False,
        detection_model=False,
        has_experiment=False,
        distilled_detector=False,
        keypoint_student=False,
        keypoints_skipped=False,
        shap=False,
    )


def test_readiness_signals_all_true_when_fully_set_up(monkeypatch):
    _pin(
        monkeypatch,
        gpu=True,
        hf=True,
        sam3_weights=True,
        sam3_backend=True,
        imported=["m1"],
        experiments=["exp1"],
    )
    assert readiness_signals() == ReadinessSignals(
        gpu=True,
        nvidia_gpu=True,
        hf_token=True,
        sam3=True,
        sam3_backend=True,
        sam3_weights=True,
        detection_model=True,
        has_experiment=True,
        distilled_detector=False,
        keypoint_student=False,
        keypoints_skipped=False,
        shap=False,
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


def test_an_nvidia_gpu_torch_cannot_use_is_not_reported_as_no_gpu(monkeypatch):
    # A first install's CPU-only torch wheel on an NVIDIA box: `cuda` warns while the
    # driver answers. That is a GPU PyTorch can't use yet, not a machine without one.
    common = dict(
        hf=False, sam3_weights=False, sam3_backend=False, imported=[], experiments=[]
    )
    _pin(monkeypatch, gpu=False, nvidia=True, **common)
    signals = readiness_signals()
    assert signals.gpu is False and signals.nvidia_gpu is True
    _pin(monkeypatch, gpu=False, nvidia=False, **common)
    signals = readiness_signals()
    assert signals.gpu is False and signals.nvidia_gpu is False
    # A CUDA device torch can use is an NVIDIA GPU even with no nvidia-smi on PATH.
    _pin(monkeypatch, gpu=True, nvidia=False, **common)
    assert readiness_signals().nvidia_gpu is True


# --- the keypoint setup step's signals (§26, #717) ---------------------------------- #
def _fresh(monkeypatch, **over):
    args = dict(
        gpu=True,
        hf=True,
        sam3_weights=True,
        sam3_backend=True,
        imported=[],
        experiments=[],
    )
    args.update(over)
    _pin(monkeypatch, **args)
    return readiness_signals()


def test_a_distilled_model_in_the_catalog_counts_by_its_role(monkeypatch):
    signals = _fresh(
        monkeypatch,
        records=[
            {"role": "detection", "source": "distilled"},
            {
                "role": "keypoints",
                "source": "imported",
            },  # an ONNX import, not a student
        ],
    )
    assert (signals.distilled_detector, signals.keypoint_student) == (True, False)
    signals = _fresh(
        monkeypatch, records=[{"role": "keypoints", "source": "distilled"}]
    )
    assert (signals.distilled_detector, signals.keypoint_student) == (False, True)


def test_a_detector_trained_in_an_experiment_counts(monkeypatch, tmp_path):
    from detail.storage import models as models_store

    def model(experiment, model_id, **record):
        model_dir = models_store.model_dir(experiment, model_id)
        model_dir.mkdir(parents=True)
        models_store.write_model_yaml(model_dir, {"id": model_id, **record})

    student_only = tmp_path / "a"
    model(student_only, "model_001", status="complete", role="keypoints")
    model(student_only, "model_002", status="running")
    assert not _fresh(monkeypatch, experiments=[student_only]).distilled_detector
    trained = tmp_path / "b"
    model(trained, "model_001", status="complete")
    assert _fresh(monkeypatch, experiments=[student_only, trained]).distilled_detector


def test_a_keypoint_student_trained_in_an_experiment_counts(monkeypatch, tmp_path):
    """On main's classes a keypoint distill registers its student in the experiment
    (``task: pose``) rather than the catalog, so the checklist's keypoint step reads both —
    and a pose model is never counted as a detector."""
    from detail.storage import models as models_store

    def model(experiment, model_id, **record):
        model_dir = models_store.model_dir(experiment, model_id)
        model_dir.mkdir(parents=True)
        models_store.write_model_yaml(model_dir, {"id": model_id, **record})

    pose_only = tmp_path / "pose"
    model(
        pose_only, "model_001", status="complete", task="pose"
    )  # trained, unregistered
    signals = _fresh(monkeypatch, experiments=[pose_only])
    assert (signals.keypoint_student, signals.distilled_detector) == (True, False)
    running = tmp_path / "running"
    model(running, "model_001", status="running", task="pose", role="keypoints")
    assert not _fresh(monkeypatch, experiments=[running]).keypoint_student


def test_skipping_keypoints_is_remembered_and_can_be_withdrawn(monkeypatch):
    from detail.storage import guidance

    assert not _fresh(monkeypatch).keypoints_skipped
    guidance.set_keypoints_skipped(True)
    assert _fresh(monkeypatch).keypoints_skipped
    guidance.set_keypoints_skipped(False)
    assert not _fresh(monkeypatch).keypoints_skipped


def test_a_damaged_guidance_file_reads_as_not_skipped(monkeypatch):
    from detail.storage import guidance

    guidance.guidance_path().parent.mkdir(parents=True, exist_ok=True)
    guidance.guidance_path().write_text(
        "keypoints_skipped: [unclosed", encoding="utf-8"
    )
    assert not _fresh(monkeypatch).keypoints_skipped
