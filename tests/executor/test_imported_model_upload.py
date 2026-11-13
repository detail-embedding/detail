"""A model the job needs that lives outside the experiment, and nothing sends it (§6).

**The class: a dependency that exists only on this machine, checked for on the cluster.** The
whole-experiment upload carries what is inside the experiment, and the video roots have their own
sync for what is outside it (`_sync_video_roots`). An **imported** detection model (a distilled
or imported YOLO) is outside it too, in this machine's model library, and had neither: the
cluster found no ``model.yaml`` and every recording of a 77-recording C57 detection failed at
"could not check model readiness on remote", the real cause (``FileNotFoundError: no model.yaml in
~/.cache/detail/models/imported/…``) buried in a conda banner. A model trained *inside* the
experiment never showed it, because it travels with the upload, which is why CD1 ran and C57 did
not.

**It hid because the readiness check was always run against a model already on the cluster**: a
built-in the cluster downloads itself, or an experiment model it had been sent, never the one
case that has to be carried. These tests make the model exist only locally and assert it is on its
way before the check, from where this machine keeps it to where the *cluster* says it keeps
imported models, and that the sources that are already carried (``trained``) or fetched by the
cluster (``pretrained``) are not sent a second copy.

**It came back one job family over.** The catalog holds keypoint models too — an imported ONNX
model, and a YOLO-pose *student* distilled from a teacher run — in the same directory, and the
upload asked only detection's question (``weights_source == "imported"``). A remote keypoints run
naming a student passed ``validate`` here and met ``unknown keypoint model`` on the cluster; an
ONNX import was refused up front, with "its weights are not uploaded" as the reason, which the
upload now makes false. The shepherd asks each job family its own resolver's question
(``_CATALOG_MODEL_OF``), and the guard at the bottom enumerates every family that names a model,
and every catalog role, so a family or a role added later is either sent or says why not.
"""

import dataclasses
import json

import pytest

from detail.core import dataclasses as job_types
from detail.core.dataclasses import (
    ComputeTarget,
    DetectionJob,
    JobStatus,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.executor import remote_shepherd, rundir, ssh_transport
from detail.executor.schedulers.base import JobState
from detail.models import imported as imported_store

# Fixtures and helpers of the shepherd tests: the same experiment and fake scheduler.
from tests.executor.test_remote_shepherd import (  # noqa: F401
    _FakeCompletedProcess,
    _FakeScheduler,
    _install_scheduler,
    _target,
    experiment,
)

MODEL_ID = "C57_model"
REMOTE_DIR = "/gpfs/home/sagolden/.cache/detail/models/imported/C57_model"
BANNER = "Miniforge (conda) has been loaded.\nWARNING: Login nodes are CPU ONLY.\n"
READY = json.dumps(
    {"ready": True, "reason": "ok", "message": "", "action_url": None, "repo_id": None}
)


def _job(weights_source="imported", model_id=MODEL_ID):
    return DetectionJob(
        job_id="det_01",
        recording_id="recording_001",
        model_id=model_id,
        prompt=Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        conf_threshold=0.5,
        compute=ComputeTarget("remote", None, None, None),
        created_at="2026-01-01T00:00:00Z",
        weights_source=weights_source,
    )


def _library_model(model_id=MODEL_ID):
    """The model in this machine's library (the per-test ``DETAIL_MODEL_DIR``)."""
    model_dir = imported_store.model_dir(model_id)
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "model.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (model_dir / "weights" / "best.pt").write_bytes(b"weights")
    return model_dir


class _Cluster:
    """The ssh surface: what the cluster says about its model directory, and what is sent."""

    def __init__(self, monkeypatch, *, answer=None):
        self.events: list = []
        self.copies: list = []
        answer = answer or _FakeCompletedProcess(0, BANNER + REMOTE_DIR + "\n")

        def run_remote(host, command, *, stdin_text=None, **kwargs):
            if command.startswith("cat ") and rundir.JOB_FILENAME in command:
                # The shepherd asking whose run directory is there (nothing: a fresh run).
                self.events.append("own-run-dir")
                return _FakeCompletedProcess(0, "")
            if stdin_text and "imported.model_dir" in stdin_text:
                self.events.append("ask-where")
                return answer
            self.events.append("readiness")
            return _FakeCompletedProcess(0, READY)

        def copy_to_remote(host, local_dir, remote_dir, **kwargs):
            self.events.append(
                ("copy", remote_dir)
            )  # the experiment's own upload is one too
            self.copies.append((host, local_dir, remote_dir))

        monkeypatch.setattr(ssh_transport, "run_remote", run_remote)
        monkeypatch.setattr(ssh_transport, "copy_to_remote", copy_to_remote)


def _imported_job_run(monkeypatch, experiment):
    """``experiment``'s run, but for an imported-model job, and a scheduler that finishes."""
    experiment_root, run_dir = experiment
    monkeypatch.setattr(rundir, "read_job", lambda _run_dir: _job())
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )
    return experiment_root, run_dir


def test_an_imported_model_is_uploaded_to_where_the_cluster_keeps_them(monkeypatch):
    local = _library_model()
    cluster = _Cluster(monkeypatch)

    remote_shepherd._sync_catalog_model(_target(), _job())

    # to the cluster's own answer (the banner above it ignored), from the local library
    assert cluster.copies == [("tillicum", local, REMOTE_DIR)]


def test_the_phase_names_the_model_and_the_cluster(monkeypatch):
    _library_model()
    _Cluster(monkeypatch)
    said = []

    remote_shepherd._sync_catalog_model(_target(), _job(), announce=said.append)

    assert said == [f"uploading the model {MODEL_ID} to tillicum"]


@pytest.mark.parametrize("weights_source", ["pretrained", "trained"])
def test_a_model_the_cluster_fetches_or_the_experiment_carries_is_not_sent(
    monkeypatch, weights_source
):
    """Built-ins are downloaded on the cluster; an experiment's own model rode the upload."""
    _library_model()  # even with one of the same name sitting in the library
    cluster = _Cluster(monkeypatch)

    remote_shepherd._sync_catalog_model(_target(), _job(weights_source))

    assert cluster.events == [] and cluster.copies == []


def test_a_model_missing_here_says_so_instead_of_uploading_nothing(monkeypatch):
    cluster = _Cluster(monkeypatch)

    with pytest.raises(FileNotFoundError, match="not in this machine's model library"):
        remote_shepherd._sync_catalog_model(_target(), _job())

    assert cluster.copies == []


@pytest.mark.parametrize(
    "answer",
    [
        _FakeCompletedProcess(1, "", "conda: command not found"),
        _FakeCompletedProcess(0, BANNER, ""),  # prints no path at all
        _FakeCompletedProcess(0, "models/imported/C57_model\n"),  # not absolute
    ],
    ids=["ssh-failed", "no-path", "relative-path"],
)
def test_a_cluster_that_cannot_say_where_is_a_clear_error_not_a_guess(
    monkeypatch, answer
):
    _library_model()
    cluster = _Cluster(monkeypatch, answer=answer)

    with pytest.raises(RuntimeError, match="could not find where 'tillicum'"):
        remote_shepherd._sync_catalog_model(_target(), _job())

    assert cluster.copies == []  # never to a path it had to make up


def test_a_run_uploads_the_model_before_it_checks_the_model(monkeypatch, experiment):
    """The order is the fix: the check that failed on its absence runs after the upload."""
    _library_model()
    cluster = _Cluster(monkeypatch)
    experiment_root, run_dir = _imported_job_run(monkeypatch, experiment)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    events = cluster.events
    asked = events.index("ask-where")
    uploaded = events.index(("copy", REMOTE_DIR))
    checked = events.index("readiness")
    assert asked < uploaded < checked


def test_a_run_whose_model_is_missing_here_fails_before_anything_is_queued(
    monkeypatch, experiment
):
    _Cluster(monkeypatch)
    experiment_root, run_dir = _imported_job_run(monkeypatch, experiment)
    scheduler = _FakeScheduler([JobState.RUNNING])
    _install_scheduler(monkeypatch, scheduler)

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.FAILED
    assert (
        "not in this machine's model library" in rundir.read_envelope(run_dir)["error"]
    )
    assert not scheduler.submitted


def test_an_ordinary_run_asks_nothing_about_imported_models(monkeypatch, experiment):
    """A pretrained job (the fixture's own) pays no extra ssh for a feature it does not use."""
    cluster = _Cluster(monkeypatch)
    experiment_root, run_dir = experiment
    _install_scheduler(monkeypatch, _FakeScheduler([JobState.RUNNING]))
    monkeypatch.setattr(
        ssh_transport,
        "read_remote_file",
        lambda *a, **k: json.dumps({"status": "complete", "progress": None}),
    )

    assert (
        remote_shepherd.run(run_dir, "tillicum", experiment_root) == JobStatus.COMPLETE
    )

    assert "ask-where" not in cluster.events
    assert ("copy", REMOTE_DIR) not in cluster.events


def test_a_failed_readiness_check_leads_with_its_cause_not_the_banner(monkeypatch):
    """The second half of the incident: the real error was the last line of eleven."""
    stderr = (
        BANNER
        + "\nTraceback (most recent call last):\n"
        + '  File "<stdin>", line 32, in <module>\n'
        + "FileNotFoundError: no model.yaml in /gpfs/home/x/.cache/detail/models/imported/m"
    )
    monkeypatch.setattr(
        ssh_transport,
        "run_remote",
        lambda *a, **k: _FakeCompletedProcess(1, "", stderr),
    )

    with pytest.raises(RuntimeError) as caught:
        remote_shepherd._check_model_readiness(_target(), _job(), "/x/exp", "")

    first = str(caught.value).splitlines()[0]
    assert first.endswith(
        "FileNotFoundError: no model.yaml in /gpfs/home/x/.cache/detail/models/imported/m"
    )
    assert "Miniforge" in str(caught.value)  # the rest is still there, below


# --------------------------------------------------------------------------- #
# Keypoint models: the same catalog, one job family over
# --------------------------------------------------------------------------- #
def _keypoints_job(model_id):
    from detail.core.keypoints import KeypointDetectJob

    return KeypointDetectJob(
        job_id="kd_01",
        recording_id="recording_001",
        detect_run_id="kd_01",
        detection_run_id="det_01",
        min_pose_confidence=None,
        model_id=model_id,
        model_variant=None,
        backbone=None,
        example_set=None,
        skeleton={"names": ["nose", "tail"], "edges": [], "pairs": []},
        n_candidates=1,
        every_n_frames=1,
        crop_px=None,
        decode=None,
        side_rule=None,
        matching=None,
        batch_size=32,
        compute=ComputeTarget("remote", None, None, None, "tillicum"),
        created_at="2026-10-06T00:00:00Z",
    )


def _student(tmp_path):
    """A YOLO-pose student in the catalog (``role: keypoints``), the shape a promoted one has."""
    del tmp_path
    model_id = "rat_pose_student"
    model_dir = imported_store.model_dir(model_id)
    (model_dir / "weights").mkdir(parents=True, exist_ok=True)
    (model_dir / "weights" / "best.pt").write_bytes(b"weights")
    (model_dir / "model.yaml").write_text(
        "schema_version: 1\n"
        f"id: {model_id}\n"
        "role: keypoints\n"
        "task: pose\n"
        "model_family: yolo_pose\n"
        "status: complete\n"
        "weights: {best: weights/best.pt}\n",
        encoding="utf-8",
    )
    return model_id


def _cluster_dir(model_id):
    return f"/gpfs/home/sagolden/.cache/detail/models/imported/{model_id}"


def _answer(model_id):
    """The cluster's ``imported.model_dir``, under its login banner."""
    return _FakeCompletedProcess(0, BANNER + _cluster_dir(model_id) + "\n")


def _onnx_import(tmp_path):
    """An imported sleap-nn ONNX keypoint model, through the real importer."""
    del tmp_path
    from tests.pipeline.test_keypoint_detect import _import_onnx_fixture

    return _import_onnx_fixture()


@pytest.mark.parametrize("make", [_student, _onnx_import])
def test_a_catalog_keypoint_model_is_uploaded(monkeypatch, tmp_path, make):
    model_id = make(tmp_path)
    cluster = _Cluster(monkeypatch, answer=_answer(model_id))

    remote_shepherd._sync_catalog_model(_target(), _keypoints_job(model_id))

    assert cluster.copies == [
        ("tillicum", imported_store.model_dir(model_id), _cluster_dir(model_id))
    ]


def test_a_built_in_keypoint_method_is_not_sent(monkeypatch):
    """``dinov3_match`` is fetched by the cluster; its example set rides in the job."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    cluster = _Cluster(monkeypatch)

    remote_shepherd._sync_catalog_model(_target(), _keypoints_job("dinov3_match"))

    assert cluster.events == [] and cluster.copies == []


def test_a_keypoints_run_uploads_its_student_before_it_checks_it(
    monkeypatch, experiment, tmp_path
):
    """The incident's shape for keypoints: the cluster's resolver never heard of the id."""
    model_id = _student(tmp_path)
    cluster = _Cluster(monkeypatch, answer=_answer(model_id))
    experiment_root, run_dir = _imported_job_run(monkeypatch, experiment)
    monkeypatch.setattr(rundir, "read_job", lambda _run_dir: _keypoints_job(model_id))

    status = remote_shepherd.run(run_dir, "tillicum", experiment_root)

    assert status == JobStatus.COMPLETE
    events = cluster.events
    assert (
        events.index("ask-where")
        < events.index(("copy", _cluster_dir(model_id)))
        < events.index("readiness")
    )


# --------------------------------------------------------------------------- #
# The guard: every job family that names a model, and every catalog role
# --------------------------------------------------------------------------- #
#: Job families that carry a ``model_id`` and are not in ``_CATALOG_MODEL_OF``, each with the
#: reason no catalog model can reach it. An entry that stops being true fails below.
_NAMES_NO_CATALOG_MODEL = {
    "embedding": "resolved from EMBEDDING_MODELS only; the catalog holds no embedding role",
    "labeling": (
        "a prompt-capable MODELS member (SAM 3); it has no weights_source, and a catalog "
        "detector is closed-set, so it can never label"
    ),
    "training": "its model_id is the model this job creates; base_model is a hub weights name",
}

#: Which job family runs a catalog model of each role.
_ROLE_RUN_BY = {"detection": "detection", "keypoints": "keypoint_detect"}


def _model_job_families():
    for cls, tag in job_types._JOB_TAGS.items():
        names = {f.name for f in dataclasses.fields(cls)}
        if "model_id" in names and "compute" in names:
            yield tag


def test_every_job_family_that_names_a_model_is_sent_its_catalog_model_or_says_why_not():
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(
        PIPELINE_RUNNERS
    )  # keypoint_detect registers its job type on import
    families = set(_model_job_families())
    assert (
        "keypoint_detect" in families
    )  # the enumeration sees the family that was missed
    for tag in families:
        assert (tag in remote_shepherd._CATALOG_MODEL_OF) != (
            tag in _NAMES_NO_CATALOG_MODEL
        ), (
            f"job family {tag!r} names a model: add it to remote_shepherd._CATALOG_MODEL_OF "
            "if a catalog model can reach it, or to _NAMES_NO_CATALOG_MODEL with the reason"
        )
    assert set(_NAMES_NO_CATALOG_MODEL) <= families, "an entry that excuses nothing"


def test_every_catalog_role_is_run_by_a_family_that_sends_it():
    """A new role (a catalog embedding model, say) must say which job family uploads it."""
    assert set(_ROLE_RUN_BY) == set(imported_store.ROLES)
    assert set(_ROLE_RUN_BY.values()) == set(remote_shepherd._CATALOG_MODEL_OF)


def test_detection_sends_only_an_imported_weights_source():
    assert remote_shepherd._catalog_model_id(_job()) == MODEL_ID
    assert remote_shepherd._catalog_model_id(_job("trained")) is None
    assert remote_shepherd._catalog_model_id(_job("pretrained")) is None
