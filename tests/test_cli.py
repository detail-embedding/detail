"""Smoke tests for the CLI — exercises the click wrappers end-to-end on tmp_path.

The point isn't to re-test the API (covered elsewhere); it's to confirm the CLI
plumbing (arg parsing, experiment/recording lookups, error → non-zero exit) does
what each command's docstring says.

**The class: a behaviour promised in ``--help`` that the shared function underneath has no
parameter for.** Ctrl-C handling was ``cancel_on_interrupt``, on **two of the three** progress
drivers and defaulting to cancel — so it reached the two callers somebody was working on, while
the five single-job ``attach`` commands all stream through the third, which had none. Each of
those five had printed its Ctrl-C promise since the day it was written: correct, prominent, and
unimplementable at the call site.

The fix is to ask the **handle** (``RunHandle.owns_worker`` — whose worker is it) rather than add
a third opt-in, because an opt-in defaulting to "as before" reaches the surface being edited and
no other.

Then **guard the prose, not only the mechanism**: one command shipped promising to leave a run
alone while launching its own worker, and only a sweep that reads what each command *claims* can
see that. That sweep is ``tests/api/test_job_reattach.py``, and it matches **exact sentences** —
substring-sniffing for the word "detach" reads a docstring explaining why a command does *not*
detach as a promise that it does, which is what happened to the docstring written for this guard.

"""

import contextlib
import io
import json
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from detail.cli import (
    _coerce_hyperparam,
    _progress_postfix,
    _stream_distill_progress,
    _stream_processing_progress,
    _stream_progress,
    main,
)
from detail.executor.base import RunHandle
from detail.core.dataclasses import (
    Detection,
    DetectionProgressEvent,
    FrameResult,
    EmbeddingProgressEvent,
    ModelReadiness,
    TextCue,
    TrackingProgressEvent,
)
from detail.core.protocols import DetectionModel
from detail.configs.labeling import LabelingConfig
from detail.configs.tracking import DEFAULT_TRACKER
from detail.core.registries import MODELS, TRACKERS, register_model
from detail.models.hf_access import HFHubModel
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store
from detail.storage.zarr_io import DetectionPartWriter

H, W = 24, 32


def _write_video(path, n_frames=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()


def _runner():
    return CliRunner()


# --- experiment create / status ----------------------------------------------- #
def test_experiment_create(tmp_path):
    result = _runner().invoke(
        main, ["experiment", "create", "p1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "created experiment" in result.output
    assert (tmp_path / "p1" / "config.yaml").is_file()


def test_experiment_create_already_exists(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    result = r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_experiment_status(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    result = r.invoke(main, ["experiment", "status", "--experiment", pdir])
    assert result.exit_code == 0, result.output
    assert "recording_001" in result.output


def test_experiment_status_missing_experiment(tmp_path):
    result = _runner().invoke(
        main, ["experiment", "status", "--experiment", str(tmp_path / "nope")]
    )
    assert result.exit_code != 0


# --- recording create / add-video / status ----------------------------------- #
def test_recording_create(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    result = r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "cage_a"]
    )
    assert result.exit_code == 0, result.output
    # The name is the user's; the id is minted (CLAUDE.md, "A recording's id is the tool's").
    assert result.output.strip() == "created recording: cage_a  (id recording_000)"
    assert (
        tmp_path / "p1" / "recordings" / "recording_000" / "recording.yaml"
    ).is_file()


def test_recording_add_video(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    video_path = tmp_path / "in.mp4"
    _write_video(video_path, n_frames=4)
    result = r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(video_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # `--recording` took the name; the files live under the id the tool minted.
    assert (
        tmp_path / "p1" / "recordings" / "recording_000" / "video" / "part_001.mp4"
    ).is_file()


def _isolate_storage(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")


def test_cli_video_root_set_and_list(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    root = tmp_path / "nas"
    root.mkdir()
    r = _runner()
    res = r.invoke(main, ["video-root", "set", "lab", str(root)])
    assert res.exit_code == 0, res.output
    res = r.invoke(main, ["video-root", "list"])
    # `str`, not `as_posix`: this is a machine-local path shown to the user, so the CLI
    # prints it with the platform's own separators and the assertion has to match.
    assert "lab" in res.output and str(root.resolve()) in res.output


def test_cli_video_root_set_rejects_missing_dir(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    res = _runner().invoke(main, ["video-root", "set", "lab", str(tmp_path / "nope")])
    assert res.exit_code != 0


def test_cli_recording_add_video_reference(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    from detail.storage import recording as recording_store

    root = tmp_path / "nas"
    root.mkdir()
    video_path = root / "m.mp4"
    _write_video(video_path, n_frames=4)
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    result = r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(video_path),
            "--reference",  # a flag now; the location is remembered automatically
        ],
    )
    assert result.exit_code == 0, result.output
    assert "referenced" in result.output
    sdir = tmp_path / "p1" / "recordings" / "recording_000"  # the minted id
    assert not (sdir / "video" / "part_001.mp4").exists()  # not copied
    part = recording_store.read_recording(sdir)["video"]["files"][0]
    assert part["source"]["rel"] == "m.mp4"  # relative to the auto-remembered folder
    assert "root" in part["source"]


def _reference_recording_cli(tmp_path, r, n_frames=4):
    """Create p1/recording_001 with one referenced video under root 'nas'. Returns the src."""
    root = tmp_path / "nas"
    root.mkdir(exist_ok=True)
    src = root / "m.mp4"
    _write_video(src, n_frames=n_frames)
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(src),
            "--reference",
        ],
    )
    return pdir, src


def test_cli_recording_verify(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    r = _runner()
    pdir, src = _reference_recording_cli(tmp_path, r)
    clean = r.invoke(
        main,
        ["recording", "verify", "--experiment", pdir, "--recording", "recording_001"],
    )
    assert clean.exit_code == 0 and "present and unchanged" in clean.output
    with open(src, "ab") as fh:
        fh.write(b"tampered")
    bad = r.invoke(
        main,
        ["recording", "verify", "--experiment", pdir, "--recording", "recording_001"],
    )
    assert bad.exit_code != 0 and "changed_fingerprint" in bad.output


def test_cli_recording_relink(tmp_path, monkeypatch):
    _isolate_storage(tmp_path, monkeypatch)
    r = _runner()
    pdir, src = _reference_recording_cli(tmp_path, r)
    moved = tmp_path / "nas" / "sub" / "m2.mp4"
    moved.parent.mkdir()
    src.rename(moved)
    res = r.invoke(
        main,
        [
            "recording",
            "relink",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--part",
            "0",
            "--video",
            str(moved),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "registered video root" not in res.output
    verify = r.invoke(
        main,
        ["recording", "verify", "--experiment", pdir, "--recording", "recording_001"],
    )
    assert verify.exit_code == 0


def test_cli_recording_relink_registers_a_root_this_machine_lacks(
    tmp_path, monkeypatch
):
    from detail.storage import locations

    _isolate_storage(tmp_path, monkeypatch)
    r = _runner()
    pdir, src = _reference_recording_cli(tmp_path, r)
    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path / "cfg2")
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg2" / "storage.yaml")
    moved = tmp_path / "mounted" / "m.mp4"
    moved.parent.mkdir()
    src.rename(moved)
    res = r.invoke(
        main,
        [
            "recording",
            "relink",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--part",
            "0",
            "--video",
            str(moved),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "registered video root 'nas'" in res.output


def test_recording_import_labels(tmp_path):
    import yaml

    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    video_path = tmp_path / "in.mp4"
    _write_video(video_path, n_frames=4)
    r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(video_path),
        ],
    )
    (tmp_path / "s1.csv").write_text(
        "start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8"
    )
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(
        yaml.dump(
            # The labels manifest keys on the id, which the tool minted.
            {"schema_version": 1, "labels": {"recording_000": "s1.csv"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = r.invoke(
        main,
        [
            "recording",
            "import-labels",
            "--experiment",
            pdir,
            "--manifest",
            str(manifest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "imported 1/1" in result.output
    assert (
        "recording_000: done" in result.output
    )  # per-recording progress callback wired


def test_recording_add_video_drives_progress_bar(tmp_path, monkeypatch):
    # add-video is synchronous; the CLI must pass a progress callback so a long
    # transcode isn't silent. Assert the callback is wired and survives a stage
    # transition (the recurring "long op looks hung" gap).
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    video_path = tmp_path / "in.mp4"
    _write_video(video_path, n_frames=4)
    seen = {}

    def fake_add(self, path, *args, progress=None, **kwargs):
        seen["progress"] = progress
        assert callable(progress)
        progress("Reading timestamps", 0.5, "")
        progress("Copying", 0.2, "1.0 MB / 5.0 MB")  # stage change
        progress("Copying", 1.0, "5.0 MB / 5.0 MB")

    monkeypatch.setattr("detail.api.recording.Recording.add_video_part", fake_add)
    result = r.invoke(
        main,
        [
            "recording", "add-video",
            "--experiment", pdir,
            "--recording", "recording_001",
            "--video", str(video_path),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen["progress"] is not None


def test_docs_command_builds_with_progress_and_opens(monkeypatch):
    # `detail docs` builds the offline site on first use (a long op → must pass a
    # progress callback) then opens it in the browser.
    seen = {}

    def fake_build(progress=None, force=False):
        seen["progress"] = progress
        assert callable(progress)
        return None

    launched = {}
    monkeypatch.setattr("detail.cli.docs_access.ensure_site_built", fake_build)
    monkeypatch.setattr(
        "detail.cli.docs_access.help_url", lambda page="index": "file:///x"
    )
    monkeypatch.setattr(
        "detail.cli.click.launch", lambda url: launched.setdefault("url", url)
    )
    result = _runner().invoke(main, ["docs"])
    assert result.exit_code == 0, result.output
    assert seen["progress"] is not None
    assert launched["url"] == "file:///x"


def test_docs_command_rejects_unknown_topic(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "detail.cli.docs_access.ensure_site_built", lambda progress=None: None
    )
    monkeypatch.setattr("detail.cli.docs_access.built_site_dir", lambda: tmp_path)
    monkeypatch.setattr("detail.cli.click.launch", lambda url: None)
    result = _runner().invoke(main, ["docs", "no-such-page"])
    assert result.exit_code != 0
    assert "no-such-page" in result.output


def test_docs_command_serve_runs_mkdocs_serve(monkeypatch, tmp_path):
    (tmp_path / "mkdocs.yml").write_text("site_name: x\n", encoding="utf-8")
    ran = {}
    monkeypatch.setattr("detail.cli.docs_access.docs_source_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "detail.cli.subprocess.run", lambda cmd, **kw: ran.update(cmd=cmd, kw=kw)
    )
    result = _runner().invoke(main, ["docs", "--serve"])
    assert result.exit_code == 0, result.output
    assert "mkdocs" in ran["cmd"] and "serve" in ran["cmd"]


def test_cli_query_eval_drives_progress_bar(tmp_path, monkeypatch):
    # query-eval is synchronous but loads embeddings + runs a similarity search per
    # behavior — the CLI must pass a progress callback so a long run isn't silent.
    # Assert the callback is wired and survives a stage transition.
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    seen = {}

    def fake_eval(experiment_dir, config, progress=None):
        seen["progress"] = progress
        assert callable(progress)
        progress("loading", 0.0, "groom: finding eligible recordings")
        progress("querying", 0.45, "groom: querying 1 recording(s)")  # stage change
        progress("scoring", 1.0, "done")
        return []

    monkeypatch.setattr("detail.pipeline.query_eval.evaluate_query", fake_eval)
    result = r.invoke(
        main,
        [
            "analysis", "query-eval",
            "--experiment", pdir,
            "--embed-run", "emb_01",
            "--behavior", "groom",
            "--out", str(tmp_path / "m.csv"),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert callable(seen["progress"])


def test_recording_status_shows_video(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    video_path = tmp_path / "in.mp4"
    _write_video(video_path, n_frames=4)
    r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(video_path),
        ],
    )
    result = r.invoke(
        main,
        ["recording", "status", "--experiment", pdir, "--recording", "recording_001"],
    )
    assert result.exit_code == 0, result.output
    assert "video parts: 1" in result.output
    assert "32x24" in result.output


# --- detection run validation only (no model invocation) ------------------- #
def test_detection_unknown_model(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    video_path = tmp_path / "in.mp4"
    _write_video(video_path)
    r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(video_path),
        ],
    )
    result = r.invoke(
        main,
        [
            "detection",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--model",
            "no_such_model",
            "--class",
            "mouse",
            "--prompt",
            "mouse",
        ],
    )
    assert result.exit_code != 0
    assert "unknown model" in result.output


# --- tracking run validation ---------------------------------------------- #
def test_tracking_unknown_detection(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    result = r.invoke(
        main,
        [
            "tracking",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--detection",
            "det_99",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_tracking_anchor_tracker_flags_parse(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    result = r.invoke(
        main,
        [
            "tracking",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--detection",
            "det_99",
            "--tracker",
            "anchor_tracker",
            "--num-subjects",
            "1",
            "--secondary-conf",
            "0.05",
            "--max-secondary",
            "2",
            "--require-contiguous",
        ],
    )
    # anchor flags parse cleanly (else click exits 2 "no such option"); the run
    # then fails at the detection lookup, proving it reached the validation layer.
    assert result.exit_code != 0
    assert "not found" in result.output


# --- embedding bad --tracks parsing --------------------------------------- #
def test_embedding_bad_tracks_arg(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    result = r.invoke(
        main,
        [
            "embedding",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--tracking",
            "track_01",
            "--tracks",
            "not,an,int",
            "--model",
            "dinov3_convnext_base",
        ],
    )
    assert result.exit_code != 0
    assert "comma-separated ints" in result.output


def test_embedding_run_rejects_unknown_constructor(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    result = r.invoke(
        main,
        [
            "embedding",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--tracking",
            "track_01",
            "--tracks",
            "0",
            "--model",
            "dinov3_convnext_base",
            "--constructor",
            "bogus",
        ],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output  # click.Choice rejects unknown recipes


def _proj_with_recording(r, tmp_path):
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    return pdir


def test_embedding_run_rejects_batch_size_below_one(tmp_path):
    r = _runner()
    pdir = _proj_with_recording(r, tmp_path)
    result = r.invoke(
        main,
        [
            "embedding",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--tracking",
            "track_01",
            "--tracks",
            "0",
            "--model",
            "dinov3_convnext_base",
            "--batch-size",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "batch-size" in result.output  # click.IntRange(min=1) rejects 0


def test_detection_run_rejects_batch_size_below_one(tmp_path):
    r = _runner()
    pdir = _proj_with_recording(r, tmp_path)
    result = r.invoke(
        main,
        [
            "detection",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--model",
            "sam3",
            "--class",
            "mouse",
            "--prompt",
            "mouse",
            "--batch-size",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "batch-size" in result.output  # click.IntRange(min=1) rejects 0


def test_embedding_run_tracks_and_all_tracks_mutually_exclusive(tmp_path):
    r = _runner()
    pdir = _proj_with_recording(r, tmp_path)
    result = r.invoke(
        main,
        [
            "embedding",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--tracking",
            "track_01",
            "--tracks",
            "0",
            "--all-tracks",
            "--model",
            "dinov3_convnext_base",
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_embedding_run_requires_tracks_or_all_tracks(tmp_path):
    r = _runner()
    pdir = _proj_with_recording(r, tmp_path)
    result = r.invoke(
        main,
        [
            "embedding",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--tracking",
            "track_01",
            "--model",
            "dinov3_convnext_base",
        ],
    )
    assert result.exit_code != 0
    assert "provide --tracks or --all-tracks" in result.output


def test_tracking_run_hyperparam_unknown_key(tmp_path):
    r = _runner()
    pdir = _proj_with_recording(r, tmp_path)
    result = r.invoke(
        main,
        [
            "tracking",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--detection",
            "det_01",
            "--hyperparam",
            "bogus=1",
        ],
    )
    assert result.exit_code != 0
    assert "unknown hyperparam" in result.output
    # The command names no --tracker, so the schema `bogus` is rejected against is the
    # *default* tracker's. Asserting the listed keys rather than the phrase, because the
    # phrase is satisfied by whichever tracker happens to be the default.
    schema = TRACKERS[DEFAULT_TRACKER].HYPERPARAMS_SCHEMA
    for name in schema:
        assert name in result.output, f"the allowed list does not name {name!r}"


def test_max_missed_writes_whichever_key_the_chosen_tracker_declares():
    """One flag, one idea, two spellings — and a flag pinned to either one fails on the other.

    `iou_tracker` called it `max_missed_frames` (until it was removed); the single-subject
    trackers call it `max_missed`. `--max-missed` used to write the first unconditionally, so
    `--tracker iou_single_tracker --max-missed 45` failed with "unknown hyperparams" — a flag
    the command offers, rejected by the tracker the command defaults to once `iou_tracker`
    was hidden. Driven over every registered tracker, so a new one is covered the day it lands.
    """
    from detail.cli import _build_tracking_hyperparams

    for tracker_id, cls in sorted(TRACKERS.items()):
        schema = cls.HYPERPARAMS_SCHEMA
        built = _build_tracking_hyperparams(
            tracker_id, None, 45, None, None, None, None, None, ()
        )
        if not {"max_missed", "max_missed_frames"} & set(schema):
            continue  # a tracker with no such concept; nothing to spell
        assert len(built) == 1
        [key] = built
        assert key in schema, f"--max-missed writes {key!r}, which {tracker_id} rejects"
        # And the tracker really accepts it, rather than the key merely being in the dict.
        assert cls().validate_hyperparams(built)[key] == 45


def test_tracking_run_hyperparam_conflicts_with_typed_flag(tmp_path):
    r = _runner()
    pdir = _proj_with_recording(r, tmp_path)
    result = r.invoke(
        main,
        [
            "tracking",
            "run",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--detection",
            "det_01",
            "--tracker",
            "anchor_tracker",
            "--max-secondary",
            "2",
            "--hyperparam",
            "max_secondary=3",
        ],
    )
    assert result.exit_code != 0
    assert "set twice" in result.output


def test_coerce_hyperparam_coerces_per_schema_type():
    schema = {
        "max_missed_frames": (int, 5),
        "iou_threshold": (float, 0.5),
        "require_contiguous": (bool, True),
    }
    assert _coerce_hyperparam(schema, "max_missed_frames", "3") == 3
    assert _coerce_hyperparam(schema, "iou_threshold", "0.25") == 0.25
    assert _coerce_hyperparam(schema, "require_contiguous", "false") is False
    assert _coerce_hyperparam(schema, "require_contiguous", "TRUE") is True


def test_coerce_hyperparam_unknown_key_lists_allowed():
    with pytest.raises(click.ClickException, match="unknown hyperparam"):
        _coerce_hyperparam({"iou_threshold": (float, 0.5)}, "bogus", "1")


def test_coerce_hyperparam_bad_value():
    with pytest.raises(click.ClickException, match="as int"):
        _coerce_hyperparam({"max_missed_frames": (int, 5)}, "max_missed_frames", "x")


# --- model check + install with a fake model ------------------------------- #
class _FakeCheckedModel(DetectionModel):
    model_id = "fake_check"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]
    installed = False

    def load(self, device):
        pass

    def readiness(self):
        return ModelReadiness(True, "ok", "", None, "fake/repo")

    def install(self, progress_cb=None):
        type(self).installed = True

    def predict(self, frame, prompt):
        return FrameResult.from_detections(0, 0, [])

    def predict_batch(self, frames, prompts):
        return []


@contextlib.contextmanager
def _registered():
    register_model("fake_check")(_FakeCheckedModel)
    try:
        yield
    finally:
        MODELS.pop("fake_check", None)


class _FakeSizedModel(HFHubModel, DetectionModel):
    """A varianted model on the real HF readiness path — each size its own repo."""

    model_id = "fake_sized"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]
    VARIANTS = {"small": "fake/repo-small", "large": "fake/repo-large"}
    DEFAULT_VARIANT = "large"
    gated = False
    hf_files = ["config.json"]

    def __init__(self, variant=None):
        super().__init__(variant)
        self.hf_repo = self.weight

    def load(self, device):
        pass

    def predict(self, frame, prompt):
        return FrameResult.from_detections(0, 0, [])

    def predict_batch(self, frames, prompts):
        return []


@contextlib.contextmanager
def _registered_sized():
    register_model("fake_sized")(_FakeSizedModel)
    try:
        yield
    finally:
        MODELS.pop("fake_sized", None)


class _FakeClosedSetModel(DetectionModel):
    """A trained (closed-set) model for prompt-test: no cues, its own classes;
    one predict returns detections for every class."""

    supported_cue_types = []
    supported_backends = ["cpu"]

    def __init__(self, weights_path=None, classes=None):
        super().__init__(None)
        self._classes = list(classes or [])

    @property
    def classes(self):
        return list(self._classes)

    def load(self, device): ...

    def readiness(self):
        return ModelReadiness(True, "ok", "", None, None)

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 2, : w // 2] = True
        return FrameResult.from_detections(
            0,
            0,
            [
                Detection(mask, (0.1, 0.1, 0.3, 0.3), "mouse", 0.9, None),
                Detection(mask, (0.4, 0.4, 0.6, 0.6), "cage", 0.8, None),
            ],
        )

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


def _write_trained_model(proj_dir):
    from detail.storage import models as models_store

    mdir = models_store.model_dir(proj_dir, "model_001")
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "m",
            "status": "complete",
            "model_family": "fake_closed",
            "classes": [{"id": 0, "name": "mouse"}, {"id": 1, "name": "cage"}],
            "weights": {"best": "best.pt", "last": None},
        },
    )


def test_model_check_ok():
    with _registered():
        result = _runner().invoke(main, ["model", "check", "fake_check"])
    assert result.exit_code == 0, result.output
    assert "ready=True" in result.output
    assert "reason=ok" in result.output


def test_model_check_unknown():
    result = _runner().invoke(main, ["model", "check", "no_such_model"])
    assert result.exit_code != 0
    assert "unknown model_id" in result.output


def test_model_install_calls_install():
    _FakeCheckedModel.installed = False
    with _registered():
        result = _runner().invoke(main, ["model", "install", "fake_check"])
    assert result.exit_code == 0, result.output
    assert _FakeCheckedModel.installed is True


def test_model_install_targets_the_requested_variant(monkeypatch):
    """Each size is its own download, so `model install <id>` alone installs the
    default variant. Without --variant there was no way to install any other size —
    a run configured for one failed its pre-flight with nothing the user could do."""
    from detail.api import models as api_models

    seen = {}
    monkeypatch.setattr(
        api_models,
        "install_model",
        lambda model_id, progress_cb=None, *, variant=None: seen.update(
            args=(model_id, variant)
        ),
    )
    result = _runner().invoke(
        main,
        ["model", "install", "dinov3_convnext_base", "--variant", "convnext_tiny"],
    )
    assert result.exit_code == 0, result.output
    assert seen["args"] == ("dinov3_convnext_base", "convnext_tiny")
    assert "dinov3_convnext_base/convnext_tiny" in result.output


def test_model_check_reports_the_requested_variant():
    # The isolated cache is empty, so the variant is genuinely not installed — and
    # the message must name the command that installs *it*, not the default size.
    # Uses a stub rather than DINOv3: without the torch extra (the CI environment)
    # DINOv3 reports missing_dependency before it ever looks at the weights.
    with _registered_sized():
        result = _runner().invoke(
            main, ["model", "check", "fake_sized", "--variant", "small"]
        )
    assert result.exit_code == 0, result.output
    assert "fake_sized/small: ready=False" in result.output
    assert "detail model install fake_sized --variant small" in result.output


def test_model_list_shows_a_row_per_variant():
    """A run gates on the exact size it was configured with, so the catalog must
    report status per size — one line per model hid three of DINOv3's four."""
    with _registered():
        result = _runner().invoke(main, ["model", "list"])
    assert result.exit_code == 0, result.output
    assert "dinov3_convnext_base/convnext_tiny" in result.output
    assert "dinov3_convnext_base/convnext_large" in result.output
    assert "sam3 " in result.output  # no size axis → listed bare, no trailing variant


def test_model_list_shows_built_ins_with_install_status():
    """`model list` is the pre-flight check — a run never downloads weights, so the
    catalog must say what's installed. It listed imported models only, which left
    list_models() (what the GUI's Models tab shows) with no CLI surface at all."""
    with _registered():
        result = _runner().invoke(main, ["model", "list"])
    assert result.exit_code == 0, result.output
    assert "fake_check" in result.output  # a built-in, not just imported models
    assert "sam3" in result.output and "dinov3_convnext_base" in result.output
    # The isolated cache is empty, so nothing is installed — and saying so must not
    # have required the network.
    assert "not installed" in result.output


def _bundle(tmp_path, manifest):
    import yaml

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.yaml").write_text(
        yaml.dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return bundle


def test_recording_import_bundle(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    bundle = _bundle(
        tmp_path,
        {
            "recordings": [
                {"id": "recording_001", "metadata": {"animal_id": "M001"}},
                {"id": "recording_002", "metadata": {"animal_id": "M002"}},
            ]
        },
    )
    result = r.invoke(
        main, ["recording", "import", "--experiment", pdir, "--bundle", str(bundle)]
    )
    assert result.exit_code == 0, result.output
    assert "imported 2/2" in result.output
    for sid in ("recording_001", "recording_002"):
        assert (tmp_path / "p1" / "recordings" / sid / "recording.yaml").is_file()


def test_recording_import_bundle_draws_one_bar_over_the_batch(tmp_path, monkeypatch):
    """A bundle import used to pass no callback at all: silent for the whole batch.

    One bar for the batch, never reset between recordings, moving inside one, and always
    saying how many of how many are done (§6 rendering invariants).
    """
    import detail.cli as cli_mod
    from detail.core import progress as progress_mod

    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "1")
    calls = []

    class _Spy:
        def __call__(self, stage, fraction, detail):
            calls.append((stage, fraction, detail))

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "_StageProgress", _Spy)
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    bundle = _bundle(
        tmp_path,
        {
            "recordings": [
                {"id": "recording_001", "video": ["a.mp4"]},
                {"id": "recording_002", "video": ["b.mp4"]},
            ]
        },
    )
    _write_video(bundle / "a.mp4", n_frames=4)
    _write_video(bundle / "b.mp4", n_frames=4)
    result = r.invoke(
        main,
        ["recording", "import", "--experiment", str(tmp_path / "p1"),
         "--bundle", str(bundle)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "imported 2/2" in result.output
    assert {stage for stage, _, _ in calls} == {
        "Importing"
    }  # one bar, never re-labelled
    fractions = [f for _, f, _ in calls]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any(0 < f < 0.5 for f in fractions), "no movement inside the first recording"
    assert all(" done" in d for _, _, d in calls)
    assert any(d.startswith("0/2 done · recording_001: ") for _, _, d in calls)


def test_recording_import_dry_run_creates_nothing(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    bundle = _bundle(tmp_path, {"recordings": [{"id": "recording_001"}]})
    result = r.invoke(
        main,
        [
            "recording",
            "import",
            "--experiment",
            pdir,
            "--bundle",
            str(bundle),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert not (tmp_path / "p1" / "recordings" / "recording_001").exists()


def test_recording_import_validation_failure_imports_nothing(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    bundle = _bundle(tmp_path, {"recordings": [{"id": "ok"}, {"id": "ok"}]})  # dup id
    result = r.invoke(
        main, ["recording", "import", "--experiment", pdir, "--bundle", str(bundle)]
    )
    assert result.exit_code != 0
    assert "duplicate id" in result.output
    assert not (tmp_path / "p1" / "recordings" / "ok").exists()


def test_experiment_import_metadata(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(main, ["recording", "create", "--experiment", pdir, "--name", "s1"])
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "recording_id,treatment\nrecording_000,drug_A\n", encoding="utf-8"
    )
    result = r.invoke(
        main,
        ["experiment", "import-metadata", "--experiment", pdir, "--csv", str(csv_path)],
    )
    assert result.exit_code == 0, result.output
    assert "updated 1" in result.output


def test_experiment_import_metadata_keys_on_a_named_column(tmp_path):
    """No recognised key column: refused with the columns listed; `--key` names the one
    holding the video file name (#330)."""
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    exp = Experiment.create("p1", tmp_path)
    recording = exp.add_recording("m27")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"files": [{"index": 0, "origin_name": "/rig/m27_day3.mp4"}]}
    recording_store.write_recording(recording.dir, data)
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("Session Clip,dose\nm27_day3,0.5\n", encoding="utf-8")
    base = ["experiment", "import-metadata", "--experiment", str(exp.dir)]

    refused = _runner().invoke(main, [*base, "--csv", str(csv_path)])
    assert refused.exit_code != 0
    assert "Session Clip, dose" in refused.output

    result = _runner().invoke(
        main, [*base, "--csv", str(csv_path), "--key", "Session Clip"]
    )
    assert result.exit_code == 0, result.output
    assert exp.recording_metadata[recording.id] == {"dose": "0.5"}


def test_experiment_metadata_columns_round_trip(tmp_path):
    """The API had add/delete-column and set-metadata with no CLI at all, and no entry in
    the "Not in the CLI" ledger either — so filling 30 animals from a script was
    impossible."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    recording = exp.add_recording("m27_day3")
    pdir = str(exp.dir)
    r = CliRunner()

    assert (
        r.invoke(
            main, ["experiment", "add-column", "--experiment", pdir, "dose"]
        ).exit_code
        == 0
    )
    result = r.invoke(
        main,
        # by *name* — a script says what the recording is called, not its serial number
        [
            "experiment",
            "set-metadata",
            "--experiment",
            pdir,
            "--recording",
            "m27_day3",
            "--set",
            "dose=0.5",
            "--set",
            "day=3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert exp.recording_metadata[recording.id] == {"dose": "0.5", "day": "3"}

    assert (
        r.invoke(
            main, ["experiment", "delete-column", "--experiment", pdir, "dose"]
        ).exit_code
        == 0
    )
    assert "dose" not in exp.metadata_columns
    assert exp.recording_metadata[recording.id] == {"day": "3"}


def test_experiment_set_metadata_rejects_a_malformed_assignment(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.add_recording("m27_day3")
    result = CliRunner().invoke(
        main,
        [
            "experiment",
            "set-metadata",
            "--experiment",
            str(exp.dir),
            "--recording",
            "m27_day3",
            "--set",
            "justakey",
        ],
    )
    assert result.exit_code != 0 and "KEY=VALUE" in result.output


def test_recording_option_takes_a_name_or_an_id(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    recording = exp.add_recording("m27_day3")
    pdir = str(exp.dir)
    r = CliRunner()

    by_name = r.invoke(
        main, ["recording", "status", "--experiment", pdir, "--recording", "m27_day3"]
    )
    by_id = r.invoke(
        main, ["recording", "status", "--experiment", pdir, "--recording", recording.id]
    )
    assert by_name.exit_code == 0 and by_id.exit_code == 0
    assert by_name.output == by_id.output
    assert f"recording: m27_day3  (id {recording.id})" in by_name.output

    missing = r.invoke(
        main, ["recording", "status", "--experiment", pdir, "--recording", "ghost"]
    )
    assert missing.exit_code != 0
    assert "m27_day3" in missing.output  # says what does exist


def test_experiment_export_metadata(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(main, ["recording", "create", "--experiment", pdir, "--name", "s1"])
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "recording_id,treatment\nrecording_000,drug_A\n", encoding="utf-8"
    )
    r.invoke(
        main,
        ["experiment", "import-metadata", "--experiment", pdir, "--csv", str(csv_path)],
    )
    out = tmp_path / "template.csv"
    result = r.invoke(
        main,
        ["experiment", "export-metadata", "--experiment", pdir, "--csv", str(out)],
    )
    assert result.exit_code == 0, result.output
    # name and file ride along for whoever fills the template in; they are keys on the
    # way back, never stored columns. No video yet, so the file cell is blank.
    assert (
        out.read_text(encoding="utf-8-sig")
        == "recording_id,name,file,treatment\nrecording_000,s1,,drug_A\n"
    )


def _tri_kf():
    return [(0, [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])]


def _square_kf():
    return [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])]


def _shifted_kf():
    return [(0, [(0.15, 0.15), (0.9, 0.1), (0.9, 0.9), (0.1, 0.95)])]


def _set_resolution(recording, width, height):
    from detail.storage import recording as recording_store

    data = recording_store.read_recording(recording.dir)
    data["video"] = {"resolution": [width, height], "files": [], "storage": "default"}
    recording_store.write_recording(recording.dir, data)


def test_recording_export_import_spatial(tmp_path):
    """A setup exports once and can be pulled from either source onto a fresh recording."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    exp.create_recording("r2")
    src.set_rois([{"name": "zone", "keyframes": _tri_kf()}])
    r = _runner()
    pdir = str(exp.dir)
    out = tmp_path / "regions.csv"
    res = r.invoke(
        main,
        # fmt: off
        [
            "recording", "export-spatial",
            "--experiment", pdir, "--recording", "r1", "--csv", str(out),
        ],
        # fmt: on
    )
    assert res.exit_code == 0, res.output
    assert out.is_file()

    # ...onto r2 from the source recording
    res2 = r.invoke(
        main,
        # fmt: off
        [
            "recording", "import-spatial",
            "--experiment", pdir, "--recording", "r2", "--from-recording", "r1",
        ],
        # fmt: on
    )
    assert res2.exit_code == 0, res2.output
    assert res2.stdout.strip() == "imported 1/1 region(s)/point(s)"
    assert [e["name"] for e in exp.get_recording("r2").rois] == ["zone"]

    # ...and onto a third from the CSV, which is the same door.
    exp.create_recording("r3")
    res3 = r.invoke(
        main,
        # fmt: off
        [
            "recording", "import-spatial",
            "--experiment", pdir, "--recording", "r3", "--csv", str(out),
        ],
        # fmt: on
    )
    assert res3.exit_code == 0, res3.output
    assert res3.stdout.strip() == "imported 1/1 region(s)/point(s)"
    assert [e["name"] for e in exp.get_recording("r3").rois] == ["zone"]

    # giving both sources is an error
    res4 = r.invoke(
        main,
        # fmt: off
        [
            "recording", "import-spatial",
            "--experiment", pdir, "--recording", "r2",
            "--from-recording", "r1", "--csv", str(out),
        ],
        # fmt: on
    )
    assert res4.exit_code != 0 and "exactly one" in res4.output


def test_recording_import_spatial_takes_the_names_that_do_not_collide(tmp_path):
    """One colliding name refuses *that name*, and the rest still arrive.

    The refusal grain follows the kind, and regions are a set by design: refusing the
    whole set because one name clashes would kill "pull one more zone from a template",
    which is the ordinary use. A count alone cannot tell the two grains apart on a
    one-region source, so this drives a two-region one and asserts both halves — the
    non-colliding region lands, and the collision leaves *this* recording's geometry
    where it was rather than gaining a second "zone" beside it.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    src.define_roi("nest", _square_kf())
    src.define_roi("zone", _square_kf())
    dst = exp.create_recording("r2")
    mine = _tri_kf()
    dst.define_roi("zone", mine)

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "recording", "import-spatial",
            "--experiment", str(exp.dir), "--recording", "r2", "--from-recording", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "imported 1/2 region(s)/point(s)" in res.stdout
    assert sorted(e["name"] for e in dst.rois) == ["nest", "zone"]
    assert "not imported: zone" in res.stderr
    assert "already exists on" in res.stderr
    kept = next(e for e in dst.rois if e["name"] == "zone")
    assert dst.roi_keyframes(kept["id"]) == mine  # not overwritten by the source's


def test_recording_import_spatial_leaves_a_windowed_region_behind(tmp_path):
    """An active frame window names one recording's timeline, so it does not travel.

    Neither answer is safe on the other recording — keeping (5000, 9000) on a shorter one
    leaves a region that applies to no frame, dropping it widens the claim to the whole
    video — so the region is refused *by name*, loudly, instead of arriving either way.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    src.define_roi("nest", _square_kf())
    src.define_roi("timed zone", _square_kf(), active=(5000, 9000))
    dst = exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "recording", "import-spatial",
            "--experiment", str(exp.dir), "--recording", "r2", "--from-recording", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "imported 1/2 region(s)/point(s)" in res.stdout
    assert "not imported: timed zone" in res.stderr
    assert "active frame window" in res.stderr
    assert [e["name"] for e in dst.rois] == ["nest"]


def test_recording_import_spatial_flattens_camera_motion_and_says_so(tmp_path):
    """A keyframed polygon arrives at frame 0, and the command names it.

    The flattening is silent in the data — the region looks like any other static one — so
    the report is the only place a user can learn that what they pulled is not what they
    drew. Asserting the stored keyframes as well as the line keeps the two honest: a
    message with the motion still stored, or motion dropped with nothing said, both fail.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    at_zero = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    later = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    src.define_roi("panning rig", [(0, at_zero), (300, later)])
    dst = exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "recording", "import-spatial",
            "--experiment", str(exp.dir), "--recording", "r2", "--from-recording", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "camera motion dropped (copied at frame 0): panning rig" in res.stdout
    assert dst.roi_keyframes(dst.rois[0]["id"]) == [(0, at_zero)]


def test_recording_import_fov_and_calibration(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    dst = exp.create_recording("r2")
    _set_resolution(src, 1920, 1080)
    src.define_arena("rig", _square_kf())
    src.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    r = _runner()
    pdir = str(exp.dir)

    res = r.invoke(
        main,
        ["recording", "import-fov", "--experiment", pdir, "--recording", "r2",
         "--from-recording", "r1"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert res.stdout.strip() == "imported 1/1 field(s) of view"
    assert [a["name"] for a in dst.arenas] == ["rig"]

    res2 = r.invoke(
        main,
        ["recording", "import-calibration", "--experiment", pdir, "--recording", "r2",
         "--from-recording", "r1"],
    )  # fmt: skip
    assert res2.exit_code == 0, res2.output
    assert "4.8 px/mm at 1920x1080" in res2.output

    # Both round-trip through the one exported CSV as well.
    out = tmp_path / "setup.csv"
    r.invoke(
        main,
        ["recording", "export-spatial", "--experiment", pdir, "--recording", "r1",
         "--csv", str(out)],
    )  # fmt: skip
    third = exp.create_recording("r3")
    for cmd in ("import-fov", "import-calibration"):
        res3 = r.invoke(
            main,
            ["recording", cmd, "--experiment", pdir, "--recording", "r3",
             "--csv", str(out)],
        )  # fmt: skip
        assert res3.exit_code == 0, res3.output
    assert [a["name"] for a in third.arenas] == ["rig"]
    assert third.distance_calibration["pixels_per_mm"] == pytest.approx(4.8)

    # Same "exactly one source" guard as import-spatial.
    res4 = r.invoke(
        main,
        ["recording", "import-fov", "--experiment", pdir, "--recording", "r2"],
    )  # fmt: skip
    assert res4.exit_code != 0 and "exactly one" in res4.output


def test_recording_import_fov_is_refused_onto_a_recording_that_has_one(tmp_path):
    """A field of view is refused *whole*, and the one already there survives untouched.

    The grain differs from the regions on purpose, so the two need separate guards: a
    second field of view is not a richer setup — the batch tracking picker intersects
    arena *ids* across recordings — so having one at all is the refusal. This used to
    clear and copy over the top, which is the same operation the push direction refuses;
    asserting the surviving polygon's keyframes, not just its name, is what separates
    "refused" from "replaced by a same-named one".
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    src.define_arena("rig", _square_kf())
    dst = exp.create_recording("r2")
    mine = _shifted_kf()
    dst.define_arena("old", mine)

    res = _runner().invoke(
        main,
        ["recording", "import-fov", "--experiment", str(exp.dir), "--recording", "r2",
         "--from-recording", "r1"],
    )  # fmt: skip

    assert res.exit_code != 0
    assert "already has a field of view" in res.output
    assert [a["name"] for a in dst.arenas] == ["old"]
    assert dst.arena_keyframes(dst.arenas[0]["id"]) == mine


def test_recording_import_fov_refuses_while_a_tracking_run_uses_it(tmp_path):
    """The polygon a completed tracking run filtered on is never swapped out from under it.

    Under the "never replaces" rule this is refused for having a field of view at all,
    which is the stronger reason — but the property that matters is the one this asserts:
    the run's own arena id still resolves to the polygon it recorded.
    """
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    dst = exp.create_recording("r2")
    src.define_arena("rig", _square_kf())
    in_use = _shifted_kf()
    arena_id = dst.define_arena("in use", in_use)
    data = recording_store.read_recording(dst.dir)
    data["tracking_runs"] = [
        {"id": "track_01", "status": "complete", "arena_ids": [arena_id]}
    ]
    recording_store.write_recording(dst.dir, data)

    res = _runner().invoke(
        main,
        ["recording", "import-fov", "--experiment", str(exp.dir), "--recording", "r2",
         "--from-recording", "r1"],
    )  # fmt: skip
    assert res.exit_code != 0
    assert [a["name"] for a in dst.arenas] == ["in use"]
    assert dst.arena_keyframes(arena_id) == in_use
    assert dst.arena_dependents(arena_id) == ["track_01"]


def test_recording_import_calibration_is_refused_onto_a_calibrated_recording(tmp_path):
    """One pixel scale per recording, so an existing calibration is kept, not replaced."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)
    dst = exp.create_recording("r2")
    _set_resolution(dst, 1920, 1080)
    dst.set_pixels_per_mm(99.0)

    res = _runner().invoke(
        main,
        ["recording", "import-calibration", "--experiment", str(exp.dir),
         "--recording", "r2", "--from-recording", "r1"],
    )  # fmt: skip

    assert res.exit_code != 0
    assert "already has a distance calibration" in res.output
    assert dst.distance_calibration["pixels_per_mm"] == pytest.approx(99.0)


def test_recording_import_calibration_is_refused_onto_a_video_of_another_shape(
    tmp_path,
):
    """One pixels-per-mm cannot describe a video of another aspect ratio; and a
    calibration left behind by a video swap is shown as unusable, not as a number."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    _set_resolution(src, 1920, 1080)
    src.set_distance_calibration([(0.5, 0.25), (0.5, 0.75)], 200.0)
    dst = exp.create_recording("r2")
    _set_resolution(dst, 1280, 1024)

    res = _runner().invoke(
        main,
        ["recording", "import-calibration", "--experiment", str(exp.dir),
         "--recording", "r2", "--from-recording", "r1"],
    )  # fmt: skip
    assert res.exit_code != 0
    assert "different video shape" in res.output
    assert dst.distance_calibration is None

    _set_resolution(src, 1280, 1024)
    st = _runner().invoke(
        main, ["recording", "status", "--experiment", str(exp.dir), "--recording", "r1"]
    )
    assert st.exit_code == 0, st.output
    assert "calibration: unusable" in st.output


def test_recording_add_stream_timeseries_csv(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text("seconds,bpm\n0.0,72\n0.5,73\n", encoding="utf-8")
    r = _runner()
    res = r.invoke(
        main,
        [
            "recording",
            "add-stream",
            "--experiment",
            str(exp.dir),
            "--recording",
            "r1",
            "--name",
            "hr",
            "--importer",
            "timeseries_csv",
            "--data",
            str(csv_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "hr" in res.output and "2 rows" in res.output
    entry = exp.get_recording("r1").streams[0]
    assert entry["shape"] == "continuous"
    assert entry["clock"] == "stream_seconds"


def test_recording_add_stream_maps_columns_and_sets_the_offset(tmp_path):
    """The CLI reaches the whole importer config, not just the sniffing default.

    The mapping flags are what make an arbitrary export importable without the GUI, and
    ``--offset`` is what makes a 100-recording import a shell loop rather than 100 scrubs.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    csv_path = tmp_path / "pose.csv"
    csv_path.write_text(
        "t_sec,nose_x,nose_y,tail_x\n0,1,2,3\n0.1,4,5,6\n", encoding="utf-8"
    )
    res = _runner().invoke(
        main,
        # fmt: off
        [
            "recording", "add-stream",
            "--experiment", str(exp.dir), "--recording", "r1",
            "--name", "pose", "--importer", "timeseries_csv", "--data", str(csv_path),
            "--time-column", "t_sec", "--time-is", "seconds",
            "--channel", "nose_x", "--channel", "nose_y=ny",
            "--offset", "3.48",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    entry = exp.get_recording("r1").streams[0]
    assert entry["columns"] == ["t_ns", "nose_x", "ny"]  # tail_x was not asked for
    assert entry["alignment"]["offset_ns"] == 3_480_000_000


def test_recording_set_stream_offset(tmp_path):
    """The scriptable half of alignment. Without it the only way to set an offset on an
    existing stream is the GUI or Python, which is unusable over a whole experiment."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text("seconds,bpm\n0.0,72\n0.5,73\n", encoding="utf-8")
    r = _runner()
    common = ["--experiment", str(exp.dir), "--recording", "r1"]
    r.invoke(
        main,
        [
            "recording",
            "add-stream",
            *common,
            "--name",
            "hr",
            "--importer",
            "timeseries_csv",
            "--data",
            str(csv_path),
        ],
    )

    res = r.invoke(
        main,
        [
            "recording",
            "set-stream-offset",
            *common,
            "--name",
            "hr",
            "--offset",
            "-1.25",
        ],
    )

    assert res.exit_code == 0, res.output
    assert "-1.250s" in res.output
    entry = exp.get_recording("r1").streams[0]
    assert entry["alignment"]["offset_ns"] == -1_250_000_000


def test_recording_set_stream_offset_names_an_unknown_stream(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    res = _runner().invoke(
        main,
        # fmt: off
        [
            "recording", "set-stream-offset",
            "--experiment", str(exp.dir), "--recording", "r1",
            "--name", "nope", "--offset", "1.0",
        ],
        # fmt: on
    )

    assert res.exit_code != 0
    assert "nope" in res.output


def test_stream_preset_saved_then_applied_in_bulk(tmp_path):
    """Map once, apply to the rest — the whole reason presets exist, from the CLI alone.

    The sources table carries an offset *per recording*, because that is exactly what a
    preset cannot: each video and logger pair has its own skew.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    for rid in ("r1", "r2", "r3"):
        exp.create_recording(rid)
        (tmp_path / f"pose_{rid}.csv").write_text(
            "t_sec,nose_x,nose_y,extra\n0,1,2,9\n0.1,3,4,9\n", encoding="utf-8"
        )
    (tmp_path / "sources.csv").write_text(
        "recording,file,offset_s\nr2,pose_r2.csv,3.512\nr3,pose_r3.csv,\n",
        encoding="utf-8",
    )
    r = _runner()

    saved = r.invoke(
        main,
        # fmt: off
        [
            "recording", "add-stream",
            "--experiment", str(exp.dir), "--recording", "r1",
            "--name", "pose", "--importer", "timeseries_csv",
            "--data", str(tmp_path / "pose_r1.csv"),
            "--time-column", "t_sec", "--time-is", "seconds",
            "--channel", "nose_x", "--channel", "nose_y",
            "--offset", "3.48", "--save-preset", "dlc_pose",
        ],
        # fmt: on
    )
    assert saved.exit_code == 0, saved.output

    listed = r.invoke(
        main, ["experiment", "list-stream-presets", "--experiment", str(exp.dir)]
    )
    assert "dlc_pose" in listed.output and "nose_x, nose_y" in listed.output

    applied = r.invoke(
        main,
        # fmt: off
        [
            "experiment", "apply-stream-preset",
            "--experiment", str(exp.dir), "--preset", "dlc_pose", "--name", "pose",
            "--sources", str(tmp_path / "sources.csv"),
        ],
        # fmt: on
    )

    assert applied.exit_code == 0, applied.output
    assert "imported 2, skipped 0" in applied.output
    offsets = [
        exp.get_recording(rid).streams[0]["alignment"]["offset_ns"]
        for rid in ("r1", "r2", "r3")
    ]
    assert offsets == [3_480_000_000, 3_512_000_000, 0]


def test_add_stream_takes_a_preset_instead_of_an_importer(tmp_path):
    """One recording at a time, ``--preset`` replaces ``--importer``.

    ``--importer`` was ``required=True`` while ``--preset``'s own help read "instead of
    --importer", and the body branches on ``preset is not None`` and reads the importer
    back off the preset. So click refused the documented single-recording form before
    the branch could run: ``Missing option '--importer'``.

    It survived because the preset tests drive ``experiment apply-stream-preset``, the
    *batch* form, which has no ``--importer`` to be required. The one-at-a-time form in
    ``docs/cli.md`` had no test at all. Both refusals are asserted too: passing neither
    and passing both are the two ways to mean nothing.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    for rid in ("r1", "r2"):
        exp.create_recording(rid)
        (tmp_path / f"pose_{rid}.csv").write_text(
            "t_sec,nose_x,nose_y,extra\n0,1,2,9\n0.1,3,4,9\n", encoding="utf-8"
        )
    r = _runner()
    saved = r.invoke(
        main,
        # fmt: off
        [
            "recording", "add-stream",
            "--experiment", str(exp.dir), "--recording", "r1",
            "--name", "pose", "--importer", "timeseries_csv",
            "--data", str(tmp_path / "pose_r1.csv"),
            "--time-column", "t_sec", "--time-is", "seconds",
            "--channel", "nose_x", "--channel", "nose_y",
            "--save-preset", "dlc_pose",
        ],
        # fmt: on
    )
    assert saved.exit_code == 0, saved.output

    applied = r.invoke(
        main,
        # fmt: off
        [
            "recording", "add-stream",
            "--experiment", str(exp.dir), "--recording", "r2",
            "--name", "pose", "--data", str(tmp_path / "pose_r2.csv"),
            "--preset", "dlc_pose", "--offset", "3.49",
        ],
        # fmt: on
    )

    assert applied.exit_code == 0, applied.output
    entry = exp.get_recording("r2").streams[0]
    assert entry["importer"] == "timeseries_csv"
    # The preset's *mapping* applied, not a default sniff: `extra` is dropped because the
    # preset names two channels, and the time column is the one the preset recorded.
    assert entry["columns"] == ["t_ns", "nose_x", "nose_y"]
    assert entry["metadata"]["time_column"] == "t_sec"
    assert entry["alignment"]["offset_ns"] == 3_490_000_000

    base = [
        "recording", "add-stream",
        "--experiment", str(exp.dir), "--recording", "r2",
        "--name", "again", "--data", str(tmp_path / "pose_r2.csv"),
    ]  # fmt: skip
    neither = r.invoke(main, base)
    both = r.invoke(
        main, base + ["--importer", "timeseries_csv", "--preset", "dlc_pose"]
    )
    for res in (neither, both):
        assert res.exit_code != 0
        assert "exactly one of --importer or --preset" in res.output


def test_apply_stream_preset_reports_a_skipped_recording(tmp_path):
    """One unreadable file is reported and skipped, not fatal to the batch."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    exp.create_recording("r2")
    (tmp_path / "good.csv").write_text("t_sec,nose_x\n0,1\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_text("t_sec,other\n0,1\n", encoding="utf-8")
    (tmp_path / "sources.csv").write_text("r1,good.csv\nr2,bad.csv\n", encoding="utf-8")
    exp.save_stream_preset(
        "p",
        "timeseries_csv",
        {
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [{"column": "nose_x", "name": "nose_x"}],
        },
    )

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "apply-stream-preset",
            "--experiment", str(exp.dir), "--preset", "p", "--name", "pose",
            "--sources", str(tmp_path / "sources.csv"),
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "imported 1, skipped 1" in res.output
    assert exp.get_recording("r1").streams
    assert not exp.get_recording("r2").streams


def test_apply_stream_preset_keys_on_a_video_column(tmp_path):
    """A `video` header on the first column keys each row on the video file name (#330)."""
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    exp = Experiment.create("p1", tmp_path)
    recording = exp.create_recording("r1")
    data = recording_store.read_recording(recording.dir)
    data["video"] = {"files": [{"index": 0, "origin_name": "/rig/M27_day3.mp4"}]}
    recording_store.write_recording(recording.dir, data)
    (tmp_path / "good.csv").write_text("t_sec,nose_x\n0,1\n", encoding="utf-8")
    (tmp_path / "sources.csv").write_text(
        "Video,file,offset_s\nm27_day3,good.csv,\n", encoding="utf-8"
    )
    exp.save_stream_preset(
        "p",
        "timeseries_csv",
        {
            "time_column": "t_sec",
            "time_kind": "seconds",
            "channels": [{"column": "nose_x", "name": "nose_x"}],
        },
    )

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "apply-stream-preset",
            "--experiment", str(exp.dir), "--preset", "p", "--name", "pose",
            "--sources", str(tmp_path / "sources.csv"),
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "imported 1, skipped 0" in res.output
    assert exp.get_recording("r1").streams


# --- experiment propagate-* (Section 10, issue #342) ------------------------- #
def test_propagate_fov_defaults_to_every_other_recording(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1").define_arena("rig", _square_kf())
    exp.create_recording("r2")
    exp.create_recording("r3")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-fov",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "copied to 2/2 recording(s)" in res.output
    assert [a["name"] for a in exp.get_recording("r2").arenas] == ["rig"]
    assert [a["name"] for a in exp.get_recording("r3").arenas] == ["rig"]


def test_propagate_fov_only_limits_the_targets(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1").define_arena("rig", _square_kf())
    exp.create_recording("r2")
    exp.create_recording("r3")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-fov",
            "--experiment", str(exp.dir), "--source", "r1", "--only", "r2",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "copied to 1/1 recording(s)" in res.output
    assert [a["name"] for a in exp.get_recording("r2").arenas] == ["rig"]
    assert exp.get_recording("r3").arenas == []


def test_propagate_counts_what_it_did_not_what_was_asked_for(tmp_path):
    """`--only <the source>` asks for one recording and copies to none.

    The count comes from the result, never the request. Counting the request reported
    "1/1" for exactly this — a number a person reads and acts on, describing a copy that
    never happened.
    """
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1").define_arena("rig", _square_kf())
    exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-fov",
            "--experiment", str(exp.dir), "--source", "r1", "--only", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "copied to 0/0 recording(s)" in res.output
    assert exp.get_recording("r2").arenas == []


def test_propagate_fov_reports_an_occupied_target_and_keeps_going(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1").define_arena("rig", _square_kf())
    exp.create_recording("r2")
    exp.create_recording("r3").define_arena("its own", _square_kf())

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-fov",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "copied to 1/2 recording(s)" in res.output
    assert "skipped r3" in res.output and "already has" in res.output
    # The one that could take it still did — a refusal is not fatal to the batch.
    assert [a["name"] for a in exp.get_recording("r2").arenas] == ["rig"]
    assert [a["name"] for a in exp.get_recording("r3").arenas] == ["its own"]


def test_propagate_fov_reports_flattened_camera_motion(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    moving = _square_kf() + [(900, _square_kf()[0][1])]
    exp.create_recording("r1").define_arena("panning rig", moving)
    exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-fov",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "camera motion dropped" in res.output and "panning rig" in res.output


def test_propagate_fov_requires_a_source(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-fov",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code != 0
    assert "no field of view to copy" in res.output


def test_propagate_calibration_via_cli(tmp_path):
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    exp = Experiment.create("p1", tmp_path)
    for rid in ("r1", "r2"):
        rec = exp.create_recording(rid)
        data = recording_store.read_recording(rec.dir)
        data["video"] = {
            "resolution": [1920, 1080],
            "files": [],
            "storage": "default",
        }
        recording_store.write_recording(rec.dir, data)
    exp.get_recording("r1").set_distance_calibration([(0.25, 0.5), (0.75, 0.5)], 200.0)

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-calibration",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "copied to 1/1 recording(s)" in res.output
    assert abs(exp.get_recording("r2").pixels_per_mm - 4.8) < 1e-9


def test_propagate_rois_via_cli(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1").define_roi("zone", _square_kf())
    exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-rois",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert [r["name"] for r in exp.get_recording("r2").rois] == ["zone"]


def test_propagate_rois_leaves_a_windowed_region_behind(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    src = exp.create_recording("r1")
    src.define_roi("nest", _square_kf())
    src.define_roi("timed zone", _square_kf(), active=(5000, 9000))
    exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-rois",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert "not copied: timed zone" in res.output
    assert [r["name"] for r in exp.get_recording("r2").rois] == ["nest"]


def test_propagate_points_via_cli(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1").define_point("lever", [(0, [(0.5, 0.5)])])
    exp.create_recording("r2")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "propagate-points",
            "--experiment", str(exp.dir), "--source", "r1",
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert [p["name"] for p in exp.get_recording("r2").points] == ["lever"]


@pytest.mark.parametrize(
    "table, match",
    [
        ("r1\n", "expected"),
        ("r1,f.csv,notanumber\n", "not a number of seconds"),
        ("\n", "lists no recordings"),
    ],
)
def test_a_malformed_sources_table_is_refused(tmp_path, table, match):
    """Reported before any recording is touched, naming the line (Tenet 10)."""
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.save_stream_preset("p", "timeseries_csv", {})
    (tmp_path / "sources.csv").write_text(table, encoding="utf-8")

    res = _runner().invoke(
        main,
        # fmt: off
        [
            "experiment", "apply-stream-preset",
            "--experiment", str(exp.dir), "--preset", "p", "--name", "pose",
            "--sources", str(tmp_path / "sources.csv"),
        ],
        # fmt: on
    )

    assert res.exit_code != 0
    assert match in res.output


def test_stream_preset_export_import_between_experiments(tmp_path):
    from detail.api.experiment import Experiment

    source = Experiment.create("src", tmp_path)
    dest = Experiment.create("dst", tmp_path)
    source.save_stream_preset("dlc", "timeseries_csv", {"time_column": "t_sec"})
    r = _runner()

    r.invoke(
        main,
        # fmt: off
        [
            "experiment", "export-stream-preset", "--experiment", str(source.dir),
            "dlc", "--out", str(tmp_path / "dlc.yaml"),
        ],
        # fmt: on
    )
    res = r.invoke(
        main,
        # fmt: off
        [
            "experiment", "import-stream-preset", "--experiment", str(dest.dir),
            str(tmp_path / "dlc.yaml"),
        ],
        # fmt: on
    )

    assert res.exit_code == 0, res.output
    assert dest.get_stream_preset("dlc")["config"]["time_column"] == "t_sec"


def test_recording_set_and_clear_calibration(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    rec = exp.create_recording("r1")
    data = recording_store.read_recording(rec.dir)
    data["video"] = {"resolution": [1920, 1080], "files": [], "storage": "default"}
    recording_store.write_recording(rec.dir, data)
    r = _runner()
    pdir = str(exp.dir)
    res = r.invoke(
        main,
        [
            "recording",
            "set-calibration",
            "--experiment",
            pdir,
            "--recording",
            "r1",
            "--pixels-per-mm",
            "4.8",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "4.8 px/mm" in res.output
    assert exp.get_recording("r1").distance_calibration["pixels_per_mm"] == 4.8
    st = r.invoke(
        main, ["recording", "status", "--experiment", pdir, "--recording", "r1"]
    )
    assert "calibration: 4.8 px/mm" in st.output
    res2 = r.invoke(
        main,
        ["recording", "clear-calibration", "--experiment", pdir, "--recording", "r1"],
    )
    assert res2.exit_code == 0
    assert exp.get_recording("r1").distance_calibration is None


def test_experiment_import_metadata_unknown_fails(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    csv_path = tmp_path / "m.csv"
    csv_path.write_text("recording_id,treatment\nghost,x\n", encoding="utf-8")
    result = r.invoke(
        main,
        ["experiment", "import-metadata", "--experiment", pdir, "--csv", str(csv_path)],
    )
    assert result.exit_code != 0
    assert "ghost" in result.output


def test_experiment_export_import(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "ws")
    r = _runner()
    r.invoke(main, ["experiment", "create", "study", "--dir", str(tmp_path / "src")])
    pdir = str(tmp_path / "src" / "study")
    r.invoke(main, ["recording", "create", "--experiment", pdir, "--name", "s1"])
    archive = tmp_path / "study.tar"
    res = r.invoke(
        main, ["experiment", "export", "--experiment", pdir, "--out", str(archive)]
    )
    assert res.exit_code == 0, res.output
    assert "exported" in res.output and archive.is_file()
    res2 = r.invoke(main, ["experiment", "import", str(archive)])
    assert res2.exit_code == 0, res2.output
    assert (tmp_path / "ws" / "study" / "config.yaml").is_file()
    assert "video not found" not in res2.output


def test_experiment_import_names_the_recordings_whose_video_was_not_found(
    tmp_path, monkeypatch
):
    """An import looks for referenced video where the exporting machine had it (#916), and says
    which recordings it could not find, by name, and what to run about it."""
    from detail.storage import locations
    from detail.storage import recording as recording_store

    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "ws")
    r = _runner()
    r.invoke(main, ["experiment", "create", "study", "--dir", str(tmp_path / "src")])
    pdir = tmp_path / "src" / "study"
    r.invoke(main, ["recording", "create", "--experiment", str(pdir), "--name", "s1"])
    directory = pdir / "recordings" / "recording_000"  # named s1; the id is minted
    data = recording_store.read_recording(directory)
    data["video"] = {
        "files": [{"id": "part_001", "source": {"root": "rig", "rel": "a.mp4"}}]
    }
    recording_store.write_recording(directory, data)
    archive = tmp_path / "study.tar"
    r.invoke(
        main, ["experiment", "export", "--experiment", str(pdir), "--out", str(archive)]
    )

    res = r.invoke(main, ["experiment", "import", str(archive)])

    assert res.exit_code == 0, res.output
    assert (
        "video not found on this machine for 1 recording(s): s1. (A file at the same "
        "path that is a different video counts as not found.) Relink one file "
        "(`detail recording relink`) and every recording whose video is in the same "
        "folder finds it too."
    ) in res.output


def _study_with_copied_video(tmp_path, monkeypatch):
    """``study`` in the workspace with one recording whose video was copied in, and an export
    record: the experiment the "moving experiments between machines" loop brings home.
    """
    from detail.storage import experiment as experiment_store
    from detail.storage import locations

    (tmp_path / "cfg").mkdir()
    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "ws")
    r = _runner()
    r.invoke(main, ["experiment", "create", "study", "--dir", str(tmp_path / "ws")])
    pdir = tmp_path / "ws" / "study"
    r.invoke(main, ["recording", "create", "--experiment", str(pdir), "--name", "m1"])
    video = tmp_path / "trial.mp4"
    _write_video(video)
    res = r.invoke(
        main,
        ["recording", "add-video", "--experiment", str(pdir), "--recording", "m1",
         "--video", str(video), "--no-preview"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    (record := experiment_store.exports_dir(pdir) / "export_01").mkdir(parents=True)
    (record / "export.yaml").write_text("id: export_01\n", encoding="utf-8")
    (copied,) = (pdir / "recordings").rglob("*.mp4")
    return r, pdir, copied


def test_experiment_import_replace_prints_what_it_deletes_first(tmp_path, monkeypatch):
    """`--replace` with a slim bundle deleted the experiment's copied video and export records
    and printed only ``imported:`` (#29). It still deletes them (the owner's decision), and
    says so first, in the words the GUI's prompt uses (`api.experiment.replace_warning`).
    """
    from detail.api.exporters import format_size

    r, pdir, copied = _study_with_copied_video(tmp_path, monkeypatch)
    size = format_size(copied.stat().st_size)
    archive = tmp_path / "study.tar"
    r.invoke(
        main, ["experiment", "export", "--experiment", str(pdir), "--out", str(archive)]
    )

    refused = r.invoke(main, ["experiment", "import", str(archive)])
    assert refused.exit_code != 0
    assert "--replace would also delete:\n" in refused.output
    assert f"  - 1 video file(s), {size}: the bundle carries no video" in refused.output

    res = r.invoke(main, ["experiment", "import", str(archive), "--replace"])

    assert res.exit_code == 0, res.output
    assert res.stdout.startswith(
        "replacing experiment 'study' deletes what the bundle does not bring back:\n"
        f"  - 1 video file(s), {size}: the bundle carries no video\n"
        "  - 1 export record(s), export_01: no bundle carries them (the exported files "
        "themselves are not touched)\n"
        "imported: "
    ), res.stdout
    assert not copied.exists()


def test_experiment_import_replace_with_nothing_lost_prints_no_list(
    tmp_path, monkeypatch
):
    """A full bundle brings the video back, and with no export record nothing is lost: no
    list, so the list means something when it appears."""
    r, pdir, copied = _study_with_copied_video(tmp_path, monkeypatch)
    import shutil

    shutil.rmtree(pdir / "exports")
    archive = tmp_path / "study.tar"
    r.invoke(
        main,
        ["experiment", "export", "--experiment", str(pdir), "--out", str(archive),
         "--include-video"],
    )  # fmt: skip

    res = r.invoke(main, ["experiment", "import", str(archive), "--replace"])

    assert res.exit_code == 0, res.output
    assert res.stdout.startswith("imported: "), res.stdout
    assert copied.is_file()


def test_experiment_export_names_a_symlinked_folder_it_left_out(tmp_path):
    import os

    r = _runner()
    r.invoke(main, ["experiment", "create", "study", "--dir", str(tmp_path / "src")])
    pdir = tmp_path / "src" / "study"
    (tmp_path / "elsewhere").mkdir()
    try:
        os.symlink(tmp_path / "elsewhere", pdir / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("this machine cannot make a symlink")
    archive = tmp_path / "study.tar"

    res = r.invoke(
        main, ["experiment", "export", "--experiment", str(pdir), "--out", str(archive)]
    )

    assert res.exit_code == 0, res.output
    assert "left out linked: it is a symlinked folder, which is not followed\n" in (
        res.output
    )


def test_experiment_export_and_import_drive_a_bar_through_every_phase(
    tmp_path, monkeypatch
):
    """Both commands printed nothing until they were done (#916): a slim export of a real
    experiment is 30k files and 17 GB, read twice. Each drives one bar now, and it is relabelled
    as the phase changes rather than holding the first label throughout."""
    import detail.cli as cli_mod
    from detail.core import progress as progress_mod
    from detail.storage import locations

    monkeypatch.setattr(locations, "STORAGE_PATH", tmp_path / "cfg" / "storage.yaml")
    monkeypatch.setattr(locations, "DEFAULT_ROOT", tmp_path / "ws")
    monkeypatch.setattr(progress_mod, "PROGRESS_INTERVAL_S", 0.0)
    calls = []

    class _Spy:
        def __call__(self, stage, fraction, detail):
            calls.append((stage, fraction))

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "_StageProgress", _Spy)
    r = _runner()
    r.invoke(main, ["experiment", "create", "study", "--dir", str(tmp_path / "src")])
    pdir = str(tmp_path / "src" / "study")
    archive = tmp_path / "study.tar"
    res = r.invoke(
        main, ["experiment", "export", "--experiment", pdir, "--out", str(archive)]
    )
    assert res.exit_code == 0, res.output
    assert {s for s, _ in calls} == {"scanning", "checking", "hashing", "packing"}
    assert calls[-1] == ("packing", 1.0)

    calls.clear()
    res = r.invoke(main, ["experiment", "import", str(archive)])
    assert res.exit_code == 0, res.output
    assert {s for s, _ in calls} == {"reading", "extracting", "verifying", "placing"}
    assert calls[-1] == ("placing", 1.0)


# --- progress rendering (tqdm bar, not a line per event) -------------------- #
class _FakeRun:
    def __init__(self, events):
        self._events = events
        self.status = "complete"
        self.id = "run_01"

    def progress(self):
        yield from self._events

    def wait(self):
        return self


def _capture_stream_progress(events, label):
    cap = io.StringIO()
    real = sys.stderr
    sys.stderr = cap
    try:
        _stream_progress(_FakeRun(events), label)
    finally:
        sys.stderr = real
    return cap.getvalue()


def test_stream_progress_redraws_in_place_not_one_line_per_event():
    events = [
        DetectionProgressEvent(
            parts_done=0,
            parts_total=1,
            frames_done=f,
            frames_total=1000,
            elapsed_s=0.1,
            eta_s=1.0,
        )
        for f in (200, 400, 600, 800, 1000)
    ]
    out = _capture_stream_progress(events, "detection")
    # one bar redrawn in place (carriage returns), not five separate lines
    assert out.count("\n") <= 1
    assert "\r" in out
    assert "detection" in out


def test_stream_progress_embedding_uses_frame_bar_with_unit_postfix():
    events = [
        EmbeddingProgressEvent(
            track_id=3,
            parts_done=0,
            parts_total=1,
            frames_done=1840,
            frames_total=2000,
            elapsed_s=0.1,
            eta_s=5.0,
            units_done=2,
            units_total=5,
        )
    ]
    out = _capture_stream_progress(events, "embedding")
    assert "1840/2000" in out  # bar keyed on the real frame total -> %/ETA
    assert "frame" in out  # the bar's unit label
    assert "track 3" in out and "unit 2/5" in out  # postfix carries the unit pos


def test_stream_progress_writes_phase_message_above_bar():
    events = [
        DetectionProgressEvent(
            parts_done=0,
            parts_total=1,
            frames_done=0,
            frames_total=100,
            elapsed_s=0.0,
            eta_s=None,
            message="loading model",
        ),
        DetectionProgressEvent(
            parts_done=0,
            parts_total=1,
            frames_done=50,
            frames_total=100,
            elapsed_s=0.1,
            eta_s=None,
        ),
    ]
    out = _capture_stream_progress(events, "detection")
    assert "loading model" in out


def _capture(fn, run, label):
    cap = io.StringIO()
    real = sys.stderr
    sys.stderr = cap
    try:
        fn(run, label)
    finally:
        sys.stderr = real
    return cap.getvalue()


def test_stream_processing_progress_renders_message_and_recording():
    # #18: the multi-recording bar must show the inner phase message AND the recording
    # position — this mapping was previously untested.
    from detail.cli import _stream_processing_progress
    from detail.core.dataclasses import DetectionProgressEvent, ProcessingProgress

    inner = DetectionProgressEvent(0, 1, 5, 10, 1.0, None, message="loading model")
    events = [
        ProcessingProgress(0, 2, "s1", inner, elapsed_s=1.0),
        ProcessingProgress(
            0, 2, "s1", DetectionProgressEvent(0, 1, 10, 10, 2.0, None), elapsed_s=2.0
        ),
    ]
    out = _capture(_stream_processing_progress, _FakeRun(events), "detection")
    assert "loading model" in out  # inner phase message above the bar
    assert "recording s1 (1/2)" in out  # recording position in the postfix


def test_stream_training_progress_renders_message_and_metrics():
    # #18: the training bar must show the phase message (e.g. "loading base model")
    # and the epoch metrics — previously untested.
    from detail.cli import _stream_training_progress
    from detail.core.dataclasses import TrainingProgressEvent

    events = [
        TrainingProgressEvent(0, 10, {}, 0.0, None, message="loading base model"),
        TrainingProgressEvent(1, 10, {"box_map50": 0.5}, 1.0, None),
    ]
    out = _capture(_stream_training_progress, _FakeRun(events), "train")
    assert "loading base model" in out
    assert "box_map50" in out


def test_progress_postfix_hides_single_part_shows_multi_part_and_tracks():
    single_part = DetectionProgressEvent(0, 1, 10, 100, 0.0, None)
    assert _progress_postfix(vars(single_part)) == ""
    tracking = TrackingProgressEvent(1, 3, 10, 100, 4, 0.0, None)
    postfix = _progress_postfix(vars(tracking))
    assert "part 1/3" in postfix and "active 4" in postfix
    embedding = EmbeddingProgressEvent(
        2, 1, 2, 50, 50, 0.0, None, units_done=1, units_total=1
    )
    assert "track 2" in _progress_postfix(vars(embedding))
    # #18: a §16 analysis run (recordings, no parts) surfaces its recording position.
    assert "recording 2/3" in _progress_postfix(
        {"recordings_done": 1, "recordings_total": 3}
    )


# --- detection prompt-test -------------------------------------------------- #
def _make_detection_recording(tmp_path, n_frames=5):
    """Experiment + recording with a tiny video and a hand-written det_01 run.

    Mirrors tests/api/test_run_preview.py's setup (frames 0=2 dets, 2=1, rest
    none) but lives here since the test modules aren't importable packages.
    """
    proj_dir = experiment_store.create_experiment(tmp_path, "p1")
    recording_store.create_recording(proj_dir, "recording_001")
    sdir = proj_dir / "recordings" / "recording_001"
    _write_video(sdir / "video" / "part_001.mp4", n_frames=n_frames)
    cap = cv2.VideoCapture(str(sdir / "video" / "part_001.mp4"))
    m = 0
    while cap.read()[0]:
        m += 1
    cap.release()
    recording_store.write_timestamps_csv(
        sdir / "video" / "timestamps" / "generated" / "part_001.csv",
        np.arange(m, dtype=np.int64) * int(1e9 / 30),
    )
    data = recording_store.read_recording(sdir)
    data["video"] = {
        "storage": "default",
        "fps_container": 30.0,
        "fps_actual": 30.0,
        "fps_source": "container",
        "resolution": [W, H],
        "total_duration_s": m / 30,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    }
    ddir = sdir / "detections" / "det_01"
    ddir.mkdir(parents=True)
    with DetectionPartWriter(ddir / "part_001.zarr") as w:
        w.add_row(0, 0, 0, "mouse", 0.9, (0.1, 0.1, 0.2, 0.2), "")
        w.add_row(0, 0, 1, "mouse", 0.8, (0.3, 0.3, 0.4, 0.4), "")
        w.add_row(2, int(2e9 / 30), 0, "mouse", 0.7, (0.5, 0.5, 0.6, 0.6), "")
    data.setdefault("detection_runs", []).append(
        {"id": "det_01", "status": "complete", "every_n_frames": 1}
    )
    recording_store.write_recording(sdir, data)
    return proj_dir


def test_detection_prompt_test_cli_summary_and_save(tmp_path):
    proj_dir = _make_detection_recording(tmp_path)
    with _registered():  # fake_check: cpu, readiness ok, predict -> 0 detections
        result = _runner().invoke(
            main,
            [
                "detection",
                "prompt-test",
                "--experiment",
                str(proj_dir),
                "--recording",
                "recording_001",
                "--model",
                "fake_check",
                "--backend",
                "cpu",
                "--class",
                "mouse",
                "--prompt",
                "mouse with long tail",
                "--prompt",
                "a mouse",
                "--n",
                "3",
                "--save-as",
                "try01",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "prompt_0" in result.output and "prompt_1" in result.output
    save_dir = (
        proj_dir
        / "recordings"
        / "recording_001"
        / "detections"
        / "prompt_tests"
        / "try01"
    )
    assert (save_dir / "test.json").is_file()


def test_detection_prompt_test_cli_rejects_zero_frames(tmp_path):
    proj_dir = _make_detection_recording(tmp_path)
    with _registered():
        result = _runner().invoke(
            main,
            [
                "detection",
                "prompt-test",
                "--experiment",
                str(proj_dir),
                "--recording",
                "recording_001",
                "--model",
                "fake_check",
                "--backend",
                "cpu",
                "--class",
                "mouse",
                "--prompt",
                "a mouse",
                "--n",
                "0",
            ],
        )
    assert result.exit_code != 0
    assert "must be > 0" in result.output


def test_detection_prompt_test_cli_closed_set(tmp_path, monkeypatch):
    from detail.core.registries import TRAINED_LOADERS

    monkeypatch.setitem(TRAINED_LOADERS, "fake_closed", _FakeClosedSetModel)
    proj_dir = _make_detection_recording(tmp_path)
    _write_trained_model(proj_dir)
    result = _runner().invoke(
        main,
        [
            "detection",
            "prompt-test",
            "--experiment",
            str(proj_dir),
            "--recording",
            "recording_001",
            "--model",
            "model_001",  # a trained model id — no --class/--prompt
            "--backend",
            "cpu",
            "--show-class",
            "mouse",
            "--n",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "prompt_0" in result.output


def test_detection_prompt_test_cli_requires_both_prompt_flags(tmp_path):
    proj_dir = _make_detection_recording(tmp_path)
    result = _runner().invoke(
        main,
        [
            "detection",
            "prompt-test",
            "--experiment",
            str(proj_dir),
            "--recording",
            "recording_001",
            "--model",
            "model_001",
            "--backend",
            "cpu",
            "--class",
            "mouse",  # no matching --prompt
        ],
    )
    assert result.exit_code != 0
    assert "both --class and --prompt" in result.output


def test_detection_prompt_test_cli_rejects_mixed_flags(tmp_path):
    proj_dir = _make_detection_recording(tmp_path)
    result = _runner().invoke(
        main,
        [
            "detection",
            "prompt-test",
            "--experiment",
            str(proj_dir),
            "--recording",
            "recording_001",
            "--model",
            "model_001",
            "--backend",
            "cpu",
            "--class",
            "mouse",
            "--prompt",
            "a mouse",
            "--show-class",
            "mouse",  # text prompts + --show-class → error
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


# --- Ctrl-C: cancel our own worker, let go of somebody else's --------------- #
class _InterruptingRun(RunHandle):
    """A handle whose progress stream raises KeyboardInterrupt — a terminal Ctrl-C while the
    blocking CLI waits.

    A real `RunHandle` subclass, not a duck type. The duck typed version is how this test could
    not see the bug it was written for: `owns_worker` is the fact the CLI reads, and a stand-in
    that simply does not have the attribute answers no question about what happens when it is
    False. A subclass inherits the default and has to *override* to be the other thing, which is
    the same obligation the real handles have.
    """

    def __init__(self, owns_worker=True):
        self.id = "run_01"
        self.cancelled = False
        self._owns = owns_worker

    @property
    def owns_worker(self) -> bool:
        return self._owns

    def progress(self):
        raise KeyboardInterrupt
        yield  # makes this a generator (matches a real progress() iterator)

    def cancel(self):
        self.cancelled = True

    def wait(self):
        pass


#: Every driver that turns a handle into a progress bar. Parametrised rather than tested one at
#: a time, because being *two of three* is what the bug was: `cancel_on_interrupt` existed on the
#: processing and distill drivers and not on `_stream_progress`, which is the one all five
#: single-job `attach` commands use.
_DRIVERS = [
    _stream_progress,
    _stream_processing_progress,
    _stream_distill_progress,
]


@pytest.mark.parametrize("driver", _DRIVERS, ids=lambda d: d.__name__)
def test_ctrl_c_cancels_a_worker_this_process_started(driver):
    """Otherwise it is orphaned: the worker is in its own process group, so the terminal's
    interrupt never reached it and nothing else is going to stop it."""
    run = _InterruptingRun(owns_worker=True)
    with pytest.raises(click.Abort):
        driver(run, "detection")
    assert run.cancelled


@pytest.mark.parametrize("driver", _DRIVERS, ids=lambda d: d.__name__)
def test_ctrl_c_lets_go_of_a_worker_this_process_did_not_start(driver):
    """The promise five `attach` commands have printed in their own ``--help`` since they were
    written — *"Ctrl-C detaches; it does not cancel the run"* — and that `_stream_progress` had
    no way to keep, because the parameter carrying it was on the other two drivers.

    Detaching is a normal exit, so it returns rather than aborting: a user who reattached to
    watch a four-hour training run and pressed Ctrl-C got a cancelled run *and* a non-zero exit
    status saying they had asked for it.
    """
    run = _InterruptingRun(owns_worker=False)
    driver(run, "detection")  # returns; Abort would be the CLI reporting a failure
    assert not run.cancelled, "detaching cancelled the run it promised to leave alone"


def test_a_handle_that_says_nothing_is_treated_as_ours():
    """The default is the safe one, and it is asserted rather than assumed.

    A new *attached* handle that forgets to override cancels a run the user was told it would
    not — loud, immediate, and reported. A new *launch* handle that forgot, under the opposite
    default, would orphan a worker silently, which is the failure nobody files.
    """

    class _Bare(RunHandle):
        id = "run_01"
        status = "running"

        def progress(self):
            raise KeyboardInterrupt
            yield

        def wait(self):
            pass

        def cancel(self):
            pass

    assert _Bare().owns_worker is True


# --- process / dataset / train (training workflow, §12 / §15) -------------- #
def _setup_complete_detection_run(tmp_path, n_frames=6):
    """A experiment + recording + complete detection run (det_01) + a experiment run record
    (run_001) indexing it — mirrors tests/pipeline/test_dataset_export._setup, so a
    ``dataset create --source-run run_001`` has real masks to export."""
    import json

    from pycocotools import mask as mask_utils

    from detail.storage import runs as runs_store

    proj = experiment_store.create_experiment(tmp_path, "p")
    recording_store.create_recording(proj, "recording_001")
    recording_dir = experiment_store.recording_path(proj, "recording_001")
    _write_video(recording_dir / "video" / "part_001.mp4", n_frames)

    data = recording_store.read_recording(recording_dir)
    data["video"] = {"files": [{"index": 0, "file": "video/part_001.mp4"}]}
    data["detection_runs"] = [
        {"id": "det_01", "status": "complete", "every_n_frames": 1}
    ]
    recording_store.write_recording(recording_dir, data)

    det_dir = recording_dir / "detections" / "det_01"
    det_dir.mkdir(parents=True)
    with DetectionPartWriter(det_dir / "part_001.zarr") as writer:
        for f in range(n_frames):
            m = np.zeros((H, W), np.uint8)
            top = f % (H - 6)
            m[top : top + 6, 4:14] = 1
            counts = mask_utils.encode(np.asfortranarray(m))["counts"].decode("ascii")
            writer.add_row(f, f * 1000, 0, "mouse", 0.9, (0.1, 0.1, 0.5, 0.5), counts)
    (det_dir / "job.json").write_text(
        json.dumps(
            {"schema_version": 1, "status": "complete", "mask_resolution": [H, W]}
        ),
        encoding="utf-8",
    )

    runs_store.write_run(
        proj,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "config": {},
            "status": "complete",
            "per_recording": {
                "recording_001": {"run_id": "det_01", "status": "complete"}
            },
            "created_at": "2026-06-09T00:00:00Z",
        },
    )
    return proj


def test_process_list_and_status(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    pdir = str(proj)
    r = _runner()
    res = r.invoke(main, ["process", "list", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "run_001" in res.output and "detection" in res.output
    res = r.invoke(main, ["process", "status", "run_001", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "recording_001" in res.output and "det_01" in res.output


def test_process_status_missing(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["process", "status", "run_999", "--experiment", str(proj)]
    )
    assert res.exit_code != 0
    assert "run_999" in res.output


def test_process_attach_terminal_run(tmp_path):
    # attach to a finished run — no live coordinator, just prints the final status.
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["process", "attach", "run_001", "--experiment", str(proj)]
    )
    assert res.exit_code == 0, res.output
    assert "status: complete" in res.output


def test_process_attach_missing(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["process", "attach", "nope", "--experiment", str(proj)]
    )
    assert res.exit_code != 0
    assert "no processing run" in res.output


def test_process_retry_refuses_a_fully_complete_run(tmp_path):
    # The fixture's run has nothing incomplete, so retry surfaces the API's reason
    # rather than spawning a coordinator that would have nothing to drive.
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["process", "retry", "run_001", "--experiment", str(proj)]
    )
    assert res.exit_code != 0
    assert "nothing to retry" in res.output


def test_process_retry_missing(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["process", "retry", "nope", "--experiment", str(proj)]
    )
    assert res.exit_code != 0
    assert "no processing run" in res.output


def test_process_cancel_calls_the_api(tmp_path, monkeypatch):
    proj = _setup_complete_detection_run(tmp_path)
    cancelled = []
    monkeypatch.setattr(
        "detail.api.experiment.Experiment.cancel_run",
        lambda self, rid: cancelled.append(rid),
    )
    res = _runner().invoke(
        main, ["process", "cancel", "run_001", "--experiment", str(proj)]
    )
    assert res.exit_code == 0, res.output
    assert cancelled == ["run_001"]
    assert "cancelled processing run: run_001" in res.output


def test_process_cancel_missing(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["process", "cancel", "nope", "--experiment", str(proj)]
    )
    assert res.exit_code != 0
    assert "no processing run" in res.output


def test_process_detection_unknown_model(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    r.invoke(
        main, ["recording", "create", "--experiment", pdir, "--name", "recording_001"]
    )
    video = tmp_path / "in.mp4"
    _write_video(video)
    r.invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--video",
            str(video),
        ],
    )
    res = r.invoke(
        main,
        [
            "process",
            "detection",
            "--experiment",
            pdir,
            "--recording",
            "recording_001",
            "--model",
            "no_such_model",
            "--class",
            "mouse",
            "--prompt",
            "mouse",
        ],
    )
    assert res.exit_code != 0
    assert "unknown model" in res.output


def test_dataset_create_list_status_delete(tmp_path):
    proj = _setup_complete_detection_run(tmp_path, n_frames=10)
    pdir = str(proj)
    r = _runner()
    res = r.invoke(
        main,
        [
            "dataset",
            "create",
            "--experiment",
            pdir,
            "--name",
            "mouse_set",
            "--source-run",
            "run_001",
            "--val-split",
            "0.2",
            "--min-contour-area",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "dataset_001" in res.output

    res = r.invoke(main, ["dataset", "list", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "dataset_001" in res.output and "mouse_set" in res.output

    res = r.invoke(main, ["dataset", "status", "dataset_001", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "mouse_set" in res.output

    res = r.invoke(
        main, ["dataset", "delete", "dataset_001", "--experiment", pdir, "--yes"]
    )
    assert res.exit_code == 0, res.output
    res = r.invoke(main, ["dataset", "list", "--experiment", pdir])
    assert "(no datasets)" in res.output


def test_dataset_create_missing_run(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main,
        [
            "dataset",
            "create",
            "--experiment",
            str(proj),
            "--name",
            "d",
            "--source-run",
            "run_999",
        ],
    )
    assert res.exit_code != 0


def test_dataset_status_missing(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main, ["dataset", "status", "nope", "--experiment", str(proj)]
    )
    assert res.exit_code != 0


def test_dataset_label_rejects_unknown_sampling(tmp_path):
    # --sampling is a click.Choice derived from LabelingConfig's Literal — a bad value is a parse
    # error, before the command touches SAM3.
    proj = experiment_store.create_experiment(tmp_path, "p")
    res = _runner().invoke(
        main,
        [
            "dataset",
            "label",
            "--experiment",
            str(proj),
            "--name",
            "d",
            "--class",
            "mouse:a mouse",
            "--n-frames",
            "4",
            "--sampling",
            "middle-out",
        ],
    )
    assert res.exit_code != 0
    assert "sampling" in res.output.lower()


def _capture_label_dataset(monkeypatch):
    """Record ``Experiment.label_dataset`` kwargs then abort before submit."""
    from detail.api.experiment import Experiment

    captured: dict = {}

    def _fake(self, *args, **kwargs):
        captured.update(kwargs)
        raise ValueError("stop-before-submit")

    monkeypatch.setattr(Experiment, "label_dataset", _fake)
    return captured


#: Where ``distill run`` / ``Experiment.distill`` name a LabelingConfig field differently: they
#: carry a training model too, so the teacher's two are prefixed.
_DISTILL_LABEL_NAMES = {
    "model_id": "label_model_id",
    "model_variant": "label_model_variant",
}


@pytest.mark.parametrize("field", sorted(LabelingConfig.model_fields))
def test_every_labeling_field_reaches_every_labeling_surface(field):
    """LabelingConfig's fields are placed by hand on four surfaces — two CLI commands and two
    API methods — so a new field reaches whichever ones its author remembered. Driven from the
    model, so the next field is checked the day it is added (`background_share` is the one this
    was written for)."""
    import inspect

    from detail.api.experiment import Experiment

    distill_name = _DISTILL_LABEL_NAMES.get(field, field)
    label_cmd = main.commands["dataset"].commands["label"]
    distill_cmd = main.commands["distill"].commands["run"]
    assert field in {p.name for p in label_cmd.params}, "dataset label"
    assert distill_name in {p.name for p in distill_cmd.params}, "distill run"
    assert field in inspect.signature(Experiment.label_dataset).parameters
    assert distill_name in inspect.signature(Experiment.distill).parameters


def test_dataset_label_background_share_reaches_label_dataset(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    captured = _capture_label_dataset(monkeypatch)
    args = ["dataset", "label", "--experiment", str(proj), "--name", "d"]
    args += ["--class", "mouse:a mouse", "--n-frames", "4"]
    _runner().invoke(main, args)
    assert (
        captured["background_share"]
        == LabelingConfig.model_fields["background_share"].default
    )
    _runner().invoke(main, args + ["--background-share", "0.25"])
    assert captured["background_share"] == 0.25
    res = _runner().invoke(main, args + ["--background-share", "0.9"])
    assert res.exit_code != 0 and "0.9" in res.output  # above the model's 0.5 bound


def test_dataset_label_model_flags_reach_label_dataset(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    captured = _capture_label_dataset(monkeypatch)
    res = _runner().invoke(
        main,
        [
            "dataset",
            "label",
            "--experiment",
            str(proj),
            "--name",
            "d",
            "--class",
            "mouse:a mouse",
            "--n-frames",
            "4",
            "--model",
            "fake_yolo",
            "--model-variant",
            "11l",
        ],
    )
    assert res.exit_code != 0  # our stop-before-submit
    assert captured["model_id"] == "fake_yolo"
    assert captured["model_variant"] == "11l"


def test_dataset_label_defaults_model_to_sam3(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    captured = _capture_label_dataset(monkeypatch)
    res = _runner().invoke(
        main,
        [
            "dataset",
            "label",
            "--experiment",
            str(proj),
            "--name",
            "d",
            "--class",
            "mouse:a mouse",
            "--n-frames",
            "4",
        ],
    )
    assert res.exit_code != 0
    assert captured["model_id"] == "sam3" and captured["model_variant"] is None


def test_dataset_label_rejects_non_prompt_capable_model(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    res = _runner().invoke(
        main,
        [
            "dataset",
            "label",
            "--experiment",
            str(proj),
            "--name",
            "d",
            "--class",
            "mouse:a mouse",
            "--n-frames",
            "4",
            "--model",
            "trained_yolo",
        ],
    )
    assert res.exit_code != 0
    assert "cannot label" in res.output


def test_train_run_unknown_dataset(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main,
        [
            "train",
            "run",
            "--experiment",
            str(proj),
            "--name",
            "m",
            "--dataset",
            "ds_999",
        ],
    )
    assert res.exit_code != 0
    assert "ds_999" in res.output


def test_train_run_rejects_unknown_hyperparam(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(
        main,
        [
            "train",
            "run",
            "--experiment",
            str(proj),
            "--name",
            "m",
            "--dataset",
            "ds_999",
            "--hyperparam",
            "bogus=1",
        ],
    )
    assert res.exit_code != 0
    assert "unknown hyperparam" in res.output


def _capture_train_model(monkeypatch):
    """Monkeypatch ``Experiment.train_model`` to record its kwargs then abort before
    submit (raising a ``ValueError`` the CLI turns into a clean exit), so a
    training-flag test needs no real dataset or run streaming."""
    from detail.api.experiment import Experiment

    captured: dict = {}

    def _fake(self, **kwargs):
        captured.update(kwargs)
        raise ValueError("stop-before-submit")

    monkeypatch.setattr(Experiment, "train_model", _fake)
    return captured


def test_train_run_patience_flag_reaches_train_model(tmp_path, monkeypatch):
    proj = _setup_complete_detection_run(tmp_path)
    captured = _capture_train_model(monkeypatch)
    res = _runner().invoke(
        main,
        [
            "train",
            "run",
            "--experiment",
            str(proj),
            "--name",
            "m",
            "--dataset",
            "dataset_001",
            "--patience",
            "7",
        ],
    )
    assert res.exit_code != 0  # our stop-before-submit
    assert captured["hyperparams"] == {"patience": 7}


def test_train_run_hyperparam_patience_coerces_via_model(tmp_path, monkeypatch):
    proj = _setup_complete_detection_run(tmp_path)
    captured = _capture_train_model(monkeypatch)
    res = _runner().invoke(
        main,
        [
            "train",
            "run",
            "--experiment",
            str(proj),
            "--name",
            "m",
            "--dataset",
            "dataset_001",
            "--hyperparam",
            "patience=9",
        ],
    )
    assert res.exit_code != 0
    assert captured["hyperparams"] == {"patience": 9}  # coerced to int via config_model


def test_train_list_empty(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(main, ["train", "list", "--experiment", str(proj)])
    assert res.exit_code == 0, res.output
    assert "(no trained models)" in res.output


def test_train_status_missing(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(main, ["train", "status", "nope", "--experiment", str(proj)])
    assert res.exit_code != 0


def test_train_resume_unknown_model(tmp_path):
    proj = _setup_complete_detection_run(tmp_path)
    res = _runner().invoke(main, ["train", "resume", "nope", "--experiment", str(proj)])
    assert res.exit_code != 0
    assert "No such command" not in res.output  # the command exists; the model doesn't


def test_embedding_batch_size_cli_bound_matches_model():
    # the CLI range is sourced from EmbeddingConfig (ge=1, le=1024), not hand-written (A17)
    from detail.cli import _EMB_DOC, _int_range

    r = _int_range(_EMB_DOC["batch_size"])
    assert r.min == 1 and r.max == 1024


# --- imported models (model import / list / remove) ------------------------ #
class _StubImportLoader(DetectionModel):
    model_id = "stub"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def __init__(self, weights_path, classes):
        super().__init__(None)
        self.weights_path = weights_path
        self.classes = classes

    @staticmethod
    def introspect(weights_path):
        return ["mouse"], "segment"

    def load(self, device): ...

    def predict(self, frame, prompt):
        raise NotImplementedError

    def predict_batch(self, frames, prompts):
        raise NotImplementedError


def test_cli_model_import_list_remove(tmp_path, monkeypatch):
    from detail.core.registries import TRAINED_LOADERS

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubImportLoader)
    w = tmp_path / "best.pt"
    w.write_bytes(b"w")
    r = _runner()
    res = r.invoke(
        main,
        ["model", "import", str(w), "--family", "stubfam", "--name", "mouse_yolo11s"],
    )
    assert res.exit_code == 0, res.output
    assert "mouse_yolo11s" in res.output
    res = r.invoke(main, ["model", "list"])
    assert "mouse_yolo11s" in res.output and "mouse" in res.output
    assert "source=imported" in res.output  # listed alongside the built-ins
    res = r.invoke(main, ["model", "remove", "mouse_yolo11s"])
    assert res.exit_code == 0, res.output
    # Gone from the catalog. (The built-ins always remain, so the list is never empty.)
    assert "mouse_yolo11s" not in r.invoke(main, ["model", "list"]).output


def test_cli_model_import_rejects_reserved_id(tmp_path, monkeypatch):
    from detail.core.registries import TRAINED_LOADERS

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubImportLoader)
    w = tmp_path / "best.pt"
    w.write_bytes(b"w")
    res = _runner().invoke(
        main, ["model", "import", str(w), "--family", "stubfam", "--name", "model_002"]
    )
    assert res.exit_code != 0
    assert "reserved" in res.output


def _import_stub_model(r, tmp_path, monkeypatch, name="m", extra=()):
    from detail.core.registries import TRAINED_LOADERS

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubImportLoader)
    w = tmp_path / "best.pt"
    w.write_bytes(b"w")
    res = r.invoke(
        main,
        ["model", "import", str(w), "--family", "stubfam", "--name", name, *extra],
    )
    assert res.exit_code == 0, res.output
    return res


def test_cli_model_import_with_metadata_and_show(tmp_path, monkeypatch):
    img = tmp_path / "shot.png"
    img.write_bytes(b"png")
    r = _runner()
    _import_stub_model(
        r,
        tmp_path,
        monkeypatch,
        name="operant_v2",
        extra=[
            "--made-by",
            "labmate",
            "--description",
            "operant chamber",
            "--notes",
            "room 1",
            "--preview",
            str(img),
        ],
    )
    out = r.invoke(main, ["model", "show", "operant_v2"]).output
    assert "labmate" in out and "operant chamber" in out and "room 1" in out
    assert "preview:" in out and "preview.png" in out


def test_cli_model_set_meta_edits(tmp_path, monkeypatch):
    r = _runner()
    _import_stub_model(r, tmp_path, monkeypatch)
    res = r.invoke(main, ["model", "set-meta", "m", "--description", "new desc"])
    assert res.exit_code == 0, res.output
    assert "new desc" in r.invoke(main, ["model", "show", "m"]).output


def test_cli_model_export_restore_round_trip(tmp_path, monkeypatch):
    r = _runner()
    _import_stub_model(r, tmp_path, monkeypatch, extra=["--made-by", "kev"])
    dest = tmp_path / "m.zip"
    res = r.invoke(main, ["model", "export", "m", str(dest)])
    assert res.exit_code == 0, res.output
    assert dest.is_file()
    r.invoke(main, ["model", "remove", "m"])
    res = r.invoke(main, ["model", "restore", str(dest)])
    assert res.exit_code == 0 and "restored model: m" in res.output
    assert "kev" in r.invoke(main, ["model", "show", "m"]).output


# --- a pose checkpoint (§12 *Keypoint models in the catalog*) ------------------- #
@pytest.fixture
def pose_pt(tmp_path, monkeypatch):
    """A ``.pt`` whose task reads ``pose``, placing 3 points with no left/right swap. Only
    the two Ultralytics readers are stubbed; the catalog and library are real (isolated).
    """
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.core.registries import TRAINED_LOADERS
    from detail.executor.runners import import_runner_modules
    from detail.models import imported as imported_store
    from detail.models import yolo_pose
    from tests.models.test_imported import _PoseStubLoader

    import_runner_modules(
        PIPELINE_RUNNERS
    )  # register first, so the stub is not replaced
    monkeypatch.setitem(TRAINED_LOADERS, imported_store._PROBE_FAMILY, _PoseStubLoader)
    monkeypatch.setattr(
        yolo_pose.TrainedYoloPose,
        "introspect",
        staticmethod(lambda weights_path: (3, [0, 1, 2])),
    )
    w = tmp_path / "rat-pose.pt"
    w.write_bytes(b"w")
    return w


def test_cli_model_import_family_defaults_to_none():
    """A default family of "yolo" refused every pose checkpoint imported without --family:
    the task decides, so the option must default to nothing."""
    command = main.commands["model"].commands["import"]
    family = next(p for p in command.params if p.name == "family")
    assert family.default is None


def test_cli_model_import_a_pose_checkpoint_with_a_skeleton(pose_pt):
    r = _runner()
    res = r.invoke(main, ["model", "import", str(pose_pt), "--skeleton", "midline_3"])
    assert res.exit_code == 0, res.output
    assert "imported model: rat-pose" in res.output
    assert (
        "skeleton: Midline (3 points) (midline_3)" in res.output
    )  # matched the preset
    show = r.invoke(main, ["model", "show", "rat-pose"]).output
    assert "role:        keypoint" in show
    assert "skeleton:    Midline (3 points) (midline_3)" in show
    assert "points:      head, midpoint, tail" in show
    listed = r.invoke(main, ["model", "list"]).output
    line = next(ln for ln in listed.splitlines() if "rat-pose" in ln)
    assert "role=keypoint" in line and "skeleton=3 points" in line


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "pass the skeleton its points follow"),
        (["--skeleton", "quadruped_side_5"], "places 3 points but the skeleton has 5"),
        (["--skeleton", "no_such_skeleton"], "no skeleton preset"),
    ],
    ids=["no skeleton", "wrong point count", "unknown skeleton"],
)
def test_cli_model_import_refuses_a_pose_checkpoint_without_a_fitting_skeleton(
    pose_pt, args, message
):
    res = _runner().invoke(main, ["model", "import", str(pose_pt), *args])
    assert res.exit_code != 0
    assert message in res.output


def test_cli_model_import_refuses_a_skeleton_for_a_detection_checkpoint(
    tmp_path, monkeypatch
):
    from detail.core.registries import TRAINED_LOADERS

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubImportLoader)
    w = tmp_path / "best.pt"
    w.write_bytes(b"w")
    res = _runner().invoke(
        main,
        ["model", "import", str(w), "--family", "stubfam", "--skeleton", "midline_3"],
    )
    assert res.exit_code != 0
    assert "a skeleton applies only to a pose (keypoint) checkpoint" in res.output


def test_cli_model_show_a_detector_has_no_skeleton(tmp_path, monkeypatch):
    r = _runner()
    _import_stub_model(r, tmp_path, monkeypatch)
    out = r.invoke(main, ["model", "show", "m"]).output
    assert "role:        detection" in out and "skeleton:" not in out


def test_cli_model_restore_names_the_skeleton_library_entry(tmp_path, pose_pt):
    from detail.storage import skeleton_library

    import yaml

    from detail.core.skeleton import Skeleton

    r = _runner()
    yaml_path = tmp_path / "line.yaml"
    line = Skeleton(names=("head", "body", "tail"), edges=(("head", "body"),))
    yaml_path.write_text(
        yaml.safe_dump({"name": "rat line", "skeleton": line.to_dict()}),
        encoding="utf-8",
    )
    res = r.invoke(
        main, ["model", "import", str(pose_pt), "--skeleton", str(yaml_path)]
    )
    assert res.exit_code == 0, res.output
    assert (
        r.invoke(
            main, ["model", "export", "rat-pose", str(tmp_path / "m.zip")]
        ).exit_code
        == 0
    )
    r.invoke(main, ["model", "remove", "rat-pose"])
    for skeleton_id in skeleton_library.discover():  # a machine that has never seen it
        skeleton_library.remove(skeleton_id)

    res = r.invoke(main, ["model", "restore", str(tmp_path / "m.zip")])
    assert res.exit_code == 0, res.output
    assert "restored model: rat-pose" in res.output
    entry = skeleton_library.list_records()[0]
    assert f"skeleton: rat line ({entry['id']}) in the library" in res.output


def test_cli_model_export_drives_progress(tmp_path, monkeypatch):
    # The synchronous long-op rule: the command must wire a non-None progress callback.
    from detail.api import models as api_models

    captured = {}

    def fake_export(model_id, dest, progress=None):
        captured["progress"] = progress
        Path(dest).write_bytes(b"zip")
        return Path(dest)

    monkeypatch.setattr(api_models, "export_model", fake_export)
    res = _runner().invoke(main, ["model", "export", "any", str(tmp_path / "o.zip")])
    assert res.exit_code == 0, res.output
    assert captured["progress"] is not None


def test_process_tracking_detection_selector_exclusivity(tmp_path):
    """Exactly one of --detection / --detection-ref (§6.5 CLI parity)."""
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    base = ["process", "tracking", "--experiment", pdir, "--recording", "recording_001"]
    both = r.invoke(
        main,
        base + ["--detection", "det_01", "--detection-ref", "recording_001:det_01"],
    )
    assert both.exit_code != 0 and "exactly one" in both.output
    neither = r.invoke(main, base)
    assert neither.exit_code != 0 and "exactly one" in neither.output


def test_process_embedding_tracking_selector_exclusivity(tmp_path):
    """`process embedding` (batch embedding, §6.5 CLI parity) takes exactly one of
    --tracking / --tracking-ref, and --tracks / --all-tracks are mutually exclusive."""
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    base = [
        "process", "embedding", "--experiment", pdir, "--recording", "recording_001",
        "--model", "dinov3_convnext_base", "--all-tracks",
    ]  # fmt: skip
    both = r.invoke(
        main,
        base + ["--tracking", "track_01", "--tracking-ref", "recording_001:track_01"],
    )
    assert both.exit_code != 0 and "exactly one" in both.output
    neither = r.invoke(main, base)
    assert neither.exit_code != 0 and "exactly one" in neither.output
    both_tracks = r.invoke(
        main,
        [
            "process", "embedding", "--experiment", pdir, "--recording", "recording_001",
            "--model", "dinov3_convnext_base", "--tracking", "track_01",
            "--all-tracks", "--tracks", "0",
        ],  # fmt: skip
    )
    assert both_tracks.exit_code != 0 and "not both" in both_tracks.output


def test_embedding_run_resize_defaults_to_model_input_not_224():
    """`--resize` defaults to unset (None) so the renderer defers to the embedding
    model's own input size, not a hard-coded 224 — parity with the GUI/API (§14)."""
    from detail.cli import embedding_run, process_embedding

    for cmd in (embedding_run, process_embedding):
        opt = next(p for p in cmd.params if p.name == "renderer_size")
        assert opt.default is None


def test_analysis_run_tracking_selector_exclusivity(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    base = ["analysis", "run", "--experiment", pdir, "--recording", "recording_001"]
    both = r.invoke(
        main,
        base + ["--tracking", "track_01", "--tracking-ref", "recording_001:track_01"],
    )
    assert both.exit_code != 0 and "exactly one" in both.output
    neither = r.invoke(main, base)
    assert neither.exit_code != 0 and "exactly one" in neither.output


def test_a_repeated_ref_names_several_settings(monkeypatch):
    """`--embed-ref` / `--tracking-ref` repeated is how a CLI user pools recordings made
    through different chains (§6.5): one ref keeps the single-spec shape every earlier run
    recorded, several become `<upstream>_specs`, and naming an id as well is refused."""
    import click

    from detail import cli

    monkeypatch.setattr(
        cli,
        "_spec_from_ref",
        lambda _proj, module_id, ref: {"module": module_id, "r": ref},
    )
    assert cli._embedding_selection(None, None, ("s1:emb_01",)) == {
        "embedding_spec": {"module": "embedding", "r": "s1:emb_01"}
    }
    assert cli._embedding_selection(None, None, ("s1:emb_01", "s9:emb_02")) == {
        "embedding_specs": [
            {"module": "embedding", "r": "s1:emb_01"},
            {"module": "embedding", "r": "s9:emb_02"},
        ]
    }
    assert cli._embedding_selection(None, "emb_01", ()) == {"embed_run_id": "emb_01"}
    assert cli._embedding_selection(None, None, ()) == {}
    with pytest.raises(click.ClickException, match="at most one"):
        cli._embedding_selection(None, "emb_01", ("s1:emb_01",))

    for command, name in (
        (cli.analysis_temporal_clustering, "embed_refs"),
        (cli.analysis_run, "tracking_refs"),
    ):
        option = next(p for p in command.params if p.name == name)
        assert option.multiple, f"{command.name} must accept its ref more than once"


def _det_rec(rid, **over):
    return {
        "id": rid, "status": "complete", "output": f"detections/{rid}/", "created_at": "t",
        "model_id": "sam3", "model_variant": "large", "weights_source": "pretrained",
        "conf_threshold": 0.25, "prompt_fingerprint": "fp", "prompt_classes": ["mouse"],
        **over,
    }  # fmt: skip


def test_spec_from_ref_derives_and_resolves_per_recording(tmp_path):
    """`--detection-ref RECORDING:det_NN` derives the detection kind and resolves each
    recording's OWN run — even at divergent ordinals (the CLI half of §6.5)."""
    from detail.api import run_spec
    from detail.api.experiment import Experiment
    from detail.cli import _spec_from_ref

    proj = Experiment.create("p", tmp_path)
    for sid, dets in [
        (
            "s1",
            [_det_rec("det_01", conf_threshold=0.9), _det_rec("det_02")],
        ),  # kind at det_02
        ("s2", [_det_rec("det_01")]),  # same kind at det_01
    ]:
        proj.create_recording(sid)
        sdir = proj.get_recording(sid).dir
        data = recording_store.read_recording(sdir)
        data["detection_runs"] = dets
        recording_store.write_recording(sdir, data)

    spec = _spec_from_ref(proj, "detection", "s2:det_01")
    assert (
        run_spec.resolve_spec_for_recording(proj.get_recording("s1"), "detection", spec)
        == "det_02"
    )
    assert (
        run_spec.resolve_spec_for_recording(proj.get_recording("s2"), "detection", spec)
        == "det_01"
    )


def test_spec_from_ref_missing_run_errors(tmp_path):
    from detail.api.experiment import Experiment
    from detail.cli import _spec_from_ref

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("s1")
    with pytest.raises(click.ClickException, match="no detection run"):
        _spec_from_ref(proj, "detection", "s1:det_99")


def test_detection_ref_bad_format_errors(tmp_path):
    r = _runner()
    r.invoke(main, ["experiment", "create", "p1", "--dir", str(tmp_path)])
    pdir = str(tmp_path / "p1")
    res = r.invoke(
        main,
        [
            "process",
            "tracking",
            "--experiment",
            pdir,
            "--recording",
            "s1",
            "--detection-ref",
            "no-colon",
        ],
    )
    assert res.exit_code != 0 and "RECORDING:run_id" in res.output


def test_analysis_list_status_cancel_delete(tmp_path):
    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "artifacts": [{"path": "recording_001/tracks.csv", "kind": "csv"}],
        },
    )
    pdir = str(proj.dir)
    r = _runner()

    res = r.invoke(main, ["analysis", "list", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "an_01" in res.output

    res = r.invoke(main, ["analysis", "status", "an_01", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "recording_001/tracks.csv" in res.output

    res = r.invoke(main, ["analysis", "cancel", "an_01", "--experiment", pdir])
    assert res.exit_code == 0, res.output  # terminal run → no-op, still succeeds

    res = r.invoke(main, ["analysis", "delete", "an_01", "--experiment", pdir, "--yes"])
    assert res.exit_code == 0, res.output
    assert (
        "(no analysis runs)"
        in r.invoke(main, ["analysis", "list", "--experiment", pdir]).output
    )


def test_analysis_status_reports_a_failure_its_own_record_missed(tmp_path):
    """The CLI reads the record, so it inherits whatever the record got wrong.

    An authored run writes `status: open` before projecting its artifact, so a job that died
    there leaves `analysis.yaml` reading `open` over a `job.json` reading `failed`. Reconciled
    on the way in, or `analysis status` reports a dead run as one somebody is working on.
    """
    import json

    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    run_dir = ar.analysis_dir(proj.dir, "an_01")
    ar.write_analysis_yaml(
        run_dir, {"id": "an_01", "module_id": "pose_clustering", "status": "open"}
    )
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "job": {"job_id": "an_01", "config": {}},
                "progress": None,
                "error": "PermissionError: [WinError 32]",
            }
        ),
        encoding="utf-8",
    )

    res = _runner().invoke(
        main, ["analysis", "status", "an_01", "--experiment", str(proj.dir)]
    )

    assert res.exit_code == 0, res.output
    assert "status: failed" in res.output
    assert "WinError 32" in res.output, "it says why, not only that"


def test_analysis_status_names_out_of_date_results_and_rebuild_puts_them_right(
    authored_cohort, monkeypatch
):
    """The stale half, through both commands rather than through the API underneath them.

    The projection itself is stubbed — `authored_cohort` is the authored run without the
    twenty-second fit, and what is under test here is the wiring: that `status` names the
    state and the command that clears it, and that `rebuild` reaches `reproject` and the
    mark comes up. The projection's own failure modes are
    `tests/api/test_authored_edit_ordering.py`.
    """
    from detail.api.temporal_clustering import TemporalClusteringRun
    from detail.storage import analysis_runs as ar

    monkeypatch.setattr(TemporalClusteringRun, "_project", lambda self, **_: None)
    cohort = authored_cohort
    pdir = str(cohort.experiment.dir)
    ar.set_artifact_stale(cohort.run_dir(), True)
    r = _runner()

    res = r.invoke(main, ["analysis", "status", "an_01", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert "out of date" in res.output
    assert "analysis rebuild an_01" in res.output, "it names the command that fixes it"

    res = r.invoke(main, ["analysis", "rebuild", "an_01", "--experiment", pdir])
    assert res.exit_code == 0, res.output
    assert not ar.artifact_stale(cohort.run_dir())

    res = r.invoke(main, ["analysis", "status", "an_01", "--experiment", pdir])
    assert "out of date" not in res.output


def test_analysis_rebuild_refuses_a_run_it_cannot_re_derive(tmp_path):
    """A computed run is re-run, not re-derived — and says so rather than half-working."""
    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {"id": "an_01", "module_id": "location_export", "status": "complete"},
    )

    res = _runner().invoke(
        main, ["analysis", "rebuild", "an_01", "--experiment", str(proj.dir)]
    )

    assert res.exit_code != 0
    assert "not an authored run" in res.output


def test_analysis_status_shows_the_settings_and_marks_the_selections(tmp_path):
    """The CLI's half of the GUI's Refine: what this run was launched with.

    Without it the only way to see a finished run's parameters is to open `analysis.yaml` by
    hand, so the two surfaces would disagree about whether "start from this run" is reachable
    at all. The `[selection]` mark comes from the same `scope` declaration the GUI reads, so
    the CLI cannot claim a different set of reusable settings than the form does.
    """
    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "temporal_clustering",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "params": {"embed_run_id": "emb_03", "track_id": 0, "method": "dtw_finch"},
        },
    )
    res = _runner().invoke(
        main, ["analysis", "status", "an_01", "--experiment", str(proj.dir)]
    )
    assert res.exit_code == 0, res.output
    assert "method: dtw_finch" in res.output
    assert "embed_run_id: emb_03 [selection]" in res.output
    # A setting is not marked — if everything were marked the mark would say nothing.
    assert "track_id: 0\n" in res.output


def test_analysis_status_does_not_print_a_structure_as_one_long_line(tmp_path):
    """A nested setting is a block or a count, never a dumped repr.

    A real behavior scan carries every example clip's recording, track and span under
    `behaviors`, which printed as a value is a single 2,000-character line — output nobody
    reads, and the kind of thing that passes a test asserting only that the key appears.
    """
    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_02"),
        {
            "id": "an_02",
            "module_id": "query",
            "recording_ids": ["recording_001"],
            "status": "open",
            "params": {
                "method_params": {"k": 20, "tau": 0.07},
                "behaviors": [
                    {"name": "rear", "examples": [{"x": 1}]},
                    {"name": "spin"},
                ],
                "trajectory_split_gap_ns": None,
            },
        },
    )
    res = _runner().invoke(
        main, ["analysis", "status", "an_02", "--experiment", str(proj.dir)]
    )
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert max(len(line) for line in lines) < 100, max(lines, key=len)

    # A dict becomes an indented block, so each knob is on its own line and readable.
    assert "    method_params:" in lines
    assert "      k: 20" in lines
    # A list of structures becomes its length — the clips stay in the record for a reader
    # that wants them.
    assert "    behaviors: 2 item(s)" in lines
    # `None` is a real value on an optional knob ("work it out for me"), so it is shown —
    # but not as the word `None`, which reads as a missing value rather than a chosen one.
    assert "    trajectory_split_gap_ns: (not set)" in lines


def test_analysis_delete_missing(tmp_path):
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    res = _runner().invoke(
        main, ["analysis", "delete", "an_99", "--experiment", str(proj.dir), "--yes"]
    )
    assert res.exit_code != 0  # FileNotFoundError → ClickException


def test_recording_position_never_exceeds_the_total():
    """`recordings_done` counts *finished* recordings, so the in-flight one is done+1 —
    but on the terminal event done == total, which rendered "recording 2/1" on a real
    single-recording clusterer fit."""
    from detail.cli import _recording_position

    assert _recording_position(0, 1) == 1  # working on the only recording
    assert _recording_position(1, 1) == 1  # terminal event, not 2
    assert _recording_position(0, 5) == 1
    assert _recording_position(3, 5) == 4  # mid-run stays 1-based
    assert _recording_position(5, 5) == 5  # terminal event, not 6


def test_analysis_postfix_shows_a_sane_recording_counter():
    from detail.cli import _progress_postfix

    assert "recording 1/1" in _progress_postfix(
        {"recordings_done": 1, "recordings_total": 1}
    )
    assert "recording 4/5" in _progress_postfix(
        {"recordings_done": 3, "recordings_total": 5}
    )


# --- distill (§12) --------------------------------------------------------- #
def test_train_promote_calls_promote_model(tmp_path, monkeypatch):
    from detail.api.experiment import Experiment

    proj = experiment_store.create_experiment(tmp_path, "p")
    captured = {}

    def _fake(self, model_id, name=None):
        captured["model_id"] = model_id
        captured["name"] = name
        return "mouse_seg"

    monkeypatch.setattr(Experiment, "promote_model", _fake)
    res = _runner().invoke(
        main,
        [
            "train",
            "promote",
            "model_001",
            "--experiment",
            str(proj),
            "--name",
            "shared",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured == {"model_id": "model_001", "name": "shared"}
    assert "global model: mouse_seg" in res.output


def _capture_distill(monkeypatch):
    """Record ``Experiment.distill`` kwargs then abort before submit."""
    from detail.api.experiment import Experiment

    captured: dict = {}

    def _fake(self, *args, **kwargs):
        captured.update(kwargs)
        raise ValueError("stop-before-submit")

    monkeypatch.setattr(Experiment, "distill", _fake)
    return captured


def test_distill_run_wires_classes_computes_and_hyperparams(tmp_path, monkeypatch):
    proj = experiment_store.create_experiment(tmp_path, "p")
    captured = _capture_distill(monkeypatch)
    res = _runner().invoke(
        main,
        [
            "distill",
            "run",
            "--experiment",
            str(proj),
            "--name",
            "m",
            "--class",
            "mouse:a mouse",
            "--n-frames",
            "50",
            "--epochs",
            "5",
            "--label-backend",
            "cuda",
            "--train-backend",
            "cpu",
        ],
    )
    assert res.exit_code != 0  # our stop-before-submit
    assert [c.name for c in captured["classes"]] == ["mouse"]
    assert captured["classes"][0].prompt == "a mouse"
    assert captured["n_frames"] == 50
    assert captured["hyperparams"] == {"epochs": 5}
    assert captured["label_compute"].backend == "cuda"
    assert captured["train_compute"].backend == "cpu"


def test_distill_run_rejects_bad_class(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    res = _runner().invoke(
        main,
        [
            "distill",
            "run",
            "--experiment",
            str(proj),
            "--name",
            "m",
            "--class",
            "noprompt",
            "--n-frames",
            "4",
        ],
    )
    assert res.exit_code != 0
    assert "NAME:PROMPT" in res.output


def test_distill_run_requires_a_class(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    res = _runner().invoke(
        main,
        ["distill", "run", "--experiment", str(proj), "--name", "m", "--n-frames", "4"],
    )
    assert res.exit_code != 0
    assert "--class" in res.output


def test_stream_distill_progress_renders_phase_message_and_eta():
    from detail.cli import _stream_distill_progress
    from detail.core.dataclasses import DistillProgress

    events = [
        DistillProgress("labeling", 0, 2, 0.5, 0.225, 0.1, 5.0, "labeling frames"),
        DistillProgress("training", 1, 2, 0.4, 0.72, 0.5, 2.0, "epoch 2/5"),
    ]
    out = _capture(_stream_distill_progress, _FakeRun(events), "distill")
    assert "labeling frames" in out  # phase message above the bar
    assert "epoch 2/5" in out  # phase advanced
    assert "eta=" in out  # ETA in the postfix


def test_distill_list_empty(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    res = _runner().invoke(main, ["distill", "list", "--experiment", str(proj)])
    assert res.exit_code == 0
    assert "no distill runs" in res.output


def test_distill_status_missing(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    res = _runner().invoke(
        main, ["distill", "status", "distill_999", "--experiment", str(proj)]
    )
    assert res.exit_code != 0


def test_distill_cancel_calls_the_api(tmp_path, monkeypatch):
    from detail.api.experiment import Experiment

    proj = experiment_store.create_experiment(tmp_path, "p")
    called = {}
    monkeypatch.setattr(
        Experiment, "cancel_distill", lambda self, rid: called.update(id=rid)
    )
    res = _runner().invoke(
        main, ["distill", "cancel", "distill_001", "--experiment", str(proj)]
    )
    assert res.exit_code == 0, res.output
    assert called == {"id": "distill_001"}


def test_cli_doctor_reports_checks_and_fix(monkeypatch):
    from detail.api import checks as api_checks
    from detail.core.dataclasses import CheckResult, CheckStatus

    monkeypatch.setattr(
        api_checks,
        "run_checks",
        lambda **k: [
            CheckResult("sam3_backend", CheckStatus.FAIL, "not installed"),
            CheckResult("hf_token", CheckStatus.OK, "configured"),
        ],
    )
    res = _runner().invoke(main, ["doctor"])
    assert "SAM 3 backend package" in res.output and "not installed" in res.output
    assert "detail docs getting-started" in res.output  # the fix pointer
    # the full copy-paste install sequence for this machine, printed once
    assert "Install commands" in res.output
    assert "git clone https://github.com/facebookresearch/sam3 " in res.output
    assert res.exit_code == 1  # a failing check exits non-zero


def test_cli_doctor_prints_no_commands_when_all_pass(monkeypatch):
    from detail.api import checks as api_checks
    from detail.core.dataclasses import CheckResult, CheckStatus

    infos = api_checks.list_checks()
    monkeypatch.setattr(
        api_checks,
        "run_checks",
        lambda **k: [CheckResult(i.check_id, CheckStatus.OK, "") for i in infos],
    )
    res = _runner().invoke(main, ["doctor"])
    assert "Install commands" not in res.output  # nothing broken → no install sequence


def test_cli_doctor_exits_zero_without_failures(monkeypatch):
    from detail.api import checks as api_checks
    from detail.core.dataclasses import CheckResult, CheckStatus

    infos = api_checks.list_checks()
    monkeypatch.setattr(
        api_checks,
        "run_checks",
        lambda **k: [CheckResult(i.check_id, CheckStatus.OK, "") for i in infos],
    )
    assert _runner().invoke(main, ["doctor"]).exit_code == 0


# --- analysis attach: the CLI half of the analysis reattach (§16) ------------ #
def _complete_analysis_run(tmp_path, run_id="an_01"):
    from detail.api.experiment import Experiment
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, run_id)
    ar.write_analysis_yaml(
        adir,
        {
            "id": run_id,
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "created_at": "2026-08-01T10:00:00Z",
        },
    )
    (adir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return proj.dir


def test_analysis_attach_terminal_run(tmp_path):
    # attach to a finished analysis — no live worker, just prints the final status.
    proj = _complete_analysis_run(tmp_path)
    res = _runner().invoke(
        main, ["analysis", "attach", "an_01", "--experiment", str(proj)]
    )
    assert res.exit_code == 0, res.output
    assert "status: complete" in res.output


def test_analysis_attach_missing(tmp_path):
    proj = _complete_analysis_run(tmp_path)
    res = _runner().invoke(
        main, ["analysis", "attach", "nope", "--experiment", str(proj)]
    )
    assert res.exit_code != 0
    assert "no analysis run" in res.output


# --- attach parity: every single-job run kind streams from the CLI too ------- #
@pytest.mark.parametrize(
    "group,run_id,subdir,writer",
    [
        ("dataset", "dataset_001", "datasets", "write_dataset_yaml"),
        ("train", "model_001", "models", "write_model_yaml"),
    ],
)
def test_attach_terminal_run_prints_status(tmp_path, group, run_id, subdir, writer):
    from detail.api.experiment import Experiment
    from detail.storage import datasets as datasets_store
    from detail.storage import models as models_store

    stores = {
        "datasets": datasets_store,
        "models": models_store,
    }
    proj = Experiment.create("p", tmp_path)
    run_dir = proj.dir / subdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    getattr(stores[subdir], writer)(
        run_dir, {"id": run_id, "name": run_id, "status": "complete"}
    )
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    res = _runner().invoke(
        main, [group, "attach", run_id, "--experiment", str(proj.dir)]
    )
    assert res.exit_code == 0, res.output
    assert "status: complete" in res.output


# --- §11 in the CLI: a name is accepted, and a name is printed ---------------- #
def _named_experiment(tmp_path):
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001").set_name("Fish tank A")
    return proj


def test_experiment_status_lists_recording_names(tmp_path):
    """The CLI's only recording listing. It printed ids while the GUI printed names."""
    proj = _named_experiment(tmp_path)
    res = _runner().invoke(
        main, ["experiment", "status", "--experiment", str(proj.dir)]
    )
    assert res.exit_code == 0, res.output
    assert "Fish tank A" in res.output


def test_a_recording_name_resolves_to_its_id(tmp_path):
    """`--recording` advertises "Recording name (or id)"; only ids used to resolve, so the
    advertised name failed with "no recording" on every processing command."""
    from detail.cli import _resolve_recording_refs

    proj = _named_experiment(tmp_path)
    assert _resolve_recording_refs(proj, ["Fish tank A"]) == ["recording_001"]
    assert _resolve_recording_refs(proj, ["recording_001"]) == ["recording_001"]


def test_an_unknown_recording_is_a_clean_error_not_a_traceback(tmp_path):
    from detail.cli import _resolve_recording_refs

    proj = _named_experiment(tmp_path)
    with pytest.raises(click.ClickException):
        _resolve_recording_refs(proj, ["no such recording"])


def test_every_processing_command_resolves_what_the_user_typed():
    """Guard: a fourth processing command must not pass raw refs to the run API.

    Scans both CLI modules: ``cli.py`` and ``cli_keypoints.py`` (the ``keypoints`` group,
    §26 "Surfaces"), so a keypoint processing command is held to the same rule.
    """
    from pathlib import Path

    import detail

    package = Path(detail.__file__).parent
    source = "\n".join(
        (package / name).read_text(encoding="utf-8-sig")
        for name in ("cli.py", "cli_keypoints.py")
    )
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "run_processing(" in line and "_resolve_recording_refs" not in line
    ]
    # the call is wrapped, so the refs land on the following line; check the whole call
    calls = source.count("proj.run_processing(")
    resolved = source.count("_resolve_recording_refs(proj, recording_ids)")
    assert resolved == calls, (
        f"{calls} run_processing call(s) but {resolved} resolve through the name→id "
        f"translation: {offenders}. --recording advertises a name; it has to work."
    )


def test_the_keypoints_group_holds_exactly_the_subcommands_that_have_landed():
    """§26 *Sequencing*: the ``keypoints`` group lives in ``cli_keypoints.py`` and is added to
    the root group; each PR adds its subcommands (PR 2: ``examples``; PR 4:
    ``detect``; PR 5: ``track``; PR 8: ``distill``), so this set grows one PR at a time
    and a subcommand that lands early, or vanishes, fails here. PR 9a's ``audit`` group was
    removed (``docs/removed_capabilities.md``).
    """
    result = CliRunner().invoke(main, ["keypoints", "--help"])
    assert result.exit_code == 0, result.output
    assert "Keypoints" in result.output
    from detail.cli_keypoints import keypoints

    assert set(keypoints.commands) == {
        "examples",
        "detect",
        "track",
        "distill",
        "export",
        "test",
    }
    assert main.commands["keypoints"] is keypoints


# --------------------------------------------------------------------------- #
# comparison: the unit of analysis is demanded only when it exists
# --------------------------------------------------------------------------- #
def test_comparison_names_the_animal_column_when_it_refuses(tmp_path):
    """A refusal that says "pass a column" without saying *which* leaves the user exactly where
    the GUI's first version left them. The GUI pre-selects the inferred column; the CLI cannot,
    so it names it — same capability, different surface (CLAUDE.md surface parity)."""
    from tests.api.test_comparison import UNBALANCED, _cohort

    exp = _cohort(tmp_path, UNBALANCED)
    result = CliRunner().invoke(
        main,
        [
            "comparison",
            "run",
            "--experiment",
            str(exp.dir),
            "--source-run",
            "an_01",
            "--group-by",
            "drug",
        ],
    )

    assert result.exit_code != 0
    assert "'animal'" in result.output and "3 times" in result.output
    assert "--aggregate-to animal" in result.output


def test_comparison_does_not_demand_an_answer_when_nothing_repeats(tmp_path):
    """Four animals, four recordings. There is one possible answer, so asking for it is a
    question with one possible response — the CLI takes `[]` and runs."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(
        tmp_path,
        {
            "r1": ({"animal": "A", "drug": "veh"}, 2),
            "r2": ({"animal": "B", "drug": "veh"}, 4),
            "r3": ({"animal": "C", "drug": "fent"}, 11),
            "r4": ({"animal": "D", "drug": "fent"}, 20),
        },
    )
    result = CliRunner().invoke(
        main,
        [
            "comparison",
            "run",
            "--experiment",
            str(exp.dir),
            "--source-run",
            "an_01",
            "--group-by",
            "drug",
        ],
    )

    assert result.exit_code == 0, result.output
    assert exp.comparisons and exp.comparisons[-1].record()["aggregate_to"] == []


def test_the_window_options_come_from_the_model():
    """`--windows`' kinds, and every window knob's default and help, were typed into
    `_window_options` beside a `WindowSpec` that already declared them.

    A fifth kind, or a changed default, would have left `--help` confidently listing the old
    ones — the failure `configs/` exists to make impossible, and one a passing suite cannot see
    because a stale list is still a valid list. The GUI's copy of the same list is guarded in
    `tests/gui/test_comparison_gui.py`.
    """
    from typing import get_args

    from click.testing import CliRunner

    from detail.cli import main
    from detail.configs.windows import WindowSpec

    text = CliRunner().invoke(main, ["comparison", "run", "--help"]).output
    fields = WindowSpec.model_fields

    for kind in get_args(fields["kind"].annotation):
        assert kind in text, f"--windows does not offer {kind!r}"
    for name in ("every_s", "parts", "length_s", "step_s"):
        assert (
            str(fields[name].default) in text
        ), f"--help does not show {name}'s default of {fields[name].default}"


def test_keypoints_detect_is_process_keypoint_detect_and_shapes_the_config(
    monkeypatch, tmp_path
):
    """§26 *Surfaces* names both routes; they are one click command registered twice, so
    the options and their defaults cannot drift. The config the command hands
    ``run_processing`` is the module's (`KeypointDetectModule.validate` resolves the rest).
    """
    from detail.api.experiment import Experiment
    from detail.cli_keypoints import keypoints

    detect = keypoints.commands["detect"]
    assert main.commands["process"].commands["keypoint_detect"] is detect
    for path in (
        ["keypoints", "detect", "--help"],
        ["process", "keypoint_detect", "--help"],
    ):
        result = CliRunner().invoke(main, path)
        assert result.exit_code == 0, result.output
        assert "--example-set" in result.output and "--side-rule" in result.output

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    seen: dict = {}

    def _capture(self, module_id, recording_ids, config, progress=None):
        seen.update(module_id=module_id, recording_ids=recording_ids, config=config)
        raise ValueError("stop here")  # the run itself is the pipeline tests' job

    monkeypatch.setattr(Experiment, "run_processing", _capture)
    base = [
        "keypoints",
        "detect",
        "--experiment",
        str(proj.dir),
        "--recording",
        "recording_001",
    ]
    result = CliRunner().invoke(
        main,
        base
        + [
            "--detection",
            "det_01",
            "--model",
            "dinov3_match",
            "--example-set",
            "ex_01",
            "--every-n-frames",
            "2",
            "--crop-px",
            "672",
            "--side-rule",
            "hard",
        ],
    )
    assert result.exit_code == 1 and "stop here" in result.output
    assert seen["module_id"] == "keypoint_detect"
    assert seen["recording_ids"] == ["recording_001"]
    config = seen["config"]
    assert (
        config["detection_run_id"] == "det_01" and config["model_id"] == "dinov3_match"
    )
    assert "tracking_run_id" not in config and "track_ids" not in config
    assert config["example_set_id"] == "ex_01" and config["every_n_frames"] == 2
    assert (config["crop_px"], config["side_rule"], config["decode"]) == (
        672,
        "hard",
        None,
    )
    assert config["min_pose_confidence"] is None  # validate resolves it per method
    assert config["batch_size"] == 16  # the model's default, read off the model
    assert config["compute"].backend == "cpu"

    result = CliRunner().invoke(
        main,
        base
        + [
            "--detection",
            "det_01",
            "--model",
            "model_007",
            "--min-pose-confidence",
            "0.4",
        ],
    )
    assert "stop here" in result.output
    assert seen["config"]["min_pose_confidence"] == 0.4

    import detail.cli as cli_module

    monkeypatch.setattr(cli_module, "_spec_from_ref", lambda p, m, ref: {m: ref})
    result = CliRunner().invoke(
        main,
        base + ["--model", "dinov3_match", "--detection-ref", "recording_001:det_01"],
    )
    assert "stop here" in result.output
    assert seen["config"]["detection_spec"] == {"detection": "recording_001:det_01"}
    assert "detection_run_id" not in seen["config"]

    for extra, fragment in (
        (["--model", "dinov3_match"], "exactly one of --detection"),
        (
            [
                "--model",
                "dinov3_match",
                "--detection",
                "det_01",
                "--detection-ref",
                "r:det_01",
            ],
            "exactly one of --detection",
        ),
        (
            ["--model", "dinov3_match", "--detection", "det_01", "--tracks", "0"],
            "No such option",
        ),
        (
            [
                "--model",
                "x",
                "--detection",
                "det_01",
                "--min-pose-confidence",
                "1.5",
            ],
            "--min-pose-confidence",
        ),
    ):
        result = CliRunner().invoke(main, base + extra)
        assert result.exit_code != 0 and fragment in result.output, result.output


def test_keypoints_track_is_process_keypoint_track_and_types_method_params(
    monkeypatch, tmp_path
):
    """§26 *Surfaces*: one click command registered under both routes. A ``--method-param``
    is typed by the method's config model, so the command knows no field by name."""
    from detail.api.experiment import Experiment
    from detail.cli_keypoints import keypoints

    track = keypoints.commands["track"]
    assert main.commands["process"].commands["keypoint_track"] is track
    for path in (
        ["keypoints", "track", "--help"],
        ["process", "keypoint_track", "--help"],
    ):
        result = CliRunner().invoke(main, path)
        assert result.exit_code == 0, result.output
        assert "--method-param" in result.output and "--detect-ref" in result.output
        assert "--tracking" in result.output and "--match-min-inside" in result.output

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    seen: dict = {}

    def _capture(self, module_id, recording_ids, config, progress=None):
        seen.update(module_id=module_id, recording_ids=recording_ids, config=config)
        raise ValueError("stop here")  # the run itself is the pipeline tests' job

    monkeypatch.setattr(Experiment, "run_processing", _capture)
    base = [
        "keypoints",
        "track",
        "--experiment",
        str(proj.dir),
        "--recording",
        "recording_001",
    ]
    result = CliRunner().invoke(
        main,
        base
        + [
            "--detect",
            "kd_01",
            "--method",
            "temporal_decode",
            "--method-param",
            "smooth_window_s=0.2",
            "--method-param",
            "despike=false",
            "--method-param",
            "decode_cutoff=none",
        ],
    )
    assert result.exit_code == 1 and "stop here" in result.output
    assert seen["module_id"] == "keypoint_track"
    assert seen["recording_ids"] == ["recording_001"]
    assert seen["config"] == {
        "tracking_run_id": None,
        "match_min_inside": 0.5,
        "match_mask_grow": 0.05,
        "method": "temporal_decode",
        "config": {"smooth_window_s": 0.2, "despike": False, "decode_cutoff": None},
        "detect_run_id": "kd_01",
    }
    result = CliRunner().invoke(main, base + ["--detect", "kd_01"])
    assert "stop here" in result.output
    # Unnamed tracking run and method stay None: validate resolves both (§26).
    assert seen["config"] == {
        "tracking_run_id": None,
        "match_min_inside": 0.5,
        "match_mask_grow": 0.05,
        "method": None,
        "config": {},
        "detect_run_id": "kd_01",
    }
    result = CliRunner().invoke(
        main,
        base
        + [
            "--detect",
            "kd_01",
            "--tracking",
            "track_02",
            "--match-min-inside",
            "0.7",
            "--match-mask-grow",
            "0",
        ],
    )
    assert "stop here" in result.output
    assert (
        seen["config"]["tracking_run_id"],
        seen["config"]["match_min_inside"],
        seen["config"]["match_mask_grow"],
    ) == ("track_02", 0.7, 0.0)

    import detail.cli as cli_module

    monkeypatch.setattr(cli_module, "_spec_from_ref", lambda p, m, ref: {"ref": ref})
    result = CliRunner().invoke(main, base + ["--detect-ref", "recording_001:kd_01"])
    assert "stop here" in result.output
    assert seen["config"]["keypoint_detect_spec"] == {"ref": "recording_001:kd_01"}
    assert "detect_run_id" not in seen["config"]
    result = CliRunner().invoke(
        main,
        base + ["--detect", "kd_01", "--tracking-ref", "recording_001:track_02"],
    )
    assert "stop here" in result.output
    # An aux spec (§6.5): run_spec.resolve_into_config resolves it per recording.
    assert seen["config"]["tracking_spec"] == {"ref": "recording_001:track_02"}
    assert seen["config"]["tracking_run_id"] is None

    for extra, fragment in [
        (["--detect", "kd_01", "--detect-ref", "r:kd_01"], "exactly one of"),
        ([], "exactly one of"),
        (["--detect", "kd_01", "--method-param", "despike=false"], "needs --method"),
        (["--detect", "kd_01", "--match-min-inside", "0"], "--match-min-inside"),
        (
            [
                "--detect",
                "kd_01",
                "--tracking",
                "track_01",
                "--tracking-ref",
                "r:track_01",
            ],
            "one of --tracking or --tracking-ref",
        ),
        (["--detect", "kd_01", "--match-mask-grow", "-0.1"], "--match-mask-grow"),
        (
            [
                "--detect",
                "kd_01",
                "--method",
                "despike_smooth",
                "--method-param",
                "x=1",
            ],
            "has no parameter 'x'",
        ),
        (
            [
                "--detect",
                "kd_01",
                "--method",
                "despike_smooth",
                "--method-param",
                "despike",
            ],
            "expected NAME=VALUE",
        ),
        (
            [
                "--detect",
                "kd_01",
                "--method",
                "despike_smooth",
                "--method-param",
                "despike_k=abc",
            ],
            "despike_k",
        ),
    ]:
        result = CliRunner().invoke(main, base + extra)
        assert result.exit_code != 0 and fragment in result.output, (
            extra,
            result.output,
        )


def test_keypoints_export_routes_to_run_export_with_only_the_options_given(
    monkeypatch, tmp_path
):
    """§26 *Surfaces*: the keypoint targets are on `export run` too; this command adds the
    keypoint options and sends only what was typed, so a target's own default is never
    overridden by a flag the user did not touch."""
    from detail.api.experiment import Experiment
    from detail.cli_keypoints import keypoints

    assert "export" in keypoints.commands
    result = CliRunner().invoke(main, ["keypoints", "export", "--help"])
    assert result.exit_code == 0, result.output
    for word in ("keypoints_dlc", "keypoints_nwb", "--cutoff", "--source", "--overlay"):
        assert word in result.output

    proj = Experiment.create("p", tmp_path)
    seen: dict = {}

    def _capture(self, kind, subject_id, target_ids, parent, **options):
        seen.update(
            kind=kind, subject_id=subject_id, targets=target_ids, options=options
        )
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "run_export", _capture)
    base = ["keypoints", "export", "--experiment", str(proj.dir), "--run", "run_011"]
    result = CliRunner().invoke(main, base + ["--out", str(tmp_path / "out")])
    assert result.exit_code == 1 and "stop here" in result.output
    assert (seen["kind"], seen["subject_id"]) == ("run", "run_011")
    assert seen["targets"] == ["keypoints"] and seen["options"] == {}
    result = CliRunner().invoke(
        main,
        base
        + [
            "--out",
            str(tmp_path / "out"),
            "--target",
            "keypoints_dlc",
            "--target",
            "keypoints_video",
            "--source",
            "detect",
            "--cutoff",
            "0.3",
            "--blank-below-cutoff",
            "--overlay",
        ],
    )
    assert "stop here" in result.output
    assert seen["targets"] == ["keypoints_dlc", "keypoints_video"]
    assert seen["options"] == {
        "source": "detect",
        "cutoff": 0.3,
        "blank_below_cutoff": True,
        "overlay": True,
    }
    result = CliRunner().invoke(main, base + ["--out", "x", "--target", "tracks"])
    assert result.exit_code != 0 and "Invalid value" in result.output


def test_process_embedding_carries_the_alignment_options_and_the_keypoint_ref(
    monkeypatch, tmp_path
):
    """§14 egocentric alignment on the CLI: the five options reach the config, the keypoint
    run by id or by ref, and a setting left out takes the model's default rather than a hole.
    """
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    seen: dict = {}

    def _capture(self, module_id, recording_ids, config, progress=None):
        seen.update(module_id=module_id, config=config)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "run_processing", _capture)
    base = [
        "process",
        "embedding",
        "--experiment",
        str(proj.dir),
        "--recording",
        "recording_001",
        "--tracking",
        "track_01",
        "--tracks",
        "0",
        "--model",
        "dinov3_vit",
    ]
    result = CliRunner().invoke(main, base)
    assert "stop here" in result.output
    assert seen["config"]["align"] == "none" and seen["config"]["heading"] is None
    assert seen["config"]["keypoint_run_id"] is None
    assert seen["config"]["max_gap_s"] == 0.5  # the model's default, recorded
    result = CliRunner().invoke(
        main,
        base
        + [
            "--align",
            "rotate",
            "--keypoint-run",
            "kt_01",
            "--heading",
            "nose_to_tail",
            "--max-gap-s",
            "0.25",
            "--facing-camera-ratio",
            "0.4",
        ],
    )
    assert "stop here" in result.output, result.output
    config = seen["config"]
    assert (config["align"], config["keypoint_run_id"], config["heading"]) == (
        "rotate",
        "kt_01",
        "nose_to_tail",
    )
    assert (
        config["max_gap_s"],
        config["mirror_hold_s"],
        config["facing_camera_ratio"],
    ) == (
        0.25,
        1.0,
        0.4,
    )
    import detail.cli as cli_module

    monkeypatch.setattr(
        cli_module, "_spec_from_ref", lambda p, m, ref: {"ref": ref, "m": m}
    )
    result = CliRunner().invoke(
        main, base + ["--align", "mirror", "--keypoint-ref", "recording_001:kt_01"]
    )
    assert "stop here" in result.output, result.output
    assert seen["config"]["keypoint_track_spec"] == {
        "ref": "recording_001:kt_01",
        "m": "keypoint_track",
    }
    assert "keypoint_run_id" not in seen["config"]
    result = CliRunner().invoke(
        main, base + ["--keypoint-run", "kt_01", "--keypoint-ref", "r:kt_01"]
    )
    assert (
        result.exit_code != 0
        and "one of --keypoint-run or --keypoint-ref" in result.output
    )
    result = CliRunner().invoke(main, ["embedding", "run", "--help"])
    assert result.exit_code == 0 and "--align" in result.output


def test_heading_definitions_round_trip_through_the_experiment_commands(tmp_path):
    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    base = ["experiment", "--experiment", str(proj.dir)]

    def run(*args):
        result = CliRunner().invoke(
            main, ["experiment", args[0], "--experiment", str(proj.dir), *args[1:]]
        )
        return result

    result = run("list-headings")
    assert result.exit_code == 0 and "no saved heading definitions" in result.output
    result = run(
        "save-heading",
        "nose_to_tail",
        "--front",
        "nose, left_ear",
        "--back",
        "tail_base",
    )
    assert result.exit_code == 0, result.output
    assert proj.heading_definitions() == [
        {"name": "nose_to_tail", "front": ["nose", "left_ear"], "back": ["tail_base"]}
    ]
    result = run("list-headings")
    assert "nose_to_tail  front: nose, left_ear  back: tail_base" in result.output
    result = run(
        "save-heading", "nose_to_tail", "--front", "nose", "--back", "tail_base"
    )
    assert result.exit_code != 0 and "already exists" in result.output
    out = tmp_path / "h.yaml"
    assert run("export-heading", "nose_to_tail", "--out", str(out)).exit_code == 0
    assert run("delete-heading", "nose_to_tail").exit_code == 0
    assert proj.heading_definitions() == []
    result = run("import-heading", str(out), "--name", "h2")
    assert result.exit_code == 0 and "imported heading definition: h2" in result.output
    assert proj.get_heading_definition("h2")["front"] == ["nose", "left_ear"]
    result = run("save-heading", "x", "--front", "nose", "--back", "nose")
    assert result.exit_code != 0 and "both front and back" in result.output
    del base


_LAUNCHES = {
    "process detection": ["process", "detection", "--model", "fake_dl"],
    "process tracking": ["process", "tracking", "--detection", "det_01"],
    "process embedding": [
        "process",
        "embedding",
        "--tracking",
        "track_01",
        "--tracks",
        "0",
        "--model",
        "dinov3_vit",
    ],
    "keypoints detect": [
        "keypoints",
        "detect",
        "--detection",
        "det_01",
        "--model",
        "dinov3_match",
        "--example-set",
        "ex_01",
    ],
    "keypoints track": [
        "keypoints",
        "track",
        "--detect",
        "kd_01",
        "--method",
        "temporal_decode",
    ],
}


@pytest.mark.parametrize("command", [*_LAUNCHES, "process retry"])
def test_every_launch_names_the_recording_it_is_validating(
    command, monkeypatch, tmp_path
):
    """Validation before a run is O(recordings) disk reads, minutes over a large selection,
    and the CLI printed nothing until it was over. Every command that launches or retries a
    processing run drives the API's progress callback onto a bar, which names the recording.
    """
    from detail.api.experiment import Experiment
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    called = []

    def _reporting(self, *args, progress=None):
        called.append(args)
        progress("Validating", 0.0, "Rat 7 (1/1)")
        raise ValueError("stop here")

    if command == "process retry":
        monkeypatch.setattr(Experiment, "retry_run", _reporting)
        runs_store.write_run(
            proj.dir,
            "run_001",
            {
                "id": "run_001",
                "module_id": "detection",
                "recording_ids": ["recording_001"],
                "config": {},
                "per_recording": {},
                "status": "failed",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )
        argv = ["process", "retry", "run_001", "--experiment", str(proj.dir)]
    else:
        monkeypatch.setattr(Experiment, "run_processing", _reporting)
        argv = [
            *_LAUNCHES[command],
            "--experiment",
            str(proj.dir),
            "--recording",
            "recording_001",
        ]
    result = CliRunner().invoke(main, argv)
    assert called, result.output  # reached the API, not refused by click first
    assert "stop here" in result.output
    assert "Validating" in result.output and "Rat 7 (1/1)" in result.output


_TWO_BOX_MEDPC = """\
Start Date: 02/16/26
Subject: RAT1
Box: 1
H:      10.000
E:
     0:        2.000      100.000        0.000
T:
     0:        1.000       10.000        0.000

Start Date: 02/16/26
Subject: RAT2
Box: 2
H:       5.000
E:
     0:        2.000      100.000
T:
     0:        0.500        5.000
"""
_MEDPC_PROGRAM = "\\  Event stamps ( array E )\n\\   2 = Lever\n\\   100 = End\n\\\n"


def _add_medpc(tmp_path, *extra):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    exp.create_recording("r1")
    data = tmp_path / "day.txt"
    data.write_text(_TWO_BOX_MEDPC, encoding="utf-8")
    program = tmp_path / "task.MPC"
    program.write_text(_MEDPC_PROGRAM, encoding="utf-8")
    res = _runner().invoke(
        main,
        [
            "recording", "add-stream", "--experiment", str(exp.dir), "--recording", "r1",
            "--name", "medpc", "--importer", "medpc", "--data", str(data),
            "--program", str(program), *extra,
        ],
    )  # fmt: skip
    return exp, res


def test_add_stream_refuses_a_several_session_medpc_file_naming_them(tmp_path):
    exp, res = _add_medpc(tmp_path)
    assert res.exit_code != 0
    assert (
        "holds 2 sessions (1: Subject RAT1, box 1, 02/16/26; 2: Subject RAT2"
        in res.output
    )
    assert exp.get_recording("r1").streams == []


def test_add_stream_medpc_box_picks_one_session_and_reports_dropped_slots(tmp_path):
    exp, res = _add_medpc(tmp_path, "--medpc-box", "1")
    assert res.exit_code == 0, res.output
    assert "2 rows" in res.output
    assert (
        "note: dropped 1 unused (code 0, time 0) slot(s) after the last event"
        in res.output
    )
    assert exp.get_recording("r1").streams[0]["metadata"]["subject"] == "RAT1"


def test_a_selector_flag_for_another_importer_is_refused_not_ignored(tmp_path):
    exp, res = _add_medpc(tmp_path, "--boris-subject", "mA")
    assert res.exit_code != 0
    assert "--boris-subject only apply to the importer they are named for" in res.output


def test_set_stream_offset_refuses_behavior_labels(tmp_path):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p1", tmp_path)
    recording = exp.create_recording("r1")
    vid = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    for _ in range(10):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    recording.add_video_part(vid)
    labels = tmp_path / "labels.csv"
    labels.write_text("start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8")
    recording.add_stream("behavior", raw_path=labels, importer="behavior_csv")
    res = _runner().invoke(
        main,
        [
            "recording", "set-stream-offset", "--experiment", str(exp.dir),
            "--recording", "r1", "--name", "behavior", "--offset", "5",
        ],
    )  # fmt: skip
    assert res.exit_code != 0
    assert "already the video's own" in res.output
    assert exp.get_recording("r1").streams[0]["alignment"]["offset_ns"] == 0
