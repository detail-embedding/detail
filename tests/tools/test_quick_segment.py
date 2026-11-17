"""Quick-segment engine + Tools API (§9): capped sequential segmentation over a raw clip.

**A class guarded here: a compute default the selected model cannot run on.** The CLI's
``tools quick-segment`` picks SAM 3 on a machine with an NVIDIA GPU, and SAM 3 runs on CUDA
only, while ``--backend`` defaulted to ``cpu`` like the stage commands. So the command as
documented failed on every machine it could run on, and the docs told users to pass
``--backend cuda`` by hand. An omitted ``--backend`` now resolves with
``best_supported_backend`` over the chosen model's own backends, as the GUI's compute row
does, and the manifest records the backend chosen
(``test_cli_quick_segment_with_no_backend_runs_on_one_the_model_supports``).
"""

import cv2
import numpy as np
import pytest
import zarr

from detail.api import tools as api_tools
from detail.core.dataclasses import (
    ComputeTarget,
    Detection,
    FrameResult,
    ModelReadiness,
    Prompt,
    PromptTarget,
    TextCue,
)
from detail.core.protocols import DetectionModel, ModelNotReadyError
from detail.core.registries import register_model
from detail.storage import locations, tool_runs
from detail.tools import quick_segment

W, H = 64, 48


def _write_video(path, n_frames, fps=30.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert vw.isOpened()
    for i in range(n_frames):
        vw.write(np.full((H, W, 3), (i * 17) % 256, dtype=np.uint8))
    vw.release()


class _StubDetector(DetectionModel):
    """One detection per prompt target each frame — a mask + normalized bbox, no backend."""

    model_id = "stub_seg"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu"]

    def load(self, device):
        self.device = device

    def set_conf_threshold(self, threshold):
        self._conf = threshold

    def predict(self, frame, prompt):
        dets = []
        for target in prompt.targets:
            mask = np.zeros((H, W), dtype=bool)
            mask[H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = True
            dets.append(
                Detection(
                    mask=mask,
                    bbox=(0.25, 0.25, 0.75, 0.75),
                    class_name=target.class_name,
                    confidence=0.9,
                    object_id=None,
                )
            )
        return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=dets)

    def predict_batch(self, frames, prompts):
        return [self.predict(f, p) for f, p in zip(frames, prompts)]


class _VariedConfidenceDetector(_StubDetector):
    """Two proposals per target — one strong, one weak.

    The plain stub returns a single 0.9 detection, so a display threshold could be broken
    (or ignored) without any test noticing. A run keeps every proposal now, so the weak one
    is exactly what the slider exists to reveal.
    """

    model_id = "stub_varied"

    def predict(self, frame, prompt):
        dets = []
        for target in prompt.targets:
            for confidence, corner in ((0.9, 0), (0.2, H // 2)):
                mask = np.zeros((H, W), dtype=bool)
                mask[corner : corner + H // 4, corner : corner + W // 4] = True
                dets.append(
                    Detection(
                        mask=mask,
                        bbox=(0.05, 0.05, 0.3, 0.3),
                        class_name=target.class_name,
                        confidence=confidence,
                        object_id=None,
                    )
                )
        return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=dets)


@pytest.fixture
def varied_model():
    register_model("stub_varied")(_VariedConfidenceDetector)
    return "stub_varied"


@pytest.fixture
def stub_model():
    register_model("stub_seg")(
        _StubDetector
    )  # restored by the registry-isolation fixture
    return "stub_seg"


@pytest.fixture
def isolated_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return tmp_path


def _config(
    video,
    n_frames=8,
    classes=("mouse", "cricket"),
    mode="consecutive",
    videos=None,
    prompts=None,
    whole_video=False,
):
    """A tool config. Defaults to ``consecutive`` — the mode these zarr/mp4 tests are
    about; the tool's own default is ``sample`` (see the sampling tests)."""
    one = Prompt(targets=[PromptTarget(c, TextCue(c)) for c in classes])
    paths = [str(v) for v in (videos or [video])]
    return {
        "video": paths[0],
        "videos": paths,
        "prompt": (prompts or [one])[0],
        "prompts": prompts or [one],
        "model_id": "stub_seg",
        "model_variant": None,
        "compute": ComputeTarget("cpu", None, None, None),
        "mode": mode,
        "whole_video": whole_video,
        "n_frames": n_frames,
    }


def test_run_tool_writes_detection_zarr_with_the_pipeline_schema(
    tmp_path, stub_model, isolated_tools
):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=12)
    record = api_tools.run_tool("quick_segment", _config(video, n_frames=8))

    assert record["status"] == "complete"
    assert record["source_video"] == str(video.resolve())  # absolute path (exemption)
    result = record["result"]
    assert result["n_processed"] == 8  # cap (8) < total (~12)
    assert result["frames_total"] >= 8
    assert result["class_counts"] == {"mouse": 8, "cricket": 8}  # 2 classes/frame

    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    root = zarr.open_group(str(run_dir / "detections.zarr"), mode="r")
    assert root.attrs["schema_version"] == 1
    assert root["frame_idx"].dtype == np.int32
    assert root["t_ns"].dtype == np.int64
    assert root["bbox"].shape[1] == 4 and root["bbox"].dtype == np.float32
    assert str(root["class_name"].dtype).startswith("str") or root[
        "class_name"
    ].dtype.kind in ("U", "T", "O")
    assert root["mask_rle"].chunks == (1000,)
    # 2 detections per frame × 8 frames
    assert root["frame_idx"].shape[0] == 16
    # t_ns now comes from the clip's generated PTS map, not frame_idx / fps — on this
    # uniformly-timed 30 fps clip the two agree, which is what makes the map safe to adopt
    assert root["t_ns"][:].max() == round(7 * 1e9 / 30.0)
    # ...and the map is saved so playback/export resolve the same frames the run counted
    assert (run_dir / "timestamps.npy").is_file()
    assert result["outputs"]["timestamps"] == "timestamps.npy"


def test_cap_is_min_of_request_and_video_length(tmp_path, stub_model, isolated_tools):
    video = tmp_path / "short.mp4"
    _write_video(video, n_frames=5)  # shorter than the requested cap
    record = api_tools.run_tool("quick_segment", _config(video, n_frames=100))
    assert record["result"]["n_processed"] == 5  # min(100, 5)


def test_run_tool_records_running_then_failed_on_error(
    tmp_path, stub_model, isolated_tools
):
    video = tmp_path / "bad.mp4"
    _write_video(video, n_frames=4)
    cfg = _config(video)
    cfg["model_id"] = "does_not_exist"
    with pytest.raises(ValueError, match="unknown model"):
        api_tools.run_tool("quick_segment", cfg)
    runs = api_tools.list_tool_runs()
    assert len(runs) == 1 and runs[0]["status"] == "failed"


class _BackendMissingDetector(_StubDetector):
    """A model whose backend package is missing: readiness says so, and load must not run."""

    model_id = "stub_unready"

    def readiness(self):
        return ModelReadiness(
            ready=False,
            reason="missing_dependency",
            message="stub needs its backend package: install it",
            action_url=None,
            repo_id=None,
        )

    def load(self, device):
        raise AssertionError("load() ran past a failed readiness pre-flight")


def test_a_model_that_is_not_ready_fails_with_its_readiness_message(
    tmp_path, isolated_tools
):
    # The tool loads the model in the app's own process. Without the pre-flight a missing
    # backend surfaced as a bare ModuleNotFoundError, and the probe inside readiness is
    # what lets a backend installed with the app open import here (#506).
    register_model("stub_unready")(_BackendMissingDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    cfg = _config(video)
    cfg["model_id"] = "stub_unready"
    with pytest.raises(ModelNotReadyError) as raised:
        api_tools.run_tool("quick_segment", cfg)
    assert str(raised.value) == "stub needs its backend package: install it"
    runs = api_tools.list_tool_runs()
    assert len(runs) == 1 and runs[0]["status"] == "failed"


def test_resolve_pretrained_or_imported_finds_an_imported_model(tmp_path, monkeypatch):
    # A tool resolves a machine-global imported/distilled model (no experiment scope) —
    # the closed-set YOLO path that lets a tool run without SAM 3 / a GPU.
    from detail.api import models as api_models
    from detail.core.registries import TRAINED_LOADERS
    from detail.models.resolve import resolve_pretrained_or_imported
    from tests.models.test_imported import _StubLoader

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubLoader)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    mid = api_models.import_model(weights, family="stubfam", name="my_yolo")
    model = resolve_pretrained_or_imported(mid)
    assert isinstance(model, _StubLoader)  # resolved from the imported catalog


def test_progress_streams_intermediate_events(
    tmp_path, stub_model, isolated_tools, monkeypatch
):
    # With the throttle interval at 0, every frame emits — assert a mid-run fraction, not
    # just the terminal one (the live-bar cadence rule).
    monkeypatch.setattr("detail.core.progress.PROGRESS_INTERVAL_S", 0.0)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=10)
    events = []
    api_tools.run_tool(
        "quick_segment",
        _config(video, n_frames=6),
        progress=lambda stage, frac, detail: events.append((stage, frac, detail)),
    )
    fractions = [f for _s, f, _d in events]
    assert any(0 < f < 1 for f in fractions)  # a mid-run event, not only 1.0
    assert fractions[-1] == 1.0


def test_export_mp4_renders_a_clip(tmp_path, stub_model, isolated_tools):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=8)
    record = api_tools.run_tool("quick_segment", _config(video, n_frames=6))
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    out = tmp_path / "overlay.mp4"
    quick_segment.export_mp4(run_dir, record, out)
    assert out.is_file() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.read()[0]  # at least one frame decodes
    cap.release()


def test_delete_tool_run_removes_it(tmp_path, stub_model, isolated_tools):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    record = api_tools.run_tool("quick_segment", _config(video, n_frames=4))
    assert api_tools.list_tool_runs()
    api_tools.delete_tool_run(record["id"])
    assert api_tools.list_tool_runs() == []


def test_cli_quick_segment_runs_lists_exports_deletes(
    tmp_path, stub_model, isolated_tools
):
    from click.testing import CliRunner

    from detail import cli

    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=6)
    export = tmp_path / "overlay.mp4"
    runner = CliRunner()

    # --export needs contiguous frames, so this is the consecutive mode. One prompt
    # carrying both classes: repeated --prompt now means separate prompts to compare.
    res = runner.invoke(
        cli.main,
        [
            "tools", "quick-segment", str(video),
            "--consecutive",
            "--prompt", "mouse,cricket",
            "--model", "stub_seg", "--backend", "cpu",
            "--frames", "5",
            "--export", str(export),
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "quick-segment run_001" in res.output and export.is_file()

    listed = runner.invoke(cli.main, ["tools", "list"])
    assert listed.exit_code == 0 and "quick_segment" in listed.output

    deleted = runner.invoke(cli.main, ["tools", "delete", "run_001"])
    assert deleted.exit_code == 0
    assert api_tools.list_tool_runs() == []


@pytest.mark.parametrize(
    "supported,cuda,mps,expected",
    [
        (
            ["cuda"],
            True,
            False,
            "cuda",
        ),  # SAM 3's shape: the default model on a GPU box
        (["cpu", "mps", "cuda"], False, True, "mps"),
        (["cpu", "mps", "cuda"], False, False, "cpu"),
    ],
)
def test_cli_quick_segment_with_no_backend_runs_on_one_the_model_supports(
    tmp_path, isolated_tools, monkeypatch, supported, cuda, mps, expected
):
    """The bare command — no ``--backend``, no ``--model`` — runs, on the best backend the
    machine has that the model accepts, and the manifest records the one it chose.

    ``--backend`` used to default to ``cpu``, and the model the command picks for a GPU
    machine is SAM 3, which runs on CUDA only: the command as documented failed on every
    machine it could run on. The GUI's compute row already picked this way."""
    from click.testing import CliRunner

    from detail import cli
    from detail.core import utils

    class _Picky(_StubDetector):
        model_id = "stub_picky"
        supported_backends = supported

    register_model("stub_picky")(_Picky)
    monkeypatch.setattr(api_tools, "default_model_for_machine", lambda: "stub_picky")
    monkeypatch.setattr(utils, "cuda_available", lambda: cuda)
    monkeypatch.setattr(utils, "mps_available", lambda: mps)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=6)

    res = CliRunner().invoke(
        cli.main, ["tools", "quick-segment", str(video), "--prompt", "mouse"]
    )
    assert res.exit_code == 0, res.output
    (record,) = api_tools.list_tool_runs()
    assert record["compute"]["backend"] == expected
    assert record["config"]["mode"] == "sample"


def test_sampled_mode_spreads_frames_across_the_whole_clip(
    tmp_path, stub_model, isolated_tools
):
    # The point of the default mode: reading the first N frames of a 60-frame clip shows
    # only the opening two seconds, where a still animal or a blank lead-in shows nothing.
    import json

    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=60)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=6, mode="sample")
    )
    assert record["status"] == "complete"
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    meta = json.loads((run_dir / "test.json").read_text(encoding="utf-8-sig"))
    sampled = [f["frame_idx"] for f in meta["sampled_frames"]]
    assert len(sampled) == 6
    assert min(sampled) < 10 and max(sampled) > 45  # both ends represented
    # Detections, not pictures: a run writes no images at all.
    assert not list(run_dir.rglob("*.jpg"))
    assert sampled == sorted(sampled) and len(set(sampled)) == 6
    # Even sampling is deterministic — no seed needed to reproduce a run.
    again = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=6, mode="sample")
    )
    again_dir = tool_runs.tool_run_dir("quick_segment", again["id"])
    assert json.loads((again_dir / "test.json").read_text(encoding="utf-8-sig"))[
        "sampled_frames"
    ] == (meta["sampled_frames"])


def test_sampled_mode_compares_several_prompts(tmp_path, stub_model, isolated_tools):
    import json

    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=30)
    prompts = [
        Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        Prompt(targets=[PromptTarget("mouse with tail", TextCue("mouse with tail"))]),
    ]
    record = api_tools.run_tool(
        "quick_segment",
        _config(video, n_frames=4, mode="sample", prompts=prompts),
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    meta = json.loads((run_dir / "test.json").read_text(encoding="utf-8-sig"))
    assert [p["label"] for p in meta["prompts"]] == ["prompt_0", "prompt_1"]
    # One detection store per (prompt, clip) — the grid renders overlays from these.
    assert (run_dir / "prompt_0" / "part_000.zarr").is_dir()
    assert (run_dir / "prompt_1" / "part_000.zarr").is_dir()
    assert not list(run_dir.rglob("*.jpg"))  # no pictures until an explicit export
    assert record["result"]["outputs"]["prompt_test"] == "test.json"
    assert "2 prompt(s)" in record["result"]["summary"]
    # Both prompts are recorded on the manifest, with the singular alias kept.
    assert len(record["prompts"]) == 2
    assert record["prompt"] == record["prompts"][0]


class _ClosedSetDetector(DetectionModel):
    """A trained/imported model: no text cue, returns its own classes regardless of
    prompt — the YOLO shape."""

    model_id = "stub_closed"
    supported_cue_types = []
    supported_backends = ["cpu"]

    def load(self, device):
        self._device = device

    def set_conf_threshold(self, threshold):
        self._conf = threshold

    def _dets(self, shape):
        h, w = shape[:2]
        out = []
        for i, name in enumerate(("rat", "bedding")):
            mask = np.zeros((h, w), dtype=bool)
            mask[i * 4 : i * 4 + 3, :3] = True
            out.append(
                Detection(
                    mask=mask,
                    bbox=(0.1, 0.1, 0.2, 0.2),
                    class_name=name,
                    confidence=0.9,
                    object_id=None,
                )
            )
        return out

    def predict(self, frame, prompt):
        return FrameResult.from_detections(
            frame_idx=0, t_ns=0, detections=self._dets(frame.shape)
        )

    def predict_batch(self, frames, prompt):
        return [self.predict(f, prompt) for f in frames]


def test_sampled_mode_runs_a_closed_set_model_with_an_empty_prompt(
    tmp_path, isolated_tools
):
    # Reported: type prompts, switch the model from SAM 3 to a YOLO, run -> "list index
    # out of range". A closed-set model takes no text, so the form sends an EMPTY prompt,
    # and the closed-set branch indexed targets[0]. Empty now means "whatever it finds".
    register_model("stub_closed")(_ClosedSetDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=12)
    config = _config(video, n_frames=3, mode="sample")
    config["model_id"] = "stub_closed"
    config["prompts"] = [Prompt(targets=[])]
    config["prompt"] = config["prompts"][0]

    record = api_tools.run_tool("quick_segment", config)
    assert record["status"] == "complete"
    # Both of the model's own classes survive; nothing was filtered against a typed name.
    assert record["result"]["class_counts"] == {"rat": 3, "bedding": 3}


def test_closed_set_prompt_can_still_narrow_to_named_classes(tmp_path, isolated_tools):
    register_model("stub_closed")(_ClosedSetDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=12)
    config = _config(video, n_frames=3, mode="sample")
    config["model_id"] = "stub_closed"
    # A prompt naming classes narrows the column — and names *all* of them, not just the
    # first, now that a prompt may carry several.
    config["prompts"] = [
        Prompt(targets=[PromptTarget("rat", TextCue("rat"))]),
        Prompt(
            targets=[
                PromptTarget("rat", TextCue("rat")),
                PromptTarget("bedding", TextCue("bedding")),
            ]
        ),
    ]
    config["prompt"] = config["prompts"][0]
    record = api_tools.run_tool("quick_segment", config)
    assert record["status"] == "complete"
    # 3 frames: prompt_0 keeps rat only (3), prompt_1 keeps both (6).
    assert record["result"]["class_counts"] == {"rat": 6, "bedding": 3}


def test_export_frames_renders_the_overlays_on_request(
    tmp_path, stub_model, isolated_tools
):
    # The run holds detections; images are made only here, when asked for.
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=20)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=3, mode="sample")
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    assert not list(run_dir.rglob("*.jpg"))  # nothing rendered by the run itself

    out = tmp_path / "share"
    written = quick_segment.export_frames(run_dir, record, out)
    jpgs = sorted(p.name for p in out.glob("*.jpg"))
    assert len(jpgs) == 3 and (out / "test.json").is_file()
    assert written == 4  # 3 overlays + the summary


def test_export_run_makes_one_named_folder_per_mode(
    tmp_path, stub_model, isolated_tools
):
    # An export leaves the app, so it has to say what it is a week later: one folder,
    # named from the clip + the run, not loose files in someone's Downloads.
    video = tmp_path / "openfield_01.mp4"
    _write_video(video, n_frames=16)
    parent = tmp_path / "share"
    parent.mkdir()

    sampled = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=3, mode="sample")
    )
    out = quick_segment.export_run(
        tool_runs.tool_run_dir("quick_segment", sampled["id"]), sampled, parent
    )
    assert out.name == f"openfield_01_quick-segment_{sampled['id']}"
    assert out.parent == parent
    assert len(list(out.glob("*.jpg"))) == 3 and (out / "test.json").is_file()

    consecutive = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=4, mode="consecutive")
    )
    out2 = quick_segment.export_run(
        tool_runs.tool_run_dir("quick_segment", consecutive["id"]), consecutive, parent
    )
    # Same shape for a video run: its own folder, and the clip named after it rather
    # than a bare overlay.mp4.
    assert out2.name == f"openfield_01_quick-segment_{consecutive['id']}"
    assert out2 != out  # two exports never land on top of each other
    mp4s = list(out2.glob("*.mp4"))
    assert [p.name for p in mp4s] == [f"{out2.name}.mp4"]
    assert mp4s[0].stat().st_size > 0


def test_exporting_one_run_twice_makes_two_folders(
    tmp_path, stub_model, isolated_tools
):
    """``docs/tools.md``: "Two exports never land on top of each other." The folder is named
    by the run, so a second export of the same run — at another threshold — reused it and
    replaced the first one's pictures (#29). It gets ``_2`` instead."""
    video = tmp_path / "openfield_01.mp4"
    _write_video(video, n_frames=16)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=3, mode="sample")
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    first = quick_segment.export_run(run_dir, record, tmp_path / "share")
    (first / "note.txt").write_text("the first export", encoding="utf-8")
    second = quick_segment.export_run(run_dir, record, tmp_path / "share", min_conf=0.9)
    assert second == first.with_name(f"{first.name}_2")
    assert (first / "note.txt").is_file() and not (second / "note.txt").exists()


def test_export_frames_refuses_a_consecutive_run(tmp_path, stub_model, isolated_tools):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=8)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=4, mode="consecutive")
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    with pytest.raises(ValueError, match="Export mp4"):
        quick_segment.export_frames(run_dir, record, tmp_path / "share")


def test_sampled_overlays_draw_the_saved_detections(
    tmp_path, stub_model, isolated_tools
):
    # Rendering on demand has to reach the same picture the run segmented, which is what
    # the saved timestamp map buys.
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=16)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=3, mode="sample")
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    assert (run_dir / "timestamps" / "part_000.npy").is_file()
    overlays = quick_segment.sampled_overlays(run_dir, [str(video)], 0)
    assert set(overlays) == {"prompt_0"}
    frame = overlays["prompt_0"]
    assert frame is not None and frame.ndim == 3 and frame.shape[2] == 3


def test_export_mp4_refuses_a_sampled_run(tmp_path, stub_model, isolated_tools):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=20)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=4, mode="sample")
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    with pytest.raises(ValueError, match="not contiguous"):
        quick_segment.export_mp4(run_dir, record, tmp_path / "out.mp4")


def test_a_stopped_run_keeps_its_partial_output(tmp_path, stub_model, isolated_tools):
    # An uncapped run can cover hours, so stopping has to leave something openable
    # rather than throwing the work away.
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=20)
    calls = {"n": 0}

    def stop_after_3() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    record = api_tools.run_tool(
        "quick_segment",
        _config(video, n_frames=20, mode="consecutive"),
        stop_check=stop_after_3,
    )
    assert record["status"] == "stopped"
    assert record["result"]["stopped"] is True
    assert record["result"]["n_processed"] == 3
    assert "stopped early" in record["result"]["summary"]
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    root = zarr.open_group(str(run_dir / "detections.zarr"), mode="r")
    assert root["frame_idx"].shape[0] == 6  # 2 detections × the 3 frames it managed


def test_whole_video_ignores_the_frame_count(tmp_path, stub_model, isolated_tools):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=14)
    record = api_tools.run_tool(
        "quick_segment",
        _config(video, n_frames=2, mode="consecutive", whole_video=True),
    )
    assert record["result"]["n_processed"] == 14  # every frame, not the requested 2


def test_cli_quick_segment_reports_unknown_model_cleanly(
    tmp_path, stub_model, isolated_tools
):
    from click.testing import CliRunner

    from detail import cli

    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    res = CliRunner().invoke(
        cli.main,
        ["tools", "quick-segment", str(video), "--prompt", "mouse", "--model", "nope"],
    )
    assert res.exit_code != 0 and "unknown model" in res.output


# --------------------------------------------------------------------------- #
# Display threshold — the run keeps everything, the view chooses what to show
# --------------------------------------------------------------------------- #
def _varied_config(video, **kw):
    cfg = _config(video, **kw)
    cfg["model_id"] = "stub_varied"
    return cfg


def test_the_run_stores_every_proposal_including_the_weak_ones(
    tmp_path, varied_model, isolated_tools
):
    """The model is asked wide open, so a 0.2 proposal reaches the zarr. If the run
    filtered, no later threshold could bring it back — that is the whole feature."""
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    record = api_tools.run_tool(
        "quick_segment", _varied_config(video, n_frames=4, classes=("mouse",))
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    columns = quick_segment.load_detections(run_dir / "detections.zarr")
    confidences = sorted(set(float(c) for c in columns["confidence"]))
    assert confidences == [pytest.approx(0.2), pytest.approx(0.9)]


def test_frame_detection_count_recounts_at_a_threshold(
    tmp_path, varied_model, isolated_tools
):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    record = api_tools.run_tool(
        "quick_segment", _varied_config(video, n_frames=4, classes=("mouse",))
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    columns = quick_segment.load_detections(run_dir / "detections.zarr")

    total = quick_segment.frame_detection_count(columns, 0.0)
    assert total == 8  # 4 frames × 2 proposals
    assert quick_segment.frame_detection_count(columns, 0.5) == 4  # the 0.9s only
    assert quick_segment.frame_detection_count(columns, 0.9) == 4  # inclusive
    assert quick_segment.frame_detection_count(columns, 0.95) == 0


def test_overlay_frame_draws_less_as_the_threshold_rises(
    tmp_path, varied_model, isolated_tools
):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    record = api_tools.run_tool(
        "quick_segment", _varied_config(video, n_frames=4, classes=("mouse",))
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    columns = quick_segment.load_detections(run_dir / "detections.zarr")
    colors = quick_segment.class_colors([str(c) for c in columns["class_name"]])
    timestamps = quick_segment.load_timestamps(run_dir, record["result"])

    everything = quick_segment.overlay_frame(
        video, 0, columns, colors, timestamps, min_conf=0.0
    )
    strict = quick_segment.overlay_frame(
        video, 0, columns, colors, timestamps, min_conf=0.5
    )
    nothing = quick_segment.overlay_frame(
        video, 0, columns, colors, timestamps, min_conf=0.99
    )
    assert everything is not None and strict is not None and nothing is not None
    assert not np.array_equal(everything, strict)  # the weak proposal disappeared
    assert not np.array_equal(strict, nothing)  # then the strong one did


def test_a_filtered_row_never_pays_for_its_mask_decode(
    tmp_path, varied_model, isolated_tools, monkeypatch
):
    """Filtering after decoding would decode every proposal the run stored on every
    redraw — the cost the threshold is supposed to avoid."""
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    record = api_tools.run_tool(
        "quick_segment", _varied_config(video, n_frames=4, classes=("mouse",))
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    columns = quick_segment.load_detections(run_dir / "detections.zarr")

    decodes = []
    real = quick_segment.decode_mask
    monkeypatch.setattr(
        quick_segment,
        "decode_mask",
        lambda rle, size: decodes.append(rle) or real(rle, size),
    )
    quick_segment._frame_detections(columns, 0, (H, W), 0.5)
    assert (
        len(decodes) == 1
    )  # only the 0.9 row; the 0.2 row was skipped before decoding


def test_sampled_frame_returns_columns_so_a_redraw_costs_no_decode(
    tmp_path, varied_model, isolated_tools
):
    """The gather/draw split: one video decode, then any number of thresholds."""
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=16)
    record = api_tools.run_tool(
        "quick_segment",
        _varied_config(video, n_frames=3, mode="sample", classes=("mouse",)),
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])

    frame_rgb, per_label, frame_idx = quick_segment.sampled_frame(
        run_dir, [str(video)], 0
    )
    assert frame_rgb is not None and frame_rgb.ndim == 3
    assert set(per_label) == {"prompt_0"}
    # Raw columns, not decoded Detections — that is what keeps the cache small.
    assert "mask_rle" in per_label["prompt_0"]

    colors = quick_segment.sampled_prompt_colors(run_dir)
    loose = quick_segment.draw_overlays(frame_rgb, per_label, frame_idx, colors, 0.0)
    strict = quick_segment.draw_overlays(frame_rgb, per_label, frame_idx, colors, 0.5)
    assert not np.array_equal(loose["prompt_0"], strict["prompt_0"])


def test_sampled_confidences_expose_every_stored_score(
    tmp_path, varied_model, isolated_tools
):
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=16)
    record = api_tools.run_tool(
        "quick_segment",
        _varied_config(video, n_frames=3, mode="sample", classes=("mouse",)),
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    per_prompt = quick_segment.sampled_confidences(run_dir)
    assert set(per_prompt) == {"prompt_0"}
    scores = per_prompt["prompt_0"]
    assert scores.size == 6  # 3 sampled frames × 2 proposals
    assert int((scores >= 0.5).sum()) == 3


def test_export_frames_bakes_the_threshold_it_was_given(
    tmp_path, varied_model, isolated_tools
):
    """A JPG cannot be re-thresholded, so the export has to carry the one on screen."""
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=16)
    record = api_tools.run_tool(
        "quick_segment",
        _varied_config(video, n_frames=2, mode="sample", classes=("mouse",)),
    )
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    loose_dir, strict_dir = tmp_path / "loose", tmp_path / "strict"
    quick_segment.export_frames(run_dir, record, loose_dir, None, 0.0)
    quick_segment.export_frames(run_dir, record, strict_dir, None, 0.5)

    loose = sorted(loose_dir.glob("*.jpg"))
    strict = sorted(strict_dir.glob("*.jpg"))
    assert loose and len(loose) == len(strict)
    a = cv2.imread(str(loose[0]))
    b = cv2.imread(str(strict[0]))
    assert not np.array_equal(
        a, b
    )  # the weak proposal is drawn in one and not the other


def test_cli_min_conf_filters_the_report_without_changing_the_run(
    tmp_path, varied_model, isolated_tools
):
    """The CLI's twin of the GUI slider. It must move what is *counted*, never what is
    stored — the stored total is the same run either way."""
    from click.testing import CliRunner

    from detail import cli

    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=4)
    runner = CliRunner()

    def run(min_conf):
        result = runner.invoke(
            cli.main,
            [
                "tools",
                "quick-segment",
                str(video),
                "--prompt",
                "mouse",
                "--consecutive",
                "--frames",
                "4",
                "--model",
                "stub_varied",
                "--min-conf",
                str(min_conf),
            ],
        )
        assert result.exit_code == 0, result.output
        return result.output

    loose = run(0.0)
    strict = run(0.5)
    assert "8 of 8 at --min-conf 0.0" in loose
    assert "4 of 8 at --min-conf 0.5" in strict
    # Both runs stored the same thing: the threshold is a view, not a filter on the run.
    for record in api_tools.list_tool_runs():
        run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
        columns = quick_segment.load_detections(run_dir / "detections.zarr")
        assert len(columns["frame_idx"]) == 8
