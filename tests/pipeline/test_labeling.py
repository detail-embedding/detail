"""Fresh labeling dataset source (§12): sample frames + label with a prompt-capable model.

**The class: a dataset's split is drawn over the examples written, never over the frames
asked for.** A source chooses which frames to try before it knows which yield an example — the
teacher can find nothing, a frame can fail to decode exactly, a stop can cut the run short.
Every dataset source used to split the *requested* frames first and then drop some, so each
drop took its train/val slot with it and the realised validation share landed wherever the
drops fell (in ``test_realised_val_share_is_the_configured_one``, 5 validation frames where
0.25 of the frames kept is 4). The fix is one mechanism, ``_dataset_common.StagedExamples``:
examples are staged and the split is drawn when the source commits. Labeling, run export and
the keypoint dataset all write through it; ``tests/pipeline/test_keypoint_dataset.py`` guards
the keypoint path.

**And the frame under the index is the frame at the index.** The labeling sampler seeked with
cv2's ``CAP_PROP_POS_FRAMES``, which converts the index to a time with the advertised fps and
can hand back a neighbouring frame while reporting the requested index. It now decodes each
frame through ``read_frame_at`` with the part's timestamp map and redraws a frame it cannot
land on (``test_frames_are_decoded_exactly_and_a_miss_is_redrawn``).

**Background frames** — where the teacher found none of the classes — are kept with an empty
label up to ``background_share`` of the dataset; a frame where it found something too small to
keep is not background, since an animal may be there.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from detail.api.experiment import Experiment
from detail.configs.labeling import with_negative_prompt
from detail.core.dataclasses import (
    ComputeTarget,
    Detection,
    FrameResult,
    LabelingClass,
    LabelingJob,
    TextCue,
)
from detail.core.protocols import DetectionModel
from detail.core.registries import MODELS
from detail.pipeline import labeling
from detail.storage import experiment as experiment_store
from detail.storage import recording as recording_store

H, W = 24, 32  # video is (W, H)


class _FakeSAM3(DetectionModel):
    """A deterministic stand-in for gated SAM3 — one centered mask per prompted class."""

    model_id = "sam3"
    supported_cue_types = [TextCue]
    supported_backends = ["cpu", "cuda"]

    def load(self, device):
        self.device = device

    def set_conf_threshold(self, threshold):
        self.threshold = threshold

    def predict(self, frame, prompt):
        h, w = frame.shape[:2]
        dets = []
        for target in prompt.targets:
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
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


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    assert vw.isOpened()
    for f in range(n_frames):
        vw.write(np.full((H, W, 3), (f * 7) % 250 + 1, np.uint8))
    vw.release()


def _video_part_with_timestamps(recording_dir, n_frames: int) -> dict:
    """Write a video part + its generated timestamp CSV (the authoritative frame count the
    labeling runner reads) and return the recording.yaml files entry pointing at both.
    """
    _write_video(recording_dir / "video" / "part_001.mp4", n_frames)
    ts_csv = recording_dir / "video" / "timestamps" / "generated" / "part_001.csv"
    ts_csv.parent.mkdir(parents=True, exist_ok=True)
    recording_store.write_timestamps_csv(
        ts_csv, np.arange(n_frames, dtype=np.int64) * 1000
    )
    return {
        "index": 0,
        "file": "video/part_001.mp4",
        "timestamps": {
            "generated": "video/timestamps/generated/part_001.csv",
            "user": None,
            "active": "generated",
        },
    }


def _add_video_recording(proj_dir, sid: str, n_frames: int) -> None:
    recording_store.create_recording(proj_dir, sid)
    recording_dir = experiment_store.recording_path(proj_dir, sid)
    data = recording_store.read_recording(recording_dir)
    data["video"] = {"files": [_video_part_with_timestamps(recording_dir, n_frames)]}
    recording_store.write_recording(recording_dir, data)


def _setup(tmp_path, counts=(20, 60)):
    """A experiment with one video recording per entry in ``counts`` (no detection run needed)."""
    proj_dir = experiment_store.create_experiment(tmp_path, "p")
    for i, n in enumerate(counts, start=1):
        _add_video_recording(proj_dir, f"recording_{i:03d}", n)
    return proj_dir


def _job(**kw) -> LabelingJob:
    base = dict(
        job_id="lbl_01",
        dataset_id="dataset_001",
        name="mouse_set",
        source_recording_ids=["recording_001", "recording_002"],
        classes=[LabelingClass(class_id=0, name="mouse", prompt="a mouse")],
        n_frames=16,
        logit_threshold=0.5,
        min_contour_area=1,
        val_split=0.0,
        compute=ComputeTarget("cpu", None, None, None),
        created_at="2026-06-09T00:00:00Z",
    )
    base.update(kw)
    return LabelingJob(**base)


def _run(proj_dir, job, progress_cb=lambda e: None):
    run_dir = labeling.run_dir_for(proj_dir, job)
    run_dir.mkdir(parents=True)
    labeling.run(job, run_dir, progress_cb=progress_cb, stop_event=None)
    return run_dir


# --------------------------------------------------------------------------- #
# Runner (in-process, fake SAM3)
# --------------------------------------------------------------------------- #
def test_writes_yolo_seg_images_and_labels(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    run_dir = _run(_setup(tmp_path), _job(n_frames=16, val_split=0.0))

    imgs = sorted((run_dir / "images" / "train").glob("*.jpg"))
    lbls = sorted((run_dir / "labels" / "train").glob("*.txt"))
    assert imgs and len(imgs) == len(lbls) == 16
    tokens = lbls[0].read_text(encoding="utf-8-sig").strip().split("\n")[0].split()
    assert tokens[0] == "0"  # class_id
    assert (len(tokens) - 1) % 2 == 0 and (len(tokens) - 1) >= 6  # a normalized polygon
    assert all(0.0 <= float(v) <= 1.0 for v in tokens[1:])

    rec = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    assert rec["status"] == "complete"
    assert rec["source"]["kind"] == "labeling"
    assert rec["source"]["model"] == "sam3"  # the default model is recorded
    assert rec["classes"] == [{"id": 0, "name": "mouse"}]
    assert rec["n_train"] == 16 and rec["n_val"] == 0
    # data.yaml (trainer-facing) is written with the class names
    data_yaml = yaml.safe_load((run_dir / "data.yaml").read_text(encoding="utf-8-sig"))
    assert data_yaml["names"] == {0: "mouse"}


def test_labels_with_a_non_sam3_model(tmp_path, monkeypatch):
    """The runner labels with whatever prompt-capable model the job names, not a hardcoded SAM3."""

    class _FakeBatchModel(_FakeSAM3):
        model_id = "fake_yolo"
        supports_batching = True

    monkeypatch.setitem(MODELS, "fake_yolo", _FakeBatchModel)
    run_dir = _run(
        _setup(tmp_path), _job(n_frames=16, val_split=0.0, model_id="fake_yolo")
    )

    assert sorted((run_dir / "images" / "train").glob("*.jpg"))  # frames were labeled
    rec = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    assert rec["source"]["model"] == "fake_yolo"  # the chosen model is recorded


def test_sampling_is_proportional_to_recording_length(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    run_dir = _run(_setup(tmp_path, counts=(20, 60)), _job(n_frames=16, val_split=0.0))
    stems = [p.stem for p in (run_dir / "images" / "train").glob("*.jpg")]
    s1 = sum(s.startswith("recording_001") for s in stems)
    s2 = sum(s.startswith("recording_002") for s in stems)
    assert (s1, s2) == (4, 12)  # 16 * 20/80 and 16 * 60/80


def test_stratified_sampling_covers_each_time_bin():
    # Stratified draws one frame per equal-width temporal bin: 10 picks over 100 frames must land
    # exactly one per decile. Uniform-random (the old behavior) clumps and leaves gaps — reverting
    # _draw_indices to rng.sample makes this fail.
    parts = [{"recording_id": "s1", "frame_count": 100}]
    job = _job(n_frames=10, sampling="stratified", source_recording_ids=["s1"])
    idxs = labeling._sample_frames(parts, job)[0]
    assert len(idxs) == 10
    assert idxs == sorted(idxs)
    for i in range(10):
        in_bin = [x for x in idxs if i * 10 <= x < (i + 1) * 10]
        assert len(in_bin) == 1, f"decile {i} got {in_bin}"


def test_uniform_sampling_returns_k_distinct_in_range():
    parts = [{"recording_id": "s1", "frame_count": 100}]
    job = _job(n_frames=10, sampling="uniform", source_recording_ids=["s1"])
    idxs = labeling._sample_frames(parts, job)[0]
    assert len(idxs) == len(set(idxs)) == 10
    assert all(0 <= x < 100 for x in idxs)


def test_val_split_is_stratified_per_recording(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    run_dir = _run(_setup(tmp_path, counts=(20, 60)), _job(n_frames=16, val_split=0.5))
    val = [p.stem for p in (run_dir / "images" / "val").glob("*.jpg")]
    assert any(
        s.startswith("recording_001") for s in val
    )  # no recording absent from val
    assert any(s.startswith("recording_002") for s in val)


def test_multi_class_writes_a_line_per_class(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    classes = [
        LabelingClass(class_id=0, name="mouse", prompt="a mouse"),
        LabelingClass(class_id=1, name="pellet", prompt="a food pellet"),
    ]
    run_dir = _run(_setup(tmp_path), _job(classes=classes, n_frames=8, val_split=0.0))
    lbl = (
        sorted((run_dir / "labels" / "train").glob("*.txt"))[0]
        .read_text(encoding="utf-8-sig")
        .strip()
    )
    class_ids = {line.split()[0] for line in lbl.split("\n")}
    assert class_ids == {"0", "1"}  # both classes labeled in the frame


def test_reproducible_given_dataset_id(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    a = _run(_setup(tmp_path / "a"), _job(n_frames=12, val_split=0.0))
    b = _run(_setup(tmp_path / "b"), _job(n_frames=12, val_split=0.0))
    names = lambda d: sorted(p.name for p in (d / "images" / "train").glob("*.jpg"))
    assert names(a) == names(b)  # same dataset_id → same sampled frames


def test_progress_is_incremental(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    monkeypatch.setattr("detail.core.progress.PROGRESS_INTERVAL_S", 0.0)
    events = []
    _run(_setup(tmp_path), _job(n_frames=16, val_split=0.0), progress_cb=events.append)
    assert any(0 < e.frames_done < e.frames_total for e in events)  # not just terminal
    # #18: the runner populates eta_s (not None) on a mid-run event.
    assert any(
        e.eta_s is not None for e in events if 0 < e.frames_done < e.frames_total
    )
    # #18: the (slow) model load is announced, not silent.
    assert any(e.message == "loading model" for e in events)


# --------------------------------------------------------------------------- #
# Exact sampler, background frames, split after labelling
# --------------------------------------------------------------------------- #
class _FindsNothingOnDarkFrames(_FakeSAM3):
    """Finds the class only on bright frames; a dark frame (``_write_video`` paints frame f
    with ``(f * 7) % 250 + 1``) has nothing on it — a background frame."""

    def predict(self, frame, prompt):
        if frame.mean() < 128:
            return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=[])
        return super().predict(frame, prompt)


def _dataset_record(run_dir) -> dict:
    return yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))


def _label_files(run_dir) -> list[Path]:
    return sorted((run_dir / "labels").glob("*/*.txt"))


def test_background_frames_are_kept_up_to_the_share(tmp_path, monkeypatch):
    """Frames where the teacher found nothing become empty-label examples, at most
    ``background_share`` of the dataset — the rest of them are dropped, never mislabelled.
    """
    monkeypatch.setitem(MODELS, "sam3", _FindsNothingOnDarkFrames)
    run_dir = _run(
        _setup(tmp_path, counts=(20, 60)),
        _job(n_frames=40, val_split=0.0, background_share=0.1),
    )
    labels = _label_files(run_dir)
    empty = [p for p in labels if not p.read_text(encoding="utf-8-sig").strip()]
    labelled = len(labels) - len(empty)
    rec = _dataset_record(run_dir)
    # floor(0.1 * L / 0.9): the most that keeps background at or under 10% of the total.
    assert len(empty) == int(0.1 * labelled / 0.9) > 0
    assert len(empty) / len(labels) <= 0.1
    assert rec["n_background"] == len(empty)
    assert rec["background_share"] == 0.1
    assert rec["n_train"] == len(labels)
    # Every staged frame was moved into place or dropped: nothing is left beside images/.
    assert not (run_dir / "staging").exists()


def test_no_background_frames_at_share_zero(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FindsNothingOnDarkFrames)
    run_dir = _run(
        _setup(tmp_path), _job(n_frames=40, val_split=0.0, background_share=0.0)
    )
    assert all(p.read_text(encoding="utf-8-sig").strip() for p in _label_files(run_dir))
    assert _dataset_record(run_dir)["n_background"] == 0


def test_a_mask_too_small_to_keep_is_not_background(tmp_path, monkeypatch):
    """The teacher found something, but every mask fell under ``min_contour_area``: an animal
    may be there, so teaching the frame as empty would be teaching a wrong label."""

    class _TinyOnDarkFrames(_FakeSAM3):
        def predict(self, frame, prompt):
            result = super().predict(frame, prompt)
            if frame.mean() >= 128:
                return result
            tiny = []
            for det in result.detections():
                mask = np.zeros_like(det.mask)
                mask[:2, :2] = True
                tiny.append(
                    Detection(
                        mask=mask,
                        bbox=det.bbox,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        object_id=None,
                    )
                )
            return FrameResult.from_detections(frame_idx=0, t_ns=0, detections=tiny)

    monkeypatch.setitem(MODELS, "sam3", _TinyOnDarkFrames)
    run_dir = _run(
        _setup(tmp_path),
        _job(n_frames=40, val_split=0.0, background_share=0.5, min_contour_area=50),
    )
    rec = _dataset_record(run_dir)
    assert rec["n_background"] == 0
    assert rec["skipped"]["below_min_area"] > 0
    assert all(p.read_text(encoding="utf-8-sig").strip() for p in _label_files(run_dir))


def test_realised_val_share_is_the_configured_one(tmp_path, monkeypatch):
    """The split is drawn over the frames that became examples. Drawn over the sampled frames
    first (as it was), every frame the teacher left empty took its train/val slot with it,
    and the realised share landed wherever those drops fell."""
    monkeypatch.setitem(MODELS, "sam3", _FindsNothingOnDarkFrames)
    run_dir = _run(
        _setup(tmp_path, counts=(80,)),
        _job(
            n_frames=40,
            val_split=0.25,
            background_share=0.0,
            source_recording_ids=["recording_001"],
        ),
    )
    n_train = len(list((run_dir / "images" / "train").glob("*.jpg")))
    n_val = len(list((run_dir / "images" / "val").glob("*.jpg")))
    kept = n_train + n_val
    assert 0 < kept < 40  # some sampled frames made no example
    assert n_val == round(kept * 0.25)
    rec = _dataset_record(run_dir)
    assert (rec["n_train"], rec["n_val"]) == (n_train, n_val)


def test_frames_are_decoded_exactly_and_a_miss_is_redrawn(tmp_path, monkeypatch):
    """Each sampled frame goes through ``read_frame_at`` with the part's timestamp map. A frame
    the decoder lands beside is not written under the requested index; another frame from the
    same part replaces it, so the sample keeps its size."""
    from detail.pipeline import _dataset_common
    from detail.storage.video import read_frame_at

    calls: list[tuple[int, bool]] = []

    def _spy(video, frame_idx, *, timestamps=None):
        calls.append((frame_idx, timestamps is not None))
        decoded = read_frame_at(video, frame_idx, timestamps=timestamps)
        if len(calls) == 1:  # the first read lands on the neighbouring frame
            return frame_idx + 1, decoded[1]
        return decoded

    monkeypatch.setattr(_dataset_common, "read_frame_at", _spy)
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    run_dir = _run(
        _setup(tmp_path, counts=(60,)),
        _job(n_frames=10, val_split=0.0, source_recording_ids=["recording_001"]),
    )
    missed = calls[0][0]
    stems = {p.stem for p in (run_dir / "images" / "train").glob("*.jpg")}
    assert all(with_map for _, with_map in calls)  # always with the timestamp map
    assert not any(s.endswith(f"_{missed:07d}") for s in stems)
    assert len(stems) == 10  # the miss was redrawn
    assert _dataset_record(run_dir)["skipped"]["undecodable"] == 1


def test_exact_sampler_redraw_is_seeded_and_bounded(tmp_path, monkeypatch):
    import random

    from detail.pipeline import _dataset_common

    vid = tmp_path / "v.mp4"
    _write_video(vid, 60)
    timestamps = np.arange(60, dtype=np.int64) * 1000

    def _odd_fail(video, frame_idx, *, timestamps=None):
        return None if frame_idx % 2 else (frame_idx, np.zeros((H, W, 3), np.uint8))

    monkeypatch.setattr(_dataset_common, "read_frame_at", _odd_fail)
    draw = lambda: [
        i
        for i, _ in _dataset_common.exact_sampled_frames(
            vid, [1, 3, 10], timestamps, random.Random("seed")
        )
    ]
    first = draw()
    assert first == draw()  # same seed, same replacements
    assert 10 in first and all(i % 2 == 0 for i in first)

    # A file nothing can be decoded from ends, it does not redraw forever.
    monkeypatch.setattr(_dataset_common, "read_frame_at", lambda *a, **k: None)
    stats: dict = {}
    got = list(
        _dataset_common.exact_sampled_frames(
            vid, [1, 3, 10], timestamps, random.Random("seed"), stats
        )
    )
    assert got == [] and stats["undecodable"] == 6  # 3 targets + 3 redraws, then stop


def test_background_cap_keeps_background_at_or_under_the_share():
    from detail.pipeline._dataset_common import background_cap

    assert background_cap(90, 0.1) == 10  # 10 / 100
    assert background_cap(89, 0.1) == 9  # 10 would be 10.1%
    assert background_cap(100, 0.5) == 100
    assert background_cap(0, 0.5) == 0  # nothing labelled, nothing to set beside it
    assert background_cap(100, 0.0) == 0
    for labelled in range(1, 200):
        for share in (0.05, 0.1, 0.25, 0.5):
            b = background_cap(labelled, share)
            assert b / (labelled + b) <= share + 1e-12
            assert (b + 1) / (
                labelled + b + 1
            ) > share  # and no smaller than it must be


def test_sam3_runs_per_frame_not_batched(tmp_path, monkeypatch):
    """supports_batching=False (SAM3) → batch_size 1 → predict_batch gets one frame."""
    seen = []

    class _Recording(_FakeSAM3):
        def predict_batch(self, frames, prompts):
            seen.append(len(frames))
            return super().predict_batch(frames, prompts)

    monkeypatch.setitem(MODELS, "sam3", _Recording)
    _run(_setup(tmp_path), _job(n_frames=8, val_split=0.0))
    assert seen and set(seen) == {1}


def test_batch_capable_model_gets_real_batches(tmp_path, monkeypatch):
    """The batched path exists for prompt-capable models that support it, even though
    SAM3 itself does not."""
    seen = []

    class _Batchable(_FakeSAM3):
        supports_batching = True

        def predict_batch(self, frames, prompts):
            seen.append(len(frames))
            return super().predict_batch(frames, prompts)

    monkeypatch.setitem(MODELS, "sam3", _Batchable)
    _run(_setup(tmp_path, counts=(20, 60)), _job(n_frames=16, val_split=0.0))
    assert max(seen) > 1  # actually batched, not per-frame
    assert max(seen) <= labeling._BATCH_SIZE


# --------------------------------------------------------------------------- #
# Experiment.label_dataset validation (raises before the SAM3 readiness gate)
# --------------------------------------------------------------------------- #
def _api_experiment(tmp_path, counts=(20, 60)):
    proj = Experiment.create("p", tmp_path)
    for i, n in enumerate(counts, start=1):
        recording = proj.create_recording(f"recording_{i:03d}")
        data = recording_store.read_recording(recording.dir)
        data["video"] = {"files": [_video_part_with_timestamps(recording.dir, n)]}
        recording_store.write_recording(recording.dir, data)
    return proj


def test_label_dataset_rejects_too_many_frames(tmp_path):
    proj = _api_experiment(tmp_path, counts=(20, 60))
    with pytest.raises(ValueError, match="exceeds the 80 frames"):
        proj.label_dataset("d", [LabelingClass(0, "mouse", "a mouse")], n_frames=100)


def test_label_dataset_rejects_duplicate_prompts(tmp_path):
    proj = _api_experiment(tmp_path)
    classes = [LabelingClass(0, "a", "same"), LabelingClass(1, "b", "same")]
    with pytest.raises(ValueError, match="share a text prompt"):
        proj.label_dataset("d", classes, n_frames=4)


def test_label_dataset_rejects_non_prompt_capable_model(tmp_path):
    proj = _api_experiment(tmp_path)
    # a closed-set / unknown model can't take a text prompt, so it can't label
    with pytest.raises(ValueError, match="cannot label"):
        proj.label_dataset(
            "d", [LabelingClass(0, "m", "m")], n_frames=4, model_id="trained_yolo"
        )


def test_label_dataset_rejects_bad_val_split(tmp_path):
    proj = _api_experiment(tmp_path)
    # val_split bounds now live on LabelingConfig — a pydantic ValidationError (a ValueError).
    with pytest.raises(ValueError, match="less than 1"):
        proj.label_dataset("d", [LabelingClass(0, "m", "m")], n_frames=4, val_split=1.0)


def test_label_dataset_rejects_bad_sampling(tmp_path):
    proj = _api_experiment(tmp_path)
    with pytest.raises(ValueError, match="sampling"):
        proj.label_dataset(
            "d", [LabelingClass(0, "m", "m")], n_frames=4, sampling="middle-out"
        )


def test_label_dataset_rejects_recording_without_video(tmp_path):
    """A recording *named* is taken as named, so one with no video is refused — by the name
    the user gave it, not the id the tool minted (#35)."""
    proj = _api_experiment(tmp_path, counts=(20,))
    bare = proj.add_recording("m27 day3")  # no video imported
    with pytest.raises(ValueError, match="no video") as raised:
        proj.label_dataset(
            "d",
            [LabelingClass(0, "m", "m")],
            n_frames=4,
            source_recording_ids=[bare.id],
        )
    assert "'m27 day3'" in str(raised.value)
    assert bare.id not in str(raised.value)


def test_label_dataset_with_none_named_samples_every_recording_with_video(tmp_path):
    """#35 finding 5. "None given" meant every recording, so one metadata-only recording —
    which Add recordings allows — refused the whole job, and the GUI's "none = all" could not
    be used. Now it means every recording with video: the frame check below counts 20 + 60
    from the two that have it, and the bare one is not reached."""
    proj = _api_experiment(tmp_path, counts=(20, 60))
    proj.add_recording("metadata only")
    with pytest.raises(ValueError, match="exceeds the 80 frames"):
        proj.label_dataset("d", [LabelingClass(0, "m", "m")], n_frames=100)


def test_sample_size_is_exact(tmp_path, monkeypatch):
    # The timestamp map is the authoritative frame count, so the sample is exactly n_frames —
    # no cv2 over-estimate leaving sampled indices past the true end unyielded (closes review A8).
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    run_dir = _run(_setup(tmp_path, counts=(20, 60)), _job(n_frames=25, val_split=0.0))
    imgs = list((run_dir / "images" / "train").glob("*.jpg"))
    assert len(imgs) == 25  # exactly what was requested
    rec = yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8-sig"))
    assert rec["n_train"] == 25 and rec["status"] == "complete"


def test_a_negative_prompt_is_labelled_and_kept_out_of_the_classes(
    tmp_path, monkeypatch
):
    """§12 *Negative prompt*: the trainer's ``data.yaml`` names both classes so the student
    learns the cable apart, and ``dataset.yaml`` lists it under ``negative_classes`` only, so
    a model trained on it outputs the user's classes."""
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    classes, negatives = with_negative_prompt(
        [LabelingClass(class_id=0, name="mouse", prompt="a mouse")], "cable"
    )
    run_dir = _run(
        _setup(tmp_path),
        _job(classes=classes, negative_classes=negatives, n_frames=8, val_split=0.0),
    )
    label_ids = {
        line.split()[0]
        for path in _label_files(run_dir)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
    }
    assert label_ids == {"0", "1"}  # the cable is labelled as its own class
    data = yaml.safe_load((run_dir / "data.yaml").read_text(encoding="utf-8-sig"))
    assert data["names"] == {0: "mouse", 1: "cable"}
    record = _dataset_record(run_dir)
    assert record["classes"] == [{"id": 0, "name": "mouse"}]
    assert record["negative_classes"] == [{"id": 1, "name": "cable"}]
    assert [c["name"] for c in record["source"]["classes"]] == ["mouse"]
    assert record["source"]["negative_classes"] == [
        {"class_id": 1, "name": "cable", "prompt": "cable"}
    ]
    assert set(record["per_class"]) == {"0", "1"}


def test_a_dataset_without_a_negative_prompt_records_none(tmp_path, monkeypatch):
    monkeypatch.setitem(MODELS, "sam3", _FakeSAM3)
    record = _dataset_record(_run(_setup(tmp_path), _job(n_frames=4, val_split=0.0)))
    assert record["negative_classes"] == []
    assert record["source"]["negative_classes"] == []


def test_label_dataset_refuses_a_negative_prompt_that_is_a_class(tmp_path):
    proj = _api_experiment(tmp_path)
    with pytest.raises(
        ValueError, match="negative prompt 'A Mouse' is also class 'mouse'"
    ):
        proj.label_dataset(
            "d",
            [LabelingClass(0, "mouse", "a mouse")],
            n_frames=4,
            negative_prompt="A Mouse",
        )
