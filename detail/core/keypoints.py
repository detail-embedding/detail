"""Keypoint job, progress and result dataclasses (§26), kept out of ``core/dataclasses.py``.

Five PRs add dataclasses for the keypoint stages (§26 *Sequencing*); in ``core/dataclasses.py``
they would all anchor after ``EmbeddingJob`` and ``EmbeddingProgressEvent`` and conflict on the
same lines, so they live here. ``FrameResult`` (§5 "Standardized output") stays in
``core/dataclasses.py`` beside ``Detection``: it is the shared container every model
returns, not a keypoint type. Each PR fills its own block below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detail.core.dataclasses import ComputeTarget


# -- keypoints (§26): PR 3 -- what a KeypointModel is handed, and what its scores mean ---- #
@dataclass
class InstanceFrame:
    """One decoded frame with every detection's box (and mask) in it (§26 "KeypointModel
    protocol").

    A per-detection method places one instance per entry of ``boxes``; a whole-frame method is
    handed ``{}`` and returns every animal it finds. Neither assigns a subject: identity is
    tracking's, given by ``keypoint_track``'s assignment step. Keys are the detection's **row
    index within its detection part**, which the method writes back as ``detection_row``.
    ``masks`` is filled only for a model that declares ``needs_mask``, at the detection run's
    mask resolution.
    """

    frame_idx: int
    t_ns: int
    frame: np.ndarray  # RGB uint8 (H, W, 3), the full decoded frame
    boxes: dict[
        int, tuple[float, float, float, float]
    ]  # detection_row -> normalized xyxy
    masks: dict[int, np.ndarray] | None = None  # detection_row -> (h, w) bool
    # detection_row -> the detection's class, filled only for a method matching each class
    # against its own example set (§26 *One example set per detection class*)
    classes: dict[int, str] | None = None


def student_input_of(record: dict) -> dict:
    """``{student_input, crop_px}`` a student is trained and run with, from a stored record:
    a model's ``model.yaml``, a pose dataset's ``dataset.yaml`` or a dataset job.

    The one reader of the choice. A record written before the choice existed is a whole-frame
    one, unless it carries ``crop_px`` and no ``student_input``: the stage before §26 trained
    on a crop and recorded only its size. Defaulting that record to the whole frame trains or
    runs a crop-trained student on whole frames, which is wrong without failing.
    """
    mode = record.get("student_input")
    crop = record.get("student_crop_px") or record.get("crop_px")
    if mode is None:
        mode = "crop" if record.get("crop_px") else "whole_frame"
    return {"student_input": str(mode), "crop_px": int(crop or 448)}


@dataclass(frozen=True)
class ScoreSpec:
    """What a method's point score means — its own, never comparable across methods."""

    meaning: str  # "cosine similarity" | "heatmap peak" | "detector confidence"
    low: float  # the range a score can take
    high: float
    default_cutoff: (
        float | None
    )  # below this a point reads as not visible; None = none set


# -- keypoints (§26): PR 4 -- the detect stage's job, and both stages' progress event ---- #
@dataclass
class KeypointDetectJob:
    """One ``keypoint_detect`` run on one recording (§26 "Job dataclasses"). Carries a full
    copy of what the run learned from, so it never depends on the example set's later edits.
    """

    job_id: str
    recording_id: str
    detect_run_id: str
    # the detections a per-detection method crops; the cadence and lineage for every method
    detection_run_id: str
    # the one run-time threshold (§26 *Visibility*): 0.25 for a method that reports_confidence,
    # None for one that does not
    min_pose_confidence: float | None
    model_id: str
    model_variant: str | None
    backbone: (
        dict | None
    )  # {"id": "dinov3_vit", "variant": "vit_large"}; None: closed-set
    example_set: (
        dict | None
    )  # {"id", "digest", "frames": [...], "marks": [...]}; None: none
    skeleton: dict  # the copy the arrays follow
    n_candidates: int
    every_n_frames: int
    crop_px: int | None
    decode: str | None
    side_rule: str | None
    matching: dict | None  # the method's constants, as resolved
    batch_size: int
    compute: ComputeTarget
    created_at: str
    # The library skeleton the example set was made from ({id, name, digest}), or None for a
    # set made before the library or a method that takes no set. Provenance, not identity:
    # the skeleton digest already is; this is what the library's dependents search reads.
    skeleton_ref: dict | None = None
    # Where a student's weights live (§12 *Keypoint models in the catalog*): "trained" (this
    # experiment's models/) or "imported" (the machine catalog); None for a registered method.
    # The worker builds from it rather than searching again, since the two can share an id.
    weights_source: str | None = None
    # The detection class a per-detection method places poses in (§26); None: every class.
    # Defaulted, so a job.json written before the field reads as every class, as it ran.
    class_name: str | None = None
    # One example-set snapshot per detection class ({class: {id, name, digest, frames,
    # marks}}, §26 *One example set per detection class*), in place of ``example_set`` and
    # ``class_name``. Defaulted, so a job.json written before the field reads as one set.
    example_sets: dict | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "recording_id": self.recording_id,
            "detect_run_id": self.detect_run_id,
            "detection_run_id": self.detection_run_id,
            "class_name": self.class_name,
            "min_pose_confidence": (
                None
                if self.min_pose_confidence is None
                else float(self.min_pose_confidence)
            ),
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "backbone": None if self.backbone is None else dict(self.backbone),
            "example_set": None if self.example_set is None else dict(self.example_set),
            "example_sets": (
                None
                if self.example_sets is None
                else {cls: dict(s) for cls, s in self.example_sets.items()}
            ),
            "skeleton": dict(self.skeleton),
            "n_candidates": int(self.n_candidates),
            "every_n_frames": int(self.every_n_frames),
            "crop_px": self.crop_px,
            "decode": self.decode,
            "side_rule": self.side_rule,
            "matching": None if self.matching is None else dict(self.matching),
            "batch_size": int(self.batch_size),
            "compute": self.compute.to_dict(),
            "created_at": self.created_at,
            "skeleton_ref": (
                None if self.skeleton_ref is None else dict(self.skeleton_ref)
            ),
            "weights_source": self.weights_source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeypointDetectJob":
        d = dict(d)
        if "detection_run_id" not in d:
            raise ValueError(
                f"keypoint detect job {d.get('job_id')!r} ran on a tracking run (the per-track "
                "layout before 2026-10-21); re-run keypoint detection"
            )
        d["compute"] = ComputeTarget.from_dict(d["compute"])
        return cls(**d)


@dataclass
class KeypointProgressEvent:
    """Both keypoint stages' progress (§6's contract): frames across the run, the phase as a
    message that advances at each boundary (``counting frames`` → ``loading model`` →
    ``placing keypoints``; the track stage's ``assigning`` → ``decoding`` → ``filtering``).
    ``track_id`` is None in the detect stage, which has no subjects."""

    track_id: int | None
    parts_done: int
    parts_total: int
    frames_done: int
    frames_total: int
    elapsed_s: float
    eta_s: float | None
    message: str | None = None


# -- keypoints (§26): PR 5 -- the track stage's job and a method's output for one part ---- #
@dataclass
class KeypointTrackJob:
    """One ``keypoint_track`` run on one recording: a detect run settled over time by one
    method whose every setting is resolved and carried (§26 "Job dataclasses")."""

    job_id: str
    recording_id: str
    track_run_id: str
    detect_run_id: str
    tracking_run_id: str  # on the detect run's detection run (§26 *Assignment*)
    match: dict  # {"min_inside": 0.5, "mask_grow": 0.05}
    method: str
    config: dict  # the method's config, every field resolved
    skeleton: dict
    created_at: str
    # Carried on from the detect run's record (§26 *The Keypoints manager*): the library
    # skeleton the points follow, read by the skeleton dependents search. None when the
    # detect run had none.
    skeleton_ref: dict | None = None
    # Where it runs: this machine's CPU, or a registered cluster (§6). None for a job written
    # before the stage could go remote, which ran here.
    compute: ComputeTarget | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "recording_id": self.recording_id,
            "track_run_id": self.track_run_id,
            "detect_run_id": self.detect_run_id,
            "tracking_run_id": self.tracking_run_id,
            "match": dict(self.match),
            "method": self.method,
            "config": dict(self.config),
            "skeleton": dict(self.skeleton),
            "created_at": self.created_at,
            "skeleton_ref": (
                None if self.skeleton_ref is None else dict(self.skeleton_ref)
            ),
            "compute": None if self.compute is None else self.compute.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeypointTrackJob":
        if "tracking_run_id" not in d:
            raise ValueError(
                f"keypoint track job {d.get('job_id')!r} was made from a per-track detect run "
                "(before 2026-10-21); re-run keypoint detection, then keypoint track"
            )
        compute = d.get("compute")
        return cls(
            **{
                **d,
                "compute": (
                    None if compute is None else ComputeTarget.from_dict(compute)
                ),
            }
        )


@dataclass
class TrackPart:
    """What a track method returns for one (track, part): arrays in, arrays out (§26 "Track
    methods"); the stage reads and writes."""

    points: np.ndarray  # (M, K, 2) float32 normalized full-frame; NaN = absent
    point_scores: np.ndarray  # (M, K) float32 the chosen detection's score
    changed: (
        np.ndarray
    )  # (M, K) int8: 0 as detected · 1 another candidate · 2 moved by a filter


# -- keypoints (§26): PR 8 -- the label stage of a keypoint distill (§12 *Distill keypoints*) -- #
@dataclass
class KeypointDatasetJob:
    """One ``keypoint_dataset`` run: frames sampled and each written as a YOLO-pose example,
    labelled by one of two teachers, both **raw** (no smoothing over time; §12 *Distill
    keypoints*, the user's decision 2026-10-09). Every threshold is resolved before the job exists
    (Tenet 9).

    - ``teacher = {kind: "keypoint_detect", detect_runs: {recording_id: kd_NN}, tracking_runs,
      match}``: finished keypoint detect runs, whose stored instances are the labels.
    - ``teacher = {kind: "model", model_id, model_variant, settings, example_set, skeleton,
      tracking_runs, match}``: the patch method runs on the sampled frames' detections, built
      from the run's own copy of the example set (``example_set`` is the snapshot keypoint
      detect records). ``compute`` is where it runs.

    Either way the instances are unassigned until §26 *Assignment*'s function gives them to the
    subjects of ``tracking_runs[recording_id]`` with ``match`` (``{min_inside, mask_grow}``;
    a job written before it was recorded takes the defaults).

    An old job may carry ``track_runs`` (the retired keypoint-track teacher); it is read as
    ``kind: keypoint_track`` so its record still opens, and is never run again.
    """

    job_id: str
    dataset_id: str
    name: str
    teacher: dict
    n_frames: int
    val_split: float
    score_cutoff: float  # resolved: below it a point is taught as not visible
    exclude_audit_s: float
    every_n_frames: int
    created_at: str
    compute: ComputeTarget | None = None
    # What each training image is (§12): the whole frame (the default, and every dataset
    # written before the choice existed), or a crop around each tracked animal.
    student_input: str = "whole_frame"
    student_crop_px: int = 448

    @property
    def teacher_kind(self) -> str:
        return str(self.teacher.get("kind"))

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "teacher": dict(self.teacher),
            "compute": None if self.compute is None else self.compute.to_dict(),
            "n_frames": int(self.n_frames),
            "val_split": float(self.val_split),
            "score_cutoff": float(self.score_cutoff),
            "exclude_audit_s": float(self.exclude_audit_s),
            "every_n_frames": int(self.every_n_frames),
            "created_at": self.created_at,
            "student_input": self.student_input,
            "student_crop_px": int(self.student_crop_px),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeypointDatasetJob":
        teacher = dict(d.get("teacher") or {})
        if not teacher:  # a job written before the teacher field: the track-run teacher
            teacher = {
                "kind": "keypoint_track",
                "track_runs": {
                    str(k): str(v) for k, v in (d.get("track_runs") or {}).items()
                },
            }
        return cls(
            job_id=d["job_id"],
            dataset_id=d["dataset_id"],
            name=d["name"],
            teacher=teacher,
            n_frames=int(d["n_frames"]),
            val_split=float(d["val_split"]),
            score_cutoff=float(d["score_cutoff"]),
            exclude_audit_s=float(d["exclude_audit_s"]),
            every_n_frames=int(d.get("every_n_frames", 1)),
            created_at=d["created_at"],
            compute=(
                ComputeTarget.from_dict(d["compute"]) if d.get("compute") else None
            ),
            student_input=student_input_of(d)["student_input"],
            student_crop_px=student_input_of(d)["crop_px"],
        )
