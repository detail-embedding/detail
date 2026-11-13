"""Modules the executor worker imports so its runners and model backends self-register.

Shared between ``Experiment`` (so ``check_model_access`` / ``install_model`` see every
first-party model) and ``Recording`` (so the worker subprocess registers the right
runners). Backend imports inside these modules are lazy (Tenet 13), so listing a
module here is safe even when its backend is not installed.

A build that wants to exclude a backend drops it from this tuple (or overrides via
``DETAIL_RUNNER_MODULES``); a config-driven registry is the cleanup that the
"Decision to revisit" note in DESIGN_PLAN tracks.
"""

PIPELINE_RUNNERS: tuple[str, ...] = (
    "detail.pipeline.tracking",
    "detail.pipeline.track_merge",
    "detail.pipeline.detection",
    "detail.pipeline.embedding",
    "detail.pipeline.keypoint_detect",
    "detail.pipeline.keypoint_track",  # also registers the two track methods
    "detail.pipeline.keypoint_dataset",  # the label stage of a keypoint distill (§12)
    "detail.pipeline.dataset_export",
    "detail.pipeline.export_run",
    "detail.pipeline.labeling",
    "detail.pipeline.training",
    "detail.pipeline.location_analysis",
    "detail.pipeline.query",
    "detail.pipeline.query_eval",
    "detail.pipeline.temporal_clustering",
    "detail.pipeline.profile_describe",
    "detail.pipeline.cluster_lookalikes",
    "detail.trainers.ultralytics_yolo",
    "detail.models.sam3",
    "detail.models.trained_yolo",
    "detail.models.dinov3",
    "detail.models.patch_match",  # dinov3_match (§26); its package import registers it
    "detail.models.superanimal",  # the two DeepLabCut SuperAnimal methods (§26)
    "detail.models.yolo_pose",  # the distilled keypoint student (§12), resolved by record
)
