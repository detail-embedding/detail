"""Embedding config model (§14) — the single source for its scalar parameter (batch size) and the
typed shape of an embedding run. The nested compute target, the resolved renderer_spec, and the
(experiment-populated, all_tracks-resolved) track_ids stay outside the model — ``EmbeddingModule.validate``
resolves / checks them and returns the enriched config; the model only enforces the modelled scalars
and carries the metadata each surface (GUI tooltip, CLI help, docs) reads.

A ``track_ids`` entry may be an int (a track's raw position) or a string (an animal id,
resolved per recording through that recording's own ``subject_labels`` — §13) —
``core.utils.resolve_track_id`` is the resolver and ``validate`` writes the resolved
integers back, the same way it resolves ``all_tracks``."""

from typing import Literal

from pydantic import BaseModel, Field

from detail.transforms.constructors import DEFAULT_CONSTRUCTOR

# The centered canvas is derived from the data, and resolved either from each recording alone
# or once across the whole selection. Per recording normalizes it away (every animal's own
# largest mask becomes the canvas, so body-size differences vanish); shared keeps the
# recordings on one absolute scale at the cost of resolution for the smaller subjects (§14).
# The max-contrast background has no such choice: a multi-recording run always resolves one
# colour for the whole selection, because a colour that differs per recording is a cue that
# identifies the recording, and the encoder sees it in every frame.
DerivedScope = Literal["per_recording", "shared"]
# §14 pose in the embedding: how a keypoint track run's points enter it (EmbeddingConfig.pose).
PoseMode = Literal["none", "draw", "append"]

#: The three ways a keypoint track run's pose can enter an embedding (``EmbeddingConfig.pose``):
#: a short label and one plain-language explanation each. The one wording every surface that
#: offers the choice shows (the launch form, ``--help``, the docs), so they cannot drift apart.
POSE_CHOICES: dict[str, tuple[str, str]] = {
    "none": (
        "Mask only",
        "The model sees the mask's shape and nothing else.",
    ),
    "draw": (
        "Points drawn on the mask",
        "The body points (coloured by side, joined by the skeleton's lines) are painted onto "
        "the image the model sees, so a single vector describes the mask and the pose together "
        "as the model reads them. The vector keeps its usual size.",
    ),
    "append": (
        "Pose numbers added to the vector",
        "The pose is written as numbers (each point's place relative to the body's centre, "
        "turned so the heading points right, in body lengths) and added after the mask's "
        "vector. Mask and pose stay two separate parts of one vector, weighed against each "
        "other by the pose weight, and the vector is that much longer.",
    ),
}


class EmbeddingConfig(BaseModel):
    tracking_run_id: str = Field(
        ...,
        description="Tracking run whose tracks to embed (e.g. track_01).",
        json_schema_extra={"label": "Tracking run"},
    )
    all_tracks: bool = Field(
        False,
        description="Embed every track in the run instead of a specific track list.",
        json_schema_extra={"label": "All tracks"},
    )
    model_id: str = Field(
        "dinov3_vit",
        description="Embedding model to run (a registered model id). The default is DINOv3 "
        "ViT (dinov3_vit).",
        json_schema_extra={"label": "Model"},
    )
    model_variant: str | None = Field(
        None,
        description="Model size/variant (e.g. vit_base); omit for the model's default "
        "size (ViT-L for DINOv3 ViT).",
        json_schema_extra={"label": "Size / variant"},
    )
    readout: str | None = Field(
        None,
        description="Feature read-out: how the backbone output becomes one vector "
        "(model-dependent, e.g. pooler / cls_patch / gap / gmp / gap_gmp). Omit for the "
        "model's default.",
        json_schema_extra={"advanced": True, "label": "Read-out"},
    )
    constructor: str | None = Field(
        DEFAULT_CONSTRUCTOR,
        description="Named input-image constructor. The centered ones crop each mask and center it "
        "at true scale in a shared canvas. Ignored when a renderer_spec is given directly.",
        json_schema_extra={"label": "Input image"},
    )
    input_size: int | None = Field(
        None,
        ge=1,
        description="Model input resolution (px), the square the model actually sees. Sets the "
        "backbone's image-processor size and the renderer's emitted size together. Omit for the "
        "model's own input size (e.g. 224 for DINOv3). Not every model is resizable.",
        json_schema_extra={
            "advanced": True,
            "label": "Input size",
            "ui_max": 4096,
            "ui_step": 16,
        },
    )
    resize: int | None = Field(
        None,
        ge=1,
        description="Final square size the renderer emits (px), independent of the model "
        "input size. Omit to follow input_size or the model's own input size. The renderer then "
        "hands the model exactly that, instead of a larger canvas it re-downscales every frame.",
        json_schema_extra={"advanced": True, "label": "Resize"},
    )
    canvas_scope: DerivedScope = Field(
        "per_recording",
        description="How the centered constructors size their shared canvas across a "
        "multi-recording run: per_recording sizes it to each recording's own largest mask "
        "(normalizes away body-size differences between subjects); shared sizes it once to "
        "the largest mask over the whole selection (keeps true relative size, but a smaller "
        "subject then fills less of the model input). No effect on a single recording.",
        json_schema_extra={"advanced": True, "label": "Canvas scope"},
    )
    every_n_frames: int | None = Field(
        None,
        ge=1,
        description=(
            "Use every Nth frame of the video. Omit to embed every frame the tracking run "
            "produced. Counted in source frames, so it must be a multiple of what tracking "
            "used: after tracking at 2, 4 embeds half of what it produced."
        ),
        json_schema_extra={"label": "Use every"},
    )
    batch_size: int = Field(
        32,
        ge=1,
        le=1024,
        description="Images per forward pass.",
        json_schema_extra={"advanced": True, "label": "Batch size"},
    )
    # §14 *Egocentric alignment*: turn each subject image so the animal faces one way before
    # the model sees it, from a keypoint track run through a heading definition.
    align: Literal["none", "rotate", "mirror"] = Field(
        "none",
        description=(
            "Egocentric alignment: make every animal face the same way before embedding. "
            "'rotate' (overhead camera) turns each image so the head points right; "
            "'mirror' (side camera) flips an animal facing left so it faces right; "
            "'none' embeds the image as cropped. "
            "Needs a keypoint track run and a heading definition. A frame with no heading "
            "within the gap is not embedded, so an aligned run never mixes turned and "
            "unturned pictures."
        ),
        json_schema_extra={"label": "Egocentric alignment"},
    )
    keypoint_run_id: str | None = Field(
        None,
        description=(
            "The keypoint track run (kt_NN) the heading and the pose are read from; it must "
            "come from this embedding's own tracking run."
        ),
        json_schema_extra={"label": "Keypoint track run"},
    )
    heading: str | None = Field(
        None,
        description=(
            "A saved heading definition: the skeleton points whose mean is the front and "
            "the points whose mean is the back. The heading is the direction from back to "
            "front, in pixels; the resolved definition is copied onto the run."
        ),
        json_schema_extra={"label": "Heading"},
    )
    max_gap_s: float = Field(
        0.5,
        ge=0.0,
        le=30.0,
        description=(
            "A frame without its own heading takes one interpolated across the gap to the "
            "nearest defined frames, when that gap is no longer than this, in seconds; in a "
            "longer gap the frame is not embedded and is counted as lost to heading."
        ),
        json_schema_extra={
            "advanced": True,
            "label": "Fill heading gaps up to",
            "unit": "s",
        },
    )
    mirror_hold_s: float = Field(
        1.0,
        ge=0.0,
        le=30.0,
        description=(
            "mirror only: the animal must face the new way for this long, in seconds, "
            "before the image flips, so a heading hovering near straight-on does not flicker."
        ),
        json_schema_extra={"advanced": True, "label": "Mirror hold", "unit": "s"},
    )
    facing_camera_ratio: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "A heading axis shorter than this fraction of the track's typical axis length "
            "(its 95th percentile over the run) is undefined: an animal facing the camera "
            "foreshortens its axis to noise, which is where headings flip."
        ),
        json_schema_extra={"advanced": True, "label": "Facing-camera ratio"},
    )
    # §14 *Pose in the embedding* (#659): the keypoint track run's points in the vector too.
    pose: PoseMode = Field(
        "none",
        description=(
            "Put the pose, not only the mask's shape, into the embedding, from the keypoint "
            "track run. none: the mask alone. draw: the body points are painted onto the "
            "image before the model sees it, so one vector holds the mask and the pose "
            "together. append: the pose is written as numbers (each point's place on the "
            "body, turned so the heading points right) and added after the mask's vector, "
            "weighed against it by 'Pose weight'; append needs a heading. Either needs points "
            "on every frame this run embeds."
        ),
        json_schema_extra={"label": "Pose"},
    )
    pose_weight: float = Field(
        1.0,
        gt=0.0,
        le=10.0,
        description=(
            "append only: how much the pose numbers count against the mask's vector when "
            "frames are compared. 1 gives them equal say, 0.5 half as much, 2 twice as much. "
            "Both parts are scaled to the same length first, so this alone sets the balance."
        ),
        json_schema_extra={"label": "Pose weight"},
    )
    pose_min_score: float | None = Field(
        None,
        # every method's score lies in [-1, 1]: cosine similarity, heatmap peak, confidence
        ge=-1.0,
        le=1.0,
        description=(
            "A point scoring below this is left out of the pose (not drawn, and counted as "
            "missing in the numbers). Omit for the keypoint track run's own cut-off, the one "
            "the heading uses."
        ),
        json_schema_extra={"advanced": True, "label": "Pose min score"},
    )
