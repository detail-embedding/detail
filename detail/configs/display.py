"""View parameters — the knobs that change what a finished result *shows*, not what a run
computes.

A view threshold still has a range, a default, and help text, so it lives in a config model
like every other parameter (§6.5): the GUI slider reads its bounds through
``_pydantic_form.field_slider`` and the CLI reads its help and default through
``describe_config``. Hand-writing the range in a widget or a ``@click.option`` is the bug
this avoids.

Deliberately *not* part of any run's config, and deliberately absent from the generated
parameter reference (``iter_config_models`` walks the module registries): a preview runs at
0.0 and keeps every proposal, so this filters the picture, not the run. Nothing here is
recorded in a run manifest — two people looking at the same run at different thresholds are
looking at the same run.
"""

from pydantic import BaseModel, Field

# What a preview / tool passes to ``DetectionModel.set_conf_threshold``: keep everything the
# model proposed, so the threshold can be chosen afterwards. This is the run-side half of the
# split — SAM3 drops proposals inside its own processor before returning them, so a result
# filtered at run time has genuinely lost the rows a slider would want back.
KEEP_ALL_PROPOSALS = 0.0


class ConfidenceView(BaseModel):
    """The confidence threshold a detection result is *displayed* at."""

    min_conf: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Only show detections at or above this confidence. The model kept every "
            "proposal it found, so moving this changes the picture, not the run — slide "
            "down to see what a looser threshold would have caught, up to see what a "
            "stricter one would have cut."
        ),
        json_schema_extra={
            "label": "Detection confidence threshold",
            "ui_step": 0.01,
        },
    )
