"""Config model for tuning a detector's detection and tracking settings (§12 *Tune detection and
tracking settings*) — the single source for the sample size's bounds, default, help and label,
read by the distill form, the model page's *Tune settings…*, the CLI options and the API
defaults alike.

The model and the recordings are experiment-aware inputs and stay arguments of
``Experiment.tune_tracking`` / ``Experiment.distill``, as ``label_dataset`` keeps its classes.
"""

from pydantic import BaseModel, ConfigDict, Field

#: Windows of video taken from each recording, spread evenly across its timestamp map. A
#: method constant rather than a field: the sample's size is the knob, its spread is not.
SAMPLE_WINDOWS = 5


class TuningConfig(BaseModel):
    """How much video the tuning measures the model on."""

    model_config = ConfigDict(extra="forbid")

    sample_seconds: float = Field(
        300.0,
        ge=10.0,
        le=3600.0,
        description=(
            "Seconds of video to measure the model on, per recording, taken as "
            f"{SAMPLE_WINDOWS} windows spread across it (all of it when the recording is "
            "shorter). More is a steadier answer and costs detection time in proportion."
        ),
        json_schema_extra={
            "advanced": True,
            "label": "Tuning sample (s per recording)",
            "unit": "s",
        },
    )


#: The help a surface shows beside the switch that turns tuning on in a distill.
TUNE_HELP = (
    "After training, run the new model on a sample of the source recordings at a low "
    "confidence, try a grid of detection confidence and tracker merging settings through the "
    "real tracker, and recommend the most stable one. It adds detection time on the sample "
    "(a few minutes per recording on a GPU) and a short sweep on the CPU. Worth it when masks "
    "flicker or come out in pieces, or when there is time to spare. Nothing changes until you "
    "press Use tuned settings on the detection or tracking form."
)
