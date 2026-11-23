"""Query run config (§20) — the run-level knobs, single-sourced for CLI help, GUI tooltips
and the generated parameter reference.

Covers the **run-level** scalars only. The two nested halves stay outside the model and are
checked in ``QueryModule.validate``: the search recordings + embedding run, and the per-behavior
``behaviors`` list (a name and its example clips). That split is the section's, not an
implementation convenience — §20 argues it at length:

**There is no threshold field, and its absence is the design.** "The search config has no
threshold field at all, so there is no way to spend hours searching and then discover a change
of mind costs another search." The cutoff is authored afterwards, lives in ``query.yaml``, and
is **per behavior** because each behavior's confidence distribution is its own and a shared
cutoff would mean nothing. Adding one here would undo the module's whole shape, so if a future
edit is tempted, that is the paragraph to read first.

**Method parameters are run-level, not per behavior.** Instances produced under different
parameters are not comparable, and a run emits *one* §19 artifact the measures layer treats as
homogeneous. With no ground truth a user has no signal to tune per behavior against either, so
a per-behavior knob would be adjusted by impression. Two settings means two runs, which is
honest about what happened.

The method's *own* parameters are a second model, declared by the method
(``QueryMethod.config_model`` — e.g. ``configs/query_soft_knn.py``) and carried here as an opaque
dict, so adding a method needs no change to this file.

``method`` is a plain ``str`` rather than a ``Literal``. Query eval, a module since removed,
typed its ``method`` as a ``Literal`` over the *old* engine's fixed matcher list; §20's methods
are a registry (``QUERY_METHODS``) whose whole point is that a method can be added without
editing the module. So the choices come from the registry at the point each surface offers
them — the CLI's ``click.Choice``, the GUI's combo — and ``validate()`` refuses an unknown id,
naming what is registered.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

#: The method a run gets when nobody picks one. Named here rather than written into each
#: surface: the CLI's option default, the GUI's opening selection and the API's model default
#: are one fact, and three copies of it drift.
DEFAULT_METHOD = "soft_knn"


class QueryConfig(BaseModel):
    """Run-level configuration for one Query run (§20)."""

    embed_run_id: str = Field(
        ...,
        description="Embedding run id shared by the searched recordings (e.g. emb_01).",
        json_schema_extra={"scope": "selection", "label": "Embed run"},
    )
    track_id: int = Field(
        0,
        ge=0,
        description="Subject track id to search (single-subject default 0).",
        json_schema_extra={"label": "Subject", "ui_max": 9999},
    )
    method: str = Field(
        DEFAULT_METHOD,
        description="Search method, by id. The default is chosen on measured cost, not on "
        "accuracy: soft_knn is fast. Which method "
        "retrieves *better* has never been measured, so each method's page says what is and "
        "is not known about it (§20).",
        json_schema_extra={"label": "Method", "advanced": True},
    )
    trajectory_split_gap_s: float | None = Field(
        None,
        ge=0.0,
        description="A break in tracking longer than this (seconds) splits the trajectory "
        "into separate stretches to search, and an example clip must sit inside one. "
        "0 = never split. Default: ~4x the subject's frame period.",
        json_schema_extra={
            "advanced": True,
            "label": "Trajectory split gap",
            "unit": "s",
            "ui_max": 60.0,
        },
    )
    min_match_s: float = Field(
        0.0,
        ge=0.0,
        description="Shortest match to return, in seconds. A match is never formed shorter "
        "than this, so a brief spike cannot crowd out a longer match beside it. A match of "
        "n frames lasts n frame periods. Applied during the search, so changing it means "
        "searching again. 0 = no minimum.",
        json_schema_extra={
            "label": "Min match length",
            "unit": "s",
            "ui_max": 60.0,
            "ui_step": 0.1,
        },
    )
    background_seed: int = Field(
        0,
        ge=0,
        description="Seed for the background sample, when the chosen method needs one. The "
        "module draws the background, never the method, so the draw is recorded and a run "
        "reproduces; change this to check how much a result depends on which background "
        "frames were drawn. Ignored by a method that needs no background.",
        json_schema_extra={
            "advanced": True,
            "label": "Background seed",
            "ui_max": 1_000_000,
        },
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_none_defaults(cls, data: Any) -> Any:
        """Let a caller pass ``None`` for a defaulted scalar to mean "use the default".

        Every surface builds this dict from widgets and flags that may be unset, and a
        ``None`` reaching a non-optional field is rejected as a type error rather than
        falling back — the same accommodation the removed ``clip_query`` module made for
        ``threshold``.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name in ("track_id",):
            if name in out and out[name] is None:
                out.pop(name)
        return out
