"""The window model (§23) — where a measure is evaluated.

Measures own this and Comparison consumes it: "did it change over the session?" is the same
operation as "did it differ between groups?", with the window as the grouping key rather than a
metadata column. It lived in ``configs/comparison.py`` until Measures grew its own surface, which
would otherwise have made the lower layer import the higher one's config.
"""

from typing import Literal

from pydantic import BaseModel, Field

_NS_PER_S = 1_000_000_000


class WindowSpec(BaseModel):
    """Which windows each recording is measured over (§23).

    ``whole`` is the default and the common case. The others exist because "did it change over the
    session?" is the same operation as "did it differ between groups?", with the window as the
    grouping key rather than a metadata column.

    **Durations are seconds here and nanoseconds downstream.** A window declared in frames or rows
    is a different duration on every recording, which is what
    ``tests/configs/test_unit_conventions.py`` exists to refuse; the conversion happens once, in
    :meth:`windows_for`, where the recording's span is known.
    """

    kind: Literal["whole", "absolute", "normalized", "sliding"] = Field(
        "whole",
        description="Whole recording, fixed-duration tiles, equal fractions, or a sliding window.",
        json_schema_extra={"label": "Windows"},
    )
    every_s: float = Field(
        600.0,
        gt=0.0,
        description="For kind=absolute: tile width in seconds.",
        json_schema_extra={"label": "Tile width", "unit": "s", "ui_max": 86_400.0},
    )
    parts: int = Field(
        3,
        ge=1,
        description="For kind=normalized: how many equal fractions of each recording.",
        json_schema_extra={"label": "Parts", "ui_max": 100},
    )
    length_s: float = Field(
        300.0,
        gt=0.0,
        description="For kind=sliding: window length in seconds.",
        json_schema_extra={"label": "Window length", "unit": "s", "ui_max": 86_400.0},
    )
    step_s: float = Field(
        60.0,
        gt=0.0,
        description="For kind=sliding: how far the window advances each step.",
        json_schema_extra={"label": "Step", "unit": "s", "ui_max": 86_400.0},
    )

    def windows_for(self, start_t_ns: int, end_t_ns: int):
        """This spec resolved against one recording's span — seconds converted here."""
        from detail.analysis import windows as windows_mod

        if self.kind == "whole":
            return windows_mod.whole(start_t_ns, end_t_ns)
        if self.kind == "absolute":
            return windows_mod.absolute(
                start_t_ns, end_t_ns, every_ns=int(self.every_s * _NS_PER_S)
            )
        if self.kind == "normalized":
            return windows_mod.normalized(start_t_ns, end_t_ns, parts=self.parts)
        return windows_mod.sliding(
            start_t_ns,
            end_t_ns,
            length_ns=int(self.length_s * _NS_PER_S),
            step_ns=int(self.step_s * _NS_PER_S),
        )
