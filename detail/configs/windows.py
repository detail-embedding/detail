"""The window model (§23) — where a measure is evaluated.

Measures own this and Comparison consumes it: "did it change over the session?" is the same
operation as "did it differ between groups?", with the window as the grouping key rather than a
metadata column. It lived in ``configs/comparison.py`` until Measures grew its own surface, which
would otherwise have made the lower layer import the higher one's config.
"""

from typing import Literal

from pydantic import BaseModel, Field

from detail.core.errors import UserInputError

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

    def windows_for(
        self, start_t_ns: int, end_t_ns: int, *, frame_ns: int | None = None
    ):
        """This spec resolved against one recording's span — seconds converted here.

        ``frame_ns`` is the recording's frame period, when known: a tile, window or step shorter
        than one frame holds no frame of its own, so it is refused in seconds, naming the frame
        rate, rather than as the nanosecond count it truncated to.

        Raises:
            UserInputError: A duration shorter than one frame (or than a nanosecond).
        """
        from detail.analysis import windows as windows_mod

        if self.kind == "whole":
            return windows_mod.whole(start_t_ns, end_t_ns)
        if self.kind == "absolute":
            return windows_mod.absolute(
                start_t_ns, end_t_ns, every_ns=self._duration_ns("every_s", frame_ns)
            )
        if self.kind == "normalized":
            return windows_mod.normalized(start_t_ns, end_t_ns, parts=self.parts)
        return windows_mod.sliding(
            start_t_ns,
            end_t_ns,
            length_ns=self._duration_ns("length_s", frame_ns),
            step_ns=self._duration_ns("step_s", frame_ns),
        )

    def windows_for_coverage(self, coverage):
        """:meth:`windows_for` over one coverage row, its frame period taken from the row."""
        frame_ns = (
            coverage.observed_ns // coverage.n_observed
            if coverage.n_observed > 0
            else None
        )
        return self.windows_for(
            coverage.start_t_ns, coverage.end_t_ns, frame_ns=frame_ns or None
        )

    def _duration_ns(self, field: str, frame_ns: int | None) -> int:
        """``field`` (seconds) in ns, refused when it is shorter than one frame."""
        seconds = getattr(self, field)
        ns = int(seconds * _NS_PER_S)
        floor = frame_ns if frame_ns else 1
        if ns >= floor:
            return ns
        label = type(self).model_fields[field].json_schema_extra["label"]
        flag = "--" + field.removesuffix("_s")
        if frame_ns:
            frame_s = frame_ns / _NS_PER_S
            raise UserInputError(
                f"{label} ({flag}) {seconds:g} s is shorter than one frame of this recording "
                f"({frame_s:.3f} s at {1 / frame_s:.0f} fps); use at least {frame_s:.3f} s"
            )
        raise UserInputError(
            f"{label} ({flag}) {seconds:g} s is shorter than a nanosecond; use a longer one"
        )
