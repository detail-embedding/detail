"""Finite numbers for the command line (#33, section C).

``click``'s ``float`` and ``FloatRange`` accept ``nan`` and ``inf``: Python's ``float()`` does,
and a range check compares ``nan`` false against both bounds, so ``FloatRange(min=0, max=10)``
lets ``nan`` through and ``FloatRange(min=0)`` lets ``inf`` through. Such a value then reached
``round(seconds * 1e9)`` and printed ``ValueError: cannot convert float NaN to integer`` or an
``OverflowError`` traceback, or was stored as a cut-off nothing could ever pass.

**Every float option in ``detail/cli*.py`` takes its type from here**, and a number parsed out of
an option's text (a ``--cutoff BEHAVIOR=VALUE``) goes through
:func:`parse_finite`. ``tests/storage/test_user_file_readers.py`` fails on a ``type=float`` or a
bare ``click.FloatRange(`` in the CLI modules.

Not the int64 check: a finite number of seconds can still be too large to be a ``t_ns``. That is
``core.utils.seconds_to_t_ns``'s job, where seconds become ``t_ns``, not the option type's.
"""

import math

import click


def parse_finite(text: str) -> float:
    """``text`` as a finite float.

    Raises:
        ValueError: If ``text`` is not a number, or is ``nan`` / ``inf``.
    """
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text!r} is not a finite number")
    return value


class _Finite:
    """Refuses ``nan`` and ``inf`` after the base type has converted (and range-checked)."""

    def convert(self, value, param, ctx):
        rv = super().convert(value, param, ctx)
        if not math.isfinite(rv):
            self.fail(f"{value!r} is not a finite number.", param, ctx)
        return rv


class FiniteFloat(_Finite, click.types.FloatParamType):
    """``type=float``, without ``nan`` and ``inf``."""


class FiniteFloatRange(_Finite, click.FloatRange):
    """``click.FloatRange``, without ``nan`` and ``inf``."""


#: The type for a float option with no bounds.
FINITE_FLOAT = FiniteFloat()


def finite_float_range(
    min: float | None = None,  # noqa: A002 - click.FloatRange's own argument names
    max: float | None = None,  # noqa: A002
    **kwargs,
) -> click.ParamType:
    """A finite float type with ``click.FloatRange``'s arguments; unbounded is plain FLOAT."""
    if min is None and max is None and not kwargs:
        return FINITE_FLOAT
    return FiniteFloatRange(min=min, max=max, **kwargs)
