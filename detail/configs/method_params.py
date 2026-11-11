"""Resolving a registry method's parameters against the model the method declares.

§20's query methods carry an opaque ``method_params`` dict (as §22's pose-clustering methods did
until they were removed) that the module never looks inside — the method's own ``config_model``
is what gives it types, bounds and defaults. Turning the raw dict into the resolved one is
therefore the same job for every method, and it is the place Tenet 9's rule applies: *the record
carries the value that was used, not the request that asked for one.*

**Unknown keys are refused, and that is the whole reason this is a function.** Pydantic
ignores extras by default, so ``model_validate({"durationmin": 2})`` returns a valid config
with the typo silently dropped — the parameter reads as accepted and the run uses the default.
That is indistinguishable from the defect this module exists to close (a knob a surface offers
and the run never receives), and it is exactly what a hand-typed
``--method-param name=value`` invites. The error names the method and lists what it does take.
"""

from typing import Any

from pydantic import BaseModel, ValidationError

#: Parameters a method used to take and no longer does, dropped silently rather than refused:
#: old runs saved them in ``method_params`` (the clustering scatter's layout, removed in §22's
#: redesign, and ``seed`` for the methods that only used it to seed that layout), and "start
#: from a previous run" must still validate. A name is dropped only for a method that no longer
#: declares it: ``codebook_viterbi`` still takes a ``seed``.
RETIRED_PARAMS = frozenset({"projection_points", "projection_layout", "seed"})


def resolve_method_params(method: Any, raw: dict | None) -> dict:
    """``raw`` validated against ``method.config_model``, with every default filled in.

    Args:
        method: the registered method (an instance or the class) declaring ``config_model``.
        raw: what a surface collected, or ``None`` for "nothing was set".

    Returns:
        The resolved parameters, ready to be written into the run record.

    Raises:
        ValueError: if a key is not one of the method's parameters (a ``RETIRED_PARAMS``
            name is dropped instead), or a value is out of the
            bounds the model declares. Both name the method, because a user picking a method
            from a registry has no other way to know whose parameter list they got wrong.
    """
    model_cls: type[BaseModel] = method.config_model
    method_id = getattr(method, "method_id", model_cls.__name__)
    values = {
        key: value
        for key, value in (raw or {}).items()
        if key not in RETIRED_PARAMS or key in model_cls.model_fields
    }
    unknown = sorted(set(values) - set(model_cls.model_fields))
    if unknown:
        raise ValueError(
            f"{method_id} has no parameter(s) {unknown}; it takes "
            f"{sorted(model_cls.model_fields)}"
        )
    try:
        return model_cls.model_validate(values).model_dump()
    except ValidationError as exc:
        raise ValueError(f"{method_id} parameters are invalid: {exc}") from exc
