"""``None`` for a setting with a default means that default, at every entry a run's config takes.

Internal callers spell "use the default" as ``None``: an API keyword such as keypoint setup's
``tracker_id: str | None = None`` is passed straight on into a stage's config. Most config
models declare those fields as a plain type with a default (``tracker_id: str =
DEFAULT_TRACKER``), so pydantic fills the default for an *omitted* key and refuses an explicit
``None``. Each module then validated the same intent two ways, and keypoint setup failed at its
tracking stage on the spelling the tests never used. ``tests/configs/test_none_means_default.py``
is the guard.
"""

from __future__ import annotations

import types
import typing

from pydantic import BaseModel


def _accepts_none(annotation) -> bool:
    if annotation is typing.Any or annotation is None or annotation is type(None):
        return True
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return any(_accepts_none(arg) for arg in typing.get_args(annotation))
    if origin is typing.Annotated:
        return _accepts_none(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        return None in typing.get_args(annotation)
    return False


def none_means_default(model: type[BaseModel] | None, config: dict) -> dict:
    """``config`` with each ``None`` the model would refuse replaced by that field's default.

    Only a field that has a default and cannot hold ``None`` is touched: there ``None`` can only
    ever have been a validation error, so the substitution changes no config that validates
    today. A required field, or one whose ``None`` means something (``model_variant``,
    ``arena_name``), is left as given. The default is written into the dict rather than the key
    dropped, so the run record names what was used (Tenet 9).
    """
    if model is None:
        return dict(config)
    out = dict(config)
    for name, field in model.model_fields.items():
        if out.get(name, ...) is not None or field.is_required():
            continue
        if _accepts_none(field.annotation):
            continue
        out[name] = field.get_default(call_default_factory=True)
    return out
