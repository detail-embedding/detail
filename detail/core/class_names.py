"""A detection class's stored name and the user's name for it (§1 *Class names*).

The stored name is the ``class_name`` a detection run wrote; it is in every row and record
downstream and never changes. The name is the user's, set per experiment in ``config.yaml``'s
``class_names`` (``{stored: name}``, only renamed classes), and is what every surface shows.
So there are two directions and one function each: :func:`label` for display, and
:func:`resolve` / :func:`resolve_config` for input, where a user may type either.

Pure: dicts in, strings out. The storage layer reads and writes the map; the API knows the
roster.
"""

from collections.abc import Iterable

NAMES_KEY = "class_names"

#: How a merged tracking run's ``class_name`` joins its source classes (``a+b``).
MERGED_SEPARATOR = "+"

#: Every config field that **refers to** an existing class, and where the class sits in it:
#: the value itself, each item of a list, or each key of a map. A detection prompt target's
#: ``class_name`` is absent on purpose — it defines a stored name rather than naming one.
CLASS_REF_FIELDS: dict[str, str] = {
    "class_name": "value",
    "class_names": "list",
    "class_example_sets": "keys",
    "class_conf_thresholds": "keys",
}


def label(stored: str, names: dict[str, str]) -> str:
    """What a person reads for the class stored as ``stored`` — each source class's name
    for a merged run's joined classes."""
    if stored in names or MERGED_SEPARATOR not in stored:
        return names.get(stored, stored)
    return MERGED_SEPARATOR.join(
        names.get(part, part) for part in stored.split(MERGED_SEPARATOR)
    )


def resolve(ref: str, names: dict[str, str], *, known: Iterable[str]) -> str:
    """The stored name ``ref`` means: itself when it is a stored name (in ``known`` or
    renamed), else the one class named ``ref``, else ``ref`` unchanged so the consumer's own
    check says the class does not exist."""
    if ref in names or ref in set(known):
        return ref
    matches = [stored for stored, name in names.items() if name == ref]
    return matches[0] if len(matches) == 1 else ref


def resolve_config(
    config: dict, names: dict[str, str], *, known: Iterable[str]
) -> dict:
    """``config`` with every :data:`CLASS_REF_FIELDS` reference resolved to its stored name.

    Returns a copy; a field that is absent or ``None`` is left as it is.
    """
    known = set(known)
    if not names:
        return dict(config)
    out = dict(config)
    for field, shape in CLASS_REF_FIELDS.items():
        value = out.get(field)
        if value is None:
            continue
        if shape == "value":
            out[field] = resolve(value, names, known=known)
        elif shape == "list":
            out[field] = [resolve(ref, names, known=known) for ref in value]
        else:
            out[field] = {
                resolve(ref, names, known=known): item for ref, item in value.items()
            }
    return out


def check(names: dict[str, str], *, known: Iterable[str]) -> None:
    """Refuse a map in which a name is blank, two classes show the same name, or a name is
    another class's stored name (a user typing it would mean two classes).

    Raises:
        ValueError: Naming the class at fault.
    """
    known = set(known) | set(names)
    for stored, name in names.items():
        if not name.strip():
            raise ValueError(
                f"class {stored!r} needs a name; leave it unset to keep it"
            )
        if name != stored and name in known:
            raise ValueError(
                f"{name!r} is already a class in this experiment; {stored!r} cannot "
                "take its name"
            )
    shown: dict[str, str] = {}
    for stored in sorted(known):
        name = label(stored, names)
        if name in shown:
            raise ValueError(
                f"{shown[name]!r} and {stored!r} would both be called {name!r}; give "
                "each class its own name"
            )
        shown[name] = stored


def renamed(names: dict[str, str], stored: str, name: str | None) -> dict[str, str]:
    """``names`` with ``stored`` renamed to ``name`` — stripped, and removed when ``name``
    is empty, ``None`` or the stored name itself (it is then not renamed)."""
    out = dict(names)
    name = (name or "").strip()
    if not name or name == stored:
        out.pop(stored, None)
    else:
        out[stored] = name
    return out
