"""The one reader for YAML a user hands in or edits by hand.

Importer files (skeletons, example sets, heading definitions, stream presets), the bulk
import and labels manifests, an export archive's manifest, and the ``config.yaml`` /
``recording.yaml`` a user may edit or a sync conflict may damage. Each used to call
``yaml.safe_load`` bare, so a missing bracket reached the user as a ``ParserError``
traceback and a PNG picked by mistake as ``Error: utf-8``.

Here a syntax error becomes a :class:`~detail.core.errors.UserInputError` naming the file,
the line and column and the problem; bytes that are not UTF-8 text become "is not a text
file". A wrong top-level shape is the caller's to name (:func:`require_mapping` is the
common one). ``UserInputError`` is a ``ValueError``, so ``dirscan.UNREADABLE_RECORD_ERRORS``
still skips such a record in a listing.

**A key written twice is refused** (#31). ``yaml.safe_load`` keeps the last of two equal keys
in one mapping, so a labels manifest listing ``recording_001:`` twice imported only the second
file and said nothing. :func:`refuse_unknown_keys` is the other half of reading a hand-written
mapping: a misspelled key (``vidoe:``) is named rather than ignored.

Run records DETAIL writes and reads back for itself keep their own readers;
``tests/storage/test_user_file_readers.py`` lists every other ``yaml.safe_load`` with why.
"""

from pathlib import Path
from typing import Any

import yaml

from detail.core import atomic
from detail.core.errors import UserInputError


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a key repeated in one mapping.

    Keys a ``<<`` merge brings in may be overridden, as YAML intends; only the keys written
    in the mapping itself are compared.
    """

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            first_line: dict = {}
            for key_node, _value in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = self.construct_object(key_node, deep=True)
                try:
                    seen = first_line.get(key)
                except TypeError:  # an unhashable key: the base class names it
                    continue
                if seen is not None:
                    raise yaml.constructor.ConstructorError(
                        None,
                        None,
                        f"key {key!r} appears twice in one mapping (first on line {seen}); "
                        "keep one",
                        key_node.start_mark,
                    )
                first_line[key] = key_node.start_mark.line + 1
        return super().construct_mapping(node, deep=deep)


def _syntax_message(source: str, exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None) or str(exc)
    if mark is None:
        return f"{source} is not valid YAML: {problem}"
    return (
        f"{source} is not valid YAML (line {mark.line + 1}, column {mark.column + 1}): "
        f"{problem}"
    )


def parse_user_yaml(text: str | bytes, source: Path | str) -> Any:
    """``yaml.safe_load(text)``, with a mistake reported against ``source``.

    Raises:
        UserInputError: ``text`` is not valid YAML, repeats a key in one mapping, or is
            bytes that are not UTF-8 text.
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise UserInputError(f"{source} is not a text file") from None
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 — a SafeLoader
    except yaml.YAMLError as exc:
        raise UserInputError(_syntax_message(str(source), exc)) from exc


def read_user_yaml(path: Path | str) -> Any:
    """Read and parse the YAML file at ``path`` (``None`` for an empty file).

    Raises:
        FileNotFoundError: No file at ``path``.
        UserInputError: The file is not UTF-8 text, or not valid YAML.
    """
    try:
        text = atomic.read_text(path)
    except UnicodeDecodeError:
        raise UserInputError(f"{path} is not a text file") from None
    return parse_user_yaml(text, path)


def require_mapping(data: Any, source: Path | str, what: str) -> dict:
    """``data`` when it is a mapping; otherwise a message naming what was expected.

    Raises:
        UserInputError: ``data`` is not a mapping.
    """
    if not isinstance(data, dict):
        found = "empty" if data is None else f"a {type(data).__name__}"
        raise UserInputError(
            f"{source} is not {what}: expected a YAML mapping (key: value lines), "
            f"found {found}"
        )
    return data


def refuse_unknown_keys(
    data: dict, allowed: tuple[str, ...], source: Path | str, what: str
) -> None:
    """Refuse a key ``data`` has that ``allowed`` does not name.

    A misspelled key is otherwise ignored, and the reader goes on as if it were absent: a
    manifest's ``vidoe:`` made a recording with no video and called the manifest valid.

    Raises:
        UserInputError: Naming each unknown key and the keys ``what`` takes.
    """
    unknown = [k for k in data if k not in allowed]
    if unknown:
        raise UserInputError(
            f"{source}: {what} has unknown key(s) {', '.join(map(repr, unknown))}; "
            f"the keys it takes are {', '.join(allowed)}"
        )
