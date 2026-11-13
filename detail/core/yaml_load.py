"""The one YAML reader: ``yaml.safe_load``'s exact output, parsed by libyaml when it is there.

PyYAML ships two parsers behind one constructor. ``yaml.safe_load`` uses the pure-Python one,
which costs about 2 ms per kilobyte: a 35 KB ``recording.yaml`` took 64 ms to parse on the WSL
box and 97 ms on Windows, and the 72 of one experiment cost 5-7 s the first time a cluster's
clips were listed. ``CSafeLoader`` is the same ``SafeConstructor`` and resolver fed by libyaml's
scanner and parser, so the tags, the types and the values it builds are the ones ``safe_load``
builds; only the tokenizing moves to C. Measured on the same file: 7.8 ms, and 6.4x over every
YAML file in the user's experiments (3,311 files, 18 MB, every result type-for-type equal, the
two broken ones refused by both with the same exception class).

Every PyYAML wheel this project installs carries libyaml; a build without it falls back to the
pure-Python loader with the same result, only slower. Errors stay ``yaml.YAMLError`` subclasses
(``ReaderError`` for a zero-filled file, ``ParserError`` / ``ScannerError`` for a torn one) under
either loader, so every ``except yaml.YAMLError`` keeps catching what it caught.

``tests/core/test_yaml_load.py`` fails on a ``yaml.safe_load`` / ``yaml.load`` call anywhere else
in ``detail/``: a reader that bypasses this one is 8x slower and nothing at its line says so.
"""

from __future__ import annotations

import yaml

#: libyaml's parser under ``yaml.safe_load``'s constructor, or the pure-Python loader without it.
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def safe_load(stream):
    """``yaml.safe_load(stream)``, parsed by libyaml when it is installed.

    ``stream`` is a ``str``, ``bytes`` or an open file, as for ``yaml.safe_load``.

    Raises:
        yaml.YAMLError: the document is not valid YAML, or holds a tag the safe constructor
            refuses (``!!python/tuple``).
    """
    return yaml.load(stream, Loader=LOADER)  # noqa: S506 - a safe loader, see LOADER
