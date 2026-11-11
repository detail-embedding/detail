"""Read one pixel of a PNG with the standard library — for tests that must not import a plotter.

A test that asks "is this exported figure on the dark surface" needs the colour of its corner and
nothing else, and importing matplotlib to decode it moves the whole test file out of the ``unit``
tier (``tests/tiers.py`` counts the import). The top-left pixel is recoverable without decoding
anything else: every PNG row filter predicts the first pixel of the first row from zeros, so its
bytes are the raw values.
"""

import struct
import zlib
from pathlib import Path

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
#: Bytes per pixel for the 8-bit colour types matplotlib writes: RGB and RGBA.
_CHANNELS = {2: 3, 6: 4}


def top_left_rgb(path: Path | str) -> tuple[int, int, int]:
    """The ``(r, g, b)`` of a PNG's top-left pixel.

    Raises:
        ValueError: If the file is not an 8-bit RGB or RGBA PNG.
    """
    data = Path(path).read_bytes()
    if not data.startswith(_SIGNATURE):
        raise ValueError(f"{path} is not a PNG")
    offset, idat, header = len(_SIGNATURE), b"", None
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            header = body
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        offset += 12 + length
    if header is None:
        raise ValueError(f"{path} has no header")
    bit_depth, colour_type = header[8], header[9]
    if bit_depth != 8 or colour_type not in _CHANNELS:
        raise ValueError(f"{path}: only 8-bit RGB/RGBA is read (got {colour_type})")
    raw = zlib.decompressobj().decompress(idat, 1 + _CHANNELS[colour_type])
    return tuple(raw[1:4])
