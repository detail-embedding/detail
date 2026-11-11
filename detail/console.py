"""The CLI's output streams, made unable to abort a command (§15).

Python takes ``sys.stdout``'s encoding from the locale: ``cp1252`` on a default Windows
console, ASCII under ``LANG=C``. Neither can represent the check mark ``detail doctor``
prints, and the default error handler *raises* — so the command printed its "Environment"
header and died on the first status row with a ``UnicodeEncodeError``. The one command
``.github/ISSUE_TEMPLATE/bug_report.md`` asks a reporter to paste was the one command a
Windows reporter could not run.

The fix belongs at the stream, not at the ~250 non-ASCII literals the package can print:
which of them reaches stdout is not a property of any one of them, and the next one added
would be written exactly as correctly as the rest. So the streams get an encode error
handler that spells an unencodable character in ASCII instead of raising.

**A transliteration, not** ``errors="replace"``, because ``?`` is not always merely
degraded. ``?`` is already ``doctor``'s *unknown* glyph, so a passing check and an unknown
one would print identically — wrong information, not missing information — and ``≥ 2``
would read as ``? 2``. A character the table does not name still becomes ``?``: unfamiliar
prose degrades, and the command completes either way.

Click does not do this for us. ``_force_correct_text_stream`` re-wraps a stream only when
``_stream_is_misconfigured``, which is *ASCII encoding only* — a cp1252 stream looks
healthy to click and is passed through untouched.
"""

import codecs
import sys
from typing import Any

#: The name the error handler is registered under, for ``reconfigure(errors=...)``.
ERROR_HANDLER = "detail_ascii"

#: How an unencodable character is spelled in ASCII. Everything not named here becomes
#: ``?`` — this table is for the characters where that would read as something other than
#: "a character did not fit".
_ASCII_FALLBACKS = {
    # Status glyphs. One column wide, so `doctor`'s status column still lines up: the
    # glyphs it can already encode on a narrow console (`•`, `–`) are one column too.
    "✓": "+",
    "✔": "+",
    "✗": "x",
    "✘": "x",
    "•": "*",
    "⚠": "!",
    "★": "*",
    "↗": "^",
    # Relations and arrows. These carry meaning — a bound printed as `? 2` says the
    # opposite of what `≥ 2` says.
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "≥": ">=",
    "≤": "<=",
    "≫": ">>",
    "≈": "~",
    "≠": "!=",
    "−": "-",
    "×": "x",
    "±": "+/-",
    "∈": "in",
    "∪": "U",
    # Punctuation. Every one of these is in cp1252, so they matter on a `LANG=C` box
    # rather than on Windows.
    "—": "--",
    "–": "-",
    "…": "...",
    "·": "-",
    "§": "S",
    "°": "deg",
    "µ": "u",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def _ascii_fallback(error: UnicodeError) -> tuple[str, int]:
    """Spell the unencodable run in ASCII and carry on from its end."""
    if not isinstance(error, UnicodeEncodeError):
        raise error  # a decode/translate error is not this handler's business
    unmappable = error.object[error.start : error.end]
    return "".join(_ASCII_FALLBACKS.get(ch, "?") for ch in unmappable), error.end


def configure_output_streams() -> None:
    """Install the ASCII fallback on ``stdout`` and ``stderr``.

    Idempotent. A stream that cannot be reconfigured — a pytest capture object, a plain
    file substitute, ``None`` under ``pythonw`` — is left alone rather than raising:
    hardening the output must never be the thing that breaks it.
    """
    codecs.register_error(ERROR_HANDLER, _ascii_fallback)
    _harden(sys.stdout)
    _harden(sys.stderr)


def _harden(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None or getattr(stream, "closed", False):
        return
    reconfigure(errors=ERROR_HANDLER)
