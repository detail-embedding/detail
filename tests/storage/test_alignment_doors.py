"""Fail-closed scan: every write of a stream's alignment offset goes through a clock gate.

**The class.** *A value written under one key or clock and read back under another*, so the
stored data and what a reader or control acts on disagree. Its alignment instance (#30):
``set_stream_offset`` stored an offset on a behavior-label stream, which is on the video's own
clock. The preview applied it and drew the labels five seconds late; scoring ignored it. Only
``add_stream`` checked the clock, and the GUI greyed its buttons, so the CLI and the API went
straight through.

**How it hid.** Each surface gated itself, and the one that did not was the one nobody
looked at. The check now lives in the storage door, ``streams_store.check_offset_allowed``,
called by the only two places allowed to write ``alignment``: ``add_stream`` (a new stream)
and ``streams_store.set_stream_offset`` (an existing one). This scan fails on a third writer,
so a new surface cannot store an offset without going through one of them.

**The rule.** Write a stream's ``alignment`` only through ``streams_store.set_stream_offset``
(or ``Recording.add_stream`` at import), and read the offset only through
``streams_store.stream_offset_ns``.
"""

import re
from pathlib import Path

import detail

_ROOT = Path(detail.__file__).parent

#: The allowed writers, by file and the function they sit in. Each calls the clock gate.
ALLOWED = {
    ("storage/streams.py", "set_stream_offset"),
    ("api/recording.py", "add_stream"),
}

# Writing the block: an assignment into it, a setdefault of it, or a dict literal carrying it.
_WRITE = re.compile(
    r"""\[\s*["']alignment["']\s*\]\s*(\[[^\]]*\]\s*)?=(?!=)"""
    r"""|setdefault\(\s*["']alignment["']"""
    r"""|["']alignment["']\s*:\s*\{"""
)
# Reading the offset anywhere but the one reader: `.get("offset_ns"` on a manifest.
_STRAY_READ = re.compile(r"""manifest\.get\(\s*["']offset_ns["']""")
_DEF = re.compile(r"^\s*def (\w+)\(")


def alignment_writes(source: str) -> list[tuple[int, str | None]]:
    """``(line, enclosing function)`` for every alignment write in ``source``."""
    found = []
    current = None
    for number, line in enumerate(source.splitlines(), start=1):
        match = _DEF.match(line)
        if match:
            current = match.group(1)
        if _WRITE.search(line):
            found.append((number, current))
    return found


def _offenders() -> list[str]:
    out = []
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        for number, func in alignment_writes(source):
            if (rel, func) not in ALLOWED:
                out.append(f"{rel}:{number} (in {func}) writes a stream's alignment")
        for number, line in enumerate(source.splitlines(), start=1):
            if _STRAY_READ.search(line):
                out.append(
                    f"{rel}:{number} reads offset_ns off a manifest: {line.strip()}"
                )
    return out


def test_every_alignment_write_goes_through_a_clock_gate():
    assert _offenders() == []


def test_every_allowed_writer_still_exists_and_calls_the_gate():
    """An entry that excuses nothing is a hole waiting for a new writer to land in it."""
    for rel, func in ALLOWED:
        source = (_ROOT / rel).read_text(encoding="utf-8-sig")
        writers = {f for _n, f in alignment_writes(source)}
        assert func in writers, f"{rel}::{func} no longer writes alignment; drop it"
        body = source[source.index(f"def {func}(") :]
        following = re.search(r"\n\s*def \w+\(", body)
        body = body[: following.start()] if following else body
        assert "check_offset_allowed(" in body, f"{rel}::{func} skips the clock gate"


def test_the_scan_would_catch_each_spelling_of_a_write():
    spellings = [
        'match["alignment"]["offset_ns"] = 5',
        "entry['alignment'] = {'offset_ns': 5}",
        'stream.setdefault("alignment", {})["offset_ns"] = 5',
        'manifest = {"alignment": {"offset_ns": 5}}',
        'm["alignment"] = new',
    ]
    for spelling in spellings:
        source = f"def sneaky():\n    {spelling}\n"
        assert alignment_writes(source) == [(2, "sneaky")], spelling
    # And reads are not writes.
    assert alignment_writes('x = m["alignment"]["offset_ns"] == 5\n') == []
    assert alignment_writes('x = m.get("alignment")\n') == []


def test_the_scan_would_catch_a_stray_offset_read():
    assert _STRAY_READ.search('int(manifest.get("offset_ns", 0) or 0)')
    assert not _STRAY_READ.search(
        '(manifest.get("alignment") or {}).get("offset_ns", 0)'
    )
