"""A nanosecond time or a byte count never travels in a signal argument typed ``int``.

**The class: a 64-bit value in a 32-bit signal argument.** ``Signal(int)`` declares a C++
``int``: 32 bits, so it stops at 2**31. In nanoseconds that is 2.1 s; in bytes it is 2 GB.
PyQt6 wrapped such a value silently, so the number arrived wrong and nothing said so: a click on
the stream strip 3 s in sent a negative time, and the montage's offsets past 2.1 s were wrong
(its slot ignores them, which hid the bug). PySide6 warns (``libshiboken: Overflow``), then runs
the slot with the error still pending. Reported: the app segfaulted a few seconds into playing
any montage, as each tile's clip passed 2.1 s, right after that warning in
``GuiWorker._deliver``. Reproduced with the real montage playing 10 s clips: exit 139 at the
first offset past 2**31.

**The rule.** An argument that carries a time in nanoseconds or a byte count is declared
``object``, which passes the Python int through whole. ``int`` stays right for a frame number,
an index, a node id or a per-mille fraction.

**The scan** (:func:`violations`) reads every emission in ``detail/``, ``self.x.emit(...)`` and
``self.emit_to_ui(self.x, ...)``, and flags an argument whose source names a nanosecond or byte
quantity (``t_ns``, ``offset_ns``, ``self._t_min``, ``bytes_done``) where a ``Signal`` named
``x`` in the same file declares ``int`` at that position. It sees names, not values: an int
argument computed from an unnamed quantity passes it. The tests below it send a value past
2**32 through each signal that carries one and check it arrives unchanged.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECOND = 1_000_000_000

#: Source text naming a nanosecond time or a byte count.
_WIDE = re.compile(r"(?<![A-Za-z0-9])t_(?:ns|min|max)\b|_ns\b|bytes")


def _declarations(tree: ast.AST) -> dict[str, list[list[str]]]:
    """``{signal name: [argument types, one list per declaration]}`` in one module."""
    found: dict[str, list[list[str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Name) and func.id == "Signal"):
            continue
        types = [ast.unparse(arg) for arg in node.value.args]
        for target in node.targets:
            if isinstance(target, ast.Name):
                found.setdefault(target.id, []).append(types)
    return found


def _emissions(tree: ast.AST):
    """``(line, signal name, argument nodes)`` for each emission through ``self``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        if func.attr == "emit" and isinstance(func.value, ast.Attribute):
            yield node.lineno, func.value.attr, node.args
        elif func.attr == "emit_to_ui" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Attribute):
                yield node.lineno, first.attr, node.args[1:]


def violations(sources: dict[str, str]) -> list[str]:
    """Each emission that sends a nanosecond or byte value into an ``int`` argument."""
    found = []
    for path, source in sources.items():
        tree = ast.parse(source)
        declared = _declarations(tree)
        for line, name, args in _emissions(tree):
            for position, arg in enumerate(args):
                if isinstance(arg, ast.Starred):
                    break
                text = ast.unparse(arg)
                if not _WIDE.search(text):
                    continue
                if any(
                    position < len(types) and types[position] == "int"
                    for types in declared.get(name, [])
                ):
                    found.append(
                        f"{path}:{line}: {name} argument {position} is int: {text}"
                    )
    return found


def _sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8-sig")
        for path in sorted((ROOT / "detail").rglob("*.py"))
    }


def test_no_nanosecond_or_byte_value_is_sent_through_an_int_argument():
    assert violations(_sources()) == []


def test_the_scan_catches_each_way_a_wide_value_is_sent():
    """The negative cases, as literal source: each shape the scan exists for is flagged, and
    the same emission into an ``object`` argument, or a frame number into ``int``, is not.
    """
    flagged = {
        "direct": "class A:\n    s = Signal(int)\n    def f(self, t_ns):\n"
        "        self.s.emit(t_ns)\n",
        "through the worker hop": "class A:\n    d = Signal(int, int, int, object)\n"
        "    def f(self, g, i, offset_ns, img):\n"
        "        self.emit_to_ui(self.d, g, i, offset_ns, img)\n",
        "a time computed from the strip's range": "class A:\n    c = Signal(int)\n"
        "    def f(self, frac):\n"
        "        self.c.emit(int(self._t_min + frac * (self._t_max - self._t_min)))\n",
        "a byte count": "class A:\n    p = Signal(int, int, str)\n    def f(self, p):\n"
        "        self.emit_to_ui(self.p, p.bytes_done, p.bytes_total, '')\n",
    }
    for case, source in flagged.items():
        assert violations({"x.py": source}), f"missed: {case}"
    passed = {
        "object argument": "class A:\n    s = Signal(object)\n    def f(self, t_ns):\n"
        "        self.s.emit(t_ns)\n",
        "a frame number": "class A:\n    s = Signal(int)\n    def f(self, frame):\n"
        "        self.s.emit(frame)\n",
    }
    for case, source in passed.items():
        assert violations({"x.py": source}) == [], f"flagged: {case}"


# --- the values themselves, through each signal that carries one ------------------------ #


def test_a_montage_offset_past_32_bits_arrives_whole(qapp):
    """The picture worker's delivery, through the real UI-thread hop (``_deliver``): the
    path that crashed."""
    from detail.gui.live_montage import _PictureWorker

    worker = _PictureWorker(receiver=None)
    got = []
    worker.delivered.connect(lambda *args: got.append(args))
    worker._deliver(worker.delivered, (3, 7, 5 * SECOND, None))
    assert got == [(3, 7, 5 * SECOND, None)]


def test_a_click_on_the_stream_strip_sends_its_time_whole(qapp):
    """A click at the right end of a 10 s strip sends a time near 10 s, not a wrapped one."""
    from detail.gui.qt import QPoint, QTest, Qt
    from detail.gui.streams import _EventStrip

    strip = _EventStrip(["lever"], {"lever": []}, 0, 10 * SECOND)
    strip.resize(600, strip.height())
    got = []
    strip.clicked.connect(got.append)
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(strip.width() - 1, 5))
    assert len(got) == 1 and 9 * SECOND < got[0] <= 10 * SECOND, got


def test_a_download_past_2gb_reports_its_bytes_whole(qapp):
    from detail.gui.workers import InstallModelWorker

    worker = InstallModelWorker("any", receiver=None)
    got = []
    worker.progress.connect(lambda *args: got.append(args))
    worker._deliver(worker.progress, (3_000_000_000, 4_500_000_000, "weights.pt"))
    assert got == [(3_000_000_000, 4_500_000_000, "weights.pt")]
