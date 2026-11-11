"""A whole-image channel swap goes through ``core.visualize``'s cv2 helpers, never numpy indexing.

**The class.** ``frame[:, :, ::-1]`` reverses a frame's channels as a strided view, and the
``np.ascontiguousarray(...)`` / ``.copy()`` that usually follows it is a fancy-index copy with no
SIMD path, many times slower than ``cv2.cvtColor`` with bit-identical output. Once per frame
in a loop over a recording, it is the loop: a profile of a ``keypoint_detect`` worker put
more wall time on that one line than on decoding the frame, with the GPU idle while it ran.

**How it hid.** The helpers existed, with the measurement in their docstring, and nothing made
anyone use them. ``frame[:, :, ::-1]`` is the idiom every OpenCV answer teaches, it is correct, and
it is fast enough on the one frame a test hands it — so it came back at 19 sites, every new
keypoint stage among them. A profile is the only thing that sees it, and a test does not run one.

**The rule.** Swap channels with ``visualize.bgr_to_rgb`` / ``rgb_to_bgr``. The scan reads the
parse tree (the ``test_path_serialization.py`` idiom), so the docstring that quotes the slow
spelling is not a hit, and it matches the spellings of a *whole-image* channel reversal only:
``[:, :, ::-1]``, ``[..., ::-1]`` and the fancy-index ``[:, :, [2, 1, 0]]``. A two-axis
``[:, ::-1]`` — a horizontal flip of an image, or an ``(N, 2)`` xy swap — is a different
operation and is left alone.

The sibling bug found beside it — a ``SubjectFrame`` (RGB) handed to Ultralytics, which reads
numpy as BGR — is a channel-*order* defect, not a cost one; its guard is
``tests/models/test_yolo_pose.py::test_the_student_is_handed_bgr_the_order_it_was_trained_on``.
"""

import ast
import pathlib
import textwrap

import detail


def _is_full_slice(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _is_reversed_slice(node: ast.AST) -> bool:
    """``::-1``."""
    if not (isinstance(node, ast.Slice) and node.lower is None and node.upper is None):
        return False
    step = node.step
    return (
        isinstance(step, ast.UnaryOp)
        and isinstance(step.op, ast.USub)
        and isinstance(step.operand, ast.Constant)
        and step.operand.value == 1
    ) or (isinstance(step, ast.Constant) and step.value == -1)


def _is_bgr_permutation(node: ast.AST) -> bool:
    """``[2, 1, 0]`` (list or tuple)."""
    return (
        isinstance(node, (ast.List, ast.Tuple))
        and [getattr(e, "value", None) for e in node.elts] == [2, 1, 0]
        and all(isinstance(e, ast.Constant) for e in node.elts)
    )


def _swaps_channels(node: ast.AST) -> bool:
    """Whether ``node`` indexes an image's channel axis in reverse."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Tuple):
        return False
    *leading, last = node.slice.elts
    if not (_is_reversed_slice(last) or _is_bgr_permutation(last)):
        return False
    if len(leading) == 1 and isinstance(leading[0], ast.Constant):
        return leading[0].value is Ellipsis  # [..., ::-1]
    return len(leading) == 2 and all(_is_full_slice(n) for n in leading)


def _offenders_in(source: str, label: str) -> list[str]:
    return [
        f"{label}:{line}"
        for line in sorted(
            node.lineno for node in ast.walk(ast.parse(source)) if _swaps_channels(node)
        )
    ]


def _offenders(root: pathlib.Path) -> list[str]:
    found = []
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root.parent).as_posix()
        found += _offenders_in(py.read_text(encoding="utf-8-sig"), rel)
    return found


def test_no_numpy_channel_swap_in_the_package():
    root = pathlib.Path(detail.__file__).resolve().parent
    found = _offenders(root)
    assert not found, (
        "numpy channel swaps (use detail.core.visualize.bgr_to_rgb / rgb_to_bgr — "
        f"same pixels, ~45x cheaper per frame): {found}"
    )


def test_the_scan_catches_every_spelling_of_a_channel_swap():
    """The negative case: each spelling a person writes is reported."""
    planted = textwrap.dedent("""
        a = np.ascontiguousarray(bgr[:, :, ::-1])
        b = frame[:, :, ::-1].copy()
        c = frame[..., ::-1]
        d = frame[:, :, [2, 1, 0]]
        e = frame[:, :, ::-1]
        """)
    assert _offenders_in(planted, "planted") == [
        f"planted:{n}" for n in (2, 3, 4, 5, 6)
    ]


def test_the_scan_leaves_other_reversals_alone():
    """A flip, an xy swap, a reversed vector and the helpers themselves are not channel swaps."""
    clean = textwrap.dedent('''
        """frame[:, :, ::-1].copy() costs 45x more"""  # a docstring quoting the rule
        flipped = image[:, ::-1]
        yx = points[:, ::-1]
        order = np.argsort(values)[::-1]
        rgb = bgr_to_rgb(bgr)
        last = frame[:, :, 0]
        ''')
    assert _offenders_in(clean, "clean") == []
