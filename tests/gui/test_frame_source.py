"""The one classification of "this scrub position produced no pixels"
(``detail/gui/frame_source.py``), and the scan that keeps every scrubber decoder on it.

**The class.** A frame-preview surface that cannot produce pixels renders an empty pane
instead of naming why, and where the surface also captures input, accepts that input
against a frame nobody saw.

``storage/video.py::read_frame_at`` already separates the two ways a scrub position
yields no picture, expressly "so previews can show a clear message":
``FileNotFoundError`` for a video-excluded export (bring the video here; nothing in this
recording will decode) and ``None`` for an undecodable position (every other position may
be fine). Ten of the twelve GUI scrub surfaces threw that away, three ways:

- ``FrameScrubber`` drops a ``_DecodeFailed`` payload without emitting, so a decoder that
  *raised* reached no display at all.
- A ``_display`` written as ``if rgb is not None: setPixmap(...)`` has no ``else``, so a
  decoder that *returned nothing* left the pane untouched.
- ``analysis_query_eval`` rendered a failed decode as "(no detection here)", a different
  fact; arena left the **previous** frame up, so the next polygon went onto a frame the
  user was not looking at.

**Why it stayed hidden.** Two surfaces handled it correctly and hand-rolled the same
``try``/``None`` pair with the same literal strings. Correct-looking precedent in the
codebase is not coverage: nobody counted the surfaces that lacked it.

**The input half.** ``ClipExampleDialog`` gated marking on ``track_continuity``, which
reads the embedding zarr — and a video-excluded export carries that in full. Derived data
is not a presence check for its source, so marking unlocked over a pane that never
rendered and committed example clips to frames nobody saw. It now resolves
``Recording.video_available`` (off-thread, for the selected recording only, never a sweep
over every recording in the picker) and refuses **in the handler**, not only by greying
the button. Arena is the same shape: gating its entry points on ``video_available`` never
covered an undecodable frame, which happens with the video present.

**The rule.** Classify every scrubber decode through ``frame_source.decode`` and render
the reason with ``unavailable_text``. Never hand-roll the pair per surface. The scan
below is data-driven over ``set_decoder`` targets, so a new scrub surface is covered the
day it is added.
"""

import ast
import pathlib

from detail.api.run_preview import SubjectNotDetectedError
from detail.gui import frame_source

_GUI = pathlib.Path(frame_source.__file__).parent

# Empty on purpose: every scrubber decoder classifies. An entry here needs a written
# reason, because the default is that a decoder routes through.
ALLOWED: set[tuple[str, str]] = set()


def _decoder_names(tree: ast.AST) -> set[str]:
    """The ``self._x`` method names handed to ``set_decoder`` in one module."""
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "set_decoder"):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
            names.add(arg.attr)
    return names


def _violations(root: pathlib.Path) -> set[tuple[str, str]]:
    """Every ``set_decoder`` target whose body never classifies a failed decode.

    A name can be defined more than once in a module (each dialog has its own
    ``_compute``); *all* of them must route through, so one raw definition is a
    violation even when a sibling is correct.
    """
    found = set()
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        wanted = _decoder_names(tree)
        if not wanted:
            continue
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in wanted:
                continue
            if "frame_source.decode" not in ast.get_source_segment(source, node):
                found.add((rel, node.name))
    return found


def test_decode_names_a_missing_video():
    def gone():
        raise FileNotFoundError("video not present: /nope.mp4")

    value, reason = frame_source.decode(gone)
    assert value is None
    assert reason == frame_source.MISSING
    assert "not in this export" in frame_source.unavailable_text(reason)


def test_decode_names_an_undecodable_position():
    for producer in (lambda: None, _raise_index_error):
        value, reason = frame_source.decode(producer)
        assert value is None
        assert reason == frame_source.UNDECODABLE
        assert "could not decode" in frame_source.unavailable_text(reason)


def test_decode_names_a_subject_that_was_not_detected():
    """The frame decoded and tracking had no box there — not a decode failure, so not called one.

    Both are ``LookupError``s, so the order of the handlers is the whole of this: caught as an
    ``IndexError`` it would read "could not decode" and send the user to check a video that is
    fine.
    """

    def lost():
        raise SubjectNotDetectedError("track 0 has no detection at frame 9")

    value, reason = frame_source.decode(lost)
    assert value is None
    assert reason == frame_source.NOT_DETECTED
    text = frame_source.unavailable_text(reason)
    assert "not detected" in text and "decode" not in text


def _raise_index_error():
    raise IndexError("could not decode frame 9 from part 0")


def test_decode_passes_a_real_frame_through_untouched():
    frame = object()
    value, reason = frame_source.decode(lambda: frame)
    assert value is frame and reason is None


def test_a_falsy_but_real_value_is_not_mistaken_for_a_failure():
    """An empty array is a decoded frame, not an absent one — ``decode`` tests ``is
    None``, never truthiness."""
    value, reason = frame_source.decode(lambda: [])
    assert value == [] and reason is None


def test_every_scrub_decoder_classifies_a_failed_decode():
    new = _violations(_GUI) - ALLOWED
    assert not new, (
        "a scrubber decoder that does not route through frame_source.decode — a frame "
        "it cannot produce becomes a blank pane instead of a reason: " + repr(new)
    )


def test_the_scan_would_catch_a_raw_decoder(tmp_path):
    """Guard the guard: a scan that matches nothing passes forever."""
    (tmp_path / "raw_surface.py").write_text(
        "class V:\n"
        "    def build(self):\n"
        "        self._scrubber.set_decoder(self._compute)\n"
        "    def _compute(self, frame):\n"
        "        return self._preview.frame(frame)\n",
        encoding="utf-8",
    )
    assert _violations(tmp_path) == {("raw_surface.py", "_compute")}
