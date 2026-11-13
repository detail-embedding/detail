"""A cached video reader lets go of the file when nobody is asking for frames.

**The class: a cached reader holds an OS file handle past the operation that needed it.** On
POSIX that is invisible — an open file can be renamed and unlinked under its reader. On Windows
an open handle pins the file: it cannot be moved, renamed or deleted until the handle closes.

**How it hid.** `FrameCursor` (#337) holds a PyAV container open so playback walks forward at
1.48 ms a frame instead of re-seeking at 33.5 — and `Recording` and `DetectionPreview` each keep
one for their own lifetime. Opening a recording page decodes frame 0 for the geometry preview, so
from then on the user could not move, rename or delete the page's video in Explorer while the
page stayed open: the exact move *Locate video* is built around. The suite's own test of that
flow (`tests/gui/test_pages.py::test_locate_button_enables_only_when_a_referenced_video_is_missing`)
failed 3/3 on Windows with WinError 32 and was green on Linux, where CI runs, because
`unlink` there succeeds on an open file. A test that *tries the delete* can only ever catch this
on the platform it affects.

**The rule.** A reader that outlives one operation holds its file only while it is in use:
`FrameCursor` closes its container `IDLE_CLOSE_S` after the last request and reopens on the next.
The release lives in the cursor rather than in each owner, so every holder inherits it; a new
long-lived holder of a raw `av.open` / `cv2.VideoCapture` is refused by the source scan below and
has to go through the cursor.

**The guard asks the OS, not the delete** (`tests/file_handles.py`): on Linux that is
`/proc/self/fd` through psutil, so the Windows failure is visible on a box that cannot have it.
Each behavioural test has its negative: the probe reports an open cursor as held, and the
deterministic idle test shows a cursor still in use is *not* released.
"""

from __future__ import annotations

import ast
import functools
import time
from pathlib import Path

import pytest

from detail.storage import recording as recording_store
from detail.storage import video
from tests.file_handles import file_is_held
from tests.storage.test_video import FPS
from tests.storage.test_video import _write_distinct_video

REPO = Path(__file__).resolve().parents[2]
DETAIL = REPO / "detail"

#: A failsafe, never a budget: a release takes ``IDLE_CLOSE_S`` plus a timer wake-up.
_RELEASE_TIMEOUT_S = 10.0


def _wait_released(path: Path) -> bool:
    deadline = time.monotonic() + _RELEASE_TIMEOUT_S
    while time.monotonic() < deadline:
        if not file_is_held(path):
            return True
        time.sleep(0.02)
    return not file_is_held(path)


def _cursor_on(tmp_path, n_frames=12):
    p = tmp_path / "v.mp4"
    _write_distinct_video(p, n_frames)
    return p, video.FrameCursor(p, timestamps=video.generate_timestamps(p, FPS))


# -- the probe ------------------------------------------------------------------------------- #
def test_the_probe_sees_a_cursor_holding_its_file(tmp_path):
    """Guard the guard: every release assertion below is "the probe says free", which a probe
    that always says free would pass."""
    p, cursor = _cursor_on(tmp_path)
    assert not file_is_held(p), "nothing has opened the file yet"
    try:
        assert cursor.frame(0)[0] == 0
        assert cursor.is_open
        assert file_is_held(p), (
            "an open cursor's video read as not held — the probe cannot see the handle this "
            "module exists to release, so every test here would pass with the bug in place"
        )
    finally:
        cursor.close()
    assert not file_is_held(p), "and close() lets go"


# -- the mechanism --------------------------------------------------------------------------- #
def test_an_idle_cursor_lets_go_of_its_file_and_reopens_on_the_next_request(
    tmp_path, monkeypatch
):
    """The real timer, end to end: decode, stop asking, and the file is free to move."""
    monkeypatch.setattr(video.FrameCursor, "IDLE_CLOSE_S", 0.05)
    p, cursor = _cursor_on(tmp_path)
    assert cursor.frame(3)[0] == 3

    assert _wait_released(p), (
        f"the video was still held {_RELEASE_TIMEOUT_S}s after the last request — on Windows "
        "the user cannot move, rename or delete it while the surface stays open"
    )
    assert not cursor.is_open
    assert cursor.frame(4)[0] == 4, "a released cursor reopens, at the frame asked for"
    cursor.close()


def test_a_cursor_in_use_keeps_its_file_and_one_timer(tmp_path):
    """Deterministic: a fake clock, and the timer replaced by a recorder.

    Playback is the reason the cursor exists, so the release must not fire *during* it — and
    must not cost a thread per frame either. One pending release while the container is open,
    re-armed for the remainder when it finds the cursor recently used.
    """
    p, cursor = _cursor_on(tmp_path)
    now = [100.0]
    armed: list[float] = []
    cursor._clock = lambda: now[0]
    cursor._arm_idle_release = lambda delay: (
        armed.append(delay),
        setattr(cursor, "_idle_timer", object()),
    )
    idle = cursor.IDLE_CLOSE_S
    try:
        for i in range(8):  # playback: one request after another
            now[0] += 0.016
            assert cursor.frame(i)[0] == i
        assert armed == [idle], f"{len(armed)} timers for 8 frames; one is the design"

        now[0] += idle / 2
        cursor._release_if_idle()
        assert cursor.is_open, "released while still inside the idle window"
        assert armed[-1] == pytest.approx(idle / 2), "re-armed for the remainder"

        now[0] += idle / 2 + 0.001
        cursor._release_if_idle()
        assert (
            not cursor.is_open
        ), "idle past the window, and still holding the container"
        assert not file_is_held(p)
    finally:
        cursor._idle_timer = None  # the recorder's placeholder has nothing to cancel
        cursor.close()


# -- every holder ---------------------------------------------------------------------------- #
def _experiment_recording(tmp_path):
    from detail.api.experiment import Experiment

    src = tmp_path / "src.mp4"
    _write_distinct_video(src, 12)
    recording = Experiment.create("p", tmp_path / "exp").create_recording("s1")
    recording.add_video_part(src)
    data = recording_store.read_recording(recording.dir)
    path = recording_store.resolve_part_video_path(
        recording.dir, data["video"]["files"][0]
    )
    return recording, path


def _decode_through_recording(tmp_path):
    recording, path = _experiment_recording(tmp_path)
    recording.get_frame(0, 2)
    return recording, path


def _decode_through_detection_preview(tmp_path):
    recording, path = _experiment_recording(tmp_path)
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [{"id": "det_01", "status": "complete"}]
    recording_store.write_recording(recording.dir, data)
    preview = recording.detection_preview("det_01")
    assert preview.frame(0, 2) is not None
    return preview, path


def _decode_through_track_preview_window(tmp_path):
    from types import SimpleNamespace

    from detail.gui.keypoint_track_preview import _WindowFrames

    recording, path = _experiment_recording(tmp_path)
    data = recording_store.read_recording(recording.dir)
    t_ns = recording_store.load_active_timestamps(recording.dir, data, 0).values
    window = SimpleNamespace(
        recording_id=recording.id,
        part_index=0,
        start_ns=int(t_ns[0]),
        end_ns=int(t_ns[-1]),
    )
    frames = _WindowFrames(SimpleNamespace(dir=recording.experiment_dir), window)
    assert frames.read(frames.first_frame + 1) is not None
    return frames, path


def _decode_through_preview_clip(tmp_path):
    from detail.api import previews

    recording, _source = _experiment_recording(tmp_path)
    recording.build_preview()
    reader = previews.PreviewReader.open(recording, 0)
    clip = reader.clip(None, 0, 10**9, 32)
    assert clip.frame_at(2).shape == (32, 32, 3)
    return clip, reader.frames.video


#: Every class that keeps a `FrameCursor` on itself, keyed as the scan below finds them. A new
#: holder fails `test_every_cursor_holder_is_exercised` until it has a row here.
HOLDERS = {
    ("detail/api/recording.py", "Recording"): _decode_through_recording,
    (
        "detail/api/run_preview.py",
        "DetectionPreview",
    ): _decode_through_detection_preview,
    (
        "detail/gui/keypoint_track_preview.py",
        "_WindowFrames",
    ): _decode_through_track_preview_window,
    ("detail/api/previews.py", "PreviewClip"): _decode_through_preview_clip,
}


@pytest.mark.parametrize("holder", sorted(HOLDERS), ids=lambda h: h[1])
def test_a_holder_lets_go_of_the_video_while_it_is_still_alive(
    holder, tmp_path, monkeypatch
):
    """Through the owner's own read path, with the owner kept alive — the open page's case.

    Not "after the owner is collected": a page holds its `Recording` for as long as it is open,
    and that is when the user goes to move the file.
    """
    monkeypatch.setattr(video.FrameCursor, "IDLE_CLOSE_S", 0.05)
    owner, path = HOLDERS[holder](tmp_path)
    assert _wait_released(
        path
    ), f"{holder[1]} still holds {path.name} {_RELEASE_TIMEOUT_S}s after its last decode"
    assert (
        owner is not None
    )  # alive to here, so the release was not the garbage collector


# -- the source scan ------------------------------------------------------------------------- #
_HANDLE_CALLS = {("av", "open"), ("cv2", "VideoCapture")}


def _is_call_to(node, names: set[tuple[str, str]]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr) in names
        and not _opens_for_writing(node)
    )


def _opens_for_writing(call: ast.Call) -> bool:
    """``av.open(path, "w")``: the export writer's output, which its one export releases
    (``storage.video.open_video_writer``), not a reader cached past the read."""
    mode = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    return isinstance(mode, ast.Constant) and mode.value == "w"


def _is_frame_cursor_call(node) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name == "FrameCursor"


def _stored(target) -> bool:
    """A target that outlives the function: an attribute or a container slot."""
    return isinstance(target, (ast.Attribute, ast.Subscript)) or (
        isinstance(target, ast.Tuple) and any(_stored(t) for t in target.elts)
    )


def _escaping_handles(tree: ast.AST, exempt_class: str | None = "FrameCursor"):
    """``(lineno, how)`` for every raw reader handle that escapes the function opening it.

    Escaping means stored on an attribute or into a container, or returned — directly, or
    through the local it was bound to. A handle used and released in its own function, the
    shape of every ``with av.open`` and ``cap = cv2.VideoCapture`` in the package, is fine.
    """
    found = []

    def visit_function(fn):
        locals_: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and _is_call_to(node.value, _HANDLE_CALLS):
                for target in node.targets:
                    if _stored(target):
                        found.append((node.lineno, "stored"))
                    elif isinstance(target, ast.Name):
                        locals_.add(target.id)
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None:
                if _is_call_to(node.value, _HANDLE_CALLS) or (
                    isinstance(node.value, ast.Name) and node.value.id in locals_
                ):
                    found.append((node.lineno, "returned"))
            if isinstance(node, ast.Assign) and any(_stored(t) for t in node.targets):
                names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                if names & locals_:
                    found.append((node.lineno, "stored via a local"))

    def walk(node, in_exempt: bool):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, in_exempt or child.name == exempt_class)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not in_exempt:
                    visit_function(child)
            else:
                walk(child, in_exempt)

    walk(tree, False)
    return found


@functools.cache
def _parsed() -> tuple[tuple[str, ast.Module], ...]:
    """Every module under ``detail/``, parsed once for both scans."""
    return tuple(
        (
            path.relative_to(REPO).as_posix(),
            ast.parse(path.read_text(encoding="utf-8-sig")),
        )
        for path in sorted(DETAIL.rglob("*.py"))
    )


def test_no_raw_reader_handle_outlives_the_function_that_opened_it():
    """Fail-closed: a long-lived reader goes through `FrameCursor`, which knows to let go."""
    offenders = [
        f"{rel}:{line} ({how})"
        for rel, tree in _parsed()
        for line, how in _escaping_handles(tree)
    ]
    assert not offenders, (
        "a raw av.open / cv2.VideoCapture handle escapes the function that opened it, so "
        "nothing closes it when the caller stops reading — on Windows the file then cannot be "
        "moved or deleted. Hold a storage.video.FrameCursor instead:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source",
    [
        "class S:\n    def f(self, p):\n        self._cap = cv2.VideoCapture(p)\n",
        "class S:\n    def f(self, p):\n        c = av.open(p)\n        self._c, self._s = c, 1\n",
        "def f(p):\n    return cv2.VideoCapture(p)\n",
        "def f(p, cache):\n    cache[p] = av.open(p)\n",
    ],
    ids=["stored", "stored-via-local", "returned", "cached"],
)
def test_the_scan_refuses_each_escape(source):
    assert _escaping_handles(ast.parse(source))


def test_a_stored_writer_is_not_a_reader_but_a_stored_reader_still_is():
    """The writer exemption is exactly ``mode "w"``: the same store of a read handle is
    still refused."""
    store = "class W:\n    def f(self, p):\n        self._c = {}\n"
    writer = store.format("av.open(p, 'w')")
    by_keyword = store.format("av.open(p, mode='w')")
    reader = store.format("av.open(p, 'r')")
    assert not _escaping_handles(ast.parse(writer))
    assert not _escaping_handles(ast.parse(by_keyword))
    assert _escaping_handles(ast.parse(reader))


def test_the_scan_would_refuse_the_cursor_itself():
    """The exemption is doing work: `FrameCursor` stores its container through a local, the
    indirect form, and the scan has to be able to see that form on real code."""
    text = (DETAIL / "storage" / "video.py").read_text(encoding="utf-8-sig")
    assert _escaping_handles(ast.parse(text), exempt_class=None)
    assert not _escaping_handles(ast.parse(text))


def _cursor_holders():
    holders = set()
    for rel, tree in _parsed():
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for node in ast.walk(cls):
                if (
                    isinstance(node, ast.Assign)
                    and _is_frame_cursor_call(node.value)
                    and any(_stored(t) for t in node.targets)
                ):
                    holders.add((rel, cls.name))
    return holders


def test_every_cursor_holder_is_exercised():
    """The table above is driven by the source, so a holder cannot be added without a test."""
    found = _cursor_holders()
    assert found, "the scan found no FrameCursor holder at all — it is matching nothing"
    assert found == set(HOLDERS), (
        f"FrameCursor holders in the source {sorted(found)} differ from the ones tested "
        f"{sorted(HOLDERS)}: give each a row in HOLDERS (or drop a row whose holder is gone)"
    )
