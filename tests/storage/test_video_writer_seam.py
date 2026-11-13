"""Export video is written in one place, ``storage/video.open_video_writer`` (Tenet 15).

The class: a codec decision made at a call site. Every MP4 export once opened its own
``cv2.VideoWriter`` through this seam's ``avc1`` → ``mp4v`` fallback, and on the Linux cv2 wheel
(WSL included) that fallback fired every time, silently: every MP4 the tool wrote there was
MPEG-4 Part 2, which PowerPoint on Windows will not reliably play, and nothing said so. H.264 now
goes through PyAV's linked ``libx264``, the fallback is visible (``codec_note``, ``export.yaml``),
and the choice is one option on one writer, which only holds while no other module constructs a
writer of its own. A second ``cv2.VideoWriter`` would write ``mp4v`` whatever the user chose; a
second PyAV output would encode without the profile, pixel format and ``+faststart`` that make
the file play.

Tenet 15's other half is unchanged: nothing runs an ``ffmpeg`` executable. Encoding through
PyAV is a library call into linked libav, which the amended tenet allows; a ``subprocess`` or
``os.system`` naming ``ffmpeg`` is still a bug, and this scan rejects it anywhere in ``detail/``.

Fail-closed source scan, the ``test_path_serialization.py`` idiom: the shape is invisible to a
behavioural test, because a second writer works.
"""

import ast
from pathlib import Path

import detail

PACKAGE = Path(detail.__file__).resolve().parent
SEAM = PACKAGE / "storage" / "video.py"


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else ""
        return f"{owner}.{func.attr}" if owner else func.attr
    return getattr(func, "id", "")


def _mode(call: ast.Call) -> str | None:
    if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
        return call.args[1].value
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _names_ffmpeg(call: ast.Call) -> bool:
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip().lower().startswith("ffmpeg")
        for node in ast.walk(call)
    )


def offenders(source: str) -> list[str]:
    """What in ``source`` writes video outside the seam, or shells out to FFmpeg."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name.endswith("VideoWriter") or name.endswith("VideoWriter_fourcc"):
            found.append(f"line {node.lineno}: {name}")
        elif name == "av.open" and _mode(node) == "w":
            found.append(f"line {node.lineno}: av.open(..., 'w')")
        elif name.split(".")[-1] in (
            "run",
            "Popen",
            "call",
            "check_call",
            "check_output",
            "system",
        ) and _names_ffmpeg(node):
            found.append(f"line {node.lineno}: {name} runs ffmpeg")
    return found


def test_only_the_seam_writes_video_and_nothing_runs_ffmpeg():
    bad = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        found = offenders(path.read_text(encoding="utf-8-sig"))
        if path == SEAM:
            found = [f for f in found if "ffmpeg" in f]
        if found:
            bad[str(path.relative_to(PACKAGE))] = found
    assert not bad, (
        "video is written outside storage/video.open_video_writer, or ffmpeg is run "
        f"(Tenet 15): {bad}"
    )


def test_the_seam_itself_holds_both_writers():
    """The scan's exemption excuses something: the two writers really are in the seam."""
    found = offenders(SEAM.read_text(encoding="utf-8-sig"))
    assert any("VideoWriter" in f for f in found)
    assert any("av.open" in f for f in found)


def test_the_scan_rejects_each_shape():
    """The negative case: each way around the seam is reported, and a read is not."""
    assert offenders("w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 30, s)")
    assert offenders("c = av.open(str(p), 'w')")
    assert offenders("c = av.open(str(p), mode='w', format='mp4')")
    assert offenders("subprocess.run(['ffmpeg', '-i', src, dst])")
    assert offenders("os.system('ffmpeg -i a.mp4 b.mp4')")
    assert not offenders("c = av.open(str(p))")
    assert not offenders("subprocess.run(['git', 'status'])")
