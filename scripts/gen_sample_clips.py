"""Generate the small sample video clips the docs screenshots are rendered from.

``scripts/gen_docs_screens.py more-ways`` builds a small experiment from a handful of tiny
clips committed under ``docs/assets/sample-data/``, for the user guide's steps on a recording
in several files and on importing a bundle.

This script clips those fixtures from a full recording, so the committed videos
stay small (a few seconds, downscaled) and are reproducible from a known source.
It reads the source sequentially from frame 0 (no seeking) and writes each clip
with cv2 (no FFmpeg), matching the pipeline's decode discipline.

Usage::

    python scripts/gen_sample_clips.py --source /path/to/recording.mp4

The source is any behavioural recording. Pass ``--out`` to write elsewhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

_OUT_DEFAULT = (
    Path(__file__).resolve().parent.parent / "docs" / "assets" / "sample-data"
)
_WIDTH = 160
_HEIGHT = 120
_FPS = 30.0

# (filename, first frame, frame count). boxA_am / boxA_pm are two parts of one
# recording, so they share resolution and fps; boxB is a separate single-part
# recording. Consecutive frame ranges let us read the source in one pass.
_CLIPS = [
    ("boxA_am.mp4", 0, 120),
    ("boxA_pm.mp4", 120, 120),
    ("boxB.mp4", 240, 120),
]


def _write_clips(source: Path, out_dir: Path) -> list[Path]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open source video: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    written = []
    last_frame = max(start + count for _, start, count in _CLIPS)
    index = 0
    for name, start, count in _CLIPS:
        path = out_dir / name
        writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_WIDTH, _HEIGHT))
        while index < start + count:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"source ran out at frame {index}; need {last_frame} frames"
                )
            if index >= start:
                writer.write(cv2.resize(frame, (_WIDTH, _HEIGHT)))
            index += 1
        writer.release()
        written.append(path)
    capture.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="full recording to clip from"
    )
    parser.add_argument(
        "--out", type=Path, default=_OUT_DEFAULT, help="output directory"
    )
    args = parser.parse_args()
    for path in _write_clips(args.source, args.out):
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
