"""Shared helpers for building a YOLO dataset — used by every dataset source: the run-export
source (`dataset_export.py`), the fresh SAM3-labeling source (`labeling.py`) and the keypoint
distill's label stage (`keypoint_dataset.py`). Everything below the labels — frame reading,
train/val split, image+label writing — is written once here (§12, "built once, shared across
sources").

**A split is assigned to the examples that were written, never to the frames that were asked
for.** A source decides which frames to try before it knows which of them yield an example: a
frame can carry no label, fail to decode exactly, or be cut off by a stop. Splitting the
requested frames first and dropping some afterwards left the realised validation share
wherever the drops happened to fall. :class:`StagedExamples` is the one way a source writes:
examples are staged, and the split is drawn over what was kept when the source commits.
"""

import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from detail.core import atomic
from detail.storage.video import read_frame_at

#: Beside ``images/`` while a source runs: examples whose split is not drawn yet. Gone once the
#: source commits; left behind only by a crash, inside a dataset marked failed.
STAGING_DIR = "staging"


def make_split_dirs(run_dir: Path) -> None:
    """Create the ``images/{train,val}`` + ``labels/{train,val}`` tree under ``run_dir``."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def sequential_frames(video: Path, targets: list[int]):
    """Yield ``(frame_idx, BGR frame)`` for each target, reading **forward only** (Tenet 11).

    Seeking H.264 can land on a neighbouring keyframe, so the extracted image would not match the
    frame index; reading start-to-finish and picking off targets guarantees frame-exactness.
    """
    if not targets:
        return
    cap = cv2.VideoCapture(str(video))
    try:
        targets = sorted(targets)
        ti, idx = 0, 0
        while ti < len(targets):
            ok, frame = cap.read()
            if not ok:
                break
            if idx == targets[ti]:
                yield idx, frame
                ti += 1
            idx += 1
    finally:
        cap.release()


def exact_sampled_frames(
    video: Path,
    targets: list[int],
    timestamps: np.ndarray,
    rng: random.Random,
    stats: dict | None = None,
):
    """Yield ``(frame_idx, BGR frame)`` for a sparse sample, each frame decoded **exactly**.

    Each target goes through ``read_frame_at`` with the part's timestamp map, so the picture is
    the ``frame_idx``-th presentation frame — the Tenet 11 labeling-sampler exception: a sampled
    frame's index is a label, not a join key, so it may seek, but it may not land somewhere else
    and keep the requested index (cv2's ``CAP_PROP_POS_FRAMES`` seek did both). Cost is one
    seek-and-decode per frame (~33 ms on a 1080p part), independent of video length.

    A target the decoder cannot land on exactly is **skipped and redrawn**: a fresh index from
    this part's untried frames, drawn from ``rng`` (so a seeded caller gets the same sample every
    time). Redraws stop after ``len(targets)`` misses, so a file that cannot be decoded at all
    ends rather than retrying forever. Every miss is counted in ``stats["undecodable"]``.
    """
    if not targets:
        return
    count = len(timestamps)
    queue = sorted(targets)
    tried = set(queue)
    misses = 0
    i = 0
    while i < len(queue):
        frame_idx = queue[i]
        i += 1
        decoded = read_frame_at(video, frame_idx, timestamps=timestamps)
        if decoded is not None and decoded[0] == frame_idx:
            yield frame_idx, decoded[1]
            continue
        misses += 1
        if stats is not None:
            stats["undecodable"] = stats.get("undecodable", 0) + 1
        if misses > len(targets) or len(tried) >= count:
            continue
        replacement = rng.choice([f for f in range(count) if f not in tried])
        tried.add(replacement)
        queue.append(replacement)


def assign_splits(by_recording: dict[str, list], val_split: float, seed: str) -> dict:
    """``{key: "train"|"val"}`` for the per-recording frame keys. The val fraction is drawn **per
    recording** (stratified) so no recording is missing from validation. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    split: dict = {}
    for keys in by_recording.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_val = int(round(len(keys) * val_split))
        for i, key in enumerate(keys):
            split[key] = "val" if i < n_val else "train"
    return split


def background_cap(n_labelled: int, share: float) -> int:
    """The most empty-label frames ``n_labelled`` labelled ones can carry so that background is at
    most ``share`` of the dataset: ``B / (L + B) <= share``."""
    if share <= 0 or n_labelled <= 0:
        return 0
    return int(math.floor(share * n_labelled / (1.0 - share) + 1e-9))


class StagedExamples:
    """A dataset's examples, written as they are labelled and split once the set is known.

    ``add`` writes an image + label into ``staging/``; ``commit`` decides which background
    frames to keep, draws the train/val split over the examples that remain (per recording,
    seeded — :func:`assign_splits`), and moves each into ``images/<split>`` / ``labels/<split>``.
    The realised validation share is therefore the configured one, whatever was skipped.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._staging = self._run_dir / STAGING_DIR
        make_split_dirs(self._run_dir)
        self._staging.mkdir(parents=True, exist_ok=True)
        self._items: list[tuple[str, object, str, bool]] = []
        self.n_background = 0

    def add(self, group: str, key, stem: str, frame, lines: list[str]) -> None:
        """Stage one image + its YOLO label. ``group`` is what the split is stratified over (the
        recording); ``key`` names the example in the split ``commit`` returns. No ``lines`` is
        a background frame: an empty label file, which Ultralytics trains on as one."""
        cv2.imwrite(str(self._staging / f"{stem}.jpg"), frame)
        (self._staging / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        self._items.append((group, key, stem, not lines))

    def commit(
        self, val_split: float, seed: str, background_share: float | None = None
    ) -> dict:
        """Keep at most ``background_share`` of the dataset as background (``None`` keeps every
        staged example — a source that stages no background, or decides it itself), split what
        is kept, and move it into place. Returns ``{key: "train"|"val"}`` for every kept
        example."""
        labelled = [item for item in self._items if not item[3]]
        background = sorted(
            (item for item in self._items if item[3]), key=lambda item: item[2]
        )
        cap = (
            len(background)
            if background_share is None
            else min(len(background), background_cap(len(labelled), background_share))
        )
        kept_background = random.Random(f"{seed}:background").sample(background, cap)
        kept_stems = {item[2] for item in kept_background}
        for item in background:
            if item[2] not in kept_stems:
                (self._staging / f"{item[2]}.jpg").unlink()
                (self._staging / f"{item[2]}.txt").unlink()
        kept = labelled + kept_background
        by_group: dict[str, list] = {}
        for group, key, _stem, _bg in kept:
            by_group.setdefault(group, []).append(key)
        split = assign_splits(by_group, val_split, seed)
        for _group, key, stem, _bg in kept:
            which = split[key]
            atomic.replace(
                self._staging / f"{stem}.jpg",
                self._run_dir / "images" / which / f"{stem}.jpg",
            )
            atomic.replace(
                self._staging / f"{stem}.txt",
                self._run_dir / "labels" / which / f"{stem}.txt",
            )
        shutil.rmtree(self._staging, ignore_errors=True)
        self.n_background = len(kept_background)
        return split
