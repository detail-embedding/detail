"""The clip window's **Aligned** turn, its keypoints and its view as one picture (§22).

- **One turn** — :func:`turn_matrix` (with :func:`turn_picture` / :func:`turn_mask` /
  :func:`turn_points`, and :func:`mark_no_axis` for a frame with no body axis) — is what every
  surface that draws a clip aligned uses: the clip window (``ClipQueryPreview.clip_views``'
  ``aligned``) and the live montage's tiles (``api.previews``, ``api.clip_tiles``).
- :class:`ClipPoints` reads the keypoints the embedding read, for the turn's axis and the tiles'
  points; :func:`pose_notes` / :func:`alignment_note` say, in the window's words, why a run has
  no keypoints or no axis.
- :func:`view_picture` lays one frame out as the window shows it, for the exports
  (``api.clip_pane_export``).

The clip window's pictures themselves are drawn by ``api.run_preview`` (``ClipQueryPreview.
clip_views``); main's picture layer that stood here beside it (a ``ClipFrame`` read and three pure
drawers) lost its callers when the merge kept remote-detail's window, and was removed.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

from detail.analysis import body_axis
from detail.analysis.body_axis import BodyAxis
from detail.core import visualize
from detail.core.skeleton import Skeleton

if TYPE_CHECKING:
    from detail.api.run_preview import ClipViews

# --- the aligned orientation: one turn, every picture --------------------------------------- #
#: The mark on a picture drawn as recorded because its frame gives no body axis.
NO_AXIS_MARK = "no axis"
_NO_AXIS_TAG = (40, 40, 40)


def turn_matrix(
    front: np.ndarray, back: np.ndarray, side: float, out_wh: tuple[int, int]
) -> np.ndarray:
    """The ``2 × 3`` affine (``cv2.warpAffine``'s forward map) that turns a picture about the
    body's middle so the axis from ``back`` to ``front`` points straight up — the direction the
    clip window's **Align pose** faces — puts that middle at the centre of an ``out_wh``
    picture, and scales a ``side``-pixel span to the output's height.

    The montage passes its crop's side (a tile shows as much around the animal as it does as
    recorded); the clip window passes the frame's height, which turns without scaling.
    """
    front = np.asarray(front, dtype=np.float64)
    back = np.asarray(back, dtype=np.float64)
    centre = (front + back) / 2.0
    dx, dy = front - back
    angle = -np.pi / 2 - np.arctan2(
        dy, dx
    )  # onto (0, -1): screen up, image y points down
    scale = float(out_wh[1]) / max(float(side), 1e-6)
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    linear = np.array([[c, -s], [s, c]])
    shift = np.array([out_wh[0] / 2.0, out_wh[1] / 2.0]) - linear @ centre
    return np.hstack([linear, shift[:, None]])


def turn_picture(
    picture: np.ndarray, matrix: np.ndarray, out_wh: tuple[int, int], fill
) -> np.ndarray:
    """``picture`` through ``matrix`` into an ``out_wh`` picture; ``fill`` where the turned
    frame does not reach (black for footage, white for the panes)."""
    return cv2.warpAffine(
        picture,
        matrix,
        (int(out_wh[0]), int(out_wh[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(v) for v in fill),
    )


def turn_mask(
    mask: np.ndarray, matrix: np.ndarray, out_wh: tuple[int, int], *, at=(0, 0)
) -> np.ndarray | None:
    """A bool ``mask`` through ``matrix``, or ``None`` when nothing of it lands in the output.
    ``at`` is ``(y0, x0)``: where a cropped mask (a trail layer) sits in the picture."""
    y0, x0 = int(at[0]), int(at[1])
    moved = matrix.copy()
    moved[:, 2] += moved[:, :2] @ np.array([x0, y0], dtype=np.float64)
    turned = cv2.warpAffine(
        mask.astype(np.uint8) * 255,
        moved,
        (int(out_wh[0]), int(out_wh[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out = turned > 127
    return out if out.any() else None


def turn_points(
    points: np.ndarray, matrix: np.ndarray, picture_wh, out_wh
) -> np.ndarray:
    """Normalized ``(K, 2)`` points of a ``picture_wh`` picture, through ``matrix``, as
    normalized points of the ``out_wh`` output (NaN stays NaN)."""
    px = np.asarray(points, dtype=np.float64) * np.array(picture_wh, dtype=np.float64)
    moved = px @ matrix[:, :2].T + matrix[:, 2]
    return moved / np.array(out_wh, dtype=np.float64)


def mark_no_axis(picture: np.ndarray) -> np.ndarray:
    """``picture`` with a small "no axis" box in its top-left corner: drawn as recorded
    because this frame gives no body axis — said, rather than left blank or turned wrongly.
    """
    out = picture.copy()
    scale = max(out.shape[0] / 700.0, 0.32)
    visualize.draw_text_label(
        out, NO_AXIS_MARK, (0, 0), _NO_AXIS_TAG, scale=scale, above=False
    )
    return out


def pose_notes(
    experiment_dir: Path | str, embeds: dict
) -> tuple[str | None, str | None]:
    """``(keypoints, aligned)``: ``None`` each when the clips of these
    ``{recording_id: embed_run_id}`` can have their keypoints drawn / be drawn **Aligned**,
    else why not, in the clip window's words (``ClipPoints.note`` / ``axis_note``): an
    embedding that read no keypoints, a keypoint run that cannot be read, no heading definition
    that fits, several that do. The first recording that cannot answers for all: a montage
    mixes them. Reads each recording's embedding record and keypoint run — call it off the UI
    thread."""
    from detail.api.run_preview import _embedding_meta
    from detail.storage import experiment as experiment_store

    if not embeds:
        nothing = "There are no clips here."
        return nothing, nothing
    points_note = align_note = None
    for recording_id, embed_run_id in sorted(embeds.items()):
        recording_dir = experiment_store.recording_path(experiment_dir, recording_id)
        try:
            meta = _embedding_meta(recording_dir, embed_run_id)
        except (ValueError, FileNotFoundError) as exc:
            why = f"The embedding run of {recording_id} cannot be read: {exc}"
            return why, why
        points = ClipPoints.of(Path(experiment_dir), recording_dir, meta, embed_run_id)
        points_note = points_note or points.note
        align_note = align_note or points.axis_note
        if points_note and align_note:
            break
    return points_note, align_note


def alignment_note(experiment_dir: Path | str, embeds: dict) -> str | None:
    """``None`` when these clips can be drawn **Aligned**, else why not
    (:func:`pose_notes`' second half)."""
    return pose_notes(experiment_dir, embeds)[1]


# --- the keypoints a run read ------------------------------------------------------------ #
class ClipPoints:
    """The keypoint run an embedding run read, held for drawing: each (subject, part) series
    is read whole the first time a frame of it is asked for, so playback looks a frame up
    rather than reopening a store per frame.

    It also holds the **heading definition** the body's axis is measured by (which points are
    the front and which the back): the embedding run's own when it was aligned, else the
    experiment's one saved definition that fits this skeleton. Several that fit is a choice
    this window does not make for you, so it uses none and ``axis_note`` says why.
    """

    def __init__(
        self,
        run,
        note: str | None,
        part_stems: list[str],
        *,
        frame_wh: tuple[int, int] = (0, 0),
        definition: dict | None = None,
        axis_note: str | None = None,
        cutoff: float | None = None,
    ) -> None:
        self.run = run
        self.note = note
        self.part_stems = part_stems
        self.frame_wh = (int(frame_wh[0]), int(frame_wh[1]))
        self.definition = definition
        self.axis_note = note if run is None else axis_note
        self.cutoff = cutoff
        self._series: dict[tuple[int, str], tuple | None] = {}
        self._typical: dict[int, float] = {}
        self._lock = threading.Lock()

    @property
    def skeleton(self) -> Skeleton | None:
        return None if self.run is None else self.run.skeleton

    @property
    def axis_columns(self) -> tuple[list[int], list[int], list[int]] | None:
        """``(front, back, middle)`` skeleton columns, or None without a definition."""
        if self.run is None or self.definition is None:
            return None
        names = tuple(self.skeleton.names)
        front, back = body_axis.definition_columns(self.definition, names)
        return front, back, body_axis.midline_columns(self.skeleton, self.definition)

    def axis(self, track_id: int) -> BodyAxis | None:
        """The subject's body axis for the curves and the aligned pose, or ``None`` without a
        heading definition (``axis_note`` says why)."""
        columns = self.axis_columns
        if columns is None:
            return None
        front, back, middle = columns
        return BodyAxis(
            front=tuple(front),
            back=tuple(back),
            middle=tuple(middle),
            frame_wh=self.frame_wh,
            cutoff=self.cutoff,
            typical_length_px=self.typical_length(int(track_id)),
        )

    @classmethod
    def of(
        cls, experiment_dir: Path, recording_dir: Path, meta: dict, embed_run_id: str
    ) -> "ClipPoints":
        """For one embedding run, from its record ``meta`` (``embedding.yaml``)."""
        from detail.storage import recording as recording_store
        from detail.storage.keypoint_runs import KeypointTrackRun

        data = recording_store.read_recording(recording_dir)
        stems = [
            recording_store.part_stem(f) for f in recording_store.video_files(data)
        ]
        resolution = (data.get("video") or {}).get("resolution") or (0, 0)
        run_id = str(meta.get("keypoint_run_id") or "")
        if not run_id:
            return cls(
                None,
                f"Embedding run {embed_run_id} used no keypoints, so there is no pose to "
                "show. An embedding aligned by, or drawing, a keypoint track run carries one.",
                stems,
            )
        try:
            run = KeypointTrackRun(recording_dir, run_id)
        except (FileNotFoundError, ValueError) as exc:
            return cls(
                None,
                f"The keypoint run embedding run {embed_run_id} used ({run_id}) cannot be "
                f"read: {exc}",
                stems,
            )
        definition, axis_note = heading_definition_for(
            experiment_dir, meta, tuple(run.skeleton.names)
        )
        cutoff = meta.get("heading_cutoff")
        if cutoff is None:
            cutoff = meta.get("pose_min_score")
        return cls(
            run,
            None,
            stems,
            frame_wh=(int(resolution[0]), int(resolution[1])),
            definition=definition,
            axis_note=axis_note,
            cutoff=None if cutoff is None else float(cutoff),
        )

    def window(
        self, track_id: int, part_frame_counts: list[int], first: int, last: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(points (n, K, 2), scores (n, K))`` for one subject over the inclusive global
        frames ``[first, last]``, NaN on a frame it has no keypoints."""
        n = int(last) - int(first) + 1
        k = len(self.skeleton.names)
        points = np.full((n, k, 2), np.nan, dtype=np.float64)
        scores = np.full((n, k), np.nan, dtype=np.float64)
        start = 0
        for stem, count in zip(self.part_stems, part_frame_counts):
            lo, hi = max(int(first), start), min(int(last), start + int(count) - 1)
            if lo <= hi:
                series = self._load(int(track_id), stem)
                if series is not None:
                    frames, pts, scs = series
                    keep = (frames >= lo - start) & (frames <= hi - start)
                    at = frames[keep].astype(np.int64) + start - int(first)
                    points[at] = pts[keep]
                    scores[at] = scs[keep]
            start += int(count)
        return points, scores

    def rows_at(
        self, track_id: int, part_stem: str, frames
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(points (n, K, 2), scores (n, K))`` for one subject on part-local ``frames`` of
        one part, NaN on a frame it has no keypoints — a montage tile's frames, which are a
        proxy's samples of the part rather than a run of global frames."""
        frames = np.asarray(frames, dtype=np.int64)
        k = len(self.skeleton.names)
        points = np.full((len(frames), k, 2), np.nan, dtype=np.float64)
        scores = np.full((len(frames), k), np.nan, dtype=np.float64)
        series = self._load(int(track_id), part_stem)
        if series is None or not len(series[0]) or not len(frames):
            return points, scores
        known, pts, scs = series
        at = np.clip(np.searchsorted(known, frames), 0, len(known) - 1)
        hit = known[at] == frames
        points[hit] = pts[at[hit]]
        scores[hit] = scs[at[hit]]
        return points, scores

    def typical_length(self, track_id: int) -> float:
        """The subject's median body-axis length in pixels over the whole run (0 when it has
        none): what a body length is relative to, and what the aligned pose is scaled by, so
        neither changes with the window."""
        columns = self.axis_columns
        if columns is None:
            return 0.0
        with self._lock:
            found = self._typical.get(int(track_id))
        if found is not None:
            return found
        front, back, _middle = columns
        lengths = []
        for stem in self.part_stems:
            series = self._load(int(track_id), stem)
            if series is not None:
                lengths.append(
                    body_axis.row_headings(
                        series[1], series[2], front, back, self.frame_wh, self.cutoff
                    )[1]
                )
        every = np.concatenate(lengths) if lengths else np.zeros(0)
        every = every[np.isfinite(every) & (every > 0)]
        typical = float(np.median(every)) if every.size else 0.0
        with self._lock:
            self._typical[int(track_id)] = typical
        return typical

    def at(
        self, part_id: str, frame_idx: int
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """``{subject: (points (K, 2) normalized, scores (K,))}`` on one frame of a part."""
        out = {}
        for track_id in self.run.track_ids:
            series = self._load(int(track_id), part_id)
            if series is None:
                continue
            frames, points, scores = series
            i = int(np.searchsorted(frames, frame_idx))
            if i < len(frames) and int(frames[i]) == int(frame_idx):
                out[int(track_id)] = (points[i], scores[i])
        return out

    def _load(self, track_id: int, part_id: str):
        key = (track_id, part_id)
        with self._lock:
            if key not in self._series:
                if part_id in self.run.part_ids(track_id):
                    rows = self.run.rows(track_id, part_id)
                    self._series[key] = (
                        np.asarray(rows["frame_idx"]),
                        np.asarray(rows["points"], dtype=np.float32),
                        np.asarray(rows["point_scores"], dtype=np.float32),
                    )
                else:
                    self._series[key] = None
            return self._series[key]


def heading_definition_for(
    experiment_dir: Path, meta: dict, names: tuple[str, ...]
) -> tuple[dict | None, str | None]:
    """``(definition, None)`` or ``(None, why there is none)`` for the clip window's body axis:
    the aligned embedding's own, else the experiment's single saved definition that fits.
    """
    from detail.storage import heading_definitions

    own = meta.get("heading_definition")
    if own:
        try:
            return body_axis.validate_definition(dict(own), names), None
        except ValueError as exc:
            return None, f"The embedding's heading definition does not fit: {exc}"
    fitting = [
        d
        for d in heading_definitions.list_definitions(experiment_dir)
        if heading_definitions.fits(d, names) is None
    ]
    if len(fitting) == 1:
        return body_axis.validate_definition(fitting[0], names), None
    if not fitting:
        return None, (
            "No heading definition says which keypoints are the front and the back. Make "
            "one with Define... beside Heading on the embedding form (it is saved for the "
            "experiment) to measure the body's length, bend and turning and to align the "
            "pose."
        )
    return None, (
        "Several saved heading definitions fit this skeleton "
        f"({', '.join(sorted(str(d.get('name')) for d in fitting))}) and the embedding was "
        "not aligned by one, so none is chosen for you."
    )


# --- the clip window's view, exported as shown ------------------------------------------- #
#: Pixels between the footage and the side column, and between the side panes — the window's.
VIEW_GAP_PX = 6


def view_picture(views: ClipViews, *, panes: bool) -> np.ndarray:
    """What the clip window shows of one frame, as one picture: the footage alone (``panes``
    False — the plain player), or the footage with the masks pane over the keypoints pane
    beside it at half its height (the panels player's layout, ``gui/clip_panes.py``). A run
    with no keypoints leaves the keypoints pane white, as the window leaves it empty.

    Raises:
        ValueError: ``views`` was read without the footage.
    """
    if views.frame is None:
        raise ValueError("this view was read without the footage; read it with footage")
    footage = views.frame
    if not panes:
        return footage
    h, w = footage.shape[:2]
    cell_h = max((h - VIEW_GAP_PX) // 2, 1)
    cell_w = max(int(round(cell_h * w / max(h, 1))), 1)
    out = np.full((h, w + VIEW_GAP_PX + cell_w, 3), 255, dtype=np.uint8)
    out[:, :w] = footage
    x = w + VIEW_GAP_PX
    for row, pane in enumerate((views.masks, views.pose)):
        if pane is None:
            continue
        small = cv2.resize(pane, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y = row * (cell_h + VIEW_GAP_PX)
        out[y : y + cell_h, x : x + cell_w] = small
    return out
