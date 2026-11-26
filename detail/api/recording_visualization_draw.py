"""Drawing one recording-visualization frame (DESIGN_PLAN §22 "Recording visualization").

A :class:`Canvas` is one matplotlib figure laid out for one tile size, video shape and theme
(``analysis.recording_visualization.layout_for``). Everything that does not change frame to frame
— the grey cloud, the ethogram's bars, the floor of the location panel — is drawn once; a frame
moves the picture, the outlines, the trails, the playhead and the text. No Qt: the window and
the export both call :meth:`Canvas.draw` from worker threads or processes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from detail.analysis import recording_visualization as rv  # noqa: E402

#: Points at the figure's 100 dpi: a px size times this is matplotlib's font size.
PT_PER_PX = 0.72
#: Keypoint colours, front to back (Figure 1's red, green, blue), cycled for longer skeletons.
POINT_COLOURS = ("#e34948", "#1baf7a", "#2a78d6", "#eda100", "#6250d6")
#: Outlines in the trail: every this many seconds over the trail.
OUTLINE_EVERY_S = 0.4
#: Masks kept for the outline trail (a frame's past outlines recur for many frames).
MASK_CACHE = 400


class Canvas:
    """One visualization drawn at ``width`` x ``height`` in ``theme``."""

    def __init__(self, data, reader, theme: str, width: int, height: int) -> None:
        self.data, self.reader, self.theme = data, reader, theme
        self.t = rv.THEMES[theme]
        self.width, self.height = int(width), int(height)
        self.layout = rv.layout_for(
            self.width, self.height, data.aspect, data.positions is not None
        )
        self._lock = threading.Lock()
        self._masks: OrderedDict[int, np.ndarray] = OrderedDict()
        fig = plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        fig.patch.set_facecolor(self.t["bg"])
        self.fig = fig
        fp = self.layout.font_px
        self._fs = {k: v * PT_PER_PX for k, v in fp.items()}
        self._build_video()
        self._build_ethogram()
        self._build_now()
        self._build_outline()
        self._build_pose()
        self._build_location()
        self._clock = fig.text(
            1 - self.layout.pad / self.width,
            self.layout.pad * 0.4 / self.height,
            "",
            color=self.t["muted"],
            fontsize=self._fs["time"],
            ha="right",
            va="bottom",
        )

    # -- geometry --------------------------------------------------------------------------- #
    def _axes(self, box: rv.Box, **kw):
        a = self.fig.add_axes(
            [
                box.x / self.width,
                1 - (box.y + box.h) / self.height,
                box.w / self.width,
                box.h / self.height,
            ],
            **kw,
        )
        a.set_axis_off()
        a.patch.set_alpha(0)
        return a

    def _caption(self, box: rv.Box, text: str) -> None:
        """A small spaced caption on the line just above ``box``."""
        self.fig.text(
            box.x / self.width,
            1 - (box.y - 4) / self.height,
            " ".join(text.upper()),
            color=self.t["muted"],
            fontsize=self._fs["label"],
            weight="bold",
            va="bottom",
        )

    def _captioned(self, box: rv.Box, text: str) -> rv.Box:
        """Caption ``box`` on its own first line and return what is left under it, so a
        panel's caption never reaches into the panel above."""
        line = self.layout.font_px["label"] * 1.9
        inner = rv.Box(box.x, box.y + line, box.w, max(box.h - line, 1.0))
        self._caption(inner, text)
        return inner

    # -- the video ---------------------------------------------------------------------------- #
    def _build_video(self) -> None:
        d, box = self.data, self.layout.video
        w, h = d.size
        a = self._axes(box)
        self._im = a.imshow(
            np.zeros((h, w, 3)), extent=(0, w, h, 0), interpolation="bilinear"
        )
        a.set_xlim(0, w), a.set_ylim(h, 0)
        r = 10 * w / max(box.w, 1)
        clip = FancyBboxPatch(
            (0, 0),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            transform=a.transData,
            facecolor="none",
            edgecolor="none",
        )
        a.add_patch(clip)
        self._im.set_clip_path(clip)
        lw = max(1.2, box.w / 900)
        self._contour = a.plot([], [], lw=lw * 1.6)[0]
        self._bones = a.plot(
            [], [], color="white", lw=lw, alpha=0.9, solid_capstyle="round"
        )[0]
        self._dots = a.scatter(
            [], [], s=(4 * lw) ** 2, edgecolors="white", linewidths=lw * 0.7, zorder=5
        )
        span = ""
        if d.recipe.get("start_s") is not None:
            span = f" · {_clock(d.recipe['start_s'])}–{_clock(d.recipe['end_s'])}"
        self._caption(box, f"{d.recording_name}{span}"[:80])

    def _draw_video(self, rgb, mask, kp, colour) -> None:
        img = rgb.astype(np.float32) / 255
        if mask is not None and mask.any():
            c = np.array(matplotlib.colors.to_rgb(colour))
            img[mask] = img[mask] * 0.68 + c * 0.32
        self._im.set_data(img)
        xs, ys = [], []
        if mask is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for cc in contours:
                cc = cc[:, 0]
                xs += [*cc[:, 0], cc[0, 0], np.nan]
                ys += [*cc[:, 1], cc[0, 1], np.nan]
        self._contour.set_data(xs, ys)
        self._contour.set_color(colour)
        if kp is None:
            self._bones.set_data([], [])
            self._dots.set_offsets(np.empty((0, 2)))
            return
        self._bones.set_data(kp[:, 0], kp[:, 1])
        self._dots.set_offsets(np.nan_to_num(kp, nan=-1e4))
        self._dots.set_facecolor(
            [POINT_COLOURS[i % len(POINT_COLOURS)] for i in range(len(kp))]
        )

    # -- the ethogram ------------------------------------------------------------------------ #
    def _build_ethogram(self) -> None:
        d, box = self.data, self.layout.ethogram
        a = self._axes(box)
        a.set_ylim(0, 1)
        half = int(d.recipe.get("ethogram_half_s", 10.0) * d.fps)
        lo, hi = max(0, d.first - half), min(len(d.codes), d.last + half)
        codes = d.codes[lo:hi]
        edges = np.flatnonzero(np.r_[1, np.diff(codes) != 0, 1])
        gap = max(0.5, 2 * len(codes) / max(box.w, 1) / 3)
        for s0, s1 in zip(edges[:-1], edges[1:]):
            a.add_patch(
                Rectangle(
                    (lo + s0, 0),
                    max(0.5, s1 - s0 - gap),
                    1,
                    facecolor=rv.colour_of(self.theme, int(codes[s0])),
                    lw=0,
                )
            )
        self._fade = a.add_patch(
            Rectangle(
                (0, -0.1),
                half + 10,
                1.2,
                facecolor=self.t["bg"],
                alpha=self.t["fade"],
                lw=0,
                zorder=5,
            )
        )
        self._head = a.plot(
            [0, 0],
            [-0.9, 1.9],
            color=self.t["ink"],
            lw=max(1.5, box.w / 900),
            zorder=6,
            clip_on=False,
        )[0]
        self._eth, self._half = a, half

    # -- now ----------------------------------------------------------------------------------- #
    def _build_now(self) -> None:
        box = self.layout.now
        fb, fd = self.layout.font_px["behavior"], self.layout.font_px["duration"]
        a = self._axes(rv.Box(box.x, box.y + fb * 0.1, fb * 0.9, fb * 1.1))
        a.set_xlim(0, 1), a.set_ylim(0, 1)
        self._now_dot = a.scatter([0.5], [0.5], s=(fb * 0.55) ** 2, linewidths=0)
        x = (box.x + fb * 1.25) / self.width
        self._now_name = self.fig.text(
            x,
            1 - (box.y + fb * 0.65) / self.height,
            "",
            fontsize=self._fs["behavior"],
            weight="bold",
            va="center",
            color=self.t["ink"],
        )
        self._now_dur = self.fig.text(
            x,
            1 - (box.y + fb * 1.3 + fd * 0.6) / self.height,
            "",
            fontsize=self._fs["duration"],
            va="center",
            color=self.t["muted"],
        )

    def _draw_now(self, fno: int) -> None:
        d = self.data
        code = int(d.codes[fno])
        self._now_dot.set_facecolor(rv.colour_of(self.theme, code))
        if code == rv.NO_POSE:
            self._now_name.set_text("—")
            self._now_dur.set_text("no pose this frame")
            return
        name = d.names[code] if 0 <= code < len(d.names) else d.grey_name
        start = fno
        while start > 0 and int(d.codes[start - 1]) == code:
            start -= 1
        self._now_name.set_text(name)
        self._now_dur.set_text(f"{(fno - start) / d.fps:.1f} s")

    # -- the outline trail ----------------------------------------------------------------------- #
    def _build_outline(self) -> None:
        d = self.data
        trail = float(d.recipe.get("trail_s", 2.0))
        box = self._captioned(self.layout.outline, f"last {trail:g} s")
        a = self._axes(box)
        a.set_aspect("equal", adjustable="datalim")
        half = max(d.body_half, 1.0)
        # The shorter side holds the whole animal; the longer side shows more around it.
        short = max(min(box.w, box.h), 1.0)
        ext = half * box.w / short, half * box.h / short
        a.set_xlim(-ext[0], ext[0]), a.set_ylim(ext[1], -ext[1])
        k = max(2, int(round(trail / OUTLINE_EVERY_S)) + 1)
        g0 = np.array(matplotlib.colors.to_rgb(self.t["old"]))
        g1 = np.array(matplotlib.colors.to_rgb(self.t["new"]))
        lw = max(0.8, box.w / 300)
        self._fill = a.fill([], [], color=self.t["fill"], zorder=1)[0]
        self._outlines = [
            a.plot(
                [],
                [],
                color=tuple(g0 + (g1 - g0) * i / (k - 1)),
                lw=lw * (0.6 + i / (k - 1)),
                zorder=2,
            )[0]
            for i in range(k)
        ]
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def _mask_at(self, fno: int) -> np.ndarray | None:
        cached = self._masks.get(fno)
        if cached is None:
            try:
                cached = self.reader.mask_only(fno)
            except (IndexError, ValueError, OSError):
                return None
            self._masks[fno] = cached
            while len(self._masks) > MASK_CACHE:
                self._masks.popitem(last=False)
        return cached

    def _outline_of(self, fno: int):
        mask = self._mask_at(fno)
        if mask is None or not mask.any():
            return None
        m = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, self._kernel
        )  # drops the tail
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)[:, 0].astype(float)
        mom = cv2.moments(m, binaryImage=True)
        if mom["m00"] == 0:
            return None
        c -= (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"])
        cols = self.reader.axis_columns
        kp = self.reader.keypoints(fno) if cols is not None else None
        if kp is not None and np.isfinite(kp).all():
            heading = kp[cols[0]].mean(0) - kp[cols[1]].mean(0)
            if self.data.align == "rotate":
                phi = -np.pi / 2 - np.arctan2(heading[1], heading[0])
                rot = np.array(
                    [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
                )
                c = c @ rot.T
            elif self.data.align == "mirror" and heading[0] < 0:
                c[:, 0] *= -1
        return np.vstack([c, c[:1]])

    def _draw_outline(self, fno: int) -> None:
        step = int(round(OUTLINE_EVERY_S * self.data.fps))
        n = len(self._outlines)
        for i, line in enumerate(self._outlines):
            f = fno - step * (n - 1 - i)
            c = self._outline_of(f) if f >= 0 else None
            line.set_data(*(c.T if c is not None else ([], [])))
            if i == n // 2:
                self._fill.set_xy(c if c is not None else np.zeros((1, 2)))

    # -- pose space ----------------------------------------------------------------------------- #
    def _build_pose(self) -> None:
        d = self.data
        box = self._captioned(self.layout.pose, "pose space")
        a = self._axes(box)
        cloud = d.cloud
        lo, hi = (
            np.percentile(cloud, [0.5, 99.5], axis=0)
            if len(cloud)
            else (np.zeros(2), np.ones(2))
        )
        pad = (hi - lo) * 0.04 + 1e-9
        lims = ((lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]))
        a.imshow(
            self._cloud_image(int(box.w), int(box.h), lims),
            extent=(*lims[0], *lims[1]),
            aspect="auto",
            zorder=0,
        )
        a.set_xlim(*lims[0]), a.set_ylim(*lims[1])
        lw = max(0.8, box.w / 500)
        self._ghost = a.plot(
            [], [], color=self.t["ghost"], lw=lw * 0.8, alpha=0.8, zorder=2
        )[0]
        self._trail = LineCollection([], capstyle="round", zorder=3)
        a.add_collection(self._trail)
        dot = (max(6.0, box.w / 45)) ** 2
        self._halo = a.scatter(
            [0], [0], s=dot * 3.5, alpha=0.25, linewidths=0, zorder=5
        )
        self._dot = a.scatter(
            [0], [0], s=dot, edgecolors=self.t["ring"], linewidths=lw * 1.5, zorder=6
        )
        self._pose_lw = lw

    def _cloud_image(self, w: int, h: int, lims) -> np.ndarray:
        fig = plt.figure(figsize=(max(w, 2) / 100, max(h, 2) / 100), dpi=100)
        fig.patch.set_alpha(0)
        a = fig.add_axes([0, 0, 1, 1])
        a.set_axis_off()
        a.patch.set_alpha(0)
        s = max(0.6, w / 230)
        a.scatter(
            self.data.cloud[:, 0],
            self.data.cloud[:, 1],
            s=s,
            color=self.t["cloud"],
            alpha=self.t["cloud_alpha"],
            linewidths=0,
        )
        a.set_xlim(*lims[0]), a.set_ylim(*lims[1])
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        return img

    def _draw_pose(self, fno: int) -> None:
        d = self.data
        ghost = int(d.recipe.get("ghost_s", 8.0) * d.fps)
        trail = int(d.recipe.get("trail_s", 2.0) * d.fps)
        g = d.xy[max(0, fno - ghost) : fno + 1]
        self._ghost.set_data(g[:, 0], g[:, 1])
        t0 = max(0, fno - trail)
        pts = d.xy[t0 : fno + 1]
        segs = np.stack([pts[:-1], pts[1:]], 1) if len(pts) > 1 else np.empty((0, 2, 2))
        n = len(segs)
        colours = [
            matplotlib.colors.to_rgba(
                rv.colour_of(self.theme, int(d.codes[t0 + i + 1])), al
            )
            for i, al in enumerate(np.linspace(0.1, 1, n))
        ]
        self._trail.set_segments(segs)
        self._trail.set_colors(colours)
        self._trail.set_linewidths(
            np.linspace(self._pose_lw * 0.6, self._pose_lw * 2.8, n) if n else []
        )
        here = d.xy[fno]
        ok = bool(np.isfinite(here).all())
        for s in (self._dot, self._halo):
            s.set_visible(ok)
            if ok:
                s.set_offsets([here])
                s.set_facecolor(rv.colour_of(self.theme, int(d.codes[fno])))

    # -- location ---------------------------------------------------------------------------- #
    def _build_location(self) -> None:
        self._loc = None
        d, box = self.data, self.layout.location
        if box is None or d.positions is None:
            return
        box = self._captioned(box, "location")
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        X, Y = d.floor
        window = float(d.recipe.get("location_window_s", 10.0))
        side = min(box.w, box.h) * 1.0
        sq = rv.Box(box.x + (box.w - side) / 2, box.y + (box.h - side) / 2, side, side)
        # 3-D axes stay square, so the box sits in a square sized to the panel's short side and
        # the time axis is stretched to fill a tall panel or squeezed for a wide one.
        tall = box.h / max(box.w, 1)
        a = self.fig.add_axes(
            [
                sq.x / self.width,
                1 - (sq.y + sq.h) / self.height,
                sq.w / self.width,
                sq.h / self.height,
            ],
            projection="3d",
            computed_zorder=False,
        )
        a.set_axis_off()
        a.patch.set_alpha(0)
        a.view_init(elev=18, azim=-76)
        a.set_xlim(0, X), a.set_ylim(Y, 0), a.set_zlim(-window, 0)
        a.set_box_aspect((X, Y, X * min(2.0, max(0.6, 1.2 * tall))), zoom=1.0)
        muted = self.t["muted"]
        floor = np.array([[0, 0], [X, 0], [X, Y], [0, Y], [0, 0]])
        a.plot(floor[:, 0], floor[:, 1], -window, color=muted, lw=1, alpha=0.8)
        for cx, cy in floor[:4]:
            a.plot([cx, cx], [cy, cy], [-window, 0], color=muted, lw=0.5, alpha=0.3)
        a.plot(floor[:, 0], floor[:, 1], 0, color=muted, lw=0.5, alpha=0.3)
        fs = self._fs["label"]
        a.text(
            -X * 0.03,
            Y,
            0,
            "now",
            color=self.t["ink"],
            fontsize=fs,
            ha="right",
            va="center",
        )
        a.text(
            -X * 0.03,
            Y,
            -window,
            f"−{window:g} s",
            color=muted,
            fontsize=fs,
            ha="right",
            va="center",
        )
        bar = rv.nice_scale_bar(X)
        a.plot([X - bar, X], [Y * 1.12, Y * 1.12], -window, color=self.t["ink"], lw=2)
        a.text(
            X - bar / 2,
            Y * 1.3,
            -window,
            f"{bar:g} {d.pos_units}",
            color=muted,
            fontsize=fs,
            ha="center",
            va="top",
        )
        self._shadow = a.plot(
            [], [], [], color=self.t["ghost"], lw=1, alpha=0.6, zorder=1
        )[0]
        self._path = Line3DCollection(
            [[(0, 0, -window), (0, 0, -window)]], zorder=3, linewidths=0
        )
        a.add_collection3d(self._path, autolim=False)
        self._drop = a.plot([], [], [], color=muted, lw=0.8, ls=(0, (2, 2)), zorder=2)[
            0
        ]
        self._ldot = a.scatter(
            [0],
            [0],
            [0],
            s=(max(6.0, box.w / 40)) ** 2,
            edgecolors=self.t["ring"],
            linewidths=2,
            zorder=6,
            depthshade=False,
        )
        self._loc, self._window = a, window

    def _draw_location(self, fno: int) -> None:
        if self._loc is None:
            return
        d, window = self.data, self._window
        win = int(window * d.fps)
        lo = max(0, fno - win)
        x, y = d.positions[lo : fno + 1, 0], d.positions[lo : fno + 1, 1]
        z = (np.arange(lo, fno + 1) - fno) / d.fps
        self._shadow.set_data_3d(x, y, np.full_like(z, -window))
        pts = np.c_[x, y, z]
        segs = np.stack([pts[:-1], pts[1:]], 1) if len(pts) > 1 else np.empty((0, 2, 3))
        n = len(segs)
        codes = d.codes[lo + 1 : fno + 1]
        colours = [
            matplotlib.colors.to_rgba(
                rv.colour_of(self.theme, int(c)) if c >= 0 else self.t["muted"], al
            )
            for c, al in zip(codes, np.linspace(0.3, 1, n))
        ]
        self._path.set_segments(segs)
        self._path.set_colors(colours)
        self._path.set_linewidths(np.linspace(0.8, 3.2, n) if n else [])
        cx, cy = d.positions[fno]
        ok = bool(np.isfinite(cx) and np.isfinite(cy))
        self._ldot.set_visible(ok)
        self._drop.set_visible(ok)
        if ok:
            self._ldot._offsets3d = ([cx], [cy], [0])
            code = int(d.codes[fno])
            self._ldot.set_facecolor(
                rv.colour_of(self.theme, code) if code >= 0 else self.t["muted"]
            )
            self._drop.set_data_3d([cx, cx], [cy, cy], [0, -window])

    # -- one frame --------------------------------------------------------------------------- #
    def draw(self, fno: int, rgb, mask, kp) -> np.ndarray:
        """The picture at recording frame ``fno`` as RGB uint8 (``height``, ``width``, 3)."""
        with self._lock:
            d = self.data
            if mask is not None:
                self._masks[int(fno)] = mask
            colour = rv.colour_of(self.theme, int(d.codes[fno]))
            if int(d.codes[fno]) < 0:
                colour = self.t["muted"]
            self._draw_video(rgb, mask, kp, colour)
            self._eth.set_xlim(fno - self._half, fno + self._half)
            self._fade.set_x(fno)
            self._head.set_xdata([fno, fno])
            self._draw_now(fno)
            self._draw_outline(fno)
            self._draw_pose(fno)
            self._draw_location(fno)
            self._clock.set_text(_clock(d.t_ns[fno] / 1e9))
            self.fig.canvas.draw()
            return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def blank(self) -> np.ndarray:
        """A tile of background, for a montage tile with nothing to show yet."""
        c = (np.array(matplotlib.colors.to_rgb(self.t["bg"])) * 255).astype(np.uint8)
        return np.broadcast_to(c, (self.height, self.width, 3)).copy()

    def dim(self, picture: np.ndarray) -> np.ndarray:
        """``picture`` faded towards the background: a montage tile that has ended."""
        bg = self.blank().astype(np.float32)
        return (picture.astype(np.float32) * 0.45 + bg * 0.55).astype(np.uint8)

    def close(self) -> None:
        plt.close(self.fig)


def _clock(seconds: float) -> str:
    seconds = float(seconds)
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
