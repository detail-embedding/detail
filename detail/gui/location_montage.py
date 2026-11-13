"""A location export's recordings playing side by side — its preview (one recording) and its
montage (several) (DESIGN_PLAN §16 "Preview and montage").

The cluster montage's engine (``api/clip_montage.ClusterMontage``) over clips found through a
``ClipSource``, so the tiles, the clock, the cache and the exports are the cluster window's. The
window itself is this export's: a window start and length (or each recording's full length), the
full-frame tick, the layer ticks, and the exports — GIF / MP4 at the player's speed, the picture
on screen, and the last tracing frame. The cluster window's sorting, paging, selection, profile
and query belong to clusters and are not here.

Every control is built from ``configs/location_plots.LocationMontageView``, so its range and help
match the CLI's ``locations montage``. Building the montage reads every clip's timestamps and
boxes, and every picture is decoded off the UI thread (``run_progress`` / the scrubber's decoder).
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from detail.configs.location_plots import LocationMontageView
from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
from detail.gui import frame_source
from detail.gui._pydantic_form import any_field_value, any_field_widget, field_meta
from detail.gui.busy import run_progress
from detail.gui.dialog import Dialog
from detail.gui.forms import ConfigForm
from detail.gui.image_view import ImageView, PictureWithControls
from detail.gui.run_preview import _to_pixmap
from detail.gui.scrubber import FrameScrubber
from detail.gui.sizing import lines, set_available
from detail.gui.video_codec import codec_of_filter, mp4_filters, with_codec_note

#: Fields that decide which clips are built (a change rebuilds the montage); the rest only
#: change how they are drawn.
_REBUILDS = ("start_s", "length_s", "full_length", "full_frame")
#: Fields the window takes from elsewhere: the player's speed menu is the speed.
_NOT_A_CONTROL = ("speed", "trail_points")
#: Each tile is drawn at this many pixels; the picture is scaled to the pane.
DRAW_PX = 224


class LocationMontageDialog(Dialog):
    """Play ``recording_ids`` of a location-export run side by side, every chosen animal
    (``subjects``, by label) in its colour."""

    def __init__(
        self,
        experiment,
        plots,
        recording_ids: list[str],
        subjects: list[str] | None,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._plots = plots
        self._recording_ids = list(recording_ids)
        self._subjects = subjects
        self._built = None
        self._generation = 0
        what = "preview" if len(self._recording_ids) == 1 else "montage"
        self.setWindowTitle(f"{plots.run_id} · {what}")

        layout = self.body()
        form_host = QWidget()
        self._form = ConfigForm(form_host, LocationMontageView)
        self._widgets: dict[str, QWidget] = {}
        for name in LocationMontageView.model_fields:
            if name in _NOT_A_CONTROL:
                continue
            widget = any_field_widget(LocationMontageView, name)
            meta = field_meta(LocationMontageView, name)
            self._widgets[name] = widget
            self._form.addRow(f"{meta['label']}:", widget)
            signal = getattr(widget, "toggled", None) or getattr(widget, "valueChanged")
            signal.connect(
                self._rebuild if name in _REBUILDS else lambda *_a: self._redraw()
            )
        self._form.finish()
        layout.addWidget(form_host)

        self._view = ImageView(zoomable=True)
        self._view.setMinimumSize(QSize(lines(20), lines(12)))
        self._view.clear_frame("Preparing the recordings…")
        self._scrubber = FrameScrubber(playable=True, loop=True)
        self._scrubber.set_decoder(self._compute)
        self._scrubber.rendered.connect(self._display)
        self._block = PictureWithControls(self._view, self._scrubber)
        layout.addWidget(self._block, 1)

        exports = QHBoxLayout()
        self._export_btn = QPushButton("Export GIF / MP4…")
        self._export_btn.clicked.connect(self._export_animation)
        self._picture_btn = QPushButton("Save picture…")
        self._picture_btn.clicked.connect(self._save_picture)
        self._last_btn = QPushButton("Last tracing frame…")
        self._last_btn.setToolTip(
            "A PNG of the last picture with every trail covering the whole clip — one "
            "picture of where each animal went."
        )
        self._last_btn.clicked.connect(self._export_last_frame)
        for button in (self._export_btn, self._picture_btn, self._last_btn):
            exports.addWidget(button)
        exports.addStretch(1)
        layout.addLayout(exports)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)
        self.finished.connect(self._shut_down)
        self._sync_buttons()
        self._rebuild()

    # -- the view ------------------------------------------------------------------ #
    def view(self) -> LocationMontageView:
        """The controls as a ``LocationMontageView``; the speed is the player's."""
        values = {
            name: any_field_value(LocationMontageView, name, widget)
            for name, widget in self._widgets.items()
        }
        values["speed"] = min(max(abs(self._scrubber.speed()), 0.25), 4.0)
        return LocationMontageView(**values)

    def _sync_buttons(self) -> None:
        ready = self._built is not None
        why = "Opens once the recordings are prepared."
        for button in (self._export_btn, self._picture_btn, self._last_btn):
            set_available(button, ready, why)

    # -- building ------------------------------------------------------------------ #
    def _rebuild(self, *_args) -> None:
        """Build the montage for the current window off the UI thread, then play it."""
        self._generation += 1
        generation = self._generation
        self._scrubber.pause()
        old, self._built = self._built, None
        if old is not None:
            old.montage.close()
        self._sync_buttons()
        try:
            view = self.view()
            layers = view.layers()
        except ValueError as exc:
            self._view.clear_frame(str(exc))
            return
        plots, ids, subjects = self._plots, self._recording_ids, self._subjects
        self._view.clear_frame("Preparing the recordings…")

        def work(progress):
            built = plots.montage(view, ids, subjects, progress)
            built.montage.prepare(progress)  # off-thread (run_progress)
            built.montage.draw_first(layers, DRAW_PX)  # off-thread (run_progress)
            return built

        run_progress(
            self._block,
            work,
            lambda built: self._on_built(generation, built),
            label="Preparing the recordings…",
            on_error=self._failed,
            supersedes="location montage",
        )

    def _on_built(self, generation: int, built) -> None:
        if generation != self._generation:
            built.montage.close()
            return
        self._built = built
        montage = built.montage
        n_steps = montage.n_steps
        self._scrubber.set_total(int(n_steps))
        self._scrubber.set_playback_rate(montage.fps)
        self._scrubber.set_timestamps(
            np.asarray([montage.step_offset_ns(i) for i in range(n_steps)], np.int64)
        )
        self._status.setText("; ".join(built.notes))
        self._sync_buttons()
        self._scrubber.set_frame(0)
        if n_steps > 1:
            self._scrubber.play()

    def _failed(self, exc: BaseException) -> None:
        self._view.clear_frame(str(exc))

    def _redraw(self) -> None:
        try:
            self.view().layers()
        except ValueError as exc:
            self._view.clear_frame(str(exc))
            return
        if self._built is not None:
            self._scrubber.set_frame(self._scrubber.frame())

    def _shut_down(self, *_args) -> None:
        self._generation += 1
        if self._built is not None:
            self._built.montage.close()
            self._built = None

    # -- drawing ------------------------------------------------------------------- #
    def _grid(self) -> int:
        return max(1, math.ceil(math.sqrt(len(self._built.montage.clips))))

    def _compute(self, step: int):
        built = self._built
        if built is None:
            return None
        try:
            layers = self.view().layers()
        except ValueError:
            return None
        columns, generation = self._grid(), self._generation
        picture = frame_source.decode(  # off-thread (scrubber decoder)
            lambda: built.montage.render(
                built.montage.step_offset_ns(step),
                layers,
                built.labels,
                columns=columns,
                tile_px=DRAW_PX,
            )
        )
        return generation, picture

    def _display(self, _step: int, result) -> None:
        if result is None:
            return
        generation, (picture, reason) = result
        if generation != self._generation:
            return
        if reason is not None:
            self._view.clear_frame(frame_source.unavailable_text(reason))
            return
        self._last_picture = picture
        self._view.set_frame(_to_pixmap(picture))
        self._block.set_aspect(self._view.pixmap_size())

    # -- exporting ----------------------------------------------------------------- #
    def _ask(self, title: str, default: str, filters: str) -> str | None:
        path, _filter = QFileDialog.getSaveFileName(self, title, default, filters)
        return path or None

    def _export_animation(self) -> None:
        # The mp4 codecs are file types of their own, so the one dialog asks both.
        path, chosen = QFileDialog.getSaveFileName(
            self,
            "Export montage",
            f"{self._plots.run_id}_montage.gif",
            ";;".join(["GIF (*.gif)", *mp4_filters()]),
        )
        if path:
            self.export_to(path, codec=codec_of_filter(chosen))

    def _export_last_frame(self) -> None:
        path = self._ask(
            "Last tracing frame", f"{self._plots.run_id}_last_frame.png", "PNG (*.png)"
        )
        if path:
            self.export_to(path, last_frame=True)

    def export_to(
        self, path: str, *, last_frame: bool = False, codec: str = DEFAULT_VIDEO_CODEC
    ) -> None:
        """Write the montage (``.gif`` / ``.mp4``, an mp4 encoded as ``codec``) or its last
        tracing frame (``.png``) to ``path``, off the UI thread, at the player's speed.
        """
        plots, view = self._plots, self.view()
        ids, subjects = self._recording_ids, self._subjects
        run_progress(
            self,
            lambda emit: plots.export_montage(
                view,
                path,
                ids,
                subjects,
                last_frame=last_frame,
                progress=emit,
                codec=codec,
            ),
            lambda written: self._status.setText(
                with_codec_note(f"Wrote {written}", written, codec)
            ),
            label="Exporting…",
            on_error=self._failed,
        )

    def _save_picture(self) -> None:
        path = self._ask(
            "Save picture", f"{self._plots.run_id}_picture.png", "PNG (*.png)"
        )
        if path:
            self.save_picture(path)

    def save_picture(self, path: str) -> None:
        """Write the picture on screen as a PNG."""
        import cv2

        picture = getattr(self, "_last_picture", None)
        if picture is None:
            self._status.setText("No picture yet.")
            return
        cv2.imwrite(path, cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
        self._status.setText(f"Wrote {path}")
