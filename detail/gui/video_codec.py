"""How an exported MP4 is encoded, asked the same way on every surface that writes one (Tenet 15).

Two shapes, because the surfaces ask in two ways. A dialog with a form holds a
:class:`VideoCodecPicker`: the codecs in a combo, and beneath it the sentence saying so when this
machine cannot write the one chosen. A window that asks only where to save offers the codecs as
the save dialog's file types (:func:`mp4_filters` / :func:`codec_of_filter`), so choosing
H.264 or mp4v costs no extra dialog, and :func:`with_codec_note` adds the fallback sentence to the
line that says where the file went. Either way a fallback is never silent.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from detail.core.video_codecs import DEFAULT_VIDEO_CODEC
from detail.core.video_codecs import VIDEO_CODEC_HELP
from detail.core.video_codecs import VIDEO_CODEC_LABELS
from detail.core.video_codecs import VIDEO_CODECS


def _note(codec: str) -> str | None:
    from detail.storage import video as video_store

    return video_store.codec_note(codec)


class VideoCodecPicker(QWidget):
    """The codec combo, and the note when this machine writes another codec than the one
    chosen. ``changed`` fires with the codec when the choice moves."""

    changed = pyqtSignal(str)

    def __init__(self, parent=None, *, codec: str = DEFAULT_VIDEO_CODEC) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._combo = QComboBox()
        self._combo.setObjectName("video_codec")
        for name in VIDEO_CODECS:
            self._combo.addItem(VIDEO_CODEC_LABELS[name], name)
        self._combo.setCurrentIndex(VIDEO_CODECS.index(codec))
        self._combo.setToolTip(VIDEO_CODEC_HELP)
        layout.addWidget(self._combo)
        self._note = QLabel("")
        self._note.setObjectName("video_codec_note")
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self._combo.currentIndexChanged.connect(self._on_changed)
        self._show_note()

    def codec(self) -> str:
        """The codec chosen (what was asked for; ``storage.video.written_codec`` says what a
        file will actually hold)."""
        return str(self._combo.currentData())

    def set_codec(self, codec: str) -> None:
        """Choose ``codec`` (one of ``VIDEO_CODECS``)."""
        self._combo.setCurrentIndex(VIDEO_CODECS.index(codec))

    def note(self) -> str:
        """The fallback sentence shown, or ``""`` when the chosen codec is written as asked."""
        return self._note.text()

    def _on_changed(self, _index: int) -> None:
        self._show_note()
        self.changed.emit(self.codec())

    def _show_note(self) -> None:
        note = _note(self.codec()) or ""
        self._note.setText(note)
        self._note.setVisible(bool(note))


def mp4_filters() -> dict[str, str]:
    """``{save-dialog file type: codec}`` for an mp4, the default codec first."""
    return {
        f"MP4 video, {VIDEO_CODEC_LABELS[name]} (*.mp4)": name for name in VIDEO_CODECS
    }


def codec_of_filter(selected: str) -> str:
    """The codec a save dialog's chosen file type names; the default for any other type."""
    return mp4_filters().get(selected, DEFAULT_VIDEO_CODEC)


def with_codec_note(text: str, path, codec: str) -> str:
    """``text`` (the line that says where a file went), with the fallback sentence after it
    when ``path`` is an mp4 this machine wrote in another codec than ``codec``."""
    if str(path).lower().endswith(".mp4"):
        note = _note(codec)
        if note:
            return f"{text} {note}"
    return text
