"""The codec picker and the save dialog's codec file types (Tenet 15).

Two shapes of one question: a form holds ``VideoCodecPicker``, a window that asks only where to
save offers the codecs as file types. Both say so when this machine writes another codec than
the one chosen; that note is what makes the fallback visible on the surface.
"""

import pytest

from detail.core.video_codecs import VIDEO_CODECS
from detail.gui import video_codec
from detail.storage import video as video_store


def test_the_picker_offers_every_codec_h264_first(qapp):
    picker = video_codec.VideoCodecPicker()
    combo = picker._combo
    assert [combo.itemData(i) for i in range(combo.count())] == list(VIDEO_CODECS)
    assert picker.codec() == "h264"
    seen = []
    picker.changed.connect(seen.append)
    picker.set_codec("mp4v")
    assert picker.codec() == "mp4v" and seen == ["mp4v"]


def test_the_picker_says_so_when_h264_cannot_be_written(qapp, monkeypatch):
    monkeypatch.setattr(video_store, "h264_available", lambda: False)
    picker = video_codec.VideoCodecPicker()
    picker.show()
    assert "PowerPoint" in picker.note() and picker._note.isVisible()
    picker.set_codec("mp4v")
    assert picker.note() == "" and not picker._note.isVisible()
    picker.close()


def test_no_note_when_h264_is_written(qapp, monkeypatch):
    monkeypatch.setattr(video_store, "h264_available", lambda: True)
    assert video_codec.VideoCodecPicker().note() == ""


@pytest.mark.parametrize("codec", VIDEO_CODECS)
def test_each_save_file_type_names_its_codec(codec):
    (chosen,) = [f for f, c in video_codec.mp4_filters().items() if c == codec]
    assert video_codec.codec_of_filter(chosen) == codec
    assert video_codec.codec_of_filter("GIF (*.gif)") == "h264"


def test_the_written_line_carries_the_fallback(monkeypatch):
    monkeypatch.setattr(video_store, "h264_available", lambda: False)
    line = video_codec.with_codec_note("Wrote a.mp4", "a.mp4", "h264")
    assert line.startswith("Wrote a.mp4 ") and "PowerPoint" in line
    assert video_codec.with_codec_note("Wrote a.gif", "a.gif", "h264") == "Wrote a.gif"


def test_the_export_clips_dialog_writes_the_codec_chosen(qapp, tmp_path):
    """**Export ▾ → Export clips…** on a clip grid, opened the way a user opens it."""
    from tests.gui.test_export_clips import _close, _grid, _open

    montage = _grid(qapp, tmp_path)
    dialog = _open(qapp, montage)
    dialog._mp4_codec.set_codec("mp4v")
    (mp4,) = [f for f in dialog.formats() if f.kind == "mp4"]
    assert mp4.codec == "mp4v"
    dialog._mp4.setChecked(False)
    assert not dialog._mp4_codec.isEnabled(), "the codec greys out with its MP4 tick"
    _close(montage)
