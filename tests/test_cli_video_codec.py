"""Every CLI command that writes an MP4 takes ``--codec``, and a fallback is printed (Tenet 15).

Data-driven over the click tree: a command whose help or options mention mp4 writes video, so it
offers the codec through the one shared option (``_VIDEO_CODEC_OPT``), the way the API takes
``codec=`` and the GUI shows ``VideoCodecPicker``. A command added tomorrow that writes an mp4 is
held to it the day it lands. Hidden commands (§25) are exempt by ``surfaces.HIDDEN``, not by a
list here; they write the default codec through the same writer.
"""

import click
from click.testing import CliRunner

from detail import surfaces
from detail.api.experiment import Experiment
from detail.cli import main
from detail.core.video_codecs import VIDEO_CODECS
from detail.storage import video as video_store


def _commands(group, path=()):
    for name, command in group.commands.items():
        if isinstance(command, click.Group):
            yield from _commands(command, (*path, name))
        else:
            yield " ".join((*path, name)), command


def _hidden(path: str) -> bool:
    hidden = {
        p for entry in surfaces.HIDDEN.values() if entry.in_effect for p in entry.cli
    }
    return any(path == p or path.startswith(p + " ") for p in hidden)


def _writes_mp4(command: click.Command) -> bool:
    text = " ".join(
        [
            command.help or "",
            *((getattr(p, "help", None) or "") for p in command.params),
        ]
    )
    return "mp4" in text.lower()


def test_every_command_that_writes_an_mp4_takes_the_codec():
    missing, found = [], []
    for path, command in _commands(main):
        if not _writes_mp4(command):
            continue
        codec = next((p for p in command.params if p.name == "codec"), None)
        if codec is None:
            if not _hidden(path):
                missing.append(path)
            continue
        found.append(path)
        assert list(codec.type.choices) == list(VIDEO_CODECS), path
        assert codec.default == "h264", path
    assert not missing, f"these commands write an mp4 and take no --codec: {missing}"
    assert len(found) >= 4, found


def _export_run(tmp_path, monkeypatch, *extra):
    from tests.api.test_export_preview_targets import _experiment_dir, _write_run
    from tests.api.test_run_preview import _recording_with_detection

    recording = _recording_with_detection(tmp_path, n_frames=5)
    experiment_dir = _experiment_dir(recording)
    _write_run(experiment_dir, "det_run_01", "detection", "s1", "det_01")
    seen: dict = {}

    def estimate(self, kind, subject_id, target_ids, out_dir, scope=None, **options):
        seen.update(options)
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "estimate_export", estimate)
    result = CliRunner().invoke(
        main,
        [
            "export", "run", "--experiment", str(experiment_dir), "--run", "det_run_01",
            "--out", str(tmp_path / "out"), "--for-recording", "s1", *extra,
        ],
    )  # fmt: skip
    return result, seen


def test_export_run_passes_the_codec_only_to_a_video_export(tmp_path, monkeypatch):
    result, seen = _export_run(
        tmp_path, monkeypatch, "--target", "preview_video", "--codec", "mp4v"
    )
    assert "stop here" in result.output and seen["codec"] == "mp4v"
    _result, seen = _export_run(
        tmp_path / "frame", monkeypatch, "--target", "preview_frame", "--codec", "mp4v"
    )
    assert "codec" not in seen


def test_a_fallback_is_printed(tmp_path, monkeypatch):
    monkeypatch.setattr(video_store, "h264_available", lambda: False)
    result, seen = _export_run(tmp_path, monkeypatch, "--target", "preview_video")
    assert seen["codec"] == "h264"
    assert "note:" in result.output and "PowerPoint" in result.output
