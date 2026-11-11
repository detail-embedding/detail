"""The CLI's preview-cache surface (§3 *Video preview cache*): ``add-video`` builds the proxy,
``--no-preview`` skips it, and ``cache previews build|size`` / ``cache clear --previews``.
"""

from __future__ import annotations

from click.testing import CliRunner

from detail.api import previews
from detail.api.experiment import Experiment
from detail.cli import main


def _add_video(experiment, video, *extra):
    return CliRunner().invoke(
        main,
        [
            "recording",
            "add-video",
            "--experiment",
            str(experiment.dir),
            "--recording",
            "first",
            "--video",
            str(video),
            *extra,
        ],
    )


def test_add_video_builds_the_proxy(tmp_path, write_vfr_video):
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("first")
    result = _add_video(experiment, write_vfr_video(tmp_path / "a.mp4"))
    assert result.exit_code == 0, result.output
    assert "Building previews" in result.output  # the bar's label, on stderr
    assert experiment.missing_previews() == []


def test_a_failed_preview_warns_and_the_import_still_succeeds(
    tmp_path, write_vfr_video, monkeypatch
):
    """The video was added; only the display-only preview failed. A non-zero exit told a script
    the import had failed. It warns — what failed, and the command that rebuilds it — and exits
    0."""

    def broken(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(previews, "_build_proxy", broken)
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("first")
    result = _add_video(experiment, write_vfr_video(tmp_path / "a.mp4"))
    assert result.exit_code == 0, result.output
    assert "added video to" in result.output
    assert "preview failed: first part 1 proxy: OSError: no space left" in result.output
    assert (
        "warning: the video was added, but its preview was not built" in result.output
    )
    assert "detail cache previews build --experiment" in result.output
    assert len(experiment.resolve_recording("first").video_parts) == 1


def test_a_preview_that_raises_outright_still_only_warns(
    tmp_path, write_vfr_video, monkeypatch
):
    from detail.api.recording import Recording

    def broken(*_args, **_kwargs):
        raise RuntimeError("cache folder unwritable")

    monkeypatch.setattr(Recording, "build_preview", broken)
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("first")
    result = _add_video(experiment, write_vfr_video(tmp_path / "a.mp4"))
    assert result.exit_code == 0, result.output
    assert "preview failed: first: cache folder unwritable" in result.output
    assert "warning: the video was added" in result.output


def test_no_preview_skips_it_and_build_fills_it_in(tmp_path, write_vfr_video):
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("first")
    result = _add_video(experiment, write_vfr_video(tmp_path / "a.mp4"), "--no-preview")
    assert result.exit_code == 0, result.output
    assert [t.kind for t in experiment.missing_previews()] == ["proxy"]

    runner = CliRunner()
    built = runner.invoke(
        main,
        ["cache", "previews", "build", "--experiment", str(experiment.dir)],
    )
    assert built.exit_code == 0, built.output
    assert "previews: built 1" in built.output
    assert experiment.missing_previews() == []
    again = runner.invoke(
        main,
        [
            "cache",
            "previews",
            "build",
            "--experiment",
            str(experiment.dir),
            "--recording",
            "first",
        ],
    )
    assert again.exit_code == 0, again.output
    assert "nothing to do" in again.output


def test_size_and_clear(tmp_path, write_vfr_video):
    experiment = Experiment.create("p", tmp_path)
    experiment.add_recording("first")
    assert _add_video(experiment, write_vfr_video(tmp_path / "a.mp4")).exit_code == 0
    runner = CliRunner()
    size = runner.invoke(main, ["cache", "previews", "size"])
    assert size.exit_code == 0, size.output
    assert str(previews.preview_cache_dir()) in size.output
    assert " KB" in size.output

    refused = runner.invoke(main, ["cache", "clear"])
    assert refused.exit_code != 0
    assert "--previews" in refused.output

    cleared = runner.invoke(main, ["cache", "clear", "--previews"])
    assert cleared.exit_code == 0, cleared.output
    assert "freed" in cleared.output
    assert previews.preview_cache_size() == 0
    assert [t.kind for t in experiment.missing_previews()] == ["proxy"]


def test_build_reports_each_failure_and_exits_non_zero(tmp_path, write_vfr_video):
    """A part that cannot be read is named on stderr; the other recordings still build."""
    from detail.storage import recording as recording_store

    experiment = Experiment.create("p", tmp_path)
    for name in ("broken", "good"):
        experiment.add_recording(name).add_video_part(
            write_vfr_video(tmp_path / f"{name}.mp4")
        )
    broken = experiment.resolve_recording("broken")
    part = recording_store.resolve_part_video_path(broken.dir, broken.video_parts[0])
    part.write_bytes(b"not a video" * 1000)
    result = CliRunner().invoke(
        main, ["cache", "previews", "build", "--experiment", str(experiment.dir)]
    )
    assert result.exit_code != 0
    assert "preview failed: broken part 1 proxy:" in result.output
    assert "previews: built 1" in result.output
    assert experiment.resolve_recording("good").missing_previews() == []
