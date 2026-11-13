"""``detail analysis cluster-montage``: the clip montage from the command line (§22).

Surface parity: the montage the cluster tree's right-click opens is a capability a script and a
terminal reach too, through the same ``api.clip_montage`` exports — the montage, one GIF per clip,
and the clip list — with the window's defaults (25 clips, most typical first, the mask and its
trail). Pinned on the session's real temporal fit: the file is written with the width asked for, the
tick rules the window enforces are refused here with the same reason, and a run that is not a
clustering run is named as such.
"""

import pytest
from click.testing import CliRunner
from PIL import Image

from detail.cli import main


def _invoke(proj, run_id, *extra):
    return CliRunner().invoke(
        main,
        ["analysis", "cluster-montage", "--experiment", str(proj.dir), "--run", run_id,
         *extra],
    )  # fmt: skip


def test_a_montage_gif_is_written_at_the_width_asked(fitted_temporal, tmp_path):
    proj, run_id = fitted_temporal
    root = proj.temporal_clustering_run(run_id).node_view().root
    out = tmp_path / "montage.gif"
    result = _invoke(
        proj, run_id, "--node", str(root), "--out", str(out), "--mask", "--clips", "4",
        "--max-width", "120", "--fps", "5", "--blind",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert f"wrote {out}" in result.output
    with Image.open(out) as gif:
        assert 100 <= gif.size[0] <= 120
        assert gif.info["duration"] == 200


def test_a_trail_without_what_it_trails_is_refused(fitted_temporal, tmp_path):
    proj, run_id = fitted_temporal
    result = _invoke(
        proj, run_id, "--node", "0", "--out", str(tmp_path / "m.mp4"), "--raw", "--no-mask",
        "--mask-trail",
    )  # fmt: skip
    assert result.exit_code != 0
    assert "mask trail trails the mask" in result.output
    assert not (tmp_path / "m.mp4").exists()


def test_a_run_that_is_not_a_clustering_run_is_named(fitted_temporal, tmp_path):
    proj, _run_id = fitted_temporal
    result = _invoke(proj, "an_99", "--node", "0", "--out", str(tmp_path / "m.gif"))
    assert result.exit_code != 0
    assert "not a temporal-clustering run" in result.output


def test_the_cli_defaults_are_the_windows():
    """`--help` reads the API's defaults, so the terminal and the window open on one montage."""
    from detail.api import clip_montage as cm
    from detail.cli import analysis_cluster_montage

    defaults = {p.name: p.default for p in analysis_cluster_montage.params}
    # Unset, --clips is the montage's 25, or the overlay's 12 with --overlay; the help says so.
    assert defaults["limit"] is None
    clips = next(p for p in analysis_cluster_montage.params if p.name == "limit")
    assert f"{cm.DEFAULT_CLIPS}, or {cm.OVERLAY_CLIPS} with --overlay" in clips.help
    assert (cm.DEFAULT_CLIPS, cm.OVERLAY_CLIPS) == (25, 12)
    assert defaults["align"] is False and defaults["overlay"] is False
    assert defaults["sort"] == cm.DEFAULT_SORT
    assert (defaults["raw"], defaults["mask"], defaults["keypoints"]) == (
        False, True, False,
    )  # fmt: skip
    assert (
        defaults["mask_trail"] is None
    ), "the trail follows --mask, so it is on by default"
    assert defaults["keypoint_trail"] is False


def test_nothing_to_write_is_refused(fitted_temporal):
    proj, run_id = fitted_temporal
    result = _invoke(proj, run_id, "--node", "0")
    assert result.exit_code != 0
    assert (
        "at least one of --out, --per-clip-gifs, --csv and --export-dir"
        in result.output
    )


def test_per_clip_gifs_and_a_list_in_the_order_asked(fitted_temporal, tmp_path):
    import csv

    from detail.api import clip_montage as cm

    proj, run_id = fitted_temporal
    run = proj.temporal_clustering_run(run_id)
    root = run.node_view().root
    gifs, listing = tmp_path / "gifs", tmp_path / "clips.csv"
    result = _invoke(
        proj, run_id, "--node", str(root), "--per-clip-gifs", str(gifs), "--csv",
        str(listing), "--clips", "3", "--sort", "least-typical", "--max-width", "80",
        "--fps", "5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert f"wrote 3 GIFs to {gifs}" in result.output
    expected = cm.cluster_montage_clips(proj, run, root, limit=3, sort="least-typical")
    assert [c.rank for c in expected][0] > [c.rank for c in expected][-1]
    assert sorted(p.name for p in gifs.iterdir()) == sorted(
        f"{cm.clip_file_stem(c)}.gif" for c in expected
    )
    with Image.open(next(gifs.iterdir())) as gif:
        assert gif.size[0] == 80 and gif.info["duration"] == 200
    with open(listing, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["rank"]) for r in rows] == [c.rank for c in expected]
    assert {(r["run_id"], r["node"]) for r in rows} == {(run_id, str(root))}


# --- --export-dir: the window's Export clips... ------------------------------------------- #
def test_export_dir_writes_each_visual_and_format_in_its_own_folder(
    fitted_temporal, tmp_path
):
    import cv2

    from detail.api import clip_montage as cm

    proj, run_id = fitted_temporal
    run = proj.temporal_clustering_run(run_id)
    root = run.node_view().root
    out = tmp_path / "export"
    result = _invoke(
        proj, run_id, "--node", str(root), "--clips", "2", "--export-dir", str(out),
        "--visual", "current", "--visual", "raw-outline", "--format", "mp4",
        "--format", "gif", "--mp4-width", "160", "--mp4-fps", "10", "--gif-width", "96",
        "--gif-fps", "5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    clips = cm.cluster_montage_clips(proj, run, root, limit=2)
    stems = ["montage", *(cm.clip_file_stem(c) for c in clips)]
    expected = sorted(
        f"{folder}/{stem}.{ext}"
        for folder in ("current_Mask_Mask_trail", "Raw_mask_outline")
        for ext in ("mp4", "gif")
        for stem in stems
    )
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*")) == expected
    assert f"wrote {len(expected)} files into {out}" in result.output
    for gif_path in out.rglob("*.gif"):
        with Image.open(gif_path) as gif:
            assert gif.size[0] == 96 and gif.info["duration"] == 200
    for mp4 in out.rglob("*.mp4"):
        cap = cv2.VideoCapture(str(mp4))
        try:
            assert round(cap.get(cv2.CAP_PROP_FPS)) == 10
            assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
        finally:
            cap.release()


def test_export_dir_writes_only_the_montage_when_asked(fitted_temporal, tmp_path):
    proj, run_id = fitted_temporal
    root = proj.temporal_clustering_run(run_id).node_view().root
    out = tmp_path / "only"
    result = _invoke(
        proj, run_id, "--node", str(root), "--clips", "2", "--export-dir", str(out),
        "--write", "montage", "--format", "gif", "--visual", "mask",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert [p.relative_to(out).as_posix() for p in out.rglob("*.*")] == [
        "Mask/montage.gif"
    ]


def test_the_export_defaults_are_the_dialogs():
    """Both formats, both kinds of file, the current ticks; a 640 px 12 fps GIF and a
    full-size mp4 at the footage's own rate."""
    from detail.api import montage_export as me
    from detail.cli import analysis_cluster_montage

    # Parsed, not read off the options: a repeatable option's unset default is a sentinel.
    defaults = analysis_cluster_montage.make_context(
        "cluster-montage", ["--experiment", ".", "--run", "an_01", "--node", "0"]
    ).params
    assert defaults["write_what"] == me.DEFAULT_WRITE == "both"
    assert defaults["visuals"] in ((), None), "empty means the current ticks"
    assert defaults["export_formats"] in ((), None), "empty means mp4 and gif"
    assert (defaults["gif_width"], defaults["gif_fps"]) == (640, 12.0)
    assert (defaults["mp4_width"], defaults["mp4_fps"]) == (None, None)
    choices = next(p for p in analysis_cluster_montage.params if p.name == "visuals")
    assert list(choices.type.choices) == list(me.VISUAL_CHOICES)


# --- --align and --overlay (§22 *Align to heading, and Overlay*) ----------------------------- #
def test_overlay_without_out_is_refused(fitted_temporal, tmp_path):
    proj, run_id = fitted_temporal
    result = _invoke(
        proj, run_id, "--node", "0", "--overlay", "--csv", str(tmp_path / "c.csv")
    )
    assert result.exit_code != 0
    assert "--overlay writes --out" in result.output


@pytest.mark.parametrize(
    "extra", [("--align", "--out", "m.gif"), ("--overlay", "--out", "o.png")]
)
def test_turning_clips_with_no_heading_is_refused_with_the_reason(
    fitted_temporal, tmp_path, extra
):
    """No clip of the session's fit has a heading here: nothing can be turned, and the command
    says why instead of writing a file of grey tiles."""
    proj, run_id = fitted_temporal
    root = proj.temporal_clustering_run(run_id).node_view().root
    flag, out_flag, name = extra
    result = _invoke(
        proj, run_id, "--node", str(root), "--clips", "2", flag, out_flag,
        str(tmp_path / name),
    )  # fmt: skip
    assert result.exit_code != 0
    assert "No clip can be aligned: " in result.output
    assert not (tmp_path / name).exists()


def test_align_reaches_every_visual_of_an_export(
    monkeypatch, fitted_temporal, tmp_path
):
    """--align is passed to Export clips' looks, the ticks' and every --visual's."""
    from detail.api import montage_export

    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(montage_export, "export_cluster_clips", fake)
    proj, run_id = fitted_temporal
    result = _invoke(
        proj, run_id, "--node", "0", "--align", "--export-dir", str(tmp_path / "e"),
        "--visual", "mask",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen["align"] is True and seen["current"].aligned is True
