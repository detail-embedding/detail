"""``detail analysis cluster-montage``: the clip montage from the command line (§22).

Surface parity: the montage the cluster tree's right-click opens is a capability a script and a
terminal reach too, through the same ``api.clip_montage`` — the montage, one GIF per clip, and
the clip list — with the window's defaults (25 clips, most typical first, the mask and its
trail; the defaults themselves are pinned in ``tests/api/test_montage_surface_parity.py``).
Run over **both** clustering topologies (``fitted_clustering``): a pose fit, whose members are
frames, and a temporal one, whose members are bouts, which is the montage's real shape.
"""

import csv

from click.testing import CliRunner
from PIL import Image

from detail.cli import main


def _invoke(proj, run_id, *extra):
    return CliRunner().invoke(
        main,
        ["analysis", "cluster-montage", "--experiment", str(proj.dir), "--run", run_id,
         *extra],
    )  # fmt: skip


def test_a_montage_gif_is_written_at_the_width_asked(fitted_clustering, tmp_path):
    proj, run = fitted_clustering
    root = run.node_view().root
    out = tmp_path / "montage.gif"
    result = _invoke(
        proj, run.id, "--node", str(root), "--out", str(out), "--mask", "--clips", "4",
        "--max-width", "120", "--fps", "5", "--blind",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert f"wrote {out}" in result.output
    with Image.open(out) as gif:
        assert 100 <= gif.size[0] <= 120
        assert gif.info["duration"] == 200


def test_a_trail_without_what_it_trails_is_refused(fitted_clustering, tmp_path):
    proj, run = fitted_clustering
    result = _invoke(
        proj, run.id, "--node", "0", "--out", str(tmp_path / "m.mp4"), "--raw", "--no-mask",
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


def test_nothing_to_write_is_refused(fitted_temporal):
    proj, run_id = fitted_temporal
    result = _invoke(proj, run_id, "--node", "0")
    assert result.exit_code != 0
    assert "at least one of --out, --per-clip-gifs and --csv" in result.output


def test_a_picture_size_without_a_picture_is_a_usage_error(fitted_temporal, tmp_path):
    """``--max-width`` and ``--fps`` shape the montage or the per-clip GIFs; given with only
    ``--csv`` they would be silently ignored, which is the worse failure."""
    proj, run_id = fitted_temporal
    for extra in (("--max-width", "300"), ("--fps", "5")):
        result = _invoke(
            proj, run_id, "--node", "0", "--csv", str(tmp_path / "c.csv"), *extra
        )
        assert result.exit_code != 0, extra
        assert "shape a picture" in result.output
    assert not (tmp_path / "c.csv").exists()


def test_per_clip_gifs_and_a_list_in_the_order_asked(fitted_clustering, tmp_path):
    from detail.api import clip_montage as cm

    proj, run = fitted_clustering
    view = run.node_view()
    # A node with a few clips to order: a pose run's root is one long clip once frames join.
    root = max(range(view.tree.n_nodes), key=lambda n: len(view.clips(n)))
    assert len(view.clips(root)) >= 3
    gifs, listing = tmp_path / "gifs", tmp_path / "clips.csv"
    result = _invoke(
        proj, run.id, "--node", str(root), "--per-clip-gifs", str(gifs), "--csv",
        str(listing), "--clips", "3", "--sort", "least-typical", "--max-width", "80",
        "--fps", "5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert f"wrote 3 GIFs to {gifs}" in result.output
    expected = cm.montage_clips(proj, run, root, limit=3, sort="least-typical")
    assert [c.rank for c in expected][0] > [c.rank for c in expected][-1]
    assert sorted(p.name for p in gifs.iterdir()) == sorted(
        f"{cm.clip_file_stem(c)}.gif" for c in expected
    )
    with Image.open(next(gifs.iterdir())) as gif:
        assert gif.size[0] == 80 and gif.info["duration"] == 200
    with open(listing, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["rank_typical"]) for r in rows] == [c.rank for c in expected]
    assert {(r["run_id"], r["node"]) for r in rows} == {(run.id, str(root))}
    assert {r["kind"] for r in rows} == {
        "query"
    }, "a clip-query CSV: every row an example"


def test_context_widens_the_tiles_and_the_list_still_names_the_clips(
    fitted_clustering, tmp_path, monkeypatch
):
    """``--context`` plays footage either side of each clip (the cluster window's Context);
    ``--aligned`` turns the tiles (on a run with no keypoints, every frame plays as recorded).
    The CSV list names each clip's own span, never its context."""
    from detail.api import clip_montage as cm

    proj, run = fitted_clustering
    view = run.node_view()
    root = max(range(view.tree.n_nodes), key=lambda n: len(view.clips(n)))
    built = []
    real = cm.ClusterMontage

    def recording(experiment_dir, clips, **kw):
        built.append(list(clips))
        return real(experiment_dir, clips, **kw)

    monkeypatch.setattr(cm, "ClusterMontage", recording)
    out, listing = tmp_path / "m.gif", tmp_path / "clips.csv"
    result = _invoke(
        proj, run.id, "--node", str(root), "--out", str(out), "--csv", str(listing),
        "--clips", "2", "--context", "1", "--aligned", "--max-width", "80", "--fps", "5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    expected = cm.montage_clips(proj, run, root, limit=2)
    assert built[0] == cm.with_context(expected, 1.0), "the montage plays the context"
    with open(listing, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["start_t_ns"]) for r in rows] == [c.start_t_ns for c in expected]
