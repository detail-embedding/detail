"""The CLI's profile-all and duplicate commands (``detail/cli.py``).

- **One bar over the whole job.** A job over 159 clusters asks for one bar (``fold_units``):
  its count is clusters done plus the running cluster's share, so tqdm's time left is the
  whole job's. A bar reset per cluster would show 159 short bars and a time left for one.
- **A half-second heartbeat does not scroll the terminal.** Only a change of phase or cluster
  is printed above the bar; the elapsed seconds ride in the postfix.
- **The estimate is shown before anything starts**, and ``--estimate`` starts nothing.
"""

from __future__ import annotations

import io

from click.testing import CliRunner

from detail.cli import _fold_tick, main as cli


def _event(done, inner, message):
    return {
        "recordings_done": done,
        "recordings_total": 4,
        "frames_done": inner,
        "frames_total": 1000 if inner is not None else 0,
        "message": message,
        "unit_label": "cluster",
        "fold_units": True,
    }


def test_the_bar_folds_the_running_cluster_into_the_clusters_done(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "stderr", io.StringIO())
    bar, last = None, None
    bar, last = _fold_tick(
        bar, _event(0, 0, "measuring · cluster 1 of 4 · node 9"), "pd", last
    )
    assert bar.total == 4 and bar.n == 0
    bar, last = _fold_tick(
        bar, _event(1, 250, "measuring · cluster 2 of 4 · node 7 · 120 members · 3s"),
        "pd", last,
    )  # fmt: skip
    assert bar.n == 1.25
    bar.close()


def test_only_a_change_of_phase_or_cluster_is_printed(monkeypatch):
    import sys

    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    bar, last = None, None
    for seconds in range(5):
        bar, last = _fold_tick(
            bar,
            _event(
                0, 100 + seconds, f"measuring · cluster 1 of 4 · node 9 · {seconds}s"
            ),
            "pd",
            last,
        )
    bar, last = _fold_tick(
        bar, _event(0, 900, "fitting · cluster 1 of 4 · node 9 · 1s"), "pd", last
    )
    bar.close()
    printed = [
        line for line in err.getvalue().splitlines() if line.startswith("  pd: ")
    ]
    assert printed == [
        "  pd: measuring · cluster 1 of 4 · node 9",
        "  pd: fitting · cluster 1 of 4 · node 9",
    ]


def test_estimate_prints_the_time_and_starts_nothing(
    fitted_temporal, tmp_path, monkeypatch
):
    from detail.storage import locations

    monkeypatch.setattr(locations, "CONFIG_DIR", tmp_path)
    experiment, run_id = fitted_temporal
    experiment.temporal_clustering_run(run_id).rebuild_node_cache()
    before = [r.id for r in experiment.analysis_runs]
    # --clusters all: the default floor (NodeView.min_cluster_s, 30 s for a cache built now)
    # is more than this whole fixture, so it would estimate no cluster at all.
    result = CliRunner().invoke(
        cli,
        [
            "analysis", "profile-all", "--experiment", str(experiment.dir),
            "--run", run_id, "--clusters", "all", "--estimate",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "profiles:" in result.output
    assert [r.id for r in experiment.analysis_runs] == before


def test_duplicate_prints_the_new_id(fitted_temporal):
    experiment, run_id = fitted_temporal
    result = CliRunner().invoke(
        cli, ["analysis", "duplicate", run_id, "--experiment", str(experiment.dir)]
    )
    assert result.exit_code == 0, result.output
    new_id = result.output.strip().rsplit(" ", 1)[-1]
    assert experiment.analysis_record(new_id)["duplicated_from"] == run_id
