"""End-to-end construction of RunDetailPage → a module's result_widget on REAL
on-disk run data — the strongest version of the GUI construction net.

The earlier tracking-preview crash (a result_widget building a run-preview widget
wired to a renamed method) lived exactly on this path: RunDetailPage._build →
GUI_MODULES[module].result_widget(experiment, record) → the run-preview widget. This
builds a experiment with a completed run for every module and asserts the page (hence
the result view) constructs.

The runs are synthesized without model inference: the detection + embedding zarrs
are written directly, and tracking runs the real AnchorTracker over them.

One module's page is covered elsewhere: pose clustering needs a real fit rather than a
synthesized run record, so ``tests/gui/test_cluster_workspace.py`` builds its
``AnalysisRunDetailPage`` off the one fit that file already makes, instead of paying for a
second one here.
"""

import cv2
import numpy as np
import pytest

from tests.gui.conftest import drain_busy_calls, drain_thread_pool  # noqa: E402
import yaml

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.core.dataclasses import TrackingJob  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.pages import RunDetailPage  # noqa: E402
from detail.gui.tables import row_id  # noqa: E402
from detail.gui.workers import StageProgressWorker  # noqa: E402
from detail.gui.run_preview import (  # noqa: E402
    DetectionRunPreview,
    EmbeddingRunPreview,
    TrackingRunPreview,
)
from detail.storage import runs as runs_store  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402
from detail.storage.zarr_io import (  # noqa: E402
    DetectionPartWriter,
    EmbeddingPartWriter,
)
from tests.gui import subjects  # noqa: E402
from tests import waits  # noqa: E402


@pytest.fixture
def completed_runs_experiment(tmp_path):
    """A experiment whose single recording has a completed run for every module, with
    real outputs on disk + the experiment-level run records RunDetailPage reads.

    The body lives in `tests/gui/subjects.py` because the conformance walk needs the same
    experiment — a detail page cannot be walked without a record to open. One builder, two
    callers, rather than the same 100 lines written twice and drifting.
    """
    return subjects.an_experiment_with_completed_runs(tmp_path)


_RESULT_VIEW = {
    "detection": DetectionRunPreview,
    "tracking": TrackingRunPreview,
    "embedding": EmbeddingRunPreview,
}


@pytest.mark.parametrize("module_id", ["detection", "tracking", "embedding"])
def test_run_detail_builds_result_widget(qapp, completed_runs_experiment, module_id):
    proj = completed_runs_experiment
    record = next(r for r in proj.processing_runs if r["module_id"] == module_id)
    page = RunDetailPage(proj, record["id"])  # _build → result_widget on real data
    # the complete-run result view is actually present (not the not-found path)
    if module_id in _RESULT_VIEW:
        assert page.findChildren(_RESULT_VIEW[module_id])


def test_run_detail_shows_next_stage_button(qapp, completed_runs_experiment):
    from PyQt6.QtWidgets import QPushButton

    proj = completed_runs_experiment

    def _labels(module_id):
        run = next(r for r in proj.processing_runs if r["module_id"] == module_id)
        page = RunDetailPage(proj, run["id"])
        return [b.text() for b in page.findChildren(QPushButton)]

    assert "Run tracking →" in _labels("detection")
    assert "Run embedding →" in _labels("tracking")
    # embedding is the last core stage — no next-step button
    assert not any(t.startswith("Run ") for t in _labels("embedding"))
    # The offered stages are the registry's (upstream == this module, or a second input it
    # reads, per-recording, surfaced) in pipeline order — not a hand map, which named one next
    # stage per module and would have hidden the second stage that reads a run.
    from detail.gui.pages import _next_stages

    # One straight line, whatever the data reads (§26, revised 2026-10-21): keypoint detect
    # reads only a detection run but is offered after tracking (`offered_after`), and keypoint
    # track — which reads tracking as a second input — is offered only from a keypoint detect
    # run, never from tracking, where it would be a button with nothing to run.
    assert _next_stages("detection") == ["tracking"]
    assert _next_stages("tracking") == ["keypoint_detect", "embedding"]
    assert _next_stages("keypoint_detect") == ["keypoint_track"]
    assert (
        _next_stages("embedding") == []
    )  # nothing reads an embedding run as a per-recording stage
    offered = [t for t in _labels("detection") if t.startswith("Run ")]
    assert offered == [f"Run {s} →" for s in _next_stages("detection")]
    assert "Run keypoint_detect →" not in offered
    tracking_offers = _labels("tracking")
    assert "Run keypoint_detect →" in tracking_offers
    assert "Run keypoint_track →" not in tracking_offers


# --- a run's model and recordings show names, not raw ids (§11) ------------- #
def test_run_detail_shows_the_model_name_and_recording_names(
    qapp, completed_runs_experiment
):
    """The reported "model name is getting truncated": there is no truncation anywhere —
    the label rendered the raw model id. Same rule for the per-recording table, which
    listed recording ids while `recording_names()` was right there.

    Built on the fixture with **real** run outputs: a page whose detection run has no zarr
    fails its result-widget load off-thread and pops a modal error, which then breaks
    whatever spins the event loop next.
    """
    from PyQt6.QtWidgets import QLabel, QTableView

    from detail.gui.pages import RunDetailPage
    from detail.storage import models as models_store
    from detail.storage import recording as recording_store

    proj = completed_runs_experiment
    recording = proj.get_recording("recording_001")
    recording.set_name("Fish tank A")

    # Point the detection run at a *named* trained model, as a distilled run would be.
    mdir = models_store.model_dir(proj.dir, "model_001")
    mdir.mkdir(parents=True, exist_ok=True)
    models_store.write_model_yaml(
        mdir, {"id": "model_001", "name": "Fish_YOLO", "status": "complete"}
    )
    data = recording_store.read_recording(recording.dir)
    for entry in data["detection_runs"]:
        entry["model_id"] = "model_001"
        entry["weights_source"] = "trained"
    recording_store.write_recording(recording.dir, data)

    detection_run = next(
        r for r in proj.processing_runs if r.get("module_id") == "detection"
    )
    page = RunDetailPage(proj, detection_run["id"])

    texts = [lb.text() for lb in page.findChildren(QLabel)]
    assert any("Fish_YOLO" in t for t in texts)  # the name the user gave it...
    assert any("model_001" in t for t in texts)  # ...with the id still greppable

    per_table = next(
        t
        for t in page.findChildren(QTableView)
        if t.model() and t.model().columnCount() >= 4
    )
    cells = [
        per_table.model().index(r, c).data()
        for r in range(per_table.model().rowCount())
        for c in range(per_table.model().columnCount())
    ]
    assert "Fish tank A" in cells  # the recording's name, not recording_001


@pytest.mark.parametrize("module_id", ["detection", "tracking", "embedding"])
def test_the_results_list_names_its_recordings_and_opens_one(
    qapp, completed_runs_experiment, module_id
):
    """The list *is* the picker now.

    It used to be a table of recordings with, underneath it, a result view that opened with a
    **dropdown of the same recordings** — two controls for one choice, and together taller than
    the results they were choosing between. Picking a row shows that recording's results; the
    row still carries the id it is about, so what opens is not decided by where the row sits.
    """
    from PyQt6.QtWidgets import QComboBox

    from detail.gui.pages import RunResults

    proj = completed_runs_experiment
    proj.get_recording("recording_001").set_name("Fish tank A")
    record = next(r for r in proj.processing_runs if r["module_id"] == module_id)
    page = RunDetailPage(proj, record["id"])

    results = page.findChild(RunResults)
    model = results._table.view.model()
    assert model.index(0, 0).data() == "Fish tank A"
    assert row_id(model.index(0, 0)) == "recording_001"  # the id rides along
    drain_busy_calls(qapp)
    drain_thread_pool(qapp)
    # The predicate names what a *recording* dropdown is, rather than "a populated combo carrying
    # item data". The frame scrubber's playback-speed picker is also that — `scrubber_speed`, items
    # 0.25x..8x with float data — so the broad version fires on it the moment the scrubber is built.
    # Which is why this ran before any wait: it passed only while the scrubber did not exist yet,
    # and would have false-fired as soon as it did. Unable to do its job in either timing (#533).
    ids = set(proj.recording_ids)
    assert not [
        c
        for c in results.findChildren(QComboBox)
        if any(c.itemData(i) in ids for i in range(c.count()))
    ], "a recording dropdown is back beside the list that replaced it"

    # Opening on a recording rather than on an empty space.
    assert isinstance(results._shown, _RESULT_VIEW[module_id])


def test_a_recording_that_failed_says_so_where_its_results_would_be(qapp, tmp_path):
    """The Error column's job, moved to where the results go.

    The column was as wide as its worst row on every run, including the runs where every row
    was `complete` and it was empty. The reason is worth a paragraph and is only ever wanted
    for one recording at a time — the one whose row you just picked.
    """
    from detail.gui.pages import RunResults
    from detail.storage import runs as runs_store

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001").set_name("Fish tank A")
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "tracking",
            "recording_ids": ["recording_001"],
            "config": {},
            "status": "failed",
            "per_recording": {
                "recording_001": {
                    "run_id": "track_01",
                    "status": "failed",
                    "error": "the detector found no subject",
                }
            },
            "created_at": "2026-06-09T14:00:00Z",
        },
    )
    page = RunDetailPage(proj, "run_001")
    results = page.findChild(RunResults)
    shown = results._shown.text()
    assert "failed" in shown
    assert "the detector found no subject" in shown


def test_a_run_with_no_per_recording_results_says_so(qapp, tmp_path):
    """The results list's empty state, asserted where it means something.

    `test_empty_states.py` exempts this list from its sweep of collection pages — a run's
    recordings are not a collection you fill — and points here for the empty state instead.
    """
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import RunResults

    proj = Experiment.create("p", tmp_path)
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "tracking",
            "recording_ids": [],
            "config": {},
            "status": "failed",
            "per_recording": {},
            "created_at": "2026-06-09T14:00:00Z",
        },
    )
    page = RunDetailPage(proj, "run_001")
    results = page.findChild(RunResults)
    shown = [label.text() for label in results.findChildren(QLabel)]
    assert any("no per-recording results" in text for text in shown), shown


def test_analysis_run_detail_builds_result_widget(qapp, tmp_path):
    # AnalysisRunDetailPage._build → the module's result_widget on a completed
    # analysis.yaml record (the §16 analogue of the processing run-detail net).
    from PyQt6.QtWidgets import QTableView

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "tracking_run_id": "track_01",
            "status": "complete",
            "artifacts": [
                {"path": "recording_001/tracks.csv", "kind": "tracks", "label": "x"}
            ],
        },
    )
    page = AnalysisRunDetailPage(proj, "an_01")
    assert page.findChildren(QTableView)  # the artifacts result table is present
    from PyQt6.QtWidgets import QPushButton

    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Open output folder" in labels  # reveals analysis/<id>/ in the file manager


def test_the_measures_panel_shows_durations_in_seconds():
    """The artifact stores nanoseconds (Tenet 8) and the screen is for a person.

    Rendered raw, a 20 ms mean bout reads `2e+07` — a number nobody can act on, in a column
    that also holds counts and ratios. The CSV keeps nanoseconds; this is the difference
    between the machine's format and the reader's.
    """
    from detail.analysis.comparison_report import shown_value

    assert shown_value(2e7, "ns") == ("0.02", "s")
    assert shown_value(20.0, "1/s") == ("20", "1/s")
    # A fraction reads as a percent. The panel had its own copy of this function, written
    # before that branch existed, so `time_fraction` printed `0.125 fraction` here and
    # `12.5 %` on the comparison page for the same number.
    assert shown_value(0.125, "fraction") == ("12.5", "%")
    assert shown_value(2.0, "") == ("2", "")
    # A refused measurement stays blank rather than becoming a zero.
    assert shown_value("", "ns") == ("", "ns")


def test_the_measures_panel_prints_what_the_comparison_page_prints(qapp):
    """The class is a second formatter, so assert on what the *panel* shows, not the formatter.

    The test above pins `shown_value`; a copy of it inside `MeasuresPanel` would leave that green
    — which is exactly how the `0.125 fraction` / `12.5 %` split shipped. So drive the panel's own
    row builder with the two units that have a conversion and compare its cells, whole, with the
    comparison page's.
    """
    from detail.gui.pages import MeasuresPanel

    rows = [
        {
            "recording_id": "recording_001",
            "subject": 0,
            "label": "rear",
            "label_to": None,
            "measure": "time_fraction",
            "value": 0.125,
            "n": 3,
        },
        {
            "recording_id": "recording_001",
            "subject": 0,
            "label": "rear",
            "label_to": None,
            "measure": "total_time",
            "value": 2e7,
            "n": 3,
        },
    ]

    class _Measures:
        def table(self, progress=None):
            return rows

    class _Experiment:
        def measures(self, run_id):
            return _Measures()

    shown = MeasuresPanel(_Experiment(), "an_00")._compute(lambda *a, **k: None)
    assert [(r[4], r[5]) for r in shown] == [("12.5", "%"), ("0.02", "s")], (
        "the measures panel renders a number differently from the comparison page — it has "
        "its own formatter again, rather than calling comparison_report.shown_value"
    )


def test_analysis_run_detail_shows_measures_for_a_run_with_an_artifact(
    qapp, tmp_path, monkeypatch
):
    """The §23 panel, through the page a user actually opens.

    Built here rather than in isolation, and against a run that really wrote the §19 artifact:
    every other analysis case in this file writes a record without one, so a panel tested
    through them would only ever exercise its absent-artifact path and pass while the table
    never built.

    `run_busy` is driven on this thread — a real QThread here segfaults the GUI suite
    (`test_clusterers.py::_sync_busy` is the same stand-in, for the same reason).
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QTableView

    from detail.gui import pages as gui_pages
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from tests.api.test_comparison import UNBALANCED, _cohort

    monkeypatch.setattr(
        gui_pages,
        "run_busy",
        lambda target, fn, on_done, **kw: on_done(fn()),
    )
    # Its determinate twin, which the panel uses since measuring may first rebuild frame views
    # an edit deferred and so has a real bar to show.
    monkeypatch.setattr(
        gui_pages,
        "run_progress",
        lambda target, work, on_done, **kw: on_done(work(lambda *_: None)),
    )
    proj = _cohort(tmp_path, UNBALANCED)
    # `_cohort` writes the artifact into analysis/an_01/ but no run record; the page reads one.
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "query",
            "recording_ids": sorted(UNBALANCED),
            "status": "complete",
            "artifacts": [],
        },
    )

    page = AnalysisRunDetailPage(proj, "an_01")
    page.show()  # the panel measures on first show, not on construction
    tables = [t for t in page.findChildren(QTableView) if t.model()]
    assert tables, "the measures panel never built a table"
    model = tables[-1].model()
    headers = [
        model.headerData(c, Qt.Orientation.Horizontal)
        for c in range(model.columnCount())
    ]
    assert "Measure" in headers and "Unit" in headers, headers
    assert model.rowCount() > 0, "the artifact has instances, so it has measures"


def test_a_clip_query_run_opens_as_a_bare_record(qapp, tmp_path):
    """A Behavior query (`clip_query`) run on disk, now that the module is removed.

    It used to build a matches table here. The record still lists and its page still opens,
    with its status and output folder and no result view.
    """
    from PyQt6.QtWidgets import QLabel, QPushButton, QTableView

    from detail.executor.runners import ANALYSIS_MODULES

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage.clip_query_csv import ClipRow, write_clip_csv

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    write_clip_csv(
        adir / "matches.csv",
        [
            ClipRow("query", 0, "recording_001", 0, "emb_01", 0, 100, 0, 3, 0.0),
            ClipRow("match", 1, "recording_001", 0, "emb_01", 900, 1000, 27, 30, 0.02),
        ],
    )
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "clip_query",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "artifacts": [
                {"path": "matches.csv", "kind": "clip_matches", "label": "matches"}
            ],
        },
    )
    assert "clip_query" not in ANALYSIS_MODULES
    assert [(r.id, r.module_id) for r in proj.analysis_runs] == [
        ("an_01", "clip_query")
    ]
    page = AnalysisRunDetailPage(proj, "an_01")
    assert not page.findChildren(
        QTableView
    ), "a result view was built for a removed module"
    assert "complete" in " ".join(label.text() for label in page.findChildren(QLabel))
    assert [
        b for b in page.findChildren(QPushButton) if b.text() == "Open output folder"
    ]


def test_analysis_run_detail_builds_query_result_widget(qapp, tmp_path):
    """A §20 query run renders its **workspace** while `open` — the authored status.

    What this pins beyond construction: that `pages.py`'s status gate lets an authored run
    through, so the calibration surface exists while the run is being worked on rather than
    only when it is finished. §21's whole point is that the statistics are what the user is
    deciding against, so a workspace that appeared only at `complete` would arrive after the
    decision it exists to support.
    """
    from PyQt6.QtWidgets import QLabel, QTabBar

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage import query_runs

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    query_runs.write_query_state(
        adir,
        query_runs.QueryState(
            embed_run_id="emb_01",
            track_id=0,
            method="soft_knn",
            behaviors=[
                query_runs.Behavior(
                    id="q01",
                    name="rearing",
                    examples=[{"recording_id": "recording_001", "track_id": 0,
                               "start_t_ns": 0, "end_t_ns": 3_000_000_000}],
                )
            ],  # fmt: skip
            search_recording_ids=["recording_001"],
        ),
    )
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "query",
            "recording_ids": ["recording_001"],
            "status": "open",
            "behaviors_searched": ["q01"],
            "behaviors_failed": {},
        },
    )
    page = AnalysisRunDetailPage(proj, "an_01")
    # The behavior is the selector's tab, not a label: switching behavior keeps the current
    # tab (§20), so the selector lives outside the tab widget rather than inside a summary.
    selectors = [
        bar
        for bar in page.findChildren(QTabBar)
        if [bar.tabText(i) for i in range(bar.count())] == ["rearing"]
    ]
    assert selectors, "the workspace has no behavior selector"
    text = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "an_01 is open" in text  # the authored status, stated


def test_a_syllable_discovery_run_opens_as_a_bare_record(qapp, tmp_path):
    """A Behavior discovery (`syllable_discovery`) run on disk, now that the module is removed.

    It used to build the syllables and members tables here. The record still lists and its page
    still opens, with its status and no result view.
    """
    from PyQt6.QtWidgets import QLabel, QTableView

    from detail.executor.runners import ANALYSIS_MODULES

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage.clip_query_csv import ClipRow, write_clip_csv

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    write_clip_csv(
        adir / "syllables.csv",
        [
            ClipRow("match", 1, "recording_001", 0, "emb_01", 0, 400, 0, 12, 0.0,
                    query_id="recording_001:00", query_name="recording_001 · syllable 00"),
            ClipRow("match", 2, "recording_001", 0, "emb_01", 900, 1300, 27, 39, 0.0,
                    query_id="recording_001:00", query_name="recording_001 · syllable 00"),
            ClipRow("match", 1, "recording_001", 0, "emb_01", 2000, 2400, 60, 72, 0.0,
                    query_id="recording_001:01", query_name="recording_001 · syllable 01"),
        ],
    )  # fmt: skip
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "syllable_discovery",
            "recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "n_syllables": 2,
            "params": {"detector": "novelty"},
            "recordings": [
                {
                    "recording_id": "recording_001",
                    "track_id": 0,
                    "n_bouts": 3,
                    "n_syllables": 2,
                    "sweep_counts": {4: 2},
                    "syllable_scales": {"recording_001:00": 4, "recording_001:01": 8},
                    "cadence_fps": 15.0,
                    "window_rows": 8,
                    "min_len_rows": 4,
                    "band_rows": 4,
                    "max_gap_ns": 100_000_000,
                    "n_runs": 5,
                }
            ],
            "status": "complete",
            "artifacts": [
                {
                    "path": "syllables.csv",
                    "kind": "syllable_exemplars",
                    "label": "syllables",
                }
            ],
        },
    )
    assert "syllable_discovery" not in ANALYSIS_MODULES
    assert [r.module_id for r in proj.analysis_runs] == ["syllable_discovery"]
    page = AnalysisRunDetailPage(proj, "an_01")
    assert not page.findChildren(
        QTableView
    ), "a result view was built for a removed module"
    assert "complete" in " ".join(lbl.text() for lbl in page.findChildren(QLabel))


def _cluster_eval_run(tmp_path, *, unlabeled=False):
    """An experiment with one completed cluster-eval run: a real long-format metrics.csv plus the
    per-recording summaries the record carries."""
    import csv as _csv

    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    rows = [
        # the label-free panel is written for every recording, labeled or not
        ("recording_001", "overall", "", "internal.coverage", 0.72),
        ("recording_001", "overall", "", "internal.n_clusters", 5),
        ("recording_001", "overall", "", "internal.transition_entropy", 1.85),
        ("recording_001", "overall", "", "internal.duration_median", 14),
    ]
    if not unlabeled:
        rows += [
            ("recording_001", "overall", "excluded", "ami", 0.61),
            ("recording_001", "overall", "singletons", "ami", 0.55),
            ("recording_001", "overall", "excluded", "ari", 0.48),
            ("recording_001", "overall", "excluded", "bcubed_f1", 0.66),
            ("recording_001", "overall", "", "segmental_f1@0.5", 0.51),
            ("recording_001", "overall", "", "covering", 0.70),
            ("recording_001", "groom", "", "precision", 0.80),
            ("recording_001", "groom", "", "recall", 0.60),
            ("recording_001", "groom", "", "f1", 0.686),
        ]
    with open(adir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["recording_id", "scope", "noise_policy", "metric", "value"])
        writer.writerows(rows)
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "cluster_eval",
            "recording_ids": ["recording_001"],
            "clusterer_id": "clusterer_001",
            "params": {
                "clusterer_id": "clusterer_001",
                "behaviors": ["groom"],
                "boundary_margins": [0, 1, 2, 5],
                "point_window_s": 0.0,
                "use_names": False,
                "require_ground_truth": not unlabeled,
            },
            "recordings": [
                {
                    "recording_id": "recording_001",
                    "has_ground_truth": not unlabeled,
                    "n_frames": 100,
                }
            ],
            "status": "complete",
            "artifacts": [{"path": "metrics.csv", "kind": "cluster_metrics"}],
        },
    )
    return proj


def test_a_cluster_eval_run_opens_as_a_bare_record(qapp, tmp_path):
    """A Clustering eval (`cluster_eval`) run on disk, now that the clusterer chain is removed.

    It used to build the three metric panels here. The record still lists and deletes, and its
    page still opens with its status and no result view. The fitted clusterer it scored lives
    under `clusterers/`, which nothing lists any more; the run does not need it to open.
    """
    from PyQt6.QtWidgets import QLabel, QTableView

    from detail.executor.runners import ANALYSIS_MODULES
    from detail.gui.pages import AnalysisRunDetailPage

    proj = _cluster_eval_run(tmp_path)
    assert "cluster_eval" not in ANALYSIS_MODULES
    assert [r.module_id for r in proj.analysis_runs] == ["cluster_eval"]
    page = AnalysisRunDetailPage(proj, "an_01")
    assert not page.findChildren(
        QTableView
    ), "a result view was built for a removed module"
    assert "complete" in " ".join(label.text() for label in page.findChildren(QLabel))
    proj.delete_analysis("an_01")
    assert proj.analysis_runs == []


def _table_cells(page) -> set[str]:
    """Every cell of every ``pages._table`` on a page (a QTableView + QStandardItemModel)."""
    from PyQt6.QtGui import QStandardItemModel
    from PyQt6.QtWidgets import QTableView

    cells = set()
    for table in page.findChildren(QTableView):
        model = table.model()
        if not isinstance(model, QStandardItemModel):
            continue
        for row in range(model.rowCount()):
            for column in range(model.columnCount()):
                item = model.item(row, column)
                if item is not None:
                    cells.add(item.text())
    return cells


def test_analysis_run_detail_builds_query_eval_result_widget(qapp, tmp_path):
    # The query-eval result view reads the aggregate metrics + per-fold detail + PR
    # curves through QueryEvalPreview and builds the report (per-behavior table, a
    # behavior-driven per-fold table, and the mean±std summary). A skipped behavior
    # renders its reason; a 2-fold run shows both folds.
    from PyQt6.QtWidgets import QLabel, QTableView

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage import query_eval_csv as qc

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    qc.write_metrics_csv(
        adir / "metrics.csv",
        [
            qc.MetricRow(
                "emb_01", "groom", "state", 1, "min", 0.5, 0.5, 0.5, 1, 0, None, None,
                3, 4, 3, 1, 0, 0.8, 0.75, 1.0, 0.85, 0.1, n_folds=2, ap_std=0.05,
            )
        ],
    )  # fmt: skip
    qc.write_folds_csv(
        adir / "folds.csv",
        [
            qc.FoldMetricRow("groom", 0, 3, 4, 3, 1, 0, 0.8, 0.75, 1.0, 0.85, 0.1),
            qc.FoldMetricRow("groom", 1, 3, 4, 2, 2, 1, 0.7, 0.5, 0.66, 0.6, 0.2),
        ],
    )
    qc.write_predictions_csv(
        adir / "predictions.csv",
        [
            qc.PredictionRow(
                "groom", "s1", 0, 1, "tp", 0.1, 100, 200, 0.8, None, 90, 210
            )
        ],
    )
    qc.write_ground_truth_csv(
        adir / "ground_truth.csv",
        [
            qc.GroundTruthRow("groom", "s1", 0, 100, 200, "scored", "matched"),
            qc.GroundTruthRow("groom", "s1", 0, 500, 600, "exemplar", ""),
        ],
    )
    qc.write_pr_points(adir / "pr" / "groom" / "fold_00.csv", [(0.5, 1.0), (1.0, 0.75)])
    qc.write_pr_points(adir / "pr" / "groom" / "fold_01.csv", [(0.5, 1.0), (1.0, 0.66)])
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "query_eval",
            "recording_ids": ["recording_001"],
            "status": "complete",
            "embed_run_id": "emb_01",
            "track_id": 0,
            "params": {"fusion": "min", "band_s": 0.5, "iou_threshold": 0.5,
                       "n_exemplars": 1, "seed": 0, "folds": 2},
            "behaviors": [
                {"behavior": "groom", "behavior_type": "state", "n_recordings": 1,
                 "n_folds": 2, "skipped": None},
                {"behavior": "rear", "behavior_type": "state", "n_recordings": 0,
                 "n_folds": 0, "skipped": "no eligible recordings"},
            ],
            "artifacts": [
                {"path": "metrics.csv", "kind": "query_eval_metrics", "label": "metrics"}
            ],
        },
    )  # fmt: skip
    page = AnalysisRunDetailPage(proj, "an_01")
    # inline report: the mean±std summary + the per-fold table + the Explore button
    from PyQt6.QtWidgets import QPushButton

    text = " ".join(lbl.text() for lbl in page.findChildren(QLabel))
    assert "±0.050" in text
    fold_model = next(
        t.model()
        for t in page.findChildren(QTableView)
        if t.model().horizontalHeaderItem(0)
        and t.model().horizontalHeaderItem(0).text() == "Fold"
    )
    assert fold_model.rowCount() == 2
    assert "Explore clips" in [b.text() for b in page.findChildren(QPushButton)]

    # the real entry point — the report's own "Explore clips" handler (non-blocking
    # show, not a modal exec) opens the window and keeps it alive on the result widget.
    from detail.gui.modules.analysis_query_eval import (
        _QueryEvalExploreDialog,
        _QueryEvalResult,
        _SpanStrip,
    )

    result = page.findChildren(_QueryEvalResult)[0]
    result._open_explore()
    assert isinstance(result._explore, _QueryEvalExploreDialog)

    # the explorer window: the scrubbable GT/TP/FP/FN timeline + a clips table (TP + the
    # held-out query, decode-safe over the fake recording), and selecting a clip plays it.
    from detail.api.run_preview import QueryEvalPreview
    from detail.gui.behavior_query import _ClipPlayer

    by_behavior = {
        m.behavior: m for m in QueryEvalPreview(proj.dir).read_metrics("an_01")
    }
    dialog = _QueryEvalExploreDialog(
        proj, proj.analysis_record("an_01"), by_behavior, "groom"
    )
    assert dialog.findChildren(_SpanStrip)  # timeline built
    assert dialog._masks.isChecked(), "masks open on, as in every player"
    clips = next(
        t.model()
        for t in dialog.findChildren(QTableView)
        if t.model().horizontalHeaderItem(0)
        and t.model().horizontalHeaderItem(0).text() == "Rank"
    )
    outcomes = {clips.item(r, 1).text() for r in range(clips.rowCount())}
    assert {
        "TP",
        "Query",
    } <= outcomes  # TP + the held-out query, both listed + playable
    player = dialog.findChildren(_ClipPlayer)[0]
    dialog._clips_table.selectRow(0)  # select a clip → scrubs the recording + plays it
    assert player._clip is not None

    # The same class `f9350f0a` fixed on the two Browse tabs and this branch fixed on the
    # audit tab, at its fourth site: the clips table and the two pictures share one splitter
    # row, and a splitter hands every child the whole height. The table drew as many rows as
    # the dialog was tall and both pictures were centred halfway down panes whose neighbour
    # starts at the top. Measured through the real dialog, shown and laid out — an unshown
    # widget has no geometry to be wrong.
    from tests.gui.layout_checks import assert_player_level_with_its_list
    from detail.gui.tables import BROWSE_ROWS

    clips_heading = next(
        lbl for lbl in dialog.findChildren(QLabel) if lbl.text() == "Clips"
    )
    dialog.resize(1200, 900)
    dialog.show()
    qapp.processEvents()
    try:
        assert_player_level_with_its_list(
            dialog,
            dialog._clips_table,
            player._src,
            above=clips_heading,
            max_rows=BROWSE_ROWS,
        )
    finally:
        dialog.hide()


def test_a_run_of_a_removed_module_opens_as_a_bare_record(qapp, tmp_path):
    """A clip-annotation run on disk still opens, as its record, now that the module is gone.

    This test used to build the run's Review/Summary workspace while the run was still
    ``annotating``. The module was removed (``docs/removed_capabilities.md``); the runs it
    left in experiments were not, and ``annotating`` is a status no registered module writes
    any more. The page has to take a module id it cannot resolve and a status nothing else
    uses, and show the record rather than raise.
    """
    from PyQt6.QtWidgets import QLabel, QTabWidget

    from detail.executor.runners import ANALYSIS_MODULES
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar

    assert "clip_annotation" not in ANALYSIS_MODULES
    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "annotation.yaml").write_text(
        "schema_version: 1\niteration: 1\n", encoding="utf-8"
    )
    (adir / "labels.csv").write_text("label_key,decision\n", encoding="utf-8")
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "clip_annotation",
            "recording_ids": ["recording_001"],
            "status": "annotating",
            "artifacts": [
                {"path": "labels.csv", "kind": "annotation_labels", "label": "labels"}
            ],
        },
    )

    assert [(r.id, r.module_id) for r in proj.analysis_runs] == [
        ("an_01", "clip_annotation")
    ]
    assert proj.authored_run("an_01") is None
    page = AnalysisRunDetailPage(proj, "an_01")
    assert not page.findChildren(
        QTabWidget
    ), "a workspace was built for a removed module"
    shown = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "annotating" in shown, "the record's status is not on the page"


@pytest.mark.parametrize(
    "module_id, run_id",
    [("detection", "det_01"), ("tracking", "track_01"), ("embedding", "emb_01")],
)
def test_recording_run_detail_builds_result(
    qapp, completed_runs_experiment, module_id, run_id
):
    # A per-recording stage run (no experiment record) previews directly via the
    # recording-scoped page — the headless-VM run-viewing gap.
    from detail.gui.pages import RecordingRunDetailPage

    page = RecordingRunDetailPage(
        completed_runs_experiment, "recording_001", module_id, run_id
    )
    assert page.findChildren(_RESULT_VIEW[module_id])  # the per-recording preview built


def test_analysis_run_detail_shows_output_for_partial(qapp, tmp_path):
    # A "partial" run (some recordings failed) wrote CSVs for the ones that finished —
    # its detail page must show those artifacts + Open output folder, not only Delete.
    from PyQt6.QtWidgets import QPushButton, QTableView

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001", "recording_002"],
            "tracking_run_id": "track_01",
            "status": "partial",
            "per_recording": {
                "recording_001": {"status": "complete"},
                "recording_002": {"status": "failed", "error": "boom"},
            },
            "artifacts": [
                {"path": "recording_001/tracks.csv", "kind": "tracks", "label": "x"}
            ],
        },
    )
    page = AnalysisRunDetailPage(proj, "an_01")
    assert page.findChildren(
        QTableView
    )  # the surviving recording's artifacts are shown
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Open output folder" in labels  # not only Delete


def test_recording_run_detail_shows_failure_reason(qapp, tmp_path):
    # A failed per-recording stage run carries its reason in the run dir's job.json,
    # not the recording.yaml record — the detail page reads it from there (M4).
    import json

    import detail.pipeline.detection  # noqa: F401 — registers the module
    from PyQt6.QtWidgets import QLabel

    from detail.executor.runners import PROCESSING_MODULES
    from detail.gui.pages import RecordingRunDetailPage

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("recording_001")
    data = recording_store.read_recording(recording.dir)
    data["detection_runs"] = [{"id": "det_01", "status": "failed"}]
    recording_store.write_recording(recording.dir, data)
    run_dir = recording.dir / PROCESSING_MODULES["detection"].output_path("det_01")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(
        json.dumps(
            {"schema_version": 1, "status": "failed", "error": "RuntimeError: boom"}
        ),
        encoding="utf-8",
    )

    page = RecordingRunDetailPage(proj, "recording_001", "detection", "det_01")
    labels = [w.text() for w in page.findChildren(QLabel)]
    assert any("RuntimeError: boom" in t for t in labels)


def test_recording_run_detail_missing_run(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import RecordingRunDetailPage

    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    page = RecordingRunDetailPage(proj, "recording_001", "detection", "det_99")
    labels = [w.text() for w in page.findChildren(QLabel)]
    assert any("run not found" in t for t in labels)


# --- retry button (§15) ----------------------------------------------------- #
class _SyncStageProgressWorker(StageProgressWorker):
    """Run inline so done/failed fire deterministically in the test thread (real
    QObject signals; only the OS thread is skipped). Mirrors tests/gui/test_launch.py.
    """

    def start(self) -> None:  # type: ignore[override]
        self.run()


def _retry_record(proj, run_id="run_009", status="failed", per_recording=None):
    per = per_recording or {
        "s1": {"run_id": "det_01", "status": "complete"},
        "s2": {"run_id": None, "status": "failed", "error": "HfHubHTTPError: 504"},
        "s3": {"run_id": None, "status": "failed", "error": "HfHubHTTPError: 504"},
    }
    runs_store.write_run(
        proj.dir,
        run_id,
        {
            "id": run_id,
            "module_id": "detection",
            "recording_ids": list(per),
            "config": {},
            "per_recording": per,
            "status": status,
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    return run_id


def _buttons(page):
    from PyQt6.QtWidgets import QPushButton

    return [b.text() for b in page.findChildren(QPushButton)]


def test_retry_button_counts_incomplete_recordings(qapp, tmp_path):
    proj = Experiment.create("p", tmp_path)
    run_id = _retry_record(proj)
    page = RunDetailPage(proj, run_id)
    assert "Retry 2 incomplete recordings" in _buttons(page)


def test_retry_button_is_singular_for_one_recording(qapp, tmp_path):
    proj = Experiment.create("p", tmp_path)
    run_id = _retry_record(
        proj,
        per_recording={
            "s1": {"run_id": "det_01", "status": "complete"},
            "s2": {"run_id": None, "status": "failed"},
        },
    )
    page = RunDetailPage(proj, run_id)
    assert "Retry 1 incomplete recording" in _buttons(page)


def test_retry_button_offered_on_a_cancelled_run(qapp, tmp_path):
    """A cancelled batch is the other case where you pick up where it stopped — the
    gate is the incomplete count, not the run-wide status."""
    proj = Experiment.create("p", tmp_path)
    run_id = _retry_record(proj, status="cancelled")
    assert "Retry 2 incomplete recordings" in _buttons(RunDetailPage(proj, run_id))


def test_no_retry_button_when_every_recording_completed(
    qapp, completed_runs_experiment
):
    for run_id in ("run_001", "run_002", "run_003", "run_004"):
        page = RunDetailPage(completed_runs_experiment, run_id)
        assert not [b for b in _buttons(page) if b.startswith("Retry")]


def test_retry_runs_off_thread_and_goes_live(qapp, tmp_path, monkeypatch):
    """Retry validates every incomplete recording + spawns — O(recordings) disk work that
    must not run on the UI thread (unlike _resume, which calls its API inline)."""
    from detail.gui import pages as gui_pages
    from PyQt6.QtWidgets import QPushButton

    spawned, captured = [], {}

    class _RecordingCall(_SyncStageProgressWorker):
        def start(self):
            # Snapshot before the work runs: if retry_run were called inline (the
            # _resume shape) it would already be recorded here. Asserting merely that
            # *a* worker started would not distinguish the two.
            spawned.append(dict(captured))
            super().start()

    class _Handle:
        def progress(self):
            return iter(())

    def _fake_retry(self, rid, progress=None):
        captured["rid"] = rid
        progress("Validating", 0.0, "s2 (1/2)")
        captured["status"] = page._status.text()
        return _Handle()

    monkeypatch.setattr(gui_pages, "StageProgressWorker", _RecordingCall)
    monkeypatch.setattr(Experiment, "retry_run", _fake_retry)
    proj = Experiment.create("p", tmp_path)
    run_id = _retry_record(proj)
    page = RunDetailPage(proj, run_id)
    button = next(
        b for b in page.findChildren(QPushButton) if b.text().startswith("Retry")
    )
    button.click()  # the real button → slot wiring, not the slot directly

    assert spawned == [{}]  # retry_run had NOT run when the worker started...
    assert captured["rid"] == run_id  # ...the worker is what ran it
    # Each report replaces the status line: the recording it is on, not a static line.
    assert captured["status"] == "Validating — s2 (1/2)"
    # isVisibleTo, not isVisible: an unshown top-level makes every child report
    # hidden regardless of its own flag.
    assert page._progress.isVisibleTo(page)  # _start_live ran: the page is live again
    waits.finished(page._worker, "the run-detail load")


def test_retry_failure_surfaces_and_re_enables(qapp, tmp_path, monkeypatch):
    from detail.gui import pages as gui_pages
    from PyQt6.QtWidgets import QPushButton

    shown = []
    monkeypatch.setattr(gui_pages, "StageProgressWorker", _SyncStageProgressWorker)
    monkeypatch.setattr(
        gui_pages.QMessageBox, "critical", lambda *a, **k: shown.append(a[2])
    )

    def _boom(self, rid, progress=None):
        raise ValueError("cannot retry run_009 — s2: no video parts")

    monkeypatch.setattr(Experiment, "retry_run", _boom)
    proj = Experiment.create("p", tmp_path)
    run_id = _retry_record(proj)
    page = RunDetailPage(proj, run_id)
    button = next(
        b for b in page.findChildren(QPushButton) if b.text().startswith("Retry")
    )
    button.click()

    assert shown == ["cannot retry run_009 — s2: no video parts"]  # verbatim
    assert button.isEnabled()  # re-enabled so the user can fix and retry
    assert not page._progress.isVisibleTo(page)  # did not go live


def test_query_eval_result_shows_each_score_beside_its_null(qapp, tmp_path):
    """The report's reason for existing: a number with nothing to compare it against
    can't be read. The per-overlap table renders actual / null / lift."""
    from PyQt6.QtWidgets import QLabel, QTableView

    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage import query_eval_csv as qc

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    qc.write_metrics_csv(
        adir / "metrics.csv",
        [
            qc.MetricRow(
                "emb_01", "climb", "state", 1, "min", 0.5, 0.5, 0.5, 1, 0, None, None,
                3, 4, 3, 1, 0, 0.8, 0.75, 1.0, 0.85, 0.1, method="dtw",
                recall_null=0.15, recall_lift=0.85, ap_null=0.12, ap_lift=0.68,
            )
        ],
    )  # fmt: skip
    qc.write_folds_csv(adir / "folds.csv", [])
    qc.write_overlap_csv(
        adir / "overlap.csv",
        [
            qc.OverlapRow("climb", "state", 0.1, 3, 4, 3, 0.9, 0.2, 0.7, 0.75, 0.63, 0.16, 0.47),
            qc.OverlapRow("climb", "state", 0.25, 3, 4, 3, 0.81, 0.15, 0.66, 0.7, 0.52, 0.12, 0.40),
            qc.OverlapRow("climb", "state", 0.5, 3, 4, 2, 0.51, 0.11, 0.40, 0.5, 0.21, 0.05, 0.16),
        ],
    )  # fmt: skip
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "query_eval",
            "status": "complete",
            "recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "params": {"method": "dtw", "band_s": 0.5},
            "behaviors": [
                {"behavior": "climb", "behavior_type": "state", "n_recordings": 1,
                 "n_folds": 1, "skipped": None}
            ],
        },
    )  # fmt: skip
    page = AnalysisRunDetailPage(proj, "an_01")
    overlap_model = next(
        t.model()
        for t in page.findChildren(QTableView)
        if t.model().horizontalHeaderItem(0)
        and t.model().horizontalHeaderItem(0).text() == "Overlap"
    )
    assert overlap_model.rowCount() == 3
    strict = [overlap_model.item(2, c).text() for c in range(8)]
    assert strict[0] == "0.50"
    assert strict[5] == "0.210" and strict[6] == "0.050"  # AP beside its null
    # The matcher is named in the header, so a report can't be misattributed.
    labels = [w.text() for w in page.findChildren(QLabel)]
    assert any("dtw" in t for t in labels)


def test_query_eval_result_survives_a_run_recorded_before_nulls(qapp, tmp_path):
    """An older run has no overlap.csv; the report must still open."""
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage import query_eval_csv as qc

    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    qc.write_metrics_csv(
        adir / "metrics.csv",
        [
            qc.MetricRow(
                "emb_01", "groom", "state", 1, "min", 0.5, 0.5, 0.5, 1, 0, None, None,
                3, 4, 3, 1, 0, 0.8, 0.75, 1.0, 0.85, 0.1,
            )
        ],
    )  # fmt: skip
    qc.write_folds_csv(adir / "folds.csv", [])
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01", "module_id": "query_eval", "status": "complete",
            "recording_ids": ["recording_001"], "embed_run_id": "emb_01", "track_id": 0,
            "params": {},
            "behaviors": [
                {"behavior": "groom", "behavior_type": "state", "n_recordings": 1,
                 "n_folds": 1, "skipped": None}
            ],
        },
    )  # fmt: skip
    assert AnalysisRunDetailPage(proj, "an_01") is not None  # no overlap.csv, no crash


def test_run_detail_states_the_runs_cadence_and_its_lineage(
    qapp, completed_runs_experiment
):
    """A finished run has to say what it looked at. The rate alone is meaningless without
    the source, and the lineage carries each ancestor's cadence so where the thinning
    happened is visible without opening three pages."""
    from PyQt6.QtWidgets import QLabel

    proj = completed_runs_experiment
    run = next(r for r in proj.processing_runs if r["module_id"] == "tracking")
    page = RunDetailPage(proj, run["id"])
    texts = [label.text() for label in page.findChildren(QLabel)]

    assert any(t.startswith("every frame") for t in texts), texts
    assert any("anchor_tracker" in t and "from" in t for t in texts), texts


# --- reattach: navigating away and back must not kill the bar (§16) ---------- #
def _running_analysis_dir(proj, run_id="an_01", module_id="location_export"):
    """An analysis run left mid-flight, the way a live worker leaves one."""
    import json
    import os

    from detail.storage import analysis_runs as ar

    adir = ar.analysis_dir(proj.dir, run_id)
    adir.mkdir(parents=True, exist_ok=True)
    ar.write_analysis_yaml(
        adir,
        {
            "id": run_id,
            "module_id": module_id,
            "recording_ids": ["recording_001"],
            "status": "running",
            "created_at": "2026-08-01T10:00:00Z",
        },
    )
    (adir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
                "pid": os.getpid(),  # alive, so nothing reconciles it out from under us
            }
        ),
        encoding="utf-8",
    )
    return adir


def test_reopening_a_running_analysis_shows_its_progress_bar(
    qapp, tmp_path, monkeypatch
):
    """The reported bug: start an analysis, leave the page, come back — the bar was frozen
    and never completed, because the reopen passed no live handle at all (the launch path
    was the only one that did) and nothing re-read the record.

    Asserts the bar is *visible*, not merely that the page constructed: it is hidden
    outright when live is None, which is exactly what the bug looked like.
    """
    from detail.gui.app import MainWindow
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    _running_analysis_dir(proj)

    window = MainWindow()
    window._experiment = proj
    window._open_analysis_run("an_01")
    page = window._content.widget(0)

    assert page._progress.isVisibleTo(page)  # a live bar, not a hidden one
    assert page._worker is not None  # ...actually streaming the run


def test_reopening_a_finished_analysis_shows_no_bar(qapp, tmp_path, monkeypatch):
    """The other half: a terminal run must still render as a static record."""
    from detail.gui.app import MainWindow
    from detail.storage import analysis_runs as ar
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    adir = _running_analysis_dir(proj)
    ar.set_analysis_status(adir, "complete")

    window = MainWindow()
    window._experiment = proj
    window._open_analysis_run("an_01")
    page = window._content.widget(0)

    assert not page._progress.isVisibleTo(page)


def test_a_live_launched_analysis_page_is_complete(qapp, tmp_path, monkeypatch):
    """A run's page must be whole the moment it is launched, not after a round trip.

    The bug was `_on_analysis_launched` building the page without one of its callbacks, so a
    button appeared only if you navigated away and came back. The callback in question was
    `on_refine`, which no longer exists — the Start-from picker replaced that button — so the
    assertion moves to the one still passed the same way. The shape of the mistake is the
    point: a page built at two call sites, complete at one of them.
    """
    from detail.gui.app import MainWindow
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    _running_analysis_dir(proj)

    window = MainWindow()
    window._experiment = proj
    handle = proj.attach_analysis_run("an_01")
    window._on_analysis_launched(handle)
    page = window._content.widget(0)

    assert page._on_new is not None


# --- 4c: a run page shows which phase it is in, and what is still ahead (§16, issue #393) --- #
def test_an_analysis_page_shows_a_row_per_declared_phase(qapp, tmp_path, monkeypatch):
    """The feature itself: the page draws one row per phase the module declared, not just the
    aggregate bar — the same complaint #372 fixed for a processing run's recordings."""
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    _running_analysis_dir(proj, module_id="query")

    page = AnalysisRunDetailPage(proj, "an_01")

    assert page._panel.units.isVisibleTo(page)
    assert len(page._panel._units_drawn()) == 4, page._panel._units_drawn()


def test_a_live_message_moves_the_running_phase_row(qapp, tmp_path, monkeypatch):
    """The structured event has to reach the rows, the way it does for a processing run's
    recordings — a test that only checks the aggregate bar moves passes with the rows
    unconnected to anything."""
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.gui.run_units import DONE, QUEUED, RUNNING
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    _running_analysis_dir(proj, module_id="query")
    page = AnalysisRunDetailPage(proj, "an_01")

    class _Event:
        message = "searching 3/4 units"
        frames_done = 700
        frames_total = 1000

    page._on_progress(_Event())

    drawn = page._panel._units_drawn()
    assert [u.state for u in drawn] == [DONE, DONE, RUNNING, QUEUED]
    running = [u for u in drawn if u.state == RUNNING]
    assert len(running) == 1, drawn
    assert running[0].name == "searching"
    assert running[0].fraction == 0.7


def test_the_phase_rows_disappear_once_the_run_is_over(qapp, tmp_path, monkeypatch):
    """Two surfaces, one claim (the rule `RunDetailPage._refresh_units` states for a finished
    processing run): the phase rows are a *live* progress display, not a record of what a
    finished run did — "Results" above already says what it produced."""
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    adir = _running_analysis_dir(proj, module_id="query")
    ar.set_analysis_status(adir, "complete")

    page = AnalysisRunDetailPage(proj, "an_01")

    assert page._panel._units_drawn() == []


# --- 4b: the other fit pages reattach too ----------------------------------- #
def _running_job(proj, kind, run_id):
    """A dataset / model run left mid-flight, record + job envelope."""
    import json
    import os

    from detail.storage import datasets as datasets_store
    from detail.storage import models as models_store

    writers = {
        "dataset": (datasets_store.dataset_dir, datasets_store.write_dataset_yaml),
        "model": (models_store.model_dir, models_store.write_model_yaml),
    }
    run_dir_for, write_record = writers[kind]
    run_dir = run_dir_for(proj.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_record(run_dir, {"id": run_id, "name": run_id, "status": "running"})
    (run_dir / "job.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "job": {"job_id": run_id, "config": {}},
                "progress": None,
                "error": None,
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


@pytest.mark.parametrize(
    "kind,run_id,opener",
    [
        ("dataset", "dataset_001", "_open_dataset"),
        ("model", "model_001", "_open_model"),
    ],
)
def test_reopening_a_running_fit_page_shows_its_progress_bar(
    qapp, tmp_path, monkeypatch, kind, run_id, opener
):
    """All five detail pages inherit _FitDetailPage, which hides its bar when live is None.
    Only the launch paths passed a handle, so every one of these reopened dead."""
    from detail.gui.app import MainWindow
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = Experiment.create("p", tmp_path)
    _running_job(proj, kind, run_id)

    window = MainWindow()
    window._experiment = proj
    getattr(window, opener)(run_id)
    page = window._content.widget(0)

    assert page._progress.isVisibleTo(page)  # a live bar, not a hidden one
    assert page._worker is not None


def test_distill_stage_rows_track_the_run_not_the_launch(qapp, tmp_path):
    """The stage rows are built once and (in the base) rebuilt only on terminal status, so
    a 90-minute distill showed "Label: pending · dataset —" for its whole duration."""
    from PyQt6.QtWidgets import QLabel

    from detail.gui.pages import DistillRunDetail
    from detail.storage import distill_runs as distill_store

    proj = Experiment.create("p", tmp_path)
    distill_store.write_run(
        proj.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "d",
            "status": "running",
            "stages": {"label": {"status": "pending"}, "train": {"status": "pending"}},
            "config": {},
            "created_at": "2026-08-01T10:00:00Z",
        },
    )

    class _Handle:
        id = "distill_001"
        status = "running"
        error = None

        def progress(self):
            return iter(())

    def _rows(page):
        """The text of the rows currently rendered (the page's own body, not every child
        widget: _build deleteLater()s the old rows, which linger until Qt collects them).
        """
        return [
            label.text()
            for widget in page._body_widgets
            for label in widget.findChildren(QLabel)
        ]

    page = DistillRunDetail(proj, "distill_001", live=_Handle())
    assert any("pending" in t for t in _rows(page))

    # the coordinator finishes labeling and starts training, rewriting the record
    distill_store.write_run(
        proj.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "d",
            "status": "running",
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "running", "model_id": "model_001"},
            },
            "config": {},
            "created_at": "2026-08-01T10:00:00Z",
        },
    )
    page._on_tick(500, "training (2/2) — epoch 1/10")

    texts = _rows(page)
    assert any("dataset_001" in t for t in texts)  # the rows followed the run
    assert any("model_001" in t for t in texts)
    assert not any("pending" in t for t in texts)  # ...and dropped the launch snapshot


def _completed_analysis(tmp_path):
    """An experiment with one finished analysis run — the shape the detail page renders."""
    from detail.storage import analysis_runs as ar

    proj = Experiment.create("p", tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "tracking_run_id": "track_01",
            "status": "complete",
            "artifacts": [],
        },
    )
    return proj


def test_an_analysis_run_offers_starting_another(qapp, tmp_path):
    """Finishing a run is where you decide what to do next, and it is usually another run.

    Distinct from Refine, which seeds *this* run's config: a different module is the common
    case, so this opens the launch dialog empty. It needs no §25 gate for that reason — the
    dialog only offers modules this build surfaces — where Refine does, because it pre-loads
    one specific module by id.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import AnalysisRunDetailPage

    proj = _completed_analysis(tmp_path)
    started = []
    page = AnalysisRunDetailPage(proj, "an_01", on_new=lambda: started.append(True))

    buttons = [
        b for b in page.findChildren(QPushButton) if b.text().startswith("Start new")
    ]
    assert buttons, "no way to start another analysis from a finished one"
    buttons[0].click()
    assert started == [True]


def test_the_new_analysis_button_is_absent_without_a_handler(qapp, tmp_path):
    """Guard the guard: the button is gated on the callback, so a page built without one
    must not show a dead control — the failure `clip_clicked` already shipped once."""
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import AnalysisRunDetailPage

    page = AnalysisRunDetailPage(_completed_analysis(tmp_path), "an_01")
    assert not [
        b for b in page.findChildren(QPushButton) if b.text().startswith("Start new")
    ]


def test_a_query_run_made_with_a_removed_method_opens_as_a_bare_record(qapp, tmp_path):
    """A Query run whose record names `dtw`, which this version no longer has.

    The run lists, its page opens and says why there is no workspace, its behavior output
    still exports, and it can still be deleted; calibrating it is refused by name. The workspace resolves the
    method to read the stored search, so building it on such a run would raise.
    """
    from PyQt6.QtWidgets import QLabel

    from detail.core.registries import QUERY_METHODS
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import query_runs

    from tests.api.test_query import _searched

    exp, run = _searched(tmp_path, n_rows=300)
    run.set_cutoff("q01", 0.5)
    state = run.state()
    state.method = "dtw"
    query_runs.write_query_state(run._run_dir(), state)
    assert "dtw" not in QUERY_METHODS

    assert [(r.id, r.module_id) for r in exp.analysis_runs] == [("an_01", "query")]
    page = AnalysisRunDetailPage(exp, "an_01")
    shown = " ".join(label.text() for label in page.findChildren(QLabel))
    assert "dtw" in shown and "no longer" in shown, shown

    with pytest.raises(ValueError, match="dtw"):
        exp.query_run("an_01").set_cutoff("q01", 0.6)
    # The behavior output it already wrote is a file, not a search, and still exports.
    out = exp.query_run("an_01").export(tmp_path / "export")
    assert sorted(p.name for p in out.iterdir()) == ["s1.csv", "s2.csv"]
    exp.delete_analysis("an_01")
    assert exp.analysis_runs == []


def test_a_pose_clustering_run_opens_as_a_bare_record(qapp, tmp_path):
    """A pose-clustering run on disk, now that the module is removed (it is expected back).

    The record lists, `authored_run` answers None (its accessor went with the module, so the
    import error it would raise is never reached), the page opens with its status and no
    workspace, and the run deletes.
    """
    from PyQt6.QtWidgets import QLabel, QTabWidget

    from detail.executor.runners import ANALYSIS_MODULES
    from detail.gui.pages import AnalysisRunDetailPage
    from detail.storage import analysis_runs as ar

    assert "pose_clustering" not in ANALYSIS_MODULES
    proj = Experiment.create("p", tmp_path)
    adir = ar.analysis_dir(proj.dir, "an_01")
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "pose_clustering.yaml").write_text(
        "schema_version: 1\nmethod: two_step\ngranularity: 4\nnames: {}\n",
        encoding="utf-8",
    )
    ar.write_analysis_yaml(
        adir,
        {
            "id": "an_01",
            "module_id": "pose_clustering",
            "recording_ids": ["recording_001"],
            "status": "open",
        },
    )

    assert [(r.id, r.module_id) for r in proj.analysis_runs] == [
        ("an_01", "pose_clustering")
    ]
    assert proj.authored_run("an_01") is None
    page = AnalysisRunDetailPage(proj, "an_01")
    assert not page.findChildren(
        QTabWidget
    ), "a workspace was built for a removed module"
    assert "open" in " ".join(label.text() for label in page.findChildren(QLabel))
    proj.delete_analysis("an_01")
    assert proj.analysis_runs == []
