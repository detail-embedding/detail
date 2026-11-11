"""The training curve, and what makes a score actionable rather than merely present.

**The gap this closes.** The model page showed one number — the final mAP50 — and a bar counting
epochs. Neither says whether the score is *still climbing*, so a run improving for an hour and a
run that plateaued forty epochs ago rendered identically. "Stop training if the metrics look
good" is undecidable from that, which is why the page grew a series beside its summary.

**The class underneath it, which is older than this feature.** `final_metrics` answers *how good
is this* and was treated as if it also answered *is it still getting better*. One number cannot,
and nothing fails when it is asked to: the page renders, the value is real, and the reader draws
a conclusion the data does not support. A summary and a series are two questions, and a surface
that needs the second is not served by the first.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")


RISING = [{"epoch": e, "box_map50": 0.30 + 0.01 * e} for e in range(1, 40)]
PEAKED = [
    {"epoch": e, "box_map50": 0.80 - 0.004 * abs(e - 31)} for e in range(1, 39)
]  # peaks at epoch 31, then drifts down — what overfitting looks like


# --- the number and the sentence --------------------------------------------- #
def test_the_best_epoch_is_the_peak_not_the_last():
    """The property that makes the caption true of the weights on disk.

    `best.pt` is saved from the peak, so naming the tail would report a score the saved model
    does not have. A monotone fixture cannot tell the two apart — every rising history has its
    maximum at the end — so this runs on one that turns over.
    """
    from detail.gui.training_curve import best_epoch

    assert best_epoch(PEAKED) == (
        31,
        pytest.approx(0.80),
    ), "the best epoch is not the peak, so the caption names a score best.pt does not carry"
    assert best_epoch(RISING)[0] == 39, "a still-rising run's best is its latest"


def test_the_caption_names_the_epoch_and_how_far_back_it_is():
    """What makes 0.847 actionable is not the number.

    A score alone cannot be compared with anything. The decision is "it peaked at 31 and we are
    at 38", so both ends are in the sentence — and the gap is stated as a **count of epochs**
    rather than as "plateaued", which would be the tool making the call the user is here to make.
    """
    from detail.gui.training_curve import best_line

    line = best_line(PEAKED, total_epochs=60)

    assert "0.800" in line, line
    assert "epoch 31" in line, line
    assert "7 epochs" in line, f"the gap since the peak is not stated: {line}"
    assert "60" in line, f"the run's length is not stated: {line}"
    assert "plateau" not in line.lower(), f"the tool is making the call: {line}"


def test_a_still_improving_run_is_not_told_it_has_stalled():
    """The counterexample for the clause above — it must not print on every run."""
    from detail.gui.training_curve import best_line

    line = best_line(RISING, total_epochs=60)

    assert (
        "no improvement" not in line
    ), f"a run that improved on its latest epoch is reported as stalled: {line}"


def test_one_epoch_behind_is_singular():
    from detail.gui.training_curve import best_line

    line = best_line(
        [{"epoch": 1, "box_map50": 0.5}, {"epoch": 2, "box_map50": 0.4}],
        total_epochs=10,
    )
    assert "1 epoch " in line or line.endswith("1 epoch"), line
    assert "1 epochs" not in line, line


def test_no_completed_epoch_says_so_rather_than_showing_a_zero():
    """An empty history is a claim about the data, not a score of 0.0."""
    from detail.gui.training_curve import best_line

    assert "no completed epoch" in best_line([])


# --- the plot ----------------------------------------------------------------- #
def _curve(qapp, history, total_epochs=60):
    from detail.gui.training_curve import TrainingCurve

    widget = TrainingCurve()
    widget.resize(700, 300)
    widget.show()
    qapp.processEvents()
    widget.set_history(history, total_epochs=total_epochs)
    qapp.processEvents()
    return widget


def test_the_axis_spans_the_run_not_the_epochs_drawn_so_far(qapp):
    """Otherwise the curve rescales under the reader on every tick and a rising line looks flat.

    This is the live case and the one a static fixture never reaches: at epoch 2 of 60 the plot
    is mostly empty, deliberately, so that the shape at epoch 40 is the shape at epoch 2 with
    more of it drawn.
    """
    widget = _curve(
        qapp, [{"epoch": 1, "box_map50": 0.36}, {"epoch": 2, "box_map50": 0.41}]
    )
    low, high = widget._plot.getViewBox().viewRange()[0]

    assert high >= 60, (
        f"x-axis stops at {high:.0f} with 60 epochs to run — the curve will rescale under the "
        "reader at every tick"
    )


def test_the_score_axis_states_its_domain(qapp):
    """mAP is a fraction, so the view may not be scrolled to coordinates it cannot take."""
    widget = _curve(qapp, PEAKED)
    limits = widget._plot.getViewBox().state["limits"]["yLimits"]

    assert (
        limits[0] is not None and limits[1] is not None
    ), f"the mAP axis is unbounded: {limits}"
    assert limits[0] <= 0.0 and 1.0 <= limits[1] <= 1.1, limits


def test_redrawing_the_same_history_does_not_stack_curves(qapp):
    """Idempotence, because this is redrawn on every tick of a live run.

    Its sibling failure is a caption *appended* to rather than set, which prints twice on a
    second delivery and which no test that draws once can see.
    """
    widget = _curve(qapp, PEAKED)
    first = len(widget._plot.listDataItems())
    caption = widget._caption.text()

    widget.set_history(PEAKED, total_epochs=60)
    qapp.processEvents()

    assert (
        len(widget._plot.listDataItems()) == first
    ), "a second draw stacked another curve"
    assert widget._caption.text() == caption, "the caption was appended to, not set"


def test_the_plot_does_not_take_the_page(qapp):
    """The regression that made the feature counterproductive.

    A `PlotWidget`'s sizeHint is a constant 600x480 and its policy is `Expanding`, so it asks for
    480px and then takes the page's slack on top: measured at 455px on a 900px window, which put
    *Stop and keep* — the button the curve exists to inform — below the fold. You read the curve,
    decide to stop, and cannot see the control.
    """
    from detail.gui.sizing import lines
    from detail.gui.training_curve import PLOT_LINES

    widget = _curve(qapp, PEAKED)
    hint = widget._plot.sizeHint().height()

    assert hint == lines(PLOT_LINES), (
        f"the plot asks for {hint}px rather than its {lines(PLOT_LINES)}px — the constant 480 "
        "is back"
    )
    assert (
        widget._plot.minimumHeight() > 0
    ), "no floor, so a squeeze crops the x-axis away"


# --- through the real entry point --------------------------------------------- #
# Constructing `TrainingCurve` proves it builds, not that anyone reaches it. These drive the page
# a user actually opens, because "a feature that renders in a private harness but sits behind an
# unrelated surface is not done" — and this one is reached by a method that landed, at first, in
# the wrong class entirely (there is more than one `_stop` in pages.py) and still passed the
# construction smoke, because the smoke fixture's model has no history to draw.


def _model_with_history(tmp_path, epochs, *, status="running"):
    from detail.api.experiment import Experiment
    from detail.storage import models as models_store

    exp = Experiment.create("p", tmp_path)
    mdir = models_store.model_dir(exp.dir, "model_001")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "mouse-detector",
            "status": status,
            "trainer_id": "ultralytics_yolo",
            "dataset_id": "dataset_001",
            "hyperparams": {"epochs": 60},
        },
    )
    if epochs:
        run = mdir / "run"
        run.mkdir(parents=True, exist_ok=True)
        rows = ["epoch,metrics/mAP50(B),metrics/mAP50(M)"]
        rows += [
            f"{e},{0.80 - 0.004 * abs(e - 31):.4f},{0.75 - 0.004 * abs(e - 31):.4f}"
            for e in range(1, epochs + 1)
        ]
        (run / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return exp


def test_the_model_page_shows_the_curve_while_training(qapp, tmp_path, monkeypatch):
    from detail.gui.pages import ModelDetailPage
    from detail.gui.training_curve import TrainingCurveBand
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 38)

    page = ModelDetailPage(exp, "model_001")
    bands = page.findChildren(TrainingCurveBand)

    assert bands, "a training run in progress shows no curve on its own page"
    caption = bands[0].curve._caption.text()
    assert "epoch 31" in caption, f"the peak is not named on the page: {caption!r}"


def test_the_model_page_shows_no_curve_when_there_is_no_history(
    qapp, tmp_path, monkeypatch
):
    """An imported model has no epochs at all rather than zero of them, and a plot of nothing is
    the "renders the frame of the thing it does not have" failure."""
    from detail.gui.pages import ModelDetailPage
    from detail.gui.training_curve import TrainingCurveBand
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 0, status="complete")

    page = ModelDetailPage(exp, "model_001")

    assert not page.findChildren(
        TrainingCurveBand
    ), "an empty axis is drawn for a model that never trained here"


def test_the_page_puts_the_stop_button_where_it_can_be_seen(
    qapp, tmp_path, monkeypatch
):
    """The curve informs a decision whose control is on the same page, so the two must fit on it.

    Measured failure: a `PlotWidget` asks for its constant 480px and then takes the page's slack,
    which put *Stop and keep* 455px below the fold on a 900px window. Assert the rendered
    geometry, not that both widgets exist — they always did.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.app import MainWindow
    from detail.gui.pages import ModelDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 38)

    window = MainWindow()
    page = ModelDetailPage(exp, "model_001")
    window._show(page, "Model")
    window.resize(1000, 800)
    window.show()
    for _ in range(20):
        qapp.processEvents()

    stop = [b for b in page.findChildren(QPushButton) if b.text() == "Stop and keep"]
    assert stop, "no Stop and keep button on a running model"
    bottom = stop[0].mapTo(window, stop[0].rect().bottomLeft()).y()

    assert bottom <= 800, (
        f"Stop and keep sits {bottom}px down an 800px window — you read the curve, decide to "
        "stop, and cannot see the control"
    )


def _write_distill_run(exp, train_status, *, model_id="model_001"):
    from detail.storage import distill_runs as distill_store

    distill_store.write_run(
        exp.dir,
        "distill_001",
        {
            "id": "distill_001",
            "status": "running",
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": train_status, "model_id": model_id},
            },
        },
    )


def _distill_run(tmp_path, train_status, *, model_id="model_001"):
    from detail.api.experiment import Experiment

    exp = Experiment.create("p", tmp_path)
    _write_distill_run(exp, train_status, model_id=model_id)
    return exp


@pytest.mark.parametrize(
    "train_status,reachable",
    [("running", True), ("complete", True), ("pending", False)],
)
def test_the_distill_page_reaches_the_model_while_it_is_training(
    qapp, tmp_path, monkeypatch, train_status, reachable
):
    """A capability with no way in, caught before it shipped.

    The model page is where full model management lives (rename, Promote to global…, weights,
    full lineage) — the curve and *Stop and keep* are inline on the distill page too now (see
    `test_the_distill_page_shows_the_curve_while_training` below), but the link to the rest of
    the model page was gated on the training being **complete** — so during training the only
    button on screen was *Cancel distill*, which kills the run and loses the model. The
    capability existed and was unreachable from the surface that showed it was needed.

    Parametrised over all three states rather than asserting the new one, because the fix is a
    *pair*: the link must appear while running and must still not appear before training starts,
    where there are no epochs to look at. Either half alone passes with the condition wrong the
    other way.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_run(tmp_path, train_status)

    page = DistillRunDetail(exp, "distill_001", on_open_model=lambda mid: None)
    labels = [b.text() for b in page.findChildren(QPushButton)]
    opens = [text for text in labels if text.startswith("Open tra")]

    if reachable:
        assert opens, (
            f"train status {train_status!r} offers no way to the model page — where the curve "
            f"and Stop and keep are. Buttons: {labels}"
        )
    else:
        assert (
            not opens
        ), f"train status {train_status!r} links to a model page with nothing on it: {labels}"


# --- the curve and Stop and keep, inline on the distill page itself ----------- #
# The above kept the *link* reachable while training; these are what the link used to be the
# only way to see — watching a distill run no longer means leaving this page to find out
# whether training is stuck (issue #392's follow-up).


def test_the_distill_page_shows_the_curve_while_training(qapp, tmp_path, monkeypatch):
    from detail.gui.pages import DistillRunDetail
    from detail.gui.training_curve import TrainingCurveBand
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 38)
    _write_distill_run(exp, "running")

    page = DistillRunDetail(exp, "distill_001")

    bands = page.findChildren(TrainingCurveBand)
    assert bands, "the distill page shows no curve for a model that is training"
    caption = bands[0].curve._caption.text()
    assert "epoch 31" in caption, f"the peak is not named on the page: {caption!r}"


def test_the_distill_page_offers_stop_and_keep_while_training(
    qapp, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 38, status="running")
    _write_distill_run(exp, "running")

    page = DistillRunDetail(exp, "distill_001")

    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Stop and keep" in labels, (
        "no way to stop a training and keep its weights without leaving the distill page: "
        f"{labels}"
    )


def test_the_distill_page_hides_stop_and_keep_once_training_is_done(
    qapp, tmp_path, monkeypatch
):
    """The curve stays (how the score got where it did); the control to stop it does not, since
    a finished model has nothing left to stop."""
    from PyQt6.QtWidgets import QPushButton

    from detail.gui.pages import DistillRunDetail
    from detail.gui.training_curve import TrainingCurveBand
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 38, status="complete")
    _write_distill_run(exp, "complete")

    page = DistillRunDetail(exp, "distill_001")

    assert page.findChildren(TrainingCurveBand), "a completed run lost its own curve"
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert (
        "Stop and keep" not in labels
    ), f"nothing left to stop, but offered anyway: {labels}"


def test_stopping_training_from_the_distill_page_calls_stop_model(
    qapp, tmp_path, monkeypatch
):
    """Wired to the real `Experiment.stop_model` — the same call `ModelDetailPage._stop` makes,
    not a reimplementation that could drift from it."""
    from PyQt6.QtWidgets import QPushButton

    from detail.api import experiment as experiment_module
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 38, status="running")
    _write_distill_run(exp, "running")
    page = DistillRunDetail(exp, "distill_001")

    stopped = []
    monkeypatch.setattr(
        experiment_module.Experiment,
        "stop_model",
        lambda self, mid: stopped.append(mid),
    )
    stop = next(
        b for b in page.findChildren(QPushButton) if b.text() == "Stop and keep"
    )
    stop.click()

    assert stopped == ["model_001"]
    assert "Stopping" in page._panel.status.text()


def test_a_distill_stage_naming_a_missing_model_shows_no_curve(
    qapp, tmp_path, monkeypatch
):
    """A stage names its model the moment it starts, before this page has any occasion to read
    it — a dangling reference (deleted model, or a race) is nothing to show, not a crash.
    """
    from detail.gui.pages import DistillRunDetail
    from detail.gui.training_curve import TrainingCurveBand
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _distill_run(tmp_path, "running", model_id="model_gone")

    page = DistillRunDetail(exp, "distill_001")  # must not raise

    assert not page.findChildren(TrainingCurveBand)


@pytest.mark.parametrize(
    "stale, model_id",
    [("pending", None), ("cancelled", "model_001"), ("failed", "model_001")],
    ids=["launch", "resume-after-cancel", "resume-after-failure"],
)
def test_stop_and_keep_appears_when_the_stage_write_trails_the_phase(
    qapp, tmp_path, monkeypatch, stale, model_id
):
    """The coordinator's real order: it announces ``training``, *then* launches the job, *then*
    writes ``train: running`` with the model id. The page rebuilds on the phase change and — before
    it learned to re-read a lagging record — trusted that snapshot for the whole stage, so Stop and
    keep, the curve and the link to the model never appeared while training ran.

    The snapshot is not always ``pending``. On **Resume** the stage still carries the status the
    last attempt left — ``cancelled`` or ``failed`` — and the first fix, which recognised only
    ``pending``, drew the resumed training as a failed row under a moving bar (seen in the app).
    Every test above writes the record before the page opens; this one writes it after.
    """
    from PyQt6.QtWidgets import QPushButton

    from detail.core.dataclasses import DistillProgress
    from detail.gui.pages import DistillRunDetail
    from detail.gui.workers import fit_progress_fraction
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 0)
    _write_distill_run(exp, stale, model_id=model_id)
    page = DistillRunDetail(exp, "distill_001", on_open_model=lambda mid: None)
    page._LAG_REREAD_S = 0.0  # these ticks arrive microseconds apart, not a second

    def deliver(inner, message):  # what FitProgressWorker.run does per event
        event = DistillProgress(
            "training", 1, 2, inner, 0.5 + inner / 2, message=message
        )
        fraction, status = fit_progress_fraction(event)
        page._on_tick(round(fraction * 1000), status)
        page._on_progress(event)

    deliver(0.0, "starting training")  # rebuilds on a record that still says `stale`
    _write_distill_run(exp, "running")  # _set_stage("train", "running", model_id=...)
    deliver(0.1, "epoch 6/60")

    shown = [b.text() for b in page._body_widgets if isinstance(b, QPushButton)]
    assert "Stop and keep" in shown, shown
    assert any(text.startswith("Open training") for text in shown), shown
    train_row = page._panel._units_drawn()[1]
    assert train_row.state == "running", train_row


# --- the curve moves while training does ------------------------------------ #
# Drawn once when the page was built and not again until the run ended, the curve answered
# "is it still improving" with whatever was true when the page opened — and a distill page,
# built as training began, had no curve at all. These deliver epochs to an open page.


def _write_epochs(exp, n):
    from detail.storage import models as models_store

    run = models_store.model_dir(exp.dir, "model_001") / "run"
    run.mkdir(parents=True, exist_ok=True)
    rows = ["epoch,metrics/mAP50(B),metrics/mAP50(M)"]
    rows += [f"{e},{0.10 + e / 100:.3f},0.40" for e in range(1, n + 1)]
    (run / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _drawn_band(page):
    """The curve band on the page now — not one a rebuild has scheduled for deletion."""
    from detail.gui.training_curve import TrainingCurveBand

    bands = [w for w in page._body_widgets if isinstance(w, TrainingCurveBand)]
    assert len(bands) <= 1, bands
    return bands[0] if bands else None


def test_the_model_page_redraws_its_curve_as_epochs_finish(qapp, tmp_path, monkeypatch):
    from detail.core.dataclasses import TrainingProgressEvent
    from detail.gui.pages import ModelDetailPage
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 5)
    page = ModelDetailPage(exp, "model_001")
    band = _drawn_band(page)
    assert len(band.curve._history) == 5

    _write_epochs(exp, 25)
    page._on_progress(TrainingProgressEvent(25, 60, {}, 900.0, None))

    assert _drawn_band(page) is band, "redrawn by rebuilding the page, not in place"
    assert [row["epoch"] for row in band.curve._history][-1] == 25


def test_the_distill_page_grows_a_curve_once_training_has_an_epoch(
    qapp, tmp_path, monkeypatch
):
    from detail.core.dataclasses import DistillProgress
    from detail.gui.pages import DistillRunDetail
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    exp = _model_with_history(tmp_path, 0)
    _write_distill_run(exp, "running")
    page = DistillRunDetail(exp, "distill_001")
    assert _drawn_band(page) is None  # no epoch yet: nothing to plot

    def epoch_done(n):
        _write_epochs(exp, n)
        page._on_progress(
            DistillProgress("training", 1, 2, n / 60, 0.5 + n / 120, message="")
        )

    epoch_done(1)
    band = _drawn_band(page)
    assert band is not None, "the first finished epoch did not put a curve on the page"
    assert len(band.curve._history) == 1

    epoch_done(25)
    assert _drawn_band(page) is band, "redrawn by rebuilding the page, not in place"
    assert len(band.curve._history) == 25
