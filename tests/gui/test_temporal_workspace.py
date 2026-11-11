"""The interpretation workspace over a **temporal** run — what differs from the pose one.

`tests/gui/test_cluster_workspace.py` covers the workspace's behaviour generically, over a pose
fit; nothing there is re-run here. What this file covers is the part that is not generic:

- **The subsample line.** §22 requires a capped fit to say so, because it is identical to a full
  one on every other part of this screen — same table, same scatter, same confident-looking
  clusters. If it is not said here it is not said anywhere.
- **The nouns.** A behavior is not a posture — and it is not a *syllable*, which is MoSeq's
  word for MoSeq's claim. The workspace takes its wording from ``WorkspaceTerms`` rather than
  hardcoding one module's vocabulary.
- **The way in.** A temporal run reached the way a user reaches it, through the run page.

A separate file rather than a parametrization of the pose one: the two fixtures are different
kinds of fit — a codebook over sampled rows, and a DTW matrix over bouts — and one fixture
pretending to be both would obscure exactly the difference these tests are about.
"""

import numpy as np
import pytest

from detail.gui.cluster_workspace import TEMPORAL_TERMS, ClusterWorkspace

from tests.pipeline.test_temporal_clustering_module import (
    STEP,
    _bouty_trajectory,
    _method_params,
)


def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    """Spin the event loop until ``predicate`` holds; every load here is off-thread."""
    from tests.gui.conftest import drain_until

    return drain_until(qapp, predicate, timeout_ms)


def _fitted(experiment_dir, recording_dir, make_embedding_run, **overrides):
    """One real temporal fit — segmentation and a genuine DTW pass, through the module."""
    from detail.api.experiment import Experiment
    from detail.storage import recording as recording_store

    features = _bouty_trajectory()
    frame_idx = np.arange(len(features), dtype=np.int64)
    t_ns = frame_idx * STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01")

    experiment = Experiment(experiment_dir)
    run = experiment.run_analysis(
        "temporal_clustering",
        {
            "source_recording_ids": ["recording_001"],
            "embed_run_id": "emb_01",
            "track_id": 0,
            "method": "dtw_finch",
            "method_params": _method_params(**overrides),
        },
    )
    run.wait()
    return experiment, run.id


@pytest.fixture
def temporal_run(experiment_dir, recording_dir, make_embedding_run):
    return _fitted(experiment_dir, recording_dir, make_embedding_run)


def _workspace(qapp, experiment, run_id):
    handle = experiment.temporal_clustering_run(run_id)
    workspace = ClusterWorkspace(experiment, handle, TEMPORAL_TERMS)
    interpret = workspace._interpret
    assert _drain(qapp, lambda: interpret._view is not None), "the cache never loaded"
    return handle, workspace


# --- the scale wall, on screen -------------------------------------------------------------


def test_a_full_fit_shows_no_subsample_line(qapp, temporal_run):
    """The negative case. A line that always appeared would be noise, and the guard below would
    prove nothing."""
    experiment, run_id = temporal_run
    _handle, workspace = _workspace(qapp, experiment, run_id)
    # `isHidden`, not `isVisible`: an unshown widget is never "visible" regardless of what it
    # was told, so an `isVisible() is False` assertion here would pass on a line that could never
    # appear — and the positive case below could never pass at all.
    assert workspace._interpret._subsample.isHidden()
    assert workspace._interpret._subsample.text() == ""


def test_a_capped_fit_says_so_on_the_screen(
    qapp, experiment_dir, recording_dir, make_embedding_run
):
    """§22: a fit over a fraction of the bouts is identical to a full one in every other respect.

    This is the whole reason ``subsample_note`` exists — without it a user reads a vocabulary
    built on four bouts as if it were built on all of them, and nothing on the screen disagrees.
    """
    experiment, run_id = _fitted(
        experiment_dir, recording_dir, make_embedding_run, max_segments=4
    )
    _handle, workspace = _workspace(qapp, experiment, run_id)
    line = workspace._interpret._subsample

    assert not line.isHidden(), "a capped fit must say what it left out"
    assert "4" in line.text()
    assert "bout" in line.text()
    assert line.isEnabled(), "a disabled label cannot be selected for copying"


# --- the nouns -------------------------------------------------------------------------------


def test_the_screen_calls_them_behaviors_not_syllables(qapp, temporal_run):
    """A behavior is not a posture, and it is not a syllable either.

    The workspace is generic over the run, so the wording is a parameter. "Syllable" is MoSeq's
    word and carries MoSeq's claim, which this tool does not make. Every visible word is read,
    after a node is selected so the card is filled, because a walk over a half-built screen is a
    check whose reach depends on scheduling (#554).
    """
    from PyQt6.QtWidgets import QLabel

    experiment, run_id = temporal_run
    _handle, workspace = _workspace(qapp, experiment, run_id)
    interpret = workspace._interpret
    view = interpret._view
    interpret.select(view.children(view.root)[0])
    text = " ".join(label.text() for label in interpret.findChildren(QLabel))
    text += " " + interpret._label_field.edit.placeholderText()
    assert "behavior" in text
    assert "syllable" not in text.lower()
    assert "posture" not in text
    assert "Cluster" not in text and "cluster" not in text
    assert interpret.label_bar().name.text().startswith("Behavior #")


def test_no_sentence_says_the_same_noun_twice(qapp, temporal_run):
    """``unit`` and ``subject`` now hold the same word, and copy built from both went circular.

    The hint under the table was ``one {subject} usually has more than one {unit}`` — true and
    readable while a unit was a "syllable", and "one behavior usually has more than one behavior"
    the moment the unit took the subject's word. Nothing failed: the label rendered, every other
    assertion passed, and the sentence was simply meaningless. It was found by looking at a
    screenshot, which is the only thing that could have found it.

    So: a label may name the noun, and may not need it twice to make its point.
    """
    experiment, run_id = temporal_run
    _handle, workspace = _workspace(qapp, experiment, run_id)

    from PyQt6.QtWidgets import QLabel

    for label in workspace._interpret.findChildren(QLabel):
        text = label.text()
        assert text.lower().count("behavior") <= 1, (
            "this sentence names the same thing twice, which reads as a tautology once "
            f"WorkspaceTerms.unit and .subject hold one word — reword it against rows: {text!r}"
        )


# --- it loads, and it loads a real vocabulary ------------------------------------------------


def test_the_workspace_opens_on_a_vocabulary(qapp, temporal_run):
    """Assert the content, not that it built: a tree of more than one behavior, root selected."""
    experiment, run_id = temporal_run
    _handle, workspace = _workspace(qapp, experiment, run_id)
    interpret = workspace._interpret
    view = interpret._view
    assert view.tree.n_leaves > 1
    assert interpret._selected == view.root
    first = view.children(view.root)[0]
    interpret.select(first)
    assert interpret.label_bar().name.text().startswith("Behavior #")
    assert view.summary(first).time_ns > 0


# --- the way a user actually reaches it ------------------------------------------------------


def test_the_run_page_opens_the_workspace_while_the_run_is_open(qapp, temporal_run):
    """Through the real entry point. A temporal fit stays **open** by design (§21), so a page
    that only rendered a result view at ``complete`` would show the interpretation surface after
    the interpretation."""
    from detail.gui.pages import AnalysisRunDetailPage

    experiment, run_id = temporal_run
    assert experiment.temporal_clustering_run(run_id).status == "open"

    page = AnalysisRunDetailPage(experiment, run_id)
    workspaces = page.findChildren(ClusterWorkspace)
    assert workspaces, "the run's page does not open the interpretation workspace"
    # AUDIT arrived by this module emitting the §19 artifact, not by being built here —
    # `gui/audit_tab.py` reads only the artifact and the run. DIFFERENCES is still absent
    # because §24 has no GUI at all, and shipping it empty would claim more than is here.
    assert workspaces[0].view_names() == [
        "Interpret",
        "Audit",
        "Browse",
        "Measures",
    ]
    assert _drain(
        qapp, lambda: workspaces[0]._interpret._view is not None
    ), "the workspace opened but never loaded the node cache"


# --- the launch form, with two methods to choose between ------------------------------------


def test_the_form_offers_only_the_selected_methods_knobs(qapp):
    """The two temporal methods have **disjoint** knob sets: one segments and compares bouts, the
    other builds a codebook and decodes. A single flat list would show someone the DTW band while
    they configure a method that has none — and would send it as a parameter the method's config
    model has never heard of, which `validate` rejects.
    """
    import detail.gui.modules  # noqa: F401  (registers the module UIs)
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES

    widget = ANALYSIS_GUI_MODULES["temporal_clustering"].config_widget()

    widget._method.setCurrentText("dtw_finch")
    bout = set(widget.config()["method_params"])
    widget._method.setCurrentText("codebook_viterbi")
    codebook = set(widget.config()["method_params"])

    assert "band_s" in bout and "detector" in bout
    assert "band_s" not in codebook and "detector" not in codebook
    assert (
        "fit_share" in codebook and "codewords" in codebook and "min_run_s" in codebook
    )
    assert "fit_share" not in bout and "codewords" not in bout
    # ...and each set is exactly what its own model declares, so a knob cannot be sent to a method
    # that does not have it.
    from detail.configs.temporal_codebook import TemporalCodebookConfig
    from detail.configs.temporal_dtw_finch import TemporalDtwFinchConfig

    assert bout <= set(TemporalDtwFinchConfig.model_fields)
    assert codebook <= set(TemporalCodebookConfig.model_fields)


def test_both_methods_configs_validate(
    qapp, experiment_dir, recording_dir, make_embedding_run
):
    """Surface parity, end to end: what the form produces must be something the module accepts.

    A form that emitted a plausible-looking dict the module then refused would fail only when
    somebody pressed Run.
    """
    import numpy as np

    import detail.gui.modules  # noqa: F401
    from detail.gui.modules.registry import ANALYSIS_GUI_MODULES
    from detail.pipeline.temporal_clustering import TemporalClusteringModule
    from detail.storage import recording as recording_store

    features = _bouty_trajectory()
    frame_idx = np.arange(len(features), dtype=np.int64)
    t_ns = frame_idx * STEP
    recording_store.write_timestamps_csv(
        recording_dir / "video/timestamps/generated/part_001.csv", t_ns
    )
    make_embedding_run(recording_dir, features, frame_idx, t_ns, embed_run_id="emb_01")

    widget = ANALYSIS_GUI_MODULES["temporal_clustering"].config_widget()
    for method_id in ("dtw_finch", "codebook_viterbi"):
        widget._method.setCurrentText(method_id)
        config = dict(widget.config())
        config["source_recording_ids"] = ["recording_001"]
        config["embed_run_id"] = "emb_01"
        resolved = TemporalClusteringModule().validate(experiment_dir, config)
        assert resolved["method"] == method_id


# --- the share column, and the time column that had to exist for it to be honest ------------


# --- where it happens ------------------------------------------------------------------------


def test_a_temporal_runs_clusters_play_every_clip_ranked_off_the_ui_thread(
    qapp, temporal_run
):
    """A temporal run once listed **no** clips for any node. Every node lists its clips now, and
    a bout method's are ranked on demand by medoid — on a worker, so the montage fills when the
    ranking lands, in the order the run gives (``run.clips``)."""
    experiment, run_id = temporal_run
    handle, workspace = _workspace(qapp, experiment, run_id)
    interpret = workspace._interpret
    view = interpret._view
    assert view.ranks_on_demand, "a bout method's typicality is ranked on demand"
    node = view.children(view.root)[0]
    interpret.select(node)
    expected = handle.clips(node).order("typical", seed=node).tolist()
    assert expected, "the node lists clips"
    assert _drain(qapp, lambda: interpret.montage_order() == expected)
    assert len(interpret.montage()._clips) == len(expected)
