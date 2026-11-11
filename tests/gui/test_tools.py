"""Tools GUI (§9): landing, run page, overlay-clip result player, sidebar navigation.

One recorded class is guarded here, because this is where quick_segment's sampled grid can be
built: *a pane handed more height than its hint with nowhere to put it* (`gui_design` §3). A
`setWidgetResizable(True)` scroll area grows its inner widget to the viewport, and a grid with no
stretched row shares that growth among its rows — the header laid out at 241px against a 17px
hint. `test_the_sampled_grid_packs_its_rows_at_the_top` checks it; the check itself, and why a
grid label beside a thumbnail is not spread, is `tests/gui/layout_checks.py`.
"""

import numpy as np
import pytest

from tests import waits

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QPushButton  # noqa: E402

from detail.api import tools as api_tools  # noqa: E402
from detail.core.registries import register_model  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.gui.tools.registry import TOOL_GUIS, import_tool_uis  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.tools.test_quick_segment import (  # noqa: E402
    _StubDetector,
    _VariedConfidenceDetector,
    _config,
    _write_video,
)


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    """A real completed quick_segment run in an isolated tools workspace."""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    register_model("stub_seg")(_StubDetector)  # restored by the isolation fixture
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=8)
    record = api_tools.run_tool("quick_segment", _config(video, n_frames=6))
    return tmp_path, record


@pytest.fixture
def sampled_run(tmp_path, monkeypatch):
    """A real completed *sampled* run — the multi-prompt shape, whose result view is the
    comparison grid rather than the overlay player."""
    from detail.core.dataclasses import Prompt, PromptTarget, TextCue

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    register_model("stub_seg")(_StubDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=24)
    prompts = [
        Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        Prompt(targets=[PromptTarget("mouse with tail", TextCue("mouse with tail"))]),
    ]
    record = api_tools.run_tool(
        "quick_segment",
        _config(video, n_frames=4, mode="sample", prompts=prompts),
    )
    return tmp_path, record


def test_sampled_run_result_shows_the_comparison_and_the_frame_view(qapp, sampled_run):
    # A sampled run has no detection zarr, so the result view must be the prompt grid —
    # picking the player would raise on the missing output.
    from detail.gui.tools.quick_segment import _QuickSegmentComparison

    _root, record = sampled_run
    import_tool_uis()
    widget = TOOL_GUIS["quick_segment"].result_widget(record)
    assert isinstance(widget, _QuickSegmentComparison)
    tabs = [widget._tabs.tabText(i) for i in range(widget._tabs.count())]
    assert tabs == ["Compare", "Play"]  # two prompts, so comparison leads
    assert widget._matrix.cell_count() == 8  # 4 frames × 2 prompts
    # The frame view steps the sampled frames and renders the saved overlays.
    assert len(widget._frames_view._frames) == 4
    assert widget._frames_view._texts == ["mouse", "mouse with tail"]


def test_single_prompt_sampled_run_opens_on_the_frame_view(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    register_model("stub_seg")(_StubDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=12)
    record = api_tools.run_tool(
        "quick_segment", _config(video, n_frames=3, mode="sample", classes=("mouse",))
    )
    import_tool_uis()
    widget = TOOL_GUIS["quick_segment"].result_widget(record)
    # Nothing to compare against, so the grid tab is not offered.
    assert [widget._tabs.tabText(i) for i in range(widget._tabs.count())] == ["Play"]


def test_a_row_is_one_prompt_and_a_comma_names_classes_within_it(qapp, tmp_path):
    # Two levels that read alike in prose: rows are prompts to compare, commas are
    # classes inside one prompt. Pinning it so the wording and the behavior can't drift.
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    import_tool_uis()
    cfg = TOOL_GUIS["quick_segment"].config_widget()
    cfg._videos.addItem(str(clip))
    cfg._prompts.clear()
    cfg._prompts.addItem("mouse,cricket")  # one prompt, two classes
    cfg._prompts.addItem("mouse with tail")  # a second prompt to compare

    prompts = cfg.config()["prompts"]
    assert len(prompts) == 2
    assert [t.class_name for t in prompts[0].targets] == ["mouse", "cricket"]
    assert [t.class_name for t in prompts[1].targets] == ["mouse with tail"]
    # Spaces around the comma are the natural way to type it and must not survive.
    cfg._prompts.clear()
    cfg._prompts.addItem("mouse, cricket ")
    assert [t.class_name for t in cfg.config()["prompts"][0].targets] == [
        "mouse",
        "cricket",
    ]


def test_tools_landing_lists_and_opens_a_tool(qapp):
    opened = []
    page = pages.ToolsPage(opened.append)
    button = next(b for b in page.findChildren(QPushButton) if b.text() == "Open")
    button.click()
    assert opened == ["quick_segment"]


def test_run_page_backend_combo_is_restricted_to_the_model(
    qapp, monkeypatch, real_machine
):
    # A cuda-only model (SAM3 is one) must offer only cuda in the combo, even on a Mac —
    # the #79 pattern. Self-contained: register a cuda-only text-cue model rather than rely
    # on sam3 being in MODELS (registry contents vary by which tests were collected).
    from detail.core.dataclasses import FrameResult, TextCue
    from detail.core.protocols import DetectionModel

    class _CudaOnlySeg(DetectionModel):
        model_id = "cuda_only_seg"
        supported_cue_types = [TextCue]
        supported_backends = ["cuda"]

        def load(self, device):
            pass

        def predict(self, frame, prompt):
            return FrameResult.from_detections(0, 0, [])

        def predict_batch(self, frames, prompts):
            return [self.predict(f, p) for f, p in zip(frames, prompts)]

    register_model("cuda_only_seg")(_CudaOnlySeg)  # restored by the isolation fixture
    monkeypatch.setattr("detail.api.models.is_installed", lambda *a, **k: True)
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    import_tool_uis()
    cfg = TOOL_GUIS["quick_segment"].config_widget()
    # A cuda-only model on a Mac: the row is offered but greyed with its reason, and the
    # backend combo never lists a device this machine does not have. It used to list
    # "cuda" alone — which Qt then made stick, since setCurrentText past it is a no-op.
    row = next(
        cfg._model.itemText(i)
        for i in range(cfg._model.count())
        if (cfg._model.itemData(i) or "") == "cuda_only_seg"
    )
    assert row.endswith("needs CUDA")
    assert "cuda" not in [
        cfg._compute.backend.itemText(i) for i in range(cfg._compute.backend.count())
    ]


def test_combo_lists_your_models_and_closed_set_hides_prompt(
    qapp, tmp_path, monkeypatch
):
    # The tool now runs any model: SAM 3 + your distilled/imported ones. A closed-set model
    # has no text prompt — the field hides and no class is required.
    from detail.api import models as api_models
    from detail.core.registries import TRAINED_LOADERS
    from tests.models.test_imported import _StubLoader

    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubLoader)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    mid = api_models.import_model(weights, family="stubfam", name="my_yolo")

    import_tool_uis()
    cfg = TOOL_GUIS["quick_segment"].config_widget()
    items = [cfg._model.itemText(i) for i in range(cfg._model.count())]
    assert "sam3" in items and mid in items  # SAM 3 + your model

    cfg._prompts.clear()  # no prompts
    cfg._videos.addItem(str(weights))  # any existing file (only existence is checked)
    cfg._model.setCurrentText(mid)  # a closed-set model
    assert not cfg._prompts.isVisibleTo(cfg)  # the prompt row hides
    # ...including its Add…/Remove buttons, which live in a sibling holder: hiding only
    # the list left them stretched across the form with nothing to add a prompt to.
    prompt_buttons = [
        b
        for b in cfg._prompt_row.findChildren(QPushButton)
        if b.text() in ("Add", "Remove")
    ]
    assert len(prompt_buttons) == 2
    assert not any(b.isVisibleTo(cfg) for b in prompt_buttons)
    assert cfg.validation_error() is None  # and no prompt is required
    assert cfg.config()["prompt"].targets == []  # an empty prompt is sent

    cfg._model.setCurrentText("sam3")  # prompt-capable again
    assert cfg._prompts.isVisibleTo(cfg)
    assert all(b.isVisibleTo(cfg) for b in prompt_buttons)
    assert "prompt" in (cfg.validation_error() or "")  # now a prompt is required


def test_config_validation_and_unbounded_run_confirmation(qapp):
    import_tool_uis()
    cfg = TOOL_GUIS["quick_segment"].config_widget()
    assert "Add a video" in cfg.validation_error()  # no video yet
    cfg._frame_count = 500  # a probed long clip

    # A bounded run needs no confirmation however long the clip is: sampling covers it.
    assert cfg.over_cap_message() is None
    cfg._mode_consecutive.setChecked(True)
    cfg._frames.setValue(100)
    assert cfg.over_cap_message() is None

    # The unbounded one does - it is the run that can take hours.
    cfg._whole_video.setChecked(True)
    assert "500" in cfg.over_cap_message()
    assert "stop it while it runs" in cfg.over_cap_message()


def test_consecutive_mode_rejects_several_videos_or_prompts(qapp, tmp_path):
    import_tool_uis()
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    other = tmp_path / "b.mp4"
    other.write_bytes(b"x")
    cfg = TOOL_GUIS["quick_segment"].config_widget()
    cfg._videos.addItem(str(clip))
    cfg._videos.addItem(str(other))
    cfg._mode_consecutive.setChecked(True)
    assert "one clip straight through" in cfg.validation_error()

    cfg._videos.takeItem(1)
    cfg._prompts.addItem("mouse with tail")
    assert "single prompt" in cfg.validation_error()
    # Sampling is what comparison is for, so it accepts both.
    cfg._mode_sample.setChecked(True)
    cfg._videos.addItem(str(other))
    assert cfg.validation_error() is None
    assert cfg.config()["mode"] == "sample"
    assert len(cfg.config()["prompts"]) == 2 and len(cfg.config()["videos"]) == 2


def test_a_frame_count_read_for_a_removed_video_is_not_shown(
    qapp, tmp_path, monkeypatch
):
    """A video added and removed before its frames are counted: the count read for it lands
    after the list emptied, and used to be shown as the frames available — an estimate for a
    video no longer listed. A count is shown only for the list it was read for."""
    import threading

    from detail.tools import quick_segment as engine

    import_tool_uis()
    release = threading.Event()

    def slow_count(_path):
        assert release.wait(10), "the test never released the count"
        return 1000

    monkeypatch.setattr(engine, "approx_frame_count", slow_count)
    clip = tmp_path / "a.mp4"
    clip.write_bytes(b"x")
    cfg = TOOL_GUIS["quick_segment"].config_widget()
    cfg._videos.addItem(str(clip))
    cfg._probe_videos()
    worker = cfg._probe_worker
    delivered = []
    worker.done.connect(delivered.append)

    cfg._videos.setCurrentRow(0)
    cfg._remove_video()
    assert cfg._video_paths() == []
    release.set()
    waits.until(qapp, lambda: delivered == [1000], "the stale count's delivery")
    assert cfg._frame_count is None, "a count for a removed video is shown"
    assert cfg._cap.text() == ""


def test_result_widget_builds_and_reports_frame_count(qapp, complete_run):
    _root, record = complete_run
    import_tool_uis()
    widget = TOOL_GUIS["quick_segment"].result_widget(record)
    assert widget._n == record["result"]["n_processed"]
    assert widget._columns["frame_idx"].shape[0] > 0  # detections loaded for overlay


def test_run_detail_page_hosts_result_then_deletes(qapp, complete_run, monkeypatch):
    from detail.gui.tools.quick_segment import _QuickSegmentResult

    _root, record = complete_run
    deleted = []
    page = pages.ToolRunDetailPage(
        "quick_segment", record["id"], on_deleted=lambda: deleted.append(True)
    )
    assert page.findChild(_QuickSegmentResult) is not None
    monkeypatch.setattr(
        "detail.gui.pages.QMessageBox.question",
        lambda *a, **k: pages.QMessageBox.StandardButton.Yes,
    )
    page._delete()
    assert deleted == [True] and api_tools.list_tool_runs() == []


def test_quick_tools_mode_navigates_to_tool_and_run(qapp, complete_run):
    # Quick tools' landing lists the tools, a tool button drills into its run page, and a
    # saved run opens its result — the real callback chain.
    from detail.gui.app import MainWindow

    _root, record = complete_run
    window = MainWindow()
    window._switch_mode("tools")
    assert isinstance(window._content.currentWidget(), pages.ToolsPage)

    window._go_to_tool("quick_segment")  # a tool button on the landing → its run page
    page = window._content.currentWidget()
    assert isinstance(page, pages.ToolRunPage)
    assert record["id"] in page._run_ids  # the saved run is listed below the form

    window._open_tool_run("quick_segment", record["id"])  # open a saved run's result
    assert isinstance(window._content.currentWidget(), pages.ToolRunDetailPage)


def test_run_page_lists_saved_runs(qapp, complete_run):
    from PyQt6.QtWidgets import QTableView

    _root, record = complete_run
    import_tool_uis()
    page = pages.ToolRunPage(
        "quick_segment", on_launched=lambda r: None, on_open_run=lambda r: None
    )
    assert page._run_ids == [record["id"]]  # the saved run is listed below the form
    table = page.findChild(QTableView)  # the navigable saved-runs table
    assert table is not None and table.model().rowCount() == 1


# --------------------------------------------------------------------------- #
# The confidence slider on a result view
# --------------------------------------------------------------------------- #
@pytest.fixture
def varied_consecutive_run(tmp_path, monkeypatch):
    """A consecutive run whose stored proposals straddle a threshold, so a slider has
    something to reveal and hide."""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    register_model("stub_varied")(_VariedConfidenceDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=6)
    cfg = _config(video, n_frames=6, classes=("mouse",))
    cfg["model_id"] = "stub_varied"
    return tmp_path, api_tools.run_tool("quick_segment", cfg)


@pytest.fixture
def varied_sampled_run(tmp_path, monkeypatch):
    from detail.core.dataclasses import Prompt, PromptTarget, TextCue

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    register_model("stub_varied")(_VariedConfidenceDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=24)
    prompts = [
        Prompt(targets=[PromptTarget("mouse", TextCue("mouse"))]),
        Prompt(targets=[PromptTarget("rat", TextCue("rat"))]),
    ]
    cfg = _config(video, n_frames=3, mode="sample", prompts=prompts)
    cfg["model_id"] = "stub_varied"
    return tmp_path, api_tools.run_tool("quick_segment", cfg)


def test_the_playback_view_recounts_and_redraws_as_the_threshold_moves(
    qapp, varied_consecutive_run
):
    _root, record = varied_consecutive_run
    import_tool_uis()
    widget = TOOL_GUIS["quick_segment"].result_widget(record)

    # 6 frames × 2 proposals stored; the caption counts what the threshold keeps.
    assert widget._total_stored == 12
    widget._confidence.set_value(0.0)
    widget._on_min_conf(0.0)
    assert widget._confidence._caption.text() == "showing 12 of 12"
    widget._on_min_conf(0.5)
    assert widget._confidence._caption.text() == "showing 6 of 12"
    widget._on_min_conf(0.95)
    assert widget._confidence._caption.text() == "showing 0 of 12"

    # The decoder draws at the live threshold, not the one the view opened on.
    strict, strict_reason = widget._decode(0)
    widget._on_min_conf(0.0)
    everything, everything_reason = widget._decode(0)
    assert strict_reason is None and everything_reason is None
    assert not (strict == everything).all()

    # And the view actually *asks* for that redraw. Testing _decode alone passed while
    # the picture on screen never changed, because nothing re-ran it.
    requested = []
    widget._scrubber.set_frame = lambda f, **kw: requested.append(f)
    widget._on_min_conf(0.7)
    assert requested, "a threshold change must re-render the frame on screen"


def test_the_comparison_view_shares_one_threshold_across_both_tabs(
    qapp, varied_sampled_run
):
    """Compare and Play show the same detections; two sliders that disagreed would give
    two answers to the same question."""
    _root, record = varied_sampled_run
    import_tool_uis()
    widget = TOOL_GUIS["quick_segment"].result_widget(record)

    widget._on_min_conf(0.5)
    assert widget._matrix._min_conf == 0.5
    assert widget._frames_view._min_conf == 0.5
    widget._on_min_conf(0.1)
    assert widget._matrix._min_conf == 0.1
    assert widget._frames_view._min_conf == 0.1

    # The Play tab re-renders too, not just records the number.
    requested = []
    widget._frames_view._scrubber.set_frame = lambda f, **kw: requested.append(f)
    widget._on_min_conf(0.6)
    assert requested, "the Play tab must re-render at the new threshold"


def test_the_comparison_caption_counts_every_prompt_at_the_threshold(
    qapp, varied_sampled_run
):
    _root, record = varied_sampled_run
    import_tool_uis()
    widget = TOOL_GUIS["quick_segment"].result_widget(record)

    # 3 sampled frames × 2 prompts × 2 proposals = 12 stored; half clear 0.5.
    widget._on_min_conf(0.0)
    assert widget._confidence._caption.text() == "showing 12 of 12"
    widget._on_min_conf(0.5)
    assert widget._confidence._caption.text() == "showing 6 of 12"


def test_the_grid_headers_follow_the_slider_not_the_run(qapp, varied_sampled_run):
    """The run's own per-prompt totals were counted over every proposal. A header that
    kept them would overstate every column at any threshold above zero.

    Builds the grid on its own rather than through the comparison widget: the sibling
    frame view keeps a decoder thread open, and waiting on the grid's gather while that
    one decodes the same clip aborts the interpreter natively.
    """
    from detail.gui.tools.quick_segment import _SampledMatrix
    from detail.storage import tool_runs
    from detail.tools import quick_segment as engine

    _root, record = varied_sampled_run
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    sources = record.get("source_videos") or [record["source_video"]]
    matrix = _SampledMatrix(run_dir, engine.read_sampled(run_dir), sources, 0.0)

    matrix._gather_worker.wait(10_000)
    for _ in range(50):
        qapp.processEvents()
        if matrix._gathered is not None:
            break
    assert matrix._gathered is not None, "the gather never landed"

    matrix.set_min_conf(0.0)
    assert "6 total" in matrix._headers["prompt_0"].text()  # 3 frames x 2 proposals
    matrix.set_min_conf(0.5)
    assert "3 total" in matrix._headers["prompt_0"].text()
    matrix.set_min_conf(0.95)
    assert "0 total" in matrix._headers["prompt_0"].text()


def test_a_grid_cell_opens_its_frame_at_the_current_threshold(qapp, varied_sampled_run):
    """220px is not enough to judge a mask, so a cell opens the frame behind it full size.

    The cell holds the **recipe**: it re-renders when asked, at whatever threshold the slider
    is on then. Caching the drawn overlay instead would pin the enlarged frame to the moment
    the cell was filled — and would hold decoded masks for the whole grid, which is what the
    gather/draw split exists to avoid.

    Built directly rather than through the comparison widget, for the reason the sibling test
    above gives: its frame view keeps a decoder thread open on the same clip.
    """
    from detail.gui.tools.quick_segment import _SampledMatrix
    from detail.storage import tool_runs
    from detail.tools import quick_segment as engine

    _root, record = varied_sampled_run
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    sources = record.get("source_videos") or [record["source_video"]]
    matrix = _SampledMatrix(run_dir, engine.read_sampled(run_dir), sources, 0.0)

    matrix._gather_worker.wait(10_000)
    for _ in range(50):
        qapp.processEvents()
        if matrix._gathered is not None:
            break
    assert matrix._gathered is not None, "the gather never landed"

    matrix.set_min_conf(0.0)
    pane = matrix._panes[(0, "prompt_0")]
    # The draw is off-thread too, so the cell is wired when that lands, not when set_min_conf
    # returns — a bare `processEvents` spin is green on an idle box and flaky in a loaded run.
    waits.until(qapp, lambda: bool(pane.toolTip()), "the grid draw")
    assert "full size" in pane.toolTip(), "the cell offers no way to see the frame"

    at_zero = pane._render()
    assert at_zero is not None and at_zero.ndim == 3, "the cell renders no frame"
    matrix.set_min_conf(0.95)  # every proposal dropped — a different picture, same cell
    waits.until(
        qapp,
        lambda: not np.array_equal(pane._render(), at_zero),
        "the cell to follow the slider",
    )


# --------------------------------------------------------------------------- #
# The Quick tools rail (§9) — one folder per tool, its saved runs beneath
# --------------------------------------------------------------------------- #
def _rail_roles(tree) -> list:
    """Every (kind, payload) role in the tree, parents before their children."""
    from detail.gui.app import _ROLE

    roles = []
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        roles.append(item.data(0, _ROLE))
        roles += [item.child(j).data(0, _ROLE) for j in range(item.childCount())]
    return roles


def _tool_node(tree, tool_id: str):
    from detail.gui.app import _ROLE

    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        if item.data(0, _ROLE) == ("tool", tool_id):
            return item
    return None


@pytest.fixture
def two_runs(tmp_path, monkeypatch):
    """Two real completed quick_segment runs — enough to pin the rail's ordering."""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    register_model("stub_seg")(_StubDetector)
    video = tmp_path / "clip.mp4"
    _write_video(video, n_frames=8)
    first = api_tools.run_tool("quick_segment", _config(video, n_frames=4))
    second = api_tools.run_tool("quick_segment", _config(video, n_frames=6))
    return tmp_path, first, second


def test_the_quick_tools_rail_is_up_on_the_landing(qapp, two_runs):
    # The rail shows for the whole mode, landing included — not only once a tool is open.
    from detail.gui.app import MainWindow

    _root, first, second = two_runs
    window = MainWindow()
    window._switch_mode("tools")

    assert isinstance(window._content.currentWidget(), pages.ToolsPage)
    assert not window._tree.isHidden()
    assert _rail_roles(window._tree) == [
        ("tool", "quick_segment"),
        ("tool_run", ("quick_segment", first["id"])),  # store order: oldest first
        ("tool_run", ("quick_segment", second["id"])),
    ]
    node = _tool_node(window._tree, "quick_segment")
    assert node.isExpanded()  # a collapsed rail would show nothing with one tool
    assert node.child(0).text(0) == f"{first['id']}  ·  complete"


def test_a_rail_click_opens_the_node_and_toggles_a_folder(qapp, two_runs):
    # The uniform rule (`_on_tree_click`), now exercised on the tool kinds: a leaf
    # navigates, a folder navigates *and* toggles.
    from detail.gui.app import MainWindow

    _root, first, _second = two_runs
    window = MainWindow()
    window._switch_mode("tools")
    node = _tool_node(window._tree, "quick_segment")

    window._on_tree_click(node.child(0), 0)  # a run leaf
    assert isinstance(window._content.currentWidget(), pages.ToolRunDetailPage)

    window._on_tree_click(node, 0)  # the tool folder: its page, and it collapses
    assert isinstance(window._content.currentWidget(), pages.ToolRunPage)
    assert not node.isExpanded()
    window._on_tree_click(node, 0)  # and back open, still showing its page
    assert isinstance(window._content.currentWidget(), pages.ToolRunPage)
    assert node.isExpanded()
    assert first["id"] in window._content.currentWidget()._run_ids


def test_the_rail_selection_follows_every_way_into_a_page(qapp, two_runs):
    # A drill-in from the landing button or the saved-runs table must move the rail, or it
    # highlights one thing while the content shows another.
    from detail.gui.app import _ROLE
    from detail.gui.app import MainWindow

    _root, first, _second = two_runs
    window = MainWindow()
    window._switch_mode("tools")

    window._go_to_tool("quick_segment")  # the landing's Open button
    assert window._tree.currentItem().data(0, _ROLE) == ("tool", "quick_segment")

    window._open_tool_run("quick_segment", first["id"])  # the saved-runs table
    assert window._tree.currentItem().data(0, _ROLE) == (
        "tool_run",
        ("quick_segment", first["id"]),
    )


def test_launching_and_deleting_a_run_keep_the_rail_current(qapp, two_runs):
    from detail.gui.app import MainWindow

    root, first, second = two_runs
    window = MainWindow()
    window._switch_mode("tools")

    video = root / "clip.mp4"
    third = api_tools.run_tool("quick_segment", _config(video, n_frames=2))
    window._on_tool_run_launched("quick_segment", third["id"])
    assert ("tool_run", ("quick_segment", third["id"])) in _rail_roles(window._tree)
    assert isinstance(window._content.currentWidget(), pages.ToolRunDetailPage)

    api_tools.delete_tool_run(second["id"])
    window._on_tool_run_deleted("quick_segment")
    assert ("tool_run", ("quick_segment", second["id"])) not in _rail_roles(
        window._tree
    )
    # ...and you land on that tool's page, which still has the rail's other runs.
    page = window._content.currentWidget()
    assert isinstance(page, pages.ToolRunPage)
    assert page._run_ids == [first["id"], third["id"]]


def test_a_collapsed_tool_stays_collapsed_across_a_refresh(qapp, two_runs):
    from detail.gui.app import MainWindow

    root, _first, _second = two_runs
    window = MainWindow()
    window._switch_mode("tools")
    _tool_node(window._tree, "quick_segment").setExpanded(False)

    api_tools.run_tool("quick_segment", _config(root / "clip.mp4", n_frames=2))
    window._refresh_tools_rail()
    assert not _tool_node(window._tree, "quick_segment").isExpanded()


def test_a_project_refresh_does_not_overwrite_the_tools_rail(qapp, two_runs):
    """The guard for the class: with two rails sharing one tree, a refresh helper must key
    off *which rail is shown*, not off "the tree is visible". A background import finishing
    while Quick tools is on screen used to be indistinguishable from one finishing in a
    project — and would rebuild the project rail into the tools tree."""
    from detail.api.experiment import Experiment
    from detail.gui.app import MainWindow

    root, _first, _second = two_runs
    experiment = Experiment.create("p", root)
    experiment.create_recording("recording_001")
    window = MainWindow()
    window._ensure_active(experiment)  # a project is active, as after an import
    window._switch_mode("tools")
    before = _rail_roles(window._tree)

    window._refresh_active_experiment()  # what _on_import_item_done calls
    assert _rail_roles(window._tree) == before
    assert window._rail == "tools"


def test_leaving_quick_tools_drops_the_rail(qapp, two_runs):
    from detail.gui.app import MainWindow

    window = MainWindow()
    window._switch_mode("tools")
    assert not window._tree.isHidden()

    window._switch_mode("models")
    assert window._tree.isHidden()
    assert window._rail is None
    assert _rail_roles(window._tree) == []


# --- what a sampled run may be played at ------------------------------------- #
# Three shapes reach `_SampledFrames` and they are not interchangeable: consecutive (stride 1),
# evenly sampled (a constant stride > 1) and randomly sampled (no stride). The rate is derived
# from the positions rather than from the mode the run was launched in, so these test the
# derivation over all three — and over the two shapes that have no sequence at all.


def _entries(frames, part=0):
    return [{"part_index": part, "frame_idx": f} for f in frames]


def test_consecutive_frames_play_at_the_source_rate():
    from detail.gui.tools.quick_segment import _uniform_stride

    assert _uniform_stride(_entries([10, 11, 12, 13])) == 1


def test_evenly_sampled_frames_play_at_the_rate_that_spacing_represents():
    """A run sampled every 30th frame is 30x through the footage, and says so by its stride."""
    from detail.gui.tools.quick_segment import _uniform_stride

    assert _uniform_stride(_entries([0, 30, 60, 90])) == 30


def test_randomly_sampled_frames_have_no_rate():
    """The case the rule exists for: scattered positions, so any speed would be invented."""
    from detail.gui.tools.quick_segment import _uniform_stride

    assert _uniform_stride(_entries([3, 41, 42, 900])) is None


def test_positions_in_different_parts_are_not_a_sequence():
    """Part-local indices restart, so a gap across a part boundary is not a gap in the footage."""
    from detail.gui.tools.quick_segment import _uniform_stride

    entries = _entries([0, 10], part=0) + _entries([20, 30], part=1)
    assert _uniform_stride(entries) is None


def test_one_frame_is_not_a_sequence():
    from detail.gui.tools.quick_segment import _uniform_stride

    assert _uniform_stride(_entries([7])) is None
    assert _uniform_stride([]) is None


def test_a_sampled_view_plays_only_when_its_positions_form_a_sequence(qapp, tmp_path):
    """End to end through the widget, not just the helper.

    The helper being right says nothing about the view reading it: the first version of this
    change wired the derivation into the wrong class entirely and every helper test still passed.
    """
    from detail.gui.tools.quick_segment import _SampledFrames

    even = _SampledFrames(
        tmp_path,
        {"fps": 30.0, "sampled_frames": _entries([0, 30, 60]), "prompts": []},
        [],
        0.0,
    )
    assert even._scrubber.is_playable()
    assert even._scrubber._fps == 1.0  # 30 fps over a 30-frame gap

    scattered = _SampledFrames(
        tmp_path,
        {"fps": 30.0, "sampled_frames": _entries([0, 7, 9000]), "prompts": []},
        [],
        0.0,
    )
    assert not scattered._scrubber.is_playable()
    assert scattered._scrubber.play_button() is None


def test_the_sampled_grid_packs_its_rows_at_the_top(qapp, varied_sampled_run):
    """`gui_design` §3's top-justify rule, in a grid's form: a trailing stretch row.

    A `setWidgetResizable(True)` scroll area grows its inner widget to the viewport, and a grid
    with no stretched row shares that growth out among its rows — the header laid out at 175px
    against a 15px hint, and worse the wider the window. `test_pane_space.py` said this was
    asserted here; it was not, and removing the stretch left every test green.

    Built directly, for the reason the sibling tests above give.
    """
    from detail.gui.tools.quick_segment import _SampledMatrix
    from detail.storage import tool_runs
    from detail.tools import quick_segment as engine
    from tests.gui.layout_checks import assert_top_justified

    _root, record = varied_sampled_run
    run_dir = tool_runs.tool_run_dir("quick_segment", record["id"])
    sources = record.get("source_videos") or [record["source_video"]]
    matrix = _SampledMatrix(run_dir, engine.read_sampled(run_dir), sources, 0.0)
    matrix._gather_worker.wait(10_000)
    for _ in range(50):
        qapp.processEvents()
        if matrix._gathered is not None:
            break

    assert_top_justified(matrix, size=(1400, 1000))
