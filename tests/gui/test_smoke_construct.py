"""Smoke tests: every navigable GUI surface must *construct* without error.

This is the safety net for refactors that rename/remove a method on a shared
widget or base class. A missed call site shows up here — including the two kinds
static review misses:
  * a subclass **hook override** (e.g. ``_build_overview``) that only runs when
    the *leaf* class is constructed, not the base;
  * ``signal.connect(self._method)`` where ``_method`` was renamed — this raises
    at construction time, never at import/lint.

When you add a page, dialog, run-preview, or module UI, add it to a list here.
(Run-preview widgets — DetectionRunPreview / TrackingRunPreview / EmbeddingRunPreview
— are constructed in ``test_run_preview.py``.)
"""

import dataclasses
import threading

import pytest

pytest.importorskip("PyQt6")

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui import sections as gui_sections
from detail.gui import comparison as gui_comparison  # noqa: E402
from detail.gui import example_set_picker as gui_example_set_picker  # noqa: E402
from detail.gui import keypoints_manager as gui_keypoints_manager  # noqa: E402
from detail.gui import pages as gui_pages  # noqa: E402
from detail.gui.modules.registry import (  # noqa: E402
    ANALYSIS_GUI_MODULES,
    GUI_MODULES,
)
from detail.storage import locations  # noqa: E402
from tests.gui.conftest import drain_until  # noqa: E402
from tests.gui.layout_checks import (  # noqa: E402
    assert_one_content_column,
    assert_top_justified,
)


def _noop(*_args, **_kwargs):
    return None


def _drain(qapp, predicate, timeout_ms: int = 10_000) -> bool:
    """Spin the event loop until ``predicate`` holds — see ``conftest.drain_until``."""
    return drain_until(qapp, predicate, timeout_ms)


def _experiment(tmp_path):
    proj = Experiment.create("p", tmp_path)
    proj.create_recording("recording_001")
    return proj


# --- every module's config form builds (data-driven → new modules covered) -- #
@pytest.mark.parametrize("module_id", sorted(GUI_MODULES))
def test_module_config_widget_builds(qapp, module_id):
    widget = GUI_MODULES[module_id].config_widget()
    assert widget is not None  # raises above if the form's construction is broken


@pytest.mark.parametrize("module_id", sorted(ANALYSIS_GUI_MODULES))
def test_analysis_module_config_widget_builds(qapp, module_id):
    widget = ANALYSIS_GUI_MODULES[module_id].config_widget()
    assert widget is not None


def test_tracking_form_rebuilds_for_iou_single_tracker(qapp):
    # The tracking form is schema-driven; selecting a tracker rebuilds its
    # hyperparam widgets from HYPERPARAMS_SCHEMA. Exercise the new tracker's schema
    # (its own float/int/bool knobs) so a change that breaks the rebuild is caught.
    widget = GUI_MODULES["tracking"].config_widget()
    widget._tracker.setCurrentText("iou_single_tracker")
    assert {"iou_weight", "max_missed", "merge_mask_iou", "require_contiguous"} <= set(
        widget._hp_widgets
    )


# --- every experiment / recording page builds ------------------------------------ #
#: How each page is built, so both the construction smoke and the layout check below are
#: driven from one list and a new page inherits the pair.
PAGES = {
    "HomePage": lambda proj: gui_pages.HomePage(_noop),
    # absorbs the former Data tab
    "ExperimentOverviewPage": lambda proj: gui_pages.ExperimentOverviewPage(
        proj, _noop
    ),
    "ExperimentRecordingsPage": lambda proj: gui_pages.ExperimentRecordingsPage(
        proj, _noop
    ),
    "MetadataPage": lambda proj: gui_pages.MetadataPage(proj),
    "ExperimentRunsPage": lambda proj: gui_pages.ExperimentRunsPage(proj, _noop),
    # top-level, machine-global (no experiment)
    "GlobalModelsPage": lambda proj: gui_pages.GlobalModelsPage(),
    "DatasetsPage": lambda proj: gui_pages.DatasetsPage(proj, _noop),
    # per-experiment "Trained models"
    "ModelsPage": lambda proj: gui_pages.ModelsPage(proj, _noop),
    "ExperimentAnalysisPage": lambda proj: gui_pages.ExperimentAnalysisPage(
        proj, _noop
    ),
    "RecordingDetailPage": lambda proj: gui_pages.RecordingDetailPage(
        proj.get_recording("recording_001"), experiment=proj
    ),
    # Collection pages the shell opens from the rail and the banner. Every one of these builds
    # from a bare experiment and none was here, because the list is hand-written and was complete
    # on the day it was written.
    # Opens its **not-found** state on a bare experiment, like the other fit detail pages here —
    # a real screen (a deleted export, a stale Active-runs row), and a different one from the
    # populated page `tests/gui/test_export_run_detail.py` drives.
    "ExportRunDetailPage": lambda proj: gui_pages.ExportRunDetailPage(
        proj, "export_001"
    ),
    "ComparisonsPage": lambda proj: gui_comparison.ComparisonsPage(proj, _noop),
    "ModelManagerPage": lambda proj: gui_pages.ModelManagerPage(
        proj, _noop, _noop, _noop
    ),
    "ToolRunPage": lambda proj: gui_pages.ToolRunPage("quick_segment", _noop),
    # -- keypoints (§26): PR 2 the Keypoints manager and the example-set picker · PR 4/5
    #    run viewers --
    "KeypointsManagerPage": lambda proj: gui_keypoints_manager.KeypointsManagerPage(),
    "ExampleSetPicker": lambda proj: gui_example_set_picker.ExampleSetPicker(proj),
}

#: Pages the shell opens that this file does not build, and where they *are* built. Each is a
#: claim, checked by `test_every_page_the_shell_opens_is_constructed_somewhere`: a detail page
#: needs a real record, and the file that already builds one is the right place for it.
COVERED_ELSEWHERE = {
    "RunDetailPage": "tests/gui/test_run_detail.py (needs a completed run's outputs)",
    "AnalysisRunDetailPage": "tests/gui/test_run_detail.py",
    "RecordingRunDetailPage": "tests/gui/test_run_detail.py",
    "ComparisonDetailPage": "tests/gui/test_comparison_gui.py (needs a saved comparison)",
    "DatasetDetailPage": "tests/gui/test_datasets.py (needs a generated dataset)",
    "ModelDetailPage": "tests/gui/test_models.py (needs a trained model record)",
    "GlobalModelDetailPage": "tests/gui/test_model_manager.py (needs a model record)",
    "ToolRunDetailPage": "tests/gui/test_tools.py (needs a saved tool run)",
    # The Configuration drill-ins and the Docs mode probe the *machine* — a CUDA check, a
    # backend import, the docs build. `test_configuration_pages.py` and
    # `test_configuration_panel.py` build them with that isolation in place; building them here
    # as well ran the real probes and aborted the whole gui tier on CI's Linux runner (exit 134,
    # a core dump, not an assertion). "Constructed somewhere" is what this guard asks for.
    "GpuCudaPage": "tests/gui/test_configuration_pages.py (probes CUDA; needs its fixtures)",
    "HuggingFacePage": "tests/gui/test_configuration_pages.py (reads the credential store)",
    "Sam3BackendPage": "tests/gui/test_configuration_panel.py (probes the sam3 import)",
    "DocumentationPage": "tests/gui/test_configuration_panel.py (builds the docs site)",
}


def _pages_the_shell_opens() -> set[str]:
    """Every `*Page` class constructed inside one of `app.py`'s navigation openers.

    Derived, because `PAGES` is hand-written: it held eleven of the twenty-six the shell can
    show, and nine of the missing ones needed no fixture at all — they were simply never added.
    """
    import ast
    import pathlib

    import detail.gui.app

    source = pathlib.Path(detail.gui.app.__file__).read_text(encoding="utf-8-sig")
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name.startswith(("_open_", "_go_to_"))
        ):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name.endswith("Page"):
                found.add(name)
    return found


def test_every_page_the_shell_opens_is_constructed_somewhere():
    """A page nothing constructs is a page whose `signal.connect` to a renamed method raises
    only when a user opens it — the whole reason this file exists."""
    opens = _pages_the_shell_opens()

    assert len(opens) >= 20, f"only {len(opens)} pages found; the AST walk broke"
    uncovered = sorted(opens - set(PAGES) - set(COVERED_ELSEWHERE))
    assert not uncovered, (
        f"the shell opens {uncovered} and no test constructs them. Add them to PAGES, or to "
        f"COVERED_ELSEWHERE naming the file that does."
    )


def test_every_covered_elsewhere_entry_is_still_a_page():
    """A stale claim is how the list stopped being complete in the first place."""
    stale = sorted(set(COVERED_ELSEWHERE) - _pages_the_shell_opens())
    assert (
        not stale
    ), f"COVERED_ELSEWHERE names pages the shell no longer opens: {stale}"


def test_experiment_and_recording_pages_build(qapp, tmp_path, monkeypatch):
    proj = _experiment(tmp_path)
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)  # isolate Home
    for make in PAGES.values():
        make(proj)


@pytest.mark.parametrize("name", sorted(PAGES))
def test_pages_top_justify_rather_than_spreading(qapp, tmp_path, monkeypatch, name):
    """A page in the content area is handed the whole window. If nothing in it declares that
    it wants the slack — a fill child, or a trailing stretch — Qt spreads it across the labels
    and the page reads as three hundred pixels of air between a heading and its table.
    """
    proj = _experiment(tmp_path)
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    assert_top_justified(PAGES[name](proj))


@pytest.mark.parametrize("name", sorted(PAGES))
def test_pages_share_one_left_margin(qapp, tmp_path, monkeypatch, name):
    """The horizontal twin of the rule above, and the half nothing measured.

    `sizing.clamp_width` promises that a capped block leaves its surplus "as margin on the
    right rather than centring the content, because everything else on these pages is
    left-aligned and a centred block would not line up with the heading above it". That is a
    claim about an *x*, and no test in this suite had ever read one — so the page a user
    lands on could drift out of its own column and only a person opening the app would see
    it. Every page here obeys it today; this is what keeps the next one honest.
    """
    proj = _experiment(tmp_path)
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    assert_one_content_column(PAGES[name](proj))


def test_the_margin_check_would_catch_an_indented_row(qapp):
    """Guard the guard. The check passes on all of `PAGES` today, so without this it would
    be indistinguishable from one whose measurement had quietly stopped working."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from tests.gui.layout_checks import off_the_content_column

    page = QWidget()
    column = QVBoxLayout(page)
    column.addWidget(QLabel("flush with the margin"))
    # A centred block — the exact failure `clamp_width`'s docstring says it avoids, and the
    # one a capped widget falls into when its item carries an alignment.
    adrift = QLabel("centred, so it lines up with nothing above it")
    adrift.setMaximumWidth(200)
    column.addWidget(adrift, alignment=Qt.AlignmentFlag.AlignHCenter)
    assert off_the_content_column(page), "a centred block must be reported"


def test_placeholder_pages_build(qapp):
    # Tools + Documentation are both real landing pages — both construct.
    gui_pages.ToolsPage(_noop)
    gui_pages.DocumentationPage()


class _Sig:
    """A minimal signal stand-in (connect/emit) for the sync worker below."""

    def __init__(self):
        self._fns = []

    def connect(self, fn):
        self._fns.append(fn)

    def emit(self, *args):
        for fn in self._fns:
            fn(*args)


class _SyncWorker:
    """Runs the work callable inline (no real QThread — the full GUI suite segfaults
    on a live thread here), forwarding progress + result through signal stand-ins."""

    def __init__(self, work, *, receiver):
        self._work = work
        self.tick = _Sig()
        self.done = _Sig()
        self.failed = _Sig()

    def start(self):
        try:
            result = self._work(lambda s, f, d: self.tick.emit(round(f * 1000), s))
        except Exception as exc:  # noqa: BLE001 — mirror the worker boundary
            self.failed.emit(exc)
            return
        self.done.emit(result)


def test_documentation_page_opens_built_site(qapp, monkeypatch):
    # When the site is already built + fresh, clicking opens it straight away (no rebuild).
    opened = {}
    monkeypatch.setattr(gui_pages.docs_access, "needs_build", lambda: False)
    monkeypatch.setattr(
        gui_pages.docs_access,
        "local_url",
        lambda page="index", anchor=None: "http://x/y",
    )
    monkeypatch.setattr(
        gui_pages.QDesktopServices,
        "openUrl",
        lambda url: opened.setdefault("url", url.toString()),
    )
    page = gui_pages.DocumentationPage()
    page._open.click()
    assert opened["url"] == "http://x/y"


def test_documentation_page_builds_off_thread_when_missing(qapp, monkeypatch):
    # First open (site absent/stale) builds off-thread, then opens the result.
    built = {}
    opened = {}
    monkeypatch.setattr(gui_pages.docs_access, "needs_build", lambda: True)
    monkeypatch.setattr(
        gui_pages.docs_access,
        "ensure_site_built",
        lambda progress=None: built.setdefault("progress", progress),
    )
    monkeypatch.setattr(
        gui_pages.docs_access,
        "local_url",
        lambda page="index", anchor=None: "http://x/z",
    )
    monkeypatch.setattr(gui_pages, "StageProgressWorker", _SyncWorker)
    monkeypatch.setattr(
        gui_pages.QDesktopServices,
        "openUrl",
        lambda url: opened.setdefault("url", url.toString()),
    )
    page = gui_pages.DocumentationPage()
    page._open.click()
    assert built["progress"] is not None  # progress callback wired through the build
    assert opened["url"] == "http://x/z"


def test_help_button_pops_a_panel_with_the_topic_prose(qapp):
    from detail.gui.help_ui import HelpPopover, help_button

    button = help_button("text-prompt")
    button.click()
    assert isinstance(button._popover, HelpPopover)
    assert "searches for" in button._popover._label.text().lower()  # from the glossary


def test_detection_form_prompt_help_opens_its_popover(qapp):
    # Verify through the real entry point: the detection config form's ? pops the
    # text-prompt definition (not a widget built in isolation).
    from PyQt6.QtWidgets import QToolButton

    form = GUI_MODULES["detection"].config_widget()
    info = [b for b in form.findChildren(QToolButton) if b.text() == "?"]
    assert info, "no ? help button on the detection form"
    # The form now carries several ?s (model, backend, prompt, …); assert one of them
    # pops the text-prompt definition rather than relying on button order.
    texts = []
    for button in info:
        button.click()
        texts.append(button._popover._label.text().lower())
    assert any(
        "prompt-capable" in text for text in texts
    ), "no ? on the detection form popped the text-prompt definition"


def test_field_help_button_pops_the_field_doc(qapp):
    # A field ? renders that field's entry from its module page (label + meta + help).
    from detail.gui.help_ui import field_help_button

    button = field_help_button("detection", "conf_threshold")
    button.click()
    text = button._popover._label.text().lower()
    assert "confidence threshold" in text and "conf_threshold" in text


def test_help_popover_opens_full_docs_when_built(qapp, monkeypatch):
    from detail.gui import help_ui

    opened = {}
    monkeypatch.setattr(help_ui.docs_access, "needs_build", lambda: False)
    monkeypatch.setattr(
        help_ui.docs_access,
        "local_url",
        lambda page="index", anchor=None: "http://x/tp",
    )
    monkeypatch.setattr(
        help_ui.QDesktopServices,
        "openUrl",
        lambda url: opened.setdefault("url", url.toString()),
    )
    button = help_ui.help_button("text-prompt")
    button.click()
    button._popover._open_full()
    assert opened["url"] == "http://x/tp"


def test_open_full_docs_shows_progress_while_rebuilding(qapp, monkeypatch):
    # "Open full docs" when the site must rebuild is a several-second mkdocs build; it
    # now runs behind a progress bar over the window instead of a silent wait.
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QWidget

    from detail import help as help_mod
    from detail.gui import help_ui
    from detail.gui.busy import BusyOverlay

    opened = {}
    monkeypatch.setattr(help_ui.docs_access, "needs_build", lambda: True)

    def fake_build(progress=None, force=False):
        if progress:
            progress("Building documentation", 0.5, "half")
        return "site"

    monkeypatch.setattr(help_ui.docs_access, "ensure_site_built", fake_build)
    monkeypatch.setattr(
        help_ui.docs_access, "local_url", lambda page="index", anchor=None: "http://x/b"
    )
    monkeypatch.setattr(
        help_ui.QDesktopServices,
        "openUrl",
        lambda url: opened.setdefault("url", url.toString()),
    )

    host = QWidget()
    host.resize(320, 220)
    help_ui.build_and_open_docs(host, help_mod.HelpTopic("gui", "runs"))
    assert host.findChildren(BusyOverlay), "no progress shown during the docs rebuild"
    for _ in range(300):  # let the off-thread build finish → the browser opens
        qapp.processEvents()
        if opened:
            break
        QTest.qWait(10)
    assert opened.get("url") == "http://x/b"


def test_configuration_pages_build(qapp):
    # GPU / CUDA and Hugging Face are now real Configuration surfaces (§1); building
    # them exercises the off-thread accelerator probe wiring and the token form.
    gui_pages.ConfigurationPage(_noop)  # the setup-checklist landing page
    gui_pages.GpuCudaPage()
    gui_pages.Sam3BackendPage()  # the SAM 3 backend install-command page
    gui_pages.HuggingFacePage()
    gui_pages.ModelManagerPage(
        None, _noop, _noop, _noop
    )  # the unified model-manager page
    gui_pages.GlobalModelDetailPage(
        "nope"
    )  # a machine-global model detail (not-found path)
    gui_pages.ImportModelDialog()  # the import-with-metadata dialog


def _walk_items(tree) -> list:
    """Every tree item, top-level and nested."""
    result = []
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        result.append(item)
        for c in range(item.childCount()):
            stack.append(item.child(c))
    return result


def _items_by_kind(tree) -> dict:
    """First tree item for each ``(kind, payload)`` role, walking the whole tree."""
    from detail.gui.app import _ROLE

    result = {}
    for item in _walk_items(tree):
        result.setdefault(item.data(0, _ROLE)[0], item)
    return result


def test_page_top_help_opens_the_current_screen_doc(qapp, tmp_path, monkeypatch):
    # Every navigable screen gets a page-top ? that pops its doc section.
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()
    window._set_page_help("metadata")
    assert window._page_help_kind == "metadata"
    assert window._page_help.isVisibleTo(window)  # a screen with a doc shows the ?
    window._open_page_help()
    assert window._page_help_popover._label.text().strip()  # rendered the screen's doc

    # The Documentation screen is itself the docs, so it shows no redundant page-top Info ?.
    window._set_page_help("documentation")
    assert not window._page_help.isVisibleTo(window)


def test_version_history_shows_the_running_build_and_opens_its_history(
    qapp, tmp_path, monkeypatch
):
    # The build label sits at the right end of the mode banner and the Version History button
    # left of Info; both are always visible — neither is tied to a page. Git is stubbed here
    # (tests/core/test_buildinfo.py runs the real reads); what this checks is that the real
    # button reaches all of it.
    from PyQt6.QtWidgets import QLabel

    from detail.core import buildinfo
    from detail.gui.app import MainWindow
    from detail.gui.version_history import VersionHistoryPopover

    info = buildinfo.BuildInfo(
        version="0.1.0",
        commit="abc1234",
        date="2026-09-20",
        commit_count=832,
        branch="main",
    )
    # Held until the window's width without it has been measured.
    release = threading.Event()
    monkeypatch.setattr(buildinfo, "build_info", lambda: release.wait(10) and info)
    monkeypatch.setattr(
        buildinfo,
        "history",
        lambda: [buildinfo.HistoryEntry("abc1234", "2026-09-20", "gui: a change (#9)")],
    )
    monkeypatch.setattr(
        buildinfo,
        "check_for_updates",
        lambda: buildinfo.UpdateStatus(behind=3, ahead=0, upstream_date="2026-09-22"),
    )
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()
    window.show()
    qapp.processEvents()
    narrowest = window.minimumSizeHint().width()
    assert window._version_history.isVisibleTo(window)
    assert window._build_label.isVisibleTo(window)
    # In the banner, not the strip beside Info, where it pushed Export off its edge.
    assert window._build_label.parent() is window._banner
    release.set()
    assert _drain(qapp, lambda: window._build_label.build_text() == info.label())
    # Always present, so it must not widen the window: at full width it raised the minimum
    # from 572px to 732px and squeezed the rail out of room to widen (test_window_state).
    # A new minimum climbs label -> banner -> window one posted layout pass at a time, so give
    # it room to arrive rather than reading it after a single pump.
    widened = drain_until(
        qapp,
        lambda: window.minimumSizeHint().width() != narrowest,
        300,
        raise_on_timeout=False,
    )
    assert not widened, f"{narrowest}px -> {window.minimumSizeHint().width()}px"

    window._version_history.click()
    popover = window._version_history_popover
    assert isinstance(popover, VersionHistoryPopover)
    assert "abc1234 on main, 2026-09-20" in popover._build.text()
    assert _drain(
        qapp,
        lambda: any(
            "gui: a change (#9)" in label.text()
            for label in popover.findChildren(QLabel)
        ),
    )
    popover._check.click()
    assert _drain(
        qapp, lambda: "3 commits behind origin/main" in popover._update_status.text()
    )
    popover.close()
    window.close()


def test_app_shell_banner_modes_and_in_project_rail(qapp, tmp_path, monkeypatch):
    # The banner shell: five modes, each reached by clicking the real banner button — a
    # renamed/removed handler would raise on click. Configuration steps drill into full-width
    # sub-pages; a left rail appears inside a project and in Quick tools, nowhere else.
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    window = MainWindow()

    # The banner lists the five modes in order; Home is the default landing (no rail).
    assert [
        window._banner.button(m).text()
        for m in ("home", "config", "models", "keypoints", "tools", "docs")
    ] == [
        "Experiments",
        "Configuration",
        "Model manager",
        "Keypoints manager",
        "Quick tools",
        "Docs",
    ]
    assert window._banner.button("home").isChecked()
    assert isinstance(window._content.currentWidget(), gui_pages.HomePage)
    assert window._tree.isHidden()  # Experiments is full-width — no rail

    # Each mode swaps the full-width content stack; drive the real banner buttons.
    window._banner.button("config").click()
    assert isinstance(window._content.currentWidget(), gui_pages.ConfigurationPage)
    assert window._tree.isHidden()
    # Configuration steps drill into full-width sub-pages (the checklist's Open buttons).
    window._go_to_node("gpu_cuda")
    assert isinstance(window._content.currentWidget(), gui_pages.GpuCudaPage)
    window._go_to_node("sam3_backend")
    assert isinstance(window._content.currentWidget(), gui_pages.Sam3BackendPage)
    window._go_to_node("hugging_face")
    assert isinstance(window._content.currentWidget(), gui_pages.HuggingFacePage)

    window._banner.button("models").click()
    assert isinstance(window._content.currentWidget(), gui_pages.ModelManagerPage)
    assert window._tree.isHidden()
    # Both Configuration checklist steps land on the one Model manager page.
    window._go_to_node("global_models")
    assert isinstance(window._content.currentWidget(), gui_pages.ModelManagerPage)
    window._go_to_node("distill_model")
    assert isinstance(window._content.currentWidget(), gui_pages.ModelManagerPage)

    window._banner.button("keypoints").click()
    assert isinstance(
        window._content.currentWidget(), gui_keypoints_manager.KeypointsManagerPage
    )
    assert window._tree.isHidden()  # skeletons are machine-wide: no rail

    window._banner.button("tools").click()
    assert isinstance(window._content.currentWidget(), gui_pages.ToolsPage)
    assert not window._tree.isHidden()  # Quick tools carries its own rail

    window._banner.button("docs").click()
    assert isinstance(window._content.currentWidget(), gui_pages.DocumentationPage)
    assert window._tree.isHidden()

    # A project is the other place a rail appears — open the experiment from Home.
    window._banner.button("home").click()
    window._open_experiment(proj)
    items = _items_by_kind(window._tree)
    # From the declaration, not a literal: this build hides one of them (§25), and a hardcoded
    # list here would be a fourth copy of `gui/sections.py`.
    assert {s.kind for s in gui_sections.sections()} <= set(items)
    assert "benchmarks" not in items  # removed (§18)
    assert "data" not in items  # the Data tab was removed
    assert not window._tree.isHidden()  # the project rail
    assert isinstance(window._content.currentWidget(), gui_pages.ExperimentOverviewPage)
    assert window._banner.button(
        "home"
    ).isChecked()  # the banner stays on Home in a project


def test_home_row_opens_experiment(qapp, tmp_path, monkeypatch):
    # Double-clicking an experiment row on Home opens its project view (rail + overview).
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    _experiment(tmp_path)
    proj2 = Experiment.create("p2", tmp_path)
    proj2.create_recording("recording_001")

    window = MainWindow()  # boots on Home (the experiments list)
    home = window._content.currentWidget()
    assert isinstance(home, gui_pages.HomePage)
    row = next(i for i, e in enumerate(home._experiments) if e.name == "p2")
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    # The real path: double-click the experiment's card.
    QTest.mouseDClick(home._cards[row], Qt.MouseButton.LeftButton)
    assert window._experiment.name == "p2"
    assert not window._tree.isHidden()  # the project rail appears
    assert isinstance(window._content.currentWidget(), gui_pages.ExperimentOverviewPage)


def test_click_toggle_is_uniform_across_folders(qapp, tmp_path, monkeypatch):
    # Inside the project rail the toggle rule is one rule for every folder: a section
    # (Recordings) opens on a name-click, always shows its page, and closes on the next.
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)  # one recording → Recordings has a child (a folder)

    window = MainWindow()
    window._open_experiment(proj)
    recordings = _items_by_kind(window._tree)["recordings"]
    window._on_tree_click(recordings, 0)  # first click opens the section
    assert recordings.isExpanded()
    # a click always shows the clicked node's page — even the one that collapses it.
    assert isinstance(
        window._content.currentWidget(), gui_pages.ExperimentRecordingsPage
    )
    window._on_tree_click(recordings, 0)  # a nested section closes on click too
    assert not recordings.isExpanded()
    window._on_tree_click(recordings, 0)
    assert recordings.isExpanded()


def test_navigation_preserves_tree_expansion(qapp, tmp_path, monkeypatch):
    # The file-explorer promise inside a project: several rail sections expand at once and
    # navigating to a sibling page never collapses them or rebuilds the rail (same node
    # objects). The old design rebuilt the whole tree on every click — revert and this fails.
    from detail.gui.app import MainWindow
    from detail.storage import runs as runs_store

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)  # "p" + one recording
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "detection",
            "recording_ids": ["recording_001"],
            "per_recording": {},
            "config": {},
            "status": "complete",
            "created_at": "2026-08-01T10:00:00Z",
        },
    )

    window = MainWindow()
    window._open_experiment(proj)
    items = _items_by_kind(window._tree)
    items["recordings"].setExpanded(True)  # both are folders by declaration (#316)
    items["runs"].setExpanded(True)
    assert items["recordings"].isExpanded() and items["runs"].isExpanded()

    # Navigate to a sibling page (Metadata): neither section collapses and the rail nodes
    # are the same objects (no rebuild on a navigation click).
    window._on_tree_click(items["metadata"], 0)
    assert isinstance(window._content.currentWidget(), gui_pages.MetadataPage)
    after = _items_by_kind(window._tree)
    assert after["recordings"] is items["recordings"] and after["runs"] is items["runs"]
    assert after["recordings"].isExpanded() and after["runs"].isExpanded()


# --- the rail reads a section when it opens, off the UI thread (#316) ------- #
def _spy_section_thread(monkeypatch, kind: str) -> list:
    """Wrap one section's child enumeration; return the threads it is called on.

    The wrap goes on the *section*, not on ``Experiment.recordings``: the experiment overview
    reads the same properties from its own worker, so a spy on the property records an
    off-thread call whether or not the rail made it — a guard that would pass with the rail
    still inline.
    """
    declared = gui_sections.sections()
    real = {s.kind: s.children for s in declared}[kind]
    threads: list = []

    def wrapped(experiment):
        threads.append(threading.current_thread())
        return real(experiment)

    patched = tuple(
        dataclasses.replace(s, children=wrapped) if s.kind == kind else s
        for s in declared
    )
    monkeypatch.setattr(gui_sections, "sections", lambda: patched)
    return threads


def test_a_closed_rail_section_reads_nothing(qapp, tmp_path, monkeypatch):
    """Opening an experiment builds folders, not their contents.

    Building them cost one record read per recording and per run — on the UI thread, for
    children invisible until a folder is expanded. Nothing looked wrong: every folder is
    collapsed, so the work was correct, complete and unseen, and only a workspace slow enough
    to make it visible (seconds per rebuild on a network share) exposed it.
    """
    from detail.gui.app import _FOLDER, MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    threads = _spy_section_thread(monkeypatch, "recordings")

    window = MainWindow()
    window._open_experiment(proj)
    node = _items_by_kind(window._tree)["recordings"]

    assert bool(
        node.data(0, _FOLDER)
    ), "a section with children must declare itself a folder"
    # The property, not a proxy for it: a *folder* whose contents have not been read. Drained
    # rather than checked immediately, or a fill merely queued would pass this.
    assert not drain_until(
        qapp, lambda: node.childCount() > 0, 500, raise_on_timeout=False
    )
    assert (
        threads == []
    ), f"the rail read its children with every folder closed: {threads}"


def test_the_rail_reads_an_opened_section_off_the_ui_thread(
    qapp, tmp_path, monkeypatch
):
    """The property is which thread the read happens on — not that it is fast here, where it
    is a handful of local stats either way (template: Home's card counts).
    """
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    proj.add_recording("second")
    threads = _spy_section_thread(monkeypatch, "recordings")

    window = MainWindow()
    window._open_experiment(proj)
    node = _items_by_kind(window._tree)["recordings"]
    window._on_tree_click(node, 0)  # the real door: a click opens the folder

    assert _drain(qapp, lambda: node.childCount() == 2), "the rail never filled"
    assert threads and threading.main_thread() not in threads, (
        "the rail read its section on the UI thread — route it through _fill_section's "
        f"BackgroundCall: {threads}"
    )


def test_a_closed_section_still_shows_it_can_be_opened(qapp, tmp_path, monkeypatch):
    """The ▸ is declared too, for the same reason the toggle is.

    Qt draws the arrow only when an item *has* children, so making them lazy silently took it
    off every section — the one cue that says the row opens at all, gone until you clicked a
    row that looked like a leaf. Found by rendering the rail and looking at it; every
    behavioural assertion here (child counts, expansion, threads) passed with it missing.
    """
    from PyQt6.QtWidgets import QTreeWidgetItem

    from detail.gui.app import MainWindow

    show = QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
    when_childless = QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicatorWhenChildless

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)  # one recording; no comparisons at all

    window = MainWindow()
    window._open_experiment(proj)
    items = _items_by_kind(window._tree)

    assert items["recordings"].childIndicatorPolicy() == show
    assert items["metadata"].childIndicatorPolicy() != show  # a leaf never claims one

    # ...and once a section's contents are known, Qt decides again — so a section that is
    # genuinely empty stops advertising rows it does not have.
    window._on_tree_click(items["comparisons"], 0)
    assert _drain(
        qapp,
        lambda: items["comparisons"].childIndicatorPolicy() == when_childless,
    ), "an empty section kept the arrow after it was read"
    assert items["comparisons"].childCount() == 0


def test_a_folder_opens_before_it_has_any_children(qapp, tmp_path, monkeypatch):
    """The toggle asks the node's **declaration**, never its children.

    ``childCount()`` is a property of the data, not of the section: correct only while
    children are built eagerly, and silently wrong the moment they are not — the click
    navigates, the folder never opens, and no test that clicked an already-filled folder
    could see it.
    """
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)

    window = MainWindow()
    window._open_experiment(proj)
    node = _items_by_kind(window._tree)["recordings"]
    assert node.childCount() == 0  # the state the predicate has to survive

    window._on_tree_click(node, 0)
    assert node.isExpanded(), "a folder with no children yet did not open on click"
    assert node.childCount() == 0, "the fill is meant to be off-thread, not inline"
    assert _drain(qapp, lambda: node.childCount() == 1)
    assert node.isExpanded()  # ...and the arriving children do not close it


def test_a_refresh_refills_the_sections_that_were_open(qapp, tmp_path, monkeypatch):
    """A structural change rebuilds the folders and re-reads exactly what was open.

    The re-expand *is* the invalidation, so this is the one path where a lazy rail could
    quietly stop updating: the rail would keep its shape, keep its expansion, and go on
    showing the runs from before the launch.
    """
    from detail.gui.app import MainWindow
    from detail.storage import runs as runs_store

    def _write(proj, run_id):
        runs_store.write_run(
            proj.dir,
            run_id,
            {
                "id": run_id,
                "module_id": "detection",
                "recording_ids": ["recording_001"],
                "per_recording": {},
                "config": {},
                "status": "complete",
                "created_at": "2026-08-01T10:00:00Z",
            },
        )

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    _write(proj, "run_001")

    window = MainWindow()
    window._open_experiment(proj)
    runs = _items_by_kind(window._tree)["runs"]
    window._on_tree_click(runs, 0)
    assert _drain(qapp, lambda: runs.childCount() == 1)

    _write(proj, "run_002")
    window._refresh_active_experiment()
    runs = _items_by_kind(window._tree)["runs"]  # a rebuild — a new node object
    assert (
        runs.isExpanded()
    ), "the refresh dropped the expansion it is meant to preserve"
    assert _drain(
        qapp, lambda: runs.childCount() == 2
    ), "the reopened section never refilled"
    labels = [runs.child(i).text(0) for i in range(runs.childCount())]
    assert [label.split("·")[-1].strip() for label in labels] == ["run_001", "run_002"]

    # ...and a section the user never opened is still not read.
    recordings = _items_by_kind(window._tree)["recordings"]
    assert recordings.childCount() == 0

    # The Overview node survives the rebuild. It is not a `Section`, so it is the one rail
    # node `_build_project_rail` could drop without any section test noticing — and a way
    # back that disappears the first time a run is launched is no way back at all.
    assert "experiment" in _items_by_kind(window._tree)


def test_overview_is_the_rail_node_you_can_come_back_to(qapp, tmp_path, monkeypatch):
    """The way back (#340/#331). Entering a project shows the overview; clicking a section
    replaces it; clicking **Overview** returns — with the Export control and the page help
    it carries, both of which left with it before this node existed.
    """
    from detail.gui.app import MainWindow
    from detail import help as help_mod

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    window = MainWindow()
    window._open_experiment(proj)
    assert window._tree.topLevelItem(0).text(0) == "Overview", "it leads the rail"

    window._navigate(_items_by_kind(window._tree)["recordings"])
    assert not isinstance(
        window._content.currentWidget(), gui_pages.ExperimentOverviewPage
    )

    window._navigate(_items_by_kind(window._tree)["experiment"])
    assert isinstance(window._content.currentWidget(), gui_pages.ExperimentOverviewPage)
    assert not window._export.isHidden(), "the bundle is exportable again"
    # Its own doc, not whichever kind happened to be set last. `PAGE_DOCS["experiment"]`
    # existed and resolved the whole time; nothing ever pointed at it, so the ? answered
    # about the experiments *list* instead.
    assert help_mod.PAGE_DOCS[window._page_help_kind].anchor == "experiment"
    assert (
        window._tree.currentItem().text(0) == "Overview"
    ), "the rail says where you are"


def test_reopening_a_section_does_not_duplicate_its_children(
    qapp, tmp_path, monkeypatch
):
    """Collapse and reopen is free, and a second delivery replaces rather than appends."""
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    threads = _spy_section_thread(monkeypatch, "recordings")

    window = MainWindow()
    window._open_experiment(proj)
    node = _items_by_kind(window._tree)["recordings"]
    for _ in range(3):
        window._on_tree_click(node, 0)  # open / close / open
    assert _drain(qapp, lambda: node.childCount() == 1)
    assert (
        node.childCount() == 1
    ), "reopening a section appended a second copy of its rows"
    assert len(threads) == 1, f"one tree, one read per section: {threads}"


def test_launch_analysis_dialog_builds(qapp, tmp_path):
    from detail.gui.launch import LaunchAnalysisDialog

    dialog = LaunchAnalysisDialog(_experiment(tmp_path))
    assert (
        dialog._module_id() in ANALYSIS_GUI_MODULES
    )  # picker populated + config built
    assert dialog._config is not None


def test_app_shell_analysis_navigation(qapp, tmp_path, monkeypatch):
    # Open a experiment (builds the tree incl. the Analysis node + its run children)
    # and route through the analysis handlers — covers the app-shell glue.
    from detail.gui.app import MainWindow
    from detail.storage import analysis_runs as ar

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    ar.write_analysis_yaml(
        ar.analysis_dir(proj.dir, "an_01"),
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "status": "complete",
        },
    )
    window = MainWindow()
    # The Model manager banner mode, which is the only route to this surface: it embeds
    # `GlobalModelsPage` for its Hugging Face table. The standalone `_open_global_models`
    # opener this line used to call was reachable from nothing in `detail/` and was deleted
    # — `test_nav_dispatch` fails on an opener nobody names.
    window._open_model_manager()
    window._open_experiment(proj)  # populates the folder incl. its Analysis section
    window._open_analysis()  # → ExperimentAnalysisPage
    window._open_analysis_run("an_01")  # → AnalysisRunDetailPage
    window._open_recording_run(
        "recording_001", "detection", "det_99"
    )  # → RecordingRunDetailPage


def test_open_experiment_reconciles_runs(qapp, tmp_path, monkeypatch):
    # experiment-open must fold orphaned processing-run status back (§15), beside the
    # existing import reconcile.
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    called = []
    monkeypatch.setattr(type(proj), "reconcile_runs", lambda self: called.append(self))
    window = MainWindow()
    window._open_experiment(proj)
    assert called == [proj]


def test_distill_surfaces_build(qapp, tmp_path, monkeypatch):
    from detail.gui import pages as gui_pages
    from detail.gui.fit import DistillDialog
    from detail.storage import distill_runs as distill_store
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = Experiment.create("p", tmp_path)
    DistillDialog(experiment)  # pre-selected experiment
    DistillDialog()  # reachable with nothing open — the experiment picker fills itself
    # a completed run record → DistillRunDetail (the lazy leaf that only builds when opened)
    distill_store.write_run(
        experiment.dir,
        "distill_001",
        {
            "id": "distill_001",
            "name": "m",
            "status": "complete",
            "stages": {
                "label": {"status": "complete", "dataset_id": "dataset_001"},
                "train": {"status": "complete", "model_id": "model_001"},
            },
        },
    )
    gui_pages.DistillRunDetail(
        experiment, "distill_001", on_open_model=lambda _id: None
    )


def test_experiment_card_shows_recordings_and_completed_runs(
    qapp, tmp_path, monkeypatch
):
    # The card answers "how far has this one got" - from the experiment's own run records,
    # counting only completed ones. A failed run is not progress.
    from PyQt6.QtWidgets import QLabel

    from detail.storage import runs as runs_store

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    for run_id, module_id, status in (
        ("run_001", "detection", "complete"),
        ("run_002", "detection", "complete"),
        ("run_003", "tracking", "complete"),
        ("run_004", "embedding", "failed"),
    ):
        runs_store.write_run(
            proj.dir,
            run_id,
            {
                "id": run_id,
                "module_id": module_id,
                "recording_ids": ["recording_001"],
                "per_recording": {},
                "config": {},
                "status": status,
                "created_at": "2026-08-01T10:00:00Z",
            },
        )

    card = gui_pages._ExperimentCard(proj, lambda _e: None)
    card.set_facts(gui_pages._card_facts(proj))  # what HomePage's worker delivers
    labels = [lbl.text() for lbl in card.findChildren(QLabel)]
    assert "p" in labels  # the name
    assert any(
        "1 recording \u00b7" in t for t in labels
    )  # singular, not "1 recordings"
    assert (
        "detect 2  \u00b7  track 1  \u00b7  embed 0" in labels
    )  # the failure doesn't count


def test_experiment_card_without_runs_or_video_says_so(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QLabel

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    card = gui_pages._ExperimentCard(proj, lambda _e: None)
    card.set_facts(gui_pages._card_facts(proj))
    labels = [lbl.text() for lbl in card.findChildren(QLabel)]
    assert "no runs yet" in labels  # not "detect 0 . track 0 . embed 0"

    # A recording with no video renders a placeholder rather than an empty box.
    card.set_preview(None)
    assert "no video" in [lbl.text() for lbl in card.findChildren(QLabel)]


def test_home_lists_a_card_per_experiment(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    _experiment(tmp_path)
    Experiment.create("p2", tmp_path)
    page = gui_pages.HomePage(lambda _e: None)
    assert len(page._cards) == 2
    assert {c._experiment.name for c in page._cards} == {"p", "p2"}


def test_home_reads_a_cards_counts_off_the_ui_thread(qapp, tmp_path, monkeypatch):
    """A card's counts are disk, and Home must not go to disk while someone is looking.

    Measured on a workspace held on an SMB share: ~9 s of blocking I/O for a *single*
    card, scaling with the number of experiments, which Windows retitles "not responding".
    The property is which thread the read happens on — not that it is fast here, where it
    is a handful of local stats either way.
    """
    import threading

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    _experiment(tmp_path)
    reading_threads = []
    real = Experiment.processing_runs.fget

    def spy(self):
        reading_threads.append(threading.current_thread())
        return real(self)

    monkeypatch.setattr(Experiment, "processing_runs", property(spy))

    page = gui_pages.HomePage(lambda _e: None)
    card = page._cards[0]
    # A cross-thread `done` is queued, so nothing can have been delivered before the loop
    # is pumped: the card is on screen with its placeholder while the reads are in flight.
    assert card._summary.text() == "…"
    assert card._stages.text() == ""

    assert _drain(qapp, lambda: bool(reading_threads))
    assert reading_threads and threading.main_thread() not in reading_threads, (
        "the card's counts were read on the UI thread — route them through card_facts "
        "in a BackgroundCall"
    )
    # ...and the delivery lands, so off-threading did not simply lose the numbers.
    assert _drain(qapp, lambda: card._summary.text() != "…")
    assert "1 recording ·" in card._summary.text()
    assert card._stages.text() == "no runs yet"


def test_home_drops_a_delivery_computed_for_cards_that_are_gone(
    qapp, tmp_path, monkeypatch
):
    """The second delivery path into destroyed cards is a second instance of that crash.

    A refresh deletes the cards a result was computed for; touching a deleted widget takes
    the process down. Both of Home's workers carry the same generation, so a result from
    before a refresh is dropped rather than applied.
    """
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    _experiment(tmp_path)
    page = gui_pages.HomePage(lambda _e: None)
    stale = page._preview_gen
    page._refresh()  # deletes the cards the stale generation was computed for

    page._apply_facts(stale, [{"n_recordings": 99, "counts": {}, "modified_at": 0}])
    assert page._cards[0]._summary.text() == "…", "a stale result was applied anyway"
    page._apply_previews(stale, [None])
    assert page._cards[0]._thumb.text() == "…"


def test_experiment_cards_are_all_the_same_height(qapp, tmp_path, monkeypatch):
    # Rows scan as a list only if they are uniform: a long description, a missing still or
    # a missing run count must not change a card's height.
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    plain = _experiment(tmp_path)
    wordy = Experiment.create("p2", tmp_path)
    wordy.set_metadata(description="a description " * 40)  # far wider than the card
    Experiment.create("p3", tmp_path)  # no recordings, no runs, no still

    page = gui_pages.HomePage(lambda _e: None)
    heights = {c.height() for c in page._cards}
    assert len(page._cards) == 3
    assert len(heights) == 1, f"cards differ in height: {heights}"

    # And the long description is elided rather than widening the card or wrapping.
    card = next(c for c in page._cards if c._experiment.name == "p2")
    card.resize(600, card.height())
    assert card._description is not None
    assert card._description.text() != wordy.description
    assert card._description.toolTip() == wordy.description
    assert plain.name  # (keeps the fixture referenced)


def test_experiment_card_is_visually_separated_from_the_page(
    qapp, tmp_path, monkeypatch
):
    # StyledPanel renders as a hairline under Fusion, which left the rows looking like
    # page background with text floating on it. A card needs a real edge and its own fill.
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QFrame

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    card = gui_pages._ExperimentCard(_experiment(tmp_path), lambda _e: None)
    assert card.frameShape() == QFrame.Shape.Box
    assert card.lineWidth() >= 1
    assert card.autoFillBackground()
    # Filled from a palette pair, so the separation survives a dark theme rather than
    # being a colour that only works in one.
    assert card.palette().color(QPalette.ColorRole.Window) == card.palette().color(
        QPalette.ColorRole.Base
    )


def test_double_click_opens_from_anywhere_on_the_card(qapp, tmp_path, monkeypatch):
    # Reported: double-clicking an experiment with no video just selected text. Page text
    # is selectable (MainWindow._show), and a selectable QLabel swallows mouse events, so
    # a card that is mostly text had no live area at all.
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLabel

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)  # no video: the card is name + counts + placeholder
    opened = []
    card = gui_pages._ExperimentCard(proj, opened.append)
    card.set_facts(gui_pages._card_facts(proj))
    card.set_preview(None)
    gui_pages._make_selectable(card)  # exactly what showing the page does

    labels = {lbl.text(): lbl for lbl in card.findChildren(QLabel)}
    for text in ("p", "no runs yet", "no video"):
        opened.clear()
        QTest.mouseDClick(labels[text], Qt.MouseButton.LeftButton)
        assert opened == [proj], f"double-clicking {text!r} did not open the experiment"

    # ...and the card's own background still works.
    opened.clear()
    QTest.mouseDClick(card, Qt.MouseButton.LeftButton)
    assert opened == [proj]


def test_rail_lists_runs_in_the_order_they_ran(qapp, tmp_path, monkeypatch):
    """The Processing-pipeline node is chronological, like every sibling section.

    It used to sort by ``(module_id, id)``, which grouped runs alphabetically by module —
    so the rail read detection · run_002, embedding · run_004, tracking · run_003 while the
    Runs *table* on the same folder page (and ``detail process list``) read them in the
    order they ran. ``_items_by_kind`` cannot see this: it keeps one item per kind, so it
    is structurally blind to order. This walks the node's children instead.
    """
    from detail.gui.app import MainWindow
    from detail.storage import runs as runs_store

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    proj = _experiment(tmp_path)
    # Module-alphabetical order (detection, detection, embedding, tracking) differs from
    # run order here, so the two orderings are distinguishable — the reporter's exact case.
    for run_id, module_id in (
        ("run_001", "detection"),
        ("run_002", "detection"),
        ("run_003", "tracking"),
        ("run_004", "embedding"),
    ):
        runs_store.write_run(
            proj.dir,
            run_id,
            {
                "id": run_id,
                "module_id": module_id,
                "recording_ids": ["recording_001"],
                "per_recording": {},
                "config": {},
                "status": "complete",
                "created_at": "2026-08-01T10:00:00Z",
            },
        )

    window = MainWindow()
    window._open_experiment(proj)
    runs_node = _items_by_kind(window._tree)["runs"]
    window._on_tree_click(
        runs_node, 0
    )  # a section reads its children when it opens (#316)
    assert _drain(qapp, lambda: runs_node.childCount() == 4), "the rail never filled"
    labels = [runs_node.child(i).text(0) for i in range(runs_node.childCount())]

    assert [label.split("·")[-1].strip() for label in labels] == [
        "run_001",
        "run_002",
        "run_003",
        "run_004",
    ]  # under the old key: run_001, run_002, run_004, run_003
    # The label itself is unchanged — module first, then the id.
    assert labels[0].split("·")[0].strip() == "detection"


# --------------------------------------------------------------------------- #
# §20 Query — the launch form
# --------------------------------------------------------------------------- #
def _query_embeddings(tmp_path):
    """An experiment with one recording carrying a complete embedding run."""
    import numpy as np
    import yaml

    from detail.storage.zarr_io import EmbeddingPartWriter

    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    run_dir = sess.dir / "embeddings" / "emb_01"
    store = run_dir / "track_0" / "part_001.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with EmbeddingPartWriter(store, output_dim=8) as w:
        for i in range(200):
            w.add_row(i, i * 1_000_000, 0, rng.normal(size=8).astype(np.float32))
    (run_dir / "embedding.yaml").write_text(
        yaml.dump({"schema_version": 1, "id": "emb_01", "tracking_run_id": "track_01",
                   "model_id": "dinov3_convnext_base", "model_variant": "base",
                   "constructor": "full_frame", "output_dim": 8, "track_ids": [0],
                   "renderer_spec": [{"type": "init_binary_mask"}], "status": "complete"},
                  sort_keys=False)
    , encoding="utf-8")  # fmt: skip
    # The store alone is not enough: the embed-run combo reads the recording's *record*, so
    # a fixture that writes only the zarr leaves the picker empty and the form unusable.
    from detail.storage import recording as recording_store

    data = recording_store.read_recording(sess.dir)
    data.setdefault("embedding_runs", []).append(
        {
            "id": "emb_01",
            "status": "complete",
            "output": "embeddings/emb_01/",
            "model_id": "dinov3_convnext_base",
            "tracking_run_id": "track_01",
        }
    )
    recording_store.write_recording(sess.dir, data)
    return proj


def _clip(start_s, end_s):
    return {
        "recording_id": "recording_001",
        "track_id": 0,
        "start_t_ns": round(start_s * 1e9),
        "end_t_ns": round(end_s * 1e9),
    }


def _query_form_with_method(method_id=None):
    """The query form with a method chosen; the declared default when none is named.

    The default is `soft_knn` and the form opens on it, so most callers pass nothing. A
    caller that names one must name a **surfaced** method: `findData` returns -1 for a method
    §25 hides, and `setCurrentIndex(-1)` blanks the picker — which every test then fails on,
    somewhere far from the line that chose it. Asserted here so the failure names its cause.
    """
    from detail.configs.query import DEFAULT_METHOD

    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    index = widget._method.findData(method_id or DEFAULT_METHOD)
    assert index >= 0, (
        f"{method_id!r} is not in this build's method picker (hidden by §25?); use the "
        f"DETAIL_SHOW_ALL escape hatch if the test is about a hidden method"
    )
    widget._method.setCurrentIndex(index)
    return widget


def test_query_form_config_is_what_validate_accepts(qapp, tmp_path):
    """Surface parity, concretely: a form that *builds* is not a form that produces a
    runnable config. Drive the form as a user does, then feed the result to the same
    `validate()` the CLI and API call."""
    from detail.configs.query import DEFAULT_METHOD
    from detail.pipeline.query import QueryModule

    proj = _query_embeddings(tmp_path)
    widget = _query_form_with_method()
    widget.on_experiment_set(proj)
    widget.on_recordings_changed(list(proj.recordings))
    widget._behaviors = [
        {"name": "rearing", "examples": [_clip(0.010, 0.029)], "negatives": []}
    ]
    assert widget.validation_error() is None

    config = {"source_recording_ids": ["recording_001"], **widget.config()}
    normalized = QueryModule().validate(proj.dir, config)  # raises if the form lies
    assert normalized["behaviors"][0]["name"] == "rearing"
    assert normalized["method"] == DEFAULT_METHOD


def test_query_form_opens_on_the_declared_default_method(qapp):
    """The form no longer opens blank.

    §20 used to declare no default at all and have the form refuse until a method was picked,
    on the grounds that nothing separated the two. That held while nothing did; `soft_knn` is
    now far faster than `dtw` was, so the default is
    chosen on cost and the field's help says so. An empty picker does not leave the choice
    open — it makes everyone make it, on a field that lives in advanced settings.
    """
    from detail.configs.query import DEFAULT_METHOD

    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    assert widget._method.currentData() == DEFAULT_METHOD
    assert "" not in [
        widget._method.itemText(i) for i in range(widget._method.count())
    ], "the blank placeholder is still in the picker"
    # ...and the form is launchable without touching the method at all.
    widget._embed.addItem("emb_01", "emb_01")
    widget._behaviors = [
        {"name": "rearing", "examples": [_clip(0.0, 3.0)], "negatives": []}
    ]
    assert widget.validation_error() is None


def test_the_form_still_refuses_when_no_method_is_registered(qapp, monkeypatch):
    """The refusal path stays reachable. With an empty registry the picker has nothing to
    select, and saying so beats launching a run that cannot resolve a method."""
    from detail.core import registries

    monkeypatch.setattr(registries, "QUERY_METHODS", {})
    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    widget._embed.addItem("emb_01", "emb_01")
    widget._behaviors = [
        {"name": "rearing", "examples": [_clip(0.0, 3.0)], "negatives": []}
    ]
    assert "method" in (widget.validation_error() or "").lower()


def test_query_form_picks_the_only_method_when_there_is_one(qapp, monkeypatch):
    """Offering the single available option is not a claim about which is better, which is
    what the no-default rule guards against.

    The registry is narrowed to one rather than assumed to hold one: this used to read the
    real registry, so it silently stopped testing its own subject the day a second method was
    registered — and failed, which is the better of the two ways for that to go.

    The one it is narrowed to must be **surfaced**, since the picker filters through §25 as
    well: narrowing to a hidden method leaves the combo empty, which is a different subject.
    """
    from detail.core import registries
    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    monkeypatch.setattr(registries, "QUERY_METHODS", {"soft_knn": SoftKnnQueryMethod})
    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    assert widget._method.currentData() == "soft_knn"


def test_query_form_advises_on_short_and_few_clips_without_blocking(qapp):
    """§20's exemplar guardrails are warnings, not refusals: a user who knows their behavior
    is brief must still be able to mark a brief clip."""
    widget = _query_form_with_method()
    widget._behaviors = [
        {"name": "rearing", "examples": [_clip(0.0, 0.5)], "negatives": []}
    ]
    widget._refresh_table()
    assert widget._advice.text().startswith("⚠")
    text = widget._advice.text()
    assert "1 clip" in text and "under 2 s" in text
    # ...and it blocks nothing: what remains are the refusals that really are refusals.
    widget._embed.addItem("emb_01", "emb_01")
    assert widget.validation_error() is None


def test_query_form_advice_is_silent_on_a_healthy_set(qapp):
    # The paired negative: an advisory that is always on is an advisory nobody reads.
    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    widget._behaviors = [
        {
            "name": "rearing",
            "examples": [_clip(i * 10, i * 10 + 3) for i in range(4)],
            "negatives": [],
        }
    ]
    widget._refresh_table()
    assert widget._advice.text() == ""


def test_query_cli_and_gui_build_the_same_config(qapp, tmp_path, monkeypatch):
    """Surface parity as a value, not a promise.

    The CLI and the GUI are two front ends over one config dict. They drift silently: a key
    renamed on one side leaves the other building something the module still accepts but that
    means something else. So this captures what each actually passes to `run_analysis` and
    compares them, rather than asserting each is individually plausible.
    """
    from click.testing import CliRunner

    from detail.api.experiment import Experiment as _Experiment
    from detail.cli import main

    proj = _query_embeddings(tmp_path)

    captured: dict = {}

    class _Stop(Exception):
        """Halt the command once the config exists — running the search is not the point."""

    def capture(self, module_id, config):
        captured["module_id"] = module_id
        captured["config"] = config
        raise _Stop

    monkeypatch.setattr(_Experiment, "run_analysis", capture)
    # Neither side names a method: the point is that the two front ends agree, and letting
    # each take its own default is a stronger check than pinning both to the same string.
    CliRunner().invoke(
        main,
        ["analysis", "query", "--experiment", str(proj.dir),
         "--search", "recording_001", "--embed-run", "emb_01",
         "--behavior", "rearing", "--example", "rearing=recording_001:0:0.010-0.029"],
    )  # fmt: skip
    assert captured["module_id"] == "query"

    widget = _query_form_with_method()  # both sides on the declared default
    widget.on_experiment_set(proj)
    widget.on_recordings_changed(list(proj.recordings))
    widget._behaviors = [
        {"name": "rearing", "examples": [_clip(0.010, 0.029)], "negatives": []}
    ]
    gui_config = {"source_recording_ids": ["recording_001"], **widget.config()}

    assert set(gui_config) == set(captured["config"])
    for key in ("source_recording_ids", "embed_run_id", "track_id", "method"):
        assert gui_config[key] == captured["config"][key], key
    cli_behavior = captured["config"]["behaviors"][0]
    gui_behavior = gui_config["behaviors"][0]
    assert cli_behavior["name"] == gui_behavior["name"] == "rearing"
    assert cli_behavior["examples"] == gui_behavior["examples"]


def test_query_form_greys_non_examples_for_a_method_that_cannot_use_them(
    qapp, monkeypatch
):
    """Greyed with the reason, never hidden. A user who marks fifteen careful non-examples
    into a method that silently discards them has been misled by an empty affordance.

    The reason comes from the method itself (`NegativeSupport.reason`), so the sentence a user
    reads is about the method they picked rather than about whichever one happened to be
    written into the form.

    No registered method refuses non-examples since `dtw` was removed, so the refusal is
    declared on the default method here: the form reads the method's declaration and reason,
    and that is what this checks.
    """
    from detail.core.dataclasses import NegativeSupport
    from detail.query_methods.soft_knn import SoftKnnQueryMethod

    monkeypatch.setattr(
        SoftKnnQueryMethod,
        "negatives",
        NegativeSupport(
            support="unusable",
            reason="it scores distance to the examples and has nothing to score them against",
        ),
    )
    widget = _query_form_with_method("soft_knn")
    reason = widget._negatives_reason()
    assert reason is not None and "distance to the examples" in reason
    # Non-examples already on a behavior are kept, and the table says they will not be used.
    widget._behaviors = [{"name": "rear", "examples": [], "negatives": [{}, {}]}]
    widget._refresh_table()
    cell = widget._model.item(0, 2)
    assert (
        cell.text() == "2 (not used)" and "distance to the examples" in cell.toolTip()
    )


def test_query_form_offers_non_examples_to_a_method_that_can_use_them(qapp):
    """The other half of the same rule, which was untestable while every registered method
    refused them: a method declaring `optional` support must actually enable the control.
    """
    widget = _query_form_with_method("soft_knn")
    assert widget._negatives_reason() is None


def test_query_form_says_when_a_seeded_runs_method_is_not_on_offer(qapp):
    """A seeded form must not silently substitute.

    `findData` returns -1 for a method §25 hides, and both `setCurrentIndex(-1)` and skipping
    leave the picker on the default while the row above says the form was seeded from that
    run — the form is then part one thing and part another with nothing on screen to say so,
    which is the failure `tests/gui/test_start_from.py` records. Qt makes this the easy bug:
    selecting a value a combo does not have is a no-op, not an error.

    `isHidden`, not `isVisible`: nothing here was ever shown, so `isVisible` is False for the
    notice whether it applies or not — the idiom that passes against its own bug.
    """
    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    widget.load_config({"method": "dtw", "track_id": 0, "behaviors": []})

    assert not widget._advice.isHidden()
    text = widget._advice.text()
    assert "dtw" in text
    assert "does not offer" in text
    # ...and it names what is selected instead, rather than only what is missing.
    assert widget._method.currentData() in text
    assert widget.config()["method"] == widget._method.currentData()


def test_the_seeded_method_note_clears_once_the_user_chooses(qapp, monkeypatch):
    """The note is about a substitution, so it must not outlive one. `_refresh_advice` rebuilds
    the notice from scratch on every table change, which is why the note is held as state
    rather than written once — and held state is what goes stale if nothing clears it.

    Two things this fixture has to arrange, both of which a comfortable one would hide. The
    picker needs **two** entries, or `setCurrentIndex` cannot be a change at all: Qt fires no
    signal for a no-op, so with one method the handler never runs and the test asserts the
    initial state. And the seeded method must be one that is missing from the picker under the
    escape hatch too, or hiding nothing means substituting nothing.

    Only one method is registered since `dtw` was removed, so a second id is registered for
    the length of the test.
    """
    from detail.core.registries import QUERY_METHODS
    from detail.surfaces import SHOW_ALL_ENV

    monkeypatch.setenv(SHOW_ALL_ENV, "1")
    monkeypatch.setitem(QUERY_METHODS, "soft_knn_twin", QUERY_METHODS["soft_knn"])
    widget = ANALYSIS_GUI_MODULES["query"].config_widget()
    assert widget._method.count() > 1, "the fixture cannot produce a change"

    widget.load_config({"method": "gone_entirely", "track_id": 0, "behaviors": []})
    assert not widget._advice.isHidden()

    starting = widget._method.currentIndex()
    widget._method.setCurrentIndex(1 - starting)  # the user picks for themselves
    assert widget._advice.isHidden()
    assert "gone_entirely" not in widget._advice.text()
