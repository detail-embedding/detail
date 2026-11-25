"""The app's widgets, discovered once — the subject list every structural requirement runs over.

**You cannot enumerate buttons.** There is no registry of them; you find them by walking the
things that contain them. So conformance is one discovery rather than seventeen: build the roots
(the pages the shell opens, and every dialog), walk everything beneath, and classify what you
find. Full coverage of the tool is then a property of the **roots** — miss one and everything
under it is invisible, which puts the entire risk in one place where it can be watched.

Both halves of the discovery already existed and are consolidated here rather than reinvented:

- pages, from ``test_smoke_construct._pages_the_shell_opens`` — an AST walk of ``app.py``'s
  ``_open_*`` / ``_go_to_*`` openers, because a page *is* what the shell shows;
- dialogs, from ``test_dialog_space._dialogs`` — ``QDialog.__subclasses__()`` walked
  transitively, so one added later is covered the day it exists.

**The floor is on the number actually built, never the number discovered.** That lesson is
`test_dialog_space`'s, learned the hard way: a floor on the discovered count lets coverage fall to
nothing one exemption at a time while the guard goes on passing.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import pathlib
from dataclasses import dataclass

import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSplitter,
    QTableView,
    QTableWidget,
    QWidget,
)

import detail.gui.app  # noqa: E402,F401  (imports the package's pages and dialogs)
import detail.gui.arena  # noqa: E402,F401
import detail.gui.behavior_query  # noqa: E402,F401
import detail.gui.comparison  # noqa: E402,F401
import detail.gui.fit  # noqa: E402,F401
import detail.gui.launch  # noqa: E402,F401
import detail.gui.pages  # noqa: E402,F401
import detail.gui.preview  # noqa: E402,F401
import detail.gui.streams  # noqa: E402,F401
from detail.api.experiment import Experiment
from detail.gui.image_view import ImageView  # noqa: E402
from detail.gui.scrubber import FrameScrubber  # noqa: E402
from detail.core.dataclasses import ComputeTarget, TrackingJob  # noqa: E402
from detail.core.registries import MODELS  # noqa: E402
from detail.transforms.constructors import CONSTRUCTORS  # noqa: E402
from detail.storage import analysis_runs as analysis_runs_store  # noqa: E402
from detail.gui import lifetime  # noqa: E402
from detail.storage import recording as recording_store  # noqa: E402
from detail.storage import runs as runs_store  # noqa: E402
from detail.storage.zarr_io import (  # noqa: E402
    DetectionPartWriter,
    EmbeddingPartWriter,
)
from detail.trackers.anchor_tracker import AnchorTracker  # noqa: E402

# --------------------------------------------------------------------------- #
# kinds — what a walked widget is, for the purpose of a requirement
# --------------------------------------------------------------------------- #
#: Ordered most-specific first: a `QTableWidget` is a `QTableView`, and a spin box is not a
#: plain field. The first match wins, so a widget has exactly one kind and a rule for "table"
#: cannot silently also run over something that merely inherits from one.
_KINDS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    # The dialog itself is a subject: `walk` includes its root, so a rule about a window's own
    # structure — one scroll, buttons outside it — has something to run on.
    ("dialog", QDialog),
    # Before the plain-widget kinds: a `FrameScrubber` is a `QWidget` that contains buttons and
    # a slider, so the video rules speak about the whole surface while its children go on being
    # walked as buttons and labels in their own right.
    ("video", FrameScrubber),
    # Before `label`: an `ImageView` *is* a `QLabel`, and first match wins — without this the
    # image rules would never run and the label rules would run over a picture.
    ("image", ImageView),
    ("number box", QAbstractSpinBox),
    ("dropdown", QComboBox),
    ("button", QPushButton),
    ("table", (QTableView, QTableWidget)),
    ("scroll area", QScrollArea),
    # Panes side by side (`gui/grip_splitter.py`): whether a pane can be dragged to nothing.
    ("splitter", QSplitter),
    # Every bar the page builds — a scroll area's own two included — so "one scrollbar look" is
    # checked on the bars a user meets, not on one built for the test.
    ("scrollbar", QScrollBar),
    ("label", QLabel),
)


def kind_of(widget: QWidget) -> str | None:
    """What kind ``widget`` is, or ``None`` if no requirement speaks about it yet."""
    for name, types in _KINDS:
        if isinstance(widget, types):
            return name
    return None


@dataclass(frozen=True)
class Subject:
    """One walked widget, and where it was found — so a failure names something findable."""

    kind: str
    widget: QWidget
    root: str
    #: "page" or "dialog" — which of the two routes to the screen this subject sits on. Some
    #: rules hold on one and not (yet) the other, and that is a fact about the app rather than an
    #: accident: a page is shown by the shell, a dialog shows itself.
    container: str = "page"

    @property
    def alive(self) -> bool:
        """Whether Qt still has this widget.

        A page may replace part of itself after it is built — `ExperimentRecordingsPage` puts up
        an empty table and swaps in the filled one when its off-thread read lands, `deleteLater`
        ing the first. The walk sees both; only the survivor is on screen, and touching the other
        raises ``RuntimeError: wrapped C/C++ object ... has been deleted`` rather than answering.
        Reuses `gui/lifetime.py`, which exists for exactly this and is what `busy.py` and
        `workers.py` already ask.
        """
        return lifetime.alive(self.widget)

    @property
    def label(self) -> str:
        """``RootClass.text`` — stable enough to put in an exemption list."""
        if not self.alive:
            return f"{self.root}.<deleted {type(self.widget).__name__}>"
        text = getattr(self.widget, "text", lambda: "")() or ""
        text = text.replace("\n", " ").strip()
        return (
            f"{self.root}.{text}"
            if text
            else f"{self.root}.<{type(self.widget).__name__}>"
        )


def walk(root: QWidget, root_name: str, container: str = "page") -> list[Subject]:
    """Every classifiable widget under ``root``, including ``root`` itself.

    The root of a page is classified as a **page**, which no type test can do: a dialog is a
    `QDialog` and a table is a `QTableView`, but a page is any `QWidget` the shell happens to
    show. Its kind comes from its position — it is the thing `walk` was handed — which is the
    only thing that distinguishes it. Without this there is nowhere to hang a rule about a
    page's *arrangement*, only about the widgets on it, and "the New… button is above the list"
    is a fact about neither button nor table alone.
    """
    found = []
    for widget in [root, *root.findChildren(QWidget)]:
        kind = kind_of(widget)
        if kind is None and widget is root and container == "page":
            kind = "page"
        if kind is not None:
            found.append(Subject(kind, widget, root_name, container))
    return found


# --------------------------------------------------------------------------- #
# roots — pages
# --------------------------------------------------------------------------- #
def pages_the_shell_opens() -> set[str]:
    """Every class the shell passes to ``MainWindow._show`` — the definition of "a page".

    Derived rather than listed, because a hand-written list held eleven of twenty-six once and
    nine of the missing ones needed no fixture at all; they were simply never added.

    **Keyed on ``_show``, not on a ``*Page`` name.** The obvious filter — a call whose name ends
    in "Page" inside an ``_open_*`` opener — misses ``DistillRunDetail``, a real page whose class
    name does not carry the suffix. Naming is a convention; being shown is the actual property,
    and `app.py` is where it is decided. A helper passed instead of a constructor (``_show`` also
    takes ``self._recordings_page()``) resolves to no class here and is covered by the page it
    returns.
    """
    source = pathlib.Path(detail.gui.app.__file__).read_text(encoding="utf-8-sig")
    named: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_show"
            and node.args
        ):
            continue
        first = node.args[0]
        if not isinstance(first, ast.Call):
            continue
        func = first.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name:
            named.add(name)
    return {name for name in named if _is_gui_class(name)}


@functools.cache
def _gui_class_names() -> frozenset[str]:
    """Every class name defined anywhere in ``detail.gui``, parsed once.

    Per-name it was a full re-parse of the package — ~130 files each time — which put 8.6s of the
    grid's 14.5s into two tests that answer the same question repeatedly. The grid is in
    `GUARD_FILES` and so runs at every stage of the gate, which is what makes this worth a cache
    rather than a shrug.
    """
    root = pathlib.Path(detail.gui.app.__file__).parent
    names = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names.update(
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        )
    return frozenset(names)


def _is_gui_class(name: str) -> bool:
    """Whether ``name`` resolves to a class defined in ``detail.gui`` — so a helper method
    passed to ``_show`` is not mistaken for a page class."""
    return name in _gui_class_names()


def _noop(*_args, **_kwargs) -> None:
    return None


def _active_runs_page():
    """The Active runs page, built the way the banner builds it."""
    from detail.gui.active_processes import ActiveProcessesPage

    return ActiveProcessesPage(on_open=_noop)


def _keypoints_manager():
    """Imported when a builder runs, after the caller has imported the GUI package."""
    from detail.gui import keypoints_manager

    return keypoints_manager


def _example_set_picker():
    from detail.gui import example_set_picker

    return example_set_picker


def _with_view_open(page):
    """``page`` (an analysis run page) with its first view's card clicked."""
    next(iter(page._cards.values())).click()
    assert page.open_view(), "the run page built no view to open"
    return page


def page_builders(pages_mod, comparison_mod) -> dict:
    """``{class name: build(experiment)}`` for the pages that build from a bare experiment.

    Passed the modules rather than importing them at call time so the caller controls when the
    GUI package is imported (some pages probe the machine at construction).
    """
    return {
        "HomePage": lambda p: pages_mod.HomePage(_noop),
        "ExperimentOverviewPage": lambda p: pages_mod.ExperimentOverviewPage(p, _noop),
        "ExperimentRecordingsPage": lambda p: pages_mod.ExperimentRecordingsPage(
            p, _noop
        ),
        "MetadataPage": lambda p: pages_mod.MetadataPage(p),
        "ExperimentRunsPage": lambda p: pages_mod.ExperimentRunsPage(p, _noop),
        "GlobalModelsPage": lambda p: pages_mod.GlobalModelsPage(),
        "DatasetsPage": lambda p: pages_mod.DatasetsPage(p, _noop),
        "ModelsPage": lambda p: pages_mod.ModelsPage(p, _noop),
        "KeypointsManagerPage": lambda p: _keypoints_manager().KeypointsManagerPage(),
        "ExampleSetPicker": lambda p: _example_set_picker().ExampleSetPicker(p),
        "ExperimentAnalysisPage": lambda p: pages_mod.ExperimentAnalysisPage(p, _noop),
        "RecordingDetailPage": lambda p: pages_mod.RecordingDetailPage(
            p.get_recording("recording_001"), experiment=p
        ),
        "ComparisonsPage": lambda p: comparison_mod.ComparisonsPage(p, _noop),
        "ModelManagerPage": lambda p: pages_mod.ModelManagerPage(
            p, _noop, _noop, _noop
        ),
        "ToolRunPage": lambda p: pages_mod.ToolRunPage("quick_segment", _noop),
        # Reached from `_switch_mode` rather than an `_open_*` opener, which is why the older
        # `*Page`-suffix-inside-an-opener derivation never saw them — two banner landings, one
        # of which is the first screen a new user meets.
        "ConfigurationPage": lambda p: pages_mod.ConfigurationPage(_noop),
        "ToolsPage": lambda p: pages_mod.ToolsPage(_noop),
        # Reached from the banner like the two above. It scans every experiment off the UI
        # thread, so the walk sees it mid-load — which is the state a user sees too, and the
        # state its empty-state notice must not lie about.
        "ActiveProcessesPage": lambda p: _active_runs_page(),
        # The Configuration drill-ins and Docs. They were excused as "probes the machine", which
        # is what `test_smoke_construct` says about *where* they are tested rather than about
        # whether they can be built — and each takes no arguments and builds in milliseconds.
        # An excuse inherited from a neighbouring list is still an excuse nobody checked.
        "GpuCudaPage": lambda p: pages_mod.GpuCudaPage(),
        "Sam3BackendPage": lambda p: pages_mod.Sam3BackendPage(),
        "HuggingFacePage": lambda p: pages_mod.HuggingFacePage(),
        "DocumentationPage": lambda p: pages_mod.DocumentationPage(),
        # The not-found path is a real screen a user reaches, and the one this walk most wants:
        # an empty state still has to obey every rule about how it says it is empty.
        "GlobalModelDetailPage": lambda p: pages_mod.GlobalModelDetailPage("nope"),
        # The detail pages. They are where most of the app's tables and buttons live, and the
        # walk could not reach any of them: each needs a record to open. It was never a hard
        # problem — every one of them was *already* built by some test, with the setup written
        # once per test file and shared with nothing. `an_experiment_with_completed_runs` is
        # that setup, in one place, and it costs 0.24s.
        "RunDetailPage": lambda p: pages_mod.RunDetailPage(p, "run_001"),
        "RecordingRunDetailPage": lambda p: pages_mod.RecordingRunDetailPage(
            p, "recording_001", "detection", "det_01"
        ),
        "AnalysisRunDetailPage": lambda p: pages_mod.AnalysisRunDetailPage(p, "an_01"),
        # The same page with its one view open full window: the view host and its row of view
        # tabs are a second screen of this page, and a walk of the landing page alone never
        # meets them.
        "AnalysisRunDetailPage.view": lambda p: _with_view_open(
            pages_mod.AnalysisRunDetailPage(p, "an_01")
        ),
        # These three open their **not-found** state, which is a screen a user really reaches
        # (a deleted run, a stale link) and which no walk covered before. It is not the same
        # screen as the populated one, and saying so here is the point: a fixture that never
        # diverges cannot tell a working page from a broken one.
        "DatasetDetailPage": lambda p: pages_mod.DatasetDetailPage(p, "dataset_001"),
        "ModelDetailPage": lambda p: pages_mod.ModelDetailPage(p, "model_001"),
        "DistillRunDetail": lambda p: pages_mod.DistillRunDetail(p, "distill_001"),
        "ExportRunDetailPage": lambda p: pages_mod.ExportRunDetailPage(p, "export_001"),
        "ToolRunDetailPage": lambda p: pages_mod.ToolRunDetailPage(
            "quick_segment", "tr_01", _noop
        ),
    }


def a_clustering_workspace(experiment_dir, fit_template, qapp):
    """The clustering run's workspace — INTERPRET's label field, filters, tree, key and montage —
    on a copy of the session's temporal fit, loaded and with a filter on, so the walk meets the
    fields in use with their Reset, the shaded key and the montage's tiles rather than an empty
    page.

    The filter bar, live montage and match key were walked on their own while no page embedded
    them; this page is their root now.
    """
    import shutil

    from detail.api.cluster_filter import Selection
    from detail.gui.cluster_workspace import ClusterWorkspace
    from tests.gui.conftest import drain_until

    template, run_id = fit_template
    shutil.copytree(template, experiment_dir, dirs_exist_ok=True)
    experiment = Experiment(experiment_dir)
    handle = experiment.authored_run(run_id)
    workspace = ClusterWorkspace(experiment, handle)
    interpret = workspace._interpret
    drain_until(qapp, lambda: interpret._clips is not None, 20_000)
    view = interpret._view
    interpret.filters().set_selection(
        Selection(recording_ids=frozenset(view.recording_ids[:1]))
    )
    # A cluster's pop-out window is the page's child: walked with it, over the real fit.
    interpret.open_window(interpret._selected)
    return workspace


#: Pages the shell opens that this walk does not build, and why. Detail pages need a real record;
#: the Configuration drill-ins and Docs probe the *machine* (a CUDA check, a backend import, a
#: docs build) and building them here once aborted the whole gui tier on CI with a core dump.
#: Each is a claim, checked by the stale-entry test in `test_requirements.py`.
PAGES_NEEDING_CONTEXT = {
    "ComparisonDetailPage": (
        "needs a cohort of completed runs to compare — `test_comparison_gui`: its content is "
        "the comparison, so an empty one walks nothing"
    ),
}


# --------------------------------------------------------------------------- #
# roots — dialogs
# --------------------------------------------------------------------------- #
def _import_every_gui_module() -> None:
    """Execute every module under ``detail/gui`` so its classes exist.

    ``QDialog.__subclasses__()`` only knows about classes Python has actually run. This package
    breaks import cycles by importing dialogs **lazily, inside the handler that opens them** —
    a real convention here, not an accident — so such a dialog does not exist until somebody
    clicks, and every guard that discovers dialogs passes over it by never seeing it. It is the
    quietest way for a guard to be green: not a wrong assertion, an absent subject.

    Caught when `ClipPopup` was added and `test_dialog_space` / `test_smoke_construct` /
    the requirements grid all stayed green over a dialog none of them had ever built.

    Cheap, and already known safe: `tests/gui/test_import_cycles.py` asserts every one of these
    imports standalone.
    """
    import importlib

    package = pathlib.Path(detail.gui.app.__file__).parent
    root = package.parents[1]
    for path in sorted(package.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        importlib.import_module(module.removesuffix(".__init__"))


def dialog_classes() -> list[type]:
    """Every ``QDialog`` subclass reachable from ``detail.gui``, transitively."""
    _import_every_gui_module()
    seen, out = set(), []
    stack = list(QDialog.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        # `gui.dialog.Dialog` is the shared base every dialog inherits, not a dialog of its own:
        # it holds no widgets, so walking it contributes nothing and its emptiness is not a bug.
        if (
            getattr(cls, "__module__", "").startswith("detail.gui")
            and cls.__name__ != "Dialog"
        ):
            out.append(cls)
    return sorted(out, key=lambda c: c.__name__)


def an_example_set(experiment) -> str:
    """A keypoint example set (§26) in ``experiment``, made once and reused: what the labeling
    screen opens on."""
    existing = experiment.example_sets()
    if existing:
        return existing[0].id
    return experiment.create_example_set("examples", "midline_2").id


def an_imported_stream(recording) -> str:
    """A stream with a canonical CSV, so the two stream dialogs have one to show."""
    from detail.storage import streams as streams_store

    name = "lever_press"
    listed = [m.get("name") for m in streams_store.list_streams(recording.dir)]
    if name in listed:
        return name
    streams_store.write_canonical_csv(
        streams_store.stream_dir(recording.dir, name) / streams_store.CANONICAL_DIRNAME,
        ["t_ns", "value"],
        [[0, 1.0], [1_000_000, 0.0]],
    )
    streams_store.append_stream_to_recording_yaml(
        recording.dir,
        {
            "schema_version": 1,
            "name": name,
            "role": "behavior",
            "columns": ["t_ns", "value"],
            "parts": ["part_001.csv"],
        },
    )
    return name


def a_saved_comparison(experiment):
    """A real §24 comparison on disk, so the export dialog is handed one rather than a stand-in.

    The only helper here that runs a real pipeline, and deliberately: this dialog used to be
    *built* by passing an `Experiment` positionally, which is the wrong object and did not raise,
    so it counted as covered while nothing about a comparison was ever exercised. A stub with a
    `.dir` would reproduce that mistake in a politer form.

    Writes the §19 behavior artifact the comparison measures, two metadata values to group by,
    and runs it. Two recordings in one group and one in the other: a cohort where every group has
    one member cannot produce a contrast, and several statistics are withheld on exactly that.
    """
    import numpy as np

    from detail.api.comparison import run_comparison
    from detail.storage import analysis_runs as ar
    from detail.storage import behavior_output as bo
    from detail.storage import frame_view as fv
    from detail.storage import metadata as md

    existing = experiment.comparisons
    if existing:
        return existing[-1]

    period, n_frames = 10_000_000, 100
    axis = (np.arange(n_frames) * period).astype(np.int64)
    plan = {"recording_001": ("veh", 2), "cmp_002": ("veh", 4), "cmp_003": ("fent", 6)}

    run_id = "cmp_src_01"
    run_dir = ar.analysis_dir(experiment.dir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    instances, coverage = [], []
    for recording_id, (drug, n_bouts) in plan.items():
        if recording_id not in experiment.recording_names():
            experiment.create_recording(recording_id)
        md.upsert_row(experiment.dir, recording_id, {"drug": drug})
        for i in range(n_bouts):
            instances.append(
                bo.BehaviorInstance(
                    recording_id,
                    0,
                    int(axis[i * 3]),
                    int(axis[i * 3 + 1]),
                    i * 3,
                    i * 3 + 1,
                    "q00",
                    "rearing",
                )
            )
        coverage.append(
            bo.CoverageRow(
                recording_id, 0, 0, n_frames * period, n_frames * period, n_frames
            )
        )
    bo.write_behavior_output(
        run_dir,
        instances=instances,
        coverage=coverage,
        manifest={
            "producer": {"module_id": "test", "run_id": run_id},
            "origin": "predicted",
            "label_space": "pooled",
            "vocabulary": ["rearing"],
            "names": {"q00": ["rearing"]},
            "overlap": "exclusive",
            "behavior_types": {"rearing": "state"},
        },
    )
    for recording_id in plan:
        fv.write_frame_view(
            run_dir,
            recording_id=recording_id,
            subject=0,
            t_ns=axis,
            observed=np.ones(n_frames, bool),
            instances=[i for i in instances if i.recording_id == recording_id],
        )
    return run_comparison(
        experiment,
        {
            "source_run_id": run_id,
            "group_by": ["drug"],
            "aggregate_to": [],
            "measures": ["n_instances"],
        },
    )


def a_model_catalog_entry():
    """One imported-model record — what the model manager's Edit action opens on."""
    from detail.api.models import ImportedModelRef

    return ImportedModelRef(
        id="mdl_01",
        name="a model",
        model_family="yolo",
        classes=["mouse"],
        made_by="a person",
        description="a description",
        notes="notes",
    )


def a_clip_query_config() -> dict:
    """A query config with one named query. `_normalize_queries` fills the rest from DEFAULTS,
    so this is the minimum a config has to carry rather than a copy of the defaults."""
    return {"queries": [{"name": "rearing", "examples": []}]}


def clip_csv_rows() -> list[dict]:
    """Parsed clip-CSV rows, one usable and one not.

    **Both**, because the dialog's whole job is telling them apart: a fixture of all-good rows
    never renders a verdict, never leaves a box unchecked, and would pass with the check gutted.
    """
    return [
        {
            "ok": True,
            "query_name": "rearing",
            "recording_id": "recording_001",
            "track_id": 0,
            "start_t_ns": 0,
            "end_t_ns": 1_000_000,
            "reason": "",
        },
        {
            "ok": False,
            "query_name": "rearing",
            "recording_id": "not_a_recording",
            "track_id": 0,
            "start_t_ns": None,
            "end_t_ns": None,
            "reason": "no such recording in this experiment",
        },
    ]


def _a_clip(recording):
    """One short clip of ``recording``'s first track, in the shape the players take.

    Built here rather than imported from a test, so the walk owns its own fixtures and a dialog
    that wants a clip stops being exempt.
    """
    from detail.gui.behavior_query import ClipSpec

    return ClipSpec(
        recording.id, 0, "emb_01", 0, 200_000_000, f"{recording.id} 0.0–0.2s"
    )


def _an_advanced_player(experiment, recording):
    """The clip window's Advanced player with the fixture's clip on it — what an enlarged pane
    follows. Not playing: the walk looks at widgets, not at frames arriving."""
    from detail.gui.advanced_player import AdvancedClipPlayer

    player = AdvancedClipPlayer(experiment, context_s=1, autoplay=False)
    player.set_clip(_a_clip(recording), autoplay=False)
    return player


def _providers(experiment, recording) -> dict:
    """What each constructor argument is filled with, keyed by the parameter's **name**.

    A table rather than a chain of ``if params[0] == ...``, because the chain could only ever
    answer for a dialog's *first* argument and nineteen dialogs take two or more — they were all
    exempted as "needs a fitted clusterer", "needs an imported stream", when what they needed was
    a second argument the dispatcher had no way to supply.

    Keyed on the name because that is what says which object is wanted: passing an `Experiment`
    positionally to everything once built dialogs holding the wrong object *without raising*,
    which is worse than failing, because the exemption list then looks justified.

    Lazy values, so an expensive fixture is only paid for by a dialog that asks for it.
    """
    from detail.api import exporters
    from detail.api.exporters import ExportSubject

    def export_subject():
        return ExportSubject("experiment", experiment.dir)

    return {
        "experiment": lambda: experiment,
        "recording": lambda: recording,
        "recordings": lambda: [recording],
        # Locate video's parts come from the page's off-thread scan.
        "locatable": lambda: recording.parts_to_locate(),
        "stream_name": lambda: an_imported_stream(recording),
        # A dialog that reports back through a callback needs one that does nothing; what it
        # reports is the caller's business, not this walk's.
        "on_submit": lambda: _noop,
        # `full_frame.FullFrameDialog` renders on demand rather than being handed a picture, so
        # what it needs is a callable and a name for the window. A real ndarray, not a stub:
        # the dialog puts it on screen, and a walk over a dialog showing nothing checks nothing.
        "render": lambda: (lambda: np.full((90, 160, 3), 40, dtype=np.uint8)),
        "title": lambda: "A frame",
        "subject": export_subject,
        "targets": lambda: exporters.targets_for(export_subject()),
        "scope": lambda: None,
        # The fixture's synthesized embedding run — what the input-image and clip dialogs
        # scrub through.
        "embed_run_id": lambda: "emb_01",
        # One clip of the fixture's own recording, for the window a strip's thumbnail opens
        # into. A provider rather than an exemption: the dialog is then really built and really
        # checked, which is the difference the ratchet on NEEDS_CONTEXT exists to push towards.
        "clip": lambda: _a_clip(recording),
        # The context a clip window opens with. A second, not "off": the padded window is the
        # shape with a span drawn on its bar, which is the one a walk should look at.
        "context_s": lambda: 1,
        # The enlarged pane window follows a player and shows one of its two panes; the GIF
        # dialog says which frames it will write.
        "player": lambda: _an_advanced_player(experiment, recording),
        "pane": lambda: "masks",
        "frames_note": lambda: "The clip itself, 5 frames.",
        # A **mapping** keyed by recording id, not a list: the run differs per recording
        # (§6.5), which is the whole reason the dialog takes it separately.
        "run_ids": lambda: {recording.id: "emb_01"},
        "track_ids": lambda: [0],
        "constructor": lambda: next(iter(CONSTRUCTORS)),
        # A prompt-test dialog is built from what the launch form had typed in it. None of it
        # has to be loadable: the dialog under test is the form, and its model load is a button
        # press this walk never makes.
        "model_id": lambda: next(iter(MODELS)),
        "prompt_text": lambda: "mouse",
        "compute": lambda: ComputeTarget("cpu", None, None, None),
        # The rest. Every name below is unique to one dialog — checked, because this table
        # is keyed globally and a name two dialogs share would hand one of them the other's
        # object. That is the failure `ComparisonExportDialog` sat in for months.
        "comparison": lambda: a_saved_comparison(experiment),
        "ref": a_model_catalog_entry,
        "config": a_clip_query_config,
        "results": clip_csv_rows,
        "query": lambda: {"name": "rearing", "fusion": "min", "band_ns": 0},
        "queries": lambda: [{"name": "rearing", "examples": [], "matches": []}],
        # The keypoint labeling screen (§26) opens on an example set; one is made in the
        # fixture experiment the first time a dialog asks.
        "set_id": lambda: an_example_set(experiment),
        # The skeleton editor (§26 Keypoints manager) opens on one library entry, with what
        # uses it already read by the page: a preset that something uses is the shape with the
        # most on screen (Save as new, the used-by line).
        "skeleton_entry": lambda: _a_skeleton_entry(),
        "skeleton_dependents": lambda: [{"kind": "example set", "experiment": "lab"}],
        "taken_skeleton_names": lambda: set(),
        # *Test cut-off…* (§12) tests the teacher it is handed, off the UI thread, as it opens.
        # A detect run the fixture does not have: the test refuses at once, so the walk sees the
        # window in its refused state rather than racing a real model load.
        "teacher": lambda: {"detect_runs": {recording.id: "kd_99"}},
        # *Test keypoints* on the detect form (§26 PR 4b): the Recordings choice's names.
        "recording_names": lambda: {recording.id: recording.name},
        # The New example set window (§26): the library, the picker's skeleton, a name.
        "skeleton_entries": lambda: _skeleton_entries(),
        "chosen_skeleton_id": lambda: "midline_2",
        "default_set_name": lambda: "example set 1",
        # A cluster's pop-out window (`cluster_window.ClusterWindow`) is opened by INTERPRET and
        # reads its labels and write queue. A page with no run loaded: the window opens on its
        # empty state with the label row greyed, which is what this walk checks the structure
        # of. The populated window, over a real fit, is `test_cluster_window.py`'s and the page
        # walk's (`a_clustering_workspace` opens one).
        "interpret": lambda: _an_interpret_page_with_nothing_loaded(experiment),
        "node": lambda: 0,
    }


def _an_interpret_page_with_nothing_loaded(experiment):
    """What a cluster window reads of INTERPRET, before the page has loaded its run."""
    from types import SimpleNamespace

    from PyQt6.QtCore import QObject
    from PyQt6.QtGui import QColor

    from detail.gui.cluster_workspace import POSE_TERMS
    from detail.gui.label_panel import LabelBar

    def nothing(*_args, **_kwargs):
        return None

    host = SimpleNamespace(
        _terms=POSE_TERMS,
        _experiment=experiment,
        _run=SimpleNamespace(id="an_01"),
        _view=None,
        _tracking={},
        _embeds={},
        _taken={},
        _names={},
        _selected=None,
        _can_edit=False,
        _align_note="This run is still loading.",
        _points_note="This run is still loading.",
        _label=nothing,
        _rename_label=nothing,
        _merge_label=nothing,
        _select_label=nothing,
        _label_field_state=lambda node: ("", "", [], []),
        _owner=lambda node: None,
        _colour_of=lambda node: QColor("grey"),
        _colour_of_root=lambda root: QColor("grey"),
        _label_rows=lambda: [],
        _quick_labels=lambda rows: [],
        _edit_reason=lambda: "The run is still loading.",
        _cluster_name=lambda node: f"Cluster #{int(node)}",
        # The page's own bar: a window shows its status line, kept in step by its signal.
        _label_bar=LabelBar(POSE_TERMS),
        _windows=SimpleNamespace(forget=nothing),
        undo=nothing,
        redo=nothing,
        _keep=QObject(),
    )
    return host


def _skeleton_entries() -> list[dict]:
    from detail.api import skeletons

    return skeletons.skeletons()


def _a_skeleton_entry() -> dict:
    from detail.api import skeletons

    return skeletons.skeleton("midline_2")


#: Dialogs :func:`build_dialog` cannot construct, and what each is missing. An entry is a
#: claim, so it names the argument rather than waving at "context" — and
#: `test_dialog_space.test_no_stale_dialog_exemptions` fails on one the builder *can* now build,
#: which is how this list stopped being right the first time: nine entries sat here describing
#: dialogs that had become buildable, and nothing said so.
#:
#: **It is empty, and that is the point.** Every entry it held turned out to be a dialog whose
#: argument was a plain dict, a list of rows, or one real artifact — and six of the seven were
#: *not compliant*, because a dialog no sweep can build is a dialog no rule reaches. The
#: exemption was not recording a hard fixture; it was hiding the bug.
DIALOGS_NEEDING_CONTEXT: dict[str, str] = {}


def buildable_dialogs() -> list[type]:
    """The dialogs a sweep actually runs over: discovered, minus the ones with no fixture.

    One derivation, not one per sweep. `test_dialog_space` and `test_requirements` each wrote
    ``[c for c in dialog_classes() if c.__name__ not in DIALOGS_NEEDING_CONTEXT]``, which is the
    same list comprehension twice — and two copies of "which dialogs are covered" is how the
    exemption list and the builder drifted in the first place. A third sweep gets it right by
    calling this rather than by being written correctly.
    """
    return [c for c in dialog_classes() if c.__name__ not in DIALOGS_NEEDING_CONTEXT]


def build_dialog(cls, experiment, recording):
    """Construct ``cls`` by filling every required argument from :func:`_providers`.

    Returns ``None`` when some required argument has no provider — that dialog genuinely needs
    context this walk cannot make, and belongs in ``NEEDS_CONTEXT`` with a reason. Anything whose
    arguments are all named things we can supply is built, which is the point: the exemption list
    should hold only the dialogs that really need a fixture nobody has written.
    """
    parameters = list(inspect.signature(cls.__init__).parameters.items())[1:]
    providers = _providers(experiment, recording)
    args, kwargs = [], {}
    positionals_done = False
    for name, parameter in parameters:
        # A keyword-only argument is passed by name, so a defaulted one does not make the ones
        # after it optional the way a defaulted positional does — `break` on the first would
        # have skipped a *required* one and built nothing. `FullFrameDialog(render, *, title)`
        # was the first dialog to have one; the walk reported it as needing context.
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if parameter.default is not inspect.Parameter.empty:
                continue
            if name not in providers:
                return None
            kwargs[name] = providers[name]()
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Not `break`: the positionals after a defaulted one are optional, but a keyword-only
        # argument after them can still be required. `ClipPopup(experiment, clip, parent=None,
        # *, context_s)` was the first dialog with that shape, and a `break` here built it
        # without `context_s` and raised.
        if positionals_done or parameter.default is not inspect.Parameter.empty:
            positionals_done = True
            continue
        if name not in providers:
            return None
        args.append(providers[name]())
    return cls(*args, **kwargs)


def an_experiment(tmp_path) -> Experiment:
    """The experiment every buildable root is constructed from.

    A strict superset of the bare one it replaced — same `recording_001`, plus a completed run of
    every module and one completed analysis. That is what lets the walk open a **detail page**,
    which is where most of the app's tables and buttons are, and it is why the walk went from 18
    roots to 29.

    Richer is not free of risk: a page built over data behaves differently from one built over
    nothing, and both are real. The pages whose record this does *not* create are built on their
    not-found path deliberately, and `page_builders` says which those are.
    """
    return an_experiment_with_completed_runs(tmp_path)


# --------------------------------------------------------------------------- #
# roots — the package itself
# --------------------------------------------------------------------------- #
GUI_PACKAGE = pathlib.Path(detail.gui.app.__file__).parent


@dataclass(frozen=True)
class PackageSubject:
    """The `detail/gui/` source tree, as one subject the grid can run rules over.

    Four house rules are about what the package **never does** — no stylesheets, no icon sets, no
    animations — and one about what it always does. None of them is a property of a widget you can
    walk to: an icon set on a `QListWidgetItem` is invisible to a walk over `QWidget`s, and a rule
    of the form "this appears nowhere" cannot be checked by looking at what happens to be built.
    So the subject is the source.

    It answers the same three questions a walked widget does — kind, liveness, label — so the grid
    stays one loop and a source rule sits in the same list as every other rule, which is the whole
    point of having a list.
    """

    kind: str = "package"
    widget: pathlib.Path = GUI_PACKAGE
    root: str = "detail.gui"
    container: str = "package"

    @property
    def alive(self) -> bool:
        """Source does not get deleted out from under the grid the way a Qt object does."""
        return True

    @property
    def label(self) -> str:
        return self.root


_FRAMES = (0, 2, 4)  # frames with a detection (video has 6)


def _tiny_video(path, n=6, w=32, h=24):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 30) % 255, np.uint8))
    vw.release()
    return path


def an_experiment_with_completed_runs(tmp_path):
    """A experiment whose single recording has a completed run for every module, with
    real outputs on disk + the experiment-level run records RunDetailPage reads."""
    proj = Experiment.create("p", tmp_path)
    sess = proj.create_recording("recording_001")
    sess.add_video_part(_tiny_video(tmp_path / "clip.mp4"))
    sdir = sess.dir

    # detection det_01 (synthesized, on the original video; one box per frame)
    det_dir = sdir / "detections" / "det_01"
    with DetectionPartWriter(det_dir / "part_001.zarr") as w:
        for f in _FRAMES:
            w.add_row(f, f * 1000, 0, "mouse", 0.9, (0.1, 0.1, 0.4, 0.4), "")
    # The visible trackers read the mask grid from the detection run's envelope.
    (det_dir / "job.json").write_text(
        json.dumps({"schema_version": 1, "mask_resolution": [24, 32]}), encoding="utf-8"
    )
    _append_run(
        sdir,
        "detection_runs",
        {
            "id": "det_01",
            "status": "complete",
            "output": "detections/det_01/",
            "model_id": "sam3",
        },
    )

    # tracking track_01 (real AnchorTracker over det_01)
    AnchorTracker().track(
        TrackingJob(
            job_id="track_01",
            recording_id="recording_001",
            track_id="track_01",
            detection_run_id="det_01",
            tracker_id="anchor_tracker",
            hyperparams={},
            arena_ids=[],
            created_at="2026-01-01T00:00:00Z",
        ),
        det_dir,
        sdir / "tracking" / "track_01",
    )
    _append_run(
        sdir,
        "tracking_runs",
        {
            "id": "track_01",
            "status": "complete",
            "output": "tracking/track_01/",
            "detection_run_id": "det_01",
            "tracker_id": "anchor_tracker",
            "arena_ids": [],
        },
    )

    # embedding emb_01 (synthesized: one track, a few rows)
    emb_dir = sdir / "embeddings" / "emb_01"
    rng = np.random.default_rng(0)
    with EmbeddingPartWriter(emb_dir / "track_0" / "part_001.zarr", output_dim=8) as w:
        for di, f in enumerate(_FRAMES):
            w.add_row(f, f * 1000, di, rng.standard_normal(8).astype(np.float32))
    (emb_dir / "embedding.yaml").write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "id": "emb_01",
                "tracking_run_id": "track_01",
                "model_id": "dinov3_convnext_base",
                "model_variant": None,
                "constructor": "full_frame",
                "output_dim": 8,
                "track_ids": [0],
                "renderer_spec": [{"type": "init_binary_mask"}],
                "status": "complete",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _append_run(
        sdir,
        "embedding_runs",
        {
            "id": "emb_01",
            "status": "complete",
            "output": "embeddings/emb_01/",
            "tracking_run_id": "track_01",
        },
    )

    # experiment-level run records — one per module (what RunDetailPage reads)
    for i, (module_id, rid) in enumerate(
        [
            ("detection", "det_01"),
            ("tracking", "track_01"),
            ("embedding", "emb_01"),
        ],
        start=1,
    ):
        runs_store.write_run(
            proj.dir,
            f"run_{i:03d}",
            {
                "id": f"run_{i:03d}",
                "module_id": module_id,
                "recording_ids": ["recording_001"],
                "per_recording": {
                    "recording_001": {"run_id": rid, "status": "complete"}
                },
                "config": {},
                "status": "complete",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    # A completed analysis run, so `AnalysisRunDetailPage` opens on its *result* rather than on
    # the not-found path. Which state a page is walked in matters: the empty and populated
    # screens are different screens, and the rules about how a table cues a clickable row have
    # nothing to run over when the table has no rows.
    # Its tables are real, so its Plots view draws a plot rather than an error.
    from tests.location_fixtures import track, write_run

    run_dir = analysis_runs_store.analysis_dir(proj.dir, "an_01")
    write_run(
        run_dir,
        {"recording_001": {0: track(10.0 + np.arange(50) * 4.0, np.full(50, 60.0))}},
    )
    analysis_runs_store.write_analysis_yaml(
        run_dir,
        {
            "id": "an_01",
            "module_id": "location_export",
            "recording_ids": ["recording_001"],
            "tracking_run_id": "track_01",
            "status": "complete",
            "per_recording": {"recording_001": {"status": "complete"}},
            "params": {"source_recording_ids": ["recording_001"]},
            "artifacts": [
                {"path": "recording_001/tracks.csv", "kind": "tracks", "label": "x"}
            ],
        },
    )
    return proj


def _append_run(recording_dir, key, entry):
    data = recording_store.read_recording(recording_dir)
    data.setdefault(key, []).append(entry)
    recording_store.write_recording(recording_dir, data)
