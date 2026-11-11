"""The experiment's sections, declared once (§25).

The rail, the overview dashboard and the navigation dispatcher all need the same list of
sections, and before this module each declared its own. They had already drifted: the rail
called one section "Processing pipeline" and the overview called the same ``kind`` "Runs", so a
user met two names for one thing depending on which door they came through.

Declaring them here is also what makes the §25 deploy toggle testable. A surface's visibility is
checked where it **enumerates** what exists, and a guard can only be data-driven over a list —
not over a sequence of hardcoded ``_item(...)`` calls. Filtering happens at each enumeration
point via ``surfaces.is_surfaced``; this module holds no policy, only the declaration.

Qt-free on purpose: the label and the child enumeration are plain data, so a test can walk every
section without a QApplication.

**The child enumerations run off the UI thread, and only for a section that is open.** Each is
O(recordings) or O(runs) small-file reads — individually trivial, seconds in total on a
workspace held on a network share — so ``child_items`` has exactly one caller,
``MainWindow._fill_section``, which runs it in a ``BackgroundCall`` when its folder opens
(#316). That is what the ``# off-thread`` markers below assert; calling one of these on the UI
thread is the bug, and ``tests/gui/test_ui_thread_blocking.py`` is where it would be caught.

``kind`` is the stable identifier — it is the nav key the dispatcher already switches on, and it
outlives the display title. Renaming a section's ``title`` is a user-visible change and nothing
else; renaming its ``kind`` changes the toggle id and the nav contract.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from detail import surfaces


@dataclass(frozen=True)
class Section:
    """One experiment section: its stable id, its label, and the rail children beneath it."""

    kind: str
    title: str
    child_kind: str | None = None
    children: Callable[[object], Sequence[tuple[str, object]]] | None = None

    def child_items(self, experiment) -> list[tuple[str, object]]:
        """``(label, child_id)`` pairs for the rail, or empty for a leaf section."""
        return [] if self.children is None else list(self.children(experiment))


def _recordings(experiment):
    # The name is the handle (§11) — the id is the tool's, opaque, and shown on the
    # recording's own page for when you need to quote it.
    return [
        (r.name, (experiment, r.id))
        for r in experiment.recordings  # off-thread (see the docstring)
    ]


def _runs(experiment):
    # Chronological, like every sibling section: the store already orders runs by their
    # monotonic id. Re-sorting by module_id groups them alphabetically, so the rail
    # disagrees with the Runs table on the same folder page.
    return [
        (f"{rec.get('module_id')}  ·  {rec['id']}", (experiment, rec["id"]))
        for rec in experiment.processing_runs  # off-thread (see the docstring)
    ]


def _analysis_runs(experiment):
    return [
        (f"{ref.module_id}  ·  {ref.id}", (experiment, ref.id))
        for ref in experiment.analysis_runs  # off-thread (see the docstring)
    ]


def _datasets(experiment):
    return [
        (f"{ref.name or ref.id}  ·  {ref.id}", (experiment, ref.id))
        for ref in experiment.datasets  # off-thread (see the docstring)
    ]


def _models(experiment):
    return [
        (f"{ref.name or ref.id}  ·  {ref.id}", (experiment, ref.id))
        for ref in experiment.models  # off-thread (see the docstring)
    ]


def _comparisons(experiment):
    return [(c.id, (experiment, c.id)) for c in experiment.comparisons]


EXPERIMENT_SECTIONS: tuple[Section, ...] = (
    Section("metadata", "Recording information"),
    Section("recordings", "Recordings", "recording", _recordings),
    # Before the pipeline, in the order the work is done: a dataset and a trained model are
    # what a detection run is given. They were drill-ins off the Model manager until its
    # rework dropped the Trained-models link, leaving the page reachable only by deleting a
    # model — a rail node is a door no page rework can take away.
    Section("datasets", "Datasets", "dataset", _datasets),
    Section("models", "Models", "model", _models),
    Section("runs", "Processing pipeline", "run", _runs),
    Section("analysis", "Analysis", "analysis_run", _analysis_runs),
    Section("comparisons", "Comparisons", "comparison", _comparisons),
)

SECTION_KINDS: frozenset[str] = frozenset(s.kind for s in EXPERIMENT_SECTIONS)


def sections():
    """The sections this build shows, in declared order (§25)."""
    return tuple(s for s in EXPERIMENT_SECTIONS if surfaces.is_surfaced(s.kind))
