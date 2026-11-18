"""The gui-side module-UI registry (docs/gui_design.md §3.2).

Each processing module contributes a bespoke UI — a config form for the launch
dialog and a result view for run detail — registered by ``module_id`` and paired
with the backend ``PROCESSING_MODULES`` entry. The shell stays generic: it
enumerates this registry, hosts the config form, launches via
``Experiment.run_processing``, and hosts the result view. Adding a module's UI is a
file here + one ``register_module_ui`` call — no shell change.
"""

from abc import ABC, abstractmethod

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from detail.configs.describe import describe_config, module_display_name
from detail.gui._pydantic_form import set_any_field_value

# The result-view protocol is part of this registry's contract and is re-exported from here; it
# lives in its own module only because a workspace module imports it while this package is still
# importing the module UIs that import that workspace.
from detail.gui.result_views import (  # noqa: F401 — re-exported
    DEFAULT_PURPOSE,
    RESULTS_VIEW,
    MultiViewWorkspace,
    ResultView,
    result_views,
)


class ConfigWidget(QWidget):
    """A module's run-config form. Subclasses build the form and expose the
    ``config`` dict passed to ``Experiment.run_processing`` plus a validation hook."""

    # Emitted when the form's choice of *upstream run* changes, so the launch dialog can
    # re-mark which recordings that upstream actually covers. A form whose config has no
    # upstream picker never emits it.
    coverage_changed = pyqtSignal()

    def config(self) -> dict:
        raise NotImplementedError

    def validation_error(self) -> str | None:
        """A user-facing message if the form is invalid, else None."""
        return None

    def on_recordings_changed(self, recordings: list) -> None:
        """Optional hook: react to the launch dialog's recording selection (e.g.
        show the source resolution). The argument is the selected ``Recording``
        objects. Default no-op. Runs on the UI thread — keep it cheap (recording.yaml
        reads are cached); if it does O(recordings) *disk* work (e.g. per-recording zarr
        reads), split it into ``resolve_recordings`` (off-thread) + ``apply_recordings``
        (UI) instead (CLAUDE.md — no O(recordings) disk I/O on the UI thread)."""

    def resolve_recordings(self, recordings: list) -> object:
        """Optional hook: the heavy, Qt-free half of reacting to a selection — the
        launch dialog runs it OFF the UI thread and hands the result to
        ``apply_recordings``. Return any payload ``apply_recordings`` understands.
        Override this (with ``apply_recordings``) only when the reaction reads disk per
        recording; the default does nothing (cheap widgets just override
        ``on_recordings_changed``)."""
        return None

    def apply_recordings(self, recordings: list, resolved: object) -> None:
        """Optional hook: apply a ``resolve_recordings`` payload to the widgets, on the
        UI thread. Default delegates to ``on_recordings_changed`` so a widget that only
        implements the cheap synchronous hook still updates."""
        self.on_recordings_changed(recordings)

    def covered_recording_ids(self) -> set[str] | None:
        """Optional hook: the subset of the last ``on_recordings_changed`` recordings the
        current config can actually run over — e.g. those that have the selected
        upstream spec (§6.5). The launch dialog narrows the run to this set and notes
        any dropped recordings. ``None`` (default) means "all selected are covered"."""
        return None

    def on_experiment_set(self, experiment) -> None:
        """Optional hook: receive the launch dialog's ``Experiment`` so the form can
        offer experiment-scoped choices (e.g. trained models). Default no-op."""

    #: Settings this form explains in place when a loaded value cannot be restored (tracking
    #: names a tracker this build does not offer beside the tracker row). The launch dialog's
    #: shared note (`start_from.unrestored_settings`) leaves these to the form.
    noted_by_form: frozenset[str] = frozenset()

    #: The pydantic model behind this form's rows, when it has one. Set it and the base
    #: ``load_config`` below restores every field the form built with ``build_fields``, so a
    #: subclass only writes the setters for widgets it built by hand.
    config_model: type | None = None

    def load_config(self, config: dict) -> None:
        """Pre-fill the form from a previous run's **config** — Refine, and the settings
        picker (§16 keeps runs immutable, so both seed a *new* run).

        The argument is a plain config dict, the same shape :meth:`config` returns. It used to
        be the run *record*, which meant every implementation dug
        ``params.get(k) or record.get(k)`` because the runner promotes some keys to the top
        level — a two-level fallback that made ``config`` and ``load_config`` impossible to
        state as inverses, and impossible to test as a round trip. The dialog flattens the
        record once now.

        Restores only the fields the model marks ``scope: "setting"``. A **selection** field —
        ``embed_run_id``, ``source_recording_ids``, ``tracking_run_id`` — names an artifact this
        experiment may not have, and the dialog re-asks for it; carrying a stale data reference
        is what kills this feature in every tool that has tried it.

        A subclass whose widgets do not all live in ``self._fields`` overrides this, sets those
        widgets, and calls ``super().load_config(config)`` for the rest. The round trip is
        guarded by ``tests/gui/test_config_round_trip.py``, so an override that forgets a knob
        fails rather than silently opening at the default.
        """
        if self.config_model is None:
            return
        fields = getattr(self, "_fields", {})
        for field in describe_config(self.config_model):
            name = field["name"]
            widget = fields.get(name)
            # `build_fields` returns `{name: (label, widget)}` and half the forms keep it that
            # way while the other half strip the label as they build their rows. Both are in
            # `self._fields`, so unwrap here rather than churn every read site.
            if isinstance(widget, tuple):
                widget = widget[1]
            if widget is None or field["scope"] != "setting" or name not in config:
                continue
            set_any_field_value(self.config_model, name, widget, config[name])


class ModuleUI(ABC):
    module_id: str  # pairs with the backend PROCESSING_MODULES entry
    mutable: bool = (
        False  # an authored, in-place run (§16) — no "Refine into a new run"
    )

    @property
    def label(self) -> str:
        """Display name in the launch picker — from ``configs.describe``, which is also what
        titles the module's docs page.

        Declared there rather than here so the two cannot disagree: they did, silently, because
        the docs title was derived from the id while this was a hand-written string, and
        ``tests/test_docs_modules_nav.py`` compares nav to pages by *path*. Behavior scan's page
        was titled "Query" the whole time. ``detail.configs`` imports no Qt, so the dependency
        runs the only safe direction.
        """
        return module_display_name(self.module_id)

    #: One line on what the module's results are for — the card for a module with one view.
    purpose: str = DEFAULT_PURPOSE

    def planned_views(self) -> list[tuple[str, str]]:
        """``(name, purpose)`` per view, known **before** the results exist.

        The landing page shows a run's cards greyed while it is still running, and there is no
        result widget to ask then. A module whose result widget is a `MultiViewWorkspace`
        returns that workspace's names; `tests/gui/test_analysis_landing.py` checks the two
        agree, so this cannot drift from what the built widget offers.
        """
        return [(RESULTS_VIEW, self.purpose)]

    @abstractmethod
    def config_widget(self) -> ConfigWidget:
        """A fresh run-config form for the launch dialog."""

    def result_widget(self, experiment, run_record: dict) -> QWidget | None:
        """A view of the **whole** run's results, for the run-detail page.

        ``None`` (the default) for a module whose results are per recording: those answer
        :meth:`preview_widget` instead, and the page builds one for the recording whose row is
        selected. Every analysis module has a whole-run view; no processing module does (the
        removed derive did, because what it produced was one derived run id, not a video to
        scrub).
        """
        return None

    def preview_widget(self, recording, run_id: str) -> QWidget | None:
        """One **recording's** results — the frames, tracks or embeddings it produced.

        The run-detail page lists the run's recordings and builds this for the selected one,
        replacing it when another row is picked; so it is built often, and its expensive reads
        belong off the UI thread (every one of these opens behind `busy.run_busy`).

        ``None`` (the default) means the module has no per-recording view, and the page falls
        back to :meth:`result_widget`.
        """
        return None

    def snapshot_widget(
        self, experiment, run_record: dict, snapshot_id: str
    ) -> QWidget | None:
        """A **read-only** view of one pinned snapshot (§21), or ``None`` if this module has none.

        ``None`` is a declaration rather than an omission: it is what tells the run-detail page to
        list the module's snapshots without a cue that opens them, so the chevron appears exactly
        where it leads somewhere. A module answers with a widget where reading a snapshot back *as
        an interpretation* means something — which is every module that authors one.
        """
        return None


GUI_MODULES: dict[str, ModuleUI] = {}

# Analysis-module UIs (§16) live in a *separate* registry so the processing launch
# dialog (which enumerates GUI_MODULES and reads PROCESSING_MODULES[id].upstream)
# never lists an analysis module — datasets/training avoid the same clash with
# bespoke dialogs. The ModuleUI / ConfigWidget contract is shared; only the
# enumeration is kept apart.
ANALYSIS_GUI_MODULES: dict[str, ModuleUI] = {}


def surfaced(registry: dict) -> list[str]:
    """The module ids in ``registry`` this build shows, sorted (§25).

    Called where a surface **enumerates** what a user may launch — never where one is looked up
    by id. A completed run of a hidden module still renders its result view: the toggle governs
    launching, not reading, so hiding a module never makes someone's existing work vanish.
    """
    from detail import surfaces

    return sorted(m for m in registry if surfaces.is_surfaced(m))


def register_module_ui(cls: type) -> type:
    """Register a ModuleUI (instantiated once), keyed by its ``module_id``."""
    instance = cls()
    GUI_MODULES[instance.module_id] = instance
    return cls


def register_analysis_module_ui(cls: type) -> type:
    """Register an analysis ModuleUI (§16), keyed by its ``module_id`` — separate
    from ``GUI_MODULES`` (see ``ANALYSIS_GUI_MODULES``)."""
    instance = cls()
    ANALYSIS_GUI_MODULES[instance.module_id] = instance
    return cls
