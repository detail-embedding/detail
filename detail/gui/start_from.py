"""The **Start from** row — begin a run at defaults, a shipped baseline, or a previous run.

Every launch form is a set of knobs a user has to fill in from nothing, every time. `PR #295`
made each form able to *take a configuration back* (`ConfigWidget.load_config`) and left the
question of where one comes from; this is that surface, and it is the only one — the Refine
button it replaces reached 2 of 9 analysis modules and no processing module at all, because it
was gated on a run being non-mutable.

Three sources, in the order a person looks for them:

- **Defaults** — the pydantic model's own, always first and always present.
- **Baselines** — named starting points shipped in `configs/baselines/`. They hold settings and
  never selections, so one travels between experiments and machines.
- **Previous runs** — newest first. A run id is minted sequentially, so descending id is
  recency and costs no extra read.

A run recorded before `#295` carries no settings. It is offered **disabled, with the reason**
rather than hidden: "Grey out, don't hide" (gui_design.md §3.1) — a user who made that run and cannot
find it in this list would reasonably conclude the list is broken, where the truth is that the
run predates the tool recording what it was launched with.

The payload rides in `itemData`; the label is display only. That is the `apply_availability` /
`selected_model_id` lesson one surface over — once a label carries a summary, `currentText()`
is a sentence, not an id.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QWidget

from detail.configs.baselines import baselines_for
from detail.configs.describe import iter_config_models, summarize_settings

#: What the first item says, and what a form opens on.
DEFAULTS_LABEL = "Defaults"

#: Shown on a run that recorded no settings — every analysis run made before `#295`, and the
#: reason the item is there at all rather than quietly missing.
NO_SETTINGS = "no settings recorded"


class StartFromRow(QWidget):
    """A combo listing the starting points for one module. Emits the config that was picked.

    Not a `QComboBox` subclass: the launch dialogs add it with `form.addRow(label, widget)`,
    and a widget is what that takes. `combo` stays public for the guards, which need to read
    the item list rather than drive a signal to learn what is offered.
    """

    picked = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QVBoxLayout

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setToolTip(
            "Begin from this tool's defaults, a shipped baseline, or the settings of a run "
            "you already made. The recordings and the upstream run are always chosen here, "
            "never carried."
        )
        layout.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(self._on_pick)
        self._loading = False

    # -- population --------------------------------------------------------- #
    def set_module(self, experiment, module_id: str, runs: list[dict]) -> None:
        """Rebuild the list for ``module_id``.

        ``runs`` is the caller's records — processing and analysis store theirs in different
        places and under different keys (`config` vs `params`), so the dialog that knows which
        it is passes them in rather than this widget guessing.
        """
        self._loading = True  # a rebuild moves the index; that is not a user's pick
        self.combo.clear()
        self.combo.addItem(DEFAULTS_LABEL, None)

        baselines = baselines_for(module_id)
        if baselines:
            self._add_heading("baselines")
            for baseline in baselines:
                self.combo.addItem(baseline.name, baseline.for_module(module_id))

        if runs:
            self._add_heading("previous runs")
            for record in runs:
                label, settings = _run_item(module_id, record)
                self.combo.addItem(label, settings)
                if settings is None:
                    self._disable_last()
        self.combo.setCurrentIndex(0)
        self._loading = False

    def _add_heading(self, text: str) -> None:
        """A disabled row that groups what follows. No styling — a disabled item greys itself."""
        self.combo.addItem(f"── {text} ──", None)
        self._disable_last()

    def _disable_last(self) -> None:
        from PyQt6.QtCore import Qt

        index = self.combo.count() - 1
        model = self.combo.model()
        item = model.item(index)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

    # -- picking ------------------------------------------------------------ #
    def _on_pick(self, index: int) -> None:
        """Every enabled item is a pick, **including Defaults**.

        The first version skipped it, reasoning that the form is already at its defaults when
        built and that loading nothing over a form the user had edited would be a reset nobody
        asked for. That is exactly backwards: choosing *Defaults* is asking for the reset. Go
        to a run, change your mind, go back, and the form kept the run's values — with the row
        above it reading "Defaults".

        The comment was more convincing than the behaviour, which is the useful part of the
        mistake: it sounded like a considered choice, so nothing went looking. What the tests
        missed is the same shape — every one of them picked *from* Defaults, and none picked
        back *to* it.
        """
        if self._loading or index < 0:
            return
        settings = self.combo.itemData(index)
        if settings is None and self.combo.itemText(index) != DEFAULTS_LABEL:
            return  # a heading, which is disabled anyway
        self.picked.emit(dict(settings or {}))


def _run_item(module_id: str, record: dict) -> tuple[str, dict | None]:
    """``(label, settings)`` for one previous run, or ``(label, None)`` when it recorded none."""
    settings = record.get("params") or record.get("config") or {}
    head = record.get("id", "?")
    recordings = record.get("recording_ids") or []
    bits = [head]
    if recordings:
        bits.append(f"{len(recordings)} recording(s)")
    if not settings:
        bits.append(f"({NO_SETTINGS})")
        return "  ·  ".join(bits), None
    model = dict(iter_config_models()).get(module_id)
    if model is not None:
        bits.append(summarize_settings(model, settings))
    created = (record.get("created_at") or "")[:10]
    if created:
        bits.append(created)
    return "  ·  ".join(bits), settings


def analysis_runs_for(experiment, module_id: str) -> list[dict]:
    """Completed-enough analysis runs of one module, newest first.

    Every status except a run still going is offered: a failed run's settings are exactly what
    you want to start from when you are fixing why it failed.
    """
    records = []
    for ref in experiment.analysis_runs:
        if ref.module_id != module_id or ref.status == "running":
            continue
        records.append(experiment.analysis_record(ref.id))
    return sorted(records, key=lambda r: r.get("id", ""), reverse=True)


def processing_runs_for(experiment, module_id: str) -> list[dict]:
    """Processing runs of one module, newest first. Their config lives under ``config``."""
    return sorted(
        (
            record
            for record in experiment.processing_runs
            if record.get("module_id") == module_id
            and record.get("status") != "running"
        ),
        key=lambda r: r.get("id", ""),
        reverse=True,
    )
