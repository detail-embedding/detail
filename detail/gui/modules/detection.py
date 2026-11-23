"""Detection module UI (§7): config form (+ result view).

Recording-agnostic config (one model + prompt applies across the selected
recordings). "Frame stride" thins the run to every Nth source frame (§4) and shows the
rate that lands at; "Test prompts…" opens the inline prompt-test (§8).

**One model list.** Built-in, experiment-trained and machine-imported models sit in a
single picker, each row carrying why it cannot run here if it cannot (`gui/model_availability`).
It used to be two pickers behind a *Weights source* mode switch that hid one or the other,
so "is there anything I can run?" could only be answered by toggling it — and on a Mac the
answer was yes, under the branch you were not looking at, while the visible branch offered
a CUDA-only SAM 3 with an enabled CUDA device box.

A model's own kind drives the rest: a prompted model (SAM 3) shows the prompt rows, a
closed-set one (trained / imported) shows its class list instead.
"""

from detail.api import models as api_models
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.configs.detection import (
    BATCHABLE_DEFAULT_BATCH_SIZE,
    DetectionConfig,
)
from detail.core.dataclasses import Prompt, PromptTarget, TextCue
from detail.core.registries import MODELS
from detail.executor.runners import import_runner_modules
from detail.gui import sizing
from detail.gui.compute_target import ComputeTargetRow
from detail.gui.forms import ConfigForm
from detail.gui.help_ui import field_label, help_button, label_with
from detail.gui.model_availability import (
    apply_availability,
    availability_note,
    fill_variants,
    show_note,
)
from detail.gui.modules._every_n import EveryNRow
from detail.gui._pydantic_form import field_meta, field_widget
from detail.gui.modules.registry import ConfigWidget, ModuleUI, register_module_ui
from detail.gui.qt import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

# NOT imported at module scope: gui.preview reaches back into gui.modules (for the
# pydantic form helpers and the confidence row), so a top-level import here closes a
# cycle whose outcome depends on which module the process happens to import first —
# importing gui.preview before gui.modules raised ImportError. Imported where used.


class _DetectionConfig(ConfigWidget):
    def __init__(self):
        super().__init__()
        import_runner_modules(PIPELINE_RUNNERS)  # populate the MODELS registry
        self._experiment = None
        self._unavailable: str | None = None
        form = ConfigForm(self, DetectionConfig)
        self._form = form
        self._model = QComboBox()
        form.addRow(field_label("detection", "model_id", "Model:"), self._model)
        self._no_models = availability_note()
        form.addRow("", self._no_models)
        self._classes = QLabel("")  # a closed-set model's own classes
        self._classes.setWordWrap(True)
        form.addRow(label_with("Classes:", help_button("class-name")), self._classes)
        self._variant = QComboBox()
        form.addRow(
            field_label("detection", "model_variant", "Size / variant:"), self._variant
        )
        self._prompt_note = QLabel(
            "Text prompt = what the model searches for; class name = the label stored on "
            "each detection (often the same word)."
        )
        self._prompt_note.setWordWrap(True)
        form.addRow(self._prompt_note)
        self._prompt = QLineEdit("mouse")
        self._prompt.setToolTip(
            "What the detection model searches for, e.g. 'mouse'. Used by prompted "
            "models (SAM 3); ignored by closed-set local models."
        )
        prompt_label = (
            QWidget()
        )  # "Text prompt:" + an ?, so the field stays self._prompt
        label_box = QHBoxLayout(prompt_label)
        label_box.setContentsMargins(0, 0, 0, 0)
        label_box.addWidget(QLabel("Text prompt:"))
        label_box.addWidget(
            help_button("text-prompt")
        )  # ? → the text-prompt definition
        label_box.addStretch()
        form.addRow(prompt_label, self._prompt)
        self._class = QLineEdit("mouse")
        self._class.setToolTip(
            "The label stored on each detection — how it appears in the outputs."
        )
        form.addRow(label_with("Class name:", help_button("class-name")), self._class)
        conf_meta = field_meta(DetectionConfig, "conf_threshold")  # label
        self._conf = field_widget(DetectionConfig, "conf_threshold")
        form.addRow(
            field_label("detection", "conf_threshold", conf_meta["label"] + ":"),
            self._conf,
        )
        batch_meta = field_meta(DetectionConfig, "batch_size")  # label
        self._batch = field_widget(DetectionConfig, "batch_size")
        form.addRow(
            field_label("detection", "batch_size", batch_meta["label"] + ":"),
            self._batch,
        )
        self._compute = ComputeTargetRow()
        self._compute.add_to(form.section())
        every_n_meta = field_meta(DetectionConfig, "every_n_frames")
        self._every_n = EveryNRow(DetectionConfig)
        form.addRow(
            field_label("detection", "every_n_frames", every_n_meta["label"] + ":"),
            self._every_n,
        )
        self._recordings: list = []
        self._test_btn = QPushButton("Test prompts")
        self._test_btn.clicked.connect(self._open_prompt_test)
        form.addRow("", self._test_btn)
        form.finish()
        # Connected only once every row its selection drives exists.
        self._model.currentIndexChanged.connect(self._sync_model_kind)
        self._rebuild_models()

    # -- the model list ------------------------------------------------------ #
    def _entries(self) -> list[dict]:
        """Every model this experiment could detect with: built-ins, then its own trained
        models, then the machine's imported ones. Built-ins lead, so a runnable SAM 3 is
        still what the form opens on."""

        def label(name: str | None, model_id: str, kind: str) -> str:
            # Most imported models are named after their id; printing it twice is noise.
            lead = f"{name}  ·  {model_id}" if name and name != model_id else model_id
            return f"{lead}  ·  {kind}"

        entries = []
        for model_id in sorted(MODELS):
            cls = MODELS[model_id]
            entries.append(
                {
                    "label": model_id,
                    "id": model_id,
                    "kind": "built-in",
                    "prompt_capable": TextCue in cls.supported_cue_types,
                    "classes": [],
                    "variants": list(cls.variants()),
                }
            )
        if self._experiment is None:  # built before an experiment is attached
            return entries
        for model in self._experiment.models:
            if model.status != "complete" or model.role != "detection":
                continue  # a pose student (§12) is a keypoint method, not a detector
            record = self._experiment.model_record(model.id)
            entries.append(
                {
                    "label": label(model.name, model.id, "trained"),
                    "id": model.id,
                    "kind": "trained",
                    "prompt_capable": False,
                    "classes": [c["name"] for c in record.get("classes", [])],
                    "variants": [],
                }
            )
        for im in self._experiment.imported_models:  # machine-global, shared
            if im.role != "detection":
                continue  # a catalog keypoint model (§12) is a keypoint method
            entries.append(
                {
                    "label": label(im.name, im.id, "imported"),
                    "id": im.id,
                    "kind": "imported",
                    "prompt_capable": False,
                    "classes": list(im.classes),
                    "variants": [],
                }
            )
        return entries

    def _rebuild_models(self) -> None:
        self._model.blockSignals(True)
        self._model.clear()
        for entry in self._entries():
            self._model.addItem(entry["label"], entry)
        self._model.blockSignals(False)
        self._unavailable = apply_availability(self._model, "detection")
        show_note(self._no_models, self._unavailable)
        self._sync_model_kind()  # marking may or may not have moved the selection

    def on_experiment_set(self, experiment) -> None:
        self._experiment = experiment
        self._rebuild_models()

    def _entry(self) -> dict | None:
        data = self._model.currentData()
        return data if isinstance(data, dict) else None

    def _prompt_capable(self) -> bool:
        entry = self._entry()
        return bool(entry and entry["prompt_capable"])

    def _sync_model_kind(self, *_args) -> None:
        """Show the rows the selected model actually has. Idempotent — ``_rebuild_models``
        calls it whether or not the marking moved the selection."""
        entry = self._entry()
        prompted = self._prompt_capable()
        closed_set = entry is not None and not prompted
        self._form.set_row_visible(self._classes, closed_set)
        self._form.set_row_visible(self._variant, bool(entry and entry["variants"]))
        for widget in (self._prompt_note, self._prompt, self._class):
            self._form.set_row_visible(widget, prompted)
        names = entry["classes"] if entry else []
        self._classes.setText(
            ", ".join(names) if names else "(none)" if closed_set else ""
        )
        # A closed-set model has no prompt, but it can still be previewed — the dialog
        # offers a per-class preview instead of candidate prompts.
        self._test_btn.setText("Test prompts" if prompted else "Test detection")
        sizing.set_available(
            self._test_btn, entry is not None, "Choose a detection model first"
        )
        self._sync_variants()
        self._sync_backends()
        self._sync_batch_default()

    def _sync_batch_default(self) -> None:
        # Batch-capable models (YOLO, incl. trained/imported) default to a tunable batch
        # size; SAM3 (and any non-batching model) is fixed + disabled at 1.
        entry = self._entry()
        if entry is None:
            batchable = False
        elif entry["kind"] == "built-in":
            batchable = bool(
                getattr(MODELS.get(entry["id"]), "supports_batching", False)
            )
        else:
            batchable = True  # trained/imported are YOLO
        self._batch.setEnabled(batchable)
        self._batch.setValue(BATCHABLE_DEFAULT_BATCH_SIZE if batchable else 1)

    def _selected_backends(self) -> list[str]:
        # Built-ins report their own; trained/imported resolve by family and take the api
        # fallback of all three. ComputeTargetRow then keeps only what this machine has.
        entry = self._entry()
        return api_models.supported_backends(entry["id"]) if entry else []

    def _sync_backends(self) -> None:
        self._compute.set_backends(self._selected_backends())

    def _sync_variants(self) -> None:
        entry = self._entry()
        variants = entry["variants"] if entry else []
        self._variant.clear()
        if not variants:
            self._variant.addItem("(default)", None)
            self._variant.setEnabled(False)
            return
        # The model's declared default, not dict order. Landing on the first entry
        # meant a form nobody touched ran a size nobody installed.
        model_cls = MODELS.get(entry["id"])
        default = model_cls.default_variant() if model_cls is not None else None
        fill_variants(self._variant, entry["id"], variants, default)
        self._variant.setEnabled(True)

    def on_recordings_changed(self, recordings: list) -> None:
        self._recordings = recordings
        self._every_n.set_recordings(recordings)

    def _open_prompt_test(self) -> None:
        prompted = self._prompt_capable()
        title = "Prompt test" if prompted else "Test detection"
        entry = self._entry()
        if entry is None:
            QMessageBox.information(
                self, title, self._unavailable or "Pick a model first."
            )
            return
        if not self._recordings:
            QMessageBox.information(
                self, title, "Check the recordings to test on first."
            )
            return
        from detail.gui.preview import PromptTestDialog

        compute = self._compute.compute_target()
        if not prompted:
            PromptTestDialog(
                self._recordings,
                model_id=entry["id"],
                prompt_text="",
                compute=compute,
                parent=self,
                closed_set=True,
                classes=entry["classes"],
            ).exec()
            return
        PromptTestDialog(
            self._recordings,
            model_id=entry["id"],
            model_variant=self._variant.currentData(),
            prompt_text=self._prompt.text(),
            compute=compute,
            parent=self,
        ).exec()

    def validation_error(self) -> str | None:
        return self._unavailable

    def load_config(self, config: dict) -> None:
        """Restore a previous run's detector settings (Refine, and the settings picker).

        ``compute`` is deliberately not restored. ``ComputeTargetRow`` offers the intersection
        of the model's backends with what *this* host has, so a stored ``cuda`` is unofferable
        on a Mac — and Qt makes ``setCurrentText`` past a one-entry combo a silent no-op, which
        is the misbind this whole flag exists to avoid. The device is a property of where you
        are running, not of the run you are copying.
        """
        index = self._model.findData(config.get("model_id"))
        if index >= 0:
            self._model.setCurrentIndex(index)
        index = self._variant.findData(config.get("model_variant"))
        if index >= 0:
            self._variant.setCurrentIndex(index)
        self._load_prompt(config.get("prompt"))
        if config.get("conf_threshold") is not None:
            self._conf.setValue(float(config["conf_threshold"]))
        if config.get("batch_size") is not None:
            self._batch.setValue(int(config["batch_size"]))
        self._every_n.set_value(config.get("every_n_frames"))

    def _load_prompt(self, prompt) -> None:
        """Fill the class + text rows from a stored prompt, in either of its two forms.

        `config()` emits a live `Prompt`; a run record on disk holds `Prompt.to_dict()`'s
        serialized form. Both reach `load_config`, so both are accepted — a dict goes through
        `Prompt.from_dict` rather than being read key by key, which is how the field name got
        written as `name` here when the dataclass calls it `class_name`. Only a `TextCue` has a
        row on this form; a point or image cue is left to the prompt-test dialog that authored
        it.
        """
        if isinstance(prompt, dict):
            prompt = Prompt.from_dict(prompt)
        targets = getattr(prompt, "targets", None)
        if not targets:
            return
        target = targets[0]
        self._class.setText(target.class_name)
        if isinstance(target.cue, TextCue):
            self._prompt.setText(target.cue.text)

    def config(self) -> dict:
        # weights_source is omitted: the detection validator resolves the source
        # (pretrained / trained / imported) from the model id.
        entry = self._entry()
        prompt = (
            Prompt(
                targets=[
                    PromptTarget(
                        self._class.text().strip(), TextCue(self._prompt.text().strip())
                    )
                ]
            )
            if self._prompt_capable()
            else Prompt(targets=[])  # closed-set: it detects its trained classes
        )
        return {
            "model_id": entry["id"] if entry else None,
            "model_variant": self._variant.currentData(),
            "prompt": prompt,
            "compute": self._compute.compute_target(),
            "conf_threshold": self._conf.value(),
            "batch_size": self._batch.value(),
            "every_n_frames": self._every_n.value(),
        }


@register_module_ui
class DetectionModuleUI(ModuleUI):
    module_id = "detection"

    def config_widget(self) -> ConfigWidget:
        return _DetectionConfig()

    def preview_widget(self, recording, run_id: str) -> QWidget | None:
        from detail.gui.run_preview import DetectionRunPreview

        return DetectionRunPreview(recording, run_id)
