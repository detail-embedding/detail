"""Helpers for config widgets that reference a per-recording upstream run.

A batch run passes one config to every selected recording, but upstream run ids
(and arena ids) are per-recording. These return the ids present in *every* selected
recording — the ones a batch run can reference uniformly across the selection.
"""

import re
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from detail.api import run_spec
from detail.core import class_names as class_names_mod
from detail.gui.dialog import Dialog
from detail.gui.sizing import fit_list_height
from detail.pipeline.detection import detected_classes
from detail.storage import experiment as experiment_store

# A derived embedding run (§14) encodes its transform in ``model_variant`` as
# ``derived:<transform>(<params>)(src=<run>)`` — one segment per transform applied
# (a derive of a derive chains them). Extract the transform + params for the dropdown
# so a derived run is legible, not indistinguishable from its base run.
_DERIVED_RE = re.compile(r"derived:(\w+)\(([^)]*)\)")


def derived_descriptor(model_variant: str | None) -> str | None:
    """A compact ``transform(params)`` summary of a derived run's provenance, or ``None``
    for a model run. Chained derives join with ``→`` (e.g. ``center → drop_top(k=1)``).
    """
    if not model_variant:
        return None
    parts = [
        f"{transform}({params})" if params else transform
        for transform, params in _DERIVED_RE.findall(model_variant)
    ]
    return " → ".join(parts) if parts else None


def repopulate_combo(combo: QComboBox, items: list[tuple[str, object]]) -> None:
    """Refill a combo with (label, data) pairs, keeping the prior selection if
    its data value is still present."""
    previous = combo.currentData()
    combo.clear()
    for label, data in items:
        combo.addItem(label, data)
    index = combo.findData(previous)
    if index >= 0:
        combo.setCurrentIndex(index)


def common_complete_run_ids(recordings: list, module: str) -> list[str]:
    """Run ids of ``module`` that are complete in every selected recording."""
    if not recordings:
        return []
    id_sets = [
        {
            r["id"]
            for r in getattr(s, f"{module}_runs", [])
            if r.get("status") == "complete"
        }
        for s in recordings
    ]
    return sorted(set.intersection(*id_sets)) if id_sets else []


def run_label(module: str, record: dict, model_names: dict | None = None) -> str:
    """A human label for a per-recording run record: the originating experiment run
    (``run_NNN``) when known, else the per-recording id, plus the key params and
    date so users can tell runs apart (the raw id is kept as the combo's data).

    ``model_names`` (``Experiment.model_names``) turns a stored model **id** into the name
    its owner gave it (§11); without it, or for an id it doesn't cover, the id renders as
    before.
    """
    head = record.get("processing_run_id") or record.get("id", "")
    bits = [head]
    if module == "detection":
        bits += [_model_name(record.get("model_id"), model_names)]
        bits += [record.get("model_variant")]
    elif module == "tracking":
        bits += [record.get("tracker_id"), record.get("detection_run_id")]
    elif module == "embedding":
        bits += [_model_name(record.get("model_id"), model_names)]
        bits += [record.get("constructor")]
        # A derived run appends its transform, so it reads e.g.
        # "emb_02 · dinov3_convnext_base · masked_color_centered · drop_top(k=1)".
        bits.append(derived_descriptor(record.get("model_variant")))
    created = (record.get("created_at") or "")[:16].replace("T", " ")
    bits.append(created or None)
    return " · ".join(str(b) for b in bits if b)


def _model_name(model_id, model_names: dict | None) -> str | None:
    """``name · model_id`` when the id has a name, else the id (see run_spec's twin)."""
    if not model_id:
        return model_id
    name = (model_names or {}).get(model_id)
    return f"{name} · {model_id}" if name and name != model_id else model_id


def common_complete_run_items(
    recordings: list, module: str, model_names: dict | None = None
) -> list[tuple[str, str]]:
    """``(label, run_id)`` pairs for ``module`` runs complete in every selected
    recording — rich labels for the combo, the per-recording id as the data value.

    When the id maps to different experiment runs across recordings the label is taken
    from the first recording's record; the id still uniquely selects the upstream.
    """
    common = common_complete_run_ids(recordings, module)
    if not common:
        return []
    first = {r["id"]: r for r in getattr(recordings[0], f"{module}_runs", [])}
    return [
        (run_label(module, first.get(rid, {"id": rid}), model_names), rid)
        for rid in common
    ]


def arena_name_groups(recordings: list) -> list[dict]:
    """``[{name, recordings}]`` — one per distinct arena *name* across the selection,
    with the recording ids that have an arena by that name.

    **A name, not an id, because an id is only unique within one recording.** Arena ids
    are per-recording mint order, so ``arena_001`` on two recordings is two unrelated
    polygons that happen to share a number. Intersecting ids — which is what this
    replaced — therefore offered one dropdown entry that meant a different field of view
    on different recordings, and every run completed looking fine.

    Latent while every recording had exactly one arena. It became reachable when a field
    of view held by a tracking run started refusing to be replaced (#350): correcting one
    means adding a second arena beside it, and then the corrected one is ``arena_002``
    where there was already one and ``arena_001`` where there was not.

    Shaped like :func:`spec_groups` on purpose — a name need not be present in *every*
    selected recording, so coverage is surfaced per option and the run narrows to the
    recordings that have it.
    """
    by_name: dict[str, list[str]] = {}
    for recording in recordings:
        for arena in recording.arenas:
            by_name.setdefault(arena["name"], []).append(recording.id)
    return [{"name": name, "recordings": by_name[name]} for name in sorted(by_name)]


def arena_items(groups: list[dict], total: int) -> list[tuple[str, str]]:
    """``(label, name)`` combo items, each annotated with ``N/total`` coverage — the
    arena twin of :func:`spec_items`. A group's ``label`` (a class's name, §1) is shown
    when it has one; the item's data is always the stored ``name``."""
    items = []
    for group in groups:
        covered = len(group["recordings"])
        suffix = f"  ·  {covered}/{total} recordings" if total else ""
        shown = group.get("label", group["name"])
        items.append((f"{shown}{suffix}", group["name"]))
    return items


def experiment_class_names(recordings: list) -> dict[str, str]:
    """The experiment's ``{stored class: name}`` (§1 *Class names*) read beside the
    selection's recordings — every recording of a selection is in one experiment. ``{}``
    for no recordings, or recordings with no directory (a test fake)."""
    for recording in recordings:
        recording_dir = getattr(recording, "dir", None)
        if recording_dir is not None:
            return experiment_store.read_class_names(Path(recording_dir).parent.parent)
    return {}


def arena_covered_ids(groups: list[dict], name: str | None) -> set[str] | None:
    """Recording ids that have an arena called ``name`` — ``None`` (all) when ``name`` is
    ``None``, which is the whole-frame choice, and an empty set when it matches none."""
    if name is None:
        return None
    for group in groups:
        if group["name"] == name:
            return set(group["recordings"])
    return set()


def detected_class_groups(recordings: list, detection_spec: dict | None) -> list[dict]:
    """``[{name, recordings}]`` — one per distinct class actually detected across the
    selection's own run of ``detection_spec`` (§13's tracking class filter).

    Shaped exactly like :func:`arena_name_groups` so :func:`arena_items` /
    :func:`arena_covered_ids` serve both — but a class is not a property of the
    recording itself the way an arena is; it has to be read off whichever detection
    run ``detection_spec`` resolves to *on that recording* (``run_spec.
    resolve_spec_for_recording``), since the same spec can be a different per-recording
    run id. ``None`` (no detection spec chosen yet) has no classes to offer.
    """
    if detection_spec is None:
        return []
    names = experiment_class_names(recordings)
    by_name: dict[str, list[str]] = {}
    for recording in recordings:
        run_id = run_spec.resolve_spec_for_recording(
            recording, "detection", detection_spec
        )
        if run_id is None:
            continue
        detection_run_dir = Path(recording.dir) / "detections" / run_id
        for class_name in detected_classes(detection_run_dir):
            by_name.setdefault(class_name, []).append(recording.id)
    return [
        {
            "name": name,
            "label": class_names_mod.label(name, names),
            "recordings": by_name[name],
        }
        for name in sorted(by_name)
    ]


# --------------------------------------------------------------------------- #
# Spec-based upstream selection (§6.5). A picker offers upstream *specs* (a run's
# full-lineage identity) rather than a per-recording run id, and the launch dialog
# resolves each recording to its own matching run. Unlike the "common id" helpers
# above, a spec need not be present in *every* selected recording — the coverage is
# surfaced per option and the run narrows to the recordings that have it.
# --------------------------------------------------------------------------- #
def spec_groups(
    recordings: list, module_id: str, model_names: dict | None = None
) -> list[dict]:
    """``[{spec, label, recordings}]`` — one per distinct upstream spec across the
    selected recordings (``recordings`` are the covered recording ids). A class in a label
    reads as the experiment names it (§1)."""
    return run_spec.group_specs(
        recordings,
        module_id,
        model_names,
        class_names=experiment_class_names(recordings),
    )


def spec_items(groups: list[dict], total: int) -> list[tuple[str, dict]]:
    """``(label, spec)`` combo items, each label annotated with ``N/total`` coverage."""
    items = []
    for group in groups:
        covered = len(group["recordings"])
        suffix = f"  ·  {covered}/{total} recordings" if total else ""
        items.append((f"{group['label']}{suffix}", group["spec"]))
    return items


def spec_runs(recordings: list, module_id: str, groups: list[dict]) -> dict:
    """Off the UI thread: ``{frozen spec: {recording_id: run id}}``, each group's own run in
    each recording it covers (a run id is a per-recording counter, §6.5), for a caller that
    hands per-recording runs to the API rather than a spec. ``recordings`` are the
    ``Recording`` objects ``groups`` was built from."""
    by_id = {r.id: r for r in recordings}
    runs = {}
    for group in groups:
        resolved = {}
        for recording_id in group["recordings"]:
            run_id = run_spec.resolve_spec_for_recording(
                by_id[recording_id], module_id, group["spec"]
            )
            if run_id:
                resolved[recording_id] = run_id
        runs[run_spec.freeze_spec(group["spec"])] = resolved
    return runs


def covered_ids(groups: list[dict], spec: dict | None) -> set[str] | None:
    """Recording ids covered by ``spec`` among ``groups`` — ``None`` (all) when ``spec``
    is ``None`` (a root/"original" choice), an empty set when it matches no group."""
    if spec is None:
        return None
    key = run_spec.freeze_spec(spec)
    for group in groups:
        if run_spec.freeze_spec(group["spec"]) == key:
            return set(group["recordings"])
    return set()


def repopulate_spec_combo(
    combo: QComboBox, items: list[tuple[str, dict]], *, fallback: dict | None
) -> None:
    """Refill a combo with ``(label, spec)`` pairs, keeping the prior spec selection
    (matched by ``freeze_spec`` — spec dicts are rebuilt each refresh, so identity
    comparison would not find them), else selecting ``fallback``.

    ``fallback`` has no default: left to the combo, a refill lands on its first row, which
    is label order — the oldest experiment run — and that is a choice nobody made. A caller
    holding groups passes :func:`widest_group`'s spec; ``None`` keeps the first row."""
    previous = combo.currentData()
    combo.clear()
    for label, spec in items:
        combo.addItem(label, spec)
    for wanted in (previous, fallback):
        if not isinstance(wanted, dict):
            continue
        key = run_spec.freeze_spec(wanted)
        for index in range(combo.count()):
            data = combo.itemData(index)
            if isinstance(data, dict) and run_spec.freeze_spec(data) == key:
                combo.setCurrentIndex(index)
                return


def widest_group(groups: list[dict]) -> dict:
    """The group covering the most recordings, the **newest** on a tie (its ``created_at``).

    A tie was broken by list order, which is label order, and a label leads with its
    experiment run — so two detection runs of one model over the same recordings opened on
    ``run_001``, the older, and a keypoint detect run went onto it unnoticed (reported). The
    newest settings are the ones a person most likely just made."""
    return max(groups, key=lambda g: (len(g["recordings"]), g.get("created_at") or ""))


def group_runs(groups: list[dict], spec: dict | None) -> dict[str, str]:
    """``{recording_id: run id}`` of the group ``spec`` names — its own run on each recording
    it covers — or ``{}`` for no spec, or one no group matches."""
    if spec is None:
        return {}
    key = run_spec.freeze_spec(spec)
    for group in groups:
        if run_spec.freeze_spec(group["spec"]) == key:
            return {rec: str(run) for rec, run in group["runs"].items()}
    return {}


def spec_reads(
    recordings: list, module_id: str, groups: list[dict], field: str
) -> dict:
    """Off the UI thread: ``{frozen spec: {recording_id: run id}}`` — the upstream run each
    group's own run read on each recording it covers, by its record's ``field`` (a tracking
    run's ``detection_run_id``, a keypoint track run's ``tracking_run_id``).

    What :func:`lineage_fits` compares. A stage that takes two inputs checks that they share
    a run **per recording**, and their settings cannot answer that: two runs of one spec are
    still two runs."""
    by_id = {r.id: r for r in recordings}
    reads = {}
    for group in groups:
        per = {}
        for recording_id, run_id in group["runs"].items():
            records = getattr(by_id[recording_id], f"{module_id}_runs", None) or []
            record = next((r for r in records if r.get("id") == run_id), None)
            if record is not None and record.get(field) is not None:
                per[recording_id] = str(record[field])
        reads[run_spec.freeze_spec(group["spec"])] = per
    return reads


def lineage_fits(reads: dict, wanted: dict[str, str]) -> dict[tuple, set[str]]:
    """``{frozen spec: recording ids}`` — for each group in ``reads`` (:func:`spec_reads`),
    the recordings on which its run read the run ``wanted`` names there. What
    :meth:`UpstreamSpecPicker.restrict` takes."""
    return {
        key: {rec for rec, up in per.items() if wanted.get(rec) == up}
        for key, per in reads.items()
    }


class UpstreamSpecPicker:
    """An analysis form's upstream picker: one option per set of settings across the
    selection, each with its ``N/M`` coverage, resolved to every recording's own run (§6.5).

    What replaced offering the ids *every* selected recording shares. An id is a
    per-recording counter, so the same settings are ``emb_01`` in one recording and
    ``emb_02`` in the one that was retried or added later — which that picker could not
    offer at all — and two unrelated runs can share an ``emb_01``, which it offered as one.

    The widget half of the ``resolve_recordings`` / ``apply_recordings`` split: grouping
    reads every processing module's records per recording, so :meth:`resolve` runs off the
    UI thread and :meth:`apply` only fills the combo. The form puts :attr:`combo` in its
    layout and emits :meth:`spec` as ``<upstream>_spec``.

    ``auto_label`` adds a first row, selected by default, whose :meth:`spec` is ``None``: the
    stage resolves the run itself (keypoint track's tracking run — the one on the detect
    run's detection run, refused when there are several, §26). A choice, not a fact: the
    other pickers have no such row.

    :meth:`restrict` ties the options to another input of the same stage: one whose run reads
    a different upstream run than the form's other choice is greyed with the reason, and its
    coverage counts only the recordings where the pair fits — the stage refuses the rest.
    """

    def __init__(self, module_id: str, *, auto_label: str | None = None) -> None:
        self.module_id = module_id
        self.combo = QComboBox()
        self._groups: list[dict] = []
        self._total = 0
        self._fits: dict[tuple, set[str]] | None = None
        self._why = ""
        self._preferred: dict | None = None
        self._auto_label = auto_label
        if auto_label is not None:
            self.combo.addItem(auto_label, None)

    @property
    def _offset(self) -> int:
        return 0 if self._auto_label is None else 1

    def resolve(self, recordings: list, model_names: dict) -> list[dict]:
        """Off the UI thread: the groups for ``recordings``. Pinned, so the read cache's
        per-property ``stat`` is paid once per recording, not once per field.

        ``model_names`` (``Experiment.model_names``) has no default: the forms that left it
        out showed a model's id where every other picker showed its name."""
        from detail.storage import read_cache

        with read_cache.pinned():
            return spec_groups(recordings, self.module_id, model_names)

    def apply(self, groups: list[dict], total: int) -> None:
        """UI thread: install ``groups``, unrestricted until the next :meth:`restrict`.
        Keeps the previous choice when it is still on offer; otherwise selects the widest
        coverage (:func:`widest_group`), so a recording missing a run does not decide what
        the other forty-seven are offered first."""
        previous = self.spec()
        self._groups = groups
        self._total = total
        self._fits = None
        self._fill(previous)

    def restrict(self, fits: dict[tuple, set[str]] | None, why: str = "") -> None:
        """UI thread: offer each group only on the recordings ``fits`` gives it
        (:func:`lineage_fits`); one with none is greyed, saying ``why``. ``None`` lifts it.
        A choice that no longer fits moves to the widest that does."""
        previous = self.spec()
        self._fits = fits
        self._why = why
        self._fill(previous)

    def _fitting(self, group: dict) -> set[str]:
        covered = set(group["recordings"])
        if self._fits is None:
            return covered
        return covered & self._fits.get(run_spec.freeze_spec(group["spec"]), set())

    def _fill(self, previous: dict | None) -> None:
        shown = [{**g, "recordings": sorted(self._fitting(g))} for g in self._groups]
        # A greyed row says why in place of its coverage, which would read 0/M.
        items = [
            (
                label if group["recordings"] else f"{group['label']}  ·  {self._why}",
                spec,
            )
            for (label, spec), group in zip(spec_items(shown, self._total), shown)
        ]
        if self._auto_label is not None:
            items = [(self._auto_label, None)] + items
        # The default is chosen below, among the rows that fit.
        repopulate_spec_combo(self.combo, items, fallback=None)
        model = self.combo.model()
        for index, group in enumerate(shown):
            item = model.item(index + self._offset)
            item.setEnabled(bool(group["recordings"]))
            item.setData(
                None if group["recordings"] else self._why,
                Qt.ItemDataRole.ToolTipRole,
            )
        if self._preferred is not None and self._select(self._preferred):
            return
        if self._auto_label is not None:
            if previous is None or not self._select(previous):
                self.combo.setCurrentIndex(0)
            return
        if previous is not None and self._select(previous):
            return
        offered = [g for g in shown if g["recordings"]]
        if offered:
            self._select(widest_group(offered)["spec"])
        else:  # a greyed row sitting in a closed combo reads as chosen
            self.combo.setCurrentIndex(-1)

    def prefer(self, spec: dict) -> None:
        """Select ``spec`` whenever it is on offer (a next-step launch from that run), in
        place of the widest-coverage default."""
        self._preferred = spec
        self._select(spec)

    def _select(self, spec: dict) -> bool:
        """Select ``spec`` if it is on offer and fits; False when it is not."""
        key = run_spec.freeze_spec(spec)
        for index, group in enumerate(self._groups):
            if run_spec.freeze_spec(group["spec"]) == key:
                if not self._fitting(group):
                    return False
                self.combo.setCurrentIndex(index + self._offset)
                return True
        return False

    @property
    def groups(self) -> list[dict]:
        """The groups last installed by :meth:`apply`."""
        return self._groups

    def spec(self) -> dict | None:
        data = self.combo.currentData()
        return data if isinstance(data, dict) else None

    def none_fit(self) -> bool:
        """Whether there are runs on offer and :meth:`restrict` greyed every one of them —
        so a form can say "none on this <other input>" rather than "run it first"."""
        return bool(self._groups) and not any(self._fitting(g) for g in self._groups)

    def covered(self) -> set[str] | None:
        """The selected recordings the chosen settings cover; ``None`` (nothing narrowed
        yet) when no settings are chosen.

        Never the empty set for "nothing chosen": the launch dialog greys every recording
        outside this set, the groups are computed from the *checked* recordings, and the
        recordings start unchecked, so an empty answer greyed every row before one could be
        checked and the form could never be filled. A selection with no matching run is
        refused by the form's ``validation_error`` instead, where it can say why.

        Under :meth:`restrict`, only the recordings where the choice fits.
        """
        spec = self.spec()
        if spec is None:
            return None
        key = run_spec.freeze_spec(spec)
        for group in self._groups:
            if run_spec.freeze_spec(group["spec"]) == key:
                return self._fitting(group)
        return set()


def add_run_on_row(form, tooltip: str) -> QComboBox:
    """A remote-capable analysis module's *Run on* row (§6 "Remote analysis"): this computer,
    then each registered remote target. Added to ``form``; returns the combo, whose data is the
    target's name (``None`` for this computer)."""
    from detail.api import remote as api_remote
    from detail.gui.help_ui import help_button, label_with

    combo = QComboBox()
    combo.setToolTip(tooltip)
    combo.addItem("This computer", None)
    for target in api_remote.list_remote_targets():
        combo.addItem(f"{target.name} (remote)", target.name)
    form.addRow(label_with("Run on:", help_button("compute-target")), combo)
    return combo


def run_on_compute(combo: QComboBox) -> dict | None:
    """The ``compute`` a *Run on* row asks for, or ``None`` for this computer."""
    from detail.core.dataclasses import ComputeTarget

    target = combo.currentData()
    if target is None:
        return None
    return ComputeTarget("remote", None, None, None, remote_target=target).to_dict()


def select_run_on(combo: QComboBox, config: dict) -> None:
    """Show where ``config`` ran on a *Run on* row: its remote target when that is still
    registered here, else this computer."""
    index = combo.findData((config.get("compute") or {}).get("remote_target"))
    combo.setCurrentIndex(index if index >= 0 else 0)


def pooled_settings_note(n_settings: int) -> str:
    """`` from N embedding settings`` when a fit pools several, else nothing — so the advice
    line says out loud that the recordings came through different chains."""
    return f" from {n_settings} embedding settings" if n_settings > 1 else ""


def recording_names(recordings: list) -> dict[str, str]:
    """Off the UI thread: ``{recording_id: name}`` for ``recordings``, what a per-recording
    choice is listed under (§11: never an id where a name will do). Pinned, so the read cache's
    ``stat`` is paid once per recording."""
    from detail.storage import read_cache

    with read_cache.pinned():
        return {r.id: getattr(r, "name", None) or r.id for r in recordings}


def run_options(groups: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """``{recording_id: [(run id, settings label), ...]}`` — every complete run a recording has
    of ``groups`` (``run_spec.group_specs``), sorted by id: what a per-recording choice offers.
    """
    options: dict[str, list[tuple[str, str]]] = {}
    for group in groups:
        for sid, run_id in group["runs"].items():
            options.setdefault(sid, []).append((str(run_id), group["label"]))
    return {sid: sorted(runs) for sid, runs in options.items()}


#: A per-recording choice's first row: no override, the recording resolves with the rest.
AS_RESOLVED = "(as the chosen settings resolve)"


class RunOverridesDialog(Dialog):
    """Name the exact ``module_id`` run each recording reads (§6.5 *per-recording overrides*).

    One row per recording with a choice of its complete runs; the first row of each leaves it
    to the picker's settings. Recordings with more than one run come first: they are the ones
    the settings cannot tell apart, and the reason this dialog exists.
    """

    def __init__(
        self,
        module_id: str,
        run_options: dict[str, list[tuple[str, str]]],
        recording_names: dict[str, str],
        chosen_runs: dict[str, str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Choose the {module_id} run per recording")
        intro = QLabel(
            f"Pick the {module_id} run a recording reads when the chosen settings would read "
            f"the wrong one — for example a run redone beside a bad one. The other recordings "
            f"resolve as before."
        )
        intro.setWordWrap(True)
        self.body().addWidget(intro)
        form = QFormLayout()
        self.body().addLayout(form)
        self._combos: dict[str, QComboBox] = {}
        order = sorted(
            run_options,
            key=lambda sid: (len(run_options[sid]) < 2, recording_names.get(sid, sid)),
        )
        for sid in order:
            combo = QComboBox()
            combo.addItem(AS_RESOLVED, None)
            for run_id, label in run_options[sid]:
                combo.addItem(f"{run_id}  ·  {label}", run_id)
                combo.setItemData(combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
            index = combo.findData(chosen_runs.get(sid))
            combo.setCurrentIndex(max(0, index))
            form.addRow(recording_names.get(sid, sid), combo)
            self._combos[sid] = combo
        if not run_options:
            self.body().addWidget(
                QLabel(f"No recording has a completed {module_id} run.")
            )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.set_buttons(buttons)

    def chosen(self) -> dict[str, str]:
        """``{recording_id: run id}`` for every recording given a run of its own."""
        return {
            sid: combo.currentData()
            for sid, combo in self._combos.items()
            if combo.currentData() is not None
        }


class RunOverrides(QObject):
    """A *Per recording…* button and a note saying which recordings read a run named for them
    — the GUI half of ``<upstream>_overrides`` (§6.5), which ``run_spec.resolve_overrides``
    checks and puts in front of the settings-based resolution.

    For the recording whose run was redone beside a bad one: two complete runs with different
    settings, so ticking both settings is refused and ticking one reads the bad run on some
    recordings. :class:`PooledSpecPicker` embeds one; a form may place another beside a second
    upstream (the location export's keypoint track picker). Same split as the pickers: the
    options are installed on the UI thread from groups resolved off it, so opening the dialog
    reads no disk.
    """

    changed = pyqtSignal()

    def __init__(self, module_id: str) -> None:
        super().__init__()
        self.module_id = module_id
        self.widget = QWidget()
        row = QHBoxLayout(self.widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton("Per recording…")
        self._tip = (
            f"Name the exact {module_id} run a recording reads, for one whose run was redone "
            "beside a run you do not want"
        )
        self.button.clicked.connect(self.open_dialog)
        row.addWidget(self.button)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        row.addWidget(self.note, 1)
        self._options: dict[str, list[tuple[str, str]]] = {}
        self._names: dict[str, str] = {}
        self._chosen: dict[str, str] = {}
        self._refresh()

    def set_options(self, groups: list[dict], names: dict[str, str]) -> None:
        """UI thread: offer each recording's complete runs of ``groups``. A choice whose run
        is no longer offered is dropped rather than carried to a run that would refuse it.
        """
        self._options = run_options(groups)
        self._names = dict(names)
        kept = {
            sid: run_id
            for sid, run_id in self._chosen.items()
            if run_id in {r for r, _label in self._options.get(sid, [])}
        }
        changed = kept != self._chosen
        self._chosen = kept
        self._refresh()
        if changed:
            self.changed.emit()

    def set_chosen(self, chosen: dict[str, str]) -> None:
        """Set the per-recording runs, as the dialog does on OK."""
        if chosen == self._chosen:
            return
        self._chosen = dict(chosen)
        self._refresh()
        self.changed.emit()

    def open_dialog(self) -> None:
        dialog = RunOverridesDialog(
            self.module_id,
            self._options,
            self._names,
            self._chosen,
            self.widget.window(),
        )
        if dialog.exec():
            self.set_chosen(dialog.chosen())

    def chosen(self) -> dict[str, str]:
        """``{recording_id: run id}`` — the recordings given a run of their own."""
        return dict(self._chosen)

    def selection(self) -> dict:
        """``<upstream>_overrides`` when any recording has a run of its own, else nothing."""
        return {f"{self.module_id}_overrides": self.chosen()} if self._chosen else {}

    def _refresh(self) -> None:
        self.button.setEnabled(bool(self._options))
        self.button.setToolTip(
            self._tip
            if self._options
            else f"No recording has a completed {self.module_id} run to choose from."
        )
        if not self._chosen:
            self.note.setText("")
            return
        shown = [
            f"{self._names.get(sid, sid)} → {run_id}"
            for sid, run_id in sorted(self._chosen.items())
        ]
        more = f" (+{len(shown) - 3} more)" if len(shown) > 3 else ""
        self.note.setText(f"Chosen per recording: {', '.join(shown[:3])}{more}")


class PooledSpecPicker(QObject):
    """An *analysis* form's upstream picker: one checkable row per set of settings across the
    selection, each with its ``N/M`` coverage, and **several may be ticked** (§6.5).

    :class:`UpstreamSpecPicker`'s analysis twin. Recordings whose upstream came through
    different chains — two strains, each detected by its own model — have no one set of
    settings in common, so a single choice could only ever pool half of them. Ticking each
    group pools them all: :meth:`selection` emits ``<upstream>_specs``, which
    ``run_spec.resolve_analysis_selection`` resolves per recording, refusing one with a run of
    two ticked groups rather than letting list order pick, and checking pooled embeddings are
    comparable. One tick emits ``<upstream>_spec``, the record shape every earlier run wrote.

    A separate class, not a mode of :class:`UpstreamSpecPicker`, because a *processing* form's
    spec is resolved per recording by ``run_spec.resolve_into_config``, which takes exactly
    one: a multi-select there would offer a choice the run cannot honour.

    Below the list sits a :class:`RunOverrides` (:attr:`overrides`): a recording given a run
    of its own reads that run, whatever the ticks resolve to, and a recording with a run of two
    ticked settings is no longer refused once it has one.

    Same ``resolve`` (off the UI thread) / ``apply`` (UI thread, no disk) split. The form puts
    :attr:`panel` (the list and the per-recording row) in its layout, connects
    :attr:`changed`, and merges :meth:`selection`. :attr:`widget` is the list itself.
    """

    #: A tick changed (the item) or a per-recording choice did (``None``).
    changed = pyqtSignal(object)

    def __init__(self, module_id: str) -> None:
        super().__init__()
        self.module_id = module_id
        self.widget = QListWidget()
        self.widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        # A lineage label opens with what every row shares (the embedding recipe) and ends with
        # what tells them apart (the detection model it descends from). Clipped at the right,
        # two strains' rows read identically; elided in the middle, both ends stay on screen.
        self.widget.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.widget.itemChanged.connect(self.changed.emit)
        self.overrides = RunOverrides(module_id)
        self.overrides.changed.connect(lambda: self.changed.emit(None))
        self.panel = QWidget()
        column = QVBoxLayout(self.panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.widget)
        column.addWidget(self.overrides.widget)
        self._groups: list[dict] = []
        self._names: dict[str, str] = {}

    def resolve(self, recordings: list, model_names: dict) -> list[dict]:
        """Off the UI thread: the groups for ``recordings``. Pinned, so the read cache's
        per-property ``stat`` is paid once per recording, not once per field. ``model_names``
        has no default, for the reason :meth:`UpstreamSpecPicker.resolve` gives."""
        from detail.storage import read_cache

        with read_cache.pinned():
            return spec_groups(recordings, self.module_id, model_names)

    def apply(self, groups: list[dict], total: int, *, names: dict[str, str]) -> None:
        """UI thread: install ``groups``. Keeps the previous ticks still on offer; when none
        are, ticks the widest coverage, so a recording missing a run does not decide what the
        other forty-seven are offered first. ``names`` (:func:`recording_names`, resolved off
        the UI thread) is what the per-recording choices are listed under."""
        self._names = dict(names)
        previous = {run_spec.freeze_spec(s) for s in self.specs()}
        self._groups = groups
        ticked = [run_spec.freeze_spec(g["spec"]) in previous for g in groups]
        if groups and not any(ticked):
            ticked[groups.index(widest_group(groups))] = True
        self.widget.blockSignals(True)
        self.widget.clear()
        for group, on in zip(groups, ticked):
            # Coverage leads, because the middle of the label is what elides away.
            item = QListWidgetItem(
                f"{len(group['recordings'])}/{total} recordings  ·  {group['label']}"
            )
            item.setToolTip(group["label"])
            item.setData(Qt.ItemDataRole.UserRole, group["spec"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.widget.addItem(item)
        self.widget.blockSignals(False)
        fit_list_height(self.widget, max_rows=6)
        # Last: dropping a choice no longer on offer emits `changed`, and a form's handler
        # then reads the list just installed rather than the one it replaced.
        self.overrides.set_options(groups, names)

    def specs(self) -> list[dict]:
        """The ticked settings, in list order."""
        items = (self.widget.item(i) for i in range(self.widget.count()))
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in items
            if item.checkState() == Qt.CheckState.Checked
        ]

    def selection(self) -> dict:
        """The config keys the ticks become — ``<upstream>_spec`` for one, ``<upstream>_specs``
        for several, nothing for none — and ``<upstream>_overrides`` when a recording has a
        run of its own."""
        specs = self.specs()
        if len(specs) == 1:
            ticks = {f"{self.module_id}_spec": specs[0]}
        else:
            ticks = {f"{self.module_id}_specs": specs} if specs else {}
        return {**ticks, **self.overrides.selection()}

    def covered(self) -> set[str] | None:
        """The selected recordings the ticked settings or a per-recording choice cover;
        ``None`` (nothing narrowed yet) when none are ticked — never the empty set, for the
        reason :meth:`UpstreamSpecPicker.covered` gives."""
        specs = self.specs()
        if not specs:
            return None
        return set(self.overrides.chosen()).union(
            *(covered_ids(self._groups, spec) for spec in specs)
        )

    def overlap(self) -> list[str]:
        """Recording ids with a run of more than one ticked setting and no run chosen for
        them — what the run refuses, so the form can say so before launch."""
        chosen = self.overrides.chosen()
        counts: dict[str, int] = {}
        for spec in self.specs():
            for sid in covered_ids(self._groups, spec):
                counts[sid] = counts.get(sid, 0) + 1
        return sorted(sid for sid, n in counts.items() if n > 1 and sid not in chosen)

    def validation_error(self) -> str | None:
        """Why the ticks cannot launch, or ``None``."""
        if not self.specs():
            return f"These recordings have no completed {self.module_id} run."
        both = [self._names.get(sid, sid) for sid in self.overlap()]
        if both:
            more = " …" if len(both) > 3 else ""
            return (
                f"{', '.join(both[:3])}{more} have a run of more than one ticked "
                f"{self.module_id} setting. Tick only one of those settings, or choose "
                "their run with Per recording…"
            )
        return None
