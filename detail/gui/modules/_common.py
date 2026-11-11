"""Helpers for config widgets that reference a per-recording upstream run.

A batch run passes one config to every selected recording, but upstream run ids
(and arena ids) are per-recording. These return the ids present in *every* selected
recording — the ones a batch run can reference uniformly across the selection.
"""

import re

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QComboBox, QListWidget, QListWidgetItem

from detail.api import run_spec
from detail.gui.sizing import fit_list_height

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
    arena twin of :func:`spec_items`."""
    items = []
    for group in groups:
        covered = len(group["recordings"])
        suffix = f"  ·  {covered}/{total} recordings" if total else ""
        items.append((f"{group['name']}{suffix}", group["name"]))
    return items


def arena_covered_ids(groups: list[dict], name: str | None) -> set[str] | None:
    """Recording ids that have an arena called ``name`` — ``None`` (all) when ``name`` is
    ``None``, which is the whole-frame choice, and an empty set when it matches none."""
    if name is None:
        return None
    for group in groups:
        if group["name"] == name:
            return set(group["recordings"])
    return set()


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
    selected recordings (``recordings`` are the covered recording ids)."""
    return run_spec.group_specs(recordings, module_id, model_names)


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


def repopulate_spec_combo(combo: QComboBox, items: list[tuple[str, dict]]) -> None:
    """Refill a combo with ``(label, spec)`` pairs, keeping the prior spec selection
    (matched by ``freeze_spec`` — spec dicts are rebuilt each refresh, so identity
    comparison would not find them)."""
    previous = combo.currentData()
    prev_key = run_spec.freeze_spec(previous) if isinstance(previous, dict) else None
    combo.clear()
    for label, spec in items:
        combo.addItem(label, spec)
    if prev_key is None:
        return
    for index in range(combo.count()):
        data = combo.itemData(index)
        if isinstance(data, dict) and run_spec.freeze_spec(data) == prev_key:
            combo.setCurrentIndex(index)
            return


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
    """

    def __init__(self, module_id: str, *, auto_label: str | None = None) -> None:
        self.module_id = module_id
        self.combo = QComboBox()
        self._groups: list[dict] = []
        self._preferred: dict | None = None
        self._auto_label = auto_label
        if auto_label is not None:
            self.combo.addItem(auto_label, None)

    @property
    def _offset(self) -> int:
        return 0 if self._auto_label is None else 1

    def resolve(self, recordings: list, model_names: dict | None = None) -> list[dict]:
        """Off the UI thread: the groups for ``recordings``. Pinned, so the read cache's
        per-property ``stat`` is paid once per recording, not once per field."""
        from detail.storage import read_cache

        with read_cache.pinned():
            return spec_groups(recordings, self.module_id, model_names)

    def apply(self, groups: list[dict], total: int) -> None:
        """UI thread: install ``groups``. Keeps the previous choice when it is still on
        offer; otherwise selects the widest coverage, so a recording missing a run does
        not decide what the other forty-seven are offered first."""
        previous = self.spec()
        self._groups = groups
        items = spec_items(groups, total)
        if self._auto_label is not None:
            items = [(self._auto_label, None)] + list(items)
        repopulate_spec_combo(self.combo, items)
        if self._preferred is not None and self._select(self._preferred):
            return
        if self._auto_label is not None:
            if previous is None or not self._select(previous):
                self.combo.setCurrentIndex(0)
            return
        current = self.spec()
        if previous is not None and current is not None:
            if run_spec.freeze_spec(current) == run_spec.freeze_spec(previous):
                return
        if groups:
            widest = max(groups, key=lambda g: len(g["recordings"]))
            self.combo.setCurrentIndex(groups.index(widest))

    def prefer(self, spec: dict) -> None:
        """Select ``spec`` whenever it is on offer (a next-step launch from that run), in
        place of the widest-coverage default."""
        self._preferred = spec
        self._select(spec)

    def _select(self, spec: dict) -> bool:
        key = run_spec.freeze_spec(spec)
        for index, group in enumerate(self._groups):
            if run_spec.freeze_spec(group["spec"]) == key:
                self.combo.setCurrentIndex(index + self._offset)
                return True
        return False

    def spec(self) -> dict | None:
        data = self.combo.currentData()
        return data if isinstance(data, dict) else None

    def covered(self) -> set[str] | None:
        """The selected recordings the chosen settings cover; ``None`` (nothing narrowed
        yet) when no settings are chosen.

        Never the empty set for "nothing chosen": the launch dialog greys every recording
        outside this set, the groups are computed from the *checked* recordings, and the
        recordings start unchecked, so an empty answer greyed every row before one could be
        checked and the form could never be filled. A selection with no matching run is
        refused by the form's ``validation_error`` instead, where it can say why.
        """
        spec = self.spec()
        return None if spec is None else covered_ids(self._groups, spec)


def pooled_settings_note(n_settings: int) -> str:
    """`` from N embedding settings`` when a fit pools several, else nothing — so the advice
    line says out loud that the recordings came through different chains."""
    return f" from {n_settings} embedding settings" if n_settings > 1 else ""


class PooledSpecPicker:
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

    Same ``resolve`` (off the UI thread) / ``apply`` (UI thread, no disk) split. The form puts
    :attr:`widget` in its layout, connects :attr:`changed`, and merges :meth:`selection`.
    """

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self.widget = QListWidget()
        self.widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        # A lineage label opens with what every row shares (the embedding recipe) and ends with
        # what tells them apart (the detection model it descends from). Clipped at the right,
        # two strains' rows read identically; elided in the middle, both ends stay on screen.
        self.widget.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.changed = self.widget.itemChanged
        self._groups: list[dict] = []

    def resolve(self, recordings: list, model_names: dict | None = None) -> list[dict]:
        """Off the UI thread: the groups for ``recordings``. Pinned, so the read cache's
        per-property ``stat`` is paid once per recording, not once per field."""
        from detail.storage import read_cache

        with read_cache.pinned():
            return spec_groups(recordings, self.module_id, model_names)

    def apply(self, groups: list[dict], total: int) -> None:
        """UI thread: install ``groups``. Keeps the previous ticks still on offer; when none
        are, ticks the widest coverage, so a recording missing a run does not decide what the
        other forty-seven are offered first."""
        previous = {run_spec.freeze_spec(s) for s in self.specs()}
        self._groups = groups
        ticked = [run_spec.freeze_spec(g["spec"]) in previous for g in groups]
        if groups and not any(ticked):
            widest = max(groups, key=lambda g: len(g["recordings"]))
            ticked[groups.index(widest)] = True
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

    def specs(self) -> list[dict]:
        """The ticked settings, in list order."""
        items = (self.widget.item(i) for i in range(self.widget.count()))
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in items
            if item.checkState() == Qt.CheckState.Checked
        ]

    def selection(self) -> dict:
        """The config key the ticks become: ``<upstream>_spec`` for one, ``<upstream>_specs``
        for several, nothing for none."""
        specs = self.specs()
        if len(specs) == 1:
            return {f"{self.module_id}_spec": specs[0]}
        return {f"{self.module_id}_specs": specs} if specs else {}

    def covered(self) -> set[str] | None:
        """The selected recordings the ticked settings cover; ``None`` (nothing narrowed yet)
        when none are ticked — never the empty set, for the reason
        :meth:`UpstreamSpecPicker.covered` gives."""
        specs = self.specs()
        if not specs:
            return None
        return set().union(*(covered_ids(self._groups, spec) for spec in specs))

    def overlap(self) -> list[str]:
        """Recording ids with a run of more than one ticked setting — what the run refuses, so
        the form can say so before launch."""
        counts: dict[str, int] = {}
        for spec in self.specs():
            for sid in covered_ids(self._groups, spec):
                counts[sid] = counts.get(sid, 0) + 1
        return sorted(sid for sid, n in counts.items() if n > 1)

    def validation_error(self) -> str | None:
        """Why the ticks cannot launch, or ``None``."""
        if not self.specs():
            return f"These recordings have no completed {self.module_id} run."
        both = self.overlap()
        if both:
            more = " …" if len(both) > 3 else ""
            return (
                f"{', '.join(both[:3])}{more} have a run of more than one ticked "
                f"{self.module_id} setting. Tick only one of those settings."
            )
        return None
