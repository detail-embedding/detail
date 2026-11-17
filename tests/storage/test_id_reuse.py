"""A deleted thing's id is never handed to the next thing of its kind.

**The class.** An identity minted as "one past the highest that still exists" (or the lowest free
number, or a list position), so a delete or a reorder hands it to a different thing while records
elsewhere still name it. Deleting the newest model and training another gave the new one the old
one's id, and a finished detection run that recorded ``model_id: model_002`` was relabelled with
the new model's name; a distill record's ``example_set.id`` came to name a different set; a Quick
segment run's export landed in a deleted run's folder; and the scene editor renumbered every region
on each Save, so ``roi_001`` named ``nest`` in one location run and ``food`` in the next.

**How it hid.** Every allocator was correct while nothing was deleted, and every test of one
minted, created and minted again. The id only goes wrong after a delete of the *newest* member,
and nothing reads wrong at the line that mints it: the wrong answer appears later, in a record
written long before, which still parses.

**The rule.** Every family mints through ``storage/id_allocator.py``: a high-water mark beside the
family that only goes up, claimed under a lock, so neither a delete nor two processes racing can
hand out an id twice. A family that does not (recordings, which keep their own counter in
``config.yaml``; arenas; snapshots; a clustering run's labels) is listed in ``_NOT_SHARED`` with
its reason. DESIGN_PLAN §3 *Ids are never reused* states the rule.

Two guards:

- ``test_a_deleted_id_is_never_minted_again`` runs every family in ``_FAMILIES`` through its real
  mint and its real delete. A family that joins the table is covered the day it joins.
- ``test_every_allocator_is_the_shared_one_or_says_why_not`` is a fail-closed source scan: every
  ``def next_*_id`` / ``reserve_*_id`` / ``_reserve_id`` / ``mint`` and every
  ``max(..., default=0) + 1`` under ``detail/`` either goes through the shared allocator or is in
  ``_NOT_SHARED``.
"""

import ast
import re
from pathlib import Path

import pytest

import detail
from detail.api.recording import Recording
from detail.core.skeleton import Skeleton
from detail.storage import analysis_runs, comparisons, datasets, distill_runs
from detail.storage import example_sets, export_runs, id_allocator, models, runs
from detail.storage import skeleton_library, spatial, tool_runs

_SKELETON = Skeleton(names=["a", "b"], edges=[("a", "b")])
_SQUARE = [(0, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])]


def _record_dir(directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text("schema_version: 1\n", encoding="utf-8")


def _define(kind: str):
    def mint(exp: Path) -> str:
        rec = exp / "recordings" / "recording_001"
        def_id = spatial.next_id(rec, kind)
        keyframes = _SQUARE if kind == "roi" else [(0, [(0.5, 0.5)])]
        n = spatial.write_keyframes(rec, kind, def_id, keyframes)
        spatial.append_to_manifest(rec, kind, def_id, f"{kind} {def_id}", n)
        return def_id

    return mint


def _reserve_detection(exp: Path) -> str:
    recording = Recording(exp, "recording_001")
    return recording._reserve_run(
        "detection_runs", "det", lambda rid: {"id": rid, "status": "complete"}
    )


def _mint_model(exp: Path) -> str:
    model_id = models.next_model_id(exp)
    _record_dir(models.model_dir(exp, model_id), models.MODEL_FILENAME)
    return model_id


def _mint_dataset(exp: Path) -> str:
    dataset_id = datasets.next_dataset_id(exp)
    _record_dir(datasets.dataset_dir(exp, dataset_id), datasets.DATASET_FILENAME)
    return dataset_id


def _mint_run(exp: Path) -> str:
    with runs.runs_lock(exp):
        run_id = runs.next_run_id(exp)
        runs.write_run(exp, run_id, {"id": run_id})
    return run_id


def _mint_distill(exp: Path) -> str:
    run_id = distill_runs.next_distill_id(exp)
    distill_runs.write_run(exp, run_id, {"id": run_id})
    return run_id


def _mint_analysis(exp: Path) -> str:
    run_id = analysis_runs.reserve_analysis_id(exp)
    _record_dir(
        analysis_runs.analysis_dir(exp, run_id), analysis_runs.ANALYSIS_FILENAME
    )
    return run_id


def _mint_export(exp: Path) -> str:
    run_id = export_runs.reserve_export_id(exp)
    _record_dir(export_runs.export_dir(exp, run_id), export_runs.EXPORT_FILENAME)
    return run_id


def _mint_comparison(exp: Path) -> str:
    cid = comparisons.next_comparison_id(exp)
    _record_dir(comparisons.comparison_dir(exp, cid), comparisons.COMPARISON_FILENAME)
    return cid


def _mint_tool_run(_exp: Path) -> str:
    run_id = tool_runs.next_run_id("quick_segment")
    tool_runs.write_tool_run("quick_segment", run_id, {"id": run_id})
    return run_id


#: family -> (mint one and write its record, delete it through the real delete function)
_FAMILIES = {
    "model": (_mint_model, models.delete_model),
    "dataset": (_mint_dataset, datasets.delete_dataset),
    "processing run": (_mint_run, runs.delete_run),
    "distill run": (_mint_distill, distill_runs.delete_run),
    "analysis run": (_mint_analysis, analysis_runs.delete_analysis),
    "export run": (_mint_export, export_runs.delete_export),
    "comparison": (_mint_comparison, comparisons.delete_comparison),
    "tool run": (_mint_tool_run, lambda _exp, rid: tool_runs.delete_tool_run(rid)),
    "skeleton": (
        lambda _exp: skeleton_library.create("s", _SKELETON)["id"],
        lambda _exp, sid: skeleton_library.remove(sid),
    ),
    "example set": (
        lambda exp: example_sets.create_set(exp, "set", _SKELETON).id,
        example_sets.delete_set,
    ),
    "region": (
        _define("roi"),
        lambda exp, rid: spatial.delete_def(
            exp / "recordings" / "recording_001", "roi", rid
        ),
    ),
    "point": (
        _define("point"),
        lambda exp, pid: spatial.delete_def(
            exp / "recordings" / "recording_001", "point", pid
        ),
    ),
    "per-recording run": (
        _reserve_detection,
        lambda exp, rid: Recording(exp, "recording_001").delete_run("detection", rid),
    ),
}


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_a_deleted_id_is_never_minted_again(recording_dir, family):
    exp = recording_dir.parent.parent
    mint, delete = _FAMILIES[family]
    mint(exp)
    newest = mint(exp)
    delete(exp, newest)
    after = mint(exp)
    assert after != newest, f"{family}: {newest} was deleted and handed out again"


def test_a_region_kept_by_save_keeps_its_id_and_a_new_one_never_takes_a_freed_id(
    recording_dir,
):
    """The scene editor's Save (``replace_defs``) hands back each surviving region's id."""
    nest = spatial.next_id(recording_dir, "roi")
    spatial.replace_defs(
        recording_dir,
        "roi",
        [
            {"name": "nest", "keyframes": _SQUARE},
            {"name": "food", "keyframes": _SQUARE},
        ],
    )
    by_name = {e["name"]: e["id"] for e in spatial.list_defs(recording_dir, "roi")}
    assert nest not in by_name.values()  # the id claimed above stays spent
    spatial.replace_defs(
        recording_dir,
        "roi",
        [
            {"id": by_name["food"], "name": "food", "keyframes": _SQUARE},
            {"name": "shelter", "keyframes": _SQUARE},
        ],
    )
    after = {e["name"]: e["id"] for e in spatial.list_defs(recording_dir, "roi")}
    assert after["food"] == by_name["food"]
    assert after["shelter"] not in {nest, by_name["nest"], by_name["food"]}


# --------------------------------------------------------------------------- #
# the allocator itself
# --------------------------------------------------------------------------- #
def test_an_existing_family_starts_past_the_highest_id_on_disk(tmp_path):
    """No stored id changes: a family made before the mark existed continues from its
    highest id, and a hand-made directory past the mark is stepped over."""
    (tmp_path / "model_007").mkdir()
    assert id_allocator.claim_dir(tmp_path, "model", digits=3) == "model_008"
    (tmp_path / "model_020").mkdir()
    assert id_allocator.claim_dir(tmp_path, "model", digits=3) == "model_021"


def test_the_mark_survives_deleting_every_member(tmp_path):
    first = id_allocator.claim_dir(tmp_path, "ex", digits=2)
    (tmp_path / first).rmdir()
    assert id_allocator.claim_dir(tmp_path, "ex", digits=2) == "ex_02"


def test_peek_predicts_without_spending(tmp_path):
    assert id_allocator.peek(tmp_path, "an", digits=2) == "an_01"
    assert id_allocator.peek(tmp_path, "an", digits=2) == "an_01"
    assert id_allocator.claim_dir(tmp_path, "an", digits=2) == "an_01"
    assert id_allocator.peek(tmp_path, "an", digits=2) == "an_02"


def test_stems_sharing_a_home_count_separately(tmp_path):
    assert id_allocator.claim(tmp_path, "roi", digits=3, taken=[]) == "roi_001"
    assert id_allocator.claim(tmp_path, "point", digits=3, taken=[]) == "point_001"
    assert id_allocator.claim(tmp_path, "roi", digits=3, taken=[]) == "roi_002"


def test_an_unreadable_mark_falls_back_to_what_is_on_disk(tmp_path):
    (tmp_path / "run_004").mkdir()
    (tmp_path / id_allocator.MARK_FILENAME).write_text("[not, a, mapping", "utf-8")
    assert id_allocator.claim_dir(tmp_path, "run", digits=3) == "run_005"


# --------------------------------------------------------------------------- #
# the source scan
# --------------------------------------------------------------------------- #
_DETAIL = Path(detail.__file__).parent

#: (file under detail/, function) -> why it does not mint through ``id_allocator``.
_NOT_SHARED = {
    ("storage/experiment.py", "next_recording_id"): (
        "the model the shared allocator copies: its own counter in config.yaml, never "
        "reused; the issue that made the shared one says not to change how recording ids "
        "are minted"
    ),
    ("storage/spatial.py", "next_arena_id"): (
        "every reference to an arena id is a tracking run, and while any exists the arena "
        "cannot be deleted or redefined (Recording.delete_arena / clear_arenas refuse), so a "
        "freed arena id is named by nothing; the scene editor redefines the whole set on "
        "Save and relies on the lowest free number handing the same ids back"
    ),
    ("core/examples.py", "next_round"): (
        "not an id: an audit set's draw-round number, named only by that set's own frames, "
        "and no record outside the set refers to a round"
    ),
    ("storage/snapshots.py", "next_snapshot_id"): (
        "nothing deletes a snapshot, so no snapshot id is ever freed"
    ),
    ("storage/run_labels.py", "mint"): (
        "a clustering run's label ids (L<n>) live in the run's own state; never reusing one "
        "after the highest is deleted needs a high-water field in that state, left for after "
        "the label-editing rework (#24) lands"
    ),
}

_ALLOCATOR_DEF = re.compile(
    r"^(next_\w*id|reserve_\w*id|_reserve_id|_next_run_id|mint)$"
)
_MAX_PLUS_ONE = re.compile(r"default=0\)\s*\+\s*1\b")


def _offenders(source: str, filename: str) -> list[str]:
    """Allocators in ``source`` that neither call ``id_allocator`` nor are listed.

    Pure over the text, so the negative cases can hand it a literal.
    """
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.get_source_segment(source, node) or ""
        mints = _ALLOCATOR_DEF.match(node.name) or _MAX_PLUS_ONE.search(body)
        if not mints or (filename, node.name) in _NOT_SHARED:
            continue
        if "id_allocator." not in body:
            out.append(f"{filename}::{node.name}")
    return out


def test_every_allocator_is_the_shared_one_or_says_why_not():
    offenders = []
    for path in sorted(_DETAIL.rglob("*.py")):
        rel = path.relative_to(_DETAIL).as_posix()
        if rel == "storage/id_allocator.py":
            continue
        offenders += _offenders(path.read_text(encoding="utf-8-sig"), rel)
    assert not offenders, (
        "these mint an id without storage/id_allocator.py, so deleting the newest member "
        "hands its id to the next one; route them through it or list them in _NOT_SHARED "
        f"with the reason: {offenders}"
    )


def test_every_exemption_still_names_a_function():
    """An entry that excuses nothing is removed, so the list cannot grow stale."""
    missing = []
    for filename, name in _NOT_SHARED:
        source = (_DETAIL / filename).read_text(encoding="utf-8-sig")
        if not re.search(rf"^\s*def {re.escape(name)}\(", source, re.M):
            missing.append((filename, name))
    assert not missing


def test_the_scan_would_catch_a_highest_plus_one_allocator():
    source = (
        "def next_widget_id(root):\n"
        "    nums = [int(p.name[7:]) for p in root.iterdir()]\n"
        "    return f'widget_{max(nums, default=0) + 1:03d}'\n"
    )
    assert _offenders(source, "storage/widgets.py") == [
        "storage/widgets.py::next_widget_id"
    ]


def test_the_scan_would_catch_an_unnamed_function_that_formats_max_plus_one():
    source = (
        "def make(labels):\n"
        "    return f'L{max((int(k[1:]) for k in labels), default=0) + 1}'\n"
    )
    assert _offenders(source, "storage/labels.py") == ["storage/labels.py::make"]


def test_the_scan_passes_an_allocator_that_uses_the_shared_one():
    source = (
        "def next_widget_id(root):\n"
        "    return id_allocator.claim_dir(root, 'widget', digits=3)\n"
    )
    assert _offenders(source, "storage/widgets.py") == []
