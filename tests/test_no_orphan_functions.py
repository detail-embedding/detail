"""A public function with tests and no caller is a countermeasure that isn't protecting anything.

The recurring shape behind several bugs this codebase has already paid for: someone builds the
right thing, leaves the last wire off, and it is invisible from then on because nothing fails.

- ``cross_seed_stability`` measured whether a clustering survives resampling — exactly the
  instability that made one configuration look like the best method when it was the luckiest
  split. Defined, unit-tested, called by nothing, for as long as it existed.
- ``query_eval_run`` was one module's in-process scripting entry point. Its two siblings are
  wired (``query_clips`` has seven callers and drives the GUI preview); this one had no caller
  and no test, and predated the package rename.
- ``discover_tool_runs`` was superseded by ``list_tool_runs`` and left behind.

None of these fail a test, break a build, or show up in review — the code is correct, it just
does not run. So this scans for the shape instead.

**What it does not flag**, because these are wired by something other than a call:
decorated functions (``@register_*``, ``@click.command`` — the registry *is* the caller),
everything in ``cli.py`` for the same reason, private ``_`` helpers, and anything ``scripts/``
or another caller root uses (doc-generation scripts and drivers are legitimate consumers).

**Methods, for a named set of classes.** The same shape outlived three GUI deletions in one
branch (2026-11-09): the montage window, the export dialog's page section and the cluster card's
breadcrumb went, and their helpers stayed on ``ClusterMontage``, ``NodeView``, ``NodeIcicle``,
``CollapsibleBand``, ``AuthoredRun`` and ``FilterBar`` — invisible to a scan of module-level
defs. :data:`_METHOD_SCANNED_CLASSES` says why the method scan is scoped rather than whole-tree.

The allow-list is claims, not exemptions: each entry says why that function has no caller. It
is checked against the live scan, so an entry for a function that has since been wired — or
deleted — fails rather than sitting there excusing nothing.
"""

import ast
import collections
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Folders beside ``scripts/`` whose code is a legitimate caller of ``detail``.
_EXTRA_CALLER_ROOTS = ()

# Each entry is a claim about why this function is uncalled. Three honest categories; none of
# them is "we don't know", which is what the guard exists to stop accumulating.
_ORPHANS_WITH_REASONS = {
    "match_curve": "called by evaluation drivers kept outside this repository",
    "smooth_min_run": "called by evaluation drivers kept outside this repository",
    "compute_metrics": "called by evaluation drivers kept outside this repository",
    "detect_curve_transitions": "called by evaluation drivers kept outside this repository",
    "recording_frame_labels": "called by evaluation drivers kept outside this repository",
    "labels_to_bout_rows": "called by evaluation drivers kept outside this repository",
    "finch_hierarchy_from_vectors": "called by evaluation drivers kept outside this repository",
    "cluster_labels": "called by evaluation drivers kept outside this repository",
    "small_multiples": "called by evaluation drivers kept outside this repository",
    "agreement_panel": "called by evaluation drivers kept outside this repository",
    # --- a registration point waiting for its first implementation -------------------- #
    "register_stream_consumer": "STREAM_CONSUMERS is empty by design (CLAUDE.md); this is the "
    "hook a first consumer will register through",
    # --- the earlier one-by-one audit's answer vocabulary ------------------------------ #
    # AUDIT's one-by-one check was replaced by clip review (every clip of a cluster); its
    # `audit.csv` stays readable and `label_audit(answer=)` still takes these answers, so what an
    # answer means stays defined and tested (tests/storage/test_behavior_audit.py).
    "answer_options": "the one-by-one audit's answers for a scope; no GUI asks them since clip "
    "review replaced the check, and label_audit(answer=) still takes them",
    "human_label_for": "what a one-by-one audit answer says about the clip's own label; read "
    "for audit.csv rows and label_audit(answer=), no GUI since clip review",
    # --- built and tested, no surface reaches it yet ---------------------------------- #
    "resolve_track_id": "DESIGN_PLAN §13's animal-id selector; every config types track_id as "
    "int, so the str branch is unreachable. Real gap, tracked in docs/clustering_experiments.md",
    "module_topic": "help-system lookup; field_topic is wired, this one is not yet",
    "anchor_exists": "help-system link validation, used by the docs tests, not yet by a surface",
    "marked_capabilities": "reads the §25 prose markers back out of a docs page; the "
    "surfaces guard is its only caller by design — the build strips blocks rather than "
    "enumerating them, so nothing in detail/ needs to ask which are marked",
    "doc_url": "help-system link builder, same",
    "topic_body": "help-system topic renderer, same",
    "apply_alignment": "constant-offset stream alignment; raises NotImplementedError for the "
    "unbuilt §3 correction table, so it is half a feature that declares itself",
    "all_targets": "the export registry sweep's enumeration seam "
    "(tests/api/test_export_registry.py). Deliberately unfiltered, which is why no surface "
    "can use it: a surface asks targets_for(subject), which drops what has no inputs on "
    "disk — and the sweep has to check the *declarations* of targets whose subject does not "
    "exist yet",
    "fit_table_width": "one half of `gui_design` §3's table rule — a table fills its pane or "
    "hugs its content — and the half `tests/gui/test_table_width.py` lets a table declare "
    "itself by. Its one caller was the audit tab's two summaries, which turned out to want the "
    "other half: capped to ~200px of content they left the rest of a 400px column blank beside "
    "them. Kept rather than deleted because the rule is still two-sided and this is the only "
    "way to express one side; if a second release passes with no caller, delete it and say so "
    "in that guard",
    # --- diagnostic accessors on the deploy toggle (§25) ------------------------------ #
    "hidden_names": "the parity guard's enumeration seam (tests/test_surfaces.py). No runtime "
    "caller by design: surfaces ask is_surfaced() about one name, and only a guard needs the "
    "whole set",
    "hidden_reason": "answers why a surface is absent — for a developer running with "
    "DETAIL_SHOW_ALL, where a revealed surface still reports why it is normally hidden. No "
    "user-facing surface shows it, because a hidden surface has nowhere to show it",
    # --- the §23 measures layer, landed with §24 Comparison as its consumer ----------- #
    # `measure_run`, `absolute` and `sliding` came off this list when §24's record wired them.
    # --- a shared artifact whose producer has landed ------------------------------------ #
    # DESIGN_PLAN §19: the behavior output is written *once* and every analysis module that
    # produces labeled behavior emits it. §20's Query is that first producer, so the writer,
    # the frame view, its observed mask and the stream projection all came off this list when
    # `pipeline/query.py` grew a reprojection. What remains is discovery-by-scanning, which
    # needs a surface that *lists* artifacts across an experiment rather than one that writes
    # one — that arrives with the calibration workspace.
    # The §19 reliability audit. Auditing is an optional step a user drives from the AUDIT tab
    # of §20's calibration workspace. The workspace shipped (#218/#222); that tab did not, so
    # these have their storage layer and their tests and no caller until it does.
    # `append_audit` came off this list when `AuthoredRun.audit_sample` / `label_audit` wired
    # it — which is the second test below doing its job, not a coincidence worth trusting.
    "draw_audit_sample": "the fixed per-bin draw, superseded by `draw_sample`'s budget for "
    "every caller; kept because runs audited before the budget existed were drawn with it, "
    "and its bin semantics are what those samples mean",
    # --- a reference implementation the fast path is checked against ------------------ #
    # Kept deliberately: `_dtw_numba` documents itself as "the compiled form of the loop in
    # retrieval.dtw_distance (bit-identical)", and the tests compare the two. Deleting the
    # readable version would leave the compiled one unverifiable.
    "dtw_distance": "the pure-Python DTW the numba path is checked bit-identical against",
    "cosine_cost": "the readable cost surface `_dtw_numba` states it reproduces",
    "match_ranked": "the reference ranked matcher the other scorers are described in terms of",
    # --- superseded, kept until someone confirms nothing external calls it ------------- #
    "next_analysis_id": "same, one registry over: superseded by reserve_analysis_id after two "
    "concurrent analysis runs took the same an_NN and the second overwrote the first",
    # --- public API offered for scripting, correct and delegating --------------------- #
    "run_check": "declared in api/checks.__all__ as public onboarding API",
    "readiness_signals": "the §1 onboarding readiness entry point, public API",
    "run_recording_checks": "per-recording integrity checks; api/checks.py:62 says they are "
    "'surfaced separately via run_recording_checks', and that surface is not built yet",
    "query_subsequence": "the single-query case of query_multi, which is what production calls",
    "t_ns_to_seconds": "unit helper on the public core API",
    "experiment_path": "path helper on the public storage API",
    "box_map": "detection-row helper on the public clip-render API",
    "read_location_csv": "the reader half of the location-export format, for consumers of it",
    # --- the floor a rule names, with no report to ship it in ---------------------------- #
    # The behavior clusterers were removed (docs/removed_capabilities.md, section 2) and the
    # shared functions only they called were deleted after them. This one stays because
    # CLAUDE.md names it as the worked example of "a headline metric ships with its floor" and
    # tests/analysis/test_cluster_floor.py is that rule's guard.
    "one_cluster_floor": "the per-behavior floor for a clustering score; its only caller was "
    "the removed clusterer evaluation's report. Kept as the reference CLAUDE.md's floor rule "
    "points at, with its guard",
    # Pose clustering was removed (section 12) and is expected back: its registry stays, empty.
    "register_pose_cluster_method": "the pose-cluster-method registry's decorator; the registry "
    "stays, empty, for pose clustering's return (docs/removed_capabilities.md, section 12)",
    "feature_rows": "its only caller was the removed pose clustering (section 12); kept for "
    "its return",
}


def _orphans() -> dict[str, str]:
    """Public, undecorated functions in ``detail/`` that nothing in ``detail/`` or ``scripts/``
    calls. Returns ``{name: "file:line"}``.

    References are collected from the **AST**, not by matching words in the text. Counting
    tokens over raw source lets a docstring or comment that merely *names* a function count as
    using it — which happened: writing "``resolve_track_id`` is the designed way out" in a new
    docstring silently un-flagged it. Prose about a function is the opposite of a caller, so
    only real ``Name`` / ``Attribute`` nodes count.
    """

    def referenced(root: pathlib.Path) -> collections.Counter:
        names: collections.Counter = collections.Counter()
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Name):
                    names[node.id] += 1
                elif isinstance(node, ast.Attribute):
                    names[node.attr] += 1
        return names

    src = {
        p: p.read_text(encoding="utf-8-sig") for p in (REPO / "detail").rglob("*.py")
    }
    in_detail = referenced(REPO / "detail")
    # A second caller root holds sweep drivers, moved out of `scripts/` so the tool's own
    # utilities and those drivers stop sharing a directory. Both are legitimate consumers;
    # leaving it out made every function only they call look orphaned.
    in_scripts = referenced(REPO / "scripts")
    in_research = collections.Counter()
    for root in _EXTRA_CALLER_ROOTS:
        in_research += referenced(REPO / root)
    sources = {str(path.relative_to(REPO)): text for path, text in src.items()}
    return uncalled_functions(sources, in_detail + in_scripts + in_research)


def uncalled_functions(
    sources: dict[str, str], references: collections.Counter
) -> dict[str, str]:
    """``{name: "file:line"}`` for each public, undecorated module-level function in
    ``sources`` whose name ``references`` never uses. Pure, so the negative case can hand it a
    literal."""
    found: dict[str, str] = {}
    for where, text in sources.items():
        if pathlib.PurePath(where).name == "cli.py":  # bound by @click, not by name
            continue
        for node in ast.parse(text).body:  # module level only
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_") or node.decorator_list:
                continue
            # A def is a FunctionDef, not a Name, so any hit here is a genuine reference.
            if references[node.name]:
                continue
            found[node.name] = f"{where}:{node.lineno}"
    return found


def test_the_function_scan_would_catch_an_unwired_function():
    """The negative case: a public function nobody calls is caught; a called one, a private
    one, a decorated one and anything in ``cli.py`` are not; a name in a docstring does not
    count as a call."""
    source = (
        "def cross_seed_stability():\n"
        '    """resolve_track_id is the designed way out."""\n'
        "def resolve_track_id(): ...\n"
        "def _helper(): ...\n"
        "@register\n"
        "def registered(): ...\n"
        "def wired(): ...\n"
        "wired()\n"
    )
    refs = collections.Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            refs[node.id] += 1
    found = uncalled_functions({"x.py": source, "cli.py": "def command(): ..."}, refs)
    assert found == {"cross_seed_stability": "x.py:1", "resolve_track_id": "x.py:3"}


def test_no_new_function_is_built_and_left_unwired():
    orphans = _orphans()
    unexplained = {n: w for n, w in orphans.items() if n not in _ORPHANS_WITH_REASONS}
    assert not unexplained, (
        "these public functions have no caller in detail/, scripts/ or another caller root. "
        "Either wire them up, "
        "delete them, or add an entry to _ORPHANS_WITH_REASONS saying why they have none: "
        f"{unexplained}"
    )


def test_the_allow_list_does_not_excuse_functions_that_are_wired_or_gone():
    """A stale entry is the same bug one level up — a claim nothing checks."""
    orphans = _orphans()
    stale = sorted(set(_ORPHANS_WITH_REASONS) - set(orphans))
    assert not stale, (
        "these are listed as uncalled but are now called (or no longer exist); drop the entry: "
        f"{stale}"
    )


# --- methods ---------------------------------------------------------------------------- #
#: The classes whose public methods are scanned too. **Not every class**: over all of
#: ``detail/`` the same scan finds ~90 uncalled public methods (2026-11-09), most of them
#: deliberate — a widget's read-back of its own state for a test (``FrameScrubber.time_labels``),
#: the scripting API a notebook calls (``Recording.define_roi``), a protocol member a registry
#: reaches by name — so a whole-tree allow-list would be ninety claims long and read as noise,
#: which is the state this guard exists to end. These are the classes where a montage window, a
#: page-exports section and a breadcrumb were deleted and their helpers were left behind
#: (``NodeView.positions_by_recording``, ``NodeIcicle.shown_path``,
#: ``CollapsibleBand.detach_header``, ``AuthoredRun.audit_clusters``, nine ``ClusterMontage``
#: methods, ``FilterBar.add_leading``). Add a class when a sweep of it comes back clean.
_METHOD_SCANNED_CLASSES = frozenset(
    {
        "AuthoredRun",
        "ClusterMontage",
        "CollapsibleBand",
        "FilterBar",
        "NodeIcicle",
        "NodeView",
    }
)

#: ``Class.method`` → why it has no caller in ``detail/``. Claims, checked both ways like
#: :data:`_ORPHANS_WITH_REASONS`.
_METHOD_ORPHANS_WITH_REASONS = {
    "NodeIcicle.shades": "a test's read of the shading the tree was handed, which the paint "
    "step reads off the private dict",
    "NodeIcicle.window_marks": "a test's read of which nodes carry a cluster window's number",
    "NodeIcicle.window_mark_rects": "a test's read of where the last paint put each window "
    "mark, so a click can be aimed at one",
    "NodeIcicle.fits_rows": "a test's read of whether the rows still fit the height (until a "
    "drag), the state a resize keeps",
    "FilterBar.constant_rows": "a test's read of the greyed one-value lines",
    "FilterBar.field_rows": "a test's read of the field lines in their arranged order",
    "FilterBar.field_row": "a test's handle on one field's line by name",
}


def _references(roots) -> collections.Counter:
    """Every name ``roots`` uses as a ``Name``, an ``Attribute``, or an identifier-shaped string
    (``getattr(page, "export_scope")`` reaches a method by a string)."""
    names: collections.Counter = collections.Counter()
    for root in roots:
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Name):
                    names[node.id] += 1
                elif isinstance(node, ast.Attribute):
                    names[node.attr] += 1
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value.isidentifier()
                ):
                    names[node.value] += 1
    return names


def uncalled_methods(
    sources: dict[str, str], references: collections.Counter, classes
) -> dict[str, str]:
    """``{"Class.method": "file:line"}`` for each public, undecorated method of ``classes`` in
    ``sources`` whose name ``references`` never uses. Pure, so the negative case can hand it a
    literal. Qt's own virtuals are camelCase (``paintEvent``) and skipped by that shape; a
    decorated method (a property, a slot) is reached by its decorator's machinery."""
    found: dict[str, str] = {}
    for where, text in sources.items():
        for cls in ast.walk(ast.parse(text)):
            if not isinstance(cls, ast.ClassDef) or cls.name not in classes:
                continue
            for node in cls.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                name = node.name
                if name.startswith("_") or node.decorator_list:
                    continue
                if re.search(r"[A-Z]", name) or references[name]:
                    continue
                found[f"{cls.name}.{name}"] = f"{where}:{node.lineno}"
    return found


def _method_orphans() -> dict[str, str]:
    sources = {
        str(p.relative_to(REPO)): p.read_text(encoding="utf-8-sig")
        for p in (REPO / "detail").rglob("*.py")
    }
    roots = (
        REPO / "detail",
        REPO / "scripts",
        *(REPO / root for root in _EXTRA_CALLER_ROOTS),
    )
    return uncalled_methods(sources, _references(roots), _METHOD_SCANNED_CLASSES)


def test_no_scanned_class_keeps_a_method_nothing_calls():
    unexplained = {
        n: w
        for n, w in _method_orphans().items()
        if n not in _METHOD_ORPHANS_WITH_REASONS
    }
    assert not unexplained, (
        "these public methods have no caller in detail/, scripts/ or another caller root. "
        "Delete them, make them private, or add an entry to _METHOD_ORPHANS_WITH_REASONS "
        f"saying why they have none: {unexplained}"
    )


def test_the_method_allow_list_names_only_uncalled_methods_of_scanned_classes():
    orphans = _method_orphans()
    stale = sorted(set(_METHOD_ORPHANS_WITH_REASONS) - set(orphans))
    assert not stale, f"listed as uncalled but now called or gone: {stale}"
    unscanned = sorted(
        n
        for n in _METHOD_ORPHANS_WITH_REASONS
        if n.split(".")[0] not in _METHOD_SCANNED_CLASSES
    )
    assert not unscanned, f"entries for classes the scan does not read: {unscanned}"


def test_the_method_scan_would_catch_a_left_behind_helper():
    """The negative case, in the shapes the sweep found: a plain helper nobody calls is caught;
    a called one, a private one, a property, a Qt virtual and a method of an unscanned class
    are not; and a name used only in prose (a comment, a docstring sentence) does not count.
    """
    source = (
        "class NodeView:\n"
        "    def positions_by_recording(self, node):\n"
        '        """Nobody calls positions_by_recording."""\n'
        "    def positions(self, node): ...\n"
        "    def _floor(self): ...\n"
        "    @property\n"
        "    def root(self): ...\n"
        "    def paintEvent(self, event): ...\n"
        "class Elsewhere:\n"
        "    def unused(self): ...\n"
        "view.positions(3)  # positions_by_recording is mentioned here only\n"
    )
    refs = _references_of(source)
    assert uncalled_methods({"x.py": source}, refs, {"NodeView"}) == {
        "NodeView.positions_by_recording": "x.py:2"
    }


def _references_of(source: str) -> collections.Counter:
    """:func:`_references` over one literal, for the negative case."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        root = pathlib.Path(folder)
        (root / "x.py").write_text(source, encoding="utf-8")
        return _references([root])


#: A failsafe, not a budget. The point of the check is to catch the scan becoming
#: *pathological* — an accidental quadratic, a re-read per symbol — not to police a second:
#: a duration on a busy machine measures whoever else is busy rather than the code.
_PATHOLOGICAL_S = 60.0


def test_the_scan_does_not_become_pathological():
    """It reads every source file on every run, so it must not grow a term that compounds.

    Timed rather than counted because what is being guarded against is *algorithmic*, and a file
    count cannot see an accidental quadratic over symbols. So the bound is set where only a real
    change in complexity reaches it — twenty-odd times the measured cost — and the run's own
    watchdog (`faulthandler_timeout`, 120s) sits behind it.
    """
    import time

    start = time.monotonic()
    _orphans()
    _method_orphans()
    elapsed = time.monotonic() - start
    assert elapsed < _PATHOLOGICAL_S, (
        f"the orphan scan took {elapsed:.1f}s against a {_PATHOLOGICAL_S:.0f}s failsafe. That is "
        "not a slow machine — something in it has changed complexity."
    )
