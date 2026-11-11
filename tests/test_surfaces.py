"""The deploy toggle keeps a capability out of *every* door (§25).

Three failures this guards, all of which ship green:

**A leaked door.** Hiding a capability means hiding it wherever a user could *enumerate* it — the
rail, the overview dashboard, a launch dialog's module list, the CLI's help, and the user guide.
Miss one and the capability is still on offer while the table claims it is hidden; the door nobody
re-checks is the one reached by a table double-click rather than by the rail. So the parity test is
driven by a list of enumeration points, and a surface added without adding its point fails here.

**A vacuous entry.** ``HIDDEN["syllable_discovry"]`` hides nothing, raises nothing, and reads
correctly in review. The only way to catch a typo'd name is to assert it resolves to something real.

**Documentation that outlives the surface.** ``docs/cli.md`` and ``docs/gui.md`` describe surfaces
in hand-written prose no registry knows about, so hiding a capability everywhere else while the
guide still teaches it is worse than either state alone.

The toggle governs **launching, not reading** — a completed run of a hidden module still renders,
so hiding one never makes a user's existing work vanish — and a hidden CLI command stays
executable, which is what lets a research sweep keep running against a build that hides it.
"""

import re
from pathlib import Path

import pytest

from detail import surfaces
from detail.surfaces import strip_hidden
from detail.cli import main as cli_main
from detail.gui import sections as gui_sections
from detail.gui.modules import registry as gui_registry


def _rail_sections():
    return [s.kind for s in gui_sections.sections()]


def _overview_sections():
    # The overview dashboard reads the same declared list — the second door onto a section.
    return [s.kind for s in gui_sections.sections()]


def _processing_launch():
    return gui_registry.surfaced(gui_registry.GUI_MODULES)


def _analysis_launch():
    return gui_registry.surfaced(gui_registry.ANALYSIS_GUI_MODULES)


def _gui_query_methods():
    """The method picker on the Behavior scan form — the GUI door onto a §20 method."""
    from detail.gui.modules.analysis_query import _query_methods

    return sorted(_query_methods())


def _cli_query_methods():
    """`analysis query --method`'s choices — the CLI door onto the same registry."""
    from detail.cli import _query_methods

    return _query_methods()


def _gui_trackers():
    """The tracker picker on the Tracking launch form — the GUI door onto a §13 tracker."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.core.registries import TRACKERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)
    return gui_registry.surfaced(TRACKERS)


def _cli_trackers():
    """What `tracking run` lists as available — the CLI door onto the same registry."""
    from detail.cli import _surfaced_trackers

    return _surfaced_trackers()


def _gui_embedding_models():
    """The model picker on the Embedding launch form — the GUI door onto `EMBEDDING_MODELS`."""
    from detail.core.registries import EMBEDDING_MODELS

    _register_embedding_models()
    return gui_registry.surfaced(EMBEDDING_MODELS)


def _gui_model_manager():
    """The Model manager's install rows — a hidden model offered for install is still offered."""
    from detail.gui.pages import surfaced_model_rows

    return [r.id for r in surfaced_model_rows()]


def _cli_model_list():
    """`model list` — the CLI's catalog of everything a user can install and run."""
    from detail.cli import _surfaced_model_variants

    return [r.id for r in _surfaced_model_variants()]


def _gui_constructors():
    """The Input image picker on the Embedding launch form."""
    from detail.transforms.constructors import surfaced_constructors

    return surfaced_constructors()


def _cli_constructors():
    """`--constructor`'s choices on both CLI commands that take one."""
    import click

    out = []
    for path in (("embedding", "run"), ("process", "embedding")):
        command = cli_main
        for part in path:
            command = command.commands[part]
        (option,) = [p for p in command.params if p.name == "constructor"]
        assert isinstance(option.type, click.Choice)
        out.extend(option.type.choices)
    return out


def _register_embedding_models():
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules

    import_runner_modules(PIPELINE_RUNNERS)


def _generated_module_pages():
    """The ids that get a published `docs/modules/<id>.md`, methods included.

    The user guide is an enumeration point (§25) and this is the one that spans *both* kinds
    of manifestation: a hidden module and a hidden method are equally reachable by URL if the
    generator still writes their page.
    """
    from detail.configs.describe import surfaced_config_models

    return [module_id for module_id, _model in surfaced_config_models()]


def _cli_help(group=cli_main, prefix=""):
    """Full command paths a user meets in ``--help``, walked to the leaves.

    Nested on purpose: a capability can hide a subcommand of a visible group
    (``clusterer sweep`` was one, until it was removed), and a scan that only read the top
    level would call that hidden.
    """
    out = []
    for name, command in getattr(group, "commands", {}).items():
        if command.hidden:
            continue
        path = f"{prefix}{name}"
        out.append(path)
        out.extend(_cli_help(command, f"{path} "))
    return out


ENUMERATION_POINTS = {
    "gui rail sections": _rail_sections,
    "gui overview sections": _overview_sections,
    "gui processing launch": _processing_launch,
    "gui analysis launch": _analysis_launch,
    "cli help": _cli_help,
    "gui query methods": _gui_query_methods,
    "cli query methods": _cli_query_methods,
    "gui trackers": _gui_trackers,
    "cli trackers": _cli_trackers,
    "generated module pages": _generated_module_pages,
    "gui embedding models": _gui_embedding_models,
    "gui model manager": _gui_model_manager,
    "cli model list": _cli_model_list,
    "gui constructors": _gui_constructors,
    "cli constructors": _cli_constructors,
}


def _method_registries() -> set[str]:
    """Every pluggable method id a ``Hidden.methods`` entry could name.

    Imported through the registries' own loaders rather than listed, so a method added
    tomorrow is resolvable the day it is registered.
    """
    from detail.cluster_methods import (
        import_pose_cluster_methods,
        import_temporal_cluster_methods,
    )
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.core.registries import (
        EMBEDDING_MODELS,
        POSE_CLUSTER_METHODS,
        QUERY_METHODS,
        TEMPORAL_CLUSTER_METHODS,
        TRACKERS,
    )
    from detail.executor.runners import import_runner_modules
    from detail.query_methods import import_query_methods
    from detail.transforms.constructors import CONSTRUCTORS

    import_query_methods()
    import_pose_cluster_methods()
    import_temporal_cluster_methods()
    # Trackers and embedding models have no `import_*` helper of their own; importing the
    # stages registers them.
    import_runner_modules(PIPELINE_RUNNERS)
    return (
        set(QUERY_METHODS)
        | set(POSE_CLUSTER_METHODS)
        | set(TEMPORAL_CLUSTER_METHODS)
        | set(TRACKERS)
        | set(EMBEDDING_MODELS)
        | set(CONSTRUCTORS)
    )


def _raw_runners() -> set[str]:
    """Runner ids registered with ``register_runner`` rather than as a module.

    ``JOB_RUNNERS`` is populated by decorator at module import. The two modules this used to
    import explicitly, because the registry walks above did not pull them in, were the removed
    clusterer fit and apply.
    """
    from detail.executor.runners import JOB_RUNNERS

    return set(JOB_RUNNERS)


def _resolvable() -> set[str]:
    from detail.executor import runners

    return (
        set(gui_sections.SECTION_KINDS)
        | set(gui_registry.GUI_MODULES)
        | set(gui_registry.ANALYSIS_GUI_MODULES)
        | set(getattr(runners, "PROCESSING_MODULES", {}))
        | set(getattr(runners, "ANALYSIS_MODULES", {}))
        # The raw `register_runner` runners — real launchable things with config models and
        # generated parameter pages, in neither module registry. Left out, an entry naming one
        # read as a typo, which is why the clusterer chain could only ever declare the one of
        # its three ids that happens to be an AnalysisModule.
        | _raw_runners()
        | _method_registries()
        | set(_cli_help())
        | {p for e in surfaces.HIDDEN.values() for p in e.cli}
    )


@pytest.fixture(autouse=True)
def _no_escape_hatch(monkeypatch):
    monkeypatch.delenv(surfaces.SHOW_ALL_ENV, raising=False)


def _in_effect():
    return {c: e for c, e in surfaces.HIDDEN.items() if e.in_effect}


def _staged():
    return {c: e for c, e in surfaces.HIDDEN.items() if not e.in_effect}


def test_every_hidden_manifestation_resolves_to_something_real():
    """A typo hides nothing and errors nowhere — resolution is the only way to catch it.

    Covers **staged** entries too, which is most of the value of staging one: a declaration
    nobody validates until the day it is switched on is a declaration that has quietly rotted.
    """
    declared = {
        name for entry in surfaces.HIDDEN.values() for name in entry.manifestations()
    }
    unresolved = sorted(declared - _resolvable())
    assert not unresolved, (
        f"hidden manifestations matching no section, module or command: {unresolved}. "
        f"A typo'd name silently hides nothing."
    )


@pytest.mark.parametrize("point", sorted(ENUMERATION_POINTS))
def test_a_hidden_capability_is_absent_from_every_enumeration_point(point):
    listed = set(ENUMERATION_POINTS[point]())
    still_offered = sorted(surfaces.hidden_names() & listed)
    assert not still_offered, f"{still_offered} still offered at {point!r}"


def test_the_parity_scan_would_catch_a_leaked_surface(monkeypatch):
    """Guard the guard: with everything filtered the scan passes vacuously, so prove it bites."""
    leaked = _rail_sections()[0]
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", section=leaked)
    )
    monkeypatch.setitem(ENUMERATION_POINTS, "leaky point", lambda: [leaked])
    with pytest.raises(AssertionError):
        test_a_hidden_capability_is_absent_from_every_enumeration_point("leaky point")


def test_a_staged_entry_points_at_docs_that_still_exist():
    """The inverse assertion, and the reason staging is worth anything.

    A staged capability is still surfaced, so its docs *should* be there — which means the
    anchor can be checked today. Left unchecked, a doc rewrite would silently invalidate it and
    the entry would enforce nothing on the day it was switched on, which looks exactly like a
    working toggle.
    """
    for capability, entry in _staged().items():
        for path, anchor in entry.docs:
            assert anchor in Path(path).read_text(encoding="utf-8-sig"), (
                f"{capability!r} declares {anchor!r} in {path}, which no longer contains it — "
                f"the anchor drifted, so switching this entry on would enforce nothing"
            )


def test_a_staged_entry_hides_nothing_yet():
    for capability, entry in _staged().items():
        assert entry.awaiting, capability
        for name in entry.manifestations():
            assert surfaces.is_surfaced(name), (
                f"{capability!r} is staged (awaiting {entry.awaiting!r}) but {name!r} is "
                f"already hidden"
            )


def test_user_docs_do_not_teach_a_capability_in_effect():
    """Anchors are declared on the entry, not grepped for: "comparison" also appears in prose
    about DTW comparisons, so a keyword scan would fire on unrelated text."""
    for capability, entry in _in_effect().items():
        for path, anchor in entry.docs:
            # Post-strip: prose may satisfy this by being deleted *or* by being marked for
            # this capability (scripts/docs_surface_hook.py), which is what lets it stay in
            # the tree and come back with the toggle. Same function the build uses, so the
            # guard and the site cannot disagree about what a marker means.
            text = strip_hidden(Path(path).read_text(encoding="utf-8-sig"), where=path)
            assert anchor not in text, (
                f"{path} still documents {capability!r} at {anchor!r}; a build that hides a "
                f"capability while its guide teaches it is worse than either state alone"
            )


def test_a_hidden_entry_declares_the_docs_it_owns():
    """An entry with no doc anchors has either checked the guide or forgotten it, and the two
    are indistinguishable — so declaring none is only allowed for a capability the user docs
    never mentioned, which is asserted here by name."""
    undocumented_by_design: set[str] = set()
    for capability, entry in surfaces.HIDDEN.items():
        assert entry.docs or capability in undocumented_by_design, (
            f"{capability!r} declares no doc anchors — if the user guide really never "
            f"mentions it, add it to undocumented_by_design with a reason"
        )


def test_hiding_a_section_removes_it_from_the_declared_list(monkeypatch):
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", section="comparisons")
    )
    assert "comparisons" not in _rail_sections()
    assert "comparisons" in gui_sections.SECTION_KINDS  # declared, just not surfaced


@pytest.mark.parametrize(
    "registry_name, enumerate_point",
    [("GUI_MODULES", _processing_launch), ("ANALYSIS_GUI_MODULES", _analysis_launch)],
)
def test_hiding_a_module_removes_it_from_its_launch_list(
    registry_name, enumerate_point, monkeypatch
):
    """Exercise the registry filter against a real registered module.

    Every hidden capability today is a section plus CLI paths, so nothing asserts the module
    path — reverting ``registry.surfaced`` would break no test, and the filter would be guarded
    vacuously. Hiding a live module here makes that path bite.
    """
    registry = getattr(gui_registry, registry_name)
    module_id = sorted(registry)[0]
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", modules=(module_id,))
    )
    assert module_id not in enumerate_point()
    assert module_id in registry  # registered, just not offered


def test_the_escape_hatch_reveals_everything(monkeypatch):
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", section="comparisons")
    )
    monkeypatch.setenv(surfaces.SHOW_ALL_ENV, "1")
    assert surfaces.is_surfaced("comparisons")
    assert "comparisons" in _rail_sections()


def test_the_reason_survives_the_escape_hatch(monkeypatch):
    """A revealed surface still says why it is normally hidden."""
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("because", section="comparisons")
    )
    monkeypatch.setenv(surfaces.SHOW_ALL_ENV, "1")
    assert surfaces.hidden_reason("comparisons") == "because"


def test_show_all_is_read_at_call_time_not_snapshotted(monkeypatch):
    """A module constant would freeze whatever the environment held at import — the cache-bug
    shape recorded in CLAUDE.md — and would make the setting untestable."""
    assert not surfaces.show_all()
    monkeypatch.setenv(surfaces.SHOW_ALL_ENV, "1")
    assert surfaces.show_all()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "FALSE"])
def test_falsey_env_values_do_not_reveal(monkeypatch, value):
    monkeypatch.setenv(surfaces.SHOW_ALL_ENV, value)
    assert not surfaces.show_all()


def test_a_hidden_cli_command_stays_executable():
    """``hidden`` keeps a command out of help and the generated reference; it does not disable
    it, which is what lets a research sweep run against a build that hides it."""
    for entry in _in_effect().values():
        for path in entry.cli:
            command = cli_main
            for part in path.split():
                command = command.commands[part]
            assert command.hidden, path
            assert command.callback is not None or command.commands, path


def test_apply_cli_visibility_is_re_appliable(monkeypatch):
    """Applied from the table after import rather than as ``hidden=`` per decorator, so a test
    can change the table and re-apply instead of needing a fresh import.

    Probes a surfaced group (`comparison`) hidden for the test, so it does not depend on what
    the shipped table happens to hide.
    """
    command = cli_main.commands["comparison"]
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", cli=("comparison",))
    )
    try:
        surfaces.apply_cli_visibility(cli_main)
        assert command.hidden
        monkeypatch.setenv(surfaces.SHOW_ALL_ENV, "1")
        surfaces.apply_cli_visibility(cli_main)
        assert not command.hidden
    finally:
        command.hidden = (
            False  # the probe is gone with the test; leave the group as shipped
        )


def test_hidden_entries_carry_a_reason():
    for capability, entry in surfaces.HIDDEN.items():
        assert entry.reason.strip(), capability


def test_the_design_plan_names_exactly_what_is_hidden():
    """The prose copy of `HIDDEN` cannot drift from the table.

    §25's status row in `DESIGN_PLAN.md` lists what this build hides. It is the only place a
    reader who is not reading `surfaces.py` finds that out — and it had **already gone stale**:
    it named three entries in effect and one staged, while `behavior_discovery` had been
    switched on. Nobody noticed, because a sentence that is wrong reads exactly like a sentence
    that is right.

    That is the whole failure mode of a permissioning toggle. It carries no removal debt and no
    expiry (`surfaces.py`), so nothing ever forces a re-read; what is hidden is remembered only
    for as long as someone remembers it. This makes the record self-maintaining instead: hide
    something without saying so here, or un-hide it without amending this, and the guard fails.

    Matched on the human-readable name each entry goes by in that sentence, not on the dict key,
    because the sentence is prose for a person and should stay so.
    """
    plan = (Path(__file__).resolve().parents[1] / "DESIGN_PLAN.md").read_text(
        encoding="utf-8-sig"
    )
    row = next(
        line for line in plan.splitlines() if "| Surface Visibility (§25)" in line
    )
    in_effect = {key for key, entry in surfaces.HIDDEN.items() if entry.in_effect}
    staged = {key for key, entry in surfaces.HIDDEN.items() if not entry.in_effect}

    spoken = {key: key.replace("_", " ") for key in surfaces.HIDDEN}
    missing = [spoken[key] for key in in_effect if spoken[key] not in row]
    assert not missing, (
        f"§25's status row does not name these hidden capabilities: {missing}. It is the only "
        "place a reader outside surfaces.py learns what this build withholds."
    )
    wrongly_named = [
        spoken[key] for key in staged if spoken[key] in row and "Staged" not in row
    ]
    assert (
        not wrongly_named
    ), f"{wrongly_named} are staged, but the row reads as in effect"
    if not staged:
        assert "Nothing is staged" in row, (
            "nothing is staged, so the row must say so — a row that silently drops the staged "
            "clause reads as though it were never updated"
        )
    if not in_effect:
        assert (
            "Nothing is hidden" in row
        ), "nothing is in effect, so the row must say so rather than list what used to be"


#: The count word the rulebook's hidden-capability sentence opens with.
_COUNT_WORDS = {
    0: "No",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
    11: "Eleven",
    12: "Twelve",
}


#: Every agent-facing file that states, in prose, how many capabilities this build hides. Both are
#: loaded or read at the start of work, and both had gone stale the same way — `CLAUDE.md` said six
#: with `tracking_iou` missing, and `docs/how_we_work.md` said six under a *"Read this first"*
#: heading. The first pass of this guard read only `CLAUDE.md`, so fixing one site and guarding one
#: site left the other wrong: the sweep is the point of the rule, and it was not done.
#: The second copy is private: its "Read this first" section is stripped from the public copy
#: (`scripts/export_public.py`), where a test asserting the sentence is there would fail.
_PROSE_COPIES = ("CLAUDE.md",)


def _count_phrase(n: int) -> str:
    """How the sentence states ``n``: "One capability is hidden", "Two capabilities are hidden"."""
    return (
        f"{_COUNT_WORDS[n]} capability is hidden"
        if n == 1
        else (f"{_COUNT_WORDS[n]} capabilities are hidden")
    )


_COUNT_SENTENCE = re.compile(r"capabilit(?:y is|ies are) hidden", re.IGNORECASE)


def _hidden_prose(name: str) -> str:
    """The passage in `name` that states how many capabilities are hidden, through its section."""
    lines = (
        (Path(__file__).resolve().parents[1] / name)
        .read_text(encoding="utf-8-sig")
        .split("\n")
    )
    start = next(
        (i for i, line in enumerate(lines) if _COUNT_SENTENCE.search(line)),
        None,
    )
    if start is None:
        return ""
    end = next(
        (
            j
            for j in range(start + 1, len(lines))
            if lines[j].startswith(("## ", "# ", "---"))
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _manifestations_missing_from(para: str) -> list[str]:
    """Ids of in-effect entries that `para` does not name.

    Matched case-insensitively as a substring rather than as a backticked token, because the two
    prose copies spell the same manifestation differently — `CLAUDE.md` writes "the `clusterers`
    section", `how_we_work.md` writes "the Clusterers section". One matcher over both beats a
    stricter one that only works on the file it was written against.

    Section, modules and method — not `cli`. A reader identifies a capability by its module or
    method id; requiring every command group would make the sentence unreadable and is already
    covered by the enumeration-point sweep above.

    A pure function so the negative case below can feed it the stale sentence CLAUDE.md really
    shipped, the way `tests/test_docs_nav.py::_undeclared` is factored.
    """
    required = [
        name
        for entry in surfaces.HIDDEN.values()
        if entry.in_effect
        for name in (entry.section, *entry.methods, *entry.modules)
        if name
    ]
    lowered = para.lower()
    return sorted({name for name in required if name.lower() not in lowered})


@pytest.mark.parametrize("doc", _PROSE_COPIES)
def test_the_prose_copies_name_exactly_what_is_hidden(doc):
    """Every prose copy of `HIDDEN` cannot drift — the sibling of the DESIGN_PLAN test above.

    The same class, now over each file that states the count. `CLAUDE.md` is loaded into every
    session and `docs/how_we_work.md` opens with *"Read this first"*, so a wrong sentence in either
    is read before an agent touches anything — and **both had gone stale the same way**. Each
    counted six while seven entries were declared, omitting `tracking_iou`; `CLAUDE.md` also named
    only `cluster_eval` where the declaration names the whole `cluster_fit` / `cluster_apply` /
    `cluster_eval` chain, which is the recorded class about a declaration that could not express the
    truth, re-made in prose beside its own write-up.

    **This guard is parametrized because its first version was not.** It read `CLAUDE.md` alone, so
    the change that fixed and guarded that file left `how_we_work.md` wrong and unguarded — a fix at
    the site rather than at the class, in the guard written for exactly that class. A sweep for the
    sentence is what found the second copy.
    """
    para = _hidden_prose(doc)
    assert para, (
        f"{doc} no longer says how many capabilities this build hides. It is what an agent reads "
        "before treating a surface as real; deleting the sentence is not the fix."
    )

    in_effect = [entry for entry in surfaces.HIDDEN.values() if entry.in_effect]
    phrase = _count_phrase(len(in_effect))
    assert phrase.lower() in para.lower(), (
        f"{len(in_effect)} capabilities are hidden in effect, so {doc} must say "
        f'"{phrase}". Both copies read "six" for as long as `tracking_iou` '
        "had been declared, and nothing noticed."
    )

    unnamed = _manifestations_missing_from(para)
    assert not unnamed, (
        f"{doc}'s hidden-capability passage does not name {unnamed}. A manifestation missing from "
        "it is a surface an agent will go on treating as real."
    )


def test_the_rulebook_scan_would_catch_a_dropped_capability(monkeypatch):
    """Guard the guard, on the exact sentence that shipped.

    Without this, the check above passes in two different worlds — the paragraph is complete, or
    the backtick match stopped matching — and they look identical from outside.
    """
    shipped = (
        "**Six capabilities are hidden from a default build** (`detail/surfaces.py`, §25): "
        "`benchmarks`, the `clusterers` section + `cluster_eval`, `clip_query`, "
        "`clip_annotation`, `syllable_discovery`, and the `dtw` query method."
    )
    # The entry that sentence misdescribed, declared for the test: the clusterer chain has since
    # been removed (`docs/removed_capabilities.md`), and the scan reads the live table.
    monkeypatch.setitem(
        surfaces.HIDDEN,
        "probe",
        surfaces.Hidden(
            "test",
            section="clusterers",
            modules=("cluster_fit", "cluster_apply", "cluster_eval"),
        ),
    )
    unnamed = _manifestations_missing_from(shipped)
    # That sentence also left out the multi-subject `iou_tracker`, which this asserted on until
    # the tracker was deleted from the tree and so stopped being something to name.
    assert {"cluster_apply", "cluster_fit"} <= set(unnamed), (
        "it must also notice the two clusterer-chain modules that sentence collapses into "
        "`cluster_eval` — the half of the drift that is easiest to read past"
    )


# --- prose markers: the docs' half of the toggle (§25) ---------------------- #
def _doc_pages() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return sorted((root / "docs").rglob("*.md"))


def test_every_marker_names_a_real_capability():
    """A typo'd ``<!--surface:behaviour_clusterers-->`` matches no entry, so nothing is ever
    stripped and the prose publishes — the same class as a typo'd manifestation above, one
    surface over, and just as silent."""
    known = set(surfaces.HIDDEN)
    for path in _doc_pages():
        for name in surfaces.marked_capabilities(path.read_text(encoding="utf-8-sig")):
            assert name in known, (
                f"{path.name} marks {name!r}, which is not a capability in HIDDEN — "
                f"the block will never be stripped"
            )


def test_markers_are_balanced_in_every_page():
    """An unclosed opener would swallow the rest of the page, which reads as "the docs lost a
    section" long after the edit that did it."""
    for path in _doc_pages():
        surfaces.check_balanced(path.read_text(encoding="utf-8-sig"), where=str(path))


def test_an_unclosed_marker_is_an_error_not_a_silent_truncation():
    """Guard the guard."""
    with pytest.raises(surfaces.UnbalancedMarkerError):
        surfaces.check_balanced("<!--surface:benchmarks-->\nbody\n", where="probe")


def test_a_marked_block_comes_back_when_the_capability_is_shown():
    """The point of marking rather than deleting: the prose is a toggle, not an edit.

    Deletion satisfies the absence guard just as well, so nothing else in this file would
    notice if the strip simply dropped every marked block unconditionally.
    """
    text = "before\n<!--surface:benchmarks-->\nBODY\n<!--/surface-->\nafter"
    assert "BODY" not in surfaces.strip_hidden(text, lambda _name: False)

    shown = surfaces.strip_hidden(text, lambda _name: True)
    assert "BODY" in shown
    assert "<!--surface" not in shown  # markers never reach the reader either way


# --------------------------------------------------------------------------- #
# hiding a *method*: one choice inside a surfaced capability (`Hidden.methods`)
# --------------------------------------------------------------------------- #


def test_hiding_a_method_removes_it_from_every_picker(monkeypatch):
    """Exercise the method filter against a real registered method.

    The module twin of `test_hiding_a_module_removes_it_from_its_launch_list`, and needed for
    the same reason: a manifestation kind nothing hides is a code path that passes vacuously.
    Probes a method that is *not* already hidden, so the assertion is about the filter rather
    than about the entry that ships.
    """
    from detail.core.registries import QUERY_METHODS

    candidates = [m for m in sorted(QUERY_METHODS) if surfaces.is_surfaced(m)]
    assert candidates, "no surfaced query method left to probe with"
    probe = candidates[0]
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", methods=(probe,))
    )

    assert probe not in _gui_query_methods()
    assert probe not in _cli_query_methods()
    assert probe not in _generated_module_pages()
    assert probe in QUERY_METHODS  # registered, just not offered


#: Pickers no in-effect entry reaches any more, by the kind of id they offer. The scan below
#: (`test_the_method_scan_would_catch_a_leak`) proves a picker consults the table by finding a
#: really-hidden id in it once the table is stubbed open, which needs a hidden id of that kind
#: to exist. When the last one is deleted from the tree the picker's filter is still there and
#: still has to work the day something is hidden again, so it is probed instead of dropped.
_PICKERS_WITH_NOTHING_HIDDEN = {
    "embedding model": (_gui_embedding_models, _gui_model_manager, _cli_model_list),
    "tracker": (_gui_trackers, _cli_trackers),
    # `--constructor`'s choices are fixed when the command is declared, so a probe hidden
    # after import cannot reach them; `ENUMERATION_POINTS` still sweeps that door.
    "input image": (_gui_constructors,),
    "query method": (_gui_query_methods, _cli_query_methods),
}


@pytest.mark.parametrize("kind", sorted(_PICKERS_WITH_NOTHING_HIDDEN))
def test_a_picker_with_nothing_hidden_still_consults_the_table(kind, monkeypatch):
    """Hide an id the picker offers today, and it must leave every door onto that registry."""
    pickers = _PICKERS_WITH_NOTHING_HIDDEN[kind]
    offered = sorted(set.intersection(*(set(picker()) for picker in pickers)))
    assert (
        offered
    ), f"no {kind} is offered by every picker, so there is nothing to probe with"
    probe = offered[0]

    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", methods=(probe,))
    )
    for picker in pickers:
        assert probe not in picker(), f"{picker.__name__} ignores surfaces"


def test_a_hidden_method_still_resolves_so_a_saved_run_opens(monkeypatch):
    """§25 governs launching, not reading — and for a method that is the whole point.

    A completed run's record names the method it used. If `resolve_method` consulted this
    table, hiding one would make every run that used it unopenable, unrenderable and
    unexportable: the toggle would destroy work rather than withhold a choice.
    """
    hidden_methods = [m for entry in _in_effect().values() for m in entry.methods]
    if not hidden_methods:
        # Nothing ships hidden, so the property is checked on a probe: hide a registered method
        # and it must still resolve. Without this the guard would pass vacuously.
        probe = sorted(_method_registries())[0]
        monkeypatch.setitem(
            surfaces.HIDDEN, "probe", surfaces.Hidden("test", methods=(probe,))
        )
        hidden_methods = [probe]
    # Resolved through whichever registry owns the id, so this covers every kind of hidden
    # implementation rather than the one kind that existed when it was written. `method` now
    # spans the query, cluster-method and tracker registries; a guard that hardcoded
    # `resolve_method` would have started failing the day a tracker was hidden and read as a
    # broken tracker rather than as a test with a narrow assumption.
    resolvable = _method_registries()
    for method_id in hidden_methods:
        assert not surfaces.is_surfaced(method_id)
        assert method_id in resolvable, (
            f"{method_id!r} is hidden but no registry still resolves it — a saved run that "
            "used it could not open"
        )


def test_a_marked_capabilitys_anchors_are_findable_in_the_source():
    """A stale anchor on an in-effect entry enforces nothing, silently.

    `test_user_docs_do_not_teach_a_capability_in_effect` asserts each anchor is *absent after
    stripping* — which a string that was never in the file at all satisfies forever. For prose
    that was **deleted** that is unavoidable and correct; for prose that was **marked** it is a
    typo waiting to be believed, and marking is what this repo prefers ("editable, tested, and
    switchable back on is not true of prose someone has to rewrite from memory").

    So: where a file carries a `surface:<capability>` marker, **at least one** of that
    capability's anchors on that file must exist in the raw source. Caught a real one —
    `([dtw](modules/dtw.md))`, declared with parentheses the prose does not have, alongside a
    second anchor that had been wrapped across a line break.

    At-least-one rather than all, and the weakness is real: a file may mark one passage and
    delete another, so `behavior_clusterers` legitimately had 1-of-4 findable in `gui.md`
    before it was removed. That makes this a check on a *set* of anchors, not on each one. It
    catches the case that actually happens — a hand-written anchor for prose the author just
    marked, which is the only moment the string is typed from memory.
    """
    stale = []
    for capability, entry in surfaces.HIDDEN.items():
        by_path: dict[str, list[str]] = {}
        for path, anchor in entry.docs:
            by_path.setdefault(path, []).append(anchor)
        for path, anchors in by_path.items():
            raw = Path(path).read_text(encoding="utf-8-sig")
            if capability not in surfaces.marked_capabilities(raw):
                continue  # this file deletes rather than marks; nothing to find
            if not any(anchor in raw for anchor in anchors):
                stale.append(f"{capability}: none of {anchors} appear in {path}")
    assert not stale, (
        "a capability whose prose is *marked* in a file, whose declared anchors for that file "
        f"all fail to appear in it — so none of them assert anything: {stale}"
    )


def test_the_stale_anchor_scan_would_catch_one(monkeypatch, tmp_path):
    """Guard the guard: a scan matching nothing today passes vacuously forever.

    Nothing ships marked, so the marked page is a temporary one: a capability whose prose is
    marked in it, declaring an anchor that is not there.
    """
    page = tmp_path / "page.md"
    page.write_text(
        "<!--surface:probe-->\nsomething hidden\n<!--/surface-->\n", encoding="utf-8"
    )
    monkeypatch.setitem(
        surfaces.HIDDEN,
        "probe",
        surfaces.Hidden("test", docs=((str(page), "not in this file at all"),)),
    )
    with pytest.raises(AssertionError):
        test_a_marked_capabilitys_anchors_are_findable_in_the_source()


def _undeclared_behind_shut_doors() -> list[str]:
    """Modules whose only launch surface is hidden while §25 does not name the module itself."""
    from detail.configs.describe import LAUNCHED_BY

    behind_a_shut_door = sorted(
        module_id
        for module_id, surface in LAUNCHED_BY.items()
        if not surfaces.is_surfaced(surface)
    )
    return [m for m in behind_a_shut_door if surfaces.is_surfaced(m)]


def test_a_module_behind_a_hidden_door_is_itself_declared_hidden():
    """The declaration half, which the derivation does **not** cover.

    ``surfaced_config_models`` consults ``LAUNCHED_BY``, so the docs leak is closed whatever
    §25 says. But ``is_surfaced`` and ``hidden_names`` cannot: ``surfaces.py`` is deliberately
    dependency-free — ``os`` and ``dataclasses`` only, so both the GUI and the CLI can import
    it — so it has no way to reach that table. With the clusterer entry naming only
    ``cluster_eval``, ``is_surfaced("cluster_fit")`` answered **True** and ``hidden_names()``
    omitted it, which is what every *parity* sweep reads: a surface listing that module would
    have been counted as legitimately offered.

    So the two have to agree, and this is where they are compared. A module whose only way in
    is a hidden surface is a hidden module, and the entry must say so. Since the clusterers
    were removed no shipped entry hides a launch surface, so the live table passes trivially
    and the test below keeps the check able to fail.
    """
    from detail.configs.describe import LAUNCHED_BY

    assert LAUNCHED_BY, "no module declares a launch surface; this guard tests nothing"
    undeclared = _undeclared_behind_shut_doors()
    assert not undeclared, (
        f"{undeclared} can only be launched through a surface this build hides, but §25 does "
        f"not name them — so is_surfaced() says True and hidden_names() omits them, and every "
        f"parity sweep counts them as offered. Add them to the entry's `modules`."
    )


def test_the_shut_door_check_would_catch_an_undeclared_module(monkeypatch):
    """Guard the guard: hide a launch surface without naming its module, and it must notice."""
    from detail.configs.describe import LAUNCHED_BY

    module_id, surface = next(
        (m, door) for m, door in sorted(LAUNCHED_BY.items()) if m != door
    )
    monkeypatch.setitem(
        surfaces.HIDDEN, "probe", surfaces.Hidden("test", cli=(surface,))
    )
    assert module_id in _undeclared_behind_shut_doors()
