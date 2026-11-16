"""Which surfaces this build shows (§25).

A build need not offer every capability the tree contains: one can stay in the tree — editable,
tested, and switchable back on — while this table decides whether a user meets it. **The table is
empty today.** The capabilities it hid (Benchmarks, which scored method variants against ground
truth no user of the querying or clustering modules has; ``clip_query`` and ``clip_annotation``,
superseded by Query; ``syllable_discovery`` and pose clustering, superseded by temporal
clustering; and others) were removed from the tree instead. Their record,
``docs/removed_capabilities.md``, is not published in this repository.

**What kind of toggle this is.** A *permissioning* toggle keyed on build audience: long-lived,
statically decided, never per-request. Three things follow, and they remove most of the design
space:

- It is **source-controlled and developer-edited**. No runtime store, no admin UI, no dynamic
  evaluation — the decision travels with the build.
- It carries **no removal debt**. A *release* toggle is technical debt to rip out as soon as its
  feature lands; here the hidden state is the correct permanent one, so the usual "file the
  removal ticket when you add the flag" advice is inverted.
- It is **not a user preference**. There is no "show advanced features" checkbox; an expert mode
  owned by the user is a different feature.

**An entry is a capability, and a capability has manifestations.** One thing a user might reach
appears under different names in different surfaces: Benchmarks (since removed) was the GUI
section ``benchmarks`` and the CLI group ``benchmark``. Declaring those names on the entry keeps the mapping beside the
decision. An earlier draft used one flat id space, and the correspondence had to be re-stated as
an alias map inside the test — a mapping in the wrong place, and one that would go stale silently.

**Ask this at enumeration points, not at use sites.** A surface consults the table exactly once —
where it *lists* what exists — never in the code that uses one. That is what makes the parity
guard possible: a scattered check cannot be enumerated, so its completeness cannot be tested, and
the door nobody re-checks is the one reached by a table double-click rather than by the rail. It
is also why hiding a capability never makes existing work vanish: the toggle governs **launching,
not reading**, so a completed run of a hidden module still renders its result.

**User documentation is one of those enumeration points.** ``docs/cli.md`` and ``docs/gui.md``
describe surfaces in hand-written prose that no registry knows about. A build that hides a
capability everywhere else while its user guide still teaches it is worse than either state alone,
so an entry names the doc anchors it owns and a guard asserts they are gone.

**The Python API never consults this.** Tenet 2 makes the API the boundary and the GUI and CLI
surfaces over it; research studies drive the API directly, and a toggle that broke a
study is a toggle nobody would dare use.

**Working in a tree where something is hidden.** Four rules, for whoever changes code while an
entry is in effect:

- **Ask** ``is_surfaced(name)`` **before treating a surface as real.** A page reachable only from
  a hidden surface is unreachable, and a parity sweep that counts one is counting a surface no
  user has. Count ``HIDDEN`` rather than restating its contents from memory; a prose copy of the
  list goes stale, and ``tests/test_surfaces.py`` checks the copies that exist.
- **Hiding governs launching, not reading.** A saved run of a hidden module still opens and
  exports; never "clean up" its read path.
- **Hidden is not dead — do not drop its guards.** Hidden code that stops being guarded decays,
  and switching it back on becomes a surprise rather than a toggle.
- **Do not generalize a fix into one.** A sweep that reaches a hidden capability has found a
  deliberate non-fix. Say so where the fix records how far it reached (the ``fix-the-class``
  skill, step 3.5).

Kept dependency-free (``os`` and ``dataclasses`` only) so both the GUI and the CLI can import it,
and so resolution stays the guard's job rather than this module's.
"""

import os
import re
from dataclasses import dataclass, field

SHOW_ALL_ENV = "DETAIL_SHOW_ALL"


@dataclass(frozen=True)
class Hidden:
    """A capability this build does not surface, and where it would otherwise appear.

    ``until`` is **not an expiry date**. A permissioning toggle does not expire; the field records
    *what would unhide it*, which is what a later reader needs in order to tell a permanent entry
    (``None``) from one waiting on something.

    The manifestation fields are the names the capability goes by in each surface — a GUI
    ``section`` kind, its ``modules`` (plural: one capability can be a chain of registered
    module ids), a ``method`` id from one of the pluggable-implementation
    registries (``QUERY_METHODS``, the cluster-method ones, and ``TRACKERS``), ``cli`` command
    paths
    (space-separated, several because
    one capability can span groups — the removed benchmarks were both ``benchmark`` and
    ``clusterer sweep``, whose output only ``benchmark export`` could read), and ``docs`` as ``(path, anchor)`` pairs
    that must not appear while it is hidden. Anchors are declared rather
    than guessed at, because "comparison" names three different things in this repo — §18's old
    name for a benchmark, §24's group contrast, and the O(bouts²) DTW bout comparison — and a
    grep would fire on all of them.

    **``awaiting`` stages an entry without putting it in effect.** A capability that should be
    hidden once its replacement ships is declared now and left surfaced: the manifestations are
    validated and the doc anchors are checked to **exist**, so switching it on later is one line
    rather than fresh archaeology of where the thing appears. That is the failure staging
    prevents — a stale anchor string enforces nothing, and nobody notices until the surface was
    meant to be gone and is not.

    ``awaiting`` and ``until`` read alike and are opposites: ``awaiting`` is what must happen
    before this takes effect, ``until`` is what would undo it once it has.
    """

    reason: str
    until: str | None = None
    awaiting: str | None = None
    section: str | None = None
    #: Module ids, plural, because a capability is not always one module. The (removed) clusterer
    #: chain was
    #: ``cluster_fit`` -> ``cluster_apply`` -> ``cluster_eval``: one thing a user reaches, three
    #: registered ids. This was a single ``str``, so the entry could name only ``cluster_eval``
    #: — and the other two therefore answered ``is_surfaced`` with **True** while the GUI section
    #: and the CLI group that launch them were both hidden, which published two parameter pages
    #: to the user site for a capability no user can reach. A guard is only as complete as the
    #: declaration it reads, so the field has to be able to say the true thing.
    modules: tuple[str, ...] = field(default_factory=tuple)
    #: Pluggable implementation ids (``QUERY_METHODS`` / the cluster-method registries /
    #: ``TRACKERS`` / ``EMBEDDING_MODELS`` / the input-image ``CONSTRUCTORS``). Hiding one takes it out of the pickers and the
    #: generated method page; it stays resolvable, so a saved run that names it still opens and
    #: still reads (the toggle governs launching). One field rather than one per registry: they
    #: are the same shape — an implementation behind a protocol, picked by id in a form — and a
    #: parallel field per registry would need its own filter in every guard that sweeps this
    #: table. Plural for the reason ``modules`` is: a family of models or recipes is one choice a
    #: user meets as several registered ids, and a single-id field could only hide one.
    methods: tuple[str, ...] = field(default_factory=tuple)
    cli: tuple[str, ...] = field(default_factory=tuple)
    docs: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def in_effect(self) -> bool:
        """Whether this entry hides anything yet. A staged entry (``awaiting`` set) does not."""
        return self.awaiting is None

    def manifestations(self) -> tuple[str, ...]:
        """Every name this capability answers to, for the reverse index."""
        return tuple(
            n for n in (self.section, *self.methods, *self.modules, *self.cli) if n
        )


#: Empty. Every capability this table hid was removed from the tree on 2026-11-10 (benchmarks,
#: the behavior clusterers, behavior query, behavior discovery, the `dtw` query method, pose
#: clustering and six earlier ones); `docs/removed_capabilities.md`, which says what each was and
#: how to restore it, is not published in this repository. The mechanism stays, tested on probe entries, for the next thing a build should
#: carry without offering.
HIDDEN: dict[str, Hidden] = {}


def show_all() -> bool:
    """Whether the escape hatch is set.

    Read from the environment **at call time**, never snapshotted into a module constant: that
    would freeze whatever the environment held when this module was first imported — the cache-bug
    shape recorded in ``CLAUDE.md`` — and would make the setting untestable.
    """
    return os.environ.get(SHOW_ALL_ENV, "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def _index() -> dict[str, str]:
    """``manifestation name -> capability id``, including each capability's own id.

    Built per call rather than cached, so a test that edits ``HIDDEN`` sees its edit — the same
    reason :func:`show_all` reads the environment late. The table is a handful of entries.
    """
    out: dict[str, str] = {}
    for capability, entry in HIDDEN.items():
        if not entry.in_effect:
            continue
        for name in (capability, *entry.manifestations()):
            out[name] = capability
    return out


def entry_for(name: str) -> Hidden | None:
    """The hidden entry ``name`` belongs to — by capability id or by any manifestation."""
    capability = _index().get(name)
    return None if capability is None else HIDDEN[capability]


# --------------------------------------------------------------------------- #
# Prose markers — the docs' equivalent of the filters above (§25)
#
# The module pages and the CLI hide themselves by *filtering* a registry. The general pages —
# gui.md, concepts.md, cli.md, glossary.md — are hand-written, so the only lever used to be
# deleting the sentence and declaring the deleted string on the entry's `docs=`. That
# contradicts what this table is for: "editable, tested, and switchable back on" is not true of
# prose someone has to rewrite from memory.
#
# So the prose is marked and stripped at build time instead. The logic lives *here*, not in the
# mkdocs hook, because three readers need to agree about what a marker means: the site build
# (scripts/docs_surface_hook.py), the guard (tests/test_surfaces.py), and **the in-app help
# panel** — `help.extract_section` renders from the same source files, and prose it kept while
# the site dropped it would be a second door onto a hidden capability, which is the failure
# this module exists to prevent.
# --------------------------------------------------------------------------- #

_OPEN = r"<!--\s*surface:(?P<name>[A-Za-z0-9_]+)\s*-->"
_CLOSE = r"<!--\s*/surface\s*-->"

#: One marked block: the opening marker, its content, and the close. Non-greedy so adjacent
#: blocks do not swallow each other, and ``DOTALL`` so a block spans paragraphs.
BLOCK = re.compile(_OPEN + r"(?P<body>.*?)" + _CLOSE, re.DOTALL)

OPEN_MARKER = re.compile(_OPEN)
CLOSE_MARKER = re.compile(_CLOSE)


class UnbalancedMarkerError(ValueError):
    """A ``surface:`` marker with no partner.

    Raised rather than tolerated: an unclosed opener would otherwise swallow the rest of the
    page silently, which reads as "the docs lost a section" long after the edit that did it.
    """


def check_balanced(text: str, where: str = "<text>") -> None:
    """Raise if the markers in ``text`` do not pair up."""
    opens = len(OPEN_MARKER.findall(text))
    closes = len(CLOSE_MARKER.findall(text))
    if opens != closes:
        raise UnbalancedMarkerError(
            f"{where}: {opens} <!--surface:…--> opener(s) and {closes} <!--/surface--> "
            "closer(s); an unclosed marker hides the rest of the page"
        )


def marked_capabilities(text: str) -> set[str]:
    """Every capability id named by a marker in ``text``."""
    return {m.group("name") for m in OPEN_MARKER.finditer(text)}


def strip_hidden(text: str, is_surfaced=None, where: str = "<text>") -> str:
    """Remove blocks whose capability this build hides; unwrap the rest.

    ``is_surfaced`` defaults to :func:`detail.surfaces.is_surfaced` and is injectable so a test
    can strip against a hypothetical build without mutating the table.
    """
    if is_surfaced is None:
        is_surfaced = globals()["is_surfaced"]
    check_balanced(text, where)

    def _replace(match: re.Match) -> str:
        # Shown: keep the body, drop the markers. Hidden: drop the lot.
        return match.group("body") if is_surfaced(match.group("name")) else ""

    return BLOCK.sub(_replace, text)


def is_surfaced(name: str) -> bool:
    """Whether this build offers ``name`` to a user.

    Accepts a capability id or any of its manifestation names, so a call site passes whatever it
    already has — a section kind, a module id — and needs no translation table of its own.
    """
    return show_all() or entry_for(name) is None


def hidden_reason(name: str) -> str | None:
    """Why ``name`` is hidden, or ``None`` if nothing hides it.

    Answers even when :func:`show_all` is set, so a developer looking at a revealed surface can
    still see why it is normally absent.
    """
    entry = entry_for(name)
    return None if entry is None else entry.reason


def hidden_names() -> frozenset[str]:
    """Every capability id **and** manifestation name that is hidden, for parity scanning."""
    return frozenset(_index())


def surfaced(names):
    """``names`` filtered to those this build shows, order preserved."""
    return [n for n in names if is_surfaced(n)]


def apply_cli_visibility(group) -> list[str]:
    """Mark every hidden capability's click command ``hidden``; return the paths marked.

    Click's ``hidden`` keeps a command out of ``--help`` **and** out of the generated reference —
    mkdocs-click decides on ``ctx.command.hidden``, not on ``Group.list_commands``, so a group
    subclass would hide it from the terminal and still publish it to the docs site — while leaving
    it executable, which is what a research sweep needs and a user never encounters.

    Applied here from the table rather than as ``hidden=`` on each decorator: one seam instead of a
    surface id hardcoded per command, and re-callable, so a test can change the table and re-apply
    rather than needing a fresh import.
    """
    marked: list[str] = []
    for entry in HIDDEN.values():
        if not entry.in_effect:
            continue
        for path in entry.cli:
            command = group
            for part in path.split():
                command = getattr(command, "commands", {}).get(part)
                if command is None:
                    break
            if command is None:
                continue
            command.hidden = not show_all()
            marked.append(path)
    return marked
