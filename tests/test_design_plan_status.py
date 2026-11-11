"""A Status row's verdict cannot contradict the code it describes.

The class: **a hand-maintained claim about what exists, with nothing checking it.**

`DESIGN_PLAN.md`'s Status table is written when a section is *designed* and nobody thinks to
edit it when the section is *built*, because building it touches no line of that table. It
drifted across seven merged PRs -- #207 and #209 (measures, comparison core), #210 (the §18
rename), #213 (the saved comparison), #218/#222 (query calibration), #223/#226/#228 (pose
clustering) -- and went on telling every reader that §20, §21, §22, §23 and §24 were "not
built" and that the §18 rename was "not applied in code yet", months after all of it shipped.

Nothing failed, which is the whole problem. The table is prose, the code is correct, and the
only symptom is a person planning work that already exists. `CLAUDE.md` names
`DESIGN_PLAN.md` as the authoritative specification and tells every agent to read it first,
so a stale row is not a cosmetic defect: it is the first thing anyone is told, and it is
wrong.

**Why a declared verdict and not a grep for "not built".** That was the first version of this
guard and it failed twice over in one run. It fired on rows that were *correct* -- §22's
"Temporal not built" and §23's "the standalone API ... are not built" both report an unbuilt
**sub-part** of a built section, which is exactly what a good row does. And it would have
missed the drift it was written for: §19's stale row said "**No producer emits the artifact
yet**", which matches no spelling of "not built". Scanning free prose for a phrase is leaky in
both directions, so the row declares its verdict in a closed vocabulary instead and the prose
after it is free to say anything -- the same "declare it, don't guess it" trade `surfaces.py`
makes for its doc anchors.

The residual limit is worth naming: a verdict check reaches the verdict. §19's own drift sat
in *trailing* prose ("No producer emits the artifact yet") under a verdict of "Storage layer
built" that was true as far as it went, and nothing short of reading the sentence would have
caught it. Seven of the eight probed rows drifted in the verdict and are covered.

**One direction is still all that is asserted.** "Built" is machine-checkable; "not built" is
not, and a probe that tried to prove a feature's absence would pass against a plausible stub.
So a section with no probe below is unchecked, and §17 is the honest example: there is no
registry to ask about a COCO exporter that does not exist.
"""

import pathlib
import re

import pytest

from detail.analysis import measures
from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.api.experiment import Experiment
from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

import_runner_modules(PIPELINE_RUNNERS)

REPO = pathlib.Path(__file__).resolve().parents[1]
DESIGN_PLAN = REPO / "DESIGN_PLAN.md"

#: A probed row opens with a bolded verdict. Anything inside it that reads as an absence
#: ("not built", "not applied") makes the verdict negative; otherwise a verdict naming
#: "built" or "implemented" is positive. Prose *after* the verdict is unconstrained, which
#: is what lets a row report an unbuilt sub-part without tripping this.
VERDICT = re.compile(r"^\*\*(?P<verdict>[^*]+)\*\*")
ABSENT = re.compile(r"not (?:yet )?(?:built|applied|implemented)", re.IGNORECASE)
PRESENT = re.compile(r"built|implemented", re.IGNORECASE)


def _writes_behavior_output() -> bool:
    """a pipeline module calls write_behavior_output"""
    return any(
        "write_behavior_output(" in path.read_text(encoding="utf-8-sig")
        for path in (REPO / "detail" / "pipeline").glob("*.py")
    )


#: Section marker -> (probe, what the probe means). Keep each probe to the cheapest fact
#: that could not be true of an unbuilt section.
PROBES = {
    "§16": (
        lambda: "location_export" in ANALYSIS_MODULES,
        "an analysis module is registered",
    ),
    "§19": (_writes_behavior_output, "a pipeline module writes the artifact"),
    "§20": (lambda: "query" in ANALYSIS_MODULES, "the query module is registered"),
    "§21": (
        lambda: any(m.mutable for m in ANALYSIS_MODULES.values()),
        "a module declares itself an authored run",
    ),
    "§22": (
        lambda: "temporal_clustering" in ANALYSIS_MODULES,
        "the temporal-clustering module is registered",
    ),
    "§23": (lambda: bool(measures.MEASURES), "the measure registry is populated"),
    "§24": (
        lambda: hasattr(Experiment, "run_comparison"),
        "Experiment.run_comparison exists",
    ),
}


def _status_rows(text: str) -> dict[str, str]:
    """``{section marker: the row's status cell}`` for every Status row carrying one."""
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        marker = re.search(r"§[\d.]+", cells[0])
        if marker:
            rows[marker.group()] = cells[1]
    return rows


def _polarity(claim: str) -> bool | None:
    """``True`` built, ``False`` not built, ``None`` when the row declares no verdict."""
    match = VERDICT.match(claim)
    if not match:
        return None
    verdict = match.group("verdict")
    if ABSENT.search(verdict):
        return False
    return True if PRESENT.search(verdict) else None


ROWS = _status_rows(DESIGN_PLAN.read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("section", sorted(PROBES))
def test_a_built_section_declares_itself_built(section):
    probe, means = PROBES[section]
    claim = ROWS[section]
    polarity = _polarity(claim)
    assert polarity is not None, (
        f"DESIGN_PLAN.md's Status row for {section} opens with no verdict this guard can "
        f"read. A probed row starts with a bolded verdict naming 'built' or 'implemented' "
        f"(or their negation); everything after it is free prose.\n  row: {claim[:200]}"
    )
    if not probe():
        return  # genuinely unbuilt: the row may say so, and nothing here can prove it wrong
    assert polarity, (
        f"DESIGN_PLAN.md's Status row for {section} reports it unbuilt, but {means}. "
        f"CLAUDE.md sends every reader to this table first, so a stale row is the first "
        f"thing they are told.\n  row: {claim[:200]}"
    )


def test_every_probed_section_has_a_row():
    """Guard the guard, half one. Every check above hangs off finding the row, so a
    renumbered or reformatted table would silently check nothing rather than failing."""
    missing = sorted(set(PROBES) - set(ROWS))
    assert not missing, (
        f"no Status row found for {missing} -- the table's shape changed and this guard "
        f"stopped covering those sections rather than failing"
    )


def test_the_verdict_reader_tells_the_cases_apart():
    """Guard the guard, half two: the shapes the table actually drifted in, and the ones it
    legitimately uses, have to come out on opposite sides."""
    # The drift, in the three spellings it really took.
    assert _polarity("**Not built.** Approved (design) only") is False
    assert _polarity("**Rename not applied in code yet** — §24 takes the word") is False
    assert _polarity("**Not yet implemented**; the spec is settled") is False
    # A built section, including one that reports an unbuilt *sub-part* afterwards.
    assert (
        _polarity("**Built** — ten measures. The standalone API is not built.") is True
    )
    assert _polarity("**Pose built** — temporal not built") is True
    assert _polarity("**Built except the GUI** — every other surface ships") is True
    # A row with no verdict is reported as unreadable, never quietly passed.
    assert _polarity("Approved (design); not built.") is None
