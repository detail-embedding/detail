"""A module that declares a §19 artifact writes one, and something proves it does.

The class: **a producer emits the shared artifact but not one of its components, so a whole layer
silently refuses on its runs.**

Every §19 producer returns `behavior_output_artifacts()` from `output_manifest`, and that one list
declares every component — so **declaring is free and writing is not**, for each of them
independently. That is the whole defect, and it has now happened three times:

- Pose clustering declared `frames` and never wrote one. Nothing failed: the run completed,
  `instances.csv` and `coverage.csv` were correct, and the run page offered a "Per-frame view"
  artifact that was not on disk. The only symptom was one layer up, where `analysis/measures.py`
  refused every *windowed* measure on a pose run while answering the same question on a Query run.
- Temporal clustering landed hours after that fix with the same defect, and the first version of
  this test caught it — red in CI on the very PR that added it.
- Pose clustering then declared `instance_features` and never wrote one, which is why this file is
  no longer about frames. The symptom was again one layer up and again invisible: the "Where it
  happens" band hides itself when a run reports no positions, so the spatial map users had
  asked for by name was simply absent from the pose screen, with nothing to click and nothing to
  read.

Three instances, one shape. So the guard is **parameterized over the components** rather than
written once per component: a fourth kind added to `behavior_output_artifacts()` fails here until
`_COVERED` names a test for every producer of it, which is the only version of this that a new
component inherits for free.

**Why this file holds only the registry check.** The invariants themselves — that a frame view's
denominator agrees with coverage, that a position is the mean over its bout — have to be asserted
against a *real* run, and each producer already builds one in its own test module. Rebuilding them
here to centralize the assertion would pay for a second real fit to say the same thing twice. What
is missing without this file is the part none of those can know: that they are the *complete* set.
So `_COVERED` is a claim in the `tests/tiers.py` spirit — an entry names the test that does the
proving, and a producer with no entry fails here rather than being quietly uncovered.
"""

import ast
import pathlib

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.executor.runners import ANALYSIS_MODULES, import_runner_modules

import_runner_modules(PIPELINE_RUNNERS)

REPO = pathlib.Path(__file__).resolve().parents[2]

#: artifact kind -> {module_id: the test that proves this producer writes it}. Adding a producer
#: means adding a line here *and* the test it names; neither alone passes. Adding a *kind* to
#: `behavior_output_artifacts()` means adding a block, which is the point — the omission fails
#: loudly instead of being discovered a layer up, months later.
_COVERED = {
    "frames": {
        "query": (
            "tests/api/test_query.py",
            "test_the_frame_view_denominator_agrees_with_coverage",
        ),
        "temporal_clustering": (
            "tests/pipeline/test_temporal_clustering_module.py",
            "test_the_frame_view_denominator_agrees_with_coverage",
        ),
    },
    "instance_features": {
        "query": (
            "tests/api/test_query.py",
            "test_a_run_writes_where_each_instance_happened",
        ),
        "temporal_clustering": (
            "tests/pipeline/test_temporal_clustering_module.py",
            "test_a_fit_writes_where_each_instance_happened",
        ),
    },
}


def _declares(kind: str) -> set[str]:
    """Modules whose `output_manifest` advertises artifact `kind`."""
    return {
        module_id
        for module_id, module in ANALYSIS_MODULES.items()
        if any(e.get("kind") == kind for e in module.output_manifest("an_01", {}))
    }


def test_the_registry_is_populated():
    """A data-driven guard over an empty registry passes by testing nothing."""
    assert (
        ANALYSIS_MODULES
    ), "no analysis modules registered — the sweep below proves nothing"
    assert _declares("frames"), "no module declares the frames artifact"


@pytest.mark.parametrize("kind", sorted(_COVERED))
def test_every_producer_of_an_artifact_is_covered(kind):
    declaring = _declares(kind)
    claimed = set(_COVERED[kind])
    assert declaring == claimed, (
        f"for artifact {kind!r}: these modules declare it but no test proves they write one: "
        f"{sorted(declaring - claimed)}; and these are claimed but no longer declare it: "
        f"{sorted(claimed - declaring)}. Declaring is free and writing is not — the gap shows up "
        f"a layer away, on a surface that refuses or hides itself, with nothing failing at the "
        f"line where it was written."
    )


def test_every_declared_kind_is_guarded():
    """The kinds are declared in one list, so a new component must arrive here too. Without this
    the sweep above only covers what someone remembered to add, which is the failure it exists to
    prevent, one level up."""
    declared = {
        e.get("kind")
        for module in ANALYSIS_MODULES.values()
        for e in module.output_manifest("an_01", {})
    }
    producer_kinds = declared & {"frames", "instance_features", "instances", "coverage"}
    unguarded = producer_kinds - set(_COVERED) - {"instances", "coverage"}
    assert not unguarded, (
        f"these artifact components are declared by a producer and guarded by nothing: "
        f"{sorted(unguarded)}. `instances` and `coverage` are written by "
        f"`write_behavior_output` itself, so a producer cannot declare them without writing them."
    )


@pytest.mark.parametrize(
    "kind,module_id",
    sorted((k, m) for k, mods in _COVERED.items() for m in mods),
)
def test_the_claimed_test_exists(kind, module_id):
    """Guard the guard. `_COVERED` is prose until something checks that the tests it names are
    real — a renamed test would leave this file passing while covering nothing."""
    rel, name = _COVERED[kind][module_id]
    path = REPO / rel
    assert path.is_file(), f"{rel} does not exist"
    names = {
        node.name
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert name in names, (
        f"{rel} has no test named {name!r}, which _COVERED claims proves {module_id!r} writes "
        f"the {kind!r} artifact"
    )
