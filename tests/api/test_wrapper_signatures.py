"""A method that forwards to a function forwards *all* of it, or says why not.

**The class.** `Experiment` and `Recording` are thin doors onto the `api.*` / `pipeline.*`
functions that do the work, and each door hand-lists the keywords it passes through. Add a
parameter to the function and the door does not grow one: the parameter is reachable from the
function and unreachable from the object every caller actually holds.

**How it hides.** The function's own tests keep passing, because a test written next to a
function calls that function. `run_comparison` gained a `stop_check` so a long comparison could
be cancelled; it shipped with a test that called `run_comparison(exp, ...)` directly and a GUI
that called `exp.run_comparison(...)`, and the first person to press Cancel got
`Experiment.run_comparison() got an unexpected keyword argument 'stop_check'`. Nothing in the
suite could have caught it, because nothing in the suite used the door the app uses. This is the
"verify through the real entry point" rule in CLAUDE.md, in the one place where the real entry
point is a different callable with the same name.

**The rule.** A keyword-only parameter on the target is a keyword-only parameter on the wrapper,
unless `ALLOWED` below says otherwise and says why. The exemptions are claims, not opt-outs: an
entry naming a parameter the target no longer has fails, so a stale excuse cannot sit here
quietly the way a dropped keyword did.
"""

import ast
import inspect
import pathlib

import pytest

import detail  # noqa: F401 (populates the registries the api modules import)

#: (class, method, parameter) -> why the wrapper legitimately does not forward it.
ALLOWED = {
    (
        "PreviewClip",
        "_draw_points",
        "edge_color",
    ): "A private drawing step, not a door: a tile draws the subject's keypoints the one way the "
    "montage shows them (scores dim, the edge colour picked by footage or white); the keypoints' "
    "look is the drawer's own, chosen once in ``visualize``.",
    (
        "PreviewClip",
        "_draw_points",
        "hollow",
    ): "A private drawing step, not a door: a tile draws the subject's keypoints the one way the "
    "montage shows them (scores dim, the edge colour picked by footage or white); the keypoints' "
    "look is the drawer's own, chosen once in ``visualize``.",
    (
        "PreviewClip",
        "_draw_points",
        "radius",
    ): "A private drawing step, not a door: a tile draws the subject's keypoints the one way the "
    "montage shows them (scores dim, the edge colour picked by footage or white); the keypoints' "
    "look is the drawer's own, chosen once in ``visualize``.",
    (
        "PreviewClip",
        "_draw_points",
        "scores",
    ): "A private drawing step, not a door: a tile draws the subject's keypoints the one way the "
    "montage shows them (scores dim, the edge colour picked by footage or white); the keypoints' "
    "look is the drawer's own, chosen once in ``visualize``.",
    (
        "PreviewClip",
        "_draw_points",
        "visible",
    ): "A private drawing step, not a door: a tile draws the subject's keypoints the one way the "
    "montage shows them (scores dim, the edge colour picked by footage or white); the keypoints' "
    "look is the drawer's own, chosen once in ``visualize``.",
    (
        "PreviewClip",
        "_matrix",
        "recorded",
    ): "A private placement step, not a door: ``recorded`` is the tile's own as-recorded "
    "placement, the clip's fixed crop (``_crop_matrix``), which a side view keeps; a caller has "
    "no other placement to give it.",
}


def _sources():
    """Every module under `detail/api/` — the objects a scripted caller holds.

    Derived, not listed. The first version named `experiment.py` and `recording.py`, so the
    wrapper shape it exists to catch went unchecked everywhere else: `Comparison.export` and
    `Comparison.drift` live in `api/comparison.py` and were never swept.
    """
    root = pathlib.Path(detail.__file__).parent / "api"
    return sorted(
        p.relative_to(root.parent.parent).as_posix()
        for p in root.glob("*.py")
        if not p.name.startswith("_")
    )


_SOURCES = _sources()


def _delegations():
    """(class, method, target) for every method whose body imports a detail.* callable and
    returns a call to it — the wrapper shape this file is about."""
    root = pathlib.Path(detail.__file__).parent.parent
    for relative in _SOURCES:
        tree = ast.parse((root / relative).read_text(encoding="utf-8-sig"))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                imported = {}
                for node in ast.walk(fn):
                    if isinstance(node, ast.ImportFrom) and (
                        node.module or ""
                    ).startswith("detail."):
                        for alias in node.names:
                            imported[alias.asname or alias.name] = (
                                node.module,
                                alias.name,
                            )
                for node in ast.walk(fn):
                    if not (
                        isinstance(node, ast.Return)
                        and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                    ):
                        continue
                    key = node.value.func.id
                    if key not in imported:
                        continue
                    module_name, target_name = imported[key]
                    target = getattr(
                        __import__(module_name, fromlist=[target_name]),
                        target_name,
                        None,
                    )
                    if not callable(target) or inspect.isclass(target):
                        continue
                    yield cls.name, fn, target


CASES = [pytest.param(c, fn, t, id=f"{c}.{fn.name}") for c, fn, t in _delegations()]


def test_the_sweep_finds_wrappers():
    """Guard the guard: an AST walk that stops matching passes vacuously forever.

    The shape matched is deliberately narrow — a local `from detail... import f` plus a
    `return f(...)` — so this count is a floor on that shape, not a census of every wrapper.
    It exists to fail loudly if a refactor moves the imports to module scope and the sweep
    silently starts covering nothing.
    """
    assert len(CASES) >= 5, f"only {len(CASES)} wrappers found; the AST shape changed"


def _dropped(cls_name, fn, target) -> set[str]:
    """Keyword-only parameters the target has and the wrapper does not forward."""
    theirs = {
        name
        for name, p in inspect.signature(target).parameters.items()
        if p.kind is p.KEYWORD_ONLY
    }
    ours = {a.arg for a in fn.args.kwonlyargs}
    return {name for name in theirs - ours if (cls_name, fn.name, name) not in ALLOWED}


@pytest.mark.parametrize("cls_name,fn,target", CASES)
def test_a_wrapper_forwards_every_keyword(cls_name, fn, target):
    dropped = _dropped(cls_name, fn, target)
    assert not dropped, (
        f"{cls_name}.{fn.name} does not forward {sorted(dropped)} to "
        f"{target.__module__}.{target.__name__} — a caller holding the object cannot reach "
        f"it. Forward it, or add an entry to ALLOWED saying why not."
    )


def test_every_exemption_still_excuses_something():
    """A reason for a parameter that no longer exists is a stale claim, and stale claims are
    how the dropped keyword survived in the first place."""
    live = {
        (cls_name, fn.name, name)
        for cls_name, fn, target in _delegations()
        for name, p in inspect.signature(target).parameters.items()
        if p.kind is p.KEYWORD_ONLY and name not in {a.arg for a in fn.args.kwonlyargs}
    }
    stale = set(ALLOWED) - live
    assert not stale, f"ALLOWED entries excusing nothing: {sorted(stale)}"


def test_a_dropped_keyword_would_be_caught():
    """Guard the guard: every wrapper in the repo forwards everything today, so the sweep above
    is a set of passing assertions with nothing proving they *can* fail. Hand the predicate a
    wrapper that drops one and watch it say so.
    """

    def target(a, *, kept=1, dropped=2):
        return a

    wrapper = ast.parse(
        "def method(self, a, *, kept=1):\n"
        "    from detail.x import target\n"
        "    return target(a, kept=kept)\n"
    ).body[0]

    assert _dropped("Fake", wrapper, target) == {"dropped"}
    # And the positive: a wrapper that forwards both is clean.
    full = ast.parse(
        "def method(self, a, *, kept=1, dropped=2):\n"
        "    return target(a, kept=kept, dropped=dropped)\n"
    ).body[0]
    assert _dropped("Fake", full, target) == set()


def test_the_source_sweep_reaches_every_api_module():
    """`_SOURCES` was two hand-written filenames, so `api/comparison.py`'s wrappers -- which had
    a real dropped keyword -- were never swept. Deriving it is the fix; this is the claim that
    the derivation still reaches more than the two it started with."""
    assert len(_SOURCES) >= 5, _SOURCES
    assert any(s.endswith("comparison.py") for s in _SOURCES), _SOURCES
