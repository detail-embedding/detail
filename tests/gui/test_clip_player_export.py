"""Every clip player offers to write the clip it is showing.

**The class: a capability added to a shared widget that reaches one surface.** This repo has
paid for it repeatedly — the spatial map's field-of-view outline, the decode scrim's
suppression, `ConfigWidget.load_config`. The shape is always the same: the capability arrives as
something an *embedder* opts into, so the surface being worked on gets it and the other twelve
go on without, and nothing fails.

So the export is not opt-in. It is a control `_ClipPlayer` builds unconditionally, which makes
"every surface has it" true by construction rather than by thirteen call sites remembering. What
is asserted here is that it *stays* unconditional: a constructor flag that could turn it off
would put the class straight back.

**And it opens the shared dialog rather than exporting.** The rule the export design actually
protects is one *implementation* — "seven bespoke export dialogs, each on the page whose author
needed one, is how the capability ended up existing twice and reachable from neither page a user
was on." A second trigger is fine; a second writer is the bug. `test_the_button_opens_the_one
_shared_dialog` is that line.
"""

import ast
import inspect
import pathlib

import pytest

pytest.importorskip("PyQt6")

from detail.api.exporters import ExportScope  # noqa: E402
from detail.gui.behavior_query import ClipSpec, _ClipPlayer  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
GUI = REPO / "detail" / "gui"


def _embedders() -> set[str]:
    """Files that construct a `_ClipPlayer`, found by parsing rather than by substring.

    A name-in-text sweep is the weaker form and this repo has one that proves it: matching a
    module's filename stem against the concatenated test tree accepts almost anything, because
    an unrelated mention of the stem satisfies it.
    """
    found = set()
    for path in sorted(GUI.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                == "_ClipPlayer"
            ):
                found.add(path.relative_to(REPO).as_posix())
    return found


def _clip() -> ClipSpec:
    return ClipSpec(
        recording_id="recording_001",
        track_id=0,
        embed_run_id="emb_01",
        start_t_ns=1_000_000_000,
        end_t_ns=3_000_000_000,
        label="grooming",
    )


def test_the_export_is_not_something_an_embedder_opts_into(qapp):
    """The guard for the class. Thirteen surfaces get the button because none of them choose it.

    If a `no_export=` / `exportable=` parameter ever appears, the capability becomes opt-in and
    the twelve surfaces nobody was working on lose it silently.
    """
    embedders = _embedders()
    # Five since the hidden capabilities' GUIs (clusterers, behavior query and discovery) were
    # removed; the floor is there to catch the sweep going blind, not to count surfaces.
    assert (
        len(embedders) >= 5
    ), f"the sweep stopped finding embedders: {sorted(embedders)}"

    params = set(inspect.signature(_ClipPlayer.__init__).parameters)
    opt_outs = {p for p in params if "export" in p.lower()}
    assert not opt_outs, (
        f"`_ClipPlayer.__init__` grew {sorted(opt_outs)}, which makes the export something an "
        f"embedder opts into — the shape that reaches one surface and leaves the other "
        f"{len(embedders) - 1} without it."
    )


def test_a_player_offers_the_export_once_it_has_a_clip(qapp, authored_cohort):
    """Disabled with nothing on screen, enabled once there is. A button that offers to export
    nothing is the empty-state bug one control over."""
    player = _ClipPlayer(authored_cohort.experiment, context_s=0)
    assert not player._export.isEnabled(), "offered an export before there was a clip"
    assert player.export_scope() is None

    player.set_clip(_clip())
    assert player._export.isEnabled()


def test_the_scope_is_the_clip_on_screen(qapp, authored_cohort):
    """What the button exports and what the shell's `Export…` would narrow to are the same span.

    Two doors onto one answer: `gui/export.py::scope_of` searches a page's children, so a
    surface holding a player answers with the player's clip without forwarding anything.
    """
    player = _ClipPlayer(authored_cohort.experiment, context_s=0)
    player.set_clip(_clip())

    scope = player.export_scope()
    assert isinstance(scope, ExportScope)
    assert scope.frames == player.clip_span()
    assert not scope.is_all(), "a clip is not the whole recording"
    assert scope.describe().startswith("frames "), scope.describe()


def test_the_flavour_follows_the_picture(qapp, authored_cohort, monkeypatch):
    """Masks on writes the overlay, masks off writes the raw footage.

    The file matches what was on screen when the button was pressed — the alternative is a
    second style control that can disagree with the one the user is already looking at.
    """
    from detail.api import clip_render
    import detail.gui.export as gui_export

    # Patched on the module, not on a name in `behavior_query`: `_export_clip` imports
    # `open_export` when it runs, so the module attribute is what it will reach.
    seen: list[dict] = []
    monkeypatch.setattr(
        gui_export,
        "open_export",
        lambda *a, **k: seen.append(k.get("seed") or {}),
    )

    player = _ClipPlayer(authored_cohort.experiment, masked=True, context_s=0)
    player.set_clip(_clip())
    player._export_clip()
    assert seen[-1]["renders"] == [clip_render.OVERLAY]

    player._masks.setChecked(False)
    player._export_clip()
    assert seen[-1]["renders"] == [clip_render.RAW]


def test_the_button_opens_the_one_shared_dialog(qapp, authored_cohort, monkeypatch):
    """It submits through `open_export`, never through a writer of its own.

    `exporters.write_export` is what a bespoke exporter would reach for; the player must not.

    **Driven by clicking**, not by calling `_export_clip`. Calling the handler proves the
    handler works and says nothing about whether anything reaches it — this repo's recorded
    class is a `signal.connect(self._method)` whose method was renamed, which only a
    construction (or a click) catches. Written the other way first; disconnecting the button
    left it green.
    """
    from detail.api import exporters
    import detail.gui.export as gui_export

    opened: list[tuple] = []
    monkeypatch.setattr(
        gui_export, "open_export", lambda *a, **k: opened.append((a, k))
    )
    monkeypatch.setattr(
        exporters,
        "write_export",
        lambda *a, **k: pytest.fail("the player wrote an export itself"),
    )

    player = _ClipPlayer(authored_cohort.experiment, context_s=0)
    player.set_clip(_clip())
    player._export.click()

    assert len(opened) == 1
    _args, kwargs = opened[0]
    assert kwargs["only"] == ("clip",), "the dialog should open on the clip target"
    seed = kwargs["seed"]
    assert seed["embed_run_id"] == "emb_01"
    assert seed["track_id"] == 0
    assert (
        seed["crop"] is False
    ), "a clip keeps the field of view unless asked otherwise"


def test_the_seed_survives_a_yaml_round_trip(qapp, authored_cohort, monkeypatch):
    """Everything the button seeds rides an `ExportJob` through YAML.

    A tuple there writes as `!!python/tuple`, which `yaml.safe_load` then refuses — the run does
    the work and dies reporting it. That has happened once already, to `renders`.
    """
    import yaml

    import detail.gui.export as gui_export

    seen: list[dict] = []
    monkeypatch.setattr(
        gui_export, "open_export", lambda *a, **k: seen.append(k.get("seed") or {})
    )
    player = _ClipPlayer(authored_cohort.experiment, context_s=0)
    player.set_clip(_clip())
    player._export_clip()

    text = yaml.safe_dump(seen[-1])
    assert "!!python" not in text, text
    assert yaml.safe_load(text) == seen[-1]
