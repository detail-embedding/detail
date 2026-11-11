"""A parameter's default is declared on its config model, never typed into a surface.

**The class.** ``detail/configs/`` is the single source for a parameter's type, bounds, help
and *default*, and 84 of the CLI's options already read theirs from it
(``default=_QRY["method"].default``). A surface that writes the value in by hand instead is
a second copy of a fact, and the two drift without anything failing: the run launches, the
config validates, the result is merely computed with a setting nobody chose.

**How it hid.** ``TrackingConfig.tracker_id`` was declared *required* — so the model had no
default to read — and each surface then supplied its own anyway. The CLI hardcoded
``default="iou_tracker"`` in the decorator; the GUI combo called ``addItems(sorted(TRACKERS))``
and inherited index 0, which is ``anchor_tracker``. So the same tool answered "what does a
default tracking run do?" two different ways depending on which surface you asked, one of them
decided by alphabetical order, and no test could see it because both answers are legal
trackers that run to completion. It surfaced only when a user reported losing frames "on the
default settings" and the default turned out not to be a single thing.

**The rule.** Where a CLI option and a config-model field name the same parameter, the option
may not supply a default the model does not declare, and where both declare one they must
agree. Data-driven over ``iter_config_models()`` and the command tree, so a module added
tomorrow is covered the day it lands rather than the day someone remembers this file.

**Reading a default correctly matters.** ``FieldInfo.default`` is ``PydanticUndefined`` for a
required field *and* for one declared with ``default_factory``. Comparing against it naively
reports ``boundary_margins``, ``min_cluster_sizes`` and ``windows`` as hardcoded when they are
not — three false positives that would have been silenced with three exemptions, which is how
a guard stops biting. Use ``is_required()`` / ``get_default(call_default_factory=True)``.
"""

import click
import pytest
from click.core import UNSET
from pydantic import BaseModel

from detail.cli import main as cli_main
from detail.configs.describe import iter_config_models

#: An option whose click default is deliberately *not* its model's, with the reason. An entry
#: that stops excusing anything fails below — the standard ``FILE_TIER`` and
#: ``UNIT_EXEMPTIONS`` are held to, so this cannot quietly become a way of muting the guard.
#:
#: **Empty, and that is the finding.** Every apparent exception turned out to be a shape the
#: comparison should understand rather than a case to excuse: click's ``UNSET``, a
#: ``multiple=True`` tuple against a list, an option that is a different representation of a
#: nested model. Each of those would have been a plausible entry here, and each would have
#: muted a real check for every other option of its kind.
EXEMPT: dict[tuple[str, str], str] = {}


def _commands(group, prefix=""):
    """Every command in the tree, keyed by its full path."""
    for name, command in getattr(group, "commands", {}).items():
        path = f"{prefix}{name}"
        yield path, command
        yield from _commands(command, f"{path} ")


def _pairs():
    """``(command path, command, model)`` for every config model that owns a CLI command.

    Derived from the module id rather than listed, so this cannot go stale against a module
    someone adds. A model with no command of its own — a method config reached through a
    parent's ``--method-param`` — simply has no options to disagree with.

    A processing module is also launched over a recording *selection* by ``process <module>``,
    a second command with its own copy of every option: ``embedding run`` and ``process
    embedding`` each declared ``--model`` separately. Pairing only the first match left the
    second copy free to invent a default, so every ``process`` command is paired too.
    """
    commands = dict(_commands(cli_main))
    out = []
    for module_id, model in sorted(iter_config_models()):
        dashed = module_id.replace("_", "-")
        for path in (f"{dashed} run", f"analysis {dashed}", dashed):
            if path in commands:
                out.append((path, commands[path], model))
                break
        selection = f"process {dashed}"
        if selection in commands:
            out.append((selection, commands[selection], model))
    return out


PAIRS = _pairs()


def _model_default(field):
    """``(declares_a_default, value)`` — resolving ``default_factory``."""
    if field.is_required():
        return False, None
    return True, field.get_default(call_default_factory=True)


def _click_supplies_nothing(param) -> bool:
    """Whether click decides no value of its own, leaving the model's to stand.

    Three spellings mean this, and all three are fine: ``required=True``, an explicit
    ``default=None``, and click's own ``UNSET`` — what a ``multiple=True`` option carries
    before it resolves to an empty tuple at parse time.
    """
    return (
        bool(getattr(param, "required", False))
        or param.default is None
        or param.default is UNSET
    )


def _same_default(click_value, model_value) -> bool:
    """Whether the two describe the same default, allowing for how each side spells it.

    ``multiple=True`` hands back a tuple where the model declares a list; that is click's
    container, not a different setting. Everything else compares by value.
    """
    if isinstance(click_value, (list, tuple)) and isinstance(
        model_value, (list, tuple)
    ):
        return list(click_value) == list(model_value)
    return click_value == model_value


def _offences(path, command, model):
    found = []
    for param in command.params:
        field = model.model_fields.get(param.name)
        if field is None or (path, param.name) in EXEMPT:
            continue
        declares, value = _model_default(field)
        if _click_supplies_nothing(param):
            continue  # nothing is being decided behind the model's back
        if isinstance(value, BaseModel):
            # The option is a different *representation* of the field — `--windows whole` is
            # a spec string the command parses into a WindowSpec — so there is no value here
            # to compare. That pairing has its own guard, keyed to the nested model's fields:
            # tests/test_cli.py::test_the_window_options_come_from_the_model.
            continue
        if not declares:
            found.append(
                f"{path} --{param.name}: click defaults to {param.default!r} while "
                f"{model.__name__} declares the field required — the surface is deciding "
                f"what the model does not"
            )
        elif not _same_default(param.default, value):
            found.append(
                f"{path} --{param.name}: click defaults to {param.default!r}, "
                f"{model.__name__} to {value!r}"
            )
    return found


def test_the_sweep_pairs_real_commands():
    """Guard the guard: a pairing that matched nothing would pass vacuously forever."""
    assert len(PAIRS) >= 10, f"only {len(PAIRS)} model/command pairs found"
    assert any(path == "tracking run" for path, _cmd, _model in PAIRS)
    assert any(path == "process embedding" for path, _cmd, _model in PAIRS)


@pytest.mark.parametrize("path,command,model", PAIRS, ids=[p[0] for p in PAIRS])
def test_a_cli_option_never_invents_a_default(path, command, model):
    offences = _offences(path, command, model)
    assert not offences, "\n".join(offences)


def test_the_scan_would_catch_a_hardcoded_default():
    """The negative case: feed it the bug and watch it bite.

    Without this the sweep passes in two different worlds — every surface reads its model, or
    the comparison stopped comparing — and they are one green test from the outside.
    """
    path, command, model = next(p for p in PAIRS if p[0] == "tracking run")
    field = model.model_fields["tracker_id"]
    _declares, real = _model_default(field)

    # Both decls, so the option is *named* `tracker_id` — a bare `--tracker` is named
    # `tracker`, matches no model field, and would be skipped rather than judged.
    planted = click.Option(["--tracker", "tracker_id"], default=f"not-{real}")
    probe = click.Command(command.name, params=[planted])
    assert planted.name == "tracker_id"
    assert _offences(path, probe, model), "a disagreeing default was not caught"

    honest = click.Option(["--tracker", "tracker_id"], default=real)
    assert not _offences(path, click.Command(command.name, params=[honest]), model)


def test_every_exemption_still_excuses_something():
    """An entry that no longer describes a real disagreement is a muted guard, not a note."""
    stale = []
    by_path = {path: (command, model) for path, command, model in PAIRS}
    for (path, name), reason in EXEMPT.items():
        assert reason.strip(), f"{path} --{name} has no reason"
        pair = by_path.get(path)
        assert pair is not None, f"{path} names no command this sweep pairs"
        command, model = pair
        param = next((p for p in command.params if p.name == name), None)
        assert param is not None, f"{path} has no --{name} option"
        assert name in model.model_fields, f"{model.__name__} has no field {name!r}"
        if not _offences(path, click.Command(command.name, params=[param]), model):
            stale.append(f"{path} --{name}")
    assert not stale, f"exemptions that no longer excuse anything: {stale}"
