"""Which stages a user can send to a cluster, and that every surface agrees (§6).

**The class: a capability wired at some surfaces and silent at the rest.** `--backend remote`
reaches a stage through its pipeline module's `validate`, the CLI's two commands for it
(`<stage> run` and `process <stage>`), and the GUI form's compute row. Detection and embedding were
wired at all of them; keypoints, the next stage to want it, was wired at none, and a user found
that by reading the flag list. Nothing could fail: a command without `--remote-target` is a
well-formed command. The stage table below is the one place that says which stages go remote; each
test reads it against one surface, so a stage wired at the CLI and not the GUI (or the reverse)
fails here, and so does a stage that quietly gains the flag without being in the table.

Every processing module is in exactly one of the two tables, so a *new* stage has to decide.
"""

import pytest

pytest.importorskip("PyQt6")

from detail.api._runner_modules import PIPELINE_RUNNERS  # noqa: E402
from detail.cli import main as cli  # noqa: E402
from detail.executor.runners import (  # noqa: E402
    PROCESSING_MODULES,
    import_runner_modules,
)
from detail.gui import modules as _gui_modules  # noqa: E402,F401 (registers UIs)
from detail.gui.modules.registry import GUI_MODULES  # noqa: E402

#: stage -> its CLI commands as command paths (the keypoint stages have no ``process`` twin;
#: they are launched from their own ``keypoints`` group, §26).
REMOTE_STAGES = {
    "detection": (("detection", "run"), ("process", "detection")),
    "embedding": (("embedding", "run"), ("process", "embedding")),
    "keypoint_detect": (("keypoints", "detect"),),
    # CPU work with no model; a cluster runs a batch's recordings side by side.
    "keypoint_track": (("keypoints", "track"), ("process", "keypoint_track")),
}

#: stage -> why it takes no compute target. A stage here that gains one fails the CLI test below.
LOCAL_STAGES = {
    "tracking": "a CPU-only algorithm over detections already computed; no backend to redirect",
    "track_merge": "joins existing tracking runs on this machine's CPU",
}


def _command(path):
    command = cli
    for name in path:
        command = command.commands[name]
    return command


def _option(command, name):
    return next((p for p in command.params if p.name == name), None)


def test_every_processing_stage_decides_whether_it_goes_to_a_cluster():
    import_runner_modules(PIPELINE_RUNNERS)
    decided = set(REMOTE_STAGES) | set(LOCAL_STAGES)
    assert set(PROCESSING_MODULES) == decided, (
        "a processing stage is in neither REMOTE_STAGES nor LOCAL_STAGES: "
        f"{sorted(set(PROCESSING_MODULES) ^ decided)}"
    )
    assert not set(REMOTE_STAGES) & set(LOCAL_STAGES)


@pytest.mark.parametrize("stage", sorted(REMOTE_STAGES))
def test_a_remote_stage_offers_remote_on_each_of_its_cli_commands(stage):
    for path in REMOTE_STAGES[stage]:
        command = _command(path)
        backend = _option(command, "backend")
        assert (
            backend is not None and "remote" in backend.type.choices
        ), f"`detail {' '.join(path)}` does not offer --backend remote"
        assert (
            _option(command, "remote_target") is not None
        ), f"`detail {' '.join(path)}` has no --remote-target"


@pytest.mark.parametrize("stage", sorted(LOCAL_STAGES))
def test_a_local_stage_offers_no_remote_on_its_cli_commands(stage):
    for path in (
        ("process", stage.replace("_", "-")),
        (stage.replace("_", "-"), "run"),
    ):
        try:
            command = _command(path)
        except KeyError:
            continue
        backend = _option(command, "backend")
        assert backend is None or "remote" not in backend.type.choices, (
            f"`detail {' '.join(path)}` offers remote but {stage!r} is listed local-only: "
            f"{LOCAL_STAGES[stage]}"
        )
        assert _option(command, "remote_target") is None


@pytest.mark.parametrize("stage", sorted(REMOTE_STAGES))
def test_a_remote_stage_form_opts_into_the_remote_compute_row(qapp, stage):
    form = GUI_MODULES[stage].config_widget()
    row = form._compute
    assert row.remote_target is not None, (
        f"the {stage} launch form's compute row was built without allow_remote, so a "
        "registered cluster is never offered there"
    )
