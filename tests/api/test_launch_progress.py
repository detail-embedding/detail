"""A launch's up-front checks report which recording they are on (§6, §15 *Validation*).

**The class: a long synchronous op reported as one static line.** ``run_processing`` validates
every selected recording before it spawns anything, and ``retry_run`` re-validates the incomplete
ones. Both were silent: the GUI showed *Validating N recording(s)…* and the CLI nothing at
all. An aligned embedding launch read every stored mask of every recording, so the line sat
unchanged for a long time and read as a hang. The pre-pass is fast now
(``tests/pipeline/test_embedding.py``), but validation is still
O(recordings) disk reads, and a selection is only getting larger.

**The rule.** Each per-recording loop at launch reports through the ``(stage, fraction, detail)``
callback before it reads each recording, by **name**, with its place in the selection, and names
the phase it is in: ``Resolving shared settings`` for a module's ``resolve_shared_config``, then
``Validating``. A module that resolves nothing shared reports no such phase.

These drive a fake module that refuses the last recording, so every check runs and nothing spawns.
"""

import pytest

from detail.api.experiment import Experiment
from detail.executor.runners import PROCESSING_MODULES, ProcessingModule
from detail.storage import runs as runs_store


class _Checked(ProcessingModule):
    """Refuses ``refuse``; records each check in ``events`` beside the progress reports."""

    module_id = "_checked"
    id_prefix = "chk"
    runs_key = "_checked_runs"
    upstream = "video"
    job_class = dict
    progress_event_class = None

    def __init__(self, events: list, refuse: str) -> None:
        self.events = events
        self.refuse = refuse

    def validate(self, recording_dir, recording_data, config):
        self.events.append(("validate", recording_dir.name))
        if recording_dir.name == self.refuse:
            raise ValueError(f"{recording_dir.name} is refused")
        return config

    def build_job(self, recording_id, run_id, config):
        raise AssertionError("a refused launch never builds a job")

    def runner(self, job, run_dir, progress_cb, stop_event):
        raise AssertionError("a refused launch never runs")

    def run_dir_for(self, experiment_root, job):
        return experiment_root

    def output_path(self, run_id):
        return f"_checked/{run_id}/"


class _SharedChecked(_Checked):
    def resolve_shared_config(
        self, experiment_dir, recording_ids, config, on_recording=None
    ):
        for sid in recording_ids:
            on_recording(sid)
            self.events.append(("share", sid))
        return config


def _experiment(tmp_path) -> tuple[Experiment, list[str]]:
    proj = Experiment.create("p", tmp_path)
    ids = [proj.add_recording(name).id for name in ("Rat 1", "Rat 2", "Rat 3")]
    return proj, ids


def _launch(proj, ids, events) -> None:
    """The one ``run_processing`` call site: ``tests/test_real_run_budget.py`` counts call
    sites, not spawns, and a refused launch spawns nothing."""
    with pytest.raises(ValueError, match="is refused"):
        proj.run_processing(
            "_checked", ids, {}, progress=lambda *report: events.append(report)
        )


def test_each_launch_phase_names_the_recording_before_reading_it(tmp_path, monkeypatch):
    proj, ids = _experiment(tmp_path)
    events: list = []
    monkeypatch.setitem(
        PROCESSING_MODULES, "_checked", _SharedChecked(events, refuse=ids[-1])
    )
    _launch(proj, ids, events)
    share, check = "Resolving shared settings", "Validating"
    assert events == [
        (share, 0.0, "Rat 1 (1/3)"),
        ("share", ids[0]),
        (share, 1 / 3, "Rat 2 (2/3)"),
        ("share", ids[1]),
        (share, 2 / 3, "Rat 3 (3/3)"),
        ("share", ids[2]),
        (share, 1.0, "3/3"),
        (check, 0.0, "Rat 1 (1/3)"),
        ("validate", ids[0]),
        (check, 1 / 3, "Rat 2 (2/3)"),
        ("validate", ids[1]),
        (check, 2 / 3, "Rat 3 (3/3)"),
        ("validate", ids[2]),
    ]


def test_a_module_with_nothing_shared_reports_only_validation(tmp_path, monkeypatch):
    """The base hook reads nothing, so naming a phase for it would be a line that claims
    work no one does."""
    proj, ids = _experiment(tmp_path)
    events: list = []
    monkeypatch.setitem(PROCESSING_MODULES, "_checked", _Checked(events, refuse=ids[0]))
    _launch(proj, ids, events)
    assert events == [("Validating", 0.0, "Rat 1 (1/3)"), ("validate", ids[0])]


def test_a_retry_names_each_recording_it_re_checks(tmp_path, monkeypatch):
    """A recording deleted since the run is still reported (by its id, having no name)
    and still lands in the refusal — reporting it must not be what raises."""
    proj, ids = _experiment(tmp_path)
    gone = "recording_099"
    events: list = []
    monkeypatch.setitem(PROCESSING_MODULES, "_checked", _Checked(events, refuse=ids[1]))
    runs_store.write_run(
        proj.dir,
        "run_001",
        {
            "id": "run_001",
            "module_id": "_checked",
            "recording_ids": [*ids, gone],
            "config": {},
            "per_recording": {
                ids[0]: {"run_id": "chk_01", "status": "complete"},
                ids[1]: {"run_id": None, "status": "failed"},
                ids[2]: {"run_id": None, "status": "failed"},
                gone: {"run_id": None, "status": "failed"},
            },
            "status": "failed",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    with pytest.raises(ValueError) as refused:
        proj.retry_run("run_001", progress=lambda *report: events.append(report))
    assert [e for e in events if len(e) == 3] == [
        ("Validating", 0.0, "Rat 2 (1/3)"),
        ("Validating", 1 / 3, "Rat 3 (2/3)"),
        ("Validating", 2 / 3, f"{gone} (3/3)"),
        ("Validating", 1.0, "3/3"),
    ]
    assert ids[1] in str(refused.value) and gone in str(refused.value)
