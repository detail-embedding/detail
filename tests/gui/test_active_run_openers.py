"""Every kind the Active-runs listing can show, the shell can open — and can cancel on quit.

**The class.** *A family of run is added to one enumeration and not its siblings.* `api.active`
now declares the families once (`RUN_KINDS`), but two things about a family live in the GUI and
cannot be derived from that table: which **page** shows it, and — through the table's `cancel`
name — that quitting mid-run offers to stop it. Both were hand-written lists, and both had
drifted:

- the double-click router (`gui/app.py::_open_active_run`) knew `run` and `analysis_run`. The
  other four kinds drew the pointing hand and the trailing `›` that `navigable=True` promises,
  and clicking one did nothing at all. Every missing opener already existed as a method; only
  the mapping was absent.
- the quit guard (`_active_runs`) knew four families, missing `clusterer` and `distill`. Closing
  the window during a distill run — hours of GPU training — accepted the close silently.

Neither could fail a behavioural test: a row nobody can open looks exactly like a row nobody
clicked, and a quit prompt that does not appear looks exactly like having nothing to say. So the
check is parity against the table, data-driven, so a seventh kind is covered the day it is added.
"""

import pytest

pytest.importorskip("PyQt6")

from detail.api import active  # noqa: E402
from detail.api.experiment import Experiment  # noqa: E402
from detail.gui.app import MainWindow  # noqa: E402

pytestmark = pytest.mark.guard


def _window(qapp, tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    return MainWindow()


def test_every_listed_kind_has_a_page_to_open(qapp, tmp_path, monkeypatch):
    window = _window(qapp, tmp_path, monkeypatch)
    openers = window._active_run_openers()
    missing = sorted({spec.kind for spec in active.RUN_KINDS} - set(openers))
    assert not missing, (
        f"these run kinds are listed on Active runs and open nothing when double-clicked: "
        f"{missing}. The row still draws the pointing hand and the `›`, so it reads as "
        f"clickable and is not."
    )


def test_no_opener_names_a_kind_that_is_never_listed(qapp, tmp_path, monkeypatch):
    """The other direction: an opener for a kind no listing produces is dead code."""
    window = _window(qapp, tmp_path, monkeypatch)
    extra = sorted(
        set(window._active_run_openers()) - {s.kind for s in active.RUN_KINDS}
    )
    assert not extra, f"openers for kinds that are never listed: {extra}"


@pytest.mark.parametrize("spec", active.RUN_KINDS, ids=lambda s: s.kind)
def test_every_listed_kind_is_offered_for_cancellation_on_quit(
    qapp, tmp_path, monkeypatch, spec
):
    """A run in flight is named in the quit prompt and reachable by `_cancel_all_active`.

    Driven through the **real** `_active_runs` → `_cancel_all_active` pair rather than the
    table, because the bug was that those two read a different list from the one that decides
    what is shown as running.
    """
    window = _window(qapp, tmp_path, monkeypatch)
    experiment = Experiment.create("p", tmp_path / "exp")

    class _Ref:
        id, status = "x_01", "running"

    monkeypatch.setattr(
        Experiment, spec.records, property(lambda self: [_Ref()]), raising=False
    )
    window._experiment = experiment

    active_ids = window._active_runs()
    assert active_ids.get(spec.kind) == ["x_01"], (
        f"a running {spec.kind} is not counted by the quit guard, so closing the window "
        f"during one neither warns nor offers to cancel it"
    )

    cancelled = []
    monkeypatch.setattr(
        experiment, spec.cancel, lambda run_id: cancelled.append(run_id), raising=False
    )
    window._cancel_all_active(active_ids)
    assert cancelled == [
        "x_01"
    ], f"'Cancel runs and quit' never reached the {spec.kind}"


@pytest.mark.parametrize("spec", active.RUN_KINDS, ids=lambda s: s.kind)
def test_double_clicking_a_row_reaches_that_kinds_page(
    qapp, tmp_path, monkeypatch, spec
):
    """Through `_open_active_run`, the method the table's `doubleClicked` is wired to.

    The dict having the key is not the claim — the claim is that a row a user double-clicks
    gets somewhere. Asserted by spying on the opener rather than on the resulting page, so this
    stays about the routing and does not depend on each detail page's fixtures.
    """
    window = _window(qapp, tmp_path, monkeypatch)
    experiment = Experiment.create("p", tmp_path / "exp")

    opened = []
    monkeypatch.setattr(
        MainWindow,
        "_active_run_openers",
        lambda self: {
            s.kind: (lambda rid, k=s.kind: opened.append((k, rid)))
            for s in active.RUN_KINDS
        },
    )
    monkeypatch.setattr(MainWindow, "_open_experiment", lambda self, e: None)

    row = active.ActiveRun(
        kind=spec.kind,
        experiment_dir=experiment.dir,
        experiment_name="p",
        run_id="x_01",
        label=spec.label,
        status="running",
        created_at="2026-09-11T12:00:00Z",
    )
    window._open_active_run(row)
    assert opened == [
        (spec.kind, "x_01")
    ], f"double-clicking a {spec.kind} row went nowhere"


def test_the_sweep_would_catch_a_kind_with_no_opener(qapp, tmp_path, monkeypatch):
    """Guard the guard: drop a kind from the router and the parity check must fail."""
    window = _window(qapp, tmp_path, monkeypatch)
    full = window._active_run_openers()
    monkeypatch.setattr(
        MainWindow,
        "_active_run_openers",
        lambda self: {k: v for k, v in full.items() if k != "distill"},
    )
    with pytest.raises(AssertionError, match="distill"):
        test_every_listed_kind_has_a_page_to_open(qapp, tmp_path, monkeypatch)
