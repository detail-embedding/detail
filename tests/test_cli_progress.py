"""The CLI's progress bars must not invent a number.

Split out of test_cli.py so it appends at the end of its own file: several branches add
CLI tests at once, and two appends to one file conflict for no reason.
"""

import pytest

import detail.cli as cli_mod


# --- progress truth in the CLI bars ----------------------------------------- #
class _FakeBar:
    """Records what the driver does to its bar. tqdm is a rendering library, not storage —
    stubbing it here is how we observe the driver's decisions."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.total = kwargs.get("total")
        self.n = 0
        self.resets = []
        self.postfix = ""
        _FakeBar.instances.append(self)

    def reset(self, total=None):
        self.resets.append(total)
        self.total = total
        self.n = 0

    def write(self, *a, **k):
        pass

    def set_postfix_str(self, s, refresh=False):
        self.postfix = s

    def refresh(self):
        pass

    def close(self):
        pass


class _FakeRun:
    id = "run_001"
    status = "complete"
    error = None

    def __init__(self, events):
        self._events = events

    def progress(self):
        return iter(self._events)

    def wait(self):
        return self

    def cancel(self):
        pass


@pytest.fixture
def fake_bars(monkeypatch):
    _FakeBar.instances = []
    import detail.cli as cli_mod

    monkeypatch.setattr(cli_mod.tqdm, "tqdm", _FakeBar)
    return _FakeBar.instances


def test_the_bar_resets_when_the_unit_changes(fake_bars):
    """A multi-recording analysis run restarts its frame counter per recording. Only moving
    `total` left tqdm dividing recording 2's count by the whole run's clock, which printed
    remaining times like 266:07:21."""
    from detail.cli import _stream_progress
    from detail.core.dataclasses import LabelingProgressEvent

    events = [
        LabelingProgressEvent(50, 100, 0, 1, 5.0, None, "recording 1"),
        LabelingProgressEvent(10, 40, 0, 1, 9.0, None, "recording 2"),  # a new unit
    ]
    _stream_progress(_FakeRun(events), "analysis")

    assert fake_bars[0].resets == [40], "the bar must restart its clock on a new unit"


def test_the_distill_bar_does_not_print_its_own_remaining(fake_bars):
    """`n` is the stage-weighted fraction, not a count of work, so tqdm's rate estimate over
    it re-derives the ~5x under-report PR #171 removed — printed next to the correct eta=.
    """
    from detail.cli import _stream_distill_progress
    from detail.core.dataclasses import DistillProgress

    events = [
        DistillProgress(
            phase="training",
            stage_index=1,
            stage_count=2,
            inner_fraction=0.14,
            overall_fraction=0.527,
            elapsed_s=1238.0,
            eta_s=5332.0,
            message="epoch 14/100",
        )
    ]
    _stream_distill_progress(_FakeRun(events), "distill")

    fmt = fake_bars[0].kwargs.get("bar_format") or ""
    assert "{remaining}" not in fmt and "remaining" not in fmt
    assert "eta=5332s" in fake_bars[0].postfix  # the runner's number is still shown
