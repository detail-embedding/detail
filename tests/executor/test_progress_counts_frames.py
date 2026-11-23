"""Every module says whether its progress total is a frame count.

**The class.** A report read a progress bar's ``frames_total`` as the number of frames a run
processed and divided it by seconds to print fps. But the bar's pair is whatever makes the bar
move: most analyses count a fraction out of 1000 (``core.progress.split_units``), location export
counts table rows, and keypoint tracking counted both of its passes, so it reported twice its
frames. The performance report printed all of these as frame rates. The pipeline-throughput
benchmark caught one: keypoint tracking at 1108 fps on 192,816 "frames", where the run itself
recorded 96,408.

**How it hid.** Nothing is wrong where it is written. The bar moves correctly, the report's
arithmetic is right, and the rate it prints is a plausible number.

**The rule.** ``progress_counts_frames`` is declared by every processing and analysis module, and
has no default, so a new module has to answer. The performance report shows frames and fps only
for a module that answers yes (``api/export_targets.py::_progress_counts_frames``); anything else
gets a blank. A module that answers yes must not build its progress from the per-mille fraction.
That is checked against its source here. The doubled shape has its own behavioural test in
``tests/pipeline/test_keypoint_track.py``.
"""

from __future__ import annotations

import inspect
import re

import pytest

from detail.api._runner_modules import PIPELINE_RUNNERS
from detail.executor.runners import ANALYSIS_MODULES
from detail.executor.runners import PROCESSING_MODULES
from detail.executor.runners import AnalysisModule
from detail.executor.runners import ProcessingModule
from detail.executor.runners import import_runner_modules

import_runner_modules(PIPELINE_RUNNERS)

_MODULES = {**PROCESSING_MODULES, **ANALYSIS_MODULES}

#: The shapes of a progress total that is a fraction, not a count of frames.
_FRACTION = re.compile(r"split_units\(|frames_total=1000\b|WITHIN_RESOLUTION")


def _declared(cls: type) -> object:
    """The value a module class declares itself, never one it inherits from the base."""
    for klass in cls.__mro__:
        if klass in (ProcessingModule, AnalysisModule):
            break
        if "progress_counts_frames" in vars(klass):
            return vars(klass)["progress_counts_frames"]
    return None


def _counts_a_fraction(source: str) -> bool:
    return bool(_FRACTION.search(source))


def test_the_registries_are_populated():
    assert len(PROCESSING_MODULES) >= 5 and len(ANALYSIS_MODULES) >= 3


@pytest.mark.parametrize("module_id", sorted(_MODULES))
def test_every_module_declares_what_its_progress_counts(module_id):
    value = _declared(type(_MODULES[module_id]))
    assert isinstance(value, bool), (
        f"{module_id} does not declare progress_counts_frames: say whether its progress "
        "frames_total is a count of frames processed (the performance report's fps uses it)"
    )


@pytest.mark.parametrize(
    "module_id",
    sorted(m for m in _MODULES if _declared(type(_MODULES[m])) is True),
)
def test_a_module_counting_frames_does_not_report_a_fraction(module_id):
    source = inspect.getsource(inspect.getmodule(type(_MODULES[module_id])))
    assert not _counts_a_fraction(source), (
        f"{module_id} declares progress_counts_frames = True but its source builds a "
        "per-mille progress fraction; the performance report would print it as fps"
    )


def test_the_checks_reject_their_counterexamples():
    """Each check fails on the thing it exists to catch."""

    class Forgot(ProcessingModule):
        module_id = "forgot"

    class Inherits(Forgot):
        pass

    class Says(AnalysisModule):
        module_id = "says"
        progress_counts_frames = False

    assert _declared(Forgot) is None
    assert _declared(Inherits) is None
    assert _declared(Says) is False
    assert _counts_a_fraction("outer, inner, total = split_units(done, n)")
    assert _counts_a_fraction("Event(frames_done=k, frames_total=1000)")
    assert not _counts_a_fraction("Event(frames_done=k, frames_total=len(frames))")
