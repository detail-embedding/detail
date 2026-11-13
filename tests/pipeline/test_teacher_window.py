"""Keypoint teacher runs (§26, #717): a windowed run is consumed by the teacher chain only.

The class this guards: a run that covers only a recording's first minutes, read by a stage
that assumes it covers all of it. Nothing fails — an embedding or a location table over five
minutes of a two-hour recording is well formed and plausible — so the guard is structural:
every registered processing or analysis module that reads a detection, tracking or keypoints
run is either a teacher consumer (``teacher_window.TEACHER_CONSUMERS``) or calls
``refuse_teacher_run`` itself. A module added later that reads one of those runs fails here
until it decides which it is.

**Accepting is two obligations, not one.** A teacher consumer records the window it inherits,
or its own run launders the teacher run: it reads as covering the whole recording, and every
refusal after it passes. That is why ``track_merge`` could not simply stop refusing when the
two-animal keypoint setup (whose own chain is windowed detection → a track per animal → a
merge) was found to fail at its merge stage with that refusal: it accepts sources that share
one window and records it (``tests/pipeline/test_track_merge.py``), and
``test_every_teacher_consumer_records_the_window_it_inherits`` holds every consumer to it. The
setup's side is ``tests/api/test_keypoint_setup.py``: every stage it launches after detection
must be a teacher consumer, which its stand-ins had never asked.
"""

import inspect

import pytest

from detail.api.experiment import _PIPELINE_RUNNERS
from detail.executor import runners
from detail.pipeline import teacher_window

_RUN_PRODUCERS = {"detection", "tracking", "keypoint_detect", "keypoint_track"}


def _reads_a_run_producer(module) -> bool:
    upstream = getattr(module, "upstream", None)
    selects = getattr(module, "selects", None) or (None,)
    aux = [ref[0] for ref in (getattr(module, "aux_refs", None) or ())]
    return bool({upstream, selects[0], *aux} & _RUN_PRODUCERS)


def _modules():
    runners.import_runner_modules(_PIPELINE_RUNNERS)
    return {**runners.PROCESSING_MODULES, **runners.ANALYSIS_MODULES}


def _class_of(module) -> type:
    return module if inspect.isclass(module) else type(module)


def _undecided(modules: dict) -> list[str]:
    """Modules that read a detection / tracking / keypoints run and neither consume a teacher
    run nor refuse one."""
    return sorted(
        module_id
        for module_id, module in modules.items()
        if _reads_a_run_producer(module)
        and module_id not in teacher_window.TEACHER_CONSUMERS
        and "refuse_teacher_run(" not in inspect.getsource(_class_of(module))
    )


def test_every_consumer_of_a_run_decides_about_teacher_runs():
    undecided = _undecided(_modules())
    assert not undecided, (
        "these modules read a detection, tracking or keypoints run but neither accept a "
        "keypoint teacher run (TEACHER_CONSUMERS) nor refuse one (refuse_teacher_run): "
        f"{undecided}"
    )


def test_the_guard_reaches_the_consumers_it_exists_for():
    reads = {m for m, mod in _modules().items() if _reads_a_run_producer(mod)}
    assert {"embedding", "track_merge", "location_export"} <= reads


def _forgets_the_window(modules: dict) -> list[str]:
    """Teacher consumers whose run record does not carry the window forward. Such a module
    launders a teacher run: its own run reads as covering the whole recording, and every
    refusal downstream of it passes."""
    return sorted(
        module_id
        for module_id in teacher_window.TEACHER_CONSUMERS
        if "teacher_window.WINDOW_KEY"
        not in inspect.getsource(_class_of(modules[module_id]).extra_record_fields)
    )


def test_every_teacher_consumer_records_the_window_it_inherits():
    assert not _forgets_the_window(_modules()), (
        "these modules take a keypoint teacher run but do not record its window, so their "
        f"own runs read as whole-recording runs: {_forgets_the_window(_modules())}"
    )


def test_the_window_guard_would_catch_a_consumer_that_drops_it():
    class _Drops:
        def extra_record_fields(self, config):
            return {"every_n_frames": 1}

    modules = {m: _Drops for m in teacher_window.TEACHER_CONSUMERS}
    assert _forgets_the_window(modules) == sorted(teacher_window.TEACHER_CONSUMERS)


def test_the_guard_would_catch_a_consumer_that_never_decided():
    class _NewConsumer:
        upstream = "tracking"

    class _Unrelated:
        upstream = "embedding"

    assert _undecided({"new_consumer": _NewConsumer, "unrelated": _Unrelated}) == [
        "new_consumer"
    ]


def test_a_whole_recording_run_is_never_refused():
    teacher_window.refuse_teacher_run(
        {"id": "trk_001"}, consumer="embedding", upstream="tracking run 'trk_001'"
    )
    teacher_window.refuse_teacher_run(None, consumer="embedding", upstream="nothing")


def test_a_teacher_run_is_refused_with_its_window_and_the_way_out():
    with pytest.raises(ValueError) as caught:
        teacher_window.refuse_teacher_run(
            {"id": "trk_001", "teacher_window_s": 300.0},
            consumer="embedding",
            upstream="tracking run 'trk_001'",
        )
    message = str(caught.value)
    assert "tracking run 'trk_001' is a keypoint teacher run (first 5 min" in message
    assert "embedding needs a run over the whole recording" in message
    assert "without a teacher window" in message


@pytest.mark.parametrize(
    "window_s, label",
    [
        (300.0, "first 5 min"),
        (90.0, "first 90 s"),
        (60.0, "first 1 min"),
        (2.5, "first 2.5 s"),
    ],
)
def test_window_label(window_s, label):
    assert teacher_window.window_label(window_s) == label
