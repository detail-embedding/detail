"""Taking clips out of a label on a temporal clustering run (§22, ``api/clip_edits.py``).

A clip a user has watched and knows is wrong is taken out of its label, or moved to another, and
the export stops carrying it. The fit is never touched. Run over both temporal topologies: the
codebook decode, whose members are frames and whose instances are runs of them (a clip taken out of
the middle splits an instance), and the bout method, whose clip is a whole bout.
"""

import json

import pytest

from detail.api.authored_run import ReadOnlyRunError
from detail.api.clip_edits import ClipRef
from detail.api.clip_edits import clip_ref
from detail.storage import behavior_output as bo
from detail.storage import temporal_runs


def _labelled(experiment, run_id):
    """The run with its root's two children labelled climb and rear."""
    run = experiment.temporal_clustering_run(run_id)
    view = run.node_view()
    climb, rear = view.children(view.root)
    run.name_nodes({climb: "climb", rear: "rear"})
    return run, climb, rear


def _spans(run, name: str) -> list[tuple[int, int]]:
    return sorted(
        (row.t_ns, row.end_t_ns)
        for row in run.artifact().instances
        if row.label == name
    )


def _middle_clip(run, node: int) -> tuple[dict, tuple[int, int]]:
    """A clip of ``node`` strictly inside one of its label's instances, with that instance."""
    clips = run.node_view().clips(node).rows()
    for instance in _spans(run, "climb"):
        inside = [
            c
            for c in clips
            if instance[0] < c["start_t_ns"] and c["end_t_ns"] < instance[1]
        ]
        if inside:
            return inside[len(inside) // 2], instance
    raise AssertionError("no clip lies strictly inside a climb instance")


def _history(run) -> list[dict]:
    path = run._run_dir() / temporal_runs.HISTORY_FILENAME
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return [{k: v for k, v in e.items() if k != "at"} for e in entries]


def test_taking_out_a_middle_clip_splits_its_instance(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clip, instance = _middle_clip(run, climb)

    run.take_out_clips([clip], "climb")

    spans = _spans(run, "climb")
    assert instance not in spans
    left = [s for s in spans if s[0] == instance[0]]
    right = [s for s in spans if s[1] == instance[1]]
    assert left and left[0][1] < clip["start_t_ns"], spans
    assert right and right[0][0] > clip["end_t_ns"], spans
    assert not any(
        s[0] <= clip["end_t_ns"] and clip["start_t_ns"] <= s[1] for s in spans
    ), "an instance still covers the taken-out clip"
    assert _history(run)[-1] == {
        "op": "take clips out",
        "clips": 1,
        "from_label": run.state().node_labels[str(climb)],
        "to_label": None,
    }
    [taken] = run.taken_out("climb")
    assert clip_ref(taken) == clip_ref(clip) and taken["from_name"] == "climb"
    assert taken["to_name"] is None


def test_a_moved_clip_shows_up_under_the_target_label(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clip, _instance = _middle_clip(run, climb)

    run.take_out_clips([clip], "climb", "rear")

    assert (clip["start_t_ns"], clip["end_t_ns"]) in _spans(run, "rear")
    assert run.taken_out()[0]["to_name"] == "rear"


def test_putting_back_restores_a_byte_identical_instances_csv(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    path = run._run_dir() / bo.INSTANCES_FILENAME
    before = path.read_bytes()
    clip, _instance = _middle_clip(run, climb)

    run.take_out_clips([clip], "climb")
    assert path.read_bytes() != before
    run.put_back_clips(
        [ClipRef(clip["recording_id"], 0, clip["start_t_ns"], clip["end_t_ns"])]
    )

    assert path.read_bytes() == before
    assert run.taken_out() == []
    assert _history(run)[-1] == {"op": "put clips back", "clips": 1}


def test_putting_back_a_clip_never_taken_out_writes_nothing(fitted_codebook):
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    clip, _instance = _middle_clip(run, climb)
    lines = len(_history(run))
    run.put_back_clips([clip])
    assert len(_history(run)) == lines


def test_a_bout_taken_out_reports_unlabelled_whole(fitted_temporal):
    experiment, run_id = fitted_temporal
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    span = (clip["start_t_ns"], clip["end_t_ns"])
    assert span in _spans(run, "climb")

    run.take_out_clips(clip, "climb")

    assert span not in _spans(run, "climb")
    assert span in [
        (r.t_ns, r.end_t_ns) for r in run.artifact().instances if not r.label
    ]


def test_an_edit_whose_label_changed_no_longer_applies(fitted_temporal):
    """Taken out of "climb", then its cluster relabelled "walk": the clip is walk again — the
    edit said "not climb", and says nothing about walk."""
    experiment, run_id = fitted_temporal
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    run.take_out_clips(clip, "climb")
    run.label_nodes({climb: run.create_label("walk")})
    assert (clip["start_t_ns"], clip["end_t_ns"]) in _spans(run, "walk")


@pytest.mark.parametrize("call", ["take_out_clips", "put_back_clips"])
def test_a_snapshot_refuses_clip_edits_naming_why(fitted_temporal, call):
    experiment, run_id = fitted_temporal
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    view = run.view(run.pin(reason="finish"))
    args = ([clip], "climb") if call == "take_out_clips" else ([clip],)
    with pytest.raises(ReadOnlyRunError, match="snap.*will not (remove|put) clips"):
        getattr(view, call)(*args)


def test_a_finalized_run_refuses_a_take_out_naming_the_way_back(fitted_temporal):
    experiment, run_id = fitted_temporal
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    run.finalize()
    with pytest.raises(ValueError, match="finalized.*reopen"):
        run.take_out_clips(clip, "climb")
    assert run.taken_out() == []


def test_a_bad_take_out_is_refused_before_anything_is_written(fitted_temporal):
    experiment, run_id = fitted_temporal
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    with pytest.raises(ValueError, match="'nope' is not a label"):
        run.take_out_clips(clip, "nope")
    with pytest.raises(ValueError, match="the label the clips are removed from"):
        run.take_out_clips(clip, "climb", "climb")
    with pytest.raises(ValueError, match="at least one clip"):
        run.take_out_clips([], "climb")
    assert run.taken_out() == []


def test_a_clip_that_is_not_in_the_label_is_refused(fitted_temporal):
    """A clip of another label, a made-up span and a recording the run never saw were each
    stored and listed by `taken_out()`, changing nothing (#31, finding 47)."""
    experiment, run_id = fitted_temporal
    run, climb, rear = _labelled(experiment, run_id)
    of_rear = run.node_view().clips(rear).rows()[0]
    real = run.node_view().clips(climb).rows()[0]
    made_up = ClipRef(real["recording_id"], 0, 1, 2)
    elsewhere = ClipRef("not_a_recording", 0, real["start_t_ns"], real["end_t_ns"])
    for stray in (of_rear, made_up, elsewhere):
        with pytest.raises(ValueError, match="is not in 'climb'"):
            run.take_out_clips([real, stray], "climb")
    assert run.taken_out() == []

    run.take_out_clips(of_rear, "rear", "climb")
    run.take_out_clips(of_rear, "climb")  # moved in, so it is in climb now
    assert [clip_ref(e) for e in run.taken_out()] == [clip_ref(of_rear)]


def test_the_snapshot_carries_the_edits_it_was_pinned_with(fitted_temporal):
    experiment, run_id = fitted_temporal
    run, climb, _rear = _labelled(experiment, run_id)
    clip = run.node_view().clips(climb).rows()[0]
    run.take_out_clips(clip, "climb")
    view = run.view(run.pin(reason="finish"))
    run.reopen()
    run.put_back_clips(clip)
    assert [clip_ref(e) for e in view.taken_out()] == [clip_ref(clip)]
    assert run.taken_out() == []


def test_the_label_bar_and_browse_preview_agree_with_the_export(fitted_codebook):
    """``label_time`` and ``labelled_runs`` read the cache, not ``instances.csv``; they must
    follow a take-out the way the export does, or the label bar and BROWSE disagree with it.
    """
    experiment, run_id = fitted_codebook
    run, climb, _rear = _labelled(experiment, run_id)
    names = run.node_names()
    clip, _instance = _middle_clip(run, climb)
    before = run.node_view().label_time(names)

    run.take_out_clips([clip], "climb")

    view = run.node_view()
    after = view.label_time(names)
    moved = before[str(climb)] - after[str(climb)]
    assert moved > 0
    assert after["unlabelled"] - before.get("unlabelled", 0) == moved
    covering = [
        r
        for r in view.labelled_runs(names)
        if r["node"] == climb
        and r["start_ns"] <= clip["end_t_ns"]
        and clip["start_t_ns"] < r["end_ns"]
    ]
    assert covering == [], "BROWSE's preview still shows the clip under climb"
    raw = view.labelled_runs(names, apply_edits=False)
    assert any(
        r["start_ns"] <= clip["start_t_ns"] < r["end_ns"] and r["node"] == climb
        for r in raw
    ), "apply_edits=False should show the clusters as fitted"
    export = _spans(run, "climb")
    assert not any(
        s[0] <= clip["end_t_ns"] and clip["start_t_ns"] <= s[1] for s in export
    )


def _side(change, index):
    return tuple({key: pair[index] for key, pair in part.items()} for part in change)


def _unlabelled(run) -> list[tuple[int, int]]:
    return sorted((r.t_ns, r.end_t_ns) for r in run.artifact().instances if not r.label)


def test_a_merge_carries_the_clip_edits_and_its_undo_puts_them_back(fitted_temporal):
    """**A label id removed while a clip edit still points at it.** A merge deletes the source
    label's id; an edit naming it as ``from_label`` then stopped applying (the clip taken out
    counted again) and one naming it as ``to_label`` sent the clip to "unlabelled". A merge says
    the two labels are one behavior, so the edits now name the target; an edit whose two ends
    become the same label is a move within one label, and goes. ⌘Z of the merge
    (``restore_labels`` with the *before* side of ``label_change``) puts the edits back too.
    """
    from detail.api import authored_run as authored

    experiment, run_id = fitted_temporal
    run, climb, rear = _labelled(experiment, run_id)
    taken = run.node_view().clips(rear).rows()[0]  # taken out of rear
    moved = run.node_view().clips(climb).rows()[0]  # moved from climb into rear
    run.take_out_clips(taken, "rear")
    run.take_out_clips(moved, "climb", "rear")
    span_taken = (taken["start_t_ns"], taken["end_t_ns"])
    span_moved = (moved["start_t_ns"], moved["end_t_ns"])
    before = run.label_maps()
    edits_before = run.taken_out()

    from detail.storage import run_labels

    labels = run.state().labels
    run.merge_labels(run_labels.find(labels, "rear"), run_labels.find(labels, "climb"))

    assert span_taken in _unlabelled(run), "the clip taken out of rear came back"
    assert span_moved in _spans(run, "climb"), "the clip moved into rear lost its label"
    [edit] = run.taken_out()
    assert clip_ref(edit) == clip_ref(taken)
    assert (edit["from_name"], edit["to_name"]) == ("climb", None)

    change = authored.label_change(before, run.label_maps())
    run.restore_labels(*_side(change, 0))  # ⌘Z
    assert run.label_maps() == before
    assert run.taken_out() == edits_before
    assert span_taken in _unlabelled(run)
    assert span_moved in _spans(run, "rear")
