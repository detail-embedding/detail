"""An export never ignores what it was asked to narrow to, and never reports it as applied.

**The class.** *An option a person gave is dropped, and the run says it was done.* For export:
`--target instances --frames 0:3` wrote every instance and `export.yaml` recorded a frames
scope; `--target tracks --cluster 3` wrote every track; `--target clip` with no `--frames`
exited 0 with "nothing to write"; `--cluster zz` (a cluster the run does not have) and
`--frames 5000:6000` (past the end) wrote header-only files reported as written (#31,
findings 40-43).

**How it hid.** Every target takes the same ``ExportScope`` and reads only the parts it
understands. Nothing compared the scope with the target's own ``scopes`` declaration, so a
narrowing a target does not implement fell through to "everything", and the manifest's
``scope`` line — written from the request, not from what each target did — said the
narrowing had happened.

**The rule.** ``exporters.check_request`` runs on every way into an export
(`Experiment.run_export`, `Experiment.estimate_export`, `exporters.write_export`) before
anything is written. It refuses a scope kind a chosen target does not list, a scope kind a
target ``requires`` and was not given, a target asked for twice, a narrowing that no chosen
target can apply, and a cluster, selection or frame range the subject does not have. A
whole-subject report (a target with no ``scopes``, such as the performance report) is
written whole beside narrowed targets, and the manifest's ``whole`` list says so.

Data-driven over the catalogue through the pure ``exporters.scope_refusal``, so a target added
tomorrow is covered the day it is registered. The behaviour on real runs (cluster, frames,
selection, the manifest) is in ``tests/api/test_exporters.py`` and ``tests/test_cli_export.py``.
"""

import dataclasses

import pytest

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportScope

pytestmark = pytest.mark.guard

#: One scope per kind, each setting exactly that kind.
_ONE_KIND = {
    "cluster": ExportScope(cluster="c01"),
    "selection": ExportScope(instance_ids=("i1",)),
    "frames": ExportScope(frames=(0, 9)),
    "recordings": ExportScope(recording_ids=("r1",)),
}


def test_every_scope_kind_has_a_probe():
    """The sweep below covers every kind there is, so a new kind cannot skip it."""
    assert set(_ONE_KIND) == set(exporters.SCOPE_KINDS)
    for kind, scope in _ONE_KIND.items():
        assert scope.kinds() == (kind,)


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
def test_a_target_declares_only_real_scope_kinds(target):
    assert set(target.scopes) <= set(exporters.SCOPE_KINDS), target.scopes
    assert set(target.requires) <= set(target.scopes), (
        f"{target.id!r} requires {target.requires} but can only be narrowed to "
        f"{target.scopes}: a required scope it cannot take could never be satisfied"
    )


@pytest.mark.parametrize(
    "target", exporters.all_targets(), ids=lambda t: f"{t.subject_kind}.{t.id}"
)
@pytest.mark.parametrize("kind", exporters.SCOPE_KINDS)
def test_a_scope_kind_is_taken_or_refused_never_ignored(target, kind):
    """For a target that narrows at all, a kind it does not list is refused by name."""
    given = dict(_ONE_KIND)
    for needed in target.requires:
        given = {k: _merge(v, _ONE_KIND[needed]) for k, v in given.items()}
    reason = exporters.scope_refusal(target, given[kind])
    if not target.scopes or kind in target.scopes:
        assert reason is None, f"{target.id!r} takes {kind} but is refused: {reason}"
    else:
        assert reason is not None, (
            f"{target.id!r} does not list {kind!r} in its scopes {target.scopes}, so it "
            "would write everything while the manifest records the narrowing"
        )
        assert "cannot be narrowed to" in reason


@pytest.mark.parametrize(
    "target",
    [t for t in exporters.all_targets() if t.requires],
    ids=lambda t: f"{t.subject_kind}.{t.id}",
)
def test_a_required_scope_left_out_is_refused(target):
    reason = exporters.scope_refusal(target, ExportScope())
    assert reason is not None and reason.startswith("needs "), reason


def test_the_clip_target_requires_its_frame_range():
    """The one shipped `requires` today: without it the sweep above iterates zero times."""
    clip = exporters.target("recording", "clip")
    assert clip.requires == ("frames",)


def _merge(a: ExportScope, b: ExportScope) -> ExportScope:
    return ExportScope(
        cluster=a.cluster if a.cluster is not None else b.cluster,
        instance_ids=a.instance_ids or b.instance_ids,
        frames=a.frames if a.frames is not None else b.frames,
        recording_ids=a.recording_ids or b.recording_ids,
    )


# --------------------------------------------------------------------------- #
# negative cases: the predicate bites on the shapes the bug took
# --------------------------------------------------------------------------- #
def test_the_sweep_would_catch_a_frames_scope_on_instances():
    """Finding 40's own example: instances narrow by cluster and selection, not frames."""
    instances = exporters.target("analysis_run", "instances")
    reason = exporters.scope_refusal(instances, ExportScope(frames=(0, 3)))
    assert reason == (
        "cannot be narrowed to a frame range "
        "(only to one cluster, a selection, some recordings)"
    )


def test_the_sweep_would_catch_a_cluster_on_a_target_that_only_takes_recordings():
    tracks = exporters.target("run", "tracks")
    assert exporters.scope_refusal(tracks, ExportScope(cluster="3")) is not None


def test_the_sweep_would_catch_a_fake_target_given_a_kind_it_lacks():
    instances = exporters.target("analysis_run", "instances")
    narrow = dataclasses.replace(instances, scopes=("cluster",))
    assert exporters.scope_refusal(narrow, ExportScope(cluster="c01")) is None
    assert exporters.scope_refusal(narrow, ExportScope(instance_ids=("i",))) is not None


def test_a_recordings_scope_naming_the_recording_itself_is_not_a_narrowing(tmp_path):
    """`--for-recording` on a recording's own export names the subject; it narrows nothing."""
    subject = exporters.ExportSubject("recording", tmp_path, "recording_000")
    clip = exporters.target("recording", "clip")
    own = ExportScope(frames=(0, 9), recording_ids=("recording_000",))
    other = ExportScope(frames=(0, 9), recording_ids=("recording_001",))
    assert exporters.scope_refusal(clip, own, subject) is None
    assert exporters.scope_refusal(clip, other, subject) is not None


def test_describe_names_every_part_that_is_set():
    """The manifest's scope line used to name one part: cluster *or* frames, never both."""
    scope = ExportScope(cluster="c01", frames=(0, 9), recording_ids=("r1",))
    assert scope.describe() == "cluster c01, frames 0-9, recording r1"
    assert ExportScope().describe() == "everything"
