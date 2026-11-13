"""Varied frames from a stratified subset of the recordings (issue #671, §26).

**The class: a cost that grows with the input and has no way to be bounded.** A varied sample
reads the shape of every tracked frame (up to ``SHAPE_POOL`` per recording) of every recording it
is given *before* anything is grouped, so on an 80-recording experiment the first pass waited on
80 recordings of masks to choose five frames, and the recordings past the first few dozen add
almost nothing: the groups are formed from the pooled shapes. Nothing was wrong, only slow, and
the only control was to select fewer recordings by hand in the launch form.

**The subset must be a sample of the set, not its head.** Taking the first ``k`` would sample the
first strain, the first day. So the recordings are cut into ``k`` equal runs and one is picked from
each, deterministically for a seed. The tests pin that, that exactly ``k`` recordings are read
(counted at the read, the expensive step), that the run says so (a shorter wait must not look like a
bug), and what must *not* change: ``spread`` still draws from every recording, ``None`` still reads
them all, and a set of ``k`` or fewer is untouched.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from detail.api.experiment import Experiment
from detail.core.skeleton import PRESETS
from detail.pipeline import subject_frames as kr

IDS = [f"recording_{i:03d}" for i in range(40)]


# --------------------------------------------------------------------------- #
# choose_recordings
# --------------------------------------------------------------------------- #
def test_k_or_fewer_recordings_are_all_kept_in_order():
    assert kr.choose_recordings(IDS[:5], 5) == IDS[:5]
    assert kr.choose_recordings(IDS[:5], 20) == IDS[:5]
    assert kr.choose_recordings([], 3) == []


def test_exactly_k_are_chosen_one_from_each_equal_run():
    chosen = kr.choose_recordings(IDS, 4, seed=3)

    assert len(chosen) == 4
    runs = [IDS[0:10], IDS[10:20], IDS[20:30], IDS[30:40]]
    assert [c in run for c, run in zip(chosen, runs)] == [
        True
    ] * 4  # end to end, not the head
    assert chosen == sorted(chosen)  # in the set's own order


@pytest.mark.parametrize("n, k", [(21, 20), (41, 20), (97, 13), (5, 4), (1000, 20)])
def test_every_run_is_non_empty_for_any_size(n, k):
    ids = [f"r{i:04d}" for i in range(n)]

    chosen = kr.choose_recordings(ids, k, seed=1)

    assert len(chosen) == k and len(set(chosen)) == k


def test_the_same_seed_picks_the_same_recordings_and_another_seed_others():
    assert kr.choose_recordings(IDS, 8, seed=5) == kr.choose_recordings(IDS, 8, seed=5)
    assert kr.choose_recordings(IDS, 8, seed=5) != kr.choose_recordings(IDS, 8, seed=6)


def test_one_recording_is_one_recording():
    assert len(kr.choose_recordings(IDS, 1)) == 1


@pytest.mark.parametrize("k", [0, -3])
def test_fewer_than_one_is_refused(k):
    with pytest.raises(ValueError, match="from_recordings must be 1 or more"):
        kr.choose_recordings(IDS, k)


# --------------------------------------------------------------------------- #
# what varied_frames reads
# --------------------------------------------------------------------------- #
@pytest.fixture
def reads(monkeypatch, tmp_path):
    """Count the recordings ``varied_frames`` reads masks of, and stop there.

    The mask read is the expensive step the option exists to bound, so it is what is counted;
    the sampler is then made to find nothing left, which ends it with a ``ValueError`` after the
    shape pass and before anything needs a video.
    """
    read: list[str] = []

    def tracked_rows(recording_dir, tracking_run_id, track_ids=None):
        read.append(recording_dir.name)
        return [], (40, 40)

    monkeypatch.setattr(kr, "_tracked_rows", tracked_rows)
    monkeypatch.setattr(kr, "_require_videos", lambda *a, **k: None)

    def run(n_sources, **kwargs):
        sources = {rid: "track_01" for rid in IDS[:n_sources]}
        events = []
        with pytest.raises(ValueError):
            kr.varied_frames(
                tmp_path,
                SimpleNamespace(id="ex_01", frames=[], rejected=[]),
                sources,
                3,
                n_typical=0,
                add_image=None,
                progress=lambda *event: events.append(event),
                **kwargs,
            )
        return list(read), events

    return run


def test_more_recordings_than_the_default_reads_only_the_default(reads):
    read, _events = reads(40)

    assert len(read) == kr.DEFAULT_FROM_RECORDINGS == 20


def test_the_recordings_read_are_spread_through_the_whole_set(reads):
    read, _events = reads(40, from_recordings=4, seed=2)

    indexes = [IDS.index(rid) for rid in read]
    assert [i // 10 for i in indexes] == [0, 1, 2, 3]  # one from each quarter


def test_none_reads_every_recording(reads):
    read, _events = reads(40, from_recordings=None)

    assert len(read) == 40


def test_the_default_or_fewer_recordings_are_all_read_and_nothing_is_announced(reads):
    read, events = reads(20)

    assert len(read) == 20
    assert "choosing recordings" not in [stage for stage, _f, _d in events]


def test_a_shorter_wait_says_why(reads):
    _read, events = reads(40, from_recordings=5)

    stages = [stage for stage, _f, _d in events]
    assert stages[0] == "choosing recordings"  # before the first thing that takes time
    assert events[0][2] == "sampling from 5 of 40 recordings"


# --------------------------------------------------------------------------- #
# what must not change, and the surfaces
# --------------------------------------------------------------------------- #
def _stub_set(monkeypatch):
    monkeypatch.setattr(
        kr.example_sets,
        "read_set",
        lambda experiment_dir, set_id: SimpleNamespace(
            id=set_id, frames=[], rejected=[]
        ),
    )


def test_spread_still_draws_from_every_recording(monkeypatch, tmp_path):
    _stub_set(monkeypatch)
    seen = {}

    def draw(experiment_dir, sources, n_frames, **kwargs):
        seen["sources"] = len(sources)
        return []

    monkeypatch.setattr(kr, "draw_frames", draw)
    sources = {rid: "track_01" for rid in IDS}

    kr.sample_frames(tmp_path, "ex_01", sources, 3, strategy="spread")

    assert seen["sources"] == 40


def test_sample_frames_hands_the_option_to_the_varied_sampler(monkeypatch, tmp_path):
    _stub_set(monkeypatch)
    seen = []
    monkeypatch.setattr(
        kr, "varied_frames", lambda *a, **k: seen.append(k["from_recordings"]) or []
    )

    for kwargs in ({}, {"from_recordings": 7}, {"from_recordings": None}):
        kr.sample_frames(tmp_path, "ex_01", {IDS[0]: "track_01"}, 1, **kwargs)

    assert seen == [20, 7, None]


@pytest.mark.parametrize("bad", [0, -1])
def test_a_from_recordings_below_one_is_refused_before_anything_is_read(
    monkeypatch, tmp_path, bad
):
    _stub_set(monkeypatch)

    with pytest.raises(ValueError, match="from_recordings must be 1 or more"):
        kr.sample_frames(
            tmp_path, "ex_01", {IDS[0]: "track_01"}, 1, from_recordings=bad
        )


def test_the_api_forwards_the_option_and_defaults_to_twenty(monkeypatch, tmp_path):
    exp = Experiment.create("e", tmp_path)
    exp.create_recording("recording_001")
    ref = exp.create_example_set("set", PRESETS["quadruped_overhead_12"][1])
    seen = []
    monkeypatch.setattr(
        kr, "sample_frames", lambda *a, **k: seen.append(k["from_recordings"]) or []
    )
    sources = {"recording_001": "track_01"}

    exp.sample_example_frames(ref.id, sources, 1)
    exp.sample_example_frames(ref.id, sources, 1, from_recordings=7)
    exp.sample_example_frames(ref.id, sources, 1, from_recordings=None)

    assert seen == [20, 7, None]


def test_the_cli_zero_means_every_recording_and_the_default_is_twenty(
    monkeypatch, tmp_path
):
    from click.testing import CliRunner

    from detail.cli import main

    seen = []

    def fake_sample(self, set_id, sources, n_frames, **kwargs):
        seen.append(kwargs["from_recordings"])
        raise ValueError("stop here")

    monkeypatch.setattr(Experiment, "sample_example_frames", fake_sample)
    Experiment.create("e", tmp_path).create_recording("recording_001")
    base = [
        "keypoints", "examples", "sample", "ex_01",
        "--experiment", str(tmp_path / "e"),
        "--recording", "recording_001",
        "--tracking-run", "track_01",
    ]  # fmt: skip

    for extra in ([], ["--from-recordings", "0"], ["--from-recordings", "9"]):
        result = CliRunner().invoke(main, [*base, *extra])
        assert "stop here" in result.output, result.output

    assert seen == [20, None, 9]
    assert np is not None  # (numpy is what the sampler's index maths is made of)
