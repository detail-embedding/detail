import numpy as np
import pytest

from detail.core import utils

from detail.core.dataclasses import ComputeTarget, PolygonKeyframes
from detail.core.utils import (
    NS_PER_SECOND,
    active_polygon_at_frame,
    best_supported_backend,
    check_track_is_unambiguous,
    coerce_series_key,
    cuda_device_names,
    default_compute,
    effective_backend,
    parse_series_dirname,
    polygon_at_frame,
    resolve_device,
    resolve_track_id,
    seconds_to_t_ns,
    series_dirname,
    t_ns_to_seconds,
    transcode_parallelism,
)


def _compute(backend, indices=None):
    return ComputeTarget(
        backend=backend,
        device_indices=indices,
        cloud_provider=None,
        cloud_instance=None,
    )


@pytest.mark.parametrize(
    "backend,indices,expected",
    [
        ("cuda", [0, 1], "0,1"),
        ("cuda", [0], "0"),
        ("cuda", None, "0"),
        ("cuda", [], "0"),
        ("mps", None, "mps"),
        ("cpu", None, "cpu"),
    ],
)
def test_resolve_device_mappings(backend, indices, expected):
    supported = ["cpu", "cuda", "mps"]
    assert resolve_device(_compute(backend, indices), supported) == expected


def test_resolve_device_unsupported_backend_raises():
    with pytest.raises(ValueError, match="not supported"):
        resolve_device(_compute("cuda"), ["cpu", "mps"])


def test_resolve_device_unmappable_backend_raises():
    # "cloud" passes the support check but has no local device mapping.
    with pytest.raises(ValueError, match="cannot map backend"):
        resolve_device(_compute("cloud"), ["cloud"])


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("remote", "cuda"),
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        ("mps", "mps"),
    ],
)
def test_effective_backend(backend, expected):
    assert effective_backend(_compute(backend)) == expected


# --- auto-detect backend (cuda > mps > cpu) ---------------------------------- #
# Over every backend, "the best a component supports" is "the best this machine has" —
# the property the deleted best_available_backend used to hold on its own.
_ALL = ["cuda", "mps", "cpu"]


def test_best_backend_prefers_cuda(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    assert best_supported_backend(_ALL) == "cuda"


def test_best_backend_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    assert best_supported_backend(_ALL) == "mps"


def test_best_backend_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: False)
    assert best_supported_backend(_ALL) == "cpu"


def test_best_supported_backend_skips_unavailable_and_unsupported(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: False)
    # cuda is present but this component doesn't accept it -> falls back to cpu
    assert best_supported_backend(["cpu"]) == "cpu"


def test_best_supported_backend_picks_available_supported_accelerator(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    # cuda available but not accepted; mps accepted and available -> mps
    assert best_supported_backend(["cpu", "mps"]) == "mps"


def test_default_compute_sets_device_index_only_for_cuda(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    cuda = default_compute()
    assert cuda.backend == "cuda"
    assert cuda.device_indices == [0]
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: False)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: True)
    mps = default_compute()
    assert mps.backend == "mps"
    assert mps.device_indices is None  # a device index is cuda-only


def test_default_compute_respects_supported_backends(monkeypatch):
    monkeypatch.setattr("detail.core.utils.cuda_available", lambda: True)
    monkeypatch.setattr("detail.core.utils.mps_available", lambda: False)
    # a cpu-only component never gets cuda even when it is present
    assert default_compute(["cpu"]).backend == "cpu"


def test_cuda_device_names_returns_a_list_of_strings():
    # Real call — empty on a CPU/MPS host, device names on an NVIDIA host.
    names = cuda_device_names()
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)


def test_resolve_track_id_int_passthrough():
    assert resolve_track_id(3, None) == 3
    assert resolve_track_id(0, {0: "M001"}) == 0


def test_resolve_track_id_resolves_animal_id():
    assert resolve_track_id("M002", {0: "M001", 1: "M002"}) == 1


def test_resolve_track_id_no_subject_labels_raises():
    with pytest.raises(ValueError, match="no subject_labels"):
        resolve_track_id("M001", None)


def test_resolve_track_id_unknown_animal_raises():
    with pytest.raises(ValueError, match="not found in subject_labels"):
        resolve_track_id("M999", {0: "M001"})


def test_series_dirname_round_trips_a_track_id():
    assert series_dirname(0) == "track_0"
    assert parse_series_dirname("track_0") == 0


def test_parse_series_dirname_rejects_a_non_track_name():
    with pytest.raises(ValueError, match="not a track directory name"):
        parse_series_dirname("embeddings")


def test_coerce_series_key_accepts_ints_and_digit_strings():
    assert coerce_series_key(5) == 5
    assert coerce_series_key(np.int64(5)) == 5
    assert coerce_series_key("5") == 5


def test_coerce_series_key_rejects_garbage():
    with pytest.raises(ValueError, match="not a valid track selector"):
        coerce_series_key("black mouse")


def test_check_track_is_unambiguous_passes_for_a_single_track():
    check_track_is_unambiguous(3, "r1", [3], run_id="emb_01")


def test_check_track_is_unambiguous_raises_on_several_tracks():
    with pytest.raises(ValueError, match="is ambiguous"):
        check_track_is_unambiguous(0, "r1", [0, 1, 2], run_id="emb_01")


def test_t_ns_seconds_conversions():
    assert t_ns_to_seconds(NS_PER_SECOND) == 1.0
    assert seconds_to_t_ns(1.0) == NS_PER_SECOND
    assert seconds_to_t_ns(0.0) == 0


def test_t_ns_seconds_round_trip():
    # 30 fps -> frame 1 at ~33.333 ms
    t_ns = 33_333_333
    assert seconds_to_t_ns(t_ns_to_seconds(t_ns)) == t_ns


# --- polygon_at_frame (Section 10) ------------------------------------------ #
def _square(d):
    return [(d, d), (1 - d, d), (1 - d, 1 - d), (d, 1 - d)]


def test_polygon_at_frame_single_keyframe_always_returns_that():
    pk = PolygonKeyframes(n_vertices=4, keyframes=[(50, _square(0.1))])
    np.testing.assert_array_equal(polygon_at_frame(pk, 0), _square(0.1))
    np.testing.assert_array_equal(polygon_at_frame(pk, 50), _square(0.1))
    np.testing.assert_array_equal(polygon_at_frame(pk, 9999), _square(0.1))


def test_polygon_at_frame_exact_keyframe_returns_it():
    pk = PolygonKeyframes(
        n_vertices=4, keyframes=[(0, _square(0.0)), (10, _square(0.2))]
    )
    np.testing.assert_array_equal(polygon_at_frame(pk, 0), _square(0.0))
    np.testing.assert_array_equal(polygon_at_frame(pk, 10), _square(0.2))


def test_polygon_at_frame_linear_interp_between():
    pk = PolygonKeyframes(
        n_vertices=4, keyframes=[(0, _square(0.0)), (10, _square(0.2))]
    )
    # at frame 5, expect halfway -> inset 0.1
    np.testing.assert_allclose(polygon_at_frame(pk, 5), _square(0.1))


def test_polygon_at_frame_clamps_before_first():
    pk = PolygonKeyframes(
        n_vertices=4, keyframes=[(10, _square(0.1)), (20, _square(0.2))]
    )
    np.testing.assert_array_equal(polygon_at_frame(pk, 0), _square(0.1))


def test_polygon_at_frame_clamps_after_last():
    pk = PolygonKeyframes(
        n_vertices=4, keyframes=[(10, _square(0.1)), (20, _square(0.2))]
    )
    np.testing.assert_array_equal(polygon_at_frame(pk, 9999), _square(0.2))


def test_polygon_at_frame_empty_keyframes_raises():
    pk = PolygonKeyframes(n_vertices=4, keyframes=[])
    with pytest.raises(ValueError, match="empty"):
        polygon_at_frame(pk, 0)


# --- active_polygon_at_frame (Section 10 active interval) ------------------- #


def test_active_polygon_none_interval_is_whole_video():
    pk = PolygonKeyframes(n_vertices=4, keyframes=[(50, _square(0.1))])
    # active=None -> present everywhere, clamped like polygon_at_frame
    np.testing.assert_array_equal(active_polygon_at_frame(pk, None, 0), _square(0.1))
    np.testing.assert_array_equal(active_polygon_at_frame(pk, None, 9999), _square(0.1))


def test_active_polygon_inside_interval_returns_shape():
    pk = PolygonKeyframes(n_vertices=4, keyframes=[(0, _square(0.1))])
    np.testing.assert_array_equal(
        active_polygon_at_frame(pk, (100, 200), 150), _square(0.1)
    )


def test_active_polygon_outside_interval_is_none():
    pk = PolygonKeyframes(n_vertices=4, keyframes=[(0, _square(0.1))])
    assert active_polygon_at_frame(pk, (100, 200), 50) is None
    assert active_polygon_at_frame(pk, (100, 200), 250) is None


def test_active_polygon_interval_boundaries_inclusive():
    pk = PolygonKeyframes(n_vertices=4, keyframes=[(0, _square(0.1))])
    assert active_polygon_at_frame(pk, (100, 200), 100) is not None
    assert active_polygon_at_frame(pk, (100, 200), 200) is not None


def test_active_polygon_works_for_point():
    pk = PolygonKeyframes(n_vertices=1, keyframes=[(0, [(0.5, 0.5)])])
    got = active_polygon_at_frame(pk, None, 10)
    assert got.shape == (1, 2)
    np.testing.assert_array_equal(got, [[0.5, 0.5]])


def test_transcode_parallelism_default_and_env(monkeypatch):
    monkeypatch.delenv("DETAIL_MAX_TRANSCODE_PARALLELISM", raising=False)
    assert 1 <= transcode_parallelism() <= 4  # conservative floor: min(4, cpu_count)
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "7")
    assert transcode_parallelism() == 7  # env overrides, no upper clamp to cpu count
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "0")
    assert transcode_parallelism() == 1  # clamped to >= 1
    monkeypatch.setenv("DETAIL_MAX_TRANSCODE_PARALLELISM", "nonsense")
    assert 1 <= transcode_parallelism() <= 4  # bad value -> fall back to the default


# --------------------------------------------------------------------------- #
# a user's free text used as a path component
# --------------------------------------------------------------------------- #
def test_a_name_that_is_already_safe_is_left_alone():
    """The common case has to keep reading the way the user wrote it, or every exported file
    grows noise for the sake of the rare one."""
    for name in ("rearing", "Supported-Rearing", "grooming_2", "climb wall"):
        assert utils.safe_stem(name) == name.replace(" ", "_")


def test_two_names_that_sanitize_alike_still_get_two_files():
    """The failure that made this shared: collapsing everything awkward to `_` maps `rear/sit`
    and `rear.sit` onto one stem, so two behaviors write to one file and one is gone with nothing
    raised. Sanitizing without disambiguating is the same silent merge as not sanitizing.
    """
    stems = {
        utils.safe_stem(n) for n in ("rear/sit", "rear.sit", "rear:sit", "rear sit")
    }

    assert len(stems) == 4, stems


def test_a_stem_never_carries_a_path_separator_or_a_suffix_dot():
    """`/` made a stem into a directory that does not exist and aborted the export; `.` was
    eaten by the `Path.with_suffix` that follows it."""
    for name in ("rear/sit", "../escape", "rear.sit", "a\\b"):
        stem = utils.safe_stem(name)
        assert "/" not in stem and "\\" not in stem and "." not in stem, stem


def test_a_directory_name_may_keep_its_dot():
    """`safe_path_part(allow_dot=True)` is for a directory, where a dot means nothing; a stem
    about to have a suffix appended is the case that collides."""
    assert utils.safe_path_part("v1.2 run", allow_dot=True) == "v1.2_run"
    assert "." not in utils.safe_path_part("v1.2 run")


@pytest.mark.parametrize(
    "resolved,expected",
    [("cpu", "cpu"), ("mps", "mps"), ("0", "cuda:0"), ("0,1", "cuda:0")],
)
def test_torch_device_translation(resolved, expected):
    """The one translation every torch-backed model's ``load`` uses (was three private
    copies in sam3, dinov3 and pecore; §26 *Sequencing*, PR 0.5)."""
    assert utils.torch_device(resolved) == expected
