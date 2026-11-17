import pytest
import yaml

from detail.storage import recording as recording_store
from detail.storage import spatial


def _square(d=0.0):
    return [(d, d), (1 - d, d), (1 - d, 1 - d), (d, 1 - d)]


def test_write_and_read_single_keyframe(recording_dir):
    n = spatial.write_arena_keyframes(recording_dir, "arena_001", [(0, _square())])
    assert n == 4
    pk = spatial.read_arena_keyframes(recording_dir, "arena_001", 4)
    assert pk.n_vertices == 4
    assert len(pk.keyframes) == 1
    assert pk.keyframes[0][0] == 0
    assert pk.keyframes[0][1] == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_write_csv_header_matches_n_vertices(recording_dir):
    spatial.write_arena_keyframes(recording_dir, "arena_001", [(0, _square())])
    csv_path = recording_dir / "arenas" / "arena_001" / "keyframes.csv"
    header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert header == [
        "frame_idx",
        "v0x",
        "v0y",
        "v1x",
        "v1y",
        "v2x",
        "v2y",
        "v3x",
        "v3y",
    ]


def test_write_multiple_keyframes_roundtrip(recording_dir):
    keyframes = [
        (0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        (100, [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
    ]
    spatial.write_arena_keyframes(recording_dir, "arena_001", keyframes)
    pk = spatial.read_arena_keyframes(recording_dir, "arena_001", 4)
    assert pk.keyframes == keyframes


def test_write_rejects_empty_keyframes(recording_dir):
    with pytest.raises(ValueError, match="non-empty"):
        spatial.write_arena_keyframes(recording_dir, "arena_001", [])


def test_write_rejects_too_few_vertices(recording_dir):
    with pytest.raises(ValueError, match="at least 3 vertices"):
        spatial.write_arena_keyframes(
            recording_dir, "arena_001", [(0, [(0.0, 0.0), (1.0, 1.0)])]
        )


def test_write_rejects_out_of_range_coords(recording_dir):
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        spatial.write_arena_keyframes(
            recording_dir, "arena_001", [(0, [(0.0, 0.0), (1.5, 0.0), (1.0, 1.0)])]
        )


def test_write_rejects_negative_frame_idx(recording_dir):
    with pytest.raises(ValueError, match=">= 0"):
        spatial.write_arena_keyframes(recording_dir, "arena_001", [(-1, _square())])


def test_write_rejects_non_monotonic_frame_idx(recording_dir):
    with pytest.raises(ValueError, match="strictly increasing"):
        spatial.write_arena_keyframes(
            recording_dir, "arena_001", [(0, _square()), (0, _square())]
        )


def test_write_rejects_inconsistent_vertex_count(recording_dir):
    keyframes = [
        (0, _square()),
        (10, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]),  # 3 verts, not 4
    ]
    with pytest.raises(ValueError, match="expected 4"):
        spatial.write_arena_keyframes(recording_dir, "arena_001", keyframes)


def test_read_rejects_header_mismatch(recording_dir):
    spatial.write_arena_keyframes(recording_dir, "arena_001", [(0, _square())])
    with pytest.raises(ValueError, match="expected 13 columns"):
        spatial.read_arena_keyframes(recording_dir, "arena_001", 6)


def test_read_missing_file(recording_dir):
    with pytest.raises(FileNotFoundError):
        spatial.read_arena_keyframes(recording_dir, "arena_999", 4)


def test_append_to_recording_yaml_keeps_schema_version_first(recording_dir):
    spatial.write_arena_keyframes(recording_dir, "arena_001", [(0, _square())])
    spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "main", 4)
    text = (recording_dir / "recording.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0].startswith("schema_version:")
    data = yaml.safe_load(text)
    assert data["arenas"] == [{"id": "arena_001", "name": "main", "n_vertices": 4}]


def test_append_rejects_duplicate_id(recording_dir):
    spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "main", 4)
    with pytest.raises(ValueError, match="id 'arena_001' already exists"):
        spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "other", 4)


def test_append_rejects_duplicate_name(recording_dir):
    spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "main", 4)
    with pytest.raises(ValueError, match="name 'main' already exists"):
        spatial.append_arena_to_recording_yaml(recording_dir, "arena_002", "main", 4)


def test_append_rejects_empty_name(recording_dir):
    with pytest.raises(ValueError, match="non-empty"):
        spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "", 4)


def test_next_arena_id_increments(recording_dir):
    assert spatial.next_arena_id(recording_dir) == "arena_001"
    spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "main", 4)
    assert spatial.next_arena_id(recording_dir) == "arena_002"


def test_list_arenas_empty_recording(recording_dir):
    assert spatial.list_arenas(recording_dir) == []


# --- generalized kinds: ROIs (polygons) and POIs (points) ------------------- #


def test_write_point_single_vertex(recording_dir):
    n = spatial.write_keyframes(
        recording_dir, "point", "point_001", [(0, [(0.5, 0.5)])]
    )
    assert n == 1
    pk = spatial.read_keyframes(recording_dir, "point", "point_001", 1)
    assert pk.n_vertices == 1
    assert pk.keyframes == [(0, [(0.5, 0.5)])]


def test_point_rejects_multiple_vertices(recording_dir):
    with pytest.raises(ValueError, match="exactly 1 vertex"):
        spatial.write_keyframes(
            recording_dir, "point", "point_001", [(0, [(0.1, 0.1), (0.2, 0.2)])]
        )


def test_roi_polygon_min_vertices(recording_dir):
    with pytest.raises(ValueError, match="at least 3 vertices"):
        spatial.write_keyframes(
            recording_dir, "roi", "roi_001", [(0, [(0.0, 0.0), (1.0, 1.0)])]
        )


def test_roi_written_under_rois_dir(recording_dir):
    spatial.write_keyframes(recording_dir, "roi", "roi_001", [(0, _square())])
    assert (recording_dir / "rois" / "roi_001" / "keyframes.csv").is_file()


def test_unknown_kind_rejected(recording_dir):
    with pytest.raises(ValueError, match="unknown spatial kind"):
        spatial.write_keyframes(recording_dir, "bogus", "x_001", [(0, _square())])


def test_next_id_per_kind(recording_dir):
    assert spatial.next_id(recording_dir, "roi") == "roi_001"
    spatial.append_to_manifest(recording_dir, "roi", "roi_001", "zone", 4)
    assert spatial.next_id(recording_dir, "roi") == "roi_002"
    assert spatial.next_id(recording_dir, "point") == "point_001"  # independent counter


def test_a_deleted_region_id_is_not_handed_to_the_next_region(recording_dir):
    """The lowest free number gave a new region a deleted one's id, so a finished location
    run's ``roi_001`` rows joined to a different region."""
    first = spatial.next_id(recording_dir, "roi")
    spatial.write_keyframes(recording_dir, "roi", first, [(0, _square())])
    spatial.append_to_manifest(recording_dir, "roi", first, "nest", 4)
    spatial.delete_def(recording_dir, "roi", first)
    assert spatial.next_id(recording_dir, "roi") != first


def test_append_roi_with_active_interval(recording_dir):
    spatial.append_to_manifest(
        recording_dir, "roi", "roi_001", "phase", 4, active=(1500, 3000)
    )
    assert spatial.list_defs(recording_dir, "roi") == [
        {
            "id": "roi_001",
            "name": "phase",
            "n_vertices": 4,
            "active": {"start": 1500, "end": 3000},
        }
    ]


def test_append_roi_active_none_is_whole_video(recording_dir):
    spatial.append_to_manifest(recording_dir, "roi", "roi_001", "zone", 4)
    assert spatial.list_defs(recording_dir, "roi")[0]["active"] is None


def test_arena_rejects_active_interval(recording_dir):
    with pytest.raises(ValueError, match="does not support an active interval"):
        spatial.append_to_manifest(
            recording_dir, "arena", "arena_001", "a", 4, active=(0, 10)
        )


def test_active_interval_rejects_start_after_end(recording_dir):
    with pytest.raises(ValueError, match="must be <= end"):
        spatial.append_to_manifest(
            recording_dir, "roi", "roi_001", "z", 4, active=(3000, 1500)
        )


def test_active_interval_rejects_negative(recording_dir):
    with pytest.raises(ValueError, match=">= 0"):
        spatial.append_to_manifest(
            recording_dir, "roi", "roi_001", "z", 4, active=(-1, 10)
        )


def test_delete_def_removes_dir_and_manifest_entry(recording_dir):
    spatial.write_keyframes(recording_dir, "point", "point_001", [(0, [(0.5, 0.5)])])
    spatial.append_to_manifest(recording_dir, "point", "point_001", "lever", 1)
    spatial.delete_def(recording_dir, "point", "point_001")
    assert not (recording_dir / "points" / "point_001").exists()
    assert spatial.list_defs(recording_dir, "point") == []


# --- bulk replace (manager "Save all") -------------------------------------- #
def test_replace_defs_replaces_all_and_reassigns_ids(recording_dir):
    spatial.replace_defs(
        recording_dir,
        "roi",
        [
            {"name": "a", "keyframes": [(0, _square())], "active": None},
            {"name": "b", "keyframes": [(0, _square(0.2))], "active": (0, 900)},
        ],
    )
    assert [e["id"] for e in spatial.list_defs(recording_dir, "roi")] == [
        "roi_001",
        "roi_002",
    ]
    # replacing with a single new def drops the rest; the new one takes an id never used
    # before, not a dropped region's
    ids = spatial.replace_defs(
        recording_dir, "roi", [{"name": "c", "keyframes": [(0, _square())]}]
    )
    assert ids == ["roi_003"]
    entries = spatial.list_defs(recording_dir, "roi")
    assert [e["name"] for e in entries] == ["c"]
    assert not (recording_dir / "rois" / "roi_001").exists()  # old dirs gone
    assert not (recording_dir / "rois" / "roi_002").exists()


def test_replace_defs_keeps_a_surviving_regions_id(recording_dir):
    """The scene editor's Save: delete ``nest``, keep ``food``. ``food`` stays ``roi_002``
    and a region added in the same Save gets ``roi_003``, never the freed ``roi_001``.
    """
    spatial.replace_defs(
        recording_dir,
        "roi",
        [
            {"name": "nest", "keyframes": [(0, _square())]},
            {"name": "food", "keyframes": [(0, _square(0.2))]},
        ],
    )
    ids = spatial.replace_defs(
        recording_dir,
        "roi",
        [
            {"id": "roi_002", "name": "food", "keyframes": [(0, _square(0.2))]},
            {"name": "water", "keyframes": [(0, _square(0.3))]},
        ],
    )
    assert ids == ["roi_002", "roi_003"]
    assert {e["id"]: e["name"] for e in spatial.list_defs(recording_dir, "roi")} == {
        "roi_002": "food",
        "roi_003": "water",
    }


def test_replace_defs_refuses_a_duplicated_id(recording_dir):
    spatial.replace_defs(
        recording_dir, "roi", [{"name": "a", "keyframes": [(0, _square())]}]
    )
    with pytest.raises(ValueError, match="roi id 'roi_001' is duplicated"):
        spatial.replace_defs(
            recording_dir,
            "roi",
            [
                {"id": "roi_001", "name": "a", "keyframes": [(0, _square())]},
                {"id": "roi_001", "name": "b", "keyframes": [(0, _square())]},
            ],
        )


def test_replace_defs_gives_a_fresh_id_for_one_no_longer_defined(recording_dir):
    """A stale editor naming an id deleted elsewhere does not resurrect it."""
    first = spatial.replace_defs(
        recording_dir, "roi", [{"name": "a", "keyframes": [(0, _square())]}]
    )[0]
    spatial.delete_def(recording_dir, "roi", first)
    ids = spatial.replace_defs(
        recording_dir,
        "roi",
        [{"id": first, "name": "a", "keyframes": [(0, _square())]}],
    )
    assert ids != [first]


def test_replace_defs_validates_before_writing(recording_dir):
    spatial.replace_defs(
        recording_dir, "roi", [{"name": "keep", "keyframes": [(0, _square())]}]
    )
    with pytest.raises(ValueError, match="non-empty"):
        spatial.replace_defs(
            recording_dir,
            "roi",
            [
                {"name": "ok", "keyframes": [(0, _square())]},
                {"name": "", "keyframes": [(0, _square())]},  # invalid → abort
            ],
        )
    # nothing was applied — the original set is intact
    assert [e["name"] for e in spatial.list_defs(recording_dir, "roi")] == ["keep"]


# --- CSV export / import (carry-over) --------------------------------------- #
def test_export_spatial_roundtrips(recording_dir):
    spatial.replace_defs(
        recording_dir,
        "roi",
        [{"name": "zone", "keyframes": [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])]}],
    )
    spatial.replace_defs(
        recording_dir,
        "point",
        [{"name": "lever", "keyframes": [(0, [(0.5, 0.5)])], "active": (0, 30)}],
    )
    out = recording_dir / "regions.csv"
    spatial.export_spatial(recording_dir, out)
    assert out.read_text(encoding="utf-8-sig") == (
        "kind,name,active_start,active_end,frame_idx,vertex_index,x,y,"
        "distance_mm,pixels_per_mm,ref_width,ref_height\n"
        "roi,zone,,,0,0,0.1,0.1,,,,\n"
        "roi,zone,,,0,1,0.9,0.1,,,,\n"
        "roi,zone,,,0,2,0.5,0.9,,,,\n"
        "point,lever,0,30,0,0,0.5,0.5,,,,\n"
    )
    defs = spatial.load_spatial_csv(out)
    assert [(d["kind"], d["name"]) for d in defs] == [
        ("roi", "zone"),
        ("point", "lever"),
    ]
    assert defs[0]["keyframes"] == [(0, [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)])]
    assert defs[1]["active"] == (0, 30)
    assert spatial.load_calibration_csv(out) is None  # none set on this recording


def test_export_spatial_includes_the_field_of_view(recording_dir):
    # One file is the whole spatial setup, so the field of view rides along with the
    # regions rather than having to be redrawn on every recording.
    spatial.write_arena_keyframes(recording_dir, "arena_001", [(0, _square())])
    spatial.append_arena_to_recording_yaml(recording_dir, "arena_001", "fov", 4)
    spatial.replace_defs(
        recording_dir, "roi", [{"name": "z", "keyframes": [(0, _square())]}]
    )
    out = recording_dir / "regions.csv"
    spatial.export_spatial(recording_dir, out)
    defs = spatial.load_spatial_csv(out)
    assert [(d["kind"], d["name"]) for d in defs] == [("arena", "fov"), ("roi", "z")]
    assert defs[0]["keyframes"] == [(0, _square())]
    # ...and callers that want only part of it can still narrow.
    spatial.export_spatial(recording_dir, out, kinds=("roi", "point"))
    assert [d["kind"] for d in spatial.load_spatial_csv(out)] == ["roi"]


def test_export_spatial_carries_a_two_point_calibration(recording_dir):
    spatial.write_calibration_from_points(
        recording_dir, [(0.25, 0.5), (0.75, 0.5)], 100.0, (1920, 1080)
    )
    out = recording_dir / "setup.csv"
    spatial.export_spatial(recording_dir, out)
    block = spatial.load_calibration_csv(out)
    assert block["points"] == [[0.25, 0.5], [0.75, 0.5]]
    assert block["distance_mm"] == 100.0
    assert block["ref_resolution"] == [1920, 1080]
    assert block["pixels_per_mm"] == pytest.approx(0.5 * 1920 / 100.0)
    # The polygon loader ignores the calibration rows rather than choking on them.
    assert spatial.load_spatial_csv(out) == []


def test_export_spatial_carries_a_scalar_calibration(recording_dir):
    # A scalar calibration has no pick points, so it exports as one point-less row.
    spatial.write_calibration_scalar(recording_dir, 3.5, (640, 480))
    out = recording_dir / "setup.csv"
    spatial.export_spatial(recording_dir, out)
    block = spatial.load_calibration_csv(out)
    assert block["points"] is None and block["distance_mm"] is None
    assert block["pixels_per_mm"] == 3.5 and block["ref_resolution"] == [640, 480]


def test_load_calibration_csv_tolerates_a_pre_calibration_export(tmp_path):
    # A CSV written before the calibration columns existed must still load — the four
    # columns were appended for exactly this reason.
    old = tmp_path / "old.csv"
    old.write_text(
        "kind,name,active_start,active_end,frame_idx,vertex_index,x,y\n"
        "roi,zone,,,0,0,0.1,0.1\n"
        "roi,zone,,,0,1,0.9,0.1\n"
        "roi,zone,,,0,2,0.5,0.9\n",
        encoding="utf-8",
    )
    assert spatial.load_calibration_csv(old) is None
    assert [d["name"] for d in spatial.load_spatial_csv(old)] == ["zone"]


def test_copy_calibration_replaces_and_requires_a_source(experiment_dir, recording_dir):
    dest = experiment_dir / "recordings" / "dest"
    (dest / "rois").mkdir(parents=True)
    recording_store.write_recording(dest, {"schema_version": 1, "id": "dest"})
    with pytest.raises(ValueError, match="no distance calibration"):
        spatial.copy_calibration(recording_dir, dest)

    spatial.write_calibration_scalar(recording_dir, 2.0, (1920, 1080))
    spatial.write_calibration_scalar(dest, 9.0, (640, 480))
    spatial.copy_calibration(recording_dir, dest)
    block = spatial.read_calibration(dest)
    # Replaced outright, and it keeps the resolution it was measured at so
    # pixels_per_mm_at can rescale for this recording's video.
    assert block["pixels_per_mm"] == 2.0 and block["ref_resolution"] == [1920, 1080]
    assert spatial.pixels_per_mm_at(block, 960, 540) == 1.0


def test_load_spatial_csv_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("kind,name,x,y\nroi,z,0.1,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        spatial.load_spatial_csv(bad)


def test_import_spatial_appends_and_suffixes_duplicate_names(recording_dir):
    spatial.replace_defs(
        recording_dir, "roi", [{"name": "zone", "keyframes": [(0, _square())]}]
    )
    spatial.import_spatial(
        recording_dir,
        [{"kind": "roi", "name": "zone", "keyframes": [(0, _square(0.1))]}],
    )
    names = [e["name"] for e in spatial.list_defs(recording_dir, "roi")]
    assert names == ["zone", "zone (2)"]  # appended, not clobbered
    assert [e["id"] for e in spatial.list_defs(recording_dir, "roi")] == [
        "roi_001",
        "roi_002",
    ]


def test_arrival_policy_flattens_a_moving_polygon(recording_dir):
    """Keyframe motion belongs to the source's own timeline, so it does not travel.

    What survives is the polygon at frame 0 — which is the *first keyframe's* vertices,
    since `polygon_at_frame` clamps below it. A moving polygon tracks one camera's pan;
    a static one states the rig geometry, and only the second is true of another
    recording.
    """
    spatial.replace_defs(
        recording_dir,
        "arena",
        [
            {
                "name": "panning rig",
                "keyframes": [(0, _square()), (900, _square(0.05))],
            }
        ],
    )
    defs, flattened, refused = spatial.prepare_for_arrival(
        spatial.read_defs(recording_dir, "arena")
    )
    assert flattened == ["panning rig"] and refused == []
    assert len(defs) == 1
    assert defs[0]["keyframes"] == [(0, [(x, y) for x, y in _square()])]


def test_arrival_policy_leaves_a_static_polygon_alone(recording_dir):
    """The common case reports nothing, so a notice means something actually happened."""
    spatial.replace_defs(
        recording_dir, "roi", [{"name": "nest", "keyframes": [(0, _square())]}]
    )
    defs, flattened, refused = spatial.prepare_for_arrival(
        spatial.read_defs(recording_dir, "roi")
    )
    assert flattened == [] and refused == []
    assert [d["name"] for d in defs] == ["nest"]


def test_arrival_policy_refuses_an_active_window(recording_dir):
    """An active window cannot be carried and cannot be dropped, so it is left behind.

    Keeping `(5000, 9000)` on a shorter recording leaves a region that exists and applies
    to no frame; dropping it widens the user's claim to the whole video and invents
    analysis rows they never asked for. Widening is the worse of the two — wrong
    information rather than missing — and unlike the geometry there is no way to *see*
    it happened, because the preview draws frame 0 only.
    """
    spatial.replace_defs(
        recording_dir,
        "roi",
        [
            {"name": "nest", "keyframes": [(0, _square())]},
            {
                "name": "post-injection zone",
                "keyframes": [(0, _square(0.05))],
                "active": (5000, 9000),
            },
        ],
    )
    defs, flattened, refused = spatial.prepare_for_arrival(
        spatial.read_defs(recording_dir, "roi")
    )
    assert [d["name"] for d in defs] == ["nest"]
    assert flattened == []
    assert [name for name, _ in refused] == ["post-injection zone"]
    assert "5000" in refused[0][1] and "9000" in refused[0][1]


def test_arrival_policy_carries_no_active_key_at_all(recording_dir):
    """Guard the guard: a def that reaches `import_spatial` must not smuggle an active
    window through in its payload — the refusal above is the only handling there is."""
    spatial.replace_defs(
        recording_dir, "roi", [{"name": "nest", "keyframes": [(0, _square())]}]
    )
    defs, _flattened, _refused = spatial.prepare_for_arrival(
        spatial.read_defs(recording_dir, "roi")
    )
    assert "active" not in defs[0]


def test_reading_defs_rejects_an_unknown_kind(recording_dir):
    with pytest.raises(ValueError, match="unknown spatial kind"):
        spatial.read_defs(recording_dir, "nonsense")


def test_spatial_exports_root(tmp_path, monkeypatch):
    from detail.storage import locations

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    root = spatial.spatial_exports_root()
    assert root == tmp_path / "roi_exports" and root.is_dir()


# --- distance calibration (§10) --------------------------------------------- #
def test_write_calibration_from_points_derives_pixels_per_mm(recording_dir):
    # two points 0.5 of the width apart at 1920×1080 → 960 px across 200 mm → 4.8 px/mm
    block = spatial.write_calibration_from_points(
        recording_dir, [(0.25, 0.5), (0.75, 0.5)], 200.0, (1920, 1080)
    )
    assert block["schema_version"] == 1
    assert abs(block["pixels_per_mm"] - 4.8) < 1e-9
    assert block["ref_resolution"] == [1920, 1080]
    assert spatial.read_calibration(recording_dir) == block


def test_pixels_per_mm_at_rescales_with_resolution(recording_dir):
    block = spatial.write_calibration_from_points(
        recording_dir, [(0.25, 0.5), (0.75, 0.5)], 200.0, (1920, 1080)
    )
    # points-based: recomputed exactly at a downsampled resolution (half → 2.4 px/mm)
    assert abs(spatial.pixels_per_mm_at(block, 960, 540) - 2.4) < 1e-9


def test_write_calibration_scalar_rescales_by_width(recording_dir):
    block = spatial.write_calibration_scalar(recording_dir, 5.0, (1920, 1080))
    assert block["points"] is None
    assert abs(spatial.pixels_per_mm_at(block, 960, 540) - 2.5) < 1e-9  # 5 · 960/1920


def test_delete_calibration_is_idempotent(recording_dir):
    spatial.write_calibration_scalar(recording_dir, 5.0, (1920, 1080))
    spatial.delete_calibration(recording_dir)
    assert spatial.read_calibration(recording_dir) is None
    spatial.delete_calibration(recording_dir)  # no error when already absent


def test_calibration_rejects_bad_input(recording_dir):
    with pytest.raises(ValueError, match="distinct"):
        spatial.write_calibration_from_points(
            recording_dir, [(0.5, 0.5), (0.5, 0.5)], 200.0, (1920, 1080)
        )
    with pytest.raises(ValueError, match="distance_mm must be"):
        spatial.write_calibration_from_points(
            recording_dir, [(0.2, 0.5), (0.8, 0.5)], 0.0, (1920, 1080)
        )
    with pytest.raises(ValueError, match="pixels_per_mm must be"):
        spatial.write_calibration_scalar(recording_dir, -1.0, (1920, 1080))
