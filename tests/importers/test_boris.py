"""BORIS tabular-events importer (Section 3).

Fixtures mirror the real "Export events" table: one row per START/STOP/POINT marker with a
``Time`` in seconds; state behaviors span a START→STOP pair, points are single POINT rows.
"""

import pytest

from detail.importers.boris import BorisImporter

# A minimal real-shaped events export: one state behavior (groom, two bouts) + one point
# behavior (bite). Header uses BORIS's real column names; extra columns are ignored.
_HEADER = "Observation id,Subject,Behavior,Behavior type,Time,Comment"
_EVENTS_CSV = (
    "\n".join(
        [
            _HEADER,
            "obs1,,groom,START,1.0,",
            "obs1,,groom,STOP,2.0,",
            "obs1,,bite,POINT,3.0,",
            "obs1,,groom,START,5.0,",
            "obs1,,groom,STOP,6.5,",
        ]
    )
    + "\n"
)


def _write(tmp_path, text, name="events.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_declares_role_and_clock():
    imp = BorisImporter()
    assert imp.importer_id == "boris"
    assert imp.role == "behavior"
    assert imp.clock == "video_seconds"


def test_can_import(tmp_path):
    good = _write(tmp_path, _EVENTS_CSV)
    bad = _write(tmp_path, "x,y,z\n1,2,3\n", "n.csv")
    # the aggregated export (has 'Start (s)') is deliberately not this importer
    aggregated = _write(
        tmp_path,
        "Behavior,Behavior type,Start (s),Stop (s)\ngroom,STATE,1,2\n",
        "agg.csv",
    )
    assert BorisImporter().can_import(good)
    assert not BorisImporter().can_import(bad)
    assert not BorisImporter().can_import(aggregated)


def test_pairs_start_stop_and_point(tmp_path):
    result = BorisImporter().import_stream(_write(tmp_path, _EVENTS_CSV), tmp_path, {})
    assert result.shape == "interval"
    assert result.clock == "video_seconds"
    assert result.time_columns == {"start_frame": "t_ns", "end_frame": "end_t_ns"}
    assert result.frame_columns is None
    # START/STOP -> spans; POINT -> zero-length; emitted in time order
    assert result.rows == [
        (1_000_000_000, 2_000_000_000, "groom", None, None, None),
        (3_000_000_000, 3_000_000_000, "bite", None, None, None),
        (5_000_000_000, 6_500_000_000, "groom", None, None, None),
    ]


def test_metadata_covered_and_types(tmp_path):
    md = (
        BorisImporter()
        .import_stream(_write(tmp_path, _EVENTS_CSV), tmp_path, {})
        .metadata
    )
    assert md["covered_behaviors"] == ["bite", "groom"]
    assert md["behavior_types"] == {"groom": "state", "bite": "point"}
    assert md["subjects"] == []


def test_subjects_recorded(tmp_path):
    body = (
        "\n".join([_HEADER, "obs1,m1,groom,START,1.0,", "obs1,m1,groom,STOP,2.0,"])
        + "\n"
    )
    md = (
        BorisImporter()
        .import_stream(_write(tmp_path, body, "subj.csv"), tmp_path, {})
        .metadata
    )
    # subjects are recorded in metadata; the per-row int subject stays null
    assert md["subjects"] == ["m1"]


def test_bom_and_non_ascii_labels(tmp_path):
    """A UTF-8 export with a BOM and accented labels — what BORIS/Excel actually emit.

    Written with an explicit encoding because the locale codec on Windows (cp1252)
    cannot represent these characters; the importer must decode them the same way on
    every OS rather than falling back to the locale.
    """
    body = (
        "\n".join(
            [
                "﻿" + _HEADER,
                "obs1,софия,tête-à-tête,START,1.0,°C",
                "obs1,софия,tête-à-tête,STOP,2.0,",
            ]
        )
        + "\n"
    )
    path = tmp_path / "bom.csv"
    path.write_text(body, encoding="utf-8")
    result = BorisImporter().import_stream(path, tmp_path, {})
    # The BOM is stripped, so the first header name still matches
    assert result.rows == [
        (1_000_000_000, 2_000_000_000, "tête-à-tête", None, None, None)
    ]
    assert result.metadata["subjects"] == ["софия"]


def test_can_import_tolerates_a_bom(tmp_path):
    path = tmp_path / "bom_detect.csv"
    path.write_text("﻿" + _EVENTS_CSV, encoding="utf-8")
    assert BorisImporter().can_import(path)


def test_tab_delimited(tmp_path):
    body = "Behavior\tBehavior type\tTime\n" "groom\tSTART\t1.0\n" "groom\tSTOP\t2.0\n"
    result = BorisImporter().import_stream(
        _write(tmp_path, body, "e.tsv"), tmp_path, {}
    )
    assert result.rows == [(1_000_000_000, 2_000_000_000, "groom", None, None, None)]


def test_empty_behavior(tmp_path):
    body = "\n".join([_HEADER, "obs1,,,START,1.0,"]) + "\n"
    with pytest.raises(ValueError, match="empty Behavior"):
        BorisImporter().import_stream(_write(tmp_path, body, "e.csv"), tmp_path, {})


def test_unknown_marker(tmp_path):
    body = "\n".join([_HEADER, "obs1,,groom,MIDDLE,1.0,"]) + "\n"
    with pytest.raises(ValueError, match="unexpected Behavior type"):
        BorisImporter().import_stream(_write(tmp_path, body, "e.csv"), tmp_path, {})


def test_stop_without_start(tmp_path):
    body = "\n".join([_HEADER, "obs1,,groom,STOP,2.0,"]) + "\n"
    with pytest.raises(ValueError, match="no open START"):
        BorisImporter().import_stream(_write(tmp_path, body, "e.csv"), tmp_path, {})


def test_unpaired_start(tmp_path):
    body = "\n".join([_HEADER, "obs1,,groom,START,1.0,"]) + "\n"
    with pytest.raises(ValueError, match="no matching STOP"):
        BorisImporter().import_stream(_write(tmp_path, body, "e.csv"), tmp_path, {})


def test_missing_required_column(tmp_path):
    body = "Subject,Behavior\nm1,groom\n"
    with pytest.raises(ValueError, match="Behavior type"):
        BorisImporter().import_stream(_write(tmp_path, body, "m.csv"), tmp_path, {})


# -- several observations or subjects, mixed kinds, int64 (#30 findings 4, 5, 7, 8) --------- #
def _boris(rows, header="Observation id,Subject,Behavior,Behavior type,Time"):
    return header + "\n" + "\n".join(rows) + "\n"


_TWO_OBSERVATIONS = _boris(
    [
        "day1,mA,groom,START,0.5",
        "day1,mA,groom,STOP,1.0",
        "day2,mA,groom,START,0.7",
        "day2,mA,groom,STOP,2.0",
    ]
)
_TWO_SUBJECTS = _boris(
    [
        "obs1,mA,groom,START,0.5",
        "obs1,mB,groom,START,0.7",
        "obs1,mA,groom,STOP,1.5",
        "obs1,mB,groom,STOP,2.0",
    ]
)


def _spans(result):
    return [(r[0], r[1], r[2]) for r in result.rows]


def test_several_observations_are_refused_naming_them(tmp_path):
    """Merging them put two sessions on one timeline; overlapping ones failed with a START
    error that pointed at the wrong cause."""
    with pytest.raises(ValueError) as err:
        BorisImporter().import_stream(_write(tmp_path, _TWO_OBSERVATIONS), tmp_path, {})
    assert str(err.value) == (
        "this BORIS export holds 2 observations ('day1', 'day2'). A recording is one "
        "session, so choose one observation; they are never merged"
    )


def test_an_observation_selector_imports_only_its_rows(tmp_path):
    result = BorisImporter().import_stream(
        _write(tmp_path, _TWO_OBSERVATIONS), tmp_path, {}, {"observation": "day2"}
    )
    assert _spans(result) == [(700_000_000, 2_000_000_000, "groom")]
    assert result.metadata["observation"] == "day2"
    assert result.metadata["observations_in_file"] == 2


def test_several_subjects_are_refused_naming_them(tmp_path):
    """Pooled, animal B's grooming scored as a miss or false positive for animal A."""
    with pytest.raises(ValueError) as err:
        BorisImporter().import_stream(_write(tmp_path, _TWO_SUBJECTS), tmp_path, {})
    assert str(err.value) == (
        "this BORIS export scores 2 subjects ('mA', 'mB'). A behavior stream is one "
        "animal's labels, so choose one subject; their bouts are never pooled"
    )


def test_a_subject_selector_imports_only_that_animal(tmp_path):
    result = BorisImporter().import_stream(
        _write(tmp_path, _TWO_SUBJECTS), tmp_path, {}, {"subject": "mB"}
    )
    assert _spans(result) == [(700_000_000, 2_000_000_000, "groom")]
    assert result.metadata["subjects"] == ["mB"]


def test_a_named_observation_the_file_lacks_is_refused(tmp_path):
    with pytest.raises(
        ValueError, match="observation 'day9' is not in this BORIS export"
    ):
        BorisImporter().import_stream(
            _write(tmp_path, _TWO_OBSERVATIONS), tmp_path, {}, {"observation": "day9"}
        )


def test_a_one_observation_one_subject_file_imports_unchanged(tmp_path):
    result = BorisImporter().import_stream(_write(tmp_path, _EVENTS_CSV), tmp_path, {})
    assert len(result.rows) == 3
    assert result.metadata["observation"] == "obs1"


def test_a_label_scored_as_both_bouts_and_points_is_refused(tmp_path):
    """Marked all-point, the 1 s bout was scored as an instant."""
    text = _boris(["o,m,rear,START,0.5", "o,m,rear,STOP,1.5", "o,m,rear,POINT,2.0"])
    with pytest.raises(ValueError) as err:
        BorisImporter().import_stream(_write(tmp_path, text), tmp_path, {})
    assert str(err.value) == (
        "BORIS behavior 'rear' is scored both as START/STOP bouts (rows 2, 3) and as POINT "
        "events (rows 4). A behavior is one kind or the other; rename one of them in BORIS, "
        "or fix its behavior type, and export again"
    )


def test_a_time_beyond_int64_is_refused_with_its_row(tmp_path):
    text = _boris(["o,m,bite,POINT,1e12"])
    with pytest.raises(
        ValueError, match=r"BORIS row 2, Time: 1000000000000.0 seconds is too large"
    ):
        BorisImporter().import_stream(_write(tmp_path, text), tmp_path, {})
