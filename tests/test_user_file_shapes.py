"""A file a person wrote by hand is read as written, or refused naming what is wrong.

**The class.** A reader took a hand-written file and went on with a different meaning
from the one written: a misspelled key ignored (``vidoe:`` made a recording with no video),
one word read as a list (``names: nose`` became the points ``n, o, s, e``), a key written
twice keeping the last, two same-named CSV columns collapsing to the second, a header-only
labels file imported as "scored, nothing happened", a half-filled active window widened to
the whole video, a padded region name merged with another, marks in pixels or ``nan``
stored as positions, two fields of view taken where a recording has one, and a model zip
with no weights restored as "installed" (#31, Part 2). Nothing is wrong at the line that
reads the file: every fixture a test writes has the right shape, and the reader is correct
on it. Only a person's file shows it.

**The rule.** A reader of a hand-written file checks each value's shape and each key's
name before using it, and refuses a mistake with a message naming the key, row or value:

- YAML goes through ``storage/user_yaml.py``, which refuses a repeated key;
  ``refuse_unknown_keys`` names a key a reader does not take (labels manifest, skeleton,
  heading definition and stream preset files); the bundle manifest's models are
  ``extra="forbid"``.
- A list is a list: ``Skeleton.from_dict``, ``body_axis.validate_definition``.
- A CSV's header is checked with ``_csv_common.reject_duplicate_columns`` (folded where the
  importer matches headers case-insensitively) and its rows with ``require_data_rows``.
- Example marks pass ``api.keypoints.check_frame_marks`` however they arrive.

**Where it stops.** An example set file keeps keys it does not know under ``extra`` on
purpose (a newer version's fields survive a round trip), so it is not refused for one; the
records DETAIL writes and reads back for itself (``config.yaml``, ``recording.yaml``,
``model.yaml``) are not checked for unknown keys for the same reason. The metadata CSV is
#31's Part 3, and run configs read as lists are its Part 4.

**How this guards it.** Each reader is fed the mistakes a person makes (the cases below),
and each must raise naming the key or row; each reader's documented example must still load
(the negative cases). Two parts are data-driven so a new member is covered the day it is
added: every bundle-manifest model must forbid unknown keys, and every registered stream
importer must be classified in ``_CSV_IMPORTERS`` (and then refuse a repeated column and a
header-only file) or in ``_NOT_CSV`` with the reason.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from detail.analysis.body_axis import validate_definition
from detail.api import keypoints as kp
from detail.api import models as models_api
from detail.api import skeletons as skeletons_api
from detail.api.experiment import Experiment
from detail.api.imports import build_specs
from detail.core.examples import ExampleMark
from detail.core.registries import STREAM_IMPORTERS
from detail.core.skeleton import Skeleton
from detail.importers import import_stream_importers
from detail.models import imported as imported_store
from detail.storage import heading_definitions
from detail.storage import labels_manifest
from detail.storage import manifest as manifest_store
from detail.storage import models as models_store
from detail.storage import spatial as spatial_store
from detail.storage.user_yaml import parse_user_yaml


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- YAML: a key written twice ------------------------------------------------------- #
def test_a_key_written_twice_is_refused_naming_it_and_its_lines():
    with pytest.raises(ValueError) as exc:
        parse_user_yaml("labels:\n  r2: a.csv\n  r2: b.csv\n", "labels.yaml")
    assert "key 'r2' appears twice" in str(exc.value)
    assert "line 3" in str(exc.value) and "first on line 2" in str(exc.value)


def test_a_merged_key_may_still_be_overridden():
    data = parse_user_yaml("base: &b {x: 1, y: 2}\nc:\n  <<: *b\n  x: 3\n", "s.yaml")
    assert data["c"] == {"x": 3, "y": 2}


# --- the bundle manifest ------------------------------------------------------------- #
def test_every_manifest_model_forbids_unknown_keys():
    for model in manifest_store._MANIFEST_MODELS:
        assert model.model_config.get("extra") == "forbid", model.__name__


@pytest.mark.parametrize(
    "manifest, expected",
    [
        (
            {"schema_version": 1, "recordings": [{"id": "r1", "vidoe": ["c.mp4"]}]},
            "recording 'r1': unknown key 'vidoe' (it takes id, metadata, video, streams)",
        ),
        (
            {"schema_version": 1, "recordings": [{"id": "r1", "name": "Mouse 1"}]},
            "recording 'r1': unknown key 'name'",
        ),
        (
            {"schema_version": 1, "recording": [], "recordings": [{"id": "r1"}]},
            "unknown key 'recording'",
        ),
        (
            {
                "schema_version": 1,
                "recordings": [
                    {
                        "id": "r1",
                        "streams": [
                            {
                                "name": "s",
                                "importer": "medpc",
                                "raw": "d",
                                "ofset_ns": 5,
                            }
                        ],
                    }
                ],
            },
            "recording 'r1' stream[0]: unknown key 'ofset_ns'",
        ),
        (
            {
                "schema_version": 1,
                "recordings": [{"id": "r1", "metadata": {"recording_id": "other"}}],
            },
            "metadata cannot set 'recording_id'",
        ),
    ],
    ids=["recording-key", "name", "top-level-key", "stream-key", "recording_id"],
)
def test_the_manifest_refuses_a_key_it_does_not_take(tmp_path, manifest, expected):
    errors, _warnings = manifest_store.validate_manifest(tmp_path, manifest, [])
    assert any(expected in e for e in errors), errors


def test_a_blank_manifest_metadata_value_is_stored_blank_not_none(tmp_path):
    manifest = {
        "schema_version": 1,
        "recordings": [{"id": "r1", "metadata": {"sex": None, "box": 4}}],
    }
    assert manifest_store.validate_manifest(tmp_path, manifest, []) == ([], [])
    (spec,) = build_specs(tmp_path, manifest)
    assert spec.metadata == {"sex": "", "box": "4"}
    assert (spec.id, spec.name) == ("r1", "r1")


# --- the labels manifest ------------------------------------------------------------- #
_LABELS_DOC = """\
schema_version: 1
labels:
  recording_001: labels/s1.csv
  recording_002: {file: labels/s2.csv, covered_behaviors: [groom, rear]}
"""


def _labels(tmp_path: Path, text: str):
    for name in ("s1.csv", "s2.csv"):
        _write(tmp_path / "labels" / name, "start_frame,end_frame,label\n0,1,groom\n")
    path = _write(tmp_path / "labels.yaml", text)
    data = labels_manifest.load_labels_manifest(path)
    return labels_manifest.resolve_labels_manifest(
        path, data, ["recording_001", "recording_002"]
    )


def test_the_documented_labels_manifest_loads(tmp_path):
    mapping = _labels(tmp_path, _LABELS_DOC)
    assert [(sid, covered) for sid, _f, covered in mapping] == [
        ("recording_001", None),
        ("recording_002", ["groom", "rear"]),
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "schema_version: 1\nlabels:\n  recording_001: {file: labels/s1.csv, "
            "covered: [groom]}\n",
            "labels['recording_001']: unknown key(s) 'covered'",
        ),
        (
            "schema_version: 1\nlabel: {}\nlabels:\n  recording_001: labels/s1.csv\n",
            "has unknown key(s) 'label'",
        ),
        (
            "schema_version: 1\nlabels:\n  recording_001: labels/s1.csv\n"
            "  recording_001: labels/s2.csv\n",
            "key 'recording_001' appears twice",
        ),
        (
            "schema_version: 1\nlabels:\n  recording_001: {file: labels/s1.csv, "
            "covered_behaviors: groom}\n",
            "covered_behaviors must be a list",
        ),
    ],
    ids=["entry-key", "top-level-key", "recording-twice", "string-for-list"],
)
def test_the_labels_manifest_refuses_a_mistake(tmp_path, text, expected):
    with pytest.raises(ValueError) as exc:
        _labels(tmp_path, text)
    assert expected in str(exc.value)


# --- skeleton and heading files ------------------------------------------------------ #
def test_an_exported_skeleton_file_and_the_documented_bare_one_load(tmp_path):
    bare = _write(
        tmp_path / "bare.yaml", "names: [nose, tail]\nedges: [[nose, tail]]\n"
    )
    assert skeletons_api.resolve_skeleton(bare) == Skeleton(
        names=("nose", "tail"), edges=(("nose", "tail"),)
    )
    made = skeletons_api.create_skeleton("rat", bare)
    out = skeletons_api.export_skeleton(made["id"], tmp_path / "out.yaml")
    assert skeletons_api.resolve_skeleton(out) == made["skeleton"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("names: nose\n", "'names' must be a list of point names"),
        ("names: [a, b]\npairs: [ab]\n", "'pairs' must be a list of [two point names]"),
        ("names: [a, b]\nedges: a-b\n", "'edges' must be a list of [two point names]"),
        ("names: [a, b]\nedges: [[a, b, c]]\n", "[two point names]"),
        ("names: [a, b]\nedgse: [[a, b]]\n", "unknown key(s) 'edgse'"),
        ("skeleton: {names: [a], pairz: []}\n", "unknown key(s) 'pairz'"),
    ],
    ids=[
        "names-word",
        "pair-word",
        "edge-word",
        "edge-of-three",
        "bare-key",
        "body-key",
    ],
)
def test_a_skeleton_file_refuses_a_mistake(tmp_path, text, expected):
    path = _write(tmp_path / "s.yaml", text)
    with pytest.raises(ValueError) as exc:
        skeletons_api.resolve_skeleton(path)
    assert expected in str(exc.value)
    assert str(path) in str(exc.value)


def test_a_heading_file_reads_one_word_as_one_point(experiment_dir, tmp_path):
    path = _write(
        tmp_path / "h.yaml", "schema_version: 1\nname: h1\nfront: nose\nback: [tail]\n"
    )
    heading_definitions.import_definition(experiment_dir, path)
    saved = heading_definitions.get_definition(experiment_dir, "h1")
    assert (saved["front"], saved["back"]) == (["nose"], ["tail"])
    heading_definitions.export_definition(experiment_dir, "h1", tmp_path / "out.yaml")
    assert heading_definitions.import_definition(
        experiment_dir, tmp_path / "out.yaml", name="h2"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("front: {nose: 1}\nback: [tail]\n", "'front' must be a list of point names"),
        ("front: [[nose]]\nback: [tail]\n", "['nose'] is not one"),
        ("front: [nose]\nbak: [tail]\n", "unknown key(s) 'bak'"),
    ],
    ids=["mapping", "nested", "key"],
)
def test_a_heading_file_refuses_a_mistake(experiment_dir, tmp_path, text, expected):
    path = _write(tmp_path / "h.yaml", "schema_version: 1\nname: h1\n" + text)
    with pytest.raises(ValueError) as exc:
        heading_definitions.import_definition(experiment_dir, path)
    assert expected in str(exc.value)


def test_validate_definition_still_takes_a_lone_name():
    assert validate_definition({"front": "nose", "back": ["tail"]}) == {
        "front": ["nose"],
        "back": ["tail"],
    }


# --- example marks ------------------------------------------------------------------- #
def _example_file(tmp_path: Path, marks: list[dict]) -> Path:
    cv2.imwrite(str(tmp_path / "f.png"), np.zeros((8, 8, 3), np.uint8))
    body = {
        "schema_version": 1,
        "name": "s",
        "skeleton": {"names": ["nose", "tail"]},
        "frames": [
            {
                "image": "f.png",
                "recording_id": "r1",
                "tracking_run_id": "t",
                "track_id": 0,
                "part_id": "part_001",
                "frame_idx": 0,
                "t_ns": 0,
                "bbox": [0, 0, 1, 1],
                "marks": marks,
            }
        ],
    }
    return _write(tmp_path / "examples.yaml", yaml.safe_dump(body, sort_keys=False))


def test_a_prepared_example_set_with_good_marks_imports(tmp_path):
    proj = Experiment.create("p", tmp_path / "ws")
    path = _example_file(
        tmp_path,
        [
            {"target": "nose", "x": 0.2, "y": 0.3},
            {"target": "tail", "x": 1.0, "y": 0.0},
        ],
    )
    made = proj.import_example_set(path)
    assert [(m.target, m.x, m.y) for m in made.frames[0].marks] == [
        ("nose", 0.2, 0.3),
        ("tail", 1.0, 0.0),
    ]


@pytest.mark.parametrize(
    "marks, expected",
    [
        ([{"target": "nose", "x": 5, "y": -3}], "is off the frame"),
        ([{"target": "nose", "x": 412, "y": 300}], "not pixels"),
        ([{"target": "nose", "x": math.nan, "y": 0.5}], "is not a position"),
        ([{"target": "nose", "x": 0.1, "y": math.inf}], "is not a position"),
        (
            [
                {"target": "nose", "x": 0.1, "y": 0.1},
                {"target": "nose", "x": 0.2, "y": 0.2},
            ],
            "'nose' is marked twice",
        ),
        ([{"target": "ear", "x": 0.1, "y": 0.1}], "no keypoint 'ear'"),
    ],
    ids=["off-frame", "pixels", "nan", "inf", "twice", "unknown-target"],
)
def test_an_example_import_refuses_what_labeling_refuses(tmp_path, marks, expected):
    proj = Experiment.create("p", tmp_path / "ws")
    with pytest.raises(ValueError) as exc:
        proj.import_example_set(_example_file(tmp_path, marks))
    assert expected in str(exc.value)
    assert "frame 0" in str(exc.value)
    with pytest.raises(ValueError, match=expected.replace("'", ".")):
        kp.check_frame_marks(
            [ExampleMark.from_dict(m) for m in marks],
            Skeleton(names=("nose", "tail")),
            "a set",
        )


# --- CSV stream importers ------------------------------------------------------------ #
#: ``{importer: (a header with one column, a data row, the header with that column
#: repeated as a person types it)}``. Every registered importer is here or in _NOT_CSV.
_CSV_IMPORTERS = {
    "timeseries_csv": ("seconds,bpm", "0.0,72", "seconds,bpm,bpm"),
    "events_csv": ("seconds,label", "1.0,press", "seconds,label,Label"),
    "behavior_csv": (
        "start_frame,end_frame,label",
        "0,5,groom",
        "start_frame,end_frame,label,LABEL",
    ),
    "boris": (
        "Time,Behavior,Behavior type",
        "1.0,groom,POINT",
        "Time,Behavior,Behavior type,behavior",
    ),
}
_NOT_CSV = {"medpc": "reads a MED-PC data file, which has no CSV header"}


def test_every_stream_importer_is_classified():
    import_stream_importers()
    assert set(STREAM_IMPORTERS) == set(_CSV_IMPORTERS) | set(_NOT_CSV)


def _import(importer: str, tmp_path: Path, text: str):
    path = _write(tmp_path / f"{importer}.csv", text)
    return STREAM_IMPORTERS[importer]().import_stream(path, tmp_path / "out", {}, None)


@pytest.mark.parametrize("importer", sorted(_CSV_IMPORTERS))
def test_a_csv_importer_reads_its_documented_shape(importer, tmp_path):
    import_stream_importers()
    header, row, _dup = _CSV_IMPORTERS[importer]
    assert len(_import(importer, tmp_path, f"{header}\n{row}\n").rows) == 1


@pytest.mark.parametrize("importer", sorted(_CSV_IMPORTERS))
def test_a_csv_importer_refuses_a_repeated_column(importer, tmp_path):
    import_stream_importers()
    _header, row, dup = _CSV_IMPORTERS[importer]
    with pytest.raises(ValueError, match="more than once"):
        _import(importer, tmp_path, f"{dup}\n{row},x\n")


@pytest.mark.parametrize("importer", sorted(_CSV_IMPORTERS))
def test_a_csv_importer_refuses_a_header_only_file(importer, tmp_path):
    import_stream_importers()
    header, _row, _dup = _CSV_IMPORTERS[importer]
    with pytest.raises(ValueError, match="no data rows"):
        _import(importer, tmp_path, f"{header}\n")


# --- the spatial CSV ----------------------------------------------------------------- #
_SPATIAL = "kind,name,active_start,active_end,frame_idx,vertex_index,x,y\n"
_TRIANGLE = "0,0,0.1,0.1\n{k},{n},{w},0,1,0.2,0.1\n{k},{n},{w},0,2,0.2,0.2\n"


def _spatial(tmp_path: Path, body: str):
    return spatial_store.load_spatial_csv(_write(tmp_path / "rig.csv", _SPATIAL + body))


def test_an_exported_spatial_setup_loads_back_as_it_was(recording_dir, tmp_path):
    triangle = [(0, [(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)])]
    for kind, def_id, name, active in [
        ("roi", "roi_001", " padded ", (5, 9)),
        ("point", "point_001", "lever", None),
    ]:
        keyframes = triangle if kind == "roi" else [(0, [(0.5, 0.5)])]
        n = spatial_store.write_keyframes(recording_dir, kind, def_id, keyframes)
        spatial_store.append_to_manifest(recording_dir, kind, def_id, name, n, active)
    spatial_store.export_spatial(recording_dir, tmp_path / "out.csv")
    back = spatial_store.load_spatial_csv(tmp_path / "out.csv")
    assert [(d["kind"], d["name"], d["active"]) for d in back] == [
        ("roi", "padded", (5, 9)),
        ("point", "lever", None),
    ]
    assert [d["name"] for d in spatial_store.list_defs(recording_dir, "roi")] == [
        "padded"
    ]


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            "roi,nest,5,," + _TRIANGLE.format(k="roi", n="nest", w="5,"),
            "row 2: active_start and active_end",
        ),
        (
            "roi,nest,,,"
            + _TRIANGLE.format(k="roi", n="nest", w=",")
            + "roi,nest ,,,0,3,0.3,0.3\n",
            "row 5: name 'nest ' has a space",
        ),
        (
            "roi,z,,,0,0,0.1,0.1\nroi,z,,,0,1,0.2,0.1\nroi,z,,,0,5,0.2,0.2\n",
            "vertex indexes [0, 1, 5]",
        ),
        (
            "roi,z,,,"
            + _TRIANGLE.format(k="roi", n="z", w=",")
            + "roi,z,,,0,2,0.9,0.9\n",
            "row 5: roi 'z' frame 0 vertex 2 is given twice (first on row 4)",
        ),
        (
            "roi,z,5,9,"
            + _TRIANGLE.format(k="roi", n="z", w=",").replace(",,,0,2", ",1,9,0,2"),
            "has active window",
        ),
    ],
    ids=["half-window", "padded-name", "skipped-vertex", "repeated-row", "two-windows"],
)
def test_the_spatial_csv_refuses_a_hand_edit_it_would_misread(tmp_path, body, expected):
    with pytest.raises(ValueError) as exc:
        _spatial(tmp_path, body)
    assert expected in str(exc.value)


def test_the_spatial_csv_refuses_a_repeated_column(tmp_path):
    path = _write(
        tmp_path / "rig.csv", _SPATIAL.rstrip("\n") + ",x\nroi,z,,,0,0,0.1,0.1,0.2\n"
    )
    with pytest.raises(ValueError, match="more than once"):
        spatial_store.load_spatial_csv(path)


def test_two_fields_of_view_are_refused_naming_both():
    square = [(0, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])]
    defs = [
        {"kind": "arena", "name": name, "keyframes": square, "active": None}
        for name in ("a1", "a2")
    ]
    with pytest.raises(ValueError, match=r"holds 2 fields of view \('a1', 'a2'\)"):
        spatial_store.prepare_for_arrival(defs)
    assert spatial_store.prepare_for_arrival(defs[:1])[0][0]["name"] == "a1"


# --- model restore ------------------------------------------------------------------- #
def _catalog_model(model_id: str) -> Path:
    mdir = imported_store.model_dir(model_id)
    (mdir / "weights").mkdir(parents=True)
    (mdir / imported_store.WEIGHTS_RELPATH).write_bytes(b"weights")
    models_store.write_model_yaml(
        mdir,
        {
            "id": model_id,
            "name": model_id,
            "source": "imported",
            "model_family": "yolo",
            "task": "detect",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": imported_store.WEIGHTS_RELPATH},
            "status": "complete",
        },
    )
    return mdir


def _rezip(source: Path, dest: Path, rename=None, drop=()) -> Path:
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(dest, "w") as out:
        for member in src.namelist():
            if member in drop:
                continue
            out.writestr(rename(member) if rename else member, src.read(member))
    return dest


def test_a_model_export_restores(tmp_path):
    _catalog_model("mine")
    archive = models_api.export_model("mine", tmp_path / "mine.zip")
    imported_store.remove("mine")
    assert models_api.import_model_archive(archive) == "mine"


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda a, d: _rezip(a, d, drop={"mine/weights/best.pt"}), "holds no weights"),
        (
            lambda a, d: _rezip(a, d, rename=lambda m: m.replace("mine/", "other/")),
            "is model 'mine', but its folder in the archive is 'other'",
        ),
        (
            lambda a, d: _rezip(
                a,
                d,
                rename=lambda m: m.removeprefix("mine/"),
                drop={"mine/weights/best.pt"},
            ),
            "holds a model folder's contents (model.yaml is at the top of the zip)",
        ),
    ],
    ids=["no-weights", "renamed-folder", "one-file"],
)
def test_a_damaged_model_zip_is_refused_and_leaves_nothing(tmp_path, mutate, expected):
    _catalog_model("mine")
    archive = models_api.export_model("mine", tmp_path / "mine.zip")
    imported_store.remove("mine")
    bad = mutate(archive, tmp_path / "bad.zip")
    with pytest.raises(ValueError) as exc:
        models_api.import_model_archive(bad)
    assert expected in str(exc.value)
    assert list(imported_store.catalog_dir().iterdir()) == []


# --- stream preset files ------------------------------------------------------------- #
def test_a_stream_preset_file_round_trips_and_refuses_an_unknown_key(
    experiment_dir, tmp_path
):
    from detail.storage import stream_presets

    stream_presets.save_preset(
        experiment_dir, "lab", "timeseries_csv", {"time_column": "t"}, offset_ns=5
    )
    stream_presets.export_preset(experiment_dir, "lab", tmp_path / "p.yaml")
    assert stream_presets.import_preset(experiment_dir, tmp_path / "p.yaml", name="b")
    text = (tmp_path / "p.yaml").read_text(encoding="utf-8-sig")
    bad = _write(tmp_path / "bad.yaml", text.replace("offset_ns:", "ofset_ns:"))
    with pytest.raises(ValueError, match="unknown key\\(s\\) 'ofset_ns'"):
        stream_presets.import_preset(experiment_dir, bad, name="c")
