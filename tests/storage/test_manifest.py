"""Bundle-manifest loading + validation (Section 3) — the bulk-import robustness core."""

import cv2
import numpy as np
import pytest
import yaml

from detail.api.imports import build_specs
from detail.api.experiment import Experiment
from detail.storage import manifest as mf

DATA_TXT = (
    "Start Date: 02/16/26\nSubject: TEST1\nBox: 4\nStart Time: 14:00:38\n"
    "MSN: Test_MSN\nG:       3.000\nH:      10.000\nE:\n"
    "     0:       19.000       11.000      100.000\nT:\n"
    "     0:        0.000        1.500       10.000\n"
)
PROGRAM_MPC = "\\  Event stamps ( array E )\n\\   11 = a\n\\   19 = b\n\\   100 = end\n"


def _video(path, w=32, h=24, n=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 9) % 250 + 1, np.uint8))
    vw.release()


def _bundle(tmp_path, manifest):
    bundle = tmp_path / "bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "manifest.yaml").write_text(
        yaml.dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return bundle


def _validate(tmp_path, manifest, existing=None):
    bundle = _bundle(tmp_path, manifest)
    return bundle, mf.validate_manifest(bundle, manifest, existing or [])


# --- load ------------------------------------------------------------------- #
def test_load_missing_manifest_raises(tmp_path):
    (tmp_path / "bundle").mkdir()
    with pytest.raises(FileNotFoundError):
        mf.load_manifest(tmp_path / "bundle")


def test_load_bad_structure_raises(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.yaml").write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError, match="recordings"):
        mf.load_manifest(bundle)


# --- happy path + build_specs ----------------------------------------------- #
def test_valid_manifest_builds_specs(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _video(bundle / "videos" / "a.mp4")
    _video(bundle / "videos" / "b.mp4")
    (bundle / "medpc").mkdir()
    (bundle / "medpc" / "m1.txt").write_text(DATA_TXT, encoding="utf-8")
    (bundle / "medpc" / "prog.MPC").write_text(PROGRAM_MPC, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "recordings": [
            {
                "id": "recording_001",
                "metadata": {"animal_id": "M001", "treatment": "drug_A"},
                "video": ["videos/a.mp4", "videos/b.mp4"],
                "streams": [
                    {
                        "name": "medpc",
                        "importer": "medpc",
                        "raw": "medpc/m1.txt",
                        "aux": {"program": "medpc/prog.MPC"},
                        "offset_ns": 250,
                    }
                ],
            },
            {"id": "recording_002", "metadata": {"animal_id": "M002"}},  # metadata-only
        ],
    }
    (bundle / "manifest.yaml").write_text(
        yaml.dump(manifest, sort_keys=False), encoding="utf-8"
    )
    errors, warnings = mf.validate_manifest(bundle, manifest, [])
    assert errors == []

    specs = build_specs(bundle, manifest)
    assert [s.id for s in specs] == ["recording_001", "recording_002"]
    assert len(specs[0].video_parts) == 2  # ordered parts
    assert specs[0].streams[0].importer == "medpc"
    assert specs[0].streams[0].offset_ns == 250
    assert specs[0].streams[0].raw == bundle / "medpc/m1.txt"
    assert specs[1].video_parts == [] and specs[1].metadata == {"animal_id": "M002"}

    # end-to-end through the API
    experiment = Experiment.create("p", tmp_path)
    from detail.api.imports import import_specs

    failed, _ = import_specs(experiment, specs)
    assert failed == []
    assert {s.id for s in experiment.recordings} == {"recording_001", "recording_002"}


# --- error classes (all reported up front) ---------------------------------- #
def test_duplicate_id_in_manifest(tmp_path):
    _, (errors, _) = _validate(tmp_path, {"recordings": [{"id": "s1"}, {"id": "s1"}]})
    assert any("duplicate id" in e for e in errors)


def test_collision_with_existing_recording(tmp_path):
    _, (errors, _) = _validate(
        tmp_path, {"recordings": [{"id": "s1"}]}, existing=["s1"]
    )
    assert any("already exists" in e for e in errors)


def test_bad_id_charset(tmp_path):
    _, (errors, _) = _validate(tmp_path, {"recordings": [{"id": "bad id/x"}]})
    assert any("must match" in e for e in errors)


def test_video_not_found(tmp_path):
    _, (errors, _) = _validate(
        tmp_path, {"recordings": [{"id": "s1", "video": ["nope.mp4"]}]}
    )
    assert any("video not found" in e for e in errors)


def test_non_homogeneous_parts(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _video(bundle / "a.mp4", w=32, h=24)
    _video(bundle / "b.mp4", w=64, h=48)  # different resolution
    manifest = {"recordings": [{"id": "s1", "video": ["a.mp4", "b.mp4"]}]}
    errors, _ = mf.validate_manifest(bundle, manifest, [])
    assert any("share fps + resolution" in e for e in errors)


def test_unknown_importer_and_missing_aux(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "m.txt").write_text(DATA_TXT, encoding="utf-8")
    manifest = {
        "recordings": [
            {
                "id": "s1",
                "streams": [
                    {"name": "x", "importer": "nope", "raw": "m.txt"},
                    {"name": "y", "importer": "medpc", "raw": "m.txt"},  # missing aux
                ],
            }
        ]
    }
    errors, _ = mf.validate_manifest(bundle, manifest, [])
    assert any("unknown importer" in e for e in errors)
    assert any("missing aux 'program'" in e for e in errors)


def test_stream_config_is_validated_upfront(tmp_path):
    """A bad importer config is a manifest error, not a per-recording failure an hour in.

    The bundle is the only bulk path, so a 60-recording import that discovers a mistyped
    column on recording 41 has already written 40 recordings' streams (Tenet 10).
    """
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "s.csv").write_text("t_sec,v\n0,1\n", encoding="utf-8")
    manifest = {
        "recordings": [
            {
                "id": "s1",
                "streams": [
                    {
                        "name": "bad",
                        "importer": "timeseries_csv",
                        "raw": "s.csv",
                        "config": {"time_kind": "furlongs"},
                    },
                    {
                        "name": "none_taken",
                        "importer": "boris",
                        "raw": "s.csv",
                        "config": {"time_column": "t_sec"},
                    },
                ],
            }
        ]
    }
    errors, _ = mf.validate_manifest(bundle, manifest, [])
    assert any("invalid config" in e for e in errors), errors
    assert any("takes no config" in e for e in errors), errors


def test_a_valid_stream_config_passes(tmp_path):
    """Guard the guard: the check above must not reject every config it sees."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "s.csv").write_text("t_sec,v\n0,1\n", encoding="utf-8")
    manifest = {
        "recordings": [
            {
                "id": "s1",
                "streams": [
                    {
                        "name": "ok",
                        "importer": "timeseries_csv",
                        "raw": "s.csv",
                        "config": {
                            "time_column": "t_sec",
                            "time_kind": "seconds",
                            "channels": [{"column": "v", "name": "v"}],
                        },
                        "offset_ns": 5,
                    }
                ],
            }
        ]
    }
    errors, _ = mf.validate_manifest(bundle, manifest, [])
    assert not errors, errors


def test_aux_wrong_extension(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "m.txt").write_text(DATA_TXT, encoding="utf-8")
    (bundle / "prog.txt").write_text(PROGRAM_MPC, encoding="utf-8")  # should be .mpc
    manifest = {
        "recordings": [
            {
                "id": "s1",
                "streams": [
                    {
                        "name": "medpc",
                        "importer": "medpc",
                        "raw": "m.txt",
                        "aux": {"program": "prog.txt"},
                    }
                ],
            }
        ]
    }
    errors, _ = mf.validate_manifest(bundle, manifest, [])
    assert any("must be ['.mpc']" in e for e in errors)


def test_bad_schema_version_rejected(tmp_path):
    _, (errors, _) = _validate(
        tmp_path, {"schema_version": 2, "recordings": [{"id": "s1"}]}
    )
    assert any("schema_version" in e for e in errors)


def test_offset_ns_must_be_int(tmp_path):
    manifest = {
        "recordings": [
            {
                "id": "s1",
                "streams": [
                    {"name": "a", "importer": "medpc", "raw": "m.txt", "offset_ns": "x"}
                ],
            }
        ]
    }
    _, (errors, _) = _validate(tmp_path, manifest)
    assert any("offset_ns must be an integer" in e for e in errors)


# --- schema is the single source for the generated reference (§7) ----------- #
def test_metadata_values_coerced_to_strings():
    model = mf.Manifest.model_validate(
        {"recordings": [{"id": "s1", "metadata": {"box": 4}}]}
    )
    assert model.recordings[0].metadata == {"box": "4"}


def test_describe_manifest_reflects_the_models():
    """describe_manifest() renders straight from the models, so a new field is
    documented automatically and the reference cannot drift from validation."""
    sections = {s["model"]: s for s in mf.describe_manifest()}
    assert set(sections) == {"Manifest", "RecordingSpec", "StreamSpec"}
    for model in (mf.Manifest, mf.RecordingSpec, mf.StreamSpec):
        fields = sections[model.__name__]["fields"]
        assert [f["name"] for f in fields] == list(model.model_fields)
        assert all(f["help"] for f in fields)  # every field carries help text
