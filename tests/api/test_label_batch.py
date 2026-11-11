"""Batch behavior-label import onto existing recordings (Section 3)."""

import numpy as np
import pytest
import yaml

from detail.api.experiment import Experiment
from detail.storage import recording as recording_store

_RECORDING_YAML = {
    "schema_version": 1,
    "name": "s",
    "created_at": "2026-01-01T00:00:00Z",
    "video": {
        "storage": "default",
        "fps_container": 30.0,
        "fps_actual": 30.0,
        "fps_source": "container",
        "resolution": [640, 480],
        "total_duration_s": 1.0,
        "files": [
            {
                "index": 0,
                "file": "video/part_001.mp4",
                "timestamps": {
                    "generated": "video/timestamps/generated/part_001.csv",
                    "user": None,
                    "active": "generated",
                },
            }
        ],
    },
}


def _make_recording(experiment_dir, sid):
    s = experiment_dir / "recordings" / sid
    (s / "video/timestamps/generated").mkdir(parents=True)
    data = {**_RECORDING_YAML, "id": sid, "name": sid}
    (s / "recording.yaml").write_text(
        yaml.dump(data, sort_keys=False), encoding="utf-8"
    )
    recording_store.write_timestamps_csv(
        s / "video/timestamps/generated/part_001.csv",
        np.array([0, 100, 200, 300, 400], dtype=np.int64),
    )


def _labels(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_batch_imports_and_skips_bad_file(experiment_dir, tmp_path):
    _make_recording(experiment_dir, "recording_001")
    _make_recording(experiment_dir, "recording_002")
    good = _labels(tmp_path, "s1.csv", "start_frame,end_frame,label\n1,3,groom\n")
    bad = _labels(tmp_path, "s2.csv", "start_frame,end_frame,label\n1,99,groom\n")
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "labels": {"recording_001": "s1.csv", "recording_002": "s2.csv"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    events = []
    skipped = Experiment(experiment_dir).import_behavior_labels(
        manifest, progress=lambda i, t, sid, f, st: events.append((sid, f, st))
    )

    # one bad file is recorded + skipped; the batch still imports the good one
    assert [sid for sid, _ in skipped] == ["recording_002"]
    proj = Experiment(experiment_dir)
    assert proj.get_recording("recording_001").is_annotated is True
    assert proj.get_recording("recording_002").is_annotated is False
    # each recording resolved against its own map (frame 1..3 -> t_ns 100..300)
    assert proj.get_recording("recording_001").stream_events("behavior") == [
        (100, 300, "groom", None, 1, 3)
    ]
    # per-recording progress fired terminal states for both
    terminal = {(sid, st) for sid, f, st in events if f >= 1.0}
    assert ("recording_001", "done") in terminal
    assert ("recording_002", "skipped") in terminal


def test_batch_unknown_recording_rejected(experiment_dir, tmp_path):
    _make_recording(experiment_dir, "recording_001")
    _labels(tmp_path, "s1.csv", "start_frame,end_frame,label\n1,3,groom\n")
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(
        yaml.dump({"schema_version": 1, "labels": {"nope": "s1.csv"}}, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not in the experiment"):
        Experiment(experiment_dir).import_behavior_labels(manifest)


def test_batch_missing_file_rejected(experiment_dir, tmp_path):
    _make_recording(experiment_dir, "recording_001")
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(
        yaml.dump(
            {"schema_version": 1, "labels": {"recording_001": "missing.csv"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found"):
        Experiment(experiment_dir).import_behavior_labels(manifest)


def test_batch_bad_schema_version(experiment_dir, tmp_path):
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(yaml.dump({"labels": {}}, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        Experiment(experiment_dir).import_behavior_labels(manifest)


def test_batch_manifest_dict_form_declares_covered(experiment_dir, tmp_path):
    _make_recording(experiment_dir, "recording_001")
    _labels(tmp_path, "s1.csv", "start_frame,end_frame,label\n1,3,groom\n")
    manifest = tmp_path / "labels.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "labels": {
                    "recording_001": {
                        "file": "s1.csv",
                        "covered_behaviors": ["groom", "rear"],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    Experiment(experiment_dir).import_behavior_labels(manifest)
    sess = Experiment(experiment_dir).get_recording("recording_001")
    assert set(sess.covered_behaviors()) == {"groom", "rear"}
