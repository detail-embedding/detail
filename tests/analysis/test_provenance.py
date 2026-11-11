"""Provenance capture: lineage walk, hashing/fingerprinting, and drift detection."""

import yaml

from detail.analysis import provenance
from detail.storage import experiment as experiment_store


def _recording_record():
    return {
        "schema_version": 1,
        "id": "s1",
        "video": {"files": [{"index": 0, "file": "video/part_001.mp4"}]},
        "streams": [{"name": "beh", "role": "behavior"}],
        "detection_runs": [
            {"id": "det_01", "model_id": "model_002", "weights_source": "trained"}
        ],
        "tracking_runs": [
            {"id": "track_01", "detection_run_id": "det_01", "tracker_id": "iou"}
        ],
        "embedding_runs": [
            {
                "id": "emb_01",
                "tracking_run_id": "track_01",
                "model_id": "dinov3_convnext_base",
                "model_variant": "convnext_base",
                "constructor": "silhouette_centered",
            }
        ],
    }


def _build_recording(
    experiment_dir, labels_text: str = "t_ns,end_t_ns,code\n1,2,climb\n"
):
    sdir = experiment_store.recording_path(experiment_dir, "s1")
    (sdir / "video").mkdir(parents=True, exist_ok=True)
    (sdir / "video" / "part_001.mp4").write_bytes(b"\x00" * ((1 << 20) + 8))
    (sdir / "streams" / "beh" / "canonical").mkdir(parents=True, exist_ok=True)
    (sdir / "streams" / "beh" / "canonical" / "part_001.csv").write_text(
        labels_text, encoding="utf-8"
    )
    (sdir / "recording.yaml").write_text(
        yaml.dump(_recording_record(), sort_keys=False), encoding="utf-8"
    )
    return sdir


def test_embedding_lineage_walks_the_dag():
    lineage = provenance.embedding_lineage(_recording_record(), "emb_01")
    assert lineage["embedding"]["constructor"] == "silhouette_centered"
    assert lineage["tracking"]["id"] == "track_01"
    assert lineage["detection"]["id"] == "det_01"
    assert lineage["detection"]["weights_source"] == "trained"
    assert lineage["video"] == ["video/part_001.mp4"]


def test_embedding_lineage_unknown_run_raises():
    import pytest

    with pytest.raises(ValueError, match="not in recording record"):
        provenance.embedding_lineage(_recording_record(), "emb_99")


def test_file_fingerprint_and_full_hash(tmp_path):
    small = tmp_path / "labels.csv"
    small.write_text("a,b\n1,2\n", encoding="utf-8")
    assert len(provenance.sha256_file(small)) == 64
    big = tmp_path / "video.mp4"
    big.write_bytes(b"x" * (2 << 20))
    fp = provenance.file_fingerprint(big)
    assert fp["size"] == 2 << 20 and len(fp["sha256_ends"]) == 64


def test_capture_assembles_lineage_and_hashes(tmp_path):
    _build_recording(tmp_path)
    prov = provenance.capture(
        tmp_path, recording_ids=["s1"], members=[("silhouette", "emb_01")]
    )
    assert prov["environment"]["python"]  # env captured (no backend load)
    sess = prov["recordings"]["s1"]
    assert len(sess["labels_hash"]) == 64
    assert "video/part_001.mp4" in sess["videos"]
    assert (
        sess["members"]["silhouette"]["embedding"]["model_id"] == "dinov3_convnext_base"
    )


def test_drift_detects_label_edit(tmp_path):
    _build_recording(tmp_path)
    recorded = provenance.capture(
        tmp_path, recording_ids=["s1"], members=[("s", "emb_01")]
    )
    assert provenance.drift(recorded, recorded) == []  # identical => no drift
    # edit the labels, re-capture: drift is reported
    _build_recording(tmp_path, labels_text="t_ns,end_t_ns,code\n1,2,climb\n3,4,climb\n")
    current = provenance.capture(
        tmp_path, recording_ids=["s1"], members=[("s", "emb_01")]
    )
    assert any("labels changed" in c for c in provenance.drift(recorded, current))


def test_drift_reports_code_and_video_change():
    recorded = {
        "code_sha": "aaa",
        "recordings": {"s1": {"labels_hash": "h", "videos": {"v.mp4": {"size": 1}}}},
    }
    current = {
        "code_sha": "bbb",
        "recordings": {"s1": {"labels_hash": "h", "videos": {"v.mp4": {"size": 2}}}},
    }
    changes = provenance.drift(recorded, current)
    assert any("code changed" in c for c in changes)
    assert any("video v.mp4 changed" in c for c in changes)
