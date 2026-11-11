"""The YOLO trainer's pose task (§12 *Distill keypoints*, PR 8): a base checkpoint that can
learn the dataset's labels, and metrics that rank a pose run by its keypoint mAP."""

import pytest

from detail.storage import datasets as datasets_store
from detail.trainers.ultralytics_yolo import UltralyticsYOLOTrainer


def _dataset(root, *, pose: bool):
    root.mkdir(parents=True, exist_ok=True)
    datasets_store.write_data_yaml(
        root,
        {0: "mouse"},
        kpt_shape=(3, 3) if pose else None,
        flip_idx=[0, 2, 1] if pose else None,
    )
    return root


def test_check_base_model_matches_the_dataset_task(tmp_path):
    t = UltralyticsYOLOTrainer()
    pose, seg = _dataset(tmp_path / "pose", pose=True), _dataset(
        tmp_path / "seg", pose=False
    )
    assert t.dataset_task(pose) == "pose"
    assert t.dataset_task(seg) == "segment"
    t.check_base_model("yolo11s-pose.pt", pose)
    t.check_base_model("yolo11s-seg.pt", seg)
    t.check_base_model("yolo11n.pt", seg)
    with pytest.raises(ValueError, match="not a pose checkpoint"):
        t.check_base_model("yolo11s-seg.pt", pose)
    with pytest.raises(
        ValueError, match="is a pose checkpoint, but the dataset has no"
    ):
        t.check_base_model("yolo11s-pose.pt", seg)


def test_final_metrics_report_pose_map_and_rank_the_best_epoch_by_it(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50(P)\n"
        "1,0.50,0.20\n"
        "2,0.90,0.40\n"  # the best box epoch
        "3,0.60,0.70\n"  # the best pose epoch: what best.pt is for in a pose run
        "4,0.55,0.65\n",
        encoding="utf-8",
    )
    out = UltralyticsYOLOTrainer().final_metrics(tmp_path)
    assert out["final_epoch"] == 4
    assert out["pose_map50"] == 0.65 and out["box_map50"] == 0.55
    assert out["best_epoch"] == 3
    assert out["best_pose_map50"] == 0.70 and out["best_box_map50"] == 0.60
    assert "mask_map50" not in out


def test_final_metrics_without_pose_keep_ranking_by_box(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50(M)\n1,0.5,0.4\n2,0.9,0.8\n3,0.6,0.5\n",
        encoding="utf-8",
    )
    assert UltralyticsYOLOTrainer().final_metrics(tmp_path) == {
        "final_epoch": 3,
        "box_map50": 0.6,
        "mask_map50": 0.5,
        "best_epoch": 2,
        "best_box_map50": 0.9,
        "best_mask_map50": 0.8,
    }


def test_data_yaml_carries_kpt_shape_and_flip_idx_only_for_a_pose_dataset(tmp_path):
    import yaml

    datasets_store.write_data_yaml(
        tmp_path, {0: "mouse"}, kpt_shape=(4, 3), flip_idx=[0, 2, 1, 3]
    )
    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8-sig"))
    assert data["kpt_shape"] == [4, 3] and data["flip_idx"] == [0, 2, 1, 3]
    assert data["path"] == "." and data["names"] == {0: "mouse"}

    datasets_store.write_data_yaml(tmp_path, {0: "mouse"})
    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8-sig"))
    assert "kpt_shape" not in data and "flip_idx" not in data
    with pytest.raises(ValueError, match="give kpt_shape"):
        datasets_store.write_data_yaml(tmp_path, {0: "mouse"}, flip_idx=[0])
    with pytest.raises(ValueError, match="2 entries for 4 keypoints"):
        datasets_store.write_data_yaml(
            tmp_path, {0: "mouse"}, kpt_shape=(4, 3), flip_idx=[0, 1]
        )
