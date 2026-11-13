"""Regenerate the SuperAnimal known-answer fixture from deeplabcut itself.

``tests/models/test_superanimal.py`` holds DETAIL's native SuperAnimal code to what deeplabcut
predicts on the same frame and box. This script produces that answer. It needs only
``deeplabcut``, ``numpy`` and ``opencv`` — run it in an environment that has deeplabcut (DETAIL's
own never does, by design, §26):

    <deeplabcut-env>/python tests/models/make_superanimal_fixture.py --weights-dir <dir>

``--weights-dir`` holds ``superanimal_quadruped_hrnet_w32.pt`` and
``superanimal_topviewmouse_hrnet_w32.pt`` (from the mwmathis Hugging Face repos). The frame is
drawn here and saved beside the answer, so the test compares pixels, not a drawing recipe.
Generated 2026-10-03 with deeplabcut 3.0.2 on the CPU.
"""

import argparse
import importlib.metadata
import json
import pathlib

import cv2
import numpy as np

DATA = pathlib.Path(__file__).resolve().parent / "data"
FRAME = DATA / "superanimal_frame.png"
ANSWER = DATA / "superanimal_dlc302.json"
BOX_NORM = (0.18, 0.28, 0.78, 0.74)  # x1, y1, x2, y2 — the tracked box a run would pass
# Boxes (x, y, w, h in pixels) that exercise deeplabcut's crop: inside, wide, tall, off every
# edge, fractional. Their offsets, scales and crop pixels need no weights to compare.
CROP_BOXES = (
    (57.6, 67.2, 192.0, 110.4),
    (100.0, 50.0, 80.0, 40.0),
    (10.5, 30.25, 40.0, 150.0),
    (-20.0, -20.0, 60.0, 60.0),
    (280.0, 200.0, 70.0, 50.0),
    (0.0, 0.0, 320.0, 240.0),
)
HEATMAP_SEED = 7  # numpy default_rng: the same heatmap on every platform
HEATMAP_SHAPE = (2, 5, 64, 64)


def draw_frame() -> np.ndarray:
    """A grey arena with a dark, rat-shaped silhouette seen from above: body, head, ears, tail."""
    rng = np.random.default_rng(20261003)
    img = np.full((240, 320, 3), 150, np.uint8)
    img = np.clip(img.astype(int) + rng.integers(-12, 13, img.shape), 0, 255).astype(
        np.uint8
    )
    cv2.ellipse(img, (150, 125), (62, 32), 12, 0, 360, (60, 55, 50), -1)
    cv2.ellipse(img, (212, 140), (26, 18), 20, 0, 360, (58, 52, 48), -1)
    cv2.circle(img, (205, 118), 8, (90, 80, 85), -1)
    cv2.circle(img, (218, 158), 8, (90, 80, 85), -1)
    cv2.circle(img, (236, 148), 3, (20, 20, 20), -1)
    cv2.line(img, (92, 112), (40, 80), (70, 65, 60), 5)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig

    DATA.mkdir(parents=True, exist_ok=True)
    bgr = draw_frame()
    cv2.imwrite(str(FRAME), bgr)
    rgb = cv2.cvtColor(cv2.imread(str(FRAME)), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = BOX_NORM
    box = np.array([[x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]], dtype=np.float32)
    import hashlib

    import torch
    from deeplabcut.pose_estimation_pytorch.data.image import top_down_crop
    from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import (
        HeatmapPredictor,
    )

    crops = []
    for bx in CROP_BOXES:
        crop, offset, scale = top_down_crop(rgb, np.array(bx), (256, 256), margin=0)
        crops.append(
            {
                "box_xywh": list(bx),
                "offset": [int(offset[0]), int(offset[1])],
                "scale": [float(scale[0]), float(scale[1])],
                "crop_sha256": hashlib.sha256(
                    np.ascontiguousarray(crop).tobytes()
                ).hexdigest(),
            }
        )
    heat = (
        np.random.default_rng(HEATMAP_SEED).random(HEATMAP_SHAPE, dtype=np.float32)
        * 1.4
    )
    predictor = HeatmapPredictor(
        apply_sigmoid=False, clip_scores=True, location_refinement=False
    )
    poses = predictor(4, {"heatmap": torch.from_numpy(heat)})["poses"][:, 0].numpy()
    answer: dict = {
        "deeplabcut": importlib.metadata.version("deeplabcut"),
        "device": "cpu",
        "box_norm": list(BOX_NORM),
        "crops": crops,
        "decode": {
            "seed": HEATMAP_SEED,
            "shape": list(HEATMAP_SHAPE),
            "scale": 1.4,
            "poses": poses.tolist(),
        },
        "models": {},
    }
    for name in ("quadruped", "topviewmouse"):
        # Naming a detector selects deeplabcut's top-down path; only the pose runner is
        # built, so the detector never runs and the box above is what the model sees.
        cfg = PoseConfig.build_for_superanimal_inference(
            f"superanimal_{name}",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            max_individuals=1,
            device="cpu",
        )
        runner = get_pose_inference_runner(
            cfg,
            snapshot_path=args.weights_dir / f"superanimal_{name}_hrnet_w32.pt",
            batch_size=1,
            device="cpu",
            max_individuals=1,
        )
        poses = np.asarray(runner.inference([(rgb, {"bboxes": box})])[0]["bodyparts"])[
            0
        ]
        answer["models"][f"superanimal_{name}"] = {
            "bodyparts": list(cfg.metadata.bodyparts),
            "xy_norm": [[float(x) / w, float(y) / h] for x, y, _ in poses],
            "score": [float(s) for _, _, s in poses],
        }
    ANSWER.write_text(json.dumps(answer, indent=1), encoding="utf-8")
    print(f"wrote {FRAME.name} and {ANSWER.name}")


if __name__ == "__main__":
    main()
