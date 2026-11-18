"""A writer that reduces a multi-piece artifact to its largest piece.

The class: a serializer that must emit one shape per object, handed an object whose mask has
come apart — where the pieces that detach are the ones setting its *extent*. A YOLO-seg label
line is one polygon, which constrains the *count* and says nothing about *which* one; bridging
the pieces into a single ring is what Ultralytics' own COCO converter does.

How it hid, twice over. Most of the area survives, so a rendered label looks right. And
every fixture here used a single blob, on which "keep the largest contour" and "keep every
contour" are the same function. On real data an animal's mask often splits at a neck or a
tail (#407 — masks splitting into multiple contours), and each time the label
silently shrank; because Ultralytics derives a label's **box** from its polygon
(`segments2boxes`), the box target shrank with it and `crop_mask` sliced the prediction flat
along that edge.

The rule: emit every contour, and test a mask that is genuinely in pieces. Both the spec's own
neighbouring sentence and the parameter's help text already said every contour — only the code
picked one, which is why reading either one past it looked like agreement.
"""

import numpy as np

from detail.core.utils import mask_to_polygon
from detail.storage import datasets as ds
from detail.storage import experiment as experiment_store


# --- mask_to_polygon -------------------------------------------------------- #
def test_mask_to_polygon_square_is_normalized():
    mask = np.zeros((100, 200), bool)  # h=100, w=200
    mask[20:60, 40:120] = True  # a rectangle
    poly = mask_to_polygon(mask)
    assert poly is not None
    assert len(poly) >= 6 and len(poly) % 2 == 0  # >=3 (x, y) points
    xs, ys = poly[0::2], poly[1::2]
    assert all(0.0 <= x <= 1.0 for x in xs)  # x normalized by width
    assert all(0.0 <= y <= 1.0 for y in ys)  # y normalized by height
    assert max(xs) <= (120 / 200) + 1e-6  # within the rect's right edge
    assert max(ys) <= (60 / 100) + 1e-6


def test_mask_to_polygon_empty_is_none():
    assert mask_to_polygon(np.zeros((50, 50), bool)) is None


def test_mask_to_polygon_below_min_area_is_none():
    mask = np.zeros((50, 50), bool)
    mask[10:13, 10:13] = True  # 9 px
    assert mask_to_polygon(mask, min_area=100) is None
    assert mask_to_polygon(mask, min_area=0) is not None


def test_mask_to_polygon_bounds_every_piece_of_a_split_mask():
    """A multi-piece artifact reduced to its largest piece — where the pieces that
    detach are exactly the ones that set the object's extent.

    A YOLO-seg label line is one polygon, so ``mask_to_polygon`` took the largest
    external contour and dropped the rest. An animal's mask splits constantly — a thin
    dark neck, a tail, a limb across the body (#407) — and the piece that detaches is
    always an extremity, never the
    middle. Ultralytics derives the label's *box* from the polygon it is handed
    (``ops.segments2boxes``), so the box shrank with the polygon, the distilled model
    learned to stop at the neck, and ``ops.crop_mask`` — which clips every predicted
    mask to its predicted box — then sliced the mask flat along it. Nothing failed
    anywhere on the way: the label parses, most of the area survives, training
    converges, the run completes, and the only symptom is an animal with no head.

    Both halves are asserted. The bounds are the box that gets derived; the fill is the
    union, because a convex hull would bound every piece too and would be a worse label
    than the one this replaces.
    """
    import cv2

    h, w = 200, 400
    mask = np.zeros((h, w), bool)
    mask[80:140, 150:330] = True  # body
    mask[95:125, 118:146] = True  # head, detached by a 4-px dark neck
    mask[70:78, 330:390] = True  # tail, detached

    poly = np.array(mask_to_polygon(mask, min_area=50)).reshape(-1, 2) * [w, h]
    assert poly[:, 0].min() <= 118  # the head sets the left edge, not the body
    assert poly[:, 0].max() >= 389  # the tail sets the right edge
    assert poly[:, 1].min() <= 70  # the tail sets the top edge

    filled = np.zeros((h, w), np.uint8)
    cv2.fillPoly(filled, [np.round(poly).astype(np.int32)], 1)
    filled = filled.astype(bool)
    assert filled[mask].all()  # every piece is inside the polygon
    assert (filled & ~mask).sum() <= mask.sum() // 100  # a union, not a hull


def test_mask_to_polygon_leaves_a_single_piece_alone():
    """The splice is reached only when there is something to splice: one contour in,
    the same ring out, so the ordinary frame pays nothing for the split-mask one."""
    mask = np.zeros((100, 200), bool)
    mask[20:60, 40:120] = True
    poly = np.array(mask_to_polygon(mask)).reshape(-1, 2) * [200, 100]
    assert (poly[:, 0].min(), poly[:, 0].max()) == (40, 119)
    assert (poly[:, 1].min(), poly[:, 1].max()) == (20, 59)


# --- dataset storage -------------------------------------------------------- #
def test_dataset_yaml_roundtrip_schema_first(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    ddir = ds.dataset_dir(proj, "dataset_001")
    ddir.mkdir(parents=True)
    ds.write_dataset_yaml(ddir, {"id": "dataset_001", "name": "d", "status": "running"})
    text = (ddir / "dataset.yaml").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == "schema_version: 1"  # Tenet 7
    rec = ds.read_dataset_yaml(ddir)
    assert rec["schema_version"] == 1
    assert rec["id"] == "dataset_001"
    ds.set_dataset_status(ddir, "complete")
    assert ds.read_dataset_yaml(ddir)["status"] == "complete"


def test_set_dataset_status_records_error(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    ddir = ds.dataset_dir(proj, "dataset_001")
    ddir.mkdir(parents=True)
    ds.write_dataset_yaml(ddir, {"id": "dataset_001", "status": "running"})
    ds.set_dataset_status(ddir, "failed", error="ValueError: nope")
    rec = ds.read_dataset_yaml(ddir)
    assert rec["status"] == "failed"
    assert rec["error"] == "ValueError: nope"


def test_discover_and_next_id(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    assert ds.discover_datasets(proj) == []
    assert ds.next_dataset_id(proj) == "dataset_001"
    for did in ("dataset_001", "dataset_002"):
        d = ds.dataset_dir(proj, did)
        d.mkdir(parents=True, exist_ok=True)  # the first was claimed by next_dataset_id
        ds.write_dataset_yaml(d, {"id": did})
    assert ds.discover_datasets(proj) == ["dataset_001", "dataset_002"]
    assert ds.next_dataset_id(proj) == "dataset_003"


def test_data_yaml_is_relative(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    ddir = ds.dataset_dir(proj, "dataset_001")
    ddir.mkdir(parents=True)
    ds.write_data_yaml(ddir, {0: "mouse", 1: "pellet"})
    import yaml

    data = yaml.safe_load((ddir / "data.yaml").read_text(encoding="utf-8-sig"))
    assert data["path"] == "."  # portable (Tenet 6)
    assert data["names"] == {0: "mouse", 1: "pellet"}


def test_delete_dataset_removes_dir_idempotent(tmp_path):
    proj = experiment_store.create_experiment(tmp_path, "p")
    d = ds.dataset_dir(proj, "dataset_001")
    d.mkdir(parents=True)
    ds.write_dataset_yaml(d, {"id": "dataset_001"})
    ds.delete_dataset(proj, "dataset_001")
    assert not d.exists()
    ds.delete_dataset(proj, "dataset_001")  # missing dir is a no-op
