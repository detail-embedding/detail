"""mkdocs-gen-files — generate the Input images page from the constructor registry (§7, §14).

The Constructors table (options, descriptions from build-fn docstrings, render steps) comes from
``detail.transforms.constructor_docs.describe_constructors`` — the code is the single source;
the example image in each row is rendered from a committed sample frame through the real
``MaskRenderer``, so adding a constructor makes it appear here with no doc edit. The intro prose is
the hand-written ``docs/input-images-intro.md``.
"""

import json
from pathlib import Path

import cv2
import numpy as np

import mkdocs_gen_files

from detail import surfaces
from detail.api.models import list_models
from detail.core.dataclasses import RenderContext
from detail.core.utils import decode_mask
from detail.transforms import basic as _basic  # noqa: F401 — registers transforms
from detail.transforms.constructor_docs import (
    DOC_BG_COLOR,
    DOC_CANVAS_PX,
    describe_constructors,
)
from detail.transforms.constructors import CONSTRUCTORS
from detail.transforms.renderer import MaskRenderer

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_FIXTURE = _DOCS / "assets" / "input-images"
_GEN = "assets/input-images/generated"  # site path for the rendered example PNGs
_RESIZE = 300

_meta = json.loads((_FIXTURE / "source_meta.json").read_text(encoding="utf-8-sig"))
_frame = cv2.cvtColor(cv2.imread(str(_FIXTURE / "source_frame.jpg")), cv2.COLOR_BGR2RGB)
_mask = decode_mask(_meta["mask_rle"], tuple(_meta["size"]))
_bbox = tuple(_meta["bbox"])


def _render(key: str) -> np.ndarray:
    recipe = CONSTRUCTORS[key]
    build_kwargs = {
        "canvas_px": DOC_CANVAS_PX if recipe.needs_canvas else None,
        "resize": _RESIZE,
    }
    if recipe.needs_bg_color:
        build_kwargs["bg_color"] = DOC_BG_COLOR
    spec = recipe.build(**build_kwargs)
    ctx = RenderContext(
        frame=_frame.copy(),
        mask=_mask.copy(),
        bbox=_bbox,
        image=None,
        frame_idx=_meta["frame_idx"],
        t_ns=0,
        class_name="",
        confidence=0.0,
        object_id=0,
        aux={},
    )
    return MaskRenderer.from_spec(spec).render(ctx)


def _write_png(site_path: str, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"failed to encode {site_path}")
    with mkdocs_gen_files.open(site_path, "wb") as handle:
        handle.write(buf.tobytes())


lines = [
    (_DOCS / "input-images-intro.md").read_text(encoding="utf-8-sig").rstrip(),
    "",
    "## Image size",
    "",
    "Every constructor emits a square RGB image whose side is the embedding model's input size "
    "(or `--resize`, if set):",
    "",
    "| Model | Input size |",
    "| --- | ---: |",
]

_embedding = [
    row
    for row in list_models()
    if row.role == "embedding" and surfaces.is_surfaced(row.id)
]
for row in sorted(_embedding, key=lambda r: (r.input_size or 0, r.id)):
    lines.append(f"| `{row.id}` | {row.input_size} |")

lines += [
    "",
    "## Constructors",
    "",
    "| Constructor | Example |",
    "| --- | ---: |",
]

for cd in describe_constructors(resize=_RESIZE):
    if not surfaces.is_surfaced(cd.key):
        continue
    _write_png(f"{_GEN}/{cd.key}.png", _render(cd.key))
    default = " *(default)*" if cd.is_default else ""
    steps = " → ".join(f"`{step}`" for step in cd.steps)
    mask_note = (
        "uses the subject mask" if cd.uses_mask else "no mask (keeps the background)"
    )
    text = (
        f"**`{cd.key}`**{default} · {cd.label}<br>{cd.description}"
        f"<br>**Steps:** {steps}<br>**Mask:** {mask_note}"
    )
    image = f'![{cd.key} example]({_GEN}/{cd.key}.png){{ width="200" }}'
    lines.append(f"| {text} | {image} |")

with mkdocs_gen_files.open("input-images.md", "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
