"""mkdocs-gen-files — generate the Embedding models page from the model catalog (§7).

Both sections are built from ``api.models`` (the single source), which registers the models and
reads their static class attributes, so a new embedding model appears with no doc edit:
- "Embedding size" — a Model / input-size / embedding-size table (``list_models``).
- "Models" — one block per model (``describe_embedding_models``), including DINOv3's per-variant
  embedding sizes.
The intro prose is the hand-written ``docs/embedding-models-intro.md``.
"""

from pathlib import Path

import mkdocs_gen_files

from detail import surfaces
from detail.api.models import describe_embedding_models

_DOCS = Path(__file__).resolve().parent.parent / "docs"

_models = sorted(
    (m for m in describe_embedding_models() if surfaces.is_surfaced(m.model_id)),
    key=lambda m: (m.architecture, m.input_size or 0, m.model_id),
)

lines = [
    (_DOCS / "embedding-models-intro.md").read_text(encoding="utf-8-sig").rstrip(),
    "",
    "## Embedding size",
    "",
    "| Model | Input size | Embedding size |",
    "| --- | ---: | ---: |",
]

for model in _models:
    if (
        model.variant_sizes
    ):  # one row per size variant (e.g. DINOv3 tiny/small/base/large)
        for variant, dim in model.variant_sizes:
            default = " (default)" if variant == model.default_variant else ""
            lines.append(
                f"| `{model.model_id}` · `{variant}`{default} | {model.input_size} | {dim} |"
            )
    else:
        lines.append(
            f"| `{model.model_id}` | {model.input_size} | {model.output_dim} |"
        )

lines += ["", "## Models", ""]

for model in _models:
    if model.variant_sizes:
        variants = ", ".join(
            f"{v} {d}" + (" (default)" if v == model.default_variant else "")
            for v, d in model.variant_sizes
        )
        embed = f"by variant: {variants}"
    else:
        embed = str(model.output_dim)
    lines += [
        f"### `{model.model_id}`",
        "",
        model.description,
        "",
        f"- **Architecture:** {model.architecture}",
        f"- **Input size:** {model.input_size}",
        f"- **Embedding size:** {embed}",
        f"- **Backends:** {', '.join(model.backends)}",
        f"- **Weights:** {'gated (Hugging Face)' if model.gated else 'public'}",
        "",
    ]

with mkdocs_gen_files.open("embedding-models.md", "w") as handle:
    handle.write("\n".join(lines) + "\n")
