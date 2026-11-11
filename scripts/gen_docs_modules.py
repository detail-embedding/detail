"""mkdocs-gen-files — one doc page per module (``modules/<id>.md``).

Each page = an optional hand-written intro (``docs/modules/_intros/<id>.md``, prose only) + the
module's parameter reference with per-field anchored headings, all generated from the pydantic
config model (§7). Registered in ``mkdocs.yml`` under ``gen-files.scripts``. The intro snippets
are excluded from being standalone pages (``mkdocs.yml`` ``exclude_docs``).

**A module this build hides gets no page** (§25). Removing its nav line alone would not be enough:
the page would still be generated and published, reachable by URL, documenting a capability the
user cannot reach. `mkdocs.yml`'s Modules nav is static, so the two are kept in step by
``tests/test_docs_modules_nav.py`` rather than by hope.
"""

from pathlib import Path

import mkdocs_gen_files

from detail.configs.describe import render_module_markdown, surfaced_config_models

_INTROS = Path(__file__).resolve().parent.parent / "docs" / "modules" / "_intros"

for module_id, model_cls in surfaced_config_models():
    intro_path = _INTROS / f"{module_id}.md"
    intro = (
        intro_path.read_text(encoding="utf-8-sig").rstrip()
        if intro_path.is_file()
        else ""
    )
    page = render_module_markdown(module_id, model_cls, intro=intro)
    with mkdocs_gen_files.open(f"modules/{module_id}.md", "w") as handle:
        handle.write(page + "\n")
