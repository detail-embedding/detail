"""mkdocs-gen-files script — generate the Parameters reference page from the config models (§7).

Runs at ``mkdocs build`` (wired in ``mkdocs.yml`` under the ``gen-files`` plugin). Writes
``reference/parameters.md`` into the virtual docs tree from a single source: each module's
pydantic config model, via ``detail.configs.describe.render_parameters_markdown``.
"""

import mkdocs_gen_files

from detail.configs.describe import render_parameters_markdown

with mkdocs_gen_files.open("reference/parameters.md", "w", encoding="utf-8") as _f:
    _f.write(render_parameters_markdown())
