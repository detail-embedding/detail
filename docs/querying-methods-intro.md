<!--
Hand-written headline for the generated Querying methods page. The method table and figure below
it are built from code by scripts/gen_docs_querying_methods.py; this file and
querying-methods-controls.md are the only prose. Excluded from the site as standalone pages
(mkdocs.yml `exclude_docs`).
-->

# Querying methods

A behavior query takes one or more example clips and finds the moments of a recording that look
like them. The method determines how each moment is scored, and the methods fall in two families.
**Windowed matchers** return matches about the example's length and are scored by how much a match
overlaps a labeled interval (a span-detection average precision). **Per-frame classifiers**, used
for evaluation, instead score every frame and rank all of them, integrating a threshold-free
per-frame average precision; a match is not tied to the example's length, and retrieval is measured
as a per-frame classification.
