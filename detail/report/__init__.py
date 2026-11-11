"""Module-agnostic reporting: publication-style figures + tables, and the Exporter seam.

This package holds only the *generic* pieces every module's "export figures/tables" capability
reuses: a publication figure/table style (``style``), tidy-data table writers (``tables`` — CSV /
LaTeX booktabs / Markdown), styled matplotlib primitives (``figures``), and the ``Exporter``
protocol + registry (``exporter``). A module's *domain* figures (which chart it draws for its own
results) live with that module, not here.
"""
