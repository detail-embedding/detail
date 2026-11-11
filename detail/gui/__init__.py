"""Qt GUI for DETAIL.

The only package allowed Qt imports (CLAUDE.md hard rule). It reaches the
backend exclusively through ``Experiment`` / ``Recording`` (Tenet 2). This package
``__init__`` imports no Qt, so machine-local helpers like ``recent`` can be used
(and tested) without PyQt6 installed.
"""
