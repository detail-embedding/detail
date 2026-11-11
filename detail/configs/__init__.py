"""Typed config models — one pydantic model per module, the single source for each
parameter's type, bounds, default, help, and label (feeds validation + CLI + GUI + docs).

Kept dependency-light (pydantic only, no backends / cv2 / Qt) so the CLI and GUI can import a
module's config model without pulling the runner's heavy imports.
"""
