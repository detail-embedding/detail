"""Patch matching (§26 "Patch matching"): one engine, its decoding private to this package.

::

    PatchFeatures backbone -> ExampleBank -> similarity maps -> point readout (private)
      dinov3_vit               one row per      bank @ grid       peaks -> 3x3 soft-argmax ->
      (any ViT later)          visible click    max per target    per_point | skeleton decode

``view.py`` is the square crop a backbone sees and the map back to frame pixels; ``bank.py``
the example descriptors and the similarity maps; ``decode.py`` the candidates and the two
decodes; ``engine.py`` the ``PatchMatch`` model that strings them together over any
``PatchFeatures`` backbone; ``dinov3_match.py`` the one registered method, ``PatchMatch`` on
``dinov3_vit``. Importing this package registers it (``api/_runner_modules.py`` names the
package).
"""

from detail.models.patch_match import dinov3_match as _registered  # noqa: F401
