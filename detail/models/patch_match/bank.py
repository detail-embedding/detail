"""The example bank: one row per visible click, and the similarity maps a frame scores against it.

Rows are never averaged (§26 "Patch matching", step 2): a mean blurs varied appearance, and the
systems that average need ~16 examples. A frame's map for a target is the **max** over the
target's rows — a cell scores as high as its single closest example — and the row that won is
kept, so the result can say which example placed each candidate (``extras.ref_index``).

``torch`` is imported inside the functions that need it (Tenet 13); the grids come from a
``PatchFeatures`` backbone and stay on its device until the maps are read back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_at(grid, u: float, v: float, side_px: int):
    """The grid bilinearly sampled at a crop-pixel point ``(u, v)``, renormalised: ``(C,)``."""
    import torch
    import torch.nn.functional as F

    pos = torch.tensor(
        [[[[2.0 * u / side_px - 1.0, 2.0 * v / side_px - 1.0]]]],
        device=grid.device,
        dtype=torch.float32,
    )
    d = F.grid_sample(grid[None].float(), pos, mode="bilinear", align_corners=False)
    return F.normalize(d[0, :, 0, 0], dim=0)


@dataclass
class ExampleBank:
    """Every target's example descriptors in one ``(R, C)`` tensor, each row's target, and
    where each row came from — built once at ``load()``, since a run's examples never change.
    """

    names: tuple[str, ...]  # targets with at least one row, in skeleton order
    bank: object  # torch (R, C) float32
    owner: object  # torch (R,) long, index into ``names``
    sources: tuple[tuple[int, str], ...]  # per row: (example frame index, target)
    #: ``owner`` on the host, kept from before it moved to the device, so the host half
    #: (``maps_from_sims``) never copies from the GPU.
    owner_host: np.ndarray  # (R,) int64

    @classmethod
    def build(
        cls, names: tuple[str, ...], rows: dict[str, list[tuple[int, object]]]
    ) -> "ExampleBank":
        """``rows`` maps a target to its ``(example frame index, descriptor)`` pairs; targets
        with no row are left out, in skeleton order."""
        import torch

        kept = tuple(n for n in names if rows.get(n))
        if not kept:
            raise ValueError(
                "no example marks a keypoint visible; nothing to match against"
            )
        stacked = [torch.stack([d for _, d in rows[n]]).float() for n in kept]
        owner = torch.cat(
            [torch.full((len(r),), i, dtype=torch.long) for i, r in enumerate(stacked)]
        )
        bank = torch.cat(stacked)
        sources = tuple((frame, n) for n in kept for frame, _ in rows[n])
        return cls(
            names=kept,
            bank=bank,
            owner=owner.to(bank.device),
            sources=sources,
            owner_host=owner.numpy(),
        )

    def score_maps(
        self, grid, allowed: np.ndarray | None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """``({target: (g, g) float32}, {target: (g, g) int16})`` — each cell's cosine to the
        target's closest example and that example's bank row; ``-inf`` where forbidden.

        One matmul over every target's rows per frame, one copy to the host: ``sims`` then
        ``maps_from_sims``, the two halves a pipelined caller runs on different threads.
        """
        sims = self.sims(grid).cpu().numpy()
        return self.maps_from_sims(sims, grid.shape[1], grid.shape[2], allowed)

    def sims(self, grid):
        """The device half: every bank row's cosine to every cell, ``(R, g*g)`` on the grid's
        device. No copy to the host."""
        flat = grid.float().reshape(grid.shape[0], -1)  # (C, g*g)
        return self.bank @ flat

    def maps_from_sims(
        self, sims: np.ndarray, g_h: int, g_w: int, allowed: np.ndarray | None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The host half of ``score_maps``, on ``sims`` already copied to numpy."""
        owner = self.owner_host
        maps: dict[str, np.ndarray] = {}
        winners: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.names):
            rows = np.nonzero(owner == i)[0]
            sub = sims[rows]  # (R_i, g*g)
            arg = sub.argmax(axis=0)
            best = (
                sub[arg, np.arange(sub.shape[1])].reshape(g_h, g_w).astype(np.float32)
            )
            won = rows[arg].reshape(g_h, g_w).astype(np.int16)
            if allowed is not None:
                best = np.where(allowed, best, -np.inf).astype(np.float32)
            maps[name] = best
            winners[name] = won
        return maps, winners

    def best_scores(self, grid, allowed: np.ndarray | None) -> dict[str, float]:
        """Per target, the best cosine anywhere allowed (NaN when nothing is)."""
        maps, _ = self.score_maps(grid, allowed)
        return self.best_from_maps(maps)

    @staticmethod
    def best_from_maps(maps: dict[str, np.ndarray]) -> dict[str, float]:
        """``best_scores``' reduction over maps already computed."""
        return {
            name: (float(m.max()) if np.isfinite(m).any() else float("nan"))
            for name, m in maps.items()
        }
