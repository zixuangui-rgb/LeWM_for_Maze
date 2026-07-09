"""Loss modules shared by HDWM training baselines."""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by LE-WM.

    The input shape follows the reference implementation: ``[T, B, D]``.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2")
        if num_proj <= 0:
            raise ValueError("num_proj must be positive")

        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"expected proj rank 3, got {proj.ndim}")

        proj = proj.float()
        random_projection = torch.randn(
            proj.size(-1),
            self.num_proj,
            device=proj.device,
            dtype=proj.dtype,
        )
        random_projection = random_projection / random_projection.norm(
            p=2, dim=0, keepdim=True
        ).clamp_min(torch.finfo(proj.dtype).eps)

        x_t = (proj @ random_projection).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square()
        err = err + x_t.sin().mean(dim=-3).square()
        statistic = err @ self.weights
        return statistic.mean()


class WassersteinSIGReg(nn.Module):
    """Sliced Wasserstein Gaussian regularizer matching SIGReg input semantics.

    The input shape follows ``SIGReg``: ``[T, B, D]``. Each random 1D projection
    is matched against standard-normal quantiles along the batch dimension with
    squared 2-Wasserstein distance, keeping time steps separate.
    """

    def __init__(self, num_proj: int = 1024) -> None:
        super().__init__()
        if num_proj <= 0:
            raise ValueError("num_proj must be positive")

        self.num_proj = num_proj

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"expected proj rank 3, got {proj.ndim}")

        proj = proj.float()
        batch_size = proj.shape[1]
        if batch_size < 2:
            raise ValueError("Wasserstein SIGReg requires at least two batch samples")

        random_projection = torch.randn(
            proj.size(-1),
            self.num_proj,
            device=proj.device,
            dtype=proj.dtype,
        )
        random_projection = random_projection / random_projection.norm(
            p=2, dim=0, keepdim=True
        ).clamp_min(torch.finfo(proj.dtype).eps)

        projected = (proj @ random_projection).sort(dim=1).values
        quantile_positions = (
            torch.arange(batch_size, device=proj.device, dtype=proj.dtype) + 0.5
        ) / float(batch_size)
        normal_quantiles = torch.erfinv(2.0 * quantile_positions - 1.0)
        normal_quantiles = normal_quantiles * (2.0**0.5)
        return (projected - normal_quantiles.view(1, batch_size, 1)).square().mean()
