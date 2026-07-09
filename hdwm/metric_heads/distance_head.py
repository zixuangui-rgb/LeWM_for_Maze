"""Distance Head for predicting BFS shortest-path distance from LeWM latents.

Architecture:
    Input:  concat(z_current, z_goal)  → [2 * latent_dim]
    Network: MLP with configurable hidden layers
    Output: scalar predicted BFS distance

Supports both "encoded" (CNN output) and "embedding" (projector output) latents.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceHead(nn.Module):
    """Predict BFS distance from a pair of frozen LeWM latents.

    Args:
        latent_dim: Dimension of the input latent vectors.
        hidden_dims: List of hidden layer dimensions. Default [256, 128].
        dropout: Dropout rate after each hidden layer. Default 0.0.
        input_mode: How to combine the two latents.
            "concat" (default): cat(z1, z2) → MLP
            "diff": abs(z1 - z2) → MLP
            "concat_diff": cat(z1, z2, abs(z1-z2)) → MLP
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        input_mode: str = "concat",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.latent_dim = latent_dim
        self.input_mode = input_mode

        if input_mode == "concat":
            input_dim = latent_dim * 2
        elif input_mode == "diff":
            input_dim = latent_dim
        elif input_mode == "concat_diff":
            input_dim = latent_dim * 3
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def _prepare_input(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """Combine two latent vectors into network input.

        Args:
            z1: [B, D] current latent
            z2: [B, D] goal latent

        Returns:
            [B, input_dim]
        """
        if self.input_mode == "concat":
            return torch.cat([z1, z2], dim=-1)
        elif self.input_mode == "diff":
            return torch.abs(z1 - z2)
        elif self.input_mode == "concat_diff":
            return torch.cat([z1, z2, torch.abs(z1 - z2)], dim=-1)
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Predict BFS distance between two latent states.

        Args:
            z1: [B, D] latent of current state
            z2: [B, D] latent of goal state

        Returns:
            [B] predicted BFS distance (non-negative)
        """
        x = self._prepare_input(z1, z2)
        out = self.net(x).squeeze(-1)  # [B]
        # Ensure non-negative output (distance is always >= 0).
        # Use softplus instead of clamp to avoid a zero-gradient dead zone when
        # all pre-activation outputs are negative.
        return F.softplus(out)
