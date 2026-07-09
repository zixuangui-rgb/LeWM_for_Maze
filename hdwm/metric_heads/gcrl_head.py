"""GCRL Reachability Head — predict whether a goal is reachable within horizon h.

Input:  (z_current, z_goal, horizon_h)
Output: P(reachable within h steps) ∈ [0, 1]

Training labels: positive if BFS_distance <= h, negative otherwise.
Hard negatives: same maze, different branch; dead-end states.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GCRLHead(nn.Module):
    """Binary reachability classifier conditioned on planning horizon.

    Args:
        latent_dim: Dimension of input latent vectors.
        hidden_dims: Hidden layer dimensions.
        num_horizons: Number of horizon buckets.
        horizons: List of horizon values (e.g., [1,2,4,8,16,32,64]).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        horizons: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if horizons is None:
            horizons = [1, 2, 4, 8, 16, 32, 64]
        self.horizons = horizons
        self.num_horizons = len(horizons)

        # Horizon embedding
        self.horizon_embed = nn.Embedding(self.num_horizons, 32)

        # Input: concat(z1, z2) + horizon_embed
        input_dim = latent_dim * 2 + 32

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

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor, horizon_idx: int | torch.Tensor
    ) -> torch.Tensor:
        """Predict reachability probability.

        Args:
            z1: [B, D] current latent
            z2: [B, D] goal latent
            horizon_idx: int (same for all) or [B] tensor of horizon indices

        Returns:
            [B] logits (use BCEWithLogitsLoss for training, sigmoid for probability)
        """
        if isinstance(horizon_idx, int):
            h_idx = torch.full((z1.shape[0],), horizon_idx, dtype=torch.long, device=z1.device)
        else:
            h_idx = horizon_idx.to(z1.device)

        h_emb = self.horizon_embed(h_idx)  # [B, 32]
        x = torch.cat([z1, z2, h_emb], dim=-1)
        return self.net(x).squeeze(-1)  # [B] logits

    def predict_reachable(
        self, z1: torch.Tensor, z2: torch.Tensor, horizon_idx: int
    ) -> torch.Tensor:
        """Return P(reachable) ∈ [0,1]."""
        return torch.sigmoid(self.forward(z1, z2, horizon_idx))

    def get_horizon_idx(self, bfs_distance: int | torch.Tensor) -> int | torch.Tensor:
        """Map a BFS distance to the smallest horizon bucket >= distance."""
        if isinstance(bfs_distance, int):
            for i, h in enumerate(self.horizons):
                if bfs_distance <= h:
                    return i
            return self.num_horizons - 1
        else:
            # Tensor version
            result = torch.full_like(bfs_distance, self.num_horizons - 1, dtype=torch.long)
            for i, h in enumerate(self.horizons):
                result = torch.where(bfs_distance <= h, torch.tensor(i, dtype=torch.long, device=bfs_distance.device), result)
            return result
