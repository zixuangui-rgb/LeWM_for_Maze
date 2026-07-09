"""QRL / Quasimetric Distance Head.

Learns a distance function Q(z_a, z_b) that:
- Approximates BFS distance (regression loss)
- Satisfies quasimetric constraints: Q(a,c) ≤ Q(a,b) + Q(b,c) (triangle inequality)
- Uses contrastive pairs to learn relative distances

Architecture: asymmetric MLP on concat(z1, z2) with separate weights for "forward" direction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QRLHead(nn.Module):
    """Quasimetric distance predictor with contrastive training.

    Uses an asymmetric architecture (different weights for Q(a,b) vs Q(b,a))
    to capture the directed nature of navigation distance.

    Args:
        latent_dim: Dimension of input latent vectors.
        hidden_dims: Hidden layer dimensions.
        temperature: Temperature for contrastive loss.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        temperature: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.temperature = temperature

        # Asymmetric: different projection for z1 (source) and z2 (target)
        self.src_proj = nn.Linear(latent_dim, latent_dim)
        self.tgt_proj = nn.Linear(latent_dim, latent_dim)

        input_dim = latent_dim * 2
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

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Predict quasimetric distance Q(z1 → z2).

        Args:
            z1: [B, D] source latent
            z2: [B, D] target latent

        Returns:
            [B] predicted distance (non-negative)
        """
        # Asymmetric projections
        s = self.src_proj(z1)  # [B, D]
        t = self.tgt_proj(z2)  # [B, D]
        x = torch.cat([s, t], dim=-1)  # [B, 2D]
        out = self.net(x).squeeze(-1)  # [B]
        return F.softplus(out)

    def triangle_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_c: torch.Tensor,
    ) -> torch.Tensor:
        """Soft triangle inequality loss: max(0, Q(a,c) - (Q(a,b) + Q(b,c))).

        Enforces: Q(a,c) ≤ Q(a,b) + Q(b,c)
        """
        q_ac = self.forward(z_a, z_c)
        q_ab = self.forward(z_a, z_b)
        q_bc = self.forward(z_b, z_c)
        violation = F.relu(q_ac - (q_ab + q_bc + 1e-6))
        return violation.mean()

    def contrastive_loss(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor,  # nearby (small BFS)
        z_negative: torch.Tensor,  # far away (large BFS)
    ) -> torch.Tensor:
        """Contrastive loss: Q(anchor, positive) should be smaller than Q(anchor, negative)."""
        q_pos = self.forward(z_anchor, z_positive)
        q_neg = self.forward(z_anchor, z_negative)
        # Hinge: Q(anchor, neg) should be > Q(anchor, pos) + margin
        margin = 1.0
        loss = F.relu(q_pos - q_neg + margin)
        return loss.mean()
