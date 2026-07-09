"""Action Validity Head — predict which actions are legal from a latent state.

Input:  z_current (latent embedding)
Output: 5 independent logits P(action_i is valid | z)

Used at CEM sampling time to mask invalid actions from the categorical distribution,
preventing the planner from selecting wall-hitting actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValidityHead(nn.Module):
    """Multi-label classifier: P(action_i is valid | z_current).

    Args:
        latent_dim: Dimension of input latent vector (128 for embedding).
        hidden_dims: Hidden layer dimensions. Default [128].
        num_actions: Number of discrete actions. Default 5.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        num_actions: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]
        self.num_actions = num_actions

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict validity logits for each action.

        Args:
            z: [B, D] latent embedding

        Returns:
            [B, num_actions] logits (use BCEWithLogitsLoss for training)
        """
        return self.net(z)  # [B, 5]

    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        """Return P(valid) ∈ [0,1] for each action."""
        return torch.sigmoid(self.forward(z))

    def predict_mask(self, z: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return boolean mask [B, 5], True = valid."""
        return self.predict_proba(z) > threshold

    def mask_distribution(
        self, z: torch.Tensor, probs: torch.Tensor, min_prob: float = 0.01
    ) -> torch.Tensor:
        """Apply validity masking to a categorical action distribution.

        Args:
            z: [1, D] current latent
            probs: [H, num_actions] CEM categorical distribution at each timestep
            min_prob: minimum probability to assign to masked-out actions

        Returns:
            [H, num_actions] renormalized distribution with invalid actions suppressed
        """
        validity = self.predict_proba(z)  # [1, 5]
        # Broadcast across horizon
        validity = validity.expand(probs.shape[0], -1)  # [H, 5]

        # Suppress invalid actions but don't zero them (keep exploration)
        masked_probs = probs * validity
        # Ensure minimum probability for exploration
        masked_probs = torch.clamp(masked_probs, min=min_prob / self.num_actions)
        # Renormalize
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        return masked_probs
