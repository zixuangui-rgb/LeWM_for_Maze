"""Metric heads for world model planning.

Modules:
    distance_head:  Predict BFS shortest-path distance from latent pairs.
    gcrl_head:      Goal-Conditioned RL reachability classifier (binary).
    qrl_head:       Quasimetric RL distance with triangle inequality + contrastive loss.
    validity_head:  Action validity classifier for planner masking.
"""

from hdwm.metric_heads.distance_head import DistanceHead
from hdwm.metric_heads.gcrl_head import GCRLHead
from hdwm.metric_heads.qrl_head import QRLHead
from hdwm.metric_heads.validity_head import ValidityHead

__all__ = ["DistanceHead", "GCRLHead", "QRLHead", "ValidityHead"]
