#!/usr/bin/env python3
"""Shared utilities: model loading and environment creation for eval/probe scripts."""

import json, sys, torch, numpy as np
from pathlib import Path

_HDWM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDWM_ROOT))

from hdwm.config import ProcgenMazeConfig
from hdwm.envs.procgen_maze import ProcgenMazeEnv
from scripts.train.train_dim256 import Unisize256, SizeCondEnc as SizeCondEnc256
from scripts.train.train_canonical_lewm import UnisizeLEWMAux, SizeConditionedEncoder
from scripts.train.train_dim256_pred_aux import Unisize256PredAux


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """Load model from checkpoint, auto-detecting model type.

    Returns:
        (model, ckpt_dict, model_type_str)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config")
    max_size = ckpt.get("max_size", 31)
    aux_type = ckpt.get("aux_type", "")

    # Try Unisize256PredAux (predictor-side aux loss)
    if aux_type == "predictor_only":
        try:
            model = Unisize256PredAux(cfg, max_size=max_size).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            return model, ckpt, "unisize256_pred_aux"
        except Exception:
            pass

    # Try Unisize256 (train_dim256 output, encoder-side aux loss)
    try:
        model = Unisize256(cfg, max_size=max_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt, "unisize256"
    except Exception:
        pass

    # Try UnisizeLEWMAux (train_canonical_lewm output)
    try:
        model = UnisizeLEWMAux(cfg, max_size=max_size).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt, "unisize_lewm_aux"
    except Exception:
        pass

    raise RuntimeError(f"Cannot load model from {ckpt_path}: unknown architecture")


def create_env_from_entry(entry: dict, device=None) -> ProcgenMazeEnv:
    """Create a ProcgenMazeEnv from a manifest entry."""
    sz = entry["maze_size"]
    cfg = ProcgenMazeConfig(
        height=sz, width=sz, observation_channels=5,
        p_noise=0.0, p_noop=0.0, p_action_turn=0.0, p_action_stay=0.0,
        resample_maze_per_sequence=False,
        topology_seed=entry["topology_seed"],
    )
    return ProcgenMazeEnv(cfg, seed=entry.get("level_seed", 42))
