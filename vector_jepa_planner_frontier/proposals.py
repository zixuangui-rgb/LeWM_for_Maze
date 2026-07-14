"""Auditable candidate generators for pooled-vector planners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from spatial_jepa_planning.common import canonical_json_sha256
from vector_jepa_planner_frontier import ACTION_IDS
from vector_jepa_planner_frontier.common import atomic_torch_save
from vector_jepa_planner_frontier.heads import (
    AutoregressiveProposal,
    DiscreteDenoisingProposal,
)
from vector_jepa_planner_frontier.schemas import ProposalConfig


class CandidateProposal(Protocol):
    def sample(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        count: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


class UniformProposal:
    def sample(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        count: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del source, goal
        if count <= 0 or horizon <= 0:
            raise ValueError("proposal count and horizon must be positive")
        return rng.choice(np.asarray(ACTION_IDS, dtype=np.int64), size=(count, horizon))


@dataclass(frozen=True)
class RetrievalBank:
    source_latents: torch.Tensor
    goal_latents: torch.Tensor
    action_chunks: torch.Tensor
    task_hashes: tuple[str, ...]
    topology_role: str = "train"

    def __post_init__(self) -> None:
        count = int(self.source_latents.shape[0])
        if self.topology_role != "train":
            raise ValueError("retrieval bank may contain train topologies only")
        if (
            self.source_latents.ndim != 2
            or self.goal_latents.shape != self.source_latents.shape
        ):
            raise ValueError("retrieval latent tensors must share shape [count, dim]")
        if self.action_chunks.ndim != 2 or self.action_chunks.shape[0] != count:
            raise ValueError("retrieval action chunks have an invalid shape")
        if len(self.task_hashes) != count:
            raise ValueError("retrieval task-hash provenance is incomplete")
        if len(set(self.task_hashes)) == 0:
            raise ValueError("retrieval bank requires task provenance")
        if bool(((self.action_chunks < 1) | (self.action_chunks > 4)).any()):
            raise ValueError("retrieval bank contains out-of-protocol actions")
        if (
            not torch.isfinite(self.source_latents).all()
            or not torch.isfinite(self.goal_latents).all()
        ):
            raise FloatingPointError("retrieval bank contains non-finite vectors")

    @property
    def fingerprint(self) -> str:
        payload = {
            "topology_role": self.topology_role,
            "task_hashes": list(self.task_hashes),
            "source_shape": list(self.source_latents.shape),
            "goal_shape": list(self.goal_latents.shape),
            "action_shape": list(self.action_chunks.shape),
            "source_sha": _tensor_sha256(self.source_latents),
            "goal_sha": _tensor_sha256(self.goal_latents),
            "action_sha": _tensor_sha256(self.action_chunks),
        }
        return canonical_json_sha256(payload)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        atomic_torch_save(
            output,
            {
                "format_version": 1,
                "topology_role": self.topology_role,
                "source_latents": self.source_latents.detach().cpu(),
                "goal_latents": self.goal_latents.detach().cpu(),
                "action_chunks": self.action_chunks.detach().cpu(),
                "task_hashes": self.task_hashes,
                "fingerprint": self.fingerprint,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> RetrievalBank:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if int(value.get("format_version", -1)) != 1:
            raise ValueError("unsupported retrieval-bank format")
        bank = cls(
            source_latents=value["source_latents"],
            goal_latents=value["goal_latents"],
            action_chunks=value["action_chunks"],
            task_hashes=tuple(value["task_hashes"]),
            topology_role=str(value["topology_role"]),
        )
        if value.get("fingerprint") != bank.fingerprint:
            raise ValueError("retrieval-bank fingerprint mismatch")
        return bank


def _tensor_sha256(value: torch.Tensor) -> str:
    import hashlib

    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class RetrievalProposal:
    def __init__(self, bank: RetrievalBank, *, top_k: int = 32) -> None:
        if top_k <= 0:
            raise ValueError("retrieval top_k must be positive")
        self.bank = bank
        self.top_k = int(top_k)

    def sample(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        count: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if source.shape != goal.shape or source.shape[0] != 1:
            raise ValueError("retrieval expects one source-goal query")
        source_cpu = source.detach().cpu()
        goal_cpu = goal.detach().cpu()
        distances = (self.bank.source_latents - source_cpu).square().mean(dim=-1) + (
            self.bank.goal_latents - goal_cpu
        ).square().mean(dim=-1)
        top_k = min(self.top_k, int(distances.numel()))
        nearest = torch.topk(distances, k=top_k, largest=False).indices.numpy()
        choices = rng.choice(nearest, size=count, replace=count > top_k)
        chunks = self.bank.action_chunks[torch.as_tensor(choices)].numpy()
        if chunks.shape[1] >= horizon:
            return chunks[:, :horizon].astype(np.int64, copy=True)
        padding = rng.choice(
            np.asarray(ACTION_IDS, dtype=np.int64),
            size=(count, horizon - chunks.shape[1]),
        )
        return np.concatenate([chunks, padding], axis=1).astype(np.int64)


class LearnedAutoregressiveSampler:
    def __init__(
        self, model: AutoregressiveProposal, *, temperature: float = 1.0
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("proposal temperature must be positive")
        self.model = model
        self.temperature = float(temperature)

    def sample(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        count: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        source_batch = source.expand(count, -1)
        goal_batch = goal.expand(count, -1)
        prefix = torch.empty((count, 0), dtype=torch.long, device=source.device)
        sampled: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(horizon):
                logits = self.model.next_logits(source_batch, goal_batch, prefix)
                probabilities = torch.softmax(logits / self.temperature, dim=-1)
                probabilities_np = probabilities.detach().cpu().numpy()
                slots = np.asarray(
                    [rng.choice(4, p=row) for row in probabilities_np],
                    dtype=np.int64,
                )
                actions = slots + 1
                sampled.append(actions)
                prefix = torch.cat(
                    [
                        prefix,
                        torch.as_tensor(
                            actions,
                            dtype=torch.long,
                            device=source.device,
                        ).reshape(-1, 1),
                    ],
                    dim=1,
                )
        return np.stack(sampled, axis=1)


class DenoisingSampler:
    def __init__(
        self,
        model: DiscreteDenoisingProposal,
        *,
        steps: int = 8,
        temperature: float = 1.0,
    ) -> None:
        if steps <= 0 or temperature <= 0.0:
            raise ValueError("denoising steps and temperature must be positive")
        self.model = model
        self.steps = int(steps)
        self.temperature = float(temperature)

    def sample(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        count: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if horizon != self.model.config.horizon:
            raise ValueError("denoising model horizon does not match planner horizon")
        source_batch = source.expand(count, -1)
        goal_batch = goal.expand(count, -1)
        tokens = torch.full(
            (count, horizon),
            self.model.mask_slot,
            dtype=torch.long,
            device=source.device,
        )
        order = rng.permutation(horizon)
        groups = np.array_split(order, min(self.steps, horizon))
        with torch.no_grad():
            for group in groups:
                logits = self.model(source_batch, goal_batch, tokens)
                probabilities = torch.softmax(logits / self.temperature, dim=-1)
                for position in group.tolist():
                    rows = probabilities[:, position].detach().cpu().numpy()
                    slots = [rng.choice(4, p=row) for row in rows]
                    tokens[:, position] = torch.as_tensor(
                        slots, dtype=torch.long, device=tokens.device
                    )
        return tokens.detach().cpu().numpy().astype(np.int64) + 1


class MixtureProposal:
    """Fixed-weight uniform/retrieval/learned mixture with exact sample count."""

    def __init__(
        self,
        config: ProposalConfig,
        *,
        retrieval: CandidateProposal | None = None,
        learned: CandidateProposal | None = None,
    ) -> None:
        self.config = config
        self.uniform = UniformProposal()
        self.retrieval = retrieval
        self.learned = learned
        if config.retrieval_weight > 0.0 and retrieval is None:
            raise ValueError(
                "retrieval weight is positive but no retrieval proposal exists"
            )
        if config.learned_weight > 0.0 and learned is None:
            raise ValueError(
                "learned weight is positive but no learned proposal exists"
            )

    def sample(
        self,
        source: torch.Tensor,
        goal: torch.Tensor,
        *,
        count: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        weights = np.asarray(
            [
                self.config.uniform_weight,
                self.config.retrieval_weight,
                self.config.learned_weight,
            ],
            dtype=np.float64,
        )
        raw = weights * count
        counts = np.floor(raw).astype(np.int64)
        remainder = count - int(counts.sum())
        if remainder:
            order = np.argsort(-(raw - counts), kind="stable")
            counts[order[:remainder]] += 1
        proposals: tuple[CandidateProposal | None, ...] = (
            self.uniform,
            self.retrieval,
            self.learned,
        )
        chunks: list[np.ndarray] = []
        for amount, proposal in zip(counts.tolist(), proposals, strict=True):
            if amount:
                if proposal is None:
                    raise RuntimeError("proposal mixture is internally inconsistent")
                chunks.append(
                    proposal.sample(
                        source,
                        goal,
                        count=amount,
                        horizon=horizon,
                        rng=rng,
                    )
                )
        combined = np.concatenate(chunks, axis=0)
        if combined.shape != (count, horizon):
            raise RuntimeError("proposal mixture produced the wrong candidate count")
        return combined[rng.permutation(count)]


__all__ = [
    "CandidateProposal",
    "DenoisingSampler",
    "LearnedAutoregressiveSampler",
    "MixtureProposal",
    "RetrievalBank",
    "RetrievalProposal",
    "UniformProposal",
]
