"""Lightning training pipeline for the minimal HDWM experiment."""

from __future__ import annotations

import warnings
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from hdwm.config import (
    BWMConfig,
    BWMLossConfig,
    BWMPriorForm,
    BWMV2Config,
    BWMV2LossConfig,
    EnvConfig,
    GridWorld2DConfig,
    HDWMConfig,
    HDWMLossConfig,
    HWMConfig,
    HWMLossConfig,
    ICWMConfig,
    ICWMLossConfig,
    LEWMCNNConfig,
    LEWMConfig,
    LEWMLossConfig,
    LEWMV2Config,
    LEWMV2LossConfig,
    LEWMV3Config,
    LEWMV3LossConfig,
    LEWMViTConfig,
    LIWMConfig,
    LIWMLossConfig,
    MetricsConfig,
    OptimizerConfig,
    PRISMConfig,
    PRISMLossConfig,
    PRISMV2Config,
    PRISMV2LossConfig,
    RuntimeMonitorConfig,
    WorldModelConfig,
    WorldModelLossConfig,
)
from hdwm.data import (
    RingWorldSequenceDataset,
    SequenceDataConfig,
    move_sequence_batch,
    sequence_batch_collate,
)
from hdwm.envs import (
    GridWorld2DEnv,
    RingWorldConfig,
    RingWorldEnv,
    SequenceBatch,
    make_env,
)
from hdwm.losses import SIGReg, WassersteinSIGReg
from hdwm.models import (
    BWM,
    BWMV2,
    HDWM,
    HWM,
    ICWM,
    LEWM,
    LEWMCNN,
    LEWMV2,
    LEWMV3,
    LEWMViT,
    LIWM,
    PRISM,
    PRISMV2,
    BWMOutput,
    BWMV2Output,
    HDWMOutput,
    HWMOutput,
    ICWMOutput,
    LEWMOutput,
    LEWMV2Output,
    LEWMV3Output,
    LIWMOutput,
    PRISMOutput,
    PRISMV2Output,
)
from hdwm.monitoring import RuntimeMonitor


class RingWorldDataModule(L.LightningDataModule):
    """Lightning data module backed by online environment sampling."""

    def __init__(
        self,
        env_config: EnvConfig,
        data_config: SequenceDataConfig,
        seed: int = 0,
        num_workers: int | None = None,
    ) -> None:
        super().__init__()
        resolved_num_workers = (
            data_config.num_workers if num_workers is None else num_workers
        )
        if resolved_num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.env_config = env_config
        self.data_config = data_config
        self.seed = seed
        self.num_workers = resolved_num_workers

    def train_dataloader(self) -> DataLoader[SequenceBatch]:
        dataset = RingWorldSequenceDataset(
            env_config=self.env_config,
            data_config=self.data_config,
            seed=self.seed,
            split="train",
        )
        # The dataset yields complete SequenceBatch objects, so DataLoader batching is
        # disabled by using batch_size=1 and unwrapping in sequence_batch_collate.
        return DataLoader(
            dataset,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader[SequenceBatch]:
        dataset = RingWorldSequenceDataset(
            env_config=self.env_config,
            data_config=self.data_config,
            seed=self.seed + 10_000,
            max_batches=self.data_config.validation_batches,
            split="validation",
        )
        return DataLoader(
            dataset,
            **self._dataloader_kwargs(),
        )

    def _dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": 1,
            "collate_fn": sequence_batch_collate,
            "num_workers": self.num_workers,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.data_config.persistent_workers
            if self.data_config.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.data_config.prefetch_factor
        return kwargs


class HDWMLightningModule(L.LightningModule):
    """LightningModule wrapping the minimal HDWM network."""

    def __init__(
        self,
        model_config: WorldModelConfig,
        optimizer_config: OptimizerConfig,
        loss_config: WorldModelLossConfig,
        metrics_config: MetricsConfig | None = None,
        runtime_monitor_config: RuntimeMonitorConfig | None = None,
        env_config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = self._build_model(model_config)
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config
        self.metrics_config = (
            metrics_config if metrics_config is not None else MetricsConfig()
        )
        self.runtime_monitor = RuntimeMonitor(
            runtime_monitor_config
            if runtime_monitor_config is not None
            else RuntimeMonitorConfig()
        )
        self._optimizer_base_lrs: list[float] | None = None
        self._position_probe_warning_keys: set[str] = set()
        self.render_env = make_env(
            env_config
            if env_config is not None
            else RingWorldConfig(
                length=model_config.observation_size,
                render_mode="ansi",
            )
        )
        self.sigreg = (
            SIGReg(knots=loss_config.sigreg_knots, num_proj=loss_config.sigreg_num_proj)
            if isinstance(
                loss_config,
                (
                    LEWMLossConfig,
                    ICWMLossConfig,
                    LEWMV2LossConfig,
                    LEWMV3LossConfig,
                    PRISMLossConfig,
                    PRISMV2LossConfig,
                    BWMLossConfig,
                    HWMLossConfig,
                ),
            )
            else None
        )
        self.wasserstein_sigreg = (
            WassersteinSIGReg(num_proj=loss_config.sigreg_num_proj)
            if isinstance(
                loss_config,
                (
                    LEWMLossConfig,
                    ICWMLossConfig,
                    LEWMV2LossConfig,
                    LEWMV3LossConfig,
                    PRISMLossConfig,
                    PRISMV2LossConfig,
                    BWMLossConfig,
                    HWMLossConfig,
                ),
            )
            else None
        )
        self.save_hyperparameters()

    def training_step(self, batch: SequenceBatch, _batch_idx: int) -> torch.Tensor:
        with self.runtime_monitor.time_block("batch_to_device", self.device):
            batch = move_sequence_batch(batch, self.device)
        if isinstance(self.model.config, LIWMConfig):
            return self._liwm_step(batch, stage="train")
        if isinstance(self.model.config, ICWMConfig):
            return self._icwm_step(batch, stage="train")
        if isinstance(self.model.config, LEWMV3Config):
            return self._lewmv3_step(batch, stage="train")
        if isinstance(self.model.config, LEWMV2Config):
            return self._lewmv2_step(batch, stage="train")
        if isinstance(self.model.config, LEWMCNNConfig):
            return self._lewm_step(batch, stage="train")
        if isinstance(self.model.config, LEWMViTConfig):
            return self._lewm_step(batch, stage="train")
        if isinstance(self.model.config, LEWMConfig):
            return self._lewm_step(batch, stage="train")
        if isinstance(self.model.config, PRISMV2Config):
            return self._prismv2_step(batch, stage="train")
        if isinstance(self.model.config, PRISMConfig):
            return self._prism_step(batch, stage="train")
        if isinstance(self.model.config, BWMV2Config):
            return self._bwmv2_step(batch, stage="train")
        if isinstance(self.model.config, BWMConfig):
            return self._bwm_step(batch, stage="train")
        if isinstance(self.model.config, HWMConfig):
            return self._hwm_step(batch, stage="train")
        return self._hdwm_step(batch, stage="train")

    def validation_step(self, batch: SequenceBatch, _batch_idx: int) -> torch.Tensor:
        batch = move_sequence_batch(batch, self.device)
        if isinstance(self.model.config, LIWMConfig):
            return self._liwm_step(batch, stage="val")
        if isinstance(self.model.config, ICWMConfig):
            return self._icwm_step(batch, stage="val")
        if isinstance(self.model.config, LEWMV3Config):
            return self._lewmv3_step(batch, stage="val")
        if isinstance(self.model.config, LEWMV2Config):
            return self._lewmv2_step(batch, stage="val")
        if isinstance(self.model.config, LEWMCNNConfig):
            return self._lewm_step(batch, stage="val")
        if isinstance(self.model.config, LEWMViTConfig):
            return self._lewm_step(batch, stage="val")
        if isinstance(self.model.config, LEWMConfig):
            return self._lewm_step(batch, stage="val")
        if isinstance(self.model.config, PRISMV2Config):
            return self._prismv2_step(batch, stage="val")
        if isinstance(self.model.config, PRISMConfig):
            return self._prism_step(batch, stage="val")
        if isinstance(self.model.config, BWMV2Config):
            return self._bwmv2_step(batch, stage="val")
        if isinstance(self.model.config, BWMConfig):
            return self._bwm_step(batch, stage="val")
        if isinstance(self.model.config, HWMConfig):
            return self._hwm_step(batch, stage="val")
        return self._hdwm_step(batch, stage="val")

    def _build_model(
        self, model_config: WorldModelConfig
    ) -> (
        HDWM
        | LIWM
        | ICWM
        | LEWM
        | LEWMCNN
        | LEWMV2
        | LEWMV3
        | PRISM
        | PRISMV2
        | BWM
        | BWMV2
        | HWM
    ):
        if isinstance(model_config, HDWMConfig):
            return HDWM(model_config)
        if isinstance(model_config, LIWMConfig):
            return LIWM(model_config)
        if isinstance(model_config, ICWMConfig):
            return ICWM(model_config)
        if isinstance(model_config, LEWMV3Config):
            return LEWMV3(model_config)
        if isinstance(model_config, LEWMV2Config):
            return LEWMV2(model_config)
        if isinstance(model_config, LEWMCNNConfig):
            return LEWMCNN(model_config)
        if isinstance(model_config, LEWMViTConfig):
            return LEWMViT(model_config)
        if isinstance(model_config, LEWMConfig):
            return LEWM(model_config)
        if isinstance(model_config, PRISMV2Config):
            return PRISMV2(model_config)
        if isinstance(model_config, PRISMConfig):
            return PRISM(model_config)
        if isinstance(model_config, BWMV2Config):
            return BWMV2(model_config)
        if isinstance(model_config, BWMConfig):
            return BWM(model_config)
        if isinstance(model_config, HWMConfig):
            return HWM(model_config)
        raise TypeError(f"unsupported model config: {type(model_config).__name__}")

    def _liwm_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, LIWMLossConfig):
            raise TypeError(
                f"expected LIWMLossConfig, got {type(self.loss_config).__name__}"
            )
        if not isinstance(self.model, LIWM):
            raise TypeError(f"expected LIWM, got {type(self.model).__name__}")
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for LIWM training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, LIWMOutput):
            raise TypeError(f"expected LIWMOutput, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            if output.prediction.numel() > 0:
                pred_loss = torch.nn.functional.mse_loss(
                    output.prediction,
                    output.target,
                )
                pos_pred_loss = torch.nn.functional.mse_loss(
                    output.pos_prediction,
                    output.pos_target,
                )
                equivariance_loss = self.model.equivariance_loss(
                    output,
                    batch.actions,
                    epsilon=self.loss_config.equivariance_epsilon,
                )
            else:
                pred_loss = output.embedding.new_tensor(0.0)
                pos_pred_loss = output.pos_embedding.new_tensor(0.0)
                equivariance_loss = output.pos_embedding.new_tensor(0.0)
            prediction_loss = pred_loss + pos_pred_loss
            sparse_loss = self.model.generator_group_lasso()
            sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * prediction_loss
                    + self.loss_config.equivariance_weight * equivariance_loss
                    + self.loss_config.sparse_weight * sparse_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/pred_loss", pred_loss)
        self.log(f"{stage}/prediction_loss", prediction_loss)
        self.log(f"{stage}/pos_pred_loss", pos_pred_loss)
        self.log(f"{stage}/equivariance_loss", equivariance_loss)
        self.log(f"{stage}/sparse_loss", sparse_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        if output.prediction.numel() > 0:
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prediction,
                    output.target,
                    dim=-1,
                ).mean(),
            )
            self.log(
                f"{stage}/pos_pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.pos_prediction,
                    output.pos_target,
                    dim=-1,
                ).mean(),
            )
        generator_norms = (
            self.model.effective_generators().flatten(start_dim=1).norm(dim=-1)
        )
        for index, generator_norm in enumerate(generator_norms):
            self.log(f"{stage}/generator_norm/{index}", generator_norm)
        self.log(
            f"{stage}/active_generator_count",
            (generator_norms > 1e-6).float().sum(),
        )
        self.log(f"{stage}/pos_norm", output.pos_embedding.norm(dim=-1).mean())
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
                probe_name="embedding",
            )
            self._position_probe_step(
                embedding=output.pos_embedding,
                states=batch.states,
                stage=stage,
                probe_name="pos",
            )
            self._position_probe_step(
                embedding=output.prediction,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="pred_z",
            )
            self._position_probe_step(
                embedding=output.pos_prediction,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="pred_pos",
            )
        if stage == "train":
            self._log_batch_visibility(batch)
        return loss

    def _hdwm_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, HDWMLossConfig):
            raise TypeError(
                f"expected HDWMLossConfig, got {type(self.loss_config).__name__}"
            )
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(
                batch.observations,
                batch.actions,
                normalize_prior_for_readout=self.loss_config.normalize_prior_for_readout,
            )
        if not isinstance(output, HDWMOutput):
            raise TypeError(f"expected HDWMOutput, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            # train both prior and posterior
            prior_for_align = output.prior
            posterior_for_align = output.posterior
            if self.loss_config.normalize_latents_for_align:
                prior_for_align = torch.nn.functional.normalize(output.prior, dim=-1)
                posterior_for_align = torch.nn.functional.normalize(
                    output.posterior, dim=-1
                )
            align_loss = torch.nn.functional.mse_loss(
                prior_for_align, posterior_for_align
            )
            cmi_logits = None
            cmi_loss = output.posterior.new_tensor(0.0)
            cmi_acc = output.posterior.new_tensor(0.0)
            if self.loss_config.cmi_weight > 0.0:
                cmi_logits = self.model.conditional_mi_logits(
                    prior=output.prior,
                    encoded=output.encoded,
                    posterior=output.posterior,
                    temperature=self.loss_config.cmi_temperature,
                    normalize_prior=self.loss_config.normalize_prior_for_readout,
                    normalize_posterior=self.loss_config.normalize_posterior_for_cmi,
                    normalize_evidence=self.loss_config.normalize_evidence_for_cmi,
                )
                # Pairwise logits are ordered so the positive sample for row t
                # is column t within each batch. cmi_logits shape: [B, T, T].
                batch_size, sequence_length, _ = cmi_logits.shape
                cmi_labels = (
                    torch.arange(sequence_length, device=cmi_logits.device)
                    .unsqueeze(0)
                    .expand(batch_size, sequence_length)
                )
                cmi_loss = torch.nn.functional.cross_entropy(
                    cmi_logits.reshape(-1, sequence_length), cmi_labels.reshape(-1)
                )
                cmi_acc = (cmi_logits.argmax(dim=-1) == cmi_labels).float().mean()
            readout_acc = (
                (output.readout_attention.argmax(dim=-1) == batch.states).float().mean()
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.posterior,
                output.posterior,
            )
            loss = (
                self.loss_config.align_weight * align_loss
                + self.loss_config.cmi_weight * cmi_loss
                + self._weighted_temporal_vicreg_loss(vicreg_loss)
            )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/align_loss", align_loss)
        self.log(f"{stage}/cmi_loss", cmi_loss)
        self.log(f"{stage}/cmi_acc", cmi_acc)
        self.log(f"{stage}/readout_acc", readout_acc)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_frame_embedding_temporal_metrics(
            output.posterior,
            batch.observations,
            stage,
        )
        if stage == "train":
            self._log_hdwm_visibility(
                batch=batch,
                output=output,
                cmi_logits=cmi_logits,
            )
        return loss

    def _lewm_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, LEWMLossConfig):
            raise TypeError(
                f"expected LEWMLossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for LE-WM training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, LEWMOutput):
            raise TypeError(f"expected LEWMOutput, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            pred_loss = (
                torch.nn.functional.mse_loss(output.prediction, output.target)
                if output.prediction.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            (
                sigreg_loss,
                wasserstein_sigreg_loss,
                vicreg_loss,
            ) = self._lewm_sequence_regularizer_losses(
                sigreg_embedding=output.sigreg_embedding,
                temporal_embedding=output.embedding,
                fallback=output.embedding,
                stage=stage,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * pred_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/pred_loss", pred_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        if output.prediction.numel() > 0:
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prediction, output.target, dim=-1
                ).mean(),
            )
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
            )
            self._position_probe_step(
                embedding=output.target,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="encoded_z",
            )
            self._position_probe_step(
                embedding=output.prediction,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="pred_z",
            )
        if stage == "train":
            self._log_lewm_visibility(batch=batch, output=output)
        return loss

    def _icwm_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, ICWMLossConfig):
            raise TypeError(
                f"expected ICWMLossConfig, got {type(self.loss_config).__name__}"
            )
        if not isinstance(self.model, ICWM):
            raise TypeError(f"expected ICWM, got {type(self.model).__name__}")
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for ICWM training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, ICWMOutput):
            raise TypeError(f"expected ICWMOutput, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            pred_loss = self._masked_prediction_mse(
                output.prediction,
                output.target,
                output.valid_prediction_mask,
            )
            context_batch_size, context_length, sequence_length = (
                batch.observations.shape[:3]
            )
            trajectory_sigreg_embedding = self._icwm_raw_sequence_embedding(
                output.sigreg_embedding,
                context_batch_size=context_batch_size,
                context_length=context_length,
                sequence_length=sequence_length,
            )
            trajectory_embedding = self._icwm_raw_sequence_embedding(
                output.observation_embedding,
                context_batch_size=context_batch_size,
                context_length=context_length,
                sequence_length=sequence_length,
            )
            (
                sigreg_loss,
                wasserstein_sigreg_loss,
                vicreg_loss,
            ) = self._lewm_sequence_regularizer_losses(
                sigreg_embedding=trajectory_sigreg_embedding,
                temporal_embedding=trajectory_embedding,
                fallback=output.observation_embedding,
                stage=stage,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * pred_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/pred_loss", pred_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self.log(
            f"{stage}/valid_transition_count",
            output.valid_prediction_mask.sum().to(dtype=output.prediction.dtype),
        )
        self.log(
            f"{stage}/packed_sequence_length",
            float(output.packed_embedding.shape[1]),
        )
        self.log(f"{stage}/context_length", float(self.model.config.context_length))
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.observation_embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            trajectory_embedding,
            batch.observations.reshape(
                context_batch_size * context_length,
                sequence_length,
                *batch.observations.shape[3:],
            ),
            stage,
        )
        if output.valid_prediction_mask.any():
            selected_prediction = output.prediction[output.valid_prediction_mask]
            selected_target = output.target[output.valid_prediction_mask]
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    selected_prediction,
                    selected_target,
                    dim=-1,
                ).mean(),
            )
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.observation_embedding,
                states=batch.states.reshape(
                    batch.states.shape[0],
                    -1,
                ),
                stage=stage,
            )
            if output.valid_prediction_mask.any():
                pred_embedding, pred_states = self._icwm_prediction_probe_inputs(
                    prediction=output.prediction,
                    states=batch.states,
                    valid_prediction_mask=output.valid_prediction_mask,
                )
                self._position_probe_step(
                    embedding=pred_embedding,
                    states=pred_states,
                    stage=stage,
                    probe_name="pred_z",
                )
        if stage == "train":
            self.log(
                "train/embedding_norm",
                output.observation_embedding.norm(dim=-1).mean(),
            )
            self.log(
                "train/packed_embedding_norm",
                output.packed_embedding.norm(dim=-1).mean(),
            )
            self._log_batch_visibility(batch)
            if self.trainer.optimizers:
                self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
            if self.metrics_config.log_layer_weight_stats:
                self._log_layer_weight_stats()
        return loss

    @staticmethod
    def _masked_prediction_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"expected target shape {tuple(prediction.shape)}, "
                f"got {tuple(target.shape)}"
            )
        if mask.shape != prediction.shape[:2]:
            raise ValueError(
                f"expected mask shape {tuple(prediction.shape[:2])}, "
                f"got {tuple(mask.shape)}"
            )
        if not mask.any():
            return prediction.new_tensor(0.0)
        per_token_error = (prediction - target).square().mean(dim=-1)
        return per_token_error[mask].mean()

    @staticmethod
    def _icwm_prediction_probe_inputs(
        prediction: torch.Tensor,
        states: torch.Tensor,
        valid_prediction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.ndim != 3:
            raise ValueError(f"expected prediction rank 3, got {prediction.ndim}")
        if states.ndim != 3:
            raise ValueError(f"expected states rank 3, got {states.ndim}")
        if valid_prediction_mask.shape != prediction.shape[:2]:
            raise ValueError(
                f"expected valid_prediction_mask shape {tuple(prediction.shape[:2])}, "
                f"got {tuple(valid_prediction_mask.shape)}"
            )

        batch_size, context_length, sequence_length = states.shape
        if prediction.shape[0] != batch_size:
            raise ValueError(
                f"expected prediction batch size {batch_size}, "
                f"got {prediction.shape[0]}"
            )
        packed_length = context_length * sequence_length + context_length + 1
        if prediction.shape[1] != packed_length - 1:
            raise ValueError(
                f"expected prediction length {packed_length - 1}, "
                f"got {prediction.shape[1]}"
            )

        packed_target_states = states.new_full(valid_prediction_mask.shape, -1)
        write_index = 1
        for context_index in range(context_length):
            if sequence_length > 1:
                packed_target_states[
                    :,
                    write_index : write_index + sequence_length - 1,
                ] = states[:, context_index, 1:]
            write_index += sequence_length
            if context_index < context_length - 1:
                write_index += 1

        selected_states = packed_target_states[valid_prediction_mask]
        if selected_states.numel() == 0:
            raise ValueError("ICWM prediction probe requires valid transitions")
        if (selected_states < 0).any():
            raise ValueError("ICWM prediction probe received invalid packed positions")

        selected_prediction = prediction[valid_prediction_mask]
        return selected_prediction.unsqueeze(0), selected_states.unsqueeze(0)

    @staticmethod
    def _weighted_mse_over_tensors(
        predictions: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if len(predictions) != len(targets):
            raise ValueError(
                f"expected equal prediction/target counts, got "
                f"{len(predictions)} and {len(targets)}"
            )

        total_error = fallback.new_tensor(0.0)
        total_count = 0
        for prediction, target in zip(predictions, targets, strict=True):
            if prediction.shape != target.shape:
                raise ValueError(
                    f"expected target shape {tuple(prediction.shape)}, "
                    f"got {tuple(target.shape)}"
                )
            if prediction.numel() == 0:
                continue
            total_error = total_error + (prediction - target).square().sum()
            total_count += prediction.numel()
        if total_count == 0:
            return fallback.new_tensor(0.0)
        return total_error / float(total_count)

    @staticmethod
    def _sample_rows(
        values: torch.Tensor,
        max_samples: int | None,
    ) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"expected rank-2 values, got {values.ndim}")
        if max_samples is None or values.shape[0] <= max_samples:
            return values
        indices = torch.linspace(
            0,
            values.shape[0] - 1,
            steps=max_samples,
            device=values.device,
        ).round()
        return values.index_select(0, indices.long())

    @classmethod
    def _concept_cauchy_loss(
        cls,
        concepts: torch.Tensor,
        tau: float | None,
        max_samples: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if concepts.ndim != 3:
            raise ValueError(f"expected concepts rank 3, got {concepts.ndim}")
        flat_concepts = concepts.reshape(-1, concepts.shape[-1])
        flat_concepts = cls._sample_rows(flat_concepts, max_samples)
        if flat_concepts.shape[0] < 2:
            zero = concepts.new_tensor(0.0)
            fallback_tau = concepts.new_tensor(1.0 if tau is None else tau)
            return zero, fallback_tau

        similarity = flat_concepts @ flat_concepts.transpose(0, 1)
        distance = (1.0 - similarity).clamp_min(0.0)
        pair_mask = torch.ones_like(distance, dtype=torch.bool).triu(diagonal=1)
        pair_distances = distance[pair_mask]
        if pair_distances.numel() == 0:
            zero = concepts.new_tensor(0.0)
            fallback_tau = concepts.new_tensor(1.0 if tau is None else tau)
            return zero, fallback_tau

        if tau is None:
            tau_tensor = pair_distances.detach().median().clamp_min(1e-6)
        else:
            tau_tensor = concepts.new_tensor(tau)
        return -(1.0 / (1.0 + pair_distances / tau_tensor)).mean(), tau_tensor

    @staticmethod
    def _rank_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError(
                f"expected matching rank inputs, got {x.shape} and {y.shape}"
            )
        if x.numel() < 2:
            return x.new_tensor(0.0)
        x_rank = torch.argsort(torch.argsort(x)).to(dtype=x.dtype)
        y_rank = torch.argsort(torch.argsort(y)).to(dtype=y.dtype)
        x_centered = x_rank - x_rank.mean()
        y_centered = y_rank - y_rank.mean()
        denom = x_centered.norm() * y_centered.norm()
        if denom <= 0:
            return x.new_tensor(0.0)
        return (x_centered * y_centered).sum() / denom

    @staticmethod
    def _consecutive_frame_embedding_cosine_metrics(
        embeddings: torch.Tensor,
        observations: torch.Tensor,
        exclude_identical_pairs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError(f"expected embeddings rank 3, got {embeddings.ndim}")
        if observations.ndim < 3:
            raise ValueError(
                f"expected observations rank at least 3, got {observations.ndim}"
            )
        if embeddings.shape[:2] != observations.shape[:2]:
            raise ValueError(
                "expected embeddings and observations to share batch and sequence "
                "dims, "
                f"got {embeddings.shape[:2]} and {observations.shape[:2]}"
            )

        zero = embeddings.new_tensor(0.0)
        if embeddings.shape[1] <= 1:
            return zero, zero, zero, zero

        flattened_observations = observations.reshape(*observations.shape[:2], -1)
        current_observations = flattened_observations[:, :-1]
        next_observations = flattened_observations[:, 1:]
        identical_pairs = (current_observations == next_observations).all(dim=-1)
        valid_pairs = (
            ~identical_pairs
            if exclude_identical_pairs
            else torch.ones_like(
                identical_pairs,
                dtype=torch.bool,
            )
        )
        pair_count = valid_pairs.sum().to(dtype=embeddings.dtype)
        identical_count = identical_pairs.sum().to(dtype=embeddings.dtype)
        if not valid_pairs.any():
            return zero, zero, pair_count, identical_count

        cosine = torch.nn.functional.cosine_similarity(
            embeddings[:, :-1],
            embeddings[:, 1:],
            dim=-1,
        )
        selected_cosine = cosine[valid_pairs]
        return (
            selected_cosine.mean(),
            selected_cosine.min(),
            pair_count,
            identical_count,
        )

    @staticmethod
    def _frame_temporal_pairwise_cosine_mean(
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean pairwise cosine similarity among time steps in each sequence."""
        if embeddings.ndim != 3:
            raise ValueError(f"expected embeddings rank 3, got {embeddings.ndim}")
        if embeddings.shape[1] < 2:
            return embeddings.new_tensor(0.0)

        normalized = torch.nn.functional.normalize(embeddings, dim=-1)
        seq_len = normalized.shape[1]

        # Sample time steps if sequence is very long to avoid O(seq^2) memory.
        max_seq_len = 256
        if seq_len > max_seq_len:
            indices = torch.randperm(seq_len, device=normalized.device)[:max_seq_len]
            normalized = normalized[:, indices, :]
            seq_len = max_seq_len

        # (batch, seq, dim) @ (batch, dim, seq) -> (batch, seq, seq)
        sim = torch.bmm(normalized, normalized.transpose(1, 2))
        mask = ~torch.eye(seq_len, dtype=torch.bool, device=sim.device)
        return sim.masked_select(mask.unsqueeze(0).expand(sim.shape[0], -1, -1)).mean()

    def _log_frame_embedding_temporal_metrics(
        self,
        embeddings: torch.Tensor,
        observations: torch.Tensor,
        stage: str,
    ) -> None:
        exclude_identical_pairs = getattr(
            self.loss_config,
            "exclude_identical_frame_pairs_from_cosine_monitor",
            True,
        )
        with torch.no_grad():
            (
                cosine_mean,
                cosine_min,
                pair_count,
                identical_count,
            ) = self._consecutive_frame_embedding_cosine_metrics(
                embeddings,
                observations,
                exclude_identical_pairs,
            )
            temporal_pairwise_cosine_mean = self._frame_temporal_pairwise_cosine_mean(
                embeddings
            )

        self.log(f"{stage}/frame_temporal_cosine_mean", cosine_mean)
        self.log(f"{stage}/frame_temporal_cosine_min", cosine_min)
        self.log(f"{stage}/frame_temporal_cosine_pair_count", pair_count)
        self.log(f"{stage}/frame_temporal_identical_pair_count", identical_count)
        self.log(
            f"{stage}/frame_temporal_pairwise_cosine_mean",
            temporal_pairwise_cosine_mean,
        )

    def _lewm_sequence_regularizer_losses(
        self,
        *,
        sigreg_embedding: torch.Tensor,
        temporal_embedding: torch.Tensor,
        fallback: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigreg_loss = (
            self._sigreg_loss(sigreg_embedding, stage)
            if sigreg_embedding.numel() > 0
            else fallback.new_tensor(0.0)
        )
        wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
            sigreg_embedding,
            stage,
            fallback,
        )
        vicreg_loss = self._maybe_temporal_vicreg_loss(
            temporal_embedding,
            fallback,
        )
        return sigreg_loss, wasserstein_sigreg_loss, vicreg_loss

    @staticmethod
    def _icwm_raw_sequence_embedding(
        embedding: torch.Tensor,
        *,
        context_batch_size: int,
        context_length: int,
        sequence_length: int,
    ) -> torch.Tensor:
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        if context_batch_size <= 0:
            raise ValueError("context_batch_size must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        expected_shape = (context_batch_size, context_length * sequence_length)
        if embedding.shape[:2] != expected_shape:
            raise ValueError(
                f"expected ICWM embedding shape prefix {expected_shape}, "
                f"got {tuple(embedding.shape[:2])}"
            )

        return embedding.reshape(
            context_batch_size,
            context_length,
            sequence_length,
            embedding.shape[-1],
        ).reshape(
            context_batch_size * context_length,
            sequence_length,
            embedding.shape[-1],
        )

    def _maybe_temporal_vicreg_loss(
        self,
        embeddings: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(getattr(self.loss_config, "vicreg_enabled", False)):
            return fallback.new_tensor(0.0)
        return self._temporal_vicreg_loss(embeddings)

    def _temporal_vicreg_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"expected embeddings rank 3, got {embeddings.ndim}")
        if embeddings.shape[-1] <= 0:
            raise ValueError("embedding dim must be positive")
        if embeddings.shape[1] <= 1:
            return embeddings.new_tensor(0.0)

        variance_target = float(
            getattr(self.loss_config, "vicreg_variance_target", 1.0)
        )
        epsilon = float(getattr(self.loss_config, "vicreg_epsilon", 1e-4))
        std = embeddings.float().var(dim=1, unbiased=True).add(epsilon).sqrt()
        return torch.relu(variance_target - std).mean()

    def _weighted_temporal_vicreg_loss(
        self,
        temporal_vicreg_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            float(getattr(self.loss_config, "vicreg_weight", 0.0))
            * temporal_vicreg_loss
        )

    def _log_temporal_vicreg(
        self,
        stage: str,
        temporal_vicreg_loss: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.loss_config, "vicreg_enabled", False)):
            return
        self.log(f"{stage}/vicreg_loss", temporal_vicreg_loss)
        self.log(
            f"{stage}/vicreg_weighted_loss",
            self._weighted_temporal_vicreg_loss(temporal_vicreg_loss),
        )

    def _log_lewmv3_concept_metrics(
        self,
        output: LEWMV3Output,
        stage: str,
        loss_config: LEWMV3LossConfig,
    ) -> None:
        if int(self.global_step) % loss_config.concept_monitor_every_n_steps != 0:
            return
        with torch.no_grad():
            concepts = output.concepts.reshape(-1, output.concepts.shape[-1])
            dynamics = (
                output.rotation_angles
                if output.rotation_angles.shape[-1] > 0
                else output.transitions
            )
            dynamics = dynamics.reshape(-1, dynamics.shape[-1])
            concepts = self._sample_rows(concepts, loss_config.concept_pair_max_samples)
            dynamics = self._sample_rows(
                dynamics,
                loss_config.concept_pair_max_samples,
            )
            if concepts.shape[0] < 2:
                return

            similarity = concepts @ concepts.transpose(0, 1)
            distance = (1.0 - similarity).clamp_min(0.0)
            pair_mask = torch.ones_like(distance, dtype=torch.bool).triu(diagonal=1)
            pair_distances = distance[pair_mask]
            distance_mean = pair_distances.mean()
            distance_std = pair_distances.std(unbiased=False)
            centered = pair_distances - distance_mean
            safe_std = distance_std.clamp_min(1e-6)
            skewness = (centered / safe_std).pow(3).mean()
            kurtosis = (centered / safe_std).pow(4).mean().clamp_min(1e-6)
            bimodality = (skewness.square() + 1.0) / kurtosis

            eigenvalues = torch.linalg.eigvalsh(similarity).flip(0)
            if eigenvalues.numel() > 1:
                gaps = eigenvalues[:-1] - eigenvalues[1:]
                effective_clusters = torch.argmax(gaps).to(dtype=concepts.dtype) + 1.0
            else:
                effective_clusters = concepts.new_tensor(1.0)

            dynamics_similarity = -torch.cdist(dynamics, dynamics)[pair_mask]
            dynamics_alignment = self._rank_correlation(
                similarity[pair_mask],
                dynamics_similarity,
            )

            if output.concepts.shape[1] > 1:
                temporal_similarity = (
                    output.concepts[:, :-1] * output.concepts[:, 1:]
                ).sum(dim=-1)
                temporal_mean = temporal_similarity.mean()
                temporal_min = temporal_similarity.min()
                temporal_jumps = (
                    temporal_similarity < loss_config.concept_jump_similarity_threshold
                ).sum()
            else:
                temporal_mean = output.concepts.new_tensor(0.0)
                temporal_min = output.concepts.new_tensor(0.0)
                temporal_jumps = output.concepts.new_tensor(0.0)

        self.log(f"{stage}/concept_distance_mean", distance_mean)
        self.log(f"{stage}/concept_distance_std", distance_std)
        self.log(f"{stage}/concept_distance_min", pair_distances.min())
        self.log(f"{stage}/concept_distance_max", pair_distances.max())
        self.log(f"{stage}/concept_bimodality_coefficient", bimodality)
        self.log(f"{stage}/concept_effective_clusters", effective_clusters)
        self.log(f"{stage}/concept_dynamics_spearman", dynamics_alignment)
        self.log(f"{stage}/concept_temporal_similarity_mean", temporal_mean)
        self.log(f"{stage}/concept_temporal_similarity_min", temporal_min)
        self.log(
            f"{stage}/concept_temporal_jumps",
            temporal_jumps.to(dtype=concepts.dtype),
        )

    def _lewmv3_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, LEWMV3LossConfig):
            raise TypeError(
                f"expected LEWMV3LossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for LE-WMv3 training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(
                batch.observations,
                batch.actions,
                prediction_steps=self.loss_config.prediction_steps,
            )
        if not isinstance(output, LEWMV3Output):
            raise TypeError(f"expected LEWMV3Output, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            rollout_hidden_targets = output.rollout_hidden_targets
            if self.loss_config.detach_rollout_hidden_targets:
                rollout_hidden_targets = tuple(
                    target.detach() for target in rollout_hidden_targets
                )
            dynamics_loss = self._weighted_mse_over_tensors(
                output.rollout_hidden_predictions,
                rollout_hidden_targets,
                output.embedding,
            )
            predicted_obs_loss = self._weighted_mse_over_tensors(
                output.rollout_decoded_predictions,
                output.rollout_obs_targets,
                output.embedding,
            )
            computed_obs_loss = torch.nn.functional.mse_loss(
                output.decoded_hidden,
                output.obs_target,
            )
            obs_loss = (
                self.loss_config.predicted_obs_weight * predicted_obs_loss
                + self.loss_config.computed_obs_weight * computed_obs_loss
            )
            cauchy_loss, cauchy_tau = self._concept_cauchy_loss(
                output.concepts,
                self.loss_config.cauchy_tau,
                self.loss_config.concept_pair_max_samples,
            )
            sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.dynamics_weight * dynamics_loss
                    + self.loss_config.obs_weight * obs_loss
                    + self.loss_config.cauchy_weight * cauchy_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/dynamics_loss", dynamics_loss)
        self.log(f"{stage}/pred_loss", predicted_obs_loss)
        self.log(f"{stage}/obs_loss", obs_loss)
        self.log(f"{stage}/predicted_obs_loss", predicted_obs_loss)
        self.log(f"{stage}/computed_obs_loss", computed_obs_loss)
        self.log(f"{stage}/cauchy_loss", cauchy_loss)
        self.log(f"{stage}/cauchy_tau", cauchy_tau)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        self.log(f"{stage}/h_dim", float(output.hidden.shape[-1]))
        self.log(f"{stage}/z_dim", float(output.embedding.shape[-1]))
        self.log(f"{stage}/concept_dim", float(output.concepts.shape[-1]))
        self.log(f"{stage}/prediction_steps", float(self.loss_config.prediction_steps))
        self.log(
            f"{stage}/detach_rollout_hidden_targets",
            float(self.loss_config.detach_rollout_hidden_targets),
        )
        self.log(
            f"{stage}/effective_prediction_steps",
            float(len(output.rollout_hidden_predictions)),
        )
        for horizon, (hidden_prediction, hidden_target) in enumerate(
            zip(
                output.rollout_hidden_predictions,
                rollout_hidden_targets,
                strict=True,
            ),
            start=1,
        ):
            if hidden_prediction.numel() > 0:
                self.log(
                    f"{stage}/dynamics_loss_h{horizon}",
                    torch.nn.functional.mse_loss(hidden_prediction, hidden_target),
                )
        for horizon, (decoded_prediction, obs_target) in enumerate(
            zip(
                output.rollout_decoded_predictions,
                output.rollout_obs_targets,
                strict=True,
            ),
            start=1,
        ):
            if decoded_prediction.numel() > 0:
                self.log(
                    f"{stage}/predicted_obs_loss_h{horizon}",
                    torch.nn.functional.mse_loss(decoded_prediction, obs_target),
                )
        if output.hidden_prediction.numel() > 0:
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.hidden_prediction,
                    output.hidden_target,
                    dim=-1,
                ).mean(),
            )
        self.log(
            f"{stage}/computed_obs_pred_target_cosine",
            torch.nn.functional.cosine_similarity(
                output.decoded_hidden,
                output.obs_target,
                dim=-1,
            ).mean(),
        )
        if output.decoded_prediction.numel() > 0:
            self.log(
                f"{stage}/predicted_obs_pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.decoded_prediction,
                    output.obs_target[:, 1:],
                    dim=-1,
                ).mean(),
            )
        self._log_lewmv3_concept_metrics(output, stage, self.loss_config)
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
                probe_name="encoded_z",
            )
            self._position_probe_step(
                embedding=output.hidden,
                states=batch.states,
                stage=stage,
                probe_name="computed_h",
            )
            self._position_probe_step(
                embedding=output.decoded_hidden,
                states=batch.states,
                stage=stage,
                probe_name="decoded_z",
            )
            if output.hidden_prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.hidden_prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_h",
                )
                self._position_probe_step(
                    embedding=output.decoded_prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_z",
                )
        if stage == "train":
            self._log_lewm_visibility(batch=batch, output=output)
        return loss

    def _lewmv2_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, LEWMV2LossConfig):
            raise TypeError(
                f"expected LEWMV2LossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for LE-WMv2 training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(
                batch.observations,
                batch.actions,
                prediction_steps=self.loss_config.prediction_steps,
            )
        if not isinstance(output, LEWMV2Output):
            raise TypeError(f"expected LEWMV2Output, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            rollout_hidden_targets = output.rollout_hidden_targets
            if self.loss_config.detach_rollout_hidden_targets:
                rollout_hidden_targets = tuple(
                    target.detach() for target in rollout_hidden_targets
                )
            dynamics_loss = self._weighted_mse_over_tensors(
                output.rollout_hidden_predictions,
                rollout_hidden_targets,
                output.embedding,
            )
            predicted_obs_loss = self._weighted_mse_over_tensors(
                output.rollout_decoded_predictions,
                output.rollout_obs_targets,
                output.embedding,
            )
            computed_obs_loss = torch.nn.functional.mse_loss(
                output.decoded_hidden,
                output.obs_target,
            )
            obs_loss = (
                self.loss_config.predicted_obs_weight * predicted_obs_loss
                + self.loss_config.computed_obs_weight * computed_obs_loss
            )
            sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.dynamics_weight * dynamics_loss
                    + self.loss_config.obs_weight * obs_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/dynamics_loss", dynamics_loss)
        self.log(f"{stage}/pred_loss", predicted_obs_loss)
        self.log(f"{stage}/obs_loss", obs_loss)
        self.log(f"{stage}/predicted_obs_loss", predicted_obs_loss)
        self.log(f"{stage}/computed_obs_loss", computed_obs_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        self.log(f"{stage}/h_dim", float(output.hidden.shape[-1]))
        self.log(f"{stage}/z_dim", float(output.embedding.shape[-1]))
        self.log(f"{stage}/prediction_steps", float(self.loss_config.prediction_steps))
        self.log(
            f"{stage}/detach_rollout_hidden_targets",
            float(self.loss_config.detach_rollout_hidden_targets),
        )
        self.log(
            f"{stage}/effective_prediction_steps",
            float(len(output.rollout_hidden_predictions)),
        )
        for horizon, (hidden_prediction, hidden_target) in enumerate(
            zip(
                output.rollout_hidden_predictions,
                rollout_hidden_targets,
                strict=True,
            ),
            start=1,
        ):
            if hidden_prediction.numel() > 0:
                self.log(
                    f"{stage}/dynamics_loss_h{horizon}",
                    torch.nn.functional.mse_loss(hidden_prediction, hidden_target),
                )
        for horizon, (decoded_prediction, obs_target) in enumerate(
            zip(
                output.rollout_decoded_predictions,
                output.rollout_obs_targets,
                strict=True,
            ),
            start=1,
        ):
            if decoded_prediction.numel() > 0:
                self.log(
                    f"{stage}/predicted_obs_loss_h{horizon}",
                    torch.nn.functional.mse_loss(decoded_prediction, obs_target),
                )
        if output.hidden_prediction.numel() > 0:
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.hidden_prediction,
                    output.hidden_target,
                    dim=-1,
                ).mean(),
            )
        self.log(
            f"{stage}/computed_obs_pred_target_cosine",
            torch.nn.functional.cosine_similarity(
                output.decoded_hidden,
                output.obs_target,
                dim=-1,
            ).mean(),
        )
        if output.decoded_prediction.numel() > 0:
            self.log(
                f"{stage}/predicted_obs_pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.decoded_prediction,
                    output.obs_target[:, 1:],
                    dim=-1,
                ).mean(),
            )
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
                probe_name="encoded_z",
            )
            self._position_probe_step(
                embedding=output.hidden,
                states=batch.states,
                stage=stage,
                probe_name="computed_h",
            )
            self._position_probe_step(
                embedding=output.decoded_hidden,
                states=batch.states,
                stage=stage,
                probe_name="decoded_z",
            )
            if output.hidden_prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.hidden_prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_h",
                )
                self._position_probe_step(
                    embedding=output.decoded_prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_z",
                )
        if stage == "train":
            self._log_lewm_visibility(batch=batch, output=output)
        return loss

    def _prism_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, PRISMLossConfig):
            raise TypeError(
                f"expected PRISMLossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for PRISM training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, PRISMOutput):
            raise TypeError(f"expected PRISMOutput, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            pred_loss = (
                torch.nn.functional.mse_loss(output.prediction, output.target)
                if output.prediction.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            clean_pred_loss = (
                torch.nn.functional.mse_loss(output.clean_prediction, output.target)
                if output.clean_prediction.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * (pred_loss + clean_pred_loss)
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/pred_loss", pred_loss)
        self.log(f"{stage}/clean_pred_loss", clean_pred_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        if output.prediction.numel() > 0:
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prediction, output.target, dim=-1
                ).mean(),
            )
        if output.clean_prediction.numel() > 0:
            self.log(
                f"{stage}/clean_pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.clean_prediction, output.target, dim=-1
                ).mean(),
            )
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.clean_state,
                states=batch.states,
                stage=stage,
                probe_name="clean_z",
            )
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
                probe_name="encoded_z",
            )
            if output.prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_z",
                )
            if output.clean_prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.clean_prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="clean_pred_z",
                )
        if stage == "train":
            self._log_prism_visibility(batch=batch, output=output)
        return loss

    def _prismv2_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, PRISMV2LossConfig):
            raise TypeError(
                f"expected PRISMV2LossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for PRISMv2 training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, PRISMV2Output):
            raise TypeError(f"expected PRISMV2Output, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            belief_loss = (
                torch.nn.functional.mse_loss(
                    output.prior_belief,
                    output.posterior_belief[:, 1:],
                )
                if output.prior_belief.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            embedding_sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            posterior_belief_sigreg_enabled = (
                self.loss_config.effective_posterior_belief_sigreg_enabled
            )
            posterior_belief_sigreg_loss = output.embedding.new_tensor(0.0)
            if (
                posterior_belief_sigreg_enabled
                and output.posterior_belief_sigreg.numel() > 0
            ):
                posterior_belief_sigreg_loss = self._sigreg_loss(
                    output.posterior_belief_sigreg,
                    stage,
                )
            sigreg_loss = (
                self.loss_config.z_sigreg_weight * embedding_sigreg_loss
                + self.loss_config.posterior_belief_sigreg_weight
                * posterior_belief_sigreg_loss
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            posterior_obs_loss = torch.nn.functional.mse_loss(
                output.posterior_obs_prediction,
                output.obs_target,
            )
            prior_obs_loss = (
                torch.nn.functional.mse_loss(
                    output.prior_obs_prediction,
                    output.obs_target[:, 1:],
                )
                if output.prior_obs_prediction.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            obs_loss = (
                self.loss_config.posterior_obs_weight * posterior_obs_loss
                + self.loss_config.prior_obs_weight * prior_obs_loss
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * belief_loss
                    + self.loss_config.obs_weight * obs_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/belief_loss", belief_loss)
        self.log(f"{stage}/pred_loss", belief_loss)
        self.log(f"{stage}/obs_loss", obs_loss)
        self.log(f"{stage}/posterior_obs_loss", posterior_obs_loss)
        self.log(f"{stage}/prior_obs_loss", prior_obs_loss)
        self.log(
            f"{stage}/posterior_obs_pred_target_cosine",
            torch.nn.functional.cosine_similarity(
                output.posterior_obs_prediction,
                output.obs_target,
                dim=-1,
            ).mean(),
        )
        if output.prior_obs_prediction.numel() > 0:
            self.log(
                f"{stage}/prior_obs_pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prior_obs_prediction,
                    output.obs_target[:, 1:],
                    dim=-1,
                ).mean(),
            )
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self.log(f"{stage}/embedding_sigreg_loss", embedding_sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self.log(
            f"{stage}/posterior_belief_sigreg_loss",
            posterior_belief_sigreg_loss,
        )
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        self.log(f"{stage}/belief_dim", float(output.prior_belief.shape[-1]))
        self.log(f"{stage}/z_dim", float(output.embedding.shape[-1]))
        self.log(
            f"{stage}/posterior_belief_pairwise_cosine_mean",
            self._pairwise_cosine_mean(output.posterior_belief),
        )
        self.log(
            f"{stage}/prior_belief_pairwise_cosine_mean",
            self._pairwise_cosine_mean(output.prior_belief),
        )
        if output.prior_belief.numel() > 0:
            self.log(
                f"{stage}/prior_target_belief_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prior_belief,
                    output.posterior_belief[:, 1:],
                    dim=-1,
                ).mean(),
            )
        if stage == "train":
            self._log_prismv2_training_text(
                batch=batch,
                output=output,
                loss=loss,
                belief_loss=belief_loss,
                obs_loss=obs_loss,
                posterior_obs_loss=posterior_obs_loss,
                prior_obs_loss=prior_obs_loss,
                sigreg_loss=sigreg_loss,
                embedding_sigreg_loss=embedding_sigreg_loss,
                posterior_belief_sigreg_loss=posterior_belief_sigreg_loss,
                sigreg_only_warmup_active=sigreg_only_warmup_active,
            )
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.posterior_obs_prediction,
                states=batch.states,
                stage=stage,
                probe_name="posterior_pred_obs",
            )
            if output.prior_obs_prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.prior_obs_prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="prior_pred_obs",
                )
            self._position_probe_step(
                embedding=output.posterior_belief,
                states=batch.states,
                stage=stage,
                probe_name="posterior_belief",
            )
            if output.prior_belief.numel() > 0:
                self._position_probe_step(
                    embedding=output.prior_belief,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="prior_belief",
                )
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
                probe_name="encoded_z",
            )
        if stage == "train":
            self._log_prismv2_visibility(batch=batch, output=output)
        return loss

    def _bwm_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, BWMLossConfig):
            raise TypeError(
                f"expected BWMLossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for BWM training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, BWMOutput):
            raise TypeError(f"expected BWMOutput, got {type(output).__name__}")

        log_bwm_diagnostics = self._should_log_bwm_diagnostics(stage)
        with self.runtime_monitor.time_block("loss", self.device):
            no_prior_pred_loss = output.embedding.new_tensor(0.0)
            no_prior_pred_target_cosine = output.embedding.new_tensor(0.0)
            shuffle_pred_loss = output.embedding.new_tensor(0.0)
            shuffle_pred_target_cosine = output.embedding.new_tensor(0.0)
            shuffle_no_prior_pred_loss = output.embedding.new_tensor(0.0)
            shuffle_no_prior_pred_target_cosine = output.embedding.new_tensor(0.0)
            shuffle_cosine_gap = output.embedding.new_tensor(0.0)
            prior_lift = output.embedding.new_tensor(0.0)
            shuffle_prior_lift = output.embedding.new_tensor(0.0)
            if output.prediction.numel() > 0:
                pred_z_norm = output.pred_z.norm(dim=-1).mean()
                prediction_norm = pred_z_norm
                pred_z = torch.nn.functional.normalize(output.pred_z, dim=-1)
                modulated_z = torch.nn.functional.normalize(output.modulated_z, dim=-1)
                prediction_for_modulated_loss = (
                    pred_z.detach()
                    if self.loss_config.detach_prediction_for_modulated_loss
                    else pred_z
                )
                pred_loss = torch.nn.functional.mse_loss(
                    prediction_for_modulated_loss,
                    modulated_z,
                )
                pred_target_cosine = torch.nn.functional.cosine_similarity(
                    pred_z, modulated_z, dim=-1
                ).mean()
                encoded_z = torch.nn.functional.normalize(
                    output.encoded_target_z,
                    dim=-1,
                )
                encoded_pred_cosine_distance = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        encoded_z,
                        pred_z,
                        dim=-1,
                    ).mean()
                )
                encoded_modulated_cosine_distance = (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        encoded_z,
                        modulated_z,
                        dim=-1,
                    ).mean()
                )
                pred_modulated_cosine_distance = 1.0 - pred_target_cosine
                if log_bwm_diagnostics:
                    with torch.no_grad():
                        prediction_for_diagnostics = pred_z.detach()
                        no_prior_target = torch.nn.functional.normalize(
                            self.model.target_without_prior(output),
                            dim=-1,
                        )
                        no_prior_pred_loss = torch.nn.functional.mse_loss(
                            prediction_for_diagnostics,
                            no_prior_target,
                        )
                        no_prior_pred_target_cosine = (
                            torch.nn.functional.cosine_similarity(
                                prediction_for_diagnostics,
                                no_prior_target,
                                dim=-1,
                            ).mean()
                        )
                        shuffled_target = torch.nn.functional.normalize(
                            self.model.shuffled_target(output),
                            dim=-1,
                        )
                        shuffle_pred_loss = torch.nn.functional.mse_loss(
                            prediction_for_diagnostics,
                            shuffled_target,
                        )
                        shuffle_pred_target_cosine = (
                            torch.nn.functional.cosine_similarity(
                                prediction_for_diagnostics,
                                shuffled_target,
                                dim=-1,
                            ).mean()
                        )
                        shuffled_no_prior_target = torch.nn.functional.normalize(
                            self.model.shuffled_target_without_prior(output),
                            dim=-1,
                        )
                        shuffle_no_prior_pred_loss = torch.nn.functional.mse_loss(
                            prediction_for_diagnostics,
                            shuffled_no_prior_target,
                        )
                        shuffle_no_prior_pred_target_cosine = (
                            torch.nn.functional.cosine_similarity(
                                prediction_for_diagnostics,
                                shuffled_no_prior_target,
                                dim=-1,
                            ).mean()
                        )
                        shuffle_cosine_gap = (
                            pred_target_cosine.detach() - shuffle_pred_target_cosine
                        )
                        prior_lift = (
                            pred_target_cosine.detach() - no_prior_pred_target_cosine
                        )
                        shuffle_prior_lift = (
                            shuffle_pred_target_cosine
                            - shuffle_no_prior_pred_target_cosine
                        )
            else:
                pred_loss = output.embedding.new_tensor(0.0)
                pred_target_cosine = output.embedding.new_tensor(0.0)
                prediction_norm = output.embedding.new_tensor(0.0)
                encoded_pred_cosine_distance = output.embedding.new_tensor(0.0)
                encoded_modulated_cosine_distance = output.embedding.new_tensor(0.0)
                pred_modulated_cosine_distance = output.embedding.new_tensor(0.0)
            sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            use_obs_cmi = (
                self.loss_config.obs_cmi_enabled
                or self.loss_config.obs_cmi_weight > 0.0
            )
            use_prior_cmi = (
                self.loss_config.prior_cmi_enabled
                or self.loss_config.prior_cmi_weight > 0.0
            )
            use_original_pred = (
                self.loss_config.original_pred_enabled
                or self.loss_config.original_pred_weight > 0.0
            )
            use_original_sigreg = (
                self.loss_config.original_sigreg_enabled
                or self.loss_config.original_sigreg_weight > 0.0
            )
            original_pred_loss = output.embedding.new_tensor(0.0)
            original_pred_target_cosine = output.embedding.new_tensor(0.0)
            if output.prediction.numel() > 0 and use_original_pred:
                original_pred_target = output.encoded_target_z
                if original_pred_target.shape != output.prediction.shape:
                    raise ValueError(
                        "expected original prediction target shape "
                        f"{tuple(output.prediction.shape)}, "
                        f"got {tuple(original_pred_target.shape)}"
                    )
                original_pred_loss = torch.nn.functional.mse_loss(
                    output.prediction,
                    original_pred_target,
                )
                original_pred_target_cosine = torch.nn.functional.cosine_similarity(
                    output.prediction,
                    original_pred_target,
                    dim=-1,
                ).mean()
            original_sigreg_loss = (
                self._sigreg_loss(output.encoded_sigreg_z, stage)
                if output.encoded_sigreg_z.numel() > 0 and use_original_sigreg
                else output.embedding.new_tensor(0.0)
            )
            obs_cmi_loss = output.embedding.new_tensor(0.0)
            obs_cmi_acc = output.embedding.new_tensor(0.0)
            obs_cmi_logit_margin = output.embedding.new_tensor(0.0)
            prior_cmi_loss = output.embedding.new_tensor(0.0)
            prior_cmi_acc = output.embedding.new_tensor(0.0)
            prior_cmi_logit_margin = output.embedding.new_tensor(0.0)
            obs_mask_gt_loss = output.embedding.new_tensor(0.0)
            obs_mask_gt_prob = output.embedding.new_tensor(0.0)
            obs_mask_gt_acc = output.embedding.new_tensor(0.0)
            obs_mask_entropy_loss = output.embedding.new_tensor(0.0)
            use_obs_mask_gt = (
                self.model.modulated_encoder is not None
                and self.model.modulated_encoder.prior_form
                == BWMPriorForm.OBSERVATION_SOFTMAX_MASK
                and output.prior.numel() > 0
            )
            if use_obs_mask_gt:
                (
                    obs_mask_gt_loss,
                    obs_mask_gt_prob,
                    obs_mask_gt_acc,
                    obs_mask_entropy_loss,
                ) = self._bwm_obs_mask_metrics(
                    output=output,
                    states=batch.states[:, 1:],
                )
            if output.prediction.numel() > 0 and use_obs_cmi:
                obs_cmi_logits = self.model.obs_cmi_logits(
                    output=output,
                    fixed_negatives=self.loss_config.obs_cmi_fixed_negatives,
                    temperature=self.loss_config.cmi_temperature,
                    negative_source=self.loss_config.obs_cmi_negative_source,
                    noise_ratio=self.loss_config.obs_cmi_noise_ratio,
                    normalize_prediction=self.loss_config.normalize_prediction_for_cmi,
                    normalize_target=self.loss_config.normalize_target_for_cmi,
                )
                obs_cmi_labels = torch.zeros(
                    obs_cmi_logits.shape[0],
                    dtype=torch.long,
                    device=obs_cmi_logits.device,
                )
                obs_cmi_loss, obs_cmi_acc, obs_cmi_logit_margin = (
                    self._contrastive_loss_metrics(obs_cmi_logits, obs_cmi_labels)
                )
            if output.prediction.numel() > 0 and use_prior_cmi:
                prior_cmi_logits = self.model.prior_cmi_logits(
                    output=output,
                    fixed_negatives=self.loss_config.prior_cmi_fixed_negatives,
                    temperature=self.loss_config.cmi_temperature,
                    negative_source=self.loss_config.prior_cmi_negative_source,
                    normalize_prediction=self.loss_config.normalize_prediction_for_cmi,
                    normalize_target=self.loss_config.normalize_target_for_cmi,
                )
                prior_cmi_labels = torch.zeros(
                    prior_cmi_logits.shape[0],
                    dtype=torch.long,
                    device=prior_cmi_logits.device,
                )
                prior_cmi_loss, prior_cmi_acc, prior_cmi_logit_margin = (
                    self._contrastive_loss_metrics(prior_cmi_logits, prior_cmi_labels)
                )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * pred_loss
                    + self.loss_config.original_pred_weight * original_pred_loss
                    + self.loss_config.original_sigreg_weight * original_sigreg_loss
                    + sigreg_regularizer_loss
                    + self.loss_config.obs_cmi_weight * obs_cmi_loss
                    + self.loss_config.prior_cmi_weight * prior_cmi_loss
                    + self.loss_config.obs_mask_gt_weight * obs_mask_gt_loss
                    + self.loss_config.obs_mask_entropy_weight * obs_mask_entropy_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/pred_loss", pred_loss)
        self.log(f"{stage}/pred_z_modulated_z_loss", pred_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self.log(f"{stage}/prediction_norm", prediction_norm)
        if use_original_pred:
            self.log(f"{stage}/original_pred_loss", original_pred_loss)
            self.log(f"{stage}/pred_z_encoded_z_loss", original_pred_loss)
            self.log(
                f"{stage}/original_pred_target_cosine",
                original_pred_target_cosine,
            )
        if use_original_sigreg:
            self.log(f"{stage}/original_sigreg_loss", original_sigreg_loss)
        if use_obs_cmi:
            self.log(f"{stage}/obs_cmi_loss", obs_cmi_loss)
            self.log(f"{stage}/obs_cmi_acc", obs_cmi_acc)
            self.log(f"{stage}/obs_cmi_logit_margin", obs_cmi_logit_margin)
        if use_prior_cmi:
            self.log(f"{stage}/prior_cmi_loss", prior_cmi_loss)
            self.log(f"{stage}/prior_cmi_acc", prior_cmi_acc)
            self.log(f"{stage}/prior_cmi_logit_margin", prior_cmi_logit_margin)
        if use_obs_mask_gt:
            self.log(f"{stage}/obs_mask_gt_loss", obs_mask_gt_loss)
            self.log(f"{stage}/obs_mask_gt_prob", obs_mask_gt_prob)
            self.log(f"{stage}/obs_mask_gt_acc", obs_mask_gt_acc)
            self.log(f"{stage}/obs_mask_entropy_loss", obs_mask_entropy_loss)
            self.log(f"{stage}/obs_mask_entropy", obs_mask_entropy_loss)
        self.log(f"{stage}/pred_target_cosine", pred_target_cosine)
        self.log(f"{stage}/pred_z_modulated_z_cosine", pred_target_cosine)
        self.log(f"{stage}/encoded_pred_cosine_distance", encoded_pred_cosine_distance)
        self.log(
            f"{stage}/encoded_z_pred_z_cosine_distance",
            encoded_pred_cosine_distance,
        )
        self.log(
            f"{stage}/encoded_modulated_cosine_distance",
            encoded_modulated_cosine_distance,
        )
        self.log(
            f"{stage}/encoded_z_modulated_z_cosine_distance",
            encoded_modulated_cosine_distance,
        )
        self.log(
            f"{stage}/pred_modulated_cosine_distance",
            pred_modulated_cosine_distance,
        )
        self.log(
            f"{stage}/pred_z_modulated_z_cosine_distance",
            pred_modulated_cosine_distance,
        )
        if log_bwm_diagnostics:
            self.log(f"{stage}/no_prior_pred_loss", no_prior_pred_loss)
            self.log(
                f"{stage}/no_prior_pred_target_cosine",
                no_prior_pred_target_cosine,
            )
            self.log(f"{stage}/prior_lift", prior_lift)
            self.log(f"{stage}/shuffle_pred_loss", shuffle_pred_loss)
            self.log(f"{stage}/shuffle_pred_target_cosine", shuffle_pred_target_cosine)
            self.log(
                f"{stage}/shuffle_no_prior_pred_loss",
                shuffle_no_prior_pred_loss,
            )
            self.log(
                f"{stage}/shuffle_no_prior_pred_target_cosine",
                shuffle_no_prior_pred_target_cosine,
            )
            self.log(f"{stage}/shuffle_cosine_gap", shuffle_cosine_gap)
            self.log(f"{stage}/shuffle_prior_lift", shuffle_prior_lift)
        self.log(f"{stage}/prior_norm", self._mean_or_zero(output.prior.norm(dim=-1)))
        self._log_embedding_collapse_metrics(output.target, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        self._log_bwm_z_space_metrics(output=output, stage=stage)
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.modulated_z,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="modulated_z",
            )
            self._position_probe_step(
                embedding=output.encoded_target_z,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="encoded_z",
            )
            if output.prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_z",
                )
        if stage == "train":
            self._log_bwm_visibility(batch=batch, output=output)
        return loss

    def _bwmv2_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, BWMV2LossConfig):
            raise TypeError(
                f"expected BWMV2LossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for BWMv2 training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, BWMV2Output):
            raise TypeError(f"expected BWMV2Output, got {type(output).__name__}")

        use_bwm_prediction = (
            self.loss_config.bwm_prediction_enabled
            or self.loss_config.bwm_prediction_weight > 0.0
        )
        use_bwm_sigreg = (
            self.loss_config.bwm_sigreg_enabled
            or self.loss_config.bwm_sigreg_weight > 0.0
        )
        if (use_bwm_prediction or use_bwm_sigreg) and not output.bwm_branch_available:
            raise ValueError(
                "BWMv2 BWM branch loss requires model.bwm_branch_enabled=true"
            )

        with self.runtime_monitor.time_block("loss", self.device):
            pred_loss = (
                torch.nn.functional.mse_loss(output.prediction, output.target)
                if output.prediction.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            sigreg_loss = (
                self._sigreg_loss(output.sigreg_embedding, stage)
                if output.sigreg_embedding.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            bwm_pred_loss = output.embedding.new_tensor(0.0)
            bwm_pred_target_cosine = output.embedding.new_tensor(0.0)
            bwm_sigreg_loss = output.embedding.new_tensor(0.0)
            if output.prediction.numel() > 0 and output.bwm_branch_available:
                prediction = torch.nn.functional.normalize(output.prediction, dim=-1)
                modulated_target = torch.nn.functional.normalize(
                    output.modulated_embedding,
                    dim=-1,
                )
                prediction_for_bwm_loss = (
                    prediction.detach()
                    if self.loss_config.detach_prediction_for_modulated_loss
                    else prediction
                )
                bwm_pred_loss = torch.nn.functional.mse_loss(
                    prediction_for_bwm_loss,
                    modulated_target,
                )
                bwm_pred_target_cosine = torch.nn.functional.cosine_similarity(
                    prediction,
                    modulated_target,
                    dim=-1,
                ).mean()
            if output.modulated_sigreg_embedding.numel() > 0 and use_bwm_sigreg:
                bwm_sigreg_loss = self._sigreg_loss(
                    output.modulated_sigreg_embedding,
                    stage,
                )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.prediction_weight * pred_loss
                    + sigreg_regularizer_loss
                    + self.loss_config.bwm_prediction_weight * bwm_pred_loss
                    + self.loss_config.bwm_sigreg_weight * bwm_sigreg_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/pred_loss", pred_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self.log(
            f"{stage}/prediction_norm",
            self._mean_or_zero(output.prediction.norm(dim=-1)),
        )
        if output.prediction.numel() > 0:
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prediction, output.target, dim=-1
                ).mean(),
            )
        if output.bwm_branch_available:
            self.log(f"{stage}/bwm_pred_loss", bwm_pred_loss)
            self.log(f"{stage}/bwm_sigreg_loss", bwm_sigreg_loss)
            self.log(f"{stage}/bwm_pred_target_cosine", bwm_pred_target_cosine)
            self.log(f"{stage}/pred_z_modulated_z_loss", bwm_pred_loss)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        self._log_bwm_z_space_metrics(output=output, stage=stage)
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
            )
            self._position_probe_step(
                embedding=output.target,
                states=batch.states[:, 1:],
                stage=stage,
                probe_name="encoded_z",
            )
            if output.prediction.numel() > 0:
                self._position_probe_step(
                    embedding=output.prediction,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="pred_z",
                )
            if output.bwm_branch_available:
                self._position_probe_step(
                    embedding=output.modulated_embedding,
                    states=batch.states[:, 1:],
                    stage=stage,
                    probe_name="modulated_z",
                )
        if stage == "train":
            self._log_bwm_visibility(batch=batch, output=output)
        return loss

    def _should_log_bwm_diagnostics(self, stage: str) -> bool:
        if stage != "train":
            return True
        try:
            log_every_n_steps = self.trainer.log_every_n_steps
        except RuntimeError:
            log_every_n_steps = 1
        return self.global_step % max(int(log_every_n_steps), 1) == 0

    def _sigreg_only_warmup_active(self, stage: str) -> bool:
        warmup_steps = int(getattr(self.loss_config, "sigreg_only_warmup_steps", 0))
        return stage == "train" and self.global_step < warmup_steps

    def _wasserstein_sigreg_active(self, stage: str) -> bool:
        weight = float(getattr(self.loss_config, "wasserstein_sigreg_weight", 0.0))
        if weight <= 0.0:
            return False
        if bool(getattr(self.loss_config, "wasserstein_sigreg_only_warmup", False)):
            return self._sigreg_only_warmup_active(stage)
        return True

    def _prepare_sigreg_input(
        self,
        embedding: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        sigreg_input = embedding.transpose(0, 1)
        if bool(getattr(self.loss_config, "sigreg_input_batch_norm", False)):
            sigreg_input = self._batch_norm_sigreg_input(sigreg_input)
        jitter_size = float(getattr(self.loss_config, "sigreg_warmup_jitter_size", 0.0))
        if jitter_size > 0.0 and self._sigreg_only_warmup_active(stage):
            sigreg_input = sigreg_input + torch.randn_like(sigreg_input) * jitter_size
        if bool(getattr(self.loss_config, "sigreg_input_scale_sqrt_dim", False)):
            sigreg_input = self._scale_sigreg_input_by_sqrt_dim(sigreg_input)
        return sigreg_input

    def _sigreg_loss(self, embedding: torch.Tensor, stage: str) -> torch.Tensor:
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized before computing its loss")
        sigreg_input = self._prepare_sigreg_input(embedding, stage)
        return self.sigreg(sigreg_input)

    def _wasserstein_sigreg_loss(
        self,
        embedding: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        if self.wasserstein_sigreg is None:
            raise RuntimeError(
                "Wasserstein SIGReg must be initialized before computing its loss"
            )
        sigreg_input = self._prepare_sigreg_input(embedding, stage)
        return self.wasserstein_sigreg(sigreg_input)

    def _maybe_wasserstein_sigreg_loss(
        self,
        embedding: torch.Tensor,
        stage: str,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if not self._wasserstein_sigreg_active(stage) or embedding.numel() == 0:
            return fallback.new_tensor(0.0)
        return self._wasserstein_sigreg_loss(embedding, stage)

    def _weighted_sigreg_regularizer_loss(
        self,
        sigreg_loss: torch.Tensor,
        wasserstein_sigreg_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.loss_config.sigreg_weight * sigreg_loss
            + float(getattr(self.loss_config, "wasserstein_sigreg_weight", 0.0))
            * wasserstein_sigreg_loss
        )

    def _log_wasserstein_sigreg(
        self,
        stage: str,
        wasserstein_sigreg_loss: torch.Tensor,
    ) -> None:
        if float(getattr(self.loss_config, "wasserstein_sigreg_weight", 0.0)) <= 0.0:
            return
        self.log(f"{stage}/wasserstein_sigreg_loss", wasserstein_sigreg_loss)
        active = torch.tensor(
            float(self._wasserstein_sigreg_active(stage)),
            device=self.device,
        )
        self.log(f"{stage}/wasserstein_sigreg_active", active)

    def _scale_sigreg_input_by_sqrt_dim(
        self,
        sigreg_input: torch.Tensor,
    ) -> torch.Tensor:
        feature_dim = sigreg_input.shape[-1]
        if feature_dim <= 0:
            raise ValueError("SIGReg input feature dim must be positive")
        return sigreg_input * (feature_dim**0.5)

    def _batch_norm_sigreg_input(self, sigreg_input: torch.Tensor) -> torch.Tensor:
        if sigreg_input.ndim != 3:
            raise ValueError(f"expected SIGReg input rank 3, got {sigreg_input.ndim}")
        feature_dim = sigreg_input.shape[-1]
        flat = sigreg_input.reshape(-1, feature_dim)
        if flat.shape[0] < 2:
            raise ValueError("SIGReg input batch norm requires at least two samples")
        normalized = torch.nn.functional.batch_norm(
            flat,
            running_mean=None,
            running_var=None,
            training=True,
        )
        return normalized.view_as(sigreg_input)

    def _log_sigreg_only_warmup(self, stage: str, active: bool) -> None:
        warmup_steps = int(getattr(self.loss_config, "sigreg_only_warmup_steps", 0))
        if warmup_steps <= 0:
            return
        value = torch.tensor(float(active), device=self.device)
        self.log(f"{stage}/sigreg_only_warmup_active", value)

    def _contrastive_loss_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.ndim != 2:
            raise ValueError(f"expected logits rank 2, got {logits.ndim}")
        if labels.shape != (logits.shape[0],):
            raise ValueError(
                f"expected labels shape {(logits.shape[0],)}, got {tuple(labels.shape)}"
            )
        if logits.shape[1] <= 0:
            raise ValueError("logits must have at least one candidate")

        loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        positive_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        if logits.shape[1] == 1:
            margin = positive_logits.new_tensor(0.0)
        else:
            positive_mask = torch.nn.functional.one_hot(
                labels,
                num_classes=logits.shape[1],
            ).bool()
            negative_logits = logits.masked_fill(positive_mask, -torch.inf)
            margin = (positive_logits - negative_logits.max(dim=-1).values).mean()
        return loss, acc, margin

    def _bwm_obs_mask_metrics(
        self,
        output: BWMOutput,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model.modulated_encoder is None:
            raise ValueError("BWM observation mask metrics require the BWM branch")
        if (
            self.model.modulated_encoder.prior_form
            != BWMPriorForm.OBSERVATION_SOFTMAX_MASK
        ):
            raise ValueError("BWM observation mask metrics require obs mask prior form")
        if states.shape != output.prior.shape[:2]:
            raise ValueError(
                f"expected states shape {tuple(output.prior.shape[:2])}, "
                f"got {tuple(states.shape)}"
            )
        observation_size = self.model.config.observation_size
        if ((states < 0) | (states >= observation_size)).any():
            raise ValueError(f"states must be in [0, {observation_size - 1}]")

        mask = self.model.modulated_encoder.observation_mask_from_prior(
            output.observations[:, 1:],
            output.prior,
        )
        gt_prob = mask.gather(dim=-1, index=states.long().unsqueeze(-1)).squeeze(-1)
        eps = torch.finfo(gt_prob.dtype).eps
        loss = -gt_prob.clamp_min(eps).log().mean()
        prob = gt_prob.mean()
        acc = (mask.argmax(dim=-1) == states).float().mean()
        entropy = -(mask * mask.clamp_min(eps).log()).sum(dim=-1)
        max_entropy = torch.log(mask.new_tensor(mask.shape[-1]))
        normalized_entropy = (entropy / max_entropy.clamp_min(eps)).mean()
        return loss, prob, acc, normalized_entropy

    def _hwm_step(self, batch: SequenceBatch, stage: str) -> torch.Tensor:
        if not isinstance(self.loss_config, HWMLossConfig):
            raise TypeError(
                f"expected HWMLossConfig, got {type(self.loss_config).__name__}"
            )
        if self.sigreg is None:
            raise RuntimeError("SIGReg must be initialized for HWM training")
        with self.runtime_monitor.time_block("forward", self.device):
            output = self.model(batch.observations, batch.actions)
        if not isinstance(output, HWMOutput):
            raise TypeError(f"expected HWMOutput, got {type(output).__name__}")

        with self.runtime_monitor.time_block("loss", self.device):
            nll_loss = (
                self._heteroscedastic_loss(
                    output.prediction, output.target, output.logvar
                )
                if output.prediction.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            var_prior_loss = (
                output.logvar.square().mean()
                if output.logvar.numel() > 0
                else output.embedding.new_tensor(0.0)
            )
            sigreg_loss = self._sigreg_loss(output.sigreg_embedding, stage)
            wasserstein_sigreg_loss = self._maybe_wasserstein_sigreg_loss(
                output.sigreg_embedding,
                stage,
                output.embedding,
            )
            sigreg_regularizer_loss = self._weighted_sigreg_regularizer_loss(
                sigreg_loss,
                wasserstein_sigreg_loss,
            )
            vicreg_loss = self._maybe_temporal_vicreg_loss(
                output.embedding,
                output.embedding,
            )
            sigreg_only_warmup_active = self._sigreg_only_warmup_active(stage)
            if sigreg_only_warmup_active:
                loss = sigreg_regularizer_loss
            else:
                loss = (
                    self.loss_config.nll_weight * nll_loss
                    + self.loss_config.var_prior_weight * var_prior_loss
                    + sigreg_regularizer_loss
                    + self._weighted_temporal_vicreg_loss(vicreg_loss)
                )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/nll_loss", nll_loss)
        self.log(f"{stage}/var_prior_loss", var_prior_loss)
        self.log(f"{stage}/sigreg_loss", sigreg_loss)
        self._log_wasserstein_sigreg(stage, wasserstein_sigreg_loss)
        self._log_temporal_vicreg(stage, vicreg_loss)
        self._log_sigreg_only_warmup(stage, sigreg_only_warmup_active)
        self._log_embedding_collapse_metrics(output.embedding, stage)
        self._log_frame_embedding_temporal_metrics(
            output.embedding,
            batch.observations,
            stage,
        )
        if stage == "val" and self.metrics_config.position_probe_enabled:
            self._position_probe_step(
                embedding=output.embedding,
                states=batch.states,
                stage=stage,
            )
        if output.prediction.numel() > 0:
            error2 = (output.prediction - output.target).square()
            precision = torch.exp(-output.logvar)
            per_dim_nll = 0.5 * (error2 * precision + output.logvar)
            per_axis_mse = error2.mean(dim=(0, 1))
            per_axis_nll = per_dim_nll.mean(dim=(0, 1))
            self._log_hwm_predictable_subspace_metrics(
                stage=stage,
                prediction=output.prediction,
                target=output.target,
                error2=error2,
                logvar=output.logvar,
                precision=precision,
                per_axis_mse=per_axis_mse,
                per_axis_nll=per_axis_nll,
            )
            self.log(
                f"{stage}/pred_target_cosine",
                torch.nn.functional.cosine_similarity(
                    output.prediction, output.target, dim=-1
                ).mean(),
            )
            self.log(f"{stage}/logvar_mean", output.logvar.mean())
            self.log(f"{stage}/logvar_min", output.logvar.min())
            self.log(f"{stage}/logvar_max", output.logvar.max())
            self.log(f"{stage}/precision_min", precision.min())
            self.log(f"{stage}/precision_max", precision.max())
            self.log(
                f"{stage}/mse",
                error2.mean(),
            )
            self.log(f"{stage}/per_axis_mse_mean", per_axis_mse.mean())
            self.log(f"{stage}/per_axis_mse_max", per_axis_mse.max())
            self.log(f"{stage}/per_axis_nll_mean", per_axis_nll.mean())
            self.log(f"{stage}/per_axis_nll_max", per_axis_nll.max())
        if stage == "train":
            self._log_hwm_visibility(batch=batch, output=output)
        return loss

    def _position_probe_step(
        self,
        embedding: torch.Tensor,
        states: torch.Tensor,
        stage: str,
        probe_name: str = "embedding",
    ) -> None:
        try:
            self._run_position_probe_step(
                embedding=embedding,
                states=states,
                stage=stage,
                probe_name=probe_name,
            )
        except Exception as error:
            self._warn_position_probe_failure(
                stage=stage,
                probe_name=probe_name,
                error=error,
            )

    def _run_position_probe_step(
        self,
        embedding: torch.Tensor,
        states: torch.Tensor,
        stage: str,
        probe_name: str = "embedding",
    ) -> None:
        if stage != "val":
            raise ValueError("position probe is only supported for validation")
        if not probe_name:
            raise ValueError("probe_name must be non-empty")
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        if states.shape != embedding.shape[:2]:
            raise ValueError(
                f"expected states shape {tuple(embedding.shape[:2])}, "
                f"got {tuple(states.shape)}"
            )

        features = embedding.detach().reshape(-1, embedding.shape[-1]).float().cpu()
        labels = states.reshape(-1).long().cpu()
        env_config = self.model.config.env_config
        num_positions = env_config.observation_size
        if ((labels < 0) | (labels >= num_positions)).any():
            raise ValueError(f"states must be in [0, {num_positions - 1}]")

        if isinstance(env_config, GridWorld2DConfig):
            self._fit_and_log_position_probe(
                features=features,
                labels=labels % env_config.width,
                num_classes=env_config.width,
                stage=stage,
                probe_name=f"{probe_name}_x",
            )
            self._fit_and_log_position_probe(
                features=features,
                labels=labels // env_config.width,
                num_classes=env_config.height,
                stage=stage,
                probe_name=f"{probe_name}_y",
            )
            return

        self._fit_and_log_position_probe(
            features=features,
            labels=labels,
            num_classes=num_positions,
            stage=stage,
            probe_name=probe_name,
        )

    def _fit_and_log_position_probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        stage: str,
        probe_name: str,
    ) -> None:
        if num_classes <= 1:
            raise ValueError("position probe requires at least two classes")

        sample_count = features.shape[0]
        if sample_count <= 1:
            raise ValueError("position probe requires at least two samples")
        bias = torch.ones(sample_count, 1, dtype=features.dtype)
        features = torch.cat([features, bias], dim=-1)
        support_mask = torch.arange(sample_count) % 2 == 0
        query_mask = ~support_mask
        if not query_mask.any():
            query_mask = support_mask

        support_features = features[support_mask]
        support_targets = torch.nn.functional.one_hot(
            labels[support_mask],
            num_classes=num_classes,
        ).to(dtype=features.dtype)
        eye = torch.eye(features.shape[-1], dtype=features.dtype)
        eye[-1, -1] = 0.0
        ridge = self.metrics_config.position_probe_ridge
        weights = torch.linalg.solve(
            support_features.T @ support_features + ridge * eye,
            support_features.T @ support_targets,
        )

        logits = features[query_mask] @ weights
        query_labels = labels[query_mask]
        probe_loss = torch.nn.functional.cross_entropy(
            logits,
            query_labels,
        )
        probe_acc = (logits.argmax(dim=-1) == query_labels).float().mean()
        self.log(f"{stage}/position_probe_{probe_name}_loss", probe_loss)
        self.log(f"{stage}/position_probe_{probe_name}_acc", probe_acc)

    def _warn_position_probe_failure(
        self,
        stage: str,
        probe_name: str,
        error: Exception,
    ) -> None:
        warning_key = f"{stage}:{probe_name}:{type(error).__name__}:{error}"
        if warning_key in self._position_probe_warning_keys:
            return
        self._position_probe_warning_keys.add(warning_key)
        warnings.warn(
            "Skipping position probe "
            f"{stage}/{probe_name} after {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _log_hwm_predictable_subspace_metrics(
        self,
        stage: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        error2: torch.Tensor,
        logvar: torch.Tensor,
        precision: torch.Tensor,
        per_axis_mse: torch.Tensor,
        per_axis_nll: torch.Tensor,
    ) -> None:
        if error2.ndim != 3:
            raise ValueError(f"expected error2 rank 3, got {error2.ndim}")
        if prediction.shape != target.shape:
            raise ValueError(
                f"expected target shape {tuple(prediction.shape)}, "
                f"got {tuple(target.shape)}"
            )
        if prediction.shape != error2.shape:
            raise ValueError(
                f"expected prediction shape {tuple(error2.shape)}, "
                f"got {tuple(prediction.shape)}"
            )
        latent_dim = error2.shape[-1]
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        per_axis_precision = precision.mean(dim=(0, 1))
        per_axis_variance = torch.exp(logvar).mean(dim=(0, 1))
        calibration = per_axis_mse / per_axis_variance.clamp_min(1e-8)
        precision_total = per_axis_precision.sum().clamp_min(1e-8)
        effective_dim = (
            precision_total.square() / per_axis_precision.square().sum().clamp_min(1e-8)
        )

        top_k = min(8, latent_dim)
        top_indices = torch.topk(per_axis_precision, k=top_k, largest=True).indices
        bottom_indices = torch.topk(per_axis_precision, k=top_k, largest=False).indices
        weighted_dot = (precision * prediction * target).sum(dim=-1)
        weighted_prediction_norm = (precision * prediction.square()).sum(dim=-1).sqrt()
        weighted_target_norm = (precision * target.square()).sum(dim=-1).sqrt()
        weighted_cosine = weighted_dot / (
            weighted_prediction_norm * weighted_target_norm
        ).clamp_min(1e-8)

        self.log(f"{stage}/weighted_mse", (error2 * precision).mean())
        self.log(f"{stage}/precision_weighted_cosine", weighted_cosine.mean())
        self.log(f"{stage}/calibration_mean", calibration.mean())
        self.log(f"{stage}/calibration_min", calibration.min())
        self.log(f"{stage}/calibration_max", calibration.max())
        self.log(f"{stage}/effective_predictable_dim", effective_dim)
        self.log(
            f"{stage}/effective_predictable_dim_ratio",
            effective_dim / float(latent_dim),
        )
        self.log(
            f"{stage}/top_precision_mass",
            per_axis_precision[top_indices].sum() / precision_total,
        )
        self.log(f"{stage}/top_precision_mse", per_axis_mse[top_indices].mean())
        self.log(f"{stage}/bottom_precision_mse", per_axis_mse[bottom_indices].mean())
        self.log(f"{stage}/top_precision_nll", per_axis_nll[top_indices].mean())
        self.log(f"{stage}/bottom_precision_nll", per_axis_nll[bottom_indices].mean())
        self.log(
            f"{stage}/top_precision_cosine",
            torch.nn.functional.cosine_similarity(
                prediction[..., top_indices],
                target[..., top_indices],
                dim=-1,
            ).mean(),
        )
        self.log(
            f"{stage}/bottom_precision_cosine",
            torch.nn.functional.cosine_similarity(
                prediction[..., bottom_indices],
                target[..., bottom_indices],
                dim=-1,
            ).mean(),
        )
        self.log(
            f"{stage}/top_precision_mean",
            per_axis_precision[top_indices].mean(),
        )
        self.log(
            f"{stage}/bottom_precision_mean",
            per_axis_precision[bottom_indices].mean(),
        )
        # Norm metrics for detecting norm shrinking trick
        top_pred = prediction[..., top_indices]
        top_target = target[..., top_indices]
        self.log(
            f"{stage}/top_precision_pred_norm",
            top_pred.norm(dim=-1).mean(),
        )
        self.log(
            f"{stage}/top_precision_target_norm",
            top_target.norm(dim=-1).mean(),
        )
        self.log(
            f"{stage}/top_precision_pred_target_norm_ratio",
            top_pred.norm(dim=-1).mean()
            / top_target.norm(dim=-1).mean().clamp_min(1e-8),
        )

    def _log_embedding_collapse_metrics(
        self, embedding: torch.Tensor, stage: str
    ) -> None:
        """Log metrics to detect embedding collapse to constant."""
        if embedding.ndim != 3:
            raise ValueError(f"expected embedding rank 3, got {embedding.ndim}")
        # Flatten to [B*T, D]
        emb_flat = embedding.reshape(-1, embedding.shape[-1])
        if emb_flat.shape[0] < 2:
            return  # Skip if insufficient samples

        # Dimension-level std: std across samples for each dimension
        dim_std = emb_flat.std(dim=0).mean()
        self.log(f"{stage}/embedding_dim_std", dim_std)

        self.log(f"{stage}/pairwise_cosine_mean", self._pairwise_cosine_mean(embedding))

    def _pairwise_cosine_mean(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"expected values rank 3, got {values.ndim}")
        flat_values = values.reshape(-1, values.shape[-1])
        if flat_values.shape[0] < 2:
            return values.new_tensor(0.0)

        # Sample a subset if too large to avoid O(N^2) computation.
        max_samples = min(128, flat_values.shape[0])
        if flat_values.shape[0] > max_samples:
            indices = torch.randperm(flat_values.shape[0], device=flat_values.device)[
                :max_samples
            ]
            sample = flat_values[indices]
        else:
            sample = flat_values

        cosine = torch.nn.functional.cosine_similarity(
            sample.unsqueeze(1),
            sample.unsqueeze(0),
            dim=-1,
        )
        item_count = cosine.shape[0]
        mask = ~torch.eye(item_count, dtype=torch.bool, device=cosine.device)
        return cosine[mask].mean()

    def _log_bwm_z_space_metrics(self, output: BWMV2Output, stage: str) -> None:
        """Log BWMv2 z-space norms, spread, and pairwise distances."""

        if output.encoded_z.ndim != 3:
            raise ValueError(f"expected encoded_z rank 3, got {output.encoded_z.ndim}")
        self.log(
            f"{stage}/encoded_z_norm",
            self._mean_or_zero(output.encoded_z.norm(dim=-1)),
        )
        self.log(
            f"{stage}/encoded_z_std",
            output.encoded_z.reshape(-1, output.encoded_z.shape[-1])
            .std(dim=0, unbiased=False)
            .mean(),
        )
        self.log(
            f"{stage}/modulated_z_norm",
            self._mean_or_zero(output.modulated_z.norm(dim=-1)),
        )
        if output.modulated_z.numel() > 0:
            self.log(
                f"{stage}/modulated_z_std",
                output.modulated_z.reshape(-1, output.modulated_z.shape[-1])
                .std(dim=0, unbiased=False)
                .mean(),
            )
        else:
            self.log(f"{stage}/modulated_z_std", output.encoded_z.new_tensor(0.0))
        self.log(
            f"{stage}/pred_z_norm",
            self._mean_or_zero(output.pred_z.norm(dim=-1)),
        )
        if output.pred_z.numel() > 0:
            self.log(
                f"{stage}/pred_z_std",
                output.pred_z.reshape(-1, output.pred_z.shape[-1])
                .std(dim=0, unbiased=False)
                .mean(),
            )
            encoded_target_z = output.encoded_target_z
            if encoded_target_z.shape != output.pred_z.shape:
                raise ValueError(
                    "expected encoded_target_z shape "
                    f"{tuple(output.pred_z.shape)}, got {tuple(encoded_target_z.shape)}"
                )
            self.log(
                f"{stage}/pred_z_encoded_z_mse_distance",
                torch.nn.functional.mse_loss(output.pred_z, encoded_target_z),
            )
            self.log(
                f"{stage}/encoded_z_pred_z_cosine_distance",
                1.0
                - torch.nn.functional.cosine_similarity(
                    encoded_target_z,
                    output.pred_z,
                    dim=-1,
                ).mean(),
            )
            if output.bwm_branch_available:
                self.log(
                    f"{stage}/encoded_z_modulated_z_cosine_distance",
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        encoded_target_z,
                        output.modulated_z,
                        dim=-1,
                    ).mean(),
                )
                self.log(
                    f"{stage}/pred_z_modulated_z_cosine_distance",
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        output.pred_z,
                        output.modulated_z,
                        dim=-1,
                    ).mean(),
                )
                self.log(
                    f"{stage}/pred_z_modulated_z_mse_distance",
                    torch.nn.functional.mse_loss(output.pred_z, output.modulated_z),
                )
                self.log(
                    f"{stage}/encoded_z_modulated_z_mse_distance",
                    torch.nn.functional.mse_loss(encoded_target_z, output.modulated_z),
                )
            else:
                zero = output.encoded_z.new_tensor(0.0)
                self.log(f"{stage}/encoded_z_modulated_z_cosine_distance", zero)
                self.log(f"{stage}/pred_z_modulated_z_cosine_distance", zero)
                self.log(f"{stage}/pred_z_modulated_z_mse_distance", zero)
                self.log(f"{stage}/encoded_z_modulated_z_mse_distance", zero)
        else:
            zero = output.encoded_z.new_tensor(0.0)
            self.log(f"{stage}/pred_z_std", zero)
            self.log(f"{stage}/encoded_z_pred_z_cosine_distance", zero)
            self.log(f"{stage}/encoded_z_modulated_z_cosine_distance", zero)
            self.log(f"{stage}/pred_z_modulated_z_cosine_distance", zero)
            self.log(f"{stage}/pred_z_encoded_z_mse_distance", zero)
            self.log(f"{stage}/pred_z_modulated_z_mse_distance", zero)
            self.log(f"{stage}/encoded_z_modulated_z_mse_distance", zero)

    def _heteroscedastic_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"expected target shape {tuple(prediction.shape)}, "
                f"got {tuple(target.shape)}"
            )
        if logvar.shape != prediction.shape:
            raise ValueError(
                f"expected logvar shape {tuple(prediction.shape)}, "
                f"got {tuple(logvar.shape)}"
            )
        if not torch.isfinite(logvar).all():
            raise ValueError("logvar must be finite")
        error2 = (target - prediction).square()
        per_dim_loss = 0.5 * (error2 * torch.exp(-logvar) + logvar)
        return per_dim_loss.mean()

    def _log_hdwm_visibility(
        self,
        batch: SequenceBatch,
        output: HDWMOutput,
        cmi_logits: torch.Tensor | None,
    ) -> None:
        """Log diagnostic metrics that explain the training objective."""

        self.log("train/prior_norm", output.prior.norm(dim=-1).mean())
        self.log("train/posterior_norm", output.posterior.norm(dim=-1).mean())
        self.log(
            "train/prior_posterior_cosine",
            torch.nn.functional.cosine_similarity(
                output.prior, output.posterior, dim=-1
            ).mean(),
        )

        readout_confidence = output.readout_attention.max(dim=-1).values.mean()
        readout_entropy = (
            -(output.readout_attention * output.readout_attention.clamp_min(1e-8).log())
            .sum(dim=-1)
            .mean()
        )
        self.log("train/readout_confidence", readout_confidence)
        self.log("train/readout_entropy", readout_entropy)

        self._log_batch_visibility(batch)

        if cmi_logits is not None:
            if cmi_logits.ndim != 3 or cmi_logits.shape[1] != cmi_logits.shape[2]:
                raise ValueError(
                    "expected cmi_logits with shape [B, T, T], "
                    f"got {tuple(cmi_logits.shape)}"
                )
            sequence_length = cmi_logits.shape[1]
            if sequence_length > 1:
                positive_logits = cmi_logits.diagonal(dim1=1, dim2=2)  # [B, T]
                negative_logits = cmi_logits.masked_fill(
                    torch.eye(
                        sequence_length,
                        dtype=torch.bool,
                        device=cmi_logits.device,
                    ).unsqueeze(0),
                    -torch.inf,
                )
                cmi_margin = positive_logits - negative_logits.max(dim=-1).values
                self.log("train/cmi_logit_margin", cmi_margin.mean())

        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.metrics_config.log_layer_weight_stats:
            self._log_layer_weight_stats()

    def _log_lewm_visibility(
        self,
        batch: SequenceBatch,
        output: LEWMOutput | LEWMV2Output,
    ) -> None:
        self.log("train/embedding_norm", output.embedding.norm(dim=-1).mean())
        self.log(
            "train/embedding_std",
            output.embedding.reshape(-1, output.embedding.shape[-1])
            .std(dim=0, unbiased=False)
            .mean(),
        )
        self._log_batch_visibility(batch)

        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.metrics_config.log_layer_weight_stats:
            self._log_layer_weight_stats()

    def _log_prism_visibility(
        self,
        batch: SequenceBatch,
        output: PRISMOutput,
    ) -> None:
        self.log("train/embedding_norm", output.embedding.norm(dim=-1).mean())
        self.log("train/clean_state_norm", output.clean_state.norm(dim=-1).mean())
        self.log(
            "train/clean_state_std",
            output.clean_state.reshape(-1, output.clean_state.shape[-1])
            .std(dim=0, unbiased=False)
            .mean(),
        )
        self._log_batch_visibility(batch)

        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.metrics_config.log_layer_weight_stats:
            self._log_layer_weight_stats()

    def _log_prismv2_visibility(
        self,
        batch: SequenceBatch,
        output: PRISMV2Output,
    ) -> None:
        self.log("train/embedding_norm", output.embedding.norm(dim=-1).mean())
        self.log(
            "train/posterior_belief_norm",
            output.posterior_belief.norm(dim=-1).mean(),
        )
        self.log(
            "train/prior_belief_norm",
            self._mean_or_zero(output.prior_belief.norm(dim=-1)),
        )
        self.log(
            "train/posterior_belief_std",
            output.posterior_belief.reshape(-1, output.posterior_belief.shape[-1])
            .std(dim=0, unbiased=False)
            .mean(),
        )
        self.log(
            "train/posterior_obs_prediction_norm",
            output.posterior_obs_prediction.norm(dim=-1).mean(),
        )
        self.log(
            "train/prior_obs_prediction_norm",
            self._mean_or_zero(output.prior_obs_prediction.norm(dim=-1)),
        )
        self._log_batch_visibility(batch)

        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.metrics_config.log_layer_weight_stats:
            self._log_layer_weight_stats()

    def _log_bwm_visibility(
        self,
        batch: SequenceBatch,
        output: BWMV2Output,
    ) -> None:
        self.log("train/plain_embedding_norm", output.encoded_z.norm(dim=-1).mean())
        self.log(
            "train/modulated_embedding_norm",
            self._mean_or_zero(output.modulated_z.norm(dim=-1)),
        )
        self.log("train/encoded_z_branch_norm", output.encoded_z.norm(dim=-1).mean())
        self.log(
            "train/modulated_z_branch_norm",
            self._mean_or_zero(output.modulated_z.norm(dim=-1)),
        )
        self.log("train/prior_bottleneck_std", output.prior.std(unbiased=False))
        self.log(
            "train/plain_embedding_std",
            output.encoded_z.reshape(-1, output.encoded_z.shape[-1])
            .std(dim=0, unbiased=False)
            .mean(),
        )
        self.log(
            "train/modulated_embedding_std",
            (
                output.modulated_z.reshape(-1, output.modulated_z.shape[-1])
                .std(dim=0, unbiased=False)
                .mean()
                if output.modulated_z.numel() > 0
                else output.encoded_z.new_tensor(0.0)
            ),
        )
        self._log_batch_visibility(batch)

        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        if self.metrics_config.log_layer_weight_stats:
            self._log_layer_weight_stats()

    def _log_hwm_visibility(
        self,
        batch: SequenceBatch,
        output: HWMOutput,
    ) -> None:
        self.log("train/embedding_norm", output.embedding.norm(dim=-1).mean())
        self.log(
            "train/embedding_std",
            output.embedding.reshape(-1, output.embedding.shape[-1])
            .std(dim=0, unbiased=False)
            .mean(),
        )
        if output.logvar.numel() > 0:
            self.log("train/logvar_std", output.logvar.std(unbiased=False))
        self._log_batch_visibility(batch)

        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
            if len(self.trainer.optimizers[0].param_groups) > 1:
                self.log(
                    "train/variance_lr",
                    self.trainer.optimizers[0].param_groups[1]["lr"],
                )
        if self.metrics_config.log_layer_weight_stats:
            self._log_layer_weight_stats()

    def _log_batch_visibility(self, batch: SequenceBatch) -> None:
        self.log("env/observation_density", batch.observations.float().mean())
        self.log("env/action_abs_mean", self._mean_or_zero(batch.actions.float().abs()))
        self.log("env/noop_rate", self._mean_or_zero(batch.noop_masks.float()))
        self.log(
            "env/actual_delta_abs_mean",
            self._mean_or_zero(batch.actual_deltas.float().abs()),
        )
        self._log_virtual_border_state_stats(batch)
        if self.metrics_config.log_gt_pos_distribution:
            self._log_ground_truth_position_distribution(batch)
        if self._is_swanlab_log_step():
            self._log_train_sequence_visualization(batch)

    def _log_virtual_border_state_stats(self, batch: SequenceBatch) -> None:
        env_config = self.model.config.env_config
        if not isinstance(env_config, GridWorld2DConfig):
            return
        virtual_border = env_config.train_virtual_border
        if virtual_border is None:
            return

        out_mask = self._virtual_border_out_state_mask(
            states=batch.states,
            width=env_config.width,
            virtual_border=virtual_border,
        )
        out_count = out_mask.sum().float()
        self.log("env/virtual_border_oob_state_count", out_count)
        self.log(
            "env/virtual_border_oob_state_ratio",
            self._mean_or_zero(out_mask.float()),
        )
        validation_border = env_config.validation_virtual_border
        if validation_border is None:
            return

        validation_mask = ~self._virtual_border_out_state_mask(
            states=batch.states,
            width=env_config.width,
            virtual_border=validation_border,
        )
        validation_count = validation_mask.sum().float()
        self.log("env/virtual_border_test_state_count", validation_count)
        self.log(
            "env/virtual_border_test_state_ratio",
            self._mean_or_zero(validation_mask.float()),
        )

    @staticmethod
    def _virtual_border_out_state_mask(
        states: torch.Tensor,
        width: int,
        virtual_border: tuple[int, int, int, int],
    ) -> torch.Tensor:
        top, left, bottom, right = virtual_border
        state_y = states // width
        state_x = states % width
        return (
            (state_y < top)
            | (state_y >= bottom)
            | (state_x < left)
            | (state_x >= right)
        )

    def on_train_batch_start(
        self,
        batch: SequenceBatch,
        batch_idx: int,
    ) -> None:
        del batch, batch_idx
        self._apply_sigreg_warmup_lr_factor()
        self.runtime_monitor.begin_train_batch(self.global_step, self.device)

    def on_train_batch_end(
        self,
        outputs: torch.Tensor,
        batch: SequenceBatch,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        self.runtime_monitor.end_optimizer_step(self.device)
        for name, value in self.runtime_monitor.end_train_batch(self.device).items():
            self.log(name, value, on_step=True, on_epoch=False)

    def on_before_backward(self, loss: torch.Tensor) -> None:
        del loss
        self.runtime_monitor.begin_backward(self.device)

    def on_after_backward(self) -> None:
        self.runtime_monitor.end_backward(self.device)

    def _apply_sigreg_warmup_lr_factor(self) -> None:
        factor = self.optimizer_config.sigreg_warmup_lr_factor
        if factor == 1.0 or not self.trainer.optimizers:
            return
        optimizer = self.trainer.optimizers[0]
        if self._optimizer_base_lrs is None:
            self._optimizer_base_lrs = [
                float(param_group["lr"]) for param_group in optimizer.param_groups
            ]
        if len(self._optimizer_base_lrs) != len(optimizer.param_groups):
            raise ValueError("optimizer param group count changed during training")
        lr_factor = factor if self._sigreg_only_warmup_active("train") else 1.0
        for base_lr, param_group in zip(
            self._optimizer_base_lrs,
            optimizer.param_groups,
            strict=True,
        ):
            param_group["lr"] = base_lr * lr_factor

    def on_before_optimizer_step(self, _optimizer: torch.optim.Optimizer) -> None:
        grad_norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        grad_norm = (
            torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)
            if grad_norms
            else torch.zeros((), device=self.device)
        )
        self.log("train/grad_norm", grad_norm)
        self.runtime_monitor.begin_optimizer_step(self.device)

    def _log_layer_weight_stats(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.endswith(".weight"):
                continue

            metric_name = name.replace(".", "/")
            values = parameter.detach().float()
            self.log(f"weights/{metric_name}/mean", values.mean())
            self.log(f"weights/{metric_name}/std", values.std(unbiased=False))
            self.log(f"weights/{metric_name}/norm", values.norm(2))

    def _log_ground_truth_position_distribution(self, batch: SequenceBatch) -> None:
        observation_size = self.model.config.observation_size
        states = batch.states.reshape(-1)
        if ((states < 0) | (states >= observation_size)).any():
            raise ValueError(f"batch states must be in [0, {observation_size - 1}]")

        counts = torch.bincount(states, minlength=observation_size).float()
        total = counts.sum()
        if total <= 0:
            raise ValueError("batch states must contain at least one item")

        frequencies = counts / total
        for position, count in enumerate(counts):
            self.log(f"env/gt_pos_count/{position}", count)
        self.log(
            "env/gt_pos_entropy",
            -(frequencies * frequencies.clamp_min(1e-8).log()).sum(),
        )

    def _is_swanlab_log_step(self) -> bool:
        if self.trainer is None or self.logger is None:
            return False
        log_every_n_steps = getattr(self.trainer, "log_every_n_steps", 1)
        return self.global_step % log_every_n_steps == 0

    def _log_train_sequence_visualization(self, batch: SequenceBatch) -> None:
        visualization = self.render_env.render(
            batch=self._renderable_sequence_batch(batch),
            batch_index=0,
        )
        if visualization is None:
            raise RuntimeError("sequence render must produce text in ansi mode")
        self._log_text_artifact(
            key="env/train_sequence",
            text=visualization,
            caption="batch_index=0",
        )

    @staticmethod
    def _renderable_sequence_batch(batch: SequenceBatch) -> SequenceBatch:
        if batch.observations.ndim not in (4, 6):
            return batch
        return SequenceBatch(
            observations=batch.observations[:, 0],
            states=batch.states[:, 0],
            noise_masks=batch.noise_masks[:, 0],
            actions=batch.actions[:, 0],
            actual_deltas=batch.actual_deltas[:, 0],
            noop_masks=batch.noop_masks[:, 0],
            obstacle_masks=batch.obstacle_masks[:, 0]
            if batch.obstacle_masks is not None
            else None,
        )

    def _log_prismv2_training_text(
        self,
        batch: SequenceBatch,
        output: PRISMV2Output,
        loss: torch.Tensor,
        belief_loss: torch.Tensor,
        obs_loss: torch.Tensor,
        posterior_obs_loss: torch.Tensor,
        prior_obs_loss: torch.Tensor,
        sigreg_loss: torch.Tensor,
        embedding_sigreg_loss: torch.Tensor,
        posterior_belief_sigreg_loss: torch.Tensor,
        sigreg_only_warmup_active: bool,
    ) -> None:
        if not self._is_swanlab_log_step():
            return

        text = self._format_prismv2_training_text(
            batch=batch,
            output=output,
            loss=loss,
            belief_loss=belief_loss,
            obs_loss=obs_loss,
            posterior_obs_loss=posterior_obs_loss,
            prior_obs_loss=prior_obs_loss,
            sigreg_loss=sigreg_loss,
            embedding_sigreg_loss=embedding_sigreg_loss,
            posterior_belief_sigreg_loss=posterior_belief_sigreg_loss,
            sigreg_only_warmup_active=sigreg_only_warmup_active,
            batch_index=0,
        )
        self._log_text_artifact(
            key="env/train_prismv2_diagnostics",
            text=text,
            caption="batch_index=0",
        )

    def _format_prismv2_training_text(
        self,
        batch: SequenceBatch,
        output: PRISMV2Output,
        loss: torch.Tensor,
        belief_loss: torch.Tensor,
        obs_loss: torch.Tensor,
        posterior_obs_loss: torch.Tensor,
        prior_obs_loss: torch.Tensor,
        sigreg_loss: torch.Tensor,
        embedding_sigreg_loss: torch.Tensor,
        posterior_belief_sigreg_loss: torch.Tensor,
        sigreg_only_warmup_active: bool,
        batch_index: int = 0,
    ) -> str:
        visualization = self.render_env.render(batch=batch, batch_index=batch_index)
        if visualization is None:
            raise RuntimeError("sequence render must produce text in ansi mode")

        states = batch.states[batch_index].detach().long().cpu()
        posterior = output.posterior_belief[batch_index].detach().float().cpu()
        prior = output.prior_belief[batch_index].detach().float().cpu()
        posterior_obs_prediction = (
            output.posterior_obs_prediction[batch_index].detach().float().cpu()
        )
        prior_obs_prediction = (
            output.prior_obs_prediction[batch_index].detach().float().cpu()
        )
        obs_encode = output.obs_target[batch_index].detach().float().cpu()
        sequence_length = states.shape[0]

        lines = [
            "train sequence",
            visualization,
            "",
            "loss calculation",
        ]
        if sigreg_only_warmup_active:
            lines.append(
                "total_loss="
                f"{self._as_float(loss):.6f} = "
                f"sigreg_weight({self.loss_config.sigreg_weight:.6f}) * "
                f"sigreg_loss({self._as_float(sigreg_loss):.6f})"
            )
        else:
            lines.append(
                "total_loss="
                f"{self._as_float(loss):.6f} = "
                f"prediction_weight({self.loss_config.prediction_weight:.6f}) * "
                f"belief_loss({self._as_float(belief_loss):.6f}) + "
                f"obs_weight({self.loss_config.obs_weight:.6f}) * "
                f"obs_loss({self._as_float(obs_loss):.6f}) + "
                f"sigreg_weight({self.loss_config.sigreg_weight:.6f}) * "
                f"sigreg_loss({self._as_float(sigreg_loss):.6f})"
            )
        lines.append(
            "obs_loss="
            f"{self._as_float(obs_loss):.6f} = "
            f"posterior_obs_weight({self.loss_config.posterior_obs_weight:.6f}) * "
            f"posterior_obs_loss({self._as_float(posterior_obs_loss):.6f}) + "
            f"prior_obs_weight({self.loss_config.prior_obs_weight:.6f}) * "
            f"prior_obs_loss({self._as_float(prior_obs_loss):.6f})"
        )
        lines.append(
            "sigreg_loss="
            f"{self._as_float(sigreg_loss):.6f} = "
            f"z_sigreg_weight({self.loss_config.z_sigreg_weight:.6f}) * "
            f"embedding_sigreg_loss({self._as_float(embedding_sigreg_loss):.6f}) + "
            "posterior_belief_sigreg_weight"
            f"({self.loss_config.posterior_belief_sigreg_weight:.6f}) * "
            "posterior_belief_sigreg_loss"
            f"({self._as_float(posterior_belief_sigreg_loss):.6f})"
        )
        lines.extend(
            [
                "",
                "per timestep",
                (
                    "t state char prior->post posterior_obs_pred->obs_encode "
                    "prior_obs_pred->next_obs_encode prior post "
                    "posterior_obs_pred prior_obs_pred obs_encode"
                ),
            ]
        )

        for step in range(sequence_length):
            char = self._format_observation_text(
                batch=batch,
                batch_index=batch_index,
                step=step,
            )
            prior_summary = "n/a"
            if step > 0 and prior.numel() > 0:
                prior_summary = self._format_sequence_match(
                    query=prior[step - 1],
                    candidates=posterior[1:],
                    candidate_steps=torch.arange(1, sequence_length),
                    candidate_states=states[1:],
                    true_candidate_index=step - 1,
                )
            obs_summary = self._format_sequence_match(
                query=posterior_obs_prediction[step],
                candidates=obs_encode,
                candidate_steps=torch.arange(sequence_length),
                candidate_states=states,
                true_candidate_index=step,
            )
            prior_obs_summary = "n/a"
            if step > 0 and prior_obs_prediction.numel() > 0:
                prior_obs_summary = self._format_sequence_match(
                    query=prior_obs_prediction[step - 1],
                    candidates=obs_encode[1:],
                    candidate_steps=torch.arange(1, sequence_length),
                    candidate_states=states[1:],
                    true_candidate_index=step - 1,
                )
            prior_preview = "n/a" if step == 0 else self._format_vector(prior[step - 1])
            prior_obs_preview = (
                "n/a"
                if step == 0
                else self._format_vector(prior_obs_prediction[step - 1])
            )
            lines.append(
                f"{step:02d} {int(states[step]):02d} {char} "
                f"{prior_summary} {obs_summary} {prior_obs_summary} "
                f"{prior_preview} {self._format_vector(posterior[step])} "
                f"{self._format_vector(posterior_obs_prediction[step])} "
                f"{prior_obs_preview} "
                f"{self._format_vector(obs_encode[step])}"
            )
        return "\n".join(lines)

    def _format_observation_text(
        self,
        batch: SequenceBatch,
        batch_index: int,
        step: int,
    ) -> str:
        observations = batch.observations[batch_index].detach().cpu()
        noise_masks = batch.noise_masks[batch_index].detach().bool().cpu()
        states = batch.states[batch_index].detach().long().cpu()
        if observations.ndim == 2:
            return "".join(
                RingWorldEnv._render_cells(
                    observation=observations[step],
                    noise_mask=noise_masks[step],
                    state=int(states[step]),
                )
            )
        if observations.ndim == 4:
            if batch.obstacle_masks is None:
                raise ValueError("2D observation text requires obstacle_masks")
            obstacle_masks = batch.obstacle_masks[batch_index].detach().bool().cpu()
            return "/".join(
                GridWorld2DEnv._render_grid(
                    noise_mask=noise_masks[step],
                    obstacle_mask=obstacle_masks[step],
                    state=int(states[step]),
                )
            )
        raise ValueError(f"unsupported observation rank {observations.ndim + 1}")

    def _format_sequence_match(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_steps: torch.Tensor,
        candidate_states: torch.Tensor,
        true_candidate_index: int,
    ) -> str:
        if candidates.numel() == 0:
            return "n/a"
        logits = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0),
            candidates,
            dim=-1,
        )
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_index = int(probs.argmax().item())
        true_prob = float(probs[true_candidate_index].item())
        return (
            f"top=t{int(candidate_steps[top_index])}/"
            f"s{int(candidate_states[top_index])}/"
            f"p{float(probs[top_index]):.3f};"
            f"true_p={true_prob:.3f}"
        )

    def _format_vector(self, vector: torch.Tensor, limit: int = 4) -> str:
        preview = ",".join(f"{float(value):+.2f}" for value in vector[:limit])
        return f"n{float(vector.norm()):.2f}[{preview}]"

    def _as_float(self, value: torch.Tensor) -> float:
        return float(value.detach().float().cpu().item())

    def _log_text_artifact(self, key: str, text: str, caption: str) -> None:
        logged = False
        for logger in self._iter_loggers():
            log_text = getattr(logger, "log_text", None)
            if log_text is None:
                continue
            log_text(
                key=key,
                texts=[text],
                step=self.global_step,
                caption=[caption],
            )
            logged = True
        if not logged and hasattr(getattr(self.logger, "experiment", None), "log"):
            self.logger.experiment.log(  # type: ignore[union-attr]
                {key: text},
                step=self.global_step,
            )

    def _iter_loggers(self) -> list[object]:
        loggers = getattr(self.trainer, "loggers", None)
        if loggers is not None:
            return list(loggers)
        if self.logger is None:
            return []
        return [self.logger]

    def _mean_or_zero(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        return values.mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if isinstance(self.model, HWM):
            variance_params = [self.model.predictor.global_logvar]
            variance_param_ids = {id(parameter) for parameter in variance_params}
            mean_params = [
                parameter
                for parameter in self.parameters()
                if id(parameter) not in variance_param_ids
            ]
            variance_lr = self.optimizer_config.variance_lr
            if variance_lr is None:
                variance_lr = self.optimizer_config.lr * 0.1
            return torch.optim.AdamW(
                [
                    {"params": mean_params, "lr": self.optimizer_config.lr},
                    {"params": variance_params, "lr": variance_lr},
                ],
                weight_decay=self.optimizer_config.weight_decay,
            )
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
