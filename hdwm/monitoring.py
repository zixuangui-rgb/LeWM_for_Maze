"""Runtime timing and resource monitoring for training diagnostics."""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator

import torch

from hdwm.config import RuntimeMonitorConfig

try:
    import psutil
except ImportError:  # pragma: no cover - optional diagnostic dependency
    psutil = None  # type: ignore[assignment]


class RuntimeMonitor:
    """Collect sparse timing and resource metrics during training."""

    def __init__(self, config: RuntimeMonitorConfig) -> None:
        self.config = config
        self.metrics: dict[str, float] = {}
        self._active = False
        self._step_start_time: float | None = None
        self._last_batch_end_time: float | None = None
        self._backward_start_time: float | None = None
        self._optimizer_step_start_time: float | None = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        if self._process is not None:
            self._process.cpu_percent(interval=None)

    def begin_train_batch(self, global_step: int, device: torch.device) -> None:
        """Start a monitored train batch when the configured cadence matches."""

        self.metrics = {}
        self._active = (
            self.config.enabled
            and global_step >= self.config.warmup_steps
            and global_step % self.config.log_every_n_steps == 0
        )
        if not self._active:
            self._step_start_time = None
            return

        now = time.perf_counter()
        if self._last_batch_end_time is not None:
            self.metrics["time/data_wait_ms"] = (now - self._last_batch_end_time) * 1000
        self._synchronize(device)
        self._step_start_time = time.perf_counter()

    def end_train_batch(self, device: torch.device) -> dict[str, float]:
        """Finish a monitored train batch and return metrics ready for logging."""

        now = time.perf_counter()
        self._last_batch_end_time = now
        if not self._active:
            return {}

        self._synchronize(device)
        end_time = time.perf_counter()
        if self._step_start_time is not None:
            self.metrics["time/train_step_ms"] = (
                end_time - self._step_start_time
            ) * 1000
        self.metrics.update(self.system_metrics(device))
        metrics = dict(self.metrics)
        self._active = False
        return metrics

    @contextlib.contextmanager
    def time_block(self, metric_name: str, device: torch.device) -> Iterator[None]:
        """Record elapsed milliseconds for a named block when monitoring is active."""

        if not self._active:
            yield
            return

        self._synchronize(device)
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.metrics[f"time/{metric_name}_ms"] = elapsed_ms

    def begin_backward(self, device: torch.device) -> None:
        if not self._active:
            return
        self._synchronize(device)
        self._backward_start_time = time.perf_counter()

    def end_backward(self, device: torch.device) -> None:
        if not self._active or self._backward_start_time is None:
            return
        self._synchronize(device)
        self.metrics["time/backward_ms"] = (
            time.perf_counter() - self._backward_start_time
        ) * 1000
        self._backward_start_time = None

    def begin_optimizer_step(self, device: torch.device) -> None:
        if not self._active:
            return
        self._synchronize(device)
        self._optimizer_step_start_time = time.perf_counter()

    def end_optimizer_step(self, device: torch.device) -> None:
        if not self._active or self._optimizer_step_start_time is None:
            return
        self._synchronize(device)
        self.metrics["time/optimizer_step_ms"] = (
            time.perf_counter() - self._optimizer_step_start_time
        ) * 1000
        self._optimizer_step_start_time = None

    def system_metrics(self, device: torch.device) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.config.log_system:
            rss_mb = self._process_rss_mb()
            if rss_mb is not None:
                metrics["system/process_rss_mb"] = rss_mb
            if self._process is not None:
                metrics["system/cpu_percent"] = float(
                    self._process.cpu_percent(interval=None)
                )

        if (
            self.config.log_cuda_memory
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            metrics["system/cuda_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(device) / 1024**2
            )
            metrics["system/cuda_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(device) / 1024**2
            )
        if (
            self.config.log_gpu_utilization
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            metrics.update(self._gpu_utilization_metrics(device))
        return metrics

    def _gpu_utilization_metrics(self, device: torch.device) -> dict[str, float]:
        metrics: dict[str, float] = {}
        gpu_utilization = self._cuda_metric("utilization", device)
        if gpu_utilization is not None:
            metrics["system/gpu_utilization_percent"] = gpu_utilization
        gpu_memory_utilization = self._cuda_metric("memory_usage", device)
        if gpu_memory_utilization is not None:
            metrics["system/gpu_memory_utilization_percent"] = gpu_memory_utilization
        return metrics

    def _cuda_metric(self, metric_name: str, device: torch.device) -> float | None:
        metric = getattr(torch.cuda, metric_name, None)
        if metric is None:
            return None
        try:
            return float(metric(device))
        except (AttributeError, ImportError, OSError, RuntimeError):
            return None

    def _synchronize(self, device: torch.device) -> None:
        if (
            self.config.synchronize_device
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(device)

    def _process_rss_mb(self) -> float | None:
        if self._process is not None:
            return float(self._process.memory_info().rss / 1024**2)

        statm_path = "/proc/self/statm"
        if not os.path.exists(statm_path):
            return None
        with open(statm_path) as statm_file:
            fields = statm_file.read().split()
        if len(fields) < 2:
            return None
        return float(int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2)
