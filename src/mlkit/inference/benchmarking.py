"""Reproducible latency measurements with synchronized CUDA timing."""

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from mlkit.calibration.session import forward_batch
from mlkit.models.model import Model
from mlkit.timing import synchronize


@dataclass
class BenchmarkResult:
    median_ms: float
    minimum_ms: float
    percentile_95_ms: float
    repetitions: int
    device: str
    peak_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    operation: Callable[[], Any],
    *,
    warmup: int = 10,
    repetitions: int = 50,
    inference_mode: bool = True,
) -> BenchmarkResult:
    """Synchronize outside each timed interval; exclude warmup and compilation.

    Operations that train parameters, such as conversion with block passes, need
    ``inference_mode=False``.
    """
    selected_device = torch.device("cuda", torch.cuda.current_device())
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions must be positive")
    samples = []
    with torch.inference_mode(inference_mode):
        for _ in range(warmup):
            operation()
        synchronize(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
        for _ in range(repetitions):
            synchronize(selected_device)
            start = time.perf_counter()
            operation()
            synchronize(selected_device)
            samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    memory = torch.cuda.max_memory_allocated(selected_device)
    return BenchmarkResult(
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        percentile_95_ms=ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        repetitions=repetitions,
        device=str(selected_device),
        peak_memory_bytes=memory,
    )


def benchmark_model(model: Model | nn.Module, batch: Any, **options: Any) -> BenchmarkResult:
    wrapped = model if isinstance(model, Model) else Model(model)
    return benchmark(lambda: forward_batch(wrapped.module, batch), **options)
