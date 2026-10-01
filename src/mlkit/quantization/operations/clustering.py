"""Deterministic weighted Lloyd iterations for learned codebooks."""

import torch
from torch import Tensor

from mlkit.quantization.operations.search import nearest


def kmeans(
    value: Tensor,
    k: int,
    *,
    weights: Tensor | None = None,
    iters: int = 20,
    seed: int = 0,
    chunk: int = 1024,
) -> Tensor:
    """Deterministic weighted Lloyd iterations for scalar or vector codebooks."""
    scalar = value.ndim == 1
    samples = value.reshape(-1, 1) if scalar else value
    if samples.ndim != 2 or not 1 <= k <= len(samples) or iters < 1:
        raise ValueError("kmeans requires a matrix, 1 <= k <= sample count, and iters >= 1")
    samples = samples.float()
    if not torch.isfinite(samples).all():
        raise ValueError("kmeans samples must be finite")
    generator = torch.Generator(device=value.device).manual_seed(seed)
    initial = torch.randperm(len(samples), device=value.device, generator=generator)[:k]
    centers = samples[initial].clone()
    importance = torch.ones(len(samples), device=value.device) if weights is None else weights
    importance = importance.reshape(-1).to(device=value.device, dtype=torch.float32)
    if (
        len(importance) != len(samples)
        or (importance < 0).any()
        or not torch.isfinite(importance).all()
    ):
        raise ValueError("weights must be finite and nonnegative with one value per sample")
    for _ in range(iters):
        assignments = nearest(samples, centers, chunk=chunk, return_indices=True)
        order = assignments.argsort(stable=True)
        cluster_sizes = torch.bincount(assignments, minlength=k)
        ordered_importance = importance[order]
        totals = torch.segment_reduce(
            samples[order] * ordered_importance[:, None], "sum", lengths=cluster_sizes
        )
        cluster_weights = torch.segment_reduce(
            ordered_importance, "sum", lengths=cluster_sizes
        )
        centers = torch.where(
            cluster_weights[:, None] > 0,
            totals / cluster_weights[:, None].clamp_min(1e-12),
            centers,
        )
    return centers[:, 0].sort().values if scalar else centers
