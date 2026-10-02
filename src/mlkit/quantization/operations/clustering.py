"""Deterministic weighted Lloyd iterations for learned codebooks."""

import torch
from torch import Tensor

from mlkit.kernels import scalar_clustering
from mlkit.quantization.operations.search import nearest


def kmeans(
    value: Tensor,
    k: int,
    *,
    weights: Tensor | None = None,
    iters: int = 20,
    seed: int = 0,
    chunk: int | None = None,
) -> Tensor:
    """Deterministic weighted Lloyd iterations for scalar or vector codebooks.

    Scalar codebooks of at most 256 centers are fitted on CUDA by a fused kernel.
    A cluster that receives no samples keeps its center.
    """
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
    if scalar and samples.is_cuda and k <= scalar_clustering.MAXIMUM_CLUSTERS:
        return fused_scalar_kmeans(
            samples[:, 0].contiguous(),
            None if weights is None else importance.contiguous(),
            centers[:, 0],
            iters,
        )
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


def fused_scalar_kmeans(
    samples: Tensor,
    weights: Tensor | None,
    centers: Tensor,
    iterations: int,
) -> Tensor:
    """Lloyd iterations over ascending scalar centers with one kernel pass each."""
    centers = centers.sort().values
    for _ in range(iterations):
        sums, totals = scalar_clustering.accumulate(samples, weights, centers)
        centers = torch.where(totals > 0, sums / totals.clamp_min(1e-12), centers).sort().values
    return centers
