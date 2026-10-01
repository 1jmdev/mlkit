"""Numerical building blocks with bounded working memory."""

import math

import torch
from torch import Tensor

from mlkit.quantization.context import Ctx


def groups(weight: Tensor, group: int | None) -> Tensor:
    if weight.ndim != 2:
        raise ValueError("groups requires a two-dimensional matrix")
    size = weight.shape[1] if group is None else group
    if size <= 0 or weight.shape[1] % size:
        raise ValueError(
            f"group size {size} must divide the row width {weight.shape[1]}; "
            "use scaled for automatic final-group padding"
        )
    return weight.reshape(-1, size)


def absmax(value: Tensor, qmax: float = 1.0) -> Tensor:
    if qmax <= 0:
        raise ValueError("qmax must be positive")
    return value.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax


def snap(value: Tensor, codebook: Tensor) -> Tensor:
    """Find the nearest scalar entry without an elements-by-codebook temporary."""
    codebook = codebook.to(device=value.device, dtype=value.dtype).flatten().sort().values
    if codebook.numel() == 0:
        raise ValueError("codebook cannot be empty")
    upper = torch.searchsorted(codebook, value.contiguous()).clamp_max(codebook.numel() - 1)
    lower = (upper - 1).clamp_min(0)
    return torch.where(
        (value - codebook[lower]).abs() <= (value - codebook[upper]).abs(),
        codebook[lower],
        codebook[upper],
    )


def nearest(
    value: Tensor,
    codebook: Tensor,
    *,
    chunk: int = 1024,
    codebook_chunk: int = 4096,
    return_indices: bool = False,
) -> Tensor:
    """Exact vector search, tiled across samples and codewords."""
    if value.ndim != 2 or codebook.ndim != 2 or value.shape[1] != codebook.shape[1]:
        raise ValueError("value and codebook must be matrices with equal vector dimensions")
    if not codebook.shape[0] or chunk <= 0 or codebook_chunk <= 0:
        raise ValueError("codebook and chunk sizes must be nonempty and positive")
    codebook = codebook.to(value.device)
    if value.shape[1] == 1:
        ordered, permutation = codebook[:, 0].float().sort(stable=True)
        samples = value[:, 0].float().contiguous()
        right = torch.searchsorted(ordered, samples).clamp_max(len(ordered) - 1)
        left = (right - 1).clamp_min(0)
        first_occurrences = torch.searchsorted(ordered, ordered)
        left_indices = permutation[first_occurrences[left]]
        right_indices = permutation[first_occurrences[right]]
        left_distance = (samples - ordered[left]).abs()
        right_distance = (samples - ordered[right]).abs()
        select_left = (left_distance < right_distance) | (
            (left_distance == right_distance) & (left_indices < right_indices)
        )
        indices = torch.where(select_left, left_indices, right_indices)
        return indices if return_indices else codebook[indices].to(value.dtype)
    index_parts = []
    for samples in value.float().split(chunk):
        minimum = torch.full((len(samples),), float("inf"), device=value.device)
        indices = torch.zeros(len(samples), dtype=torch.long, device=value.device)
        for offset in range(0, len(codebook), codebook_chunk):
            candidates = codebook[offset : offset + codebook_chunk].float()
            distances = (
                samples.square().sum(1, keepdim=True)
                + candidates.square().sum(1)[None, :]
                - 2 * samples @ candidates.T
            )
            costs, local_indices = distances.min(1)
            improved = costs < minimum
            indices = torch.where(improved, local_indices + offset, indices)
            minimum = torch.minimum(minimum, costs)
        index_parts.append(indices)
    indices = torch.cat(index_parts) if index_parts else torch.empty(
        0, dtype=torch.long, device=value.device
    )
    return indices if return_indices else codebook[indices].to(value.dtype)


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
    if (len(importance) != len(samples) or (importance < 0).any()
            or not torch.isfinite(importance).all()):
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


def hadamard(value: Tensor, *, normalize: bool = True) -> Tensor:
    """Fast Walsh-Hadamard transform along the final dimension."""
    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("hadamard currently requires a power-of-two final dimension")
    output = value.clone()
    stride = 1
    while stride < width:
        pairs = output.reshape(*value.shape[:-1], -1, 2, stride)
        first, second = pairs[..., 0, :], pairs[..., 1, :]
        output = torch.stack((first + second, first - second), dim=-2).reshape_as(value)
        stride *= 2
    return output / math.sqrt(width) if normalize else output


def rht(value: Tensor, *, seed: int = 0, inverse: bool = False) -> Tensor:
    from mlkit.quantization.rotations import randomized_transform

    return randomized_transform(value, seed=seed, inverse=inverse)


def proxy_loss(weight: Tensor, reconstruction: Tensor, ctx: Ctx | None = None) -> Tensor:
    """Mean squared output error per output channel (or weight MSE without H)."""
    error = weight.float() - reconstruction.float()
    if ctx is None:
        return error.square().mean()
    return ((error @ ctx.H.to(error.device)) * error).sum() / weight.shape[0]
