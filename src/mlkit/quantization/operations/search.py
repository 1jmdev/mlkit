"""Exact nearest-entry search over scalar and vector codebooks with bounded working memory."""

import torch
from torch import Tensor

from mlkit.kernels import scalar_encode


def snap(value: Tensor, codebook: Tensor) -> Tensor:
    """Find the nearest scalar entry without an elements-by-codebook temporary.

    Ties select the lower entry. FP32 CUDA tensors that do not track gradients
    are rounded by a fused kernel.
    """
    codebook = codebook.to(device=value.device, dtype=value.dtype).flatten().sort().values
    if codebook.numel() == 0:
        raise ValueError("codebook cannot be empty")
    differentiable = torch.is_grad_enabled() and (value.requires_grad or codebook.requires_grad)
    if value.is_cuda and value.dtype == torch.float32 and not differentiable and value.numel():
        return scalar_encode.snap(value.contiguous(), codebook)
    upper = torch.searchsorted(codebook, value.contiguous()).clamp_max(codebook.numel() - 1)
    lower = (upper - 1).clamp_min(0)
    return torch.where(
        (value - codebook[lower]).abs() <= (value - codebook[upper]).abs(),
        codebook[lower],
        codebook[upper],
    )


DISTANCE_BUDGET = 2**23
"""Distances materialized at once by the vector search: 32 MiB of FP32 values."""


def nearest(
    value: Tensor,
    codebook: Tensor,
    *,
    chunk: int | None = None,
    codebook_chunk: int = 4096,
    return_indices: bool = False,
) -> Tensor:
    """Exact vector search, tiled across samples and codewords.

    ``chunk`` bounds the samples compared at once; by default it is chosen so that
    one tile of distances stays within a fixed memory budget.
    """
    if value.ndim != 2 or codebook.ndim != 2 or value.shape[1] != codebook.shape[1]:
        raise ValueError("value and codebook must be matrices with equal vector dimensions")
    if not codebook.shape[0] or codebook_chunk <= 0 or (chunk is not None and chunk <= 0):
        raise ValueError("codebook and chunk sizes must be nonempty and positive")
    codebook = codebook.to(value.device)
    if value.shape[1] == 1:
        indices = nearest_scalar(value[:, 0], codebook[:, 0])
        return indices if return_indices else codebook[indices].to(value.dtype)
    if chunk is None:
        chunk = max(1, DISTANCE_BUDGET // min(codebook_chunk, len(codebook)))
    tiles = []
    for offset in range(0, len(codebook), codebook_chunk):
        candidates = codebook[offset : offset + codebook_chunk].float()
        tiles.append((offset, candidates.T.contiguous(), candidates.square().sum(1)[None, :]))
    index_parts = []
    for samples in value.float().split(chunk):
        minimum = torch.full((len(samples),), float("inf"), device=value.device)
        indices = torch.zeros(len(samples), dtype=torch.long, device=value.device)
        for offset, transposed, norms in tiles:
            costs, local_indices = torch.addmm(norms, samples, transposed, alpha=-2).min(1)
            improved = costs < minimum
            indices = torch.where(improved, local_indices + offset, indices)
            minimum = torch.minimum(minimum, costs)
        index_parts.append(indices)
    indices = torch.cat(index_parts) if index_parts else torch.empty(
        0, dtype=torch.long, device=value.device
    )
    return indices if return_indices else codebook[indices].to(value.dtype)


def nearest_scalar(value: Tensor, codebook: Tensor) -> Tensor:
    """Indices of the nearest scalar codewords; ties select the lowest original index."""
    ordered, permutation = codebook.float().sort(stable=True)
    samples = value.float().contiguous()
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
    return torch.where(select_left, left_indices, right_indices)
