"""Exact nearest-entry search over scalar and vector codebooks with bounded working memory."""

import torch
from torch import Tensor


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
