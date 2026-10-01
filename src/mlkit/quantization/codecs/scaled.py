"""Decoders for grouped scalar and vector codes with per-group scales."""

import torch
from torch import Tensor

from mlkit.quantization.codecs.registry import codec
from mlkit.quantization.grids.lattice import device_points


@codec("scaled")
def decode_scaled(
    codes: Tensor,
    *,
    scales: Tensor,
    values: Tensor,
    group: int,
    offset: int = 0,
    zero: Tensor | None = None,
) -> Tensor:
    columns = torch.arange(offset, offset + codes.shape[1], device=codes.device) // group
    reconstruction = values[codes.long()] * scales[:, columns]
    return reconstruction if zero is None else reconstruction + zero[:, columns]


@codec("feedback")
def decode_feedback(
    codes: Tensor,
    *,
    scales: Tensor,
    values: Tensor,
    group: int,
    refit: int,
    zero: Tensor | None = None,
    permutation: Tensor | None = None,
) -> Tensor:
    columns = torch.arange(codes.shape[1], device=codes.device)
    groups_per_region = (refit + group - 1) // group
    scale_indices = (columns // refit) * groups_per_region + (columns % refit) // group
    reconstruction = values[codes.long()] * scales[:, scale_indices]
    if zero is not None:
        reconstruction = reconstruction + zero[:, scale_indices]
    return reconstruction if permutation is None else reconstruction[:, permutation.argsort()]


@codec("vector_feedback")
@codec("vector_scaled")
def decode_vector_scaled(
    codes: Tensor,
    *,
    scales: Tensor,
    values: Tensor | None,
    group: int,
    dim: int,
    offset: int = 0,
    zero: Tensor | None = None,
    refit: int | None = None,
    permutation: Tensor | None = None,
) -> Tensor:
    if values is None:
        values = device_points(codes.device)
    width = codes.shape[1] * dim
    columns = torch.arange(offset, offset + width, device=codes.device)
    positions = columns // group if refit is None else (
        (columns // refit) * ((refit + group - 1) // group) + (columns % refit) // group
    )
    reconstruction = values[codes.long()].reshape(codes.shape[0], width) * scales[:, positions]
    if zero is not None:
        reconstruction = reconstruction + zero[:, positions]
    return reconstruction if permutation is None else reconstruction[:, permutation.argsort()]
