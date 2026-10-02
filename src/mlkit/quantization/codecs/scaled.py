"""Decoders for grouped scalar and vector codes with per-group scales."""

import torch
from torch import Tensor

from mlkit.kernels import scalar_decode
from mlkit.quantization.codecs.registry import codec
from mlkit.quantization.grids.lattice import device_points

FUSED_CODE_DTYPES = {torch.uint8, torch.int32}


def requires_gradients(*tensors: Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def decode_grouped(
    codes: Tensor,
    scales: Tensor,
    values: Tensor,
    group: int,
    zero: Tensor | None,
) -> Tensor:
    """Reconstruct codes whose column ``c`` uses scale ``c // group`` of its row.

    Inference on CUDA uses the fused kernel. Differentiable decoding, used when
    codec parameters are trained, keeps to tensor operations.
    """
    rows, width = codes.shape
    tile = scalar_decode.tile_for(group)
    fused = (
        codes.is_cuda
        and tile is not None
        and codes.dtype in FUSED_CODE_DTYPES
        and scales.dtype == torch.float32
        and values.dtype == torch.float32
        and (zero is None or zero.dtype == torch.float32)
        and codes.is_contiguous()
        and not requires_gradients(scales, values, zero)
    )
    if fused:
        assert tile is not None
        return scalar_decode.decode(
            codes,
            scales.contiguous(),
            values.contiguous(),
            None if zero is None else zero.contiguous(),
            group,
            tile,
        )
    reconstruction = values[codes.int()]
    if width % group == 0:
        grouped = reconstruction.reshape(rows, -1, group) * scales[:, :, None]
        if zero is not None:
            grouped = grouped + zero[:, :, None]
        return grouped.reshape(rows, width)
    columns = torch.arange(width, device=codes.device) // group
    reconstruction = reconstruction * scales[:, columns]
    return reconstruction if zero is None else reconstruction + zero[:, columns]


@codec("scaled", row_parameters=("scales", "zero"))
def decode_scaled(
    codes: Tensor,
    *,
    scales: Tensor,
    values: Tensor,
    group: int,
    offset: int = 0,
    zero: Tensor | None = None,
) -> Tensor:
    if offset == 0:
        return decode_grouped(codes, scales, values, group, zero)
    columns = torch.arange(offset, offset + codes.shape[1], device=codes.device) // group
    reconstruction = values[codes.int()] * scales[:, columns]
    return reconstruction if zero is None else reconstruction + zero[:, columns]


@codec("feedback", row_parameters=("scales", "zero"))
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
    if refit % group == 0:
        reconstruction = decode_grouped(codes, scales, values, group, zero)
    else:
        columns = torch.arange(codes.shape[1], device=codes.device)
        groups_per_region = (refit + group - 1) // group
        scale_indices = (columns // refit) * groups_per_region + (columns % refit) // group
        reconstruction = values[codes.int()] * scales[:, scale_indices]
        if zero is not None:
            reconstruction = reconstruction + zero[:, scale_indices]
    return reconstruction if permutation is None else reconstruction[:, permutation.argsort()]


@codec("vector_feedback", row_parameters=("scales", "zero"))
@codec("vector_scaled", row_parameters=("scales", "zero"))
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
    reconstruction = values[codes.int()].reshape(codes.shape[0], width) * scales[:, positions]
    if zero is not None:
        reconstruction = reconstruction + zero[:, positions]
    return reconstruction if permutation is None else reconstruction[:, permutation.argsort()]
