"""Fused dynamic group scaling and scalar-grid reconstruction for online tensors.

The kernel reproduces ``Scaled`` rounding bit for bit. Division uses the
round-to-nearest intrinsic because half-precision inputs frequently land on
exact rounding ties, where an approximate quotient selects the wrong code.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

from mlkit.kernels.scalar_rounding import (
    SCALE_FORMAT_CODES,
    nearest_codebook_index,
    nearest_integer,
    rounded_scale,
)


@triton.jit
def reconstruct_groups(
    inputs,
    output,
    values,
    width,
    group,
    groups_per_row,
    scale_reciprocal: tl.constexpr,
    minimum_value: tl.constexpr,
    maximum_value: tl.constexpr,
    format: tl.constexpr,
    asymmetric: tl.constexpr,
    integer: tl.constexpr,
    codebook_size: tl.constexpr,
    search_steps: tl.constexpr,
    tile: tl.constexpr,
):
    identifier = tl.program_id(0)
    row = (identifier // groups_per_row).to(tl.int64)
    offsets = tl.arange(0, tile)
    columns = identifier % groups_per_row * group + offsets
    valid = (columns < width) & (offsets < group)
    samples = tl.load(inputs + row * width + columns, valid, other=0).to(tl.float32)
    zero = tl.full((), 0, tl.float32)
    if asymmetric:
        minimum = tl.min(tl.where(valid, samples, float("inf")), 0)
        maximum = tl.max(tl.where(valid, samples, -float("inf")), 0)
        raw_scale = tl.maximum(maximum - minimum, 1.0e-12) * scale_reciprocal
        zero = minimum - minimum_value * raw_scale
        scale = rounded_scale(raw_scale, format)
        magnitude = rounded_scale(tl.maximum(tl.abs(zero), 1.0e-12), format)
        zero = tl.where(zero < 0, -magnitude, tl.where(zero > 0, magnitude, 0))
    else:
        magnitude = tl.maximum(tl.max(tl.abs(samples), 0), 1.0e-12)
        scale = rounded_scale(magnitude * scale_reciprocal, format)
    normalized = libdevice.div_rn(samples - zero, scale)
    if integer:
        rounded = nearest_integer(normalized, minimum_value, maximum_value)
    else:
        index = nearest_codebook_index(normalized, values, codebook_size, search_steps)
        rounded = tl.load(values + index)
    tl.store(output + row * width + columns, rounded * scale + zero, valid)


def reconstruct(
    inputs: Tensor,
    values: Tensor,
    group: int,
    scale_format: str,
    asymmetric: bool,
    integer: bool,
    minimum: float,
    maximum: float,
    scale_reciprocal: float,
) -> Tensor:
    """Round ``inputs`` to a scaled scalar grid; ``scale_reciprocal`` maps a range to a scale."""
    output = torch.empty_like(inputs)
    group = min(group, inputs.shape[1])
    groups_per_row = triton.cdiv(inputs.shape[1], group)
    reconstruct_groups[(len(inputs) * groups_per_row,)](
        inputs,
        output,
        values,
        inputs.shape[1],
        group,
        groups_per_row,
        scale_reciprocal,
        minimum,
        maximum,
        SCALE_FORMAT_CODES[scale_format],
        asymmetric,
        integer,
        values.numel(),
        math.ceil(math.log2(values.numel() + 1)),
        triton.next_power_of_2(group),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
