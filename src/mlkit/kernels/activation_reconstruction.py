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

SCALE_FORMAT_CODES = {"fp32": 0, "fp16": 1, "bf16": 2, "fp8": 3, "e8m0": 4}


@triton.jit
def rounded_e4m3(value):
    """Round a positive FP32 value to FP8 E4M3 with ties to even.

    Triton's own conversion rounds through FP16 first, which misplaces values
    that lie just beyond a tie. The quantum is derived from the exponent bits;
    values below the smallest normal number share the subnormal spacing.
    """
    clamped = tl.minimum(tl.maximum(value, 2.0**-9), 448.0)
    exponent = ((clamped.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    quantum_exponent = tl.maximum(exponent, -6) - 3
    quantum = ((quantum_exponent + 127) << 23).to(tl.float32, bitcast=True)
    reciprocal = ((127 - quantum_exponent) << 23).to(tl.float32, bitcast=True)
    return libdevice.nearbyint(clamped * reciprocal) * quantum


@triton.jit
def rounded_scale(value, format: tl.constexpr):
    if format == 1:
        return tl.minimum(tl.maximum(value, 2.0**-24), 65504.0).to(tl.float16).to(tl.float32)
    if format == 2:
        return value.to(tl.bfloat16).to(tl.float32)
    if format == 3:
        return rounded_e4m3(value)
    if format == 4:
        exponent = libdevice.nearbyint(tl.log2(tl.maximum(value, 2.0**-127)))
        return tl.exp2(tl.minimum(tl.maximum(exponent, -127.0), 127.0)).to(tl.float32)
    return value


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
        rounded = libdevice.nearbyint(normalized)
        rounded = tl.minimum(tl.maximum(rounded, minimum_value), maximum_value)
    else:
        lower = tl.full((tile,), 0, tl.int32)
        upper = tl.full((tile,), codebook_size, tl.int32)
        for _ in range(search_steps):
            middle = (lower + upper) // 2
            candidate = tl.load(values + middle, middle < codebook_size, other=float("inf"))
            move_right = candidate < normalized
            lower = tl.where(move_right, middle + 1, lower)
            upper = tl.where(move_right, upper, middle)
        upper = tl.minimum(lower, codebook_size - 1)
        lower = tl.maximum(upper - 1, 0)
        low_value = tl.load(values + lower)
        high_value = tl.load(values + upper)
        rounded = tl.where(
            tl.abs(normalized - low_value) <= tl.abs(normalized - high_value),
            low_value,
            high_value,
        )
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
