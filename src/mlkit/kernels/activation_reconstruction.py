"""Fused dynamic group scaling and scalar-grid reconstruction for online tensors."""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice


@triton.jit
def rounded_scale(value, format: tl.constexpr):
    if format == 1:
        return tl.minimum(tl.maximum(value, 2.0**-24), 65504.0).to(tl.float16).to(tl.float32)
    if format == 2:
        return value.to(tl.bfloat16).to(tl.float32)
    if format == 3:
        return tl.minimum(tl.maximum(value, 2.0**-9), 448.0).to(tl.float8e4nv).to(tl.float32)
    if format == 4:
        exponent = libdevice.nearbyint(tl.log2(tl.maximum(value, 2.0**-127)))
        return tl.exp2(tl.minimum(tl.maximum(exponent, -127.0), 127.0)).to(tl.float32)
    return value


@triton.jit
def reconstruct_groups(
    inputs, output, values,
    width: tl.constexpr, group: tl.constexpr, groups_per_row: tl.constexpr,
    maximum: tl.constexpr, minimum_value: tl.constexpr, maximum_value: tl.constexpr,
    format: tl.constexpr, asymmetric: tl.constexpr, integer: tl.constexpr,
    codebook_size: tl.constexpr, search_steps: tl.constexpr, tile: tl.constexpr,
):
    identifier = tl.program_id(0)
    row = identifier // groups_per_row
    columns = identifier % groups_per_row * group + tl.arange(0, tile)
    valid = (columns < width) & (tl.arange(0, tile) < group)
    samples = tl.load(inputs + row * width + columns, valid, other=0).to(tl.float32)
    zero = tl.full((), 0, tl.float32)
    if asymmetric:
        minimum = tl.min(tl.where(tl.arange(0, tile) < group, samples, float("inf")), 0)
        maximum_group = tl.max(tl.where(tl.arange(0, tile) < group, samples, -float("inf")), 0)
        raw_scale = tl.maximum(maximum_group - minimum, 1.0e-12) / (maximum_value - minimum_value)
        zero = minimum - minimum_value * raw_scale
        scale = rounded_scale(raw_scale, format)
        magnitude = rounded_scale(tl.maximum(tl.abs(zero), 1.0e-12), format)
        zero = tl.where(zero < 0, -magnitude, tl.where(zero > 0, magnitude, 0))
    else:
        scale = rounded_scale(tl.maximum(tl.max(tl.abs(samples), 0), 1.0e-12) / maximum, format)
    normalized = (samples - zero) / scale
    if integer:
        rounded = tl.minimum(tl.maximum(libdevice.nearbyint(normalized), minimum_value),
                             maximum_value)
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
        rounded = tl.where(tl.abs(normalized - low_value) <= tl.abs(normalized - high_value),
                           low_value, high_value)
    tl.store(output + row * width + columns, rounded * scale + zero, valid)


def reconstruct(
    inputs: Tensor, values: Tensor, group: int, scale_format: str, asymmetric: bool,
    integer: bool, minimum: float, maximum: float, absolute_maximum: float,
) -> Tensor:
    formats = {"fp32": 0, "fp16": 1, "bf16": 2, "fp8": 3, "e8m0": 4}
    output = torch.empty_like(inputs)
    group = min(group, inputs.shape[1])
    groups_per_row = triton.cdiv(inputs.shape[1], group)
    reconstruct_groups[(len(inputs) * groups_per_row,)](
        inputs, output, values, inputs.shape[1], group, groups_per_row,
        absolute_maximum, minimum, maximum, formats[scale_format], asymmetric, integer,
        values.numel(), math.ceil(math.log2(values.numel() + 1)),
        triton.next_power_of_2(group), num_warps=4, enable_fp_fusion=False,
    )
    return output
