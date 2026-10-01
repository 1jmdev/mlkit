"""Kernel helpers that round scales and normalized values exactly as the tensor reference does.

Division by a scale uses the round-to-nearest intrinsic at every call site,
because inputs that are exact rounding ties must select the same code on both
paths. Scale storage rounding reproduces the conversions of ``store_scale``.
"""

import triton
import triton.language as tl
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
    """Round a positive scale to its storage format; ``format`` indexes SCALE_FORMAT_CODES."""
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
def nearest_integer(normalized, minimum_value: tl.constexpr, maximum_value: tl.constexpr):
    """The nearest integer of a consecutive grid, with ties to even and saturation."""
    rounded = libdevice.nearbyint(normalized)
    return tl.minimum(tl.maximum(rounded, minimum_value), maximum_value)


@triton.jit
def nearest_codebook_index(
    normalized,
    values,
    codebook_size: tl.constexpr,
    search_steps: tl.constexpr,
):
    """Index of the nearest entry of an ascending codebook; ties select the lower entry."""
    lower = tl.zeros_like(normalized).to(tl.int32)
    upper = lower + codebook_size
    for _ in range(search_steps):
        middle = (lower + upper) // 2
        candidate = tl.load(values + middle, middle < codebook_size, other=float("inf"))
        move_right = candidate < normalized
        lower = tl.where(move_right, middle + 1, lower)
        upper = tl.where(move_right, upper, middle)
    above = tl.minimum(lower, codebook_size - 1)
    below = tl.maximum(above - 1, 0)
    below_distance = tl.abs(normalized - tl.load(values + below))
    above_distance = tl.abs(normalized - tl.load(values + above))
    return tl.where(below_distance <= above_distance, below, above)
