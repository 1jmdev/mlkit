"""Walsh-Hadamard transform along the last dimension with several stages per pass.

A butterfly stage with stride ``s`` replaces each pair ``(x[i], x[i + s])`` by its
sum and difference. Three consecutive stages act on independent groups of eight
elements, so one pass loads eight values per group, applies all three stages in
registers and stores them. The stages run in ascending stride order, which
reproduces the element-wise arithmetic of a stage-by-stage transform exactly.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 512
RUNTIME_ARGUMENTS = ["source", "destination", "groups", "stride", "stride_bits", "scale"]


@triton.jit(do_not_specialize=RUNTIME_ARGUMENTS, do_not_specialize_on_alignment=RUNTIME_ARGUMENTS)
def hadamard_pass(
    source,
    destination,
    groups,
    stride,
    stride_bits,
    scale,
    stages: tl.constexpr,
    scaled: tl.constexpr,
    block: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = group < groups
    low = group & (stride - 1)
    base = ((group >> stride_bits) << (stride_bits + stages)) + low
    x0 = tl.load(source + base, valid, other=0)
    x1 = tl.load(source + base + stride, valid, other=0)
    if stages == 1:
        y0 = x0 + x1
        y1 = x0 - x1
        if scaled:
            y0 *= scale
            y1 *= scale
        tl.store(destination + base, y0, valid)
        tl.store(destination + base + stride, y1, valid)
    else:
        x2 = tl.load(source + base + 2 * stride, valid, other=0)
        x3 = tl.load(source + base + 3 * stride, valid, other=0)
        a0 = x0 + x1
        a1 = x0 - x1
        a2 = x2 + x3
        a3 = x2 - x3
        b0 = a0 + a2
        b1 = a1 + a3
        b2 = a0 - a2
        b3 = a1 - a3
        if stages == 2:
            if scaled:
                b0 *= scale
                b1 *= scale
                b2 *= scale
                b3 *= scale
            tl.store(destination + base, b0, valid)
            tl.store(destination + base + stride, b1, valid)
            tl.store(destination + base + 2 * stride, b2, valid)
            tl.store(destination + base + 3 * stride, b3, valid)
        else:
            x4 = tl.load(source + base + 4 * stride, valid, other=0)
            x5 = tl.load(source + base + 5 * stride, valid, other=0)
            x6 = tl.load(source + base + 6 * stride, valid, other=0)
            x7 = tl.load(source + base + 7 * stride, valid, other=0)
            a4 = x4 + x5
            a5 = x4 - x5
            a6 = x6 + x7
            a7 = x6 - x7
            b4 = a4 + a6
            b5 = a5 + a7
            b6 = a4 - a6
            b7 = a5 - a7
            c0 = b0 + b4
            c1 = b1 + b5
            c2 = b2 + b6
            c3 = b3 + b7
            c4 = b0 - b4
            c5 = b1 - b5
            c6 = b2 - b6
            c7 = b3 - b7
            if scaled:
                c0 *= scale
                c1 *= scale
                c2 *= scale
                c3 *= scale
                c4 *= scale
                c5 *= scale
                c6 *= scale
                c7 *= scale
            tl.store(destination + base, c0, valid)
            tl.store(destination + base + stride, c1, valid)
            tl.store(destination + base + 2 * stride, c2, valid)
            tl.store(destination + base + 3 * stride, c3, valid)
            tl.store(destination + base + 4 * stride, c4, valid)
            tl.store(destination + base + 5 * stride, c5, valid)
            tl.store(destination + base + 6 * stride, c6, valid)
            tl.store(destination + base + 7 * stride, c7, valid)


def transform(value: Tensor, *, normalize: bool) -> Tensor:
    """Transform a contiguous FP32 CUDA tensor whose last dimension is a power of two."""
    width = value.shape[-1]
    output = torch.empty_like(value)
    if width == 1:
        return output.copy_(value)
    remaining = width.bit_length() - 1
    scale = float(torch.tensor(1.0) / torch.tensor(math.sqrt(width)))
    source = value
    stride_bits = 0
    while remaining:
        stages = min(3, remaining)
        remaining -= stages
        groups = value.numel() >> stages
        hadamard_pass[(triton.cdiv(groups, BLOCK),)](
            source,
            output,
            groups,
            1 << stride_bits,
            stride_bits,
            scale,
            stages,
            normalize and remaining == 0,
            BLOCK,
            enable_fp_fusion=False,
        )
        source = output
        stride_bits += stages
    return output
