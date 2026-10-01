"""Rounding of group scales and offsets to their declared storage formats."""

import torch
from torch import Tensor

from mlkit.quantization.grids import Grid

SCALE_FORMAT_BITS = {"fp32": 32, "fp16": 16, "bf16": 16, "fp8": 8, "e8m0": 8}


def store_scale(value: Tensor, format: str | Grid) -> tuple[Tensor, int]:
    if isinstance(format, Grid):
        if format.dim != 1:
            raise ValueError("scale grids must be scalar")
        return format(value).clamp_min(torch.finfo(torch.float32).tiny), format.bits
    if format == "fp32":
        return value.float(), 32
    if format == "fp16":
        return value.clamp(2**-24, 65504).half().float(), 16
    if format == "bf16":
        return value.bfloat16().float(), 16
    if format == "fp8":
        return value.clamp(2**-9, 448).to(torch.float8_e4m3fn).float(), 8
    if format == "e8m0":
        return torch.pow(2.0, value.clamp_min(2**-127).log2().round().clamp(-127, 127)), 8
    raise ValueError(f"unsupported scale storage format {format!r}")
