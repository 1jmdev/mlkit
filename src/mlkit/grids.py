"""Scalar and vector representable sets, independent of scaling algorithms."""

from __future__ import annotations

import builtins
import math
from collections.abc import Callable

import torch
from torch import Tensor

from mlkit.operations import nearest, snap


class Grid:
    def __init__(
        self,
        function: Callable[[Tensor], Tensor],
        bits: int,
        dim: int = 1,
        *,
        values: Tensor | None = None,
        name: str = "custom",
    ) -> None:
        if bits < 1 or dim < 1:
            raise ValueError("grid bits and dimension must be positive")
        self.function = function
        self.bits = bits
        self.dim = dim
        self.values = values
        self.name = name

    def __call__(self, value: Tensor) -> Tensor:
        return self.function(value)

    def __repr__(self) -> str:
        return f"Grid({self.name}, bits={self.bits}, dim={self.dim})"


class GridFactory:
    def __call__(self, *, bits: int, dim: int = 1) -> Callable[[Callable], Grid]:
        def decorate(function: Callable[[Tensor], Tensor]) -> Grid:
            return Grid(function, bits, dim, name=function.__name__)

        return decorate

    def int(self, bits: int) -> Grid:
        if not 2 <= bits <= 16:
            raise ValueError("signed integer grids support 2 through 16 bits")
        minimum, maximum = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        values = torch.arange(minimum, maximum + 1, dtype=torch.float32)
        return Grid(
            lambda x: x.round().clamp(minimum, maximum),
            bits, values=values, name=f"int{bits}",
        )

    def values(self, values: Tensor, *, bits: builtins.int | None = None) -> Grid:
        values = torch.as_tensor(values, dtype=torch.float32).flatten().sort().values
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError("a scalar codebook must contain finite values")
        precision = max(1, math.ceil(math.log2(values.numel()))) if bits is None else bits
        if values.numel() > 2**precision:
            raise ValueError("codebook size exceeds the declared bit capacity")
        return Grid(lambda x: snap(x, values), precision, values=values, name="values")

    def vector(self, codebook: Tensor, *, bits: builtins.int | None = None) -> Grid:
        codebook = torch.as_tensor(codebook, dtype=torch.float32)
        if codebook.ndim != 2 or min(codebook.shape) < 1 or not torch.isfinite(codebook).all():
            raise ValueError("a vector codebook must be a nonempty finite matrix")
        precision = max(1, math.ceil(math.log2(len(codebook)))) if bits is None else bits
        if len(codebook) > 2**precision:
            raise ValueError("codebook size exceeds the declared bit capacity")

        def round_vectors(value: Tensor) -> Tensor:
            return nearest(value.reshape(-1, codebook.shape[1]), codebook).reshape_as(value)

        return Grid(round_vectors, precision, codebook.shape[1], values=codebook, name="vector")

    def fp(self, format: str) -> Grid:
        if format == "e2m1":
            return self.values(torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0,
                                             0.5, 1, 1.5, 2, 3, 4, 6]), bits=4)
        if format not in {"e3m2", "e2m3", "e4m3", "e5m2"}:
            raise ValueError(f"unsupported floating-point grid {format!r}")
        exponent_bits = builtins.int(format[1])
        mantissa_bits = builtins.int(format[3])
        bias = 2 ** (exponent_bits - 1) - 1
        values = [0.0]
        for exponent in range(2**exponent_bits - 1):
            for mantissa in range(2**mantissa_bits):
                value = (
                    mantissa / 2**mantissa_bits * 2 ** (1 - bias)
                    if exponent == 0
                    else (1 + mantissa / 2**mantissa_bits) * 2 ** (exponent - bias)
                )
                values.extend((-value, value))
        return self.values(
            torch.tensor(sorted(set(values))), bits=1 + exponent_bits + mantissa_bits
        )


grid = GridFactory()

NF4_VALUES = torch.tensor([
    -1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138, -0.18477343,
    -0.09105004, 0.0, 0.07958030, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261700, 0.72295684, 1.0,
])
