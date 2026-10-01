"""Standard scalar formats built from grouped scaling."""

import builtins
from typing import Any

from mlkit.quantization.formats.grouped_scaling import Scaled, scaled
from mlkit.quantization.grids import NF4_VALUES, grid


def int(
    bits: builtins.int = 4,
    group: builtins.int | None = 128,
    **options: Any,
) -> Scaled:
    return scaled(grid.int(bits), group=group, **options)


def nf4(group: builtins.int | None = 64, **options: Any) -> Scaled:
    return scaled(grid.values(NF4_VALUES, bits=4), group=group, **options)


def mxfp4(group: builtins.int = 32, **options: Any) -> Scaled:
    return scaled(grid.fp("e2m1"), group=group, scale_fmt="e8m0", **options)
