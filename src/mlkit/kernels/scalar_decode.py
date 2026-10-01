"""Fused reconstruction of grouped scalar codes into a dense FP32 matrix.

Each program reconstructs a tile of rows. A column tile lies inside one scale
group, so scales and offsets are loaded once per tile rather than gathered for
every element, and no index or scale temporaries are materialized.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

ROW_TILE = 8
MAXIMUM_TILE = 128
MINIMUM_TILE = 8

RUNTIME_ARGUMENTS = [
    "codes",
    "scales",
    "values",
    "zeros",
    "output",
    "width",
    "rows",
    "groups_per_row",
    "group_width",
    "tiles_per_row",
]


def tile_for(group: int) -> int | None:
    """Columns per kernel tile: the largest power of two dividing a scale group."""
    tile = min(MAXIMUM_TILE, group & -group)
    return tile if tile >= MINIMUM_TILE else None


@triton.jit(do_not_specialize=RUNTIME_ARGUMENTS, do_not_specialize_on_alignment=RUNTIME_ARGUMENTS)
def decode_scalar_codes(
    codes,
    scales,
    values,
    zeros,
    output,
    width,
    rows,
    groups_per_row,
    group_width,
    tiles_per_row,
    has_zero: tl.constexpr,
    row_tile: tl.constexpr,
    tile: tl.constexpr,
):
    row_indices = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    row_valid = row_indices < rows
    offsets = tl.arange(0, tile)
    row_starts = row_indices.to(tl.int64) * width
    for tile_index in range(tiles_per_row):
        columns = tile_index * tile + offsets
        valid = row_valid[:, None] & (columns < width)[None, :]
        positions = row_starts[:, None] + columns[None, :]
        indices = tl.load(codes + positions, valid, other=0).to(tl.int32)
        scale_indices = row_indices * groups_per_row + (tile_index * tile) // group_width
        scale = tl.load(scales + scale_indices, row_valid, other=0)
        weights = tl.load(values + indices) * scale[:, None]
        if has_zero:
            weights += tl.load(zeros + scale_indices, row_valid, other=0)[:, None]
        tl.store(output + positions, weights, valid)


def decode(
    codes: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    group: int,
    tile: int,
) -> Tensor:
    """Reconstruct ``values[codes] * scales + zeros`` with one scale per row and group."""
    rows, width = codes.shape
    output = torch.empty((rows, width), device=codes.device, dtype=torch.float32)
    decode_scalar_codes[(triton.cdiv(rows, ROW_TILE),)](
        codes,
        scales,
        values,
        zeros,
        output,
        width,
        rows,
        scales.shape[1],
        group,
        triton.cdiv(width, tile),
        zeros is not None,
        ROW_TILE,
        tile,
        enable_fp_fusion=False,
    )
    return output
