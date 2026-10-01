"""Fused sequential scalar rounding inside a GPTQ update tile.

Matrix dimensions and tile positions are runtime arguments, so one compiled
kernel serves every layer shape and every tile of a layer.
"""

import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

ROW_TILE = 16


@triton.jit
def round_feedback_tile(
    working,
    upper,
    scales,
    zeros,
    values,
    output,
    encoded,
    errors,
    input_width,
    output_width,
    region_start,
    tile_start,
    tile_width,
    group_width,
    groups_per_row,
    minimum_code: tl.constexpr,
    maximum_code: tl.constexpr,
    has_zero: tl.constexpr,
    integer_grid: tl.constexpr,
    codebook_size: tl.constexpr,
    codebook_tile: tl.constexpr,
    row_tile: tl.constexpr,
    column_tile: tl.constexpr,
):
    rows = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    columns = tl.arange(0, column_tile)
    row_valid = rows < output_width
    valid = row_valid[:, None] & (columns[None, :] < tile_width)
    row_offsets = rows.to(tl.int64) * input_width
    weights = tl.load(
        working + row_offsets[:, None] + tile_start + columns[None, :], valid, other=0
    )
    for column in range(tile_width):
        position = tile_start + column
        original = tl.sum(tl.where(columns[None, :] == column, weights, 0), axis=1)
        scale_indices = rows * groups_per_row + (position - region_start) // group_width
        scale_values = tl.load(scales + scale_indices, row_valid, other=1)
        zero_values = tl.full((row_tile,), 0, tl.float32)
        if has_zero:
            zero_values = tl.load(zeros + scale_indices, row_valid, other=0)
        normalized = libdevice.div_rn(original - zero_values, scale_values)
        if integer_grid:
            codes = libdevice.nearbyint(normalized)
            codes = tl.minimum(tl.maximum(codes, minimum_code), maximum_code)
            indices = codes - minimum_code
            reconstruction = codes * scale_values + zero_values
        else:
            entries = tl.arange(0, codebook_tile)
            representable = tl.load(values + entries, entries < codebook_size, other=0)
            distances = tl.where(
                entries[None, :] < codebook_size,
                tl.abs(normalized[:, None] - representable[None, :]),
                float("inf"),
            )
            minimum_distance = tl.min(distances, axis=1)
            indices = tl.min(
                tl.where(distances == minimum_distance[:, None], entries[None, :], codebook_tile),
                axis=1,
            )
            indices = tl.minimum(indices, codebook_size - 1)
            selected = tl.load(values + indices)
            reconstruction = selected * scale_values + zero_values
        upper_row = upper + position.to(tl.int64) * input_width + tile_start
        diagonal = tl.load(upper_row + column)
        error = libdevice.div_rn(original - reconstruction, diagonal)
        coefficients = tl.load(
            upper_row + columns, (columns > column) & (columns < tile_width), other=0
        )
        weights -= error[:, None] * coefficients[None, :]
        tl.store(output + row_offsets + position, reconstruction, row_valid)
        tl.store(encoded + row_offsets + position, indices.to(tl.int32), row_valid)
        tl.store(errors + rows * tile_width + column, error, row_valid)


def round_tile(
    working: Tensor,
    upper: Tensor,
    scales: Tensor,
    zeros: Tensor | None,
    output: Tensor,
    encoded: Tensor,
    errors: Tensor,
    *,
    region_start: int,
    tile_start: int,
    group: int,
    bits: int,
    values: Tensor,
    integer_grid: bool,
) -> None:
    round_feedback_tile[(triton.cdiv(working.shape[0], ROW_TILE),)](
        working,
        upper,
        scales,
        zeros,
        values,
        output,
        encoded,
        errors,
        working.shape[1],
        working.shape[0],
        region_start,
        tile_start,
        errors.shape[1],
        group,
        scales.shape[1],
        -(2 ** (bits - 1)),
        2 ** (bits - 1) - 1,
        zeros is not None,
        integer_grid,
        values.numel(),
        triton.next_power_of_2(values.numel()),
        ROW_TILE,
        triton.next_power_of_2(errors.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
