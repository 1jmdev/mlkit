"""Fused sequential scalar rounding inside an error-feedback tile.

Each column is rounded after adding the feedback of every earlier column:
``target = weight + correction``, and the deviation ``weight - reconstruction``
is propagated to later columns of the tile through the LDL feedback
coefficients. Matrix dimensions and tile positions are runtime arguments, so
one compiled kernel serves every layer shape and every tile of a layer.
"""

import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

ROW_TILE = 16


@triton.jit
def round_feedback_tile(
    original,
    feedback,
    coefficients,
    scales,
    zeros,
    values,
    output,
    encoded,
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
    corrections = tl.load(
        feedback + row_offsets[:, None] + tile_start + columns[None, :], valid, other=0
    )
    for column in range(tile_width):
        position = tile_start + column
        weight = tl.load(original + row_offsets + position, row_valid, other=0)
        correction = tl.sum(tl.where(columns[None, :] == column, corrections, 0), axis=1)
        target = weight + correction
        scale_indices = rows * groups_per_row + (position - region_start) // group_width
        scale_values = tl.load(scales + scale_indices, row_valid, other=1)
        zero_values = tl.full((row_tile,), 0, tl.float32)
        if has_zero:
            zero_values = tl.load(zeros + scale_indices, row_valid, other=0)
        normalized = libdevice.div_rn(target - zero_values, scale_values)
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
        deviation = weight - reconstruction
        coefficient_row = tl.load(
            coefficients + position.to(tl.int64) * input_width + tile_start + columns,
            (columns > column) & (columns < tile_width),
            other=0,
        )
        corrections += deviation[:, None] * coefficient_row[None, :]
        tl.store(output + row_offsets + position, reconstruction, row_valid)
        tl.store(encoded + row_offsets + position, indices.to(tl.int32), row_valid)


def round_tile(
    original: Tensor,
    feedback: Tensor,
    coefficients: Tensor,
    scales: Tensor,
    zeros: Tensor | None,
    output: Tensor,
    encoded: Tensor,
    *,
    region_start: int,
    tile_start: int,
    tile_width: int,
    group: int,
    bits: int,
    values: Tensor,
    integer_grid: bool,
) -> None:
    """Round columns ``tile_start`` to ``tile_start + tile_width`` of every row."""
    round_feedback_tile[(triton.cdiv(original.shape[0], ROW_TILE),)](
        original,
        feedback,
        coefficients,
        scales,
        zeros,
        values,
        output,
        encoded,
        original.shape[1],
        original.shape[0],
        region_start,
        tile_start,
        tile_width,
        group,
        scales.shape[1],
        -(2 ** (bits - 1)),
        2 ** (bits - 1) - 1,
        zeros is not None,
        integer_grid,
        values.numel(),
        triton.next_power_of_2(values.numel()),
        ROW_TILE,
        triton.next_power_of_2(tile_width),
        num_warps=4,
        enable_fp_fusion=False,
    )
