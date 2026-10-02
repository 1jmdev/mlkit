"""Fused sequential rounding inside an error-feedback tile.

Each column, or each block of columns of a vector grid, is rounded after adding
the feedback of everything rounded before it: ``target = weight + correction``,
and the deviation ``weight - reconstruction`` is propagated to later columns of
the tile through the LDL feedback coefficients. Matrix dimensions and tile
positions are runtime arguments, so one compiled kernel serves every layer
shape and every tile of a layer.
"""

import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

from mlkit.kernels.lattice_search import nearest_lattice_code

ROW_TILE = 16
CODEBOOK_TILE = 256


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


@triton.jit
def round_vector_feedback_tile(
    original,
    feedback,
    coefficients,
    scales,
    codebook,
    output,
    encoded,
    input_width,
    region_start,
    tile_start,
    tile_width,
    group_width,
    groups_per_row,
    codebook_size,
    codebook_tiles,
    lattice: tl.constexpr,
    dimension: tl.constexpr,
    codebook_tile: tl.constexpr,
    column_tile: tl.constexpr,
):
    """Round one row of a tile in blocks of ``dimension`` columns.

    ``codebook`` holds the reconstruction of every code. For the E8P lattice its
    first 256 rows of magnitudes follow at offset ``codebook_size`` and the
    search is structured; any other codebook is searched exhaustively.
    """
    row = tl.program_id(0)
    columns = tl.arange(0, column_tile)
    coordinates = tl.arange(0, dimension)
    row_offset = row.to(tl.int64) * input_width
    corrections = tl.load(
        feedback + row_offset + tile_start + columns, columns < tile_width, other=0
    )
    if lattice:
        candidates = tl.arange(0, 256)
        absolute_points = tl.load(
            codebook + (codebook_size + candidates[:, None]) * 8 + coordinates[None, :]
        )
        point_norms = tl.sum(absolute_points * absolute_points, 1)
        point_parities = tl.sum(absolute_points, 1).to(tl.int32) & 1
    entries = tl.arange(0, codebook_tile)
    for block in range(tile_width // dimension):
        local = block * dimension
        position = tile_start + local
        weight = tl.load(original + row_offset + position + coordinates)
        selection = columns[None, :] == (local + coordinates)[:, None]
        target = weight + tl.sum(tl.where(selection, corrections[None, :], 0), axis=1)
        scale = tl.load(scales + row * groups_per_row + (position - region_start) // group_width)
        normalized = libdevice.div_rn(target, scale)
        if lattice:
            code = nearest_lattice_code(normalized, absolute_points, point_norms, point_parities)
        else:
            minimum_cost = tl.full((), float("inf"), tl.float32)
            code = tl.full((), 0, tl.int32)
            for tile in range(codebook_tiles):
                indices = tile * codebook_tile + entries
                present = indices < codebook_size
                codewords = tl.load(
                    codebook + indices[:, None] * dimension + coordinates[None, :],
                    present[:, None],
                    other=0,
                )
                differences = codewords - normalized[None, :]
                costs = tl.where(present, tl.sum(differences * differences, axis=1), float("inf"))
                cost = tl.min(costs, axis=0)
                index = tl.argmin(costs, axis=0, tie_break_left=True)
                code = tl.where(cost < minimum_cost, tile * codebook_tile + index, code)
                minimum_cost = tl.minimum(cost, minimum_cost)
        reconstruction = tl.load(codebook + code * dimension + coordinates) * scale
        deviation = weight - reconstruction
        later = (columns >= local + dimension) & (columns < tile_width)
        coefficient_block = tl.load(
            coefficients
            + (position + coordinates).to(tl.int64)[:, None] * input_width
            + tile_start
            + columns[None, :],
            later[None, :],
            other=0,
        )
        corrections += tl.sum(deviation[:, None] * coefficient_block, axis=0)
        tl.store(output + row_offset + position + coordinates, reconstruction)
        tl.store(
            encoded + row.to(tl.int64) * (input_width // dimension) + position // dimension, code
        )


def round_vector_tile(
    original: Tensor,
    feedback: Tensor,
    coefficients: Tensor,
    scales: Tensor,
    codebook: Tensor,
    output: Tensor,
    encoded: Tensor,
    *,
    region_start: int,
    tile_start: int,
    tile_width: int,
    group: int,
    dimension: int,
    codebook_size: int,
    lattice: bool,
) -> None:
    """Round columns ``tile_start`` to ``tile_start + tile_width`` of every row in blocks."""
    round_vector_feedback_tile[(original.shape[0],)](
        original,
        feedback,
        coefficients,
        scales,
        codebook,
        output,
        encoded,
        original.shape[1],
        region_start,
        tile_start,
        tile_width,
        group,
        scales.shape[1],
        codebook_size,
        triton.cdiv(codebook_size, CODEBOOK_TILE),
        lattice,
        dimension,
        CODEBOOK_TILE,
        triton.next_power_of_2(tile_width),
        num_warps=2,
        enable_fp_fusion=False,
    )
