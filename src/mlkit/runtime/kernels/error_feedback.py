"""Fused sequential scalar integer rounding inside a GPTQ update tile."""

import triton
import triton.language as language
from torch import Tensor
from triton.language.extra.cuda import libdevice


@triton.jit
def round_feedback_tile(
    working,
    upper,
    scales,
    zeros,
    output,
    encoded,
    errors,
    input_width: language.constexpr,
    output_width: language.constexpr,
    region_start: language.constexpr,
    tile_start: language.constexpr,
    tile_width: language.constexpr,
    group_width: language.constexpr,
    groups_per_row: language.constexpr,
    minimum_code: language.constexpr,
    maximum_code: language.constexpr,
    has_zero: language.constexpr,
    row_tile: language.constexpr,
    column_tile: language.constexpr,
):
    rows = language.program_id(0) * row_tile + language.arange(0, row_tile)
    columns = language.arange(0, column_tile)
    valid = (rows[:, None] < output_width) & (columns[None, :] < tile_width)
    weights = language.load(
        working + rows[:, None] * input_width + tile_start + columns[None, :], valid, other=0
    )
    for column in range(tile_width):
        original = language.sum(language.where(columns[None, :] == column, weights, 0), axis=1)
        scale_indices = rows * groups_per_row + (tile_start + column - region_start) // group_width
        scale_values = language.load(scales + scale_indices, rows < output_width, other=1)
        zero_values = language.full((row_tile,), 0, language.float32)
        if has_zero:
            zero_values = language.load(zeros + scale_indices, rows < output_width, other=0)
        codes = libdevice.nearbyint((original - zero_values) / scale_values)
        codes = language.minimum(language.maximum(codes, minimum_code), maximum_code)
        reconstruction = codes * scale_values + zero_values
        diagonal = language.load(upper + (tile_start + column) * input_width + tile_start + column)
        error = (original - reconstruction) / diagonal
        coefficients = language.load(
            upper + (tile_start + column) * input_width + tile_start + columns,
            (columns > column) & (columns < tile_width), other=0,
        )
        weights -= error[:, None] * coefficients[None, :]
        language.store(output + rows * input_width + tile_start + column,
                       reconstruction, rows < output_width)
        language.store(encoded + rows * input_width + tile_start + column,
                       (codes - minimum_code).to(language.int32), rows < output_width)
        language.store(errors + rows * tile_width + column, error, rows < output_width)


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
) -> None:
    row_tile = 16
    round_feedback_tile[(triton.cdiv(working.shape[0], row_tile),)](
        working, upper, scales, zeros, output, encoded, errors,
        working.shape[1], working.shape[0], region_start, tile_start,
        errors.shape[1], group, scales.shape[1], -(2 ** (bits - 1)), 2 ** (bits - 1) - 1,
        zeros is not None, row_tile, triton.next_power_of_2(errors.shape[1]),
        num_warps=4, enable_fp_fusion=False,
    )
