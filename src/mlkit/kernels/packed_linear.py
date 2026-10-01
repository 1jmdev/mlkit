"""Fused scalar-grid INT4 decoding and matrix-vector multiplication."""

import triton
import triton.language as language
from torch import Tensor


@triton.jit
def packed_matrix_vector(
    inputs,
    packed,
    scales,
    values,
    zeros,
    bias,
    output,
    input_width: language.constexpr,
    output_width: language.constexpr,
    group_width: language.constexpr,
    groups_per_row: language.constexpr,
    has_zero: language.constexpr,
    has_bias: language.constexpr,
    row_tile: language.constexpr,
    column_tile: language.constexpr,
):
    rows = language.program_id(0) * row_tile + language.arange(0, row_tile)
    sample = language.program_id(1)
    columns = language.arange(0, column_tile)
    accumulator = language.full((row_tile,), 0, language.float32)
    for offset in range(language.cdiv(input_width, column_tile)):
        positions = offset * column_tile + columns
        weight_indices = rows[:, None] * input_width + positions[None, :]
        valid = (rows[:, None] < output_width) & (positions[None, :] < input_width)
        bytes = language.load(packed + weight_indices // 2, valid, other=0)
        codes = (bytes >> ((weight_indices % 2) * 4)) & 15
        code_values = language.load(values + codes)
        scale_indices = rows[:, None] * groups_per_row + positions[None, :] // group_width
        scale_values = language.load(scales + scale_indices, valid, other=0)
        weights = code_values * scale_values
        if has_zero:
            weights += language.load(zeros + scale_indices, valid, other=0)
        weights = weights.to(inputs.dtype.element_ty).to(language.float32)
        activations = language.load(
            inputs + sample * input_width + positions, positions < input_width, other=0
        ).to(language.float32)
        accumulator += language.sum(weights * activations[None, :], axis=1)
    if has_bias:
        accumulator += language.load(bias + rows, rows < output_width, other=0)
    language.store(output + sample * output_width + rows, accumulator, rows < output_width)


def matrix_vector(
    inputs: Tensor,
    packed: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    group: int,
    *,
    row_tile: int = 4,
    column_tile: int = 1024,
    warps: int = 4,
) -> None:
    packed_matrix_vector[(triton.cdiv(output.shape[-1], row_tile), inputs.shape[0])](
        inputs, packed, scales, values, zeros, bias, output,
        inputs.shape[-1], output.shape[-1], group, scales.shape[1],
        zeros is not None, bias is not None, row_tile, column_tile, num_warps=warps,
    )
