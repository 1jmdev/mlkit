"""Fused scalar-grid INT4 decoding and matrix-vector multiplication."""

import triton
import triton.language as tl
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({"row_tile": 1, "column_tile": 1024}, num_warps=4),
        triton.Config({"row_tile": 1, "column_tile": 4096}, num_warps=4),
        triton.Config({"row_tile": 1, "column_tile": 8192}, num_warps=4),
        triton.Config({"row_tile": 2, "column_tile": 2048}, num_warps=4),
        triton.Config({"row_tile": 2, "column_tile": 4096}, num_warps=4),
        triton.Config({"row_tile": 4, "column_tile": 1024}, num_warps=4),
        triton.Config({"row_tile": 8, "column_tile": 1024}, num_warps=4),
    ],
    key=["input_width", "output_width", "group_width"],
)
@triton.jit
def packed_matrix_vector(
    inputs,
    packed,
    scales,
    values,
    zeros,
    bias,
    output,
    input_width: tl.constexpr,
    output_width: tl.constexpr,
    group_width: tl.constexpr,
    groups_per_row: tl.constexpr,
    has_zero: tl.constexpr,
    has_bias: tl.constexpr,
    uniform_grid: tl.constexpr,
    grid_minimum: tl.constexpr,
    grid_step: tl.constexpr,
    row_tile: tl.constexpr,
    column_tile: tl.constexpr,
):
    rows = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    sample = tl.program_id(1)
    columns = tl.arange(0, column_tile)
    accumulator = tl.full((row_tile,), 0, tl.float32)
    for offset in range(tl.cdiv(input_width, column_tile)):
        positions = offset * column_tile + columns
        weight_indices = rows[:, None] * input_width + positions[None, :]
        valid = (rows[:, None] < output_width) & (positions[None, :] < input_width)
        bytes = tl.load(packed + weight_indices // 2, valid, other=0)
        codes = (bytes >> ((weight_indices % 2) * 4)) & 15
        if uniform_grid:
            code_values = grid_minimum + codes.to(tl.float32) * grid_step
        else:
            code_values = tl.load(values + codes)
        scale_indices = rows[:, None] * groups_per_row + positions[None, :] // group_width
        scale_values = tl.load(scales + scale_indices, valid, other=0)
        weights = code_values * scale_values
        if has_zero:
            weights += tl.load(zeros + scale_indices, valid, other=0)
        weights = weights.to(inputs.dtype.element_ty).to(tl.float32)
        activations = tl.load(
            inputs + sample * input_width + positions, positions < input_width, other=0
        ).to(tl.float32)
        accumulator += tl.sum(weights * activations[None, :], axis=1)
    if has_bias:
        accumulator += tl.load(bias + rows, rows < output_width, other=0)
    tl.store(output + sample * output_width + rows, accumulator, rows < output_width)


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
    uniform_grid: bool = False,
    grid_minimum: float = 0.0,
    grid_step: float = 1.0,
) -> None:
    def launch_grid(parameters):
        return triton.cdiv(output.shape[-1], parameters["row_tile"]), inputs.shape[0]

    packed_matrix_vector[launch_grid](
        inputs, packed, scales, values, zeros, bias, output,
        inputs.shape[-1], output.shape[-1], group, scales.shape[1],
        zeros is not None, bias is not None, uniform_grid, grid_minimum, grid_step,
    )


@triton.jit
def decode_matrix(
    packed,
    scales,
    values,
    zeros,
    output,
    elements: tl.constexpr,
    input_width: tl.constexpr,
    group_width: tl.constexpr,
    groups_per_row: tl.constexpr,
    has_zero: tl.constexpr,
    uniform_grid: tl.constexpr,
    grid_minimum: tl.constexpr,
    grid_step: tl.constexpr,
    tile: tl.constexpr,
):
    indices = tl.program_id(0) * tile + tl.arange(0, tile)
    valid = indices < elements
    bytes = tl.load(packed + indices // 2, valid, other=0)
    codes = (bytes >> ((indices % 2) * 4)) & 15
    if uniform_grid:
        code_values = grid_minimum + codes.to(tl.float32) * grid_step
    else:
        code_values = tl.load(values + codes)
    rows, columns = indices // input_width, indices % input_width
    scale_indices = rows * groups_per_row + columns // group_width
    weights = code_values * tl.load(scales + scale_indices, valid, other=0)
    if has_zero:
        weights += tl.load(zeros + scale_indices, valid, other=0)
    tl.store(output + indices, weights, valid)


def decode(
    packed: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    output: Tensor,
    group: int,
    uniform_grid: bool,
    grid_minimum: float,
    grid_step: float,
) -> None:
    decode_matrix[(triton.cdiv(output.numel(), 4096),)](
        packed, scales, values, zeros, output, output.numel(), output.shape[1],
        group, scales.shape[1], zeros is not None, uniform_grid, grid_minimum,
        grid_step, 4096, num_warps=4,
    )

