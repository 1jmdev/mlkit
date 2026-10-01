"""Fused scalar-grid INT4 decoding and matrix-vector multiplication."""

import triton
import triton.language as language
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
    input_width: language.constexpr,
    output_width: language.constexpr,
    group_width: language.constexpr,
    groups_per_row: language.constexpr,
    has_zero: language.constexpr,
    has_bias: language.constexpr,
    uniform_grid: language.constexpr,
    grid_minimum: language.constexpr,
    grid_step: language.constexpr,
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
        if uniform_grid:
            code_values = grid_minimum + codes.to(language.float32) * grid_step
        else:
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
    elements: language.constexpr,
    input_width: language.constexpr,
    group_width: language.constexpr,
    groups_per_row: language.constexpr,
    has_zero: language.constexpr,
    uniform_grid: language.constexpr,
    grid_minimum: language.constexpr,
    grid_step: language.constexpr,
    tile: language.constexpr,
):
    indices = language.program_id(0) * tile + language.arange(0, tile)
    valid = indices < elements
    bytes = language.load(packed + indices // 2, valid, other=0)
    codes = (bytes >> ((indices % 2) * 4)) & 15
    if uniform_grid:
        code_values = grid_minimum + codes.to(language.float32) * grid_step
    else:
        code_values = language.load(values + codes)
    rows, columns = indices // input_width, indices % input_width
    scale_indices = rows * groups_per_row + columns // group_width
    weights = code_values * language.load(scales + scale_indices, valid, other=0)
    if has_zero:
        weights += language.load(zeros + scale_indices, valid, other=0)
    language.store(output + indices, weights, valid)


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


@triton.autotune(
    configs=[
        triton.Config({"batch_tile": 16, "output_tile": 32, "column_tile": 64}, num_warps=4),
        triton.Config({"batch_tile": 16, "output_tile": 64, "column_tile": 128}, num_warps=4),
        triton.Config({"batch_tile": 32, "output_tile": 64, "column_tile": 64}, num_warps=4),
        triton.Config({"batch_tile": 32, "output_tile": 64, "column_tile": 128}, num_warps=4),
        triton.Config({"batch_tile": 64, "output_tile": 64, "column_tile": 128}, num_warps=4),
    ],
    key=["batch_size", "input_width", "output_width", "group_width"],
)
@triton.jit
def packed_matrix_multiply(
    inputs,
    packed,
    scales,
    values,
    zeros,
    bias,
    output,
    batch_size: language.constexpr,
    input_width: language.constexpr,
    output_width: language.constexpr,
    group_width: language.constexpr,
    groups_per_row: language.constexpr,
    has_zero: language.constexpr,
    has_bias: language.constexpr,
    batch_tile: language.constexpr,
    output_tile: language.constexpr,
    column_tile: language.constexpr,
):
    samples = language.program_id(0) * batch_tile + language.arange(0, batch_tile)
    rows = language.program_id(1) * output_tile + language.arange(0, output_tile)
    columns = language.arange(0, column_tile)
    accumulator = language.full((batch_tile, output_tile), 0, language.float32)
    for offset in range(language.cdiv(input_width, column_tile)):
        positions = offset * column_tile + columns
        activations = language.load(
            inputs + samples[:, None] * input_width + positions[None, :],
            (samples[:, None] < batch_size) & (positions[None, :] < input_width), other=0,
        )
        weight_indices = rows[:, None] * input_width + positions[None, :]
        valid = (rows[:, None] < output_width) & (positions[None, :] < input_width)
        bytes = language.load(packed + weight_indices // 2, valid, other=0)
        codes = (bytes >> ((weight_indices % 2) * 4)) & 15
        code_values = language.load(values + codes)
        scale_indices = rows[:, None] * groups_per_row + positions[None, :] // group_width
        weights = code_values * language.load(scales + scale_indices, valid, other=0)
        if has_zero:
            weights += language.load(zeros + scale_indices, valid, other=0)
        weights = weights.to(inputs.dtype.element_ty)
        accumulator = language.dot(
            activations, language.trans(weights), accumulator, input_precision="tf32x3"
        )
    if has_bias:
        accumulator += language.load(bias + rows, rows < output_width, other=0)[None, :]
    language.store(
        output + samples[:, None] * output_width + rows[None, :], accumulator,
        (samples[:, None] < batch_size) & (rows[None, :] < output_width),
    )


def matrix_multiply(
    inputs: Tensor,
    packed: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    group: int,
) -> None:
    def launch_grid(parameters):
        return (
            triton.cdiv(inputs.shape[0], parameters["batch_tile"]),
            triton.cdiv(output.shape[-1], parameters["output_tile"]),
        )

    packed_matrix_multiply[launch_grid](
        inputs, packed, scales, values, zeros, bias, output,
        inputs.shape[0], inputs.shape[-1], output.shape[-1], group, scales.shape[1],
        zeros is not None, bias is not None,
    )
