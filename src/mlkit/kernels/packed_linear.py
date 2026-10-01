"""Fused four-bit scalar-grid decoding and matrix-vector multiplication.

Both kernels read every packed byte once, yielding two codes, and process whole
byte tiles that lie inside one scale group, so each scale is loaded once per
tile instead of once per element. Layer dimensions are runtime arguments without
value or alignment specialization, which lets one compiled kernel serve every
layer and be launched directly.
"""

import triton
import triton.language as tl
from torch import Tensor

ROW_TILE = 8
MAXIMUM_TILE_BYTES = 64
MINIMUM_TILE_BYTES = 8
MAXIMUM_SAMPLE_TILE = 8

RUNTIME_ARGUMENTS = [
    "inputs",
    "packed",
    "scales",
    "values",
    "zeros",
    "bias",
    "output",
    "input_width",
    "output_width",
    "groups_per_row",
    "group_width",
    "tiles_per_row",
    "samples",
]


def tile_bytes_for(input_width: int, group: int) -> int | None:
    """Bytes per kernel tile, or ``None`` when rows or groups are not byte aligned.

    Two codes share a byte, so rows and scale groups must contain an even number
    of codes. A tile is the largest power-of-two byte count dividing a group.
    """
    if input_width % 2 or group % 2:
        return None
    group_bytes = group // 2
    tile_bytes = min(MAXIMUM_TILE_BYTES, group_bytes & -group_bytes)
    return tile_bytes if tile_bytes >= MINIMUM_TILE_BYTES else None


@triton.jit(do_not_specialize=RUNTIME_ARGUMENTS, do_not_specialize_on_alignment=RUNTIME_ARGUMENTS)
def packed_matrix_vector(
    inputs,
    packed,
    scales,
    values,
    zeros,
    bias,
    output,
    input_width,
    output_width,
    groups_per_row,
    group_width,
    tiles_per_row,
    samples,
    has_zero: tl.constexpr,
    has_bias: tl.constexpr,
    uniform_grid: tl.constexpr,
    grid_minimum: tl.constexpr,
    grid_step: tl.constexpr,
    row_tile: tl.constexpr,
    tile_bytes: tl.constexpr,
    sample_tile: tl.constexpr,
):
    rows = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    sample_indices = tl.program_id(1) * sample_tile + tl.arange(0, sample_tile)
    row_valid = rows < output_width
    sample_valid = sample_indices < samples
    offsets = tl.arange(0, tile_bytes)
    row_bytes = input_width // 2
    row_starts = rows.to(tl.int64) * row_bytes
    input_starts = sample_indices.to(tl.int64) * input_width
    accumulator = tl.zeros((sample_tile, row_tile, tile_bytes), dtype=tl.float32)
    for tile in range(tiles_per_row):
        byte_positions = tile * tile_bytes + offsets
        valid = byte_positions < row_bytes
        codes = tl.load(
            packed + row_starts[:, None] + byte_positions[None, :],
            row_valid[:, None] & valid[None, :],
            other=0,
        )
        low = (codes & 15).to(tl.int32)
        high = (codes >> 4).to(tl.int32)
        if uniform_grid:
            low_weights = grid_minimum + low.to(tl.float32) * grid_step
            high_weights = grid_minimum + high.to(tl.float32) * grid_step
        else:
            low_weights = tl.load(values + low)
            high_weights = tl.load(values + high)
        scale_indices = rows * groups_per_row + (2 * tile * tile_bytes) // group_width
        scale = tl.load(scales + scale_indices, row_valid, other=0).to(tl.float32)
        low_weights = low_weights * scale[:, None]
        high_weights = high_weights * scale[:, None]
        if has_zero:
            zero = tl.load(zeros + scale_indices, row_valid, other=0).to(tl.float32)
            low_weights += zero[:, None]
            high_weights += zero[:, None]
        # Positions beyond a row or a sample read zero inputs and contribute nothing.
        input_valid = sample_valid[:, None] & valid[None, :]
        input_positions = inputs + input_starts[:, None] + 2 * byte_positions[None, :]
        even = tl.load(input_positions, input_valid, other=0).to(tl.float32)
        odd = tl.load(input_positions + 1, input_valid, other=0).to(tl.float32)
        accumulator += (
            low_weights[None, :, :] * even[:, None, :] + high_weights[None, :, :] * odd[:, None, :]
        )
    result = tl.sum(accumulator, axis=2)
    if has_bias:
        result += tl.load(bias + rows, row_valid, other=0).to(tl.float32)[None, :]
    output_positions = (
        output + sample_indices.to(tl.int64)[:, None] * output_width + rows[None, :]
    )
    tl.store(output_positions, result, sample_valid[:, None] & row_valid[None, :])


def sample_tile_for(samples: int) -> int:
    """Input rows multiplied by one program; each program decodes its weights once."""
    return min(MAXIMUM_SAMPLE_TILE, triton.next_power_of_2(samples))


def matrix_vector(
    inputs: Tensor,
    packed: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    bias: Tensor | None,
    output: Tensor,
    group: int,
    tile_bytes: int,
    *,
    uniform_grid: bool = False,
    grid_minimum: float = 0.0,
    grid_step: float = 1.0,
) -> None:
    """Multiply rows of ``inputs`` by packed weights through the public Triton call path."""
    samples, input_width = inputs.shape
    output_width = output.shape[-1]
    sample_tile = sample_tile_for(samples)
    grid = (triton.cdiv(output_width, ROW_TILE), triton.cdiv(samples, sample_tile))
    packed_matrix_vector[grid](
        inputs,
        packed,
        scales,
        values,
        zeros,
        bias,
        output,
        input_width,
        output_width,
        scales.shape[1],
        group,
        triton.cdiv(input_width // 2, tile_bytes),
        samples,
        zeros is not None,
        bias is not None,
        uniform_grid,
        grid_minimum,
        grid_step,
        ROW_TILE,
        tile_bytes,
        sample_tile,
    )


@triton.jit
def decode_rows(
    packed,
    scales,
    values,
    zeros,
    output,
    input_width,
    output_width,
    groups_per_row,
    group_width,
    tiles_per_row,
    has_zero: tl.constexpr,
    uniform_grid: tl.constexpr,
    grid_minimum: tl.constexpr,
    grid_step: tl.constexpr,
    row_tile: tl.constexpr,
    tile_bytes: tl.constexpr,
):
    rows = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    row_valid = rows < output_width
    offsets = tl.arange(0, tile_bytes)
    row_bytes = input_width // 2
    packed_starts = rows.to(tl.int64) * row_bytes
    output_starts = rows.to(tl.int64) * input_width
    for tile in range(tiles_per_row):
        byte_positions = tile * tile_bytes + offsets
        valid = row_valid[:, None] & (byte_positions < row_bytes)[None, :]
        codes = tl.load(packed + packed_starts[:, None] + byte_positions[None, :], valid, other=0)
        low = (codes & 15).to(tl.int32)
        high = (codes >> 4).to(tl.int32)
        if uniform_grid:
            low_values = grid_minimum + low.to(tl.float32) * grid_step
            high_values = grid_minimum + high.to(tl.float32) * grid_step
        else:
            low_values = tl.load(values + low)
            high_values = tl.load(values + high)
        scale_indices = rows * groups_per_row + (2 * tile * tile_bytes) // group_width
        scale = tl.load(scales + scale_indices, row_valid, other=0).to(tl.float32)
        low_weights = low_values * scale[:, None]
        high_weights = high_values * scale[:, None]
        if has_zero:
            zero = tl.load(zeros + scale_indices, row_valid, other=0).to(tl.float32)
            low_weights += zero[:, None]
            high_weights += zero[:, None]
        positions = output + output_starts[:, None] + 2 * byte_positions[None, :]
        tl.store(positions, low_weights, valid)
        tl.store(positions + 1, high_weights, valid)


def decode(
    packed: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    output: Tensor,
    group: int,
    tile_bytes: int,
    uniform_grid: bool,
    grid_minimum: float,
    grid_step: float,
) -> None:
    """Reconstruct the dense weight matrix from byte-aligned packed four-bit codes."""
    output_width, input_width = output.shape
    decode_rows[(triton.cdiv(output_width, ROW_TILE),)](
        packed,
        scales,
        values,
        zeros,
        output,
        input_width,
        output_width,
        scales.shape[1],
        group,
        triton.cdiv(input_width // 2, tile_bytes),
        zeros is not None,
        uniform_grid,
        grid_minimum,
        grid_step,
        ROW_TILE,
        tile_bytes,
    )
