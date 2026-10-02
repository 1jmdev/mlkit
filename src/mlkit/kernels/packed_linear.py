"""Fused scalar-grid decoding and matrix-vector multiplication of packed codes.

Codes of one to eight bits are stored as a little-endian bit stream. The stream
repeats with a period of one word: the fewest whole bytes that hold a whole
number of codes, for example one byte and two codes at four bits, or three
bytes and eight codes at three bits. Both kernels assemble every word once and
process tiles of words that lie inside one scale group, so each scale is loaded
once per tile instead of once per element. Layer dimensions are runtime
arguments without value or alignment specialization, which lets one compiled
kernel serve every layer and be launched directly.
"""

import math
from dataclasses import dataclass

import triton
import triton.language as tl
from torch import Tensor

ROW_TILE = 8
MAXIMUM_TILE_CODES = 128
MINIMUM_TILE_CODES = 16
MAXIMUM_SAMPLE_TILE = 8
MAXIMUM_CODE_BITS = 8

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


@dataclass(frozen=True)
class PackedLayout:
    """The word structure of a packed code stream and the kernel tile it permits."""

    bits: int
    bytes_per_word: int
    codes_per_word: int
    tile_words: int

    @property
    def tile_codes(self) -> int:
        return self.tile_words * self.codes_per_word

    def row_bytes(self, input_width: int) -> int:
        return input_width // self.codes_per_word * self.bytes_per_word

    def tiles_per_row(self, input_width: int) -> int:
        return triton.cdiv(input_width, self.tile_codes)


def layout_for(bits: int, input_width: int, group: int) -> PackedLayout | None:
    """The layout of a layer, or ``None`` when rows or scale groups split a word.

    A tile is the largest power-of-two count of words dividing a scale group.
    """
    if not 1 <= bits <= MAXIMUM_CODE_BITS:
        return None
    divisor = math.gcd(bits, 8)
    bytes_per_word = bits // divisor
    codes_per_word = 8 // divisor
    if input_width % codes_per_word or group % codes_per_word:
        return None
    group_words = group // codes_per_word
    tile_words = min(MAXIMUM_TILE_CODES // codes_per_word, group_words & -group_words)
    if tile_words * codes_per_word < MINIMUM_TILE_CODES:
        return None
    return PackedLayout(bits, bytes_per_word, codes_per_word, tile_words)


@triton.jit
def packed_words(
    packed,
    row_starts,
    word_positions,
    valid,
    bytes_per_word: tl.constexpr,
):
    """Assemble the words at ``word_positions`` of every row from their bytes."""
    byte_positions = row_starts[:, None] + (word_positions * bytes_per_word)[None, :]
    if bytes_per_word > 4:
        words = tl.load(packed + byte_positions, valid, other=0).to(tl.int64)
        for byte in tl.static_range(1, bytes_per_word):
            words |= tl.load(packed + byte_positions + byte, valid, other=0).to(tl.int64) << (
                8 * byte
            )
    else:
        words = tl.load(packed + byte_positions, valid, other=0).to(tl.int32)
        for byte in tl.static_range(1, bytes_per_word):
            words |= tl.load(packed + byte_positions + byte, valid, other=0).to(tl.int32) << (
                8 * byte
            )
    return words


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
    code_bits: tl.constexpr,
    bytes_per_word: tl.constexpr,
    codes_per_word: tl.constexpr,
    tile_words: tl.constexpr,
    sample_tile: tl.constexpr,
):
    rows = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    sample_indices = tl.program_id(1) * sample_tile + tl.arange(0, sample_tile)
    row_valid = rows < output_width
    sample_valid = sample_indices < samples
    offsets = tl.arange(0, tile_words)
    row_words = input_width // codes_per_word
    row_starts = rows.to(tl.int64) * (row_words * bytes_per_word)
    input_starts = sample_indices.to(tl.int64) * input_width
    accumulator = tl.zeros((sample_tile, row_tile, tile_words), dtype=tl.float32)
    for tile in range(tiles_per_row):
        word_positions = tile * tile_words + offsets
        valid = word_positions < row_words
        words = packed_words(
            packed, row_starts, word_positions, row_valid[:, None] & valid[None, :], bytes_per_word
        )
        scale_indices = rows * groups_per_row + (tile * tile_words * codes_per_word) // group_width
        scale = tl.load(scales + scale_indices, row_valid, other=0).to(tl.float32)
        if has_zero:
            zero = tl.load(zeros + scale_indices, row_valid, other=0).to(tl.float32)
        input_valid = sample_valid[:, None] & valid[None, :]
        code_positions = word_positions * codes_per_word
        input_positions = inputs + input_starts[:, None] + code_positions[None, :]
        for code in tl.static_range(codes_per_word):
            codes = ((words >> (code * code_bits)) & ((1 << code_bits) - 1)).to(tl.int32)
            if uniform_grid:
                weights = grid_minimum + codes.to(tl.float32) * grid_step
            else:
                weights = tl.load(values + codes)
            weights = weights * scale[:, None]
            if has_zero:
                weights += zero[:, None]
            operands = tl.load(input_positions + code, input_valid, other=0).to(tl.float32)
            accumulator += weights[None, :, :] * operands[:, None, :]
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
    layout: PackedLayout,
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
        layout.tiles_per_row(input_width),
        samples,
        zeros is not None,
        bias is not None,
        uniform_grid,
        grid_minimum,
        grid_step,
        ROW_TILE,
        layout.bits,
        layout.bytes_per_word,
        layout.codes_per_word,
        layout.tile_words,
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
    code_bits: tl.constexpr,
    bytes_per_word: tl.constexpr,
    codes_per_word: tl.constexpr,
    tile_words: tl.constexpr,
):
    rows = tl.program_id(0) * row_tile + tl.arange(0, row_tile)
    row_valid = rows < output_width
    offsets = tl.arange(0, tile_words)
    row_words = input_width // codes_per_word
    packed_starts = rows.to(tl.int64) * (row_words * bytes_per_word)
    output_starts = rows.to(tl.int64) * input_width
    for tile in range(tiles_per_row):
        word_positions = tile * tile_words + offsets
        valid = row_valid[:, None] & (word_positions < row_words)[None, :]
        words = packed_words(packed, packed_starts, word_positions, valid, bytes_per_word)
        scale_indices = rows * groups_per_row + (tile * tile_words * codes_per_word) // group_width
        scale = tl.load(scales + scale_indices, row_valid, other=0).to(tl.float32)
        if has_zero:
            zero = tl.load(zeros + scale_indices, row_valid, other=0).to(tl.float32)
        code_positions = word_positions * codes_per_word
        positions = output + output_starts[:, None] + code_positions[None, :]
        for code in tl.static_range(codes_per_word):
            codes = ((words >> (code * code_bits)) & ((1 << code_bits) - 1)).to(tl.int32)
            if uniform_grid:
                weights = grid_minimum + codes.to(tl.float32) * grid_step
            else:
                weights = tl.load(values + codes)
            weights = weights * scale[:, None]
            if has_zero:
                weights += zero[:, None]
            tl.store(positions + code, weights, valid)


def decode(
    packed: Tensor,
    scales: Tensor,
    values: Tensor,
    zeros: Tensor | None,
    output: Tensor,
    group: int,
    layout: PackedLayout,
    uniform_grid: bool,
    grid_minimum: float,
    grid_step: float,
) -> None:
    """Reconstruct the dense weight matrix from word-aligned packed codes."""
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
        layout.tiles_per_row(input_width),
        zeros is not None,
        uniform_grid,
        grid_minimum,
        grid_step,
        ROW_TILE,
        layout.bits,
        layout.bytes_per_word,
        layout.codes_per_word,
        layout.tile_words,
    )
