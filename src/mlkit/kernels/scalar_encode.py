"""Fused encoding of a matrix into grouped scalar codes and search for group scales.

Encoding reads the matrix once and writes one code per element. The scale search
evaluates every candidate scale of a group while its elements are in registers,
so all candidates cost one pass over the matrix instead of one pass each.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda import libdevice

from mlkit.kernels.scalar_rounding import (
    SCALE_FORMAT_CODES,
    nearest_codebook_index,
    nearest_integer,
    rounded_scale,
)

ROW_TILE = 8
MAXIMUM_TILE = 128
MINIMUM_TILE = 8

ENCODE_RUNTIME_ARGUMENTS = [
    "value",
    "scales",
    "zeros",
    "values",
    "codes",
    "width",
    "rows",
    "groups_per_row",
    "group_width",
    "tiles_per_row",
]
SEARCH_RUNTIME_ARGUMENTS = [
    "value",
    "scales",
    "zeros",
    "values",
    "fractions",
    "selected",
    "width",
    "group_width",
    "groups_per_row",
    "tiles_per_group",
    "candidates",
]


def tile_for(group: int) -> int | None:
    """Columns per kernel tile: the largest power of two dividing a scale group."""
    tile = min(MAXIMUM_TILE, group & -group)
    return tile if tile >= MINIMUM_TILE else None


def search_steps_for(codebook_size: int) -> int:
    return math.ceil(math.log2(codebook_size + 1))


@triton.jit(
    do_not_specialize=ENCODE_RUNTIME_ARGUMENTS,
    do_not_specialize_on_alignment=ENCODE_RUNTIME_ARGUMENTS,
)
def encode_scalar_codes(
    value,
    scales,
    zeros,
    values,
    codes,
    width,
    rows,
    groups_per_row,
    group_width,
    tiles_per_row,
    has_zero: tl.constexpr,
    integer: tl.constexpr,
    minimum_value: tl.constexpr,
    maximum_value: tl.constexpr,
    codebook_size: tl.constexpr,
    search_steps: tl.constexpr,
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
        samples = tl.load(value + positions, valid, other=0)
        scale_indices = row_indices * groups_per_row + (tile_index * tile) // group_width
        scale = tl.load(scales + scale_indices, row_valid, other=1)
        if has_zero:
            samples -= tl.load(zeros + scale_indices, row_valid, other=0)[:, None]
        normalized = libdevice.div_rn(samples, scale[:, None])
        if integer:
            indices = nearest_integer(normalized, minimum_value, maximum_value) - minimum_value
        else:
            indices = nearest_codebook_index(normalized, values, codebook_size, search_steps)
        tl.store(codes + positions, indices.to(tl.int32), valid)


def encode(
    value: Tensor,
    scales: Tensor,
    zeros: Tensor | None,
    values: Tensor,
    group: int,
    tile: int,
    *,
    integer: bool,
    minimum: float,
    maximum: float,
) -> Tensor:
    """Codes of the nearest grid values of ``(value - zeros) / scales``, one byte each."""
    rows, width = value.shape
    codes = torch.empty((rows, width), device=value.device, dtype=torch.uint8)
    encode_scalar_codes[(triton.cdiv(rows, ROW_TILE),)](
        value,
        scales,
        zeros,
        values,
        codes,
        width,
        rows,
        scales.shape[1],
        group,
        triton.cdiv(width, tile),
        zeros is not None,
        integer,
        minimum,
        maximum,
        values.numel(),
        search_steps_for(values.numel()),
        ROW_TILE,
        tile,
        enable_fp_fusion=False,
    )
    return codes


@triton.jit
def group_squared_error(
    value,
    values,
    row_start,
    group_start,
    scale,
    zero,
    width,
    group_width,
    tiles_per_group,
    integer: tl.constexpr,
    minimum_value: tl.constexpr,
    maximum_value: tl.constexpr,
    codebook_size: tl.constexpr,
    search_steps: tl.constexpr,
    tile: tl.constexpr,
):
    """Squared rounding error of one row group at one scale."""
    offsets = tl.arange(0, tile)
    error = tl.zeros((tile,), dtype=tl.float32)
    for tile_index in range(tiles_per_group):
        within_group = tile_index * tile + offsets
        columns = group_start + within_group
        valid = (within_group < group_width) & (columns < width)
        samples = tl.load(value + row_start + columns, valid, other=0) - zero
        normalized = libdevice.div_rn(samples, scale)
        if integer:
            rounded = nearest_integer(normalized, minimum_value, maximum_value)
        else:
            index = nearest_codebook_index(normalized, values, codebook_size, search_steps)
            rounded = tl.load(values + index)
        difference = tl.where(valid, rounded * scale - samples, 0.0)
        error += difference * difference
    return tl.sum(error, axis=0)


@triton.jit(
    do_not_specialize=SEARCH_RUNTIME_ARGUMENTS,
    do_not_specialize_on_alignment=SEARCH_RUNTIME_ARGUMENTS,
)
def search_group_scales(
    value,
    scales,
    zeros,
    values,
    fractions,
    selected,
    width,
    group_width,
    groups_per_row,
    tiles_per_group,
    candidates,
    has_zero: tl.constexpr,
    integer: tl.constexpr,
    minimum_value: tl.constexpr,
    maximum_value: tl.constexpr,
    codebook_size: tl.constexpr,
    search_steps: tl.constexpr,
    format: tl.constexpr,
    tile: tl.constexpr,
):
    identifier = tl.program_id(0)
    row_start = (identifier // groups_per_row).to(tl.int64) * width
    group_start = (identifier % groups_per_row) * group_width
    base = tl.load(scales + identifier)
    zero = tl.full((), 0, tl.float32)
    if has_zero:
        zero = tl.load(zeros + identifier)
    best_scale = base
    best_error = group_squared_error(
        value,
        values,
        row_start,
        group_start,
        base,
        zero,
        width,
        group_width,
        tiles_per_group,
        integer,
        minimum_value,
        maximum_value,
        codebook_size,
        search_steps,
        tile,
    )
    for candidate in range(candidates):
        scale = rounded_scale(base * tl.load(fractions + candidate), format)
        error = group_squared_error(
            value,
            values,
            row_start,
            group_start,
            scale,
            zero,
            width,
            group_width,
            tiles_per_group,
            integer,
            minimum_value,
            maximum_value,
            codebook_size,
            search_steps,
            tile,
        )
        improved = error < best_error
        best_scale = tl.where(improved, scale, best_scale)
        best_error = tl.minimum(best_error, error)
    tl.store(selected + identifier, best_scale)


def search_scales(
    value: Tensor,
    scales: Tensor,
    zeros: Tensor | None,
    values: Tensor,
    fractions: Tensor,
    group: int,
    tile: int,
    scale_format: str,
    *,
    integer: bool,
    minimum: float,
    maximum: float,
) -> Tensor:
    """For every row group, the candidate ``scales * fraction`` with least squared error.

    The stored scale itself is the first candidate and is kept on ties. Candidates
    are rounded to ``scale_format`` before they are evaluated.
    """
    width = value.shape[1]
    selected = torch.empty_like(scales)
    search_group_scales[(scales.numel(),)](
        value,
        scales,
        zeros,
        values,
        fractions,
        selected,
        width,
        group,
        scales.shape[1],
        triton.cdiv(min(group, width), tile),
        fractions.numel(),
        zeros is not None,
        integer,
        minimum,
        maximum,
        values.numel(),
        search_steps_for(values.numel()),
        SCALE_FORMAT_CODES[scale_format],
        tile,
        enable_fp_fusion=False,
    )
    return selected
