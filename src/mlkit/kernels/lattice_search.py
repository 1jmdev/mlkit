"""Exact E8P magnitude search and minimum-cost sign-parity correction on CUDA."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def nearest_lattice_code(values, absolute_points, point_norms, point_parities):
    """The code of the E8P point nearest to eight coordinates.

    Every magnitude pattern is compared in both shifted cosets. A pattern whose
    parity disagrees with the signs of the input pays for flipping the sign of
    its cheapest coordinate.
    """
    coordinates = tl.arange(0, 8)
    candidates = tl.arange(0, 256)
    minimum_cost = tl.full((), float("inf"), tl.float32)
    selected = tl.full((), 0, tl.int32)
    for shift_index in tl.static_range(2):
        shift = -0.25 if shift_index == 0 else 0.25
        centered = values - shift
        negative = centered < 0
        sign_parity = tl.sum(negative.to(tl.int32), 0) & 1
        products = tl.abs(centered)[None, :] * absolute_points
        correction_cost = tl.min(products, 1)
        correction_axis = tl.argmin(products, 1, tie_break_left=True)
        mismatch = point_parities != sign_parity
        costs = (
            tl.sum(centered * centered, 0)
            + point_norms
            - 2 * tl.sum(products, 1)
            + 4 * correction_cost * mismatch
        )
        index = tl.argmin(costs, 0, tie_break_left=True)
        cost = tl.min(costs, 0)
        correction = tl.sum(tl.where(candidates == index, correction_axis, 0), 0)
        parity_correction = tl.sum(tl.where(candidates == index, mismatch.to(tl.int32), 0), 0)
        sign_code = tl.sum(tl.where(coordinates < 7, negative.to(tl.int32) << coordinates, 0), 0)
        sign_code ^= tl.where((parity_correction != 0) & (correction < 7), 1 << correction, 0)
        code = index * 128 + sign_code + (32768 if shift_index == 0 else 0)
        selected = tl.where(cost < minimum_cost, code, selected)
        minimum_cost = tl.minimum(cost, minimum_cost)
    return selected


@triton.jit
def nearest_lattice(inputs, magnitudes, output):
    sample = tl.program_id(0).to(tl.int64)
    coordinates = tl.arange(0, 8)
    candidates = tl.arange(0, 256)
    values = tl.load(inputs + sample * 8 + coordinates).to(tl.float32)
    absolute_points = tl.load(magnitudes + candidates[:, None] * 8 + coordinates[None, :])
    point_norms = tl.sum(absolute_points * absolute_points, 1)
    point_parities = tl.sum(absolute_points, 1).to(tl.int32) & 1
    code = nearest_lattice_code(values, absolute_points, point_norms, point_parities)
    tl.store(output + sample, code)


def search(inputs: Tensor, magnitudes: Tensor) -> Tensor:
    flattened = inputs.reshape(-1, 8).contiguous()
    indices = torch.empty(len(flattened), device=inputs.device, dtype=torch.int32)
    # One vector occupies a program; two warps hold its candidates with the least idle width.
    nearest_lattice[(len(flattened),)](
        flattened,
        magnitudes.contiguous(),
        indices,
        num_warps=2,
        enable_fp_fusion=False,
    )
    return indices.reshape(inputs.shape[:-1])
