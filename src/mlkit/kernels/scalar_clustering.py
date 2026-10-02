"""Fused Lloyd iteration for scalar codebooks.

Each program assigns a block of samples to their nearest centers and writes the
weighted sum and the total weight of every cluster within the block. The
partial results of all blocks are then added in a fixed order, so a clustering
is reproducible from run to run.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from mlkit.kernels.scalar_rounding import nearest_codebook_index

MAXIMUM_CLUSTERS = 256
TILE_ELEMENTS = 4096

RUNTIME_ARGUMENTS = ["samples", "weights", "centers", "sums", "totals", "count"]


@triton.jit(do_not_specialize=RUNTIME_ARGUMENTS, do_not_specialize_on_alignment=RUNTIME_ARGUMENTS)
def accumulate_clusters(
    samples,
    weights,
    centers,
    sums,
    totals,
    count,
    has_weights: tl.constexpr,
    cluster_count: tl.constexpr,
    search_steps: tl.constexpr,
    cluster_tile: tl.constexpr,
    block: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    positions = program * block + tl.arange(0, block)
    valid = positions < count
    values = tl.load(samples + positions, valid, other=0)
    index = nearest_codebook_index(values, centers, cluster_count, search_steps)
    if has_weights:
        importance = tl.load(weights + positions, valid, other=0)
    else:
        importance = tl.where(valid, 1.0, 0.0)
    clusters = tl.arange(0, cluster_tile)
    member = index[:, None] == clusters[None, :]
    weighted = (values * importance)[:, None]
    outputs = program * cluster_tile + clusters
    tl.store(sums + outputs, tl.sum(tl.where(member, weighted, 0.0), axis=0))
    tl.store(totals + outputs, tl.sum(tl.where(member, importance[:, None], 0.0), axis=0))


def accumulate(samples: Tensor, weights: Tensor | None, centers: Tensor) -> tuple[Tensor, Tensor]:
    """Weighted sums and total weights of the samples nearest to each ascending center."""
    count, cluster_count = samples.numel(), centers.numel()
    cluster_tile = triton.next_power_of_2(cluster_count)
    block = TILE_ELEMENTS // cluster_tile
    programs = triton.cdiv(count, block)
    sums = torch.empty((programs, cluster_tile), device=samples.device)
    totals = torch.empty_like(sums)
    accumulate_clusters[(programs,)](
        samples,
        weights,
        centers,
        sums,
        totals,
        count,
        weights is not None,
        cluster_count,
        math.ceil(math.log2(cluster_count + 1)),
        cluster_tile,
        block,
        enable_fp_fusion=False,
    )
    return sums.sum(0)[:cluster_count], totals.sum(0)[:cluster_count]
