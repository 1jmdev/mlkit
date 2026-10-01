"""Parallel decoding of independent bitshift-trellis states into matrix weights."""

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 1024


@triton.jit
def decode_states(
    transitions,
    initial_states,
    scale,
    codebook,
    output,
    elements,
    height,
    width,
    tile,
    state_bits: tl.constexpr,
    transition_bits: tl.constexpr,
    custom_codebook: tl.constexpr,
    block: tl.constexpr,
):
    indices = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = indices < elements
    rows = (indices // width).to(tl.int32)
    columns = (indices % width).to(tl.int32)
    sequence = (columns // tile) * (height // tile) + rows // tile
    time = (rows % tile) * tile + columns % tile
    initial = tl.load(initial_states + sequence, valid, other=0).to(tl.int32)
    state = tl.where(
        time * transition_bits < state_bits,
        initial << tl.minimum(time * transition_bits, state_bits),
        0,
    )
    for offset in tl.static_range((state_bits + transition_bits - 1) // transition_bits):
        position = time - 1 - offset
        incoming = tl.load(
            transitions + sequence.to(tl.int64) * (tile * tile - 1) + position,
            valid & (position >= 0),
            other=0,
        ).to(tl.int32)
        state |= incoming << (offset * transition_bits)
    state &= (1 << state_bits) - 1
    if custom_codebook:
        reconstruction = tl.load(codebook + state).to(tl.float32)
    else:
        scrambled = (34038481 * state.to(tl.uint32) + 76625530).to(tl.uint32)
        byte_sum = (
            (scrambled & 255)
            + ((scrambled >> 8) & 255)
            + ((scrambled >> 16) & 255)
            + ((scrambled >> 24) & 255)
        )
        reconstruction = (byte_sum.to(tl.float32) - 510.0) / 147.8
    tl.store(output + indices, reconstruction * tl.load(scale), valid)


def decode(
    transitions: Tensor,
    initial_states: Tensor,
    scale: Tensor,
    shape: tuple[int, int] | list[int],
    tile: int,
    L: int,
    k: int,
    codebook: Tensor | None,
) -> Tensor:
    output = torch.empty(tuple(shape), device=transitions.device, dtype=torch.float32)
    decode_states[(triton.cdiv(output.numel(), BLOCK),)](
        transitions.contiguous(),
        initial_states.contiguous(),
        scale,
        codebook,
        output,
        output.numel(),
        shape[0],
        shape[1],
        tile,
        L,
        k,
        codebook is not None,
        BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
