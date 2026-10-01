"""One-program-per-sequence fused free-start Viterbi search."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def search_sequence(
    inputs,
    codes,
    traceback,
    output,
    length: tl.constexpr,
    state_bits: tl.constexpr,
    transition_bits: tl.constexpr,
):
    sample = tl.program_id(0)
    state_count: tl.constexpr = 1 << state_bits
    predecessor_count: tl.constexpr = 1 << (state_bits - transition_bits)
    states = tl.arange(0, state_count)
    predecessors = tl.arange(0, predecessor_count)
    values = tl.load(codes + states)
    initial = tl.load(inputs + sample * length)
    costs = (initial - values) * (initial - values)
    for time in range(1, length):
        grouped = tl.reshape(costs, (1 << transition_bits, predecessor_count))
        minimum = tl.min(grouped, axis=0)
        indices = tl.argmin(grouped, axis=0, tie_break_left=True)
        tl.store(
            traceback + sample * (length - 1) * predecessor_count
            + (time - 1) * predecessor_count + predecessors, indices,
        )
        previous = tl.gather(minimum, states >> transition_bits, axis=0)
        value = tl.load(inputs + sample * length + time)
        costs = previous + (value - values) * (value - values)
    state = tl.argmin(costs, axis=0, tie_break_left=True)
    tl.store(output + sample * length + length - 1, state)
    for offset in range(length - 1):
        time = length - 2 - offset
        lower = state >> transition_bits
        high = tl.load(
            traceback + sample * (length - 1) * predecessor_count + time * predecessor_count + lower
        ).to(tl.int32)
        state = (high << (state_bits - transition_bits)) | lower
        tl.store(output + sample * length + time, state)


def search(inputs: Tensor, codes: Tensor, L: int, k: int) -> Tensor:
    traceback = torch.empty(
        (len(inputs), inputs.shape[1] - 1, 2 ** (L - k)), device=inputs.device, dtype=torch.uint8
    )
    output = torch.empty_like(inputs, dtype=torch.int32)
    search_sequence[(len(inputs),)](
        inputs, codes, traceback, output, inputs.shape[1], L, k,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
