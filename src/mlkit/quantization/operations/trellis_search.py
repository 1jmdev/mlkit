"""Computed Gaussian codes and exact free-start Viterbi search on a bitshift trellis."""

import torch
from torch import Tensor

from mlkit.kernels.viterbi_search import search


def one_mad(states: Tensor) -> Tensor:
    scrambled = (34_038_481 * states.long() + 76_625_530) & 0xFFFFFFFF
    byte_sum = torch.zeros_like(scrambled)
    for shift in (0, 8, 16, 24):
        byte_sum += (scrambled >> shift) & 255
    return (byte_sum.float() - 510) / 147.8


def viterbi(
    value: Tensor,
    codes: Tensor,
    L: int,
    k: int,
    *,
    backend: str = "auto",
    return_states: bool = False,
) -> Tensor:
    """Find the exact least-squares free-start path on a bitshift trellis."""
    if not 1 <= k <= min(8, L) or not 1 <= L <= 16:
        raise ValueError("trellis requires 1 <= k <= min(8, L) and 1 <= L <= 16")
    if value.ndim != 2 or value.shape[1] < 1 or codes.numel() != 2**L:
        raise ValueError("viterbi requires [sequences, time] inputs and exactly 2**L code values")
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("Viterbi backend must be auto, torch, or triton")
    codes = codes.to(device=value.device, dtype=torch.float32)
    if value.device.type == "cuda" and backend != "torch" and L <= 12:
        states = search(value.float().contiguous(), codes, L, k)
    else:
        if backend == "triton":
            raise ValueError("fused Viterbi requires CUDA and L <= 12")
        batch, length = value.shape
        lower_bits = L - k
        predecessors = 2**lower_bits
        cost = (value[:, :1].float() - codes).square()
        back = torch.empty(
            length - 1, batch, predecessors, dtype=torch.uint8, device=value.device
        )
        for time in range(1, length):
            minimum, indices = cost.reshape(batch, 2**k, predecessors).min(1)
            back[time - 1] = indices.to(torch.uint8)
            cost = minimum.repeat_interleave(2**k, dim=1) + (
                value[:, time : time + 1].float() - codes
            ).square()
        state = cost.argmin(1)
        states = torch.empty(batch, length, dtype=torch.int32, device=value.device)
        states[:, -1] = state
        for time in range(length - 2, -1, -1):
            lower = state >> k
            predecessor = back[time].gather(1, lower[:, None])[:, 0].long()
            state = predecessor * predecessors + lower
            states[:, time] = state
    return states if return_states else codes[states.long()].to(value.dtype)
