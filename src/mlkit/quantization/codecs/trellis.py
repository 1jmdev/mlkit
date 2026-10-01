"""Decoder for tiled bitshift-trellis transitions."""

import torch
from torch import Tensor

from mlkit.kernels.trellis_decode import decode
from mlkit.quantization.codecs.registry import codec
from mlkit.quantization.operations.trellis_search import one_mad


@codec("trellis")
def decode_trellis(
    transitions: Tensor,
    *,
    initial_states: Tensor,
    scale: Tensor,
    L: int,
    k: int,
    shape: tuple[int, int] | list[int],
    tile: int,
    codebook: Tensor | None = None,
) -> Tensor:
    needs_gradients = torch.is_grad_enabled() and (
        scale.requires_grad or (codebook is not None and codebook.requires_grad)
    )
    if transitions.is_cuda and not needs_gradients:
        return decode(transitions, initial_states, scale, shape, tile, L, k, codebook)
    time = torch.arange(transitions.shape[1] + 1, device=transitions.device)
    shifts = (time * k).clamp_max(L)
    states = torch.where(time * k < L, initial_states.long()[:, None] << shifts, 0)
    for offset in range((L + k - 1) // k if transitions.shape[1] else 0):
        positions = (time - 1 - offset).clamp_min(0)
        incoming = transitions[:, positions.clamp_max(transitions.shape[1] - 1)].long()
        states |= torch.where(time > offset, incoming << (offset * k), 0)
    states &= 2**L - 1
    reconstruction = one_mad(states) if codebook is None else codebook.to(states.device)[states]
    height, width = shape
    decoded = reconstruction * scale.to(states.device)
    return decoded.reshape(width // tile, height // tile, tile, tile).permute(
        1, 2, 0, 3
    ).reshape(shape)
