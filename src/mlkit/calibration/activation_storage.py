"""Placement of captured activations on CUDA or the host."""

import torch
from torch import Tensor

STORAGE_POLICIES = ("auto", "cuda", "host")
MINIMUM_RESERVE_BYTES = 2**30


class ActivationStorage:
    """Keeps captured activations on CUDA while memory allows and on the host otherwise.

    ``auto`` keeps a tensor on its device only if a quarter of the device memory,
    and at least one gibibyte, remains available afterwards for statistics,
    factorizations and block passes. ``cuda`` never moves tensors and ``host``
    always does.
    """

    def __init__(self, policy: str = "auto", reserve_fraction: float = 0.25) -> None:
        if policy not in STORAGE_POLICIES:
            raise ValueError("calibration storage must be auto, cuda, or host")
        if not 0 <= reserve_fraction < 1:
            raise ValueError("the reserved memory fraction must be in [0, 1)")
        self.policy = policy
        self.reserve_fraction = reserve_fraction

    def fits(self, tensor: Tensor) -> bool:
        """Whether the device retains its reserve after keeping ``tensor`` resident."""
        free, total = torch.cuda.mem_get_info(tensor.device)
        cached = torch.cuda.memory_reserved(tensor.device) - torch.cuda.memory_allocated(
            tensor.device
        )
        reserve = max(MINIMUM_RESERVE_BYTES, int(self.reserve_fraction * total))
        return free + cached - tensor.numel() * tensor.element_size() >= reserve

    def store(self, tensor: Tensor, *, copy: bool) -> Tensor:
        """Detach ``tensor`` for later replay; ``copy`` protects against in-place reuse."""
        tensor = tensor.detach()
        if not tensor.is_cuda:
            return tensor
        if self.policy == "host" or (self.policy == "auto" and not self.fits(tensor)):
            return tensor.cpu()
        return tensor.clone() if copy else tensor
