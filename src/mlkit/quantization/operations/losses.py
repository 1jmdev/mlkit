"""Layer reconstruction losses used to compare candidate quantizations."""

import torch
from torch import Tensor

from mlkit.quantization.context import Ctx


def proxy_loss(weight: Tensor, reconstruction: Tensor, ctx: Ctx | None = None) -> Tensor:
    """Mean squared output error per output channel (or weight MSE without H)."""
    error = weight.float() - reconstruction.float()
    if ctx is None:
        return error.square().mean()
    weighted = error @ ctx.H.to(error.device)
    return torch.vdot(weighted.flatten(), error.flatten()) / weight.shape[0]
