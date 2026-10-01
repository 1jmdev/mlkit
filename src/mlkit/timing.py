"""Wall-clock measurement that accounts for asynchronous CUDA execution."""

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
