"""Small helpers for inspecting and substituting PyTorch modules."""

from typing import Any

import torch
from torch import Tensor, nn


def extract_hidden(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError("block output must be a tensor, tuple, or last_hidden_state result")


def preserve_input_processing(original: nn.Module, replacement: nn.Module) -> None:
    """Preserve online transforms when an inference or training layer is substituted."""
    replacement._forward_pre_hooks = original._forward_pre_hooks.copy()
    replacement._forward_pre_hooks_with_kwargs = original._forward_pre_hooks_with_kwargs.copy()
    for name, value in original.named_buffers(recurse=False):
        if name.startswith("_mlkit_"):
            replacement.register_buffer(name, value.detach().clone())


def weight_name(module_name: str) -> str:
    return f"{module_name}.weight" if module_name else "weight"


def module_device(module: nn.Module) -> torch.device:
    tensor: Tensor | None = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return torch.device("cuda") if tensor is None else tensor.device
