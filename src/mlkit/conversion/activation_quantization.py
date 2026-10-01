"""Online activation hooks and explicit scalar-format descriptors."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from mlkit.quantization.context import Ctx
from mlkit.quantization.formats import Scaled, scaled
from mlkit.quantization.grids import grid
from mlkit.quantization.representation import as_q


def describe_quantizer(quantizer: Any) -> dict[str, Any] | None:
    if (
        not isinstance(quantizer, Scaled)
        or quantizer.grid.dim != 1
        or quantizer.grid.values is None
        or not isinstance(quantizer.scale, str)
        or not isinstance(quantizer.scale_fmt, str)
    ):
        return None
    return {
        "kind": "scaled",
        "bits": quantizer.grid.bits,
        "grid": quantizer.grid.name,
        "integer": quantizer.grid.integer,
        "values": quantizer.grid.values.detach().cpu().tolist(),
        "group": quantizer.group,
        "scale": quantizer.scale,
        "scale_fmt": quantizer.scale_fmt,
        "asym": quantizer.asym,
        "search_steps": quantizer.search_steps,
    }


def restore_quantizer(descriptor: dict[str, Any]) -> Scaled:
    if descriptor["kind"] != "scaled":
        raise ValueError(f"unsupported online quantizer {descriptor['kind']!r}")
    # Checkpoints written before the explicit flag identify integer grids by name.
    if descriptor.get("integer", descriptor["grid"].startswith("int")):
        representation = grid.int(descriptor["bits"])
    else:
        representation = grid.values(
            torch.tensor(descriptor["values"]), bits=descriptor["bits"]
        )
    return scaled(
        representation,
        group=descriptor["group"],
        scale=descriptor["scale"],
        scale_fmt=descriptor["scale_fmt"],
        asym=descriptor["asym"],
        search_steps=descriptor["search_steps"],
    )


class ActivationHook:
    """A copyable hook whose context follows the converted model's modules."""

    def __init__(self, quantizer: Callable, context: Ctx) -> None:
        self.quantizer = quantizer
        self.context = context

    def __call__(self, layer: nn.Module, arguments: tuple) -> tuple:
        inputs = arguments[0]
        shape = inputs.shape
        flattened = inputs.reshape(-1, shape[-1])
        if isinstance(self.quantizer, Scaled):
            reconstruction = self.quantizer.reconstruct_activations(flattened)
        else:
            reconstruction = as_q(self.quantizer(flattened.float(), self.context)).w
            reconstruction = reconstruction.to(inputs.dtype)
        return (reconstruction.reshape(shape), *arguments[1:])


def install_activation_quantization(
    module: nn.Module, quantizer: Callable, context: Ctx,
) -> Any:
    return module.register_forward_pre_hook(ActivationHook(quantizer, context))
