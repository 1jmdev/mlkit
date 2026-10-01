"""A linear layer that executes four-bit scalar-grid codes with fused CUDA kernels."""

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mlkit.kernels.packed_linear import decode, matrix_vector
from mlkit.models.module_utilities import preserve_input_processing
from mlkit.quantization.operations import pack
from mlkit.quantization.representation import Q


class PackedLinear(nn.Module):
    """Exact scalar-grid codes, fused CUDA decode, and dense prefill fallback."""

    packed: Tensor
    scales: Tensor
    values: Tensor
    zeros: Tensor | None
    bias: Tensor | None
    _dense_weight: Tensor | None

    def __init__(self, original: nn.Linear, quantized: Q, *, cache_dense: bool = False) -> None:
        super().__init__()
        if (quantized.codec not in {"scaled", "feedback"}
                or quantized.metadata.get("code_bits") != 4):
            raise ValueError("PackedLinear requires a four-bit scalar-grid codec")
        if quantized.params.get("offset", 0) != 0:
            raise ValueError("PackedLinear requires a complete fitted region")
        if quantized.params.get("permutation") is not None:
            raise ValueError("PackedLinear requires weights in their original column order")
        if quantized.params.get("refit", quantized.params["group"]) % quantized.params["group"]:
            raise ValueError("PackedLinear requires refit boundaries aligned with scale groups")
        assert quantized.codes is not None
        device = original.weight.device
        if device.type != "cuda":
            raise ValueError("PackedLinear requires CUDA weights")
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.group = quantized.params["group"]
        self.storage_dtype = original.weight.dtype
        self.cache_dense = cache_dense
        self.register_buffer("packed", pack(quantized.codes, 4).to(device))
        scale_dtype = {
            "fp16": torch.float16, "bf16": torch.bfloat16,
        }.get(str(quantized.metadata.get("scale_fmt")), quantized.params["scales"].dtype)
        self.register_buffer(
            "scales", quantized.params["scales"].to(device=device, dtype=scale_dtype).contiguous()
        )
        self.register_buffer("values", quantized.params["values"].to(device).contiguous())
        zero = quantized.params.get("zero")
        self.register_buffer("zeros", None if zero is None else zero.to(
            device=device, dtype=scale_dtype
        ).contiguous())
        self.register_buffer(
            "bias", None if original.bias is None else original.bias.detach().clone()
        )
        self.register_buffer("_dense_weight", None, persistent=False)
        self.maximum_fused_rows = 1
        scalar_values = quantized.params["values"].cpu().float()
        differences = scalar_values.diff()
        self.uniform_grid = bool(len(differences) and torch.allclose(
            differences, differences[0].expand_as(differences)
        ))
        self.grid_minimum = float(scalar_values[0])
        self.grid_step = float(differences[0]) if len(differences) else 1.0
        preserve_input_processing(original, self)

    @property
    def weight(self) -> Tensor:
        if self._dense_weight is not None:
            return self._dense_weight
        reconstruction = torch.empty(
            (self.out_features, self.in_features), device=self.packed.device,
            dtype=self.storage_dtype,
        )
        decode(self.packed, self.scales, self.values, self.zeros, reconstruction,
               self.group, self.uniform_grid, self.grid_minimum, self.grid_step)
        if self.cache_dense:
            self._dense_weight = reconstruction
        return reconstruction

    def forward(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape
        if inputs.device.type != "cuda":
            raise ValueError("packed inference requires CUDA inputs")
        flattened = inputs.reshape(-1, shape[-1]).contiguous()
        if flattened.shape[1] != self.in_features:
            raise ValueError("input width does not match the packed linear")
        if len(flattened) <= self.maximum_fused_rows:
            if torch.is_grad_enabled() and inputs.requires_grad:
                raise RuntimeError("packed CUDA inference does not support autograd")
            output = inputs.new_empty((len(flattened), self.out_features))
            matrix_vector(
                flattened, self.packed, self.scales, self.values, self.zeros, self.bias,
                output, self.group, uniform_grid=self.uniform_grid,
                grid_minimum=self.grid_minimum, grid_step=self.grid_step,
            )
            return output.reshape(*shape[:-1], self.out_features)
        return functional.linear(inputs, self.weight.to(inputs.dtype), self.bias)


def packed_compatible(quantized: Q) -> bool:
    """Require a complete scalar codec with group-aligned fitting regions."""
    if quantized.codec not in {"scaled", "feedback"}:
        return False
    group = quantized.params["group"]
    return (
        quantized.metadata.get("code_bits") == 4
        and quantized.params.get("offset", 0) == 0
        and quantized.params.get("permutation") is None
        and quantized.params.get("refit", group) % group == 0
    )
