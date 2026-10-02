"""A linear layer that executes packed scalar-grid codes with fused CUDA kernels."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
import triton
from torch import Tensor, nn

from mlkit.kernels import launching
from mlkit.kernels.packed_linear import (
    MAXIMUM_CODE_BITS,
    ROW_TILE,
    decode,
    layout_for,
    matrix_vector,
    packed_matrix_vector,
    sample_tile_for,
)
from mlkit.models.module_utilities import preserve_input_processing
from mlkit.quantization.codecs import decode_scaled
from mlkit.quantization.operations import pack, unpack
from mlkit.quantization.representation import Q

SCALE_STORAGE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
MAXIMUM_FUSED_ROWS = 8


class DirectLaunch:
    """Compiled matrix-vector kernels shared by every layer, by specialization.

    A specialization is selected by tensor dtypes and constant kernel arguments.
    ``available`` turns false if this Triton release cannot be launched directly.
    """

    available = True
    compiled: dict[tuple[Any, ...], Any] = {}


class PackedLinear(nn.Module):
    """Exact scalar-grid codes of one to eight bits, fused CUDA decode, and dense prefill.

    Up to ``maximum_fused_rows`` input rows are multiplied by the packed codes
    directly. Larger inputs reconstruct the dense weight once and use an
    ordinary matrix product. A layer whose rows or scale groups split a packing
    word keeps its packed storage and always reconstructs the dense weight.
    """

    packed: Tensor
    scales: Tensor
    values: Tensor
    zeros: Tensor | None
    bias: Tensor | None
    _dense_weight: Tensor | None

    def __init__(self, original: nn.Linear, quantized: Q, *, cache_dense: bool = False) -> None:
        super().__init__()
        if not packed_compatible(quantized):
            raise ValueError(
                "PackedLinear requires a complete scalar-grid codec of at most eight bits in "
                "its original column order with refit boundaries aligned to scale groups"
            )
        assert quantized.codes is not None
        device = original.weight.device
        if device.type != "cuda":
            raise ValueError("PackedLinear requires CUDA weights")
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.group = quantized.params["group"]
        self.code_bits = int(quantized.metadata["code_bits"])
        self.storage_dtype = original.weight.dtype
        self.cache_dense = cache_dense
        self.register_buffer("packed", pack(quantized.codes, self.code_bits).to(device))
        scales = quantized.params["scales"]
        scale_dtype = SCALE_STORAGE_DTYPES.get(
            str(quantized.metadata.get("scale_fmt")), scales.dtype
        )
        self.register_buffer("scales", scales.to(device=device, dtype=scale_dtype).contiguous())
        self.register_buffer("values", quantized.params["values"].to(device).contiguous())
        zero = quantized.params.get("zero")
        self.register_buffer(
            "zeros",
            None if zero is None else zero.to(device=device, dtype=scale_dtype).contiguous(),
        )
        self.register_buffer(
            "bias", None if original.bias is None else original.bias.detach().clone()
        )
        self.register_buffer("_dense_weight", None, persistent=False)
        scalar_values = quantized.params["values"].cpu().float()
        differences = scalar_values.diff()
        self.uniform_grid = bool(
            len(differences)
            and torch.allclose(differences, differences[0].expand_as(differences))
        )
        self.grid_minimum = float(scalar_values[0])
        self.grid_step = float(differences[0]) if len(differences) else 1.0
        self.layout = layout_for(self.code_bits, self.in_features, self.group)
        self.maximum_fused_rows = 0 if self.layout is None else MAXIMUM_FUSED_ROWS
        self._constant_arguments: tuple[Any, ...] = ()
        self._specialization: tuple[Any, ...] = ()
        preserve_input_processing(original, self)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> "PackedLinear":
        self._constant_arguments = ()
        self._specialization = ()
        self._dense_weight = None
        return super()._apply(fn, recurse)

    @property
    def weight(self) -> Tensor:
        if self._dense_weight is not None:
            return self._dense_weight
        if self.layout is None:
            codes = unpack(self.packed, self.code_bits, (self.out_features, self.in_features))
            reconstruction = decode_scaled(
                codes,
                scales=self.scales.float(),
                values=self.values,
                group=self.group,
                zero=None if self.zeros is None else self.zeros.float(),
            ).to(self.storage_dtype)
        else:
            reconstruction = torch.empty(
                (self.out_features, self.in_features),
                device=self.packed.device,
                dtype=self.storage_dtype,
            )
            decode(
                self.packed,
                self.scales,
                self.values,
                self.zeros,
                reconstruction,
                self.group,
                self.layout,
                self.uniform_grid,
                self.grid_minimum,
                self.grid_step,
            )
        if self.cache_dense:
            self._dense_weight = reconstruction
        return reconstruction

    def forward(self, inputs: Tensor) -> Tensor:
        width = self.in_features
        shape = inputs.shape
        if shape[-1] != width:
            raise ValueError("input width does not match the packed linear")
        if not inputs.is_cuda:
            raise ValueError("packed inference requires CUDA inputs")
        rows = inputs.numel() // width
        if rows > self.maximum_fused_rows:
            return functional.linear(inputs, self.weight.to(inputs.dtype), self.bias)
        if inputs.requires_grad and torch.is_grad_enabled():
            raise RuntimeError("packed CUDA inference does not support autograd")
        flattened = inputs.reshape(rows, width)
        if not flattened.is_contiguous():
            flattened = flattened.contiguous()
        output = inputs.new_empty((rows, self.out_features))
        if torch.compiler.is_compiling() or not DirectLaunch.available:
            assert self.layout is not None
            matrix_vector(
                flattened,
                self.packed,
                self.scales,
                self.values,
                self.zeros,
                self.bias,
                output,
                self.group,
                self.layout,
                uniform_grid=self.uniform_grid,
                grid_minimum=self.grid_minimum,
                grid_step=self.grid_step,
            )
        else:
            self._launch_directly(flattened, output, rows)
        return output.view(shape[:-1] + (self.out_features,))

    def _launch_directly(self, inputs: Tensor, output: Tensor, rows: int) -> None:
        """Launch the matrix-vector kernel without Triton's per-call argument binding.

        The constant arguments and the specialization they select are cached and
        rebuilt after the buffers change. The compiler path uses the public
        Triton call instead, because that is what it captures into its graph.
        """
        scales, zeros, bias = self.scales, self.zeros, self.bias
        constants = self._constant_arguments
        if not constants:
            layout = self.layout
            assert layout is not None
            constants = self._constant_arguments = (
                self.in_features,
                self.out_features,
                scales.shape[1],
                self.group,
                layout.tiles_per_row(self.in_features),
                zeros is not None,
                bias is not None,
                self.uniform_grid,
                self.grid_minimum,
                self.grid_step,
                ROW_TILE,
                layout.bits,
                layout.bytes_per_word,
                layout.codes_per_word,
                layout.tile_words,
            )
            self._specialization = (
                scales.device.index,
                scales.dtype,
                None if zeros is None else zeros.dtype,
                None if bias is None else bias.dtype,
                *constants[5:],
            )
        sample_tile = sample_tile_for(rows)
        arguments = (
            inputs,
            self.packed,
            scales,
            self.values,
            zeros,
            bias,
            output,
            *constants[:5],
            rows,
            *constants[5:],
            sample_tile,
        )
        grid = (triton.cdiv(self.out_features, ROW_TILE), triton.cdiv(rows, sample_tile))
        specialization = (inputs.dtype, sample_tile, self._specialization)
        compiled = DirectLaunch.compiled.get(specialization)
        if compiled is None:
            compiled = launching.compile_and_launch(packed_matrix_vector, grid, arguments)
            if compiled is None:
                DirectLaunch.available = False
            else:
                DirectLaunch.compiled[specialization] = compiled
            return
        try:
            launching.launch(compiled, grid, arguments)
        except TypeError:
            DirectLaunch.available = False
            packed_matrix_vector[grid](*arguments)


def packed_compatible(quantized: Q) -> bool:
    """Require a complete scalar codec with group-aligned fitting regions."""
    if quantized.codec not in {"scaled", "feedback"}:
        return False
    group = quantized.params["group"]
    code_bits = quantized.metadata.get("code_bits")
    return (
        isinstance(code_bits, int)
        and 1 <= code_bits <= MAXIMUM_CODE_BITS
        and quantized.codes is not None
        and quantized.params["values"].numel() <= 2**code_bits
        and quantized.params.get("offset", 0) == 0
        and quantized.params.get("permutation") is None
        and quantized.params.get("refit", group) % group == 0
    )
