"""Explicit inference backends and reproducible latency measurements."""

import copy
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mlkit.engine import forward_batch, synchronize
from mlkit.models import Model, QModel, resolve_device
from mlkit.packing import pack
from mlkit.representation import Q


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
        self.register_buffer("scales", quantized.params["scales"].to(device).contiguous())
        self.register_buffer("values", quantized.params["values"].to(device).contiguous())
        zero = quantized.params.get("zero")
        self.register_buffer("zeros", None if zero is None else zero.to(device).contiguous())
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

    @property
    def weight(self) -> Tensor:
        if self._dense_weight is not None:
            return self._dense_weight
        from mlkit.kernels.packed_linear import decode

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
            from mlkit.kernels.packed_linear import matrix_vector

            output = inputs.new_empty((len(flattened), self.out_features))
            matrix_vector(
                flattened, self.packed, self.scales, self.values, self.zeros, self.bias,
                output, self.group, uniform_grid=self.uniform_grid,
                grid_minimum=self.grid_minimum, grid_step=self.grid_step,
            )
            return output.reshape(*shape[:-1], self.out_features)
        return functional.linear(inputs, self.weight.to(inputs.dtype), self.bias)


def optimize(
    model: Model | nn.Module,
    backend: str = "auto",
    *,
    compile: bool = False,
    mode: str = "reduce-overhead",
    cache_dense: bool = False,
    inplace: bool = False,
) -> Model:
    """Preserve reconstructions while choosing dense or packed execution.

    Packed execution supports scalar four-bit Q codecs. Other formats remain
    dense. TorchAO exports are a separate operation because they requantize.
    """
    if backend not in {"auto", "dense", "packed"}:
        raise ValueError("backend must be auto, dense, or packed")
    wrapped = model if isinstance(model, Model) else Model(model)
    resolve_device(wrapped.device)
    converted = wrapped if inplace else copy.deepcopy(wrapped)
    if isinstance(converted, QModel):
        _ = converted.model_bpw
    if backend == "auto":
        use_packed = isinstance(converted, QModel)
        backend = "packed" if use_packed else "dense"
    if backend == "packed":
        if not isinstance(converted, QModel):
            raise ValueError("packed inference requires a QModel with codec state")
        try:
            import triton  # noqa: F401
        except ImportError as error:
            raise ImportError("packed CUDA inference requires Triton") from error
        replaced = 0
        for name, quantized in converted.quantized.items():
            module = converted.module.get_submodule(name)
            compatible = (
                quantized.codec in {"scaled", "feedback"}
                and quantized.metadata.get("code_bits") == 4
                and quantized.params.get("permutation") is None
                and not quantized.params.get("refit", quantized.params["group"])
                % quantized.params["group"]
            ) if quantized.codec in {"scaled", "feedback"} else False
            if isinstance(module, nn.Linear) and compatible:
                converted.module.set_submodule(name, PackedLinear(
                    module, quantized, cache_dense=cache_dense
                ))
                replaced += 1
        converted.execution_backend = f"packed:{replaced}"
    else:
        converted.execution_backend = "dense"
    if compile:
        converted.module.compile(mode=mode)
        converted.execution_backend += "+compiled"
    return converted


def export_torchao(
    model: Model | nn.Module,
    *,
    bits: int = 4,
    group: int = 128,
    packing_format: str = "tile_packed_to_4d",
) -> Model:
    """Requantize a separate model using a maintained TorchAO execution backend."""
    try:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
        from torchao.quantization.quant_api import Int4PackingFormat
    except ImportError as error:
        raise ImportError("TorchAO export requires uv add 'mlkit[torchao]'") from error
    wrapped = model if isinstance(model, Model) else Model(model)
    module = copy.deepcopy(wrapped.module)
    if bits == 4:
        configuration = Int4WeightOnlyConfig(
            group_size=group, int4_packing_format=Int4PackingFormat(packing_format),
            set_inductor_config=False,
        )
    elif bits == 8:
        configuration = Int8WeightOnlyConfig(version=2, set_inductor_config=False)
    else:
        raise ValueError("TorchAO export supports four-bit or eight-bit weights")
    quantize_(module, configuration)
    converted = Model(module, wrapped.tokenizer, name=wrapped.name)
    converted.execution_backend = f"torchao-int{bits}"
    return converted


@dataclass
class BenchmarkResult:
    median_ms: float
    minimum_ms: float
    percentile_95_ms: float
    repetitions: int
    device: str
    peak_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    operation: Callable[[], Any],
    *,
    device: str | torch.device = "cuda",
    warmup: int = 10,
    repetitions: int = 50,
) -> BenchmarkResult:
    """Synchronize outside each timed interval; exclude warmup and compilation."""
    selected_device = resolve_device(device)
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions must be positive")
    samples = []
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize(selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
        for _ in range(repetitions):
            synchronize(selected_device)
            start = time.perf_counter()
            operation()
            synchronize(selected_device)
            samples.append((time.perf_counter() - start) * 1000)
    ordered = sorted(samples)
    memory = torch.cuda.max_memory_allocated(selected_device)
    return BenchmarkResult(
        statistics.median(samples), min(samples),
        ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        repetitions, str(selected_device), memory,
    )


def benchmark_model(model: Model | nn.Module, batch: Any, **options: Any) -> BenchmarkResult:
    wrapped = model if isinstance(model, Model) else Model(model)
    return benchmark(lambda: forward_batch(wrapped.module, batch), device=wrapped.device, **options)
