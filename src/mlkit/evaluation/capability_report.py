"""A report of the fused kernels, checkpoints and backends a quantizer qualifies for."""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor

from mlkit.inference.packed_linear import packed_compatible
from mlkit.kernels.packed_linear import layout_for
from mlkit.quantization.algorithms.error_feedback import (
    FUSED_CODEBOOK_LIMIT,
    FUSED_VECTOR_CODEBOOK_LIMIT,
    FUSED_VECTOR_DIMENSIONS,
)
from mlkit.quantization.codecs import registered
from mlkit.quantization.context import Ctx
from mlkit.quantization.formats.grouped_scaling import Scaled
from mlkit.quantization.formats.scale_storage import SCALE_FORMAT_BITS
from mlkit.quantization.protocol import FittedRounder, FunctionQuantizer, Quantizer
from mlkit.quantization.representation import Q, as_q


@dataclass
class Capabilities:
    """What one quantizer supports, measured on a random matrix.

    ``checkpoint`` tells whether codes are stored or the dense reconstruction.
    ``error_feedback`` tells how ``gptq`` and ``ldlq`` round with it as their
    inner format. ``packed_inference`` tells whether ``optimize`` executes its
    codes without reconstructing the weight.
    """

    quantizer: str
    bits_per_weight: float | None
    codec: str | None
    checkpoint: str
    error_feedback: str
    packed_inference: str
    online_activations: str

    def __str__(self) -> str:
        names = [field.name.replace("_", " ") for field in fields(self)]
        width = max(len(name) for name in names)
        values = [getattr(self, field.name) for field in fields(self)]
        return "\n".join(
            f"{name.ljust(width)}  {format_capability(value)}"
            for name, value in zip(names, values, strict=True)
        )


def format_capability(value: Any) -> str:
    if value is None:
        return "unknown"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def capabilities(quantization: Any, shape: tuple[int, int] = (64, 256)) -> Capabilities:
    """Round a random matrix with ``quantization`` and report what it qualifies for.

    The matrix and its calibration inputs are generated on CUDA with a fixed
    seed, so calibrated quantizers can be inspected without a model. The bit
    cost includes side information and therefore depends on ``shape``.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(shape, device="cuda", generator=generator)
    inputs = torch.randn((4 * shape[1], shape[1]), device="cuda", generator=generator)
    context = Ctx("capabilities", X=inputs)
    with torch.no_grad():
        result = as_q(quantization(weight, context))
        reconstruction = result.w
    if reconstruction.shape != weight.shape or not torch.isfinite(reconstruction).all():
        raise ValueError("the quantizer returned an invalid reconstruction")
    bits = None if result.bits is None else result.bits + context.additional_bits
    portable = result.codes is not None and registered(result.codec)
    return Capabilities(
        quantizer=repr(quantization),
        bits_per_weight=None if bits is None else bits / weight.numel(),
        codec=result.codec,
        checkpoint="codes with a registered codec" if portable else "dense reconstruction",
        error_feedback=error_feedback_path(quantization, weight),
        packed_inference=packed_inference_path(result, shape[1]),
        online_activations=online_activation_path(quantization),
    )


def error_feedback_path(quantization: Any, weight: Tensor) -> str:
    if isinstance(quantization, FunctionQuantizer) or not isinstance(quantization, Quantizer):
        return "reference rounding; the function is called on every column block"
    if type(quantization).fit is Quantizer.fit:
        return "not an inner format; it does not implement Quantizer.fit"
    context = Ctx("capabilities", X=weight.new_zeros((1, weight.shape[1])))
    with torch.no_grad():
        rounder = quantization.fit(weight, context)
    if isinstance(rounder, FittedRounder):
        scalar, vector = rounder.scalar, rounder.vector
        if (
            scalar is not None
            and scalar.grid.bits <= 8
            and scalar.values.numel() <= FUSED_CODEBOOK_LIMIT
        ):
            return "fused scalar kernel at step=1"
        if (
            vector is not None
            and vector.grid.dim in FUSED_VECTOR_DIMENSIONS
            and vector.codebook_size <= FUSED_VECTOR_CODEBOOK_LIMIT
        ):
            return f"fused vector kernel at step={vector.grid.dim}"
    return "reference rounding with the fitted rounder"


def packed_inference_path(result: Q, input_width: int) -> str:
    if not registered(result.codec) or result.codec not in {"scaled", "feedback"}:
        return "no; dense weights"
    if not packed_compatible(result):
        return "no; dense weights"
    layout = layout_for(result.metadata["code_bits"], input_width, result.params["group"])
    if layout is None:
        return "packed storage with dense execution; rows or groups split a packing word"
    return "fused kernel"


def online_activation_path(quantization: Any) -> str:
    if not isinstance(quantization, Scaled):
        return "reference rounding; not restorable from a checkpoint"
    fused = (
        quantization.grid.dim == 1
        and quantization.grid.values is not None
        and quantization.scale == "absmax"
        and isinstance(quantization.scale_fmt, str)
        and quantization.scale_fmt in SCALE_FORMAT_BITS
    )
    return "fused kernel" if fused else "reference rounding"
