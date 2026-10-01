"""Flattened composition of portable codecs with basis changes and channel scales."""

from typing import Any

import torch
from torch import Tensor

from mlkit.quantization.codecs.registry import codec, decoder, registered
from mlkit.quantization.operations.orthogonal_transforms import structured_transform
from mlkit.quantization.representation import Q


def deterministic_signs(width: int, seed: int, device: torch.device) -> Tensor:
    """Generate the same Rademacher signs on every storage and execution device."""
    values = (torch.arange(width, device=device, dtype=torch.int64) + seed) & 0xFFFFFFFF
    for _ in range(2):
        values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values ^= values >> 16
    return (values & 1).float() * 2 - 1


@codec("basis")
def decode_basis(
    codes: Tensor,
    *,
    inner_codec: str,
    shape: tuple[int, int] | list[int],
    left_seed: int | None,
    right_seed: int | None,
    left_signs: Tensor | None,
    right_signs: Tensor | None,
    **parameters: Any,
) -> Tensor:
    inner = {name.removeprefix("inner_"): value for name, value in parameters.items()}
    reconstruction = decoder(inner_codec)(codes, **inner)
    if right_seed is not None:
        signs = right_signs if right_signs is not None else deterministic_signs(
            shape[1], right_seed, codes.device
        )
        reconstruction = structured_transform(reconstruction, inverse=True) * signs
    if left_seed is not None:
        signs = left_signs if left_signs is not None else deterministic_signs(
            shape[0], left_seed, codes.device
        )
        reconstruction = structured_transform(reconstruction.T, inverse=True).T * signs[:, None]
    return reconstruction


@codec("channel_scaled")
def decode_channel_scaled(
    codes: Tensor,
    *,
    inner_codec: str,
    channel_scales: Tensor,
    **parameters: Any,
) -> Tensor:
    inner = {name.removeprefix("inner_"): value for name, value in parameters.items()}
    return decoder(inner_codec)(codes, **inner) / channel_scales


def compose(quantized: Q, name: str, parameters: dict[str, Any]) -> Q:
    if quantized.codes is None or quantized.codec is None or not registered(quantized.codec):
        raise ValueError("portable composition requires a registered inner codec")
    metadata = dict(quantized.metadata)
    metadata["trainable"] = [
        f"inner_{parameter}" for parameter in metadata.get("trainable", [])
    ]
    formats = dict(metadata.get("parameter_formats", {}))
    if quantized.codec in {"scaled", "feedback", "vector_scaled", "vector_feedback"}:
        formats["scales"] = formats["zero"] = metadata.get("scale_fmt", "fp32")
    metadata["parameter_formats"] = {f"inner_{key}": value for key, value in formats.items()}
    metadata["parameter_bits"] = {
        f"inner_{key}": value for key, value in metadata.get("parameter_bits", {}).items()
    }
    inner_parameters = {f"inner_{key}": value for key, value in quantized.params.items()}
    return Q(
        codes=quantized.codes,
        params=inner_parameters | {"inner_codec": quantized.codec} | parameters,
        decode=decoder(name),
        codec=name,
        bits=quantized.bits,
        metadata=metadata,
    )
