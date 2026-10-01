"""Named decoders and flattened composition for portable quantized representations."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from mlkit.quantization.formats import decode_feedback, decode_scaled, decode_vector_scaled
from mlkit.quantization.representation import Q
from mlkit.quantization.rotations import structured_transform
from mlkit.quantization.trellis import decode_trellis


def deterministic_signs(width: int, seed: int, device: torch.device) -> Tensor:
    """Generate the same Rademacher signs on every storage and execution device."""
    values = (torch.arange(width, device=device, dtype=torch.int64) + seed) & 0xFFFFFFFF
    for _ in range(2):
        values = ((values ^ (values >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    values ^= values >> 16
    return (values & 1).float() * 2 - 1


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


def decode_channel_scaled(
    codes: Tensor, *, inner_codec: str, channel_scales: Tensor, **parameters: Any,
) -> Tensor:
    inner = {name.removeprefix("inner_"): value for name, value in parameters.items()}
    return decoder(inner_codec)(codes, **inner) / channel_scales


_DECODERS: dict[str, Callable[..., Tensor]] = {
    "scaled": decode_scaled,
    "feedback": decode_feedback,
    "trellis": decode_trellis,
    "basis": decode_basis,
    "channel_scaled": decode_channel_scaled,
    "vector_scaled": decode_vector_scaled,
    "vector_feedback": decode_vector_scaled,
}


def decoder(name: str) -> Callable[..., Tensor]:
    if name not in _DECODERS:
        raise ValueError(f"checkpoint requires registered codec {name!r}")
    return _DECODERS[name]


def codec(name: str) -> Callable[[Callable], Callable]:
    """Register a decoder by name; checkpoints store its name rather than its code."""
    def register(function: Callable[..., Tensor]) -> Callable:
        if not name or name in _DECODERS:
            raise ValueError(f"codec name {name!r} is empty or already registered")
        _DECODERS[name] = function
        return function

    return register


def registered(name: str | None) -> bool:
    return name in _DECODERS


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
    return Q(
        codes=quantized.codes,
        params={f"inner_{key}": value for key, value in quantized.params.items()} | {
            "inner_codec": quantized.codec,
        } | parameters,
        decode=decoder(name), codec=name, bits=quantized.bits, metadata=metadata,
    )
