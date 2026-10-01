"""Named decoders for portable quantized representations.

Importing this package registers every built-in codec.
"""

from mlkit.quantization.codecs.composition import (
    compose,
    decode_basis,
    decode_channel_scaled,
    deterministic_signs,
)
from mlkit.quantization.codecs.registry import codec, decoder, registered
from mlkit.quantization.codecs.scaled import decode_feedback, decode_scaled, decode_vector_scaled
from mlkit.quantization.codecs.trellis import decode_trellis

__all__ = [
    "codec",
    "compose",
    "decode_basis",
    "decode_channel_scaled",
    "decode_feedback",
    "decode_scaled",
    "decode_trellis",
    "decode_vector_scaled",
    "decoder",
    "deterministic_signs",
    "registered",
]
