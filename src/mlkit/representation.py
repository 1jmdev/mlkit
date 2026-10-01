"""Quantized reconstructions and optional differentiable codec state."""

from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor


class Q:
    """A reconstruction, its logical bit cost, and an optional executable codec.

    ``bits`` is the total number of bits, including side information. ``None``
    means unknown. Decoder functions are never pickled into checkpoints.
    """

    def __init__(
        self,
        w: Tensor | None = None,
        *,
        bits: int | float | None = None,
        codes: Tensor | None = None,
        params: Mapping[str, Any] | None = None,
        decode: Callable[..., Tensor] | None = None,
        codec: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if w is None and (codes is None or decode is None):
            raise ValueError("Q requires a reconstruction or codes and a decoder")
        if bits is not None and bits < 0:
            raise ValueError("bits must be nonnegative")
        self._weight = w
        self.bits = bits
        self.codes = codes
        self.params = dict(params or {})
        self.decode = decode
        self.codec = codec
        self.metadata = dict(metadata or {})

    @property
    def w(self) -> Tensor:
        """Decode on access so trainable codec parameters remain differentiable."""
        if self.decode is not None and self.codes is not None:
            return self.decode(self.codes, **self.params)
        assert self._weight is not None
        return self._weight

    @property
    def bpw(self) -> float | None:
        return None if self.bits is None else float(self.bits) / self.w.numel()

    def __repr__(self) -> str:
        return f"Q(shape={tuple(self.w.shape)}, bits={self.bits}, codec={self.codec!r})"


def as_q(value: Q | Tensor) -> Q:
    if isinstance(value, Q):
        return value
    if isinstance(value, Tensor):
        return Q(value)
    raise TypeError(f"quantizers must return a Tensor or Q, received {type(value).__name__}")
