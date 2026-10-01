"""The registry of named decoders; checkpoints store a decoder name rather than its code."""

from collections.abc import Callable

from torch import Tensor

_DECODERS: dict[str, Callable[..., Tensor]] = {}


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
