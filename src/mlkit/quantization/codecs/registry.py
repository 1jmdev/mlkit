"""The registry of named decoders; checkpoints store a decoder name rather than its code."""

from collections.abc import Callable

from torch import Tensor

_DECODERS: dict[str, Callable[..., Tensor]] = {}
_ROW_PARAMETERS: dict[str, tuple[str, ...]] = {}


def decoder(name: str) -> Callable[..., Tensor]:
    if name not in _DECODERS:
        raise ValueError(f"checkpoint requires registered codec {name!r}")
    return _DECODERS[name]


def codec(
    name: str,
    *,
    row_parameters: tuple[str, ...] | None = None,
) -> Callable[[Callable], Callable]:
    """Register a decoder by name; checkpoints store its name rather than its code.

    ``row_parameters`` names the decoder parameters that hold one entry per
    weight row, alongside codes that hold one row per weight row. Declaring them
    lets representations of row chunks be joined, which is how wide layers are
    converted within bounded memory.
    """
    def register(function: Callable[..., Tensor]) -> Callable:
        if not name or name in _DECODERS:
            raise ValueError(f"codec name {name!r} is empty or already registered")
        _DECODERS[name] = function
        if row_parameters is not None:
            _ROW_PARAMETERS[name] = row_parameters
        return function

    return register


def registered(name: str | None) -> bool:
    return name in _DECODERS


def row_parameters(name: str | None) -> tuple[str, ...] | None:
    """The per-row parameters of a codec, or ``None`` if its rows cannot be joined."""
    return None if name is None else _ROW_PARAMETERS.get(name)
