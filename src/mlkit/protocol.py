"""Function-first quantizer and pass configuration protocols."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from torch import Tensor

from mlkit.context import Ctx
from mlkit.representation import Q, as_q


class Quantizer:
    """Optional fitting protocol for algorithms that round narrow column slices."""

    def fit(self, w: Tensor, ctx: Ctx) -> Callable[[Tensor, slice], Q]:
        raise NotImplementedError

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        return as_q(self.fit(w, ctx or Ctx(device=w.device))(w, slice(None)))


class FunctionQuantizer(Quantizer):
    __name__: str

    def __init__(self, function: Callable[..., Q | Tensor], parameters: dict[str, Any]) -> None:
        functools.update_wrapper(self, function)
        self.function = function
        self.parameters = parameters
        self.signature = inspect.signature(function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args and not {"w", "ctx"}.intersection(kwargs):
            parameters = self.parameters | kwargs
            self.signature.bind_partial(None, None, **parameters)
            return FunctionQuantizer(self.function, parameters)
        return as_q(self.function(*args, **(self.parameters | kwargs)))

    def fit(self, w: Tensor, ctx: Ctx) -> Callable[[Tensor, slice], Q]:
        def round_columns(value: Tensor, columns: slice) -> Q:
            try:
                return self(value, ctx)
            except (ValueError, RuntimeError) as error:
                raise ValueError(
                    f"{self.__name__} cannot round a narrow slice {tuple(value.shape)}; "
                    "use mk.scaled or implement Quantizer.fit(w, ctx) with round(x, cols)"
                ) from error

        return round_columns

    def __repr__(self) -> str:
        arguments = ", ".join(f"{name}={value!r}" for name, value in self.parameters.items())
        return f"{self.__name__}({arguments})"


def quantizer(function: Callable[..., Q | Tensor]) -> FunctionQuantizer:
    return FunctionQuantizer(function, {})


def fit_quantizer(
    quantization: Callable[..., Q | Tensor], w: Tensor, ctx: Ctx
) -> Callable[[Tensor, slice], Q]:
    if isinstance(quantization, Quantizer):
        return quantization.fit(w, ctx)
    return FunctionQuantizer(quantization, {}).fit(w, ctx)
