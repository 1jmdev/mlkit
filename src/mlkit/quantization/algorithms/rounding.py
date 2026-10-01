"""Direct rounding without calibration statistics."""

from torch import Tensor

from mlkit.quantization.context import Ctx
from mlkit.quantization.protocol import Quantizer, QuantizerFunction
from mlkit.quantization.representation import Q, as_q


class RoundToNearest(Quantizer):
    def __init__(self, inner: QuantizerFunction) -> None:
        self.inner = inner

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        return as_q(self.inner(w, ctx or Ctx(device=w.device)))

    def __repr__(self) -> str:
        return f"rtn({self.inner!r})"


def rtn(inner: QuantizerFunction) -> RoundToNearest:
    return RoundToNearest(inner)
