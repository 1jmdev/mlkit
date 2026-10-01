"""Per-layer selection of the candidate quantizer with the lowest reconstruction loss."""

from torch import Tensor

from mlkit.quantization.context import Ctx, copy_cache
from mlkit.quantization.operations.losses import proxy_loss
from mlkit.quantization.protocol import Quantizer, QuantizerFunction
from mlkit.quantization.representation import Q, as_q


class BestOf(Quantizer):
    def __init__(self, candidates: tuple[QuantizerFunction, ...], by: str = "proxy") -> None:
        if not candidates or by not in {"proxy", "mse"}:
            raise ValueError("best_of requires candidates and by='proxy' or 'mse'")
        self.candidates = candidates
        self.by = by

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        ctx = ctx or Ctx(device=w.device)
        selected = None
        minimum_loss = float("inf")
        base_bits, selected_bits = ctx._additional_bits, 0.0
        for quantization in self.candidates:
            candidate_context = ctx.replace(cache=copy_cache(ctx.cache))
            candidate = as_q(quantization(w.clone(), candidate_context))
            loss = float(proxy_loss(w, candidate.w, ctx if self.by == "proxy" else None))
            if loss < minimum_loss:
                selected, minimum_loss = candidate, loss
                selected_bits = candidate_context._additional_bits
                selected_cache = candidate_context.cache
        ctx._additional_bits = base_bits + selected_bits
        ctx.cache.clear()
        ctx.cache.update(selected_cache)
        assert selected is not None
        return selected


def best_of(*candidates: QuantizerFunction, by: str = "proxy") -> BestOf:
    return BestOf(candidates, by=by)
