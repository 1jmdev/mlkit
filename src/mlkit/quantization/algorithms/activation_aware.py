"""Activation-aware channel scale search, optionally shared across sibling projections."""

import torch
from torch import Tensor

from mlkit.quantization.codecs import compose, registered
from mlkit.quantization.context import Ctx, copy_cache
from mlkit.quantization.operations.losses import proxy_loss
from mlkit.quantization.protocol import Quantizer, QuantizerFunction
from mlkit.quantization.representation import Q, as_q


class ActivationAware(Quantizer):
    def __init__(self, inner: QuantizerFunction, *, grid: int = 20, shared: bool = False) -> None:
        if grid < 1:
            raise ValueError("AWQ search grid must be positive")
        self.inner = inner
        self.grid = grid
        self.shared = shared

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        ctx = ctx or Ctx(device=w.device)
        siblings = ctx.siblings if self.shared and ctx.siblings else (ctx.name,)
        group_key = (repr(self), siblings)
        results = ctx.cache.setdefault("_mlkit_awq_results", {})
        if self.shared and group_key in results:
            result, additional_bits = results[group_key][ctx.name]
            ctx.add_bits(additional_bits)
            return result
        modules = ctx.cache.get("_mlkit_sibling_modules", {})
        algorithms = ctx.cache.get("_mlkit_layer_algorithms", {})
        if len(siblings) > 1 and any(repr(algorithms.get(name)) != repr(self) for name in siblings):
            raise ValueError("shared AWQ requires the same quantizer on every sibling projection")
        weights = {
            name: w if name == ctx.name else modules[name].weight.detach().float()
            for name in siblings
        }
        importance = ctx.act_absmean.to(w.device).clamp_min(1e-5)
        hessian = ctx.H.to(w.device)
        selected = None
        minimum_loss = float("inf")
        for alpha in torch.linspace(0, 1, self.grid).tolist():
            scales = importance.pow(alpha)
            scales /= (scales.max() * scales.min()).sqrt()
            scales = scales.clamp(2**-24, 65504).half().float()
            transformed_hessian = hessian / scales[:, None] / scales[None, :]
            trial_cache = copy_cache(ctx.cache)
            candidates = {}
            loss = 0.0
            for index, (name, weight) in enumerate(weights.items()):
                scaled_context = ctx.replace(H=transformed_hessian, cache=trial_cache, name=name)
                candidate = as_q(self.inner(weight * scales, scaled_context))
                reconstruction = candidate.w / scales
                if candidate.codes is not None and registered(candidate.codec):
                    candidate = compose(candidate, "channel_scaled", {"channel_scales": scales})
                    candidate.metadata["parameter_formats"]["channel_scales"] = "fp16"
                else:
                    candidate = Q(reconstruction, bits=candidate.bits)
                scale_bits = 16 * scales.numel() if index == 0 else 0
                candidates[name] = (candidate, scaled_context._additional_bits + scale_bits)
                loss += float(proxy_loss(weight, reconstruction, ctx))
            if loss < minimum_loss:
                selected = candidates
                minimum_loss = loss
                selected_cache = trial_cache
        assert selected is not None
        ctx.cache.update(selected_cache)
        if self.shared:
            results[group_key] = selected
        result, additional_bits = selected[ctx.name]
        ctx.add_bits(additional_bits)
        return result

    def __repr__(self) -> str:
        return f"awq({self.inner!r}, grid={self.grid}, shared={self.shared})"


def awq(inner: QuantizerFunction, grid: int = 20, shared: bool = False) -> ActivationAware:
    return ActivationAware(inner, grid=grid, shared=shared)
