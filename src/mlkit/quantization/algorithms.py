"""Composable calibration algorithms and changes of basis."""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from mlkit.quantization.codecs import compose, deterministic_signs, registered
from mlkit.quantization.context import Ctx, layer_seed
from mlkit.quantization.formats import Scaled, decode_feedback
from mlkit.quantization.operations import proxy_loss
from mlkit.quantization.protocol import Quantizer, fit_quantizer
from mlkit.quantization.representation import Q, as_q
from mlkit.quantization.rotations import structured_transform

QuantizerFunction = Callable[[Tensor, Ctx], Q | Tensor]


class RoundToNearest(Quantizer):
    def __init__(self, inner: QuantizerFunction) -> None:
        self.inner = inner

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        return as_q(self.inner(w, ctx or Ctx(device=w.device)))

    def __repr__(self) -> str:
        return f"rtn({self.inner!r})"


class ErrorFeedback(Quantizer):
    """Inverse-Cholesky error feedback with tiled trailing matrix updates."""

    def __init__(
        self,
        inner: QuantizerFunction,
        *,
        step: int = 8,
        refit: int | None = None,
        damp: float = 0.01,
        order: str = "natural",
        block_size: int = 128,
        backend: str = "auto",
    ) -> None:
        if step < 1 or block_size < step or block_size % step:
            raise ValueError("block_size must be a positive multiple of step")
        if refit is not None and (refit < step or refit % step):
            raise ValueError("refit must be a positive multiple of step or None")
        if damp < 0 or order not in {"natural", "activation"}:
            raise ValueError("damp must be nonnegative and order must be natural or activation")
        self.inner = inner
        self.step = step
        self.refit = refit
        self.damp = damp
        self.order = order
        self.block_size = block_size
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("error-feedback backend must be auto, torch, or triton")
        self.backend = backend

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        ctx = ctx or Ctx(device=w.device)
        working = w.float().clone()
        hessian = ctx.H.to(device=w.device, dtype=torch.float32).clone()
        width = w.shape[1]
        if hessian.shape != (width, width):
            raise ValueError("Hessian dimensions must match the weight input width")
        if width % self.step:
            raise ValueError("weight input width must be divisible by the error-feedback step")
        dead = hessian.diagonal() == 0
        hessian[dead, dead] = 1
        working[:, dead] = 0
        permutation = None
        if self.order == "activation":
            permutation = hessian.diagonal().argsort(descending=True)
            working = working[:, permutation]
            hessian = hessian[permutation][:, permutation]
        diagonal = hessian.diagonal()
        diagonal.add_(self.damp * diagonal.mean().clamp_min(1e-12))
        factor, status = torch.linalg.cholesky_ex(hessian)
        if status.item() != 0:
            raise ValueError("calibration Hessian is not positive definite; increase damp")
        inverse = torch.cholesky_inverse(factor)
        upper = torch.linalg.cholesky(inverse, upper=True).contiguous()
        output = torch.empty_like(working)
        total_bits: float | None = 0.0
        encoded = None
        codec_parameters = None
        codec_metadata = None
        region_scales = []
        region_zeros = []
        retain_codec = True
        trellis_parts = []
        refit_width = width if self.refit is None else self.refit
        for region_start in range(0, width, refit_width):
            region_stop = min(width, region_start + refit_width)
            fitted_context = ctx
            if permutation is not None:
                fitted_context = ctx.replace(H=hessian)
            rounder = fit_quantizer(
                self.inner, working[:, region_start:region_stop], fitted_context
            )
            if fitted_context is not ctx:
                ctx.add_bits(fitted_context._additional_bits)
            use_fused = (
                working.device.type == "cuda" and self.backend != "torch" and self.step == 1
                and isinstance(self.inner, Scaled) and self.inner.grid.name.startswith("int")
            )
            if self.backend == "triton" and not use_fused:
                raise ValueError("fused GPTQ requires CUDA, step=1, and a scaled integer grid")
            if use_fused:
                assert isinstance(self.inner, Scaled)
                from mlkit.runtime.kernels.error_feedback import round_tile

                initial = rounder(working[:, region_start:region_stop], slice(None))
                assert initial.codes is not None
                if encoded is None:
                    encoded = torch.empty_like(working, dtype=initial.codes.dtype)
                    codec_parameters, codec_metadata = initial.params, initial.metadata
                region_scales.append(initial.params["scales"])
                if initial.params["zero"] is not None:
                    region_zeros.append(initial.params["zero"])
                total_bits = None if total_bits is None or initial.bits is None else (
                    total_bits + initial.bits
                )
                for tile_start in range(region_start, region_stop, self.block_size):
                    tile_stop = min(region_stop, tile_start + self.block_size)
                    errors = working.new_empty((working.shape[0], tile_stop - tile_start))
                    round_tile(
                        working, upper, initial.params["scales"], initial.params["zero"],
                        output, encoded, errors, region_start=region_start,
                        tile_start=tile_start, group=initial.params["group"],
                        bits=self.inner.grid.bits,
                    )
                    working[:, tile_stop:] -= errors @ upper[tile_start:tile_stop, tile_stop:]
                continue
            for tile_start in range(region_start, region_stop, self.block_size):
                tile_stop = min(region_stop, tile_start + self.block_size)
                tile = working[:, tile_start:tile_stop].clone()
                errors = torch.zeros_like(tile)
                for column in range(tile_start, tile_stop, self.step):
                    local = column - tile_start
                    stop = column + self.step
                    local_stop = local + self.step
                    quantized = as_q(rounder(
                        tile[:, local:local_stop],
                        slice(column - region_start, stop - region_start),
                    ))
                    reconstruction = quantized.w
                    if quantized.codec == "trellis" and self.refit is None and permutation is None:
                        trellis_parts.append(quantized)
                    if retain_codec:
                        if quantized.codec != "scaled" or quantized.codes is None:
                            retain_codec = False
                        else:
                            if encoded is None:
                                encoded = torch.empty_like(working, dtype=quantized.codes.dtype)
                                codec_parameters = quantized.params
                                codec_metadata = quantized.metadata
                            encoded[:, column:stop] = quantized.codes
                            if column == region_start:
                                region_scales.append(quantized.params["scales"])
                                if quantized.params["zero"] is not None:
                                    region_zeros.append(quantized.params["zero"])
                    if reconstruction.shape != tile[:, local:local_stop].shape:
                        raise ValueError("inner quantizer changed the shape of a column slice")
                    output[:, column:stop] = reconstruction
                    if total_bits is not None:
                        total_bits = None if quantized.bits is None else total_bits + quantized.bits
                    residual = tile[:, local:local_stop] - reconstruction
                    if self.step == 1:
                        error = residual / upper[column, column]
                    else:
                        error = torch.linalg.solve_triangular(
                            upper[column:stop, column:stop].T,
                            residual.T,
                            upper=False,
                        ).T
                    errors[:, local:local_stop] = error
                    tile[:, local_stop:] -= error @ upper[column:stop, stop:tile_stop]
                working[:, tile_stop:] -= errors @ upper[tile_start:tile_stop, tile_stop:]
        if permutation is not None:
            output = output[:, permutation.argsort()]
        if retain_codec and encoded is not None:
            assert codec_parameters is not None and codec_metadata is not None
            if permutation is not None:
                ctx.add_bits(width * math.ceil(math.log2(width)))
            return Q(
                codes=encoded, bits=total_bits, codec="feedback", decode=decode_feedback,
                params={
                    "scales": torch.cat(region_scales, dim=1),
                    "values": codec_parameters["values"], "group": codec_parameters["group"],
                    "refit": refit_width,
                    "zero": torch.cat(region_zeros, dim=1) if region_zeros else None,
                    "permutation": permutation,
                },
                metadata=codec_metadata,
            )
        if trellis_parts and len(trellis_parts) == width // self.step:
            from mlkit.quantization.trellis import decode_trellis

            parameters = trellis_parts[0].params | {
                "shape": tuple(w.shape),
                "initial_states": torch.cat([
                    part.params["initial_states"] for part in trellis_parts
                ]),
            }
            return Q(
                codes=torch.cat([part.codes for part in trellis_parts if part.codes is not None]),
                params=parameters,
                decode=decode_trellis, codec="trellis", bits=total_bits,
                metadata=trellis_parts[0].metadata,
            )
        return Q(output, bits=total_bits)

    def __repr__(self) -> str:
        return f"ldlq({self.inner!r}, step={self.step}, refit={self.refit})"


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
        weights = {name: w if name == ctx.name else modules[name].weight.detach().float()
                   for name in siblings}
        importance = ctx.act_absmean.to(w.device).clamp_min(1e-5)
        hessian = ctx.H.to(w.device)
        selected = None
        minimum_loss = float("inf")
        for alpha in torch.linspace(0, 1, self.grid).tolist():
            scales = importance.pow(alpha)
            scales /= (scales.max() * scales.min()).sqrt()
            scales = scales.clamp(2**-24, 65504).half().float()
            transformed_hessian = hessian / scales[:, None] / scales[None, :]
            trial_cache = dict(ctx.cache)
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
                candidates[name] = (
                    candidate, scaled_context._additional_bits + (16 * scales.numel() if index == 0
                                                                 else 0),
                )
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


class Incoherent(Quantizer):
    def __init__(
        self,
        inner: QuantizerFunction,
        *,
        left: str | None = "rht",
        right: str | None = "rht",
        train_signs: bool = False,
    ) -> None:
        if left not in {None, "rht"} or right not in {None, "rht"}:
            raise ValueError("incoherence transforms must be rht or None")
        self.inner = inner
        self.left = left
        self.right = right
        self.train_signs = train_signs

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        ctx = ctx or Ctx(device=w.device)
        seed = layer_seed(ctx.name, ctx._seed) % 2**32
        left_seed = seed if self.left else None
        right_seed = (seed + 1) % 2**32 if self.right else None
        left_signs = deterministic_signs(w.shape[0], seed, w.device) if self.left else None
        right_signs = deterministic_signs(w.shape[1], seed + 1, w.device) if self.right else None
        transformed = w.float()
        if left_signs is not None:
            transformed = structured_transform((transformed * left_signs[:, None]).T).T
        if right_signs is not None:
            transformed = structured_transform(transformed * right_signs)
        transformed_context = ctx.replace()
        if right_signs is not None:
            original_provider = transformed_context._provider

            def provider(name: str, fn: Callable | None, reduce: str) -> Tensor:
                if name == "H":
                    hessian = ctx.H.to(w.device) * right_signs[:, None] * right_signs[None, :]
                    return structured_transform(structured_transform(hessian).T).T
                if name == "X":
                    return structured_transform(ctx.X.to(w.device) * right_signs)
                inputs = structured_transform(ctx.X.to(w.device) * right_signs)
                if name == "act_absmean":
                    return inputs.abs().mean(0)
                if name == "act_absmax":
                    return inputs.abs().amax(0)
                if fn is not None:
                    return fn(inputs)
                if original_provider is not None:
                    return original_provider(name, fn, reduce)
                raise KeyError(name)

            transformed_context._stats = {}
            transformed_context._provider = provider
        quantized = as_q(self.inner(transformed, transformed_context))
        ctx.add_bits(transformed_context._additional_bits)
        if quantized.codes is not None and registered(quantized.codec):
            result = compose(quantized, "basis", {
                "shape": tuple(w.shape), "left_seed": left_seed, "right_seed": right_seed,
                "left_signs": left_signs if self.train_signs else None,
                "right_signs": right_signs if self.train_signs else None,
            })
            if self.train_signs:
                for name, signs in [("left_signs", left_signs), ("right_signs", right_signs)]:
                    if signs is not None:
                        result.metadata["trainable"].append(name)
                        result.metadata["parameter_formats"][name] = "fp16"
                        ctx.add_bits(16 * signs.numel())
            return result
        if self.train_signs:
            raise ValueError("trainable signs require an inner quantizer with a registered codec")
        reconstruction = quantized.w
        if right_signs is not None:
            reconstruction = structured_transform(reconstruction, inverse=True) * right_signs
        if left_signs is not None:
            reconstruction = (
                structured_transform(reconstruction.T, inverse=True).T * left_signs[:, None]
            )
        return Q(reconstruction, bits=quantized.bits)

    def __repr__(self) -> str:
        return f"incoherent({self.inner!r})"


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
            candidate_context = ctx.replace(cache=dict(ctx.cache))
            candidate = as_q(quantization(w, candidate_context))
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


def rtn(inner: QuantizerFunction) -> RoundToNearest:
    return RoundToNearest(inner)


def ldlq(inner: QuantizerFunction, **options: Any) -> ErrorFeedback:
    return ErrorFeedback(inner, **options)


def gptq(
    inner: QuantizerFunction,
    refit: int | None = 128,
    act_order: bool = False,
    **options: Any,
) -> ErrorFeedback:
    return ldlq(
        inner, step=1, refit=refit,
        order="activation" if act_order else "natural", **options,
    )


def awq(inner: QuantizerFunction, grid: int = 20, shared: bool = False) -> ActivationAware:
    return ActivationAware(inner, grid=grid, shared=shared)


def incoherent(inner: QuantizerFunction, **options: Any) -> Incoherent:
    return Incoherent(inner, **options)


def best_of(*candidates: QuantizerFunction, by: str = "proxy") -> BestOf:
    return BestOf(candidates, by=by)
