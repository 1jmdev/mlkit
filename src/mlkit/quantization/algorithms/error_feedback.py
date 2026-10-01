"""Hessian error feedback: GPTQ single-column and LDLQ block rounding."""

import math
from typing import Any

import torch
from torch import Tensor

from mlkit.kernels.error_feedback import round_tile
from mlkit.quantization.codecs import decode_feedback, decode_trellis, decode_vector_scaled
from mlkit.quantization.context import Ctx
from mlkit.quantization.protocol import (
    FittedRounder,
    Quantizer,
    QuantizerFunction,
    fit_quantizer,
)
from mlkit.quantization.representation import Q, as_q


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
            raise ValueError(
                f"calibration Hessian for {ctx.name!r} is not positive definite; increase damp"
            )
        if permutation is None:
            del diagonal, hessian
        if factor.requires_grad:
            inverse = torch.cholesky_inverse(factor)
            upper = torch.linalg.cholesky(inverse, upper=True).contiguous()
        else:
            inverse = torch.cholesky_inverse(factor, out=factor)
            torch.linalg.cholesky(inverse, upper=True, out=inverse)
            upper = inverse.contiguous()
        del factor
        del inverse
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
            scalar_grid = rounder.scalar_grid if isinstance(rounder, FittedRounder) else None
            use_fused = (
                working.device.type == "cuda" and self.backend != "torch" and self.step == 1
                and scalar_grid is not None and scalar_grid.bits <= 8
                and scalar_grid.values is not None and 1 <= scalar_grid.values.numel() <= 256
            )
            if self.backend == "triton" and not use_fused:
                raise ValueError("fused GPTQ requires CUDA, step=1, and a native scalar rounder")
            if use_fused:
                assert scalar_grid is not None
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
                        bits=scalar_grid.bits, values=initial.params["values"],
                        integer_grid=scalar_grid.integer,
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
                        if (quantized.codec not in {"scaled", "vector_scaled"}
                                or quantized.codes is None):
                            retain_codec = False
                        else:
                            if encoded is None:
                                dimension = quantized.params.get("dim", 1)
                                encoded = torch.empty(
                                    (working.shape[0], width // dimension),
                                    device=working.device, dtype=quantized.codes.dtype,
                                )
                                codec_parameters = quantized.params
                                codec_metadata = quantized.metadata
                            dimension = quantized.params.get("dim", 1)
                            encoded[:, column // dimension : stop // dimension] = quantized.codes
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
            dimension = codec_parameters.get("dim")
            parameters = {
                "scales": torch.cat(region_scales, dim=1),
                "values": codec_parameters["values"], "group": codec_parameters["group"],
                "refit": refit_width,
                "zero": torch.cat(region_zeros, dim=1) if region_zeros else None,
                "permutation": permutation,
            }
            if dimension is not None:
                parameters["dim"] = dimension
            return Q(
                codes=encoded, bits=total_bits,
                codec="feedback" if dimension is None else "vector_feedback",
                decode=decode_feedback if dimension is None else decode_vector_scaled,
                params=parameters,
                metadata=codec_metadata,
            )
        if trellis_parts and len(trellis_parts) == width // self.step:
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


def ldlq(inner: QuantizerFunction, **options: Any) -> ErrorFeedback:
    return ErrorFeedback(inner, **options)


def gptq(
    inner: QuantizerFunction,
    refit: int | None = 128,
    act_order: bool = False,
    **options: Any,
) -> ErrorFeedback:
    return ldlq(
        inner,
        step=1,
        refit=refit,
        order="activation" if act_order else "natural",
        **options,
    )
