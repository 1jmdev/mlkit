"""Hessian error feedback: GPTQ single-column and LDLQ block rounding.

Rounding is computed in LDL form. With the damped Hessian factored as
``H = R Rᵀ`` for an upper-triangular ``R``, the proxy loss of a row deviation
``d = w - q`` is ``‖d R‖²``. Column block ``k`` is therefore rounded after adding
``Σ_{j<k} d_j R_jk R_kk⁻¹`` to its weights. One Cholesky factorization of the
index-reversed Hessian supplies every coefficient and no inverse is formed. The
recursion is algebraically identical to the inverse-Cholesky updates of GPTQ.
"""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from mlkit.kernels.error_feedback import round_tile, round_vector_tile
from mlkit.quantization.codecs import decode_feedback, decode_trellis, decode_vector_scaled
from mlkit.quantization.context import Ctx
from mlkit.quantization.protocol import (
    FittedRounder,
    Quantizer,
    QuantizerFunction,
    ScalarRounding,
    VectorRounding,
    fit_quantizer,
)
from mlkit.quantization.representation import Q, as_q

FUSED_CODEBOOK_LIMIT = 256
FUSED_VECTOR_DIMENSIONS = {2, 4, 8, 16}
FUSED_VECTOR_CODEBOOK_LIMIT = 65536
UPPER_FACTORIZATION_WIDTH = 4096


def feedback_coefficients(
    hessian: Tensor,
    step: int,
    *,
    damp: float = 0.01,
    permutation: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Coefficients ``R blockdiag(R)⁻¹`` of the damped ``H = R Rᵀ`` and the factorization status.

    Columns are processed in the order given by ``permutation``. Zero diagonal
    entries mark channels without activations and are replaced by one before
    damping. The lower Cholesky factor of the index-reversed Hessian, reversed
    again, is the upper factor ``R``. The result is row-major and contiguous; only
    entries above the block diagonal are meaningful. A nonzero status reports an
    indefinite matrix.

    cuSOLVER factors column-major matrices. From ``UPPER_FACTORIZATION_WIDTH``
    upward the upper factorization is faster, and its transpose is already the
    row-major lower factor.
    """
    width = hessian.shape[0]
    if permutation is None:
        reversed_hessian = hessian.flip(0, 1)
    else:
        order = permutation.flip(0)
        reversed_hessian = hessian[order][:, order]
    diagonal = reversed_hessian.diagonal()
    diagonal.masked_fill_(diagonal == 0, 1)
    diagonal.add_(damp * diagonal.mean().clamp_min(1e-12))
    if width >= UPPER_FACTORIZATION_WIDTH:
        factor, status = torch.linalg.cholesky_ex(reversed_hessian, upper=True)
        lower = factor.T
    else:
        lower, status = torch.linalg.cholesky_ex(reversed_hessian)
    del reversed_hessian, diagonal
    if status.item() != 0:
        return lower, int(status.item())
    if step == 1:
        lower.div_(lower.diagonal().clone())
        return lower.flip(0, 1).contiguous(), 0
    upper = lower.flip(0, 1).contiguous()
    blocks = width // step
    selection = torch.arange(blocks, device=upper.device)
    diagonal_blocks = upper.reshape(blocks, step, blocks, step)[selection, :, selection, :]
    block_columns = upper.reshape(width, blocks, step).transpose(0, 1)
    solved = torch.linalg.solve_triangular(diagonal_blocks, block_columns, upper=True, left=False)
    return solved.transpose(0, 1).reshape(width, width).contiguous(), 0


class PermutedStatistics:
    """Supplies calibration statistics in the column order of permuted weights."""

    def __init__(self, ctx: Ctx, permutation: Tensor) -> None:
        self.ctx = ctx
        self.permutation = permutation
        self.values: dict[str, Tensor] = {}

    def __call__(self, name: str, function: Callable | None, reduction: str) -> Tensor:
        if name not in self.values:
            value = self.ctx.stat(name, function, reduction)
            if name == "H":
                value = value[self.permutation][:, self.permutation]
            elif name == "X":
                value = value[:, self.permutation]
            elif name in {"act_absmean", "act_absmax"}:
                value = value[self.permutation]
            self.values[name] = value
        return self.values[name]


class CodecAccumulator:
    """Collects codes and fitted parameters so that the result remains a portable codec."""

    def __init__(self, rows: int, width: int, device: torch.device) -> None:
        self.rows = rows
        self.width = width
        self.device = device
        self.codes: Tensor | None = None
        self.parameters: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.scales: list[Tensor] = []
        self.zeros: list[Tensor] = []
        self.retained = True
        self.trellis_parts: list[Q] = []
        self.bits: float | None = 0.0

    def add_bits(self, bits: float | None) -> None:
        self.bits = None if self.bits is None or bits is None else self.bits + bits

    def begin_scalar_region(self, scalar: ScalarRounding) -> Tensor:
        """Record a region rounded by the fused kernel and return its code matrix."""
        if self.codes is None:
            self.codes = torch.empty((self.rows, self.width), dtype=torch.uint8, device=self.device)
            self.parameters = {"values": scalar.values, "group": scalar.group}
            self.metadata = scalar.metadata
        self.scales.append(scalar.scales)
        if scalar.zero is not None:
            self.zeros.append(scalar.zero)
        self.add_bits(scalar.bits)
        return self.codes

    def begin_vector_region(self, vector: VectorRounding) -> Tensor:
        """Record a region rounded in blocks by the fused kernel and return its code matrix."""
        dimension = vector.grid.dim
        if self.codes is None:
            self.codes = torch.empty(
                (self.rows, self.width // dimension), dtype=torch.int32, device=self.device
            )
            self.parameters = {
                "values": None if vector.lattice else vector.codebook,
                "group": vector.group,
                "dim": dimension,
            }
            self.metadata = vector.metadata
        self.scales.append(vector.scales)
        self.add_bits(vector.bits)
        return self.codes

    def add_block(
        self,
        quantized: Q,
        column: int,
        stop: int,
        region_start: int,
        collect_trellis: bool,
    ) -> None:
        """Record one block rounded by the reference path."""
        self.add_bits(quantized.bits)
        if collect_trellis and quantized.codec == "trellis":
            self.trellis_parts.append(quantized)
        if not self.retained:
            return
        if quantized.codec not in {"scaled", "vector_scaled"} or quantized.codes is None:
            self.retained = False
            return
        dimension = quantized.params.get("dim", 1)
        if self.codes is None:
            self.codes = torch.empty(
                (self.rows, self.width // dimension),
                dtype=quantized.codes.dtype,
                device=self.device,
            )
            self.parameters = quantized.params
            self.metadata = quantized.metadata
        self.codes[:, column // dimension : stop // dimension] = quantized.codes
        if column == region_start:
            self.scales.append(quantized.params["scales"])
            if quantized.params["zero"] is not None:
                self.zeros.append(quantized.params["zero"])

    def result(
        self,
        output: Tensor,
        permutation: Tensor | None,
        refit_width: int,
        blocks: int,
        ctx: Ctx,
    ) -> Q:
        if self.retained and self.codes is not None:
            if permutation is not None:
                ctx.add_bits(self.width * math.ceil(math.log2(self.width)))
            dimension = self.parameters.get("dim")
            parameters = {
                "scales": torch.cat(self.scales, dim=1),
                "values": self.parameters["values"],
                "group": self.parameters["group"],
                "refit": refit_width,
                "zero": torch.cat(self.zeros, dim=1) if self.zeros else None,
                "permutation": permutation,
            }
            if dimension is not None:
                parameters["dim"] = dimension
            return Q(
                codes=self.codes,
                bits=self.bits,
                codec="feedback" if dimension is None else "vector_feedback",
                decode=decode_feedback if dimension is None else decode_vector_scaled,
                params=parameters,
                metadata=self.metadata,
            )
        if self.trellis_parts and len(self.trellis_parts) == blocks:
            parts = self.trellis_parts
            parameters = parts[0].params | {
                "shape": (self.rows, self.width),
                "initial_states": torch.cat([part.params["initial_states"] for part in parts]),
            }
            return Q(
                codes=torch.cat([part.codes for part in parts if part.codes is not None]),
                params=parameters,
                decode=decode_trellis,
                codec="trellis",
                bits=self.bits,
                metadata=parts[0].metadata,
            )
        if permutation is not None:
            output = output[:, permutation.argsort()]
        return Q(output, bits=self.bits)


class ErrorFeedback(Quantizer):
    """LDL error feedback with fused tile rounding and lazy trailing matrix updates."""

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
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("error-feedback backend must be auto, torch, or triton")
        self.inner = inner
        self.step = step
        self.refit = refit
        self.damp = damp
        self.order = order
        self.block_size = block_size
        self.backend = backend

    def __call__(self, w: Tensor, ctx: Ctx | None = None) -> Q:
        ctx = ctx or Ctx(device=w.device)
        rows, width = w.shape
        hessian = ctx.H.to(device=w.device, dtype=torch.float32).detach()
        if hessian.shape != (width, width):
            raise ValueError("Hessian dimensions must match the weight input width")
        if width % self.step:
            raise ValueError("weight input width must be divisible by the error-feedback step")
        dead = hessian.diagonal() == 0
        original = w.detach().to(torch.float32, copy=True)
        original[:, dead] = 0
        permutation = None
        if self.order == "activation":
            permutation = hessian.diagonal().masked_fill(dead, 1).argsort(descending=True)
            original = original[:, permutation]
        coefficients, status = feedback_coefficients(
            hessian, self.step, damp=self.damp, permutation=permutation
        )
        if status != 0:
            raise ValueError(
                f"calibration Hessian for {ctx.name!r} is not positive definite; increase damp"
            )
        del hessian
        statistics = None if permutation is None else PermutedStatistics(ctx, permutation)
        feedback = torch.zeros_like(original)
        output = torch.empty_like(original)
        accumulator = CodecAccumulator(rows, width, w.device)
        refit_width = width if self.refit is None else self.refit
        collect_trellis = self.refit is None and permutation is None
        for region_start in range(0, width, refit_width):
            region_stop = min(width, region_start + refit_width)
            targets = original[:, region_start:region_stop]
            if region_start:
                targets = targets + self._pending_continuation(
                    feedback, coefficients, region_start, region_stop
                )
            fitted_context = ctx if statistics is None else ctx.derive(statistics)
            rounder = fit_quantizer(self.inner, targets, fitted_context)
            del targets
            if fitted_context is not ctx:
                ctx.add_bits(fitted_context.additional_bits)
            fused = self._fused_rounding(rounder, w.device)
            scalar = fused if isinstance(fused, ScalarRounding) else None
            vector = fused if isinstance(fused, VectorRounding) else None
            encoded = None
            if scalar is not None:
                encoded = accumulator.begin_scalar_region(scalar)
            elif vector is not None:
                encoded = accumulator.begin_vector_region(vector)
            for tile_start in range(region_start, region_stop, self.block_size):
                tile_stop = min(region_stop, tile_start + self.block_size)
                if vector is not None:
                    assert encoded is not None
                    round_vector_tile(
                        original,
                        feedback,
                        coefficients,
                        vector.scales,
                        vector.codebook,
                        output,
                        encoded,
                        region_start=region_start,
                        tile_start=tile_start,
                        tile_width=tile_stop - tile_start,
                        group=vector.group,
                        dimension=vector.grid.dim,
                        codebook_size=vector.codebook_size,
                        lattice=vector.lattice,
                    )
                elif scalar is not None:
                    assert encoded is not None
                    round_tile(
                        original,
                        feedback,
                        coefficients,
                        scalar.scales,
                        scalar.zero,
                        output,
                        encoded,
                        region_start=region_start,
                        tile_start=tile_start,
                        tile_width=tile_stop - tile_start,
                        group=scalar.group,
                        bits=scalar.grid.bits,
                        values=scalar.values,
                        integer_grid=scalar.grid.integer,
                    )
                else:
                    self._round_tile(
                        rounder,
                        original,
                        feedback,
                        coefficients,
                        output,
                        accumulator,
                        region_start,
                        tile_start,
                        tile_stop,
                        collect_trellis,
                    )
                if tile_stop < width:
                    deviations = original[:, tile_start:tile_stop] - output[:, tile_start:tile_stop]
                    feedback[:, tile_stop:].addmm_(
                        deviations, coefficients[tile_start:tile_stop, tile_stop:]
                    )
        return accumulator.result(output, permutation, refit_width, width // self.step, ctx)

    @staticmethod
    def _pending_continuation(
        feedback: Tensor,
        coefficients: Tensor,
        region_start: int,
        region_stop: int,
    ) -> Tensor:
        """The adjustment that gives a region its optimal unrounded continuation.

        Formats are fitted to the weights each column would take if no further
        rounding occurred. Accumulated feedback equals that adjustment multiplied
        by the unit-triangular coefficient block of the region, so it is recovered
        with one triangular solve.
        """
        return torch.linalg.solve_triangular(
            coefficients[region_start:region_stop, region_start:region_stop],
            feedback[:, region_start:region_stop],
            upper=True,
            left=False,
            unitriangular=True,
        )

    def _fused_rounding(
        self,
        rounder: Callable[[Tensor, slice], Q],
        device: torch.device,
    ) -> ScalarRounding | VectorRounding | None:
        """The fitted state when a fused CUDA kernel can round this region.

        Scalar grids are rounded one column at a time. Vector grids are rounded in
        blocks, which requires a step equal to the grid dimension.
        """
        fitted = rounder if isinstance(rounder, FittedRounder) else None
        available = fitted is not None and device.type == "cuda" and self.backend != "torch"
        scalar = fitted.scalar if fitted is not None and available else None
        if (
            scalar is not None
            and self.step == 1
            and scalar.grid.bits <= 8
            and 1 <= scalar.values.numel() <= FUSED_CODEBOOK_LIMIT
        ):
            return scalar
        vector = fitted.vector if fitted is not None and available else None
        if (
            vector is not None
            and self.step == vector.grid.dim
            and vector.grid.dim in FUSED_VECTOR_DIMENSIONS
            and vector.codebook_size <= FUSED_VECTOR_CODEBOOK_LIMIT
            and vector.scales.dtype == torch.float32
        ):
            return vector
        if self.backend == "triton":
            raise ValueError(
                "fused error feedback requires CUDA and a native scalar rounder with step=1, "
                "or a native vector rounder with step equal to its dimension"
            )
        return None

    def _round_tile(
        self,
        rounder: Callable[[Tensor, slice], Q],
        original: Tensor,
        feedback: Tensor,
        coefficients: Tensor,
        output: Tensor,
        accumulator: CodecAccumulator,
        region_start: int,
        tile_start: int,
        tile_stop: int,
        collect_trellis: bool,
    ) -> None:
        """Round one tile block by block with an arbitrary fitted rounder."""
        corrections = feedback[:, tile_start:tile_stop].clone()
        for column in range(tile_start, tile_stop, self.step):
            stop = column + self.step
            local_stop = stop - tile_start
            weights = original[:, column:stop]
            targets = weights + corrections[:, column - tile_start : local_stop]
            quantized = as_q(rounder(targets, slice(column - region_start, stop - region_start)))
            reconstruction = quantized.w
            if reconstruction.shape != weights.shape:
                raise ValueError("inner quantizer changed the shape of a column slice")
            output[:, column:stop] = reconstruction
            accumulator.add_block(quantized, column, stop, region_start, collect_trellis)
            if stop < tile_stop:
                deviation = weights - reconstruction
                block_coefficients = coefficients[column:stop, stop:tile_stop]
                if self.step == 1:
                    corrections[:, local_stop:] += deviation * block_coefficients
                else:
                    corrections[:, local_stop:] += deviation @ block_coefficients

    @property
    def row_separable(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.inner, "row_separable", False))

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
