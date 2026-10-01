"""Grouped scaling and fitting of scalar and vector grids."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from mlkit.kernels.activation_reconstruction import reconstruct
from mlkit.quantization.codecs.scaled import decode_scaled, decode_vector_scaled
from mlkit.quantization.context import Ctx
from mlkit.quantization.formats.scale_storage import SCALE_FORMAT_BITS, store_scale
from mlkit.quantization.grids import Grid
from mlkit.quantization.grids.lattice import nearest_e8p
from mlkit.quantization.operations.search import nearest
from mlkit.quantization.protocol import FittedRounder, Quantizer, ScalarRounding
from mlkit.quantization.representation import Q


def single_precision_reciprocal(value: float) -> float:
    """The FP32 reciprocal, so that host arithmetic and fused kernels agree bit for bit."""
    one = torch.tensor(1.0, dtype=torch.float32)
    return float(one / torch.tensor(value, dtype=torch.float32))


class Scaled(Quantizer):
    def __init__(
        self,
        grid: Grid,
        group: int | None = 128,
        scale: str | Callable[[Tensor], Tensor] = "absmax",
        scale_fmt: str | Grid = "fp16",
        asym: bool = False,
        *,
        search_steps: int = 20,
    ) -> None:
        if group is not None and group <= 0:
            raise ValueError("group must be positive or None")
        if not callable(scale) and scale not in {"absmax", "mse"}:
            raise ValueError("scale must be absmax, mse, or callable")
        if search_steps < 1:
            raise ValueError("search_steps must be positive")
        if asym and (grid.dim != 1 or grid.values is None):
            raise ValueError("asymmetric quantization requires an explicit scalar grid")
        if asym and (scale_fmt == "e8m0" or isinstance(scale_fmt, Grid)):
            raise ValueError("asymmetric offsets require a signed floating-point storage format")
        self.grid = grid
        self.group = group
        self.scale = scale
        self.scale_fmt = scale_fmt
        self.asym = asym
        self.search_steps = search_steps
        self._activation_values: dict[torch.device, Tensor] = {}
        values = None if grid.values is None or grid.dim != 1 else grid.values.detach().cpu()
        self._value_minimum = 0.0 if values is None else float(values.min())
        self._value_maximum = 0.0 if values is None else float(values.max())
        magnitude = 1.0 if grid.values is None else float(grid.values.abs().max())
        value_range = self._value_maximum - self._value_minimum
        if asym and value_range <= 0:
            raise ValueError("asymmetric quantization requires at least two distinct grid values")
        # Scales are a range multiplied by one of these reciprocals, on the host and in kernels.
        self._magnitude_reciprocal = single_precision_reciprocal(magnitude)
        self._range_reciprocal = single_precision_reciprocal(value_range) if asym else 1.0

    def reconstruct_activations(self, value: Tensor) -> Tensor:
        """Reconstruct online scalar grids without codec allocation or host synchronization."""
        supported = (
            value.is_cuda
            and self.grid.dim == 1
            and self.grid.values is not None
            and self.scale == "absmax"
            and isinstance(self.scale_fmt, str)
            and self.scale_fmt in SCALE_FORMAT_BITS
        )
        if not supported:
            return self(value.float()).w.to(value.dtype)
        assert isinstance(self.scale_fmt, str)
        values = self._activation_values.get(value.device)
        if values is None:
            assert self.grid.values is not None
            values = self.grid.values.to(value.device).contiguous()
            self._activation_values[value.device] = values
        return reconstruct(
            value.contiguous(),
            values,
            self.group or value.shape[1],
            self.scale_fmt,
            self.asym,
            self.grid.integer,
            self._value_minimum,
            self._value_maximum,
            self._range_reciprocal if self.asym else self._magnitude_reciprocal,
        )

    def logical_bits(self, shape: tuple[int, int]) -> float:
        width = shape[1]
        group = min(self.group or width, width)
        scale_bits = (
            self.scale_fmt.bits
            if isinstance(self.scale_fmt, Grid)
            else SCALE_FORMAT_BITS[self.scale_fmt]
        )
        groups = shape[0] * ((width + group - 1) // group)
        return self.grid.bits * shape[0] * width / self.grid.dim + (
            scale_bits * groups * (2 if self.asym else 1)
        )

    def fit(self, w: Tensor, ctx: Ctx) -> FittedRounder:
        if w.ndim != 2 or not w.is_floating_point() or not torch.isfinite(w).all():
            raise ValueError("scaled requires a finite floating-point weight matrix")
        width = w.shape[1]
        group = width if self.group is None else min(self.group, width)
        if group % self.grid.dim:
            raise ValueError("group width must be divisible by the vector grid dimension")
        padding = (-width) % group
        grouped = functional.pad(w.float(), (0, padding)).reshape(w.shape[0], -1, group)
        # Padding completes the final group and must not influence its range or its error.
        occupied = None
        if padding:
            positions = torch.arange(grouped.shape[1] * group, device=w.device)
            occupied = (positions < width).reshape(1, -1, group)
        values = self.grid.values
        if self.grid.dim > 1 and values is not None and self.grid.name != "e8p":
            accounted = ctx.cache.setdefault("_mlkit_vector_codebooks", set())
            if id(self.grid) not in accounted:
                ctx.add_bits(values.numel() * values.element_size() * 8)
                accounted.add(id(self.grid))
        zero = None
        if self.asym:
            if occupied is None:
                minimum, maximum = grouped.amin(-1), grouped.amax(-1)
            else:
                minimum = grouped.masked_fill(~occupied, float("inf")).amin(-1)
                maximum = grouped.masked_fill(~occupied, -float("inf")).amax(-1)
            scales = (maximum - minimum).clamp_min(1e-12) * self._range_reciprocal
            zero = minimum - self._value_minimum * scales
        elif callable(self.scale):
            scales = self.scale(grouped.reshape(-1, group)).reshape(w.shape[0], -1)
        else:
            scales = grouped.abs().amax(-1).clamp_min(1e-12) * self._magnitude_reciprocal
        scales, scale_bits = store_scale(scales, self.scale_fmt)

        def round_grouped(value: Tensor, candidate: Tensor) -> Tensor:
            normalized = value / candidate[..., None]
            if self.grid.dim > 1:
                normalized = normalized.reshape(*normalized.shape[:-1], -1, self.grid.dim)
                return self.grid(normalized).reshape_as(value) * candidate[..., None]
            return self.grid(normalized) * candidate[..., None]

        def group_error(value: Tensor, candidate: Tensor) -> Tensor:
            error = (round_grouped(value, candidate) - value).square()
            return (error if occupied is None else error * occupied).sum(-1)

        if self.scale == "mse":
            centered = grouped if zero is None else grouped - zero[..., None]
            selected = scales.clone()
            best_error = group_error(centered, scales)
            for fraction in torch.linspace(0.5, 1.0, self.search_steps).tolist():
                candidate, _ = store_scale(scales * fraction, self.scale_fmt)
                error = group_error(centered, candidate)
                improved = error < best_error
                selected = torch.where(improved, candidate, selected)
                best_error = torch.minimum(best_error, error)
            scales = selected
        side_bits = scales.numel() * scale_bits * (2 if zero is not None else 1)
        if zero is not None:
            magnitude, _ = store_scale(zero.abs().clamp_min(1e-12), self.scale_fmt)
            zero = magnitude * zero.sign()
        scalar = None
        if self.grid.dim == 1 and values is not None:
            scalar = ScalarRounding(
                grid=self.grid,
                scales=scales,
                zero=zero,
                group=group,
                values=values.to(w.device),
                bits=self.grid.bits * w.numel() + side_bits,
                metadata={
                    "code_bits": self.grid.bits,
                    "scale_fmt": self.scale_fmt,
                    "trainable": ["scales"] + (["zero"] if zero is not None else []),
                },
            )

        def round_columns(value: Tensor, columns: slice) -> Q:
            start, stop, stride = columns.indices(width)
            if stride != 1 or stop - start != value.shape[1]:
                raise ValueError(
                    "rounder columns must describe a contiguous slice of its fitted region"
                )
            positions = torch.arange(start, stop, device=value.device) // group
            local_scales = scales[:, positions]
            centered = value.float() if zero is None else value.float() - zero[:, positions]
            normalized = centered / local_scales
            bits = (
                self.grid.bits * value.numel() / self.grid.dim
                + side_bits * value.shape[1] / width
            )
            if self.grid.dim == 1:
                rounded = self.grid(normalized)
                if scalar is not None:
                    if self.grid.integer:
                        codes = rounded - self._value_minimum
                    else:
                        codes = torch.searchsorted(scalar.values, rounded.contiguous())
                    storage_dtype = torch.uint8 if self.grid.bits <= 8 else torch.int32
                    parameters: dict[str, Any] = {
                        "scales": scales,
                        "values": scalar.values,
                        "group": group,
                        "offset": start,
                        "zero": zero,
                    }
                    return Q(
                        bits=bits,
                        codes=codes.to(storage_dtype),
                        params=parameters,
                        decode=decode_scaled,
                        codec="scaled",
                        metadata=scalar.metadata,
                    )
            else:
                if value.shape[1] % self.grid.dim:
                    raise ValueError("column slices must align to the vector grid dimension")
                if values is not None:
                    vectors = normalized.reshape(-1, self.grid.dim)
                    lattice = self.grid.name == "e8p"
                    if lattice:
                        indices = nearest_e8p(vectors, return_indices=True)
                    else:
                        indices = nearest(vectors, values, return_indices=True)
                    return Q(
                        codes=indices.reshape(value.shape[0], -1).to(torch.int32),
                        bits=bits,
                        params={
                            "scales": scales,
                            "values": None if lattice else values.to(value.device),
                            "dim": self.grid.dim,
                            "group": group,
                            "offset": start,
                            "zero": zero,
                        },
                        decode=decode_vector_scaled,
                        codec="vector_scaled",
                        metadata={
                            "code_bits": self.grid.bits,
                            "scale_fmt": self.scale_fmt,
                            "trainable": ["scales"],
                        },
                    )
                rounded = self.grid(normalized.reshape(-1, self.grid.dim)).reshape_as(value)
            reconstruction = rounded * local_scales
            if zero is not None:
                reconstruction = reconstruction + zero[:, positions]
            return Q(reconstruction, bits=bits)

        return FittedRounder(round_columns, scalar=scalar)

    def __repr__(self) -> str:
        return f"scaled({self.grid!r}, group={self.group}, scale={self.scale!r})"


def scaled(
    grid: Grid,
    group: int | None = 128,
    scale: str | Callable[[Tensor], Tensor] = "absmax",
    scale_fmt: str | Grid = "fp16",
    asym: bool = False,
    **options: Any,
) -> Scaled:
    return Scaled(grid, group, scale, scale_fmt, asym, **options)
