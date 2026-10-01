"""Grouped scaling, fitting, and portable scalar-grid codecs."""

import builtins
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from mlkit.quantization.context import Ctx
from mlkit.quantization.grids import NF4_VALUES, Grid, grid
from mlkit.quantization.operations import nearest
from mlkit.quantization.protocol import Quantizer
from mlkit.quantization.representation import Q


def decode_scaled(
    codes: Tensor,
    *,
    scales: Tensor,
    values: Tensor,
    group: builtins.int,
    offset: builtins.int = 0,
    zero: Tensor | None = None,
) -> Tensor:
    columns = torch.arange(offset, offset + codes.shape[1], device=codes.device) // group
    reconstruction = values[codes.long()] * scales[:, columns]
    return reconstruction if zero is None else reconstruction + zero[:, columns]


def decode_feedback(
    codes: Tensor,
    *,
    scales: Tensor,
    values: Tensor,
    group: builtins.int,
    refit: builtins.int,
    zero: Tensor | None = None,
    permutation: Tensor | None = None,
) -> Tensor:
    columns = torch.arange(codes.shape[1], device=codes.device)
    groups_per_region = (refit + group - 1) // group
    scale_indices = (columns // refit) * groups_per_region + (columns % refit) // group
    reconstruction = values[codes.long()] * scales[:, scale_indices]
    if zero is not None:
        reconstruction = reconstruction + zero[:, scale_indices]
    return reconstruction if permutation is None else reconstruction[:, permutation.argsort()]


def decode_vector_scaled(
    codes: Tensor, *, scales: Tensor, values: Tensor | None, group: builtins.int,
    dim: builtins.int, offset: builtins.int = 0, zero: Tensor | None = None,
    refit: builtins.int | None = None, permutation: Tensor | None = None,
) -> Tensor:
    if values is None:
        from mlkit.quantization.lattice import device_points

        values = device_points(codes.device)
    width = codes.shape[1] * dim
    columns = torch.arange(offset, offset + width, device=codes.device)
    positions = columns // group if refit is None else (
        (columns // refit) * ((refit + group - 1) // group) + (columns % refit) // group
    )
    reconstruction = values[codes.long()].reshape(codes.shape[0], width) * scales[:, positions]
    if zero is not None:
        reconstruction = reconstruction + zero[:, positions]
    return reconstruction if permutation is None else reconstruction[:, permutation.argsort()]


def store_scale(value: Tensor, format: str | Grid) -> tuple[Tensor, builtins.int]:
    if isinstance(format, Grid):
        if format.dim != 1:
            raise ValueError("scale grids must be scalar")
        return format(value).clamp_min(torch.finfo(torch.float32).tiny), format.bits
    if format == "fp32":
        return value.float(), 32
    if format == "fp16":
        return value.clamp(2**-24, 65504).half().float(), 16
    if format == "bf16":
        return value.bfloat16().float(), 16
    if format == "fp8":
        return value.clamp(2**-9, 448).to(torch.float8_e4m3fn).float(), 8
    if format == "e8m0":
        return torch.pow(2.0, value.clamp_min(2**-127).log2().round().clamp(-127, 127)), 8
    raise ValueError(f"unsupported scale storage format {format!r}")


class Scaled(Quantizer):
    def __init__(
        self,
        grid: Grid,
        group: builtins.int | None = 128,
        scale: str | Callable[[Tensor], Tensor] = "absmax",
        scale_fmt: str | Grid = "fp16",
        asym: bool = False,
        *,
        search_steps: builtins.int = 20,
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
        self._value_absolute_maximum = 1.0 if values is None else float(values.abs().max())

    def reconstruct_activations(self, value: Tensor) -> Tensor:
        """Reconstruct online scalar grids without codec allocation or host synchronization."""
        supported = (
            value.is_cuda and self.grid.dim == 1 and self.grid.values is not None
            and self.scale == "absmax" and isinstance(self.scale_fmt, str)
            and self.scale_fmt in {"fp32", "fp16", "bf16", "fp8", "e8m0"}
        )
        if not supported:
            return self(value.float()).w.to(value.dtype)
        assert isinstance(self.scale_fmt, str)
        from mlkit.runtime.kernels.activations import reconstruct

        values = self._activation_values.get(value.device)
        if values is None:
            assert self.grid.values is not None
            values = self.grid.values.to(value.device).contiguous()
            self._activation_values[value.device] = values
        return reconstruct(
            value.contiguous(), values, self.group or value.shape[1], self.scale_fmt, self.asym,
            self.grid.name.startswith("int"), self._value_minimum, self._value_maximum,
            self._value_absolute_maximum,
        )

    def logical_bits(self, shape: tuple[builtins.int, builtins.int]) -> float:
        width = shape[1]
        group = min(self.group or width, width)
        formats = {"fp32": 32, "fp16": 16, "bf16": 16, "fp8": 8, "e8m0": 8}
        scale_bits = (self.scale_fmt.bits if isinstance(self.scale_fmt, Grid)
                      else formats[self.scale_fmt])
        groups = shape[0] * ((width + group - 1) // group)
        return self.grid.bits * shape[0] * width / self.grid.dim + (
            scale_bits * groups * (2 if self.asym else 1)
        )

    def fit(self, w: Tensor, ctx: Ctx) -> Callable[[Tensor, slice], Q]:
        if w.ndim != 2 or not w.is_floating_point() or not torch.isfinite(w).all():
            raise ValueError("scaled requires a finite floating-point weight matrix")
        width = w.shape[1]
        group = width if self.group is None else min(self.group, width)
        if group % self.grid.dim:
            raise ValueError("group width must be divisible by the vector grid dimension")
        padding = (-width) % group
        grouped = functional.pad(w.float(), (0, padding)).reshape(w.shape[0], -1, group)
        values = self.grid.values
        if self.grid.dim > 1 and values is not None and self.grid.name != "e8p":
            accounted = ctx.cache.setdefault("_mlkit_vector_codebooks", set())
            if id(self.grid) not in accounted:
                ctx.add_bits(values.numel() * values.element_size() * 8)
                accounted.add(id(self.grid))
        maximum = 1.0 if values is None else float(values.abs().max())
        zero = None
        if self.asym:
            assert values is not None
            minimum_value, maximum_value = float(values.min()), float(values.max())
            minimum, maximum_group = grouped.amin(-1), grouped.amax(-1)
            scales = (maximum_group - minimum).clamp_min(1e-12) / (maximum_value - minimum_value)
            zero = minimum - minimum_value * scales
        elif callable(self.scale):
            scales = self.scale(grouped.reshape(-1, group)).reshape(w.shape[0], -1)
        else:
            scales = grouped.abs().amax(-1).clamp_min(1e-12) / maximum
        scales, scale_bits = store_scale(scales, self.scale_fmt)

        def round_grouped(value: Tensor, candidate: Tensor) -> Tensor:
            normalized = value / candidate[..., None]
            if self.grid.dim > 1:
                normalized = normalized.reshape(*normalized.shape[:-1], -1, self.grid.dim)
                return self.grid(normalized).reshape_as(value) * candidate[..., None]
            return self.grid(normalized) * candidate[..., None]

        if self.scale == "mse":
            centered = grouped if zero is None else grouped - zero[..., None]
            selected = scales.clone()
            best_error = (round_grouped(centered, scales) - centered).square().sum(-1)
            for fraction in torch.linspace(0.5, 1.0, self.search_steps).tolist():
                candidate, _ = store_scale(scales * fraction, self.scale_fmt)
                error = (round_grouped(centered, candidate) - centered).square().sum(-1)
                improved = error < best_error
                selected = torch.where(improved, candidate, selected)
                best_error = torch.minimum(best_error, error)
            scales = selected
        side_bits = scales.numel() * scale_bits * (2 if zero is not None else 1)
        if zero is not None:
            magnitude, _ = store_scale(zero.abs().clamp_min(1e-12), self.scale_fmt)
            zero = magnitude * zero.sign()

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
                if values is not None:
                    local_values = values.to(device=value.device)
                    codes = torch.searchsorted(local_values, rounded.contiguous())
                    storage_dtype = torch.uint8 if self.grid.bits <= 8 else torch.int32
                    parameters: dict[str, Any] = {
                        "scales": scales,
                        "values": local_values,
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
                        metadata={
                            "code_bits": self.grid.bits, "scale_fmt": self.scale_fmt,
                            "trainable": ["scales"] + (["zero"] if zero is not None else []),
                        },
                    )
            else:
                if value.shape[1] % self.grid.dim:
                    raise ValueError("column slices must align to the vector grid dimension")
                if values is not None:
                    vectors = normalized.reshape(-1, self.grid.dim)
                    if self.grid.name == "e8p":
                        from mlkit.quantization.lattice import nearest_e8p

                        indices = nearest_e8p(vectors, return_indices=True)
                    else:
                        indices = nearest(vectors, values, return_indices=True)
                    return Q(
                        codes=indices.reshape(value.shape[0], -1).to(torch.int32), bits=bits,
                        params={
                            "scales": scales, "values": None if self.grid.name == "e8p" else
                            values.to(value.device), "dim": self.grid.dim, "group": group,
                            "offset": start, "zero": zero,
                        },
                        decode=decode_vector_scaled, codec="vector_scaled",
                        metadata={
                            "code_bits": self.grid.bits, "scale_fmt": self.scale_fmt,
                            "trainable": ["scales"],
                        },
                    )
                rounded = self.grid(normalized.reshape(-1, self.grid.dim)).reshape_as(value)
            reconstruction = rounded * local_scales
            if zero is not None:
                reconstruction = reconstruction + zero[:, positions]
            return Q(reconstruction, bits=bits)

        return round_columns

    def __repr__(self) -> str:
        return f"scaled({self.grid!r}, group={self.group}, scale={self.scale!r})"


def scaled(
    grid: Grid,
    group: builtins.int | None = 128,
    scale: str | Callable[[Tensor], Tensor] = "absmax",
    scale_fmt: str | Grid = "fp16",
    asym: bool = False,
    **options: Any,
) -> Scaled:
    return Scaled(grid, group, scale, scale_fmt, asym, **options)


def int(
    bits: builtins.int = 4,
    group: builtins.int | None = 128,
    **options: Any,
) -> Scaled:
    return scaled(grid.int(bits), group=group, **options)


def nf4(group: builtins.int | None = 64, **options: Any) -> Scaled:
    return scaled(grid.values(NF4_VALUES, bits=4), group=group, **options)


def mxfp4(group: builtins.int = 32, **options: Any) -> Scaled:
    return scaled(grid.fp("e2m1"), group=group, scale_fmt="e8m0", **options)
