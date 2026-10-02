"""Grouped scaling and fitting of scalar and vector grids."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from mlkit.kernels import scalar_encode
from mlkit.kernels.activation_reconstruction import reconstruct
from mlkit.quantization.codecs.scaled import decode_scaled, decode_vector_scaled
from mlkit.quantization.context import Ctx
from mlkit.quantization.formats.scale_storage import SCALE_FORMAT_BITS, store_scale
from mlkit.quantization.grids import Grid
from mlkit.quantization.grids.lattice import device_table, nearest_e8p
from mlkit.quantization.operations.search import nearest
from mlkit.quantization.protocol import FittedRounder, Quantizer, ScalarRounding, VectorRounding
from mlkit.quantization.representation import Q

FUSED_CODEBOOK_LIMIT = 256


def single_precision_reciprocal(value: float) -> float:
    """The FP32 reciprocal, so that host arithmetic and fused kernels agree bit for bit."""
    one = torch.tensor(1.0, dtype=torch.float32)
    return float(one / torch.tensor(value, dtype=torch.float32))


def group_extrema(weight: Tensor, group: int) -> tuple[Tensor, Tensor]:
    """Minimum and maximum of every row group; a partial final group holds only its columns."""
    rows, width = weight.shape
    whole = width - width % group
    minima, maxima = [], []
    if whole:
        grouped = weight[:, :whole].reshape(rows, -1, group)
        minima.append(grouped.amin(-1))
        maxima.append(grouped.amax(-1))
    if whole < width:
        remainder = weight[:, whole:]
        minima.append(remainder.amin(-1, keepdim=True))
        maxima.append(remainder.amax(-1, keepdim=True))
    return torch.cat(minima, dim=1), torch.cat(maxima, dim=1)


def nearest_codes(normalized: Tensor, values: Tensor) -> Tensor:
    """Indices of the nearest ascending codebook values; ties select the lower value."""
    upper = torch.searchsorted(values, normalized.contiguous()).clamp_max(values.numel() - 1)
    lower = (upper - 1).clamp_min(0)
    select_lower = (normalized - values[lower]).abs() <= (normalized - values[upper]).abs()
    return torch.where(select_lower, lower, upper)


class Scaled(Quantizer):
    """A grid applied to row groups that each carry a scale and, optionally, an offset.

    Scalar grids with explicit values round to the nearest value. On CUDA they are
    fitted and encoded by fused kernels that reproduce the tensor reference exactly.
    """

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
        self._device_values: dict[torch.device, Tensor] = {}
        values = None if grid.values is None or grid.dim != 1 else grid.values.detach().cpu()
        self._value_minimum = 0.0 if values is None else float(values.min())
        self._value_maximum = 0.0 if values is None else float(values.max())
        magnitude = 1.0 if grid.values is None else float(grid.values.abs().max())
        value_range = self._value_maximum - self._value_minimum
        if asym and value_range <= 0:
            raise ValueError("asymmetric quantization requires at least two distinct grid values")
        self._magnitude_reciprocal = single_precision_reciprocal(magnitude)
        self._range_reciprocal = single_precision_reciprocal(value_range) if asym else 1.0

    def values_on(self, device: torch.device) -> Tensor:
        """The grid values on ``device``, transferred once."""
        values = self._device_values.get(device)
        if values is None:
            assert self.grid.values is not None
            values = self.grid.values.to(device).contiguous()
            self._device_values[device] = values
        return values

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
        return reconstruct(
            value.contiguous(),
            self.values_on(value.device),
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
        if w.ndim != 2 or not w.is_floating_point():
            raise ValueError("scaled requires a finite floating-point weight matrix")
        rows, width = w.shape
        group = width if self.group is None else min(self.group, width)
        if group % self.grid.dim:
            raise ValueError("group width must be divisible by the vector grid dimension")
        values = self.grid.values
        if self.grid.dim > 1 and values is not None and self.grid.name != "e8p":
            accounted = ctx.cache.setdefault("_mlkit_vector_codebooks", set())
            if id(self.grid) not in accounted:
                ctx.add_bits(values.numel() * values.element_size() * 8)
                accounted.add(id(self.grid))
        weight = w.float()
        scales, zero = self._initial_scales(weight, group)
        scales, scale_bits = store_scale(scales, self.scale_fmt)
        scalar_values = None
        if self.grid.dim == 1 and values is not None:
            scalar_values = self.values_on(w.device)
        tile = None
        if w.is_cuda and scalar_values is not None and len(scalar_values) <= FUSED_CODEBOOK_LIMIT:
            tile = scalar_encode.tile_for(group)
        if self.scale == "mse":
            scales = self._search_scales(weight, scales, zero, group, scalar_values, tile)
        side_bits = scales.numel() * scale_bits * (2 if zero is not None else 1)
        if zero is not None:
            magnitude, _ = store_scale(zero.abs().clamp_min(1e-12), self.scale_fmt)
            zero = magnitude * zero.sign()
        scalar = None
        if scalar_values is not None:
            scalar = ScalarRounding(
                grid=self.grid,
                scales=scales,
                zero=zero,
                group=group,
                values=scalar_values,
                bits=self.grid.bits * w.numel() + side_bits,
                metadata={
                    "code_bits": self.grid.bits,
                    "scale_fmt": self.scale_fmt,
                    "trainable": ["scales"] + (["zero"] if zero is not None else []),
                },
            )
        vector = None
        vector_metadata = {
            "code_bits": self.grid.bits,
            "scale_fmt": self.scale_fmt,
            "trainable": ["scales"],
        }
        if self.grid.dim > 1 and values is not None and w.is_cuda:
            lattice = self.grid.name == "e8p"
            vector = VectorRounding(
                grid=self.grid,
                scales=scales,
                group=group,
                codebook=device_table(w.device) if lattice else self.values_on(w.device),
                codebook_size=len(values),
                lattice=lattice,
                bits=self.grid.bits * w.numel() / self.grid.dim + side_bits,
                metadata=vector_metadata,
            )

        def round_columns(value: Tensor, columns: slice) -> Q:
            start, stop, stride = columns.indices(width)
            if stride != 1 or stop - start != value.shape[1]:
                raise ValueError(
                    "rounder columns must describe a contiguous slice of its fitted region"
                )
            bits = (
                self.grid.bits * value.numel() / self.grid.dim
                + side_bits * value.shape[1] / width
            )
            if scalar is not None:
                codes = self._scalar_codes(value, scalar, start, stop, tile)
                return Q(
                    bits=bits,
                    codes=codes,
                    params={
                        "scales": scales,
                        "values": scalar.values,
                        "group": group,
                        "offset": start,
                        "zero": zero,
                    },
                    decode=decode_scaled,
                    codec="scaled",
                    metadata=scalar.metadata,
                )
            positions = torch.arange(start, stop, device=value.device) // group
            local_scales = scales[:, positions]
            centered = value.float() if zero is None else value.float() - zero[:, positions]
            normalized = centered / local_scales
            if self.grid.dim == 1:
                rounded = self.grid(normalized)
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
                        metadata=vector_metadata,
                    )
                rounded = self.grid(normalized.reshape(-1, self.grid.dim)).reshape_as(value)
            reconstruction = rounded * local_scales
            if zero is not None:
                reconstruction = reconstruction + zero[:, positions]
            return Q(reconstruction, bits=bits)

        return FittedRounder(round_columns, scalar=scalar, vector=vector)

    def _initial_scales(self, weight: Tensor, group: int) -> tuple[Tensor, Tensor | None]:
        """Unrounded group scales and offsets from the range of each group."""
        rows, width = weight.shape
        if callable(self.scale) and not self.asym:
            if not torch.isfinite(weight).all():
                raise ValueError("scaled requires a finite floating-point weight matrix")
            padded = functional.pad(weight, (0, (-width) % group))
            return self.scale(padded.reshape(-1, group)).reshape(rows, -1), None
        minimum, maximum = group_extrema(weight, group)
        if not (torch.isfinite(minimum).all() and torch.isfinite(maximum).all()):
            raise ValueError("scaled requires a finite floating-point weight matrix")
        if self.asym:
            scales = (maximum - minimum).clamp_min(1e-12) * self._range_reciprocal
            return scales, minimum - self._value_minimum * scales
        magnitude = torch.maximum(maximum, -minimum)
        return magnitude.clamp_min(1e-12) * self._magnitude_reciprocal, None

    def _search_scales(
        self,
        weight: Tensor,
        scales: Tensor,
        zero: Tensor | None,
        group: int,
        scalar_values: Tensor | None,
        tile: int | None,
    ) -> Tensor:
        """Among shrunken candidates, the stored scale of every group with least squared error."""
        fractions = torch.linspace(0.5, 1.0, self.search_steps, device=weight.device)
        fused = (
            tile is not None
            and scalar_values is not None
            and isinstance(self.scale_fmt, str)
            and weight.is_contiguous()
        )
        if fused:
            assert tile is not None and scalar_values is not None
            assert isinstance(self.scale_fmt, str)
            return scalar_encode.search_scales(
                weight,
                scales.contiguous(),
                None if zero is None else zero.contiguous(),
                scalar_values,
                fractions,
                group,
                tile,
                self.scale_fmt,
                integer=self.grid.integer,
                minimum=self._value_minimum,
                maximum=self._value_maximum,
            )
        rows, width = weight.shape
        padding = (-width) % group
        grouped = functional.pad(weight, (0, padding)).reshape(rows, -1, group)
        occupied = None
        if padding:
            positions = torch.arange(grouped.shape[1] * group, device=weight.device)
            occupied = (positions < width).reshape(1, -1, group)
        centered = grouped if zero is None else grouped - zero[..., None]

        def group_error(candidate: Tensor) -> Tensor:
            normalized = centered / candidate[..., None]
            if self.grid.dim > 1:
                vectors = normalized.reshape(*normalized.shape[:-1], -1, self.grid.dim)
                rounded = self.grid(vectors).reshape_as(centered)
            else:
                rounded = self.grid(normalized)
            error = (rounded * candidate[..., None] - centered).square()
            return (error if occupied is None else error * occupied).sum(-1)

        selected = scales.clone()
        best_error = group_error(scales)
        for fraction in fractions.tolist():
            candidate, _ = store_scale(scales * fraction, self.scale_fmt)
            error = group_error(candidate)
            improved = error < best_error
            selected = torch.where(improved, candidate, selected)
            best_error = torch.minimum(best_error, error)
        return selected

    def _scalar_codes(
        self,
        value: Tensor,
        scalar: ScalarRounding,
        start: int,
        stop: int,
        tile: int | None,
    ) -> Tensor:
        """Codes of the nearest scalar grid values for columns ``start`` to ``stop``."""
        fused = (
            tile is not None
            and start == 0
            and value.is_cuda
            and value.dtype == torch.float32
            and value.is_contiguous()
        )
        if fused:
            assert tile is not None
            return scalar_encode.encode(
                value,
                scalar.scales,
                scalar.zero,
                scalar.values,
                scalar.group,
                tile,
                integer=self.grid.integer,
                minimum=self._value_minimum,
                maximum=self._value_maximum,
            )
        positions = torch.arange(start, stop, device=value.device) // scalar.group
        centered = value.float()
        if scalar.zero is not None:
            centered = centered - scalar.zero[:, positions]
        normalized = centered / scalar.scales[:, positions]
        if self.grid.integer:
            codes = normalized.round().clamp(self._value_minimum, self._value_maximum)
            codes = codes - self._value_minimum
        else:
            codes = nearest_codes(normalized, scalar.values)
        return codes.to(torch.uint8 if self.grid.bits <= 8 else torch.int32)

    def __repr__(self) -> str:
        scale = self.scale if isinstance(self.scale, str) else getattr(
            self.scale, "__name__", type(self.scale).__name__
        )
        description = f"scaled({self.grid!r}, group={self.group}, scale={scale!r}"
        if self.scale_fmt != "fp16":
            description += f", scale_fmt={self.scale_fmt!r}"
        if self.asym:
            description += ", asym=True"
        return description + ")"


def scaled(
    grid: Grid,
    group: int | None = 128,
    scale: str | Callable[[Tensor], Tensor] = "absmax",
    scale_fmt: str | Grid = "fp16",
    asym: bool = False,
    **options: Any,
) -> Scaled:
    return Scaled(grid, group, scale, scale_fmt, asym, **options)
