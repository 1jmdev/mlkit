"""Streaming calibration reductions collected on demand for one block at a time.

Second moments of half-precision activations are multiplied on tensor cores in
TF32, because such values fit the TF32 significand. The result differs from the
FP32 product by a few parts in 100,000, which is below the error of the FP32
factorization that consumes a Hessian and far below its damping. Setting
``TENSOR_FLOAT_PRODUCTS`` to false multiplies every second moment in FP32.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from mlkit.calibration.session import CalibrationSession, StopForward
from mlkit.models.architecture import identify_siblings

SECOND_MOMENT_BLOCK = 512
TENSOR_FLOAT_PRODUCTS = True
HALF_PRECISION_DTYPES = {torch.float16, torch.bfloat16}


def limit_token_energy(inputs: Tensor, limit: float | None) -> Tensor:
    """Scale down rows whose energy exceeds ``limit`` times the median row energy.

    A few tokens of a transformer carry activations orders of magnitude larger
    than all others. Left unlimited, they dominate every second-moment statistic,
    and error feedback then trades the accuracy of ordinary tokens for theirs.
    Rows within the limit are returned unchanged; ``None`` disables the limit.
    """
    if limit is None or inputs.shape[0] < 2:
        return inputs
    energy = inputs.square().sum(1)
    threshold = limit * energy.median()
    if threshold <= 0 or not bool((energy > threshold).any()):
        return inputs
    return inputs * (threshold / energy).clamp_max(1).sqrt()[:, None]


def add_upper_second_moment(total: Tensor, inputs: Tensor, *, tensor_float: bool) -> None:
    """Add ``inputsᵀ inputs`` to the blocks of ``total`` on and above its block diagonal.

    The product is symmetric, so the blocks below the diagonal are never computed;
    ``mirror_upper_blocks`` fills them once after the last batch. This costs a little
    more than half of the arithmetic of the full product. ``tensor_float`` permits
    TF32 products, for inputs that hold half-precision values.
    """
    width = inputs.shape[1]
    allowed = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tensor_float and TENSOR_FLOAT_PRODUCTS
    try:
        for start in range(0, width, SECOND_MOMENT_BLOCK):
            stop = min(width, start + SECOND_MOMENT_BLOCK)
            total[start:stop, start:].addmm_(inputs[:, start:stop].T, inputs[:, start:])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allowed


def mirror_upper_blocks(total: Tensor) -> None:
    """Copy the blocks above the block diagonal of ``total`` to their transposed positions."""
    width = total.shape[0]
    for start in range(0, width, SECOND_MOMENT_BLOCK):
        stop = min(width, start + SECOND_MOMENT_BLOCK)
        total[stop:, start:stop] = total[start:stop, stop:].T


class StatisticAccumulator:
    """A streaming reduction over layer inputs whose state stays on the input device."""

    def __init__(
        self,
        name: str,
        function: Callable[[Tensor], Tensor] | None = None,
        reduction: str = "mean",
        sample_rows: int = 4096,
    ) -> None:
        self.name = name
        self.function = function
        self.reduction = reduction
        self.sample_rows = sample_rows
        self.rows = 0
        self.total: Tensor | None = None
        self.samples: list[Tensor] = []
        self.finalized = False

    def update(self, inputs: Tensor, *, half_precision: bool | None = None) -> None:
        """Accumulate one batch of layer inputs.

        ``half_precision`` states that the values originate from half-precision
        activations; by default it is inferred from the dtype of ``inputs``.
        """
        if self.finalized:
            raise RuntimeError("a finalized statistic cannot accumulate further inputs")
        if half_precision is None:
            half_precision = inputs.dtype in HALF_PRECISION_DTYPES
        inputs = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        rows = inputs.shape[0]
        if self.name == "X":
            remaining = self.sample_rows - self.rows
            if remaining > 0:
                sample = inputs[:remaining].clone()
                self.samples.append(sample)
                self.rows += len(sample)
            return
        if self.name == "H":
            if self.total is None:
                width = inputs.shape[1]
                self.total = torch.zeros((width, width), device=inputs.device)
            add_upper_second_moment(self.total, inputs, tensor_float=half_precision)
            self.rows += rows
            return
        if self.name == "act_absmean":
            value = inputs.abs().sum(0)
        elif self.name == "act_absmax":
            value = inputs.abs().amax(0)
        elif self.function is not None:
            value = self.function(inputs)
            if self.reduction == "mean":
                value = value * rows
        else:
            raise KeyError(f"custom statistic {self.name!r} requires a function")
        if self.total is None:
            self.total = value
        elif self.reduction == "max" or self.name == "act_absmax":
            self.total = torch.maximum(self.total, value)
        else:
            self.total.add_(value)
        self.rows += rows

    def result(self) -> Tensor:
        """Finalize the reduction in place; the accumulator accepts no further inputs."""
        if self.name == "X":
            if not self.samples:
                raise ValueError("no calibration rows reached the selected layer")
            return torch.cat(self.samples)
        if self.total is None or self.rows == 0:
            raise ValueError("no calibration rows reached the selected layer")
        averaged = self.name in {"H", "act_absmean"} or (
            self.reduction == "mean" and self.name != "act_absmax"
        )
        if self.name == "H" and not self.finalized:
            mirror_upper_blocks(self.total)
        if averaged and not self.finalized:
            self.total.div_(self.rows)
        self.finalized = True
        return self.total


class BlockStatistics:
    def __init__(
        self,
        session: CalibrationSession,
        original_block: nn.Module,
        index: int,
        prefix: str,
    ) -> None:
        self.session = session
        self.block = original_block
        self.index = index
        self.prefix = prefix
        self.values: dict[tuple[str, str], Tensor] = {}

    def collect(self, statistic: str, function: Callable | None, reduction: str) -> None:
        self.collect_many({statistic: (function, reduction)})

    def collect_many(self, requirements: dict[str, tuple[Callable | None, str]]) -> None:
        session = self.session
        session.requirements.update(requirements)
        if not requirements:
            return
        session.prepare_data()
        layers = [
            (name, module) for name, module in self.block.named_modules()
            if isinstance(module, nn.Linear)
        ]
        siblings = identify_siblings(layers, self.prefix)
        modules = {f"{self.prefix}.{name}".strip("."): module for name, module in layers}
        aliases = {
            name: names[0] for name, names in siblings.items()
            if not any(modules[sibling]._forward_pre_hooks for sibling in names)
        }
        accumulators: dict[tuple[str, str], StatisticAccumulator] = {}
        handles = []
        for name, module in modules.items():
            if aliases.get(name, name) != name:
                continue
            collectors = []
            for statistic, (function, reduction) in requirements.items():
                if (name, statistic) in self.values:
                    continue
                cache_statistic = f"X:{session.sample_rows}" if statistic == "X" else statistic
                if session.disk_cache is not None and function is None:
                    cached = session.disk_cache.get(name, cache_statistic)
                    if cached is not None:
                        self.values[name, statistic] = cached
                        continue
                accumulator = StatisticAccumulator(
                    statistic, function, reduction, session.sample_rows
                )
                accumulators[name, statistic] = accumulator
                collectors.append(accumulator)
            if not collectors:
                continue

            def observe(
                layer: nn.Module,
                arguments: tuple,
                collectors: list[StatisticAccumulator] = collectors,
            ) -> None:
                half_precision = arguments[0].dtype in HALF_PRECISION_DTYPES
                inputs = arguments[0].detach().reshape(-1, arguments[0].shape[-1]).float()
                inputs = limit_token_energy(inputs, session.token_energy_limit)
                for collector in collectors:
                    collector.update(inputs, half_precision=half_precision)

            handles.append(module.register_forward_pre_hook(observe))
        try:
            if handles:
                with torch.no_grad():
                    for call in session.block_calls(self.index):
                        call.run(self.block)
            for (name, statistic), accumulator in accumulators.items():
                result = accumulator.result()
                self.values[name, statistic] = result
                if session.disk_cache is not None and requirements[statistic][0] is None:
                    cache_statistic = f"X:{session.sample_rows}" if statistic == "X" else statistic
                    session.disk_cache.put(name, cache_statistic, result)
            for name, canonical in aliases.items():
                for statistic in requirements:
                    if (canonical, statistic) in self.values:
                        self.values[name, statistic] = self.values[canonical, statistic]
        finally:
            for handle in handles:
                handle.remove()

    def provider(self, name: str, device: torch.device) -> Callable:
        def retrieve(statistic: str, function: Callable | None, reduction: str) -> Tensor:
            if (name, statistic) not in self.values:
                self.collect(statistic, function, reduction)
            return self.values[name, statistic].to(device)

        return retrieve


class ModelStatistics:
    """Statistics of layers outside the repeated blocks, from complete forward passes.

    A pass ends at the observed layer, so a wide output head is never evaluated.
    """

    def __init__(self, session: CalibrationSession) -> None:
        self.session = session
        self.values: dict[tuple[str, str], Tensor] = {}

    def collect(self, name: str, statistic: str, function: Callable | None, reduction: str) -> None:
        session = self.session
        accumulator = StatisticAccumulator(statistic, function, reduction, session.sample_rows)

        def observe(layer: nn.Module, arguments: tuple) -> None:
            half_precision = arguments[0].dtype in HALF_PRECISION_DTYPES
            inputs = arguments[0].detach().reshape(-1, arguments[0].shape[-1]).float()
            inputs = limit_token_energy(inputs, session.token_energy_limit)
            accumulator.update(inputs, half_precision=half_precision)
            raise StopForward

        handle = session.model.module.get_submodule(name).register_forward_pre_hook(observe)
        try:
            session.run_model()
        finally:
            handle.remove()
        self.values[name, statistic] = accumulator.result()

    def provider(self, name: str, device: torch.device) -> Callable:
        def retrieve(statistic: str, function: Callable | None, reduction: str) -> Tensor:
            if (name, statistic) not in self.values:
                self.collect(name, statistic, function, reduction)
            return self.values[name, statistic].to(device)

        return retrieve
