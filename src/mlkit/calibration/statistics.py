"""Streaming calibration reductions collected on demand for one block at a time."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from mlkit.calibration.session import CalibrationSession
from mlkit.models.architecture import identify_siblings


class StatisticAccumulator:
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

    def update(self, inputs: Tensor) -> None:
        inputs = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        rows = inputs.shape[0]
        if self.name == "X":
            remaining = self.sample_rows - self.rows
            if remaining > 0:
                sample = inputs[:remaining].cpu()
                self.samples.append(sample)
                self.rows += len(sample)
            return
        if self.name == "H":
            value = inputs.T @ inputs
        elif self.name == "act_absmean":
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
        if self.name == "X":
            if not self.samples:
                raise ValueError("no calibration rows reached the selected layer")
            return torch.cat(self.samples)
        if self.total is None or self.rows == 0:
            raise ValueError("no calibration rows reached the selected layer")
        if (self.name in {"H", "act_absmean"} or self.reduction == "mean") and (
            self.name != "act_absmax"
        ):
            return (self.total / self.rows).cpu()
        return self.total.cpu()


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
                inputs = arguments[0].detach().reshape(-1, arguments[0].shape[-1]).float()
                for collector in collectors:
                    collector.update(inputs)

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
