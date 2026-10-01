"""Streaming calibration reductions and safe, content-addressed disk caching."""

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


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


def model_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class StatisticsCache:
    def __init__(self, directory: str | Path, identity: str) -> None:
        self.directory = Path(directory) / identity

    def path(self, layer: str, statistic: str) -> Path:
        digest = hashlib.sha256(f"{layer}:{statistic}".encode()).hexdigest()
        return self.directory / f"{digest}.safetensors"

    def get(self, layer: str, statistic: str) -> Tensor | None:
        path = self.path(layer, statistic)
        return load_file(str(path))["value"] if path.is_file() else None

    def put(self, layer: str, statistic: str, value: Tensor) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, suffix=".safetensors")
        os.close(descriptor)
        try:
            save_file({"value": value.contiguous().cpu()}, temporary)
            os.replace(temporary, self.path(layer, statistic))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
