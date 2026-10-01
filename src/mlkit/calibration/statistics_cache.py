"""Safe, content-addressed disk caching of calibration statistics."""

import hashlib
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


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
