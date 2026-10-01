"""Model wrappers that place modules on CUDA and record quantization results."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_map

from mlkit.models.architecture import ArchitectureAdapter, architecture_adapter
from mlkit.models.module_utilities import weight_name
from mlkit.models.reports import BlockPassReport, LayerReport
from mlkit.quantization.representation import Q


class Model(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        tokenizer: Any = None,
        *,
        name: str | None = None,
        architecture: ArchitectureAdapter | None = None,
    ) -> None:
        super().__init__()
        self.module = module.cuda()
        self.tokenizer = tokenizer
        self.name = name or type(module).__name__
        self.architecture = architecture or architecture_adapter(module)
        self.execution_backend = "dense"
        self._parameter_accounting: tuple[int, int] | None = None
        first_parameter = next(module.parameters(), None)
        self._initial_dtype = torch.float32 if first_parameter is None else first_parameter.dtype

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arguments, keywords = tree_map(
            lambda value: value.to(self.device) if isinstance(value, Tensor) else value,
            (args, kwargs),
        )
        return self.module(*arguments, **keywords)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        generate = cast(Callable[..., Any], self.module.generate)
        arguments, keywords = tree_map(
            lambda value: value.to(self.device) if isinstance(value, Tensor) else value,
            (args, kwargs),
        )
        return generate(*arguments, **keywords)

    @property
    def device(self) -> torch.device:
        tensor: Tensor | None = next(self.module.parameters(), None)
        if tensor is None:
            tensor = next(self.module.buffers(), None)
        return torch.device("cuda") if tensor is None else tensor.device

    @property
    def dtype(self) -> torch.dtype:
        tensor = next(self.module.parameters(), None)
        return self._initial_dtype if tensor is None else tensor.dtype

    @property
    def storage_bytes(self) -> int:
        """Bytes in distinct registered parameter and buffer storages, excluding KV caches."""
        storages = {}
        pending = [*self.module.parameters(), *self.module.buffers()]
        visited = set()
        while pending:
            tensor = pending.pop()
            if id(tensor) in visited:
                continue
            visited.add(id(tensor))
            flatten = getattr(tensor, "__tensor_flatten__", None)
            if flatten is not None:
                names, _ = flatten()
                pending.extend(getattr(tensor, name) for name in names)
                continue
            storage = tensor.untyped_storage()
            storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    @property
    def config(self) -> Any:
        return self.module.config

    @property
    def blocks(self) -> Sequence[nn.Module]:
        return self.architecture.blocks

    @property
    def residual_readers(self) -> list[nn.Linear]:
        return self.architecture.residual_readers

    @property
    def residual_writers(self) -> list[nn.Linear]:
        return self.architecture.residual_writers

    @property
    def norms(self) -> list[nn.Module]:
        return self.architecture.norms


class QModel(Model):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quantized: dict[str, Q] = {}
        self.layer_reports: list[LayerReport] = []
        self.pass_reports: list[BlockPassReport] = []
        self.activation_handles: list[Any] = []
        self.activation_specs: dict[str, dict[str, Any] | None] = {}
        self.kv_spec: dict[str, Any] | None = None
        self.execution_backend = "dense"

    @property
    def bpw(self) -> float | None:
        if not self.layer_reports:
            return None
        if any(record.bits is None for record in self.layer_reports):
            return None
        bits = sum(record.bits for record in self.layer_reports if record.bits is not None)
        return bits / sum(record.elements for record in self.layer_reports)

    @property
    def model_bpw(self) -> float | None:
        """Include untouched embeddings, output heads, norms, and biases."""
        if self.bpw is None:
            return None
        if self._parameter_accounting is None:
            selected = {weight_name(record.name) for record in self.layer_reports}
            original_bits = 0
            original_elements = 0
            for name, parameter in self.module.named_parameters():
                original_elements += parameter.numel()
                if name not in selected:
                    original_bits += parameter.numel() * parameter.element_size() * 8
            self._parameter_accounting = original_elements, original_bits
        original_elements, original_bits = self._parameter_accounting
        quantized_bits = sum(
            record.bits for record in self.layer_reports if record.bits is not None
        )
        return (original_bits + quantized_bits) / original_elements

    def report(self, *, print_table: bool = True) -> list[dict[str, Any]]:
        records = [asdict(record) | {"bpw": record.bpw} for record in self.layer_reports]
        if print_table:
            print(f"{'layer':58} {'bpw':>7} {'loss':>12} {'seconds':>9}")
            for record in self.layer_reports:
                precision = "?" if record.bpw is None else f"{record.bpw:.3f}"
                print(f"{record.name:58} {precision:>7} {record.loss:12.5g} {record.seconds:9.3f}")
        return records

    def save(self, path: str | Path, **options: Any) -> None:
        # Checkpoints depend on this wrapper, so the convenience entry point imports lazily.
        from mlkit.checkpoints.saving import save

        save(self, path, **options)
