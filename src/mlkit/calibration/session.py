"""Capture and replay of block inputs across calibration batches."""

import hashlib
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mlkit.calibration.statistics_cache import StatisticsCache, model_fingerprint
from mlkit.calibration.token_batches import DataSource, TokenBatches, data, normalize_batches
from mlkit.models.model import Model
from mlkit.models.module_utilities import extract_hidden, module_device


def map_tensors(value: Any, function: Callable[[Tensor], Tensor]) -> Any:
    if isinstance(value, Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    if isinstance(value, Mapping):
        return {name: map_tensors(item, function) for name, item in value.items()}
    return value


@dataclass
class BlockCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def with_hidden(self, hidden: Tensor) -> "BlockCall":
        if "hidden_states" in self.kwargs:
            return BlockCall(self.args, self.kwargs | {"hidden_states": hidden})
        if not self.args:
            raise ValueError("architecture block must receive hidden_states or a positional tensor")
        return BlockCall((hidden, *self.args[1:]), self.kwargs)

    def hidden(self) -> Tensor:
        return self.kwargs["hidden_states"] if "hidden_states" in self.kwargs else self.args[0]

    def run(self, block: nn.Module) -> Any:
        device = module_device(block)
        arguments = map_tensors(self.args, lambda tensor: tensor.to(device))
        keywords = map_tensors(self.kwargs, lambda tensor: tensor.to(device))
        return block(*arguments, **keywords)


def forward_batch(model: nn.Module, batch: Any) -> Any:
    device = module_device(model)
    batch = map_tensors(batch, lambda tensor: tensor.to(device))
    if isinstance(batch, Mapping):
        if hasattr(model, "config"):
            batch = dict(batch) | {"use_cache": False}
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    return model(batch)


class CalibrationSession:
    def __init__(
        self,
        model: Model,
        calibration: Any,
        *,
        sequential: bool,
        sample_rows: int,
        cache_dir: str | Path | None,
        need_targets: bool,
        retain_history: bool = True,
        selected_blocks: tuple[int, ...] | None = None,
    ) -> None:
        if selected_blocks is not None:
            if sequential:
                raise ValueError("selected block capture requires nonsequential calibration")
            if any(index < 0 or index >= len(model.blocks) for index in selected_blocks):
                raise ValueError("selected calibration block index is outside the model")
        self.model = model
        self.calibration = calibration
        self.sequential = sequential
        self.sample_rows = sample_rows
        self.cache_dir = cache_dir
        self.need_targets = need_targets
        self.retain_history = retain_history
        self.selected_blocks = selected_blocks
        self.calls: list[list[BlockCall]] | None = None
        self.targets: list[list[Tensor]] = [[] for _ in model.blocks]
        self.requirements: dict[str, tuple[Callable | None, str]] = {}
        self.disk_cache: StatisticsCache | None = None
        self.batches: list[Any] | None = None

    def prepare_data(self) -> None:
        if self.batches is not None:
            return
        calibration = self.calibration
        if isinstance(calibration, str):
            calibration = data(calibration, tokenizer=self.model.tokenizer)
        if isinstance(calibration, DataSource):
            calibration = calibration.bind(self.model.tokenizer)
        if calibration is None:
            raise ValueError("this quantizer requires calibration; supply calib token batches")
        self.batches = normalize_batches(calibration)
        if not self.batches:
            raise ValueError("calibration must contain at least one batch")
        if not self.sequential and self.cache_dir is not None:
            token_batches = TokenBatches(
                batch if isinstance(batch, dict) else {"inputs": batch} for batch in self.batches
            )
            identity = hashlib.sha256(
                (model_fingerprint(self.model.module) + token_batches.fingerprint).encode()
            ).hexdigest()
            self.disk_cache = StatisticsCache(self.cache_dir, identity)

    def prepare(self) -> None:
        if self.calls is not None:
            return
        self.prepare_data()
        assert self.batches is not None
        self.calls = [[] for _ in self.model.blocks]
        handles = []
        host_tensors: dict[int, tuple[weakref.ReferenceType[Tensor], Tensor]] = {}

        def transfer_to_host(tensor: Tensor) -> Tensor:
            stored = host_tensors.get(id(tensor))
            if stored is not None and stored[0]() is tensor:
                return stored[1]
            transferred = tensor.detach().cpu()
            host_tensors[id(tensor)] = weakref.ref(tensor), transferred
            return transferred

        for index, block in enumerate(self.model.blocks):
            if self.selected_blocks is not None and index not in self.selected_blocks:
                continue

            def capture_inputs(
                module: nn.Module,
                arguments: tuple,
                keywords: dict,
                block_index: int = index,
            ) -> None:
                assert self.calls is not None
                call = BlockCall(arguments, keywords)
                if self.sequential and block_index > 0 and not self.need_targets:
                    call = call.with_hidden(torch.empty(0, device="cpu"))
                self.calls[block_index].append(BlockCall(
                    map_tensors(call.args, transfer_to_host),
                    map_tensors(call.kwargs, transfer_to_host),
                ))

            handles.append(block.register_forward_pre_hook(capture_inputs, with_kwargs=True))
            if self.need_targets:

                def capture_targets(
                    module: nn.Module,
                    arguments: tuple,
                    output: Any,
                    block_index: int = index,
                ) -> None:
                    self.targets[block_index].append(transfer_to_host(extract_hidden(output)))

                handles.append(block.register_forward_hook(capture_targets))
        try:
            training_states = [(module, module.training) for module in self.model.module.modules()]
            self.model.module.eval()
            with torch.no_grad():
                for batch in self.batches:
                    host_tensors.clear()
                    forward_batch(self.model.module, batch)
        finally:
            for module, training in training_states:
                module.training = training
            for handle in handles:
                handle.remove()

    def block_calls(self, index: int) -> list[BlockCall]:
        self.prepare()
        assert self.calls is not None
        return self.calls[index]

    def propagate(self, index: int, block: nn.Module) -> None:
        if not self.sequential or self.calls is None or index + 1 >= len(self.model.blocks):
            return
        with torch.no_grad():
            for sample, call in enumerate(self.calls[index]):
                hidden = extract_hidden(call.run(block)).detach().cpu()
                self.calls[index + 1][sample] = self.calls[index + 1][sample].with_hidden(hidden)

    def release(self, index: int) -> None:
        """Discard completed block data unless a model pass requires the history."""
        if not self.retain_history:
            if self.calls is not None:
                self.calls[index] = []
            self.targets[index] = []
