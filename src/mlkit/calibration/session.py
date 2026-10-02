"""Capture and replay of block inputs across calibration batches."""

import contextlib
import hashlib
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mlkit.calibration.activation_storage import ActivationStorage
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


class StopForward(Exception):
    """Ends a model forward inside a hook once the required inputs have been observed."""


@contextlib.contextmanager
def evaluation_mode(module: nn.Module) -> Iterator[None]:
    """Run ``module`` in evaluation mode and restore the training state of every submodule."""
    training_states = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, training in training_states:
            submodule.training = training


def equivalent_arguments(left: Any, right: Any) -> bool:
    """Whether two captured arguments hold the same tensors and equal values."""
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return left is right
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                equivalent_arguments(first, second)
                for first, second in zip(left, right, strict=True)
            )
        )
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(equivalent_arguments(left[name], right[name]) for name in left)
        )
    if left is right:
        return True
    try:
        return bool(left == right)
    except (RuntimeError, TypeError, ValueError):
        return False


class CalibrationSession:
    """Captures the inputs of every block once and replays them block by block."""

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
        storage: str = "auto",
        token_energy_limit: float | None = 100.0,
    ) -> None:
        if token_energy_limit is not None and token_energy_limit <= 0:
            raise ValueError("the token energy limit must be positive or None")
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
        self.storage = ActivationStorage(storage)
        self.token_energy_limit = token_energy_limit
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
            description = (
                model_fingerprint(self.model.module)
                + token_batches.fingerprint
                + f"token_energy_limit={self.token_energy_limit}"
            )
            identity = hashlib.sha256(description.encode()).hexdigest()
            self.disk_cache = StatisticsCache(self.cache_dir, identity)

    def prepare(self) -> None:
        """Run the calibration batches once and record the inputs of every block."""
        if self.calls is not None:
            return
        self.prepare_data()
        assert self.batches is not None
        self.calls = [[] for _ in self.model.blocks]
        with evaluation_mode(self.model.module), torch.no_grad():
            self._capture(self.batches[0], first_block_only=False)
            first_block_only = self._later_blocks_repeat_first_block_arguments()
            for batch in self.batches[1:]:
                self._capture(batch, first_block_only=first_block_only)

    def run_model(self) -> None:
        """Run every calibration batch through the model; a hook may end a pass early."""
        self.prepare_data()
        assert self.batches is not None
        with evaluation_mode(self.model.module), torch.no_grad():
            for batch in self.batches:
                with contextlib.suppress(StopForward):
                    forward_batch(self.model.module, batch)

    def _later_blocks_repeat_first_block_arguments(self) -> bool:
        """Whether one batch showed every block receiving the first block's side inputs.

        Sequential calibration replaces the hidden state of later blocks, so when
        their remaining arguments are the very tensors given to the first block,
        the forward pass can end as soon as the first block has been reached.
        """
        assert self.calls is not None
        if not self.sequential or self.need_targets or len(self.calls) < 2:
            return False
        if len(self.calls[0]) != 1:
            return False
        placeholder = torch.empty(0, device="cpu")
        reference = self.calls[0][0].with_hidden(placeholder)
        for calls in self.calls[1:]:
            if len(calls) != 1:
                return False
            candidate = calls[0].with_hidden(placeholder)
            if not equivalent_arguments(reference.args, candidate.args):
                return False
            if not equivalent_arguments(reference.kwargs, candidate.kwargs):
                return False
        return True

    def _capture(self, batch: Any, *, first_block_only: bool) -> None:
        assert self.calls is not None
        calls = self.calls
        handles = []
        stored_tensors: dict[int, tuple[weakref.ReferenceType[Tensor], Tensor]] = {}
        placeholder = torch.empty(0, device="cpu")

        def store(tensor: Tensor) -> Tensor:
            """Store each distinct tensor of a batch once, however many blocks receive it."""
            stored = stored_tensors.get(id(tensor))
            if stored is not None and stored[0]() is tensor:
                return stored[1]
            if tensor is placeholder:
                return tensor
            kept = self.storage.store(tensor, copy=True)
            stored_tensors[id(tensor)] = weakref.ref(tensor), kept
            return kept

        for index, block in enumerate(self.model.blocks):
            if self.selected_blocks is not None and index not in self.selected_blocks:
                continue
            if first_block_only and index > 0:
                break

            def capture_inputs(
                module: nn.Module,
                arguments: tuple,
                keywords: dict,
                block_index: int = index,
            ) -> None:
                call = BlockCall(arguments, keywords)
                if self.sequential and block_index > 0 and not self.need_targets:
                    call = call.with_hidden(placeholder)
                captured = BlockCall(
                    map_tensors(call.args, store), map_tensors(call.kwargs, store)
                )
                calls[block_index].append(captured)
                if first_block_only:
                    for later_calls in calls[1:]:
                        later_calls.append(captured.with_hidden(placeholder))
                    raise StopForward

            handles.append(block.register_forward_pre_hook(capture_inputs, with_kwargs=True))
            if self.need_targets:

                def capture_targets(
                    module: nn.Module,
                    arguments: tuple,
                    output: Any,
                    block_index: int = index,
                ) -> None:
                    self.targets[block_index].append(store(extract_hidden(output)))

                handles.append(block.register_forward_hook(capture_targets))
        try:
            forward_batch(self.model.module, batch)
        except StopForward:
            pass
        finally:
            for handle in handles:
                handle.remove()

    def block_calls(self, index: int) -> list[BlockCall]:
        self.prepare()
        assert self.calls is not None
        return self.calls[index]

    def propagate(self, index: int, block: nn.Module) -> None:
        """Replace the next block's inputs with the outputs of the converted block."""
        if not self.sequential or self.calls is None or index + 1 >= len(self.model.blocks):
            return
        with torch.no_grad():
            for sample, call in enumerate(self.calls[index]):
                hidden = self.storage.store(extract_hidden(call.run(block)), copy=False)
                self.calls[index + 1][sample] = self.calls[index + 1][sample].with_hidden(hidden)

    def release(self, index: int) -> None:
        """Discard completed block data unless a model pass requires the history."""
        if not self.retain_history:
            if self.calls is not None:
                self.calls[index] = []
            self.targets[index] = []
