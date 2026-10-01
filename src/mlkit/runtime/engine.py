"""Demand-driven, blockwise model conversion."""

import copy
import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mlkit.experiments.data import DataSource, TokenBatches, data, normalize_batches
from mlkit.quantization.context import Ctx
from mlkit.quantization.operations import proxy_loss
from mlkit.quantization.recipes import Recipe, normalize_recipe
from mlkit.quantization.representation import as_q
from mlkit.runtime.models import (
    LayerReport,
    Model,
    QModel,
    architecture_adapter,
    extract_hidden,
    module_device,
)
from mlkit.runtime.statistics import StatisticAccumulator, StatisticsCache, model_fingerprint


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
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.sequential = sequential
        self.sample_rows = sample_rows
        self.cache_dir = cache_dir
        self.need_targets = need_targets
        self.calls: list[list[BlockCall]] | None = None
        self.targets: list[list[Tensor]] = [[] for _ in model.blocks]
        self.requirements: dict[str, tuple[Callable | None, str]] = {}
        self.disk_cache: StatisticsCache | None = None
        self.batches: list[Any] | None = None

    def prepare(self) -> None:
        if self.calls is not None:
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
        self.calls = [[] for _ in self.model.blocks]
        handles = []
        for index, block in enumerate(self.model.blocks):
            def capture_inputs(
                module: nn.Module,
                arguments: tuple,
                keywords: dict,
                block_index: int = index,
            ) -> None:
                assert self.calls is not None
                call = BlockCall(
                    map_tensors(arguments, lambda tensor: tensor.detach().cpu()),
                    map_tensors(keywords, lambda tensor: tensor.detach().cpu()),
                )
                if self.sequential and block_index > 0 and not self.need_targets:
                    call = call.with_hidden(torch.empty(0))
                self.calls[block_index].append(call)

            handles.append(block.register_forward_pre_hook(capture_inputs, with_kwargs=True))
            if self.need_targets:
                def capture_targets(
                    module: nn.Module, arguments: tuple, output: Any, block_index: int = index,
                ) -> None:
                    self.targets[block_index].append(extract_hidden(output).detach().cpu())

                handles.append(block.register_forward_hook(capture_targets))
        try:
            training_states = [(module, module.training) for module in self.model.module.modules()]
            self.model.module.eval()
            with torch.no_grad():
                for batch in self.batches:
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
        session = self.session
        session.requirements[statistic] = (function, reduction)
        calls = session.block_calls(self.index)
        accumulators = {}
        handles = []
        for relative_name, module in self.block.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            name = f"{self.prefix}.{relative_name}".strip(".")
            if (name, statistic) in self.values:
                continue
            if session.disk_cache is not None and function is None:
                cached = session.disk_cache.get(name, statistic)
                if cached is not None:
                    self.values[name, statistic] = cached
                    continue
            accumulator = StatisticAccumulator(statistic, function, reduction, session.sample_rows)
            accumulators[name] = accumulator

            def observe(
                layer: nn.Module,
                arguments: tuple,
                collector: StatisticAccumulator = accumulator,
            ) -> None:
                collector.update(arguments[0])

            handles.append(module.register_forward_pre_hook(observe))
        try:
            if handles:
                with torch.no_grad():
                    for call in calls:
                        call.run(self.block)
            for name, accumulator in accumulators.items():
                result = accumulator.result()
                self.values[name, statistic] = result
                if session.disk_cache is not None and function is None:
                    session.disk_cache.put(name, statistic, result)
        finally:
            for handle in handles:
                handle.remove()

    def provider(self, name: str, device: torch.device) -> Callable:
        def retrieve(statistic: str, function: Callable | None, reduction: str) -> Tensor:
            if (name, statistic) not in self.values:
                self.collect(statistic, function, reduction)
            return self.values[name, statistic].to(device)

        return retrieve


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def quantize(
    model: Model | nn.Module,
    recipe: Any,
    calib: Any = "c4",
    sequential: bool = True,
    *,
    seed: int = 0,
    sample_rows: int = 4096,
    cache_dir: str | Path | None = "~/.cache/mlkit/statistics",
) -> QModel:
    """Convert a separate model copy; statistics and datasets remain lazy."""
    source = model if isinstance(model, Model) else Model(model)
    definition: Recipe = normalize_recipe(recipe)
    if definition.kv is not None:
        raise NotImplementedError("KV quantization requires an architecture-specific cache adapter")
    converted_module = copy.deepcopy(source.module).eval()
    normalize_affine_layers(converted_module)
    converted = QModel(converted_module, source.tokenizer, name=source.name)
    converted.architecture = architecture_adapter(converted_module)
    for transform in definition.transforms:
        transform(converted, calib)
    calibration_model = converted if sequential or definition.transforms else source
    session = CalibrationSession(
        calibration_model, calib, sequential=sequential, sample_rows=sample_rows,
        cache_dir=None if cache_dir is None else Path(cache_dir).expanduser(),
        need_targets=bool(definition.passes),
    )
    if definition.passes or (definition.transforms and not sequential):
        session.prepare()
    shared_cache: dict[str, Any] = {}
    for index, block in enumerate(converted.blocks):
        prefix = converted.architecture.block_name(index)
        original_block = copy.deepcopy(block)
        statistics = BlockStatistics(session, original_block, index, prefix)
        for statistic, (function, reduction) in list(session.requirements.items()):
            statistics.collect(statistic, function, reduction)
        layers = [(name, layer) for name, layer in block.named_modules()
                  if isinstance(layer, nn.Linear)]
        contexts = {}
        sibling_groups = identify_siblings(layers, prefix)
        for relative_name, layer in layers:
            name = f"{prefix}.{relative_name}".strip(".")
            context = Ctx(
                name, layer, block, index, provider=statistics.provider(name, layer.weight.device),
                cache=shared_cache, seed=seed, device=layer.weight.device,
                siblings=sibling_groups.get(name, ()),
            )
            contexts[name] = context
        algorithms = {name: definition.select(name, context) for name, context in contexts.items()}
        shared_cache["_mlkit_sibling_modules"] = {
            f"{prefix}.{name}".strip("."): original_block.get_submodule(name)
            for name, _ in layers
        }
        shared_cache["_mlkit_layer_algorithms"] = algorithms
        for relative_name, layer in layers:
            name = f"{prefix}.{relative_name}".strip(".")
            context = contexts[name]
            quantization = algorithms[name]
            if quantization is None:
                continue
            weight = layer.weight.detach().float()
            synchronize(weight.device)
            start = time.perf_counter()
            with torch.no_grad():
                result = as_q(quantization(weight, context))
                reconstruction = result.w
                if reconstruction.shape != weight.shape or not torch.isfinite(reconstruction).all():
                    raise ValueError(f"quantizer returned an invalid reconstruction for {name}")
                known_hessian = context._stats.get("H")
                loss_context = Ctx(H=known_hessian) if known_hessian is not None else None
                loss = float(proxy_loss(weight, reconstruction, loss_context))
                layer.weight.copy_(reconstruction.to(layer.weight.dtype))
            synchronize(weight.device)
            duration = time.perf_counter() - start
            bits = None if result.bits is None else result.bits + context._additional_bits
            result.bits = bits
            converted.quantized[name] = (
                result if definition.passes else result.to("cpu", detach=True)
            )
            converted.layer_reports.append(LayerReport(
                name, (weight.shape[0], weight.shape[1]), bits,
                weight.numel(), loss, duration, repr(quantization),
            ))
            if definition.acts is not None:
                activation_quantizer = resolve_activation(definition.acts, name, context)
                if activation_quantizer is not None:
                    def quantize_inputs(
                        module: nn.Module,
                        arguments: tuple,
                        algorithm: Callable = activation_quantizer,
                        layer_context: Ctx = context,
                    ) -> tuple:
                        inputs = arguments[0]
                        shape = inputs.shape
                        value = as_q(algorithm(
                            inputs.reshape(-1, shape[-1]).float(), layer_context
                        ))
                        return (value.w.reshape(shape).to(inputs.dtype), *arguments[1:])

                    converted.activation_handles.append(layer.register_forward_pre_hook(
                        quantize_inputs
                    ))
        if definition.passes:
            from mlkit.runtime.passes import run_block_passes

            run_block_passes(converted, index, block, original_block, session, definition.passes)
        session.propagate(index, block)
        for key in ["_mlkit_sibling_modules", "_mlkit_layer_algorithms", "_mlkit_awq_results"]:
            shared_cache.pop(key, None)
        if definition.acts is not None:
            original_block.to("cpu")
            for context in contexts.values():
                context._stats = {name: value.cpu() for name, value in context._stats.items()}
        del original_block, statistics, contexts
    for model_pass in definition.model_passes:
        model_pass(converted, session)
    return converted


def resolve_activation(value: Any, name: str, context: Ctx) -> Any:
    if isinstance(value, Mapping):
        return Recipe(weights=value).select(name, context)
    return value


def identify_siblings(
    layers: list[tuple[str, nn.Linear]], prefix: str,
) -> dict[str, tuple[str, ...]]:
    """Identify projections known to consume the same transformer activations."""
    groups: dict[tuple[str, str], list[str]] = {}
    for name, _ in layers:
        parent, _, field = name.rpartition(".")
        family = "attention" if field in {"q_proj", "k_proj", "v_proj"} else (
            "feedforward" if field in {"gate_proj", "up_proj"} else None
        )
        if family is not None:
            groups.setdefault((parent, family), []).append(f"{prefix}.{name}".strip("."))
    return {name: tuple(names) for names in groups.values() if len(names) > 1 for name in names}


def normalize_affine_layers(model: nn.Module) -> None:
    """Convert Hugging Face GPT-2 Conv1D projections to ordinary Linear layers."""
    for name, module in list(model.named_modules()):
        if (type(module).__name__ != "Conv1D"
                or not type(module).__module__.startswith("transformers.")):
            continue
        weight = module.weight
        converted = nn.Linear(weight.shape[0], weight.shape[1], bias=module.bias is not None,
                              device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            converted.weight.copy_(weight.T)
            if converted.bias is not None:
                converted.bias.copy_(module.bias)
        converted.weight.requires_grad_(weight.requires_grad)
        model.set_submodule(name, converted)
