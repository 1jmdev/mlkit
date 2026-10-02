"""Demand-driven, blockwise model conversion."""

import copy
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mlkit.calibration.session import CalibrationSession
from mlkit.calibration.statistics import BlockStatistics, ModelStatistics
from mlkit.conversion.activation_quantization import (
    describe_quantizer,
    install_activation_quantization,
)
from mlkit.conversion.block_passes import run_block_passes
from mlkit.conversion.key_value_quantization import install_kv_quantization
from mlkit.models.architecture import (
    architecture_adapter,
    identify_siblings,
    normalize_affine_layers,
)
from mlkit.models.model import Model, QModel
from mlkit.models.module_utilities import weight_name
from mlkit.models.reports import LayerReport
from mlkit.quantization.context import Ctx
from mlkit.quantization.operations.losses import proxy_loss
from mlkit.quantization.recipes import Recipe, normalize_recipe
from mlkit.quantization.representation import as_q
from mlkit.timing import synchronize


def quantize(
    model: Model | nn.Module,
    recipe: Any,
    calib: Any = "c4",
    sequential: bool = True,
    *,
    seed: int = 0,
    sample_rows: int = 4096,
    cache_dir: str | Path | None = "~/.cache/mlkit/statistics",
    calibration_storage: str = "auto",
    token_energy_limit: float | None = 100.0,
) -> QModel:
    """Convert a separate model copy; statistics and datasets remain lazy.

    ``calibration_storage`` places captured block inputs: ``"auto"`` keeps them on
    CUDA while a memory reserve remains, ``"cuda"`` always does, and ``"host"``
    moves them to host memory. ``token_energy_limit`` bounds the energy of one
    calibration token relative to the median token of its batch, so that a few
    massive activations cannot dominate the statistics; ``None`` disables it.
    """
    source = model if isinstance(model, Model) else Model(model)
    definition: Recipe = normalize_recipe(recipe)
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
        retain_history=bool(definition.model_passes),
        storage=calibration_storage,
        token_energy_limit=token_energy_limit,
    )
    if definition.passes or (definition.transforms and not sequential):
        session.prepare()
    shared_cache: dict[str, Any] = {}
    for index, block in enumerate(converted.blocks):
        prefix = converted.architecture.block_name(index)
        original_block = copy.deepcopy(block)
        statistics = BlockStatistics(session, original_block, index, prefix)
        statistics.collect_many(dict(session.requirements))
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
        activation_contexts = []
        for relative_name, layer in layers:
            name = f"{prefix}.{relative_name}".strip(".")
            context = contexts[name]
            quantization = algorithms[name]
            if quantization is None:
                continue
            convert_layer(
                converted, name, layer, quantization, context,
                retain_device=bool(definition.passes),
            )
            if definition.acts is not None:
                activation_quantizer = resolve_activation(definition.acts, name, context)
                if activation_quantizer is not None:
                    activation_context = Ctx(
                        name, layer, block, index, provider=context._provider, cache=shared_cache,
                        siblings=context.siblings, seed=seed, device=layer.weight.device,
                    )
                    activation_contexts.append(activation_context)
                    with torch.no_grad():
                        sample = torch.zeros(1, layer.in_features, device=layer.weight.device)
                        as_q(activation_quantizer(sample, activation_context))

                    converted.activation_specs[name] = describe_quantizer(activation_quantizer)
                    converted.activation_handles.append(install_activation_quantization(
                        layer, activation_quantizer, activation_context
                    ))
            if not definition.passes:
                context._stats.clear()
        if definition.passes:
            run_block_passes(converted, index, block, original_block, session, definition.passes)
            reports = {report.name: report for report in converted.layer_reports}
            with torch.no_grad():
                for relative_name, layer in layers:
                    name = f"{prefix}.{relative_name}".strip(".")
                    if name not in reports:
                        continue
                    original_layer = original_block.get_submodule(relative_name)
                    assert isinstance(original_layer, nn.Linear)
                    hessian = contexts[name]._stats.get("H")
                    loss_context = Ctx(H=hessian) if hessian is not None else None
                    reports[name].loss = float(proxy_loss(
                        original_layer.weight, layer.weight, loss_context
                    ))
        session.propagate(index, block)
        for key in ["_mlkit_sibling_modules", "_mlkit_layer_algorithms", "_mlkit_awq_results"]:
            shared_cache.pop(key, None)
        if definition.acts is not None:
            original_block.to("cpu")
            for activation_context in activation_contexts:
                activation_context._provider = None
        del original_block, statistics, contexts
        session.release(index)
    if definition.head is not None:
        convert_head_layers(converted, definition, session, shared_cache, seed)
    for model_pass in definition.model_passes:
        model_pass(converted, session)
    if definition.kv is not None:
        converted.activation_handles.append(install_kv_quantization(
            converted.module, definition.kv
        ))
        converted.kv_spec = describe_quantizer(definition.kv)
    return converted


def convert_layer(
    converted: QModel,
    name: str,
    layer: nn.Linear,
    quantization: Any,
    context: Ctx,
    *,
    retain_device: bool,
) -> None:
    """Round one layer in place and record its representation and report."""
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
    converted.quantized[name] = result if retain_device else result.to("cpu", detach=True)
    converted.layer_reports.append(LayerReport(
        name, (weight.shape[0], weight.shape[1]), bits,
        weight.numel(), loss, duration, repr(quantization),
    ))


def convert_head_layers(
    converted: QModel,
    definition: Recipe,
    session: CalibrationSession,
    shared_cache: dict[str, Any],
    seed: int,
) -> None:
    """Round the linear layers outside the repeated blocks, after every block.

    Their statistics come from complete forward passes of the calibration model.
    A parameter that shares the storage of a rounded weight, such as a tied input
    embedding, takes the rounded values and is recorded in ``tied_weights``.
    """
    block_path = converted.architecture.block_path
    if block_path is None:
        return
    statistics = ModelStatistics(session)
    for name, layer in list(converted.module.named_modules()):
        if not isinstance(layer, nn.Linear) or name.startswith(f"{block_path}."):
            continue
        context = Ctx(
            name, layer, None, len(converted.blocks),
            provider=statistics.provider(name, layer.weight.device),
            cache=shared_cache, seed=seed, device=layer.weight.device,
        )
        quantization = definition.select_head(name, context)
        if quantization is None:
            continue
        convert_layer(converted, name, layer, quantization, context, retain_device=False)
        context._stats.clear()
        for parameter_name, parameter in converted.module.named_parameters(
            remove_duplicate=False
        ):
            if parameter is layer.weight and parameter_name != weight_name(name):
                converted.tied_weights[parameter_name] = name


def resolve_activation(value: Any, name: str, context: Ctx) -> Any:
    if isinstance(value, Mapping):
        return Recipe(weights=value).select(name, context)
    return value
