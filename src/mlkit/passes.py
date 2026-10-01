"""Differentiable block passes for codecs and normalization parameters."""

import functools
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mlkit.context import layer_seed
from mlkit.engine import BlockCall, CalibrationSession
from mlkit.models import QModel, extract_hidden
from mlkit.representation import Q


class ConfiguredPass:
    def __init__(self, function: Callable, parameters: dict[str, Any] | None = None) -> None:
        functools.update_wrapper(self, function)
        self.function = function
        self.parameters = parameters or {}

    def __call__(self, *arguments: Any, **parameters: Any) -> Any:
        if not arguments:
            return ConfiguredPass(self.function, self.parameters | parameters)
        return self.function(*arguments, **(self.parameters | parameters))


def block_pass(function: Callable) -> ConfiguredPass:
    return ConfiguredPass(function)


def model_pass(function: Callable) -> ConfiguredPass:
    return ConfiguredPass(function)


def norm_params(block: nn.Module) -> list[nn.Parameter]:
    parameters = []
    for module in block.modules():
        if isinstance(module, nn.LayerNorm) or "rmsnorm" in type(module).__name__.lower():
            parameters.extend(module.parameters(recurse=False))
    return parameters


class CodecLinear(nn.Module):
    def __init__(self, original: nn.Linear, quantized: Q) -> None:
        super().__init__()
        if quantized.codes is None or quantized.decode is None:
            raise ValueError("differentiable linear requires a codec")
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.dtype = original.weight.dtype
        self.bias = original.bias
        self.register_buffer("codes", quantized.codes)
        self.decoder = quantized.decode
        self.constants = {}
        self.parameters_by_name = nn.ParameterDict()
        self.parameter_names = {}
        trainable = quantized.metadata.get("trainable")
        for index, (name, value) in enumerate(quantized.params.items()):
            if (isinstance(value, Tensor) and value.is_floating_point()
                    and (trainable is None or name in trainable)):
                identifier = f"parameter_{index}"
                self.parameters_by_name[identifier] = nn.Parameter(value.detach().clone())
                self.parameter_names[name] = identifier
                quantized.params[name] = self.parameters_by_name[identifier]
            else:
                self.constants[name] = value

    @property
    def weight(self) -> Tensor:
        parameters = {name: self.parameters_by_name[identifier]
                      for name, identifier in self.parameter_names.items()}
        return self.decoder(self.codes, **(self.constants | parameters))

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.linear(inputs, self.weight.to(self.dtype), self.bias)


class BlockPassCtx:
    def __init__(
        self,
        block: nn.Module,
        original: nn.Module,
        calls: list[BlockCall],
        targets: list[Tensor],
        parameters: list[nn.Parameter],
        index: int,
    ) -> None:
        self.calls = calls
        self.fp_block = original
        self.qparams = parameters
        self.block_idx = index
        device = next(block.parameters()).device
        self.rng = torch.Generator(device=device).manual_seed(layer_seed(f"block.{index}"))
        shapes = {tuple(call.hidden().shape[1:]) for call in calls}
        if len(shapes) != 1:
            raise ValueError(
                "block passes require calibration batches with matching sequence shapes"
            )
        self.inputs = torch.cat([call.hidden() for call in calls]).to(device)
        self.targets = torch.cat(targets).to(device)
        self.loss_history: list[float] = []

    def forward(self, block: nn.Module, inputs: Tensor) -> Tensor:
        return extract_hidden(self.calls[0].with_hidden(inputs).run(block))


def run_block_passes(
    model: QModel,
    index: int,
    block: nn.Module,
    original: nn.Module,
    session: CalibrationSession,
    passes: Sequence[Callable],
) -> None:
    calls = session.block_calls(index)
    targets = session.targets[index]
    prefix = model.architecture.block_name(index)
    replacements: list[tuple[str, nn.Linear, CodecLinear]] = []
    qparams = []
    original_gradients = {
        name: parameter.requires_grad for name, parameter in block.named_parameters()
    }
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    for name, module in list(block.named_modules()):
        full_name = f"{prefix}.{name}".strip(".")
        quantized = model.quantized.get(full_name)
        if not isinstance(module, nn.Linear) or quantized is None or quantized.decode is None:
            continue
        replacement = CodecLinear(module, quantized)
        if not name:
            raise ValueError("block passes require a containing block, rather than a bare Linear")
        block.set_submodule(name, replacement)
        replacements.append((name, module, replacement))
        qparams.extend(replacement.parameters_by_name.values())
    for parameter in norm_params(block):
        parameter.requires_grad_(True)
    context = BlockPassCtx(block, original, calls, targets, qparams, index)
    try:
        with torch.enable_grad():
            for operation in passes:
                operation(block, context)
    finally:
        with torch.no_grad():
            for name, module, replacement in replacements:
                module.weight.copy_(replacement.weight.to(module.weight.dtype))
                block.set_submodule(name, module)
            for name, parameter in block.named_parameters():
                parameter.requires_grad_(original_gradients[name])
        for name in list(model.quantized):
            if not prefix or name.startswith(prefix + "."):
                model.quantized[name] = model.quantized[name].to("cpu", detach=True)


@block_pass
def finetune(
    block: nn.Module,
    ctx: BlockPassCtx,
    steps: int = 200,
    lr: float = 1e-4,
    bs: int = 8,
) -> None:
    if steps < 0 or lr <= 0 or bs < 1:
        raise ValueError("finetuning requires steps >= 0, lr > 0, and bs >= 1")
    parameters = ctx.qparams + norm_params(block)
    if not parameters or steps == 0:
        return
    optimizer = torch.optim.Adam(parameters, lr=lr)
    device = next(block.parameters()).device
    for _ in range(steps):
        indices = torch.randint(len(ctx.calls), (bs,), generator=ctx.rng, device=device).tolist()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index in indices:
            prediction = extract_hidden(ctx.calls[index].run(block))
            offset = sum(ctx.calls[previous].hidden().shape[0] for previous in range(index))
            count = ctx.calls[index].hidden().shape[0]
            target = ctx.targets[offset : offset + count]
            loss = functional.mse_loss(prediction.float(), target.float()) / bs
            loss.backward()
            total_loss += float(loss.detach())
        optimizer.step()
        ctx.loss_history.append(total_loss)
