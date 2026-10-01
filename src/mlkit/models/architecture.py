"""Extensible discovery of repeated blocks, projections and normalization sites."""

from collections.abc import Callable, Sequence

import torch
from torch import nn


class ArchitectureAdapter:
    """Access repeated blocks without coupling formats to a transformer family."""

    def __init__(self, model: nn.Module, block_path: str | None = None) -> None:
        self.model = model
        self.block_path = block_path or self.discover_blocks(model)

    @staticmethod
    def discover_blocks(model: nn.Module) -> str | None:
        for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
            try:
                sequence = model.get_submodule(path)
            except AttributeError:
                continue
            if isinstance(sequence, (nn.ModuleList, nn.Sequential)):
                return path
        return None

    @property
    def blocks(self) -> Sequence[nn.Module]:
        if self.block_path is None:
            return (self.model,)
        sequence = self.model.get_submodule(self.block_path)
        if not isinstance(sequence, (nn.ModuleList, nn.Sequential)):
            raise TypeError("architecture block path must identify a ModuleList or Sequential")
        return list(sequence)

    def block_name(self, index: int) -> str:
        return "" if self.block_path is None else f"{self.block_path}.{index}"

    @property
    def norms(self) -> list[nn.Module]:
        return [
            module for module in self.model.modules()
            if isinstance(module, nn.LayerNorm) or "rmsnorm" in type(module).__name__.lower()
        ]

    @property
    def residual_readers(self) -> list[nn.Linear]:
        suffixes = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "lm_head"}
        return [
            module for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear) and name.split(".")[-1] in suffixes
        ]

    @property
    def residual_writers(self) -> list[nn.Linear]:
        suffixes = {"o_proj", "down_proj"}
        return [
            module for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear) and name.split(".")[-1] in suffixes
        ]


_ADAPTER_FACTORIES: dict[str, Callable[[nn.Module], ArchitectureAdapter]] = {}


def adapter(
    model_type: str,
) -> Callable[[Callable[[nn.Module], ArchitectureAdapter]], Callable]:
    def register(factory: Callable[[nn.Module], ArchitectureAdapter]) -> Callable:
        _ADAPTER_FACTORIES[model_type] = factory
        return factory

    return register


def architecture_adapter(model: nn.Module) -> ArchitectureAdapter:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    factory = _ADAPTER_FACTORIES.get(model_type, ArchitectureAdapter)
    return factory(model)


def identify_siblings(
    layers: list[tuple[str, nn.Linear]],
    prefix: str,
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
        if (
            type(module).__name__ != "Conv1D"
            or not type(module).__module__.startswith("transformers.")
        ):
            continue
        weight = module.weight
        converted = nn.Linear(
            weight.shape[0],
            weight.shape[1],
            bias=module.bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            converted.weight.copy_(weight.T)
            if converted.bias is not None:
                converted.bias.copy_(module.bias)
        converted.weight.requires_grad_(weight.requires_grad)
        model.set_submodule(name, converted)
