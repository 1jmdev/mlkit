"""Model wrappers and extensible architecture discovery."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_map

from mlkit.quantization.representation import Q


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


@dataclass
class LayerReport:
    name: str
    shape: tuple[int, int]
    bits: float | None
    elements: int
    loss: float
    seconds: float
    method: str

    @property
    def bpw(self) -> float | None:
        return None if self.bits is None else self.bits / self.elements


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
        from mlkit.runtime.serialization import save

        save(self, path, **options)


def load(
    name: str | Path,
    dtype: str | torch.dtype = "auto",
    **options: Any,
) -> Model:
    path = Path(name)
    if (path / "mlkit.json").is_file():
        from mlkit.runtime.serialization import load_checkpoint

        return load_checkpoint(path, **options)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Hugging Face loading requires uv add 'mlkit[transformers]'") from error
    selected_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else dtype
    module: nn.Module = AutoModelForCausalLM.from_pretrained(
        str(name), dtype=selected_dtype, **options
    )
    module.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(name), trust_remote_code=False)
    return Model(module, tokenizer, name=str(name))


def extract_hidden(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError("block output must be a tensor, tuple, or last_hidden_state result")


def preserve_input_processing(original: nn.Module, replacement: nn.Module) -> None:
    """Preserve online transforms when an inference or training layer is substituted."""
    replacement._forward_pre_hooks = original._forward_pre_hooks.copy()
    replacement._forward_pre_hooks_with_kwargs = original._forward_pre_hooks_with_kwargs.copy()
    for name, value in original.named_buffers(recurse=False):
        if name.startswith("_mlkit_"):
            replacement.register_buffer(name, value.detach().clone())


def weight_name(module_name: str) -> str:
    return f"{module_name}.weight" if module_name else "weight"


def module_device(module: nn.Module) -> torch.device:
    tensor: Tensor | None = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return torch.device("cuda") if tensor is None else tensor.device
