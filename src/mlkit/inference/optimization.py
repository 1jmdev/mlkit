"""Selection of dense or packed execution and optional compilation."""

import copy

from torch import nn

from mlkit.inference.packed_linear import PackedEmbedding, PackedLinear, packed_compatible
from mlkit.models.model import Model, QModel


def optimize(
    model: Model | nn.Module,
    backend: str = "auto",
    *,
    compile: bool = False,
    mode: str = "reduce-overhead",
    cache_dense: bool = False,
    inplace: bool = False,
) -> Model:
    """Preserve reconstructions while choosing dense or packed execution.

    Packed execution supports scalar Q codecs of one to eight bits. Other formats
    remain dense. An embedding tied to a packed output head reads its rows from
    the packed codes of the head. TorchAO exports are a separate operation
    because they requantize.
    """
    if backend not in {"auto", "dense", "packed"}:
        raise ValueError("backend must be auto, dense, or packed")
    wrapped = model if isinstance(model, Model) else Model(model)
    if isinstance(wrapped, QModel):
        _ = wrapped.model_bpw
    if backend == "auto":
        use_packed = isinstance(wrapped, QModel)
        backend = "packed" if use_packed else "dense"
    converted: Model
    if backend == "packed":
        if not isinstance(wrapped, QModel):
            raise ValueError("packed inference requires a QModel with codec state")
        named_replacements: dict[str, nn.Module] = {}
        for name, quantized in wrapped.quantized.items():
            module = wrapped.module.get_submodule(name)
            if isinstance(module, nn.Linear) and packed_compatible(quantized):
                named_replacements[name] = PackedLinear(
                    module, quantized, cache_dense=cache_dense
                )
        replacement_count = len(named_replacements)
        for parameter_name, layer_name in wrapped.tied_weights.items():
            owner_name = parameter_name.rpartition(".")[0]
            owner = wrapped.module.get_submodule(owner_name)
            projection = named_replacements.get(layer_name)
            if (
                isinstance(owner, nn.Embedding)
                and isinstance(projection, PackedLinear)
                and projection.layout is not None
            ):
                named_replacements[owner_name] = PackedEmbedding(projection, owner)
        replacements = {
            id(wrapped.module.get_submodule(name)): replacement
            for name, replacement in named_replacements.items()
        }
        if inplace:
            for name, replacement in named_replacements.items():
                if name:
                    wrapped.module.set_submodule(name, replacement)
                else:
                    wrapped.module = replacement
                    wrapped.architecture.model = replacement
            converted = wrapped
        else:
            converted = copy.deepcopy(wrapped, replacements)
            for module in converted.module.modules():
                if isinstance(module, PackedLinear):
                    module._forward_pre_hooks = copy.deepcopy(
                        module._forward_pre_hooks, replacements
                    )
        converted.execution_backend = f"packed:{replacement_count}"
    else:
        converted = wrapped if inplace else copy.deepcopy(wrapped)
        converted.execution_backend = "dense"
    if not inplace or backend == "packed":
        for attribute in (
            "_compiled_call", "_compiled_call_impl", "_last_compile_config", "_cache",
        ):
            if attribute in converted.module.__dict__:
                delattr(converted.module, attribute)
    if compile:
        if getattr(converted.module, "_mlkit_kv_quantizer", None) is not None:
            raise ValueError("online KV quantization currently supports eager generation")
        configuration = getattr(converted.module, "generation_config", None)
        if configuration is not None and hasattr(converted.module, "get_compiled_call"):
            from transformers import CompileConfig

            configuration = copy.deepcopy(configuration)
            configuration.cache_implementation = "static"
            configuration.compile_config = CompileConfig(mode=mode)
            converted.module.generation_config = configuration
        else:
            converted.module.compile(mode=mode)
        converted.execution_backend += "+compiled"
    return converted
