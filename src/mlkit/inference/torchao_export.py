"""Optional requantization of a model with a maintained TorchAO execution backend."""

import copy

from torch import nn

from mlkit.models.model import Model


def export_torchao(
    model: Model | nn.Module,
    *,
    bits: int = 4,
    group: int = 128,
    packing_format: str = "tile_packed_to_4d",
) -> Model:
    """Requantize a separate model using a maintained TorchAO execution backend."""
    try:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
        from torchao.quantization.quant_api import Int4PackingFormat
    except ImportError as error:
        raise ImportError("TorchAO export requires uv add 'mlkit[torchao]'") from error
    wrapped = model if isinstance(model, Model) else Model(model)
    module = copy.deepcopy(wrapped.module)
    if bits == 4:
        if packing_format == "tile_packed_to_4d":
            for parameter in module.parameters():
                if parameter.is_floating_point():
                    parameter.data = parameter.data.bfloat16()
        configuration = Int4WeightOnlyConfig(
            group_size=group, int4_packing_format=Int4PackingFormat(packing_format),
            set_inductor_config=False,
        )
    elif bits == 8:
        configuration = Int8WeightOnlyConfig(version=2, set_inductor_config=False)
    else:
        raise ValueError("TorchAO export supports four-bit or eight-bit weights")
    quantize_(module, configuration, filter_fn=quantizable_linear)
    converted = Model(module, wrapped.tokenizer, name=wrapped.name)
    converted.execution_backend = f"torchao-int{bits}"
    return converted


def quantizable_linear(module: nn.Module, name: str) -> bool:
    return isinstance(module, nn.Linear) and name.split(".")[-1] not in {
        "lm_head", "embed_out", "output",
    }
