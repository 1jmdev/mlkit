"""Reading of mlkit checkpoints into a quantized model on CUDA."""

import json
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from mlkit.checkpoints.manifest import FORMAT_VERSION, MANIFEST_FILE, WEIGHTS_FILE
from mlkit.conversion.activation_quantization import (
    install_activation_quantization,
    restore_quantizer,
)
from mlkit.conversion.key_value_quantization import install_kv_quantization
from mlkit.conversion.model_transforms import install_transform, record_transform
from mlkit.models.model import QModel
from mlkit.models.module_utilities import weight_name
from mlkit.models.reports import BlockPassReport, LayerReport
from mlkit.quantization.codecs import decoder
from mlkit.quantization.context import Ctx
from mlkit.quantization.operations import unpack
from mlkit.quantization.representation import Q


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | Callable[[], nn.Module] | None = None,
) -> QModel:
    directory = Path(path)
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported mlkit checkpoint format version")
    tokenizer = None
    external_architecture = model is not None
    if model is None:
        if manifest["architecture"] != "transformers":
            raise ValueError(
                "external architecture checkpoints require model=module or model=factory"
            )
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("checkpoint loading requires 'mlkit[transformers]'") from error
        configuration = AutoConfig.from_pretrained(directory, trust_remote_code=False)
        model = AutoModelForCausalLM.from_config(
            configuration, trust_remote_code=False, dtype=getattr(torch, manifest["dtype"])
        )
        if (directory / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
    elif not isinstance(model, nn.Module):
        model = model()
    assert isinstance(model, nn.Module)
    if external_architecture:
        model.to(dtype=getattr(torch, manifest["dtype"]))
    tensors = load_file(str(directory / WEIGHTS_FILE))
    state = {name.removeprefix("state."): value for name, value in tensors.items()
             if name.startswith("state.")}
    for alias, canonical_name in manifest.get("state_aliases", {}).items():
        state[alias] = state[canonical_name]
    converted = QModel(model, tokenizer, name=manifest["name"])
    for name, layer in manifest["layers"].items():
        if layer["codec"] is not None:
            codes = unpack(tensors[f"{name}.codes"], layer["code_bits"], tuple(layer["shape"]))
            parameters = {
                parameter_name: (
                    decode_parameter(tensors, value) if "tensor" in value else value["value"]
                )
                for parameter_name, value in layer["params"].items()
            }
            quantized = Q(
                codes=codes, params=parameters, decode=decoder(layer["codec"]),
                codec=layer["codec"], bits=layer["bits"],
                metadata=layer.get("metadata", {}) | {"code_bits": layer["code_bits"]},
            )
        else:
            quantized = Q(tensors[f"{name}.reconstruction"], bits=layer["bits"])
        converted.quantized[name] = quantized
        state[weight_name(name)] = quantized.to("cuda").w.detach().cpu()
    for descriptor in manifest.get("transforms", []):
        install_transform(model, descriptor, state)
        record_transform(model, descriptor)
    for name, value in state.items():
        owner, _, field = name.rpartition(".")
        if field != "bias":
            continue
        module = model.get_submodule(owner)
        if isinstance(module, nn.Linear) and module.bias is None:
            if tuple(value.shape) != (module.out_features,):
                raise ValueError(f"checkpoint bias shape does not match linear layer {owner!r}")
            module.bias = nn.Parameter(value.to(module.weight.device, module.weight.dtype))
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    for name, specification in manifest.get("activations", {}).items():
        module = model.get_submodule(name)
        quantizer = restore_quantizer(specification)
        context = Ctx(name, module, device=converted.device)
        converted.activation_handles.append(install_activation_quantization(
            module, quantizer, context
        ))
        converted.activation_specs[name] = specification
    if manifest.get("kv") is not None:
        converted.kv_spec = manifest["kv"]
        converted.activation_handles.append(install_kv_quantization(
            model, restore_quantizer(converted.kv_spec)
        ))
    converted.layer_reports = [
        LayerReport(**(record | {"shape": tuple(record["shape"])}))
        for record in manifest["reports"]
    ]
    converted.pass_reports = [
        BlockPassReport(**record) for record in manifest.get("pass_reports", [])
    ]
    if manifest.get("parameter_accounting") is not None:
        converted._parameter_accounting = tuple(manifest["parameter_accounting"])
    return converted


def decode_parameter(tensors: dict[str, Tensor], descriptor: dict) -> Tensor:
    value = tensors[descriptor["tensor"]]
    if descriptor.get("encoding") == "e8m0":
        return torch.pow(2.0, value.float() - 127)
    if descriptor.get("encoding") == "packed":
        return unpack(value, descriptor["bits"], tuple(descriptor["shape"]))
    return value.float() if value.dtype in {
        torch.float16, torch.bfloat16, torch.float8_e4m3fn,
    } else value
