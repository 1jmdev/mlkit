"""Reading of mlkit checkpoints into a quantized model on CUDA.

Tensors are read straight onto the GPU, codes are unpacked and decoded there,
and a Hugging Face architecture is constructed without initializing weights
that the checkpoint is about to replace.
"""

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

HALF_PRECISION_STORAGE = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | Callable[[], nn.Module] | None = None,
) -> QModel:
    directory = Path(path)
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported mlkit checkpoint format version")
    dtype = getattr(torch, manifest["dtype"])
    tokenizer = None
    uninitialized = False
    if model is None:
        if manifest["architecture"] != "transformers":
            raise ValueError(
                "external architecture checkpoints require model=module or model=factory"
            )
        module, tokenizer, uninitialized = construct_transformers_model(directory, dtype)
    else:
        module = model if isinstance(model, nn.Module) else model()
        module.to(dtype=dtype)
    tensors = load_file(str(directory / WEIGHTS_FILE), device="cuda")
    state = {
        name.removeprefix("state."): value
        for name, value in tensors.items()
        if name.startswith("state.")
    }
    for alias, canonical_name in manifest.get("state_aliases", {}).items():
        state[alias] = state[canonical_name]
    quantized_layers: dict[str, Q] = {}
    for name, layer in manifest["layers"].items():
        quantized = restore_layer(tensors, name, layer)
        with torch.no_grad():
            state[weight_name(name)] = quantized.w.to(dtype)
        quantized_layers[name] = quantized.to("cpu", detach=True)
    tied_weights = manifest.get("tied_weights", {})
    for parameter_name, layer_name in tied_weights.items():
        state[parameter_name] = state[weight_name(layer_name)]
    del tensors
    for descriptor in manifest.get("transforms", []):
        install_transform(module, descriptor, state)
        record_transform(module, descriptor)
    restore_missing_biases(module, state)
    module.load_state_dict(state, strict=True, assign=uninitialized)
    tie_weights = getattr(module, "tie_weights", None)
    if uninitialized and callable(tie_weights):
        tie_weights()
    del state
    module.cuda().eval()
    converted = QModel(module, tokenizer, name=manifest["name"])
    converted.quantized = quantized_layers
    converted.tied_weights = dict(tied_weights)
    for name, specification in manifest.get("activations", {}).items():
        target = module.get_submodule(name)
        quantizer = restore_quantizer(specification)
        context = Ctx(name, target, device=converted.device)
        converted.activation_handles.append(
            install_activation_quantization(target, quantizer, context)
        )
        converted.activation_specs[name] = specification
    if manifest.get("kv") is not None:
        converted.kv_spec = manifest["kv"]
        converted.activation_handles.append(
            install_kv_quantization(module, restore_quantizer(converted.kv_spec))
        )
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


def construct_transformers_model(
    directory: Path,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any, bool]:
    """Build the saved architecture; the flag reports parameters without storage."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("checkpoint loading requires 'mlkit[transformers]'") from error
    try:
        from accelerate import init_empty_weights
    except ImportError:
        init_empty_weights = None
    configuration = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    uninitialized = init_empty_weights is not None
    context = init_empty_weights() if uninitialized else contextlib.nullcontext()
    with context:
        module = AutoModelForCausalLM.from_config(
            configuration, trust_remote_code=False, dtype=dtype
        )
    tokenizer = None
    if (directory / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
    return module, tokenizer, uninitialized


def restore_layer(tensors: dict[str, Tensor], name: str, layer: dict[str, Any]) -> Q:
    """Rebuild the representation of one quantized layer on the device of its tensors."""
    if layer["codec"] is None:
        return Q(tensors[f"{name}.reconstruction"], bits=layer["bits"])
    codes = unpack(tensors[f"{name}.codes"], layer["code_bits"], tuple(layer["shape"]))
    parameters = {
        parameter_name: (
            decode_parameter(tensors, value) if "tensor" in value else value["value"]
        )
        for parameter_name, value in layer["params"].items()
    }
    return Q(
        codes=codes,
        params=parameters,
        decode=decoder(layer["codec"]),
        codec=layer["codec"],
        bits=layer["bits"],
        metadata=layer.get("metadata", {}) | {"code_bits": layer["code_bits"]},
    )


def restore_missing_biases(module: nn.Module, state: dict[str, Tensor]) -> None:
    """Create biases that normalization fusion introduced on projections without one."""
    for name, value in state.items():
        owner, _, field = name.rpartition(".")
        if field != "bias":
            continue
        target = module.get_submodule(owner)
        if isinstance(target, nn.Linear) and target.bias is None:
            if tuple(value.shape) != (target.out_features,):
                raise ValueError(f"checkpoint bias shape does not match linear layer {owner!r}")
            target.bias = nn.Parameter(value.detach().clone())


def decode_parameter(tensors: dict[str, Tensor], descriptor: dict) -> Tensor:
    value = tensors[descriptor["tensor"]]
    if descriptor.get("encoding") == "e8m0":
        return torch.pow(2.0, value.float() - 127)
    if descriptor.get("encoding") == "packed":
        return unpack(value, descriptor["bits"], tuple(descriptor["shape"]))
    return value.float() if value.dtype in HALF_PRECISION_STORAGE else value
