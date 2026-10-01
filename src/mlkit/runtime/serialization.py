"""Versioned JSON manifests and tensor-only, portable checkpoints."""

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from mlkit.quantization.formats import decode_feedback, decode_scaled
from mlkit.quantization.packing import pack, unpack
from mlkit.quantization.representation import Q
from mlkit.quantization.trellis import decode_trellis
from mlkit.runtime.models import LayerReport, QModel

_CODECS: dict[str, Callable[..., Tensor]] = {
    "scaled": decode_scaled, "feedback": decode_feedback, "trellis": decode_trellis,
}
FORMAT_VERSION = 1


def codec(name: str) -> Callable[[Callable], Callable]:
    """Register an explicit decoder; checkpoints contain its name, never code."""
    def register(decoder: Callable[..., Tensor]) -> Callable:
        if not name or name in _CODECS:
            raise ValueError(f"codec name {name!r} is empty or already registered")
        _CODECS[name] = decoder
        return decoder

    return register


def save(model: QModel, path: str | Path, *, overwrite: bool = False) -> None:
    destination = Path(path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    if model.activation_handles:
        raise ValueError(
            "activation hooks need a serializable recipe; weight-only save is supported"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mlkit_checkpoint_", dir=destination.parent))
    try:
        tensors: dict[str, Tensor] = {}
        layers = {}
        for name, quantized in model.quantized.items():
            if quantized.codec in _CODECS and quantized.codes is not None:
                precision = quantized.metadata.get("code_bits")
                if precision is None:
                    precision = max(1, int(quantized.codes.max()).bit_length())
                tensors[f"{name}.codes"] = pack(quantized.codes.cpu(), precision)
                parameters = {}
                for parameter_name, value in quantized.params.items():
                    if isinstance(value, Tensor):
                        identifier = f"{name}.params.{parameter_name}"
                        encoding = None
                        parameter_bits = quantized.metadata.get("parameter_bits", {}).get(
                            parameter_name
                        )
                        format = quantized.metadata.get("parameter_formats", {}).get(parameter_name)
                        descriptor: dict[str, Any] = {"tensor": identifier, "encoding": None}
                        if parameter_bits is not None:
                            descriptor = {"tensor": identifier, "encoding": "packed",
                                          "bits": parameter_bits, "shape": list(value.shape)}
                            value = pack(value, parameter_bits)
                        else:
                            if parameter_name in {"scales", "zero"}:
                                format = quantized.metadata.get("scale_fmt", "fp32")
                            if format == "fp16":
                                value = value.half()
                            elif format == "bf16":
                                value = value.bfloat16()
                            elif format == "fp8":
                                value = value.to(torch.float8_e4m3fn)
                            elif format == "e8m0":
                                value = (value.float().log2().round() + 127).to(torch.uint8)
                                encoding = "e8m0"
                        tensors[identifier] = value.detach().cpu().contiguous().clone()
                        if encoding is not None:
                            descriptor["encoding"] = encoding
                        parameters[parameter_name] = descriptor
                    else:
                        parameters[parameter_name] = {"value": value}
                layers[name] = {
                    "codec": quantized.codec, "shape": list(quantized.codes.shape),
                    "code_bits": precision, "bits": quantized.bits, "params": parameters,
                    "metadata": {
                        key: value for key, value in quantized.metadata.items()
                        if isinstance(value, (str, int, float, bool, list, dict, type(None)))
                    },
                }
            else:
                reconstruction = quantized.w.detach().cpu()
                tensors[f"{name}.reconstruction"] = reconstruction.contiguous()
                layers[name] = {"codec": None, "bits": quantized.bits}
        omitted = {f"{name}.weight" for name in model.quantized}
        for name, value in model.module.state_dict().items():
            owner, _, field = name.rpartition(".")
            if owner in model.quantized and field in {"packed", "scales", "values", "zeros"}:
                continue
            if name not in omitted:
                tensors[f"state.{name}"] = value.detach().cpu().contiguous().clone()
        save_file(tensors, str(temporary / "weights.safetensors"))
        configuration = getattr(model.module, "config", None)
        if configuration is not None:
            configuration.save_pretrained(temporary)
        if model.tokenizer is not None:
            model.tokenizer.save_pretrained(temporary)
        manifest = {
            "format_version": FORMAT_VERSION, "name": model.name,
            "architecture": "transformers" if configuration is not None else "external",
            "dtype": str(model.dtype).removeprefix("torch."),
            "layers": layers, "reports": [asdict(record) for record in model.layer_reports],
            "logical_bpw": model.bpw, "model_bpw": model.model_bpw,
            "parameter_accounting": model._parameter_accounting,
            "transforms": getattr(model.module, "_mlkit_transforms", []),
            "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
        }
        (temporary / "mlkit.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            backup = destination.with_name(f"{destination.name}.previous")
            if backup.exists():
                raise FileExistsError(f"checkpoint backup already exists: {backup}")
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | Callable[[], nn.Module] | None = None,
) -> QModel:
    directory = Path(path)
    manifest = json.loads((directory / "mlkit.json").read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported mlkit checkpoint format version")
    tokenizer = None
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
        model = AutoModelForCausalLM.from_config(configuration, trust_remote_code=False)
        if (directory / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
    elif not isinstance(model, nn.Module):
        model = model()
    assert isinstance(model, nn.Module)
    tensors = load_file(str(directory / "weights.safetensors"))
    state = {name.removeprefix("state."): value for name, value in tensors.items()
             if name.startswith("state.")}
    converted = QModel(model, tokenizer, name=manifest["name"])
    for name, layer in manifest["layers"].items():
        if layer["codec"] is not None:
            if layer["codec"] not in _CODECS:
                raise ValueError(f"checkpoint requires registered codec {layer['codec']!r}")
            codes = unpack(tensors[f"{name}.codes"], layer["code_bits"], tuple(layer["shape"]))
            parameters = {
                parameter_name: (
                    decode_parameter(tensors, value) if "tensor" in value else value["value"]
                )
                for parameter_name, value in layer["params"].items()
            }
            quantized = Q(
                codes=codes, params=parameters, decode=_CODECS[layer["codec"]],
                codec=layer["codec"], bits=layer["bits"],
                metadata=layer.get("metadata", {}) | {"code_bits": layer["code_bits"]},
            )
        else:
            quantized = Q(tensors[f"{name}.reconstruction"], bits=layer["bits"])
        converted.quantized[name] = quantized
        state[f"{name}.weight"] = quantized.w
    model.to(dtype=getattr(torch, manifest["dtype"]))
    from mlkit.runtime.transforms import install_transform, record_transform

    for descriptor in manifest.get("transforms", []):
        install_transform(model, descriptor, state)
        record_transform(model, descriptor)
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    converted.layer_reports = [
        LayerReport(**(record | {"shape": tuple(record["shape"])}))
        for record in manifest["reports"]
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
    return value.float() if value.dtype == torch.float8_e4m3fn else value
