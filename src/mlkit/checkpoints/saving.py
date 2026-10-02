"""Writing of tensor-only, portable checkpoints with a JSON manifest."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor

from mlkit.checkpoints.manifest import FORMAT_VERSION, MANIFEST_FILE, WEIGHTS_FILE
from mlkit.models.model import QModel
from mlkit.models.module_utilities import weight_name
from mlkit.quantization.codecs import registered
from mlkit.quantization.operations import pack


def save(model: QModel, path: str | Path, *, overwrite: bool = False) -> None:
    destination = Path(path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    if any(specification is None for specification in model.activation_specs.values()):
        raise ValueError(
            "custom online activation quantizers require an explicit deployment recipe"
        )
    if getattr(model.module, "_mlkit_kv_quantizer", None) is not None and model.kv_spec is None:
        raise ValueError("custom online KV quantizers require an explicit deployment recipe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="mlkit_checkpoint_", dir=destination.parent))
    try:
        tensors: dict[str, Tensor] = {}
        parameter_fingerprints: dict[tuple, str] = {}
        layers = {}
        for name, quantized in model.quantized.items():
            if registered(quantized.codec) and quantized.codes is not None:
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
                        stored_value = value.detach().cpu().contiguous()
                        fingerprint = (
                            str(stored_value.dtype),
                            tuple(stored_value.shape),
                            hashlib.sha256(
                                stored_value.reshape(-1).view(torch.uint8).numpy().tobytes()
                            ).digest(),
                        )
                        canonical_identifier = parameter_fingerprints.get(fingerprint)
                        if canonical_identifier is None:
                            tensors[identifier] = stored_value.clone()
                            parameter_fingerprints[fingerprint] = identifier
                        else:
                            descriptor["tensor"] = canonical_identifier
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
        omitted = {weight_name(name) for name in model.quantized} | set(model.tied_weights)
        state_storages: dict[tuple, str] = {}
        state_aliases = {}
        for name, value in model.module.state_dict().items():
            owner, _, field = name.rpartition(".")
            if owner in model.quantized and field in {"packed", "scales", "values", "zeros"}:
                continue
            if name not in omitted:
                identity = (
                    value.device, value.data_ptr(), value.dtype,
                    tuple(value.shape), tuple(value.stride()),
                )
                canonical_name = state_storages.get(identity)
                if canonical_name is None:
                    tensors[f"state.{name}"] = value.detach().cpu().contiguous().clone()
                    state_storages[identity] = name
                else:
                    state_aliases[name] = canonical_name
        save_file(tensors, str(temporary / WEIGHTS_FILE))
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
            "pass_reports": [asdict(record) for record in model.pass_reports],
            "logical_bpw": model.bpw, "model_bpw": model.model_bpw,
            "parameter_accounting": model._parameter_accounting,
            "transforms": getattr(model.module, "_mlkit_transforms", []),
            "activations": model.activation_specs, "kv": model.kv_spec,
            "state_aliases": state_aliases,
            "tied_weights": model.tied_weights,
            "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
        }
        (temporary / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
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
