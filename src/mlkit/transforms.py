"""Function-preserving normalization fusion, rotations, and input balancing."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from mlkit.context import layer_seed
from mlkit.data import DataSource, data, normalize_batches
from mlkit.engine import forward_batch
from mlkit.models import Model
from mlkit.operations import rht


def normalization_groups(model: Model) -> list[tuple[nn.Module, list[nn.Linear]]]:
    groups: list[tuple[nn.Module, list[nn.Linear]]] = []
    for block in model.blocks:
        for norm_name, projection_names in [
            ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
            ("pre_feedforward_layernorm" if hasattr(block, "pre_feedforward_layernorm")
             else "post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
        ]:
            try:
                normalization = block.get_submodule(norm_name)
                readers = [block.get_submodule(name) for name in projection_names]
            except AttributeError:
                continue
            if all(isinstance(reader, nn.Linear) for reader in readers):
                groups.append((normalization, [cast(nn.Linear, reader) for reader in readers]))
    try:
        normalization = model.module.get_submodule("model.norm")
        head = model.module.get_submodule("lm_head")
    except AttributeError:
        return groups
    if isinstance(head, nn.Linear):
        groups.append((normalization, [head]))
    return groups


def untie_output_embeddings(model: Model) -> None:
    if not hasattr(model.module, "get_input_embeddings"):
        return
    architecture = cast(Any, model.module)
    embedding = architecture.get_input_embeddings()
    output = architecture.get_output_embeddings()
    if output is not None and embedding.weight is output.weight:
        output.weight = nn.Parameter(output.weight.detach().clone(),
                                     requires_grad=output.weight.requires_grad)
        model.config.tie_word_embeddings = False


@dataclass
class FuseNorms:
    def __call__(self, model: Model, calib: Any = None) -> None:
        groups = normalization_groups(model)
        if not groups:
            raise ValueError(
                "normalization fusion requires an adapter with recognized norm readers"
            )
        untie_output_embeddings(model)
        with torch.no_grad():
            for normalization, readers in groups:
                parameter = cast(Tensor, normalization.weight)
                weight = parameter.detach().float()
                offset = 1 if type(normalization).__name__.startswith("Gemma") else 0
                gain = weight + offset
                bias = getattr(normalization, "bias", None)
                for reader in readers:
                    original_weight = reader.weight.detach().float()
                    if bias is not None:
                        contribution = original_weight @ bias.float()
                        if reader.bias is None:
                            reader.bias = nn.Parameter(contribution.to(reader.weight.dtype))
                        else:
                            reader.bias.add_(contribution.to(reader.bias.dtype))
                    reader.weight.copy_((original_weight * gain).to(reader.weight.dtype))
                parameter.fill_(1 - offset)
                if bias is not None:
                    bias.zero_()


@dataclass
class Rotate:
    method: str = "hadamard"
    seed: int = 0
    online: bool = True

    def __call__(self, model: Model, calib: Any = None) -> None:
        if self.method != "hadamard":
            raise ValueError("rotate supports the hadamard structured orthogonal basis")
        if any(isinstance(normalization, nn.LayerNorm) for normalization in model.norms):
            raise ValueError("residual rotations currently require RMS normalization")
        grouped = {id(normalization) for normalization, _ in normalization_groups(model)}
        if any(id(normalization) not in grouped for normalization in model.norms):
            raise ValueError("architecture contains normalization sites without fusion readers")
        FuseNorms()(model, calib)
        embedding = cast(Any, model.module).get_input_embeddings()
        width = embedding.weight.shape[1]
        with torch.no_grad():
            embedding.weight.copy_(
                rht(embedding.weight.float(), seed=self.seed).to(embedding.weight.dtype)
            )
            for reader in model.residual_readers:
                reader.weight.copy_(
                    rht(reader.weight.float(), seed=self.seed).to(reader.weight.dtype)
                )
            for writer in model.residual_writers:
                writer.weight.copy_(
                    rht(writer.weight.float().T, seed=self.seed).T.to(writer.weight.dtype)
                )
                if writer.bias is not None:
                    writer.bias.copy_(
                        rht(writer.bias.float(), seed=self.seed).to(writer.bias.dtype)
                    )
            if self.online:
                for name, module in model.module.named_modules():
                    if (not isinstance(module, nn.Linear)
                            or name.split(".")[-1] not in {"o_proj", "down_proj"}):
                        continue
                    seed = layer_seed(name, self.seed)
                    module.weight.copy_(
                        rht(module.weight.float(), seed=seed).to(module.weight.dtype)
                    )
                    descriptor = {"kind": "rotation", "name": name, "seed": seed}
                    install_transform(model.module, descriptor)
                    record_transform(model.module, descriptor)
        model.module.__dict__["_mlkit_residual_rotation"] = {"width": width, "seed": self.seed}


@dataclass
class Smooth:
    alpha: float = 0.5

    def __call__(self, model: Model, calib: Any) -> None:
        if not 0 <= self.alpha <= 1:
            raise ValueError("smoothing alpha must be between zero and one")
        if isinstance(calib, str):
            calib = data(calib, tokenizer=model.tokenizer)
        if isinstance(calib, DataSource):
            calib = calib.bind(model.tokenizer)
        if calib is None:
            raise ValueError("smoothing requires calibration data")
        observed: dict[str, Tensor] = {}
        handles = []
        selected = [(name, module) for name, module in model.module.named_modules()
                    if isinstance(module, nn.Linear) and name.split(".")[-1] != "lm_head"]
        for name, module in selected:
            def observe(layer: nn.Module, arguments: tuple, name: str = name) -> None:
                inputs = arguments[0].detach().reshape(-1, arguments[0].shape[-1]).float()
                maximum = inputs.abs().amax(0)
                observed[name] = (
                    maximum if name not in observed else torch.maximum(observed[name], maximum)
                )

            handles.append(module.register_forward_pre_hook(observe))
        try:
            with torch.no_grad():
                for batch in normalize_batches(calib):
                    forward_batch(model.module, batch)
        finally:
            for handle in handles:
                handle.remove()
        with torch.no_grad():
            for name, module in selected:
                if name not in observed:
                    raise ValueError(f"calibration did not reach smoothing layer {name!r}")
                activation_maximum = observed[name].clamp_min(1e-5)
                weight_maximum = module.weight.float().abs().amax(0).clamp_min(1e-5)
                scales = (activation_maximum.pow(self.alpha)
                          / weight_maximum.pow(1 - self.alpha)).clamp(1e-5, 1e5)
                module.weight.copy_((module.weight.float() * scales).to(module.weight.dtype))
                module.register_buffer("_mlkit_input_scale", scales)
                descriptor = {"kind": "smooth", "name": name}
                install_transform(model.module, descriptor)
                record_transform(model.module, descriptor)


def record_transform(model: nn.Module, descriptor: dict[str, Any]) -> None:
    descriptors = getattr(model, "_mlkit_transforms", [])
    descriptors.append(descriptor)
    model.__dict__["_mlkit_transforms"] = descriptors


def install_transform(
    model: nn.Module, descriptor: dict[str, Any], state: dict[str, Tensor] | None = None,
) -> None:
    module = model.get_submodule(descriptor["name"])
    if descriptor["kind"] == "rotation":
        seed = descriptor["seed"]

        def rotate_inputs(layer: nn.Module, arguments: tuple) -> tuple:
            inputs = arguments[0]
            return (rht(inputs.float(), seed=seed).to(inputs.dtype), *arguments[1:])

        module.register_forward_pre_hook(rotate_inputs)
    elif descriptor["kind"] == "smooth":
        if state is not None:
            module.register_buffer(
                "_mlkit_input_scale", state[f"{descriptor['name']}._mlkit_input_scale"]
            )

        def balance_inputs(layer: nn.Module, arguments: tuple) -> tuple:
            inputs = arguments[0]
            return ((inputs.float() / layer._mlkit_input_scale).to(inputs.dtype), *arguments[1:])

        module.register_forward_pre_hook(balance_inputs)
    else:
        raise ValueError(f"unknown model transform {descriptor['kind']!r}")


def fuse_norms() -> FuseNorms:
    return FuseNorms()


def rotate(method: str = "hadamard", *, seed: int = 0, online: bool = True) -> Rotate:
    return Rotate(method, seed, online)


def smooth(alpha: float = 0.5) -> Smooth:
    return Smooth(alpha)
