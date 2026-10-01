"""Layer selection and composable model conversion recipes."""

import fnmatch
import inspect
import re
from builtins import int as builtins_int
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mlkit.quantization.algorithms import awq, gptq
from mlkit.quantization.context import Ctx
from mlkit.quantization.formats import int, mxfp4, nf4
from mlkit.quantization.protocol import Quantizer


@dataclass
class Recipe:
    weights: Any = None
    acts: Any = None
    kv: Any = None
    transforms: Sequence[Callable] = field(default_factory=tuple)
    passes: Sequence[Callable] = field(default_factory=tuple)
    model_passes: Sequence[Callable] = field(default_factory=tuple)
    name: str | None = None

    def replace(self, **changes: Any) -> "Recipe":
        return replace(self, **changes)

    def select(self, name: str, ctx: Ctx) -> Any:
        if isinstance(self.weights, Mapping):
            for pattern, quantization in self.weights.items():
                if fnmatch.fnmatchcase(name, pattern):
                    return resolve_selector(quantization, ctx)
            return None
        if name.split(".")[-1] in {"lm_head", "embed_out", "output"}:
            return None
        return resolve_selector(self.weights, ctx)


def resolve_selector(value: Any, context: Ctx) -> Any:
    if value is None or isinstance(value, Quantizer):
        return value
    if callable(value):
        signature = inspect.signature(value)
        positional = [
            parameter for parameter in signature.parameters.values()
            if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
        ]
        if len(positional) == 1:
            return value(context)
    return value


def preset(name: str) -> Recipe:
    match = re.fullmatch(r"(rtn|gptq|awq)-int([2-8])-g([1-9][0-9]*)", name)
    if match:
        method, bits, group = match.groups()
        quantization: Quantizer = int(builtins_int(bits), group=builtins_int(group))
        if method == "gptq":
            quantization = gptq(quantization, refit=builtins_int(group))
        elif method == "awq":
            quantization = awq(quantization)
        return Recipe(weights=quantization, name=name)
    match = re.fullmatch(r"nf4-g([1-9][0-9]*)", name)
    if match:
        return Recipe(weights=nf4(group=builtins_int(match[1])), name=name)
    if name == "mxfp4-g32":
        return Recipe(weights=mxfp4(), name=name)
    if name == "rtn-w4a4":
        return Recipe(weights=int(4), acts=int(4, group=None), name=name)
    raise ValueError(
        f"unknown preset {name!r}; expected rtn-int4-g128, gptq-int4-g128, "
        "awq-int4-g128, nf4-g64, mxfp4-g32, or rtn-w4a4"
    )


def normalize_recipe(value: Any) -> Recipe:
    if isinstance(value, Recipe):
        return value
    if isinstance(value, str):
        return preset(value)
    return Recipe(weights=value)
