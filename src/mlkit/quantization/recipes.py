"""Layer selection and composable model conversion recipes."""

import fnmatch
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from mlkit.quantization.algorithms import awq, gptq
from mlkit.quantization.context import Ctx
from mlkit.quantization.formats import standard
from mlkit.quantization.protocol import Quantizer


@dataclass
class Recipe:
    """What to quantize and how.

    ``weights`` selects the format of linear layers inside the repeated blocks.
    ``head`` selects the format of linear layers outside them, such as the output
    head of a language model; it is ``None`` by default, which leaves them dense.
    Both accept a quantizer, a function of the layer context, or a mapping from
    layer name patterns to either.
    """

    weights: Any = None
    acts: Any = None
    kv: Any = None
    head: Any = None
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

    def select_head(self, name: str, ctx: Ctx) -> Any:
        if isinstance(self.head, Mapping):
            for pattern, quantization in self.head.items():
                if fnmatch.fnmatchcase(name, pattern):
                    return resolve_selector(quantization, ctx)
            return None
        return resolve_selector(self.head, ctx)


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


_PRESETS: dict[str, Callable[[], Any]] = {}

BUILTIN_PRESETS = (
    "rtn-int4-g128, gptq-int4-g128, awq-int4-g128, nf4-g64, mxfp4-g32, or rtn-w4a4"
)


def preset(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Register a recipe factory under a name accepted wherever a recipe is expected.

    The factory takes no arguments and returns a recipe or a weight quantizer.
    It is called each time the name is resolved, so every conversion receives a
    fresh recipe.
    """
    def register(factory: Callable[[], Any]) -> Callable[[], Any]:
        if not name or name in _PRESETS or builtin_preset(name) is not None:
            raise ValueError(f"preset name {name!r} is empty or already defined")
        _PRESETS[name] = factory
        return factory

    return register


def builtin_preset(name: str) -> Recipe | None:
    match = re.fullmatch(r"(rtn|gptq|awq)-int([2-8])-g([1-9][0-9]*)", name)
    if match:
        method, bits, group = match.groups()
        quantization: Quantizer = standard.int(int(bits), group=int(group))
        if method == "gptq":
            quantization = gptq(quantization, refit=int(group))
        elif method == "awq":
            quantization = awq(quantization)
        return Recipe(weights=quantization, name=name)
    match = re.fullmatch(r"nf4-g([1-9][0-9]*)", name)
    if match:
        return Recipe(weights=standard.nf4(group=int(match[1])), name=name)
    if name == "mxfp4-g32":
        return Recipe(weights=standard.mxfp4(), name=name)
    if name == "rtn-w4a4":
        return Recipe(weights=standard.int(4), acts=standard.int(4, group=None), name=name)
    return None


def resolve_preset(name: str) -> Recipe:
    """The recipe of a registered or built-in preset name."""
    factory = _PRESETS.get(name)
    if factory is not None:
        definition = normalize_recipe(factory())
        return definition if definition.name is not None else definition.replace(name=name)
    builtin = builtin_preset(name)
    if builtin is None:
        registered = "".join(f", {registered_name}" for registered_name in sorted(_PRESETS))
        raise ValueError(f"unknown preset {name!r}; expected {BUILTIN_PRESETS}{registered}")
    return builtin


def normalize_recipe(value: Any) -> Recipe:
    if isinstance(value, Recipe):
        return value
    if isinstance(value, str):
        return resolve_preset(value)
    return Recipe(weights=value)
