"""Comparison of recipes against a baseline and Cartesian parameter sweeps."""

import gc
import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import nn

from mlkit.conversion.quantize import quantize
from mlkit.evaluation.perplexity import ppl
from mlkit.evaluation.tables import Table
from mlkit.models.model import Model
from mlkit.quantization.recipes import normalize_recipe
from mlkit.timing import synchronize


def baseline_bpw(model: Model) -> float:
    parameters = list(model.module.parameters())
    return sum(value.numel() * value.element_size() * 8 for value in parameters) / sum(
        value.numel() for value in parameters
    )


def compare(
    model: Model | nn.Module,
    recipes: Sequence[Any],
    data: Any = "wikitext2",
    *,
    calib: Any = "c4",
    budget: str = "fast",
    seq: int = 2048,
    print_table: bool = True,
    quantize_options: Mapping[str, Any] | None = None,
    **evaluation_options: Any,
) -> Table:
    wrapped = model if isinstance(model, Model) else Model(model)
    baseline = ppl(wrapped, data, budget=budget, seq=seq, **evaluation_options)
    if not isinstance(baseline, float):
        raise ValueError("compare expects one evaluation dataset and scalar perplexity")
    records = [{"method": "baseline", "bpw": baseline_bpw(wrapped),
                "perplexity": baseline, "delta": 0.0, "seconds": 0.0}]
    for recipe in recipes:
        synchronize(wrapped.device)
        start = time.perf_counter()
        converted = quantize(wrapped, recipe, calib=calib, **dict(quantize_options or {}))
        synchronize(wrapped.device)
        duration = time.perf_counter() - start
        score = ppl(converted, data, budget=budget, seq=seq, **evaluation_options)
        definition = normalize_recipe(recipe)
        records.append({
            "method": definition.name or repr(definition.weights), "bpw": converted.bpw,
            "perplexity": score, "delta": score - baseline, "seconds": duration,
        })
        del converted
        gc.collect()
    result = Table(records)
    if print_table:
        print(result)
    return result


def sweep(
    model: Model | nn.Module,
    factory: Callable,
    *,
    options: Mapping[str, Any] | None = None,
    **parameters: Sequence[Any],
) -> Table:
    if not parameters:
        raise ValueError("sweep requires at least one parameter grid")
    names = list(parameters)
    recipes = [
        factory(**dict(zip(names, values, strict=True)))
        for values in itertools.product(*(parameters[name] for name in names))
    ]
    return compare(model, recipes, **dict(options or {}))
