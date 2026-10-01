"""Reproducible perplexity, layer probes, and experiment tables."""

import csv
import fnmatch
import gc
import itertools
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mlkit.experiments.data import DataSource, normalize_batches
from mlkit.experiments.data import data as tokenize_data
from mlkit.quantization.context import Ctx
from mlkit.quantization.operations import proxy_loss
from mlkit.quantization.recipes import normalize_recipe
from mlkit.quantization.representation import as_q
from mlkit.runtime.engine import BlockStatistics, CalibrationSession, quantize, synchronize
from mlkit.runtime.models import Model


@dataclass
class PerplexityResult:
    perplexity: float
    tokens: int
    negative_log_likelihood: float
    seconds: float
    dataset: str
    sequence_length: int


class Table(Sequence[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __getitem__(self, index: Any) -> Any:
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def __str__(self) -> str:
        if not self.records:
            return ""
        columns = list(self.records[0])
        values = [
            [format_value(record.get(column)) for column in columns] for record in self.records
        ]
        widths = [
            max(len(column), *(len(row[index]) for row in values))
            for index, column in enumerate(columns)
        ]
        return "\n".join(
            "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
            for row in [columns, *values]
        )

    def to_csv(self, path: str | Path) -> None:
        if not self.records:
            raise ValueError("cannot export an empty comparison table")
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)


def format_value(value: Any) -> str:
    if value is None:
        return "?"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def ppl(
    model: Model | nn.Module,
    data: Any = "wikitext2",
    budget: str = "fast",
    seq: int = 2048,
    *,
    max_tokens: int | None = None,
    seed: int = 0,
    return_details: bool = False,
) -> Any:
    """Token-weighted causal perplexity on contiguous, nonoverlapping windows.

    The first token in each window has no preceding context and is excluded.
    Padding and labels of -100 are excluded. Custom data is never silently
    replaced with a downloaded dataset.
    """
    if budget not in {"fast", "full"} or seq < 2:
        raise ValueError("budget must be fast or full and seq must be at least two")
    wrapped = model if isinstance(model, Model) else Model(model)
    if isinstance(data, (list, tuple)) and data and all(isinstance(item, str) for item in data):
        return {
            name: ppl(wrapped, name, budget, seq, max_tokens=max_tokens,
                      seed=seed, return_details=return_details)
            for name in data
        }
    token_limit = max_tokens if max_tokens is not None else (20_000 if budget == "fast" else None)
    if token_limit is not None and token_limit < 1:
        raise ValueError("max_tokens must be positive")
    dataset_name = data if isinstance(data, str) else getattr(data, "name", "tokens")
    if not isinstance(dataset_name, str):
        dataset_name = "text"
    if isinstance(data, str):
        windows = None if token_limit is None else math.ceil(token_limit / (seq - 1))
        if data == "c4" and windows is None:
            windows = 256
        data = tokenize_data(
            data, n=windows, seq=seq, split="test", tokenizer=wrapped.tokenizer, seed=seed
        )
    if isinstance(data, Tensor):
        data = {"input_ids": data.unsqueeze(0) if data.ndim == 1 else data}
    if isinstance(data, DataSource):
        data = data.bind(wrapped.tokenizer)
    batches = normalize_batches(data)
    training_states = [(module, module.training) for module in wrapped.modules()]
    wrapped.eval()
    negative_log_likelihood = torch.zeros((), dtype=torch.float64, device=wrapped.device)
    count = 0
    maximum_sequence_length = 0
    synchronize(wrapped.device)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch in batches:
                if not isinstance(batch, Mapping) or "input_ids" not in batch:
                    raise ValueError("perplexity data must contain input_ids tensors")
                maximum_sequence_length = max(
                    maximum_sequence_length, batch["input_ids"].shape[-1]
                )
                inputs = {name: value.to(wrapped.device) for name, value in batch.items()
                          if name in {"input_ids", "attention_mask", "position_ids"}}
                labels = batch.get("labels", batch["input_ids"]).to(wrapped.device).clone()
                if "attention_mask" in inputs:
                    labels[inputs["attention_mask"] == 0] = -100
                output = wrapped(**inputs, use_cache=False)
                logits = output.logits if hasattr(output, "logits") else output[0]
                shifted_labels = labels[:, 1:].reshape(-1)
                shifted_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                valid_indices = (shifted_labels != -100).nonzero().flatten()
                if token_limit is not None:
                    valid_indices = valid_indices[: token_limit - count]
                for indices in valid_indices.split(128):
                    loss = functional.cross_entropy(
                        shifted_logits[indices].float(), shifted_labels[indices], reduction="sum"
                    )
                    negative_log_likelihood += loss.double()
                count += valid_indices.numel()
                if token_limit is not None and count >= token_limit:
                    break
    finally:
        for module, training in training_states:
            module.training = training
    synchronize(wrapped.device)
    if count == 0:
        raise ValueError("evaluation contained no valid next-token targets")
    total_loss = float(negative_log_likelihood)
    perplexity = math.exp(total_loss / count) if total_loss / count < 709 else float("inf")
    result = PerplexityResult(
        perplexity, count, total_loss, time.perf_counter() - start,
        dataset_name, maximum_sequence_length,
    )
    return result if return_details else result.perplexity


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


def probe(
    quantizers: Sequence[Callable],
    model: Model | nn.Module,
    layers: str = "*.mlp.down_proj",
    *,
    calib: Any = "c4",
    cache_dir: str | Path | None = "~/.cache/mlkit/statistics",
    seed: int = 0,
    print_table: bool = True,
) -> Table:
    wrapped = model if isinstance(model, Model) else Model(model)
    session = CalibrationSession(
        wrapped, calib, sequential=False, sample_rows=4096,
        cache_dir=None if cache_dir is None else Path(cache_dir).expanduser(), need_targets=False,
    )
    records = []
    for index, block in enumerate(wrapped.blocks):
        prefix = wrapped.architecture.block_name(index)
        statistics = BlockStatistics(session, block, index, prefix)
        for relative_name, module in block.named_modules():
            name = f"{prefix}.{relative_name}".strip(".")
            if not isinstance(module, nn.Linear) or not fnmatch.fnmatchcase(name, layers):
                continue
            weight = module.weight.detach().float()
            for algorithm in quantizers:
                context = Ctx(name, module, block, index, seed=seed, device=weight.device,
                              provider=statistics.provider(name, weight.device))
                synchronize(weight.device)
                start = time.perf_counter()
                with torch.no_grad():
                    quantized = as_q(algorithm(weight, context))
                    loss = float(proxy_loss(weight, quantized.w, context))
                synchronize(weight.device)
                bits = None if quantized.bits is None else quantized.bits + context._additional_bits
                records.append({"layer": name, "method": repr(algorithm),
                                "bpw": None if bits is None else bits / weight.numel(),
                                "loss": loss, "seconds": time.perf_counter() - start})
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


def eval(
    model: Model,
    tasks: Sequence[str],
    *,
    batch_size: int | str = 1,
    max_length: int | None = None,
    **options: Any,
) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError("task evaluation requires uv add 'mlkit[evaluation]'") from error
    language_model = HFLM(
        pretrained=model.module, tokenizer=model.tokenizer,
        batch_size=batch_size, max_length=max_length,
    )
    result = simple_evaluate(model=language_model, tasks=list(tasks), **options)
    if result is None:
        raise RuntimeError("task evaluation returned no results")
    return result
