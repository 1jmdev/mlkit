"""Reproducible token-weighted causal perplexity."""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from mlkit.calibration.token_batches import DataSource, normalize_batches
from mlkit.calibration.token_batches import data as tokenize_data
from mlkit.models.model import Model
from mlkit.timing import synchronize


@dataclass
class PerplexityResult:
    perplexity: float
    tokens: int
    negative_log_likelihood: float
    seconds: float
    dataset: str
    sequence_length: int


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
