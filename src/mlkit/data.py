"""Seeded calibration and evaluation token sequences."""

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


class TokenBatches(Sequence[dict[str, Tensor]]):
    def __init__(self, batches: Iterable[dict[str, Tensor]], *, name: str = "tokens") -> None:
        self.batches = list(batches)
        self.name = name

    def __getitem__(self, index: Any) -> Any:
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        return iter(self.batches)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for batch in self:
            for name, tensor in sorted(batch.items()):
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class DataSource:
    name: str | Sequence[str]
    n: int | None = 128
    seq: int = 2048
    split: str = "train"
    seed: int = 0
    streaming: bool = True

    def bind(self, tokenizer: Any) -> TokenBatches:
        if tokenizer is None:
            raise ValueError("text calibration requires a model tokenizer or pretokenized batches")
        result = data(self.name, self.n, self.seq, self.split, tokenizer=tokenizer,
                      seed=self.seed, streaming=self.streaming)
        assert isinstance(result, TokenBatches)
        return result


def normalize_batches(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Tensor):
        if value.ndim == 1:
            return [{"input_ids": value.unsqueeze(0)}]
        return [value]
    return list(value)


def data(
    name: str | Sequence[str],
    n: int | None = 128,
    seq: int = 2048,
    split: str = "train",
    *,
    tokenizer: Any = None,
    seed: int = 0,
    streaming: bool = True,
) -> TokenBatches | DataSource:
    """Tokenize text using an explicit tokenizer and deterministic sampling.

    Evaluation splits use contiguous, nonoverlapping windows. Training uses
    seeded random windows. No tokenizer is guessed for model-independent data.
    """
    if seq < 2 or (n is not None and n < 1):
        raise ValueError("sequence length must be at least two and n must be positive or None")
    if tokenizer is None:
        return DataSource(name, n, seq, split, seed, streaming)
    texts: Iterable[str]
    if isinstance(name, str):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError("named datasets require uv add 'mlkit[datasets]'") from error
        if name == "wikitext2":
            corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif name == "c4":
            selected_split = "validation" if split == "test" else split
            corpus = load_dataset("allenai/c4", "en", split=selected_split, streaming=streaming)
        elif name == "redpajama":
            if split != "train":
                raise ValueError("redpajama calibration supports only the train split")
            corpus = load_dataset(
                "togethercomputer/RedPajama-Data-1T-Sample", split=split, streaming=streaming
            )
        else:
            raise ValueError("unknown dataset; use wikitext2, c4, redpajama, or a list of strings")
        texts = (record["text"] for record in corpus)
        dataset_name = name
        if n is None and name != "wikitext2" and streaming:
            raise ValueError("unbounded streaming evaluation requires an explicit n")
    else:
        texts = iter(name)
        dataset_name = "text"
    token_parts = []
    count = 0
    target = None if n is None else (n + 1) * seq
    for text in texts:
        if not text.strip():
            continue
        encoded = tokenizer(text + "\n\n", add_special_tokens=False, return_attention_mask=False)
        tokens = encoded["input_ids"]
        token_parts.extend(tokens)
        count += len(tokens)
        if target is not None and count >= target:
            break
    if count < seq:
        raise ValueError(f"corpus has only {count} tokens, fewer than sequence length {seq}")
    tokens = torch.tensor(token_parts, dtype=torch.long)
    number = count // seq if n is None else n
    starts: Iterable[int]
    if split == "train":
        generator = torch.Generator().manual_seed(seed)
        starts = torch.randint(count - seq + 1, (number,), generator=generator).tolist()
    else:
        number = min(number, count // seq)
        starts = range(0, number * seq, seq)
    return TokenBatches(
        ({"input_ids": tokens[start : start + seq].unsqueeze(0)} for start in starts),
        name=dataset_name,
    )
