import sys
from types import SimpleNamespace

import torch

import mlkit as mk


def test_wikitext_uses_one_contiguous_tokenization(monkeypatch) -> None:
    texts = ["", "Alpha", "", "Beta"]
    dataset = SimpleNamespace(load_dataset=lambda *arguments, **keywords: {"text": texts})
    monkeypatch.setitem(sys.modules, "datasets", dataset)
    observed = []

    def tokenize(text, **options):
        observed.append(text)
        return {"input_ids": [ord(character) for character in text]}

    batches = mk.data("wikitext2", n=None, seq=4, split="test", tokenizer=tokenize)
    expected = "\n\n".join(texts)
    assert observed == [expected]
    tokens = torch.cat([batch["input_ids"].flatten() for batch in batches])
    assert tokens.tolist() == [ord(character) for character in expected[: len(tokens)]]
