from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import mlkit as mk
from mlkit.evaluation import perplexity

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


class CausalModel(nn.Module):
    def __init__(self, vocabulary: int = 19) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 8)
        self.projection = nn.Linear(8, vocabulary)

    def forward(self, input_ids, use_cache=False, **kwargs):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def test_perplexity_matches_cross_entropy() -> None:
    model = CausalModel()
    tokens = torch.randint(19, (2, 17))
    logits = model(tokens).logits[:, :-1]
    expected = functional.cross_entropy(logits.reshape(-1, 19), tokens[:, 1:].reshape(-1)).exp()
    score = mk.ppl(model, data=tokens, budget="full", return_details=True)
    assert abs(score.perplexity - float(expected.detach())) < 1e-5
    assert score.tokens == 32
    assert score.sequence_length == 17
    assert model.training


def test_perplexity_budget_and_padding() -> None:
    model = CausalModel()
    tokens = torch.randint(19, (2, 8))
    mask = torch.ones_like(tokens)
    mask[1, 4:] = 0
    score = mk.ppl(model, data={"input_ids": tokens, "attention_mask": mask}, return_details=True)
    assert score.tokens == 10
    score = mk.ppl(model, data=tokens, max_tokens=3, return_details=True)
    assert score.tokens == 3


def test_comparison_table_and_sweep(tmp_path) -> None:
    model = CausalModel()
    tokens = torch.randint(19, (2, 12))
    table = mk.compare(model, [mk.int(4, group=4)], data=tokens, calib=None, print_table=False)
    assert len(table) == 2
    assert table[0]["method"] == "baseline"
    assert table[1]["bpw"] == 8
    table.to_csv(tmp_path / "comparison.csv")
    assert "perplexity" in (tmp_path / "comparison.csv").read_text()
    table = mk.sweep(model, lambda bits: mk.int(bits, group=4), bits=[2, 3],
                     options={"data": tokens, "calib": None, "print_table": False})
    assert len(table) == 3


def test_full_c4_evaluation_uses_a_bounded_protocol(monkeypatch) -> None:
    requested_windows = []

    def tokenize(name, n, **options):
        requested_windows.append((name, n))
        return mk.TokenBatches([{"input_ids": torch.randint(19, (1, 8))}], name=name)

    monkeypatch.setattr(perplexity, "tokenize_data", tokenize)
    score = mk.ppl(CausalModel(), data="c4", budget="full", return_details=True)
    assert requested_windows == [("c4", 256)]
    assert score.dataset == "c4"
    assert score.sequence_length == 8


def test_perplexity_preserves_the_token_batch_dataset_name() -> None:
    batches = mk.TokenBatches([{"input_ids": torch.randint(19, (1, 8))}], name="validation")
    score = mk.ppl(CausalModel(), data=batches, return_details=True)
    assert score.dataset == "validation"
