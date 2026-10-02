"""Small architectures shared by conversion, calibration and evaluation tests."""

from types import SimpleNamespace

import torch
from torch import nn


class CausalModel(nn.Module):
    """A token embedding followed by an output projection, returning ``logits``."""

    def __init__(self, vocabulary: int = 19) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 8)
        self.projection = nn.Linear(8, vocabulary)

    def forward(self, input_ids, use_cache=False, **kwargs):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


class RepeatedModel(nn.Module):
    """Three sequential blocks under ``model.layers`` followed by an output head."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
            for _ in range(3)
        ])
        self.lm_head = nn.Linear(16, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            inputs = block(inputs)
        return self.lm_head(inputs)


class SharedProjectionModel(nn.Module):
    """Attention projections that consume the same input, as recognized sibling layers."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(16, 16, bias=False)
        self.self_attn.k_proj = nn.Linear(16, 16, bias=False)
        self.self_attn.v_proj = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (
            self.self_attn.q_proj(inputs)
            + self.self_attn.k_proj(inputs)
            + self.self_attn.v_proj(inputs)
        )


def create_tiny_llama(*, tie_word_embeddings: bool = False):
    """A two-block Llama with grouped key/value heads, created without downloads."""
    import transformers

    configuration = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=tie_word_embeddings,
    )
    return transformers.LlamaForCausalLM(configuration)
