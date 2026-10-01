import pytest
import torch
from reference_models import create_tiny_llama

import mlkit as mk

pytest.importorskip("transformers")


def test_compiled_generation_uses_static_cache_without_mutating_source() -> None:
    model = mk.Model(create_tiny_llama())
    original_cache = model.module.generation_config.cache_implementation
    compiled = mk.optimize(model, backend="dense", compile=True)
    assert compiled.module.generation_config.cache_implementation == "static"
    assert model.module.generation_config.cache_implementation == original_cache
    tokens = torch.randint(128, (1, 7))
    options = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0}
    expected = model.generate(tokens, **options)
    result = compiled.generate(tokens, **options)
    assert torch.equal(result, expected)
