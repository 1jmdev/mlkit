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


def test_compiled_generation_reads_a_tied_embedding_from_the_packed_head() -> None:
    recipe = mk.Recipe(weights=mk.int(4, group=32), head=mk.int(8, group=32))
    converted = mk.quantize(create_tiny_llama(tie_word_embeddings=True), recipe, calib=None)
    compiled = mk.optimize(converted, backend="packed", compile=True)
    assert compiled.execution_backend == "packed:15+compiled"
    tokens = torch.randint(128, (1, 7))
    options = {
        "max_new_tokens": 3,
        "do_sample": False,
        "pad_token_id": 0,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    expected = converted.generate(tokens, **options)
    result = compiled.generate(tokens, **options)
    for actual_logits, expected_logits in zip(result.logits, expected.logits, strict=True):
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-3, atol=1e-4)
