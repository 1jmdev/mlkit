import pytest
import torch
from reference_models import create_tiny_llama

import mlkit as mk

pytest.importorskip("transformers")


def test_cache_quantizer_observes_post_rope_keys_and_values() -> None:
    observations = []

    @mk.quantizer
    def record(values, context):
        observations.append((context.name, tuple(values.shape)))
        return mk.Q(values, bits=16 * values.numel())

    model = create_tiny_llama()
    converted = mk.quantize(model, mk.Recipe(weights=lambda w, ctx: w, kv=record), calib=None)
    tokens = torch.randint(128, (1, 5))
    with torch.inference_mode():
        expected = model(tokens, use_cache=False).logits
        result = converted(tokens, use_cache=True)
    torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
    assert len(observations) == 4
    assert all(shape == (10, 8) for _, shape in observations)
    with torch.inference_mode():
        continued = converted(tokens[:, :1], past_key_values=result.past_key_values, use_cache=True)
    assert continued.past_key_values.get_seq_length() == 6
    assert all(bits > 0 for bits in continued.past_key_values.logical_bits.values())


def test_cache_quantization_runs_during_perplexity_and_generation() -> None:
    model = create_tiny_llama()
    converted = mk.quantize(
        model,
        mk.Recipe(weights=mk.int(4, group=16), kv=mk.int(4, group=None)),
        calib=None,
    )
    tokens = torch.randint(128, (1, 7))
    assert mk.ppl(converted, data=tokens) > 0
    generated = converted.generate(tokens, max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert generated.shape[1] >= tokens.shape[1] + 1


def test_online_scalar_formats_are_saved_and_restored(tmp_path) -> None:
    model = create_tiny_llama()
    recipe = mk.Recipe(
        weights=mk.int(4, group=16),
        acts=mk.nf4(group=None),
        kv=mk.int(4, group=None),
    )
    converted = mk.quantize(model, recipe, calib=None)
    tokens = torch.randint(128, (1, 7))
    with torch.inference_mode():
        expected = converted(tokens, use_cache=False).logits
    converted.save(tmp_path / "online")
    restored = mk.load(tmp_path / "online", model=create_tiny_llama)
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(tokens, use_cache=False).logits, expected, rtol=0, atol=0
        )
