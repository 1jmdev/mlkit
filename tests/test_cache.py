import pytest
import torch

import mlkit as mk

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


def create_model():
    transformers = pytest.importorskip("transformers")
    configuration = transformers.LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    )
    return transformers.LlamaForCausalLM(configuration)


def test_cache_quantizer_observes_post_rope_keys_and_values() -> None:
    observations = []

    @mk.quantizer
    def record(values, context):
        observations.append((context.name, tuple(values.shape)))
        return mk.Q(values, bits=16 * values.numel())

    model = create_model()
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
    model = create_model()
    converted = mk.quantize(model, mk.Recipe(weights=mk.int(4, group=16), kv=mk.int(4, group=None)),
                            calib=None)
    tokens = torch.randint(128, (1, 7))
    assert mk.ppl(converted, data=tokens) > 0
    generated = converted.generate(tokens, max_new_tokens=3, do_sample=False, pad_token_id=0)
    assert generated.shape[1] >= tokens.shape[1] + 1


def test_compiled_generation_uses_static_cache_without_mutating_source() -> None:
    model = mk.Model(create_model())
    original_cache = model.module.generation_config.cache_implementation
    compiled = mk.optimize(model, backend="dense", compile=True)
    assert compiled.module.generation_config.cache_implementation == "static"
    assert model.module.generation_config.cache_implementation == original_cache
    tokens = torch.randint(128, (1, 7))
    options = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0}
    expected = model.generate(tokens, **options)
    result = compiled.generate(tokens, **options)
    assert torch.equal(result, expected)
