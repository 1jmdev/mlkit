import pytest
import torch
from torch import nn

import mlkit as mk


def identity_quantizer(w, ctx):
    return mk.Q(w, bits=32 * w.numel())


def test_input_smoothing_preserves_outputs_and_serializes(tmp_path) -> None:
    module = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    inputs = torch.randn(4, 16)
    definition = mk.Recipe(weights=identity_quantizer, transforms=[mk.smooth(0.5)])
    converted = mk.quantize(module, definition, calib=[inputs])
    torch.testing.assert_close(converted(inputs), module(inputs), rtol=1e-5, atol=1e-6)
    converted.save(tmp_path / "smoothed")
    restored = mk.load(
        tmp_path / "smoothed",
        model=lambda: nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8)),
    )
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=0, atol=0)


@pytest.mark.integration
def test_llama_norm_fusion_and_rotation_preserve_logits(tmp_path) -> None:
    model = mk.load("hf-internal-testing/tiny-random-LlamaForCausalLM")
    tokens = torch.randint(100, 2000, (1, 17))
    with torch.no_grad():
        expected = model(tokens).logits
    for transform in [mk.fuse_norms(), mk.rotate(seed=7)]:
        definition = mk.Recipe(weights=identity_quantizer, transforms=[transform])
        converted = mk.quantize(model, definition, calib=None)
        with torch.no_grad():
            torch.testing.assert_close(converted(tokens).logits, expected, rtol=1e-4, atol=1e-5)
        directory = tmp_path / type(transform).__name__
        converted.save(directory)
        restored = mk.load(directory)
        with torch.no_grad():
            torch.testing.assert_close(
                restored(tokens).logits, converted(tokens).logits, rtol=0, atol=0
            )


def test_norm_fusion_restores_new_projection_biases(tmp_path) -> None:
    transformers = pytest.importorskip("transformers")

    def create_model():
        configuration = transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        module = transformers.LlamaForCausalLM(configuration)
        normalization = nn.LayerNorm(32)
        with torch.no_grad():
            normalization.bias.normal_(std=0.1)
        module.model.layers[0].input_layernorm = normalization
        return module

    module = create_model()
    tokens = torch.randint(128, (1, 8))
    converted = mk.quantize(
        module,
        mk.Recipe(weights=identity_quantizer, transforms=[mk.fuse_norms()]),
        calib=None,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            converted(tokens).logits, module(tokens).logits, rtol=1e-5, atol=1e-6
        )
    converted.save(tmp_path / "fused")
    restored = mk.load(tmp_path / "fused", model=create_model)
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(tokens).logits, converted(tokens).logits, rtol=0, atol=0
        )
    assert restored.module.model.layers[0].self_attn.q_proj.bias is not None
