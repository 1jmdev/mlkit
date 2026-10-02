import json

import pytest
import torch
from reference_models import RepeatedModel, create_tiny_llama
from safetensors.torch import load_file

import mlkit as mk
from mlkit.inference import PackedEmbedding


def test_head_recipe_rounds_layers_outside_the_blocks() -> None:
    original = RepeatedModel()
    recipe = mk.Recipe(weights=mk.int(4, group=8), head=mk.int(8, group=16))
    converted = mk.quantize(original, recipe, calib=None)
    assert list(converted.quantized)[-1] == "lm_head"
    assert len(converted.layer_reports) == 7
    assert converted.layer_reports[-1].bpw == 9
    assert not torch.equal(converted.module.lm_head.weight, original.lm_head.weight)
    torch.testing.assert_close(
        converted.module.lm_head.weight.cpu(), converted.quantized["lm_head"].w, rtol=0, atol=0
    )
    assert converted.tied_weights == {}


def test_head_mapping_selects_layers_by_name() -> None:
    original = RepeatedModel()
    recipe = mk.Recipe(weights=mk.int(4, group=8), head={"other_*": mk.int(8)})
    converted = mk.quantize(original, recipe, calib=None)
    assert "lm_head" not in converted.quantized
    recipe = mk.Recipe(weights=mk.int(4, group=8), head={"lm_*": mk.int(8, group=16)})
    assert "lm_head" in mk.quantize(original, recipe, calib=None).quantized


@pytest.mark.parametrize("sequential", [True, False])
def test_head_statistics_come_from_the_calibration_model(sequential: bool) -> None:
    original = RepeatedModel()
    calibration = [torch.randn(2, 8, 16) for _ in range(3)]
    observed = {}

    @mk.quantizer
    def record(weight, context):
        observed[context.name] = context.H.clone()
        return mk.Q(weight, bits=32 * weight.numel())

    recipe = mk.Recipe(weights=mk.int(4, group=8), head=record)
    converted = mk.quantize(
        original, recipe, calib=calibration, sequential=sequential, cache_dir=None
    )
    reference = converted.module if sequential else original
    inputs = torch.cat(calibration).reshape(-1, 16)
    with torch.no_grad():
        for block in reference.model.layers:
            inputs = block(inputs)
    torch.testing.assert_close(
        observed["lm_head"], inputs.T @ inputs / len(inputs), rtol=1e-4, atol=1e-5
    )


def test_tied_embedding_shares_the_rounded_head(tmp_path) -> None:
    tokens = [torch.randint(0, 128, (2, 12)) for _ in range(2)]
    original = create_tiny_llama(tie_word_embeddings=True)
    recipe = mk.Recipe(weights=mk.int(4, group=32), head=mk.int(4, group=32))
    converted = mk.quantize(original, recipe, calib=tokens, cache_dir=None)
    module = converted.module
    assert converted.tied_weights == {"model.embed_tokens.weight": "lm_head"}
    assert module.lm_head.weight is module.model.embed_tokens.weight
    assert converted.model_bpw < 5
    with torch.inference_mode():
        expected = converted(tokens[0]).logits
        packed = mk.optimize(converted, backend="packed")
        assert isinstance(packed.module.model.embed_tokens, PackedEmbedding)
        assert isinstance(packed.module.lm_head, mk.PackedLinear)
        torch.testing.assert_close(packed(tokens[0]).logits, expected, rtol=1e-4, atol=1e-5)
        assert packed.storage_bytes < converted.storage_bytes / 4
        torch.testing.assert_close(
            packed.module.model.embed_tokens(tokens[0]),
            module.model.embed_tokens(tokens[0]),
            rtol=0,
            atol=0,
        )
        prompt = tokens[0][:1, :4]
        assert torch.equal(
            packed.generate(prompt, max_new_tokens=4, do_sample=False),
            converted.generate(prompt, max_new_tokens=4, do_sample=False),
        )
        packed.save(tmp_path / "checkpoint")
        stored = load_file(tmp_path / "checkpoint" / "weights.safetensors")
        assert "state.model.embed_tokens.weight" not in stored
        manifest = json.loads((tmp_path / "checkpoint" / "mlkit.json").read_text())
        assert manifest["tied_weights"] == converted.tied_weights
        restored = mk.load(tmp_path / "checkpoint")
        assert restored.tied_weights == converted.tied_weights
        assert restored.module.lm_head.weight is restored.module.model.embed_tokens.weight
        torch.testing.assert_close(restored(tokens[0]).logits, expected, rtol=0, atol=0)
        assert restored.model_bpw == converted.model_bpw


def test_untied_embedding_remains_dense() -> None:
    original = create_tiny_llama()
    recipe = mk.Recipe(weights=mk.int(4, group=32), head=mk.int(4, group=32))
    converted = mk.quantize(original, recipe, calib=None)
    assert converted.tied_weights == {}
    packed = mk.optimize(converted, backend="packed")
    assert isinstance(packed.module.model.embed_tokens, torch.nn.Embedding)
    assert torch.equal(
        packed.module.model.embed_tokens.weight, original.model.embed_tokens.weight
    )
