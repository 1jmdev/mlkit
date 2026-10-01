import pytest
import torch

import mlkit as mk
from mlkit.calibration import StatisticAccumulator

pytestmark = pytest.mark.integration

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def token_batches(sequence: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cuda").manual_seed(37)
    return [
        {"input_ids": torch.randint(100, 2000, (1, sequence), generator=generator)}
        for _ in range(3)
    ]


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_downloaded_llama_conversion(tmp_path, dtype: str) -> None:
    model = mk.load(TINY_LLAMA, dtype=dtype)
    batches = token_batches(32)
    baseline = mk.ppl(model, data=batches, budget="full")
    converted = mk.quantize(
        model, mk.gptq(mk.int(4, group=16), refit=16), calib=batches, cache_dir=None
    )
    assert len(converted.layer_reports) == 14
    assert converted.bpw is not None
    score = mk.ppl(converted, data=batches, budget="full")
    assert abs(score - baseline) / baseline < 0.02
    converted.save(tmp_path / "llama_checkpoint")
    restored = mk.load(tmp_path / "llama_checkpoint")
    assert restored.bpw == converted.bpw
    assert restored.storage_bytes == converted.storage_bytes
    assert restored.dtype == converted.dtype
    tied = converted.module.lm_head.weight is converted.module.get_input_embeddings().weight
    restored_tied = restored.module.lm_head.weight is restored.module.get_input_embeddings().weight
    assert restored_tied == tied
    assert all(parameter.is_cuda for parameter in restored.parameters())
    assert not restored.module.training
    tokens = batches[0]["input_ids"]
    with torch.no_grad():
        torch.testing.assert_close(
            converted(tokens).logits, restored(tokens).logits, rtol=0, atol=0
        )


def test_sequential_llama_replay_preserves_calibration_statistics() -> None:
    model = mk.load(TINY_LLAMA, dtype="float16")
    batches = token_batches(128)
    collectors = {}
    handles = []
    for name, module in model.module.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.layers."):
            continue
        collector = StatisticAccumulator("H")
        collectors[name] = collector

        def observe(module, arguments, collector=collector):
            collector.update(arguments[0])

        handles.append(module.register_forward_pre_hook(observe))
    try:
        with torch.no_grad():
            for batch in batches:
                model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    observed = {}

    @mk.quantizer
    def identity(weight, context):
        observed[context.name] = context.H.cpu()
        return mk.Q(weight.clone(), bits=16 * weight.numel())

    mk.quantize(model, identity, calib=batches, cache_dir=None, token_energy_limit=None)
    assert observed.keys() == collectors.keys()
    for name, value in observed.items():
        torch.testing.assert_close(value, collectors[name].result().cpu(), rtol=1e-5, atol=1e-6)
