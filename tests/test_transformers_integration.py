import pytest
import torch

import mlkit as mk


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_downloaded_llama_conversion(tmp_path, dtype: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = mk.load("hf-internal-testing/tiny-random-LlamaForCausalLM", dtype=dtype)
    generator = torch.Generator().manual_seed(37)
    batches = [{"input_ids": torch.randint(100, 2000, (1, 32), generator=generator)}
               for _ in range(3)]
    baseline = mk.ppl(model, data=batches, budget="full")
    converted = mk.quantize(model, mk.gptq(mk.int(4, group=16), refit=16),
                            calib=batches, cache_dir=None)
    assert len(converted.layer_reports) == 14
    assert converted.bpw is not None
    score = mk.ppl(converted, data=batches, budget="full")
    assert abs(score - baseline) / baseline < 0.02
    converted.save(tmp_path / "llama_checkpoint")
    restored = mk.load(tmp_path / "llama_checkpoint")
    assert restored.bpw == converted.bpw
    tokens = batches[0]["input_ids"].cuda()
    with torch.no_grad():
        torch.testing.assert_close(
            converted(tokens).logits, restored(tokens).logits, rtol=0, atol=0
        )
