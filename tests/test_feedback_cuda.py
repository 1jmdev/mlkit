import pytest
import torch

import mlkit as mk


@pytest.mark.cuda
@pytest.mark.parametrize("refit", [16, 64, None])
@pytest.mark.parametrize("act_order", [False, True])
def test_fused_feedback_matches_torch(refit: int | None, act_order: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(79)
    weights = torch.randn(71, 96, device="cuda", generator=generator)
    inputs = torch.randn(256, 96, device="cuda", generator=generator)
    context = mk.Ctx(X=inputs)
    quantization = mk.int(4, group=16)
    expected = mk.gptq(quantization, refit=refit, act_order=act_order,
                       backend="torch")(weights, context)
    result = mk.gptq(quantization, refit=refit, act_order=act_order,
                     backend="triton")(weights, context)
    torch.testing.assert_close(result.w, expected.w, rtol=1e-5, atol=1e-6)
    assert result.bits == expected.bits
    assert result.codec == "feedback"
