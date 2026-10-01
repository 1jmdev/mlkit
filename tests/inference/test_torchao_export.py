import pytest
import torch
from torch import nn

import mlkit as mk

pytest.importorskip("torchao")


@pytest.mark.parametrize("bits", [4, 8])
def test_torchao_export_executes_and_reports_physical_storage(bits: int) -> None:
    dtype = torch.float16
    module = nn.Sequential(nn.Linear(1024, 128, dtype=dtype))
    module.register_buffer("precision_reference", torch.ones(4, dtype=torch.float32))
    original = module[0].weight.detach().clone()
    exported = mk.export_torchao(module, bits=bits, group=32)
    inputs = torch.randn(3, 1024, dtype=exported.dtype)
    with torch.inference_mode():
        actual = exported(inputs)
        expected = torch.nn.functional.linear(
            inputs, original.to(inputs.dtype), module[0].bias.to(inputs.dtype)
        )
    relative_error = (actual - expected).float().square().mean().sqrt() / expected.float().std()
    assert float(relative_error) < 0.2
    assert exported.storage_bytes < original.numel() * original.element_size()
    assert exported.module.precision_reference.dtype == torch.float32
    torch.testing.assert_close(module[0].weight, original, rtol=0, atol=0)
