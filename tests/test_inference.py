import pytest
import torch
from torch import nn

import mlkit as mk


@pytest.mark.parametrize("format", [mk.int(4, group=32), mk.nf4(group=32)])
@pytest.mark.parametrize("batch", [1, 3, 7])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_linear_matches_dense(format, batch: int, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.float16 if device == "cuda" else torch.float32
    module = nn.Sequential(nn.Linear(130, 71, device=device, dtype=dtype))
    converted = mk.quantize(module, format, calib=None)
    packed = mk.optimize(converted, backend="packed")
    inputs = torch.randn(batch, 130, device=device, dtype=dtype)
    with torch.inference_mode():
        torch.testing.assert_close(packed(inputs), converted(inputs), rtol=0.002, atol=0.002)
    assert isinstance(packed.module[0], mk.PackedLinear)
    assert packed.module[0].packed.numel() == (130 * 71 + 1) // 2


def test_benchmark_has_synchronized_measurements() -> None:
    weight = torch.randn(64, 64)
    result = mk.benchmark(lambda: weight @ weight, repetitions=5, warmup=1)
    assert result.median_ms >= result.minimum_ms > 0
    assert result.repetitions == 5
    assert result.percentile_95_ms >= result.median_ms
