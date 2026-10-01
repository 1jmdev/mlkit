import torch

import mlkit as mk


def test_benchmark_has_synchronized_measurements() -> None:
    weight = torch.randn(64, 64)
    result = mk.benchmark(lambda: weight @ weight, repetitions=5, warmup=1)
    assert result.median_ms >= result.minimum_ms > 0
    assert result.repetitions == 5
    assert result.percentile_95_ms >= result.median_ms
