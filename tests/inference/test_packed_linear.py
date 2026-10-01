import pytest
import torch
from torch import nn

import mlkit as mk

FORMATS = {
    "int4": mk.int(4, group=32),
    "nf4": mk.nf4(group=32),
}


@pytest.mark.parametrize("name", FORMATS)
@pytest.mark.parametrize("batch", [1, 3, 7])
def test_packed_linear_matches_dense(name: str, batch: int) -> None:
    dtype = torch.float16
    module = nn.Sequential(nn.Linear(130, 71, dtype=dtype))
    converted = mk.quantize(module, FORMATS[name], calib=None)
    packed = mk.optimize(converted, backend="packed")
    inputs = torch.randn(batch, 130, dtype=dtype)
    with torch.inference_mode():
        torch.testing.assert_close(packed(inputs), converted(inputs), rtol=0.002, atol=0.002)
    assert isinstance(packed.module[0], mk.PackedLinear)
    assert packed.module[0].packed.numel() == (130 * 71 + 1) // 2


def test_packed_transforms_and_checkpoint_preserve_outputs(tmp_path) -> None:
    module = nn.Sequential(nn.Linear(32, 16, bias=False))
    inputs = torch.randn(4, 32)
    converted = mk.quantize(
        module,
        mk.Recipe(weights=mk.int(4, group=8), transforms=[mk.smooth()]),
        calib=[inputs],
    )
    packed = mk.optimize(converted)
    torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-5, atol=1e-6)
    packed.save(tmp_path / "packed")
    restored = mk.load(
        tmp_path / "packed",
        model=lambda: nn.Sequential(nn.Linear(32, 16, bias=False)),
    )
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=0, atol=0)
    measurement = mk.benchmark_model(packed, inputs, warmup=1, repetitions=2)
    assert measurement.median_ms > 0


def test_packed_copy_rebinds_activation_context_and_owns_its_storage() -> None:
    observed_modules = []

    @mk.quantizer
    def record_inputs(inputs, context):
        observed_modules.append(context.module)
        return inputs

    module = nn.Sequential(nn.Linear(128, 64, bias=False))
    converted = mk.quantize(
        module,
        mk.Recipe(weights=mk.int(4, group=32), acts=record_inputs),
        calib=None,
    )
    packed = mk.optimize(converted, backend="packed")
    inputs = torch.randn(2, 128)
    with torch.inference_mode():
        expected = converted(inputs)
        observed_modules.clear()
        actual = packed(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert observed_modules == [packed.module[0]]
        converted.module[0].weight.zero_()
        torch.testing.assert_close(packed(inputs), actual, rtol=0, atol=0)
    assert packed.storage_bytes < converted.storage_bytes
