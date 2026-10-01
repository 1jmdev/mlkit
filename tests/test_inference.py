import pytest
import torch
from torch import nn

import mlkit as mk


@pytest.mark.parametrize("format", [mk.int(4, group=32), mk.nf4(group=32)])
@pytest.mark.parametrize("batch", [1, 3, 7])
@pytest.mark.cuda
def test_packed_linear_matches_dense(format, batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = "cuda"
    dtype = torch.float16
    module = nn.Sequential(nn.Linear(130, 71, device=device, dtype=dtype))
    converted = mk.quantize(module, format, calib=None)
    packed = mk.optimize(converted, backend="packed")
    inputs = torch.randn(batch, 130, device=device, dtype=dtype)
    with torch.inference_mode():
        torch.testing.assert_close(packed(inputs), converted(inputs), rtol=0.002, atol=0.002)
    assert isinstance(packed.module[0], mk.PackedLinear)
    assert packed.module[0].packed.numel() == (130 * 71 + 1) // 2


@pytest.mark.cuda
def test_benchmark_has_synchronized_measurements() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    weight = torch.randn(64, 64, device="cuda")
    result = mk.benchmark(lambda: weight @ weight, repetitions=5, warmup=1)
    assert result.median_ms >= result.minimum_ms > 0
    assert result.repetitions == 5
    assert result.percentile_95_ms >= result.median_ms


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_packed_transforms_and_checkpoint_preserve_outputs(tmp_path) -> None:
    module = nn.Sequential(nn.Linear(32, 16, bias=False))
    inputs = torch.randn(4, 32)
    converted = mk.quantize(
        module, mk.Recipe(weights=mk.int(4, group=8), transforms=[mk.smooth()]),
        calib=[inputs],
    )
    packed = mk.optimize(converted)
    torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-5, atol=1e-6)
    packed.save(tmp_path / "packed")
    restored = mk.load(tmp_path / "packed", model=lambda: nn.Sequential(
        nn.Linear(32, 16, bias=False)
    ))
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=0, atol=0)
    measurement = mk.benchmark_model(packed, inputs, warmup=1, repetitions=2)
    assert measurement.median_ms > 0


@pytest.mark.cuda
def test_wrapper_places_input_tensors_automatically() -> None:
    model = mk.Model(nn.Linear(8, 4))
    output = model(torch.randn(3, 8))
    assert output.is_cuda


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_packed_copy_rebinds_activation_context_and_owns_its_storage() -> None:
    observed_modules = []

    @mk.quantizer
    def record_inputs(inputs, context):
        observed_modules.append(context.module)
        return inputs

    module = nn.Sequential(nn.Linear(128, 64, bias=False))
    converted = mk.quantize(
        module, mk.Recipe(weights=mk.int(4, group=32), acts=record_inputs), calib=None
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


@pytest.mark.cuda
@pytest.mark.parametrize("bits", [4, 8])
def test_torchao_export_executes_and_reports_physical_storage(bits: int) -> None:
    pytest.importorskip("torchao")
    device = "cuda"
    dtype = torch.float16
    module = nn.Sequential(nn.Linear(1024, 128, device=device, dtype=dtype))
    module.register_buffer("precision_reference", torch.ones(4, device=device, dtype=torch.float32))
    original = module[0].weight.detach().clone()
    exported = mk.export_torchao(module, bits=bits, group=32)
    inputs = torch.randn(3, 1024, device=device, dtype=exported.dtype)
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
