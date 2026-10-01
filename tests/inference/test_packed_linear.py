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


def packed_and_reference(
    format,
    *,
    input_width: int = 256,
    output_width: int = 37,
    dtype: torch.dtype = torch.float32,
    calibration: torch.Tensor | None = None,
):
    module = nn.Sequential(nn.Linear(input_width, output_width, dtype=dtype))
    converted = mk.quantize(
        module, format, calib=None if calibration is None else [calibration], cache_dir=None
    )
    packed = mk.optimize(converted, backend="packed")
    assert isinstance(packed.module[0], mk.PackedLinear)
    return packed, converted


@pytest.mark.parametrize("rows", [1, 2, 5, 8, 9, 33])
def test_packed_linear_handles_every_row_count(rows: int) -> None:
    packed, converted = packed_and_reference(mk.int(4, group=64))
    inputs = torch.randn(rows, 256)
    with torch.inference_mode():
        torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-4, atol=1e-5)
        nested = inputs.reshape(1, rows, 256)
        torch.testing.assert_close(packed(nested), converted(nested), rtol=1e-4, atol=1e-5)


def test_packed_linear_applies_asymmetric_offsets() -> None:
    packed, converted = packed_and_reference(mk.int(4, group=64, asym=True))
    assert packed.module[0].zeros is not None
    for rows in (1, 3, 20):
        inputs = torch.randn(rows, 256)
        with torch.inference_mode():
            torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-4, atol=1e-5)


def test_packed_linear_executes_error_feedback_codecs() -> None:
    calibration = torch.randn(512, 256)
    packed, converted = packed_and_reference(
        mk.gptq(mk.int(4, group=64), refit=64), calibration=calibration
    )
    assert converted.quantized["0"].codec == "feedback"
    inputs = torch.randn(3, 256)
    with torch.inference_mode():
        torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("input_width", "group"),
    [(255, 64), (256, 5), (250, 20)],
)
def test_layouts_without_byte_alignment_use_dense_reconstruction(
    input_width: int,
    group: int,
) -> None:
    packed, converted = packed_and_reference(mk.int(4, group=group), input_width=input_width)
    layer = packed.module[0]
    assert layer.tile_bytes is None
    assert layer.maximum_fused_rows == 0
    inputs = torch.randn(2, input_width)
    with torch.inference_mode():
        torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(layer.weight, converted.module[0].weight, rtol=1e-6, atol=1e-7)


def test_packed_weight_reconstruction_matches_the_converted_layer() -> None:
    for format in (mk.int(4, group=64), mk.nf4(group=32), mk.int(4, group=64, asym=True)):
        packed, converted = packed_and_reference(format)
        torch.testing.assert_close(
            packed.module[0].weight, converted.module[0].weight, rtol=1e-6, atol=1e-7
        )


def test_packed_linear_can_be_copied_and_cast_after_launching() -> None:
    import copy

    packed, converted = packed_and_reference(mk.int(4, group=64), dtype=torch.float16)
    inputs = torch.randn(1, 256, dtype=torch.float16)
    with torch.inference_mode():
        expected = packed(inputs)
        duplicate = copy.deepcopy(packed)
        torch.testing.assert_close(duplicate(inputs), expected, rtol=0, atol=0)
        single = duplicate.module[0].float()
        result = single(inputs.float())
        torch.testing.assert_close(result, expected.float(), rtol=2e-3, atol=2e-3)
        assert result.dtype == torch.float32
