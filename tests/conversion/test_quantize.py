import copy
import weakref

import pytest
import torch
from reference_models import RepeatedModel

import mlkit as mk


def test_conversion_preserves_source_and_skips_head() -> None:
    original = RepeatedModel()
    state = copy.deepcopy(original.state_dict())
    converted = mk.quantize(original, mk.int(4, group=8), calib=None)
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, state[name])
    assert len(converted.layer_reports) == 6
    assert converted.bpw == 6
    assert converted.model_bpw > converted.bpw
    assert torch.equal(converted.module.lm_head.weight, original.lm_head.weight)
    assert converted(torch.randn(2, 16)).shape == (2, 5)


def test_sequential_statistics_use_quantized_prefix() -> None:
    original = RepeatedModel()
    calibration = [torch.randn(2, 8, 16) for _ in range(3)]
    observed = {}

    @mk.quantizer
    def record(w, ctx):
        observed[ctx.name] = ctx.H.clone()
        return mk.Q(torch.zeros_like(w), bits=2 * w.numel())

    mk.quantize(original, record, calib=calibration, cache_dir=None)
    first_bias = original.model.layers[0][2].bias
    expected = first_bias[:, None] @ first_bias[None, :]
    torch.testing.assert_close(observed["model.layers.1.0"], expected)
    assert observed["model.layers.0.0"].diagonal().mean() > 0.5


def test_pattern_order_and_per_layer_selector() -> None:
    original = RepeatedModel()
    converted = mk.quantize(
        original,
        {
            "*.layers.0.*": None,
            "*.layers.1.*": lambda ctx: mk.int(3, group=8),
            "*.layers.*": mk.int(4, group=8),
        },
        calib=None,
    )
    assert len(converted.layer_reports) == 4
    assert converted.layer_reports[0].bpw == 5
    assert converted.layer_reports[2].bpw == 6


def test_conversion_releases_completed_layer_hessians() -> None:
    hessians = []

    @mk.quantizer
    def record(weight, context):
        hessians.append(weakref.ref(context.H))
        return mk.Q(weight.clone(), bits=16 * weight.numel())

    mk.quantize(RepeatedModel(), record, calib=[torch.randn(2, 16)], cache_dir=None)
    assert len(hessians) == 6
    assert all(reference() is None for reference in hessians)


def test_custom_statistics_and_disk_cache(tmp_path) -> None:
    model = RepeatedModel()
    calibration = [torch.randn(2, 16), torch.randn(3, 16)]
    observed = {}

    @mk.quantizer
    def statistics(w, ctx):
        observed[ctx.name] = ctx.stat("m4", lambda x: x.pow(4).mean(0))
        _ = ctx.H
        return mk.Q(w, bits=w.numel() * 32)

    mk.quantize(model, statistics, calib=calibration, sequential=False, cache_dir=tmp_path)
    expected = torch.cat(calibration).pow(4).mean(0)
    torch.testing.assert_close(observed["model.layers.0.0"], expected)
    assert len(list(tmp_path.rglob("*.safetensors"))) == 6


def test_missing_calibration_is_actionable() -> None:
    with pytest.raises(ValueError, match="calibration"):
        mk.quantize(RepeatedModel(), mk.gptq(mk.int(4, group=8)), calib=None)


def test_cached_statistics_skip_model_calibration_forward(tmp_path) -> None:
    model = RepeatedModel()
    calls = []
    model.register_forward_hook(lambda *arguments: calls.append(True))
    calibration = [torch.randn(4, 16)]

    @mk.quantizer
    def record(values, context):
        _ = context.H
        return mk.Q(values, bits=32 * values.numel())

    mk.quantize(model, record, calib=calibration, sequential=False, cache_dir=tmp_path)
    assert calls
    calls.clear()
    mk.quantize(model, record, calib=calibration, sequential=False, cache_dir=tmp_path)
    assert not calls
