import copy

import pytest
import torch
from torch import nn

import mlkit as mk
from mlkit.runtime.engine import CalibrationSession

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


class RepeatedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
            for _ in range(3)
        ])
        self.lm_head = nn.Linear(16, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            inputs = block(inputs)
        return self.lm_head(inputs)


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


def test_calibration_shares_metadata_and_releases_completed_blocks() -> None:
    class MetadataBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(16, 16)

        def forward(self, hidden_states, position_ids):
            return self.projection(hidden_states) + position_ids.mean()

    class MetadataModel(RepeatedModel):
        def __init__(self):
            super().__init__()
            self.model.layers = nn.ModuleList([MetadataBlock() for _ in range(3)])

        def forward(self, inputs):
            positions = torch.arange(16, dtype=inputs.dtype, device=inputs.device)
            for block in self.model.layers:
                inputs = block(inputs, position_ids=positions)
            return self.lm_head(inputs)

    wrapped = mk.Model(MetadataModel())
    inputs = torch.randn(2, 16)
    session = CalibrationSession(
        wrapped, [inputs], sequential=True, sample_rows=32, cache_dir=None,
        need_targets=False, retain_history=False,
    )
    session.prepare()
    positions = [session.block_calls(index)[0].kwargs["position_ids"] for index in range(3)]
    assert len({value.untyped_storage().data_ptr() for value in positions}) == 1
    with torch.no_grad():
        expected = wrapped.blocks[0](inputs, position_ids=positions[0].cuda())
    session.propagate(0, wrapped.blocks[0])
    session.release(0)
    assert session.block_calls(0) == []
    torch.testing.assert_close(session.block_calls(1)[0].hidden(), expected.cpu(), rtol=0, atol=0)


def test_pattern_order_and_per_layer_selector() -> None:
    original = RepeatedModel()
    converted = mk.quantize(original, {
        "*.layers.0.*": None,
        "*.layers.1.*": lambda ctx: mk.int(3, group=8),
        "*.layers.*": mk.int(4, group=8),
    }, calib=None)
    assert len(converted.layer_reports) == 4
    assert converted.layer_reports[0].bpw == 5
    assert converted.layer_reports[2].bpw == 6


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


def test_activation_quantization_hooks() -> None:
    original = RepeatedModel()
    converted = mk.quantize(
        original, mk.Recipe(weights=mk.int(4, group=8), acts=mk.int(4, group=None)),
        calib=None,
    )
    assert len(converted.activation_handles) == 6
    assert torch.isfinite(converted(torch.randn(3, 16))).all()


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
