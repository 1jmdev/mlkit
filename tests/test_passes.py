import pytest
import torch
from torch import nn

import mlkit as mk

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


def test_codec_parameters_are_trained_and_cached() -> None:
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    calibration = [torch.randn(4, 16) for _ in range(4)]
    observations = {}

    @mk.block_pass
    def train_block(block, ctx):
        assert len(ctx.qparams) == 2
        with torch.no_grad():
            observations["before"] = sum(
                (ctx.forward(block, call.hidden()) - target).square().mean()
                for call, target in zip(ctx.calls, ctx.targets.split(4), strict=True)
            )
        mk.finetune(block, ctx, steps=30, lr=0.002, bs=2)
        with torch.no_grad():
            observations["after"] = sum(
                (ctx.forward(block, call.hidden()) - target).square().mean()
                for call, target in zip(ctx.calls, ctx.targets.split(4), strict=True)
            )

    recipe = mk.Recipe(weights=mk.int(2, group=8), passes=[train_block])
    converted = mk.quantize(model, recipe, calib=calibration)
    assert observations["after"] < observations["before"]
    assert isinstance(converted.module[0], nn.Linear)
    assert converted.module[0].weight.requires_grad
    torch.testing.assert_close(converted.module[0].weight.cpu(), converted.quantized["0"].w)


def test_block_pass_configuration() -> None:
    observed = []

    @mk.block_pass
    def record(block, ctx, count=1):
        observed.append(count)

    record(count=3)(None, None)
    assert observed == [3]


def test_half_precision_normalization_finetuning_stays_finite() -> None:
    dtype = torch.float16
    module = nn.Sequential(nn.LayerNorm(16, dtype=dtype), nn.Linear(16, 16, dtype=dtype))
    inputs = [torch.randn(4, 16, dtype=dtype) for _ in range(4)]
    original_norm = module[0].weight.detach().clone()
    converted = mk.quantize(
        module,
        mk.Recipe(weights=mk.int(2, group=8), passes=[mk.finetune(steps=10, bs=2)]),
        calib=inputs,
    )
    assert all(torch.isfinite(parameter).all() for parameter in converted.parameters())
    assert torch.isfinite(converted(inputs[0])).all()
    assert not torch.equal(converted.module[0].weight, original_norm)
    assert converted.module[0].weight.dtype == dtype


def test_shared_vector_grid_stays_fixed_during_block_finetuning() -> None:
    codebook = torch.randn(16, 2)
    format = mk.scaled(mk.grid.vector(codebook), group=8)
    inputs = [torch.randn(4, 8) for _ in range(2)]
    module = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    converted = mk.quantize(
        module, mk.Recipe(weights=format, passes=[mk.finetune(steps=3, bs=1)]), calib=inputs
    )
    for representation in converted.quantized.values():
        torch.testing.assert_close(representation.params["values"], codebook.cpu(), rtol=0, atol=0)
    assert sum(record.bits for record in converted.layer_reports) == 2 * (2 * 64 + 16 * 8) + 1024
