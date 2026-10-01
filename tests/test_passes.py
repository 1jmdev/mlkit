import torch
from torch import nn

import mlkit as mk


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
    torch.testing.assert_close(converted.module[0].weight, converted.quantized["0"].w)


def test_block_pass_configuration() -> None:
    observed = []

    @mk.block_pass
    def record(block, ctx, count=1):
        observed.append(count)

    record(count=3)(None, None)
    assert observed == [3]
