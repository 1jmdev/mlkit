import torch
from reference_models import RepeatedModel

import mlkit as mk


def test_activation_quantization_hooks() -> None:
    original = RepeatedModel()
    converted = mk.quantize(
        original,
        mk.Recipe(weights=mk.int(4, group=8), acts=mk.int(4, group=None)),
        calib=None,
    )
    assert len(converted.activation_handles) == 6
    assert torch.isfinite(converted(torch.randn(3, 16))).all()


def test_custom_activation_statistics_are_collected_before_inference() -> None:
    observed = []

    @mk.quantizer
    def keep_channels(values, context):
        importance = context.stat("channel_max", lambda x: x.abs().amax(0), reduce="max")
        observed.append(importance.clone())
        indices = importance.topk(2).indices
        reconstructed = mk.int(4, group=None)(values).w
        reconstructed[:, indices] = values[:, indices]
        return mk.Q(reconstructed, bits=4 * values.numel() + 12 * 2 * len(values))

    module = torch.nn.Sequential(torch.nn.Linear(16, 8))
    inputs = torch.randn(4, 16)
    converted = mk.quantize(
        module,
        mk.Recipe(weights=mk.int(4, group=8), acts=keep_channels),
        calib=[inputs],
        cache_dir=None,
    )
    assert len(observed) == 1
    converted(inputs)
    assert len(observed) == 2
    torch.testing.assert_close(observed[0], inputs.abs().amax(0), rtol=0, atol=0)
