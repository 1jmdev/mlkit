import pytest
import torch

import mlkit as mk

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


@pytest.mark.parametrize("format", [
    mk.int(3, group=32), mk.int(4, group=None), mk.nf4(group=64),
    mk.mxfp4(), mk.int(4, group=32, asym=True), mk.int(4, group=32, scale_fmt="bf16"),
    mk.int(4, group=32, scale_fmt="fp8"),
])
def test_fused_activation_reconstruction_matches_fitted_format(format) -> None:
    inputs = torch.randn(7, 130, dtype=torch.float16)
    expected = format(inputs.float()).w.half()
    result = format.reconstruct_activations(inputs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


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
    converted = mk.quantize(module, mk.Recipe(weights=mk.int(4, group=8), acts=keep_channels),
                            calib=[inputs], cache_dir=None)
    assert len(observed) == 1
    converted(inputs)
    assert len(observed) == 2
    torch.testing.assert_close(observed[0], inputs.abs().amax(0), rtol=0, atol=0)
