import pytest
import torch
from torch import nn

import mlkit as mk

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


def test_input_smoothing_preserves_outputs_and_serializes(tmp_path) -> None:
    module = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    inputs = torch.randn(4, 16)
    definition = mk.Recipe(weights=lambda w, ctx: mk.Q(w, bits=32 * w.numel()),
                           transforms=[mk.smooth(0.5)])
    converted = mk.quantize(module, definition, calib=[inputs])
    torch.testing.assert_close(converted(inputs), module(inputs), rtol=1e-5, atol=1e-6)
    converted.save(tmp_path / "smoothed")
    restored = mk.load(tmp_path / "smoothed", model=lambda: nn.Sequential(
        nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8)
    ))
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=0, atol=0)


@pytest.mark.integration
def test_llama_norm_fusion_and_rotation_preserve_logits(tmp_path) -> None:
    model = mk.load("hf-internal-testing/tiny-random-LlamaForCausalLM")
    tokens = torch.randint(100, 2000, (1, 17))
    with torch.no_grad():
        expected = model(tokens).logits
    for transform in [mk.fuse_norms(), mk.rotate(seed=7)]:
        definition = mk.Recipe(weights=lambda w, ctx: mk.Q(w, bits=32 * w.numel()),
                               transforms=[transform])
        converted = mk.quantize(model, definition, calib=None)
        with torch.no_grad():
            torch.testing.assert_close(converted(tokens).logits, expected, rtol=1e-4, atol=1e-5)
        directory = tmp_path / type(transform).__name__
        converted.save(directory)
        restored = mk.load(directory)
        with torch.no_grad():
            torch.testing.assert_close(restored(tokens).logits, converted(tokens).logits,
                                       rtol=0, atol=0)
