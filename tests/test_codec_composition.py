import pytest
import torch
from torch import nn

import mlkit as mk

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("cuda_tensors")]


@pytest.mark.parametrize("train_signs", [False, True])
def test_incoherent_codec_is_portable_and_differentiable(tmp_path, train_signs: bool) -> None:
    module = nn.Sequential(nn.Linear(32, 16, bias=False))
    recipe = mk.Recipe(
        weights=mk.incoherent(mk.int(4, group=8), train_signs=train_signs),
        passes=[mk.finetune(steps=2, bs=1)] if train_signs else [],
    )
    inputs = torch.randn(4, 32)
    converted = mk.quantize(module, recipe, calib=[inputs])
    representation = converted.quantized["0"]
    assert representation.codec == "basis"
    expected_bits = 4 * 512 + 16 * 64 + (16 * 48 if train_signs else 0)
    assert representation.bits == expected_bits
    converted.save(tmp_path / "basis")
    restored = mk.load(tmp_path / "basis", model=lambda: nn.Sequential(
        nn.Linear(32, 16, bias=False)
    ))
    torch.testing.assert_close(converted(inputs), restored(inputs), rtol=1e-5, atol=1e-6)
    assert restored.quantized["0"].metadata == representation.metadata
