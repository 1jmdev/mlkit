import pytest
import torch
from reference_models import SharedProjectionModel
from torch import nn

import mlkit as mk


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
    restored = mk.load(
        tmp_path / "basis",
        model=lambda: nn.Sequential(nn.Linear(32, 16, bias=False)),
    )
    torch.testing.assert_close(converted(inputs), restored(inputs), rtol=1e-5, atol=1e-6)
    assert restored.quantized["0"].metadata == representation.metadata


def test_shared_awq_uses_one_scale_and_counts_it_once(tmp_path) -> None:
    inputs = torch.randn(32, 16)
    model = SharedProjectionModel()
    converted = mk.quantize(
        model,
        mk.awq(mk.int(4, group=8), grid=4, shared=True),
        calib=[inputs],
        cache_dir=None,
    )
    representations = list(converted.quantized.values())
    for representation in representations:
        assert representation.codec == "channel_scaled"
        torch.testing.assert_close(
            representation.params["channel_scales"],
            representations[0].params["channel_scales"],
        )
    assert sum(record.bits for record in converted.layer_reports) == 3 * (4 * 256 + 16 * 32) + 256
    converted.save(tmp_path / "awq")
    restored = mk.load(tmp_path / "awq", model=SharedProjectionModel)
    torch.testing.assert_close(converted(inputs), restored(inputs), rtol=0, atol=0)


def test_e8p_feedback_codec_supports_sign_finetuning(tmp_path) -> None:
    inputs = torch.randn(4, 16)
    format = mk.scaled(mk.grid.e8p(), group=None)
    algorithm = mk.incoherent(mk.ldlq(format, step=8), train_signs=True)
    recipe = mk.Recipe(weights=algorithm, passes=[mk.finetune(steps=2, bs=1)])
    converted = mk.quantize(
        nn.Sequential(nn.Linear(16, 8, bias=False)),
        recipe,
        calib=[inputs],
        cache_dir=None,
    )
    representation = converted.quantized["0"]
    assert representation.codec == "basis"
    assert representation.params["inner_codec"] == "vector_feedback"
    assert representation.codes.shape == (8, 2)
    converted.save(tmp_path / "e8p")
    restored = mk.load(
        tmp_path / "e8p",
        model=lambda: nn.Sequential(nn.Linear(16, 8, bias=False)),
    )
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=1e-5, atol=1e-6)
