import json

import pytest
import torch
from torch import nn

import mlkit as mk


@pytest.mark.parametrize("bits", range(1, 17))
@pytest.mark.parametrize("elements", [0, 1, 7, 17, 64])
def test_bitstream_round_trip(bits: int, elements: int) -> None:
    codes = torch.randint(2**bits, (elements,), dtype=torch.int32)
    packed = mk.pack(codes, bits)
    assert packed.numel() == (elements * bits + 7) // 8
    assert torch.equal(mk.unpack(packed, bits, codes.shape).int(), codes)


def test_unsigned_storage_validation() -> None:
    with pytest.raises(ValueError, match="capacity"):
        mk.pack(torch.tensor([16]), 4)
    with pytest.raises(ValueError, match="byte count"):
        mk.unpack(torch.zeros(1, dtype=torch.uint8), 4, (4,))


def test_checkpoint_round_trip(tmp_path) -> None:
    def create_model():
        return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))

    model = create_model()
    converted = mk.quantize(model, mk.int(4, group=8), calib=None)
    directory = tmp_path / "checkpoint"
    converted.save(directory)
    restored = mk.load_checkpoint(directory, model=create_model, device="cpu")
    inputs = torch.randn(4, 32)
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=0, atol=0)
    assert restored.bpw == converted.bpw
    manifest = json.loads((directory / "mlkit.json").read_text())
    assert manifest["tensor_bytes"] < sum(
        value.numel() * value.element_size() for value in converted.module.state_dict().values()
    )
    with pytest.raises(FileExistsError):
        converted.save(directory)
    converted.save(directory, overwrite=True)


def test_custom_reconstruction_checkpoint(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(8, 4))
    converted = mk.quantize(model, lambda w, ctx: w.round(), calib=None)
    converted.save(tmp_path / "checkpoint")
    restored = mk.load_checkpoint(
        tmp_path / "checkpoint", model=lambda: nn.Sequential(nn.Linear(8, 4)), device="cpu"
    )
    assert restored.bpw is None
    torch.testing.assert_close(converted.module[0].weight, restored.module[0].weight)
