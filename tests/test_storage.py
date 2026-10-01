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


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_checkpoint_round_trip(tmp_path) -> None:
    def create_model():
        return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))

    model = create_model()
    converted = mk.quantize(model, mk.int(4, group=8), calib=None)
    directory = tmp_path / "checkpoint"
    converted.save(directory)
    restored = mk.load_checkpoint(directory, model=create_model)
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
    restored.save(tmp_path / "restored")
    second_manifest = json.loads((tmp_path / "restored" / "mlkit.json").read_text())
    assert second_manifest["tensor_bytes"] == manifest["tensor_bytes"]


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_custom_reconstruction_checkpoint(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(8, 4))
    converted = mk.quantize(model, lambda w, ctx: w.round(), calib=None)
    converted.save(tmp_path / "checkpoint")
    restored = mk.load_checkpoint(
        tmp_path / "checkpoint", model=lambda: nn.Sequential(nn.Linear(8, 4))
    )
    assert restored.bpw is None
    torch.testing.assert_close(converted.module[0].weight, restored.module[0].weight)


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_root_linear_conversion_checkpoint_and_inference(tmp_path) -> None:
    converted = mk.quantize(nn.Linear(32, 16, bias=False), mk.int(4, group=8), calib=None)
    assert converted.model_bpw == converted.bpw
    inputs = torch.randn(3, 32)
    converted.save(tmp_path / "root")
    restored = mk.load_checkpoint(tmp_path / "root", model=lambda: nn.Linear(32, 16, bias=False))
    torch.testing.assert_close(restored(inputs), converted(inputs), rtol=0, atol=0)
    packed = mk.optimize(restored, inplace=True)
    assert isinstance(packed.module, mk.PackedLinear)
    assert packed.architecture.model is packed.module
    torch.testing.assert_close(packed(inputs), converted(inputs), rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_checkpoint_stores_tied_weights_and_shared_parameters_once(tmp_path) -> None:
    class TiedLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 16)
            self.projections = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
            self.lm_head = nn.Linear(16, 32, bias=False)
            self.lm_head.weight = self.embedding.weight

        def forward(self, tokens):
            return self.lm_head(self.projections(self.embedding(tokens)))

    converted = mk.quantize(TiedLanguageModel(), mk.int(4, group=8), calib=None)
    converted.save(tmp_path / "shared")
    manifest = json.loads((tmp_path / "shared" / "mlkit.json").read_text())
    assert manifest["state_aliases"] == {"lm_head.weight": "embedding.weight"}
    first = manifest["layers"]["projections.0"]["params"]["values"]["tensor"]
    second = manifest["layers"]["projections.1"]["params"]["values"]["tensor"]
    assert first == second
    restored = mk.load_checkpoint(tmp_path / "shared", model=TiedLanguageModel)
    tokens = torch.randint(32, (2, 7))
    torch.testing.assert_close(restored(tokens), converted(tokens), rtol=0, atol=0)
    assert restored.module.embedding.weight is restored.module.lm_head.weight
