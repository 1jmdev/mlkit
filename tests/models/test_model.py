import torch
from torch import nn

import mlkit as mk


def test_wrapper_places_module_and_input_tensors_on_cuda() -> None:
    model = mk.Model(nn.Linear(8, 4, device="cpu"))
    output = model(torch.randn(3, 8, device="cpu"))
    assert output.is_cuda
    assert model.device.type == "cuda"
