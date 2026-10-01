import pytest
import torch


@pytest.fixture
def cuda_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with torch.device("cuda"):
        yield
