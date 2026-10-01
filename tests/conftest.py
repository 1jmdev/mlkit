"""Shared test configuration: mlkit targets CUDA, so tests allocate on CUDA by default."""

import pytest
import torch

REQUIRES_CUDA = pytest.mark.skip(reason="mlkit requires a CUDA device")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    for item in items:
        if "device_independent" not in item.keywords:
            item.add_marker(REQUIRES_CUDA)


@pytest.fixture(autouse=True)
def deterministic_cuda_defaults(request: pytest.FixtureRequest):
    """Seed every generator and make CUDA the default device for tensor factories."""
    torch.manual_seed(0)
    if "device_independent" in request.keywords:
        yield
        return
    with torch.device("cuda"):
        yield
