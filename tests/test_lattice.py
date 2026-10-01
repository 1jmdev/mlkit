import pytest
import torch

import mlkit as mk
from mlkit.quantization.grids.lattice import absolute_points, e8p_points, nearest_e8p
from mlkit.quantization.operations import structured_transform


def test_e8p_point_set_and_optimal_search() -> None:
    assert absolute_points().shape == (256, 8)
    points = e8p_points()
    assert points.shape == (65536, 8)
    assert len(points.unique(dim=0)) == 65536
    samples = torch.randn(17, 8, generator=torch.Generator().manual_seed(61))
    result = mk.grid.e8p()(samples)
    indices = nearest_e8p(samples, return_indices=True)
    torch.testing.assert_close(points.to(result.device)[indices], result, rtol=0, atol=0)
    reference = mk.nearest(samples, points, chunk=17)
    torch.testing.assert_close((samples - result).square().sum(1),
                               (samples - reference).square().sum(1), rtol=1e-5, atol=1e-6)


def test_structured_transforms_support_transformer_widths() -> None:
    for width in [12, 20, 28, 96, 896, 4864, 7]:
        values = torch.randn(3, width)
        transformed = structured_transform(values)
        torch.testing.assert_close(structured_transform(transformed, inverse=True), values,
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(transformed.square().sum(1), values.square().sum(1),
                                   rtol=1e-5, atol=1e-5)


def test_incoherence_on_non_power_of_two_dimensions() -> None:
    weights = torch.randn(28, 96)
    result = mk.incoherent(lambda w, ctx: mk.Q(w, bits=32 * w.numel()))(weights)
    torch.testing.assert_close(result.w, weights, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
def test_fused_lattice_search_has_the_reference_optimal_cost() -> None:
    samples = torch.randn(1024, 8)
    reference = nearest_e8p(samples, backend="torch")
    result = nearest_e8p(samples, backend="triton")
    torch.testing.assert_close((samples - result).square().sum(1),
                               (samples - reference).square().sum(1), rtol=1e-5, atol=1e-5)
