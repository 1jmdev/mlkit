import torch

import mlkit as mk
from mlkit.quantization.grids.lattice import absolute_points, e8p_points, nearest_e8p


def test_e8p_point_set_and_optimal_search() -> None:
    assert absolute_points().shape == (256, 8)
    points = e8p_points()
    assert points.shape == (65536, 8)
    assert len(points.unique(dim=0)) == 65536
    samples = torch.randn(17, 8, generator=torch.Generator(device="cuda").manual_seed(61))
    result = mk.grid.e8p()(samples)
    indices = nearest_e8p(samples, return_indices=True)
    torch.testing.assert_close(points.to(result.device)[indices], result, rtol=0, atol=0)
    reference = mk.nearest(samples, points, chunk=17)
    torch.testing.assert_close(
        (samples - result).square().sum(1),
        (samples - reference).square().sum(1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fused_lattice_search_has_the_reference_optimal_cost() -> None:
    samples = torch.randn(1024, 8)
    reference = nearest_e8p(samples, backend="torch")
    result = nearest_e8p(samples, backend="triton")
    torch.testing.assert_close(
        (samples - result).square().sum(1),
        (samples - reference).square().sum(1),
        rtol=1e-5,
        atol=1e-5,
    )
