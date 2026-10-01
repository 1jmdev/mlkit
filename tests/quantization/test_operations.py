import pytest
import torch

import mlkit as mk
from mlkit.quantization.operations import structured_transform


def test_hadamard_and_random_transform_inverses() -> None:
    values = torch.randn(12, 64)
    torch.testing.assert_close(mk.hadamard(mk.hadamard(values)), values)
    torch.testing.assert_close(mk.rht(mk.rht(values, seed=4), seed=4, inverse=True), values)
    with pytest.raises(ValueError, match="power-of-two"):
        mk.hadamard(torch.randn(3, 12))


def test_structured_transforms_support_transformer_widths() -> None:
    for width in [12, 20, 28, 96, 896, 4864, 7]:
        values = torch.randn(3, width)
        transformed = structured_transform(values)
        torch.testing.assert_close(
            structured_transform(transformed, inverse=True), values, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            transformed.square().sum(1), values.square().sum(1), rtol=1e-5, atol=1e-5
        )


def test_scalar_and_vector_search() -> None:
    values = torch.randn(64)
    codebook = torch.tensor([-1.0, -0.2, 0.0, 0.5, 1.0])
    expected = codebook[(values[:, None] - codebook).square().argmin(1)]
    torch.testing.assert_close(mk.snap(values, codebook), expected)
    samples, vectors = torch.randn(30, 8), torch.randn(21, 8)
    expected_indices = torch.cdist(samples, vectors).argmin(1)
    indices = mk.nearest(samples, vectors, chunk=7, codebook_chunk=5, return_indices=True)
    assert torch.equal(indices, expected_indices)


def test_scalar_vector_search_preserves_original_order_ties() -> None:
    samples = torch.tensor([[0.0], [1.0], [-1.0], [3.0], [-3.0]])
    codebook = torch.tensor([[2.0], [-2.0], [-2.0], [2.0], [0.0]])
    expected = (samples - codebook.T).square().argmin(1)
    actual = mk.nearest(samples, codebook, return_indices=True)
    assert torch.equal(actual, expected)


def test_weighted_scalar_kmeans_matches_known_centroids() -> None:
    samples = torch.tensor([-4.0, -2.0, 8.0, 10.0])
    weights = torch.tensor([1.0, 3.0, 3.0, 1.0])
    centers = mk.kmeans(samples, k=2, weights=weights, iters=10)
    torch.testing.assert_close(centers, torch.tensor([-2.5, 8.5]), rtol=0, atol=0)


@pytest.mark.parametrize("dimension", [1, 8])
def test_weighted_kmeans_is_reproducible(dimension: int) -> None:
    samples = torch.randn(4096, dimension)
    if dimension == 1:
        samples = samples[:, 0]
    weights = torch.rand(4096)
    expected = mk.kmeans(samples, k=16, weights=weights, iters=3, seed=79)
    actual = mk.kmeans(samples, k=16, weights=weights, iters=3, seed=79)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.device_independent
@pytest.mark.parametrize("bits", range(1, 17))
@pytest.mark.parametrize("elements", [0, 1, 7, 17, 64])
def test_bitstream_round_trip(bits: int, elements: int) -> None:
    codes = torch.randint(2**bits, (elements,), dtype=torch.int32)
    packed = mk.pack(codes, bits)
    assert packed.numel() == (elements * bits + 7) // 8
    assert torch.equal(mk.unpack(packed, bits, codes.shape).int(), codes)


@pytest.mark.parametrize("bits", [1, 3, 4, 7, 12, 16])
def test_bitstream_round_trip_on_cuda(bits: int) -> None:
    codes = torch.randint(2**bits, (1021,), dtype=torch.int32)
    packed = mk.pack(codes, bits)
    assert packed.is_cuda
    assert torch.equal(mk.unpack(packed, bits, codes.shape).int(), codes)


@pytest.mark.device_independent
def test_unsigned_storage_validation() -> None:
    with pytest.raises(ValueError, match="capacity"):
        mk.pack(torch.tensor([16]), 4)
    with pytest.raises(ValueError, match="byte count"):
        mk.unpack(torch.zeros(1, dtype=torch.uint8), 4, (4,))
