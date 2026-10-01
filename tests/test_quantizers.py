import pytest
import torch

import mlkit as mk


@pytest.fixture
def weights() -> torch.Tensor:
    return torch.randn(16, 64, generator=torch.Generator().manual_seed(7))


def test_scaled_codec_and_bits(weights: torch.Tensor) -> None:
    result = mk.int(4, group=16)(weights)
    assert result.codes is not None
    assert result.bits == 4 * weights.numel() + 16 * 64
    assert result.bpw == 5
    assert result.w.shape == weights.shape
    assert (weights - result.w).square().mean() < 0.015
    assert torch.equal(result.w, result.decode(result.codes, **result.params))


@pytest.mark.parametrize("group", [None, 5, 64, 128])
def test_scaled_handles_partial_groups(weights: torch.Tensor, group: int | None) -> None:
    result = mk.nf4(group=group)(weights)
    assert result.w.shape == weights.shape
    assert torch.isfinite(result.w).all()


def test_function_configuration(weights: torch.Tensor) -> None:
    @mk.quantizer
    def uniform(w, ctx, bits=4):
        return mk.Q(w.round(), bits=bits * w.numel())

    configured = uniform(bits=3)
    assert configured(weights, mk.Ctx()).bits == 3 * weights.numel()
    assert uniform(weights, mk.Ctx()).bits == 4 * weights.numel()
    with pytest.raises(TypeError):
        uniform(unknown=True)


def test_lazy_context_statistics() -> None:
    inputs = torch.randn(30, 8)
    context = mk.Ctx(X=inputs)
    torch.testing.assert_close(context.H, inputs.T @ inputs / len(inputs))
    torch.testing.assert_close(context.act_absmean, inputs.abs().mean(0))
    torch.testing.assert_close(
        context.stat("m4", lambda x: x.pow(4).mean(0)), inputs.pow(4).mean(0)
    )
    assert context.replace(H=torch.eye(8)).cache is context.cache
    with pytest.raises(RuntimeError, match="calibration"):
        _ = mk.Ctx().H


def reference_gptq(weights: torch.Tensor, context: mk.Ctx, inner: mk.Quantizer) -> torch.Tensor:
    working, hessian = weights.clone(), context.H.clone()
    hessian.diagonal().add_(0.01 * hessian.diagonal().mean())
    upper = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
    )
    output = torch.zeros_like(working)
    for column in range(working.shape[1]):
        if column % 16 == 0:
            rounder = inner.fit(working[:, column : column + 16], context)
        local = column % 16
        quantized = rounder(working[:, column : column + 1], slice(local, local + 1)).w
        output[:, column : column + 1] = quantized
        error = (working[:, column : column + 1] - quantized) / upper[column, column]
        working[:, column + 1 :] -= error * upper[column, column + 1 :]
    return output


def test_gptq_matches_reference(weights: torch.Tensor) -> None:
    context = mk.Ctx(X=torch.randn(256, 64, generator=torch.Generator().manual_seed(11)))
    inner = mk.int(3, group=16, scale_fmt="fp32")
    expected = reference_gptq(weights, context, inner)
    result = mk.gptq(inner, refit=16, block_size=16)(weights, context)
    torch.testing.assert_close(result.w, expected, rtol=1e-5, atol=1e-6)
    assert result.bits == 3 * weights.numel() + 32 * 64


def test_feedback_handles_dead_channels(weights: torch.Tensor) -> None:
    inputs = torch.randn(128, 64)
    inputs[:, 3] = 0
    result = mk.gptq(mk.int(4, group=16), refit=16)(weights, mk.Ctx(X=inputs))
    assert torch.isfinite(result.w).all()
    assert (result.w[:, 3] == 0).all()


def test_incoherent_identity(weights: torch.Tensor) -> None:
    result = mk.incoherent(lambda w, ctx: mk.Q(w, bits=32 * w.numel()))(weights)
    torch.testing.assert_close(result.w, weights, rtol=1e-5, atol=1e-6)


def test_vector_grid_and_feedback(weights: torch.Tensor) -> None:
    codebook = torch.randn(32, 8)
    quantization = mk.scaled(mk.grid.vector(codebook), group=None)
    context = mk.Ctx(H=torch.eye(64))
    result = mk.ldlq(quantization, step=8)(weights, context)
    assert result.w.shape == weights.shape
    assert result.bits == 5 * weights.numel() / 8 + 16 * weights.shape[0]


def test_awq_search_includes_baseline(weights: torch.Tensor) -> None:
    context = mk.Ctx(X=torch.randn(256, 64))
    quantization = mk.int(3, group=16)
    result = mk.awq(quantization, grid=5)(weights, context)
    assert mk.proxy_loss(weights, result.w, context) <= mk.proxy_loss(
        weights, quantization(weights).w, context
    ) + 1e-6


def test_hadamard_and_random_transform_inverses() -> None:
    values = torch.randn(12, 64)
    torch.testing.assert_close(mk.hadamard(mk.hadamard(values)), values)
    torch.testing.assert_close(mk.rht(mk.rht(values, seed=4), seed=4, inverse=True), values)
    with pytest.raises(ValueError, match="power-of-two"):
        mk.hadamard(torch.randn(3, 12))


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


@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_tensors")
@pytest.mark.parametrize("dimension", [1, 8])
def test_weighted_kmeans_is_reproducible_on_cuda(dimension: int) -> None:
    samples = torch.randn(4096, dimension)
    if dimension == 1:
        samples = samples[:, 0]
    weights = torch.rand(4096)
    expected = mk.kmeans(samples, k=16, weights=weights, iters=3, seed=79)
    actual = mk.kmeans(samples, k=16, weights=weights, iters=3, seed=79)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_best_of_isolates_nested_side_information_accounting() -> None:
    weight = torch.randn(8, 16)

    @mk.quantizer
    def candidate(values, context, reconstruct=False):
        accounted = context.cache.setdefault("accounted", set())
        if "codebook" not in accounted:
            context.add_bits(128)
            accounted.add("codebook")
        result = values if reconstruct else torch.zeros_like(values)
        return mk.Q(result, bits=4 * values.numel())

    context = mk.Ctx(cache={"accounted": set()})
    result = mk.best_of(candidate, candidate(reconstruct=True), by="mse")(weight, context)
    torch.testing.assert_close(result.w, weight)
    assert context.additional_bits == 128
    assert context.cache["accounted"] == {"codebook"}
