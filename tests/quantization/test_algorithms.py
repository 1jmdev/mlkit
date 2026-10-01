import pytest
import torch

import mlkit as mk


@pytest.fixture
def weights() -> torch.Tensor:
    return torch.randn(16, 64, generator=torch.Generator(device="cuda").manual_seed(7))


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


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_gptq_matches_reference(weights: torch.Tensor, backend: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(11)
    context = mk.Ctx(X=torch.randn(256, 64, generator=generator))
    inner = mk.int(3, group=16, scale_fmt="fp32")
    expected = reference_gptq(weights, context, inner)
    result = mk.gptq(inner, refit=16, block_size=16, backend=backend)(weights, context)
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


def test_incoherence_on_non_power_of_two_dimensions() -> None:
    weights = torch.randn(28, 96)
    result = mk.incoherent(lambda w, ctx: mk.Q(w, bits=32 * w.numel()))(weights)
    torch.testing.assert_close(result.w, weights, rtol=1e-5, atol=1e-5)


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
