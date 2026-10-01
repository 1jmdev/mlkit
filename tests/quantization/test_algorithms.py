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


def reference_block_feedback(
    weights: torch.Tensor,
    hessian: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Block error feedback written with the inverse-Cholesky updates of GPTQ."""
    working, hessian = weights.clone(), hessian.clone()
    hessian.diagonal().add_(0.01 * hessian.diagonal().mean())
    upper = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
    )
    output = torch.zeros_like(working)
    for column in range(0, working.shape[1], step):
        stop = column + step
        reconstruction = (working[:, column:stop] * 4).round() / 4
        output[:, column:stop] = reconstruction
        residual = working[:, column:stop] - reconstruction
        error = torch.linalg.solve_triangular(
            upper[column:stop, column:stop].T, residual.T, upper=False
        ).T
        working[:, stop:] -= error @ upper[column:stop, stop:]
    return output


@pytest.mark.parametrize("step", [1, 4, 8])
@pytest.mark.parametrize("block_size", [8, 16, 64])
def test_block_feedback_matches_inverse_cholesky_reference(
    weights: torch.Tensor,
    step: int,
    block_size: int,
) -> None:
    @mk.quantizer
    def quarters(values, context):
        return mk.Q((values * 4).round() / 4, bits=8 * values.numel())

    generator = torch.Generator(device="cuda").manual_seed(13)
    context = mk.Ctx(X=torch.randn(256, 64, generator=generator))
    expected = reference_block_feedback(weights, context.H, step)
    result = mk.ldlq(quarters, step=step, block_size=block_size)(weights, context)
    torch.testing.assert_close(result.w, expected, rtol=1e-5, atol=1e-6)
    assert mk.proxy_loss(weights, result.w, context) < mk.proxy_loss(
        weights, (weights * 4).round() / 4, context
    )


@pytest.mark.parametrize("step", [1, 8])
def test_feedback_coefficients_agree_between_factorization_layouts(monkeypatch, step: int) -> None:
    from mlkit.quantization.algorithms import error_feedback

    generator = torch.Generator(device="cuda").manual_seed(17)
    inputs = torch.randn(256, 64, generator=generator)
    hessian = inputs.T @ inputs / 256
    lower_layout, status = error_feedback.feedback_coefficients(hessian, step)
    assert status == 0
    monkeypatch.setattr(error_feedback, "UPPER_FACTORIZATION_WIDTH", 1)
    upper_layout, status = error_feedback.feedback_coefficients(hessian, step)
    assert status == 0
    hessian.diagonal().add_(0.01 * hessian.diagonal().mean())
    assert lower_layout.is_contiguous() and upper_layout.is_contiguous()
    torch.testing.assert_close(
        upper_layout.triu(step), lower_layout.triu(step), rtol=1e-4, atol=1e-5
    )
    factor = torch.linalg.inv(
        torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)
    )
    blocks = torch.block_diag(*[
        factor[start : start + step, start : start + step] for start in range(0, 64, step)
    ])
    expected = factor @ torch.linalg.inv(blocks)
    torch.testing.assert_close(lower_layout.triu(step), expected.triu(step), rtol=1e-3, atol=1e-4)


def test_activation_order_supplies_statistics_in_permuted_column_order(
    weights: torch.Tensor,
) -> None:
    inputs = torch.randn(256, 64) * torch.linspace(0.1, 3.0, 64)
    context = mk.Ctx(X=inputs)
    observed = {}

    class Recording(mk.Quantizer):
        def fit(self, weight, fitted_context):
            observed["weight"] = weight.clone()
            observed["H"] = fitted_context.H.clone()
            observed["act_absmean"] = fitted_context.act_absmean.clone()
            return mk.int(4, group=16).fit(weight, fitted_context)

    mk.gptq(Recording(), refit=None, act_order=True)(weights, context)
    permutation = context.H.diagonal().argsort(descending=True)
    torch.testing.assert_close(observed["weight"], weights[:, permutation])
    torch.testing.assert_close(observed["H"], context.H[permutation][:, permutation])
    torch.testing.assert_close(observed["act_absmean"], context.act_absmean[permutation])
    assert (observed["H"].diagonal().diff() <= 0).all()


@pytest.mark.parametrize("act_order", [False, True])
def test_feedback_reduces_proxy_loss_below_direct_rounding(
    weights: torch.Tensor,
    act_order: bool,
) -> None:
    inputs = torch.randn(256, 64) * torch.linspace(0.1, 3.0, 64)
    context = mk.Ctx(X=inputs)
    format = mk.int(3, group=16)
    direct = mk.proxy_loss(weights, format(weights).w, context)
    feedback = mk.gptq(format, refit=16, act_order=act_order)(weights, context)
    assert mk.proxy_loss(weights, feedback.w, context) < direct
    assert feedback.codec == "feedback"
