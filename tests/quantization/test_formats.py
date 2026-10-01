import pytest
import torch

import mlkit as mk


@pytest.fixture
def weights() -> torch.Tensor:
    return torch.randn(16, 64, generator=torch.Generator(device="cuda").manual_seed(7))


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


def test_asymmetric_partial_group_ignores_padding() -> None:
    weights = torch.rand(4, 70) + 10
    result = mk.int(4, group=32, asym=True, scale_fmt="fp32")(weights)
    final_group = weights[:, 64:]
    step = (final_group.amax(1) - final_group.amin(1)) / 15
    error = (result.w - weights)[:, 64:].abs().amax(1)
    assert (error <= step / 2 + 1e-5).all()


def test_integer_grid_is_identified_by_attribute_rather_than_name() -> None:
    assert mk.grid.int(4).integer

    @mk.grid(bits=4)
    def integer_like(values):
        return values.round().clamp(-8, 7)

    assert not integer_like.integer
    assert not mk.grid.values(torch.arange(-8.0, 8.0), bits=4).integer
